# Kind model-routing rehearsal — graphs and what happened

Runs of 2026-06-11 on the `kind-sim-multi` stack (section C of the main
README): two sim backends — `facebook/opt-350m` ("fast": 1s TTFT / 50ms ITL,
~4.5–5 RPS capacity) and `facebook/opt-125m` ("slow": 3s TTFT / 200ms ITL,
~0.8 RPS). IPP runs `model-selector` + `model-name-filter`; the workload's
`model` field is a string-encoded array offering both, so every routing
decision below was made by the scorer/picker. Timelines are 60s monitoring
windows of IPP "Model selected" log lines (pod logs rotate — capture live).
Regenerate with `tools/plot_kind_rehearsal.py`.

## 1. `routing_avgttft_only.png` — why max-score + avg-ttft alone fails

Config: `avg-ttft-scorer` (1.0) + `max-score-picker`. Load: 3 → 12 → 40 RPS,
two 60s stages each.

- **3 RPS**: everything goes to the fast sim — correct: its TTFT EMA (~1s)
  beats the slow sim's (~3s).
- **12 RPS**: the fast sim saturates (~4.5 RPS capacity), its queue grows,
  but the EMA only learns this after queued requests produce first tokens —
  so for ~2 minutes nothing overflows. When the EMA finally crosses 3s, the
  picker (winner-take-all) moves **100% of traffic** onto the 0.8 RPS slow
  sim — a stampede, not an overflow. The slow sim drowns, the comparison
  flips back, and the system oscillates at full amplitude.
- **40 RPS**: the drowned sim stops emitting first tokens → its EMA stops
  updating → after the 30s staleness threshold the scorer treats it as
  *idle* (score 1.0) → it attracts **all** traffic (the 805/819 minute).
  "No data = idle" is right at cold start and exactly backwards under
  overload.

## 2. `routing_blended.png` — adding the inflight scorer

Config: `avg-ttft-scorer` (1.0) + `inflight-requests-scorer` (1.0) +
`max-score-picker`. Same load.

- **Treatment 1 (fresh state)**: the inflight count increments at *request*
  time, so each pick immediately lowers the picked model's next score —
  per-request negative feedback. Result: a stable, load-proportional split
  at every stage; no flips, no lock-ins.
- **Treatment 2 (right panel)**: started with the in-flight counters leaked
  by treatment 1's failure storm (see finding below) — the healthy fast sim
  scored 0 and was locked out for minutes even at 3 RPS.

## 3. `stage_outcomes.png` — the quantified trade-off

- While the small model is saturated (12 RPS stages), the blend cuts
  failures ~40% (182+268 vs 340+413): the overflow genuinely uses the second
  backend's capacity.
- The cost: while *unsaturated* (3 RPS stages), the inflight term sends
  ~30% of requests to the 3s-TTFT sim unnecessarily — successes' TTFT p90 is
  3.0s vs 1.0s. Tunable via the inflight weight (try 0.3 instead of 1.0).
- Beyond combined capacity (40 RPS) both configs fail equally — nothing
  routes your way out of insufficient capacity.

## 4. `spike_timeline.png` — parallel harnesses + pinning

Planner harness (2nd namespace, run-only mode) pins `facebook/opt-125m` by
naming it; summarizer harness offers both models. The gray solo window shows
100% pinned routing (the filter guarantee). When the summarizer joins, the
leaked-counter state initially sends all its traffic onto the planner's
model (degrading the pinned workload: 51 ok / 99 fail); after the state
flipped, the healthy pattern appears — summarizer on the fast sim, planner
trickle pinned. Lesson: **pinning is not reservation** — array-carrying
traffic may legally land on a pinned workload's backend.

## Findings to bring upstream

1. **Stale-EMA-as-idle** (avg-ttft scorer): an overloaded backend that emits
   no first tokens goes "stale" and scores like an idle one, attracting more
   load. Staleness decay must distinguish idle from drowning (in-flight
   count is the signal — but see 2).
2. **In-flight counter leak** (`request-metadata-extractor`): `Requests++`
   per request, decrement only on *response events* — Envoy-local 504s and
   shed 503s never decrement, so failure storms permanently poison both the
   idleness signal and the inflight scorer until an IPP restart.
