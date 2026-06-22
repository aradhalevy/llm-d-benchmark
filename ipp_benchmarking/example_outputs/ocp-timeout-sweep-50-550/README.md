# IPP routing A/B — concurrency sweep 50→550, NO timeouts (OCP Qwen3 8B+32B)

`max-score-picker` with **no scorer (random ≈50/50)** vs **avg-ttft-scorer (smart, load-aware)**.
Single summarizer harness, IPP chooses among `["Qwen/Qwen3-8B","Qwen/Qwen3-32B"]`. 6 stages
× 1500 req (2048-in/256-out). Gateway HTTPRoute timeout lifted 30s→1200s so **nothing is
shed** — every request runs to completion. Stats trim 10% time-edges per stage.

## Headline

**0 failures in both arms** (no timeouts at any concurrency; random max latency 373s, smart 80s — both < 1200s).
The avg-ttft scorer routes **85–96% to the fast 8B**, avoiding the slow dense 32B that bottlenecks
the random arm. Result: **~5× lower p95 latency and ~3–4× higher throughput** at high concurrency.

| C | 8B% rnd | 8B% smt | p50 rnd | p50 smt | p95 rnd | p95 smt | tok/s rnd | tok/s smt |
|----|----|----|----|----|----|----|----|----|
| 50 | 48% | 96% | 13.7s | 5.2s | 34.8s | 8.1s | 809 | 2425 |
| 150 | 51% | 87% | 6.7s | 10.9s | 100s | 17.5s | 899 | 3646 |
| 250 | 50% | 87% | 13.4s | 18.2s | 169s | 26.1s | 848 | 3247 |
| 350 | 49% | 87% | 21.9s | 25.1s | 238s | 39.5s | 793 | 2977 |
| 450 | 50% | 86% | 27.5s | 30.9s | 306s | 57.9s | 789 | 2771 |
| 550 | 48% | 85% | 32.7s | 37.3s | 366s | 71.6s | 783 | 2646 |

Nuance: smart's **p50 is slightly higher** at mid-load (it concentrates ~87% on the 8B, so the
median 8B request is a bit busier), but its **p95/tail is 4–5× better** because it keeps the slow
32B lightly loaded (only ~13%) instead of drowning 50% of traffic there. The 32B share rises
4%→15% as concurrency grows (the scorer offloads overflow to 32B as the 8B fills).

## Files
- `latency_rps_ab.png` — **main A/B figure**: top = per-backend p50/p95 latency vs concurrency (linear); bottom = actual RPS sent vs time (y capped 100), stacked by served backend, stages labeled `C=N`, gaps `cooldown`. The bottom time-axis tells the efficiency story directly: random needs **~2700s** of load for the 6 stages, smart only **~850s** (same 9000 reqs, 3× faster — it rides the fast 8B).
- `latency_rps_ab_trim.png` — same, but the RPS panel drops the first/last 10% of each stage (removes the slot-fill burst + drain) and caps y at 30 → clean steady-state RPS: smart holds ~10-15 RPS (mostly 8B) vs random ~2-10 RPS split across both backends.
- `latency_ab_linear.png` — per-backend p50/p95 vs concurrency, p50|p95 panels (linear; the 32B-slow dashed curve is the random arm's killer)
- `routing_share_vs_concurrency_ab.png` — 8B routing share vs concurrency, both arms
- `AB_summary_table.txt` — the table above
- `{random,smart}/per_request_slim.json` — per-request (t, lat, fail, served-model); 9000 each, 0 fails
- `{random,smart}/stage_*_lifecycle_metrics.json` — per-stage inference-perf metrics
- `smart/ipp-decisions.log` — IPP decision log (smart arm; random's was lost to rotation but slim==decision at 0 fails)
- `../../collected-logs-sweep-{random,smart}/` — IPP/EPP/vLLM/gateway logs + GPU DCGM

Tools: `extract_per_request_slim.py`, `trim_stages.py`, `plot_latency_ab_linear.py`, `plot_routing_share_from_slim.py`.
