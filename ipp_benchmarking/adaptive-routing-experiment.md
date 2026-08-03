# Adaptive-routing experiment — Gemma-4 26B vs Qwen3.6 35B (FP8)

Show the power of **latency-based routing** between two similar models. The IPP
`ttft-aware-scorer` routes a shared, model-agnostic stream to whichever pool has spare
capacity, and we prove it **recovers** when dedicated per-model load clears.

**Models** (both FP8, MoE, 1× H100 each, similar quality & perf ±20% — measured):
`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` and `Qwen/Qwen3.6-35B-A3B-FP8`.

**What we simulate:** a shared traffic stream both pools can serve, plus dedicated per-model
load we toggle on/off. The IPP shifts shared traffic to the pool with spare capacity, then
recovers when the load clears.

**Workloads** — all three streams are the SAME synthetic summarization shape (~2048 in / ~256 out,
random, no prefix reuse) at the same rate, generated with the same tokenizer. Only the `model`
field differs:
- `shared` = `model: "auto"` → routed to **both** pools by the scorer
- `gemma-only` = exact Gemma name → pinned to Gemma
- `qwen-only` = exact Qwen name → pinned to Qwen

Identical shapes are the point: a shared request and a pinned request are then the same unit of
work, so per-pool latencies are directly comparable. (The first run used shareGPT for shared,
summarization for Gemma and code-completion for Qwen, which made the pools' latencies
incomparable — Gemma's higher e2e was mostly its longer outputs, not the pool.)

The IPP `model-group-name-filter` reads the workload's `model` field: `"auto"` keeps both pools
as candidates (adaptive); an **exact model name** pins to that pool. (`model-name-filter` is
exact-match only — it 429s on `"auto"`.)

**Runs — poisson RPS per 300s stage** (shared `model:"auto"` adaptive; pinned = exact model name):

| Stage        | Shared (const) | Gemma-only | Qwen-only | Shows                    |
|--------------|----------------|------------|-----------|--------------------------|
| -1 baseline1 | off            | 10         | 10        | baseline with no shared  |
|  0 baseline2 | 10             | off        | off       | reference split          |
|  1 pin Gemma | 10             | 10         | off       | shared shifts to Qwen    |
|  2 release   | 10             | off        | off       | recovery                 |
|  3 pin Qwen  | 10             | off        | 10        | shared shifts to Gemma   |
|  4 both      | 10             | 10         | 10        | overload (30 rps offered) |
|  5 release   | 10             | off        | off       | full recovery            |

~60 min (36,000 requests). `adaptive_toggle_run.sh` runs the stages discretely: each stage launches
its own 300s runs, waits for all of them, then the next stage starts. Nothing crosses a stage
boundary. Concurrent runs within a stage each get their own harness namespace (`-p $NS,$NS-2` /
`$NS-3`) — llmdbenchmark deletes harness pods by a shared label per namespace, so co-located runs
would truncate each other. The driver seeds those namespaces (PVC, data-access pod, SA+RBAC) itself.

Each pool sits just under saturation on its own (Gemma 10 of a measured 11.1 rps ceiling for this
shape; Qwen 10 of a predicted ~14). **Stage 4 is deliberately oversubscribed**: 30 rps offered
against ~25 rps of combined capacity, so expect queue growth and a long tail there — that stage
tests routing under overload, not steady state. See
`example_outputs/gemma-qwen-adaptive/saturation-ramps/` for the ceilings and how they were measured.

## Random-routing baseline (ablation)

Same seven stages, same profiles, same image — only `ttft-aware-scorer` removed from the IPP
pipeline (`ipp_configs/adaptive-gemma-qwen-random-values.yaml`, a 6-line diff from the smart
values). With no scorer every candidate keeps score 0, and `max-score-picker` shuffles before its
stable sort ("needed for random tie break when scores are equal",
`picker/maxscore/picker.go`), so each `"auto"` request is a uniform coin flip between the pools.
Pinned requests are unaffected — the filter still leaves exactly one candidate.

`ttft-percentile-extractor` is kept deliberately: nothing consumes it, but its `ttft-observation`
line is the per-request actual-TTFT source *and* the end-of-stream marker the per-request e2e
derivation needs on this image (the per-chunk lines are gone).

Verified before running — 60 shared requests gave 27/33 (45/55, 0.8σ from 50/50) with zero
`ttft-aware score` lines emitted.

    # switch the IPP to the random config (helm upgrade hits an SSA conflict on the ConfigMap;
    # patch it directly, which is what the existing kubectl-patch field manager already did)
    python3 -c "import yaml;c=yaml.safe_load(open('ipp_benchmarking/ipp_configs/adaptive-gemma-qwen-random-values.yaml'))['payloadProcessor']['customConfig'];print(yaml.safe_dump({'apiVersion':'llm-d.ai/v1alpha1','kind':'PayloadProcessorConfig','datalayer':c['datalayer'],'plugins':c['plugins'],'profiles':c['profiles']}))" > /tmp/random-ipp-config.yaml
    oc create cm ipp-cfg-tmp -n "$NAMESPACE" --from-file=custom-ipp-config.yaml=/tmp/random-ipp-config.yaml --dry-run=client -o json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print(json.dumps([{'op':'replace','path':'/data/custom-ipp-config.yaml','value':d['data']['custom-ipp-config.yaml']}]))" > /tmp/patch.json
    oc patch cm payload-processor -n "$NAMESPACE" --type=json --patch-file=/tmp/patch.json
    oc rollout restart deploy/payload-processor -n "$NAMESPACE"

    # then the same seven stages, into their own output directory
    TAG=stage-300s-random-baseline bash /tmp/stage.sh s-1_baseline1 adaptive_gemma_summarization.yaml adaptive_qwen_summarization.yaml
    ...

Results land in `example_outputs/gemma-qwen-adaptive/stage-300s-random-baseline/`, directly
comparable to `stage-300s-summarization/` stage for stage.

---

## Files

| Purpose | Path |
|---|---|
| Two-pool scenario / spec | `config/scenarios/cicd/ocp-gemma-qwen-adaptive.yaml` (+ `config/specification/…`) |
| IPP values (image `ttft-aware-p25-expl`) | `ipp_benchmarking/ipp_configs/adaptive-gemma-qwen-smart-values.yaml` |
| Base-model ConfigMaps | `ipp_benchmarking/ipp_configs/{gemma-26b,qwen36-35b}-base-model.yaml` |
| Workloads | `workload/profiles/inference-perf/adaptive_{shared,gemma,qwen}_summarization.yaml.in` |
| Run driver | `ipp_benchmarking/tools/adaptive_toggle_run.sh` |
| In-pod stage extractor (IPP slice + per-request e2e) | `ipp_benchmarking/tools/ipp_extract_stage.sh` |
| Analysis / figures | `ipp_benchmarking/tools/plot_adaptive_stages.py` |

## Run procedure (from repo root)

```bash
export NAMESPACE=<your-namespace>
export IPP_PATH=<path to a llm-d-inference-payload-processor checkout>   # for the helm chart only

# 1. Standup BOTH pools (2 GPUs). Long download + Qwen ~5-min engine init.
llmdbenchmark --spec cicd/ocp-gemma-qwen-adaptive standup -p "$NAMESPACE"

# 2. Install the IPP. Image: ghcr.io/mohammad-nassar10/llm-d-inference-payload-processor:ttft-aware-p25-expl
#    (ttft-aware scorer + exploration fix, and the per-chunk log flood demoted so --v=4 is usable).
#    The tag is MUTABLE -- `helm upgrade` alone will not re-pull it if the pod is already running;
#    the rollout restart below is what picks up a new build.
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio \
  -f ipp_benchmarking/ipp_configs/adaptive-gemma-qwen-smart-values.yaml \
  --set provider.supportedEvents.requestBody=true --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseBody=true --set 'payloadProcessor.flags.v=4' \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=1200s
oc rollout restart deploy/payload-processor -n "$NAMESPACE"
oc rollout status  deploy/payload-processor -n "$NAMESPACE" --timeout=180s

# 3. Base-model ConfigMaps (feed X-Gateway-Base-Model-Name; NOT /config/models.json).
oc apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/gemma-26b-base-model.yaml \
                         -f ipp_benchmarking/ipp_configs/qwen36-35b-base-model.yaml

# 4. Header-match routes (scenario sets httpRoute.enabled: false).
ipp_benchmarking/tools/gen_httproutes.sh "$NAMESPACE" infra-llmdbench-inference-gateway \
  RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic Qwen/Qwen3.6-35B-A3B-FP8 | oc apply -f -

# 5. IPP capture must be running IN-CLUSTER before the timeline starts (a laptop-side
#    `oc logs -f` loses 75% of lines under load). Truncate the previous capture first -- the PVC
#    is 96Gi and this run adds ~5-10 GB.
oc exec -n "$NAMESPACE" access-to-harness-data-workload-pvc -- \
  truncate -s 0 /requests/smart-logs/ipp-full-live.log
oc get pod ipp-logtail -n "$NAMESPACE"        # must be Running

# 6. Run the 7-stage timeline, then slice each stage window out of the capture.
ipp_benchmarking/tools/adaptive_toggle_run.sh adaptive

# 7. Figures + tables.
python3 ipp_benchmarking/tools/plot_adaptive_stages.py \
  ipp_benchmarking/example_outputs/gemma-qwen-adaptive/adaptive -o /tmp/adaptive

# teardown
llmdbenchmark --spec cicd/ocp-gemma-qwen-adaptive teardown -p "$NAMESPACE"
helm uninstall payload-processor -n "$NAMESPACE"   # IPP is not torn down automatically
```

**Headline result:** shared-probe p99 with the smart scorer vs a static 50/50 baseline across
stages 1–4; plus per-pool routing share over time from `ipp-full-live.log` (`analyze_routing.py`).

## Running one stage at a time (`tools/stage.sh`)

`adaptive_toggle_run.sh` runs the whole timeline on a schedule. `stage.sh` runs **one** stage and
archives everything for it — per-stream results, IPP slice, per-request e2e — retrying if the
gpu-reaper scales a pool mid-stage. Deploy names are standup-generated, so override them:

```bash
export NAMESPACE=<your-namespace> TAG=<arm name>          # TAG = output dir under example_outputs/
export G_DEPLOY=$(oc get deploy -n $NAMESPACE -o name | grep -i gemma | grep decode | cut -d/ -f2)
export Q_DEPLOY=$(oc get deploy -n $NAMESPACE -o name | grep -i qwen  | grep decode | cut -d/ -f2)
export NS=$NAMESPACE

ipp_benchmarking/tools/stage.sh s0_shared20 adaptive_shared_summ20.yaml
ipp_benchmarking/tools/stage.sh s1_pinGemma adaptive_gemma_summarization.yaml \
                                            adaptive_shared_summarization.yaml
```

Each stage prints `DONE … gemma_pct=` from the vLLM counters and a capture-completeness check.
A stage that produced **0 requests still exits 0** — check the `total=` in that line, and read the
`llmdbenchmark` log in `/tmp/stage_<label>_<profile>.log` if it is zero.

## Arm: `ttft-aware-fix` image (no `model-group-name-filter`)

Newer IPP builds may not register `model-group-name-filter`. Check what a build actually has before
trusting a values file — the pod names the missing plugin and crashloops:

```bash
oc logs -n "$NAMESPACE" deploy/payload-processor | grep -E "build|not registered"
```

`ttft-aware-fix` (build `8f4a606`) registers `auto-group-model-name-filter` instead, which pins by
group rather than by exact model name:

| request `model` | candidates |
|---|---|
| `"auto"` | both pools → the scorer chooses (shared stream) |
| `"auto/gemma"` | Gemma only (pinned stream — use `adaptive_gemma_autogroup_summarization.yaml`) |
| exact model name | **none → 429** |

Use `ipp_configs/adaptive-gemma-qwen-ttft-aware-fix-values.yaml`, and additionally:

```bash
# harness step 04 verifies with a bodyless GET /v1/models, which IPP cannot route -> 404
sed "s|<inference-pool>|$(oc get inferencepool -n $NAMESPACE -o name | head -1 | cut -d/ -f2)|" \
  ipp_benchmarking/ipp_configs/models-probe-route.yaml | oc apply -n "$NAMESPACE" -f -

# full-fidelity IPP capture (needs SA `ipp-logtail` with get/list on pods + pods/log)
oc apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/ipp-logtail-pod.yaml
```

Verify **both** before starting a stage — a dead capture only produces a warning, and `2/2 Ready`
does not mean vLLM is serving:

```bash
GW=http://infra-llmdbench-inference-gateway-istio.$NAMESPACE.svc.cluster.local:80
oc exec -n "$NAMESPACE" ipp-logtail -- curl -s -o /dev/null -w '%{http_code}\n' $GW/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","prompt":"warm up","max_tokens":8,"stream":true}'   # want 200
oc exec -n "$NAMESPACE" ipp-logtail -- stat -c %s /requests/smart-logs/ipp-full-live.log  # want it
                                                    # to grow across the probe above
```

Then slice each stage's window (from `stage_marks.log`) out of the capture, in-cluster:

```bash
oc exec -i -n "$NAMESPACE" ipp-logtail -- sh -s s1_pinGemma:<start_ts>:<end_ts+60> \
  < ipp_benchmarking/tools/slice_raw_window.sh
oc rsync -n "$NAMESPACE" ipp-logtail:/requests/smart-logs/windows/ ./   # oc cp truncates
```

**Confirm the scorer is not blind.** With no TTFT observations it scores every model equally and
`max-score-picker` shuffles — indistinguishable from a random picker:

```bash
python3 -c "
import gzip,json,collections
c=collections.Counter()
for l in gzip.open('<stage>/ipp-raw.log.gz','rt',errors='ignore'):
    if '\"msg\":\"ttft-percentile wrote attribute\"' in l: c[json.loads(l).get('RecentN')]+=1
print(c)"        # all-zero RecentN => blind; add session-affinity (see Gotchas) and re-run
```

## Gotchas (learned)
- **RPS is a starting point** anchored to a light shape (490/128); the workloads here are
  heavier (2048-input). Run a per-model poisson RPS sweep first to set the real saturation, then
  scale the `10` rps in the profiles.
- **IPP registration is mandatory** — without both models registered + `model-group-name-filter`,
  every request dies at `no models available after filtering`.
- **`models.json` may be wiped on helm upgrade** — if the IPP crashloops, re-inject and restart
  (patch command in `README.md`, OCP step 2).
- **Sequential standup only** — the two pools collide on the shared gateway secret if run in parallel.
- **`session-affinity`** is omitted from `-smart-values.yaml` (as the archived runs had it). If the
  IPP log shows blind routing (RecentN=0 / no recorded TTFT), use `-ttft-aware-fix-values.yaml`. This
  bit on `ttft-aware-fix`: the extractor only fires on the buffered response path, so without a
  ResponseProcessor every request scored (1,1) and routing was uniformly random.
- Set `LLMDBENCH_WAIT_TIMEOUT=6000`; wait for real vLLM readiness (poll `/v1/completions`, not `/health`).
