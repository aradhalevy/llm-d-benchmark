# IPP benchmarking

Measure how IPP routing changes latency and throughput: a load-aware **smart**
scorer vs. a static or random baseline. Three experiments below.

This uses `llmdbenchmark` (the repo this lives in). Detailed instructions on how to use that are in the main [README](../README.md) of the repo.

Run `llmdbenchmark` **from the repo root** — it auto-discovers this bundle's
scenarios, specs and profiles. Plotters need matplotlib/numpy, so run them with
`.venv/bin/python3`.

Troubleshooting, gotchas and findings live in [`AGENTS.md`](./AGENTS.md).

## Run data

`example_outputs/` is the directory results land in. There is one directory per arm:
`example_outputs/<experiment>/<arm>/`. Plotters take arms as positional
`label=dir` plus `-o out.png`, and read:

| file | produced by |
|---|---|
| `stage_<N>_lifecycle_metrics.json`, `summary_lifecycle_metrics.json` (summary of the runs) | `llmdbenchmark run` |
| `harness_stdout.log` (stage bands) | `llmdbenchmark run` |
| `per_request_slim.json` (per request data, like TTFT and ITL) | `tools/extract_per_request_slim.py` |
| `ipp-full-live.log` (the ipp logs) | `tools/ab_routing_run.sh` |

`ab_routing_run.sh` and `adaptive_toggle_run.sh` run the expirements and write this layout themselves.

---

## 1. Kind simulator (no GPU)

This expirement uses Kind with [vllm simulators ](https://github.com/llm-d/llm-d-inference-sim).
We use two fake-latency simulators — `facebook/opt-125m` (slow) and `opt-350m` (fast) — for a
fast local smoke test of routing. A/B the scorer by re-running with
`maxscore-baseline-values.yaml` (no scorer → random ~50/50) as the second arm.

```bash
kind create cluster
docker pull ghcr.io/llm-d/llm-d-benchmark:v0.6.3 && kind load docker-image ghcr.io/llm-d/llm-d-benchmark:v0.6.3
export IPP_PATH=/path/to/llm-d-inference-payload-processor
make -C "$IPP_PATH" image-build REGISTRY=ghcr.io/<you> VERSION=ttft-scorer
kind load docker-image ghcr.io/<you>/llm-d-inference-payload-processor:ttft-scorer

./install.sh && source .venv/bin/activate
llmdbenchmark --spec cicd/kind-sim-multi standup -p llmdbench

# values file carries the plugin pipeline, listModels and image wiring -- set
# payloadProcessor.image.registry in it to your own
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n llmdbench -f ipp_benchmarking/ipp_configs/kind-sim-smart-values.yaml
kubectl rollout restart deploy/payload-processor -n llmdbench   # config-only upgrades don't restart it
kubectl apply -n llmdbench -f ipp_benchmarking/ipp_configs/opt-125m-base-model.yaml \
                           -f ipp_benchmarking/ipp_configs/opt-350m-base-model.yaml

# give the sims different TTFT/ITL so routing has something to optimize; not in the
# scenario, so re-apply after every standup
for d in $(kubectl get deploy -n llmdbench -o name | grep decode); do
  m=$(kubectl get $d -n llmdbench -o jsonpath='{.spec.template.spec.containers[0].args[1]}')
  case $m in *125m*) t=3s i=200ms;; *) t=1s i=50ms;; esac
  kubectl patch $d -n llmdbench --type=json -p="[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":[\"--model\",\"$m\",\"--port\",\"8200\",\"--served-model-name\",\"$m\",\"--time-to-first-token=$t\",\"--inter-token-latency=$i\",\"--max-num-seqs=10\"]}]"
done

llmdbenchmark --spec cicd/kind-sim-multi run -l inference-perf -w sanity_random.yaml
NAMESPACE=llmdbench ./ipp_benchmarking/collect_logs.sh        # -> ./collected-logs-<N>/

llmdbenchmark --spec cicd/kind-sim-multi teardown -p llmdbench
helm uninstall payload-processor -n llmdbench                 # IPP isn't torn down automatically
```

**Plot** — both plotters accept an arm dir or a whole `collected-logs-<N>/` bundle:

```bash
C=30,50,70,90,110,130,150,170,190,210,230
A="random"=collected-logs-random "smart"=collected-logs-smart
.venv/bin/python3 ipp_benchmarking/tools/plot_latency_vs_concurrency_kind.py $A \
  --concurrencies $C --bars split -o latency_vs_concurrency_kind.png
.venv/bin/python3 ipp_benchmarking/tools/plot_routing_vs_concurrency_kind.py $A \
  --concurrencies $C -o routing_vs_concurrency_kind.png
```

---

## 2. OCP Qwen3-8B + 32B — smart routing vs static baselines

A deep-research agent hammers the fast 8B and leaves the 32B mostly idle. Smart
routing spills 8B overflow onto the 32B as load climbs.

Three runs, one arm per `ab_routing_run.sh` invocation — it patches the arm's values into the IPP, restarts it, runs the profile, streams the full IPP log to `ipp-full-live.log` (the container log rotates away under load), then collects.

All three profiles are open-loop **Poisson RPS** ladders, 11 stages × 300s (~55
min per arm), same request shape (~2048 in / ~256 out):

- **`static_8b` / `static_32b`** — the values file registers a single model, so
  there is no routing choice. `half_8b_poisson.yaml` runs 2.5→15→2.5 RPS and
  `half_32b_poisson.yaml` 1→6→1 RPS, so per stage `8B + 32B` sums to the smart
  arm's rate.
- **`smart`** — both models registered, no `model-group-name-filter`, so
  candidates come from `listModels` and `queue-ttft-scorer` (+
  `ttft-percentile-extractor`, `explorationRate: 0.1`) with `max-score-picker`
  choose per request. `sweep_8stage_poisson.yaml` runs the summed ladder
  3.5→21→3.5 RPS. Needs an image built with `queue-ttft-scorer`. For the avg-ttft
  scorer swap in `avgttft-ocp-values.yaml`.

All timeouts are lifted to 1200s so nothing is shed — the delta is latency and
throughput, not failures.

```bash
# Prereqs: OCP w/ H100-80GB, `oc login`, HF_TOKEN with Qwen access, IPP image pullable.
export NAMESPACE=llm-d-<you>
export IPP_PATH=/path/to/llm-d-inference-payload-processor

# 1. Standup both pools, then the two required decode fixes (crashloop otherwise).
llmdbenchmark --spec cicd/ocp-qwen3-8b-32b standup -p "$NAMESPACE"
for d in $(oc get deploy -n "$NAMESPACE" -o name | grep decode); do
  oc patch "$d" -n "$NAMESPACE" -p '{"spec":{"strategy":{"type":"Recreate","rollingUpdate":null}}}'
  oc set env "$d" -n "$NAMESPACE" -c vllm USER=vllm LOGNAME=vllm
  oc patch "$d" -n "$NAMESPACE" --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-model-len 8192"}]'
done

# 2. Install IPP. VALUES selects which model(s) are REGISTERED -- re-run this same
#    upgrade with the matching values file before each arm below.
export VALUES=ipp_benchmarking/ipp_configs/static-8b-only-values.yaml
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio -f "$VALUES" \
  --set provider.supportedEvents.requestBody=true --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseBody=true --set 'payloadProcessor.flags.v=4' \
  --set payloadProcessor.image.registry=ghcr.io/<you> --set payloadProcessor.image.tag=ttft-scorer \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=1200s
oc apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/qwen3-8b-base-model.yaml \
                         -f ipp_benchmarking/ipp_configs/qwen3-32b-base-model.yaml

# 3. Header-match routes (the scenario disables the default PathPrefix route).
ipp_benchmarking/tools/gen_httproutes.sh "$NAMESPACE" | oc apply -f -

# 4. Set NS and GW at the top of tools/ab_routing_run.sh, then run ONE arm at a time,
#    re-running step 2 with the matching values file before each.
ipp_benchmarking/tools/ab_routing_run.sh static_8b \
    ipp_benchmarking/ipp_configs/static-8b-only-values.yaml half_8b_poisson.yaml
ipp_benchmarking/tools/ab_routing_run.sh static_32b \
    ipp_benchmarking/ipp_configs/static-32b-only-values.yaml half_32b_poisson.yaml
ipp_benchmarking/tools/ab_routing_run.sh smart \
    ipp_benchmarking/ipp_configs/median-ttft-ocp-values.yaml sweep_8stage_poisson.yaml

llmdbenchmark --spec cicd/ocp-qwen3-8b-32b teardown -p "$NAMESPACE"
helm uninstall payload-processor -n "$NAMESPACE"
```

**Plot.** `plot_ttft_per_stage.py` reads each stage's requested RPS from the
stage file and overlays the arms by stage, printing the per-stage % change.
The arms are comparable **stage for stage** (`8B stage N + 32B stage N = smart
stage N`), not rate for rate — and the x labels come from the **first** arm
listed, so put `smart` first and read the static curves as its two halves.

```bash
D=ipp_benchmarking/example_outputs/ocp-research-agent-routing
RPS=3.5,7,10.5,14,17.5,21,17.5,14,10.5,7,3.5      # smart ladder; static legs are its halves
P=.venv/bin/python3

# per-stage latency vs requested RPS -- TTFT (default) and e2e
$P ipp_benchmarking/tools/plot_ttft_per_stage.py "smart"=$D/smart "static_8b"=$D/static_8b \
  -o $D/ttft_per_stage_ab.png
$P ipp_benchmarking/tools/plot_ttft_per_stage.py "smart"=$D/smart "static_8b"=$D/static_8b \
  --metric e2e -o $D/e2e_per_stage_ab.png

# 8B/32B split per stage from the IPP decision log
$P ipp_benchmarking/tools/analyze_routing.py $D/smart/ipp-full-live.log

# predicted-vs-actual TTFT for one arm; also writes a zoomed PNG per stage into <out>_stages/
$P ipp_benchmarking/tools/plot_ttft_actual_vs_predicted.py "smart"=$D/smart/ipp-full-live.log \
  --unit RPS --concurrencies $RPS -o $D/ttft_smart_fullrun.png
```

---

## 3. OCP Gemma-4-26B vs Qwen3.6-35B — adaptive routing

Two comparable FP8 MoE models (`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`,
`Qwen/Qwen3.6-35B-A3B-FP8`), 1× H100 each. A shared `model: "auto"` stream is
routed by `ttft-aware-scorer` to whichever pool has spare capacity; dedicated
per-model load is toggled on and off to show the shift **and the recovery**.

All three streams are the same synthetic summarization shape (~2048 in / ~256
out) at the same rate — only the `model` field differs, so a shared request and a
pinned request are the same unit of work. `model-group-name-filter` reads it:
`"auto"` keeps both pools as candidates, an exact model name pins to one pool.

Poisson RPS per 300s stage, ~60 min total (36,000 requests):

| Stage        | Shared | Gemma-only | Qwen-only | Shows                     |
|--------------|--------|------------|-----------|---------------------------|
| -1 baseline1 | off    | 10         | 10        | baseline with no shared   |
|  0 baseline2 | 10     | off        | off       | reference split           |
|  1 pin Gemma | 10     | 10         | off       | shared shifts to Qwen     |
|  2 release   | 10     | off        | off       | recovery                  |
|  3 pin Qwen  | 10     | off        | 10        | shared shifts to Gemma    |
|  4 both      | 10     | 10         | 10        | overload (30 rps offered) |
|  5 release   | 10     | off        | off       | full recovery             |

Each pool sits just under saturation on its own; **stage 4 is deliberately
oversubscribed** (30 rps against ~25 rps of capacity) and tests routing under
overload, not steady state. Ablation arm: the same stages with
`adaptive-gemma-qwen-random-values.yaml` (identical pipeline minus the scorer, so
every `"auto"` request is a coin flip).

```bash
export NAMESPACE=<your-namespace>
export IPP_PATH=<path to a llm-d-inference-payload-processor checkout>   # helm chart only

# 1. Standup both pools (2 GPUs). Long download + ~5-min Qwen engine init.
llmdbenchmark --spec cicd/ocp-gemma-qwen-adaptive standup -p "$NAMESPACE"

# 2. Install the IPP. Image tag is MUTABLE -- the rollout restart is what re-pulls it.
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio \
  -f ipp_benchmarking/ipp_configs/adaptive-gemma-qwen-smart-values.yaml \
  --set provider.supportedEvents.requestBody=true --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseBody=true --set 'payloadProcessor.flags.v=4' \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=1200s
oc rollout restart deploy/payload-processor -n "$NAMESPACE"
oc rollout status  deploy/payload-processor -n "$NAMESPACE" --timeout=180s

# 3. Base-model ConfigMaps (feed X-Gateway-Base-Model-Name) + header-match routes.
oc apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/gemma-26b-base-model.yaml \
                         -f ipp_benchmarking/ipp_configs/qwen36-35b-base-model.yaml
ipp_benchmarking/tools/gen_httproutes.sh "$NAMESPACE" infra-llmdbench-inference-gateway \
  RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic Qwen/Qwen3.6-35B-A3B-FP8 | oc apply -f -

# 4. IPP capture must run IN-CLUSTER before the timeline starts (a laptop-side
#    `oc logs -f` loses most lines under load). Truncate the previous capture first.
oc apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/ipp-logtail-pod.yaml
oc exec -n "$NAMESPACE" access-to-harness-data-workload-pvc -- \
  truncate -s 0 /requests/smart-logs/ipp-full-live.log
oc get pod ipp-logtail -n "$NAMESPACE"        # must be Running

# 5. Run the 7-stage timeline and slice each stage window out of the capture.
#    Stages are discrete: each launches its own 300s runs and waits for all of them.
ipp_benchmarking/tools/adaptive_toggle_run.sh adaptive

llmdbenchmark --spec cicd/ocp-gemma-qwen-adaptive teardown -p "$NAMESPACE"
helm uninstall payload-processor -n "$NAMESPACE"
```

**Plot** — per-stage figures and tables (shared-probe percentiles per pool,
routing share over time):

```bash
.venv/bin/python3 ipp_benchmarking/tools/plot_adaptive_stages.py \
  ipp_benchmarking/example_outputs/gemma-qwen-adaptive/adaptive -o /tmp/adaptive
```

### Files

| Purpose | Path |
|---|---|
| Scenario / spec | `config/scenarios/cicd/ocp-gemma-qwen-adaptive.yaml` (+ `config/specification/…`) |
| IPP values | `ipp_benchmarking/ipp_configs/adaptive-gemma-qwen-{smart,random}-values.yaml` |
| Base-model ConfigMaps | `ipp_benchmarking/ipp_configs/{gemma-26b,qwen36-35b}-base-model.yaml` |
| Workloads | `workload/profiles/inference-perf/adaptive_{shared,gemma,qwen}_summarization.yaml.in` |
| Run driver / single stage | `ipp_benchmarking/tools/adaptive_toggle_run.sh`, `tools/stage.sh` |
| Stage extractor, figures | `ipp_benchmarking/tools/ipp_extract_stage.sh`, `tools/plot_adaptive_stages.py` |
