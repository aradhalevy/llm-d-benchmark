# Running multi-model IPP benchmarks

Evaluate IPP routing (a load-aware **smart** picker vs. a static baseline)
end-to-end with [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark).

**Read [`AGENTS.md`](./AGENTS.md) first** — the required fixes, gotchas, and
findings live there, not here.

Run `llmdbenchmark` **from the repo root** — it auto-discovers this bundle's
scenarios/specs/profiles (e.g. `--spec cicd/ocp-qwen3-8b-32b`,
`-w summarization_concurrency_8b32b.yaml`). `example_outputs/` ships sample plots
+ writeups only; raw run data is too large to commit.

---

## Kind quick-start (simulator, no GPU)

Two fake-latency sims — `facebook/opt-125m` (slow) + `opt-350m` (fast) — for a
fast local smoke test of routing.

```bash
kind create cluster
docker pull ghcr.io/llm-d/llm-d-benchmark:v0.6.3 && kind load docker-image ghcr.io/llm-d/llm-d-benchmark:v0.6.3
# build + side-load the IPP image from the IPP repo (tag per scorer-picker):
make image-build && kind load docker-image ghcr.io/llm-d/llm-d-inference-payload-processor:smartRouting

./install.sh && source .venv/bin/activate
llmdbenchmark --spec cicd/kind-sim-multi standup -p llmdbench

export IPP_PATH=/path/to/llm-d-inference-payload-processor
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n llmdbench --set provider.name=istio \
  --set payloadProcessor.image.tag=smartRouting --set payloadProcessor.image.pullPolicy=Never \
  --set provider.supportedEvents.requestBody=true --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseBody=true --set provider.messageTimeout=1200s \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway \
  --set 'payloadProcessor.listModels[0]=facebook/opt-125m' --set 'payloadProcessor.listModels[1]=facebook/opt-350m'
kubectl apply -n llmdbench -f ipp_benchmarking/ipp_configs/opt-125m-base-model.yaml \
                           -f ipp_benchmarking/ipp_configs/opt-350m-base-model.yaml
# give the sims different TTFT/ITL so routing has something to optimize — see AGENTS.md.

llmdbenchmark --spec cicd/kind-sim-multi run -l inference-perf -w sanity_random.yaml
NAMESPACE=llmdbench ./ipp_benchmarking/collect_logs.sh        # -> ./collected-logs-<N>/
llmdbenchmark --spec cicd/kind-sim-multi teardown -p llmdbench
helm uninstall payload-processor -n llmdbench                 # IPP isn't torn down automatically
```

---

## OpenShift A/B: smart routing vs static single-model baselines (ocp-qwen3-8b-32b)

**One script, one run at a time.** `tools/ab_routing_run.sh <arm> <ipp_values_file>
[profile]` runs exactly one thing per invocation — it patches the values file's
routing config into the IPP, restarts, runs the given profile (no planner, so the
decision log is clean), warms the backend(s), then collects logs + the slim extract +
routing analysis. The A/B is three such runs:

- **static baselines** — register **only one model** (the values file's `listModels`
  has a single entry, so there is no routing choice) and drive its **half-conc**
  profile. `static-8b-only-values.yaml` + `half_8b.yaml` and
  `static-32b-only-values.yaml` + `half_32b.yaml`, each C=25→275→25 so that
  **8B@C/2 + 32B@C/2 sums to the smart run's full C**. The static split has no spill
  path: when the 8B leg saturates there is nowhere to offload.
- **smart** (`median-ttft-ocp-values.yaml`, **both** models registered, **no**
  `model-name-filter` so `sweep_8stage`'s single-string `model_name` lets the
  model-selector pick from both `listModels`): `queue-ttft-scorer` (predicts
  per-request TTFT under load via the `ttft-percentile-extractor`, routes to the
  lowest predicted TTFT, with `explorationRate: 0.1`) + `max-score-picker` keeps
  most traffic on the fast 8B and offloads overflow to the 32B as load climbs.
  Drives the full `sweep_8stage.yaml` (C=50→550→50, 11 stages, ~43k reqs,
  ~5 min/stage; the symmetric up/down legs expose routing hysteresis). Needs an
  IPP image built with the queue-ttft-scorer (set its tag in step 2). For the
  avg-ttft scorer instead, swap in `avgttft-ocp-values.yaml` (also no filter).

All timeouts are lifted (route + client `request_timeout` to 1200s, ext-proc
`messageTimeout` to 1200s) so nothing is shed — the delta is latency/throughput,
not failures. Smart keeps **85–96%** of load on the fast 8B and spills overflow
to the 32B; the static split cannot.

```bash
# Prereqs: OCP cluster w/ H100-80GB, `oc login`, HF_TOKEN with Qwen access,
#          IPP image in a registry the cluster pulls.
export NAMESPACE=llm-d-<you>
export IPP_PATH=/path/to/llm-d-inference-payload-processor

# 1. Standup BOTH pools. (AGENTS.md must-do: patch --max-model-len 8192 on the
#    32B decode deploy after standup or it crash-loops.)
llmdbenchmark --spec cicd/ocp-qwen3-8b-32b standup -p "$NAMESPACE"

# 2. Install IPP. `VALUES` selects which model(s) are REGISTERED (listModels): the
#    single-model baselines register one model, the smart run registers both. Re-run
#    this same helm upgrade with the matching values file before each phase below.
export VALUES=$IPP_PATH/ipp_benchmarking/ipp_configs/static-8b-only-values.yaml   # then -32b-only, then median-ttft-ocp-values
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio -f "$VALUES" \
   --set provider.supportedEvents.requestBody=true --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseBody=true
    --set 'payloadProcessor.flags.v=4' \
  --set payloadProcessor.image.registry=ghcr.io/<you> --set payloadProcessor.image.tag=<tag> \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=1200s
kubectl apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/qwen3-8b-base-model.yaml \
                              -f ipp_benchmarking/ipp_configs/qwen3-32b-base-model.yaml
# Header-match routes (scenario sets httpRoute.enabled: false; standup renders none).
# Pool names are derived from $NAMESPACE -- no hand-editing.
ipp_benchmarking/tools/gen_httproutes.sh "$NAMESPACE" | kubectl apply -f -

# No timeouts: lift the per-route 30s request timeout (the sole load-shedder) so requests
# complete instead of being shed. ab_routing_run.sh re-applies this per arm.
for r in $(kubectl get httproute -n "$NAMESPACE" -o name | grep -E 'qwen3-(8b|32b)'); do
  kubectl patch "$r" -n "$NAMESPACE" --type=json \
    -p='[{"op":"replace","path":"/spec/rules/0/timeouts/request","value":"1200s"}]'
done

# 3. Edit the vars at the top of tools/ab_routing_run.sh: REPO (repo path), NS
#    (namespace), GW (in-cluster gateway URL — embeds the namespace).

# 4. Run ONE arm at a time. Each block below is a SINGLE command (note the trailing
#    backslashes) with three positional args:
#        ab_routing_run.sh  <arm>  <ipp_values_file>  <profile>
#    Before each, re-run step 2's helm upgrade with the matching VALUES so the IPP
#    registers the right model(s).
#    Single-model baselines first (each registers one model -> no routing choice):
ipp_benchmarking/tools/ab_routing_run.sh \
    static_8b \
    ipp_benchmarking/ipp_configs/static-8b-only-values.yaml \
    half_8b.yaml
ipp_benchmarking/tools/ab_routing_run.sh \
    static_32b \
    ipp_benchmarking/ipp_configs/static-32b-only-values.yaml \
    half_32b.yaml
#    Then smart routing across both (pass the full sweep profile explicitly):
ipp_benchmarking/tools/ab_routing_run.sh \
    smart \
    ipp_benchmarking/ipp_configs/median-ttft-ocp-values.yaml \
    sweep_8stage.yaml

# 5. Plot (exact usage is in each script's docstring header). Run plotters with
#    the venv python — they need matplotlib/numpy: `source .venv/bin/activate` (or
#    prefix `.venv/bin/python3`). ab_routing_run.sh already built each arm's inputs:
#    per_request_slim.json (latency plotter) + decisions_*.log (routing plotter).
#    The static legs ran at HALF conc (25→275→25) but correspond to the smart run's
#    full C (8B@25 == full C=50), so label all runs with the same full-equivalent list.
D=ipp_benchmarking/example_outputs/ocp-research-agent-routing
.venv/bin/python3 ipp_benchmarking/tools/plot_routing_vs_concurrency_ocp.py "smart"=$D/smart --concurrencies 50,150,250,350,450,550,450,350,250,150,50 -o $D/routing_vs_concurrency_ab.png
.venv/bin/python3 ipp_benchmarking/tools/plot_latency_vs_concurrency_ocp.py "smart"=$D/smart "static_8b"=$D/static_8b "static_32b"=$D/static_32b --concurrencies 50,150,250,350,450,550,450,350,250,150,50 -o $D/latency_vs_concurrency_ab.png
```

`ab_routing_run.sh` writes into `example_outputs/ocp-research-agent-routing/<arm>/`
(a re-run overwrites the committed sample). The smart run drives `sweep_8stage.yaml`;
the static baselines drive `half_8b.yaml` / `half_32b.yaml` — the IPP config + the
registered model(s) are what change between runs.

**Plotting a plain `llmdbenchmark run`** (not via `ab_routing_run.sh`): the raw
`per_request_lifecycle_metrics.json` is 100MB–1GB and often truncated, so build the
slim extract the OCP **latency** plotter needs first (the **routing** plotter needs
the IPP decision log, which only `ab_routing_run.sh` captures):

```bash
.venv/bin/python3 ipp_benchmarking/tools/extract_per_request_slim.py \
  <results>/.../per_request_lifecycle_metrics.json  <arm_dir>/per_request_slim.json
# -> "<n> records  8B=<x> 32B=<y> fail=<z>"; then point the latency plotter at <arm_dir>
```

### OpenShift: decode pods crash with `getpwuid(): uid not found`

OpenShift runs the vLLM container under an arbitrary high UID (the namespace's
range) that isn't in `/etc/passwd`, so torch's inductor cache setup calls
`getpass.getuser()` -> `pwd.getpwuid()` and dies with
`KeyError: getpwuid(): uid not found`. Give `getuser()` a name via the
environment so it skips the passwd lookup -- patch every decode deployment
after standup:

```bash
for d in $(oc get deploy -n "$NAMESPACE" -o name | grep decode); do
  oc patch "$d" -n "$NAMESPACE" -p \
    '{"spec":{"template":{"spec":{"containers":[{"name":"vllm","env":[{"name":"USER","value":"vllm"},{"name":"LOGNAME","value":"vllm"}]}]}}}}'
done
```

### No HTTPRoutes after standup -> gateway 404s everything

IPP routes by the `X-Gateway-Base-Model-Name` header, which the default
PathPrefix route can't express, so the scenario sets `httpRoute.enabled: false`
and standup renders no route. Generate the header-match routes (pool names are
derived deterministically from the namespace + model, so nothing to hand-edit):

```bash
ipp_benchmarking/tools/gen_httproutes.sh "$NAMESPACE" | kubectl apply -f -
kubectl get httproute -n "$NAMESPACE"   # both routes should be Accepted/ResolvedRefs
```
