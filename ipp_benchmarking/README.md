# Running multi-model IPP benchmarks

Evaluate IPP routing (a load-aware **smart** picker vs. a static baseline)
end-to-end with [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark).

**Read [`AGENTS.md`](./AGENTS.md) first** — the required fixes, gotchas, and
findings live there, not here.

Run `llmdbenchmark` **from the repo root**; it auto-discovers this bundle's
scenarios/specs/profiles (e.g. `--spec cicd/ocp-qwen3-8b-32b`,
`-w summarization_concurrency_8b32b.yaml`). Contents: `tools/` (plotting +
GPU/MFU), `ipp_configs/` (base-model ConfigMaps + IPP helm values), `config/`
(scenarios & specs), `workload/profiles/inference-perf/`, `collect_logs.sh`,
and `example_outputs/` (sample plots + writeups; raw run data is too large to
ship).

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

## OpenShift A/B: smart routing vs random (ocp-qwen3-8b-32b)

**One workload, two models.** A single **summarizer** harness offers *both*
models on every request — a JSON **string-array** `model_name`
(`'["Qwen/Qwen3-8B","Qwen/Qwen3-32B"]'`) means "IPP, pick one". The 8B is the
fast default; the 32B sits idle until the 8B saturates. Ramp C=100→500 (6000
reqs). The **A/B delta is only the IPP config** — same workload, same models:

- **smart** (`blog-ocp-ttft-only-values.yaml`): `avg-ttft-scorer` +
  `max-score-picker` — keeps most traffic on the fast 8B and offloads only the
  overflow to the 32B as load climbs (8B share 80%, 32B 20%, rising with
  concurrency).
- **random** (`maxscore-baseline-values.yaml`): `max-score-picker`, no scorer —
  ties break randomly, so it acts as a random picker: a load-blind **~50/50**
  split that floods the 32B from the start.

All timeouts are lifted (route + client `request_timeout` to 1200s, ext-proc
`messageTimeout` to 1200s) so nothing is shed — the delta is latency/throughput,
not failures.

Result ([`example_outputs/ocp-timeout-sweep-50-550/`](./example_outputs/ocp-timeout-sweep-50-550/)):
**0 failures both arms.** smart keeps **85–96%** on the fast 8B (random ~50/50),
giving **~5× lower p95** (smart 72s vs random 366s at peak) and **~3–4×
throughput** (2400–3600 vs ~800 tok/s) — the load-blind 50/50 bottlenecks on the
slow 32B instead.

One script per arm (`tools/ab_routing_run.sh <arm> <ipp_config>`) swaps the IPP
ConfigMap + restarts, runs the **single** summarizer harness (no planner → the
IPP decision log is uncontaminated), warms both backends through the cold-start
window, then collects logs + the slim per-request extract + routing analysis.

```bash
# Prereqs: OCP cluster w/ H100-80GB, `oc login`, HF_TOKEN with Qwen access,
#          IPP image in a registry the cluster pulls.
export NAMESPACE=llm-d-<you>
export IPP_PATH=/path/to/llm-d-inference-payload-processor

# 1. Standup BOTH pools. (AGENTS.md must-do: patch --max-model-len 8192 on the
#    32B decode deploy after standup or it crash-loops.)
llmdbenchmark --spec cicd/ocp-qwen3-8b-32b standup -p "$NAMESPACE"

# 2. Install IPP with the smart (avg-ttft) values; ab_routing_run.sh swaps the
#    config per arm afterwards.
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio \
  -f ipp_benchmarking/ipp_configs/blog-ocp-ttft-only-values.yaml \
  --set payloadProcessor.image.registry=ghcr.io/<you> --set payloadProcessor.image.tag=<tag> \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=1200s
kubectl apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/qwen3-8b-base-model.yaml \
                              -f ipp_benchmarking/ipp_configs/qwen3-32b-base-model.yaml

# No timeouts: lift the per-route 30s request timeout (the sole load-shedder) so requests
# complete instead of being shed. ab_routing_run.sh re-applies this per arm.
for r in $(kubectl get httproute -n "$NAMESPACE" -o name | grep -E 'qwen3-(8b|32b)'); do
  kubectl patch "$r" -n "$NAMESPACE" --type=json \
    -p='[{"op":"replace","path":"/spec/rules/0/timeouts/request","value":"1200s"}]'
done

# 3. Edit the vars at the top of tools/ab_routing_run.sh: REPO, NS, GW.

# 4. Run each arm (one per invocation). The 2nd arg is the IPP values file; the
#    script patches its customConfig into the cm + restarts IPP for that arm.
ipp_benchmarking/tools/ab_routing_run.sh smart  ipp_benchmarking/ipp_configs/blog-ocp-ttft-only-values.yaml
ipp_benchmarking/tools/ab_routing_run.sh random ipp_benchmarking/ipp_configs/maxscore-baseline-values.yaml

# 5. Plot (exact usage is in each script's docstring header).
D=ipp_benchmarking/example_outputs/ocp-research-agent-routing
ipp_benchmarking/tools/plot_routing_vs_concurrency_ocp.py "smart"=$D/smart "random"=$D/random --concurrencies 100,200,300,400,500 -o $D/routing_vs_concurrency_ab.png
ipp_benchmarking/tools/plot_latency_vs_concurrency_ocp.py "smart"=$D/smart "random"=$D/random --concurrencies 100,200,300,400,500 -o $D/latency_vs_concurrency_ab.png
```

`ab_routing_run.sh` writes into `example_outputs/ocp-research-agent-routing/<arm>/`
(a re-run overwrites the committed sample). Both arms drive the same
`summarization_concurrency_8b32b.yaml` profile — only the IPP config changes.

---

**Troubleshooting, required fixes, and findings → [`AGENTS.md`](./AGENTS.md).**
