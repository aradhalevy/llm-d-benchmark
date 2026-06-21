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
  --set provider.supportedEvents.responseBody=true --set provider.messageTimeout=10s \
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

## OpenShift research-agent A/B (ocp-qwen3-8b-32b)

A deep-research agent: a **Qwen3-8B summarizer** (the saturating workload,
hundreds of page-summary calls) + a **Qwen3-32B planner** (rare, pinned). The
inflight-aware **avg-ttft scorer** offloads summarizer overflow onto the idle
32B when the 8B saturates. Headline result
([`example_outputs/ocp-research-agent-blog/`](./example_outputs/ocp-research-agent-blog/README.md)):
**+35% summaries completed, 33% vs 51% failures, ~2.2–2.4× goodput** at the
C=400/500 ceiling — paid for by planner contention on the shared 32B.

One script per arm (`tools/ab_blog_run.sh`) drives the summarizer via
`llmdbenchmark` **and** an in-pod curl planner (no second namespace), captures
IPP decisions through two unioned/deduped log nets, pulls the OpenShift logs,
and slim-extracts the giant per-request file in-pod. Standup deploys **both**
pools (`cicd/ocp-qwen3-8b-32b`); the run uses the single-harness `-summarizer`
spec so one harness is the only load source (IPP still routes across both). The
A/B delta is just the summarizer's `model_name`: a JSON **string-array**
(`'["Qwen/Qwen3-8B","Qwen/Qwen3-32B"]'` → scorer picks) vs. a pinned single
name.

```bash
# Prereqs: OCP cluster w/ H100-80GB, `oc login`, HF_TOKEN with Qwen access,
#          IPP image (smart avg-ttft build) in a registry the cluster pulls.
export NAMESPACE=llm-d-<you>
export IPP_PATH=/path/to/llm-d-inference-payload-processor

# 1. Standup BOTH pools. (AGENTS.md must-do: patch --max-model-len 8192 on the
#    32B decode deploy after standup or it crash-loops.)
llmdbenchmark --spec cicd/ocp-qwen3-8b-32b standup -p "$NAMESPACE"

# 2. Install IPP — pure avg-ttft inflight-aware scorer.
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "$NAMESPACE" --set provider.name=istio \
  -f ipp_benchmarking/ipp_configs/avgttft-blog-values.yaml \
  --set payloadProcessor.image.registry=ghcr.io/<you> --set payloadProcessor.image.tag=<tag> \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway --set provider.messageTimeout=10s
kubectl apply -n "$NAMESPACE" -f ipp_benchmarking/ipp_configs/qwen3-8b-base-model.yaml \
                              -f ipp_benchmarking/ipp_configs/qwen3-32b-base-model.yaml

# 3. Edit the vars at the top of tools/ab_blog_run.sh: REPO, NS, GW.

# 4. Run each arm (one arm per invocation). Restart IPP between arms — the
#    in-flight counter leak freezes a saturated EMA otherwise (AGENTS.md).
ipp_benchmarking/tools/ab_blog_run.sh arm_a_smart  summarization_concurrency_8b32b.yaml   # smart: model array -> scorer offloads
kubectl rollout restart deploy/payload-processor -n "$NAMESPACE"
ipp_benchmarking/tools/ab_blog_run.sh arm_b_static summarization_static_8b.yaml           # baseline: pinned 8B

# 5. Plot (exact usage is in each script's docstring header).
D=ipp_benchmarking/example_outputs/ocp-research-agent-blog
ipp_benchmarking/tools/plot_latency_vs_concurrency_ocp.py "smart"=$D/arm_a_smart "static"=$D/arm_b_static --concurrencies 100,200,300,400,500 -o $D/latency_vs_concurrency_ab.png
ipp_benchmarking/tools/plot_routing_vs_concurrency_ocp.py "smart"=$D/arm_a_smart "static"=$D/arm_b_static --concurrencies 100,200,300,400,500 --planner 150 -o $D/routing_vs_concurrency_ab.png
ipp_benchmarking/tools/plot_planner_latency_ab.py "smart"=$D/arm_a_smart/planner "static"=$D/arm_b_static/planner --concurrencies 1,2,3,4,5 -o $D/planner_latency_ab.png
```

`ab_blog_run.sh` writes into `example_outputs/ocp-research-agent-blog/<arm>/`
(a re-run overwrites the committed sample). The smart-vs-**random** routing
variant uses `ab_routing_run.sh` with `ipp_configs/blog-ocp-random-values.yaml`
→ `example_outputs/ocp-research-agent-routing/`.

---

**Troubleshooting, required fixes, and findings → [`AGENTS.md`](./AGENTS.md).**
