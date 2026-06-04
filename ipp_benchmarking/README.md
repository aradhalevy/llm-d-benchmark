# Running multi-model IPP benchmarks

Evaluate IPP routing (a **smart** load-aware picker vs. a **random** baseline)
end-to-end with [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark).

**Read [`AGENTS.md`](./AGENTS.md) first** — it has the required fixes, gotchas,
and findings (the "must-dos" and troubleshooting are there, not here).

`llmdbenchmark` discovers this folder automatically:
`--spec cicd/ocp-qwen3-multi` and `-w saturation_sweep_500.yaml` resolve the
bundle's scenarios/specs/profiles, while shared templates still come from the
repo root. Just **run `llmdbenchmark` from the repo root** (its default
`base_dir` is the cwd). 

Contents: `tools/` (GPU/MFU + plotting), `ipp_configs/` (base-model ConfigMaps),
`config/` (the `kind-sim-multi` + `ocp-qwen3-multi` scenarios & specs),
`workload/profiles/inference-perf/` (saturation/scrape profiles),
`collect_logs.sh`, and `example_outputs/` (sample comparison plots + story
HTMLs). Raw `collected-logs-` run data isn't included (too large).

Two scenarios:
- **Kind** (`cicd/kind-sim-multi`) — `llm-d-inference-sim` pods, no GPU. Two
  fake-latency models: `facebook/opt-125m` (slow) + `opt-350m` (fast).
- **OpenShift** (`cicd/ocp-qwen3-multi`) — real vLLM on H100-80GB.
  `Qwen3-30B-A3B` (MoE, fast) + `Qwen3-32B` (dense, slow).

The IPP wiring (helm install, base-model ConfigMaps) is shared; only the cluster
/ scenario / image-loading differs. IPP images: build a **smart** and a
**random** variant and swap with `helm upgrade` (see AGENTS.md).

---

# A. Kind (simulator)

```bash
# 1. cluster + harness image
kind create cluster
docker pull ghcr.io/llm-d/llm-d-benchmark:v0.6.3
kind load docker-image ghcr.io/llm-d/llm-d-benchmark:v0.6.3

# 2. build + side-load the IPP image (from the IPP repo; tag per scorer-picker)
make image-build
kind load docker-image ghcr.io/llm-d/llm-d-inference-payload-processor:smartRouting

# 3. install llmdbenchmark, then stand up the stack
./install.sh && source .venv/bin/activate
llmdbenchmark --spec cicd/kind-sim-multi standup -p llmdbench
```

Deploy IPP:

```bash
export IPP_PATH=/path/to/llm-d-inference-payload-processor
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n llmdbench --set provider.name=istio \
  --set payloadProcessor.image.tag=smartRouting \
  --set payloadProcessor.image.pullPolicy=Never \
  --set provider.supportedEvents.requestHeaders=true \
  --set provider.supportedEvents.requestBody=true \
  --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseHeaders=false \
  --set provider.supportedEvents.responseBody=true \
  --set provider.supportedEvents.responseTrailers=false \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway \
  --set provider.messageTimeout=10s \
  --set 'payloadProcessor.listModels[0]=facebook/opt-125m' \
  --set 'payloadProcessor.listModels[1]=facebook/opt-350m'
kubectl apply -n llmdbench -f ipp_benchmarking/ipp_configs/opt-125m-base-model.yaml \
                           -f ipp_benchmarking/ipp_configs/opt-350m-base-model.yaml
```

Give the sim models different latency so routing has something to optimize
(slow `opt-125m`: TTFT 3s / ITL 200ms; fast `opt-350m`: TTFT 1s / ITL 50ms):

```bash
for dep in facebook-16adac56-opt-125m-decode facebook-16adac56-opt-125m-prefill; do
  kubectl patch deployment "$dep" -n llmdbench --type=json -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--time-to-first-token=3s"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--inter-token-latency=200ms"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-num-seqs=10"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-waiting-queue-length=20"}]'
done
for dep in facebook-a5673f1c-opt-350m-decode facebook-a5673f1c-opt-350m-prefill; do
  kubectl patch deployment "$dep" -n llmdbench --type=json -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--time-to-first-token=1s"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--inter-token-latency=50ms"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-num-seqs=10"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-waiting-queue-length=20"}]'
done
```

Run, collect, tear down:

```bash
llmdbenchmark --spec cicd/kind-sim-multi run -l inference-perf -w sanity_random.yaml
NAMESPACE=llmdbench ./ipp_benchmarking/collect_logs.sh     # -> ./collected-logs-<N>/
llmdbenchmark --spec cicd/kind-sim-multi teardown -p llmdbench
helm uninstall payload-processor -n llmdbench               # IPP isn't torn down automatically
```

---

# B. OpenShift (real GPUs)

Prereqs: `oc whoami` works, ≥2 nodes with `NVIDIA-H100-80GB-HBM3`, NVIDIA GPU
Operator, a ≥200Gi StorageClass. Set your project once: `export NAMESPACE=llm-d-<user_name>`.

```bash
# namespace + HF secret
oc new-project "${NAMESPACE}" || oc project "${NAMESPACE}"
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

build IPP to a registry the cluster can pull from (if its a private ghcr -> [make public in your account](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility#configuring-visibility-of-packages-for-your-personal-account))
```bash
echo $GHCR_PAT | docker login ghcr.io -u <user_name> --password-stdin
VERSION=smartRouting make image-build
docker push ghcr.io/<user_name>/llm-d-inference-payload-processor:smartRouting

# stand up (pulls ~60GB + ~64GB Qwen weights to the model PVC)
llmdbenchmark --spec cicd/ocp-qwen3-multi standup -p "${NAMESPACE}"
```

Deploy IPP:

```bash
export IPP_PATH=/path/to/llm-d-inference-payload-processor
helm upgrade --install payload-processor "$IPP_PATH/config/charts/payload-processor/" \
  -n "${NAMESPACE}" --set provider.name=istio \
  --set payloadProcessor.image.registry=ghcr.io/<user_name> \
  --set payloadProcessor.image.repository=llm-d-inference-payload-processor \
  --set payloadProcessor.image.tag=smartRouting \
  --set payloadProcessor.image.pullPolicy=IfNotPresent \
  --set provider.supportedEvents.requestHeaders=true \
  --set provider.supportedEvents.requestBody=true \
  --set provider.supportedEvents.requestTrailers=true \
  --set provider.supportedEvents.responseHeaders=false \
  --set provider.supportedEvents.responseBody=true \
  --set provider.supportedEvents.responseTrailers=false \
  --set inferenceGateway.name=infra-llmdbench-inference-gateway \
  --set provider.messageTimeout=10s \
  --set 'payloadProcessor.listModels[0]=Qwen/Qwen3-30B-A3B' \
  --set 'payloadProcessor.listModels[1]=Qwen/Qwen3-32B'
kubectl apply -n "${NAMESPACE}" -f ipp_benchmarking/ipp_configs/qwen3-30b-a3b-base-model.yaml \
                                -f ipp_benchmarking/ipp_configs/qwen3-32b-base-model.yaml
```

**Required after standup** — patch `--max-model-len 8192` on both decode pods or
Qwen3-32B crash-loops (see AGENTS.md for why):

```bash
for dep in qwen-qwe-de682216-wen3-32b-decode qwen-qwe-65acc96e--30b-a3b-decode; do
  kubectl patch deployment "$dep" -n "${NAMESPACE}" --type=json -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--max-model-len"},
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"8192"}]'
done
# deployment names carry a per-standup hash — confirm with: kubectl get deploy -n "${NAMESPACE}"
```

Run, collect, tear down:

```bash
llmdbenchmark --spec cicd/ocp-qwen3-multi run -l inference-perf -w saturation_sharegpt.yaml -p "${NAMESPACE}"
NAMESPACE="${NAMESPACE}" ./ipp_benchmarking/collect_logs.sh   # IPP/EPP/decode logs + benchmark-results + gpu-dcgm/
llmdbenchmark --spec cicd/ocp-qwen3-multi teardown -p "${NAMESPACE}"   # add -d to also clear the model PVC
helm uninstall payload-processor -n "${NAMESPACE}"
```

## Saturation sweeps

Profiles in `workload/profiles/inference-perf/`: `saturation_sweep.yaml`
(200/250/300) and `saturation_sweep_500.yaml` (200/300/400/500), each stage 25 s
load + 25 s cooldown. A/B the pickers by `helm upgrade`-ing `image.tag` between
runs (no teardown). **A single harness caps ~150 RPS — use `-j N` for higher**
(see AGENTS.md for the RWX/node requirements).

```bash
llmdbenchmark --spec cicd/ocp-qwen3-multi run -l inference-perf \
  -w saturation_sweep_500.yaml -p "${NAMESPACE}" -j 4
```

Compare with the plot tools (in `ipp_benchmarking/tools/`), passing
`random=collected-logs-N smart=collected-logs-M`: `plot_saturation_comparison.py`,
`plot_latency_and_failures.py`, `plot_picker_decisions_v2.py`.

## GPU compute / FLOPs (MFU)

`ipp_benchmarking/collect_logs.sh` auto-archives per-GPU DCGM metrics to
`collected-logs-<N>/gpu-dcgm/`. Plot with `ipp_benchmarking/tools/query_gpu_utilization.py`;
compute MFU with `ipp_benchmarking/tools/compute_mfu.py`. (Dense = compute-bound,
MoE = memory-bound — see AGENTS.md.)

---

**Troubleshooting, required fixes, and findings → [`AGENTS.md`](./AGENTS.md).**
