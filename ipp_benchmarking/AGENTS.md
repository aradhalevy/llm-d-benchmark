# AGENTS.md — IPP benchmarking handoff

Knowledge-transfer notes for an AI agent (or person) picking up these IPP
benchmarks. `README.md` is the step-by-step how-to; **this file is the
non-obvious stuff**: required fixes, gotchas that cost hours, and the findings
worth knowing before you start. Read it first.

The goal of every run: **did the IPP picker change routing and latency the way
we expected?** Collect both benchmark results *and* control-plane evidence (IPP
logs, EPP scores, decode KV/queue) — one without the other can't answer it.

---

## Orientation

- **IPP** is an Envoy `ext_proc` gRPC server between the Gateway and the llm-d
  Router/EPP. Default plugins inject `X-Gateway-Model-Name` (from request JSON
  `model`) and `X-Gateway-Base-Model-Name` (from labeled ConfigMaps) before the
  router schedules. With a scorer it also picks the backend model.
- **Picker = the experiment variable**, and the smart-vs-random difference is
  **baked into the image build**, not set via helm flags. Build two tagged
  variants and swap with `helm upgrade --set payloadProcessor.image.tag=...`
  (no teardown needed between pickers):
  - **smart** — inflight-requests scorer + max-score picker (load-aware; routes
    away from a busy backend).
  - **random** — max-score picker, **no scorer** (load-blind; spreads traffic
    across all listed pools regardless of load).
- Base-model ConfigMaps consumed by the default plugins **must** carry label
  `inference.llm-d.ai/ipp-managed: "true"`.
- Supplying **any** `--plugin` (or `listModels`) **replaces the entire default
  set** — include everything you need explicitly.
- Scorers must return scores in `[0, 1]`; out-of-range values are clamped. Some
  metric names still use the older `bbr` prefix.
- The gateway name stays `infra-llmdbench-inference-gateway` regardless of your
  namespace — it's derived from the release (`llmdbench`), not the project.

---

## Must-dos (skip one and the run breaks or lies)

1. **List only real backends.** The random picker routes proportionally to
   *every* listed model — list one with no backing pod and ~half the traffic
   404s into phantom pools and corrupts the run. On **OCP use the Qwen set only**
   (`Qwen/Qwen3-30B-A3B` + `Qwen/Qwen3-32B`), **never `facebook/opt-*`**; on Kind
   use the opt set. Keep `listModels` and the applied `ipp_configs/` in sync.
2. **OCP: patch `--max-model-len 8192` on both decode deployments** after standup
   or **Qwen3-32B crash-loops** (needs 10 GiB KV for the 40960 default, only
   ~8.25 available on one H100). The scenario's `maxModelLen: 8192` is ignored
   because `modelCommand: imageDefault` only exports `VLLM_MAX_MODEL_LEN`, which
   this vLLM doesn't read. Patch **both** pods for a fair comparison.
3. **IPP is NOT removed by `llmdbenchmark teardown`** — always
   `helm uninstall payload-processor -n <ns>` separately.
4. **Scope everything to your namespace** — `-p <ns>` on every `run`/`teardown`,
   `-n <ns>` on every `oc`/`kubectl`/`helm`. Omitting `-p` on `run` silently
   deploys the harness into the shared default namespace (`llmdbench`).
5. The `ocp-qwen3-multi` scenario sets `skipSmoketest: true` on purpose —
   header-match routing 404s until IPP injects `X-Gateway-Base-Model-Name`, so
   the auto-smoketest would fail.

---

## High-RPS / saturation runs

- **Harness sizing:** the default (cpu `1` / mem `16Gi`) thrashes at the shareGPT
  dataset-prep step. The `ocp-qwen3-multi` scenario already uses **cpu `8` / mem
  `32Gi`**.
- **A single inference-perf pod caps ~130–155 req/s** under saturation no matter
  the requested rate (it issues the full request count but stretches it over
  wall-clock). To truly offer 300–600 RPS, use **`-j N` parallel pods** (~150 RPS
  each). Always verify achieved vs requested in `stage_*_lifecycle_metrics.json`
  (`load_summary.count / benchmark_time_seconds`) before trusting a "500 RPS"
  label.
- **`-j N` needs RWX `workload-pvc`.** The parallel harness pods + the
  data-access helper all mount `workload-pvc`. With **ReadWriteOnce** they must
  co-locate on one node or hit Multi-Attach (see troubleshooting). The
  `ibm-spectrum-scale-fileset` storage class supports **ReadWriteMany** — set
  `shared.storage.workloadPvc.accessModes: [ReadWriteMany]` (recreate the PVC).
- Keep `--max-model-len` identical across compared runs — a smaller cap fits more
  requests in KV and **moves the saturation knee**.

---

## Findings worth knowing (so you can sanity-check results)

- **Smart vs random:** the random picker saturates the dense **Qwen3-32B by
  ~100 RPS** (throughput plateaus ~1830 tok/s, first 504s ~250 RPS) while the MoE
  sits idle. Smart routing offloads to the fast MoE and sustains far more
  (~2620 tok/s, 0 failures at 100 RPS). At true ~600 RPS (`-j 4`) both backends
  saturate with multi-thousand-deep queues.
- **Compute vs memory bound (GPU FLOPs):** the dense **Qwen3-32B is
  compute-bound** (DCGM tensor-active ~64%, MFU ~59%); the **MoE Qwen3-30B-A3B is
  memory-bound** (DRAM-active ~47% but tensor-active only ~22%, MFU ~20%). The
  MoE saturates on **HBM/KV**, not compute — it has large FLOP headroom (few
  active params/token). Both GPUs hit the ~700 W H100 TDP.
- **GPU FLOPs are NOT in the harness metrics scrape.** `metricsScrapeEnabled`
  scrapes vLLM + EPP `/metrics` (KV, queue, throughput, latency, prefix-cache,
  `inference_pool_*`, `inference_extension_*`) — useful, but **no DCGM/FLOPs**.
  vLLM exposes `vllm:estimated_flops_per_gpu_total` but it reads **0** on the
  pinned build. GPU FLOPs come **only** from `tools/collect_dcgm.py` (DCGM via
  Thanos, auto-run by `collect_logs.sh`). Use `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`
  — **not** `DCGM_FI_DEV_GPU_UTIL` (pinned ~100%, useless) nor `PIPE_FP16_ACTIVE`
  (~0, vLLM is BF16). Analyze with `tools/compute_mfu.py` (MFU =
  `2·active_params·tokens / peak_FLOPs`, H100 BF16 989.4 TFLOP/s).

---

## Troubleshooting

**Namespace scoping.** See must-do #4. Cluster-scoped `llmdbench-modelservice-*`
ClusterRoles are shared — teardown removes the set it created; don't hand-delete
them while another project uses the stack.

**RWO `workload-pvc` Multi-Attach + the stale helper pod.** With RWO, if a
harness pod lands on a different node than the `access-to-harness-data-workload-pvc`
helper it hangs in `ContainerCreating` (`Multi-Attach error ... already used by
pod(s) access-to-harness-data-workload-pvc`) until the 3600 s timeout. Unblock by
moving the helper onto the harness's node (dump YAML, set `spec.nodeName`,
force-delete, re-apply). **Gotcha:** a manually-recreated helper then blocks the
next run's step 02 (`Forbidden: pod updates may not change fields other than
image...`) — so **delete any leftover `access-to-harness-data-workload-pvc` pod
before a new run.** Durable fix: RWX `workload-pvc` (above).

**A node without the storage CSI driver.** With RWX, a pod can land on a node
missing the `spectrum-scale` CSI driver and fail to attach
(`CSINode ... does not contain driver spectrumscale.csi.ibm.com`). Cordoning the
node is cluster-scoped (may be disallowed). Namespace-scoped fix: pin pods to
storage-capable nodes via a label only they carry —
`oc annotate namespace <ns> openshift.io/node-selector="scale=true"` (revert
after). Confirm your good nodes have the label first.

**Harness HuggingFace egress fails on some nodes.** Some workers can't reach HF
over IPv4 → tokenizer init fails (IPv6 nodes succeed). The router serves both
models regardless of which stack's harness launched, so a single healthy stack's
results are usable; re-running usually reschedules onto a working node.

**`route already exists` during standup** — non-fatal leftover route from a prior
run; `All standup steps complete` still prints. Ignore, or
`oc delete route llmdbench-inference-gateway-route` first.

**`collect_logs.sh` finds no workspace.** It greps `/tmp` at depth 1, but the
workspace may live under `/tmp/<user-or-runtime>/`. Point it explicitly:
`export LLMDBENCH_WORKSPACE=$(ls -dt /tmp/**/workspace_llmdbench_* | head -1)`
then re-run.

**Benign warmup noise** (not fatal): vLLM `Unknown vLLM environment variable
detected: VLLM_MAX_MODEL_LEN` (expected — that's why the `--max-model-len` patch
is needed); EPP `metric family "vllm:lora_requests_info" not found`; brief
`503 "decode node is not ready"` during pod warmup.

**Custom plugins need their Envoy events.** Supplying `--plugin` overrides all
defaults — include every plugin. A plugin that reads the body sees nothing unless
`provider.supportedEvents.requestBody` / `responseBody` is `true`.

**`provider.messageTimeout=10s` is required.** Without it Envoy's ~200 ms default
trips `HTTP 504 ext_proc_error_per-message_timeout_exceeded` once IPP defers
header ACKs under load.

**helm-diff must be ≥ v3.14 with Helm 4.** Helm 4 removed `--validate`; helm-diff
≤ v3.13 still passes it and fails. Reinstall:
`helm plugin install https://github.com/databus23/helm-diff --version v3.15.7`.

**Don't bump `llm-d-inference-sim` to `latest` on Kind.** `latest` needs an HTTP
render sidecar on `localhost:8082` and crash-loops without one. The
`kind-sim-multi` scenario is pinned to a UDS-based tag that works as-is.

---

## Environment specifics (these were ours — adapt to yours)

- OpenShift project `llm-d-arad` on cluster `api.pokprod001.ete14.res.ibm.com`.
- IPP image at `ghcr.io/<user>/llm-d-inference-payload-processor` with distinct
  tags per picker; new ghcr packages are **private** by default (make public or
  add a pull secret).
- Model PVCs are **not reliably preserved** across teardown — expect a fresh
  ~60–64 GB download per Qwen model on standup.
- Secrets/config in repo `.env` (gitignored): `HF_TOKEN`, `IPP_PATH`, `GHCR_PAT`.
  `source .venv/bin/activate && source .env` before running.
