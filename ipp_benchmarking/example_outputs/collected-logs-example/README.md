# Example `collected-logs-` directory

This is one full output directory produced by `collect_logs.sh` after a run — a
reference for what a benchmark run captures. It happens to be the most complete
example available because the harness metrics scrape was enabled, so it contains
**every** collection type at once.

## The run that produced it

| | |
|---|---|
| Scenario | `cicd/ocp-qwen3-multi` on OpenShift (2×H100-80GB cluster) |
| Model | **Qwen/Qwen3-30B-A3B** (MoE) — single-model standup |
| IPP picker | **smart** (inflight-requests scorer + max-score picker) |
| Harness | `inference-perf` |
| Workload | `scrape_smoke.yaml` — Poisson **20 RPS for 30 s**, shareGPT (≤256 in / ≤32 out) |
| Metrics scrape | **on** (`monitoring.metricsScrapeEnabled: true`) — scrapes vLLM + EPP `/metrics` every 10 s |
| Routing path | client → Gateway/Istio → IPP ext_proc → EPP/router → vLLM |

It was a short smoke run to demonstrate collection, not a saturation sweep — so
the numbers are light (≈600 requests, KV barely moves). For real comparisons see
the saturation plots in `../run_comparison_saturation/`.

## What's in here

```
collected-logs-example/
├── payload-processor-*.log              # IPP (ext_proc) logs — routing decisions
├── *-gaie-epp-*.log                     # EPP / inference-scheduler logs
├── *-decode-*.log                       # vLLM decode pod logs (Running/Waiting/GPU KV cache usage)
├── infra-llmdbench-inference-gateway-*  # Istio gateway logs
├── gpu-dcgm/                            # per-GPU DCGM metrics for the run window
│   ├── DCGM_FI_PROF_PIPE_TENSOR_ACTIVE.json   # tensor-core duty cycle (FLOP usage)
│   ├── DCGM_FI_PROF_SM_ACTIVE.json / DRAM_ACTIVE.json
│   ├── DCGM_FI_DEV_GPU_UTIL.json / POWER_USAGE.json / PIPE_FP16_ACTIVE.json
│   └── summary.txt                      # per-backend mean(busy)/peak per metric
└── benchmark-results/
    ├── results/<experiment-id>/
    │   ├── config.yaml                  # rendered run config (model, endpoint, load)
    │   ├── run_metadata.yaml
    │   ├── scrape_smoke.yaml            # the rendered workload profile
    │   ├── summary_lifecycle_metrics.json      # aggregate latency/throughput
    │   ├── stage_0_lifecycle_metrics.json      # per-stage stats (rate, count, p50/p95, failures)
    │   ├── per_request_lifecycle_metrics.json  # one record per request (timings, tokens, served model)
    │   ├── benchmark_report*.yaml       # formatted reports (v0.1 + v0.2)
    │   ├── stdout.log / stderr.log / metrics_collection.log
    │   └── metrics/                     # harness metrics scrape (because it was enabled)
    │       ├── raw/                      # per-pod /metrics snapshots every 10 s (vLLM + EPP)
    │       ├── processed/                # aggregated per-pod stats + replica/startup snapshots
    │       └── graphs/                   # time-series PNGs
    └── analysis/<experiment-id>/        # post-run analysis artifacts
```

## Where each source comes from
- **Pod logs** (`*.log`) — `kubectl logs` of the IPP, EPP, decode, and gateway pods.
- **`benchmark-results/`** — copied from the harness PVC (the `results/` + `analysis/`
  the run wrote).
- **`gpu-dcgm/`** — queried from OpenShift monitoring (Thanos) by `tools/collect_dcgm.py`,
  which `collect_logs.sh` invokes automatically. The harness scrape does **not** capture
  GPU FLOPs; this is the only source for tensor-core / MFU analysis.
- **`metrics/`** (inside `results/`) — the in-cluster harness scrape of vLLM + EPP
  `/metrics`, present only when `metricsScrapeEnabled` is on.
