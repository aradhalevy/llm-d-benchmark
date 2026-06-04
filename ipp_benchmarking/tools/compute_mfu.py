#!/usr/bin/env python3
"""Model FLOPs Utilization (MFU) per backend from collected benchmark logs.

MFU = achieved_FLOPs / peak_FLOPs — the literal "fraction of the GPU's compute
we actually use." We get achieved FLOPs from the tokens each backend processed:

    FLOPs/token  ~=  2 * active_params       (forward pass, matmul-dominated)
    achieved FLOPs/s = 2 * active_params * (input_tokens/s + output_tokens/s)
    MFU(t) = achieved_FLOPs/s(t) / peak_FLOPs

This is the standard first-order estimate (it ignores attention and other
non-GEMM work, so it slightly *under*-counts FLOPs; in practice MFU is the
accepted efficiency number). The MoE vs dense contrast is the point: the MoE
spends ~2*3.3e9 FLOPs/token, the dense model ~2*32.8e9 — ~10x more compute for
the same token, so for equal throughput the dense GPU shows ~10x the MFU.

Per-backend token counts come from each request's served model + token counts
in per_request_lifecycle_metrics.json, so MFU is split per backend (the stage
summaries only give an aggregate across both pools).

Usage:
    compute_mfu.py random=collected-logs-43 smart=collected-logs-44 -o mfu.png
    compute_mfu.py smart=collected-logs-45 \
        --peak-tflops 989.4 \
        --active-params "Qwen3-30B-A3B=3.3e9,Qwen3-32B=32.8e9"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

MOE = "Qwen3-30B-A3B"
DENSE = "Qwen3-32B"
COLORS = {MOE: "#1f77b4", DENSE: "#d62728"}
# H100-80GB-HBM3 (SXM) BF16 Tensor Core peak, no sparsity (NVIDIA datasheet).
DEFAULT_PEAK_TFLOPS = 989.4
DEFAULT_ACTIVE_PARAMS = {MOE: 3.3e9, DENSE: 32.8e9}


def short(m: str) -> str:
    return m.rsplit("/", 1)[-1]


def backend_of(rec: dict) -> str:
    """Which backend actually SERVED this request. Read the served model from
    the response (vLLM echoes the pod's served_model_name), NOT the request
    model — the IPP picker can route a request to either backend, so the
    response is the source of truth for which GPU did the work."""
    resp = rec.get("response")
    s = resp if isinstance(resp, str) else json.dumps(resp) if resp else ""
    if "32B" not in s and "30B" not in s:
        # fall back to the captured raw_response inside info
        raw = ((rec.get("info") or {}).get("extra_info") or {}).get("raw_response")
        if raw:
            s = raw
    if "32B" in s or "32b" in s:
        return DENSE
    if "30B" in s or "30b" in s:
        return MOE
    return "?"


def load(run: Path):
    files = glob.glob(str(run / "benchmark-results" / "results" / "*" / "per_request_lifecycle_metrics.json"))
    if not files:
        return []
    best = max(files, key=lambda f: len(json.load(open(f))))
    out = []
    for r in json.load(open(best)):
        if r.get("error"):
            continue
        st, et = r.get("start_time"), r.get("end_time")
        info = r.get("info") or {}
        inp = info.get("input_tokens")
        ri = info.get("response_info") or {}
        outp = ri.get("output_tokens")
        if outp is None:
            outp = (ri.get("server_usage") or {}).get("completion_tokens")
        if not isinstance(st, (int, float)) or not isinstance(et, (int, float)):
            continue
        out.append({
            "t": st, "end": et,
            "in": int(inp or 0), "out": int(outp or 0),
            "m": backend_of(r),
        })
    out.sort(key=lambda x: x["t"])
    return out


def detect_stages(times, tmax, bin_s=5.0, thresh=40.0):
    """[(start,end)] of load stages from arrival density (gaps=cooldowns)."""
    bins = np.arange(0, tmax + bin_s, bin_s)
    cnt, _ = np.histogram(times, bins=bins)
    active = (cnt / bin_s) > thresh
    stages, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and (active[j + 1] or (j + 2 < len(active) and active[j + 2])):
                j += 1
            stages.append((bins[i], bins[min(j + 1, len(bins) - 1)]))
            i = j + 1
        else:
            i += 1
    return stages


def stage_mfu(recs, active_params, peak_flops, stages, t0):
    """Average MFU per backend within each detected load stage.

    MFU = (2*active_params * tokens_in_stage) / (stage_seconds * peak_FLOPs).
    Averaging over the whole stage (>=25s) avoids the prefill-burst artifact of
    per-second binning, and cannot exceed 100% (the GPU can't beat peak on
    average). Requests are attributed to the stage their prefill starts in."""
    out = []
    for (s, e) in stages:
        dur = e - s
        tok = sum(r["in"] + r["out"] for r in recs if s <= (r["t"] - t0) < e)
        flops_s = 2.0 * active_params * tok / dur if dur else 0.0
        out.append(flops_s / peak_flops * 100.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="mfu.png")
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS,
                    help="Per-GPU peak FLOPs in TFLOP/s (default H100 BF16 989.4).")
    ap.add_argument("--active-params", default=None,
                    help='Override active params, e.g. "Qwen3-30B-A3B=3.3e9,Qwen3-32B=32.8e9".')
    ap.add_argument("--rates", default="200,300,400,500",
                    help="comma-separated load-stage rates in order (x-axis labels)")
    args = ap.parse_args()
    rates = [r.strip() for r in args.rates.split(",") if r.strip()]

    peak = args.peak_tflops * 1e12
    active = dict(DEFAULT_ACTIVE_PARAMS)
    if args.active_params:
        for kv in args.active_params.split(","):
            k, v = kv.split("=")
            active[k.strip()] = float(v)

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        recs = load(Path(path))
        if recs:
            runs.append((label, recs))
    if not runs:
        print("no per-request data", file=sys.stderr)
        sys.exit(1)

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.0), squeeze=False)
    axes = axes[0]

    print(f"Peak per-GPU compute: {args.peak_tflops:.1f} TFLOP/s (BF16). "
          f"Active params: " + ", ".join(f"{short(k)}={v/1e9:.1f}B" for k, v in active.items()))
    for ax, (label, recs) in zip(axes, runs):
        print(f"\n=== {label} ===")
        t0 = min(r["t"] for r in recs)
        times = [r["t"] - t0 for r in recs]
        tmax = max(times)
        stages = detect_stages(times, tmax)
        xr = rates[:len(stages)] + [f"s{k}" for k in range(len(rates), len(stages))]
        x = np.arange(len(stages))
        for m in (MOE, DENSE):
            mr = [r for r in recs if r["m"] == m]
            if not mr:
                continue
            ap_m = active.get(m, active.get(short(m), 0))
            mfu = stage_mfu(recs and mr, ap_m, peak, stages, t0)
            ax.plot(x, mfu, marker="o", linewidth=2, color=COLORS[m], label=short(m))
            for xi, yi in zip(x, mfu):
                ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=COLORS[m])
            tin = sum(r["in"] for r in mr); tout = sum(r["out"] for r in mr)
            overall = 2.0 * ap_m * (tin + tout) / (sum(e - s for s, e in stages) or 1) / peak * 100
            print(f"  {short(m):16s} reqs={len(mr):6d} in_tok={tin:8d} out_tok={tout:7d} "
                  f"FLOPs/token={2*ap_m/1e9:5.1f}G  per-stage MFU%="
                  f"{[round(float(v),1) for v in mfu]}  load-avg={overall:.1f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(xr)
        ax.set_xlabel("offered load (RPS)")
        ax.set_ylabel("Model FLOPs Utilization (% of peak)")
        ax.set_title(f"{label}: MFU per backend vs load")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Model FLOPs Utilization — achieved / {args.peak_tflops:.0f} TFLOP/s peak "
                 f"(2·active_params·tokens, averaged per load stage)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
