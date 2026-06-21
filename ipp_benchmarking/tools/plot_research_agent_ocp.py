#!/usr/bin/env python3
"""Deep-research-agent A/B plots: latency scorer vs random model routing (OCP).

Per run (one column) the summarizer crawl-phase metrics vs concurrent clients:
  * e2e latency p50/p90
  * TTFT p50/p90
  * throughput (requests/s) + failures
and a companion routing-share figure (% of each stage routed to 8B vs 32B).

Aggregates come from inference-perf's own per-stage summaries
(stage_*_lifecycle_metrics.json: request_latency, time_to_first_token, throughput,
success/fail counts). Routing share comes from per_request_lifecycle_metrics.json by
the served model echoed in each response (ground truth); the stage summaries are not
split by model so routing is recovered separately and zipped by stage order.

    plot_research_agent_ocp.py "PR latency scorer"=.../scorer "random"=.../random \
        -o ocp_research_agent.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BIG, SMALL = "big", "small"          # 32B (planner/overflow), 8B (summarizer)
MODEL_NAME = {SMALL: "Qwen3-8B (small)", BIG: "Qwen3-32B (big)"}
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}   # green / purple
P50_C, P90_C = "#1f77b4", "#d62728"          # e2e/ttft percentile colors


def served_model(rec: dict):
    """Which backend served this request, from the response/info blob (ground truth)."""
    blob = json.dumps(rec.get("response")) + json.dumps(rec.get("info") or {})
    # qwen family first (32B before 8B so the substring check is unambiguous), then
    # the kind opt family so this tool can be smoke-tested on the kind sweeps.
    if "32B" in blob or "350m" in blob:
        return BIG
    if "8B" in blob or "125m" in blob:
        return SMALL
    return None


def _stage_idx(path: str) -> int:
    m = re.search(r"stage_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _pick(d: dict, *keys, default=float("nan")):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d if isinstance(d, (int, float)) else default


def load_stages(run: Path):
    """Per-stage aggregate metrics from inference-perf stage summaries, in stage order."""
    files = sorted(glob.glob(str(run / "stage_*_lifecycle_metrics.json"))
                   or glob.glob(str(run / "**" / "stage_*_lifecycle_metrics.json"), recursive=True),
                   key=_stage_idx)
    out = []
    for f in files:
        d = json.load(open(f))
        succ, fail = d.get("successes", {}), d.get("failures", {})
        out.append({
            "conc":    _pick(d, "load_summary", "concurrency"),
            "e2e_p50": _pick(succ, "latency", "request_latency", "median"),
            "e2e_p90": _pick(succ, "latency", "request_latency", "p90"),
            "ttft_p50": _pick(succ, "latency", "time_to_first_token", "median"),
            "ttft_p90": _pick(succ, "latency", "time_to_first_token", "p90"),
            "rps":     _pick(succ, "throughput", "requests_per_sec"),
            "succ":    int(_pick(succ, "count", default=0)),
            "fail":    int(_pick(fail, "count", default=0)),
        })
    return out


def load_routing(run: Path):
    """Overall (small%, big%, n) from the IPP decision log.

    per_request routing is unusable here: inference-perf truncates the streamed
    per_request file mid-write (json.load -> "Unterminated string"). The IPP
    decision log (one `Qwen3-8B`/`Qwen3-32B` per pick) is the reliable ground truth.
    """
    logs = glob.glob(str(run / "ipp-decisions.log")) + glob.glob(str(run / "**" / "ipp-decisions.log"), recursive=True)
    small = big = 0
    for lg in logs[:1]:
        with open(lg, errors="ignore") as fh:
            for line in fh:
                for m in re.findall(r"Qwen3-(8B|32B)", line):
                    if m == "8B":
                        small += 1
                    else:
                        big += 1
    n = small + big
    if not n:
        return None
    return (100.0 * small / n, 100.0 * big / n, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path entries")
    ap.add_argument("-o", "--output", default="ocp_research_agent.png")
    ap.add_argument("--routing-output", default=None, help="routing-share png (default: <output>_routing.png)")
    args = ap.parse_args()

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        runs.append((label, load_stages(Path(path)), load_routing(Path(path))))
    if not runs:
        sys.exit("no runs")

    nr = len(runs)
    # ---- main figure: 3 rows (e2e, ttft, throughput+fail) x one col per arm
    fig, axes = plt.subplots(3, nr, figsize=(6.8 * nr, 9.6), squeeze=False, sharex="col")
    for col, (label, stages, _) in enumerate(runs):
        xs = [s["conc"] for s in stages]
        axE, axT, axR = axes[0][col], axes[1][col], axes[2][col]

        axE.plot(xs, [s["e2e_p50"] for s in stages], "-o", color=P50_C, lw=2.2, label="p50")
        axE.plot(xs, [s["e2e_p90"] for s in stages], "--o", color=P90_C, lw=2.2, label="p90")
        axE.set_ylabel("e2e latency (s)"); axE.set_title(label, fontsize=11)
        axE.grid(True, alpha=0.2); axE.legend(fontsize=8.5)

        axT.plot(xs, [s["ttft_p50"] for s in stages], "-o", color=P50_C, lw=2.2, label="p50")
        axT.plot(xs, [s["ttft_p90"] for s in stages], "--o", color=P90_C, lw=2.2, label="p90")
        axT.set_ylabel("TTFT (s)"); axT.grid(True, alpha=0.2); axT.legend(fontsize=8.5)

        w = (xs[1] - xs[0]) * 0.6 if len(xs) > 1 else 6
        axR.bar(xs, [s["succ"] for s in stages], width=w, color="#7f7f7f", label="success")
        axR.bar(xs, [s["fail"] for s in stages], width=w, bottom=[s["succ"] for s in stages],
                color="#d62728", label="failed")
        axRt = axR.twinx()
        axRt.plot(xs, [s["rps"] for s in stages], "-D", color="#2ca02c", lw=2.0, ms=5, label="req/s")
        axRt.set_ylabel("throughput (req/s)", color="#2ca02c")
        axR.set_ylabel("requests"); axR.set_xlabel("concurrent clients")
        axR.grid(True, axis="y", alpha=0.2); axR.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Deep-research agent — summarizer crawl-phase: latency scorer vs random", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")

    # ---- routing-share figure: overall % to each backend per arm (grouped bars)
    rout = args.routing_output or args.output.replace(".png", "") + "_routing.png"
    fig2, ax = plt.subplots(figsize=(2.2 * nr + 3, 4.8))
    labels = [label for label, _, _ in runs]
    x = np.arange(nr); w = 0.38
    small_pct = [r[0] if r else 0 for _, _, r in runs]
    big_pct = [r[1] if r else 0 for _, _, r in runs]
    ax.bar(x - w / 2, small_pct, w, color=COLOR[SMALL], label=MODEL_NAME[SMALL])
    ax.bar(x + w / 2, big_pct, w, color=COLOR[BIG], label=MODEL_NAME[BIG])
    for i, (_, _, r) in enumerate(runs):
        if not r:
            continue
        ax.text(i - w / 2, small_pct[i] + 1, f"{small_pct[i]:.0f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, big_pct[i] + 1, f"{big_pct[i]:.0f}%", ha="center", fontsize=9)
    ax.axhline(50, color="#999", ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100); ax.set_ylabel("% of summarizer picks")
    ax.set_title("Routing share — where summarizer requests went (overall)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.2); ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(rout, dpi=130)
    print(f"wrote {rout}")


if __name__ == "__main__":
    main()
