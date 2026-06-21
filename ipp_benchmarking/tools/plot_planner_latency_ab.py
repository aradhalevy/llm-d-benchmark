#!/usr/bin/env python3
"""Planner e2e latency + success/fail vs. concurrency, Arm A (smart) vs Arm B (static).

Reads the planner's own inference-perf STAGE files (per-stage percentiles + succ/fail), since
the planner's per_request is too sparse to time-cluster. Same two-panel style as the summarizer
latency plot:
  * TOP — successful-request e2e p50 (solid) / p95 (dashed). Timed-out requests (504, ~31s) are
    drawn as separate markers in a TIMEOUT BAND above the 30s line, one spaced level per stage so
    they don't overlap, NOT connected by a line, each annotated with its concurrency + fail count.
  * BOTTOM — request volume per concurrency: green = success, red = timed out.

    plot_planner_latency_ab.py "smart"=.../arm_a_smart/planner "static"=.../arm_b_static/planner \
        --concurrencies 1,2,3,4,5 -o planner_latency_ab.png
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
from matplotlib.patches import Patch

OK_C, TMO_C = "#2ca02c", "#d62728"
E2E_C = "#9467bd"            # 32B (planner is pinned to the big model)
CEIL = 30.0
GAP = 1.5
SPACING = 1.4               # vertical spacing of per-stage timeout markers in the band


def load_stages(run: Path):
    out = []
    for f in sorted(glob.glob(str(run / "stage_*_lifecycle_metrics.json")),
                    key=lambda x: int(re.search(r"stage_(\d+)", x).group(1))):
        d = json.load(open(f))
        s, fa = d["successes"], d["failures"]
        rl = s.get("latency", {}).get("request_latency", {})
        out.append({
            "conc": round(d["load_summary"].get("concurrency", 0)),
            "succ": s["count"], "fail": fa["count"],
            "p50": rl.get("median", float("nan")), "p95": rl.get("p95", float("nan")),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path")
    ap.add_argument("-o", "--output", default="planner_latency_ab.png")
    ap.add_argument("--concurrencies", default="1,2,3,4,5")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    runs = [(e.split("=", 1)[0], load_stages(Path(e.split("=", 1)[1]))) for e in args.inputs if "=" in e]
    if not runs:
        sys.exit("no runs")

    n_band = len(conc)
    y_top = CEIL + GAP + (n_band - 1) * SPACING + 1.5
    bar_max = max((st["succ"] + st["fail"]) for _, stages in runs for st in stages) * 1.18
    nr = len(runs)
    fig, axes = plt.subplots(2, nr, figsize=(7.2 * nr, 7.4), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2.5, 1.0]})

    for col, (label, stages) in enumerate(runs):
        axL, axB = axes[0][col], axes[1][col]
        xs = [s["conc"] for s in stages]

        axL.axhspan(CEIL + GAP, y_top, color="#eee", zorder=0)
        axL.axhline(CEIL, color="#333", lw=1.4, zorder=2)
        axL.text(xs[-1], CEIL, "30s timeout ", color="#333", fontsize=8, va="bottom", ha="right", zorder=3)

        # success latency curves (real region)
        axL.plot(xs, [s["p50"] for s in stages], "-o", color=E2E_C, lw=2.3, ms=5, label="e2e p50", zorder=5)
        axL.plot(xs, [s["p95"] for s in stages], "--o", color=E2E_C, lw=2.3, ms=5, label="e2e p95", zorder=5)

        # timed-out requests: a separate spaced marker per stage in the band (NOT connected)
        for i, s in enumerate(stages):
            if s["fail"] > 0:
                yb = CEIL + GAP + i * SPACING
                axL.plot([s["conc"]], [yb], marker="X", color=TMO_C, ms=11, zorder=6, linestyle="None")
                axL.annotate(f"{s['fail']} timed out", (s["conc"], yb), textcoords="offset points",
                             xytext=(8, 0), va="center", fontsize=8, color=TMO_C)

        axL.set_ylim(0, y_top)
        axL.set_yticks([0, 5, 10, 15, 20, 25, 30])
        axL.set_ylabel("planner e2e latency (s)")
        axL.set_title(label, fontsize=11)
        axL.grid(True, axis="y", alpha=0.2)
        h = [Line2D([0], [0], color=E2E_C, lw=2.3, marker="o", label="e2e p50"),
             Line2D([0], [0], color=E2E_C, lw=2.3, ls="--", marker="o", label="e2e p95"),
             Line2D([0], [0], color=TMO_C, marker="X", ls="None", ms=10, label="timed out (504)")]
        axL.legend(handles=h, loc="upper left", fontsize=8.5, framealpha=0.92)

        # volume bars
        xa = np.array(xs, float)
        axB.bar(xa, [s["succ"] for s in stages], width=0.55, color=OK_C, label="success")
        axB.bar(xa, [s["fail"] for s in stages], width=0.55, bottom=[s["succ"] for s in stages],
                color=TMO_C, label="timed out")
        axB.set_ylim(0, bar_max)
        axB.set_xticks(conc)
        axB.set_xlabel("planner concurrent clients (1/100 of summarizer)")
        axB.set_ylabel("requests")
        axB.grid(True, axis="y", alpha=0.2)
        if col == 0:
            axB.legend(handles=[Patch(color=OK_C, label="success"), Patch(color=TMO_C, label="timed out")],
                       loc="upper left", fontsize=8.5)

    fig.suptitle("Planner (pinned Qwen3-32B) e2e latency + timeouts vs. concurrency — smart vs. no routing",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
