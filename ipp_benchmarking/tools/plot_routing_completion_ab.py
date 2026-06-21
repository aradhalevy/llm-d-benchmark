#!/usr/bin/env python3
"""Routing share vs. concurrency from the SUMMARIZER's own per_request (completion-based).

Used when a concurrent planner harness contaminates the IPP decision log (its pre/post-summarizer
edges break the per-stage clustering). The summarizer per_request_slim contains ONLY summarizer
requests, so it clusters cleanly into the 5 stages. Per stage we show the share of *successful
completions* served by each backend (8B vs 32B) — this is "where summaries actually ran", which
undercounts the picker's 32B intent (the 32B fails more under overflow) but is planner-free.

    plot_routing_completion_ab.py "smart"=.../arm_a_smart "static"=.../arm_b_static \
        --concurrencies 100,200,300,400,500 -o routing.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SMALL, BIG = "small", "big"
NAME = {SMALL: "Qwen3-8B (small)", BIG: "Qwen3-32B (big)"}
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}


def stages_from_slim(run: Path, n):
    rows = json.load(open(run / "per_request_slim.json"))
    rows.sort(key=lambda r: r["t"])
    stages, cur = [], [rows[0]]
    for p, r in zip(rows, rows[1:]):
        if r["t"] - p["t"] > 8:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= 20]
    # merge clusters split by 30s-timeout lulls back down to n concurrency levels (smallest gap first)
    while len(stages) > n:
        gaps = [stages[i + 1][0]["t"] - stages[i][-1]["t"] for i in range(len(stages) - 1)]
        i = gaps.index(min(gaps))
        stages[i] = stages[i] + stages[i + 1]
        del stages[i + 1]
    return stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default="routing_completion_ab.png")
    ap.add_argument("--concurrencies", default="100,200,300,400,500")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    runs = [(e.split("=", 1)[0], stages_from_slim(Path(e.split("=", 1)[1]), len(conc))) for e in args.inputs if "=" in e]

    nr = len(runs)
    fig, axes = plt.subplots(1, nr, figsize=(7.2 * nr, 5.0), squeeze=False, sharey=True)
    axes = axes[0]
    for ax, (label, stages) in zip(axes, runs):
        xs = conc[:len(stages)]
        pct = {SMALL: [], BIG: []}
        for s in stages:
            so = sum(1 for r in s if r["m"] == SMALL and not r["fail"])
            bo = sum(1 for r in s if r["m"] == BIG and not r["fail"])
            ok = so + bo or 1
            pct[SMALL].append(100 * so / ok); pct[BIG].append(100 * bo / ok)
        for m in (SMALL, BIG):
            ax.plot(xs, pct[m], "-o", color=COLOR[m], lw=2.4, ms=5)
            ax.annotate(f"{pct[m][-1]:.0f}%", (xs[-1], pct[m][-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8.5, color=COLOR[m])
        ax.axhline(50, color="#999", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(0, 100); ax.set_xticks(conc)
        ax.set_xlabel("concurrent clients"); ax.set_ylabel("% of completed summaries on backend")
        ax.set_title(label, fontsize=11); ax.grid(True, alpha=0.2)
        ax.legend(handles=[Line2D([0], [0], color=COLOR[m], lw=2.4, marker="o", label=NAME[m])
                           for m in (SMALL, BIG)], loc="center left", fontsize=9, framealpha=0.92)
    fig.suptitle("Routing share vs. concurrency — % of completed summaries per backend (summarizer per-request)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
