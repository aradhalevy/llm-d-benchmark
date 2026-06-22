#!/usr/bin/env python3
"""Routing share vs. concurrency, per model — OCP Qwen3-8B/32B research-agent.

Adapted from plot_routing_vs_concurrency_kind.py, but routing is taken from the IPP
DECISION log (the picker's actual choice per request), not per_request. That matters
on OCP: the 32B offload target gets swamped at high load and fails, so per_request
*completions* understate where the picker actually sent traffic — only the decision
log shows true picker behaviour (the "does 32B share rise with load" question).

  * x = concurrent clients
  * y = % of that stage's picks routed to each backend (green 8B / purple 32B)

Decisions are deduped by x-request-id (rolling-snapshot capture overlaps). With
--planner N, N planner picks (all pinned to 32B) are subtracted pro-rata per stage so
the curve reflects SUMMARIZER routing only. Stages are recovered by clustering on idle
gaps, then merged down to the requested number of concurrency levels.

    plot_routing_vs_concurrency_ocp.py "random"=.../random "smart"=.../smart_planner \
        --concurrencies 100,200,300,400,500 --planner 0,159 -o routing.png
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

SMALL, BIG = "small", "big"
MODEL_NAME = {SMALL: "Qwen3-8B (small)", BIG: "Qwen3-32B (big)"}
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}


def load_decisions(run: Path):
    """(ts, model) per pick, deduped by x-request-id, sorted by ts."""
    logs = glob.glob(str(run / "ipp-decisions.log")) + glob.glob(str(run / "ipp-model-selected.log"))
    sel, ts = {}, {}
    for lg in logs[:1]:
        for line in open(lg, errors="ignore"):
            if '"msg":"Model selected"' not in line:
                continue
            rid = re.search(r'"x-request-id":"([^"]+)"', line)
            m = BIG if "Qwen3-32B" in line else (SMALL if "Qwen3-8B" in line else None)
            if not rid or not m:
                continue
            rid = rid.group(1)
            sel[rid] = m
            t = re.search(r'"ts":([0-9.]+)', line)
            ts[rid] = float(t.group(1)) if t else 0.0
    return sorted((ts[r], sel[r]) for r in sel)


def cluster_to_n(rows, n, gap=8.0, min_size=15):
    """Gap-cluster, drop tiny clusters, then merge adjacent (smallest inter-gap first)
    until exactly n stages remain (the 5 concurrency levels)."""
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r[0] - prev[0] > gap:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= min_size]
    # merge the two adjacent clusters separated by the smallest time gap until len==n
    while len(stages) > n:
        gaps = [stages[i + 1][0][0] - stages[i][-1][0] for i in range(len(stages) - 1)]
        i = int(np.argmin(gaps))
        stages[i] = stages[i] + stages[i + 1]
        del stages[i + 1]
    return stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path entries")
    ap.add_argument("-o", "--output", default="routing_vs_concurrency_ocp.png")
    ap.add_argument("--concurrencies", default="100,200,300,400,500")
    ap.add_argument("--planner", default="", help="comma list, one planner-count per input (all 32B), subtracted pro-rata")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    planners = [int(p) for p in args.planner.split(",")] if args.planner else []

    runs = []
    for idx, entry in enumerate(args.inputs):
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        rows = load_decisions(Path(path))
        if not rows:
            sys.exit(f"no decisions in {path}")
        stages = cluster_to_n(rows, len(conc))
        pl = planners[idx] if idx < len(planners) else 0
        total = sum(len(s) for s in stages)
        runs.append((label, stages, pl, total))

    nr = len(runs)
    fig, axes = plt.subplots(1, nr, figsize=(7.2 * nr, 5.0), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, (label, stages, pl, total) in zip(axes, runs):
        xs = conc[:len(stages)]
        pct = {SMALL: [], BIG: []}
        for st in stages[:len(conc)]:
            n = len(st)
            nb = sum(1 for _, m in st if m == BIG)
            # subtract this stage's pro-rata share of planner picks (all 32B) -> summarizer only
            pl_here = pl * n / total if total else 0
            nb_s = max(0.0, nb - pl_here)
            n_s = max(1e-9, n - pl_here)
            pct[BIG].append(100.0 * nb_s / n_s)
            pct[SMALL].append(100.0 * (n_s - nb_s) / n_s)
        for m in (SMALL, BIG):
            ax.plot(xs, pct[m], "-o", color=COLOR[m], lw=2.4, ms=5, zorder=4)
            ax.annotate(f"{pct[m][-1]:.0f}%", (xs[-1], pct[m][-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8.5, color=COLOR[m])
        ax.axhline(50, color="#999", ls=":", lw=0.8, alpha=0.7, zorder=1)
        ax.set_ylim(0, 100)
        ax.set_xticks(conc)
        ax.set_xlabel("concurrent clients")
        ax.set_ylabel("% of picks per backend")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(handles=[Line2D([0], [0], color=COLOR[m], lw=2.4, marker="o", label=MODEL_NAME[m])
                           for m in (SMALL, BIG)], loc="center left", fontsize=9, framealpha=0.92)

    fig.suptitle("Routing share vs. concurrency", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
