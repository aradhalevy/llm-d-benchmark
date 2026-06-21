#!/usr/bin/env python3
"""Planner success / timeout-fail, Arm A (smart) vs Arm B (static).

The planner is a fixed ~3-concurrency curl loop pinned to Qwen3-32B; we only logged HTTP
codes (200 = ok, 504 = gateway timeout). So this is totals per arm, not per-concurrency
(the planner has no ramp). Grouped bars: green = success, red = timeout, with the success
rate annotated.

    plot_planner_ab.py "smart"=.../arm_a_smart "static"=.../arm_b_static -o planner_ab.png
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OK_C, TMO_C = "#2ca02c", "#d62728"


def counts(run: Path):
    """Handle both formats: old '<code>' per line, new '<epoch> <code> <time_total>'."""
    f = run / "planner-http-codes.txt"
    ok = tmo = other = 0
    for line in open(f):
        toks = line.split()
        code = next((t for t in toks if t.isdigit() and len(t) == 3), None)
        if code == "200":
            ok += 1
        elif code == "504":
            tmo += 1
        elif code:
            other += 1
    return ok, tmo, other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path")
    ap.add_argument("-o", "--output", default="planner_ab.png")
    args = ap.parse_args()
    runs = [(e.split("=", 1)[0], Path(e.split("=", 1)[1])) for e in args.inputs if "=" in e]
    if not runs:
        sys.exit("no runs")

    labels = [lbl for lbl, _ in runs]
    data = [counts(p) for _, p in runs]
    x = np.arange(len(runs)); w = 0.34

    fig, ax = plt.subplots(figsize=(2.6 * len(runs) + 3, 5.2))
    oks = [d[0] for d in data]; tmos = [d[1] + d[2] for d in data]
    ax.bar(x - w / 2, oks, w, color=OK_C, label="success (200)")
    ax.bar(x + w / 2, tmos, w, color=TMO_C, label="timeout / fail (504)")
    for i, (ok, tmo) in enumerate(zip(oks, tmos)):
        tot = ok + tmo
        ax.text(i - w / 2, ok + 2, str(ok), ha="center", fontsize=9)
        ax.text(i + w / 2, tmo + 2, str(tmo), ha="center", fontsize=9)
        ax.text(i, max(ok, tmo) + max(oks + tmos) * 0.06,
                f"{100 * ok / tot:.0f}% ok" if tot else "n/a", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("planner requests (pinned to Qwen3-32B)")
    ax.set_ylim(0, max(oks + tmos) * 1.22)
    ax.set_title("Planner success vs. timeout — smart routing steals the 32B the planner needs",
                 fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(handles=[Patch(color=OK_C, label="success (200)"),
                       Patch(color=TMO_C, label="timeout / fail (504)")], fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
