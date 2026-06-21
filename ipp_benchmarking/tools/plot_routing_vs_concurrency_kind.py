#!/usr/bin/env python3
"""Routing share vs. concurrency, per model — companion to the latency plot.

Same 2-panel layout (one per run), but instead of latency percentiles it shows the
fraction of requests the picker routed to each backend at each concurrency level.
No per-request dots — just the routing between models.
  * x = concurrent clients
  * y = % of that stage's requests routed to each backend (green fast / red slow)

Routing is taken from per_request: successes by the served model echoed in the
response, failures (no served model) attributed to the slow backend (exact here —
the fast sim never times out). Stages are recovered from the idle gaps load.interval
leaves between concurrency stages.

    plot_routing_vs_concurrency_kind.py "random"=.../random "PR#46"=.../inflight \
        --concurrencies 30,40,50,60,70,80,90,100,110,120 -o routing.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FAST, SLOW = "fast", "slow"
MODEL_NAME = {FAST: "opt-350m (fast)", SLOW: "opt-125m (slow)"}
COLOR = {FAST: "#2ca02c", SLOW: "#9467bd"}   # green / purple


def served_model(rec: dict):
    blob = json.dumps(rec.get("response")) + json.dumps(rec.get("info") or {})
    if "350m" in blob:
        return FAST
    if "125m" in blob:
        return SLOW
    return None


def load_rows(run: Path):
    # accept a trimmed arm dir OR a full collect_logs.sh bundle: find the file at any depth
    cands = glob.glob(str(run / "**" / "per_request_lifecycle_metrics.json"), recursive=True)
    if not cands:
        sys.exit(f"no per_request_lifecycle_metrics.json under {run}")
    pf = max(cands, key=lambda f: len(json.load(open(f))))
    rows = []
    for r in json.load(open(pf)):
        st = r.get("start_time")
        if isinstance(st, (int, float)):
            served = served_model(r)
            rows.append({"t": st, "m": served if served else SLOW})
    rows.sort(key=lambda x: x["t"])
    return rows


def cluster_stages(rows, gap, min_size, trim=0.1):
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["t"] - prev["t"] > gap:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    # drop the first/last `trim` of each (time-sorted) stage to mitigate ramp outliers
    out = []
    for s in stages:
        if len(s) < min_size:
            continue
        k = int(len(s) * trim)
        out.append(s[k:len(s) - k] if k else s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path entries")
    ap.add_argument("-o", "--output", default="routing_vs_concurrency_kind.png")
    ap.add_argument("--concurrencies", default="30,40,50,60,70,80,90,100,110,120")
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--min-size", type=int, default=20)
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        stages = cluster_stages(load_rows(Path(path)), args.gap, args.min_size)
        if len(stages) != len(conc):
            print(f"warn {path}: {len(stages)} stages vs {len(conc)} concurrencies", file=sys.stderr)
        runs.append((label, stages))
    if not runs:
        sys.exit("no runs")

    nr = len(runs)
    fig, axes = plt.subplots(1, nr, figsize=(7.2 * nr, 5.0), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, (label, stages) in zip(axes, runs):
        xs = conc[:len(stages)]
        pct = {FAST: [], SLOW: []}
        for st in stages[:len(conc)]:
            n = len(st)
            nf = sum(1 for r in st if r["m"] == FAST)
            pct[FAST].append(100.0 * nf / n)
            pct[SLOW].append(100.0 * (n - nf) / n)
        for m in (FAST, SLOW):
            ax.plot(xs, pct[m], "-o", color=COLOR[m], lw=2.4, ms=5, zorder=4)
            ax.annotate(f"{pct[m][-1]:.0f}%", (xs[-1], pct[m][-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8.5, color=COLOR[m])
        ax.axhline(50, color="#999", ls=":", lw=0.8, alpha=0.7, zorder=1)
        ax.set_ylim(0, 100)
        ax.set_xticks(conc)
        ax.set_xlabel("concurrent clients")
        ax.set_ylabel("% of requests routed to backend")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(handles=[Line2D([0], [0], color=COLOR[m], lw=2.4, marker="o", label=MODEL_NAME[m])
                           for m in (FAST, SLOW)], loc="center left", fontsize=9, framealpha=0.92)

    fig.suptitle("Routing share vs. concurrency — % of each stage routed to each backend", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
