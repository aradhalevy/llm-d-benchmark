#!/usr/bin/env python3
"""Show which backend model each picker chose over time.

For each run (LABEL=collected-logs-N) it plots, over wall-clock time:
  * a faint dot per request at the chosen backend (two rows: top = one model,
    bottom = the other), so the raw decision stream is visible, and
  * a rolling share line: fraction of recent requests routed to each backend
    (sliding window), so the routing tendency over time is clear.

Random routing hovers around an even split regardless of load; the smart
picker (inflight-requests-scorer) skews toward the faster backend and shifts
with load. Runs are drawn side by side.

Usage:
    plot_model_choice_over_time.py random=collected-logs-39 smart=collected-logs-40 -o model_choice.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def served_model(rec: dict) -> str:
    resp = rec.get("response")
    s = resp if isinstance(resp, str) else json.dumps(resp)
    m = re.search(r'"model":"([^"]+)"', s or "")
    return m.group(1) if m else "?"


def load_run(path: Path) -> list[dict]:
    files = list((path / "benchmark-results" / "results").glob("*/per_request_lifecycle_metrics.json"))
    if not files:
        return []
    best = max(files, key=lambda f: len(json.load(open(f))))
    recs = []
    for r in json.load(open(best)):
        st = r.get("start_time")
        if isinstance(st, (int, float)):
            recs.append({"st": st, "model": served_model(r)})
    recs.sort(key=lambda r: r["st"])
    return recs


def rolling_share(times, flags, grid, window):
    """Fraction of requests with flag==1 within +/- window/2 of each grid point."""
    times = np.asarray(times)
    flags = np.asarray(flags, dtype=float)
    out = []
    for t in grid:
        mask = (times >= t - window / 2) & (times <= t + window / 2)
        out.append(flags[mask].mean() if mask.any() else np.nan)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="model_choice_over_time.png")
    parser.add_argument("--window", type=float, default=6.0,
                        help="Sliding window (s) for the routing-share line.")
    args = parser.parse_args()

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        recs = load_run(Path(path))
        if not recs:
            print(f"no per-request data in {path}", file=sys.stderr)
            continue
        runs.append((label, recs))
    if not runs:
        print("no usable inputs", file=sys.stderr)
        sys.exit(1)

    models = sorted({r["model"] for _, recs in runs for r in recs})
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    primary = models[0]  # share line tracks fraction routed to this backend
    # Place the primary model (tracked by the % line) on the TOP row so the
    # high share line sits next to its own label; others below.
    row_order = [m for m in models if m != primary] + [primary]

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.0), squeeze=False)
    axes = axes[0]

    for ax, (label, recs) in zip(axes, runs):
        t0 = min(r["st"] for r in recs)
        times = [r["st"] - t0 for r in recs]
        tmax = max(times)

        # raw decision stream: dot per request, one row per model
        row = {m: i for i, m in enumerate(row_order)}
        for m in row_order:
            xs = [t for t, r in zip(times, recs) if r["model"] == m]
            ys = [row[m] + np.random.uniform(-0.12, 0.12) for _ in xs]
            ax.scatter(xs, ys, s=10, alpha=0.45, color=colors[m], edgecolors="none", zorder=2)
        ax.set_yticks(list(row.values()))
        ax.set_yticklabels([short(m) for m in row_order])
        ax.set_ylim(-0.6, len(row_order) - 0.4)
        ax.set_xlim(0, tmax)
        ax.set_xlabel("time since first request (s)")
        ax.set_title(f"{label}: backend chosen over time")
        ax.grid(True, axis="x", alpha=0.25)

        # rolling share of traffic to the primary backend (right axis, 0-100%)
        ax2 = ax.twinx()
        grid = np.linspace(0, tmax, 240)
        flags = [1 if r["model"] == primary else 0 for r in recs]
        share = rolling_share(times, flags, grid, args.window) * 100
        ax2.plot(grid, share, color="#222222", linewidth=1.8, zorder=4,
                 label=f"% to {short(primary)} ({args.window:.0f}s window)")
        ax2.axhline(50, color="#888888", linestyle="--", linewidth=1, alpha=0.7, zorder=3)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel(f"% routed to {short(primary)}")
        overall = 100.0 * sum(flags) / len(flags)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.9,
                   title=f"overall {overall:.0f}% {short(primary)}")

    fig.suptitle("Picker decisions over time: random vs smart", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
