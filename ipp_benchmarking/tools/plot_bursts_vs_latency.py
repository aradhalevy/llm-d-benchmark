#!/usr/bin/env python3
"""Show request bursts over time and how they affect latency.

For each run (LABEL=collected-logs-N), plots a time series:
  * scatter: one dot per request at its arrival time, y = end-to-end latency,
    colored by the backend model that actually served it (from the response).
  * overlaid step line (right axis): total in-flight requests over time
    (concurrency) -- the "burst" signal.

Runs are drawn side by side with a shared y-scale so random vs smart are
directly comparable: bursts that pile onto one backend under random routing
produce latency spikes that the load-aware picker avoids.

Usage:
    plot_bursts_vs_latency.py random=collected-logs-39 smart=collected-logs-40 -o bursts.png
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

MODEL_COLORS = {}
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def served_model(rec: dict) -> str:
    resp = rec.get("response")
    s = resp if isinstance(resp, str) else json.dumps(resp)
    m = re.search(r'"model":"([^"]+)"', s or "")
    return m.group(1) if m else "?"


def load_run(path: Path) -> list[dict]:
    """Return per-request records {start, end, lat, model} for the busiest treatment."""
    files = list((path / "benchmark-results" / "results").glob("*/per_request_lifecycle_metrics.json"))
    if not files:
        return []
    best = max(files, key=lambda f: len(json.load(open(f))))
    recs = []
    for r in json.load(open(best)):
        st, et = r.get("start_time"), r.get("end_time")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            recs.append({"st": st, "et": et, "lat": et - st, "model": served_model(r)})
    return recs


def concurrency_series(recs: list[dict], t0: float, t1: float, step: float = 0.25):
    """Sample total in-flight count (requests with st<=t<=et) over a time grid."""
    grid = np.arange(0.0, t1 - t0 + step, step)
    starts = np.array([r["st"] - t0 for r in recs])
    ends = np.array([r["et"] - t0 for r in recs])
    inflight = np.array([np.sum((starts <= t) & (ends >= t)) for t in grid])
    return grid, inflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="bursts_vs_latency.png")
    parser.add_argument("--step", type=float, default=0.25,
                        help="Concurrency sampling interval in seconds.")
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

    # Assign stable colors per backend model across all runs.
    models = sorted({r["model"] for _, recs in runs for r in recs})
    for i, m in enumerate(models):
        MODEL_COLORS.setdefault(m, PALETTE[i % len(PALETTE)])

    lat_max = max(r["lat"] for _, recs in runs for r in recs) * 1.08
    conc_max = 0
    series = {}
    for label, recs in runs:
        t0 = min(r["st"] for r in recs)
        t1 = max(r["et"] for r in recs)
        grid, inflight = concurrency_series(recs, t0, t1, args.step)
        series[label] = (t0, grid, inflight)
        conc_max = max(conc_max, inflight.max())
    conc_max *= 1.1

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (label, recs) in zip(axes, runs):
        t0, grid, inflight = series[label]
        # right axis: in-flight concurrency (the burst signal)
        ax2 = ax.twinx()
        ax2.fill_between(grid, inflight, step="mid", color="#999999", alpha=0.18, zorder=0)
        ax2.plot(grid, inflight, color="#666666", linewidth=1.0, alpha=0.7, zorder=1,
                 label="in-flight (concurrency)")
        ax2.set_ylim(0, conc_max)
        ax2.set_ylabel("in-flight requests (concurrency)", color="#555555")
        ax2.tick_params(axis="y", colors="#555555")

        # left axis: per-request latency scatter, colored by served backend
        for m in models:
            xs = [r["st"] - t0 for r in recs if r["model"] == m]
            ys = [r["lat"] for r in recs if r["model"] == m]
            ax.scatter(xs, ys, s=14, alpha=0.6, color=MODEL_COLORS[m],
                       label=short(m), zorder=3, edgecolors="none")
        ax.set_ylim(0, lat_max)
        ax.set_xlim(0, max(grid))
        ax.set_xlabel("time since first request (s)")
        ax.set_ylabel("request latency (s)")
        ax.set_title(f"{label}: bursts vs latency")
        ax.grid(True, alpha=0.25)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        # merged legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("Incoming request bursts (concurrency) and resulting latency", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
