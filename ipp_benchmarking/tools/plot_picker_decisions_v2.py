#!/usr/bin/env python3
"""Picker decisions over time, v2: latency dots + routing-share area.

For each run (LABEL=collected-logs-N), one panel:
  * dots = per-request end-to-end latency over time (left axis), colored by the
    backend model that served the request -> shows the p50/p95 spread directly.
  * filled area (right axis, 0-100%) = the routing share over time: the band
    from 0 to %-routed-to-MoE is the MoE's colour, the band above (to 100%) is
    the dense model's colour. So the coloured regions show the proportion each
    model received at each moment.

Random vs smart side by side, shared latency scale.

Usage:
    plot_picker_decisions_v2.py random=collected-logs-43 smart=collected-logs-44 -o picker_v2.png
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


def short(m: str) -> str:
    return m.rsplit("/", 1)[-1]


def served(rec: dict) -> str:
    s = rec.get("response") if isinstance(rec.get("response"), str) else json.dumps(rec.get("response"))
    m = re.search(r'"model":"([^"]+)"', s or "")
    if not m:
        return "?"
    return MOE if "30B" in m.group(1) else (DENSE if "32B" in m.group(1) else short(m.group(1)))


def load(run: Path):
    files = glob.glob(str(run / "benchmark-results" / "results" / "*" / "per_request_lifecycle_metrics.json"))
    if not files:
        return []
    best = max(files, key=lambda f: len(json.load(open(f))))
    recs = []
    for r in json.load(open(best)):
        st, et = r.get("start_time"), r.get("end_time")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            recs.append({"t": st, "lat": et - st, "m": served(r)})
    recs.sort(key=lambda x: x["t"])
    return recs


def detect_stages(times, tmax, bin_s=5.0, thresh=40.0):
    """Return [(start,end)] of load stages from arrival density (gaps=cooldowns)."""
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


def share_to_moe(times, is_moe, grid, window):
    times = np.asarray(times); is_moe = np.asarray(is_moe, float)
    out = []
    for t in grid:
        mask = (times >= t - window / 2) & (times <= t + window / 2)
        out.append(is_moe[mask].mean() if mask.any() else np.nan)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="picker_decisions_v2.png")
    ap.add_argument("--window", type=float, default=4.0, help="sliding window (s) for routing share")
    ap.add_argument("--rates", default="200,250,300", help="comma-separated load-stage rates in order")
    args = ap.parse_args()
    rates = [r.strip() for r in args.rates.split(",") if r.strip()]

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

    lat_max = max(r["lat"] for _, recs in runs for r in recs) * 1.05
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (label, recs) in zip(axes, runs):
        t0 = min(r["t"] for r in recs)
        times = [r["t"] - t0 for r in recs]
        tmax = max(times)
        # routing-share area (right axis, %), drawn as background bands
        ax2 = ax.twinx()
        grid = np.linspace(0, tmax, 250)
        frac = share_to_moe(times, [r["m"] == MOE for r in recs], grid, args.window) * 100
        frac = np.nan_to_num(frac, nan=np.nanmean(frac))
        ax2.fill_between(grid, 0, frac, color=COLORS[MOE], alpha=0.16, zorder=0)
        ax2.fill_between(grid, frac, 100, color=COLORS[DENSE], alpha=0.16, zorder=0)
        ax2.plot(grid, frac, color=COLORS[MOE], linewidth=1.0, alpha=0.5, zorder=1)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel(f"% routed  (lower band = {MOE}, upper = {DENSE})", fontsize=9)
        overall = 100.0 * sum(r["m"] == MOE for r in recs) / len(recs)
        ax2.text(0.99, 0.02, f"{overall:.0f}% {MOE}", transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=9, color="#333")

        # latency dots (left axis), colored by served model
        for m in (MOE, DENSE):
            xs = [t for t, r in zip(times, recs) if r["m"] == m]
            ys = [r["lat"] for r in recs if r["m"] == m]
            ax.scatter(xs, ys, s=9, alpha=0.5, color=COLORS[m], edgecolors="none",
                       zorder=3, label=short(m))
        ax.set_ylim(0, lat_max)
        ax.set_xlim(0, tmax)
        ax.set_xlabel("time since load start (s)")
        ax.set_ylabel("request latency (s)")
        ax.set_title(f"{label}: latency dots + routing share")
        ax.grid(True, alpha=0.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # annotate load stages (200/250/300 RPS) and cooldowns on the x-axis
        stages = detect_stages(times, tmax)
        for k, (s, e) in enumerate(stages[:len(rates)]):
            ax.axvline(s, color="#444", ls=":", lw=0.8, zorder=2)
            ax.axvline(e, color="#444", ls=":", lw=0.8, zorder=2)
            ax.text((s + e) / 2, lat_max * 0.035, f"{rates[k]} RPS", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#111", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        for k in range(len(stages[:len(rates)]) - 1):
            gs, ge = stages[k][1], stages[k + 1][0]
            if ge - gs > 6:
                ax.text((gs + ge) / 2, lat_max * 0.035, "cooldown", ha="center", va="bottom",
                        fontsize=8, style="italic", color="#444", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    fig.suptitle("Picker decisions over time — latency by backend (dots) over the routing share (bands)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
