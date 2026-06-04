#!/usr/bin/env python3
"""Plot p95 latency vs rate (left) and failure counts vs rate (right)
for multiple collected-logs runs, side by side.

Usage:
    plot_latency_and_failures.py LABEL1=collected-logs-N1 LABEL2=collected-logs-N2 ... -o out.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def collect_stage_stats(logs_dir: Path) -> list[dict]:
    """Aggregate per-stage stats across all treatments in a logs dir.

    Returns list of dicts with: rate, total, failures, lat_p50, lat_p95.
    Latency stats are averaged (count-weighted) across treatments.

    NOTE: latency stats are computed only over successful requests
    (`stage_*_lifecycle_metrics.json` -> successes.latency.request_latency).
    Timed-out / failed requests are NOT included. This means a scenario
    that fails many slow requests can show a lower successful-p95 than a
    scenario that lets them complete; cross-reference the failure counts.
    """
    by_stage: dict[int, dict] = {}
    for treatment_dir in sorted((logs_dir / "benchmark-results" / "results").iterdir()):
        for stage_file in sorted(treatment_dir.glob("stage_*_lifecycle_metrics.json")):
            idx = int(stage_file.stem.split("_")[1])
            meta = json.load(open(stage_file))
            ls = meta["load_summary"]
            su = meta["successes"]
            fa = meta["failures"]
            slot = by_stage.setdefault(
                idx,
                {"rate": ls["requested_rate"], "total": 0, "failures": 0,
                 "lat_p50_num": 0.0, "lat_p95_num": 0.0, "lat_w": 0},
            )
            slot["total"] += ls["count"]
            slot["failures"] += fa["count"]
            if su["count"]:
                lat = su["latency"]["request_latency"]
                p50 = lat.get("median") or lat.get("p50")
                p95 = lat.get("p95")
                if p50 is not None:
                    slot["lat_p50_num"] += p50 * su["count"]
                if p95 is not None:
                    slot["lat_p95_num"] += p95 * su["count"]
                slot["lat_w"] += su["count"]
    out = []
    for idx in sorted(by_stage):
        s = by_stage[idx]
        p50 = s["lat_p50_num"] / s["lat_w"] if s["lat_w"] else None
        p95 = s["lat_p95_num"] / s["lat_w"] if s["lat_w"] else None
        out.append({
            "rate": float(s["rate"]),
            "total": int(s["total"]),
            "failures": int(s["failures"]),
            "lat_p50": p50,
            "lat_p95": p95,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="latency_and_failures.png")
    parser.add_argument("--percentile", choices=["p50", "p95"], default="p95",
                        help="Which request-latency percentile to plot on the left panel.")
    args = parser.parse_args()

    scenarios = []
    for entry in args.inputs:
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        stats = collect_stage_stats(Path(path))
        if not stats:
            print(f"no stages found in {path}", file=sys.stderr)
            continue
        scenarios.append((label, stats))

    if not scenarios:
        print("no usable inputs", file=sys.stderr)
        sys.exit(1)

    fig, (ax_lat, ax_fail) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"random": "#1f77b4", "smart": "#ff7f0e"}
    fallback = ["#2ca02c", "#d62728", "#9467bd"]

    # Left: latency (chosen percentile) vs rate
    lat_key = "lat_" + args.percentile
    pct = args.percentile
    for i, (label, stats) in enumerate(scenarios):
        c = colors.get(label, fallback[i % len(fallback)])
        xs = [s["rate"] for s in stats]
        ys = [s[lat_key] for s in stats]
        ax_lat.plot(xs, ys, marker="o", linewidth=2, label=label, color=c)
    ax_lat.set_xlabel("rate per second (RPS)")
    ax_lat.set_ylabel(f"{pct} request latency (s, successful requests only)")
    ax_lat.set_title(f"{pct} latency vs load")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend()

    # Right: failure counts per rate (grouped bars)
    rates = sorted({s["rate"] for _, stats in scenarios for s in stats})
    x = np.arange(len(rates))
    bar_w = 0.8 / max(1, len(scenarios))
    for i, (label, stats) in enumerate(scenarios):
        c = colors.get(label, fallback[i % len(fallback)])
        by_rate = {s["rate"]: s for s in stats}
        ys = [by_rate.get(r, {}).get("failures", 0) for r in rates]
        offsets = x + (i - (len(scenarios) - 1) / 2) * bar_w
        bars = ax_fail.bar(offsets, ys, bar_w, label=label, color=c, alpha=0.85)
        for bar, val in zip(bars, ys):
            if val > 0:
                ax_fail.annotate(str(val), (bar.get_x() + bar.get_width() / 2, val),
                                 textcoords="offset points", xytext=(0, 3),
                                 ha="center", fontsize=8)
    ax_fail.set_xticks(x)
    ax_fail.set_xticklabels([f"{int(r)}" for r in rates])
    ax_fail.set_xlabel("rate per second (RPS)")
    ax_fail.set_ylabel("Failed requests (count, HTTP 504)")
    ax_fail.set_title("Failures per rate")
    ax_fail.grid(True, axis="y", alpha=0.3)
    ax_fail.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
