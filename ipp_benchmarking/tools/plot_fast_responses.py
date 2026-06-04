#!/usr/bin/env python3
"""Plot count of successful requests completing under a latency threshold
vs requested rate, for multiple collected-logs scenarios.

Uses each stage's latency percentile distribution to estimate the fraction
of successes that finished within `--threshold` seconds, then multiplies by
the success count and aggregates across treatments. Higher curves = more
fast responses delivered as load grows.

Usage:
    plot_fast_responses.py LABEL1=collected-logs-N1 LABEL2=... \
        --threshold 13.0 -o out.png
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


# Percentile keys present in inference-perf stage latency dicts.
# Order matters: must be ascending.
PERCENTILES = [
    ("min",   0.0),
    ("p0.1",  0.001),
    ("p1",    0.01),
    ("p5",    0.05),
    ("p10",   0.10),
    ("p25",   0.25),
    ("median", 0.50),
    ("p75",   0.75),
    ("p90",   0.90),
    ("p95",   0.95),
    ("p99",   0.99),
    ("p99.9", 0.999),
    ("max",   1.0),
]


def fraction_below(lat_dist: dict, threshold: float) -> float:
    """Estimate fraction of successes whose request_latency <= threshold,
    by linearly interpolating between adjacent percentile points."""
    samples = []  # (latency, fraction)
    for key, frac in PERCENTILES:
        v = lat_dist.get(key)
        if v is None:
            continue
        samples.append((v, frac))
    if not samples:
        return 0.0
    samples.sort()
    # Below the minimum -> 0; above the maximum -> 1
    if threshold <= samples[0][0]:
        return 0.0
    if threshold >= samples[-1][0]:
        return 1.0
    for i in range(1, len(samples)):
        x0, f0 = samples[i - 1]
        x1, f1 = samples[i]
        if x0 <= threshold <= x1:
            if x1 == x0:
                return f1
            return f0 + (f1 - f0) * (threshold - x0) / (x1 - x0)
    return 1.0


def collect_per_stage(logs_dir: Path, threshold_s: float) -> list[tuple[float, int, int]]:
    """Aggregate per-stage stats across treatments.
    Returns (requested_rate, fast_count, issued) per stage.
    fast_count = sum across treatments of round(success_count * P(latency <= threshold))."""
    by_stage: dict[int, dict] = {}
    for treatment_dir in sorted((logs_dir / "benchmark-results" / "results").iterdir()):
        for stage_file in sorted(treatment_dir.glob("stage_*_lifecycle_metrics.json")):
            idx = int(stage_file.stem.split("_")[1])
            meta = json.load(open(stage_file))
            ls = meta["load_summary"]
            su = meta["successes"]
            slot = by_stage.setdefault(idx,
                {"rate": ls["requested_rate"], "fast": 0, "issued": 0})
            slot["issued"] += ls["count"]
            if su["count"]:
                frac = fraction_below(su["latency"]["request_latency"], threshold_s)
                slot["fast"] += int(round(su["count"] * frac))
    out = []
    for idx in sorted(by_stage):
        s = by_stage[idx]
        if s["rate"] < 1:  # skip cooldown stages
            continue
        out.append((float(s["rate"]), int(s["fast"]), int(s["issued"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="fast_responses.png")
    parser.add_argument("--threshold", type=float, default=13.0,
                        help="Latency threshold in seconds.")
    args = parser.parse_args()

    colors = {"random": "#1f77b4", "smart": "#ff7f0e"}
    fallback = ["#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(8, 5))
    for i, entry in enumerate(args.inputs):
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        stats = collect_per_stage(Path(path), args.threshold)
        if not stats:
            print(f"no stages in {path}", file=sys.stderr)
            continue
        xs = [s[0] for s in stats]
        ys = [s[1] for s in stats]
        c = colors.get(label, fallback[i % len(fallback)])
        plt.plot(xs, ys, marker="o", linewidth=2, label=label, color=c)
        for x, y, fast, total in zip(xs, ys, [s[1] for s in stats], [s[2] for s in stats]):
            plt.annotate(f"{fast}/{total}", (x, y), textcoords="offset points",
                         xytext=(5, 5), fontsize=8, alpha=0.7)

    plt.xlabel("Requested rate (RPS)")
    plt.ylabel(f"Successful requests with latency <= {args.threshold:g}s")
    plt.title(f"Fast responses delivered vs offered load")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
