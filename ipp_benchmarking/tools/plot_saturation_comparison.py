#!/usr/bin/env python3
"""Plot timeout-failure rate vs request rate for multiple collected-logs runs.

Usage:
    plot_saturation_comparison.py LABEL1=collected-logs-N1 LABEL2=collected-logs-N2 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt


def collect_stage_stats(logs_dir: Path) -> list[tuple[float, float, int, int, float]]:
    """Return list of (requested_rate, timeout_pct, total, timeouts, active_seconds) per stage.

    Uses inference-perf's official per-stage stage_*_lifecycle_metrics.json files
    for accurate stage counts. Timeout count is everything in `failures` (these
    are dominated by 504 ext_proc timeouts under saturation). active_seconds is
    the summed measured benchmark_time across treatments for the stage.
    Aggregates across all treatment dirs in the run.
    """
    by_stage: dict[int, dict[str, float | int]] = {}
    for treatment_dir in sorted((logs_dir / "benchmark-results" / "results").iterdir()):
        for stage_file in sorted(treatment_dir.glob("stage_*_lifecycle_metrics.json")):
            stage_idx = int(stage_file.stem.split("_")[1])
            meta = json.load(open(stage_file))
            ls = meta["load_summary"]
            slot = by_stage.setdefault(
                stage_idx,
                {"rate": ls["requested_rate"], "total": 0, "failures": 0, "seconds": 0.0},
            )
            slot["total"] += ls["count"]
            slot["failures"] += meta["failures"]["count"]
            slot["seconds"] += float(meta.get("benchmark_time_seconds") or 0.0)
    out = []
    for stage_idx in sorted(by_stage):
        slot = by_stage[stage_idx]
        total = int(slot["total"])
        fails = int(slot["failures"])
        pct = fails / total * 100 if total else 0
        out.append((float(slot["rate"]), pct, total, fails, float(slot["seconds"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="saturation_comparison.png")
    parser.add_argument("--metric",
                        choices=["failure", "success", "count", "throughput"],
                        default="failure",
                        help="Plot failure% (HTTP 504), success% (100 - failure%), "
                             "count of successful requests, "
                             "or throughput in successful requests/second.")
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))
    # Stagger annotation offsets so labels don't collide where lines converge.
    # Known labels get fixed sides ("smart" above, "random" below); other
    # labels fall back to index-based positions.
    fixed_offsets = {"smart": (0, 16), "random": (0, -22)}
    fallback_offsets = [(0, 16), (0, -22)]
    for i, entry in enumerate(args.inputs):
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        stats = collect_stage_stats(Path(path))
        if not stats:
            print(f"no stages found in {path}", file=sys.stderr)
            continue
        xs = [s[0] for s in stats]
        if args.metric == "success":
            ys = [100 - s[1] for s in stats]
            ann = [f"{s[2]-s[3]}/{s[2]}" for s in stats]
        elif args.metric == "count":
            ys = [s[2] - s[3] for s in stats]
            ann = [f"{s[2]-s[3]}/{s[2]}" for s in stats]
        elif args.metric == "throughput":
            ys = [(s[2] - s[3]) / s[4] if s[4] > 0 else 0 for s in stats]
            ann = [f"{y:.1f}" for y in ys]
        else:
            ys = [s[1] for s in stats]
            ann = [f"{s[3]}/{s[2]}" for s in stats]
        line, = plt.plot(xs, ys, marker="o", linewidth=2, label=label)
        ox, oy = fixed_offsets.get(label, fallback_offsets[i % len(fallback_offsets)])
        for x, y, a in zip(xs, ys, ann):
            plt.annotate(a, (x, y), textcoords="offset points",
                         xytext=(ox, oy), fontsize=8, alpha=0.95,
                         ha="center", va="center",
                         color=line.get_color())

    plt.xlabel("rate per second (RPS)")
    if args.metric == "success":
        plt.ylabel("Successful requests (% of issued)")
        plt.title("Success rate vs load")
        plt.ylim(0, 105)
    elif args.metric == "count":
        plt.ylabel("Successful requests handled")
        plt.title("Successful requests")
        plt.ylim(bottom=0)
    elif args.metric == "throughput":
        plt.ylabel("Successful requests / second")
        plt.title("Throughput")
        plt.ylim(bottom=0)
    else:
        plt.ylabel("Timeout failures (% of requests, HTTP 504)")
        plt.title("Saturation: timeout-failure rate vs load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
