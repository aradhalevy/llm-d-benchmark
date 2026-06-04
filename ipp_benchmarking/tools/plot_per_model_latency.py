#!/usr/bin/env python3
"""Plot per-model e2e latency distributions for one or more collected-logs runs.

Reads the `info.extra_info.raw_response` field of inference-perf's
per_request_lifecycle_metrics.json and extracts the served model name
returned by vLLM. This is the only reliable source of per-pod routing in
the collected results because inference-perf doesn't tag requests with
the gateway routing decision.

Usage:
    plot_per_model_latency.py LABEL=collected-logs-N [LABEL=...] [-o out.png]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",
                      str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt


MODEL_RE = re.compile(r'"model":"([^"]+)"')


def collect_per_model_latency(logs_dir: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    results = logs_dir / "benchmark-results" / "results"
    if not results.is_dir():
        return out
    for tdir in sorted(results.iterdir()):
        path = tdir / "per_request_lifecycle_metrics.json"
        if not path.exists():
            continue
        for r in json.load(open(path)):
            info = r.get("info") or {}
            raw = (info.get("extra_info") or {}).get("raw_response", "")
            m = MODEL_RE.search(raw)
            if not m:
                continue
            served = m.group(1)
            if r.get("error"):
                continue
            start, end = r.get("start_time"), r.get("end_time")
            if not (start and end and end > start):
                continue
            out[served].append(end - start)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="LABEL=path entries")
    parser.add_argument("-o", "--output", default="per_model_latency.png")
    parser.add_argument("--ylim", type=float, default=None,
                        help="Clip y-axis at this value (seconds)")
    args = parser.parse_args()

    scenarios = []
    for entry in args.inputs:
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        data = collect_per_model_latency(Path(path))
        if not data:
            print(f"no per-request data in {path}", file=sys.stderr)
            continue
        scenarios.append((label, data))

    if not scenarios:
        sys.exit(1)

    # Order models consistently: fast pod first, slow pod second.
    model_order = ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B"]
    all_models = sorted(
        {m for _, d in scenarios for m in d if m in model_order},
        key=lambda m: model_order.index(m) if m in model_order else 99,
    )
    model_colors = {
        "Qwen/Qwen3-30B-A3B": "#1f77b4",   # blue — MoE / fast
        "Qwen/Qwen3-32B":     "#d62728",   # red  — dense / slow
    }

    n_scenarios = len(scenarios)
    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5.5 * n_scenarios, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (label, data) in zip(axes, scenarios):
        positions = list(range(1, n_models + 1))
        box_data = [data.get(m, []) for m in all_models]
        bp = ax.boxplot(box_data, positions=positions,
                        showfliers=False, patch_artist=True,
                        widths=0.55, whis=(5, 95))
        for patch, m in zip(bp["boxes"], all_models):
            patch.set_facecolor(model_colors.get(m, "#888"))
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

        # Annotate counts + median + p95
        for x, m in zip(positions, all_models):
            lats = data.get(m, [])
            if not lats:
                continue
            lats_s = sorted(lats)
            p50 = lats_s[len(lats_s) // 2]
            p95 = lats_s[int(len(lats_s) * 0.95)]
            ax.text(x, ax.get_ylim()[1] * 0.97,
                    f"n={len(lats)}\np50={p50:.2f}s\np95={p95:.2f}s",
                    ha="center", va="top", fontsize=9, alpha=0.85)

        ax.set_title(label)
        ax.set_xticks(positions)
        ax.set_xticklabels([m.split("/")[-1] for m in all_models], rotation=0)
        ax.grid(True, axis="y", alpha=0.3)
        if args.ylim is not None:
            ax.set_ylim(0, args.ylim)

    axes[0].set_ylabel("End-to-end latency (s, successful requests)")
    fig.suptitle("Per-model e2e latency — boxes show p25/median/p75, whiskers p5/p95")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
