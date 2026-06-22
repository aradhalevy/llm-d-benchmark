#!/usr/bin/env python3
"""Latency percentiles vs. concurrency, per model — OCP Qwen3-8B/32B research-agent.

Adapted from plot_latency_vs_concurrency_kind.py. Two stacked panels per run:
  * TOP — e2e latency p50 (solid) / p95 (dashed); colour = backend
    (green = Qwen3-8B small/fast, purple = Qwen3-32B big). A percentile dominated
    by timeouts leaves the main 0..30s region and reappears in the separated
    TIMEOUT band above the break.
  * BOTTOM — request volume per concurrency: green = served by 8B, purple = served
    by 32B, red = failed (timed out). NOTE the OCP difference vs Kind: failures are
    NOT attributed to a model (no served model in the response, and unlike the Kind
    fast sim the 8B *does* fail at high load), so the red bar is a single combined
    category, not split per model.

Per-model split = served model echoed in each response (ground truth, successes
only). Input is the slim extract from extract_per_request_slim.py.

    plot_latency_vs_concurrency_ocp.py "smart router"=.../smart_planner \
        --concurrencies 100,200,300,400,500 -o out.png
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
from matplotlib.patches import Patch

SMALL, BIG = "small", "big"
MODEL_NAME = {SMALL: "Qwen3-8B (small)", BIG: "Qwen3-32B (big)"}
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}   # green / purple
TMO_COLOR = "#d62728"                          # red, failed
PCTS = [("p50", 50, "-"), ("p95", 95, (0, (5, 3)))]
CEIL = 30.0
BAND_ORDER = [(SMALL, "p50"), (SMALL, "p95"), (BIG, "p50"), (BIG, "p95")]
GAP = 1.5
SPACING = 1.15


def load_rows(run: Path):
    slim = run / "per_request_slim.json"
    if not slim.exists():
        sys.exit(f"missing {slim} — run extract_per_request_slim.py first")
    rows = [{"t": r["t"], "lat": r["lat"], "fail": r["fail"], "m": r["m"]}
            for r in json.load(open(slim))]
    rows.sort(key=lambda x: x["t"])
    return rows


def cluster_stages(rows, gap, min_size, n=None, trim=0.1):
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["t"] - prev["t"] > gap:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= min_size]
    # a high-fail stage has 31s-timeout lulls that split it; merge adjacent clusters
    # (smallest inter-gap first) down to the expected number of concurrency levels
    while n and len(stages) > n:
        gaps = [stages[i + 1][0]["t"] - stages[i][-1]["t"] for i in range(len(stages) - 1)]
        i = int(np.argmin(gaps))
        stages[i] = stages[i] + stages[i + 1]
        del stages[i + 1]
    out = []
    for s in stages:
        k = int(len(s) * trim)
        out.append(s[k:len(s) - k] if k else s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path entries")
    ap.add_argument("-o", "--output", default="latency_vs_concurrency_ocp.png")
    ap.add_argument("--concurrencies", default="100,200,300,400,500")
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--bars", choices=("split", "stacked"), default="split")
    ap.add_argument("--connect", action="store_true",
                    help="draw each percentile as ONE continuous line through the band "
                         "(in-band stages sit at the band level) instead of breaking at the cutoff")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        stages = cluster_stages(load_rows(Path(path)), args.gap, args.min_size, n=len(conc))
        if len(stages) != len(conc):
            print(f"warn {path}: {len(stages)} stages vs {len(conc)} concurrencies", file=sys.stderr)
        runs.append((label, stages))
    if not runs:
        sys.exit("no runs")

    level = {key: CEIL + GAP + i * SPACING for i, key in enumerate(BAND_ORDER)}
    y_top = CEIL + GAP + (len(BAND_ORDER) - 1) * SPACING + 1.0
    if args.bars == "stacked":
        bar_max = max(len(s) for _, stages in runs for s in stages) * 1.05
    else:
        bar_max = max(sum(1 for r in s if r["m"] == m and not r["fail"])
                      for _, stages in runs for s in stages for m in (SMALL, BIG)) * 1.15
        bar_max = max(bar_max, max(sum(1 for r in s if r["fail"])
                      for _, stages in runs for s in stages) * 1.15)

    nr = len(runs)
    fig, axes = plt.subplots(2, nr, figsize=(7.4 * nr, 7.6), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2.5, 1.0]})

    for col, (label, stages) in enumerate(runs):
        axL = axes[0][col]
        axB = axes[1][col]
        xs = conc[:len(stages)]

        axL.axhspan(CEIL + GAP, y_top, color="#eee", zorder=0)
        axL.axhline(CEIL, color="#333", lw=1.4, zorder=2)
        axL.text(xs[-1], CEIL, "30s timeout ", color="#333", fontsize=8, va="bottom", ha="right", zorder=3)

        for model in (SMALL, BIG):
            for name, q, ls in PCTS:
                real, band = [], []
                for st in stages[:len(conc)]:
                    lat = [r["lat"] for r in st if r["m"] == model and not r["fail"]]
                    v = np.percentile(lat, q) if lat else np.nan
                    if not np.isnan(v) and v > CEIL:
                        real.append(np.nan); band.append(level[(model, name)])
                    else:
                        real.append(v); band.append(np.nan)
                if args.connect:
                    # one continuous line: in-band stages take the band level, so the
                    # curve steps up into the band and back down without a gap
                    y = [b if np.isnan(r) else r for r, b in zip(real, band)]
                    axL.plot(xs, y, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=4, zorder=5)
                else:
                    axL.plot(xs, real, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=4, zorder=5)
                    axL.plot(xs, band, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=5, zorder=5)

        axL.set_ylim(0, y_top)
        axL.set_yticks([0, 5, 10, 15, 20, 25, 30])
        axL.set_ylabel("e2e latency (s)")
        axL.set_title(label, fontsize=11)
        axL.grid(True, axis="y", alpha=0.2)
        model_h = [Line2D([0], [0], color=COLOR[m], lw=2.4, label=MODEL_NAME[m]) for m in (SMALL, BIG)]
        style_h = [Line2D([0], [0], color="#444", lw=2.0, ls=ls, label=name) for name, _, ls in PCTS]
        leg1 = axL.legend(handles=model_h, loc="upper left", fontsize=8.5, framealpha=0.92)
        axL.add_artist(leg1)
        axL.legend(handles=style_h, loc="upper left", bbox_to_anchor=(0.0, 0.78),
                   fontsize=8.5, framealpha=0.92, title="percentile")

        sub = stages[:len(conc)]
        xa = np.array(xs, float)
        ss = [sum(1 for r in st if r["m"] == SMALL and not r["fail"]) for st in sub]
        bs = [sum(1 for r in st if r["m"] == BIG and not r["fail"]) for st in sub]
        fl = [sum(1 for r in st if r["fail"]) for st in sub]
        if args.bars == "stacked":
            w = (xs[1] - xs[0]) * 0.6 if len(xs) > 1 else 6
            axB.bar(xa, ss, width=w, color=COLOR[SMALL])
            axB.bar(xa, bs, width=w, bottom=ss, color=COLOR[BIG])
            axB.bar(xa, fl, width=w, bottom=np.add(ss, bs), color=TMO_COLOR)
        else:  # split: 8B-served and 32B-served side by side, failed (combined) as a 3rd bar
            w = (xs[1] - xs[0]) * 0.26 if len(xs) > 1 else 3
            axB.bar(xa - w, ss, width=w, color=COLOR[SMALL])
            axB.bar(xa, bs, width=w, color=COLOR[BIG])
            axB.bar(xa + w, fl, width=w, color=TMO_COLOR)
        axB.set_ylim(0, bar_max)
        axB.set_xticks(conc)
        axB.set_xlabel("concurrent clients")
        axB.set_ylabel("requests")
        axB.grid(True, axis="y", alpha=0.2)
        if col == 0:
            axB.legend(handles=[Patch(color=COLOR[SMALL], label="8B served"),
                                Patch(color=COLOR[BIG], label="32B served"),
                                Patch(color=TMO_COLOR, label="failed (either)")],
                       loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Summarizer e2e latency + request volume vs. concurrency", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
