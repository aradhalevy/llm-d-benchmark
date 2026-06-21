#!/usr/bin/env python3
"""Latency percentiles vs. concurrency for the Kind sims, per model.

Per run (one column): two stacked panels sharing the x axis (concurrent clients):
  * TOP — latency percentiles p50 (solid) / p95 (dashed); colour = backend
    (green = fast opt-350m, purple = slow opt-125m). Real latency is drawn in the
    main 0..30s region; the moment a percentile is dominated by timeouts it leaves
    the main region (cut off at the 30s line) and reappears as a flat marker in the
    separated TIMEOUT band above the break, one spaced level per (model,percentile)
    so every timed-out curve stays visible.
  * BOTTOM — a stacked request-volume bar per concurrency: green = requests served
    by fast, purple = served by slow, red = timed out (across both pools).

Timeouts are real: a failed request gives up at the ~31s gateway-504 ceiling and
contributes that wall-time. Attribution: successes by the served model in the
response; failures (no served model) → slow (the fast sim never times out here).

    plot_latency_vs_concurrency_kind.py "random"=.../random "PR#46"=.../inflight \
        --concurrencies 30,40,50,60,70,80,90,100,110,120 -o out.png
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

FAST, SLOW = "fast", "slow"
MODEL_NAME = {FAST: "opt-350m (fast)", SLOW: "opt-125m (slow)"}
COLOR = {FAST: "#2ca02c", SLOW: "#9467bd"}   # green / purple
TMO_COLOR = "#d62728"                          # red, for timed-out requests
PCTS = [("p50", 50, "-"), ("p95", 95, (0, (5, 3)))]
CEIL = 30.0                                    # timeout threshold / top bar line
# one spaced level per (model, percentile) in the band above the break, low -> high
BAND_ORDER = [(FAST, "p50"), (FAST, "p95"), (SLOW, "p50"), (SLOW, "p95")]
GAP = 1.5            # blank cutoff between the main region and the band
SPACING = 1.15       # vertical spacing of timed-out curves in the band


def served_model(rec: dict):
    blob = json.dumps(rec.get("response")) + json.dumps(rec.get("info") or {})
    if "350m" in blob:
        return FAST
    if "125m" in blob:
        return SLOW
    return None


def load_rows(run: Path):
    pf = max(glob.glob(str(run / "per_request_lifecycle_metrics.json"))
             or glob.glob(str(run / "benchmark-results" / "results" / "*" / "per_request_lifecycle_metrics.json")),
             key=lambda f: len(json.load(open(f))))
    rows = []
    for r in json.load(open(pf)):
        st, et = r.get("start_time"), r.get("end_time")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            served = served_model(r)
            rows.append({"t": st, "lat": et - st, "fail": bool(r.get("error")),
                         "m": served if served else SLOW})
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
    ap.add_argument("-o", "--output", default="latency_vs_concurrency_kind.png")
    ap.add_argument("--concurrencies", default="30,40,50,60,70,80,90,100,110,120")
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--bars", choices=("split", "stacked"), default="split",
                    help="bottom panel: per-model side-by-side bars (split) or one stacked bar (stacked)")
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

    level = {key: CEIL + GAP + i * SPACING for i, key in enumerate(BAND_ORDER)}
    y_top = CEIL + GAP + (len(BAND_ORDER) - 1) * SPACING + 1.0
    if args.bars == "stacked":
        bar_max = max(len(s) for _, stages in runs for s in stages) * 1.05
    else:
        bar_max = max(sum(1 for r in s if r["m"] == m) for _, stages in runs for s in stages
                      for m in (FAST, SLOW)) * 1.08

    nr = len(runs)
    fig, axes = plt.subplots(2, nr, figsize=(7.4 * nr, 7.6), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [2.5, 1.0]})

    for col, (label, stages) in enumerate(runs):
        axL = axes[0][col]   # latency
        axB = axes[1][col]   # stacked request bars
        xs = conc[:len(stages)]

        # ---- timeout band scaffold (top panel)
        axL.axhspan(CEIL + GAP, y_top, color="#eee", zorder=0)
        axL.axhline(CEIL, color="#333", lw=1.4, zorder=2)
        axL.text(xs[-1], CEIL, "30s timeout ", color="#333", fontsize=8, va="bottom", ha="right", zorder=3)

        # ---- percentile curves: real (<=30) as lines, timeouts in the band
        for model in (FAST, SLOW):
            for name, q, ls in PCTS:
                real, band = [], []
                for st in stages[:len(conc)]:
                    lat = [r["lat"] for r in st if r["m"] == model]
                    v = np.percentile(lat, q) if lat else np.nan
                    if not np.isnan(v) and v > CEIL:
                        real.append(np.nan); band.append(level[(model, name)])
                    else:
                        real.append(v); band.append(np.nan)
                axL.plot(xs, real, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=4, zorder=5)
                axL.plot(xs, band, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=5, zorder=5)

        axL.set_ylim(0, y_top)
        axL.set_yticks([0, 5, 10, 15, 20, 25, 30])
        axL.set_ylabel("end-to-end latency (s)")
        axL.set_title(label, fontsize=11)
        axL.grid(True, axis="y", alpha=0.2)
        model_h = [Line2D([0], [0], color=COLOR[m], lw=2.4, label=MODEL_NAME[m]) for m in (FAST, SLOW)]
        style_h = [Line2D([0], [0], color="#444", lw=2.0, ls=ls, label=name) for name, _, ls in PCTS]
        leg1 = axL.legend(handles=model_h, loc="upper left", fontsize=8.5, framealpha=0.92)
        axL.add_artist(leg1)
        axL.legend(handles=style_h, loc="upper left", bbox_to_anchor=(0.0, 0.78),
                   fontsize=8.5, framealpha=0.92, title="percentile")

        # ---- bottom: request volume; green=fast served, purple=slow served, red=timed out
        sub = stages[:len(conc)]
        xa = np.array(xs, float)
        fs = [sum(1 for r in st if r["m"] == FAST and not r["fail"]) for st in sub]
        ff = [sum(1 for r in st if r["m"] == FAST and r["fail"]) for st in sub]
        ps = [sum(1 for r in st if r["m"] == SLOW and not r["fail"]) for st in sub]
        pf = [sum(1 for r in st if r["m"] == SLOW and r["fail"]) for st in sub]
        if args.bars == "stacked":
            w = (xs[1] - xs[0]) * 0.6 if len(xs) > 1 else 6
            axB.bar(xa, fs, width=w, color=COLOR[FAST])
            axB.bar(xa, ps, width=w, bottom=fs, color=COLOR[SLOW])
            axB.bar(xa, np.add(ff, pf), width=w, bottom=np.add(fs, ps), color=TMO_COLOR)
        else:  # split: one bar per model, side by side, red fails on top of each
            w = (xs[1] - xs[0]) * 0.34 if len(xs) > 1 else 3
            off = w * 0.55
            axB.bar(xa - off, fs, width=w, color=COLOR[FAST])
            axB.bar(xa - off, ff, width=w, bottom=fs, color=TMO_COLOR)
            axB.bar(xa + off, ps, width=w, color=COLOR[SLOW])
            axB.bar(xa + off, pf, width=w, bottom=ps, color=TMO_COLOR)
        axB.set_ylim(0, bar_max)
        axB.set_xticks(conc)
        axB.set_xlabel("concurrent clients")
        axB.set_ylabel("requests")
        axB.grid(True, axis="y", alpha=0.2)
        if col == 0:
            axB.legend(handles=[Patch(color=COLOR[FAST], label="fast served"),
                                Patch(color=COLOR[SLOW], label="slow served"),
                                Patch(color=TMO_COLOR, label="timed out")],
                       loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Latency percentiles (p50/p95) + request volume vs. concurrency, per model",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
