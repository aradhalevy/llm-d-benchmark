#!/usr/bin/env python3
"""Summarizer latency vs concurrency (STACKED bars) with a per-model TIMEOUT BAND.

Top panel: per-model e2e p50 (solid) / p95 (dashed). The percentile is computed over that model's
requests INCLUDING its timed-out ones (assigned the ~31s route-timeout value), so when a model's
requests are mostly timing out, its percentile rises past 30s and is drawn as a flat LINE in the
band above the cutoff (same colour as the model, same dash as the percentile) — the curve simply
continues above the 30s line, separated by the gap. Green = 8B, purple = 32B.

Per-model timeout count = (picker routed this model, from the IPP decision log) − (completed on it,
from per_request). This is correct even when one backend fails far more than the other (e.g. random
routing: ~50% goes to the 32B but ~98% of THOSE time out — the band shows the 32B line up top, not
the 8B).

Bottom panel: STRICTLY the summarizer's own requests — stacked 8B-served / 32B-served / timed-out.

    plot_latency_stacked_timeoutband_ab.py "smart"=.../smart "random"=.../random \
        --concurrencies 100,200,300,400,500 -o out.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SMALL, BIG = "small", "big"
NAME = {SMALL: "Qwen3-8B (small)", BIG: "Qwen3-32B (big)"}
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}
TMO_C = "#d62728"
PCTS = [("p50", 50, "-"), ("p95", 95, (0, (5, 3)))]
CEIL = 30.0
TIMEOUT_LAT = 31.0
GAP = 1.5
SPACING = 1.5
BAND_ORDER = [(SMALL, "p50"), (SMALL, "p95"), (BIG, "p50"), (BIG, "p95")]


def _trim(stages, frac):
    """Drop the first/last `frac` of each (time-sorted) stage — keep the middle, so
    warmup/cooldown points don't bias the percentiles or the routed/timeout counts."""
    if not frac:
        return stages
    out = []
    for s in stages:
        k = int(len(s) * frac)
        out.append(s[k:len(s) - k] if k else s)
    return out


def _merge_to_n(stages, n):
    while len(stages) > n:
        gaps = [stages[i + 1][0] - stages[i][-1] for i in range(len(stages) - 1)]  # by first/last key
        i = gaps.index(min(gaps))
        stages[i] = stages[i] + stages[i + 1]
        del stages[i + 1]
    return stages


def slim_stages(run: Path, n, trim=0.0):
    rows = json.load(open(run / "per_request_slim.json"))
    rows.sort(key=lambda r: r["t"])
    stages, cur = [], [rows[0]]
    for p, r in zip(rows, rows[1:]):
        if r["t"] - p["t"] > 8:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= 20]
    while len(stages) > n:
        gaps = [stages[i + 1][0]["t"] - stages[i][-1]["t"] for i in range(len(stages) - 1)]
        i = gaps.index(min(gaps)); stages[i] = stages[i] + stages[i + 1]; del stages[i + 1]
    return _trim(stages, trim)


def picker_per_stage(run: Path, n, trim=0.0):
    """(8B,32B) picker counts per stage from the IPP decision log, deduped by x-request-id."""
    f = run / "ipp-decisions.log"
    if not f.exists():
        return None
    sel, ts = {}, {}
    for line in open(f, errors="ignore"):
        if '"msg":"Model selected"' not in line:
            continue
        rid = re.search(r'"x-request-id":"([^"]+)"', line)
        m = BIG if "Qwen3-32B" in line else (SMALL if "Qwen3-8B" in line else None)
        if not rid or not m:
            continue
        sel[rid.group(1)] = m
        t = re.search(r'"ts":([0-9.]+)', line); ts[rid.group(1)] = float(t.group(1)) if t else 0.0
    rows = sorted((ts[r], sel[r]) for r in sel)
    if not rows:
        return None
    stages, cur = [], [rows[0]]
    for p, r in zip(rows, rows[1:]):
        if r[0] - p[0] > 8:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= 15]
    while len(stages) > n:
        gaps = [stages[i + 1][0][0] - stages[i][-1][0] for i in range(len(stages) - 1)]
        i = gaps.index(min(gaps)); stages[i] = stages[i] + stages[i + 1]; del stages[i + 1]
    stages = _trim(stages, trim)
    return [{SMALL: sum(1 for _, m in s if m == SMALL), BIG: sum(1 for _, m in s if m == BIG)} for s in stages]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default="latency_stacked_timeoutband_ab.png")
    ap.add_argument("--concurrencies", default="100,200,300,400,500")
    ap.add_argument("--connect", action="store_true",
                    help="draw each percentile as ONE continuous line through the band "
                         "(in-band stages sit at the band level) instead of breaking at the cutoff")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="drop this fraction from the start AND end of each stage "
                         "(0.1 = keep the middle 80%%) to remove warmup/cooldown bias")
    ap.add_argument("--bottom-tick", type=int, default=0,
                    help="y-tick spacing on the request-volume panel (e.g. 200); 0 = auto")
    ap.add_argument("--height-ratios", default="2.6,1.0",
                    help="top,bottom panel height ratio")
    ap.add_argument("--drop-last", type=int, default=0,
                    help="drop this many trailing concurrency stages (cluster to the full "
                         "set first, then drop — so the remaining stages aren't merged)")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    n_full = len(conc)
    runs = []
    for e in args.inputs:
        if "=" not in e:
            continue
        lbl, p = e.split("=", 1)
        st = slim_stages(Path(p), n_full, args.trim)
        pk = picker_per_stage(Path(p), n_full, args.trim)
        if args.drop_last:
            st = st[:-args.drop_last]
            pk = pk[:-args.drop_last] if pk else pk
        runs.append((lbl, st, pk))
    if args.drop_last:
        conc = conc[:-args.drop_last]

    level = {k: CEIL + GAP + i * SPACING for i, k in enumerate(BAND_ORDER)}
    y_top = CEIL + GAP + (len(BAND_ORDER) - 1) * SPACING + 1.2
    bar_max = max(len(s) for _, stages, _ in runs for s in stages) * 1.08
    nr = len(runs)
    hr = [float(x) for x in args.height_ratios.split(",")]
    fig, axes = plt.subplots(2, nr, figsize=(7.4 * nr, 7.8), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": hr})

    for col, (label, stages, pick) in enumerate(runs):
        axL, axB = axes[0][col], axes[1][col]
        xs = conc[:len(stages)]

        axL.axhspan(CEIL + GAP, y_top, color="#eee", zorder=0)
        axL.axhline(CEIL, color="#333", lw=1.4, zorder=2)
        axL.text(xs[-1], CEIL, "30s timeout ", color="#333", fontsize=8, va="bottom", ha="right", zorder=3)

        for model in (SMALL, BIG):
            for name, q, ls in PCTS:
                real, band = [], []
                for i, st in enumerate(stages):
                    succ = [r["lat"] for r in st if r["m"] == model and not r["fail"]]
                    # per-model timeouts = routed here (picker) − completed here; assign ~31s.
                    # Cold-start keepalive/warm probes leak extra picks into the decision log
                    # (stage 0 only): scale the stage's picker counts down to the slim stage
                    # total so the leaked picks don't masquerade as fast-model timeouts.
                    if pick and i < len(pick):
                        pick_tot = pick[i][SMALL] + pick[i][BIG]
                        scale = min(1.0, len(st) / pick_tot) if pick_tot else 1.0
                        routed = round(pick[i][model] * scale)
                    else:
                        routed = len(succ)
                    n_tmo = max(0, routed - len(succ))
                    lat = succ + [TIMEOUT_LAT] * n_tmo
                    v = float(np.percentile(lat, q)) if lat else np.nan
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
                    axL.plot(xs, band, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=4, zorder=5)

        axL.set_ylim(0, y_top)
        axL.set_yticks([0, 5, 10, 15, 20, 25, 30])
        axL.set_ylabel("e2e latency (s)")
        axL.set_title(label, fontsize=11)
        axL.grid(True, axis="y", alpha=0.2)
        mh = [Line2D([0], [0], color=COLOR[m], lw=2.4, label=NAME[m]) for m in (SMALL, BIG)]
        sh = [Line2D([0], [0], color="#444", lw=2.0, ls=lsx, label=nm) for nm, _, lsx in PCTS]
        leg1 = axL.legend(handles=mh, loc="upper left", fontsize=8.5, framealpha=0.92)
        axL.add_artist(leg1)
        axL.legend(handles=sh, loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=8.5,
                   framealpha=0.92, title="percentile")

        # bottom: STRICTLY summarizer, stacked served + timed-out
        xa = np.array(xs, float)
        ss = [sum(1 for r in st if r["m"] == SMALL and not r["fail"]) for st in stages]
        bs = [sum(1 for r in st if r["m"] == BIG and not r["fail"]) for st in stages]
        fl = [sum(1 for r in st if r["fail"]) for st in stages]
        w = (xs[1] - xs[0]) * 0.6 if len(xs) > 1 else 6
        axB.bar(xa, ss, width=w, color=COLOR[SMALL])
        axB.bar(xa, bs, width=w, bottom=ss, color=COLOR[BIG])
        axB.bar(xa, fl, width=w, bottom=np.add(ss, bs), color=TMO_C)
        axB.set_ylim(0, bar_max)
        if args.bottom_tick:
            axB.set_yticks(np.arange(0, bar_max + args.bottom_tick, args.bottom_tick))
        axB.set_xticks(conc)
        axB.set_xlabel("concurrent clients")
        axB.set_ylabel("requests")
        axB.grid(True, axis="y", alpha=0.2)
        if col == 0:
            axB.legend(handles=[Patch(color=COLOR[SMALL], label="8B served"),
                                Patch(color=COLOR[BIG], label="32B served"),
                                Patch(color=TMO_C, label="timed out")],
                       loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Summarizer e2e latency + request volume vs. concurrency", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
