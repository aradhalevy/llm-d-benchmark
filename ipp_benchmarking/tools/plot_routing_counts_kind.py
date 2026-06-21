#!/usr/bin/env python3
"""Routing counts over time for the Kind sims — absolute requests routed per model.

Companion to plot_picker_decisions_kind.py. Same left axis (per-request latency
dots of successes, coloured by served backend), but the right axis replaces the
0-100% routing-share band with, for each backend, a LINE of the absolute number
of requests the picker routed to it per time bin (from the IPP "Model selected"
log = every decision, incl. requests that later failed). One panel per run.

  * dots  = latency of SUCCESSFUL requests (left axis), green=fast, red=slow
  * lines = absolute requests routed per --bin seconds (right axis), same colours
  * cooldown phases (no offered load) read as both lines at ~0

    plot_routing_counts_kind.py "PR#46 inflight-aware"=collected-logs-1 \
        "maxScore (load-blind)"=collected-logs-2 --rates 3,6,12 -o routing_counts.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
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

FAST = "facebook/opt-350m"   # TTFT 1s / ITL 50ms
SLOW = "facebook/opt-125m"   # TTFT 3s / ITL 200ms
COLORS = {FAST: "#2ca02c", SLOW: "#d62728"}


def short(m: str) -> str:
    return m.rsplit("/", 1)[-1]


def _model_of(text: str) -> str | None:
    if "350m" in text:
        return FAST
    if "125m" in text:
        return SLOW
    return None


def served(rec: dict) -> str | None:
    s = rec.get("response") if isinstance(rec.get("response"), str) else json.dumps(rec.get("response"))
    m = re.search(r'"model":"([^"]+)"', s or "")
    return _model_of(m.group(1)) if m else None


def load_requests(run: Path):
    """Return (successes [(t,lat,m)], all_start_times) for the latency dots."""
    files = glob.glob(str(run / "benchmark-results" / "results" / "*" / "per_request_lifecycle_metrics.json"))
    if not files:
        return [], []
    best = max(files, key=lambda f: len(json.load(open(f))))
    recs, starts = [], []
    for r in json.load(open(best)):
        st = r.get("start_time")
        if isinstance(st, (int, float)):
            starts.append(st)
        et = r.get("end_time")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            m = served(r)
            if m in (FAST, SLOW):
                recs.append({"t": st, "lat": et - st, "m": m})
    recs.sort(key=lambda x: x["t"])
    return recs, sorted(starts)


def load_selections(run: Path):
    """Parse the IPP 'Model selected' log -> [(ts_epoch, model)] true decisions."""
    log = run / "ipp-model-selected.log"
    if not log.exists():
        return []
    sels = []
    for line in open(log):
        if "Model selected" not in line:
            continue
        mt = re.search(r'"ts":([0-9.]+)', line)
        m = _model_of(line)
        if mt and m:
            sels.append((float(mt.group(1)), m))
    sels.sort()
    return sels


def smooth(y, win_bins):
    """Centered moving average over win_bins (>=1 = no smoothing)."""
    win_bins = int(round(win_bins))
    if win_bins <= 1:
        return np.asarray(y, float)
    k = np.ones(win_bins) / win_bins
    return np.convolve(np.asarray(y, float), k, mode="same")


def nice_ticks(vmax):
    """A small set of 'nice' tick values from 0..vmax (for the count axis)."""
    if vmax <= 0:
        return np.array([0])
    raw = vmax / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return np.arange(0, vmax + step * 0.5, step)


def detect_stages(times, tmax, bin_s=5.0, thresh=2.0):
    """[(start,end)] of load stages from arrival density (gaps=cooldowns)."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="routing_counts_kind.png")
    ap.add_argument("--bin", type=float, default=1.0, help="time bin width (s) for the counts")
    ap.add_argument("--smooth", type=float, default=5.0,
                    help="moving-average window (s) applied to the count lines; <=bin disables")
    ap.add_argument("--rates", default="3,6,12", help="comma-separated load-stage rates in order")
    args = ap.parse_args()
    rates = [r.strip() for r in args.rates.split(",") if r.strip()]

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        run = Path(path)
        dots, starts = load_requests(run)
        sels = load_selections(run)
        if dots:
            runs.append((label, dots, starts, sels))
    if not runs:
        print("no per-request data", file=sys.stderr)
        sys.exit(1)

    lat_max = max(d["lat"] for _, dots, _, _ in runs for d in dots) * 1.05

    # shared count y-scale: prep the per-bin counts up front
    prepped, count_max = [], 0.0
    for label, dots, starts, sels in runs:
        # dots clock (per-request, monotonic): aligned to its own load start
        t0 = min([d["t"] for d in dots] + starts)
        starts_rel = [s - t0 for s in starts]
        tmax_dots = max([d["t"] - t0 for d in dots] + (starts_rel or [0]))
        # count lines from IPP selections (epoch clock): aligned to its own load start
        if sels:
            ts0 = min(s[0] for s in sels)
            stimes = np.array([s[0] - ts0 for s in sels])
            smodels = [s[1] for s in sels]
        else:  # fallback: use successful served model on the dots clock
            stimes = np.array([d["t"] - t0 for d in dots])
            smodels = [d["m"] for d in dots]
        tmax = max(tmax_dots, float(stimes.max()) if len(stimes) else 0.0)
        edges = np.arange(0, tmax + args.bin, args.bin)
        centers = edges[:-1] + args.bin / 2
        win_bins = max(1, round(args.smooth / args.bin))
        counts = {}
        for m in (FAST, SLOW):
            mt = stimes[[mm == m for mm in smodels]]
            raw, _ = np.histogram(mt, bins=edges)
            counts[m] = smooth(raw, win_bins)
        count_max = max(count_max, max(counts[FAST].max(), counts[SLOW].max()))
        prepped.append((label, dots, t0, starts_rel, tmax, centers, counts))

    n = len(prepped)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.2), squeeze=False)
    axes = axes[0]

    # Dots are squeezed into the lower ~half (extra latency headroom); the count
    # lines are offset into the upper ~half so the two never overlap.
    lat_top = lat_max * 2.0
    for ax, (label, dots, t0, starts_rel, tmax, centers, counts) in zip(axes, prepped):
        # ---- right axis: absolute routing-count lines, floated into the top band
        ax2 = ax.twinx()
        ax2.axhline(0, color="#999", lw=0.7, alpha=0.5, zorder=0)  # count baseline
        for m in (FAST, SLOW):
            ax2.plot(centers, counts[m], "-", color=COLORS[m], linewidth=2.2, alpha=0.95, zorder=4)
        ax2.set_ylim(-1.6 * count_max, count_max * 1.1)  # 0 sits ~60% up -> lines on top
        ax2.set_yticks(nice_ticks(count_max))             # only 0..count_max labelled
        smooth_note = f"  ({args.smooth:g}s moving avg)" if args.smooth > args.bin else ""
        ax2.set_ylabel(f"requests routed per {args.bin:g}s{smooth_note}  (lines, upper band)", fontsize=9)

        # ---- left axis: latency dots of successes, coloured by served backend
        times = [d["t"] - t0 for d in dots]
        for m in (FAST, SLOW):
            xs = [t for t, d in zip(times, dots) if d["m"] == m]
            ys = [d["lat"] for d in dots if d["m"] == m]
            ax.scatter(xs, ys, s=9, alpha=0.45, color=COLORS[m], edgecolors="none",
                       zorder=3, label=f"{short(m)} ({'fast' if m == FAST else 'slow'} model)")
        ax.set_ylim(0, lat_top)
        ax.set_yticks(np.arange(0, lat_max + 1e-9, max(5, round(lat_max / 4 / 5) * 5)))
        ax.set_xlim(0, tmax)
        ax.set_xlabel("time since load start (s)")
        ax.set_ylabel("request latency of successes (s)  (dots, lower band)")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        stages = detect_stages(starts_rel or times, tmax)
        for k, (s, e) in enumerate(stages[:len(rates)]):
            ax.axvline(s, color="#444", ls=":", lw=0.8, zorder=2)
            ax.axvline(e, color="#444", ls=":", lw=0.8, zorder=2)
            ax.text((s + e) / 2, lat_top * 0.985, f"{rates[k]} RPS", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#111", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        for k in range(len(stages[:len(rates)]) - 1):
            gs, ge = stages[k][1], stages[k + 1][0]
            if ge - gs > 6:
                ax.text((gs + ge) / 2, lat_top * 0.985, "cooldown", ha="center", va="top",
                        fontsize=8, style="italic", color="#444", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    fig.suptitle("Routing decisions over time — latency of successes (dots) + absolute requests routed to each backend (lines)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
