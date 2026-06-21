#!/usr/bin/env python3
"""Picker decisions over time for the Kind sims — opt-350m (fast) vs opt-125m (slow).

Kind variant of plot_picker_decisions_v2.py (the benchmark_story_4 graph). One
panel per run (LABEL=collected-logs-N):
  * dots = per-request end-to-end latency over time (left axis), colored by the
    backend that served the request -> the p50/p95 spread of SUCCESSFUL requests.
  * filled bands (right axis, 0-100%) = the TRUE routing share over time, taken
    from the IPP "Model selected" log (every picker decision, incl. requests that
    later failed) -> 0..%-to-fast is the fast model's colour, the band above is
    the slow model's. Sourcing the band from selections (not from successes)
    avoids the survivorship bias where a saturating backend that fails most of
    its requests vanishes from a success-only share.
  * cooldown phases (rate ~0) are left blank/white — no band is drawn where there
    is no offered load.

The contrast it surfaces: a load-blind picker keeps choosing the slow sim ~50% of
the time (persistent band) even as its successful dots disappear under load —
i.e. half the traffic is being sent to a backend that craters.

    plot_picker_decisions_kind.py "PR#46 inflight-aware"=collected-logs-1 \
        "maxScore (load-blind)"=collected-logs-2 --rates 3,6,12 -o out.png

Falls back to the success-only share for a run that has no ipp-model-selected.log.
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

# Lower band / "preferred when unsaturated" = the fast sim; upper band = slow sim.
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
    """Return (successes [(t,lat,m)], all_start_times) — successes for the dots,
    all starts (incl. failures) for stage detection and the time origin."""
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
            if m in (FAST, SLOW):  # successful, served-model known
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


def detect_stages(times, tmax, bin_s=5.0, thresh=2.0):
    """Return [(start,end)] of load stages from arrival density (gaps=cooldowns).

    Kind RPS is ~100x lower than the OCP runs, so the density threshold is set
    just above the 0.001 RPS cooldown rate rather than OCP's 40.
    """
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


def share_to_fast(times, is_fast, grid, window):
    times = np.asarray(times); is_fast = np.asarray(is_fast, float)
    out = []
    for t in grid:
        mask = (times >= t - window / 2) & (times <= t + window / 2)
        out.append(is_fast[mask].mean() if mask.any() else np.nan)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="picker_decisions_kind.png")
    ap.add_argument("--window", type=float, default=6.0, help="sliding window (s) for routing share")
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

    lat_max = max(r["lat"] for _, dots, _, _ in runs for r in dots) * 1.05
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (label, dots, starts, sels) in zip(axes, runs):
        # The harness per-request clock (start_time, monotonic) and the IPP "ts"
        # (epoch) are DIFFERENT clocks, so each source is aligned to its own load
        # start (min) — both then begin at t=0, on a shared "since load start" axis.
        t0 = min([d["t"] for d in dots] + starts)   # per-request clock (dots + stages)
        starts_rel = [s - t0 for s in starts]
        tmax = max([d["t"] - t0 for d in dots] + (starts_rel or [0]))

        # ---- true routing-share band from IPP selections, blank during cooldowns
        ax2 = ax.twinx()
        stages = detect_stages(starts_rel or [d["t"] - t0 for d in dots], tmax)
        band_src = "IPP selections" if sels else "successes (no IPP log)"
        if sels:
            ts0 = min(s[0] for s in sels)           # IPP epoch clock, own origin
            st_times = [s[0] - ts0 for s in sels]
            st_fast = [s[1] == FAST for s in sels]
        else:  # fallback: success-only (same clock as dots)
            st_times = [d["t"] - t0 for d in dots]
            st_fast = [d["m"] == FAST for d in dots]
        grid = np.linspace(0, tmax, 300)
        frac = share_to_fast(st_times, st_fast, grid, args.window) * 100
        # blank out cooldowns (and any gap with no offered load): NaN -> no fill
        in_stage = np.array([any(s <= t <= e for s, e in stages) for t in grid])
        frac = np.where(in_stage, frac, np.nan)
        valid = ~np.isnan(frac)
        ax2.fill_between(grid, 0, frac, where=valid, color=COLORS[FAST], alpha=0.16, zorder=0)
        ax2.fill_between(grid, frac, 100, where=valid, color=COLORS[SLOW], alpha=0.16, zorder=0)
        ax2.plot(grid, frac, color=COLORS[FAST], linewidth=1.0, alpha=0.55, zorder=1)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel(f"% routed  (lower band = {short(FAST)}, upper = {short(SLOW)})", fontsize=9)
        denom = len(sels) if sels else len(dots)
        overall = 100.0 * (sum(st_fast) / denom) if denom else float("nan")
        ax2.text(0.99, 0.02, f"{overall:.0f}% {short(FAST)}  (band: {band_src})", transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=8.5, color="#333")

        # ---- latency dots for SUCCESSFUL requests, coloured by served backend
        times = [d["t"] - t0 for d in dots]
        for m in (FAST, SLOW):
            xs = [t for t, d in zip(times, dots) if d["m"] == m]
            ys = [d["lat"] for d in dots if d["m"] == m]
            ax.scatter(xs, ys, s=9, alpha=0.5, color=COLORS[m], edgecolors="none",
                       zorder=3, label=f"{short(m)} ({'fast' if m == FAST else 'slow'} model, successes)")
        ax.set_ylim(0, lat_max)
        ax.set_xlim(0, tmax)
        ax.set_xlabel("time since load start (s)")
        ax.set_ylabel("request latency (s)")
        ax.set_title(f"{label}: latency dots (successes) + true routing share")
        ax.grid(True, alpha=0.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # ---- stage / cooldown annotations
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

    fig.suptitle("Picker decisions over time — latency of successes (dots) over the true routing share (bands)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
