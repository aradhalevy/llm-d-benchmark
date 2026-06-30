#!/usr/bin/env python3
"""Plot the queue-ttft scorer's PREDICTED effectiveTTFT against the ACTUAL
measured TTFT, from an IPP ipp-tail.log.

  plot_ttft_actual_vs_predicted.py <label>=<ipp-tail.log> [<label>=<log> ...] -o out.png

Predicted: "queue-ttft score" lines  -> effectiveTTFT (s), per request, ts.
Actual:    "ttft-observation" lines   -> ttft_s,        per request, ts.
Both share the IPP wall-clock `ts`, so we bin both into time windows and compare
the per-window medians (left: time series through the concurrency sweep;
right: actual-vs-predicted scatter with y=x and MAE/correlation).

Besides the full-run figure, writes one zoomed PNG per stage (the stage window
+ ~100s pad each side, dense time ticks) into `<out-stem>_stages/`.
"""
import datetime, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

STAGE_PAD_S = 100.0  # seconds of context shown before/after each stage in zooms

BIN_S = 5.0  # time-window for median aggregation
# half-concurrency sweep for the static legs (half_8b/half_32b); override with --concurrencies
CONC_DEFAULT = [25, 75, 125, 175, 225, 275, 225, 175, 125, 75, 25]
_STAGE_RE = re.compile(r"([0-9-]+ [0-9:,]+).*Stage ([0-9]+) - run (started|completed)")


def stage_windows(stdout_path, conc):
    """Exact stage windows from the inference-perf harness log (UTC epoch, ground
    truth) mapped to each stage's profile concurrency. Returns [(conc, start, end)]."""
    starts, ends = {}, {}
    for line in open(stdout_path, errors="replace"):
        m = _STAGE_RE.search(line)
        if not m:
            continue
        ep = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        (starts if m.group(3) == "started" else ends)[int(m.group(2))] = ep
    return [(conc[n], starts[n], ends[n]) for n in sorted(starts)
            if n in ends and n < len(conc)]


def annotate_stages(ax, stdout_path, conc, t0, xmax, xmin=0):
    ytop = ax.get_ylim()[1]
    prev_end = None
    for c, s, e in stage_windows(stdout_path, conc):
        s0, e0 = s - t0, e - t0
        if e0 >= xmin and s0 <= xmax:  # stage (partially) visible
            lo, hi = max(s0, xmin), min(e0, xmax)
            ax.axvspan(lo, hi, color="0.90", zorder=0)
            ax.text((lo + hi) / 2, ytop * 0.97, f"C={c}",
                    ha="center", va="top", fontsize=8, color="dimgray")
            if prev_end is not None and xmin < s0 and s0 - prev_end > 4 and prev_end < xmax:
                ax.text((max(prev_end, xmin) + s0) / 2, ytop * 0.97, "cooldown",
                        ha="center", va="top", fontsize=7, color="silver", style="italic")
        prev_end = e0


def plot_series(ax, at, av, pt, pv):
    if len(at):
        ax.plot(at, av, color="tab:green", lw=1.8, label="actual TTFT (median/5s)")
    if len(pt):
        ax.plot(pt, pv, color="tab:purple", lw=1.8, label="predicted effectiveTTFT (median/5s)")
    ax.set_xlabel("elapsed (s)"); ax.set_ylabel("TTFT (s)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)


def per_stage_zooms(out, label, at, av, pt, pv, stdout_path, conc, t0):
    """One zoomed PNG per stage: stage window +/- STAGE_PAD_S, dense time ticks."""
    stem, _ = os.path.splitext(out)
    d = stem + "_stages"; os.makedirs(d, exist_ok=True)
    wrote = 0
    for n, (c, s, e) in enumerate(stage_windows(stdout_path, conc)):
        x0, x1 = (s - t0) - STAGE_PAD_S, (e - t0) + STAGE_PAD_S
        win = [v[(t >= x0) & (t <= x1)] for t, v in ((at, av), (pt, pv)) if len(t)]
        if not any(len(w) for w in win):
            continue  # truncated log: no data in this stage's window
        ymax = max([np.nanmax(w) for w in win if len(w)] or [1])
        fig, ax = plt.subplots(figsize=(11, 4.2))
        plot_series(ax, at, av, pt, pv)
        ax.set_xlim(x0, x1); ax.set_ylim(0, ymax * 1.12)
        annotate_stages(ax, stdout_path, conc, t0, x1, xmin=x0)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(f"{label}: stage {n} (C={c}) predicted vs actual TTFT")
        fig.tight_layout(); fig.savefig(os.path.join(d, f"stage{n:02d}_C{c}.png"), dpi=130)
        plt.close(fig); wrote += 1
    print(f"{label}: wrote {wrote} stage zooms to {d}/")


def parse(path):
    pred, act = [], []  # (ts, value)
    with open(path, errors="replace") as f:
        for line in f:
            if "queue-ttft score" not in line and "ttft-observation" not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("msg")
            if m == "queue-ttft score" and "effectiveTTFT" in r:
                pred.append((r["ts"], r["effectiveTTFT"]))
            elif m == "ttft-observation" and "ttft_s" in r:
                act.append((r["ts"], r["ttft_s"]))
    return np.array(pred), np.array(act)


def binned_median(ts, val, t0, bin_s=BIN_S):
    if len(ts) == 0:
        return np.array([]), np.array([])
    b = ((ts - t0) // bin_s).astype(int)
    out_t, out_v = [], []
    for k in np.unique(b):
        out_t.append(k * bin_s)
        out_v.append(np.median(val[b == k]))
    return np.array(out_t), np.array(out_v)


def main():
    args = sys.argv[1:]
    out = "ttft_actual_vs_predicted.png"
    conc = CONC_DEFAULT
    if "-o" in args:
        i = args.index("-o"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    if "--concurrencies" in args:
        i = args.index("--concurrencies")
        conc = [int(c) for c in args[i + 1].split(",")]; args = args[:i] + args[i + 2:]
    runs = [a.split("=", 1) for a in args]

    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.2 * n), squeeze=False)
    for row, (label, path) in enumerate(runs):
        pred, act = parse(path)
        ax = axes[row][0]
        if len(pred) == 0 and len(act) == 0:
            ax.set_title(f"{label}: no data"); continue
        t0 = min([x[0] for x in (pred, act) if len(x)][0][0],
                 *([act[0][0]] if len(act) else []), *([pred[0][0]] if len(pred) else []))
        pt, pv = binned_median(pred[:, 0], pred[:, 1], t0) if len(pred) else (np.array([]), np.array([]))
        at, av = binned_median(act[:, 0], act[:, 1], t0) if len(act) else (np.array([]), np.array([]))

        plot_series(ax, at, av, pt, pv)
        ax.set_title(f"{label}: predicted vs actual TTFT over time")

        # certain stage annotation from the harness stage log (sibling of oc-logs/)
        xmax = max([v[-1] for v in (at, pt) if len(v)] or [0])
        ymax = max([np.nanmax(v) for v in (av, pv) if len(v)] or [1])
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax * 1.12)
        d = os.path.dirname(path)  # log may sit in <arm>/ or <arm>/oc-logs/
        stdout_path = next((c for c in (os.path.join(d, "harness_stdout.log"),
                            os.path.join(os.path.dirname(d), "harness_stdout.log"))
                            if os.path.exists(c)), None)
        if stdout_path:
            annotate_stages(ax, stdout_path, conc, t0, xmax)
            per_stage_zooms(out, label, at, av, pt, pv, stdout_path, conc, t0)
        print(f"{label}: predicted={len(pred)} actual={len(act)} bins={len(pt)}")

    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
