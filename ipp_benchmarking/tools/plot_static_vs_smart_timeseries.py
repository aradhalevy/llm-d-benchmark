#!/usr/bin/env python3
"""Throughput + latency over elapsed time (not per-stage-aggregated), for the
static 8B, static 32B, and smart runs on one shared, stage-aligned timeline.

Runs' stages have different durations, so each stage is stretched to the LONGEST
of the three runs; a run that finishes its stage early simply stops plotting
until the next stage (no stretching of its data, just a gap). Stage starts get a
C= marking; the fixed inter-stage cooldown is shaded grey.

  plot_static_vs_smart_timeseries.py --static-8b <dir> --static-32b <dir> \
      --smart <dir> --concurrencies 50,150,... [--bin 15] -o out.png

Each <dir> has per_request_slim.json + harness_stdout.log.
"""
import argparse, datetime, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STAGE_RE = re.compile(r"([0-9-]+ [0-9:,]+).*Stage ([0-9]+) - run (started|completed)")


def windows(stdout_path):
    """[(stage, start, end)] elapsed from stage-0 start."""
    s, e = {}, {}
    for line in open(stdout_path, errors="replace"):
        m = _STAGE_RE.search(line)
        if not m:
            continue
        ep = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        (s if m.group(3) == "started" else e)[int(m.group(2))] = ep
    t0 = s[0]
    return [(n, s[n] - t0, e[n] - t0) for n in sorted(s) if n in e]


def layout(all_wins):
    """Canonical stage-aligned timeline from per-run windows: each stage width is
    the max across runs; cooldown after a stage is the max gap. Returns
    (canon_start[k], width[k], cooldown[k])."""
    nst = min(len(w) for w in all_wins)
    W = [max(w[k][2] - w[k][1] for w in all_wins) for k in range(nst)]
    G = [max(w[k + 1][1] - w[k][2] for w in all_wins) for k in range(nst - 1)] + [0.0]
    canon = [0.0]
    for k in range(nst - 1):
        canon.append(canon[k] + W[k] + G[k])
    return canon, W, G


def run_segments(run_dir, canon, bin_s, metric="latency"):
    """Per-stage binned series mapped onto the canonical timeline. Requests keep
    their real within-stage offset (1:1), so shorter runs just leave a tail gap.
    metric=latency -> e2e latency (s); normlat -> latency/output-token (s/tok).
    Yields (xs, thr, p50, p95) per stage; thr is always req/s."""
    rows = json.load(open(f"{run_dir}/per_request_slim.json"))
    t0 = min(r["t"] for r in rows)
    wins = windows(f"{run_dir}/harness_stdout.log")
    val = (lambda r: r["lat"] / r["ot"]) if metric == "normlat" else (lambda r: r["lat"])
    segs = []
    for k, (_, s, e) in enumerate(wins):
        if k >= len(canon):
            break
        rel = [(r["t"] - t0 - s, val(r)) for r in rows
               if not r["fail"] and s <= r["t"] - t0 <= e and (metric != "normlat" or r["ot"])]
        xs, thr, p50, p95 = [], [], [], []
        for b in range(int((e - s) // bin_s) + 1):
            lat = [l for te, l in rel if b * bin_s <= te < (b + 1) * bin_s]
            if not lat:
                continue
            xs.append(canon[k] + b * bin_s + bin_s / 2)
            thr.append(len(lat) / bin_s)
            p50.append(np.percentile(lat, 50)); p95.append(np.percentile(lat, 95))
        segs.append((np.array(xs), np.array(thr), np.array(p50), np.array(p95)))
    return segs


def mark_stages(ax, canon, W, G, conc):
    for k, x in enumerate(canon):
        ax.axvline(x, color="0.7", lw=.8, ls="--", zorder=0)
        if k < len(conc):
            ax.text(x + 2, ax.get_ylim()[1] * .98, f"C={conc[k]}", fontsize=7,
                    color="dimgray", va="top", rotation=90)
        if G[k]:  # cooldown after this stage
            ax.axvspan(x + W[k], x + W[k] + G[k], color="0.88", zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-8b", required=True)
    ap.add_argument("--static-32b", required=True)
    ap.add_argument("--smart", required=True)
    ap.add_argument("-o", "--output", default="static_vs_smart_timeseries.png")
    ap.add_argument("--bin", type=float, default=15.0, help="time-bin seconds")
    ap.add_argument("--metric", choices=["latency", "normlat"], default="latency",
                    help="latency panel: e2e latency (s) or normalized per output token (s/tok)")
    ap.add_argument("--lat-ymax", type=float, default=None, help="clip latency panel y-axis")
    ap.add_argument("--concurrencies", default="50,150,250,350,450,550,450,350,250,150,50")
    a = ap.parse_args()
    conc = [int(c) for c in a.concurrencies.split(",")]
    runs = [("Static 8B", a.static_8b, "#2ca02c"),
            ("Static 32B", a.static_32b, "#9467bd"),
            ("Smart (pooled)", a.smart, "#d62728")]

    all_wins = [windows(f"{d}/harness_stdout.log") for _, d, _ in runs]
    canon, W, G = layout(all_wins)

    fig, (axt, axl) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for label, d, color in runs:
        first = True
        for xs, thr, p50, p95 in run_segments(d, canon, a.bin, a.metric):
            if not len(xs):
                continue
            axt.plot(xs, thr, "-", color=color, lw=1.3, label=label if first else None)
            axl.plot(xs, p50, "-", color=color, lw=1.3, label=f"{label} p50" if first else None)
            axl.plot(xs, p95, "--", color=color, lw=1.0, alpha=.6, label=f"{label} p95" if first else None)
            first = False

    lat_lab = "normalized latency (s / output token)" if a.metric == "normlat" else "e2e latency (s)"
    axt.set_ylabel("completed throughput (req/s)"); axt.set_ylim(0, None)
    axl.set_ylabel(lat_lab); axl.set_ylim(0, a.lat_ymax)
    axl.set_xlabel("elapsed since start (s) — stages aligned to the longest run per stage; grey = cooldown")
    for ax in (axt, axl):
        mark_stages(ax, canon, W, G, conc)
        ax.grid(alpha=.3); ax.set_xlim(0, canon[-1] + W[-1])
        ax.legend(fontsize=8, ncol=3, loc="upper right")
    axt.set_title(f"Throughput and {lat_lab.split(' (')[0]} over time — static 8B / 32B vs smart routing")
    fig.tight_layout(); fig.savefig(a.output, dpi=130)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
