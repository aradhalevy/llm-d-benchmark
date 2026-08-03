#!/usr/bin/env python3
"""Per-stage wall-clock timeline: smart routing vs static 8B vs static 32B.

  plot_stage_timeline.py <smart_stdout> <static8b_stdout> <static32b_stdout> \
      -o out.png [--concurrencies 50,150,...]

Each stage's slot width is the LONGEST of the three runs' stage durations (from
the inference-perf harness "Stage N - run started/completed" markers). Every run
draws a bar of its own duration inside the slot; the faster runs finish early and
get a marker with the seconds saved vs the slowest.
"""
import argparse, datetime, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STAGE_RE = re.compile(r"([0-9-]+ [0-9:,]+).*Stage ([0-9]+) - run (started|completed)")
RUNS = [("static 8B", "#1f77b4"), ("static 32B", "#9467bd"), ("smart routing", "#2ca02c")]


def durations(path):
    """{stage: duration_s} from the harness stdout; stages missing a completion
    marker (truncated log) are skipped."""
    starts, ends = {}, {}
    for line in open(path, errors="replace"):
        m = _STAGE_RE.search(line)
        if not m:
            continue
        ep = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        (starts if m.group(3) == "started" else ends)[int(m.group(2))] = ep
    return {s: ends[s] - starts[s] for s in starts if s in ends}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smart"); ap.add_argument("static8b"); ap.add_argument("static32b")
    ap.add_argument("-o", "--output", default="stage_timeline.png")
    ap.add_argument("--concurrencies", default="50,150,250,350,450,550,450,350,250,150,50")
    a = ap.parse_args()
    conc = [int(c) for c in a.concurrencies.split(",")]

    d8, d32, dsm = durations(a.static8b), durations(a.static32b), durations(a.smart)
    per_run = [d8, d32, dsm]
    stages = sorted(set(d8) | set(d32) | set(dsm))

    fig, ax = plt.subplots(figsize=(14, 6))
    lane_h, gap = 0.24, 0.04
    cum = 0.0
    for st in stages:
        durs = [d.get(st) for d in per_run]
        slot = max(x for x in durs if x is not None)
        for i, (dur, (label, color)) in enumerate(zip(durs, RUNS)):
            y = -(i * (lane_h + gap))
            if dur is None:
                continue
            ax.barh(y, dur, left=cum, height=lane_h, color=color,
                    edgecolor="white", label=label if st == stages[0] else None)
            if dur < slot - 1:  # finished early
                ax.plot(cum + dur, y, "|", color="black", ms=10, mew=1.5)
                ax.text(cum + dur + slot * 0.01, y, f"-{slot - dur:.0f}s",
                        va="center", ha="left", fontsize=7, color="black")
        ax.axvline(cum, color="0.7", lw=0.6, ls="--")
        ax.text(cum + slot / 2, lane_h + 0.06, f"C={conc[st] if st < len(conc) else '?'}",
                ha="center", va="bottom", fontsize=8)
        cum += slot
    ax.axvline(cum, color="0.7", lw=0.6, ls="--")

    ax.set_yticks([-(i * (lane_h + gap)) for i in range(len(RUNS))])
    ax.set_yticklabels([r[0] for r in RUNS])
    ax.set_xlabel("wall-clock time (s) — each stage sized to the slowest run")
    ax.set_title("Per-stage runtime: smart routing vs static 8B vs static 32B")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(a.output, dpi=130)
    print("wrote", a.output)
    tot = {lbl: sum(d.values()) for (lbl, _), d in zip(RUNS, per_run)}
    for lbl, t in tot.items():
        print(f"{lbl:14s} total stage time: {t:7.0f}s")
    print(f"{'slowest-per-stage':14s} total slot time: {cum:7.0f}s")


if __name__ == "__main__":
    main()
