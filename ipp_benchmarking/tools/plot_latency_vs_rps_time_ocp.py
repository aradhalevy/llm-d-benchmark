#!/usr/bin/env python3
"""Latency vs concurrency (TOP) + actual RPS sent vs time (BOTTOM) — OCP A/B.

Same top panel as plot_latency_vs_concurrency_ocp.py (e2e p50/p95 per backend with
the separated 30s-timeout band), except timeouts are INCLUDED in the percentile and
attributed to their routed backend. The BOTTOM panel changes: instead of request
volume binned by concurrency level, it plots the *actual requests-per-second sent*
over time since load start. Send time is just slim `t` (= start_time); no t-lat
reconstruction. Each ramp stage is annotated with its concurrent-client count, and
the idle gap between stages is labeled with its MEASURED width (drain + the
configured `interval` cooldown), which runs well past the bare cooldown.

    plot_latency_vs_rps_time_ocp.py "smart router"=.../smart "random"=.../random \\
        --concurrencies 100,200,300,400,500 --cooldown 15 -o out.png
"""
from __future__ import annotations

import argparse
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
COLOR = {SMALL: "#2ca02c", BIG: "#9467bd"}
TMO_COLOR = "#d62728"                          # red, failed/timeout
PCTS = [("p50", 50, "-"), ("p95", 95, (0, (5, 3)))]
CEIL = 30.0
BAND_ORDER = [(SMALL, "p50"), (SMALL, "p95"), (BIG, "p50"), (BIG, "p95")]
GAP, SPACING = 1.5, 1.15


def load_routed_models(run: Path):
    """Ordered routed-model decisions from the IPP log (deduped by x-request-id,
    earliest ts). Each summarizer request's *routed* model — known even for the ones
    that later time out, unlike the served model which is empty on a 504."""
    log = run / "ipp-full-live.log"
    if not log.exists():
        log = run / "ipp-decisions.log"
    if not log.exists():
        return []
    seen = {}
    for line in open(log, errors="ignore"):
        if '"msg":"Model selected"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid, m = d.get("x-request-id"), d.get("model")
        if rid and m and rid not in seen:
            seen[rid] = (d["ts"], BIG if "32B" in m else SMALL)
    return [m for _, m in sorted(seen.values())]


def load_rows(run: Path):
    slim = run / "per_request_slim.json"
    if not slim.exists():
        sys.exit(f"missing {slim} — run extract_per_request_slim.py first")
    # NB: slim `t` is start_time = the SEND time (lat = end - start), so `t` is what to
    # bin for RPS-sent and to order stages by; no `t - lat` reconstruction needed.
    rows = [{"t": r["t"], "lat": r["lat"], "fail": r["fail"], "m": r["m"]}
            for r in json.load(open(slim))]
    rows.sort(key=lambda x: x["t"])
    return rows


def stage_model_latencies(stage, routed_32b):
    """e2e latencies per *routed* backend for one ramp stage, INCLUDING timeouts.
    Successes carry their served model (routed==served when a response comes back), so
    they're attributed exactly. Timeouts have no served model, so they're split by the
    per-stage deficit: routed_32b - served_32b_successes = 32B's timeouts (the rest are
    8B's). A timed-out request's latency (~31s) is a real e2e outcome and counts in the
    percentile — that's what lifts a saturated backend's line into the timeout band."""
    succ = {SMALL: [], BIG: []}
    fail_lats = []
    for r in stage:
        if r["fail"]:
            fail_lats.append(r["lat"])
        elif r["m"] in succ:
            succ[r["m"]].append(r["lat"])
    f32 = min(max(routed_32b - len(succ[BIG]), 0), len(fail_lats))   # 32B's timeouts
    # WHICH specific timeout goes to which backend is arbitrary here (fail_lats isn't
    # sorted), but harmless: every timeout latency is >30s, so it lands in the band
    # regardless of partition. Only the per-backend timeout COUNT matters, and that's
    # what f32 fixes. (Would matter if timeout latencies ever straddled the 30s line.)
    return {SMALL: succ[SMALL] + fail_lats[f32:],
            BIG:   succ[BIG] + fail_lats[:f32]}


def split_by_count(rows, counts, key="t"):
    """Split requests (sorted by `key`) into ramp stages by known per-stage request
    counts — exact and robust, unlike re-clustering on send-time gaps which merges
    adjacent stages whenever the cooldown isn't a clean lull in the send stream."""
    rs = sorted(rows, key=lambda r: r[key])
    stages, i = [], 0
    for c in counts:
        stages.append(rs[i:i + c]); i += c
    if i < len(rs):                       # leftover (count mismatch) -> append to last
        stages[-1] += rs[i:]
    return [s for s in stages if s]


def annotate_stages(ax, sub, conc, rps_max, cooldown):
    """Shade each ramp stage, label it with its concurrency, and mark the idle gap
    (drain + cooldown) between stages with its measured width."""
    for i, st in enumerate(sub):
        s0, s1 = st[0]["rel"], st[-1]["rel"]
        ax.axvspan(s0, s1, color="#888", alpha=0.06, zorder=0)
        ax.text((s0 + s1) / 2, rps_max * 0.93, f"C={conc[i]}", ha="center", va="top",
                fontsize=9, fontweight="bold", color="#333")
        if i:  # the idle gap between stages
            p1 = sub[i - 1][-1]["rel"]
            ax.axvspan(p1, s0, color=COLOR[BIG], alpha=0.10, zorder=0)
            ax.annotate("cooldown", ((p1 + s0) / 2, rps_max * 0.45),
                        ha="center", va="center", fontsize=7, color="#6a3d9a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=path entries")
    ap.add_argument("-o", "--output", default="latency_vs_rps_time_ocp.png")
    ap.add_argument("--concurrencies", default="100,200,300,400,500")
    ap.add_argument("--requests", default="400,800,1200,1600,2000",
                    help="num_requests per stage (from the workload profile)")
    ap.add_argument("--cooldown", type=float, default=15.0,
                    help="configured inter-stage cooldown (s) for the annotation note")
    ap.add_argument("--bin", type=float, default=2.0, help="RPS bin width (s)")
    ap.add_argument("--no-total-rps", action="store_true",
                    help="drop the middle total-RPS panel; keep only latency + by-outcome RPS")
    args = ap.parse_args()
    conc = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    reqs = [int(c) for c in args.requests.split(",") if c.strip()]

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        rows = load_rows(Path(path))
        t0 = min(r["t"] for r in rows)                 # load start = first send
        for r in rows:
            r["rel"] = r["t"] - t0                     # send time since load start
        stages = split_by_count(rows, reqs, key="t")   # ramp stages, by send order
        # routed-32B count per stage: the decision stream is already ts-ordered (same
        # request sequence as send order), so slice it by the same cumulative counts.
        # There are usually a few extra decisions (warmup/planner) *before* load start;
        # drop that leading offset so the per-stage slices line up with the requests
        # (otherwise every stage is shifted by a constant ~30, per the reviewer's MEDIUM).
        routed = load_routed_models(Path(path))
        off = max(len(routed) - sum(reqs), 0)
        r32, j = [], off
        for c in reqs:
            r32.append(sum(m == BIG for m in routed[j:j + c]) if routed else 0)
            j += c
        runs.append((label, rows, stages, r32))
    if not runs:
        sys.exit("no runs")

    level = {key: CEIL + GAP + i * SPACING for i, key in enumerate(BAND_ORDER)}
    y_top = CEIL + GAP + (len(BAND_ORDER) - 1) * SPACING + 1.0
    rps_max = 0.0
    rel_max = 0.0
    for _, rows, _, _ in runs:
        rel = np.array([r["rel"] for r in rows])
        rel_max = max(rel_max, rel.max())
        edges = np.arange(0, rel.max() + args.bin, args.bin)
        rps_max = max(rps_max, (np.histogram(rel, bins=edges)[0] / args.bin).max())
    rps_max *= 1.12

    nr = len(runs)
    heights = [2.0, 2.0] if args.no_total_rps else [2.3, 1.1, 1.1]
    fig, axes = plt.subplots(len(heights), nr, figsize=(7.4 * nr, 4.0 + 1.8 * len(heights)),
                             squeeze=False, gridspec_kw={"height_ratios": heights})

    for col, (label, rows, stages, r32) in enumerate(runs):
        axL = axes[0][col]
        axB = None if args.no_total_rps else axes[1][col]
        axS = axes[-1][col]
        sub = stages[:len(conc)]
        xs = conc[:len(sub)]
        lat_by_model = [stage_model_latencies(st, r32[i]) for i, st in enumerate(sub)]

        # ---- TOP: e2e latency p50/p95 per ROUTED backend, INCLUDING timeouts ----
        # A timeout is a real ~31s e2e outcome, so it counts in the percentile. Requests
        # are bucketed by the model they were *routed* to (decision log), not served, so
        # timed-out requests (no served model) still land on their backend. When a
        # percentile is a timeout it sits in the separated band above the 30s line.
        axL.axhspan(CEIL + GAP, y_top, color="#eee", zorder=0)
        axL.axhline(CEIL, color="#333", lw=1.4, zorder=2)
        axL.text(xs[-1], CEIL, "30s timeout ", color="#333", fontsize=8, va="bottom", ha="right", zorder=3)
        for model in (SMALL, BIG):
            for name, q, ls in PCTS:
                y = []
                for lm in lat_by_model:
                    lat = lm[model]                                    # incl. timeouts
                    v = np.percentile(lat, q) if lat else np.nan
                    y.append(level[(model, name)] if (not np.isnan(v) and v > CEIL) else v)
                axL.plot(xs, y, ls=ls, color=COLOR[model], lw=2.3, marker="o", ms=4, zorder=5)
        axL.set_ylim(0, y_top)
        axL.set_yticks([0, 5, 10, 15, 20, 25, 30])
        axL.set_ylabel("e2e latency (s)  —  band = timeout")
        axL.set_xlabel("concurrent clients")
        axL.set_xticks(conc)
        axL.set_title(label, fontsize=11)
        axL.grid(True, axis="y", alpha=0.2)
        model_h = [Line2D([0], [0], color=COLOR[m], lw=2.4, label=MODEL_NAME[m]) for m in (SMALL, BIG)]
        style_h = [Line2D([0], [0], color="#444", lw=2.0, ls=ls, label=name) for name, _, ls in PCTS]
        leg1 = axL.legend(handles=model_h, loc="upper left", fontsize=8.5, framealpha=0.92)
        axL.add_artist(leg1)
        axL.legend(handles=style_h, loc="upper left", bbox_to_anchor=(0.0, 0.78),
                   fontsize=8.5, framealpha=0.92, title="percentile (routed)")

        # ---- MIDDLE (optional): total RPS sent vs time, w/ per-stage conc + cooldown ----
        if axB is not None:
            rel = np.array([r["rel"] for r in rows])
            edges = np.arange(0, rel.max() + args.bin, args.bin)
            rps = np.histogram(rel, bins=edges)[0] / args.bin
            centers = edges[:-1] + args.bin / 2
            axB.fill_between(centers, rps, step="mid", color="#4c72b0", alpha=0.25, zorder=1)
            axB.step(centers, rps, where="mid", color="#1f4e79", lw=1.6, zorder=2)
            annotate_stages(axB, sub, conc, rps_max, args.cooldown)
            axB.set_ylim(0, rps_max)
            axB.set_xlim(0, rel_max * 1.01)
            axB.set_ylabel("requests sent / s")
            axB.grid(True, axis="y", alpha=0.2)
            axB.tick_params(labelbottom=False)
            if col == 0:
                axB.legend(handles=[Line2D([0], [0], color="#1f4e79", lw=1.8, label="RPS sent (total)"),
                                    Line2D([0], [0], color=COLOR[BIG], lw=6, alpha=0.3,
                                           label=f"inter-stage idle (drain + {args.cooldown:.0f}s cooldown)")],
                           loc="upper left", fontsize=8, framealpha=0.92)

        # ---- BOTTOM: same rate, decomposed by OUTCOME — 8B-served / 32B-served / timed
        # out — as a stacked area (stack total == the sent rate above). Served bins use
        # the ground-truth served model; timeouts (no served model) are one combined red
        # band. All binned by send time, so a request shows where it was submitted.
        edges = np.arange(0, rel_max + args.bin, args.bin)
        centers = edges[:-1] + args.bin / 2
        rel_s = np.array([r["rel"] for r in rows if r["m"] == SMALL and not r["fail"]])
        rel_b = np.array([r["rel"] for r in rows if r["m"] == BIG and not r["fail"]])
        rel_f = np.array([r["rel"] for r in rows if r["fail"]])
        h_s = np.histogram(rel_s, bins=edges)[0] / args.bin
        h_b = np.histogram(rel_b, bins=edges)[0] / args.bin
        h_f = np.histogram(rel_f, bins=edges)[0] / args.bin
        axS.stackplot(centers, h_s, h_b, h_f,
                      colors=[COLOR[SMALL], COLOR[BIG], TMO_COLOR], step="mid", alpha=0.85)
        if axB is None:        # no middle panel -> carry the stage/cooldown labels here
            annotate_stages(axS, sub, conc, rps_max, args.cooldown)
        axS.set_ylim(0, rps_max)
        axS.set_xlim(0, rel_max * 1.01)
        axS.set_xlabel("time since load start (s)")
        axS.set_ylabel("requests / s  (by outcome)")
        axS.grid(True, axis="y", alpha=0.2)
        if col == 0:
            axS.legend(handles=[Patch(color=COLOR[SMALL], label="8B served"),
                                Patch(color=COLOR[BIG], label="32B served"),
                                Patch(color=TMO_COLOR, label="timed out")],
                       loc="upper left", fontsize=8, framealpha=0.92)

    fig.suptitle("Summarizer e2e latency vs concurrency  +  RPS sent over time (total & by outcome)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
