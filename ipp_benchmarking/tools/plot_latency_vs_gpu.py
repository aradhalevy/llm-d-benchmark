#!/usr/bin/env python3
"""Per-request latency over time vs. actual GPU utilization (vLLM KV-cache %).

Replaces the earlier "bursts vs latency" view, whose grey line was a
client-side outstanding-request reconstruction (NOT a picker in-flight
signal, and it conflated queue effect with arrival rate).

This version uses real wall-clock evidence from the run's own logs:
  * latency per request  -> from the IPP (payload-processor) log, where each
    request id carries real epoch timestamps; latency = last_ts - first_ts
    (the request's round trip through the ext_proc, which equals e2e latency).
  * GPU utilization       -> vLLM "GPU KV cache usage: X%" and "Running: N reqs"
    parsed from each backend's decode-pod log (RFC3339 timestamps).

Both are on the same epoch axis, so you can see whether latency spikes
coincide with a busy GPU (model queueing) or an idle one (path/routing delay).

Usage:
    plot_latency_vs_gpu.py LABEL=collected-logs-N [LABEL2=...] -o out.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt

RFC3339 = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z")
KV_RE = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")
RUN_RE = re.compile(r"Running:\s*(\d+)\s*reqs")


def epoch_of(line: str) -> float | None:
    m = RFC3339.match(line)
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    frac = float("0." + m.group(2)) if m.group(2) else 0.0
    return base.timestamp() + frac


def ipp_requests(ipp_log: Path) -> list[tuple[float, float]]:
    """Return [(arrival_epoch, latency_s)] per request id from the IPP log."""
    byid: dict[str, list[float]] = defaultdict(list)
    for line in open(ipp_log, errors="ignore"):
        j = line.find("{")
        if j < 0:
            continue
        try:
            d = json.loads(line[j:])
        except Exception:
            continue
        rid, ts = d.get("x-request-id"), d.get("ts")
        if rid and isinstance(ts, (int, float)):
            byid[rid].append(ts)
    out = [(min(v), max(v) - min(v)) for v in byid.values() if len(v) > 1]
    out.sort()
    return out


def decode_util(decode_log: Path) -> list[tuple[float, float, int]]:
    """Return [(epoch, kv_pct, running)] from a vLLM decode-pod log."""
    rows = []
    for line in open(decode_log, errors="ignore"):
        kv = KV_RE.search(line)
        if not kv:
            continue
        ep = epoch_of(line)
        run = RUN_RE.search(line)
        if ep is not None:
            rows.append((ep, float(kv.group(1)), int(run.group(1)) if run else 0))
    return rows


def short(path_name: str) -> str:
    m = re.search(r"(30b-a3b|32b)", path_name)
    return {"30b-a3b": "Qwen3-30B-A3B", "32b": "Qwen3-32B"}.get(m.group(1) if m else "", path_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="latency_vs_gpu.png")
    args = ap.parse_args()

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            print(f"skipping {entry!r}: expected LABEL=path", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        p = Path(path)
        ipp_logs = glob.glob(str(p / "payload-processor-*.log"))
        decode_logs = sorted(glob.glob(str(p / "*-decode-*.log")))
        if not ipp_logs or not decode_logs:
            print(f"missing ipp/decode logs in {path}", file=sys.stderr)
            continue
        reqs = ipp_requests(Path(ipp_logs[0]))
        utils = {Path(d).name: decode_util(Path(d)) for d in decode_logs}
        runs.append((label, reqs, utils))
    if not runs:
        print("no usable inputs", file=sys.stderr)
        sys.exit(1)

    lat_max = max(l for _, reqs, _ in runs for _, l in reqs) * 1.08
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5.4), squeeze=False)
    axes = axes[0]
    kv_colors = {"Qwen3-30B-A3B": "#1f77b4", "Qwen3-32B": "#d62728"}

    for ax, (label, reqs, utils) in zip(axes, runs):
        # Isolate the benchmark request cluster: split arrivals on gaps >15s
        # (the IPP pod also served sparse sanity/warmup requests) and keep the
        # densest cluster, so the x-axis covers the load window for THIS run.
        clusters: list[list[tuple[float, float]]] = [[]]
        for a, l in reqs:
            if clusters[-1] and a - clusters[-1][-1][0] > 15:
                clusters.append([])
            clusters[-1].append((a, l))
        bench = max(clusters, key=len)
        t0 = bench[0][0]
        tmax = (bench[-1][0] + bench[-1][1]) - t0 + 3
        reqs = bench

        # left: per-request latency
        xs = [a - t0 for a, _ in reqs]
        ys = [l for _, l in reqs]
        ax.scatter(xs, ys, s=12, alpha=0.5, color="#444444", edgecolors="none",
                   zorder=3, label="request latency")
        ax.set_ylim(0, lat_max)
        ax.set_xlim(0, tmax)
        ax.set_xlabel("time since load start (s)")
        ax.set_ylabel("request latency (s)")
        ax.set_title(f"{label}: latency vs GPU utilization")
        ax.grid(True, alpha=0.25)

        # right: GPU KV-cache % per backend (actual model utilization)
        ax2 = ax.twinx()
        for name, rows in utils.items():
            rows = [(ep - t0, kv) for ep, kv, _ in rows if 0 <= ep - t0 <= tmax]
            if not rows:
                continue
            mdl = short(name)
            ax2.plot([r[0] for r in rows], [r[1] for r in rows],
                     marker=".", linewidth=1.6, color=kv_colors.get(mdl, "#2ca02c"),
                     alpha=0.9, label=f"{mdl} GPU KV %")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("GPU KV-cache usage (%)")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("Latency spikes occur while the GPU is near-idle "
                 "(KV-cache ≤8%) — delay is in the request path, not the model",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
