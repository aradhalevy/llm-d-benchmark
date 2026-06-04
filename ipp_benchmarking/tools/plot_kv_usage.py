#!/usr/bin/env python3
"""Plot GPU KV-cache usage (and queue depth) over time per backend, per run.

Reads each run's collected decode-pod logs (`*-decode-*.log`), which carry
vLLM's periodic `Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%`
lines with RFC3339 timestamps. Draws KV% over time for each backend, one
panel per run, so saturation (KV pegged near 100% with a deep Waiting queue)
is directly visible and comparable across runs (e.g. random vs smart).

Usage:
    plot_kv_usage.py random=collected-logs-43 smart=collected-logs-44 -o kv_usage.png
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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt

RFC3339 = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z")
KV = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")
WAIT = re.compile(r"Waiting:\s*(\d+)\s*reqs")
COLORS = {"Qwen3-30B-A3B": "#1f77b4", "Qwen3-32B": "#d62728"}


def epoch(line: str):
    m = RFC3339.match(line)
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return base.timestamp() + (float("0." + m.group(2)) if m.group(2) else 0.0)


def backend(fname: str) -> str:
    m = re.search(r"(30b-a3b|32b)", fname)
    return {"30b-a3b": "Qwen3-30B-A3B", "32b": "Qwen3-32B"}.get(m.group(1) if m else "", fname)


def ipp_window(run_dir: Path):
    """Derive (start_epoch, end_epoch) for THIS run from the IPP log's last
    dense request cluster (real epoch `ts`), so KV samples can be windowed even
    when the decode pod served several runs. Returns None if unavailable."""
    logs = glob.glob(str(run_dir / "payload-processor-*.log"))
    if not logs:
        return None
    ts = []
    for line in open(logs[0], errors="ignore"):
        j = line.find('{"level"')
        if j < 0:
            continue
        try:
            d = json.loads(line[j:])
        except Exception:
            continue
        t = d.get("ts")
        if isinstance(t, (int, float)) and d.get("x-request-id"):
            ts.append(t)
    if not ts:
        return None
    ts.sort()
    # last cluster of requests separated by >30s gaps = the most recent run
    start = ts[0]
    clusters = [[ts[0]]]
    for t in ts[1:]:
        if t - clusters[-1][-1] > 30:
            clusters.append([])
        clusters[-1].append(t)
    win = max(clusters, key=len) if len(clusters[-1]) < 20 else clusters[-1]
    return win[0] - 3, win[-1] + 65  # pad start; extend for post-load drain


def series(decode_log: Path):
    """Return [(epoch, kv%, waiting)] from a decode log."""
    rows = []
    for line in open(decode_log, errors="ignore"):
        k = KV.search(line)
        if not k:
            continue
        e = epoch(line)
        w = WAIT.search(line)
        if e is not None:
            rows.append((e, float(k.group(1)), int(w.group(1)) if w else 0))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="kv_usage.png")
    ap.add_argument("--show-queue", action="store_true",
                    help="Overlay Waiting-queue depth on a right axis.")
    args = ap.parse_args()

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            print(f"skipping {entry!r}", file=sys.stderr)
            continue
        label, path = entry.split("=", 1)
        decode = sorted(glob.glob(str(Path(path) / "*-decode-*.log")))
        data = {}
        for d in decode:
            rows = series(Path(d))
            # keep only samples where the GPU was active (load window), trim idle tails
            active = [r for r in rows if r[1] > 0 or r[2] > 0]
            if active:
                data[backend(os.path.basename(d))] = (rows, active)
        if data:
            runs.append((label, data, Path(path)))
    if not runs:
        print("no decode KV data found", file=sys.stderr)
        sys.exit(1)

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 5.0), squeeze=False)
    axes = axes[0]
    for ax, (label, data, run_dir) in zip(axes, runs):
        ippwin = ipp_window(run_dir)
        # The decode pods may have served several runs; isolate the LAST active
        # cluster (split all active samples on >90s idle gaps, keep the latest)
        # so each panel shows only this run's window.
        if ippwin is not None:
            t0, tend = ippwin
        else:
            # fallback: last substantial active cluster from the decode samples
            all_active = sorted(e for _, (rows, a) in data.items() for e, kv, w in a)
            clusters = [[all_active[0]]]
            for e in all_active[1:]:
                if e - clusters[-1][-1] > 90:
                    clusters.append([])
                clusters[-1].append(e)
            substantial = [c for c in clusters if len(c) >= 5]
            win = substantial[-1] if substantial else max(clusters, key=len)
            t0, tend = win[0], win[-1]
        tmax = tend - t0
        ax2 = ax.twinx() if args.show_queue else None
        for mdl, (rows, active) in data.items():
            xs = [e - t0 for e, kv, w in rows if 0 <= e - t0 <= tmax + 5]
            kvs = [kv for e, kv, w in rows if 0 <= e - t0 <= tmax + 5]
            ax.plot(xs, kvs, marker=".", linewidth=1.8, color=COLORS.get(mdl, "#2ca02c"),
                    label=f"{mdl} KV%")
            if ax2 is not None:
                ws = [w for e, kv, w in rows if 0 <= e - t0 <= tmax + 5]
                ax2.plot(xs, ws, linestyle="--", linewidth=1.0, alpha=0.5,
                         color=COLORS.get(mdl, "#2ca02c"))
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(tmax, 1))
        ax.set_xlabel("time since load start (s)")
        ax.set_ylabel("GPU KV-cache usage (%)")
        ax.set_title(f"{label}: GPU KV-cache usage")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
        if ax2 is not None:
            ax2.set_ylabel("Waiting queue (reqs)")
    fig.suptitle("GPU KV-cache usage per backend (saturation = pegged near 100%)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
