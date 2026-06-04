#!/usr/bin/env python3
"""Measured GPU compute utilization per backend, from DCGM via Prometheus/Thanos.

The NVIDIA GPU Operator's DCGM exporter is scraped into OpenShift monitoring;
the profiling counters are attributed to each decode pod's GPU. This pulls them
for a run's time window and reports, per backend (MoE vs dense):

  * DCGM_FI_PROF_PIPE_TENSOR_ACTIVE - fraction of time the Tensor Cores are
    issuing work. This is the hardware proxy for "are we using the FLOP units"
    (a duty cycle: active != necessarily at peak FLOPs, so read it next to the
    computed MFU from compute_mfu.py).
  * DCGM_FI_PROF_SM_ACTIVE   - fraction of time SMs have a warp resident.
  * DCGM_FI_PROF_DRAM_ACTIVE - HBM bandwidth duty cycle (decode is memory-bound).
  * DCGM_FI_DEV_GPU_UTIL     - coarse "any kernel running" %.

The run window is derived from the collected decode-pod logs' RFC3339 timestamps
(KV-active samples), so it lines up with the actual load. Override with
--start/--end (unix epoch) if needed.

Auth: uses `oc whoami -t` for the token and the thanos-querier route by default;
override with env THANOS_HOST / OC_TOKEN.

Usage:
    query_gpu_utilization.py smart=collected-logs-45 -o gpu_util.png
    query_gpu_utilization.py random=collected-logs-43 smart=collected-logs-44 -o gpu_util.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

RFC3339 = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z")
KV = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")

MOE = "Qwen3-30B-A3B"
DENSE = "Qwen3-32B"
COLORS = {MOE: "#1f77b4", DENSE: "#d62728"}
# pod-name regex per backend (decode pod carries the model short name)
POD_RE = {MOE: "30b-a3b.*decode", DENSE: "32b.*decode"}

METRICS = [
    ("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "Tensor-core active"),
    ("DCGM_FI_PROF_SM_ACTIVE", "SM active"),
    ("DCGM_FI_PROF_DRAM_ACTIVE", "DRAM (HBM) active"),
    ("DCGM_FI_DEV_GPU_UTIL", "GPU util"),
]


def oc(*args: str) -> str:
    return subprocess.run(["oc", *args], capture_output=True, text=True).stdout.strip()


def epoch(line: str):
    m = RFC3339.search(line)
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return base.timestamp() + (float("0." + m.group(2)) if m.group(2) else 0.0)


def window_from_decode_logs(run: Path):
    """[t0, t1] unix-epoch (UTC) when the decode pods were actively serving."""
    active = []
    for d in glob.glob(str(run / "*-decode-*.log")):
        for line in open(d, errors="ignore"):
            k = KV.search(line)
            if k and float(k.group(1)) > 0:
                e = epoch(line)
                if e:
                    active.append(e)
    if not active:
        return None
    return min(active), max(active)


def query_range(host, token, query, start, end, step=10):
    params = urllib.parse.urlencode({"query": query, "start": int(start), "end": int(end), "step": step})
    url = f"https://{host}/api/v1/query_range?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        return json.load(r)["data"]["result"]


def series(host, token, metric, pod_re, start, end, step):
    q = f'{metric}{{exported_namespace="llm-d-arad",exported_pod=~"qwen.*{pod_re}.*"}}'
    res = query_range(host, token, q, start, end, step)
    if not res:  # some clusters label the pod as `pod` not `exported_pod`
        q = f'{metric}{{namespace="llm-d-arad",pod=~"qwen.*{pod_re}.*"}}'
        res = query_range(host, token, q, start, end, step)
    if not res:
        return np.array([]), np.array([])
    vals = res[0]["values"]
    t = np.array([float(v[0]) for v in vals])
    y = np.array([float(v[1]) for v in vals])
    return t, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=collected-logs-N entries")
    ap.add_argument("-o", "--output", default="gpu_util.png")
    ap.add_argument("--start", type=float, default=None, help="override window start (unix epoch)")
    ap.add_argument("--end", type=float, default=None, help="override window end (unix epoch)")
    ap.add_argument("--step", type=int, default=10, help="query step seconds")
    ap.add_argument("--pad", type=float, default=20.0, help="seconds of pad around the window")
    args = ap.parse_args()

    host = os.environ.get("THANOS_HOST") or oc("get", "route", "thanos-querier",
                                               "-n", "openshift-monitoring", "-o",
                                               "jsonpath={.spec.host}")
    token = os.environ.get("OC_TOKEN") or oc("whoami", "-t")
    if not host or not token:
        print("could not resolve Thanos host / token (set THANOS_HOST, OC_TOKEN)", file=sys.stderr)
        sys.exit(1)

    runs = []
    for entry in args.inputs:
        if "=" not in entry:
            continue
        label, path = entry.split("=", 1)
        run = Path(path)
        if args.start and args.end:
            win = (args.start, args.end)
        else:
            win = window_from_decode_logs(run)
        if not win:
            print(f"no decode-log window for {path}; pass --start/--end", file=sys.stderr)
            continue
        runs.append((label, run, win))
    if not runs:
        sys.exit(1)

    # one row per metric, one column per run
    nrow, ncol = len(METRICS), len(runs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 2.6 * nrow), squeeze=False)

    for col, (label, run, (t0, t1)) in enumerate(runs):
        s, e = t0 - args.pad, t1 + args.pad
        print(f"\n=== {label}  window={dt.datetime.utcfromtimestamp(t0):%H:%M:%S}–"
              f"{dt.datetime.utcfromtimestamp(t1):%H:%M:%S} UTC ({t1-t0:.0f}s) ===")
        for row, (metric, nice) in enumerate(METRICS):
            ax = axes[row][col]
            scale = 1.0 if metric == "DCGM_FI_DEV_GPU_UTIL" else 100.0  # PROF_* are 0..1
            for m in (MOE, DENSE):
                t, y = series(host, token, metric, POD_RE[m], s, e, args.step)
                if t.size == 0:
                    continue
                y = y * scale
                ax.plot(t - t0, y, color=COLORS[m], linewidth=1.4, label=short(m))
                busy = y[y > 1.0]
                print(f"  {nice:18s} {short(m):14s} mean(busy)={busy.mean() if busy.size else 0:5.1f}%  "
                      f"peak={y.max() if y.size else 0:5.1f}%")
            ax.set_ylim(0, 105)
            ax.set_ylabel(f"{nice}\n(% )", fontsize=8)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(f"{label}", fontsize=11)
                ax.legend(loc="upper right", fontsize=8)
            if row == nrow - 1:
                ax.set_xlabel("time since load start (s)")

    fig.suptitle("Measured GPU utilization per backend (DCGM) — tensor-core active = FLOP-unit duty cycle",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=130)
    print(f"\nwrote {args.output}")


def short(m: str) -> str:
    return m.rsplit("/", 1)[-1]


if __name__ == "__main__":
    main()
