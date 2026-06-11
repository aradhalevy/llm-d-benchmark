#!/usr/bin/env python3
"""Per-stage model routing split from IPP logs.

Buckets the IPP "Model selected" log lines into benchmark stages and prints the
per-stage selection count for each model, so the routing behavior of the
model-selector (scorer overflow, pinning) can be checked against expectations.

Usage:
  kubectl logs <ipp-pod> -n <ns> --since=30m | \
      routing_split_by_stage.py --start <epoch> --stages 60,60,60,60,60,60

  --start   epoch seconds when stage 1 began (defaults to first selection seen)
  --stages  comma-separated stage durations in seconds
"""
import argparse
import json
import re
import sys
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=float, default=None)
parser.add_argument("--stages", default="60,60,60,60,60,60")
args = parser.parse_args()

durations = [float(d) for d in args.stages.split(",")]
selections = []  # (ts, model)
for line in sys.stdin:
    if '"Model selected"' not in line:
        continue
    try:
        rec = json.loads(line[line.index("{"):])
        selections.append((rec["ts"], rec["model"]))
    except (ValueError, KeyError):
        m = re.search(r'"ts":([0-9.]+).*"model":"([^"]+)"', line)
        if m:
            selections.append((float(m.group(1)), m.group(2)))

if not selections:
    sys.exit("no 'Model selected' lines found on stdin")

start = args.start if args.start is not None else selections[0][0]
bounds = []
t = start
for d in durations:
    bounds.append((t, t + d))
    t += d

print(f"{'stage':>5} {'window':>13} {'total':>6}  per-model")
for i, (lo, hi) in enumerate(bounds, 1):
    in_stage = [m for ts, m in selections if lo <= ts < hi]
    counts = Counter(in_stage)
    split = "  ".join(f"{m.split('/')[-1]}={c}" for m, c in sorted(counts.items()))
    print(f"{i:>5} {int(lo - start):>5}s-{int(hi - start):>4}s {len(in_stage):>6}  {split}")

after = [m for ts, m in selections if ts >= bounds[-1][1]]
if after:
    counts = Counter(after)
    split = "  ".join(f"{m.split('/')[-1]}={c}" for m, c in sorted(counts.items()))
    print(f"{'after':>5} {'':>13} {len(after):>6}  {split}")
