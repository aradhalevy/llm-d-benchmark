#!/usr/bin/env python3
"""Summarizer routing breakdown from IPP "Model selected" decision lines.

Single-model arms send everything to their one model; the smart arm splits 8B/32B.
Input may contain duplicate lines (rolling-snapshot capture overlaps) -> dedup by
x-request-id.  usage: analyze_routing.py <log>
"""
import sys, json, re

log = sys.argv[1]
sel, ts = {}, {}
for line in open(log, errors="ignore"):
    if '"msg":"Model selected"' not in line:
        continue
    try: d = json.loads(line)
    except: continue
    rid = d.get("x-request-id")
    m = "32B" if "Qwen3-32B" in line else ("8B" if "Qwen3-8B" in line else None)
    if not rid or not m: continue
    sel[rid] = m
    mt = re.search(r'"ts":([0-9.]+)', line)
    ts[rid] = float(mt.group(1)) if mt else 0.0

T8 = sum(1 for m in sel.values() if m == "8B")
T32 = sum(1 for m in sel.values() if m == "32B")
tot = T8 + T32 or 1
print(f"summarizer decisions: {T8+T32}   8B={T8} ({100*T8/tot:.1f}%)   32B={T32} ({100*T32/tot:.1f}%)")

# per-stage split: cluster by idle gaps
rows = sorted((ts[r], sel[r]) for r in sel)
if rows:
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r[0] - prev[0] > 8: stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    print("\nper-stage 8B/32B split:")
    print(f"  {'stage':>5s} {'n':>6s} {'8B':>6s} {'8B%':>6s} {'32B':>6s} {'32B%':>6s}")
    for i, s in enumerate([s for s in stages if len(s) >= 20]):
        n = len(s); b = sum(1 for _, mm in s if mm == "32B"); e = n - b
        print(f"  {i:>5d} {n:6d} {e:6d} {100*e/n:5.1f}% {b:6d} {100*b/n:5.1f}%")
