#!/usr/bin/env python3
"""Routing breakdown from IPP decision lines (deduped by x-request-id) + known planner count.

The planner is pinned to the 32B (single model -> one candidate, curl-verified) and never
routes to the 8B, so with the known planner count:
  summarizer -> 8B  = total -> 8B   (minus manual 8B test curls)
  summarizer -> 32B = total -> 32B  (minus planner, minus manual 32B curls)
  total to BIG (incl planning) = total -> 32B (minus manual 32B curls)

Input may contain duplicate decision lines (rolling-snapshot capture overlaps) -> dedup by
x-request-id. usage: analyze_routing.py <log> <planner_n> [manual_32b] [manual_8b]
"""
import sys, json, re

log = sys.argv[1]
planner_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m32 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
m8 = int(sys.argv[4]) if len(sys.argv) > 4 else 0

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
total = T8 + T32
summ8 = T8 - m8
summ32 = T32 - planner_n - m32
summ = summ8 + summ32
big = T32 - m32

print(f"unique decisions: {total}   (->8B {T8}, ->32B {T32})")
print(f"adjustments: planner={planner_n} (all 32B), manual curls 32B={m32} 8B={m8}\n")
print(f"PLANNER:    {planner_n:5d}  -> 32B 100%  (single-model, pinned)")
if summ > 0:
    print(f"SUMMARIZE:  {summ:5d}  -> 8B={summ8} ({100*summ8/summ:.1f}%)   32B={summ32} ({100*summ32/summ:.1f}%)")
    print(f"\nTOTAL to BIG model (32B) incl. planning: {big} / {big+summ8} ({100*big/(big+summ8):.1f}% of all real requests)")
    print(f"  = planner {planner_n}  +  summarization-overflow {summ32}")

# per-stage summarizer 32B-share: cluster by idle gaps, subtract a pro-rata planner share
rows = sorted((ts[r], sel[r]) for r in sel)
if rows:
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r[0] - prev[0] > 8: stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    big_stages = [s for s in stages if len(s) >= 20]
    print("\nper-stage 32B-share of ALL decisions (incl. steady planner trickle):")
    print(f"  {'stage':5s} {'n':>5s} {'32B':>6s} {'32B%':>6s}")
    for i, s in enumerate(big_stages):
        n = len(s); b = sum(1 for _, mm in s if mm == "32B")
        print(f"  {i:<5d} {n:5d} {b:6d} {100*b/n:5.1f}%")
