#!/usr/bin/env python3
"""Merge the static 8B + static 32B per_request_slim extracts into ONE
"combined static split" run, so it can be plotted as a single workload against
the smart run (deliverable: 8B+32B-as-one-workload vs smart routing).

The two static legs ran at HALF concurrency (25->275) at different wall-clock
times; they correspond stage-for-stage to the smart run's FULL concurrency
(8B@25 + 32B@25 == smart@50). We segment each leg into its N stages by time-gap,
then pool 8B-stage-k with 32B-stage-k and re-base timestamps into one clean
N-stage timeline (latencies untouched -- only `t` is synthesized so the
downstream plotter's gap-based segmentation recovers exactly N stages).

  merge_static_slims.py <static_8b_dir> <static_32b_dir> <out_dir> [--stages 11]
"""
import argparse, json, sys
from pathlib import Path


def cluster(rows, n, gap=8.0, min_size=20):
    rows = sorted(rows, key=lambda r: r["t"])
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["t"] - prev["t"] > gap:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= min_size]
    while len(stages) > n:  # merge smallest inter-stage gaps down to n
        gaps = [stages[i + 1][0]["t"] - stages[i][-1]["t"] for i in range(len(stages) - 1)]
        i = min(range(len(gaps)), key=lambda j: gaps[j])
        stages[i] = stages[i] + stages[i + 1]; del stages[i + 1]
    return stages


def load(d):
    p = Path(d) / "per_request_slim.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("static_8b"); ap.add_argument("static_32b"); ap.add_argument("out")
    ap.add_argument("--stages", type=int, default=11)
    a = ap.parse_args()
    n = a.stages
    s8 = cluster(load(a.static_8b), n)
    s32 = cluster(load(a.static_32b), n)
    if len(s8) != n or len(s32) != n:
        print(f"warn: 8b={len(s8)} 32b={len(s32)} stages vs {n}", file=sys.stderr)
    merged = []
    STAGE_DT = 100.0  # >> gap=8s so stages stay distinct
    for k in range(min(len(s8), len(s32), n)):
        base = k * STAGE_DT
        pool = s8[k] + s32[k]
        for i, r in enumerate(pool):
            t = base + (i / max(len(pool), 1)) * 1.0  # 1s-wide window, < gap
            merged.append({"t": t, "lat": r["lat"], "fail": r["fail"], "m": r["m"],
                           "ot": r.get("ot")})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(merged, open(out / "per_request_slim.json", "w"))
    small = sum(1 for r in merged if r["m"] == "small" and not r["fail"])
    big = sum(1 for r in merged if r["m"] == "big" and not r["fail"])
    fail = sum(1 for r in merged if r["fail"])
    print(f"merged {len(merged)} records -> {out}/per_request_slim.json  "
          f"8B={small} 32B={big} fail={fail} stages={min(len(s8),len(s32))}")


if __name__ == "__main__":
    main()
