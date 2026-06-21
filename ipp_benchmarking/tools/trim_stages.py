#!/usr/bin/env python3
"""Trim the first/last 10% (by TIME) of each stage from a per_request_slim.json.

Stages are recovered by COUNT, not idle-gap clustering: the sweep sends exactly
`--per-stage` requests per stage, sequentially (stage k+1 starts only after stage
k drains), so sorting by start_time and chunking into N groups is exact. Idle-gap
clustering is WRONG here because the saturated high-C stages stall mid-stage with
multi-minute zero-arrival gaps that would falsely split a stage.

Trimming removes the slot-fill ramp + drain tail + cold-EMA transient, leaving the
steady-state middle 80% for clean per-stage latency/routing stats.

    trim_stages.py <slim.json> <per_stage> <out_trimmed.json> [trim_frac=0.1]
    trim_stages.py --selfcheck
"""
import json, sys


def trim(records, per_stage, frac=0.1):
    recs = sorted(records, key=lambda r: r["t"])
    out, stages = [], []
    for i in range(0, len(recs), per_stage):
        chunk = recs[i:i + per_stage]
        if not chunk:
            continue
        t0, t1 = chunk[0]["t"], chunk[-1]["t"]
        span = t1 - t0
        lo, hi = t0 + frac * span, t1 - frac * span
        kept = [r for r in chunk if lo <= r["t"] <= hi]
        out.extend(kept)
        stages.append((len(chunk), len(kept)))
    return out, stages


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        # 2 stages of 10 evenly spaced in time; 10% trim drops the edges of each
        recs = [{"t": float(s * 100 + i), "lat": 1.0, "fail": False, "m": "small"}
                for s in range(2) for i in range(10)]
        out, stages = trim(recs, 10, 0.1)
        assert stages == [(10, 8), (10, 8)], stages          # each stage keeps middle 8
        ts = [r["t"] for r in out]
        assert min(ts) == 1.0 and 100.0 not in ts, ts        # stage0 t=0 trimmed, stage1 t=100 trimmed
        print("selfcheck ok:", stages)
        sys.exit(0)

    slim, per_stage, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    recs = json.load(open(slim))
    out, stages = trim(recs, per_stage, frac)
    json.dump(out, open(outp, "w"))
    print(f"{len(recs)} -> {len(out)} records ({frac:.0%} time-edge trim/stage)")
    for k, (n, kept) in enumerate(stages):
        print(f"  stage {k}: {n} -> {kept}")
