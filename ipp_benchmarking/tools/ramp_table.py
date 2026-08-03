#!/usr/bin/env python3
"""Print the rate-vs-latency table for a multi-stage inference-perf run (saturation ramp).

Usage: ramp_table.py <dir containing stage_N_lifecycle_metrics.json>

Saturation shows up as achieved_rate falling below requested, throughput plateauing while the
requested rate climbs, or e2e latency growing superlinearly.
"""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
files = sorted(d.glob("stage_*_lifecycle_metrics.json"),
               key=lambda p: int(re.search(r"stage_(\d+)", p.name).group(1)))
if not files:
    sys.exit(f"no stage_*_lifecycle_metrics.json in {d}")

print(f"{'rate':>6} {'achv':>6} {'rps_out':>8} {'ok':>6} {'fail':>5} "
      f"{'e2e_med':>8} {'e2e_p99':>8} {'ttft_med':>9} {'ttft_p99':>9} {'itl_med':>8}")
prev = None
for f in files:
    m = json.loads(f.read_text())
    load = m["load_summary"]
    # past the cliff every request fails, and inference-perf emits successes/latency as null
    # rather than omitting them -- `or {}` keeps those stages in the table instead of crashing.
    ok = m.get("successes") or {}
    lat = ok.get("latency") or {}
    req, achv = load["requested_rate"], load.get("achieved_rate", 0)
    med = (lat.get("request_latency") or {}).get("median", 0)
    row = (f'{req:6.1f} {achv:6.1f} {(ok.get("throughput") or {}).get("requests_per_sec", 0):8.1f} '
           f'{ok.get("count", 0):6d} {(m.get("failures") or {}).get("count", 0):5d} '
           f'{med:8.2f} {(lat.get("request_latency") or {}).get("p99", 0):8.2f} '
           f'{(lat.get("time_to_first_token") or {}).get("median", 0):9.3f} '
           f'{(lat.get("time_to_first_token") or {}).get("p99", 0):9.3f} '
           f'{(lat.get("inter_token_latency") or {}).get("median", 0):8.4f}')
    # flag the knee: rate not kept up with, or median e2e latency jumping >2x in one step
    flags = []
    if achv < req * 0.95:
        flags.append(f"rate short {100 * achv / req:.0f}%")
    if prev and prev > 0 and med > 2 * prev:
        flags.append(f"e2e x{med / prev:.1f}")
    print(row + ("   <-- " + ", ".join(flags) if flags else ""))
    prev = med
