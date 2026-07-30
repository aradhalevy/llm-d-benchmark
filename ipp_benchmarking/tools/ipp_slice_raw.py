#!/usr/bin/env python3
"""Slice the full raw IPP capture into per-stage windows (gzipped) from stage_marks.log."""
import gzip, re, sys
from pathlib import Path

SRC = Path(sys.argv[1])          # raw-ipp/random-arm-full.log.gz
ROOT = Path(sys.argv[2])         # arm dir with stage_marks.log + stage dirs
PAD = 60                         # s past stage end: in-flight requests still finishing
S5_FALLBACK = 2200               # s5_release has no end_ts mark; bound it like s2 (1994 s) + pad

marks = {}
for ln in (ROOT / "stage_marks.log").read_text().split("\n"):
    p = ln.split()
    if len(p) < 2 or "=" not in p[1]:
        continue
    k, v = p[1].split("=")
    marks.setdefault(p[0], {})[k] = int(v)

win = {}
for st, m in marks.items():
    lo = m["start_ts"]
    hi = m.get("end_ts", lo + S5_FALLBACK) + PAD
    if (ROOT / st).is_dir():
        win[st] = (lo, hi)

fh = {st: gzip.open(ROOT / st / "ipp-raw.log.gz", "wt") for st in win}
n = {st: 0 for st in win}
sel = {st: 0 for st in win}
ts_re = re.compile(rb'"ts":(\d+)')
other = 0
with gzip.open(SRC, "rb") as f:
    for raw in f:
        m = ts_re.search(raw)
        if not m:
            other += 1
            continue
        t = int(m.group(1))
        for st, (lo, hi) in win.items():
            if lo <= t <= hi:
                fh[st].write(raw.decode("utf-8", "replace"))
                n[st] += 1
                if b'"msg":"Model selected"' in raw:
                    sel[st] += 1
                break
        else:
            other += 1
for h in fh.values():
    h.close()

print(f"{'stage':16s} {'window_s':>9s} {'raw_lines':>10s} {'Model selected':>15s} {'gz':>8s}")
for st in sorted(win, key=lambda s: win[s][0]):
    lo, hi = win[st]
    sz = (ROOT / st / "ipp-raw.log.gz").stat().st_size
    print(f"{st:16s} {hi-lo:9d} {n[st]:10d} {sel[st]:15d} {sz/1e6:7.1f}M")
print(f"outside all windows / no ts: {other}")
