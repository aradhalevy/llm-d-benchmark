#!/usr/bin/env python3
"""5s-window p50 e2e latency: static single-model vs smart-routed, per model.

  plot_p50_static_vs_smart.py --smart <dir> --static8b <dir> --static32b <dir> \
      -o <out_prefix> [--bin 5] [--concurrencies 50,150,...]

e2e latency is reconstructed per request from each run's ipp-full-live.log
("processing request headers complete" -> "processing response body complete",
tagged by the "Model selected" model); this matches inference-perf's client
per_request latency to <1% and is the only per-request source the smart run has
(its per_request_lifecycle file OOM'd on write). Static requests are all one
model; smart requests are split by the routed model.

Writes two full-run figures (p50 vs run-relative time, stage bands) — one for 8B,
one for 32B — plus a per-stage zoom PNG (stage window +/-100s) into <prefix>_<m>_stages/.
"""
import argparse, datetime, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

STAGE_PAD_S = 100.0
_STAGE_RE = re.compile(r"([0-9-]+ [0-9:,]+).*Stage ([0-9]+) - run (started|completed)")
_TS = re.compile(r'"ts":([0-9.]+)')
_RID = re.compile(r'"x-request-id":"([^"]+)"')
_MDL = re.compile(r'"model":"Qwen/Qwen3-(8B|32B)"')
STATIC_C, SMART_C = "#1f77b4", "#2ca02c"


def ipp_requests(path):
    """[(ts_start, e2e_s, model)] per request from an ipp-full-live.log."""
    start, end, mdl = {}, {}, {}
    for line in open(path, errors="replace"):
        if "x-request-id" not in line:
            continue
        r = _RID.search(line)
        if not r:
            continue
        rid = r.group(1)
        if "processing request headers complete" in line:
            t = _TS.search(line); start[rid] = float(t.group(1)) if t else start.get(rid)
        elif "processing response body complete" in line:
            t = _TS.search(line); end[rid] = float(t.group(1)) if t else end.get(rid)
        elif "Model selected" in line:
            m = _MDL.search(line)
            if m: mdl[rid] = m.group(1)
    rows = [(start[r], end[r] - start[r], mdl.get(r)) for r in end if r in start]
    return sorted(rows)


def slim_rows(path):
    """[(t, e2e_s, model)] from inference-perf per_request_slim.json (client latency)."""
    mp = {"small": "8B", "big": "32B"}
    return sorted((r["t"], r["lat"], mp.get(r["m"]))
                  for r in json.load(open(path)) if not r["fail"])


_TTFT = re.compile(r'"ttft_s":([0-9.]+)')


def ttft_observations(path):
    """[(dispatch_ts, ttft_s, model)] from IPP "ttft-observation" lines; binned at
    dispatch time (ts - ttft_s, when load was applied) to align with stage windows."""
    out = []
    for line in open(path, errors="replace"):
        if '"msg":"ttft-observation"' not in line:
            continue
        t, v, m = _TS.search(line), _TTFT.search(line), _MDL.search(line)
        if t and v and m:
            out.append((float(t.group(1)) - float(v.group(1)), float(v.group(1)), m.group(1)))
    return sorted(out)


def stage_windows(stdout_path):
    """[(conc, start, end)] wall-clock UTC epoch per stage from the harness log."""
    starts, ends = {}, {}
    for line in open(stdout_path, errors="replace"):
        m = _STAGE_RE.search(line)
        if not m:
            continue
        ep = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        (starts if m.group(3) == "started" else ends)[int(m.group(2))] = ep
    return {s: (starts[s], ends.get(s)) for s in sorted(starts)}


def binned_p50(rows, t0, bin_s):
    d = {}
    for t, lat, _ in rows:
        d.setdefault(int((t - t0) // bin_s), []).append(lat)
    xs = sorted(d)
    x = np.array([(b + 0.5) * bin_s for b in xs]); y = np.array([np.median(d[b]) for b in xs])
    # break the line across idle inter-stage gaps so it isn't bridged diagonally
    gaps = np.where(np.diff(x) > 2.5 * bin_s)[0]
    for g in gaps[::-1]:
        x = np.insert(x, g + 1, np.nan); y = np.insert(y, g + 1, np.nan)
    return x, y


def draw_bands(ax, win, t0, conc):
    for s, (st, en) in win.items():
        en = en if en else st
        ax.axvspan(st - t0, en - t0, color="0.85", alpha=.35, lw=0)
        ax.axvline(st - t0, color="0.6", lw=0.5, ls="--")
        ax.text((st - t0 + (en - t0)) / 2, 0.98, f"C={conc[s] if s < len(conc) else '?'}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="0.4")


def make(model, s_rows, s_win, m_rows, m_win, conc, bin_s, out, ylabel):
    band_t0 = min(v[0] for v in s_win.values())
    # model=None -> pool all requests ("total"). static line uses its own clock (slim
    # monotonic ts, or IPP epoch) anchored at its first request ~= stage 0 start.
    slabel = f"static {model}" if model else "static (8B+32B pooled)"
    mlabel = f"smart routing ({model} requests)" if model else "smart routing (all requests)"
    s_rows = [r for r in s_rows if model is None or r[2] == model or r[2] is None]
    st0 = min(r[0] for r in s_rows)
    mt0 = min(v[0] for v in m_win.values())
    sx, sy = binned_p50(s_rows, st0, bin_s)
    mx, my = binned_p50([r for r in m_rows if model is None or r[2] == model], mt0, bin_s)

    fig, ax = plt.subplots(figsize=(14, 5))
    draw_bands(ax, s_win, band_t0, conc)
    ax.plot(sx, sy, "-", color=STATIC_C, lw=1.3, label=slabel)
    ax.plot(mx, my, "-", color=SMART_C, lw=1.3, label=mlabel)
    ax.set_xlabel("time since stage 0 start (s)"); ax.set_ylabel(ylabel)
    ax.set_title(f"{model or 'total'}: static vs smart routing — {bin_s:g}s-window {ylabel}")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=130); plt.close(fig)
    print("wrote", f"{out}.png")

    zdir = f"{out}_stages"; os.makedirs(zdir, exist_ok=True)
    for s in sorted(set(s_win) | set(m_win)):
        se = s_win.get(s); me = m_win.get(s)
        lo = min([(se[0] - band_t0) if se else np.inf, (me[0] - mt0) if me else np.inf]) - STAGE_PAD_S
        hi = max([((se[1] or se[0]) - band_t0) if se else -np.inf,
                  ((me[1] or me[0]) - mt0) if me else -np.inf]) + STAGE_PAD_S
        fig, ax = plt.subplots(figsize=(12, 5))
        draw_bands(ax, s_win, band_t0, conc)
        ax.plot(sx, sy, "-o", color=STATIC_C, ms=3, lw=1, label=slabel)
        ax.plot(mx, my, "-o", color=SMART_C, ms=3, lw=1, label=mlabel)
        ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.set_xlabel("time since stage 0 start (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{model or 'total'} stage {s} (C={conc[s] if s < len(conc) else '?'}): static vs smart p50")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{zdir}/stage_{s}.png", dpi=120); plt.close(fig)
    print("wrote", f"{zdir}/ ({len(set(s_win) | set(m_win))} stages)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart", required=True); ap.add_argument("--static8b", required=True)
    ap.add_argument("--static32b", required=True)
    ap.add_argument("-o", "--output", default="p50_static_vs_smart")
    ap.add_argument("--bin", type=float, default=5.0)
    ap.add_argument("--metric", choices=("latency", "ttft"), default="latency")
    ap.add_argument("--combined", action="store_true",
                    help="one 'total' graph: all smart requests vs both static runs (8B+32B) pooled")
    ap.add_argument("--concurrencies", default="50,150,250,350,450,550,450,350,250,150,50")
    a = ap.parse_args()
    conc = [int(c) for c in a.concurrencies.split(",")]
    ttft = a.metric == "ttft"
    ylabel = "TTFT p50 (s)" if ttft else "e2e latency p50 (s)"

    def static_src(sdir):  # TTFT: IPP for all. latency: slim (client) for static.
        slim = f"{sdir}/per_request_slim.json"
        if ttft: return ttft_observations(f"{sdir}/ipp-full-live.log")
        return slim_rows(slim) if os.path.exists(slim) else ipp_requests(f"{sdir}/ipp-full-live.log")

    m_rows = ttft_observations(f"{a.smart}/ipp-full-live.log") if ttft else ipp_requests(f"{a.smart}/ipp-full-live.log")
    m_win = stage_windows(f"{a.smart}/harness_stdout.log")

    if a.combined:
        rel = lambda rows: (lambda t0: [(t - t0, l, mdl) for t, l, mdl in rows])(min(r[0] for r in rows))
        relw = lambda win: (lambda t0: {s: (st - t0, en - t0 if en else None) for s, (st, en) in win.items()})(
            min(v[0] for v in win.values()))
        # pool both static runs on run-relative time; bands from static_8b (durations ~match)
        s_pool = rel(static_src(a.static8b)) + rel(static_src(a.static32b))
        make(None, sorted(s_pool), relw(stage_windows(f"{a.static8b}/harness_stdout.log")),
             rel(m_rows), relw(m_win), conc, a.bin, f"{a.output}_total", ylabel)
        return
    for model, sdir in (("8B", a.static8b), ("32B", a.static32b)):
        make(model, static_src(sdir), stage_windows(f"{sdir}/harness_stdout.log"),
             m_rows, m_win, conc, a.bin, f"{a.output}_{model.lower()}", ylabel)


if __name__ == "__main__":
    main()
