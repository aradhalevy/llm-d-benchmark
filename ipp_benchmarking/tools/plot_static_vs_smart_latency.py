#!/usr/bin/env python3
"""Deliverable: the two static single-model legs (8B + 32B) pooled "as one
workload" vs the smart-routed workload, at matching full-equivalent concurrency.

  plot_static_vs_smart_latency.py <static_combined_dir> <smart_stages_dir> \
      <smart_decisions.log> -o out.png --concurrencies 50,150,...

* combined static: per-request latency pooled from the merged slim (8B@C/2 +
  32B@C/2), percentiles computed per stage.
* smart: per-stage request_latency percentiles from inference-perf's
  stage_*_lifecycle_metrics.json (its giant per-request file OOM'd on write, but
  the authoritative per-stage aggregates are intact).
* routing split: combined = 8B/32B request mix from the slim; smart = 32B-offload
  share per stage from the IPP decision log (deduped by x-request-id).
"""
import argparse, glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, PURPLE = "#2ca02c", "#9467bd"


def cluster(rows, n, gap=8.0, min_size=20):
    rows = sorted(rows, key=lambda r: r["t"])
    stages, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["t"] - prev["t"] > gap:
            stages.append(cur); cur = []
        cur.append(r)
    stages.append(cur)
    stages = [s for s in stages if len(s) >= min_size]
    while len(stages) > n:
        g = [stages[i + 1][0]["t"] - stages[i][-1]["t"] for i in range(len(stages) - 1)]
        i = min(range(len(g)), key=lambda j: g[j]); stages[i] += stages[i + 1]; del stages[i + 1]
    return stages


def combined_static(d, n):
    rows = json.load(open(f"{d}/per_request_slim.json"))
    out = []
    for st in cluster(rows, n):
        lat = np.array([r["lat"] for r in st if not r["fail"]])
        b = sum(1 for r in st if r["m"] == "big")
        out.append({"p50": np.percentile(lat, 50), "p95": np.percentile(lat, 95),
                    "n": len(st), "share32": b / len(st)})
    return out


def smart_stages(d):
    out = []
    for f in sorted(glob.glob(f"{d}/stage_*.json"),
                    key=lambda p: int(p.split("stage_")[1].split(".")[0])):
        x = json.load(open(f)); lat = x["successes"]["latency"]["request_latency"]
        out.append({"C": x["load_summary"]["concurrency"], "p50": lat["median"],
                    "p95": lat["p95"], "n": x["successes"]["count"],
                    "fail": x.get("failures", {}).get("count", 0)})
    return out


def smart_split(logpath, n):
    seen = {}
    import re
    pat = re.compile(r'"ts":([0-9.]+).*?"x-request-id":"([^"]+)","model":"Qwen/Qwen3-(8B|32B)"')
    for line in open(logpath, errors="replace"):
        if "Model selected" not in line:
            continue
        m = pat.search(line)
        if m:
            seen[m.group(2)] = (float(m.group(1)), m.group(3))
    rows = [{"t": t, "big": mdl == "32B"} for t, mdl in seen.values()]
    return [sum(r["big"] for r in s) / len(s) for s in cluster(rows, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("static_combined"); ap.add_argument("smart_stages"); ap.add_argument("smart_decisions")
    ap.add_argument("-o", "--output", default="static_vs_smart_latency.png")
    ap.add_argument("--concurrencies", default="50,150,250,350,450,550,450,350,250,150,50")
    a = ap.parse_args()
    conc = [int(c) for c in a.concurrencies.split(",")]
    n = len(conc)
    cs = combined_static(a.static_combined, n)
    sm = smart_stages(a.smart_stages)
    sm_split = smart_split(a.smart_decisions, n)
    x = list(range(n))

    fig, (axL, axB) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.3, 1]})
    # latency
    axL.plot(x, [s["p50"] for s in cs], "-o", color=PURPLE, label="static split (8B+32B) p50")
    axL.plot(x, [s["p95"] for s in cs], "--o", color=PURPLE, alpha=.6, label="static split p95")
    axL.plot(x, [s["p50"] for s in sm], "-s", color=GREEN, label="smart routing p50")
    axL.plot(x, [s["p95"] for s in sm], "--s", color=GREEN, alpha=.6, label="smart routing p95")
    axL.set_ylabel("end-to-end request latency (s)")
    axL.set_title("Static split (8B+32B as one workload) vs smart routing — latency vs concurrency")
    axL.legend(fontsize=9); axL.grid(alpha=.3)
    # routing split (32B share)
    w = 0.38
    axB.bar([i - w/2 for i in x], [s["share32"]*100 for s in cs], w, color=PURPLE, label="static split: % on 32B")
    axB.bar([i + w/2 for i in x], [sm_split[i]*100 if i < len(sm_split) else 0 for i in x], w,
            color=GREEN, label="smart: % offloaded to 32B")
    axB.set_ylabel("% requests on 32B"); axB.set_xlabel("full-equivalent concurrency")
    axB.set_xticks(x); axB.set_xticklabels(conc)
    axB.legend(fontsize=9); axB.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.output, dpi=130)
    print("wrote", a.output)
    print("static combined p50:", [round(s["p50"],1) for s in cs])
    print("smart          p50:", [round(s["p50"],1) for s in sm])
    print("static combined p95:", [round(s["p95"],1) for s in cs])
    print("smart          p95:", [round(s["p95"],1) for s in sm])


if __name__ == "__main__":
    main()
