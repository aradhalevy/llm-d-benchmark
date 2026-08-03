#!/usr/bin/env python3
"""Per-model e2e latency by stage: static split (left) vs smart routing (right).

Left panel stacks two independent static runs "as if" run together -- 8B from
--static-8b, 32B from --static-32b. Right panel splits one smart run by backend.
Each request is assigned to a stage by its dispatch time falling inside the
harness stdout stage window (elapsed from stage-0 start); x ticks are the stage
concurrency (C=...). 8B green / 32B purple, p50 solid / p95 dashed.

  plot_static_vs_smart_permodel.py --static-8b <dir> --static-32b <dir> \
      --smart <dir> --concurrencies 50,150,... -o out.png

Each <dir> has per_request_slim.json (m = "small"/"big") + harness_stdout.log.
"""
import argparse, datetime, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"small": "#2ca02c", "big": "#9467bd"}
NAME = {"small": "Qwen3-8B", "big": "Qwen3-32B"}
_STAGE_RE = re.compile(r"([0-9-]+ [0-9:,]+).*Stage ([0-9]+) - run (started|completed)")


def stage_windows(stdout_path):
    """[(stage, start, end)] in elapsed seconds from stage-0 start."""
    starts, ends = {}, {}
    for line in open(stdout_path, errors="replace"):
        m = _STAGE_RE.search(line)
        if not m:
            continue
        ep = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        (starts if m.group(3) == "started" else ends)[int(m.group(2))] = ep
    t0 = starts[0]
    return [(n, starts[n] - t0, ends[n] - t0) for n in sorted(starts) if n in ends]


def model_stage(run_dir, model, n, metric):
    """Per-stage backend metric; each request binned into a harness stage window
    (dispatch elapsed = t - first dispatch). metric=latency -> (p50, p95);
    metric=throughput -> (completed req/s, None); metric=normlat -> (p50, p95) of
    per-request latency/output-token (s/tok)."""
    allrows = json.load(open(f"{run_dir}/per_request_slim.json"))
    t0 = min(r["t"] for r in allrows)
    rows = [r for r in allrows if r["m"] == model and not r["fail"]]
    a, b = [np.nan] * n, [np.nan] * n
    for stage, s, e in stage_windows(f"{run_dir}/harness_stdout.log"):
        if stage >= n:
            continue
        sel = [r for r in rows if s <= r["t"] - t0 <= e]
        if not sel:
            continue
        if metric == "throughput":
            a[stage] = len(sel) / (e - s)
        elif metric == "normlat":
            v = np.array([r["lat"] / r["ot"] for r in sel if r["ot"]])
            if len(v):
                a[stage], b[stage] = np.percentile(v, 50), np.percentile(v, 95)
        else:
            lat = np.array([r["lat"] for r in sel])
            a[stage], b[stage] = np.percentile(lat, 50), np.percentile(lat, 95)
    return a, b


def draw(ax, series, x, n, metric):
    ymax = 0
    for run_dir, model in series:
        a, b = model_stage(run_dir, model, n, metric)
        lbl = "req/s" if metric == "throughput" else "p50"
        ax.plot(x, a, "-o", color=COLOR[model], label=f"{NAME[model]} {lbl}")
        if metric != "throughput":
            ax.plot(x, b, "--o", color=COLOR[model], alpha=.55, label=f"{NAME[model]} p95")
        top = a if metric == "throughput" else b
        ymax = max(ymax, np.nanmax(top) if np.any(~np.isnan(top)) else 0)
    return ymax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-8b", required=True)
    ap.add_argument("--static-32b", required=True)
    ap.add_argument("--smart", required=True)
    ap.add_argument("-o", "--output", default="static_vs_smart_permodel.png")
    ap.add_argument("--metric", choices=["latency", "throughput", "throughput_total", "normlat"], default="latency")
    ap.add_argument("--concurrencies", default="50,150,250,350,450,550,450,350,250,150,50")
    a = ap.parse_args()
    conc = [int(c) for c in a.concurrencies.split(",")]
    n = len(conc); x = list(range(n))

    if a.metric == "throughput_total":  # one panel: static (8B+32B summed) vs smart total
        total = lambda series: np.nansum([model_stage(d, m, n, "throughput")[0] for d, m in series], axis=0)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(x, total([(a.static_8b, "small"), (a.static_32b, "big")]), "-o",
                color="#1f77b4", label="Static total (8B + 32B)")
        ax.plot(x, total([(a.smart, "small"), (a.smart, "big")]), "-o",
                color="#d62728", label="Smart routing total")
        ax.set_xlabel("stage concurrency"); ax.set_ylabel("completed throughput (req/s)")
        ax.set_xticks(x); ax.set_xticklabels([f"C={c}" for c in conc], rotation=45, fontsize=8)
        ax.set_ylim(0, None); ax.grid(alpha=.3); ax.legend()
        ax.set_title("Combined throughput by stage — static split vs smart routing")
        fig.tight_layout(); fig.savefig(a.output, dpi=130); print("wrote", a.output)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), squeeze=False, sharey=True)
    panels = [("Static (8B + 32B run separately)", [(a.static_8b, "small"), (a.static_32b, "big")]),
              ("Smart routing (single run)", [(a.smart, "small"), (a.smart, "big")])]
    ymax = 0
    for ax, (title, series) in zip(axes[0], panels):
        ymax = max(ymax, draw(ax, series, x, n, a.metric))
        ax.set_title(title); ax.set_xlabel("stage concurrency")
        ax.set_xticks(x); ax.set_xticklabels([f"C={c}" for c in conc], rotation=45, fontsize=8)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    ylab = {"throughput": "completed throughput (req/s)",
            "normlat": "normalized latency (s / output token)"}.get(a.metric, "end-to-end request latency (s)")
    axes[0][0].set_ylabel(ylab)
    for ax in axes[0]:
        ax.set_ylim(0, ymax * 1.05)
    what = {"throughput": "throughput",
            "normlat": "normalized latency (per output token)"}.get(a.metric, "e2e latency")
    fig.suptitle(f"Per-model {what} by stage — static split vs smart routing")
    fig.tight_layout(); fig.savefig(a.output, dpi=130)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
