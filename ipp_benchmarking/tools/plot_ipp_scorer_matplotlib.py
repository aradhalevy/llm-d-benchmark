#!/usr/bin/env python3
"""Generate matplotlib figures for IPP scorer evaluation logs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llmdbenchmark-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from visualize_ipp_scorer_run import (
    align_decisions,
    parse_decisions,
    parse_runs,
    short_model,
    summarize_run,
)


COLORS = {
    "facebook/opt-125m": "#b44e3b",
    "facebook/opt-350m": "#287b6a",
}
MARKERS = {
    "facebook/opt-125m": "o",
    "facebook/opt-350m": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default="collected-logs-18",
        help="Collected logs directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory. Defaults to <logs_dir>/matplotlib-figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def color(model: str) -> str:
    return COLORS.get(model, "#4b6f9f")


def marker(model: str) -> str:
    return MARKERS.get(model, "o")


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.8,
        }
    )


def compute_itl(row: dict) -> float | None:
    times = row.get("output_token_times")
    if times is None:
        return None
    diffs = np.diff(np.asarray(times, dtype=float))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 0.0
    return float(np.mean(positive))


def enrich_rows_for_plotting(runs: list[dict]) -> None:
    for run in runs:
        for row in run["requests"]:
            # Reconstruct ITL from token timestamps if the parser did not expose it.
            row["itl_s"] = compute_itl(row)


def add_token_timestamps_to_rows(runs: list[dict]) -> None:
    # parse_runs keeps only compact fields. Pull token timestamps from the raw
    # per-request file by original_index so the plotting script can compute ITL.
    import json

    for run in runs:
        raw_path = run["result_dir"] / "per_request_lifecycle_metrics.json"
        raw_rows = json.loads(raw_path.read_text())
        by_original = {}
        for idx, raw in enumerate(raw_rows):
            info = raw.get("info", {}).get("response_info") or {}
            by_original[idx] = info.get("output_token_times") or []
        for row in run["requests"]:
            row["output_token_times"] = by_original.get(row["original_index"], [])


def collect_models(runs: list[dict]) -> list[str]:
    return sorted(
        {
            model
            for run in runs
            for row in run["requests"]
            for model in [
                row.get("requested_model"),
                row.get("chosen_model"),
                row.get("selected_model_log"),
                *(row.get("scores") or {}).keys(),
                *(row.get("weighted_scores") or {}).keys(),
            ]
            if model
        }
    )


def finite_metric(rows: list[dict], key: str) -> np.ndarray:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return np.asarray(values, dtype=float)


def scatter_metric_by_model(
    ax: plt.Axes,
    rows: list[dict],
    models: list[str],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for model in models:
        model_rows = [row for row in rows if row.get("chosen_model") == model and row.get(metric) is not None]
        if not model_rows:
            continue
        x = [row["index"] for row in model_rows]
        y = [max(float(row[metric]), 1e-6) for row in model_rows]
        redirected = [bool(row.get("redirected")) for row in model_rows]
        ax.scatter(
            x,
            y,
            s=[26 if r else 18 for r in redirected],
            c=color(model),
            marker=marker(model),
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            label=short_model(model),
        )
    values = finite_metric(rows, metric)
    if len(values):
        ax.set_ylim(bottom=0, top=float(np.max(values)) * 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Request order")
    ax.legend(loc="best", frameon=True)


def plot_all_runs_timeseries(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), constrained_layout=True)

    fig.suptitle("Per-request latency metrics — all runs", fontweight="bold")

    metrics = [
        ("latency_s", "Request latency (s)", "End-to-end latency"),
        ("ttft_s", "TTFT (s)", "Time to first token"),
        ("itl_s", "Mean ITL (s)", "Mean inter-token latency"),
    ]

    all_rows = [row for run in runs for row in run["requests"]]

    for ax, (metric, ylabel, title) in zip(axes, metrics, strict=True):
        for model in models:
            model_rows = [r for r in all_rows if r.get("chosen_model") == model and r.get(metric) is not None]
            if not model_rows:
                continue
            x = [r["index"] for r in model_rows]
            y = [max(float(r[metric]), 1e-6) for r in model_rows]
            redirected = [bool(r.get("redirected")) for r in model_rows]
            ax.scatter(
                x,
                y,
                s=[26 if red else 18 for red in redirected],
                c=color(model),
                marker=marker(model),
                alpha=0.65,
                edgecolors="white",
                linewidths=0.35,
                label=short_model(model),
            )

    for ax, (metric, ylabel, title) in zip(axes, metrics, strict=True):
        all_rows = [row for run in runs for row in run["requests"]]
        values = finite_metric(all_rows, metric)
        if len(values):
            ax.set_ylim(bottom=0, top=float(np.max(values)) * 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", axis="y")
        ax.margins(x=0.01)
        ax.legend(loc="best", frameon=True)
    axes[-1].set_xlabel("Request order")

    save_figure(fig, output_dir / "all_runs_latency_ttft_itl", formats)


def plot_all_runs_decisions(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.5, 6.8), constrained_layout=True)

    fig.suptitle("Scorer signals — all runs", fontweight="bold")

    all_rows = [row for run in runs for row in run["requests"]]

    for model in models:
        x = [row["index"] for row in all_rows if model in (row.get("inflight") or {})]
        y = [(row.get("inflight") or {}).get(model) for row in all_rows if model in (row.get("inflight") or {})]
        ax0.scatter(x, y, color=color(model), marker=marker(model), label=short_model(model), s=18, alpha=0.65, edgecolors="white", linewidths=0.35)

    for model in models:
        x = [row["index"] for row in all_rows if model in (row.get("weighted_scores") or row.get("scores") or {})]
        y = [
            (row.get("weighted_scores") or row.get("scores") or {}).get(model)
            for row in all_rows
            if model in (row.get("weighted_scores") or row.get("scores") or {})
        ]
        ax1.scatter(x, y, color=color(model), marker=marker(model), label=short_model(model), s=18, alpha=0.65, edgecolors="white", linewidths=0.35)

    ax0.set_title("Inflight load observed by scorer")
    ax0.set_ylabel("Inflight requests")
    ax0.legend(loc="best", frameon=True)
    ax0.grid(True, axis="y")
    ax0.margins(x=0.01)

    ax1.set_ylabel("Weighted score")
    ax1.set_xlabel("Request order")
    ax1.set_ylim(-0.08, 1.08)
    ax1.set_title("Scorer output used by picker")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, axis="y")
    ax1.margins(x=0.01)

    save_figure(fig, output_dir / "all_runs_scorer_signals", formats)


def plot_distribution_summary(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    # Write three separate figures: CDF, histogram, and strip+box
    _plot_distribution_cdf(runs, models, output_dir, formats)
    _plot_distribution_histogram(runs, models, output_dir, formats)
    _plot_distribution_stripbox(runs, models, output_dir, formats)


def _plot_distribution_cdf(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    """CDF: cumulative distribution function — read percentiles directly."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    metrics = [
        ("latency_s", "Request latency (s)", "Latency"),
        ("ttft_s", "TTFT (s)", "TTFT"),
        ("itl_s", "Mean ITL (s)", "ITL"),
    ]
    all_rows = [row for run in runs for row in run["requests"]]

    for ax, (metric, ylabel, title) in zip(axes, metrics, strict=True):
        for model in models:
            vals = sorted([max(float(row[metric]), 1e-6) for row in all_rows if row.get("chosen_model") == model and row.get(metric) is not None])
            if not vals:
                continue
            n = len(vals)
            y = np.arange(1, n + 1) / n
            ax.plot(vals, y, color=color(model), label=short_model(model), linewidth=1.8)
            # mark p50, p90, p99
            for p, ls in [(50, "-"), (90, "--"), (99, ":")]:
                idx = int(n * p / 100) - 1
                ax.axvline(vals[idx], color=color(model), linestyle=ls, alpha=0.35, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Cumulative fraction of requests")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle("Latency metric distributions — CDF (log scale)", fontweight="bold")
    save_figure(fig, output_dir / "all_runs_latency_metric_distributions_cdf", formats)


def _plot_distribution_histogram(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    """Histogram: overlapping bars with log x-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    metrics = [
        ("latency_s", "Request latency (s)", "Latency"),
        ("ttft_s", "TTFT (s)", "TTFT"),
        ("itl_s", "Mean ITL (s)", "ITL"),
    ]
    all_rows = [row for run in runs for row in run["requests"]]

    for ax, (metric, ylabel, title) in zip(axes, metrics, strict=True):
        data = []
        labels = []
        colors_list = []
        for model in models:
            vals = [max(float(row[metric]), 1e-6) for row in all_rows if row.get("chosen_model") == model and row.get(metric) is not None]
            if vals:
                data.append(vals)
                labels.append(short_model(model))
                colors_list.append(color(model))

        if not data:
            continue

        # automatic bins based on data range
        all_vals = np.concatenate(data)
        bins = np.logspace(np.log10(max(all_vals.min(), 1e-5)), np.log10(all_vals.max() * 1.05), 35)

        for vals, label, c in zip(data, labels, colors_list, strict=True):
            ax.hist(vals, bins=bins, alpha=0.55, label=label, color=c, edgecolor="none", density=False)

        ax.set_xscale("log")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Request count")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", frameon=True)

    fig.suptitle("Latency metric distributions — Histogram (log scale)", fontweight="bold")
    save_figure(fig, output_dir / "all_runs_latency_metric_distributions_histogram", formats)


def _plot_distribution_stripbox(runs: list[dict], models: list[str], output_dir: Path, formats: list[str]) -> None:
    """Strip + box: classic boxplot with jittered raw points alongside."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    metrics = [
        ("latency_s", "Request latency (s)", "Latency"),
        ("ttft_s", "TTFT (s)", "TTFT"),
        ("itl_s", "Mean ITL (s)", "ITL"),
    ]
    all_rows = [row for run in runs for row in run["requests"]]

    for ax, (metric, ylabel, title) in zip(axes, metrics, strict=True):
        data = []
        labels = []
        colors_list = []
        for model in models:
            vals = [max(float(row[metric]), 1e-6) for row in all_rows if row.get("chosen_model") == model and row.get(metric) is not None]
            if vals:
                data.append(vals)
                labels.append(short_model(model))
                colors_list.append(color(model))

        if not data:
            continue

        # boxplot
        bp = ax.boxplot(data, positions=range(1, len(data) + 1), widths=0.5, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors_list, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for median in bp["medians"]:
            median.set_color("#1f2933")
            median.set_linewidth(2.0)

        # jittered strip of actual points
        rng = np.random.default_rng(42)
        for pos, (vals, c) in enumerate(zip(data, colors_list, strict=True), start=1):
            x = rng.normal(pos, 0.06, size=len(vals))
            ax.scatter(x, vals, color=c, s=8, alpha=0.35, edgecolors="none", zorder=3)

        ax.set_yscale("log")
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", alpha=0.35)

    fig.suptitle("Latency metric distributions — Box + Strip (log scale)", fontweight="bold")
    save_figure(fig, output_dir / "all_runs_latency_metric_distributions_boxstrip", formats)


def write_text_summary(runs: list[dict], models: list[str], output_dir: Path) -> None:
    lines = ["IPP scorer matplotlib summary", ""]
    for run in runs:
        summary = run["summary"]
        lines.append(run["run_id"])
        lines.append(f"  requests: {summary['count']}")
        lines.append(f"  redirected: {summary['redirected']}")
        hsr = summary['highest_score_rate']
        lir = summary['lower_inflight_rate']
        lines.append(f"  highest-score selection rate: {hsr:.3f}" if hsr is not None else "  highest-score selection rate: -")
        lir_str = f"{lir:.3f}" if lir is not None else "-"
        lines.append(
            f"  lower-inflight selection rate when counts differed: "
            f"{lir_str} (n={summary['lower_inflight_n']})"
        )
        for model in models:
            stats = summary["by_model"][model]
            lines.append(
                f"  {short_model(model)}: n={stats['count']}, "
                f"mean latency={stats['mean']:.3f}s, p90={stats['p90']:.3f}s"
            )
        lines.append("")
    (output_dir / "matplotlib_summary.txt").write_text("\n".join(lines))


def save_figure(fig: plt.Figure, stem: Path, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else logs_dir / "matplotlib-figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    runs = parse_runs(logs_dir)
    decisions = parse_decisions(logs_dir)
    align_decisions(runs, decisions)
    add_token_timestamps_to_rows(runs)
    enrich_rows_for_plotting(runs)
    models = collect_models(runs)
    for run in runs:
        run["summary"] = summarize_run(run, models)

    plot_all_runs_timeseries(runs, models, output_dir, args.formats)
    plot_all_runs_decisions(runs, models, output_dir, args.formats)
    plot_distribution_summary(runs, models, output_dir, args.formats)
    write_text_summary(runs, models, output_dir)

    print(f"Wrote matplotlib figures to {output_dir}")


if __name__ == "__main__":
    main()
