#!/usr/bin/env python3
"""Build an IPP scorer evaluation report from a collected-logs directory.

The report intentionally uses only the Python standard library plus PyYAML,
which is already a project dependency. It emits a self-contained HTML file
with inline SVG charts and CSV files with the parsed request/decision data.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


MODEL_COLORS = {
    "facebook/opt-125m": "#b44e3b",
    "facebook/opt-350m": "#287b6a",
}
FALLBACK_COLORS = ["#3b6fb6", "#8a5bb8", "#b3842f", "#5b6c7b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default="collected-logs-18",
        help="Collected logs directory, for example collected-logs-18.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory. Defaults to <logs_dir>/visualizations.",
    )
    return parser.parse_args()


def parse_iso_epoch(value: str) -> float:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def load_json_line(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        return json.loads(line[start:])
    except json.JSONDecodeError:
        return None


def short_model(model: str | None) -> str:
    if not model:
        return "unknown"
    return model.rsplit("/", 1)[-1]


def color_for(model: str | None, models: list[str]) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if model in models:
        return FALLBACK_COLORS[models.index(model) % len(FALLBACK_COLORS)]
    return "#67717c"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def extract_response_model(response: str) -> str | None:
    match = re.search(r'"model":"([^"]+)"', response or "")
    return match.group(1) if match else None


def parse_benchmark_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    request = json.loads(row["request"])
    response_info = row.get("info", {}).get("response_info") or {}
    chunk_times = response_info.get("chunk_times") or []
    start = float(row["start_time"])
    end = float(row["end_time"])
    return {
        "original_index": index,
        "index": index,
        "benchmark_start": start,
        "benchmark_end": end,
        "requested_model": request.get("model"),
        "chosen_model": extract_response_model(row.get("response", "")),
        "latency_s": end - start,
        "ttft_s": (float(chunk_times[0]) - start) if chunk_times else None,
        "output_tokens": response_info.get("output_tokens"),
        "input_tokens": row.get("info", {}).get("input_tokens"),
        "error": row.get("error"),
    }


def parse_decisions(logs_dir: Path) -> list[dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": {},
            "weighted_scores": {},
            "inflight": {},
            "original_model": None,
            "selected_model": None,
            "parsed_model": None,
            "request_start_ts": None,
            "request_complete_ts": None,
            "response_complete_ts": None,
        }
    )

    for log_path in sorted(logs_dir.glob("payload-processor-*.log")):
        with log_path.open() as handle:
            for line in handle:
                obj = load_json_line(line)
                if not obj:
                    continue
                request_id = obj.get("x-request-id")
                if not request_id:
                    continue
                decision = decisions[request_id]
                decision["request_id"] = request_id
                decision["log_file"] = log_path.name
                msg = obj.get("msg")
                ts = obj.get("ts")
                if msg == "captured request headers, deferring response until body arrives...":
                    decision["request_start_ts"] = ts
                elif msg == "inflight-requests score":
                    decision["request_start_ts"] = decision["request_start_ts"] or ts
                    decision["scores"][obj.get("model")] = obj.get("score")
                    decision["inflight"][obj.get("model")] = obj.get("inflightRequests")
                elif msg == "weighted score":
                    decision["weighted_scores"][obj.get("model")] = obj.get("score")
                elif msg == "model selector redirected request":
                    decision["original_model"] = obj.get("original")
                    decision["selected_model"] = obj.get("selected")
                elif msg == "parsed field from body":
                    decision["parsed_model"] = obj.get("value")
                elif msg == "processing request body complete":
                    decision["request_complete_ts"] = ts
                elif msg == "processing response body complete":
                    decision["response_complete_ts"] = ts

    complete = []
    for decision in decisions.values():
        selected = decision["selected_model"] or decision["parsed_model"]
        if not selected or not decision["request_start_ts"]:
            continue
        decision["effective_selected_model"] = selected
        decision["redirected"] = bool(decision["selected_model"])
        complete.append(decision)
    return sorted(complete, key=lambda d: (d["request_start_ts"], d["request_id"]))


def parse_runs(logs_dir: Path) -> list[dict[str, Any]]:
    result_root = logs_dir / "benchmark-results" / "results"
    runs = []
    for result_dir in sorted(result_root.glob("*/")):
        per_request_path = result_dir / "per_request_lifecycle_metrics.json"
        metadata_path = result_dir / "run_metadata.yaml"
        if not per_request_path.exists() or not metadata_path.exists():
            continue
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
        raw_rows = json.loads(per_request_path.read_text())
        rows = [parse_benchmark_row(row, index) for index, row in enumerate(raw_rows)]
        rows.sort(key=lambda row: (row["benchmark_start"], row["original_index"]))
        for request_order, row in enumerate(rows):
            row["index"] = request_order
        runs.append(
            {
                "run_id": result_dir.name,
                "result_dir": result_dir,
                "metadata": metadata,
                "start_epoch": parse_iso_epoch(metadata["harness_start"]),
                "stop_epoch": parse_iso_epoch(metadata["harness_stop"]),
                "requests": rows,
            }
        )
    return sorted(runs, key=lambda run: run["start_epoch"])


def align_decisions(runs: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    for run in runs:
        run_decisions = [
            decision
            for decision in decisions
            if run["start_epoch"] <= decision["request_start_ts"] <= run["stop_epoch"]
        ]
        run_decisions.sort(key=lambda d: (d["request_start_ts"], d["request_id"]))
        matched_decisions = match_decisions_to_rows(run["requests"], run_decisions)
        run["decisions"] = run_decisions
        for row, decision in zip(run["requests"], matched_decisions):
            if decision is None:
                continue
            row["request_id"] = decision["request_id"]
            row["decision_ts"] = decision["request_start_ts"]
            row["selected_model_log"] = decision["effective_selected_model"]
            row["redirected"] = decision["redirected"]
            row["original_model_log"] = decision["original_model"] or decision["parsed_model"]
            row["scores"] = decision["scores"]
            row["weighted_scores"] = decision["weighted_scores"]
            row["inflight"] = decision["inflight"]
            row["selected_highest_score"] = selected_highest_score(row)
            row["selected_lower_inflight"] = selected_lower_inflight(row)
        run["alignment_warning"] = None
        if len(run["requests"]) != len(run_decisions):
            run["alignment_warning"] = (
                f"Aligned {min(len(run['requests']), len(run_decisions))} decisions "
                f"for {len(run['requests'])} benchmark rows and {len(run_decisions)} IPP decisions."
            )


def match_decisions_to_rows(
    rows: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    if not rows or not decisions:
        return [None for _ in rows]
    offset = decisions[0]["request_start_ts"] - rows[0]["benchmark_start"]
    unmatched = set(range(len(decisions)))
    matched: list[dict[str, Any] | None] = []
    for row in rows:
        projected_ts = row["benchmark_start"] + offset
        same_model = [
            i
            for i in unmatched
            if decisions[i].get("effective_selected_model") == row.get("chosen_model")
        ]
        candidates = same_model or list(unmatched)
        if not candidates:
            matched.append(None)
            continue
        best_index = min(
            candidates,
            key=lambda i: (
                abs(decisions[i]["request_start_ts"] - projected_ts),
                decisions[i]["request_start_ts"],
            ),
        )
        unmatched.remove(best_index)
        matched.append(decisions[best_index])
    return matched


def selected_highest_score(row: dict[str, Any]) -> bool | None:
    scores = row.get("weighted_scores") or row.get("scores") or {}
    selected = row.get("selected_model_log") or row.get("chosen_model")
    if not scores or not selected:
        return None
    best = max(scores.values())
    return scores.get(selected) == best


def selected_lower_inflight(row: dict[str, Any]) -> bool | None:
    inflight = row.get("inflight") or {}
    selected = row.get("selected_model_log") or row.get("chosen_model")
    if selected not in inflight or len(inflight) < 2:
        return None
    selected_value = inflight[selected]
    other_values = [value for model, value in inflight.items() if model != selected]
    if not other_values or all(selected_value == value for value in other_values):
        return None
    return all(selected_value < value for value in other_values)


def summarize_run(run: dict[str, Any], models: list[str]) -> dict[str, Any]:
    rows = run["requests"]
    chosen_counts = Counter(row.get("chosen_model") for row in rows)
    redirected = sum(1 for row in rows if row.get("redirected"))
    highest = [row.get("selected_highest_score") for row in rows if row.get("selected_highest_score") is not None]
    lower = [row.get("selected_lower_inflight") for row in rows if row.get("selected_lower_inflight") is not None]
    score_ties = 0
    for row in rows:
        scores = row.get("weighted_scores") or row.get("scores") or {}
        if scores and len(set(scores.values())) == 1:
            score_ties += 1
    by_model = {}
    for model in models:
        vals = [row["latency_s"] for row in rows if row.get("chosen_model") == model]
        by_model[model] = {
            "count": len(vals),
            "mean": mean(vals) if vals else None,
            "p50": percentile(vals, 0.50) if vals else None,
            "p90": percentile(vals, 0.90) if vals else None,
            "p95": percentile(vals, 0.95) if vals else None,
        }
    return {
        "count": len(rows),
        "chosen_counts": chosen_counts,
        "redirected": redirected,
        "highest_score_rate": (sum(highest) / len(highest)) if highest else None,
        "lower_inflight_rate": (sum(lower) / len(lower)) if lower else None,
        "lower_inflight_n": len(lower),
        "score_ties": score_ties,
        "by_model": by_model,
    }


def svg_latency_scatter(run: dict[str, Any], models: list[str]) -> str:
    rows = run["requests"]
    width, height = 980, 330
    left, right, top, bottom = 58, 18, 20, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    max_latency = max(row["latency_s"] for row in rows) if rows else 1
    p95 = percentile([row["latency_s"] for row in rows], 0.95)
    y_max = max(max_latency, p95, 1)

    def x(index: int) -> float:
        return left + (index / max(1, len(rows) - 1)) * plot_w

    def y(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = [svg_open(width, height)]
    parts.append(axis(left, top, plot_w, plot_h, f"latency, seconds (max {fmt(max_latency, 1)}s)"))
    for tick in range(0, 5):
        value = y_max * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e6eaee"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt(value, 1)}</text>')
    for row in rows:
        cx = x(row["index"])
        cy = y(row["latency_s"])
        fill = color_for(row.get("chosen_model"), models)
        stroke = "#1f2933" if row.get("selected_highest_score") is False else "white"
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{fill}" stroke="{stroke}" stroke-width="1">'
            f'<title>#{row["index"]} {short_model(row.get("chosen_model"))}: {fmt(row["latency_s"])}s, '
            f'inflight={row.get("inflight")}</title></circle>'
        )
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="axislabel">request order</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_score_heatmap(run: dict[str, Any], models: list[str]) -> str:
    rows = run["requests"]
    width, height = 980, 150
    left, right, top, bottom = 145, 18, 18, 25
    plot_w = width - left - right
    cell_w = plot_w / max(1, len(rows))
    row_h = 38
    parts = [svg_open(width, height)]
    for model_index, model in enumerate(models):
        yy = top + model_index * row_h
        parts.append(f'<text x="{left - 10}" y="{yy + 24}" text-anchor="end" class="tick">{html.escape(short_model(model))}</text>')
        for request in rows:
            score = (request.get("weighted_scores") or request.get("scores") or {}).get(model)
            if score is None:
                fill = "#eef1f4"
            else:
                intensity = max(0, min(1, float(score)))
                fill = blend("#f1f5f9", color_for(model, models), intensity)
            xx = left + request["index"] * cell_w
            parts.append(f'<rect x="{xx:.2f}" y="{yy}" width="{max(cell_w, 1):.2f}" height="{row_h - 4}" fill="{fill}"/>')
        parts.append(f'<line x1="{left}" y1="{yy + row_h - 2}" x2="{left + plot_w}" y2="{yy + row_h - 2}" stroke="#ffffff"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" class="axislabel">weighted score by request, darker means higher</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_inflight_timeline(run: dict[str, Any], models: list[str]) -> str:
    rows = run["requests"]
    width, height = 980, 260
    left, right, top, bottom = 58, 110, 20, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    max_inflight = max(
        [value for row in rows for value in (row.get("inflight") or {}).values()] or [1]
    )

    def x(index: int) -> float:
        return left + (index / max(1, len(rows) - 1)) * plot_w

    def y(value: float) -> float:
        return top + plot_h - (value / max(1, max_inflight)) * plot_h

    parts = [svg_open(width, height), axis(left, top, plot_w, plot_h, "inflight requests seen by scorer")]
    for tick in range(max_inflight + 1):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e6eaee"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    for model in models:
        points = []
        for row in rows:
            value = (row.get("inflight") or {}).get(model)
            if value is not None:
                points.append(f'{x(row["index"]):.1f},{y(value):.1f}')
        if points:
            parts.append(
                f'<polyline fill="none" stroke="{color_for(model, models)}" stroke-width="2" '
                f'points="{" ".join(points)}"><title>{html.escape(model)}</title></polyline>'
            )
    for i, model in enumerate(models):
        yy = top + 20 + i * 22
        parts.append(f'<rect x="{width - right + 18}" y="{yy - 10}" width="12" height="12" fill="{color_for(model, models)}"/>')
        parts.append(f'<text x="{width - right + 36}" y="{yy}" class="tick">{html.escape(short_model(model))}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" class="axislabel">request order</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_latency_boxplot(run: dict[str, Any], models: list[str]) -> str:
    width, height = 620, 230
    left, right, top, bottom = 70, 20, 18, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    rows = run["requests"]
    max_latency = max([row["latency_s"] for row in rows] or [1])

    def y(value: float) -> float:
        return top + plot_h - (value / max_latency) * plot_h

    parts = [svg_open(width, height), axis(left, top, plot_w, plot_h, "latency distribution")]
    for tick in range(0, 5):
        value = max_latency * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e6eaee"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt(value, 1)}</text>')
    gap = plot_w / max(1, len(models))
    for i, model in enumerate(models):
        vals = [row["latency_s"] for row in rows if row.get("chosen_model") == model]
        if not vals:
            continue
        q10, q25, q50, q75, q90 = [percentile(vals, pct) for pct in (0.10, 0.25, 0.50, 0.75, 0.90)]
        cx = left + gap * i + gap / 2
        box_w = min(70, gap * 0.45)
        parts.append(f'<line x1="{cx:.1f}" y1="{y(q10):.1f}" x2="{cx:.1f}" y2="{y(q90):.1f}" stroke="{color_for(model, models)}" stroke-width="2"/>')
        parts.append(f'<rect x="{cx - box_w / 2:.1f}" y="{y(q75):.1f}" width="{box_w:.1f}" height="{max(1, y(q25) - y(q75)):.1f}" fill="{color_for(model, models)}" opacity="0.30" stroke="{color_for(model, models)}"/>')
        parts.append(f'<line x1="{cx - box_w / 2:.1f}" y1="{y(q50):.1f}" x2="{cx + box_w / 2:.1f}" y2="{y(q50):.1f}" stroke="{color_for(model, models)}" stroke-width="3"/>')
        parts.append(f'<text x="{cx:.1f}" y="{height - 18}" text-anchor="middle" class="tick">{html.escape(short_model(model))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_quality_bars(summary: dict[str, Any]) -> str:
    width, height = 340, 230
    left, right, top, bottom = 135, 18, 24, 24
    plot_w = width - left - right
    rows = [
        ("highest score", summary["highest_score_rate"], summary["count"]),
        ("lower inflight*", summary["lower_inflight_rate"], summary["lower_inflight_n"]),
    ]
    parts = [svg_open(width, height)]
    for i, (label, rate, count) in enumerate(rows):
        yy = top + i * 58
        rate = rate or 0
        parts.append(f'<text x="{left - 8}" y="{yy + 21}" text-anchor="end" class="tick">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{yy}" width="{plot_w}" height="28" rx="4" fill="#edf1f5"/>')
        parts.append(f'<rect x="{left}" y="{yy}" width="{plot_w * rate:.1f}" height="28" rx="4" fill="#287b6a"/>')
        parts.append(f'<text x="{left + plot_w + 8}" y="{yy + 19}" class="tick">{rate * 100:.0f}%</text>')
        parts.append(f'<text x="{left}" y="{yy + 45}" class="axislabel">n={count}</text>')
    parts.append(f'<text x="{left}" y="{height - 12}" class="axislabel">*only requests where inflight counts differed</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_open(width: int, height: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        f'role="img" xmlns="http://www.w3.org/2000/svg">'
    )


def axis(left: int, top: int, width: int, height: int, label: str) -> str:
    return (
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d8dee5"/>'
        f'<text x="{left}" y="{top - 6}" class="axislabel">{html.escape(label)}</text>'
    )


def blend(start_hex: str, end_hex: str, amount: float) -> str:
    def parse(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    start = parse(start_hex)
    end = parse(end_hex)
    mixed = [round(start[i] + (end[i] - start[i]) * amount) for i in range(3)]
    return "#" + "".join(f"{part:02x}" for part in mixed)


def write_csv(output_dir: Path, runs: list[dict[str, Any]], models: list[str]) -> None:
    request_fields = [
        "run_id",
        "index",
        "request_id",
        "requested_model",
        "chosen_model",
        "selected_model_log",
        "redirected",
        "latency_s",
        "ttft_s",
        "input_tokens",
        "output_tokens",
        "selected_highest_score",
        "selected_lower_inflight",
    ]
    for model in models:
        request_fields.extend([f"score_{short_model(model)}", f"inflight_{short_model(model)}"])
    with (output_dir / "requests_enriched.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=request_fields)
        writer.writeheader()
        for run in runs:
            for row in run["requests"]:
                out = {field: row.get(field) for field in request_fields}
                out["run_id"] = run["run_id"]
                for model in models:
                    out[f"score_{short_model(model)}"] = (row.get("weighted_scores") or row.get("scores") or {}).get(model)
                    out[f"inflight_{short_model(model)}"] = (row.get("inflight") or {}).get(model)
                writer.writerow(out)

    with (output_dir / "run_summary.csv").open("w", newline="") as handle:
        fields = ["run_id", "requests", "redirected", "score_ties", "highest_score_rate", "lower_inflight_rate"]
        for model in models:
            fields.extend([f"{short_model(model)}_count", f"{short_model(model)}_mean_latency", f"{short_model(model)}_p90_latency"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            summary = run["summary"]
            row = {
                "run_id": run["run_id"],
                "requests": summary["count"],
                "redirected": summary["redirected"],
                "score_ties": summary["score_ties"],
                "highest_score_rate": summary["highest_score_rate"],
                "lower_inflight_rate": summary["lower_inflight_rate"],
            }
            for model in models:
                row[f"{short_model(model)}_count"] = summary["by_model"][model]["count"]
                row[f"{short_model(model)}_mean_latency"] = summary["by_model"][model]["mean"]
                row[f"{short_model(model)}_p90_latency"] = summary["by_model"][model]["p90"]
            writer.writerow(row)


def render_table(summary: dict[str, Any], models: list[str]) -> str:
    rows = []
    for model in models:
        stats = summary["by_model"][model]
        rows.append(
            "<tr>"
            f"<td><span class='swatch' style='background:{color_for(model, models)}'></span>{html.escape(short_model(model))}</td>"
            f"<td>{stats['count']}</td>"
            f"<td>{fmt(stats['mean'])}</td>"
            f"<td>{fmt(stats['p50'])}</td>"
            f"<td>{fmt(stats['p90'])}</td>"
            f"<td>{fmt(stats['p95'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>chosen model</th><th>requests</th><th>mean s</th>"
        "<th>p50 s</th><th>p90 s</th><th>p95 s</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_html(logs_dir: Path, runs: list[dict[str, Any]], models: list[str]) -> str:
    total_requests = sum(run["summary"]["count"] for run in runs)
    total_redirected = sum(run["summary"]["redirected"] for run in runs)
    sections = []
    for run in runs:
        summary = run["summary"]
        metadata = run["metadata"]
        warning = f"<p class='warning'>{html.escape(run['alignment_warning'])}</p>" if run.get("alignment_warning") else ""
        sections.append(
            f"""
            <section class="run">
              <h2>{html.escape(run["run_id"])}</h2>
              <p class="meta">
                {html.escape(metadata.get("harness_start", ""))} to {html.escape(metadata.get("harness_stop", ""))}
                · workload {html.escape(metadata.get("harness_workload", ""))}
                · {summary["count"]} requests
                · {summary["redirected"]} redirected
                · {summary["score_ties"]} score ties
              </p>
              {warning}
              <div class="cards">
                <div class="card"><div class="metric">{(f'{summary["highest_score_rate"]*100:.0f}%' if summary["highest_score_rate"] is not None else '-')}</div><div class="label">chosen model had highest score</div></div>
                <div class="card"><div class="metric">{(f'{summary["lower_inflight_rate"]*100:.0f}%' if summary["lower_inflight_rate"] is not None else '-')}</div><div class="label">chose lower-inflight model when counts differed</div></div>
                <div class="card"><div class="metric">{summary["redirected"]}</div><div class="label">requests redirected by IPP</div></div>
              </div>
              <div class="grid">
                <div class="panel wide"><h3>Latency By Request</h3>{svg_latency_scatter(run, models)}</div>
                <div class="panel"><h3>Decision Quality</h3>{svg_quality_bars(summary)}</div>
                <div class="panel"><h3>Latency Distribution</h3>{svg_latency_boxplot(run, models)}</div>
                <div class="panel wide"><h3>Inflight Counts Seen By Scorer</h3>{svg_inflight_timeline(run, models)}</div>
                <div class="panel wide"><h3>Weighted Score Heatmap</h3>{svg_score_heatmap(run, models)}</div>
                <div class="panel wide"><h3>Latency Summary</h3>{render_table(summary, models)}</div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IPP scorer evaluation: {html.escape(logs_dir.name)}</title>
  <style>
    :root {{ --text: #1f2933; --muted: #65717d; --line: #d8dee5; --bg: #f6f8fa; --panel: #ffffff; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }}
    header {{ padding: 28px 34px 18px; background: #ffffff; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 6px; font-size: 22px; }}
    h3 {{ margin: 0 0 12px; font-size: 15px; }}
    p {{ max-width: 1100px; line-height: 1.45; }}
    .meta, .label, .axislabel, .tick {{ color: var(--muted); font-size: 12px; }}
    .run {{ padding: 0 34px 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 12px; max-width: 980px; margin: 16px 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(300px, 1fr)); gap: 14px; }}
    .wide {{ grid-column: 1 / -1; }}
    .warning {{ color: #8a4b12; background: #fff7e8; border: 1px solid #f0d2a2; padding: 10px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 4px; }}
    @media (max-width: 760px) {{ .grid, .cards {{ grid-template-columns: 1fr; }} .wide {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <header>
    <h1>IPP Inflight-Requests Scorer Evaluation</h1>
    <p>
      Source: <code>{html.escape(str(logs_dir))}</code>. This report joins benchmark per-request latency with
      payload-processor scorer logs by run window and request order. It is meant to show whether IPP selected the
      highest-scoring model, whether that corresponded to lower inflight load, and what latency outcome followed.
    </p>
    <p><strong>{len(runs)}</strong> benchmark run(s), <strong>{total_requests}</strong> requests, <strong>{total_redirected}</strong> IPP redirects.</p>
  </header>
  {"".join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else logs_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(logs_dir)
    decisions = parse_decisions(logs_dir)
    align_decisions(runs, decisions)
    models = sorted(
        {
            model
            for run in runs
            for row in run["requests"]
            for model in [
                row.get("requested_model"),
                row.get("chosen_model"),
                *(row.get("scores") or {}).keys(),
                *(row.get("weighted_scores") or {}).keys(),
            ]
            if model
        }
    )
    for run in runs:
        run["summary"] = summarize_run(run, models)

    write_csv(output_dir, runs, models)
    (output_dir / "ipp_scorer_report.html").write_text(render_html(logs_dir, runs, models))
    print(f"Wrote {output_dir / 'ipp_scorer_report.html'}")
    print(f"Wrote {output_dir / 'requests_enriched.csv'}")
    print(f"Wrote {output_dir / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
