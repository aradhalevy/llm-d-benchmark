#!/usr/bin/env python3
"""Plots for the Kind model-routing rehearsal (README section C).

Data sources:
- Routing timelines: 60s monitoring windows of IPP "Model selected" log lines
  captured live during the runs of 2026-06-11 (the kind node rotates pod logs,
  so the full per-request timeline is not recoverable afterwards — always
  monitor live or dump logs right after a run).
- Stage outcomes: stage_*_lifecycle_metrics.json from the inference-perf runs.

Outputs PNGs into ipp_benchmarking/example_outputs/kind-model-routing/.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "example_outputs" / "kind-model-routing"
OUT.mkdir(parents=True, exist_ok=True)

SLOW = "facebook/opt-125m (slow, ~0.8 RPS cap)"
FAST = "facebook/opt-350m (fast, ~4.5-5 RPS cap)"
C_SLOW, C_FAST = "#d62728", "#2ca02c"

# ---------------------------------------------------------------- timeline 1
# avg-ttft-scorer (1.0) + max-score-picker, hot sweep treatment 1
# per-minute selections [slow, fast]
avg = np.array([
    [2, 134], [1, 99],          # 3 RPS: all to the fast sim (correct)
    [1, 241], [3, 304],         # 12 RPS: fast saturating, EMA lags, no overflow yet
    [223, 0], [0, 156],         # EMA crosses -> 100% stampede to slow, then flip back
    [441, 188], [805, 14],      # 40 RPS: oscillation, then stale-EMA lock-in on slow
    [0, 377], [0, 440],         # drain: locked back on fast
])

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(avg))
ax.bar(x, avg[:, 0], color=C_SLOW, label=SLOW)
ax.bar(x, avg[:, 1], bottom=avg[:, 0], color=C_FAST, label=FAST)
for xpos, label in [(0, "3 RPS"), (2, "12 RPS"), (6, "40 RPS"), (8, "drain")]:
    ax.axvline(xpos - 0.5, color="gray", ls=":", lw=1)
    ax.text(xpos - 0.4, ax.get_ylim()[1] * 0.02 + avg.sum(axis=1).max(), label, fontsize=9, color="gray")
ax.annotate("stampede:\n100% -> slow", xy=(4, 223), xytext=(3.0, 520),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate("stale-EMA lock-in:\n805/819 -> drowned slow sim", xy=(7, 805), xytext=(4.6, 760),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_xlabel("minute"); ax.set_ylabel("selections / min")
ax.set_title("avg-ttft + max-score-picker: winner-take-all flips and staleness lock-in")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "routing_avgttft_only.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- timeline 2
# avg-ttft (1.0) + inflight-requests (1.0) + max-score-picker
blend_t1 = np.array([
    [27, 61], [34, 130],        # 3 RPS: stable split (inflight term sends some to slow)
    [60, 152], [186, 363], [149, 365],   # 12 RPS: proportional overflow, no flips
    [231, 271], [315, 309], [453, 383],  # 40 RPS: balanced saturation
    [244, 139],                 # drain
])
blend_t2 = np.array([
    [87, 0], [73, 0], [106, 0], [180, 0],   # poisoned start: leaked counters lock out fast sim
    [255, 145], [47, 53], [258, 308], [355, 323], [413, 411],  # partial recovery
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), width_ratios=[1.1, 1])
for ax, data, title in [
    (axes[0], blend_t1, "treatment 1 (fresh state): stable proportional split"),
    (axes[1], blend_t2, "treatment 2 (leaked counters): fast sim locked out"),
]:
    x = np.arange(len(data))
    ax.bar(x, data[:, 0], color=C_SLOW, label=SLOW)
    ax.bar(x, data[:, 1], bottom=data[:, 0], color=C_FAST, label=FAST)
    ax.set_xlabel("minute"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("selections / min")
for xpos, label in [(0, "3 RPS"), (2, "12 RPS"), (5, "40 RPS")]:
    axes[0].axvline(xpos - 0.5, color="gray", ls=":", lw=1)
    axes[0].text(xpos - 0.4, blend_t1.sum(axis=1).max() * 1.02, label, fontsize=9, color="gray")
axes[1].annotate("in-flight counter leak from t1 failures:\nfast sim scores 0 despite being idle",
                 xy=(1.5, 90), xytext=(0.6, 420), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("avg-ttft + inflight blend + max-score-picker", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "routing_blended.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------ stage outcomes
stages = ["1\n3 RPS", "2\n3 RPS", "3\n12 RPS", "4\n12 RPS", "5\n40 RPS", "6\n40 RPS"]
fails_avg = [0, 0, 340, 413, 2129, 2345]
fails_blend = [0, 0, 182, 268, 2114, 2210]
ttft_avg = [1.0, 1.0, 25.5, 24.6, 24.3, 0.0]
ttft_blend = [3.0, 3.0, 22.1, 25.2, 24.2, 26.2]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(stages)); w = 0.38
axes[0].bar(x - w / 2, fails_avg, w, label="avg-ttft only", color="#1f77b4")
axes[0].bar(x + w / 2, fails_blend, w, label="ttft + inflight blend", color="#ff7f0e")
axes[0].set_xticks(x, stages); axes[0].set_ylabel("failed requests / stage")
axes[0].set_title("failures per stage (treatment 1)")
axes[0].annotate("blend cuts failures ~40%\nwhile small model saturated", xy=(2.2, 300),
                 xytext=(0.4, 1500), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[0].legend(fontsize=8)
axes[1].plot(x, ttft_avg, "o-", label="avg-ttft only", color="#1f77b4")
axes[1].plot(x, ttft_blend, "s-", label="ttft + inflight blend", color="#ff7f0e")
axes[1].set_xticks(x, stages); axes[1].set_ylabel("TTFT p90 of successes (s)")
axes[1].set_title("TTFT p90 per stage (treatment 1)")
axes[1].annotate("blend's cost while unsaturated:\ninflight term routes ~30% to the 3s sim",
                 xy=(0, 3.0), xytext=(0.3, 12), arrowprops=dict(arrowstyle="->"), fontsize=9)
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "stage_outcomes.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------ spike timeline
# 30s windows [slow, fast]; planner pinned to slow at 0.5 RPS, summarizer offers both
spike = np.array([
    [7, 0], [13, 0],                       # planner solo: 100% pinned (proof)
    [35, 0], [89, 0], [76, 0], [62, 0],    # summarizer joins; poisoned state -> all to slow
    [28, 18], [18, 0],                     # flip starts; summarizer t1 ends
    [0, 62], [2, 66], [14, 59], [18, 58],  # t2: summarizer -> fast, planner trickle pinned
    [11, 39], [15, 58], [5, 0],
])
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(spike))
ax.bar(x, spike[:, 0], color=C_SLOW, label=SLOW + "  [planner pinned here]")
ax.bar(x, spike[:, 1], bottom=spike[:, 0], color=C_FAST, label=FAST)
ax.set_ylim(0, 150)
ax.axvspan(-0.5, 1.5, color="gray", alpha=0.12)
ax.text(-0.3, 100, "planner solo:\n100% pinned", fontsize=9)
ax.annotate("summarizer joins; leaked-counter state\nsends ALL its traffic onto planner's model",
            xy=(3, 92), xytext=(2.2, 125), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate("healthy window: summarizer -> fast,\nplanner trickle stays pinned",
            xy=(10, 76), xytext=(9.0, 115), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_xlabel("30s window"); ax.set_ylabel("selections / 30s")
ax.set_title("parallel-harness spike: pinned planner + array-carrying summarizer", pad=12)
ax.legend(loc="center left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "spike_timeline.png", dpi=130); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob('*.png'))])
