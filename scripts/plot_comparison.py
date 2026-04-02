"""Plot comparison of RL vs Combined training runs.

Reads two JSON report files (from rl_run.py) and generates a 4-panel
comparison chart:
  1. Per-step reward (both methods overlaid)
  2. Training loss (both methods overlaid)
  3. Cumulative pass rate (both methods overlaid)
  4. Final evaluation comparison bar chart

Usage:
    uv run python scripts/plot_comparison.py
    uv run python scripts/plot_comparison.py --rl path/to/rl.json --combine path/to/combine.json
    uv run python scripts/plot_comparison.py --output assets/custom_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_comparison(
    rl_report: dict,
    combine_report: dict,
    output_path: str = "assets/openclaw_rl_vs_combined.png",
) -> None:
    model_name = rl_report.get("model_name", "Unknown")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"RL vs Combined Training on mlx-tinker ({model_name})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    reports = {"RL": rl_report, "Combined": combine_report}
    colors = {"RL": "#3498db", "Combined": "#e67e22"}
    markers = {"RL": "o", "Combined": "s"}

    # ---- Panel 1: Per-step reward ----
    ax1 = axes[0, 0]
    for label, report in reports.items():
        steps_data = report["steps"]
        steps = [s["step"] for s in steps_data]
        rewards = [s["mean_reward"] for s in steps_data]

        # Rolling average (window=3)
        w = min(3, len(rewards))
        rolling = np.convolve(rewards, np.ones(w) / w, mode="valid")
        rolling_x = steps[w - 1 :]

        ax1.bar(
            [s + (0.2 if label == "Combined" else -0.2) for s in steps],
            rewards,
            width=0.35,
            color=colors[label],
            alpha=0.4,
            label=f"{label} (step)",
        )
        if len(rolling) > 1:
            ax1.plot(
                rolling_x,
                rolling,
                color=colors[label],
                linewidth=2,
                marker=markers[label],
                markersize=4,
                label=f"{label} (rolling avg)",
            )

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Per-Step Reward")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_ylim(-1.3, 1.3)

    # ---- Panel 2: Training loss ----
    ax2 = axes[0, 1]
    for label, report in reports.items():
        steps_data = report["steps"]
        steps = [s["step"] for s in steps_data]
        losses = [s["losses"][0] if s["losses"] else 0 for s in steps_data]

        ax2.plot(
            steps,
            losses,
            color=colors[label],
            linewidth=2,
            marker=markers[label],
            markersize=4,
            label=label,
        )
        ax2.fill_between(steps, losses, alpha=0.1, color=colors[label])

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss (sum)")
    ax2.set_title("Training Loss")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    # ---- Panel 3: Cumulative pass rate ----
    ax3 = axes[1, 0]
    for label, report in reports.items():
        steps_data = report["steps"]
        steps = [s["step"] for s in steps_data]
        rewards = [s["mean_reward"] for s in steps_data]

        cum_passes = np.cumsum([1 if r > 0 else 0 for r in rewards])
        cum_total = np.arange(1, len(rewards) + 1)
        cum_rate = cum_passes / cum_total * 100

        ax3.plot(
            steps,
            cum_rate,
            color=colors[label],
            linewidth=2,
            marker=markers[label],
            markersize=4,
            label=f"{label} (training)",
        )
        ax3.fill_between(steps, cum_rate, alpha=0.1, color=colors[label])

        # Baseline and final eval markers
        baseline_pr = report["baseline_eval"]["pass_rate"] * 100
        final_pr = report["final_eval"]["pass_rate"] * 100
        ax3.axhline(
            y=baseline_pr,
            color=colors[label],
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )
        ax3.scatter(
            [steps[-1] + 1],
            [final_pr],
            color=colors[label],
            marker="*",
            s=150,
            zorder=5,
            label=f"{label} final eval: {final_pr:.0f}%",
        )

    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Cumulative Pass Rate (%)")
    ax3.set_title("Training Success + Final Eval")
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=7, loc="lower right")

    # ---- Panel 4: Final evaluation comparison ----
    ax4 = axes[1, 1]

    # Gather all unique task IDs from both final evals
    rl_final = {
        r["task_id"]: r for r in rl_report["final_eval"]["results"]
    }
    combine_final = {
        r["task_id"]: r for r in combine_report["final_eval"]["results"]
    }
    all_task_ids = list(dict.fromkeys(
        list(rl_final.keys()) + list(combine_final.keys())
    ))

    x = np.arange(len(all_task_ids))
    width = 0.35

    rl_rewards = [rl_final.get(t, {}).get("reward", 0) for t in all_task_ids]
    combine_rewards = [
        combine_final.get(t, {}).get("reward", 0) for t in all_task_ids
    ]

    rl_colors = ["#aed6f1" if r > 0 else "#f5b7b1" for r in rl_rewards]
    combine_colors = ["#fad7a0" if r > 0 else "#f5b7b1" for r in combine_rewards]

    bars1 = ax4.bar(x - width / 2, rl_rewards, width, color=rl_colors, edgecolor=colors["RL"], linewidth=1.5, label="RL")
    bars2 = ax4.bar(x + width / 2, combine_rewards, width, color=combine_colors, edgecolor=colors["Combined"], linewidth=1.5, label="Combined")

    short_names = [t.replace("_", "\n")[:20] for t in all_task_ids]
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, fontsize=6, rotation=45, ha="right")
    ax4.set_ylabel("Reward")
    ax4.set_title("Final Eval: Per-Task Comparison")
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.legend(fontsize=8)
    ax4.set_ylim(-1.3, 1.3)

    # Overall stats annotation
    rl_pr = rl_report["final_eval"]["pass_rate"] * 100
    combine_pr = combine_report["final_eval"]["pass_rate"] * 100
    rl_base = rl_report["baseline_eval"]["pass_rate"] * 100
    combine_base = combine_report["baseline_eval"]["pass_rate"] * 100

    stats_text = (
        f"RL:       {rl_base:.0f}% baseline -> {rl_pr:.0f}% final\n"
        f"Combined: {combine_base:.0f}% baseline -> {combine_pr:.0f}% final"
    )
    fig.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Comparison plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot RL vs Combined comparison")
    parser.add_argument(
        "--rl",
        default="workspace_reports/openclaw_local/rl_report.json",
        help="Path to RL report JSON",
    )
    parser.add_argument(
        "--combine",
        default="workspace_reports/openclaw_local/combine_report.json",
        help="Path to Combined report JSON",
    )
    parser.add_argument(
        "--output",
        default="assets/openclaw_rl_vs_combined.png",
        help="Output plot path",
    )
    args = parser.parse_args()

    rl_report = load_report(args.rl)
    combine_report = load_report(args.combine)
    plot_comparison(rl_report, combine_report, args.output)


if __name__ == "__main__":
    main()
