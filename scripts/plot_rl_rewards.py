"""Plot RL training rewards and loss curve from local_rl_report.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rl_report(report_path: str, output_path: str = "assets/openclaw_rl_curve.png") -> None:
    with open(report_path) as f:
        report = json.load(f)

    steps_data = report["steps"]
    final = report["final_eval"]

    steps = [s["step"] for s in steps_data]
    rewards = [s["mean_reward"] for s in steps_data]
    losses = [s["losses"][0] if s["losses"] else 0 for s in steps_data]

    method = report.get("method", "rl").upper()
    method_label = {"RL": "RL (PPO)", "COMBINE": "Combined (OPD+RL)"}.get(method, method)
    rollouts_per_step = report.get("rollouts_per_step", 4)

    # Compute rolling mean reward (window=3)
    rolling_window = min(3, len(rewards))
    rolling_reward = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode="valid")
    rolling_steps = steps[rolling_window - 1 :]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1.3]})
    fig.suptitle(
        f"OpenClaw-RL {method_label} on mlx-tinker ({report['model_name']})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # --- Panel 1: Per-step reward ---
    ax1 = axes[0, 0]
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in rewards]
    ax1.bar(steps, rewards, color=colors, alpha=0.7, width=0.6, label="Step reward")
    if len(rolling_reward) > 1:
        ax1.plot(rolling_steps, rolling_reward, "k-o", linewidth=2, markersize=4, label=f"Rolling avg (w={rolling_window})")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Per-Step Reward")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-1.3, 1.3)

    # --- Panel 2: Loss curve ---
    ax2 = axes[0, 1]
    ax2.plot(steps, losses, "b-o", linewidth=2, markersize=5)
    ax2.fill_between(steps, losses, alpha=0.15, color="blue")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss (sum)")
    ax2.set_title("Training Loss")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # --- Bottom row: table spans both columns ---
    axes[1, 0].remove()
    axes[1, 1].remove()
    ax_table = fig.add_subplot(2, 1, 2)
    ax_table.axis("off")
    ax_table.set_title("Per-Task Results", fontsize=11, pad=15)

    # Shorter task names for the table
    short_names = {
        "Arxiv Paper Digest": "Arxiv Digest",
        "Pdf Batch Classification": "PDF Classification",
        "Conflicting Information Resolution": "Conflict Resolution",
        "Financial Data Extraction": "Financial Extract.",
        "Escalation Routing": "Escalation Route",
        "Cross Department Updates": "Cross-Dept Update",
    }

    table_data = []
    for s in steps_data:
        raw_name = s["task_id"].replace("_", " ").title()
        task = short_names.get(raw_name, raw_name[:22])
        result = "Pass" if s["mean_reward"] > 0 else "Fail"
        loss = f"{s['losses'][0]:.0f}" if s["losses"] else "N/A"
        rollouts = f"{s['passed_rollouts']}/{rollouts_per_step}"
        table_data.append([s["step"], task, result, rollouts, loss])

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Step", "Task", "Result", "Rollouts", "Loss"],
        cellLoc="center",
        loc="upper center",
        colWidths=[0.06, 0.22, 0.08, 0.10, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor("#d5d8dc")
        table[0, j].set_text_props(fontweight="bold")

    # Color code result rows
    for i, row in enumerate(table_data):
        color = "#d5f5e3" if row[2] == "Pass" else "#fadbd8"
        for j in range(len(row)):
            table[i + 1, j].set_facecolor(color)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else "workspace_reports/openclaw_local/local_rl_report.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "assets/openclaw_rl_curve.png"
    plot_rl_report(report_path, output_path)
