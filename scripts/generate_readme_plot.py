#!/usr/bin/env python3
"""Generate the README benchmark comparison plot.

Reads Tinker (official) and mlx-tinker SFT benchmark JSONs and produces a
comparison figure: loss curves + step-time comparison for 4B (cloud vs local),
plus 9B local-only training curve.

Usage:
    uv run python scripts/generate_readme_plot.py
    uv run python scripts/generate_readme_plot.py \
        --cloud-4b workspace_reports/readme_benchmark_4b_cloud/tinker_results.json \
        --mlx-4b   workspace_reports/readme_benchmark_4b/mlx_results.json \
        --mlx-9b   workspace_reports/readme_benchmark_9b/mlx_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_COLOR = "#1565C0"
LOCAL_COLOR = "#E65100"
LOCAL_9B_COLOR = "#2E7D32"


def load_sft_logs(path: Path, max_steps: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    logs = data["sft"]["training_logs"]
    if max_steps is not None:
        logs = logs[:max_steps]
    return logs


def load_sft_summary(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["sft"]


def extract(logs: list[dict]) -> tuple[list[int], list[float], list[float]]:
    steps = [e["step"] for e in logs]
    losses = [e["loss_sum"] for e in logs]
    times = [e["step_time_s"] for e in logs]
    return steps, losses, times


def main():
    parser = argparse.ArgumentParser(description="Generate README benchmark plot")
    parser.add_argument(
        "--cloud-4b",
        type=str,
        default=str(ROOT / "workspace_reports/readme_benchmark_4b_cloud/tinker_results.json"),
    )
    parser.add_argument(
        "--mlx-4b",
        type=str,
        default=str(ROOT / "workspace_reports/readme_benchmark_4b/mlx_results.json"),
    )
    parser.add_argument(
        "--mlx-9b",
        type=str,
        default=str(ROOT / "workspace_reports/readme_benchmark_9b/mlx_results.json"),
    )
    parser.add_argument("--steps", type=int, default=50, help="Max steps to plot")
    parser.add_argument(
        "--output", type=str, default=str(ROOT / "assets/benchmark_sft_comparison.png")
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_loss, ax_time, ax_9b = axes

    # Load data
    cloud_4b = load_sft_logs(Path(args.cloud_4b), args.steps)
    mlx_4b = load_sft_logs(Path(args.mlx_4b), args.steps)
    mlx_9b = load_sft_logs(Path(args.mlx_9b), args.steps)

    c_steps, c_losses, c_times = extract(cloud_4b)
    m4_steps, m4_losses, m4_times = extract(mlx_4b)
    m9_steps, m9_losses, m9_times = extract(mlx_9b)

    # --- Panel 1: 4B Loss Curves (Cloud vs Local) ---
    ax_loss.semilogy(
        c_steps, c_losses, color=OFFICIAL_COLOR, linewidth=1.6, alpha=0.85, label="Tinker (official)"
    )
    ax_loss.semilogy(
        m4_steps,
        m4_losses,
        color=LOCAL_COLOR,
        linewidth=1.6,
        alpha=0.85,
        label="mlx-tinker (Apple Silicon)",
    )
    ax_loss.set_xlabel("Training Step", fontsize=10)
    ax_loss.set_ylabel("Loss (log scale)", fontsize=10)
    ax_loss.set_title("Qwen3.5-4B  \u2014  SFT Loss", fontsize=11, fontweight="bold")
    ax_loss.legend(fontsize=8.5, loc="upper right")
    ax_loss.grid(True, alpha=0.3, linewidth=0.5)
    ax_loss.tick_params(labelsize=8)

    # --- Panel 2: 4B Step Time (Cloud vs Local) ---
    ax_time.scatter(c_steps, c_times, color=OFFICIAL_COLOR, s=14, alpha=0.5, zorder=3)
    ax_time.scatter(m4_steps, m4_times, color=LOCAL_COLOR, s=14, alpha=0.5, zorder=3)

    c_mean = np.mean(c_times)
    m4_mean = np.mean(m4_times)
    ax_time.axhline(
        c_mean,
        color=OFFICIAL_COLOR,
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label=f"Official avg: {c_mean:.1f}s",
    )
    ax_time.axhline(
        m4_mean,
        color=LOCAL_COLOR,
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label=f"Local avg: {m4_mean:.1f}s",
    )

    ax_time.set_xlabel("Training Step", fontsize=10)
    ax_time.set_ylabel("Step Time (seconds)", fontsize=10)
    ax_time.set_title("Qwen3.5-4B  \u2014  Step Time", fontsize=11, fontweight="bold")
    ax_time.legend(fontsize=8.5, loc="upper right")
    ax_time.grid(True, alpha=0.3, linewidth=0.5)
    ax_time.tick_params(labelsize=8)

    # --- Panel 3: All Models Loss (Local) ---
    ax_9b.semilogy(
        m4_steps,
        m4_losses,
        color=LOCAL_COLOR,
        linewidth=1.6,
        alpha=0.85,
        label="Qwen3.5-4B (17.6 GB)",
    )
    ax_9b.semilogy(
        m9_steps,
        m9_losses,
        color=LOCAL_9B_COLOR,
        linewidth=1.6,
        alpha=0.85,
        label="Qwen3.5-9B (36.4 GB)",
    )
    ax_9b.set_xlabel("Training Step", fontsize=10)
    ax_9b.set_ylabel("Loss (log scale)", fontsize=10)
    ax_9b.set_title("mlx-tinker  \u2014  Model Comparison", fontsize=11, fontweight="bold")
    ax_9b.legend(fontsize=8.5, loc="upper right")
    ax_9b.grid(True, alpha=0.3, linewidth=0.5)
    ax_9b.tick_params(labelsize=8)

    # Suptitle
    fig.suptitle(
        "SFT on WikiSQL  \u2014  50 steps, QLoRA rank-8, 4-bit quantization, batch_size=2",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    # Footer with summary stats
    m4_summary = load_sft_summary(Path(args.mlx_4b))
    m9_summary = load_sft_summary(Path(args.mlx_9b))
    c4_summary = load_sft_summary(Path(args.cloud_4b))
    c4_t = c4_summary["total_train_time_s"]
    c4_a = c4_summary["eval_accuracy"] * 100
    m4_t = m4_summary["total_train_time_s"]
    m4_m = m4_summary["peak_memory_gb"]
    m4_a = m4_summary["eval_accuracy"] * 100
    m9_t = m9_summary["total_train_time_s"]
    m9_m = m9_summary["peak_memory_gb"]
    m9_a = m9_summary["eval_accuracy"] * 100
    footer = (
        f"Tinker (official) (4B): {c4_t:.0f}s, {c4_a:.0f}% acc  |  "
        f"mlx-tinker (4B): {m4_t:.0f}s, {m4_m:.1f} GB, {m4_a:.0f}% acc  |  "
        f"mlx-tinker (9B): {m9_t:.0f}s, {m9_m:.1f} GB, {m9_a:.0f}% acc"
    )
    fig.text(0.5, -0.04, footer, ha="center", fontsize=8, color="#555555")

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
