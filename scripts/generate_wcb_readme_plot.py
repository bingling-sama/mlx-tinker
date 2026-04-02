#!/usr/bin/env python3
"""Generate a README plot for WildClawBench RL with OpenClaw + OpenClaw-RL.

Parses a training log from a WildClawBench/OpenClaw-RL run and produces a
compact figure showing reward progression during training.

Usage:
    uv run python scripts/generate_wcb_readme_plot.py
    uv run python scripts/generate_wcb_readme_plot.py \
        --train-log workspace_reports/openclaw_upstream/runs/<run>/train/logs/train.log \
        --output assets/wcb_openclaw_rl_learning.png
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_LOG = ROOT / (
    "workspace_reports/openclaw_upstream/runs/"
    "wcb_full_2b_37tasks_p4_20260402_022452/train/logs/train.log"
)
DEFAULT_OUTPUT = ROOT / "assets/wcb_openclaw_rl_learning.png"

NEG_COLOR = "#B3261E"
NEU_COLOR = "#7A7A7A"
POS_COLOR = "#1B7F3B"
LINE_COLOR = "#0B57D0"
BG_COLOR = "#FAF8F4"


def rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    q: deque[float] = deque()
    total = 0.0
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.popleft()
        out.append(total / len(q))
    return out


def parse_train_log(path: Path) -> dict:
    step_rows: list[dict] = []
    submitted = 0
    included = 0
    excluded = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    step_done_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[Trainer\] step "
        r"(\d+) done \| batch=(\d+) mean_reward=([-.0-9]+) success=([.0-9]+)"
    )
    submitted_re = re.compile(r"submitted session=.* exclude=(True|False)")
    ts_re = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ts_match = ts_re.match(line)
        if ts_match:
            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        m = step_done_re.search(line)
        if m:
            step_rows.append(
                {
                    "step": int(m.group(2)),
                    "batch": int(m.group(3)),
                    "mean_reward": float(m.group(4)),
                    "success": float(m.group(5)),
                    "timestamp": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                }
            )

        m = submitted_re.search(line)
        if m:
            submitted += 1
            if m.group(1) == "True":
                excluded += 1
            else:
                included += 1

    if not step_rows:
        raise ValueError(f"No trainer step data found in {path}")

    if first_ts is None or last_ts is None:
        raise ValueError(f"No timestamps found in {path}")

    return {
        "steps": step_rows,
        "submitted": submitted,
        "included": included,
        "excluded": excluded,
        "duration": last_ts - first_ts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WildClawBench RL README plot")
    parser.add_argument("--train-log", type=str, default=str(DEFAULT_TRAIN_LOG))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = parse_train_log(Path(args.train_log))
    steps = data["steps"]

    x = [row["step"] for row in steps]
    rewards = [row["mean_reward"] for row in steps]
    success = [row["success"] for row in steps]
    reward_avg = rolling_mean(rewards, window=5)
    success_avg = rolling_mean(success, window=5)

    pos_cum = np.cumsum([1 if r > 0 else 0 for r in rewards])
    neu_cum = np.cumsum([1 if r == 0 else 0 for r in rewards])
    neg_cum = np.cumsum([1 if r < 0 else 0 for r in rewards])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_reward, ax_mix) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor(BG_COLOR)

    bar_colors = [POS_COLOR if r > 0 else NEG_COLOR if r < 0 else NEU_COLOR for r in rewards]
    ax_reward.bar(x, rewards, color=bar_colors, width=0.82, alpha=0.9)
    ax_reward.plot(x, reward_avg, color=LINE_COLOR, linewidth=2.4, label="5-step moving average")
    ax_reward.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    ax_reward.set_title("WildClawBench Reward During RL", fontsize=12, fontweight="bold")
    ax_reward.set_xlabel("Training Step")
    ax_reward.set_ylabel("Mean Reward")
    ax_reward.set_ylim(-1.15, 1.15)
    ax_reward.set_yticks([-1, 0, 1])
    ax_reward.legend(loc="lower right", fontsize=9)

    first_positive = next((row["step"] for row in steps if row["mean_reward"] > 0), None)
    if first_positive is not None:
        ax_reward.annotate(
            f"first positive step: {first_positive}",
            xy=(first_positive, 1.0),
            xytext=(first_positive + 1.5, 0.7),
            arrowprops={"arrowstyle": "->", "color": POS_COLOR, "lw": 1.4},
            fontsize=9,
            color=POS_COLOR,
        )

    ax_mix.plot(x, neg_cum, color=NEG_COLOR, linewidth=2.0, label="Negative-reward steps")
    ax_mix.plot(x, neu_cum, color=NEU_COLOR, linewidth=2.0, label="Zero-reward steps")
    ax_mix.plot(x, pos_cum, color=POS_COLOR, linewidth=2.0, label="Positive-reward steps")
    ax_mix.set_title("Outcome Mix Shifts Over Time", fontsize=12, fontweight="bold")
    ax_mix.set_xlabel("Training Step")
    ax_mix.set_ylabel("Cumulative Step Count")

    ax_mix_r = ax_mix.twinx()
    ax_mix_r.plot(
        x,
        success_avg,
        color=LINE_COLOR,
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        label="5-step success rate",
    )
    ax_mix_r.set_ylabel("Success Rate")
    ax_mix_r.set_ylim(-0.02, 1.02)

    lines1, labels1 = ax_mix.get_legend_handles_labels()
    lines2, labels2 = ax_mix_r.get_legend_handles_labels()
    ax_mix.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

    duration = data["duration"]
    footer = (
        "OpenClaw agent episodes -> OpenClaw-RL proxy scoring -> mlx-tinker local PPO updates\n"
        f"{len(steps)} PPO steps | {data['submitted']} scored trajectories "
        f"({data['included']} trainable / {data['excluded']} excluded) | "
        f"duration {duration}"
    )
    fig.suptitle(
        "End-to-End Tool-Use RL on WildClawBench with OpenClaw + OpenClaw-RL",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(0.5, -0.02, footer, ha="center", fontsize=9, color="#444444")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
