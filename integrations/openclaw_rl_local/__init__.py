"""Full-local OpenClaw continual-learning integration for mlx-tinker.

Supports three training methods:
  - rl: Binary RL with GRPO advantages
  - opd: On-Policy Distillation with teacher logprobs
  - combine: Combined OPD + Binary RL (recommended)
"""

from .config import LocalRLConfig
from .scorers import LocalTeacherLogprobExtractor
from .trainer import LocalRLTrainer

__all__ = ["LocalRLConfig", "LocalRLTrainer", "LocalTeacherLogprobExtractor"]
