"""Engine and server configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class LRScheduleConfig(BaseModel):
    """Learning rate schedule configuration."""

    warmup_steps: int = 0
    schedule: Literal["constant", "cosine", "linear"] = "constant"
    min_lr_ratio: float = 0.1
    total_steps: int = 0


class EngineConfig(BaseModel):
    """Configuration for the MLX-Tinker engine."""

    # Model
    base_model: str = "Qwen/Qwen3.5-0.8B"
    quantize_bits: int = 4
    quantize_group_size: int = 64

    # LoRA defaults
    default_lora_rank: int = 16
    default_lora_alpha: float = 32.0

    # Engine
    engine_cycle_ms: int = 100
    max_batch_size: int = 8

    # Training
    optimizer_type: Literal["adamw_8bit", "adamw", "adafactor", "lion"] = "adamw"
    gradient_checkpointing: bool = True
    lr_schedule: LRScheduleConfig = LRScheduleConfig()

    # Checkpoints
    checkpoints_base: Path = Path("checkpoints")

    # Database
    database_path: Path = Path("tinker.db")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Session management
    session_cleanup_interval_sec: int = 60
    session_timeout_sec: int = 300

    # Memory
    max_kv_cache_size: int | None = None
    kv_cache_bits: int | None = 4
    kv_cache_group_size: int = 64
    quantized_kv_start: int = 0
    prefix_cache_disk_limit_gb: float = 2.0
