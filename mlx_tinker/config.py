"""Engine and server configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


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
    gradient_checkpointing: bool = True
