"""Configuration for the Hermes live-RL bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


@dataclass
class HermesRLConfig:
    """Configuration shared by the Hermes proxy and trainer."""

    # Method
    method: str = os.environ.get("HERMES_RL_METHOD", "rl")

    # Model
    model_name: str = os.environ.get("HERMES_RL_MODEL_NAME", "Qwen/Qwen3.5-4B")
    served_model_name: str = os.environ.get("HERMES_RL_SERVED_MODEL", "hermes-local")
    lora_rank: int = _env_int("HERMES_RL_LORA_RANK", 32)
    teacher_model_name: str = os.environ.get("HERMES_RL_TEACHER_MODEL", "")

    # mlx-tinker backend
    tinker_base_url: str = os.environ.get("TINKER_BASE_URL", "http://127.0.0.1:8010")
    tinker_api_key: str = os.environ.get(
        "TINKER_API_KEY",
        os.environ.get("MLX_TINKER_API_KEY", "tml-local"),
    )

    # Training
    learning_rate: float = _env_float("HERMES_RL_LR", 1e-4)
    batch_size: int = _env_int("HERMES_RL_BATCH_SIZE", 4)
    max_steps: int = _env_int("HERMES_RL_MAX_STEPS", 1000)
    loss_fn: str = os.environ.get("HERMES_RL_LOSS_FN", "ppo")
    kl_loss_coef: float = _env_float("HERMES_RL_KL_LOSS_COEF", 0.0)
    save_weights_timeout: float = _env_float("HERMES_RL_SAVE_TIMEOUT", 200.0)
    save_interval: int = _env_int("HERMES_RL_SAVE_INTERVAL", 20)
    resume_from_ckpt: str = os.environ.get("HERMES_RL_RESUME_FROM_CKPT", "")
    train_epochs: int = _env_int("HERMES_RL_TRAIN_EPOCHS", 1)

    # Combined / OPD weights
    w_opd: float = _env_float("HERMES_RL_W_OPD", 1.0)
    w_rl: float = _env_float("HERMES_RL_W_RL", 1.0)
    eval_mode: bool = os.environ.get("HERMES_RL_EVAL_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    # PRM / hint judge
    prm_m: int = _env_int("HERMES_RL_PRM_M", 3)
    prm_temperature: float = _env_float("HERMES_RL_PRM_TEMPERATURE", 0.6)
    prm_max_tokens: int = _env_int("HERMES_RL_PRM_MAX_TOKENS", 4096)

    # Proxy
    proxy_host: str = os.environ.get("HERMES_RL_PROXY_HOST", "0.0.0.0")
    proxy_port: int = _env_int("HERMES_RL_PROXY_PORT", 30050)
    proxy_api_key: str = os.environ.get("HERMES_RL_PROXY_API_KEY", "")
    max_context_tokens: int = _env_int("HERMES_RL_MAX_CONTEXT_TOKENS", 8192)

    # Logging / output
    record_dir: str = os.environ.get("HERMES_RL_RECORD_DIR", "records/")
    wandb_project: str = os.environ.get("HERMES_RL_WANDB_PROJECT", "hermes-tinker")

    def resolved_teacher_model(self) -> str:
        return self.teacher_model_name or self.model_name


TinkerConfig = HermesRLConfig
