"""Configuration for the local OpenClaw RL integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .runtime import ContainerRuntime, RuntimeUrls, build_runtime_urls


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class LocalRLConfig:
    model_name: str = os.environ.get("OPENCLAW_LOCAL_MODEL_NAME", "Qwen/Qwen3.5-4B")
    method: str = os.environ.get("OPENCLAW_LOCAL_METHOD", "combine")
    loss_fn: str = os.environ.get("OPENCLAW_LOCAL_LOSS_FN", "ppo")
    curriculum_path: Path = Path(
        os.environ.get(
            "OPENCLAW_LOCAL_CURRICULUM",
            "integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml",
        )
    )
    container_runtime: ContainerRuntime = ContainerRuntime(
        os.environ.get("OPENCLAW_CONTAINER_RUNTIME", "docker").lower()
    )
    mlx_tinker_base_url: str = os.environ.get("MLX_TINKER_BASE_URL", "http://127.0.0.1:8010")
    mlx_tinker_api_key: str = os.environ.get("MLX_TINKER_API_KEY", "tml-local")
    proxy_host: str = os.environ.get("OPENCLAW_PROXY_HOST", "0.0.0.0")
    proxy_port: int = _env_int("OPENCLAW_PROXY_PORT", 30000)
    provider_base_url: str | None = os.environ.get("OPENCLAW_PROVIDER_BASE_URL")
    gateway_ws_url: str = os.environ.get("OPENCLAW_GATEWAY_WS_URL", "ws://127.0.0.1:18789")
    gateway_token: str | None = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
    served_model_name: str = os.environ.get("OPENCLAW_SERVED_MODEL_NAME", "qwen3.5-local")
    lora_rank: int = _env_int("OPENCLAW_LORA_RANK", 8)
    rollout_count: int = _env_int("OPENCLAW_RL_ROLLOUTS", 4)
    batch_size: int = _env_int("OPENCLAW_RL_BATCH_SIZE", 2)
    learning_rate: float = _env_float("OPENCLAW_RL_LR", 5e-5)
    max_steps: int = _env_int("OPENCLAW_RL_MAX_STEPS", 8)
    save_name_prefix: str = os.environ.get("OPENCLAW_RL_SAVE_PREFIX", "openclaw_local")
    request_timeout_ms: int = _env_int("OPENCLAW_GATEWAY_TIMEOUT_MS", 600_000)
    proxy_max_tokens: int = _env_int("OPENCLAW_PROXY_MAX_TOKENS", 512)
    train_max_prompt_tokens: int = _env_int("OPENCLAW_TRAIN_MAX_PROMPT_TOKENS", 256)
    train_max_response_tokens: int = _env_int("OPENCLAW_TRAIN_MAX_RESPONSE_TOKENS", 256)
    output_dir: Path = Path(
        os.environ.get("OPENCLAW_LOCAL_OUTPUT_DIR", "workspace_reports/openclaw_local")
    )

    # -- Teacher model (base model, no LoRA) for OPD / combine --
    teacher_model_name: str | None = os.environ.get("OPENCLAW_TEACHER_MODEL_NAME")

    # -- Combined method: advantage weights --
    w_opd: float = _env_float("OPENCLAW_COMBINE_W_OPD", 1.0)
    w_rl: float = _env_float("OPENCLAW_COMBINE_W_RL", 1.0)
    train_epochs: int = _env_int("OPENCLAW_TRAIN_EPOCHS", 1)

    # -- KL penalty --
    kl_loss_coef: float = _env_float("OPENCLAW_KL_LOSS_COEF", 0.02)

    # -- PRM / hint judge settings --
    prm_m: int = _env_int("OPENCLAW_PRM_M", 3)
    prm_temperature: float = _env_float("OPENCLAW_PRM_TEMPERATURE", 0.6)
    prm_max_tokens: int = _env_int("OPENCLAW_PRM_MAX_TOKENS", 2048)

    def resolved_teacher_model(self) -> str:
        """Return the teacher model name, defaulting to the policy model if unset."""
        return self.teacher_model_name or self.model_name

    @property
    def runtime_urls(self) -> RuntimeUrls:
        return build_runtime_urls(
            runtime=self.container_runtime,
            mlx_tinker_base_url=self.mlx_tinker_base_url,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            provider_base_url=self.provider_base_url,
        )
