#!/usr/bin/env python3
"""CLI entrypoint for the Hermes live-RL bridge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .config import HermesRLConfig
from .trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> HermesRLConfig:
    parser = argparse.ArgumentParser(
        description="Hermes live RL on mlx-tinker (RL / OPD / Combined)"
    )
    parser.add_argument(
        "--method",
        default=os.getenv("HERMES_RL_METHOD", "rl"),
        choices=["rl", "opd", "combine"],
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("HERMES_RL_MODEL_NAME", "Qwen/Qwen3.5-4B"),
    )
    parser.add_argument(
        "--served-model-name",
        default=os.getenv("HERMES_RL_SERVED_MODEL", "hermes-local"),
    )
    parser.add_argument(
        "--tinker-base-url",
        default=os.getenv("TINKER_BASE_URL", "http://127.0.0.1:8010"),
    )
    parser.add_argument(
        "--tinker-api-key",
        default=os.getenv("TINKER_API_KEY", os.getenv("MLX_TINKER_API_KEY", "tml-local")),
    )
    parser.add_argument(
        "--proxy-api-key",
        default=os.getenv("HERMES_RL_PROXY_API_KEY", ""),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=int(os.getenv("HERMES_RL_LORA_RANK", "32")),
    )
    parser.add_argument(
        "--teacher-model-name",
        default=os.getenv("HERMES_RL_TEACHER_MODEL", ""),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("HERMES_RL_LR", "1e-4")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("HERMES_RL_BATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("HERMES_RL_MAX_STEPS", "1000")),
    )
    parser.add_argument("--loss-fn", default=os.getenv("HERMES_RL_LOSS_FN", "ppo"))
    parser.add_argument(
        "--kl-loss-coef",
        type=float,
        default=float(os.getenv("HERMES_RL_KL_LOSS_COEF", "0.0")),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=int(os.getenv("HERMES_RL_SAVE_INTERVAL", "20")),
    )
    parser.add_argument(
        "--resume-from-ckpt",
        default=os.getenv("HERMES_RL_RESUME_FROM_CKPT", ""),
    )
    parser.add_argument("--w-opd", type=float, default=float(os.getenv("HERMES_RL_W_OPD", "1.0")))
    parser.add_argument("--w-rl", type=float, default=float(os.getenv("HERMES_RL_W_RL", "1.0")))
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=int(os.getenv("HERMES_RL_TRAIN_EPOCHS", "1")),
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        default=os.getenv("HERMES_RL_EVAL_MODE", "0").strip().lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--prm-m", type=int, default=int(os.getenv("HERMES_RL_PRM_M", "3")))
    parser.add_argument(
        "--prm-temperature",
        type=float,
        default=float(os.getenv("HERMES_RL_PRM_TEMPERATURE", "0.6")),
    )
    parser.add_argument(
        "--prm-max-tokens",
        type=int,
        default=int(os.getenv("HERMES_RL_PRM_MAX_TOKENS", "4096")),
    )
    parser.add_argument(
        "--proxy-host",
        default=os.getenv("HERMES_RL_PROXY_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=int(os.getenv("HERMES_RL_PROXY_PORT", "30050")),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=int(os.getenv("HERMES_RL_MAX_CONTEXT_TOKENS", "8192")),
    )
    parser.add_argument(
        "--record-dir",
        default=os.getenv("HERMES_RL_RECORD_DIR", "records/"),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("HERMES_RL_WANDB_PROJECT", "hermes-tinker"),
    )

    args = parser.parse_args()
    return HermesRLConfig(
        method=args.method,
        model_name=args.model_name,
        served_model_name=args.served_model_name,
        tinker_base_url=args.tinker_base_url,
        tinker_api_key=args.tinker_api_key,
        proxy_api_key=args.proxy_api_key,
        lora_rank=args.lora_rank,
        teacher_model_name=args.teacher_model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        loss_fn=args.loss_fn,
        kl_loss_coef=args.kl_loss_coef,
        save_interval=args.save_interval,
        resume_from_ckpt=args.resume_from_ckpt,
        w_opd=args.w_opd,
        w_rl=args.w_rl,
        train_epochs=args.train_epochs,
        eval_mode=args.eval_mode,
        prm_m=args.prm_m,
        prm_temperature=args.prm_temperature,
        prm_max_tokens=args.prm_max_tokens,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        max_context_tokens=args.max_context_tokens,
        record_dir=args.record_dir,
        wandb_project=args.wandb_project,
    )


def main() -> None:
    config = parse_args()
    if not config.tinker_api_key:
        print("ERROR: TINKER_API_KEY or --tinker-api-key is required.", file=sys.stderr)
        sys.exit(1)
    trainer = Trainer(config)
    try:
        asyncio.run(trainer.run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("[main] Ctrl+C received, cleaning up")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
