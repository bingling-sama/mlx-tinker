"""CLI entrypoint for the local OpenClaw RL integration.

Supports all three methods via --method {rl, opd, combine}.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .config import LocalRLConfig
from .runtime import ContainerRuntime
from .trainer import LocalRLTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local OpenClaw RL trainer for mlx-tinker")

    # Method
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--method",
        choices=["rl", "opd", "combine"],
        default="combine",
        help="Training method: rl, opd, or combine (default: combine)",
    )
    parser.add_argument(
        "--loss-fn",
        choices=["ppo", "importance_sampling", "cispo"],
        default="ppo",
        help="Loss function (default: ppo)",
    )

    # Curriculum and infrastructure
    parser.add_argument(
        "--curriculum",
        default="integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml",
    )
    parser.add_argument("--runtime", choices=["podman", "docker", "host"], default="docker")
    parser.add_argument("--mlx-tinker-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--gateway-ws-url", default="ws://127.0.0.1:18789")
    parser.add_argument("--gateway-token", default=None)
    parser.add_argument("--proxy-host", default="0.0.0.0")
    parser.add_argument("--proxy-port", type=int, default=30000)
    parser.add_argument("--provider-base-url", default=None)
    parser.add_argument("--served-model-name", default="qwen3.5-local")

    # Training
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=8)

    # Combined method weights
    parser.add_argument(
        "--w-opd",
        type=float,
        default=1.0,
        help="OPD advantage weight (combine method only, default: 1.0)",
    )
    parser.add_argument(
        "--w-rl",
        type=float,
        default=1.0,
        help="RL advantage weight (combine method only, default: 1.0)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Duplicate samples N times per batch (combine typically uses 2, default: 1)",
    )

    # Teacher / KL
    parser.add_argument(
        "--teacher-model-name",
        default=None,
        help="Teacher model name (defaults to same as --model-name)",
    )
    parser.add_argument(
        "--kl-loss-coef",
        type=float,
        default=0.02,
        help="KL penalty coefficient (default: 0.02)",
    )

    # Logging and output
    parser.add_argument("--output-dir", default="workspace_reports/openclaw_local")
    parser.add_argument("--log-level", default="INFO")
    return parser


async def _main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = LocalRLConfig(
        model_name=args.model_name,
        method=args.method,
        loss_fn=args.loss_fn,
        curriculum_path=Path(args.curriculum),
        container_runtime=ContainerRuntime(args.runtime),
        mlx_tinker_base_url=args.mlx_tinker_base_url,
        gateway_ws_url=args.gateway_ws_url,
        gateway_token=args.gateway_token,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        provider_base_url=args.provider_base_url,
        served_model_name=args.served_model_name,
        lora_rank=args.lora_rank,
        rollout_count=args.rollouts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        w_opd=args.w_opd,
        w_rl=args.w_rl,
        train_epochs=args.train_epochs,
        teacher_model_name=args.teacher_model_name,
        kl_loss_coef=args.kl_loss_coef,
        output_dir=Path(args.output_dir),
    )
    trainer = LocalRLTrainer(config)
    await trainer.run()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
