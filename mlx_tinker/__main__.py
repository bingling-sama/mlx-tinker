"""CLI entry point: python -m mlx_tinker"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from mlx_tinker.config import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX-Tinker: Tinker API backend for Apple Silicon")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Base model name or path")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--db", default="tinker.db", help="Database path")
    parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--quantize-bits", type=int, default=4, help="Quantization bits (0=none)")
    parser.add_argument("--lora-rank", type=int, default=16, help="Default LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="Default LoRA alpha")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument("--cycle-ms", type=int, default=100, help="Engine cycle time in ms")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = EngineConfig(
        base_model=args.model,
        host=args.host,
        port=args.port,
        database_path=args.db,
        checkpoints_base=args.checkpoints,
        quantize_bits=args.quantize_bits,
        default_lora_rank=args.lora_rank,
        default_lora_alpha=args.lora_alpha,
        max_batch_size=args.max_batch_size,
        engine_cycle_ms=args.cycle_ms,
    )

    from mlx_tinker.api.server import create_app

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
