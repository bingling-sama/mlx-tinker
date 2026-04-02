#!/usr/bin/env python3
"""Render an OpenClaw models.providers block for the local proxy."""

from __future__ import annotations

import argparse
import json

from integrations.openclaw_rl_local.runtime import ContainerRuntime, resolve_provider_base_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["podman", "docker", "host"], default="podman")
    parser.add_argument("--proxy-port", type=int, default=30000)
    parser.add_argument("--provider-id", default="mlx-tinker-local")
    parser.add_argument("--model-id", default="qwen3.5-local")
    parser.add_argument("--model-name", default="Qwen 3.5 Local")
    parser.add_argument("--provider-base-url", default=None)
    parser.add_argument("--api-key-env", default="MLX_TINKER_API_KEY")
    args = parser.parse_args()

    base_url = resolve_provider_base_url(
        ContainerRuntime(args.runtime),
        args.proxy_port,
        args.provider_base_url,
    )
    payload = {
        "mode": "merge",
        "providers": {
            args.provider_id: {
                "baseUrl": base_url,
                "apiKey": f"${{{args.api_key_env}}}",
                "api": "openai-completions",
                "models": [
                    {
                        "id": args.model_id,
                        "name": args.model_name,
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 32768,
                        "maxTokens": 4096,
                    }
                ],
            }
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
