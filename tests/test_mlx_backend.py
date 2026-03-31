from __future__ import annotations

from unittest.mock import patch

from mlx_tinker.backend.gradient_checkpointing import enable_gradient_checkpointing
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import CreateModelInput, LoraConfig
from tests.helpers import FakeTokenizer, TinyLM


def _backend(tmp_path, gradient_checkpointing: bool) -> MLXBackend:
    return MLXBackend(
        EngineConfig(
            base_model="test-model",
            checkpoints_base=tmp_path / "checkpoints",
            database_path=tmp_path / "tinker.db",
            gradient_checkpointing=gradient_checkpointing,
        )
    )


def test_create_model_enables_gradient_checkpointing_when_configured(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=True)

    with patch(
        "mlx_tinker.backend.mlx_backend.mlx_load",
        side_effect=[(TinyLM(), FakeTokenizer()), (TinyLM(), FakeTokenizer())],
    ):
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )

    assert enable_gradient_checkpointing(backend.models["model-1"]) == 0


def test_create_model_leaves_gradient_checkpointing_off_when_disabled(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with patch(
        "mlx_tinker.backend.mlx_backend.mlx_load",
        side_effect=[(TinyLM(), FakeTokenizer()), (TinyLM(), FakeTokenizer())],
    ):
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )

    assert enable_gradient_checkpointing(backend.models["model-1"]) == len(
        backend.models["model-1"].layers
    )


def test_create_model_enables_longlora_when_requested(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with (
        patch(
            "mlx_tinker.backend.mlx_backend.mlx_load",
            side_effect=[(TinyLM(), FakeTokenizer()), (TinyLM(), FakeTokenizer())],
        ),
        patch("mlx_tinker.backend.mlx_backend.enable_longlora_attention") as enable_longlora,
    ):
        backend.create_model(
            "model-1",
            CreateModelInput(
                lora_config=LoraConfig(
                    rank=4,
                    alpha=8.0,
                    use_longlora=True,
                    longlora_group_size_ratio=0.25,
                )
            ),
        )

    enable_longlora.assert_called_once_with(backend.models["model-1"], group_size_ratio=0.25)
