from __future__ import annotations

from unittest.mock import MagicMock

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import AdamParams, LoraConfig, OptimStepInput
from tests.helpers import FakeTokenizer, TinyModel


def _make_backend(tmp_path):
    backend = MLXBackend(
        EngineConfig(
            base_model="test-model",
            checkpoints_base=tmp_path / "checkpoints",
            database_path=tmp_path / "tinker.db",
        )
    )
    backend.models["model-1"] = TinyModel()
    backend.tokenizers["model-1"] = FakeTokenizer()
    backend.lora_configs["model-1"] = LoraConfig(rank=8)
    return backend


def test_optim_step_clears_cached_sampling_models(tmp_path):
    backend = _make_backend(tmp_path)
    backend.current_loaded_sampler_path = "cached-path"
    backend.training.optim_step = MagicMock(return_value=MagicMock(metrics={"ok": 1.0}))

    backend.optim_step(
        "model-1",
        OptimStepInput(adam_params=AdamParams(learning_rate=1e-3)),
    )

    assert backend.current_loaded_sampler_path is None


def test_unload_model_clears_cached_sampling_models(tmp_path):
    backend = _make_backend(tmp_path)
    backend.current_loaded_sampler_path = "cached-path"

    backend.unload_model("model-1", MagicMock())

    assert backend.current_loaded_sampler_path is None


def test_optim_step_invalidates_student_transcript_cache(tmp_path):
    backend = _make_backend(tmp_path)
    backend.training.optim_step = MagicMock(return_value=MagicMock(metrics={"ok": 1.0}))
    backend.transcript_cache.invalidate_namespace_prefix = MagicMock()

    backend.optim_step(
        "model-1",
        OptimStepInput(adam_params=AdamParams(learning_rate=1e-3)),
    )

    backend.transcript_cache.invalidate_namespace_prefix.assert_called_with("student:")


def test_unload_model_invalidates_student_transcript_cache(tmp_path):
    backend = _make_backend(tmp_path)
    backend.transcript_cache.invalidate_namespace_prefix = MagicMock()

    backend.unload_model("model-1", MagicMock())

    backend.transcript_cache.invalidate_namespace_prefix.assert_called_with("student:")
