from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlx_tinker.backend.gradient_checkpointing import enable_gradient_checkpointing
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    CreateModelInput,
    EncodedTextChunk,
    GeneratedSequence,
    LoraConfig,
    ModelInput,
    SampleInput,
    SampleOutput,
    SamplingParams,
)
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


def _sample_request(**updates) -> SampleInput:
    payload = {
        "prompt": ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
        "sampling_params": SamplingParams(temperature=0.0, max_tokens=4),
    }
    payload.update(updates)
    return SampleInput(**payload)


def _sample_output() -> SampleOutput:
    return SampleOutput(
        sequences=[GeneratedSequence(stop_reason="stop", tokens=[4], logprobs=[0.0])]
    )


def test_create_model_enables_gradient_checkpointing_when_configured(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=True)

    with patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())):
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )

    assert enable_gradient_checkpointing(backend.models["model-1"]) == 0


def test_create_model_leaves_gradient_checkpointing_off_when_disabled(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())):
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
        patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())),
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


def test_teacher_sampling_uses_same_model_with_zeroed_scales_and_restores_afterward(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())):
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )

    model = backend.models["model-1"]
    original_scales = backend.lora_manager.get_lora_scales(model)
    observed_scales: list[tuple[bool, list[float], str | None]] = []

    def _capture_scales(model_arg, _tokenizer, _request, namespace=None):
        observed_scales.append(
            (
                model_arg is model,
                backend.lora_manager.get_lora_scales(model_arg),
                namespace,
            )
        )
        return _sample_output()

    backend.inference.sample = MagicMock(side_effect=_capture_scales)

    result = backend.sample(None, _sample_request())

    assert result.sequences[0].tokens == [4]
    assert observed_scales == [
        (
            True,
            [0.0] * len(original_scales),
            "base:test-model:max_kv=None:kv_bits=4:kv_group_size=64:quantized_kv_start=0",
        )
    ]
    assert backend.lora_manager.get_lora_scales(model) == original_scales


def test_teacher_sampling_does_not_reload_base_model_after_create_model(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with patch(
        "mlx_tinker.backend.mlx_backend.mlx_load",
        return_value=(TinyLM(), FakeTokenizer()),
    ) as load_mock:
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )
        backend.inference.sample = MagicMock(return_value=_sample_output())

        backend.sample(None, _sample_request())

    assert load_mock.call_count == 1


@pytest.mark.parametrize(
    ("config_updates", "patch_longlora"),
    [
        ({"train_embeddings": True}, False),
        ({"train_norms": True}, False),
        ({"use_longlora": True}, True),
    ],
)
def test_teacher_sampling_rejects_non_lora_only_configs(tmp_path, config_updates, patch_longlora):
    backend = _backend(tmp_path, gradient_checkpointing=False)
    lora_config = LoraConfig(rank=4, alpha=8.0, **config_updates)

    with (
        patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())),
        patch("mlx_tinker.backend.mlx_backend.enable_longlora_attention", new=MagicMock())
        if patch_longlora
        else patch("mlx_tinker.backend.mlx_backend.enable_longlora_attention"),
    ):
        backend.create_model("model-1", CreateModelInput(lora_config=lora_config))

    with pytest.raises(ValueError, match="LoRA-only tuning"):
        backend.sample(None, _sample_request())


def test_path_backed_sampling_is_rejected_while_live_model_is_resident(tmp_path):
    backend = _backend(tmp_path, gradient_checkpointing=False)

    with patch("mlx_tinker.backend.mlx_backend.mlx_load", return_value=(TinyLM(), FakeTokenizer())):
        backend.create_model(
            "model-1",
            CreateModelInput(lora_config=LoraConfig(rank=4, alpha=8.0)),
        )

    with pytest.raises(ValueError, match="Path-backed sampling is unavailable"):
        backend.sample(None, _sample_request(model_path="checkpoints/model-1/sampler/latest"))


def test_inject_eos_stop_token_converts_stop_strings_and_qwen_eos(tmp_path):
    class MockQwenTokenizer:
        eos_token_id = 151643

        def encode(self, text, add_special_tokens=False):
            if text == "<|im_end|>":
                return [151645]
            if text == "<|endoftext|>":
                return [151643]
            return [1]

    req = _sample_request(
        sampling_params=SamplingParams(
            temperature=1.0, max_tokens=64, stop_strings=["<|im_end|>"]
        )
    )

    injected = MLXBackend._inject_eos_stop_token(req, MockQwenTokenizer())
    stop_tokens = injected.sampling_params.stop_tokens
    assert stop_tokens is not None
    assert 151643 in stop_tokens  # eos_token_id
    assert 151645 in stop_tokens  # <|im_end|> encoded
