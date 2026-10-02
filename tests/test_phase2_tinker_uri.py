"""Unit tests for Phase 2: tinker:// Virtual URI & Checkpoint Path System."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import tinker
from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.backend.uri import (
    format_tinker_path,
    is_tinker_path,
    parse_tinker_path,
    relative_path_to_tinker_path,
    tinker_path_to_relative_path,
)
from mlx_tinker.config import EngineConfig
from mlx_tinker.types import (
    CheckpointType,
    LoadWeightsInput,
    LoraConfig,
    SaveWeightsForSamplerInput,
    SaveWeightsInput,
)


@pytest.fixture
def temp_checkpoints_dir(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


@pytest.fixture
def backend(temp_checkpoints_dir):
    config = EngineConfig(
        base_model="test-model",
        checkpoints_base=temp_checkpoints_dir,
    )
    return MLXBackend(config)


class TestTinkerURIEncoderDecoder:
    """Test parse_tinker_path, format_tinker_path, and bidirectional conversions."""

    def test_is_tinker_path(self):
        assert is_tinker_path("tinker://run-1/weights/step-1") is True
        assert is_tinker_path("tinker://run-1/sampler_weights/export-1") is True
        assert is_tinker_path("checkpoints/run-1/step-1") is False
        assert is_tinker_path("/tmp/weights") is False

    def test_parse_tinker_path_training(self):
        model_id, ckpt_type, ckpt_id = parse_tinker_path("tinker://my-run/weights/step-0010")
        assert model_id == "my-run"
        assert ckpt_type == CheckpointType.TRAINING
        assert ckpt_id == "step-0010"

    def test_parse_tinker_path_sampler(self):
        model_id, ckpt_type, ckpt_id = parse_tinker_path("tinker://my-run/sampler_weights/export-xyz")
        assert model_id == "my-run"
        assert ckpt_type == CheckpointType.SAMPLER
        assert ckpt_id == "export-xyz"

    def test_parse_tinker_path_invalid(self):
        with pytest.raises(ValueError, match="must start with 'tinker://'"):
            parse_tinker_path("file://local/weights")

        with pytest.raises(ValueError, match="Invalid tinker path format"):
            parse_tinker_path("tinker://my-run/weights")

        with pytest.raises(ValueError, match="Invalid checkpoint category"):
            parse_tinker_path("tinker://my-run/unknown_type/id")

    def test_format_tinker_path(self):
        # Training
        p1 = format_tinker_path("run-123", "step-5", CheckpointType.TRAINING)
        assert p1 == "tinker://run-123/weights/step-5"

        # Strips duplicate weights/
        p2 = format_tinker_path("run-123", "weights/step-5", CheckpointType.TRAINING)
        assert p2 == "tinker://run-123/weights/step-5"

        # Sampler
        p3 = format_tinker_path("run-123", "sampler-5", CheckpointType.SAMPLER)
        assert p3 == "tinker://run-123/sampler_weights/sampler-5"

        # Strips duplicate sampler/ or sampler_weights/
        p4 = format_tinker_path("run-123", "sampler_weights/sampler-5", CheckpointType.SAMPLER)
        assert p4 == "tinker://run-123/sampler_weights/sampler-5"

        p5 = format_tinker_path("run-123", "sampler/sampler-5", CheckpointType.SAMPLER)
        assert p5 == "tinker://run-123/sampler_weights/sampler-5"

    def test_official_sdk_compatibility(self):
        """Ensure paths formatted by us are perfectly accepted by Tinker SDK's ParsedCheckpointTinkerPath."""
        training_path = format_tinker_path("model-abc", "step-100", CheckpointType.TRAINING)
        sdk_parsed = tinker.ParsedCheckpointTinkerPath.from_tinker_path(training_path)
        assert sdk_parsed.training_run_id == "model-abc"
        assert sdk_parsed.checkpoint_type == "training"
        assert sdk_parsed.checkpoint_id == "weights/step-100"

        sampler_path = format_tinker_path("model-abc", "eval-100", CheckpointType.SAMPLER)
        sdk_parsed_sampler = tinker.ParsedCheckpointTinkerPath.from_tinker_path(sampler_path)
        assert sdk_parsed_sampler.training_run_id == "model-abc"
        assert sdk_parsed_sampler.checkpoint_type == "sampler"
        assert sdk_parsed_sampler.checkpoint_id == "sampler_weights/eval-100"

    def test_bidirectional_roundtrip(self):
        rel_train = "my-model/step_0042"
        uri_train = relative_path_to_tinker_path(rel_train)
        assert uri_train == "tinker://my-model/weights/step_0042"
        rel_back, ckpt_type = tinker_path_to_relative_path(uri_train)
        assert rel_back == rel_train
        assert ckpt_type == CheckpointType.TRAINING

        rel_sampler = "my-model/sampler/export-v1"
        uri_sampler = relative_path_to_tinker_path(rel_sampler)
        assert uri_sampler == "tinker://my-model/sampler_weights/export-v1"
        rel_sampler_back, ckpt_type_sampler = tinker_path_to_relative_path(uri_sampler)
        assert rel_sampler_back == rel_sampler
        assert ckpt_type_sampler == CheckpointType.SAMPLER


class TestValidateCheckpointPathWithTinkerURI:
    """Test MLXBackend._validate_checkpoint_path with tinker:// URIs and standard paths."""

    def test_resolve_tinker_training_uri(self, backend):
        tinker_uri = "tinker://test-model/weights/step-0001"
        resolved = backend._validate_checkpoint_path(tinker_uri)
        expected = (backend.config.checkpoints_base / "test-model" / "step-0001").resolve()
        assert resolved == expected

    def test_resolve_tinker_sampler_uri(self, backend):
        tinker_uri = "tinker://test-model/sampler_weights/export-0001"
        resolved = backend._validate_checkpoint_path(tinker_uri)
        expected = (backend.config.checkpoints_base / "test-model" / "sampler" / "export-0001").resolve()
        assert resolved == expected

    def test_tinker_uri_traversal_protection(self, backend):
        malicious_uri = "tinker://test-model/weights/../../etc/passwd"
        with pytest.raises(ValueError, match="outside the allowed"):
            backend._validate_checkpoint_path(malicious_uri)


class TestSaveAndLoadWeightsTinkerURI:
    """Test save_weights, save_weights_for_sampler, and load_weights returning / resolving tinker:// URIs."""

    def test_save_weights_returns_tinker_uri(self, backend, monkeypatch):
        model_id = "test-training-model"
        backend.models[model_id] = MagicMock()
        backend.training.optimizers[model_id] = MagicMock()

        # Mock save_training_checkpoint to avoid heavy MLX IO
        mock_save = MagicMock()
        monkeypatch.setattr("mlx_tinker.backend.mlx_backend.save_training_checkpoint", mock_save)

        # 1. Without explicit path
        out1 = backend.save_weights(model_id, SaveWeightsInput())
        assert is_tinker_path(out1.path)
        parsed1 = parse_tinker_path(out1.path)
        assert parsed1[0] == model_id
        assert parsed1[1] == CheckpointType.TRAINING

        # 2. With named checkpoint
        out2 = backend.save_weights(model_id, SaveWeightsInput(path="step_0020"))
        assert out2.path == f"tinker://{model_id}/weights/step_0020"

        # 3. With full tinker URI as requested path
        out3 = backend.save_weights(model_id, SaveWeightsInput(path=f"tinker://{model_id}/weights/custom_step"))
        assert out3.path == f"tinker://{model_id}/weights/custom_step"

    def test_save_weights_for_sampler_returns_tinker_uri(self, backend, monkeypatch):
        model_id = "test-sampler-model"
        backend.models[model_id] = MagicMock()
        backend.lora_configs[model_id] = LoraConfig(rank=8, alpha=16.0)

        mock_save_sampler = MagicMock()
        monkeypatch.setattr("mlx_tinker.backend.mlx_backend.save_sampler_weights", mock_save_sampler)

        # 1. Ephemeral save
        out1 = backend.save_weights_for_sampler(
            model_id,
            SaveWeightsForSamplerInput(ephemeral=True, sampling_session_id="sess-1"),
        )
        assert out1.path is None
        assert out1.sampling_session_id == "sess-1"

        # 2. Named sampler save
        out2 = backend.save_weights_for_sampler(
            model_id,
            SaveWeightsForSamplerInput(path="eval_chk_1"),
        )
        assert out2.path == f"tinker://{model_id}/sampler_weights/eval_chk_1"

        # 3. With tinker:// URI
        out3 = backend.save_weights_for_sampler(
            model_id,
            SaveWeightsForSamplerInput(path=f"tinker://{model_id}/sampler_weights/eval_chk_2"),
        )
        assert out3.path == f"tinker://{model_id}/sampler_weights/eval_chk_2"

    def test_load_weights_resolves_tinker_uri(self, backend, monkeypatch):
        model_id = "test-load-model"
        backend.models[model_id] = MagicMock()

        mock_load = MagicMock(return_value={"optimizer_step": 10})
        monkeypatch.setattr("mlx_tinker.backend.mlx_backend.load_training_checkpoint", mock_load)

        # Load using tinker:// URI
        tinker_uri = f"tinker://{model_id}/weights/step_0020"
        out = backend.load_weights(
            model_id,
            LoadWeightsInput(path=tinker_uri, optimizer=True),
        )
        assert out.path == tinker_uri
        mock_load.assert_called_once()
        expected_dir = (backend.config.checkpoints_base / model_id / "step_0020").resolve()
        assert mock_load.call_args[0][1] == expected_dir
