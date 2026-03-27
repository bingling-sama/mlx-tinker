"""Tests for checkpoint save/load: training checkpoints and sampler weights."""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_tinker.backend.checkpointing import (
    load_training_checkpoint,
    save_sampler_weights,
    save_training_checkpoint,
)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.embed(x))


@pytest.fixture
def model():
    m = TinyModel()
    mx.eval(m.parameters())
    return m


class TestSaveTrainingCheckpoint:
    def test_creates_files(self, model, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        save_training_checkpoint(model, None, ckpt_dir)

        assert (ckpt_dir / "model.safetensors").exists()
        assert (ckpt_dir / "metadata.json").exists()

    def test_saves_optimizer_state(self, model, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        opt_state = {"step": mx.array(10), "lr": mx.array(0.001)}
        save_training_checkpoint(model, opt_state, ckpt_dir)

        assert (ckpt_dir / "optimizer" / "state.npz").exists()

    def test_saves_metadata(self, model, tmp_path):
        import json

        ckpt_dir = tmp_path / "ckpt"
        meta = {"step": 42, "loss": 0.5}
        save_training_checkpoint(model, None, ckpt_dir, metadata=meta)

        saved_meta = json.loads((ckpt_dir / "metadata.json").read_text())
        assert saved_meta["step"] == 42
        assert saved_meta["loss"] == 0.5


class TestLoadTrainingCheckpoint:
    def test_roundtrip(self, model, tmp_path):
        """Save and load should preserve model weights."""
        ckpt_dir = tmp_path / "ckpt"
        original_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}

        save_training_checkpoint(model, None, ckpt_dir)

        # Zero out weights
        for _, p in tree_flatten(model.parameters()):
            p *= 0
        mx.eval(model.parameters())

        # Verify weights are zeroed
        for _, p in tree_flatten(model.parameters()):
            assert mx.all(p == 0).item()

        # Load and verify restoration
        load_training_checkpoint(model, ckpt_dir)
        mx.eval(model.parameters())

        loaded_weights = {k: v.tolist() for k, v in tree_flatten(model.parameters())}
        for key in original_weights:
            assert original_weights[key] == loaded_weights[key], f"Weight mismatch for {key}"

    def test_missing_checkpoint_raises(self, model, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_training_checkpoint(model, tmp_path / "nonexistent")

    def test_returns_none_without_optimizer_state(self, model, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        save_training_checkpoint(model, None, ckpt_dir)

        opt_state = load_training_checkpoint(model, ckpt_dir)
        assert opt_state is None

    def test_returns_optimizer_state(self, model, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        opt_state = {"step": mx.array(10)}
        save_training_checkpoint(model, opt_state, ckpt_dir)

        loaded_opt = load_training_checkpoint(model, ckpt_dir)
        assert loaded_opt is not None


class TestSaveSamplerWeights:
    def test_creates_model_file(self, model, tmp_path):
        out_dir = tmp_path / "sampler"
        save_sampler_weights(model, out_dir)

        assert (out_dir / "model.safetensors").exists()

    def test_saves_config(self, model, tmp_path):
        import json

        out_dir = tmp_path / "sampler"
        config = {"hidden_size": 16, "vocab_size": 32}
        save_sampler_weights(model, out_dir, model_config=config)

        saved = json.loads((out_dir / "config.json").read_text())
        assert saved["hidden_size"] == 16

    def test_saves_tokenizer_config(self, model, tmp_path):
        import json

        out_dir = tmp_path / "sampler"
        tok_config = {"bos_token_id": 1, "eos_token_id": 2}
        save_sampler_weights(model, out_dir, tokenizer_config=tok_config)

        saved = json.loads((out_dir / "tokenizer_config.json").read_text())
        assert saved["eos_token_id"] == 2

    def test_weight_count_matches_model(self, model, tmp_path):
        out_dir = tmp_path / "sampler"
        save_sampler_weights(model, out_dir)

        loaded = mx.load(str(out_dir / "model.safetensors"))
        model_params = dict(tree_flatten(model.parameters()))
        assert len(loaded) == len(model_params)
