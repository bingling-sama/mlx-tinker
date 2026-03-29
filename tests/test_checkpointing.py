"""Tests for checkpoint save/load: training checkpoints and sampler weights."""


import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_tinker.backend.checkpointing import (
    load_training_checkpoint,
    save_sampler_weights,
    save_training_checkpoint,
)
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import AdamParams, ForwardBackwardInput, OptimStepInput
from tests.helpers import TinyModel, make_datum


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
    def test_creates_adapter_file(self, model, tmp_path):
        out_dir = tmp_path / "sampler"
        save_sampler_weights(model, out_dir, base_model="test-model")

        assert (out_dir / "adapters.safetensors").exists()

    def test_saves_base_model_and_lora_config(self, model, tmp_path):
        import json

        out_dir = tmp_path / "sampler"
        lora_config = {"rank": 8, "alpha": 16.0}
        save_sampler_weights(model, out_dir, base_model="test-model", lora_config=lora_config)

        saved = json.loads((out_dir / "config.json").read_text())
        assert saved["base_model"] == "test-model"
        assert saved["lora_config"]["rank"] == 8

    def test_saves_tokenizer_config(self, model, tmp_path):
        import json

        out_dir = tmp_path / "sampler"
        tok_config = {"bos_token_id": 1, "eos_token_id": 2}
        save_sampler_weights(model, out_dir, base_model="test-model", tokenizer_config=tok_config)

        saved = json.loads((out_dir / "tokenizer_config.json").read_text())
        assert saved["eos_token_id"] == 2

    def test_weight_count_matches_trainable_parameters(self, model, tmp_path):
        out_dir = tmp_path / "sampler"
        save_sampler_weights(model, out_dir, base_model="test-model")

        loaded = mx.load(str(out_dir / "adapters.safetensors"))
        trainable_params = dict(tree_flatten(model.trainable_parameters()))
        assert len(loaded) == len(trainable_params)


# ---------------------------------------------------------------------------
# Checkpoint determinism (Tier 2)
# ---------------------------------------------------------------------------


class TestCheckpointDeterminism:
    def test_save_reload_produces_same_loss(self, tmp_path):
        """Save → reload → forward must produce identical loss (weights restored exactly)."""
        mx.random.seed(42)
        model = TinyModel(vocab_size=32, dim=16)
        mx.eval(model.parameters())
        training = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0))

        # Train 5 steps
        for _ in range(5):
            training.forward_backward("test", model, fb_req)
            training.optim_step("test", model, opt_req)

        # Record loss at current weights
        result_before = training.forward_backward("loss_check", model, fb_req)
        loss_before = result_before.metrics["loss:sum"]

        # Save checkpoint
        ckpt_dir = tmp_path / "ckpt"
        save_training_checkpoint(model, None, ckpt_dir)

        # Record exact params
        params_before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}

        # Corrupt model weights
        for _, p in tree_flatten(model.parameters()):
            p *= 0
        mx.eval(model.parameters())

        # Reload
        load_training_checkpoint(model, ckpt_dir)
        mx.eval(model.parameters())

        # Verify exact parameter match
        params_after = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        for key in params_before:
            np.testing.assert_array_equal(
                params_before[key], params_after[key],
                err_msg=f"Parameter {key} differs after reload",
            )

        # Verify same loss
        training2 = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
        result_after = training2.forward_backward("loss_check2", model, fb_req)
        loss_after = result_after.metrics["loss:sum"]

        assert abs(loss_before - loss_after) < 1e-6, (
            f"Loss differs after reload: {loss_before} vs {loss_after}"
        )


# ---------------------------------------------------------------------------
# Continuation determinism (P2)
# ---------------------------------------------------------------------------


class TestContinuationDeterminism:
    """Train N -> save -> reload -> train M must match training N+M straight through."""

    def test_train_N_save_train_M_matches_N_plus_M(self, tmp_path):
        """Resumed training must produce same params+loss as uninterrupted training."""
        N, M = 5, 5
        datum = make_datum([1, 2, 3, 4], [2, 3, 4, 5], [0.0, 1.0, 1.0, 1.0])
        fb_req = ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
        opt_req = OptimStepInput(
            adam_params=AdamParams(learning_rate=0.01, weight_decay=0.0)
        )

        # --- Path A: train N+M straight through ---
        mx.random.seed(42)
        model_a = TinyModel(vocab_size=32, dim=16)
        mx.eval(model_a.parameters())
        training_a = TrainingBackend(
            optimizer_type="adamw", gradient_checkpointing=False
        )

        for _ in range(N + M):
            training_a.forward_backward("ref", model_a, fb_req)
            training_a.optim_step("ref", model_a, opt_req)

        result_a = training_a.forward_backward("ref_final", model_a, fb_req)
        loss_a = result_a.metrics["loss:sum"]
        params_a = {
            k: np.array(v) for k, v in tree_flatten(model_a.parameters())
        }

        # --- Path B: train N, save, reload, train M ---
        mx.random.seed(42)
        model_b = TinyModel(vocab_size=32, dim=16)
        mx.eval(model_b.parameters())
        training_b = TrainingBackend(
            optimizer_type="adamw", gradient_checkpointing=False
        )

        for _ in range(N):
            training_b.forward_backward("resume", model_b, fb_req)
            training_b.optim_step("resume", model_b, opt_req)

        ckpt_dir = tmp_path / "continuation_ckpt"
        opt_state = training_b.get_optimizer_state("resume")
        save_training_checkpoint(model_b, opt_state, ckpt_dir)

        loaded_opt = load_training_checkpoint(model_b, ckpt_dir)
        mx.eval(model_b.parameters())

        training_b.load_optimizer_state("resume", loaded_opt)

        for _ in range(M):
            training_b.forward_backward("resume", model_b, fb_req)
            training_b.optim_step("resume", model_b, opt_req)

        result_b = training_b.forward_backward("resume_final", model_b, fb_req)
        loss_b = result_b.metrics["loss:sum"]
        params_b = {
            k: np.array(v) for k, v in tree_flatten(model_b.parameters())
        }

        # --- Compare ---
        for key in params_a:
            np.testing.assert_allclose(
                params_a[key],
                params_b[key],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Parameter {key} diverged after continuation",
            )

        assert abs(loss_a - loss_b) < 1e-5, (
            f"Loss diverged: straight={loss_a} resumed={loss_b}"
        )
