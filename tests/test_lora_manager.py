"""Unit tests for the LoRA manager.

These tests use a tiny model to verify LoRA application, save, and load
without requiring a full Qwen3.5-9B download.
"""

import tempfile

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.types import LoraConfig


class TinyTransformerLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.mlp = TinyMLP(dim)


class TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim * 2, dim, bias=False)


class TinyLM(nn.Module):
    """Minimal LM structure matching Qwen/Llama architecture for LoRA testing.

    Uses dim=64 so all weight matrices are divisible by group_size=32.
    Has a .layers property matching real mlx-lm models (e.g., Qwen3Model).
    """

    def __init__(self, vocab_size: int = 128, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = TinyModelInner(vocab_size, dim, num_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.model(x)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers


class TinyModelInner(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [TinyTransformerLayer(dim) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x)


@pytest.fixture
def model():
    m = TinyLM(vocab_size=128, dim=64, num_layers=2)
    mx.eval(m.parameters())
    return m


@pytest.fixture
def lora_config():
    return LoraConfig(rank=4, alpha=8.0, seed=42, train_attn=True, train_mlp=True)


@pytest.fixture
def manager():
    return LoRAManager()


class TestApplyQLoRA:
    def test_quantizes_and_adds_lora(self, model, lora_config, manager):
        from mlx.utils import tree_flatten

        total_before = sum(p.size for _, p in tree_flatten(model.parameters()))

        model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=32)

        trainable, total = manager.get_trainable_param_count(model)
        assert trainable > 0
        assert trainable < total
        # LoRA params should be a small fraction
        assert trainable / total < 0.5

    def test_only_lora_params_trainable(self, model, lora_config, manager):
        from mlx.utils import tree_flatten

        model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=32)

        for name, _ in tree_flatten(model.trainable_parameters()):
            assert "lora" in name.lower(), f"Non-LoRA param is trainable: {name}"

    def test_attn_only(self, model, manager):
        from mlx.utils import tree_flatten

        config = LoraConfig(rank=4, alpha=8.0, train_attn=True, train_mlp=False)
        model = manager.apply_qlora(model, config, quantize_bits=4, quantize_group_size=32)

        for name, _ in tree_flatten(model.trainable_parameters()):
            assert "self_attn" in name or "lora" in name


class TestSaveLoadAdapter:
    def test_roundtrip(self, model, lora_config, manager):
        model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.save_adapter(model, tmpdir, lora_config)

            # Verify files exist
            from pathlib import Path

            p = Path(tmpdir)
            assert (p / "adapters.safetensors").exists()
            assert (p / "adapter_config.json").exists()

            # Modify weights, then reload
            from mlx.utils import tree_flatten

            for _, param in tree_flatten(model.trainable_parameters()):
                param *= 0  # Zero out
            mx.eval(model.parameters())

            manager.load_adapter(model, tmpdir)
            mx.eval(model.parameters())

            # Weights should be restored (not zero)
            has_nonzero = False
            for _, param in tree_flatten(model.trainable_parameters()):
                if mx.any(param != 0).item():
                    has_nonzero = True
                    break
            assert has_nonzero, "Loaded adapter should have non-zero weights"
