from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import scaled_dot_product_attention

from mlx_tinker.backend.longlora import enable_longlora_attention
from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.types import LoraConfig


class IdentityRope(nn.Module):
    def __call__(self, x, offset: int = 0):
        del offset
        return x


class TinyLongAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = IdentityRope()

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        del cache
        batch_size, seq_len, _ = x.shape
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rope(queries)
        keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1))


class TinyLongMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim * 2, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLongLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyLongAttention(dim)
        self.mlp = TinyLongMLP(dim)
        self.input_layernorm = nn.RMSNorm(dim, eps=1e-5)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=1e-5)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        attn = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        hidden = x + attn
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TinyLongInner(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [TinyLongLayer(dim) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(dim, eps=1e-5)

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        for layer in self.layers:
            hidden = layer(hidden, mask=mask, cache=None)
        return self.norm(hidden)


class TinyLongLM(nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.model = TinyLongInner(vocab_size, dim, num_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.lm_head(self.model(tokens))

    @property
    def layers(self):
        return self.model.layers


def _clone_model(model: nn.Module) -> nn.Module:
    clone = TinyLongLM(vocab_size=64, dim=32, num_layers=1)
    clone.load_weights(list(dict(tree_flatten(model.parameters())).items()), strict=False)
    mx.eval(clone.parameters())
    return clone


def test_longlora_eval_matches_full_attention():
    mx.random.seed(0)
    baseline = TinyLongLM()
    patched = _clone_model(baseline)
    enable_longlora_attention(patched, group_size_ratio=0.25)

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    baseline.eval()
    patched.eval()

    base_output = baseline(tokens)
    patched_output = patched(tokens)
    mx.eval(base_output, patched_output)

    assert np.allclose(np.array(base_output), np.array(patched_output), atol=1e-6)


def test_longlora_training_changes_divisible_sequences():
    mx.random.seed(1)
    baseline = TinyLongLM()
    patched = _clone_model(baseline)
    enabled = enable_longlora_attention(patched, group_size_ratio=0.25)

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    baseline.train()
    patched.train()

    base_output = baseline(tokens)
    patched_output = patched(tokens)
    mx.eval(base_output, patched_output)

    assert enabled == 1
    assert base_output.shape == patched_output.shape
    assert not np.allclose(np.array(base_output), np.array(patched_output), atol=1e-6)


def test_longlora_training_falls_back_for_non_divisible_sequences():
    mx.random.seed(2)
    baseline = TinyLongLM()
    patched = _clone_model(baseline)
    enable_longlora_attention(patched, group_size_ratio=0.25)

    tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    baseline.train()
    patched.train()

    base_output = baseline(tokens)
    patched_output = patched(tokens)
    mx.eval(base_output, patched_output)

    assert np.allclose(np.array(base_output), np.array(patched_output), atol=1e-6)


def test_apply_qlora_can_train_embeddings_and_norms():
    model = TinyLongLM()
    manager = LoRAManager()
    config = LoraConfig(rank=4, alpha=8.0, train_attn=True, train_mlp=True)

    model = manager.apply_qlora(
        model,
        config,
        quantize_bits=4,
        quantize_group_size=32,
        train_embeddings=True,
        train_norms=True,
    )

    trainable_names = [name for name, _ in tree_flatten(model.trainable_parameters())]

    assert any("lora_" in name for name in trainable_names)
    assert any("embed_tokens.weight" in name for name in trainable_names)
    assert any("input_layernorm.weight" in name for name in trainable_names)
    assert any("post_attention_layernorm.weight" in name for name in trainable_names)
    assert isinstance(model.model.embed_tokens, nn.Embedding)
