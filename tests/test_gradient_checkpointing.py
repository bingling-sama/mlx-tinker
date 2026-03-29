"""Tests for runtime gradient checkpointing wiring."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_tinker.backend.gradient_checkpointing import enable_gradient_checkpointing
from mlx_tinker.backend.lora_manager import LoRAManager
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import AdamParams, ForwardBackwardInput, LoraConfig, OptimStepInput
from tests.helpers import TinyLM, make_datum


def _clone_model_weights(src: TinyLM, dst: TinyLM) -> None:
    weights = list(dict(tree_flatten(src.parameters())).items())
    dst.load_weights(weights, strict=True)
    mx.eval(dst.parameters())


def test_enable_gradient_checkpointing_preserves_parameter_names():
    model = TinyLM(vocab_size=64, dim=64, num_layers=2)
    mx.eval(model.parameters())

    before = [name for name, _ in tree_flatten(model.parameters())]

    patched = enable_gradient_checkpointing(model)
    after = [name for name, _ in tree_flatten(model.parameters())]

    assert patched == len(model.layers)
    assert before == after
    assert enable_gradient_checkpointing(model) == 0


def test_checkpointed_training_matches_uncheckpointed_training():
    model_plain = TinyLM(vocab_size=64, dim=64, num_layers=2)
    model_ckpt = TinyLM(vocab_size=64, dim=64, num_layers=2)
    mx.eval(model_plain.parameters(), model_ckpt.parameters())
    _clone_model_weights(model_plain, model_ckpt)
    enable_gradient_checkpointing(model_ckpt)

    training_plain = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
    training_ckpt = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=True)

    request = ForwardBackwardInput(
        data=[make_datum([1, 2, 3, 4], [2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0])],
        loss_fn="cross_entropy",
    )

    model_plain.train()
    model_ckpt.train()
    result_plain = training_plain.forward_backward("plain", model_plain, request)
    result_ckpt = training_ckpt.forward_backward("ckpt", model_ckpt, request)

    assert abs(result_plain.metrics["loss:sum"] - result_ckpt.metrics["loss:sum"]) < 1e-5

    plain_grads = dict(tree_flatten(training_plain.accumulated_grads["plain"]))
    ckpt_grads = dict(tree_flatten(training_ckpt.accumulated_grads["ckpt"]))
    assert plain_grads.keys() == ckpt_grads.keys()
    for key in plain_grads:
        assert mx.allclose(plain_grads[key], ckpt_grads[key], atol=1e-5, rtol=1e-5).item(), key

    opt_request = OptimStepInput(
        adam_params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=0.0)
    )
    training_plain.optim_step("plain", model_plain, opt_request)
    training_ckpt.optim_step("ckpt", model_ckpt, opt_request)

    plain_params = dict(tree_flatten(model_plain.parameters()))
    ckpt_params = dict(tree_flatten(model_ckpt.parameters()))
    for key in plain_params:
        assert mx.allclose(plain_params[key], ckpt_params[key], atol=1e-5, rtol=1e-5).item(), key


def test_checkpointing_keeps_lora_trainable_parameter_names_stable():
    model = TinyLM(vocab_size=64, dim=64, num_layers=2)
    mx.eval(model.parameters())

    manager = LoRAManager()
    lora_config = LoraConfig(rank=4, alpha=8.0, train_attn=True, train_mlp=True)
    model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=32)

    before = [name for name, _ in tree_flatten(model.trainable_parameters())]
    patched = enable_gradient_checkpointing(model)
    after = [name for name, _ in tree_flatten(model.trainable_parameters())]

    assert patched == len(model.layers)
    assert before == after
