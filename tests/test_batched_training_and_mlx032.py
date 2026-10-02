from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_tinker.backend.inference import InferenceBackend
from mlx_tinker.backend.training import TrainingBackend
from mlx_tinker.types import (
    EncodedTextChunk,
    ForwardBackwardInput,
    ForwardInput,
    ModelInput,
    SampleInput,
    SamplingParams,
)
from tests.helpers import TinyModelInner, make_datum


@pytest.fixture
def tiny_training_model():
    m = TinyModelInner(vocab_size=100, dim=32, num_layers=2)
    mx.eval(m.parameters())
    return m


def test_forward_backward_batch_gradient_and_metric_equivalence(tiny_training_model):
    """Batched forward_backward must produce mathematically identical gradients."""
    backend_seq = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)
    backend_batch = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

    d1 = make_datum([10, 11, 12], [11, 12, 13], [1.0, 1.0, 1.0])
    d2 = make_datum([20, 21, 22, 23], [21, 22, 23, 24], [1.0, 1.0, 1.0, 1.0])
    d3 = make_datum([30, 31], [31, 32], [1.0, 1.0])

    r1 = ForwardBackwardInput(data=[d1], loss_fn="cross_entropy")
    r2 = ForwardBackwardInput(data=[d2, d3], loss_fn="cross_entropy")

    # 1. Sequential run
    out1_seq = backend_seq.forward_backward("m_seq", tiny_training_model, r1)
    out2_seq = backend_seq.forward_backward("m_seq", tiny_training_model, r2)
    grads_seq = dict(tree_flatten(backend_seq.accumulated_grads["m_seq"]))

    # 2. Batched run
    batch_outs = backend_batch.forward_backward_batch("m_batch", tiny_training_model, [r1, r2])
    assert len(batch_outs) == 2
    out1_batched, out2_batched = batch_outs
    grads_batch = dict(tree_flatten(backend_batch.accumulated_grads["m_batch"]))

    # Verify gradients are identical
    for k in grads_seq:
        max_diff = float(mx.max(mx.abs(grads_seq[k] - grads_batch[k])).item())
        assert max_diff < 1e-5, f"Gradient difference too large for {k}: {max_diff}"

    # Verify individual sequence loss metrics match
    assert abs(out1_seq.metrics["loss:sum"] - out1_batched.metrics["loss:sum"]) < 1e-4
    assert abs(out2_seq.metrics["loss:sum"] - out2_batched.metrics["loss:sum"]) < 1e-4

    # Verify logprobs match
    assert len(out1_batched.loss_fn_outputs) == 1
    assert len(out2_batched.loss_fn_outputs) == 2


def test_forward_batch_equivalence(tiny_training_model):
    """Batched forward pass must return the exact same logprobs as sequential forward."""
    backend = TrainingBackend(optimizer_type="adamw", gradient_checkpointing=False)

    d1 = make_datum([5, 6, 7], [6, 7, 8], [1.0, 1.0, 1.0])
    d2 = make_datum([15, 16], [16, 17], [1.0, 1.0])

    r1 = ForwardInput(data=[d1])
    r2 = ForwardInput(data=[d2])

    seq_out1 = backend.forward("m", tiny_training_model, r1)
    seq_out2 = backend.forward("m", tiny_training_model, r2)

    batched_outs = backend.forward_batch("m", tiny_training_model, [r1, r2])
    assert len(batched_outs) == 2

    assert batched_outs[0].logprobs == seq_out1.logprobs
    assert batched_outs[1].logprobs == seq_out2.logprobs


@pytest.mark.skipif(
    not Path("./Qwen3.5-4B").exists(),
    reason="Local Qwen3.5-4B checkpoint not found",
)
def test_batch_generator_mlx032_no_numeric_collapse():
    """BatchGenerator under mlx-lm 0.32 handles heterogeneous prompt lengths."""
    import mlx_lm

    model, tokenizer = mlx_lm.load("./Qwen3.5-4B")
    backend = InferenceBackend()

    prompts = [
        "2 + 2 =",
        "Explain quantum computing in one short sentence.",
        "Hello",
    ]
    tokenized = [tokenizer.encode(p) for p in prompts]
    requests = [
        SampleInput(
            prompt=ModelInput(chunks=[EncodedTextChunk(tokens=toks)]),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=10),
            num_samples=1,
        )
        for toks in tokenized
    ]

    outputs = backend.sample_batch(model, tokenizer, requests)
    assert len(outputs) == 3

    for i, out in enumerate(outputs):
        tokens = out.sequences[0].tokens
        assert len(tokens) > 0
        # In Qwen token 0 is '!'
        assert tokens != [0] * len(tokens), f"Prompt {i} collapsed into zeros/!"
        assert not all(t == tokens[0] for t in tokens), f"Prompt {i} repeated single token"
