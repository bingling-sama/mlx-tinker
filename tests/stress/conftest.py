"""Shared fixtures for stress tests (MLX vs HuggingFace equivalence)."""

from __future__ import annotations

import os

import pytest

MODEL_NAME = "Qwen/Qwen3.5-9B"
MIN_RAM_GB = 12  # Minimum RAM for stress tests (~6GB quantized + HF model)


def _check_memory():
    """Check if the system has enough RAM for stress tests."""
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        return ram_gb >= MIN_RAM_GB
    except ImportError:
        return True  # Assume enough if psutil not available


skip_insufficient_ram = pytest.mark.skipif(
    not _check_memory(),
    reason=f"Insufficient RAM (need {MIN_RAM_GB}GB+)",
)


@pytest.fixture(scope="session")
def model_name():
    return os.environ.get("MLX_TINKER_TEST_MODEL", MODEL_NAME)


@pytest.fixture(scope="session")
def shared_tokenizer(model_name):
    """Load tokenizer once for all stress tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


@pytest.fixture(scope="session")
def wikipedia_dataset():
    """Load a small Wikipedia subset for testing."""
    from datasets import load_dataset

    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    samples = []
    for i, example in enumerate(ds):
        if i >= 100:
            break
        text = example["text"][:512]  # Truncate to 512 chars
        if len(text) > 50:
            samples.append(text)
    return samples


@pytest.fixture(scope="session")
def mlx_model(model_name):
    """Load model via mlx-lm (session-scoped, expensive)."""
    from mlx_lm import load

    model, tokenizer = load(model_name)
    return model, tokenizer


@pytest.fixture(scope="session")
def hf_model(model_name):
    """Load model via HuggingFace transformers (session-scoped, expensive)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model
