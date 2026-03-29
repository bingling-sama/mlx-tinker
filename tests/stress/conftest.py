"""Shared fixtures for stress tests (MLX vs HuggingFace equivalence)."""

from __future__ import annotations

import os

import pytest

MODEL_NAME = "Qwen/Qwen3.5-4B"
MIN_RAM_GB = 4  # Minimum RAM for stress tests (~1GB quantized + HF model)

_allowed_from_env = os.environ.get("MLX_TINKER_ALLOWED_SKIP_TESTS", "").strip()
ALLOWED_SKIP_TESTS = {
    name.strip() for name in _allowed_from_env.split(",") if name.strip()
}


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

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
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


_skipped_tests: list[str] = []


def pytest_runtest_makereport(item, call):
    """Track skipped tests for budget enforcement."""
    if call.when == "call" and call.excinfo is not None:
        if call.excinfo.typename == "Skipped":
            _skipped_tests.append(item.name)
    elif call.when == "setup" and call.excinfo is not None:
        if call.excinfo.typename == "Skipped":
            _skipped_tests.append(item.name)


def pytest_sessionfinish(session, exitstatus):
    """Enforce skip budget: fail if unexpected tests were skipped."""
    if not os.environ.get("ENFORCE_SKIP_BUDGET"):
        return

    unexpected = [n for n in _skipped_tests if n not in ALLOWED_SKIP_TESTS]
    if unexpected:
        session.exitstatus = 1
        print(
            f"\nSKIP BUDGET VIOLATED: {len(unexpected)} unexpected skip(s): "
            f"{unexpected}\nAllowed: {ALLOWED_SKIP_TESTS}"
        )
