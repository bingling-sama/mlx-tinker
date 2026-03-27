"""Shared fixtures for cookbook verification tests."""

from __future__ import annotations

import os

import pytest

MODEL_NAME = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="session")
def model_name():
    return os.environ.get("MLX_TINKER_TEST_MODEL", MODEL_NAME)
