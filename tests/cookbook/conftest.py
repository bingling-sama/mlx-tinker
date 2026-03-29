"""Shared fixtures for cookbook verification tests.

Uses the atropos sql_query_env for WikiSQL data loading and SQL execution,
and tinker-atropos patterns for training data conversion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
WIKISQL_SUBSET_SIZE = 50
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def model_name():
    return os.environ.get("MLX_TINKER_TEST_MODEL", MODEL_NAME)


@pytest.fixture(scope="session")
def wikisql_data():
    """Load WikiSQL subset from local fixture."""
    fixture_path = FIXTURES_DIR / "wikisql_subset.json"
    if fixture_path.exists():
        with open(fixture_path) as f:
            return json.load(f)[:WIKISQL_SUBSET_SIZE]

    try:
        from environments.community.sql_query_env.wikisql_loader import load_wikisql

        return load_wikisql(n_items=WIKISQL_SUBSET_SIZE)
    except ImportError:
        pytest.fail(
            f"WikiSQL fixture not found at {fixture_path} and "
            "atropos sql_query_env not available"
        )
