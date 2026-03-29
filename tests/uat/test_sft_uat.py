"""Opt-in local-only SFT UAT smoke tests."""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from tests.uat.helpers import (
    BATCH_SIZE,
    EVAL_EXAMPLE_LIMIT,
    LOCAL_BASE_URL,
    MODEL_NAME,
    SFT_STEPS,
    SFT_REGRESSION_TOLERANCE,
    TOTAL_TIMEOUT_S,
    TRAIN_EXAMPLE_LIMIT,
    WIKISQL_SPEC,
    get_local_service_client,
    load_examples,
    run_sft_backend_uat,
    write_report,
)

pytestmark = pytest.mark.uat


def _require_uat_prereqs() -> None:
    if os.environ.get("MLX_TINKER_RUN_UAT") != "1":
        pytest.skip("Set MLX_TINKER_RUN_UAT=1 to run opt-in UAT.")
    try:
        response = httpx.get(f"{LOCAL_BASE_URL}/api/v1/healthz", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Local mlx-tinker server is not healthy at {LOCAL_BASE_URL}.")
    except Exception:
        pytest.skip(f"Local mlx-tinker server is not reachable at {LOCAL_BASE_URL}.")


@pytest.fixture(scope="session")
def uat_prereqs():
    _require_uat_prereqs()


@pytest.fixture(scope="session")
def hf_tokenizer(uat_prereqs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.mark.asyncio
async def test_local_wikisql_sft_smoke(uat_prereqs, hf_tokenizer):
    train_examples = load_examples("wikisql", "train", limit=TRAIN_EXAMPLE_LIMIT)
    eval_examples = load_examples("wikisql", "eval", limit=EVAL_EXAMPLE_LIMIT)

    local_result = await asyncio.wait_for(
        run_sft_backend_uat(
            get_local_service_client(),
            hf_tokenizer,
            WIKISQL_SPEC,
            train_examples,
            eval_examples,
        ),
        timeout=TOTAL_TIMEOUT_S,
    )

    report = {
        "mode": "local_only_sft_smoke",
        "dataset": WIKISQL_SPEC.name,
        "model": MODEL_NAME,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "mlx_tinker": local_result,
    }
    report_path = write_report("wikisql_sft_smoke", report)
    print(f"\nUAT report written to {report_path}")
    print(report)

    assert local_result["total_time_s"] <= TOTAL_TIMEOUT_S
    assert local_result["improvement"] >= SFT_REGRESSION_TOLERANCE
    assert local_result["num_steps"] == SFT_STEPS
    assert local_result["batch_size"] == BATCH_SIZE
    assert 0.0 <= local_result["base_accuracy"] <= 1.0
    assert 0.0 <= local_result["tuned_accuracy"] <= 1.0
