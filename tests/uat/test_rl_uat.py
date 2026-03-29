"""Opt-in local-only RL UAT smoke/extended tests."""

from __future__ import annotations

import asyncio
import math
import os

import httpx
import pytest

from tests.uat.helpers import (
    BATCH_SIZE,
    EVAL_EXAMPLE_LIMIT,
    HEAVY_REQUEST_TIMEOUT_S,
    LOCAL_BASE_URL,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    RL_NUM_ROLLOUTS,
    RL_REGRESSION_TOLERANCE,
    RL_STEPS,
    SFT_STEPS,
    TOTAL_TIMEOUT_S,
    TRAIN_EXAMPLE_LIMIT,
    WIKISQL_SPEC,
    get_local_service_client,
    load_examples,
    run_sft_then_rl_backend_uat,
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
async def test_local_wikisql_sft_then_rl(uat_prereqs, hf_tokenizer):
    train_examples = load_examples("wikisql", "train", limit=TRAIN_EXAMPLE_LIMIT)
    eval_examples = load_examples("wikisql", "eval", limit=EVAL_EXAMPLE_LIMIT)

    result = await asyncio.wait_for(
        run_sft_then_rl_backend_uat(
            get_local_service_client(),
            hf_tokenizer,
            WIKISQL_SPEC,
            train_examples,
            eval_examples,
            sft_steps=SFT_STEPS,
            rl_steps=RL_STEPS,
            num_rollouts=RL_NUM_ROLLOUTS,
        ),
        timeout=TOTAL_TIMEOUT_S,
    )

    report = {
        "mode": "local_only_sft_then_rl",
        "dataset": WIKISQL_SPEC.name,
        "model": MODEL_NAME,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "heavy_request_timeout_s": HEAVY_REQUEST_TIMEOUT_S,
        "mlx_tinker": result,
    }
    report_path = write_report("wikisql_sft_then_rl", report)
    print(f"\nRL UAT report written to {report_path}")
    print(report)

    assert result["sft"]["num_steps"] == SFT_STEPS
    assert result["sft"]["batch_size"] == BATCH_SIZE
    assert result["rl"]["num_steps"] == RL_STEPS
    assert result["rl"]["num_rollouts"] == RL_NUM_ROLLOUTS
    assert result["rl"]["group_size"] == RL_NUM_ROLLOUTS
    assert result["rl"]["non_zero_advantage_steps"] >= 1
    assert len(result["sft"]["losses"]) == SFT_STEPS
    assert len(result["rl"]["losses"]) == RL_STEPS
    assert all(math.isfinite(loss) for loss in result["sft"]["losses"])
    assert all(math.isfinite(loss) for loss in result["rl"]["losses"])
    assert all(math.isfinite(reward) for reward in result["rl"]["mean_rewards"])
    assert result["rl_improvement_over_base"] >= RL_REGRESSION_TOLERANCE
