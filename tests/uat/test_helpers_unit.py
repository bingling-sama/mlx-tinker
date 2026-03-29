"""Offline unit tests for UAT helper logic."""

from __future__ import annotations

from tests.uat.helpers import (
    CUAD_SPEC,
    build_rl_datum,
    compute_advantages,
    WIKISQL_SPEC,
    cuad_score_prediction,
    normalize_text,
    wikisql_reward_prediction,
    wikisql_score_prediction,
)


def test_normalize_text_uses_first_line_and_whitespace():
    assert normalize_text("  Foo   Bar \nBaz  ") == "foo bar"


def test_wikisql_score_prediction_matches_on_equivalent_table_name():
    example = {
        "columns": ["Country", "Capital"],
        "rows": [["France", "Paris"], ["Japan", "Tokyo"]],
        "sql": "SELECT \"Capital\" FROM table WHERE \"Country\" = 'France'",
    }
    text = 'SELECT "Capital" FROM data WHERE "Country" = \'France\''
    assert wikisql_score_prediction(text, example) is True


def test_cuad_score_prediction_normalizes_case():
    example = {"answer": "Distributor Agreement"}
    assert cuad_score_prediction("distributor agreement\nextra", example) is True


def test_wikisql_reward_prediction_is_dense():
    example = {
        "columns": ["Country", "Capital"],
        "rows": [["France", "Paris"], ["Japan", "Tokyo"]],
        "sql": 'SELECT "Capital" FROM table WHERE "Country" = \'France\'',
    }

    assert wikisql_reward_prediction("nonsense", example) == -1.0
    invalid_reward = wikisql_reward_prediction('SELECT nope FROM data', example)
    assert -0.5 <= invalid_reward < 0.0

    wrong_reward = wikisql_reward_prediction(
        'SELECT "Capital" FROM data WHERE "Country" = \'Japan\'',
        example,
    )
    assert 0.0 < wrong_reward < 1.0

    assert wikisql_reward_prediction(
        'SELECT "Capital" FROM data WHERE "Country" = \'France\'',
        example,
    ) == 1.0


def test_dataset_specs_build_expected_prompts():
    wikisql_prompt = WIKISQL_SPEC.make_prompt(
        {
            "columns": ["Country", "Capital"],
            "rows": [["France", "Paris"]],
            "question": "What is the capital of France?",
        }
    )
    assert "SQL:" in wikisql_prompt

    cuad_prompt = CUAD_SPEC.make_prompt(
        {
            "clause_type": "Document Name",
            "question": "Extract the document name.",
            "context": "This agreement is titled Distributor Agreement.",
        }
    )
    assert "Answer:" in cuad_prompt


def test_compute_advantages_centers_rewards():
    assert compute_advantages([1.0, -1.0, -1.0, 1.0]) == [1.0, -1.0, -1.0, 1.0]


def test_build_rl_datum_masks_prompt_positions():
    class FakeTokenizer:
        def encode(self, text):
            return [11, 12, 13]

    datum = build_rl_datum(
        FakeTokenizer(),
        WIKISQL_SPEC,
        {
            "columns": ["Country"],
            "rows": [["France"]],
            "question": "Country?",
            "sql": 'SELECT "Country" FROM table',
        },
        generated_tokens=[21, 22],
        generated_logprobs=[-0.2, -0.3],
        advantage=0.5,
    )

    loss_inputs = datum.loss_fn_inputs
    assert loss_inputs["advantages"].data[:2] == [0.0, 0.0]
    assert loss_inputs["advantages"].data[-1] == 0.5
