#!/usr/bin/env python3
"""CUAD Atropos environment for legal clause extraction."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from pydantic import Field

FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "uat"


def _normalize_text(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


class CuadEnvConfig(BaseEnvConfig):
    """Configuration for CUAD extraction."""

    data_path: str = Field(
        default=str(FIXTURES_DIR / "cuad_train.json"),
        description="Path to frozen CUAD UAT fixture",
    )
    max_examples: int = Field(default=500, description="Max examples to use for training")
    eval_examples: int = Field(default=100, description="Number of examples reserved for eval")
    system_prompt: str = Field(
        default=(
            "You extract exact clause values from legal contract excerpts. "
            "Return only the exact answer span, or NO_ANSWER if absent."
        ),
        description="System prompt for extraction",
    )


class CuadEnv(BaseEnv):
    """Atropos environment for legal clause extraction."""

    env_config_cls = CuadEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[dict] = []
        self.eval_data: list[dict] = []
        self.iter: int = 0

    async def setup(self):
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"CUAD data not found at {data_path}")

        with open(data_path) as f:
            all_examples = json.load(f)

        self.data = all_examples[: self.config.max_examples]
        self.eval_data = all_examples[
            self.config.max_examples : self.config.max_examples + self.config.eval_examples
        ]
        print(f"CuadEnv: loaded {len(self.data)} train + {len(self.eval_data)} eval examples")

    def get_next_item(self) -> dict:
        item = self.data[self.iter % len(self.data)]
        self.iter += 1
        return item

    def _format_messages(self, example: dict) -> list[dict]:
        user_msg = (
            f'Clause type: {example["clause_type"]}\n'
            f'Question: {example["question"]}\n'
            f'Contract excerpt:\n{example["context"]}\n\n'
            "Answer:"
        )
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def _score_completion(self, completion_text: str, example: dict) -> float:
        predicted = _normalize_text(completion_text)
        gold = _normalize_text(example["answer"])
        return 1.0 if predicted == gold else 0.0

    async def collect_trajectories(self, item: Any):
        example = item
        messages = self._format_messages(example)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=64,
                temperature=0.3,
            )
            state = managed.get_state()
            nodes = state.get("nodes", [])

        if not nodes:
            return None, []

        tokens_list: list[list[int]] = []
        masks_list: list[list[int]] = []
        scores: list[float] = []
        logprobs_list: list[list[float]] = []

        for node in nodes:
            completion_text = node.get("text", "")
            score = self._score_completion(completion_text, example)
            toks = node.get("tokens", [])
            mask = node.get("mask", [1] * len(toks))
            lps = node.get("logprobs", [0.0] * len(toks))
            if not toks:
                continue
            tokens_list.append(toks)
            masks_list.append(mask)
            scores.append(score)
            logprobs_list.append(lps)

        if not tokens_list:
            return None, []

        return ScoredDataGroup(
            tokens=tokens_list,
            masks=masks_list,
            scores=scores,
            inference_logprobs=logprobs_list,
        ), []

    async def evaluate(self, *args, **kwargs):
        if not self.eval_data:
            return

        correct = 0
        total = len(self.eval_data)
        for example in self.eval_data:
            messages = self._format_messages(example)
            try:
                async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                    await managed.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=64,
                        temperature=0.0,
                    )
                    state = managed.get_state()
                    nodes = state.get("nodes", [])
                if nodes and self._score_completion(nodes[0].get("text", ""), example) > 0:
                    correct += 1
            except Exception:
                pass

        accuracy = correct / total if total > 0 else 0.0
        print(f"CUAD Eval: {correct}/{total} = {accuracy:.1%}")
        if hasattr(self, "wandb_log"):
            self.wandb_log({"eval/accuracy": accuracy, "eval/correct": correct})


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python scripts/environments/cuad_env.py [serve|process|evaluate] ...")
        sys.exit(2)
    CuadEnv.cli()
