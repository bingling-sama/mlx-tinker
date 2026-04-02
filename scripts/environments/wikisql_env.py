#!/usr/bin/env python3
"""WikiSQL Atropos environment for tinker-atropos RL training.

Implements the tinker-atropos environment pattern for text-to-SQL generation
on WikiSQL data. Model generates SQL queries, which are executed against
in-memory SQLite tables to compute rewards.

Usage with tinker-atropos (3-terminal setup):
  Terminal 1: run-api
  Terminal 2: python launch_training.py --config configs/benchmark_tinker.yaml
  Terminal 3: python scripts/environments/wikisql_env.py serve --config configs/benchmark_tinker.yaml

Can also be used standalone for evaluation:
  python scripts/environments/wikisql_env.py evaluate --config configs/benchmark_tinker.yaml
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from pydantic import Field


FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


class WikiSQLEnvConfig(BaseEnvConfig):
    """Configuration for the WikiSQL environment."""

    data_path: str = Field(
        default=str(FIXTURES_DIR / "wikisql_subset.json"),
        description="Path to WikiSQL JSON data file",
    )
    max_examples: int = Field(default=40, description="Max examples to use for training")
    eval_examples: int = Field(default=10, description="Number of examples reserved for eval")
    system_prompt: str = Field(
        default=(
            "You are a SQL query generator. Given a table schema and a question, "
            "write a SQL query to answer the question. Output only the SQL query."
        ),
        description="System prompt for the model",
    )


class WikiSQLEnv(BaseEnv):
    """Atropos environment for WikiSQL text-to-SQL generation."""

    env_config_cls = WikiSQLEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[dict] = []
        self.eval_data: list[dict] = []
        self.iter: int = 0

    async def setup(self):
        """Load WikiSQL data and prepare for training."""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"WikiSQL data not found at {data_path}")

        with open(data_path) as f:
            all_examples = json.load(f)

        self.data = all_examples[: self.config.max_examples]
        self.eval_data = all_examples[
            self.config.max_examples : self.config.max_examples + self.config.eval_examples
        ]
        print(
            f"WikiSQLEnv: loaded {len(self.data)} train + {len(self.eval_data)} eval examples"
        )

    def get_next_item(self) -> dict:
        """Return the next training example (cycling through data)."""
        item = self.data[self.iter % len(self.data)]
        self.iter += 1
        return item

    def _format_messages(self, example: dict) -> list[dict]:
        """Format a WikiSQL example as chat messages."""
        cols_str = " | ".join(example.get("columns", []))
        rows = example.get("rows", [])
        sample_rows = ""
        for row_data in rows[:3]:
            sample_rows += " | ".join(str(v) for v in row_data) + "\n"

        user_msg = (
            f"Table columns: {cols_str}\n"
            f"Sample data:\n{sample_rows}\n"
            f"Question: {example.get('question', '')}\n"
            f"SQL: "
        )

        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        """Extract SQL from generated text."""
        match = re.search(r"(SELECT\b[^;]*)", text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _execute_sql(sql: str, example: dict) -> list | None:
        """Execute SQL against in-memory SQLite table."""
        header = example.get("columns", [])
        rows = example.get("rows", [])
        if not header or not rows:
            return None
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        try:
            cols_def = ", ".join(f'"{h}" TEXT' for h in header)
            cur.execute(f"CREATE TABLE data ({cols_def})")
            placeholders = ", ".join("?" * len(header))
            cur.executemany(
                f"INSERT INTO data VALUES ({placeholders})",
                [tuple(str(v) for v in r) for r in rows],
            )
            cur.execute(sql)
            return cur.fetchall()
        except Exception:
            return None
        finally:
            conn.close()

    def _score_completion(self, completion_text: str, example: dict) -> float:
        """Score a SQL completion: 1.0 if correct, 0.0 otherwise."""
        pred_sql = self._extract_sql(completion_text)
        if pred_sql is None:
            return 0.0

        pred_result = self._execute_sql(pred_sql, example)
        if pred_result is None:
            return 0.0

        # Compare with gold SQL
        gold_sql = example.get("sql", "")
        adjusted_gold = re.sub(
            r"\bFROM\s+table\b", "FROM data", gold_sql, flags=re.IGNORECASE
        )
        gold_result = self._execute_sql(adjusted_gold, example)
        if gold_result is None:
            # Gold SQL doesn't execute, just check if pred returns something
            return 1.0 if pred_result else 0.0

        # Execution match
        return 1.0 if set(map(tuple, pred_result)) == set(map(tuple, gold_result)) else 0.0

    async def collect_trajectories(
        self, item: Any
    ) -> Tuple[Optional[ScoredDataGroup], List[Any]]:
        """Generate SQL completions and score them."""
        example = item
        messages = self._format_messages(example)

        # Use managed server for generation (atropos pattern)
        async with self.server.managed_server(
            tokenizer=self.tokenizer
        ) as managed:
            completions = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=128,
                temperature=0.8,
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

            # Filter very short completions
            if len(toks) < 5:
                continue

            tokens_list.append(toks)
            masks_list.append(mask)
            scores.append(score)
            logprobs_list.append(lps)

        if not tokens_list:
            return None, []

        scored_data = ScoredDataGroup(
            tokens=tokens_list,
            masks=masks_list,
            scores=scores,
            inference_logprobs=logprobs_list,
        )

        return scored_data, []

    async def evaluate(self, *args, **kwargs):
        """Evaluate current model on held-out WikiSQL examples."""
        if not self.eval_data:
            return

        correct = 0
        total = len(self.eval_data)

        for example in self.eval_data:
            messages = self._format_messages(example)
            try:
                async with self.server.managed_server(
                    tokenizer=self.tokenizer
                ) as managed:
                    completions = await managed.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=128,
                        temperature=0.0,  # greedy for eval
                    )
                    state = managed.get_state()
                    nodes = state.get("nodes", [])

                if nodes:
                    text = nodes[0].get("text", "")
                    score = self._score_completion(text, example)
                    if score > 0:
                        correct += 1
            except Exception:
                pass

        accuracy = correct / total if total > 0 else 0.0
        print(f"WikiSQL Eval: {correct}/{total} = {accuracy:.1%}")

        if hasattr(self, "wandb_log"):
            self.wandb_log({"eval/accuracy": accuracy, "eval/correct": correct})


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python scripts/environments/wikisql_env.py [serve|process|evaluate] ...")
        sys.exit(2)
    WikiSQLEnv.cli()
