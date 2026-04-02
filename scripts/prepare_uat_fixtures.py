#!/usr/bin/env python3
"""Build frozen UAT fixtures for WikiSQL and CUAD."""

from __future__ import annotations

import json
import random
import re
import tarfile
import tempfile
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "uat"

WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
WIKISQL_TRAIN_SIZE = 500
WIKISQL_EVAL_SIZE = 100
CUAD_TRAIN_SIZE = 500
CUAD_EVAL_SIZE = 100

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<"]
CLAUSE_RE = re.compile(r'"([^"]+)"')


def _write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _wikisql_sql_to_string(sql: dict, headers: list[str]) -> str:
    sel = int(sql["sel"])
    agg = AGG_OPS[int(sql["agg"])]
    select_column = headers[sel]
    select_expr = f'{agg}("{select_column}")' if agg else f'"{select_column}"'

    where_parts = []
    for col_idx, op_idx, value in sql.get("conds", []):
        column = headers[int(col_idx)]
        operator = COND_OPS[int(op_idx)]
        try:
            float(value)
            value_repr = str(value)
        except (TypeError, ValueError):
            value_repr = f"'{value}'"
        where_parts.append(f'"{column}" {operator} {value_repr}')

    query = f"SELECT {select_expr} FROM table"
    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)
    return query


def _build_wikisql_split(main_path: Path, tables_path: Path, limit: int, seed: int) -> list[dict]:
    with tables_path.open() as f:
        table_map = {row["id"]: row for row in (json.loads(line) for line in f)}

    with main_path.open() as f:
        rows = [json.loads(line) for line in f]

    rng = random.Random(seed)
    rng.shuffle(rows)

    examples = []
    for row in rows:
        table = table_map[row["table_id"]]
        examples.append(
            {
                "id": f'{row["table_id"]}:{row["question"]}',
                "table_id": row["table_id"],
                "question": row["question"].strip(),
                "sql": _wikisql_sql_to_string(row["sql"], table["header"]),
                "columns": [str(x) for x in table["header"]],
                "rows": [[str(cell) for cell in r] for r in table["rows"][:8]],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def build_wikisql() -> tuple[list[dict], list[dict]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "wikisql.tar.bz2"
        urllib.request.urlretrieve(WIKISQL_URL, tar_path)
        with tarfile.open(tar_path, mode="r:bz2") as tf:
            tf.extractall(Path(tmpdir), filter="data")

        data_dir = Path(tmpdir) / "data"
        train = _build_wikisql_split(
            data_dir / "train.jsonl",
            data_dir / "train.tables.jsonl",
            WIKISQL_TRAIN_SIZE,
            seed=7,
        )
        eval_examples = _build_wikisql_split(
            data_dir / "dev.jsonl",
            data_dir / "dev.tables.jsonl",
            WIKISQL_EVAL_SIZE,
            seed=11,
        )
        return train, eval_examples


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _extract_clause_type(question: str) -> str:
    match = CLAUSE_RE.search(question)
    return match.group(1).strip() if match else question.strip()


def _make_cuad_window(context: str, answer: str, answer_start: int) -> str:
    if answer_start < 0:
        window = context[:1200]
    else:
        left = max(0, answer_start - 500)
        right = min(len(context), answer_start + len(answer) + 500)
        window = context[left:right]
    return _normalize_space(window)[:1400]


def _iter_cuad_candidates() -> list[dict]:
    path = hf_hub_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json", repo_type="dataset")
    with open(path) as f:
        raw = json.load(f)

    examples = []
    for contract in raw["data"]:
        title = contract["title"]
        for paragraph in contract["paragraphs"]:
            context = _normalize_space(paragraph["context"])
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    continue
                answer = _normalize_space(answers[0]["text"])
                if not answer:
                    continue
                if len(answer) > 160 or len(answer.split()) > 20:
                    continue
                answer_start = int(answers[0].get("answer_start", -1))
                window = _make_cuad_window(context, answer, answer_start)
                if answer not in window:
                    continue
                examples.append(
                    {
                        "id": qa["id"],
                        "contract_title": title,
                        "clause_type": _extract_clause_type(qa["question"]),
                        "question": qa["question"].strip(),
                        "context": window,
                        "answer": answer,
                    }
                )
    return examples


def _balanced_select(examples: list[dict], limit: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        grouped[example["clause_type"]].append(example)

    rng = random.Random(seed)
    clause_types = sorted(grouped)
    queues: dict[str, deque[dict]] = {}
    for clause_type in clause_types:
        items = grouped[clause_type]
        rng.shuffle(items)
        queues[clause_type] = deque(items)

    selected = []
    seen_ids = set()
    while len(selected) < limit:
        advanced = False
        for clause_type in clause_types:
            queue = queues[clause_type]
            while queue and queue[0]["id"] in seen_ids:
                queue.popleft()
            if not queue:
                continue
            example = queue.popleft()
            selected.append(example)
            seen_ids.add(example["id"])
            advanced = True
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return selected


def build_cuad() -> tuple[list[dict], list[dict]]:
    all_examples = _iter_cuad_candidates()
    train = _balanced_select(all_examples, CUAD_TRAIN_SIZE, seed=13)
    used_ids = {example["id"] for example in train}
    remaining = [example for example in all_examples if example["id"] not in used_ids]
    eval_examples = _balanced_select(remaining, CUAD_EVAL_SIZE, seed=17)
    return train, eval_examples


def main() -> None:
    wikisql_train, wikisql_eval = build_wikisql()
    cuad_train, cuad_eval = build_cuad()

    _write_json(FIXTURE_DIR / "wikisql_train.json", wikisql_train)
    _write_json(FIXTURE_DIR / "wikisql_eval.json", wikisql_eval)
    _write_json(FIXTURE_DIR / "cuad_train.json", cuad_train)
    _write_json(FIXTURE_DIR / "cuad_eval.json", cuad_eval)

    print(f"Wrote {len(wikisql_train)} WikiSQL train examples")
    print(f"Wrote {len(wikisql_eval)} WikiSQL eval examples")
    print(f"Wrote {len(cuad_train)} CUAD train examples")
    print(f"Wrote {len(cuad_eval)} CUAD eval examples")


if __name__ == "__main__":
    main()
