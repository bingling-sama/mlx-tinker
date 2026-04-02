"""Curriculum loading and lightweight task grading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UserTurn:
    content: str


@dataclass(frozen=True)
class SuccessCheck:
    kind: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    source: str
    system_setup: str
    user_turns: tuple[UserTurn, ...]
    success_checks: tuple[SuccessCheck, ...]
    reward_rule: str
    trainable: bool


@dataclass(frozen=True)
class TaskEvaluation:
    passed: bool
    reward: float
    failed_checks: tuple[str, ...]


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        # The shipped curriculum file intentionally uses JSON-compatible YAML.
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Curriculum root must be an object: {path}")
    return loaded


def load_curriculum(path: Path) -> list[TaskSpec]:
    loaded = _load_raw(path)
    tasks = loaded.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Curriculum is missing 'tasks': {path}")
    parsed: list[TaskSpec] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"Task entry must be an object: {task!r}")
        parsed.append(
            TaskSpec(
                task_id=str(task["task_id"]),
                category=str(task["category"]),
                source=str(task.get("source", "")),
                system_setup=str(task.get("system_setup", "")),
                user_turns=tuple(
                    UserTurn(content=str(turn["content"]))
                    for turn in task.get("user_turns", [])
                ),
                success_checks=tuple(
                    SuccessCheck(
                        kind=str(check["kind"]),
                        values=tuple(str(value) for value in check.get("values", [])),
                    )
                    for check in task.get("success_checks", [])
                ),
                reward_rule=str(task.get("reward_rule", "binary_all_checks")),
                trainable=bool(task.get("trainable", True)),
            )
        )
    return parsed


def split_curriculum(tasks: list[TaskSpec]) -> tuple[list[TaskSpec], list[TaskSpec]]:
    train = [task for task in tasks if task.trainable]
    eval_only = [task for task in tasks if not task.trainable]
    return train, eval_only


def _check_passes(text: str, check: SuccessCheck) -> bool:
    haystack = text.lower()
    values = tuple(value.lower() for value in check.values)
    if check.kind == "contains_all":
        return all(value in haystack for value in values)
    if check.kind == "contains_any":
        return any(value in haystack for value in values)
    if check.kind == "regex":
        return any(re.search(value, text, re.IGNORECASE) is not None for value in check.values)
    raise ValueError(f"Unknown success check kind: {check.kind}")


def evaluate_task_output(task: TaskSpec, text: str) -> TaskEvaluation:
    failures = [
        f"{check.kind}:{','.join(check.values)}"
        for check in task.success_checks
        if not _check_passes(text, check)
    ]
    passed = not failures
    if task.reward_rule == "binary_all_checks":
        reward = 1.0 if passed else -1.0
    else:
        raise ValueError(f"Unsupported reward rule: {task.reward_rule}")
    return TaskEvaluation(passed=passed, reward=reward, failed_checks=tuple(failures))
