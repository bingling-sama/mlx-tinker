"""Scripted user harness that talks to OpenClaw over the Gateway WebSocket."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .curriculum import TaskEvaluation, TaskSpec, evaluate_task_output
from .gateway_client import OpenClawGatewayClient


@dataclass(frozen=True)
class TaskRunResult:
    task: TaskSpec
    session_key: str
    history: tuple[dict[str, Any], ...]
    final_assistant_text: str
    evaluation: TaskEvaluation


def _extract_text_blocks(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    text = message.get("text")
    return str(text) if isinstance(text, str) else ""


class OpenClawUserSimulator:
    def __init__(self, gateway_client: OpenClawGatewayClient) -> None:
        self.gateway_client = gateway_client

    async def run_task(self, task: TaskSpec) -> TaskRunResult:
        session_key = f"wildclaw-{task.task_id}-{uuid.uuid4().hex[:8]}"
        first_turn = True
        for turn in task.user_turns:
            message = turn.content
            if first_turn and task.system_setup:
                message = f"{task.system_setup}\n\nUser request:\n{turn.content}"
                first_turn = False
            sent = await self.gateway_client.send_chat(session_key=session_key, message=message)
            await self.gateway_client.wait_for_run(sent.run_id)

        history = await self.gateway_client.load_history(session_key=session_key, limit=100)
        assistant_messages = [
            message for message in history if str(message.get("role", "")).lower() == "assistant"
        ]
        final_text = _extract_text_blocks(assistant_messages[-1]) if assistant_messages else ""
        evaluation = evaluate_task_output(task, final_text)
        return TaskRunResult(
            task=task,
            session_key=session_key,
            history=tuple(history),
            final_assistant_text=final_text,
            evaluation=evaluation,
        )
