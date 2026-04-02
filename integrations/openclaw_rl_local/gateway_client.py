"""Minimal OpenClaw Gateway WebSocket client for scripted user turns."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from itertools import count
from typing import Any

import websockets


PROTOCOL_VERSION = 3
TUI_CLIENT_ID = "openclaw-tui"
TUI_CLIENT_MODE = "ui"


@dataclass(frozen=True)
class GatewayChatResult:
    run_id: str


class GatewayRequestError(RuntimeError):
    pass


class OpenClawGatewayClient:
    def __init__(
        self,
        *,
        ws_url: str,
        token: str | None,
        request_timeout_ms: int = 120_000,
    ) -> None:
        self.ws_url = ws_url
        self.token = token
        self.request_timeout_ms = request_timeout_ms
        self._ws: Any = None
        self._seq = count(1)

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.ws_url)
        await self._request(
            "connect",
            {
                "minProtocol": PROTOCOL_VERSION,
                "maxProtocol": PROTOCOL_VERSION,
                "client": {
                    "id": TUI_CLIENT_ID,
                    "displayName": "mlx-tinker-local-sim",
                    "version": "0.1.0",
                    "platform": "python",
                    "mode": TUI_CLIENT_MODE,
                },
                "role": "operator",
                "scopes": ["operator.read", "operator.write"],
                **({"auth": {"token": self.token}} if self.token else {}),
            },
        )

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send_chat(self, *, session_key: str, message: str) -> GatewayChatResult:
        payload = await self._request(
            "chat.send",
            {
                "sessionKey": session_key,
                "message": message,
                "idempotencyKey": f"{session_key}-{uuid.uuid4().hex}",
            },
        )
        if payload.get("status") not in {"started", "ok"} or not isinstance(
            payload.get("runId"), str
        ):
            raise GatewayRequestError(f"chat.send did not start correctly: {payload}")
        return GatewayChatResult(run_id=payload["runId"])

    async def wait_for_run(self, run_id: str) -> dict[str, Any]:
        payload = await self._request(
            "agent.wait",
            {"runId": run_id, "timeoutMs": self.request_timeout_ms},
            timeout_ms=self.request_timeout_ms + 10_000,
        )
        if payload.get("status") != "ok":
            raise GatewayRequestError(f"agent.wait failed for {run_id}: {payload}")
        return payload

    async def load_history(self, *, session_key: str, limit: int = 50) -> list[dict[str, Any]]:
        payload = await self._request(
            "chat.history",
            {"sessionKey": session_key, "limit": limit},
        )
        messages = payload.get("messages", [])
        return list(messages) if isinstance(messages, list) else []

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Gateway client is not connected")
        request_id = f"req-{next(self._seq)}"
        frame = {"type": "req", "id": request_id, "method": method, "params": params}
        await self._ws.send(json.dumps(frame))
        while True:
            raw = await asyncio_wait_for(self._ws.recv(), timeout_ms or self.request_timeout_ms)
            message = json.loads(raw)
            if message.get("type") != "res" or message.get("id") != request_id:
                continue
            if not message.get("ok", False):
                raise GatewayRequestError(str(message.get("error") or message))
            payload = message.get("payload", {})
            return payload if isinstance(payload, dict) else {"value": payload}


async def asyncio_wait_for(awaitable, timeout_ms: int):
    import asyncio

    return await asyncio.wait_for(awaitable, timeout=timeout_ms / 1000.0)
