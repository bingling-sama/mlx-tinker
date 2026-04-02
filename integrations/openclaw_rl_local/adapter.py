"""Thin local SDK adapters for mlx-tinker-backed Tinker clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tinker


@dataclass
class LocalSamplingClientAdapter:
    client: Any
    model_name: str

    async def sample_async(self, *args, **kwargs):
        return await self.client.sample_async(*args, **kwargs)


@dataclass
class LocalTrainingClientAdapter:
    client: Any
    model_name: str

    async def forward_backward_async(self, *args, **kwargs):
        return await self.client.forward_backward_async(*args, **kwargs)

    async def optim_step_async(self, *args, **kwargs):
        return await self.client.optim_step_async(*args, **kwargs)

    async def save_state_async(self, *args, **kwargs):
        return await self.client.save_state_async(*args, **kwargs)

    async def load_state_async(self, *args, **kwargs):
        return await self.client.load_state_async(*args, **kwargs)

    async def save_weights_and_get_sampling_client_async(self, *args, **kwargs) -> LocalSamplingClientAdapter:
        sampling_client = await self.client.save_weights_and_get_sampling_client_async(*args, **kwargs)
        return LocalSamplingClientAdapter(sampling_client, self.model_name)


class LocalTinkerClientFactory:
    """Factory for local Tinker SDK clients pointed at mlx-tinker."""

    def __init__(self, *, base_url: str, api_key: str, model_name: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self._service_client = tinker.ServiceClient(base_url=base_url, api_key=api_key)

    @property
    def service_client(self):
        return self._service_client

    async def create_training_client(self, *, rank: int) -> LocalTrainingClientAdapter:
        client = await self._service_client.create_lora_training_client_async(
            base_model=self.model_name,
            rank=rank,
        )
        return LocalTrainingClientAdapter(client, self.model_name)

    async def create_sampling_client(self) -> LocalSamplingClientAdapter:
        client = await self._service_client.create_sampling_client_async(base_model=self.model_name)
        return LocalSamplingClientAdapter(client, self.model_name)

    async def create_teacher_client(
        self, *, teacher_model_name: str | None = None
    ) -> LocalSamplingClientAdapter:
        """Create a sampling client for the base model (teacher, no LoRA).

        The teacher model is the base model without any LoRA adapters applied.
        This is used for OPD and Combined methods to extract per-token teacher
        logprobs that serve as the distillation signal.

        Args:
            teacher_model_name: Override model name. Defaults to self.model_name.
        """
        model = teacher_model_name or self.model_name
        client = await self._service_client.create_sampling_client_async(base_model=model)
        return LocalSamplingClientAdapter(client, model)

    def close(self) -> None:
        holder = getattr(self._service_client, "holder", None)
        close = getattr(holder, "close", None)
        if callable(close):
            close()
