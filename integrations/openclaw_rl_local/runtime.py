"""Runtime URL helpers for host/container networking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ContainerRuntime(str, Enum):
    PODMAN = "podman"
    DOCKER = "docker"
    HOST = "host"


def default_container_host(runtime: ContainerRuntime) -> str:
    if runtime == ContainerRuntime.HOST:
        return "127.0.0.1"
    if runtime == ContainerRuntime.PODMAN:
        return "host.containers.internal"
    return "host.docker.internal"


def resolve_provider_base_url(
    runtime: ContainerRuntime,
    proxy_port: int,
    explicit_base_url: str | None = None,
) -> str:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    return f"http://{default_container_host(runtime)}:{proxy_port}/v1"


@dataclass(frozen=True)
class RuntimeUrls:
    mlx_tinker_base_url: str
    proxy_base_url: str
    provider_base_url: str


def build_runtime_urls(
    *,
    runtime: ContainerRuntime,
    mlx_tinker_base_url: str,
    proxy_host: str,
    proxy_port: int,
    provider_base_url: str | None = None,
) -> RuntimeUrls:
    proxy_base_url = f"http://{proxy_host}:{proxy_port}/v1".rstrip("/")
    return RuntimeUrls(
        mlx_tinker_base_url=mlx_tinker_base_url.rstrip("/"),
        proxy_base_url=proxy_base_url,
        provider_base_url=resolve_provider_base_url(runtime, proxy_port, provider_base_url),
    )
