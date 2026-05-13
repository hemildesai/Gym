# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-neutral public sandbox API.

This module is the boundary Gym code should use when it needs a sandbox.
Provider packages implement the lower-level protocol; callers create, execute
inside, move files through, and delete sandboxes through this facade.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from nemo_gym.sandbox.config import SandboxProviderConfig
from nemo_gym.sandbox.providers import (
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    create_provider,
)


def rewrite_image(image: str | None, rewrites: list[dict[str, str]]) -> str | None:
    """Apply ordered image-prefix rewrites used by sandbox configs."""
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image


class Sandbox:
    """Public facade for provider-backed sandbox operations."""

    def __init__(self, provider: SandboxProviderConfig | SandboxProvider) -> None:
        self._provider = (
            create_provider(cast(SandboxProviderConfig, provider)) if isinstance(provider, Mapping) else provider
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return await self._provider.create(spec)

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        return await self._provider.create_batch(spec, count, allow_partial=allow_partial)

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        return await self._provider.connect(sandbox_id)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        return await self._provider.exec(
            handle,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
        )

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        await self._provider.write_file(handle, target_path, data)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        return await self._provider.read_file(handle, source_path)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await self._provider.upload_file(handle, source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        await self._provider.download_file(handle, source_path, target_path)

    async def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        await self._provider.close(handle, delete=delete)

    async def delete(self, handle: SandboxHandle) -> None:
        await self.close(handle, delete=True)

    async def aclose(self) -> None:
        close_provider = getattr(self._provider, "aclose", None)
        if close_provider is not None:
            await close_provider()

    def handle_reference(self, handle: SandboxHandle) -> Any:
        make_reference = getattr(self._provider, "handle_reference", None)
        if make_reference is None:
            return handle
        return make_reference(handle)

    async def materialize_handle(self, value: Any) -> SandboxHandle:
        materialize = getattr(self._provider, "materialize_handle", None)
        if materialize is None:
            if isinstance(value, SandboxHandle):
                return value
            raise ValueError(f"Provider {self.provider_name!r} cannot materialize handle references")
        result = materialize(value)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, SandboxHandle):
            raise TypeError(f"materialize_handle must return SandboxHandle, got {type(result).__name__}")
        return result
