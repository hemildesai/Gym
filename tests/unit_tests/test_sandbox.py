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

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_gym.sandbox import Sandbox, SandboxExecResult, SandboxHandle, SandboxSpec, register_provider
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider_module
from nemo_gym.sandbox.providers.opensandbox.provider import (
    IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY,
    IMAGE_PULL_POLICY_EXTENSION_KEY,
    OpenSandboxProvider,
)
from responses_api_agents.mini_swe_agent.sandbox_environment import MiniSWESandboxEnvironment


class FakeSandboxProvider:
    name = "fake"
    last_instance: "FakeSandboxProvider | None" = None

    def __init__(self, marker: str = "default") -> None:
        self.marker = marker
        self.created_specs: list[SandboxSpec] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.closed: list[tuple[SandboxHandle, bool]] = []
        FakeSandboxProvider.last_instance = self

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created_specs.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw={"spec": spec})

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        del allow_partial
        return [await self.create(spec) for _ in range(count)]

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw={})

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
        self.exec_calls.append(
            {
                "handle": handle,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        del handle, target_path, data

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        del handle, source_path
        return b""

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        del handle, source_path, target_path

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        del handle, source_path, target_path

    async def close(self, handle: SandboxHandle, *, delete: bool) -> None:
        self.closed.append((handle, delete))


def test_sandbox_facade_uses_public_provider_api() -> None:
    asyncio.run(_assert_sandbox_facade_uses_public_provider_api())


async def _assert_sandbox_facade_uses_public_provider_api() -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)

    sandbox = Sandbox({"name": provider_name, "kwargs": {"marker": "configured"}})
    handle = await sandbox.create(SandboxSpec(image="image:tag", metadata={"suite": "unit"}))

    provider = FakeSandboxProvider.last_instance
    assert provider is not None
    assert provider.marker == "configured"
    assert provider.created_specs[0].image == "image:tag"
    assert provider.created_specs[0].metadata == {"suite": "unit"}

    result = await sandbox.exec(handle, "pytest -q", cwd="/repo", timeout_s=60, user="agent")
    assert result == SandboxExecResult(stdout="ok", stderr=None, return_code=0)
    assert provider.exec_calls[0] == {
        "handle": handle,
        "command": "pytest -q",
        "cwd": "/repo",
        "env": None,
        "timeout_s": 60,
        "user": "agent",
    }

    await sandbox.delete(handle)
    assert provider.closed[0] == (handle, True)


def test_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch))


async def _assert_opensandbox_sdk_create_receives_default_image_pull_policy(monkeypatch) -> None:
    class FakeSDKSandbox:
        create_calls: list[dict[str, Any]] = []

        def __init__(self, sandbox_id: str) -> None:
            self.id = sandbox_id

        @classmethod
        async def create(cls, **kwargs: Any) -> "FakeSDKSandbox":
            cls.create_calls.append(kwargs)
            return cls("sdk-sandbox-1")

    monkeypatch.setattr(
        opensandbox_provider_module,
        "_require_opensandbox_sdk",
        lambda: (FakeSDKSandbox, object, object, object, object),
    )

    provider = OpenSandboxProvider(create_probe_command=None, sdk_max_connections=None)
    monkeypatch.setattr(provider, "_connection_config", lambda request_timeout_s=None: object())

    handle = await provider.create(
        SandboxSpec(
            image="image:tag",
            metadata={
                "harbor_instance_id": "swebench::django__django-10880",
                "long": f"bad:{'x' * 80}:",
            },
        )
    )

    assert handle.sandbox_id == "sdk-sandbox-1"
    metadata = FakeSDKSandbox.create_calls[0]["metadata"]
    assert metadata["harbor_instance_id"] == "swebench_django__django-10880"
    assert metadata["long"] == ("bad_" + "x" * 59)
    extensions = FakeSDKSandbox.create_calls[0]["extensions"]
    assert extensions[IMAGE_PULL_POLICY_EXTENSION_KEY] == "IfNotPresent"
    assert extensions[IMAGE_PULL_POLICY_ANNOTATION_EXTENSION_KEY] == "IfNotPresent"


def test_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    asyncio.run(_assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch))


async def _assert_opensandbox_create_probe_can_require_stable_successes(monkeypatch) -> None:
    provider = OpenSandboxProvider(
        create_probe_command="true",
        create_probe_expected_stdout=None,
        create_probe_stable_count=3,
        create_probe_stable_delay_s=0,
        sdk_max_connections=None,
    )
    calls: list[dict[str, Any]] = []

    async def fake_exec(
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        calls.append(
            {
                "handle": handle,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
            }
        )
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(provider, "exec", fake_exec)
    handle = SandboxHandle(sandbox_id="sdk-sandbox-0", provider_name="opensandbox", raw=object())

    await provider._verify_created_handle(handle)

    assert [call["command"] for call in calls] == ["true", "true", "true"]
    assert all(call["timeout_s"] == 30 for call in calls)
    assert all(call["user"] == "root" for call in calls)


def test_mini_swe_sandbox_environment_owns_conda_setup(monkeypatch) -> None:
    provider_name = f"fake-{uuid4().hex}"
    register_provider(provider_name, FakeSandboxProvider)
    monkeypatch.setenv("FORWARDED_KEY", "forwarded-value")

    env = MiniSWESandboxEnvironment(
        image="upstream/image:tag",
        cwd="/testbed",
        provider={"name": provider_name, "kwargs": {"marker": "configured"}},
        spec={
            "image_rewrites": [{"from": "upstream/", "to": "mirror/"}],
            "metadata": {"suite": "unit"},
            "resources": {"cpu": "1"},
        },
        env={"STATIC_KEY": "static-value"},
        forward_env=["FORWARDED_KEY"],
        conda_env="testbed",
        activate_conda=True,
        user="agent",
        delete=True,
    )

    try:
        provider = FakeSandboxProvider.last_instance
        assert provider is not None
        assert provider.marker == "configured"
        assert provider.created_specs[0].image == "mirror/image:tag"
        assert provider.created_specs[0].env == {
            "FORWARDED_KEY": "forwarded-value",
            "STATIC_KEY": "static-value",
        }

        result = env.execute("pytest -q", is_eval=True)
        assert result == {"output": "ok", "returncode": 0, "exception_info": ""}
        exec_call = provider.exec_calls[0]
        assert exec_call["cwd"] == "/"
        assert exec_call["timeout_s"] == 1800
        assert exec_call["user"] == "agent"
        assert "conda activate testbed" in exec_call["command"]
        assert exec_call["command"].endswith("pytest -q")
    finally:
        env.cleanup()

    assert FakeSandboxProvider.last_instance is not None
    assert FakeSandboxProvider.last_instance.closed[0][1] is True
