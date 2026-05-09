# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mini-swe-agent environment backed by the NeMo-RL sandbox provider API."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import os
import shlex
import threading
from typing import Any


@dataclass
class OpenSandboxMiniSWEEnvironmentConfig:
    image: str
    cwd: str = "/testbed"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    step_timeout: int = 600
    eval_timeout: int = 1800
    instance_id: str | None = None
    provider: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    conda_env: str | None = "testbed"
    activate_conda: bool = True
    user: str | int | None = "root"
    delete: bool = True
    cache_dir_template: str | None = None


class _AsyncLoopRunner:
    """Own one event loop for all async provider calls in this sync adapter."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro: Any, timeout_s: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


def _rewrite_image(image: str, rewrites: list[dict[str, str]]) -> str:
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image


class OpenSandboxMiniSWEEnvironment:
    """Sync mini-swe-agent environment using ``SandboxProvider`` / ``SandboxSpec``.

    mini-swe-agent expects a small synchronous object with ``execute`` and
    ``cleanup`` methods. The provider remains async underneath; this adapter
    bridges that boundary without introducing another runtime provider layer.
    """

    def __init__(
        self,
        *,
        config_class: type = OpenSandboxMiniSWEEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        from nemo_rl.sandbox.providers import SandboxSpec, create_provider

        self.config = config_class(**kwargs)
        if not self.config.provider:
            raise ValueError("OpenSandbox mini-swe-agent environment requires provider")

        spec_config = dict(self.config.spec)
        image = spec_config.pop("image", None) or self.config.image
        image = _rewrite_image(image, spec_config.pop("image_rewrites", []))

        env = dict(spec_config.pop("env", {}))
        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                env[key] = value
        env.update(self.config.env)

        self._loop_runner = _AsyncLoopRunner()
        self._provider = create_provider(self.config.provider)
        self._handle = self._loop_runner.run(
            self._provider.create(
                SandboxSpec(
                    image=image,
                    snapshot_id=spec_config.pop("snapshot_id", None),
                    timeout_s=spec_config.pop("timeout_s", None),
                    ready_timeout_s=spec_config.pop("ready_timeout_s", None),
                    env=env,
                    metadata={
                        **spec_config.pop("metadata", {}),
                        "nemo_gym_agent": "mini_swe_agent",
                        "instance_id": (self.config.instance_id or "unknown")[:63],
                    },
                    resources=spec_config.pop("resources", {}),
                    entrypoint=spec_config.pop("entrypoint", None),
                    extensions=spec_config.pop("extensions", {}),
                    platform=spec_config.pop("platform", None),
                    volumes=spec_config.pop("volumes", None),
                    skip_health_check=spec_config.pop("skip_health_check", None),
                )
            )
        )
        self._closed = False

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {**self.config.__dict__, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.__dict__,
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _command(self, command: str, cwd: str) -> str:
        if not self.config.activate_conda or not self.config.conda_env:
            return command
        quoted_cwd = shlex.quote(cwd)
        quoted_env = shlex.quote(self.config.conda_env)
        return (
            f"cd {quoted_cwd} && "
            "source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {quoted_env} && "
            f"{command}"
        )

    def execute(self, command: str, cwd: str = "", is_eval: bool = False) -> dict[str, Any]:
        timeout_s = self.config.eval_timeout if is_eval else self.config.step_timeout
        exec_cwd = cwd or self.config.cwd
        result = self._loop_runner.run(
            self._provider.exec(
                self._handle,
                self._command(command, exec_cwd),
                cwd="/",
                timeout_s=timeout_s,
                user=self.config.user,
            )
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return {
            "output": output,
            "returncode": result.return_code,
            "exception_info": "",
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop_runner.run(self._provider.close(self._handle, delete=self.config.delete))
            aclose = getattr(self._provider, "aclose", None)
            if aclose is not None:
                self._loop_runner.run(aclose())
        finally:
            self._loop_runner.close()

    def __enter__(self) -> "OpenSandboxMiniSWEEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.cleanup()
            except Exception:
                pass
