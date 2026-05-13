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

"""mini-swe-agent environment adapter backed by the Gym sandbox API."""

import asyncio
import os
import shlex
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.sandbox import Sandbox, SandboxSpec, rewrite_image
from nemo_gym.sandbox.config import SandboxProviderConfig


@dataclass
class MiniSWESandboxEnvironmentConfig:
    """Configuration for mini-swe-agent runs inside a sandbox."""

    image: str
    cwd: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    step_timeout: int = 600
    eval_timeout: int = 1800
    instance_id: str | None = None
    provider: SandboxProviderConfig | dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    conda_env: str | None = None
    activate_conda: bool = False
    user: str | int | None = "root"
    delete: bool = True


class _AsyncLoopRunner:
    """Own one event loop for async provider calls used by sync harnesses."""

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


class MiniSWESandboxEnvironment:
    """mini-swe-agent sync environment implemented with ``nemo_gym.sandbox.Sandbox``."""

    def __init__(
        self,
        *,
        config_class: type = MiniSWESandboxEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        if not self.config.provider:
            raise ValueError("MiniSWESandboxEnvironment requires provider")

        spec_config = dict(self.config.spec)
        image = spec_config.pop("image", None) or self.config.image
        image = rewrite_image(image, spec_config.pop("image_rewrites", []))

        env = dict(spec_config.pop("env", {}))
        for key in self.config.forward_env:
            value = os.getenv(key)
            if value is not None:
                env[key] = value
        env.update(self.config.env)

        self._loop_runner = _AsyncLoopRunner()
        self._sandbox = Sandbox(self.config.provider)
        self._handle = self._loop_runner.run(
            self._sandbox.create(
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
            self._sandbox.exec(
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
            self._loop_runner.run(self._sandbox.close(self._handle, delete=self.config.delete))
            self._loop_runner.run(self._sandbox.aclose())
        finally:
            self._loop_runner.close()

    def __enter__(self) -> "MiniSWESandboxEnvironment":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.cleanup()
            except Exception:
                pass
