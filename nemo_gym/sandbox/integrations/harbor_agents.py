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

"""Harbor installed-agent shims used by sandbox trajectory PoCs."""

import json
import shlex
from typing import Any

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.qwen_code import QwenCode
from harbor.environments.base import BaseSandbox
from harbor.models.task.config import MCPServerConfig


class NpmClaudeCode(ClaudeCode):
    """Claude Code agent that installs through npm on Debian uv images.

    Harbor's default Debian path uses Anthropic's native installer. On the
    uv-based sandbox image used for SWE tasks, that installer currently reports
    success without leaving a callable ``claude`` binary. The npm install path
    is already used by Harbor on Alpine images; this shim reuses that path while
    preserving Harbor's Claude Code run and ATIF parsing logic.
    """

    async def install(self, environment: BaseSandbox) -> None:
        version = f"@{self._version}" if self._version else ""
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apk >/dev/null 2>&1; then "
                "  apk add --no-cache curl bash nodejs npm; "
                "elif command -v apt-get >/dev/null 2>&1; then "
                "  apt-get update && "
                "  DEBIAN_FRONTEND=noninteractive apt-get install -y "
                "  curl bash nodejs npm; "
                "elif command -v yum >/dev/null 2>&1; then "
                "  yum install -y curl bash nodejs npm; "
                "else "
                "  echo 'Warning: no known package manager found' >&2; "
                "fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"npm install -g @anthropic-ai/claude-code{version}; "
                "claude --version"
            ),
        )


class CachedMiniSweAgent(MiniSweAgent):
    """Mini SWE Agent shim that trusts a pre-installed sandbox CLI.

    OpenSandbox SWE runs can prepare sandboxes before Harbor borrows them. When
    that prewarm step has already installed ``mini-swe-agent``, repeating the
    full apt/uv install in Harbor's per-trial setup only burns rollout time.
    """

    async def install(self, environment: BaseSandbox) -> None:
        version_check = "true"
        if self._version:
            expected_version = shlex.quote(self._version)
            version_check = (
                "installed_version=$(uv tool list 2>/dev/null | "
                "awk '/^mini-swe-agent / {print $2; exit}' | sed 's/^v//'); "
                f'test "$installed_version" = {expected_version}'
            )
        version_probe = (
            '. "$HOME/.local/bin/env" 2>/dev/null || true; '
            "command -v mini-swe-agent >/dev/null 2>&1 && "
            "mini-swe-agent --help >/dev/null 2>&1 && "
            f"{version_check}"
        )
        result = await self.exec_as_agent(environment, command=version_probe)
        if result.return_code == 0:
            return
        await super().install(environment)


class ConfiguredQwenCode(QwenCode):
    """Qwen Code agent with explicit OpenAI-compatible generation settings.

    Harbor's stock Qwen Code integration leaves the CLI's content-generator
    timeout at the tool default. In the uv SWE images this has produced
    immediate ``Request timeout after 0s`` failures against vLLM. This shim
    keeps Harbor's install, run, and trajectory parsing behavior intact, but
    always writes a Qwen settings file before launching the CLI.
    """

    def __init__(
        self,
        *args: Any,
        content_generator_timeout_ms: int = 1_200_000,
        content_generator_retries: int = 2,
        context_window_size: int = 65_536,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._content_generator_timeout_ms = content_generator_timeout_ms
        self._content_generator_retries = content_generator_retries
        self._context_window_size = context_window_size
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._enable_thinking = enable_thinking

    def _qwen_settings(self) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "timeout": self._content_generator_timeout_ms,
            "maxRetries": self._content_generator_retries,
            "contextWindowSize": self._context_window_size,
            "extra_body": {"enable_thinking": self._enable_thinking},
            "samplingParams": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                "top_k": self._top_k,
            },
        }
        settings: dict[str, Any] = {
            "contentGenerator": {
                "timeout": self._content_generator_timeout_ms,
                "maxRetries": self._content_generator_retries,
            },
            "modelProviders": {
                "openai": [
                    {
                        "id": self.model_name,
                        "name": self.model_name,
                        "baseUrl": "$OPENAI_BASE_URL",
                        "envKey": "OPENAI_API_KEY",
                        "generationConfig": generation_config,
                    }
                ]
            },
            "security": {"auth": {"selectedType": "openai"}},
            "model": {
                "name": self.model_name,
                "generationConfig": generation_config,
            },
        }

        if self.mcp_servers:
            settings["mcpServers"] = {
                server.name: _qwen_mcp_server_config(server)
                for server in self.mcp_servers
            }

        return settings

    def _build_register_mcp_servers_command(self) -> str:
        config = json.dumps(self._qwen_settings(), indent=2)
        return f"mkdir -p ~/.qwen && printf %s {shlex.quote(config)} > ~/.qwen/settings.json"


def _qwen_mcp_server_config(server: MCPServerConfig) -> dict[str, Any]:
    if server.transport == "stdio":
        return {"command": server.command, "args": server.args}
    if server.transport == "streamable-http":
        return {"httpUrl": server.url}
    return {"url": server.url}
