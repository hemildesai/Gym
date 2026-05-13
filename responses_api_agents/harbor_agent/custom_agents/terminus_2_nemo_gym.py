# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import base64
import json
import shlex
import time
from pathlib import Path
from typing import Any, Literal

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseSandbox
from harbor.llms.base import BaseLLM, LLMBackend
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


class MemoryLimitExceededError(Exception):
    """Compatibility shim for non-Singularity Harbor environments."""


class GymTmuxSession(TmuxSession):
    """Tmux session variant that pastes literal text instead of flag-like keys."""

    _TMUX_LITERAL_CHUNK_SIZE = 4096
    _TMUX_CONTROL_CWD = "/"
    _TMUX_SPECIAL_KEYS = {
        "BSpace",
        "C-c",
        "C-d",
        "Delete",
        "Down",
        "End",
        "Escape",
        "Home",
        "Left",
        "PageDown",
        "PageUp",
        "Right",
        "Space",
        "Tab",
        "Up",
    }

    def _tmux_send_keys(self, keys: list[str]) -> list[str]:
        """Build tmux commands, avoiding ``send-keys`` for arbitrary text."""
        commands: list[str] = []
        for key in keys:
            if self._is_special_tmux_key(key):
                commands.append(
                    "tmux send-keys -t "
                    + shlex.quote(self._session_name)
                    + " "
                    + shlex.quote(key)
                )
            else:
                commands.extend(self._tmux_paste_literal(key))
        return commands

    def _is_special_tmux_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS or key in self._TMUX_SPECIAL_KEYS or key.startswith(("C-", "M-"))

    def _tmux_paste_literal(self, text: str) -> list[str]:
        commands = []
        for index in range(0, len(text), self._TMUX_LITERAL_CHUNK_SIZE):
            chunk = text[index : index + self._TMUX_LITERAL_CHUNK_SIZE]
            encoded = base64.b64encode(chunk.encode("utf-8")).decode("ascii")
            buffer_name = f"nemo-gym-keys-{abs(hash((self._session_name, index)))}"
            commands.append(
                "printf %s "
                + shlex.quote(encoded)
                + " | base64 -d | tmux load-buffer -b "
                + shlex.quote(buffer_name)
                + " - && tmux paste-buffer -b "
                + shlex.quote(buffer_name)
                + " -t "
                + shlex.quote(self._session_name)
                + " && tmux delete-buffer -b "
                + shlex.quote(buffer_name)
            )
        return commands

    async def is_session_alive(self) -> bool:
        result = await self.environment.exec(
            command="tmux has-session -t {}".format(self._session_name),
            user=self._user,
            cwd=self._TMUX_CONTROL_CWD,
        )
        return result.return_code == 0

    async def _send_blocking_keys(
        self,
        keys: list[str],
        max_timeout_sec: float,
    ) -> None:
        start_time_sec = time.time()

        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(
                command=command,
                user=self._user,
                cwd=self._TMUX_CONTROL_CWD,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self.environment.session_id}: failed to send blocking keys: {result.stderr}"
                )

        result = await self.environment.exec(
            f"timeout {max_timeout_sec}s tmux wait done",
            user=self._user,
            cwd=self._TMUX_CONTROL_CWD,
        )
        if result.return_code != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")

        elapsed_time_sec = time.time() - start_time_sec
        self._logger.debug(f"Blocking command completed in {elapsed_time_sec:.2f}s.")

    async def _send_non_blocking_keys(
        self,
        keys: list[str],
        min_timeout_sec: float,
    ) -> None:
        start_time_sec = time.time()

        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(
                command=command,
                user=self._user,
                cwd=self._TMUX_CONTROL_CWD,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self.environment.session_id}: failed to send non-blocking keys: {result.stderr}"
                )

        elapsed_time_sec = time.time() - start_time_sec
        if elapsed_time_sec < min_timeout_sec:
            await asyncio.sleep(min_timeout_sec - elapsed_time_sec)

    async def capture_pane(self, capture_entire: bool = False) -> str:
        result = await self.environment.exec(
            self._tmux_capture_pane(capture_entire=capture_entire),
            user=self._user,
            cwd=self._TMUX_CONTROL_CWD,
        )
        return result.stdout or ""


class Terminus2NemoGym(Terminus2):
    """Terminus2 variant that uses a NeMo Gym model server-compatible BaseLLM."""

    @staticmethod
    def name() -> str:
        return "terminus-2-nemo-gym"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "default"] | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        trajectory_config: dict | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        llm: BaseLLM | None = None,
        interleaved_thinking: bool = False,
        responses_create_params: dict[str, Any] | None = None,
        nemo_model_server_timeout_sec: float = 120.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._provided_nemo_gym_llm: BaseLLM | None = llm
        if self._provided_nemo_gym_llm is None:
            if model_name is None:
                raise ValueError("model_name is required for Terminus2NemoGym")
            if api_base is None:
                raise ValueError("api_base is required for Terminus2NemoGym when llm is not provided")

            self._provided_nemo_gym_llm = NemoGymLLM(
                model_name=model_name,
                api_base=api_base,
                collect_rollout_details=collect_rollout_details,
                model_info=model_info,
                responses_create_params=responses_create_params,
                timeout_sec=nemo_model_server_timeout_sec,
            )

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            enable_summarize=enable_summarize,
            proactive_summarization_threshold=proactive_summarization_threshold,
            max_thinking_tokens=max_thinking_tokens,
            model_info=model_info,
            trajectory_config=trajectory_config,
            tmux_pane_width=tmux_pane_width,
            tmux_pane_height=tmux_pane_height,
            store_all_messages=store_all_messages,
            record_terminal_session=record_terminal_session,
            interleaved_thinking=interleaved_thinking,
            *args,
            **kwargs,
        )

    def _init_llm(
        self,
        llm_backend: LLMBackend | str,
        model_name: str,
        temperature: float,
        collect_rollout_details: bool,
        llm_kwargs: dict | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        """Return the prebuilt Gym LLM instead of Terminus2's LiteLLM backend."""
        if self._provided_nemo_gym_llm is None:
            raise ValueError("Terminus2NemoGym LLM was not initialized")
        return self._provided_nemo_gym_llm

    async def setup(self, environment: BaseSandbox) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = GymTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    async def run(self, instruction: str, environment: BaseSandbox, context: AgentContext) -> None:
        """Override run() to gracefully handle agent errors.

        The parent's run() has a finally block that saves rollout_details and
        dumps the trajectory before any exception propagates. By catching
        exceptions here, we let Harbor's trial system proceed normally with the
        verifier — returning the agent's conversation history from all completed
        turns (reward will be 0 for incomplete work) instead of crashing the
        entire rollout batch.
        """
        self._memory_limit_exceeded = False
        try:
            await super().run(instruction, environment, context)
        except MemoryLimitExceededError as e:
            self._memory_limit_exceeded = True
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        except Exception as e:
            self.logger.info(f"Agent error: {type(e).__name__}: {e}. Returning history from completed turns.")
        finally:
            self._write_agent_error_flags()

    def _write_agent_error_flags(self) -> None:
        """Write agent error flags to disk for app.py to pick up."""
        try:
            flags: dict[str, bool] = {
                "memory_limit_exceeded": self._memory_limit_exceeded,
            }
            llm = getattr(self, "_llm", None)
            if llm and isinstance(llm, NemoGymLLM):
                flags["context_length_exceeded"] = llm.context_length_exceeded
            (self.logs_dir / "agent_error_flags.json").write_text(json.dumps(flags))
        except Exception:
            pass  # Don't let flag-writing failures break the agent
