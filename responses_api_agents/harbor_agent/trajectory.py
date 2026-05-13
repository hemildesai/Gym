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

"""Harbor sandbox trajectory and policy-trace conversion helpers.

Installed agents can run unmodified inside a sandbox, but GRPO needs
assistant token IDs and generation logprobs. A policy proxy or Harbor-managed
LLM client should emit the shape consumed here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict


@dataclass(frozen=True)
class SandboxRolloutContext:
    """Policy endpoint context supplied to sandbox-backed rollout consumers."""

    model_name: str
    base_urls: list[str | None]


class RolloutDetail(TypedDict):
    """Per-turn token and logprob trace for one linear chat segment."""

    prompt_token_ids: list[list[int]]
    completion_token_ids: list[list[int]]
    logprobs: NotRequired[list[list[float]]]
    extra: NotRequired[dict[str, list[Any]]]


class SandboxTrajectory(TypedDict):
    """One completed sandbox trajectory in trainable NeMo-RL form."""

    rollout_details: list[RolloutDetail]
    reward: float
    full_result: NotRequired[dict[str, Any]]
    agent_name: NotRequired[str]
    truncated: NotRequired[bool]
    loss_multiplier: NotRequired[float]
    stop_reason: NotRequired[str]


@dataclass(frozen=True)
class ConvertedTrajectory:
    """Trajectory converted into NeMo-RL message-log fields."""

    message_log: list[dict[str, Any]]
    input_message_log: list[dict[str, Any]]
    reward: float
    full_result: dict[str, Any]
    agent_name: str | None
    truncated: bool
    loss_multiplier: float


def load_policy_trace_jsonl(path: Path) -> list[RolloutDetail]:
    """Load proxy-recorded policy trace records from JSONL."""
    rollout_detail: RolloutDetail = {
        "prompt_token_ids": [],
        "completion_token_ids": [],
    }
    logprobs: list[list[float]] = []
    extra: dict[str, list[Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("prompt_token_ids") is not None:
                rollout_detail["prompt_token_ids"].append(record["prompt_token_ids"])
            if record.get("completion_token_ids") is not None:
                rollout_detail["completion_token_ids"].append(record["completion_token_ids"])
            if record.get("logprobs") is not None:
                logprobs.append(record["logprobs"])
            if isinstance(record.get("extra"), dict):
                for key, value in record["extra"].items():
                    extra.setdefault(key, []).append(value)
    if logprobs:
        rollout_detail["logprobs"] = logprobs
    if extra:
        rollout_detail["extra"] = extra
    return [rollout_detail]


def _validate_turn(
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    logprobs: list[float] | None,
    require_trainable: bool,
    turn_idx: int,
) -> None:
    if not prompt_ids:
        raise ValueError(f"Sandbox trajectory turn {turn_idx} has empty prompt tokens")
    if not completion_ids:
        raise ValueError(f"Sandbox trajectory turn {turn_idx} has empty completion tokens")
    if logprobs is None:
        if require_trainable:
            raise ValueError(
                "Sandbox trajectory is missing generation logprobs. "
                "Set env.sandbox.trajectory.require_trainable=false only for offline collection."
            )
        return
    if len(completion_ids) != len(logprobs):
        raise ValueError(
            f"Sandbox trajectory turn {turn_idx} has {len(completion_ids)} "
            f"completion tokens but {len(logprobs)} logprobs"
        )


def convert_rollout_details_to_message_log(
    rollout_details: list[RolloutDetail],
    *,
    require_trainable: bool,
) -> list[dict[str, Any]]:
    """Convert Harbor/proxy rollout details into NeMo-RL message logs.

    When a turn prompt is a contiguous extension of earlier turns, only the
    delta prompt tokens are emitted. Some installed agents compact, summarize,
    or resend shorter contexts between tool calls; those turns are represented
    as a fresh user prompt so assistant token/logprob ownership stays explicit.
    """
    if len(rollout_details) != 1:
        raise ValueError("Sandbox PoC currently supports one linear rollout detail segment per trajectory")
    import torch

    rollout_detail = rollout_details[0]
    prompts = rollout_detail["prompt_token_ids"]
    completions = rollout_detail["completion_token_ids"]
    logprobs_by_turn = rollout_detail.get("logprobs", None)
    if len(prompts) != len(completions):
        raise ValueError(f"Sandbox trajectory has {len(prompts)} prompt turns and {len(completions)} completion turns")
    if logprobs_by_turn is not None and len(logprobs_by_turn) != len(completions):
        raise ValueError(
            f"Sandbox trajectory has {len(completions)} completion turns and {len(logprobs_by_turn)} logprob turns"
        )

    seen_token_ids: list[int] = []
    message_log: list[dict[str, Any]] = []
    for turn_idx, completion_ids in enumerate(completions):
        prompt_ids = prompts[turn_idx]
        turn_logprobs = None
        if logprobs_by_turn is not None:
            turn_logprobs = logprobs_by_turn[turn_idx]

        _validate_turn(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=turn_logprobs,
            require_trainable=require_trainable,
            turn_idx=turn_idx,
        )

        if seen_token_ids == prompt_ids[: len(seen_token_ids)]:
            prompt_delta = prompt_ids[len(seen_token_ids) :]
        else:
            prompt_delta = prompt_ids
        message_log.append(
            {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(prompt_delta, dtype=torch.long),
            }
        )
        assistant_message = {
            "role": "assistant",
            "content": "",
            "token_ids": torch.tensor(completion_ids, dtype=torch.long),
        }
        if turn_logprobs is not None:
            assistant_message["generation_logprobs"] = torch.tensor(turn_logprobs, dtype=torch.float32)
        message_log.append(assistant_message)

        seen_token_ids = [*prompt_ids, *completion_ids]

    return message_log


def convert_sandbox_trajectory(
    trajectory: SandboxTrajectory,
    *,
    require_trainable: bool,
) -> ConvertedTrajectory:
    """Convert one sandbox trajectory to the NeMo-RL rollout contract."""
    message_log = convert_rollout_details_to_message_log(
        trajectory["rollout_details"], require_trainable=require_trainable
    )
    full_result = dict(trajectory.get("full_result", {}))
    full_result["reward"] = trajectory["reward"]
    if "agent_name" in trajectory:
        full_result["agent_name"] = trajectory["agent_name"]
    if "stop_reason" in trajectory:
        full_result["stop_reason"] = trajectory["stop_reason"]

    return ConvertedTrajectory(
        message_log=message_log,
        input_message_log=message_log[:1],
        reward=trajectory["reward"],
        full_result=full_result,
        agent_name=trajectory.get("agent_name", None),
        truncated=bool(trajectory.get("truncated", False)),
        loss_multiplier=float(trajectory.get("loss_multiplier", 1.0)),
    )
