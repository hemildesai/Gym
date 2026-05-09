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

"""Summarize NeMo Gym rollout JSONL files produced by sandbox eval jobs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--sample-limit", default=20, type=int)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _assistant_outputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    response = row.get("response") or {}
    return [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]


def _status(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    return str(response.get("status") or "missing")


def _error_key(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    error = response.get("error")
    if error is None:
        return "none"
    if isinstance(error, dict):
        return str(error.get("type") or error.get("code") or "error")
    return type(error).__name__


def _exception_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    exception_info = metadata.get("exception_info") or {}
    if isinstance(exception_info, dict) and exception_info.get("exception_type"):
        return str(exception_info["exception_type"])
    return "none"


def _stats(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "mean": None, "max": None}

    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        p50 = sorted_values[midpoint]
    else:
        p50 = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2

    return {
        "min": sorted_values[0],
        "p50": p50,
        "mean": mean(values),
        "max": sorted_values[-1],
    }


def _iter_rows(path: Path) -> Iterator[dict[str, Any] | None]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            if isinstance(row, dict):
                yield row
            else:
                yield None


def main() -> None:
    args = parse_args()

    completed_rows = 0
    parse_errors = 0
    reward_sum = 0.0
    assistant_turn_counts: list[int] = []
    prompt_token_lengths: list[int] = []
    generation_token_lengths: list[int] = []
    logprob_lengths: list[int] = []
    turns_with_prompt_token_ids = 0
    turns_with_generation_token_ids = 0
    turns_with_generation_log_probs = 0
    turns_with_all_trainable_fields = 0
    reward_distribution: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    exception_counts: Counter[str] = Counter()
    zero_reward_samples = []

    for maybe_row in _iter_rows(args.jsonl):
        if maybe_row is None:
            parse_errors += 1
            continue

        row = maybe_row
        completed_rows += 1
        reward = float(row.get("reward") or 0.0)
        reward_sum += reward
        reward_key = str(row.get("reward") if "reward" in row else "missing")
        reward_distribution[reward_key] += 1
        status_counts[_status(row)] += 1
        error_counts[_error_key(row)] += 1
        exception_counts[_exception_key(row)] += 1

        assistant_outputs = _assistant_outputs(row)
        assistant_turn_counts.append(len(assistant_outputs))
        for item in assistant_outputs:
            prompt_token_ids = item.get("prompt_token_ids") or []
            generation_token_ids = item.get("generation_token_ids") or []
            generation_log_probs = item.get("generation_log_probs") or []

            if prompt_token_ids:
                turns_with_prompt_token_ids += 1
                prompt_token_lengths.append(len(prompt_token_ids))
            if generation_token_ids:
                turns_with_generation_token_ids += 1
                generation_token_lengths.append(len(generation_token_ids))
            if generation_log_probs:
                turns_with_generation_log_probs += 1
                logprob_lengths.append(len(generation_log_probs))
            if prompt_token_ids and generation_token_ids and generation_log_probs:
                turns_with_all_trainable_fields += 1

        if reward == 0.0 and len(zero_reward_samples) < args.sample_limit:
            zero_reward_samples.append(
                {
                    "instance_id": row.get("instance_id"),
                    "status": _status(row),
                    "error": _error_key(row),
                    "exception": _exception_key(row),
                }
            )

    report = {
        "rollouts_jsonl": str(args.jsonl),
        "completed_rows": completed_rows,
        "jsonl_parse_errors": parse_errors,
        "reward_mean": reward_sum / completed_rows if completed_rows else None,
        "reward_sum": reward_sum,
        "reward_distribution": dict(sorted(reward_distribution.items())),
        "response_status_counts": dict(sorted(status_counts.items())),
        "response_error_counts": dict(sorted(error_counts.items())),
        "exception_counts": dict(sorted(exception_counts.items())),
        "assistant_turns_per_row": _stats(assistant_turn_counts),
        "assistant_turns_total": sum(assistant_turn_counts),
        "turns_with_prompt_token_ids": turns_with_prompt_token_ids,
        "turns_with_generation_token_ids": turns_with_generation_token_ids,
        "turns_with_generation_log_probs": turns_with_generation_log_probs,
        "turns_with_all_trainable_fields": turns_with_all_trainable_fields,
        "prompt_token_len": _stats(prompt_token_lengths),
        "generation_token_len": _stats(generation_token_lengths),
        "logprob_len": _stats(logprob_lengths),
        "zero_reward_samples": zero_reward_samples,
    }
    report_json = json.dumps(report, indent=2, sort_keys=True)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
