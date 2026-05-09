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

"""Summarize Harbor agent trajectory timing from NeMo Gym rollout JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


DEFAULT_JOBS_ROOT = Path(os.environ.get("HARBOR_JOBS_DIR", "/tmp/harbor_jobs"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _iter_rows(path: Path) -> Iterator[dict[str, Any] | None]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            yield row if isinstance(row, dict) else None


def _parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _message_timestamp(message: dict[str, Any]) -> float | None:
    extra = message.get("extra")
    if isinstance(extra, dict):
        timestamp = _parse_timestamp(extra.get("timestamp"))
        if timestamp is not None:
            return timestamp
    for key in ("timestamp", "created_at", "time"):
        timestamp = _parse_timestamp(message.get(key))
        if timestamp is not None:
            return timestamp
    return None


def _nested(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _agent_started_at(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    paths = (
        ("agent_execution", "started_at"),
        ("agent_result", "started_at"),
        ("agent_result", "start_time"),
        ("timing", "agent_started_at"),
    )
    for path in paths:
        timestamp = _parse_timestamp(_nested(metadata, path))
        if timestamp is not None:
            return timestamp
    return None


def _agent_finished_at(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    paths = (
        ("agent_execution", "finished_at"),
        ("agent_result", "finished_at"),
        ("agent_result", "end_time"),
        ("timing", "agent_finished_at"),
    )
    for path in paths:
        timestamp = _parse_timestamp(_nested(metadata, path))
        if timestamp is not None:
            return timestamp
    return None


def _metadata_trial_path(row: dict[str, Any]) -> Path | None:
    metadata = row.get("metadata") or {}
    keys = (
        "trial_dir",
        "trial_path",
        "trial_uri",
        "result_dir",
        "result_path",
    )
    for key in keys:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value.startswith("file://"):
            value = value[len("file://") :]
        path = Path(value)
        if path.name == "result.json":
            path = path.parent
        if path.exists():
            return path
    return None


def _job_name(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    for key in ("job_name", "job_id", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _build_job_index(jobs_root: Path) -> dict[str, Path]:
    if not jobs_root.exists():
        return {}
    index: dict[str, Path] = {}
    for result_path in jobs_root.rglob("result.json"):
        trial_dir = result_path.parent
        job_dir = trial_dir.parent
        index.setdefault(job_dir.name, trial_dir)
    return index


def _trial_dir(row: dict[str, Any], job_index: dict[str, Path]) -> Path | None:
    metadata_path = _metadata_trial_path(row)
    if metadata_path is not None:
        return metadata_path
    job_name = _job_name(row)
    if job_name is None:
        return None
    return job_index.get(job_name)


def _load_trajectory(trial_dir: Path) -> dict[str, Any] | None:
    candidates = (
        trial_dir / "agent" / "mini-swe-agent.trajectory.json",
        trial_dir / "agent" / "trajectory.json",
        trial_dir / "trajectory.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(trajectory, dict):
            return trajectory | {"_trajectory_path": str(path)}
    return None


def _messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    messages = trajectory.get("messages")
    if isinstance(messages, list):
        return [item for item in messages if isinstance(item, dict)]
    return []


def _is_assistant_message(message: dict[str, Any]) -> bool:
    if message.get("role") == "assistant" or message.get("source") == "agent":
        return True
    extra = message.get("extra")
    return (
        message.get("type") is None
        and isinstance(extra, dict)
        and isinstance(extra.get("actions"), list)
    )


def _is_tool_output(message: dict[str, Any]) -> bool:
    return message.get("type") == "function_call_output"


def _action_count(message: dict[str, Any]) -> int:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return 0
    actions = extra.get("actions")
    return len(actions) if isinstance(actions, list) else 0


def _usage(row: dict[str, Any]) -> dict[str, int]:
    usage = (row.get("response") or {}).get("usage") or {}
    output_details = usage.get("output_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int(input_details.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _analyze_timing(row: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    messages = _messages(trajectory)
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if _is_assistant_message(message) and _message_timestamp(message) is not None
    ]
    all_timestamps = [
        timestamp
        for message in messages
        if (timestamp := _message_timestamp(message)) is not None
    ]
    start_ts = _agent_started_at(row)
    finish_ts = _agent_finished_at(row)
    if start_ts is None and all_timestamps:
        start_ts = min(all_timestamps)
    if finish_ts is None and all_timestamps:
        finish_ts = max(all_timestamps)
    if start_ts is not None and all_timestamps:
        start_ts = min(start_ts, min(all_timestamps))
    if finish_ts is not None and all_timestamps:
        finish_ts = max(finish_ts, max(all_timestamps))

    inference_wait_s = 0.0
    tool_execution_s = 0.0
    previous_ready_ts = start_ts
    tool_outputs = 0
    tool_calls = 0
    parallel_tool_turns = 0

    for position, assistant_index in enumerate(assistant_indices):
        assistant_message = messages[assistant_index]
        assistant_ts = _message_timestamp(assistant_message)
        if assistant_ts is None:
            continue
        if previous_ready_ts is not None and assistant_ts >= previous_ready_ts:
            inference_wait_s += assistant_ts - previous_ready_ts

        actions = _action_count(assistant_message)
        tool_calls += actions
        if actions > 1:
            parallel_tool_turns += 1

        next_assistant_index = (
            assistant_indices[position + 1]
            if position + 1 < len(assistant_indices)
            else len(messages)
        )
        tool_timestamps = [
            timestamp
            for message in messages[assistant_index + 1 : next_assistant_index]
            if _is_tool_output(message)
            and (timestamp := _message_timestamp(message)) is not None
        ]
        tool_outputs += len(tool_timestamps)
        if tool_timestamps:
            last_tool_ts = max(tool_timestamps)
            if last_tool_ts >= assistant_ts:
                tool_execution_s += last_tool_ts - assistant_ts
            previous_ready_ts = last_tool_ts
        else:
            previous_ready_ts = assistant_ts

    agent_execution_s = None
    if start_ts is not None and finish_ts is not None and finish_ts >= start_ts:
        agent_execution_s = finish_ts - start_ts
    attributed_s = inference_wait_s + tool_execution_s
    unknown_agent_s = (
        max(0.0, agent_execution_s - attributed_s)
        if agent_execution_s is not None
        else None
    )
    usage = _usage(row)

    return {
        "instance_id": row.get("instance_id"),
        "reward": row.get("reward"),
        "agent_timeout_error": row.get("agent_timeout_error", 0),
        "trajectory_path": trajectory.get("_trajectory_path"),
        "agent_execution_s": agent_execution_s,
        "inference_wait_s": inference_wait_s,
        "tool_execution_s": tool_execution_s,
        "unknown_agent_s": unknown_agent_s,
        "assistant_turns": len(assistant_indices),
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "parallel_tool_turns": parallel_tool_turns,
        **usage,
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    sorted_values = sorted(values)

    def percentile(percent: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (len(sorted_values) - 1) * percent
        lower = int(rank)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = rank - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    return {
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": mean(values),
        "max": sorted_values[-1],
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def summarize_timing(
    jsonl: Path,
    *,
    jobs_root: Path,
    top_limit: int,
) -> dict[str, Any]:
    """Summarize Harbor trajectory timing from a rollout JSONL file."""
    rows = list(_iter_rows(jsonl))
    job_index: dict[str, Path] = {}
    analyses: list[dict[str, Any]] = []
    missing_trajectory = []
    parse_errors = 0
    reward_counts: Counter[str] = Counter()

    for maybe_row in rows:
        if maybe_row is None:
            parse_errors += 1
            continue
        row = maybe_row
        reward_counts[str(row.get("reward", "missing"))] += 1
        trial_dir = _metadata_trial_path(row)
        if trial_dir is None:
            if not job_index:
                job_index = _build_job_index(jobs_root)
            trial_dir = _trial_dir(row, job_index)
        if trial_dir is None:
            missing_trajectory.append(
                {"instance_id": row.get("instance_id"), "reason": "trial_dir_not_found"}
            )
            continue
        trajectory = _load_trajectory(trial_dir)
        if trajectory is None:
            missing_trajectory.append(
                {
                    "instance_id": row.get("instance_id"),
                    "reason": "trajectory_not_found",
                    "trial_dir": str(trial_dir),
                }
            )
            continue
        analyses.append(_analyze_timing(row, trajectory))

    duration_keys = (
        "agent_execution_s",
        "inference_wait_s",
        "tool_execution_s",
        "unknown_agent_s",
    )
    count_keys = (
        "assistant_turns",
        "tool_calls",
        "tool_outputs",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    aggregate: dict[str, Any] = {
        key: _stats(
            [
                float(analysis[key])
                for analysis in analyses
                if analysis.get(key) is not None
            ]
        )
        for key in duration_keys + count_keys
    }

    total_agent_s = sum(
        float(analysis["agent_execution_s"])
        for analysis in analyses
        if analysis.get("agent_execution_s") is not None
    )
    total_inference_s = sum(float(analysis["inference_wait_s"]) for analysis in analyses)
    total_tool_s = sum(float(analysis["tool_execution_s"]) for analysis in analyses)
    total_unknown_s = sum(
        float(analysis["unknown_agent_s"])
        for analysis in analyses
        if analysis.get("unknown_agent_s") is not None
    )

    top_slowest = sorted(
        analyses,
        key=lambda item: float(item.get("agent_execution_s") or 0.0),
        reverse=True,
    )[:top_limit]

    return {
        "rollouts_jsonl": str(jsonl),
        "jobs_root": str(jobs_root),
        "rows_seen": len(rows),
        "rows_analyzed": len(analyses),
        "jsonl_parse_errors": parse_errors,
        "missing_trajectory_count": len(missing_trajectory),
        "reward_distribution": dict(sorted(reward_counts.items())),
        "aggregate": aggregate,
        "wall_time_totals": {
            "agent_execution_s": total_agent_s,
            "inference_wait_s": total_inference_s,
            "tool_execution_s": total_tool_s,
            "unknown_agent_s": total_unknown_s,
            "inference_wait_fraction": _ratio(total_inference_s, total_agent_s),
            "tool_execution_fraction": _ratio(total_tool_s, total_agent_s),
            "unknown_agent_fraction": _ratio(total_unknown_s, total_agent_s),
        },
        "top_slowest": top_slowest,
        "missing_trajectory_samples": missing_trajectory[:top_limit],
    }


def main() -> None:
    args = parse_args()

    report = summarize_timing(
        args.jsonl,
        jobs_root=args.jobs_root,
        top_limit=args.top_limit,
    )
    report_json = json.dumps(report, indent=2, sort_keys=True)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
