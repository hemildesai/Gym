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

"""Summarize sandbox eval observability into a compact metrics JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    numeric = sorted(float(value) for value in values)
    if not numeric:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "min": numeric[0],
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
        "mean": mean(numeric),
        "max": numeric[-1],
    }


def _duration_stats(
    events: list[dict[str, Any]],
    *,
    name: str | None = None,
    phase: str | None = None,
    command_class: str | None = None,
    status: str | None = None,
) -> dict[str, float | int | None]:
    values = []
    for event in events:
        if event.get("event_type") != "span_end":
            continue
        if name is not None and event.get("name") != name:
            continue
        attrs = event.get("attributes") or {}
        if phase is not None and attrs.get("phase") != phase:
            continue
        if command_class is not None and attrs.get("command_class") != command_class:
            continue
        event_status = str(attrs.get("status") or "ok")
        if status == "ok" and event_status == "error":
            continue
        if status == "error" and event_status != "error":
            continue
        duration = attrs.get("duration_s")
        if isinstance(duration, (int, float)):
            values.append(float(duration))
    return _stats(values)


def _event_duration_stats(
    events: list[dict[str, Any]],
    *,
    name: str,
) -> dict[str, float | int | None]:
    values = []
    for event in events:
        if event.get("name") != name:
            continue
        attrs = event.get("attributes") or {}
        duration = attrs.get("duration_s")
        if isinstance(duration, (int, float)):
            values.append(float(duration))
    return _stats(values)


def _rollout_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    rows = _load_jsonl(path)
    rewards = []
    statuses = Counter()
    exceptions = Counter()
    for row in rows:
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        statuses[str(row.get("status") or "unknown")] += 1
        exception = row.get("exception")
        exceptions[str(exception) if exception else "none"] += 1
    return {
        "rows": len(rows),
        "reward_sum": sum(rewards),
        "score": (sum(rewards) / len(rows)) if rows else None,
        "reward": _stats(rewards),
        "status_counts": dict(sorted(statuses.items())),
        "exception_counts": dict(sorted(exceptions.items())),
    }


def _sandbox_lifetimes(events: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    bounds: dict[str, list[float]] = {}
    for event in events:
        attrs = event.get("attributes") or {}
        sandbox_id = attrs.get("sandbox_id")
        elapsed = event.get("elapsed_time_s")
        if not isinstance(sandbox_id, str) or not isinstance(elapsed, (int, float)):
            continue
        item = bounds.setdefault(sandbox_id, [float(elapsed), float(elapsed)])
        item[0] = min(item[0], float(elapsed))
        item[1] = max(item[1], float(elapsed))

    edges: list[tuple[float, int]] = []
    lifetimes = []
    for start, end in bounds.values():
        lifetimes.append(max(0.0, end - start))
        edges.append((start, 1))
        edges.append((end, -1))

    peak = 0
    active = 0
    for _, delta in sorted(edges, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return peak, {"count": len(bounds), "duration_s": _stats(lifetimes)}


def summarize(observability_dir: Path, *, rollouts: Path | None = None) -> dict[str, Any]:
    events = _load_jsonl(observability_dir / "events.jsonl")
    resources = _load_jsonl(observability_dir / "resource_samples.jsonl")

    spans_by_name = Counter()
    errors = Counter()
    stop_reasons = Counter()
    sandbox_ids = set()
    command_classes = Counter()
    max_elapsed_time_s = 0.0
    for event in events:
        elapsed = event.get("elapsed_time_s")
        if isinstance(elapsed, (int, float)):
            max_elapsed_time_s = max(max_elapsed_time_s, float(elapsed))
        attrs = event.get("attributes") or {}
        sandbox_id = attrs.get("sandbox_id")
        if isinstance(sandbox_id, str):
            sandbox_ids.add(sandbox_id)
        if event.get("event_type") == "span_end":
            spans_by_name[str(event.get("name") or "unknown")] += 1
            if attrs.get("status") == "error":
                errors[str(attrs.get("error_type") or "error")] += 1
        if event.get("name") == "trajectory.complete":
            stop_reasons[str(attrs.get("stop_reason") or "unknown")] += 1
        if event.get("name") == "trajectory.masked":
            stop_reasons[str(attrs.get("stop_reason") or "masked")] += 1
        command_class = attrs.get("command_class")
        if isinstance(command_class, str):
            command_classes[command_class] += 1

    peak_sandbox_concurrency, sandbox_lifetime = _sandbox_lifetimes(events)
    memory_values = [
        int(sample["memory_usage_bytes"])
        for sample in resources
        if isinstance(sample.get("memory_usage_bytes"), int)
    ]
    cpu_values = [
        float(sample["cpu_utilization"])
        for sample in resources
        if isinstance(sample.get("cpu_utilization"), (int, float))
    ]
    process_counts = [
        int(sample["process_count"])
        for sample in resources
        if isinstance(sample.get("process_count"), int)
    ]

    sandbox_exec_by_command_class = {
        command_class: _duration_stats(
            events,
            name="sandbox.exec",
            command_class=command_class,
        )
        for command_class in sorted(command_classes)
    }

    durations = {
        "sandbox_create_api_s": _duration_stats(events, name="sandbox.create_api"),
        "sandbox_create_probe_s": _duration_stats(events, name="sandbox.create_probe"),
        "sandbox_start_s": _duration_stats(events, name="sandbox.start"),
        "sandbox_setup_s": _duration_stats(events, name="sandbox.setup"),
        "sandbox_borrow_setup_s": _duration_stats(events, name="sandbox.borrow.setup"),
        "sandbox_prewarm_setup_s": _duration_stats(events, name="sandbox.prewarm.setup"),
        "sandbox_upload_environment_s": _duration_stats(
            events,
            name="sandbox.setup.upload_environment",
        ),
        "sandbox_exec_s": _duration_stats(events, name="sandbox.exec"),
        "phase_setup_s": _duration_stats(events, phase="setup"),
        "phase_execution_s": _duration_stats(events, phase="execution"),
        "harbor_trial_run_s": _duration_stats(events, name="harbor.trial.run"),
        "llm_request_s": _duration_stats(events, name="llm.request"),
        "trajectory_duration_s": _event_duration_stats(
            events,
            name="trajectory.complete",
        ),
        "sandbox_exec_by_command_class_s": sandbox_exec_by_command_class,
    }

    return {
        "schema_version": 1,
        "observability_dir": str(observability_dir),
        "events_count": len(events),
        "resource_samples_count": len(resources),
        "wall_time_s": max_elapsed_time_s,
        "sandbox_count": len(sandbox_ids),
        "peak_sandbox_concurrency": peak_sandbox_concurrency,
        "sandbox_lifetime": sandbox_lifetime,
        "durations": durations,
        "startup_breakdown_s": {
            "sandbox_readiness": durations["sandbox_start_s"],
            "opensandbox_create_api": durations["sandbox_create_api_s"],
            "first_exec_probe": durations["sandbox_create_probe_s"],
            "environment_setup": durations["sandbox_setup_s"],
            "borrow_setup": durations["sandbox_borrow_setup_s"],
            "prewarm_setup": durations["sandbox_prewarm_setup_s"],
            "upload_environment": durations["sandbox_upload_environment_s"],
            "description": {
                "sandbox_readiness": "OpenSandbox allocation through first successful sandbox start probe.",
                "environment_setup": "Harbor/Gym environment bootstrap after the sandbox is reachable.",
                "borrow_setup": "Rollout-path Harbor environment reset/upload/setup after borrowing a prewarmed handle.",
                "prewarm_setup": "Pre-rollout Harbor environment reset/upload/setup performed against idle pool handles.",
                "upload_environment": "Upload of task environment files into the sandbox.",
                "opensandbox_create_api": "OpenSandbox SDK/API create wait before NeMo-RL's explicit probe.",
                "first_exec_probe": "NeMo-RL verification command proving execd is reachable.",
            },
        },
        "spans_by_name": dict(sorted(spans_by_name.items())),
        "errors": dict(sorted(errors.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "resource_peaks": {
            "memory_usage_bytes": max(memory_values) if memory_values else None,
            "cpu_utilization": max(cpu_values) if cpu_values else None,
            "process_count": max(process_counts) if process_counts else None,
        },
        "rollouts": _rollout_metrics(rollouts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observability_dir", type=Path)
    parser.add_argument("--rollouts", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = summarize(args.observability_dir, rollouts=args.rollouts)
    text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
