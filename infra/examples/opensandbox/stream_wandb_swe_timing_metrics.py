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

"""Stream SWE trajectory timing and vLLM serving metrics to W&B."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from typing import Any

from summarize_harbor_trajectory_timing import summarize_timing


STOP_REQUESTED = False
VLLM_THROUGHPUT_RE = re.compile(
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, "
    r"Waiting: (?P<waiting>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts_jsonl", type=Path)
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=Path(os.environ.get("HARBOR_JOBS_DIR", "/tmp/harbor_jobs")),
    )
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--latest-output", type=Path, default=None)
    parser.add_argument("--top-limit", type=int, default=3)
    parser.add_argument("--metric-prefix", default="swe_timing")
    parser.add_argument("--wandb-project", default="nemo-gym-sandbox-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--vllm-namespace", default="default")
    parser.add_argument(
        "--vllm-label-selector",
        default="app.kubernetes.io/name=vllm",
    )
    parser.add_argument("--vllm-log-since-seconds", type=int, default=180)
    parser.add_argument("--vllm-log-tail-lines", type=int, default=2000)
    return parser.parse_args()


def _request_stop(signum: int, frame: object | None) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _default_wandb_run_id(run_name: str) -> str:
    return hashlib.sha1(run_name.encode()).hexdigest()[:16]


def _safe_metric_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (float, int)) else None


def _add_stats(
    output: dict[str, float | int],
    *,
    prefix: str,
    name: str,
    stats: dict[str, Any] | None,
) -> None:
    if not isinstance(stats, dict):
        return
    for stat in ("count", "mean", "p50", "p95", "p99", "max"):
        value = _number(stats.get(stat))
        if value is None:
            continue
        suffix = "count" if stat == "count" else f"{stat}_s"
        if name.endswith("_tokens") or name in {"assistant_turns", "tool_calls"}:
            suffix = stat
        output[f"{prefix}/{name}_{suffix}"] = value


def _flatten_timing_report(
    report: dict[str, Any],
    *,
    metric_prefix: str,
    target_rows: int | None,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key in ("rows_seen", "rows_analyzed", "jsonl_parse_errors", "missing_trajectory_count"):
        value = _number(report.get(key))
        if value is not None:
            output[f"{metric_prefix}/{key}"] = value

    rows_seen = report.get("rows_seen")
    if target_rows and isinstance(rows_seen, int):
        output[f"{metric_prefix}/rollouts_progress"] = rows_seen / target_rows
        output[f"{metric_prefix}/rollouts_remaining"] = max(0, target_rows - rows_seen)

    reward_distribution = report.get("reward_distribution")
    if isinstance(reward_distribution, dict):
        for reward, count in sorted(reward_distribution.items()):
            if isinstance(count, int):
                output[f"{metric_prefix}/reward/{_safe_metric_name(str(reward))}"] = count

    wall_time_totals = report.get("wall_time_totals")
    if isinstance(wall_time_totals, dict):
        for key, value in sorted(wall_time_totals.items()):
            number = _number(value)
            if number is not None:
                output[f"{metric_prefix}/wall_time/{key}"] = number

    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict):
        for key in (
            "agent_execution_s",
            "inference_wait_s",
            "tool_execution_s",
            "unknown_agent_s",
            "assistant_turns",
            "tool_calls",
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "total_tokens",
        ):
            _add_stats(
                output,
                prefix=metric_prefix,
                name=key.removesuffix("_s"),
                stats=aggregate.get(key),
            )
    return output


def _load_kubernetes_client() -> Any | None:
    try:
        from kubernetes import client, config
        from kubernetes.config.config_exception import ConfigException
    except ImportError:
        return None

    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException:
            return None
    return client.CoreV1Api()


def _latest_vllm_line(log_text: str) -> dict[str, float | int] | None:
    latest: dict[str, float | int] | None = None
    for line in log_text.splitlines():
        match = VLLM_THROUGHPUT_RE.search(line)
        if match is None:
            continue
        latest = {
            "prompt_tps": float(match.group("prompt")),
            "generation_tps": float(match.group("generation")),
            "running_reqs": int(match.group("running")),
            "waiting_reqs": int(match.group("waiting")),
            "kv_cache_usage_pct": float(match.group("kv")),
            "prefix_cache_hit_rate_pct": float(match.group("prefix")),
        }
    return latest


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _vllm_metrics(args: argparse.Namespace) -> dict[str, float | int]:
    api = _load_kubernetes_client()
    if api is None:
        return {}

    from kubernetes.client.exceptions import ApiException

    try:
        pod_list = api.list_namespaced_pod(
            args.vllm_namespace,
            label_selector=args.vllm_label_selector,
        )
    except ApiException:
        return {}

    samples: list[dict[str, float | int]] = []
    for pod in pod_list.items:
        pod_name = pod.metadata.name
        if not pod_name:
            continue
        try:
            log_text = api.read_namespaced_pod_log(
                pod_name,
                args.vllm_namespace,
                since_seconds=args.vllm_log_since_seconds,
                tail_lines=args.vllm_log_tail_lines,
            )
        except ApiException:
            continue
        sample = _latest_vllm_line(log_text)
        if sample is not None:
            samples.append(sample)

    if not samples:
        return {"vllm/pods_sampled": 0}

    prompt_tps = [float(sample["prompt_tps"]) for sample in samples]
    generation_tps = [float(sample["generation_tps"]) for sample in samples]
    waiting_reqs = [float(sample["waiting_reqs"]) for sample in samples]
    running_reqs = [float(sample["running_reqs"]) for sample in samples]
    kv_cache = [float(sample["kv_cache_usage_pct"]) for sample in samples]
    prefix_hit = [float(sample["prefix_cache_hit_rate_pct"]) for sample in samples]
    return {
        "vllm/pods_sampled": len(samples),
        "vllm/prompt_tps_sum": sum(prompt_tps),
        "vllm/prompt_tps_mean": _mean(prompt_tps) or 0.0,
        "vllm/generation_tps_sum": sum(generation_tps),
        "vllm/generation_tps_mean": _mean(generation_tps) or 0.0,
        "vllm/running_reqs_sum": sum(running_reqs),
        "vllm/waiting_reqs_sum": sum(waiting_reqs),
        "vllm/waiting_reqs_max": max(waiting_reqs),
        "vllm/kv_cache_usage_pct_mean": _mean(kv_cache) or 0.0,
        "vllm/kv_cache_usage_pct_max": max(kv_cache),
        "vllm/prefix_cache_hit_rate_pct_mean": _mean(prefix_hit) or 0.0,
        "vllm/prefix_cache_hit_rate_pct_min": min(prefix_hit),
    }


def _write_latest(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    import wandb

    run_name = args.wandb_run_name or args.rollouts_jsonl.parent.name
    run_id = args.wandb_run_id or _default_wandb_run_id(run_name)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=run_id,
        name=run_name,
        resume="allow",
        dir=str(args.wandb_dir) if args.wandb_dir else None,
        config={
            "rollouts_jsonl": str(args.rollouts_jsonl),
            "jobs_root": str(args.jobs_root),
            "target_rows": args.target_rows,
            "vllm_label_selector": args.vllm_label_selector,
        },
    )
    while not STOP_REQUESTED:
        try:
            report = summarize_timing(
                args.rollouts_jsonl,
                jobs_root=args.jobs_root,
                top_limit=args.top_limit,
            )
            metrics = _flatten_timing_report(
                report,
                metric_prefix=args.metric_prefix,
                target_rows=args.target_rows,
            )
            metrics.update(_vllm_metrics(args))
            wandb.log(metrics)
            _write_latest(
                args.latest_output,
                {
                    "metrics": metrics,
                    "rows_analyzed": report.get("rows_analyzed"),
                    "rows_seen": report.get("rows_seen"),
                    "top_slowest": report.get("top_slowest"),
                    "wall_time_totals": report.get("wall_time_totals"),
                    "wandb_url": run.get_url(),
                },
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            wandb.log({f"{args.metric_prefix}/stream_error": 1})
            _write_latest(args.latest_output, payload)
        time.sleep(args.interval_s)
    wandb.finish()


if __name__ == "__main__":
    main()
