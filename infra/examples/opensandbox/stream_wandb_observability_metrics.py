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

"""Stream sandbox eval observability metrics to W&B while a job is running."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from summarize_observability_metrics import summarize


GLOBAL_STOP_REQUESTED = False


def _request_stop(signum: int, frame: object | None) -> None:
    del signum, frame
    global GLOBAL_STOP_REQUESTED
    GLOBAL_STOP_REQUESTED = True


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _default_wandb_run_id(run_id: str) -> str:
    return hashlib.sha1(run_id.encode()).hexdigest()[:16]


def _get_nested(data: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _add_stat(
    output: dict[str, float | int],
    metrics: dict[str, Any],
    name: str,
    path: Sequence[str],
    field: str,
) -> None:
    value = _get_nested(metrics, (*path, field))
    if isinstance(value, (int, float)):
        output[name] = value


def _flatten_metrics(
    metrics: dict[str, Any],
    *,
    target_rows: int | None,
    metric_prefix: str,
) -> dict[str, float | int]:
    logged: dict[str, float | int] = {}

    scalar_paths = {
        "wall_time_s": ("wall_time_s",),
        "events_count": ("events_count",),
        "resource_samples_count": ("resource_samples_count",),
        "sandbox_count": ("sandbox_count",),
        "peak_sandbox_concurrency": ("peak_sandbox_concurrency",),
        "rollouts_rows": ("rollouts", "rows"),
        "rollouts_score": ("rollouts", "score"),
        "rollouts_reward_sum": ("rollouts", "reward_sum"),
        "resource_memory_peak_bytes": ("resource_peaks", "memory_usage_bytes"),
        "resource_cpu_peak": ("resource_peaks", "cpu_utilization"),
        "resource_process_peak": ("resource_peaks", "process_count"),
    }
    for metric_name, path in scalar_paths.items():
        value = _get_nested(metrics, path)
        if isinstance(value, (int, float)):
            logged[f"{metric_prefix}/{metric_name}"] = value

    rows = _get_nested(metrics, ("rollouts", "rows"))
    if target_rows and isinstance(rows, int):
        logged[f"{metric_prefix}/rollouts_progress"] = rows / target_rows
        logged[f"{metric_prefix}/rollouts_remaining"] = max(0, target_rows - rows)

    duration_paths = {
        "startup_readiness": ("startup_breakdown_s", "sandbox_readiness"),
        "startup_opensandbox_create_api": (
            "startup_breakdown_s",
            "opensandbox_create_api",
        ),
        "startup_first_exec_probe": ("startup_breakdown_s", "first_exec_probe"),
        "startup_environment_setup": (
            "startup_breakdown_s",
            "environment_setup",
        ),
        "startup_borrow_setup": ("startup_breakdown_s", "borrow_setup"),
        "startup_prewarm_setup": ("startup_breakdown_s", "prewarm_setup"),
        "startup_upload_environment": (
            "startup_breakdown_s",
            "upload_environment",
        ),
        "sandbox_exec": ("durations", "sandbox_exec_s"),
        "phase_setup": ("durations", "phase_setup_s"),
        "phase_execution": ("durations", "phase_execution_s"),
        "llm_request": ("durations", "llm_request_s"),
        "trajectory_duration": ("durations", "trajectory_duration_s"),
    }
    for metric_name, path in duration_paths.items():
        for stat in ("count", "p50", "p95", "p99", "mean", "max"):
            _add_stat(
                logged,
                metrics,
                f"{metric_prefix}/{metric_name}_{stat}_s"
                if stat != "count"
                else f"{metric_prefix}/{metric_name}_count",
                path,
                stat,
            )

    errors = metrics.get("errors")
    if isinstance(errors, dict):
        for error_type, count in sorted(errors.items()):
            if isinstance(count, int):
                safe_error = str(error_type).replace("/", "_")
                logged[f"{metric_prefix}/errors/{safe_error}"] = count

    stop_reasons = metrics.get("stop_reasons")
    if isinstance(stop_reasons, dict):
        for reason, count in sorted(stop_reasons.items()):
            if isinstance(count, int):
                safe_reason = str(reason).replace("/", "_")
                logged[f"{metric_prefix}/stop_reasons/{safe_reason}"] = count

    return logged


def _first_server_config(global_config_dict: Any, server_name: str) -> Any:
    server_config = global_config_dict[server_name]
    for value in server_config.values():
        for inner_value in value.values():
            return inner_value
    raise ValueError(f"Could not resolve server config for {server_name!r}")


def _fetch_sandbox_pool_snapshot(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.pool_agent_name:
        return None

    try:
        import requests
        from omegaconf import OmegaConf

        head_server_url = f"http://{args.pool_head_host}:{args.pool_head_port}"
        response = requests.get(
            f"{head_server_url}/global_config_dict_yaml",
            timeout=args.pool_request_timeout_s,
        )
        response.raise_for_status()
        global_config_dict = OmegaConf.create(json.loads(response.content.decode()))
        server_config = _first_server_config(
            global_config_dict,
            args.pool_agent_name,
        )
        snapshot_url = f"http://{server_config.host}:{server_config.port}/sandbox_pool_snapshot"
        snapshot_response = requests.get(
            snapshot_url,
            timeout=args.pool_request_timeout_s,
        )
        snapshot_response.raise_for_status()
        snapshot = snapshot_response.json()
        if isinstance(snapshot, dict):
            return snapshot
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return None


def _flatten_pool_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    metric_prefix: str,
) -> dict[str, float | int]:
    if not snapshot:
        return {}

    logged: dict[str, float | int] = {}
    for key in (
        "idle",
        "borrowed",
        "total",
        "prewarm_inflight",
        "failed",
        "pool_exhausted_total",
        "direct_create_total",
        "acquire_total",
        "acquire_hit_total",
        "stale_handle_total",
        "release_failure_total",
    ):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            logged[f"{metric_prefix}/pool_{key}"] = value

    state = snapshot.get("state")
    state_index = {
        "empty": 0,
        "healthy": 1,
        "running": 2,
        "degraded": 3,
        "warming": 4,
        "draining": 5,
        "stopped": 6,
    }.get(str(state), -1)
    logged[f"{metric_prefix}/pool_state"] = state_index

    if "error" in snapshot:
        logged[f"{metric_prefix}/pool_snapshot_error"] = 1
    return logged


def _write_latest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observability_dir", type=Path)
    parser.add_argument("--rollouts", type=Path, default=None)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--metric-prefix", default="sandbox/live")
    parser.add_argument("--latest-output", type=Path, default=None)
    parser.add_argument("--pool-agent-name", default=None)
    parser.add_argument("--pool-head-host", default="127.0.0.1")
    parser.add_argument("--pool-head-port", type=int, default=11000)
    parser.add_argument("--pool-request-timeout-s", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.interval_s <= 0:
        raise ValueError("--interval-s must be positive")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    run_id = _env("NEMO_RL_SANDBOX_OBSERVABILITY_RUN_ID")
    wandb_run_id = os.environ.get("WANDB_RUN_ID") or _default_wandb_run_id(run_id)
    os.environ.setdefault("WANDB_RUN_ID", wandb_run_id)
    os.environ.setdefault("WANDB_RESUME", "allow")
    os.environ.setdefault("WANDB_MODE", "online")

    import wandb

    wandb_dir = Path(_env("WANDB_DIR"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=_env("NEMO_RL_SANDBOX_OBSERVABILITY_WANDB_PROJECT"),
        name=_env("NEMO_RL_SANDBOX_OBSERVABILITY_WANDB_RUN_NAME"),
        id=wandb_run_id,
        resume=os.environ["WANDB_RESUME"],
        job_type="sandbox-observability-live",
        dir=str(wandb_dir),
        config={
            "benchmark": args.benchmark,
            "harness": args.harness,
            "run_id": run_id,
            "target_rows": args.target_rows,
            "metric_prefix": args.metric_prefix,
        },
    )
    print(
        json.dumps(
            {
                "event": "wandb.live_metrics.start",
                "wandb_url": run.get_url(),
                "run_id": run_id,
                "wandb_run_id": wandb_run_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    step = 0
    try:
        while True:
            metrics = summarize(args.observability_dir, rollouts=args.rollouts)
            flattened = _flatten_metrics(
                metrics,
                target_rows=args.target_rows,
                metric_prefix=args.metric_prefix,
            )
            pool_snapshot = _fetch_sandbox_pool_snapshot(args)
            flattened.update(
                _flatten_pool_snapshot(
                    pool_snapshot,
                    metric_prefix=args.metric_prefix,
                )
            )
            flattened[f"{args.metric_prefix}/heartbeat"] = step + 1
            wandb.log(flattened)
            payload = {
                "step": step,
                "logged_metric_count": len(flattened),
                "pool_snapshot": pool_snapshot,
                "rows": _get_nested(metrics, ("rollouts", "rows")),
                "score": _get_nested(metrics, ("rollouts", "score")),
                "startup_readiness_p95_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "sandbox_readiness", "p95"),
                ),
                "startup_readiness_p99_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "sandbox_readiness", "p99"),
                ),
                "environment_setup_p95_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "environment_setup", "p95"),
                ),
                "environment_setup_p99_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "environment_setup", "p99"),
                ),
                "borrow_setup_p95_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "borrow_setup", "p95"),
                ),
                "borrow_setup_p99_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "borrow_setup", "p99"),
                ),
                "prewarm_setup_p95_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "prewarm_setup", "p95"),
                ),
                "prewarm_setup_p99_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "prewarm_setup", "p99"),
                ),
                "upload_environment_p95_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "upload_environment", "p95"),
                ),
                "upload_environment_p99_s": _get_nested(
                    metrics,
                    ("startup_breakdown_s", "upload_environment", "p99"),
                ),
                "wall_time_s": metrics.get("wall_time_s"),
            }
            if args.latest_output is not None:
                _write_latest(args.latest_output, payload)
            print(
                json.dumps(
                    {"event": "wandb.live_metrics.log", **payload},
                    sort_keys=True,
                ),
                flush=True,
            )
            step += 1
            if GLOBAL_STOP_REQUESTED:
                break
            time.sleep(args.interval_s)
    except Exception as exc:
        print(
            json.dumps(
                {"event": "wandb.live_metrics.error", "error": repr(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        wandb.log({f"{args.metric_prefix}/finished": 1})
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
