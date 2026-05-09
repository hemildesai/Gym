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

"""Prewarm or cleanup Gym Harbor-agent OpenSandbox handles."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import requests
from omegaconf import OmegaConf


TASK_INDEX_KEY_NAME = "_ng_task_index"
ROLLOUT_INDEX_KEY_NAME = "_ng_rollout_index"


def _load_prewarm_items(
    input_jsonl: Path,
    *,
    limit: int | None,
    num_repeats: int,
) -> list[dict[str, Any]]:
    if num_repeats < 1:
        raise ValueError("num_repeats must be >= 1")

    row_to_task_idx: dict[str, int] = {}
    task_idx_to_rollout_idx: Counter[int] = Counter()
    items = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for row_idx, row_str in enumerate(f):
            if limit is not None and row_idx >= limit:
                break
            stripped = row_str.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"Row {row_idx} is missing string instance_id")
            task_index = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))
            for _ in range(num_repeats):
                rollout_index = task_idx_to_rollout_idx[task_index]
                task_idx_to_rollout_idx[task_index] += 1
                items.append(
                    {
                        "instance_id": instance_id,
                        "task_index": task_index,
                        "rollout_index": rollout_index,
                        TASK_INDEX_KEY_NAME: task_index,
                        ROLLOUT_INDEX_KEY_NAME: rollout_index,
                    }
                )
    return items


def _prewarm_item_key(item: dict[str, Any]) -> str:
    task_index = item.get(TASK_INDEX_KEY_NAME, item.get("task_index"))
    rollout_index = item.get(ROLLOUT_INDEX_KEY_NAME, item.get("rollout_index"))
    if task_index is None or rollout_index is None:
        return str(item["instance_id"])
    return f"{int(task_index)}:{int(rollout_index)}:{item['instance_id']}"


def _load_global_config_dict(*, head_host: str, head_port: int) -> Any:
    head_server_url = f"http://{head_host}:{head_port}"
    response = requests.get(f"{head_server_url}/global_config_dict_yaml", timeout=30)
    response.raise_for_status()
    return OmegaConf.create(json.loads(response.content.decode()))


def _first_server_config(global_config_dict: Any, server_name: str) -> Any:
    server_config = global_config_dict[server_name]
    for value in server_config.values():
        for inner_value in value.values():
            return inner_value
    raise ValueError(f"Could not resolve server config for {server_name!r}")


async def _post(
    *,
    agent_name: str,
    url_path: str,
    payload: dict[str, Any],
    head_host: str,
    head_port: int,
) -> dict[str, Any]:
    global_config_dict = _load_global_config_dict(
        head_host=head_host,
        head_port=head_port,
    )
    server_config = _first_server_config(global_config_dict, agent_name)
    url = f"http://{server_config.host}:{server_config.port}{url_path}"
    response = await asyncio.to_thread(requests.post, url, json=payload, timeout=None)
    response.raise_for_status()
    return dict(response.json())


async def _prewarm(args: argparse.Namespace) -> dict[str, Any]:
    items = _load_prewarm_items(
        args.input_jsonl,
        limit=args.limit,
        num_repeats=args.num_repeats,
    )
    items_by_key = {_prewarm_item_key(item): item for item in items}
    pending_items = items
    attempts = []
    created = 0
    reused = 0
    final_result: dict[str, Any] = {}
    delay_s = args.retry_delay_s

    for attempt in range(args.retries + 1):
        result = await _post(
            agent_name=args.agent_name,
            url_path="/prewarm_sandboxes",
            payload={
                "items": pending_items,
                "concurrency": args.concurrency,
                "create_concurrency": args.create_concurrency,
                "prepare_concurrency": args.prepare_concurrency,
                "replace_existing": args.replace_existing and attempt == 0,
                "prepare_environment": args.prepare_environment,
                "start_policy_proxy": args.start_policy_proxy,
            },
            head_host=args.head_host,
            head_port=args.head_port,
        )
        errors = {
            key: value
            for key, value in dict(result.get("errors", {})).items()
            if key in items_by_key
        }
        attempts.append(
            {
                "attempt": attempt + 1,
                "requested": len(pending_items),
                "created": result.get("created", 0),
                "reused": result.get("reused", 0),
                "failed": len(errors),
            }
        )
        created += int(result.get("created", 0) or 0)
        reused += int(result.get("reused", 0) or 0)
        final_result = result
        if not errors:
            break
        if attempt >= args.retries:
            break
        pending_items = [items_by_key[key] for key in errors]
        await asyncio.to_thread(time.sleep, delay_s)
        delay_s = min(args.retry_max_delay_s, delay_s * args.retry_backoff)

    final_errors = {
        key: value
        for key, value in dict(final_result.get("errors", {})).items()
        if key in {_prewarm_item_key(item) for item in pending_items}
    }
    return {
        **final_result,
        "requested": len(items),
        "created": created,
        "reused": reused,
        "failed": len(final_errors),
        "errors": final_errors,
        "attempts": attempts,
    }


async def _cleanup(args: argparse.Namespace) -> dict[str, Any]:
    return await _post(
        agent_name=args.agent_name,
        url_path="/cleanup_prewarmed_sandboxes",
        payload={"delete": args.delete},
        head_host=args.head_host,
        head_port=args.head_port,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prewarm = subparsers.add_parser("prewarm")
    prewarm.add_argument("--agent-name", required=True)
    prewarm.add_argument("--input-jsonl", type=Path, required=True)
    prewarm.add_argument("--limit", type=int, default=None)
    prewarm.add_argument("--num-repeats", type=int, default=1)
    prewarm.add_argument("--concurrency", type=int, default=None)
    prewarm.add_argument("--create-concurrency", type=int, default=None)
    prewarm.add_argument("--prepare-concurrency", type=int, default=None)
    prewarm.add_argument("--replace-existing", action="store_true")
    prewarm.add_argument("--retries", type=int, default=0)
    prewarm.add_argument("--retry-delay-s", type=float, default=5.0)
    prewarm.add_argument("--retry-backoff", type=float, default=2.0)
    prewarm.add_argument("--retry-max-delay-s", type=float, default=60.0)
    prewarm.add_argument(
        "--prepare-environment",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prewarm.add_argument(
        "--start-policy-proxy",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    prewarm.add_argument("--head-host", default="127.0.0.1")
    prewarm.add_argument("--head-port", type=int, default=11000)

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--agent-name", required=True)
    cleanup.add_argument("--delete", action=argparse.BooleanOptionalAction, default=True)
    cleanup.add_argument("--head-host", default="127.0.0.1")
    cleanup.add_argument("--head-port", type=int, default=11000)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prewarm":
        result = asyncio.run(_prewarm(args))
    elif args.command == "cleanup":
        result = asyncio.run(_cleanup(args))
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
