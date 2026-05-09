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

"""Prepare a Gym input JSONL for local Harbor task directories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-alias", default="tbench2")
    parser.add_argument("--limit", default=1, type=int)
    parser.add_argument(
        "--offset",
        default=0,
        type=int,
        help="Start from this offset after sorting task names.",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Take every Nth task after applying --offset.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="Shuffle task names with this seed before applying offset/stride/limit.",
    )
    parser.add_argument(
        "--exclude-rollouts",
        action="append",
        default=[],
        type=Path,
        help="Existing Gym rollout JSONL files whose instance_id values should be skipped.",
    )
    return parser.parse_args()


def _excluded_instance_ids(paths: list[Path]) -> set[str]:
    instance_ids = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if isinstance(instance_id, str):
                    instance_ids.add(instance_id)
    return instance_ids


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    excluded_instance_ids = _excluded_instance_ids(args.exclude_rollouts)
    task_names = sorted(
        task_dir.name
        for task_dir in args.tasks_dir.iterdir()
        if task_dir.is_dir() and (task_dir / "task.toml").exists()
    )
    if not task_names:
        raise RuntimeError(f"No Harbor task.toml files found under {args.tasks_dir}")
    if args.sample_seed is not None:
        rng = random.Random(args.sample_seed)
        rng.shuffle(task_names)
    selected_task_names = task_names[args.offset :: args.stride]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as f:
        for task_name in selected_task_names[: args.limit]:
            instance_id = f"{args.dataset_alias}::{task_name}"
            if instance_id in excluded_instance_ids:
                continue
            row = {
                "instance_id": instance_id,
                "responses_create_params": {"input": []},
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    print(
        f"Wrote {written} Gym rows for dataset {args.dataset_alias!r} to {args.output} "
        f"(offset={args.offset}, stride={args.stride}, sample_seed={args.sample_seed})"
    )


if __name__ == "__main__":
    main()
