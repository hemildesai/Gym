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

"""Lightweight verifier for Gym rollout JSONL smoke outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--min-rows", default=1, type=int)
    parser.add_argument("--require-trainable", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _assistant_outputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    response = row.get("response") or {}
    return [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]


def _has_trainable_fields(item: dict[str, Any]) -> bool:
    return bool(
        item.get("prompt_token_ids")
        and item.get("generation_token_ids")
        and item.get("generation_log_probs")
    )


def main() -> None:
    args = parse_args()
    rows = 0
    trainable_turns = 0
    assistant_turns = 0
    rows_without_assistant_outputs = []
    untrainable_turns = []
    rewards = []

    with args.jsonl.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1

            if "reward" not in row:
                raise RuntimeError(f"Row {index} missing reward")
            rewards.append(row["reward"])

            assistant_outputs = _assistant_outputs(row)
            if not assistant_outputs:
                rows_without_assistant_outputs.append(index)
                continue

            assistant_turns += len(assistant_outputs)
            for turn_index, item in enumerate(assistant_outputs):
                if _has_trainable_fields(item):
                    trainable_turns += 1
                elif args.require_trainable:
                    untrainable_turns.append(
                        {"row": index, "assistant_turn": turn_index}
                    )

    if rows < args.min_rows:
        raise RuntimeError(f"Expected at least {args.min_rows} rows, found {rows}")

    if args.require_trainable and untrainable_turns:
        sample = untrainable_turns[:5]
        raise RuntimeError(f"Assistant turns missing trainable fields: {sample}")

    if args.require_trainable and trainable_turns == 0:
        raise RuntimeError("No assistant output contains token IDs and logprobs")

    report = {
        "rows": rows,
        "rewards": rewards,
        "assistant_turns": assistant_turns,
        "trainable_turns": trainable_turns,
        "rows_without_assistant_outputs": rows_without_assistant_outputs,
        "untrainable_turns": untrainable_turns,
    }
    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
