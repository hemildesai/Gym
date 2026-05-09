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

"""Create a W&B observability run at job startup and emit its URL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _default_wandb_run_id(run_id: str) -> str:
    return hashlib.sha1(run_id.encode()).hexdigest()[:16]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--harness", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = _env("NEMO_RL_SANDBOX_OBSERVABILITY_RUN_ID")
    wandb_run_id = os.environ.get("WANDB_RUN_ID") or _default_wandb_run_id(run_id)
    os.environ.setdefault("WANDB_RUN_ID", wandb_run_id)
    os.environ.setdefault("WANDB_RESUME", "allow")

    meta: dict[str, object] = {
        "run_id": run_id,
        "wandb_project": _env("NEMO_RL_SANDBOX_OBSERVABILITY_WANDB_PROJECT"),
        "wandb_run_name": _env("NEMO_RL_SANDBOX_OBSERVABILITY_WANDB_RUN_NAME"),
        "wandb_run_id": wandb_run_id,
        "benchmark": args.benchmark,
        "harness": args.harness,
    }

    try:
        import wandb

        wandb_dir = Path(_env("WANDB_DIR"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=str(meta["wandb_project"]),
            name=str(meta["wandb_run_name"]),
            id=wandb_run_id,
            resume=os.environ["WANDB_RESUME"],
            job_type="sandbox-observability",
            dir=str(wandb_dir),
            config={"benchmark": args.benchmark, "harness": args.harness},
        )
        meta.update(
            {
                "wandb_entity": run.entity,
                "wandb_url": run.get_url(),
            }
        )
        wandb.log({"lifecycle/startup": 1})
        wandb.finish()
    except Exception as exc:
        meta["wandb_error"] = repr(exc)

    observability_dir = Path(_env("NEMO_RL_SANDBOX_OBSERVABILITY_DIR"))
    observability_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    (observability_dir / "wandb_start.json").write_text(payload)
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
