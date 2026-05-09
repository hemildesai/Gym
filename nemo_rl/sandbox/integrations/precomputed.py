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

"""Precomputed sandbox trajectories for unit tests and offline replay."""

from typing import Any

from nemo_rl.sandbox.config import SandboxConfig
from nemo_rl.sandbox.integrations.policy_proxy import SandboxTrajectory


def collect_precomputed_trajectories(
    rows: list[dict[str, Any]],
    sandbox_config: SandboxConfig,
) -> list[SandboxTrajectory]:
    """Collect trajectories embedded in ``extra_env_info`` rows."""
    del sandbox_config
    trajectories: list[SandboxTrajectory] = []
    for row_idx, row in enumerate(rows):
        if "sandbox_trajectory" in row:
            trajectories.append(row["sandbox_trajectory"])
        elif "rollout_details" in row and "reward" in row:
            trajectories.append(
                {
                    "rollout_details": row["rollout_details"],
                    "reward": row["reward"],
                    "full_result": row.get("full_result", {}),
                    "agent_name": row.get("agent_name", "precomputed"),
                    "truncated": row.get("truncated", False),
                }
            )
        else:
            raise ValueError(
                "Precomputed sandbox integration requires each row to contain "
                f"`sandbox_trajectory` or `rollout_details`+`reward`; row_idx={row_idx}"
            )
    return trajectories
