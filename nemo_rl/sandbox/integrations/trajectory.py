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

"""Dispatch sandbox rollout collection to the configured integration."""

from dataclasses import dataclass
from typing import Any

from nemo_rl.sandbox.config import SandboxConfig
from nemo_rl.sandbox.integrations.policy_proxy import SandboxTrajectory
from nemo_rl.sandbox.integrations.precomputed import collect_precomputed_trajectories


@dataclass(frozen=True)
class SandboxRolloutContext:
    """Policy endpoint context supplied by NeMo-RL to sandbox integrations."""

    model_name: str
    base_urls: list[str | None]


def collect_sandbox_trajectories(
    rows: list[dict[str, Any]],
    sandbox_config: SandboxConfig,
    context: SandboxRolloutContext,
) -> list[SandboxTrajectory]:
    """Collect sandbox trajectories with the configured integration."""
    integration = sandbox_config["integration"]
    integration_name = integration["name"]
    if integration_name == "precomputed":
        return collect_precomputed_trajectories(rows, sandbox_config)
    if integration_name == "harbor":
        from nemo_rl.sandbox.integrations.harbor import collect_harbor_trajectories

        return collect_harbor_trajectories(rows, sandbox_config, context)

    raise ValueError(f"Unknown sandbox integration {integration_name!r}")
