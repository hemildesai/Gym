# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import re
import sys
import threading
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

import ray
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class HarborDatasetSourceConfig(BaseModel):
    local_dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    workdir: Optional[str] = None


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    concurrency: int
    runner_backend: Literal["async", "ray", "thread"] = "ray"
    runner_num_cpus: float = 0.0
    runner_thread_workers: Optional[int] = None
    response_output_mode: Literal["trajectory", "policy_trace"] = "trajectory"

    # --- Harbor agent settings ---
    # Name of a built-in Harbor agent (e.g. "terminus-2", "claude-code", "aider").
    harbor_agent_name: Optional[str] = "terminus-2"
    # Python import path for a custom agent class (e.g. "my_pkg.my_mod:MyAgent").
    # Overrides harbor_agent_name when set.
    harbor_agent_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor AgentConfig (e.g. collect_rollout_details,
    # model_info). See harbor_agent.yaml for examples.
    harbor_agent_kwargs: Optional[dict[str, Any]] = None
    # Optional env forwarded to the Harbor AgentConfig.
    harbor_agent_env: Optional[dict[str, str]] = None
    # Optional model-name template for installed agents that expect provider/model.
    # Supports "{model_name}" and "{target_base_url}".
    harbor_agent_model_name: Optional[str] = None
    # Optional policy proxy config for installed agents. When set, the proxy is
    # started inside the sandbox and its policy_trace.jsonl is used for
    # trainable token IDs/logprobs.
    harbor_policy_proxy: Optional[dict[str, Any]] = None

    # --- Dataset routing ---
    # Map of dataset aliases to source definitions. Each alias must define exactly
    # one source:
    # 1) local: {"local_dataset_path": "..."}
    # 2) registry: {"dataset_name": "...", "dataset_version": "..."} (version optional)
    # Requests must provide instance_id in the form "<dataset_alias>::<task_name>".
    harbor_datasets: dict[str, HarborDatasetSourceConfig]

    # --- Environment ---
    # Harbor environment type: "singularity", "docker", "daytona", "modal", etc.
    harbor_environment_type: Optional[str] = "singularity"
    # Python import path for a custom environment class (e.g. "my_pkg.my_mod:MyEnv").
    # Overrides harbor_environment_type when set.
    harbor_environment_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor EnvironmentConfig (e.g.
    # singularity_image_cache_dir, singularity_force_pull).
    harbor_environment_kwargs: Optional[dict[str, Any]] = None

    # --- Timeouts ---
    # Override agent timeout (seconds). Replaces the task's own timeout entirely.
    # Use this to set a fixed timeout for all tasks regardless of task.toml.
    harbor_agent_override_timeout: Optional[int] = None
    # Override agent setup timeout (seconds). Replaces Harbor's setup default.
    harbor_agent_override_setup_timeout: Optional[int] = None
    # Cap agent timeout (seconds). Uses the task's own timeout but clamps it
    # to this maximum. Respects shorter per-task timeouts unlike harbor_agent_override_timeout.
    harbor_agent_max_timeout: Optional[int] = None
    # Override verifier timeout (seconds). Replaces the task's own verifier timeout.
    harbor_verifier_override_timeout: Optional[int] = None
    # Cap verifier timeout (seconds). Uses the task's own verifier timeout but
    # clamps it to this maximum.
    harbor_verifier_max_timeout: Optional[int] = None
    # Multiplier applied to all Harbor timeouts after override/cap. None = 1.0.
    harbor_timeout_multiplier: Optional[float] = None
    # Outer timeout for the complete /run request. This is a guardrail around
    # Harbor and sandbox cleanup paths; normal agent/verifier timeouts remain
    # controlled by Harbor itself.
    harbor_run_timeout: Optional[int] = None

    # --- Job output ---
    # Directory where Harbor writes job results and trial artifacts.
    harbor_jobs_dir: str = "jobs"

    # Optional in-process sandbox pool controls for prewarmed OpenSandbox handles.
    sandbox_pool: Optional[dict[str, Any]] = None

    # --- Model routing ---
    # NeMo Gym model server reference used to resolve Harbor model base URL.
    model_server: ModelServerRef


class HarborSandboxPrewarmItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    instance_id: str
    task_index: Optional[int] = None
    rollout_index: Optional[int] = None


class HarborSandboxPrewarmRequest(BaseModel):
    items: list[HarborSandboxPrewarmItem]
    concurrency: Optional[int] = None
    create_concurrency: Optional[int] = None
    prepare_concurrency: Optional[int] = None
    replace_existing: bool = False
    prepare_environment: Optional[bool] = None
    start_policy_proxy: Optional[bool] = None


class HarborSandboxCleanupRequest(BaseModel):
    delete: bool = True


class HarborSandboxProgressRequest(BaseModel):
    keys: Optional[list[str]] = None
    limit: int = 20
    timeout_s: int = 20


class HarborSandboxPoolSnapshot(BaseModel):
    configured: bool
    state: str
    idle: int
    borrowed: int
    total: int
    prewarm_inflight: int
    failed: int
    pool_exhausted_total: int
    direct_create_total: int
    acquire_total: int
    acquire_hit_total: int
    stale_handle_total: int
    release_failure_total: int
    errors: dict[str, str]


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


async def run_harbor_job(job_config_dict: dict) -> str:
    """Runs a single Harbor Job and returns the trial directory path.

    The trial directory contains:
    - result.json: Summary result with reward, agent_result, verifier_result, etc.
    - agent/trajectory.json: Full ATIF trajectory with per-step messages, tool
      calls, observations, and per-token logprobs.

    Harbor writes result.json and trajectory.json to disk even when the trial
    fails (e.g. verifier timeout, reward file not found, OOM).  We recover the
    trial directory after an exception so the caller can still use the partial
    trajectory for training.
    """
    _patch_litellm_model_list_compat()

    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    config = JobConfig(**job_config_dict)
    job = await Job.create(config)

    job_error = None
    try:
        await job.run()
    except Exception as e:
        job_error = e

    # Find the trial directory from the job output directory.  Harbor writes
    # result.json before propagating most exceptions, so we can usually
    # recover the trial even when job.run() raised.
    job_dir = config.jobs_dir / config.job_name
    if job_dir.exists():
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            result_path = trial_dir / "result.json"
            if result_path.exists():
                return str(trial_dir)

    # No trial directory found — re-raise the original error if there was one,
    # otherwise raise FileNotFoundError.
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No trial result found in {job_dir}")


def _patch_litellm_model_list_compat() -> None:
    """Provide optional LiteLLM model-list attributes expected by Harbor."""
    import litellm

    for attr in ("zai_models",):
        if not hasattr(litellm, attr):
            setattr(litellm, attr, set())


_RUNNER_EVENT_LOOP_LOCAL = threading.local()


def _run_harbor_job_sync(job_config_dict: dict) -> str:
    """Synchronous wrapper for run_harbor_job for use in Ray remote.

    Ray workers and thread-pool workers are long-lived. Reusing one event loop
    per worker thread avoids cross-loop issues with global async state while
    keeping concurrent thread-pool jobs isolated from each other.
    """
    event_loop = getattr(_RUNNER_EVENT_LOOP_LOCAL, "event_loop", None)
    if event_loop is None or event_loop.is_closed():
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        _RUNNER_EVENT_LOOP_LOCAL.event_loop = event_loop
    return event_loop.run_until_complete(run_harbor_job(job_config_dict))


async def _run_harbor_job_with_backend(
    *,
    backend: str,
    job_config_dict: dict[str, Any],
    runner_num_cpus: float,
    thread_executor: Optional[ThreadPoolExecutor] = None,
) -> str:
    """Run one Harbor job using the configured coordinator-side backend."""
    if backend == "async":
        return await run_harbor_job(job_config_dict)
    if backend == "thread":
        if thread_executor is None:
            raise ValueError("runner_backend='thread' requires a configured thread executor")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            thread_executor,
            _run_harbor_job_sync,
            job_config_dict,
        )
    if backend == "ray":
        future = runner_ray_remote.options(num_cpus=runner_num_cpus).remote(
            _run_harbor_job_sync,
            {"job_config_dict": job_config_dict},
        )
        return await asyncio.to_thread(ray.get, future)
    raise ValueError(f"Unsupported Harbor runner_backend={backend!r}")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig
    sem: Semaphore = None
    runner_executor: Optional[ThreadPoolExecutor] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _sandbox_pool_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _sandbox_pool_provider: Any = PrivateAttr(default=None)
    _sandbox_pool_idle_tokens: dict[str, str] = PrivateAttr(default_factory=dict)
    _sandbox_pool_borrowed_tokens: dict[str, str] = PrivateAttr(default_factory=dict)
    _sandbox_pool_handles: dict[str, Any] = PrivateAttr(default_factory=dict)
    _sandbox_pool_progress_probe_tokens: set[str] = PrivateAttr(default_factory=set)
    _sandbox_pool_errors: dict[str, str] = PrivateAttr(default_factory=dict)
    _sandbox_pool_exhausted_total: int = PrivateAttr(default=0)
    _sandbox_pool_direct_create_total: int = PrivateAttr(default=0)
    _sandbox_pool_acquire_total: int = PrivateAttr(default=0)
    _sandbox_pool_acquire_hit_total: int = PrivateAttr(default=0)
    _sandbox_pool_stale_handle_total: int = PrivateAttr(default=0)
    _sandbox_pool_release_failure_total: int = PrivateAttr(default=0)
    _sandbox_pool_prewarm_inflight: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        if self.config.runner_backend == "thread":
            max_workers = self.config.runner_thread_workers or self.config.concurrency
            self.runner_executor = ThreadPoolExecutor(
                max_workers=max(1, int(max_workers)),
                thread_name_prefix="harbor-runner",
            )

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/prewarm_sandboxes")(self.prewarm_sandboxes)
        app.post("/cleanup_prewarmed_sandboxes")(self.cleanup_prewarmed_sandboxes)
        app.get("/sandbox_pool_snapshot")(self.sandbox_pool_snapshot)
        app.post("/sandbox_pool_progress")(self.sandbox_pool_progress)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self.sem:
            global_config_dict = get_global_config_dict()

            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)
            run_timestamp = datetime.now(timezone.utc)
            run_id = self._build_run_id(run_timestamp)

            instance_id = body.instance_id
            dataset_alias, task_name = self._parse_instance_id(instance_id)

            output_file_dir = self._get_results_output_dir(policy_model_name, dataset_alias, run_timestamp)
            jobs_dir = self._get_jobs_output_dir(policy_model_name, dataset_alias, run_timestamp)
            job_name = self._build_job_name(run_id)

            responses_create_params = body.responses_create_params.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )
            preallocated_handle_token, borrowed_handle_key = await self._take_prewarmed_handle(body)

            try:
                job_config_dict = self._build_job_config(
                    dataset_alias,
                    task_name,
                    policy_model_name,
                    base_url,
                    job_name=job_name,
                    jobs_dir=jobs_dir,
                    responses_create_params=responses_create_params,
                    policy_target_base_url=global_config_dict.get("policy_base_url"),
                    preallocated_handle_token=preallocated_handle_token,
                )

                trial_dir_path: Optional[str] = None
                try:
                    run_job = _run_harbor_job_with_backend(
                        backend=self.config.runner_backend,
                        job_config_dict=job_config_dict,
                        runner_num_cpus=self.config.runner_num_cpus,
                        thread_executor=self.runner_executor,
                    )
                    if self.config.harbor_run_timeout is not None:
                        trial_dir_path = await asyncio.wait_for(
                            run_job,
                            timeout=float(self.config.harbor_run_timeout),
                        )
                    else:
                        trial_dir_path = await run_job
                    trial_dir = Path(trial_dir_path)

                    # Read the trial result (summary: reward, agent_result, verifier_result)
                    with open(trial_dir / "result.json", "r") as f:
                        trial_result = json.load(f)

                    # Read the ATIF trajectory (full conversation with per-token logprobs)
                    trajectory = None
                    trajectory_path = trial_dir / "agent" / "trajectory.json"
                    if trajectory_path.exists():
                        with open(trajectory_path, "r") as f:
                            trajectory = json.load(f)

                    # Read agent error flags written by the agent
                    agent_error_flags = {}
                    agent_error_flags_path = trial_dir / "agent" / "agent_error_flags.json"
                    if agent_error_flags_path.exists():
                        with open(agent_error_flags_path, "r") as f:
                            agent_error_flags = json.load(f)

                    # Extract reward from verifier result
                    verifier_result = trial_result.get("verifier_result")
                    reward = HarborAgentUtils.extract_reward(verifier_result)

                    policy_trace_rollout_details = self._load_policy_trace_rollout_details(trial_dir)

                    # Convert Harbor outputs to NeMo Gym response items:
                    # keep rich trajectory details, then overlay rollout token details when present.
                    output_items = HarborAgentUtils.trial_result_to_responses(
                        trial_result,
                        trajectory,
                        policy_trace_rollout_details,
                        output_mode=self.config.response_output_mode,
                    )

                    # Extract the initial instruction from the trajectory as input messages
                    input_messages = HarborAgentUtils.extract_input_from_trajectory(trajectory)

                    # Populate usage from trajectory final_metrics or agent_result
                    usage = HarborAgentUtils.extract_usage(trial_result, trajectory)

                except Exception as e:
                    print(f"Error running Harbor job: {e}")
                    trial_result = None
                    trajectory = None
                    policy_trace_rollout_details = None
                    agent_error_flags = {}
                    output_items = []
                    input_messages = []
                    usage = None
                    reward = 0.0

                metadata = self._compact_trial_result_metadata(trial_result)
                if trial_dir_path is not None:
                    metadata["trial_dir"] = trial_dir_path

                response = HarborAgentUtils.get_default_response_object()
                response["model"] = policy_model_name
                response["temperature"] = responses_create_params.get("temperature")
                response["top_p"] = responses_create_params.get("top_p")
                response["output"] = output_items
                if usage:
                    response["usage"] = usage

                # Update responses_create_params with the actual input sent to the agent
                updated_params = body.responses_create_params
                if input_messages:
                    updated_params = body.responses_create_params.model_copy(update={"input": input_messages})

                verify_response = HarborVerifyResponse(
                    responses_create_params=updated_params,
                    reward=reward,
                    response=response,
                    instance_id=instance_id,
                    metadata=metadata,
                    context_length_exceeded_error=int(agent_error_flags.get("context_length_exceeded", False)),
                    memory_limit_exceeded_error=int(agent_error_flags.get("memory_limit_exceeded", False)),
                    agent_timeout_error=int(
                        ((trial_result or {}).get("exception_info") or {}).get("exception_type") == "AgentTimeoutError"
                    ),
                )

                # Save result to disk (folder = run_id, file = task name)
                output_path = output_file_dir / run_id
                output_path.mkdir(parents=True, exist_ok=True)

                safe_instance_id = self._sanitize_path_component(instance_id)
                with open(output_path / f"{safe_instance_id}.json", "w") as f:
                    json.dump(verify_response.model_dump(), f, indent=2)

                return verify_response
            finally:
                if preallocated_handle_token is not None:
                    try:
                        await self._release_prewarmed_handle(
                            borrowed_handle_key,
                            preallocated_handle_token,
                            delete=True,
                        )
                    except Exception as e:
                        key = borrowed_handle_key or preallocated_handle_token
                        async with self._sandbox_pool_lock:
                            self._sandbox_pool_release_failure_total += 1
                            self._sandbox_pool_errors[key] = (
                                f"{type(e).__name__}: {e}"
                            )
                        print(f"Error releasing prewarmed sandbox {key}: {e}")

    async def prewarm_sandboxes(self, body: HarborSandboxPrewarmRequest = Body()) -> dict[str, Any]:
        if body.concurrency is not None and body.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if body.create_concurrency is not None and body.create_concurrency < 1:
            raise ValueError("create_concurrency must be >= 1")
        if body.prepare_concurrency is not None and body.prepare_concurrency < 1:
            raise ValueError("prepare_concurrency must be >= 1")
        async with self._sandbox_pool_lock:
            if self._sandbox_pool_prewarm_inflight:
                raise RuntimeError("Sandbox prewarm is already running")
            if body.replace_existing:
                await self._cleanup_prewarmed_sandboxes_locked(delete=True)
            provider = self._sandbox_pool_provider
            if provider is None:
                provider = self._create_sandbox_pool_provider()
                self._sandbox_pool_provider = provider

            concurrency = body.concurrency or self.config.concurrency
            pool_config = self._sandbox_pool_config()
            create_concurrency = (
                body.create_concurrency
                or pool_config.get("create_concurrency")
                or concurrency
            )
            prepare_concurrency = (
                body.prepare_concurrency
                or pool_config.get("prepare_concurrency")
                or concurrency
            )
            create_semaphore = asyncio.Semaphore(max(1, int(create_concurrency)))
            prepare_semaphore = asyncio.Semaphore(max(1, int(prepare_concurrency)))
            prepare_environment = self._sandbox_pool_prepare_environment(body)
            start_policy_proxy = self._sandbox_pool_start_policy_proxy(body)
            policy_proxy_config = self._sandbox_pool_policy_proxy_config()
            start_policy_proxy = start_policy_proxy and policy_proxy_config is not None
            if start_policy_proxy:
                prepare_environment = True
            created = 0
            reused = 0
            errors: dict[str, str] = {}

            pending_by_instance_id: dict[str, list[tuple[str, HarborSandboxPrewarmItem]]] = {}
            seen_pending_keys: set[str] = set()
            for item in body.items:
                key = self._sandbox_pool_key(
                    item.instance_id,
                    self._sandbox_pool_task_index(item),
                    self._sandbox_pool_rollout_index(item),
                )
                if key in self._sandbox_pool_idle_tokens or key in self._sandbox_pool_borrowed_tokens:
                    reused += 1
                    continue
                if key in seen_pending_keys:
                    errors[key] = "duplicate sandbox pool key in one prewarm request"
                    continue
                seen_pending_keys.add(key)
                pending_by_instance_id.setdefault(item.instance_id, []).append((key, item))
            self._sandbox_pool_prewarm_inflight = sum(
                len(keyed_items)
                for keyed_items in pending_by_instance_id.values()
            )

        async def _prepare_handle(handle: Any, instance_id: str) -> None:
            async with prepare_semaphore:
                await self._prepare_prewarmed_handle(
                    instance_id,
                    handle,
                    policy_proxy_config=(
                        policy_proxy_config
                        if start_policy_proxy
                        else None
                    ),
                )

        async def _prewarm_group(instance_id: str, keyed_items: list[tuple[str, HarborSandboxPrewarmItem]]) -> None:
            nonlocal created, reused
            handles: list[Any] = []
            try:
                spec = self._build_sandbox_pool_spec(instance_id)
                async with create_semaphore:
                    handles = await provider.create_batch(
                        spec,
                        len(keyed_items),
                        allow_partial=False,
                    )
                if len(handles) != len(keyed_items):
                    raise RuntimeError(
                        f"Expected {len(keyed_items)} prewarmed handles, got {len(handles)}"
                    )
                if prepare_environment:
                    await asyncio.gather(
                        *(_prepare_handle(handle, instance_id) for handle in handles)
                    )
                async with self._sandbox_pool_lock:
                    for (key, _), handle in zip(keyed_items, handles):
                        token = uuid4().hex
                        errors.pop(key, None)
                        self._sandbox_pool_errors.pop(key, None)
                        self._register_prewarmed_handle(
                            key,
                            token,
                            handle,
                            prepared_environment=prepare_environment,
                            policy_proxy_started=start_policy_proxy,
                        )
                created += len(handles)
            except Exception as e:
                if handles:
                    try:
                        from nemo_rl.sandbox.integrations.harbor import (
                            _close_preallocated_handles,
                        )

                        await _close_preallocated_handles(
                            provider,
                            handles,
                            delete_batch=True,
                        )
                    except Exception:
                        pass
                message = f"{type(e).__name__}: {e}"
                for key, _ in keyed_items:
                    errors[key] = message

        try:
            await asyncio.gather(
                *(
                    _prewarm_group(instance_id, keyed_items)
                    for instance_id, keyed_items in pending_by_instance_id.items()
                )
            )
        finally:
            async with self._sandbox_pool_lock:
                self._sandbox_pool_prewarm_inflight = 0

        async with self._sandbox_pool_lock:
            self._sandbox_pool_errors.update(errors)
            return {
                "requested": len(body.items),
                "created": created,
                "reused": reused,
                "failed": len(errors),
                "create_concurrency": max(1, int(create_concurrency)),
                "prepare_concurrency": max(1, int(prepare_concurrency)),
                "errors": errors,
                "snapshot": self._sandbox_pool_snapshot_dict(),
            }

    async def cleanup_prewarmed_sandboxes(self, body: HarborSandboxCleanupRequest = Body()) -> dict[str, Any]:
        async with self._sandbox_pool_lock:
            if self._sandbox_pool_prewarm_inflight:
                raise RuntimeError("Cannot cleanup sandboxes while prewarm is running")
            cleaned = await self._cleanup_prewarmed_sandboxes_locked(delete=body.delete)
            return {"cleaned": cleaned, "snapshot": self._sandbox_pool_snapshot_dict()}

    async def sandbox_pool_snapshot(self) -> HarborSandboxPoolSnapshot:
        return HarborSandboxPoolSnapshot(**self._sandbox_pool_snapshot_dict())

    async def sandbox_pool_progress(
        self,
        body: Optional[HarborSandboxProgressRequest] = Body(default=None),
    ) -> dict[str, Any]:
        body = body or HarborSandboxProgressRequest()
        limit = min(
            max(1, int(body.limit)),
            self._sandbox_pool_progress_probe_limit(),
        )
        timeout_s = max(1, int(body.timeout_s))
        requested_keys = set(body.keys or [])
        async with self._sandbox_pool_lock:
            borrowed_items = list(self._sandbox_pool_borrowed_tokens.items())
            if requested_keys:
                borrowed_items = [
                    item for item in borrowed_items if item[0] in requested_keys
                ]
            selected_items = borrowed_items[:limit]
            selected = [
                (key, token, self._sandbox_pool_handles.get(token))
                for key, token in selected_items
            ]
            missing = [
                key
                for key, token in selected_items
                if token not in self._sandbox_pool_handles
            ]
            provider = self._sandbox_pool_provider

        if provider is None:
            return {
                "snapshot": self._sandbox_pool_snapshot_dict(),
                "progress": {},
                "errors": {"provider": "sandbox pool provider is not initialized"},
            }

        progress_semaphore = asyncio.Semaphore(
            self._sandbox_pool_progress_probe_concurrency()
        )

        async def _probe(key: str, token: str, handle_ref: Any) -> tuple[str, dict[str, Any]]:
            async with progress_semaphore:
                if handle_ref is None:
                    return key, {"error": "handle_not_found"}
                try:
                    from nemo_rl.sandbox.integrations.harbor import (
                        _PROGRESS_PROBE_PATH,
                        _PROGRESS_PROBE_SCRIPT,
                    )

                    handle = await self._sandbox_pool_materialize_handle(handle_ref)
                    if token not in self._sandbox_pool_progress_probe_tokens:
                        await provider.write_file(
                            handle,
                            _PROGRESS_PROBE_PATH,
                            _PROGRESS_PROBE_SCRIPT,
                        )
                        self._sandbox_pool_progress_probe_tokens.add(token)
                    result = await provider.exec(
                        handle,
                        f"python3 {_PROGRESS_PROBE_PATH}",
                        user="root",
                        timeout_s=timeout_s,
                    )
                    if result.return_code != 0:
                        return key, {
                            "error": "progress_probe_failed",
                            "return_code": result.return_code,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    return key, json.loads(result.stdout or "{}")
                except Exception as e:
                    return key, {"error": type(e).__name__, "message": str(e)}

        progress_pairs = await asyncio.gather(
            *(_probe(key, token, handle) for key, token, handle in selected)
        )
        return {
            "snapshot": self._sandbox_pool_snapshot_dict(),
            "progress": dict(progress_pairs),
            "missing": missing,
        }

    def _get_results_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build immutable run output directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path.cwd() / "results" / "runs" / date_key / dataset_key / model_key

    def _get_jobs_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build Harbor jobs directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path(self.config.harbor_jobs_dir) / date_key / dataset_key / model_key

    @staticmethod
    def _parse_instance_id(instance_id: str) -> tuple[str, str]:
        """Parse instance id in the required form: <dataset_alias>::<task_name>."""
        dataset_alias, sep, task_name = instance_id.partition("::")
        dataset_alias = dataset_alias.strip()
        task_name = task_name.strip()
        if not sep or not dataset_alias or not task_name:
            raise ValueError(f"instance_id must be in the form '<dataset_alias>::<task_name>' (got: {instance_id!r})")
        return dataset_alias, task_name

    def _build_run_id(self, run_timestamp: datetime) -> str:
        """Build a compact run id (time + short hash) for immutable file naming."""
        time_key = run_timestamp.strftime("%H%M%S")
        return f"{time_key}_{uuid4().hex[:8]}"

    def _build_job_name(self, run_id: str) -> str:
        """Build a Harbor job name from run id only."""
        return run_id

    @staticmethod
    def _extract_model_name(policy_model_name: str) -> str:
        """Extract the final model name from a full path or HF-style identifier.

        '/lustre/.../nano-v3-sft-...-hf'  -> 'nano-v3-sft-...-hf'
        'Qwen/Qwen3-8B'                   -> 'Qwen3-8B'
        'my-model'                         -> 'my-model'
        """
        return Path(policy_model_name).name or policy_model_name

    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize path components to avoid accidental nested directories."""
        sanitized = value.replace("/", "__").replace("\\", "__").replace(":", "__")
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    @staticmethod
    def _compact_trial_result_metadata(trial_result: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Drop duplicated trajectory payloads while preserving result metadata."""
        if not trial_result:
            return {}

        metadata = dict(trial_result)
        agent_result = metadata.get("agent_result")
        if isinstance(agent_result, dict):
            compact_agent_result = dict(agent_result)
            for key in (
                "rollout_details",
                "trajectory",
                "messages",
                "conversation",
                "history",
                "steps",
                "raw_trajectory",
            ):
                compact_agent_result.pop(key, None)
            metadata["agent_result"] = compact_agent_result
        return metadata

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    def _load_policy_trace_rollout_details(self, trial_dir: Path) -> Optional[list[dict[str, Any]]]:
        if not self.config.harbor_policy_proxy:
            return None
        trace_file = str(self.config.harbor_policy_proxy.get("trace_file", "policy_trace.jsonl"))
        trace_path = trial_dir / "agent" / trace_file
        if not trace_path.exists():
            return None
        from nemo_rl.sandbox.integrations.policy_proxy import load_policy_trace_jsonl

        return list(load_policy_trace_jsonl(trace_path))

    def _sandbox_pool_config(self) -> dict[str, Any]:
        return dict(self.config.sandbox_pool or {})

    def _sandbox_pool_enabled(self) -> bool:
        return bool(self._sandbox_pool_config().get("enabled", False))

    def _sandbox_pool_acquire_policy(self) -> str:
        policy = str(
            self._sandbox_pool_config().get("acquire_policy", "direct_create")
        ).lower()
        if policy not in {"direct_create", "fail_fast"}:
            raise ValueError(
                "sandbox_pool.acquire_policy must be either 'direct_create' "
                f"or 'fail_fast', got {policy!r}"
            )
        return policy

    def _sandbox_pool_state(self) -> str:
        if self._sandbox_pool_prewarm_inflight:
            return "warming"
        if self._sandbox_pool_borrowed_tokens:
            return "running"
        if self._sandbox_pool_errors:
            return "degraded"
        if self._sandbox_pool_idle_tokens:
            return "healthy"
        return "empty"

    def _sandbox_pool_progress_probe_limit(self) -> int:
        return max(
            1,
            int(self._sandbox_pool_config().get("progress_probe_limit", 32)),
        )

    def _sandbox_pool_progress_probe_concurrency(self) -> int:
        return max(
            1,
            int(self._sandbox_pool_config().get("progress_probe_concurrency", 16)),
        )

    def _sandbox_pool_snapshot_dict(self) -> dict[str, Any]:
        return {
            "configured": self._sandbox_pool_enabled(),
            "state": self._sandbox_pool_state(),
            "idle": len(self._sandbox_pool_idle_tokens),
            "borrowed": len(self._sandbox_pool_borrowed_tokens),
            "total": len(self._sandbox_pool_handles),
            "prewarm_inflight": self._sandbox_pool_prewarm_inflight,
            "failed": len(self._sandbox_pool_errors),
            "pool_exhausted_total": self._sandbox_pool_exhausted_total,
            "direct_create_total": self._sandbox_pool_direct_create_total,
            "acquire_total": self._sandbox_pool_acquire_total,
            "acquire_hit_total": self._sandbox_pool_acquire_hit_total,
            "stale_handle_total": self._sandbox_pool_stale_handle_total,
            "release_failure_total": self._sandbox_pool_release_failure_total,
            "errors": dict(self._sandbox_pool_errors),
        }

    @staticmethod
    def _sandbox_pool_key(instance_id: str, task_index: Optional[int], rollout_index: Optional[int]) -> str:
        if task_index is None or rollout_index is None:
            return instance_id
        return f"{task_index}:{rollout_index}:{instance_id}"

    @staticmethod
    def _sandbox_pool_task_index(item: BaseModel) -> Optional[int]:
        model_extra = item.model_extra or {}
        value = model_extra.get(TASK_INDEX_KEY_NAME, getattr(item, "task_index", None))
        return int(value) if value is not None else None

    @staticmethod
    def _sandbox_pool_rollout_index(item: BaseModel) -> Optional[int]:
        model_extra = item.model_extra or {}
        value = model_extra.get(ROLLOUT_INDEX_KEY_NAME, getattr(item, "rollout_index", None))
        return int(value) if value is not None else None

    def _prewarm_item_from_run_request(self, body: HarborRunRequest) -> HarborSandboxPrewarmItem:
        model_extra = body.model_extra or {}
        return HarborSandboxPrewarmItem(
            instance_id=body.instance_id,
            task_index=model_extra.get(TASK_INDEX_KEY_NAME, getattr(body, "task_index", None)),
            rollout_index=model_extra.get(ROLLOUT_INDEX_KEY_NAME, getattr(body, "rollout_index", None)),
        )

    async def _take_prewarmed_handle(self, body: HarborRunRequest) -> tuple[Optional[str], Optional[str]]:
        item = self._prewarm_item_from_run_request(body)
        key = self._sandbox_pool_key(
            item.instance_id,
            self._sandbox_pool_task_index(item),
            self._sandbox_pool_rollout_index(item),
        )
        async with self._sandbox_pool_lock:
            if self._sandbox_pool_enabled():
                self._sandbox_pool_acquire_total += 1
            token = self._sandbox_pool_idle_tokens.pop(key, None)
            if token is None:
                token = self._sandbox_pool_idle_tokens.pop(item.instance_id, None)
            if token is not None:
                self._sandbox_pool_borrowed_tokens[key] = token
                self._sandbox_pool_acquire_hit_total += 1
                return token, key

        if self._sandbox_pool_enabled() and self._sandbox_pool_acquire_policy() == "fail_fast":
            self._sandbox_pool_exhausted_total += 1
            raise RuntimeError(f"No prewarmed sandbox available for {key!r}")
        if self._sandbox_pool_enabled():
            self._sandbox_pool_direct_create_total += 1
        return None, None

    def _register_prewarmed_handle(
        self,
        key: str,
        token: str,
        handle: Any,
        *,
        prepared_environment: bool = False,
        policy_proxy_started: bool = False,
    ) -> None:
        from nemo_rl.sandbox.integrations.harbor import (
            _PREALLOCATED_HANDLES,
            _preallocated_handle_reference,
        )

        if self._sandbox_pool_provider is None:
            raise RuntimeError("Sandbox pool provider is not initialized")
        handle_ref = _preallocated_handle_reference(
            self._sandbox_pool_provider,
            handle,
            prepared_environment=prepared_environment,
            policy_proxy_started=policy_proxy_started,
        )
        _PREALLOCATED_HANDLES[token] = handle_ref
        self._sandbox_pool_idle_tokens[key] = token
        self._sandbox_pool_handles[token] = handle_ref

    async def _sandbox_pool_materialize_handle(self, handle_ref: Any) -> Any:
        from nemo_rl.sandbox.integrations.harbor import _materialize_preallocated_handle

        if self._sandbox_pool_provider is None:
            raise RuntimeError("Sandbox pool provider is not initialized")
        return await _materialize_preallocated_handle(
            self._sandbox_pool_provider,
            handle_ref,
        )

    @staticmethod
    def _sandbox_pool_batch_name(handle_ref: Any) -> Optional[str]:
        if isinstance(handle_ref, dict):
            batch_name = handle_ref.get("batch_name")
            return str(batch_name) if batch_name is not None else None
        batch_name = getattr(getattr(handle_ref, "raw", None), "batch_name", None)
        return str(batch_name) if batch_name is not None else None

    async def _release_prewarmed_handle(
        self,
        key: Optional[str],
        token: str,
        *,
        delete: bool,
    ) -> None:
        from nemo_rl.sandbox.integrations.harbor import (
            _PREALLOCATED_HANDLES,
            _close_preallocated_handles,
        )

        async with self._sandbox_pool_lock:
            if key is not None:
                borrowed_token = self._sandbox_pool_borrowed_tokens.get(key)
                if borrowed_token is not None and borrowed_token != token:
                    return
            for borrowed_key, borrowed_token in list(
                self._sandbox_pool_borrowed_tokens.items()
            ):
                if borrowed_token == token:
                    key = borrowed_key
                    break
            for idle_key, idle_token in list(self._sandbox_pool_idle_tokens.items()):
                if idle_token == token:
                    key = idle_key
                    break
            handle_ref = self._sandbox_pool_handles.get(token)
            provider = self._sandbox_pool_provider
            batch_name = self._sandbox_pool_batch_name(handle_ref)
            batch_has_siblings = bool(
                batch_name
                and any(
                    other_token != token
                    and self._sandbox_pool_batch_name(other_handle_ref) == batch_name
                    for other_token, other_handle_ref in self._sandbox_pool_handles.items()
                )
            )

        if handle_ref is None:
            async with self._sandbox_pool_lock:
                self._sandbox_pool_stale_handle_total += 1
        elif provider is not None:
            handle = await self._sandbox_pool_materialize_handle(handle_ref)
            await _close_preallocated_handles(
                provider,
                [handle],
                delete_batch=delete and not batch_has_siblings,
            )
        async with self._sandbox_pool_lock:
            if key is not None:
                self._sandbox_pool_borrowed_tokens.pop(key, None)
                self._sandbox_pool_idle_tokens.pop(key, None)
            for borrowed_key, borrowed_token in list(
                self._sandbox_pool_borrowed_tokens.items()
            ):
                if borrowed_token == token:
                    self._sandbox_pool_borrowed_tokens.pop(borrowed_key, None)
            for idle_key, idle_token in list(self._sandbox_pool_idle_tokens.items()):
                if idle_token == token:
                    self._sandbox_pool_idle_tokens.pop(idle_key, None)
            self._sandbox_pool_handles.pop(token, None)
            self._sandbox_pool_progress_probe_tokens.discard(token)
            _PREALLOCATED_HANDLES.pop(token, None)

    def _create_sandbox_pool_provider(self) -> Any:
        from nemo_rl.sandbox.providers import create_provider

        environment_kwargs = self.config.harbor_environment_kwargs or {}
        provider_config = environment_kwargs.get("provider")
        if provider_config is None:
            raise ValueError("harbor_environment_kwargs.provider is required for sandbox prewarm")
        return create_provider(provider_config)

    def _sandbox_pool_prepare_environment(self, body: HarborSandboxPrewarmRequest) -> bool:
        if body.prepare_environment is not None:
            return body.prepare_environment
        return bool(self._sandbox_pool_config().get("prewarm_environment_setup", False))

    def _sandbox_pool_start_policy_proxy(self, body: HarborSandboxPrewarmRequest) -> bool:
        if body.start_policy_proxy is not None:
            return body.start_policy_proxy
        return bool(self._sandbox_pool_config().get("prewarm_policy_proxy", False))

    def _sandbox_pool_task_dir(self, instance_id: str) -> Path:
        dataset_alias, task_name = self._parse_instance_id(instance_id)
        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None:
            raise ValueError(f"Unknown dataset alias in instance_id: {dataset_alias!r}")
        if not dataset_source.local_dataset_path:
            raise ValueError("Sandbox prewarm currently requires local Harbor datasets")
        return Path(dataset_source.local_dataset_path) / task_name

    async def _prepare_prewarmed_handle(
        self,
        instance_id: str,
        handle: Any,
        *,
        policy_proxy_config: Optional[dict[str, Any]] = None,
    ) -> None:
        from nemo_rl.sandbox.integrations.harbor import (
            install_policy_proxy_client_config_for_handle,
            prepare_harbor_sandbox_environment,
            start_policy_proxy_for_handle,
        )

        if self._sandbox_pool_provider is None:
            raise RuntimeError("Sandbox pool provider is not initialized")
        environment_kwargs = dict(self.config.harbor_environment_kwargs or {})
        commands = environment_kwargs.get("pre_agent_setup_commands") or []
        if not isinstance(commands, list):
            raise ValueError("harbor_environment_kwargs.pre_agent_setup_commands must be a list")
        await prepare_harbor_sandbox_environment(
            provider=self._sandbox_pool_provider,
            handle=handle,
            environment_dir=self._sandbox_pool_task_dir(instance_id) / "environment",
            environment_target_dir=str(environment_kwargs.get("environment_target_dir", "/app")),
            upload_environment_dir=bool(environment_kwargs.get("upload_environment_dir", True)),
            pre_agent_setup_commands=[str(command) for command in commands],
            span_name="sandbox.prewarm.setup",
            phase="prewarm",
        )
        if policy_proxy_config is not None:
            from nemo_rl.sandbox.observability import observability_span

            async with observability_span(
                "sandbox.policy_proxy.prewarm_start",
                phase="prewarm",
            ):
                await start_policy_proxy_for_handle(
                    self._sandbox_pool_provider,
                    handle,
                    policy_proxy_config,
                )
                await install_policy_proxy_client_config_for_handle(
                    self._sandbox_pool_provider,
                    handle,
                    policy_proxy_config,
                )

    def _build_sandbox_pool_spec(self, instance_id: str) -> Any:
        from harbor.models.task.config import TaskConfig
        from nemo_rl.sandbox.integrations.harbor import _kubernetes_dns_label, _rewrite_sandbox_image
        from nemo_rl.sandbox.providers import SandboxSpec

        _, task_name = self._parse_instance_id(instance_id)
        task_dir = self._sandbox_pool_task_dir(instance_id)
        task_config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
        task_environment = task_config.environment
        environment_kwargs = dict(self.config.harbor_environment_kwargs or {})
        spec_config = dict(environment_kwargs["spec"])

        image = spec_config.get("image") or task_environment.docker_image
        image = _rewrite_sandbox_image(image, spec_config.get("image_rewrites", []))

        resources = dict(spec_config.get("resources", {}))
        if not resources:
            resources = {
                "cpu": str(task_environment.cpus),
                "memory": f"{task_environment.memory_mb}Mi",
            }

        extensions = dict(spec_config.get("extensions", {}))
        pool_ref_template = environment_kwargs.get("pool_ref_template")
        if pool_ref_template is not None and "poolRef" not in extensions:
            rendered_pool_ref = str(pool_ref_template).format(
                environment_name=task_name,
                task_name=task_name,
            )
            extensions["poolRef"] = _kubernetes_dns_label(rendered_pool_ref)

        metadata = dict(spec_config.get("metadata", {}))
        metadata.setdefault("harbor_environment_name", task_name)
        metadata.setdefault("harbor_instance_id", instance_id)

        return SandboxSpec(
            image=image,
            snapshot_id=spec_config.get("snapshot_id", None),
            timeout_s=spec_config.get("timeout_s", None),
            ready_timeout_s=spec_config.get("ready_timeout_s", None),
            env=dict(spec_config.get("env", {})),
            metadata=metadata,
            resources=resources,
            entrypoint=spec_config.get("entrypoint", None),
            extensions=extensions,
            platform=spec_config.get("platform", None),
            volumes=spec_config.get("volumes", None),
            skip_health_check=spec_config.get("skip_health_check", None),
        )

    async def _cleanup_prewarmed_sandboxes_locked(self, *, delete: bool) -> int:
        from nemo_rl.sandbox.integrations.harbor import (
            _PREALLOCATED_HANDLES,
            _close_preallocated_handles,
            _close_provider_resources,
        )

        provider = self._sandbox_pool_provider
        handles = list(self._sandbox_pool_handles.values())
        tokens = list(self._sandbox_pool_handles.keys())
        if provider is not None and handles:
            materialized_handles = await asyncio.gather(
                *(self._sandbox_pool_materialize_handle(handle) for handle in handles)
            )
            await _close_preallocated_handles(
                provider,
                list(materialized_handles),
                delete_batch=delete,
            )
        for token in tokens:
            _PREALLOCATED_HANDLES.pop(token, None)
        self._sandbox_pool_idle_tokens.clear()
        self._sandbox_pool_borrowed_tokens.clear()
        self._sandbox_pool_handles.clear()
        self._sandbox_pool_progress_probe_tokens.clear()
        if provider is not None:
            await _close_provider_resources(provider)
        self._sandbox_pool_provider = None
        return len(handles)

    def _format_config_value(self, value: Any, format_values: dict[str, str]) -> Any:
        if isinstance(value, str):
            return value.format(**format_values)
        if isinstance(value, dict):
            return {key: self._format_config_value(inner, format_values) for key, inner in value.items()}
        if isinstance(value, list):
            return [self._format_config_value(inner, format_values) for inner in value]
        return value

    def _build_environment_policy_proxy_config(
        self,
        *,
        model_name: str,
        api_base: str,
        policy_target_base_url: Optional[str],
    ) -> tuple[Optional[dict[str, Any]], dict[str, str], dict[str, str]]:
        if not self.config.harbor_policy_proxy:
            return None, {}, {
                "model_name": model_name,
                "target_base_url": api_base,
                "policy_base_url": policy_target_base_url or api_base,
            }

        proxy_config = dict(self.config.harbor_policy_proxy)
        trace_file = str(proxy_config.get("trace_file", "policy_trace.jsonl"))
        port = int(proxy_config["port"])
        proxy_root_url = f"http://127.0.0.1:{port}"
        proxy_base_url = f"{proxy_root_url}/v1"
        format_values = {
            "model_name": model_name,
            "target_base_url": api_base,
            "policy_base_url": policy_target_base_url or api_base,
            "proxy_root_url": proxy_root_url,
            "proxy_base_url": proxy_base_url,
        }

        proxy_env = {
            key: self._format_config_value(value, format_values)
            for key, value in (proxy_config.get("env") or {}).items()
        }
        environment_policy_proxy = {
            "target_base_url": self._format_config_value(
                proxy_config.get("target_base_url", "{policy_base_url}"),
                format_values,
            ),
            "port": port,
            "trace_path": str(PurePosixPath("/logs/agent") / trace_file),
            "script_path": proxy_config["script_path"],
            "env": proxy_env,
        }
        for key in (
            "backend",
            "litellm_provider",
            "upstream_model_name",
            "responses_upstream_api",
            "generation_temperature",
            "generation_top_p",
            "generation_top_k",
            "generation_chat_template_kwargs",
            "force_generation_params",
        ):
            if key in proxy_config:
                environment_policy_proxy[key] = self._format_config_value(
                    proxy_config[key],
                    format_values,
                )
        return environment_policy_proxy, proxy_env, format_values

    def _sandbox_pool_policy_proxy_config(self) -> Optional[dict[str, Any]]:
        if not self.config.harbor_policy_proxy:
            return None
        global_config_dict = get_global_config_dict()
        policy_model_name = global_config_dict["policy_model_name"]
        environment_policy_proxy, _, _ = self._build_environment_policy_proxy_config(
            model_name=policy_model_name,
            api_base=self._resolve_model_base_url(global_config_dict),
            policy_target_base_url=global_config_dict.get("policy_base_url"),
        )
        return environment_policy_proxy

    def _build_job_config(
        self,
        dataset_alias: str,
        task_name: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: Optional[dict[str, Any]] = None,
        policy_target_base_url: Optional[str] = None,
        preallocated_handle_token: Optional[str] = None,
    ) -> dict:
        """Build a Harbor JobConfig dict for a single task."""
        from harbor.models.job.config import (
            DatasetConfig,
            JobConfig,
        )
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            VerifierConfig,
        )

        (
            environment_policy_proxy,
            proxy_env,
            format_values,
        ) = self._build_environment_policy_proxy_config(
            model_name=model_name,
            api_base=api_base,
            policy_target_base_url=policy_target_base_url,
        )

        agent_kwargs: dict[str, Any] = {"api_base": api_base}
        if responses_create_params:
            agent_kwargs["responses_create_params"] = responses_create_params
            # Terminus-2 accepts temperature as a top-level kwarg for trajectory metadata.
            if "temperature" in responses_create_params:
                agent_kwargs["temperature"] = responses_create_params["temperature"]
        if self.config.harbor_agent_kwargs:
            agent_kwargs.update(self.config.harbor_agent_kwargs)

        environment_kwargs = {}
        if self.config.harbor_environment_kwargs:
            environment_kwargs.update(self.config.harbor_environment_kwargs)
        if preallocated_handle_token is not None:
            pool_config = self._sandbox_pool_config()
            environment_kwargs["preallocated_handle_token"] = preallocated_handle_token
            if "verify_preallocated_handle" in pool_config:
                environment_kwargs["verify_preallocated_handle"] = bool(
                    pool_config["verify_preallocated_handle"]
                )
            if "fallback_create_for_preallocated_handle" in pool_config:
                environment_kwargs["fallback_create_for_preallocated_handle"] = bool(
                    pool_config["fallback_create_for_preallocated_handle"]
                )

        agent_env: dict[str, str] = {}
        if self.config.harbor_agent_env:
            agent_env.update(self.config.harbor_agent_env)

        if self.config.harbor_policy_proxy and environment_policy_proxy is not None:
            environment_kwargs["policy_proxy"] = environment_policy_proxy
            agent_env.update(proxy_env)

            for key, value in (self.config.harbor_policy_proxy.get("agent_kwargs") or {}).items():
                agent_kwargs.setdefault(key, self._format_config_value(value, format_values))

        agent_model_name = model_name
        if self.config.harbor_agent_model_name:
            agent_model_name = self._format_config_value(self.config.harbor_agent_model_name, format_values)

        agent_config = AgentConfig(
            name=self.config.harbor_agent_name if not self.config.harbor_agent_import_path else None,
            import_path=self.config.harbor_agent_import_path,
            model_name=agent_model_name,
            override_timeout_sec=(
                float(self.config.harbor_agent_override_timeout)
                if self.config.harbor_agent_override_timeout is not None
                else None
            ),
            override_setup_timeout_sec=(
                float(self.config.harbor_agent_override_setup_timeout)
                if self.config.harbor_agent_override_setup_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_agent_max_timeout)
                if self.config.harbor_agent_max_timeout is not None
                else None
            ),
            kwargs=agent_kwargs,
            env=agent_env,
        )

        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None:
            available = ", ".join(sorted(self.config.harbor_datasets.keys()))
            raise ValueError(
                f"Unknown dataset alias in instance_id: {dataset_alias!r}. Available aliases: [{available}]"
            )

        has_local = bool(dataset_source.local_dataset_path)
        has_registry = bool(dataset_source.dataset_name)
        if has_local == has_registry:
            raise ValueError(
                f"Dataset alias {dataset_alias!r} must define exactly one source: "
                "local_dataset_path OR dataset_name[/dataset_version]."
            )

        # Dataset alias-level workdir overrides global harbor_environment_kwargs.workdir.
        if dataset_source.workdir is not None:
            environment_kwargs["workdir"] = dataset_source.workdir

        environment_config = EnvironmentConfig(
            type=self.config.harbor_environment_type if not self.config.harbor_environment_import_path else None,
            import_path=self.config.harbor_environment_import_path,
            kwargs=environment_kwargs,
        )

        verifier_config = VerifierConfig(
            override_timeout_sec=(
                float(self.config.harbor_verifier_override_timeout)
                if self.config.harbor_verifier_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_verifier_max_timeout)
                if self.config.harbor_verifier_max_timeout is not None
                else None
            ),
        )

        if has_registry:
            dataset_config = DatasetConfig(
                name=dataset_source.dataset_name,
                version=dataset_source.dataset_version,
                task_names=[task_name],
            )
        else:
            dataset_config = DatasetConfig(
                path=Path(dataset_source.local_dataset_path),
                task_names=[task_name],
            )

        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            timeout_multiplier=(
                self.config.harbor_timeout_multiplier if self.config.harbor_timeout_multiplier is not None else 1.0
            ),
            n_concurrent_trials=1,
            quiet=True,
            environment=environment_config,
            verifier=verifier_config,
            agents=[agent_config],
            datasets=[dataset_config],
        )

        return job_config.model_dump(mode="json")


if __name__ == "__main__":
    HarborAgent.run_webserver()
