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

"""Harbor environment backed by the public NeMo Gym sandbox API."""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import uuid4

from nemo_gym.sandbox import (
    Sandbox,
    SandboxBatchCreateError,
    SandboxCreateVerificationError,
    SandboxHandle,
    SandboxSpec,
)
from nemo_gym.sandbox.config import SandboxProviderConfig
from nemo_gym.sandbox.observability import (
    SandboxResourceSampler,
    build_recorder_from_config,
    current_recorder,
    ensure_env_recorder,
    event_context,
    ingest_agent_trajectory_events,
    observability_span,
    push_event_context,
    record_event,
    reset_current_recorder,
    reset_event_context,
    set_current_recorder,
    use_recorder,
)
from nemo_gym.sandbox.observability.render import safe_report_name
from responses_api_agents.harbor_agent.trajectory import (
    SandboxRolloutContext,
    SandboxTrajectory,
)


G_LOGGER = logging.getLogger(__name__)
SandboxConfig = dict[str, Any]


def _require_harbor() -> dict[str, Any]:
    try:
        from harbor.environments.base import BaseSandbox, ExecResult
        from harbor.models.trial.config import TrialConfig
        from harbor.models.trial.paths import EnvironmentPaths
        from harbor.trial.trial import Trial
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Harbor is required for the Gym harbor_agent sandbox environment. "
            "Install Harbor in the runtime image before using this agent."
        ) from e

    return {
        "BaseSandbox": BaseSandbox,
        "EnvironmentPaths": EnvironmentPaths,
        "ExecResult": ExecResult,
        "Trial": Trial,
        "TrialConfig": TrialConfig,
    }


class _SandboxTypeValue:
    value = "sandbox"


_OPENSANDBOX_METADATA_VALUE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_KUBERNETES_DNS_LABEL_RE = re.compile(r"[^a-z0-9-]+")


def _opensandbox_metadata_value(value: Any) -> str:
    """Return an OpenSandbox/Kubernetes-safe metadata value.

    Harbor task names intentionally use package names like ``org/name``. The
    OpenSandbox API accepts metadata values with a Kubernetes-label-like shape,
    so integration-owned metadata must be normalized at this adapter boundary.
    """
    sanitized = _OPENSANDBOX_METADATA_VALUE_RE.sub("_", str(value)).strip("._-")
    sanitized = sanitized[:63].strip("._-")
    return sanitized or "metadata"


def _kubernetes_dns_label(value: Any, *, max_length: int = 63) -> str:
    """Return a stable Kubernetes DNS label for generated resource names."""
    sanitized = _KUBERNETES_DNS_LABEL_RE.sub("-", str(value).lower()).strip("-")
    if not sanitized:
        return "resource"
    if len(sanitized) <= max_length:
        return sanitized

    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:8]
    prefix = sanitized[: max_length - len(digest) - 1].rstrip("-")
    return f"{prefix}-{digest}" if prefix else digest


def _harbor_task_name(environment_name: str) -> str:
    """Return the Harbor task name for pool routing."""
    return environment_name


def _rewrite_sandbox_image(
    image: str | None,
    rewrites: list[dict[str, str]],
) -> str | None:
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image


try:
    _HARBOR_IMPORTS = _require_harbor()
    _BaseSandbox = _HARBOR_IMPORTS["BaseSandbox"]
except ModuleNotFoundError:
    _BaseSandbox = object


_PREALLOCATED_HANDLE_ROW_KEY = "_nemo_rl_preallocated_handle_token"
_PREALLOCATED_HANDLES: dict[str, Any] = {}
_PREALLOCATED_ENVIRONMENT_PREPARED_KEY = "prepared_environment"


def _preallocated_handle_reference(
    provider: Any,
    handle: SandboxHandle,
    *,
    prepared_environment: bool = False,
) -> Any:
    """Return a loop-neutral preallocated-handle value when supported."""
    make_reference = getattr(provider, "handle_reference", None)
    if make_reference is None:
        return handle
    reference = make_reference(handle)
    if isinstance(reference, dict) and prepared_environment:
        reference = dict(reference)
        reference[_PREALLOCATED_ENVIRONMENT_PREPARED_KEY] = True
    return reference


def _preallocated_handle_environment_prepared(value: Any) -> bool:
    """Return whether deterministic environment setup was done in prewarm."""
    return bool(isinstance(value, dict) and value.get(_PREALLOCATED_ENVIRONMENT_PREPARED_KEY, False))


async def _materialize_preallocated_handle(
    provider: Any,
    value: Any,
) -> SandboxHandle:
    """Resolve a preallocated-handle value in the current event loop."""
    if isinstance(value, SandboxHandle):
        return value

    materialize = getattr(provider, "materialize_handle", None)
    if materialize is None:
        raise ValueError(
            f"This sandbox provider cannot materialize preallocated handle references: {type(provider).__name__}"
        )
    result = materialize(value)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, SandboxHandle):
        raise TypeError(f"materialize_handle must return SandboxHandle, got {type(result).__name__}")
    return result


_START_PROBE_COMMAND = "printf nemo-rl-sandbox-start-ready"
_START_PROBE_EXPECTED_STDOUT = "nemo-rl-sandbox-start-ready"
_AGENT_LOG_DIR = "/logs/agent"
_AGENT_LOG_TRANSFER_EXCLUDED_PARTS = frozenset(
    {
        ".cache",
        ".local",
        ".npm",
        ".tmp",
        "node_modules",
    }
)
_SANDBOX_RUNTIME_ERROR_MARKERS = (
    "502",
    "503",
    "504",
    "bad gateway",
    "connection refused",
    "connection reset",
    "endpoint",
    "failed create probe",
    "failed start probe",
    "gateway timeout",
    "server disconnected",
    "service unavailable",
    "start probe",
    "timed out",
    "timeout",
)


def _is_agent_log_transfer(target_dir: str) -> bool:
    return PurePosixPath(target_dir).as_posix().rstrip("/") == _AGENT_LOG_DIR


def _should_skip_agent_log_transfer(relative_path: str | PurePosixPath) -> bool:
    return bool(_AGENT_LOG_TRANSFER_EXCLUDED_PARTS.intersection(PurePosixPath(relative_path).parts))


async def _exec_provider_checked(
    provider: Any,
    handle: SandboxHandle,
    command: str,
    *,
    phase: str,
    cwd: str | None = "/",
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
    user: str | int | None = None,
) -> Any:
    result = await provider.exec(
        handle,
        command,
        cwd=cwd,
        env=env,
        timeout_s=timeout_s,
        user=user,
    )
    if result.return_code != 0:
        raise RuntimeError(
            "OpenSandbox bootstrap command failed with exit "
            f"{result.return_code}; phase={phase}; command={command}; "
            f"stdout={(result.stdout or '')[:2000]}; "
            f"stderr={(result.stderr or '')[:2000]}"
        )
    return result


async def _upload_dir_to_handle(
    provider: Any,
    handle: SandboxHandle,
    source_dir: Path | str,
    target_dir: str,
) -> None:
    source_root = Path(source_dir)
    await _exec_provider_checked(
        provider,
        handle,
        f"mkdir -p {shlex.quote(target_dir)}",
        phase="upload_environment_mkdir",
        user="root",
    )
    if await _upload_dir_tar_to_handle(provider, handle, source_root, target_dir):
        return
    await _upload_dir_file_by_file(provider, handle, source_root, target_dir)


def _iter_upload_paths(
    source_root: Path,
    *,
    skip_agent_caches: bool,
) -> Iterator[tuple[Path, PurePosixPath]]:
    for dir_path, dir_names, file_names in os.walk(source_root):
        relative_dir = Path(dir_path).relative_to(source_root)
        if skip_agent_caches:
            dir_names[:] = [
                dirname
                for dirname in dir_names
                if not _should_skip_agent_log_transfer(PurePosixPath(relative_dir.as_posix()) / dirname)
            ]
        for file_name in file_names:
            local_path = Path(dir_path) / file_name
            if relative_dir.as_posix() == ".":
                relative_path = PurePosixPath(file_name)
            else:
                relative_path = PurePosixPath(relative_dir.as_posix()) / file_name
            if skip_agent_caches and _should_skip_agent_log_transfer(relative_path):
                continue
            yield local_path, relative_path


async def _upload_dir_tar_to_handle(
    provider: Any,
    handle: SandboxHandle,
    source_root: Path,
    target_dir: str,
) -> bool:
    """Upload a directory as one archive when the sandbox has tar."""
    skip_agent_caches = _is_agent_log_transfer(target_dir)
    archive_remote_path = f"/tmp/nemo_rl_upload_{uuid4().hex}.tar.gz"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, mode="w:gz") as tar:
            for local_path, relative_path in _iter_upload_paths(source_root, skip_agent_caches=skip_agent_caches):
                tar.add(local_path, arcname=relative_path.as_posix(), recursive=False)
        archive.flush()
        await provider.upload_file(handle, Path(archive.name), archive_remote_path)

    result = await provider.exec(
        handle,
        "command -v tar >/dev/null 2>&1 && "
        f"tar -xzf {shlex.quote(archive_remote_path)} "
        f"-C {shlex.quote(target_dir)} && "
        f"rm -f {shlex.quote(archive_remote_path)}",
        user="root",
        cwd="/",
    )
    if result.return_code == 0:
        return True
    await provider.exec(
        handle,
        f"rm -f {shlex.quote(archive_remote_path)}",
        user="root",
        cwd="/",
    )
    G_LOGGER.warning(
        "Falling back to per-file sandbox directory upload; tar upload failed with exit %s; stdout=%s; stderr=%s",
        result.return_code,
        (result.stdout or "")[:500],
        (result.stderr or "")[:500],
    )
    return False


async def _upload_dir_file_by_file(
    provider: Any,
    handle: SandboxHandle,
    source_root: Path,
    target_dir: str,
) -> None:
    skip_agent_caches = _is_agent_log_transfer(target_dir)
    for local_path, relative_path in _iter_upload_paths(source_root, skip_agent_caches=skip_agent_caches):
        remote_path = str(PurePosixPath(target_dir) / relative_path)
        await _exec_provider_checked(
            provider,
            handle,
            f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))}",
            phase="upload_environment_parent",
            user="root",
        )
        await provider.upload_file(handle, local_path, remote_path)


async def prepare_harbor_sandbox_environment(
    *,
    provider: Any,
    handle: SandboxHandle,
    environment_dir: Path | str,
    environment_target_dir: str,
    upload_environment_dir: bool,
    pre_agent_setup_commands: list[str] | None,
    span_name: str,
    phase: str,
) -> None:
    """Prepare deterministic Harbor sandbox filesystem state.

    This helper intentionally excludes per-trial policy-proxy startup and
    verifier work. Those remain on the borrow/execution path so test files and
    trial-local traces are not leaked into the agent phase.
    """
    imports = _require_harbor()
    environment_paths = imports["EnvironmentPaths"]

    agent_dir = shlex.quote(str(environment_paths.agent_dir))
    verifier_dir = shlex.quote(str(environment_paths.verifier_dir))
    artifacts_dir = shlex.quote(str(environment_paths.artifacts_dir))
    target_dir = shlex.quote(environment_target_dir.rstrip("/") or "/app")
    async with observability_span(span_name, phase=phase):
        await _exec_provider_checked(
            provider,
            handle,
            "rm -rf "
            f"{agent_dir} {verifier_dir} {artifacts_dir}; "
            "mkdir -p "
            f"{agent_dir} {verifier_dir} {artifacts_dir} {target_dir} "
            '"$HOME/.local/bin"; '
            "chmod 777 "
            f"{agent_dir} {verifier_dir} {artifacts_dir}",
            phase="bootstrap_dirs",
            user="root",
        )
        if upload_environment_dir:
            async with observability_span(
                "sandbox.setup.upload_environment",
                phase=phase,
            ):
                await _upload_dir_to_handle(
                    provider,
                    handle,
                    environment_dir,
                    environment_target_dir,
                )
        for command in pre_agent_setup_commands or []:
            async with observability_span(
                "sandbox.pre_agent_setup",
                phase=phase,
            ):
                result = await provider.exec(
                    handle,
                    command,
                    user="root",
                    cwd="/",
                )
            if result.return_code != 0:
                raise RuntimeError(
                    "OpenSandbox pre-agent setup command failed with exit "
                    f"{result.return_code}: {command}\n"
                    f"stdout: {(result.stdout or '')[:2000]}\n"
                    f"stderr: {(result.stderr or '')[:2000]}"
                )


class SandboxHarborEnvironment(_BaseSandbox):
    """Harbor ``BaseSandbox`` implemented through NeMo Gym's sandbox layer.

    Harbor owns the agent and verifier workflow. The underlying runtime and
    infrastructure provider is the configured NeMo Gym sandbox provider.
    """

    def __init__(
        self,
        *args: Any,
        provider: SandboxProviderConfig | None = None,
        spec: dict[str, Any] | None = None,
        environment_target_dir: str = "/app",
        upload_environment_dir: bool = True,
        preallocated_handle_token: str | None = None,
        pre_agent_setup_commands: list[str] | None = None,
        default_cwd: str | None = None,
        default_exec_timeout_s: int | None = None,
        pool_ref_template: str | None = None,
        verify_preallocated_handle: bool = True,
        fallback_create_for_preallocated_handle: bool = True,
        observability_context: dict[str, Any] | None = None,
        start_probe_command: str | None = _START_PROBE_COMMAND,
        start_probe_timeout_s: int = 30,
        start_probe_expected_stdout: str | None = _START_PROBE_EXPECTED_STDOUT,
        **kwargs: Any,
    ) -> None:
        if provider is None:
            raise ValueError("SandboxHarborEnvironment requires environment.kwargs.provider")
        if spec is None:
            raise ValueError("SandboxHarborEnvironment requires environment.kwargs.spec")
        self._provider_config = provider
        self._sandbox = Sandbox(provider)
        self._spec_config = spec
        self._environment_target_dir = environment_target_dir.rstrip("/") or "/app"
        self._upload_environment_dir = upload_environment_dir
        self._preallocated_handle_token = preallocated_handle_token
        self._pre_agent_setup_commands = pre_agent_setup_commands or []
        self._default_cwd = default_cwd
        self._pool_ref_template = pool_ref_template
        self._verify_preallocated_handle = verify_preallocated_handle
        self._fallback_create_for_preallocated_handle = fallback_create_for_preallocated_handle
        if start_probe_command is not None and start_probe_timeout_s <= 0:
            raise ValueError("start_probe_timeout_s must be > 0")
        if default_exec_timeout_s is not None and default_exec_timeout_s <= 0:
            raise ValueError("default_exec_timeout_s must be > 0")
        self._start_probe_command = start_probe_command
        self._start_probe_timeout_s = start_probe_timeout_s
        self._start_probe_expected_stdout = start_probe_expected_stdout
        self._default_exec_timeout_s = default_exec_timeout_s
        self._default_cwd_enabled = False
        self._handle: SandboxHandle | None = None
        self._observability_context = observability_context or {}
        self._observability_recorder_token: Any | None = None
        self._observability_context_token: Any | None = None
        self._resource_sampler: SandboxResourceSampler | None = None

        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> Any:
        return _SandboxTypeValue()

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return True

    def _validate_definition(self) -> None:
        if "image" not in self._spec_config and "snapshot_id" not in self._spec_config:
            docker_image = getattr(self.task_env_config, "docker_image", None)
            if docker_image is None:
                raise ValueError(
                    "SandboxHarborEnvironment requires spec.image, spec.snapshot_id, or task environment docker_image"
                )

    def _build_spec(self) -> SandboxSpec:
        spec_config = dict(self._spec_config)
        image = spec_config.get("image", None)
        if image is None:
            image = getattr(self.task_env_config, "docker_image", None)
        image = _rewrite_sandbox_image(
            image,
            spec_config.get("image_rewrites", []),
        )

        resources = dict(spec_config.get("resources", {}))
        if not resources:
            resources = {
                "cpu": str(self.task_env_config.cpus),
                "memory": f"{self.task_env_config.memory_mb}Mi",
            }

        env = dict(spec_config.get("env", {}))
        persistent_env = self._merge_env(None)
        if persistent_env is not None:
            env.update(persistent_env)

        metadata = {
            key: _opensandbox_metadata_value(value)
            for key, value in {
                **spec_config.get("metadata", {}),
                "harbor_session_id": self.session_id,
                "harbor_environment_name": self.environment_name,
            }.items()
        }
        extensions = dict(spec_config.get("extensions", {}))
        if self._pool_ref_template is not None and "poolRef" not in extensions:
            task_name = _harbor_task_name(self.environment_name)
            try:
                rendered_pool_ref = self._pool_ref_template.format(
                    environment_name=self.environment_name,
                    task_name=task_name,
                )
            except KeyError as exc:
                raise ValueError("pool_ref_template may only reference {environment_name} and {task_name}") from exc
            extensions["poolRef"] = _kubernetes_dns_label(rendered_pool_ref)

        return SandboxSpec(
            image=image,
            snapshot_id=spec_config.get("snapshot_id", None),
            timeout_s=spec_config.get("timeout_s", None),
            ready_timeout_s=spec_config.get("ready_timeout_s", None),
            env=env,
            metadata=metadata,
            resources=resources,
            entrypoint=spec_config.get("entrypoint", None),
            extensions=extensions,
            platform=spec_config.get("platform", None),
            volumes=spec_config.get("volumes", None),
            skip_health_check=spec_config.get("skip_health_check", None),
        )

    def _activate_observability(self) -> None:
        recorder = current_recorder() or ensure_env_recorder()
        if recorder is not None and current_recorder() is None:
            self._observability_recorder_token = set_current_recorder(recorder)
        attrs = {
            "trajectory_id": self._observability_context.get("trajectory_id", self.environment_name),
            "harbor_session_id": self.session_id,
            "environment_name": self.environment_name,
            **self._observability_context,
        }
        self._observability_context_token = push_event_context(attrs)

    def _deactivate_observability(self) -> None:
        if self._observability_context_token is not None:
            try:
                reset_event_context(self._observability_context_token)
            except ValueError:
                pass
            self._observability_context_token = None
        if self._observability_recorder_token is not None:
            try:
                reset_current_recorder(self._observability_recorder_token)
            except ValueError:
                pass
            self._observability_recorder_token = None

    def _start_resource_sampler(self) -> None:
        recorder = current_recorder()
        if recorder is None or recorder.resource_sampler_interval_s() <= 0:
            return
        if self._handle is None or self._resource_sampler is not None:
            return
        self._resource_sampler = SandboxResourceSampler(
            provider=self._sandbox,
            handle=self._handle,
            recorder=recorder,
            interval_s=recorder.resource_sampler_interval_s(),
            process_trace=recorder.process_trace,
            attributes={
                "trajectory_id": self.environment_name,
                "harbor_session_id": self.session_id,
                "environment_name": self.environment_name,
            },
        )
        self._resource_sampler.start()

    async def start(self, force_build: bool) -> None:
        del force_build
        self._activate_observability()
        preallocated_environment_prepared = False
        async with observability_span(
            "sandbox.start",
            phase="startup",
            attributes={
                "environment_name": self.environment_name,
                "harbor_session_id": self.session_id,
            },
        ):
            if self._preallocated_handle_token is not None:
                try:
                    preallocated_value = _PREALLOCATED_HANDLES[self._preallocated_handle_token]
                except KeyError as e:
                    raise ValueError(
                        f"Unknown preallocated OpenSandbox handle token {self._preallocated_handle_token!r}"
                    ) from e
                preallocated_environment_prepared = _preallocated_handle_environment_prepared(preallocated_value)
                self._handle = await _materialize_preallocated_handle(
                    self._sandbox,
                    preallocated_value,
                )
                if self._verify_preallocated_handle:
                    try:
                        await self._verify_start_probe()
                    except RuntimeError:
                        if not self._fallback_create_for_preallocated_handle:
                            raise
                        G_LOGGER.warning(
                            "Preallocated OpenSandbox handle failed start probe; "
                            "sandbox_id=%s; creating a replacement sandbox",
                            self._handle.sandbox_id,
                            exc_info=True,
                        )
                        self._handle = None
                        self._preallocated_handle_token = None
                        self._handle = await self._sandbox.create(self._build_spec())
                        preallocated_environment_prepared = False
            else:
                self._handle = await self._sandbox.create(self._build_spec())
            self._start_resource_sampler()
        async with observability_span("sandbox.setup", phase="setup"):
            if not preallocated_environment_prepared:
                await prepare_harbor_sandbox_environment(
                    provider=self._sandbox,
                    handle=self._require_handle(),
                    environment_dir=self.environment_dir,
                    environment_target_dir=self._environment_target_dir,
                    upload_environment_dir=self._upload_environment_dir,
                    pre_agent_setup_commands=self._pre_agent_setup_commands,
                    span_name="sandbox.borrow.setup",
                    phase="setup",
                )
            self._default_cwd_enabled = self._default_cwd is not None

    async def _exec_checked(
        self,
        command: str,
        *,
        phase: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        result = await self.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "OpenSandbox bootstrap command failed with exit "
                f"{result.return_code}; phase={phase}; command={command}; "
                f"stdout={(result.stdout or '')[:2000]}; "
                f"stderr={(result.stderr or '')[:2000]}"
            )
        return result

    async def _verify_start_probe(self) -> None:
        if self._start_probe_command is None:
            return

        try:
            result = await asyncio.wait_for(
                self.exec(
                    self._start_probe_command,
                    user="root",
                    timeout_sec=self._start_probe_timeout_s,
                ),
                timeout=self._start_probe_timeout_s + 5,
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenSandbox handle failed start probe command; command={self._start_probe_command!r}"
            ) from e

        stdout = result.stdout or ""
        expected = self._start_probe_expected_stdout
        if result.return_code != 0 or (expected is not None and expected not in stdout):
            raise RuntimeError(
                "OpenSandbox handle start probe command returned an "
                f"unexpected result; return_code={result.return_code}, "
                f"expected_stdout={expected!r}, stdout={stdout[:200]!r}, "
                f"stderr={(result.stderr or '')[:200]!r}"
            )

    async def stop(self, delete: bool) -> None:
        if self._resource_sampler is not None:
            await self._resource_sampler.stop()
            self._resource_sampler = None
        try:
            if self._handle is not None:
                if self._preallocated_handle_token is not None:
                    self._handle = None
                    return
                await self._sandbox.close(
                    self._handle,
                    delete=delete,
                )
                self._handle = None
        finally:
            await _close_provider_resources(self._sandbox)
            self._deactivate_observability()

    async def prepare_logs_for_host(self) -> None:
        return None

    def _require_handle(self) -> SandboxHandle:
        if self._handle is None:
            raise RuntimeError("SandboxHarborEnvironment has not been started")
        return self._handle

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        await self._sandbox.upload_file(self._require_handle(), source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source_root = Path(source_dir)
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", cwd="/", user="root")
        skip_agent_caches = _is_agent_log_transfer(target_dir)
        for dir_path, dir_names, file_names in os.walk(source_root):
            relative_dir = Path(dir_path).relative_to(source_root)
            if skip_agent_caches:
                dir_names[:] = [
                    dirname
                    for dirname in dir_names
                    if not _should_skip_agent_log_transfer(PurePosixPath(relative_dir.as_posix()) / dirname)
                ]
            for file_name in file_names:
                local_path = Path(dir_path) / file_name
                if relative_dir.as_posix() == ".":
                    relative_path = PurePosixPath(file_name)
                else:
                    relative_path = PurePosixPath(relative_dir.as_posix()) / file_name
                if skip_agent_caches and _should_skip_agent_log_transfer(relative_path):
                    continue
                remote_path = str(PurePosixPath(target_dir) / relative_path)
                await self.exec(
                    f"mkdir -p {shlex.quote(str(PurePosixPath(remote_path).parent))}",
                    cwd="/",
                    user="root",
                )
                await self.upload_file(local_path, remote_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._sandbox.download_file(self._require_handle(), source_path, Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        is_agent_log_transfer = _is_agent_log_transfer(source_dir)
        result = await self.exec(f"find {shlex.quote(source_dir)} -type f -print", cwd="/", user="root")
        if result.return_code != 0 or not result.stdout:
            return
        for remote_file in result.stdout.splitlines():
            relative = PurePosixPath(remote_file).relative_to(PurePosixPath(source_dir))
            if is_agent_log_transfer and _should_skip_agent_log_transfer(relative):
                continue
            await self.download_file(remote_file, target / relative.as_posix())
        if is_agent_log_transfer:
            recorder = current_recorder()
            if recorder is not None:
                trajectory_id = self._observability_context.get("trajectory_id", self.environment_name)
                ingest_agent_trajectory_events(
                    target,
                    recorder=recorder,
                    trajectory_id=str(trajectory_id),
                )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        imports = _require_harbor()
        exec_result_type = imports["ExecResult"]
        merged_env = self._merge_env(env)
        user = self._resolve_user(user)
        exec_cwd = cwd
        if exec_cwd is None and self._default_cwd_enabled:
            exec_cwd = self._default_cwd
        if timeout_sec is None:
            timeout_sec = self._default_exec_timeout_s
        result = await self._sandbox.exec(
            self._require_handle(),
            command,
            cwd=exec_cwd,
            env=merged_env,
            timeout_s=timeout_sec,
            user=user,
        )
        return exec_result_type(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _format_config_value(value: Any, format_values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**format_values)
    if isinstance(value, list):
        return [_format_config_value(item, format_values) for item in value]
    if isinstance(value, dict):
        return {key: _format_config_value(item, format_values) for key, item in value.items()}
    return value


def _first_policy_base_url(context: SandboxRolloutContext) -> str:
    if not context.base_urls or context.base_urls[0] is None:
        raise ValueError(
            "Sandbox trajectory collection requires an exposed policy HTTP "
            "endpoint. Set generation.vllm_cfg.expose_http_server=true."
        )
    return context.base_urls[0]


def _reward_from_result(result: Any, sandbox_config: SandboxConfig) -> tuple[float, dict[str, float | int]]:
    rewards = None
    if result.verifier_result is not None:
        rewards = result.verifier_result.rewards
    if not rewards:
        return 0.0, {}

    trajectory_config = sandbox_config["trajectory"]
    reward_key = trajectory_config.get("reward_key")
    if reward_key is not None:
        if reward_key not in rewards:
            raise ValueError(f"Harbor verifier result has no reward key {reward_key!r}")
        return float(rewards[reward_key]), rewards

    if len(rewards) == 1:
        value = next(iter(rewards.values()))
        return float(value), rewards

    raise ValueError("Harbor verifier emitted multiple rewards. Set env.sandbox.trajectory.reward_key.")


def _stop_reason_from_result(result: Any) -> str:
    if result.exception_info is None:
        return "complete"

    exception_type = result.exception_info.exception_type
    if exception_type == "ContextLengthExceededError":
        return "context_length"
    if exception_type == "AgentTimeoutError":
        return "agent_timeout"
    return "error"


def _is_retryable_sandbox_runtime_error(exception: BaseException) -> bool:
    if isinstance(
        exception,
        (
            ConnectionError,
            TimeoutError,
            OSError,
            SandboxCreateVerificationError,
        ),
    ):
        return True

    status_code = getattr(exception, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status_code is not None and status_code < 500:
        return False

    message = f"{type(exception).__name__}: {exception}".lower()
    return any(marker in message for marker in _SANDBOX_RUNTIME_ERROR_MARKERS)


def _observability_artifacts(trajectory_id: str | None) -> dict[str, str] | None:
    recorder = current_recorder() or ensure_env_recorder()
    if recorder is None:
        return None
    artifacts = {
        "output_dir": str(recorder.output_dir),
        "events_jsonl": str(recorder.events_path),
        "resource_samples_jsonl": str(recorder.resource_samples_path),
        "summary_json": str(recorder.output_dir / "summary.json"),
        "reports_index": str(recorder.output_dir / "reports" / "index.html"),
    }
    if trajectory_id:
        report_stem = safe_report_name(trajectory_id)
        artifacts["trajectory_html"] = str(recorder.output_dir / "reports" / f"{report_stem}.html")
        artifacts["trajectory_png"] = str(recorder.output_dir / "reports" / f"{report_stem}.png")
    return artifacts


def _record_trajectory_event(
    *,
    name: str,
    trajectory_id: str | None,
    reward: float | None,
    stop_reason: str,
    duration_s: float,
    loss_multiplier: float | None = None,
) -> None:
    record_event(
        "trajectory",
        name,
        attributes={
            "trajectory_id": trajectory_id,
            "reward": reward,
            "stop_reason": stop_reason,
            "duration_s": duration_s,
            "loss_multiplier": loss_multiplier,
        },
    )


def _full_result_from_harbor(
    result: Any,
    rewards: dict[str, float | int],
    *,
    stop_reason: str,
    attempt_idx: int,
) -> dict[str, Any]:
    full_result = cast(dict[str, Any], result.model_dump(mode="json"))
    full_result["rewards"] = rewards
    full_result["stop_reason"] = stop_reason
    full_result["attempt_idx"] = attempt_idx
    if result.exception_info is not None:
        full_result["exception_info"] = result.exception_info.model_dump(mode="json")
    observability = _observability_artifacts(result.trial_name)
    if observability is not None:
        full_result["observability"] = observability
    return full_result


def _agent_name_from_result_or_config(result: Any | None, trial_config: Any) -> str:
    if result is not None and result.agent_info is not None:
        return result.agent_info.name
    return trial_config.agent.name or trial_config.agent.import_path or "harbor"


def _masked_trajectory(
    *,
    trial_config: Any,
    result: Any | None,
    stop_reason: str,
    error_message: str | None,
) -> SandboxTrajectory:
    full_result: dict[str, Any] = {
        "masked": True,
        "stop_reason": stop_reason,
    }
    if error_message is not None:
        full_result["error_message"] = error_message
    if result is not None:
        full_result["result"] = result.model_dump(mode="json")
    observability = _observability_artifacts(getattr(trial_config, "trial_name", None))
    if observability is not None:
        full_result["observability"] = observability

    return {
        "rollout_details": [
            {
                "prompt_token_ids": [[0]],
                "completion_token_ids": [[0]],
                "logprobs": [[0.0]],
            }
        ],
        "reward": 0.0,
        "full_result": full_result,
        "agent_name": _agent_name_from_result_or_config(result, trial_config),
        "truncated": True,
        "loss_multiplier": 0.0,
        "stop_reason": stop_reason,
    }


def _validate_harbor_config(sandbox_config: SandboxConfig) -> None:
    integration_kwargs = sandbox_config["integration"]["kwargs"]
    for required_key in ("trial_config", "environment_spec", "max_retries"):
        if required_key not in integration_kwargs:
            raise ValueError(
                f"env.sandbox.integration.kwargs.{required_key} is required for the Harbor sandbox integration"
            )

    if not integration_kwargs.get("eval_only", False):
        agent_kwargs = dict(integration_kwargs["trial_config"].get("agent", {}).get("kwargs", {}))
        agent_kwargs.update(integration_kwargs.get("agent_kwargs", {}))
        if not agent_kwargs.get("collect_rollout_details", False):
            raise ValueError(
                "Harbor-native trainable trajectories require "
                "env.sandbox.integration.kwargs.trial_config.agent.kwargs."
                "collect_rollout_details=true, or the same key under "
                "env.sandbox.integration.kwargs.agent_kwargs. Set integration_kwargs."
                "eval_only=true to bypass this check for evaluation runs "
                "that do not need trainable token IDs/logprobs."
            )

    rate_limit_config = integration_kwargs.get("rate_limit", None)
    if isinstance(rate_limit_config, dict) and rate_limit_config.get("enabled", False):
        trajectories_per_second = rate_limit_config.get("trajectories_per_second", None)
        if trajectories_per_second is not None and float(trajectories_per_second) <= 0.0:
            raise ValueError("rate_limit.trajectories_per_second must be > 0")
        max_concurrency = rate_limit_config.get("max_concurrency", None)
        if max_concurrency is not None and int(max_concurrency) < 1:
            raise ValueError("rate_limit.max_concurrency must be >= 1")


def _trial_config_for_row(
    row: dict[str, Any],
    sandbox_config: SandboxConfig,
    context: SandboxRolloutContext,
) -> Any:
    imports = _require_harbor()
    trial_config_type = imports["TrialConfig"]

    integration_kwargs = sandbox_config["integration"]["kwargs"]
    trial_template = integration_kwargs["trial_config"]
    row_override = row.get("harbor_trial_config", {})
    config_dict = _deep_merge_dict(trial_template, row_override)

    environment_cfg = config_dict.setdefault("environment", {})
    environment_cfg["import_path"] = "responses_api_agents.harbor_agent.sandbox:SandboxHarborEnvironment"
    environment_cfg["type"] = None
    environment_kwargs = environment_cfg.setdefault("kwargs", {})
    environment_kwargs["provider"] = sandbox_config["provider"]
    environment_kwargs.setdefault(
        "observability_context",
        {
            "prompt_group_id": row.get("sandbox_prompt_group_id"),
            "prompt_idx": row.get("prompt_idx"),
            "generation_idx": row.get("generation_idx"),
        },
    )
    preallocated_handle_token = row.get(_PREALLOCATED_HANDLE_ROW_KEY)
    if preallocated_handle_token is not None:
        environment_kwargs["preallocated_handle_token"] = preallocated_handle_token
    if "spec" in environment_kwargs:
        environment_kwargs["spec"] = _deep_merge_dict(
            environment_kwargs["spec"], integration_kwargs["environment_spec"]
        )
    else:
        environment_kwargs["spec"] = integration_kwargs["environment_spec"]

    policy_base_url = _first_policy_base_url(context)
    format_values = {
        "model_name": context.model_name,
        "target_base_url": policy_base_url,
    }
    agent_cfg = config_dict.setdefault("agent", {})
    if "model_name" in agent_cfg:
        agent_cfg["model_name"] = _format_config_value(agent_cfg["model_name"], format_values)
    if "agent_model_name" in integration_kwargs:
        agent_cfg["model_name"] = _format_config_value(integration_kwargs["agent_model_name"], format_values)
    if "agent_kwargs" in integration_kwargs:
        agent_kwargs = agent_cfg.setdefault("kwargs", {})
        for key, value in integration_kwargs["agent_kwargs"].items():
            agent_kwargs[key] = _format_config_value(value, format_values)

    if "pre_agent_setup_commands" in integration_kwargs:
        environment_kwargs["pre_agent_setup_commands"] = _format_config_value(
            integration_kwargs["pre_agent_setup_commands"], format_values
        )
    if "default_exec_timeout_s" in integration_kwargs:
        environment_kwargs["default_exec_timeout_s"] = integration_kwargs["default_exec_timeout_s"]

    if "policy_endpoint_env" in integration_kwargs:
        agent_cfg = config_dict.setdefault("agent", {})
        agent_env = agent_cfg.setdefault("env", {})
        for env_name in integration_kwargs["policy_endpoint_env"]:
            agent_env[env_name] = policy_base_url

    return trial_config_type.model_validate(config_dict)


def _ingest_agent_trajectory_observability(trial: Any) -> None:
    recorder = current_recorder()
    if recorder is None:
        return
    ingest_agent_trajectory_events(
        trial._trial_paths.agent_dir,
        recorder=recorder,
        trajectory_id=trial.config.trial_name,
    )


async def _run_harbor_trial(
    row: dict[str, Any],
    sandbox_config: SandboxConfig,
    context: SandboxRolloutContext,
) -> SandboxTrajectory:
    imports = _require_harbor()
    trial_type = imports["Trial"]
    integration_kwargs = sandbox_config["integration"]["kwargs"]
    max_retries = integration_kwargs["max_retries"]
    attempt_row = dict(row)
    trial_config = _trial_config_for_row(attempt_row, sandbox_config, context)
    result = None
    last_error_message = None
    stop_reason = "error"
    trajectory_start_s = time.monotonic()

    for attempt_idx in range(max_retries):
        trial_config = _trial_config_for_row(attempt_row, sandbox_config, context)
        async with observability_span(
            "harbor.trial.create",
            phase="setup",
            attributes={"attempt_idx": attempt_idx},
        ):
            trial = await trial_type.create(trial_config)
        with event_context(
            trajectory_id=trial.config.trial_name,
            trial_name=trial.config.trial_name,
            attempt_idx=attempt_idx,
        ):
            try:
                async with observability_span(
                    "harbor.trial.run",
                    phase="execution",
                    attributes={"attempt_idx": attempt_idx},
                ):
                    result = await trial.run()
            except Exception as e:
                last_error_message = f"{type(e).__name__}: {e}"
                if _is_retryable_sandbox_runtime_error(e):
                    stop_reason = "sandbox_runtime_lost"
                    if _PREALLOCATED_HANDLE_ROW_KEY in attempt_row:
                        G_LOGGER.warning(
                            "Retrying Harbor trial without preallocated sandbox "
                            "after sandbox runtime loss; trial=%s attempt=%s error=%r",
                            trial.config.trial_name,
                            attempt_idx,
                            e,
                        )
                        attempt_row = dict(attempt_row)
                        attempt_row.pop(_PREALLOCATED_HANDLE_ROW_KEY, None)
                else:
                    stop_reason = "error"
                continue
            finally:
                _ingest_agent_trajectory_observability(trial)

        stop_reason = _stop_reason_from_result(result)
        if stop_reason == "agent_timeout":
            _record_trajectory_event(
                name="trajectory.masked",
                trajectory_id=trial.config.trial_name,
                reward=0.0,
                stop_reason=stop_reason,
                duration_s=time.monotonic() - trajectory_start_s,
                loss_multiplier=0.0,
            )
            return _masked_trajectory(
                trial_config=trial_config,
                result=result,
                stop_reason=stop_reason,
                error_message=None,
            )

        rollout_details = None
        if result.agent_result is not None:
            rollout_details = result.agent_result.rollout_details

        if not rollout_details:
            last_error_message = f"Harbor trial {result.trial_name} did not produce rollout_details"
            continue

        if stop_reason == "context_length":
            reward = 0.0
            rewards = {}
        elif result.verifier_result is None:
            last_error_message = f"Harbor trial {result.trial_name} did not produce verifier_result"
            continue
        else:
            reward, rewards = _reward_from_result(result, sandbox_config)

        _record_trajectory_event(
            name="trajectory.complete",
            trajectory_id=result.trial_name,
            reward=reward,
            stop_reason=stop_reason,
            duration_s=time.monotonic() - trajectory_start_s,
            loss_multiplier=1.0,
        )
        return {
            "rollout_details": rollout_details,
            "reward": reward,
            "full_result": _full_result_from_harbor(
                result,
                rewards,
                stop_reason=stop_reason,
                attempt_idx=attempt_idx,
            ),
            "agent_name": result.agent_info.name,
            "truncated": result.exception_info is not None,
            "stop_reason": stop_reason,
        }

    _record_trajectory_event(
        name="trajectory.masked",
        trajectory_id=getattr(trial_config, "trial_name", None),
        reward=0.0,
        stop_reason=stop_reason,
        duration_s=time.monotonic() - trajectory_start_s,
        loss_multiplier=0.0,
    )
    return _masked_trajectory(
        trial_config=trial_config,
        result=result,
        stop_reason=stop_reason,
        error_message=last_error_message,
    )


async def _rate_limit(
    *,
    rate_limit_config: dict[str, Any] | None,
    lock: asyncio.Lock,
    next_start_time: list[float],
) -> None:
    if rate_limit_config is None:
        return
    if "enabled" not in rate_limit_config or not rate_limit_config["enabled"]:
        return

    trajectories_per_second = rate_limit_config.get("trajectories_per_second", None)
    if trajectories_per_second is None:
        return

    min_interval_s = 1.0 / float(trajectories_per_second)
    loop = asyncio.get_running_loop()
    async with lock:
        now = loop.time()
        sleep_s = max(0.0, next_start_time[0] - now)
        next_start_time[0] = max(now, next_start_time[0]) + min_interval_s

    if sleep_s > 0.0:
        await asyncio.sleep(sleep_s)


def _semaphore_for_rate_limit(
    rate_limit_config: dict[str, Any] | None,
) -> asyncio.Semaphore | None:
    if rate_limit_config is None:
        return None
    if "enabled" not in rate_limit_config or not rate_limit_config["enabled"]:
        return None
    max_concurrency = rate_limit_config.get("max_concurrency", None)
    if max_concurrency is None:
        return None
    return asyncio.Semaphore(int(max_concurrency))


def _prompt_group_id(row: dict[str, Any], row_idx: int) -> Any:
    if "sandbox_prompt_group_id" in row:
        return row["sandbox_prompt_group_id"]
    if "idx" in row:
        return row["idx"]
    return json.dumps(row, sort_keys=True, default=str)


def _mask_failed_prompt_groups(
    trajectories: list[SandboxTrajectory],
    rows: list[dict[str, Any]],
) -> list[SandboxTrajectory]:
    prompt_group_ids = [_prompt_group_id(row, row_idx) for row_idx, row in enumerate(rows)]
    failed_groups = {
        group_id
        for group_id, trajectory in zip(prompt_group_ids, trajectories)
        if trajectory.get("loss_multiplier", 1.0) == 0.0
    }
    if not failed_groups:
        return trajectories

    masked_trajectories: list[SandboxTrajectory] = []
    for group_id, trajectory in zip(prompt_group_ids, trajectories):
        if group_id not in failed_groups:
            masked_trajectories.append(trajectory)
            continue

        masked = dict(trajectory)
        full_result = dict(masked.get("full_result", {}))
        full_result["masked_prompt_group_id"] = group_id
        masked["full_result"] = full_result
        masked["loss_multiplier"] = 0.0
        masked_trajectories.append(cast(SandboxTrajectory, masked))

    return masked_trajectories


def _spec_from_environment_spec(spec_config: dict[str, Any]) -> SandboxSpec:
    return SandboxSpec(
        image=spec_config.get("image", None),
        snapshot_id=spec_config.get("snapshot_id", None),
        timeout_s=spec_config.get("timeout_s", None),
        ready_timeout_s=spec_config.get("ready_timeout_s", None),
        env=dict(spec_config.get("env", {})),
        metadata=dict(spec_config.get("metadata", {})),
        resources=dict(spec_config.get("resources", {})),
        entrypoint=spec_config.get("entrypoint", None),
        extensions=dict(spec_config.get("extensions", {})),
        platform=spec_config.get("platform", None),
        volumes=spec_config.get("volumes", None),
        skip_health_check=spec_config.get("skip_health_check", None),
    )


def _task_environment_for_row(row: dict[str, Any]) -> dict[str, Any]:
    task_path = row.get("harbor_trial_config", {}).get("task", {}).get("path", None)
    if task_path is None:
        return {}

    task_toml_path = Path(task_path) / "task.toml"
    try:
        with task_toml_path.open("rb") as f:
            task_config = tomllib.load(f)
    except FileNotFoundError:
        return {}

    environment = task_config.get("environment", {})
    if not isinstance(environment, dict):
        return {}
    return environment


def _preallocation_environment_spec(
    rows: list[dict[str, Any]],
    integration_kwargs: dict[str, Any],
) -> dict[str, Any]:
    spec_config = deepcopy(integration_kwargs["environment_spec"])
    if spec_config.get("image") is not None or spec_config.get("snapshot_id") is not None:
        return spec_config

    environments = [_task_environment_for_row(row) for row in rows]
    images = {
        environment.get("docker_image") for environment in environments if environment.get("docker_image") is not None
    }
    if not images:
        return spec_config
    if len(images) != 1:
        raise ValueError(
            "Preallocated Harbor sandboxes require one image per preallocation "
            "chunk. Use a static environment_spec.image, or choose a "
            "preallocate_batch_size that keeps SWE task-image chunks together."
        )

    spec_config["image"] = _rewrite_sandbox_image(
        next(iter(images)),
        spec_config.get("image_rewrites", []),
    )
    if not spec_config.get("resources"):
        first_environment = environments[0]
        resources = {}
        if first_environment.get("cpus") is not None:
            resources["cpu"] = str(first_environment["cpus"])
        if first_environment.get("memory") is not None:
            resources["memory"] = str(first_environment["memory"])
        if resources:
            spec_config["resources"] = resources
    return spec_config


def _rows_with_preallocated_handles(
    rows: list[dict[str, Any]],
    handles: list[SandboxHandle],
    provider: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(rows) != len(handles):
        raise ValueError("Preallocated handle count must match row count")

    tokens = []
    rows_with_handles = []
    for row, handle in zip(rows, handles, strict=True):
        token = uuid4().hex
        _PREALLOCATED_HANDLES[token] = _preallocated_handle_reference(provider, handle)
        tokens.append(token)
        row_with_handle = dict(row)
        row_with_handle[_PREALLOCATED_HANDLE_ROW_KEY] = token
        rows_with_handles.append(row_with_handle)
    return rows_with_handles, tokens


async def _close_preallocated_handles(
    provider: Any,
    handles: list[SandboxHandle],
    *,
    delete: bool,
) -> None:
    if not handles:
        return

    close_concurrency = max(1, int(getattr(provider, "_batch_create_concurrency", 8)))
    close_semaphore = asyncio.Semaphore(close_concurrency)

    async def _close_one(handle: SandboxHandle) -> Any:
        async with close_semaphore:
            return await provider.close(
                handle,
                delete=delete,
            )

    close_results = await asyncio.gather(
        *(_close_one(handle) for handle in handles),
        return_exceptions=True,
    )

    cleanup_errors = [result for result in close_results if isinstance(result, Exception)]
    if cleanup_errors:
        raise RuntimeError(
            "Failed to clean up one or more preallocated sandbox handles: "
            + "; ".join(repr(error) for error in cleanup_errors[:3])
        ) from cleanup_errors[0]


async def _close_provider_resources(provider: Any) -> None:
    """Close optional provider-owned resources without masking caller errors."""
    close_provider = getattr(provider, "aclose", None)
    if not callable(close_provider):
        return
    try:
        await close_provider()
    except Exception:
        G_LOGGER.warning(
            "Failed to close sandbox provider resources",
            exc_info=True,
        )


def _masked_preallocation_trajectories(
    rows: list[dict[str, Any]],
    sandbox_config: SandboxConfig,
    context: SandboxRolloutContext,
    error: SandboxBatchCreateError,
) -> list[SandboxTrajectory]:
    return [
        _masked_trajectory(
            trial_config=_trial_config_for_row(row, sandbox_config, context),
            result=None,
            stop_reason="sandbox_preallocate_error",
            error_message=f"{type(error).__name__}: {error}",
        )
        for row in rows
    ]


def _append_checkpoint_trajectory(
    checkpoint_path: str | None,
    row: dict[str, Any],
    trajectory: SandboxTrajectory,
) -> None:
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"row": row, "trajectory": trajectory}) + "\n")


def _preallocate_batch_size(
    integration_kwargs: dict[str, Any],
    rate_limit_config: dict[str, Any] | None,
    row_count: int,
) -> int:
    configured = integration_kwargs.get("preallocate_batch_size", None)
    if configured is None and rate_limit_config is not None:
        configured = rate_limit_config.get("max_concurrency", None)
    batch_size = row_count if configured is None else int(configured)
    if batch_size < 1:
        raise ValueError("preallocate_batch_size must be >= 1")
    return min(batch_size, row_count)


def collect_harbor_trajectories(
    rows: list[dict[str, Any]],
    sandbox_config: SandboxConfig,
    context: SandboxRolloutContext,
) -> list[SandboxTrajectory]:
    """Run Harbor trials and return trainable sandbox trajectories."""
    _validate_harbor_config(sandbox_config)
    integration_kwargs = sandbox_config["integration"]["kwargs"]
    rate_limit_config = None
    if "rate_limit" in integration_kwargs:
        rate_limit_config = integration_kwargs["rate_limit"]
    checkpoint_path = integration_kwargs.get("checkpoint_output_path", None)
    owned_recorder = build_recorder_from_config(
        sandbox_config.get("observability", None),
    )

    async def _collect() -> list[SandboxTrajectory]:
        semaphore = _semaphore_for_rate_limit(rate_limit_config)
        rate_limit_lock = asyncio.Lock()
        next_start_time = [0.0]

        async def _run_one(row: dict[str, Any]) -> SandboxTrajectory:
            await _rate_limit(
                rate_limit_config=rate_limit_config,
                lock=rate_limit_lock,
                next_start_time=next_start_time,
            )
            if semaphore is None:
                return await _run_harbor_trial(row, sandbox_config, context)
            async with semaphore:
                return await _run_harbor_trial(row, sandbox_config, context)

        async def _run_row_guarded(row: dict[str, Any]) -> SandboxTrajectory:
            try:
                return await _run_one(row)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return _masked_trajectory(
                    trial_config=_trial_config_for_row(row, sandbox_config, context),
                    result=None,
                    stop_reason="error",
                    error_message=f"{type(e).__name__}: {e}",
                )

        async def _run_rows(run_rows: list[dict[str, Any]]) -> list[SandboxTrajectory]:
            async def _run_indexed(row_idx: int, row: dict[str, Any]) -> tuple[int, SandboxTrajectory]:
                return row_idx, await _run_row_guarded(row)

            tasks = [asyncio.create_task(_run_indexed(row_idx, row)) for row_idx, row in enumerate(run_rows)]
            results: list[SandboxTrajectory | None] = [None] * len(run_rows)
            try:
                for task in asyncio.as_completed(tasks):
                    row_idx, trajectory = await task
                    results[row_idx] = trajectory
                    _append_checkpoint_trajectory(checkpoint_path, run_rows[row_idx], trajectory)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return [trajectory for trajectory in results if trajectory is not None]

        if integration_kwargs.get("preallocate_sandboxes", False):
            provider = Sandbox(sandbox_config["provider"])
            try:
                batch_size = _preallocate_batch_size(integration_kwargs, rate_limit_config, len(rows))
                trajectories = []
                for offset in range(0, len(rows), batch_size):
                    chunk = rows[offset : offset + batch_size]
                    spec = _spec_from_environment_spec(_preallocation_environment_spec(chunk, integration_kwargs))
                    G_LOGGER.info(
                        "Preallocating sandbox chunk offset=%s size=%s total_rows=%s",
                        offset,
                        len(chunk),
                        len(rows),
                    )
                    try:
                        async with observability_span(
                            "sandbox.preallocate_chunk",
                            phase="startup",
                            attributes={
                                "offset": offset,
                                "count": len(chunk),
                            },
                        ):
                            handles = await cast(Any, provider).create_batch(
                                spec,
                                len(chunk),
                                allow_partial=True,
                            )
                    except SandboxBatchCreateError as e:
                        G_LOGGER.warning(
                            "Failed to preallocate sandbox chunk offset=%s size=%s: %s",
                            offset,
                            len(chunk),
                            e,
                        )
                        if integration_kwargs.get("mask_failed_preallocation", True):
                            masked = _masked_preallocation_trajectories(chunk, sandbox_config, context, e)
                            for row, trajectory in zip(chunk, masked, strict=True):
                                _append_checkpoint_trajectory(checkpoint_path, row, trajectory)
                            trajectories.extend(masked)
                            continue
                        raise
                    G_LOGGER.info(
                        "Preallocated sandbox chunk offset=%s size=%s",
                        offset,
                        len(handles),
                    )
                    rows_with_handles, tokens = _rows_with_preallocated_handles(
                        chunk[: len(handles)],
                        handles,
                        provider,
                    )
                    masked_partial: list[SandboxTrajectory] = []
                    if len(handles) < len(chunk):
                        failed_rows = chunk[len(handles) :]
                        error = SandboxBatchCreateError(
                            "OpenSandbox batch preallocation returned a partial "
                            f"chunk: requested={len(chunk)}, created={len(handles)}"
                        )
                        masked_partial = _masked_preallocation_trajectories(
                            failed_rows, sandbox_config, context, error
                        )
                    try:
                        trajectories.extend(await _run_rows(rows_with_handles))
                        if masked_partial:
                            for row, trajectory in zip(chunk[len(handles) :], masked_partial, strict=True):
                                _append_checkpoint_trajectory(checkpoint_path, row, trajectory)
                            trajectories.extend(masked_partial)
                    finally:
                        G_LOGGER.info(
                            "Cleaning preallocated sandbox chunk offset=%s size=%s",
                            offset,
                            len(handles),
                        )
                        for token in tokens:
                            _PREALLOCATED_HANDLES.pop(token, None)
                        async with observability_span(
                            "sandbox.preallocate_cleanup",
                            phase="cleanup",
                            attributes={
                                "offset": offset,
                                "count": len(handles),
                            },
                        ):
                            await _close_preallocated_handles(
                                provider,
                                handles,
                                delete=integration_kwargs.get("delete_preallocated_batch", True),
                            )
                        G_LOGGER.info(
                            "Cleaned preallocated sandbox chunk offset=%s size=%s",
                            offset,
                            len(handles),
                        )
            finally:
                await _close_provider_resources(provider)
        else:
            trajectories = await _run_rows(rows)

        if integration_kwargs.get("mask_failed_prompt_group", None):
            return _mask_failed_prompt_groups(trajectories, rows)
        return trajectories

    try:
        with use_recorder(owned_recorder):
            return asyncio.run(_collect())
    finally:
        if owned_recorder is not None:
            owned_recorder.finalize()
