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

"""OpenSandbox provider implementation."""

import asyncio
from dataclasses import replace
import logging
from datetime import timedelta
from pathlib import Path
import shlex
from typing import Any, Awaitable, Callable

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
)
from nemo_gym.sandbox.providers.opensandbox.batchsandbox import (
    BatchSandboxClient,
    BatchSandboxReplica,
    DEFAULT_BATCHSANDBOX_NAMESPACE,
    DEFAULT_EXECD_IMAGE,
    DEFAULT_EXECD_PORT,
    batchsandbox_name,
    build_batchsandbox_manifest,
    endpoint_with_port,
)
from nemo_gym.sandbox.observability import (
    command_attributes,
    current_recorder,
    observability_span,
)

LOGGER = logging.getLogger(__name__)


class OpenSandboxBatchCreateError(RuntimeError):
    """Raised when a batch sandbox preallocation cannot be completed."""


class OpenSandboxCreateTimeoutError(TimeoutError):
    """Raised when OpenSandbox sandbox creation exceeds the client timeout."""


class OpenSandboxCreateVerificationError(ConnectionError):
    """Raised when a newly-created sandbox cannot execute a probe command."""


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
    "connection refused",
    "connection reset",
    "gateway timeout",
    "http 408",
    "http 409",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "service unavailable",
    "server disconnected",
    "status code: 408",
    "status code: 409",
    "status code: 425",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


def _require_opensandbox_sdk() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.sandboxes import PlatformSpec, Volume
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenSandbox SDK is required for the opensandbox sandbox provider. "
            "Install it in the NeMo-RL runtime image before using "
            "env.sandbox.provider.name=opensandbox."
        ) from e

    return Sandbox, ConnectionConfig, RunCommandOpts, PlatformSpec, Volume


def _has_retryable_error_marker(exception: BaseException) -> bool:
    message = str(exception).lower()
    return any(marker in message for marker in RETRYABLE_ERROR_MARKERS)


def _is_retryable_create_error(exception: BaseException) -> bool:
    """Return whether a sandbox create failure is likely transient."""
    if isinstance(exception, (ConnectionError, OSError, TimeoutError)):
        return True

    try:
        from opensandbox.exceptions import (
            InvalidArgumentException,
            SandboxApiException,
            SandboxException,
            SandboxInternalException,
            SandboxReadyTimeoutException,
            SandboxUnhealthyException,
        )
    except ModuleNotFoundError:
        return _has_retryable_error_marker(exception)

    if isinstance(exception, InvalidArgumentException):
        return False
    if isinstance(
        exception,
        (
            SandboxInternalException,
            SandboxReadyTimeoutException,
            SandboxUnhealthyException,
        ),
    ):
        return True
    if isinstance(exception, SandboxApiException):
        status_code = getattr(exception, "status_code", None)
        if status_code in RETRYABLE_HTTP_STATUS_CODES:
            return True
        if status_code is not None and status_code < 500:
            return False
    if not isinstance(exception, SandboxException):
        return _has_retryable_error_marker(exception)

    return _has_retryable_error_marker(exception)


def _is_retryable_sdk_operation_error(exception: BaseException) -> bool:
    """Return whether a sandbox SDK operation failure is safe to retry.

    Command timeouts are intentionally not retried here: a timeout can mean the
    command is still running in the sandbox, and replaying long agent commands
    can duplicate work. HTTP 5xx and connection-level failures are retried
    because they indicate the command/file request did not complete cleanly
    through the OpenSandbox control plane.
    """
    if isinstance(exception, TimeoutError):
        return False
    if isinstance(exception, (ConnectionError, OSError)):
        return True
    return _is_retryable_create_error(exception)


def _is_missing_sandbox_delete_error(exception: BaseException) -> bool:
    message = str(exception).lower()
    return "sandbox" in message and "not found" in message


def _log_create_retry(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox sandbox create after attempt %s; "
        "next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def _log_operation_retry(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox SDK operation after attempt %s; "
        "next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def _string_map(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def _normalize_spec(spec: SandboxSpec) -> SandboxSpec:
    return replace(
        spec,
        env=_string_map(spec.env),
        metadata=_string_map(spec.metadata),
        resources=_string_map(spec.resources),
        extensions=_string_map(spec.extensions),
    )


def _to_platform_spec(platform: dict[str, Any]) -> Any:
    _, _, _, PlatformSpec, _ = _require_opensandbox_sdk()
    return PlatformSpec(**platform)


def _to_volumes(volumes: list[dict[str, Any]]) -> list[Any]:
    _, _, _, _, Volume = _require_opensandbox_sdk()
    return [Volume(**volume) for volume in volumes]


class OpenSandboxProvider:
    """Provider backed by OpenSandbox.

    ``batch_mode="sdk"`` uses ``opensandbox.Sandbox.create`` for each sandbox.
    ``batch_mode="batchsandbox"`` creates one Kubernetes ``BatchSandbox`` CR per
    batch and exposes each ready endpoint through the same provider-neutral
    ``SandboxHandle`` interface.
    """

    name = "opensandbox"

    def __init__(
        self,
        *,
        domain: str | None = None,
        api_key: str | None = None,
        protocol: str | None = None,
        use_server_proxy: bool | None = None,
        request_timeout_s: int | None = None,
        create_request_timeout_s: int | None = None,
        create_timeout_s: float | None = None,
        create_probe_command: str | None = "printf nemo-rl-sandbox-ready",
        create_probe_expected_stdout: str | None = "nemo-rl-sandbox-ready",
        create_probe_timeout_s: int = 30,
        create_probe_sample_count: int | None = None,
        batch_mode: str = "sdk",
        batch_create_concurrency: int = 4,
        batch_create_progress_timeout_s: float | None = None,
        batch_create_retries: int = 2,
        batch_create_retry_delay_s: float = 5.0,
        batch_create_retry_max_delay_s: float = 60.0,
        operation_retries: int = 3,
        operation_retry_delay_s: float = 1.0,
        operation_retry_max_delay_s: float = 15.0,
        sdk_max_connections: int | None = 512,
        sdk_max_keepalive_connections: int | None = 128,
        sdk_keepalive_expiry_s: float = 30.0,
        batchsandbox_namespace: str = DEFAULT_BATCHSANDBOX_NAMESPACE,
        batchsandbox_ready_timeout_s: float | None = None,
        batchsandbox_progress_timeout_s: float | None = None,
        batchsandbox_endpoint_only_ready_after_s: float | None = None,
        batchsandbox_poll_interval_s: float = 2.0,
        batchsandbox_execd_port: int = DEFAULT_EXECD_PORT,
        batchsandbox_execd_image: str = DEFAULT_EXECD_IMAGE,
    ) -> None:
        if batch_mode not in {"sdk", "batchsandbox"}:
            raise ValueError("batch_mode must be either 'sdk' or 'batchsandbox'")
        self._domain = domain
        self._api_key = api_key
        self._protocol = protocol
        self._use_server_proxy = use_server_proxy
        self._request_timeout_s = request_timeout_s
        self._create_request_timeout_s = create_request_timeout_s
        self._create_timeout_s = create_timeout_s
        self._create_probe_command = create_probe_command
        self._create_probe_expected_stdout = create_probe_expected_stdout
        self._create_probe_timeout_s = create_probe_timeout_s
        self._create_probe_sample_count = create_probe_sample_count
        self._batch_mode = batch_mode
        if batch_create_concurrency < 1:
            raise ValueError("batch_create_concurrency must be >= 1")
        if (
            batch_create_progress_timeout_s is not None
            and batch_create_progress_timeout_s <= 0
        ):
            raise ValueError("batch_create_progress_timeout_s must be > 0")
        if create_timeout_s is not None and create_timeout_s <= 0:
            raise ValueError("create_timeout_s must be > 0")
        if create_probe_command is not None and create_probe_timeout_s <= 0:
            raise ValueError("create_probe_timeout_s must be > 0")
        if create_probe_sample_count is not None and create_probe_sample_count < 1:
            raise ValueError("create_probe_sample_count must be >= 1")
        if batch_create_retries < 0:
            raise ValueError("batch_create_retries must be >= 0")
        if batch_create_retry_delay_s < 0:
            raise ValueError("batch_create_retry_delay_s must be >= 0")
        if batch_create_retry_max_delay_s < 0:
            raise ValueError("batch_create_retry_max_delay_s must be >= 0")
        if operation_retries < 0:
            raise ValueError("operation_retries must be >= 0")
        if operation_retry_delay_s < 0:
            raise ValueError("operation_retry_delay_s must be >= 0")
        if operation_retry_max_delay_s < 0:
            raise ValueError("operation_retry_max_delay_s must be >= 0")
        if sdk_max_connections is not None and sdk_max_connections < 1:
            raise ValueError("sdk_max_connections must be >= 1")
        if (
            sdk_max_keepalive_connections is not None
            and sdk_max_keepalive_connections < 0
        ):
            raise ValueError("sdk_max_keepalive_connections must be >= 0")
        if sdk_keepalive_expiry_s <= 0:
            raise ValueError("sdk_keepalive_expiry_s must be > 0")
        if not batchsandbox_namespace:
            raise ValueError("batchsandbox_namespace must be non-empty")
        if (
            batchsandbox_ready_timeout_s is not None
            and batchsandbox_ready_timeout_s <= 0
        ):
            raise ValueError("batchsandbox_ready_timeout_s must be > 0")
        if (
            batchsandbox_progress_timeout_s is not None
            and batchsandbox_progress_timeout_s <= 0
        ):
            raise ValueError("batchsandbox_progress_timeout_s must be > 0")
        if (
            batchsandbox_endpoint_only_ready_after_s is not None
            and batchsandbox_endpoint_only_ready_after_s <= 0
        ):
            raise ValueError("batchsandbox_endpoint_only_ready_after_s must be > 0")
        if batchsandbox_poll_interval_s <= 0:
            raise ValueError("batchsandbox_poll_interval_s must be > 0")
        if batchsandbox_execd_port <= 0:
            raise ValueError("batchsandbox_execd_port must be > 0")
        if not batchsandbox_execd_image:
            raise ValueError("batchsandbox_execd_image must be non-empty")
        self._batch_create_concurrency = batch_create_concurrency
        self._batch_create_progress_timeout_s = batch_create_progress_timeout_s
        self._batch_create_retries = batch_create_retries
        self._batch_create_retry_delay_s = batch_create_retry_delay_s
        self._batch_create_retry_max_delay_s = batch_create_retry_max_delay_s
        self._operation_retries = operation_retries
        self._operation_retry_delay_s = operation_retry_delay_s
        self._operation_retry_max_delay_s = operation_retry_max_delay_s
        self._sdk_max_connections = sdk_max_connections
        self._sdk_max_keepalive_connections = sdk_max_keepalive_connections
        self._sdk_keepalive_expiry_s = sdk_keepalive_expiry_s
        self._sdk_transport: Any | None = None
        self._sdk_transport_loop: asyncio.AbstractEventLoop | None = None
        self._batchsandbox_namespace = batchsandbox_namespace
        self._batchsandbox_ready_timeout_s = batchsandbox_ready_timeout_s
        self._batchsandbox_progress_timeout_s = batchsandbox_progress_timeout_s
        self._batchsandbox_endpoint_only_ready_after_s = (
            batchsandbox_endpoint_only_ready_after_s
        )
        self._batchsandbox_execd_port = batchsandbox_execd_port
        self._batchsandbox_execd_image = batchsandbox_execd_image
        self._batchsandbox_client = BatchSandboxClient(
            namespace=batchsandbox_namespace,
            retries=batch_create_retries,
            retry_delay_s=batch_create_retry_delay_s,
            retry_max_delay_s=batch_create_retry_max_delay_s,
            poll_interval_s=batchsandbox_poll_interval_s,
        )

    def _sdk_shared_transport(self) -> Any | None:
        if self._sdk_max_connections is None:
            return None

        loop = asyncio.get_running_loop()
        if self._sdk_transport is not None and self._sdk_transport_loop is loop:
            return self._sdk_transport

        if self._sdk_transport is not None:
            LOGGER.warning(
                "Replacing OpenSandbox SDK shared transport for a new event loop. "
                "Call OpenSandboxProvider.aclose() before reusing a provider across "
                "event loops."
            )

        import httpx

        self._sdk_transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(
                max_connections=self._sdk_max_connections,
                max_keepalive_connections=self._sdk_max_keepalive_connections,
                keepalive_expiry=self._sdk_keepalive_expiry_s,
            )
        )
        self._sdk_transport_loop = loop
        return self._sdk_transport

    def _connection_config(self, request_timeout_s: int | float | None = None) -> Any:
        _, ConnectionConfig, _, _, _ = _require_opensandbox_sdk()
        kwargs: dict[str, Any] = {}
        if self._domain is not None:
            kwargs["domain"] = self._domain
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._protocol is not None:
            kwargs["protocol"] = self._protocol
        if self._use_server_proxy is not None:
            kwargs["use_server_proxy"] = self._use_server_proxy
        if request_timeout_s is None:
            request_timeout_s = self._request_timeout_s
        if request_timeout_s is not None:
            kwargs["request_timeout"] = timedelta(seconds=request_timeout_s)
        transport = self._sdk_shared_transport()
        if transport is not None:
            kwargs["transport"] = transport
        return ConnectionConfig(**kwargs)

    async def aclose(self) -> None:
        """Close provider-owned shared SDK transport, if one was created."""
        if self._sdk_transport is None:
            return
        await self._sdk_transport.aclose()
        self._sdk_transport = None
        self._sdk_transport_loop = None

    async def _await_sdk_call(
        self,
        awaitable: Any,
        *,
        operation: str,
        sandbox_id: str,
        timeout_s: float | None,
    ) -> Any:
        if timeout_s is None:
            return await awaitable

        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Timed out during OpenSandbox {operation} after {timeout_s:g}s; "
                f"sandbox_id={sandbox_id!r}"
            ) from e

    async def _await_sdk_operation(
        self,
        operation_factory: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        sandbox_id: str,
        timeout_s: float | None,
    ) -> Any:
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_sdk_operation_error),
            stop=stop_after_attempt(self._operation_retries + 1),
            wait=wait_random_exponential(
                multiplier=self._operation_retry_delay_s,
                max=self._operation_retry_max_delay_s,
            ),
            before_sleep=_log_operation_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                return await self._await_sdk_call(
                    operation_factory(),
                    operation=operation,
                    sandbox_id=sandbox_id,
                    timeout_s=timeout_s,
                )

        raise RuntimeError("OpenSandbox SDK operation retry loop did not run")

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        if self._create_probe_command is None:
            return

        try:
            async with observability_span(
                "sandbox.create_probe",
                phase="startup",
                attributes={
                    "provider": self.name,
                    "sandbox_id": handle.sandbox_id,
                },
            ):
                result = await asyncio.wait_for(
                    self.exec(
                        handle,
                        self._create_probe_command,
                        timeout_s=self._create_probe_timeout_s,
                        user="root",
                    ),
                    timeout=self._create_probe_timeout_s,
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise OpenSandboxCreateVerificationError(
                "OpenSandbox sandbox failed create probe command; "
                f"sandbox_id={handle.sandbox_id!r}, "
                f"command={self._create_probe_command!r}"
            ) from e

        stdout = result.stdout or ""
        expected = self._create_probe_expected_stdout
        if result.return_code != 0 or (expected is not None and expected not in stdout):
            raise OpenSandboxCreateVerificationError(
                "OpenSandbox sandbox create probe command returned an "
                f"unexpected result; sandbox_id={handle.sandbox_id!r}, "
                f"return_code={result.return_code}, expected_stdout={expected!r}, "
                f"stdout={stdout[:200]!r}, stderr={(result.stderr or '')[:200]!r}"
            )

    async def _verify_created_handles(
        self,
        handles: list[SandboxHandle],
    ) -> None:
        """Verify a batch of created handles with bounded probe concurrency."""
        if self._create_probe_command is None or not handles:
            return

        handles_to_probe = handles
        if (
            self._create_probe_sample_count is not None
            and self._create_probe_sample_count < len(handles)
        ):
            sample_count = self._create_probe_sample_count
            if sample_count == 1:
                sampled_indices = [0]
            else:
                sampled_indices = [
                    round(index * (len(handles) - 1) / (sample_count - 1))
                    for index in range(sample_count)
                ]
            handles_to_probe = [handles[index] for index in sampled_indices]

        semaphore = asyncio.Semaphore(self._batch_create_concurrency)

        async def _verify_one(handle: SandboxHandle) -> None:
            async with semaphore:
                await self._verify_created_handle(handle)

        results = await asyncio.gather(
            *(_verify_one(handle) for handle in handles_to_probe),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise OpenSandboxCreateVerificationError(
                "One or more OpenSandbox sandboxes failed create probe "
                f"verification; failed={len(errors)}, total={len(handles)}"
            ) from errors[0]

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        try:
            await self.close(handle, delete=True)
        except Exception as e:
            LOGGER.warning(
                "Failed to clean up OpenSandbox sandbox after create probe "
                "failure; sandbox_id=%s; error=%r",
                handle.sandbox_id,
                e,
            )

    async def _create_once(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox through ``opensandbox.Sandbox.create``."""
        if spec.extensions.get("poolRef") and self._use_server_proxy is False:
            raise ValueError(
                "OpenSandbox pooled BatchSandbox creation requires "
                "use_server_proxy=True so SDK calls are routed through the "
                "server proxy and do not rely on stale cached pod endpoints."
            )

        Sandbox, _, _, _, _ = _require_opensandbox_sdk()

        kwargs: dict[str, Any] = {
            "env": spec.env,
            "metadata": spec.metadata,
            "resource": spec.resources,
            "extensions": spec.extensions,
            "connection_config": self._connection_config(
                request_timeout_s=self._create_request_timeout_s
            ),
        }
        if spec.image is not None:
            kwargs["image"] = spec.image
        if spec.snapshot_id is not None:
            kwargs["snapshot_id"] = spec.snapshot_id
        if spec.timeout_s is not None:
            kwargs["timeout"] = timedelta(seconds=spec.timeout_s)
        if spec.ready_timeout_s is not None:
            kwargs["ready_timeout"] = timedelta(seconds=spec.ready_timeout_s)
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        if spec.platform is not None:
            kwargs["platform"] = _to_platform_spec(spec.platform)
        if spec.volumes is not None:
            kwargs["volumes"] = _to_volumes(spec.volumes)
        if spec.skip_health_check is not None:
            kwargs["skip_health_check"] = spec.skip_health_check

        timeout_s = self._create_timeout_s
        if timeout_s is None and self._request_timeout_s is not None:
            timeout_s = float(self._request_timeout_s)

        try:
            async with observability_span(
                "sandbox.create_api",
                phase="startup",
                attributes={
                    "provider": self.name,
                    "batch_mode": self._batch_mode,
                    "image": spec.image,
                    "pool_ref": spec.extensions.get("poolRef"),
                },
            ):
                if timeout_s is None:
                    sandbox = await Sandbox.create(**kwargs)
                else:
                    sandbox = await asyncio.wait_for(
                        Sandbox.create(**kwargs),
                        timeout=timeout_s,
                    )
        except TimeoutError as e:
            raise OpenSandboxCreateTimeoutError(
                "Timed out creating OpenSandbox sandbox after "
                f"{timeout_s:g}s; image={spec.image!r}, "
                f"poolRef={spec.extensions.get('poolRef')!r}, "
                f"ready_timeout_s={spec.ready_timeout_s!r}"
            ) from e
        handle = SandboxHandle(
            sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox
        )
        try:
            await self._verify_created_handle(handle)
        except OpenSandboxCreateVerificationError:
            await self._cleanup_failed_create_handle(handle)
            raise
        return handle

    async def _create_with_retries(
        self,
        spec: SandboxSpec,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> SandboxHandle:
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_create_error),
            stop=stop_after_attempt(self._batch_create_retries + 1),
            wait=wait_random_exponential(
                multiplier=self._batch_create_retry_delay_s,
                max=self._batch_create_retry_max_delay_s,
            ),
            before_sleep=_log_create_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                if semaphore is None:
                    return await self._create_once(spec)
                async with semaphore:
                    return await self._create_once(spec)

        raise OpenSandboxBatchCreateError("OpenSandbox create retry loop did not run")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create one sandbox through the configured OpenSandbox batch mode."""
        spec = _normalize_spec(spec)
        async with observability_span(
            "sandbox.create",
            phase="startup",
            attributes={
                "provider": self.name,
                "batch_mode": self._batch_mode,
                "image": spec.image,
                "pool_ref": spec.extensions.get("poolRef"),
            },
        ):
            if self._batch_mode == "batchsandbox":
                return (await self.create_batch(spec, 1))[0]
            return await self._create_with_retries(spec)

    async def _close_many(
        self,
        handles: list[SandboxHandle],
        *,
        delete: bool,
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(self._batch_create_concurrency)

        async def _close_one(handle: SandboxHandle) -> Any:
            async with semaphore:
                return await self.close(handle, delete=delete)

        return list(
            await asyncio.gather(
                *(_close_one(handle) for handle in handles),
                return_exceptions=True,
            )
        )

    async def _create_batch_sdk(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        """Create several sandboxes through the OpenSandbox SDK.

        When the OpenSandbox server is configured with the BatchSandbox workload
        provider and ``spec.extensions`` contains ``poolRef``, each SDK create
        request is backed by a BatchSandbox allocation without NeMo-RL touching
        raw Kubernetes manifests. By default a batch is all-or-nothing. Callers
        that can make progress with fewer sandboxes may set ``allow_partial``.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        semaphore = asyncio.Semaphore(self._batch_create_concurrency)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(count):
            queue.put_nowait(index)
        handles: list[SandboxHandle | None] = [None] * count
        created_handles: list[SandboxHandle] = []
        create_errors: list[BaseException] = []
        loop = asyncio.get_running_loop()
        last_progress_at = loop.time()
        progress_timeout_error: OpenSandboxCreateTimeoutError | None = None

        async def _worker() -> None:
            nonlocal last_progress_at
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    LOGGER.info(
                        "Creating OpenSandbox sandbox %s/%s with max_attempts=%s",
                        index + 1,
                        count,
                        self._batch_create_retries + 1,
                    )
                    handle = await self._create_with_retries(
                        spec,
                        semaphore=semaphore,
                    )
                    handles[index] = handle
                    created_handles.append(handle)
                    last_progress_at = loop.time()
                    LOGGER.info(
                        "Created OpenSandbox sandbox %s/%s: %s",
                        index + 1,
                        count,
                        handle.sandbox_id,
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    create_errors.append(e)
                    last_progress_at = loop.time()
                finally:
                    queue.task_done()

        worker_count = min(count, self._batch_create_concurrency)
        workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
        queue_join: asyncio.Task[None] | None = None
        try:
            queue_join = asyncio.create_task(queue.join())
            while not queue_join.done():
                wait_timeout_s = 1.0
                if self._batch_create_progress_timeout_s is not None:
                    idle_s = loop.time() - last_progress_at
                    remaining_progress_s = (
                        self._batch_create_progress_timeout_s - idle_s
                    )
                    if remaining_progress_s <= 0:
                        progress_timeout_error = OpenSandboxCreateTimeoutError(
                            "Timed out waiting for OpenSandbox SDK batch create "
                            "progress after "
                            f"{self._batch_create_progress_timeout_s:g}s; "
                            f"requested={count}, created={len(created_handles)}, "
                            f"failed={len(create_errors)}, pending={queue.qsize()}"
                        )
                        queue_join.cancel()
                        break
                    wait_timeout_s = min(wait_timeout_s, remaining_progress_s)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(queue_join), timeout=wait_timeout_s
                    )
                except asyncio.TimeoutError:
                    if self._batch_create_progress_timeout_s is None:
                        continue
                    idle_s = loop.time() - last_progress_at
                    if idle_s < self._batch_create_progress_timeout_s:
                        continue
                    progress_timeout_error = OpenSandboxCreateTimeoutError(
                        "Timed out waiting for OpenSandbox SDK batch create "
                        "progress after "
                        f"{self._batch_create_progress_timeout_s:g}s; "
                        f"requested={count}, created={len(created_handles)}, "
                        f"failed={len(create_errors)}, pending={queue.qsize()}"
                    )
                    queue_join.cancel()
                    break
        finally:
            if queue_join is not None:
                queue_join.cancel()
                await asyncio.gather(queue_join, return_exceptions=True)
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        if progress_timeout_error is not None:
            cleanup_results = await self._close_many(created_handles, delete=True)
            cleanup_errors = [
                repr(result)
                for result in cleanup_results
                if isinstance(result, BaseException)
            ]
            if cleanup_errors:
                progress_timeout_error.args = (
                    f"{progress_timeout_error.args[0]}, "
                    f"cleanup_errors={cleanup_errors[:3]}",
                )
            raise progress_timeout_error
        complete_handles = [handle for handle in handles if handle is not None]
        prefix_handles: list[SandboxHandle] = []
        non_prefix_handles: list[SandboxHandle] = []
        saw_gap = False
        for handle in handles:
            if handle is None:
                saw_gap = True
                continue
            if saw_gap:
                non_prefix_handles.append(handle)
            else:
                prefix_handles.append(handle)

        if create_errors:
            if allow_partial and prefix_handles:
                await self._close_many(non_prefix_handles, delete=True)
                LOGGER.warning(
                    "Partially preallocated OpenSandbox sandboxes after retries: "
                    "requested=%s, returned_prefix=%s, cleaned_non_prefix=%s, "
                    "failed=%s",
                    count,
                    len(prefix_handles),
                    len(non_prefix_handles),
                    len(create_errors),
                )
                return prefix_handles

            cleanup_results = await self._close_many(created_handles, delete=True)
            cleanup_errors = [
                repr(result)
                for result in cleanup_results
                if isinstance(result, BaseException)
            ]
            error = create_errors[0]
            message = (
                "Failed to preallocate OpenSandbox sandboxes after retries: "
                f"requested={count}, created={len(created_handles)}, "
                f"failed={len(create_errors)}"
            )
            if cleanup_errors:
                message += f", cleanup_errors={cleanup_errors[:3]}"
            if isinstance(error, Exception):
                raise OpenSandboxBatchCreateError(message) from error
            raise OpenSandboxBatchCreateError(message)

        if len(complete_handles) != count:
            if allow_partial and prefix_handles:
                await self._close_many(non_prefix_handles, delete=True)
                LOGGER.warning(
                    "Partially preallocated OpenSandbox sandboxes without an "
                    "explicit create error: requested=%s, returned_prefix=%s, "
                    "cleaned_non_prefix=%s",
                    count,
                    len(prefix_handles),
                    len(non_prefix_handles),
                )
                return prefix_handles
            await self._close_many(complete_handles, delete=True)
            raise OpenSandboxBatchCreateError(
                "OpenSandbox batch create completed without errors but returned "
                f"{len(complete_handles)} of {count} handles"
            )

        return complete_handles

    async def _create_batch_batchsandbox(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        name = batchsandbox_name(spec)
        manifest = build_batchsandbox_manifest(
            spec,
            name=name,
            namespace=self._batchsandbox_namespace,
            count=count,
            execd_image=self._batchsandbox_execd_image,
            provider_name=self.name,
        )
        timeout_s = (
            self._batchsandbox_ready_timeout_s
            or spec.ready_timeout_s
            or self._create_timeout_s
            or self._request_timeout_s
            or 300.0
        )
        try:
            await self._batchsandbox_client.create(manifest)
            _, endpoints = await self._batchsandbox_client.wait_ready(
                name,
                count=count,
                timeout_s=float(timeout_s),
                progress_timeout_s=self._batchsandbox_progress_timeout_s,
                endpoint_only_ready_after_s=(
                    self._batchsandbox_endpoint_only_ready_after_s
                ),
                allow_partial=allow_partial,
            )
            connection_config = self._connection_config()
            handles = []
            for index, endpoint in enumerate(endpoints):
                raw = BatchSandboxReplica(
                    batch_name=name,
                    replica_index=index,
                    endpoint=endpoint_with_port(
                        endpoint, self._batchsandbox_execd_port
                    ),
                    connection_config=connection_config,
                    delete_batch=self.delete_batch,
                )
                handle = SandboxHandle(
                    sandbox_id=raw.id,
                    provider_name=self.name,
                    raw=raw,
                )
                handles.append(handle)
            await self._verify_created_handles(handles)
            return handles
        except TimeoutError as e:
            await self.delete_batch(name)
            raise OpenSandboxCreateTimeoutError(str(e)) from e
        except Exception:
            await self.delete_batch(name)
            raise

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        """Create several equivalent OpenSandbox sandboxes."""
        if count < 1:
            raise ValueError("count must be >= 1")
        spec = _normalize_spec(spec)
        async with observability_span(
            "sandbox.create_batch",
            phase="startup",
            attributes={
                "provider": self.name,
                "batch_mode": self._batch_mode,
                "count": count,
                "allow_partial": allow_partial,
                "image": spec.image,
                "pool_ref": spec.extensions.get("poolRef"),
            },
        ):
            if self._batch_mode == "batchsandbox":
                return await self._create_batch_batchsandbox(
                    spec,
                    count,
                    allow_partial=allow_partial,
                )
            return await self._create_batch_sdk(
                spec,
                count,
                allow_partial=allow_partial,
            )

    def handle_reference(self, handle: SandboxHandle) -> dict[str, Any]:
        """Build a loop-neutral reference for a sandbox handle.

        OpenSandbox SDK objects own httpx transports and clients that are bound
        to the event loop where they were created. Prewarmed handles may cross
        from a FastAPI prewarm request into a thread-pool runner, so only pass a
        serializable reference across that boundary and re-materialize SDK
        adapters in the consuming event loop.
        """
        if hasattr(handle.raw, "batch_name"):
            return {
                "kind": "batchsandbox_replica",
                "provider": self.name,
                "sandbox_id": handle.sandbox_id,
                "batch_name": handle.raw.batch_name,
                "replica_index": handle.raw.replica_index,
                "endpoint": handle.raw.endpoint,
            }
        return {
            "kind": "sandbox_id",
            "provider": self.name,
            "sandbox_id": handle.sandbox_id,
        }

    async def materialize_handle(self, reference: dict[str, Any]) -> SandboxHandle:
        """Create a loop-local handle from ``handle_reference`` output."""
        kind = reference.get("kind")
        if kind == "batchsandbox_replica":
            raw = BatchSandboxReplica(
                batch_name=str(reference["batch_name"]),
                replica_index=int(reference["replica_index"]),
                endpoint=str(reference["endpoint"]),
                connection_config=self._connection_config(),
                delete_batch=self.delete_batch,
            )
            return SandboxHandle(
                sandbox_id=str(reference.get("sandbox_id") or raw.id),
                provider_name=self.name,
                raw=raw,
            )
        if kind == "sandbox_id":
            return await self.connect(str(reference["sandbox_id"]))
        raise ValueError(f"Unsupported OpenSandbox handle reference kind: {kind!r}")

    async def delete_batch(self, batch_name: str) -> None:
        """Delete one provider-created BatchSandbox CR by name."""
        async with observability_span(
            "sandbox.delete_batch",
            phase="cleanup",
            attributes={
                "provider": self.name,
                "batch_name": batch_name,
            },
        ):
            await self._batchsandbox_client.delete(batch_name)

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        """Connect to an existing OpenSandbox sandbox."""
        Sandbox, _, _, _, _ = _require_opensandbox_sdk()
        sandbox = await Sandbox.connect(
            sandbox_id, connection_config=self._connection_config()
        )
        return SandboxHandle(
            sandbox_id=str(sandbox.id), provider_name=self.name, raw=sandbox
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run a command inside an OpenSandbox sandbox."""
        _, _, RunCommandOpts, _, _ = _require_opensandbox_sdk()

        opts_kwargs: dict[str, Any] = {}
        if cwd is not None:
            opts_kwargs["working_directory"] = cwd
        if env is not None:
            opts_kwargs["envs"] = env
        if timeout_s is not None:
            opts_kwargs["timeout"] = timedelta(seconds=timeout_s)

        effective_command = command
        if isinstance(user, int):
            opts_kwargs["uid"] = user
        elif isinstance(user, str) and user != "root":
            effective_command = (
                f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(user)}"
            )

        sdk_timeout_s = (
            float(timeout_s) + 60.0
            if timeout_s is not None
            else (
                float(self._request_timeout_s)
                if self._request_timeout_s is not None
                else None
            )
        )
        recorder = current_recorder()
        include_command_text = (
            recorder.include_command_text if recorder is not None else False
        )
        async with observability_span(
            "sandbox.exec",
            phase="execution",
            attributes={
                "provider": self.name,
                "sandbox_id": handle.sandbox_id,
                **command_attributes(
                    command,
                    include_command_text=include_command_text,
                ),
            },
        ):
            execution = await self._await_sdk_operation(
                lambda: handle.raw.commands.run(
                    effective_command, opts=RunCommandOpts(**opts_kwargs)
                ),
                operation="command run",
                sandbox_id=handle.sandbox_id,
                timeout_s=sdk_timeout_s,
            )
        stdout = "\n".join(msg.text for msg in execution.logs.stdout) or None
        stderr_parts = [msg.text for msg in execution.logs.stderr]
        if execution.error is not None:
            stderr_parts.append(f"{execution.error.name}: {execution.error.value}")
        stderr = "\n".join(stderr_parts) or None
        if execution.exit_code is not None:
            return_code = execution.exit_code
        elif execution.error is not None:
            return_code = 1
        else:
            return_code = 0

        return SandboxExecResult(
            stdout=stdout, stderr=stderr, return_code=return_code
        )

    async def write_file(
        self, handle: SandboxHandle, target_path: str, data: str | bytes
    ) -> None:
        """Write one file into an OpenSandbox sandbox."""
        async with observability_span(
            "sandbox.write_file",
            phase="setup",
            attributes={
                "provider": self.name,
                "sandbox_id": handle.sandbox_id,
                "target_path": target_path,
                "bytes": len(data),
            },
        ):
            await self._await_sdk_operation(
                lambda: handle.raw.files.write_file(target_path, data),
                operation=f"write_file({target_path})",
                sandbox_id=handle.sandbox_id,
                timeout_s=float(self._request_timeout_s)
                if self._request_timeout_s is not None
                else None,
            )

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        """Read one file from an OpenSandbox sandbox."""
        async with observability_span(
            "sandbox.read_file",
            phase="execution",
            attributes={
                "provider": self.name,
                "sandbox_id": handle.sandbox_id,
                "source_path": source_path,
            },
        ):
            return await self._await_sdk_operation(
                lambda: handle.raw.files.read_bytes(source_path),
                operation=f"read_file({source_path})",
                sandbox_id=handle.sandbox_id,
                timeout_s=float(self._request_timeout_s)
                if self._request_timeout_s is not None
                else None,
            )

    async def upload_file(
        self, handle: SandboxHandle, source_path: Path, target_path: str
    ) -> None:
        """Upload one local file into an OpenSandbox sandbox."""
        await self.write_file(handle, target_path, source_path.read_bytes())

    async def download_file(
        self, handle: SandboxHandle, source_path: str, target_path: Path
    ) -> None:
        """Download one file from an OpenSandbox sandbox."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(await self.read_file(handle, source_path))

    async def close(self, handle: SandboxHandle, *, delete: bool) -> None:
        """Close local SDK resources and optionally terminate the sandbox."""
        async with observability_span(
            "sandbox.close",
            phase="cleanup",
            attributes={
                "provider": self.name,
                "sandbox_id": handle.sandbox_id,
                "delete": delete,
                "batch_name": getattr(handle.raw, "batch_name", None),
            },
        ):
            if hasattr(handle.raw, "batch_name"):
                if delete:
                    await self.delete_batch(handle.raw.batch_name)
                await handle.raw.close()
                return

            kill_error: Exception | None = None
            if delete:
                retry_policy = AsyncRetrying(
                    retry=retry_if_exception(_is_retryable_create_error),
                    stop=stop_after_attempt(self._batch_create_retries + 1),
                    wait=wait_random_exponential(
                        multiplier=self._batch_create_retry_delay_s,
                        max=self._batch_create_retry_max_delay_s,
                    ),
                    before_sleep=_log_create_retry,
                    reraise=True,
                )
                try:
                    async for attempt in retry_policy:
                        with attempt:
                            await handle.raw.kill()
                except Exception as e:
                    if not _is_missing_sandbox_delete_error(e):
                        kill_error = e
                    else:
                        LOGGER.info(
                            "OpenSandbox sandbox %r was already deleted during close",
                            handle.sandbox_id,
                        )

            close_error: Exception | None = None
            try:
                await handle.raw.close()
            except Exception as e:
                close_error = e

            if kill_error is not None:
                if close_error is not None:
                    raise RuntimeError(
                        "Failed to delete and close OpenSandbox sandbox "
                        f"{handle.sandbox_id!r}: delete_error={kill_error!r}, "
                        f"close_error={close_error!r}"
                    ) from kill_error
                raise kill_error
            if close_error is not None:
                raise close_error
