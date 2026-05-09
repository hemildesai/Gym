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

"""Direct Kubernetes BatchSandbox helpers for OpenSandbox."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
import shlex
from typing import Any, Callable, Awaitable
from uuid import uuid4

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from nemo_gym.sandbox.observability import observability_span, record_event
from nemo_gym.sandbox.providers.base import SandboxSpec

LOGGER = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
BATCHSANDBOX_GROUP = "sandbox.opensandbox.io"
BATCHSANDBOX_VERSION = "v1alpha1"
BATCHSANDBOX_PLURAL = "batchsandboxes"
BATCHSANDBOX_ENDPOINTS_ANNOTATION = "sandbox.opensandbox.io/endpoints"
DEFAULT_EXECD_PORT = 44772
DEFAULT_BATCHSANDBOX_NAMESPACE = "opensandbox"
DEFAULT_EXECD_IMAGE = "opensandbox/execd:v1.0.15"
CPU_POOL_AFFINITY = {
    "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [
                {
                    "matchExpressions": [
                        {
                            "key": "nodeGroup",
                            "operator": "In",
                            "values": ["customer-cpu"],
                        }
                    ]
                },
                {
                    "matchExpressions": [
                        {
                            "key": "karpenter.sh/nodepool",
                            "operator": "In",
                            "values": ["cpu"],
                        },
                        {
                            "key": "karpenter.k8s.aws/instance-generation",
                            "operator": "Gt",
                            "values": ["5"],
                        },
                    ]
                },
            ]
        },
        "preferredDuringSchedulingIgnoredDuringExecution": [
            {
                "weight": 100,
                "preference": {
                    "matchExpressions": [
                        {
                            "key": "nodeGroup",
                            "operator": "In",
                            "values": ["customer-cpu"],
                        }
                    ]
                },
            }
        ],
    }
}
CPU_POOL_TOLERATIONS = [{"operator": "Exists"}]
K8S_DNS_LABEL_RE = re.compile(r"[^a-z0-9-]+")
K8S_LABEL_VALUE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def require_opensandbox_endpoint_adapters() -> tuple[Any, Any, Any]:
    """Load SDK adapters used to talk directly to ready execd endpoints."""
    try:
        from opensandbox.adapters.command_adapter import CommandsAdapter
        from opensandbox.adapters.filesystem_adapter import FilesystemAdapter
        from opensandbox.models.sandboxes import SandboxEndpoint
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenSandbox SDK endpoint adapters are required for "
            "OpenSandboxProvider(batch_mode='batchsandbox'). Install the "
            "OpenSandbox SDK in the NeMo-RL runtime image."
        ) from e

    return CommandsAdapter, FilesystemAdapter, SandboxEndpoint


def require_kubernetes_client() -> tuple[Any, Any, Any]:
    """Load the official Kubernetes Python client."""
    try:
        from kubernetes import client, config
        from kubernetes.client.exceptions import ApiException
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The kubernetes Python package is required for "
            "OpenSandboxProvider(batch_mode='batchsandbox'). Install "
            "`kubernetes` in the coordinator runtime."
        ) from e

    return client, config, ApiException


def load_kubernetes_config_once() -> None:
    """Pick in-cluster config first, then kubeconfig, exactly once."""
    if getattr(load_kubernetes_config_once, "_loaded", False):
        return
    _, config, _ = require_kubernetes_client()
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    load_kubernetes_config_once._loaded = True  # type: ignore[attr-defined]


def is_retryable_kubernetes_error(exception: BaseException) -> bool:
    """Return whether a Kubernetes API failure is likely transient."""
    try:
        _, _, ApiException = require_kubernetes_client()
    except ModuleNotFoundError:
        return False
    if isinstance(exception, ApiException):
        return exception.status in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exception, (ConnectionError, OSError, TimeoutError))


def log_retry(retry_state: RetryCallState) -> None:
    """Log one BatchSandbox Kubernetes retry."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else None
    LOGGER.warning(
        "Retrying OpenSandbox BatchSandbox Kubernetes call after attempt %s; "
        "next_sleep_s=%s; error=%r",
        retry_state.attempt_number,
        sleep_s,
        exception,
    )


def dns_label(value: str, *, default: str = "nemo-rl-sandbox") -> str:
    """Return a DNS-label-safe Kubernetes resource name fragment."""
    normalized = K8S_DNS_LABEL_RE.sub("-", value.lower())[:63].strip("-")
    return normalized or default


def label_value(value: str) -> str:
    """Return a Kubernetes-label-value-safe string."""
    normalized = K8S_LABEL_VALUE_RE.sub("_", value)[:63].strip("._-")
    return normalized or "unknown"


def batchsandbox_name(spec: SandboxSpec) -> str:
    """Build a short unique BatchSandbox name."""
    run_id = spec.metadata.get("nemo-rl.nvidia.com/run-id", "nemo-rl")
    prefix = dns_label(f"{run_id}-bsbx", default="nemo-rl-bsbx")
    suffix = uuid4().hex[:10]
    return f"{prefix[:52].rstrip('-')}-{suffix}"


def batchsandbox_labels(spec: SandboxSpec, provider_name: str) -> dict[str, str]:
    """Build labels for provider-created BatchSandbox resources."""
    labels = {
        "app.kubernetes.io/name": "nemo-rl-sandbox",
        "app.kubernetes.io/component": "opensandbox-batchsandbox",
        "app.kubernetes.io/managed-by": "nemo-rl",
        "nemo-rl.nvidia.com/provider": provider_name,
        "nemo-rl.nvidia.com/batch-mode": "batchsandbox",
    }
    for key, value in spec.metadata.items():
        labels[key] = label_value(str(value))
    return labels


def batchsandbox_expire_time(spec: SandboxSpec) -> str | None:
    """Convert relative sandbox timeout into an absolute CR expireTime."""
    if spec.timeout_s is None:
        return None
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=spec.timeout_s)
    return expires_at.isoformat()


def batchsandbox_task_template(spec: SandboxSpec) -> dict[str, Any]:
    """Build the pooled BatchSandbox taskTemplate for an interactive sandbox."""
    entrypoint = spec.entrypoint or ["tail", "-f", "/dev/null"]
    escaped_entrypoint = " ".join(shlex.quote(arg) for arg in entrypoint)
    env_list = [{"name": key, "value": value} for key, value in spec.env.items()]
    return {
        "spec": {
            "process": {
                "command": [
                    "/bin/sh",
                    "-c",
                    f"exec /opt/opensandbox/bin/bootstrap.sh {escaped_entrypoint}",
                ],
                "env": env_list,
            }
        }
    }


def batchsandbox_template(
    spec: SandboxSpec,
    *,
    execd_image: str,
    provider_name: str,
) -> dict[str, Any]:
    """Build template-mode BatchSandbox pod template."""
    if spec.image is None:
        raise ValueError(
            "OpenSandbox BatchSandbox template mode requires SandboxSpec.image"
        )
    if spec.snapshot_id is not None:
        raise ValueError(
            "OpenSandbox BatchSandbox template mode does not support "
            "SandboxSpec.snapshot_id"
        )
    resources = {"limits": spec.resources, "requests": spec.resources}
    env = [
        {"name": key, "value": value}
        for key, value in {
            **spec.env,
            "EXECD": "/opt/opensandbox/bin/execd",
        }.items()
    ]
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "tolerations": CPU_POOL_TOLERATIONS,
        "affinity": CPU_POOL_AFFINITY,
        "initContainers": [
            {
                "name": "execd-installer",
                "image": execd_image,
                "command": ["/bin/sh", "-c"],
                "args": [
                    "cp ./execd /opt/opensandbox/bin/execd && "
                    "cp ./bootstrap.sh /opt/opensandbox/bin/bootstrap.sh && "
                    "chmod +x /opt/opensandbox/bin/execd && "
                    "chmod +x /opt/opensandbox/bin/bootstrap.sh"
                ],
                "volumeMounts": [
                    {
                        "name": "opensandbox-bin",
                        "mountPath": "/opt/opensandbox/bin",
                    }
                ],
            }
        ],
        "containers": [
            {
                "name": "sandbox",
                "image": spec.image,
                "command": ["/opt/opensandbox/bin/bootstrap.sh"]
                + (spec.entrypoint or ["tail", "-f", "/dev/null"]),
                "env": env,
                "volumeMounts": [
                    {
                        "name": "opensandbox-bin",
                        "mountPath": "/opt/opensandbox/bin",
                    }
                ],
            }
        ],
        "volumes": [{"name": "opensandbox-bin", "emptyDir": {}}],
    }
    if spec.resources:
        pod_spec["containers"][0]["resources"] = resources
    if spec.platform is not None:
        pod_spec["nodeSelector"] = {
            "kubernetes.io/os": spec.platform["os"],
            "kubernetes.io/arch": spec.platform["arch"],
        }
    return {
        "metadata": {
            "labels": batchsandbox_labels(spec, provider_name),
        },
        "spec": pod_spec,
    }


def build_batchsandbox_manifest(
    spec: SandboxSpec,
    *,
    name: str,
    namespace: str,
    count: int,
    execd_image: str,
    provider_name: str,
) -> dict[str, Any]:
    """Build one BatchSandbox CR for `count` equivalent sandboxes."""
    batch_spec: dict[str, Any] = {
        "replicas": count,
        "taskResourcePolicyWhenCompleted": "Retain",
    }
    if spec.extensions.get("poolRef"):
        if spec.platform is not None:
            raise ValueError(
                "OpenSandbox BatchSandbox pool mode does not support "
                "SandboxSpec.platform. Use an arch-specific poolRef."
            )
        if spec.volumes is not None:
            raise ValueError(
                "OpenSandbox BatchSandbox pool mode does not support volumes."
            )
        batch_spec["poolRef"] = spec.extensions["poolRef"]
        batch_spec["taskTemplate"] = batchsandbox_task_template(spec)
    else:
        batch_spec["template"] = batchsandbox_template(
            spec,
            execd_image=execd_image,
            provider_name=provider_name,
        )

    expire_time = batchsandbox_expire_time(spec)
    if expire_time is not None:
        batch_spec["expireTime"] = expire_time

    return {
        "apiVersion": f"{BATCHSANDBOX_GROUP}/{BATCHSANDBOX_VERSION}",
        "kind": "BatchSandbox",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": batchsandbox_labels(spec, provider_name),
        },
        "spec": batch_spec,
    }


def parse_batchsandbox_endpoints(workload: dict[str, Any]) -> list[str]:
    """Parse ordered endpoint IPs from the BatchSandbox endpoint annotation."""
    annotations = workload.get("metadata", {}).get("annotations", {}) or {}
    endpoints_json = annotations.get(BATCHSANDBOX_ENDPOINTS_ANNOTATION)
    if not endpoints_json:
        return []
    try:
        endpoints = json.loads(endpoints_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(endpoints, list):
        return []
    return [str(endpoint) for endpoint in endpoints if endpoint]


def batchsandbox_status_summary(workload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact status summary for readiness diagnostics."""
    status = workload.get("status", {}) or {}
    fields = (
        "phase",
        "replicas",
        "allocated",
        "ready",
        "taskRunning",
        "taskSucceed",
        "taskFailed",
        "taskUnknown",
        "taskPending",
    )
    summary = {field: status.get(field) for field in fields if field in status}
    conditions = status.get("conditions", [])
    if isinstance(conditions, list) and conditions:
        summary["conditions"] = [
            {
                "type": condition.get("type"),
                "status": condition.get("status"),
                "reason": condition.get("reason"),
                "message": condition.get("message"),
            }
            for condition in conditions[:3]
            if isinstance(condition, dict)
        ]
    return summary


def endpoint_with_port(endpoint: str, port: int) -> str:
    """Return `host:port`, stripping URL schemes if present."""
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        endpoint = endpoint.split("://", 1)[1]
    if ":" in endpoint:
        return endpoint
    return f"{endpoint}:{port}"


class BatchSandboxReplica:
    """SDK-shaped raw handle for one ready BatchSandbox replica endpoint."""

    def __init__(
        self,
        *,
        batch_name: str,
        replica_index: int,
        endpoint: str,
        connection_config: Any,
        delete_batch: Callable[[str], Awaitable[None]],
    ) -> None:
        CommandsAdapter, FilesystemAdapter, SandboxEndpoint = (
            require_opensandbox_endpoint_adapters()
        )
        self.id = f"{batch_name}-{replica_index}"
        self.batch_name = batch_name
        self.replica_index = replica_index
        self.endpoint = endpoint
        self._delete_batch = delete_batch
        sandbox_endpoint = SandboxEndpoint(endpoint=endpoint, headers={})
        self.commands = CommandsAdapter(connection_config, sandbox_endpoint)
        self.files = FilesystemAdapter(connection_config, sandbox_endpoint)

    async def kill(self) -> None:
        """Delete the owning BatchSandbox CR."""
        await self._delete_batch(self.batch_name)

    async def close(self) -> None:
        """Match the OpenSandbox SDK close interface."""
        return None


class BatchSandboxClient:
    """Small async wrapper around Kubernetes CustomObjectsApi."""

    def __init__(
        self,
        *,
        namespace: str,
        retries: int,
        retry_delay_s: float,
        retry_max_delay_s: float,
        poll_interval_s: float,
    ) -> None:
        self.namespace = namespace
        self._retries = retries
        self._retry_delay_s = retry_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._poll_interval_s = poll_interval_s
        self._api: Any | None = None

    def api(self) -> Any:
        """Return a cached CustomObjectsApi."""
        if self._api is None:
            load_kubernetes_config_once()
            client, _, _ = require_kubernetes_client()
            self._api = client.CustomObjectsApi()
        return self._api

    async def create(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Create one BatchSandbox CR with retries."""
        retry_policy = AsyncRetrying(
            retry=retry_if_exception(is_retryable_kubernetes_error),
            stop=stop_after_attempt(self._retries + 1),
            wait=wait_random_exponential(
                multiplier=self._retry_delay_s,
                max=self._retry_max_delay_s,
            ),
            before_sleep=log_retry,
            reraise=True,
        )
        async for attempt in retry_policy:
            with attempt:
                async with observability_span(
                    "batchsandbox.create_cr",
                    phase="startup",
                    attributes={
                        "namespace": self.namespace,
                        "batch_name": manifest.get("metadata", {}).get("name"),
                        "replicas": manifest.get("spec", {}).get("replicas"),
                    },
                ):
                    return await asyncio.to_thread(
                        self.api().create_namespaced_custom_object,
                        group=BATCHSANDBOX_GROUP,
                        version=BATCHSANDBOX_VERSION,
                        namespace=self.namespace,
                        plural=BATCHSANDBOX_PLURAL,
                        body=manifest,
                    )
        raise RuntimeError("BatchSandbox create retry loop did not run")

    async def get(self, name: str) -> dict[str, Any]:
        """Fetch one BatchSandbox CR."""
        return await asyncio.to_thread(
            self.api().get_namespaced_custom_object,
            group=BATCHSANDBOX_GROUP,
            version=BATCHSANDBOX_VERSION,
            namespace=self.namespace,
            plural=BATCHSANDBOX_PLURAL,
            name=name,
        )

    async def delete(self, name: str) -> None:
        """Delete one BatchSandbox CR, treating 404 as already cleaned."""
        try:
            async with observability_span(
                "batchsandbox.delete_cr",
                phase="cleanup",
                attributes={"namespace": self.namespace, "batch_name": name},
            ):
                await asyncio.to_thread(
                    self.api().delete_namespaced_custom_object,
                    group=BATCHSANDBOX_GROUP,
                    version=BATCHSANDBOX_VERSION,
                    namespace=self.namespace,
                    plural=BATCHSANDBOX_PLURAL,
                    name=name,
                    grace_period_seconds=0,
                )
        except Exception as e:
            _, _, ApiException = require_kubernetes_client()
            if not isinstance(e, ApiException) or e.status != 404:
                raise
            LOGGER.info("BatchSandbox %r was already deleted", name)

    async def wait_ready(
        self,
        name: str,
        *,
        count: int,
        timeout_s: float,
        progress_timeout_s: float | None = None,
        endpoint_only_ready_after_s: float | None = None,
        allow_partial: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        """Wait for replicas to be ready and return their endpoints.

        Strong readiness is preferred. When OpenSandbox has already published
        all endpoints but lags on status counters, callers may opt into an
        endpoint-only fallback. The provider still probes returned endpoints
        before treating those sandboxes as created.
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        last_progress_at = asyncio.get_running_loop().time()
        last_ready = 0
        last_endpoint_count = 0
        last_allocated = 0
        last_status_summary: dict[str, Any] = {}
        workload: dict[str, Any] | None = None
        last_recorded_state: tuple[int, int, int] | None = None
        endpoint_ready_at: float | None = None
        while True:
            workload = await self.get(name)
            status = workload.get("status", {}) or {}
            last_status_summary = batchsandbox_status_summary(workload)
            last_allocated = int(status.get("allocated", 0) or 0)
            last_ready = int(status.get("ready", 0) or 0)
            endpoints = parse_batchsandbox_endpoints(workload)
            last_endpoint_count = len(endpoints)
            state = (last_allocated, last_ready, last_endpoint_count)
            if state != last_recorded_state:
                last_progress_at = asyncio.get_running_loop().time()
                record_event(
                    "sample",
                    "batchsandbox.readiness",
                    attributes={
                        "namespace": self.namespace,
                        "batch_name": name,
                        "requested": count,
                        "allocated": last_allocated,
                        "ready": last_ready,
                        "endpoints": last_endpoint_count,
                        "phase": status.get("phase"),
                    },
                )
                last_recorded_state = state
            if (
                last_allocated >= count
                and last_ready >= count
                and last_endpoint_count >= count
            ):
                return workload, endpoints[:count]

            now = asyncio.get_running_loop().time()
            if last_endpoint_count >= count:
                if endpoint_ready_at is None:
                    endpoint_ready_at = now
                    record_event(
                        "sample",
                        "batchsandbox.endpoint_ready",
                        attributes={
                            "namespace": self.namespace,
                            "batch_name": name,
                            "requested": count,
                            "allocated": last_allocated,
                            "ready": last_ready,
                            "endpoints": last_endpoint_count,
                            "phase": status.get("phase"),
                        },
                    )
                if (
                    endpoint_only_ready_after_s is not None
                    and now - endpoint_ready_at >= endpoint_only_ready_after_s
                ):
                    LOGGER.warning(
                        "Returning BatchSandbox endpoints before strong status "
                        "readiness after %.3fs; name=%s, requested=%s, "
                        "allocated=%s, ready=%s, endpoints=%s",
                        now - endpoint_ready_at,
                        name,
                        count,
                        last_allocated,
                        last_ready,
                        last_endpoint_count,
                    )
                    record_event(
                        "sample",
                        "batchsandbox.endpoint_ready_fallback",
                        attributes={
                            "namespace": self.namespace,
                            "batch_name": name,
                            "requested": count,
                            "allocated": last_allocated,
                            "ready": last_ready,
                            "endpoints": last_endpoint_count,
                            "endpoint_ready_wait_s": now - endpoint_ready_at,
                            "phase": status.get("phase"),
                        },
                    )
                    return workload, endpoints[:count]
            else:
                endpoint_ready_at = None

            if (
                progress_timeout_s is not None
                and endpoint_ready_at is None
                and now - last_progress_at >= progress_timeout_s
            ):
                raise TimeoutError(
                    "Timed out waiting for BatchSandbox readiness progress after "
                    f"{progress_timeout_s:g}s; name={name!r}, requested={count}, "
                    f"allocated={last_allocated}, ready={last_ready}, "
                    f"endpoints={last_endpoint_count}, "
                    f"status={json.dumps(last_status_summary, sort_keys=True)}"
                )

            if now >= deadline:
                if allow_partial and workload is not None:
                    partial_count = min(
                        count,
                        last_allocated,
                        last_ready,
                        last_endpoint_count,
                    )
                    if partial_count > 0:
                        LOGGER.warning(
                            "Returning partial BatchSandbox readiness after "
                            "timeout; name=%s, requested=%s, ready=%s, "
                            "endpoints=%s, allocated=%s",
                            name,
                            count,
                            last_ready,
                            last_endpoint_count,
                            last_allocated,
                        )
                        return workload, endpoints[:partial_count]
                raise TimeoutError(
                    "Timed out waiting for BatchSandbox readiness after "
                    f"{timeout_s:g}s; name={name!r}, requested={count}, "
                    f"allocated={last_allocated}, ready={last_ready}, "
                    f"endpoints={last_endpoint_count}, "
                    f"status={json.dumps(last_status_summary, sort_keys=True)}"
                )
            await asyncio.sleep(self._poll_interval_s)
