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

"""Scale validation harness for OpenSandbox provider BatchSandbox mode."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec
from nemo_gym.sandbox.providers.opensandbox import OpenSandboxProvider
from nemo_gym.sandbox.providers.opensandbox.batchsandbox import (
    DEFAULT_EXECD_IMAGE,
    DEFAULT_EXECD_PORT,
)


@dataclass
class ScaleResult:
    run_id: str
    requested_replicas: int
    batchsandbox_namespace: str
    batchsandbox_execd_image: str
    batchsandbox_execd_port: int
    created_handles: int
    batch_names: list[str]
    creation_time_s: float | None
    progress_timeout_s: float | None
    strong_readiness_wait_s: float | None
    endpoint_count: int
    echo_success_count: int
    cleanup_time_s: float | None
    cleanup_error: str | None
    success: bool
    error: str | None
    started_at: str
    finished_at: str


def _parse_key_values(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected key=value, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key] = raw
    return parsed


def _raw_attr(raw: Any, name: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(name)
    return getattr(raw, name, None)


def _batch_names(handles: list[SandboxHandle]) -> list[str]:
    names = {
        str(name)
        for handle in handles
        if (name := _raw_attr(handle.raw, "batch_name")) is not None
    }
    return sorted(names)


def _raw_endpoint_count(handles: list[SandboxHandle]) -> int:
    return sum(
        1
        for handle in handles
        if _raw_attr(handle.raw, "endpoint") or _raw_attr(handle.raw, "execd_endpoint")
    )


def _load_k8s_client() -> Any | None:
    try:
        from kubernetes import client, config
    except ModuleNotFoundError:
        return None

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def _endpoint_count_from_annotation(batchsandbox: dict[str, Any]) -> int:
    annotations = batchsandbox.get("metadata", {}).get("annotations", {}) or {}
    raw = annotations.get("sandbox.opensandbox.io/endpoints")
    if not raw:
        return 0
    try:
        endpoints = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if isinstance(endpoints, list):
        return len([endpoint for endpoint in endpoints if endpoint])
    if isinstance(endpoints, dict):
        return len([endpoint for endpoint in endpoints.values() if endpoint])
    return 0


async def _wait_strong_readiness(
    *,
    namespace: str,
    batch_names: list[str],
    requested_replicas: int,
    timeout_s: float,
    poll_s: float,
) -> tuple[float | None, int]:
    """Wait for every owned BatchSandbox replica and endpoint to be ready."""
    if not batch_names:
        return 0.0, 0

    api = _load_k8s_client()
    if api is None:
        return None, 0

    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    last_endpoint_count = 0
    while time.monotonic() < deadline:
        ready_total = 0
        allocated_total = 0
        endpoint_total = 0
        for batch_name in batch_names:
            batchsandbox = await asyncio.to_thread(
                api.get_namespaced_custom_object,
                group="sandbox.opensandbox.io",
                version="v1alpha1",
                namespace=namespace,
                plural="batchsandboxes",
                name=batch_name,
            )
            status = batchsandbox.get("status", {}) or {}
            ready_total += int(status.get("ready", 0) or 0)
            allocated_total += int(status.get("allocated", 0) or 0)
            endpoint_total += _endpoint_count_from_annotation(batchsandbox)

        last_endpoint_count = endpoint_total
        if (
            allocated_total >= requested_replicas
            and ready_total >= requested_replicas
            and endpoint_total >= requested_replicas
        ):
            return time.monotonic() - started, endpoint_total
        await asyncio.sleep(poll_s)

    raise TimeoutError(
        "Timed out waiting for strong BatchSandbox readiness: "
        f"requested={requested_replicas}, endpoints={last_endpoint_count}"
    )


async def _verify_echoes(
    provider: OpenSandboxProvider,
    handles: list[SandboxHandle],
    *,
    concurrency: int,
    timeout_s: int,
) -> int:
    semaphore = asyncio.Semaphore(concurrency)

    async def _verify_one(index: int, handle: SandboxHandle) -> bool:
        token = f"nemo-batch-scale-{index}"
        async with semaphore:
            result = await provider.exec(
                handle,
                f"printf {token}",
                timeout_s=timeout_s,
                user="root",
            )
        return result.return_code == 0 and token in (result.stdout or "")

    results = await asyncio.gather(
        *(_verify_one(index, handle) for index, handle in enumerate(handles)),
        return_exceptions=True,
    )
    return sum(result is True for result in results)


async def _cleanup(provider: OpenSandboxProvider, handles: list[SandboxHandle]) -> float:
    started = time.monotonic()
    batch_names = _batch_names(handles)
    delete_batch = getattr(provider, "delete_batch", None)
    if batch_names and callable(delete_batch):
        await asyncio.gather(*(delete_batch(batch_name) for batch_name in batch_names))
    else:
        await asyncio.gather(
            *(provider.close(handle, delete=True) for handle in handles),
            return_exceptions=True,
        )
    return time.monotonic() - started


async def _run(args: argparse.Namespace) -> ScaleResult:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or f"batchsandbox-provider-scale-{uuid4().hex[:8]}"
    handles: list[SandboxHandle] = []
    creation_time_s: float | None = None
    strong_readiness_wait_s: float | None = None
    cleanup_time_s: float | None = None
    cleanup_error: str | None = None
    endpoint_count = 0
    echo_success_count = 0
    error: str | None = None

    provider: OpenSandboxProvider | None = None

    spec = SandboxSpec(
        image=args.image,
        timeout_s=args.sandbox_timeout_s,
        ready_timeout_s=args.sandbox_ready_timeout_s,
        resources=_parse_key_values(args.resource),
        entrypoint=["tail", "-f", "/dev/null"],
        metadata={
            "nemo-gym.nvidia.com/run-id": run_id,
            "nemo-gym.nvidia.com/harness": "batchsandbox-provider-scale",
            "nemo-gym.nvidia.com/owner": args.owner,
        },
        platform=(
            {"os": args.platform_os, "arch": args.platform_arch}
            if args.platform_arch
            else None
        ),
    )

    try:
        provider = OpenSandboxProvider(
            domain=args.domain,
            api_key=args.api_key,
            protocol=args.protocol,
            use_server_proxy=True,
            request_timeout_s=args.provider_request_timeout_s,
            create_request_timeout_s=args.provider_create_request_timeout_s,
            create_timeout_s=args.provider_create_timeout_s,
            create_probe_timeout_s=args.provider_create_probe_timeout_s,
            batch_mode=args.batch_mode,
            batch_create_concurrency=args.batch_create_concurrency,
            batch_create_retries=args.batch_create_retries,
            batch_create_retry_delay_s=args.batch_create_retry_delay_s,
            batch_create_retry_max_delay_s=args.batch_create_retry_max_delay_s,
            batchsandbox_namespace=args.opensandbox_namespace,
            batchsandbox_ready_timeout_s=args.strong_readiness_timeout_s,
            batchsandbox_progress_timeout_s=args.progress_timeout_s,
            batchsandbox_poll_interval_s=args.strong_readiness_poll_s,
            batchsandbox_execd_port=args.batchsandbox_execd_port,
            batchsandbox_execd_image=args.batchsandbox_execd_image,
        )
        create_started = time.monotonic()
        handles = await provider.create_batch(spec, args.replicas)
        creation_time_s = time.monotonic() - create_started
        batch_names = _batch_names(handles)

        strong_readiness_wait_s, endpoint_count = await _wait_strong_readiness(
            namespace=args.opensandbox_namespace,
            batch_names=batch_names,
            requested_replicas=args.replicas,
            timeout_s=args.strong_readiness_timeout_s,
            poll_s=args.strong_readiness_poll_s,
        )
        if endpoint_count == 0:
            endpoint_count = _raw_endpoint_count(handles) or len(handles)

        echo_success_count = await _verify_echoes(
            provider,
            handles,
            concurrency=args.echo_concurrency,
            timeout_s=args.echo_timeout_s,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if provider is not None:
            try:
                if handles:
                    cleanup_time_s = await _cleanup(provider, handles)
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
            await provider.aclose()

    finished_at = datetime.now(timezone.utc).isoformat()
    return ScaleResult(
        run_id=run_id,
        requested_replicas=args.replicas,
        batchsandbox_namespace=args.opensandbox_namespace,
        batchsandbox_execd_image=args.batchsandbox_execd_image,
        batchsandbox_execd_port=args.batchsandbox_execd_port,
        created_handles=len(handles),
        batch_names=_batch_names(handles),
        creation_time_s=creation_time_s,
        progress_timeout_s=args.progress_timeout_s,
        strong_readiness_wait_s=strong_readiness_wait_s,
        endpoint_count=endpoint_count,
        echo_success_count=echo_success_count,
        cleanup_time_s=cleanup_time_s,
        cleanup_error=cleanup_error,
        success=(
            error is None
            and cleanup_error is None
            and len(handles) == args.replicas
            and endpoint_count >= args.replicas
            and echo_success_count == args.replicas
        ),
        error=error,
        started_at=started_at,
        finished_at=finished_at,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicas",
        type=int,
        choices=[32, 64, 128, 512, 1024],
        required=True,
    )
    parser.add_argument("--batch-mode", default="batchsandbox")
    parser.add_argument("--run-id", default=os.getenv("JOB_NAME"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--domain",
        default="opensandbox-server.opensandbox-system.svc.cluster.local",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENSANDBOX_API_KEY"))
    parser.add_argument("--protocol", default="http")
    parser.add_argument("--owner", default="nemo-gym")
    parser.add_argument("--opensandbox-namespace", default="opensandbox")
    parser.add_argument("--batchsandbox-execd-image", default=DEFAULT_EXECD_IMAGE)
    parser.add_argument(
        "--batchsandbox-execd-port",
        type=int,
        default=DEFAULT_EXECD_PORT,
    )
    parser.add_argument(
        "--image",
        default="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=["cpu=100m", "memory=256Mi"],
    )
    parser.add_argument("--platform-os", default="linux")
    parser.add_argument("--platform-arch", default="amd64")
    parser.add_argument("--sandbox-timeout-s", type=int, default=2400)
    parser.add_argument("--sandbox-ready-timeout-s", type=int, default=900)
    parser.add_argument("--provider-request-timeout-s", type=int, default=900)
    parser.add_argument("--provider-create-request-timeout-s", type=int, default=900)
    parser.add_argument("--provider-create-timeout-s", type=float, default=1200)
    parser.add_argument("--provider-create-probe-timeout-s", type=int, default=30)
    parser.add_argument("--batch-create-concurrency", type=int, default=64)
    parser.add_argument("--batch-create-retries", type=int, default=2)
    parser.add_argument("--batch-create-retry-delay-s", type=float, default=5.0)
    parser.add_argument("--batch-create-retry-max-delay-s", type=float, default=45.0)
    parser.add_argument("--strong-readiness-timeout-s", type=float, default=1200)
    parser.add_argument(
        "--progress-timeout-s",
        type=float,
        default=300,
        help=(
            "Fail provider create if BatchSandbox allocated/ready/endpoint "
            "counts stop advancing for this many seconds."
        ),
    )
    parser.add_argument("--strong-readiness-poll-s", type=float, default=2.0)
    parser.add_argument("--echo-concurrency", type=int, default=128)
    parser.add_argument("--echo-timeout-s", type=int, default=30)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = asyncio.run(_run(args))
    payload = asdict(result)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
