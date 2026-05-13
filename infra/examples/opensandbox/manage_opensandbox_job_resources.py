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

"""OpenSandbox Pool lifecycle helpers for eval jobs."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DOMAIN = "opensandbox-server.opensandbox-system.svc.cluster.local"
DEFAULT_NAMESPACE = "opensandbox"


class OpenSandboxHttpError(RuntimeError):
    """HTTP error raised by the OpenSandbox management API."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"OpenSandbox API returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _base_url(domain: str) -> str:
    if domain.startswith(("http://", "https://")):
        base = domain.rstrip("/")
    else:
        base = f"http://{domain.strip('/')}"
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _request_json(
    method: str,
    domain: str,
    path: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any] | None:
    url = f"{_base_url(domain)}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["OPEN-SANDBOX-API-KEY"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise OpenSandboxHttpError(exc.code, body) from exc


def _load_pool_manifests(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read Pool manifests. Add `--with pyyaml` "
            "to the job's uv run invocation."
        ) from exc

    documents = list(yaml.safe_load_all(path.read_text()))
    pools = [doc for doc in documents if doc and doc.get("kind") == "Pool"]
    if not pools:
        raise ValueError(f"Expected at least one Pool manifest in {path}, got 0")
    return pools


def _capacity_from_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    capacity = dict(manifest.get("spec", {}).get("capacitySpec", {}) or {})
    required = {"bufferMax", "bufferMin", "poolMax", "poolMin"}
    missing = required - set(capacity)
    if missing:
        raise ValueError(f"Pool manifest is missing capacitySpec keys: {sorted(missing)}")
    return {key: int(capacity[key]) for key in required}


def _scaled_capacity(
    capacity: dict[str, int],
    *,
    concurrency: int | None,
    buffer_extra: int,
    pool_max_extra: int,
    allocated_headroom: int,
) -> dict[str, int]:
    if concurrency is None:
        return capacity
    if concurrency < 0:
        raise ValueError("--scale-for-concurrency must be >= 0")
    if allocated_headroom < 0:
        raise ValueError("allocated_headroom must be >= 0")
    buffer_target = concurrency
    buffer_max_target = concurrency + buffer_extra
    pool_max_target = max(
        buffer_max_target,
        allocated_headroom + concurrency + pool_max_extra,
    )
    return {
        "bufferMax": max(buffer_max_target, capacity["bufferMax"]),
        "bufferMin": max(buffer_target, capacity["bufferMin"]),
        "poolMax": max(pool_max_target, capacity["poolMax"]),
        "poolMin": capacity["poolMin"],
    }


def _pool_payload(
    manifest: dict[str, Any],
    *,
    pool_name: str | None,
    capacity: dict[str, int],
) -> dict[str, Any]:
    metadata = manifest.get("metadata", {}) or {}
    spec = manifest.get("spec", {}) or {}
    name = pool_name or metadata.get("name")
    if not name:
        raise ValueError("Pool manifest must include metadata.name or --pool-name")
    template = spec.get("template")
    if not template:
        raise ValueError("Pool manifest must include spec.template")
    return {
        "name": name,
        "template": template,
        "capacitySpec": capacity,
    }


def _status_available(pool: dict[str, Any] | None) -> int:
    if not pool:
        return 0
    status = pool.get("status") or {}
    return int(status.get("available") or 0)


def _status_allocated(pool: dict[str, Any] | None) -> int:
    if not pool:
        return 0
    status = pool.get("status") or {}
    return int(status.get("allocated") or 0)


def _wait_pool_available(
    *,
    domain: str,
    api_key: str | None,
    pool_name: str,
    min_available: int,
    timeout_s: float,
    poll_s: float,
    request_timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_pool: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_pool = _request_json(
            "GET",
            domain,
            f"/pools/{urllib.parse.quote(pool_name)}",
            api_key=api_key,
            timeout_s=request_timeout_s,
        )
        available = _status_available(last_pool)
        status = (last_pool or {}).get("status") or {}
        print(
            json.dumps(
                {
                    "event": "pool.wait",
                    "pool": pool_name,
                    "available": available,
                    "allocated": status.get("allocated"),
                    "total": status.get("total"),
                    "min_available": min_available,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if available >= min_available:
            return last_pool or {}
        time.sleep(poll_s)
    raise TimeoutError(
        f"Pool {pool_name!r} did not reach available>={min_available} "
        f"within {timeout_s:g}s; last_status={(last_pool or {}).get('status')!r}"
    )


def _create_or_update_pool(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    base_capacity = _capacity_from_manifest(manifest)
    capacity = _scaled_capacity(
        base_capacity,
        concurrency=args.scale_for_concurrency,
        buffer_extra=args.scale_buffer_extra,
        pool_max_extra=args.scale_pool_max_extra,
        allocated_headroom=0,
    )
    payload = _pool_payload(manifest, pool_name=args.pool_name, capacity=capacity)
    pool_name = payload["name"]

    created = False
    try:
        pool = _request_json(
            "POST",
            args.domain,
            "/pools",
            api_key=args.api_key,
            payload=payload,
            timeout_s=args.request_timeout_s,
        )
        created = True
    except OpenSandboxHttpError as exc:
        if exc.status_code != 409:
            raise
        pool = _request_json(
            "GET",
            args.domain,
            f"/pools/{urllib.parse.quote(pool_name)}",
            api_key=args.api_key,
            timeout_s=args.request_timeout_s,
        )
        if not args.skip_update_existing:
            allocated_headroom = (
                _status_allocated(pool)
                if args.preserve_existing_allocated
                else 0
            )
            capacity = _scaled_capacity(
                base_capacity,
                concurrency=args.scale_for_concurrency,
                buffer_extra=args.scale_buffer_extra,
                pool_max_extra=args.scale_pool_max_extra,
                allocated_headroom=allocated_headroom,
            )
            pool = _request_json(
                "PUT",
                args.domain,
                f"/pools/{urllib.parse.quote(pool_name)}",
                api_key=args.api_key,
                payload={"capacitySpec": capacity},
                timeout_s=args.request_timeout_s,
            )

    print(
        json.dumps(
            {
                "event": "pool.created",
                "pool": pool_name,
                "created": created,
                "capacitySpec": capacity,
                "preservedAllocated": (
                    _status_allocated(pool)
                    if args.preserve_existing_allocated
                    else 0
                ),
                "status": (pool or {}).get("status"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "pool": pool_name,
        "created": created,
        "capacitySpec": capacity,
    }


def _wait_one_pool(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    pool_name = result["pool"]
    capacity = result["capacitySpec"]
    min_available = args.wait_available
    if min_available is None:
        min_available = capacity["bufferMin"]
    final_pool = _wait_pool_available(
        domain=args.domain,
        api_key=args.api_key,
        pool_name=pool_name,
        min_available=min_available,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        request_timeout_s=args.request_timeout_s,
    )
    print(
        json.dumps(
            {
                "event": "pool.ready",
                "pool": pool_name,
                "created": result["created"],
                "capacitySpec": capacity,
                "status": final_pool.get("status"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "pool": pool_name,
        "created": result["created"],
        "capacitySpec": capacity,
        "status": final_pool.get("status"),
    }


def _pool_list_statuses(
    *,
    domain: str,
    api_key: str | None,
    request_timeout_s: float,
) -> dict[str, dict[str, Any]]:
    response = _request_json(
        "GET",
        domain,
        "/pools",
        api_key=api_key,
        timeout_s=request_timeout_s,
    )
    return {
        str(pool.get("name")): pool
        for pool in (response or {}).get("items", [])
        if pool.get("name")
    }


def _wait_pools_available(
    args: argparse.Namespace,
    created_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    min_available_by_name = {}
    for result in created_results:
        min_available = args.wait_available
        if min_available is None:
            min_available = result["capacitySpec"]["bufferMin"]
        min_available_by_name[result["pool"]] = int(min_available)

    deadline = time.monotonic() + args.timeout_s
    last_statuses: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        last_statuses = _pool_list_statuses(
            domain=args.domain,
            api_key=args.api_key,
            request_timeout_s=args.request_timeout_s,
        )
        pending: list[dict[str, Any]] = []
        for pool_name, min_available in min_available_by_name.items():
            pool = last_statuses.get(pool_name)
            available = _status_available(pool)
            if available < min_available:
                status = (pool or {}).get("status") or {}
                pending.append(
                    {
                        "pool": pool_name,
                        "available": available,
                        "allocated": status.get("allocated"),
                        "total": status.get("total"),
                        "min_available": min_available,
                    }
                )

        print(
            json.dumps(
                {
                    "event": "pools.wait",
                    "ready": len(min_available_by_name) - len(pending),
                    "pending": len(pending),
                    "count": len(min_available_by_name),
                    "pending_sample": pending[:10],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not pending:
            results = []
            for result in created_results:
                pool_name = result["pool"]
                final_pool = last_statuses.get(pool_name) or {}
                print(
                    json.dumps(
                        {
                            "event": "pool.ready",
                            "pool": pool_name,
                            "created": result["created"],
                            "capacitySpec": result["capacitySpec"],
                            "status": final_pool.get("status"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                results.append(
                    {
                        "pool": pool_name,
                        "created": result["created"],
                        "capacitySpec": result["capacitySpec"],
                        "status": final_pool.get("status"),
                    }
                )
            return results
        time.sleep(args.poll_s)

    pending_summary = []
    for pool_name, min_available in min_available_by_name.items():
        pool = last_statuses.get(pool_name)
        available = _status_available(pool)
        if available < min_available:
            pending_summary.append(
                {
                    "pool": pool_name,
                    "available": available,
                    "min_available": min_available,
                    "status": (pool or {}).get("status"),
                }
            )
    raise TimeoutError(
        "Pools did not reach requested availability within "
        f"{args.timeout_s:g}s; pending={pending_summary[:20]!r}"
    )


def _map_with_concurrency(
    func: Any,
    values: Sequence[Any],
    *,
    concurrency: int,
) -> list[Any]:
    if concurrency <= 1 or len(values) <= 1:
        return [func(value) for value in values]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(values))) as executor:
        return list(executor.map(func, values))


def _write_pool_names(path: str | None, pool_names: Sequence[str]) -> None:
    if path is None:
        return
    names_out = Path(path)
    names_out.parent.mkdir(parents=True, exist_ok=True)
    existing = names_out.read_text().splitlines() if names_out.exists() else []
    seen = set(existing)
    with names_out.open("a") as f:
        for pool_name in pool_names:
            if pool_name and pool_name not in seen:
                seen.add(pool_name)
                f.write(f"{pool_name}\n")


def _read_pool_name_filter(path: str | None) -> set[str] | None:
    if path is None:
        return None
    names_path = Path(path)
    if not names_path.exists():
        raise FileNotFoundError(f"Pool filter file does not exist: {names_path}")
    names = {
        line.strip()
        for line in names_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"Pool filter file is empty: {names_path}")
    return names


def _filter_pool_manifests(
    manifests: list[dict[str, Any]],
    pool_names: set[str] | None,
) -> list[dict[str, Any]]:
    if pool_names is None:
        return manifests
    filtered = [
        manifest
        for manifest in manifests
        if (manifest.get("metadata", {}) or {}).get("name") in pool_names
    ]
    if not filtered:
        raise ValueError(
            "No Pool manifests matched --pool-name-file entries; "
            f"requested={sorted(pool_names)}"
        )
    return filtered


def ensure_pool(args: argparse.Namespace) -> int:
    manifests = _load_pool_manifests(Path(args.manifest))
    manifests = _filter_pool_manifests(
        manifests,
        _read_pool_name_filter(args.pool_name_file),
    )
    if args.pool_name and len(manifests) != 1:
        raise ValueError("--pool-name can only be used with one Pool manifest")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    created_results = _map_with_concurrency(
        lambda manifest: _create_or_update_pool(args, manifest),
        manifests,
        concurrency=args.concurrency,
    )
    _write_pool_names(args.names_out, [result["pool"] for result in created_results])
    if len(created_results) == 1:
        results = [_wait_one_pool(args, created_results[0])]
    else:
        results = _wait_pools_available(args, created_results)
    print(
        json.dumps(
            {
                "event": "pools.ready",
                "manifest": args.manifest,
                "count": len(results),
                "pools": [result["pool"] for result in results],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _pool_names(args: argparse.Namespace) -> list[str]:
    names = list(args.pool_name or [])
    if args.pool_names_file:
        path = Path(args.pool_names_file)
        if path.exists():
            names.extend(line.strip() for line in path.read_text().splitlines())
    deduped: list[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def cleanup(args: argparse.Namespace) -> int:
    pool_results: list[dict[str, Any]] = []
    if not args.keep_pools:
        for pool_name in _pool_names(args):
            try:
                _request_json(
                    "DELETE",
                    args.domain,
                    f"/pools/{urllib.parse.quote(pool_name)}",
                    api_key=args.api_key,
                    timeout_s=args.request_timeout_s,
                )
                pool_results.append({"pool": pool_name, "deleted": True})
            except OpenSandboxHttpError as exc:
                if exc.status_code == 404:
                    pool_results.append({"pool": pool_name, "deleted": False, "missing": True})
                else:
                    raise
    print(
        json.dumps(
            {
                "event": "opensandbox.cleanup",
                "run_id": args.run_id,
                "pools": pool_results,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _add_common_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domain",
        default=os.getenv("OPENSANDBOX_DOMAIN", DEFAULT_DOMAIN),
        help="OpenSandbox server domain, with or without scheme.",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENSANDBOX_API_KEY"))
    parser.add_argument("--request-timeout-s", type=float, default=60.0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pool_parser = subparsers.add_parser("ensure-pool")
    _add_common_api_args(pool_parser)
    pool_parser.add_argument("--manifest", required=True)
    pool_parser.add_argument("--pool-name")
    pool_parser.add_argument(
        "--pool-name-file",
        help="Optional newline-delimited list of Pool names to select from --manifest.",
    )
    pool_parser.add_argument("--scale-for-concurrency", type=int)
    pool_parser.add_argument(
        "--scale-buffer-extra",
        type=int,
        default=0,
        help=(
            "Extra warm idle slots above --scale-for-concurrency. Defaults to "
            "0 so eval/demo warmup does not over-provision helper image pulls."
        ),
    )
    pool_parser.add_argument(
        "--scale-pool-max-extra",
        type=int,
        default=0,
        help=(
            "Extra Pool capacity above --scale-for-concurrency. Defaults to 0; "
            "raise this only for resilience experiments that need burst room."
        ),
    )
    pool_parser.add_argument(
        "--preserve-existing-allocated",
        action="store_true",
        help=(
            "When updating an existing Pool, size poolMax for currently "
            "allocated sandboxes plus --scale-for-concurrency. This prevents "
            "overlapping eval waves from shrinking away the next wave's idle "
            "headroom."
        ),
    )
    pool_parser.add_argument("--wait-available", type=int)
    pool_parser.add_argument("--timeout-s", type=float, default=900.0)
    pool_parser.add_argument("--poll-s", type=float, default=5.0)
    pool_parser.add_argument("--concurrency", type=int, default=16)
    pool_parser.add_argument("--names-out")
    pool_parser.add_argument("--skip-update-existing", action="store_true")
    pool_parser.set_defaults(func=ensure_pool)

    cleanup_parser = subparsers.add_parser("cleanup")
    _add_common_api_args(cleanup_parser)
    cleanup_parser.add_argument("--run-id", required=True)
    cleanup_parser.add_argument("--pool-name", action="append")
    cleanup_parser.add_argument("--pool-names-file")
    cleanup_parser.add_argument("--keep-pools", action="store_true")
    cleanup_parser.set_defaults(func=cleanup)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
