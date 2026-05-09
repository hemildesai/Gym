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

"""Generate OpenSandbox Pool manifests for Harbor task directories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any


DNS_LABEL_RE = re.compile(r"[^a-z0-9-]+")
DEFAULT_NAMESPACE = "opensandbox"
DEFAULT_TASK_EXECUTOR_IMAGE = "mirror.gcr.io/opensandbox/task-executor:latest"
DEFAULT_EXECD_IMAGE = "mirror.gcr.io/opensandbox/execd:v1.0.15"
SWEBENCH_DOCKERHUB_PREFIX = "swebench/sweb.eval.x86_64."
SWEBENCH_EPOCH_GHCR_PREFIX = "ghcr.io/epoch-research/swe-bench.eval.x86_64."


def dns_label(value: Any, *, max_length: int = 63) -> str:
    """Return a stable Kubernetes DNS label."""
    sanitized = DNS_LABEL_RE.sub("-", str(value).lower()).strip("-")
    if not sanitized:
        return "resource"
    if len(sanitized) <= max_length:
        return sanitized

    digest = hashlib.sha1(sanitized.encode("utf-8")).hexdigest()[:8]
    prefix = sanitized[: max_length - len(digest) - 1].rstrip("-")
    return f"{prefix}-{digest}" if prefix else digest


def parse_rewrite(value: str) -> tuple[str, str]:
    """Parse an image rewrite in FROM=TO form."""
    source, sep, target = value.partition("=")
    if not sep or not source or not target:
        raise argparse.ArgumentTypeError(
            "image rewrites must use FROM=TO, for example "
            "alexgshaw/=mirror.gcr.io/alexgshaw/"
        )
    return source, target


def rewrite_image(image: str, rewrites: list[tuple[str, str]]) -> str:
    """Apply prefix rewrites to a task image."""
    for source, target in rewrites:
        if image.startswith(source):
            return target + image[len(source) :]
    return image


def swebench_epoch_ghcr_image(*, task_name: str, docker_image: str) -> str:
    """Return the Epoch GHCR mirror image for an official SWE-bench task."""
    if not docker_image.startswith(SWEBENCH_DOCKERHUB_PREFIX):
        raise ValueError(
            "SWE-bench Epoch GHCR rewrite only supports official SWE-bench "
            f"DockerHub images; got {docker_image!r} for task {task_name!r}"
        )
    return f"{SWEBENCH_EPOCH_GHCR_PREFIX}{task_name}:latest"


def task_repo_group(task_name: str) -> str:
    """Return a coarse repo/task-family grouping for reports."""
    return task_name.rsplit("-", 1)[0]


def cpu_to_millicores(value: str) -> int | None:
    """Parse a simple Kubernetes CPU quantity into millicores."""
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("m"):
        try:
            return int(normalized[:-1])
        except ValueError:
            return None
    try:
        return int(float(normalized) * 1000)
    except ValueError:
        return None


def cpu_limit_at_least_request(*, request_cpu: str, limit_cpu: str) -> str:
    """Raise the CPU limit when task metadata requests more than the default."""
    request_millicores = cpu_to_millicores(request_cpu)
    limit_millicores = cpu_to_millicores(limit_cpu)
    if request_millicores is None or limit_millicores is None:
        return limit_cpu
    if request_millicores <= limit_millicores:
        return limit_cpu
    return request_cpu


def task_environment(path: Path) -> dict[str, Any]:
    """Load the environment section from one Harbor task.toml."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError(f"{path} has non-dictionary [environment]")
    return environment


def pool_manifest(
    *,
    task_name: str,
    image: str,
    benchmark_label: str,
    pool_prefix: str,
    namespace: str,
    request_cpu: str,
    request_memory: str,
    request_storage: str,
    limit_cpu: str,
    limit_memory: str,
    limit_storage: str,
    pool_min: int,
    buffer_min: int,
    buffer_max: int,
    pool_max: int,
    task_executor_image: str,
    execd_image: str,
    arch: str,
    karpenter_instance_categories: list[str],
    karpenter_instance_generation: str,
    mini_swe_agent_init_install: bool,
    mini_swe_agent_package: str,
) -> dict[str, Any]:
    """Build one OpenSandbox Pool manifest."""
    task_label = dns_label(task_name)
    pool_name = dns_label(f"{pool_prefix}-{task_name}-pool")
    volumes = [
        {"name": "sandbox-storage", "emptyDir": {}},
        {"name": "opensandbox-bin", "emptyDir": {}},
        {"name": "sandbox-logs", "emptyDir": {}},
    ]
    init_containers = [
        {
            "name": "task-executor-installer",
            "image": task_executor_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/sh", "-c"],
            "args": [
                (
                    "cp /workspace/server "
                    "/opt/opensandbox/bin/task-executor\n"
                    "chmod +x "
                    "/opt/opensandbox/bin/task-executor\n"
                )
            ],
            "volumeMounts": [
                {
                    "name": "opensandbox-bin",
                    "mountPath": "/opt/opensandbox/bin",
                }
            ],
        },
        {
            "name": "execd-installer",
            "image": execd_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/sh", "-c"],
            "args": [
                (
                    "cp ./execd /opt/opensandbox/bin/execd\n"
                    "cp ./bootstrap.sh "
                    "/opt/opensandbox/bin/bootstrap.sh\n"
                    "chmod +x /opt/opensandbox/bin/execd\n"
                    "chmod +x "
                    "/opt/opensandbox/bin/bootstrap.sh\n"
                )
            ],
            "volumeMounts": [
                {
                    "name": "opensandbox-bin",
                    "mountPath": "/opt/opensandbox/bin",
                }
            ],
        },
    ]
    sandbox_volume_mounts = [
        {
            "name": "sandbox-storage",
            "mountPath": "/var/lib/sandbox",
        },
        {
            "name": "opensandbox-bin",
            "mountPath": "/opt/opensandbox/bin",
        },
        {
            "name": "sandbox-logs",
            "mountPath": "/workspace/logs",
        },
    ]
    if mini_swe_agent_init_install:
        volumes.append({"name": "mini-swe-agent-tools", "emptyDir": {}})
        init_containers.append(
            {
                "name": "mini-swe-agent-installer",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["/bin/sh", "-c"],
                "args": [
                    (
                        "set -e\n"
                        "export HOME=/root\n"
                        "mkdir -p \"$HOME/.local/bin\"\n"
                        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
                        "if ! command -v curl >/dev/null 2>&1; then\n"
                        "  if command -v apt-get >/dev/null 2>&1; then\n"
                        "    DEBIAN_FRONTEND=noninteractive apt-get update\n"
                        "    DEBIAN_FRONTEND=noninteractive apt-get install -y curl git\n"
                        "  else\n"
                        "    echo 'curl is required to install uv' >&2\n"
                        "    exit 1\n"
                        "  fi\n"
                        "fi\n"
                        "if ! command -v uv >/dev/null 2>&1; then\n"
                        "  curl -LsSf https://astral.sh/uv/0.7.13/install.sh | "
                        "UV_INSTALL_DIR=\"$HOME/.local/bin\" sh\n"
                        "fi\n"
                        "export PATH=\"$HOME/.local/bin:$PATH\"\n"
                        "command -v uv >/dev/null\n"
                        f"uv tool install {mini_swe_agent_package!r}\n"
                        "mini-swe-agent --help >/dev/null\n"
                    )
                ],
                "volumeMounts": [
                    {
                        "name": "mini-swe-agent-tools",
                        "mountPath": "/root/.local",
                    }
                ],
            }
        )
        sandbox_volume_mounts.append(
            {
                "name": "mini-swe-agent-tools",
                "mountPath": "/root/.local",
            }
        )
    return {
        "apiVersion": "sandbox.opensandbox.io/v1alpha1",
        "kind": "Pool",
        "metadata": {
            "name": pool_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "opensandbox",
                "app.kubernetes.io/component": pool_name,
                "sandbox.nemo-gym/benchmark": benchmark_label,
                "sandbox.nemo-gym/task": task_label,
            },
        },
        "spec": {
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "opensandbox",
                        "app.kubernetes.io/component": pool_name,
                        "sandbox.nemo-gym/benchmark": benchmark_label,
                        "sandbox.nemo-gym/task": task_label,
                    },
                    "annotations": {
                        "karpenter.sh/do-not-disrupt": "true",
                    },
                },
                "spec": {
                    "schedulerName": "default-scheduler",
                    "dnsPolicy": "ClusterFirst",
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "nodeGroup",
                                                "operator": "In",
                                                "values": ["customer-cpu"],
                                            },
                                            {
                                                "key": "kubernetes.io/arch",
                                                "operator": "In",
                                                "values": [arch],
                                            },
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
                                                "key": "kubernetes.io/arch",
                                                "operator": "In",
                                                "values": [arch],
                                            },
                                            {
                                                "key": (
                                                    "karpenter.k8s.aws/"
                                                    "instance-generation"
                                                ),
                                                "operator": "Gt",
                                                "values": [karpenter_instance_generation],
                                            },
                                            {
                                                "key": (
                                                    "karpenter.k8s.aws/"
                                                    "instance-category"
                                                ),
                                                "operator": "In",
                                                "values": karpenter_instance_categories,
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
                    },
                    "volumes": volumes,
                    "initContainers": init_containers,
                    "containers": [
                        {
                            "name": "sandbox",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-c"],
                            "args": [
                                (
                                    "/opt/opensandbox/bin/task-executor "
                                    "-listen-addr=0.0.0.0:5758 "
                                    ">/tmp/task-executor.log 2>&1"
                                )
                            ],
                            "env": [
                                {"name": "SANDBOX_MAIN_CONTAINER", "value": "main"},
                                {
                                    "name": "EXECD_ENVS",
                                    "value": "/opt/opensandbox/.env",
                                },
                                {
                                    "name": "EXECD",
                                    "value": "/opt/opensandbox/bin/execd",
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": request_cpu,
                                    "memory": request_memory,
                                    "ephemeral-storage": request_storage,
                                },
                                "limits": {
                                    "cpu": limit_cpu,
                                    "memory": limit_memory,
                                    "ephemeral-storage": limit_storage,
                                },
                            },
                            "volumeMounts": sandbox_volume_mounts,
                        }
                    ],
                },
            },
            "capacitySpec": {
                "bufferMax": buffer_max,
                "bufferMin": buffer_min,
                "poolMax": pool_max,
                "poolMin": pool_min,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--benchmark-label", required=True)
    parser.add_argument("--pool-prefix", required=True)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--image-rewrite", action="append", default=[], type=parse_rewrite)
    parser.add_argument(
        "--swebench-epoch-ghcr",
        action="store_true",
        help=(
            "Use public ghcr.io/epoch-research SWE-bench evaluation images for "
            "official SWE-bench task pools. This avoids DockerHub anonymous "
            "pull throttling for swebench/sweb.eval.x86_64.* images."
        ),
    )
    parser.add_argument("--task-executor-image", default=DEFAULT_TASK_EXECUTOR_IMAGE)
    parser.add_argument("--execd-image", default=DEFAULT_EXECD_IMAGE)
    parser.add_argument("--max-tasks", default=0, type=int)
    parser.add_argument("--default-cpu", default="1")
    parser.add_argument("--default-memory", default="4Gi")
    parser.add_argument("--default-storage", default="10Gi")
    parser.add_argument(
        "--override-cpu",
        help="Force CPU request for every generated Pool, ignoring task.toml.",
    )
    parser.add_argument(
        "--override-memory",
        help="Force memory request for every generated Pool, ignoring task.toml.",
    )
    parser.add_argument(
        "--override-storage",
        help="Force ephemeral-storage request for every generated Pool, ignoring task.toml.",
    )
    parser.add_argument("--limit-cpu", default="2")
    parser.add_argument("--limit-memory", default="8Gi")
    parser.add_argument("--limit-storage", default="20Gi")
    parser.add_argument("--pool-min", default=1, type=int)
    parser.add_argument("--buffer-min", default=0, type=int)
    parser.add_argument("--buffer-max", default=1, type=int)
    parser.add_argument("--pool-max", default=1, type=int)
    parser.add_argument(
        "--arch",
        default="amd64",
        choices=("amd64", "arm64"),
        help=(
            "Kubernetes architecture for generated sandbox pools. Official "
            "SWE-bench/TBench images are currently amd64; use arm64 only for "
            "multi-arch task images."
        ),
    )
    parser.add_argument(
        "--karpenter-instance-category",
        action="append",
        default=None,
        help=(
            "Allowed karpenter.k8s.aws/instance-category value. Repeat for "
            "multiple categories. Defaults to c, m, and r."
        ),
    )
    parser.add_argument(
        "--karpenter-instance-generation",
        default="5",
        help="Minimum Karpenter instance generation, used with operator Gt.",
    )
    parser.add_argument(
        "--mini-swe-agent-init-install",
        action="store_true",
        help=(
            "Add an initContainer that installs mini-swe-agent into a shared "
            "/root/.local volume before the sandbox becomes Pool-ready. Use "
            "for SWE task pools when demo latency should exclude per-sandbox "
            "CLI installation."
        ),
    )
    parser.add_argument(
        "--mini-swe-agent-package",
        default="mini-swe-agent==2.1.0",
        help="Package spec used by --mini-swe-agent-init-install.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate pool manifests and an image grouping report."""
    args = parse_args()
    task_tomls = sorted(args.tasks_dir.glob("*/task.toml"))
    if args.max_tasks > 0:
        task_tomls = task_tomls[: args.max_tasks]
    if not task_tomls:
        raise RuntimeError(f"No Harbor task.toml files found under {args.tasks_dir}")

    manifests = []
    rows = []
    karpenter_instance_categories = args.karpenter_instance_category or [
        "c",
        "m",
        "r",
    ]
    for task_toml in task_tomls:
        task_name = task_toml.parent.name
        environment = task_environment(task_toml)
        docker_image = environment.get("docker_image")
        if not isinstance(docker_image, str) or not docker_image:
            raise ValueError(f"{task_toml} does not define environment.docker_image")

        if args.swebench_epoch_ghcr:
            image = swebench_epoch_ghcr_image(
                task_name=task_name,
                docker_image=docker_image,
            )
        else:
            image = rewrite_image(docker_image, args.image_rewrite)
        request_cpu = str(
            args.override_cpu or environment.get("cpus") or args.default_cpu
        )
        request_memory = str(
            args.override_memory or environment.get("memory") or args.default_memory
        )
        request_storage = str(
            args.override_storage or environment.get("storage") or args.default_storage
        )
        limit_cpu = cpu_limit_at_least_request(
            request_cpu=request_cpu,
            limit_cpu=args.limit_cpu,
        )
        manifest = pool_manifest(
            task_name=task_name,
            image=image,
            benchmark_label=args.benchmark_label,
            pool_prefix=args.pool_prefix,
            namespace=args.namespace,
            request_cpu=request_cpu,
            request_memory=request_memory,
            request_storage=request_storage,
            limit_cpu=limit_cpu,
            limit_memory=args.limit_memory,
            limit_storage=args.limit_storage,
            pool_min=args.pool_min,
            buffer_min=args.buffer_min,
            buffer_max=args.buffer_max,
            pool_max=args.pool_max,
            task_executor_image=args.task_executor_image,
            execd_image=args.execd_image,
            arch=args.arch,
            karpenter_instance_categories=karpenter_instance_categories,
            karpenter_instance_generation=args.karpenter_instance_generation,
            mini_swe_agent_init_install=args.mini_swe_agent_init_install,
            mini_swe_agent_package=args.mini_swe_agent_package,
        )
        manifests.append(manifest)
        rows.append(
            {
                "task_name": task_name,
                "repo_group": task_repo_group(task_name),
                "pool": manifest["metadata"]["name"],
                "docker_image": docker_image,
                "pool_image": image,
                "request_cpu": request_cpu,
                "request_memory": request_memory,
                "request_storage": request_storage,
                "limit_cpu": limit_cpu,
                "task_executor_image": args.task_executor_image,
                "execd_image": args.execd_image,
                "arch": args.arch,
                "karpenter_instance_categories": karpenter_instance_categories,
                "karpenter_instance_generation": args.karpenter_instance_generation,
                "mini_swe_agent_init_install": args.mini_swe_agent_init_install,
                "mini_swe_agent_package": args.mini_swe_agent_package,
            }
        )

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required. Run with `uv run --with pyyaml`."
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump_all(manifests, sort_keys=False))

    images_by_repo_group: dict[str, set[str]] = {}
    for row in rows:
        images_by_repo_group.setdefault(row["repo_group"], set()).add(
            row["docker_image"]
        )
    repo_groups_with_multiple_images = {
        repo_group: len(images)
        for repo_group, images in sorted(images_by_repo_group.items())
        if len(images) > 1
    }

    report = {
        "task_count": len(rows),
        "pool_count": len(manifests),
        "pool_ref_template": f"{args.pool_prefix}-{{task_name}}-pool",
        "unique_images": len({row["docker_image"] for row in rows}),
        "unique_pool_images": len({row["pool_image"] for row in rows}),
        "repo_groups": Counter(row["repo_group"] for row in rows).most_common(),
        "repo_groups_with_multiple_images": repo_groups_with_multiple_images,
        "repo_level_pool_exact_safe": not repo_groups_with_multiple_images,
        "resource_request_overrides": {
            "cpu": args.override_cpu,
            "memory": args.override_memory,
            "storage": args.override_storage,
        },
        "resource_requests": {
            "cpu": Counter(row["request_cpu"] for row in rows).most_common(),
            "memory": Counter(row["request_memory"] for row in rows).most_common(),
            "storage": Counter(row["request_storage"] for row in rows).most_common(),
        },
        "helper_images": {
            "task_executor": args.task_executor_image,
            "execd": args.execd_image,
        },
        "image_transform": {
            "image_rewrites": [
                {"source": source, "target": target}
                for source, target in args.image_rewrite
            ],
            "swebench_epoch_ghcr": args.swebench_epoch_ghcr,
        },
        "scheduling": {
            "arch": args.arch,
            "karpenter_instance_categories": karpenter_instance_categories,
            "karpenter_instance_generation_gt": args.karpenter_instance_generation,
        },
        "sample": rows[:20],
    }
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
