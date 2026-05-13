# Gym OpenSandbox Eval Launchers

This directory contains Kubernetes launch assets for running Gym eval jobs
directly from a Gym checkout. The jobs do not require a NeMo-RL checkout; helper
scripts, OpenSandbox pool management, prewarm, rollout verification, and summary
tools live here.

## Prerequisites

- A Kubernetes namespace that can reach `opensandbox-server.opensandbox-system.svc.cluster.local`.
- A secret named `opensandbox-api-key` with key `api-key`.
- A PVC named `rl-workspace`, or edit the manifests to use your workspace mount.
- A mounted Gym checkout at `/mnt/rl-workspace/gym/deps/Gym`.
- Mounted Harbor and OpenSandbox Python checkouts at:
  - `/mnt/rl-workspace/gym/deps/harbor`
  - `/mnt/rl-workspace/gym/deps/opensandbox-python`
- Harbor task directories under `/mnt/rl-workspace/gym-eval/data`.
- An OpenAI-compatible model endpoint, defaulting to
  `http://vllm.default.svc.cluster.local:8000/v1`.

All of these paths can be overridden through environment variables in the job
manifest: `GYM_DIR`, `HARBOR_DIR`, `OPENSANDBOX_PYTHON_DIR`,
`HARBOR_TASKS_DIR`, `POLICY_BASE_URL`, and `POLICY_MODEL_NAME`.

## Launch SWE-bench Smoke Eval

```bash
kubectl create -f infra/examples/opensandbox/gym-harbor-mini-swe-eval-job.yaml
```

The default manifest runs a two-task smoke eval with:

- `NEMO_GYM_RAY_ENABLED=false`, so the coordinator can run without Ray startup.
- Direct model routing through `POLICY_BASE_URL`.
- OpenSandbox SDK/server-proxy sandbox routing through the unified Gym sandbox API.
- Client-side Gym sandbox prewarm enabled with `fail_fast` acquire policy.
- W&B disabled by default; set `NEMO_RL_SANDBOX_OBSERVABILITY_WANDB=1` and
  provide an optional `wandb-api-key` secret to upload observability data.

To scale beyond smoke, patch `TASK_LIMIT`, `CONCURRENCY`,
`CREATE_CONCURRENCY`, and the `GYM_SANDBOX_POOL_PREWARM_*` knobs.

## Optional OpenSandbox Pool CR Warm Starts

The eval job can run with client-side prewarm alone, which creates
SDK-backed handles before rollout collection. For exact per-task OpenSandbox
Pool CR warm starts, set:

```yaml
- name: OPENSANDBOX_GENERATE_HARBOR_TASK_POOLS
  value: "1"
- name: OPENSANDBOX_GENERATED_POOL_PREFIX
  value: swefastghcr
```

The job will generate task-specific Pool manifests from `HARBOR_TASKS_DIR`,
ensure them through the OpenSandbox API, and route sandboxes with
`swefastghcr-{task_name}-pool`.

Static sample Pool manifests are included for smoke testing and cluster setup:

- `swe-agent-pool.yaml`
- `swe-agent-arch-pools.yaml`
- `swebench-astropy-pool.yaml`
- `tbench-adaptive-rejection-sampler-pool.yaml`

## Helper Scripts

- `prepare_gym_tbench_input.py`: builds Gym rollout JSONL input from Harbor task dirs.
- `prewarm_gym_harbor_sandboxes.py`: calls the Harbor agent prewarm and cleanup APIs.
- `generate_harbor_task_pools.py`: emits per-task OpenSandbox Pool CRs.
- `manage_opensandbox_job_resources.py`: ensures and cleans up OpenSandbox Pools.
- `verify_gym_rollouts.py`, `summarize_gym_rollouts.py`, and
  `summarize_harbor_trajectory_timing.py`: post-run validation and summaries.
