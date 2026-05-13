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

"""Typed configuration for sandbox providers and observability.

Defaults for these fields belong in YAML exemplars. Code should require keys
from enabled configs instead of silently supplying behavior here.
"""

from typing import Any, NotRequired, TypedDict


class SandboxProviderConfig(TypedDict):
    """Underlying runtime and infrastructure provider.

    Keys:
        name: Provider registry name, for example ``opensandbox``.
        kwargs: Provider-specific constructor settings such as OpenSandbox
            domain, API key, or proxy mode.
    """

    name: str
    kwargs: NotRequired[dict[str, Any]]


class SandboxObservabilityArtifactsConfig(TypedDict):
    """Portable report artifact settings for sandbox observability.

    Keys:
        enabled: Writes static report artifacts under
            ``observability/reports`` when true.
        render_html: Writes HTML index, aggregate, and trajectory pages.
        render_png: Writes PNG aggregate and per-trajectory timelines.
        export_otlp_json: Writes OpenTelemetry-shaped JSON and Chrome trace
            artifacts under ``observability/traces``.
    """

    enabled: bool
    render_html: bool
    render_png: bool
    export_otlp_json: bool


class SandboxObservabilityOtelConfig(TypedDict):
    """OpenTelemetry export settings for sandbox observability.

    Keys:
        enabled: Enables best-effort OpenTelemetry metric export.
        service_name: OpenTelemetry service name for the eval job.
        endpoint: OTLP HTTP metrics endpoint. Set to ``null`` to disable
            export without changing the rest of the observability config.
        export_logs: Reserved for structured log export; metric export is the
            first supported path.
    """

    enabled: bool
    service_name: str
    endpoint: str | None
    export_logs: bool


class SandboxObservabilityWandbConfig(TypedDict):
    """W&B artifact mirroring settings.

    Keys:
        enabled: ``true`` forces W&B artifact logging when a W&B run exists,
            ``false`` disables it, and ``null`` mirrors automatically when a
            W&B run is already configured.
        artifact_name: Artifact name used when uploading the observability
            directory to W&B.
        log_artifact: Uploads the observability directory as a W&B artifact.
        log_metrics: Logs summary scalars and histograms as native W&B metrics.
        log_time_series: Replays selected event/resource samples to W&B with
            elapsed wall-clock time as the x-axis.
        max_time_series_points: Maximum number of time-series points replayed
            to W&B. Use ``0`` for no sampling.
        log_reports: Logs rendered aggregate and trajectory PNGs as native
            W&B media when present.
        max_media_items: Maximum number of per-trajectory PNGs logged as
            native W&B media.
        log_tables: Logs sampled event, resource, and model-call rows as
            native W&B tables.
        max_table_rows: Maximum rows per native W&B table. Use ``0`` for no
            sampling.
        metric_prefix: Prefix used for native W&B metric names.
        project: Optional W&B project for forced observability-only runs.
        entity: Optional W&B entity for forced observability-only runs.
        run_name: Optional W&B run name for forced observability-only runs.
    """

    enabled: bool | None
    artifact_name: str
    log_artifact: bool
    log_metrics: bool
    log_time_series: bool
    max_time_series_points: int
    log_reports: bool
    max_media_items: int
    log_tables: bool
    max_table_rows: int
    metric_prefix: str
    project: NotRequired[str | None]
    entity: NotRequired[str | None]
    run_name: NotRequired[str | None]


class SandboxObservabilityPrivacyConfig(TypedDict):
    """Privacy controls for sandbox observability artifacts.

    Keys:
        include_command_text: Includes raw shell command text in artifacts when
            true. The recommended default is false; command hashes and
            categories are still recorded.
    """

    include_command_text: bool


class SandboxObservabilityProcessTraceConfig(TypedDict):
    """In-sandbox process trace settings.

    Keys:
        enabled: Enables periodic ``/proc`` snapshots inside each sandbox.
        sample_interval_s: Target seconds between process snapshots. The
            resource sampler runs at the smaller of this value and
            ``resource_sample_interval_s`` when enabled.
        max_processes_per_sample: Maximum number of process rows retained in
            each sample. Rows are ordered by RSS and CPU time.
        include_cmdline: Includes raw process command lines in artifacts when
            true. The recommended default is false.
    """

    enabled: bool
    sample_interval_s: float
    max_processes_per_sample: int
    include_cmdline: bool


class SandboxObservabilityConfig(TypedDict):
    """Sandbox eval observability configuration.

    Keys:
        enabled: Enables portable run artifacts and optional live sinks.
        output_dir: Directory for ``events.jsonl``, ``resource_samples.jsonl``,
            ``summary.json``, and static reports.
        resource_sample_interval_s: Seconds between best-effort sandbox CPU,
            memory, and process-count samples.
        max_rendered_trajectories: Maximum per-trajectory timelines rendered
            during finalization.
        artifacts: Static report artifact settings.
        otel: OpenTelemetry metric export settings.
        wandb: W&B artifact mirroring settings.
        process_trace: In-sandbox process trace settings.
        privacy: Redaction settings.
    """

    enabled: bool
    output_dir: str | None
    resource_sample_interval_s: float
    max_rendered_trajectories: int
    artifacts: SandboxObservabilityArtifactsConfig
    otel: SandboxObservabilityOtelConfig
    wandb: SandboxObservabilityWandbConfig
    process_trace: SandboxObservabilityProcessTraceConfig
    privacy: SandboxObservabilityPrivacyConfig
