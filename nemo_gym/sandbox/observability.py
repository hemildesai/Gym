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

"""Optional OpenTelemetry tracing + per-sandbox resource metrics.

This module is fully env-gated and fail-safe: if OpenTelemetry is not
installed or tracing is disabled, every entry point degrades to a no-op so it
can never break an eval run.

Two observability features are implemented here:

1. Tool-call OTel spans. ``tool_call_span`` opens a span named
   ``sandbox.exec`` (the tool-call granularity) per sandbox command. The span
   is keyed by ``instance_id`` / ``sandbox_id`` so a trace can be assembled per
   trajectory in the collector (Jaeger / Tempo / OTLP).

2. Per-sandbox CPU + memory during a tool call. The provider brackets each
   exec with a cgroup v2 leaf read (cpu.stat usage_usec + memory.peak /
   memory.current, before+after) and calls ``record_tool_call_resources`` to
   attach the deltas (``tool_cpu_seconds`` + ``tool_mem_peak_bytes``) onto the
   active tool-call span and a bounded metric keyed by instance/sandbox.

Environment variables (read once at import-time setup):

* ``NG_OBS_EXPORT_TRACES``     -- "1"/"true" to enable the OTLP span exporter.
* ``NG_OBS_TRACES_EXPORTER``   -- "otlp_http" (default) or "otlp_grpc".
* ``NG_OBS_TRACES_ENDPOINT``   -- collector endpoint
                                  (e.g. http://jaeger.default.svc:4318/v1/traces).
* ``NG_OBS_SERVICE_NAME``      -- service.name resource attribute.
* ``NG_OBS_ENDPOINT_LABEL``    -- free-form label added as a span attribute.
* ``NG_OBS_SANDBOX_CGROUP_BRACKET`` -- "1" to enable the cgroup bracket.
* ``NG_OBS_SANDBOX_CGROUP_PATH``    -- cgroup v2 leaf mount (default
                                       /sys/fs/cgroup).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator


LOGGER = logging.getLogger(__name__)


# --- env helpers -----------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def traces_enabled() -> bool:
    """Whether the OTLP span exporter should be active."""
    return _env_flag("NG_OBS_EXPORT_TRACES", False)


def cgroup_bracket_enabled() -> bool:
    """Whether the provider should bracket execs with a cgroup read."""
    return _env_flag("NG_OBS_SANDBOX_CGROUP_BRACKET", False)


def cgroup_path() -> str:
    """cgroup v2 leaf mount inside the sandbox."""
    return os.environ.get("NG_OBS_SANDBOX_CGROUP_PATH", "/sys/fs/cgroup").rstrip("/")


# --- lazy OTel tracer + metric setup ---------------------------------------

_TRACER: Any | None = None
_CPU_HISTOGRAM: Any | None = None
_MEM_HISTOGRAM: Any | None = None
_SETUP_DONE = False
_ENDPOINT_LABEL = ""


def _setup() -> None:
    """Initialise the OTel tracer + metrics once (best-effort)."""
    global _TRACER, _CPU_HISTOGRAM, _MEM_HISTOGRAM, _SETUP_DONE, _ENDPOINT_LABEL
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    if not traces_enabled():
        return

    _ENDPOINT_LABEL = os.environ.get("NG_OBS_ENDPOINT_LABEL", "")
    service_name = os.environ.get("NG_OBS_SERVICE_NAME", "nemo-gym-sandbox")
    endpoint = os.environ.get("NG_OBS_TRACES_ENDPOINT")
    exporter_kind = os.environ.get("NG_OBS_TRACES_EXPORTER", "otlp_http").strip().lower()

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "nemo-gym",
            }
        )

        # A TracerProvider may already exist (set by the driver). Only install
        # our own if none is configured, otherwise reuse the global one so our
        # spans flow through the existing pipeline.
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            provider = existing
        else:
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

        if exporter_kind == "otlp_grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            span_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            span_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        _TRACER = trace.get_tracer("nemo_gym.sandbox")
        LOGGER.info(
            "nemo_gym sandbox OTel tracing enabled: exporter=%s endpoint=%s service=%s",
            exporter_kind,
            endpoint,
            service_name,
        )
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("nemo_gym sandbox OTel tracing setup failed (%s); disabling traces", exc)
        _TRACER = None
        return

    # Bounded per-sandbox resource metrics. Cardinality is keyed by
    # instance_id/sandbox_id (~500 instances), not 30k.
    try:
        from opentelemetry import metrics

        meter = metrics.get_meter("nemo_gym.sandbox")
        _CPU_HISTOGRAM = meter.create_histogram(
            "nemo_gym.sandbox.tool_call.cpu_seconds",
            unit="s",
            description="CPU seconds consumed by one sandbox tool call (cgroup delta).",
        )
        _MEM_HISTOGRAM = meter.create_histogram(
            "nemo_gym.sandbox.tool_call.mem_peak_bytes",
            unit="By",
            description="Peak memory during one sandbox tool call (cgroup memory.peak).",
        )
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.debug("nemo_gym sandbox OTel metrics unavailable (%s)", exc)


def _tracer() -> Any | None:
    _setup()
    return _TRACER


# --- public API ------------------------------------------------------------


class _NoopSpan:
    """Stand-in span object when tracing is disabled."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@contextmanager
def tool_call_span(
    command_class: str,
    *,
    sandbox_id: str | None,
    instance_id: str | None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a tool-call (``sandbox.exec``) span.

    Yields a span-like object that always supports ``set_attribute``; callers
    use it to attach per-sandbox cgroup deltas after the command finishes.
    Degrades to a no-op span when tracing is disabled.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NoopSpan()
        return

    span_attrs: dict[str, Any] = {
        "sandbox.command_class": command_class,
        "sandbox.id": sandbox_id or "unknown",
        "sandbox.instance_id": instance_id or "unknown",
    }
    if _ENDPOINT_LABEL:
        span_attrs["nemo_gym.endpoint_label"] = _ENDPOINT_LABEL
    if attributes:
        span_attrs.update(attributes)

    with tracer.start_as_current_span("sandbox.exec", attributes=span_attrs) as span:
        yield span


def record_tool_call_resources(
    span: Any,
    *,
    sandbox_id: str | None,
    instance_id: str | None,
    command_class: str,
    cpu_seconds: float | None,
    mem_peak_bytes: int | None,
    mem_current_bytes: int | None = None,
) -> None:
    """Attach per-sandbox CPU/mem deltas to the tool-call span + bounded metric."""
    if span is not None:
        try:
            if cpu_seconds is not None:
                span.set_attribute("tool_cpu_seconds", float(cpu_seconds))
            if mem_peak_bytes is not None:
                span.set_attribute("tool_mem_peak_bytes", int(mem_peak_bytes))
            if mem_current_bytes is not None:
                span.set_attribute("tool_mem_current_bytes", int(mem_current_bytes))
        except Exception:  # pragma: no cover - best effort
            pass

    metric_attrs = {
        "sandbox_id": sandbox_id or "unknown",
        "instance_id": instance_id or "unknown",
        "command_class": command_class,
    }
    try:
        if _CPU_HISTOGRAM is not None and cpu_seconds is not None:
            _CPU_HISTOGRAM.record(float(cpu_seconds), metric_attrs)
        if _MEM_HISTOGRAM is not None and mem_peak_bytes is not None:
            _MEM_HISTOGRAM.record(float(mem_peak_bytes), metric_attrs)
    except Exception:  # pragma: no cover - best effort
        pass


# --- cgroup bracket wrapping -----------------------------------------------

# Sentinel markers used to fence the cgroup samples we inject into the command
# so the provider can strip them back out of the captured stderr.
CGROUP_SENTINEL_BEGIN = "__NG_CGROUP_BEGIN__"
CGROUP_SENTINEL_END = "__NG_CGROUP_END__"


def wrap_command_with_cgroup_bracket(command: str) -> str:
    """Wrap ``command`` so a single exec captures cgroup deltas around it.

    Reads cpu.stat (usage_usec), memory.current and memory.peak from the
    sandbox cgroup v2 leaf before and after the real command, then emits a
    fenced, sentinel-delimited block on stderr. The real command's exit code
    is preserved. ``parse_cgroup_bracket`` parses the block out and strips it.
    """
    root = cgroup_path()
    # Use a subshell; tolerate missing files (cgroup v1 / non-fastlet) by
    # emitting empty values so parsing simply yields None.
    reader = (
        "__ng_cpu() {{ awk '/^usage_usec/{{print $2}}' {root}/cpu.stat 2>/dev/null || true; }}; "
        "__ng_memc() {{ cat {root}/memory.current 2>/dev/null || true; }}; "
        "__ng_memp() {{ cat {root}/memory.peak 2>/dev/null || true; }}"
    ).format(root=root)
    return (
        f"{reader}; "
        f"__ng_c0=$(__ng_cpu); __ng_m0=$(__ng_memc); "
        f"{{ {command}; }}; __ng_rc=$?; "
        f"__ng_c1=$(__ng_cpu); __ng_mp=$(__ng_memp); __ng_m1=$(__ng_memc); "
        f'>&2 printf "\\n%s cpu0=%s cpu1=%s memcur=%s mempeak=%s %s\\n" '
        f'"{CGROUP_SENTINEL_BEGIN}" "$__ng_c0" "$__ng_c1" "$__ng_m1" "$__ng_mp" "{CGROUP_SENTINEL_END}"; '
        f"exit $__ng_rc"
    )


def parse_cgroup_bracket(stderr: str | None) -> tuple[str | None, dict[str, Any]]:
    """Extract cgroup deltas from a bracketed stderr stream.

    Returns ``(clean_stderr, deltas)`` where ``deltas`` may contain
    ``cpu_seconds``, ``mem_peak_bytes`` and ``mem_current_bytes``. The sentinel
    line is removed from the returned stderr.
    """
    deltas: dict[str, Any] = {}
    if not stderr or CGROUP_SENTINEL_BEGIN not in stderr:
        return stderr, deltas

    clean_lines: list[str] = []
    for line in stderr.splitlines():
        if CGROUP_SENTINEL_BEGIN in line and CGROUP_SENTINEL_END in line:
            try:
                fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok and not tok.startswith("__NG"))
                cpu0 = fields.get("cpu0")
                cpu1 = fields.get("cpu1")
                if cpu0 and cpu1 and cpu0.isdigit() and cpu1.isdigit():
                    deltas["cpu_seconds"] = max(0, int(cpu1) - int(cpu0)) / 1_000_000.0
                mempeak = fields.get("mempeak")
                if mempeak and mempeak.isdigit():
                    deltas["mem_peak_bytes"] = int(mempeak)
                memcur = fields.get("memcur")
                if memcur and memcur.isdigit():
                    deltas["mem_current_bytes"] = int(memcur)
            except Exception:  # pragma: no cover - best effort
                pass
            continue
        clean_lines.append(line)

    clean = "\n".join(clean_lines) or None
    return clean, deltas
