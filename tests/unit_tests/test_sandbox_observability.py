# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_gym.sandbox import observability as obs


def test_wrap_command_embeds_real_command_and_sentinels(monkeypatch):
    monkeypatch.setenv("NG_OBS_SANDBOX_CGROUP_PATH", "/sys/fs/cgroup")
    wrapped = obs.wrap_command_with_cgroup_bracket("pytest -q tests/")
    assert "pytest -q tests/" in wrapped
    assert obs.CGROUP_SENTINEL_BEGIN in wrapped
    assert obs.CGROUP_SENTINEL_END in wrapped
    # the real command's exit code must be preserved
    assert "exit $__ng_rc" in wrapped


def test_parse_cgroup_bracket_extracts_deltas_and_strips_sentinel():
    stderr = (
        "real line 1\n"
        f"\n{obs.CGROUP_SENTINEL_BEGIN} cpu0=1000000 cpu1=3500000 "
        f"memcur=104857600 mempeak=209715200 {obs.CGROUP_SENTINEL_END}\n"
        "real line 2"
    )
    clean, deltas = obs.parse_cgroup_bracket(stderr)
    assert deltas["cpu_seconds"] == 2.5
    assert deltas["mem_peak_bytes"] == 209715200
    assert deltas["mem_current_bytes"] == 104857600
    assert obs.CGROUP_SENTINEL_BEGIN not in (clean or "")
    assert "real line 1" in clean
    assert "real line 2" in clean


def test_parse_cgroup_bracket_passthrough_without_sentinel():
    clean, deltas = obs.parse_cgroup_bracket("plain stderr")
    assert clean == "plain stderr"
    assert deltas == {}


def test_parse_cgroup_bracket_handles_missing_values():
    line = f"{obs.CGROUP_SENTINEL_BEGIN} cpu0= cpu1= memcur= mempeak= {obs.CGROUP_SENTINEL_END}"
    clean, deltas = obs.parse_cgroup_bracket(line)
    assert deltas == {}
    assert clean is None


def test_parse_cgroup_bracket_clamps_negative_cpu_delta():
    # A counter that appears to go backwards must not produce a negative delta.
    line = f"{obs.CGROUP_SENTINEL_BEGIN} cpu0=5000000 cpu1=1000000 memcur=10 mempeak=20 {obs.CGROUP_SENTINEL_END}"
    _clean, deltas = obs.parse_cgroup_bracket(line)
    assert deltas["cpu_seconds"] == 0.0


def test_tool_call_span_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("NG_OBS_EXPORT_TRACES", raising=False)
    obs._SETUP_DONE = False
    obs._TRACER = None
    with obs.tool_call_span("pytest", sandbox_id="sb1", instance_id="inst1") as span:
        # no-op span must accept set_attribute / record without raising
        obs.record_tool_call_resources(
            span,
            sandbox_id="sb1",
            instance_id="inst1",
            command_class="pytest",
            cpu_seconds=1.0,
            mem_peak_bytes=1024,
        )


def test_env_flags(monkeypatch):
    monkeypatch.setenv("NG_OBS_EXPORT_TRACES", "true")
    monkeypatch.setenv("NG_OBS_SANDBOX_CGROUP_BRACKET", "1")
    assert obs.traces_enabled() is True
    monkeypatch.setenv("NG_OBS_EXPORT_TRACES", "0")
    assert obs.traces_enabled() is False


def test_resource_capture_mode(monkeypatch):
    monkeypatch.delenv("NG_OBS_SANDBOX_RESOURCE_MODE", raising=False)
    monkeypatch.delenv("NG_OBS_SANDBOX_CGROUP_BRACKET", raising=False)
    assert obs.resource_capture_mode() == "off"
    # back-compat: the legacy bracket toggle now defaults to host-side get_metrics
    monkeypatch.setenv("NG_OBS_SANDBOX_CGROUP_BRACKET", "1")
    assert obs.resource_capture_mode() == "get_metrics"
    assert obs.cgroup_bracket_enabled() is False
    # explicit cgroup mode wins and enables the in-sandbox wrapper
    monkeypatch.setenv("NG_OBS_SANDBOX_RESOURCE_MODE", "cgroup")
    assert obs.resource_capture_mode() == "cgroup"
    assert obs.cgroup_bracket_enabled() is True


class _FakeMetrics:
    def __init__(self, cpu_count, cpu_pct, mem_mib):
        self.cpu_count = cpu_count
        self.cpu_used_percentage = cpu_pct
        self.memory_used_in_mib = mem_mib
        self.memory_total_in_mib = 2048.0
        self.timestamp = 0


def test_metrics_delta_from_samples():
    before = _FakeMetrics(cpu_count=2.0, cpu_pct=10.0, mem_mib=100.0)
    after = _FakeMetrics(cpu_count=2.0, cpu_pct=90.0, mem_mib=180.0)
    deltas = obs.metrics_delta_from_samples(before, after, elapsed_s=4.0)
    # mean cores busy = (10+90)/2/100 * 2 cores = 1.0 core; * 4s = 4.0 cpu-seconds
    assert deltas["cpu_seconds"] == 4.0
    # peak mem = max(100,180) MiB
    assert deltas["mem_peak_bytes"] == int(180.0 * 1024 * 1024)
    assert deltas["mem_current_bytes"] == int(180.0 * 1024 * 1024)


def test_metrics_delta_zero_elapsed():
    s = _FakeMetrics(cpu_count=1.0, cpu_pct=50.0, mem_mib=10.0)
    deltas = obs.metrics_delta_from_samples(s, s, elapsed_s=0.0)
    assert "cpu_seconds" not in deltas
    assert deltas["mem_peak_bytes"] == int(10.0 * 1024 * 1024)
