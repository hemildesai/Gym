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

"""Sandbox-side progress probe for installed-agent trajectory runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


G_AGENT_LOG_PATHS = (
    "/logs/agent/policy_trace.jsonl",
    "/logs/agent/policy-proxy.log",
    "/logs/agent/mini-swe-agent.txt",
    "/logs/agent/mini-swe-agent.trajectory.json",
    "/logs/agent/recording.cast",
    "/logs/agent/opencode.txt",
    "/logs/agent/openhands.txt",
    "/logs/agent/openhands_sdk.txt",
    "/logs/agent/openhands.trajectory.json",
    "/logs/agent/aider.txt",
    "/logs/agent/codex.txt",
    "/logs/agent/claude-code.txt",
    "/logs/agent/trajectory.json",
)

G_PROCESS_KEYWORDS = (
    "mini-swe-agent",
    "opencode",
    "openhands",
    "claude",
    "codex",
    "aider",
    "node",
    "python",
)
G_PROBE_BASENAME = "nemo_rl_sandbox_progress_probe.py"


def _line_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _json_size(value: Any, key: str) -> int:
    if isinstance(value, dict):
        total = 1 if key in value else 0
        return total + sum(_json_size(item, key) for item in value.values())
    if isinstance(value, list):
        return sum(_json_size(item, key) for item in value)
    return 0


def _safe_json_summary(path: Path) -> dict[str, int] | None:
    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return {
        "message_key_count": _json_size(payload, "messages"),
        "tool_call_key_count": _json_size(payload, "tool_calls"),
    }


def _policy_trace_summary(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None

    turns = 0
    completion_tokens = 0
    logprobs = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                turns += 1
                record = json.loads(line)
                completion_ids = record.get("completion_token_ids")
                if isinstance(completion_ids, list):
                    completion_tokens += len(completion_ids)
                record_logprobs = record.get("logprobs")
                if isinstance(record_logprobs, list):
                    logprobs += len(record_logprobs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"turns": turns}

    return {
        "turns": turns,
        "completion_tokens": completion_tokens,
        "logprobs": logprobs,
    }


def _file_snapshot(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        return {"path": path_value, "exists": False}

    stat = path.stat()
    snapshot: dict[str, Any] = {
        "path": path_value,
        "exists": True,
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "lines": _line_count(path),
    }
    if path.suffix == ".json":
        json_summary = _safe_json_summary(path)
        if json_summary is not None:
            snapshot["json"] = json_summary
    if path.name == "policy_trace.jsonl":
        trace_summary = _policy_trace_summary(path)
        if trace_summary is not None:
            snapshot["policy_trace"] = trace_summary
    return snapshot


def _process_snapshot() -> list[dict[str, Any]]:
    processes = []
    proc = Path("/proc")
    if not proc.exists():
        return processes
    current_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw_cmdline.replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        if not cmdline:
            continue
        if G_PROBE_BASENAME in cmdline:
            continue
        if not any(keyword in cmdline for keyword in G_PROCESS_KEYWORDS):
            continue
        try:
            stat = entry.stat()
            start_mtime = stat.st_mtime
        except OSError:
            start_mtime = None
        processes.append(
            {
                "pid": pid,
                "cmdline": cmdline[:500],
                "proc_mtime": start_mtime,
            }
        )
    return processes


def collect_progress_snapshot() -> dict[str, Any]:
    """Collect a best-effort progress snapshot from inside the sandbox."""
    files = [_file_snapshot(path) for path in G_AGENT_LOG_PATHS]
    return {
        "generated_at": time.time(),
        "processes": _process_snapshot(),
        "files": files,
    }


def main() -> None:
    print(json.dumps(collect_progress_snapshot(), sort_keys=True))


if __name__ == "__main__":
    main()
