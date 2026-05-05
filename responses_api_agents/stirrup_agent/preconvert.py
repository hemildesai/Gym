# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Preconvert Office deliverables to PDF using the agent's Apptainer SIF.

The gym evaluation container has no libreoffice; the per-run agent SIF
does. Convert here so persisted deliverables carry sibling PDFs the
GDPVal multimodal judge can read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger(__name__)


OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

DEFAULT_CONVERT_TIMEOUT_S = 120


def _needs_conversion(path: Path) -> bool:
    return path.suffix.lower() in OFFICE_EXTENSIONS and not path.with_suffix(".pdf").exists()


def _find_office_without_pdf(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.is_dir():
        return files
    for dirpath, _dirs, filenames in os.walk(root):
        for filename in filenames:
            p = Path(dirpath) / filename
            if _needs_conversion(p):
                files.append(p)
    return sorted(files)


async def _convert_one(
    path: Path,
    sif_path: str,
    *,
    timeout: int = DEFAULT_CONVERT_TIMEOUT_S,
    apptainer_binary: str = "apptainer",
) -> tuple[Path, bool, str]:
    out_dir = path.parent
    # Per-call profile keeps concurrent libreoffice instances from clashing.
    profile = f"/tmp/lo-profile-{abs(hash(str(path)))}"
    inner_cmd = (
        "libreoffice --headless --nologo --nolockcheck --nodefault --norestore "
        f"-env:UserInstallation=file://{profile} "
        f"--convert-to pdf --outdir /_pcv_io /_pcv_io/{shlex.quote(path.name)}"
    )
    cmd = (
        f"{shlex.quote(apptainer_binary)} exec --cleanenv "
        f"--bind {shlex.quote(str(out_dir))}:/_pcv_io "
        f"{shlex.quote(sif_path)} bash -c {shlex.quote(inner_cmd)}"
    )

    proc: Optional[asyncio.subprocess.Process] = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return path, False, f"{apptainer_binary} binary not found on host PATH"
    except Exception as exc:
        return path, False, f"failed to spawn apptainer: {exc!r}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return path, False, f"timeout converting {path.name} after {timeout}s"

    if path.with_suffix(".pdf").exists():
        return path, True, f"converted {path.name}"

    tail = (stderr or stdout).decode(errors="replace").strip().splitlines()[-3:]
    detail = " | ".join(tail) or "(no output captured)"
    return path, False, f"libreoffice rc={proc.returncode} did not produce {path.with_suffix('.pdf').name}: {detail}"


async def preconvert_office_in_dir_via_apptainer(
    output_dir: str | os.PathLike,
    sif_path: str,
    *,
    max_concurrent: int = 1,
    timeout: int = DEFAULT_CONVERT_TIMEOUT_S,
    apptainer_binary: str = "apptainer",
) -> tuple[int, int, list[str]]:
    """Convert every pending Office file under ``output_dir`` to PDF.

    Returns ``(num_success, num_failed, error_messages)``.
    """
    if not sif_path:
        return 0, 0, []
    if not Path(sif_path).exists():
        msg = f"SIF not found at {sif_path}; preconvert skipped"
        LOGGER.warning(msg)
        return 0, 0, [msg]

    files = _find_office_without_pdf(Path(output_dir))
    if not files:
        return 0, 0, []

    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(p: Path) -> tuple[Path, bool, str]:
        async with sem:
            return await _convert_one(p, sif_path, timeout=timeout, apptainer_binary=apptainer_binary)

    results = await asyncio.gather(*(_bounded(p) for p in files))
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    errors = [msg for _, ok, msg in results if not ok]
    return n_ok, n_fail, errors
