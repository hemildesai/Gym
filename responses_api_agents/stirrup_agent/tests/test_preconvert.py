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
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_agents.stirrup_agent import preconvert as pcv


class TestFindOfficeWithoutPdf:
    def test_skips_files_with_sibling_pdf(self, tmp_path: Path) -> None:
        (tmp_path / "a.docx").write_text("x")
        (tmp_path / "a.pdf").write_text("p")
        (tmp_path / "b.xlsx").write_text("x")
        files = pcv._find_office_without_pdf(tmp_path)
        assert [f.name for f in files] == ["b.xlsx"]

    def test_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.pptx").write_text("x")
        files = pcv._find_office_without_pdf(tmp_path)
        assert [f.name for f in files] == ["deep.pptx"]

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert pcv._find_office_without_pdf(tmp_path / "nope") == []


@pytest.mark.asyncio
async def test_skips_when_sif_missing(tmp_path: Path) -> None:
    (tmp_path / "a.docx").write_text("x")
    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), "/no/such/sif.sif")
    assert ok == 0
    assert fail == 0
    assert any("SIF not found" in m for m in errors)


@pytest.mark.asyncio
async def test_skips_when_sif_path_empty(tmp_path: Path) -> None:
    (tmp_path / "a.docx").write_text("x")
    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), "")
    assert (ok, fail, errors) == (0, 0, [])


@pytest.mark.asyncio
async def test_no_files_returns_zeros(tmp_path: Path) -> None:
    sif = tmp_path / "fake.sif"
    sif.write_text("not a real sif but file exists")
    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), str(sif))
    assert (ok, fail, errors) == (0, 0, [])


@pytest.mark.asyncio
async def test_success_path_via_mocked_apptainer(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.docx"
    src.write_text("x")
    sif = tmp_path / "fake.sif"
    sif.write_text("sif")

    async def _fake_create(cmd, stdout, stderr):
        # Mimic apptainer running libreoffice and producing the PDF.
        src.with_suffix(".pdf").write_bytes(b"%PDF-1.4 fake\n")

        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create)

    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), str(sif))
    assert (ok, fail, errors) == (1, 0, [])
    assert src.with_suffix(".pdf").exists()


@pytest.mark.asyncio
async def test_failure_when_pdf_not_produced(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.docx"
    src.write_text("x")
    sif = tmp_path / "fake.sif"
    sif.write_text("sif")

    async def _fake_create(cmd, stdout, stderr):
        proc = MagicMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"libreoffice exploded"))
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create)

    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), str(sif))
    assert ok == 0
    assert fail == 1
    assert "libreoffice exploded" in errors[0]
    assert "rc=1" in errors[0]


@pytest.mark.asyncio
async def test_apptainer_binary_missing(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.docx"
    src.write_text("x")
    sif = tmp_path / "fake.sif"
    sif.write_text("sif")

    async def _raise_fnf(cmd, stdout, stderr):
        raise FileNotFoundError("apptainer")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _raise_fnf)

    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), str(sif))
    assert ok == 0
    assert fail == 1
    assert "apptainer binary not found" in errors[0]


@pytest.mark.asyncio
async def test_timeout_kills_process(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "a.docx"
    src.write_text("x")
    sif = tmp_path / "fake.sif"
    sif.write_text("sif")

    async def _fake_create(cmd, stdout, stderr):
        proc = MagicMock()
        proc.returncode = None

        async def _hang():
            await asyncio.sleep(60)
            return b"", b""

        proc.communicate = _hang
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create)

    ok, fail, errors = await pcv.preconvert_office_in_dir_via_apptainer(str(tmp_path), str(sif), timeout=0)
    assert ok == 0
    assert fail == 1
    assert "timeout converting" in errors[0]
