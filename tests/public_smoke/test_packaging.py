from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_built_wheel_metadata_has_v011_license_and_self_contained_test_extra(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = tuple(tmp_path.glob("spatialcf-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_names = tuple(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        assert len(metadata_names) == 1
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_names[0])
        )

    assert metadata["Version"] == "0.1.1"
    assert metadata["License-Expression"] == "Apache-2.0"
    assert set(metadata.get_all("Provides-Extra", [])) == {"ai2thor", "test"}
    requirements = metadata.get_all("Requires-Dist", [])
    assert requirements.count("ai2thor<6,>=5; extra == 'ai2thor'") == 1
    assert {
        "build<2,>=1; extra == 'test'",
        "hypothesis<7,>=6; extra == 'test'",
        "pytest-xdist<4,>=3.6; extra == 'test'",
        "pytest<10,>=8; extra == 'test'",
    } <= set(requirements)


def test_public_runtime_closure_wheel_has_no_legacy_data_members(
    tmp_path: Path,
) -> None:
    if (ROOT / "release").is_dir():
        pytest.skip("private development checkout retains historical data sources")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheel = next(tmp_path.glob("spatialcf-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert not any(
            name.startswith("spatialcf/data/") for name in archive.namelist()
        )
