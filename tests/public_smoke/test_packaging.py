from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_built_wheel_has_exact_ai2thor_extra_metadata(tmp_path: Path) -> None:
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

    provided_extras = metadata.get_all("Provides-Extra", [])
    assert provided_extras.count("ai2thor") == 1
    requirements = metadata.get_all("Requires-Dist", [])
    assert requirements.count("ai2thor<6,>=5; extra == 'ai2thor'") == 1
