"""Public example and CLI work without private fixtures or repository metadata."""

import importlib.util
import json
from pathlib import Path

from typer.testing import CliRunner

from spatialcf.cli import app
from spatialcf.domain.general_dataset import GeneralDatasetInput
from spatialcf.domain.serialization import canonical_json_bytes


def test_public_general_cli_empty_roundtrip(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "dataset"
    source.write_bytes(
        canonical_json_bytes(GeneralDatasetInput.seal(sources=(), tasks=()))
    )
    runner = CliRunner()
    generated = runner.invoke(
        app, ["general", "generate", "--input", str(source), "--output", str(output)]
    )
    assert generated.exit_code == 0, generated.output
    assert json.loads(generated.output)["candidate_count"] == 0
    for command in ("verify", "inspect"):
        result = runner.invoke(app, ["general", command, str(output)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == json.loads(generated.output)
    missing = runner.invoke(app, ["general", "verify", str(tmp_path / "missing")])
    assert missing.exit_code == 2 and '"status": "ERROR"' in missing.output


def test_public_example_builds_self_contained_mixed_input():
    path = Path(__file__).resolve().parents[2] / "examples/general_dataset.py"
    spec = importlib.util.spec_from_file_location("general_dataset_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.example_input()
    assert len(value.tasks) == len(value.sources) == 2
    assert {source.kind for source in value.sources} == {
        "PLACEMENT_SNAPSHOT",
        "RIGID_SNAPSHOT",
    }
    assert canonical_json_bytes(
        GeneralDatasetInput.model_validate_json(canonical_json_bytes(value))
    ) == canonical_json_bytes(value)
