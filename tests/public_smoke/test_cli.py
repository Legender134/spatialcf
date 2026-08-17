from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner, Result

import spatialcf.cli as public_cli
from spatialcf.generation import GenerationReport

app = public_cli.app


runner = CliRunner()
ERROR_KEYS = {"error", "error_type", "request_id", "stage", "status"}


def _assert_structured_parse_error(result: Result, error_type: str) -> None:
    assert result.exit_code == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert set(error) == ERROR_KEYS
    assert error["status"] == "ERROR"
    assert error["error_type"] == error_type
    assert error["error"]
    assert error["request_id"] is None
    assert error["stage"] is None
    assert "Usage:" not in result.stderr
    assert "Traceback" not in result.stderr


def test_public_cli_has_only_three_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "generate" in result.stdout
    assert "verify" in result.stdout
    assert "inspect" in result.stdout
    assert "audit" not in result.stdout
    assert "compare" not in result.stdout
    command = get_command(app)
    assert isinstance(command, typer.core.TyperGroup)
    assert tuple(sorted(command.commands)) == ("generate", "inspect", "verify")


def test_bare_invocation_preserves_no_args_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage:" in result.stdout
    assert "generate" in result.stdout
    assert "verify" in result.stdout
    assert "inspect" in result.stdout
    assert result.stderr == ""
    assert '"status": "ERROR"' not in result.output


def test_cli_error_is_structured_without_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path / "missing")])

    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error["status"] == "ERROR"
    assert error["error_type"] == "FileNotFoundError"
    assert error["error"]
    assert error["request_id"] is None
    assert error["stage"] is None
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_missing_required_option_is_structured(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["generate", "--output", str(tmp_path / "dataset")],
    )

    _assert_structured_parse_error(result, "MissingParameter")


def test_nonexistent_config_is_a_structured_facade_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(tmp_path / "missing.toml"),
            "--output",
            str(tmp_path / "dataset"),
        ],
    )

    _assert_structured_parse_error(result, "FileNotFoundError")


def test_unknown_command_is_structured() -> None:
    result = runner.invoke(app, ["unknown-command"])

    _assert_structured_parse_error(result, "UsageError")


def test_invalid_output_path_type_is_structured(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("config_version = 1\n", encoding="utf-8")
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            str(config),
            "--output",
            str(output_file),
        ],
    )

    _assert_structured_parse_error(result, "BadParameter")


def test_verbose_parse_error_includes_traceback(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["generate", "--output", str(tmp_path / "dataset"), "--verbose"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Traceback (most recent call last)" in result.stderr
    assert json.loads(result.stderr.splitlines()[-1])["error_type"] == (
        "MissingParameter"
    )


def _report() -> GenerationReport:
    return GenerationReport(
        source_count=1,
        source_capture_rejected_count=0,
        frozen_request_count=1,
        planned_request_count=1,
        planning_rejected_request_count=0,
        accepted_request_count=1,
        execution_rejected_request_count=0,
        terminal_reasons={},
        dataset_tree_sha256="0" * 64,
    )


@pytest.mark.parametrize("command", ("generate", "verify"))
def test_report_commands_emit_json_only_on_stdout(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    if command == "generate":
        config = tmp_path / "config.toml"
        config.write_text("config_version = 1\n", encoding="utf-8")
        output = tmp_path / "dataset"
        monkeypatch.setattr(
            public_cli,
            "generate_dataset",
            lambda received_config, received_output: (
                report
                if (received_config, received_output) == (config, output)
                else pytest.fail("generate arguments changed")
            ),
        )
        arguments = ["generate", "--config", str(config), "--output", str(output)]
    else:
        dataset = tmp_path / "dataset"
        monkeypatch.setattr(
            public_cli,
            "verify_dataset",
            lambda received: (
                report
                if received == dataset
                else pytest.fail("verify argument changed")
            ),
        )
        arguments = ["verify", str(dataset)]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert json.loads(result.stdout) == report.model_dump(mode="json")
    assert result.stderr == ""


def test_inspect_emits_sorted_json_only_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    expected = {"record_count": 0, "relation_counts": {}}
    monkeypatch.setattr(
        public_cli,
        "inspect_dataset",
        lambda received: (
            expected if received == dataset else pytest.fail("inspect path changed")
        ),
    )

    result = runner.invoke(app, ["inspect", str(dataset)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected
    assert result.stdout.index('"record_count"') < result.stdout.index(
        '"relation_counts"'
    )
    assert result.stderr == ""


def test_verbose_error_includes_traceback_and_request_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RequestFailure(RuntimeError):
        stage = "execution"
        request_id = "request-example"

    def fail(_dataset: Path) -> GenerationReport:
        raise RequestFailure("native replay failed")

    monkeypatch.setattr(public_cli, "verify_dataset", fail)

    result = runner.invoke(
        app,
        ["verify", str(tmp_path / "dataset"), "--verbose"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Traceback (most recent call last)" in result.stderr
    assert json.loads(result.stderr.splitlines()[-1]) == {
        "error": "native replay failed",
        "error_type": "RequestFailure",
        "request_id": "request-example",
        "stage": "execution",
        "status": "ERROR",
    }
