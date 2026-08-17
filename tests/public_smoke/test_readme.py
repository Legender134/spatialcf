from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import shutil
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from typer.testing import CliRunner

import spatialcf.adapters.ai2thor as ai2thor_module
import spatialcf.cli as public_cli
from spatialcf.generation import generate_dataset, read_dataset_records
from tests.public_smoke._fake_runtime import READMEAdapterFactory

ROOT = Path(__file__).parents[2]
PUBLIC_DOC_ROOT = (
    ROOT / "docs" / "public"
    if (ROOT / "release" / "public-files.txt").is_file()
    else ROOT / "docs"
)
QUICKSTART = """git clone --branch v0.1.1 --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[ai2thor]"
spatialcf generate --config configs/ai2thor-example.toml --output ./dataset
spatialcf verify ./dataset
spatialcf inspect ./dataset"""
LOCAL_SETUP = '''git clone --branch v0.1.1 --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[ai2thor]"'''
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "README_EN.md",
    PUBLIC_DOC_ROOT / "installation.md",
    PUBLIC_DOC_ROOT / "quickstart.md",
    PUBLIC_DOC_ROOT / "concepts.md",
    PUBLIC_DOC_ROOT / "adapters.md",
    PUBLIC_DOC_ROOT / "api.md",
)
PUBLIC_INSTALLATION_SURFACES = (
    ROOT / "README.md",
    ROOT / "README_EN.md",
    PUBLIC_DOC_ROOT / "installation.md",
    PUBLIC_DOC_ROOT / "quickstart.md",
    PUBLIC_DOC_ROOT / "adapters.md",
)
SHELL_FENCE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
PIP_EXECUTABLE = re.compile(r"pip(?:\d+(?:\.\d+)?)?$")
PYTHON_EXECUTABLE = re.compile(r"python(?:\d+(?:\.\d+)?)?$")
PUBLIC_TEXT_FILES = (
    *PUBLIC_DOCS,
    ROOT / "configs/ai2thor-example.toml",
    ROOT / "examples/generate_ai2thor_dataset.py",
    ROOT / "examples/verify_dataset.py",
    ROOT / "examples/inspect_result.py",
)
FORBIDDEN_MARKERS = (
    "competition",
    "qwen",
    "benchmark",
    "spatialcf-" + "development",
    "docs/" + "superpowers",
    "docs/handoff",
    "spatialcf.pipelines",
    "/home/" + "pale/",
    "2.9.",
    "state-of-the-art",
    "training pipeline",
    "model training",
)


def _fenced_shell_blocks(path: Path) -> tuple[str, ...]:
    return tuple(
        block.strip() for block in SHELL_FENCE.findall(path.read_text(encoding="utf-8"))
    )


def _fenced_shell_commands(path: Path) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(shlex.split(line))
        for block in _fenced_shell_blocks(path)
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _strip_environment_prefix(command: tuple[str, ...]) -> tuple[str, ...] | None:
    index = 0
    while index < len(command) and SHELL_ASSIGNMENT.fullmatch(command[index]):
        index += 1
    if index == len(command) or command[index] != "env":
        return command[index:]

    index += 1
    while index < len(command):
        token = command[index]
        if token == "--":
            index += 1
            while index < len(command) and SHELL_ASSIGNMENT.fullmatch(command[index]):
                index += 1
            return command[index:]
        if SHELL_ASSIGNMENT.fullmatch(token) or token in {"-i", "--ignore-environment"}:
            index += 1
            continue
        if token in {"-u", "--unset"}:
            if index + 1 == len(command):
                return None
            index += 2
            continue
        if token.startswith("-"):
            return None
        return command[index:]
    return ()


def _pip_install_operands(command: tuple[str, ...]) -> tuple[str, ...] | None:
    invocation = _strip_environment_prefix(command)
    if invocation is None:
        return None
    if not invocation:
        return ()

    executable = Path(invocation[0]).name
    if PIP_EXECUTABLE.fullmatch(executable):
        pip_arguments = invocation[1:]
    elif PYTHON_EXECUTABLE.fullmatch(executable):
        module_index = next(
            (
                index
                for index in range(1, len(invocation) - 1)
                if invocation[index : index + 2] == ("-m", "pip")
            ),
            None,
        )
        if module_index is None:
            return ()
        pip_arguments = invocation[module_index + 2 :]
    else:
        return ()

    try:
        install_index = pip_arguments.index("install")
    except ValueError:
        return ()
    return tuple(
        argument
        for argument in pip_arguments[install_index + 1 :]
        if not argument.startswith("-")
    )


def _is_index_only_ai2thor_install(command: tuple[str, ...]) -> bool:
    operands = _pip_install_operands(command)
    if operands is None:
        return True
    return any(operand.lower() == "spatialcf[ai2thor]" for operand in operands)


def _assert_no_index_only_ai2thor_install(paths: tuple[Path, ...]) -> None:
    for path in paths:
        for command in _fenced_shell_commands(path):
            assert not _is_index_only_ai2thor_install(command), (path, command)


def test_public_installation_surfaces_begin_with_the_release_local_setup() -> None:
    for path in PUBLIC_INSTALLATION_SURFACES:
        assert any(
            block.startswith(LOCAL_SETUP) for block in _fenced_shell_blocks(path)
        ), path


def test_quickstart_fences_continue_from_the_release_local_setup() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "README_EN.md",
        PUBLIC_DOC_ROOT / "quickstart.md",
    ):
        assert any(
            block.startswith(QUICKSTART) for block in _fenced_shell_blocks(path)
        ), path


def test_public_documentation_shell_fences_reject_index_only_ai2thor_installs() -> None:
    _assert_no_index_only_ai2thor_install(PUBLIC_DOCS)


@pytest.mark.parametrize(
    "legacy_install",
    (
        'pip install "spatialcf[ai2thor]"',
        'pip --disable-pip-version-check install -q "spatialcf[ai2thor]"',
        'python -m pip install --no-cache-dir "spatialcf[ai2thor]"',
        'pip3 install "spatialcf[ai2thor]"',
        'python -I -m pip install "spatialcf[ai2thor]"',
        'PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "spatialcf[ai2thor]"',
        'env PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "spatialcf[ai2thor]"',
        'env --unknown-option pip install "spatialcf[ai2thor]"',
        'env -- PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "spatialcf[ai2thor]"',
        'env -u HOME -- PIP_DISABLE_PIP_VERSION_CHECK=1 pip install "spatialcf[ai2thor]"',
    ),
)
def test_shell_install_validation_rejects_mutated_index_only_commands(
    tmp_path: Path,
    legacy_install: str,
) -> None:
    mutated_doc = tmp_path / "installation.md"
    mutated_doc.write_text(
        f"```bash\n{LOCAL_SETUP}\n{legacy_install}\n```\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError):
        _assert_no_index_only_ai2thor_install((mutated_doc,))


def test_shell_install_validation_accepts_the_required_local_setup(
    tmp_path: Path,
) -> None:
    local_setup_doc = tmp_path / "installation.md"
    local_setup_doc.write_text(f"```bash\n{LOCAL_SETUP}\n```\n", encoding="utf-8")

    _assert_no_index_only_ai2thor_install((local_setup_doc,))


def test_example_configuration_is_exact() -> None:
    path = ROOT / "configs/ai2thor-example.toml"
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {
        "config_version": 1,
        "adapter": "ai2thor",
        "scene_names": ["FloorPlan2"],
        "split": "train",
        "campaign_id": "ai2thor-quickstart",
        "seed": 20260723,
        "width": 640,
        "height": 480,
        "max_requests": 12,
    }


def test_readme_fake_adapter_uses_synthetic_runtime_identity_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_distribution_metadata(_name: str) -> str:
        raise AssertionError("fake runtime consulted installed package metadata")

    monkeypatch.setattr(ai2thor_module, "package_version", reject_distribution_metadata)
    factory = READMEAdapterFactory()

    with factory(["FloorPlan2"], width=80, height=60, seed=20260723) as adapter:
        identity = adapter.runtime_identity()

    assert identity.ai2thor_version == "synthetic-public-smoke"
    assert identity.unity_commit_id == "public-readme-fake"
    assert identity.native_scene_name == "FloorPlan2"
    assert (identity.width, identity.height, identity.seed) == (80, 60, 20260723)


def test_readme_dataset_commands_run_through_the_documented_fake_adapter_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copy2(
        ROOT / "configs/ai2thor-example.toml",
        tmp_path / "configs/ai2thor-example.toml",
    )
    adapter_factory = READMEAdapterFactory()

    def generate_with_fake_adapter(config: Path, output: Path):
        return generate_dataset(
            config,
            output,
            adapter_factory=adapter_factory,
        )

    monkeypatch.setattr(public_cli, "generate_dataset", generate_with_fake_adapter)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    commands = [
        line for line in QUICKSTART.splitlines() if line.startswith("spatialcf ")
    ]
    results = [
        runner.invoke(public_cli.app, shlex.split(command)[1:]) for command in commands
    ]

    assert [result.exit_code for result in results] == [0, 0, 0], [
        (result.stdout, result.stderr, repr(result.exception)) for result in results
    ]
    assert all(result.stderr == "" for result in results)
    generated = json.loads(results[0].stdout)
    verified = json.loads(results[1].stdout)
    inspected = json.loads(results[2].stdout)
    assert generated == verified
    assert generated["frozen_request_count"] == (
        generated["planned_request_count"]
        + generated["planning_rejected_request_count"]
    )
    assert generated["planned_request_count"] == (
        generated["accepted_request_count"]
        + generated["execution_rejected_request_count"]
    )
    assert generated["frozen_request_count"] > 0
    assert generated["planned_request_count"] > 0
    assert generated["accepted_request_count"] > 0
    records = read_dataset_records(tmp_path / "dataset")
    assert len(records) == generated["accepted_request_count"]
    expected_bundle_files = {
        "after-depth.npy",
        "after-instance.png",
        "after-pointcloud.ply",
        "after-rgb.png",
        "before-depth.npy",
        "before-instance.png",
        "before-pointcloud.ply",
        "before-rgb.png",
        "bundle.json",
        "checksums.sha256",
    }
    for record in records:
        bundle_root = tmp_path / "dataset" / record.bundle_path
        assert bundle_root.is_dir()
        assert {path.name for path in bundle_root.iterdir()} == expected_bundle_files
    assert inspected["report"] == generated
    assert adapter_factory.constructions > 0


@pytest.mark.parametrize(
    "name",
    ("generate_ai2thor_dataset.py", "verify_dataset.py", "inspect_result.py"),
)
def test_examples_use_only_the_public_generation_api(name: str) -> None:
    path = ROOT / "examples" / name
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "spatialcf.generation" in imported_modules
    assert all(
        not module.startswith("spatialcf.pipelines") for module in imported_modules
    )


def test_public_docs_are_user_facing_and_platform_neutral() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_TEXT_FILES)
    lowered = combined.lower()
    for marker in FORBIDDEN_MARKERS:
        assert marker not in lowered
    assert "platform-neutral" in combined
    assert "平台无关" in combined
    assert "Unity" in combined
    assert "Adapter" in combined


def test_documented_cli_commands_use_only_the_public_surface() -> None:
    command_lines = []
    for path in PUBLIC_DOCS:
        command_lines.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("spatialcf ")
        )

    assert command_lines
    for line in command_lines:
        arguments = shlex.split(line)
        assert arguments[1] in {"--help", "generate", "verify", "inspect"}
        assert not re.search(r"\b(audit|compare|sample-review|validate-review)\b", line)


def test_every_local_readme_link_resolves() -> None:
    for readme in (ROOT / "README.md", ROOT / "README_EN.md"):
        targets = re.findall(
            r"\[[^\]]+\]\(([^)]+)\)",
            readme.read_text(encoding="utf-8"),
        )
        assert targets
        for target in targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = readme.parent / parsed.path
            assert resolved.is_file(), (readme, target)


def test_license_is_unmodified_apache_2() -> None:
    payload = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_notice_is_factual_and_has_no_guessed_third_party_notice() -> None:
    expected = (
        b"SpatialCF\n"
        b"Copyright 2026 SpatialCF contributors\n"
        b"\n"
        b"This product includes software developed by the SpatialCF "
        b"contributors.\n"
    )
    assert (ROOT / "NOTICE").read_bytes() == expected
