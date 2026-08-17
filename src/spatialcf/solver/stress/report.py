"""Deterministic publication of solver stress reports and failure evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.data.writer import _rename_no_replace
from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.geometry.obb import obb_footprint
from spatialcf.solver.stress.models import (
    SatStressOracle,
    StressCase,
    UnsatStressOracle,
)
from spatialcf.solver.stress.validation import (
    FailedStressValidationRecord,
    SatStressValidationRecord,
    StressValidationEvidence,
    StressValidationReport,
    UnsatStressValidationRecord,
    rebuild_stress_validation_record,
    stress_after_scene_digest,
    stress_solver_result_digest,
    stress_solver_result_payload,
)

_GENERATOR_VERSION = "stress-v1"
_ORACLE_VERSION = "stress-v1"
_SOLVER_VERSION = "certified-continuous-v1"
_WIDTH = 1000
_HEIGHT = 800
_MARGIN = 60
_TOP_MARGIN = 130
_PATH_DISCLAIMER = "displacement only; not a motion path"


class StressReportPublicationError(RuntimeError):
    """Publication failed with an explicit visibility and recovery location."""

    def __init__(
        self,
        output: Path,
        *,
        published: bool,
        staging_path: Path | None,
        detail: str,
    ) -> None:
        state = "published" if published else "not published"
        recovery = (
            f"; preserved staging: {staging_path}"
            if staging_path is not None
            else ""
        )
        super().__init__(f"{detail}; report is {state}: {output}{recovery}")
        self.output = output
        self.published = published
        self.staging_path = staging_path


class StressReportDurabilityError(StressReportPublicationError):
    """A verified report is visible but its parent fsync failed."""

    def __init__(self, output: Path) -> None:
        super().__init__(
            output,
            published=True,
            staging_path=None,
            detail="published but parent directory fsync failed",
        )


def _counts(
    report: StressValidationReport,
    attribute: str,
) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}
    for record in report.cases:
        key = str(getattr(record, attribute))
        counts = values.setdefault(key, {"passed": 0, "total": 0})
        counts["total"] += 1
        counts["passed"] += record.result == "PASS"
    return values


def _maximum(
    values: Sequence[tuple[str, float]],
) -> dict[str, str | float] | None:
    if not values:
        return None
    case_id, value = max(values, key=lambda item: (item[1], item[0]))
    return {"case_id": case_id, "value_m": value}


def _report_payload(report: StressValidationReport) -> dict[str, object]:
    cases = [
        record.model_dump(mode="json")
        | {
            "failure_path": (
                f"failures/{record.case_id}"
                if isinstance(record, FailedStressValidationRecord)
                else None
            )
        }
        for record in report.cases
    ]
    certificates = [
        (record.case_id, record.certificate.optimality_gap)
        for record in report.cases
        if record.certificate is not None
    ]
    realized_errors = [
        (record.case_id, record.realized_error_m)
        for record in report.cases
        if isinstance(record, SatStressValidationRecord)
    ]
    return {
        **report.model_dump(mode="json", exclude={"cases"}),
        "generator_version": _GENERATOR_VERSION,
        "oracle_version": _ORACLE_VERSION,
        "solver_version": _SOLVER_VERSION,
        "cases": cases,
        "statistics": {
            "by_direction": _counts(report, "direction"),
            "by_expected_outcome": _counts(report, "expected_outcome"),
            "by_family": _counts(report, "family"),
            "by_seed": _counts(report, "seed"),
        },
        "maximum_realized_error": _maximum(realized_errors),
        "maximum_certificate_gap": _maximum(certificates),
    }


def stress_markdown(report: StressValidationReport) -> str:
    """Return the stable human-readable projection of a stress report."""
    lines = [
        f"# Core Solver Stress Validation: {report.status}",
        "",
        f"Profile: `{report.profile}`",
        f"Cases: {report.case_count}",
        f"Failures: {report.failure_count}",
        "",
        "| case | direction | family | expected | actual | result |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        "| "
        f"{record.case_id} | {record.direction} | {record.family} | "
        f"{record.expected_outcome} | {record.actual_outcome.value} | "
        f"{record.result} |"
        for record in report.cases
    )
    return "\n".join((*lines, ""))


def _pixel_transform(scene: Scene):
    xs = [point.x for point in scene.room_polygon_xy]
    ys = [point.y for point in scene.room_polygon_xy]
    scale = min(
        (_WIDTH - 2 * _MARGIN) / (max(xs) - min(xs)),
        (_HEIGHT - _TOP_MARGIN - _MARGIN) / (max(ys) - min(ys)),
    )

    def transform(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round(_MARGIN + (point[0] - min(xs)) * scale),
            round(_HEIGHT - _MARGIN - (point[1] - min(ys)) * scale),
        )

    return transform


def _polygon_pixels(scene: Scene, object_id: str, transform) -> list[tuple[int, int]]:
    footprint = obb_footprint(scene.object_by_id(object_id).obb)
    return [transform((float(x), float(y))) for x, y in footprint.exterior.coords]


def render_stress_failure(
    case: StressCase,
    after: Scene | None,
    record: FailedStressValidationRecord,
) -> Image.Image:
    """Render deterministic endpoint geometry for one failed stress case."""
    before = case.scene
    transform = _pixel_transform(before)
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    room = [transform((point.x, point.y)) for point in before.room_polygon_xy]
    draw.polygon(room, fill="#F5F8FA", outline="#1D3557", width=4)
    display = after if after is not None else before
    subject_id = case.intervention.subject_id
    for obj in sorted(display.objects, key=lambda item: item.object_id):
        if obj.object_id == subject_id:
            continue
        draw.polygon(
            _polygon_pixels(display, obj.object_id, transform),
            fill="#B0BEC5",
            outline="#455A64",
            width=3,
        )
    draw.line(
        _polygon_pixels(before, subject_id, transform),
        fill="#277DA1",
        width=5,
        joint="curve",
    )
    if after is not None:
        draw.line(
            _polygon_pixels(after, subject_id, transform),
            fill="#F94144",
            width=5,
            joint="curve",
        )
    draw.text((30, 20), record.case_id, fill="#111827")
    draw.text((30, 45), f"FAILED: {', '.join(record.errors)}", fill="#C1121F")
    draw.text((30, 70), _PATH_DISCLAIMER, fill="#6A4C93")
    return image


def _render_png_bytes(
    case: StressCase,
    after: Scene,
    record: FailedStressValidationRecord,
) -> bytes:
    stream = BytesIO()
    render_stress_failure(case, after, record).save(
        stream,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return stream.getvalue()


def _solver_result_payload(item: StressValidationEvidence) -> dict[str, object]:
    return stress_solver_result_payload(item.solve_result)


def _case_sha256(case: StressCase) -> str:
    return hashlib.sha256(
        canonical_json_bytes(case.model_dump(mode="json"))
    ).hexdigest()


def _validate_evidence_binding(
    report: StressValidationReport,
    evidence: Sequence[StressValidationEvidence],
) -> None:
    if len(report.cases) != len(evidence):
        raise ValueError("stress report evidence membership mismatch")
    for record, item in zip(report.cases, evidence, strict=True):
        case = item.case
        result = item.solve_result
        expected_metadata = (
            case.case_id,
            case.seed,
            case.direction,
            case.raw_slot,
            case.family,
            case.transform,
            _case_sha256(case),
            case.expected_outcome,
        )
        observed_metadata = (
            record.case_id,
            record.seed,
            record.direction,
            record.raw_slot,
            record.family,
            record.transform,
            record.case_sha256,
            record.expected_outcome,
        )
        if (
            observed_metadata != expected_metadata
            or case.scene.scene_id != case.case_id
        ):
            raise ValueError("stress report evidence case identity mismatch")
        if (
            record.solver_result_digest != stress_solver_result_digest(result)
            or record.after_scene_digest != stress_after_scene_digest(item.after_scene)
        ):
            raise ValueError("stress report evidence digest mismatch")

        rebuilt = rebuild_stress_validation_record(
            case,
            result,
            item.after_scene,
        )
        if rebuilt != record:
            raise ValueError("stress report evidence validation record mismatch")

        after = item.after_scene
        if isinstance(record, SatStressValidationRecord):
            if after is None or result.subject_position is None:
                raise ValueError("stress report evidence SAT replay mismatch")
            before_subject = case.scene.object_by_id(case.intervention.subject_id)
            after_subject = after.object_by_id(case.intervention.subject_id)
            if (
                record.before_xy.x != before_subject.position.x
                or record.before_xy.y != before_subject.position.y
                or record.after_xy.x != after_subject.position.x
                or record.after_xy.y != after_subject.position.y
            ):
                raise ValueError("stress report evidence SAT record mismatch")
        elif isinstance(record, UnsatStressValidationRecord) and after is not None:
            raise ValueError("stress report evidence UNSAT replay mismatch")

        if after is not None:
            position = result.subject_position
            if after.scene_id != case.scene.scene_id or position is None:
                raise ValueError("stress report evidence after-scene identity mismatch")
            replayed = after.object_by_id(case.intervention.subject_id).position
            if replayed.x != position.x or replayed.y != position.y:
                raise ValueError("stress report evidence after-scene identity mismatch")


def _relative_parts(relative: str) -> tuple[str, ...]:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError("stress report entry path must stay relative")
    return tuple(part for part in value.parts if part != ".")


def _open_directory_at(root_descriptor: int, relative: str = ".") -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for stress report publication")
    descriptor = os.dup(root_descriptor)
    try:
        for part in _relative_parts(relative):
            child = os.open(
                part,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        result = os.fstat(descriptor)
        if not stat.S_ISDIR(result.st_mode):
            raise RuntimeError("stress report entry is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent_directory_at(
    root_descriptor: int,
    relative: str,
) -> tuple[int, str]:
    parts = _relative_parts(relative)
    if not parts:
        raise RuntimeError("stress report file path is empty")
    parent = "/".join(parts[:-1]) or "."
    return _open_directory_at(root_descriptor, parent), parts[-1]


def _write_bytes(root_descriptor: int, relative: str, payload: bytes) -> None:
    parent_descriptor, name = _open_parent_directory_at(root_descriptor, relative)
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    try:
        label = Path(relative)
        _require_regular_single_link(os.fstat(descriptor), label)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("stress report write made no progress")
            offset += written
        _fsync_file(descriptor)
        _require_regular_single_link(os.fstat(descriptor), label)
    finally:
        os.close(descriptor)


def _read_bytes(
    root_descriptor: int,
    relative: str,
    expected_identity: tuple[int, int, int],
) -> bytes:
    parent_descriptor, name = _open_parent_directory_at(root_descriptor, relative)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        before = os.fstat(descriptor)
        label = Path(relative)
        _require_regular_single_link(before, label)
        if (
            before.st_dev,
            before.st_ino,
            stat.S_IFMT(before.st_mode),
        ) != expected_identity:
            raise RuntimeError("stress report file identity changed before read")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require_regular_single_link(after, label)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("stress report file identity changed during read")
        bound = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (bound.st_dev, bound.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("stress report file pathname changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _decode_json(payload: bytes) -> object:
    return json.loads(payload.decode("utf-8"))


def _require_regular_single_link(result: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(result.st_mode):
        raise RuntimeError(f"stress report entry is not a regular file: {path.name}")
    if result.st_nlink != 1:
        raise RuntimeError(f"stress report file link count must be one: {path.name}")


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_tree_directory(root_descriptor: int, relative: str) -> None:
    descriptor = _open_directory_at(root_descriptor, relative)
    try:
        before = os.fstat(descriptor)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("stress report directory identity changed during fsync")
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for stress report publication")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError("stress report fsync target is not a directory")
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("stress report directory identity changed during fsync")
    finally:
        os.close(descriptor)


def _write_json(root_descriptor: int, relative: str, value: object) -> None:
    _write_bytes(
        root_descriptor,
        relative,
        canonical_json_bytes(value, pretty=True),
    )


def _write_text(root_descriptor: int, relative: str, value: str) -> None:
    _write_bytes(root_descriptor, relative, value.encode("utf-8"))


def _failure_entries(
    report: StressValidationReport,
    evidence: Sequence[StressValidationEvidence],
) -> tuple[set[str], set[str]]:
    files = {"report.json", "report.md"}
    directories: set[str] = set()
    for record, item in zip(report.cases, evidence, strict=True):
        if not isinstance(record, FailedStressValidationRecord):
            continue
        directories.update({"failures", f"failures/{record.case_id}"})
        prefix = f"failures/{record.case_id}"
        files.update(
            {
                f"{prefix}/before.json",
                f"{prefix}/checks.json",
                f"{prefix}/intervention.json",
                f"{prefix}/oracle.json",
                f"{prefix}/replay.txt",
                f"{prefix}/solver-result.json",
            }
        )
        if item.after_scene is not None:
            files.update({f"{prefix}/after.json", f"{prefix}/topdown.png"})
    return files, directories


def _mkdir_directory_at(root_descriptor: int, relative: str) -> None:
    parent_descriptor, name = _open_parent_directory_at(root_descriptor, relative)
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        created = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(created.st_mode):
            raise RuntimeError("stress report created entry is not a directory")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            if (created.st_dev, created.st_ino) != (opened.st_dev, opened.st_ino):
                raise RuntimeError("stress report directory changed before open")
            os.fchmod(descriptor, 0o700)
            final = os.fstat(descriptor)
            if (
                (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)
                or stat.S_IMODE(final.st_mode) != 0o700
            ):
                raise RuntimeError("stress report directory changed during fchmod")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _collect_tree_identity_manifest(
    root_descriptor: int,
) -> tuple[dict[str, tuple[int, int, int]], set[str], set[str]]:
    root_result = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_result.st_mode):
        raise RuntimeError("stress report root is not a directory")
    identity_manifest = {
        ".": (
            root_result.st_dev,
            root_result.st_ino,
            stat.S_IFMT(root_result.st_mode),
        )
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    observed_inodes = {(root_result.st_dev, root_result.st_ino): "."}

    def visit(directory_descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_descriptor)):
            result = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISREG(result.st_mode):
                _require_regular_single_link(result, Path(relative))
                observed_files.add(relative)
            elif stat.S_ISDIR(result.st_mode):
                observed_directories.add(relative)
            else:
                raise RuntimeError("stress report staging contains an unsafe entry")
            inode = (result.st_dev, result.st_ino)
            if inode in observed_inodes:
                raise RuntimeError(
                    "stress report staging contains a duplicate inode: "
                    f"{observed_inodes[inode]} and {relative}"
                )
            observed_inodes[inode] = relative
            identity_manifest[relative] = (
                result.st_dev,
                result.st_ino,
                stat.S_IFMT(result.st_mode),
            )
            if stat.S_ISDIR(result.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    opened = os.fstat(child)
                    if (result.st_dev, result.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise RuntimeError(
                            "stress report directory identity changed during traversal"
                        )
                    visit(child, relative)
                finally:
                    os.close(child)

    visit(root_descriptor, "")
    return identity_manifest, observed_files, observed_directories


def _verify_json(
    root_descriptor: int,
    relative: str,
    expected: object,
    identity_manifest: dict[str, tuple[int, int, int]],
) -> object:
    expected_bytes = canonical_json_bytes(expected, pretty=True)
    observed = _read_bytes(
        root_descriptor,
        relative,
        identity_manifest[relative],
    )
    if observed != expected_bytes:
        raise RuntimeError(
            f"stress report JSON payload mismatch: {PurePosixPath(relative).name}"
        )
    return _decode_json(observed)


def _verify_tree(
    root: Path,
    report: StressValidationReport,
    evidence: Sequence[StressValidationEvidence],
    expected_identity_manifest: dict[str, tuple[int, int, int]] | None = None,
    *,
    root_descriptor: int,
) -> dict[str, tuple[int, int, int]]:
    expected_files, expected_directories = _failure_entries(report, evidence)
    identity_manifest, observed_files, observed_directories = (
        _collect_tree_identity_manifest(root_descriptor)
    )
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise RuntimeError("stress report staging file set mismatch")

    _verify_json(
        root_descriptor,
        "report.json",
        _report_payload(report),
        identity_manifest,
    )
    markdown = _read_bytes(
        root_descriptor,
        "report.md",
        identity_manifest["report.md"],
    )
    if markdown != stress_markdown(report).encode("utf-8"):
        raise RuntimeError("stress report Markdown payload mismatch")
    markdown.decode("utf-8")

    for record, item in zip(report.cases, evidence, strict=True):
        if not isinstance(record, FailedStressValidationRecord):
            continue
        case_root = f"failures/{record.case_id}"
        before = _verify_json(
            root_descriptor,
            f"{case_root}/before.json",
            item.case.scene.model_dump(mode="json"),
            identity_manifest,
        )
        Scene.model_validate(before)
        intervention = _verify_json(
            root_descriptor,
            f"{case_root}/intervention.json",
            item.case.intervention.model_dump(mode="json"),
            identity_manifest,
        )
        InterventionSpec.model_validate(intervention)
        oracle = _verify_json(
            root_descriptor,
            f"{case_root}/oracle.json",
            item.case.oracle.model_dump(mode="json"),
            identity_manifest,
        )
        oracle_type = (
            SatStressOracle
            if isinstance(item.case.oracle, SatStressOracle)
            else UnsatStressOracle
        )
        oracle_type.model_validate(oracle)
        _verify_json(
            root_descriptor,
            f"{case_root}/solver-result.json",
            _solver_result_payload(item),
            identity_manifest,
        )
        _verify_json(
            root_descriptor,
            f"{case_root}/checks.json",
            {
                "case_id": record.case_id,
                "checks": dict(record.checks),
                "errors": record.errors,
            },
            identity_manifest,
        )
        replay_relative = f"{case_root}/replay.txt"
        replay = _read_bytes(
            root_descriptor,
            replay_relative,
            identity_manifest[replay_relative],
        )
        expected_replay = (
            "PYTHONPATH=src .venv-core/bin/python scripts/stress_core_solver.py "
            f"--case-id {record.case_id} "
            f"--output artifacts/core-solver-stress-{record.case_id}\n"
        ).encode()
        if replay != expected_replay:
            raise RuntimeError("stress report replay payload mismatch")
        replay.decode("utf-8")
        if item.after_scene is not None:
            after = _verify_json(
                root_descriptor,
                f"{case_root}/after.json",
                item.after_scene.model_dump(mode="json"),
                identity_manifest,
            )
            Scene.model_validate(after)
            image_relative = f"{case_root}/topdown.png"
            image_payload = _read_bytes(
                root_descriptor,
                image_relative,
                identity_manifest[image_relative],
            )
            expected_image = _render_png_bytes(
                item.case,
                item.after_scene,
                record,
            )
            if image_payload != expected_image:
                raise RuntimeError("stress report PNG payload mismatch")
            with Image.open(BytesIO(image_payload)) as image:
                image.verify()
            with Image.open(BytesIO(image_payload)) as image:
                if image.mode != "RGB" or image.size != (_WIDTH, _HEIGHT):
                    raise RuntimeError("stress report image contract mismatch")
    if (
        expected_identity_manifest is not None
        and identity_manifest != expected_identity_manifest
    ):
        raise RuntimeError("stress report identity manifest mismatch")
    return identity_manifest


def _created_staging_identity(path: Path) -> tuple[int, int]:
    result = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(result.st_mode):
        raise RuntimeError("stress report staging is not a directory")
    if stat.S_IMODE(result.st_mode) != 0o700:
        raise RuntimeError("stress report staging is not private")
    return (result.st_dev, result.st_ino)


def _open_staging_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
    make_private: bool = False,
) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for stress report publication")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        result = os.fstat(descriptor)
        if not stat.S_ISDIR(result.st_mode):
            raise RuntimeError("stress report staging is not a directory")
        identity = (result.st_dev, result.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise RuntimeError("stress report staging identity changed before open")
        if make_private:
            os.fchmod(descriptor, 0o700)
            result = os.fstat(descriptor)
            if (
                (result.st_dev, result.st_ino) != identity
                or stat.S_IMODE(result.st_mode) != 0o700
            ):
                raise RuntimeError("stress report staging changed during fchmod")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _staging_identity(path: Path) -> tuple[int, int]:
    descriptor = _open_staging_directory(path)
    try:
        result = os.fstat(descriptor)
        return (result.st_dev, result.st_ino)
    finally:
        os.close(descriptor)


def _has_directory_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        return _staging_identity(path) == identity
    except (OSError, RuntimeError):
        return False


def _write_tree(
    root: Path,
    report: StressValidationReport,
    evidence: Sequence[StressValidationEvidence],
    *,
    root_descriptor: int,
) -> None:
    _write_json(root_descriptor, "report.json", _report_payload(report))
    _write_text(root_descriptor, "report.md", stress_markdown(report))
    failed = {
        record.case_id: (record, item)
        for record, item in zip(report.cases, evidence, strict=True)
        if isinstance(record, FailedStressValidationRecord)
    }
    if failed:
        _mkdir_directory_at(root_descriptor, "failures")
        for case_id, (record, item) in failed.items():
            case_root = f"failures/{case_id}"
            _mkdir_directory_at(root_descriptor, case_root)
            _write_json(
                root_descriptor,
                f"{case_root}/before.json",
                item.case.scene.model_dump(mode="json"),
            )
            _write_json(
                root_descriptor,
                f"{case_root}/intervention.json",
                item.case.intervention.model_dump(mode="json"),
            )
            _write_json(
                root_descriptor,
                f"{case_root}/oracle.json",
                item.case.oracle.model_dump(mode="json"),
            )
            _write_json(
                root_descriptor,
                f"{case_root}/solver-result.json",
                _solver_result_payload(item),
            )
            _write_json(
                root_descriptor,
                f"{case_root}/checks.json",
                {
                    "case_id": case_id,
                    "checks": dict(record.checks),
                    "errors": record.errors,
                },
            )
            _write_text(
                root_descriptor,
                f"{case_root}/replay.txt",
                "PYTHONPATH=src .venv-core/bin/python "
                "scripts/stress_core_solver.py "
                f"--case-id {case_id} "
                f"--output artifacts/core-solver-stress-{case_id}\n",
            )
            if item.after_scene is not None:
                _write_json(
                    root_descriptor,
                    f"{case_root}/after.json",
                    item.after_scene.model_dump(mode="json"),
                )
                _write_bytes(
                    root_descriptor,
                    f"{case_root}/topdown.png",
                    _render_png_bytes(item.case, item.after_scene, record),
                )


def publish_stress_report(
    report: StressValidationReport,
    evidence: Sequence[StressValidationEvidence],
    output: Path,
) -> Path:
    """Atomically publish one closed deterministic stress report tree."""
    _validate_evidence_binding(report, evidence)
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
        )
    )
    published = False
    identity: tuple[int, int] | None = None
    identity_manifest: dict[str, tuple[int, int, int]] | None = None
    staging_descriptor: int | None = None
    try:
        identity = _created_staging_identity(staging)
        staging_descriptor = _open_staging_directory(
            staging,
            expected_identity=identity,
            make_private=True,
        )
        _write_tree(
            staging,
            report,
            evidence,
            root_descriptor=staging_descriptor,
        )
        identity_manifest = _verify_tree(
            staging,
            report,
            evidence,
            root_descriptor=staging_descriptor,
        )
        failure_directories = sorted(
            (
                relative
                for relative, (_, _, entry_type) in identity_manifest.items()
                if relative != "." and stat.S_ISDIR(entry_type)
            ),
            key=lambda relative: (
                -len(PurePosixPath(relative).parts),
                relative,
            ),
        )
        for relative in failure_directories:
            _fsync_tree_directory(staging_descriptor, relative)
        _fsync_tree_directory(staging_descriptor, ".")
        _fsync_directory(output.parent)
        _verify_tree(
            staging,
            report,
            evidence,
            identity_manifest,
            root_descriptor=staging_descriptor,
        )
        if not _has_directory_identity(staging, identity):
            raise RuntimeError("stress report staging ownership changed")
        _rename_no_replace(staging, output)
        if not _has_directory_identity(output, identity):
            raise RuntimeError("stress report output identity mismatch after rename")
        published = True
        _verify_tree(
            output,
            report,
            evidence,
            identity_manifest,
            root_descriptor=staging_descriptor,
        )
        if not _has_directory_identity(output, identity):
            raise RuntimeError("stress report output identity changed after verification")
        try:
            _fsync_directory(output.parent)
        except OSError as error:
            raise StressReportDurabilityError(output) from error
    except StressReportPublicationError:
        raise
    except BaseException as error:
        detail = str(error) or type(error).__name__
        recovery_staging = None
        if not published and identity is not None:
            if _has_directory_identity(output, identity):
                published = True
                recovery_staging = None
                try:
                    if identity_manifest is None or staging_descriptor is None:
                        raise RuntimeError(
                            "stress report verified tree state is unavailable"
                        )
                    _verify_tree(
                        output,
                        report,
                        evidence,
                        identity_manifest,
                        root_descriptor=staging_descriptor,
                    )
                    if not _has_directory_identity(output, identity):
                        raise RuntimeError(
                            "stress report output identity changed after verification"
                        )
                except (OSError, RuntimeError, ValueError) as verification_error:
                    verification_detail = (
                        str(verification_error) or type(verification_error).__name__
                    )
                    detail = (
                        f"{detail}; post-rename verification failed: "
                        f"{verification_detail}"
                    )
            elif _has_directory_identity(staging, identity):
                recovery_staging = staging
        raise StressReportPublicationError(
            output,
            published=published,
            staging_path=recovery_staging,
            detail=detail,
        ) from error
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
    return output
