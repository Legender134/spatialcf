"""Human-auditable publication for independent core solver validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.domain.models import Scene
from spatialcf.geometry.obb import obb_footprint
from spatialcf.solver.validation import (
    CaseValidationRecord,
    CoreValidationError,
    ValidationReport,
)


_WIDTH = 1000
_HEIGHT = 800
_MARGIN = 60
_TOP_MARGIN = 120
_CLAIM_BOUNDARY = (
    "Passing this package supports only the claim that the certified solver "
    "works on three transparent Canonical Scene problems and returns "
    "independently verified motions within its declared continuous optimality "
    "tolerance. It does not prove robustness to arbitrary scenes, correctness "
    "of a dataset adapter, perceptual accuracy, photorealism, or physical "
    "simulation fidelity."
)


def _number(value: float) -> str:
    return f"{value:.12g}"


def validation_markdown(report: ValidationReport) -> str:
    """Return a deterministic human-readable projection of a passing report."""
    lines = [
        "# Core Solver Independent Validation: PASS",
        "",
        (
            "All three canonical directions passed independent replay. The "
            "declared continuous optimality tolerance is 1e-6 m."
        ),
        "",
        "| case | relation | exact infimum (m) | realized (m) | error (m) | certificate [lower, upper] | gap (m) | verifier | result |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for case in report.cases:
        certificate = case.certificate
        lines.append(
            "| "
            f"{case.case_id} | "
            f"{case.relation_before.value} → {case.relation_after.value} | "
            f"{_number(case.exact_infimum_m)} | "
            f"{_number(case.realized_displacement_m)} | "
            f"{_number(case.realized_error_m)} | "
            f"[{_number(certificate.distance_lower_bound)}, "
            f"{_number(certificate.distance_upper_bound)}] | "
            f"{_number(certificate.optimality_gap)} | "
            f"{case.verifier_status.value} / leakage={case.leakage_count} | PASS |"
        )
    lines.extend(("", "## Closed-form derivations", ""))
    for case in report.cases:
        lines.extend((f"### {case.case_id}", "", case.derivation, ""))
    lines.extend(("## Claim boundary", "", _CLAIM_BOUNDARY, ""))
    return "\n".join(lines)


def _pixel_transform(scene: Scene):
    xs = [point.x for point in scene.room_polygon_xy]
    ys = [point.y for point in scene.room_polygon_xy]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    scale = min(
        (_WIDTH - 2 * _MARGIN) / span_x,
        (_HEIGHT - _TOP_MARGIN - _MARGIN) / span_y,
    )

    def transform(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            round(_MARGIN + (x - min(xs)) * scale),
            round(_HEIGHT - _MARGIN - (y - min(ys)) * scale),
        )

    return transform


def _polygon_pixels(scene: Scene, object_id: str, transform) -> list[tuple[int, int]]:
    footprint = obb_footprint(scene.object_by_id(object_id).obb)
    return [transform((float(x), float(y))) for x, y in footprint.exterior.coords]


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((start, end), fill="#6A4C93", width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 15.0
    spread = math.pi / 7.0
    first = (
        round(end[0] - length * math.cos(angle - spread)),
        round(end[1] - length * math.sin(angle - spread)),
    )
    second = (
        round(end[0] - length * math.cos(angle + spread)),
        round(end[1] - length * math.sin(angle + spread)),
    )
    draw.polygon((end, first, second), fill="#6A4C93")


def render_validation_topdown(
    before: Scene,
    after: Scene,
    record: CaseValidationRecord,
) -> Image.Image:
    """Render explanatory room, OBB, and movement geometry for one case."""
    transform = _pixel_transform(before)
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    room = [transform((point.x, point.y)) for point in before.room_polygon_xy]
    draw.polygon(room, fill="#F5F8FA", outline="#1D3557", width=4)

    subject_id = "subject"
    for obj in sorted(after.objects, key=lambda item: item.object_id):
        if obj.object_id == subject_id:
            continue
        polygon = _polygon_pixels(after, obj.object_id, transform)
        draw.polygon(polygon, fill="#B0BEC5", outline="#455A64", width=3)
        center = transform((obj.position.x, obj.position.y))
        draw.text((center[0] + 7, center[1] - 14), obj.object_id, fill="#263238")

    before_polygon = _polygon_pixels(before, subject_id, transform)
    after_polygon = _polygon_pixels(after, subject_id, transform)
    draw.line(before_polygon, fill="#277DA1", width=5, joint="curve")
    draw.line(after_polygon, fill="#F94144", width=5, joint="curve")
    before_subject = before.object_by_id(subject_id)
    after_subject = after.object_by_id(subject_id)
    before_center = transform((before_subject.position.x, before_subject.position.y))
    after_center = transform((after_subject.position.x, after_subject.position.y))
    _arrow(draw, before_center, after_center)

    draw.text((30, 20), record.case_id, fill="#111827")
    draw.text(
        (30, 45),
        f"{record.relation_before.value} -> {record.relation_after.value}",
        fill="#111827",
    )
    draw.text(
        (30, 70),
        (
            f"exact={_number(record.exact_infimum_m)} m  "
            f"realized={_number(record.realized_displacement_m)} m"
        ),
        fill="#111827",
    )
    axis_origin = (_MARGIN, _HEIGHT - 25)
    draw.line((axis_origin, (axis_origin[0] + 45, axis_origin[1])), fill="#111827", width=2)
    draw.line((axis_origin, (axis_origin[0], axis_origin[1] - 45)), fill="#111827", width=2)
    draw.text((axis_origin[0] + 49, axis_origin[1] - 8), "+X", fill="#111827")
    draw.text((axis_origin[0] - 9, axis_origin[1] - 60), "+Y", fill="#111827")
    return image


def _validate_scene_membership(
    report: ValidationReport,
    scenes: Mapping[str, tuple[Scene, Scene]],
) -> None:
    expected = tuple(case.case_id for case in report.cases)
    if set(scenes) != set(expected) or len(scenes) != len(expected):
        raise CoreValidationError("report scene membership mismatch")
    for case_id in expected:
        pair = scenes[case_id]
        if type(pair) is not tuple or len(pair) != 2:
            raise CoreValidationError(f"{case_id}: report scene membership malformed")
        before, after = pair
        if not isinstance(before, Scene) or not isinstance(after, Scene):
            raise CoreValidationError(f"{case_id}: report scenes must be Scene values")
        if before.scene_id != case_id or after.scene_id != case_id:
            raise CoreValidationError(f"{case_id}: report scene membership mismatch")


def _prepare_output(output_root: Path) -> None:
    if output_root.is_symlink():
        raise FileExistsError(f"output path is a symlink: {output_root}")
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"output directory is non-empty: {output_root}")
        return
    output_root.mkdir(parents=True)


def _verify_output(root: Path, report: ValidationReport) -> None:
    ValidationReport.model_validate_json((root / "report.json").read_bytes())
    (root / "report.md").read_text(encoding="utf-8")
    expected = {"report.json", "report.md"}
    for case in report.cases:
        directory = root / case.case_id
        Scene.model_validate_json((directory / "before.json").read_bytes())
        Scene.model_validate_json((directory / "after.json").read_bytes())
        with Image.open(directory / "topdown.png") as image:
            image.verify()
        expected.update(
            {
                f"{case.case_id}/before.json",
                f"{case.case_id}/after.json",
                f"{case.case_id}/topdown.png",
            }
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise CoreValidationError("published report file set is not closed")


def publish_validation_report(
    report: ValidationReport,
    scenes: Mapping[str, tuple[Scene, Scene]],
    output_root: Path,
) -> Path:
    """Write the exact report tree without replacing prior output."""
    _validate_scene_membership(report, scenes)
    _prepare_output(output_root)
    (output_root / "report.json").write_bytes(
        canonical_json_bytes(report.model_dump(mode="json"), pretty=True)
    )
    (output_root / "report.md").write_text(
        validation_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    records = {case.case_id: case for case in report.cases}
    for case_id in records:
        before, after = scenes[case_id]
        directory = output_root / case_id
        directory.mkdir()
        (directory / "before.json").write_bytes(
            canonical_json_bytes(before.model_dump(mode="json"), pretty=True)
        )
        (directory / "after.json").write_bytes(
            canonical_json_bytes(after.model_dump(mode="json"), pretty=True)
        )
        render_validation_topdown(before, after, records[case_id]).save(
            directory / "topdown.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
    _verify_output(output_root, report)
    return output_root
