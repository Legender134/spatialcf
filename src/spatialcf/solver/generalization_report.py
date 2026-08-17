"""Deterministic publication for the core solver generalization gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageDraw

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.domain.models import Scene
from spatialcf.geometry.obb import obb_footprint
from spatialcf.solver.generalization_validation import (
    FailedGeneralizationRecord,
    GeneralizationScenes,
    GeneralizationValidationError,
    GeneralizationValidationReport,
    SatGeneralizationRecord,
    UnsatGeneralizationRecord,
)

_WIDTH = 1000
_HEIGHT = 800
_MARGIN = 60
_TOP_MARGIN = 165
_PATH_DISCLAIMER = "displacement only; not a motion path"
_CLAIM_BOUNDARY = (
    "Passing this package supports only the claim that the certified solver "
    "is useful on the fixed controlled Canonical Scene matrix for LEFT to "
    "RIGHT, FRONT to BEHIND, and NEAR to FAR final-state interventions. It "
    "does not prove arbitrary-scene robustness, dataset-adapter or perception "
    "correctness, physical simulation, or collision-free motion planning. "
    "Every arrow is displacement only; not a motion path."
)


def _number(value: float) -> str:
    return f"{value:.12g}"


def generalization_markdown(report: GeneralizationValidationReport) -> str:
    """Return a deterministic human-readable projection of all 30 records."""
    passed = sum(record.result == "PASS" for record in report.cases)
    sat = [record for record in report.cases if record.expected_outcome == "SAT"]
    unsat = [record for record in report.cases if record.expected_outcome == "UNSAT"]
    sat_passed = sum(record.result == "PASS" for record in sat)
    unsat_passed = sum(record.result == "PASS" for record in unsat)
    lines = [
        f"# Core Solver Generalization Validation: {report.status}",
        "",
        (f"{passed}/30 cases passed; {sat_passed}/21 SAT; {unsat_passed}/9 UNSAT."),
        "",
        "The certified optimality and independent realized-error budget is 1e-6 m.",
        "",
        f"Movement diagrams are {_PATH_DISCLAIMER}.",
        "",
        "## Direction summary",
        "",
        "| direction | passed | expected SAT | expected UNSAT |",
        "|---|---:|---:|---:|",
    ]
    for before, after in (
        ("left", "right"),
        ("front", "behind"),
        ("near", "far"),
    ):
        records = [
            record
            for record in report.cases
            if record.case_id.startswith(
                {"left": "lr-", "front": "fb-", "near": "nf-"}[before]
            )
        ]
        lines.append(
            f"| {before} -> {after} | "
            f"{sum(record.result == 'PASS' for record in records)}/10 | "
            f"{sum(record.expected_outcome == 'SAT' for record in records)} | "
            f"{sum(record.expected_outcome == 'UNSAT' for record in records)} |"
        )

    lines.extend(
        (
            "",
            "## Satisfiable cases",
            "",
            "| case | relation | exact (m) | realized (m) | error (m) | certificate gap (m) | verifier | result |",
            "|---|---|---:|---:|---:|---:|---|---|",
        )
    )
    for record in report.cases:
        if isinstance(record, SatGeneralizationRecord):
            lines.append(
                "| "
                f"{record.case_id} | "
                f"{record.relation_before.value} -> {record.relation_after.value} | "
                f"{_number(record.exact_infimum_m)} | "
                f"{_number(record.realized_displacement_m)} | "
                f"{_number(record.realized_error_m)} | "
                f"{_number(record.certificate.optimality_gap)} | "
                f"{record.verifier_status.value} / leakage={record.leakage_count} | "
                "PASS |"
            )

    lines.extend(
        (
            "",
            "## Unsatisfiable cases",
            "",
            "| case | relation | maximum (m) | required (m) | solver reason | result |",
            "|---|---|---:|---:|---|---|",
        )
    )
    for record in report.cases:
        if isinstance(record, UnsatGeneralizationRecord):
            lines.append(
                "| "
                f"{record.case_id} | "
                f"{record.relation_before.value} -> {record.relation_after.value} | "
                f"{_number(record.maximum_possible_value_m)} | "
                f"{_number(record.required_value_m)} | "
                f"{record.reason} | PASS |"
            )

    failures = [
        record
        for record in report.cases
        if isinstance(record, FailedGeneralizationRecord)
    ]
    lines.extend(("", "## Failed checks", ""))
    if failures:
        lines.extend(
            (
                "| case | expected | solver status | reason | failed checks |",
                "|---|---|---|---|---|",
            )
        )
        for record in failures:
            lines.append(
                "| "
                f"{record.case_id} | {record.expected_outcome} | "
                f"{record.solver_status.value} | {record.reason or '-'} | "
                f"{', '.join(record.failed_checks)} |"
            )
    else:
        lines.append("None.")

    lines.extend(("", "## Independent derivations", ""))
    for record in report.cases:
        lines.extend((f"### {record.case_id}", "", record.derivation, ""))
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


def render_generalization_topdown(
    before: Scene,
    after: Scene | None,
    record: (
        SatGeneralizationRecord | UnsatGeneralizationRecord | FailedGeneralizationRecord
    ),
) -> Image.Image:
    """Render endpoint geometry while explicitly disclaiming path evidence."""
    transform = _pixel_transform(before)
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    room = [transform((point.x, point.y)) for point in before.room_polygon_xy]
    draw.polygon(room, fill="#F5F8FA", outline="#1D3557", width=4)

    display_scene = after if after is not None else before
    subject_id = "subject"
    support_id = before.object_by_id(subject_id).support_object_id
    for obj in sorted(display_scene.objects, key=lambda item: item.object_id):
        if obj.object_id == subject_id:
            continue
        polygon = _polygon_pixels(display_scene, obj.object_id, transform)
        fill = "#C5E1A5" if obj.object_id == support_id else "#B0BEC5"
        draw.polygon(polygon, fill=fill, outline="#455A64", width=3)
        center = transform((obj.position.x, obj.position.y))
        draw.text((center[0] + 7, center[1] - 14), obj.object_id, fill="#263238")

    before_polygon = _polygon_pixels(before, subject_id, transform)
    draw.line(before_polygon, fill="#277DA1", width=5, joint="curve")
    before_subject = before.object_by_id(subject_id)
    if after is not None:
        after_polygon = _polygon_pixels(after, subject_id, transform)
        draw.line(after_polygon, fill="#F94144", width=5, joint="curve")
        after_subject = after.object_by_id(subject_id)
        _arrow(
            draw,
            transform((before_subject.position.x, before_subject.position.y)),
            transform((after_subject.position.x, after_subject.position.y)),
        )

    if isinstance(record, SatGeneralizationRecord):
        if after is None:
            raise GeneralizationValidationError(
                f"{record.case_id}: SAT render scene membership mismatch"
            )
        relation = f"{record.relation_before.value} -> {record.relation_after.value}"
        measurement = (
            f"exact={_number(record.exact_infimum_m)} m  "
            f"realized={_number(record.realized_displacement_m)} m"
        )
    elif isinstance(record, UnsatGeneralizationRecord):
        if after is not None:
            raise GeneralizationValidationError(
                f"{record.case_id}: UNSAT render scene membership mismatch"
            )
        relation = f"{record.relation_before.value} -> {record.relation_after.value}"
        measurement = (
            f"NO LEGAL FINAL STATE  max={_number(record.maximum_possible_value_m)} m  "
            f"required={_number(record.required_value_m)} m"
        )
    else:
        relation = f"expected {record.expected_outcome}"
        measurement = f"VALIDATION FAILED: {', '.join(record.failed_checks)}"

    draw.text((30, 20), record.case_id, fill="#111827")
    draw.text((30, 45), relation, fill="#111827")
    draw.text((30, 70), measurement, fill="#111827")
    draw.text((30, 95), _PATH_DISCLAIMER, fill="#6A4C93")
    axis_origin = (_MARGIN, _HEIGHT - 25)
    draw.line(
        (axis_origin, (axis_origin[0] + 45, axis_origin[1])),
        fill="#111827",
        width=2,
    )
    draw.line(
        (axis_origin, (axis_origin[0], axis_origin[1] - 45)),
        fill="#111827",
        width=2,
    )
    draw.text((axis_origin[0] + 49, axis_origin[1] - 8), "+X", fill="#111827")
    draw.text((axis_origin[0] - 9, axis_origin[1] - 60), "+Y", fill="#111827")
    return image


def _validate_scene_membership(
    report: GeneralizationValidationReport,
    scenes: Mapping[str, tuple[Scene, Scene | None]],
) -> None:
    expected = tuple(record.case_id for record in report.cases)
    if set(scenes) != set(expected) or len(scenes) != len(expected):
        raise GeneralizationValidationError(
            "generalization report scene membership mismatch"
        )
    for record in report.cases:
        pair = scenes[record.case_id]
        if type(pair) is not tuple or len(pair) != 2:
            raise GeneralizationValidationError(
                f"{record.case_id}: generalization report scene membership malformed"
            )
        before, after = pair
        if not isinstance(before, Scene) or before.scene_id != record.case_id:
            raise GeneralizationValidationError(
                f"{record.case_id}: generalization report scene membership mismatch"
            )
        if isinstance(record, SatGeneralizationRecord):
            if not isinstance(after, Scene) or after.scene_id != record.case_id:
                raise GeneralizationValidationError(
                    f"{record.case_id}: generalization report scene membership mismatch"
                )
        elif isinstance(record, UnsatGeneralizationRecord):
            if after is not None:
                raise GeneralizationValidationError(
                    f"{record.case_id}: generalization report scene membership mismatch"
                )
        elif after is not None and (
            record.expected_outcome == "UNSAT"
            or after.scene_id != record.case_id
        ):
            raise GeneralizationValidationError(
                f"{record.case_id}: generalization report scene membership mismatch"
            )


def _prepare_output(output_root: Path) -> None:
    if output_root.is_symlink():
        raise FileExistsError(f"output path is a symlink: {output_root}")
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"output directory is non-empty: {output_root}")
        return
    output_root.mkdir(parents=True)


def _write_scene(path: Path, scene: Scene) -> None:
    path.write_bytes(canonical_json_bytes(scene.model_dump(mode="json"), pretty=True))


def _verify_output(
    root: Path,
    report: GeneralizationValidationReport,
    scenes: Mapping[str, tuple[Scene, Scene | None]],
) -> None:
    parsed = GeneralizationValidationReport.model_validate_json(
        (root / "report.json").read_bytes()
    )
    if parsed != report:
        raise GeneralizationValidationError("generalization report JSON mismatch")
    if (root / "report.md").read_text(encoding="utf-8") != generalization_markdown(
        report
    ):
        raise GeneralizationValidationError("generalization report Markdown mismatch")
    expected = {"report.json", "report.md"}
    for record in report.cases:
        directory = root / record.case_id
        Scene.model_validate_json((directory / "before.json").read_bytes())
        expected.add(f"{record.case_id}/before.json")
        _, after = scenes[record.case_id]
        if after is not None:
            Scene.model_validate_json((directory / "after.json").read_bytes())
            expected.add(f"{record.case_id}/after.json")
        with Image.open(directory / "topdown.png") as image:
            image.verify()
        with Image.open(directory / "topdown.png") as image:
            if image.mode != "RGB" or image.size != (_WIDTH, _HEIGHT):
                raise GeneralizationValidationError(
                    f"{record.case_id}: generalization image contract mismatch"
                )
        expected.add(f"{record.case_id}/topdown.png")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise GeneralizationValidationError(
            "generalization report file set is not closed"
        )


def publish_generalization_report(
    report: GeneralizationValidationReport,
    scenes: GeneralizationScenes,
    output_root: Path,
) -> Path:
    """Write the exact report tree without replacing any prior result."""
    _validate_scene_membership(report, scenes)
    _prepare_output(output_root)
    (output_root / "report.json").write_bytes(
        canonical_json_bytes(report.model_dump(mode="json"), pretty=True)
    )
    (output_root / "report.md").write_text(
        generalization_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    records = {record.case_id: record for record in report.cases}
    for case_id, record in records.items():
        before, after = scenes[case_id]
        case_root = output_root / case_id
        case_root.mkdir()
        _write_scene(case_root / "before.json", before)
        if after is not None:
            _write_scene(case_root / "after.json", after)
        render_generalization_topdown(before, after, record).save(
            case_root / "topdown.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
    _verify_output(output_root, report, scenes)
    return output_root
