"""Human-auditable publication for core solver challenge validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.domain.models import Scene
from spatialcf.geometry.obb import obb_footprint
from spatialcf.solver.challenge_validation import (
    ChallengeScenes,
    ChallengeValidationError,
    ChallengeValidationReport,
    SatChallengeRecord,
    UnsatChallengeRecord,
)


_WIDTH = 1000
_HEIGHT = 800
_MARGIN = 60
_TOP_MARGIN = 145
_PATH_DISCLAIMER = "displacement only; not a motion path"
_CLAIM_BOUNDARY = (
    "Passing this package supports only the claim that the certified solver "
    "respects one explicit obstacle, preserves one finite support contact, "
    "and proves one bounded FAR target infeasible on transparent Canonical "
    "Scene inputs. It does not prove arbitrary-scene robustness, dataset "
    "adapter correctness, physical simulation, or collision-free motion "
    "planning. Every arrow is displacement only; not a motion path."
)


def _number(value: float) -> str:
    return f"{value:.12g}"


def challenge_markdown(report: ChallengeValidationReport) -> str:
    """Return a deterministic readable projection of a passing report."""
    lines = [
        "# Core Solver Challenge Validation: PASS",
        "",
        (
            "Two constrained final states and one certified infeasible target "
            "passed independent replay."
        ),
        "",
        f"Movement diagrams are {_PATH_DISCLAIMER}.",
        "",
        "## Satisfiable cases",
        "",
        "| case | relation | exact infimum (m) | realized (m) | error (m) | certificate [lower, upper] | gap (m) | verifier | result |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for record in report.cases:
        if not isinstance(record, SatChallengeRecord):
            continue
        certificate = record.certificate
        lines.append(
            "| "
            f"{record.case_id} | "
            f"{record.relation_before.value} -> {record.relation_after.value} | "
            f"{_number(record.exact_infimum_m)} | "
            f"{_number(record.realized_displacement_m)} | "
            f"{_number(record.realized_error_m)} | "
            f"[{_number(certificate.distance_lower_bound)}, "
            f"{_number(certificate.distance_upper_bound)}] | "
            f"{_number(certificate.optimality_gap)} | "
            f"{record.verifier_status.value} / leakage={record.leakage_count} | "
            "PASS |"
        )
    lines.extend(
        (
            "",
            "## Unsatisfiable case",
            "",
            "| case | relation | maximum achievable gap (m) | required gap (m) | solver reason | result |",
            "|---|---|---:|---:|---|---|",
        )
    )
    for record in report.cases:
        if not isinstance(record, UnsatChallengeRecord):
            continue
        lines.append(
            "| "
            f"{record.case_id} | "
            f"{record.relation_before.value} -> {record.relation_after.value} | "
            f"{_number(record.maximum_possible_gap_m)} | "
            f"{_number(record.required_gap_m)} | "
            f"{record.reason} | PASS |"
        )
    lines.extend(("", "## Closed-form derivations", ""))
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


def _polygon_pixels(
    scene: Scene,
    object_id: str,
    transform,
) -> list[tuple[int, int]]:
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


def render_challenge_topdown(
    before: Scene,
    after: Scene | None,
    record: SatChallengeRecord | UnsatChallengeRecord,
) -> Image.Image:
    """Render explanatory endpoint geometry without implying a motion path."""
    transform = _pixel_transform(before)
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    room = [transform((point.x, point.y)) for point in before.room_polygon_xy]
    draw.polygon(room, fill="#F5F8FA", outline="#1D3557", width=4)

    display_scene = after if after is not None else before
    subject_id = "subject"
    for obj in sorted(display_scene.objects, key=lambda item: item.object_id):
        if obj.object_id == subject_id:
            continue
        polygon = _polygon_pixels(display_scene, obj.object_id, transform)
        fill = "#C5E1A5" if obj.object_id == "support" else "#B0BEC5"
        draw.polygon(polygon, fill=fill, outline="#455A64", width=3)
        center = transform((obj.position.x, obj.position.y))
        draw.text((center[0] + 7, center[1] - 14), obj.object_id, fill="#263238")

    before_polygon = _polygon_pixels(before, subject_id, transform)
    draw.line(before_polygon, fill="#277DA1", width=5, joint="curve")
    before_subject = before.object_by_id(subject_id)
    if isinstance(record, SatChallengeRecord):
        if after is None:
            raise ChallengeValidationError(
                f"{record.case_id}: SAT render scene membership mismatch"
            )
        after_polygon = _polygon_pixels(after, subject_id, transform)
        draw.line(after_polygon, fill="#F94144", width=5, joint="curve")
        after_subject = after.object_by_id(subject_id)
        before_center = transform(
            (before_subject.position.x, before_subject.position.y)
        )
        after_center = transform((after_subject.position.x, after_subject.position.y))
        _arrow(draw, before_center, after_center)
        draw.text((30, 95), _PATH_DISCLAIMER, fill="#6A4C93")
        measurement = (
            f"exact={_number(record.exact_infimum_m)} m  "
            f"realized={_number(record.realized_displacement_m)} m"
        )
    else:
        if after is not None:
            raise ChallengeValidationError(
                f"{record.case_id}: UNSAT render scene membership mismatch"
            )
        draw.text((30, 95), "NO LEGAL FINAL STATE", fill="#C62828")
        measurement = (
            f"max gap={_number(record.maximum_possible_gap_m)} m  "
            f"required={_number(record.required_gap_m)} m"
        )

    draw.text((30, 20), record.case_id, fill="#111827")
    draw.text(
        (30, 45),
        f"{record.relation_before.value} -> {record.relation_after.value}",
        fill="#111827",
    )
    draw.text((30, 70), measurement, fill="#111827")
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
    report: ChallengeValidationReport,
    scenes: Mapping[str, tuple[Scene, Scene | None]],
) -> None:
    expected = tuple(record.case_id for record in report.cases)
    if set(scenes) != set(expected) or len(scenes) != len(expected):
        raise ChallengeValidationError("challenge report scene membership mismatch")
    for record in report.cases:
        pair = scenes[record.case_id]
        if type(pair) is not tuple or len(pair) != 2:
            raise ChallengeValidationError(
                f"{record.case_id}: challenge report scene membership malformed"
            )
        before, after = pair
        if not isinstance(before, Scene) or before.scene_id != record.case_id:
            raise ChallengeValidationError(
                f"{record.case_id}: challenge report scene membership mismatch"
            )
        if isinstance(record, SatChallengeRecord):
            if not isinstance(after, Scene) or after.scene_id != record.case_id:
                raise ChallengeValidationError(
                    f"{record.case_id}: challenge report scene membership mismatch"
                )
        elif after is not None:
            raise ChallengeValidationError(
                f"{record.case_id}: challenge report scene membership mismatch"
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


def _verify_output(root: Path, report: ChallengeValidationReport) -> None:
    ChallengeValidationReport.model_validate_json((root / "report.json").read_bytes())
    (root / "report.md").read_text(encoding="utf-8")
    expected = {"report.json", "report.md"}
    for record in report.cases:
        directory = root / record.case_id
        Scene.model_validate_json((directory / "before.json").read_bytes())
        expected.add(f"{record.case_id}/before.json")
        if isinstance(record, SatChallengeRecord):
            Scene.model_validate_json((directory / "after.json").read_bytes())
            expected.add(f"{record.case_id}/after.json")
        with Image.open(directory / "topdown.png") as image:
            image.verify()
        expected.add(f"{record.case_id}/topdown.png")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ChallengeValidationError("challenge report file set is not closed")


def publish_challenge_report(
    report: ChallengeValidationReport,
    scenes: ChallengeScenes,
    output_root: Path,
) -> Path:
    """Write the exact challenge report tree without replacing prior output."""
    _validate_scene_membership(report, scenes)
    _prepare_output(output_root)
    (output_root / "report.json").write_bytes(
        canonical_json_bytes(report.model_dump(mode="json"), pretty=True)
    )
    (output_root / "report.md").write_text(
        challenge_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    records = {record.case_id: record for record in report.cases}
    for case_id, record in records.items():
        before, after = scenes[case_id]
        case_root = output_root / case_id
        case_root.mkdir()
        _write_scene(case_root / "before.json", before)
        if isinstance(record, SatChallengeRecord):
            assert after is not None
            _write_scene(case_root / "after.json", after)
        render_challenge_topdown(before, after, record).save(
            case_root / "topdown.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
    _verify_output(output_root, report)
    return output_root
