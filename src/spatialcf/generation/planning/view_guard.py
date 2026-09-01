"""Pure authenticated sampled-view admission guard."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spatialcf.domain.edit import CanonicalEdit
from spatialcf.domain.request import InterventionSpec, Relation
from spatialcf.domain.scene import OBB, Scene, Vec3
from spatialcf.domain.serialization import canonical_sha256
from spatialcf.generation.capture.models import SourceViewFact
from spatialcf.generation.planning.models import (
    _SOURCE_VIEW_GUARD_HASH_DOMAIN,
    SourceViewGuard,
    SourceViewObjectProxy,
)
from spatialcf.relations.engine import RelationEngine, ground_gap

_QUANTIZATION_M = 1e-5


@dataclass(frozen=True, slots=True)
class _ProjectedSample:
    object_id: str
    row: int
    column: int
    depth_lower: float
    depth_upper: float
    weight: int


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _rotation(obb: OBB) -> np.ndarray:
    q = obb.rotation
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source-view OBB rotation must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("source-view OBB rotation must be nonzero")
    x, y, z, w = (value / norm for value in values)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _translated_obb(obb: OBB, dx: float, dy: float) -> OBB:
    return obb.model_copy(
        update={
            "center": Vec3(
                x=obb.center.x + dx,
                y=obb.center.y + dy,
                z=obb.center.z,
            )
        }
    )


def _obb_corners(obb: OBB) -> tuple[np.ndarray, ...]:
    center = np.asarray((obb.center.x, obb.center.y, obb.center.z), dtype=np.float64)
    half = np.asarray((obb.extent.x, obb.extent.y, obb.extent.z), dtype=np.float64) / 2
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)) or np.any(half <= 0):
        raise ValueError("source-view OBB must be finite and nondegenerate")
    rotation = _rotation(obb)
    return tuple(
        center + rotation @ (half * np.asarray((sx, sy, sz), dtype=np.float64))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    )


def _camera_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = matrix @ np.asarray((*point, 1.0), dtype=np.float64)
    if not np.all(np.isfinite(homogeneous)) or homogeneous[3] == 0.0:
        raise ArithmeticError("nonfinite_projection")
    return homogeneous[:3] / homogeneous[3]


def _pixel(camera, camera_point: np.ndarray) -> tuple[float, float, float]:
    x, y, depth = (float(value) for value in camera_point)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ArithmeticError("behind_camera")
    fx, fy, cx, cy = (
        camera.intrinsics[0],
        camera.intrinsics[4],
        camera.intrinsics[2],
        camera.intrinsics[5],
    )
    values = (fx, fy, cx, cy)
    if not all(math.isfinite(value) for value in values) or fx <= 0.0 or fy <= 0.0:
        raise ArithmeticError("invalid_intrinsics")
    return fx * x / depth + cx, cy - fy * y / depth, depth


def _ray_hits_obb_before(
    origin: np.ndarray,
    endpoint: np.ndarray,
    obb: OBB,
) -> bool:
    rotation = _rotation(obb)
    center = np.asarray((obb.center.x, obb.center.y, obb.center.z), dtype=np.float64)
    half = np.asarray((obb.extent.x, obb.extent.y, obb.extent.z), dtype=np.float64) / 2
    local_origin = rotation.T @ (origin - center)
    local_direction = rotation.T @ (endpoint - origin)
    lower, upper = 0.0, 1.0
    for axis in range(3):
        direction = float(local_direction[axis])
        if direction == 0.0:
            if local_origin[axis] < -half[axis] or local_origin[axis] > half[axis]:
                return False
            continue
        first = (-half[axis] - local_origin[axis]) / direction
        second = (half[axis] - local_origin[axis]) / direction
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower > upper:
            return False
    return lower <= upper and lower <= 1.0


def _outer_projection(camera, matrix: np.ndarray, obb: OBB) -> tuple[float, float]:
    pixels = tuple(_pixel(camera, _camera_point(matrix, corner)) for corner in _obb_corners(obb))
    xs = tuple(item[0] for item in pixels)
    ys = tuple(item[1] for item in pixels)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    area = (xmax - xmin) * (ymax - ymin)
    if not math.isfinite(area) or area <= 0.0:
        raise ArithmeticError("degenerate_outer_projection")
    clipped_width = max(0.0, min(xmax, camera.width) - max(xmin, 0.0))
    clipped_height = max(0.0, min(ymax, camera.height) - max(ymin, 0.0))
    return area, 1.0 - clipped_width * clipped_height / area


def _fallback(object_id: str) -> SourceViewObjectProxy:
    return SourceViewObjectProxy(
        object_id=object_id,
        bbox_center_x_lower=0.0,
        bbox_center_x_upper=0.0,
        camera_depth_lower_m=0.0,
        camera_depth_upper_m=0.0,
        image_area_fraction_lower=0.0,
        visible_fraction_lower=0.0,
        truncated_fraction_upper=1.0,
    )


def _zbuffer(
    projected: tuple[_ProjectedSample, ...],
) -> tuple[tuple[_ProjectedSample, ...], tuple[str, ...]]:
    grouped: dict[tuple[int, int], list[_ProjectedSample]] = {}
    for sample in projected:
        grouped.setdefault((sample.row, sample.column), []).append(sample)
    survivors: list[_ProjectedSample] = []
    reasons: list[str] = []
    for cell_samples in grouped.values():
        if len(cell_samples) == 1:
            survivors.append(cell_samples[0])
            continue
        winners = tuple(
            candidate
            for candidate in cell_samples
            if all(
                candidate is other
                or candidate.depth_upper < other.depth_lower
                for other in cell_samples
            )
        )
        if len(winners) == 1:
            survivors.append(winners[0])
        else:
            reasons.append("coverage:depth_interval_tie_or_overlap")
    return tuple(survivors), tuple(reasons)


def _guard(
    *,
    fact: SourceViewFact,
    semantic_problem_sha256: str,
    solve_result_sha256: str,
    edit: CanonicalEdit,
    subject: SourceViewObjectProxy,
    reference: SourceViewObjectProxy,
    target_relation: Relation,
    target_satisfied: bool,
    old_relation_satisfied: bool,
    reasons: tuple[str, ...],
) -> SourceViewGuard:
    payload = {
        "guard_version": "competition-native-source-view-guard:2.9.5",
        "status": "PASSED" if not reasons else "UNCERTIFIED",
        "source_view_fact_sha256": fact.source_view_fact_sha256,
        "semantic_problem_sha256": semantic_problem_sha256,
        "solve_result_sha256": solve_result_sha256,
        "edit_sha256": edit.edit_sha256,
        "subject": subject,
        "reference": reference,
        "target_relation": target_relation,
        "target_satisfied": target_satisfied,
        "old_relation_satisfied": old_relation_satisfied,
        "reasons": tuple(sorted(set(reasons))),
    }
    return SourceViewGuard(
        **payload,
        source_view_guard_sha256=canonical_sha256(
            payload, domain=_SOURCE_VIEW_GUARD_HASH_DOMAIN
        ),
    )


def _definite_relation(
    relation: Relation,
    subject: SourceViewObjectProxy,
    reference: SourceViewObjectProxy,
    scene: Scene,
    subject_obb: OBB,
    reference_obb: OBB,
) -> tuple[bool, bool]:
    tolerance = RelationEngine.COMPARISON_TOLERANCE

    def at_least(value: float, threshold: float) -> bool:
        return value >= threshold or math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=tolerance
        )

    def below(value: float, threshold: float) -> bool:
        return value < threshold and not math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=tolerance
        )

    if relation is Relation.LEFT:
        lower = reference.bbox_center_x_lower - subject.bbox_center_x_upper
        upper = reference.bbox_center_x_upper - subject.bbox_center_x_lower
        threshold = scene.camera_by_id("main").width * RelationEngine.LEFT_RIGHT_FRACTION
    elif relation is Relation.RIGHT:
        lower = subject.bbox_center_x_lower - reference.bbox_center_x_upper
        upper = subject.bbox_center_x_upper - reference.bbox_center_x_lower
        threshold = scene.camera_by_id("main").width * RelationEngine.LEFT_RIGHT_FRACTION
    elif relation is Relation.FRONT:
        lower = reference.camera_depth_lower_m - subject.camera_depth_upper_m
        upper = reference.camera_depth_upper_m - subject.camera_depth_lower_m
        threshold = RelationEngine.FRONT_BEHIND_METERS
    elif relation is Relation.BEHIND:
        lower = subject.camera_depth_lower_m - reference.camera_depth_upper_m
        upper = subject.camera_depth_upper_m - reference.camera_depth_lower_m
        threshold = RelationEngine.FRONT_BEHIND_METERS
    else:
        gap = ground_gap(subject_obb, reference_obb)
        if relation is Relation.NEAR:
            return (
                gap <= RelationEngine.NEAR_METERS,
                gap > RelationEngine.NEAR_METERS,
            )
        return (
            gap >= RelationEngine.FAR_METERS,
            gap < RelationEngine.FAR_METERS,
        )
    return at_least(lower, threshold), below(upper, threshold)


def evaluate_source_view_guard(
    scene: Scene,
    intervention: InterventionSpec,
    fact: SourceViewFact,
    edit: CanonicalEdit,
    *,
    semantic_problem_sha256: str,
    solve_result_sha256: str,
) -> SourceViewGuard:
    """Evaluate one bound endpoint without importing a platform adapter."""

    if type(scene) is not Scene:
        raise TypeError("source-view guard scene must be exact")
    if type(intervention) is not InterventionSpec:
        raise TypeError("source-view guard intervention must be exact")
    if type(fact) is not SourceViewFact:
        raise TypeError("source-view guard fact must be exact")
    if type(edit) is not CanonicalEdit:
        raise TypeError("source-view guard edit must be exact")
    semantic_problem_sha256 = _digest(
        semantic_problem_sha256, "source-view semantic problem digest"
    )
    solve_result_sha256 = _digest(
        solve_result_sha256, "source-view solve result digest"
    )
    SourceViewFact.model_validate(fact.model_dump(mode="python"), strict=True)
    if edit.semantic_problem_sha256 != semantic_problem_sha256:
        raise ValueError("source-view edit does not bind semantic problem")
    if edit.subject_id != intervention.subject_id:
        raise ValueError("source-view edit does not bind intervention subject")
    if intervention.camera_id != "main":
        raise ValueError("source-view guard requires the main camera")
    try:
        subject_object = scene.object_by_id(intervention.subject_id)
        reference_object = scene.object_by_id(intervention.reference_id)
        camera = scene.camera_by_id("main")
    except KeyError as error:
        raise ValueError("source-view guard object/camera binding is incomplete") from error
    if subject_object.object_id == reference_object.object_id:
        raise ValueError("source-view guard objects must be distinct")
    if fact.scene_id != scene.scene_id:
        raise ValueError("source-view fact scene identity changed")
    binding_domain = "spatialcf.source-view-binding.v1"
    if fact.scene_sha256 != canonical_sha256(scene, domain=binding_domain):
        raise ValueError("source-view fact does not bind scene")
    if fact.camera_sha256 != canonical_sha256(camera, domain=binding_domain):
        raise ValueError("source-view fact does not bind main camera")
    rows = {row.object_id: row for row in fact.objects}
    positive_ids: list[str] = []
    for item in scene.objects:
        view = item.views.get("main")
        if view is not None and view.camera_id != "main":
            raise ValueError("source-view main view camera identity changed")
        if view is not None and view.visible_fraction > 0.0:
            positive_ids.append(item.object_id)
    if tuple(rows) != tuple(sorted(positive_ids)):
        raise ValueError("source-view fact object roster does not bind main views")
    for row in fact.objects:
        view = scene.object_by_id(row.object_id).views["main"]
        actual_bbox = (
            row.source_mask_bbox.xmin,
            row.source_mask_bbox.ymin,
            row.source_mask_bbox.xmax,
            row.source_mask_bbox.ymax,
        )
        expected_bbox = (
            view.bbox.xmin,
            view.bbox.ymin,
            view.bbox.xmax,
            view.bbox.ymax,
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in zip(actual_bbox, expected_bbox, strict=True)
        ):
            raise ValueError("source-view fact bbox does not bind main view")
        if (
            row.source_mask_bbox.xmin < 0.0
            or row.source_mask_bbox.ymin < 0.0
            or row.source_mask_bbox.xmax > camera.width
            or row.source_mask_bbox.ymax > camera.height
            or any(value < 0 or value >= camera.height for value in row.sample_rows)
            or any(value < 0 or value >= camera.width for value in row.sample_columns)
        ):
            raise ValueError("source-view fact pixels exceed main camera bounds")
    if {intervention.subject_id, intervention.reference_id} - set(rows):
        raise ValueError("source-view fact lacks intervention rows")
    try:
        for item in scene.objects:
            _obb_corners(item.obb)
    except ValueError as error:
        raise ValueError("source-view scene OBB is malformed") from error

    reasons: list[str] = []
    try:
        matrix = np.asarray(camera.world_to_camera, dtype=np.float64).reshape(4, 4)
        inverse = np.linalg.inv(matrix)
        camera_origin_h = inverse @ np.asarray((0.0, 0.0, 0.0, 1.0))
        if camera_origin_h[3] == 0.0:
            raise ArithmeticError("invalid camera homogeneous coordinate")
        camera_origin = camera_origin_h[:3] / camera_origin_h[3]
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(camera_origin)):
            raise ArithmeticError("nonfinite_camera")
    except (ValueError, np.linalg.LinAlgError, ArithmeticError, ZeroDivisionError):
        return _guard(
            fact=fact,
            semantic_problem_sha256=semantic_problem_sha256,
            solve_result_sha256=solve_result_sha256,
            edit=edit,
            subject=_fallback(intervention.subject_id),
            reference=_fallback(intervention.reference_id),
            target_relation=intervention.relation_after,
            target_satisfied=False,
            old_relation_satisfied=True,
            reasons=("projection:camera_transform_invalid",),
        )

    dx, dy = edit.translation_xy_m.x, edit.translation_xy_m.y
    moved_subject_obb = _translated_obb(subject_object.obb, dx, dy)
    obbs = {
        item.object_id: (
            moved_subject_obb if item.object_id == subject_object.object_id else item.obb
        )
        for item in scene.objects
    }
    projected: list[_ProjectedSample] = []
    for row in fact.objects:
        obj = scene.object_by_id(row.object_id)
        translation = (dx, dy) if row.object_id == subject_object.object_id else (0.0, 0.0)
        for qx, qy, qz, weight in zip(
            row.local_x_quantized,
            row.local_y_quantized,
            row.local_z_quantized,
            row.sample_weights,
            strict=True,
        ):
            corners = tuple(
                np.asarray(
                    (
                        obj.position.x + translation[0] + (qx + sx * 0.5) * _QUANTIZATION_M,
                        obj.position.y + translation[1] + (qy + sy * 0.5) * _QUANTIZATION_M,
                        obj.position.z + (qz + sz * 0.5) * _QUANTIZATION_M,
                    ),
                    dtype=np.float64,
                )
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            )
            try:
                pixels = tuple(_pixel(camera, _camera_point(matrix, corner)) for corner in corners)
            except ArithmeticError as error:
                reasons.append(f"projection:{error.args[0]}")
                continue
            cells = {(round(pixel[1]), round(pixel[0])) for pixel in pixels}
            if len(cells) != 1:
                reasons.append("projection:quantization_cell_straddle")
                continue
            row_px, column_px = next(iter(cells))
            if not (0 <= column_px < camera.width and 0 <= row_px < camera.height):
                reasons.append("projection:sample_out_of_frame")
                continue
            try:
                blocked = any(
                    _ray_hits_obb_before(camera_origin, corner, blocker)
                    for corner in corners
                    for blocker_id, blocker in obbs.items()
                    if blocker_id != row.object_id
                )
            except ValueError:
                reasons.append("coverage:blocker_invalid")
                continue
            if blocked:
                continue
            depths = tuple(pixel[2] for pixel in pixels)
            projected.append(
                _ProjectedSample(
                    object_id=row.object_id,
                    row=row_px,
                    column=column_px,
                    depth_lower=min(depths),
                    depth_upper=max(depths),
                    weight=weight,
                )
            )

    survivors, zbuffer_reasons = _zbuffer(tuple(projected))
    reasons.extend(zbuffer_reasons)

    proxies: dict[str, SourceViewObjectProxy] = {}
    for object_id in (intervention.subject_id, intervention.reference_id):
        object_samples = tuple(item for item in survivors if item.object_id == object_id)
        try:
            outer_area, truncation = _outer_projection(camera, matrix, obbs[object_id])
            scene_object = scene.object_by_id(object_id)
            translation = (dx, dy) if object_id == subject_object.object_id else (0.0, 0.0)
            anchor = _camera_point(
                matrix,
                np.asarray(
                    (
                        scene_object.position.x + translation[0],
                        scene_object.position.y + translation[1],
                        scene_object.position.z,
                    ),
                    dtype=np.float64,
                ),
            )
            if not math.isfinite(float(anchor[2])) or anchor[2] <= 0.0:
                raise ArithmeticError("behind_camera_anchor")
        except (ArithmeticError, ValueError):
            reasons.append(f"projection:{object_id}:outer_invalid")
            proxies[object_id] = _fallback(object_id)
            continue
        if not object_samples:
            reasons.append(f"coverage:{object_id}:empty")
            proxies[object_id] = _fallback(object_id)
            continue
        columns = tuple(item.column for item in object_samples)
        rows_px = tuple(item.row for item in object_samples)
        bbox_area = (max(columns) - min(columns) + 1) * (
            max(rows_px) - min(rows_px) + 1
        )
        center = (min(columns) + max(columns) + 1) / 2.0
        proxies[object_id] = SourceViewObjectProxy(
            object_id=object_id,
            bbox_center_x_lower=center,
            bbox_center_x_upper=center,
            camera_depth_lower_m=float(anchor[2]),
            camera_depth_upper_m=float(anchor[2]),
            image_area_fraction_lower=min(1.0, bbox_area / (camera.width * camera.height)),
            visible_fraction_lower=min(
                1.0, sum(item.weight for item in object_samples) / outer_area
            ),
            truncated_fraction_upper=min(1.0, max(0.0, truncation)),
        )

    subject_proxy = proxies[intervention.subject_id]
    reference_proxy = proxies[intervention.reference_id]
    target_true, _ = _definite_relation(
        intervention.relation_after,
        subject_proxy,
        reference_proxy,
        scene,
        moved_subject_obb,
        reference_object.obb,
    )
    _, old_false = _definite_relation(
        intervention.relation_before,
        subject_proxy,
        reference_proxy,
        scene,
        moved_subject_obb,
        reference_object.obb,
    )
    old_satisfied = not old_false
    for label, proxy in (("subject", subject_proxy), ("reference", reference_proxy)):
        if proxy.visible_fraction_lower < RelationEngine.MIN_VISIBLE_FRACTION:
            reasons.append(f"visibility:{label}:visible_fraction")
        if proxy.image_area_fraction_lower < RelationEngine.MIN_IMAGE_AREA_FRACTION:
            reasons.append(f"visibility:{label}:image_area_fraction")
        if proxy.truncated_fraction_upper > RelationEngine.MAX_TRUNCATED_FRACTION:
            reasons.append(f"visibility:{label}:truncated_fraction")
    if not target_true:
        reasons.append("relation:target_not_definite")
    if not old_false:
        reasons.append("relation:old_not_definitely_false")
    return _guard(
        fact=fact,
        semantic_problem_sha256=semantic_problem_sha256,
        solve_result_sha256=solve_result_sha256,
        edit=edit,
        subject=subject_proxy,
        reference=reference_proxy,
        target_relation=intervention.relation_after,
        target_satisfied=target_true,
        old_relation_satisfied=old_satisfied,
        reasons=tuple(reasons),
    )


__all__ = ("evaluate_source_view_guard",)
