"""Solver-independent rigid transformations for stress-case drafts."""

from __future__ import annotations

import math
from typing import TypeAlias

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import Camera, Quaternion, Scene, SceneObject, Vec2, Vec3
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.stress.models import (
    SatStressOracle,
    StressCaseDraft,
    StressTransform,
    UnsatStressOracle,
)

_XYMatrix: TypeAlias = tuple[int, int, int, int]


def _multiply(left: _XYMatrix, right: _XYMatrix) -> _XYMatrix:
    l00, l01, l10, l11 = left
    r00, r01, r10, r11 = right
    return (
        l00 * r00 + l01 * r10,
        l00 * r01 + l01 * r11,
        l10 * r00 + l11 * r10,
        l10 * r01 + l11 * r11,
    )


def _linear_matrix(transform: StressTransform) -> _XYMatrix:
    mirror = {
        "none": (1, 0, 0, 1),
        "camera_horizontal": (-1, 0, 0, 1),
        "camera_depth": (1, 0, 0, -1),
    }[transform.mirror]
    rotation = {
        0: (1, 0, 0, 1),
        90: (0, -1, 1, 0),
        180: (-1, 0, 0, -1),
        270: (0, 1, -1, 0),
    }[transform.rotation_degrees]
    return _multiply(rotation, mirror)


def _transform_xy(point: Vec2, matrix: _XYMatrix, translation: Vec2) -> Vec2:
    a00, a01, a10, a11 = matrix
    return Vec2(
        x=a00 * point.x + a01 * point.y + translation.x,
        y=a10 * point.x + a11 * point.y + translation.y,
    )


def _transform_position(
    point: Vec3,
    matrix: _XYMatrix,
    translation: Vec2,
) -> Vec3:
    transformed = _transform_xy(Vec2(x=point.x, y=point.y), matrix, translation)
    return Vec3(x=transformed.x, y=transformed.y, z=point.z)


def _canonical_yaw(rotation: Quaternion, matrix: _XYMatrix) -> Quaternion:
    cosine = rotation.w * rotation.w - rotation.z * rotation.z
    sine = 2.0 * rotation.w * rotation.z
    a00, a01, a10, a11 = matrix
    axis_x = a00 * cosine + a01 * sine
    axis_y = a10 * cosine + a11 * sine
    half_angle = math.atan2(axis_y, axis_x) / 2.0
    z = math.sin(half_angle)
    w = math.cos(half_angle)
    if abs(w) < 1e-15 and z < 0.0:
        z, w = -z, -w
    if abs(z) < 1e-15:
        z = 0.0
    if abs(w) < 1e-15:
        w = 0.0
    return Quaternion(x=0.0, y=0.0, z=z, w=w)


def _transform_camera(
    camera: Camera,
    linear: _XYMatrix,
    translation: Vec2,
) -> Camera:
    a00, a01, a10, a11 = linear
    matrix = list(camera.world_to_camera)
    for row in range(4):
        start = 4 * row
        old_x = camera.world_to_camera[start]
        old_y = camera.world_to_camera[start + 1]
        new_x = old_x * a00 + old_y * a01
        new_y = old_x * a10 + old_y * a11
        matrix[start] = 0.0 if new_x == 0.0 else new_x
        matrix[start + 1] = 0.0 if new_y == 0.0 else new_y
        new_offset = (
            camera.world_to_camera[start + 3]
            - new_x * translation.x
            - new_y * translation.y
        )
        matrix[start + 3] = 0.0 if new_offset == 0.0 else new_offset
    return camera.model_copy(update={"world_to_camera": tuple(matrix)})


def _transform_object(
    obj: SceneObject,
    linear: _XYMatrix,
    translation: Vec2,
) -> SceneObject:
    position = _transform_position(obj.position, linear, translation)
    center = _transform_position(obj.obb.center, linear, translation)
    rotation = _canonical_yaw(obj.rotation, linear)
    obb_rotation = _canonical_yaw(obj.obb.rotation, linear)
    return obj.model_copy(
        update={
            "position": position,
            "rotation": rotation,
            "obb": obj.obb.model_copy(
                update={"center": center, "rotation": obb_rotation}
            ),
        }
    )


def _signed_area(points: tuple[Vec2, ...]) -> float:
    return 0.5 * sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(points, (*points[1:], points[0]), strict=True)
    )


def _derivation_suffix(transform: StressTransform) -> str:
    return (
        "; rigid transform "
        f"translation=({transform.translation_xy.x:.12g},"
        f"{transform.translation_xy.y:.12g}), "
        f"mirror={transform.mirror}, rotation={transform.rotation_degrees}deg, "
        f"base={transform.base_case_digest or 'none'}"
    )


def _target_unit_normal(draft: StressCaseDraft, scene: Scene) -> Vec2 | None:
    relation = draft.intervention.relation_after
    camera = scene.camera_by_id(draft.intervention.camera_id)
    matrix = camera.world_to_camera
    if relation in {Relation.LEFT, Relation.RIGHT}:
        reference = scene.object_by_id(draft.intervention.reference_id)
        reference_view = reference.views[draft.intervention.camera_id]
        direction = 1.0 if relation is Relation.RIGHT else -1.0
        fx, _, cx, _, _, _, _, _, _ = camera.intrinsics
        target_u = reference_view.bbox.center_x + direction * (
            camera.width * RelationEngine.LEFT_RIGHT_FRACTION
        )
        pixel_offset = target_u - cx
        normal_x = direction * (fx * matrix[0] - pixel_offset * matrix[8])
        normal_y = direction * (fx * matrix[1] - pixel_offset * matrix[9])
    elif relation in {Relation.FRONT, Relation.BEHIND}:
        direction = 1.0 if relation is Relation.BEHIND else -1.0
        normal_x = direction * matrix[8]
        normal_y = direction * matrix[9]
    else:
        return None
    norm = math.hypot(normal_x, normal_y)
    if norm <= 1e-12:
        raise ValueError("stress transform target has a zero half-space normal")
    return Vec2(x=normal_x / norm, y=normal_y / norm)


def apply_stress_transform(
    draft: StressCaseDraft,
    transform: StressTransform,
) -> StressCaseDraft:
    """Apply one declared rigid world-XY transform to a stress draft."""
    translation = transform.translation_xy
    linear = _linear_matrix(transform)
    room = tuple(
        _transform_xy(point, linear, translation)
        for point in draft.scene.room_polygon_xy
    )
    if _signed_area(room) < 0.0:
        room = tuple(reversed(room))
    scene = draft.scene.model_copy(
        update={
            "room_polygon_xy": room,
            "cameras": tuple(
                _transform_camera(camera, linear, translation)
                for camera in draft.scene.cameras
            ),
            "objects": tuple(
                _transform_object(obj, linear, translation)
                for obj in draft.scene.objects
            ),
        }
    )
    oracle = draft.oracle
    if isinstance(oracle, SatStressOracle):
        oracle = SatStressOracle(
            proof_kind=oracle.proof_kind,
            exact_infimum_m=oracle.exact_infimum_m,
            exact_infimum_points=tuple(
                _transform_xy(point, linear, translation)
                for point in oracle.exact_infimum_points
            ),
            derivation=oracle.derivation + _derivation_suffix(transform),
        )
    elif isinstance(oracle, UnsatStressOracle):
        target_normal = _target_unit_normal(draft, scene)
        scalar_shift = (
            0.0
            if target_normal is None
            else (target_normal.x * translation.x + target_normal.y * translation.y)
        )
        oracle = UnsatStressOracle(
            proof_kind=oracle.proof_kind,
            maximum_possible_value_m=round(
                oracle.maximum_possible_value_m + scalar_shift,
                12,
            ),
            required_value_m=round(
                oracle.required_value_m + scalar_shift,
                12,
            ),
            expected_reason=oracle.expected_reason,
            derivation=oracle.derivation + _derivation_suffix(transform),
        )
    payload = draft.model_dump(mode="python")
    payload.update(
        {
            "scene": scene,
            "oracle": oracle,
            "transform": transform,
        }
    )
    return StressCaseDraft.model_validate(payload)
