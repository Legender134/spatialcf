"""Capture camera geometry: exact contracts and pure derivation."""

from __future__ import annotations

import math

from spatialcf.adapters.base import (
    AdapterPose,
    AdapterPosition,
)

from spatialcf.domain.scene import (
    OBB,
    Quaternion,
    Scene,
)

from spatialcf.generation.capture._models.constants import (
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8,
    _PAIR_CAMERA_DIRECTIONS,
    _PAIR_CAMERA_HORIZONS_DEGREES,
    _PAIR_CAMERA_RADIUS_M,
)


class _CameraConversionError(ValueError):
    pass


def _expected_camera_world_to_camera(
    native_position: tuple[float, float, float],
    *,
    yaw_degrees: float,
    horizon_degrees: float,
) -> tuple[float, ...]:
    """Rebuild the persisted Canonical camera matrix from adapter scalars."""

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(horizon_degrees)
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    forward = (
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        -math.sin(pitch),
    )
    up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    # The frozen scene records Canonical right-handed Z-up positions.
    position = (native_position[0], native_position[2], native_position[1])
    rows = (right, up, forward)
    translation = tuple(
        -sum(row[index] * position[index] for index in range(3)) for row in rows
    )
    return (
        *right,
        translation[0],
        *up,
        translation[1],
        *forward,
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _pair_midpoint(
    scene: Scene,
    subject_object_id: str,
    reference_object_id: str,
) -> tuple[float, float]:
    subject = scene.object_by_id(subject_object_id)
    reference = scene.object_by_id(reference_object_id)
    return (
        (subject.obb.center.x + reference.obb.center.x) / 2.0,
        (subject.obb.center.y + reference.obb.center.y) / 2.0,
    )


def _validate_pair_camera_inputs(
    scene: Scene,
    subject_object_id: str,
    reference_object_id: str,
    reachable_positions: tuple[AdapterPosition, ...],
) -> None:
    if type(scene) is not Scene:
        raise ValueError("scene must be an exact canonical Scene")
    if (
        type(subject_object_id) is not str
        or not subject_object_id
        or type(reference_object_id) is not str
        or not reference_object_id
        or subject_object_id == reference_object_id
    ):
        raise ValueError("camera pair must use two distinct non-empty object IDs")
    scene.object_by_id(subject_object_id)
    scene.object_by_id(reference_object_id)
    if (
        type(reachable_positions) is not tuple
        or not reachable_positions
        or any(type(item) is not AdapterPosition for item in reachable_positions)
    ):
        raise ValueError("reachable positions must be a non-empty exact tuple")
    if len(set(reachable_positions)) != len(reachable_positions):
        raise ValueError("reachable positions must be unique")


def _ring_positions(
    midpoint_x: float,
    midpoint_z: float,
    directions: tuple[tuple[float, float], ...],
    reachable_positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    selected: list[AdapterPosition] = []
    selected_set: set[AdapterPosition] = set()
    for direction_x, direction_z in directions:
        target_x = midpoint_x + _PAIR_CAMERA_RADIUS_M * direction_x
        target_z = midpoint_z + _PAIR_CAMERA_RADIUS_M * direction_z
        nearest = min(
            reachable_positions,
            key=lambda item: (
                (item.x - target_x) ** 2 + (item.z - target_z) ** 2,
                item.x,
                item.z,
                item.y,
            ),
        )
        if nearest not in selected_set:
            selected.append(nearest)
            selected_set.add(nearest)
    return tuple(selected)


def deterministic_pair_camera_poses(
    scene: Scene,
    subject_object_id: str,
    reference_object_id: str,
    reachable_positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPose, ...]:
    """Return at most eight frozen Tier-1 poses around one object pair."""

    _validate_pair_camera_inputs(
        scene,
        subject_object_id,
        reference_object_id,
        reachable_positions,
    )
    midpoint_x, midpoint_z = _pair_midpoint(
        scene,
        subject_object_id,
        reference_object_id,
    )
    selected = _ring_positions(
        midpoint_x,
        midpoint_z,
        _PAIR_CAMERA_DIRECTIONS,
        reachable_positions,
    )
    poses: list[AdapterPose] = []
    for position in selected:
        yaw = (
            math.degrees(math.atan2(midpoint_x - position.x, midpoint_z - position.z))
            % 360.0
        )
        for horizon in _PAIR_CAMERA_HORIZONS_DEGREES:
            poses.append(
                AdapterPose(
                    position=position,
                    yaw_degrees=yaw,
                    horizon_degrees=horizon,
                    standing=True,
                )
            )
    return tuple(poses)


def _clearance_rotation_matrix(
    rotation: Quaternion,
) -> tuple[tuple[float, float, float], ...]:
    values = (rotation.x, rotation.y, rotation.z, rotation.w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera clearance OBB rotation must be finite")
    maximum = max(abs(value) for value in values)
    if maximum == 0.0:
        raise ValueError("camera clearance OBB rotation must be nonzero")
    scaled = tuple(value / maximum for value in values)
    norm = math.sqrt(sum(value * value for value in scaled))
    x, y, z, w = (value / norm for value in scaled)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _projected_corners(obb: OBB) -> tuple[tuple[float, float], ...]:
    extents = (obb.extent.x, obb.extent.y, obb.extent.z)
    centers = (obb.center.x, obb.center.y, obb.center.z)
    if not all(math.isfinite(value) and value > 0.0 for value in extents):
        raise ValueError("camera clearance OBB extents must be finite and positive")
    if not all(math.isfinite(value) for value in centers):
        raise ValueError("camera clearance OBB center must be finite")
    rotation = _clearance_rotation_matrix(obb.rotation)
    points = set()
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                local = (
                    x_sign * extents[0] / 2.0,
                    y_sign * extents[1] / 2.0,
                    z_sign * extents[2] / 2.0,
                )
                world = tuple(
                    centers[axis]
                    + sum(rotation[axis][inner] * local[inner] for inner in range(3))
                    for axis in range(3)
                )
                points.add((world[0], world[1]))
    return tuple(sorted(points))


def _cross(
    origin: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
        right[0] - origin[0]
    )


def _convex_hull(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    unique = tuple(sorted(set(points)))
    if len(unique) < 3:
        raise ValueError("camera clearance OBB projection must have positive area")
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("camera clearance OBB projection must have positive area")
    return hull


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta = (end[0] - start[0], end[1] - start[1])
    length_squared = delta[0] * delta[0] + delta[1] * delta[1]
    if length_squared == 0.0:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * delta[0] + (point[1] - start[1]) * delta[1])
            / length_squared,
        ),
    )
    nearest = (
        start[0] + fraction * delta[0],
        start[1] + fraction * delta[1],
    )
    return math.dist(point, nearest)


def _point_polygon_distance(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> float:
    crosses = tuple(
        _cross(polygon[index], polygon[(index + 1) % len(polygon)], point)
        for index in range(len(polygon))
    )
    if all(value >= 0.0 for value in crosses) or all(value <= 0.0 for value in crosses):
        return 0.0
    return min(
        _point_segment_distance(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def _filter_competition_native_camera_positions(
    scene: Scene,
    positions: tuple[AdapterPosition, ...],
    *,
    clearance_radius_m: float,
) -> tuple[AdapterPosition, ...]:
    if type(scene) is not Scene:
        raise TypeError("camera clearance scene must be an exact Scene")
    checked_scene = Scene.model_validate(scene.model_dump(mode="python"), strict=True)
    if type(positions) is not tuple or any(
        type(position) is not AdapterPosition for position in positions
    ):
        raise TypeError("camera clearance requires an exact position tuple")
    if not positions:
        raise TypeError("camera clearance requires a non-empty exact position tuple")
    if len(set(positions)) != len(positions):
        raise ValueError("camera clearance positions must be unique")
    footprints = tuple(
        _convex_hull(_projected_corners(item.obb))
        for item in checked_scene.objects
        if item.movable
    )
    accepted = tuple(
        position
        for position in positions
        if all(
            _point_polygon_distance((position.x, position.z), footprint)
            > clearance_radius_m
            for footprint in footprints
        )
    )
    return tuple(sorted(accepted, key=lambda item: (item.x, item.z, item.y)))


def filter_competition_native_camera_positions_v2_9_5(
    scene: Scene,
    positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5,
    )


def filter_competition_native_camera_positions_v2_9_6(
    scene: Scene,
    positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6,
    )


def filter_competition_native_camera_positions_v2_9_7(
    scene: Scene,
    positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7,
    )


def filter_competition_native_camera_positions_v2_9_8(
    scene: Scene,
    positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8,
    )


def _legacy_camera(matrix: tuple[float, ...]) -> tuple[float, dict[str, float]]:
    cosine, negative_sine = matrix[0], matrix[1]
    sine, second_cosine = matrix[8], matrix[9]
    if math.isclose(math.hypot(sine, second_cosine), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise _CameraConversionError("MISSING_FACT:COMPLETE_UPRIGHT_CAMERA_DEPTH_BASIS")
    expected = (
        (matrix[2], 0.0),
        (matrix[4], 0.0),
        (matrix[5], 0.0),
        (matrix[6], 1.0),
        (matrix[10], 0.0),
        (matrix[12], 0.0),
        (matrix[13], 0.0),
        (matrix[14], 0.0),
        (matrix[15], 1.0),
        (negative_sine, -sine),
        (cosine, second_cosine),
    )
    if any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9)
        for actual, wanted in expected
    ) or not math.isclose(math.hypot(sine, cosine), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise _CameraConversionError("UNSUPPORTED_MODEL:CAMERA_NOT_EXACT_UPRIGHT")
    angle = 0.0 if sine == 0.0 else math.atan2(sine, cosine)
    return angle, {"x": matrix[3], "y": -matrix[7], "z": matrix[11]}


# Preserve supported public names and pickle lookup.
_CameraConversionError.__module__ = "spatialcf.generation.capture.models"
_expected_camera_world_to_camera.__module__ = "spatialcf.generation.capture.models"
_pair_midpoint.__module__ = "spatialcf.generation.capture.models"
_validate_pair_camera_inputs.__module__ = "spatialcf.generation.capture.models"
_ring_positions.__module__ = "spatialcf.generation.capture.models"
deterministic_pair_camera_poses.__module__ = "spatialcf.generation.capture.models"
_clearance_rotation_matrix.__module__ = "spatialcf.generation.capture.models"
_projected_corners.__module__ = "spatialcf.generation.capture.models"
_cross.__module__ = "spatialcf.generation.capture.models"
_convex_hull.__module__ = "spatialcf.generation.capture.models"
_point_segment_distance.__module__ = "spatialcf.generation.capture.models"
_point_polygon_distance.__module__ = "spatialcf.generation.capture.models"
_filter_competition_native_camera_positions.__module__ = "spatialcf.generation.capture.models"
filter_competition_native_camera_positions_v2_9_5.__module__ = "spatialcf.generation.capture.models"
filter_competition_native_camera_positions_v2_9_6.__module__ = "spatialcf.generation.capture.models"
filter_competition_native_camera_positions_v2_9_7.__module__ = "spatialcf.generation.capture.models"
filter_competition_native_camera_positions_v2_9_8.__module__ = "spatialcf.generation.capture.models"
_legacy_camera.__module__ = "spatialcf.generation.capture.models"
