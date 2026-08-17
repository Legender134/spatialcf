"""Deterministic source-only camera evidence for current capture."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import (
    AI2ThorAgentPose,
    AI2ThorCameraApplication,
    AI2ThorNativePosition,
    AI2ThorObservation,
    ai2thor_camera_world_to_camera,
)
from spatialcf.domain.models import OBB, Camera, Quaternion, Scene, Vec3
from spatialcf.domain.v2.base import FiniteFloat, Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.geometry.transforms import ai2thor_position_to_world
from spatialcf.relations.engine import RelationEngine

_PAIR_CAMERA_RADIUS_M = 2.0
_PAIR_CAMERA_DIRECTIONS = (
    (0.0, -1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
)
_PAIR_CAMERA_HORIZONS_DEGREES = (0.0, 30.0)
_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5 = 0.2
_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6 = 0.21
_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7 = 0.25
_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8 = 0.25 + math.sqrt(2.0) * 0.5e-6


class _CameraConversionError(ValueError):
    pass


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
    reachable_positions: tuple[AI2ThorNativePosition, ...],
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
        or any(type(item) is not AI2ThorNativePosition for item in reachable_positions)
    ):
        raise ValueError("reachable positions must be a non-empty exact tuple")
    if len(set(reachable_positions)) != len(reachable_positions):
        raise ValueError("reachable positions must be unique")


def _ring_positions(
    midpoint_x: float,
    midpoint_z: float,
    directions: tuple[tuple[float, float], ...],
    reachable_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    selected: list[AI2ThorNativePosition] = []
    selected_set: set[AI2ThorNativePosition] = set()
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
    reachable_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorAgentPose, ...]:
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
    poses: list[AI2ThorAgentPose] = []
    for position in selected:
        yaw = (
            math.degrees(math.atan2(midpoint_x - position.x, midpoint_z - position.z))
            % 360.0
        )
        for horizon in _PAIR_CAMERA_HORIZONS_DEGREES:
            poses.append(
                AI2ThorAgentPose(
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
    positions: tuple[AI2ThorNativePosition, ...],
    *,
    clearance_radius_m: float,
) -> tuple[AI2ThorNativePosition, ...]:
    if type(scene) is not Scene:
        raise TypeError("camera clearance scene must be an exact Scene")
    checked_scene = Scene.model_validate(scene.model_dump(mode="python"), strict=True)
    if type(positions) is not tuple or any(
        type(position) is not AI2ThorNativePosition for position in positions
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
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5,
    )


def filter_competition_native_camera_positions_v2_9_6(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6,
    )


def filter_competition_native_camera_positions_v2_9_7(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7,
    )


def filter_competition_native_camera_positions_v2_9_8(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
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


_MAX_POSE_BANK_MEMBERS = 256
_MAX_POSITION_RESIDUAL_M = 1e-5
_MAX_ANGLE_RESIDUAL_DEGREES = 1e-4
_POLICY_HASH_DOMAIN = "spatialcf.competition-native-camera-policy.v2.9.3"
_POSE_BANK_HASH_DOMAIN = "spatialcf.competition-native-camera-pose-bank.v2.9.3"
_EVIDENCE_HASH_DOMAIN = "spatialcf.competition-native-source-camera-evidence.v2.9.3"
_PLACEMENT_ROSTER_HASH_DOMAIN = (
    "spatialcf.competition-native-camera-placement-roster.v2.9.4"
)
_LEGACY_POSE_POLICY_VERSION = "deterministic-pair-camera-tier-1:1"
_SOLVER_UPRIGHT_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright:2"
)
_EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain:3"
)
_COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.2m:4"
)
_CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m:5"
)
_RESET_PER_POSE_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m-reset-per-pose:6"
)
_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m:7"
)
_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused:8"
)
_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused-final-settle:9"
)


def _editable_pose_policy_versions() -> frozenset[str]:
    return frozenset(
        {
            _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
            _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
            _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
            _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
            _GRID_MARGIN_EDITABLE_POLICY_VERSION,
            _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
            _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
        }
    )


class CompetitionNativeCameraPoseV2_9_3(V2Model):
    """One exact native TeleportFull pose on the canonical evidence wire."""

    pose_version: Literal["competition-native-camera-pose:2.9.3"] = (
        "competition-native-camera-pose:2.9.3"
    )
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    yaw_degrees: FiniteFloat
    horizon_degrees: FiniteFloat
    standing: bool


def _policy_payload(
    pose_policy_version: str = _LEGACY_POSE_POLICY_VERSION,
) -> dict[str, object]:
    if pose_policy_version not in {
        _LEGACY_POSE_POLICY_VERSION,
        _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
        _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
        _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
        _GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    }:
        raise ValueError("camera evidence pose policy version is unsupported")
    return {
        "camera_id": "main",
        "maximum_pose_bank_count": 256,
        "maximum_truncated_fraction": 0.5,
        "minimum_image_area_fraction": 0.0025,
        "minimum_visible_fraction": 0.2,
        "policy_version": "competition-native-camera-selection-policy:2.9.3",
        "pose_policy_version": pose_policy_version,
    }


class CompetitionNativeCameraPolicyV2_9_3(V2Model):
    """Frozen literal source-camera selection policy and its own digest."""

    policy_version: Literal["competition-native-camera-selection-policy:2.9.3"] = (
        "competition-native-camera-selection-policy:2.9.3"
    )
    pose_policy_version: Literal[
        "deterministic-pair-camera-tier-1:1",
        "deterministic-pair-camera-tier-1-solver-upright:2",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain:3",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.2m:4",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.21m:5",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.21m-reset-per-pose:6",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m:7",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m-physics-paused:8",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m-physics-paused-final-settle:9",
    ] = _LEGACY_POSE_POLICY_VERSION
    camera_id: Literal["main"] = "main"
    maximum_pose_bank_count: Literal[256] = 256
    minimum_visible_fraction: Literal[0.2] = 0.2
    minimum_image_area_fraction: Literal[0.0025] = 0.0025
    maximum_truncated_fraction: Literal[0.5] = 0.5
    policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy_digest(self) -> Self:
        expected = canonical_sha256_v2(
            _policy_payload(self.pose_policy_version),
            domain=_POLICY_HASH_DOMAIN,
        )
        if self.policy_sha256 != expected:
            raise ValueError("camera evidence policy digest mismatch")
        return self


class CompetitionNativeCameraScoreV2_9_3(V2Model):
    """The two literal source-only counts used by camera selection."""

    score_version: Literal["competition-native-camera-score:2.9.3"] = (
        "competition-native-camera-score:2.9.3"
    )
    movable_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)
    all_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_score_counts(self) -> Self:
        if (
            self.movable_scene_unique_category_qualifying_count
            > self.all_scene_unique_category_qualifying_count
        ):
            raise ValueError("camera evidence movable score exceeds total score")
        return self


class CompetitionNativeCameraPlacementPositionV2_9_4(V2Model):
    """One exact native subject anchor considered by camera selection."""

    position_version: Literal["competition-native-camera-placement-position:2.9.4"] = (
        "competition-native-camera-placement-position:2.9.4"
    )
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class CompetitionNativeCameraPlacementRosterEntryV2_9_4(V2Model):
    """One source subject and its complete canonical native placement roster."""

    entry_version: Literal["competition-native-camera-placement-roster-entry:2.9.4"] = (
        "competition-native-camera-placement-roster-entry:2.9.4"
    )
    subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    positions: tuple[CompetitionNativeCameraPlacementPositionV2_9_4, ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_positions(self) -> Self:
        ordered = tuple(
            sorted(
                set(self.positions),
                key=lambda item: (item.x, item.z, item.y),
            )
        )
        if self.positions != ordered:
            raise ValueError("camera placement positions must be unique and canonical")
        return self


class CompetitionNativeCameraScoreV2_9_4(V2Model):
    """Source visibility plus the native edit domain visible from one pose."""

    score_version: Literal["competition-native-camera-score:2.9.4"] = (
        "competition-native-camera-score:2.9.4"
    )
    placement_roster_sha256: Sha256Digest
    visible_native_placement_subject_count: int = Field(strict=True, ge=0)
    visible_native_placement_count: int = Field(strict=True, ge=0)
    movable_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)
    all_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_score_counts(self) -> Self:
        if (
            self.movable_scene_unique_category_qualifying_count
            > self.all_scene_unique_category_qualifying_count
            or self.visible_native_placement_subject_count
            > self.movable_scene_unique_category_qualifying_count
            or self.visible_native_placement_subject_count
            > self.visible_native_placement_count
        ):
            raise ValueError("editable camera score counts are inconsistent")
        return self


CompetitionNativeCameraScoreFamilyV2_9_3 = (
    CompetitionNativeCameraScoreV2_9_3 | CompetitionNativeCameraScoreV2_9_4
)


def _strict_placement_roster_v2_9_4(
    placement_roster: object,
) -> tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...]:
    if type(placement_roster) is not tuple or any(
        type(item) is not CompetitionNativeCameraPlacementRosterEntryV2_9_4
        for item in placement_roster
    ):
        raise TypeError("camera placement roster must be an exact entry tuple")
    checked = tuple(
        CompetitionNativeCameraPlacementRosterEntryV2_9_4.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in placement_roster
    )
    if not checked:
        raise ValueError("camera placement roster must not be empty")
    if tuple(
        sorted(checked, key=lambda item: item.subject_object_id)
    ) != checked or len({item.subject_object_id for item in checked}) != len(checked):
        raise ValueError("camera placement roster must be unique and canonical")
    return checked


def competition_native_camera_placement_roster_sha256_v2_9_4(
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
) -> Sha256Digest:
    """Hash one exact source placement roster in its independent domain."""

    checked = _strict_placement_roster_v2_9_4(placement_roster)
    payload = {
        "placement_roster_version": "competition-native-camera-placement-roster:2.9.4",
        "subjects": tuple(item.model_dump(mode="json") for item in checked),
    }
    return canonical_sha256_v2(payload, domain=_PLACEMENT_ROSTER_HASH_DOMAIN)


def select_competition_native_camera_score_index_v2_9_4(
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_4, ...],
) -> int:
    """Select edit-domain coverage, then source visibility, then bank index."""

    if (
        type(pose_scores) is not tuple
        or not pose_scores
        or any(
            type(item) is not CompetitionNativeCameraScoreV2_9_4 for item in pose_scores
        )
    ):
        raise TypeError("editable camera score ledger must be an exact nonempty tuple")
    checked = tuple(
        CompetitionNativeCameraScoreV2_9_4.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in pose_scores
    )
    if (
        len(checked) > _MAX_POSE_BANK_MEMBERS
        or len({item.placement_roster_sha256 for item in checked}) != 1
    ):
        raise ValueError("editable camera score ledger is not source-aligned")
    return min(
        range(len(checked)),
        key=lambda index: (
            -checked[index].visible_native_placement_subject_count,
            -checked[index].visible_native_placement_count,
            -checked[index].movable_scene_unique_category_qualifying_count,
            -checked[index].all_scene_unique_category_qualifying_count,
            index,
        ),
    )


def _strict_score_ledger(
    pose_scores: object,
) -> tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...]:
    if type(pose_scores) is not tuple:
        raise TypeError("camera evidence score ledger must be an exact score tuple")
    if not pose_scores or len(pose_scores) > _MAX_POSE_BANK_MEMBERS:
        raise ValueError(
            "camera evidence score ledger must contain 1 through 256 scores"
        )
    score_type: type[CompetitionNativeCameraScoreFamilyV2_9_3]
    if all(type(item) is CompetitionNativeCameraScoreV2_9_3 for item in pose_scores):
        score_type = CompetitionNativeCameraScoreV2_9_3
    elif all(type(item) is CompetitionNativeCameraScoreV2_9_4 for item in pose_scores):
        score_type = CompetitionNativeCameraScoreV2_9_4
    else:
        raise TypeError("camera evidence score ledger mixes score versions")
    return tuple(
        score_type.model_validate(item.model_dump(mode="python"), strict=True)
        for item in pose_scores
    )


def select_competition_native_camera_score_index_v2_9_3(
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_3, ...],
) -> int:
    """Select the literal score argmax, breaking complete ties by bank index."""

    checked = _strict_score_ledger(pose_scores)
    if any(type(item) is not CompetitionNativeCameraScoreV2_9_3 for item in checked):
        raise TypeError("legacy camera selector requires 2.9.3 scores")
    return min(
        range(len(checked)),
        key=lambda index: (
            -checked[index].movable_scene_unique_category_qualifying_count,
            -checked[index].all_scene_unique_category_qualifying_count,
            index,
        ),
    )


def _select_competition_native_camera_score_index(
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...],
) -> int:
    checked = _strict_score_ledger(pose_scores)
    if type(checked[0]) is CompetitionNativeCameraScoreV2_9_3:
        return select_competition_native_camera_score_index_v2_9_3(checked)  # type: ignore[arg-type]
    return select_competition_native_camera_score_index_v2_9_4(checked)  # type: ignore[arg-type]


def verify_competition_native_camera_observation_binding_v2_9_3(
    requested_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_native_camera_position: tuple[float, float, float],
    camera: Camera,
) -> None:
    """Close persisted native observation fields to one requested main Camera."""

    if (
        type(requested_pose) is not CompetitionNativeCameraPoseV2_9_3
        or type(observed_pose) is not CompetitionNativeCameraPoseV2_9_3
    ):
        raise TypeError(
            "camera evidence camera observation binding poses must be exact"
        )
    if (
        type(observed_native_camera_position) is not tuple
        or len(observed_native_camera_position) != 3
        or any(type(item) is not float for item in observed_native_camera_position)
    ):
        raise TypeError(
            "camera evidence camera observation binding native position must be exact"
        )
    if type(camera) is not Camera or camera.camera_id != "main":
        raise TypeError(
            "camera evidence camera observation binding Camera must be exact main"
        )
    position_residual_m = math.dist(
        (requested_pose.x, requested_pose.y, requested_pose.z),
        (observed_pose.x, observed_pose.y, observed_pose.z),
    )
    yaw_residual_degrees = abs(
        (observed_pose.yaw_degrees - requested_pose.yaw_degrees + 180.0) % 360.0 - 180.0
    )
    horizon_residual_degrees = abs(
        observed_pose.horizon_degrees - requested_pose.horizon_degrees
    )
    if (
        position_residual_m > _MAX_POSITION_RESIDUAL_M
        or yaw_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or horizon_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or observed_pose.standing is not requested_pose.standing
    ):
        raise ValueError(
            "camera evidence camera observation binding does not close requested pose"
        )
    native_position = Vec3(
        x=observed_native_camera_position[0],
        y=observed_native_camera_position[1],
        z=observed_native_camera_position[2],
    )
    expected_world_to_camera = ai2thor_camera_world_to_camera(
        ai2thor_position_to_world(native_position),
        yaw_degrees=observed_pose.yaw_degrees,
        horizon_degrees=observed_pose.horizon_degrees,
    )
    if camera.world_to_camera != expected_world_to_camera:
        raise ValueError(
            "camera evidence camera observation binding does not close main Camera"
        )


def _evidence_payload(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    policy_sha256: str,
    pose_bank_sha256: str,
    pose_bank_count: int,
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...],
    selected_pose_index: int,
    requested_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_native_camera_position: tuple[float, float, float],
    camera: Camera,
    score: CompetitionNativeCameraScoreFamilyV2_9_3,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
) -> dict[str, object]:
    return {
        "camera": camera.model_dump(mode="json"),
        "depth_npy_sha256": depth_npy_sha256,
        "evidence_version": "competition-native-source-camera-evidence:2.9.3",
        "instance_png_sha256": instance_png_sha256,
        "is_scene_at_rest": is_scene_at_rest,
        "observed_native_camera_position": observed_native_camera_position,
        "observed_pose": observed_pose.model_dump(mode="json"),
        "pointcloud_ply_sha256": pointcloud_ply_sha256,
        "policy_sha256": policy_sha256,
        "pose_bank_count": pose_bank_count,
        "pose_bank_sha256": pose_bank_sha256,
        "pose_scores": tuple(item.model_dump(mode="json") for item in pose_scores),
        "requested_pose": requested_pose.model_dump(mode="json"),
        "rgb_png_sha256": rgb_png_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "scene_id": scene_id,
        "score": score.model_dump(mode="json"),
        "selected_pose_index": selected_pose_index,
        "source_capture_sha256": source_capture_sha256,
        "source_id": source_id,
        "source_locator_sha256": source_locator_sha256,
    }


class CompetitionNativeSourceCameraEvidenceV2_9_3(V2Model):
    """One selected source camera with complete immutable capture lineage."""

    evidence_version: Literal["competition-native-source-camera-evidence:2.9.3"] = (
        "competition-native-source-camera-evidence:2.9.3"
    )
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_locator_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    pose_bank_sha256: Sha256Digest
    pose_bank_count: int = Field(strict=True, ge=1, le=256)
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...] = Field(
        min_length=1, max_length=256
    )
    selected_pose_index: int = Field(strict=True, ge=0, le=255)
    requested_pose: CompetitionNativeCameraPoseV2_9_3
    observed_pose: CompetitionNativeCameraPoseV2_9_3
    observed_native_camera_position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    camera: Camera
    score: CompetitionNativeCameraScoreFamilyV2_9_3
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    pointcloud_ply_sha256: Sha256Digest
    is_scene_at_rest: bool
    camera_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        checked_scores = _strict_score_ledger(self.pose_scores)
        if len(checked_scores) != self.pose_bank_count:
            raise ValueError("camera evidence score ledger is not bank-aligned")
        if self.selected_pose_index >= self.pose_bank_count:
            raise ValueError("camera evidence selected index is outside pose bank")
        if (
            self.selected_pose_index
            != _select_competition_native_camera_score_index(checked_scores)
            or self.score != checked_scores[self.selected_pose_index]
        ):
            raise ValueError("camera evidence selected index is not the literal argmax")
        if self.camera.camera_id != "main":
            raise ValueError("camera evidence must persist the main camera")
        verify_competition_native_camera_observation_binding_v2_9_3(
            self.requested_pose,
            self.observed_pose,
            self.observed_native_camera_position,
            self.camera,
        )
        expected_policies = {
            canonical_sha256_v2(
                _policy_payload(version),
                domain=_POLICY_HASH_DOMAIN,
            )
            for version in (
                _LEGACY_POSE_POLICY_VERSION,
                _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
                _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
                _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
                _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
                _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
                _GRID_MARGIN_EDITABLE_POLICY_VERSION,
                _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
                _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
            )
        }
        if self.policy_sha256 not in expected_policies:
            raise ValueError("camera evidence policy digest mismatch")
        editable_policy_sha256s = {
            canonical_sha256_v2(
                _policy_payload(version),
                domain=_POLICY_HASH_DOMAIN,
            )
            for version in _editable_pose_policy_versions()
        }
        if (type(checked_scores[0]) is CompetitionNativeCameraScoreV2_9_4) != (
            self.policy_sha256 in editable_policy_sha256s
        ):
            raise ValueError("camera evidence score version does not bind policy")
        expected = canonical_sha256_v2(
            _evidence_payload(
                source_id=self.source_id,
                scene_id=self.scene_id,
                source_locator_sha256=self.source_locator_sha256,
                runtime_identity_sha256=self.runtime_identity_sha256,
                source_capture_sha256=self.source_capture_sha256,
                policy_sha256=self.policy_sha256,
                pose_bank_sha256=self.pose_bank_sha256,
                pose_bank_count=self.pose_bank_count,
                pose_scores=checked_scores,
                selected_pose_index=self.selected_pose_index,
                requested_pose=self.requested_pose,
                observed_pose=self.observed_pose,
                observed_native_camera_position=self.observed_native_camera_position,
                camera=self.camera,
                score=self.score,
                rgb_png_sha256=self.rgb_png_sha256,
                depth_npy_sha256=self.depth_npy_sha256,
                instance_png_sha256=self.instance_png_sha256,
                pointcloud_ply_sha256=self.pointcloud_ply_sha256,
                is_scene_at_rest=self.is_scene_at_rest,
            ),
            domain=_EVIDENCE_HASH_DOMAIN,
        )
        if self.camera_evidence_sha256 != expected:
            raise ValueError("camera evidence digest mismatch")
        return self


def _strict_scene(scene: object) -> Scene:
    if type(scene) is not Scene:
        raise TypeError("camera evidence scene must be an exact Scene")
    return Scene.model_validate(scene.model_dump(mode="python"), strict=True)


def _strict_native_pose(pose: object, *, label: str) -> AI2ThorAgentPose:
    if type(pose) is not AI2ThorAgentPose:
        raise TypeError(f"{label} must be an exact AI2ThorAgentPose")
    if type(pose.position) is not AI2ThorNativePosition:
        raise TypeError(f"{label} position must be exact")
    return AI2ThorAgentPose(
        position=AI2ThorNativePosition(**asdict(pose.position)),
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _wire_pose(pose: AI2ThorAgentPose) -> CompetitionNativeCameraPoseV2_9_3:
    return CompetitionNativeCameraPoseV2_9_3(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _pose_key(
    pose: CompetitionNativeCameraPoseV2_9_3,
) -> tuple[float, float, float, float, float, bool]:
    return (
        pose.x,
        pose.y,
        pose.z,
        pose.yaw_degrees,
        pose.horizon_degrees,
        pose.standing,
    )


def build_competition_native_camera_pose_bank_v2_9_3(
    scene: Scene,
    pairs: tuple[tuple[str, str], ...],
    reachable_positions: tuple[AI2ThorNativePosition, ...],
    fallback_pose: AI2ThorAgentPose,
    *,
    policy: CompetitionNativeCameraPolicyV2_9_3 | None = None,
) -> tuple[CompetitionNativeCameraPoseV2_9_3, ...]:
    """Build the bounded, permutation-invariant Tier-1 source pose bank."""

    checked_scene = _strict_scene(scene)
    if type(pairs) is not tuple or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not str or not item for item in pair)
        for pair in pairs
    ):
        raise TypeError("camera evidence pairs must be an exact tuple of string pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("camera evidence pairs must be unique")
    if (
        type(reachable_positions) is not tuple
        or not reachable_positions
        or any(type(item) is not AI2ThorNativePosition for item in reachable_positions)
    ):
        raise TypeError("reachable positions must be a non-empty exact tuple")
    checked_positions = tuple(
        AI2ThorNativePosition(**asdict(item)) for item in reachable_positions
    )
    if len(set(checked_positions)) != len(checked_positions):
        raise ValueError("reachable positions must be unique")
    checked_fallback = _strict_native_pose(fallback_pose, label="fallback pose")
    checked_policy = (
        build_competition_native_camera_policy_v2_9_3()
        if policy is None
        else _strict_policy(policy)
    )
    solver_upright = checked_policy.pose_policy_version in {
        _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        *_editable_pose_policy_versions(),
    }
    if (
        checked_policy.pose_policy_version
        == _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION
    ):
        checked_positions = filter_competition_native_camera_positions_v2_9_5(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no collision-safe reachable positions"
            )
    elif checked_policy.pose_policy_version in {
        _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
        _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
    }:
        checked_positions = filter_competition_native_camera_positions_v2_9_6(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no contact-margin-safe reachable positions"
            )
    elif checked_policy.pose_policy_version in {
        _GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    }:
        checked_positions = filter_competition_native_camera_positions_v2_9_7(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no grid-margin-safe reachable positions"
            )
    elif (
        checked_policy.pose_policy_version
        == _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION
    ):
        checked_positions = filter_competition_native_camera_positions_v2_9_8(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no quantization-safe reachable positions"
            )

    poses: set[CompetitionNativeCameraPoseV2_9_3] = set()
    for subject_object_id, support_object_id in pairs:
        subject = checked_scene.object_by_id(subject_object_id)
        checked_scene.object_by_id(support_object_id)
        if (
            not subject.movable
            or subject.support_object_id != support_object_id
            or subject_object_id == support_object_id
        ):
            raise ValueError("camera evidence pair does not bind movable support")
        generated = deterministic_pair_camera_poses(
            checked_scene,
            subject_object_id,
            support_object_id,
            checked_positions,
        )
        poses.update(
            _wire_pose(item)
            for item in generated
            if not solver_upright or item.horizon_degrees == 0.0
        )
        if len(poses) > _MAX_POSE_BANK_MEMBERS:
            raise ValueError("camera evidence pose bank exceeds 256 members")

    if not poses:
        if solver_upright:
            checked_fallback = AI2ThorAgentPose(
                position=checked_fallback.position,
                yaw_degrees=checked_fallback.yaw_degrees,
                horizon_degrees=0.0,
                standing=checked_fallback.standing,
            )
        poses.add(_wire_pose(checked_fallback))
    return tuple(sorted(poses, key=_pose_key))


def build_competition_native_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the one frozen 2.9.3 source-camera selection policy."""

    payload = _policy_payload(_LEGACY_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_solver_upright_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the bounded camera policy supported by the certified solver."""

    payload = _policy_payload(_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_editable_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the upright policy that ranks source-native edit coverage."""

    payload = _policy_payload(_EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_collision_safe_editable_camera_policy_v2_9_5() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the source-only edit policy with fixed 0.2m movable clearance."""

    payload = _policy_payload(_COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_contact_margin_editable_camera_policy_v2_9_6() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the edit policy with 0.2m agent plus fixed 1cm margin."""

    payload = _policy_payload(_CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_reset_per_pose_editable_camera_policy_v2_9_7() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the contact-safe policy that resets the source per pose."""

    payload = _policy_payload(_RESET_PER_POSE_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_grid_margin_editable_camera_policy_v2_9_8() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the sequential source policy with one-grid movable margin."""

    payload = _policy_payload(_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_paused_camera_policy_v2_9_9() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the one-grid policy whose observation bank pauses physics."""

    payload = _policy_payload(_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_settled_camera_policy_v2_9_10() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the paused-bank policy that freezes only after final settlement."""

    payload = _policy_payload(_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256_v2(payload, domain=_POLICY_HASH_DOMAIN),
    )


def _strict_pose_bank(
    pose_bank: object,
) -> tuple[CompetitionNativeCameraPoseV2_9_3, ...]:
    if type(pose_bank) is not tuple or any(
        type(item) is not CompetitionNativeCameraPoseV2_9_3 for item in pose_bank
    ):
        raise TypeError("camera evidence pose bank must be an exact pose tuple")
    checked = tuple(
        CompetitionNativeCameraPoseV2_9_3.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in pose_bank
    )
    if not checked or len(checked) > _MAX_POSE_BANK_MEMBERS:
        raise ValueError("camera evidence pose bank must contain 1 through 256 poses")
    if tuple(sorted(set(checked), key=_pose_key)) != checked:
        raise ValueError("camera evidence pose bank is not unique and canonical")
    return checked


def competition_native_camera_pose_bank_sha256_v2_9_3(
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
) -> Sha256Digest:
    """Hash one exact canonical pose bank in its independent domain."""

    checked = _strict_pose_bank(pose_bank)
    payload = {
        "pose_bank_version": "competition-native-camera-pose-bank:2.9.3",
        "poses": tuple(item.model_dump(mode="json") for item in checked),
    }
    return canonical_sha256_v2(payload, domain=_POSE_BANK_HASH_DOMAIN)


def score_competition_native_camera_scene_v2_9_3(
    scene: Scene,
) -> CompetitionNativeCameraScoreV2_9_3:
    """Count qualifying objects whose category occurs once in the scene."""

    checked = _strict_scene(scene)
    checked.camera_by_id("main")
    category_counts: dict[str, int] = {}
    for item in checked.objects:
        category_counts[item.category] = category_counts.get(item.category, 0) + 1

    qualifying = []
    for item in checked.objects:
        view = item.views.get("main")
        if (
            category_counts[item.category] == 1
            and item.request_eligible
            and view is not None
            and view.visible_fraction >= RelationEngine.MIN_VISIBLE_FRACTION
            and view.image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION
            and view.truncated_fraction <= RelationEngine.MAX_TRUNCATED_FRACTION
        ):
            qualifying.append(item)
    return CompetitionNativeCameraScoreV2_9_3(
        movable_scene_unique_category_qualifying_count=sum(
            item.movable for item in qualifying
        ),
        all_scene_unique_category_qualifying_count=len(qualifying),
    )


def _rotation_matrix_values(obb: OBB) -> tuple[tuple[float, float, float], ...]:
    rotation = obb.rotation
    norm = math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera placement OBB rotation is invalid")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
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


def _translated_obb_fully_visible_v2_9_4(
    scene: Scene,
    subject_object_id: str,
    position: CompetitionNativeCameraPlacementPositionV2_9_4,
) -> bool:
    subject = scene.object_by_id(subject_object_id)
    camera = scene.camera_by_id("main")
    delta = (
        position.x - subject.position.x,
        position.z - subject.position.y,
        position.y - subject.position.z,
    )
    rotation = _rotation_matrix_values(subject.obb)
    center = (
        subject.obb.center.x,
        subject.obb.center.y,
        subject.obb.center.z,
    )
    matrix = camera.world_to_camera
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    projected: list[tuple[float, float]] = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                local = (
                    x_sign * subject.obb.extent.x / 2.0,
                    y_sign * subject.obb.extent.y / 2.0,
                    z_sign * subject.obb.extent.z / 2.0,
                )
                world = tuple(
                    center[axis]
                    + delta[axis]
                    + sum(rotation[axis][inner] * local[inner] for inner in range(3))
                    for axis in range(3)
                )
                homogeneous = tuple(
                    sum(matrix[row * 4 + column] * world[column] for column in range(3))
                    + matrix[row * 4 + 3]
                    for row in range(4)
                )
                if homogeneous[3] == 0.0:
                    raise ValueError("camera placement projected a point to infinity")
                camera_x = homogeneous[0] / homogeneous[3]
                camera_y = homogeneous[1] / homogeneous[3]
                camera_z = homogeneous[2] / homogeneous[3]
                if camera_z <= 1e-12:
                    return False
                projected.append(
                    (
                        fx * camera_x / camera_z + cx,
                        cy - fy * camera_y / camera_z,
                    )
                )
    min_x = min(item[0] for item in projected)
    max_x = max(item[0] for item in projected)
    min_y = min(item[1] for item in projected)
    max_y = max(item[1] for item in projected)
    if not (
        min_x >= 0.5
        and max_x <= camera.width - 0.5
        and min_y >= 0.5
        and max_y <= camera.height - 0.5
    ):
        return False
    image_area_fraction = (
        (max_x - min_x) * (max_y - min_y) / (camera.width * camera.height)
    )
    return image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION


def score_competition_native_editable_camera_scene_v2_9_4(
    scene: Scene,
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
) -> CompetitionNativeCameraScoreV2_9_4:
    """Score one frozen scene by complete visible native edit coverage."""

    observed = _strict_scene(scene)
    base = score_competition_native_camera_scene_v2_9_3(observed)
    roster = _strict_placement_roster_v2_9_4(placement_roster)
    category_counts: dict[str, int] = {}
    for item in observed.objects:
        category_counts[item.category] = category_counts.get(item.category, 0) + 1

    visible_subjects = 0
    visible_positions = 0
    for entry in roster:
        subject = observed.object_by_id(entry.subject_object_id)
        observed.object_by_id(entry.support_object_id)
        view = subject.views.get("main")
        if (
            not subject.movable
            or not subject.request_eligible
            or subject.support_object_id != entry.support_object_id
            or category_counts[subject.category] != 1
            or view is None
            or view.visible_fraction < RelationEngine.MIN_VISIBLE_FRACTION
            or view.image_area_fraction < RelationEngine.MIN_IMAGE_AREA_FRACTION
            or view.truncated_fraction > RelationEngine.MAX_TRUNCATED_FRACTION
        ):
            continue
        count = sum(
            _translated_obb_fully_visible_v2_9_4(
                observed,
                entry.subject_object_id,
                position,
            )
            for position in entry.positions
        )
        visible_positions += count
        visible_subjects += count > 0
    return CompetitionNativeCameraScoreV2_9_4(
        placement_roster_sha256=(
            competition_native_camera_placement_roster_sha256_v2_9_4(roster)
        ),
        visible_native_placement_subject_count=visible_subjects,
        visible_native_placement_count=visible_positions,
        movable_scene_unique_category_qualifying_count=(
            base.movable_scene_unique_category_qualifying_count
        ),
        all_scene_unique_category_qualifying_count=(
            base.all_scene_unique_category_qualifying_count
        ),
    )


def score_competition_native_editable_camera_application_v2_9_4(
    *,
    source_scene: Scene,
    pose: CompetitionNativeCameraPoseV2_9_3,
    application: AI2ThorCameraApplication,
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
    policy: CompetitionNativeCameraPolicyV2_9_3,
) -> CompetitionNativeCameraScoreV2_9_4:
    """Close and score one source observation by native edit coverage."""

    checked_policy = _strict_policy(policy)
    if checked_policy.pose_policy_version not in _editable_pose_policy_versions():
        raise ValueError("editable camera score requires the edit-domain policy")
    score_competition_native_source_camera_application_v2_9_3(
        source_scene=source_scene,
        pose=pose,
        application=application,
        policy=checked_policy,
    )
    checked_application = _strict_application(application)
    return score_competition_native_editable_camera_scene_v2_9_4(
        checked_application.observed_scene,
        placement_roster,
    )


def _strict_application(value: object) -> AI2ThorCameraApplication:
    if type(value) is not AI2ThorCameraApplication:
        raise TypeError("camera evidence application must be exact")
    requested_pose = _strict_native_pose(value.requested_pose, label="requested pose")
    observed_pose = _strict_native_pose(value.observed_pose, label="observed pose")
    if type(value.observed_camera_position) is not AI2ThorNativePosition:
        raise TypeError("camera evidence observed camera position must be exact")
    observed_position = AI2ThorNativePosition(**asdict(value.observed_camera_position))
    observed_scene = _strict_scene(value.observed_scene)
    if type(value.observation) is not AI2ThorObservation:
        raise TypeError("camera evidence observation must be exact")
    observation = value.observation
    if type(observation.scene) is not Scene:
        raise TypeError("camera evidence observation scene must be exact")
    if any(
        type(blob) is not bytes
        for blob in (
            observation.rgb_png,
            observation.depth_npy,
            observation.instance_png,
            observation.pointcloud_ply,
        )
    ):
        raise TypeError("camera evidence observation assets must be exact bytes")
    if type(observation.is_scene_at_rest) is not bool or any(
        type(key) is not str or type(count) is not int or count < 0
        for key, count in observation.instance_pixel_counts.items()
    ):
        raise TypeError("camera evidence observation metadata must be exact")
    rebuilt_observation = AI2ThorObservation.create(
        scene=_strict_scene(observation.scene),
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=observation.instance_pixel_counts,
        is_scene_at_rest=observation.is_scene_at_rest,
    )
    if (
        rebuilt_observation != observation
        or rebuilt_observation.scene != observed_scene
    ):
        raise ValueError("camera evidence observation binding or asset hash changed")
    residuals = (
        value.position_residual_m,
        value.yaw_residual_degrees,
        value.horizon_residual_degrees,
    )
    if any(type(item) is not float or not math.isfinite(item) for item in residuals):
        raise TypeError(
            "camera evidence application residuals must be exact finite floats"
        )
    expected_residuals = (
        math.dist(
            (
                requested_pose.position.x,
                requested_pose.position.y,
                requested_pose.position.z,
            ),
            (
                observed_pose.position.x,
                observed_pose.position.y,
                observed_pose.position.z,
            ),
        ),
        abs(
            (observed_pose.yaw_degrees - requested_pose.yaw_degrees + 180.0) % 360.0
            - 180.0
        ),
        abs(observed_pose.horizon_degrees - requested_pose.horizon_degrees),
    )
    if any(
        not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=4.0 * max(math.ulp(actual), math.ulp(expected)),
        )
        for actual, expected in zip(residuals, expected_residuals, strict=True)
    ):
        raise ValueError("camera evidence application residual changed")
    if (
        value.position_residual_m > _MAX_POSITION_RESIDUAL_M
        or value.yaw_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or value.horizon_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
    ):
        raise ValueError("camera evidence application pose drift exceeds tolerance")
    if observed_pose.standing is not requested_pose.standing:
        raise ValueError("camera evidence application standing state changed")
    if not rebuilt_observation.is_scene_at_rest:
        raise ValueError("camera evidence application is not at rest")
    return AI2ThorCameraApplication(
        requested_pose=requested_pose,
        observed_pose=observed_pose,
        observed_camera_position=observed_position,
        observed_scene=observed_scene,
        observation=rebuilt_observation,
        position_residual_m=value.position_residual_m,
        yaw_residual_degrees=value.yaw_residual_degrees,
        horizon_residual_degrees=value.horizon_residual_degrees,
    )


def _validate_application_source_closure(
    source_scene: Scene,
    application: AI2ThorCameraApplication,
) -> None:
    observed = application.observed_scene
    if (
        observed.scene_id != source_scene.scene_id
        or observed.source != source_scene.source
        or observed.coordinate_system != source_scene.coordinate_system
        or observed.room_polygon_xy != source_scene.room_polygon_xy
        or observed.collision_obstacles != source_scene.collision_obstacles
        or observed.subject_position_regions != source_scene.subject_position_regions
        or observed.pinned_object_ids != source_scene.pinned_object_ids
        or observed.generation_seed != source_scene.generation_seed
    ):
        raise ValueError("camera evidence application source root changed")

    source_object_ids = tuple(item.object_id for item in source_scene.objects)
    observed_object_ids = tuple(item.object_id for item in observed.objects)
    if source_object_ids != observed_object_ids:
        raise ValueError("camera evidence application source object roster changed")
    for source_object, observed_object in zip(
        source_scene.objects, observed.objects, strict=True
    ):
        if (
            observed_object.object_id != source_object.object_id
            or observed_object.name != source_object.name
            or observed_object.category != source_object.category
            or observed_object.movable is not source_object.movable
            or observed_object.request_eligible is not source_object.request_eligible
            or observed_object.support_object_id != source_object.support_object_id
            or observed_object.position != source_object.position
            or observed_object.rotation != source_object.rotation
            or observed_object.obb != source_object.obb
        ):
            raise ValueError("camera evidence application source object changed")

    source_camera_ids = tuple(item.camera_id for item in source_scene.cameras)
    observed_camera_ids = tuple(item.camera_id for item in observed.cameras)
    if source_camera_ids != observed_camera_ids:
        raise ValueError("camera evidence application source camera roster changed")
    for source_camera, observed_camera in zip(
        source_scene.cameras, observed.cameras, strict=True
    ):
        if (
            observed_camera.camera_id != source_camera.camera_id
            or observed_camera.width != source_camera.width
            or observed_camera.height != source_camera.height
            or observed_camera.intrinsics != source_camera.intrinsics
            or (
                source_camera.camera_id != "main"
                and observed_camera.world_to_camera != source_camera.world_to_camera
            )
        ):
            raise ValueError("camera evidence application source camera changed")


def score_competition_native_source_camera_application_v2_9_3(
    *,
    source_scene: Scene,
    pose: CompetitionNativeCameraPoseV2_9_3,
    application: AI2ThorCameraApplication,
    policy: CompetitionNativeCameraPolicyV2_9_3 | None = None,
) -> CompetitionNativeCameraScoreV2_9_3:
    """Close and score one bank-aligned application without retaining siblings."""

    source = _strict_scene(source_scene)
    source.camera_by_id("main")
    if type(pose) is not CompetitionNativeCameraPoseV2_9_3:
        raise TypeError("camera evidence pose must be exact")
    checked_pose = CompetitionNativeCameraPoseV2_9_3.model_validate(
        pose.model_dump(mode="python"), strict=True
    )
    checked_application = _strict_application(application)
    if _wire_pose(checked_application.requested_pose) != checked_pose:
        raise ValueError("camera evidence application is not bank-aligned")
    if checked_application.observed_scene.scene_id != source.scene_id:
        raise ValueError("camera evidence application scene identity mismatch")
    _validate_application_source_closure(source, checked_application)
    observed_camera = checked_application.observed_scene.camera_by_id("main")
    if policy is not None:
        verify_competition_native_solver_camera_binding_v2_9_3(
            policy,
            checked_pose,
            observed_camera,
        )
    return score_competition_native_camera_scene_v2_9_3(
        checked_application.observed_scene
    )


def _strict_policy(
    policy: object,
) -> CompetitionNativeCameraPolicyV2_9_3:
    if type(policy) is not CompetitionNativeCameraPolicyV2_9_3:
        raise TypeError("camera evidence policy must be exact")
    return CompetitionNativeCameraPolicyV2_9_3.model_validate(
        policy.model_dump(mode="python"), strict=True
    )


def verify_competition_native_solver_camera_binding_v2_9_3(
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose: CompetitionNativeCameraPoseV2_9_3,
    camera: Camera,
) -> None:
    """Require exact upright camera semantics only for the solver policy."""

    checked_policy = _strict_policy(policy)
    if type(pose) is not CompetitionNativeCameraPoseV2_9_3:
        raise TypeError("solver camera pose must be exact")
    checked_pose = CompetitionNativeCameraPoseV2_9_3.model_validate(
        pose.model_dump(mode="python"), strict=True
    )
    if type(camera) is not Camera or camera.camera_id != "main":
        raise TypeError("solver camera must be exact main")
    checked_camera = Camera.model_validate(
        camera.model_dump(mode="python"), strict=True
    )
    if checked_policy.pose_policy_version == _LEGACY_POSE_POLICY_VERSION:
        return
    if checked_pose.horizon_degrees != 0.0:
        raise ValueError("solver-upright camera pose must use horizon zero")
    try:
        _legacy_camera(checked_camera.world_to_camera)
    except _CameraConversionError as error:
        raise ValueError("solver-upright camera matrix is unsupported") from error


def build_competition_native_source_camera_evidence_v2_9_3(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_3, ...],
    selected_application: AI2ThorCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Build evidence from a complete light ledger and one replayed winner."""

    checked_source_scene = _strict_scene(source_scene)
    if checked_source_scene.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity mismatch")
    checked_source_scene.camera_by_id("main")
    checked_policy = _strict_policy(policy)
    checked_bank = _strict_pose_bank(pose_bank)
    checked_scores = _strict_score_ledger(pose_scores)
    if len(checked_scores) != len(checked_bank):
        raise ValueError("camera evidence score ledger is not bank-aligned")
    selected_index = select_competition_native_camera_score_index_v2_9_3(checked_scores)
    selected_pose = checked_bank[selected_index]
    selected = _strict_application(selected_application)
    selected_score = score_competition_native_source_camera_application_v2_9_3(
        source_scene=checked_source_scene,
        pose=selected_pose,
        application=selected,
    )
    if selected_score != checked_scores[selected_index]:
        raise ValueError("camera evidence replay score differs from frozen score")

    observed_pose = _wire_pose(selected.observed_pose)
    observed_position = (
        selected.observed_camera_position.x,
        selected.observed_camera_position.y,
        selected.observed_camera_position.z,
    )
    camera = selected.observed_scene.camera_by_id("main")
    observation = selected.observation
    pose_bank_sha256 = competition_native_camera_pose_bank_sha256_v2_9_3(checked_bank)
    payload = _evidence_payload(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        policy_sha256=checked_policy.policy_sha256,
        pose_bank_sha256=pose_bank_sha256,
        pose_bank_count=len(checked_bank),
        pose_scores=checked_scores,
        selected_pose_index=selected_index,
        requested_pose=selected_pose,
        observed_pose=observed_pose,
        observed_native_camera_position=observed_position,
        camera=camera,
        score=selected_score,
        rgb_png_sha256=observation.rgb_png_sha256,
        depth_npy_sha256=observation.depth_npy_sha256,
        instance_png_sha256=observation.instance_png_sha256,
        pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
        is_scene_at_rest=observation.is_scene_at_rest,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256_v2(
            payload, domain=_EVIDENCE_HASH_DOMAIN
        ),
    )


def build_competition_native_source_camera_evidence_v2_9_4(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_4, ...],
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
    selected_application: AI2ThorCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Build source evidence whose winner maximizes visible edit coverage."""

    checked_source_scene = _strict_scene(source_scene)
    if checked_source_scene.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity mismatch")
    checked_source_scene.camera_by_id("main")
    checked_policy = _strict_policy(policy)
    if checked_policy.pose_policy_version not in _editable_pose_policy_versions():
        raise ValueError("editable camera evidence requires the edit-domain policy")
    checked_bank = _strict_pose_bank(pose_bank)
    checked_scores = _strict_score_ledger(pose_scores)
    if any(
        type(item) is not CompetitionNativeCameraScoreV2_9_4 for item in checked_scores
    ):
        raise TypeError("editable camera evidence requires 2.9.4 scores")
    if len(checked_scores) != len(checked_bank):
        raise ValueError("camera evidence score ledger is not bank-aligned")
    checked_roster = _strict_placement_roster_v2_9_4(placement_roster)
    selected_index = select_competition_native_camera_score_index_v2_9_4(
        checked_scores  # type: ignore[arg-type]
    )
    selected_pose = checked_bank[selected_index]
    selected = _strict_application(selected_application)
    selected_score = score_competition_native_editable_camera_application_v2_9_4(
        source_scene=checked_source_scene,
        pose=selected_pose,
        application=selected,
        placement_roster=checked_roster,
        policy=checked_policy,
    )
    if selected_score != checked_scores[selected_index]:
        raise ValueError("camera evidence replay score differs from frozen score")

    observed_pose = _wire_pose(selected.observed_pose)
    observed_position = (
        selected.observed_camera_position.x,
        selected.observed_camera_position.y,
        selected.observed_camera_position.z,
    )
    camera = selected.observed_scene.camera_by_id("main")
    observation = selected.observation
    payload = _evidence_payload(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        policy_sha256=checked_policy.policy_sha256,
        pose_bank_sha256=competition_native_camera_pose_bank_sha256_v2_9_3(
            checked_bank
        ),
        pose_bank_count=len(checked_bank),
        pose_scores=checked_scores,
        selected_pose_index=selected_index,
        requested_pose=selected_pose,
        observed_pose=observed_pose,
        observed_native_camera_position=observed_position,
        camera=camera,
        score=selected_score,
        rgb_png_sha256=observation.rgb_png_sha256,
        depth_npy_sha256=observation.depth_npy_sha256,
        instance_png_sha256=observation.instance_png_sha256,
        pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
        is_scene_at_rest=observation.is_scene_at_rest,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256_v2(
            payload, domain=_EVIDENCE_HASH_DOMAIN
        ),
    )


def select_competition_native_source_camera_evidence_v2_9_3(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    applications: tuple[AI2ThorCameraApplication, ...],
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Select one complete source observation using literal scene scores only."""

    checked_source_scene = _strict_scene(source_scene)
    checked_bank = _strict_pose_bank(pose_bank)
    if type(applications) is not tuple or any(
        type(item) is not AI2ThorCameraApplication for item in applications
    ):
        raise TypeError("camera evidence applications must be an exact tuple")
    if len(applications) != len(checked_bank):
        raise ValueError("camera evidence applications are not bank-aligned")
    checked_applications = tuple(_strict_application(item) for item in applications)
    scores: list[CompetitionNativeCameraScoreV2_9_3] = []
    for index, (pose, application) in enumerate(
        zip(checked_bank, checked_applications, strict=True)
    ):
        try:
            score = score_competition_native_source_camera_application_v2_9_3(
                source_scene=checked_source_scene,
                pose=pose,
                application=application,
                policy=policy,
            )
        except ValueError as error:
            if "bank-aligned" in str(error):
                raise ValueError(
                    f"camera evidence application {index} is not bank-aligned"
                ) from error
            raise
        scores.append(score)

    pose_scores = tuple(scores)
    selected_index = select_competition_native_camera_score_index_v2_9_3(pose_scores)
    return build_competition_native_source_camera_evidence_v2_9_3(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        source_scene=checked_source_scene,
        policy=policy,
        pose_bank=checked_bank,
        pose_scores=pose_scores,
        selected_application=checked_applications[selected_index],
    )


def verify_competition_native_source_camera_evidence_v2_9_3(
    evidence: CompetitionNativeSourceCameraEvidenceV2_9_3,
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    source_scene: Scene,
    selected_scene: Scene,
    selected_camera: Camera,
    selected_application: AI2ThorCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Close persisted evidence against independently supplied capture bindings."""

    if type(evidence) is not CompetitionNativeSourceCameraEvidenceV2_9_3:
        raise TypeError("camera evidence must be exact")
    checked = CompetitionNativeSourceCameraEvidenceV2_9_3.model_validate(
        evidence.model_dump(mode="python"), strict=True
    )
    checked_policy = _strict_policy(policy)
    checked_bank = _strict_pose_bank(pose_bank)
    source = _strict_scene(source_scene)
    scene = _strict_scene(selected_scene)
    if type(selected_camera) is not Camera:
        raise TypeError("camera evidence selected camera must be exact")
    camera = Camera.model_validate(
        selected_camera.model_dump(mode="python"), strict=True
    )
    application = _strict_application(selected_application)
    if source.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity does not close")
    _validate_application_source_closure(source, application)
    expected_lineage = (
        source_id,
        scene_id,
        source_locator_sha256,
        runtime_identity_sha256,
        source_capture_sha256,
    )
    if expected_lineage != (
        checked.source_id,
        checked.scene_id,
        checked.source_locator_sha256,
        checked.runtime_identity_sha256,
        checked.source_capture_sha256,
    ):
        raise ValueError("camera evidence lineage does not close")
    if (
        checked.policy_sha256 != checked_policy.policy_sha256
        or checked.pose_bank_sha256
        != competition_native_camera_pose_bank_sha256_v2_9_3(checked_bank)
        or checked.pose_bank_count != len(checked_bank)
        or checked.selected_pose_index >= len(checked_bank)
        or checked_bank[checked.selected_pose_index] != checked.requested_pose
        or checked.requested_pose != _wire_pose(application.requested_pose)
    ):
        raise ValueError("camera evidence policy or pose bank does not close")
    if (
        scene.scene_id != scene_id
        or application.observed_scene != scene
        or application.observation.scene != scene
        or scene.camera_by_id("main") != camera
        or checked.camera != camera
        or checked.observed_pose != _wire_pose(application.observed_pose)
        or checked.observed_native_camera_position
        != (
            application.observed_camera_position.x,
            application.observed_camera_position.y,
            application.observed_camera_position.z,
        )
        or checked.score != score_competition_native_camera_scene_v2_9_3(scene)
        or checked.rgb_png_sha256 != application.observation.rgb_png_sha256
        or checked.depth_npy_sha256 != application.observation.depth_npy_sha256
        or checked.instance_png_sha256 != application.observation.instance_png_sha256
        or checked.pointcloud_ply_sha256
        != application.observation.pointcloud_ply_sha256
        or checked.is_scene_at_rest is not application.observation.is_scene_at_rest
    ):
        raise ValueError("camera evidence selected capture does not close")
    verify_competition_native_solver_camera_binding_v2_9_3(
        checked_policy,
        checked.requested_pose,
        camera,
    )
    return checked


CameraPolicy = CompetitionNativeCameraPolicyV2_9_3
SourceCameraEvidence = CompetitionNativeSourceCameraEvidenceV2_9_3
build_settled_camera_policy = build_competition_native_settled_camera_policy_v2_9_10
verify_source_camera_evidence = verify_competition_native_source_camera_evidence_v2_9_3

__all__ = (
    "CameraPolicy",
    "SourceCameraEvidence",
    "build_settled_camera_policy",
    "verify_source_camera_evidence",
)
