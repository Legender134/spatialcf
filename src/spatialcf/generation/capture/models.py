"""Flat current capture and roster contracts.

These models preserve the current wire literals and hash domains without
inheriting historical policy or compilation versions.
"""

from __future__ import annotations

# Final Task 3 owners for platform-neutral camera and surface evidence.
import math
from dataclasses import asdict
from hashlib import sha256
from typing import Literal, Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterObservation,
    AdapterPose,
    AdapterPosition,
    AdapterRuntimeIdentity,
    AdapterSpawnMap,
    AdapterSurfacePatch,
)
from spatialcf.domain.base import CanonicalModel, FiniteFloat, Sha256Digest
from spatialcf.domain.scene import OBB, Camera, Quaternion, Scene
from spatialcf.domain.serialization import canonical_sha256
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


class CompetitionNativeCameraPoseV2_9_3(CanonicalModel):
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


class CompetitionNativeCameraPolicyV2_9_3(CanonicalModel):
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
        expected = canonical_sha256(
            _policy_payload(self.pose_policy_version),
            domain=_POLICY_HASH_DOMAIN,
        )
        if self.policy_sha256 != expected:
            raise ValueError("camera evidence policy digest mismatch")
        return self


class CompetitionNativeCameraScoreV2_9_3(CanonicalModel):
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


class CompetitionNativeCameraPlacementPositionV2_9_4(CanonicalModel):
    """One exact native subject anchor considered by camera selection."""

    position_version: Literal["competition-native-camera-placement-position:2.9.4"] = (
        "competition-native-camera-placement-position:2.9.4"
    )
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class CompetitionNativeCameraPlacementRosterEntryV2_9_4(CanonicalModel):
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


class CompetitionNativeCameraScoreV2_9_4(CanonicalModel):
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
    return canonical_sha256(payload, domain=_PLACEMENT_ROSTER_HASH_DOMAIN)


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
    expected_world_to_camera = _expected_camera_world_to_camera(
        observed_native_camera_position,
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


class CompetitionNativeSourceCameraEvidenceV2_9_3(CanonicalModel):
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
            canonical_sha256(
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
            canonical_sha256(
                _policy_payload(version),
                domain=_POLICY_HASH_DOMAIN,
            )
            for version in _editable_pose_policy_versions()
        }
        if (type(checked_scores[0]) is CompetitionNativeCameraScoreV2_9_4) != (
            self.policy_sha256 in editable_policy_sha256s
        ):
            raise ValueError("camera evidence score version does not bind policy")
        expected = canonical_sha256(
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


def _strict_native_pose(pose: object, *, label: str) -> AdapterPose:
    if type(pose) is not AdapterPose:
        raise TypeError(f"{label} must be an exact AdapterPose")
    if type(pose.position) is not AdapterPosition:
        raise TypeError(f"{label} position must be exact")
    return AdapterPose(
        position=AdapterPosition(**asdict(pose.position)),
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _wire_pose(pose: AdapterPose) -> CompetitionNativeCameraPoseV2_9_3:
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
    reachable_positions: tuple[AdapterPosition, ...],
    fallback_pose: AdapterPose,
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
        or any(type(item) is not AdapterPosition for item in reachable_positions)
    ):
        raise TypeError("reachable positions must be a non-empty exact tuple")
    checked_positions = tuple(
        AdapterPosition(**asdict(item)) for item in reachable_positions
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
            checked_fallback = AdapterPose(
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
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_solver_upright_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the bounded camera policy supported by the certified solver."""

    payload = _policy_payload(_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_editable_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the upright policy that ranks source-native edit coverage."""

    payload = _policy_payload(_EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_collision_safe_editable_camera_policy_v2_9_5() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the source-only edit policy with fixed 0.2m movable clearance."""

    payload = _policy_payload(_COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_contact_margin_editable_camera_policy_v2_9_6() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the edit policy with 0.2m agent plus fixed 1cm margin."""

    payload = _policy_payload(_CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_reset_per_pose_editable_camera_policy_v2_9_7() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the contact-safe policy that resets the source per pose."""

    payload = _policy_payload(_RESET_PER_POSE_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_grid_margin_editable_camera_policy_v2_9_8() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the sequential source policy with one-grid movable margin."""

    payload = _policy_payload(_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_paused_camera_policy_v2_9_9() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the one-grid policy whose observation bank pauses physics."""

    payload = _policy_payload(_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_settled_camera_policy_v2_9_10() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the paused-bank policy that freezes only after final settlement."""

    payload = _policy_payload(_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
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
    return canonical_sha256(payload, domain=_POSE_BANK_HASH_DOMAIN)


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
    application: AdapterCameraApplication,
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


def _strict_application(value: object) -> AdapterCameraApplication:
    if type(value) is not AdapterCameraApplication:
        raise TypeError("camera evidence application must be exact")
    requested_pose = _strict_native_pose(value.requested_pose, label="requested pose")
    observed_pose = _strict_native_pose(value.observed_pose, label="observed pose")
    if type(value.observed_camera_position) is not AdapterPosition:
        raise TypeError("camera evidence observed camera position must be exact")
    observed_position = AdapterPosition(**asdict(value.observed_camera_position))
    observed_scene = _strict_scene(value.observed_scene)
    if type(value.observation) is not AdapterObservation:
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
    if type(observation.is_settled) is not bool or any(
        type(key) is not str or type(count) is not int or count < 0
        for key, count in observation.instance_pixel_counts
    ):
        raise TypeError("camera evidence observation metadata must be exact")
    rebuilt_observation = AdapterObservation.create(
        scene=_strict_scene(observation.scene),
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=observation.instance_pixel_counts,
        is_settled=observation.is_settled,
        instance_colors=observation.instance_colors,
        instance_evidence_provenance=observation.instance_evidence_provenance,
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
    if not rebuilt_observation.is_settled:
        raise ValueError("camera evidence application is not at rest")
    return AdapterCameraApplication(
        source=value.source,
        binding=value.binding,
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
    application: AdapterCameraApplication,
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
    application: AdapterCameraApplication,
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
    selected_application: AdapterCameraApplication,
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
        is_scene_at_rest=observation.is_settled,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256(payload, domain=_EVIDENCE_HASH_DOMAIN),
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
    selected_application: AdapterCameraApplication,
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
        is_scene_at_rest=observation.is_settled,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256(payload, domain=_EVIDENCE_HASH_DOMAIN),
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
    applications: tuple[AdapterCameraApplication, ...],
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Select one complete source observation using literal scene scores only."""

    checked_source_scene = _strict_scene(source_scene)
    checked_bank = _strict_pose_bank(pose_bank)
    if type(applications) is not tuple or any(
        type(item) is not AdapterCameraApplication for item in applications
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
    selected_application: AdapterCameraApplication,
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
        or checked.is_scene_at_rest is not application.observation.is_settled
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

# Final Task 3 owner for capture-bound surface evidence.

from pydantic import Field, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel

_PATCH_HASH_DOMAIN = "spatialcf.competition-native-receptacle-surface-patch.v2.9.2"
_SUBJECT_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-subject-surface-evidence.v2.9.2"
)
_SOURCE_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-source-surface-evidence.v2.9.2"
)
_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"


def _patch_payload(
    *,
    patch_index: int,
    x_min: float,
    x_max: float,
    native_y: float,
    z_min: float,
    z_max: float,
) -> dict[str, object]:
    return {
        "native_y": native_y,
        "patch_index": patch_index,
        "x_max": x_max,
        "x_min": x_min,
        "z_max": z_max,
        "z_min": z_min,
    }


class CompetitionNativeReceptacleSurfacePatchV2_9_2(CanonicalModel):
    """One ordered raw 21x21 trigger-grid patch and its independent digest."""

    patch_index: int = Field(strict=True, ge=0)
    x_min: FiniteFloat
    x_max: FiniteFloat
    native_y: FiniteFloat
    z_min: FiniteFloat
    z_max: FiniteFloat
    patch_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if self.x_min >= self.x_max or self.z_min >= self.z_max:
            raise ValueError("surface evidence patch must have positive area")
        expected = canonical_sha256(
            _patch_payload(
                patch_index=self.patch_index,
                x_min=self.x_min,
                x_max=self.x_max,
                native_y=self.native_y,
                z_min=self.z_min,
                z_max=self.z_max,
            ),
            domain=_PATCH_HASH_DOMAIN,
        )
        if self.patch_sha256 != expected:
            raise ValueError("surface evidence patch digest mismatch")
        return self


def _subject_payload(
    *,
    subject_object_id: str,
    support_object_id: str,
    native_subject_object_id: str,
    native_support_object_id: str,
    runtime_identity_sha256: str,
    scene_sha256: str,
    positions_sha256: str,
    spawn_map_source_sha256: str,
    placement_sha256: str,
    source_capture_sha256: str,
    patches: tuple[CompetitionNativeReceptacleSurfacePatchV2_9_2, ...],
) -> dict[str, object]:
    return {
        "native_support_object_id": native_support_object_id,
        "native_subject_object_id": native_subject_object_id,
        "patches": tuple(item.model_dump(mode="json") for item in patches),
        "placement_sha256": placement_sha256,
        "positions_sha256": positions_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "scene_sha256": scene_sha256,
        "source_capture_sha256": source_capture_sha256,
        "spawn_map_source_sha256": spawn_map_source_sha256,
        "subject_object_id": subject_object_id,
        "support_object_id": support_object_id,
    }


class CompetitionNativeSubjectSurfaceEvidenceV2_9_2(CanonicalModel):
    """All source-capture bindings for one receptacle-supported subject."""

    subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    native_subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    native_support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    runtime_identity_sha256: Sha256Digest
    scene_sha256: Sha256Digest
    positions_sha256: Sha256Digest
    spawn_map_source_sha256: Sha256Digest
    placement_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    patches: tuple[CompetitionNativeReceptacleSurfacePatchV2_9_2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    subject_surface_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_subject_evidence(self) -> Self:
        if self.subject_object_id == self.support_object_id:
            raise ValueError("surface evidence subject and support must differ")
        if tuple(item.patch_index for item in self.patches) != tuple(
            range(len(self.patches))
        ):
            raise ValueError("surface evidence patch indexes are not canonical")
        patch_keys = tuple(
            (
                item.native_y,
                item.x_min,
                item.z_min,
                item.x_max,
                item.z_max,
            )
            for item in self.patches
        )
        if patch_keys != tuple(sorted(set(patch_keys))):
            raise ValueError("surface evidence patches are not unique and canonical")
        expected = canonical_sha256(
            _subject_payload(
                subject_object_id=self.subject_object_id,
                support_object_id=self.support_object_id,
                native_subject_object_id=self.native_subject_object_id,
                native_support_object_id=self.native_support_object_id,
                runtime_identity_sha256=self.runtime_identity_sha256,
                scene_sha256=self.scene_sha256,
                positions_sha256=self.positions_sha256,
                spawn_map_source_sha256=self.spawn_map_source_sha256,
                placement_sha256=self.placement_sha256,
                source_capture_sha256=self.source_capture_sha256,
                patches=self.patches,
            ),
            domain=_SUBJECT_EVIDENCE_HASH_DOMAIN,
        )
        if self.subject_surface_evidence_sha256 != expected:
            raise ValueError("subject surface evidence digest mismatch")
        return self


def _source_payload(
    *,
    source_id: str,
    scene_id: str,
    source_capture_sha256: str,
    subjects: tuple[CompetitionNativeSubjectSurfaceEvidenceV2_9_2, ...],
) -> dict[str, object]:
    return {
        "evidence_version": "competition-native-source-surface-evidence:2.9.2",
        "scene_id": scene_id,
        "source_capture_sha256": source_capture_sha256,
        "source_id": source_id,
        "subjects": tuple(item.model_dump(mode="json") for item in subjects),
    }


class CompetitionNativeSourceSurfaceEvidenceV2_9_2(CanonicalModel):
    """One sibling surface-evidence row for an unchanged accepted capture."""

    evidence_version: Literal["competition-native-source-surface-evidence:2.9.2"] = (
        "competition-native-source-surface-evidence:2.9.2"
    )
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_capture_sha256: Sha256Digest
    subjects: tuple[CompetitionNativeSubjectSurfaceEvidenceV2_9_2, ...] = Field(
        max_length=96
    )
    surface_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_source_evidence(self) -> Self:
        subject_ids = tuple(item.subject_object_id for item in self.subjects)
        if subject_ids != tuple(sorted(set(subject_ids))):
            raise ValueError("source surface evidence subjects are not canonical")
        if any(
            item.source_capture_sha256 != self.source_capture_sha256
            for item in self.subjects
        ):
            raise ValueError("source surface evidence capture lineage mismatch")
        expected = canonical_sha256(
            _source_payload(
                source_id=self.source_id,
                scene_id=self.scene_id,
                source_capture_sha256=self.source_capture_sha256,
                subjects=self.subjects,
            ),
            domain=_SOURCE_EVIDENCE_HASH_DOMAIN,
        )
        if self.surface_evidence_sha256 != expected:
            raise ValueError("source surface evidence digest mismatch")
        return self


def _strict_spawn_map(value: object) -> AdapterSpawnMap:
    if type(value) is not AdapterSpawnMap:
        raise TypeError("surface evidence spawn map must be exact")
    if type(value.runtime_identity) is not AdapterRuntimeIdentity:
        raise TypeError("surface evidence runtime identity must be exact")
    if type(value.surface_patches) is not tuple or any(
        type(item) is not AdapterSurfacePatch for item in value.surface_patches
    ):
        raise TypeError("surface evidence patches must be an exact tuple")
    return AdapterSpawnMap(
        binding=value.binding,
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        runtime_identity=AdapterRuntimeIdentity(**asdict(value.runtime_identity)),
        positions=tuple(value.positions),
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=tuple(
            AdapterSurfacePatch(**asdict(item)) for item in value.surface_patches
        ),
        position_region=value.position_region,
    )


def _build_patch(
    patch_index: int,
    patch: AdapterSurfacePatch,
) -> CompetitionNativeReceptacleSurfacePatchV2_9_2:
    payload = _patch_payload(
        patch_index=patch_index,
        x_min=patch.x_min,
        x_max=patch.x_max,
        native_y=patch.native_y,
        z_min=patch.z_min,
        z_max=patch.z_max,
    )
    return CompetitionNativeReceptacleSurfacePatchV2_9_2(
        **payload,
        patch_sha256=canonical_sha256(payload, domain=_PATCH_HASH_DOMAIN),
    )


def _accepted_capture_scene_sha256(scene: Scene) -> str:
    """Digest one validated Canonical capture scene in source-roster order."""

    return sha256(
        canonical_json_bytes(normalize_competition_native_source_scene_v2_9(scene))
    ).hexdigest()


def verify_competition_native_source_surface_evidence_v2_9_2(
    capture: CompetitionNativeSourceCaptureV2_9,
    evidence: CompetitionNativeSourceSurfaceEvidenceV2_9_2,
) -> CompetitionNativeSourceSurfaceEvidenceV2_9_2:
    """Freshly replay every persisted patch binding against one capture."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("surface evidence capture must be exact")
    if type(evidence) is not CompetitionNativeSourceSurfaceEvidenceV2_9_2:
        raise TypeError("surface evidence must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"),
        strict=True,
    )
    checked_evidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2.model_validate(
        evidence.model_dump(mode="python"),
        strict=True,
    )
    if (
        checked_evidence.source_id != checked_capture.source.source_id
        or checked_evidence.scene_id != checked_capture.scene.scene_id
        or checked_evidence.source_capture_sha256
        != checked_capture.source_capture_sha256
    ):
        raise ValueError("surface evidence does not bind source capture")

    placements = {item.object_id: item for item in checked_capture.placement_facts}
    supports = {item.object_id: item for item in checked_capture.support_facts}
    expected_subject_ids = tuple(
        sorted(
            item.object_id
            for item in checked_capture.placement_facts
            if item.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        )
    )
    if tuple(item.subject_object_id for item in checked_evidence.subjects) != (
        expected_subject_ids
    ):
        raise ValueError("surface evidence does not close capture placements")

    runtime = AdapterRuntimeIdentity(
        **checked_capture.runtime_identity.model_dump(mode="python")
    )
    runtime_digest = canonical_sha256(
        CompetitionNativeRuntimeIdentityV2_9(**asdict(runtime)),
        domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
    )
    capture_scene_sha256 = _accepted_capture_scene_sha256(checked_capture.scene)
    for subject in checked_evidence.subjects:
        placement = placements[subject.subject_object_id]
        support = supports[subject.subject_object_id]
        support_object_id = support.support_object_id
        if subject.scene_sha256 != capture_scene_sha256:
            raise ValueError(
                "surface evidence subject scene digest does not bind capture"
            )
        if (
            support.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or support_object_id is None
            or subject.support_object_id != support_object_id
            or subject.native_subject_object_id != support.native_object_id
            or subject.native_support_object_id
            != supports[support_object_id].native_object_id
            or subject.runtime_identity_sha256 != runtime_digest
            or subject.placement_sha256 != placement.placement_sha256
            or subject.source_capture_sha256 != checked_capture.source_capture_sha256
            or placement.position_region is None
        ):
            raise ValueError("surface evidence subject does not bind capture facts")
        region = placement.position_region
        if (
            region.subject_object_id != subject.subject_object_id
            or region.source_kind != "ai2thor-receptacle-trigger-grid-v1"
            or region.source_sha256 != subject.spawn_map_source_sha256
        ):
            raise ValueError("surface evidence placement region changed on replay")
    return checked_evidence


def build_competition_native_source_surface_evidence_v2_9_2(
    capture: CompetitionNativeSourceCaptureV2_9,
    spawn_maps: tuple[AdapterSpawnMap, ...],
) -> CompetitionNativeSourceSurfaceEvidenceV2_9_2:
    """Close raw patch ownership against one unchanged source capture."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("surface evidence capture must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"),
        strict=True,
    )
    if type(spawn_maps) is not tuple:
        raise TypeError("surface evidence spawn_maps must be an exact tuple")
    checked_maps = tuple(
        sorted(
            (_strict_spawn_map(item) for item in spawn_maps),
            key=lambda item: item.subject_object_id,
        )
    )
    map_subject_ids = tuple(item.subject_object_id for item in checked_maps)
    if len(map_subject_ids) != len(set(map_subject_ids)):
        raise ValueError("surface evidence spawn-map subjects must be unique")

    placement_by_id = {item.object_id: item for item in checked_capture.placement_facts}
    support_by_id = {item.object_id: item for item in checked_capture.support_facts}
    expected_subject_ids = tuple(
        sorted(
            item.object_id
            for item in checked_capture.placement_facts
            if item.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        )
    )
    if map_subject_ids != expected_subject_ids:
        raise ValueError("surface evidence spawn maps do not close capture placements")

    subjects: list[CompetitionNativeSubjectSurfaceEvidenceV2_9_2] = []
    for spawn_map in checked_maps:
        placement = placement_by_id[spawn_map.subject_object_id]
        support = support_by_id[spawn_map.subject_object_id]
        support_object_id = support.support_object_id
        if (
            support.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or support_object_id is None
            or spawn_map.scene_id != checked_capture.scene.scene_id
            or spawn_map.support_object_id != support_object_id
            or spawn_map.native_subject_object_id != support.native_object_id
            or spawn_map.native_support_object_id
            != support_by_id[support_object_id].native_object_id
            or not spawn_map.surface_patches
            or placement.position_region is None
            or spawn_map.position_region != placement.position_region
        ):
            raise ValueError("surface evidence spawn map does not bind capture facts")
        expected_runtime = CompetitionNativeRuntimeIdentityV2_9(
            **asdict(spawn_map.runtime_identity)
        )
        if expected_runtime != checked_capture.runtime_identity:
            raise ValueError("surface evidence runtime does not bind source capture")
        patches = tuple(
            _build_patch(index, patch)
            for index, patch in enumerate(spawn_map.surface_patches)
        )
        runtime_identity_sha256 = canonical_sha256(
            checked_capture.runtime_identity,
            domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
        )
        payload = _subject_payload(
            subject_object_id=spawn_map.subject_object_id,
            support_object_id=spawn_map.support_object_id,
            native_subject_object_id=spawn_map.native_subject_object_id,
            native_support_object_id=spawn_map.native_support_object_id,
            runtime_identity_sha256=runtime_identity_sha256,
            scene_sha256=spawn_map.scene_sha256,
            positions_sha256=spawn_map.positions_sha256,
            spawn_map_source_sha256=spawn_map.source_sha256,
            placement_sha256=placement.placement_sha256,
            source_capture_sha256=checked_capture.source_capture_sha256,
            patches=patches,
        )
        subjects.append(
            CompetitionNativeSubjectSurfaceEvidenceV2_9_2(
                **payload,
                subject_surface_evidence_sha256=canonical_sha256(
                    payload,
                    domain=_SUBJECT_EVIDENCE_HASH_DOMAIN,
                ),
            )
        )

    subject_tuple = tuple(subjects)
    source_payload = _source_payload(
        source_id=checked_capture.source.source_id,
        scene_id=checked_capture.scene.scene_id,
        source_capture_sha256=checked_capture.source_capture_sha256,
        subjects=subject_tuple,
    )
    evidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2(
        source_id=checked_capture.source.source_id,
        scene_id=checked_capture.scene.scene_id,
        source_capture_sha256=checked_capture.source_capture_sha256,
        subjects=subject_tuple,
        surface_evidence_sha256=canonical_sha256(
            source_payload,
            domain=_SOURCE_EVIDENCE_HASH_DOMAIN,
        ),
    )
    return verify_competition_native_source_surface_evidence_v2_9_2(
        checked_capture,
        evidence,
    )


ReceptacleSurfacePatch = CompetitionNativeReceptacleSurfacePatchV2_9_2
SourceSurfaceEvidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2
SubjectSurfaceEvidence = CompetitionNativeSubjectSurfaceEvidenceV2_9_2
build_source_surface_evidence = build_competition_native_source_surface_evidence_v2_9_2
verify_source_surface_evidence = (
    verify_competition_native_source_surface_evidence_v2_9_2
)

__all__ = (
    "ReceptacleSurfacePatch",
    "SourceSurfaceEvidence",
    "SubjectSurfaceEvidence",
    "build_source_surface_evidence",
    "verify_source_surface_evidence",
)

from enum import StrEnum

from pydantic import Field, model_validator

from spatialcf.domain.base import CanonicalModel
from spatialcf.domain.request import Relation
from spatialcf.domain.scene import BBox2D, SubjectPositionRegion, Vec2
from spatialcf.domain.serialization import (
    canonical_json_bytes,
)
from spatialcf.generation.capture.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)

DatasetSplitV2_9 = Literal["train", "validation", "test"]

_CURRENT_POLICY_HASH_DOMAIN = (
    "spatialcf.competition-native-candidate-roster-policy.v2.9.4"
)
_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_SOURCE_CAPTURE_HASH_DOMAIN = "spatialcf.competition-native-source-capture.v2.9"
_MANIFEST_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-manifest.v2.9"
_SUMMARY_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-summary.v2.9"
_PLACEMENT_HASH_DOMAIN = "spatialcf.competition-native-placement-fact.v2.9"
_MAX_TEXT_CHARS = 512
_MAX_REASON_CHARS = 256
_MAX_REASONS = 8
_MAX_POLICY_SOURCES = 512
_MAX_OBJECTS_PER_SCENE = 96
_MAX_CANDIDATES_TOTAL = 40_000
_MAX_REQUESTS_TOTAL = 1_000
_MAX_NATIVE_POSITIONS = 10_000
_MAX_CAMERAS_PER_SCENE = 64
_MAX_OBSTACLES_PER_SCENE = 2_048
_MAX_REGIONS_PER_SCENE = 2_048
_MAX_POLYGON_VERTICES = 4_096
_SOURCE_CAPTURE_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024 - 16 * 1024
_MAX_PERSISTED_REQUEST_TEXT_CHARS = _MAX_TEXT_CHARS


class CompetitionNativeSupportKindV2_9(StrEnum):
    FLOOR = "FLOOR"
    RECEPTACLE = "RECEPTACLE"
    UNKNOWN = "UNKNOWN"
    MULTIPLE_AMBIGUOUS = "MULTIPLE_AMBIGUOUS"
    CYCLIC = "CYCLIC"


class CompetitionNativePlacementAvailabilityV2_9(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    KNOWN_RECEPTACLE_SPAWN = "KNOWN_RECEPTACLE_SPAWN"
    KNOWN_FLOOR_INNER_REGION = "KNOWN_FLOOR_INNER_REGION"
    MISSING = "MISSING"


class CompetitionNativeSubjectStateV2_9(StrEnum):
    ELIGIBLE_RECEPTACLE_DOMAIN = "ELIGIBLE_RECEPTACLE_DOMAIN"
    ELIGIBLE_FLOOR_INNER_DOMAIN = "ELIGIBLE_FLOOR_INNER_DOMAIN"
    REJECTED_NOT_MOVABLE = "REJECTED_NOT_MOVABLE"
    REJECTED_NOT_REQUEST_ELIGIBLE = "REJECTED_NOT_REQUEST_ELIGIBLE"
    REJECTED_PINNED = "REJECTED_PINNED"
    REJECTED_HAS_DEPENDENT_CHILD = "REJECTED_HAS_DEPENDENT_CHILD"
    REJECTED_UNKNOWN_SUPPORT = "REJECTED_UNKNOWN_SUPPORT"
    REJECTED_MISSING_PLACEMENT_DOMAIN = "REJECTED_MISSING_PLACEMENT_DOMAIN"
    REJECTED_INVALID_SOURCE_FACT = "REJECTED_INVALID_SOURCE_FACT"


class CompetitionNativeCandidateStateV2_9(StrEnum):
    SELECTED = "SELECTED"
    REJECTED_SUBJECT = "REJECTED_SUBJECT"
    REJECTED_REFERENCE = "REJECTED_REFERENCE"
    REJECTED_DEPENDENCY = "REJECTED_DEPENDENCY"
    REJECTED_NOT_VISIBLE = "REJECTED_NOT_VISIBLE"
    REJECTED_AMBIGUOUS = "REJECTED_AMBIGUOUS"
    REJECTED_RELATION_NOT_OBSERVED = "REJECTED_RELATION_NOT_OBSERVED"
    REJECTED_POLICY_CAP = "REJECTED_POLICY_CAP"
    REJECTED_TARGET_UNREACHABLE = "REJECTED_TARGET_UNREACHABLE"


class CompetitionNativeSourceRefV2_9(CanonicalModel):
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    split: DatasetSplitV2_9
    source_locator_sha256: Sha256Digest = Field(
        description=(
            "Digest of the frozen locator record; this is distinct from a "
            "procedural runtime's source content digest."
        )
    )


class RosterPolicy(CanonicalModel):
    """The only supported candidate-roster policy."""

    policy_version: Literal["competition-native-candidate-roster-policy:2.9.4"] = (
        "competition-native-candidate-roster-policy:2.9.4"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    seed: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    width: int = Field(strict=True, gt=0, le=4096)
    height: int = Field(strict=True, gt=0, le=4096)
    max_scenes: int = Field(strict=True, gt=0, le=_MAX_POLICY_SOURCES)
    max_objects_per_scene: int = Field(strict=True, gt=0, le=_MAX_OBJECTS_PER_SCENE)
    max_candidates_total: int = Field(strict=True, gt=0, le=_MAX_CANDIDATES_TOTAL)
    max_requests_total: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_scene: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_subject: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_category_pair: int = Field(
        strict=True, gt=0, le=_MAX_REQUESTS_TOTAL
    )
    target_requests_per_relation: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    sources: tuple[CompetitionNativeSourceRefV2_9, ...] = Field(
        min_length=1, max_length=_MAX_POLICY_SOURCES
    )
    require_scene_unique_referents: Literal[True] = True
    require_receptacle_surface_evidence: Literal[True] = True
    require_source_camera_evidence: Literal[True] = True
    camera_policy: CameraPolicy
    require_receptacle_native_target_reachability: Literal[True] = True

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.width * self.height > 4_194_304:
            raise ValueError("candidate roster render dimensions exceed pixel limit")
        source_ids = tuple(item.source_id for item in self.sources)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError("candidate roster sources must use canonical unique order")
        if len(self.sources) > self.max_scenes:
            raise ValueError("candidate roster source count exceeds max_scenes")
        return self

    @property
    def policy_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_CURRENT_POLICY_HASH_DOMAIN)


CompetitionNativeCandidateRosterPolicyV2_9_4 = RosterPolicy


class CompetitionNativeRuntimeIdentityV2_9(CanonicalModel):
    ai2thor_version: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    unity_commit_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_scene_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_TEXT_CHARS
    )
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    seed: int = Field(strict=True)
    render_depth_image: bool
    render_instance_segmentation: bool
    grid_size_m: float
    snap_to_grid: bool
    rotate_step_degrees: int = Field(strict=True, gt=0)
    coordinate_transform_version: str = Field(
        strict=True, min_length=1, max_length=_MAX_TEXT_CHARS
    )
    source_dataset_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_revision: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_split: Literal["train", "val", "validation", "test"] | None
    source_index: int | None = Field(default=None, ge=0)
    source_sha256: Sha256Digest | None = Field(
        description=(
            "Digest of actual procedural source content, independent of the "
            "frozen source locator digest."
        )
    )
    source_scene_alias: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_loader_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_loader_version: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_room_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_floor_xz_bounds: tuple[float, float, float, float] | None
    teleport_vertical_guard_m: float


def validate_competition_native_runtime_source_lineage_v2_9(
    source: CompetitionNativeSourceRefV2_9,
    runtime: CompetitionNativeRuntimeIdentityV2_9,
) -> None:
    """Bind legacy/procedural runtime identity to one frozen source locator."""

    runtime_split = (
        "validation" if runtime.source_split == "val" else runtime.source_split
    )
    if runtime_split is not None and runtime_split != source.split:
        raise ValueError("source capture runtime split does not match frozen source")
    if runtime.native_scene_name == "Procedural":
        if (
            runtime.source_scene_alias != source.scene_id
            or runtime.source_sha256 is None
            or runtime.source_split is None
        ):
            raise ValueError(
                "procedural runtime scene alias does not match frozen source"
            )
    elif runtime.native_scene_name not in {
        source.scene_id,
        f"{source.scene_id}_physics",
    }:
        raise ValueError("legacy runtime scene name does not match frozen source")


class CompetitionNativeSupportFactV2_9(CanonicalModel):
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    raw_parent_object_ids: tuple[str, ...] = Field(max_length=_MAX_OBJECTS_PER_SCENE)
    structural_parent_object_ids: tuple[str, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    domain_parent_object_ids: tuple[str, ...] = Field(max_length=_MAX_OBJECTS_PER_SCENE)
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None
    floor_object_id: str | None

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        for values in (
            self.raw_parent_object_ids,
            self.structural_parent_object_ids,
            self.domain_parent_object_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("captured support parent IDs are not canonical")
            if any(not value or len(value) > _MAX_TEXT_CHARS for value in values):
                raise ValueError("captured support parent ID exceeds persisted limit")
        if any(
            value is not None and len(value) > _MAX_TEXT_CHARS
            for value in (self.support_object_id, self.floor_object_id)
        ):
            raise ValueError("captured support target exceeds persisted limit")
        raw = set(self.raw_parent_object_ids)
        structural = set(self.structural_parent_object_ids)
        domain = set(self.domain_parent_object_ids)
        if not structural.issubset(raw) or not domain.issubset(raw):
            raise ValueError("captured support partitions escape raw parents")
        if structural.intersection(domain):
            raise ValueError("captured support partitions overlap")
        kind = self.support_kind
        if kind is CompetitionNativeSupportKindV2_9.RECEPTACLE:
            valid = (
                len(domain) == 1
                and not structural
                and self.support_object_id == self.domain_parent_object_ids[0]
                and self.floor_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.FLOOR:
            valid = (
                len(structural) == 1
                and not domain
                and self.floor_object_id == self.structural_parent_object_ids[0]
                and self.support_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.UNKNOWN:
            valid = (
                not domain
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS:
            valid = (
                len(structural) + len(domain) > 1
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        else:
            valid = (
                bool(domain)
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        if not valid:
            raise ValueError("captured support fact is not closed")
        return self


class CompetitionNativePositionV2_9(CanonicalModel):
    x: float
    y: float
    z: float


class CompetitionNativeFloorEnvelopeV2_9(CanonicalModel):
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    floor_object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    floor_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_aabb: OBB
    floor_top_z: float
    clearance_m: float = Field(ge=0.0)
    polygon_xy: tuple[Vec2, ...] = Field(min_length=3, max_length=_MAX_POLYGON_VERTICES)

    @model_validator(mode="after")
    def validate_polygon(self) -> Self:
        coordinates = tuple((point.x, point.y) for point in self.polygon_xy)
        twice_area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(coordinates, coordinates[1:] + coordinates[:1])
        )
        if len(set(coordinates)) < 3 or twice_area == 0.0:
            raise ValueError("floor envelope polygon is degenerate")
        return self


def _placement_payload(
    *,
    object_id: str,
    availability: CompetitionNativePlacementAvailabilityV2_9,
    support_kind: CompetitionNativeSupportKindV2_9,
    support_object_id: str | None,
    floor_object_id: str | None,
    native_positions: tuple[CompetitionNativePositionV2_9, ...],
    position_region: SubjectPositionRegion | None,
    reasons: tuple[str, ...],
) -> dict[str, object]:
    return {
        "availability": availability.value,
        "floor_object_id": floor_object_id,
        "native_positions": tuple(
            item.model_dump(mode="json") for item in native_positions
        ),
        "object_id": object_id,
        "position_region": (
            None if position_region is None else position_region.model_dump(mode="json")
        ),
        "reasons": reasons,
        "support_kind": support_kind.value,
        "support_object_id": support_object_id,
    }


class CompetitionNativeSubjectPlacementFactV2_9(CanonicalModel):
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    availability: CompetitionNativePlacementAvailabilityV2_9
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    floor_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    native_positions: tuple[CompetitionNativePositionV2_9, ...] = Field(
        max_length=_MAX_NATIVE_POSITIONS
    )
    position_region: SubjectPositionRegion | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)
    placement_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_placement(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("placement reasons must be non-empty and canonical")
        region = self.position_region
        if region is not None and (
            len(region.region_id) > _MAX_TEXT_CHARS
            or len(region.subject_object_id) > _MAX_TEXT_CHARS
            or len(region.components) > _MAX_POLYGON_VERTICES
            or any(
                len(component.exterior) > _MAX_POLYGON_VERTICES
                or len(component.holes) > _MAX_POLYGON_VERTICES
                or any(len(hole) > _MAX_POLYGON_VERTICES for hole in component.holes)
                for component in region.components
            )
        ):
            raise ValueError("placement region exceeds persisted nested limit")
        position_keys = tuple(
            (item.x, item.z, item.y) for item in self.native_positions
        )
        if position_keys != tuple(sorted(set(position_keys))):
            raise ValueError("placement native positions must be unique and canonical")
        if (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.NOT_APPLICABLE
        ):
            valid = (
                not self.native_positions
                and self.position_region is None
                and not self.reasons
            )
        elif (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        ):
            valid = (
                self.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and bool(self.native_positions)
                and (
                    self.position_region is None
                    or (
                        self.position_region.subject_object_id == self.object_id
                        and self.position_region.source_kind
                        == "ai2thor-receptacle-trigger-grid-v1"
                        and bool(self.position_region.components)
                    )
                )
                and not self.reasons
            )
        elif (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
        ):
            valid = (
                self.support_kind is CompetitionNativeSupportKindV2_9.FLOOR
                and not self.native_positions
                and self.position_region is not None
                and bool(self.position_region.components)
                and not self.reasons
            )
        else:
            valid = (
                not self.native_positions
                and self.position_region is None
                and bool(self.reasons)
            )
        if not valid:
            raise ValueError("subject placement fact is not closed")
        expected = canonical_sha256(
            _placement_payload(
                object_id=self.object_id,
                availability=self.availability,
                support_kind=self.support_kind,
                support_object_id=self.support_object_id,
                floor_object_id=self.floor_object_id,
                native_positions=self.native_positions,
                position_region=self.position_region,
                reasons=self.reasons,
            ),
            domain=_PLACEMENT_HASH_DOMAIN,
        )
        if self.placement_sha256 != expected:
            raise ValueError("subject placement digest mismatch")
        return self


def build_competition_native_subject_placement_fact_v2_9(
    *,
    object_id: str,
    availability: CompetitionNativePlacementAvailabilityV2_9,
    support_kind: CompetitionNativeSupportKindV2_9,
    support_object_id: str | None,
    floor_object_id: str | None,
    native_positions: tuple[CompetitionNativePositionV2_9, ...] = (),
    position_region: SubjectPositionRegion | None = None,
    reasons: tuple[str, ...] = (),
) -> CompetitionNativeSubjectPlacementFactV2_9:
    reasons = tuple(sorted(set(reasons)))
    native_positions = tuple(
        sorted(native_positions, key=lambda item: (item.x, item.z, item.y))
    )
    payload = _placement_payload(
        object_id=object_id,
        availability=availability,
        support_kind=support_kind,
        support_object_id=support_object_id,
        floor_object_id=floor_object_id,
        native_positions=native_positions,
        position_region=position_region,
        reasons=reasons,
    )
    return CompetitionNativeSubjectPlacementFactV2_9(
        object_id=object_id,
        availability=availability,
        support_kind=support_kind,
        support_object_id=support_object_id,
        floor_object_id=floor_object_id,
        native_positions=native_positions,
        position_region=position_region,
        reasons=reasons,
        placement_sha256=canonical_sha256(payload, domain=_PLACEMENT_HASH_DOMAIN),
    )


_SOURCE_VIEW_FACT_HASH_DOMAIN = (
    "spatialcf.competition-native-source-view-fact.v2.9.5"
)
_SOURCE_VIEW_BINDING_HASH_DOMAIN = "spatialcf.source-view-binding.v1"
_SOURCE_VIEW_SAMPLING_POLICY_SHA256 = canonical_sha256(
    {
        "local_quantization_m": 1e-5,
        "maximum_samples": 76_800,
        "render_proxy": "weighted_sampled_splat_z_buffer",
        "representative": "minimum_finite_positive_depth_then_row_column",
        "rounding": "nearest_even",
        "tile_height": 2,
        "tile_width": 2,
        "weight": "object_mask_pixel_count_in_tile",
    },
    domain="spatialcf.competition-native-source-view-policy.v2.9.5",
)


class SourceViewObjectSamples(CanonicalModel):
    object_id: CanonicalId
    source_mask_bbox: BBox2D
    source_mask_pixel_count: int = Field(strict=True, gt=0)
    sample_rows: tuple[int, ...]
    sample_columns: tuple[int, ...]
    sample_weights: tuple[int, ...]
    local_x_quantized: tuple[int, ...]
    local_y_quantized: tuple[int, ...]
    local_z_quantized: tuple[int, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        arrays = (
            self.sample_rows,
            self.sample_columns,
            self.sample_weights,
            self.local_x_quantized,
            self.local_y_quantized,
            self.local_z_quantized,
        )
        if not self.sample_rows or len({len(item) for item in arrays}) != 1:
            raise ValueError("source-view sample arrays must be equal and nonempty")
        if any(type(value) is not int for array in arrays for value in array):
            raise TypeError("source-view samples must contain exact integers")
        if any(value < 0 for value in (*self.sample_rows, *self.sample_columns)):
            raise ValueError("source-view sample pixels must be in bounds")
        keys = tuple(
            (row // 2, column // 2)
            for row, column in zip(
                self.sample_rows, self.sample_columns, strict=True
            )
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source-view sample tiles must be canonical")
        if any(weight < 1 or weight > 4 for weight in self.sample_weights):
            raise ValueError("source-view sample weights must be in [1, 4]")
        if sum(self.sample_weights) != self.source_mask_pixel_count:
            raise ValueError("source-view sample weights do not close mask count")
        bbox_values = (
            self.source_mask_bbox.xmin,
            self.source_mask_bbox.ymin,
            self.source_mask_bbox.xmax,
            self.source_mask_bbox.ymax,
        )
        if any(
            not math.isfinite(value) or value < 0 or not float(value).is_integer()
            for value in bbox_values
        ):
            raise ValueError("source-view mask bbox must be an integer envelope")
        xmin, ymin, xmax, ymax = (int(value) for value in bbox_values)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("source-view mask bbox must be nonempty and half-open")
        if any(
            row < ymin or row >= ymax or column < xmin or column >= xmax
            for row, column in zip(self.sample_rows, self.sample_columns, strict=True)
        ):
            raise ValueError("source-view sample pixels must lie inside the mask bbox")
        if self.source_mask_pixel_count > (xmax - xmin) * (ymax - ymin):
            raise ValueError("source-view mask count exceeds the mask bbox area")
        for row, column, weight in zip(
            self.sample_rows,
            self.sample_columns,
            self.sample_weights,
            strict=True,
        ):
            tile_ymin = (row // 2) * 2
            tile_xmin = (column // 2) * 2
            tile_area = max(0, min(ymax, tile_ymin + 2) - max(ymin, tile_ymin)) * max(
                0, min(xmax, tile_xmin + 2) - max(xmin, tile_xmin)
            )
            if weight > tile_area:
                raise ValueError("source-view sample weight exceeds its mask tile")
        return self


class SourceViewFact(CanonicalModel):
    fact_version: Literal["competition-native-source-view-fact:2.9.5"]
    source_id: CanonicalId
    scene_id: CanonicalId
    source_locator_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    scene_sha256: Sha256Digest
    camera_sha256: Sha256Digest
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    sampling_policy_sha256: Sha256Digest
    objects: tuple[SourceViewObjectSamples, ...]
    source_view_fact_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        object_ids = tuple(item.object_id for item in self.objects)
        if not object_ids or object_ids != tuple(sorted(set(object_ids))):
            raise ValueError("source-view object rows must be canonical")
        if sum(len(item.sample_rows) for item in self.objects) > 76_800:
            raise ValueError("source-view fact exceeds the sample cap")
        if self.sampling_policy_sha256 != _SOURCE_VIEW_SAMPLING_POLICY_SHA256:
            raise ValueError("source-view sampling policy changed")
        payload = self.model_dump(
            mode="python", exclude={"source_view_fact_sha256"}
        )
        expected = canonical_sha256(
            payload, domain=_SOURCE_VIEW_FACT_HASH_DOMAIN
        )
        if self.source_view_fact_sha256 != expected:
            raise ValueError("source-view fact digest mismatch")
        return self


def _capture_payload(
    *,
    source: CompetitionNativeSourceRefV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    scene: Scene,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
    settlement_pass_steps: int,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...],
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
    source_view_fact: SourceViewFact | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "depth_npy_sha256": depth_npy_sha256,
        "floor_envelope": None
        if floor_envelope is None
        else floor_envelope.model_dump(mode="json"),
        "instance_png_sha256": instance_png_sha256,
        "is_scene_at_rest": is_scene_at_rest,
        "placement_facts": tuple(
            item.model_dump(mode="json") for item in placement_facts
        ),
        "pointcloud_ply_sha256": pointcloud_ply_sha256,
        "reachable_positions": tuple(
            item.model_dump(mode="json") for item in reachable_positions
        ),
        "rgb_png_sha256": rgb_png_sha256,
        "runtime_identity": runtime_identity.model_dump(mode="json"),
        "scene": scene.model_dump(mode="json", warnings="error"),
        "settlement_pass_steps": settlement_pass_steps,
        "source": source.model_dump(mode="json"),
        "support_facts": tuple(item.model_dump(mode="json") for item in support_facts),
    }
    if source_view_fact is not None:
        payload["source_view_fact"] = source_view_fact.model_dump(mode="json")
    return payload


def normalize_competition_native_source_scene_v2_9(scene: Scene) -> Scene:
    """Reject duplicate source rosters and return their one canonical ordering."""

    if type(scene) is not Scene:
        raise TypeError("captured source scene must be an exact Scene")
    if len(scene.scene_id) > _MAX_TEXT_CHARS or len(scene.source) > _MAX_TEXT_CHARS:
        raise ValueError("captured source scene identity exceeds persisted limit")
    if (
        len(scene.objects) > _MAX_OBJECTS_PER_SCENE
        or len(scene.cameras) > _MAX_CAMERAS_PER_SCENE
        or len(scene.collision_obstacles) > _MAX_OBSTACLES_PER_SCENE
        or len(scene.subject_position_regions) > _MAX_REGIONS_PER_SCENE
        or len(scene.room_polygon_xy) > _MAX_POLYGON_VERTICES
    ):
        raise ValueError("captured source scene nested roster exceeds persisted limit")
    for item in scene.objects:
        if (
            any(
                len(value) > _MAX_TEXT_CHARS
                for value in (item.object_id, item.name, item.category)
            )
            or len(item.views) > _MAX_CAMERAS_PER_SCENE
            or any(
                len(camera_id) > _MAX_TEXT_CHARS
                or len(view.camera_id) > _MAX_TEXT_CHARS
                for camera_id, view in item.views.items()
            )
            or (
                item.support_object_id is not None
                and len(item.support_object_id) > _MAX_TEXT_CHARS
            )
        ):
            raise ValueError("captured source object exceeds persisted limit")
    if (
        any(len(item.camera_id) > _MAX_TEXT_CHARS for item in scene.cameras)
        or any(
            len(item.obstacle_id) > _MAX_TEXT_CHARS
            or len(item.source_object_id) > _MAX_TEXT_CHARS
            for item in scene.collision_obstacles
        )
        or any(
            len(item.region_id) > _MAX_TEXT_CHARS
            or len(item.subject_object_id) > _MAX_TEXT_CHARS
            or len(item.components) > _MAX_POLYGON_VERTICES
            or any(
                len(component.exterior) > _MAX_POLYGON_VERTICES
                or len(component.holes) > _MAX_POLYGON_VERTICES
                or any(len(hole) > _MAX_POLYGON_VERTICES for hole in component.holes)
                for component in item.components
            )
            for item in scene.subject_position_regions
        )
        or len(scene.pinned_object_ids) > _MAX_OBJECTS_PER_SCENE
        or any(
            len(object_id) > _MAX_TEXT_CHARS for object_id in scene.pinned_object_ids
        )
    ):
        raise ValueError("captured source scene strings exceed persisted limit")
    unique_rosters = (
        (tuple(item.object_id for item in scene.objects), "object IDs"),
        (tuple(item.name for item in scene.objects), "object names"),
        (tuple(item.camera_id for item in scene.cameras), "camera IDs"),
        (
            tuple(item.obstacle_id for item in scene.collision_obstacles),
            "collision obstacle IDs",
        ),
        (
            tuple(item.region_id for item in scene.subject_position_regions),
            "subject position region IDs",
        ),
    )
    for values, label in unique_rosters:
        if len(values) != len(set(values)):
            raise ValueError(f"captured source scene contains duplicate {label}")
    normalized = scene.model_copy(
        update={
            "cameras": tuple(sorted(scene.cameras, key=lambda item: item.camera_id)),
            "objects": tuple(sorted(scene.objects, key=lambda item: item.object_id)),
            "collision_obstacles": tuple(
                sorted(scene.collision_obstacles, key=lambda item: item.obstacle_id)
            ),
            "subject_position_regions": tuple(
                sorted(
                    scene.subject_position_regions,
                    key=lambda item: item.region_id,
                )
            ),
        }
    )
    return Scene.model_validate(normalized.model_dump(mode="python"), strict=True)


def _point_in_polygon_or_boundary(point: Vec2, polygon: tuple[Vec2, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        cross = (point.y - previous.y) * (current.x - previous.x) - (
            point.x - previous.x
        ) * (current.y - previous.y)
        if (
            abs(cross) <= 1e-9
            and min(previous.x, current.x) - 1e-9
            <= point.x
            <= max(previous.x, current.x) + 1e-9
            and min(previous.y, current.y) - 1e-9
            <= point.y
            <= max(previous.y, current.y) + 1e-9
        ):
            return True
        if (current.y > point.y) != (previous.y > point.y):
            intersection_x = previous.x + (
                (point.y - previous.y)
                * (current.x - previous.x)
                / (current.y - previous.y)
            )
            if point.x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _validate_floor_envelope_binding(
    scene: Scene,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
) -> None:
    floor_facts = tuple(
        item
        for item in support_facts
        if item.support_kind is CompetitionNativeSupportKindV2_9.FLOOR
    )
    known_floor_ids = {
        item.floor_object_id for item in floor_facts if item.floor_object_id is not None
    }
    known_floor_placements = tuple(
        item
        for item in placement_facts
        if item.availability
        is CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
    )
    if floor_envelope is None:
        if known_floor_placements:
            raise ValueError("known floor placement has no captured floor envelope")
        return
    native_aabb = floor_envelope.native_aabb
    floor_bounds = (
        native_aabb.center.x - native_aabb.extent.x / 2.0,
        native_aabb.center.y - native_aabb.extent.y / 2.0,
        native_aabb.center.x + native_aabb.extent.x / 2.0,
        native_aabb.center.y + native_aabb.extent.y / 2.0,
    )
    if runtime_identity.source_floor_xz_bounds is not None:
        source_bounds = runtime_identity.source_floor_xz_bounds
        floor_bounds = (
            max(source_bounds[0], floor_bounds[0]),
            max(source_bounds[1], floor_bounds[1]),
            min(source_bounds[2], floor_bounds[2]),
            min(source_bounds[3], floor_bounds[3]),
        )
    clearance = floor_envelope.clearance_m
    expected_polygon = (
        Vec2(x=floor_bounds[0] + clearance, y=floor_bounds[1] + clearance),
        Vec2(x=floor_bounds[2] - clearance, y=floor_bounds[1] + clearance),
        Vec2(x=floor_bounds[2] - clearance, y=floor_bounds[3] - clearance),
        Vec2(x=floor_bounds[0] + clearance, y=floor_bounds[3] - clearance),
    )
    if (
        floor_envelope.scene_id != scene.scene_id
        or known_floor_ids != {floor_envelope.floor_object_id}
        or not floor_facts
        or floor_envelope.polygon_xy != expected_polygon
        or floor_envelope.floor_top_z
        != native_aabb.center.z + native_aabb.extent.z / 2.0
        or any(
            not _point_in_polygon_or_boundary(point, scene.room_polygon_xy)
            for point in floor_envelope.polygon_xy
        )
        or floor_envelope.native_aabb.extent.x <= 0.0
        or floor_envelope.native_aabb.extent.y <= 0.0
        or floor_envelope.native_aabb.extent.z <= 0.0
    ):
        raise ValueError("captured floor envelope does not bind the settled scene")
    floor_fact_by_object = {item.object_id: item for item in floor_facts}
    if any(
        item.object_id not in floor_fact_by_object
        or item.floor_object_id != floor_envelope.floor_object_id
        or item.position_region is None
        or item.position_region.subject_object_id != item.object_id
        or any(
            not _point_in_polygon_or_boundary(point, floor_envelope.polygon_xy)
            for component in item.position_region.components
            for ring in (component.exterior, *component.holes)
            for point in ring
        )
        for item in known_floor_placements
    ):
        raise ValueError("captured floor placement does not bind its support fact")


def validate_competition_native_floor_envelope_v2_9(
    scene: Scene,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    *,
    expected_clearance_m: float,
) -> None:
    """Bind an adapter floor envelope to this scene, support roster and call."""

    if floor_envelope.clearance_m != expected_clearance_m:
        raise ValueError("captured floor envelope clearance does not match request")
    _validate_floor_envelope_binding(
        scene, support_facts, floor_envelope, (), runtime_identity
    )


class CompetitionNativeSourceCaptureV2_9(CanonicalModel):
    source: CompetitionNativeSourceRefV2_9
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9
    scene: Scene
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    pointcloud_ply_sha256: Sha256Digest
    is_scene_at_rest: bool
    settlement_pass_steps: int = Field(strict=True, ge=0)
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...] = Field(
        max_length=_MAX_NATIVE_POSITIONS
    )
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    source_view_fact: SourceViewFact | None = None
    source_capture_sha256: Sha256Digest

    @model_serializer(mode="wrap")
    def serialize_optional_source_view_fact(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if self.source_view_fact is None:
            payload.pop("source_view_fact", None)
        return payload

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        if self.scene.scene_id != self.source.scene_id:
            raise ValueError("source capture scene identity mismatch")
        if (self.runtime_identity.width, self.runtime_identity.height) == (0, 0):
            raise ValueError("source capture runtime dimensions are invalid")
        validate_competition_native_runtime_source_lineage_v2_9(
            self.source, self.runtime_identity
        )
        normalized_scene = normalize_competition_native_source_scene_v2_9(self.scene)
        if normalized_scene != self.scene:
            raise ValueError("captured source scene ordering is not canonical")
        object_ids = tuple(item.object_id for item in self.scene.objects)
        support_ids = tuple(item.object_id for item in self.support_facts)
        placement_ids = tuple(item.object_id for item in self.placement_facts)
        if support_ids != tuple(sorted(object_ids)) or placement_ids != tuple(
            sorted(object_ids)
        ):
            raise ValueError("source capture object fact rosters are not closed")
        object_id_set = set(object_ids)
        if any(
            parent_id not in object_id_set
            for fact in self.support_facts
            for parent_id in fact.domain_parent_object_ids
        ):
            raise ValueError("source capture domain parent has no stable object")
        support_by_id = {item.object_id: item for item in self.support_facts}
        if any(
            item.scene_id != self.scene.scene_id
            or item.object_name != self.scene.object_by_id(item.object_id).name
            for item in self.support_facts
        ):
            raise ValueError("source capture support facts do not bind the scene")
        if any(
            placement.support_kind
            is not support_by_id[placement.object_id].support_kind
            or placement.support_object_id
            != support_by_id[placement.object_id].support_object_id
            or placement.floor_object_id
            != support_by_id[placement.object_id].floor_object_id
            for placement in self.placement_facts
        ):
            raise ValueError("source capture placement facts do not bind support facts")
        _validate_floor_envelope_binding(
            self.scene,
            self.support_facts,
            self.floor_envelope,
            self.placement_facts,
            self.runtime_identity,
        )
        position_keys = tuple(
            (item.x, item.z, item.y) for item in self.reachable_positions
        )
        if position_keys != tuple(sorted(set(position_keys))):
            raise ValueError("captured reachable positions are not canonical")
        if self.source_view_fact is not None:
            fact = self.source_view_fact
            camera = self.scene.camera_by_id("main")
            if (
                fact.source_id != self.source.source_id
                or fact.scene_id != self.scene.scene_id
                or fact.source_locator_sha256 != self.source.source_locator_sha256
                or fact.runtime_identity_sha256
                != canonical_sha256(
                    self.runtime_identity,
                    domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN,
                )
                or fact.scene_sha256
                != canonical_sha256(
                    self.scene, domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN
                )
                or fact.camera_sha256
                != canonical_sha256(
                    camera, domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN
                )
                or fact.rgb_png_sha256 != self.rgb_png_sha256
                or fact.depth_npy_sha256 != self.depth_npy_sha256
                or fact.instance_png_sha256 != self.instance_png_sha256
            ):
                raise ValueError("source-view fact does not bind captured evidence")
            if any(
                view is not None and view.camera_id != "main"
                for item in self.scene.objects
                for view in (item.views.get("main"),)
            ):
                raise ValueError("source-view fact main view camera identity changed")
            expected_object_ids = tuple(
                sorted(
                    item.object_id
                    for item in self.scene.objects
                    if (view := item.views.get("main")) is not None
                    and view.visible_fraction > 0.0
                )
            )
            fact_object_ids = tuple(item.object_id for item in fact.objects)
            if fact_object_ids != expected_object_ids:
                raise ValueError("source-view fact object roster does not bind main views")
            for samples in fact.objects:
                bbox = samples.source_mask_bbox
                if bbox.xmax > camera.width or bbox.ymax > camera.height:
                    raise ValueError("source-view fact bbox exceeds main camera bounds")
                if any(
                    row >= camera.height or column >= camera.width
                    for row, column in zip(
                        samples.sample_rows,
                        samples.sample_columns,
                        strict=True,
                    )
                ):
                    raise ValueError("source-view fact samples exceed main camera bounds")
                view = self.scene.object_by_id(samples.object_id).views["main"]
                if any(
                    not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                    for actual, expected in zip(
                        (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
                        (
                            view.bbox.xmin,
                            view.bbox.ymin,
                            view.bbox.xmax,
                            view.bbox.ymax,
                        ),
                        strict=True,
                    )
                ):
                    raise ValueError("source-view fact bbox does not bind its main view")
        capture_payload = _capture_payload(
            source=self.source,
            runtime_identity=self.runtime_identity,
            scene=self.scene,
            rgb_png_sha256=self.rgb_png_sha256,
            depth_npy_sha256=self.depth_npy_sha256,
            instance_png_sha256=self.instance_png_sha256,
            pointcloud_ply_sha256=self.pointcloud_ply_sha256,
            is_scene_at_rest=self.is_scene_at_rest,
            settlement_pass_steps=self.settlement_pass_steps,
            support_facts=self.support_facts,
            floor_envelope=self.floor_envelope,
            reachable_positions=self.reachable_positions,
            placement_facts=self.placement_facts,
            source_view_fact=self.source_view_fact,
        )
        if (
            len(canonical_json_bytes(capture_payload))
            > _SOURCE_CAPTURE_PAYLOAD_MAX_BYTES
        ):
            raise ValueError("source capture exceeds persisted record byte limit")
        expected = canonical_sha256(
            capture_payload,
            domain=_SOURCE_CAPTURE_HASH_DOMAIN,
        )
        if self.source_capture_sha256 != expected:
            raise ValueError("source capture digest mismatch")
        return self


def build_competition_native_source_capture_v2_9(
    *,
    source: CompetitionNativeSourceRefV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    scene: Scene,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
    settlement_pass_steps: int,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...],
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
    source_view_fact: SourceViewFact | None = None,
) -> CompetitionNativeSourceCaptureV2_9:
    normalized_scene = normalize_competition_native_source_scene_v2_9(scene)
    support_facts = tuple(sorted(support_facts, key=lambda item: item.object_id))
    reachable_positions = tuple(
        sorted(reachable_positions, key=lambda item: (item.x, item.z, item.y))
    )
    placement_facts = tuple(sorted(placement_facts, key=lambda item: item.object_id))
    payload = _capture_payload(
        source=source,
        runtime_identity=runtime_identity,
        scene=normalized_scene,
        rgb_png_sha256=rgb_png_sha256,
        depth_npy_sha256=depth_npy_sha256,
        instance_png_sha256=instance_png_sha256,
        pointcloud_ply_sha256=pointcloud_ply_sha256,
        is_scene_at_rest=is_scene_at_rest,
        settlement_pass_steps=settlement_pass_steps,
        support_facts=support_facts,
        floor_envelope=floor_envelope,
        reachable_positions=reachable_positions,
        placement_facts=placement_facts,
        source_view_fact=source_view_fact,
    )
    return CompetitionNativeSourceCaptureV2_9(
        source=source,
        runtime_identity=runtime_identity,
        scene=normalized_scene,
        rgb_png_sha256=rgb_png_sha256,
        depth_npy_sha256=depth_npy_sha256,
        instance_png_sha256=instance_png_sha256,
        pointcloud_ply_sha256=pointcloud_ply_sha256,
        is_scene_at_rest=is_scene_at_rest,
        settlement_pass_steps=settlement_pass_steps,
        support_facts=support_facts,
        floor_envelope=floor_envelope,
        reachable_positions=reachable_positions,
        placement_facts=placement_facts,
        source_view_fact=source_view_fact,
        source_capture_sha256=canonical_sha256(
            payload, domain=_SOURCE_CAPTURE_HASH_DOMAIN
        ),
    )


def competition_native_roster_selection_identity_v2_9(
    capture: CompetitionNativeSourceCaptureV2_9,
) -> Sha256Digest:
    """Return the source identity used only for deterministic roster selection."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("roster selection identity requires an exact source capture")
    if capture.source_view_fact is None:
        return capture.source_capture_sha256
    payload = _capture_payload(
        source=capture.source,
        runtime_identity=capture.runtime_identity,
        scene=capture.scene,
        rgb_png_sha256=capture.rgb_png_sha256,
        depth_npy_sha256=capture.depth_npy_sha256,
        instance_png_sha256=capture.instance_png_sha256,
        pointcloud_ply_sha256=capture.pointcloud_ply_sha256,
        is_scene_at_rest=capture.is_scene_at_rest,
        settlement_pass_steps=capture.settlement_pass_steps,
        support_facts=capture.support_facts,
        floor_envelope=capture.floor_envelope,
        reachable_positions=capture.reachable_positions,
        placement_facts=capture.placement_facts,
        source_view_fact=None,
    )
    return canonical_sha256(payload, domain=_SOURCE_CAPTURE_HASH_DOMAIN)


class CompetitionNativeSourceCaptureOutcomeV2_9(CanonicalModel):
    source: CompetitionNativeSourceRefV2_9
    status: Literal["accepted", "rejected"]
    capture: CompetitionNativeSourceCaptureV2_9 | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("source outcome reasons must be canonical")
        if self.status == "accepted":
            if (
                self.capture is None
                or self.reasons
                or self.capture.source != self.source
            ):
                raise ValueError("accepted source outcome is not closed")
        elif self.capture is not None or not self.reasons:
            raise ValueError("rejected source outcome is not closed")
        return self


class CompetitionNativeObjectInventoryV2_9(CanonicalModel):
    inventory_id: str = Field(pattern=r"^object-[0-9a-f]{64}$")
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    category: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    subject_state: CompetitionNativeSubjectStateV2_9
    reference_eligible: bool
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    placement_sha256: Sha256Digest
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("object inventory reasons must be canonical")
        eligible = self.subject_state in {
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_RECEPTACLE_DOMAIN,
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_FLOOR_INNER_DOMAIN,
        }
        if eligible == bool(self.reasons):
            raise ValueError("object subject terminal state is not closed")
        return self


class CompetitionNativeCandidateInventoryV2_9(CanonicalModel):
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    state: CompetitionNativeCandidateStateV2_9
    selection_index: int | None = Field(default=None, strict=True, ge=0)
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.subject_id == self.reference_id:
            raise ValueError("candidate subject and reference must differ")
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("candidate reasons must be canonical")
        if self.state is CompetitionNativeCandidateStateV2_9.SELECTED:
            if self.selection_index is None or self.reasons:
                raise ValueError("selected candidate is not closed")
        elif self.selection_index is not None or not self.reasons:
            raise ValueError("rejected candidate is not closed")
        return self


class CompetitionNativeSelectedRequestV2_9(CanonicalModel):
    request_id: str = Field(pattern=r"^request-[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    selection_index: int = Field(strict=True, ge=0)
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    source_locator_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    relation_after: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    camera_id: Literal["main"] = "main"

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("selected request relation_after must be opposite")
        return self


class CompetitionNativeCandidateRosterManifestV2_9(CanonicalModel):
    manifest_version: Literal["competition-native-candidate-roster-manifest:2.9"] = (
        "competition-native-candidate-roster-manifest:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    requests: tuple[CompetitionNativeSelectedRequestV2_9, ...] = Field(
        max_length=_MAX_REQUESTS_TOTAL
    )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        indices = tuple(item.selection_index for item in self.requests)
        if indices != tuple(range(len(self.requests))):
            raise ValueError("request manifest selection indices are not contiguous")
        request_ids = tuple(item.request_id for item in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request manifest request IDs are not unique")
        return self

    @property
    def manifest_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_MANIFEST_HASH_DOMAIN)


class CompetitionNativeRosterRejectionV2_9(CanonicalModel):
    rejection_id: str = Field(pattern=r"^rejection-[0-9a-f]{64}$")
    stage: Literal["source", "object_subject", "object_reference", "candidate"]
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    candidate_id: str | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("roster rejection reasons must be canonical")
        if self.stage == "source":
            valid = self.object_id is None and self.candidate_id is None
        elif self.stage in {"object_subject", "object_reference"}:
            valid = self.object_id is not None and self.candidate_id is None
        else:
            valid = self.object_id is None and self.candidate_id is not None
        if not valid:
            raise ValueError("roster rejection identity is not closed")
        return self


class CompetitionNativeStageCountV2_9(CanonicalModel):
    stage: str = Field(strict=True, min_length=1, max_length=128)
    count: int = Field(strict=True, gt=0)


class CompetitionNativeCandidateStateCountV2_9(CanonicalModel):
    state: CompetitionNativeCandidateStateV2_9
    count: int = Field(strict=True, gt=0)


class CompetitionNativeRelationCountV2_9(CanonicalModel):
    relation: Relation
    count: int = Field(strict=True, gt=0)


class RosterSummary(CanonicalModel):
    summary_version: Literal["competition-native-candidate-roster-summary:2.9"] = (
        "competition-native-candidate-roster-summary:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    manifest_sha256: Sha256Digest
    source_count: int = Field(strict=True, ge=0)
    accepted_source_count: int = Field(strict=True, ge=0)
    rejected_source_count: int = Field(strict=True, ge=0)
    object_count: int = Field(strict=True, ge=0)
    eligible_subject_count: int = Field(strict=True, ge=0)
    rejected_subject_count: int = Field(strict=True, ge=0)
    candidate_count: int = Field(strict=True, ge=0)
    selected_request_count: int = Field(strict=True, ge=0)
    rejected_candidate_count: int = Field(strict=True, ge=0)
    candidate_state_counts: tuple[CompetitionNativeCandidateStateCountV2_9, ...] = (
        Field(max_length=len(CompetitionNativeCandidateStateV2_9))
    )
    relation_selected_counts: tuple[CompetitionNativeRelationCountV2_9, ...] = Field(
        max_length=len(Relation)
    )
    rejection_stage_counts: tuple[CompetitionNativeStageCountV2_9, ...] = Field(
        max_length=4
    )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.source_count != self.accepted_source_count + self.rejected_source_count:
            raise ValueError("summary source counts do not close")
        if (
            self.object_count
            != self.eligible_subject_count + self.rejected_subject_count
        ):
            raise ValueError("summary object counts do not close")
        if (
            self.candidate_count
            != self.selected_request_count + self.rejected_candidate_count
        ):
            raise ValueError("summary candidate counts do not close")
        if (
            sum(item.count for item in self.candidate_state_counts)
            != self.candidate_count
        ):
            raise ValueError("summary candidate state counts do not close")
        if (
            sum(item.count for item in self.relation_selected_counts)
            != self.selected_request_count
        ):
            raise ValueError("summary relation counts do not close")
        for counts, key in (
            (self.candidate_state_counts, lambda item: item.state.value),
            (self.relation_selected_counts, lambda item: item.relation.value),
            (self.rejection_stage_counts, lambda item: item.stage),
        ):
            keys = tuple(key(item) for item in counts)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise ValueError("summary count keys are not canonical")
        return self

    @property
    def summary_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_SUMMARY_HASH_DOMAIN)


CompetitionNativeCandidateRosterSummaryV2_9 = RosterSummary


class RosterCompilation(CanonicalModel):
    """The only supported complete roster compilation."""

    policy: RosterPolicy
    scene_inventory: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    object_inventory: tuple[CompetitionNativeObjectInventoryV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
    )
    candidate_inventory: tuple[CompetitionNativeCandidateInventoryV2_9, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )
    request_manifest: CompetitionNativeCandidateRosterManifestV2_9
    rejections: tuple[CompetitionNativeRosterRejectionV2_9, ...] = Field(
        max_length=(
            _MAX_POLICY_SOURCES
            + 2 * _MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
            + _MAX_CANDIDATES_TOTAL
        )
    )
    summary: RosterSummary
    surface_evidence: tuple[SourceSurfaceEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    camera_evidence: tuple[SourceCameraEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    target_reachability: tuple[CandidateTargetReachability, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )

    @model_validator(mode="after")
    def validate_compilation(self) -> Self:
        if self.summary.selected_request_count != len(self.request_manifest.requests):
            raise ValueError("compilation request count does not close")
        if self.summary.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("compilation policy digest mismatch")
        if self.summary.manifest_sha256 != self.request_manifest.manifest_sha256:
            raise ValueError("compilation manifest digest mismatch")
        return self

    @model_validator(mode="after")
    def validate_surface_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.surface_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("surface evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.surface_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_capture_sha256 != capture.source_capture_sha256
            ):
                raise ValueError("surface evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_camera_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.camera_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("camera evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.camera_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_locator_sha256 != record.source.source_locator_sha256
                or evidence.source_capture_sha256 != capture.source_capture_sha256
                or evidence.camera != capture.scene.camera_by_id("main")
                or evidence.rgb_png_sha256 != capture.rgb_png_sha256
                or evidence.depth_npy_sha256 != capture.depth_npy_sha256
                or evidence.instance_png_sha256 != capture.instance_png_sha256
                or evidence.pointcloud_ply_sha256 != capture.pointcloud_ply_sha256
                or evidence.is_scene_at_rest is not capture.is_scene_at_rest
            ):
                raise ValueError("camera evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_target_reachability(self) -> Self:
        rows = self.target_reachability
        row_ids = tuple(item.candidate_id for item in rows)
        if row_ids != tuple(sorted(set(row_ids))):
            raise ValueError("target reachability rows are not canonical")
        candidates = {item.candidate_id: item for item in self.candidate_inventory}
        expected_ids = tuple(
            sorted(
                item.candidate_id
                for item in self.candidate_inventory
                if item.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and item.state
                in {
                    CompetitionNativeCandidateStateV2_9.SELECTED,
                    CompetitionNativeCandidateStateV2_9.REJECTED_POLICY_CAP,
                    CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE,
                }
            )
        )
        if row_ids != expected_ids:
            raise ValueError(
                "target reachability does not exactly cover receptacle selection inputs"
            )
        capture_by_source = {
            item.source.source_id: item.capture
            for item in self.scene_inventory
            if item.capture is not None
        }
        surface_by_source = {item.source_id: item for item in self.surface_evidence}
        camera_by_source = {item.source_id: item for item in self.camera_evidence}
        for row in rows:
            candidate = candidates[row.candidate_id]
            capture = capture_by_source.get(candidate.source_id)
            surface = surface_by_source.get(candidate.source_id)
            camera = camera_by_source.get(candidate.source_id)
            if capture is None or surface is None or camera is None:
                raise ValueError("target reachability source evidence is absent")
            placement = next(
                (
                    item
                    for item in capture.placement_facts
                    if item.object_id == candidate.subject_id
                ),
                None,
            )
            subject_surface = next(
                (
                    item
                    for item in surface.subjects
                    if item.subject_object_id == candidate.subject_id
                ),
                None,
            )
            unreachable = (
                candidate.state
                is CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE
            )
            if (
                placement is None
                or subject_surface is None
                or row.source_id != candidate.source_id
                or row.scene_id != candidate.scene_id
                or row.source_capture_sha256 != candidate.source_capture_sha256
                or row.subject_id != candidate.subject_id
                or row.reference_id != candidate.reference_id
                or row.relation_before is not candidate.relation_before
                or row.placement_sha256 != placement.placement_sha256
                or row.surface_evidence_sha256 != surface.surface_evidence_sha256
                or row.subject_surface_evidence_sha256
                != subject_surface.subject_surface_evidence_sha256
                or row.camera_evidence_sha256 != camera.camera_evidence_sha256
                or unreachable != (row.status is TargetReachabilityStatus.UNREACHABLE)
            ):
                raise ValueError(
                    "target reachability row does not bind candidate facts"
                )
        return self


CompetitionNativeCandidateRosterCompilationV2_9_4 = RosterCompilation


__all__ = (
    "CompetitionNativeCandidateInventoryV2_9",
    "CompetitionNativeCandidateRosterCompilationV2_9_4",
    "CompetitionNativeCandidateRosterManifestV2_9",
    "CompetitionNativeCandidateRosterPolicyV2_9_4",
    "CompetitionNativeCandidateRosterSummaryV2_9",
    "CompetitionNativeCandidateStateCountV2_9",
    "CompetitionNativeCandidateStateV2_9",
    "CompetitionNativeFloorEnvelopeV2_9",
    "CompetitionNativeObjectInventoryV2_9",
    "CompetitionNativePlacementAvailabilityV2_9",
    "CompetitionNativePositionV2_9",
    "CompetitionNativeRelationCountV2_9",
    "CompetitionNativeRosterRejectionV2_9",
    "CompetitionNativeRuntimeIdentityV2_9",
    "CompetitionNativeSelectedRequestV2_9",
    "CompetitionNativeSourceCaptureOutcomeV2_9",
    "CompetitionNativeSourceCaptureV2_9",
    "CompetitionNativeSourceRefV2_9",
    "CompetitionNativeStageCountV2_9",
    "CompetitionNativeSubjectPlacementFactV2_9",
    "CompetitionNativeSubjectStateV2_9",
    "CompetitionNativeSupportFactV2_9",
    "CompetitionNativeSupportKindV2_9",
    "DatasetSplitV2_9",
    "RosterCompilation",
    "RosterPolicy",
    "RosterSummary",
    "build_competition_native_source_capture_v2_9",
    "build_competition_native_subject_placement_fact_v2_9",
    "normalize_competition_native_source_scene_v2_9",
    "validate_competition_native_floor_envelope_v2_9",
    "validate_competition_native_runtime_source_lineage_v2_9",
)
