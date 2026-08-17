from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any
from weakref import ReferenceType, ref

import numpy as np
from PIL import Image

from spatialcf.adapters.base import RenderedAssets
from spatialcf.domain.models import (
    OBB,
    BBox2D,
    Camera,
    CollisionObstacle,
    ObjectView,
    Quaternion,
    Scene,
    SceneObject,
    SubjectPositionRegion,
    Vec2,
    Vec3,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2
from spatialcf.geometry.regions import (
    conservative_navigation_position_geometry,
    conservative_receptacle_position_geometry,
    planar_polygon_payloads,
)
from spatialcf.geometry.transforms import (
    ai2thor_position_to_world,
    ai2thor_rotation_to_world,
    matrix4,
    transform_point,
)

ControllerFactory = Callable[..., Any]
_EPSILON = 1e-6
_ANGLE_TOLERANCE_DEGREES = 1e-4
# Unity round-trips placed-object Euler angles at about 3e-4 degrees while the
# independently checked quaternion geometry remains stable within 1e-5.
_OBJECT_ROTATION_TOLERANCE_DEGREES = 1e-3
_CAMERA_POSITION_TOLERANCE_M = 1e-5
_OBJECT_GEOMETRY_TOLERANCE_M = 1e-5
_NATIVE_NAVIGATION_GRID_SIZE_M = 0.05
_REACHABLE_POSITION_QUANTIZATION_M = 1e-6
_RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M = 1e-5
_RECEPTACLE_TRIGGER_GRID_SIDE = 21
_RECEPTACLE_TRIGGER_GRID_SIZE = _RECEPTACLE_TRIGGER_GRID_SIDE**2
_TELEPORT_VERTICAL_GUARD_M = 1e-6
_RUNTIME_RECEPTACLE_POSITION_RESIDUAL_M = 1e-4
_STRUCTURAL_OBJECT_TYPES = frozenset({"Ceiling", "Floor", "Wall"})


def _camera_position_residual_m(
    requested: AI2ThorNativePosition,
    observed: AI2ThorNativePosition,
) -> float:
    """Return the total native-coordinate residual for one camera pose."""
    return math.dist(
        (requested.x, requested.y, requested.z),
        (observed.x, observed.y, observed.z),
    )


def _camera_position_residual_within_tolerance(
    requested: AI2ThorNativePosition,
    observed: AI2ThorNativePosition,
) -> tuple[float, bool]:
    """Apply the shared total-position camera contract with ULP allowance."""
    residual_m = _camera_position_residual_m(requested, observed)
    rounding_allowance_m = 4.0 * max(
        math.ulp(value)
        for value in (
            requested.x,
            requested.y,
            requested.z,
            observed.x,
            observed.y,
            observed.z,
        )
    )
    return (
        residual_m,
        residual_m <= _CAMERA_POSITION_TOLERANCE_M + rounding_allowance_m,
    )


def ai2thor_camera_world_to_camera(
    position: Vec3,
    *,
    yaw_degrees: float,
    horizon_degrees: float,
) -> tuple[float, ...]:
    """Build AI2-THOR's canonical world-to-camera extrinsic matrix."""

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(horizon_degrees)
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0])
    forward = np.asarray(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            -math.sin(pitch),
        ]
    )
    up = np.cross(right, forward)
    rotation = np.stack([right, up, forward])
    translation = -rotation @ np.asarray([position.x, position.y, position.z])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return tuple(float(value) for value in world_to_camera.reshape(-1))


class AI2ThorRuntimeError(RuntimeError):
    """Expected controller transport/launcher failure."""


class AI2ThorNativeReturnError(ValueError):
    """A successful native action returned a structurally invalid state."""


class AI2ThorSettlementTimeout(RuntimeError):
    """A bounded native settlement loop exhausted its explicit Pass budget."""


class AI2ThorNativeSupportKind(StrEnum):
    """Closed classification of one native parent lineage."""

    FLOOR = "FLOOR"
    RECEPTACLE = "RECEPTACLE"
    UNKNOWN = "UNKNOWN"
    MULTIPLE_AMBIGUOUS = "MULTIPLE_AMBIGUOUS"
    CYCLIC = "CYCLIC"


@dataclass(frozen=True)
class AI2ThorNativeSupportFact:
    """Read-only native parent evidence bound to one stable scene object."""

    scene_id: str
    object_id: str
    object_name: str
    native_object_id: str
    raw_parent_object_ids: tuple[str, ...]
    structural_parent_object_ids: tuple[str, ...]
    domain_parent_object_ids: tuple[str, ...]
    support_kind: AI2ThorNativeSupportKind
    support_object_id: str | None
    floor_object_id: str | None

    def __post_init__(self) -> None:
        for name in (
            "scene_id",
            "object_id",
            "object_name",
            "native_object_id",
        ):
            _nonempty_text(getattr(self, name), f"native support {name}")
        for name in (
            "raw_parent_object_ids",
            "structural_parent_object_ids",
            "domain_parent_object_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(f"native support {name} must be a text tuple")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"native support {name} must be unique and sorted")
        raw = set(self.raw_parent_object_ids)
        structural = set(self.structural_parent_object_ids)
        domain = set(self.domain_parent_object_ids)
        if not structural.issubset(raw) or not domain.issubset(raw):
            raise ValueError("native support parent partitions must be raw subsets")
        if structural.intersection(domain):
            raise ValueError("native support parent partitions must be disjoint")
        if type(self.support_kind) is not AI2ThorNativeSupportKind:
            raise ValueError("native support kind has invalid type")
        if self.support_kind is AI2ThorNativeSupportKind.RECEPTACLE:
            if (
                len(domain) != 1
                or structural
                or self.support_object_id != self.domain_parent_object_ids[0]
                or self.floor_object_id is not None
            ):
                raise ValueError("receptacle support fact is not closed")
        elif self.support_kind is AI2ThorNativeSupportKind.FLOOR:
            if (
                len(structural) != 1
                or domain
                or self.floor_object_id != self.structural_parent_object_ids[0]
                or self.support_object_id is not None
            ):
                raise ValueError("floor support fact is not closed")
        elif self.support_kind is AI2ThorNativeSupportKind.UNKNOWN:
            if domain or self.support_object_id is not None or self.floor_object_id is not None:
                raise ValueError("unknown support fact resolved an unusable parent")
        elif self.support_kind is AI2ThorNativeSupportKind.MULTIPLE_AMBIGUOUS:
            if (
                len(domain) + len(structural) <= 1
                or self.support_object_id is not None
                or self.floor_object_id is not None
            ):
                raise ValueError("ambiguous support fact requires multiple parents")
        elif (
            not domain
            or self.support_object_id is not None
            or self.floor_object_id is not None
        ):
            raise ValueError("cyclic support fact is not closed")


def _strict_finite_float(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def _nonempty_text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _full_commit_sha(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a complete lowercase commit SHA")
    return text


def _validate_json_tree(
    value: Any,
    *,
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("house JSON numbers must be finite")
        return
    if type(value) not in (dict, list):
        raise ValueError("house must contain exact JSON values")
    active = set() if active_containers is None else active_containers
    identity = id(value)
    if identity in active:
        raise ValueError("house JSON must not contain cycles")
    active.add(identity)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("house JSON objects must have string keys")
            for item in value.values():
                _validate_json_tree(item, active_containers=active)
        else:
            for item in value:
                _validate_json_tree(item, active_containers=active)
    finally:
        active.remove(identity)


def _canonical_house_json_bytes(house: Any) -> bytes:
    if type(house) is not dict:
        raise ValueError("house must be an exact dict")
    _validate_json_tree(house)
    try:
        return (
            json.dumps(
                house,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError("house must be finite canonical UTF-8 JSON") from error


def canonical_procedural_house_sha256(house: dict[str, Any]) -> str:
    """Return the canonical source digest without imposing a room policy."""
    if type(house) is not dict:
        raise ValueError("procedural house root must be an exact dict")
    return sha256(_canonical_house_json_bytes(house)).hexdigest()


def _procedural_room_identity(
    house: dict[str, Any],
) -> tuple[str, tuple[float, float, float, float]]:
    rooms = house.get("rooms")
    if type(rooms) is not list or len(rooms) != 1:
        raise ValueError("procedural house must contain exactly one room")
    room = rooms[0]
    if type(room) is not dict:
        raise ValueError("procedural room must be an exact dict")
    room_id = _nonempty_text(room.get("id"), "room id")
    polygon = room.get("floorPolygon")
    if type(polygon) is not list or len(polygon) != 4:
        raise ValueError("room floorPolygon must contain exactly four points")
    points: list[tuple[float, float]] = []
    elevations: list[float] = []
    for point in polygon:
        if type(point) is not dict:
            raise ValueError("room floorPolygon points must be exact dicts")
        coordinates = tuple(point.get(axis) for axis in ("x", "y", "z"))
        if any(
            type(coordinate) not in (int, float)
            or not math.isfinite(float(coordinate))
            for coordinate in coordinates
        ):
            raise ValueError("room floorPolygon points must be finite")
        points.append((float(coordinates[0]), float(coordinates[2])))
        elevations.append(float(coordinates[1]))
    if any(
        not math.isclose(value, elevations[0], rel_tol=0.0, abs_tol=1e-9)
        for value in elevations[1:]
    ):
        raise ValueError("room floorPolygon must lie on one horizontal plane")
    twice_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    if abs(twice_area) <= 1e-12:
        raise ValueError("room floorPolygon must have positive area")
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_z = min(point[1] for point in points)
    maximum_z = max(point[1] for point in points)
    if maximum_x - minimum_x <= 1e-12 or maximum_z - minimum_z <= 1e-12:
        raise ValueError("room floorPolygon must have positive area")
    expected_corners = {
        (minimum_x, minimum_z),
        (maximum_x, minimum_z),
        (maximum_x, maximum_z),
        (minimum_x, maximum_z),
    }
    actual_corners = set(points)
    edges_are_axis_aligned = all(
        (
            math.isclose(points[index][0], points[(index + 1) % 4][0], abs_tol=1e-12)
            != math.isclose(
                points[index][1],
                points[(index + 1) % 4][1],
                abs_tol=1e-12,
            )
        )
        for index in range(4)
    )
    if actual_corners != expected_corners or not edges_are_axis_aligned:
        raise ValueError(
            "room floorPolygon must be a convex axis-aligned rectangle"
        )
    return room_id, (minimum_x, minimum_z, maximum_x, maximum_z)


@dataclass(frozen=True)
class AI2ThorProceduralScene:
    """Immutable provenance plus canonical source bytes for one ProcTHOR house."""

    dataset_id: str
    revision: str
    split: str
    index: int
    source_loader_id: str
    source_loader_version: str
    canonical_house_json: bytes
    house_sha256: str
    room_id: str
    floor_xz_bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _nonempty_text(self.dataset_id, "dataset_id")
        _full_commit_sha(self.revision, "revision")
        if type(self.split) is not str or self.split not in {"train", "val", "test"}:
            raise ValueError("split must be exactly train, val, or test")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be an exact non-negative integer")
        _nonempty_text(self.source_loader_id, "source_loader_id")
        _nonempty_text(self.source_loader_version, "source_loader_version")
        if type(self.canonical_house_json) is not bytes:
            raise ValueError("canonical_house_json must be exact bytes")
        try:
            decoded = json.loads(self.canonical_house_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("canonical_house_json must be valid UTF-8 JSON") from error
        if type(decoded) is not dict:
            raise ValueError("canonical_house_json root must be an exact dict")
        room_id, floor_xz_bounds = _procedural_room_identity(decoded)
        if _canonical_house_json_bytes(decoded) != self.canonical_house_json:
            raise ValueError("canonical_house_json is not canonical")
        expected_sha256 = sha256(self.canonical_house_json).hexdigest()
        if type(self.house_sha256) is not str or self.house_sha256 != expected_sha256:
            raise ValueError("house_sha256 does not match canonical_house_json")
        if self.room_id != room_id:
            raise ValueError("room_id does not match canonical_house_json")
        if (
            type(self.floor_xz_bounds) is not tuple
            or self.floor_xz_bounds != floor_xz_bounds
        ):
            raise ValueError("floor_xz_bounds do not match canonical_house_json")

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        revision: str,
        split: str,
        index: int,
        source_loader_id: str,
        source_loader_version: str,
        house: dict[str, Any],
    ) -> AI2ThorProceduralScene:
        canonical = _canonical_house_json_bytes(house)
        room_id, floor_xz_bounds = _procedural_room_identity(house)
        return cls(
            dataset_id=dataset_id,
            revision=revision,
            split=split,
            index=index,
            source_loader_id=source_loader_id,
            source_loader_version=source_loader_version,
            canonical_house_json=canonical,
            house_sha256=sha256(canonical).hexdigest(),
            room_id=room_id,
            floor_xz_bounds=floor_xz_bounds,
        )

    def decode_house(self) -> dict[str, Any]:
        """Return a fresh mutable decoding; callers never receive retained state."""
        decoded = json.loads(self.canonical_house_json)
        if type(decoded) is not dict:  # Defends the public contract after construction.
            raise RuntimeError("procedural house root changed from an exact dict")
        return decoded

@dataclass(frozen=True)
class AI2ThorNativePosition:
    """A finite position in AI2-THOR's native X/Y/Z coordinate system."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            object.__setattr__(
                self,
                axis,
                _strict_finite_float(
                    getattr(self, axis),
                    f"native position {axis}",
                ),
            )


def canonicalize_ai2thor_reachable_positions(
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    """Return stable 1-micrometre identities for one native navigation grid."""

    if (
        type(positions) is not tuple
        or not positions
        or any(type(item) is not AI2ThorNativePosition for item in positions)
    ):
        raise TypeError("reachable positions must be a non-empty exact tuple")
    keyed: list[tuple[tuple[int, int, int], AI2ThorNativePosition]] = []
    seen: set[tuple[int, int, int]] = set()
    for position in positions:
        key = tuple(
            round(value / _REACHABLE_POSITION_QUANTIZATION_M)
            for value in (position.x, position.z, position.y)
        )
        if key in seen:
            raise ValueError("reachable positions contain a duplicate canonical point")
        seen.add(key)
        keyed.append(
            (
                key,
                AI2ThorNativePosition(
                    x=round(key[0] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                    y=round(key[2] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                    z=round(key[1] * _REACHABLE_POSITION_QUANTIZATION_M, 6),
                ),
            )
        )
    keyed.sort(key=lambda item: item[0])
    return tuple(position for _, position in keyed)


def bind_ai2thor_reachable_positions(
    reference_positions: tuple[AI2ThorNativePosition, ...],
    observed_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    """Bind one noisy replay to a frozen native navigation-grid roster."""

    frozen = canonicalize_ai2thor_reachable_positions(reference_positions)
    if (
        type(observed_positions) is not tuple
        or not observed_positions
        or any(
            type(item) is not AI2ThorNativePosition for item in observed_positions
        )
    ):
        raise TypeError("observed reachable positions must be a non-empty exact tuple")
    if len(observed_positions) != len(reference_positions):
        raise ValueError("reachable position roster changed")
    maximum_coordinate_ulp_m = max(
        math.ulp(value)
        for position in (*reference_positions, *observed_positions)
        for value in (position.x, position.y, position.z)
    )
    maximum_tolerance_m = _REACHABLE_POSITION_QUANTIZATION_M + 4.0 * max(
        maximum_coordinate_ulp_m,
        math.ulp(_REACHABLE_POSITION_QUANTIZATION_M),
    )
    bucket_span = math.ceil(
        maximum_tolerance_m / _REACHABLE_POSITION_QUANTIZATION_M
    )

    def bucket(position: AI2ThorNativePosition) -> tuple[int, int, int]:
        return tuple(
            math.floor(value / _REACHABLE_POSITION_QUANTIZATION_M)
            for value in (position.x, position.y, position.z)
        )

    reference_buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, reference in enumerate(reference_positions):
        reference_buckets.setdefault(bucket(reference), []).append(index)

    matched: set[int] = set()
    for observed in observed_positions:
        center = bucket(observed)
        candidates: list[int] = []
        if bucket_span <= 8:
            for x_offset in range(-bucket_span, bucket_span + 1):
                for y_offset in range(-bucket_span, bucket_span + 1):
                    for z_offset in range(-bucket_span, bucket_span + 1):
                        candidates.extend(
                            reference_buckets.get(
                                (
                                    center[0] + x_offset,
                                    center[1] + y_offset,
                                    center[2] + z_offset,
                                ),
                                (),
                            )
                        )
        else:
            candidates.extend(range(len(reference_positions)))
        within_tolerance = tuple(
            index
            for index in candidates
            if math.dist(
                (
                    reference_positions[index].x,
                    reference_positions[index].y,
                    reference_positions[index].z,
                ),
                (observed.x, observed.y, observed.z),
            )
            <= _REACHABLE_POSITION_QUANTIZATION_M
            + 4.0
            * max(
                *(
                    math.ulp(value)
                    for value in (
                        reference_positions[index].x,
                        reference_positions[index].y,
                        reference_positions[index].z,
                        observed.x,
                        observed.y,
                        observed.z,
                    )
                ),
                math.ulp(_REACHABLE_POSITION_QUANTIZATION_M),
            )
        )
        if len(within_tolerance) != 1 or within_tolerance[0] in matched:
            raise ValueError("reachable position roster changed")
        matched.add(within_tolerance[0])
    if len(matched) != len(reference_positions):
        raise ValueError("reachable position roster changed")
    return frozen


@dataclass(frozen=True)
class AI2ThorAgentPose:
    """A complete deterministic TeleportFull request in native coordinates."""

    position: AI2ThorNativePosition
    yaw_degrees: float
    horizon_degrees: float
    standing: bool

    def __post_init__(self) -> None:
        if type(self.position) is not AI2ThorNativePosition:
            raise ValueError("agent pose position must be an AI2ThorNativePosition")
        object.__setattr__(
            self,
            "yaw_degrees",
            _strict_finite_float(self.yaw_degrees, "agent yaw"),
        )
        object.__setattr__(
            self,
            "horizon_degrees",
            _strict_finite_float(self.horizon_degrees, "camera horizon"),
        )
        if type(self.standing) is not bool:
            raise ValueError("agent standing must be an exact boolean")


@dataclass(frozen=True)
class AI2ThorRuntimeIdentity:
    """Exact package, Unity build and controller contract for one run."""

    ai2thor_version: str
    unity_commit_id: str
    native_scene_name: str
    width: int
    height: int
    seed: int
    render_depth_image: bool = True
    render_instance_segmentation: bool = True
    grid_size_m: float = 0.05
    snap_to_grid: bool = True
    rotate_step_degrees: int = 90
    coordinate_transform_version: str = "ai2thor-native-xzy-to-rh-z-up-v1"
    source_dataset_id: str | None = None
    source_revision: str | None = None
    source_split: str | None = None
    source_index: int | None = None
    source_sha256: str | None = None
    source_scene_alias: str | None = None
    source_loader_id: str | None = None
    source_loader_version: str | None = None
    source_room_id: str | None = None
    source_floor_xz_bounds: tuple[float, float, float, float] | None = None
    teleport_vertical_guard_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "ai2thor_version",
            "unity_commit_id",
            "native_scene_name",
            "coordinate_transform_version",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        for name in ("width", "height", "seed", "rotate_step_degrees"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an exact integer")
        if self.width <= 0 or self.height <= 0 or self.rotate_step_degrees <= 0:
            raise ValueError("runtime dimensions and rotation step must be positive")
        for name in (
            "render_depth_image",
            "render_instance_segmentation",
            "snap_to_grid",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact boolean")
        if (
            isinstance(self.grid_size_m, bool)
            or not isinstance(self.grid_size_m, (int, float))
            or not math.isfinite(float(self.grid_size_m))
            or self.grid_size_m <= 0.0
        ):
            raise ValueError("grid_size_m must be finite and positive")
        object.__setattr__(self, "grid_size_m", float(self.grid_size_m))
        if (
            type(self.teleport_vertical_guard_m) not in (int, float)
            or not math.isfinite(float(self.teleport_vertical_guard_m))
            or float(self.teleport_vertical_guard_m) not in {0.0, 1e-6}
        ):
            raise ValueError("teleport_vertical_guard_m must be exactly 0 or 1e-6")
        object.__setattr__(
            self,
            "teleport_vertical_guard_m",
            float(self.teleport_vertical_guard_m),
        )
        source_values = (
            self.source_dataset_id,
            self.source_revision,
            self.source_split,
            self.source_index,
            self.source_sha256,
            self.source_scene_alias,
            self.source_loader_id,
            self.source_loader_version,
            self.source_room_id,
            self.source_floor_xz_bounds,
        )
        if any(value is not None for value in source_values):
            if any(value is None for value in source_values):
                raise ValueError("procedural source provenance must be complete")
            _nonempty_text(self.source_dataset_id, "source_dataset_id")
            _full_commit_sha(self.source_revision, "source_revision")
            _nonempty_text(self.source_scene_alias, "source_scene_alias")
            _nonempty_text(self.source_loader_id, "source_loader_id")
            _nonempty_text(self.source_loader_version, "source_loader_version")
            _nonempty_text(self.source_room_id, "source_room_id")
            if self.source_split not in {"train", "val", "test"}:
                raise ValueError("source_split must be exactly train, val, or test")
            if type(self.source_index) is not int or self.source_index < 0:
                raise ValueError("source_index must be an exact non-negative integer")
            if (
                type(self.source_sha256) is not str
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.source_sha256)
            ):
                raise ValueError("source_sha256 must be lowercase SHA-256 hex")
            bounds = self.source_floor_xz_bounds
            if (
                type(bounds) is not tuple
                or len(bounds) != 4
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    for value in bounds
                )
                or not float(bounds[0]) < float(bounds[2])
                or not float(bounds[1]) < float(bounds[3])
            ):
                raise ValueError(
                    "source_floor_xz_bounds must be a finite positive rectangle"
                )
            object.__setattr__(
                self,
                "source_floor_xz_bounds",
                tuple(float(value) for value in bounds),
            )
            if self.native_scene_name != "Procedural":
                raise ValueError("procedural provenance requires native Procedural scene")
            if self.teleport_vertical_guard_m != 1e-6:
                raise ValueError("procedural provenance requires 1e-6 teleport guard")
        elif self.native_scene_name == "Procedural":
            raise ValueError("native Procedural scene requires source provenance")
        elif self.teleport_vertical_guard_m not in {
            0.0,
            _TELEPORT_VERTICAL_GUARD_M,
        }:
            raise ValueError("legacy scene has an unsupported teleport guard")


@dataclass(frozen=True)
class AI2ThorObservation:
    """Immutable, same-event frames and scene state returned by AI2-THOR."""

    scene: Scene
    rgb_png: bytes
    depth_npy: bytes
    instance_png: bytes
    pointcloud_ply: bytes
    rgb_png_sha256: str
    depth_npy_sha256: str
    instance_png_sha256: str
    pointcloud_ply_sha256: str
    instance_pixel_counts: Mapping[str, int]
    is_scene_at_rest: bool

    @classmethod
    def create(
        cls,
        *,
        scene: Scene,
        rgb_png: bytes,
        depth_npy: bytes,
        instance_png: bytes,
        pointcloud_ply: bytes,
        instance_pixel_counts: Mapping[str, int],
        is_scene_at_rest: bool,
    ) -> AI2ThorObservation:
        return cls(
            scene=scene,
            rgb_png=rgb_png,
            depth_npy=depth_npy,
            instance_png=instance_png,
            pointcloud_ply=pointcloud_ply,
            rgb_png_sha256=sha256(rgb_png).hexdigest(),
            depth_npy_sha256=sha256(depth_npy).hexdigest(),
            instance_png_sha256=sha256(instance_png).hexdigest(),
            pointcloud_ply_sha256=sha256(pointcloud_ply).hexdigest(),
            instance_pixel_counts=MappingProxyType(
                dict(sorted(instance_pixel_counts.items()))
            ),
            is_scene_at_rest=is_scene_at_rest,
        )


@dataclass(frozen=True)
class AI2ThorCameraApplication:
    """A requested camera pose and its immutable same-event observation."""

    requested_pose: AI2ThorAgentPose
    observed_pose: AI2ThorAgentPose
    observed_camera_position: AI2ThorNativePosition
    observed_scene: Scene
    observation: AI2ThorObservation
    position_residual_m: float
    yaw_residual_degrees: float
    horizon_residual_degrees: float


@dataclass(frozen=True)
class AI2ThorSettledCameraApplication:
    """One post-unpause camera application plus its final settlement count."""

    application: AI2ThorCameraApplication
    settlement_pass_steps: int

    def __post_init__(self) -> None:
        if type(self.application) is not AI2ThorCameraApplication:
            raise TypeError("settled camera application must be exact")
        if type(self.settlement_pass_steps) is not int:
            raise TypeError("settled camera pass count must be an exact integer")
        if self.settlement_pass_steps < 0:
            raise ValueError("settled camera pass count must be non-negative")


@dataclass(frozen=True)
class AI2ThorSceneSettlement:
    """A fully still source baseline captured from one final native event."""

    observed_scene: Scene
    observation: AI2ThorObservation
    pass_steps: int


@dataclass(frozen=True)
class AI2ThorIsolatedEpisode:
    """One fresh controller plus its immutable same-event source baseline."""

    adapter: AI2ThorAdapter
    baseline_settlement: AI2ThorSceneSettlement


@dataclass(frozen=True)
class AI2ThorPoseApplication:
    """Keep the commanded canonical state separate from native observation."""

    commanded_scene: Scene
    observed_scene: Scene
    commanded_position: Vec3
    observed_position: Vec3
    position_residual_m: float
    observation: AI2ThorObservation
    is_scene_at_rest: bool
    subject_is_moving: bool


@dataclass(frozen=True)
class AI2ThorFloorEnvelope:
    """Conservative convex floor evidence derived from one native floor AABB."""

    scene_id: str
    floor_object_id: str
    floor_name: str
    native_aabb: OBB
    floor_top_z: float
    clearance_m: float
    polygon_xy: tuple[Vec2, ...]


@dataclass(frozen=True)
class AI2ThorNativeFeasibilityMap:
    """Conservative native collision envelopes for one exact source event."""

    scene_id: str
    subject_object_id: str
    clearance_m: float
    obstacles: tuple[CollisionObstacle, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.scene_id, "feasibility scene_id")
        _nonempty_text(self.subject_object_id, "feasibility subject_object_id")
        if (
            type(self.clearance_m) not in (int, float)
            or not math.isfinite(float(self.clearance_m))
            or float(self.clearance_m) <= 0.0
        ):
            raise ValueError("collision clearance must be finite and positive")
        object.__setattr__(self, "clearance_m", float(self.clearance_m))
        if type(self.obstacles) is not tuple or any(
            type(item) is not CollisionObstacle for item in self.obstacles
        ):
            raise ValueError("feasibility obstacles must be a CollisionObstacle tuple")
        obstacle_ids = tuple(item.obstacle_id for item in self.obstacles)
        source_ids = tuple(item.source_object_id for item in self.obstacles)
        if len(set(obstacle_ids)) != len(obstacle_ids):
            raise ValueError("feasibility obstacle IDs must be unique")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("feasibility source object IDs must be unique")
        if any(item.clearance_m != self.clearance_m for item in self.obstacles):
            raise ValueError("feasibility obstacle clearance mismatch")


@dataclass(frozen=True)
class AI2ThorReceptacleSurfacePatch:
    """One complete native 21-by-21 receptacle trigger grid."""

    x_min: float
    x_max: float
    native_y: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.native_y, self.z_min, self.z_max)
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("receptacle surface patch values must be finite")
        for name, value in zip(
            ("x_min", "x_max", "native_y", "z_min", "z_max"), values
        ):
            object.__setattr__(self, name, float(value))
        if self.x_min >= self.x_max or self.z_min >= self.z_max:
            raise ValueError("receptacle surface patch must have positive area")


def _grid_axis(
    values: tuple[float, ...],
) -> tuple[tuple[int, ...], dict[int, tuple[float, ...]]] | None:
    grouped: dict[int, list[float]] = {}
    for value in values:
        key = round(value / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M)
        grouped.setdefault(key, []).append(value)
    keys = tuple(sorted(grouped))
    if len(keys) != _RECEPTACLE_TRIGGER_GRID_SIDE:
        return None
    actual_min = min(values)
    actual_max = max(values)
    if actual_min >= actual_max:
        return None
    tolerance = 2.0 * _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M
    for index, key in enumerate(keys):
        expected = actual_min + (actual_max - actual_min) * index / 20.0
        if any(abs(value - expected) > tolerance for value in grouped[key]):
            return None
    return keys, {key: tuple(grouped[key]) for key in keys}


def build_ai2thor_receptacle_surface_patches(
    raw_positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorReceptacleSurfacePatch, ...]:
    """Fail closed unless every raw contiguous block is one complete grid."""

    if (
        type(raw_positions) is not tuple
        or not raw_positions
        or len(raw_positions) % _RECEPTACLE_TRIGGER_GRID_SIZE
        or any(type(item) is not AI2ThorNativePosition for item in raw_positions)
    ):
        return ()
    patches: list[AI2ThorReceptacleSurfacePatch] = []
    for start in range(0, len(raw_positions), _RECEPTACLE_TRIGGER_GRID_SIZE):
        block = raw_positions[start : start + _RECEPTACLE_TRIGGER_GRID_SIZE]
        y_keys = {
            round(item.y / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M)
            for item in block
        }
        x_axis = _grid_axis(tuple(item.x for item in block))
        z_axis = _grid_axis(tuple(item.z for item in block))
        if len(y_keys) != 1 or x_axis is None or z_axis is None:
            return ()
        x_keys, _ = x_axis
        z_keys, _ = z_axis
        cells = tuple(
            (
                round(item.x / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M),
                round(item.z / _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M),
            )
            for item in block
        )
        expected_cells = {(x_key, z_key) for x_key in x_keys for z_key in z_keys}
        if len(set(cells)) != _RECEPTACLE_TRIGGER_GRID_SIZE or set(cells) != expected_cells:
            return ()
        patches.append(
            AI2ThorReceptacleSurfacePatch(
                x_min=min(item.x for item in block),
                x_max=max(item.x for item in block),
                native_y=(
                    min(item.y for item in block) + max(item.y for item in block)
                )
                / 2.0,
                z_min=min(item.z for item in block),
                z_max=max(item.z for item in block),
            )
        )
    return tuple(
        sorted(
            patches,
            key=lambda item: (
                item.native_y,
                item.x_min,
                item.z_min,
                item.x_max,
                item.z_max,
            ),
        )
    )


@dataclass(frozen=True)
class AI2ThorReceptacleSpawnMap:
    """Source-bound receptacle coordinates without a feasibility claim.

    AI2-THOR's ``GetSpawnCoordinatesAboveReceptacle`` action describes native
    receptacle coordinates.  It does not prove that a particular subject fits
    at every returned coordinate, so this value deliberately avoids the word
    ``feasible`` and remains adapter evidence rather than a solver domain.
    """

    scene_id: str
    subject_object_id: str
    support_object_id: str
    native_subject_object_id: str
    native_support_object_id: str
    runtime_identity: AI2ThorRuntimeIdentity
    positions: tuple[AI2ThorNativePosition, ...]
    positions_sha256: str
    scene_sha256: str
    source_sha256: str
    surface_patches: tuple[AI2ThorReceptacleSurfacePatch, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "scene_id",
            "subject_object_id",
            "support_object_id",
            "native_subject_object_id",
            "native_support_object_id",
        ):
            _nonempty_text(getattr(self, field_name), field_name)
        if self.subject_object_id == self.support_object_id:
            raise ValueError("receptacle spawn subject and support must differ")
        if type(self.runtime_identity) is not AI2ThorRuntimeIdentity:
            raise ValueError("receptacle spawn runtime identity has invalid type")
        checked_runtime = AI2ThorRuntimeIdentity(**asdict(self.runtime_identity))
        object.__setattr__(self, "runtime_identity", checked_runtime)
        if type(self.surface_patches) is not tuple or any(
            type(item) is not AI2ThorReceptacleSurfacePatch
            for item in self.surface_patches
        ):
            raise ValueError("receptacle surface patches must be an exact tuple")
        checked_patches = tuple(
            AI2ThorReceptacleSurfacePatch(**asdict(item))
            for item in self.surface_patches
        )
        if checked_patches != tuple(
            sorted(
                set(checked_patches),
                key=lambda item: (
                    item.native_y,
                    item.x_min,
                    item.z_min,
                    item.x_max,
                    item.z_max,
                ),
            )
        ):
            raise ValueError("receptacle surface patches must be unique and canonical")
        object.__setattr__(self, "surface_patches", checked_patches)
        if type(self.positions) is not tuple or not self.positions:
            raise ValueError("receptacle spawn positions must be a non-empty tuple")
        if any(type(item) is not AI2ThorNativePosition for item in self.positions):
            raise ValueError("receptacle spawn positions must be native positions")
        keys = tuple((item.x, item.z, item.y) for item in self.positions)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("receptacle spawn positions must be unique and sorted")
        for field_name in ("positions_sha256", "scene_sha256", "source_sha256"):
            digest = getattr(self, field_name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{field_name} must be lowercase SHA-256 hex")
        if self.positions_sha256 != _native_positions_sha256(self.positions):
            raise ValueError("receptacle spawn positions digest mismatch")
        if self.surface_patches:
            expected_positions = tuple(
                sorted(
                    (
                        patch.x_min
                        + (patch.x_max - patch.x_min) * x_index / 20.0,
                        patch.z_min
                        + (patch.z_max - patch.z_min) * z_index / 20.0,
                        patch.native_y,
                    )
                    for patch in self.surface_patches
                    for x_index in range(_RECEPTACLE_TRIGGER_GRID_SIDE)
                    for z_index in range(_RECEPTACLE_TRIGGER_GRID_SIDE)
                )
            )
            actual_positions = tuple(
                (item.x, item.z, item.y) for item in self.positions
            )
            tolerance = 2.0 * _RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M
            expected_count = (
                len(self.surface_patches) * _RECEPTACLE_TRIGGER_GRID_SIZE
            )
            if (
                len(expected_positions) != expected_count
                or len(actual_positions) != expected_count
                or any(
                    abs(expected - actual) > tolerance
                    for expected_position, actual_position in zip(
                        expected_positions,
                        actual_positions,
                        strict=True,
                    )
                    for expected, actual in zip(
                        expected_position,
                        actual_position,
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "receptacle surface patches do not close the position grid"
                )
        if self.source_sha256 != _receptacle_spawn_source_sha256(
            scene_id=self.scene_id,
            subject_object_id=self.subject_object_id,
            support_object_id=self.support_object_id,
            native_subject_object_id=self.native_subject_object_id,
            native_support_object_id=self.native_support_object_id,
            runtime_identity=self.runtime_identity,
            positions_sha256=self.positions_sha256,
            scene_sha256=self.scene_sha256,
            surface_patches=self.surface_patches,
        ):
            raise ValueError("receptacle spawn source digest mismatch")


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_scene_sha256(scene: Scene) -> str:
    """Hash source facts independently of unordered roster presentation."""

    if type(scene) is not Scene:
        raise TypeError("canonical scene digest requires an exact Scene")
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
    payload = normalized.model_dump(mode="python", warnings="error")
    payload["pinned_object_ids"] = tuple(sorted(scene.pinned_object_ids))
    return sha256(canonical_json_bytes_v2(payload)).hexdigest()


def _receptacle_scene_sha256(
    scene: Scene,
    surface_patches: tuple[AI2ThorReceptacleSurfacePatch, ...],
) -> str:
    """Preserve the legacy digest unless the new grid contract is active."""

    if surface_patches:
        return _canonical_scene_sha256(scene)
    return _canonical_json_sha256(scene.model_dump(mode="json", warnings="error"))


def _native_positions_sha256(
    positions: tuple[AI2ThorNativePosition, ...],
) -> str:
    return _canonical_json_sha256(
        tuple({"x": item.x, "y": item.y, "z": item.z} for item in positions)
    )


def _receptacle_spawn_source_sha256(
    *,
    scene_id: str,
    subject_object_id: str,
    support_object_id: str,
    native_subject_object_id: str,
    native_support_object_id: str,
    runtime_identity: AI2ThorRuntimeIdentity,
    positions_sha256: str,
    scene_sha256: str,
    surface_patches: tuple[AI2ThorReceptacleSurfacePatch, ...] = (),
) -> str:
    payload: dict[str, object] = {
            "action": "GetSpawnCoordinatesAboveReceptacle",
            "anywhere": True,
            "method": (
                "ai2thor-receptacle-trigger-grid-v1"
                if surface_patches
                else "ai2thor-receptacle-spawn-map-v1"
            ),
            "native_subject_object_id": native_subject_object_id,
            "native_support_object_id": native_support_object_id,
            "positions_sha256": positions_sha256,
            "runtime_identity": asdict(runtime_identity),
            "scene_id": scene_id,
            "scene_sha256": scene_sha256,
            "subject_object_id": subject_object_id,
            "support_object_id": support_object_id,
        }
    if surface_patches:
        payload["surface_patches"] = tuple(asdict(item) for item in surface_patches)
    return _canonical_json_sha256(payload)


def _strict_native_position(
    value: object,
    label: str,
) -> AI2ThorNativePosition:
    if type(value) is not AI2ThorNativePosition:
        raise TypeError(f"{label} must be an exact AI2ThorNativePosition")
    return AI2ThorNativePosition(x=value.x, y=value.y, z=value.z)


def _strict_receptacle_spawn_map(
    value: object,
) -> AI2ThorReceptacleSpawnMap:
    if type(value) is not AI2ThorReceptacleSpawnMap:
        raise TypeError("spawn_map must be an exact AI2ThorReceptacleSpawnMap")
    if type(value.positions) is not tuple:
        raise TypeError("spawn map positions must be an exact tuple")
    positions = tuple(
        _strict_native_position(item, "spawn map position") for item in value.positions
    )
    if type(value.runtime_identity) is not AI2ThorRuntimeIdentity:
        raise TypeError("spawn map runtime identity has invalid type")
    return AI2ThorReceptacleSpawnMap(
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        runtime_identity=AI2ThorRuntimeIdentity(**asdict(value.runtime_identity)),
        positions=positions,
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=value.surface_patches,
    )


def capture_bound_ai2thor_receptacle_spawn_map(
    spawn_map: AI2ThorReceptacleSpawnMap,
    *,
    fresh_scene: Scene,
    frozen_scene: Scene,
) -> AI2ThorReceptacleSpawnMap:
    """Rebind one exact fresh native query to a validated frozen scene digest.

    Only the scene digest is substituted. Runtime identity, native IDs, exact
    native positions and trigger-grid patches all remain those returned by the
    fresh query, so drift in any native evidence still changes the source
    digest and fails its upstream lineage comparison.
    """

    checked = _strict_receptacle_spawn_map(spawn_map)
    if type(fresh_scene) is not Scene or type(frozen_scene) is not Scene:
        raise TypeError("capture-bound spawn scenes must be exact Scene values")
    expected_fresh_scene_sha256 = _receptacle_scene_sha256(
        fresh_scene,
        checked.surface_patches,
    )
    if (
        checked.scene_id != fresh_scene.scene_id
        or checked.scene_sha256 != expected_fresh_scene_sha256
        or frozen_scene.scene_id != fresh_scene.scene_id
    ):
        raise ValueError("capture-bound spawn map does not bind the fresh scene")
    frozen_scene_sha256 = _receptacle_scene_sha256(
        frozen_scene,
        checked.surface_patches,
    )
    source_sha256 = _receptacle_spawn_source_sha256(
        scene_id=checked.scene_id,
        subject_object_id=checked.subject_object_id,
        support_object_id=checked.support_object_id,
        native_subject_object_id=checked.native_subject_object_id,
        native_support_object_id=checked.native_support_object_id,
        runtime_identity=checked.runtime_identity,
        positions_sha256=checked.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        surface_patches=checked.surface_patches,
    )
    return AI2ThorReceptacleSpawnMap(
        scene_id=checked.scene_id,
        subject_object_id=checked.subject_object_id,
        support_object_id=checked.support_object_id,
        native_subject_object_id=checked.native_subject_object_id,
        native_support_object_id=checked.native_support_object_id,
        runtime_identity=checked.runtime_identity,
        positions=checked.positions,
        positions_sha256=checked.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        source_sha256=source_sha256,
        surface_patches=checked.surface_patches,
    )


def build_receptacle_support_position_region(
    scene: Scene,
    spawn_map: AI2ThorReceptacleSpawnMap,
) -> SubjectPositionRegion:
    """Build the fixed-pose subject-anchor locus from trigger-grid evidence."""

    checked = _strict_receptacle_spawn_map(spawn_map)
    subject = scene.object_by_id(checked.subject_object_id)
    scene_sha256 = _receptacle_scene_sha256(scene, checked.surface_patches)
    if (
        checked.scene_id != scene.scene_id
        or checked.scene_sha256 != scene_sha256
        or subject.support_object_id != checked.support_object_id
        or not checked.surface_patches
    ):
        raise ValueError("receptacle surface patches do not bind the scene subject")
    geometry = conservative_receptacle_position_geometry(
        surface_patch_bounds_xy=tuple(
            (item.x_min, item.z_min, item.x_max, item.z_max)
            for item in checked.surface_patches
        ),
        subject=subject,
    )
    return SubjectPositionRegion(
        region_id=(
            f"native-receptacle-trigger-grid:{subject.object_id}:"
            f"{checked.source_sha256[:16]}"
        ),
        subject_object_id=subject.object_id,
        source_kind="ai2thor-receptacle-trigger-grid-v1",
        source_sha256=checked.source_sha256,
        components=planar_polygon_payloads(geometry),
    )


@dataclass(frozen=True)
class AI2ThorNavigationFeasibilityMap:
    """Source-bound conservative position region from one native nav grid."""

    scene_id: str
    subject_object_id: str
    agent_radius_m: float
    clearance_m: float
    reachable_positions: tuple[AI2ThorNativePosition, ...]
    reachable_positions_sha256: str
    source_sha256: str
    position_region: SubjectPositionRegion

    def __post_init__(self) -> None:
        _nonempty_text(self.scene_id, "navigation scene_id")
        _nonempty_text(self.subject_object_id, "navigation subject_object_id")
        radius = _strict_finite_float(self.agent_radius_m, "navigation agent radius")
        clearance = _strict_finite_float(
            self.clearance_m,
            "navigation clearance",
        )
        if radius <= 0.0 or clearance < 0.0 or clearance >= radius:
            raise ValueError(
                "navigation agent radius must be positive and clearance smaller"
            )
        object.__setattr__(self, "agent_radius_m", radius)
        object.__setattr__(self, "clearance_m", clearance)
        if type(self.reachable_positions) is not tuple or not self.reachable_positions:
            raise ValueError("navigation reachable positions must be a non-empty tuple")
        if any(
            type(position) is not AI2ThorNativePosition
            for position in self.reachable_positions
        ):
            raise ValueError("navigation positions must be native positions")
        keys = tuple(
            (position.x, position.z, position.y)
            for position in self.reachable_positions
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("navigation positions must be unique and sorted")
        for name in ("reachable_positions_sha256", "source_sha256"):
            digest = getattr(self, name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if type(self.position_region) is not SubjectPositionRegion:
            raise ValueError("navigation position region has invalid type")
        if (
            self.position_region.subject_object_id != self.subject_object_id
            or self.position_region.source_sha256 != self.source_sha256
        ):
            raise ValueError("navigation position region identity mismatch")


def build_navigation_feasibility_map(
    scene: Scene,
    *,
    subject_object_id: str,
    room_polygon_xy: tuple[Vec2, ...],
    reachable_positions: tuple[AI2ThorNativePosition, ...],
    agent_radius_m: float,
    clearance_m: float,
) -> AI2ThorNavigationFeasibilityMap:
    """Deterministically bind native navigation evidence to one Scene."""
    if (
        type(agent_radius_m) not in (int, float)
        or not math.isfinite(float(agent_radius_m))
        or float(agent_radius_m) <= 0.0
        or type(clearance_m) not in (int, float)
        or not math.isfinite(float(clearance_m))
        or float(clearance_m) < 0.0
        or float(clearance_m) >= float(agent_radius_m)
    ):
        raise ValueError(
            "navigation radius must be positive and clearance non-negative "
            "and smaller than the radius"
        )
    subject = scene.object_by_id(subject_object_id)
    positions_payload = [
        {"x": item.x, "y": item.y, "z": item.z}
        for item in reachable_positions
    ]
    positions_bytes = json.dumps(
        positions_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    positions_digest = sha256(positions_bytes).hexdigest()
    source_payload = {
        "agent_radius_m": float(agent_radius_m),
        "clearance_m": float(clearance_m),
        "method": "ai2thor-navigation-v1",
        "reachable_positions_sha256": positions_digest,
        "room_polygon_xy": [
            {"x": point.x, "y": point.y} for point in room_polygon_xy
        ],
        "scene_id": scene.scene_id,
        "subject_object_id": subject.object_id,
        "subject_obb": subject.obb.model_dump(mode="json"),
        "subject_position": subject.position.model_dump(mode="json"),
    }
    source_digest = sha256(
        json.dumps(
            source_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    geometry = conservative_navigation_position_geometry(
        room_polygon_xy=room_polygon_xy,
        reachable_positions_xy=tuple(
            Vec2(x=position.x, y=position.z) for position in reachable_positions
        ),
        subject=subject,
        agent_radius_m=float(agent_radius_m),
        clearance_m=float(clearance_m),
    )
    position_region = SubjectPositionRegion(
        region_id=f"native-navigation:{subject.object_id}:{source_digest[:16]}",
        subject_object_id=subject.object_id,
        source_kind="ai2thor-navigation-v1",
        source_sha256=source_digest,
        components=planar_polygon_payloads(geometry),
    )
    return AI2ThorNavigationFeasibilityMap(
        scene_id=scene.scene_id,
        subject_object_id=subject.object_id,
        agent_radius_m=float(agent_radius_m),
        clearance_m=float(clearance_m),
        reachable_positions=reachable_positions,
        reachable_positions_sha256=positions_digest,
        source_sha256=source_digest,
        position_region=position_region,
    )


def _domain_object_metadata(raw_objects: list[Any]) -> list[Any]:
    return [
        item
        for item in raw_objects
        if (
            not isinstance(item, dict)
            or item.get("objectType") not in _STRUCTURAL_OBJECT_TYPES
        )
    ]


def _validated_native_object_metadata(
    raw_objects: list[Any],
) -> list[dict[str, Any]]:
    validated_objects: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    object_names: set[str] = set()
    for item in raw_objects:
        if type(item) is not dict:
            raise AI2ThorNativeReturnError(
                "AI2-THOR object collection entries must be exact dictionaries"
            )
        object_id = item.get("objectId")
        name = item.get("name")
        category = item.get("objectType")
        if any(
            type(value) is not str or not value.strip()
            for value in (object_id, name, category)
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR object ID, name, and type must be non-empty text"
            )
        if object_id in object_ids or name in object_names:
            raise AI2ThorNativeReturnError(
                "AI2-THOR object IDs and names must be unique"
            )
        object_ids.add(object_id)
        object_names.add(name)
        if any(
            field in item and type(item[field]) is not bool
            for field in ("moveable", "pickupable")
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR object mobility fields must be exact booleans"
            )
        parents = item.get("parentReceptacles")
        if parents is not None and (
            type(parents) is not list
            or any(type(parent) is not str or not parent.strip() for parent in parents)
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR parent receptacles must be null or a list of "
                "non-empty text IDs"
            )
        validated_objects.append(item)

    for item in validated_objects:
        for parent in item.get("parentReceptacles") or []:
            if parent not in object_ids:
                raise AI2ThorNativeReturnError(
                    f"observed support {parent!r} has no stable object identity"
                )

    return [
        {
            **item,
            "parentReceptacles": list(item.get("parentReceptacles") or []),
        }
        for item in validated_objects
    ]


def _cyclic_domain_object_ids(
    graph: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    """Return every member of every directed cycle, including self-loops."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for parent in graph.get(node, ()):
            if parent not in graph:
                continue
            if parent not in indices:
                visit(parent)
                lowlinks[node] = min(lowlinks[node], lowlinks[parent])
            elif parent in active:
                lowlinks[node] = min(lowlinks[node], indices[parent])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            active.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in graph.get(node, ()):
            cyclic.update(component)

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return frozenset(cyclic)


def build_ai2thor_native_support_facts(
    scene: Scene,
    raw_objects: list[object],
) -> tuple[AI2ThorNativeSupportFact, ...]:
    """Validate and normalize native parent lineage without a platform action."""

    if type(scene) is not Scene:
        raise TypeError("native support scene must be an exact Scene")
    if type(raw_objects) is not list:
        raise TypeError("native support raw_objects must be an exact list")
    validated = _validated_native_object_metadata(raw_objects)
    raw_by_id = {
        item["objectId"]: (item["name"], item["objectType"]) for item in validated
    }
    scene_ids = tuple(item.object_id for item in scene.objects)
    scene_names = tuple(item.name for item in scene.objects)
    if len(scene_ids) != len(set(scene_ids)) or len(scene_names) != len(set(scene_names)):
        raise AI2ThorNativeReturnError(
            "stable scene object IDs and names must be unique"
        )
    scene_by_name = {item.name: item for item in scene.objects}
    raw_domain_names = {
        item["name"] for item in validated if item["objectType"] not in _STRUCTURAL_OBJECT_TYPES
    }
    if set(scene_by_name) != raw_domain_names:
        raise AI2ThorNativeReturnError(
            "native and stable scene object name rosters must match"
        )
    native_by_name = {item["name"]: item for item in validated}

    normalized: dict[
        str,
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ] = {}
    for object_name, scene_object in scene_by_name.items():
        native = native_by_name[object_name]
        raw_parents = tuple(sorted(set(native["parentReceptacles"])))
        structural: list[str] = []
        domain: list[str] = []
        normalized_raw: list[str] = []
        for raw_parent in raw_parents:
            parent_name, parent_type = raw_by_id[raw_parent]
            if parent_type in _STRUCTURAL_OBJECT_TYPES:
                structural.append(raw_parent)
                normalized_raw.append(raw_parent)
            else:
                try:
                    stable_parent = scene_by_name[parent_name].object_id
                except KeyError as error:
                    raise AI2ThorNativeReturnError(
                        f"observed support {raw_parent!r} has no stable object identity"
                    ) from error
                domain.append(stable_parent)
                normalized_raw.append(stable_parent)
        normalized[scene_object.object_id] = (
            object_name,
            native["objectId"],
            tuple(sorted(set(normalized_raw))),
            tuple(sorted(set(structural))),
            tuple(sorted(set(domain))),
        )

    graph = {object_id: values[4] for object_id, values in normalized.items()}
    cyclic = _cyclic_domain_object_ids(graph)
    facts: list[AI2ThorNativeSupportFact] = []
    for object_id in sorted(normalized):
        object_name, native_object_id, raw, structural, domain = normalized[object_id]
        floor_parents = tuple(
            parent for parent in structural if raw_by_id[parent][1] == "Floor"
        )
        plausible_count = len(domain) + len(floor_parents)
        if object_id in cyclic:
            kind = AI2ThorNativeSupportKind.CYCLIC
            support_object_id = None
            floor_object_id = None
        elif plausible_count > 1:
            kind = AI2ThorNativeSupportKind.MULTIPLE_AMBIGUOUS
            support_object_id = None
            floor_object_id = None
        elif len(domain) == 1 and not structural:
            kind = AI2ThorNativeSupportKind.RECEPTACLE
            support_object_id = domain[0]
            floor_object_id = None
        elif len(floor_parents) == 1 and not domain and len(structural) == 1:
            kind = AI2ThorNativeSupportKind.FLOOR
            support_object_id = None
            floor_object_id = floor_parents[0]
        else:
            kind = AI2ThorNativeSupportKind.UNKNOWN
            support_object_id = None
            floor_object_id = None
            # Structural parents other than Floor are raw provenance, not usable
            # support. Keep them out of the structural support partition so the
            # UNKNOWN invariant remains explicit.
            structural = ()
        facts.append(
            AI2ThorNativeSupportFact(
                scene_id=scene.scene_id,
                object_id=object_id,
                object_name=object_name,
                native_object_id=native_object_id,
                raw_parent_object_ids=raw,
                structural_parent_object_ids=structural,
                domain_parent_object_ids=domain,
                support_kind=kind,
                support_object_id=support_object_id,
                floor_object_id=floor_object_id,
            )
        )
    return tuple(facts)


def _load_default_controller_type() -> type[Any]:
    try:
        with warnings.catch_warnings():
            # AI2-THOR 5.0.0 contains escaped spaces in diagnostic-only string
            # literals. Import it inside callers' fail-closed warning scopes
            # without weakening warnings emitted by our code or at runtime.
            warnings.filterwarnings(
                "ignore",
                message=r"invalid escape sequence '\\ '",
                category=DeprecationWarning,
            )
            from ai2thor.controller import Controller
    except ImportError as exc:
        raise RuntimeError(
            "AI2-THOR is not installed. Install spatialcf[sim] with Python 3.11."
        ) from exc
    return Controller


def _default_controller_factory(**kwargs: Any) -> Any:
    return _load_default_controller_type()(**kwargs)


def _quaternion_yaw(rotation: Quaternion) -> float:
    norm = math.sqrt(
        rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2
    )
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise ValueError("object rotation must be a finite non-zero quaternion")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotation_matrix(rotation: Quaternion) -> np.ndarray:
    norm = math.sqrt(
        rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2
    )
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise ValueError("rotation must be a finite non-zero quaternion")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def _quaternions_close(left: Quaternion, right: Quaternion) -> bool:
    a = np.asarray([left.x, left.y, left.z, left.w], dtype=float)
    b = np.asarray([right.x, right.y, right.z, right.w], dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return bool(
        np.allclose(a, b, atol=1e-5, rtol=0.0)
        or np.allclose(a, -b, atol=1e-5, rtol=0.0)
    )


class AI2ThorAdapter:
    def __init__(
        self,
        scene_names: list[str],
        width: int,
        height: int,
        seed: int,
        *,
        controller_factory: ControllerFactory | None = None,
        allow_source_pose_drift: bool = False,
        procedural_scenes: Mapping[str, AI2ThorProceduralScene] | None = None,
    ) -> None:
        if type(scene_names) is not list or any(
            type(name) is not str for name in scene_names
        ):
            raise ValueError("scene_names must be an exact string list")
        if type(width) is not int or type(height) is not int:
            raise ValueError("render dimensions must be exact integers")
        if type(seed) is not int:
            raise ValueError("seed must be an exact integer")
        if type(allow_source_pose_drift) is not bool:
            raise ValueError("allow_source_pose_drift must be an exact boolean")
        if not scene_names:
            raise ValueError("at least one scene is required")
        if len(set(scene_names)) != len(scene_names):
            raise ValueError("scene names must be unique")
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        if procedural_scenes is None:
            procedural_by_alias: dict[str, AI2ThorProceduralScene] = {}
        else:
            if not isinstance(procedural_scenes, Mapping):
                raise ValueError("procedural_scenes must be a mapping")
            procedural_by_alias = {}
            for alias, source in procedural_scenes.items():
                if type(alias) is not str:
                    raise ValueError("procedural_scenes keys must be exact strings")
                if type(source) is not AI2ThorProceduralScene:
                    raise ValueError(
                        "procedural_scenes values must be AI2ThorProceduralScene"
                    )
                procedural_by_alias[alias] = source
            if not set(procedural_by_alias).issubset(scene_names):
                raise ValueError("procedural_scenes keys must be a scene_names subset")
        self.scene_names = list(scene_names)
        self.scene_name = self.scene_names[0]
        self.width = width
        self.height = height
        self.seed = seed
        self.allow_source_pose_drift = allow_source_pose_drift
        self.procedural_scenes: Mapping[str, AI2ThorProceduralScene] = (
            MappingProxyType(dict(procedural_by_alias))
        )
        self.controller: Any | None = None
        self._controller_factory = controller_factory or _default_controller_factory
        self._event: Any | None = None
        self._latest_event: Any | None = None
        self._stopped = False
        self._current_scene: Scene | None = None
        self._camera_states: dict[
            tuple[str, tuple[float, ...], tuple[float, ...]], dict[str, Any]
        ] = {}
        self._native_rotations: dict[tuple[str, str], dict[str, float]] = {}
        self._isolated_controller_refs: list[ReferenceType[Any]] = []

    def __enter__(self) -> AI2ThorAdapter:
        if self.controller is not None:
            raise RuntimeError("adapter context is already active")
        self._stopped = False
        self.scene_name = self.scene_names[0]
        self._latest_event = None
        self._camera_states.clear()
        self._native_rotations.clear()
        try:
            self.controller = self._start_controller(
                scene=self._native_scene_input(self.scene_name),
                width=self.width,
                height=self.height,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                gridSize=_NATIVE_NAVIGATION_GRID_SIZE_M,
                snapToGrid=True,
                rotateStepDegrees=90,
            )
            self._validate_controller_source_or_poison(
                self.controller, self.scene_name
            )
            event = self._step(self.controller, "Pass", action="Pass")
            self._event = event
            self._current_scene = None
            self._event = self._checked_scene_event(
                self.controller,
                event,
                "Pass",
                self.scene_name,
            )
            return self
        except BaseException as error:
            try:
                self._stop()
            except AI2ThorRuntimeError as cleanup_error:
                error.add_note(f"AI2-THOR cleanup also failed: {cleanup_error}")
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._stop()
        except AI2ThorRuntimeError as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"AI2-THOR cleanup also failed: {cleanup_error}")

    def _stop(self) -> None:
        if self.controller is not None and not self._stopped:
            self._stopped = True
            controller = self.controller
            self.controller = None
            self._event = None
            self._latest_event = None
            self._current_scene = None
            try:
                controller.stop()
            except OSError as error:
                raise AI2ThorRuntimeError(
                    f"AI2-THOR controller stop I/O failed: {error}"
                ) from error
            except MemoryError:
                raise
            except Exception as error:
                raise AI2ThorRuntimeError(
                    f"AI2-THOR controller stop failed: {error}"
                ) from error
        else:
            self.controller = None
            self._event = None
            self._latest_event = None
            self._current_scene = None

    def _start_controller(self, **kwargs: Any) -> Any:
        try:
            return self._controller_factory(**kwargs)
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR controller startup I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR controller startup failed: {error}"
            ) from error

    @staticmethod
    def _step(controller: Any, label: str, **kwargs: Any) -> Any:
        try:
            return controller.step(**kwargs)
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR {label} step I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR {label} step failed: {error}"
            ) from error

    def _native_scene_input(self, scene_id: str) -> str | dict[str, Any]:
        source = self.procedural_scenes.get(scene_id)
        return scene_id if source is None else source.decode_house()

    def _poison_scene_state(self) -> None:
        self._event = None
        self._current_scene = None

    def _validate_controller_source(
        self,
        controller: Any,
        scene_id: str,
    ) -> None:
        source = self.procedural_scenes.get(scene_id)
        if source is None:
            controller_scene = getattr(controller, "scene", None)
            if type(controller_scene) is dict:
                raise RuntimeError("legacy source cannot use a controller house dict")
            if controller_scene not in {scene_id, f"{scene_id}_physics"}:
                raise RuntimeError(
                    "legacy controller scene does not match the registered scene"
                )
            return
        try:
            canonical = _canonical_house_json_bytes(controller.scene)
        except (AttributeError, ValueError) as error:
            raise RuntimeError(
                "controller procedural source SHA-256 cannot be verified"
            ) from error
        if sha256(canonical).hexdigest() != source.house_sha256:
            raise RuntimeError("controller procedural source SHA-256 changed")

    def _validate_controller_source_or_poison(
        self,
        controller: Any,
        scene_id: str,
    ) -> None:
        try:
            self._validate_controller_source(controller, scene_id)
        except BaseException:
            self._poison_scene_state()
            raise

    def _validate_scene_source_or_poison(
        self,
        controller: Any,
        scene_id: str,
        event: Any,
    ) -> None:
        try:
            self._validate_controller_source(controller, scene_id)
            native_scene_name = self._native_scene_name(event)
            if scene_id in self.procedural_scenes:
                if native_scene_name != "Procedural":
                    raise RuntimeError(
                        "registered procedural source requires native Procedural scene"
                    )
            else:
                if native_scene_name == "Procedural":
                    raise RuntimeError(
                        "legacy source cannot use native Procedural scene"
                    )
                if native_scene_name != controller.scene:
                    raise RuntimeError(
                        "legacy controller and event native scene names differ"
                    )
        except BaseException:
            self._poison_scene_state()
            raise

    def _checked_scene_event(
        self,
        controller: Any,
        event: Any,
        action: str,
        scene_id: str,
    ) -> Any:
        # Keep the raw returned event even when action or identity validation
        # fails.  Candidate rejection classification may inspect it later, but
        # that path does not treat it as trusted without independently
        # revalidating the relevant fields.
        self._latest_event = event
        try:
            checked = self._checked_event(event, action)
            self._validate_scene_source_or_poison(
                controller,
                scene_id,
                checked,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        return checked

    def _reset(self, controller: Any, scene_id: str) -> Any:
        try:
            return controller.reset(scene=self._native_scene_input(scene_id))
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR reset {scene_id} I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR reset {scene_id} failed: {error}"
            ) from error

    def _require_active(self) -> Any:
        if self.controller is None:
            raise RuntimeError("adapter must be used as a context manager")
        return self.controller

    @staticmethod
    def _checked_event(event: Any, action: str) -> Any:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{action} returned an event without metadata")
        if metadata.get("lastActionSuccess") is not True:
            message = metadata.get("errorMessage") or f"{action} failed"
            raise RuntimeError(str(message))
        return event

    def list_scene_ids(self) -> list[str]:
        return list(self.scene_names)

    def runtime_identity(self) -> AI2ThorRuntimeIdentity:
        """Return the active package/build/configuration identity without action."""
        controller = self._require_active()
        if self._event is None:
            raise RuntimeError("no AI2-THOR event is available")
        self._validate_scene_source_or_poison(
            controller,
            self.scene_name,
            self._event,
        )
        build = getattr(controller, "_build", None)
        commit_id = getattr(build, "commit_id", None)
        if type(commit_id) is not str or not commit_id:
            raise RuntimeError("AI2-THOR controller has no Unity build identity")
        try:
            installed_version = package_version("ai2thor")
        except PackageNotFoundError as error:
            raise RuntimeError("AI2-THOR package identity is unavailable") from error
        source = self.procedural_scenes.get(self.scene_name)
        return AI2ThorRuntimeIdentity(
            ai2thor_version=installed_version,
            unity_commit_id=commit_id,
            native_scene_name=self._native_scene_name(self._event),
            width=self.width,
            height=self.height,
            seed=self.seed,
            source_dataset_id=None if source is None else source.dataset_id,
            source_revision=None if source is None else source.revision,
            source_split=None if source is None else source.split,
            source_index=None if source is None else source.index,
            source_sha256=None if source is None else source.house_sha256,
            source_scene_alias=None if source is None else self.scene_name,
            source_loader_id=None if source is None else source.source_loader_id,
            source_loader_version=(
                None if source is None else source.source_loader_version
            ),
            source_room_id=None if source is None else source.room_id,
            source_floor_xz_bounds=(
                None if source is None else source.floor_xz_bounds
            ),
            teleport_vertical_guard_m=_TELEPORT_VERTICAL_GUARD_M,
        )

    def latest_native_event(self, scene_id: str) -> Any:
        """Return latest raw event after source and native-scene revalidation."""
        controller = self._require_active()
        if scene_id not in self.scene_names:
            raise KeyError(scene_id)
        if self._latest_event is None:
            raise RuntimeError("adapter has no latest native event")
        self._validate_scene_source_or_poison(
            controller,
            scene_id,
            self._latest_event,
        )
        return self._latest_event

    def native_support_facts(
        self,
        scene: Scene,
    ) -> tuple[AI2ThorNativeSupportFact, ...]:
        """Read support lineage from the exact current event without an action."""

        event = self._current_event_for_scene(scene)
        raw_objects = event.metadata.get("objects")
        if type(raw_objects) is not list:
            raise AI2ThorNativeReturnError("invalid AI2-THOR object collection")
        return build_ai2thor_native_support_facts(scene, raw_objects)

    def current_agent_pose(self, scene: Scene) -> AI2ThorAgentPose:
        """Read the agent pose from the exact current event without an action."""

        return self._native_agent_pose(self._current_event_for_scene(scene))

    def _current_event_for_scene(self, scene: Scene) -> Any:
        controller = self._require_active()
        if (
            self._event is None
            or self.scene_name != scene.scene_id
            or self._current_scene != scene
        ):
            raise RuntimeError(
                "scene must be the adapter's exact current scene and event"
            )
        self._validate_scene_source_or_poison(
            controller,
            scene.scene_id,
            self._event,
        )
        return self._event

    @staticmethod
    def _is_analysis_overlay(
        current: Scene,
        requested: Scene,
    ) -> bool:
        if current == requested:
            return False
        return (
            requested.model_copy(
                update={
                    "room_polygon_xy": current.room_polygon_xy,
                    "collision_obstacles": current.collision_obstacles,
                    "subject_position_regions": current.subject_position_regions,
                }
            )
            == current
        )

    @staticmethod
    def _native_position(
        value: Any,
        label: str,
        *,
        error_type: type[Exception] = ValueError,
    ) -> AI2ThorNativePosition:
        if type(value) is not dict or set(value) != {"x", "y", "z"}:
            raise error_type(f"{label} must have exactly x/y/z keys")
        try:
            return AI2ThorNativePosition(
                x=_strict_finite_float(value["x"], f"{label} x"),
                y=_strict_finite_float(value["y"], f"{label} y"),
                z=_strict_finite_float(value["z"], f"{label} z"),
            )
        except ValueError as error:
            if error_type is ValueError:
                raise
            raise error_type(str(error)) from error

    @staticmethod
    def _native_scene_name(event: Any) -> str:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            raise TypeError("AI2-THOR event has no metadata")
        scene_name = metadata.get("sceneName")
        if type(scene_name) is not str or not scene_name:
            raise RuntimeError("AI2-THOR event has no valid native scene name")
        return scene_name

    @classmethod
    def _validate_native_scene_name(
        cls,
        event: Any,
        expected_native_scene_name: str,
    ) -> None:
        if cls._native_scene_name(event) != expected_native_scene_name:
            raise RuntimeError(
                "AI2-THOR event scene name changed during camera operation"
            )

    def _validate_native_scene_name_or_poison(
        self,
        event: Any,
        expected_native_scene_name: str,
    ) -> None:
        try:
            self._validate_native_scene_name(event, expected_native_scene_name)
        except BaseException:
            self._poison_scene_state()
            raise

    @classmethod
    def _native_agent_pose(cls, event: Any) -> AI2ThorAgentPose:
        metadata = event.metadata
        agent = metadata.get("agent")
        if type(agent) is not dict:
            raise ValueError("AI2-THOR event has no valid agent pose")
        position = cls._native_position(agent.get("position"), "agent position")
        rotation = agent.get("rotation")
        if type(rotation) is not dict or set(rotation) != {"x", "y", "z"}:
            raise ValueError("agent rotation must have exactly x/y/z keys")
        rotation_values = {
            axis: _strict_finite_float(rotation[axis], f"agent rotation {axis}")
            for axis in ("x", "y", "z")
        }
        if not cls._angles_close(rotation_values["x"], 0.0) or not cls._angles_close(
            rotation_values["z"], 0.0
        ):
            raise RuntimeError("camera pose returned non-zero pitch or roll")

        horizons: list[float] = []
        if "cameraHorizon" in metadata:
            horizons.append(
                _strict_finite_float(
                    metadata["cameraHorizon"],
                    "top-level camera horizon",
                )
            )
        if "cameraHorizon" in agent:
            horizons.append(
                _strict_finite_float(
                    agent["cameraHorizon"],
                    "agent camera horizon",
                )
            )
        if not horizons:
            raise ValueError("AI2-THOR event has no camera horizon")
        if any(not cls._angles_close(horizons[0], horizon) for horizon in horizons[1:]):
            raise RuntimeError("AI2-THOR camera horizon metadata is inconsistent")
        standing = agent.get("isStanding")
        if type(standing) is not bool:
            raise ValueError("AI2-THOR agent standing must be an exact boolean")
        return AI2ThorAgentPose(
            position=position,
            yaw_degrees=rotation_values["y"],
            horizon_degrees=horizons[0],
            standing=standing,
        )

    @classmethod
    def _native_scene_fully_settled(
        cls,
        event: Any,
        expected_categories_by_name: dict[str, str],
    ) -> bool:
        raw_objects = event.metadata.get("objects")
        if type(raw_objects) is not list:
            raise AI2ThorNativeReturnError(
                "scene settlement returned no object metadata"
            )
        raw_objects = _validated_native_object_metadata(raw_objects)
        domain_objects = _domain_object_metadata(raw_objects)
        categories_by_name: dict[str, str] = {}
        any_object_moving = False
        for item in domain_objects:
            if type(item) is not dict:
                raise AI2ThorNativeReturnError(
                    "scene settlement returned invalid domain object metadata"
                )
            name = item.get("name")
            category = item.get("objectType")
            if (
                type(name) is not str
                or not name
                or type(category) is not str
                or not category
            ):
                raise AI2ThorNativeReturnError(
                    "scene settlement returned invalid object name/category metadata"
                )
            if name in categories_by_name:
                raise AI2ThorNativeReturnError(
                    "scene settlement returned duplicate stable object names"
                )
            categories_by_name[name] = category
            is_moving = item.get("isMoving")
            if type(is_moving) is not bool:
                raise AI2ThorNativeReturnError(
                    f"AI2-THOR isMoving for object {name!r} must be an exact boolean"
                )
            any_object_moving = any_object_moving or is_moving
        if categories_by_name != expected_categories_by_name:
            raise AI2ThorNativeReturnError(
                "object name/category mapping changed during scene settlement"
            )
        return cls._native_scene_at_rest(event) and not any_object_moving

    @staticmethod
    def _angle_residual_degrees(left: float, right: float) -> float:
        return abs((left - right + 180.0) % 360.0 - 180.0)

    @classmethod
    def _validate_camera_object_invariants(
        cls,
        source: Scene,
        observed: Scene,
    ) -> None:
        cls._validate_camera_object_identity_invariants(source, observed)
        source_by_name = cls._objects_by_name(source.objects)
        observed_by_name = cls._objects_by_name(observed.objects)

        for name, original in source_by_name.items():
            current = observed_by_name[name]
            if (
                not np.allclose(
                    (
                        current.position.x,
                        current.position.y,
                        current.position.z,
                        current.obb.center.x,
                        current.obb.center.y,
                        current.obb.center.z,
                        current.obb.extent.x,
                        current.obb.extent.y,
                        current.obb.extent.z,
                    ),
                    (
                        original.position.x,
                        original.position.y,
                        original.position.z,
                        original.obb.center.x,
                        original.obb.center.y,
                        original.obb.center.z,
                        original.obb.extent.x,
                        original.obb.extent.y,
                        original.obb.extent.z,
                    ),
                    atol=_OBJECT_GEOMETRY_TOLERANCE_M,
                    rtol=0.0,
                )
                or not _quaternions_close(
                    current.rotation,
                    original.rotation,
                )
                or not _quaternions_close(
                    current.obb.rotation,
                    original.obb.rotation,
                )
            ):
                raise RuntimeError(
                    f"object {name!r} geometry changed during camera application"
                )

    @classmethod
    def _validate_camera_object_identity_invariants(
        cls,
        source: Scene,
        observed: Scene,
    ) -> None:
        source_by_name = cls._objects_by_name(source.objects)
        observed_by_name = cls._objects_by_name(observed.objects)
        if set(source_by_name) != set(observed_by_name):
            raise RuntimeError("stable object names changed during camera application")

        for name, original in source_by_name.items():
            current = observed_by_name[name]
            if (
                current.object_id != original.object_id
                or current.name != original.name
                or current.category != original.category
                or current.movable is not original.movable
                or current.request_eligible is not original.request_eligible
                or current.support_object_id != original.support_object_id
            ):
                raise RuntimeError(
                    f"object {name!r} identity, category, mobility, or support changed "
                    "during camera application"
                )

    def reachable_agent_positions(
        self,
        scene: Scene,
    ) -> tuple[AI2ThorNativePosition, ...]:
        """Return a deterministic, strictly validated native navigation grid."""
        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        expected_native_scene_name = self._native_scene_name(current_event)
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "GetReachablePositions",
                action="GetReachablePositions",
            )
            event = self._checked_scene_event(
                controller,
                event,
                "GetReachablePositions",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
            )
            action_return = event.metadata.get("actionReturn")
            if type(action_return) is not list:
                raise ValueError(
                    "GetReachablePositions actionReturn must be a list"
                )
            keyed_positions: list[
                tuple[tuple[int, int, int], AI2ThorNativePosition]
            ] = []
            seen_keys: set[tuple[int, int, int]] = set()
            for index, raw_position in enumerate(action_return):
                position = self._native_position(
                    raw_position,
                    f"reachable position {index}",
                )
                quantized_key = tuple(
                    round(value / _REACHABLE_POSITION_QUANTIZATION_M)
                    for value in (position.x, position.z, position.y)
                )
                if quantized_key in seen_keys:
                    raise ValueError(
                        "GetReachablePositions returned a duplicate quantized position"
                    )
                seen_keys.add(quantized_key)
                keyed_positions.append((quantized_key, position))
        except BaseException:
            self._poison_scene_state()
            raise
        keyed_positions.sort(key=lambda item: item[0])
        self._event = event
        self._current_scene = scene
        return tuple(position for _, position in keyed_positions)

    def receptacle_spawn_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
    ) -> AI2ThorReceptacleSpawnMap:
        """Capture deterministic native receptacle coordinates for one subject.

        Returned coordinates are source facts, not collision-free placements.
        The query is executed once against the exact current source event and
        cannot be used as a per-candidate platform search loop.
        """

        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        subject = scene.object_by_id(subject_object_id)
        support_object_id = subject.support_object_id
        if support_object_id is None:
            raise ValueError("receptacle spawn subject has no declared support")
        support = scene.object_by_id(support_object_id)
        runtime_identity = self.runtime_identity()
        expected_native_scene_name = self._native_scene_name(current_event)
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        native_subject_object_id = self._native_object_id_for_name(
            current_event,
            subject.name,
        )
        native_support_object_id = self._native_object_id_for_name(
            current_event,
            support.name,
        )
        try:
            event = self._step(
                controller,
                "GetSpawnCoordinatesAboveReceptacle",
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=native_support_object_id,
                anywhere=True,
            )
            event = self._checked_scene_event(
                controller,
                event,
                "GetSpawnCoordinatesAboveReceptacle",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
            )
            action_return = event.metadata.get("actionReturn")
            if type(action_return) is not list or not action_return:
                raise AI2ThorNativeReturnError(
                    "GetSpawnCoordinatesAboveReceptacle actionReturn must be "
                    "a non-empty list"
                )
            raw_positions: list[AI2ThorNativePosition] = []
            keyed_positions: list[
                tuple[tuple[int, int, int], AI2ThorNativePosition]
            ] = []
            seen_keys: set[tuple[int, int, int]] = set()
            for index, raw_position in enumerate(action_return):
                position = self._native_position(
                    raw_position,
                    f"receptacle spawn position {index}",
                    error_type=AI2ThorNativeReturnError,
                )
                raw_positions.append(position)
                quantized_key = tuple(
                    round(value / _REACHABLE_POSITION_QUANTIZATION_M)
                    for value in (position.x, position.z, position.y)
                )
                if quantized_key in seen_keys:
                    raise AI2ThorNativeReturnError(
                        "GetSpawnCoordinatesAboveReceptacle returned a duplicate "
                        "quantized position"
                    )
                seen_keys.add(quantized_key)
                keyed_positions.append((quantized_key, position))
        except BaseException:
            self._poison_scene_state()
            raise
        positions = tuple(
            sorted(
                (position for _, position in keyed_positions),
                key=lambda item: (item.x, item.z, item.y),
            )
        )
        surface_patches = build_ai2thor_receptacle_surface_patches(
            tuple(raw_positions)
        )
        positions_sha256 = _native_positions_sha256(positions)
        scene_sha256 = _receptacle_scene_sha256(scene, surface_patches)
        source_sha256 = _receptacle_spawn_source_sha256(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            support_object_id=support.object_id,
            native_subject_object_id=native_subject_object_id,
            native_support_object_id=native_support_object_id,
            runtime_identity=runtime_identity,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            surface_patches=surface_patches,
        )
        spawn_map = AI2ThorReceptacleSpawnMap(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            support_object_id=support.object_id,
            native_subject_object_id=native_subject_object_id,
            native_support_object_id=native_support_object_id,
            runtime_identity=runtime_identity,
            positions=positions,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            source_sha256=source_sha256,
            surface_patches=surface_patches,
        )
        self._event = event
        self._current_scene = scene
        return spawn_map

    def settle_scene_observed(
        self,
        scene: Scene,
        max_pass_steps: int = 30,
    ) -> AI2ThorSceneSettlement:
        """Advance physics until the scene and every domain object are still."""
        controller = self._require_active()
        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        event = self._current_event_for_scene(scene)
        expected_categories_by_name = {
            name: obj.category
            for name, obj in self._objects_by_name(scene.objects).items()
        }
        previous_camera_states = deepcopy(self._camera_states)
        previous_native_rotations = deepcopy(self._native_rotations)
        pass_steps = 0
        try:
            expected_native_scene_name = self._native_scene_name(event)
            while True:
                self._validate_native_scene_name_or_poison(
                    event,
                    expected_native_scene_name,
                )
                settled = self._native_scene_fully_settled(
                    event,
                    expected_categories_by_name,
                )
                if settled:
                    native_observed = self._scene_from_event(scene.scene_id, event)
                    observed_scene = self._stable_observed_scene(
                        scene,
                        native_observed,
                    )
                    observation = self._observation_from_event(
                        observed_scene,
                        event,
                    )
                    result = AI2ThorSceneSettlement(
                        observed_scene=observed_scene,
                        observation=observation,
                        pass_steps=pass_steps,
                    )
                    self._event = event
                    self._current_scene = observed_scene
                    return result
                if pass_steps >= max_pass_steps:
                    raise AI2ThorSettlementTimeout(
                        "AI2-THOR scene did not settle within "
                        f"{max_pass_steps} Pass steps"
                    )
                event = self._step(
                    controller,
                    "Pass",
                    action="Pass",
                )
                self._event = event
                self._current_scene = None
                pass_steps += 1
                event = self._checked_scene_event(
                    controller,
                    event,
                    "Pass",
                    scene.scene_id,
                )
        except BaseException:
            self._camera_states = previous_camera_states
            self._native_rotations = previous_native_rotations
            self._current_scene = None
            raise

    @contextmanager
    def isolated_scene_observed(
        self,
        source: Scene,
        max_pass_steps: int = 30,
    ) -> Iterator[AI2ThorIsolatedEpisode]:
        """Open one target-only controller and yield its settled source.

        The parent adapter remains the preparation owner and is never reset or
        mutated.  Native object IDs may differ in the child, so its observed
        baseline is rebound to the frozen source IDs and analysis overlays by
        unique object name.  Geometry, views, assets, and rest status all come
        from the child's same final trusted event.
        """
        parent_controller = self._require_active()
        if type(source) is not Scene:
            raise ValueError("source must be an exact canonical Scene")
        if source.scene_id not in self.scene_names:
            raise KeyError(source.scene_id)
        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        procedural = self.procedural_scenes.get(source.scene_id)

        def isolated_controller_factory(**kwargs: Any) -> Any:
            controller = self._controller_factory(**kwargs)
            if controller is parent_controller:
                raise RuntimeError(
                    "isolated AI2-THOR episode reused the parent controller"
                )
            live_refs: list[ReferenceType[Any]] = []
            for controller_ref in self._isolated_controller_refs:
                previous = controller_ref()
                if previous is None:
                    continue
                live_refs.append(controller_ref)
                if controller is previous:
                    raise RuntimeError(
                        "isolated AI2-THOR episode reused a prior child controller"
                    )
            try:
                live_refs.append(ref(controller))
            except TypeError as error:
                raise RuntimeError(
                    "isolated AI2-THOR controller must support weak references"
                ) from error
            self._isolated_controller_refs = live_refs
            return controller

        child = AI2ThorAdapter(
            [source.scene_id],
            width=self.width,
            height=self.height,
            seed=self.seed,
            controller_factory=isolated_controller_factory,
            allow_source_pose_drift=self.allow_source_pose_drift,
            procedural_scenes=(
                {source.scene_id: procedural}
                if procedural is not None
                else None
            ),
        )
        with child:
            event = child._activate_scene(source.scene_id)
            native_scene = child._scene_from_event(source.scene_id, event)
            child._current_scene = native_scene
            native_settlement = child.settle_scene_observed(
                native_scene,
                max_pass_steps=max_pass_steps,
            )
            final_event = child._current_event_for_scene(
                native_settlement.observed_scene
            )
            stable_scene = child._stable_observed_scene(
                source,
                native_settlement.observed_scene,
            )
            observation = child._observation_from_event(stable_scene, final_event)
            settlement = AI2ThorSceneSettlement(
                observed_scene=stable_scene,
                observation=observation,
                pass_steps=native_settlement.pass_steps,
            )
            child._current_scene = stable_scene
            yield AI2ThorIsolatedEpisode(
                adapter=child,
                baseline_settlement=settlement,
            )

    def apply_camera_pose_observed(
        self,
        scene: Scene,
        pose: AI2ThorAgentPose,
    ) -> AI2ThorCameraApplication:
        """Apply one unforced TeleportFull and bind its same-event observation."""
        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        expected_native_scene_name = self._native_scene_name(current_event)
        if type(pose) is not AI2ThorAgentPose:
            raise ValueError("camera pose must be an exact AI2ThorAgentPose")
        snapped_x = (
            round(pose.position.x / _NATIVE_NAVIGATION_GRID_SIZE_M)
            * _NATIVE_NAVIGATION_GRID_SIZE_M
        )
        snapped_z = (
            round(pose.position.z / _NATIVE_NAVIGATION_GRID_SIZE_M)
            * _NATIVE_NAVIGATION_GRID_SIZE_M
        )
        commanded_position = AI2ThorNativePosition(
            x=snapped_x,
            y=pose.position.y,
            z=snapped_z,
        )
        _, is_on_native_grid = _camera_position_residual_within_tolerance(
            pose.position,
            commanded_position,
        )
        if not is_on_native_grid:
            raise ValueError("camera pose is not on the native grid")
        if commanded_position.x == 0.0:
            commanded_position = AI2ThorNativePosition(
                x=0.0,
                y=commanded_position.y,
                z=commanded_position.z,
            )
        if commanded_position.z == 0.0:
            commanded_position = AI2ThorNativePosition(
                x=commanded_position.x,
                y=commanded_position.y,
                z=0.0,
            )

        previous_camera_states = deepcopy(self._camera_states)
        previous_native_rotations = deepcopy(self._native_rotations)
        try:
            event = self._step(
                controller,
                "TeleportFull",
                action="TeleportFull",
                position={
                    "x": commanded_position.x,
                    "y": commanded_position.y,
                    "z": commanded_position.z,
                },
                rotation={"x": 0.0, "y": pose.yaw_degrees, "z": 0.0},
                horizon=pose.horizon_degrees,
                standing=pose.standing,
            )
            # Unity may have changed even when validation below fails. Retain
            # the returned event but invalidate the canonical current scene
            # until every invariant and same-event artifact has been checked.
            self._event = event
            self._current_scene = None
            event = self._checked_scene_event(
                controller,
                event,
                "TeleportFull",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )

            observed_pose = self._native_agent_pose(event)
            observed_camera_position = self._native_position(
                event.metadata.get("cameraPosition"),
                "camera position",
            )
            position_residual_m, position_within_tolerance = (
                _camera_position_residual_within_tolerance(
                    pose.position,
                    observed_pose.position,
                )
            )
            yaw_residual_degrees = self._angle_residual_degrees(
                observed_pose.yaw_degrees,
                pose.yaw_degrees,
            )
            horizon_residual_degrees = abs(
                observed_pose.horizon_degrees - pose.horizon_degrees
            )
            if not position_within_tolerance:
                raise RuntimeError("camera position drift exceeds tolerance")
            if yaw_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
                raise RuntimeError("camera yaw drift exceeds tolerance")
            if horizon_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
                raise RuntimeError("camera horizon drift exceeds tolerance")
            if observed_pose.standing is not pose.standing:
                raise RuntimeError("camera standing state differs from request")

            native_observed = self._scene_from_event(scene.scene_id, event)
            observed_scene = self._canonical_camera_observed_scene(
                scene,
                native_observed,
            )
            observation = self._observation_from_event(observed_scene, event)
            result = AI2ThorCameraApplication(
                requested_pose=pose,
                observed_pose=observed_pose,
                observed_camera_position=observed_camera_position,
                observed_scene=observed_scene,
                observation=observation,
                position_residual_m=position_residual_m,
                yaw_residual_degrees=yaw_residual_degrees,
                horizon_residual_degrees=horizon_residual_degrees,
            )
            self._current_scene = observed_scene
            return result
        except BaseException:
            self._camera_states = previous_camera_states
            self._native_rotations = previous_native_rotations
            self._poison_scene_state()
            raise

    @contextmanager
    def paused_camera_observations(self, source: Scene) -> Iterator[Scene]:
        """Freeze native physics while yielding camera-only source observations."""

        with self._paused_camera_observations(
            source,
            retain_unpaused_scene=False,
        ) as paused_source:
            yield paused_source

    @contextmanager
    def paused_camera_observations_for_settlement(
        self,
        source: Scene,
    ) -> Iterator[Scene]:
        """Freeze camera ranking and retain the native scene returned by unpause."""

        with self._paused_camera_observations(
            source,
            retain_unpaused_scene=True,
        ) as paused_source:
            yield paused_source

    @contextmanager
    def _paused_camera_observations(
        self,
        source: Scene,
        *,
        retain_unpaused_scene: bool,
    ) -> Iterator[Scene]:
        if type(retain_unpaused_scene) is not bool:
            raise TypeError("retain-unpaused-scene flag must be an exact boolean")
        controller = self._require_active()
        current_event = self._current_event_for_scene(source)
        expected_native_scene_name = self._native_scene_name(current_event)
        try:
            paused_event = self._step(
                controller,
                "PausePhysicsAutoSim",
                action="PausePhysicsAutoSim",
            )
            self._event = paused_event
            self._current_scene = None
            paused_event = self._checked_scene_event(
                controller,
                paused_event,
                "PausePhysicsAutoSim",
                source.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                paused_event,
                expected_native_scene_name,
            )
            native_paused = self._scene_from_event(source.scene_id, paused_event)
            paused_source = self._canonical_camera_observed_scene(
                source,
                native_paused,
            )
            self._current_scene = paused_source
        except BaseException:
            self._poison_scene_state()
            raise

        try:
            yield paused_source
        except BaseException as error:
            try:
                self._unpause_camera_observations(
                    controller,
                    source.scene_id,
                    expected_native_scene_name,
                    retained_source=source if retain_unpaused_scene else None,
                )
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"AI2-THOR physics unpause also failed: {cleanup_error}")
            raise
        else:
            self._unpause_camera_observations(
                controller,
                source.scene_id,
                expected_native_scene_name,
                retained_source=source if retain_unpaused_scene else None,
            )

    def _unpause_camera_observations(
        self,
        controller: Any,
        scene_id: str,
        expected_native_scene_name: str,
        *,
        retained_source: Scene | None = None,
    ) -> None:
        try:
            event = self._step(
                controller,
                "UnpausePhysicsAutoSim",
                action="UnpausePhysicsAutoSim",
            )
            self._event = event
            self._current_scene = None
            event = self._checked_scene_event(
                controller,
                event,
                "UnpausePhysicsAutoSim",
                scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            if retained_source is not None:
                native_unpaused = self._scene_from_event(scene_id, event)
                stable_unpaused = self._stable_observed_scene(
                    retained_source,
                    native_unpaused,
                )
                self._validate_camera_object_identity_invariants(
                    retained_source,
                    stable_unpaused,
                )
                self._current_scene = stable_unpaused
            self._event = event
        except BaseException:
            self._poison_scene_state()
            raise

    def settle_current_camera_pose_observed(
        self,
        source: Scene,
        pose: AI2ThorAgentPose,
        *,
        max_pass_steps: int,
    ) -> AI2ThorSettledCameraApplication:
        """Settle the post-unpause scene and bind its current camera event."""

        if type(source) is not Scene:
            raise TypeError("camera settlement source must be an exact Scene")
        if type(pose) is not AI2ThorAgentPose:
            raise TypeError("camera settlement pose must be exact")
        current_scene = self._current_scene
        if current_scene is None or current_scene.scene_id != source.scene_id:
            raise RuntimeError("adapter has no current post-unpause camera scene")
        settlement = self.settle_scene_observed(
            current_scene,
            max_pass_steps=max_pass_steps,
        )
        event = self._current_event_for_scene(settlement.observed_scene)
        observed_pose = self._native_agent_pose(event)
        observed_camera_position = self._native_position(
            event.metadata.get("cameraPosition"),
            "camera position",
        )
        position_residual_m, position_within_tolerance = (
            _camera_position_residual_within_tolerance(
                pose.position,
                observed_pose.position,
            )
        )
        yaw_residual_degrees = self._angle_residual_degrees(
            observed_pose.yaw_degrees,
            pose.yaw_degrees,
        )
        horizon_residual_degrees = abs(
            observed_pose.horizon_degrees - pose.horizon_degrees
        )
        if not position_within_tolerance:
            raise RuntimeError("settled camera position drift exceeds tolerance")
        if yaw_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
            raise RuntimeError("settled camera yaw drift exceeds tolerance")
        if horizon_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
            raise RuntimeError("settled camera horizon drift exceeds tolerance")
        if observed_pose.standing is not pose.standing:
            raise RuntimeError("settled camera standing state differs from request")
        return AI2ThorSettledCameraApplication(
            application=AI2ThorCameraApplication(
                requested_pose=pose,
                observed_pose=observed_pose,
                observed_camera_position=observed_camera_position,
                observed_scene=settlement.observed_scene,
                observation=settlement.observation,
                position_residual_m=position_residual_m,
                yaw_residual_degrees=yaw_residual_degrees,
                horizon_residual_degrees=horizon_residual_degrees,
            ),
            settlement_pass_steps=settlement.pass_steps,
        )

    def apply_camera_pose_from_frozen_source_observed(
        self,
        source: Scene,
        pose: AI2ThorAgentPose,
        *,
        max_pass_steps: int,
    ) -> AI2ThorCameraApplication:
        """Reset and settle the same frozen source before one camera pose."""

        if type(source) is not Scene:
            raise TypeError("frozen camera source must be an exact Scene")
        loaded = self.load_scene(source.scene_id)
        settlement = self.settle_scene_observed(
            loaded,
            max_pass_steps=max_pass_steps,
        )
        stable_source = self._canonical_camera_observed_scene(
            source,
            settlement.observed_scene,
        )
        self._current_scene = stable_source
        return self.apply_camera_pose_observed(stable_source, pose)

    def conservative_floor_envelope(
        self,
        scene: Scene,
        clearance_m: float,
    ) -> AI2ThorFloorEnvelope:
        """Derive an inward-offset rectangle from the current native floor AABB."""
        event = self._current_event_for_scene(scene)
        if (
            isinstance(clearance_m, bool)
            or not isinstance(clearance_m, (int, float))
            or not math.isfinite(float(clearance_m))
            or clearance_m < 0.0
        ):
            raise ValueError("floor clearance must be finite and non-negative")
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError("current event has no structural Floor collection")
        floors: list[dict[str, Any]] = []
        for item in raw_objects:
            if not isinstance(item, dict) or item.get("objectType") != "Floor":
                continue
            bounds = item.get("axisAlignedBoundingBox")
            size = bounds.get("size") if isinstance(bounds, dict) else None
            if not isinstance(size, dict):
                raise ValueError("structural Floor must have a finite positive AABB")
            try:
                native_size_x = float(size["x"])
                native_size_y = float(size["y"])
                native_size_z = float(size["z"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "structural Floor must have a finite positive AABB"
                ) from exc
            if (
                not all(
                    math.isfinite(value)
                    for value in (native_size_x, native_size_y, native_size_z)
                )
                or native_size_x <= 0.0
                or native_size_z <= 0.0
                or native_size_y < 0.0
            ):
                raise ValueError("structural Floor must have a finite positive AABB")
            if native_size_y == 0.0:
                continue
            floors.append(item)
        if len(floors) != 1:
            raise ValueError("current event must contain exactly one structural Floor")
        floor = floors[0]
        bounds = floor.get("axisAlignedBoundingBox")
        if not isinstance(bounds, dict):
            raise ValueError("structural Floor must have a finite positive AABB")
        try:
            center = ai2thor_position_to_world(Vec3(**bounds["center"]))
            size = bounds["size"]
            extent = Vec3(
                x=float(size["x"]),
                y=float(size["z"]),
                z=float(size["y"]),
            )
            floor_object_id = str(floor["objectId"])
            floor_name = str(floor["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "structural Floor must have a finite positive AABB"
            ) from exc
        geometry_values = (
            center.x,
            center.y,
            center.z,
            extent.x,
            extent.y,
            extent.z,
        )
        if (
            not all(math.isfinite(value) for value in geometry_values)
            or min(extent.x, extent.y, extent.z) <= 0.0
            or not floor_object_id
            or not floor_name
        ):
            raise ValueError("structural Floor must have a finite positive AABB")
        source = self.procedural_scenes.get(scene.scene_id)
        effective_native_bounds: tuple[float, float, float, float] | None = None
        if source is not None:
            native_center = bounds.get("center")
            if not isinstance(native_center, dict):
                raise ValueError("structural Floor must have a finite positive AABB")
            try:
                native_center_x = float(native_center["x"])
                native_center_z = float(native_center["z"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "structural Floor must have a finite positive AABB"
                ) from error
            floor_aabb_bounds = (
                native_center_x - extent.x / 2.0,
                native_center_z - extent.y / 2.0,
                native_center_x + extent.x / 2.0,
                native_center_z + extent.y / 2.0,
            )
            if not all(
                math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-5)
                for expected, actual in zip(
                    source.floor_xz_bounds,
                    floor_aabb_bounds,
                    strict=True,
                )
            ):
                raise ValueError(
                    "procedural floorPolygon does not match structural Floor AABB"
                )
            effective_native_bounds = (
                max(source.floor_xz_bounds[0], floor_aabb_bounds[0]),
                max(source.floor_xz_bounds[1], floor_aabb_bounds[1]),
                min(source.floor_xz_bounds[2], floor_aabb_bounds[2]),
                min(source.floor_xz_bounds[3], floor_aabb_bounds[3]),
            )
        if effective_native_bounds is None:
            effective_native_bounds = (
                center.x - extent.x / 2.0,
                center.y - extent.y / 2.0,
                center.x + extent.x / 2.0,
                center.y + extent.y / 2.0,
            )
        minimum_x = effective_native_bounds[0] + float(clearance_m)
        minimum_y = effective_native_bounds[1] + float(clearance_m)
        maximum_x = effective_native_bounds[2] - float(clearance_m)
        maximum_y = effective_native_bounds[3] - float(clearance_m)
        if maximum_x <= minimum_x or maximum_y <= minimum_y:
            raise ValueError("floor clearance leaves no positive envelope")
        native_aabb = OBB(
            center=center,
            extent=extent,
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return AI2ThorFloorEnvelope(
            scene_id=scene.scene_id,
            floor_object_id=floor_object_id,
            floor_name=floor_name,
            native_aabb=native_aabb,
            floor_top_z=center.z + extent.z / 2.0,
            clearance_m=float(clearance_m),
            polygon_xy=(
                Vec2(x=minimum_x, y=minimum_y),
                Vec2(x=maximum_x, y=minimum_y),
                Vec2(x=maximum_x, y=maximum_y),
                Vec2(x=minimum_x, y=maximum_y),
            ),
        )

    def conservative_collision_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
        clearance_m: float,
    ) -> AI2ThorNativeFeasibilityMap:
        """Expand stationary native OBBs into view-independent obstacles."""
        self._current_event_for_scene(scene)
        if (
            isinstance(clearance_m, bool)
            or not isinstance(clearance_m, (int, float))
            or not math.isfinite(float(clearance_m))
            or float(clearance_m) <= 0.0
        ):
            raise ValueError("collision clearance must be finite and positive")
        subject = scene.object_by_id(subject_object_id)
        excluded_ids = {subject.object_id, subject.support_object_id}
        clearance = float(clearance_m)
        obstacles = tuple(
            CollisionObstacle(
                obstacle_id=f"native-clearance:{obj.object_id}",
                source_object_id=obj.object_id,
                clearance_m=clearance,
                obb=obj.obb,
            )
            for obj in sorted(scene.objects, key=lambda item: item.object_id)
            if obj.object_id not in excluded_ids
        )
        return AI2ThorNativeFeasibilityMap(
            scene_id=scene.scene_id,
            subject_object_id=subject.object_id,
            clearance_m=clearance,
            obstacles=obstacles,
        )

    def conservative_navigation_map(
        self,
        scene: Scene,
        *,
        subject_object_id: str,
        room_polygon_xy: tuple[Vec2, ...],
        agent_radius_m: float,
        clearance_m: float,
    ) -> AI2ThorNavigationFeasibilityMap:
        """Bind the current reachable grid to a conservative subject locus."""
        self._current_event_for_scene(scene)
        positions = self.reachable_agent_positions(scene)
        return build_navigation_feasibility_map(
            scene,
            subject_object_id=subject_object_id,
            room_polygon_xy=room_polygon_xy,
            reachable_positions=positions,
            agent_radius_m=agent_radius_m,
            clearance_m=clearance_m,
        )

    @staticmethod
    def _camera_key(
        scene_id: str,
        camera: Camera,
    ) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        return scene_id, camera.intrinsics, camera.world_to_camera

    def _camera(self, metadata: dict[str, Any], scene_id: str) -> Camera:
        try:
            fov = math.radians(float(metadata["fov"]))
            position = ai2thor_position_to_world(
                Vec3(**metadata["cameraPosition"])
            )
            agent_metadata = metadata["agent"]
            yaw_degrees = float(agent_metadata["rotation"]["y"])
            if "cameraHorizon" in metadata:
                horizon = metadata["cameraHorizon"]
            else:
                horizon = agent_metadata["cameraHorizon"]
            horizon_degrees = float(horizon)
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                "invalid AI2-THOR camera metadata"
            ) from exc
        if not (0.0 < fov < math.pi):
            raise AI2ThorNativeReturnError(
                "camera field of view must be between 0 and 180 degrees"
            )
        focal = self.height / (2.0 * math.tan(fov / 2.0))
        intrinsics = (
            focal,
            0.0,
            self.width / 2.0,
            0.0,
            focal,
            self.height / 2.0,
            0.0,
            0.0,
            1.0,
        )
        camera = Camera(
            camera_id="main",
            width=self.width,
            height=self.height,
            intrinsics=tuple(float(value) for value in intrinsics),
            world_to_camera=ai2thor_camera_world_to_camera(
                position,
                yaw_degrees=yaw_degrees,
                horizon_degrees=horizon_degrees,
            ),
        )
        agent = metadata.get("agent")
        if not isinstance(agent, dict):
            raise AI2ThorNativeReturnError("invalid AI2-THOR agent metadata")
        try:
            state = {
                "position": {
                    axis: float(agent["position"][axis]) for axis in ("x", "y", "z")
                },
                "rotation": {
                    axis: float(agent["rotation"][axis]) for axis in ("x", "y", "z")
                },
                "horizon": float(horizon),
                "standing": bool(agent.get("isStanding", True)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                "invalid AI2-THOR agent pose metadata"
            ) from exc
        self._camera_states[self._camera_key(scene_id, camera)] = state
        return camera

    @staticmethod
    def _oriented_bounds(
        metadata: dict[str, Any],
        position: Vec3,
        rotation: Quaternion,
    ) -> OBB:
        oriented = metadata.get("objectOrientedBoundingBox")
        corners = oriented.get("cornerPoints") if isinstance(oriented, dict) else None
        if corners is not None:
            try:
                array = np.asarray(
                    [
                        (
                            float(point["x"]),
                            float(point["y"]),
                            float(point["z"]),
                        )
                        if isinstance(point, dict)
                        else tuple(float(value) for value in point)
                        for point in corners
                    ],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AI2ThorNativeReturnError(
                    "invalid oriented bounding-box corners"
                ) from exc
            if array.shape != (8, 3) or not np.isfinite(array).all():
                raise AI2ThorNativeReturnError(
                    "oriented bounding box must have eight finite corners"
                )
            world = array[:, [0, 2, 1]]
            center_array = world.mean(axis=0)
            yaw = _quaternion_yaw(rotation)
            planar_rotation = Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(yaw / 2.0),
                w=math.cos(yaw / 2.0),
            )
            # Geometry consumers use an upright Z-up OBB. For tilted objects,
            # bound the projected corners in the object's yaw frame so the
            # ground footprint is conservative instead of under-estimated.
            local = (world - center_array) @ _rotation_matrix(planar_rotation)
            extent_array = np.ptp(local, axis=0)
            if np.any(extent_array <= _EPSILON):
                raise AI2ThorNativeReturnError(
                    "oriented bounding-box extents must be positive"
                )
            return OBB(
                center=Vec3(
                    x=float(center_array[0]),
                    y=float(center_array[1]),
                    z=float(center_array[2]),
                ),
                extent=Vec3(
                    x=float(extent_array[0]),
                    y=float(extent_array[1]),
                    z=float(extent_array[2]),
                ),
                rotation=planar_rotation,
            )

        bounds = metadata.get("axisAlignedBoundingBox")
        if not isinstance(bounds, dict):
            raise AI2ThorNativeReturnError(
                "object is missing bounding-box metadata"
            )
        try:
            center = ai2thor_position_to_world(Vec3(**bounds["center"]))
            size = bounds["size"]
            extent = Vec3(
                x=float(size["x"]),
                y=float(size["z"]),
                z=float(size["y"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                "invalid axis-aligned bounding-box metadata"
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (
                center.x,
                center.y,
                center.z,
                extent.x,
                extent.y,
                extent.z,
            )
        ) or min(extent.x, extent.y, extent.z) <= _EPSILON:
            raise AI2ThorNativeReturnError(
                "axis-aligned bounding-box values must be finite and positive"
            )
        return OBB(
            center=center,
            extent=extent,
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    @staticmethod
    def _obb_corners(obb: OBB) -> np.ndarray:
        offsets = np.asarray(
            [
                [dx * obb.extent.x / 2, dy * obb.extent.y / 2, dz * obb.extent.z / 2]
                for dx in (-1.0, 1.0)
                for dy in (-1.0, 1.0)
                for dz in (-1.0, 1.0)
            ],
            dtype=float,
        )
        center = np.asarray([obb.center.x, obb.center.y, obb.center.z])
        return offsets @ _rotation_matrix(obb.rotation).T + center

    def _projected_bounds(
        self,
        obb: OBB,
        camera: Camera,
    ) -> tuple[float, float, float, float]:
        projected: list[tuple[float, float]] = []
        extrinsics = matrix4(camera.world_to_camera)
        fx, fy = camera.intrinsics[0], camera.intrinsics[4]
        cx, cy = camera.intrinsics[2], camera.intrinsics[5]
        for point in self._obb_corners(obb):
            camera_point = transform_point(
                extrinsics,
                Vec3(x=float(point[0]), y=float(point[1]), z=float(point[2])),
            )
            if camera_point.z <= _EPSILON:
                continue
            projected.append(
                (
                    fx * camera_point.x / camera_point.z + cx,
                    cy - fy * camera_point.y / camera_point.z,
                )
            )
        if not projected:
            raise AI2ThorNativeReturnError(
                "visible object bounds are behind the camera"
            )
        xs, ys = zip(*projected)
        return min(xs), min(ys), max(xs), max(ys)

    def _view(
        self,
        metadata: dict[str, Any],
        position: Vec3,
        obb: OBB,
        camera: Camera,
        event: Any,
    ) -> dict[str, ObjectView]:
        object_id = metadata["objectId"]
        detections = getattr(event, "instance_detections2D", {}) or {}
        masks = getattr(event, "instance_masks", {}) or {}
        detection = detections.get(object_id)
        mask = masks.get(object_id)
        if detection is None or mask is None:
            return {}
        values = np.asarray(detection, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise AI2ThorNativeReturnError(f"invalid detection for {object_id!r}")
        xmin, ymin, xmax, ymax = (float(value) for value in values)
        if xmax <= xmin or ymax <= ymin:
            if metadata.get("visible") is False:
                return {}
            if (
                type(mask) is not np.ndarray
                or mask.shape != (self.height, self.width)
                or mask.dtype != np.bool_
            ):
                raise AI2ThorNativeReturnError(
                    f"invalid instance mask for {object_id!r}"
                )
            mask_array = mask
            pixels = np.argwhere(mask_array)
            if pixels.size == 0:
                raise AI2ThorNativeReturnError(
                    f"invalid detection bounds for {object_id!r}"
                )
            ymin = float(pixels[:, 0].min())
            ymax = float(pixels[:, 0].max() + 1)
            xmin = float(pixels[:, 1].min())
            xmax = float(pixels[:, 1].max() + 1)
        else:
            mask_array = np.asarray(mask)
            if mask_array.shape != (self.height, self.width):
                raise AI2ThorNativeReturnError(
                    f"invalid instance mask shape for {object_id!r}"
                )
        camera_point = transform_point(matrix4(camera.world_to_camera), position)
        if not math.isfinite(camera_point.z):
            raise AI2ThorNativeReturnError(
                f"visible object {object_id!r} has non-positive camera depth"
            )
        if camera_point.z <= _EPSILON:
            # AI2-THOR object anchors are not guaranteed to be OBB centres.
            # Large fixtures can therefore cross the camera plane while their
            # valid anchor is behind it.  They remain scene geometry, but do
            # not expose a relation view whose anchor depth is non-positive.
            # A box wholly on either side while its anchor disagrees is still
            # inconsistent native metadata and fails closed.
            extrinsics = matrix4(camera.world_to_camera)
            corner_depths = tuple(
                transform_point(
                    extrinsics,
                    Vec3(x=float(corner[0]), y=float(corner[1]), z=float(corner[2])),
                ).z
                for corner in self._obb_corners(obb)
            )
            if min(corner_depths) < 0.0 < max(corner_depths):
                return {}
            raise AI2ThorNativeReturnError(
                f"visible object {object_id!r} has non-positive camera depth"
            )

        clipped = (
            max(0.0, min(float(self.width), xmin)),
            max(0.0, min(float(self.height), ymin)),
            max(0.0, min(float(self.width), xmax)),
            max(0.0, min(float(self.height), ymax)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise AI2ThorNativeReturnError(
                f"visible detection for {object_id!r} is outside the image"
            )
        projected = self._projected_bounds(obb, camera)
        projected_area = max(
            _EPSILON,
            (projected[2] - projected[0]) * (projected[3] - projected[1]),
        )
        projected_clipped_width = max(
            0.0, min(projected[2], self.width) - max(projected[0], 0.0)
        )
        projected_clipped_height = max(
            0.0, min(projected[3], self.height) - max(projected[1], 0.0)
        )
        bbox_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        image_area = float(self.width * self.height)
        return {
            "main": ObjectView(
                camera_id="main",
                bbox=BBox2D(
                    xmin=clipped[0],
                    ymin=clipped[1],
                    xmax=clipped[2],
                    ymax=clipped[3],
                ),
                camera_depth=float(camera_point.z),
                visible_fraction=float(
                    np.clip(np.count_nonzero(mask_array) / projected_area, 0.0, 1.0)
                ),
                image_area_fraction=float(np.clip(bbox_area / image_area, 0.0, 1.0)),
                truncated_fraction=float(
                    np.clip(
                        1.0
                        - projected_clipped_width
                        * projected_clipped_height
                        / projected_area,
                        0.0,
                        1.0,
                    )
                ),
            )
        }

    def _object(
        self,
        metadata: dict[str, Any],
        camera: Camera,
        event: Any,
        structural_object_ids: frozenset[str],
    ) -> SceneObject:
        try:
            position = ai2thor_position_to_world(Vec3(**metadata["position"]))
            rotation = ai2thor_rotation_to_world(Vec3(**metadata["rotation"]))
            object_id = metadata["objectId"]
            name = metadata["name"]
            category = metadata["objectType"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                "invalid AI2-THOR object metadata"
            ) from exc
        obb = self._oriented_bounds(metadata, position, rotation)
        parents = metadata["parentReceptacles"]
        parents = [parent for parent in parents if parent not in structural_object_ids]
        return SceneObject(
            object_id=object_id,
            name=name,
            category=category,
            movable=(metadata.get("moveable") is True or metadata.get("pickupable") is True),
            position=position,
            rotation=rotation,
            obb=obb,
            support_object_id=parents[0] if parents else None,
            views=self._view(metadata, position, obb, camera, event),
        )

    @staticmethod
    def _without_cyclic_support_assignments(
        objects: tuple[SceneObject, ...],
    ) -> tuple[SceneObject, ...]:
        """Drop every edge in a cyclic native receptacle component.

        AI2-THOR's ``parentReceptacles`` describes receptacle membership, not
        a certified physical support tree, and real scenes can report cycles.
        A cycle has no honest single supporting parent, so retain the objects
        but represent those assignments as unknown instead of guessing an
        edge or passing an invalid graph to the solver.
        """
        object_ids = {obj.object_id for obj in objects}
        parents = {
            obj.object_id: (
                obj.support_object_id
                if obj.support_object_id in object_ids
                else None
            )
            for obj in objects
        }
        cyclic_ids: set[str] = set()
        for start in sorted(parents):
            path: list[str] = []
            path_index: dict[str, int] = {}
            current: str | None = start
            while current is not None and current in parents:
                if current in path_index:
                    cyclic_ids.update(path[path_index[current] :])
                    break
                path_index[current] = len(path)
                path.append(current)
                current = parents[current]
        if not cyclic_ids:
            return objects
        return tuple(
            obj.model_copy(update={"support_object_id": None})
            if obj.object_id in cyclic_ids
            else obj
            for obj in objects
        )

    def _scene_from_event(self, scene_id: str, event: Any) -> Scene:
        metadata = event.metadata
        camera = self._camera(metadata, scene_id)
        raw_objects = metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError("invalid AI2-THOR object collection")
        raw_objects = _validated_native_object_metadata(raw_objects)
        structural_object_ids = frozenset(
            item["objectId"]
            for item in raw_objects
            if item["objectType"] in _STRUCTURAL_OBJECT_TYPES
        )
        raw_objects = _domain_object_metadata(raw_objects)
        names = [item["name"] for item in raw_objects]
        for item, name in zip(raw_objects, names, strict=True):
            try:
                rotation = {
                    axis: float(item["rotation"][axis])
                    for axis in ("x", "y", "z")
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise AI2ThorNativeReturnError(
                    f"invalid native rotation for {name!r}"
                ) from exc
            if not all(math.isfinite(value) for value in rotation.values()):
                raise AI2ThorNativeReturnError(
                    f"invalid native rotation for {name!r}"
                )
            self._native_rotations[(scene_id, name)] = rotation
        objects = tuple(
            self._object(
                item,
                camera,
                event,
                structural_object_ids,
            )
            for item in raw_objects
        )
        objects = self._without_cyclic_support_assignments(objects)
        scene_bounds = metadata.get("sceneBounds")
        if not isinstance(scene_bounds, dict):
            raise AI2ThorNativeReturnError("invalid AI2-THOR scene bounds")
        try:
            center = ai2thor_position_to_world(Vec3(**scene_bounds["center"]))
            size = scene_bounds["size"]
            half_x = float(size["x"]) / 2.0
            half_y = float(size["z"]) / 2.0
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError("invalid AI2-THOR scene bounds") from exc
        if not all(
            math.isfinite(value)
            for value in (center.x, center.y, center.z, half_x, half_y)
        ) or half_x <= 0.0 or half_y <= 0.0:
            raise AI2ThorNativeReturnError(
                "scene ground bounds must be finite and positive"
            )
        return Scene(
            scene_id=scene_id,
            source="ai2thor",
            room_polygon_xy=(
                Vec2(x=center.x - half_x, y=center.y - half_y),
                Vec2(x=center.x + half_x, y=center.y - half_y),
                Vec2(x=center.x + half_x, y=center.y + half_y),
                Vec2(x=center.x - half_x, y=center.y + half_y),
            ),
            cameras=(camera,),
            objects=objects,
            generation_seed=self.seed,
        )

    def _activate_scene(
        self,
        scene_id: str,
        *,
        force_reset: bool = False,
    ) -> Any:
        controller = self._require_active()
        if scene_id not in self.scene_names:
            raise KeyError(scene_id)
        if self._event is None and not force_reset:
            raise RuntimeError("no AI2-THOR event is available")
        if (
            self._event is not None
            and self.scene_name == scene_id
            and not force_reset
        ):
            self._validate_scene_source_or_poison(
                controller,
                scene_id,
                self._event,
            )
            return self._event
        try:
            event = self._reset(controller, scene_id)
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            f"reset {scene_id}",
            scene_id,
        )
        # Commit native-scene tracking only after reset success. Callers can
        # then fail closed without applying saved poses to the wrong scene.
        self.scene_name = scene_id
        return event

    def load_scene(self, scene_id: str) -> Scene:
        # Public load means a deterministic baseline reload, even when Unity
        # already has this scene active. Internal restoration uses the
        # non-forced activation path to avoid redundant resets.
        event = self._activate_scene(scene_id, force_reset=True)
        scene = self._scene_from_event(scene_id, event)
        self._current_scene = scene
        return scene

    @staticmethod
    def _objects_by_name(
        objects: tuple[SceneObject, ...],
    ) -> dict[str, SceneObject]:
        by_name = {obj.name: obj for obj in objects}
        if len(by_name) != len(objects):
            raise ValueError("scene object names must be unique")
        return by_name

    @classmethod
    def _stable_observed_scene(
        cls,
        source: Scene,
        observed: Scene,
    ) -> Scene:
        """Map native ID churn back to source IDs through unique object names."""
        source_by_name = cls._objects_by_name(source.objects)
        observed_by_name = cls._objects_by_name(observed.objects)
        if set(source_by_name) != set(observed_by_name):
            raise AI2ThorNativeReturnError(
                "stable object names changed during pose application"
            )

        aliases: dict[str, str] = {}

        def register_alias(native_id: str, stable_id: str) -> None:
            existing = aliases.get(native_id)
            if existing is not None and existing != stable_id:
                raise AI2ThorNativeReturnError(
                    "native object IDs do not map to unique stable IDs"
                )
            aliases[native_id] = stable_id

        for name, original in source_by_name.items():
            current = observed_by_name[name]
            register_alias(current.object_id, original.object_id)
        source_object_ids = {obj.object_id for obj in source.objects}
        for original in source.objects:
            support_id = original.support_object_id
            if support_id is not None and support_id not in source_object_ids:
                register_alias(support_id, support_id)

        stable_objects: list[SceneObject] = []
        for original in source.objects:
            current = observed_by_name[original.name]
            support_id = current.support_object_id
            stable_support_id = None
            if support_id is not None:
                stable_support_id = aliases.get(support_id)
                if stable_support_id is None:
                    raise AI2ThorNativeReturnError(
                        f"observed support {support_id!r} has no stable object identity"
                    )
            stable_updates: dict[str, object] = {
                "object_id": original.object_id,
                "support_object_id": stable_support_id,
            }
            if "request_eligible" in original.model_fields_set:
                stable_updates["request_eligible"] = original.request_eligible
            stable_objects.append(current.model_copy(update=stable_updates))
        return source.model_copy(
            update={
                "cameras": observed.cameras,
                "objects": tuple(stable_objects),
            }
        )

    @classmethod
    def _canonical_camera_observed_scene(
        cls,
        source: Scene,
        native_observed: Scene,
    ) -> Scene:
        """Retain fresh camera/views after proving source geometry unchanged."""
        stable = cls._stable_observed_scene(source, native_observed)
        cls._validate_camera_object_invariants(source, stable)
        stable_by_name = cls._objects_by_name(stable.objects)
        return source.model_copy(
            update={
                "cameras": stable.cameras,
                "objects": tuple(
                    original.model_copy(
                        update={"views": stable_by_name[original.name].views}
                    )
                    for original in source.objects
                ),
            }
        )

    @classmethod
    def _commanded_scene(
        cls,
        source: Scene,
        observed: Scene,
        subject_id: str,
        commanded_position: Vec3,
    ) -> Scene:
        """Reproduce the legacy canonical command while retaining fresh views."""
        observed_by_name = cls._objects_by_name(observed.objects)
        subject = source.object_by_id(subject_id)
        delta_x = commanded_position.x - subject.position.x
        delta_y = commanded_position.y - subject.position.y
        merged: list[SceneObject] = []
        for original in source.objects:
            current = observed_by_name[original.name]
            if original.object_id == subject_id:
                merged.append(
                    original.model_copy(
                        update={
                            "position": commanded_position,
                            "obb": original.obb.model_copy(
                                update={
                                    "center": Vec3(
                                        x=original.obb.center.x + delta_x,
                                        y=original.obb.center.y + delta_y,
                                        z=original.obb.center.z,
                                    )
                                }
                            ),
                            "views": current.views,
                        }
                    )
                )
            else:
                merged.append(original.model_copy(update={"views": current.views}))
        return source.model_copy(update={"objects": tuple(merged)})

    def _native_rotation_for(
        self,
        scene_id: str,
        obj: SceneObject,
    ) -> dict[str, float]:
        rotation = self._native_rotations.get((scene_id, obj.name))
        if rotation is None:
            raise ValueError(
                f"native rotation was not captured for object {obj.name!r}"
            )
        if not _quaternions_close(
            ai2thor_rotation_to_world(Vec3(**rotation)),
            obj.rotation,
        ) and not self.allow_source_pose_drift:
            raise ValueError(
                f"scene rotation for object {obj.name!r} differs from captured native pose"
            )
        return dict(rotation)

    def _object_pose(
        self,
        scene_id: str,
        obj: SceneObject,
        position: Vec3 | None = None,
    ) -> dict[str, Any]:
        requested_position = position or obj.position
        return {
            "objectName": obj.name,
            "position": {
                "x": float(requested_position.x),
                "y": float(requested_position.z),
                "z": float(requested_position.y),
            },
            "rotation": self._native_rotation_for(scene_id, obj),
        }

    def _restore_scene_state(self, scene: Scene) -> Any:
        controller = self._require_active()
        if scene.scene_id not in self.scene_names:
            raise KeyError(scene.scene_id)
        self._activate_scene(scene.scene_id)
        self._objects_by_name(scene.objects)
        camera = scene.camera_by_id("main")
        state = self._camera_states.get(self._camera_key(scene.scene_id, camera))
        if state is None:
            raise ValueError("camera pose was not captured by this adapter")
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "SetObjectPoses",
                action="SetObjectPoses",
                objectPoses=[
                    self._object_pose(scene.scene_id, obj)
                    for obj in scene.objects
                    if obj.movable
                ],
                placeStationary=True,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "SetObjectPoses",
            scene.scene_id,
        )
        try:
            event = self._step(
                controller,
                "TeleportFull",
                action="TeleportFull",
                position=dict(state["position"]),
                rotation=dict(state["rotation"]),
                horizon=state["horizon"],
                standing=state["standing"],
                forceAction=True,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "TeleportFull",
            scene.scene_id,
        )
        self._validate_returned_state(
            scene,
            event,
            expected_positions,
            expected_rotations,
        )
        self._current_scene = scene
        return event

    @staticmethod
    def _angles_close(
        left: float,
        right: float,
        *,
        tolerance_degrees: float = _ANGLE_TOLERANCE_DEGREES,
    ) -> bool:
        difference = (left - right + 180.0) % 360.0 - 180.0
        return abs(difference) <= tolerance_degrees

    def _validate_returned_state(
        self,
        scene: Scene,
        event: Any,
        expected_positions: dict[str, Vec3],
        expected_rotations: dict[str, dict[str, float]],
        *,
        total_position_residual_limits_by_name: Mapping[str, float] | None = None,
        rotation_residual_limits_by_name: Mapping[str, float] | None = None,
    ) -> None:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError(
                "pose application returned no object metadata"
            )
        raw_objects = _validated_native_object_metadata(raw_objects)
        raw_objects = _domain_object_metadata(raw_objects)
        names = [item["name"] for item in raw_objects]
        by_name = dict(zip(names, raw_objects, strict=True))
        expected_by_name = self._objects_by_name(scene.objects)
        if set(by_name) != set(expected_by_name):
            raise AI2ThorNativeReturnError(
                "stable object names changed during pose application"
            )
        for name in expected_by_name:
            metadata = by_name[name]
            try:
                position = ai2thor_position_to_world(Vec3(**metadata["position"]))
                native_rotation = {
                    axis: float(metadata["rotation"][axis]) for axis in ("x", "y", "z")
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise AI2ThorNativeReturnError(
                    f"object {name!r} returned an invalid pose"
                ) from exc
            expected_position = expected_positions[name]
            expected_rotation = expected_rotations[name]
            expected_coordinates = (
                expected_position.x,
                expected_position.y,
                expected_position.z,
            )
            observed_coordinates = (position.x, position.y, position.z)
            residual_limit = (
                None
                if total_position_residual_limits_by_name is None
                else total_position_residual_limits_by_name.get(name)
            )
            if residual_limit is None:
                position_matches = np.allclose(
                    observed_coordinates,
                    expected_coordinates,
                    atol=1e-5,
                    rtol=0.0,
                )
            else:
                rounding_allowance_m = 4.0 * math.ulp(residual_limit)
                position_matches = (
                    math.dist(
                        observed_coordinates,
                        expected_coordinates,
                    )
                    <= residual_limit + rounding_allowance_m
                )
            rotation_residual_limit = (
                _OBJECT_ROTATION_TOLERANCE_DEGREES
                if rotation_residual_limits_by_name is None
                else rotation_residual_limits_by_name.get(
                    name,
                    _OBJECT_ROTATION_TOLERANCE_DEGREES,
                )
            )
            if not position_matches or not all(
                self._angles_close(
                    native_rotation[axis],
                    expected_rotation[axis],
                    tolerance_degrees=rotation_residual_limit,
                )
                for axis in ("x", "y", "z")
            ):
                raise AI2ThorNativeReturnError(
                    f"object {name!r} pose changed during pose application"
                )
        observed_camera = self._camera(event.metadata, scene.scene_id)
        self._validate_camera_fixed(scene.camera_by_id("main"), observed_camera)

    @staticmethod
    def _native_scene_at_rest(event: Any) -> bool:
        value = event.metadata.get("isSceneAtRest")
        if type(value) is not bool:
            raise AI2ThorNativeReturnError(
                "AI2-THOR isSceneAtRest must be an exact boolean"
            )
        return value

    @staticmethod
    def _native_object_is_moving(event: Any, object_name: str) -> bool:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError(
                "pose application returned no object metadata"
            )
        matches = [
            item
            for item in raw_objects
            if isinstance(item, dict) and item.get("name") == object_name
        ]
        if len(matches) != 1:
            raise AI2ThorNativeReturnError(
                f"expected one native object named {object_name!r}"
            )
        value = matches[0].get("isMoving")
        if type(value) is not bool:
            raise AI2ThorNativeReturnError(
                "AI2-THOR isMoving must be an exact boolean"
            )
        return value

    @staticmethod
    def _native_object_id_for_name(event: Any, object_name: str) -> str:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise RuntimeError("pose application returned no object metadata")
        matches = [
            item
            for item in raw_objects
            if isinstance(item, dict) and item.get("name") == object_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one native object named {object_name!r}"
            )
        native_object_id = matches[0].get("objectId")
        if type(native_object_id) is not str or not native_object_id:
            raise RuntimeError(
                f"native object named {object_name!r} has no valid objectId"
            )
        return native_object_id

    @staticmethod
    def _validate_camera_fixed(expected: Camera, observed: Camera) -> None:
        if expected.width != observed.width or expected.height != observed.height:
            raise AI2ThorNativeReturnError(
                "camera dimensions changed during pose application"
            )
        if not np.allclose(
            expected.intrinsics, observed.intrinsics, atol=1e-8, rtol=0.0
        ) or not np.allclose(
            expected.world_to_camera, observed.world_to_camera, atol=1e-5, rtol=0.0
        ):
            raise AI2ThorNativeReturnError(
                "camera pose or intrinsics changed during pose application"
            )

    def apply_object_xy_observed(
        self,
        scene: Scene,
        object_id: str,
        x: float,
        y: float,
    ) -> AI2ThorPoseApplication:
        """Apply one command and retain both canonical and native observations."""
        controller = self._require_active()
        target = scene.object_by_id(object_id)
        if not target.movable:
            raise ValueError(f"object {object_id!r} is not movable")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("object X/Y must be finite")
        self._objects_by_name(scene.objects)
        if self._current_scene != scene:
            current_scene = self._current_scene
            if current_scene is not None and self._is_analysis_overlay(
                current_scene, scene
            ):
                current_event = self._current_event_for_scene(current_scene)
            else:
                current_event = self._restore_scene_state(scene)
        else:
            current_event = self._current_event_for_scene(scene)
        expected_positions = {
            obj.name: (
                Vec3(x=x, y=y, z=obj.position.z)
                if obj.object_id == object_id
                else obj.position
            )
            for obj in scene.objects
        }
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        native_object_id = self._native_object_id_for_name(
            current_event,
            target.name,
        )
        commanded_position = expected_positions[target.name]
        teleport_vertical_guard_m = _TELEPORT_VERTICAL_GUARD_M
        # SetObjectPoses removes every movable object omitted from its payload.
        # TeleportObject is the native single-object edit and avoids re-emitting
        # unrelated poses, while the validation below still checks the full scene.
        try:
            event = self._step(
                controller,
                "TeleportObject",
                action="TeleportObject",
                objectId=native_object_id,
                position={
                    "x": commanded_position.x,
                    "y": commanded_position.z + teleport_vertical_guard_m,
                    "z": commanded_position.y,
                },
                rotation=dict(expected_rotations[target.name]),
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "TeleportObject",
            scene.scene_id,
        )
        self._validate_returned_state(
            scene,
            event,
            expected_positions,
            expected_rotations,
        )
        native_observed = self._scene_from_event(scene.scene_id, event)
        observed_scene = self._stable_observed_scene(scene, native_observed)
        commanded_scene = self._commanded_scene(
            scene,
            observed_scene,
            object_id,
            commanded_position,
        )
        observed_position = observed_scene.object_by_id(object_id).position
        position_residual_m = math.dist(
            (
                commanded_position.x,
                commanded_position.y,
                commanded_position.z,
            ),
            (
                observed_position.x,
                observed_position.y,
                observed_position.z,
            ),
        )
        is_scene_at_rest = self._native_scene_at_rest(event)
        subject_is_moving = self._native_object_is_moving(event, target.name)
        observation = self._observation_from_event(observed_scene, event)
        result = AI2ThorPoseApplication(
            commanded_scene=commanded_scene,
            observed_scene=observed_scene,
            commanded_position=commanded_position,
            observed_position=observed_position,
            position_residual_m=position_residual_m,
            observation=observation,
            is_scene_at_rest=is_scene_at_rest,
            subject_is_moving=subject_is_moving,
        )
        # The retained event contains the observed native pose, not the ideal
        # command.  Keep that identity exact; legacy callers still receive the
        # commanded scene from ``with_object_xy`` and restoration is explicit.
        self._current_scene = observed_scene
        return result

    def apply_receptacle_spawn_point_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        position: AI2ThorNativePosition,
    ) -> AI2ThorPoseApplication:
        """Audit one externally selected endpoint with native surface placement.

        This method is deliberately not a search API.  It accepts exactly one
        point from one source-bound map, executes one ``PlaceObjectAtPoint``,
        and fails closed if native collision or pose validation rejects it.
        """

        checked_map, subject, native_subject_object_id = (
            self._receptacle_audit_source(scene, spawn_map)
        )
        checked_position = _strict_native_position(position, "spawn audit position")
        if checked_position not in checked_map.positions:
            raise ValueError("spawn audit position is not in the source-bound map")
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            checked_position,
        )

    def apply_receptacle_endpoint_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        *,
        x: float,
        y: float,
    ) -> AI2ThorPoseApplication:
        """Audit one exact world-XY endpoint without snapping or searching.

        The source-bound map contributes only the single exact native support
        height required by this horizontal-support API.  The endpoint X/Y is
        supplied by the platform-neutral solver and is never replaced by a
        nearby returned coordinate.  Multi-height receptacles are outside this
        contract and fail before any native action.
        """

        checked_map, subject, native_subject_object_id = (
            self._receptacle_audit_source(scene, spawn_map)
        )
        endpoint_x = _strict_finite_float(x, "receptacle endpoint x")
        endpoint_y = _strict_finite_float(y, "receptacle endpoint y")
        native_heights = {item.y for item in checked_map.positions}
        if len(native_heights) != 1:
            raise ValueError(
                "exact receptacle endpoint audit requires one native support height"
            )
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            AI2ThorNativePosition(
                x=endpoint_x,
                y=next(iter(native_heights)),
                z=endpoint_y,
            ),
        )

    def apply_receptacle_endpoint_settled_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        *,
        x: float,
        y: float,
        max_pass_steps: int,
        max_subject_rotation_residual_degrees: float | None = None,
    ) -> AI2ThorPoseApplication:
        """Place once, then wait boundedly for the runtime-only endpoint to settle."""

        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        if max_subject_rotation_residual_degrees is not None:
            max_subject_rotation_residual_degrees = _strict_finite_float(
                max_subject_rotation_residual_degrees,
                "subject rotation residual limit",
            )
            if max_subject_rotation_residual_degrees <= 0.0:
                raise ValueError("subject rotation residual limit must be positive")
        checked_map, subject, native_subject_object_id = self._receptacle_audit_source(
            scene, spawn_map
        )
        endpoint_x = _strict_finite_float(x, "receptacle endpoint x")
        endpoint_y = _strict_finite_float(y, "receptacle endpoint y")
        native_heights = {item.y for item in checked_map.positions}
        if len(native_heights) != 1:
            raise ValueError(
                "exact receptacle endpoint audit requires one native support height"
            )
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            AI2ThorNativePosition(
                x=endpoint_x,
                y=next(iter(native_heights)),
                z=endpoint_y,
            ),
            max_pass_steps=max_pass_steps,
            max_subject_rotation_residual_degrees=(
                max_subject_rotation_residual_degrees
            ),
        )

    def _receptacle_audit_source(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
    ) -> tuple[AI2ThorReceptacleSpawnMap, SceneObject, str]:
        current_event = self._current_event_for_scene(scene)
        checked_map = _strict_receptacle_spawn_map(spawn_map)
        subject = scene.object_by_id(checked_map.subject_object_id)
        if subject.support_object_id != checked_map.support_object_id:
            raise ValueError("spawn map support does not match the current scene")
        runtime_identity = self.runtime_identity()
        scene_sha256 = _receptacle_scene_sha256(
            scene,
            checked_map.surface_patches,
        )
        native_subject_object_id = self._native_object_id_for_name(
            current_event,
            subject.name,
        )
        support = scene.object_by_id(checked_map.support_object_id)
        native_support_object_id = self._native_object_id_for_name(
            current_event,
            support.name,
        )
        if (
            checked_map.scene_id != scene.scene_id
            or checked_map.scene_sha256 != scene_sha256
            or checked_map.runtime_identity != runtime_identity
            or checked_map.native_subject_object_id != native_subject_object_id
            or checked_map.native_support_object_id != native_support_object_id
        ):
            raise ValueError("spawn map does not close the exact current source")
        return checked_map, subject, native_subject_object_id

    def _apply_receptacle_native_position_observed(
        self,
        scene: Scene,
        subject: SceneObject,
        native_subject_object_id: str,
        checked_position: AI2ThorNativePosition,
        *,
        max_pass_steps: int | None = None,
        max_subject_rotation_residual_degrees: float | None = None,
    ) -> AI2ThorPoseApplication:
        controller = self._require_active()
        commanded_position = Vec3(
            x=checked_position.x,
            y=checked_position.z,
            z=subject.position.z,
        )
        expected_positions = {
            obj.name: (
                commanded_position
                if obj.object_id == subject.object_id
                else obj.position
            )
            for obj in scene.objects
        }
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "PlaceObjectAtPoint",
                action="PlaceObjectAtPoint",
                objectId=native_subject_object_id,
                position={
                    "x": checked_position.x,
                    "y": checked_position.y,
                    "z": checked_position.z,
                },
                rotation=dict(expected_rotations[subject.name]),
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "PlaceObjectAtPoint",
            scene.scene_id,
        )
        if max_pass_steps is None:
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
            )
        else:
            immediate_native = self._scene_from_event(scene.scene_id, event)
            immediate_observed = self._stable_observed_scene(scene, immediate_native)
            self._current_scene = immediate_observed
            settlement = self.settle_scene_observed(
                immediate_observed,
                max_pass_steps=max_pass_steps,
            )
            event = self._current_event_for_scene(settlement.observed_scene)
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
                total_position_residual_limits_by_name={
                    subject.name: _RUNTIME_RECEPTACLE_POSITION_RESIDUAL_M
                },
                rotation_residual_limits_by_name=(
                    None
                    if max_subject_rotation_residual_degrees is None
                    else {
                        subject.name: max_subject_rotation_residual_degrees,
                    }
                ),
            )
        native_observed = self._scene_from_event(scene.scene_id, event)
        observed_scene = self._stable_observed_scene(scene, native_observed)
        commanded_scene = self._commanded_scene(
            scene,
            observed_scene,
            subject.object_id,
            commanded_position,
        )
        observed_position = observed_scene.object_by_id(subject.object_id).position
        position_residual_m = math.dist(
            (
                commanded_position.x,
                commanded_position.y,
                commanded_position.z,
            ),
            (
                observed_position.x,
                observed_position.y,
                observed_position.z,
            ),
        )
        result = AI2ThorPoseApplication(
            commanded_scene=commanded_scene,
            observed_scene=observed_scene,
            commanded_position=commanded_position,
            observed_position=observed_position,
            position_residual_m=position_residual_m,
            observation=self._observation_from_event(observed_scene, event),
            is_scene_at_rest=self._native_scene_at_rest(event),
            subject_is_moving=self._native_object_is_moving(event, subject.name),
        )
        self._current_scene = observed_scene
        return result

    def with_object_xy(
        self,
        scene: Scene,
        object_id: str,
        x: float,
        y: float,
    ) -> Scene:
        return self.apply_object_xy_observed(scene, object_id, x, y).commanded_scene

    @staticmethod
    def _png_bytes(frame: np.ndarray) -> bytes:
        stream = BytesIO()
        Image.fromarray(frame, mode="RGB").save(stream, format="PNG")
        return stream.getvalue()

    @staticmethod
    def _npy_bytes(array: np.ndarray) -> bytes:
        stream = BytesIO()
        np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    def _stable_instance_pixel_counts(
        self,
        scene: Scene,
        event: Any,
    ) -> dict[str, int]:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError(
                "observation returned no object metadata"
            )
        native_by_name = {
            str(item.get("name")): item
            for item in _domain_object_metadata(raw_objects)
            if isinstance(item, dict)
        }
        if len(native_by_name) != len(_domain_object_metadata(raw_objects)):
            raise AI2ThorNativeReturnError(
                "observation returned duplicate object names"
            )
        masks = getattr(event, "instance_masks", None)
        if not isinstance(masks, Mapping):
            raise AI2ThorNativeReturnError(
                "observation returned invalid instance masks"
            )
        validated_masks: dict[str, np.ndarray] = {}
        for native_id in masks:
            if type(native_id) is not str:
                raise AI2ThorNativeReturnError(
                    "observation returned invalid instance masks: keys must be strings"
                )
            mask = masks[native_id]
            if (
                not isinstance(mask, np.ndarray)
                or mask.shape != (self.height, self.width)
                or mask.dtype != np.bool_
            ):
                raise AI2ThorNativeReturnError(
                    "observation returned invalid instance masks: "
                    "values must be boolean HxW numpy arrays"
                )
            validated_masks[native_id] = mask
        counts: dict[str, int] = {}
        for obj in scene.objects:
            metadata = native_by_name.get(obj.name)
            if metadata is None:
                raise AI2ThorNativeReturnError(
                    f"observation is missing stable object name {obj.name!r}"
                )
            native_id = str(metadata.get("objectId"))
            mask = validated_masks.get(native_id)
            if mask is None:
                counts[obj.object_id] = 0
                continue
            counts[obj.object_id] = int(np.count_nonzero(mask))
        return counts

    def _observation_from_event(
        self,
        scene: Scene,
        event: Any,
    ) -> AI2ThorObservation:
        camera = scene.camera_by_id("main")
        rgb, depth, instance = self._validated_frames(event)
        return AI2ThorObservation.create(
            scene=scene,
            rgb_png=self._png_bytes(rgb),
            depth_npy=self._npy_bytes(depth),
            instance_png=self._png_bytes(instance),
            pointcloud_ply=self._pointcloud_bytes(camera, depth, rgb),
            instance_pixel_counts=self._stable_instance_pixel_counts(scene, event),
            is_scene_at_rest=self._native_scene_at_rest(event),
        )

    def capture_current_observation(self, scene: Scene) -> AI2ThorObservation:
        """Capture frames from the exact current event without replaying poses."""
        event = self._current_event_for_scene(scene)
        return self._observation_from_event(scene, event)

    def _validated_frames(
        self,
        event: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_event = self._event if event is None else event
        if source_event is None:
            raise RuntimeError("no AI2-THOR event is available")
        rgb = np.asarray(getattr(source_event, "frame", None))
        depth = np.asarray(getattr(source_event, "depth_frame", None))
        instance = np.asarray(
            getattr(source_event, "instance_segmentation_frame", None)
        )
        expected_color = (self.height, self.width, 3)
        expected_depth = (self.height, self.width)
        if rgb.shape != expected_color or rgb.dtype != np.uint8:
            raise AI2ThorNativeReturnError(
                "AI2-THOR RGB frame must be HxWx3 uint8"
            )
        if instance.shape != expected_color or instance.dtype != np.uint8:
            raise AI2ThorNativeReturnError(
                "AI2-THOR instance frame must be HxWx3 uint8"
            )
        if depth.shape != expected_depth or not np.issubdtype(
            depth.dtype, np.number
        ):
            raise AI2ThorNativeReturnError(
                "AI2-THOR depth frame must be a numeric HxW array"
            )
        return rgb, depth.astype(np.float32, copy=False), instance

    @staticmethod
    def _validate_stem(stem: str) -> None:
        if (
            not stem
            or stem.strip() != stem
            or stem in {".", ".."}
            or Path(stem).name != stem
            or "/" in stem
            or "\\" in stem
        ):
            raise ValueError("artifact stem must be a safe filename component")

    def _pointcloud_bytes(
        self,
        camera: Camera,
        depth: np.ndarray,
        rgb: np.ndarray,
    ) -> bytes:
        rows, columns = np.mgrid[0 : self.height : 4, 0 : self.width : 4]
        z = depth[rows, columns]
        valid = np.isfinite(z) & (z > 0.0)
        rows, columns, z = rows[valid], columns[valid], z[valid]
        fx, fy = camera.intrinsics[0], camera.intrinsics[4]
        cx, cy = camera.intrinsics[2], camera.intrinsics[5]
        x = (columns - cx) * z / fx
        y = -(rows - cy) * z / fy
        camera_points = np.stack([x, y, z, np.ones_like(z)])
        world_points = np.linalg.inv(matrix4(camera.world_to_camera)) @ camera_points
        colors = rgb[rows, columns]
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {world_points.shape[1]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        lines.extend(
            (
                f"{world_points[0, index]:.9g} "
                f"{world_points[1, index]:.9g} "
                f"{world_points[2, index]:.9g} "
                f"{int(colors[index, 0])} "
                f"{int(colors[index, 1])} "
                f"{int(colors[index, 2])}"
            )
            for index in range(world_points.shape[1])
        )
        return ("\n".join(lines) + "\n").encode("ascii")

    def _write_pointcloud(
        self,
        camera: Camera,
        depth: np.ndarray,
        rgb: np.ndarray,
        destination: Path,
    ) -> None:
        destination.write_bytes(self._pointcloud_bytes(camera, depth, rgb))

    def render_assets(
        self,
        scene: Scene,
        camera_id: str,
        destination_root: Path,
        stem: str,
    ) -> RenderedAssets:
        self._require_active()
        camera = scene.camera_by_id(camera_id)
        if camera_id != "main":
            raise KeyError(camera_id)
        self._validate_stem(stem)
        self._restore_scene_state(scene)
        rgb, depth, instance = self._validated_frames()
        destination_root.mkdir(parents=True, exist_ok=True)
        assets = RenderedAssets(
            rgb_path=destination_root / f"{stem}-rgb.png",
            depth_path=destination_root / f"{stem}-depth.npy",
            instance_path=destination_root / f"{stem}-instance.png",
            pointcloud_path=destination_root / f"{stem}-pointcloud.ply",
        )
        if len(
            {
                assets.rgb_path,
                assets.depth_path,
                assets.instance_path,
                assets.pointcloud_path,
            }
        ) != 4:
            raise ValueError("artifact paths must be distinct")
        Image.fromarray(rgb, mode="RGB").save(assets.rgb_path)
        np.save(assets.depth_path, depth, allow_pickle=False)
        Image.fromarray(instance, mode="RGB").save(assets.instance_path)
        self._write_pointcloud(camera, depth, rgb, assets.pointcloud_path)
        return assets
