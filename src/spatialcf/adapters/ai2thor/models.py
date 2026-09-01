from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from spatialcf.adapters.base import (
    AdapterBinding,
    AdapterCameraApplication,
    AdapterFloorEnvelope,
    AdapterObservation,
    AdapterPose,
    AdapterPosition,
    AdapterRuntimeIdentity,
    AdapterSpawnMap,
    AdapterSupportFact,
    AdapterSurfacePatch,
    AppliedCertifiedEdit,
    CapturedSource,
    CertifiedEditApplication,
    InstanceEvidenceProvenance,
    SettledReadback,
)
from spatialcf.domain.scene import (
    OBB,
    CollisionObstacle,
    Scene,
    SubjectPositionRegion,
    Vec2,
    Vec3,
)

_RECEPTACLE_TRIGGER_GRID_QUANTIZATION_M = 1e-5
_RECEPTACLE_TRIGGER_GRID_SIDE = 21
_RECEPTACLE_TRIGGER_GRID_SIZE = _RECEPTACLE_TRIGGER_GRID_SIDE**2
_TELEPORT_VERTICAL_GUARD_M = 1e-6


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
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
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
            type(coordinate) not in (int, float) or not math.isfinite(float(coordinate))
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
        raise ValueError("room floorPolygon must be a convex axis-aligned rectangle")
    return room_id, (minimum_x, minimum_z, maximum_x, maximum_z)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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
            if (
                domain
                or self.support_object_id is not None
                or self.floor_object_id is not None
            ):
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
                or any(
                    character not in "0123456789abcdef"
                    for character in self.source_sha256
                )
            ):
                raise ValueError("source_sha256 must be lowercase SHA-256 hex")
            bounds = self.source_floor_xz_bounds
            if (
                type(bounds) is not tuple
                or len(bounds) != 4
                or any(
                    type(value) not in (int, float) or not math.isfinite(float(value))
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
                raise ValueError(
                    "procedural provenance requires native Procedural scene"
                )
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
    instance_colors: tuple[tuple[str, tuple[int, int, int]], ...]

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
        instance_colors: tuple[tuple[str, tuple[int, int, int]], ...],
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
            instance_colors=tuple(
                (object_id, tuple(color)) for object_id, color in instance_colors
            ),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.instance_pixel_counts, Mapping) or any(
            type(object_id) is not str
            or type(count) is not int
            or count < 0
            for object_id, count in self.instance_pixel_counts.items()
        ):
            raise TypeError("AI2-THOR instance pixel counts must be exact")
        if set(self.instance_pixel_counts) != {
            obj.object_id for obj in self.scene.objects
        }:
            raise ValueError(
                "AI2-THOR instance pixel counts must cover every scene object"
            )
        if type(self.instance_colors) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not tuple
            or len(item[1]) != 3
            or any(
                type(channel) is not int or not 0 <= channel <= 255
                for channel in item[1]
            )
            for item in self.instance_colors
        ):
            raise TypeError("AI2-THOR instance colors must be exact RGB pairs")
        if (
            self.instance_colors != tuple(sorted(self.instance_colors))
            or len({item[0] for item in self.instance_colors})
            != len(self.instance_colors)
            or len({item[1] for item in self.instance_colors})
            != len(self.instance_colors)
        ):
            raise ValueError("AI2-THOR instance colors must be sorted and unique")
        positive_ids = {
            object_id
            for object_id, count in self.instance_pixel_counts.items()
            if count > 0
        }
        if {item[0] for item in self.instance_colors} != positive_ids:
            raise ValueError("AI2-THOR instance colors must cover positive pixels")


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

    adapter: AI2ThorAdapter  # noqa: F821
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
                        patch.x_min + (patch.x_max - patch.x_min) * x_index / 20.0,
                        patch.z_min + (patch.z_max - patch.z_min) * z_index / 20.0,
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
            expected_count = len(self.surface_patches) * _RECEPTACLE_TRIGGER_GRID_SIZE
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


def adapter_position_from_native(value: AI2ThorNativePosition) -> AdapterPosition:
    if type(value) is not AI2ThorNativePosition:
        raise TypeError("native position must be exact")
    return AdapterPosition(x=value.x, y=value.y, z=value.z)


def native_position_from_adapter(value: AdapterPosition) -> AI2ThorNativePosition:
    if type(value) is not AdapterPosition:
        raise TypeError("adapter position must be exact")
    return AI2ThorNativePosition(x=value.x, y=value.y, z=value.z)


def adapter_pose_from_native(value: AI2ThorAgentPose) -> AdapterPose:
    if type(value) is not AI2ThorAgentPose:
        raise TypeError("native pose must be exact")
    return AdapterPose(
        position=adapter_position_from_native(value.position),
        yaw_degrees=value.yaw_degrees,
        horizon_degrees=value.horizon_degrees,
        standing=value.standing,
    )


def native_pose_from_adapter(value: AdapterPose) -> AI2ThorAgentPose:
    if type(value) is not AdapterPose:
        raise TypeError("adapter pose must be exact")
    return AI2ThorAgentPose(
        position=native_position_from_adapter(value.position),
        yaw_degrees=value.yaw_degrees,
        horizon_degrees=value.horizon_degrees,
        standing=value.standing,
    )


def adapter_runtime_identity_from_native(
    value: AI2ThorRuntimeIdentity,
) -> AdapterRuntimeIdentity:
    if type(value) is not AI2ThorRuntimeIdentity:
        raise TypeError("native runtime identity must be exact")
    return AdapterRuntimeIdentity(**asdict(value))


def adapter_observation_from_native(value: AI2ThorObservation) -> AdapterObservation:
    if type(value) is not AI2ThorObservation:
        raise TypeError("native observation must be exact")
    return AdapterObservation.create(
        scene=value.scene,
        rgb_png=value.rgb_png,
        depth_npy=value.depth_npy,
        instance_png=value.instance_png,
        pointcloud_ply=value.pointcloud_ply,
        instance_pixel_counts=tuple(sorted(value.instance_pixel_counts.items())),
        is_settled=value.is_scene_at_rest,
        instance_colors=value.instance_colors,
        instance_evidence_provenance=(
            InstanceEvidenceProvenance.SAME_EVENT_INSTANCE_SEGMENTATION
        ),
    )


def adapter_support_fact_from_native(
    value: AI2ThorNativeSupportFact,
) -> AdapterSupportFact:
    if type(value) is not AI2ThorNativeSupportFact:
        raise TypeError("native support fact must be exact")
    return AdapterSupportFact(
        scene_id=value.scene_id,
        object_id=value.object_id,
        object_name=value.object_name,
        native_object_id=value.native_object_id,
        raw_parent_object_ids=value.raw_parent_object_ids,
        structural_parent_object_ids=value.structural_parent_object_ids,
        domain_parent_object_ids=value.domain_parent_object_ids,
        support_kind=value.support_kind.value,
        support_object_id=value.support_object_id,
        floor_object_id=value.floor_object_id,
    )


def adapter_floor_envelope_from_native(
    value: AI2ThorFloorEnvelope,
) -> AdapterFloorEnvelope:
    if type(value) is not AI2ThorFloorEnvelope:
        raise TypeError("native floor envelope must be exact")
    return AdapterFloorEnvelope(
        scene_id=value.scene_id,
        floor_object_id=value.floor_object_id,
        floor_name=value.floor_name,
        native_aabb=value.native_aabb,
        floor_top_z=value.floor_top_z,
        clearance_m=value.clearance_m,
        polygon_xy=value.polygon_xy,
    )


def adapter_spawn_map_from_native(
    value: AI2ThorReceptacleSpawnMap,
    *,
    binding: AdapterBinding,
    position_region: SubjectPositionRegion | None,
) -> AdapterSpawnMap:
    if type(value) is not AI2ThorReceptacleSpawnMap:
        raise TypeError("native spawn map must be exact")
    if type(binding) is not AdapterBinding:
        raise TypeError("spawn binding must be exact")
    return AdapterSpawnMap(
        binding=binding,
        runtime_identity=adapter_runtime_identity_from_native(value.runtime_identity),
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        positions=tuple(adapter_position_from_native(item) for item in value.positions),
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=tuple(
            AdapterSurfacePatch(
                x_min=item.x_min,
                x_max=item.x_max,
                native_y=item.native_y,
                z_min=item.z_min,
                z_max=item.z_max,
            )
            for item in value.surface_patches
        ),
        position_region=position_region,
    )


def ai2thor_spawn_map_from_adapter(value: AdapterSpawnMap) -> AI2ThorReceptacleSpawnMap:
    """Reconstruct one concrete map only inside the AI2-THOR boundary."""
    if type(value) is not AdapterSpawnMap:
        raise TypeError("adapter spawn map must be exact")
    return AI2ThorReceptacleSpawnMap(
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        runtime_identity=AI2ThorRuntimeIdentity(**asdict(value.runtime_identity)),
        positions=tuple(
            AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
            for item in value.positions
        ),
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=tuple(
            AI2ThorReceptacleSurfacePatch(
                x_min=item.x_min,
                x_max=item.x_max,
                native_y=item.native_y,
                z_min=item.z_min,
                z_max=item.z_max,
            )
            for item in value.surface_patches
        ),
    )


def applied_certified_edit_from_native(
    value: AI2ThorPoseApplication,
    *,
    application: CertifiedEditApplication,
) -> AppliedCertifiedEdit:
    """Erase native event types from a single acknowledged placement."""
    if type(value) is not AI2ThorPoseApplication:
        raise TypeError("native applied pose must be exact")
    if type(application) is not CertifiedEditApplication:
        raise TypeError("certified application must be exact")
    return AppliedCertifiedEdit(
        application=application,
        edit=application.edit,
        commanded_scene=value.commanded_scene,
        observed_scene=value.observed_scene,
        commanded_position=AdapterPosition(
            x=value.commanded_position.x,
            y=value.commanded_position.y,
            z=value.commanded_position.z,
        ),
        observed_position=AdapterPosition(
            x=value.observed_position.x,
            y=value.observed_position.y,
            z=value.observed_position.z,
        ),
        position_residual_m=value.position_residual_m,
        observation=adapter_observation_from_native(value.observation),
        is_scene_at_rest=value.is_scene_at_rest,
        subject_is_moving=value.subject_is_moving,
        settlement_pass_steps=0,
        binding=application.source.binding,
    )


def settled_readback_from_native(
    value: AI2ThorSceneSettlement,
    *,
    applied: AppliedCertifiedEdit,
) -> SettledReadback:
    """Erase concrete settlement state after exactly one bounded readback."""
    if type(value) is not AI2ThorSceneSettlement:
        raise TypeError("native settlement must be exact")
    if type(applied) is not AppliedCertifiedEdit:
        raise TypeError("applied certified edit must be exact")
    observed = value.observed_scene.object_by_id(applied.edit.subject_id).position
    observed_position = AdapterPosition(
        x=observed.x,
        y=observed.y,
        z=observed.z,
    )
    return SettledReadback(
        applied=applied,
        commanded_scene=applied.commanded_scene,
        observed_scene=value.observed_scene,
        commanded_position=applied.commanded_position,
        observed_position=observed_position,
        position_residual_m=math.dist(
            (
                applied.commanded_position.x,
                applied.commanded_position.y,
                applied.commanded_position.z,
            ),
            (
                observed_position.x,
                observed_position.y,
                observed_position.z,
            ),
        ),
        observation=adapter_observation_from_native(value.observation),
        is_scene_at_rest=value.observation.is_scene_at_rest,
        subject_is_moving=False,
        settlement_pass_steps=value.pass_steps,
        binding=applied.binding,
    )


def adapter_camera_application_from_native(
    value: AI2ThorCameraApplication,
    *,
    source: CapturedSource,
) -> AdapterCameraApplication:
    if type(value) is not AI2ThorCameraApplication:
        raise TypeError("native camera application must be exact")
    if type(source) is not CapturedSource:
        raise TypeError("camera application source must be exact")
    return AdapterCameraApplication(
        source=source,
        binding=source.binding,
        requested_pose=adapter_pose_from_native(value.requested_pose),
        observed_pose=adapter_pose_from_native(value.observed_pose),
        observed_camera_position=adapter_position_from_native(
            value.observed_camera_position
        ),
        observed_scene=value.observed_scene,
        observation=adapter_observation_from_native(value.observation),
        position_residual_m=value.position_residual_m,
        yaw_residual_degrees=value.yaw_residual_degrees,
        horizon_residual_degrees=value.horizon_residual_degrees,
    )
