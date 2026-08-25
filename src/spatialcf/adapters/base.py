import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from spatialcf.domain.edit import CanonicalEdit
from spatialcf.domain.request import InterventionSpec
from spatialcf.domain.scene import OBB, Scene, SubjectPositionRegion, Vec2, Vec3
from spatialcf.domain.serialization import canonical_json_bytes


@dataclass(frozen=True)
class RenderedAssets:
    rgb_path: Path
    depth_path: Path
    instance_path: Path
    pointcloud_path: Path


@dataclass(frozen=True)
class AdapterBinding:
    """Opaque, Canonical-only binding between one capture and its readbacks."""

    scene_id: str
    token: str

    def __post_init__(self) -> None:
        if type(self.scene_id) is not str or not self.scene_id:
            raise TypeError("binding scene_id must be a non-empty exact string")
        if type(self.token) is not str or not self.token:
            raise TypeError("binding token must be a non-empty exact string")


@dataclass(frozen=True)
class CaptureRequest:
    scene_id: str
    camera_id: str

    def __post_init__(self) -> None:
        if type(self.scene_id) is not str or not self.scene_id:
            raise TypeError("capture scene_id must be a non-empty exact string")
        if type(self.camera_id) is not str or not self.camera_id:
            raise TypeError("capture camera_id must be a non-empty exact string")


@dataclass(frozen=True)
class CapturedSource:
    request: CaptureRequest
    scene: Scene
    binding: AdapterBinding

    def __post_init__(self) -> None:
        if type(self.request) is not CaptureRequest or type(self.scene) is not Scene:
            raise TypeError("captured source must contain exact Canonical values")
        if type(self.binding) is not AdapterBinding:
            raise TypeError("captured source binding must be exact")
        if (
            self.scene.scene_id != self.request.scene_id
            or self.binding.scene_id != self.scene.scene_id
        ):
            raise ValueError("captured source binding does not match its scene")
        self.scene.camera_by_id(self.request.camera_id)


@dataclass(frozen=True)
class CapturedSupport:
    source: CapturedSource
    scene: Scene
    binding: AdapterBinding

    def __post_init__(self) -> None:
        if type(self.source) is not CapturedSource or type(self.scene) is not Scene:
            raise TypeError("captured support must contain exact Canonical values")
        if type(self.binding) is not AdapterBinding:
            raise TypeError("captured support binding must be exact")
        if self.scene != self.source.scene or self.binding != self.source.binding:
            raise ValueError("captured support must retain its source binding")


@dataclass(frozen=True)
class CertifiedEditApplication:
    source: CapturedSource
    intervention: InterventionSpec
    edit: CanonicalEdit
    spawn_map: "AdapterSpawnMap"
    max_settlement_steps: int

    def __post_init__(self) -> None:
        if type(self.source) is not CapturedSource:
            raise TypeError("certified application source must be exact")
        if type(self.intervention) is not InterventionSpec:
            raise TypeError("certified application intervention must be exact")
        if type(self.edit) is not CanonicalEdit:
            raise TypeError("certified application edit must be exact")
        if type(self.spawn_map) is not AdapterSpawnMap:
            raise TypeError("certified application spawn map must be exact")
        if type(self.max_settlement_steps) is not int or self.max_settlement_steps <= 0:
            raise ValueError("certified application settlement steps must be positive")
        if (
            self.edit.subject_id != self.intervention.subject_id
            or self.source.request.camera_id != self.intervention.camera_id
        ):
            raise ValueError("certified application does not bind its edit")
        subject = self.source.scene.object_by_id(self.intervention.subject_id)
        self.source.scene.object_by_id(self.intervention.reference_id)
        if (
            self.spawn_map.binding != self.source.binding
            or self.spawn_map.scene_id != self.source.scene.scene_id
            or self.spawn_map.subject_object_id != subject.object_id
            or self.spawn_map.support_object_id != subject.support_object_id
        ):
            raise ValueError("certified application spawn map does not bind its source")


@dataclass(frozen=True)
class AppliedCertifiedEdit:
    application: CertifiedEditApplication
    edit: CanonicalEdit
    commanded_scene: Scene
    observed_scene: Scene
    commanded_position: "AdapterPosition"
    observed_position: "AdapterPosition"
    position_residual_m: float
    observation: "AdapterObservation"
    is_scene_at_rest: bool
    subject_is_moving: bool
    settlement_pass_steps: int
    binding: AdapterBinding

    def __post_init__(self) -> None:
        if type(self.application) is not CertifiedEditApplication:
            raise TypeError("applied application must be exact")
        if (
            type(self.edit) is not CanonicalEdit
            or type(self.commanded_scene) is not Scene
            or type(self.observed_scene) is not Scene
        ):
            raise TypeError("applied edit must contain exact Canonical values")
        if (
            type(self.commanded_position) is not AdapterPosition
            or type(self.observed_position) is not AdapterPosition
            or type(self.observation) is not AdapterObservation
        ):
            raise TypeError("applied edit must carry exact observed facts")
        if type(self.binding) is not AdapterBinding:
            raise TypeError("applied edit binding must be exact")
        if (
            type(self.is_scene_at_rest) is not bool
            or type(self.subject_is_moving) is not bool
        ):
            raise TypeError("applied edit rest and motion flags must be exact")
        if (
            type(self.settlement_pass_steps) is not int
            or self.settlement_pass_steps < 0
        ):
            raise ValueError("applied edit settlement count must be non-negative")
        if self.edit != self.application.edit:
            raise ValueError("applied edit must equal the certified edit")
        if (
            self.commanded_scene.scene_id != self.application.source.scene.scene_id
            or self.observed_scene.scene_id != self.application.source.scene.scene_id
            or self.binding != self.application.source.binding
            or self.binding != self.application.spawn_map.binding
        ):
            raise ValueError("applied edit does not retain its source binding")
        commanded = self.commanded_scene.object_by_id(self.edit.subject_id).position
        observed = self.observed_scene.object_by_id(self.edit.subject_id).position
        expected_commanded = self.application.source.scene.object_by_id(
            self.edit.subject_id
        ).position
        if (
            self.commanded_position
            != AdapterPosition(
                x=commanded.x,
                y=commanded.y,
                z=commanded.z,
            )
            or self.observed_position
            != AdapterPosition(
                x=observed.x,
                y=observed.y,
                z=observed.z,
            )
            or self.commanded_position
            != AdapterPosition(
                x=expected_commanded.x + self.edit.translation_xy_m.x,
                y=expected_commanded.y + self.edit.translation_xy_m.y,
                z=expected_commanded.z,
            )
        ):
            raise ValueError("applied edit positions do not bind its Canonical scenes")
        residual = math.dist(
            (
                self.commanded_position.x,
                self.commanded_position.y,
                self.commanded_position.z,
            ),
            (
                self.observed_position.x,
                self.observed_position.y,
                self.observed_position.z,
            ),
        )
        value = _strict_float(self.position_residual_m, "applied edit residual")
        if value < 0.0 or not math.isclose(value, residual, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("applied edit residual does not match observed position")
        object.__setattr__(self, "position_residual_m", value)
        if (
            self.observation.scene != self.observed_scene
            or self.observation.is_settled != self.is_scene_at_rest
            or (self.subject_is_moving and self.is_scene_at_rest)
        ):
            raise ValueError("applied edit observation does not match motion state")

    @property
    def scene(self) -> Scene:
        """Compatibility view of the concrete post-action Canonical scene."""
        return self.observed_scene


@dataclass(frozen=True)
class SettledReadback:
    applied: AppliedCertifiedEdit
    commanded_scene: Scene
    observed_scene: Scene
    commanded_position: "AdapterPosition"
    observed_position: "AdapterPosition"
    position_residual_m: float
    observation: "AdapterObservation"
    is_scene_at_rest: bool
    subject_is_moving: bool
    settlement_pass_steps: int
    binding: AdapterBinding

    def __post_init__(self) -> None:
        if (
            type(self.applied) is not AppliedCertifiedEdit
            or type(self.commanded_scene) is not Scene
            or type(self.observed_scene) is not Scene
        ):
            raise TypeError("settled readback must contain exact Canonical values")
        if (
            type(self.commanded_position) is not AdapterPosition
            or type(self.observed_position) is not AdapterPosition
            or type(self.observation) is not AdapterObservation
        ):
            raise TypeError("settled readback must carry exact observed facts")
        if type(self.binding) is not AdapterBinding:
            raise TypeError("settled readback binding must be exact")
        if (
            type(self.is_scene_at_rest) is not bool
            or type(self.subject_is_moving) is not bool
        ):
            raise TypeError("settled readback rest and motion flags must be exact")
        if (
            type(self.settlement_pass_steps) is not int
            or self.settlement_pass_steps < 0
        ):
            raise ValueError("settled readback pass count must be non-negative")
        if (
            self.commanded_scene != self.applied.commanded_scene
            or self.commanded_position != self.applied.commanded_position
            or self.observed_scene.scene_id != self.applied.observed_scene.scene_id
            or self.binding != self.applied.binding
        ):
            raise ValueError("settled readback does not retain its application binding")
        observed = self.observed_scene.object_by_id(
            self.applied.edit.subject_id
        ).position
        if self.observed_position != AdapterPosition(
            x=observed.x,
            y=observed.y,
            z=observed.z,
        ):
            raise ValueError("settled readback position does not bind observed scene")
        residual = math.dist(
            (
                self.commanded_position.x,
                self.commanded_position.y,
                self.commanded_position.z,
            ),
            (
                self.observed_position.x,
                self.observed_position.y,
                self.observed_position.z,
            ),
        )
        value = _strict_float(self.position_residual_m, "settled readback residual")
        if value < 0.0 or not math.isclose(value, residual, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "settled readback residual does not match observed position"
            )
        object.__setattr__(self, "position_residual_m", value)
        if (
            self.observation.scene != self.observed_scene
            or self.observation.is_settled != self.is_scene_at_rest
            or not self.is_scene_at_rest
            or self.subject_is_moving
        ):
            raise ValueError("settled readback does not carry a settled observation")

    @property
    def application(self) -> CertifiedEditApplication:
        return self.applied.application

    @property
    def scene(self) -> Scene:
        """Compatibility view of the settled Canonical scene."""
        return self.observed_scene


class AdapterOperationError(RuntimeError):
    """A normalized adapter operation could not produce its bound result."""


class AdapterActionRejected(AdapterOperationError):
    """The adapter received one explicit rejected platform action."""

    def __init__(self, reason: str) -> None:
        _strict_text(reason, "adapter action rejection")
        self.reason = " ".join(reason.split())
        super().__init__(self.reason)


class AdapterReturnRejected(AdapterOperationError):
    """A successful platform action returned invalid adapter evidence."""

    def __init__(self, reason: str) -> None:
        _strict_text(reason, "adapter return rejection")
        self.reason = " ".join(reason.split())
        super().__init__(self.reason)


class AdapterSettlementTimeout(AdapterOperationError):
    """An explicitly bounded adapter settlement did not complete."""


def _strict_text(value: str, label: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty exact string")


def _strict_digest(value: str, label: str) -> None:
    _strict_text(value, label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")


def _strict_float(value: float, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(f"{label} must be an exact finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class SourceCaptureOptions:
    """Bounded source-observation settings selected by generation."""

    max_settlement_steps: int
    floor_clearance_m: float
    navigation_agent_radius_m: float
    navigation_clearance_m: float

    def __post_init__(self) -> None:
        if type(self.max_settlement_steps) is not int or self.max_settlement_steps <= 0:
            raise ValueError("source capture max_settlement_steps must be positive")
        for name, raw_value in (
            ("floor_clearance_m", self.floor_clearance_m),
            ("navigation_agent_radius_m", self.navigation_agent_radius_m),
            ("navigation_clearance_m", self.navigation_clearance_m),
        ):
            value = _strict_float(raw_value, f"source capture {name}")
            if value < 0.0:
                raise ValueError(f"source capture {name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.navigation_agent_radius_m <= 0.0:
            raise ValueError("source capture navigation agent radius must be positive")


@dataclass(frozen=True)
class AdapterProceduralScene:
    """Immutable source bytes supplied to an adapter without a native model."""

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
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("revision", self.revision),
            ("split", self.split),
            ("source_loader_id", self.source_loader_id),
            ("source_loader_version", self.source_loader_version),
            ("room_id", self.room_id),
        ):
            _strict_text(value, f"procedural source {name}")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("procedural source index must be non-negative")
        if type(self.canonical_house_json) is not bytes:
            raise TypeError("procedural source bytes must be exact")
        _strict_digest(self.house_sha256, "procedural source SHA-256")
        if sha256(self.canonical_house_json).hexdigest() != self.house_sha256:
            raise ValueError("procedural source digest does not match bytes")
        if type(self.floor_xz_bounds) is not tuple or len(self.floor_xz_bounds) != 4:
            raise TypeError("procedural source bounds must be an exact tuple")
        bounds = tuple(
            _strict_float(value, "procedural source floor bound")
            for value in self.floor_xz_bounds
        )
        if not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
            raise ValueError("procedural source bounds must have positive area")
        object.__setattr__(self, "floor_xz_bounds", bounds)
        self.decode_house()

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
        house: dict,
    ) -> Self:
        if type(house) is not dict:
            raise TypeError("procedural source house must be an exact dictionary")
        encoded = (
            json.dumps(
                house,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        rooms = house.get("rooms")
        if type(rooms) is not list or len(rooms) != 1 or type(rooms[0]) is not dict:
            raise ValueError("procedural source must contain one room")
        room = rooms[0]
        room_id = room.get("id")
        polygon = room.get("floorPolygon")
        if type(room_id) is not str or type(polygon) is not list or len(polygon) < 3:
            raise ValueError("procedural source room is invalid")
        try:
            xs = tuple(float(point["x"]) for point in polygon)
            zs = tuple(float(point["z"]) for point in polygon)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("procedural source room polygon is invalid") from error
        return cls(
            dataset_id=dataset_id,
            revision=revision,
            split=split,
            index=index,
            source_loader_id=source_loader_id,
            source_loader_version=source_loader_version,
            canonical_house_json=encoded,
            house_sha256=sha256(encoded).hexdigest(),
            room_id=room_id,
            floor_xz_bounds=(min(xs), min(zs), max(xs), max(zs)),
        )

    def decode_house(self) -> dict:
        try:
            decoded = json.loads(self.canonical_house_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("procedural source bytes are not UTF-8 JSON") from error
        if type(decoded) is not dict:
            raise ValueError("procedural source JSON root must be a dictionary")
        return decoded


@dataclass(frozen=True)
class AdapterPosition:
    """One immutable adapter-coordinate position carried without native objects."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _strict_float(self.x, "adapter position x"))
        object.__setattr__(self, "y", _strict_float(self.y, "adapter position y"))
        object.__setattr__(self, "z", _strict_float(self.z, "adapter position z"))


@dataclass(frozen=True)
class AdapterPose:
    position: AdapterPosition
    yaw_degrees: float
    horizon_degrees: float
    standing: bool

    def __post_init__(self) -> None:
        if type(self.position) is not AdapterPosition:
            raise TypeError("adapter pose position must be exact")
        object.__setattr__(
            self, "yaw_degrees", _strict_float(self.yaw_degrees, "adapter yaw")
        )
        object.__setattr__(
            self,
            "horizon_degrees",
            _strict_float(self.horizon_degrees, "adapter camera horizon"),
        )
        if type(self.standing) is not bool:
            raise TypeError("adapter pose standing must be an exact boolean")


@dataclass(frozen=True)
class AdapterRuntimeProvenance:
    dataset_id: str
    revision: str
    split: str
    index: int
    source_sha256: str
    scene_alias: str
    loader_id: str
    loader_version: str
    room_id: str
    floor_xz_bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("revision", self.revision),
            ("split", self.split),
            ("source_sha256", self.source_sha256),
            ("scene_alias", self.scene_alias),
            ("loader_id", self.loader_id),
            ("loader_version", self.loader_version),
            ("room_id", self.room_id),
        ):
            _strict_text(value, f"runtime provenance {name}")
        _strict_digest(self.source_sha256, "runtime provenance source_sha256")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("runtime provenance index must be non-negative")
        if type(self.floor_xz_bounds) is not tuple or len(self.floor_xz_bounds) != 4:
            raise TypeError("runtime provenance floor bounds must be an exact tuple")
        bounds = tuple(
            _strict_float(value, "runtime provenance floor bound")
            for value in self.floor_xz_bounds
        )
        if not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
            raise ValueError("runtime provenance floor bounds must be positive")
        object.__setattr__(self, "floor_xz_bounds", bounds)


@dataclass(frozen=True)
class AdapterRuntimeIdentity:
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
        for name, value in (
            ("ai2thor_version", self.ai2thor_version),
            ("unity_commit_id", self.unity_commit_id),
            ("native_scene_name", self.native_scene_name),
            ("coordinate_transform_version", self.coordinate_transform_version),
        ):
            _strict_text(value, f"runtime identity {name}")
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("seed", self.seed),
            ("rotate_step_degrees", self.rotate_step_degrees),
        ):
            if type(value) is not int:
                raise TypeError(f"runtime identity {name} must be an exact integer")
        if self.width <= 0 or self.height <= 0 or self.rotate_step_degrees <= 0:
            raise ValueError("runtime identity dimensions must be positive")
        for name, value in (
            ("render_depth_image", self.render_depth_image),
            ("render_instance_segmentation", self.render_instance_segmentation),
            ("snap_to_grid", self.snap_to_grid),
        ):
            if type(value) is not bool:
                raise TypeError(f"runtime identity {name} must be an exact boolean")
        object.__setattr__(
            self, "grid_size_m", _strict_float(self.grid_size_m, "runtime grid size")
        )
        object.__setattr__(
            self,
            "teleport_vertical_guard_m",
            _strict_float(self.teleport_vertical_guard_m, "runtime teleport guard"),
        )
        provenance_values = (
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
        if any(value is not None for value in provenance_values):
            if any(value is None for value in provenance_values):
                raise ValueError("runtime identity provenance must be complete")
            AdapterRuntimeProvenance(
                dataset_id=self.source_dataset_id,
                revision=self.source_revision,
                split=self.source_split,
                index=self.source_index,
                source_sha256=self.source_sha256,
                scene_alias=self.source_scene_alias,
                loader_id=self.source_loader_id,
                loader_version=self.source_loader_version,
                room_id=self.source_room_id,
                floor_xz_bounds=self.source_floor_xz_bounds,
            )


@dataclass(frozen=True)
class AdapterObservation:
    scene: Scene
    rgb_png: bytes
    depth_npy: bytes
    instance_png: bytes
    pointcloud_ply: bytes
    rgb_png_sha256: str
    depth_npy_sha256: str
    instance_png_sha256: str
    pointcloud_ply_sha256: str
    instance_pixel_counts: tuple[tuple[str, int], ...]
    is_settled: bool

    @classmethod
    def create(
        cls,
        *,
        scene: Scene,
        rgb_png: bytes,
        depth_npy: bytes,
        instance_png: bytes,
        pointcloud_ply: bytes,
        instance_pixel_counts: tuple[tuple[str, int], ...],
        is_settled: bool,
    ) -> Self:
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
            instance_pixel_counts=instance_pixel_counts,
            is_settled=is_settled,
        )

    def __post_init__(self) -> None:
        if type(self.scene) is not Scene:
            raise TypeError("adapter observation scene must be exact")
        for name, value in (
            ("rgb_png", self.rgb_png),
            ("depth_npy", self.depth_npy),
            ("instance_png", self.instance_png),
            ("pointcloud_ply", self.pointcloud_ply),
        ):
            if type(value) is not bytes:
                raise TypeError(f"adapter observation {name} must be exact bytes")
        for name, value in (
            ("rgb_png_sha256", self.rgb_png_sha256),
            ("depth_npy_sha256", self.depth_npy_sha256),
            ("instance_png_sha256", self.instance_png_sha256),
            ("pointcloud_ply_sha256", self.pointcloud_ply_sha256),
        ):
            _strict_digest(value, f"adapter observation {name}")
        expected = (
            sha256(self.rgb_png).hexdigest(),
            sha256(self.depth_npy).hexdigest(),
            sha256(self.instance_png).hexdigest(),
            sha256(self.pointcloud_ply).hexdigest(),
        )
        if expected != (
            self.rgb_png_sha256,
            self.depth_npy_sha256,
            self.instance_png_sha256,
            self.pointcloud_ply_sha256,
        ):
            raise ValueError("adapter observation asset digests do not match bytes")
        if (
            type(self.instance_pixel_counts) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.instance_pixel_counts
            )
            or self.instance_pixel_counts != tuple(sorted(self.instance_pixel_counts))
            or len({item[0] for item in self.instance_pixel_counts})
            != len(self.instance_pixel_counts)
        ):
            raise TypeError(
                "adapter observation pixel counts must be sorted exact pairs"
            )
        if type(self.is_settled) is not bool:
            raise TypeError("adapter observation settled flag must be exact")


@dataclass(frozen=True)
class AdapterSupportFact:
    scene_id: str
    object_id: str
    object_name: str
    native_object_id: str
    raw_parent_object_ids: tuple[str, ...]
    structural_parent_object_ids: tuple[str, ...]
    domain_parent_object_ids: tuple[str, ...]
    support_kind: str
    support_object_id: str | None
    floor_object_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("scene_id", self.scene_id),
            ("object_id", self.object_id),
            ("object_name", self.object_name),
            ("native_object_id", self.native_object_id),
            ("support_kind", self.support_kind),
        ):
            _strict_text(value, f"adapter support {name}")
        for name, values in (
            ("raw_parent_object_ids", self.raw_parent_object_ids),
            ("structural_parent_object_ids", self.structural_parent_object_ids),
            ("domain_parent_object_ids", self.domain_parent_object_ids),
        ):
            if (
                type(values) is not tuple
                or any(type(value) is not str or not value for value in values)
                or values != tuple(sorted(set(values)))
            ):
                raise TypeError(f"adapter support {name} must be sorted exact strings")
        for name, value in (
            ("support_object_id", self.support_object_id),
            ("floor_object_id", self.floor_object_id),
        ):
            if value is not None:
                _strict_text(value, f"adapter support {name}")


@dataclass(frozen=True)
class AdapterFloorEnvelope:
    scene_id: str
    floor_object_id: str
    floor_name: str
    native_aabb: OBB
    floor_top_z: float
    clearance_m: float
    polygon_xy: tuple[Vec2, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("scene_id", self.scene_id),
            ("floor_object_id", self.floor_object_id),
            ("floor_name", self.floor_name),
        ):
            _strict_text(value, f"adapter floor {name}")
        if type(self.native_aabb) is not OBB:
            raise TypeError("adapter floor native_aabb must be exact")
        object.__setattr__(
            self, "floor_top_z", _strict_float(self.floor_top_z, "adapter floor top")
        )
        object.__setattr__(
            self,
            "clearance_m",
            _strict_float(self.clearance_m, "adapter floor clearance"),
        )
        if (
            type(self.polygon_xy) is not tuple
            or len(self.polygon_xy) < 3
            or any(type(item) is not Vec2 for item in self.polygon_xy)
        ):
            raise TypeError("adapter floor polygon must be exact canonical points")


@dataclass(frozen=True)
class AdapterSurfacePatch:
    x_min: float
    x_max: float
    native_y: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x_min", self.x_min),
            ("x_max", self.x_max),
            ("native_y", self.native_y),
            ("z_min", self.z_min),
            ("z_max", self.z_max),
        ):
            object.__setattr__(
                self, name, _strict_float(value, f"adapter surface {name}")
            )
        if self.x_min >= self.x_max or self.z_min >= self.z_max:
            raise ValueError("adapter surface patch must have positive area")


@dataclass(frozen=True)
class AdapterSpawnMap:
    binding: AdapterBinding
    runtime_identity: AdapterRuntimeIdentity
    scene_id: str
    subject_object_id: str
    support_object_id: str
    native_subject_object_id: str
    native_support_object_id: str
    positions: tuple[AdapterPosition, ...]
    positions_sha256: str
    scene_sha256: str
    source_sha256: str
    surface_patches: tuple[AdapterSurfacePatch, ...]
    position_region: SubjectPositionRegion | None = None

    def __post_init__(self) -> None:
        if type(self.binding) is not AdapterBinding:
            raise TypeError("adapter spawn map binding must be exact")
        if type(self.runtime_identity) is not AdapterRuntimeIdentity:
            raise TypeError("adapter spawn map runtime identity must be exact")
        for name, value in (
            ("scene_id", self.scene_id),
            ("subject_object_id", self.subject_object_id),
            ("support_object_id", self.support_object_id),
            ("native_subject_object_id", self.native_subject_object_id),
            ("native_support_object_id", self.native_support_object_id),
        ):
            _strict_text(value, f"adapter spawn map {name}")
        if self.scene_id != self.binding.scene_id:
            raise ValueError("adapter spawn map does not retain its source binding")
        if (
            type(self.positions) is not tuple
            or any(type(item) is not AdapterPosition for item in self.positions)
            or self.positions
            != tuple(sorted(self.positions, key=lambda item: (item.x, item.z, item.y)))
        ):
            raise TypeError("adapter spawn map positions must be sorted exact values")
        for name, value in (
            ("positions_sha256", self.positions_sha256),
            ("scene_sha256", self.scene_sha256),
            ("source_sha256", self.source_sha256),
        ):
            _strict_digest(value, f"adapter spawn map {name}")
        if type(self.surface_patches) is not tuple or any(
            type(item) is not AdapterSurfacePatch for item in self.surface_patches
        ):
            raise TypeError("adapter spawn map patches must be exact")
        if (
            self.position_region is not None
            and type(self.position_region) is not SubjectPositionRegion
        ):
            raise TypeError("adapter spawn map position region must be canonical")


@dataclass(frozen=True)
class SourceCaptureFacts:
    source: CapturedSource
    binding: AdapterBinding
    scene: Scene
    runtime_identity: AdapterRuntimeIdentity
    observation: AdapterObservation
    support_facts: tuple[AdapterSupportFact, ...]
    floor_envelope: AdapterFloorEnvelope | None
    floor_position_regions: tuple[tuple[str, SubjectPositionRegion], ...]
    reachable_positions: tuple[AdapterPosition, ...]
    current_pose: AdapterPose
    settlement_pass_steps: int

    def __post_init__(self) -> None:
        if (
            type(self.source) is not CapturedSource
            or type(self.binding) is not AdapterBinding
        ):
            raise TypeError("source capture facts must retain exact source and binding")
        if self.binding != self.source.binding:
            raise ValueError("source capture facts binding does not match source")
        if (
            type(self.scene) is not Scene
            or type(self.runtime_identity) is not AdapterRuntimeIdentity
        ):
            raise TypeError("source capture facts require exact canonical values")
        if (
            type(self.observation) is not AdapterObservation
            or self.observation.scene != self.scene
        ):
            raise ValueError("source capture facts observation does not bind scene")
        if not self.observation.is_settled:
            raise ValueError("source capture facts require a settled observation")
        if (
            type(self.support_facts) is not tuple
            or any(type(item) is not AdapterSupportFact for item in self.support_facts)
            or tuple(item.object_id for item in self.support_facts)
            != tuple(item.object_id for item in self.scene.objects)
        ):
            raise ValueError("source capture facts support roster does not bind scene")
        if (
            self.floor_envelope is not None
            and type(self.floor_envelope) is not AdapterFloorEnvelope
        ):
            raise TypeError("source capture facts floor must be exact when present")
        if self.scene.scene_id != self.binding.scene_id:
            raise ValueError(
                "source capture facts scene does not retain source binding"
            )
        if (
            self.floor_envelope is not None
            and self.floor_envelope.scene_id != self.scene.scene_id
        ):
            raise ValueError("source capture facts floor does not bind observed scene")
        if (
            type(self.floor_position_regions) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not SubjectPositionRegion
                for item in self.floor_position_regions
            )
            or self.floor_position_regions
            != tuple(sorted(self.floor_position_regions, key=lambda item: item[0]))
        ):
            raise TypeError(
                "source capture facts floor regions must be sorted exact pairs"
            )
        if any(item.scene_id != self.scene.scene_id for item in self.support_facts):
            raise ValueError(
                "source capture facts support values do not bind observed scene"
            )
        if (
            type(self.reachable_positions) is not tuple
            or any(
                type(item) is not AdapterPosition for item in self.reachable_positions
            )
            or self.reachable_positions
            != tuple(
                sorted(
                    self.reachable_positions, key=lambda item: (item.x, item.z, item.y)
                )
            )
        ):
            raise TypeError("source capture facts reachable positions must be sorted")
        if type(self.current_pose) is not AdapterPose:
            raise TypeError("source capture facts current pose must be exact")
        if (
            type(self.settlement_pass_steps) is not int
            or self.settlement_pass_steps < 0
        ):
            raise ValueError(
                "source capture facts settlement pass count must be non-negative"
            )


@dataclass(frozen=True)
class CameraObservationHandle:
    source: CapturedSource
    binding: AdapterBinding
    scene: Scene
    token: str
    settle_after_resume: bool

    def __post_init__(self) -> None:
        if (
            type(self.source) is not CapturedSource
            or type(self.binding) is not AdapterBinding
        ):
            raise TypeError("camera handle source and binding must be exact")
        if type(self.scene) is not Scene:
            raise TypeError("camera handle scene must be exact")
        if (
            self.binding != self.source.binding
            or self.scene.scene_id != self.binding.scene_id
        ):
            raise ValueError("camera handle does not bind its source")
        _strict_text(self.token, "camera handle token")
        if type(self.settle_after_resume) is not bool:
            raise TypeError("camera handle settle_after_resume must be exact")


@dataclass(frozen=True)
class AdapterCameraApplication:
    source: CapturedSource
    binding: AdapterBinding
    requested_pose: AdapterPose
    observed_pose: AdapterPose
    observed_camera_position: AdapterPosition
    observed_scene: Scene
    observation: AdapterObservation
    position_residual_m: float
    yaw_residual_degrees: float
    horizon_residual_degrees: float

    def __post_init__(self) -> None:
        if (
            type(self.source) is not CapturedSource
            or type(self.binding) is not AdapterBinding
        ):
            raise TypeError("camera application source and binding must be exact")
        if self.binding != self.source.binding:
            raise ValueError("camera application does not retain source binding")
        for name, value in (
            ("requested_pose", self.requested_pose),
            ("observed_pose", self.observed_pose),
        ):
            if type(value) is not AdapterPose:
                raise TypeError(f"camera application {name} must be exact")
        if type(self.observed_camera_position) is not AdapterPosition:
            raise TypeError("camera application observed position must be exact")
        if (
            type(self.observed_scene) is not Scene
            or type(self.observation) is not AdapterObservation
        ):
            raise TypeError("camera application observed facts must be exact")
        if self.observation.scene != self.observed_scene:
            raise ValueError(
                "camera application observation does not bind observed scene"
            )
        self.observed_scene.camera_by_id(self.source.request.camera_id)
        for name, raw_value in (
            ("position_residual_m", self.position_residual_m),
            ("yaw_residual_degrees", self.yaw_residual_degrees),
            ("horizon_residual_degrees", self.horizon_residual_degrees),
        ):
            value = _strict_float(raw_value, f"camera application {name}")
            if value < 0.0:
                raise ValueError(f"camera application {name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AdapterSettledCameraApplication:
    application: AdapterCameraApplication
    settlement_pass_steps: int

    def __post_init__(self) -> None:
        if type(self.application) is not AdapterCameraApplication:
            raise TypeError("settled camera application must be exact")
        if (
            type(self.settlement_pass_steps) is not int
            or self.settlement_pass_steps < 0
        ):
            raise ValueError("settled camera pass count must be non-negative")


@runtime_checkable
class XYSceneTransformer(Protocol):
    """Create a canonical scene with one object's XY position changed."""

    def with_object_xy(
        self, scene: Scene, object_id: str, x: float, y: float
    ) -> Scene: ...


@runtime_checkable
class SceneAdapter(XYSceneTransformer, Protocol):
    def list_scene_ids(self) -> list[str]: ...

    def load_scene(self, scene_id: str) -> Scene: ...

    def render_assets(
        self,
        scene: Scene,
        camera_id: str,
        destination_root: Path,
        stem: str,
    ) -> RenderedAssets: ...


@runtime_checkable
class EnvironmentAdapter(SceneAdapter, Protocol):
    """Lifecycle and Canonical readback contract for platform boundaries."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def capture_source(self, request: CaptureRequest) -> CapturedSource: ...

    def capture_support(self, source: CapturedSource) -> CapturedSupport: ...

    def apply_certified_edit(
        self,
        application: CertifiedEditApplication,
    ) -> AppliedCertifiedEdit: ...

    def settle_readback(self, applied: AppliedCertifiedEdit) -> SettledReadback: ...

    def observe_source(
        self,
        source: CapturedSource,
        *,
        options: SourceCaptureOptions,
        settle: bool,
    ) -> SourceCaptureFacts: ...

    def capture_spawn_maps(
        self,
        facts: SourceCaptureFacts,
        *,
        subject_object_ids: tuple[str, ...],
    ) -> tuple[AdapterSpawnMap, ...]: ...

    def pause_camera_observations(
        self,
        facts: SourceCaptureFacts,
        *,
        settle_after_resume: bool,
    ) -> CameraObservationHandle: ...

    def resume_camera_observations(self, handle: CameraObservationHandle) -> None: ...

    def apply_camera_pose(
        self,
        facts: SourceCaptureFacts,
        pose: AdapterPose,
        *,
        handle: CameraObservationHandle | None,
        source_scene: Scene,
        reset_from_source: bool,
        max_settlement_steps: int,
    ) -> AdapterCameraApplication: ...

    def settle_camera_pose(
        self,
        facts: SourceCaptureFacts,
        pose: AdapterPose,
        *,
        source_scene: Scene,
        max_settlement_steps: int,
    ) -> AdapterSettledCameraApplication: ...


def capture_source_with_scene_adapter(
    adapter: SceneAdapter,
    request: CaptureRequest,
) -> CapturedSource:
    scene = adapter.load_scene(request.scene_id)
    return CapturedSource(
        request=request,
        scene=scene,
        binding=AdapterBinding(
            scene_id=scene.scene_id, token=f"scene:{scene.scene_id}"
        ),
    )


def capture_support_with_environment_adapter(
    adapter: EnvironmentAdapter,
    source: CapturedSource,
) -> CapturedSupport:
    del adapter
    return CapturedSupport(source=source, scene=source.scene, binding=source.binding)


def apply_certified_edit_with_scene_adapter(
    adapter: XYSceneTransformer,
    application: CertifiedEditApplication,
) -> AppliedCertifiedEdit:
    subject = application.source.scene.object_by_id(application.edit.subject_id)
    edit = application.edit
    scene = adapter.with_object_xy(
        application.source.scene,
        edit.subject_id,
        subject.position.x + edit.translation_xy_m.x,
        subject.position.y + edit.translation_xy_m.y,
    )
    commanded_position = AdapterPosition(
        x=scene.object_by_id(edit.subject_id).position.x,
        y=scene.object_by_id(edit.subject_id).position.y,
        z=scene.object_by_id(edit.subject_id).position.z,
    )
    return AppliedCertifiedEdit(
        application=application,
        edit=edit,
        commanded_scene=scene,
        observed_scene=scene,
        commanded_position=commanded_position,
        observed_position=commanded_position,
        position_residual_m=0.0,
        observation=_protocol_observation(scene, settled=False),
        is_scene_at_rest=False,
        subject_is_moving=True,
        settlement_pass_steps=0,
        binding=application.source.binding,
    )


def settle_readback_with_environment_adapter(
    adapter: EnvironmentAdapter,
    applied: AppliedCertifiedEdit,
) -> SettledReadback:
    del adapter
    return SettledReadback(
        applied=applied,
        commanded_scene=applied.commanded_scene,
        observed_scene=applied.observed_scene,
        commanded_position=applied.commanded_position,
        observed_position=applied.observed_position,
        position_residual_m=applied.position_residual_m,
        observation=_protocol_observation(applied.observed_scene, settled=True),
        is_scene_at_rest=True,
        subject_is_moving=False,
        settlement_pass_steps=0,
        binding=applied.binding,
    )


def _protocol_observation(scene: Scene, *, settled: bool) -> AdapterObservation:
    """Use deterministic opaque bytes for the Canonical-only fallback adapter."""
    prefix = scene.scene_id.encode("utf-8")
    return AdapterObservation.create(
        scene=scene,
        rgb_png=b"protocol-rgb:" + prefix,
        depth_npy=b"protocol-depth:" + prefix,
        instance_png=b"protocol-instance:" + prefix,
        pointcloud_ply=b"protocol-ply:" + prefix,
        instance_pixel_counts=(),
        is_settled=settled,
    )


def capture_bound_adapter_spawn_map(
    spawn_map: AdapterSpawnMap,
    *,
    fresh_scene: Scene,
    frozen_scene: Scene,
) -> AdapterSpawnMap:
    """Rebind source facts without exposing a concrete adapter representation."""
    if type(spawn_map) is not AdapterSpawnMap:
        raise TypeError("capture-bound spawn map must be exact")
    if type(fresh_scene) is not Scene or type(frozen_scene) is not Scene:
        raise TypeError("capture-bound spawn scenes must be exact Scene values")
    expected_fresh_scene_sha256 = _adapter_spawn_map_scene_sha256(
        fresh_scene,
        spawn_map.surface_patches,
    )
    if (
        spawn_map.scene_id != fresh_scene.scene_id
        or spawn_map.scene_sha256 != expected_fresh_scene_sha256
        or frozen_scene.scene_id != fresh_scene.scene_id
    ):
        raise ValueError("capture-bound spawn map does not bind the fresh scene")
    frozen_scene_sha256 = _adapter_spawn_map_scene_sha256(
        frozen_scene,
        spawn_map.surface_patches,
    )
    source_sha256 = _adapter_spawn_map_source_sha256(
        scene_id=spawn_map.scene_id,
        subject_object_id=spawn_map.subject_object_id,
        support_object_id=spawn_map.support_object_id,
        native_subject_object_id=spawn_map.native_subject_object_id,
        native_support_object_id=spawn_map.native_support_object_id,
        runtime_identity=spawn_map.runtime_identity,
        positions_sha256=spawn_map.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        surface_patches=spawn_map.surface_patches,
    )
    return AdapterSpawnMap(
        binding=spawn_map.binding,
        runtime_identity=spawn_map.runtime_identity,
        scene_id=spawn_map.scene_id,
        subject_object_id=spawn_map.subject_object_id,
        support_object_id=spawn_map.support_object_id,
        native_subject_object_id=spawn_map.native_subject_object_id,
        native_support_object_id=spawn_map.native_support_object_id,
        positions=spawn_map.positions,
        positions_sha256=spawn_map.positions_sha256,
        scene_sha256=frozen_scene_sha256,
        source_sha256=source_sha256,
        surface_patches=spawn_map.surface_patches,
        position_region=spawn_map.position_region,
    )


def _adapter_spawn_map_scene_sha256(
    scene: Scene,
    surface_patches: tuple[AdapterSurfacePatch, ...],
) -> str:
    if surface_patches:
        normalized = scene.model_copy(
            update={
                "cameras": tuple(
                    sorted(scene.cameras, key=lambda item: item.camera_id)
                ),
                "objects": tuple(
                    sorted(scene.objects, key=lambda item: item.object_id)
                ),
                "collision_obstacles": tuple(
                    sorted(
                        scene.collision_obstacles,
                        key=lambda item: item.obstacle_id,
                    )
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
        return sha256(canonical_json_bytes(payload)).hexdigest()
    payload = json.dumps(
        scene.model_dump(mode="json", warnings="error"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _adapter_spawn_map_source_sha256(
    *,
    scene_id: str,
    subject_object_id: str,
    support_object_id: str,
    native_subject_object_id: str,
    native_support_object_id: str,
    runtime_identity: AdapterRuntimeIdentity,
    positions_sha256: str,
    scene_sha256: str,
    surface_patches: tuple[AdapterSurfacePatch, ...],
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
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def move_object_xy(scene: Scene, object_id: str, x: float, y: float) -> Scene:
    """Apply the protocol's sole local operation without a platform runtime."""

    target = scene.object_by_id(object_id)
    if not target.movable:
        raise ValueError("protocol edits require a movable subject")
    delta_x = x - target.position.x
    delta_y = y - target.position.y
    objects = tuple(
        object_
        if object_.object_id != object_id
        else object_.model_copy(
            update={
                "position": Vec3(x=x, y=y, z=object_.position.z),
                "obb": OBB(
                    center=Vec3(
                        x=object_.obb.center.x + delta_x,
                        y=object_.obb.center.y + delta_y,
                        z=object_.obb.center.z,
                    ),
                    extent=object_.obb.extent,
                    rotation=object_.obb.rotation,
                ),
            }
        )
        for object_ in scene.objects
    )
    return scene.model_copy(update={"objects": objects})


__all__ = (
    "AdapterActionRejected",
    "AdapterBinding",
    "AdapterCameraApplication",
    "AdapterFloorEnvelope",
    "AdapterObservation",
    "AdapterOperationError",
    "AdapterPose",
    "AdapterPosition",
    "AdapterProceduralScene",
    "AdapterReturnRejected",
    "AdapterRuntimeIdentity",
    "AdapterRuntimeProvenance",
    "AdapterSettledCameraApplication",
    "AdapterSettlementTimeout",
    "AdapterSpawnMap",
    "AdapterSupportFact",
    "AppliedCertifiedEdit",
    "CameraObservationHandle",
    "CaptureRequest",
    "CapturedSource",
    "CapturedSupport",
    "CertifiedEditApplication",
    "EnvironmentAdapter",
    "RenderedAssets",
    "SceneAdapter",
    "SettledReadback",
    "SourceCaptureFacts",
    "SourceCaptureOptions",
    "XYSceneTransformer",
    "capture_bound_adapter_spawn_map",
)
