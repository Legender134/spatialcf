import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from spatialcf.domain.enums import Relation


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Vec2(FrozenModel):
    x: float
    y: float


class Vec3(FrozenModel):
    x: float
    y: float
    z: float


class Quaternion(FrozenModel):
    x: float
    y: float
    z: float
    w: float


class BBox2D(FrozenModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


class ObjectView(FrozenModel):
    camera_id: str
    bbox: BBox2D
    camera_depth: float
    visible_fraction: float = Field(ge=0.0, le=1.0)
    image_area_fraction: float = Field(ge=0.0, le=1.0)
    truncated_fraction: float = Field(ge=0.0, le=1.0)


class OBB(FrozenModel):
    center: Vec3
    extent: Vec3
    rotation: Quaternion


class CollisionObstacle(FrozenModel):
    """Native geometry plus solver-only clearance for collision checks."""

    obstacle_id: str = Field(min_length=1)
    source_object_id: str = Field(min_length=1)
    clearance_m: float = Field(ge=0.0, strict=True)
    obb: OBB

    def conservative_obb(self) -> OBB:
        clearance_diameter = 2.0 * self.clearance_m
        return self.obb.model_copy(
            update={
                "extent": Vec3(
                    x=self.obb.extent.x + clearance_diameter,
                    y=self.obb.extent.y + clearance_diameter,
                    z=self.obb.extent.z + clearance_diameter,
                )
            }
        )


class PlanarPolygon(FrozenModel):
    """One canonical polygon component without a repeated closing vertex."""

    exterior: tuple[Vec2, ...] = Field(min_length=3)
    holes: tuple[tuple[Vec2, ...], ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_rings(self) -> "PlanarPolygon":
        for label, ring in (
            ("exterior", self.exterior),
            *((f"hole {index}", hole) for index, hole in enumerate(self.holes)),
        ):
            if len(ring) < 3:
                raise ValueError(f"{label} must contain at least three vertices")
            coordinates = tuple((point.x, point.y) for point in ring)
            if any(
                not math.isfinite(value)
                for coordinate in coordinates
                for value in coordinate
            ):
                raise ValueError(f"{label} vertices must be finite")
            if len(set(coordinates)) < 3:
                raise ValueError(f"{label} must contain three distinct vertices")
            if coordinates[0] == coordinates[-1]:
                raise ValueError(f"{label} must omit the repeated closing vertex")
        return self


class SubjectPositionRegion(FrozenModel):
    """A source-bound allowed XY position-anchor locus for one subject."""

    region_id: str = Field(min_length=1)
    subject_object_id: str = Field(min_length=1)
    source_kind: Literal[
        "ai2thor-navigation-v1",
        "ai2thor-receptacle-trigger-grid-v1",
    ]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    components: tuple[PlanarPolygon, ...] = Field(default_factory=tuple)


class SceneObject(FrozenModel):
    object_id: str
    name: str
    category: str
    movable: bool
    request_eligible: bool = Field(default=True, strict=True)
    position: Vec3
    rotation: Quaternion
    obb: OBB
    support_object_id: str | None = None
    views: dict[str, ObjectView] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def serialize_request_eligibility(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if "request_eligible" not in self.model_fields_set:
            payload.pop("request_eligible", None)
        return payload


class Camera(FrozenModel):
    camera_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    intrinsics: tuple[float, ...] = Field(min_length=9, max_length=9)
    world_to_camera: tuple[float, ...] = Field(min_length=16, max_length=16)


class Scene(FrozenModel):
    scene_id: str
    source: str
    coordinate_system: Literal["RH_METERS_Z_UP"] = "RH_METERS_Z_UP"
    room_polygon_xy: tuple[Vec2, ...] = Field(min_length=3)
    cameras: tuple[Camera, ...]
    objects: tuple[SceneObject, ...]
    collision_obstacles: tuple[CollisionObstacle, ...] = Field(default_factory=tuple)
    subject_position_regions: tuple[SubjectPositionRegion, ...] = Field(
        default_factory=tuple
    )
    pinned_object_ids: frozenset[str] = Field(default_factory=frozenset)
    generation_seed: int

    @model_validator(mode="after")
    def validate_subject_position_regions(self) -> "Scene":
        object_ids = {obj.object_id for obj in self.objects}
        region_ids = tuple(region.region_id for region in self.subject_position_regions)
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("subject position region IDs must be unique")
        if any(
            region.subject_object_id not in object_ids
            for region in self.subject_position_regions
        ):
            raise ValueError("subject position region references an unknown object")
        return self

    @model_serializer(mode="wrap")
    def serialize_analysis_overlays(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if "collision_obstacles" not in self.model_fields_set:
            payload.pop("collision_obstacles", None)
        if "subject_position_regions" not in self.model_fields_set:
            payload.pop("subject_position_regions", None)
        return payload

    def object_by_id(self, object_id: str) -> SceneObject:
        matches = [obj for obj in self.objects if obj.object_id == object_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one object {object_id!r}, found {len(matches)}")
        return matches[0]

    def camera_by_id(self, camera_id: str) -> Camera:
        matches = [cam for cam in self.cameras if cam.camera_id == camera_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one camera {camera_id!r}, found {len(matches)}")
        return matches[0]

    def children_by_support(self) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = {}
        for obj in self.objects:
            if obj.support_object_id is not None:
                children.setdefault(obj.support_object_id, []).append(obj.object_id)
        return {
            support_id: tuple(sorted(object_ids))
            for support_id, object_ids in sorted(children.items())
        }


class InterventionSpec(FrozenModel):
    subject_id: str
    reference_id: str
    relation_before: Relation
    relation_after: Relation
    camera_id: str

    @model_validator(mode="after")
    def validate_flip(self) -> "InterventionSpec":
        if self.subject_id == self.reference_id:
            raise ValueError("subject and reference must differ")
        if self.relation_before == self.relation_after:
            raise ValueError("counterfactual must change the relation")
        if self.relation_before.axis is not self.relation_after.axis:
            raise ValueError("counterfactual relations must share one relation axis")
        if self.relation_before.opposite != self.relation_after:
            raise ValueError("MVP supports opposite relation flips only")
        return self
