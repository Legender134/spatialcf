"""Version-free scene values and Canonical semantic scene facts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictBool, model_serializer, model_validator
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    FiniteFloat,
    PositiveFiniteFloat,
    RigidTransformV2,
    SchemaIdentityV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.base import (
    Vec3 as CanonicalVec3,
)
from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    PlanarRegionV2,
)


class FrozenModel(CanonicalModel):
    """Legacy value-model behavior retained inside the current domain owner."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        allow_inf_nan=True,
        validate_default=False,
        revalidate_instances="never",
    )


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
    def validate_rings(self) -> PlanarPolygon:
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
    def validate_subject_position_regions(self) -> Scene:
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


_NormalizedFraction = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]
_PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]


class RegionBoundaryPolicy(StrEnum):
    """Whether a geometric fact includes its mathematical boundary."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"


class CameraMatrixLayout(StrEnum):
    ROW_MAJOR = "ROW_MAJOR"


class CameraAxes(StrEnum):
    X_RIGHT_Y_DOWN_Z_FORWARD = "X_RIGHT_Y_DOWN_Z_FORWARD"


class CameraPixelConvention(StrEnum):
    CENTER_AT_HALF = "CENTER_AT_HALF"


class CameraDepthConvention(StrEnum):
    POSITIVE_Z_FORWARD = "POSITIVE_Z_FORWARD"


class CameraDistortionModel(StrEnum):
    NONE = "NONE"
    BROWN_CONRADY = "BROWN_CONRADY"


class BrownConradyCoefficients(CanonicalModel):
    """Named coefficients for the normalized-coordinate Brown-Conrady model.

    The frozen order semantics are radial ``k1, k2, k3`` and tangential
    ``p1, p2`` in ``x_d = x(1+k1*r2+k2*r4+k3*r6)+2*p1*x*y+p2*(r2+2*x2)``
    and its corresponding Y equation.
    """

    k1: FiniteFloat
    k2: FiniteFloat
    p1: FiniteFloat
    p2: FiniteFloat
    k3: FiniteFloat


class ObjectPose(CanonicalModel):
    """The world pose of the object's unique pivot/edit anchor."""

    anchor_kind: Literal["OBJECT_PIVOT"] = "OBJECT_PIVOT"
    world_from_object: RigidTransformV2


class ObjectSupportAssignment(CanonicalModel):
    """Availability-aware baseline support assignment for one object."""

    availability: FactAvailabilityV2
    surface_id: CanonicalId | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.availability is FactAvailabilityV2.KNOWN:
            if self.surface_id is None:
                raise ValueError("KNOWN support assignment requires surface_id")
        elif self.surface_id is not None:
            raise ValueError(
                f"{self.availability.value} support assignment must not carry surface_id"
            )
        return self

    @classmethod
    def known(cls, surface_id: CanonicalId) -> Self:
        return cls(availability=FactAvailabilityV2.KNOWN, surface_id=surface_id)

    @classmethod
    def missing(cls) -> Self:
        return cls(availability=FactAvailabilityV2.MISSING)

    @classmethod
    def not_applicable(cls) -> Self:
        return cls(availability=FactAvailabilityV2.NOT_APPLICABLE)


class CanonicalObject(CanonicalModel):
    """A stable semantic object identity without source aliases or provenance."""

    object_id: CanonicalId
    category_id: CanonicalId
    movable: StrictBool
    pose: ObjectPose
    support_assignment: ObjectSupportAssignment


class WorkspaceBoundaryFact(CanonicalModel):
    """A claimed workspace extent; it is not itself an edit constraint."""

    fact_id: CanonicalId
    region_world_xy: PlanarRegionV2
    boundary_policy: RegionBoundaryPolicy
    region_approximation: GeometryApproximationV2
    geometry_uncertainty: UncertaintyBudgetV2


class KnownFreeSpaceFact(CanonicalModel):
    """A claimed known-free extent, distinct from workspace and subject domain."""

    fact_id: CanonicalId
    region_world_xy: PlanarRegionV2
    boundary_policy: RegionBoundaryPolicy
    region_approximation: GeometryApproximationV2
    geometry_uncertainty: UncertaintyBudgetV2


class PinholeCamera(CanonicalModel):
    """One fully specified calibrated pinhole camera convention."""

    camera_id: CanonicalId
    width_px: _PositiveStrictInt
    height_px: _PositiveStrictInt
    intrinsics_row_major: tuple[FiniteFloat, ...] = Field(
        min_length=9,
        max_length=9,
    )
    world_to_camera: RigidTransformV2
    matrix_layout: CameraMatrixLayout = CameraMatrixLayout.ROW_MAJOR
    camera_axes: CameraAxes = CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
    pixel_convention: CameraPixelConvention = CameraPixelConvention.CENTER_AT_HALF
    depth_convention: CameraDepthConvention = CameraDepthConvention.POSITIVE_Z_FORWARD
    near_clip_m: PositiveFiniteFloat
    far_clip_m: PositiveFiniteFloat
    distortion_model: CameraDistortionModel = CameraDistortionModel.NONE
    brown_conrady_coefficients: BrownConradyCoefficients | None = None
    calibration_uncertainty: UncertaintyBudgetV2

    @model_validator(mode="after")
    def validate_projection_contract(self) -> Self:
        intrinsics = self.intrinsics_row_major
        if intrinsics[0] <= 0.0 or intrinsics[4] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if intrinsics[6:] != (0.0, 0.0, 1.0):
            raise ValueError("camera intrinsics final row must be (0, 0, 1)")
        if self.far_clip_m <= self.near_clip_m:
            raise ValueError("far_clip_m must be greater than near_clip_m")
        has_coefficients = self.brown_conrady_coefficients is not None
        if has_coefficients is not (
            self.distortion_model is CameraDistortionModel.BROWN_CONRADY
        ):
            raise ValueError(
                "Brown-Conrady coefficients do not match the distortion model"
            )
        return self


class BaselineObservation(CanonicalModel):
    """One normalized, definition-bound object/camera metric interval."""

    observation_id: CanonicalId
    object_id: CanonicalId
    camera_id: CanonicalId
    metric_definition_id: CanonicalId
    metric_definition_version: CanonicalId
    normalized_value: _NormalizedFraction
    normalized_lower_bound: _NormalizedFraction
    normalized_upper_bound: _NormalizedFraction

    @model_validator(mode="after")
    def validate_normalized_interval(self) -> Self:
        if self.normalized_lower_bound > self.normalized_upper_bound:
            raise ValueError("normalized interval lower bound exceeds upper bound")
        if not (
            self.normalized_lower_bound
            <= self.normalized_value
            <= self.normalized_upper_bound
        ):
            raise ValueError("normalized value must lie inside its interval")
        return self


class SupportSurfaceFact(CanonicalModel):
    """A planar region on a collision body in an explicit anchor frame.

    ``anchor_from_surface`` maps surface coordinates into the owner object's
    frame when ``owner_object_id`` is present, and into the world frame for an
    environment body. ``normal_in_anchor`` is expressed in that same frame.
    """

    surface_id: CanonicalId
    owner_object_id: CanonicalId | None
    supporting_body_id: CanonicalId
    anchor_from_surface: RigidTransformV2
    normal_in_anchor: CanonicalVec3
    region_uv: PlanarRegionV2
    region_approximation: GeometryApproximationV2
    boundary_policy: RegionBoundaryPolicy
    geometry_uncertainty: UncertaintyBudgetV2

    @model_validator(mode="after")
    def validate_plane_and_normal(self) -> Self:
        normal = self.normal_in_anchor
        magnitude = math.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("support surface normal must be unit length")
        expected = _rotated_positive_z(self.anchor_from_surface)
        if any(
            not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12)
            for actual, target in zip(
                (normal.x, normal.y, normal.z),
                expected,
                strict=True,
            )
        ):
            raise ValueError(
                "support surface normal_in_anchor must match its surface frame +Z"
            )
        return self


class CanonicalScene(CanonicalModel):
    """Versioned platform-neutral facts with a canonical closed reference graph."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="canonical-scene")
    )
    scene_id: CanonicalId = Field(
        description=(
            "Platform-neutral stable semantic identity; never a native locator"
        )
    )
    coordinate_system: Literal["RH_METERS_Z_UP"] = "RH_METERS_Z_UP"
    objects: FactSetV2[CanonicalObject]
    geometry_instances: FactSetV2[GeometryInstanceV2]
    collision_bodies: FactSetV2[CollisionBodyFactV2]
    workspace_boundaries: FactSetV2[WorkspaceBoundaryFact]
    known_free_spaces: FactSetV2[KnownFreeSpaceFact]
    support_surfaces: FactSetV2[SupportSurfaceFact]
    cameras: FactSetV2[PinholeCamera]
    baseline_observations: FactSetV2[BaselineObservation]

    @model_validator(mode="after")
    def canonicalize_and_validate_scene(self) -> Self:
        if self.schema_identity != self._expected_schema_identity():
            raise ValueError("canonical scene schema identity must be fixed")
        if (
            self.objects.availability is not FactAvailabilityV2.KNOWN
            or self.objects.completeness is not FactCompletenessV2.EXACT
            or self.objects.values is None
        ):
            raise ValueError("objects must be a KNOWN EXACT fact set")

        keys: dict[str, tuple[str, ...]] = {
            "objects": ("object_id",),
            "geometry_instances": ("geometry_id",),
            "collision_bodies": ("body_id",),
            "workspace_boundaries": ("fact_id",),
            "known_free_spaces": ("fact_id",),
            "support_surfaces": ("surface_id",),
            "cameras": ("camera_id",),
            "baseline_observations": (
                "object_id",
                "camera_id",
                "metric_definition_id",
                "metric_definition_version",
                "observation_id",
            ),
        }
        for field_name, key_fields in keys.items():
            facts = getattr(self, field_name)
            canonical = _canonicalize_fact_ids(facts, field_name, key_fields)
            object.__setattr__(self, field_name, canonical)

        self._validate_reference_graph("inner")
        self._validate_reference_graph("outer")
        return self

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2:
        return SchemaIdentityV2(schema_name="canonical-scene")

    def _validate_reference_graph(self, branch: Literal["inner", "outer"]) -> None:
        objects = _by_id(self.objects, "object_id", branch)
        geometries = _by_id(self.geometry_instances, "geometry_id", branch)
        bodies = _by_id(self.collision_bodies, "body_id", branch)
        surfaces = _by_id(self.support_surfaces, "surface_id", branch)
        cameras = _by_id(self.cameras, "camera_id", branch)

        for geometry in _values_for_branch(self.geometry_instances, branch):
            if (
                geometry.owner_object_id is not None
                and geometry.owner_object_id not in objects
            ):
                raise ValueError(
                    f"geometry {geometry.geometry_id!r} references an unknown owner object"
                )
            if geometry.owner_object_id is not None:
                owner = objects[geometry.owner_object_id]
                if not _is_yaw_only(owner.pose.world_from_object):
                    raise ValueError(
                        "upright/extruded geometry requires a world upright object pose"
                    )

        used_collision_geometry: dict[str, str] = {}
        for body in _values_for_branch(self.collision_bodies, branch):
            if body.owner_object_id is not None and body.owner_object_id not in objects:
                raise ValueError(
                    f"collision body {body.body_id!r} references an unknown owner object"
                )
            for geometry_id in body.geometry_instance_ids:
                geometry = geometries.get(geometry_id)
                if geometry is None:
                    raise ValueError(
                        f"collision body {body.body_id!r} references unknown geometry"
                    )
                if geometry.role is not GeometryRoleV2.COLLISION:
                    raise ValueError(
                        "collision bodies must reference COLLISION geometry"
                    )
                if geometry.owner_object_id != body.owner_object_id:
                    raise ValueError(
                        "collision body and geometry owner object must match"
                    )
                previous = used_collision_geometry.setdefault(geometry_id, body.body_id)
                if previous != body.body_id:
                    raise ValueError(
                        "collision geometry must belong to exactly one collision body"
                    )
        collision_geometry_ids = {
            geometry.geometry_id
            for geometry in _values_for_branch(self.geometry_instances, branch)
            if geometry.role is GeometryRoleV2.COLLISION
        }
        orphan_ids = collision_geometry_ids.difference(used_collision_geometry)
        if orphan_ids:
            raise ValueError(
                "orphan COLLISION geometry must belong to exactly one collision body: "
                + ", ".join(sorted(orphan_ids))
            )

        for surface in _values_for_branch(self.support_surfaces, branch):
            if (
                surface.owner_object_id is not None
                and surface.owner_object_id not in objects
            ):
                raise ValueError(
                    f"support surface {surface.surface_id!r} has unknown owner object"
                )
            supporting_body = bodies.get(surface.supporting_body_id)
            if supporting_body is None:
                raise ValueError(
                    f"support surface {surface.surface_id!r} has unknown supporting body"
                )
            if supporting_body.owner_object_id != surface.owner_object_id:
                raise ValueError(
                    "support surface and supporting body owner object must match"
                )

        support_parent: dict[str, str] = {}
        for object_ in _values_for_branch(self.objects, branch):
            assignment = object_.support_assignment
            if assignment.availability is not FactAvailabilityV2.KNOWN:
                continue
            assert assignment.surface_id is not None
            surface = surfaces.get(assignment.surface_id)
            if surface is None:
                raise ValueError(
                    f"object {object_.object_id!r} references unknown support surface"
                )
            if surface.owner_object_id is not None:
                if surface.owner_object_id == object_.object_id:
                    raise ValueError("support cycle: an object cannot support itself")
                support_parent[object_.object_id] = surface.owner_object_id
        _reject_support_cycles(support_parent)

        for observation in _values_for_branch(self.baseline_observations, branch):
            if observation.object_id not in objects:
                raise ValueError(
                    f"baseline observation references unknown object {observation.object_id!r}"
                )
            if observation.camera_id not in cameras:
                raise ValueError(
                    f"baseline observation references unknown camera {observation.camera_id!r}"
                )


def _canonicalize_fact_ids(
    facts: FactSetV2,
    family_name: str,
    key_fields: tuple[str, ...],
) -> FactSetV2:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return facts
    updates: dict[str, tuple[CanonicalModel, ...]] = {}
    for value_field in ("values", "inner_values", "outer_values"):
        values = getattr(facts, value_field)
        if values is None:
            continue
        keys = tuple(
            tuple(getattr(value, key_field) for key_field in key_fields)
            for value in values
        )
        if len(set(keys)) != len(keys):
            label = family_name.replace("_", " ")
            raise ValueError(f"{label} IDs must be unique")
        identities = tuple(key[-1] for key in keys)
        if len(set(identities)) != len(identities):
            label = family_name.replace("_", " ")
            raise ValueError(f"{label} stable IDs must be unique")
        if family_name == "baseline_observations":
            semantic_keys = tuple(key[:-1] for key in keys)
            if len(set(semantic_keys)) != len(semantic_keys):
                raise ValueError(
                    "baseline observation object-camera-metric keys must be unique"
                )
        updates[value_field] = tuple(
            sorted(
                values,
                key=lambda value: tuple(
                    getattr(value, key_field) for key_field in key_fields
                ),
            )
        )
    return facts.model_copy(update=updates)


def _values_for_branch(
    facts: FactSetV2,
    branch: Literal["inner", "outer"],
) -> tuple[CanonicalModel, ...]:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return ()
    if facts.completeness is FactCompletenessV2.BRACKETED:
        assert facts.inner_values is not None
        assert facts.outer_values is not None
        return facts.inner_values if branch == "inner" else facts.outer_values
    assert facts.values is not None
    if facts.completeness is FactCompletenessV2.OUTER_BOUND and branch == "inner":
        return ()
    return facts.values


def _by_id(
    facts: FactSetV2,
    id_field: str,
    branch: Literal["inner", "outer"],
) -> dict[str, CanonicalModel]:
    return {getattr(item, id_field): item for item in _values_for_branch(facts, branch)}


def _reject_support_cycles(parent_by_child: dict[str, str]) -> None:
    for start in parent_by_child:
        seen: set[str] = set()
        current = start
        while current in parent_by_child:
            if current in seen:
                raise ValueError("support cycle detected")
            seen.add(current)
            current = parent_by_child[current]


def _rotated_positive_z(transform: object) -> tuple[float, float, float]:
    yaw_radians = getattr(transform, "yaw_radians", None)
    if (
        getattr(transform, "kind", None) == "DIRECTED_YAW_INTERVAL"
        and type(yaw_radians) is float
        and math.isfinite(yaw_radians)
    ):
        return 0.0, 0.0, 1.0
    quarter_turns = getattr(transform, "quarter_turns_ccw", None)
    if (
        getattr(transform, "kind", None) == "EXACT_CARDINAL_YAW"
        and type(quarter_turns) is int
        and quarter_turns in range(4)
    ):
        return 0.0, 0.0, 1.0
    if not isinstance(transform, RigidTransformV2):
        raise TypeError("unsupported support-surface transform")
    quaternion = transform.rotation
    return (
        2.0 * (quaternion.x * quaternion.z + quaternion.w * quaternion.y),
        2.0 * (quaternion.y * quaternion.z - quaternion.w * quaternion.x),
        1.0 - 2.0 * (quaternion.x**2 + quaternion.y**2),
    )


def _is_yaw_only(transform: object) -> bool:
    if isinstance(transform, RigidTransformV2):
        return math.isclose(
            transform.rotation.x,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) and math.isclose(
            transform.rotation.y,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    yaw_radians = getattr(transform, "yaw_radians", None)
    if (
        getattr(transform, "kind", None) == "DIRECTED_YAW_INTERVAL"
        and type(yaw_radians) is float
        and math.isfinite(yaw_radians)
    ):
        return True
    quarter_turns = getattr(transform, "quarter_turns_ccw", None)
    return (
        getattr(transform, "kind", None) == "EXACT_CARDINAL_YAW"
        and type(quarter_turns) is int
        and quarter_turns in range(4)
    )
