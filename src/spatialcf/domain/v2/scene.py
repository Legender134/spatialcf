"""Canonical v2 semantic scene facts and their closed reference graph."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    FiniteFloat,
    PositiveFiniteFloat,
    RigidTransformV2,
    SchemaIdentityV2,
    UncertaintyBudgetV2,
    V2Model,
    Vec3V2,
)
from spatialcf.domain.v2.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    PlanarRegionV2,
)

_NormalizedFraction = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]
_PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]


class RegionBoundaryPolicyV2(StrEnum):
    """Whether a geometric fact includes its mathematical boundary."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"


class CameraMatrixLayoutV2(StrEnum):
    ROW_MAJOR = "ROW_MAJOR"


class CameraAxesV2(StrEnum):
    X_RIGHT_Y_DOWN_Z_FORWARD = "X_RIGHT_Y_DOWN_Z_FORWARD"


class CameraPixelConventionV2(StrEnum):
    CENTER_AT_HALF = "CENTER_AT_HALF"


class CameraDepthConventionV2(StrEnum):
    POSITIVE_Z_FORWARD = "POSITIVE_Z_FORWARD"


class CameraDistortionModelV2(StrEnum):
    NONE = "NONE"
    BROWN_CONRADY = "BROWN_CONRADY"


class BrownConradyCoefficientsV2(V2Model):
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


class ObjectPoseV2(V2Model):
    """The world pose of the object's unique pivot/edit anchor."""

    anchor_kind: Literal["OBJECT_PIVOT"] = "OBJECT_PIVOT"
    world_from_object: RigidTransformV2


class ObjectSupportAssignmentV2(V2Model):
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


class CanonicalObjectV2(V2Model):
    """A stable semantic object identity without source aliases or provenance."""

    object_id: CanonicalId
    category_id: CanonicalId
    movable: StrictBool
    pose: ObjectPoseV2
    support_assignment: ObjectSupportAssignmentV2


class WorkspaceBoundaryFactV2(V2Model):
    """A claimed workspace extent; it is not itself an edit constraint."""

    fact_id: CanonicalId
    region_world_xy: PlanarRegionV2
    boundary_policy: RegionBoundaryPolicyV2
    region_approximation: GeometryApproximationV2
    geometry_uncertainty: UncertaintyBudgetV2


class KnownFreeSpaceFactV2(V2Model):
    """A claimed known-free extent, distinct from workspace and subject domain."""

    fact_id: CanonicalId
    region_world_xy: PlanarRegionV2
    boundary_policy: RegionBoundaryPolicyV2
    region_approximation: GeometryApproximationV2
    geometry_uncertainty: UncertaintyBudgetV2


class PinholeCameraV2(V2Model):
    """One fully specified calibrated pinhole camera convention."""

    camera_id: CanonicalId
    width_px: _PositiveStrictInt
    height_px: _PositiveStrictInt
    intrinsics_row_major: tuple[FiniteFloat, ...] = Field(
        min_length=9,
        max_length=9,
    )
    world_to_camera: RigidTransformV2
    matrix_layout: CameraMatrixLayoutV2 = CameraMatrixLayoutV2.ROW_MAJOR
    camera_axes: CameraAxesV2 = CameraAxesV2.X_RIGHT_Y_DOWN_Z_FORWARD
    pixel_convention: CameraPixelConventionV2 = CameraPixelConventionV2.CENTER_AT_HALF
    depth_convention: CameraDepthConventionV2 = (
        CameraDepthConventionV2.POSITIVE_Z_FORWARD
    )
    near_clip_m: PositiveFiniteFloat
    far_clip_m: PositiveFiniteFloat
    distortion_model: CameraDistortionModelV2 = CameraDistortionModelV2.NONE
    brown_conrady_coefficients: BrownConradyCoefficientsV2 | None = None
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
            self.distortion_model is CameraDistortionModelV2.BROWN_CONRADY
        ):
            raise ValueError(
                "Brown-Conrady coefficients do not match the distortion model"
            )
        return self


class BaselineObservationV2(V2Model):
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


class SupportSurfaceFactV2(V2Model):
    """A planar region on a collision body in an explicit anchor frame.

    ``anchor_from_surface`` maps surface coordinates into the owner object's
    frame when ``owner_object_id`` is present, and into the world frame for an
    environment body. ``normal_in_anchor`` is expressed in that same frame.
    """

    surface_id: CanonicalId
    owner_object_id: CanonicalId | None
    supporting_body_id: CanonicalId
    anchor_from_surface: RigidTransformV2
    normal_in_anchor: Vec3V2
    region_uv: PlanarRegionV2
    region_approximation: GeometryApproximationV2
    boundary_policy: RegionBoundaryPolicyV2
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


class CanonicalSceneV2(V2Model):
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
    objects: FactSetV2[CanonicalObjectV2]
    geometry_instances: FactSetV2[GeometryInstanceV2]
    collision_bodies: FactSetV2[CollisionBodyFactV2]
    workspace_boundaries: FactSetV2[WorkspaceBoundaryFactV2]
    known_free_spaces: FactSetV2[KnownFreeSpaceFactV2]
    support_surfaces: FactSetV2[SupportSurfaceFactV2]
    cameras: FactSetV2[PinholeCameraV2]
    baseline_observations: FactSetV2[BaselineObservationV2]

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
    updates: dict[str, tuple[V2Model, ...]] = {}
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
) -> tuple[V2Model, ...]:
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
) -> dict[str, V2Model]:
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
