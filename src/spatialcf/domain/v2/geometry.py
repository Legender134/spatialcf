"""Platform-neutral Canonical v2 geometry facts.

Geometry describes what is known about a scene.  Clearance, contact policy,
candidate selection, and platform provenance intentionally do not belong here.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FiniteFloat,
    RigidTransformV2,
    UncertaintyBudgetV2,
    V2Model,
    Vec2V2,
    Vec3V2,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


class RingWindingV2(StrEnum):
    """Canonical orientation of a polygon ring."""

    COUNTERCLOCKWISE = "COUNTERCLOCKWISE"
    CLOCKWISE = "CLOCKWISE"


class GeometryRoleV2(StrEnum):
    """The semantic consumer for one non-interchangeable geometry instance."""

    VISUAL = "VISUAL"
    RELATION = "RELATION"
    COLLISION = "COLLISION"
    SUPPORT = "SUPPORT"
    OCCLUDER = "OCCLUDER"


class GeometryApproximationV2(StrEnum):
    """Sound relationship between a supplied shape and the represented shape."""

    EXACT = "EXACT"
    INNER = "INNER"
    OUTER = "OUTER"


def _signed_area(coordinates: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
            strict=True,
        )
    )


def _canonical_ring_coordinates(
    coordinates: tuple[tuple[float, float], ...],
    winding: RingWindingV2,
) -> tuple[tuple[float, float], ...]:
    wants_positive_area = winding is RingWindingV2.COUNTERCLOCKWISE
    if (_signed_area(coordinates) > 0.0) is not wants_positive_area:
        coordinates = tuple(reversed(coordinates))
    first_index = min(range(len(coordinates)), key=coordinates.__getitem__)
    return coordinates[first_index:] + coordinates[:first_index]


class PlanarRingV2(V2Model):
    """A simple canonical ring without a repeated closing vertex."""

    winding: RingWindingV2
    vertices: tuple[Vec2V2, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_and_canonicalize_ring(self) -> Self:
        coordinates = tuple(
            (
                0.0 if point.x == 0.0 else point.x,
                0.0 if point.y == 0.0 else point.y,
            )
            for point in self.vertices
        )
        if coordinates[0] == coordinates[-1]:
            raise ValueError("ring must omit the repeated closing vertex")
        closed_pairs = zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
            strict=True,
        )
        if any(left == right for left, right in closed_pairs):
            raise ValueError("ring must not contain adjacent duplicate vertices")
        if len(set(coordinates)) < 3:
            raise ValueError("ring must contain at least three distinct vertices")
        triples = zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
            coordinates[2:] + coordinates[:2],
            strict=True,
        )
        if any(
            (middle[0] - left[0]) * (right[1] - middle[1])
            == (middle[1] - left[1]) * (right[0] - middle[0])
            for left, middle, right in triples
        ):
            raise ValueError("ring must not contain redundant collinear vertices")
        area = _signed_area(coordinates)
        if area == 0.0:
            raise ValueError("ring must have positive non-zero area")
        from shapely.geometry import LinearRing

        if not LinearRing(coordinates).is_simple:
            raise ValueError("ring must be simple")

        canonical = _canonical_ring_coordinates(coordinates, self.winding)
        object.__setattr__(
            self,
            "vertices",
            tuple(Vec2V2(x=x, y=y) for x, y in canonical),
        )
        return self


class PlanarPolygonComponentV2(V2Model):
    """One valid polygon component with canonical, pairwise-disjoint holes."""

    exterior: PlanarRingV2
    holes: tuple[PlanarRingV2, ...] = ()

    @model_validator(mode="after")
    def validate_and_canonicalize_component(self) -> Self:
        if self.exterior.winding is not RingWindingV2.COUNTERCLOCKWISE:
            raise ValueError("polygon exterior must be COUNTERCLOCKWISE")
        if any(hole.winding is not RingWindingV2.CLOCKWISE for hole in self.holes):
            raise ValueError("polygon holes must be CLOCKWISE")

        ordered_holes = tuple(sorted(self.holes, key=canonical_json_bytes_v2))
        if len(set(ordered_holes)) != len(ordered_holes):
            raise ValueError("polygon holes must be unique")
        polygon = _polygon(self.exterior, ordered_holes)
        if not polygon.is_valid or polygon.area <= 0.0:
            raise ValueError("polygon holes must be contained, disjoint, and valid")
        object.__setattr__(self, "holes", ordered_holes)
        return self


class PlanarRegionV2(V2Model):
    """A canonical non-empty union of disjoint polygon components."""

    components: tuple[PlanarPolygonComponentV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_and_canonicalize_components(self) -> Self:
        ordered = tuple(sorted(self.components, key=canonical_json_bytes_v2))
        if len(set(ordered)) != len(ordered):
            raise ValueError("region components must be unique")
        polygons = tuple(_polygon(item.exterior, item.holes) for item in ordered)
        for index, left in enumerate(polygons):
            for right in polygons[index + 1 :]:
                if not left.disjoint(right):
                    raise ValueError("region components must be pairwise disjoint")
        object.__setattr__(self, "components", ordered)
        return self


class UprightBox3DV2(V2Model):
    """An object-local box whose instance transform may rotate only around Z."""

    shape_type: Literal["UPRIGHT_BOX_3D"] = "UPRIGHT_BOX_3D"
    origin_convention: Literal["CENTERED_AT_GEOMETRY_FRAME"] = (
        "CENTERED_AT_GEOMETRY_FRAME"
    )
    size_m: Vec3V2

    @model_validator(mode="after")
    def validate_positive_size(self) -> Self:
        if any(value <= 0.0 for value in (self.size_m.x, self.size_m.y, self.size_m.z)):
            raise ValueError("upright box size_m components must be positive")
        return self


class ExtrudedPlanarPolygonV2(V2Model):
    """One object-local planar polygon extruded through a finite Z interval."""

    shape_type: Literal["EXTRUDED_PLANAR_POLYGON"] = "EXTRUDED_PLANAR_POLYGON"
    coordinate_convention: Literal["GEOMETRY_FRAME_XY_AND_Z"] = (
        "GEOMETRY_FRAME_XY_AND_Z"
    )
    footprint: PlanarPolygonComponentV2
    lower_z_m: FiniteFloat
    upper_z_m: FiniteFloat

    @model_validator(mode="after")
    def validate_z_interval(self) -> Self:
        if self.upper_z_m <= self.lower_z_m:
            raise ValueError("upper_z_m must be greater than lower_z_m")
        return self


GeometryShapeV2 = Annotated[
    UprightBox3DV2 | ExtrudedPlanarPolygonV2,
    Field(discriminator="shape_type"),
]


class GeometryInstanceV2(V2Model):
    """One role-specific shape anchored in an object frame or the world frame."""

    geometry_id: CanonicalId
    owner_object_id: CanonicalId | None
    role: GeometryRoleV2
    anchor_from_geometry: RigidTransformV2
    approximation: GeometryApproximationV2
    uncertainty: UncertaintyBudgetV2
    shape: GeometryShapeV2

    @model_validator(mode="after")
    def validate_upright_instance(self) -> Self:
        if isinstance(self.shape, (UprightBox3DV2, ExtrudedPlanarPolygonV2)):
            rotation = self.anchor_from_geometry.rotation
            if not math.isclose(
                rotation.x, 0.0, rel_tol=0.0, abs_tol=1e-12
            ) or not math.isclose(
                rotation.y,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "upright box anchor rotation must be a yaw-only rotation"
                )
        return self


class CollisionBodyFactV2(V2Model):
    """A collision body referencing collision-role geometry facts.

    ``owner_object_id=None`` denotes a typed environment body.  The Scene
    reference graph decides whether every referenced geometry and optional
    owner exists; collision clearance remains a constraint policy.
    """

    body_id: CanonicalId
    owner_object_id: CanonicalId | None
    composition: Literal["CLOSED_SOLID_UNION"] = "CLOSED_SOLID_UNION"
    geometry_instance_ids: tuple[CanonicalId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_geometry_ids(self) -> Self:
        if len(set(self.geometry_instance_ids)) != len(self.geometry_instance_ids):
            raise ValueError("collision body geometry_instance_ids must be unique")
        object.__setattr__(
            self,
            "geometry_instance_ids",
            tuple(sorted(self.geometry_instance_ids)),
        )
        return self


def _polygon(
    exterior: PlanarRingV2,
    holes: tuple[PlanarRingV2, ...],
) -> object:
    from shapely.geometry import Polygon

    return Polygon(
        tuple((point.x, point.y) for point in exterior.vertices),
        tuple(tuple((point.x, point.y) for point in hole.vertices) for hole in holes),
    )
