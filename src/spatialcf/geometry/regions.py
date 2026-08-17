"""Strict conversion between canonical polygon payloads and GEOS geometry."""

from __future__ import annotations

import math
from collections.abc import Iterable

import shapely
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from spatialcf.domain.models import (
    PlanarPolygon,
    SceneObject,
    SubjectPositionRegion,
    Vec2,
)
from spatialcf.geometry.obb import obb_footprint


def _finite_geometry(geometry: BaseGeometry, label: str) -> BaseGeometry:
    coordinates = shapely.get_coordinates(geometry)
    if any(
        not math.isfinite(float(value))
        for coordinate in coordinates
        for value in coordinate
    ):
        raise ValueError(f"{label} contains non-finite coordinates")
    if not geometry.is_valid:
        raise ValueError(f"{label} is not valid polygonal geometry")
    return shapely.normalize(geometry)


def planar_polygon_geometry(component: PlanarPolygon) -> Polygon:
    """Parse one strict positive-area polygon component."""
    polygon = Polygon(
        [(point.x, point.y) for point in component.exterior],
        [
            [(point.x, point.y) for point in hole]
            for hole in component.holes
        ],
    )
    polygon = _finite_geometry(polygon, "planar polygon")
    if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0.0:
        raise ValueError("planar polygon must have positive area")
    return polygon


def subject_position_region_geometry(region: SubjectPositionRegion) -> BaseGeometry:
    """Return the normalized non-overlapping union encoded by a region."""
    if not region.components:
        return Polygon()
    polygons = tuple(planar_polygon_geometry(item) for item in region.components)
    area_sum = sum(item.area for item in polygons)
    merged = _finite_geometry(unary_union(polygons), "subject position region")
    if not isinstance(merged, (Polygon, MultiPolygon)):
        raise TypeError("subject position region must be polygonal")
    if not math.isclose(merged.area, area_sum, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("subject position region components must not overlap")
    return merged


def planar_polygon_payloads(geometry: BaseGeometry) -> tuple[PlanarPolygon, ...]:
    """Serialize normalized Polygon/MultiPolygon geometry deterministically."""
    geometry = _finite_geometry(geometry, "position region")
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Polygon):
        polygons = (geometry,)
    elif isinstance(geometry, MultiPolygon):
        polygons = tuple(geometry.geoms)
    else:
        raise TypeError("position region must be polygonal")
    components = []
    for polygon in sorted(polygons, key=lambda item: item.wkb):
        components.append(
            PlanarPolygon(
                exterior=tuple(
                    Vec2(x=float(x), y=float(y))
                    for x, y in tuple(polygon.exterior.coords)[:-1]
                ),
                holes=tuple(
                    tuple(
                        Vec2(x=float(x), y=float(y))
                        for x, y in tuple(ring.coords)[:-1]
                    )
                    for ring in polygon.interiors
                ),
            )
        )
    return tuple(components)


def _polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0.0:
            yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for item in geometry.geoms:
            yield from _polygons(item)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    polygons = tuple(_polygons(geometry))
    if not polygons:
        return Polygon()
    return _finite_geometry(unary_union(polygons), "polygonal region")


def _subject_relative_vertices(subject: SceneObject) -> tuple[tuple[float, float], ...]:
    return tuple(
        (x - subject.position.x, y - subject.position.y)
        for x, y in tuple(obb_footprint(subject.obb).exterior.coords)[:-1]
    )


def _convex_center_locus(
    container: Polygon,
    relative_vertices: tuple[tuple[float, float], ...],
) -> BaseGeometry:
    locus: BaseGeometry = container
    for x, y in relative_vertices:
        locus = _polygonal(
            locus.intersection(translate(container, xoff=-x, yoff=-y))
        )
        if locus.is_empty:
            break
    return locus


def conservative_receptacle_position_geometry(
    *,
    surface_patch_bounds_xy: tuple[tuple[float, float, float, float], ...],
    subject: SceneObject,
) -> BaseGeometry:
    """Return the anchor locus whose footprint fits in one complete patch.

    Each bound is ``(xmin, ymin, xmax, ymax)`` in canonical world XY.  Patches
    are eroded independently before union, so disconnected native trigger
    volumes can never prove a placement whose footprint bridges their seam.
    The returned points locate ``subject.position`` (the canonical anchor),
    which need not coincide with the OBB center.
    """

    if type(surface_patch_bounds_xy) is not tuple or not surface_patch_bounds_xy:
        raise ValueError("receptacle evidence must contain surface patches")
    relative_vertices = _subject_relative_vertices(subject)
    loci: list[BaseGeometry] = []
    for index, bounds in enumerate(surface_patch_bounds_xy):
        if type(bounds) is not tuple or len(bounds) != 4:
            raise TypeError(f"receptacle patch {index} must be one bounds tuple")
        xmin, ymin, xmax, ymax = bounds
        if not all(
            type(value) in (int, float) and math.isfinite(float(value))
            for value in bounds
        ):
            raise ValueError(f"receptacle patch {index} bounds must be finite")
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"receptacle patch {index} must have positive area")
        patch = Polygon(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        )
        locus = _convex_center_locus(patch, relative_vertices)
        if not locus.is_empty:
            loci.append(locus)
    if not loci:
        return Polygon()
    return _polygonal(unary_union(loci))


def _configuration_obstacle(
    fixed: Polygon,
    relative_vertices: tuple[tuple[float, float], ...],
) -> Polygon:
    return MultiPoint(
        [
            (fixed_x - subject_x, fixed_y - subject_y)
            for fixed_x, fixed_y in tuple(fixed.exterior.coords)[:-1]
            for subject_x, subject_y in relative_vertices
        ]
    ).convex_hull


def conservative_navigation_position_geometry(
    *,
    room_polygon_xy: tuple[Vec2, ...],
    reachable_positions_xy: tuple[Vec2, ...],
    subject: SceneObject,
    agent_radius_m: float,
    clearance_m: float,
) -> BaseGeometry:
    """Derive a conservative subject-position region from navigation evidence.

    Every reachable agent center proves one inscribed radius-sized disk free of
    native collision geometry.  The returned region contains only subject
    positions whose complete, fixed-orientation footprint lies in the union of
    those known-free disks and in the supplied convex room envelope.  It may
    deliberately reject safe native placements; it never treats an unobserved
    navigation-grid gap as free space.
    """
    room = _finite_geometry(
        Polygon([(point.x, point.y) for point in room_polygon_xy]),
        "navigation room",
    )
    if (
        not isinstance(room, Polygon)
        or room.is_empty
        or room.area <= 0.0
        or not room.equals(room.convex_hull)
    ):
        raise ValueError("navigation room must be one positive convex polygon")
    if not reachable_positions_xy:
        raise ValueError("navigation evidence must contain reachable positions")

    # GEOS buffers approximate circles with inscribed chords, so this union is
    # a subset of the capsule-clear area proven by GetReachablePositions.
    known_free = MultiPoint(
        [(point.x, point.y) for point in reachable_positions_xy]
    ).buffer(agent_radius_m, quad_segs=16)
    if clearance_m > 0.0:
        known_free = known_free.buffer(-clearance_m, quad_segs=16)
    known_free = _polygonal(known_free.intersection(room))

    relative_vertices = _subject_relative_vertices(subject)
    room_locus = _convex_center_locus(room, relative_vertices)
    if room_locus.is_empty or known_free.is_empty:
        return Polygon()

    unknown_or_blocked = _polygonal(room.difference(known_free))
    if unknown_or_blocked.is_empty:
        return room_locus

    triangulation = shapely.constrained_delaunay_triangles(unknown_or_blocked)
    blocked_triangles = tuple(_polygons(triangulation))
    if not blocked_triangles:
        raise ArithmeticError("navigation complement triangulation is empty")
    configuration_obstacles = unary_union(
        [
            _configuration_obstacle(triangle, relative_vertices)
            for triangle in blocked_triangles
        ]
    )
    return _polygonal(room_locus.difference(configuration_obstacles))
