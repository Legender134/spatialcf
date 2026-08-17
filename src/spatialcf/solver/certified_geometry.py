"""Conservative inner and outer geometry for distance constraints."""

import math

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from spatialcf.solver.certified_models import FeasibleRegionBracket
from spatialcf.solver.feasible import (
    GEOMETRY_EPS,
    _convex_minkowski_sum,
    _minimum_side_apothem,
    _regular_disk_polygon,
)


def disk_bracket(
    radius: float,
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    """Bracket a true origin-centred disk with safe regular polygons."""
    inner = _regular_disk_polygon(
        radius,
        circumscribed=False,
        sides=disk_segments,
    )
    outer = _regular_disk_polygon(
        radius,
        circumscribed=True,
        sides=disk_segments,
    )
    inner_vertices = list(inner.exterior.coords)[:-1]
    outer_vertices = list(outer.exterior.coords)[:-1]
    radial_geometry_error = max(
        radius - _minimum_side_apothem(inner_vertices),
        max(math.hypot(x, y) for x, y in outer_vertices) - radius,
        0.0,
    )
    return FeasibleRegionBracket.create(
        inner=inner,
        outer=outer,
        radial_geometry_error=radial_geometry_error,
        disk_segments=disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def near_bracket(
    configuration_obstacle: Polygon,
    radius: float,
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    """Bracket points within ``radius`` of a convex configuration obstacle."""
    disk = disk_bracket(radius, disk_segments, numeric_tolerance)
    inner = _convex_minkowski_sum(
        configuration_obstacle,
        list(disk.inner.exterior.coords)[:-1],
    )
    outer = _convex_minkowski_sum(
        configuration_obstacle,
        list(disk.outer.exterior.coords)[:-1],
    )
    return FeasibleRegionBracket.create(
        inner=inner,
        outer=outer,
        radial_geometry_error=disk.radial_geometry_error,
        disk_segments=disk.disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def far_bracket(
    configuration_obstacle: Polygon,
    radius: float,
    disk_segments: int,
    numeric_tolerance: float,
    universe: BaseGeometry,
) -> FeasibleRegionBracket:
    """Bracket points at least ``radius`` from a configuration obstacle."""
    forbidden = near_bracket(
        configuration_obstacle,
        radius,
        disk_segments,
        numeric_tolerance,
    )
    return complement_bracket(forbidden, universe)


def intersect_brackets(
    left: FeasibleRegionBracket,
    right: FeasibleRegionBracket,
) -> FeasibleRegionBracket:
    """Conservatively intersect two region brackets."""
    return FeasibleRegionBracket.create(
        inner=left.inner.intersection(right.inner),
        outer=left.outer.intersection(right.outer),
        radial_geometry_error=max(
            left.radial_geometry_error,
            right.radial_geometry_error,
        ),
        disk_segments=min(left.disk_segments, right.disk_segments),
        numeric_tolerance=GEOMETRY_EPS,
    )


def union_brackets(
    left: FeasibleRegionBracket,
    right: FeasibleRegionBracket,
    universe: BaseGeometry | None = None,
) -> FeasibleRegionBracket:
    """Conservatively unite two brackets; accept a compositional universe."""
    inner = left.inner.union(right.inner)
    outer = left.outer.union(right.outer)
    return FeasibleRegionBracket.create(
        inner=inner,
        outer=outer,
        radial_geometry_error=max(
            left.radial_geometry_error,
            right.radial_geometry_error,
        ),
        disk_segments=min(left.disk_segments, right.disk_segments),
        numeric_tolerance=GEOMETRY_EPS,
    )


def complement_bracket(
    value: FeasibleRegionBracket,
    universe: BaseGeometry,
) -> FeasibleRegionBracket:
    """Complement a bracket within ``universe``, reversing its polarity."""
    return FeasibleRegionBracket.create(
        inner=universe.difference(value.outer),
        outer=universe.difference(value.inner),
        radial_geometry_error=value.radial_geometry_error,
        disk_segments=value.disk_segments,
        numeric_tolerance=GEOMETRY_EPS,
    )
