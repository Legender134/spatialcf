"""Opt-in native geometry extraction; legacy captures retain their old owner.

V2 encloses the convex hull of the supplied native metadata corners. It does
not establish mesh/collider containment. Its room is an inward grid domain
inside the reported scene envelope, not a measured usable-floor polygon.
"""

from __future__ import annotations

import math
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from spatialcf.adapters.ai2thor.models import AI2ThorNativeReturnError
from spatialcf.domain.scene import OBB, Quaternion, Vec2, Vec3

LEGACY_GEOMETRY = "legacy_v1"
WORLD_AABB_GEOMETRY = "world_aabb_grid_v2"
_TRANSFORMS = MappingProxyType({
    LEGACY_GEOMETRY: "ai2thor-native-xzy-to-rh-z-up-v1",
    WORLD_AABB_GEOMETRY: "ai2thor-native-xzy-to-rh-z-up-world-aabb-grid-v2",
})
_GRID_DENOMINATOR = 1024


def geometry_transform_version(policy: str) -> str:
    if type(policy) is not str or policy not in _TRANSFORMS:
        raise ValueError("unsupported AI2-THOR geometry policy")
    return _TRANSFORMS[policy]


def _number(value: Any) -> Fraction:
    if type(value) not in (int, float):
        raise AI2ThorNativeReturnError("geometry coordinates must be finite numbers")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise AI2ThorNativeReturnError("geometry coordinate is not representable") from error
    if not math.isfinite(numeric):
        raise AI2ThorNativeReturnError("geometry coordinates must be finite numbers")
    return Fraction(numeric)


def _native_corners(value: Any) -> tuple[tuple[Fraction, ...], ...]:
    if type(value) not in (list, tuple) or len(value) != 8:
        raise AI2ThorNativeReturnError("geometry requires eight native corners")
    points = []
    for point in value:
        if type(point) is dict:
            try:
                row = tuple(point[axis] for axis in ("x", "y", "z"))
            except KeyError as error:
                raise AI2ThorNativeReturnError("native corner is missing a coordinate") from error
        elif type(point) in (list, tuple) and len(point) == 3:
            row = point
        else:
            raise AI2ThorNativeReturnError("native corner must have three coordinates")
        x, y, z = map(_number, row)
        points.append((x, z, y))
    return tuple(points)


def _center_size_bounds(value: Any) -> tuple[tuple[Fraction, Fraction], ...]:
    if type(value) is not dict:
        raise AI2ThorNativeReturnError("native bounds must be an object")
    try:
        center = tuple(_number(value["center"][axis]) for axis in ("x", "z", "y"))
        size = tuple(_number(value["size"][axis]) for axis in ("x", "z", "y"))
    except (KeyError, TypeError) as error:
        raise AI2ThorNativeReturnError("native bounds need center and size") from error
    if any(length <= 0 for length in size):
        raise AI2ThorNativeReturnError("native bounds must have positive size")
    return tuple((c - s / 2, c + s / 2) for c, s in zip(center, size, strict=True))


def _directed_float(value: Fraction, *, upward: bool) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise AI2ThorNativeReturnError("geometry bound is not representable") from error
    if not math.isfinite(result):
        raise AI2ThorNativeReturnError("geometry bound is not representable")
    if (upward and Fraction(result) < value) or (not upward and Fraction(result) > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    if not math.isfinite(result):
        raise AI2ThorNativeReturnError("outward geometry bound is not representable")
    return result


def _outward_extent(center: float, lower: Fraction, upper: Fraction) -> float:
    # Reach representable outward endpoints directly. Incrementing an extent
    # one ULP at a time can take ~2**52 steps when center is large (e.g. 2**53).
    lower_float = _directed_float(lower, upward=False)
    upper_float = _directed_float(upper, upward=True)
    exact = 2 * max(Fraction(center) - Fraction(lower_float),
                    Fraction(upper_float) - Fraction(center))
    try:
        extent = float(exact)
    except OverflowError as error:
        raise AI2ThorNativeReturnError("geometry extent is not representable") from error
    if not math.isfinite(extent) or extent <= 1e-6:
        raise AI2ThorNativeReturnError("geometry extent must be finite and nondegenerate")
    if Fraction(extent) < exact:
        extent = math.nextafter(extent, math.inf)
    if not math.isfinite(extent):
        raise AI2ThorNativeReturnError("outward geometry extent is not representable")
    # Identity-frame float corner construction must retain containment too.
    ends = (center - extent / 2, center + extent / 2)
    if not all(math.isfinite(value) for value in ends):
        raise AI2ThorNativeReturnError("geometry corners are not representable")
    if Fraction(ends[0]) > lower or Fraction(ends[1]) < upper:
        raise AI2ThorNativeReturnError("float geometry corners do not enclose native bounds")
    return extent


def world_aabb(metadata: dict[str, Any]) -> OBB:
    """Enclose raw OBB corners (or the declared AABB when no OBB is supplied)."""
    oriented = metadata.get("objectOrientedBoundingBox")
    if oriented is not None:
        if type(oriented) is not dict:
            raise AI2ThorNativeReturnError("oriented native bounds must be an object")
        points = _native_corners(oriented.get("cornerPoints"))
        bounds = tuple((min(p[a] for p in points), max(p[a] for p in points))
                       for a in range(3))
    else:
        aligned = metadata.get("axisAlignedBoundingBox")
        bounds = _center_size_bounds(aligned)
        if "cornerPoints" in aligned:
            points = _native_corners(aligned["cornerPoints"])
            bounds = tuple((min(lo, *(p[a] for p in points)),
                            max(hi, *(p[a] for p in points)))
                           for a, (lo, hi) in enumerate(bounds))
    center = tuple(float((lo + hi) / 2) for lo, hi in bounds)
    if not all(math.isfinite(c) for c in center):
        raise AI2ThorNativeReturnError("geometry center is not representable")
    extent = tuple(_outward_extent(c, lo, hi)
                   for c, (lo, hi) in zip(center, bounds, strict=True))
    return OBB(
        center=Vec3(x=center[0], y=center[1], z=center[2]),
        extent=Vec3(x=extent[0], y=extent[1], z=extent[2]),
        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def inward_room(bounds: dict[str, Any]) -> tuple[Vec2, ...]:
    """Use fixed binary grid points contained in both reported envelopes."""
    center_bounds = _center_size_bounds(bounds)
    corners = _native_corners(bounds.get("cornerPoints"))
    intervals = []
    for axis in range(2):
        lower = max(center_bounds[axis][0], min(p[axis] for p in corners))
        upper = min(center_bounds[axis][1], max(p[axis] for p in corners))
        lo = Fraction(math.ceil(lower * _GRID_DENOMINATOR), _GRID_DENOMINATOR)
        hi = Fraction(math.floor(upper * _GRID_DENOMINATOR), _GRID_DENOMINATOR)
        if lo >= hi:
            raise AI2ThorNativeReturnError("inward room has no positive grid area")
        values = (float(lo), float(hi))
        if any(not math.isfinite(v) for v in values) or (
            Fraction(values[0]) != lo or Fraction(values[1]) != hi
        ):
            raise AI2ThorNativeReturnError("inward room grid is not representable")
        intervals.append(values)
    (xmin, xmax), (ymin, ymax) = intervals
    return tuple(Vec2(x=x, y=y) for x, y in (
        (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
    ))
