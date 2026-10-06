"""Pure geometric residuals used by final native audit."""

from __future__ import annotations

import math

from spatialcf.domain.scene import OBB

_OBJECT_GEOMETRY_TOLERANCE_M = 1e-5


def _vec3_values(value) -> tuple[float, float, float]:
    return value.x, value.y, value.z


def _quaternion_angle_residual_deg(left, right, label: str) -> float:
    a = tuple(float(item) for item in (left.x, left.y, left.z, left.w))
    b = tuple(float(item) for item in (right.x, right.y, right.z, right.w))
    if not all(math.isfinite(item) for item in (*a, *b)):
        raise ValueError(f"{label} must contain finite values")
    a_norm = math.sqrt(sum(item * item for item in a))
    b_norm = math.sqrt(sum(item * item for item in b))
    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError(f"{label} must contain non-zero quaternions")
    cosine = abs(
        sum(
            (a_item / a_norm) * (b_item / b_norm)
            for a_item, b_item in zip(a, b, strict=True)
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, cosine))))


def _obb_corner_coordinates(obb: OBB, label: str) -> tuple[tuple[float, ...], ...]:
    values = (
        obb.center.x,
        obb.center.y,
        obb.center.z,
        obb.extent.x,
        obb.extent.y,
        obb.extent.z,
        obb.rotation.x,
        obb.rotation.y,
        obb.rotation.z,
        obb.rotation.w,
    )
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError(f"{label} must contain finite values")
    if any(float(item) <= 0.0 for item in (obb.extent.x, obb.extent.y, obb.extent.z)):
        raise ValueError(f"{label} must contain strictly positive extents")
    x, y, z, w = (
        float(obb.rotation.x),
        float(obb.rotation.y),
        float(obb.rotation.z),
        float(obb.rotation.w),
    )
    maximum_component = max(abs(x), abs(y), abs(z), abs(w))
    if maximum_component == 0.0:
        raise ValueError(f"{label} must contain a non-zero quaternion")
    scaled = tuple(component / maximum_component for component in (x, y, z, w))
    scaled_norm = math.sqrt(sum(component * component for component in scaled))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise ValueError(f"{label} must contain a normalizable quaternion")
    x, y, z, w = (component / scaled_norm for component in scaled)
    rotation = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    center = (float(obb.center.x), float(obb.center.y), float(obb.center.z))
    half_extent = (
        float(obb.extent.x) / 2.0,
        float(obb.extent.y) / 2.0,
        float(obb.extent.z) / 2.0,
    )
    return tuple(
        tuple(
            center[row]
            + sum(
                rotation[row][column] * signs[column] * half_extent[column]
                for column in range(3)
            )
            for row in range(3)
        )
        for signs in (
            (dx, dy, dz)
            for dx in (-1.0, 1.0)
            for dy in (-1.0, 1.0)
            for dz in (-1.0, 1.0)
        )
    )


def _obb_corner_hausdorff_residual_m(left: OBB, right: OBB, label: str) -> float:
    left_corners = _obb_corner_coordinates(left, label)
    right_corners = _obb_corner_coordinates(right, label)

    def directed(source, target) -> float:
        return max(
            min(math.dist(source_corner, target_corner) for target_corner in target)
            for source_corner in source
        )

    return max(
        directed(left_corners, right_corners),
        directed(right_corners, left_corners),
    )


def _close_values(left, right, tolerance: float) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right, strict=True)
    )


def _quaternions_close(left, right) -> bool:
    a = tuple(float(item) for item in (left.x, left.y, left.z, left.w))
    b = tuple(float(item) for item in (right.x, right.y, right.z, right.w))
    if not all(math.isfinite(item) for item in (*a, *b)):
        return False
    a_norm = math.sqrt(sum(item * item for item in a))
    b_norm = math.sqrt(sum(item * item for item in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return False
    normalized_a = tuple(item / a_norm for item in a)
    normalized_b = tuple(item / b_norm for item in b)
    return _close_values(
        normalized_a,
        normalized_b,
        _OBJECT_GEOMETRY_TOLERANCE_M,
    ) or _close_values(
        normalized_a,
        tuple(-item for item in normalized_b),
        _OBJECT_GEOMETRY_TOLERANCE_M,
    )


_vec3_values.__module__ = "spatialcf.generation.execution.audit"
_quaternion_angle_residual_deg.__module__ = "spatialcf.generation.execution.audit"
_obb_corner_coordinates.__module__ = "spatialcf.generation.execution.audit"
_obb_corner_hausdorff_residual_m.__module__ = "spatialcf.generation.execution.audit"
_close_values.__module__ = "spatialcf.generation.execution.audit"
_quaternions_close.__module__ = "spatialcf.generation.execution.audit"
