"""Pure proxy geometry, exact input checks and directed numerical bounds."""

from __future__ import annotations

import math
from fractions import Fraction

from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.scene import OBB, Scene
from spatialcf.generation.planning.models import EndpointWorkspace

_NON_SUPPORT_COLLISION_MARGIN_M = 0.01


def default_planning_workspace(
    scene: Scene,
    subject_object_id: str,
) -> EndpointWorkspace:
    """Return the exact rectangular subject-anchor locus used by conversion."""

    checked_scene = _strict_legacy(scene, Scene, "scene")
    if type(subject_object_id) is not str or not subject_object_id:
        raise TypeError("subject_object_id must be a non-empty exact string")
    subject = checked_scene.object_by_id(subject_object_id)
    points = tuple((item.x, item.y) for item in checked_scene.room_polygon_xy)
    xs = sorted({item[0] for item in points})
    ys = sorted({item[1] for item in points})
    if len(points) != 4 or len(xs) != 2 or len(ys) != 2:
        raise ValueError("native planning requires one rectangular room polygon")
    yaw = _obb_yaw(subject.obb)
    half_x = subject.obb.extent.x / 2.0
    half_y = subject.obb.extent.y / 2.0
    radius_x = math.nextafter(
        abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y,
        math.inf,
    )
    radius_y = math.nextafter(
        abs(math.sin(yaw)) * half_x + abs(math.cos(yaw)) * half_y,
        math.inf,
    )
    return EndpointWorkspace(
        min_x_m=math.nextafter(xs[0] + radius_x, math.inf),
        min_y_m=math.nextafter(ys[0] + radius_y, math.inf),
        max_x_m=math.nextafter(xs[1] - radius_x, -math.inf),
        max_y_m=math.nextafter(ys[1] - radius_y, -math.inf),
    )


def _conservative_source_overlap(subject: OBB, obstacle: OBB) -> bool:
    subject_lower_z, subject_upper_z = _z_interval(subject)
    obstacle_lower_z, obstacle_upper_z = _z_interval(obstacle)
    if obstacle_upper_z <= subject_lower_z or subject_upper_z <= obstacle_lower_z:
        return False
    combined_radius = _l1_horizontal_radius(subject) + _l1_horizontal_radius(obstacle)
    return (
        abs(
            Fraction.from_float(subject.center.x)
            - Fraction.from_float(obstacle.center.x)
        )
        < combined_radius
        and abs(
            Fraction.from_float(subject.center.y)
            - Fraction.from_float(obstacle.center.y)
        )
        < combined_radius
    )


def _strict_legacy(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _strict_v2(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _fixed_margin_collision_proxies(
    scene: Scene,
    *,
    subject_object_id: str,
    support_object_id: str | None,
) -> tuple[dict[str, OBB], set[str]]:
    """Return max(existing, 1 cm) proxies without inflating direct support."""

    objects = {item.object_id: item for item in scene.objects}
    overlays = {}
    for obstacle in scene.collision_obstacles:
        if obstacle.source_object_id not in objects:
            raise ValueError("collision overlay references an unknown native object")
        if obstacle.source_object_id in overlays:
            raise ValueError("collision overlays must be unique by native object")
        overlays[obstacle.source_object_id] = obstacle

    result: dict[str, OBB] = {}
    minimum_margin_ids: set[str] = set()
    for object_id, item in objects.items():
        if object_id == subject_object_id:
            continue
        overlay = overlays.get(object_id)
        if object_id == support_object_id:
            continue
        base = item.obb if overlay is None else overlay.obb
        existing = 0.0 if overlay is None else overlay.clearance_m
        proxy = _inflate_obb(base, max(existing, _NON_SUPPORT_COLLISION_MARGIN_M))
        _require_valid_proxy_obb(proxy)
        result[object_id] = proxy
        minimum_margin_ids.add(object_id)
    return result, minimum_margin_ids


def _inflate_obb(obb: OBB, margin_m: float) -> OBB:
    diameter = 2.0 * margin_m
    return obb.model_copy(
        update={
            "extent": obb.extent.model_copy(
                update={
                    "x": obb.extent.x + diameter,
                    "y": obb.extent.y + diameter,
                    "z": obb.extent.z + diameter,
                }
            )
        }
    )


def _partition_fixed_obstacles(
    scene: Scene,
    subject_id: str,
    subject_obb: OBB,
    workspace: EndpointWorkspace,
    collision_proxies: dict[str, OBB],
) -> tuple[set[str], set[str]]:
    included: set[str] = set()
    excluded: set[str] = set()
    for item in scene.objects:
        if item.object_id == subject_id:
            continue
        proxy = collision_proxies.get(item.object_id, item.obb)
        target = (
            excluded if _proven_disjoint(subject_obb, proxy, workspace) else included
        )
        target.add(item.object_id)
    return included, excluded


def _proven_disjoint(
    subject: OBB,
    obstacle: OBB,
    workspace: EndpointWorkspace,
) -> bool:
    subject_lower_z, subject_upper_z = _z_interval(subject)
    obstacle_lower_z, obstacle_upper_z = _z_interval(obstacle)
    if obstacle_upper_z <= subject_lower_z or subject_upper_z <= obstacle_lower_z:
        return True

    # hx + hy encloses the projection of any upright yawed rectangle on both
    # world axes.  It is deliberately looser than sqrt(hx^2 + hy^2), but its
    # exact rational proof needs no trigonometric or binary64 tolerance.
    subject_radius = _l1_horizontal_radius(subject)
    obstacle_radius = _l1_horizontal_radius(obstacle)
    obstacle_x = Fraction.from_float(obstacle.center.x)
    obstacle_y = Fraction.from_float(obstacle.center.y)
    min_x = Fraction.from_float(workspace.min_x_m) - subject_radius
    max_x = Fraction.from_float(workspace.max_x_m) + subject_radius
    min_y = Fraction.from_float(workspace.min_y_m) - subject_radius
    max_y = Fraction.from_float(workspace.max_y_m) + subject_radius
    return (
        obstacle_x + obstacle_radius <= min_x
        or max_x <= obstacle_x - obstacle_radius
        or obstacle_y + obstacle_radius <= min_y
        or max_y <= obstacle_y - obstacle_radius
    )


def _z_interval(obb: OBB) -> tuple[Fraction, Fraction]:
    center = Fraction.from_float(obb.center.z)
    half = Fraction.from_float(obb.extent.z) / 2
    return center - half, center + half


def _l1_horizontal_radius(obb: OBB) -> Fraction:
    return (Fraction.from_float(obb.extent.x) + Fraction.from_float(obb.extent.y)) / 2


def _require_workspace_subset(
    problem: SemanticProblemV2_3,
    workspace: EndpointWorkspace,
) -> None:
    component = problem.scene.workspace_boundaries.values[0].region_world_xy.components[
        0
    ]
    vertices = component.exterior.vertices
    allowed = (
        min(item.x for item in vertices),
        min(item.y for item in vertices),
        max(item.x for item in vertices),
        max(item.y for item in vertices),
    )
    requested = (
        workspace.min_x_m,
        workspace.min_y_m,
        workspace.max_x_m,
        workspace.max_y_m,
    )
    if any(
        Fraction.from_float(actual) < Fraction.from_float(limit)
        for actual, limit in zip(requested[:2], allowed[:2], strict=True)
    ) or any(
        Fraction.from_float(actual) > Fraction.from_float(limit)
        for actual, limit in zip(requested[2:], allowed[2:], strict=True)
    ):
        raise ValueError("endpoint workspace exceeds the subject anchor locus")


def _clamp_support_collision_top(
    payload: dict,
    supporting_body_id: str,
    exact_contact_z: Fraction,
) -> None:
    """Keep the proxy support solid at or below its exact contact plane.

    AI2-THOR receptacle OBBs are visual/collision envelopes rather than exact
    contact surfaces.  Their top can sit slightly above an object's observed
    resting bottom, making one source snapshot claim both support and solid
    penetration.  The native proxy already replaces the receptacle surface by
    the subject-bottom plane; its collision solid must use that same boundary.
    The center is rounded downward so the closed solid can touch but never
    cross the exact rational plane.
    """

    bodies = {
        item["body_id"]: item for item in payload["scene"]["collision_bodies"]["values"]
    }
    try:
        geometry_ids = set(bodies[supporting_body_id]["geometry_instance_ids"])
    except KeyError as error:
        raise RuntimeError(
            "support proxy lost its supporting collision body"
        ) from error
    if not geometry_ids:
        raise RuntimeError("support proxy collision body must not be empty")

    matched: set[str] = set()
    for geometry in payload["scene"]["geometry_instances"]["values"]:
        geometry_id = geometry["geometry_id"]
        if geometry_id not in geometry_ids:
            continue
        matched.add(geometry_id)
        try:
            size_z = geometry["shape"]["size_m"]["z"]
            anchor = geometry["anchor_from_geometry"]
            center_z = anchor["translation"]["z"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "support collision proxy must be one anchored upright box"
            ) from error
        half_extent_z = Fraction.from_float(size_z) / 2
        current_top = Fraction.from_float(center_z) + half_extent_z
        if current_top <= exact_contact_z:
            continue
        clamped_center = _fraction_floor_binary64(exact_contact_z - half_extent_z)
        geometry["anchor_from_geometry"] = {
            **anchor,
            "translation": {
                **anchor["translation"],
                "z": clamped_center,
            },
        }
        if Fraction.from_float(clamped_center) + half_extent_z > exact_contact_z:
            raise RuntimeError("support collision top clamp rounded inward")
    if matched != geometry_ids:
        raise RuntimeError("support collision body geometry closure is incomplete")


def _fraction_floor_binary64(value: Fraction) -> float:
    try:
        published = float(value)
    except OverflowError as error:
        raise ArithmeticError("support collision center is not finite") from error
    if not math.isfinite(published):
        raise ArithmeticError("support collision center is not finite")
    if Fraction.from_float(published) > value:
        published = math.nextafter(published, -math.inf)
    if not math.isfinite(published) or Fraction.from_float(published) > value:
        raise ArithmeticError("support collision center cannot be rounded downward")
    return published


def _directed_fraction_bounds(value: Fraction) -> tuple[float, float]:
    published = float(value)
    if not math.isfinite(published):
        raise ArithmeticError("native support residual is not finite")
    exact = Fraction.from_float(published)
    lower = math.nextafter(published, -math.inf) if exact > value else published
    upper = math.nextafter(published, math.inf) if exact < value else published
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ArithmeticError("native support residual cannot be enclosed")
    return lower, upper


def _obb_transform(obb: OBB) -> dict[str, object]:
    _require_valid_proxy_obb(obb)
    yaw = _obb_yaw(obb)
    return {
        "kind": "DIRECTED_YAW_INTERVAL",
        "translation": obb.center.model_dump(mode="python"),
        "yaw_radians": yaw,
    }


def _obb_yaw(obb: OBB) -> float:
    quaternion = obb.rotation
    if quaternion.x != 0.0 or quaternion.y != 0.0:
        raise ValueError("native collision proxy must be exact upright yaw")
    norm = math.hypot(quaternion.z, quaternion.w)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("native collision proxy quaternion must be unit length")
    return 0.0 if quaternion.z == 0.0 else 2.0 * math.atan2(quaternion.z, quaternion.w)


def _require_valid_proxy_obb(obb: OBB) -> None:
    values = (
        obb.center.x,
        obb.center.y,
        obb.center.z,
        obb.extent.x,
        obb.extent.y,
        obb.extent.z,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("native collision proxy must be finite")
    if min(obb.extent.x, obb.extent.y, obb.extent.z) <= 0.0:
        raise ValueError("native collision proxy extents must be positive")
    _obb_yaw(obb)


default_planning_workspace.__module__ = "spatialcf.generation.planning.problem"
_conservative_source_overlap.__module__ = "spatialcf.generation.planning.problem"
_strict_legacy.__module__ = "spatialcf.generation.planning.problem"
_strict_v2.__module__ = "spatialcf.generation.planning.problem"
_fixed_margin_collision_proxies.__module__ = "spatialcf.generation.planning.problem"
_inflate_obb.__module__ = "spatialcf.generation.planning.problem"
_partition_fixed_obstacles.__module__ = "spatialcf.generation.planning.problem"
_proven_disjoint.__module__ = "spatialcf.generation.planning.problem"
_z_interval.__module__ = "spatialcf.generation.planning.problem"
_l1_horizontal_radius.__module__ = "spatialcf.generation.planning.problem"
_require_workspace_subset.__module__ = "spatialcf.generation.planning.problem"
_clamp_support_collision_top.__module__ = "spatialcf.generation.planning.problem"
_fraction_floor_binary64.__module__ = "spatialcf.generation.planning.problem"
_directed_fraction_bounds.__module__ = "spatialcf.generation.planning.problem"
_obb_transform.__module__ = "spatialcf.generation.planning.problem"
_obb_yaw.__module__ = "spatialcf.generation.planning.problem"
_require_valid_proxy_obb.__module__ = "spatialcf.generation.planning.problem"
