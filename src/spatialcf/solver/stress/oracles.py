"""Independent analytic geometry for deterministic solver stress cases."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from spatialcf.domain.enums import Relation, RelationAxis
from spatialcf.domain.models import Camera, SceneObject, Vec2
from spatialcf.geometry.obb import ground_gap, obb_footprint
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.stress.models import (
    StressCase,
    StressCaseDraft,
    StressOracleResult,
)

_EPS = 1e-12
_GEOMETRY_TOLERANCE = 1e-9
# Frozen from the public relation-label boundary contract.  Stress oracles keep
# this value local so their operational geometry is derived independently of
# the production relation engine implementation.
_OPERATIONAL_COMPARISON_TOLERANCE = 1e-9
_BUFFER_QUAD_SEGS = 8192
_AXIS_RELATIONS: dict[RelationAxis, tuple[Relation, Relation]] = {
    RelationAxis.HORIZONTAL: (Relation.LEFT, Relation.RIGHT),
    RelationAxis.DEPTH: (Relation.FRONT, Relation.BEHIND),
    RelationAxis.DISTANCE: (Relation.NEAR, Relation.FAR),
}


@dataclass(frozen=True)
class PointProjection:
    """The Euclidean projection of one point onto a closed primitive."""

    point: Vec2
    distance: float


@dataclass(frozen=True)
class _ConvexOffsetSegment:
    """One exact translated edge of a convex Euclidean offset."""

    start: Vec2
    end: Vec2


@dataclass(frozen=True)
class _ConvexOffsetArc:
    """One exact CCW circular fillet of a convex Euclidean offset."""

    center: Vec2
    radius: float
    start_angle: float
    end_angle: float


@dataclass(frozen=True)
class _ConvexOffsetPrimitives:
    """Finite exact boundary primitives for a convex polygon offset."""

    segments: tuple[_ConvexOffsetSegment, ...]
    arcs: tuple[_ConvexOffsetArc, ...]


class UnattainedOracleInfimumError(ValueError):
    """The closed geometric infimum has no legal exact stress witness."""


def point_to_segment(point: Vec2, start: Vec2, end: Vec2) -> PointProjection:
    """Return the exact clamped projection of ``point`` onto one segment."""
    dx = end.x - start.x
    dy = end.y - start.y
    squared_length = dx * dx + dy * dy
    if squared_length <= _EPS:
        closest = start
    else:
        parameter = (
            (point.x - start.x) * dx + (point.y - start.y) * dy
        ) / squared_length
        parameter = min(1.0, max(0.0, parameter))
        closest = Vec2(
            x=start.x + parameter * dx,
            y=start.y + parameter * dy,
        )
    return PointProjection(
        point=closest,
        distance=math.hypot(point.x - closest.x, point.y - closest.y),
    )


def project_to_half_space(
    point: Vec2,
    *,
    normal: Vec2,
    offset: float,
) -> PointProjection:
    """Project onto ``normal dot point >= offset``."""
    squared_norm = normal.x * normal.x + normal.y * normal.y
    if squared_norm <= _EPS:
        raise ValueError("half-space normal must be non-zero")
    signed_deficit = offset - (normal.x * point.x + normal.y * point.y)
    if signed_deficit <= 0.0:
        return PointProjection(point=point, distance=0.0)
    scale = signed_deficit / squared_norm
    projected = Vec2(
        x=point.x + scale * normal.x,
        y=point.y + scale * normal.y,
    )
    return PointProjection(
        point=projected,
        distance=signed_deficit / math.sqrt(squared_norm),
    )


def _relative_subject_vertices(subject: SceneObject) -> tuple[Vec2, ...]:
    return tuple(
        Vec2(x=float(x) - subject.position.x, y=float(y) - subject.position.y)
        for x, y in tuple(obb_footprint(subject.obb).exterior.coords)[:-1]
    )


def _canonical_polygon_vertices(polygon: BaseGeometry) -> tuple[Vec2, ...]:
    if polygon.is_empty:
        return ()
    if not isinstance(polygon, Polygon):
        raise TypeError("configuration-space hull must be a polygon")
    coordinates = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
    signed_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(
            coordinates,
            coordinates[1:] + coordinates[:1],
            strict=True,
        )
    )
    if signed_area < 0.0:
        coordinates.reverse()
    first = min(
        range(len(coordinates)),
        key=lambda index: (coordinates[index][1], coordinates[index][0]),
    )
    ordered = coordinates[first:] + coordinates[:first]
    return tuple(Vec2(x=x, y=y) for x, y in ordered)


def configuration_space_vertices(
    subject: SceneObject,
    obstacle: SceneObject,
) -> tuple[Vec2, ...]:
    """Return the convex obstacle OBB vertices minus relative subject vertices."""
    obstacle_vertices = tuple(obb_footprint(obstacle.obb).exterior.coords)[:-1]
    relative_vertices = _relative_subject_vertices(subject)
    hull = MultiPoint(
        [
            (float(ox) - subject_vertex.x, float(oy) - subject_vertex.y)
            for ox, oy in obstacle_vertices
            for subject_vertex in relative_vertices
        ]
    ).convex_hull
    return _canonical_polygon_vertices(hull)


def _configuration_polygon(subject: SceneObject, obstacle: SceneObject) -> Polygon:
    vertices = configuration_space_vertices(subject, obstacle)
    if len(vertices) < 3:
        return Polygon()
    return Polygon([(point.x, point.y) for point in vertices])


def _distance_buffer_envelopes(
    configuration: BaseGeometry,
    radius: float,
) -> tuple[BaseGeometry, BaseGeometry]:
    """Return inscribed and circumscribed polygonal envelopes of an exact buffer."""
    inner = configuration.buffer(radius, quad_segs=_BUFFER_QUAD_SEGS)
    # GEOS rounds a fillet's segment count as floor(angle / quantum + 1/2),
    # where quantum = pi / (2 * quad_segs).  For every emitted fillet segment,
    # its angle is therefore less than 3/2 * quantum, and hence less than
    # 2 * quantum.  The supported OBB yaw grid has no non-zero turn below 15
    # degrees, so GEOS's near-parallel endpoint shortcut cannot apply.  Thus a
    # chord's half-angle is strictly below quantum, and scaling by sec(quantum)
    # makes the polygonal chord cover the exact radius-r circular arc.
    chord_half_angle_bound = math.pi / (2.0 * _BUFFER_QUAD_SEGS)
    outer_radius = radius / math.cos(chord_half_angle_bound)
    outer = configuration.buffer(outer_radius, quad_segs=_BUFFER_QUAD_SEGS)
    return inner, outer


def _without_nearly_collinear_vertices(
    coordinates: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    simplified = coordinates.copy()
    while len(simplified) > 3:
        for index, current in enumerate(simplified):
            previous = simplified[index - 1]
            following = simplified[(index + 1) % len(simplified)]
            incoming = (current[0] - previous[0], current[1] - previous[1])
            outgoing = (following[0] - current[0], following[1] - current[1])
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            cross_scale = math.hypot(*incoming) * math.hypot(*outgoing)
            cross_tolerance = 64.0 * math.ulp(cross_scale)
            between = (
                incoming[0] * (current[0] - following[0])
                + incoming[1] * (current[1] - following[1])
                <= cross_tolerance
            )
            if abs(cross) <= cross_tolerance and between:
                del simplified[index]
                break
        else:
            return simplified
    return simplified


def _convex_offset_primitives(
    configuration: Polygon,
    radius: float,
) -> _ConvexOffsetPrimitives:
    """Return exact edge offsets and convex-vertex fillets for a convex polygon."""
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("offset radius must be finite and positive")
    if (
        configuration.is_empty
        or not configuration.is_valid
        or not configuration.equals(configuration.convex_hull)
    ):
        raise ValueError("configuration must be a non-empty convex polygon")
    coordinates = [
        (float(x), float(y)) for x, y in tuple(configuration.exterior.coords)[:-1]
    ]
    if not configuration.exterior.is_ccw:
        coordinates.reverse()
    coordinates = _without_nearly_collinear_vertices(coordinates)

    normals: list[tuple[float, float]] = []
    segments: list[_ConvexOffsetSegment] = []
    for start, end in zip(
        coordinates,
        coordinates[1:] + coordinates[:1],
        strict=True,
    ):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= _EPS:
            raise ValueError("configuration has a degenerate edge")
        normal = (dy / length, -dx / length)
        normals.append(normal)
        segments.append(
            _ConvexOffsetSegment(
                start=Vec2(
                    x=start[0] + radius * normal[0],
                    y=start[1] + radius * normal[1],
                ),
                end=Vec2(
                    x=end[0] + radius * normal[0],
                    y=end[1] + radius * normal[1],
                ),
            )
        )

    arcs: list[_ConvexOffsetArc] = []
    for index, coordinate in enumerate(coordinates):
        incoming_normal = normals[index - 1]
        outgoing_normal = normals[index]
        start_angle = math.atan2(incoming_normal[1], incoming_normal[0])
        turn = (
            math.atan2(outgoing_normal[1], outgoing_normal[0]) - start_angle
        ) % math.tau
        if not 0.0 < turn < math.pi:
            raise ValueError("configuration must have strictly convex turns")
        arcs.append(
            _ConvexOffsetArc(
                center=Vec2(x=coordinate[0], y=coordinate[1]),
                radius=radius,
                start_angle=start_angle,
                end_angle=start_angle + turn,
            )
        )
    return _ConvexOffsetPrimitives(segments=tuple(segments), arcs=tuple(arcs))


def _point_on_arc(arc: _ConvexOffsetArc, angle: float) -> Vec2:
    return Vec2(
        x=arc.center.x + arc.radius * math.cos(angle),
        y=arc.center.y + arc.radius * math.sin(angle),
    )


def _unwrapped_arc_angle(arc: _ConvexOffsetArc, angle: float) -> float:
    return arc.start_angle + (angle - arc.start_angle) % math.tau


def _arc_contains_angle(arc: _ConvexOffsetArc, angle: float) -> bool:
    unwrapped = _unwrapped_arc_angle(arc, angle)
    return unwrapped <= arc.end_angle + _GEOMETRY_TOLERANCE


def _bisect_to_adjacent_outside(
    inside: float,
    outside: float,
    is_outside,
) -> float:
    """Bisect an inside/outside interval to adjacent floating-point values."""
    if is_outside(inside) or not is_outside(outside):
        raise ValueError("invalid inside/outside bracket")
    while math.nextafter(inside, outside) != outside:
        midpoint = inside + (outside - inside) / 2.0
        if midpoint == inside:
            midpoint = math.nextafter(inside, outside)
        elif midpoint == outside:
            midpoint = math.nextafter(outside, inside)
        if is_outside(midpoint):
            outside = midpoint
        else:
            inside = midpoint
    return outside


def _segment_circle_intersections(
    start: Vec2,
    end: Vec2,
    *,
    center: Vec2,
    radius: float,
) -> tuple[tuple[float, Vec2], ...]:
    reversed_input = (end.x, end.y) < (start.x, start.y)
    if reversed_input:
        start, end = end, start
    dx, dy = end.x - start.x, end.y - start.y
    fx, fy = start.x - center.x, start.y - center.y
    squared_length = dx * dx + dy * dy
    if squared_length <= _EPS:
        return ()
    unclamped_closest = -(fx * dx + fy * dy) / squared_length
    closest_parameter = min(1.0, max(0.0, unclamped_closest))

    def point_at(parameter: float) -> Vec2:
        return Vec2(x=start.x + parameter * dx, y=start.y + parameter * dy)

    def distance_at(parameter: float) -> float:
        point = point_at(parameter)
        return math.hypot(point.x - center.x, point.y - center.y)

    def outside_endpoint(inside: float, outside: float) -> float:
        return _bisect_to_adjacent_outside(
            inside,
            outside,
            lambda parameter: distance_at(parameter) >= radius,
        )

    linear = 2.0 * (fx * dx + fy * dy)
    constant = fx * fx + fy * fy - radius * radius
    discriminant = linear * linear - 4.0 * squared_length * constant
    guard = 64.0 * math.ulp(
        max(1.0, abs(linear * linear), abs(4.0 * squared_length * constant))
    )
    if discriminant < -guard:
        return ()
    closest_distance = distance_at(closest_parameter)
    if closest_distance > radius and not 0.0 <= unclamped_closest <= 1.0:
        return ()
    if closest_distance >= radius:
        parameters = (closest_parameter,)
    else:
        parameters = tuple(
            outside_endpoint(closest_parameter, endpoint)
            for endpoint in (0.0, 1.0)
            if endpoint != closest_parameter and distance_at(endpoint) >= radius
        )
    intersections: list[tuple[float, Vec2]] = []
    for parameter in parameters:
        point = point_at(parameter)
        if math.hypot(point.x - center.x, point.y - center.y) < radius:
            raise ArithmeticError("circle intersection was not enclosed")
        original_parameter = 1.0 - parameter if reversed_input else parameter
        if not any(
            math.isclose(
                original_parameter,
                existing[0],
                rel_tol=0.0,
                abs_tol=_EPS,
            )
            for existing in intersections
        ):
            intersections.append((original_parameter, point))
    return tuple(intersections)


def _geometry_boundary_segments(
    geometry: BaseGeometry,
) -> tuple[tuple[Vec2, Vec2], ...]:
    segments: list[tuple[Vec2, Vec2]] = []
    for ring in _rings(geometry):
        coordinates = tuple((float(x), float(y)) for x, y in ring.coords)
        segments.extend(
            (Vec2(x=start[0], y=start[1]), Vec2(x=end[0], y=end[1]))
            for start, end in pairwise(coordinates)
            if start != end
        )
    return tuple(segments)


def _point_is_within_geometry_tolerance(
    point: Vec2,
    geometry: BaseGeometry,
) -> bool:
    return geometry.buffer(_GEOMETRY_TOLERANCE).covers(Point(point.x, point.y))


def _intersection_geometry_candidates(
    origin: Vec2,
    geometry: BaseGeometry,
) -> tuple[Vec2, ...]:
    if geometry.is_empty:
        return ()
    geometries = getattr(geometry, "geoms", (geometry,))
    candidates: list[Vec2] = []
    for item in geometries:
        if isinstance(item, Point):
            candidates.append(Vec2(x=float(item.x), y=float(item.y)))
        elif hasattr(item, "coords"):
            coordinates = tuple((float(x), float(y)) for x, y in item.coords)
            candidates.extend(Vec2(x=x, y=y) for x, y in coordinates)
            candidates.extend(
                point_to_segment(
                    origin,
                    Vec2(x=start[0], y=start[1]),
                    Vec2(x=end[0], y=end[1]),
                ).point
                for start, end in pairwise(coordinates)
            )
        else:
            candidates.extend(_intersection_geometry_candidates(origin, item))
    return tuple(candidates)


def _deduplicate_points(points: list[Vec2]) -> tuple[Vec2, ...]:
    unique: dict[tuple[float, float], Vec2] = {}
    for point in points:
        scale = max(1.0, abs(point.x), abs(point.y))
        zero_tolerance = 64.0 * math.ulp(scale)
        canonical = Vec2(
            x=0.0 if abs(point.x) <= zero_tolerance else point.x,
            y=0.0 if abs(point.y) <= zero_tolerance else point.y,
        )
        unique.setdefault(
            (round(canonical.x, 12), round(canonical.y, 12)),
            canonical,
        )
    return tuple(unique[key] for key in sorted(unique))


def _enclose_far_candidate(
    candidate: Vec2,
    configuration: Polygon,
    radius: float,
) -> Vec2 | None:
    candidate_point = Point(candidate.x, candidate.y)
    if candidate_point.distance(configuration) >= radius:
        return candidate
    nearest = nearest_points(configuration, candidate_point)[0]
    nearest_x, nearest_y = float(nearest.x), float(nearest.y)
    dx, dy = candidate.x - nearest_x, candidate.y - nearest_y
    distance = math.hypot(dx, dy)
    if distance <= _EPS:
        return None
    unit_x, unit_y = dx / distance, dy / distance
    excess = max(radius - distance, math.ulp(radius))
    while True:
        outside_distance = radius + excess
        if not math.isfinite(outside_distance):
            return None
        outside = Vec2(
            x=nearest_x + outside_distance * unit_x,
            y=nearest_y + outside_distance * unit_y,
        )
        if Point(outside.x, outside.y).distance(configuration) >= radius:
            break
        next_excess = excess * 2.0
        if not math.isfinite(next_excess) or next_excess == excess:
            return None
        excess = next_excess

    def point_at(parameter: float) -> Vec2:
        return Vec2(
            x=candidate.x + parameter * (outside.x - candidate.x),
            y=candidate.y + parameter * (outside.y - candidate.y),
        )

    def is_enclosed(parameter: float) -> bool:
        point = point_at(parameter)
        return Point(point.x, point.y).distance(configuration) >= radius

    parameter = _bisect_to_adjacent_outside(
        0.0,
        1.0,
        is_enclosed,
    )
    return point_at(parameter)


def _exact_far_candidates(
    origin: Vec2,
    physical: BaseGeometry,
    configuration: Polygon,
    radius: float,
) -> tuple[Vec2, ...]:
    """Return a complete finite candidate set for an exact convex FAR optimum."""
    if physical.is_empty:
        return ()
    primitives = _convex_offset_primitives(configuration, radius)
    physical_edges = _geometry_boundary_segments(physical)
    candidates: list[Vec2] = []

    if (
        _point_is_within_geometry_tolerance(origin, physical)
        and Point(origin.x, origin.y).distance(configuration) >= radius
    ):
        candidates.append(origin)

    for segment in primitives.segments:
        intersection = LineString(
            ((segment.start.x, segment.start.y), (segment.end.x, segment.end.y))
        ).intersection(physical)
        candidates.extend(_intersection_geometry_candidates(origin, intersection))

    for arc in primitives.arcs:
        events: dict[float, tuple[float, Vec2]] = {
            round(arc.start_angle, 15): (
                arc.start_angle,
                _point_on_arc(arc, arc.start_angle),
            ),
            round(arc.end_angle, 15): (
                arc.end_angle,
                _point_on_arc(arc, arc.end_angle),
            ),
        }
        for start, end in physical_edges:
            for _, point in _segment_circle_intersections(
                start,
                end,
                center=arc.center,
                radius=arc.radius,
            ):
                angle = math.atan2(point.y - arc.center.y, point.x - arc.center.x)
                if _arc_contains_angle(arc, angle):
                    unwrapped = _unwrapped_arc_angle(arc, angle)
                    events.setdefault(round(unwrapped, 15), (unwrapped, point))
        ordered_events = tuple(events[key] for key in sorted(events))
        for _, point in ordered_events:
            if _point_is_within_geometry_tolerance(point, physical):
                candidates.append(point)
        for start_event, end_event in pairwise(ordered_events):
            start_angle, start_point = start_event
            end_angle, end_point = end_event
            if end_angle - start_angle <= _EPS:
                continue
            midpoint = _point_on_arc(arc, (start_angle + end_angle) / 2.0)
            if not _point_is_within_geometry_tolerance(midpoint, physical):
                continue
            candidates.extend((start_point, end_point))
            if (
                math.hypot(
                    origin.x - arc.center.x,
                    origin.y - arc.center.y,
                )
                > _EPS
            ):
                radial_angle = _unwrapped_arc_angle(
                    arc,
                    math.atan2(
                        origin.y - arc.center.y,
                        origin.x - arc.center.x,
                    ),
                )
                if start_angle <= radial_angle <= end_angle:
                    candidates.append(_point_on_arc(arc, radial_angle))

    for start, end in physical_edges:
        parameters = [0.0, 1.0]
        edge_line = LineString(((start.x, start.y), (end.x, end.y)))
        for segment in primitives.segments:
            intersection = edge_line.intersection(
                LineString(
                    (
                        (segment.start.x, segment.start.y),
                        (segment.end.x, segment.end.y),
                    )
                )
            )
            for point in _intersection_geometry_candidates(origin, intersection):
                edge_length = math.hypot(end.x - start.x, end.y - start.y)
                if edge_length > _EPS:
                    parameters.append(
                        math.hypot(point.x - start.x, point.y - start.y) / edge_length
                    )
        for arc in primitives.arcs:
            parameters.extend(
                parameter
                for parameter, point in _segment_circle_intersections(
                    start,
                    end,
                    center=arc.center,
                    radius=arc.radius,
                )
                if _arc_contains_angle(
                    arc,
                    math.atan2(point.y - arc.center.y, point.x - arc.center.x),
                )
            )
        parameters = sorted(
            {round(min(1.0, max(0.0, parameter)), 15) for parameter in parameters}
        )
        for lower, upper in pairwise(parameters):
            if upper - lower <= _EPS:
                continue
            midpoint_parameter = (lower + upper) / 2.0
            midpoint = Vec2(
                x=start.x + midpoint_parameter * (end.x - start.x),
                y=start.y + midpoint_parameter * (end.y - start.y),
            )
            if Point(midpoint.x, midpoint.y).distance(configuration) + _EPS < radius:
                continue
            interval_start = Vec2(
                x=start.x + lower * (end.x - start.x),
                y=start.y + lower * (end.y - start.y),
            )
            interval_end = Vec2(
                x=start.x + upper * (end.x - start.x),
                y=start.y + upper * (end.y - start.y),
            )
            candidates.extend(
                (
                    interval_start,
                    interval_end,
                    point_to_segment(origin, interval_start, interval_end).point,
                )
            )

    enclosed = [
        enclosed_candidate
        for candidate in candidates
        if (
            enclosed_candidate := _enclose_far_candidate(
                candidate,
                configuration,
                radius,
            )
        )
        is not None
    ]
    return _deduplicate_points(enclosed)


def _attained_candidate_neighbor(
    case: StressCaseDraft,
    point: Vec2,
    physical: BaseGeometry,
    strict_boundaries: tuple[BaseGeometry, ...],
) -> Vec2 | None:
    """Repair only finite-precision misses around a non-strict exact candidate."""
    shapely_point = Point(point.x, point.y)

    def is_on_strict_boundary(candidate: Vec2) -> bool:
        candidate_point = Point(candidate.x, candidate.y)
        return any(
            not boundary.is_empty and boundary.distance(candidate_point) <= _EPS
            for boundary in strict_boundaries
        )

    if is_on_strict_boundary(point):
        return None

    def is_valid(candidate: Vec2) -> bool:
        return (
            not is_on_strict_boundary(candidate)
            and physical.covers(Point(candidate.x, candidate.y))
            and _candidate_projection_is_valid(case, candidate)
            and _candidate_preserves_relations(case, candidate)
            and _candidate_satisfies_target(case, candidate)
        )

    nearest_physical = nearest_points(physical, shapely_point)[0]
    projected = Vec2(x=float(nearest_physical.x), y=float(nearest_physical.y))
    repairs: list[Vec2] = []
    if is_valid(projected):
        repairs.append(projected)

    for start, end in _geometry_boundary_segments(physical):
        projection = point_to_segment(point, start, end)
        if projection.distance > _GEOMETRY_TOLERANCE:
            continue
        base = projection.point
        if is_valid(base):
            repairs.append(base)
            continue
        dx, dy = end.x - start.x, end.y - start.y
        squared_length = dx * dx + dy * dy
        if squared_length <= _EPS:
            continue
        base_parameter = (
            (base.x - start.x) * dx + (base.y - start.y) * dy
        ) / squared_length
        base_parameter = min(1.0, max(0.0, base_parameter))

        def point_at(
            parameter: float,
            segment_start: Vec2 = start,
            segment_dx: float = dx,
            segment_dy: float = dy,
        ) -> Vec2:
            return Vec2(
                x=segment_start.x + parameter * segment_dx,
                y=segment_start.y + parameter * segment_dy,
            )

        rebuilt_base = point_at(base_parameter)
        if is_valid(rebuilt_base):
            repairs.append(rebuilt_base)
            continue
        if _candidate_satisfies_target(case, rebuilt_base):
            continue

        for endpoint_parameter in (0.0, 1.0):
            if endpoint_parameter == base_parameter:
                continue
            endpoint = point_at(endpoint_parameter)
            if not _candidate_satisfies_target(case, endpoint):
                continue
            parameter = _bisect_to_adjacent_outside(
                base_parameter,
                endpoint_parameter,
                lambda value: _candidate_satisfies_target(case, point_at(value)),
            )
            repaired = point_at(parameter)
            if is_valid(repaired):
                repairs.append(repaired)

    if not repairs and physical.covers(shapely_point):
        subject = case.scene.object_by_id(case.intervention.subject_id)
        reference = case.scene.object_by_id(case.intervention.reference_id)
        configuration = _configuration_polygon(subject, reference)
        nearest_configuration = nearest_points(configuration, shapely_point)[0]
        nearest_x = float(nearest_configuration.x)
        nearest_y = float(nearest_configuration.y)
        dx, dy = point.x - nearest_x, point.y - nearest_y
        distance = math.hypot(dx, dy)
        if distance > _EPS:
            unit_x, unit_y = dx / distance, dy / distance
            excess = max(RelationEngine.FAR_METERS - distance, math.ulp(distance))
            while True:
                outside_distance = RelationEngine.FAR_METERS + excess
                if not math.isfinite(outside_distance):
                    break
                outside = Vec2(
                    x=nearest_x + outside_distance * unit_x,
                    y=nearest_y + outside_distance * unit_y,
                )
                if _candidate_satisfies_target(case, outside):
                    parameter = _bisect_to_adjacent_outside(
                        0.0,
                        1.0,
                        lambda value, outside_candidate=outside: (
                            _candidate_satisfies_target(
                                case,
                                Vec2(
                                    x=point.x + value * (outside_candidate.x - point.x),
                                    y=point.y + value * (outside_candidate.y - point.y),
                                ),
                            )
                        ),
                    )
                    repaired = Vec2(
                        x=point.x + parameter * (outside.x - point.x),
                        y=point.y + parameter * (outside.y - point.y),
                    )
                    if is_valid(repaired):
                        repairs.append(repaired)
                    break
                next_excess = excess * 2.0
                if not math.isfinite(next_excess) or next_excess == excess:
                    break
                excess = next_excess

    if not repairs:
        return None
    return min(
        repairs,
        key=lambda candidate: (
            math.hypot(candidate.x - point.x, candidate.y - point.y),
            candidate.x,
            candidate.y,
        ),
    )


def support_center_rectangle(
    subject: SceneObject,
    support: SceneObject,
) -> tuple[float, float, float, float]:
    """Return the exact subject-position rectangle fully supported by an axis-aligned OBB."""
    footprint = obb_footprint(support.obb)
    bounds = footprint.bounds
    scale = max(1.0, *(abs(value) for value in bounds))
    axis_tolerance = 64.0 * math.ulp(scale)
    min_x, min_y, max_x, max_y = bounds
    vertices_are_axis_aligned = all(
        min(abs(float(x) - min_x), abs(float(x) - max_x)) <= axis_tolerance
        and min(abs(float(y) - min_y), abs(float(y) - max_y)) <= axis_tolerance
        for x, y in footprint.exterior.coords[:-1]
    )
    if footprint.is_empty or not vertices_are_axis_aligned:
        raise ValueError("support footprint must be an axis-aligned rectangle")
    relative = _relative_subject_vertices(subject)
    lower_x = min_x - min(point.x for point in relative)
    lower_y = min_y - min(point.y for point in relative)
    upper_x = max_x - max(point.x for point in relative)
    upper_y = max_y - max(point.y for point in relative)
    if lower_x > upper_x or lower_y > upper_y:
        raise ValueError("subject footprint cannot fit on support")

    def canonical_zero(value: float) -> float:
        return 0.0 if abs(value) <= axis_tolerance else value

    return (
        canonical_zero(lower_x),
        canonical_zero(lower_y),
        canonical_zero(upper_x),
        canonical_zero(upper_y),
    )


def _geometry_vertices(geometry: BaseGeometry) -> tuple[Point, ...]:
    if geometry.is_empty:
        return ()
    geometries = getattr(geometry, "geoms", (geometry,))
    points: list[Point] = []
    for item in geometries:
        if isinstance(item, Polygon):
            points.extend(
                Point(float(x), float(y)) for x, y in item.exterior.coords[:-1]
            )
            for ring in item.interiors:
                points.extend(Point(float(x), float(y)) for x, y in ring.coords[:-1])
        elif hasattr(item, "coords"):
            points.extend(Point(float(x), float(y)) for x, y in item.coords)
    return tuple(points)


def maximum_ground_gap(
    subject: SceneObject,
    reference: SceneObject,
    center_locus: BaseGeometry,
) -> float:
    """Maximize exact OBB ground gap over every vertex of a polygonal center locus."""
    configuration = _configuration_polygon(subject, reference)
    vertices = _geometry_vertices(center_locus)
    if not vertices:
        return 0.0
    return max(float(point.distance(configuration)) for point in vertices)


def _rectangular_center_locus(
    container: BaseGeometry,
    subject: SceneObject,
) -> BaseGeometry:
    if container.is_empty or not container.equals(box(*container.bounds)):
        raise ValueError("stress oracle container must be an axis-aligned rectangle")
    min_x, min_y, max_x, max_y = container.bounds
    relative = _relative_subject_vertices(subject)
    lower_x = min_x - min(point.x for point in relative)
    lower_y = min_y - min(point.y for point in relative)
    upper_x = max_x - max(point.x for point in relative)
    upper_y = max_y - max(point.y for point in relative)
    if lower_x > upper_x or lower_y > upper_y:
        return Polygon()
    return box(lower_x, lower_y, upper_x, upper_y)


def _physical_center_locus(case: StressCaseDraft) -> BaseGeometry:
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    locus = _rectangular_center_locus(room, subject)
    support_id = subject.support_object_id
    if support_id is not None:
        support = scene.object_by_id(support_id)
        locus = locus.intersection(box(*support_center_rectangle(subject, support)))
    for obstacle in sorted(scene.objects, key=lambda item: item.object_id):
        if obstacle.object_id in {subject.object_id, support_id}:
            continue
        subject_bottom = subject.obb.center.z - subject.obb.extent.z / 2.0
        subject_top = subject.obb.center.z + subject.obb.extent.z / 2.0
        obstacle_bottom = obstacle.obb.center.z - obstacle.obb.extent.z / 2.0
        obstacle_top = obstacle.obb.center.z + obstacle.obb.extent.z / 2.0
        z_overlap = min(subject_top, obstacle_top) - max(
            subject_bottom,
            obstacle_bottom,
        )
        if z_overlap <= _GEOMETRY_TOLERANCE:
            continue
        locus = locus.difference(_configuration_polygon(subject, obstacle))
    return locus


def _camera_affine_coefficients(
    camera: Camera,
    row: int,
    z: float,
) -> tuple[float, float, float]:
    start = row * 4
    matrix = camera.world_to_camera
    return (
        matrix[start],
        matrix[start + 1],
        matrix[start + 2] * z + matrix[start + 3],
    )


def _linear_expression_half_space(
    geometry: BaseGeometry,
    coefficients: tuple[float, float, float],
    *,
    keep_greater: bool,
) -> BaseGeometry:
    x_coefficient, y_coefficient, constant = coefficients
    sign = 1.0 if keep_greater else -1.0
    if math.hypot(x_coefficient, y_coefficient) <= _EPS:
        return geometry if sign * constant >= 0.0 else Polygon()
    normal = Vec2(x=sign * x_coefficient, y=sign * y_coefficient)
    offset = -sign * constant
    return _intersect_half_space(geometry, normal, offset)


def _subject_projection_residuals(
    case: StressCaseDraft,
) -> tuple[float, float, float] | None:
    """Independently recover observed-minus-analytic subject calibration."""
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    camera = scene.camera_by_id(spec.camera_id)
    subject_view = subject.views.get(spec.camera_id)
    if subject_view is None:
        return None

    matrix = camera.world_to_camera
    position = subject.position

    def source_camera_coordinate(row: int) -> float:
        start = 4 * row
        return (
            matrix[start] * position.x
            + matrix[start + 1] * position.y
            + matrix[start + 2] * position.z
            + matrix[start + 3]
        )

    source_camera_x = source_camera_coordinate(0)
    source_camera_y = source_camera_coordinate(1)
    source_depth = source_camera_coordinate(2)
    if source_depth <= 0.0:
        return None
    fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
    source_center_x = fx * source_camera_x / source_depth + cx
    source_center_y = cy - fy * source_camera_y / source_depth
    bbox = subject_view.bbox
    observed_center_y = (bbox.ymin + bbox.ymax) / 2.0
    residuals = (
        bbox.center_x - source_center_x,
        observed_center_y - source_center_y,
        subject_view.camera_depth - source_depth,
    )
    calibration_values = (
        source_center_x,
        source_center_y,
        source_depth,
        *residuals,
    )
    if not all(math.isfinite(value) for value in calibration_values):
        return None
    return residuals


def _visibility_center_locus(
    case: StressCaseDraft,
    geometry: BaseGeometry,
) -> BaseGeometry:
    """Return a conservative positive-depth outer for reachable subject centers.

    Product visibility permits a partially clipped bounding box, whose exact
    clipped-area constraint is nonlinear when both image axes clip.  Keep this
    locus conservative and enforce that constraint in the final candidate gate.
    """
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    camera = scene.camera_by_id(spec.camera_id)
    depth = _camera_affine_coefficients(camera, 2, subject.position.z)
    return _linear_expression_half_space(
        geometry,
        depth,
        keep_greater=True,
    )


def _horizontal_relation_half_space(
    case: StressCaseDraft,
    stationary: SceneObject,
    relation: Relation,
    *,
    include_comparison_tolerance: bool = True,
) -> tuple[Vec2, float]:
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    camera = scene.camera_by_id(spec.camera_id)
    reference_view = stationary.views[spec.camera_id]
    fx, _, cx, _, _, _, _, _, _ = camera.intrinsics
    direction = 1.0 if relation is Relation.RIGHT else -1.0
    comparison_tolerance = (
        _OPERATIONAL_COMPARISON_TOLERANCE if include_comparison_tolerance else 0.0
    )
    residuals = _subject_projection_residuals(case)
    if residuals is None:
        raise ValueError("subject projection calibration is invalid")
    horizontal_residual, _, _ = residuals
    threshold = camera.width * RelationEngine.LEFT_RIGHT_FRACTION
    target_u = reference_view.bbox.center_x + direction * (
        threshold - comparison_tolerance
    )
    pixel_offset = target_u - cx - horizontal_residual
    matrix = camera.world_to_camera
    z = subject.position.z
    normal = Vec2(
        x=direction * (fx * matrix[0] - pixel_offset * matrix[8]),
        y=direction * (fx * matrix[1] - pixel_offset * matrix[9]),
    )
    constant = direction * (
        fx * (matrix[2] * z + matrix[3]) - pixel_offset * (matrix[10] * z + matrix[11])
    )
    return normal, -constant


def _depth_relation_half_space(
    case: StressCaseDraft,
    stationary: SceneObject,
    relation: Relation,
    *,
    include_comparison_tolerance: bool = True,
) -> tuple[Vec2, float]:
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    camera = scene.camera_by_id(spec.camera_id)
    reference_depth = stationary.views[spec.camera_id].camera_depth
    direction = 1.0 if relation is Relation.BEHIND else -1.0
    comparison_tolerance = (
        _OPERATIONAL_COMPARISON_TOLERANCE if include_comparison_tolerance else 0.0
    )
    residuals = _subject_projection_residuals(case)
    if residuals is None:
        raise ValueError("subject projection calibration is invalid")
    _, _, depth_residual = residuals
    threshold = RelationEngine.FRONT_BEHIND_METERS - comparison_tolerance
    target_depth = reference_depth + direction * threshold - depth_residual
    matrix = camera.world_to_camera
    z = subject.position.z
    normal = Vec2(x=direction * matrix[8], y=direction * matrix[9])
    constant = direction * (matrix[10] * z + matrix[11] - target_depth)
    return normal, -constant


def _relation_locus(
    case: StressCaseDraft,
    geometry: BaseGeometry,
    stationary: SceneObject,
    relation: Relation,
    *,
    expand_boundary: bool = True,
) -> BaseGeometry:
    if relation.axis is RelationAxis.HORIZONTAL:
        normal, offset = _horizontal_relation_half_space(case, stationary, relation)
        return _intersect_half_space(
            geometry,
            normal,
            offset,
            expand_boundary=expand_boundary,
        )
    if relation.axis is RelationAxis.DEPTH:
        normal, offset = _depth_relation_half_space(case, stationary, relation)
        return _intersect_half_space(
            geometry,
            normal,
            offset,
            expand_boundary=expand_boundary,
        )
    subject = case.scene.object_by_id(case.intervention.subject_id)
    configuration = _configuration_polygon(subject, stationary)
    radius = (
        RelationEngine.NEAR_METERS
        if relation is Relation.NEAR
        else RelationEngine.FAR_METERS
    )
    inner, outer = _distance_buffer_envelopes(configuration, radius)
    if relation is Relation.NEAR:
        return geometry.intersection(outer)
    return geometry.difference(inner).union(geometry.intersection(inner.boundary))


def _ambiguous_distance_outer_locus(
    case: StressCaseDraft,
    geometry: BaseGeometry,
    stationary: SceneObject,
) -> tuple[BaseGeometry, tuple[BaseGeometry, BaseGeometry]]:
    subject = case.scene.object_by_id(case.intervention.subject_id)
    configuration = _configuration_polygon(subject, stationary)
    near_inner, _ = _distance_buffer_envelopes(
        configuration,
        RelationEngine.NEAR_METERS,
    )
    _, far_outer = _distance_buffer_envelopes(
        configuration,
        RelationEngine.FAR_METERS,
    )
    outside_near_inner = geometry.difference(near_inner).union(
        geometry.intersection(near_inner.boundary)
    )
    return outside_near_inner.intersection(far_outer), (
        near_inner.boundary,
        far_outer.boundary,
    )


def _relation_boundary(
    case: StressCaseDraft,
    geometry: BaseGeometry,
    stationary: SceneObject,
    relation: Relation,
) -> BaseGeometry:
    if relation.axis is RelationAxis.HORIZONTAL:
        normal, offset = _horizontal_relation_half_space(case, stationary, relation)
        return _half_space_boundary_line(geometry, normal, offset)
    if relation.axis is RelationAxis.DEPTH:
        normal, offset = _depth_relation_half_space(case, stationary, relation)
        return _half_space_boundary_line(geometry, normal, offset)
    subject = case.scene.object_by_id(case.intervention.subject_id)
    radius = (
        RelationEngine.NEAR_METERS
        if relation is Relation.NEAR
        else RelationEngine.FAR_METERS
    )
    return (
        _configuration_polygon(subject, stationary)
        .buffer(
            radius,
            quad_segs=_BUFFER_QUAD_SEGS,
        )
        .boundary
    )


def _preserved_relations_center_locus(
    case: StressCaseDraft,
    geometry: BaseGeometry,
) -> tuple[BaseGeometry, tuple[BaseGeometry, ...]]:
    """Intersect the exact relation-label preservation contract used by Verifier."""
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    engine = RelationEngine()
    result = geometry
    strict_boundaries: list[BaseGeometry] = []
    for stationary in sorted(scene.objects, key=lambda item: item.object_id):
        if stationary.object_id == subject.object_id:
            continue
        stationary_view = stationary.views.get(spec.camera_id)
        subject_view = subject.views.get(spec.camera_id)
        permanently_invisible = (
            subject_view is None
            or stationary_view is None
            or subject_view.visible_fraction < engine.MIN_VISIBLE_FRACTION
            or subject_view.image_area_fraction < engine.MIN_IMAGE_AREA_FRACTION
            or subject_view.truncated_fraction > engine.MAX_TRUNCATED_FRACTION
            or stationary_view.visible_fraction < engine.MIN_VISIBLE_FRACTION
            or stationary_view.image_area_fraction < engine.MIN_IMAGE_AREA_FRACTION
            or stationary_view.truncated_fraction > engine.MAX_TRUNCATED_FRACTION
        )
        labels = engine.pair_labels(
            scene,
            subject.object_id,
            stationary.object_id,
            spec.camera_id,
        )
        for axis in RelationAxis:
            if (
                stationary.object_id == spec.reference_id
                and axis is spec.relation_after.axis
            ):
                continue
            axis_labels = tuple(
                relation for relation in _AXIS_RELATIONS[axis] if relation in labels
            )
            if len(axis_labels) == 1:
                result = _relation_locus(
                    case,
                    result,
                    stationary,
                    axis_labels[0],
                    expand_boundary=False,
                )
            elif len(axis_labels) > 1:
                raise ValueError("before scene has contradictory relation labels")
            elif not permanently_invisible:
                if axis is RelationAxis.DISTANCE:
                    result, distance_boundaries = _ambiguous_distance_outer_locus(
                        case,
                        result,
                        stationary,
                    )
                    strict_boundaries.extend(distance_boundaries)
                else:
                    first, second = _AXIS_RELATIONS[axis]
                    labelled = _relation_locus(case, result, stationary, first).union(
                        _relation_locus(case, result, stationary, second)
                    )
                    result = result.difference(labelled)
                    strict_boundaries.extend(
                        (
                            _relation_boundary(case, geometry, stationary, first),
                            _relation_boundary(case, geometry, stationary, second),
                        )
                    )
            if result.is_empty:
                return result, tuple(strict_boundaries)
    return result, tuple(strict_boundaries)


def _horizontal_half_space(case: StressCaseDraft) -> tuple[Vec2, float]:
    reference = case.scene.object_by_id(case.intervention.reference_id)
    return _horizontal_relation_half_space(
        case,
        reference,
        case.intervention.relation_after,
    )


def _depth_half_space(case: StressCaseDraft) -> tuple[Vec2, float]:
    reference = case.scene.object_by_id(case.intervention.reference_id)
    return _depth_relation_half_space(
        case,
        reference,
        case.intervention.relation_after,
    )


def _half_space_polygon(
    geometry: BaseGeometry,
    normal: Vec2,
    offset: float,
    *,
    expand_boundary: bool = True,
) -> Polygon:
    norm = math.hypot(normal.x, normal.y)
    if norm <= _EPS:
        raise ValueError("target relation has a zero half-space normal")
    unit_x, unit_y = normal.x / norm, normal.y / norm
    tangent_x, tangent_y = -unit_y, unit_x
    if geometry.is_empty:
        return Polygon()
    min_x, min_y, max_x, max_y = geometry.bounds
    magnitude = max(
        1.0,
        abs(offset),
        *(
            abs(normal.x * x) + abs(normal.y * y)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
        ),
    )
    guarded_offset = offset - 32.0 * math.ulp(magnitude) if expand_boundary else offset
    boundary_x = normal.x * guarded_offset / (norm * norm)
    boundary_y = normal.y * guarded_offset / (norm * norm)
    scale = 8.0 * max(
        1.0,
        abs(min_x),
        abs(min_y),
        abs(max_x),
        abs(max_y),
        abs(boundary_x),
        abs(boundary_y),
    )
    return Polygon(
        [
            (boundary_x - scale * tangent_x, boundary_y - scale * tangent_y),
            (boundary_x + scale * tangent_x, boundary_y + scale * tangent_y),
            (
                boundary_x + scale * tangent_x + scale * unit_x,
                boundary_y + scale * tangent_y + scale * unit_y,
            ),
            (
                boundary_x - scale * tangent_x + scale * unit_x,
                boundary_y - scale * tangent_y + scale * unit_y,
            ),
        ]
    )


def _intersect_half_space(
    geometry: BaseGeometry,
    normal: Vec2,
    offset: float,
    *,
    expand_boundary: bool = True,
) -> BaseGeometry:
    if math.hypot(normal.x, normal.y) <= _EPS:
        return geometry if offset <= 0.0 else Polygon()
    return geometry.intersection(
        _half_space_polygon(
            geometry,
            normal,
            offset,
            expand_boundary=expand_boundary,
        )
    )


def _half_space_boundary_line(
    geometry: BaseGeometry,
    normal: Vec2,
    offset: float,
) -> LineString:
    norm = math.hypot(normal.x, normal.y)
    if norm <= _EPS:
        return LineString()
    unit_x, unit_y = normal.x / norm, normal.y / norm
    tangent_x, tangent_y = -unit_y, unit_x
    boundary_x = normal.x * offset / (norm * norm)
    boundary_y = normal.y * offset / (norm * norm)
    bounds = geometry.bounds if not geometry.is_empty else (0.0, 0.0, 0.0, 0.0)
    scale = 16.0 * max(
        1.0, *(abs(value) for value in bounds), abs(boundary_x), abs(boundary_y)
    )
    return LineString(
        (
            (boundary_x - scale * tangent_x, boundary_y - scale * tangent_y),
            (boundary_x + scale * tangent_x, boundary_y + scale * tangent_y),
        )
    )


def _target_locus(
    case: StressCaseDraft,
    physical: BaseGeometry,
) -> tuple[BaseGeometry, Vec2 | None, float]:
    relation = case.intervention.relation_after
    if relation in {Relation.LEFT, Relation.RIGHT}:
        normal, offset = _horizontal_half_space(case)
        target = _intersect_half_space(physical, normal, offset)
        return target, normal, offset
    if relation in {Relation.FRONT, Relation.BEHIND}:
        normal, offset = _depth_half_space(case)
        target = _intersect_half_space(physical, normal, offset)
        return target, normal, offset
    if relation is not Relation.FAR:
        raise ValueError("stress oracle supports only opposite target relations")
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    reference = scene.object_by_id(case.intervention.reference_id)
    configuration = _configuration_polygon(subject, reference)
    excluded, _ = _distance_buffer_envelopes(
        configuration,
        RelationEngine.FAR_METERS,
    )
    return physical.difference(excluded), None, RelationEngine.FAR_METERS


def _rings(geometry: BaseGeometry):
    geometries = getattr(geometry, "geoms", (geometry,))
    for item in geometries:
        if isinstance(item, Polygon):
            yield item.exterior
            yield from item.interiors
        elif hasattr(item, "coords"):
            yield item


def _nearest_geometry_points(
    origin: Vec2,
    geometry: BaseGeometry,
    *,
    is_valid: Callable[[Vec2], bool] | None = None,
    strict_boundaries: tuple[BaseGeometry, ...] = (),
) -> tuple[Vec2, ...]:
    point = Point(origin.x, origin.y)
    if geometry.covers(point):
        return (origin,)
    candidates: list[PointProjection] = []
    for ring in _rings(geometry):
        coordinates = tuple((float(x), float(y)) for x, y in ring.coords)
        for start, end in pairwise(coordinates):
            candidates.append(
                point_to_segment(
                    origin,
                    Vec2(x=start[0], y=start[1]),
                    Vec2(x=end[0], y=end[1]),
                )
            )
    if not candidates:
        raise ValueError("target locus has no boundary")
    minimum = min(candidate.distance for candidate in candidates)
    unique: set[tuple[float, float]] = set()
    for candidate in candidates:
        if not math.isclose(candidate.distance, minimum, rel_tol=0.0, abs_tol=_EPS):
            continue
        candidate_geometry = Point(candidate.point.x, candidate.point.y)
        if any(
            not boundary.is_empty and boundary.distance(candidate_geometry) <= _EPS
            for boundary in strict_boundaries
        ):
            continue
        rounded_x = round(candidate.point.x, 12)
        rounded_y = round(candidate.point.y, 12)
        # Decimal canonicalization can round an exact operational boundary to
        # the excluded side (for example .3999999999875 -> .399999999987).
        # Select the nearest adjacent 12-place point still enclosed by the
        # independently derived locus and accepted by every caller gate.
        adjacent = [
            (x, y)
            for x in (
                round(rounded_x - 1e-12, 12),
                rounded_x,
                round(rounded_x + 1e-12, 12),
            )
            for y in (
                round(rounded_y - 1e-12, 12),
                rounded_y,
                round(rounded_y + 1e-12, 12),
            )
            if geometry.covers(Point(x, y))
            and (is_valid is None or is_valid(Vec2(x=x, y=y)))
        ]
        if adjacent:
            unique.add(
                min(
                    adjacent,
                    key=lambda point: (
                        math.hypot(
                            point[0] - candidate.point.x,
                            point[1] - candidate.point.y,
                        ),
                        point,
                    ),
                )
            )
    return tuple(Vec2(x=x, y=y) for x, y in sorted(unique))


def _maximum_half_space_value(
    physical: BaseGeometry,
    normal: Vec2,
    offset: float,
) -> tuple[float, float]:
    norm = math.hypot(normal.x, normal.y)
    vertices = _geometry_vertices(physical)
    maximum = (
        max((normal.x * point.x + normal.y * point.y) / norm for point in vertices)
        if vertices
        else 0.0
    )
    return maximum, offset / norm


def _positive_depth_boundary(
    case: StressCaseDraft,
    geometry: BaseGeometry,
) -> BaseGeometry:
    subject = case.scene.object_by_id(case.intervention.subject_id)
    camera = case.scene.camera_by_id(case.intervention.camera_id)
    x_coefficient, y_coefficient, constant = _camera_affine_coefficients(
        camera,
        2,
        subject.position.z,
    )
    if math.hypot(x_coefficient, y_coefficient) <= _EPS:
        return LineString()
    return _half_space_boundary_line(
        geometry,
        Vec2(x=x_coefficient, y=y_coefficient),
        -constant,
    )


def _candidate_projection(
    case: StressCaseDraft,
    point: Vec2,
) -> tuple[float, float, float]:
    subject = case.scene.object_by_id(case.intervention.subject_id)
    camera = case.scene.camera_by_id(case.intervention.camera_id)
    coefficients = tuple(
        _camera_affine_coefficients(camera, row, subject.position.z) for row in range(3)
    )
    camera_x, camera_y, depth = (
        x_coefficient * point.x + y_coefficient * point.y + constant
        for x_coefficient, y_coefficient, constant in coefficients
    )
    if depth <= 0.0:
        return math.nan, math.nan, depth
    fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
    return fx * camera_x / depth + cx, cy - fy * camera_y / depth, depth


def _candidate_projection_is_valid(case: StressCaseDraft, point: Vec2) -> bool:
    center_x, center_y, depth = _candidate_projection(case, point)
    if depth <= 0.0 or not all(
        math.isfinite(value) for value in (center_x, center_y, depth)
    ):
        return False
    scene = case.scene
    spec = case.intervention
    camera = scene.camera_by_id(spec.camera_id)
    subject = scene.object_by_id(spec.subject_id)
    subject_view = subject.views.get(spec.camera_id)
    residuals = _subject_projection_residuals(case)
    if subject_view is None or residuals is None:
        return False
    bbox = subject_view.bbox
    horizontal_residual, vertical_residual, depth_residual = residuals
    center_x += horizontal_residual
    center_y += vertical_residual
    calibrated_depth = depth + depth_residual
    if not math.isfinite(calibrated_depth) or calibrated_depth <= 0.0:
        return False

    half_width = (bbox.xmax - bbox.xmin) / 2.0
    half_height = (bbox.ymax - bbox.ymin) / 2.0
    xmin, xmax = center_x - half_width, center_x + half_width
    ymin, ymax = center_y - half_height, center_y + half_height
    full_area = (xmax - xmin) * (ymax - ymin)
    if full_area <= 0.0:
        return False
    clipped_width = max(
        0.0,
        min(xmax, camera.width) - max(xmin, 0.0),
    )
    clipped_height = max(
        0.0,
        min(ymax, camera.height) - max(ymin, 0.0),
    )
    truncated_fraction = 1.0 - clipped_width * clipped_height / full_area
    return (
        subject_view.visible_fraction >= RelationEngine.MIN_VISIBLE_FRACTION
        and subject_view.image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION
        and truncated_fraction <= RelationEngine.MAX_TRUNCATED_FRACTION
    )


def _candidate_axis_labels(
    case: StressCaseDraft,
    stationary: SceneObject,
    axis: RelationAxis,
    point: Vec2,
) -> frozenset[Relation]:
    scene = case.scene
    camera_id = case.intervention.camera_id
    if axis is RelationAxis.DISTANCE:
        subject = scene.object_by_id(case.intervention.subject_id)
        delta_x = point.x - subject.position.x
        delta_y = point.y - subject.position.y
        moved_obb = subject.obb.model_copy(
            update={
                "center": subject.obb.center.model_copy(
                    update={
                        "x": subject.obb.center.x + delta_x,
                        "y": subject.obb.center.y + delta_y,
                    }
                )
            }
        )
        gap = ground_gap(moved_obb, stationary.obb)
        if gap <= RelationEngine.NEAR_METERS:
            return frozenset({Relation.NEAR})
        if gap >= RelationEngine.FAR_METERS:
            return frozenset({Relation.FAR})
        return frozenset()

    stationary_view = stationary.views[camera_id]
    center_x, _, depth = _candidate_projection(case, point)
    residuals = _subject_projection_residuals(case)
    if residuals is None:
        return frozenset()
    horizontal_residual, _, depth_residual = residuals
    center_x += horizontal_residual
    depth += depth_residual
    if axis is RelationAxis.HORIZONTAL:
        delta = stationary_view.bbox.center_x - center_x
        threshold = (
            scene.camera_by_id(camera_id).width * RelationEngine.LEFT_RIGHT_FRACTION
        )
        positive_relation, negative_relation = Relation.LEFT, Relation.RIGHT
    else:
        delta = stationary_view.camera_depth - depth
        threshold = RelationEngine.FRONT_BEHIND_METERS
        positive_relation, negative_relation = Relation.FRONT, Relation.BEHIND
    distance = abs(delta)
    at_threshold = math.isclose(
        distance,
        threshold,
        rel_tol=0.0,
        abs_tol=_OPERATIONAL_COMPARISON_TOLERANCE,
    )
    if distance < threshold and not at_threshold:
        return frozenset()
    return frozenset({positive_relation if delta > 0.0 else negative_relation})


def _candidate_preserves_relations(case: StressCaseDraft, point: Vec2) -> bool:
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    engine = RelationEngine()
    for stationary in sorted(scene.objects, key=lambda item: item.object_id):
        if stationary.object_id == subject.object_id:
            continue
        subject_view = subject.views.get(spec.camera_id)
        stationary_view = stationary.views.get(spec.camera_id)
        permanently_invisible = (
            subject_view is None
            or stationary_view is None
            or subject_view.visible_fraction < engine.MIN_VISIBLE_FRACTION
            or subject_view.image_area_fraction < engine.MIN_IMAGE_AREA_FRACTION
            or subject_view.truncated_fraction > engine.MAX_TRUNCATED_FRACTION
            or stationary_view.visible_fraction < engine.MIN_VISIBLE_FRACTION
            or stationary_view.image_area_fraction < engine.MIN_IMAGE_AREA_FRACTION
            or stationary_view.truncated_fraction > engine.MAX_TRUNCATED_FRACTION
        )
        if permanently_invisible:
            continue
        before = engine.pair_labels(
            scene,
            subject.object_id,
            stationary.object_id,
            spec.camera_id,
        )
        for axis in RelationAxis:
            if (
                stationary.object_id == spec.reference_id
                and axis is spec.relation_after.axis
            ):
                continue
            before_axis = frozenset(
                relation for relation in before if relation.axis is axis
            )
            if _candidate_axis_labels(case, stationary, axis, point) != before_axis:
                return False
    return True


def _candidate_satisfies_target(case: StressCaseDraft, point: Vec2) -> bool:
    reference = case.scene.object_by_id(case.intervention.reference_id)
    relation = case.intervention.relation_after
    return relation in _candidate_axis_labels(case, reference, relation.axis, point)


def _actual_locus_is_empty(
    geometry: BaseGeometry,
    strict_boundaries: tuple[BaseGeometry, ...],
) -> bool:
    if geometry.is_empty:
        return True
    boundaries = tuple(
        boundary for boundary in strict_boundaries if not boundary.is_empty
    )
    if not boundaries:
        return False
    return geometry.difference(unary_union(boundaries)).is_empty


def recompute_stress_oracle(case: StressCaseDraft | StressCase) -> StressOracleResult:
    """Recompute outcome and optimum/bound fields from scene geometry alone."""
    draft = case.as_draft() if isinstance(case, StressCase) else case
    physical = _physical_center_locus(draft)
    physical = _visibility_center_locus(draft, physical)
    visibility_boundary = _positive_depth_boundary(draft, physical)
    physical, preservation_boundaries = _preserved_relations_center_locus(
        draft,
        physical,
    )
    strict_boundaries = (visibility_boundary, *preservation_boundaries)
    target, normal, required = _target_locus(draft, physical)
    subject = draft.scene.object_by_id(draft.intervention.subject_id)
    reference = draft.scene.object_by_id(draft.intervention.reference_id)
    origin = Vec2(x=subject.position.x, y=subject.position.y)
    if draft.intervention.relation_after is Relation.FAR and not _actual_locus_is_empty(
        target,
        strict_boundaries,
    ):
        configuration = _configuration_polygon(subject, reference)
        exact_candidates = _exact_far_candidates(
            origin,
            physical,
            configuration,
            RelationEngine.FAR_METERS,
        )
        valid_candidates = [
            point
            for point in exact_candidates
            if physical.covers(Point(point.x, point.y))
            and _candidate_projection_is_valid(draft, point)
            and _candidate_preserves_relations(draft, point)
            and _candidate_satisfies_target(draft, point)
        ]
        for point in exact_candidates:
            if point in valid_candidates:
                continue
            repaired = _attained_candidate_neighbor(
                draft,
                point,
                physical,
                strict_boundaries,
            )
            if repaired is not None:
                valid_candidates.append(repaired)
        if not valid_candidates:
            raise UnattainedOracleInfimumError("unattained_oracle_infimum")
        minimum = min(
            math.hypot(point.x - origin.x, point.y - origin.y)
            for point in valid_candidates
        )
        witnesses = tuple(
            point
            for point in valid_candidates
            if math.isclose(
                math.hypot(point.x - origin.x, point.y - origin.y),
                minimum,
                rel_tol=0.0,
                abs_tol=_EPS,
            )
        )
        return StressOracleResult(
            expected_outcome="SAT",
            exact_infimum_m=round(minimum, 12),
            exact_infimum_points=witnesses,
            maximum_possible_value_m=None,
            required_value_m=None,
        )
    if not _actual_locus_is_empty(target, strict_boundaries):
        witnesses = _nearest_geometry_points(
            origin,
            target,
            is_valid=lambda point: (
                _candidate_projection_is_valid(draft, point)
                and _candidate_preserves_relations(draft, point)
                and _candidate_satisfies_target(draft, point)
            ),
            strict_boundaries=strict_boundaries,
        )
        valid_witnesses = tuple(
            point
            for point in witnesses
            if _candidate_projection_is_valid(draft, point)
            and _candidate_preserves_relations(draft, point)
            and _candidate_satisfies_target(draft, point)
        )
        if not valid_witnesses:
            raise UnattainedOracleInfimumError("unattained_oracle_infimum")
        exact = min(
            math.hypot(point.x - origin.x, point.y - origin.y)
            for point in valid_witnesses
        )
        return StressOracleResult(
            expected_outcome="SAT",
            exact_infimum_m=round(exact, 12),
            exact_infimum_points=valid_witnesses,
            maximum_possible_value_m=None,
            required_value_m=None,
        )
    if normal is not None:
        maximum, required_value = _maximum_half_space_value(physical, normal, required)
    else:
        maximum = maximum_ground_gap(subject, reference, physical)
        required_value = required
    return StressOracleResult(
        expected_outcome="UNSAT",
        exact_infimum_m=None,
        exact_infimum_points=(),
        maximum_possible_value_m=round(maximum, 12),
        required_value_m=round(required_value, 12),
    )
