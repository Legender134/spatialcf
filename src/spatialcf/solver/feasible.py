import math
from collections.abc import Iterable

from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import Camera, InterventionSpec, Scene, SceneObject
from spatialcf.geometry.obb import (
    OBB_INTERSECTION_Z_OVERLAP_TOLERANCE,
    obb_footprint,
    obb_z_overlap_depth,
)
from spatialcf.geometry.regions import subject_position_region_geometry

PolygonalGeometry = Polygon | MultiPolygon
DISK_POLYGON_SIDES = 128
GEOMETRY_EPS = 1e-9
_MAX_SCALE_CORRECTIONS = 256


def _polygonal(geometry: BaseGeometry) -> PolygonalGeometry:
    """Keep only area-bearing pieces, including holes and split components."""
    if geometry.is_empty:
        return Polygon()
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        polygons = [polygon for polygon in geometry.geoms if not polygon.is_empty]
    elif isinstance(geometry, GeometryCollection):
        polygons = list(_polygons(geometry.geoms))
    else:
        return Polygon()
    if not polygons:
        return Polygon()
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        return merged
    if isinstance(merged, MultiPolygon):
        return merged
    return _polygonal(merged)


def _polygons(geometries: Iterable[BaseGeometry]) -> Iterable[Polygon]:
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            if not geometry.is_empty:
                yield geometry
        elif isinstance(geometry, MultiPolygon):
            yield from (polygon for polygon in geometry.geoms if not polygon.is_empty)
        elif isinstance(geometry, GeometryCollection):
            yield from _polygons(geometry.geoms)


def clip_half_plane(
    polygon: Polygon,
    nx: float,
    ny: float,
    offset: float,
    keep_greater: bool,
) -> Polygon:
    """Clip a convex mask polygon to ``nx*x + ny*y >= offset`` or its inverse."""
    coords = list(polygon.exterior.coords)[:-1]

    def value(point: tuple[float, float]) -> float:
        raw = nx * point[0] + ny * point[1] - offset
        return raw if keep_greater else -raw

    output: list[tuple[float, float]] = []
    for start, end in zip(coords, coords[1:] + coords[:1]):
        start_value, end_value = value(start), value(end)
        if start_value >= 0:
            output.append(start)
        if (start_value >= 0) != (end_value >= 0):
            ratio = start_value / (start_value - end_value)
            output.append((
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            ))
    return Polygon(output) if len(output) >= 3 else Polygon()


def _relative_vertices(subject: SceneObject) -> list[tuple[float, float]]:
    return [
        (x - subject.position.x, y - subject.position.y)
        for x, y in list(obb_footprint(subject.obb).exterior.coords)[:-1]
    ]


def _center_locus(
    container: BaseGeometry,
    relative_vertices: list[tuple[float, float]],
) -> PolygonalGeometry:
    """Centers whose translated fixed-orientation footprint lies in a container.

    This is exact for convex containers. For concave rooms, checking footprint
    vertices is deliberately a safe superset rather than a false-negative filter.
    """
    if not relative_vertices:
        return _polygonal(container)
    first_x, first_y = relative_vertices[0]
    locus = _polygonal(translate(container, xoff=-first_x, yoff=-first_y))
    for x, y in relative_vertices[1:]:
        locus = _polygonal(
            locus.intersection(translate(container, xoff=-x, yoff=-y))
        )
        if locus.is_empty:
            break
    return locus


def _convex_minkowski_sum(
    first: Polygon,
    second_vertices: list[tuple[float, float]],
) -> Polygon:
    """The exact Minkowski sum of convex polygons via vertex sums and a hull."""
    first_vertices = list(first.exterior.coords)[:-1]
    return MultiPoint([
        (first_x + second_x, first_y + second_y)
        for first_x, first_y in first_vertices
        for second_x, second_y in second_vertices
    ]).convex_hull


def _configuration_obstacle(
    fixed: Polygon,
    relative_vertices: list[tuple[float, float]],
) -> Polygon:
    """Fixed footprint Minkowski-summed with the reflected subject footprint."""
    return _convex_minkowski_sum(
        fixed,
        [(-subject_x, -subject_y) for subject_x, subject_y in relative_vertices],
    )


def _regular_disk_polygon(
    radius: float,
    *,
    circumscribed: bool,
    sides: int = DISK_POLYGON_SIDES,
) -> Polygon:
    """A deterministic regular polygon around an origin-centered disk.

    Scaling uses invariants measured from the realized floating-point unit
    vertices. The circumscribed polygon has every side apothem at least
    ``radius + GEOMETRY_EPS``; the inscribed polygon has every vertex norm at
    most ``radius - GEOMETRY_EPS``. The epsilon dominates floating-point and
    GEOS overlay noise while remaining negligible at meter-scale thresholds.
    """
    if type(sides) is not int or sides < 4 or sides & (sides - 1) != 0:
        raise ValueError("sides must be a power of two at least four")

    unit_vertices = [
        (
            math.cos(2.0 * math.pi * index / sides),
            math.sin(2.0 * math.pi * index / sides),
        )
        for index in range(sides)
    ]

    if circumscribed:
        target = radius + GEOMETRY_EPS
        unit_invariant = _minimum_side_apothem(unit_vertices)
        scale = target / unit_invariant
    else:
        target = radius - GEOMETRY_EPS
        unit_invariant = max(math.hypot(x, y) for x, y in unit_vertices)
        scale = target / unit_invariant

    for _ in range(_MAX_SCALE_CORRECTIONS):
        polygon = Polygon([(scale * x, scale * y) for x, y in unit_vertices])
        realized_vertices = list(polygon.exterior.coords)[:-1]
        if circumscribed:
            realized_invariant = _minimum_side_apothem(realized_vertices)
            if realized_invariant >= target:
                return polygon
            scale = math.nextafter(
                scale * (target / realized_invariant),
                math.inf,
            )
        else:
            if max(math.hypot(x, y) for x, y in realized_vertices) <= target:
                return polygon
            scale = math.nextafter(scale, 0.0)
    raise ArithmeticError("could not realize a numerically safe regular disk polygon")


def _minimum_side_apothem(vertices: list[tuple[float, float]]) -> float:
    """Minimum origin-to-side distance for an ordered convex polygon."""
    return min(
        abs(start_x * end_y - start_y * end_x)
        / math.hypot(end_x - start_x, end_y - start_y)
        for (start_x, start_y), (end_x, end_y)
        in zip(vertices, vertices[1:] + vertices[:1])
    )


def _half_plane_mask(
    region: PolygonalGeometry,
    nx: float,
    ny: float,
    constant: float,
    keep_greater: bool,
) -> Polygon:
    if math.isclose(nx, 0.0, abs_tol=1e-12) and math.isclose(ny, 0.0, abs_tol=1e-12):
        if (constant >= 0.0) == keep_greater:
            min_x, min_y, max_x, max_y = region.bounds
            return box(min_x - 1.0, min_y - 1.0, max_x + 1.0, max_y + 1.0)
        return Polygon()
    min_x, min_y, max_x, max_y = region.bounds
    span = max(max_x - min_x, max_y - min_y, 1.0)
    universe = box(min_x - span, min_y - span, max_x + span, max_y + span)
    return clip_half_plane(universe, nx, ny, -constant, keep_greater)


def _intersect_half_plane(
    region: PolygonalGeometry,
    nx: float,
    ny: float,
    constant: float,
    keep_greater: bool,
) -> PolygonalGeometry:
    if region.is_empty:
        return Polygon()
    # The relation thresholds are inclusive. Expand by a tiny numerical amount
    # so clipping arithmetic cannot turn an exact threshold point into a hole.
    tolerance = 1e-9
    if keep_greater:
        constant += tolerance
    else:
        constant -= tolerance
    return _polygonal(region.intersection(
        _half_plane_mask(region, nx, ny, constant, keep_greater)
    ))


def _camera_row(camera: Camera, row: int, subject_z: float) -> tuple[float, float, float]:
    matrix = camera.world_to_camera
    start = row * 4
    return matrix[start], matrix[start + 1], matrix[start + 2] * subject_z + matrix[start + 3]


def _require_query_views(subject: SceneObject, reference: SceneObject, camera_id: str) -> None:
    if subject.views.get(camera_id) is None or reference.views.get(camera_id) is None:
        raise ValueError("camera-relative feasible regions require subject and reference query views")


class FeasibleRegionBuilder:
    def build(self, scene: Scene, spec: InterventionSpec) -> PolygonalGeometry:
        room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
        subject = scene.object_by_id(spec.subject_id)
        reference = scene.object_by_id(spec.reference_id)
        relative_vertices = _relative_vertices(subject)
        region = _center_locus(room, relative_vertices)

        for position_region in scene.subject_position_regions:
            if position_region.subject_object_id == subject.object_id:
                region = _polygonal(region.intersection(
                    subject_position_region_geometry(position_region)
                ))

        support_id = subject.support_object_id
        if support_id is not None:
            support = scene.object_by_id(support_id)
            region = _polygonal(region.intersection(
                _center_locus(obb_footprint(support.obb), relative_vertices)
            ))

        if spec.relation_after in {Relation.LEFT, Relation.RIGHT, Relation.FRONT, Relation.BEHIND}:
            _require_query_views(subject, reference, spec.camera_id)
            camera = scene.camera_by_id(spec.camera_id)
            if spec.relation_after in {Relation.LEFT, Relation.RIGHT}:
                region = self._left_right_region(region, subject, reference, camera, spec)
            else:
                region = self._front_behind_region(region, subject, reference, camera, spec)
        elif spec.relation_after is Relation.NEAR:
            configuration = _configuration_obstacle(obb_footprint(reference.obb), relative_vertices)
            near_locus = _convex_minkowski_sum(
                configuration,
                list(_regular_disk_polygon(0.50, circumscribed=True).exterior.coords)[:-1],
            )
            region = _polygonal(region.intersection(near_locus))
        else:
            configuration = _configuration_obstacle(obb_footprint(reference.obb), relative_vertices)
            far_exclusion = _convex_minkowski_sum(
                configuration,
                list(_regular_disk_polygon(1.50, circumscribed=False).exterior.coords)[:-1],
            )
            region = _polygonal(region.difference(far_exclusion))

        excluded_ids = {subject.object_id, support_id}
        for obstacle in scene.objects:
            if (
                obstacle.object_id in excluded_ids
                or obb_z_overlap_depth(subject.obb, obstacle.obb)
                <= OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
            ):
                continue
            configuration = _configuration_obstacle(obb_footprint(obstacle.obb), relative_vertices)
            region = _polygonal(region.difference(configuration))
        for obstacle in scene.collision_obstacles:
            conservative_obb = obstacle.conservative_obb()
            if (
                obb_z_overlap_depth(subject.obb, conservative_obb)
                <= OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
            ):
                continue
            configuration = _configuration_obstacle(
                obb_footprint(conservative_obb), relative_vertices
            )
            region = _polygonal(region.difference(configuration))
        return _polygonal(region)

    def _left_right_region(
        self,
        region: PolygonalGeometry,
        subject: SceneObject,
        reference: SceneObject,
        camera: Camera,
        spec: InterventionSpec,
    ) -> PolygonalGeometry:
        reference_view = reference.views[spec.camera_id]
        target = reference_view.bbox.center_x + (
            camera.width * 0.05 if spec.relation_after is Relation.RIGHT else -camera.width * 0.05
        )
        x_nx, x_ny, x_constant = _camera_row(camera, 0, subject.position.z)
        depth_nx, depth_ny, depth_constant = _camera_row(camera, 2, subject.position.z)
        fx, cx = camera.intrinsics[0], camera.intrinsics[2]
        numerator_nx = fx * x_nx + (cx - target) * depth_nx
        numerator_ny = fx * x_ny + (cx - target) * depth_ny
        numerator_constant = fx * x_constant + (cx - target) * depth_constant
        region = _intersect_half_plane(
            region,
            depth_nx,
            depth_ny,
            depth_constant,
            keep_greater=True,
        )
        return _intersect_half_plane(
            region,
            numerator_nx,
            numerator_ny,
            numerator_constant,
            keep_greater=spec.relation_after is Relation.RIGHT,
        )

    def _front_behind_region(
        self,
        region: PolygonalGeometry,
        subject: SceneObject,
        reference: SceneObject,
        camera: Camera,
        spec: InterventionSpec,
    ) -> PolygonalGeometry:
        subject_view = subject.views[spec.camera_id]
        reference_view = reference.views[spec.camera_id]
        nx, ny, constant = _camera_row(camera, 2, subject.position.z)
        current_depth = nx * subject.position.x + ny * subject.position.y + constant
        calibrated_constant = constant + subject_view.camera_depth - current_depth
        target = reference_view.camera_depth + (
            0.20 if spec.relation_after is Relation.BEHIND else -0.20
        )
        return _intersect_half_plane(
            region,
            nx,
            ny,
            calibrated_constant - target,
            keep_greater=spec.relation_after is Relation.BEHIND,
        )
