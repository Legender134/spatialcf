import math

from shapely.affinity import rotate
from shapely.geometry import Polygon

from spatialcf.domain.models import OBB

OBB_INTERSECTION_XY_AREA_TOLERANCE = 1e-9
OBB_INTERSECTION_Z_OVERLAP_TOLERANCE = 1e-9


def _yaw_degrees(obb: OBB) -> float:
    q = obb.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return math.degrees(yaw)


def obb_footprint(obb: OBB) -> Polygon:
    hx = obb.extent.x / 2.0
    hy = obb.extent.y / 2.0
    polygon = Polygon([
        (obb.center.x - hx, obb.center.y - hy),
        (obb.center.x + hx, obb.center.y - hy),
        (obb.center.x + hx, obb.center.y + hy),
        (obb.center.x - hx, obb.center.y + hy),
    ])
    return rotate(polygon, _yaw_degrees(obb), origin=(obb.center.x, obb.center.y))


def ground_gap(first: OBB, second: OBB) -> float:
    return float(obb_footprint(first).distance(obb_footprint(second)))


def footprints_overlap(first: OBB, second: OBB, tolerance: float = 1e-6) -> bool:
    return obb_footprint(first).intersection(obb_footprint(second)).area > tolerance


def obb_z_overlap_depth(first: OBB, second: OBB) -> float:
    """Return signed Z overlap: positive volume, zero contact, negative gap."""
    first_bottom = first.center.z - first.extent.z / 2.0
    first_top = first.center.z + first.extent.z / 2.0
    second_bottom = second.center.z - second.extent.z / 2.0
    second_top = second.center.z + second.extent.z / 2.0
    return min(first_top, second_top) - max(first_bottom, second_bottom)


def obbs_intersect_3d(
    first: OBB,
    second: OBB,
    *,
    xy_area_tolerance: float = OBB_INTERSECTION_XY_AREA_TOLERANCE,
    z_overlap_tolerance: float = OBB_INTERSECTION_Z_OVERLAP_TOLERANCE,
) -> bool:
    """Return whether two Z-up OBBs have positive 3D volume intersection.

    Contact is allowed.  Intersection requires both XY footprint area greater
    than ``xy_area_tolerance`` square metres and Z overlap greater than
    ``z_overlap_tolerance`` metres.
    """
    return (
        footprints_overlap(first, second, tolerance=xy_area_tolerance)
        and obb_z_overlap_depth(first, second) > z_overlap_tolerance
    )


def inside_room(obb: OBB, room: Polygon, tolerance: float = 1e-6) -> bool:
    return room.buffer(tolerance).covers(obb_footprint(obb))
