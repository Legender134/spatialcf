"""Exact unchanged-reference tile persistence in the declared pinhole model.

Only source-observed pixels are credited. A whole source 2x2 tile must be
strictly separated from the destination subject's projected outer hull.
This depth-free rule deliberately discards even obstructions behind the
reference. It does not claim native mesh or renderer equivalence.
"""

from fractions import Fraction as F
from itertools import product
import math

from spatialcf.generation.planning.models import SourceViewObjectProxy


def _directed(value: F, *, upward: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("unrepresentable reference metric")
    if (upward and F(result) < value) or (not upward and F(result) > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _hull(points):
    points = sorted(set(points))
    lower, upper = [], []
    for source, target in ((points, lower), (reversed(points), upper)):
        for point in source:
            while len(target) >= 2 and _cross(target[-2], target[-1], point) <= 0:
                target.pop()
            target.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("degenerate reference projection")
    return hull


def _separated(hull, rectangle):
    axes = [(F(1), F(0)), (F(0), F(1))]
    axes.extend((q[1] - p[1], p[0] - q[0])
                for p, q in zip(hull, (*hull[1:], hull[0])))
    for x, y in axes:
        left = tuple(x * p[0] + y * p[1] for p in hull)
        right = tuple(x * p[0] + y * p[1] for p in rectangle)
        if max(left) < min(right) or max(right) < min(left):
            return True
    return False


def reference_proxy(scene, intervention, fact, edit):
    camera = scene.camera_by_id("main")
    k = tuple(F(value) for value in camera.intrinsics)
    m = tuple(tuple(F(value) for value in camera.world_to_camera[i:i + 4])
              for i in range(0, 16, 4))
    if (k[1], k[3], k[6], k[7], k[8]) != (0, 0, 0, 0, 1) or min(k[0], k[4]) <= 0:
        raise ValueError("unsupported persistence intrinsics")
    determinant = sum(m[0][i] * (m[1][(i + 1) % 3] * m[2][(i + 2) % 3]
                               - m[1][(i + 2) % 3] * m[2][(i + 1) % 3])
                      for i in range(3))
    if m[3] != (0, 0, 0, 1) or determinant == 0:
        raise ValueError("unsupported persistence camera")

    def transform(point):
        return tuple(sum(m[i][j] * point[j] for j in range(3)) + m[i][3]
                     for i in range(3))

    def projection(obj, *, moved=False):
        q = obj.obb.rotation
        if (q.x, q.y, q.z, q.w) != (0., 0., 0., 1.):
            raise ValueError("persistence requires world-axis outer boxes")
        center = [F(getattr(obj.obb.center, axis)) for axis in ("x", "y", "z")]
        if moved:
            # Match the persisted endpoint geometry's rounded center exactly.
            center[0] = F(obj.obb.center.x + edit.translation_xy_m.x)
            center[1] = F(obj.obb.center.y + edit.translation_xy_m.y)
        half = tuple(F(getattr(obj.obb.extent, axis)) / 2 for axis in ("x", "y", "z"))
        if min(half) <= 0:
            raise ValueError("degenerate persistence outer box")
        vertices = tuple(transform(tuple(c + h * s for c, h, s in zip(center, half, signs)))
                         for signs in product((-1, 1), repeat=3))
        if min(p[2] for p in vertices) <= 0:
            raise ValueError("persistence outer box crosses camera plane")
        return _hull((k[0] * x / z + k[2], k[5] - k[4] * y / z)
                     for x, y, z in vertices)

    subject = scene.object_by_id(intervention.subject_id)
    reference = scene.object_by_id(intervention.reference_id)
    subject_hull = projection(subject, moved=True)
    reference_hull = projection(reference)
    row = next(item for item in fact.objects if item.object_id == reference.object_id)
    retained = []
    for y, x, weight in zip(row.sample_rows, row.sample_columns, row.sample_weights, strict=True):
        left, top = x // 2 * 2, y // 2 * 2
        right, bottom = min(left + 1, camera.width - 1), min(top + 1, camera.height - 1)
        tile = ((left, top), (right, top), (right, bottom), (left, bottom))
        if _separated(subject_hull, tile):
            retained.append((x, y, weight))
    if not retained:
        raise ValueError("no persistent reference tiles")
    xs, ys = tuple(p[0] for p in reference_hull), tuple(p[1] for p in reference_hull)
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    clipped = (max(F(0), min(max(xs), camera.width) - max(min(xs), 0))
               * max(F(0), min(max(ys), camera.height) - max(min(ys), 0)))
    columns, rows = tuple(p[0] for p in retained), tuple(p[1] for p in retained)
    bbox_area = (max(columns) - min(columns) + 1) * (max(rows) - min(rows) + 1)
    depth = transform(tuple(F(getattr(reference.position, axis)) for axis in ("x", "y", "z")))[2]
    if depth <= 0:
        raise ValueError("persistence reference anchor behind camera")
    return SourceViewObjectProxy(
        object_id=reference.object_id,
        # Unseen newly revealed pixels may extend the bbox anywhere in frame.
        bbox_center_x_lower=_directed(F(max(columns) + 1, 2)),
        bbox_center_x_upper=_directed(F(min(columns) + camera.width, 2), upward=True),
        camera_depth_lower_m=_directed(depth),
        camera_depth_upper_m=_directed(depth, upward=True),
        image_area_fraction_lower=_directed(F(bbox_area, camera.width * camera.height)),
        visible_fraction_lower=_directed(min(F(1), sum(p[2] for p in retained) / area)),
        truncated_fraction_upper=_directed(1 - clipped / area, upward=True),
    )
