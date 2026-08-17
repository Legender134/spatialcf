import math

import numpy as np

from spatialcf.domain.models import Quaternion, Vec3


def ai2thor_position_to_world(value: Vec3) -> Vec3:
    return Vec3(x=value.x, y=value.z, z=value.y)


def ai2thor_rotation_to_world(euler_degrees: Vec3) -> Quaternion:
    # Unity Quaternion.Euler applies Z, then X, then Y. In Hamilton product
    # notation that is q_y * q_x * q_z. Conjugating the Unity orientation by
    # the Y-up -> canonical Z-up basis maps quaternion (x, y, z, w) to
    # (-x, -z, -y, w).
    half_x = math.radians(euler_degrees.x) / 2.0
    half_y = math.radians(euler_degrees.y) / 2.0
    half_z = math.radians(euler_degrees.z) / 2.0
    sx, cx = math.sin(half_x), math.cos(half_x)
    sy, cy = math.sin(half_y), math.cos(half_y)
    sz, cz = math.sin(half_z), math.cos(half_z)
    unity_x = cy * sx * cz + sy * cx * sz
    unity_y = -cy * sx * sz + sy * cx * cz
    unity_z = cy * cx * sz - sy * sx * cz
    unity_w = cy * cx * cz + sy * sx * sz
    return Quaternion(
        x=-unity_x,
        y=-unity_z,
        z=-unity_y,
        w=unity_w,
    )


def matrix4(values: tuple[float, ...]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("matrix must have 16 values")
    return np.asarray(values, dtype=float).reshape(4, 4)


def transform_point(matrix: np.ndarray, point: Vec3) -> Vec3:
    transformed = matrix @ np.asarray([point.x, point.y, point.z, 1.0])
    if transformed[3] == 0:
        raise ValueError("point transformed to infinity")
    transformed = transformed / transformed[3]
    return Vec3(x=float(transformed[0]), y=float(transformed[1]), z=float(transformed[2]))
