from __future__ import annotations

import math
from typing import Any

import numpy as np

from spatialcf.adapters.ai2thor.models import (
    AI2ThorAgentPose,
    AI2ThorNativePosition,
    _strict_finite_float,
)
from spatialcf.domain.scene import Quaternion, Vec3
from spatialcf.geometry.transforms import (
    ai2thor_position_to_world,
    ai2thor_rotation_to_world,
    matrix4,
    transform_point,
)

_ANGLE_TOLERANCE_DEGREES = 1e-4
_EPSILON = 1e-6

__all__ = (
    "ai2thor_position_to_world",
    "ai2thor_rotation_to_world",
    "matrix4",
    "transform_point",
)


def ai2thor_camera_world_to_camera(
    position: Vec3,
    *,
    yaw_degrees: float,
    horizon_degrees: float,
) -> tuple[float, ...]:
    """Build AI2-THOR's canonical world-to-camera extrinsic matrix."""

    yaw = math.radians(yaw_degrees)
    pitch = math.radians(horizon_degrees)
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0])
    forward = np.asarray(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            -math.sin(pitch),
        ]
    )
    up = np.cross(right, forward)
    rotation = np.stack([right, up, forward])
    translation = -rotation @ np.asarray([position.x, position.y, position.z])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return tuple(float(value) for value in world_to_camera.reshape(-1))


def _quaternion_yaw(rotation: Quaternion) -> float:
    norm = math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2)
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise ValueError("object rotation must be a finite non-zero quaternion")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotation_matrix(rotation: Quaternion) -> np.ndarray:
    norm = math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2)
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise ValueError("rotation must be a finite non-zero quaternion")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def _quaternions_close(left: Quaternion, right: Quaternion) -> bool:
    a = np.asarray([left.x, left.y, left.z, left.w], dtype=float)
    b = np.asarray([right.x, right.y, right.z, right.w], dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return bool(
        np.allclose(a, b, atol=1e-5, rtol=0.0)
        or np.allclose(a, -b, atol=1e-5, rtol=0.0)
    )


class AI2ThorConversionMixin:
    @staticmethod
    def _native_position(
        value: Any,
        label: str,
        *,
        error_type: type[Exception] = ValueError,
    ) -> AI2ThorNativePosition:
        if type(value) is not dict or set(value) != {"x", "y", "z"}:
            raise error_type(f"{label} must have exactly x/y/z keys")
        try:
            return AI2ThorNativePosition(
                x=_strict_finite_float(value["x"], f"{label} x"),
                y=_strict_finite_float(value["y"], f"{label} y"),
                z=_strict_finite_float(value["z"], f"{label} z"),
            )
        except ValueError as error:
            if error_type is ValueError:
                raise
            raise error_type(str(error)) from error

    @classmethod
    def _native_agent_pose(cls, event: Any) -> AI2ThorAgentPose:
        metadata = event.metadata
        agent = metadata.get("agent")
        if type(agent) is not dict:
            raise ValueError("AI2-THOR event has no valid agent pose")
        position = cls._native_position(agent.get("position"), "agent position")
        rotation = agent.get("rotation")
        if type(rotation) is not dict or set(rotation) != {"x", "y", "z"}:
            raise ValueError("agent rotation must have exactly x/y/z keys")
        rotation_values = {
            axis: _strict_finite_float(rotation[axis], f"agent rotation {axis}")
            for axis in ("x", "y", "z")
        }
        if not cls._angles_close(rotation_values["x"], 0.0) or not cls._angles_close(
            rotation_values["z"], 0.0
        ):
            raise RuntimeError("camera pose returned non-zero pitch or roll")

        horizons: list[float] = []
        if "cameraHorizon" in metadata:
            horizons.append(
                _strict_finite_float(
                    metadata["cameraHorizon"],
                    "top-level camera horizon",
                )
            )
        if "cameraHorizon" in agent:
            horizons.append(
                _strict_finite_float(
                    agent["cameraHorizon"],
                    "agent camera horizon",
                )
            )
        if not horizons:
            raise ValueError("AI2-THOR event has no camera horizon")
        if any(not cls._angles_close(horizons[0], horizon) for horizon in horizons[1:]):
            raise RuntimeError("AI2-THOR camera horizon metadata is inconsistent")
        standing = agent.get("isStanding")
        if type(standing) is not bool:
            raise ValueError("AI2-THOR agent standing must be an exact boolean")
        return AI2ThorAgentPose(
            position=position,
            yaw_degrees=rotation_values["y"],
            horizon_degrees=horizons[0],
            standing=standing,
        )

    @staticmethod
    def _angles_close(
        left: float,
        right: float,
        *,
        tolerance_degrees: float = _ANGLE_TOLERANCE_DEGREES,
    ) -> bool:
        difference = (left - right + 180.0) % 360.0 - 180.0
        return abs(difference) <= tolerance_degrees
