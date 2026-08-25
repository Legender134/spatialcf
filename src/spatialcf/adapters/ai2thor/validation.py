"""Native-only validation for canonical AI2-THOR observations."""

from __future__ import annotations

import math
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from spatialcf.adapters.ai2thor.models import AI2ThorObservation
from spatialcf.verification.integrity import (
    competition_native_observation_payload_sha256,
)


def competition_native_observation_sha256(observation: AI2ThorObservation) -> str:
    """Hash one exact native observation through the neutral payload owner."""

    if type(observation) is not AI2ThorObservation:
        raise TypeError("native observation digest requires an exact observation")
    assets = {
        "depth_npy_sha256": sha256(observation.depth_npy).hexdigest(),
        "instance_png_sha256": sha256(observation.instance_png).hexdigest(),
        "pointcloud_ply_sha256": sha256(observation.pointcloud_ply).hexdigest(),
        "rgb_png_sha256": sha256(observation.rgb_png).hexdigest(),
    }
    stored = {
        "depth_npy_sha256": observation.depth_npy_sha256,
        "instance_png_sha256": observation.instance_png_sha256,
        "pointcloud_ply_sha256": observation.pointcloud_ply_sha256,
        "rgb_png_sha256": observation.rgb_png_sha256,
    }
    if assets != stored:
        raise ValueError("native observation stored asset digests do not match bytes")
    return competition_native_observation_payload_sha256(
        scene=observation.scene,
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=observation.instance_pixel_counts,
        is_scene_at_rest=observation.is_scene_at_rest,
    )


def observation_contract_errors(
    observation: AI2ThorObservation,
    camera_id: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        camera = observation.scene.camera_by_id(camera_id)
    except KeyError:
        return ("observation_camera_missing",)

    assets = (
        ("rgb", observation.rgb_png, observation.rgb_png_sha256),
        ("depth", observation.depth_npy, observation.depth_npy_sha256),
        (
            "instance",
            observation.instance_png,
            observation.instance_png_sha256,
        ),
        (
            "pointcloud",
            observation.pointcloud_ply,
            observation.pointcloud_ply_sha256,
        ),
    )
    for name, payload, expected_digest in assets:
        if sha256(payload).hexdigest() != expected_digest:
            errors.append(f"observation_{name}_digest_mismatch")

    for name, payload in (
        ("rgb", observation.rgb_png),
        ("instance", observation.instance_png),
    ):
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.size != (camera.width, camera.height):
                    errors.append(f"observation_{name}_dimensions_mismatch")
                if image.mode != "RGB":
                    errors.append(f"observation_{name}_dtype_mismatch")
        except (OSError, UnidentifiedImageError, ValueError):
            errors.append(f"observation_{name}_invalid")

    try:
        depth = np.load(BytesIO(observation.depth_npy), allow_pickle=False)
        if depth.shape != (camera.height, camera.width):
            errors.append("observation_depth_dimensions_mismatch")
        if depth.dtype != np.float32:
            errors.append("observation_depth_dtype_mismatch")
        if not np.any(np.isfinite(depth) & (depth > 0.0)):
            errors.append("observation_depth_has_no_positive_sample")
    except (OSError, ValueError, TypeError):
        errors.append("observation_depth_invalid")

    try:
        pointcloud = observation.pointcloud_ply.decode("ascii")
        lines = pointcloud.splitlines()
        if (
            len(lines) < 4
            or lines[0] != "ply"
            or lines[1] != "format ascii 1.0"
            or "end_header" not in lines
        ):
            raise ValueError
        vertex_lines = [line for line in lines if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError
        vertex_count = int(vertex_lines[0].removeprefix("element vertex "))
        header_end = lines.index("end_header")
        vertices = lines[header_end + 1 :]
        if vertex_count <= 0 or len(vertices) != vertex_count:
            raise ValueError
        for vertex in vertices:
            values = vertex.split()
            if len(values) != 6:
                raise ValueError
            coordinates = tuple(float(value) for value in values[:3])
            colors = tuple(int(value) for value in values[3:])
            if not all(math.isfinite(value) for value in coordinates) or not all(
                0 <= value <= 255 for value in colors
            ):
                raise ValueError
    except (UnicodeDecodeError, ValueError):
        errors.append("observation_pointcloud_invalid")

    expected_ids = {obj.object_id for obj in observation.scene.objects}
    if set(observation.instance_pixel_counts) != expected_ids:
        errors.append("observation_instance_count_ids_mismatch")
    max_pixels = camera.width * camera.height
    if any(
        type(count) is not int or count < 0 or count > max_pixels
        for count in observation.instance_pixel_counts.values()
    ):
        errors.append("observation_instance_count_invalid")
    return tuple(sorted(set(errors)))
