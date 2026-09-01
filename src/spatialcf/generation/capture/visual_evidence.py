"""Build compact, platform-neutral source-view facts from captured raster bytes."""

from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image

from spatialcf.adapters.base import AdapterObservation
from spatialcf.domain.scene import BBox2D, Scene
from spatialcf.domain.serialization import canonical_sha256
from spatialcf.generation.capture.models import (
    _SOURCE_VIEW_SAMPLING_POLICY_SHA256,
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceRefV2_9,
    SourceViewFact,
    SourceViewObjectSamples,
)

SOURCE_VIEW_FACT_VERSION = "competition-native-source-view-fact:2.9.5"
SOURCE_VIEW_SAMPLE_CAP = 76_800
SOURCE_VIEW_QUANTIZATION_M = 1e-5
_FACT_DOMAIN = "spatialcf.competition-native-source-view-fact.v2.9.5"
SOURCE_VIEW_SAMPLING_POLICY_SHA256 = _SOURCE_VIEW_SAMPLING_POLICY_SHA256


class SourceViewFactError(ValueError):
    """Typed fail-closed source-view derivation error."""


def _sha(value: object) -> str:
    return canonical_sha256(value, domain="spatialcf.source-view-binding.v1")


def _decode_instance(payload: bytes, *, width: int, height: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            if image.mode != "RGB" or image.size != (width, height):
                raise SourceViewFactError("source instance PNG dimensions or mode changed")
            return np.asarray(image, dtype=np.uint8)
    except SourceViewFactError:
        raise
    except (OSError, ValueError) as error:
        raise SourceViewFactError("source instance PNG is invalid") from error


def _decode_depth(payload: bytes, *, width: int, height: int) -> np.ndarray:
    try:
        depth = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SourceViewFactError("source depth NPY is invalid") from error
    if type(depth) is not np.ndarray:
        if isinstance(depth, np.lib.npyio.NpzFile):
            depth.close()
        raise SourceViewFactError(
            "source depth NPY must contain a single float32 ndarray"
        )
    if depth.shape != (height, width) or depth.dtype != np.float32:
        raise SourceViewFactError("source depth NPY dimensions or dtype changed")
    return depth


def _validated_instance_counts(value: object) -> dict[str, int]:
    if type(value) is not tuple:
        raise SourceViewFactError("source-view counts must be an exact tuple")
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or item[1] < 0
        ):
            raise SourceViewFactError("source-view count is not an exact pair")
    if value != tuple(sorted(value)):
        raise SourceViewFactError("source-view counts are not canonical")
    if len(value) != len({item[0] for item in value}):
        raise SourceViewFactError("source-view counts contain duplicate objects")
    return dict(value)


def _validated_instance_colors(value: object) -> dict[str, tuple[int, int, int]]:
    if type(value) is not tuple:
        raise SourceViewFactError("source-view colors must be an exact tuple")
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
        ):
            raise SourceViewFactError("source-view color is not an exact pair")
        color = item[1]
        if (
            type(color) is not tuple
            or len(color) != 3
            or any(
                type(channel) is not int or not 0 <= channel <= 255
                for channel in color
            )
        ):
            raise SourceViewFactError("source-view color is not exact RGB")
    if value != tuple(sorted(value)):
        raise SourceViewFactError("source-view colors are not canonical")
    if len(value) != len({item[0] for item in value}):
        raise SourceViewFactError("source-view colors contain duplicate objects")
    if len(value) != len({item[1] for item in value}):
        raise SourceViewFactError("source-view colors reuse an RGB value")
    return dict(value)


def _quantize_local_coordinate(value: float) -> int:
    if not math.isfinite(value):
        raise SourceViewFactError("source-view local coordinate is not finite")
    return round(value / SOURCE_VIEW_QUANTIZATION_M)


def build_source_view_fact(
    source: CompetitionNativeSourceRefV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    scene: Scene,
    observation: AdapterObservation,
) -> SourceViewFact:
    """Derive one deterministic weighted 2x2-tile source-view fact."""
    if type(source) is not CompetitionNativeSourceRefV2_9:
        raise SourceViewFactError("source-view source must be exact")
    if type(runtime_identity) is not CompetitionNativeRuntimeIdentityV2_9:
        raise SourceViewFactError("source-view runtime must be exact")
    if type(scene) is not Scene or type(observation) is not AdapterObservation:
        raise SourceViewFactError("source-view scene and observation must be exact")
    if observation.scene != scene or scene.scene_id != source.scene_id:
        raise SourceViewFactError("source-view observation does not bind the scene")
    camera = scene.camera_by_id("main")
    if (camera.width, camera.height) != (
        runtime_identity.width,
        runtime_identity.height,
    ):
        raise SourceViewFactError("source-view runtime dimensions changed")
    instance = _decode_instance(
        observation.instance_png, width=camera.width, height=camera.height
    )
    depth = _decode_depth(
        observation.depth_npy, width=camera.width, height=camera.height
    )
    counts = _validated_instance_counts(observation.instance_pixel_counts)
    colors = _validated_instance_colors(observation.instance_colors)
    required = {object_id for object_id, count in counts.items() if count > 0}
    if set(colors) != required:
        raise SourceViewFactError("source-view colors do not cover visible objects")
    try:
        camera_to_world = np.linalg.inv(
            np.asarray(camera.world_to_camera, dtype=np.float64).reshape(4, 4)
        )
    except np.linalg.LinAlgError as error:
        raise SourceViewFactError("source-view camera matrix is singular") from error
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise SourceViewFactError("source-view camera intrinsics are invalid")

    rows: list[SourceViewObjectSamples] = []
    total_samples = 0
    for object_id in sorted(required):
        try:
            obj = scene.object_by_id(object_id)
        except KeyError as error:
            raise SourceViewFactError(
                "source-view evidence names an unknown object"
            ) from error
        view = obj.views.get("main")
        if view is None:
            raise SourceViewFactError("visible source object has no main view")
        if view.camera_id != "main":
            raise SourceViewFactError(
                "source-view main view camera identity changed"
            )
        raw_color = colors[object_id]
        if (
            type(raw_color) is not tuple
            or len(raw_color) != 3
            or any(type(channel) is not int or not 0 <= channel <= 255 for channel in raw_color)
        ):
            raise SourceViewFactError("source-view color is not exact RGB")
        color = np.asarray(raw_color, dtype=np.uint8)
        mask = np.all(instance == color, axis=2)
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count != counts[object_id]:
            raise SourceViewFactError("source-view mask pixel count changed")
        occupied = np.argwhere(mask)
        if occupied.size == 0:
            raise SourceViewFactError("source-view object mask is empty")
        ymin, xmin = occupied.min(axis=0)
        ymax, xmax = occupied.max(axis=0)
        bbox = BBox2D(
            xmin=float(xmin),
            ymin=float(ymin),
            xmax=float(xmax + 1),
            ymax=float(ymax + 1),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in zip(
                (view.bbox.xmin, view.bbox.ymin, view.bbox.xmax, view.bbox.ymax),
                (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
                strict=True,
            )
        ):
            raise SourceViewFactError("source-view detection and mask bbox differ")
        tile_keys = sorted({(int(row) // 2, int(column) // 2) for row, column in occupied})
        total_samples += len(tile_keys)
        if total_samples > SOURCE_VIEW_SAMPLE_CAP:
            raise SourceViewFactError("source-view sample cap exceeded")
        sample_rows: list[int] = []
        sample_columns: list[int] = []
        sample_weights: list[int] = []
        local_x: list[int] = []
        local_y: list[int] = []
        local_z: list[int] = []
        for tile_row, tile_column in tile_keys:
            tile = occupied[
                (occupied[:, 0] // 2 == tile_row)
                & (occupied[:, 1] // 2 == tile_column)
            ]
            candidates = [
                (float(depth[row, column]), int(row), int(column))
                for row, column in tile
                if math.isfinite(float(depth[row, column]))
                and float(depth[row, column]) > 0.0
            ]
            if not candidates:
                raise SourceViewFactError("occupied source tile has no valid depth")
            source_depth, row, column = min(candidates)
            camera_point = np.asarray(
                [
                    (column - cx) * source_depth / fx,
                    (cy - row) * source_depth / fy,
                    source_depth,
                    1.0,
                ],
                dtype=np.float64,
            )
            world = camera_to_world @ camera_point
            if not np.all(np.isfinite(world)) or world[3] == 0.0:
                raise SourceViewFactError("source-view back-projection is invalid")
            world = world[:3] / world[3]
            local = (
                float(world[0]) - obj.position.x,
                float(world[1]) - obj.position.y,
                float(world[2]) - obj.position.z,
            )
            sample_rows.append(row)
            sample_columns.append(column)
            sample_weights.append(len(tile))
            local_x.append(_quantize_local_coordinate(local[0]))
            local_y.append(_quantize_local_coordinate(local[1]))
            local_z.append(_quantize_local_coordinate(local[2]))
        rows.append(
            SourceViewObjectSamples(
                object_id=object_id,
                source_mask_bbox=bbox,
                source_mask_pixel_count=pixel_count,
                sample_rows=tuple(sample_rows),
                sample_columns=tuple(sample_columns),
                sample_weights=tuple(sample_weights),
                local_x_quantized=tuple(local_x),
                local_y_quantized=tuple(local_y),
                local_z_quantized=tuple(local_z),
            )
        )
    if not rows:
        raise SourceViewFactError("source-view fact has no visible objects")
    payload = {
        "fact_version": SOURCE_VIEW_FACT_VERSION,
        "source_id": source.source_id,
        "scene_id": scene.scene_id,
        "source_locator_sha256": source.source_locator_sha256,
        "runtime_identity_sha256": _sha(runtime_identity),
        "scene_sha256": _sha(scene),
        "camera_sha256": _sha(camera),
        "rgb_png_sha256": observation.rgb_png_sha256,
        "depth_npy_sha256": observation.depth_npy_sha256,
        "instance_png_sha256": observation.instance_png_sha256,
        "sampling_policy_sha256": SOURCE_VIEW_SAMPLING_POLICY_SHA256,
        "objects": tuple(rows),
    }
    return SourceViewFact(
        **payload,
        source_view_fact_sha256=canonical_sha256(payload, domain=_FACT_DOMAIN),
    )
