"""Native scene and observation capture for the staged AI2-THOR adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spatialcf.adapters.ai2thor.conversion import (
    _EPSILON,
    _quaternion_yaw,
    _rotation_matrix,
    ai2thor_camera_world_to_camera,
)
from spatialcf.adapters.ai2thor.models import (
    _TELEPORT_VERTICAL_GUARD_M,
    AI2ThorAgentPose,
    AI2ThorNativeReturnError,
    AI2ThorObservation,
    AI2ThorRuntimeIdentity,
    AI2ThorSettlementTimeout,
    _canonical_house_json_bytes,
    adapter_floor_envelope_from_native,
    adapter_observation_from_native,
    adapter_pose_from_native,
    adapter_position_from_native,
    adapter_runtime_identity_from_native,
    adapter_support_fact_from_native,
)
from spatialcf.adapters.ai2thor.support import (
    _STRUCTURAL_OBJECT_TYPES,
    _domain_object_metadata,
    _validated_native_object_metadata,
    build_navigation_feasibility_map,
)
from spatialcf.adapters.base import (
    AdapterOperationError,
    AdapterSettlementTimeout,
    CapturedSource,
    RenderedAssets,
    SourceCaptureFacts,
    SourceCaptureOptions,
)
from spatialcf.domain.scene import (
    OBB,
    BBox2D,
    Camera,
    ObjectView,
    Quaternion,
    Scene,
    SceneObject,
    Vec2,
    Vec3,
)
from spatialcf.geometry.transforms import (
    ai2thor_position_to_world,
    ai2thor_rotation_to_world,
    matrix4,
    transform_point,
)


# These module-level callables retain the exact frozen method bodies. Formatting
# stays disabled so body-level AST parity also preserves multiline docstrings.
# fmt: off
def canonical_procedural_house_sha256(house: dict[str, Any]) -> str:
    """Return the canonical source digest without imposing a room policy."""
    if type(house) is not dict:
        raise ValueError("procedural house root must be an exact dict")
    return sha256(_canonical_house_json_bytes(house)).hexdigest()

def _camera(self, metadata: dict[str, Any], scene_id: str) -> Camera:
    try:
        fov = math.radians(float(metadata["fov"]))
        position = ai2thor_position_to_world(Vec3(**metadata["cameraPosition"]))
        agent_metadata = metadata["agent"]
        yaw_degrees = float(agent_metadata["rotation"]["y"])
        if "cameraHorizon" in metadata:
            horizon = metadata["cameraHorizon"]
        else:
            horizon = agent_metadata["cameraHorizon"]
        horizon_degrees = float(horizon)
    except (KeyError, TypeError, ValueError) as exc:
        raise AI2ThorNativeReturnError("invalid AI2-THOR camera metadata") from exc
    if not (0.0 < fov < math.pi):
        raise AI2ThorNativeReturnError(
            "camera field of view must be between 0 and 180 degrees"
        )
    focal = self.height / (2.0 * math.tan(fov / 2.0))
    intrinsics = (
        focal,
        0.0,
        self.width / 2.0,
        0.0,
        focal,
        self.height / 2.0,
        0.0,
        0.0,
        1.0,
    )
    camera = Camera(
        camera_id="main",
        width=self.width,
        height=self.height,
        intrinsics=tuple(float(value) for value in intrinsics),
        world_to_camera=ai2thor_camera_world_to_camera(
            position,
            yaw_degrees=yaw_degrees,
            horizon_degrees=horizon_degrees,
        ),
    )
    agent = metadata.get("agent")
    if not isinstance(agent, dict):
        raise AI2ThorNativeReturnError("invalid AI2-THOR agent metadata")
    try:
        state = {
            "position": {
                axis: float(agent["position"][axis]) for axis in ("x", "y", "z")
            },
            "rotation": {
                axis: float(agent["rotation"][axis]) for axis in ("x", "y", "z")
            },
            "horizon": float(horizon),
            "standing": bool(agent.get("isStanding", True)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AI2ThorNativeReturnError(
            "invalid AI2-THOR agent pose metadata"
        ) from exc
    self._camera_states[self._camera_key(scene_id, camera)] = state
    return camera

def _oriented_bounds(
    metadata: dict[str, Any],
    position: Vec3,
    rotation: Quaternion,
) -> OBB:
    oriented = metadata.get("objectOrientedBoundingBox")
    corners = oriented.get("cornerPoints") if isinstance(oriented, dict) else None
    if corners is not None:
        try:
            array = np.asarray(
                [
                    (
                        float(point["x"]),
                        float(point["y"]),
                        float(point["z"]),
                    )
                    if isinstance(point, dict)
                    else tuple(float(value) for value in point)
                    for point in corners
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                "invalid oriented bounding-box corners"
            ) from exc
        if array.shape != (8, 3) or not np.isfinite(array).all():
            raise AI2ThorNativeReturnError(
                "oriented bounding box must have eight finite corners"
            )
        world = array[:, [0, 2, 1]]
        center_array = world.mean(axis=0)
        yaw = _quaternion_yaw(rotation)
        planar_rotation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )
        # Geometry consumers use an upright Z-up OBB. For tilted objects,
        # bound the projected corners in the object's yaw frame so the
        # ground footprint is conservative instead of under-estimated.
        local = (world - center_array) @ _rotation_matrix(planar_rotation)
        extent_array = np.ptp(local, axis=0)
        if np.any(extent_array <= _EPSILON):
            raise AI2ThorNativeReturnError(
                "oriented bounding-box extents must be positive"
            )
        return OBB(
            center=Vec3(
                x=float(center_array[0]),
                y=float(center_array[1]),
                z=float(center_array[2]),
            ),
            extent=Vec3(
                x=float(extent_array[0]),
                y=float(extent_array[1]),
                z=float(extent_array[2]),
            ),
            rotation=planar_rotation,
        )

    bounds = metadata.get("axisAlignedBoundingBox")
    if not isinstance(bounds, dict):
        raise AI2ThorNativeReturnError("object is missing bounding-box metadata")
    try:
        center = ai2thor_position_to_world(Vec3(**bounds["center"]))
        size = bounds["size"]
        extent = Vec3(
            x=float(size["x"]),
            y=float(size["z"]),
            z=float(size["y"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AI2ThorNativeReturnError(
            "invalid axis-aligned bounding-box metadata"
        ) from exc
    if (
        not all(
            math.isfinite(value)
            for value in (
                center.x,
                center.y,
                center.z,
                extent.x,
                extent.y,
                extent.z,
            )
        )
        or min(extent.x, extent.y, extent.z) <= _EPSILON
    ):
        raise AI2ThorNativeReturnError(
            "axis-aligned bounding-box values must be finite and positive"
        )
    return OBB(
        center=center,
        extent=extent,
        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

def _obb_corners(obb: OBB) -> np.ndarray:
    offsets = np.asarray(
        [
            [dx * obb.extent.x / 2, dy * obb.extent.y / 2, dz * obb.extent.z / 2]
            for dx in (-1.0, 1.0)
            for dy in (-1.0, 1.0)
            for dz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    center = np.asarray([obb.center.x, obb.center.y, obb.center.z])
    return offsets @ _rotation_matrix(obb.rotation).T + center

def _projected_bounds(
    self,
    obb: OBB,
    camera: Camera,
) -> tuple[float, float, float, float]:
    projected: list[tuple[float, float]] = []
    extrinsics = matrix4(camera.world_to_camera)
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    for point in self._obb_corners(obb):
        camera_point = transform_point(
            extrinsics,
            Vec3(x=float(point[0]), y=float(point[1]), z=float(point[2])),
        )
        if camera_point.z <= _EPSILON:
            continue
        projected.append(
            (
                fx * camera_point.x / camera_point.z + cx,
                cy - fy * camera_point.y / camera_point.z,
            )
        )
    if not projected:
        raise AI2ThorNativeReturnError(
            "visible object bounds are behind the camera"
        )
    xs, ys = zip(*projected)
    return min(xs), min(ys), max(xs), max(ys)

def _view(
    self,
    metadata: dict[str, Any],
    position: Vec3,
    obb: OBB,
    camera: Camera,
    event: Any,
) -> dict[str, ObjectView]:
    object_id = metadata["objectId"]
    detections = getattr(event, "instance_detections2D", {}) or {}
    masks = getattr(event, "instance_masks", {}) or {}
    detection = detections.get(object_id)
    mask = masks.get(object_id)
    if detection is None or mask is None:
        return {}
    values = np.asarray(detection, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise AI2ThorNativeReturnError(f"invalid detection for {object_id!r}")
    xmin, ymin, xmax, ymax = (float(value) for value in values)
    if xmax <= xmin or ymax <= ymin:
        if metadata.get("visible") is False:
            return {}
        if (
            type(mask) is not np.ndarray
            or mask.shape != (self.height, self.width)
            or mask.dtype != np.bool_
        ):
            raise AI2ThorNativeReturnError(
                f"invalid instance mask for {object_id!r}"
            )
        mask_array = mask
        pixels = np.argwhere(mask_array)
        if pixels.size == 0:
            raise AI2ThorNativeReturnError(
                f"invalid detection bounds for {object_id!r}"
            )
        ymin = float(pixels[:, 0].min())
        ymax = float(pixels[:, 0].max() + 1)
        xmin = float(pixels[:, 1].min())
        xmax = float(pixels[:, 1].max() + 1)
    else:
        mask_array = np.asarray(mask)
        if mask_array.shape != (self.height, self.width):
            raise AI2ThorNativeReturnError(
                f"invalid instance mask shape for {object_id!r}"
            )
    camera_point = transform_point(matrix4(camera.world_to_camera), position)
    if not math.isfinite(camera_point.z):
        raise AI2ThorNativeReturnError(
            f"visible object {object_id!r} has non-positive camera depth"
        )
    if camera_point.z <= _EPSILON:
        # AI2-THOR object anchors are not guaranteed to be OBB centres.
        # Large fixtures can therefore cross the camera plane while their
        # valid anchor is behind it.  They remain scene geometry, but do
        # not expose a relation view whose anchor depth is non-positive.
        # A box wholly on either side while its anchor disagrees is still
        # inconsistent native metadata and fails closed.
        extrinsics = matrix4(camera.world_to_camera)
        corner_depths = tuple(
            transform_point(
                extrinsics,
                Vec3(x=float(corner[0]), y=float(corner[1]), z=float(corner[2])),
            ).z
            for corner in self._obb_corners(obb)
        )
        if min(corner_depths) < 0.0 < max(corner_depths):
            return {}
        raise AI2ThorNativeReturnError(
            f"visible object {object_id!r} has non-positive camera depth"
        )

    clipped = (
        max(0.0, min(float(self.width), xmin)),
        max(0.0, min(float(self.height), ymin)),
        max(0.0, min(float(self.width), xmax)),
        max(0.0, min(float(self.height), ymax)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise AI2ThorNativeReturnError(
            f"visible detection for {object_id!r} is outside the image"
        )
    projected = self._projected_bounds(obb, camera)
    projected_area = max(
        _EPSILON,
        (projected[2] - projected[0]) * (projected[3] - projected[1]),
    )
    projected_clipped_width = max(
        0.0, min(projected[2], self.width) - max(projected[0], 0.0)
    )
    projected_clipped_height = max(
        0.0, min(projected[3], self.height) - max(projected[1], 0.0)
    )
    bbox_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    image_area = float(self.width * self.height)
    return {
        "main": ObjectView(
            camera_id="main",
            bbox=BBox2D(
                xmin=clipped[0],
                ymin=clipped[1],
                xmax=clipped[2],
                ymax=clipped[3],
            ),
            camera_depth=float(camera_point.z),
            visible_fraction=float(
                np.clip(np.count_nonzero(mask_array) / projected_area, 0.0, 1.0)
            ),
            image_area_fraction=float(np.clip(bbox_area / image_area, 0.0, 1.0)),
            truncated_fraction=float(
                np.clip(
                    1.0
                    - projected_clipped_width
                    * projected_clipped_height
                    / projected_area,
                    0.0,
                    1.0,
                )
            ),
        )
    }

def _object(
    self,
    metadata: dict[str, Any],
    camera: Camera,
    event: Any,
    structural_object_ids: frozenset[str],
) -> SceneObject:
    try:
        position = ai2thor_position_to_world(Vec3(**metadata["position"]))
        rotation = ai2thor_rotation_to_world(Vec3(**metadata["rotation"]))
        object_id = metadata["objectId"]
        name = metadata["name"]
        category = metadata["objectType"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AI2ThorNativeReturnError("invalid AI2-THOR object metadata") from exc
    obb = self._oriented_bounds(metadata, position, rotation)
    parents = metadata["parentReceptacles"]
    parents = [parent for parent in parents if parent not in structural_object_ids]
    return SceneObject(
        object_id=object_id,
        name=name,
        category=category,
        movable=(
            metadata.get("moveable") is True or metadata.get("pickupable") is True
        ),
        position=position,
        rotation=rotation,
        obb=obb,
        support_object_id=parents[0] if parents else None,
        views=self._view(metadata, position, obb, camera, event),
    )

def _without_cyclic_support_assignments(
    objects: tuple[SceneObject, ...],
) -> tuple[SceneObject, ...]:
    """Drop every edge in a cyclic native receptacle component.

        AI2-THOR's ``parentReceptacles`` describes receptacle membership, not
        a certified physical support tree, and real scenes can report cycles.
        A cycle has no honest single supporting parent, so retain the objects
        but represent those assignments as unknown instead of guessing an
        edge or passing an invalid graph to the solver.
        """
    object_ids = {obj.object_id for obj in objects}
    parents = {
        obj.object_id: (
            obj.support_object_id if obj.support_object_id in object_ids else None
        )
        for obj in objects
    }
    cyclic_ids: set[str] = set()
    for start in sorted(parents):
        path: list[str] = []
        path_index: dict[str, int] = {}
        current: str | None = start
        while current is not None and current in parents:
            if current in path_index:
                cyclic_ids.update(path[path_index[current] :])
                break
            path_index[current] = len(path)
            path.append(current)
            current = parents[current]
    if not cyclic_ids:
        return objects
    return tuple(
        obj.model_copy(update={"support_object_id": None})
        if obj.object_id in cyclic_ids
        else obj
        for obj in objects
    )

def _scene_from_event(self, scene_id: str, event: Any) -> Scene:
    metadata = event.metadata
    camera = self._camera(metadata, scene_id)
    raw_objects = metadata.get("objects")
    if not isinstance(raw_objects, list):
        raise AI2ThorNativeReturnError("invalid AI2-THOR object collection")
    raw_objects = _validated_native_object_metadata(raw_objects)
    structural_object_ids = frozenset(
        item["objectId"]
        for item in raw_objects
        if item["objectType"] in _STRUCTURAL_OBJECT_TYPES
    )
    raw_objects = _domain_object_metadata(raw_objects)
    names = [item["name"] for item in raw_objects]
    for item, name in zip(raw_objects, names, strict=True):
        try:
            rotation = {
                axis: float(item["rotation"][axis]) for axis in ("x", "y", "z")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise AI2ThorNativeReturnError(
                f"invalid native rotation for {name!r}"
            ) from exc
        if not all(math.isfinite(value) for value in rotation.values()):
            raise AI2ThorNativeReturnError(f"invalid native rotation for {name!r}")
        self._native_rotations[(scene_id, name)] = rotation
    objects = tuple(
        self._object(
            item,
            camera,
            event,
            structural_object_ids,
        )
        for item in raw_objects
    )
    objects = self._without_cyclic_support_assignments(objects)
    scene_bounds = metadata.get("sceneBounds")
    if not isinstance(scene_bounds, dict):
        raise AI2ThorNativeReturnError("invalid AI2-THOR scene bounds")
    try:
        center = ai2thor_position_to_world(Vec3(**scene_bounds["center"]))
        size = scene_bounds["size"]
        half_x = float(size["x"]) / 2.0
        half_y = float(size["z"]) / 2.0
    except (KeyError, TypeError, ValueError) as exc:
        raise AI2ThorNativeReturnError("invalid AI2-THOR scene bounds") from exc
    if (
        not all(
            math.isfinite(value)
            for value in (center.x, center.y, center.z, half_x, half_y)
        )
        or half_x <= 0.0
        or half_y <= 0.0
    ):
        raise AI2ThorNativeReturnError(
            "scene ground bounds must be finite and positive"
        )
    return Scene(
        scene_id=scene_id,
        source="ai2thor",
        room_polygon_xy=(
            Vec2(x=center.x - half_x, y=center.y - half_y),
            Vec2(x=center.x + half_x, y=center.y - half_y),
            Vec2(x=center.x + half_x, y=center.y + half_y),
            Vec2(x=center.x - half_x, y=center.y + half_y),
        ),
        cameras=(camera,),
        objects=objects,
        generation_seed=self.seed,
    )

def _objects_by_name(
    objects: tuple[SceneObject, ...],
) -> dict[str, SceneObject]:
    by_name = {obj.name: obj for obj in objects}
    if len(by_name) != len(objects):
        raise ValueError("scene object names must be unique")
    return by_name

def _stable_observed_scene(
    cls,
    source: Scene,
    observed: Scene,
) -> Scene:
    """Map native ID churn back to source IDs through unique object names."""
    source_by_name = cls._objects_by_name(source.objects)
    observed_by_name = cls._objects_by_name(observed.objects)
    if set(source_by_name) != set(observed_by_name):
        raise AI2ThorNativeReturnError(
            "stable object names changed during pose application"
        )

    aliases: dict[str, str] = {}

    def register_alias(native_id: str, stable_id: str) -> None:
        existing = aliases.get(native_id)
        if existing is not None and existing != stable_id:
            raise AI2ThorNativeReturnError(
                "native object IDs do not map to unique stable IDs"
            )
        aliases[native_id] = stable_id

    for name, original in source_by_name.items():
        current = observed_by_name[name]
        register_alias(current.object_id, original.object_id)
    source_object_ids = {obj.object_id for obj in source.objects}
    for original in source.objects:
        support_id = original.support_object_id
        if support_id is not None and support_id not in source_object_ids:
            register_alias(support_id, support_id)

    stable_objects: list[SceneObject] = []
    for original in source.objects:
        current = observed_by_name[original.name]
        support_id = current.support_object_id
        stable_support_id = None
        if support_id is not None:
            stable_support_id = aliases.get(support_id)
            if stable_support_id is None:
                raise AI2ThorNativeReturnError(
                    f"observed support {support_id!r} has no stable object identity"
                )
        stable_updates: dict[str, object] = {
            "object_id": original.object_id,
            "support_object_id": stable_support_id,
        }
        if "request_eligible" in original.model_fields_set:
            stable_updates["request_eligible"] = original.request_eligible
        stable_objects.append(current.model_copy(update=stable_updates))
    return source.model_copy(
        update={
            "cameras": observed.cameras,
            "objects": tuple(stable_objects),
        }
    )

def _canonical_camera_observed_scene(
    cls,
    source: Scene,
    native_observed: Scene,
) -> Scene:
    """Retain fresh camera/views after proving source geometry unchanged."""
    stable = cls._stable_observed_scene(source, native_observed)
    cls._validate_camera_object_invariants(source, stable)
    stable_by_name = cls._objects_by_name(stable.objects)
    return source.model_copy(
        update={
            "cameras": stable.cameras,
            "objects": tuple(
                original.model_copy(
                    update={"views": stable_by_name[original.name].views}
                )
                for original in source.objects
            ),
        }
    )

def _png_bytes(frame: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(frame, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()

def _npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()

def _stable_instance_pixel_counts(
    self,
    scene: Scene,
    event: Any,
) -> dict[str, int]:
    raw_objects = event.metadata.get("objects")
    if not isinstance(raw_objects, list):
        raise AI2ThorNativeReturnError("observation returned no object metadata")
    native_by_name = {
        str(item.get("name")): item
        for item in _domain_object_metadata(raw_objects)
        if isinstance(item, dict)
    }
    if len(native_by_name) != len(_domain_object_metadata(raw_objects)):
        raise AI2ThorNativeReturnError(
            "observation returned duplicate object names"
        )
    masks = getattr(event, "instance_masks", None)
    if not isinstance(masks, Mapping):
        raise AI2ThorNativeReturnError(
            "observation returned invalid instance masks"
        )
    validated_masks: dict[str, np.ndarray] = {}
    for native_id in masks:
        if type(native_id) is not str:
            raise AI2ThorNativeReturnError(
                "observation returned invalid instance masks: keys must be strings"
            )
        mask = masks[native_id]
        if (
            not isinstance(mask, np.ndarray)
            or mask.shape != (self.height, self.width)
            or mask.dtype != np.bool_
        ):
            raise AI2ThorNativeReturnError(
                "observation returned invalid instance masks: "
                "values must be boolean HxW numpy arrays"
            )
        validated_masks[native_id] = mask
    counts: dict[str, int] = {}
    for obj in scene.objects:
        metadata = native_by_name.get(obj.name)
        if metadata is None:
            raise AI2ThorNativeReturnError(
                f"observation is missing stable object name {obj.name!r}"
            )
        native_id = str(metadata.get("objectId"))
        mask = validated_masks.get(native_id)
        if mask is None:
            counts[obj.object_id] = 0
            continue
        counts[obj.object_id] = int(np.count_nonzero(mask))
    return counts

def _observation_from_event(
    self,
    scene: Scene,
    event: Any,
) -> AI2ThorObservation:
    camera = scene.camera_by_id("main")
    rgb, depth, instance = self._validated_frames(event)
    return AI2ThorObservation.create(
        scene=scene,
        rgb_png=self._png_bytes(rgb),
        depth_npy=self._npy_bytes(depth),
        instance_png=self._png_bytes(instance),
        pointcloud_ply=self._pointcloud_bytes(camera, depth, rgb),
        instance_pixel_counts=self._stable_instance_pixel_counts(scene, event),
        is_scene_at_rest=self._native_scene_at_rest(event),
    )

def capture_current_observation(self, scene: Scene) -> AI2ThorObservation:
    """Capture frames from the exact current event without replaying poses."""
    event = self._current_event_for_scene(scene)
    return self._observation_from_event(scene, event)

def _validated_frames(
    self,
    event: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_event = self._event if event is None else event
    if source_event is None:
        raise RuntimeError("no AI2-THOR event is available")
    rgb = np.asarray(getattr(source_event, "frame", None))
    depth = np.asarray(getattr(source_event, "depth_frame", None))
    instance = np.asarray(
        getattr(source_event, "instance_segmentation_frame", None)
    )
    expected_color = (self.height, self.width, 3)
    expected_depth = (self.height, self.width)
    if rgb.shape != expected_color or rgb.dtype != np.uint8:
        raise AI2ThorNativeReturnError("AI2-THOR RGB frame must be HxWx3 uint8")
    if instance.shape != expected_color or instance.dtype != np.uint8:
        raise AI2ThorNativeReturnError(
            "AI2-THOR instance frame must be HxWx3 uint8"
        )
    if depth.shape != expected_depth or not np.issubdtype(depth.dtype, np.number):
        raise AI2ThorNativeReturnError(
            "AI2-THOR depth frame must be a numeric HxW array"
        )
    return rgb, depth.astype(np.float32, copy=False), instance

def _validate_stem(stem: str) -> None:
    if (
        not stem
        or stem.strip() != stem
        or stem in {".", ".."}
        or Path(stem).name != stem
        or "/" in stem
        or "\\" in stem
    ):
        raise ValueError("artifact stem must be a safe filename component")

def _pointcloud_bytes(
    self,
    camera: Camera,
    depth: np.ndarray,
    rgb: np.ndarray,
) -> bytes:
    rows, columns = np.mgrid[0 : self.height : 4, 0 : self.width : 4]
    z = depth[rows, columns]
    valid = np.isfinite(z) & (z > 0.0)
    rows, columns, z = rows[valid], columns[valid], z[valid]
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    x = (columns - cx) * z / fx
    y = -(rows - cy) * z / fy
    camera_points = np.stack([x, y, z, np.ones_like(z)])
    world_points = np.linalg.inv(matrix4(camera.world_to_camera)) @ camera_points
    colors = rgb[rows, columns]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {world_points.shape[1]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        (
            f"{world_points[0, index]:.9g} "
            f"{world_points[1, index]:.9g} "
            f"{world_points[2, index]:.9g} "
            f"{int(colors[index, 0])} "
            f"{int(colors[index, 1])} "
            f"{int(colors[index, 2])}"
        )
        for index in range(world_points.shape[1])
    )
    return ("\n".join(lines) + "\n").encode("ascii")

def _write_pointcloud(
    self,
    camera: Camera,
    depth: np.ndarray,
    rgb: np.ndarray,
    destination: Path,
) -> None:
    destination.write_bytes(self._pointcloud_bytes(camera, depth, rgb))

def render_assets(
    self,
    scene: Scene,
    camera_id: str,
    destination_root: Path,
    stem: str,
) -> RenderedAssets:
    self._require_active()
    camera = scene.camera_by_id(camera_id)
    if camera_id != "main":
        raise KeyError(camera_id)
    self._validate_stem(stem)
    self._restore_scene_state(scene)
    rgb, depth, instance = self._validated_frames()
    destination_root.mkdir(parents=True, exist_ok=True)
    assets = RenderedAssets(
        rgb_path=destination_root / f"{stem}-rgb.png",
        depth_path=destination_root / f"{stem}-depth.npy",
        instance_path=destination_root / f"{stem}-instance.png",
        pointcloud_path=destination_root / f"{stem}-pointcloud.ply",
    )
    if (
        len(
            {
                assets.rgb_path,
                assets.depth_path,
                assets.instance_path,
                assets.pointcloud_path,
            }
        )
        != 4
    ):
        raise ValueError("artifact paths must be distinct")
    Image.fromarray(rgb, mode="RGB").save(assets.rgb_path)
    np.save(assets.depth_path, depth, allow_pickle=False)
    Image.fromarray(instance, mode="RGB").save(assets.instance_path)
    self._write_pointcloud(camera, depth, rgb, assets.pointcloud_path)
    return assets


# fmt: on
class AI2ThorCaptureMixin:
    _camera = _camera
    _oriented_bounds = staticmethod(_oriented_bounds)
    _obb_corners = staticmethod(_obb_corners)
    _projected_bounds = _projected_bounds
    _view = _view
    _object = _object
    _without_cyclic_support_assignments = staticmethod(
        _without_cyclic_support_assignments
    )
    _scene_from_event = _scene_from_event
    _objects_by_name = staticmethod(_objects_by_name)
    _stable_observed_scene = classmethod(_stable_observed_scene)
    _canonical_camera_observed_scene = classmethod(_canonical_camera_observed_scene)
    _png_bytes = staticmethod(_png_bytes)
    _npy_bytes = staticmethod(_npy_bytes)
    _stable_instance_pixel_counts = _stable_instance_pixel_counts
    _observation_from_event = _observation_from_event
    capture_current_observation = capture_current_observation
    _validated_frames = _validated_frames
    _validate_stem = staticmethod(_validate_stem)
    _pointcloud_bytes = _pointcloud_bytes
    _write_pointcloud = _write_pointcloud
    render_assets = render_assets

    def list_scene_ids(self) -> list[str]:
        return list(self.scene_names)

    def observe_source(
        self,
        source: CapturedSource,
        *,
        options: SourceCaptureOptions,
        settle: bool,
    ) -> SourceCaptureFacts:
        """Capture one settled source through the neutral evidence protocol."""

        if type(source) is not CapturedSource:
            raise AdapterOperationError("source capture requires an exact source")
        if type(options) is not SourceCaptureOptions or type(settle) is not bool:
            raise AdapterOperationError("source capture options must be exact")
        if not settle:
            raise AdapterOperationError("source observation must settle")
        try:
            settlement = self.settle_scene_observed(
                source.scene,
                max_pass_steps=options.max_settlement_steps,
            )
            scene = settlement.observed_scene
            support_by_id = {
                item.object_id: adapter_support_fact_from_native(item)
                for item in self.native_support_facts(scene)
            }
            if set(support_by_id) != {item.object_id for item in scene.objects}:
                raise ValueError("support fact roster does not bind observed scene")
            support_facts = tuple(
                support_by_id[item.object_id] for item in scene.objects
            )
            floor_subjects = tuple(
                item
                for item in scene.objects
                if item.movable
                and support_by_id[item.object_id].support_kind == "FLOOR"
            )
            floor = None
            if floor_subjects:
                try:
                    floor = adapter_floor_envelope_from_native(
                        self.conservative_floor_envelope(
                            scene,
                            clearance_m=options.floor_clearance_m,
                        )
                    )
                except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError):
                    floor = None
            native_reachable = self.reachable_agent_positions(scene)
            reachable = tuple(
                adapter_position_from_native(item) for item in native_reachable
            )
            floor_regions = ()
            if floor is not None:
                regions = []
                for item in floor_subjects:
                    try:
                        navigation = build_navigation_feasibility_map(
                            scene,
                            subject_object_id=item.object_id,
                            room_polygon_xy=floor.polygon_xy,
                            reachable_positions=native_reachable,
                            agent_radius_m=options.navigation_agent_radius_m,
                            clearance_m=options.navigation_clearance_m,
                        )
                    except (
                        AI2ThorNativeReturnError,
                        RuntimeError,
                        ValueError,
                        KeyError,
                    ):
                        continue
                    if navigation.position_region.components:
                        regions.append((item.object_id, navigation.position_region))
                floor_regions = tuple(sorted(regions, key=lambda item: item[0]))
            return SourceCaptureFacts(
                source=source,
                binding=source.binding,
                scene=scene,
                runtime_identity=adapter_runtime_identity_from_native(
                    self.runtime_identity()
                ),
                observation=adapter_observation_from_native(settlement.observation),
                support_facts=support_facts,
                floor_envelope=floor,
                floor_position_regions=floor_regions,
                reachable_positions=reachable,
                current_pose=adapter_pose_from_native(self.current_agent_pose(scene)),
                settlement_pass_steps=settlement.pass_steps,
            )
        except AI2ThorSettlementTimeout as error:
            raise AdapterSettlementTimeout(str(error)) from error
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error

    def runtime_identity(self) -> AI2ThorRuntimeIdentity:
        """Return the active package/build/configuration identity without action."""
        controller = self._require_active()
        if self._event is None:
            raise RuntimeError("no AI2-THOR event is available")
        self._validate_scene_source_or_poison(
            controller,
            self.scene_name,
            self._event,
        )
        build = getattr(controller, "_build", None)
        commit_id = getattr(build, "commit_id", None)
        if type(commit_id) is not str or not commit_id:
            raise RuntimeError("AI2-THOR controller has no Unity build identity")
        try:
            installed_version = package_version("ai2thor")
        except PackageNotFoundError as error:
            raise RuntimeError("AI2-THOR package identity is unavailable") from error
        source = self.procedural_scenes.get(self.scene_name)
        return AI2ThorRuntimeIdentity(
            ai2thor_version=installed_version,
            unity_commit_id=commit_id,
            native_scene_name=self._native_scene_name(self._event),
            width=self.width,
            height=self.height,
            seed=self.seed,
            source_dataset_id=None if source is None else source.dataset_id,
            source_revision=None if source is None else source.revision,
            source_split=None if source is None else source.split,
            source_index=None if source is None else source.index,
            source_sha256=None if source is None else source.house_sha256,
            source_scene_alias=None if source is None else self.scene_name,
            source_loader_id=None if source is None else source.source_loader_id,
            source_loader_version=(
                None if source is None else source.source_loader_version
            ),
            source_room_id=None if source is None else source.room_id,
            source_floor_xz_bounds=(None if source is None else source.floor_xz_bounds),
            teleport_vertical_guard_m=_TELEPORT_VERTICAL_GUARD_M,
        )

    def latest_native_event(self, scene_id: str) -> Any:
        """Return latest raw event after source and native-scene revalidation."""
        controller = self._require_active()
        if scene_id not in self.scene_names:
            raise KeyError(scene_id)
        if self._latest_event is None:
            raise RuntimeError("adapter has no latest native event")
        self._validate_scene_source_or_poison(
            controller,
            scene_id,
            self._latest_event,
        )
        return self._latest_event

    def current_agent_pose(self, scene: Scene) -> AI2ThorAgentPose:
        """Read the agent pose from the exact current event without an action."""

        return self._native_agent_pose(self._current_event_for_scene(scene))

    def _current_event_for_scene(self, scene: Scene) -> Any:
        controller = self._require_active()
        if (
            self._event is None
            or self.scene_name != scene.scene_id
            or self._current_scene != scene
        ):
            raise RuntimeError(
                "scene must be the adapter's exact current scene and event"
            )
        self._validate_scene_source_or_poison(
            controller,
            scene.scene_id,
            self._event,
        )
        return self._event

    @staticmethod
    def _is_analysis_overlay(
        current: Scene,
        requested: Scene,
    ) -> bool:
        if current == requested:
            return False
        return (
            requested.model_copy(
                update={
                    "room_polygon_xy": current.room_polygon_xy,
                    "collision_obstacles": current.collision_obstacles,
                    "subject_position_regions": current.subject_position_regions,
                }
            )
            == current
        )

    @staticmethod
    def _native_scene_name(event: Any) -> str:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            raise TypeError("AI2-THOR event has no metadata")
        scene_name = metadata.get("sceneName")
        if type(scene_name) is not str or not scene_name:
            raise RuntimeError("AI2-THOR event has no valid native scene name")
        return scene_name

    @classmethod
    def _validate_native_scene_name(
        cls,
        event: Any,
        expected_native_scene_name: str,
    ) -> None:
        if cls._native_scene_name(event) != expected_native_scene_name:
            raise RuntimeError(
                "AI2-THOR event scene name changed during camera operation"
            )

    def _validate_native_scene_name_or_poison(
        self,
        event: Any,
        expected_native_scene_name: str,
    ) -> None:
        try:
            self._validate_native_scene_name(event, expected_native_scene_name)
        except BaseException:
            self._poison_scene_state()
            raise
