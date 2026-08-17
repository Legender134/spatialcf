"""Deterministic AI2-THOR-shaped runtime used only by public smoke tests."""

from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from spatialcf.adapters.ai2thor import AI2ThorAdapter, AI2ThorRuntimeIdentity
from spatialcf.solver.analytic_motion import AnalyticMotionModel


def _box_corners(
    center: dict[str, float],
    size: dict[str, float],
    rotation_y: float,
) -> list[list[float]]:
    angle = math.radians(rotation_y)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [
            center["x"] + cosine * dx * size["x"] / 2 + sine * dz * size["z"] / 2,
            center["y"] + dy * size["y"] / 2,
            center["z"] - sine * dx * size["x"] / 2 + cosine * dz * size["z"] / 2,
        ]
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]


def _object(
    object_id: str,
    name: str,
    *,
    x: float,
    z: float,
    movable: bool,
) -> dict[str, Any]:
    center = {"x": x, "y": 0.5, "z": z}
    size = {"x": 1.0, "y": 1.0, "z": 0.5}
    return {
        "objectId": object_id,
        "name": name,
        "objectType": "Chair" if object_id == "chair-id" else "Table",
        "moveable": movable,
        "pickupable": False,
        "isMoving": False,
        "position": dict(center),
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "axisAlignedBoundingBox": {"center": dict(center), "size": dict(size)},
        "objectOrientedBoundingBox": {"cornerPoints": _box_corners(center, size, 0.0)},
        "parentReceptacles": [],
    }


def _metadata() -> dict[str, Any]:
    return {
        "lastActionSuccess": True,
        "errorMessage": "",
        "sceneName": "FloorPlan2",
        "isSceneAtRest": True,
        "fov": 90.0,
        "cameraPosition": {"x": 1.0, "y": 1.5, "z": -2.0},
        "cameraHorizon": 0.0,
        "agent": {
            "position": {"x": 1.0, "y": 0.0, "z": -2.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "cameraHorizon": 0.0,
            "isStanding": True,
        },
        "sceneBounds": {
            "center": {"x": 0.0, "y": 1.5, "z": 1.0},
            "size": {"x": 6.0, "y": 3.0, "z": 8.0},
        },
        "objects": [
            _object("chair-id", "Chair|0", x=0.0, z=1.0, movable=True),
            _object("table-id", "Table|0", x=0.0, z=1.5, movable=False),
            _object("lamp-id", "Lamp|0", x=0.0, z=4.0, movable=True),
        ],
    }


class FakeEvent:
    def __init__(self, *, width: int, height: int) -> None:
        self.metadata = _metadata()
        support = self.metadata["objects"][1]
        support["position"] = {"x": 0.0, "y": 0.45, "z": 1.5}
        support["axisAlignedBoundingBox"] = {
            "center": dict(support["position"]),
            "size": {"x": 8.0, "y": 0.1, "z": 8.0},
        }
        support["objectOrientedBoundingBox"] = None
        self.metadata["objects"][0]["parentReceptacles"] = ["table-id"]
        self.metadata["objects"][2]["objectType"] = "Lamp"
        self.frame = np.full((height, width, 3), 17, dtype=np.uint8)
        self.depth_frame = np.full((height, width), 2.5, dtype=np.float32)
        self.instance_segmentation_frame = np.full(
            (height, width, 3), (1, 2, 3), dtype=np.uint8
        )
        self.instance_detections2D = {
            "chair-id": _projected_detection(0.0, 1.0, width, height),
            "table-id": np.asarray(
                [width * 0.5, height / 6, width * 0.9, height * 0.9]
            ),
            "lamp-id": _projected_detection(0.0, 4.0, width, height),
        }
        self.instance_masks = {
            object_id: np.full((height, width), True, dtype=bool)
            for object_id in ("chair-id", "table-id", "lamp-id")
        }

    def clone(self) -> FakeEvent:
        copied = object.__new__(FakeEvent)
        copied.metadata = deepcopy(self.metadata)
        copied.frame = self.frame.copy()
        copied.depth_frame = self.depth_frame.copy()
        copied.instance_segmentation_frame = self.instance_segmentation_frame.copy()
        copied.instance_detections2D = deepcopy(self.instance_detections2D)
        copied.instance_masks = {
            object_id: mask.copy() for object_id, mask in self.instance_masks.items()
        }
        return copied


def _projected_detection(
    x: float,
    z: float,
    width: int,
    height: int,
) -> np.ndarray:
    depth = z + 2.0
    focal = width / 2
    center_x = focal * (x - 1.0) / depth + width / 2
    center_y = height / 2 + focal / depth
    return np.asarray(
        [
            center_x - width / 16,
            center_y - height / 6,
            center_x + width / 16,
            center_y + height / 6,
        ]
    )


class FakeController:
    def __init__(self, event: FakeEvent, **kwargs: Any) -> None:
        self.scene = kwargs["scene"]
        self.last_event = event.clone()
        self.last_event.metadata["sceneName"] = str(self.scene)
        self._build = SimpleNamespace(commit_id="public-readme-fake")
        self.reachable_positions: object = [
            {"x": 1.0, "y": 0.0, "z": -2.0},
            {"x": 2.0, "y": 0.0, "z": -1.0},
        ]
        self.receptacle_spawn_positions: object = []
        self.stop_calls = 0

    def reset(self, scene: Any) -> FakeEvent:
        self.scene = scene
        self.last_event.metadata["sceneName"] = str(scene)
        return self.last_event

    def step(self, **action: Any) -> FakeEvent:
        event = self.last_event.clone()
        name = action["action"]
        if name == "GetReachablePositions":
            event.metadata["actionReturn"] = deepcopy(self.reachable_positions)
        elif name == "GetSpawnCoordinatesAboveReceptacle":
            event.metadata["actionReturn"] = deepcopy(self.receptacle_spawn_positions)
        elif name == "TeleportFull":
            event.metadata["agent"]["position"] = deepcopy(action["position"])
            event.metadata["agent"]["rotation"] = deepcopy(action["rotation"])
            event.metadata["agent"]["cameraHorizon"] = action["horizon"]
            event.metadata["agent"]["isStanding"] = action["standing"]
            event.metadata["cameraHorizon"] = action["horizon"]
            event.metadata["cameraPosition"] = {
                "x": action["position"]["x"],
                "y": action["position"]["y"] + (1.5 if action["standing"] else 0.9),
                "z": action["position"]["z"],
            }
        elif name in {"PlaceObjectAtPoint", "TeleportObject"}:
            self._move_object(event, action)
        event.metadata["lastActionSuccess"] = True
        event.metadata["errorMessage"] = ""
        self.last_event = event
        return event

    def _move_object(self, event: FakeEvent, action: dict[str, Any]) -> None:
        item = next(
            current
            for current in event.metadata["objects"]
            if current["objectId"] == action["objectId"]
        )
        old_id = item["objectId"]
        old_position = item["position"]
        requested = action["position"]
        new_position = {
            "x": requested["x"],
            "y": requested.get("y", old_position["y"]),
            "z": requested["z"],
        }
        delta = {
            axis: new_position[axis] - old_position[axis] for axis in ("x", "y", "z")
        }
        item["position"] = new_position
        item["rotation"] = deepcopy(action["rotation"])
        for axis in ("x", "y", "z"):
            item["axisAlignedBoundingBox"]["center"][axis] += delta[axis]
        oriented = item.get("objectOrientedBoundingBox")
        if oriented is not None:
            for point in oriented["cornerPoints"]:
                for index, axis in enumerate(("x", "y", "z")):
                    point[index] += delta[axis]
        new_id = (
            f"{item['name']}|{new_position['x']:+.2f}|"
            f"{new_position['y']:+.2f}|{new_position['z']:+.2f}"
        )
        item["objectId"] = new_id
        for current in event.metadata["objects"]:
            current["parentReceptacles"] = [
                new_id if parent == old_id else parent
                for parent in current.get("parentReceptacles", [])
            ]
        if old_id in event.instance_detections2D:
            event.instance_detections2D[new_id] = event.instance_detections2D.pop(
                old_id
            )
        if old_id in event.instance_masks:
            event.instance_masks[new_id] = event.instance_masks.pop(old_id)
        item["isMoving"] = False

    def stop(self) -> None:
        self.stop_calls += 1


class ControllerFactory:
    def __init__(self, event: FakeEvent) -> None:
        self.event = event

    def __call__(self, **kwargs: Any) -> FakeController:
        return FakeController(self.event, **kwargs)


def _install_synthetic_runtime_identity(adapter: AI2ThorAdapter) -> None:
    """Keep the exact Adapter type while removing installed-runtime dependency."""

    def runtime_identity() -> AI2ThorRuntimeIdentity:
        controller = adapter._require_active()
        if adapter._event is None:
            raise RuntimeError("no synthetic runtime event is available")
        adapter._validate_scene_source_or_poison(
            controller,
            adapter.scene_name,
            adapter._event,
        )
        build = getattr(controller, "_build", None)
        commit_id = getattr(build, "commit_id", None)
        if type(commit_id) is not str or not commit_id:
            raise RuntimeError("synthetic runtime has no build identity")
        return AI2ThorRuntimeIdentity(
            ai2thor_version="synthetic-public-smoke",
            unity_commit_id=commit_id,
            native_scene_name=adapter._native_scene_name(adapter._event),
            width=adapter.width,
            height=adapter.height,
            seed=adapter.seed,
        )

    adapter.runtime_identity = runtime_identity  # type: ignore[method-assign]


def _project_native_edit_from_live_source(adapter: AI2ThorAdapter) -> None:
    controller = adapter.controller
    original_step = controller.step

    def projecting_step(**action: Any) -> FakeEvent:
        source = adapter._current_scene
        event = original_step(**action)
        if (
            action.get("action") == "PlaceObjectAtPoint"
            and event.metadata.get("lastActionSuccess") is True
        ):
            assert source is not None
            subject = next(
                item for item in event.metadata["objects"] if item["name"] == "Chair|0"
            )
            view = AnalyticMotionModel().projected_view(
                source,
                "chair-id",
                "main",
                float(subject["position"]["x"]),
                float(subject["position"]["z"]),
            )
            event.instance_detections2D[subject["objectId"]] = np.asarray(
                [view.bbox.xmin, view.bbox.ymin, view.bbox.xmax, view.bbox.ymax]
            )
        return event

    controller.step = projecting_step


_CAPTURE_CALL = object()


@dataclass
class READMEAdapterFactory:
    constructions: int = 0

    def __call__(
        self,
        scene_names: list[str] | tuple[str, ...],
        *,
        width: int,
        height: int,
        seed: int,
        procedural_scenes: object = _CAPTURE_CALL,
    ):
        del procedural_scenes
        self.constructions += 1
        event = FakeEvent(width=width, height=height)
        controller_factory = ControllerFactory(event)

        @contextmanager
        def manager():
            adapter = AI2ThorAdapter(
                list(scene_names),
                width=width,
                height=height,
                seed=seed,
                controller_factory=controller_factory,
            )
            _install_synthetic_runtime_identity(adapter)
            with adapter:
                adapter.controller.receptacle_spawn_positions = [
                    {
                        "x": -2.0 + 4.0 * x_index / 20.0,
                        "y": 0.5,
                        "z": -1.0 + 5.0 * y_index / 20.0,
                    }
                    for x_index in range(21)
                    for y_index in range(21)
                ]
                _project_native_edit_from_live_source(adapter)
                yield adapter

        return manager()
