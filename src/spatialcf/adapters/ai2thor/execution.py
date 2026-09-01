from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from types import TracebackType
from typing import Any, Self
from weakref import ReferenceType, ref

import numpy as np

from spatialcf.adapters.ai2thor.camera import _NATIVE_NAVIGATION_GRID_SIZE_M
from spatialcf.adapters.ai2thor.conversion import _quaternions_close
from spatialcf.adapters.ai2thor.models import (
    _TELEPORT_VERTICAL_GUARD_M,
    AI2ThorIsolatedEpisode,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorPoseApplication,
    AI2ThorReceptacleSpawnMap,
    AI2ThorRuntimeError,
    AI2ThorSceneSettlement,
    AI2ThorSettlementTimeout,
    _canonical_house_json_bytes,
    _strict_finite_float,
)
from spatialcf.adapters.ai2thor.support import (
    _domain_object_metadata,
    _receptacle_scene_sha256,
    _strict_native_position,
    _strict_receptacle_spawn_map,
    _validated_native_object_metadata,
)
from spatialcf.domain.scene import Scene, SceneObject, Vec3
from spatialcf.geometry.transforms import (
    ai2thor_position_to_world,
    ai2thor_rotation_to_world,
)

# Unity round-trips placed-object Euler angles at about 3e-4 degrees while the
# independently checked quaternion geometry remains stable within 1e-5.
_OBJECT_ROTATION_TOLERANCE_DEGREES = 1e-3
_RUNTIME_RECEPTACLE_POSITION_RESIDUAL_M = 1e-4


class AI2ThorExecutionMixin:
    def __enter__(self) -> Self:
        if self.controller is not None:
            raise RuntimeError("adapter context is already active")
        self._stopped = False
        self.scene_name = self.scene_names[0]
        self._latest_event = None
        self._camera_states.clear()
        self._native_rotations.clear()
        try:
            self.controller = self._start_controller(
                scene=self._native_scene_input(self.scene_name),
                width=self.width,
                height=self.height,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                gridSize=_NATIVE_NAVIGATION_GRID_SIZE_M,
                snapToGrid=True,
                rotateStepDegrees=90,
            )
            self._validate_controller_source_or_poison(self.controller, self.scene_name)
            event = self._step(self.controller, "Pass", action="Pass")
            self._event = event
            self._current_scene = None
            self._event = self._checked_scene_event(
                self.controller,
                event,
                "Pass",
                self.scene_name,
            )
            return self
        except BaseException as error:
            try:
                self._stop()
            except AI2ThorRuntimeError as cleanup_error:
                error.add_note(f"AI2-THOR cleanup also failed: {cleanup_error}")
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._stop()
        except AI2ThorRuntimeError as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"AI2-THOR cleanup also failed: {cleanup_error}")

    def _stop(self) -> None:
        if self.controller is not None and not self._stopped:
            self._stopped = True
            controller = self.controller
            self.controller = None
            self._event = None
            self._latest_event = None
            self._current_scene = None
            try:
                controller.stop()
            except OSError as error:
                raise AI2ThorRuntimeError(
                    f"AI2-THOR controller stop I/O failed: {error}"
                ) from error
            except MemoryError:
                raise
            except Exception as error:
                raise AI2ThorRuntimeError(
                    f"AI2-THOR controller stop failed: {error}"
                ) from error
        else:
            self.controller = None
            self._event = None
            self._latest_event = None
            self._current_scene = None

    def _start_controller(self, **kwargs: Any) -> Any:
        try:
            return self._controller_factory(**kwargs)
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR controller startup I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR controller startup failed: {error}"
            ) from error

    @staticmethod
    def _step(controller: Any, label: str, **kwargs: Any) -> Any:
        try:
            return controller.step(**kwargs)
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR {label} step I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR {label} step failed: {error}"
            ) from error

    def _native_scene_input(self, scene_id: str) -> str | dict[str, Any]:
        source = self.procedural_scenes.get(scene_id)
        return scene_id if source is None else source.decode_house()

    def _poison_scene_state(self) -> None:
        self._event = None
        self._current_scene = None

    def _validate_controller_source(
        self,
        controller: Any,
        scene_id: str,
    ) -> None:
        source = self.procedural_scenes.get(scene_id)
        if source is None:
            controller_scene = getattr(controller, "scene", None)
            if type(controller_scene) is dict:
                raise RuntimeError("legacy source cannot use a controller house dict")
            if controller_scene not in {scene_id, f"{scene_id}_physics"}:
                raise RuntimeError(
                    "legacy controller scene does not match the registered scene"
                )
            return
        try:
            canonical = _canonical_house_json_bytes(controller.scene)
        except (AttributeError, ValueError) as error:
            raise RuntimeError(
                "controller procedural source SHA-256 cannot be verified"
            ) from error
        if sha256(canonical).hexdigest() != source.house_sha256:
            raise RuntimeError("controller procedural source SHA-256 changed")

    def _validate_controller_source_or_poison(
        self,
        controller: Any,
        scene_id: str,
    ) -> None:
        try:
            self._validate_controller_source(controller, scene_id)
        except BaseException:
            self._poison_scene_state()
            raise

    def _validate_scene_source_or_poison(
        self,
        controller: Any,
        scene_id: str,
        event: Any,
    ) -> None:
        try:
            self._validate_controller_source(controller, scene_id)
            native_scene_name = self._native_scene_name(event)
            if scene_id in self.procedural_scenes:
                if native_scene_name != "Procedural":
                    raise RuntimeError(
                        "registered procedural source requires native Procedural scene"
                    )
            else:
                if native_scene_name == "Procedural":
                    raise RuntimeError(
                        "legacy source cannot use native Procedural scene"
                    )
                if native_scene_name != controller.scene:
                    raise RuntimeError(
                        "legacy controller and event native scene names differ"
                    )
        except BaseException:
            self._poison_scene_state()
            raise

    def _checked_scene_event(
        self,
        controller: Any,
        event: Any,
        action: str,
        scene_id: str,
    ) -> Any:
        # Keep the raw returned event even when action or identity validation
        # fails.  Candidate rejection classification may inspect it later, but
        # that path does not treat it as trusted without independently
        # revalidating the relevant fields.
        self._latest_event = event
        try:
            checked = self._checked_event(event, action)
            self._validate_scene_source_or_poison(
                controller,
                scene_id,
                checked,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        return checked

    def _reset(self, controller: Any, scene_id: str) -> Any:
        try:
            return controller.reset(scene=self._native_scene_input(scene_id))
        except OSError as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR reset {scene_id} I/O failed: {error}"
            ) from error
        except MemoryError:
            raise
        except Exception as error:
            raise AI2ThorRuntimeError(
                f"AI2-THOR reset {scene_id} failed: {error}"
            ) from error

    def _require_active(self) -> Any:
        if self.controller is None:
            raise RuntimeError("adapter must be used as a context manager")
        return self.controller

    @staticmethod
    def _checked_event(event: Any, action: str) -> Any:
        metadata = getattr(event, "metadata", None)
        metadata_is_dict = isinstance(metadata, dict)
        if not metadata_is_dict:
            raise RuntimeError(f"{action} returned an event without metadata")
        if metadata.get("lastActionSuccess") is not True:
            message = metadata.get("errorMessage") or f"{action} failed"
            raise RuntimeError(str(message))
        return event

    @classmethod
    def _native_scene_fully_settled(
        cls,
        event: Any,
        expected_categories_by_name: dict[str, str],
    ) -> bool:
        raw_objects = event.metadata.get("objects")
        if type(raw_objects) is not list:
            raise AI2ThorNativeReturnError(
                "scene settlement returned no object metadata"
            )
        raw_objects = _validated_native_object_metadata(raw_objects)
        domain_objects = _domain_object_metadata(raw_objects)
        categories_by_name: dict[str, str] = {}
        any_object_moving = False
        for item in domain_objects:
            if type(item) is not dict:
                raise AI2ThorNativeReturnError(
                    "scene settlement returned invalid domain object metadata"
                )
            name = item.get("name")
            category = item.get("objectType")
            if (
                type(name) is not str
                or not name
                or type(category) is not str
                or not category
            ):
                raise AI2ThorNativeReturnError(
                    "scene settlement returned invalid object name/category metadata"
                )
            if name in categories_by_name:
                raise AI2ThorNativeReturnError(
                    "scene settlement returned duplicate stable object names"
                )
            categories_by_name[name] = category
            is_moving = item.get("isMoving")
            if type(is_moving) is not bool:
                raise AI2ThorNativeReturnError(
                    f"AI2-THOR isMoving for object {name!r} must be an exact boolean"
                )
            any_object_moving = any_object_moving or is_moving
        if categories_by_name != expected_categories_by_name:
            raise AI2ThorNativeReturnError(
                "object name/category mapping changed during scene settlement"
            )
        return cls._native_scene_at_rest(event) and not any_object_moving

    @staticmethod
    def _angle_residual_degrees(left: float, right: float) -> float:
        return abs((left - right + 180.0) % 360.0 - 180.0)

    def settle_scene_observed(
        self,
        scene: Scene,
        max_pass_steps: int = 30,
    ) -> AI2ThorSceneSettlement:
        """Advance physics until the scene and every domain object are still."""
        controller = self._require_active()
        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        event = self._current_event_for_scene(scene)
        expected_categories_by_name = {
            name: obj.category
            for name, obj in self._objects_by_name(scene.objects).items()
        }
        previous_camera_states = deepcopy(self._camera_states)
        previous_native_rotations = deepcopy(self._native_rotations)
        pass_steps = 0
        try:
            expected_native_scene_name = self._native_scene_name(event)
            while True:
                self._validate_native_scene_name_or_poison(
                    event,
                    expected_native_scene_name,
                )
                settled = self._native_scene_fully_settled(
                    event,
                    expected_categories_by_name,
                )
                if settled:
                    native_observed = self._scene_from_event(scene.scene_id, event)
                    observed_scene = self._stable_observed_scene(
                        scene,
                        native_observed,
                    )
                    observation = self._observation_from_event(
                        observed_scene,
                        event,
                    )
                    result = AI2ThorSceneSettlement(
                        observed_scene=observed_scene,
                        observation=observation,
                        pass_steps=pass_steps,
                    )
                    self._event = event
                    self._current_scene = observed_scene
                    return result
                if pass_steps >= max_pass_steps:
                    raise AI2ThorSettlementTimeout(
                        "AI2-THOR scene did not settle within "
                        f"{max_pass_steps} Pass steps"
                    )
                event = self._step(
                    controller,
                    "Pass",
                    action="Pass",
                )
                self._event = event
                self._current_scene = None
                pass_steps += 1
                event = self._checked_scene_event(
                    controller,
                    event,
                    "Pass",
                    scene.scene_id,
                )
        except BaseException:
            self._camera_states = previous_camera_states
            self._native_rotations = previous_native_rotations
            self._current_scene = None
            raise

    @contextmanager
    def isolated_scene_observed(
        self,
        source: Scene,
        max_pass_steps: int = 30,
    ) -> Iterator[AI2ThorIsolatedEpisode]:
        """Open one target-only controller and yield its settled source.

        The parent adapter remains the preparation owner and is never reset or
        mutated.  Native object IDs may differ in the child, so its observed
        baseline is rebound to the frozen source IDs and analysis overlays by
        unique object name.  Geometry, views, assets, and rest status all come
        from the child's same final trusted event.
        """
        parent_controller = self._require_active()
        if type(source) is not Scene:
            raise ValueError("source must be an exact canonical Scene")
        if source.scene_id not in self.scene_names:
            raise KeyError(source.scene_id)
        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        procedural = self.procedural_scenes.get(source.scene_id)

        def isolated_controller_factory(**kwargs: Any) -> Any:
            controller = self._controller_factory(**kwargs)
            if controller is parent_controller:
                raise RuntimeError(
                    "isolated AI2-THOR episode reused the parent controller"
                )
            live_refs: list[ReferenceType[Any]] = []
            for controller_ref in self._isolated_controller_refs:
                previous = controller_ref()
                if previous is None:
                    continue
                live_refs.append(controller_ref)
                if controller is previous:
                    raise RuntimeError(
                        "isolated AI2-THOR episode reused a prior child controller"
                    )
            try:
                live_refs.append(ref(controller))
            except TypeError as error:
                raise RuntimeError(
                    "isolated AI2-THOR controller must support weak references"
                ) from error
            self._isolated_controller_refs = live_refs
            return controller

        child = type(self)(
            [source.scene_id],
            width=self.width,
            height=self.height,
            seed=self.seed,
            controller_factory=isolated_controller_factory,
            allow_source_pose_drift=self.allow_source_pose_drift,
            procedural_scenes=(
                {source.scene_id: procedural} if procedural is not None else None
            ),
        )
        with child:
            event = child._activate_scene(source.scene_id)
            native_scene = child._scene_from_event(source.scene_id, event)
            child._current_scene = native_scene
            native_settlement = child.settle_scene_observed(
                native_scene,
                max_pass_steps=max_pass_steps,
            )
            final_event = child._current_event_for_scene(
                native_settlement.observed_scene
            )
            stable_scene = child._stable_observed_scene(
                source,
                native_settlement.observed_scene,
            )
            observation = child._observation_from_event(stable_scene, final_event)
            settlement = AI2ThorSceneSettlement(
                observed_scene=stable_scene,
                observation=observation,
                pass_steps=native_settlement.pass_steps,
            )
            child._current_scene = stable_scene
            yield AI2ThorIsolatedEpisode(
                adapter=child,
                baseline_settlement=settlement,
            )

    def _activate_scene(
        self,
        scene_id: str,
        *,
        force_reset: bool = False,
    ) -> Any:
        controller = self._require_active()
        if scene_id not in self.scene_names:
            raise KeyError(scene_id)
        if self._event is None and not force_reset:
            raise RuntimeError("no AI2-THOR event is available")
        if self._event is not None and self.scene_name == scene_id and not force_reset:
            self._validate_scene_source_or_poison(
                controller,
                scene_id,
                self._event,
            )
            return self._event
        try:
            event = self._reset(controller, scene_id)
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            f"reset {scene_id}",
            scene_id,
        )
        # Commit native-scene tracking only after reset success. Callers can
        # then fail closed without applying saved poses to the wrong scene.
        self.scene_name = scene_id
        return event

    def load_scene(self, scene_id: str) -> Scene:
        # Public load means a deterministic baseline reload, even when Unity
        # already has this scene active. Internal restoration uses the
        # non-forced activation path to avoid redundant resets.
        event = self._activate_scene(scene_id, force_reset=True)
        scene = self._scene_from_event(scene_id, event)
        self._current_scene = scene
        return scene

    @classmethod
    def _commanded_scene(
        cls,
        source: Scene,
        observed: Scene,
        subject_id: str,
        commanded_position: Vec3,
    ) -> Scene:
        """Reproduce the legacy canonical command while retaining fresh views."""
        observed_by_name = cls._objects_by_name(observed.objects)
        subject = source.object_by_id(subject_id)
        delta_x = commanded_position.x - subject.position.x
        delta_y = commanded_position.y - subject.position.y
        merged: list[SceneObject] = []
        for original in source.objects:
            current = observed_by_name[original.name]
            if original.object_id == subject_id:
                merged.append(
                    original.model_copy(
                        update={
                            "position": commanded_position,
                            "obb": original.obb.model_copy(
                                update={
                                    "center": Vec3(
                                        x=original.obb.center.x + delta_x,
                                        y=original.obb.center.y + delta_y,
                                        z=original.obb.center.z,
                                    )
                                }
                            ),
                            "views": current.views,
                        }
                    )
                )
            else:
                merged.append(original.model_copy(update={"views": current.views}))
        return source.model_copy(update={"objects": tuple(merged)})

    def _native_rotation_for(
        self,
        scene_id: str,
        obj: SceneObject,
    ) -> dict[str, float]:
        rotation = self._native_rotations.get((scene_id, obj.name))
        if rotation is None:
            raise ValueError(
                f"native rotation was not captured for object {obj.name!r}"
            )
        if (
            not _quaternions_close(
                ai2thor_rotation_to_world(Vec3(**rotation)),
                obj.rotation,
            )
            and not self.allow_source_pose_drift
        ):
            raise ValueError(
                f"scene rotation for object {obj.name!r} differs from captured native pose"
            )
        return dict(rotation)

    def _object_pose(
        self,
        scene_id: str,
        obj: SceneObject,
        position: Vec3 | None = None,
    ) -> dict[str, Any]:
        requested_position = position or obj.position
        return {
            "objectName": obj.name,
            "position": {
                "x": float(requested_position.x),
                "y": float(requested_position.z),
                "z": float(requested_position.y),
            },
            "rotation": self._native_rotation_for(scene_id, obj),
        }

    def _restore_scene_state(self, scene: Scene) -> Any:
        controller = self._require_active()
        if scene.scene_id not in self.scene_names:
            raise KeyError(scene.scene_id)
        self._activate_scene(scene.scene_id)
        self._objects_by_name(scene.objects)
        camera = scene.camera_by_id("main")
        state = self._camera_states.get(self._camera_key(scene.scene_id, camera))
        if state is None:
            raise ValueError("camera pose was not captured by this adapter")
        expected_positions = {obj.name: obj.position for obj in scene.objects}
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "SetObjectPoses",
                action="SetObjectPoses",
                objectPoses=[
                    self._object_pose(scene.scene_id, obj)
                    for obj in scene.objects
                    if obj.movable
                ],
                placeStationary=True,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "SetObjectPoses",
            scene.scene_id,
        )
        try:
            event = self._step(
                controller,
                "TeleportFull",
                action="TeleportFull",
                position=dict(state["position"]),
                rotation=dict(state["rotation"]),
                horizon=state["horizon"],
                standing=state["standing"],
                forceAction=True,
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "TeleportFull",
            scene.scene_id,
        )
        self._validate_returned_state(
            scene,
            event,
            expected_positions,
            expected_rotations,
        )
        self._current_scene = scene
        return event

    def _validate_returned_state(
        self,
        scene: Scene,
        event: Any,
        expected_positions: dict[str, Vec3],
        expected_rotations: dict[str, dict[str, float]],
        *,
        deferred_pose_name: str | None = None,
        total_position_residual_limits_by_name: Mapping[str, float] | None = None,
        rotation_residual_limits_by_name: Mapping[str, float] | None = None,
    ) -> None:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError(
                "pose application returned no object metadata"
            )
        raw_objects = _validated_native_object_metadata(raw_objects)
        raw_objects = _domain_object_metadata(raw_objects)
        names = [item["name"] for item in raw_objects]
        by_name = dict(zip(names, raw_objects, strict=True))
        expected_by_name = self._objects_by_name(scene.objects)
        if set(by_name) != set(expected_by_name):
            raise AI2ThorNativeReturnError(
                "stable object names changed during pose application"
            )
        if deferred_pose_name is not None and (
            deferred_pose_name not in expected_by_name
            or deferred_pose_name not in expected_positions
            or deferred_pose_name not in expected_rotations
        ):
            raise ValueError("deferred pose name is not an expected stable object")
        for name in expected_by_name:
            metadata = by_name[name]
            try:
                position = ai2thor_position_to_world(Vec3(**metadata["position"]))
                native_rotation = {
                    axis: float(metadata["rotation"][axis]) for axis in ("x", "y", "z")
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise AI2ThorNativeReturnError(
                    f"object {name!r} returned an invalid pose"
                ) from exc
            if not all(
                math.isfinite(value)
                for value in (
                    position.x,
                    position.y,
                    position.z,
                    *native_rotation.values(),
                )
            ):
                raise AI2ThorNativeReturnError(
                    f"object {name!r} returned an invalid pose"
                )
            expected_position = expected_positions[name]
            expected_rotation = expected_rotations[name]
            expected_coordinates = (
                expected_position.x,
                expected_position.y,
                expected_position.z,
            )
            observed_coordinates = (position.x, position.y, position.z)
            residual_limit = (
                None
                if total_position_residual_limits_by_name is None
                else total_position_residual_limits_by_name.get(name)
            )
            if residual_limit is None:
                position_matches = np.allclose(
                    observed_coordinates,
                    expected_coordinates,
                    atol=1e-5,
                    rtol=0.0,
                )
            else:
                rounding_allowance_m = 4.0 * math.ulp(residual_limit)
                position_matches = (
                    math.dist(
                        observed_coordinates,
                        expected_coordinates,
                    )
                    <= residual_limit + rounding_allowance_m
                )
            rotation_residual_limit = (
                _OBJECT_ROTATION_TOLERANCE_DEGREES
                if rotation_residual_limits_by_name is None
                else rotation_residual_limits_by_name.get(
                    name,
                    _OBJECT_ROTATION_TOLERANCE_DEGREES,
                )
            )
            if name == deferred_pose_name:
                continue
            if not position_matches or not all(
                self._angles_close(
                    native_rotation[axis],
                    expected_rotation[axis],
                    tolerance_degrees=rotation_residual_limit,
                )
                for axis in ("x", "y", "z")
            ):
                raise AI2ThorNativeReturnError(
                    f"object {name!r} pose changed during pose application"
                )
        observed_camera = self._camera(event.metadata, scene.scene_id)
        self._validate_camera_fixed(scene.camera_by_id("main"), observed_camera)

    @staticmethod
    def _native_scene_at_rest(event: Any) -> bool:
        value = event.metadata.get("isSceneAtRest")
        if type(value) is not bool:
            raise AI2ThorNativeReturnError(
                "AI2-THOR isSceneAtRest must be an exact boolean"
            )
        return value

    @staticmethod
    def _native_object_is_moving(event: Any, object_name: str) -> bool:
        raw_objects = event.metadata.get("objects")
        if not isinstance(raw_objects, list):
            raise AI2ThorNativeReturnError(
                "pose application returned no object metadata"
            )
        matches = [
            item
            for item in raw_objects
            if isinstance(item, dict) and item.get("name") == object_name
        ]
        if len(matches) != 1:
            raise AI2ThorNativeReturnError(
                f"expected one native object named {object_name!r}"
            )
        value = matches[0].get("isMoving")
        if type(value) is not bool:
            raise AI2ThorNativeReturnError("AI2-THOR isMoving must be an exact boolean")
        return value

    @staticmethod
    def _native_object_id_for_name(event: Any, object_name: str) -> str:
        raw_objects = event.metadata.get("objects")
        raw_objects_is_list = isinstance(raw_objects, list)
        if not raw_objects_is_list:
            raise RuntimeError("pose application returned no object metadata")
        matches = [
            item
            for item in raw_objects
            if isinstance(item, dict) and item.get("name") == object_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one native object named {object_name!r}")
        native_object_id = matches[0].get("objectId")
        if type(native_object_id) is not str or not native_object_id:
            raise RuntimeError(
                f"native object named {object_name!r} has no valid objectId"
            )
        return native_object_id

    def apply_object_xy_observed(
        self,
        scene: Scene,
        object_id: str,
        x: float,
        y: float,
    ) -> AI2ThorPoseApplication:
        """Apply one command and retain both canonical and native observations."""
        controller = self._require_active()
        target = scene.object_by_id(object_id)
        if not target.movable:
            raise ValueError(f"object {object_id!r} is not movable")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("object X/Y must be finite")
        self._objects_by_name(scene.objects)
        if self._current_scene != scene:
            current_scene = self._current_scene
            if current_scene is not None and self._is_analysis_overlay(
                current_scene, scene
            ):
                current_event = self._current_event_for_scene(current_scene)
            else:
                current_event = self._restore_scene_state(scene)
        else:
            current_event = self._current_event_for_scene(scene)
        expected_positions = {
            obj.name: (
                Vec3(x=x, y=y, z=obj.position.z)
                if obj.object_id == object_id
                else obj.position
            )
            for obj in scene.objects
        }
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        native_object_id = self._native_object_id_for_name(
            current_event,
            target.name,
        )
        commanded_position = expected_positions[target.name]
        teleport_vertical_guard_m = _TELEPORT_VERTICAL_GUARD_M
        # SetObjectPoses removes every movable object omitted from its payload.
        # TeleportObject is the native single-object edit and avoids re-emitting
        # unrelated poses, while the validation below still checks the full scene.
        try:
            event = self._step(
                controller,
                "TeleportObject",
                action="TeleportObject",
                objectId=native_object_id,
                position={
                    "x": commanded_position.x,
                    "y": commanded_position.z + teleport_vertical_guard_m,
                    "z": commanded_position.y,
                },
                rotation=dict(expected_rotations[target.name]),
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "TeleportObject",
            scene.scene_id,
        )
        self._validate_returned_state(
            scene,
            event,
            expected_positions,
            expected_rotations,
        )
        native_observed = self._scene_from_event(scene.scene_id, event)
        observed_scene = self._stable_observed_scene(scene, native_observed)
        commanded_scene = self._commanded_scene(
            scene,
            observed_scene,
            object_id,
            commanded_position,
        )
        observed_position = observed_scene.object_by_id(object_id).position
        position_residual_m = math.dist(
            (
                commanded_position.x,
                commanded_position.y,
                commanded_position.z,
            ),
            (
                observed_position.x,
                observed_position.y,
                observed_position.z,
            ),
        )
        is_scene_at_rest = self._native_scene_at_rest(event)
        subject_is_moving = self._native_object_is_moving(event, target.name)
        observation = self._observation_from_event(observed_scene, event)
        result = AI2ThorPoseApplication(
            commanded_scene=commanded_scene,
            observed_scene=observed_scene,
            commanded_position=commanded_position,
            observed_position=observed_position,
            position_residual_m=position_residual_m,
            observation=observation,
            is_scene_at_rest=is_scene_at_rest,
            subject_is_moving=subject_is_moving,
        )
        # The retained event contains the observed native pose, not the ideal
        # command.  Keep that identity exact; legacy callers still receive the
        # commanded scene from ``with_object_xy`` and restoration is explicit.
        self._current_scene = observed_scene
        return result

    def apply_receptacle_spawn_point_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        position: AI2ThorNativePosition,
    ) -> AI2ThorPoseApplication:
        """Audit one externally selected endpoint with native surface placement.

        This method is deliberately not a search API.  It accepts exactly one
        point from one source-bound map, executes one ``PlaceObjectAtPoint``,
        and fails closed if native collision or pose validation rejects it.
        """

        checked_map, subject, native_subject_object_id = self._receptacle_audit_source(
            scene, spawn_map
        )
        checked_position = _strict_native_position(position, "spawn audit position")
        if checked_position not in checked_map.positions:
            raise ValueError("spawn audit position is not in the source-bound map")
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            checked_position,
        )

    def apply_receptacle_endpoint_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        *,
        x: float,
        y: float,
        _defer_subject_pose_validation: bool = False,
    ) -> AI2ThorPoseApplication:
        """Audit one exact world-XY endpoint without snapping or searching.

        The source-bound map contributes only the single exact native support
        height required by this horizontal-support API.  The endpoint X/Y is
        supplied by the platform-neutral solver and is never replaced by a
        nearby returned coordinate.  Multi-height receptacles are outside this
        contract and fail before any native action.
        """

        checked_map, subject, native_subject_object_id = self._receptacle_audit_source(
            scene, spawn_map
        )
        endpoint_x = _strict_finite_float(x, "receptacle endpoint x")
        endpoint_y = _strict_finite_float(y, "receptacle endpoint y")
        native_heights = {item.y for item in checked_map.positions}
        if len(native_heights) != 1:
            raise ValueError(
                "exact receptacle endpoint audit requires one native support height"
            )
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            AI2ThorNativePosition(
                x=endpoint_x,
                y=next(iter(native_heights)),
                z=endpoint_y,
            ),
            deferred_pose_name=(subject.name if _defer_subject_pose_validation else None),
        )

    def apply_receptacle_endpoint_settled_observed(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
        *,
        x: float,
        y: float,
        max_pass_steps: int,
        max_subject_rotation_residual_degrees: float | None = None,
    ) -> AI2ThorPoseApplication:
        """Place once, then wait boundedly for the runtime-only endpoint to settle."""

        if type(max_pass_steps) is not int or max_pass_steps <= 0:
            raise ValueError("max_pass_steps must be an exact positive integer")
        if max_subject_rotation_residual_degrees is not None:
            max_subject_rotation_residual_degrees = _strict_finite_float(
                max_subject_rotation_residual_degrees,
                "subject rotation residual limit",
            )
            if max_subject_rotation_residual_degrees <= 0.0:
                raise ValueError("subject rotation residual limit must be positive")
        checked_map, subject, native_subject_object_id = self._receptacle_audit_source(
            scene, spawn_map
        )
        endpoint_x = _strict_finite_float(x, "receptacle endpoint x")
        endpoint_y = _strict_finite_float(y, "receptacle endpoint y")
        native_heights = {item.y for item in checked_map.positions}
        if len(native_heights) != 1:
            raise ValueError(
                "exact receptacle endpoint audit requires one native support height"
            )
        return self._apply_receptacle_native_position_observed(
            scene,
            subject,
            native_subject_object_id,
            AI2ThorNativePosition(
                x=endpoint_x,
                y=next(iter(native_heights)),
                z=endpoint_y,
            ),
            max_pass_steps=max_pass_steps,
            max_subject_rotation_residual_degrees=(
                max_subject_rotation_residual_degrees
            ),
        )

    def _receptacle_audit_source(
        self,
        scene: Scene,
        spawn_map: AI2ThorReceptacleSpawnMap,
    ) -> tuple[AI2ThorReceptacleSpawnMap, SceneObject, str]:
        current_event = self._current_event_for_scene(scene)
        checked_map = _strict_receptacle_spawn_map(spawn_map)
        subject = scene.object_by_id(checked_map.subject_object_id)
        if subject.support_object_id != checked_map.support_object_id:
            raise ValueError("spawn map support does not match the current scene")
        runtime_identity = self.runtime_identity()
        scene_sha256 = _receptacle_scene_sha256(
            scene,
            checked_map.surface_patches,
        )
        native_subject_object_id = self._native_object_id_for_name(
            current_event,
            subject.name,
        )
        support = scene.object_by_id(checked_map.support_object_id)
        native_support_object_id = self._native_object_id_for_name(
            current_event,
            support.name,
        )
        if (
            checked_map.scene_id != scene.scene_id
            or checked_map.scene_sha256 != scene_sha256
            or checked_map.runtime_identity != runtime_identity
            or checked_map.native_subject_object_id != native_subject_object_id
            or checked_map.native_support_object_id != native_support_object_id
        ):
            raise ValueError("spawn map does not close the exact current source")
        return checked_map, subject, native_subject_object_id

    def _apply_receptacle_native_position_observed(
        self,
        scene: Scene,
        subject: SceneObject,
        native_subject_object_id: str,
        checked_position: AI2ThorNativePosition,
        *,
        max_pass_steps: int | None = None,
        max_subject_rotation_residual_degrees: float | None = None,
        deferred_pose_name: str | None = None,
    ) -> AI2ThorPoseApplication:
        controller = self._require_active()
        commanded_position = Vec3(
            x=checked_position.x,
            y=checked_position.z,
            z=subject.position.z,
        )
        expected_positions = {
            obj.name: (
                commanded_position
                if obj.object_id == subject.object_id
                else obj.position
            )
            for obj in scene.objects
        }
        expected_rotations = {
            obj.name: self._native_rotation_for(scene.scene_id, obj)
            for obj in scene.objects
        }
        try:
            event = self._step(
                controller,
                "PlaceObjectAtPoint",
                action="PlaceObjectAtPoint",
                objectId=native_subject_object_id,
                position={
                    "x": checked_position.x,
                    "y": checked_position.y,
                    "z": checked_position.z,
                },
                rotation=dict(expected_rotations[subject.name]),
            )
        except BaseException:
            self._poison_scene_state()
            raise
        self._event = event
        self._current_scene = None
        event = self._checked_scene_event(
            controller,
            event,
            "PlaceObjectAtPoint",
            scene.scene_id,
        )
        if max_pass_steps is None:
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
                deferred_pose_name=deferred_pose_name,
            )
        else:
            immediate_native = self._scene_from_event(scene.scene_id, event)
            immediate_observed = self._stable_observed_scene(scene, immediate_native)
            self._current_scene = immediate_observed
            settlement = self.settle_scene_observed(
                immediate_observed,
                max_pass_steps=max_pass_steps,
            )
            event = self._current_event_for_scene(settlement.observed_scene)
            self._validate_returned_state(
                scene,
                event,
                expected_positions,
                expected_rotations,
                total_position_residual_limits_by_name={
                    subject.name: _RUNTIME_RECEPTACLE_POSITION_RESIDUAL_M
                },
                rotation_residual_limits_by_name=(
                    None
                    if max_subject_rotation_residual_degrees is None
                    else {
                        subject.name: max_subject_rotation_residual_degrees,
                    }
                ),
            )
        native_observed = self._scene_from_event(scene.scene_id, event)
        observed_scene = self._stable_observed_scene(scene, native_observed)
        commanded_scene = self._commanded_scene(
            scene,
            observed_scene,
            subject.object_id,
            commanded_position,
        )
        observed_position = observed_scene.object_by_id(subject.object_id).position
        position_residual_m = math.dist(
            (
                commanded_position.x,
                commanded_position.y,
                commanded_position.z,
            ),
            (
                observed_position.x,
                observed_position.y,
                observed_position.z,
            ),
        )
        result = AI2ThorPoseApplication(
            commanded_scene=commanded_scene,
            observed_scene=observed_scene,
            commanded_position=commanded_position,
            observed_position=observed_position,
            position_residual_m=position_residual_m,
            observation=self._observation_from_event(observed_scene, event),
            is_scene_at_rest=self._native_scene_at_rest(event),
            subject_is_moving=self._native_object_is_moving(event, subject.name),
        )
        self._current_scene = observed_scene
        return result

    def with_object_xy(
        self,
        scene: Scene,
        object_id: str,
        x: float,
        y: float,
    ) -> Scene:
        return self.apply_object_xy_observed(scene, object_id, x, y).commanded_scene
