"""Native camera control for the staged AI2-THOR adapter."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np

from spatialcf.adapters.ai2thor.conversion import (
    _ANGLE_TOLERANCE_DEGREES,
    _quaternions_close,
)
from spatialcf.adapters.ai2thor.models import (
    AI2ThorAgentPose,
    AI2ThorCameraApplication,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorSettledCameraApplication,
    AI2ThorSettlementTimeout,
    adapter_camera_application_from_native,
    native_pose_from_adapter,
)
from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterOperationError,
    AdapterPose,
    AdapterSettledCameraApplication,
    AdapterSettlementTimeout,
    CameraObservationHandle,
    SourceCaptureFacts,
)
from spatialcf.domain.scene import Camera, Scene

_CAMERA_POSITION_TOLERANCE_M = 1e-5
_OBJECT_GEOMETRY_TOLERANCE_M = 1e-5
_NATIVE_NAVIGATION_GRID_SIZE_M = 0.05

CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5 = 0.2
CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6 = 0.21
CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7 = 0.25
CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8 = 0.25 + math.sqrt(2.0) * 0.5e-6


def _clearance_rotation_matrix(
    rotation,
) -> tuple[tuple[float, float, float], ...]:
    values = (rotation.x, rotation.y, rotation.z, rotation.w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera clearance OBB rotation must be finite")
    maximum = max(abs(value) for value in values)
    if maximum == 0.0:
        raise ValueError("camera clearance OBB rotation must be nonzero")
    scaled = tuple(value / maximum for value in values)
    norm = math.sqrt(sum(value * value for value in scaled))
    x, y, z, w = (value / norm for value in scaled)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _clearance_projected_corners(obb) -> tuple[tuple[float, float], ...]:
    extents = (obb.extent.x, obb.extent.y, obb.extent.z)
    centers = (obb.center.x, obb.center.y, obb.center.z)
    if not all(math.isfinite(value) and value > 0.0 for value in extents):
        raise ValueError("camera clearance OBB extents must be finite and positive")
    if not all(math.isfinite(value) for value in centers):
        raise ValueError("camera clearance OBB center must be finite")
    rotation = _clearance_rotation_matrix(obb.rotation)
    points = set()
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                local = (
                    x_sign * extents[0] / 2.0,
                    y_sign * extents[1] / 2.0,
                    z_sign * extents[2] / 2.0,
                )
                world = tuple(
                    centers[axis]
                    + sum(rotation[axis][inner] * local[inner] for inner in range(3))
                    for axis in range(3)
                )
                points.add((world[0], world[1]))
    return tuple(sorted(points))


def _clearance_cross(
    origin: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
        right[0] - origin[0]
    )


def _clearance_convex_hull(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    unique = tuple(sorted(set(points)))
    if len(unique) < 3:
        raise ValueError("camera clearance OBB projection must have positive area")
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _clearance_cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _clearance_cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("camera clearance OBB projection must have positive area")
    return hull


def _clearance_point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta = (end[0] - start[0], end[1] - start[1])
    length_squared = delta[0] * delta[0] + delta[1] * delta[1]
    if length_squared == 0.0:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * delta[0] + (point[1] - start[1]) * delta[1])
            / length_squared,
        ),
    )
    nearest = (
        start[0] + fraction * delta[0],
        start[1] + fraction * delta[1],
    )
    return math.dist(point, nearest)


def _clearance_point_polygon_distance(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
) -> float:
    crosses = tuple(
        _clearance_cross(polygon[index], polygon[(index + 1) % len(polygon)], point)
        for index in range(len(polygon))
    )
    if all(value >= 0.0 for value in crosses) or all(value <= 0.0 for value in crosses):
        return 0.0
    return min(
        _clearance_point_segment_distance(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def _filter_competition_native_camera_positions(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
    *,
    clearance_radius_m: float,
) -> tuple[AI2ThorNativePosition, ...]:
    if type(scene) is not Scene:
        raise TypeError("camera clearance scene must be an exact Scene")
    checked_scene = Scene.model_validate(scene.model_dump(mode="python"), strict=True)
    if type(positions) is not tuple or any(
        type(position) is not AI2ThorNativePosition for position in positions
    ):
        raise TypeError("camera clearance requires an exact position tuple")
    if not positions:
        raise TypeError("camera clearance requires a non-empty exact position tuple")
    if len(set(positions)) != len(positions):
        raise ValueError("camera clearance positions must be unique")
    footprints = tuple(
        _clearance_convex_hull(_clearance_projected_corners(item.obb))
        for item in checked_scene.objects
        if item.movable
    )
    accepted = tuple(
        position
        for position in positions
        if all(
            _clearance_point_polygon_distance((position.x, position.z), footprint)
            > clearance_radius_m
            for footprint in footprints
        )
    )
    return tuple(sorted(accepted, key=lambda item: (item.x, item.z, item.y)))


def filter_competition_native_camera_positions_v2_9_5(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5,
    )


def filter_competition_native_camera_positions_v2_9_6(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6,
    )


def filter_competition_native_camera_positions_v2_9_7(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7,
    )


def filter_competition_native_camera_positions_v2_9_8(
    scene: Scene,
    positions: tuple[AI2ThorNativePosition, ...],
) -> tuple[AI2ThorNativePosition, ...]:
    return _filter_competition_native_camera_positions(
        scene,
        positions,
        clearance_radius_m=CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8,
    )


def _camera_position_residual_m(
    requested: AI2ThorNativePosition,
    observed: AI2ThorNativePosition,
) -> float:
    """Return the total native-coordinate residual for one camera pose."""
    return math.dist(
        (requested.x, requested.y, requested.z),
        (observed.x, observed.y, observed.z),
    )


def _camera_position_residual_within_tolerance(
    requested: AI2ThorNativePosition,
    observed: AI2ThorNativePosition,
) -> tuple[float, bool]:
    """Apply the shared total-position camera contract with ULP allowance."""
    residual_m = _camera_position_residual_m(requested, observed)
    rounding_allowance_m = 4.0 * max(
        math.ulp(value)
        for value in (
            requested.x,
            requested.y,
            requested.z,
            observed.x,
            observed.y,
            observed.z,
        )
    )
    return (
        residual_m,
        residual_m <= _CAMERA_POSITION_TOLERANCE_M + rounding_allowance_m,
    )


class AI2ThorCameraMixin:
    def pause_camera_observations(
        self,
        facts: SourceCaptureFacts,
        *,
        settle_after_resume: bool,
    ) -> CameraObservationHandle:
        """Pause one captured source until the matching one-shot resume."""

        if (
            type(facts) is not SourceCaptureFacts
            or type(settle_after_resume) is not bool
        ):
            raise AdapterOperationError("camera pause arguments must be exact")
        token = (
            f"{facts.binding.token}:camera-pause:{self._protocol_camera_token_sequence}"
        )
        self._protocol_camera_token_sequence += 1
        if (
            token in self._protocol_camera_handles
            or token in self._protocol_camera_resumed
        ):
            raise AdapterOperationError("camera pause token collision")
        manager = (
            self.paused_camera_observations_for_settlement(facts.scene)
            if settle_after_resume
            else self.paused_camera_observations(facts.scene)
        )
        try:
            paused_scene = manager.__enter__()
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error
        handle = CameraObservationHandle(
            source=facts.source,
            binding=facts.binding,
            scene=paused_scene,
            token=token,
            settle_after_resume=settle_after_resume,
        )
        self._protocol_camera_handles[token] = (handle, manager)
        return handle

    def resume_camera_observations(self, handle: CameraObservationHandle) -> None:
        """Resume a paused source exactly once, even on a workflow exception."""

        if type(handle) is not CameraObservationHandle:
            raise AdapterOperationError("camera resume handle must be exact")
        if handle.token in self._protocol_camera_resumed:
            raise AdapterOperationError("camera observations already resumed")
        try:
            issued_handle, manager = self._protocol_camera_handles[handle.token]
        except KeyError as error:
            raise AdapterOperationError("camera pause handle is missing") from error
        if handle is not issued_handle or handle != issued_handle:
            raise AdapterOperationError("camera pause handle is not the issued handle")
        del self._protocol_camera_handles[handle.token]
        try:
            manager.__exit__(None, None, None)
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error
        finally:
            self._protocol_camera_resumed.add(handle.token)

    def apply_camera_pose(
        self,
        facts: SourceCaptureFacts,
        pose: AdapterPose,
        *,
        handle: CameraObservationHandle | None,
        source_scene: Scene,
        reset_from_source: bool,
        max_settlement_steps: int,
    ) -> AdapterCameraApplication:
        """Apply exactly one requested pose without choosing or replaying it."""

        if (
            type(facts) is not SourceCaptureFacts
            or type(pose) is not AdapterPose
            or type(source_scene) is not Scene
            or type(reset_from_source) is not bool
            or type(max_settlement_steps) is not int
            or max_settlement_steps <= 0
        ):
            raise AdapterOperationError("camera application arguments must be exact")
        if handle is not None and (
            type(handle) is not CameraObservationHandle
            or handle.binding != facts.binding
            or handle.token not in self._protocol_camera_handles
        ):
            raise AdapterOperationError("camera application pause handle is invalid")
        try:
            native_pose = native_pose_from_adapter(pose)
            application = (
                self.apply_camera_pose_from_frozen_source_observed(
                    source_scene,
                    native_pose,
                    max_pass_steps=max_settlement_steps,
                )
                if reset_from_source
                else self.apply_camera_pose_observed(source_scene, native_pose)
            )
            return adapter_camera_application_from_native(
                application,
                source=facts.source,
            )
        except AI2ThorSettlementTimeout as error:
            raise AdapterSettlementTimeout(str(error)) from error
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error

    def settle_camera_pose(
        self,
        facts: SourceCaptureFacts,
        pose: AdapterPose,
        *,
        source_scene: Scene,
        max_settlement_steps: int,
    ) -> AdapterSettledCameraApplication:
        """Settle one already-selected post-resume pose exactly once."""

        if (
            type(facts) is not SourceCaptureFacts
            or type(pose) is not AdapterPose
            or type(source_scene) is not Scene
            or type(max_settlement_steps) is not int
            or max_settlement_steps <= 0
        ):
            raise AdapterOperationError("camera settlement arguments must be exact")
        try:
            settled = self.settle_current_camera_pose_observed(
                source_scene,
                native_pose_from_adapter(pose),
                max_pass_steps=max_settlement_steps,
            )
            return AdapterSettledCameraApplication(
                application=adapter_camera_application_from_native(
                    settled.application,
                    source=facts.source,
                ),
                settlement_pass_steps=settled.settlement_pass_steps,
            )
        except AI2ThorSettlementTimeout as error:
            raise AdapterSettlementTimeout(str(error)) from error
        except (AI2ThorNativeReturnError, RuntimeError, ValueError, KeyError) as error:
            raise AdapterOperationError(str(error)) from error

    @classmethod
    def _validate_camera_object_invariants(
        cls,
        source: Scene,
        observed: Scene,
    ) -> None:
        cls._validate_camera_object_identity_invariants(source, observed)
        source_by_name = cls._objects_by_name(source.objects)
        observed_by_name = cls._objects_by_name(observed.objects)

        for name, original in source_by_name.items():
            current = observed_by_name[name]
            if (
                not np.allclose(
                    (
                        current.position.x,
                        current.position.y,
                        current.position.z,
                        current.obb.center.x,
                        current.obb.center.y,
                        current.obb.center.z,
                        current.obb.extent.x,
                        current.obb.extent.y,
                        current.obb.extent.z,
                    ),
                    (
                        original.position.x,
                        original.position.y,
                        original.position.z,
                        original.obb.center.x,
                        original.obb.center.y,
                        original.obb.center.z,
                        original.obb.extent.x,
                        original.obb.extent.y,
                        original.obb.extent.z,
                    ),
                    atol=_OBJECT_GEOMETRY_TOLERANCE_M,
                    rtol=0.0,
                )
                or not _quaternions_close(
                    current.rotation,
                    original.rotation,
                )
                or not _quaternions_close(
                    current.obb.rotation,
                    original.obb.rotation,
                )
            ):
                raise RuntimeError(
                    f"object {name!r} geometry changed during camera application"
                )

    @classmethod
    def _validate_camera_object_identity_invariants(
        cls,
        source: Scene,
        observed: Scene,
    ) -> None:
        source_by_name = cls._objects_by_name(source.objects)
        observed_by_name = cls._objects_by_name(observed.objects)
        if set(source_by_name) != set(observed_by_name):
            raise RuntimeError("stable object names changed during camera application")

        for name, original in source_by_name.items():
            current = observed_by_name[name]
            if (
                current.object_id != original.object_id
                or current.name != original.name
                or current.category != original.category
                or current.movable is not original.movable
                or current.request_eligible is not original.request_eligible
                or current.support_object_id != original.support_object_id
            ):
                raise RuntimeError(
                    f"object {name!r} identity, category, mobility, or support changed "
                    "during camera application"
                )

    def apply_camera_pose_observed(
        self,
        scene: Scene,
        pose: AI2ThorAgentPose,
    ) -> AI2ThorCameraApplication:
        """Apply one unforced TeleportFull and bind its same-event observation."""
        controller = self._require_active()
        current_event = self._current_event_for_scene(scene)
        expected_native_scene_name = self._native_scene_name(current_event)
        if type(pose) is not AI2ThorAgentPose:
            raise ValueError("camera pose must be an exact AI2ThorAgentPose")
        snapped_x = (
            round(pose.position.x / _NATIVE_NAVIGATION_GRID_SIZE_M)
            * _NATIVE_NAVIGATION_GRID_SIZE_M
        )
        snapped_z = (
            round(pose.position.z / _NATIVE_NAVIGATION_GRID_SIZE_M)
            * _NATIVE_NAVIGATION_GRID_SIZE_M
        )
        commanded_position = AI2ThorNativePosition(
            x=snapped_x,
            y=pose.position.y,
            z=snapped_z,
        )
        _, is_on_native_grid = _camera_position_residual_within_tolerance(
            pose.position,
            commanded_position,
        )
        if not is_on_native_grid:
            raise ValueError("camera pose is not on the native grid")
        if commanded_position.x == 0.0:
            commanded_position = AI2ThorNativePosition(
                x=0.0,
                y=commanded_position.y,
                z=commanded_position.z,
            )
        if commanded_position.z == 0.0:
            commanded_position = AI2ThorNativePosition(
                x=commanded_position.x,
                y=commanded_position.y,
                z=0.0,
            )

        previous_camera_states = deepcopy(self._camera_states)
        previous_native_rotations = deepcopy(self._native_rotations)
        try:
            event = self._step(
                controller,
                "TeleportFull",
                action="TeleportFull",
                position={
                    "x": commanded_position.x,
                    "y": commanded_position.y,
                    "z": commanded_position.z,
                },
                rotation={"x": 0.0, "y": pose.yaw_degrees, "z": 0.0},
                horizon=pose.horizon_degrees,
                standing=pose.standing,
            )
            # Unity may have changed even when validation below fails. Retain
            # the returned event but invalidate the canonical current scene
            # until every invariant and same-event artifact has been checked.
            self._event = event
            self._current_scene = None
            event = self._checked_scene_event(
                controller,
                event,
                "TeleportFull",
                scene.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )

            observed_pose = self._native_agent_pose(event)
            observed_camera_position = self._native_position(
                event.metadata.get("cameraPosition"),
                "camera position",
            )
            position_residual_m, position_within_tolerance = (
                _camera_position_residual_within_tolerance(
                    pose.position,
                    observed_pose.position,
                )
            )
            yaw_residual_degrees = self._angle_residual_degrees(
                observed_pose.yaw_degrees,
                pose.yaw_degrees,
            )
            horizon_residual_degrees = abs(
                observed_pose.horizon_degrees - pose.horizon_degrees
            )
            if not position_within_tolerance:
                raise RuntimeError("camera position drift exceeds tolerance")
            if yaw_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
                raise RuntimeError("camera yaw drift exceeds tolerance")
            if horizon_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
                raise RuntimeError("camera horizon drift exceeds tolerance")
            if observed_pose.standing is not pose.standing:
                raise RuntimeError("camera standing state differs from request")

            native_observed = self._scene_from_event(scene.scene_id, event)
            observed_scene = self._canonical_camera_observed_scene(
                scene,
                native_observed,
            )
            observation = self._observation_from_event(observed_scene, event)
            result = AI2ThorCameraApplication(
                requested_pose=pose,
                observed_pose=observed_pose,
                observed_camera_position=observed_camera_position,
                observed_scene=observed_scene,
                observation=observation,
                position_residual_m=position_residual_m,
                yaw_residual_degrees=yaw_residual_degrees,
                horizon_residual_degrees=horizon_residual_degrees,
            )
            self._current_scene = observed_scene
            return result
        except BaseException:
            self._camera_states = previous_camera_states
            self._native_rotations = previous_native_rotations
            self._poison_scene_state()
            raise

    @contextmanager
    def paused_camera_observations(self, source: Scene) -> Iterator[Scene]:
        """Freeze native physics while yielding camera-only source observations."""

        with self._paused_camera_observations(
            source,
            retain_unpaused_scene=False,
        ) as paused_source:
            yield paused_source

    @contextmanager
    def paused_camera_observations_for_settlement(
        self,
        source: Scene,
    ) -> Iterator[Scene]:
        """Freeze camera ranking and retain the native scene returned by unpause."""

        with self._paused_camera_observations(
            source,
            retain_unpaused_scene=True,
        ) as paused_source:
            yield paused_source

    @contextmanager
    def _paused_camera_observations(
        self,
        source: Scene,
        *,
        retain_unpaused_scene: bool,
    ) -> Iterator[Scene]:
        if type(retain_unpaused_scene) is not bool:
            raise TypeError("retain-unpaused-scene flag must be an exact boolean")
        controller = self._require_active()
        current_event = self._current_event_for_scene(source)
        expected_native_scene_name = self._native_scene_name(current_event)
        try:
            paused_event = self._step(
                controller,
                "PausePhysicsAutoSim",
                action="PausePhysicsAutoSim",
            )
            self._event = paused_event
            self._current_scene = None
            paused_event = self._checked_scene_event(
                controller,
                paused_event,
                "PausePhysicsAutoSim",
                source.scene_id,
            )
            self._validate_native_scene_name_or_poison(
                paused_event,
                expected_native_scene_name,
            )
            native_paused = self._scene_from_event(source.scene_id, paused_event)
            paused_source = self._canonical_camera_observed_scene(
                source,
                native_paused,
            )
            self._current_scene = paused_source
        except BaseException:
            self._poison_scene_state()
            raise

        try:
            yield paused_source
        except BaseException as error:
            try:
                self._unpause_camera_observations(
                    controller,
                    source.scene_id,
                    expected_native_scene_name,
                    retained_source=source if retain_unpaused_scene else None,
                )
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"AI2-THOR physics unpause also failed: {cleanup_error}")
            raise
        else:
            self._unpause_camera_observations(
                controller,
                source.scene_id,
                expected_native_scene_name,
                retained_source=source if retain_unpaused_scene else None,
            )

    def _unpause_camera_observations(
        self,
        controller: Any,
        scene_id: str,
        expected_native_scene_name: str,
        *,
        retained_source: Scene | None = None,
    ) -> None:
        try:
            event = self._step(
                controller,
                "UnpausePhysicsAutoSim",
                action="UnpausePhysicsAutoSim",
            )
            self._event = event
            self._current_scene = None
            event = self._checked_scene_event(
                controller,
                event,
                "UnpausePhysicsAutoSim",
                scene_id,
            )
            self._validate_native_scene_name_or_poison(
                event,
                expected_native_scene_name,
            )
            if retained_source is not None:
                native_unpaused = self._scene_from_event(scene_id, event)
                stable_unpaused = self._stable_observed_scene(
                    retained_source,
                    native_unpaused,
                )
                self._validate_camera_object_identity_invariants(
                    retained_source,
                    stable_unpaused,
                )
                self._current_scene = stable_unpaused
            self._event = event
        except BaseException:
            self._poison_scene_state()
            raise

    def settle_current_camera_pose_observed(
        self,
        source: Scene,
        pose: AI2ThorAgentPose,
        *,
        max_pass_steps: int,
    ) -> AI2ThorSettledCameraApplication:
        """Settle the post-unpause scene and bind its current camera event."""

        if type(source) is not Scene:
            raise TypeError("camera settlement source must be an exact Scene")
        if type(pose) is not AI2ThorAgentPose:
            raise TypeError("camera settlement pose must be exact")
        current_scene = self._current_scene
        if current_scene is None or current_scene.scene_id != source.scene_id:
            raise RuntimeError("adapter has no current post-unpause camera scene")
        settlement = self.settle_scene_observed(
            current_scene,
            max_pass_steps=max_pass_steps,
        )
        event = self._current_event_for_scene(settlement.observed_scene)
        observed_pose = self._native_agent_pose(event)
        observed_camera_position = self._native_position(
            event.metadata.get("cameraPosition"),
            "camera position",
        )
        position_residual_m, position_within_tolerance = (
            _camera_position_residual_within_tolerance(
                pose.position,
                observed_pose.position,
            )
        )
        yaw_residual_degrees = self._angle_residual_degrees(
            observed_pose.yaw_degrees,
            pose.yaw_degrees,
        )
        horizon_residual_degrees = abs(
            observed_pose.horizon_degrees - pose.horizon_degrees
        )
        if not position_within_tolerance:
            raise RuntimeError("settled camera position drift exceeds tolerance")
        if yaw_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
            raise RuntimeError("settled camera yaw drift exceeds tolerance")
        if horizon_residual_degrees > _ANGLE_TOLERANCE_DEGREES:
            raise RuntimeError("settled camera horizon drift exceeds tolerance")
        if observed_pose.standing is not pose.standing:
            raise RuntimeError("settled camera standing state differs from request")
        return AI2ThorSettledCameraApplication(
            application=AI2ThorCameraApplication(
                requested_pose=pose,
                observed_pose=observed_pose,
                observed_camera_position=observed_camera_position,
                observed_scene=settlement.observed_scene,
                observation=settlement.observation,
                position_residual_m=position_residual_m,
                yaw_residual_degrees=yaw_residual_degrees,
                horizon_residual_degrees=horizon_residual_degrees,
            ),
            settlement_pass_steps=settlement.pass_steps,
        )

    def apply_camera_pose_from_frozen_source_observed(
        self,
        source: Scene,
        pose: AI2ThorAgentPose,
        *,
        max_pass_steps: int,
    ) -> AI2ThorCameraApplication:
        """Reset and settle the same frozen source before one camera pose."""

        if type(source) is not Scene:
            raise TypeError("frozen camera source must be an exact Scene")
        loaded = self.load_scene(source.scene_id)
        settlement = self.settle_scene_observed(
            loaded,
            max_pass_steps=max_pass_steps,
        )
        stable_source = self._canonical_camera_observed_scene(
            source,
            settlement.observed_scene,
        )
        self._current_scene = stable_source
        return self.apply_camera_pose_observed(stable_source, pose)

    @staticmethod
    def _camera_key(
        scene_id: str,
        camera: Camera,
    ) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        return scene_id, camera.intrinsics, camera.world_to_camera

    @staticmethod
    def _validate_camera_fixed(expected: Camera, observed: Camera) -> None:
        if expected.width != observed.width or expected.height != observed.height:
            raise AI2ThorNativeReturnError(
                "camera dimensions changed during pose application"
            )
        if not np.allclose(
            expected.intrinsics, observed.intrinsics, atol=1e-8, rtol=0.0
        ) or not np.allclose(
            expected.world_to_camera, observed.world_to_camera, atol=1e-5, rtol=0.0
        ):
            raise AI2ThorNativeReturnError(
                "camera pose or intrinsics changed during pose application"
            )
