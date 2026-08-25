"""Deterministic Canonical-only implementation of :class:`EnvironmentAdapter`."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from PIL import Image, ImageDraw

from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterFloorEnvelope,
    AdapterObservation,
    AdapterOperationError,
    AdapterPose,
    AdapterPosition,
    AdapterRuntimeIdentity,
    AdapterSettledCameraApplication,
    AdapterSpawnMap,
    AdapterSupportFact,
    AppliedCertifiedEdit,
    CameraObservationHandle,
    CapturedSource,
    CapturedSupport,
    CaptureRequest,
    CertifiedEditApplication,
    RenderedAssets,
    SettledReadback,
    SourceCaptureFacts,
    SourceCaptureOptions,
    capture_source_with_scene_adapter,
    capture_support_with_environment_adapter,
    move_object_xy,
)
from spatialcf.domain.scene import OBB, Quaternion, Scene, Vec2, Vec3


def _validate_stem(stem: str) -> None:
    if (
        type(stem) is not str
        or not stem
        or stem.strip() != stem
        or stem in {".", ".."}
        or Path(stem).name != stem
        or "/" in stem
        or "\\" in stem
    ):
        raise ValueError("artifact stem must be a safe filename component")


class InMemoryEnvironmentAdapter:
    """A no-runtime adapter that retains exactly one Canonical scene."""

    def __init__(self, scene: Scene) -> None:
        if type(scene) is not Scene:
            raise TypeError("in-memory adapter requires an exact Canonical Scene")
        self._scene = scene
        self._paused_camera_handles: dict[str, CameraObservationHandle] = {}
        self._resumed_tokens: set[str] = set()
        self._camera_pause_token_sequence = 0
        self._execution_actions: list[str] = []
        self._pending_protocol_applied: AppliedCertifiedEdit | None = None

    @classmethod
    def from_scene(cls, scene: Scene) -> Self:
        return cls(scene)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def list_scene_ids(self) -> list[str]:
        return [self._scene.scene_id]

    def load_scene(self, scene_id: str) -> Scene:
        if scene_id != self._scene.scene_id:
            raise KeyError(scene_id)
        self._execution_actions.append("load_scene")
        return self._scene

    @property
    def execution_actions(self) -> tuple[str, ...]:
        return tuple(self._execution_actions)

    def with_object_xy(self, scene: Scene, object_id: str, x: float, y: float) -> Scene:
        return move_object_xy(scene, object_id, x, y)

    def capture_source(self, request: CaptureRequest) -> CapturedSource:
        return capture_source_with_scene_adapter(self, request)

    def capture_support(self, source: CapturedSource) -> CapturedSupport:
        return capture_support_with_environment_adapter(self, source)

    def apply_certified_edit(
        self,
        application: CertifiedEditApplication,
    ) -> AppliedCertifiedEdit:
        if type(application) is not CertifiedEditApplication:
            raise AdapterOperationError("certified application must be exact")
        if application.source.scene != self._scene:
            raise AdapterOperationError("certified application source is stale")
        if self._pending_protocol_applied is not None:
            raise AdapterOperationError("pending certified application is not settled")
        subject = application.source.scene.object_by_id(application.edit.subject_id)
        observed_scene = move_object_xy(
            application.source.scene,
            application.edit.subject_id,
            subject.position.x + application.edit.translation_xy_m.x,
            subject.position.y + application.edit.translation_xy_m.y,
        )
        observed = observed_scene.object_by_id(application.edit.subject_id).position
        position = AdapterPosition(x=observed.x, y=observed.y, z=observed.z)
        applied = AppliedCertifiedEdit(
            application=application,
            edit=application.edit,
            commanded_scene=observed_scene,
            observed_scene=observed_scene,
            commanded_position=position,
            observed_position=position,
            position_residual_m=0.0,
            observation=self._observation(observed_scene, settled=False),
            is_scene_at_rest=False,
            subject_is_moving=True,
            settlement_pass_steps=0,
            binding=application.source.binding,
        )
        self._pending_protocol_applied = applied
        self._execution_actions.append(
            f"apply_certified_edit:{application.edit.subject_id}"
        )
        return applied

    def settle_readback(self, applied: AppliedCertifiedEdit) -> SettledReadback:
        if self._pending_protocol_applied is not applied:
            raise AdapterOperationError(
                "pending certified application is stale or consumed"
            )
        settled = SettledReadback(
            applied=applied,
            commanded_scene=applied.commanded_scene,
            observed_scene=applied.observed_scene,
            commanded_position=applied.commanded_position,
            observed_position=applied.observed_position,
            position_residual_m=applied.position_residual_m,
            observation=self._observation(applied.observed_scene, settled=True),
            is_scene_at_rest=True,
            subject_is_moving=False,
            settlement_pass_steps=0,
            binding=applied.binding,
        )
        self._pending_protocol_applied = None
        self._execution_actions.append(f"settle_readback:{applied.edit.subject_id}")
        return settled

    def _runtime_identity(self, source: CapturedSource) -> AdapterRuntimeIdentity:
        """Bind dimensions to the caller-selected capture camera exactly."""
        camera = source.scene.camera_by_id(source.request.camera_id)
        return AdapterRuntimeIdentity(
            ai2thor_version="in-memory",
            unity_commit_id="in-memory",
            native_scene_name=source.scene.scene_id,
            width=camera.width,
            height=camera.height,
            seed=source.scene.generation_seed,
        )

    @staticmethod
    def _observation(scene: Scene, *, settled: bool) -> AdapterObservation:
        source = scene.scene_id.encode("utf-8")
        return AdapterObservation.create(
            scene=scene,
            rgb_png=b"in-memory-rgb:" + source,
            depth_npy=b"in-memory-depth:" + source,
            instance_png=b"in-memory-instance:" + source,
            pointcloud_ply=b"in-memory-ply:" + source,
            instance_pixel_counts=(),
            is_settled=settled,
        )

    @staticmethod
    def _support_facts(scene: Scene) -> tuple[AdapterSupportFact, ...]:
        return tuple(
            AdapterSupportFact(
                scene_id=scene.scene_id,
                object_id=item.object_id,
                object_name=item.name,
                native_object_id=f"in-memory:{item.object_id}",
                raw_parent_object_ids=(
                    () if item.support_object_id is None else (item.support_object_id,)
                ),
                structural_parent_object_ids=(
                    ("in-memory-floor",) if item.support_object_id is None else ()
                ),
                domain_parent_object_ids=(
                    () if item.support_object_id is None else (item.support_object_id,)
                ),
                support_kind=(
                    "FLOOR" if item.support_object_id is None else "RECEPTACLE"
                ),
                support_object_id=item.support_object_id,
                floor_object_id=(
                    "in-memory-floor" if item.support_object_id is None else None
                ),
            )
            for item in scene.objects
        )

    def observe_source(
        self,
        source: CapturedSource,
        *,
        options: SourceCaptureOptions,
        settle: bool,
    ) -> SourceCaptureFacts:
        if type(source) is not CapturedSource or source.scene != self._scene:
            raise AdapterOperationError("missing source fixture")
        if type(options) is not SourceCaptureOptions or type(settle) is not bool:
            raise AdapterOperationError("invalid source capture options")
        if not settle:
            raise AdapterOperationError("source observation must settle")
        support_facts = self._support_facts(source.scene)
        polygon = source.scene.room_polygon_xy
        floor = None
        if len(polygon) >= 3:
            xs = tuple(point.x for point in polygon)
            ys = tuple(point.y for point in polygon)
            floor = AdapterFloorEnvelope(
                scene_id=source.scene.scene_id,
                floor_object_id="in-memory-floor",
                floor_name="in-memory-floor",
                native_aabb=OBB(
                    center=Vec3(
                        x=(min(xs) + max(xs)) / 2.0,
                        y=(min(ys) + max(ys)) / 2.0,
                        z=0.0,
                    ),
                    extent=Vec3(
                        x=max(xs) - min(xs),
                        y=max(ys) - min(ys),
                        z=0.01,
                    ),
                    rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                floor_top_z=0.0,
                clearance_m=options.floor_clearance_m,
                polygon_xy=tuple(Vec2(x=point.x, y=point.y) for point in polygon),
            )
        position = AdapterPosition(x=0.0, y=0.0, z=0.0)
        facts = SourceCaptureFacts(
            source=source,
            binding=source.binding,
            scene=source.scene,
            runtime_identity=self._runtime_identity(source),
            observation=self._observation(source.scene, settled=True),
            support_facts=support_facts,
            floor_envelope=floor,
            floor_position_regions=(),
            reachable_positions=(position,),
            current_pose=AdapterPose(
                position=position,
                yaw_degrees=0.0,
                horizon_degrees=0.0,
                standing=True,
            ),
            settlement_pass_steps=0,
        )
        self._execution_actions.append("settle_source")
        return facts

    def capture_spawn_maps(
        self,
        facts: SourceCaptureFacts,
        *,
        subject_object_ids: tuple[str, ...],
    ) -> tuple[AdapterSpawnMap, ...]:
        if (
            type(facts) is not SourceCaptureFacts
            or facts.source.scene != self._scene
            or facts.scene != self._scene
        ):
            raise AdapterOperationError("missing source fixture")
        if facts.support_facts != self._support_facts(self._scene):
            raise AdapterOperationError("source support facts changed")
        if type(subject_object_ids) is not tuple or any(
            type(item) is not str or not item for item in subject_object_ids
        ):
            raise AdapterOperationError("subject IDs must be exact")
        if len(set(subject_object_ids)) != len(subject_object_ids):
            raise AdapterOperationError("subject IDs must be unique")
        maps = []
        for subject_id in subject_object_ids:
            try:
                subject = facts.scene.object_by_id(subject_id)
            except (KeyError, ValueError) as error:
                raise AdapterOperationError("missing subject fixture") from error
            if subject.support_object_id is None:
                raise AdapterOperationError("missing subject support fixture")
            digest = sha256(subject_id.encode("utf-8")).hexdigest()
            maps.append(
                AdapterSpawnMap(
                    binding=facts.binding,
                    runtime_identity=facts.runtime_identity,
                    scene_id=facts.scene.scene_id,
                    subject_object_id=subject_id,
                    support_object_id=subject.support_object_id,
                    native_subject_object_id=f"in-memory:{subject_id}",
                    native_support_object_id=(f"in-memory:{subject.support_object_id}"),
                    positions=(),
                    positions_sha256=digest,
                    scene_sha256=digest,
                    source_sha256=digest,
                    surface_patches=(),
                )
            )
        captured = tuple(maps)
        self._execution_actions.extend(
            f"capture_spawn_maps:{item.subject_object_id}" for item in captured
        )
        return captured

    def pause_camera_observations(
        self,
        facts: SourceCaptureFacts,
        *,
        settle_after_resume: bool,
    ) -> CameraObservationHandle:
        if (
            type(facts) is not SourceCaptureFacts
            or facts.source.scene != self._scene
            or facts.binding != facts.source.binding
            or type(settle_after_resume) is not bool
        ):
            raise AdapterOperationError("invalid camera pause fixture")
        token = f"in-memory:{facts.binding.token}:{self._camera_pause_token_sequence}"
        self._camera_pause_token_sequence += 1
        if token in self._paused_camera_handles or token in self._resumed_tokens:
            raise AdapterOperationError("camera pause token collision")
        handle = CameraObservationHandle(
            source=facts.source,
            binding=facts.binding,
            scene=facts.scene,
            token=token,
            settle_after_resume=settle_after_resume,
        )
        self._paused_camera_handles[token] = handle
        return handle

    def resume_camera_observations(self, handle: CameraObservationHandle) -> None:
        if type(handle) is not CameraObservationHandle:
            raise AdapterOperationError("missing camera pause fixture")
        if handle.token in self._resumed_tokens:
            raise AdapterOperationError("camera observations already resumed")
        try:
            issued_handle = self._paused_camera_handles[handle.token]
        except KeyError as error:
            raise AdapterOperationError("missing camera pause fixture") from error
        if handle is not issued_handle or handle != issued_handle:
            raise AdapterOperationError("camera pause handle is not the issued handle")
        del self._paused_camera_handles[handle.token]
        self._resumed_tokens.add(handle.token)

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
        if (
            type(facts) is not SourceCaptureFacts
            or facts.source.scene != self._scene
            or facts.binding != facts.source.binding
            or type(pose) is not AdapterPose
            or type(source_scene) is not Scene
            or type(reset_from_source) is not bool
            or type(max_settlement_steps) is not int
            or max_settlement_steps <= 0
        ):
            raise AdapterOperationError("invalid camera application fixture")
        if handle is not None and (
            type(handle) is not CameraObservationHandle
            or handle.binding != facts.binding
            or handle.token not in self._paused_camera_handles
            or handle.token in self._resumed_tokens
        ):
            raise AdapterOperationError("invalid camera pause handle")
        observation = self._observation(source_scene, settled=True)
        return AdapterCameraApplication(
            source=facts.source,
            binding=facts.binding,
            requested_pose=pose,
            observed_pose=pose,
            observed_camera_position=pose.position,
            observed_scene=source_scene,
            observation=observation,
            position_residual_m=0.0,
            yaw_residual_degrees=0.0,
            horizon_residual_degrees=0.0,
        )

    def settle_camera_pose(
        self,
        facts: SourceCaptureFacts,
        pose: AdapterPose,
        *,
        source_scene: Scene,
        max_settlement_steps: int,
    ) -> AdapterSettledCameraApplication:
        application = self.apply_camera_pose(
            facts,
            pose,
            handle=None,
            source_scene=source_scene,
            reset_from_source=False,
            max_settlement_steps=max_settlement_steps,
        )
        return AdapterSettledCameraApplication(
            application=application,
            settlement_pass_steps=0,
        )

    def render_assets(
        self,
        scene: Scene,
        camera_id: str,
        destination_root: Path,
        stem: str,
    ) -> RenderedAssets:
        camera = scene.camera_by_id(camera_id)
        _validate_stem(stem)
        destination_root.mkdir(parents=True, exist_ok=True)
        rgb_path = destination_root / f"{stem}-rgb.png"
        depth_path = destination_root / f"{stem}-depth.npy"
        instance_path = destination_root / f"{stem}-instance.png"
        pointcloud_path = destination_root / f"{stem}-pointcloud.ply"
        rgb = Image.new("RGB", (camera.width, camera.height), "white")
        instance = Image.new("RGB", (camera.width, camera.height), "black")
        rgb_draw = ImageDraw.Draw(rgb)
        instance_draw = ImageDraw.Draw(instance)
        for index, object_ in enumerate(scene.objects, start=1):
            view = object_.views.get(camera_id)
            if view is None:
                continue
            box = (view.bbox.xmin, view.bbox.ymin, view.bbox.xmax, view.bbox.ymax)
            rgb_draw.rectangle(box, outline=(40, 90, 180), width=3)
            instance_draw.rectangle(
                box,
                fill=(index % 255, (index * 17) % 255, (index * 31) % 255),
            )
        rgb.save(rgb_path)
        instance.save(instance_path)
        np.save(
            depth_path,
            np.full((camera.height, camera.width), 2.0, dtype=np.float32),
            allow_pickle=False,
        )
        points = tuple(object_.obb.center for object_ in scene.objects)
        pointcloud_path.write_text(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            + "".join(f"{point.x} {point.y} {point.z}\n" for point in points),
            encoding="ascii",
        )
        return RenderedAssets(
            rgb_path=rgb_path,
            depth_path=depth_path,
            instance_path=instance_path,
            pointcloud_path=pointcloud_path,
        )
