"""Current capture-to-fresh-source correspondence authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, model_validator

from spatialcf.domain.models import OBB, Scene
from spatialcf.domain.v2.base import NonNegativeFiniteFloat, Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    SourceCameraEvidence,
)
from spatialcf.generation._internal.evidence.surface import (
    SourceSurfaceEvidence,
    verify_source_surface_evidence,
)
from spatialcf.generation._internal.planning.campaign import (
    BatchRequest,
    RuntimePosePolicy,
)
from spatialcf.generation._internal.planning.models import EndpointPlan
from spatialcf.generation.capture.compiler import (
    score_competition_native_camera_capture_scene_v2_9_3,
)
from spatialcf.generation.capture.models import CompetitionNativeSourceCaptureV2_9

_CAPTURE_SOURCE_CORRESPONDENCE_HASH_DOMAIN_V2_9_6 = (
    "spatialcf.competition-native-capture-source-correspondence.v2.9.6"
)
_CAMERA_INTRINSIC_TOLERANCE = 1e-8
_CAMERA_EXTRINSIC_TOLERANCE = 1e-5
_CAPTURE_SOURCE_POSITION_COMPONENT_TOLERANCE_M = 1e-3
_CAPTURE_SOURCE_OBB_CORNER_TOLERANCE_M = 1e-2
_CAPTURE_SOURCE_ROTATION_TOLERANCE_DEG = 0.05
_CAPTURE_SOURCE_VIEW_BBOX_TOLERANCE_PX = 1.0
_CAPTURE_SOURCE_VIEW_DEPTH_TOLERANCE_M = 6e-3
_CAPTURE_SOURCE_VIEW_FRACTION_TOLERANCE = 5e-3
_REQUEST_LINEAGE_HASH_DOMAIN = (
    "spatialcf.competition-native-camera-patch-runtime-pose-bbox-request-lineage.v2.9.6"
)
_REQUEST_BINDING_HASH_DOMAIN = (
    "spatialcf.competition-native-camera-patch-request-binding.v2.9.3"
)
_CAMERA_RUNTIME_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"


def legacy_sha256(value: BaseModel) -> str:
    """Hash one legacy model under the frozen finite-number convention."""

    if not isinstance(value, BaseModel):
        raise TypeError("competition legacy digest requires a Pydantic model")
    payload = json.dumps(
        _stable_legacy_value(value.model_dump(mode="json")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_legacy_value(value: object) -> object:
    if isinstance(value, Enum):
        return _stable_legacy_value(value.value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("competition legacy digest requires finite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("competition legacy digest requires string mapping keys")
        return {key: _stable_legacy_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_legacy_value(item) for item in value]
    if isinstance(value, AbstractSet):
        normalized = [_stable_legacy_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    raise TypeError(
        f"unsupported competition legacy digest value {type(value).__name__!r}"
    )


def request_binding_sha256(request: BatchRequest) -> Sha256Digest:
    """Bind every exact current request field to its execution lineage."""

    if type(request) is not BatchRequest:
        raise TypeError("camera+patch batch request must be exact")
    checked = BatchRequest.model_validate(
        request.model_dump(mode="python", warnings="error"), strict=True
    )
    return canonical_sha256_v2(checked, domain=_REQUEST_BINDING_HASH_DOMAIN)


class RequestLineage(V2Model):
    """Flat current camera, patch, collision, pose, and bbox authority."""

    lineage_version: Literal[
        "competition-native-camera-patch-runtime-pose-bbox-request-lineage:2.9.6"
    ] = "competition-native-camera-patch-runtime-pose-bbox-request-lineage:2.9.6"
    frozen_source_scene_sha256: Sha256Digest
    source_capture: CompetitionNativeSourceCaptureV2_9
    source_capture_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    placement_sha256: Sha256Digest
    surface_evidence: SourceSurfaceEvidence
    surface_evidence_sha256: Sha256Digest
    subject_surface_evidence_sha256: Sha256Digest
    patch_index: int
    patch_sha256: Sha256Digest
    spawn_map_source_sha256: Sha256Digest
    semantic_problem_sha256: Sha256Digest
    proxy_bundle_sha256: Sha256Digest
    endpoint_plan_sha256: Sha256Digest
    solve_result_sha256: Sha256Digest
    batch_request_sha256: Sha256Digest
    endpoint_plan: EndpointPlan
    camera_evidence: SourceCameraEvidence
    camera_policy: CameraPolicy
    runtime_collision_delegated_native_object_ids: tuple[str, ...]
    runtime_pose_policy: RuntimePosePolicy
    visibility_semantics_id: Literal["visibility-semantics:analytic-bbox-v1"] = (
        "visibility-semantics:analytic-bbox-v1"
    )
    image_area_metric_definition_id: Literal[
        "visibility:visible-clipped-projected-bounding-box-area-fraction"
    ] = "visibility:visible-clipped-projected-bounding-box-area-fraction"
    image_area_metric_definition_version: Literal["definition:2"] = "definition:2"
    image_area_metric_formula: Literal[
        "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"
    ] = "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"

    @model_validator(mode="after")
    def validate_patch_lineage(self) -> Self:
        capture = self.source_capture
        evidence = verify_source_surface_evidence(capture, self.surface_evidence)
        if type(self.patch_index) is not int or self.patch_index < 0:
            raise ValueError("patch-bound lineage index must be non-negative and exact")
        subjects = tuple(
            item
            for item in evidence.subjects
            if item.subject_surface_evidence_sha256
            == self.subject_surface_evidence_sha256
        )
        if len(subjects) != 1 or self.patch_index >= len(subjects[0].patches):
            raise ValueError("patch-bound lineage subject patch is not unique")
        subject = subjects[0]
        patch = subject.patches[self.patch_index]
        if (
            legacy_sha256(capture.scene) != self.frozen_source_scene_sha256
            or capture.source_capture_sha256 != self.source_capture_sha256
            or evidence.surface_evidence_sha256 != self.surface_evidence_sha256
            or subject.runtime_identity_sha256 != self.runtime_identity_sha256
            or subject.placement_sha256 != self.placement_sha256
            or subject.spawn_map_source_sha256 != self.spawn_map_source_sha256
            or patch.patch_sha256 != self.patch_sha256
        ):
            raise ValueError("patch-bound request lineage is not closed")
        return self

    @model_validator(mode="after")
    def validate_camera_runtime_lineage(self) -> Self:
        capture = self.source_capture
        evidence = self.camera_evidence
        endpoint = self.endpoint_plan
        delegated = self.runtime_collision_delegated_native_object_ids
        source = capture.source
        if (
            type(endpoint) is not EndpointPlan
            or delegated != tuple(sorted(set(delegated)))
            or endpoint.runtime_collision_delegated_native_object_ids != delegated
            or endpoint.endpoint_plan_sha256 != self.endpoint_plan_sha256
            or endpoint.patch_index != self.patch_index
            or endpoint.patch_sha256 != self.patch_sha256
            or endpoint.source_capture_sha256 != self.source_capture_sha256
            or endpoint.placement_sha256 != self.placement_sha256
            or endpoint.surface_evidence_sha256 != self.surface_evidence_sha256
            or endpoint.subject_surface_evidence_sha256
            != self.subject_surface_evidence_sha256
            or endpoint.proxy_bundle_sha256 != self.proxy_bundle_sha256
            or endpoint.solve_result_sha256 != self.solve_result_sha256
        ):
            raise ValueError("camera+patch runtime endpoint plan is not closed")
        if (
            evidence.source_id != source.source_id
            or evidence.scene_id != source.scene_id
            or evidence.source_locator_sha256 != source.source_locator_sha256
            or evidence.runtime_identity_sha256
            != canonical_sha256_v2(
                capture.runtime_identity, domain=_CAMERA_RUNTIME_HASH_DOMAIN
            )
            or evidence.source_capture_sha256 != capture.source_capture_sha256
            or evidence.policy_sha256 != self.camera_policy.policy_sha256
            or evidence.camera != capture.scene.camera_by_id("main")
            or evidence.score
            != score_competition_native_camera_capture_scene_v2_9_3(
                capture, evidence, capture.scene
            )
            or evidence.rgb_png_sha256 != capture.rgb_png_sha256
            or evidence.depth_npy_sha256 != capture.depth_npy_sha256
            or evidence.instance_png_sha256 != capture.instance_png_sha256
            or evidence.pointcloud_ply_sha256 != capture.pointcloud_ply_sha256
            or evidence.is_scene_at_rest is not capture.is_scene_at_rest
        ):
            raise ValueError("camera+patch runtime request lineage is not closed")
        return self

    @property
    def request_lineage_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_REQUEST_LINEAGE_HASH_DOMAIN)

    @property
    def competition_native_camera_patch_runtime_request_lineage_sha256(
        self,
    ) -> Sha256Digest:
        return self.request_lineage_sha256


class CaptureSourceCorrespondence(V2Model):
    """Source replay certificate using physical OBB corner-set residuals."""

    validation_version: Literal[
        "competition-native-capture-source-correspondence:2.9.6"
    ] = "competition-native-capture-source-correspondence:2.9.6"
    frozen_source_scene_sha256: Sha256Digest
    fresh_source_scene_sha256: Sha256Digest
    maximum_position_component_residual_m: NonNegativeFiniteFloat
    maximum_obb_corner_residual_m: NonNegativeFiniteFloat
    maximum_object_rotation_residual_deg: NonNegativeFiniteFloat
    maximum_camera_intrinsic_residual: NonNegativeFiniteFloat
    maximum_camera_extrinsic_residual: NonNegativeFiniteFloat
    maximum_view_bbox_residual_px: NonNegativeFiniteFloat
    maximum_view_depth_residual_m: NonNegativeFiniteFloat
    maximum_view_fraction_residual: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        limits = (
            (
                "maximum_position_component_residual_m",
                self.maximum_position_component_residual_m,
                _CAPTURE_SOURCE_POSITION_COMPONENT_TOLERANCE_M,
            ),
            (
                "maximum_obb_corner_residual_m",
                self.maximum_obb_corner_residual_m,
                _CAPTURE_SOURCE_OBB_CORNER_TOLERANCE_M,
            ),
            (
                "maximum_object_rotation_residual_deg",
                self.maximum_object_rotation_residual_deg,
                _CAPTURE_SOURCE_ROTATION_TOLERANCE_DEG,
            ),
            (
                "maximum_camera_intrinsic_residual",
                self.maximum_camera_intrinsic_residual,
                _CAMERA_INTRINSIC_TOLERANCE,
            ),
            (
                "maximum_camera_extrinsic_residual",
                self.maximum_camera_extrinsic_residual,
                _CAMERA_EXTRINSIC_TOLERANCE,
            ),
            (
                "maximum_view_bbox_residual_px",
                self.maximum_view_bbox_residual_px,
                _CAPTURE_SOURCE_VIEW_BBOX_TOLERANCE_PX,
            ),
            (
                "maximum_view_depth_residual_m",
                self.maximum_view_depth_residual_m,
                _CAPTURE_SOURCE_VIEW_DEPTH_TOLERANCE_M,
            ),
            (
                "maximum_view_fraction_residual",
                self.maximum_view_fraction_residual,
                _CAPTURE_SOURCE_VIEW_FRACTION_TOLERANCE,
            ),
        )
        violations = tuple(
            f"{field}={value!r}>{limit!r}"
            for field, value, limit in limits
            if value > limit
        )
        if violations:
            raise ValueError(
                "capture source correspondence exceeds frozen limits: "
                + ", ".join(violations)
            )
        return self

    @property
    def competition_native_capture_source_correspondence_sha256(
        self,
    ) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_CAPTURE_SOURCE_CORRESPONDENCE_HASH_DOMAIN_V2_9_6,
        )


def _validate_capture_source(
    frozen_source_scene: Scene,
    fresh_source_scene: Scene,
) -> CaptureSourceCorrespondence:
    """Prove that a fresh settlement still represents one frozen capture.

    Identity, semantic, support, render, camera and non-geometry scene facts
    remain exact.  Native float residuals are measured field by field; values
    are never rounded or hidden behind a hash comparison.
    """

    frozen = _strict_legacy(frozen_source_scene, Scene, "frozen_source_scene")
    fresh = _strict_legacy(fresh_source_scene, Scene, "fresh_source_scene")
    if (
        fresh.scene_id != frozen.scene_id
        or fresh.source != frozen.source
        or fresh.coordinate_system != frozen.coordinate_system
        or fresh.room_polygon_xy != frozen.room_polygon_xy
        or fresh.collision_obstacles != frozen.collision_obstacles
        or fresh.subject_position_regions != frozen.subject_position_regions
        or fresh.pinned_object_ids != frozen.pinned_object_ids
        or fresh.generation_seed != frozen.generation_seed
    ):
        raise ValueError("capture source root facts changed")

    frozen_objects = tuple(item.object_id for item in frozen.objects)
    fresh_objects = tuple(item.object_id for item in fresh.objects)
    if (
        len(set(frozen_objects)) != len(frozen_objects)
        or len(set(fresh_objects)) != len(fresh_objects)
        or set(fresh_objects) != set(frozen_objects)
    ):
        raise ValueError("capture source object roster changed")
    frozen_by_id = {item.object_id: item for item in frozen.objects}
    fresh_by_id = {item.object_id: item for item in fresh.objects}

    maximum_position = maximum_obb_corner = 0.0
    maximum_object_rotation = 0.0
    maximum_bbox = maximum_view_depth = maximum_view_fraction = 0.0
    for object_id in sorted(frozen_by_id):
        expected = frozen_by_id[object_id]
        observed = fresh_by_id[object_id]
        if (
            observed.name != expected.name
            or observed.category != expected.category
            or observed.movable is not expected.movable
            or observed.request_eligible is not expected.request_eligible
            or observed.support_object_id != expected.support_object_id
            or tuple(sorted(observed.views)) != tuple(sorted(expected.views))
        ):
            raise ValueError(
                f"capture source object facts changed: {expected.object_id}"
            )
        maximum_position = max(
            maximum_position,
            _maximum_finite_component_residual(
                _vec3_values(observed.position),
                _vec3_values(expected.position),
                f"capture source object position: {expected.object_id}",
            ),
        )
        object_rotation = _quaternion_angle_residual_deg(
            observed.rotation,
            expected.rotation,
            f"capture source object rotation: {expected.object_id}",
        )
        maximum_object_rotation = max(maximum_object_rotation, object_rotation)
        maximum_obb_corner = max(
            maximum_obb_corner,
            _obb_corner_hausdorff_residual_m(
                observed.obb,
                expected.obb,
                f"capture source object OBB geometry: {expected.object_id}",
            ),
        )
        for camera_id in sorted(expected.views):
            expected_view = expected.views[camera_id]
            observed_view = observed.views[camera_id]
            if (
                expected_view.camera_id != camera_id
                or observed_view.camera_id != camera_id
            ):
                raise ValueError(
                    f"capture source object view camera changed: {expected.object_id}"
                )
            maximum_bbox = max(
                maximum_bbox,
                _maximum_finite_component_residual(
                    (
                        observed_view.bbox.xmin,
                        observed_view.bbox.ymin,
                        observed_view.bbox.xmax,
                        observed_view.bbox.ymax,
                    ),
                    (
                        expected_view.bbox.xmin,
                        expected_view.bbox.ymin,
                        expected_view.bbox.xmax,
                        expected_view.bbox.ymax,
                    ),
                    f"capture source object view bbox: {expected.object_id}",
                ),
            )
            maximum_view_depth = max(
                maximum_view_depth,
                _maximum_finite_component_residual(
                    (observed_view.camera_depth,),
                    (expected_view.camera_depth,),
                    f"capture source object view depth: {expected.object_id}",
                ),
            )
            maximum_view_fraction = max(
                maximum_view_fraction,
                _maximum_finite_component_residual(
                    (
                        observed_view.visible_fraction,
                        observed_view.image_area_fraction,
                        observed_view.truncated_fraction,
                    ),
                    (
                        expected_view.visible_fraction,
                        expected_view.image_area_fraction,
                        expected_view.truncated_fraction,
                    ),
                    f"capture source object view fractions: {expected.object_id}",
                ),
            )

    frozen_cameras = tuple(item.camera_id for item in frozen.cameras)
    fresh_cameras = tuple(item.camera_id for item in fresh.cameras)
    if (
        len(set(frozen_cameras)) != len(frozen_cameras)
        or len(set(fresh_cameras)) != len(fresh_cameras)
        or set(fresh_cameras) != set(frozen_cameras)
    ):
        raise ValueError("capture source camera roster changed")
    frozen_cameras_by_id = {item.camera_id: item for item in frozen.cameras}
    fresh_cameras_by_id = {item.camera_id: item for item in fresh.cameras}
    maximum_intrinsic = maximum_extrinsic = 0.0
    for camera_id in sorted(frozen_cameras_by_id):
        expected = frozen_cameras_by_id[camera_id]
        observed = fresh_cameras_by_id[camera_id]
        if observed.width != expected.width or observed.height != expected.height:
            raise ValueError(
                f"capture source camera render facts changed: {expected.camera_id}"
            )
        maximum_intrinsic = max(
            maximum_intrinsic,
            _maximum_finite_component_residual(
                observed.intrinsics,
                expected.intrinsics,
                f"capture source camera intrinsics: {expected.camera_id}",
            ),
        )
        maximum_extrinsic = max(
            maximum_extrinsic,
            _maximum_finite_component_residual(
                observed.world_to_camera,
                expected.world_to_camera,
                f"capture source camera extrinsics: {expected.camera_id}",
            ),
        )

    common = {
        "frozen_source_scene_sha256": legacy_sha256(frozen),
        "fresh_source_scene_sha256": legacy_sha256(fresh),
        "maximum_position_component_residual_m": maximum_position,
        "maximum_camera_intrinsic_residual": maximum_intrinsic,
        "maximum_camera_extrinsic_residual": maximum_extrinsic,
        "maximum_view_bbox_residual_px": maximum_bbox,
        "maximum_view_depth_residual_m": maximum_view_depth,
        "maximum_view_fraction_residual": maximum_view_fraction,
    }
    return CaptureSourceCorrespondence(
        **common,
        maximum_obb_corner_residual_m=maximum_obb_corner,
        maximum_object_rotation_residual_deg=maximum_object_rotation,
    )


def validate_capture_source(
    frozen_source_scene: Scene,
    fresh_source_scene: Scene,
) -> CaptureSourceCorrespondence:
    """Compare stable object pose and the physical world-space OBB corner set."""

    correspondence = _validate_capture_source(
        frozen_source_scene,
        fresh_source_scene,
    )
    if type(correspondence) is not CaptureSourceCorrespondence:
        raise RuntimeError("capture correspondence returned the wrong current type")
    return correspondence


def _strict_legacy(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _vec3_values(value) -> tuple[float, float, float]:
    return value.x, value.y, value.z


def _maximum_finite_component_residual(left, right, label: str) -> float:
    if len(left) != len(right) or not all(
        math.isfinite(float(item)) for item in (*left, *right)
    ):
        raise ValueError(f"{label} must contain matching finite values")
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)),
        default=0.0,
    )


def _quaternion_angle_residual_deg(left, right, label: str) -> float:
    a = tuple(float(item) for item in (left.x, left.y, left.z, left.w))
    b = tuple(float(item) for item in (right.x, right.y, right.z, right.w))
    if not all(math.isfinite(item) for item in (*a, *b)):
        raise ValueError(f"{label} must contain finite values")
    a_norm = math.sqrt(sum(item * item for item in a))
    b_norm = math.sqrt(sum(item * item for item in b))
    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError(f"{label} must contain non-zero quaternions")
    cosine = abs(
        sum(
            (a_item / a_norm) * (b_item / b_norm)
            for a_item, b_item in zip(a, b, strict=True)
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, cosine))))


def _obb_corner_coordinates(obb: OBB, label: str) -> tuple[tuple[float, ...], ...]:
    values = (
        obb.center.x,
        obb.center.y,
        obb.center.z,
        obb.extent.x,
        obb.extent.y,
        obb.extent.z,
        obb.rotation.x,
        obb.rotation.y,
        obb.rotation.z,
        obb.rotation.w,
    )
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError(f"{label} must contain finite values")
    if any(float(item) <= 0.0 for item in (obb.extent.x, obb.extent.y, obb.extent.z)):
        raise ValueError(f"{label} must contain strictly positive extents")
    x, y, z, w = (
        float(obb.rotation.x),
        float(obb.rotation.y),
        float(obb.rotation.z),
        float(obb.rotation.w),
    )
    maximum_component = max(abs(x), abs(y), abs(z), abs(w))
    if maximum_component == 0.0:
        raise ValueError(f"{label} must contain a non-zero quaternion")
    scaled = tuple(component / maximum_component for component in (x, y, z, w))
    scaled_norm = math.sqrt(sum(component * component for component in scaled))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise ValueError(f"{label} must contain a normalizable quaternion")
    x, y, z, w = (component / scaled_norm for component in scaled)
    rotation = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    center = (float(obb.center.x), float(obb.center.y), float(obb.center.z))
    half_extent = (
        float(obb.extent.x) / 2.0,
        float(obb.extent.y) / 2.0,
        float(obb.extent.z) / 2.0,
    )
    return tuple(
        tuple(
            center[row]
            + sum(
                rotation[row][column] * signs[column] * half_extent[column]
                for column in range(3)
            )
            for row in range(3)
        )
        for signs in (
            (dx, dy, dz)
            for dx in (-1.0, 1.0)
            for dy in (-1.0, 1.0)
            for dz in (-1.0, 1.0)
        )
    )


def _obb_corner_hausdorff_residual_m(left: OBB, right: OBB, label: str) -> float:
    left_corners = _obb_corner_coordinates(left, label)
    right_corners = _obb_corner_coordinates(right, label)

    def directed(source, target) -> float:
        return max(
            min(math.dist(source_corner, target_corner) for target_corner in target)
            for source_corner in source
        )

    return max(
        directed(left_corners, right_corners),
        directed(right_corners, left_corners),
    )


__all__ = (
    "CaptureSourceCorrespondence",
    "RequestLineage",
    "legacy_sha256",
    "request_binding_sha256",
    "validate_capture_source",
)
