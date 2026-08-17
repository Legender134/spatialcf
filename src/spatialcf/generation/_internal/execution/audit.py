"""Current one-shot endpoint audit authority."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import (
    AI2ThorAdapter,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorObservation,
    AI2ThorPoseApplication,
    AI2ThorReceptacleSpawnMap,
    AI2ThorRuntimeError,
    AI2ThorSettlementTimeout,
)
from spatialcf.core.v2.continuous_yaw_solve_verifier_v2_9 import (
    verify_continuous_yaw_solve_result_v2_9,
)
from spatialcf.domain.enums import QualityTier, SolverStatus
from spatialcf.domain.models import OBB, Camera, InterventionSpec, Scene
from spatialcf.domain.v2.base import (
    CanonicalId,
    FiniteFloat,
    NonNegativeFiniteFloat,
    Sha256Digest,
    V2Model,
    Vec3V2,
)
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawSolveVerificationKindV2,
)
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    SourceCameraEvidence,
    verify_competition_native_camera_observation_binding_v2_9_3,
)
from spatialcf.generation._internal.evidence.camera import (
    CompetitionNativeCameraPoseV2_9_3 as CameraPose,
)
from spatialcf.generation._internal.execution.correspondence import (
    CaptureSourceCorrespondence,
    legacy_sha256,
)
from spatialcf.generation._internal.planning.campaign import RuntimePosePolicy
from spatialcf.generation._internal.planning.models import ProxyBundle
from spatialcf.verification.verifier import VerificationResult, Verifier

_AUDIT_HASH_DOMAIN = "spatialcf.competition-native-endpoint-audit.v2.9.6"
_OBSERVATION_DOMAIN = b"spatialcf.competition-native-observation.v2.9\0"
_MAX_POSITION_RESIDUAL_M = 1e-5
_MAX_RUNTIME_POSITION_RESIDUAL_M = 1e-4
_OBJECT_GEOMETRY_TOLERANCE_M = 1e-5
_CAMERA_INTRINSIC_TOLERANCE = 1e-8
_CAMERA_EXTRINSIC_TOLERANCE = 1e-5


def observation_sha256(observation: AI2ThorObservation) -> str:
    if type(observation) is not AI2ThorObservation:
        raise TypeError("native observation digest requires an exact observation")
    assets = {
        "depth_npy_sha256": hashlib.sha256(observation.depth_npy).hexdigest(),
        "instance_png_sha256": hashlib.sha256(observation.instance_png).hexdigest(),
        "pointcloud_ply_sha256": hashlib.sha256(observation.pointcloud_ply).hexdigest(),
        "rgb_png_sha256": hashlib.sha256(observation.rgb_png).hexdigest(),
    }
    stored = {
        "depth_npy_sha256": observation.depth_npy_sha256,
        "instance_png_sha256": observation.instance_png_sha256,
        "pointcloud_ply_sha256": observation.pointcloud_ply_sha256,
        "rgb_png_sha256": observation.rgb_png_sha256,
    }
    if assets != stored:
        raise ValueError("native observation stored asset digests do not match bytes")
    counts = observation.instance_pixel_counts
    if any(
        type(key) is not str or type(value) is not int for key, value in counts.items()
    ):
        raise TypeError("native observation pixel counts must be exact")
    payload = json.dumps(
        {
            "assets": assets,
            "instance_pixel_counts": dict(sorted(counts.items())),
            "is_scene_at_rest": observation.is_scene_at_rest,
            "scene_sha256": legacy_sha256(observation.scene),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_OBSERVATION_DOMAIN + payload).hexdigest()


class EndpointAudit(V2Model):
    """Flat current bbox/runtime-pose endpoint audit."""

    audit_version: Literal["competition-native-endpoint-audit:2.9.6"] = (
        "competition-native-endpoint-audit:2.9.6"
    )
    audit_scope: Literal[
        "ONE_CAMERA_REPLAY_ONE_DELEGATED_BBOX_PATCH_SOLVE_ONE_FRESH_PATCH_"
        "ONE_NATIVE_ACTION_ONE_BOUNDED_SUBJECT_POSE"
    ] = (
        "ONE_CAMERA_REPLAY_ONE_DELEGATED_BBOX_PATCH_SOLVE_ONE_FRESH_PATCH_"
        "ONE_NATIVE_ACTION_ONE_BOUNDED_SUBJECT_POSE"
    )
    native_action: Literal["PlaceObjectAtPoint"] = "PlaceObjectAtPoint"
    native_audit_status: Literal["PASSED"] = "PASSED"
    evidence_eligible: Literal[False] = False
    case_id: CanonicalId
    native_scene_id: CanonicalId
    subject_native_object_id: CanonicalId
    reference_native_object_id: CanonicalId
    semantic_problem_sha256: Sha256Digest
    solver_config_sha256: Sha256Digest
    solve_result_sha256: Sha256Digest
    edit_sha256: Sha256Digest
    proxy_bundle_sha256: Sha256Digest
    spawn_map_source_sha256: Sha256Digest
    observed_scene_sha256: Sha256Digest
    after_observation_sha256: Sha256Digest
    commanded_position: Vec3V2
    observed_position: Vec3V2
    position_residual_m: NonNegativeFiniteFloat
    relation_before: CanonicalId
    relation_after: CanonicalId
    verification_status: Literal["success"] = "success"
    verification_quality: Literal["PURE", "LOW_LEAKAGE"]
    relation_damage_count: int = Field(strict=True, ge=0)
    relation_damage_items: tuple[CanonicalId, ...]
    semantic_normalization_scope: Literal[
        "FRESH_COLLATERAL_THEN_FROZEN_SOLVER_PROJECTION_V1"
    ] = "FRESH_COLLATERAL_THEN_FROZEN_SOLVER_PROJECTION_V1"
    frozen_source_scene_sha256: Sha256Digest
    fresh_source_scene_sha256: Sha256Digest
    source_correspondence_sha256: Sha256Digest
    fresh_spawn_map_source_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    placement_sha256: Sha256Digest
    surface_evidence_sha256: Sha256Digest
    subject_surface_evidence_sha256: Sha256Digest
    patch_index: int = Field(strict=True, ge=0)
    patch_sha256: Sha256Digest
    fresh_subject_surface_evidence_sha256: Sha256Digest
    fresh_patch_sha256: Sha256Digest
    endpoint_plan_sha256: Sha256Digest
    camera_evidence: SourceCameraEvidence
    camera_policy: CameraPolicy
    fresh_observed_pose: CameraPose
    fresh_observed_native_camera_position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    fresh_camera: Camera
    camera_replay_observation_sha256: Sha256Digest
    runtime_collision_delegated_native_object_ids: tuple[CanonicalId, ...]
    runtime_pose_policy: RuntimePosePolicy
    subject_rotation_residual_degrees: NonNegativeFiniteFloat
    subject_obb_corner_residual_m: NonNegativeFiniteFloat
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
    def validate_audit(self) -> Self:
        if self.position_residual_m > _MAX_RUNTIME_POSITION_RESIDUAL_M:
            raise ValueError("native endpoint residual exceeds the frozen limit")
        if self.relation_before == self.relation_after:
            raise ValueError("native audit must change the target relation")
        if self.relation_damage_items != tuple(sorted(set(self.relation_damage_items))):
            raise ValueError("relation damage items must be unique and canonical")
        if self.relation_damage_count != len(self.relation_damage_items):
            raise ValueError("relation damage count does not close its roster")
        return self

    @model_validator(mode="after")
    def validate_fresh_patch(self) -> Self:
        if (
            self.fresh_subject_surface_evidence_sha256
            != self.subject_surface_evidence_sha256
            or self.fresh_patch_sha256 != self.patch_sha256
        ):
            raise ValueError("fresh native patch does not close frozen patch lineage")
        return self

    @model_validator(mode="after")
    def validate_camera_replay(self) -> Self:
        if type(self.fresh_camera) is not Camera:
            raise TypeError("fresh camera replay Camera must be exact")
        camera = Camera.model_validate(
            self.fresh_camera.model_dump(mode="python"), strict=True
        )
        object.__setattr__(self, "fresh_camera", camera)
        if (
            self.camera_evidence.policy_sha256 != self.camera_policy.policy_sha256
            or self.camera_evidence.source_capture_sha256 != self.source_capture_sha256
        ):
            raise ValueError("camera replay audit frozen lineage is not closed")
        verify_competition_native_camera_observation_binding_v2_9_3(
            self.camera_evidence.requested_pose,
            self.fresh_observed_pose,
            self.fresh_observed_native_camera_position,
            camera,
        )
        return self

    @model_validator(mode="after")
    def validate_runtime_collision_authority(self) -> Self:
        delegated = self.runtime_collision_delegated_native_object_ids
        if delegated != tuple(sorted(set(delegated))):
            raise ValueError("runtime collision authority must be canonical")
        return self

    @model_validator(mode="after")
    def validate_runtime_pose_authority(self) -> Self:
        policy = self.runtime_pose_policy
        if (
            self.position_residual_m > policy.max_subject_position_residual_m
            or self.subject_rotation_residual_degrees
            > policy.max_subject_rotation_residual_degrees
            or self.subject_obb_corner_residual_m
            > policy.max_subject_obb_corner_residual_m
        ):
            raise ValueError("runtime subject pose exceeds the frozen policy")
        return self

    @property
    def endpoint_audit_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_AUDIT_HASH_DOMAIN)

    @property
    def competition_native_endpoint_audit_sha256(self) -> Sha256Digest:
        return self.endpoint_audit_sha256


class _NativeAfterStructureMismatch(ValueError):
    """The returned native scene changed facts outside the one allowed edit."""


class EndpointAuditRejected(RuntimeError):
    """A solve or native endpoint failed before an audit record was published."""

    def __init__(self, stage: str, reasons: tuple[str, ...]) -> None:
        if type(stage) is not str or not stage or not reasons:
            raise ValueError("a native audit rejection requires a stage and reasons")
        if type(reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in reasons
        ):
            raise TypeError("native audit rejection reasons must be exact strings")
        self.stage = stage
        self.reasons = tuple(sorted(set(reasons)))
        super().__init__(f"{stage}: {', '.join(self.reasons)}")


@dataclass(frozen=True)
class _EndpointExecution:
    audit: EndpointAudit | None
    application: AI2ThorPoseApplication
    capture_bound_audit_payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if (self.audit is None) == (self.capture_bound_audit_payload is None):
            raise ValueError(
                "endpoint execution requires exactly one closed audit representation"
            )


def execute_endpoint(
    adapter: AI2ThorAdapter,
    frozen_source_scene: Scene,
    fresh_source_scene: Scene,
    intervention: InterventionSpec,
    proxy_bundle: ProxyBundle,
    config: ContinuousYawSolverConfigV2_9,
    solve_result: ContinuousYawCertifiedSuccessResultV2_9,
    fresh_spawn_map: AI2ThorReceptacleSpawnMap,
    *,
    capture_bound_spawn_map_source_sha256: Sha256Digest,
    max_post_edit_pass_steps: int,
    runtime_pose_policy: RuntimePosePolicy,
    source_correspondence: CaptureSourceCorrespondence,
) -> _EndpointExecution:
    """Execute a frozen solve once against its corresponding fresh baseline."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        frozen = _strict_legacy(
            frozen_source_scene,
            Scene,
            "frozen_source_scene",
        )
        fresh = _strict_legacy(fresh_source_scene, Scene, "fresh_source_scene")
        spec = _strict_legacy(intervention, InterventionSpec, "intervention")
        bundle = _strict_v2(proxy_bundle, ProxyBundle, "proxy_bundle")
        checked_config = _strict_v2(config, ContinuousYawSolverConfigV2_9, "config")
        result = _strict_v2(
            solve_result,
            ContinuousYawCertifiedSuccessResultV2_9,
            "solve_result",
        )
        correspondence = _strict_v2(
            source_correspondence,
            CaptureSourceCorrespondence,
            "source_correspondence",
        )
        runtime_pose_policy = _strict_v2(
            runtime_pose_policy, RuntimePosePolicy, "runtime_pose_policy"
        )
    if correspondence.frozen_source_scene_sha256 != legacy_sha256(
        frozen
    ) or correspondence.fresh_source_scene_sha256 != legacy_sha256(fresh):
        raise EndpointAuditRejected(
            "native_precondition", ("capture_source_correspondence_mismatch",)
        )
    if type(capture_bound_spawn_map_source_sha256) is not str:
        raise TypeError("capture-bound spawn digest must be an exact string")

    binding = bundle.binding
    if binding.legacy_scene_sha256 != legacy_sha256(frozen):
        raise EndpointAuditRejected("input", ("legacy_scene_sha256_mismatch",))
    if binding.intervention_sha256 != legacy_sha256(spec):
        raise EndpointAuditRejected("input", ("intervention_sha256_mismatch",))
    if (
        result.semantic_problem_sha256
        != bundle.semantic_problem.semantic_problem_sha256
    ):
        raise EndpointAuditRejected("solve", ("semantic_problem_sha256_mismatch",))
    if type(fresh_spawn_map) is not AI2ThorReceptacleSpawnMap:
        raise TypeError("fresh_spawn_map must be an exact AI2ThorReceptacleSpawnMap")
    verified = verify_continuous_yaw_solve_result_v2_9(
        bundle.semantic_problem,
        checked_config,
        result,
    )
    if verified.kind is not ContinuousYawSolveVerificationKindV2.VERIFIED:
        raise EndpointAuditRejected(
            "solve",
            verified.finding_codes or ("fresh_solve_verification_failed",),
        )

    edit = result.selected_witness.edit
    expected_subject_id = f"object:{binding.subject_native_object_id}"
    if edit.subject_id != expected_subject_id:
        raise RuntimeError("verified edit subject does not match the proxy binding")
    frozen_subject = frozen.object_by_id(binding.subject_native_object_id)
    _require_native_endpoint_preconditions(
        fresh,
        frozen_subject.object_id,
        fresh_spawn_map,
    )
    endpoint_x = frozen_subject.position.x + edit.translation_xy_m.x
    endpoint_y = frozen_subject.position.y + edit.translation_xy_m.y
    fresh_spawn_map_source_sha256 = fresh_spawn_map.source_sha256
    before_native_event = adapter.latest_native_event(fresh.scene_id)
    try:
        application = adapter.apply_receptacle_endpoint_settled_observed(
            fresh,
            fresh_spawn_map,
            x=endpoint_x,
            y=endpoint_y,
            max_pass_steps=max_post_edit_pass_steps,
            max_subject_rotation_residual_degrees=(
                runtime_pose_policy.max_subject_rotation_residual_degrees
            ),
        )
    except (RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, AI2ThorSettlementTimeout):
            raise EndpointAuditRejected(
                "native_return", ("post_edit_scene_not_settled",)
            ) from error
        rejection = _native_action_rejection(
            adapter,
            fresh.scene_id,
            before_native_event,
            error,
        )
        if rejection is not None:
            raise EndpointAuditRejected(
                "native_action", (f"native_action_rejected:{rejection}",)
            ) from error
        returned = _native_return_rejection(
            adapter,
            fresh.scene_id,
            before_native_event,
            error,
        )
        if returned is not None:
            raise EndpointAuditRejected(
                "native_return", (f"native_return_rejected:{returned}",)
            ) from error
        raise
    if fresh_spawn_map.source_sha256 != fresh_spawn_map_source_sha256:
        raise RuntimeError("fresh spawn map changed during native endpoint audit")

    failures: list[str] = []
    if not application.is_scene_at_rest or not application.observation.is_scene_at_rest:
        failures.append("scene_not_at_rest")
    if application.subject_is_moving:
        failures.append("subject_is_moving")
    if application.observation.scene != application.observed_scene:
        failures.append("observation_scene_mismatch")
    if application.position_residual_m > _MAX_RUNTIME_POSITION_RESIDUAL_M:
        failures.append("position_residual_exceeded")
    try:
        fresh_normalized = _normalized_runtime_pose_observed_scene_v2_9_5(
            application.commanded_scene,
            application.observed_scene,
            binding.subject_native_object_id,
            runtime_pose_policy,
        )
    except _NativeAfterStructureMismatch as error:
        failures.append(f"after_scene_structure:{error}")
        frozen_normalized = None
    else:
        frozen_normalized = (
            _project_runtime_pose_observed_scene_to_frozen_source_v2_9_5(
                frozen,
                fresh_normalized,
                binding.subject_native_object_id,
            )
        )
    verification = None
    if frozen_normalized is not None:
        verification = _verify_minimum_cost_with_proxy_collision_authority(
            bundle,
            frozen,
            frozen_normalized,
            spec,
            runtime_pose_subject_object_id=binding.subject_native_object_id,
        )
        if verification.status is not SolverStatus.SUCCESS:
            failures.append(f"verification_status:{verification.status.value}")
        if verification.quality not in {QualityTier.PURE, QualityTier.LOW_LEAKAGE}:
            failures.append(f"verification_quality:{verification.quality.value}")
        failures.extend(f"verification_error:{item}" for item in verification.errors)
    if failures:
        raise EndpointAuditRejected("native_verification", tuple(sorted(set(failures))))
    if verification is None:
        raise RuntimeError("native verification result is missing")

    audit_payload: dict[str, object] = {
        "case_id": binding.case_id,
        "native_scene_id": binding.native_scene_id,
        "subject_native_object_id": binding.subject_native_object_id,
        "reference_native_object_id": binding.reference_native_object_id,
        "semantic_problem_sha256": (bundle.semantic_problem.semantic_problem_sha256),
        "solver_config_sha256": checked_config.config_sha256,
        "solve_result_sha256": result.solve_result_sha256,
        "edit_sha256": edit.edit_sha256,
        "proxy_bundle_sha256": bundle.proxy_bundle_sha256,
        "spawn_map_source_sha256": capture_bound_spawn_map_source_sha256,
        "fresh_spawn_map_source_sha256": fresh_spawn_map_source_sha256,
        "frozen_source_scene_sha256": legacy_sha256(frozen),
        "fresh_source_scene_sha256": legacy_sha256(fresh),
        "source_correspondence_sha256": (
            correspondence.competition_native_capture_source_correspondence_sha256
        ),
        "observed_scene_sha256": legacy_sha256(application.observed_scene),
        "after_observation_sha256": observation_sha256(application.observation),
        "commanded_position": Vec3V2(**application.commanded_position.model_dump()),
        "observed_position": Vec3V2(**application.observed_position.model_dump()),
        "position_residual_m": application.position_residual_m,
        "relation_before": spec.relation_before.value,
        "relation_after": spec.relation_after.value,
        "verification_quality": verification.quality.value,
        "relation_damage_count": verification.relation_damage_count,
        "relation_damage_items": verification.relation_damage_items,
    }
    commanded_subject = application.commanded_scene.object_by_id(
        binding.subject_native_object_id
    )
    observed_subject = application.observed_scene.object_by_id(
        binding.subject_native_object_id
    )
    audit_payload.update(
        runtime_pose_policy=runtime_pose_policy,
        subject_rotation_residual_degrees=_quaternion_angle_residual_deg(
            observed_subject.rotation,
            commanded_subject.rotation,
            "runtime subject rotation",
        ),
        subject_obb_corner_residual_m=_obb_corner_hausdorff_residual_m(
            observed_subject.obb,
            commanded_subject.obb,
            "runtime subject OBB",
        ),
    )
    return _EndpointExecution(
        audit=None,
        application=application,
        capture_bound_audit_payload=audit_payload,
    )


def _strict_legacy(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _strict_v2(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(
        value.model_dump(mode="python", warnings="error"), strict=True
    )


def _native_action_rejection(
    adapter: AI2ThorAdapter,
    scene_id: str,
    before_event: object,
    error: Exception,
) -> str | None:
    """Classify only an explicit failed native action as a request rejection."""

    if isinstance(error, AI2ThorRuntimeError):
        return None
    try:
        event = adapter.latest_native_event(scene_id)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    metadata = getattr(event, "metadata", None)
    if (
        event is before_event
        or not isinstance(metadata, dict)
        or metadata.get("lastActionSuccess") is not False
    ):
        return None
    message = metadata.get("errorMessage")
    if (
        type(message) is not str
        or not message.strip()
        or message.strip() != str(error).strip()
    ):
        return None
    return " ".join(message.split())


def _native_return_rejection(
    adapter: AI2ThorAdapter,
    scene_id: str,
    before_event: object,
    error: Exception,
    *,
    allow_existing_success_event: bool = False,
) -> str | None:
    """Classify a failed parse/validation only after a new successful event."""

    if not isinstance(error, AI2ThorNativeReturnError):
        return None
    try:
        event = adapter.latest_native_event(scene_id)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    metadata = getattr(event, "metadata", None)
    if (
        (event is before_event and not allow_existing_success_event)
        or not isinstance(metadata, dict)
        or metadata.get("lastActionSuccess") is not True
    ):
        return None
    message = " ".join(str(error).split())
    if not message:
        return None
    return f"{type(error).__name__}:{message}"


def _require_native_endpoint_preconditions(
    scene: Scene,
    subject_id: str,
    spawn_map: AI2ThorReceptacleSpawnMap,
) -> None:
    """Reject closed, expected endpoint preconditions before the native action."""

    subject = scene.object_by_id(subject_id)
    if subject.support_object_id is None:
        raise EndpointAuditRejected("native_precondition", ("subject_has_no_support",))
    if (
        spawn_map.scene_id != scene.scene_id
        or spawn_map.subject_object_id != subject.object_id
        or spawn_map.support_object_id != subject.support_object_id
    ):
        raise EndpointAuditRejected(
            "native_precondition", ("spawn_map_source_mismatch",)
        )
    positions = spawn_map.positions
    if type(positions) is not tuple:
        raise TypeError("spawn_map positions must be an exact tuple")
    if any(type(item) is not AI2ThorNativePosition for item in positions):
        raise TypeError("spawn_map positions must be exact native positions")
    if len({item.y for item in positions}) != 1:
        raise EndpointAuditRejected(
            "native_precondition", ("native_support_height_not_unique",)
        )


def _normalized_runtime_pose_observed_scene_v2_9_5(
    commanded: Scene,
    observed: Scene,
    subject_id: str,
    policy: RuntimePosePolicy,
) -> Scene:
    """Retain the bounded final subject pose/OBB and fresh object views."""

    checked_policy = RuntimePosePolicy.model_validate(
        policy.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    _require_native_after_structure(
        commanded,
        observed,
        subject_id,
        runtime_pose_policy=checked_policy,
    )
    observed_by_id = {item.object_id: item for item in observed.objects}
    normalized = []
    for expected in commanded.objects:
        current = observed_by_id[expected.object_id]
        normalized.append(
            current
            if expected.object_id == subject_id
            else expected.model_copy(update={"views": current.views})
        )
    return commanded.model_copy(update={"objects": tuple(normalized)})


def _verify_minimum_cost_with_proxy_collision_authority(
    bundle: ProxyBundle,
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
    *,
    runtime_pose_subject_object_id: str | None = None,
) -> VerificationResult:
    verifier = Verifier()
    if type(bundle) is not ProxyBundle:
        raise TypeError("proxy collision authority requires exact ProxyBundle")
    if runtime_pose_subject_object_id is not None:
        return verifier.verify_minimum_cost_with_runtime_pose_authority(
            before,
            after,
            spec,
            runtime_collision_delegated_object_ids=(
                bundle.binding.runtime_collision_delegated_native_object_ids
            ),
            runtime_pose_subject_object_id=runtime_pose_subject_object_id,
        )
    return verifier.verify_minimum_cost_with_runtime_collision_authority(
        before,
        after,
        spec,
        runtime_collision_delegated_object_ids=(
            bundle.binding.runtime_collision_delegated_native_object_ids
        ),
    )


def _project_runtime_pose_observed_scene_to_frozen_source_v2_9_5(
    frozen_source: Scene,
    fresh_after: Scene,
    subject_id: str,
) -> Scene:
    """Project actual final subject geometry onto frozen source roots."""

    fresh_by_id = {item.object_id: item for item in fresh_after.objects}
    projected = []
    for original in frozen_source.objects:
        current = fresh_by_id[original.object_id]
        projected.append(
            current
            if original.object_id == subject_id
            else original.model_copy(update={"views": current.views})
        )
    return frozen_source.model_copy(update={"objects": tuple(projected)})


def _require_native_after_structure(
    source: Scene,
    observed: Scene,
    subject_id: str,
    *,
    runtime_pose_policy: RuntimePosePolicy | None = None,
) -> None:
    """Close every returned fact except subject XY and fresh object views."""

    if (
        observed.scene_id != source.scene_id
        or observed.source != source.source
        or observed.coordinate_system != source.coordinate_system
        or observed.room_polygon_xy != source.room_polygon_xy
        or observed.collision_obstacles != source.collision_obstacles
        or observed.subject_position_regions != source.subject_position_regions
        or observed.pinned_object_ids != source.pinned_object_ids
        or observed.generation_seed != source.generation_seed
    ):
        raise _NativeAfterStructureMismatch("scene root facts changed")

    source_ids = tuple(item.object_id for item in source.objects)
    observed_ids = tuple(item.object_id for item in observed.objects)
    if (
        len(set(source_ids)) != len(source_ids)
        or len(set(observed_ids)) != len(observed_ids)
        or observed_ids != source_ids
        or subject_id not in set(source_ids)
    ):
        raise _NativeAfterStructureMismatch("object roster changed")

    for original, current in zip(source.objects, observed.objects, strict=True):
        if (
            current.object_id != original.object_id
            or current.name != original.name
            or current.category != original.category
            or current.movable is not original.movable
            or current.request_eligible is not original.request_eligible
            or current.support_object_id != original.support_object_id
        ):
            raise _NativeAfterStructureMismatch(
                f"object structural facts changed: {original.object_id}"
            )
        if original.object_id == subject_id:
            if runtime_pose_policy is not None:
                if (
                    math.dist(
                        _vec3_values(current.position),
                        _vec3_values(original.position),
                    )
                    > runtime_pose_policy.max_subject_position_residual_m
                ):
                    raise _NativeAfterStructureMismatch(
                        "subject runtime position residual exceeded"
                    )
                if (
                    _quaternion_angle_residual_deg(
                        current.rotation,
                        original.rotation,
                        "runtime subject rotation",
                    )
                    > runtime_pose_policy.max_subject_rotation_residual_degrees
                ):
                    raise _NativeAfterStructureMismatch(
                        "subject runtime rotation residual exceeded"
                    )
                if (
                    _obb_corner_hausdorff_residual_m(
                        current.obb,
                        original.obb,
                        "runtime subject OBB",
                    )
                    > runtime_pose_policy.max_subject_obb_corner_residual_m
                ):
                    raise _NativeAfterStructureMismatch(
                        "subject runtime OBB residual exceeded"
                    )
                continue
            if not _close_values(
                (current.position.z,),
                (original.position.z,),
                _OBJECT_GEOMETRY_TOLERANCE_M,
            ):
                raise _NativeAfterStructureMismatch("subject vertical position changed")
            expected_center = (
                original.obb.center.x + current.position.x - original.position.x,
                original.obb.center.y + current.position.y - original.position.y,
                original.obb.center.z,
            )
        else:
            if not _close_values(
                _vec3_values(current.position),
                _vec3_values(original.position),
                _OBJECT_GEOMETRY_TOLERANCE_M,
            ):
                raise _NativeAfterStructureMismatch(
                    f"stationary object position changed: {original.object_id}"
                )
            expected_center = _vec3_values(original.obb.center)
        if (
            not _close_values(
                _vec3_values(current.obb.center),
                expected_center,
                _OBJECT_GEOMETRY_TOLERANCE_M,
            )
            or not _close_values(
                _vec3_values(current.obb.extent),
                _vec3_values(original.obb.extent),
                _OBJECT_GEOMETRY_TOLERANCE_M,
            )
            or not _quaternions_close(current.rotation, original.rotation)
            or not _quaternions_close(current.obb.rotation, original.obb.rotation)
        ):
            raise _NativeAfterStructureMismatch(
                f"object geometry changed: {original.object_id}"
            )

    source_camera_ids = tuple(item.camera_id for item in source.cameras)
    observed_camera_ids = tuple(item.camera_id for item in observed.cameras)
    if (
        len(set(source_camera_ids)) != len(source_camera_ids)
        or len(set(observed_camera_ids)) != len(observed_camera_ids)
        or observed_camera_ids != source_camera_ids
    ):
        raise _NativeAfterStructureMismatch("camera roster changed")
    for original, current in zip(source.cameras, observed.cameras, strict=True):
        if (
            current.camera_id != original.camera_id
            or current.width != original.width
            or current.height != original.height
            or not _close_values(
                current.intrinsics,
                original.intrinsics,
                _CAMERA_INTRINSIC_TOLERANCE,
            )
            or not _close_values(
                current.world_to_camera,
                original.world_to_camera,
                _CAMERA_EXTRINSIC_TOLERANCE,
            )
        ):
            raise _NativeAfterStructureMismatch(
                f"camera facts changed: {original.camera_id}"
            )


def _vec3_values(value) -> tuple[float, float, float]:
    return value.x, value.y, value.z


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


def _close_values(left, right, tolerance: float) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right, strict=True)
    )


def _quaternions_close(left, right) -> bool:
    a = tuple(float(item) for item in (left.x, left.y, left.z, left.w))
    b = tuple(float(item) for item in (right.x, right.y, right.z, right.w))
    if not all(math.isfinite(item) for item in (*a, *b)):
        return False
    a_norm = math.sqrt(sum(item * item for item in a))
    b_norm = math.sqrt(sum(item * item for item in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return False
    normalized_a = tuple(item / a_norm for item in a)
    normalized_b = tuple(item / b_norm for item in b)
    return _close_values(
        normalized_a,
        normalized_b,
        _OBJECT_GEOMETRY_TOLERANCE_M,
    ) or _close_values(
        normalized_a,
        tuple(-item for item in normalized_b),
        _OBJECT_GEOMETRY_TOLERANCE_M,
    )


__all__ = (
    "EndpointAudit",
    "EndpointAuditRejected",
    "execute_endpoint",
    "observation_sha256",
)
