"""Final one-shot endpoint audit and execution authority."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from math import dist
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.base import (
    AdapterActionRejected,
    AdapterObservation,
    AdapterOperationError,
    AdapterPose,
    AdapterPosition,
    AdapterReturnRejected,
    AdapterSettlementTimeout,
    AdapterSpawnMap,
    CapturedSource,
    CaptureRequest,
    CertifiedEditApplication,
    EnvironmentAdapter,
    SettledReadback,
    SourceCaptureFacts,
    SourceCaptureOptions,
    capture_bound_adapter_spawn_map,
)
from spatialcf.adapters.base import AdapterObservation as AI2ThorObservation
from spatialcf.adapters.base import EnvironmentAdapter as AI2ThorAdapter
from spatialcf.core.solver import solve_minimum_cost
from spatialcf.core.verification import (
    verify_solve_result,
)
from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FiniteFloat,
    NonNegativeFiniteFloat,
    Sha256Digest,
    Vec3,
)
from spatialcf.domain.edit import CanonicalEdit
from spatialcf.domain.request import InterventionSpec, QualityTier, SolverStatus
from spatialcf.domain.scene import OBB, Camera, Scene
from spatialcf.domain.serialization import canonical_sha256
from spatialcf.domain.solver import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawSolveVerificationKindV2,
)
from spatialcf.generation.capture.compiler import (
    score_competition_native_camera_capture_scene_v2_9_3,
)
from spatialcf.generation.capture.models import (
    _SUBJECT_EVIDENCE_HASH_DOMAIN,
    CameraPolicy,
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceCaptureV2_9,
    SourceCameraEvidence,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    _build_patch,
    _subject_payload,
    build_competition_native_settled_camera_policy_v2_9_10,
    verify_competition_native_camera_observation_binding_v2_9_3,
    verify_source_surface_evidence,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeCameraPoseV2_9_3 as CameraPose,
)
from spatialcf.generation.execution.correspondence import (
    CaptureSourceCorrespondence,
    RequestLineage,
    legacy_sha256,
    request_binding_sha256,
    validate_capture_source,
)
from spatialcf.generation.planning.campaign import BatchRequest, RuntimePosePolicy
from spatialcf.generation.planning.models import (
    CollisionDelegation,
    EndpointWorkspace,
    ProxyBundle,
    SubjectPlacementFact,
)
from spatialcf.generation.planning.problem import build_proxy_bundle
from spatialcf.generation.workflows import contracts as workflow_contracts
from spatialcf.verification.integrity import (
    competition_native_observation_payload_sha256,
)
from spatialcf.verification.verifier import VerificationResult, Verifier

_AUDIT_HASH_DOMAIN = "spatialcf.competition-native-endpoint-audit.v2.9.6"
_MAX_POSITION_RESIDUAL_M = 1e-5
_MAX_RUNTIME_POSITION_RESIDUAL_M = 1e-4
_OBJECT_GEOMETRY_TOLERANCE_M = 1e-5
_CAMERA_INTRINSIC_TOLERANCE = 1e-8
_CAMERA_EXTRINSIC_TOLERANCE = 1e-5
_RUN_HASH_DOMAIN = "spatialcf.competition-native-audit-run.v2.9.7"
_EXECUTION_HASH_DOMAIN = "spatialcf.competition-native-audit-execution.v2.9.6"
_CAMERA_RUNTIME_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_MAX_SETTLEMENT_STEPS = 600


def observation_sha256(observation: AdapterObservation) -> str:
    if type(observation) is not AdapterObservation:
        raise TypeError("adapter observation digest requires an exact observation")
    return competition_native_observation_payload_sha256(
        scene=observation.scene,
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=dict(observation.instance_pixel_counts),
        is_scene_at_rest=observation.is_settled,
    )


class EndpointAudit(CanonicalModel):
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
    commanded_position: Vec3
    observed_position: Vec3
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
        return canonical_sha256(self, domain=_AUDIT_HASH_DOMAIN)

    @property
    def competition_native_endpoint_audit_sha256(self) -> Sha256Digest:
        return self.endpoint_audit_sha256


class _NativeAfterStructureMismatch(ValueError):
    """The returned native scene changed facts outside the one allowed edit."""


class EndpointAuditRejected(RuntimeError):
    """A solve or native endpoint failed before an audit record was published."""

    def __init__(
        self,
        stage: str,
        reasons: tuple[str, ...],
        *,
        readback: SettledReadback | None = None,
    ) -> None:
        if type(stage) is not str or not stage or not reasons:
            raise ValueError("a native audit rejection requires a stage and reasons")
        if type(reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in reasons
        ):
            raise TypeError("native audit rejection reasons must be exact strings")
        if stage == "native_verification":
            if type(readback) is not SettledReadback:
                raise TypeError(
                    "native verification rejection requires an exact settled readback"
                )
        elif readback is not None:
            raise ValueError(
                "only native verification rejection may retain a settled readback"
            )
        self.stage = stage
        self.reasons = tuple(sorted(set(reasons)))
        self.readback = readback
        super().__init__(f"{stage}: {', '.join(self.reasons)}")


@dataclass(frozen=True)
class _EndpointExecution:
    audit: EndpointAudit | None
    application: SettledReadback
    capture_bound_audit_payload: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if (self.audit is None) == (self.capture_bound_audit_payload is None):
            raise ValueError(
                "endpoint execution requires exactly one closed audit representation"
            )


def execute_endpoint(
    adapter: EnvironmentAdapter,
    frozen_source_scene: Scene,
    fresh_source: CapturedSource,
    intervention: InterventionSpec,
    proxy_bundle: ProxyBundle,
    config: ContinuousYawSolverConfigV2_9,
    solve_result: ContinuousYawCertifiedSuccessResultV2_9,
    fresh_spawn_map: AdapterSpawnMap,
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
        if type(fresh_source) is not CapturedSource:
            raise TypeError("fresh_source must be an exact CapturedSource")
        fresh = _strict_legacy(fresh_source.scene, Scene, "fresh_source_scene")
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
    if type(fresh_spawn_map) is not AdapterSpawnMap:
        raise TypeError("fresh_spawn_map must be an exact AdapterSpawnMap")
    verified = verify_solve_result(
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
    _require_endpoint_preconditions(
        fresh,
        frozen_subject.object_id,
        fresh_spawn_map,
    )
    fresh_spawn_map_source_sha256 = fresh_spawn_map.source_sha256
    adapter_edit = CanonicalEdit(
        semantic_problem_sha256=edit.semantic_problem_sha256,
        subject_id=frozen_subject.object_id,
        translation_xy_m=edit.translation_xy_m,
    )
    certified = CertifiedEditApplication(
        source=fresh_source,
        intervention=spec,
        edit=adapter_edit,
        spawn_map=fresh_spawn_map,
        max_settlement_steps=max_post_edit_pass_steps,
    )
    try:
        application = adapter.apply_certified_edit(certified)
        settled = adapter.settle_readback(application)
    except AdapterSettlementTimeout as error:
        raise EndpointAuditRejected(
            "native_return", ("post_edit_scene_not_settled",)
        ) from error
    except AdapterActionRejected as error:
        raise EndpointAuditRejected(
            "native_action", (f"native_action_rejected:{error.reason}",)
        ) from error
    except AdapterReturnRejected as error:
        raise EndpointAuditRejected(
            "native_return", (f"native_return_rejected:{error.reason}",)
        ) from error
    except AdapterOperationError as error:
        raise EndpointAuditRejected(
            "native_precondition",
            (f"certified_edit_failed:{type(error).__name__}:{error}",),
        ) from error
    if fresh_spawn_map.source_sha256 != fresh_spawn_map_source_sha256:
        raise RuntimeError("fresh spawn map changed during native endpoint audit")

    failures: list[str] = []
    if not settled.is_scene_at_rest or not settled.observation.is_settled:
        failures.append("scene_not_at_rest")
    if settled.subject_is_moving:
        failures.append("subject_is_moving")
    if settled.observation.scene != settled.observed_scene:
        failures.append("observation_scene_mismatch")
    if settled.position_residual_m > _MAX_RUNTIME_POSITION_RESIDUAL_M:
        failures.append("position_residual_exceeded")
    try:
        fresh_normalized = _normalized_runtime_pose_observed_scene_v2_9_5(
            settled.commanded_scene,
            settled.observed_scene,
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
        raise EndpointAuditRejected(
            "native_verification",
            tuple(sorted(set(failures))),
            readback=settled,
        )
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
        "observed_scene_sha256": legacy_sha256(settled.observed_scene),
        "after_observation_sha256": observation_sha256(settled.observation),
        "commanded_position": Vec3(
            x=settled.commanded_position.x,
            y=settled.commanded_position.y,
            z=settled.commanded_position.z,
        ),
        "observed_position": Vec3(
            x=settled.observed_position.x,
            y=settled.observed_position.y,
            z=settled.observed_position.z,
        ),
        "position_residual_m": settled.position_residual_m,
        "relation_before": spec.relation_before.value,
        "relation_after": spec.relation_after.value,
        "verification_quality": verification.quality.value,
        "relation_damage_count": verification.relation_damage_count,
        "relation_damage_items": verification.relation_damage_items,
    }
    commanded_subject = settled.commanded_scene.object_by_id(
        binding.subject_native_object_id
    )
    observed_subject = settled.observed_scene.object_by_id(
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
        application=settled,
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


def _require_endpoint_preconditions(
    scene: Scene,
    subject_id: str,
    spawn_map: AdapterSpawnMap,
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
    if any(type(item) is not AdapterPosition for item in positions):
        raise TypeError("spawn_map positions must be exact adapter positions")
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


class AuditRun(CanonicalModel):
    """Flat current replayable CPU chain plus one native endpoint audit."""

    run_version: Literal["competition-native-audit-run:2.9.7"] = (
        "competition-native-audit-run:2.9.7"
    )
    orchestration_scope: Literal[
        "ONE_CAMERA_REPLAY_ONE_DELEGATED_BBOX_PATCH_SOLVE_ONE_FRESH_PATCH_"
        "ONE_EDIT_ACTION_ONE_BOUNDED_SUBJECT_POSE"
    ] = (
        "ONE_CAMERA_REPLAY_ONE_DELEGATED_BBOX_PATCH_SOLVE_ONE_FRESH_PATCH_"
        "ONE_EDIT_ACTION_ONE_BOUNDED_SUBJECT_POSE"
    )
    evidence_eligible: Literal[False] = False
    case_id: CanonicalId
    source_scene: Scene
    intervention: InterventionSpec
    endpoint_workspace: EndpointWorkspace
    proxy_bundle: ProxyBundle
    solver_config: ContinuousYawSolverConfigV2_9
    solve_result: ContinuousYawCertifiedSuccessResultV2_9
    before_observation_sha256: Sha256Digest
    native_audit: EndpointAudit
    native_render_width_px: int = Field(strict=True, gt=0)
    native_render_height_px: int = Field(strict=True, gt=0)
    native_seed: int = Field(strict=True)
    max_settlement_steps: int = Field(strict=True, gt=0, le=_MAX_SETTLEMENT_STEPS)
    before_observation_contract: Literal[
        "FRESH_ASSETS_NORMALIZED_TO_FROZEN_SOLVER_SOURCE_V1"
    ] = "FRESH_ASSETS_NORMALIZED_TO_FROZEN_SOLVER_SOURCE_V1"
    fresh_source_scene: Scene
    fresh_before_observation_sha256: Sha256Digest
    source_correspondence: CaptureSourceCorrespondence
    source_capture: CompetitionNativeSourceCaptureV2_9
    surface_evidence: SourceSurfaceEvidence
    endpoint_plan_sha256: Sha256Digest
    camera_evidence: SourceCameraEvidence
    camera_policy: CameraPolicy
    runtime_pose_policy: RuntimePosePolicy

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        source = _strict_legacy(self.source_scene, Scene, "source_scene")
        intervention = _strict_legacy(
            self.intervention, InterventionSpec, "intervention"
        )
        binding = self.proxy_bundle.binding
        audit = self.native_audit
        if self.case_id != binding.case_id or self.case_id != audit.case_id:
            raise ValueError("native audit run case ID is not closed")
        if binding.endpoint_workspace != self.endpoint_workspace:
            raise ValueError("native audit run endpoint workspace is not closed")
        if source.scene_id != binding.native_scene_id:
            raise ValueError("native audit run source scene is not closed")
        if binding.legacy_scene_sha256 != legacy_sha256(source):
            raise ValueError("native audit run source scene hash is not closed")
        if binding.intervention_sha256 != legacy_sha256(intervention):
            raise ValueError("native audit run intervention hash is not closed")
        if intervention.subject_id != binding.subject_native_object_id:
            raise ValueError("native audit run subject is not closed")
        if intervention.reference_id != binding.reference_native_object_id:
            raise ValueError("native audit run reference is not closed")
        if (
            audit.native_scene_id != source.scene_id
            or audit.subject_native_object_id != intervention.subject_id
            or audit.reference_native_object_id != intervention.reference_id
            or audit.relation_before != intervention.relation_before.value
            or audit.relation_after != intervention.relation_after.value
        ):
            raise ValueError("native audit run native audit inputs are not closed")
        problem_sha = self.proxy_bundle.semantic_problem.semantic_problem_sha256
        if (
            self.solve_result.semantic_problem_sha256 != problem_sha
            or audit.semantic_problem_sha256 != problem_sha
        ):
            raise ValueError("native audit run problem hash is not closed")
        if (
            self.solve_result.solver_config != self.solver_config
            or audit.solver_config_sha256 != self.solver_config.config_sha256
        ):
            raise ValueError("native audit run solver config is not closed")
        if audit.solve_result_sha256 != self.solve_result.solve_result_sha256:
            raise ValueError("native audit run solve result is not closed")
        if audit.edit_sha256 != self.solve_result.selected_witness.edit.edit_sha256:
            raise ValueError("native audit run selected edit is not closed")
        if audit.proxy_bundle_sha256 != self.proxy_bundle.proxy_bundle_sha256:
            raise ValueError("native audit run proxy bundle is not closed")
        return self

    @model_validator(mode="after")
    def validate_render_policy(self) -> Self:
        source = _strict_legacy(self.source_scene, Scene, "source_scene")
        intervention = _strict_legacy(
            self.intervention, InterventionSpec, "intervention"
        )
        camera = source.camera_by_id(intervention.camera_id)
        if (
            self.native_render_width_px != camera.width
            or self.native_render_height_px != camera.height
            or self.native_seed != source.generation_seed
        ):
            raise ValueError("native audit run render policy is not closed")
        return self

    @model_validator(mode="after")
    def validate_capture_bound_sources(self) -> Self:
        frozen_sha256 = legacy_sha256(self.source_scene)
        fresh_sha256 = legacy_sha256(self.fresh_source_scene)
        correspondence = self.source_correspondence
        audit = self.native_audit
        if (
            correspondence.frozen_source_scene_sha256 != frozen_sha256
            or correspondence.fresh_source_scene_sha256 != fresh_sha256
            or audit.frozen_source_scene_sha256 != frozen_sha256
            or audit.fresh_source_scene_sha256 != fresh_sha256
            or audit.source_correspondence_sha256
            != correspondence.competition_native_capture_source_correspondence_sha256
        ):
            raise ValueError("capture-bound audit run source lineage is not closed")
        return self

    @model_validator(mode="after")
    def validate_patch_bound_lineage(self) -> Self:
        binding = self.proxy_bundle.binding
        audit = self.native_audit
        subjects = tuple(
            item
            for item in self.surface_evidence.subjects
            if item.subject_object_id == self.intervention.subject_id
            and item.subject_surface_evidence_sha256
            == binding.subject_surface_evidence_sha256
        )
        if len(subjects) != 1 or binding.patch_index >= len(subjects[0].patches):
            raise ValueError("patch-bound audit run lineage is not closed")
        subject = subjects[0]
        patch = subject.patches[binding.patch_index]
        if (
            self.source_capture.scene != self.source_scene
            or self.source_capture.source_capture_sha256
            != binding.source_capture_sha256
            or self.surface_evidence.source_capture_sha256
            != self.source_capture.source_capture_sha256
            or self.surface_evidence.surface_evidence_sha256
            != binding.surface_evidence_sha256
            or subject.runtime_identity_sha256 != binding.runtime_identity_sha256
            or subject.placement_sha256 != binding.placement_sha256
            or subject.spawn_map_source_sha256 != binding.spawn_map_source_sha256
            or patch.patch_sha256 != binding.patch_sha256
            or audit.source_capture_sha256 != binding.source_capture_sha256
            or audit.runtime_identity_sha256 != binding.runtime_identity_sha256
            or audit.placement_sha256 != binding.placement_sha256
            or audit.surface_evidence_sha256 != binding.surface_evidence_sha256
            or audit.subject_surface_evidence_sha256
            != binding.subject_surface_evidence_sha256
            or audit.patch_index != binding.patch_index
            or audit.patch_sha256 != binding.patch_sha256
            or audit.spawn_map_source_sha256 != binding.spawn_map_source_sha256
            or audit.fresh_subject_surface_evidence_sha256
            != subject.subject_surface_evidence_sha256
            or audit.fresh_patch_sha256 != patch.patch_sha256
            or self.endpoint_plan_sha256 != audit.endpoint_plan_sha256
        ):
            raise ValueError("patch-bound audit run lineage is not closed")
        return self

    @model_validator(mode="after")
    def validate_camera_replay(self) -> Self:
        audit = self.native_audit
        capture = self.source_capture
        evidence = self.camera_evidence
        if (
            evidence != audit.camera_evidence
            or self.camera_policy != audit.camera_policy
            or evidence.source_id != capture.source.source_id
            or evidence.scene_id != capture.source.scene_id
            or evidence.source_locator_sha256 != capture.source.source_locator_sha256
            or evidence.runtime_identity_sha256
            != canonical_sha256(
                capture.runtime_identity, domain=_CAMERA_RUNTIME_HASH_DOMAIN
            )
            or evidence.policy_sha256 != self.camera_policy.policy_sha256
            or evidence.source_capture_sha256 != capture.source_capture_sha256
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
            or score_competition_native_camera_capture_scene_v2_9_3(
                capture, evidence, self.fresh_source_scene
            )
            != evidence.score
            or audit.fresh_camera != self.fresh_source_scene.camera_by_id("main")
            or audit.camera_replay_observation_sha256
            != self.fresh_before_observation_sha256
        ):
            raise ValueError("camera-aware audit run lineage is not closed")
        verify_competition_native_camera_observation_binding_v2_9_3(
            evidence.requested_pose,
            audit.fresh_observed_pose,
            audit.fresh_observed_native_camera_position,
            self.fresh_source_scene.camera_by_id("main"),
        )
        return self

    @model_validator(mode="after")
    def validate_runtime_collision_authority(self) -> Self:
        delegated = (
            self.proxy_bundle.binding.runtime_collision_delegated_native_object_ids
        )
        if delegated != self.native_audit.runtime_collision_delegated_native_object_ids:
            raise ValueError("runtime collision audit run authority is not closed")
        return self

    @model_validator(mode="after")
    def validate_runtime_pose_authority(self) -> Self:
        if self.runtime_pose_policy != self.native_audit.runtime_pose_policy:
            raise ValueError("runtime pose audit run policy is not closed")
        return self

    @model_validator(mode="after")
    def validate_bbox_visibility_authority(self) -> Self:
        binding = self.proxy_bundle.binding
        audit = self.native_audit
        if (
            binding.visibility_semantics_id != audit.visibility_semantics_id
            or binding.image_area_metric_definition_id
            != audit.image_area_metric_definition_id
            or binding.image_area_metric_definition_version
            != audit.image_area_metric_definition_version
            or binding.image_area_metric_formula != audit.image_area_metric_formula
        ):
            raise ValueError("bbox visibility audit run authority is not closed")
        return self

    @property
    def audit_run_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_RUN_HASH_DOMAIN)

    @property
    def competition_native_audit_run_sha256(self) -> Sha256Digest:
        return self.audit_run_sha256


@dataclass(frozen=True)
class AuditExecution:
    """Process-local same-event observations for the current run."""

    run: AuditRun
    before_observation: AI2ThorObservation
    fresh_before_observation: AI2ThorObservation
    after_observation: AI2ThorObservation
    execution_version: Literal["competition-native-audit-execution:2.9.6"] = (
        "competition-native-audit-execution:2.9.6"
    )

    def __post_init__(self) -> None:
        if (
            type(self.execution_version) is not str
            or self.execution_version != "competition-native-audit-execution:2.9.6"
        ):
            raise TypeError("bbox visibility execution version must be exact")
        if type(self.run) is not AuditRun:
            raise TypeError("bbox visibility execution run must be exact")
        for label, observation in (
            ("before", self.before_observation),
            ("fresh before", self.fresh_before_observation),
            ("after", self.after_observation),
        ):
            if type(observation) is not AI2ThorObservation:
                raise TypeError(f"{label} observation must be exact")
        if self.before_observation.scene != self.run.source_scene:
            raise ValueError("normalized before observation does not close source")
        if self.fresh_before_observation.scene != self.run.fresh_source_scene:
            raise ValueError("fresh before observation does not close camera replay")
        if _observation_asset_identity(self.before_observation) != (
            _observation_asset_identity(self.fresh_before_observation)
        ):
            raise ValueError("normalized before observation changed camera assets")
        if (
            observation_sha256(self.before_observation)
            != self.run.before_observation_sha256
        ):
            raise ValueError("normalized before observation digest does not close run")
        if observation_sha256(self.fresh_before_observation) != (
            self.run.fresh_before_observation_sha256
        ):
            raise ValueError("camera replay observation digest does not close run")
        if legacy_sha256(self.after_observation.scene) != (
            self.run.native_audit.observed_scene_sha256
        ):
            raise ValueError("after observation does not close bbox visibility audit")
        if observation_sha256(self.after_observation) != (
            self.run.native_audit.after_observation_sha256
        ):
            raise ValueError("after observation digest does not close bbox audit")

    @property
    def audit_execution_sha256(self) -> Sha256Digest:
        return canonical_sha256(
            {
                "execution_version": self.execution_version,
                "native_audit_run_sha256": self.run.audit_run_sha256,
                "before_observation_sha256": observation_sha256(
                    self.before_observation
                ),
                "fresh_before_observation_sha256": observation_sha256(
                    self.fresh_before_observation
                ),
                "after_observation_sha256": observation_sha256(self.after_observation),
            },
            domain=_EXECUTION_HASH_DOMAIN,
        )

    @property
    def competition_native_audit_execution_sha256(self) -> Sha256Digest:
        return self.audit_execution_sha256


def _same_exact_rotation(left, right) -> bool:
    left_values = (left.x, left.y, left.z, left.w)
    right_values = (right.x, right.y, right.z, right.w)
    return left_values == right_values or left_values == tuple(
        -value for value in right_values
    )


def _competition_camera_pose_v2_9_3(pose: AdapterPose) -> CameraPose:
    return CameraPose(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _fresh_native_support_matches_capture_v2_9_3(
    facts: SourceCaptureFacts,
    capture: CompetitionNativeSourceCaptureV2_9,
) -> bool:
    observed = tuple(sorted(facts.support_facts, key=lambda item: item.object_id))
    expected = capture.support_facts

    def support_key(item):
        return (
            item.scene_id,
            item.object_id,
            item.object_name,
            item.native_object_id,
            item.raw_parent_object_ids,
            item.structural_parent_object_ids,
            item.domain_parent_object_ids,
            item.support_kind,
            item.support_object_id,
            item.floor_object_id,
        )

    return tuple(support_key(item) for item in observed) == tuple(
        support_key(item) for item in expected
    )


def _rebuild_fresh_subject_surface_evidence_v2_9_2(
    capture: CompetitionNativeSourceCaptureV2_9,
    source_evidence: SourceSurfaceEvidence,
    subject_object_id: str,
    spawn_map,
) -> SubjectSurfaceEvidence:
    """Rebuild exactly one subject row without querying unrelated subjects."""

    checked_evidence = verify_source_surface_evidence(
        capture,
        source_evidence,
    )
    subjects = tuple(
        item
        for item in checked_evidence.subjects
        if item.subject_object_id == subject_object_id
    )
    if len(subjects) != 1:
        raise ValueError("patch-bound subject surface evidence is not unique")
    expected = subjects[0]
    runtime = CompetitionNativeRuntimeIdentityV2_9(**asdict(spawn_map.runtime_identity))
    capture_runtime = CompetitionNativeRuntimeIdentityV2_9.model_validate(
        capture.runtime_identity.model_dump(mode="python"),
        strict=True,
    )
    if runtime != capture_runtime:
        raise ValueError("fresh patch runtime identity changed")
    patches = tuple(
        _build_patch(index, patch)
        for index, patch in enumerate(spawn_map.surface_patches)
    )
    payload = _subject_payload(
        subject_object_id=spawn_map.subject_object_id,
        support_object_id=spawn_map.support_object_id,
        native_subject_object_id=spawn_map.native_subject_object_id,
        native_support_object_id=spawn_map.native_support_object_id,
        runtime_identity_sha256=expected.runtime_identity_sha256,
        scene_sha256=spawn_map.scene_sha256,
        positions_sha256=spawn_map.positions_sha256,
        spawn_map_source_sha256=spawn_map.source_sha256,
        placement_sha256=expected.placement_sha256,
        source_capture_sha256=capture.source_capture_sha256,
        patches=patches,
    )
    return SubjectSurfaceEvidence(
        **payload,
        subject_surface_evidence_sha256=canonical_sha256(
            payload,
            domain=_SUBJECT_EVIDENCE_HASH_DOMAIN,
        ),
    )


def _runtime_commanded_scene_v2_9_6(
    run: AuditRun,
) -> Scene:
    """Rebuild the commanded subject translation from frozen run authority."""

    subject_id = run.intervention.subject_id
    original = run.fresh_source_scene.object_by_id(subject_id)
    commanded = run.native_audit.commanded_position
    delta = (
        commanded.x - original.position.x,
        commanded.y - original.position.y,
        commanded.z - original.position.z,
    )
    moved = original.model_copy(
        update={
            "position": original.position.model_copy(
                update={"x": commanded.x, "y": commanded.y, "z": commanded.z}
            ),
            "obb": original.obb.model_copy(
                update={
                    "center": original.obb.center.model_copy(
                        update={
                            "x": original.obb.center.x + delta[0],
                            "y": original.obb.center.y + delta[1],
                            "z": original.obb.center.z + delta[2],
                        }
                    )
                }
            ),
        }
    )
    return run.fresh_source_scene.model_copy(
        update={
            "objects": tuple(
                moved if item.object_id == subject_id else item
                for item in run.fresh_source_scene.objects
            )
        }
    )


def _unique_object_by_name(scene: Scene, name: str, label: str):
    matches = tuple(item for item in scene.objects if item.name == name)
    if len(matches) != 1:
        raise EndpointAuditRejected(
            "native_precondition",
            (f"{label}_name_match_count:{len(matches)}",),
        )
    return matches[0]


def _require_nonempty_exact_str(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty exact string")


def _observation_with_scene(
    observation: AI2ThorObservation,
    scene: Scene,
) -> AI2ThorObservation:
    return AI2ThorObservation.create(
        scene=scene,
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=observation.instance_pixel_counts,
        is_settled=observation.is_settled,
    )


def _observation_asset_identity(
    observation: AI2ThorObservation,
) -> tuple[object, ...]:
    return (
        observation.rgb_png,
        observation.depth_npy,
        observation.instance_png,
        observation.pointcloud_ply,
        observation.instance_pixel_counts,
        observation.is_settled,
    )


def _execute_audit(
    adapter: AI2ThorAdapter,
    request: BatchRequest,
    *,
    request_lineage: RequestLineage,
    fresh_transitions=None,
) -> AuditExecution:
    """Execute the single current audit route with no version dispatch."""

    if not isinstance(adapter, EnvironmentAdapter):
        raise TypeError("adapter must satisfy the EnvironmentAdapter protocol")
    if type(request) is not BatchRequest:
        raise TypeError("batch request must be exact")
    if type(request_lineage) is not RequestLineage:
        raise TypeError("RequestLineage must be exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_request = BatchRequest.model_validate(
            request.model_dump(mode="python", warnings="error"), strict=True
        )
        lineage = RequestLineage.model_validate(
            request_lineage.model_dump(mode="python", warnings="error"), strict=True
        )
    if lineage.batch_request_sha256 != request_binding_sha256(checked_request):
        raise ValueError("camera+patch runtime request binding mismatch")
    if lineage.endpoint_plan.endpoint_workspace != checked_request.endpoint_workspace:
        raise ValueError("camera+patch runtime endpoint workspace mismatch")
    settled_camera_policy = build_competition_native_settled_camera_policy_v2_9_10()
    if lineage.camera_policy != settled_camera_policy:
        raise ValueError("current audit requires the settled camera policy")

    scene_id = checked_request.scene_id
    max_settlement_steps = checked_request.max_settlement_steps
    try:
        captured_source = adapter.capture_source(
            CaptureRequest(scene_id=scene_id, camera_id="main")
        )
    except AdapterActionRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"load_scene_rejected:{error.reason}",)
        ) from error
    except AdapterReturnRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"load_scene_return_rejected:{error.reason}",)
        ) from error
    except AdapterOperationError:
        raise
    try:
        initial_facts = adapter.observe_source(
            captured_source,
            options=SourceCaptureOptions(
                max_settlement_steps=max_settlement_steps,
                floor_clearance_m=0.1,
                navigation_agent_radius_m=0.2,
                navigation_clearance_m=0.0,
            ),
            settle=True,
        )
    except AdapterActionRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"settlement_action_rejected:{error.reason}",)
        ) from error
    except AdapterSettlementTimeout as error:
        raise EndpointAuditRejected(
            "native_precondition", ("scene_not_settled",)
        ) from error
    except AdapterReturnRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"settlement_return_rejected:{error.reason}",)
        ) from error
    except AdapterOperationError:
        raise

    fresh_source = initial_facts.scene
    evidence = lineage.camera_evidence
    requested_pose = AdapterPose(
        position=AdapterPosition(
            x=evidence.requested_pose.x,
            y=evidence.requested_pose.y,
            z=evidence.requested_pose.z,
        ),
        yaw_degrees=evidence.requested_pose.yaw_degrees,
        horizon_degrees=evidence.requested_pose.horizon_degrees,
        standing=evidence.requested_pose.standing,
    )
    try:
        handle = adapter.pause_camera_observations(
            initial_facts,
            settle_after_resume=True,
        )
        try:
            adapter.apply_camera_pose(
                initial_facts,
                requested_pose,
                handle=handle,
                source_scene=fresh_source,
                reset_from_source=False,
                max_settlement_steps=max_settlement_steps,
            )
        finally:
            adapter.resume_camera_observations(handle)
        camera_settlement = adapter.settle_camera_pose(
            initial_facts,
            requested_pose,
            source_scene=fresh_source,
            max_settlement_steps=max_settlement_steps,
        )
        camera_application = camera_settlement.application
    except AdapterActionRejected as error:
        raise EndpointAuditRejected(
            "native_precondition",
            (f"camera_replay_action_rejected:{error.reason}",),
        ) from error
    except AdapterReturnRejected as error:
        raise EndpointAuditRejected(
            "native_precondition",
            (f"camera_replay_return_rejected:{error.reason}",),
        ) from error
    except AdapterOperationError as error:
        raise EndpointAuditRejected(
            "native_precondition",
            (f"camera_replay_failed:{type(error).__name__}:{error}",),
        ) from error

    fresh_source = camera_application.observed_scene
    fresh_capture = CapturedSource(
        request=captured_source.request,
        scene=fresh_source,
        binding=captured_source.binding,
    )
    fresh_facts = SourceCaptureFacts(
        source=fresh_capture,
        binding=fresh_capture.binding,
        scene=fresh_source,
        runtime_identity=initial_facts.runtime_identity,
        observation=camera_application.observation,
        support_facts=initial_facts.support_facts,
        floor_envelope=initial_facts.floor_envelope,
        floor_position_regions=initial_facts.floor_position_regions,
        reachable_positions=initial_facts.reachable_positions,
        current_pose=camera_application.observed_pose,
        settlement_pass_steps=camera_settlement.settlement_pass_steps,
    )
    fresh_before_observation = camera_application.observation
    fresh_observed_pose = _competition_camera_pose_v2_9_3(
        camera_application.observed_pose
    )
    fresh_observed_native_camera_position = (
        camera_application.observed_camera_position.x,
        camera_application.observed_camera_position.y,
        camera_application.observed_camera_position.z,
    )
    try:
        fresh_runtime = CompetitionNativeRuntimeIdentityV2_9(
            **asdict(fresh_facts.runtime_identity)
        )
        expected_camera_residuals = (
            dist(
                (
                    requested_pose.position.x,
                    requested_pose.position.y,
                    requested_pose.position.z,
                ),
                (
                    camera_application.observed_pose.position.x,
                    camera_application.observed_pose.position.y,
                    camera_application.observed_pose.position.z,
                ),
            ),
            abs(
                (
                    camera_application.observed_pose.yaw_degrees
                    - requested_pose.yaw_degrees
                    + 180.0
                )
                % 360.0
                - 180.0
            ),
            abs(
                camera_application.observed_pose.horizon_degrees
                - requested_pose.horizon_degrees
            ),
        )
        if camera_application.requested_pose != requested_pose:
            raise ValueError("camera requested pose changed")
        if fresh_facts.current_pose != camera_application.observed_pose:
            raise ValueError("camera current pose changed")
        if (
            camera_application.position_residual_m,
            camera_application.yaw_residual_degrees,
            camera_application.horizon_residual_degrees,
        ) != expected_camera_residuals:
            raise ValueError("camera application residuals changed")
        if camera_application.observation.scene != fresh_source:
            raise ValueError("camera observation scene changed")
        if camera_application.observation.is_settled is not True:
            raise ValueError("camera observation is not settled")
        if fresh_runtime != lineage.source_capture.runtime_identity:
            raise ValueError("camera replay runtime changed")
        if not _fresh_native_support_matches_capture_v2_9_3(
            fresh_facts, lineage.source_capture
        ):
            raise ValueError("camera replay support facts changed")
        if (
            score_competition_native_camera_capture_scene_v2_9_3(
                lineage.source_capture, evidence, fresh_source
            )
            != evidence.score
        ):
            raise ValueError("camera replay score changed")
        verify_competition_native_camera_observation_binding_v2_9_3(
            evidence.requested_pose,
            fresh_observed_pose,
            fresh_observed_native_camera_position,
            fresh_source.camera_by_id("main"),
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"camera_replay_closure:{error}",)
        ) from error

    source = lineage.source_capture.scene
    try:
        source_correspondence = validate_capture_source(source, fresh_source)
    except (TypeError, ValueError) as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"capture_source_correspondence:{error}",)
        ) from error
    subject = _unique_object_by_name(source, checked_request.subject_name, "subject")
    reference = _unique_object_by_name(
        source, checked_request.reference_name, "reference"
    )
    fresh_subject = fresh_source.object_by_id(subject.object_id)
    if not (
        _same_exact_rotation(fresh_subject.rotation, subject.rotation)
        and _same_exact_rotation(fresh_subject.obb.rotation, subject.obb.rotation)
    ):
        raise EndpointAuditRejected(
            "native_precondition", ("capture_source_subject_yaw_mismatch",)
        )
    if subject.support_object_id is None:
        raise EndpointAuditRejected("native_precondition", ("subject_has_no_support",))
    spec = InterventionSpec(
        subject_id=subject.object_id,
        reference_id=reference.object_id,
        relation_before=checked_request.relation_before,
        relation_after=checked_request.relation_after,
        camera_id="main",
    )
    subjects = tuple(
        item
        for item in lineage.surface_evidence.subjects
        if item.subject_object_id == subject.object_id
    )
    placements = tuple(
        item
        for item in lineage.source_capture.placement_facts
        if item.object_id == subject.object_id
    )
    if len(subjects) != 1:
        raise EndpointAuditRejected(
            "native_precondition", ("patch_subject_surface_evidence_mismatch",)
        )
    if len(placements) != 1:
        raise EndpointAuditRejected(
            "native_precondition", ("patch_subject_placement_mismatch",)
        )
    bundle = build_proxy_bundle(
        source,
        spec,
        workspace=checked_request.endpoint_workspace,
        source_surface_evidence=lineage.surface_evidence,
        subject_surface_evidence=subjects[0],
        placement=SubjectPlacementFact(
            source_capture=lineage.source_capture,
            subject_placement=placements[0],
        ),
        collision_delegation=CollisionDelegation(
            case_id=checked_request.case_id,
            patch_index=lineage.patch_index,
        ),
    )
    binding = bundle.binding
    if (
        binding.source_capture_sha256 != lineage.source_capture_sha256
        or binding.runtime_identity_sha256 != lineage.runtime_identity_sha256
        or binding.placement_sha256 != lineage.placement_sha256
        or binding.surface_evidence_sha256 != lineage.surface_evidence_sha256
        or binding.subject_surface_evidence_sha256
        != lineage.subject_surface_evidence_sha256
        or binding.patch_index != lineage.patch_index
        or binding.patch_sha256 != lineage.patch_sha256
        or binding.spawn_map_source_sha256 != lineage.spawn_map_source_sha256
        or bundle.semantic_problem.semantic_problem_sha256
        != lineage.semantic_problem_sha256
        or bundle.proxy_bundle_sha256 != lineage.proxy_bundle_sha256
        or binding.runtime_collision_delegated_native_object_ids
        != lineage.runtime_collision_delegated_native_object_ids
    ):
        raise EndpointAuditRejected(
            "native_precondition", ("patch_proxy_lineage_mismatch",)
        )

    solved = solve_minimum_cost(bundle.semantic_problem, checked_request.solver_config)
    if type(solved.result) is not ContinuousYawCertifiedSuccessResultV2_9:
        findings = solved.finding_codes or getattr(solved.result, "finding_codes", ())
        raise EndpointAuditRejected(
            "solve", findings or ("competition_native_proxy_not_certified_success",)
        )
    if solved.result.solve_result_sha256 != lineage.solve_result_sha256:
        raise EndpointAuditRejected(
            "native_precondition", ("solve_result_lineage_mismatch",)
        )

    try:
        (spawn_map,) = adapter.capture_spawn_maps(
            fresh_facts,
            subject_object_ids=(subject.object_id,),
        )
    except AdapterActionRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"spawn_map_action_rejected:{error.reason}",)
        ) from error
    except AdapterReturnRejected as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"spawn_map_return_rejected:{error.reason}",)
        ) from error
    except AdapterOperationError:
        raise
    capture_bound_spawn_map = capture_bound_adapter_spawn_map(
        spawn_map, fresh_scene=fresh_source, frozen_scene=source
    )
    if capture_bound_spawn_map.source_sha256 != lineage.spawn_map_source_sha256:
        raise EndpointAuditRejected(
            "native_precondition", ("spawn_map_source_lineage_mismatch",)
        )
    try:
        fresh_subject_evidence = _rebuild_fresh_subject_surface_evidence_v2_9_2(
            lineage.source_capture,
            lineage.surface_evidence,
            subject.object_id,
            capture_bound_spawn_map,
        )
    except (TypeError, ValueError) as error:
        raise EndpointAuditRejected(
            "native_precondition", (f"fresh_patch_evidence:{error}",)
        ) from error
    if fresh_subject_evidence != subjects[0]:
        raise EndpointAuditRejected(
            "native_precondition", ("fresh_patch_evidence_mismatch",)
        )

    executed = execute_endpoint(
        adapter,
        source,
        fresh_capture,
        spec,
        bundle,
        checked_request.solver_config,
        solved.result,
        spawn_map,
        capture_bound_spawn_map_source_sha256=capture_bound_spawn_map.source_sha256,
        max_post_edit_pass_steps=max_settlement_steps,
        runtime_pose_policy=lineage.runtime_pose_policy,
        source_correspondence=source_correspondence,
    )
    after_observation = executed.application.observation
    if executed.audit is not None or executed.capture_bound_audit_payload is None:
        raise RuntimeError("runtime endpoint audit payload is incomplete")

    audit = EndpointAudit(
        **executed.capture_bound_audit_payload,
        source_capture_sha256=lineage.source_capture_sha256,
        runtime_identity_sha256=lineage.runtime_identity_sha256,
        placement_sha256=lineage.placement_sha256,
        surface_evidence_sha256=lineage.surface_evidence_sha256,
        subject_surface_evidence_sha256=lineage.subject_surface_evidence_sha256,
        patch_index=lineage.patch_index,
        patch_sha256=lineage.patch_sha256,
        fresh_subject_surface_evidence_sha256=(
            fresh_subject_evidence.subject_surface_evidence_sha256
        ),
        fresh_patch_sha256=(
            fresh_subject_evidence.patches[lineage.patch_index].patch_sha256
        ),
        endpoint_plan_sha256=lineage.endpoint_plan_sha256,
        camera_evidence=lineage.camera_evidence,
        camera_policy=lineage.camera_policy,
        fresh_observed_pose=fresh_observed_pose,
        fresh_observed_native_camera_position=fresh_observed_native_camera_position,
        fresh_camera=fresh_source.camera_by_id("main"),
        camera_replay_observation_sha256=observation_sha256(fresh_before_observation),
        runtime_collision_delegated_native_object_ids=(
            lineage.runtime_collision_delegated_native_object_ids
        ),
        visibility_semantics_id=lineage.visibility_semantics_id,
        image_area_metric_definition_id=lineage.image_area_metric_definition_id,
        image_area_metric_definition_version=(
            lineage.image_area_metric_definition_version
        ),
        image_area_metric_formula=lineage.image_area_metric_formula,
    )
    normalized_before = _observation_with_scene(fresh_before_observation, source)
    run = AuditRun(
        case_id=checked_request.case_id,
        source_scene=source,
        intervention=spec,
        endpoint_workspace=checked_request.endpoint_workspace,
        proxy_bundle=bundle,
        solver_config=checked_request.solver_config,
        solve_result=solved.result,
        before_observation_sha256=observation_sha256(normalized_before),
        native_audit=audit,
        native_render_width_px=source.camera_by_id("main").width,
        native_render_height_px=source.camera_by_id("main").height,
        native_seed=source.generation_seed,
        max_settlement_steps=max_settlement_steps,
        fresh_source_scene=fresh_source,
        fresh_before_observation_sha256=observation_sha256(fresh_before_observation),
        source_correspondence=source_correspondence,
        source_capture=lineage.source_capture,
        surface_evidence=lineage.surface_evidence,
        endpoint_plan_sha256=lineage.endpoint_plan_sha256,
        camera_evidence=lineage.camera_evidence,
        camera_policy=lineage.camera_policy,
        runtime_pose_policy=lineage.runtime_pose_policy,
    )
    result = AuditExecution(
        run=run,
        before_observation=normalized_before,
        fresh_before_observation=fresh_before_observation,
        after_observation=after_observation,
    )
    if fresh_transitions is not None:
        transition = fresh_transitions.executed(
            checked_request.request_id,
            executed.application.application.edit,
            executed.application,
        )
        if type(transition) is not workflow_contracts.ExecutedEdit:
            raise TypeError("audit transition execution must be exact")
        fresh_transitions.verify(checked_request.request_id)
    return result


def execute_audit(
    adapter: AI2ThorAdapter,
    request: BatchRequest,
    *,
    request_lineage: RequestLineage,
) -> AuditExecution:
    return _execute_audit(adapter, request, request_lineage=request_lineage)


def _execute_audit_with_transitions(
    adapter: AI2ThorAdapter,
    request: BatchRequest,
    *,
    request_lineage: RequestLineage,
    fresh_transitions,
) -> AuditExecution:
    return _execute_audit(
        adapter,
        request,
        request_lineage=request_lineage,
        fresh_transitions=fresh_transitions,
    )


def verify_audit_run(run: AuditRun, observed_after_scene: Scene) -> AuditRun:
    """Fresh-rebuild the current proxy and independently verify the after scene."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(run) is not AuditRun:
            raise TypeError("submitted_run must be exact AuditRun")
        if type(observed_after_scene) is not Scene:
            raise TypeError("observed_after_scene must be an exact Scene")
        checked = AuditRun.model_validate(
            run.model_dump(mode="python", warnings="error"), strict=True
        )
        after = Scene.model_validate(
            observed_after_scene.model_dump(mode="python"), strict=True
        )
        subjects = tuple(
            item
            for item in checked.surface_evidence.subjects
            if item.subject_object_id == checked.intervention.subject_id
        )
        placements = tuple(
            item
            for item in checked.source_capture.placement_facts
            if item.object_id == checked.intervention.subject_id
        )
        if len(subjects) != 1 or len(placements) != 1:
            raise ValueError("patch-bound audit run subject evidence changed")
        rebuilt = build_proxy_bundle(
            checked.source_scene,
            checked.intervention,
            workspace=checked.endpoint_workspace,
            source_surface_evidence=checked.surface_evidence,
            subject_surface_evidence=subjects[0],
            placement=SubjectPlacementFact(
                source_capture=checked.source_capture,
                subject_placement=placements[0],
            ),
            collision_delegation=CollisionDelegation(
                case_id=checked.case_id,
                patch_index=checked.proxy_bundle.binding.patch_index,
            ),
        )
        if rebuilt != checked.proxy_bundle:
            raise ValueError("native audit run proxy replay mismatch")
        verified = verify_solve_result(
            rebuilt.semantic_problem,
            checked.solver_config,
            checked.solve_result,
        )
        if verified.kind is not ContinuousYawSolveVerificationKindV2.VERIFIED:
            raise ValueError("native audit run solve replay mismatch")
        if legacy_sha256(after) != checked.native_audit.observed_scene_sha256:
            raise ValueError("native audit run after scene hash mismatch")
        observed_subject = after.object_by_id(checked.intervention.subject_id)
        source_subject = checked.source_scene.object_by_id(
            checked.intervention.subject_id
        )
        edit = checked.solve_result.selected_witness.edit.translation_xy_m
        commanded = checked.native_audit.commanded_position
        expected_position = checked.native_audit.observed_position
        if (
            commanded.x != source_subject.position.x + edit.x
            or commanded.y != source_subject.position.y + edit.y
            or commanded.z
            != checked.fresh_source_scene.object_by_id(
                checked.intervention.subject_id
            ).position.z
        ):
            raise ValueError("native audit run commanded endpoint mismatch")
        if (
            observed_subject.position.x != expected_position.x
            or observed_subject.position.y != expected_position.y
            or observed_subject.position.z != expected_position.z
        ):
            raise ValueError("native audit run observed subject position mismatch")
        residual = dist(
            (commanded.x, commanded.y, commanded.z),
            (expected_position.x, expected_position.y, expected_position.z),
        )
        if residual != checked.native_audit.position_residual_m:
            raise ValueError("native audit run position residual mismatch")
        correspondence = validate_capture_source(
            checked.source_scene, checked.fresh_source_scene
        )
        if correspondence != checked.source_correspondence:
            raise ValueError("capture-bound source correspondence replay mismatch")
        commanded_scene = _runtime_commanded_scene_v2_9_6(checked)
        fresh_normalized = _normalized_runtime_pose_observed_scene_v2_9_5(
            commanded_scene,
            after,
            checked.intervention.subject_id,
            checked.runtime_pose_policy,
        )
        normalized = _project_runtime_pose_observed_scene_to_frozen_source_v2_9_5(
            checked.source_scene,
            fresh_normalized,
            checked.intervention.subject_id,
        )
        commanded_subject = commanded_scene.object_by_id(
            checked.intervention.subject_id
        )
        if (
            _quaternion_angle_residual_deg(
                observed_subject.rotation,
                commanded_subject.rotation,
                "runtime subject rotation",
            )
            != checked.native_audit.subject_rotation_residual_degrees
            or _obb_corner_hausdorff_residual_m(
                observed_subject.obb,
                commanded_subject.obb,
                "runtime subject OBB",
            )
            != checked.native_audit.subject_obb_corner_residual_m
        ):
            raise ValueError("native audit run runtime pose residual mismatch")
        semantic = _verify_minimum_cost_with_proxy_collision_authority(
            checked.proxy_bundle,
            checked.source_scene,
            normalized,
            checked.intervention,
            runtime_pose_subject_object_id=checked.intervention.subject_id,
        )
        audit = checked.native_audit
        if (
            semantic.status is not SolverStatus.SUCCESS
            or semantic.quality not in {QualityTier.PURE, QualityTier.LOW_LEAKAGE}
            or semantic.errors
            or semantic.quality.value != audit.verification_quality
            or semantic.relation_damage_count != audit.relation_damage_count
            or semantic.relation_damage_items != audit.relation_damage_items
        ):
            raise ValueError("native audit run independent semantic replay mismatch")
        return checked


__all__ = (
    "AuditExecution",
    "AuditRun",
    "EndpointAudit",
    "EndpointAuditRejected",
    "execute_audit",
    "execute_endpoint",
    "observation_sha256",
    "verify_audit_run",
)
