"""Current-only native audit orchestration and replay verification."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from math import dist
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import (
    AI2ThorAdapter,
    AI2ThorAgentPose,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorObservation,
    AI2ThorSettlementTimeout,
    capture_bound_ai2thor_receptacle_spawn_map,
)
from spatialcf.core.v2.continuous_yaw_solve_verifier_v2_9 import (
    verify_continuous_yaw_solve_result_v2_9,
)
from spatialcf.core.v2.continuous_yaw_solver_v2_9 import (
    solve_continuous_yaw_minimum_cost_v2_9,
)
from spatialcf.domain.enums import QualityTier, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.domain.v2.base import CanonicalId, Sha256Digest, V2Model
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawSolveVerificationKindV2,
)
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    SourceCameraEvidence,
    build_competition_native_settled_camera_policy_v2_9_10,
    verify_competition_native_camera_observation_binding_v2_9_3,
)
from spatialcf.generation._internal.evidence.camera import (
    CompetitionNativeCameraPoseV2_9_3 as CameraPose,
)
from spatialcf.generation._internal.evidence.surface import (
    _SUBJECT_EVIDENCE_HASH_DOMAIN,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    _build_patch,
    _subject_payload,
    verify_source_surface_evidence,
)
from spatialcf.generation._internal.execution.audit import (
    EndpointAudit,
    EndpointAuditRejected,
    _native_action_rejection,
    _native_return_rejection,
    _normalized_runtime_pose_observed_scene_v2_9_5,
    _obb_corner_hausdorff_residual_m,
    _project_runtime_pose_observed_scene_to_frozen_source_v2_9_5,
    _quaternion_angle_residual_deg,
    _verify_minimum_cost_with_proxy_collision_authority,
    execute_endpoint,
    observation_sha256,
)
from spatialcf.generation._internal.execution.correspondence import (
    CaptureSourceCorrespondence,
    RequestLineage,
    legacy_sha256,
    request_binding_sha256,
    validate_capture_source,
)
from spatialcf.generation._internal.planning.campaign import (
    BatchRequest,
    RuntimePosePolicy,
)
from spatialcf.generation._internal.planning.models import (
    CollisionDelegation,
    EndpointWorkspace,
    ProxyBundle,
    SubjectPlacementFact,
)
from spatialcf.generation._internal.planning.proxy import build_proxy_bundle
from spatialcf.generation.capture.compiler import (
    score_competition_native_camera_capture_scene_v2_9_3,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceCaptureV2_9,
)

_RUN_HASH_DOMAIN = "spatialcf.competition-native-audit-run.v2.9.7"
_EXECUTION_HASH_DOMAIN = "spatialcf.competition-native-audit-execution.v2.9.6"
_CAMERA_RUNTIME_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_MAX_SETTLEMENT_STEPS = 600


class AuditRun(V2Model):
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
            != canonical_sha256_v2(
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
        return canonical_sha256_v2(self, domain=_RUN_HASH_DOMAIN)

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
        return canonical_sha256_v2(
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


def _competition_camera_pose_v2_9_3(pose) -> CameraPose:
    return CameraPose(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _fresh_native_support_matches_capture_v2_9_3(
    adapter: AI2ThorAdapter,
    scene: Scene,
    capture: CompetitionNativeSourceCaptureV2_9,
) -> bool:
    observed = tuple(
        sorted(adapter.native_support_facts(scene), key=lambda item: item.object_id)
    )
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
            item.support_kind.value,
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
        subject_surface_evidence_sha256=canonical_sha256_v2(
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


def _latest_native_event_or_none(adapter: AI2ThorAdapter, scene_id: str) -> object:
    try:
        return adapter.latest_native_event(scene_id)
    except (KeyError, RuntimeError):
        return None


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
        is_scene_at_rest=observation.is_scene_at_rest,
    )


def _observation_asset_identity(observation: AI2ThorObservation) -> tuple[object, ...]:
    return (
        observation.rgb_png,
        observation.depth_npy,
        observation.instance_png,
        observation.pointcloud_ply,
        tuple(sorted(observation.instance_pixel_counts.items())),
        observation.is_scene_at_rest,
    )


def _strict_legacy(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def execute_audit(
    adapter: AI2ThorAdapter,
    request: BatchRequest,
    *,
    request_lineage: RequestLineage,
) -> AuditExecution:
    """Execute the single current audit route with no version dispatch."""

    if type(adapter) is not AI2ThorAdapter:
        raise TypeError("adapter must be an exact AI2ThorAdapter")
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
    before_load_event = _latest_native_event_or_none(adapter, scene_id)
    try:
        loaded_scene = adapter.load_scene(scene_id)
    except (RuntimeError, AI2ThorNativeReturnError) as error:
        rejection = _native_action_rejection(
            adapter, scene_id, before_load_event, error
        )
        if rejection is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"load_scene_rejected:{rejection}",)
            ) from error
        returned = _native_return_rejection(adapter, scene_id, before_load_event, error)
        if returned is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"load_scene_return_rejected:{returned}",)
            ) from error
        raise

    before_settlement_event = adapter.latest_native_event(scene_id)
    try:
        settlement = adapter.settle_scene_observed(
            loaded_scene, max_pass_steps=max_settlement_steps
        )
    except (RuntimeError, AI2ThorNativeReturnError) as error:
        rejection = _native_action_rejection(
            adapter, scene_id, before_settlement_event, error
        )
        if rejection is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"settlement_action_rejected:{rejection}",)
            ) from error
        if isinstance(error, AI2ThorSettlementTimeout):
            raise EndpointAuditRejected(
                "native_precondition", ("scene_not_settled",)
            ) from error
        returned = _native_return_rejection(
            adapter,
            scene_id,
            before_settlement_event,
            error,
            allow_existing_success_event=True,
        )
        if returned is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"settlement_return_rejected:{returned}",)
            ) from error
        raise

    fresh_source = settlement.observed_scene
    evidence = lineage.camera_evidence
    requested_pose = AI2ThorAgentPose(
        position=AI2ThorNativePosition(
            x=evidence.requested_pose.x,
            y=evidence.requested_pose.y,
            z=evidence.requested_pose.z,
        ),
        yaw_degrees=evidence.requested_pose.yaw_degrees,
        horizon_degrees=evidence.requested_pose.horizon_degrees,
        standing=evidence.requested_pose.standing,
    )
    before_camera_event = adapter.latest_native_event(scene_id)
    try:
        with adapter.paused_camera_observations_for_settlement(
            fresh_source
        ) as paused_source:
            camera_application = adapter.apply_camera_pose_observed(
                paused_source, requested_pose
            )
        camera_application = adapter.settle_current_camera_pose_observed(
            fresh_source,
            requested_pose,
            max_pass_steps=max_settlement_steps,
        ).application
    except (RuntimeError, AI2ThorNativeReturnError, TypeError, ValueError) as error:
        rejection = _native_action_rejection(
            adapter, scene_id, before_camera_event, error
        )
        if rejection is not None:
            raise EndpointAuditRejected(
                "native_precondition",
                (f"camera_replay_action_rejected:{rejection}",),
            ) from error
        returned = _native_return_rejection(
            adapter, scene_id, before_camera_event, error
        )
        if returned is not None:
            raise EndpointAuditRejected(
                "native_precondition",
                (f"camera_replay_return_rejected:{returned}",),
            ) from error
        raise EndpointAuditRejected(
            "native_precondition",
            (f"camera_replay_failed:{type(error).__name__}:{error}",),
        ) from error

    fresh_source = camera_application.observed_scene
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
        current_pose = adapter.current_agent_pose(fresh_source)
        fresh_runtime = CompetitionNativeRuntimeIdentityV2_9(
            **asdict(adapter.runtime_identity())
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
        if current_pose != camera_application.observed_pose:
            raise ValueError("camera current pose changed")
        if (
            camera_application.position_residual_m,
            camera_application.yaw_residual_degrees,
            camera_application.horizon_residual_degrees,
        ) != expected_camera_residuals:
            raise ValueError("camera application residuals changed")
        if camera_application.observation.scene != fresh_source:
            raise ValueError("camera observation scene changed")
        if camera_application.observation.is_scene_at_rest is not True:
            raise ValueError("camera observation is not settled")
        if fresh_runtime != lineage.source_capture.runtime_identity:
            raise ValueError("camera replay runtime changed")
        if not _fresh_native_support_matches_capture_v2_9_3(
            adapter, fresh_source, lineage.source_capture
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

    solved = solve_continuous_yaw_minimum_cost_v2_9(
        bundle.semantic_problem, checked_request.solver_config
    )
    if type(solved.result) is not ContinuousYawCertifiedSuccessResultV2_9:
        findings = solved.finding_codes or getattr(solved.result, "finding_codes", ())
        raise EndpointAuditRejected(
            "solve", findings or ("competition_native_proxy_not_certified_success",)
        )
    if solved.result.solve_result_sha256 != lineage.solve_result_sha256:
        raise EndpointAuditRejected(
            "native_precondition", ("solve_result_lineage_mismatch",)
        )

    before_spawn_query_event = adapter.latest_native_event(fresh_source.scene_id)
    try:
        spawn_map = adapter.receptacle_spawn_map(
            fresh_source, subject_object_id=subject.object_id
        )
    except (RuntimeError, TypeError, ValueError) as error:
        rejection = _native_action_rejection(
            adapter, fresh_source.scene_id, before_spawn_query_event, error
        )
        if rejection is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"spawn_map_action_rejected:{rejection}",)
            ) from error
        returned = _native_return_rejection(
            adapter, fresh_source.scene_id, before_spawn_query_event, error
        )
        if returned is not None:
            raise EndpointAuditRejected(
                "native_precondition", (f"spawn_map_return_rejected:{returned}",)
            ) from error
        raise
    capture_bound_spawn_map = capture_bound_ai2thor_receptacle_spawn_map(
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
        fresh_source,
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
    after_observation = adapter.capture_current_observation(
        executed.application.observed_scene
    )
    if after_observation != executed.application.observation:
        raise EndpointAuditRejected(
            "native_verification", ("fresh_after_observation_mismatch",)
        )
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
    return AuditExecution(
        run=run,
        before_observation=normalized_before,
        fresh_before_observation=fresh_before_observation,
        after_observation=after_observation,
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
        verified = verify_continuous_yaw_solve_result_v2_9(
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
    "execute_audit",
    "verify_audit_run",
)
