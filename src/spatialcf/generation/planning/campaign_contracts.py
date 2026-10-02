"""Explicit source campaign contracts owner."""

from __future__ import annotations

import hashlib

import os

from collections import (
    Counter,
)

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from spatialcf.core.solver import (
    solve_minimum_cost,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.request import (
    InterventionSpec,
    Relation,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.domain.solver import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
)

from spatialcf.generation.capture.compiler import (
    verify_competition_native_camera_evidence_capture_v2_9_3,
)

from spatialcf.generation.capture.models import (
    CameraPolicy,
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSupportKindV2_9,
    SourceCameraEvidence,
    SourceSurfaceEvidence,
    verify_source_surface_evidence,
)

from spatialcf.generation.capture.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)

from spatialcf.generation.planning.endpoint import (
    endpoint_workspace_within_position_region,
)

from spatialcf.generation.planning.models import (
    CollisionDelegation,
    EndpointPlan,
    EndpointWorkspace,
    ProxyBundle,
    SourceViewGuard,
    SubjectPlacementFact,
)

from spatialcf.generation.planning.problem import (
    build_proxy_bundle,
    default_planning_workspace,
)

from spatialcf.generation.planning.view_guard import (
    evaluate_source_view_guard,
)

from spatialcf.verification.filesystem import (
    DirectoryIdentity,
)

from spatialcf.verification.integrity import (
    competition_legacy_sha256,
)


_POLICY_DOMAIN = "spatialcf.competition-native-source-policy.v2.9.13"


_PLAN_DOMAIN = "spatialcf.competition-native-source-plan.v2.9.10"


_TARGET_LEDGER_DOMAIN = (
    "spatialcf.competition-native-source-target-reachability-ledger.v2.9.8"
)


_RUNTIME_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9"


_ACCEPTED_ROSTER_DOMAIN = (
    "spatialcf.competition-native-accepted-source-capture-roster.v2.9.5"
)


_RUNTIME_POSE_POLICY_DOMAIN = "spatialcf.competition-native-runtime-pose-policy.v2.9.5"


_SOURCE_POLICY_VERSION = "competition-native-source-policy:2.9.13"


_SOURCE_PLAN_VERSION = "competition-native-source-plan:2.9.10"

from spatialcf.generation.planning.native_versions import (
    PERSISTENT_POLICY, PERSISTENT_PLAN, guard_for_policy, plan_for_policy,
    policy_for_captures, endpoint_for_guard, hash_domain,
    LEGACY_CANDIDATES, CLEAR_FIRST_CANDIDATES,
)


_FILES = {"plan.json", "checksums.sha256"}


_MAX_PLAN_BYTES = 256 * 1024 * 1024


_MAX_POLICY_BYTES = 16 * 1024 * 1024


_MAX_CHECKSUM_BYTES = 1024


_MAX_REQUESTS = 1_000


_MAX_SOURCES = 512


_MAX_TARGET_ROWS = 40_000


_MAX_REASON_CHARS = 4_096


_SCREENING_DOMAIN_OPERATIONS = 100_000


_RELATIONS = tuple(sorted(Relation, key=lambda item: item.value))


def _manifest_file_sha256(manifest: object) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest) + b"\n").hexdigest()


def _runtime_identity_sha256(capture: CompetitionNativeSourceCaptureV2_9) -> str:
    return canonical_sha256(capture.runtime_identity, domain=_RUNTIME_DOMAIN)


def _legacy_sha256(value: BaseModel) -> str:
    return competition_legacy_sha256(value)


def _accepted_source_capture_roster_sha256(
    captures: tuple[CompetitionNativeSourceCaptureV2_9, ...],
) -> Sha256Digest:
    if type(captures) is not tuple or any(
        type(item) is not CompetitionNativeSourceCaptureV2_9 for item in captures
    ):
        raise TypeError("accepted source capture roster must be an exact tuple")
    checked = tuple(
        CompetitionNativeSourceCaptureV2_9.model_validate(
            item.model_dump(mode="python", warnings="error"), strict=True
        )
        for item in captures
    )
    source_ids = tuple(item.source.source_id for item in checked)
    if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(
        set(source_ids)
    ):
        raise ValueError("accepted source capture roster is not canonical")
    return canonical_sha256(
        {
            "roster_version": (
                "competition-native-accepted-source-capture-roster:2.9.5"
            ),
            "captures": tuple(
                {
                    "source": item.source.model_dump(mode="json"),
                    "source_capture_sha256": item.source_capture_sha256,
                }
                for item in checked
            ),
        },
        domain=_ACCEPTED_ROSTER_DOMAIN,
    )


def _target_reachability_ledger_sha256(
    rows: tuple[CandidateTargetReachability, ...],
) -> Sha256Digest:
    if type(rows) is not tuple or any(
        type(item) is not CandidateTargetReachability for item in rows
    ):
        raise TypeError("target reachability ledger must be an exact tuple")
    candidate_ids = tuple(item.candidate_id for item in rows)
    if candidate_ids != tuple(sorted(set(candidate_ids))):
        raise ValueError("target reachability ledger is not canonical")
    return canonical_sha256(rows, domain=_TARGET_LEDGER_DOMAIN)


class RuntimePosePolicy(CanonicalModel):
    policy_version: Literal["competition-native-runtime-pose-policy:2.9.5"] = (
        "competition-native-runtime-pose-policy:2.9.5"
    )
    claim_scope: Literal["ONE_SUBJECT_FINAL_OBSERVED_POSE_AND_OBB_REVERIFIED"] = (
        "ONE_SUBJECT_FINAL_OBSERVED_POSE_AND_OBB_REVERIFIED"
    )
    max_subject_position_residual_m: Literal[0.0001] = 0.0001
    max_subject_rotation_residual_degrees: Literal[0.05] = 0.05
    max_subject_obb_corner_residual_m: Literal[0.01] = 0.01

    @property
    def runtime_pose_policy_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_RUNTIME_POSE_POLICY_DOMAIN)


class SourcePolicy(CanonicalModel):
    """The single current bounded source-campaign policy."""

    policy_version: Literal[_SOURCE_POLICY_VERSION, PERSISTENT_POLICY] = _SOURCE_POLICY_VERSION
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    roster_manifest_sha256: Sha256Digest
    width: int = Field(strict=True, gt=0, le=4096)
    height: int = Field(strict=True, gt=0, le=4096)
    seed: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    max_endpoint_candidate_points: int = Field(default=64, strict=True, gt=0, le=256)
    candidate_wall_time_seconds: Literal[60, 600] = 60
    max_settlement_steps: int = Field(default=60, strict=True, gt=0, le=600)
    solver_config: ContinuousYawSolverConfigV2_9
    endpoint_candidate_strategy: Literal[
        "CAMERA_SELECTED_CAPTURE_BOUND_NATIVE_REPLAY_PARALLEL_PATCH_"
        "RELATION_RANKED_RUNTIME_COLLISION_DELEGATED_BBOX_VISIBILITY",
        CLEAR_FIRST_CANDIDATES,
    ] = (
        "CAMERA_SELECTED_CAPTURE_BOUND_NATIVE_REPLAY_PARALLEL_PATCH_"
        "RELATION_RANKED_RUNTIME_COLLISION_DELEGATED_BBOX_VISIBILITY"
    )
    max_included_collision_obstacles: int = Field(default=10, strict=True, gt=0, le=32)
    endpoint_proxy_policy: Literal[
        "capture_patch_runtime_collision_delegated_bbox_visibility_v2_9_4"
    ] = "capture_patch_runtime_collision_delegated_bbox_visibility_v2_9_4"
    max_parallel_endpoint_workers: Literal[4] = 4
    camera_policy: CameraPolicy
    accepted_source_capture_roster_sha256: Sha256Digest
    runtime_pose_policy: RuntimePosePolicy
    roster_target_gate: Literal["SOURCE_ONLY_NATIVE_TARGET_REACHABILITY_V2_9_4"] = (
        "SOURCE_ONLY_NATIVE_TARGET_REACHABILITY_V2_9_4"
    )
    roster_policy_sha256: Sha256Digest
    target_reachability_ledger_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.policy_version != PERSISTENT_POLICY and self.endpoint_candidate_strategy != LEGACY_CANDIDATES:
            raise ValueError("legacy policy requires legacy candidate strategy")
        expected_wall_time = 600 if self.policy_version == PERSISTENT_POLICY else 60
        if self.candidate_wall_time_seconds != expected_wall_time:
            raise ValueError("native policy wall time/version mismatch")
        if self.width * self.height > 4_194_304:
            raise ValueError("native source render policy exceeds pixel limit")
        return self

    @property
    def competition_native_source_policy_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=hash_domain(self.policy_version))


class SourceRequestOutcome(CanonicalModel):
    request_id: str = Field(pattern=r"^request-[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    selection_index: int = Field(strict=True, ge=0)
    relation_before: Relation
    relation_slot_index: int = Field(strict=True, ge=0)
    status: Literal["planned", "rejected"]
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    source_locator_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    source_scene_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    subject_id: str = Field(strict=True, min_length=1, max_length=512)
    subject_name: str = Field(strict=True, min_length=1, max_length=512)
    reference_id: str = Field(strict=True, min_length=1, max_length=512)
    reference_name: str = Field(strict=True, min_length=1, max_length=512)
    support_kind: CompetitionNativeSupportKindV2_9
    placement_sha256: Sha256Digest
    case_id: str = Field(strict=True, min_length=1, max_length=512)
    planning_workspace: EndpointWorkspace | None
    endpoint_workspace: EndpointWorkspace | None
    endpoint_candidate_index: int | None = Field(default=None, strict=True, ge=0)
    attempted_workspace_count: int | None = Field(default=None, strict=True, gt=0)
    candidate_point_count: int | None = Field(default=None, strict=True, gt=0)
    solve_result_sha256: Sha256Digest | None
    selected_edit_sha256: Sha256Digest | None = None
    source_view_fact_sha256: Sha256Digest | None = None
    source_view_guard: SourceViewGuard | None = None
    reasons: tuple[str, ...] = Field(max_length=32)
    surface_evidence_sha256: Sha256Digest | None = None
    subject_surface_evidence_sha256: Sha256Digest | None = None
    patch_index: int | None = Field(default=None, strict=True, ge=0)
    patch_sha256: Sha256Digest | None = None
    semantic_problem_sha256: Sha256Digest | None = None
    proxy_bundle_sha256: Sha256Digest | None = None
    endpoint_plan_sha256: Sha256Digest | None = None
    camera_evidence_sha256: Sha256Digest | None = None
    runtime_collision_delegated_native_object_ids: tuple[CanonicalId, ...] | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            type(item) is not str or not item or len(item) > _MAX_REASON_CHARS
            for item in self.reasons
        ):
            raise ValueError("source request outcome reasons are not canonical")
        metrics = (
            self.planning_workspace,
            self.endpoint_workspace,
            self.endpoint_candidate_index,
            self.attempted_workspace_count,
            self.candidate_point_count,
            self.solve_result_sha256,
            self.selected_edit_sha256,
            self.source_view_fact_sha256,
            self.source_view_guard,
        )
        if self.status == "planned":
            if any(item is None for item in metrics) or self.reasons:
                raise ValueError("planned endpoint outcome is not closed")
            if self.endpoint_candidate_index >= self.candidate_point_count:
                raise ValueError("planned endpoint candidate escaped its roster")
        elif any(item is not None for item in metrics) or not self.reasons:
            raise ValueError("rejected endpoint outcome is not closed")
        return self

    @model_validator(mode="after")
    def validate_patch_lineage(self) -> Self:
        lineage = (
            self.surface_evidence_sha256,
            self.subject_surface_evidence_sha256,
            self.patch_index,
            self.patch_sha256,
            self.semantic_problem_sha256,
            self.proxy_bundle_sha256,
            self.endpoint_plan_sha256,
        )
        if self.status == "planned":
            if any(item is None for item in lineage):
                raise ValueError("planned patch-bound lineage is not closed")
            if (
                self.source_view_guard.status != "PASSED"
                or self.source_view_guard.source_view_fact_sha256
                != self.source_view_fact_sha256
                or self.source_view_guard.semantic_problem_sha256
                != self.semantic_problem_sha256
                or self.source_view_guard.solve_result_sha256
                != self.solve_result_sha256
                or self.source_view_guard.edit_sha256 != self.selected_edit_sha256
            ):
                raise ValueError("planned source-view guard lineage is not closed")
        elif any(item is not None for item in lineage):
            raise ValueError("rejected patch-bound lineage must be empty")
        return self

    @model_validator(mode="after")
    def validate_camera_lineage(self) -> Self:
        if (self.status == "planned") != (self.camera_evidence_sha256 is not None):
            raise ValueError("camera evidence lineage must exist exactly when planned")
        return self

    @model_validator(mode="after")
    def validate_runtime_collision_delegation(self) -> Self:
        delegated = self.runtime_collision_delegated_native_object_ids
        if self.status == "planned":
            if delegated is None or delegated != tuple(sorted(set(delegated))):
                raise ValueError(
                    "planned runtime collision delegation must be explicit and canonical"
                )
        elif delegated is not None:
            raise ValueError("rejected runtime collision delegation must be absent")
        return self


class SourceSlotEntry(CanonicalModel):
    relation: Relation
    request_id: str | None = Field(default=None, pattern=r"^request-[0-9a-f]{64}$")


class SourceSlotOutcome(CanonicalModel):
    slot_index: int = Field(strict=True, ge=0)
    status: Literal["ready", "incomplete"]
    entries: tuple[SourceSlotEntry, ...] = Field(
        min_length=len(Relation), max_length=len(Relation)
    )
    blockers: tuple[str, ...] = Field(max_length=len(Relation))
    batch_id: str | None = Field(default=None, pattern=r"^b[0-9]{4}$")

    @model_validator(mode="after")
    def validate_slot(self) -> Self:
        if tuple(item.relation for item in self.entries) != _RELATIONS:
            raise ValueError("source slot relation roster is not canonical")
        if self.blockers != tuple(sorted(set(self.blockers))) or any(
            type(item) is not str or not item or len(item) > _MAX_REASON_CHARS
            for item in self.blockers
        ):
            raise ValueError("source slot blockers are not canonical")
        complete = all(item.request_id is not None for item in self.entries)
        if self.status == "ready":
            if (
                not complete
                or self.blockers
                or self.batch_id != f"b{self.slot_index:04d}"
            ):
                raise ValueError("ready source slot is not closed")
        elif not self.blockers or self.batch_id is not None:
            raise ValueError("incomplete source slot is not closed")
        return self


class BatchRequest(CanonicalModel):
    request_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    case_id: CanonicalId
    scene_id: str = Field(strict=True, min_length=1, max_length=256)
    subject_name: str = Field(strict=True, min_length=1, max_length=256)
    reference_name: str = Field(strict=True, min_length=1, max_length=256)
    relation_before: Relation
    relation_after: Relation
    endpoint_workspace: EndpointWorkspace
    solver_config: ContinuousYawSolverConfigV2_9
    max_settlement_steps: int = Field(default=60, strict=True, gt=0, le=600)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.subject_name == self.reference_name:
            raise ValueError("native batch subject and reference must differ")
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("native batch relation_after must be opposite")
        candidate = self.solver_config.candidate_config
        if (
            candidate.max_domain_operations > 2_000_000
            or candidate.max_so2_atomic_steps > 2_000_000
            or candidate.max_candidate_cells > 50_000
            or self.solver_config.max_objective_partition_cells > 100_000
        ):
            raise ValueError("native batch solver policy exceeds the frozen limits")
        return self


class BatchManifest(CanonicalModel):
    manifest_version: Literal["competition-native-batch-manifest:2.9.1"] = (
        "competition-native-batch-manifest:2.9.1"
    )
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    seed: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    requests: tuple[BatchRequest, ...]

    @model_validator(mode="after")
    def validate_partial_roster(self) -> Self:
        if (
            self.width > 4096
            or self.height > 4096
            or self.width * self.height > 4_194_304
        ):
            raise ValueError("native batch render policy exceeds the frozen limits")
        requests = tuple(
            BatchRequest.model_validate(item, strict=True) for item in self.requests
        )
        if not 1 <= len(requests) <= len(Relation):
            raise ValueError("native partial batch must contain one to six requests")
        request_ids = tuple(item.request_id for item in requests)
        case_ids = tuple(item.case_id for item in requests)
        relations = tuple(item.relation_before for item in requests)
        if request_ids != tuple(sorted(request_ids)):
            raise ValueError("native batch requests must use canonical request order")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("native batch request IDs must be unique")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("native batch case IDs must be unique")
        if len(set(relations)) != len(relations):
            raise ValueError("native partial batch relations must be unique")
        return self


def _case_id(
    policy: SourcePolicy, request: CompetitionNativeSelectedRequestV2_9
) -> str:
    return (
        f"case:{policy.campaign_id}:{request.selection_index:04d}:"
        f"{request.request_id[-12:]}"
    )


def _workspace_is_subset(inner: EndpointWorkspace, outer: EndpointWorkspace) -> bool:
    return (
        outer.min_x_m <= inner.min_x_m < inner.max_x_m <= outer.max_x_m
        and outer.min_y_m <= inner.min_y_m < inner.max_y_m <= outer.max_y_m
    )


def _fresh_solve_result(
    proxy: ProxyBundle,
    config: ContinuousYawSolverConfigV2_9,
    expected_solve_result_sha256: Sha256Digest,
) -> ContinuousYawCertifiedSuccessResultV2_9 | None:
    solved = solve_minimum_cost(proxy.semantic_problem, config)
    result = solved.result
    if (
        type(result) is ContinuousYawCertifiedSuccessResultV2_9
        and result.semantic_problem_sha256
        == proxy.semantic_problem.semantic_problem_sha256
        and type(result.solver_config) is ContinuousYawSolverConfigV2_9
        and result.solver_config == config
        and result.solver_config.config_sha256 == config.config_sha256
        and result.solve_result_sha256 == expected_solve_result_sha256
    ):
        return result
    return None


class SourcePlan(CanonicalModel):
    """The single current, fully replayable source campaign plan."""

    plan_version: Literal[_SOURCE_PLAN_VERSION, PERSISTENT_PLAN] = _SOURCE_PLAN_VERSION
    evidence_eligible: Literal[False] = False
    source_policy: SourcePolicy
    source_policy_sha256: Sha256Digest
    roster_manifest: CompetitionNativeCandidateRosterManifestV2_9
    roster_manifest_sha256: Sha256Digest
    roster_manifest_file_sha256: Sha256Digest
    source_captures: tuple[CompetitionNativeSourceCaptureV2_9, ...] = Field(
        max_length=_MAX_SOURCES
    )
    request_outcomes: tuple[SourceRequestOutcome, ...] = Field(max_length=_MAX_REQUESTS)
    slots: tuple[SourceSlotOutcome, ...] = Field(max_length=_MAX_REQUESTS)
    batches: tuple[BatchManifest, ...] = Field(max_length=_MAX_REQUESTS)
    surface_evidence: tuple[SourceSurfaceEvidence, ...] = Field(max_length=_MAX_SOURCES)
    camera_evidence: tuple[SourceCameraEvidence, ...] = Field(max_length=_MAX_SOURCES)
    target_reachability: tuple[CandidateTargetReachability, ...] = Field(
        max_length=_MAX_TARGET_ROWS
    )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        policy = self.source_policy
        if (policy.policy_version != policy_for_captures(self.source_captures)
                or self.plan_version != plan_for_policy(policy.policy_version)):
            raise ValueError("native runtime/policy/plan version mismatch")
        if self.source_policy_sha256 != policy.competition_native_source_policy_sha256:
            raise ValueError("native source plan policy digest mismatch")
        if (
            self.roster_manifest_sha256 != self.roster_manifest.manifest_sha256
            or policy.roster_manifest_sha256 != self.roster_manifest_sha256
            or self.roster_manifest_file_sha256
            != _manifest_file_sha256(self.roster_manifest)
            or policy.campaign_id != self.roster_manifest.campaign_id
        ):
            raise ValueError("native source plan roster manifest digest mismatch")
        requests = self.roster_manifest.requests
        selected_source_ids = {item.source_id for item in requests}
        capture_source_ids = tuple(
            item.source.source_id for item in self.source_captures
        )
        if (
            capture_source_ids != tuple(sorted(capture_source_ids))
            or len(capture_source_ids) != len(set(capture_source_ids))
            or not selected_source_ids.issubset(capture_source_ids)
        ):
            raise ValueError("camera-bound accepted capture roster is not closed")
        if tuple(item.request_id for item in self.request_outcomes) != tuple(
            item.request_id for item in requests
        ):
            raise ValueError("native source request outcomes do not cover the roster")

        captures = {item.source.source_id: item for item in self.source_captures}
        relation_counts: Counter[Relation] = Counter()
        expected_slots: dict[tuple[Relation, int], str] = {}
        for request, outcome in zip(requests, self.request_outcomes, strict=True):
            slot = relation_counts[request.relation_before]
            relation_counts[request.relation_before] += 1
            expected_slots[(request.relation_before, slot)] = request.request_id
            capture = captures.get(request.source_id)
            if capture is None:
                raise ValueError("native source request has no captured source")
            placements = tuple(
                item
                for item in capture.placement_facts
                if item.object_id == request.subject_id
            )
            if len(placements) != 1:
                raise ValueError("native source request placement is not unique")
            placement = placements[0]
            try:
                subject = capture.scene.object_by_id(request.subject_id)
                reference = capture.scene.object_by_id(request.reference_id)
            except KeyError as error:
                raise ValueError(
                    "native source request object binding changed"
                ) from error
            if (
                outcome.candidate_id != request.candidate_id
                or outcome.selection_index != request.selection_index
                or outcome.relation_before is not request.relation_before
                or outcome.relation_slot_index != slot
                or outcome.source_id != request.source_id
                or outcome.source_locator_sha256 != request.source_locator_sha256
                or outcome.source_capture_sha256 != request.source_capture_sha256
                or outcome.scene_id != request.scene_id
                or outcome.subject_id != request.subject_id
                or outcome.subject_name != request.subject_name
                or outcome.reference_id != request.reference_id
                or outcome.reference_name != request.reference_name
                or outcome.support_kind is not request.support_kind
                or capture.source.source_locator_sha256 != request.source_locator_sha256
                or capture.source_capture_sha256 != request.source_capture_sha256
                or capture.scene.scene_id != request.scene_id
                or subject.name != request.subject_name
                or reference.name != request.reference_name
                or placement.support_kind is not request.support_kind
                or outcome.source_scene_sha256 != _legacy_sha256(capture.scene)
                or outcome.runtime_identity_sha256 != _runtime_identity_sha256(capture)
                or outcome.placement_sha256 != placement.placement_sha256
                or outcome.case_id != _case_id(policy, request)
            ):
                raise ValueError("native source request outcome binding mismatch")
            if outcome.status == "planned":
                if (
                    request.support_kind
                    is not CompetitionNativeSupportKindV2_9.RECEPTACLE
                    or placement.availability
                    is not CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
                    or placement.support_object_id is None
                    or not placement.native_positions
                    or subject.support_object_id != placement.support_object_id
                    or capture.runtime_identity.native_scene_name == "Procedural"
                    or not capture.is_scene_at_rest
                    or (
                        capture.runtime_identity.width,
                        capture.runtime_identity.height,
                        capture.runtime_identity.seed,
                    )
                    != (policy.width, policy.height, policy.seed)
                ):
                    raise ValueError("planned native source outcome is not executable")
                region = placement.position_region
                if (
                    region is None
                    or not region.components
                    or region.subject_object_id != request.subject_id
                    or region.source_kind != "ai2thor-receptacle-trigger-grid-v1"
                ):
                    raise ValueError(
                        "planned native source receptacle position region is invalid"
                    )
                if not endpoint_workspace_within_position_region(
                    subject, region, outcome.endpoint_workspace
                ):
                    raise ValueError(
                        "planned native source endpoint escaped receptacle position region"
                    )
                try:
                    expected_workspace = default_planning_workspace(
                        capture.scene, request.subject_id
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "planned native source workspace cannot be replayed"
                    ) from error
                if (
                    outcome.planning_workspace != expected_workspace
                    or not _workspace_is_subset(
                        outcome.endpoint_workspace, expected_workspace
                    )
                    or outcome.candidate_point_count
                    > policy.max_endpoint_candidate_points
                    or outcome.attempted_workspace_count
                    > 4 * outcome.candidate_point_count
                ):
                    raise ValueError("planned native source endpoint metrics changed")

        slot_count = max(relation_counts.values(), default=0)
        if tuple(item.slot_index for item in self.slots) != tuple(range(slot_count)):
            raise ValueError("native source slot indices are not closed")
        outcome_by_id = {item.request_id: item for item in self.request_outcomes}
        request_by_id = {item.request_id: item for item in requests}
        expected_batches: list[BatchManifest] = []
        for slot in self.slots:
            for entry in slot.entries:
                if entry.request_id != expected_slots.get(
                    (entry.relation, slot.slot_index)
                ):
                    raise ValueError("native source slot membership changed")
            expected_blockers = tuple(
                sorted(
                    f"source_plan:missing_relation:{entry.relation.value}"
                    if entry.request_id is None
                    else f"source_plan:request_rejected:{entry.request_id}"
                    for entry in slot.entries
                    if entry.request_id is None
                    or outcome_by_id[entry.request_id].status != "planned"
                )
            )
            should_ready = not expected_blockers
            if (
                slot.blockers != expected_blockers
                or slot.status != ("ready" if should_ready else "incomplete")
                or slot.batch_id
                != (f"b{slot.slot_index:04d}" if should_ready else None)
            ):
                raise ValueError("native source slot status changed")
            planned_request_ids = tuple(
                sorted(
                    entry.request_id
                    for entry in slot.entries
                    if entry.request_id is not None
                    and outcome_by_id[entry.request_id].status == "planned"
                )
            )
            if planned_request_ids:
                batch_requests = tuple(
                    BatchRequest(
                        request_id=request_id,
                        case_id=outcome_by_id[request_id].case_id,
                        scene_id=request_by_id[request_id].scene_id,
                        subject_name=request_by_id[request_id].subject_name,
                        reference_name=request_by_id[request_id].reference_name,
                        relation_before=request_by_id[request_id].relation_before,
                        relation_after=request_by_id[request_id].relation_after,
                        endpoint_workspace=outcome_by_id[request_id].endpoint_workspace,
                        solver_config=policy.solver_config,
                        max_settlement_steps=policy.max_settlement_steps,
                    )
                    for request_id in planned_request_ids
                )
                expected_batches.append(
                    BatchManifest(
                        batch_id=f"b{slot.slot_index:04d}",
                        width=policy.width,
                        height=policy.height,
                        seed=policy.seed,
                        requests=batch_requests,
                    )
                )
        if self.batches != tuple(expected_batches):
            raise ValueError("native source ready batch manifests changed")
        return self

    @model_validator(mode="after")
    def validate_patch_bound_plan(self) -> Self:
        capture_ids = tuple(item.source.source_id for item in self.source_captures)
        if tuple(item.source_id for item in self.surface_evidence) != capture_ids:
            raise ValueError("patch-bound plan evidence roster is not canonical")
        captures = {item.source.source_id: item for item in self.source_captures}
        surfaces = {
            item.source_id: verify_source_surface_evidence(
                captures[item.source_id], item
            )
            for item in self.surface_evidence
        }
        requests = {item.request_id: item for item in self.roster_manifest.requests}
        for outcome in self.request_outcomes:
            if outcome.status != "planned":
                continue
            request = requests[outcome.request_id]
            capture = captures[outcome.source_id]
            surface = surfaces[outcome.source_id]
            subjects = tuple(
                item
                for item in surface.subjects
                if item.subject_object_id == outcome.subject_id
            )
            placements = tuple(
                item
                for item in capture.placement_facts
                if item.object_id == outcome.subject_id
            )
            if len(subjects) != 1 or len(placements) != 1:
                raise ValueError("planned patch-bound subject join is not unique")
            subject_evidence = subjects[0]
            placement = placements[0]
            if (
                outcome.surface_evidence_sha256 != surface.surface_evidence_sha256
                or outcome.subject_surface_evidence_sha256
                != subject_evidence.subject_surface_evidence_sha256
                or outcome.placement_sha256 != placement.placement_sha256
                or outcome.patch_index >= len(subject_evidence.patches)
                or outcome.patch_sha256
                != subject_evidence.patches[outcome.patch_index].patch_sha256
                or capture.source_view_fact is None
                or outcome.source_view_fact_sha256
                != capture.source_view_fact.source_view_fact_sha256
            ):
                raise ValueError("planned patch-bound evidence lineage changed")
            endpoint = EndpointPlan(
                plan_version=endpoint_for_guard(guard_for_policy(self.source_policy.policy_version)),
                planning_workspace=outcome.planning_workspace,
                endpoint_workspace=outcome.endpoint_workspace,
                candidate_index=outcome.endpoint_candidate_index,
                attempted_workspace_count=outcome.attempted_workspace_count,
                candidate_point_count=outcome.candidate_point_count,
                patch_index=outcome.patch_index,
                patch_sha256=outcome.patch_sha256,
                source_capture_sha256=outcome.source_capture_sha256,
                placement_sha256=outcome.placement_sha256,
                surface_evidence_sha256=outcome.surface_evidence_sha256,
                subject_surface_evidence_sha256=(
                    outcome.subject_surface_evidence_sha256
                ),
                semantic_problem_sha256=outcome.semantic_problem_sha256,
                proxy_bundle_sha256=outcome.proxy_bundle_sha256,
                solve_result_sha256=outcome.solve_result_sha256,
                selected_edit_sha256=outcome.selected_edit_sha256,
                source_view_fact_sha256=outcome.source_view_fact_sha256,
                source_view_guard=outcome.source_view_guard,
                runtime_collision_delegated_native_object_ids=(
                    outcome.runtime_collision_delegated_native_object_ids
                ),
            )
            if outcome.endpoint_plan_sha256 != endpoint.endpoint_plan_sha256:
                raise ValueError("planned patch-bound endpoint plan changed")
            intervention = InterventionSpec(
                subject_id=request.subject_id,
                reference_id=request.reference_id,
                relation_before=request.relation_before,
                relation_after=request.relation_after,
                camera_id=request.camera_id,
            )
            proxy = build_proxy_bundle(
                capture.scene,
                intervention,
                workspace=outcome.endpoint_workspace,
                source_surface_evidence=surface,
                subject_surface_evidence=subject_evidence,
                placement=SubjectPlacementFact(
                    source_capture=capture, subject_placement=placement
                ),
                collision_delegation=CollisionDelegation(
                    case_id=outcome.case_id, patch_index=outcome.patch_index
                ),
            )
            if (
                outcome.semantic_problem_sha256
                != proxy.semantic_problem.semantic_problem_sha256
                or outcome.proxy_bundle_sha256 != proxy.proxy_bundle_sha256
                or outcome.runtime_collision_delegated_native_object_ids
                != proxy.binding.runtime_collision_delegated_native_object_ids
            ):
                raise ValueError("planned patch-bound proxy lineage changed")
            fresh_result = _fresh_solve_result(
                proxy, self.source_policy.solver_config, outcome.solve_result_sha256
            )
            if fresh_result is None:
                raise ValueError("planned patch-bound fresh solve lineage changed")
            if (
                outcome.selected_edit_sha256
                != fresh_result.selected_witness.edit.edit_sha256
            ):
                raise ValueError("planned selected edit lineage changed")
            replayed_guard = evaluate_source_view_guard(
                capture.scene,
                intervention,
                capture.source_view_fact,
                fresh_result.selected_witness.edit,
                semantic_problem_sha256=proxy.semantic_problem.semantic_problem_sha256,
                solve_result_sha256=fresh_result.solve_result_sha256,
                proxy_binding=proxy.binding,
                guard_version=guard_for_policy(self.source_policy.policy_version),
            )
            if replayed_guard != outcome.source_view_guard:
                raise ValueError("planned source-view guard replay changed")
        return self

    @model_validator(mode="after")
    def validate_camera_bound_plan(self) -> Self:
        capture_ids = tuple(item.source.source_id for item in self.source_captures)
        if self.source_policy.accepted_source_capture_roster_sha256 != (
            _accepted_source_capture_roster_sha256(self.source_captures)
        ):
            raise ValueError("camera-bound accepted capture roster digest mismatch")
        if (
            tuple(item.source_id for item in self.surface_evidence) != capture_ids
            or tuple(item.source_id for item in self.camera_evidence) != capture_ids
        ):
            raise ValueError("camera-bound plan evidence roster is not canonical")
        captures = {item.source.source_id: item for item in self.source_captures}
        evidence_by_source = {
            item.source_id: verify_competition_native_camera_evidence_capture_v2_9_3(
                captures[item.source_id], item, self.source_policy.camera_policy
            )
            for item in self.camera_evidence
        }
        for outcome in self.request_outcomes:
            expected = (
                evidence_by_source[outcome.source_id].camera_evidence_sha256
                if outcome.status == "planned"
                else None
            )
            if outcome.camera_evidence_sha256 != expected:
                raise ValueError("planned camera evidence lineage changed")
        return self

    @model_validator(mode="after")
    def validate_target_reachability(self) -> Self:
        if self.source_policy.target_reachability_ledger_sha256 != (
            _target_reachability_ledger_sha256(self.target_reachability)
        ):
            raise ValueError("source plan target reachability ledger changed")
        captures = {item.source.source_id: item for item in self.source_captures}
        surfaces = {item.source_id: item for item in self.surface_evidence}
        cameras = {item.source_id: item for item in self.camera_evidence}
        rows = {item.candidate_id: item for item in self.target_reachability}
        selected_receptacle_ids = {
            item.candidate_id
            for item in self.roster_manifest.requests
            if item.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
        }
        if not selected_receptacle_ids.issubset(rows):
            raise ValueError("source plan target reachability misses selected request")
        requests = {item.candidate_id: item for item in self.roster_manifest.requests}
        for row in self.target_reachability:
            capture = captures.get(row.source_id)
            surface = surfaces.get(row.source_id)
            camera = cameras.get(row.source_id)
            if capture is None or surface is None or camera is None:
                raise ValueError("source plan target reachability source is absent")
            placement = next(
                (
                    item
                    for item in capture.placement_facts
                    if item.object_id == row.subject_id
                ),
                None,
            )
            subject_surface = next(
                (
                    item
                    for item in surface.subjects
                    if item.subject_object_id == row.subject_id
                ),
                None,
            )
            request = requests.get(row.candidate_id)
            if (
                placement is None
                or subject_surface is None
                or row.source_capture_sha256 != capture.source_capture_sha256
                or row.placement_sha256 != placement.placement_sha256
                or row.surface_evidence_sha256 != surface.surface_evidence_sha256
                or row.subject_surface_evidence_sha256
                != subject_surface.subject_surface_evidence_sha256
                or row.camera_evidence_sha256 != camera.camera_evidence_sha256
                or (
                    request is not None
                    and (
                        row.status is not TargetReachabilityStatus.REACHABLE
                        or row.source_id != request.source_id
                        or row.scene_id != request.scene_id
                        or row.subject_id != request.subject_id
                        or row.reference_id != request.reference_id
                        or row.relation_before is not request.relation_before
                    )
                )
            ):
                raise ValueError("source plan target reachability binding changed")
        return self

    @property
    def competition_native_source_plan_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=hash_domain(self.plan_version))


_SOURCE_PLAN_CAPABILITY = object()


_SOURCE_PLAN_VERIFICATION_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class RetainedSourcePlan:
    _capability: object = field(repr=False, compare=False)
    plan: SourcePlan
    plan_payload_sha256: Sha256Digest


@dataclass(frozen=True, slots=True)
class RetainedSourcePlanVerification:
    """One source plan bound to its original retained directory snapshot."""

    _capability: object = field(repr=False, compare=False)
    root_identity: DirectoryIdentity
    plan: SourcePlan
    _entries: tuple[tuple[str, os.stat_result], ...] = field(
        repr=False,
        compare=False,
    )


# Resolve model annotations before restoring the public type identity.
RuntimePosePolicy.model_rebuild()
SourcePolicy.model_rebuild()
SourceRequestOutcome.model_rebuild()
SourceSlotEntry.model_rebuild()
SourceSlotOutcome.model_rebuild()
BatchRequest.model_rebuild()
BatchManifest.model_rebuild()
SourcePlan.model_rebuild()


# Preserve supported public names and pickle lookup.
_manifest_file_sha256.__module__ = "spatialcf.generation.planning.campaign"
_runtime_identity_sha256.__module__ = "spatialcf.generation.planning.campaign"
_legacy_sha256.__module__ = "spatialcf.generation.planning.campaign"
_accepted_source_capture_roster_sha256.__module__ = "spatialcf.generation.planning.campaign"
_target_reachability_ledger_sha256.__module__ = "spatialcf.generation.planning.campaign"
RuntimePosePolicy.__module__ = "spatialcf.generation.planning.campaign"
SourcePolicy.__module__ = "spatialcf.generation.planning.campaign"
SourceRequestOutcome.__module__ = "spatialcf.generation.planning.campaign"
SourceSlotEntry.__module__ = "spatialcf.generation.planning.campaign"
SourceSlotOutcome.__module__ = "spatialcf.generation.planning.campaign"
BatchRequest.__module__ = "spatialcf.generation.planning.campaign"
BatchManifest.__module__ = "spatialcf.generation.planning.campaign"
_case_id.__module__ = "spatialcf.generation.planning.campaign"
_workspace_is_subset.__module__ = "spatialcf.generation.planning.campaign"
_fresh_solve_result.__module__ = "spatialcf.generation.planning.campaign"
SourcePlan.__module__ = "spatialcf.generation.planning.campaign"
RetainedSourcePlan.__module__ = "spatialcf.generation.planning.campaign"
RetainedSourcePlanVerification.__module__ = "spatialcf.generation.planning.campaign"
