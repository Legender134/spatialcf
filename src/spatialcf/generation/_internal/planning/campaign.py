"""Current-only source campaign planning and immutable plan storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from spatialcf.core.v2.continuous_yaw_solver_v2_9 import (
    solve_continuous_yaw_minimum_cost_v2_9,
)
from spatialcf.domain.enums import Relation
from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.domain.v2.base import CanonicalId, Sha256Digest, V2Model
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
)
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    SourceCameraEvidence,
)
from spatialcf.generation._internal.evidence.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)
from spatialcf.generation._internal.evidence.surface import (
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    verify_source_surface_evidence,
)
from spatialcf.generation._internal.planning.endpoint import (
    EndpointPlanRejected,
    endpoint_workspace_within_position_region,
    plan_endpoint,
)
from spatialcf.generation._internal.planning.models import (
    CollisionDelegation,
    EndpointPlan,
    EndpointWorkspace,
    ProxyBundle,
    SubjectPlacementFact,
)
from spatialcf.generation._internal.planning.proxy import (
    build_proxy_bundle,
    default_planning_workspace,
    default_solver_config,
)
from spatialcf.generation.capture.compiler import (
    compile_roster,
    verify_competition_native_camera_evidence_capture_v2_9_3,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportKindV2_9,
    RosterCompilation,
)
from spatialcf.generation.errors import require_wire_version
from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

_POLICY_DOMAIN = "spatialcf.competition-native-source-policy.v2.9.13"
_PLAN_DOMAIN = "spatialcf.competition-native-source-plan.v2.9.9"
_TARGET_LEDGER_DOMAIN = (
    "spatialcf.competition-native-source-target-reachability-ledger.v2.9.8"
)
_RUNTIME_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9"
_ACCEPTED_ROSTER_DOMAIN = (
    "spatialcf.competition-native-accepted-source-capture-roster.v2.9.5"
)
_RUNTIME_POSE_POLICY_DOMAIN = "spatialcf.competition-native-runtime-pose-policy.v2.9.5"
_SOURCE_POLICY_VERSION = "competition-native-source-policy:2.9.13"
_SOURCE_PLAN_VERSION = "competition-native-source-plan:2.9.9"
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
    return hashlib.sha256(canonical_json_bytes_v2(manifest) + b"\n").hexdigest()


def _runtime_identity_sha256(capture: CompetitionNativeSourceCaptureV2_9) -> str:
    return canonical_sha256_v2(capture.runtime_identity, domain=_RUNTIME_DOMAIN)


def _legacy_sha256(value: BaseModel) -> str:
    if not isinstance(value, BaseModel):
        raise TypeError("legacy digest requires a Pydantic model")
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
            raise ValueError("legacy digest requires finite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("legacy digest requires string mapping keys")
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
    raise TypeError(f"unsupported legacy digest value {type(value).__name__!r}")


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
    return canonical_sha256_v2(
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
    return canonical_sha256_v2(rows, domain=_TARGET_LEDGER_DOMAIN)


class RuntimePosePolicy(V2Model):
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
        return canonical_sha256_v2(self, domain=_RUNTIME_POSE_POLICY_DOMAIN)


class SourcePolicy(V2Model):
    """The single current bounded source-campaign policy."""

    policy_version: Literal[_SOURCE_POLICY_VERSION] = _SOURCE_POLICY_VERSION
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    roster_manifest_sha256: Sha256Digest
    width: int = Field(strict=True, gt=0, le=4096)
    height: int = Field(strict=True, gt=0, le=4096)
    seed: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    max_endpoint_candidate_points: int = Field(default=64, strict=True, gt=0, le=256)
    candidate_wall_time_seconds: Literal[60] = 60
    max_settlement_steps: int = Field(default=60, strict=True, gt=0, le=600)
    solver_config: ContinuousYawSolverConfigV2_9
    endpoint_candidate_strategy: Literal[
        "CAMERA_SELECTED_CAPTURE_BOUND_NATIVE_REPLAY_PARALLEL_PATCH_"
        "RELATION_RANKED_RUNTIME_COLLISION_DELEGATED_BBOX_VISIBILITY"
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
        if self.width * self.height > 4_194_304:
            raise ValueError("native source render policy exceeds pixel limit")
        return self

    @property
    def competition_native_source_policy_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_POLICY_DOMAIN)


class SourceRequestOutcome(V2Model):
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


class SourceSlotEntry(V2Model):
    relation: Relation
    request_id: str | None = Field(default=None, pattern=r"^request-[0-9a-f]{64}$")


class SourceSlotOutcome(V2Model):
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


class BatchRequest(V2Model):
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


class BatchManifest(V2Model):
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


def _fresh_solve_matches(
    proxy: ProxyBundle,
    config: ContinuousYawSolverConfigV2_9,
    expected_solve_result_sha256: Sha256Digest,
) -> bool:
    solved = solve_continuous_yaw_minimum_cost_v2_9(proxy.semantic_problem, config)
    result = solved.result
    return bool(
        type(result) is ContinuousYawCertifiedSuccessResultV2_9
        and result.semantic_problem_sha256
        == proxy.semantic_problem.semantic_problem_sha256
        and type(result.solver_config) is ContinuousYawSolverConfigV2_9
        and result.solver_config == config
        and result.solver_config.config_sha256 == config.config_sha256
        and result.solve_result_sha256 == expected_solve_result_sha256
    )


class SourcePlan(V2Model):
    """The single current, fully replayable source campaign plan."""

    plan_version: Literal[_SOURCE_PLAN_VERSION] = _SOURCE_PLAN_VERSION
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
            ):
                raise ValueError("planned patch-bound evidence lineage changed")
            endpoint = EndpointPlan(
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
                proxy_bundle_sha256=outcome.proxy_bundle_sha256,
                solve_result_sha256=outcome.solve_result_sha256,
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
            if not _fresh_solve_matches(
                proxy, self.source_policy.solver_config, outcome.solve_result_sha256
            ):
                raise ValueError("planned patch-bound fresh solve lineage changed")
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
        return canonical_sha256_v2(self, domain=_PLAN_DOMAIN)


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


def build_default_source_policy(compilation: RosterCompilation) -> SourcePolicy:
    if type(compilation) is not RosterCompilation:
        raise TypeError("candidate roster compilation must be exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked = RosterCompilation.model_validate(
            compilation.model_dump(mode="python", warnings="error"), strict=True
        )
        accepted_captures = tuple(
            item.capture
            for item in checked.scene_inventory
            if item.status == "accepted" and item.capture is not None
        )
        return SourcePolicy(
            campaign_id=checked.policy.campaign_id,
            roster_manifest_sha256=checked.request_manifest.manifest_sha256,
            width=checked.policy.width,
            height=checked.policy.height,
            seed=checked.policy.seed,
            candidate_wall_time_seconds=60,
            solver_config=default_solver_config(),
            camera_policy=checked.policy.camera_policy,
            accepted_source_capture_roster_sha256=(
                _accepted_source_capture_roster_sha256(accepted_captures)
            ),
            runtime_pose_policy=RuntimePosePolicy(),
            roster_policy_sha256=checked.policy.policy_sha256,
            target_reachability_ledger_sha256=(
                _target_reachability_ledger_sha256(checked.target_reachability)
            ),
        )


def _endpoint_rejection_reason(reason: str) -> str:
    candidate = f"source_plan:endpoint:{reason}"
    if len(candidate) <= _MAX_REASON_CHARS:
        return candidate
    digest = hashlib.sha256(reason.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"source_plan:endpoint_reason_sha256:{digest}"


def _endpoint_rejection_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted({_endpoint_rejection_reason(item) for item in reasons}))
    if len(normalized) <= 32:
        return normalized
    payload = b"\0".join(item.encode("utf-8") for item in normalized)
    return (
        f"source_plan:endpoint_reasons_sha256:{hashlib.sha256(payload).hexdigest()}",
    )


def _rejected_outcome(
    request: CompetitionNativeSelectedRequestV2_9,
    slot: int,
    case_id: str,
    reasons: str | tuple[str, ...],
    *,
    capture: CompetitionNativeSourceCaptureV2_9,
    placement: CompetitionNativeSubjectPlacementFactV2_9,
) -> SourceRequestOutcome:
    if type(reasons) is str:
        reasons = (reasons,)
    return SourceRequestOutcome(
        request_id=request.request_id,
        candidate_id=request.candidate_id,
        selection_index=request.selection_index,
        relation_before=request.relation_before,
        relation_slot_index=slot,
        status="rejected",
        source_id=request.source_id,
        source_locator_sha256=request.source_locator_sha256,
        source_capture_sha256=request.source_capture_sha256,
        source_scene_sha256=_legacy_sha256(capture.scene),
        runtime_identity_sha256=_runtime_identity_sha256(capture),
        scene_id=request.scene_id,
        subject_id=request.subject_id,
        subject_name=request.subject_name,
        reference_id=request.reference_id,
        reference_name=request.reference_name,
        support_kind=request.support_kind,
        placement_sha256=placement.placement_sha256,
        case_id=case_id,
        planning_workspace=None,
        endpoint_workspace=None,
        endpoint_candidate_index=None,
        attempted_workspace_count=None,
        candidate_point_count=None,
        solve_result_sha256=None,
        reasons=tuple(sorted(set(reasons))),
    )


def _endpoint_policy_controls(policy: SourcePolicy) -> tuple[int, int, int]:
    """Freeze the current campaign-to-endpoint optional policy seam."""

    if type(policy) is not SourcePolicy:
        raise TypeError("native source policy must be exact")
    return (
        policy.max_endpoint_candidate_points,
        policy.max_included_collision_obstacles,
        _SCREENING_DOMAIN_OPERATIONS,
    )


def _preflight_request(
    policy: SourcePolicy,
    request: CompetitionNativeSelectedRequestV2_9,
    capture: CompetitionNativeSourceCaptureV2_9,
    placement: CompetitionNativeSubjectPlacementFactV2_9,
    case_id: str,
) -> str | None:
    _endpoint_policy_controls(policy)
    runtime = capture.runtime_identity
    if runtime.native_scene_name == "Procedural":
        return "source_plan:procedural_native_audit_unsupported"
    if not capture.is_scene_at_rest:
        return "source_plan:source_scene_not_at_rest"
    if (runtime.width, runtime.height, runtime.seed) != (
        policy.width,
        policy.height,
        policy.seed,
    ):
        return "source_plan:render_policy_mismatch"
    if placement.support_kind is not request.support_kind:
        return "source_plan:placement_lineage_mismatch"
    if request.support_kind is CompetitionNativeSupportKindV2_9.FLOOR:
        return "source_plan:floor_native_audit_unsupported"
    subject = capture.scene.object_by_id(request.subject_id)
    if (
        request.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
        or placement.availability
        is not CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        or placement.support_object_id is None
        or not placement.native_positions
        or subject.support_object_id != placement.support_object_id
    ):
        return "source_plan:placement_not_executable"
    region = placement.position_region
    if (
        region is None
        or not region.components
        or region.subject_object_id != request.subject_id
        or region.source_kind != "ai2thor-receptacle-trigger-grid-v1"
    ):
        return "source_plan:receptacle_position_region_missing"
    try:
        BatchRequest(
            request_id=request.request_id,
            case_id=case_id,
            scene_id=request.scene_id,
            subject_name=request.subject_name,
            reference_name=request.reference_name,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            endpoint_workspace=EndpointWorkspace(
                min_x_m=0.0, min_y_m=0.0, max_x_m=1.0, max_y_m=1.0
            ),
            solver_config=policy.solver_config,
            max_settlement_steps=policy.max_settlement_steps,
        )
    except (TypeError, ValueError):
        return "source_plan:batch_contract_unsupported"
    return None


@dataclass(frozen=True, slots=True)
class _PlanningJob:
    manifest_index: int
    request: CompetitionNativeSelectedRequestV2_9
    slot: int
    case_id: str
    capture: CompetitionNativeSourceCaptureV2_9
    placement_fact: CompetitionNativeSubjectPlacementFactV2_9
    source_surface_evidence: SourceSurfaceEvidence
    subject_surface_evidence: SubjectSurfaceEvidence
    scene: Scene
    intervention: InterventionSpec
    workspace: EndpointWorkspace
    placement: SubjectPlacementFact
    camera_evidence_sha256: Sha256Digest
    max_candidate_points: int
    max_included_collision_obstacles: int
    screening_domain_operations: int


def _prepare_job(
    policy: SourcePolicy,
    request: CompetitionNativeSelectedRequestV2_9,
    *,
    manifest_index: int,
    slot: int,
    capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    camera_evidence_sha256: Sha256Digest,
) -> _PlanningJob | SourceRequestOutcome:
    case_id = _case_id(policy, request)
    verified_evidence = verify_source_surface_evidence(capture, source_surface_evidence)
    placements = tuple(
        item for item in capture.placement_facts if item.object_id == request.subject_id
    )
    if len(placements) != 1:
        raise ValueError("fresh roster selected a non-unique placement fact")
    placement_fact = placements[0]
    subjects = tuple(
        item
        for item in verified_evidence.subjects
        if item.subject_object_id == request.subject_id
    )
    if len(subjects) != 1:
        return _rejected_outcome(
            request,
            slot,
            case_id,
            "source_plan:subject_surface_evidence_missing",
            capture=capture,
            placement=placement_fact,
        )
    reason = _preflight_request(policy, request, capture, placement_fact, case_id)
    if reason is not None:
        return _rejected_outcome(
            request,
            slot,
            case_id,
            reason,
            capture=capture,
            placement=placement_fact,
        )
    try:
        intervention = InterventionSpec(
            subject_id=request.subject_id,
            reference_id=request.reference_id,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            camera_id=request.camera_id,
        )
        workspace = default_planning_workspace(capture.scene, request.subject_id)
        placement = SubjectPlacementFact(
            source_capture=capture, subject_placement=placement_fact
        )
    except Exception as error:  # noqa: BLE001
        return _rejected_outcome(
            request,
            slot,
            case_id,
            f"source_plan:workspace:{type(error).__name__}",
            capture=capture,
            placement=placement_fact,
        )
    (
        max_candidate_points,
        max_included_collision_obstacles,
        screening_domain_operations,
    ) = _endpoint_policy_controls(policy)
    return _PlanningJob(
        manifest_index=manifest_index,
        request=request,
        slot=slot,
        case_id=case_id,
        capture=capture,
        placement_fact=placement_fact,
        source_surface_evidence=verified_evidence,
        subject_surface_evidence=subjects[0],
        scene=capture.scene,
        intervention=intervention,
        workspace=workspace,
        placement=placement,
        camera_evidence_sha256=camera_evidence_sha256,
        max_candidate_points=max_candidate_points,
        max_included_collision_obstacles=max_included_collision_obstacles,
        screening_domain_operations=screening_domain_operations,
    )


def _endpoint_worker(connection: Connection, arguments: tuple[object, ...]) -> None:
    try:
        if len(arguments) != 11:
            connection.send(("error", "InvalidWorkerArguments"))
            return
        (
            scene,
            intervention,
            workspace,
            config,
            source_surface_evidence,
            subject_surface_evidence,
            placement,
            case_id,
            max_candidate_points,
            max_included_collision_obstacles,
            screening_domain_operations,
        ) = arguments
        try:
            result = plan_endpoint(
                scene,
                intervention,
                workspace,
                config,
                source_surface_evidence,
                subject_surface_evidence,
                placement=placement,
                case_id=case_id,
                max_candidate_points=max_candidate_points,
                max_included_collision_obstacles=(max_included_collision_obstacles),
                screening_domain_operations=screening_domain_operations,
            )
        except EndpointPlanRejected as error:
            connection.send(("rejected", error.reasons))
        except Exception as error:  # noqa: BLE001
            connection.send(("error", type(error).__name__))
        else:
            connection.send(("ok", result))
    finally:
        connection.close()


def _bounded_default_endpoint_plan(
    policy: SourcePolicy, job: _PlanningJob
) -> EndpointPlan:
    arguments = (
        job.scene,
        job.intervention,
        job.workspace,
        policy.solver_config,
        job.source_surface_evidence,
        job.subject_surface_evidence,
        job.placement,
        job.case_id,
        job.max_candidate_points,
        job.max_included_collision_obstacles,
        job.screening_domain_operations,
    )
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_endpoint_worker, args=(sender, arguments), daemon=False
    )
    try:
        try:
            process.start()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise EndpointPlanRejected(
                (f"endpoint_plan:worker_start_{type(error).__name__.lower()}",)
            ) from error
        finally:
            sender.close()
        if not receiver.poll(policy.candidate_wall_time_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            raise EndpointPlanRejected(("endpoint_plan:candidate_wall_time_exceeded",))
        try:
            payload = receiver.recv()
        except EOFError as error:
            raise EndpointPlanRejected(("endpoint_plan:worker_eof",)) from error
    finally:
        receiver.close()
        if process.pid is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        process.close()
    if type(payload) is not tuple or len(payload) < 2 or type(payload[0]) is not str:
        raise EndpointPlanRejected(("endpoint_plan:worker_invalid_payload",))
    if payload[0] == "ok" and len(payload) == 2:
        if type(payload[1]) is not EndpointPlan:
            raise EndpointPlanRejected(("endpoint_plan:worker_invalid_result",))
        return payload[1]
    if payload[0] == "rejected" and len(payload) == 2:
        reasons = payload[1]
        if (
            type(reasons) is not tuple
            or not reasons
            or any(type(item) is not str or not item for item in reasons)
        ):
            raise EndpointPlanRejected(("endpoint_plan:worker_invalid_rejection",))
        raise EndpointPlanRejected(reasons)
    raise EndpointPlanRejected(("endpoint_plan:worker_internal_error",))


def _run_job(
    policy: SourcePolicy,
    endpoint_planner: Callable[..., object],
    job: _PlanningJob,
) -> SourceRequestOutcome:
    request = job.request
    default_planner = endpoint_planner is plan_endpoint
    try:
        endpoint = (
            _bounded_default_endpoint_plan(policy, job)
            if default_planner
            else endpoint_planner(
                job.scene,
                job.intervention,
                job.workspace,
                policy.solver_config,
                job.source_surface_evidence,
                job.subject_surface_evidence,
                placement=job.placement,
                case_id=job.case_id,
                max_candidate_points=job.max_candidate_points,
                max_included_collision_obstacles=(job.max_included_collision_obstacles),
                screening_domain_operations=job.screening_domain_operations,
            )
        )
    except EndpointPlanRejected as error:
        reasons = error.reasons
        if default_planner and any(
            item.startswith("endpoint_plan:worker_") for item in reasons
        ):
            reasons = ("endpoint_plan:worker_internal_error",)
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            _endpoint_rejection_reasons(reasons),
            capture=job.capture,
            placement=job.placement_fact,
        )
    except Exception as error:  # noqa: BLE001
        reason = (
            _endpoint_rejection_reasons(("endpoint_plan:worker_internal_error",))
            if default_planner
            else f"source_plan:endpoint_exception:{type(error).__name__}"
        )
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            reason,
            capture=job.capture,
            placement=job.placement_fact,
        )
    if type(endpoint) is not EndpointPlan:
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_invalid_result",
            capture=job.capture,
            placement=job.placement_fact,
        )
    subject_evidence = job.subject_surface_evidence
    if (
        endpoint.planning_workspace != job.workspace
        or not _workspace_is_subset(endpoint.endpoint_workspace, job.workspace)
        or endpoint.candidate_point_count > policy.max_endpoint_candidate_points
        or endpoint.attempted_workspace_count > 4 * endpoint.candidate_point_count
        or endpoint.source_capture_sha256 != job.capture.source_capture_sha256
        or endpoint.placement_sha256 != job.placement_fact.placement_sha256
        or endpoint.surface_evidence_sha256
        != job.source_surface_evidence.surface_evidence_sha256
        or endpoint.subject_surface_evidence_sha256
        != subject_evidence.subject_surface_evidence_sha256
        or endpoint.patch_index >= len(subject_evidence.patches)
        or endpoint.patch_sha256
        != subject_evidence.patches[endpoint.patch_index].patch_sha256
        or endpoint.runtime_collision_delegated_native_object_ids
        != tuple(sorted(endpoint.runtime_collision_delegated_native_object_ids))
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_binding_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    region = job.placement_fact.position_region
    if region is None or not endpoint_workspace_within_position_region(
        job.scene.object_by_id(request.subject_id),
        region,
        endpoint.endpoint_workspace,
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_outside_receptacle_position_region",
            capture=job.capture,
            placement=job.placement_fact,
        )
    try:
        proxy = build_proxy_bundle(
            job.scene,
            job.intervention,
            workspace=endpoint.endpoint_workspace,
            source_surface_evidence=job.source_surface_evidence,
            subject_surface_evidence=subject_evidence,
            placement=job.placement,
            collision_delegation=CollisionDelegation(
                case_id=job.case_id, patch_index=endpoint.patch_index
            ),
        )
    except Exception as error:  # noqa: BLE001
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            f"source_plan:proxy_replay:{type(error).__name__}",
            capture=job.capture,
            placement=job.placement_fact,
        )
    if (
        endpoint.proxy_bundle_sha256 != proxy.proxy_bundle_sha256
        or endpoint.runtime_collision_delegated_native_object_ids
        != proxy.binding.runtime_collision_delegated_native_object_ids
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_proxy_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    if not _fresh_solve_matches(
        proxy, policy.solver_config, endpoint.solve_result_sha256
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_solve_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    try:
        BatchRequest(
            request_id=request.request_id,
            case_id=job.case_id,
            scene_id=request.scene_id,
            subject_name=request.subject_name,
            reference_name=request.reference_name,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            endpoint_workspace=endpoint.endpoint_workspace,
            solver_config=policy.solver_config,
            max_settlement_steps=policy.max_settlement_steps,
        )
    except (TypeError, ValueError):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:batch_contract_unsupported",
            capture=job.capture,
            placement=job.placement_fact,
        )
    return SourceRequestOutcome(
        request_id=request.request_id,
        candidate_id=request.candidate_id,
        selection_index=request.selection_index,
        relation_before=request.relation_before,
        relation_slot_index=job.slot,
        status="planned",
        source_id=request.source_id,
        source_locator_sha256=request.source_locator_sha256,
        source_capture_sha256=request.source_capture_sha256,
        source_scene_sha256=_legacy_sha256(job.scene),
        runtime_identity_sha256=_runtime_identity_sha256(job.capture),
        scene_id=request.scene_id,
        subject_id=request.subject_id,
        subject_name=request.subject_name,
        reference_id=request.reference_id,
        reference_name=request.reference_name,
        support_kind=request.support_kind,
        placement_sha256=job.placement_fact.placement_sha256,
        case_id=job.case_id,
        planning_workspace=endpoint.planning_workspace,
        endpoint_workspace=endpoint.endpoint_workspace,
        endpoint_candidate_index=endpoint.candidate_index,
        attempted_workspace_count=endpoint.attempted_workspace_count,
        candidate_point_count=endpoint.candidate_point_count,
        solve_result_sha256=endpoint.solve_result_sha256,
        reasons=(),
        surface_evidence_sha256=(job.source_surface_evidence.surface_evidence_sha256),
        subject_surface_evidence_sha256=(
            subject_evidence.subject_surface_evidence_sha256
        ),
        patch_index=endpoint.patch_index,
        patch_sha256=endpoint.patch_sha256,
        semantic_problem_sha256=proxy.semantic_problem.semantic_problem_sha256,
        proxy_bundle_sha256=proxy.proxy_bundle_sha256,
        endpoint_plan_sha256=endpoint.endpoint_plan_sha256,
        camera_evidence_sha256=job.camera_evidence_sha256,
        runtime_collision_delegated_native_object_ids=(
            endpoint.runtime_collision_delegated_native_object_ids
        ),
    )


def _plan_parallel_outcomes(
    policy: SourcePolicy,
    requests: tuple[CompetitionNativeSelectedRequestV2_9, ...],
    request_slots: dict[str, int],
    captures: dict[str, CompetitionNativeSourceCaptureV2_9],
    surfaces: dict[str, SourceSurfaceEvidence],
    cameras: dict[str, SourceCameraEvidence],
    endpoint_planner: Callable[..., object],
) -> tuple[SourceRequestOutcome, ...]:
    ordered: list[SourceRequestOutcome | None] = [None] * len(requests)
    jobs: list[_PlanningJob] = []
    for index, request in enumerate(requests):
        prepared = _prepare_job(
            policy,
            request,
            manifest_index=index,
            slot=request_slots[request.request_id],
            capture=captures[request.source_id],
            source_surface_evidence=surfaces[request.source_id],
            camera_evidence_sha256=cameras[request.source_id].camera_evidence_sha256,
        )
        if type(prepared) is SourceRequestOutcome:
            ordered[index] = prepared
        elif type(prepared) is _PlanningJob:
            jobs.append(prepared)
        else:
            raise RuntimeError("endpoint preflight returned invalid result")
    with ThreadPoolExecutor(
        max_workers=policy.max_parallel_endpoint_workers,
        thread_name_prefix="spatialcf-endpoint-current",
    ) as executor:
        futures = {
            executor.submit(_run_job, policy, endpoint_planner, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            if ordered[job.manifest_index] is not None:
                raise RuntimeError("endpoint outcome index was reused")
            ordered[job.manifest_index] = future.result()
    if any(item is None for item in ordered):
        raise RuntimeError("endpoint outcomes do not close the manifest")
    return tuple(item for item in ordered if item is not None)


def plan_source_campaign(
    compilation: RosterCompilation,
    policy: SourcePolicy,
    *,
    endpoint_planner: Callable[..., object] = plan_endpoint,
) -> SourcePlan:
    """Plan each frozen request once, without backfilling rejected slots."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(compilation) is not RosterCompilation:
            raise TypeError("candidate roster compilation must be exact")
        if type(policy) is not SourcePolicy:
            raise TypeError("native source policy must be exact")
        checked = RosterCompilation.model_validate(
            compilation.model_dump(mode="python", warnings="error"), strict=True
        )
        recomputed = compile_roster(
            checked.policy,
            checked.scene_inventory,
            checked.surface_evidence,
            checked.camera_evidence,
        )
        if checked != recomputed:
            raise ValueError("candidate roster compilation is not freshly reproducible")
        policy = SourcePolicy.model_validate(
            policy.model_dump(mode="python", warnings="error"), strict=True
        )
        manifest = checked.request_manifest
        if (
            policy.roster_manifest_sha256 != manifest.manifest_sha256
            or policy.campaign_id != manifest.campaign_id
        ):
            raise ValueError("native source policy roster manifest digest mismatch")
        if (policy.width, policy.height, policy.seed) != (
            checked.policy.width,
            checked.policy.height,
            checked.policy.seed,
        ):
            raise ValueError("native source policy render does not match roster")
        if (
            policy.roster_policy_sha256 != checked.policy.policy_sha256
            or policy.target_reachability_ledger_sha256
            != _target_reachability_ledger_sha256(checked.target_reachability)
        ):
            raise ValueError("target-reachability source policy roster binding changed")

        relation_ordinals: Counter[Relation] = Counter()
        request_slots: dict[str, int] = {}
        relation_members: dict[tuple[Relation, int], str] = {}
        for request in manifest.requests:
            slot = relation_ordinals[request.relation_before]
            relation_ordinals[request.relation_before] += 1
            request_slots[request.request_id] = slot
            relation_members[(request.relation_before, slot)] = request.request_id
        slot_count = max(relation_ordinals.values(), default=0)
        frozen_entries = tuple(
            tuple(
                SourceSlotEntry(
                    relation=relation,
                    request_id=relation_members.get((relation, slot)),
                )
                for relation in _RELATIONS
            )
            for slot in range(slot_count)
        )

        selected_source_ids = {item.source_id for item in manifest.requests}
        source_captures = tuple(
            item.capture
            for item in checked.scene_inventory
            if item.status == "accepted" and item.capture is not None
        )
        captures = {item.source.source_id: item for item in source_captures}
        if not selected_source_ids.issubset(captures):
            raise ValueError("selected request has no accepted frozen capture")
        if policy.accepted_source_capture_roster_sha256 != (
            _accepted_source_capture_roster_sha256(source_captures)
        ):
            raise ValueError("camera-bound accepted capture roster digest mismatch")
        surfaces = {item.source_id: item for item in checked.surface_evidence}
        cameras = {
            item.source_id: verify_competition_native_camera_evidence_capture_v2_9_3(
                captures[item.source_id], item, policy.camera_policy
            )
            for item in checked.camera_evidence
            if item.source_id in captures
        }
        if set(surfaces) != set(captures):
            raise ValueError("surface evidence does not cover accepted sources")
        if set(cameras) != set(captures):
            raise ValueError("camera evidence does not cover accepted sources")

        outcomes = _plan_parallel_outcomes(
            policy,
            manifest.requests,
            request_slots,
            captures,
            surfaces,
            cameras,
            endpoint_planner,
        )
        outcome_by_id = {item.request_id: item for item in outcomes}
        request_by_id = {item.request_id: item for item in manifest.requests}
        slots: list[SourceSlotOutcome] = []
        batches: list[BatchManifest] = []
        for slot_index, entries in enumerate(frozen_entries):
            blockers = tuple(
                sorted(
                    f"source_plan:missing_relation:{entry.relation.value}"
                    if entry.request_id is None
                    else f"source_plan:request_rejected:{entry.request_id}"
                    for entry in entries
                    if entry.request_id is None
                    or outcome_by_id[entry.request_id].status != "planned"
                )
            )
            slots.append(
                SourceSlotOutcome(
                    slot_index=slot_index,
                    status="incomplete" if blockers else "ready",
                    entries=entries,
                    blockers=blockers,
                    batch_id=None if blockers else f"b{slot_index:04d}",
                )
            )
            planned_request_ids = tuple(
                sorted(
                    entry.request_id
                    for entry in entries
                    if entry.request_id is not None
                    and outcome_by_id[entry.request_id].status == "planned"
                )
            )
            if not planned_request_ids:
                continue
            batches.append(
                BatchManifest(
                    batch_id=f"b{slot_index:04d}",
                    width=policy.width,
                    height=policy.height,
                    seed=policy.seed,
                    requests=tuple(
                        BatchRequest(
                            request_id=request_id,
                            case_id=outcome_by_id[request_id].case_id,
                            scene_id=request_by_id[request_id].scene_id,
                            subject_name=request_by_id[request_id].subject_name,
                            reference_name=request_by_id[request_id].reference_name,
                            relation_before=request_by_id[request_id].relation_before,
                            relation_after=request_by_id[request_id].relation_after,
                            endpoint_workspace=outcome_by_id[
                                request_id
                            ].endpoint_workspace,
                            solver_config=policy.solver_config,
                            max_settlement_steps=policy.max_settlement_steps,
                        )
                        for request_id in planned_request_ids
                    ),
                )
            )
        return SourcePlan(
            source_policy=policy,
            source_policy_sha256=policy.competition_native_source_policy_sha256,
            roster_manifest=manifest,
            roster_manifest_sha256=manifest.manifest_sha256,
            roster_manifest_file_sha256=_manifest_file_sha256(manifest),
            source_captures=source_captures,
            request_outcomes=outcomes,
            slots=tuple(slots),
            batches=tuple(batches),
            surface_evidence=tuple(
                surfaces[item.source.source_id] for item in source_captures
            ),
            camera_evidence=tuple(
                cameras[item.source.source_id] for item in source_captures
            ),
            target_reachability=checked.target_reachability,
        )


def _parse_source_plan(payload: bytes) -> SourcePlan:
    """Parse only the current plan after the caller's version preflight."""

    require_wire_version(
        payload,
        artifact_kind="source plan",
        field="plan_version",
        expected=_SOURCE_PLAN_VERSION,
    )
    plan = SourcePlan.model_validate_json(payload, strict=True)
    if type(plan) is not SourcePlan:
        raise TypeError("source plan must be exact SourcePlan")
    if (
        payload
        != canonical_json_bytes_v2(plan.model_dump(mode="json", warnings="error"))
        + b"\n"
    ):
        raise ValueError("native source plan is not canonical")
    return plan


def _load_source_policy_fd(
    descriptor: int,
    name: str,
    expected_stat: os.stat_result,
) -> SourcePolicy:
    payload = read_regular_at(
        descriptor, name, _MAX_POLICY_BYTES, expected_stat=expected_stat
    )
    require_wire_version(
        payload,
        artifact_kind="source policy",
        field="policy_version",
        expected=_SOURCE_POLICY_VERSION,
    )
    policy = SourcePolicy.model_validate_json(payload, strict=True)
    if type(policy) is not SourcePolicy:
        raise TypeError("source policy must be exact SourcePolicy")
    if payload != canonical_json_bytes_v2(policy) + b"\n":
        raise ValueError("native source policy is not canonical")
    return policy


def load_source_policy(path: Path) -> SourcePolicy:
    if not isinstance(path, Path):
        raise TypeError("source policy path must be a Path")
    absolute = Path(os.path.abspath(path))
    with bound_absolute_directory(absolute.parent) as descriptor:
        item = os.stat(absolute.name, dir_fd=descriptor, follow_symlinks=False)
        policy = _load_source_policy_fd(descriptor, absolute.name, item)
        revalidate_entries(descriptor, {absolute.name: item})
        return policy


def _load_source_plan_fd(descriptor: int) -> SourcePlan:
    entries = snapshot_exact_directory(descriptor, regular_names=_FILES)
    payload = read_regular_at(
        descriptor,
        "plan.json",
        _MAX_PLAN_BYTES,
        expected_stat=entries["plan.json"],
    )
    require_wire_version(
        payload,
        artifact_kind="source plan",
        field="plan_version",
        expected=_SOURCE_PLAN_VERSION,
    )
    plan = _parse_source_plan(payload)
    checksum = read_regular_at(
        descriptor,
        "checksums.sha256",
        _MAX_CHECKSUM_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    expected = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
    if checksum != expected:
        raise ValueError("native source plan checksum mismatch")
    revalidate_entries(descriptor, entries)
    return plan


def prepare_source_plan_verification(
    root_descriptor: int,
) -> RetainedSourcePlanVerification:
    """Verify a current source plan under an already-retained descriptor."""

    if type(root_descriptor) is not int:
        raise TypeError("source plan descriptor must be an exact integer")
    entries = snapshot_exact_directory(root_descriptor, regular_names=_FILES)
    identity = directory_identity_fd(root_descriptor)
    plan = _load_source_plan_fd(root_descriptor)
    revalidate_entries(root_descriptor, entries)
    return RetainedSourcePlanVerification(
        _capability=_SOURCE_PLAN_VERIFICATION_CAPABILITY,
        root_identity=identity,
        plan=plan,
        _entries=tuple(sorted(entries.items())),
    )


def revalidate_source_plan_verification(
    root_descriptor: int,
    retained: RetainedSourcePlanVerification,
) -> SourcePlan:
    """Revalidate a prepared source plan without a new snapshot baseline."""

    if type(root_descriptor) is not int:
        raise TypeError("source plan descriptor must be an exact integer")
    if (
        type(retained) is not RetainedSourcePlanVerification
        or retained._capability is not _SOURCE_PLAN_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained source plan verification must be exact")
    if directory_identity_fd(root_descriptor) != retained.root_identity:
        raise ValueError("retained source plan root identity changed")
    revalidate_entries(root_descriptor, dict(retained._entries))
    return SourcePlan.model_validate(
        retained.plan.model_dump(mode="python", warnings="error"),
        strict=True,
    )


def load_source_plan(root: Path) -> SourcePlan:
    if not isinstance(root, Path):
        raise TypeError("source plan root must be a Path")
    with bound_absolute_directory(root) as descriptor:
        return _load_source_plan_fd(descriptor)


def _retain_checked_source_plan(plan: SourcePlan) -> RetainedSourcePlan:
    if type(plan) is not SourcePlan:
        raise TypeError("retained source plan payload must be exact SourcePlan")
    payload = (
        canonical_json_bytes_v2(plan.model_dump(mode="json", warnings="error")) + b"\n"
    )
    return RetainedSourcePlan(
        _capability=_SOURCE_PLAN_CAPABILITY,
        plan=plan,
        plan_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_source_plan_retained(root: Path) -> RetainedSourcePlan:
    if not isinstance(root, Path):
        raise TypeError("source plan root must be a Path")
    with bound_absolute_directory(root) as descriptor:
        return _retain_checked_source_plan(_load_source_plan_fd(descriptor))


def _rollback_transaction(
    transaction: object, output: Path, active_error: object
) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native source plan failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native source plan failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="native source plan publication failed before commit",
            ) from active_error
        return
    try:
        transaction.rollback()
    except BaseException as rollback_error:
        try:
            final = transaction.reconcile()
        except BaseException:  # noqa: BLE001
            published = None
        else:
            published = (
                True
                if final.location is RenameLocation.OUTPUT
                else False
                if final.location is RenameLocation.SOURCE
                else None
            )
        raise CompetitionNativePublicationError(
            output,
            published=published,
            recovery_name=transaction.recovery_name,
            detail="native source plan publication rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="native source plan publication failed and was rolled back",
        ) from active_error


def _raise_transaction_exit_error(
    parent: object,
    transaction: object,
    output: Path,
    active_error: BaseException,
) -> None:
    try:
        reconciliation = reconcile_owned_rename_at(
            parent.parent_descriptor,
            transaction.name,
            parent.output_name,
            transaction.identity,
        )
    except BaseException as reconciliation_error:  # noqa: BLE001
        active_error.add_note(str(reconciliation_error))
        published = None
        recovery_name = transaction.recovery_name
    else:
        published = (
            True
            if reconciliation.location is RenameLocation.OUTPUT
            else False
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
        recovery_name = (
            parent.output_name
            if reconciliation.location is RenameLocation.OUTPUT
            else transaction.name
            if reconciliation.location is RenameLocation.SOURCE
            else transaction.recovery_name
        )
    raise CompetitionNativePublicationError(
        output,
        published=published,
        recovery_name=recovery_name,
        detail="native source plan transaction close failed after publication",
    ) from active_error


def publish_source_plan(plan: SourcePlan, output_root: Path) -> SourcePlan:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(plan) is not SourcePlan:
            raise TypeError("native source plan must be exact SourcePlan")
        if not isinstance(output_root, Path):
            raise TypeError("native source plan output_root must be a Path")
        checked = SourcePlan.model_validate(
            plan.model_dump(mode="python", warnings="error"), strict=True
        )
        payload = canonical_json_bytes_v2(checked) + b"\n"
        if len(payload) > _MAX_PLAN_BYTES:
            raise ValueError("native source plan exceeds byte limit")
        checksum = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
        output = Path(os.path.abspath(output_root))
        parent = open_native_output_parent(output)
        transaction = None
        publication_completed = False
        try:
            parent.ensure_absent(parent.output_name)
            try:
                with parent.create_staging(label="plan") as transaction:
                    try:
                        transaction.write("plan.json", payload)
                        transaction.write("checksums.sha256", checksum)
                        transaction.fsync()
                        seal = transaction.seal()
                        verified = _load_source_plan_fd(transaction.descriptor)
                        if verified != checked:
                            raise RuntimeError(
                                "sealed native source plan verification changed"
                            )
                        transaction.validate_seal(seal)
                        transaction.publish()
                        transaction.validate_location(RenameLocation.OUTPUT)
                        transaction.validate_seal(seal)
                        parent.validate()
                        publication_completed = True
                    except BaseException as error:
                        _rollback_transaction(transaction, output, error)
                        raise
            except CompetitionNativePublicationError:
                raise
            except BaseException as error:
                if publication_completed and transaction is not None:
                    _raise_transaction_exit_error(parent, transaction, output, error)
                raise
        except BaseException as error:
            try:
                parent.close()
            except BaseException as close_error:  # noqa: BLE001
                error.add_note(str(close_error))
            raise
        try:
            parent.close()
        except BaseException as close_error:
            published: bool | None = True
            recovery_name: str | None = output.name
            if transaction is not None:
                try:
                    reconciliation = reconcile_owned_rename_at(
                        parent.parent_descriptor,
                        transaction.name,
                        parent.output_name,
                        transaction.identity,
                    )
                except BaseException as reconciliation_error:  # noqa: BLE001
                    close_error.add_note(str(reconciliation_error))
                    published = None
                    recovery_name = transaction.recovery_name
                else:
                    published = (
                        True
                        if reconciliation.location is RenameLocation.OUTPUT
                        else False
                        if reconciliation.location is RenameLocation.SOURCE
                        else None
                    )
                    recovery_name = (
                        parent.output_name
                        if reconciliation.location is RenameLocation.OUTPUT
                        else transaction.name
                        if reconciliation.location is RenameLocation.SOURCE
                        else transaction.recovery_name
                    )
            raise CompetitionNativePublicationError(
                output,
                published=published,
                recovery_name=recovery_name,
                detail="native source plan published but retained parent close failed",
            ) from close_error
        return checked


__all__ = (
    "BatchManifest",
    "BatchRequest",
    "RetainedSourcePlan",
    "RetainedSourcePlanVerification",
    "RuntimePosePolicy",
    "SourcePlan",
    "SourcePolicy",
    "SourceRequestOutcome",
    "SourceSlotEntry",
    "SourceSlotOutcome",
    "build_default_source_policy",
    "load_source_plan",
    "load_source_plan_retained",
    "load_source_policy",
    "plan_source_campaign",
    "prepare_source_plan_verification",
    "publish_source_plan",
    "revalidate_source_plan_verification",
)
