"""Current source-campaign execution over immutable partial batches."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2
from spatialcf.generation._internal.execution.batch import (
    BatchSummary,
    execute_batch,
    prepare_batch_verification,
    revalidate_batch_verification,
    verify_batch,
)
from spatialcf.generation._internal.execution.correspondence import (
    RequestLineage,
    request_binding_sha256,
)
from spatialcf.generation._internal.planning import campaign as planning_campaign
from spatialcf.generation._internal.planning.campaign import (
    BatchManifest,
    RetainedSourcePlan,
    SourcePlan,
)
from spatialcf.generation._internal.planning.models import EndpointPlan
from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
)


class SourceExecutionSummary(V2Model):
    """Execution closure for all planned requests in their frozen slots."""

    summary_version: Literal["competition-native-source-execution-summary:2.9.3"] = (
        "competition-native-source-execution-summary:2.9.3"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str
    plan_sha256: Sha256Digest
    planned_request_count: int = Field(strict=True, ge=0)
    endpoint_planned_request_count: int = Field(strict=True, ge=0)
    endpoint_rejected_request_count: int = Field(strict=True, ge=0)
    unbatched_request_count: int = Field(strict=True, ge=0)
    ready_batch_count: int = Field(strict=True, ge=0)
    executed_batch_count: int = Field(strict=True, ge=0)
    reused_batch_count: int = Field(strict=True, ge=0)
    accepted_request_count: int = Field(strict=True, ge=0)
    native_rejected_request_count: int = Field(strict=True, ge=0)
    execution_scope: Literal["ALL_ENDPOINT_PLANNED_REQUESTS_IN_FROZEN_SLOTS"] = (
        "ALL_ENDPOINT_PLANNED_REQUESTS_IN_FROZEN_SLOTS"
    )
    batched_request_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.planned_request_count != (
            self.endpoint_planned_request_count + self.endpoint_rejected_request_count
        ):
            raise ValueError("source execution endpoint counts do not close")
        if self.endpoint_planned_request_count != (
            self.unbatched_request_count + self.batched_request_count
        ):
            raise ValueError("source execution request counts do not close")
        if self.ready_batch_count != (
            self.executed_batch_count + self.reused_batch_count
        ):
            raise ValueError("source execution run counts do not close")
        if self.batched_request_count != (
            self.accepted_request_count + self.native_rejected_request_count
        ):
            raise ValueError("source execution native counts do not close")
        if self.unbatched_request_count != 0 or not (
            self.ready_batch_count
            <= self.batched_request_count
            <= 6 * self.ready_batch_count
        ):
            raise ValueError("partial source execution did not cover every endpoint")
        return self


_SOURCE_BATCH_VERIFICATION_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class RetainedSourceBatchVerification:
    """One current source batch verified against its retained directory."""

    _capability: object = field(repr=False, compare=False)
    batch_id: str
    plan_sha256: Sha256Digest
    summary: BatchSummary
    _verification: object = field(repr=False, compare=False)


def _checked_source_plan(plan: object) -> SourcePlan:
    if type(plan) is RetainedSourcePlan:
        if plan._capability is not planning_campaign._SOURCE_PLAN_CAPABILITY:
            raise TypeError("retained source plan capability must be exact")
        checked = plan.plan
        if type(checked) is not SourcePlan:
            raise TypeError("retained source plan payload must be exact SourcePlan")
        payload_sha256 = hashlib.sha256(
            canonical_json_bytes_v2(checked.model_dump(mode="json", warnings="error"))
            + b"\n"
        ).hexdigest()
        if payload_sha256 != plan.plan_payload_sha256:
            raise ValueError("retained source plan payload changed")
        return checked
    if type(plan) is not SourcePlan:
        raise TypeError("native source plan must be exact SourcePlan")
    return SourcePlan.model_validate(
        plan.model_dump(mode="python", warnings="error"), strict=True
    )


def _batch_manifest_sha256(manifest: BatchManifest) -> Sha256Digest:
    if type(manifest) is not BatchManifest:
        raise TypeError("batch manifest digest requires exact BatchManifest")
    return hashlib.sha256(canonical_json_bytes_v2(manifest) + b"\n").hexdigest()


def _batch_lineage(
    plan: SourcePlan,
    batch: BatchManifest,
) -> dict[str, RequestLineage]:
    """Reconstruct only the current named lineage from a current source plan."""

    outcomes = {item.request_id: item for item in plan.request_outcomes}
    captures = {item.source.source_id: item for item in plan.source_captures}
    surfaces = {item.source_id: item for item in plan.surface_evidence}
    cameras = {item.source_id: item for item in plan.camera_evidence}
    lineage: dict[str, RequestLineage] = {}
    for request in batch.requests:
        outcome = outcomes[request.request_id]
        if outcome.status != "planned":
            raise RuntimeError("current batch contains an unplanned request")
        capture = captures[outcome.source_id]
        surface = surfaces[outcome.source_id]
        subjects = tuple(
            item
            for item in surface.subjects
            if item.subject_object_id == outcome.subject_id
        )
        if len(subjects) != 1:
            raise RuntimeError("current request subject evidence is not unique")
        subject = subjects[0]
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
            subject_surface_evidence_sha256=(outcome.subject_surface_evidence_sha256),
            proxy_bundle_sha256=outcome.proxy_bundle_sha256,
            solve_result_sha256=outcome.solve_result_sha256,
            runtime_collision_delegated_native_object_ids=(
                outcome.runtime_collision_delegated_native_object_ids
            ),
        )
        lineage[request.request_id] = RequestLineage(
            frozen_source_scene_sha256=outcome.source_scene_sha256,
            source_capture=capture,
            source_capture_sha256=outcome.source_capture_sha256,
            runtime_identity_sha256=subject.runtime_identity_sha256,
            placement_sha256=outcome.placement_sha256,
            surface_evidence=surface,
            surface_evidence_sha256=outcome.surface_evidence_sha256,
            subject_surface_evidence_sha256=(outcome.subject_surface_evidence_sha256),
            patch_index=outcome.patch_index,
            patch_sha256=outcome.patch_sha256,
            spawn_map_source_sha256=subject.spawn_map_source_sha256,
            semantic_problem_sha256=outcome.semantic_problem_sha256,
            proxy_bundle_sha256=outcome.proxy_bundle_sha256,
            endpoint_plan_sha256=outcome.endpoint_plan_sha256,
            solve_result_sha256=outcome.solve_result_sha256,
            batch_request_sha256=request_binding_sha256(request),
            endpoint_plan=endpoint,
            camera_evidence=cameras[outcome.source_id],
            camera_policy=plan.source_policy.camera_policy,
            runtime_collision_delegated_native_object_ids=(
                outcome.runtime_collision_delegated_native_object_ids
            ),
            runtime_pose_policy=plan.source_policy.runtime_pose_policy,
        )
    return lineage


def _campaign_children(
    descriptor: int, allowed: set[str], *, complete: bool
) -> set[str]:
    names = set(os.listdir(descriptor))
    if not names <= allowed or (complete and names != allowed):
        raise ValueError("native campaign output directory set mismatch")
    for name in names:
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(item.st_mode):
            raise ValueError("native campaign output child must be a real directory")
    return names


def _require_batch_summary(batch: BatchManifest, summary: object) -> BatchSummary:
    if type(summary) is not BatchSummary:
        raise TypeError("native campaign batch verifier returned invalid summary")
    if (
        summary.batch_id != batch.batch_id
        or summary.request_manifest_sha256 != _batch_manifest_sha256(batch)
    ):
        raise ValueError("native campaign output binds a different batch")
    return summary


def prepare_source_batch_verification(
    plan: SourcePlan | RetainedSourcePlan,
    batch: BatchManifest,
    root_descriptor: int,
) -> RetainedSourceBatchVerification:
    """Verify one plan-owned batch under an already-retained descriptor."""

    checked = _checked_source_plan(plan)
    if type(batch) is not BatchManifest or batch not in checked.batches:
        raise ValueError("source batch verification requires one plan-owned batch")
    if type(root_descriptor) is not int:
        raise TypeError("source batch descriptor must be an exact integer")
    retained = prepare_batch_verification(
        root_descriptor,
        request_lineage=_batch_lineage(checked, batch),
    )
    summary = _require_batch_summary(batch, retained.summary)
    return RetainedSourceBatchVerification(
        _capability=_SOURCE_BATCH_VERIFICATION_CAPABILITY,
        batch_id=batch.batch_id,
        plan_sha256=checked.competition_native_source_plan_sha256,
        summary=summary,
        _verification=retained,
    )


def revalidate_source_batch_verification(
    root_descriptor: int,
    retained: RetainedSourceBatchVerification,
) -> BatchSummary:
    """Revalidate one prepared source batch without a new snapshot baseline."""

    if type(root_descriptor) is not int:
        raise TypeError("source batch descriptor must be an exact integer")
    if (
        type(retained) is not RetainedSourceBatchVerification
        or retained._capability is not _SOURCE_BATCH_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained source batch verification must be exact")
    summary = revalidate_batch_verification(
        root_descriptor,
        retained._verification,
    )
    if summary != retained.summary:
        raise ValueError("retained source batch summary changed")
    return BatchSummary.model_validate(
        summary.model_dump(mode="python", warnings="error"),
        strict=True,
    )


def summarize_verified_source_campaign(
    plan: SourcePlan | RetainedSourcePlan,
    batch_summaries: Mapping[str, BatchSummary],
) -> SourceExecutionSummary:
    """Close exact current campaign counts from already-verified batches."""

    checked = _checked_source_plan(plan)
    if not isinstance(batch_summaries, Mapping) or any(
        type(batch_id) is not str for batch_id in batch_summaries
    ):
        raise TypeError("verified source batch summaries must be a string mapping")
    copied = dict(batch_summaries)
    expected_ids = {batch.batch_id for batch in checked.batches}
    if set(copied) != expected_ids:
        raise ValueError("verified source batch summaries do not cover the plan")
    summaries = tuple(
        _require_batch_summary(batch, copied[batch.batch_id])
        for batch in checked.batches
    )
    endpoint_planned = sum(
        outcome.status == "planned" for outcome in checked.request_outcomes
    )
    execution_request_count = sum(len(batch.requests) for batch in checked.batches)
    return SourceExecutionSummary(
        campaign_id=checked.source_policy.campaign_id,
        plan_sha256=checked.competition_native_source_plan_sha256,
        planned_request_count=len(checked.request_outcomes),
        endpoint_planned_request_count=endpoint_planned,
        endpoint_rejected_request_count=(
            len(checked.request_outcomes) - endpoint_planned
        ),
        unbatched_request_count=endpoint_planned - execution_request_count,
        ready_batch_count=len(checked.batches),
        executed_batch_count=0,
        reused_batch_count=len(checked.batches),
        accepted_request_count=sum(item.accepted_count for item in summaries),
        native_rejected_request_count=sum(item.rejected_count for item in summaries),
        batched_request_count=execution_request_count,
    )


def run_source_campaign(
    plan: SourcePlan | RetainedSourcePlan,
    output_parent: Path,
    *,
    execute: bool,
    batch_executor=execute_batch,
    batch_verifier=verify_batch,
) -> SourceExecutionSummary:
    """Execute missing batches or freshly verify an exact current campaign."""

    if type(execute) is not bool:
        raise TypeError("native source execute flag must be exact")
    checked = _checked_source_plan(plan)
    if not isinstance(output_parent, Path):
        raise TypeError("native source output_parent must be a Path")
    parent = Path(os.path.abspath(output_parent))
    batches = checked.batches
    allowed = {item.batch_id for item in batches}
    executed = reused = accepted = native_rejected = 0
    retained_path = batch_verifier is verify_batch
    with bound_absolute_directory(parent) as descriptor:
        parent_identity = directory_identity_fd(descriptor)
        existing = _campaign_children(descriptor, allowed, complete=not execute)
        summaries: dict[str, BatchSummary] = {}
        with ExitStack() as retained_children:
            child_identities: dict[str, tuple[int, int]] = {}
            child_descriptors: dict[str, int] = {}
            retained_verifications: dict[str, object] = {}
            for batch in batches:
                if batch.batch_id not in existing:
                    continue
                child_descriptor = retained_children.enter_context(
                    bound_child_directory(descriptor, batch.batch_id)
                )
                child_identity = directory_identity_fd(child_descriptor)
                child_identities[batch.batch_id] = child_identity
                child_descriptors[batch.batch_id] = child_descriptor
                lineage = _batch_lineage(checked, batch)
                if retained_path:
                    retained = prepare_batch_verification(
                        child_descriptor, request_lineage=lineage
                    )
                    retained_verifications[batch.batch_id] = retained
                    summaries[batch.batch_id] = _require_batch_summary(
                        batch, retained.summary
                    )
                else:
                    summaries[batch.batch_id] = _require_batch_summary(
                        batch,
                        batch_verifier(
                            parent / batch.batch_id,
                            expected_root_identity=child_identity,
                            request_lineage=lineage,
                        ),
                    )
            for batch in batches:
                output = parent / batch.batch_id
                if batch.batch_id in summaries:
                    summary = summaries[batch.batch_id]
                    reused += 1
                elif execute:
                    lineage = _batch_lineage(checked, batch)
                    batch_executor(
                        batch,
                        output,
                        expected_parent_identity=parent_identity,
                        request_lineage=lineage,
                    )
                    child_descriptor = retained_children.enter_context(
                        bound_child_directory(descriptor, batch.batch_id)
                    )
                    child_identity = directory_identity_fd(child_descriptor)
                    child_identities[batch.batch_id] = child_identity
                    child_descriptors[batch.batch_id] = child_descriptor
                    if retained_path:
                        retained = prepare_batch_verification(
                            child_descriptor, request_lineage=lineage
                        )
                        retained_verifications[batch.batch_id] = retained
                        summary = _require_batch_summary(batch, retained.summary)
                    else:
                        summary = _require_batch_summary(
                            batch,
                            batch_verifier(
                                output,
                                expected_root_identity=child_identity,
                                request_lineage=lineage,
                            ),
                        )
                    summaries[batch.batch_id] = summary
                    executed += 1
                else:
                    raise FileNotFoundError(output)
                accepted += summary.accepted_count
                native_rejected += summary.rejected_count
            _campaign_children(descriptor, allowed, complete=True)
            for batch in batches:
                if retained_path:
                    final_summary = _require_batch_summary(
                        batch,
                        revalidate_batch_verification(
                            child_descriptors[batch.batch_id],
                            retained_verifications[batch.batch_id],
                        ),
                    )
                else:
                    final_summary = _require_batch_summary(
                        batch,
                        batch_verifier(
                            parent / batch.batch_id,
                            expected_root_identity=child_identities[batch.batch_id],
                            request_lineage=_batch_lineage(checked, batch),
                        ),
                    )
                if final_summary != summaries[batch.batch_id]:
                    raise ValueError("native campaign batch changed after verification")
            _campaign_children(descriptor, allowed, complete=True)
    endpoint_planned = sum(
        item.status == "planned" for item in checked.request_outcomes
    )
    execution_request_count = sum(len(item.requests) for item in batches)
    return SourceExecutionSummary(
        campaign_id=checked.source_policy.campaign_id,
        plan_sha256=checked.competition_native_source_plan_sha256,
        planned_request_count=len(checked.request_outcomes),
        endpoint_planned_request_count=endpoint_planned,
        endpoint_rejected_request_count=(
            len(checked.request_outcomes) - endpoint_planned
        ),
        unbatched_request_count=endpoint_planned - execution_request_count,
        ready_batch_count=len(batches),
        executed_batch_count=executed,
        reused_batch_count=reused,
        accepted_request_count=accepted,
        native_rejected_request_count=native_rejected,
        batched_request_count=execution_request_count,
    )


def verify_source_campaign(
    plan: SourcePlan | RetainedSourcePlan,
    output_parent: Path,
    *,
    batch_verifier=verify_batch,
) -> SourceExecutionSummary:
    return run_source_campaign(
        plan,
        output_parent,
        execute=False,
        batch_verifier=batch_verifier,
    )


def run_planned_requests(
    plan: SourcePlan | RetainedSourcePlan,
    output_parent: Path,
    *,
    execute: bool,
    batch_executor=execute_batch,
    batch_verifier=verify_batch,
) -> SourceExecutionSummary:
    return run_source_campaign(
        plan,
        output_parent,
        execute=execute,
        batch_executor=batch_executor,
        batch_verifier=batch_verifier,
    )


__all__ = (
    "RetainedSourceBatchVerification",
    "SourceExecutionSummary",
    "prepare_source_batch_verification",
    "revalidate_source_batch_verification",
    "run_planned_requests",
    "run_source_campaign",
    "summarize_verified_source_campaign",
    "verify_source_campaign",
)
