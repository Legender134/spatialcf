"""Capture roster: exact contracts and pure derivation."""

from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    Field,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.domain.request import (
    Relation,
)

from spatialcf.generation.capture.reachability_contracts import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)

from spatialcf.generation.capture._models.camera_contracts import (
    SourceCameraEvidence,
)

from spatialcf.generation.capture._models.constants import (
    DatasetSplitV2_9,
    _MANIFEST_HASH_DOMAIN,
    _MAX_CANDIDATES_TOTAL,
    _MAX_OBJECTS_PER_SCENE,
    _MAX_PERSISTED_REQUEST_TEXT_CHARS,
    _MAX_POLICY_SOURCES,
    _MAX_REASONS,
    _MAX_REASON_CHARS,
    _MAX_REQUESTS_TOTAL,
    _MAX_TEXT_CHARS,
    _SOURCE_CAPTURE_HASH_DOMAIN,
    _SUMMARY_HASH_DOMAIN,
)

from spatialcf.generation.capture._models.contracts import (
    CompetitionNativeCandidateStateV2_9,
    CompetitionNativeSourceRefV2_9,
    CompetitionNativeSubjectStateV2_9,
    CompetitionNativeSupportKindV2_9,
    RosterPolicy,
)

from spatialcf.generation.capture._models.source import (
    CompetitionNativeSourceCaptureV2_9,
    _capture_payload,
)

from spatialcf.generation.capture._models.surfaces import (
    SourceSurfaceEvidence,
)


def competition_native_roster_selection_identity_v2_9(
    capture: CompetitionNativeSourceCaptureV2_9,
) -> Sha256Digest:
    """Return the source identity used only for deterministic roster selection."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("roster selection identity requires an exact source capture")
    if capture.source_view_fact is None:
        return capture.source_capture_sha256
    payload = _capture_payload(
        source=capture.source,
        runtime_identity=capture.runtime_identity,
        scene=capture.scene,
        rgb_png_sha256=capture.rgb_png_sha256,
        depth_npy_sha256=capture.depth_npy_sha256,
        instance_png_sha256=capture.instance_png_sha256,
        pointcloud_ply_sha256=capture.pointcloud_ply_sha256,
        is_scene_at_rest=capture.is_scene_at_rest,
        settlement_pass_steps=capture.settlement_pass_steps,
        support_facts=capture.support_facts,
        floor_envelope=capture.floor_envelope,
        reachable_positions=capture.reachable_positions,
        placement_facts=capture.placement_facts,
        source_view_fact=None,
    )
    return canonical_sha256(payload, domain=_SOURCE_CAPTURE_HASH_DOMAIN)


class CompetitionNativeSourceCaptureOutcomeV2_9(CanonicalModel):
    source: CompetitionNativeSourceRefV2_9
    status: Literal["accepted", "rejected"]
    capture: CompetitionNativeSourceCaptureV2_9 | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("source outcome reasons must be canonical")
        if self.status == "accepted":
            if (
                self.capture is None
                or self.reasons
                or self.capture.source != self.source
            ):
                raise ValueError("accepted source outcome is not closed")
        elif self.capture is not None or not self.reasons:
            raise ValueError("rejected source outcome is not closed")
        return self


class CompetitionNativeObjectInventoryV2_9(CanonicalModel):
    inventory_id: str = Field(pattern=r"^object-[0-9a-f]{64}$")
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    category: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    subject_state: CompetitionNativeSubjectStateV2_9
    reference_eligible: bool
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    placement_sha256: Sha256Digest
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("object inventory reasons must be canonical")
        eligible = self.subject_state in {
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_RECEPTACLE_DOMAIN,
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_FLOOR_INNER_DOMAIN,
        }
        if eligible == bool(self.reasons):
            raise ValueError("object subject terminal state is not closed")
        return self


class CompetitionNativeCandidateInventoryV2_9(CanonicalModel):
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    state: CompetitionNativeCandidateStateV2_9
    selection_index: int | None = Field(default=None, strict=True, ge=0)
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.subject_id == self.reference_id:
            raise ValueError("candidate subject and reference must differ")
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("candidate reasons must be canonical")
        if self.state is CompetitionNativeCandidateStateV2_9.SELECTED:
            if self.selection_index is None or self.reasons:
                raise ValueError("selected candidate is not closed")
        elif self.selection_index is not None or not self.reasons:
            raise ValueError("rejected candidate is not closed")
        return self


class CompetitionNativeSelectedRequestV2_9(CanonicalModel):
    request_id: str = Field(pattern=r"^request-[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    selection_index: int = Field(strict=True, ge=0)
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    source_locator_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    relation_after: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    camera_id: Literal["main"] = "main"

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("selected request relation_after must be opposite")
        return self


class CompetitionNativeCandidateRosterManifestV2_9(CanonicalModel):
    manifest_version: Literal["competition-native-candidate-roster-manifest:2.9"] = (
        "competition-native-candidate-roster-manifest:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    requests: tuple[CompetitionNativeSelectedRequestV2_9, ...] = Field(
        max_length=_MAX_REQUESTS_TOTAL
    )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        indices = tuple(item.selection_index for item in self.requests)
        if indices != tuple(range(len(self.requests))):
            raise ValueError("request manifest selection indices are not contiguous")
        request_ids = tuple(item.request_id for item in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request manifest request IDs are not unique")
        return self

    @property
    def manifest_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_MANIFEST_HASH_DOMAIN)


class CompetitionNativeRosterRejectionV2_9(CanonicalModel):
    rejection_id: str = Field(pattern=r"^rejection-[0-9a-f]{64}$")
    stage: Literal["source", "object_subject", "object_reference", "candidate"]
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    candidate_id: str | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("roster rejection reasons must be canonical")
        if self.stage == "source":
            valid = self.object_id is None and self.candidate_id is None
        elif self.stage in {"object_subject", "object_reference"}:
            valid = self.object_id is not None and self.candidate_id is None
        else:
            valid = self.object_id is None and self.candidate_id is not None
        if not valid:
            raise ValueError("roster rejection identity is not closed")
        return self


class CompetitionNativeStageCountV2_9(CanonicalModel):
    stage: str = Field(strict=True, min_length=1, max_length=128)
    count: int = Field(strict=True, gt=0)


class CompetitionNativeCandidateStateCountV2_9(CanonicalModel):
    state: CompetitionNativeCandidateStateV2_9
    count: int = Field(strict=True, gt=0)


class CompetitionNativeRelationCountV2_9(CanonicalModel):
    relation: Relation
    count: int = Field(strict=True, gt=0)


class RosterSummary(CanonicalModel):
    summary_version: Literal["competition-native-candidate-roster-summary:2.9"] = (
        "competition-native-candidate-roster-summary:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    manifest_sha256: Sha256Digest
    source_count: int = Field(strict=True, ge=0)
    accepted_source_count: int = Field(strict=True, ge=0)
    rejected_source_count: int = Field(strict=True, ge=0)
    object_count: int = Field(strict=True, ge=0)
    eligible_subject_count: int = Field(strict=True, ge=0)
    rejected_subject_count: int = Field(strict=True, ge=0)
    candidate_count: int = Field(strict=True, ge=0)
    selected_request_count: int = Field(strict=True, ge=0)
    rejected_candidate_count: int = Field(strict=True, ge=0)
    candidate_state_counts: tuple[CompetitionNativeCandidateStateCountV2_9, ...] = (
        Field(max_length=len(CompetitionNativeCandidateStateV2_9))
    )
    relation_selected_counts: tuple[CompetitionNativeRelationCountV2_9, ...] = Field(
        max_length=len(Relation)
    )
    rejection_stage_counts: tuple[CompetitionNativeStageCountV2_9, ...] = Field(
        max_length=4
    )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.source_count != self.accepted_source_count + self.rejected_source_count:
            raise ValueError("summary source counts do not close")
        if (
            self.object_count
            != self.eligible_subject_count + self.rejected_subject_count
        ):
            raise ValueError("summary object counts do not close")
        if (
            self.candidate_count
            != self.selected_request_count + self.rejected_candidate_count
        ):
            raise ValueError("summary candidate counts do not close")
        if (
            sum(item.count for item in self.candidate_state_counts)
            != self.candidate_count
        ):
            raise ValueError("summary candidate state counts do not close")
        if (
            sum(item.count for item in self.relation_selected_counts)
            != self.selected_request_count
        ):
            raise ValueError("summary relation counts do not close")
        for counts, key in (
            (self.candidate_state_counts, lambda item: item.state.value),
            (self.relation_selected_counts, lambda item: item.relation.value),
            (self.rejection_stage_counts, lambda item: item.stage),
        ):
            keys = tuple(key(item) for item in counts)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise ValueError("summary count keys are not canonical")
        return self

    @property
    def summary_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_SUMMARY_HASH_DOMAIN)


CompetitionNativeCandidateRosterSummaryV2_9 = RosterSummary


class RosterCompilation(CanonicalModel):
    """The only supported complete roster compilation."""

    policy: RosterPolicy
    scene_inventory: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    object_inventory: tuple[CompetitionNativeObjectInventoryV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
    )
    candidate_inventory: tuple[CompetitionNativeCandidateInventoryV2_9, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )
    request_manifest: CompetitionNativeCandidateRosterManifestV2_9
    rejections: tuple[CompetitionNativeRosterRejectionV2_9, ...] = Field(
        max_length=(
            _MAX_POLICY_SOURCES
            + 2 * _MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
            + _MAX_CANDIDATES_TOTAL
        )
    )
    summary: RosterSummary
    surface_evidence: tuple[SourceSurfaceEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    camera_evidence: tuple[SourceCameraEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    target_reachability: tuple[CandidateTargetReachability, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )

    @model_validator(mode="after")
    def validate_compilation(self) -> Self:
        if self.policy.fixed_request is not None and (
                len(self.request_manifest.requests) != 1
                or not self.policy.fixed_request.matches(self.request_manifest.requests[0])):
            raise ValueError("fixed request does not bind selected manifest")
        if self.summary.selected_request_count != len(self.request_manifest.requests):
            raise ValueError("compilation request count does not close")
        if self.summary.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("compilation policy digest mismatch")
        if self.summary.manifest_sha256 != self.request_manifest.manifest_sha256:
            raise ValueError("compilation manifest digest mismatch")
        return self

    @model_validator(mode="after")
    def validate_surface_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.surface_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("surface evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.surface_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_capture_sha256 != capture.source_capture_sha256
            ):
                raise ValueError("surface evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_camera_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.camera_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("camera evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.camera_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_locator_sha256 != record.source.source_locator_sha256
                or evidence.source_capture_sha256 != capture.source_capture_sha256
                or evidence.camera != capture.scene.camera_by_id("main")
                or evidence.rgb_png_sha256 != capture.rgb_png_sha256
                or evidence.depth_npy_sha256 != capture.depth_npy_sha256
                or evidence.instance_png_sha256 != capture.instance_png_sha256
                or evidence.pointcloud_ply_sha256 != capture.pointcloud_ply_sha256
                or evidence.is_scene_at_rest is not capture.is_scene_at_rest
            ):
                raise ValueError("camera evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_target_reachability(self) -> Self:
        rows = self.target_reachability
        row_ids = tuple(item.candidate_id for item in rows)
        if row_ids != tuple(sorted(set(row_ids))):
            raise ValueError("target reachability rows are not canonical")
        candidates = {item.candidate_id: item for item in self.candidate_inventory}
        expected_ids = tuple(
            sorted(
                item.candidate_id
                for item in self.candidate_inventory
                if item.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and item.state
                in {
                    CompetitionNativeCandidateStateV2_9.SELECTED,
                    CompetitionNativeCandidateStateV2_9.REJECTED_POLICY_CAP,
                    CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE,
                }
            )
        )
        if row_ids != expected_ids:
            raise ValueError(
                "target reachability does not exactly cover receptacle selection inputs"
            )
        capture_by_source = {
            item.source.source_id: item.capture
            for item in self.scene_inventory
            if item.capture is not None
        }
        surface_by_source = {item.source_id: item for item in self.surface_evidence}
        camera_by_source = {item.source_id: item for item in self.camera_evidence}
        for row in rows:
            candidate = candidates[row.candidate_id]
            capture = capture_by_source.get(candidate.source_id)
            surface = surface_by_source.get(candidate.source_id)
            camera = camera_by_source.get(candidate.source_id)
            if capture is None or surface is None or camera is None:
                raise ValueError("target reachability source evidence is absent")
            placement = next(
                (
                    item
                    for item in capture.placement_facts
                    if item.object_id == candidate.subject_id
                ),
                None,
            )
            subject_surface = next(
                (
                    item
                    for item in surface.subjects
                    if item.subject_object_id == candidate.subject_id
                ),
                None,
            )
            unreachable = (
                candidate.state
                is CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE
            )
            if (
                placement is None
                or subject_surface is None
                or row.source_id != candidate.source_id
                or row.scene_id != candidate.scene_id
                or row.source_capture_sha256 != candidate.source_capture_sha256
                or row.subject_id != candidate.subject_id
                or row.reference_id != candidate.reference_id
                or row.relation_before is not candidate.relation_before
                or row.placement_sha256 != placement.placement_sha256
                or row.surface_evidence_sha256 != surface.surface_evidence_sha256
                or row.subject_surface_evidence_sha256
                != subject_surface.subject_surface_evidence_sha256
                or row.camera_evidence_sha256 != camera.camera_evidence_sha256
                or unreachable != (row.status is TargetReachabilityStatus.UNREACHABLE)
            ):
                raise ValueError(
                    "target reachability row does not bind candidate facts"
                )
        return self


CompetitionNativeCandidateRosterCompilationV2_9_4 = RosterCompilation


# Resolve model annotations before restoring the public type identity.
CompetitionNativeSourceCaptureOutcomeV2_9.model_rebuild()
CompetitionNativeObjectInventoryV2_9.model_rebuild()
CompetitionNativeCandidateInventoryV2_9.model_rebuild()
CompetitionNativeSelectedRequestV2_9.model_rebuild()
CompetitionNativeCandidateRosterManifestV2_9.model_rebuild()
CompetitionNativeRosterRejectionV2_9.model_rebuild()
CompetitionNativeStageCountV2_9.model_rebuild()
CompetitionNativeCandidateStateCountV2_9.model_rebuild()
CompetitionNativeRelationCountV2_9.model_rebuild()
RosterSummary.model_rebuild()
RosterCompilation.model_rebuild()


# Preserve supported public names and pickle lookup.
competition_native_roster_selection_identity_v2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSourceCaptureOutcomeV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeObjectInventoryV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCandidateInventoryV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSelectedRequestV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCandidateRosterManifestV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeRosterRejectionV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeStageCountV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCandidateStateCountV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeRelationCountV2_9.__module__ = "spatialcf.generation.capture.models"
RosterSummary.__module__ = "spatialcf.generation.capture.models"
RosterCompilation.__module__ = "spatialcf.generation.capture.models"
