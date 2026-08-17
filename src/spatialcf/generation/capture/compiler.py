"""Single current candidate-roster compiler.

The compiler consumes only frozen source facts and never branches on solver or
native execution outcomes.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Literal

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import AI2ThorAgentPose, AI2ThorNativePosition
from spatialcf.domain.enums import Relation, SolverStatus
from spatialcf.domain.models import Scene
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    CompetitionNativeCameraPlacementPositionV2_9_4,
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraScoreFamilyV2_9_3,
    CompetitionNativeCameraScoreV2_9_4,
    SourceCameraEvidence,
    build_competition_native_camera_pose_bank_v2_9_3,
    competition_native_camera_pose_bank_sha256_v2_9_3,
    score_competition_native_camera_scene_v2_9_3,
    score_competition_native_editable_camera_scene_v2_9_4,
    select_competition_native_camera_score_index_v2_9_3,
    select_competition_native_camera_score_index_v2_9_4,
    verify_competition_native_camera_observation_binding_v2_9_3,
    verify_competition_native_solver_camera_binding_v2_9_3,
)
from spatialcf.generation._internal.evidence.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
    derive_competition_native_candidate_target_reachability_from_prepared_v2_9_4,
    prepare_competition_native_target_reachability_source_v2_9_4,
)
from spatialcf.generation._internal.evidence.surface import (
    SourceSurfaceEvidence,
    verify_source_surface_evidence,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeCandidateInventoryV2_9,
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativeCandidateStateCountV2_9,
    CompetitionNativeCandidateStateV2_9,
    CompetitionNativeObjectInventoryV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeRelationCountV2_9,
    CompetitionNativeRosterRejectionV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeSourceCaptureOutcomeV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSourceRefV2_9,
    CompetitionNativeStageCountV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSubjectStateV2_9,
    CompetitionNativeSupportFactV2_9,
    CompetitionNativeSupportKindV2_9,
    RosterCompilation,
    RosterPolicy,
    RosterSummary,
    validate_competition_native_runtime_source_lineage_v2_9,
)
from spatialcf.relations.engine import RelationEngine

_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_CANDIDATE_ID_HASH_DOMAIN = "spatialcf.competition-native-candidate-id.v2.9"
_REQUEST_ID_HASH_DOMAIN = "spatialcf.competition-native-request-id.v2.9"
_OBJECT_ID_HASH_DOMAIN = "spatialcf.competition-native-object-inventory-id.v2.9"
_REJECTION_ID_HASH_DOMAIN = "spatialcf.competition-native-roster-rejection-id.v2.9"
_MAX_POLICY_SOURCES = 512
_MAX_OBJECTS_PER_SCENE = 96
_MAX_CANDIDATES_TOTAL = 40_000


class _RosterCore(V2Model):
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
    rejections: tuple[CompetitionNativeRosterRejectionV2_9, ...]
    summary: RosterSummary

    @model_validator(mode="after")
    def validate_core(self):
        if self.summary.selected_request_count != len(self.request_manifest.requests):
            raise ValueError("compilation request count does not close")
        if self.summary.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("compilation policy digest mismatch")
        if self.summary.manifest_sha256 != self.request_manifest.manifest_sha256:
            raise ValueError("compilation manifest digest mismatch")
        return self


def _digest_order(*values: object) -> str:
    return hashlib.sha256(canonical_json_bytes_v2(values)).hexdigest()


def _rejection(
    *,
    stage: Literal["source", "object_subject", "object_reference", "candidate"],
    source_id: str,
    scene_id: str,
    object_id: str | None = None,
    candidate_id: str | None = None,
    reasons: tuple[str, ...],
) -> CompetitionNativeRosterRejectionV2_9:
    reasons = tuple(sorted(set(reasons)))
    payload = {
        "candidate_id": candidate_id,
        "object_id": object_id,
        "reasons": reasons,
        "scene_id": scene_id,
        "source_id": source_id,
        "stage": stage,
    }
    return CompetitionNativeRosterRejectionV2_9(
        rejection_id="rejection-"
        + canonical_sha256_v2(payload, domain=_REJECTION_ID_HASH_DOMAIN),
        stage=stage,
        source_id=source_id,
        scene_id=scene_id,
        object_id=object_id,
        candidate_id=candidate_id,
        reasons=reasons,
    )


def _qualifying_main_view(scene: Scene, object_id: str) -> bool:
    try:
        view = scene.object_by_id(object_id).views.get("main")
        scene.camera_by_id("main")
    except KeyError:
        return False
    return bool(
        view is not None
        and view.visible_fraction >= RelationEngine.MIN_VISIBLE_FRACTION
        and view.image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION
        and view.truncated_fraction <= RelationEngine.MAX_TRUNCATED_FRACTION
    )


def _valid_geometry(scene: Scene, object_id: str) -> bool:
    try:
        item = scene.object_by_id(object_id)
    except KeyError:
        return False
    return min(item.obb.extent.x, item.obb.extent.y, item.obb.extent.z) > 0.0


def _support_dependency_graph(
    support_by_id: dict[str, CompetitionNativeSupportFactV2_9],
) -> dict[str, frozenset[str]]:
    return {
        object_id: frozenset(fact.domain_parent_object_ids)
        for object_id, fact in support_by_id.items()
    }


def _depends_on_support(
    dependency_graph: dict[str, frozenset[str]],
    first: str,
    second: str,
) -> bool:
    pending = list(dependency_graph.get(first, ()))
    visited: set[str] = set()
    while pending:
        parent_id = pending.pop()
        if parent_id == second:
            return True
        if parent_id in visited:
            continue
        visited.add(parent_id)
        pending.extend(dependency_graph.get(parent_id, ()))
    return False


def _subject_terminal(
    scene: Scene,
    object_id: str,
    support: CompetitionNativeSupportFactV2_9 | None,
    placement: CompetitionNativeSubjectPlacementFactV2_9 | None,
    *,
    has_dependent_child: bool,
) -> tuple[CompetitionNativeSubjectStateV2_9, tuple[str, ...]]:
    item = scene.object_by_id(object_id)
    if not item.movable:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_NOT_MOVABLE,
            ("object:not_movable",),
        )
    if not item.request_eligible:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_NOT_REQUEST_ELIGIBLE,
            ("object:not_request_eligible",),
        )
    if item.object_id in scene.pinned_object_ids:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_PINNED,
            ("object:pinned",),
        )
    if has_dependent_child:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_HAS_DEPENDENT_CHILD,
            ("object:has_dependent_child",),
        )
    if (
        support is None
        or placement is None
        or support.scene_id != scene.scene_id
        or support.object_name != item.name
        or support.support_object_id != item.support_object_id
        and support.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
        or placement.support_kind is not support.support_kind
        or placement.support_object_id != support.support_object_id
        or placement.floor_object_id != support.floor_object_id
    ):
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_INVALID_SOURCE_FACT,
            ("object:invalid_source_fact",),
        )
    if support.support_kind in {
        CompetitionNativeSupportKindV2_9.UNKNOWN,
        CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS,
        CompetitionNativeSupportKindV2_9.CYCLIC,
    }:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_UNKNOWN_SUPPORT,
            (f"object:support_{support.support_kind.value.lower()}",),
        )
    if placement.availability is CompetitionNativePlacementAvailabilityV2_9.MISSING:
        return (
            CompetitionNativeSubjectStateV2_9.REJECTED_MISSING_PLACEMENT_DOMAIN,
            placement.reasons,
        )
    if (
        support.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
        and placement.availability
        is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
    ):
        return CompetitionNativeSubjectStateV2_9.ELIGIBLE_RECEPTACLE_DOMAIN, ()
    if (
        support.support_kind is CompetitionNativeSupportKindV2_9.FLOOR
        and placement.availability
        is CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
    ):
        return CompetitionNativeSubjectStateV2_9.ELIGIBLE_FLOOR_INNER_DOMAIN, ()
    return (
        CompetitionNativeSubjectStateV2_9.REJECTED_INVALID_SOURCE_FACT,
        ("object:invalid_placement_binding",),
    )


def _candidate_identity(
    policy_sha256: str,
    source: CompetitionNativeSourceRefV2_9,
    capture: CompetitionNativeSourceCaptureV2_9,
    subject_id: str,
    reference_id: str,
    relation: Relation,
    support_kind: CompetitionNativeSupportKindV2_9,
) -> str:
    payload = {
        "policy_sha256": policy_sha256,
        "reference_id": reference_id,
        "relation_before": relation.value,
        "scene_id": source.scene_id,
        "source_capture_sha256": capture.source_capture_sha256,
        "source_id": source.source_id,
        "split": source.split,
        "subject_id": subject_id,
        "support_kind": support_kind.value,
    }
    return "candidate-" + canonical_sha256_v2(payload, domain=_CANDIDATE_ID_HASH_DOMAIN)


def _candidate_record(
    *,
    candidate_id: str,
    source: CompetitionNativeSourceRefV2_9,
    capture: CompetitionNativeSourceCaptureV2_9,
    subject,
    reference,
    relation: Relation,
    support_kind: CompetitionNativeSupportKindV2_9,
    state: CompetitionNativeCandidateStateV2_9,
    reasons: tuple[str, ...],
    selection_index: int | None = None,
) -> CompetitionNativeCandidateInventoryV2_9:
    return CompetitionNativeCandidateInventoryV2_9(
        candidate_id=candidate_id,
        source_id=source.source_id,
        scene_id=source.scene_id,
        split=source.split,
        source_capture_sha256=capture.source_capture_sha256,
        subject_id=subject.object_id,
        subject_name=subject.name,
        subject_category=subject.category,
        reference_id=reference.object_id,
        reference_name=reference.name,
        reference_category=reference.category,
        relation_before=relation,
        support_kind=support_kind,
        state=state,
        selection_index=selection_index,
        reasons=tuple(sorted(set(reasons))),
    )


def _select_candidates(
    policy: RosterPolicy,
    candidates: list[CompetitionNativeCandidateInventoryV2_9],
    sources: dict[str, CompetitionNativeSourceRefV2_9],
) -> tuple[
    tuple[CompetitionNativeCandidateInventoryV2_9, ...],
    CompetitionNativeCandidateRosterManifestV2_9,
]:
    pending = {
        item.candidate_id: item
        for item in candidates
        if item.reasons == ("candidate:selection_pending",)
    }
    grouped: dict[
        tuple[str, str, str, str, str],
        deque[CompetitionNativeCandidateInventoryV2_9],
    ] = {}
    for item in pending.values():
        key = (
            item.relation_before.value,
            item.subject_category,
            item.reference_category,
            item.support_kind.value,
            item.scene_id,
        )
        grouped.setdefault(key, deque()).append(item)
    for key, values in tuple(grouped.items()):
        grouped[key] = deque(
            sorted(
                values,
                key=lambda item: _digest_order(policy.seed, item.candidate_id),
            )
        )

    selected: list[CompetitionNativeCandidateInventoryV2_9] = []
    states: dict[str, CompetitionNativeCandidateInventoryV2_9] = {
        item.candidate_id: item
        for item in candidates
        if item.candidate_id not in pending
    }
    relation_counts: Counter[Relation] = Counter()
    scene_counts: Counter[str] = Counter()
    subject_counts: Counter[tuple[str, str]] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    relations = tuple(sorted(Relation, key=lambda item: item.value))
    for relation in relations:
        keys = sorted(
            (key for key in grouped if key[0] == relation.value),
            key=lambda key: _digest_order(policy.seed, relation.value, key),
        )
        while any(grouped[key] for key in keys):
            for key in keys:
                if not grouped[key]:
                    continue
                item = grouped[key].popleft()
                subject_key = (item.source_id, item.subject_id)
                category_pair = (item.subject_category, item.reference_category)
                allowed = (
                    len(selected) < policy.max_requests_total
                    and relation_counts[relation] < policy.target_requests_per_relation
                    and scene_counts[item.scene_id] < policy.max_requests_per_scene
                    and subject_counts[subject_key] < policy.max_requests_per_subject
                    and pair_counts[category_pair]
                    < policy.max_requests_per_category_pair
                )
                if allowed:
                    frozen = item.model_copy(
                        update={
                            "state": CompetitionNativeCandidateStateV2_9.SELECTED,
                            "selection_index": len(selected),
                            "reasons": (),
                        }
                    )
                    frozen = CompetitionNativeCandidateInventoryV2_9.model_validate(
                        frozen.model_dump(mode="python"), strict=True
                    )
                    selected.append(frozen)
                    states[item.candidate_id] = frozen
                    relation_counts[relation] += 1
                    scene_counts[item.scene_id] += 1
                    subject_counts[subject_key] += 1
                    pair_counts[category_pair] += 1
                else:
                    states[item.candidate_id] = item.model_copy(
                        update={"reasons": ("candidate:policy_cap",)}
                    )

    requests: list[CompetitionNativeSelectedRequestV2_9] = []
    for item in selected:
        source = sources[item.source_id]
        request_digest = canonical_sha256_v2(
            {
                "candidate_id": item.candidate_id,
                "selection_index": item.selection_index,
            },
            domain=_REQUEST_ID_HASH_DOMAIN,
        )
        requests.append(
            CompetitionNativeSelectedRequestV2_9(
                request_id="request-" + request_digest,
                candidate_id=item.candidate_id,
                selection_index=item.selection_index,
                source_id=item.source_id,
                source_locator_sha256=source.source_locator_sha256,
                source_capture_sha256=item.source_capture_sha256,
                scene_id=item.scene_id,
                split=item.split,
                subject_id=item.subject_id,
                subject_name=item.subject_name,
                subject_category=item.subject_category,
                reference_id=item.reference_id,
                reference_name=item.reference_name,
                reference_category=item.reference_category,
                relation_before=item.relation_before,
                relation_after=item.relation_before.opposite,
                support_kind=item.support_kind,
            )
        )
    manifest = CompetitionNativeCandidateRosterManifestV2_9(
        campaign_id=policy.campaign_id,
        policy_sha256=policy.policy_sha256,
        requests=tuple(requests),
    )
    return tuple(sorted(states.values(), key=lambda item: item.candidate_id)), manifest


def _compile_competition_native_candidate_roster(
    policy: RosterPolicy,
    source_records: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...],
    *,
    require_scene_unique_referents: bool,
    target_reachability_surface_evidence: tuple[SourceSurfaceEvidence, ...]
    | None = None,
    target_reachability_camera_evidence: tuple[SourceCameraEvidence, ...] | None = None,
) -> tuple[
    _RosterCore,
    tuple[CandidateTargetReachability, ...],
]:
    """Compile all terminal records without a runtime or downstream outcome."""

    if type(source_records) is not tuple:
        raise TypeError("candidate roster source_records must be an exact tuple")
    if type(policy) is not RosterPolicy or require_scene_unique_referents is not True:
        raise TypeError("candidate roster compiler policy dispatch is invalid")
    use_target_reachability = True
    if (
        target_reachability_surface_evidence is None
        or target_reachability_camera_evidence is None
    ):
        raise TypeError("candidate roster target reachability dispatch is invalid")
    checked_policy = RosterPolicy.model_validate(
        policy.model_dump(mode="python"), strict=True
    )
    checked_records = tuple(
        CompetitionNativeSourceCaptureOutcomeV2_9.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in source_records
    )
    if tuple(item.source for item in checked_records) != checked_policy.sources:
        raise ValueError("source outcomes do not exactly match the frozen policy")

    total_candidates = 0
    for record in checked_records:
        if record.capture is None:
            continue
        if (
            record.capture.runtime_identity.width != checked_policy.width
            or record.capture.runtime_identity.height != checked_policy.height
            or record.capture.runtime_identity.seed != checked_policy.seed
        ):
            raise ValueError("source capture runtime does not match roster policy")
        validate_competition_native_runtime_source_lineage_v2_9(
            record.source, record.capture.runtime_identity
        )
        count = len(record.capture.scene.objects)
        if count > checked_policy.max_objects_per_scene:
            raise ValueError("source object count exceeds max_objects_per_scene")
        total_candidates += count * (count - 1) * len(Relation)
    if total_candidates > checked_policy.max_candidates_total:
        raise ValueError("candidate enumeration exceeds max_candidates_total")

    object_inventory: list[CompetitionNativeObjectInventoryV2_9] = []
    candidates: list[CompetitionNativeCandidateInventoryV2_9] = []
    rejections: list[CompetitionNativeRosterRejectionV2_9] = []
    target_reachability: list[CandidateTargetReachability] = []
    sources = {item.source_id: item for item in checked_policy.sources}
    surface_by_source = {
        item.source_id: item for item in target_reachability_surface_evidence or ()
    }
    camera_by_source = {
        item.source_id: item for item in target_reachability_camera_evidence or ()
    }
    engine = RelationEngine()
    eligible_states = {
        CompetitionNativeSubjectStateV2_9.ELIGIBLE_RECEPTACLE_DOMAIN,
        CompetitionNativeSubjectStateV2_9.ELIGIBLE_FLOOR_INNER_DOMAIN,
    }
    for record in checked_records:
        if record.capture is None:
            rejections.append(
                _rejection(
                    stage="source",
                    source_id=record.source.source_id,
                    scene_id=record.source.scene_id,
                    reasons=record.reasons,
                )
            )
            continue
        capture = record.capture
        scene = capture.scene
        if use_target_reachability:
            try:
                prepared_target_source = (
                    prepare_competition_native_target_reachability_source_v2_9_4(
                        capture,
                        surface_by_source[record.source.source_id],
                        camera_by_source[record.source.source_id],
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "candidate target reachability evidence is absent"
                ) from error
        else:
            prepared_target_source = None
        category_counts = Counter(item.category for item in scene.objects)
        support_by_id = {item.object_id: item for item in capture.support_facts}
        placement_by_id = {item.object_id: item for item in capture.placement_facts}
        dependency_graph = _support_dependency_graph(support_by_id)
        support_parent_ids = {
            parent_id
            for parent_ids in dependency_graph.values()
            for parent_id in parent_ids
        }
        inventory_by_id: dict[str, CompetitionNativeObjectInventoryV2_9] = {}
        for item in scene.objects:
            support = support_by_id.get(item.object_id)
            placement = placement_by_id.get(item.object_id)
            state, reasons = _subject_terminal(
                scene,
                item.object_id,
                support,
                placement,
                has_dependent_child=item.object_id in support_parent_ids,
            )
            reference_eligible = bool(
                item.request_eligible
                and _valid_geometry(scene, item.object_id)
                and _qualifying_main_view(scene, item.object_id)
            )
            support_kind = (
                CompetitionNativeSupportKindV2_9.UNKNOWN
                if support is None
                else support.support_kind
            )
            placement_sha256 = (
                "0" * 64 if placement is None else placement.placement_sha256
            )
            identity_payload = {
                "object_id": item.object_id,
                "scene_id": scene.scene_id,
                "source_capture_sha256": capture.source_capture_sha256,
                "source_id": record.source.source_id,
                "split": record.source.split,
            }
            inventory = CompetitionNativeObjectInventoryV2_9(
                inventory_id="object-"
                + canonical_sha256_v2(identity_payload, domain=_OBJECT_ID_HASH_DOMAIN),
                source_id=record.source.source_id,
                scene_id=scene.scene_id,
                split=record.source.split,
                source_capture_sha256=capture.source_capture_sha256,
                object_id=item.object_id,
                object_name=item.name,
                category=item.category,
                subject_state=state,
                reference_eligible=reference_eligible,
                support_kind=support_kind,
                support_object_id=None
                if support is None
                else support.support_object_id,
                placement_sha256=placement_sha256,
                reasons=reasons,
            )
            inventory_by_id[item.object_id] = inventory
            object_inventory.append(inventory)
            if state not in eligible_states:
                rejections.append(
                    _rejection(
                        stage="object_subject",
                        source_id=record.source.source_id,
                        scene_id=scene.scene_id,
                        object_id=item.object_id,
                        reasons=reasons,
                    )
                )
            if not reference_eligible:
                rejections.append(
                    _rejection(
                        stage="object_reference",
                        source_id=record.source.source_id,
                        scene_id=scene.scene_id,
                        object_id=item.object_id,
                        reasons=("object:reference_ineligible",),
                    )
                )

        for subject in scene.objects:
            subject_inventory = inventory_by_id[subject.object_id]
            for reference in scene.objects:
                if subject.object_id == reference.object_id:
                    continue
                reference_inventory = inventory_by_id[reference.object_id]
                for relation in Relation:
                    candidate_id = _candidate_identity(
                        checked_policy.policy_sha256,
                        record.source,
                        capture,
                        subject.object_id,
                        reference.object_id,
                        relation,
                        subject_inventory.support_kind,
                    )
                    if (
                        require_scene_unique_referents
                        and category_counts[subject.category] != 1
                    ):
                        state = CompetitionNativeCandidateStateV2_9.REJECTED_AMBIGUOUS
                        reasons = ("candidate:subject_category_not_scene_unique",)
                    elif (
                        require_scene_unique_referents
                        and category_counts[reference.category] != 1
                    ):
                        state = CompetitionNativeCandidateStateV2_9.REJECTED_AMBIGUOUS
                        reasons = ("candidate:reference_category_not_scene_unique",)
                    elif subject_inventory.subject_state not in eligible_states:
                        state = CompetitionNativeCandidateStateV2_9.REJECTED_SUBJECT
                        reasons = ("candidate:subject_ineligible",)
                    elif not reference_inventory.reference_eligible:
                        state = CompetitionNativeCandidateStateV2_9.REJECTED_REFERENCE
                        reasons = ("candidate:reference_ineligible",)
                    elif _depends_on_support(
                        dependency_graph,
                        subject.object_id,
                        reference.object_id,
                    ) or _depends_on_support(
                        dependency_graph,
                        reference.object_id,
                        subject.object_id,
                    ):
                        state = CompetitionNativeCandidateStateV2_9.REJECTED_DEPENDENCY
                        reasons = ("candidate:support_dependency",)
                    else:
                        observed = engine.observe(
                            scene,
                            subject.object_id,
                            reference.object_id,
                            relation,
                            "main",
                        )
                        if observed.status is SolverStatus.NOT_VISIBLE:
                            state = (
                                CompetitionNativeCandidateStateV2_9.REJECTED_NOT_VISIBLE
                            )
                            reasons = ("candidate:not_visible",)
                        elif observed.status is SolverStatus.AMBIGUOUS:
                            state = (
                                CompetitionNativeCandidateStateV2_9.REJECTED_AMBIGUOUS
                            )
                            reasons = ("candidate:ambiguous",)
                        elif (
                            observed.status is not SolverStatus.SUCCESS
                            or not observed.satisfied
                        ):
                            state = CompetitionNativeCandidateStateV2_9.REJECTED_RELATION_NOT_OBSERVED
                            reasons = ("candidate:relation_not_observed",)
                        else:
                            if (
                                use_target_reachability
                                and subject_inventory.support_kind
                                is CompetitionNativeSupportKindV2_9.RECEPTACLE
                            ):
                                if prepared_target_source is None:
                                    raise RuntimeError(
                                        "target reachability prepared source is absent"
                                    )
                                reachability = derive_competition_native_candidate_target_reachability_from_prepared_v2_9_4(
                                    candidate_id=candidate_id,
                                    prepared_source=prepared_target_source,
                                    subject_id=subject.object_id,
                                    reference_id=reference.object_id,
                                    relation_before=relation,
                                )
                                target_reachability.append(reachability)
                                if (
                                    reachability.status
                                    is TargetReachabilityStatus.UNREACHABLE
                                ):
                                    state = CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE
                                    reasons = (
                                        "candidate:no_relation_after_native_candidate",
                                    )
                                else:
                                    state = CompetitionNativeCandidateStateV2_9.REJECTED_POLICY_CAP
                                    reasons = ("candidate:selection_pending",)
                            else:
                                state = CompetitionNativeCandidateStateV2_9.REJECTED_POLICY_CAP
                                reasons = ("candidate:selection_pending",)
                    candidates.append(
                        _candidate_record(
                            candidate_id=candidate_id,
                            source=record.source,
                            capture=capture,
                            subject=subject,
                            reference=reference,
                            relation=relation,
                            support_kind=subject_inventory.support_kind,
                            state=state,
                            reasons=reasons,
                        )
                    )

    frozen_candidates, manifest = _select_candidates(
        checked_policy, candidates, sources
    )
    for item in frozen_candidates:
        if item.state is not CompetitionNativeCandidateStateV2_9.SELECTED:
            rejections.append(
                _rejection(
                    stage="candidate",
                    source_id=item.source_id,
                    scene_id=item.scene_id,
                    candidate_id=item.candidate_id,
                    reasons=item.reasons,
                )
            )
    object_inventory_tuple = tuple(
        sorted(object_inventory, key=lambda item: item.inventory_id)
    )
    rejections_tuple = tuple(sorted(rejections, key=lambda item: item.rejection_id))
    subject_counts = Counter(item.subject_state for item in object_inventory_tuple)
    candidate_counts = Counter(item.state for item in frozen_candidates)
    relation_counts = Counter(item.relation_before for item in manifest.requests)
    stage_counts = Counter(item.stage for item in rejections_tuple)
    eligible_count = sum(subject_counts[state] for state in eligible_states)
    summary = RosterSummary(
        campaign_id=checked_policy.campaign_id,
        policy_sha256=checked_policy.policy_sha256,
        manifest_sha256=manifest.manifest_sha256,
        source_count=len(checked_records),
        accepted_source_count=sum(
            item.status == "accepted" for item in checked_records
        ),
        rejected_source_count=sum(
            item.status == "rejected" for item in checked_records
        ),
        object_count=len(object_inventory_tuple),
        eligible_subject_count=eligible_count,
        rejected_subject_count=len(object_inventory_tuple) - eligible_count,
        candidate_count=len(frozen_candidates),
        selected_request_count=len(manifest.requests),
        rejected_candidate_count=len(frozen_candidates) - len(manifest.requests),
        candidate_state_counts=tuple(
            CompetitionNativeCandidateStateCountV2_9(state=state, count=count)
            for state, count in sorted(
                candidate_counts.items(), key=lambda item: item[0].value
            )
            if count
        ),
        relation_selected_counts=tuple(
            CompetitionNativeRelationCountV2_9(relation=relation, count=count)
            for relation, count in sorted(
                relation_counts.items(), key=lambda item: item[0].value
            )
            if count
        ),
        rejection_stage_counts=tuple(
            CompetitionNativeStageCountV2_9(stage=stage, count=count)
            for stage, count in sorted(stage_counts.items())
            if count
        ),
    )
    return (
        _RosterCore(
            policy=checked_policy,
            scene_inventory=checked_records,
            object_inventory=object_inventory_tuple,
            candidate_inventory=frozen_candidates,
            request_manifest=manifest,
            rejections=rejections_tuple,
            summary=summary,
        ),
        tuple(sorted(target_reachability, key=lambda item: item.candidate_id)),
    )


def _editable_camera_capture_roster_v2_9_4(
    capture: CompetitionNativeSourceCaptureV2_9,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
]:
    """Rebuild every patch-certified subject and every explicit exclusion."""

    support_by_id = {item.object_id: item for item in capture.support_facts}
    placement_by_id = {item.object_id: item for item in capture.placement_facts}
    try:
        source_pairs = tuple(
            sorted(
                (item.object_id, support_by_id[item.object_id].support_object_id)
                for item in capture.scene.objects
                if item.movable
                and support_by_id[item.object_id].support_kind
                is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and support_by_id[item.object_id].support_object_id is not None
            )
        )
        certified_pairs = []
        roster_entries = []
        for subject_object_id, support_object_id in source_pairs:
            placement = placement_by_id[subject_object_id]
            if (
                placement.availability
                is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
                and placement.support_object_id == support_object_id
            ):
                certified_pairs.append((subject_object_id, support_object_id))
                roster_entries.append(
                    CompetitionNativeCameraPlacementRosterEntryV2_9_4(
                        subject_object_id=subject_object_id,
                        support_object_id=support_object_id,
                        positions=tuple(
                            CompetitionNativeCameraPlacementPositionV2_9_4(
                                x=position.x,
                                y=position.y,
                                z=position.z,
                            )
                            for position in placement.native_positions
                        ),
                    )
                )
                continue
            if not (
                placement.availability
                is CompetitionNativePlacementAvailabilityV2_9.MISSING
                and placement.support_object_id == support_object_id
                and placement.reasons
                == ("source_capture:receptacle_surface_patch_missing",)
            ):
                raise ValueError("editable camera placement roster is incomplete")
    except KeyError as error:
        raise ValueError("editable camera placement roster is incomplete") from error
    return tuple(certified_pairs), tuple(roster_entries)


def _verify_competition_native_camera_evidence_capture_v2_9_3(
    capture: CompetitionNativeSourceCaptureV2_9,
    evidence: SourceCameraEvidence,
    policy: CameraPolicy,
) -> SourceCameraEvidence:
    """Rebuild the trusted bank and close one persisted camera row."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("camera evidence source capture must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"), strict=True
    )
    if type(evidence) is not SourceCameraEvidence:
        raise TypeError("camera evidence row must be exact")
    checked_evidence = SourceCameraEvidence.model_validate(
        evidence.model_dump(mode="python"), strict=True
    )
    if type(policy) is not CameraPolicy:
        raise TypeError("camera evidence policy must be exact")
    checked_policy = CameraPolicy.model_validate(
        policy.model_dump(mode="python"), strict=True
    )

    if type(checked_evidence.score) is CompetitionNativeCameraScoreV2_9_4:
        pairs, _placement_roster = _editable_camera_capture_roster_v2_9_4(
            checked_capture
        )
    else:
        support_by_id = {item.object_id: item for item in checked_capture.support_facts}
        pairs = tuple(
            sorted(
                (item.object_id, support_by_id[item.object_id].support_object_id)
                for item in checked_capture.scene.objects
                if item.movable
                and support_by_id[item.object_id].support_kind
                is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and support_by_id[item.object_id].support_object_id is not None
            )
        )
    requested_fallback = AI2ThorAgentPose(
        position=AI2ThorNativePosition(
            x=checked_evidence.requested_pose.x,
            y=checked_evidence.requested_pose.y,
            z=checked_evidence.requested_pose.z,
        ),
        yaw_degrees=checked_evidence.requested_pose.yaw_degrees,
        horizon_degrees=checked_evidence.requested_pose.horizon_degrees,
        standing=checked_evidence.requested_pose.standing,
    )
    fallback = (
        requested_fallback
        if not pairs
        else AI2ThorAgentPose(
            position=AI2ThorNativePosition(
                x=checked_capture.reachable_positions[0].x,
                y=checked_capture.reachable_positions[0].y,
                z=checked_capture.reachable_positions[0].z,
            ),
            yaw_degrees=0.0,
            horizon_degrees=0.0,
            standing=True,
        )
    )
    pose_bank = build_competition_native_camera_pose_bank_v2_9_3(
        checked_capture.scene,
        pairs,
        tuple(
            AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
            for item in checked_capture.reachable_positions
        ),
        fallback,
        policy=checked_policy,
    )
    if type(checked_evidence.score) is CompetitionNativeCameraScoreV2_9_4:
        selected_index = select_competition_native_camera_score_index_v2_9_4(
            checked_evidence.pose_scores
        )
    else:
        selected_index = select_competition_native_camera_score_index_v2_9_3(
            checked_evidence.pose_scores
        )
    expected_score = score_competition_native_camera_capture_scene_v2_9_3(
        checked_capture,
        checked_evidence,
        checked_capture.scene,
    )
    verify_competition_native_camera_observation_binding_v2_9_3(
        pose_bank[selected_index],
        checked_evidence.observed_pose,
        checked_evidence.observed_native_camera_position,
        checked_capture.scene.camera_by_id("main"),
    )
    verify_competition_native_solver_camera_binding_v2_9_3(
        checked_policy,
        pose_bank[selected_index],
        checked_capture.scene.camera_by_id("main"),
    )
    runtime_identity_sha256 = canonical_sha256_v2(
        checked_capture.runtime_identity,
        domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
    )
    if (
        checked_evidence.source_id != checked_capture.source.source_id
        or checked_evidence.scene_id != checked_capture.source.scene_id
        or checked_evidence.source_locator_sha256
        != checked_capture.source.source_locator_sha256
        or checked_evidence.runtime_identity_sha256 != runtime_identity_sha256
        or checked_evidence.source_capture_sha256
        != checked_capture.source_capture_sha256
        or checked_evidence.policy_sha256 != checked_policy.policy_sha256
        or checked_evidence.pose_bank_sha256
        != competition_native_camera_pose_bank_sha256_v2_9_3(pose_bank)
        or checked_evidence.pose_bank_count != len(pose_bank)
        or selected_index != checked_evidence.selected_pose_index
        or pose_bank[selected_index] != checked_evidence.requested_pose
        or checked_evidence.score != expected_score
        or checked_evidence.camera != checked_capture.scene.camera_by_id("main")
        or checked_evidence.rgb_png_sha256 != checked_capture.rgb_png_sha256
        or checked_evidence.depth_npy_sha256 != checked_capture.depth_npy_sha256
        or checked_evidence.instance_png_sha256 != checked_capture.instance_png_sha256
        or checked_evidence.pointcloud_ply_sha256
        != checked_capture.pointcloud_ply_sha256
        or checked_evidence.is_scene_at_rest is not checked_capture.is_scene_at_rest
    ):
        raise ValueError("camera evidence does not close source capture")
    return checked_evidence


def score_competition_native_camera_capture_scene_v2_9_3(
    capture: CompetitionNativeSourceCaptureV2_9,
    evidence: SourceCameraEvidence,
    scene: Scene,
) -> CompetitionNativeCameraScoreFamilyV2_9_3:
    """Recompute the versioned camera score from frozen capture facts."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("camera score source capture must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"), strict=True
    )
    if type(evidence) is not SourceCameraEvidence:
        raise TypeError("camera score evidence must be exact")
    checked_evidence = SourceCameraEvidence.model_validate(
        evidence.model_dump(mode="python"), strict=True
    )
    if type(scene) is not Scene:
        raise TypeError("camera score scene must be exact")
    checked_scene = Scene.model_validate(scene.model_dump(mode="python"), strict=True)
    if type(checked_evidence.score) is not CompetitionNativeCameraScoreV2_9_4:
        return score_competition_native_camera_scene_v2_9_3(checked_scene)

    _pairs, roster = _editable_camera_capture_roster_v2_9_4(checked_capture)
    return score_competition_native_editable_camera_scene_v2_9_4(
        checked_scene,
        roster,
    )


def verify_competition_native_camera_evidence_capture_v2_9_3(
    capture: CompetitionNativeSourceCaptureV2_9,
    evidence: SourceCameraEvidence,
    policy: CameraPolicy,
) -> SourceCameraEvidence:
    """Public exact verifier for either persisted camera-score generation."""

    return _verify_competition_native_camera_evidence_capture_v2_9_3(
        capture,
        evidence,
        policy,
    )


def compile_roster(
    policy: RosterPolicy,
    source_records: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...],
    surface_evidence: tuple[SourceSurfaceEvidence, ...],
    camera_evidence: tuple[SourceCameraEvidence, ...],
) -> RosterCompilation:
    """Compile target-reachable source candidates before deterministic quotas."""

    if type(policy) is not RosterPolicy:
        raise TypeError("candidate roster policy must be exact")
    if type(source_records) is not tuple:
        raise TypeError("candidate roster source_records must be an exact tuple")
    if type(surface_evidence) is not tuple or any(
        type(item) is not SourceSurfaceEvidence for item in surface_evidence
    ):
        raise TypeError("candidate roster surface evidence rows must be exact")
    if type(camera_evidence) is not tuple or any(
        type(item) is not SourceCameraEvidence for item in camera_evidence
    ):
        raise TypeError("candidate roster camera evidence rows must be exact")
    capture_by_source_id = {
        item.source.source_id: item.capture
        for item in source_records
        if item.status == "accepted" and item.capture is not None
    }
    checked_surface = tuple(
        verify_source_surface_evidence(capture_by_source_id[item.source_id], item)
        if item.source_id in capture_by_source_id
        else item
        for item in surface_evidence
    )
    checked_camera = tuple(
        _verify_competition_native_camera_evidence_capture_v2_9_3(
            capture_by_source_id[item.source_id],
            item,
            policy.camera_policy,
        )
        if item.source_id in capture_by_source_id
        else item
        for item in camera_evidence
    )
    base, target_reachability = _compile_competition_native_candidate_roster(
        policy,
        source_records,
        require_scene_unique_referents=True,
        target_reachability_surface_evidence=checked_surface,
        target_reachability_camera_evidence=checked_camera,
    )
    return RosterCompilation(
        **base.model_dump(mode="python"),
        surface_evidence=checked_surface,
        camera_evidence=checked_camera,
        target_reachability=target_reachability,
    )


__all__ = (
    "compile_roster",
    "score_competition_native_camera_capture_scene_v2_9_3",
    "verify_competition_native_camera_evidence_capture_v2_9_3",
)
