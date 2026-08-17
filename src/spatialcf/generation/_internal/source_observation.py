"""Read-only source observation capture for the current roster."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass

from pydantic import model_validator

from spatialcf.adapters.ai2thor import (
    AI2ThorAdapter,
    AI2ThorAgentPose,
    AI2ThorNativePosition,
    AI2ThorNativeReturnError,
    AI2ThorObservation,
    AI2ThorReceptacleSpawnMap,
    AI2ThorSettlementTimeout,
    bind_ai2thor_reachable_positions,
    build_navigation_feasibility_map,
    build_receptacle_support_position_region,
    canonicalize_ai2thor_reachable_positions,
)
from spatialcf.domain.models import Scene, SubjectPositionRegion
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    CompetitionNativeCameraPlacementPositionV2_9_4,
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraPolicyV2_9_3,
    CompetitionNativeCameraPoseV2_9_3,
    CompetitionNativeCameraScoreV2_9_3,
    CompetitionNativeCameraScoreV2_9_4,
    CompetitionNativeSourceCameraEvidenceV2_9_3,
    SourceCameraEvidence,
    build_competition_native_camera_policy_v2_9_3,
    build_competition_native_camera_pose_bank_v2_9_3,
    build_competition_native_source_camera_evidence_v2_9_3,
    build_competition_native_source_camera_evidence_v2_9_4,
    score_competition_native_editable_camera_application_v2_9_4,
    score_competition_native_source_camera_application_v2_9_3,
    select_competition_native_camera_score_index_v2_9_3,
    select_competition_native_camera_score_index_v2_9_4,
)
from spatialcf.generation._internal.evidence.surface import (
    CompetitionNativeSourceSurfaceEvidenceV2_9_2,
    SourceSurfaceEvidence,
    build_competition_native_source_surface_evidence_v2_9_2,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeFloorEnvelopeV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativePositionV2_9,
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceCaptureOutcomeV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSourceRefV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportFactV2_9,
    CompetitionNativeSupportKindV2_9,
    build_competition_native_source_capture_v2_9,
    build_competition_native_subject_placement_fact_v2_9,
    normalize_competition_native_source_scene_v2_9,
    validate_competition_native_floor_envelope_v2_9,
    validate_competition_native_runtime_source_lineage_v2_9,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeSourceCaptureOutcomeV2_9 as SourceCaptureOutcome,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeSourceRefV2_9 as SourceRef,
)
from spatialcf.generation.capture.plan import CaptureSettings

_EXPECTED_SOURCE_ERRORS = (
    AI2ThorNativeReturnError,
    AI2ThorSettlementTimeout,
    RuntimeError,
    ValueError,
)
_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain:3"
)
_COLLISION_SAFE_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.2m:4"
)
_CONTACT_MARGIN_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m:5"
)
_RESET_PER_POSE_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m-reset-per-pose:6"
)
_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m:7"
)
_PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused:8"
)
_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused-final-settle:9"
)


class CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2(V2Model):
    """The unchanged capture outcome plus evidence from its same native query."""

    outcome: CompetitionNativeSourceCaptureOutcomeV2_9
    surface_evidence: CompetitionNativeSourceSurfaceEvidenceV2_9_2 | None

    @model_validator(mode="after")
    def validate_capture_evidence(self):
        if self.outcome.capture is None:
            if self.surface_evidence is not None:
                raise ValueError(
                    "rejected source capture cannot carry surface evidence"
                )
        elif (
            self.surface_evidence is None
            or self.surface_evidence.source_id != self.outcome.source.source_id
            or self.surface_evidence.scene_id != self.outcome.source.scene_id
            or self.surface_evidence.source_capture_sha256
            != self.outcome.capture.source_capture_sha256
        ):
            raise ValueError("accepted source capture surface evidence is not closed")
        return self


class CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(V2Model):
    """One terminal source outcome with both required sibling evidence rows."""

    outcome: CompetitionNativeSourceCaptureOutcomeV2_9
    surface_evidence: CompetitionNativeSourceSurfaceEvidenceV2_9_2 | None
    camera_evidence: CompetitionNativeSourceCameraEvidenceV2_9_3 | None

    @model_validator(mode="after")
    def validate_capture_evidence(self):
        capture = self.outcome.capture
        if capture is None:
            if self.surface_evidence is not None or self.camera_evidence is not None:
                raise ValueError("rejected source capture cannot carry evidence")
            return self
        if self.surface_evidence is None or self.camera_evidence is None:
            raise ValueError("accepted source capture requires both evidence rows")
        if (
            self.surface_evidence.source_id != self.outcome.source.source_id
            or self.surface_evidence.scene_id != self.outcome.source.scene_id
            or self.surface_evidence.source_capture_sha256
            != capture.source_capture_sha256
            or self.camera_evidence.source_id != self.outcome.source.source_id
            or self.camera_evidence.scene_id != self.outcome.source.scene_id
            or self.camera_evidence.source_locator_sha256
            != self.outcome.source.source_locator_sha256
            or self.camera_evidence.source_capture_sha256
            != capture.source_capture_sha256
            or self.camera_evidence.camera != capture.scene.camera_by_id("main")
            or self.camera_evidence.rgb_png_sha256 != capture.rgb_png_sha256
            or self.camera_evidence.depth_npy_sha256 != capture.depth_npy_sha256
            or self.camera_evidence.instance_png_sha256 != capture.instance_png_sha256
            or self.camera_evidence.pointcloud_ply_sha256
            != capture.pointcloud_ply_sha256
            or self.camera_evidence.is_scene_at_rest is not capture.is_scene_at_rest
        ):
            raise ValueError("accepted source capture evidence is not closed")
        return self


def _rejected(
    source: CompetitionNativeSourceRefV2_9,
    reason: str,
) -> CompetitionNativeSourceCaptureOutcomeV2_9:
    return CompetitionNativeSourceCaptureOutcomeV2_9(
        source=source,
        status="rejected",
        capture=None,
        reasons=(reason,),
    )


def _support_fact(value) -> CompetitionNativeSupportFactV2_9:
    return CompetitionNativeSupportFactV2_9(
        scene_id=value.scene_id,
        object_id=value.object_id,
        object_name=value.object_name,
        native_object_id=value.native_object_id,
        raw_parent_object_ids=tuple(sorted(set(value.raw_parent_object_ids))),
        structural_parent_object_ids=tuple(
            sorted(set(value.structural_parent_object_ids))
        ),
        domain_parent_object_ids=tuple(sorted(set(value.domain_parent_object_ids))),
        support_kind=CompetitionNativeSupportKindV2_9(value.support_kind.value),
        support_object_id=value.support_object_id,
        floor_object_id=value.floor_object_id,
    )


def _positions(values) -> tuple[CompetitionNativePositionV2_9, ...]:
    return tuple(
        CompetitionNativePositionV2_9(x=item.x, y=item.y, z=item.z)
        for item in sorted(values, key=lambda item: (item.x, item.z, item.y))
    )


def _optional_receptacle_position_region(
    scene: Scene,
    spawn_map: AI2ThorReceptacleSpawnMap,
) -> SubjectPositionRegion | None:
    """Attach new trigger-grid evidence without narrowing legacy captures."""

    if not spawn_map.surface_patches:
        return None
    region = build_receptacle_support_position_region(scene, spawn_map)
    return region if region.components else None


def _missing_placement(
    object_id: str,
    support: CompetitionNativeSupportFactV2_9,
    reason: str,
) -> CompetitionNativeSubjectPlacementFactV2_9:
    return build_competition_native_subject_placement_fact_v2_9(
        object_id=object_id,
        availability=CompetitionNativePlacementAvailabilityV2_9.MISSING,
        support_kind=support.support_kind,
        support_object_id=support.support_object_id,
        floor_object_id=support.floor_object_id,
        reasons=(reason,),
    )


def _capture_competition_native_source_with_spawn_maps_v2_9(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> tuple[
    CompetitionNativeSourceCaptureOutcomeV2_9,
    tuple[AI2ThorReceptacleSpawnMap, ...],
]:
    """Load, settle, and capture one source without editing an object."""

    if type(source) is not CompetitionNativeSourceRefV2_9:
        raise TypeError("native source ref must be exact")
    try:
        loaded = adapter.load_scene(source.scene_id)
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:load_failed"), ()
    try:
        settlement = adapter.settle_scene_observed(
            loaded, max_pass_steps=max_settlement_steps
        )
    except AI2ThorSettlementTimeout:
        return _rejected(source, "source_capture:settlement_timeout"), ()
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:settlement_failed"), ()
    scene = settlement.observed_scene
    try:
        runtime = adapter.runtime_identity()
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:runtime_identity_failed"), ()
    try:
        raw_support_facts = adapter.native_support_facts(scene)
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:support_snapshot_failed"), ()
    try:
        normalized_scene = normalize_competition_native_source_scene_v2_9(scene)
        support_facts = tuple(
            sorted(
                (_support_fact(item) for item in raw_support_facts),
                key=lambda item: item.object_id,
            )
        )
        object_ids = tuple(item.object_id for item in normalized_scene.objects)
        if tuple(item.object_id for item in support_facts) != object_ids:
            raise ValueError("support fact roster does not match scene objects")
        if any(
            item.scene_id != normalized_scene.scene_id
            or item.object_name != normalized_scene.object_by_id(item.object_id).name
            for item in support_facts
        ):
            raise ValueError("support facts do not bind the settled scene")
        runtime_identity = CompetitionNativeRuntimeIdentityV2_9(**asdict(runtime))
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:support_snapshot_invalid"), ()
    try:
        validate_competition_native_runtime_source_lineage_v2_9(
            source, runtime_identity
        )
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:runtime_lineage_mismatch"), ()

    support_by_id = {item.object_id: item for item in support_facts}
    floor_subjects = tuple(
        item
        for item in normalized_scene.objects
        if item.movable
        and support_by_id[item.object_id].support_kind
        is CompetitionNativeSupportKindV2_9.FLOOR
    )
    floor = None
    floor_reason = None
    reachable = ()
    reachable_reason = None
    if floor_subjects:
        try:
            captured_floor = adapter.conservative_floor_envelope(
                scene, clearance_m=floor_clearance_m
            )
        except _EXPECTED_SOURCE_ERRORS:
            floor_reason = "source_capture:floor_envelope_missing"
        else:
            try:
                floor = CompetitionNativeFloorEnvelopeV2_9(
                    scene_id=captured_floor.scene_id,
                    floor_object_id=captured_floor.floor_object_id,
                    floor_name=captured_floor.floor_name,
                    native_aabb=captured_floor.native_aabb,
                    floor_top_z=captured_floor.floor_top_z,
                    clearance_m=captured_floor.clearance_m,
                    polygon_xy=captured_floor.polygon_xy,
                )
                validate_competition_native_floor_envelope_v2_9(
                    normalized_scene,
                    support_facts,
                    floor,
                    runtime_identity,
                    expected_clearance_m=floor_clearance_m,
                )
            except _EXPECTED_SOURCE_ERRORS:
                floor = None
                floor_reason = "source_capture:floor_envelope_mismatch"
        try:
            raw_reachable = adapter.reachable_agent_positions(scene)
            reachable = _positions(raw_reachable)
        except _EXPECTED_SOURCE_ERRORS:
            reachable_reason = "source_capture:reachable_positions_missing"

    placement_by_id: dict[str, CompetitionNativeSubjectPlacementFactV2_9] = {}
    for item in floor_subjects:
        support = support_by_id[item.object_id]
        missing_reason = floor_reason or reachable_reason
        if missing_reason is not None or floor is None or not reachable:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                missing_reason or "source_capture:floor_inner_domain_missing",
            )
            continue
        try:
            navigation = build_navigation_feasibility_map(
                normalized_scene,
                subject_object_id=item.object_id,
                room_polygon_xy=floor.polygon_xy,
                reachable_positions=tuple(
                    AI2ThorNativePosition(x=value.x, y=value.y, z=value.z)
                    for value in reachable
                ),
                agent_radius_m=navigation_agent_radius_m,
                clearance_m=navigation_clearance_m,
            )
            if not navigation.position_region.components:
                raise ValueError("navigation inner region is empty")
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    position_region=navigation.position_region,
                )
            )
        except _EXPECTED_SOURCE_ERRORS:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                "source_capture:floor_inner_domain_missing",
            )

    receptacle_subjects = tuple(
        item
        for item in normalized_scene.objects
        if item.movable
        and support_by_id[item.object_id].support_kind
        is CompetitionNativeSupportKindV2_9.RECEPTACLE
    )
    retained_spawn_maps: list[AI2ThorReceptacleSpawnMap] = []
    for item in receptacle_subjects:
        support = support_by_id[item.object_id]
        try:
            spawn_map = adapter.receptacle_spawn_map(
                scene, subject_object_id=item.object_id
            )
            if (
                spawn_map.scene_id != scene.scene_id
                or spawn_map.subject_object_id != item.object_id
                or spawn_map.support_object_id != support.support_object_id
                or not spawn_map.positions
            ):
                raise AI2ThorNativeReturnError(
                    "spawn map does not bind the captured support"
                )
            position_region = _optional_receptacle_position_region(
                scene,
                spawn_map,
            )
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    native_positions=_positions(spawn_map.positions),
                    position_region=position_region,
                )
            )
            retained_spawn_maps.append(spawn_map)
        except _EXPECTED_SOURCE_ERRORS:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                "source_capture:receptacle_spawn_missing",
            )

    for item in normalized_scene.objects:
        if item.object_id in placement_by_id:
            continue
        support = support_by_id[item.object_id]
        if not item.movable:
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.NOT_APPLICABLE
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                )
            )
        else:
            reason = {
                CompetitionNativeSupportKindV2_9.UNKNOWN: (
                    "source_capture:support_unknown"
                ),
                CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS: (
                    "source_capture:support_multiple_ambiguous"
                ),
                CompetitionNativeSupportKindV2_9.CYCLIC: (
                    "source_capture:support_cyclic"
                ),
            }.get(
                support.support_kind,
                "source_capture:placement_fact_invalid",
            )
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                reason,
            )

    observation = settlement.observation
    try:
        capture = build_competition_native_source_capture_v2_9(
            source=source,
            runtime_identity=runtime_identity,
            scene=normalized_scene,
            rgb_png_sha256=observation.rgb_png_sha256,
            depth_npy_sha256=observation.depth_npy_sha256,
            instance_png_sha256=observation.instance_png_sha256,
            pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
            is_scene_at_rest=observation.is_scene_at_rest,
            settlement_pass_steps=settlement.pass_steps,
            support_facts=support_facts,
            floor_envelope=floor,
            reachable_positions=reachable,
            placement_facts=tuple(
                placement_by_id[item.object_id] for item in normalized_scene.objects
            ),
        )
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:normalized_capture_invalid"), ()
    return (
        CompetitionNativeSourceCaptureOutcomeV2_9(
            source=source,
            status="accepted",
            capture=capture,
            reasons=(),
        ),
        tuple(sorted(retained_spawn_maps, key=lambda item: item.subject_object_id)),
    )


def capture_competition_native_source_v2_9(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureOutcomeV2_9:
    """Preserve the legacy capture API while discarding sibling evidence."""

    outcome, _ = _capture_competition_native_source_with_spawn_maps_v2_9(
        adapter,
        source,
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
    )
    return outcome


def capture_competition_native_source_v2_9_2(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2:
    """Capture once and retain the same-query receptacle patch evidence."""

    outcome, spawn_maps = _capture_competition_native_source_with_spawn_maps_v2_9(
        adapter,
        source,
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
    )
    evidence = (
        None
        if outcome.capture is None
        else build_competition_native_source_surface_evidence_v2_9_2(
            outcome.capture,
            spawn_maps,
        )
    )
    return CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2(
        outcome=outcome,
        surface_evidence=evidence,
    )


def _native_camera_pose(
    pose: CompetitionNativeCameraPoseV2_9_3,
) -> AI2ThorAgentPose:
    return AI2ThorAgentPose(
        position=AI2ThorNativePosition(x=pose.x, y=pose.y, z=pose.z),
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _camera_placement_roster_v2_9_4(
    scene: Scene,
    pairs: tuple[tuple[str, str], ...],
    spawn_maps: tuple[AI2ThorReceptacleSpawnMap, ...],
) -> tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...]:
    """Project complete read-only native spawn maps into the camera score wire."""

    if type(scene) is not Scene or type(pairs) is not tuple:
        raise TypeError("camera placement roster inputs must be exact")
    if type(spawn_maps) is not tuple or any(
        type(item) is not AI2ThorReceptacleSpawnMap for item in spawn_maps
    ):
        raise TypeError("camera placement roster spawn maps must be exact")
    by_subject = {item.subject_object_id: item for item in spawn_maps}
    if len(by_subject) != len(spawn_maps) or set(by_subject) != {
        subject_id for subject_id, _ in pairs
    }:
        raise ValueError("camera placement roster does not cover its subject pairs")
    entries = []
    for subject_id, support_id in pairs:
        subject = scene.object_by_id(subject_id)
        scene.object_by_id(support_id)
        spawn_map = by_subject[subject_id]
        if (
            subject.support_object_id != support_id
            or spawn_map.scene_id != scene.scene_id
            or spawn_map.subject_object_id != subject_id
            or spawn_map.support_object_id != support_id
            or not spawn_map.positions
        ):
            raise ValueError("camera placement roster does not bind the source")
        entries.append(
            CompetitionNativeCameraPlacementRosterEntryV2_9_4(
                subject_object_id=subject_id,
                support_object_id=support_id,
                positions=tuple(
                    CompetitionNativeCameraPlacementPositionV2_9_4(
                        x=item.x,
                        y=item.y,
                        z=item.z,
                    )
                    for item in spawn_map.positions
                ),
            )
        )
    return tuple(sorted(entries, key=lambda item: item.subject_object_id))


def _camera_capture_placement_roster_v2_9_4(
    capture: CompetitionNativeSourceCaptureV2_9,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
]:
    """Close every final receptacle pair and every allowed patchless exclusion."""

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("camera placement capture must be exact")
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
                raise ValueError("final camera placement roster is incomplete")
    except KeyError as error:
        raise ValueError("final camera placement roster is incomplete") from error
    return tuple(certified_pairs), tuple(roster_entries)


def _capture_selected_camera_source_v2_9_3(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    scene: Scene,
    observation: AI2ThorObservation,
    settlement_pass_steps: int,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    reachable: tuple[CompetitionNativePositionV2_9, ...],
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> tuple[
    CompetitionNativeSourceCaptureOutcomeV2_9,
    tuple[AI2ThorReceptacleSpawnMap, ...],
]:
    """Capture placement facts only after the camera winner is fixed."""

    normalized_scene = normalize_competition_native_source_scene_v2_9(scene)
    support_by_id = {item.object_id: item for item in support_facts}
    floor_subjects = tuple(
        item
        for item in normalized_scene.objects
        if item.movable
        and support_by_id[item.object_id].support_kind
        is CompetitionNativeSupportKindV2_9.FLOOR
    )
    floor = None
    floor_reason = None
    if floor_subjects:
        try:
            captured_floor = adapter.conservative_floor_envelope(
                scene, clearance_m=floor_clearance_m
            )
        except _EXPECTED_SOURCE_ERRORS:
            floor_reason = "source_capture:floor_envelope_missing"
        else:
            try:
                floor = CompetitionNativeFloorEnvelopeV2_9(
                    scene_id=captured_floor.scene_id,
                    floor_object_id=captured_floor.floor_object_id,
                    floor_name=captured_floor.floor_name,
                    native_aabb=captured_floor.native_aabb,
                    floor_top_z=captured_floor.floor_top_z,
                    clearance_m=captured_floor.clearance_m,
                    polygon_xy=captured_floor.polygon_xy,
                )
                validate_competition_native_floor_envelope_v2_9(
                    normalized_scene,
                    support_facts,
                    floor,
                    runtime_identity,
                    expected_clearance_m=floor_clearance_m,
                )
            except _EXPECTED_SOURCE_ERRORS:
                floor = None
                floor_reason = "source_capture:floor_envelope_mismatch"

    placement_by_id: dict[str, CompetitionNativeSubjectPlacementFactV2_9] = {}
    for item in floor_subjects:
        support = support_by_id[item.object_id]
        if floor_reason is not None or floor is None or not reachable:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                floor_reason or "source_capture:floor_inner_domain_missing",
            )
            continue
        try:
            navigation = build_navigation_feasibility_map(
                normalized_scene,
                subject_object_id=item.object_id,
                room_polygon_xy=floor.polygon_xy,
                reachable_positions=tuple(
                    AI2ThorNativePosition(x=value.x, y=value.y, z=value.z)
                    for value in reachable
                ),
                agent_radius_m=navigation_agent_radius_m,
                clearance_m=navigation_clearance_m,
            )
            if not navigation.position_region.components:
                raise ValueError("navigation inner region is empty")
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    position_region=navigation.position_region,
                )
            )
        except _EXPECTED_SOURCE_ERRORS:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                "source_capture:floor_inner_domain_missing",
            )

    receptacle_subjects = tuple(
        item
        for item in normalized_scene.objects
        if item.movable
        and support_by_id[item.object_id].support_kind
        is CompetitionNativeSupportKindV2_9.RECEPTACLE
    )
    retained_spawn_maps: list[AI2ThorReceptacleSpawnMap] = []
    for item in receptacle_subjects:
        support = support_by_id[item.object_id]
        try:
            spawn_map = adapter.receptacle_spawn_map(
                scene, subject_object_id=item.object_id
            )
            if (
                spawn_map.scene_id != scene.scene_id
                or spawn_map.subject_object_id != item.object_id
                or spawn_map.support_object_id != support.support_object_id
                or not spawn_map.positions
            ):
                raise AI2ThorNativeReturnError(
                    "spawn map does not bind the captured support"
                )
            if not spawn_map.surface_patches:
                placement_by_id[item.object_id] = _missing_placement(
                    item.object_id,
                    support,
                    "source_capture:receptacle_surface_patch_missing",
                )
                continue
            position_region = _optional_receptacle_position_region(scene, spawn_map)
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    native_positions=_positions(spawn_map.positions),
                    position_region=position_region,
                )
            )
            retained_spawn_maps.append(spawn_map)
        except _EXPECTED_SOURCE_ERRORS:
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                "source_capture:receptacle_spawn_missing",
            )

    for item in normalized_scene.objects:
        if item.object_id in placement_by_id:
            continue
        support = support_by_id[item.object_id]
        if not item.movable:
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=(
                        CompetitionNativePlacementAvailabilityV2_9.NOT_APPLICABLE
                    ),
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                )
            )
        else:
            reason = {
                CompetitionNativeSupportKindV2_9.UNKNOWN: (
                    "source_capture:support_unknown"
                ),
                CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS: (
                    "source_capture:support_multiple_ambiguous"
                ),
                CompetitionNativeSupportKindV2_9.CYCLIC: (
                    "source_capture:support_cyclic"
                ),
            }.get(support.support_kind, "source_capture:placement_fact_invalid")
            placement_by_id[item.object_id] = _missing_placement(
                item.object_id,
                support,
                reason,
            )

    try:
        capture = build_competition_native_source_capture_v2_9(
            source=source,
            runtime_identity=runtime_identity,
            scene=normalized_scene,
            rgb_png_sha256=observation.rgb_png_sha256,
            depth_npy_sha256=observation.depth_npy_sha256,
            instance_png_sha256=observation.instance_png_sha256,
            pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
            is_scene_at_rest=observation.is_scene_at_rest,
            settlement_pass_steps=settlement_pass_steps,
            support_facts=support_facts,
            floor_envelope=floor,
            reachable_positions=reachable,
            placement_facts=tuple(
                placement_by_id[item.object_id] for item in normalized_scene.objects
            ),
        )
    except _EXPECTED_SOURCE_ERRORS:
        return _rejected(source, "source_capture:normalized_capture_invalid"), ()
    return (
        CompetitionNativeSourceCaptureOutcomeV2_9(
            source=source,
            status="accepted",
            capture=capture,
            reasons=(),
        ),
        tuple(sorted(retained_spawn_maps, key=lambda item: item.subject_object_id)),
    )


def _capture_competition_native_source_v2_9_3(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
    camera_policy: CompetitionNativeCameraPolicyV2_9_3,
) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
    """Select one complete source camera before placement capture."""

    def rejected(reason: str) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
        return CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
            outcome=_rejected(source, reason),
            surface_evidence=None,
            camera_evidence=None,
        )

    if type(source) is not CompetitionNativeSourceRefV2_9:
        raise TypeError("native source ref must be exact")
    try:
        loaded = adapter.load_scene(source.scene_id)
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:load_failed")
    try:
        settlement = adapter.settle_scene_observed(
            loaded, max_pass_steps=max_settlement_steps
        )
    except AI2ThorSettlementTimeout:
        return rejected("source_capture:settlement_timeout")
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:settlement_failed")
    baseline_scene = settlement.observed_scene
    try:
        runtime = adapter.runtime_identity()
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:runtime_identity_failed")
    try:
        raw_support_facts = adapter.native_support_facts(baseline_scene)
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:support_snapshot_failed")
    try:
        normalized_baseline = normalize_competition_native_source_scene_v2_9(
            baseline_scene
        )
        support_facts = tuple(
            sorted(
                (_support_fact(item) for item in raw_support_facts),
                key=lambda item: item.object_id,
            )
        )
        object_ids = tuple(item.object_id for item in normalized_baseline.objects)
        if tuple(item.object_id for item in support_facts) != object_ids:
            raise ValueError("support fact roster does not match scene objects")
        if any(
            item.scene_id != normalized_baseline.scene_id
            or item.object_name != normalized_baseline.object_by_id(item.object_id).name
            for item in support_facts
        ):
            raise ValueError("support facts do not bind the settled scene")
        runtime_identity = CompetitionNativeRuntimeIdentityV2_9(**asdict(runtime))
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:support_snapshot_invalid")
    try:
        validate_competition_native_runtime_source_lineage_v2_9(
            source, runtime_identity
        )
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:runtime_lineage_mismatch")

    camera_stack = ExitStack()
    try:
        raw_reachable = adapter.reachable_agent_positions(baseline_scene)
        reachable = _positions(raw_reachable)
        fallback_pose = adapter.current_agent_pose(baseline_scene)
        support_by_id = {item.object_id: item for item in support_facts}
        pairs = tuple(
            sorted(
                (
                    item.object_id,
                    support_by_id[item.object_id].support_object_id,
                )
                for item in normalized_baseline.objects
                if item.movable
                and support_by_id[item.object_id].support_kind
                is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and support_by_id[item.object_id].support_object_id is not None
            )
        )
        policy = CompetitionNativeCameraPolicyV2_9_3.model_validate(
            camera_policy.model_dump(mode="python"), strict=True
        )
        editable_camera = policy.pose_policy_version in {
            _EDITABLE_CAMERA_POLICY_VERSION,
            _COLLISION_SAFE_EDITABLE_CAMERA_POLICY_VERSION,
            _CONTACT_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
            _RESET_PER_POSE_EDITABLE_CAMERA_POLICY_VERSION,
            _GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
            _PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
            _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
        }
        reset_per_pose = (
            policy.pose_policy_version == _RESET_PER_POSE_EDITABLE_CAMERA_POLICY_VERSION
        )
        pause_physics = policy.pose_policy_version in {
            _PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
            _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION,
        }
        settle_after_unpause = (
            policy.pose_policy_version
            == _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_CAMERA_POLICY_VERSION
        )
        if settle_after_unpause:
            reachable = _positions(
                canonicalize_ai2thor_reachable_positions(raw_reachable)
            )
        frozen_reachable = reachable
        paused_observations = (
            adapter.paused_camera_observations_for_settlement
            if settle_after_unpause
            else adapter.paused_camera_observations
        )
        current_scene = (
            camera_stack.enter_context(paused_observations(baseline_scene))
            if pause_physics
            else baseline_scene
        )
        queried_proposal_spawn_maps = (
            tuple(
                adapter.receptacle_spawn_map(
                    baseline_scene,
                    subject_object_id=subject_object_id,
                )
                for subject_object_id, _support_object_id in pairs
            )
            if editable_camera
            else ()
        )
        if editable_camera:
            proposal_by_subject = {
                item.subject_object_id: item for item in queried_proposal_spawn_maps
            }
            pairs = tuple(
                pair for pair in pairs if proposal_by_subject[pair[0]].surface_patches
            )
            proposal_spawn_maps = tuple(
                proposal_by_subject[subject_object_id]
                for subject_object_id, _support_object_id in pairs
            )
        else:
            proposal_spawn_maps = ()
        placement_roster = (
            _camera_placement_roster_v2_9_4(
                baseline_scene,
                pairs,
                proposal_spawn_maps,
            )
            if editable_camera
            else ()
        )
        pose_bank = build_competition_native_camera_pose_bank_v2_9_3(
            baseline_scene,
            pairs,
            tuple(
                AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
                for item in reachable
            ),
            fallback_pose,
            policy=policy,
        )
        pose_scores: list[
            CompetitionNativeCameraScoreV2_9_3 | CompetitionNativeCameraScoreV2_9_4
        ] = []
        for pose in pose_bank:
            application = (
                adapter.apply_camera_pose_from_frozen_source_observed(
                    baseline_scene,
                    _native_camera_pose(pose),
                    max_pass_steps=max_settlement_steps,
                )
                if reset_per_pose
                else adapter.apply_camera_pose_observed(
                    current_scene,
                    _native_camera_pose(pose),
                )
            )
            current_scene = application.observed_scene
            pose_scores.append(
                score_competition_native_editable_camera_application_v2_9_4(
                    source_scene=baseline_scene,
                    pose=pose,
                    application=application,
                    placement_roster=placement_roster,
                    policy=policy,
                )
                if editable_camera
                else score_competition_native_source_camera_application_v2_9_3(
                    source_scene=baseline_scene,
                    pose=pose,
                    application=application,
                    policy=policy,
                )
            )
            del application
        frozen_scores = tuple(pose_scores)
        selected_index = (
            select_competition_native_camera_score_index_v2_9_4(frozen_scores)
            if editable_camera
            else select_competition_native_camera_score_index_v2_9_3(frozen_scores)
        )
        selected_application = (
            adapter.apply_camera_pose_from_frozen_source_observed(
                baseline_scene,
                _native_camera_pose(pose_bank[selected_index]),
                max_pass_steps=max_settlement_steps,
            )
            if reset_per_pose
            else adapter.apply_camera_pose_observed(
                current_scene,
                _native_camera_pose(pose_bank[selected_index]),
            )
        )
        selected_score = (
            score_competition_native_editable_camera_application_v2_9_4(
                source_scene=baseline_scene,
                pose=pose_bank[selected_index],
                application=selected_application,
                placement_roster=placement_roster,
                policy=policy,
            )
            if editable_camera
            else score_competition_native_source_camera_application_v2_9_3(
                source_scene=baseline_scene,
                pose=pose_bank[selected_index],
                application=selected_application,
                policy=policy,
            )
        )
        if selected_score != frozen_scores[selected_index]:
            raise ValueError("camera winner replay score changed")
        evidence_source_scene = baseline_scene
        capture_settlement_pass_steps = settlement.pass_steps
        if settle_after_unpause:
            try:
                camera_stack.close()
            except _EXPECTED_SOURCE_ERRORS:
                return rejected("source_capture:camera_physics_resume_failed")
            settled_camera = adapter.settle_current_camera_pose_observed(
                baseline_scene,
                _native_camera_pose(pose_bank[selected_index]),
                max_pass_steps=max_settlement_steps,
            )
            selected_application = settled_camera.application
            capture_settlement_pass_steps = settled_camera.settlement_pass_steps
            evidence_source_scene = selected_application.observed_scene
            final_support_facts = tuple(
                sorted(
                    (
                        _support_fact(item)
                        for item in adapter.native_support_facts(evidence_source_scene)
                    ),
                    key=lambda item: item.object_id,
                )
            )
            final_object_ids = tuple(
                sorted(item.object_id for item in evidence_source_scene.objects)
            )
            if (
                tuple(item.object_id for item in final_support_facts)
                != final_object_ids
            ):
                raise ValueError("final support fact roster does not match scene")
            if any(
                item.scene_id != evidence_source_scene.scene_id
                or item.object_name
                != evidence_source_scene.object_by_id(item.object_id).name
                for item in final_support_facts
            ):
                raise ValueError("final support facts do not bind settled scene")
            support_facts = final_support_facts
            final_raw_reachable = adapter.reachable_agent_positions(
                evidence_source_scene
            )
            final_reachable = _positions(
                bind_ai2thor_reachable_positions(
                    raw_reachable,
                    final_raw_reachable,
                )
            )
            if final_reachable != frozen_reachable:
                raise ValueError("settled camera reachable position roster changed")
            reachable = final_reachable
            final_pose_bank = build_competition_native_camera_pose_bank_v2_9_3(
                evidence_source_scene,
                pairs,
                tuple(
                    AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
                    for item in reachable
                ),
                _native_camera_pose(pose_bank[selected_index]),
                policy=policy,
            )
            if final_pose_bank != pose_bank:
                raise ValueError("settled camera pose bank changed")
            final_selected_score = (
                score_competition_native_editable_camera_application_v2_9_4(
                    source_scene=evidence_source_scene,
                    pose=pose_bank[selected_index],
                    application=selected_application,
                    placement_roster=placement_roster,
                    policy=policy,
                )
                if editable_camera
                else score_competition_native_source_camera_application_v2_9_3(
                    source_scene=evidence_source_scene,
                    pose=pose_bank[selected_index],
                    application=selected_application,
                    policy=policy,
                )
            )
            if final_selected_score != frozen_scores[selected_index]:
                raise ValueError("settled camera winner score changed")
    except _EXPECTED_SOURCE_ERRORS:
        try:
            camera_stack.close()
        except _EXPECTED_SOURCE_ERRORS:
            return rejected("source_capture:camera_physics_resume_failed")
        return rejected("source_capture:camera_selection_failed")

    outcome, spawn_maps = _capture_selected_camera_source_v2_9_3(
        adapter,
        source,
        scene=selected_application.observed_scene,
        observation=selected_application.observation,
        settlement_pass_steps=capture_settlement_pass_steps,
        runtime_identity=runtime_identity,
        support_facts=support_facts,
        reachable=reachable,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
    )
    capture = outcome.capture
    if capture is None:
        del selected_application
        try:
            camera_stack.close()
        except _EXPECTED_SOURCE_ERRORS:
            return rejected("source_capture:camera_physics_resume_failed")
        return CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
            outcome=outcome,
            surface_evidence=None,
            camera_evidence=None,
        )
    try:
        if editable_camera:
            if settle_after_unpause:
                final_pairs, final_placement_roster = (
                    _camera_capture_placement_roster_v2_9_4(capture)
                )
                placement_roster_changed = (
                    final_pairs != pairs
                    or final_placement_roster != placement_roster
                    or _camera_placement_roster_v2_9_4(
                        selected_application.observed_scene,
                        final_pairs,
                        spawn_maps,
                    )
                    != final_placement_roster
                )
            else:
                placement_roster_changed = (
                    _camera_placement_roster_v2_9_4(
                        selected_application.observed_scene,
                        pairs,
                        spawn_maps,
                    )
                    != placement_roster
                )
            if placement_roster_changed:
                raise ValueError("camera placement roster changed after winner replay")
        surface_evidence = build_competition_native_source_surface_evidence_v2_9_2(
            capture,
            spawn_maps,
        )
        evidence_arguments = {
            "source_id": source.source_id,
            "scene_id": source.scene_id,
            "source_locator_sha256": source.source_locator_sha256,
            "runtime_identity_sha256": canonical_sha256_v2(
                runtime_identity,
                domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
            ),
            "source_capture_sha256": capture.source_capture_sha256,
            "source_scene": evidence_source_scene,
            "policy": policy,
            "pose_bank": pose_bank,
            "pose_scores": frozen_scores,
            "selected_application": selected_application,
        }
        camera_evidence = (
            build_competition_native_source_camera_evidence_v2_9_4(
                **evidence_arguments,
                placement_roster=placement_roster,
            )
            if editable_camera
            else build_competition_native_source_camera_evidence_v2_9_3(
                **evidence_arguments,
            )
        )
        result = CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
            outcome=outcome,
            surface_evidence=surface_evidence,
            camera_evidence=camera_evidence,
        )
    except _EXPECTED_SOURCE_ERRORS:
        del selected_application
        try:
            camera_stack.close()
        except _EXPECTED_SOURCE_ERRORS:
            return rejected("source_capture:camera_physics_resume_failed")
        return rejected("source_capture:evidence_invalid")
    del selected_application
    try:
        camera_stack.close()
    except _EXPECTED_SOURCE_ERRORS:
        return rejected("source_capture:camera_physics_resume_failed")
    return result


def capture_competition_native_source_v2_9_3(
    adapter: AI2ThorAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
    """Capture with the original 2.9.3 camera policy for API compatibility."""

    return _capture_competition_native_source_v2_9_3(
        adapter,
        source,
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
        camera_policy=build_competition_native_camera_policy_v2_9_3(),
    )


@dataclass(frozen=True, slots=True)
class SourceCaptureResult:
    outcome: SourceCaptureOutcome
    surface_evidence: SourceSurfaceEvidence | None
    camera_evidence: SourceCameraEvidence | None


def capture_source_observation(
    adapter: AI2ThorAdapter,
    *,
    source: SourceRef,
    settings: CaptureSettings,
    camera_policy: CameraPolicy,
) -> SourceCaptureResult:
    """Capture one source once; never select or retry based on endpoint success."""

    if type(settings) is not CaptureSettings:
        raise TypeError("capture settings must be exact")
    checked_settings = CaptureSettings.model_validate(
        settings.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    result = _capture_competition_native_source_v2_9_3(
        adapter,
        source,
        max_settlement_steps=checked_settings.max_settlement_steps,
        floor_clearance_m=checked_settings.floor_clearance_m,
        navigation_agent_radius_m=checked_settings.navigation_agent_radius_m,
        navigation_clearance_m=checked_settings.navigation_clearance_m,
        camera_policy=camera_policy,
    )
    return SourceCaptureResult(
        outcome=result.outcome,
        surface_evidence=result.surface_evidence,
        camera_evidence=result.camera_evidence,
    )


__all__ = (
    "SourceCaptureResult",
    "capture_source_observation",
)
