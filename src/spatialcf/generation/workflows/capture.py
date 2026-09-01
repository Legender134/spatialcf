"""Protocol-only source capture and deterministic roster compilation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from pydantic import model_validator

from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterObservation,
    AdapterOperationError,
    AdapterPose,
    AdapterPosition,
    AdapterProceduralScene,
    AdapterSettlementTimeout,
    AdapterSpawnMap,
    AdapterSupportFact,
    CapturedSource,
    CaptureRequest,
    EnvironmentAdapter,
    InstanceEvidenceProvenance,
    SourceCaptureFacts,
    SourceCaptureOptions,
)
from spatialcf.domain.base import CanonicalModel
from spatialcf.domain.request import Relation
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256
from spatialcf.domain.source import LegacyAI2ThorSource, ProceduralSource
from spatialcf.generation.capture.compiler import compile_roster
from spatialcf.generation.capture.models import (
    CameraPolicy,
    CompetitionNativeCameraPlacementPositionV2_9_4,
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraPolicyV2_9_3,
    CompetitionNativeCameraPoseV2_9_3,
    CompetitionNativeCameraScoreV2_9_3,
    CompetitionNativeCameraScoreV2_9_4,
    CompetitionNativeFloorEnvelopeV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativePositionV2_9,
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceCaptureOutcomeV2_9,
    CompetitionNativeSourceRefV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportFactV2_9,
    CompetitionNativeSupportKindV2_9,
    SourceCameraEvidence,
    SourceSurfaceEvidence,
    build_competition_native_camera_policy_v2_9_3,
    build_competition_native_camera_pose_bank_v2_9_3,
    build_competition_native_source_camera_evidence_v2_9_3,
    build_competition_native_source_camera_evidence_v2_9_4,
    build_competition_native_source_capture_v2_9,
    build_competition_native_source_surface_evidence_v2_9_2,
    build_competition_native_subject_placement_fact_v2_9,
    normalize_competition_native_source_scene_v2_9,
    score_competition_native_editable_camera_application_v2_9_4,
    score_competition_native_source_camera_application_v2_9_3,
    select_competition_native_camera_score_index_v2_9_3,
    select_competition_native_camera_score_index_v2_9_4,
    validate_competition_native_floor_envelope_v2_9,
    validate_competition_native_runtime_source_lineage_v2_9,
)
from spatialcf.generation.capture.plan import CapturePlan, CaptureSettings
from spatialcf.generation.capture.storage import publish_roster
from spatialcf.generation.capture.visual_evidence import (
    SourceViewFactError,
    build_source_view_fact,
)

_PROCTHOR_DATASET_ID = "allenai/procthor-10k"
_PROCTHOR_DATASET_NAME = "procthor-10k"
_PROCTHOR_LOADER_ID = "prior"
_PROCTHOR_LOADER_VERSION = "1.0.3"
_EXPECTED_SOURCE_LIFECYCLE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_EXPECTED_CAPTURE_ERRORS = (
    AdapterOperationError,
    AdapterSettlementTimeout,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
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


class CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2(CanonicalModel):
    outcome: CompetitionNativeSourceCaptureOutcomeV2_9
    surface_evidence: SourceSurfaceEvidence | None

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


class CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(CanonicalModel):
    outcome: CompetitionNativeSourceCaptureOutcomeV2_9
    surface_evidence: SourceSurfaceEvidence | None
    camera_evidence: SourceCameraEvidence | None

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


def load_prior_dataset(name: str, revision: str) -> object:
    try:
        import prior
    except ImportError as error:
        raise RuntimeError("ProcTHOR capture requires spatialcf[procthor]") from error
    try:
        return prior.load_dataset(name, revision=revision)
    except MemoryError:
        raise
    except Exception as error:
        raise RuntimeError("ProcTHOR dataset loader failed") from error


def _dataset_split(dataset: object, split: str) -> Sequence[dict]:
    """Return exactly the named mapping split; no attribute fallback exists."""
    if not isinstance(dataset, Mapping):
        raise TypeError("ProcTHOR dataset must expose named mapping splits")
    try:
        values = dataset[split]
    except KeyError as error:
        raise ValueError("ProcTHOR dataset split is absent") from error
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("ProcTHOR dataset split must be indexable")
    return values


def _resolve_procedural_source(
    locator: ProceduralSource,
    datasets: dict[tuple[str, str], object],
    dataset_loader: Callable[[str, str], object],
) -> AdapterProceduralScene:
    if (
        locator.dataset_id != _PROCTHOR_DATASET_ID
        or locator.loader_id != _PROCTHOR_LOADER_ID
        or locator.loader_version != _PROCTHOR_LOADER_VERSION
    ):
        raise ValueError("unsupported ProcTHOR source loader identity")
    key = (_PROCTHOR_DATASET_NAME, locator.revision)
    if key not in datasets:
        datasets[key] = dataset_loader(*key)
    values = _dataset_split(datasets[key], locator.split)
    try:
        house = values[locator.index]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("ProcTHOR source locator does not exist") from error
    if type(house) is not dict:
        raise ValueError("ProcTHOR source house must be an exact dict")
    scene = AdapterProceduralScene.create(
        dataset_id=locator.dataset_id,
        revision=locator.revision,
        split=locator.split,
        index=locator.index,
        source_loader_id=locator.loader_id,
        source_loader_version=locator.loader_version,
        house=house,
    )
    if scene.house_sha256 != locator.source_sha256:
        raise ValueError("ProcTHOR source content digest changed")
    return scene


def _rejected_source(
    source: CompetitionNativeSourceRefV2_9,
    reason: str,
) -> CompetitionNativeSourceCaptureOutcomeV2_9:
    return CompetitionNativeSourceCaptureOutcomeV2_9(
        source=source,
        status="rejected",
        capture=None,
        reasons=(reason,),
    )


def _support_fact(value: AdapterSupportFact) -> CompetitionNativeSupportFactV2_9:
    if type(value) is not AdapterSupportFact:
        raise TypeError("adapter support fact must be exact")
    return CompetitionNativeSupportFactV2_9(
        scene_id=value.scene_id,
        object_id=value.object_id,
        object_name=value.object_name,
        native_object_id=value.native_object_id,
        raw_parent_object_ids=value.raw_parent_object_ids,
        structural_parent_object_ids=value.structural_parent_object_ids,
        domain_parent_object_ids=value.domain_parent_object_ids,
        support_kind=CompetitionNativeSupportKindV2_9(value.support_kind),
        support_object_id=value.support_object_id,
        floor_object_id=value.floor_object_id,
    )


def _positions(
    values: tuple[AdapterPosition, ...],
) -> tuple[CompetitionNativePositionV2_9, ...]:
    return tuple(
        CompetitionNativePositionV2_9(x=item.x, y=item.y, z=item.z)
        for item in sorted(values, key=lambda item: (item.x, item.z, item.y))
    )


def _adapter_pose(pose: CompetitionNativeCameraPoseV2_9_3) -> AdapterPose:
    return AdapterPose(
        position=AdapterPosition(x=pose.x, y=pose.y, z=pose.z),
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _canonical_positions(
    positions: tuple[AdapterPosition, ...],
) -> tuple[AdapterPosition, ...]:
    keyed = {
        (round(item.x / 1e-6), round(item.z / 1e-6), round(item.y / 1e-6))
        for item in positions
    }
    if len(keyed) != len(positions):
        raise ValueError("reachable positions contain a duplicate canonical point")
    return tuple(
        AdapterPosition(x=key[0] * 1e-6, y=key[2] * 1e-6, z=key[1] * 1e-6)
        for key in sorted(keyed)
    )


def _camera_placement_roster(
    scene,
    pairs: tuple[tuple[str, str], ...],
    spawn_maps: tuple[AdapterSpawnMap, ...],
) -> tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...]:
    if type(spawn_maps) is not tuple or any(
        type(item) is not AdapterSpawnMap for item in spawn_maps
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


def _capture_selected_camera_source(
    source: CompetitionNativeSourceRefV2_9,
    *,
    facts: SourceCaptureFacts,
    scene,
    application: AdapterCameraApplication,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    reachable: tuple[CompetitionNativePositionV2_9, ...],
    spawn_maps: tuple[AdapterSpawnMap, ...],
    settlement_pass_steps: int,
    floor_clearance_m: float,
) -> tuple[CompetitionNativeSourceCaptureOutcomeV2_9, tuple[AdapterSpawnMap, ...]]:
    normalized_scene = normalize_competition_native_source_scene_v2_9(scene)
    support_by_id = {item.object_id: item for item in support_facts}
    runtime_identity = CompetitionNativeRuntimeIdentityV2_9(
        **asdict(facts.runtime_identity)
    )
    raw_observation = application.observation
    source_view_fact = None
    if (
        raw_observation.scene != application.observed_scene
        or application.observed_scene != scene
    ):
        return (
            _rejected_source(source, "source_capture:source_view_fact_invalid"),
            (),
        )
    try:
        observation = AdapterObservation.create(
            scene=normalized_scene,
            rgb_png=raw_observation.rgb_png,
            depth_npy=raw_observation.depth_npy,
            instance_png=raw_observation.instance_png,
            pointcloud_ply=raw_observation.pointcloud_ply,
            instance_pixel_counts=raw_observation.instance_pixel_counts,
            instance_colors=raw_observation.instance_colors,
            instance_evidence_provenance=(
                raw_observation.instance_evidence_provenance
            ),
            is_settled=raw_observation.is_settled,
        )
    except (TypeError, ValueError):
        return (
            _rejected_source(source, "source_capture:source_view_fact_invalid"),
            (),
        )
    if (
        observation.rgb_png_sha256 != raw_observation.rgb_png_sha256
        or observation.depth_npy_sha256 != raw_observation.depth_npy_sha256
        or observation.instance_png_sha256 != raw_observation.instance_png_sha256
        or observation.pointcloud_ply_sha256 != raw_observation.pointcloud_ply_sha256
    ):
        return (
            _rejected_source(source, "source_capture:source_view_fact_invalid"),
            (),
        )
    if (
        observation.instance_evidence_provenance
        is InstanceEvidenceProvenance.SAME_EVENT_INSTANCE_SEGMENTATION
    ):
        try:
            source_view_fact = build_source_view_fact(
                source,
                runtime_identity,
                normalized_scene,
                observation,
            )
        except (SourceViewFactError, TypeError, ValueError):
            return (
                _rejected_source(source, "source_capture:source_view_fact_invalid"),
                (),
            )
    elif (
        observation.instance_evidence_provenance
        is not InstanceEvidenceProvenance.LEGACY_NEUTRAL
    ):
        return (
            _rejected_source(source, "source_capture:source_view_fact_invalid"),
            (),
        )
    floor = None
    floor_reason = None
    floor_subjects = tuple(
        item
        for item in normalized_scene.objects
        if item.movable
        and support_by_id[item.object_id].support_kind
        is CompetitionNativeSupportKindV2_9.FLOOR
    )
    if floor_subjects:
        if facts.floor_envelope is None:
            floor_reason = "source_capture:floor_envelope_missing"
        else:
            try:
                floor = CompetitionNativeFloorEnvelopeV2_9(
                    scene_id=facts.floor_envelope.scene_id,
                    floor_object_id=facts.floor_envelope.floor_object_id,
                    floor_name=facts.floor_envelope.floor_name,
                    native_aabb=facts.floor_envelope.native_aabb,
                    floor_top_z=facts.floor_envelope.floor_top_z,
                    clearance_m=facts.floor_envelope.clearance_m,
                    polygon_xy=facts.floor_envelope.polygon_xy,
                )
                validate_competition_native_floor_envelope_v2_9(
                    normalized_scene,
                    support_facts,
                    floor,
                    runtime_identity,
                    expected_clearance_m=floor_clearance_m,
                )
            except (TypeError, ValueError):
                floor = None
                floor_reason = "source_capture:floor_envelope_mismatch"

    placement_by_id: dict[str, CompetitionNativeSubjectPlacementFactV2_9] = {}
    floor_regions = dict(facts.floor_position_regions)
    for item in floor_subjects:
        support = support_by_id[item.object_id]
        region = floor_regions.get(item.object_id)
        if floor_reason is not None or floor is None or not reachable or region is None:
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=CompetitionNativePlacementAvailabilityV2_9.MISSING,
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    reasons=(
                        floor_reason or "source_capture:floor_inner_domain_missing",
                    ),
                )
            )
        else:
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION,
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                    position_region=region,
                )
            )

    maps_by_subject = {item.subject_object_id: item for item in spawn_maps}
    retained_maps = []
    for item in normalized_scene.objects:
        support = support_by_id[item.object_id]
        if item.object_id in placement_by_id:
            continue
        if (
            item.movable
            and support.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
        ):
            spawn_map = maps_by_subject.get(item.object_id)
            if (
                spawn_map is None
                or spawn_map.scene_id != normalized_scene.scene_id
                or spawn_map.support_object_id != support.support_object_id
                or not spawn_map.positions
            ):
                placement_by_id[item.object_id] = (
                    build_competition_native_subject_placement_fact_v2_9(
                        object_id=item.object_id,
                        availability=CompetitionNativePlacementAvailabilityV2_9.MISSING,
                        support_kind=support.support_kind,
                        support_object_id=support.support_object_id,
                        floor_object_id=support.floor_object_id,
                        reasons=("source_capture:receptacle_spawn_missing",),
                    )
                )
            elif not spawn_map.surface_patches or spawn_map.position_region is None:
                placement_by_id[item.object_id] = (
                    build_competition_native_subject_placement_fact_v2_9(
                        object_id=item.object_id,
                        availability=CompetitionNativePlacementAvailabilityV2_9.MISSING,
                        support_kind=support.support_kind,
                        support_object_id=support.support_object_id,
                        floor_object_id=support.floor_object_id,
                        reasons=("source_capture:receptacle_surface_patch_missing",),
                    )
                )
            else:
                placement_by_id[item.object_id] = (
                    build_competition_native_subject_placement_fact_v2_9(
                        object_id=item.object_id,
                        availability=CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN,
                        support_kind=support.support_kind,
                        support_object_id=support.support_object_id,
                        floor_object_id=support.floor_object_id,
                        native_positions=_positions(spawn_map.positions),
                        position_region=spawn_map.position_region,
                    )
                )
                retained_maps.append(spawn_map)
            continue
        if not item.movable:
            placement_by_id[item.object_id] = (
                build_competition_native_subject_placement_fact_v2_9(
                    object_id=item.object_id,
                    availability=CompetitionNativePlacementAvailabilityV2_9.NOT_APPLICABLE,
                    support_kind=support.support_kind,
                    support_object_id=support.support_object_id,
                    floor_object_id=support.floor_object_id,
                )
            )
            continue
        reason = {
            CompetitionNativeSupportKindV2_9.UNKNOWN: "source_capture:support_unknown",
            CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS: "source_capture:support_multiple_ambiguous",
            CompetitionNativeSupportKindV2_9.CYCLIC: "source_capture:support_cyclic",
        }.get(support.support_kind, "source_capture:placement_fact_invalid")
        placement_by_id[item.object_id] = (
            build_competition_native_subject_placement_fact_v2_9(
                object_id=item.object_id,
                availability=CompetitionNativePlacementAvailabilityV2_9.MISSING,
                support_kind=support.support_kind,
                support_object_id=support.support_object_id,
                floor_object_id=support.floor_object_id,
                reasons=(reason,),
            )
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
            is_scene_at_rest=observation.is_settled,
            settlement_pass_steps=settlement_pass_steps,
            support_facts=support_facts,
            floor_envelope=floor,
            reachable_positions=reachable,
            placement_facts=tuple(
                placement_by_id[item.object_id] for item in normalized_scene.objects
            ),
            source_view_fact=source_view_fact,
        )
    except (TypeError, ValueError):
        return _rejected_source(source, "source_capture:normalized_capture_invalid"), ()
    return (
        CompetitionNativeSourceCaptureOutcomeV2_9(
            source=source,
            status="accepted",
            capture=capture,
            reasons=(),
        ),
        tuple(sorted(retained_maps, key=lambda item: item.subject_object_id)),
    )


def _capture_competition_native_source_v2_9_3(
    adapter: EnvironmentAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
    camera_policy: CompetitionNativeCameraPolicyV2_9_3,
    captured_source_consumer: Callable[[CapturedSource], None] | None = None,
) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
    """Capture one frozen source through six adapter operations, once."""

    def rejected(reason: str) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
        return CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
            outcome=_rejected_source(source, reason),
            surface_evidence=None,
            camera_evidence=None,
        )

    if type(source) is not CompetitionNativeSourceRefV2_9:
        raise TypeError("native source ref must be exact")
    options = SourceCaptureOptions(
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
    )
    try:
        captured_source = adapter.capture_source(
            CaptureRequest(scene_id=source.scene_id, camera_id="main")
        )
    except _EXPECTED_CAPTURE_ERRORS:
        return rejected("source_capture:load_failed")
    if type(captured_source) is not CapturedSource:
        return rejected("source_capture:load_failed")
    if captured_source_consumer is not None:
        captured_source_consumer(captured_source)
    try:
        facts = adapter.observe_source(captured_source, options=options, settle=True)
    except AdapterSettlementTimeout:
        return rejected("source_capture:settlement_timeout")
    except _EXPECTED_CAPTURE_ERRORS:
        return rejected("source_capture:settlement_failed")
    if type(facts) is not SourceCaptureFacts:
        return rejected("source_capture:support_snapshot_invalid")
    try:
        normalized_baseline = normalize_competition_native_source_scene_v2_9(
            facts.scene
        )
        support_facts = tuple(
            sorted(
                (_support_fact(item) for item in facts.support_facts),
                key=lambda item: item.object_id,
            )
        )
        object_ids = tuple(item.object_id for item in normalized_baseline.objects)
        if tuple(item.object_id for item in support_facts) != object_ids:
            raise ValueError("support fact roster does not match scene objects")
        runtime_identity = CompetitionNativeRuntimeIdentityV2_9(
            **asdict(facts.runtime_identity)
        )
        if any(
            item.scene_id != normalized_baseline.scene_id
            or item.object_name != normalized_baseline.object_by_id(item.object_id).name
            for item in support_facts
        ):
            raise ValueError("support facts do not bind the settled scene")
    except (TypeError, ValueError):
        return rejected("source_capture:support_snapshot_invalid")
    try:
        validate_competition_native_runtime_source_lineage_v2_9(
            source,
            runtime_identity,
        )
    except (TypeError, ValueError):
        return rejected("source_capture:runtime_lineage_mismatch")

    try:
        reachable_values = facts.reachable_positions
        policy = CompetitionNativeCameraPolicyV2_9_3.model_validate(
            camera_policy.model_dump(mode="python"),
            strict=True,
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
            reachable_values = _canonical_positions(reachable_values)
        reachable = _positions(reachable_values)
        support_by_id = {item.object_id: item for item in support_facts}
        pairs = tuple(
            sorted(
                (item.object_id, support_by_id[item.object_id].support_object_id)
                for item in normalized_baseline.objects
                if item.movable
                and support_by_id[item.object_id].support_kind
                is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and support_by_id[item.object_id].support_object_id is not None
            )
        )
    except (TypeError, ValueError, KeyError):
        return rejected("source_capture:camera_selection_failed")

    paused_handle = None
    camera_selection_failed = False
    final_facts = facts
    try:
        current_scene = facts.scene
        if pause_physics:
            paused_handle = adapter.pause_camera_observations(
                facts,
                settle_after_resume=settle_after_unpause,
            )
            current_scene = paused_handle.scene
        proposal_maps = (
            adapter.capture_spawn_maps(
                facts,
                subject_object_ids=tuple(subject_id for subject_id, _ in pairs),
            )
            if editable_camera
            else ()
        )
        if editable_camera:
            proposal_by_subject = {
                item.subject_object_id: item for item in proposal_maps
            }
            if set(proposal_by_subject) != {subject_id for subject_id, _ in pairs}:
                raise ValueError("proposal spawn maps do not cover source pairs")
            pairs = tuple(
                pair for pair in pairs if proposal_by_subject[pair[0]].surface_patches
            )
            proposal_maps = tuple(
                proposal_by_subject[subject_id] for subject_id, _ in pairs
            )
        placement_roster = (
            _camera_placement_roster(facts.scene, pairs, proposal_maps)
            if editable_camera
            else ()
        )
        pose_bank = build_competition_native_camera_pose_bank_v2_9_3(
            facts.scene,
            pairs,
            reachable_values,
            facts.current_pose,
            policy=policy,
        )
        pose_scores: list[
            CompetitionNativeCameraScoreV2_9_3 | CompetitionNativeCameraScoreV2_9_4
        ] = []
        for pose in pose_bank:
            application = adapter.apply_camera_pose(
                facts,
                _adapter_pose(pose),
                handle=paused_handle,
                source_scene=facts.scene if reset_per_pose else current_scene,
                reset_from_source=reset_per_pose,
                max_settlement_steps=max_settlement_steps,
            )
            current_scene = application.observed_scene
            pose_scores.append(
                score_competition_native_editable_camera_application_v2_9_4(
                    source_scene=facts.scene,
                    pose=pose,
                    application=application,
                    placement_roster=placement_roster,
                    policy=policy,
                )
                if editable_camera
                else score_competition_native_source_camera_application_v2_9_3(
                    source_scene=facts.scene,
                    pose=pose,
                    application=application,
                    policy=policy,
                )
            )
        frozen_scores = tuple(pose_scores)
        selected_index = (
            select_competition_native_camera_score_index_v2_9_4(frozen_scores)
            if editable_camera
            else select_competition_native_camera_score_index_v2_9_3(frozen_scores)
        )
        selected_pose = pose_bank[selected_index]
        selected_application = adapter.apply_camera_pose(
            facts,
            _adapter_pose(selected_pose),
            handle=paused_handle,
            source_scene=facts.scene if reset_per_pose else current_scene,
            reset_from_source=reset_per_pose,
            max_settlement_steps=max_settlement_steps,
        )
        selected_score = (
            score_competition_native_editable_camera_application_v2_9_4(
                source_scene=facts.scene,
                pose=selected_pose,
                application=selected_application,
                placement_roster=placement_roster,
                policy=policy,
            )
            if editable_camera
            else score_competition_native_source_camera_application_v2_9_3(
                source_scene=facts.scene,
                pose=selected_pose,
                application=selected_application,
                policy=policy,
            )
        )
        if selected_score != frozen_scores[selected_index]:
            raise ValueError("camera winner replay score changed")
        capture_settlement_pass_steps = facts.settlement_pass_steps
        evidence_source_scene = facts.scene
        if settle_after_unpause:
            adapter.resume_camera_observations(paused_handle)
            paused_handle = None
            settled = adapter.settle_camera_pose(
                facts,
                _adapter_pose(selected_pose),
                source_scene=facts.scene,
                max_settlement_steps=max_settlement_steps,
            )
            selected_application = settled.application
            capture_settlement_pass_steps = settled.settlement_pass_steps
            evidence_source_scene = selected_application.observed_scene
            final_score = (
                score_competition_native_editable_camera_application_v2_9_4(
                    source_scene=evidence_source_scene,
                    pose=selected_pose,
                    application=selected_application,
                    placement_roster=placement_roster,
                    policy=policy,
                )
                if editable_camera
                else score_competition_native_source_camera_application_v2_9_3(
                    source_scene=evidence_source_scene,
                    pose=selected_pose,
                    application=selected_application,
                    policy=policy,
                )
            )
            if final_score != frozen_scores[selected_index]:
                raise ValueError("settled camera winner score changed")
        final_support_by_id = {item.object_id: item for item in facts.support_facts}
        final_support_facts = tuple(
            final_support_by_id[item.object_id]
            for item in selected_application.observed_scene.objects
        )
        if any(
            support.scene_id != selected_application.observed_scene.scene_id
            or support.object_name
            != selected_application.observed_scene.object_by_id(support.object_id).name
            for support in final_support_facts
        ):
            raise ValueError("final support facts do not bind settled scene")
        final_facts = SourceCaptureFacts(
            source=facts.source,
            binding=facts.binding,
            scene=selected_application.observed_scene,
            runtime_identity=facts.runtime_identity,
            observation=selected_application.observation,
            support_facts=final_support_facts,
            floor_envelope=facts.floor_envelope,
            floor_position_regions=facts.floor_position_regions,
            reachable_positions=facts.reachable_positions,
            current_pose=selected_application.observed_pose,
            settlement_pass_steps=capture_settlement_pass_steps,
        )
    except _EXPECTED_CAPTURE_ERRORS:
        camera_selection_failed = True
    finally:
        if paused_handle is not None:
            try:
                adapter.resume_camera_observations(paused_handle)
            except _EXPECTED_CAPTURE_ERRORS:
                camera_selection_failed = True

    if camera_selection_failed:
        return rejected("source_capture:camera_selection_failed")

    try:
        final_maps = adapter.capture_spawn_maps(
            final_facts,
            subject_object_ids=tuple(
                item.object_id
                for item in normalized_baseline.objects
                if item.movable
                and support_by_id[item.object_id].support_kind
                is CompetitionNativeSupportKindV2_9.RECEPTACLE
            ),
        )
        outcome, retained_maps = _capture_selected_camera_source(
            source,
            facts=facts,
            scene=selected_application.observed_scene,
            application=selected_application,
            support_facts=support_facts,
            reachable=reachable,
            spawn_maps=final_maps,
            settlement_pass_steps=capture_settlement_pass_steps,
            floor_clearance_m=floor_clearance_m,
        )
        capture = outcome.capture
        if capture is None:
            return CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
                outcome=outcome,
                surface_evidence=None,
                camera_evidence=None,
            )
        if (
            editable_camera
            and _camera_placement_roster(
                selected_application.observed_scene,
                pairs,
                retained_maps,
            )
            != placement_roster
        ):
            raise ValueError("camera placement roster changed after winner replay")
        surface_evidence = build_competition_native_source_surface_evidence_v2_9_2(
            capture,
            retained_maps,
        )
        evidence_arguments = {
            "source_id": source.source_id,
            "scene_id": source.scene_id,
            "source_locator_sha256": source.source_locator_sha256,
            "runtime_identity_sha256": canonical_sha256(
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
        return CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3(
            outcome=outcome,
            surface_evidence=surface_evidence,
            camera_evidence=camera_evidence,
        )
    except _EXPECTED_CAPTURE_ERRORS:
        return rejected("source_capture:evidence_invalid")


def capture_competition_native_source_v2_9(
    adapter: EnvironmentAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureOutcomeV2_9:
    return _capture_competition_native_source_v2_9_3(
        adapter,
        source,
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
        camera_policy=build_competition_native_camera_policy_v2_9_3(),
    ).outcome


def capture_competition_native_source_v2_9_2(
    adapter: EnvironmentAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2:
    result = _capture_competition_native_source_v2_9_3(
        adapter,
        source,
        max_settlement_steps=max_settlement_steps,
        floor_clearance_m=floor_clearance_m,
        navigation_agent_radius_m=navigation_agent_radius_m,
        navigation_clearance_m=navigation_clearance_m,
        camera_policy=build_competition_native_camera_policy_v2_9_3(),
    )
    return CompetitionNativeSourceCaptureWithSurfaceEvidenceV2_9_2(
        outcome=result.outcome,
        surface_evidence=result.surface_evidence,
    )


def capture_competition_native_source_v2_9_3(
    adapter: EnvironmentAdapter,
    source: CompetitionNativeSourceRefV2_9,
    *,
    max_settlement_steps: int,
    floor_clearance_m: float,
    navigation_agent_radius_m: float,
    navigation_clearance_m: float,
) -> CompetitionNativeSourceCaptureWithCameraEvidenceV2_9_3:
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
    outcome: CompetitionNativeSourceCaptureOutcomeV2_9
    surface_evidence: SourceSurfaceEvidence | None
    camera_evidence: SourceCameraEvidence | None


def capture_source_observation(
    adapter: EnvironmentAdapter,
    *,
    source: CompetitionNativeSourceRefV2_9,
    settings: CaptureSettings,
    camera_policy: CameraPolicy,
) -> SourceCaptureResult:
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


def _apply_resource_caps(
    policy,
    outcomes: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...],
) -> tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...]:
    bounded = []
    candidate_count = 0
    for outcome in outcomes:
        capture = outcome.capture
        if outcome.status == "rejected" or capture is None:
            bounded.append(outcome)
            continue
        object_count = len(capture.scene.objects)
        source_candidates = object_count * max(0, object_count - 1) * len(Relation)
        if object_count > policy.max_objects_per_scene:
            bounded.append(
                _rejected_source(outcome.source, "dataset_capture:object_cap_exceeded")
            )
        elif candidate_count + source_candidates > policy.max_candidates_total:
            bounded.append(
                _rejected_source(
                    outcome.source, "dataset_capture:candidate_cap_exceeded"
                )
            )
        else:
            bounded.append(outcome)
            candidate_count += source_candidates
    return tuple(bounded)


def _capture_source_with_adapter(
    source: CompetitionNativeSourceRefV2_9,
    procedural: Mapping[str, AdapterProceduralScene] | None,
    plan: CapturePlan,
    adapter_factory: Callable[..., EnvironmentAdapter],
    fresh_transitions=None,
) -> tuple[
    CompetitionNativeSourceCaptureOutcomeV2_9,
    SourceSurfaceEvidence | None,
    SourceCameraEvidence | None,
]:
    manager = adapter_factory(
        [source.scene_id],
        width=plan.roster_policy.width,
        height=plan.roster_policy.height,
        seed=plan.roster_policy.seed,
        procedural_scenes=procedural,
    )
    try:
        adapter = manager.__enter__()
    except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
        return (
            _rejected_source(source, "dataset_capture:adapter_lifecycle_failed"),
            None,
            None,
        )
    try:
        result = _capture_competition_native_source_v2_9_3(
            adapter,
            source,
            max_settlement_steps=plan.capture_settings.max_settlement_steps,
            floor_clearance_m=plan.capture_settings.floor_clearance_m,
            navigation_agent_radius_m=plan.capture_settings.navigation_agent_radius_m,
            navigation_clearance_m=plan.capture_settings.navigation_clearance_m,
            camera_policy=plan.roster_policy.camera_policy,
            captured_source_consumer=(
                (lambda captured: fresh_transitions.capture(source.source_id, captured))
                if fresh_transitions is not None
                else None
            ),
        )
    except BaseException as error:
        try:
            manager.__exit__(type(error), error, error.__traceback__)
        except MemoryError:
            raise
        except Exception as cleanup_error:  # noqa: BLE001
            error.add_note(f"adapter cleanup also failed: {cleanup_error}")
        raise
    try:
        manager.__exit__(None, None, None)
    except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
        return (
            _rejected_source(source, "dataset_capture:adapter_lifecycle_failed"),
            None,
            None,
        )
    return result.outcome, result.surface_evidence, result.camera_evidence


def _capture_dataset(
    plan: CapturePlan,
    *,
    adapter_factory: Callable[..., EnvironmentAdapter],
    dataset_loader: Callable[[str, str], object],
    fresh_transitions=None,
):
    if type(plan) is not CapturePlan:
        raise TypeError("dataset capture plan must be exact")
    checked = CapturePlan.model_validate(
        plan.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    datasets: dict[tuple[str, str], object] = {}
    outcomes: list[CompetitionNativeSourceCaptureOutcomeV2_9] = []
    surface_evidence = []
    camera_evidence = []
    source_locators = {
        entry.source_id: entry.source for entry in checked.source_manifest.sources
    }
    for source in checked.roster_policy.sources:
        locator = source_locators[source.source_id]
        procedural: dict[str, AdapterProceduralScene] = {}
        if type(locator) is ProceduralSource:
            try:
                procedural[source.scene_id] = _resolve_procedural_source(
                    locator,
                    datasets,
                    dataset_loader,
                )
            except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
                outcomes.append(
                    _rejected_source(source, "dataset_capture:source_resolution_failed")
                )
                continue
        elif type(locator) is not LegacyAI2ThorSource:
            raise TypeError("dataset source manifest contains an unsupported source")
        outcome, source_surface, source_camera = _capture_source_with_adapter(
            source,
            procedural or None,
            checked,
            adapter_factory,
            fresh_transitions=fresh_transitions,
        )
        outcomes.append(
            CompetitionNativeSourceCaptureOutcomeV2_9.model_validate_json(
                canonical_json_bytes(outcome),
                strict=True,
            )
        )
        if source_surface is not None:
            surface_evidence.append(source_surface)
        if source_camera is not None:
            camera_evidence.append(source_camera)
    bounded = _apply_resource_caps(checked.roster_policy, tuple(outcomes))
    accepted_source_ids = {
        item.source.source_id for item in bounded if item.status == "accepted"
    }
    return compile_roster(
        checked.roster_policy,
        bounded,
        tuple(
            item for item in surface_evidence if item.source_id in accepted_source_ids
        ),
        tuple(
            item for item in camera_evidence if item.source_id in accepted_source_ids
        ),
    )


def capture_dataset(
    plan: CapturePlan,
    *,
    adapter_factory: Callable[..., EnvironmentAdapter],
    dataset_loader: Callable[[str, str], object],
):
    return _capture_dataset(
        plan,
        adapter_factory=adapter_factory,
        dataset_loader=dataset_loader,
    )


def _capture_dataset_with_transitions(
    plan: CapturePlan,
    *,
    adapter_factory: Callable[..., EnvironmentAdapter],
    dataset_loader: Callable[[str, str], object],
    fresh_transitions,
):
    return _capture_dataset(
        plan,
        adapter_factory=adapter_factory,
        dataset_loader=dataset_loader,
        fresh_transitions=fresh_transitions,
    )


def _capture_and_publish_dataset_with_transitions(
    plan: CapturePlan,
    roster_root,
    *,
    adapter_factory: Callable[..., EnvironmentAdapter],
    dataset_loader: Callable[[str, str], object],
    fresh_transitions,
):
    return publish_roster(
        _capture_dataset(
            plan,
            adapter_factory=adapter_factory,
            dataset_loader=dataset_loader,
            fresh_transitions=fresh_transitions,
        ),
        roster_root,
    )


def capture_and_publish_dataset(
    plan: CapturePlan,
    roster_root,
    *,
    adapter_factory: Callable[..., EnvironmentAdapter],
    dataset_loader: Callable[[str, str], object],
):
    return publish_roster(
        capture_dataset(
            plan,
            adapter_factory=adapter_factory,
            dataset_loader=dataset_loader,
        ),
        roster_root,
    )


__all__ = (
    "SourceCaptureResult",
    "capture_and_publish_dataset",
    "capture_competition_native_source_v2_9",
    "capture_competition_native_source_v2_9_2",
    "capture_competition_native_source_v2_9_3",
    "capture_dataset",
    "capture_source_observation",
    "load_prior_dataset",
)
