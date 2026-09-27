"""Public capture model aliases; each contract family has an explicit owner."""

from __future__ import annotations

import math

from dataclasses import (
    asdict,
)

from hashlib import (
    sha256,
)

from typing import (
    Literal,
    Self,
)

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterObservation,
    AdapterPose,
    AdapterPosition,
    AdapterRuntimeIdentity,
    AdapterSpawnMap,
    AdapterSurfacePatch,
)

from spatialcf.domain.base import (
    CanonicalModel,
    FiniteFloat,
    Sha256Digest,
)

from spatialcf.domain.scene import (
    OBB,
    Camera,
    Quaternion,
    Scene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.relations.engine import (
    RelationEngine,
)

from spatialcf.domain.base import CanonicalId

from enum import (
    StrEnum,
)

from spatialcf.domain.request import (
    Relation,
)

from spatialcf.domain.scene import (
    BBox2D,
    SubjectPositionRegion,
    Vec2,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.generation.capture.reachability_contracts import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)

from spatialcf.generation.capture._models.camera_contracts import (
    _editable_pose_policy_versions,
    CompetitionNativeCameraPoseV2_9_3,
    _policy_payload,
    CompetitionNativeCameraPolicyV2_9_3,
    CompetitionNativeCameraScoreV2_9_3,
    CompetitionNativeCameraPlacementPositionV2_9_4,
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraScoreV2_9_4,
    CompetitionNativeCameraScoreFamilyV2_9_3,
    _strict_placement_roster_v2_9_4,
    competition_native_camera_placement_roster_sha256_v2_9_4,
    select_competition_native_camera_score_index_v2_9_4,
    _strict_score_ledger,
    select_competition_native_camera_score_index_v2_9_3,
    _select_competition_native_camera_score_index,
    verify_competition_native_camera_observation_binding_v2_9_3,
    _evidence_payload,
    CompetitionNativeSourceCameraEvidenceV2_9_3,
    _strict_scene,
    CameraPolicy,
    SourceCameraEvidence,
)

from spatialcf.generation.capture._models.camera_evidence import (
    build_competition_native_source_camera_evidence_v2_9_3,
    build_competition_native_source_camera_evidence_v2_9_4,
    select_competition_native_source_camera_evidence_v2_9_3,
    verify_competition_native_source_camera_evidence_v2_9_3,
    verify_source_camera_evidence,
)

from spatialcf.generation.capture._models.camera_geometry import (
    _CameraConversionError,
    _expected_camera_world_to_camera,
    _pair_midpoint,
    _validate_pair_camera_inputs,
    _ring_positions,
    deterministic_pair_camera_poses,
    _clearance_rotation_matrix,
    _projected_corners,
    _cross,
    _convex_hull,
    _point_segment_distance,
    _point_polygon_distance,
    _filter_competition_native_camera_positions,
    filter_competition_native_camera_positions_v2_9_5,
    filter_competition_native_camera_positions_v2_9_6,
    filter_competition_native_camera_positions_v2_9_7,
    filter_competition_native_camera_positions_v2_9_8,
    _legacy_camera,
)

from spatialcf.generation.capture._models.camera_scoring import (
    _strict_native_pose,
    _wire_pose,
    _pose_key,
    build_competition_native_camera_pose_bank_v2_9_3,
    build_competition_native_camera_policy_v2_9_3,
    build_competition_native_solver_upright_camera_policy_v2_9_3,
    build_competition_native_editable_camera_policy_v2_9_3,
    build_competition_native_collision_safe_editable_camera_policy_v2_9_5,
    build_competition_native_contact_margin_editable_camera_policy_v2_9_6,
    build_competition_native_reset_per_pose_editable_camera_policy_v2_9_7,
    build_competition_native_grid_margin_editable_camera_policy_v2_9_8,
    build_competition_native_paused_camera_policy_v2_9_9,
    build_competition_native_settled_camera_policy_v2_9_10,
    _strict_pose_bank,
    competition_native_camera_pose_bank_sha256_v2_9_3,
    score_competition_native_camera_scene_v2_9_3,
    _rotation_matrix_values,
    _translated_obb_fully_visible_v2_9_4,
    score_competition_native_editable_camera_scene_v2_9_4,
    score_competition_native_editable_camera_application_v2_9_4,
    _strict_application,
    _validate_application_source_closure,
    score_competition_native_source_camera_application_v2_9_3,
    _strict_policy,
    verify_competition_native_solver_camera_binding_v2_9_3,
    build_settled_camera_policy,
)

from spatialcf.generation.capture._models.constants import (
    _PAIR_CAMERA_RADIUS_M,
    _PAIR_CAMERA_DIRECTIONS,
    _PAIR_CAMERA_HORIZONS_DEGREES,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7,
    _CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8,
    _MAX_POSE_BANK_MEMBERS,
    _MAX_POSITION_RESIDUAL_M,
    _MAX_ANGLE_RESIDUAL_DEGREES,
    _POLICY_HASH_DOMAIN,
    _POSE_BANK_HASH_DOMAIN,
    _EVIDENCE_HASH_DOMAIN,
    _PLACEMENT_ROSTER_HASH_DOMAIN,
    _LEGACY_POSE_POLICY_VERSION,
    _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
    _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
    _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
    _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
    _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
    _GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _PATCH_HASH_DOMAIN,
    _SUBJECT_EVIDENCE_HASH_DOMAIN,
    _SOURCE_EVIDENCE_HASH_DOMAIN,
    _RUNTIME_IDENTITY_HASH_DOMAIN,
    DatasetSplitV2_9,
    _CURRENT_POLICY_HASH_DOMAIN,
    _SOURCE_CAPTURE_HASH_DOMAIN,
    _MANIFEST_HASH_DOMAIN,
    _SUMMARY_HASH_DOMAIN,
    _PLACEMENT_HASH_DOMAIN,
    _MAX_TEXT_CHARS,
    _MAX_REASON_CHARS,
    _MAX_REASONS,
    _MAX_POLICY_SOURCES,
    _MAX_OBJECTS_PER_SCENE,
    _MAX_CANDIDATES_TOTAL,
    _MAX_REQUESTS_TOTAL,
    _MAX_NATIVE_POSITIONS,
    _MAX_CAMERAS_PER_SCENE,
    _MAX_OBSTACLES_PER_SCENE,
    _MAX_REGIONS_PER_SCENE,
    _MAX_POLYGON_VERTICES,
    _SOURCE_CAPTURE_PAYLOAD_MAX_BYTES,
    _MAX_PERSISTED_REQUEST_TEXT_CHARS,
    _SOURCE_VIEW_FACT_HASH_DOMAIN,
    _SOURCE_VIEW_BINDING_HASH_DOMAIN,
    _SOURCE_VIEW_SAMPLING_POLICY_SHA256,
)

from spatialcf.generation.capture._models.contracts import (
    CompetitionNativeSupportKindV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSubjectStateV2_9,
    CompetitionNativeCandidateStateV2_9,
    CompetitionNativeSourceRefV2_9,
    RosterPolicy,
    CompetitionNativeCandidateRosterPolicyV2_9_4,
    CompetitionNativeRuntimeIdentityV2_9,
    validate_competition_native_runtime_source_lineage_v2_9,
    CompetitionNativeSupportFactV2_9,
    CompetitionNativePositionV2_9,
    CompetitionNativeFloorEnvelopeV2_9,
    _placement_payload,
    CompetitionNativeSubjectPlacementFactV2_9,
    build_competition_native_subject_placement_fact_v2_9,
)

from spatialcf.generation.capture._models.roster import (
    competition_native_roster_selection_identity_v2_9,
    CompetitionNativeSourceCaptureOutcomeV2_9,
    CompetitionNativeObjectInventoryV2_9,
    CompetitionNativeCandidateInventoryV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativeRosterRejectionV2_9,
    CompetitionNativeStageCountV2_9,
    CompetitionNativeCandidateStateCountV2_9,
    CompetitionNativeRelationCountV2_9,
    RosterSummary,
    CompetitionNativeCandidateRosterSummaryV2_9,
    RosterCompilation,
    CompetitionNativeCandidateRosterCompilationV2_9_4,
)

from spatialcf.generation.capture._models.source import (
    SourceViewObjectSamples,
    SourceViewFact,
    _capture_payload,
    normalize_competition_native_source_scene_v2_9,
    _point_in_polygon_or_boundary,
    _validate_floor_envelope_binding,
    validate_competition_native_floor_envelope_v2_9,
    CompetitionNativeSourceCaptureV2_9,
    build_competition_native_source_capture_v2_9,
)

from spatialcf.generation.capture._models.surfaces import (
    _patch_payload,
    CompetitionNativeReceptacleSurfacePatchV2_9_2,
    _subject_payload,
    CompetitionNativeSubjectSurfaceEvidenceV2_9_2,
    _source_payload,
    CompetitionNativeSourceSurfaceEvidenceV2_9_2,
    _strict_spawn_map,
    _build_patch,
    _accepted_capture_scene_sha256,
    verify_competition_native_source_surface_evidence_v2_9_2,
    build_competition_native_source_surface_evidence_v2_9_2,
    ReceptacleSurfacePatch,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    build_source_surface_evidence,
    verify_source_surface_evidence,
)

__all__ = (
    "CompetitionNativeCandidateInventoryV2_9",
    "CompetitionNativeCandidateRosterCompilationV2_9_4",
    "CompetitionNativeCandidateRosterManifestV2_9",
    "CompetitionNativeCandidateRosterPolicyV2_9_4",
    "CompetitionNativeCandidateRosterSummaryV2_9",
    "CompetitionNativeCandidateStateCountV2_9",
    "CompetitionNativeCandidateStateV2_9",
    "CompetitionNativeFloorEnvelopeV2_9",
    "CompetitionNativeObjectInventoryV2_9",
    "CompetitionNativePlacementAvailabilityV2_9",
    "CompetitionNativePositionV2_9",
    "CompetitionNativeRelationCountV2_9",
    "CompetitionNativeRosterRejectionV2_9",
    "CompetitionNativeRuntimeIdentityV2_9",
    "CompetitionNativeSelectedRequestV2_9",
    "CompetitionNativeSourceCaptureOutcomeV2_9",
    "CompetitionNativeSourceCaptureV2_9",
    "CompetitionNativeSourceRefV2_9",
    "CompetitionNativeStageCountV2_9",
    "CompetitionNativeSubjectPlacementFactV2_9",
    "CompetitionNativeSubjectStateV2_9",
    "CompetitionNativeSupportFactV2_9",
    "CompetitionNativeSupportKindV2_9",
    "DatasetSplitV2_9",
    "RosterCompilation",
    "RosterPolicy",
    "RosterSummary",
    "build_competition_native_source_capture_v2_9",
    "build_competition_native_subject_placement_fact_v2_9",
    "normalize_competition_native_source_scene_v2_9",
    "validate_competition_native_floor_envelope_v2_9",
    "validate_competition_native_runtime_source_lineage_v2_9",
)
