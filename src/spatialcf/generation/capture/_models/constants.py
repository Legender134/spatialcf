"""Capture constants: exact contracts and pure derivation."""

from __future__ import annotations

import math

from typing import (
    Literal,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)


_PAIR_CAMERA_RADIUS_M = 2.0


_PAIR_CAMERA_DIRECTIONS = (
    (0.0, -1.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
)


_PAIR_CAMERA_HORIZONS_DEGREES = (0.0, 30.0)


_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_5 = 0.2


_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_6 = 0.21


_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_7 = 0.25


_CAMERA_AGENT_CLEARANCE_RADIUS_M_V2_9_8 = 0.25 + math.sqrt(2.0) * 0.5e-6


_MAX_POSE_BANK_MEMBERS = 256


_MAX_POSITION_RESIDUAL_M = 1e-5


_MAX_ANGLE_RESIDUAL_DEGREES = 1e-4


_POLICY_HASH_DOMAIN = "spatialcf.competition-native-camera-policy.v2.9.3"


_POSE_BANK_HASH_DOMAIN = "spatialcf.competition-native-camera-pose-bank.v2.9.3"


_EVIDENCE_HASH_DOMAIN = "spatialcf.competition-native-source-camera-evidence.v2.9.3"


_PLACEMENT_ROSTER_HASH_DOMAIN = (
    "spatialcf.competition-native-camera-placement-roster.v2.9.4"
)


_LEGACY_POSE_POLICY_VERSION = "deterministic-pair-camera-tier-1:1"


_SOLVER_UPRIGHT_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright:2"
)


_EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain:3"
)


_COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.2m:4"
)


_CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m:5"
)


_RESET_PER_POSE_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.21m-reset-per-pose:6"
)


_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m:7"
)


_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused:8"
)


_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION = (
    "deterministic-pair-camera-tier-1-solver-upright-edit-domain-"
    "movable-clearance-0.25m-physics-paused-final-settle:9"
)


_PATCH_HASH_DOMAIN = "spatialcf.competition-native-receptacle-surface-patch.v2.9.2"


_SUBJECT_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-subject-surface-evidence.v2.9.2"
)


_SOURCE_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-source-surface-evidence.v2.9.2"
)


_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"


DatasetSplitV2_9 = Literal["train", "validation", "test"]


_CURRENT_POLICY_HASH_DOMAIN = (
    "spatialcf.competition-native-candidate-roster-policy.v2.9.4"
)


_SOURCE_CAPTURE_HASH_DOMAIN = "spatialcf.competition-native-source-capture.v2.9"


_MANIFEST_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-manifest.v2.9"


_SUMMARY_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-summary.v2.9"


_PLACEMENT_HASH_DOMAIN = "spatialcf.competition-native-placement-fact.v2.9"


_MAX_TEXT_CHARS = 512


_MAX_REASON_CHARS = 256


_MAX_REASONS = 8


_MAX_POLICY_SOURCES = 512


_MAX_OBJECTS_PER_SCENE = 96


_MAX_CANDIDATES_TOTAL = 40_000


_MAX_REQUESTS_TOTAL = 1_000


_MAX_NATIVE_POSITIONS = 10_000


_MAX_CAMERAS_PER_SCENE = 64


_MAX_OBSTACLES_PER_SCENE = 2_048


_MAX_REGIONS_PER_SCENE = 2_048


_MAX_POLYGON_VERTICES = 4_096


_SOURCE_CAPTURE_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024 - 16 * 1024


_MAX_PERSISTED_REQUEST_TEXT_CHARS = _MAX_TEXT_CHARS


_SOURCE_VIEW_FACT_HASH_DOMAIN = (
    "spatialcf.competition-native-source-view-fact.v2.9.5"
)


_SOURCE_VIEW_BINDING_HASH_DOMAIN = "spatialcf.source-view-binding.v1"


_SOURCE_VIEW_SAMPLING_POLICY_SHA256 = canonical_sha256(
    {
        "local_quantization_m": 1e-5,
        "maximum_samples": 76_800,
        "render_proxy": "weighted_sampled_splat_z_buffer",
        "representative": "minimum_finite_positive_depth_then_row_column",
        "rounding": "nearest_even",
        "tile_height": 2,
        "tile_width": 2,
        "weight": "object_mask_pixel_count_in_tile",
    },
    domain="spatialcf.competition-native-source-view-policy.v2.9.5",
)
