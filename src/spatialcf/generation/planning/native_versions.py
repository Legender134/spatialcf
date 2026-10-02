"""Explicit compatibility matrix for native geometry and planning evidence."""

import json
from spatialcf.generation.errors import require_wire_version

LEGACY_GUARD = "competition-native-source-view-guard:2.9.5"
PERSISTENT_GUARD = "competition-native-source-view-guard:2.9.6"
LEGACY_POLICY = "competition-native-source-policy:2.9.13"
PERSISTENT_POLICY = "competition-native-source-policy:2.9.14"
LEGACY_PLAN = "competition-native-source-plan:2.9.10"
PERSISTENT_PLAN = "competition-native-source-plan:2.9.11"
LEGACY_ENDPOINT = "competition-native-endpoint-plan:2.9.5"
PERSISTENT_ENDPOINT = "competition-native-endpoint-plan:2.9.6"
WORLD_AABB_RUNTIME = "ai2thor-native-xzy-to-rh-z-up-world-aabb-grid-v2"
LEGACY_CANDIDATES = (
    "CAMERA_SELECTED_CAPTURE_BOUND_NATIVE_REPLAY_PARALLEL_PATCH_"
    "RELATION_RANKED_RUNTIME_COLLISION_DELEGATED_BBOX_VISIBILITY"
)
CLEAR_FIRST_CANDIDATES = "SOURCE_NATIVE_CLEAR_FIRST_V1"


def policy_for_captures(captures):
    flags = tuple(item.runtime_identity.coordinate_transform_version == WORLD_AABB_RUNTIME
                  for item in captures)
    if any(flags) and not all(flags):
        raise ValueError("mixed native geometry versions in source campaign")
    return PERSISTENT_POLICY if flags and all(flags) else LEGACY_POLICY


def guard_for_policy(version):
    return {LEGACY_POLICY: LEGACY_GUARD, PERSISTENT_POLICY: PERSISTENT_GUARD}[version]


def plan_for_policy(version):
    return {LEGACY_POLICY: LEGACY_PLAN, PERSISTENT_POLICY: PERSISTENT_PLAN}[version]


def endpoint_for_guard(version):
    return {LEGACY_GUARD: LEGACY_ENDPOINT, PERSISTENT_GUARD: PERSISTENT_ENDPOINT}[version]


def hash_domain(version):
    return "spatialcf." + version.replace(":", ".v")


def require_planning_wire(payload, *, artifact_kind, field, expected):
    successors = {LEGACY_PLAN: PERSISTENT_PLAN, LEGACY_POLICY: PERSISTENT_POLICY}
    try:
        decoded = json.loads(payload)
    except (ValueError, UnicodeError):
        decoded = None
    observed = decoded.get(field) if type(decoded) is dict else None
    selected = observed if observed == successors.get(expected) else expected
    return require_wire_version(payload, artifact_kind=artifact_kind, field=field, expected=selected)
