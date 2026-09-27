"""Upright compiler constants; explicit pure implementation owner."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from spatialcf.domain import (
    upright_se2 as upright,
)


_INPUT_FAMILY_REF = "definition:spatialcf/upright-se2/compiler-input/1.0"


_INPUT_SCHEMA_REF = "schema:spatialcf/upright-se2/compiler-input/1.0"


_STATE_FAMILY_REF = "definition:spatialcf/upright-se2/state/1.0"


_STATE_SCHEMA_REF = "schema:spatialcf/upright-se2/state/1.0"


_SCENE_SCHEMA_REF = "schema:spatialcf/canonical-scene/2.3"


_BACKEND_REF = "backend:spatialcf/upright-se2/cardinal"


_CONTINUOUS_BACKEND_REF = "backend:spatialcf/upright-se2/continuous"


_BACKEND_BUILD_SHA256 = "b" * 64


_CHECKER_BUILD_SHA256 = "a" * 64


_DEPENDENCY_LOCK_SHA256 = "d" * 64


_DEFINITION_CLOSURE_REF = "definition:spatialcf/upright-se2/definition-closure/1.0"


_SOLVE_POLICY_REF = "definition:spatialcf/upright-se2/solve-policy/1.0"


_DEFINITION_KIND_REF = "definition:spatialcf/upright-se2/definition-kind/1.0"


_DERIVED_RULE_REF = "definition:spatialcf/upright-se2/derived-pose-and-facts/1.0"


_CONTINUOUS_UNAVAILABLE_REF = (
    "definition:spatialcf/upright-se2/continuous-capability-unavailable/1.0"
)


_COMPILER_INPUT_FACT_KEY = "fact-key:spatialcf/upright-se2/compiler-input"


_PIVOT_STATE_HASH_DOMAIN = "spatialcf/counterfactual/upright-se2/pivot-state/3.0"


_POSE_STATE_HASH_DOMAIN = "spatialcf/counterfactual/upright-se2/base-pose/3.0"


_UNCHANGED_LEAVES_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/unchanged-leaves/3.0"
)


_RETAINED_OWNER_BOUND_SCHEMA_REF = upright.UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF


_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/finite-real/1.0"


_INTEGER_SCHEMA_REF = "schema:spatialcf/upright-se2/integer/1.0"


_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/canonical-id/1.0"


_DIGEST_SCHEMA_REF = "schema:spatialcf/upright-se2/digest/1.0"


_ENUM_SCHEMA_REF = "schema:spatialcf/upright-se2/enum-symbol/1.0"


_YAW_ARGUMENT_SCHEMA_REF = "schema:spatialcf/upright-se2/yaw-argument/1.0"


_DEFINITION_CLOSURE_SCHEMA_REF = "schema:spatialcf/upright-se2/definition-closure/1.0"


_SOLVE_POLICY_CLOSURE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/solve-policy-closure/1.0"
)


_CARDINAL_TURN_FRACTIONS = {
    0: Fraction(0),
    1: Fraction(1, 4),
    2: Fraction(-1, 2),
    3: Fraction(-1, 4),
}


_PRIMARY_ROLES = (
    "subject-world-x",
    "subject-world-y",
    "subject-explicit-yaw",
)


_DERIVED_ROLES = (
    "subject-derived-canonical-pose",
    "subject-derived-collision",
    "subject-derived-support",
    "subject-derived-relation",
    "subject-derived-visibility",
)


_SOURCE_FIELDS = (
    ("objects", "object_id"),
    ("geometry-instances", "geometry_id"),
    ("collision-bodies", "body_id"),
    ("workspace-boundaries", "fact_id"),
    ("known-free-spaces", "fact_id"),
    ("support-surfaces", "surface_id"),
    ("cameras", "camera_id"),
    ("baseline-observations", "observation_id"),
)


_WORLD_XY_FRAME_REF = "definition:spatialcf/upright-se2/world-xy/1.0"


_METRE_UNIT_REF = "definition:spatialcf/upright-se2/metre/1.0"


_CLOSED_INTERVAL_TOPOLOGY_REF = "definition:spatialcf/upright-se2/closed-interval/1.0"
