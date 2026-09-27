"""Upright profile constants contracts and intrinsic operations."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from typing import (
    TypeVar,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)


UPRIGHT_SE2_PROFILE_REF = "spatialcf/upright_se2@1"


UPRIGHT_SE2_SEMANTICS_PROFILE_REF = "spatialcf/upright_se2/semantics@1"


UPRIGHT_SE2_YAW_TO_POSE_RULE_REF = "definition:spatialcf/upright-se2/yaw-to-pose/1.0"


UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/derived-source-fact/3.0"
)


UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/proof-material/1.0"
)


UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/cardinal-proof-material/1.0"
)


UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR = "UPRIGHT_SE2_CARDINAL_PROOF_MATERIAL_V1"


UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/continuous-proof-material/1.0"
)


UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/continuous-proof-material/1.0"
)


UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR = (
    "UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_V1"
)


UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/solve-policy/1.0"
)


UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/solve-policy-closure/1.0"
)


UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/claim-certified-solution/1.0"
)


UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/claim-finite-gap-solution/1.0"
)


_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/pivot-state/3.0"
)


_UPRIGHT_SE2_UNCHANGED_LEAVES_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/unchanged-leaves/3.0"
)


_UPRIGHT_SE2_YAW_TO_POSE_MAX_ULPS = 8


_UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF = (
    "definition:spatialcf/upright-se2/compiler-input/1.0"
)


_UPRIGHT_SE2_COMPILER_INPUT_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/compiler-input/1.0"
)


_UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY = "fact-key:spatialcf/upright-se2/compiler-input"


_UPRIGHT_SE2_STATE_FAMILY_REF = "definition:spatialcf/upright-se2/state/1.0"


_UPRIGHT_SE2_STATE_SCHEMA_REF = "schema:spatialcf/upright-se2/state/1.0"


_UPRIGHT_SE2_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/finite-real/1.0"


_UPRIGHT_SE2_INTEGER_SCHEMA_REF = "schema:spatialcf/upright-se2/integer/1.0"


_UPRIGHT_SE2_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/canonical-id/1.0"


_UPRIGHT_SE2_DIGEST_SCHEMA_REF = "schema:spatialcf/upright-se2/digest/1.0"


_UPRIGHT_SE2_ENUM_SCHEMA_REF = "schema:spatialcf/upright-se2/enum-symbol/1.0"


_UPRIGHT_SE2_YAW_ARGUMENT_SCHEMA_REF = "schema:spatialcf/upright-se2/yaw-argument/1.0"


_UPRIGHT_SE2_WORLD_XY_FRAME_REF = "definition:spatialcf/upright-se2/world-xy/1.0"


_UPRIGHT_SE2_METRE_UNIT_REF = "definition:spatialcf/upright-se2/metre/1.0"


_UPRIGHT_SE2_CLOSED_INTERVAL_TOPOLOGY_REF = (
    "definition:spatialcf/upright-se2/closed-interval/1.0"
)


_UPRIGHT_SE2_DERIVED_RULE_REF = (
    "definition:spatialcf/upright-se2/derived-pose-and-facts/1.0"
)


_UPRIGHT_SE2_PRIMARY_ROLES = (
    "subject-world-x",
    "subject-world-y",
    "subject-explicit-yaw",
)


_UPRIGHT_SE2_DERIVED_ROLES = (
    "subject-derived-canonical-pose",
    "subject-derived-collision",
    "subject-derived-support",
    "subject-derived-relation",
    "subject-derived-visibility",
)


_UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS = {
    0: Fraction(0),
    1: Fraction(1, 4),
    2: Fraction(-1, 2),
    3: Fraction(-1, 4),
}


_UPRIGHT_SE2_CARDINAL_QUATERNIONS = {
    0: (0.0, 1.0),
    1: (0.7071067811865476, 0.7071067811865476),
    2: (1.0, 0.0),
    3: (-0.7071067811865476, 0.7071067811865476),
}


UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/cardinal-own-pivot/1.0"
)


UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/cardinal-reference-pivot/1.0"
)


UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/continuous-own-pivot/1.0"
)


UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/continuous-reference-pivot/1.0"
)


UPRIGHT_SE2_OPERATOR_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_PROFILE_CAPABILITY_REF = "capability:spatialcf/upright-se2/profile/1"


UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/compiler/1"
)


UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/backend/1"
)


UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/checker/1"
)


UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/compiler/1"
)


UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/backend/1"
)


UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/checker/1"
)


UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_STAGED_CAPABILITY_REFS = tuple(
    sorted(
        (
            *UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
            *UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_BACKEND_OWNER_REF = "owner:spatialcf/upright-se2/backend"


UPRIGHT_SE2_CHECKER_OWNER_REF = "owner:spatialcf/upright-se2/checker"


UPRIGHT_SE2_COMPILER_OWNER_REF = "owner:spatialcf/upright-se2/compiler"


UPRIGHT_SE2_COMPILER_BUILD_SHA256 = "c" * 64


UPRIGHT_SE2_CHECKER_BUILD_SHA256 = "a" * 64


UPRIGHT_SE2_COLLISION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/collision-clearance-contact/1.0"
)


UPRIGHT_SE2_SUPPORT_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/same-surface-support/1.0"
)


UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/target-relation/1.0"
)


UPRIGHT_SE2_PRESERVATION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/preservation/1.0"
)


UPRIGHT_SE2_VISIBILITY_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/fixed-camera-visibility/1.0"
)


UPRIGHT_SE2_PREDICATE_DEFINITION_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_COLLISION_PREDICATE_REF,
            UPRIGHT_SE2_SUPPORT_PREDICATE_REF,
            UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
            UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
            UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/objective-five-term/1.0"
)


UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-predicate-evaluator/1"
)


UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-predicate-verifier/1"
)


UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-objective-evaluator/1"
)


UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-objective-verifier/1"
)


UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)


UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)


_ValueT = TypeVar("_ValueT")


_SEMANTIC_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-id/1.0"


_SEMANTIC_SYMBOL_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-symbol/1.0"


_SEMANTIC_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-real/1.0"


_SEMANTIC_TUPLE_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-tuple/1.0"


UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/semantic-definition-closure/1.0"
)


_SEMANTIC_DEFINITION_KIND_REF = (
    "definition:spatialcf/upright-se2/semantic-definition-body/1.0"
)


_OBJECTIVE_DEFINITION_KIND_REF = (
    "definition:spatialcf/upright-se2/five-term-objective-body/1.0"
)


_SEMANTIC_CLOSURE_KIND_REF = (
    "definition:spatialcf/upright-se2/semantic-closure-body/1.0"
)


_SEMANTIC_CLOSURE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/semantic-definition-closure/1.0"
)


_OBJECTIVE_POLICY_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/five-term-objective-policy/1.0"
)


_EXECUTABLE_POLICY_FAMILY_REF = "definition:spatialcf/upright-se2/executable-policy/1.0"


_EXECUTABLE_POLICY_WIRE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/executable-policy-wire/1.0"
)


_EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/executable-policy-bundle/1.0"
)


_EXECUTABLE_POLICY_BUNDLE_FACT_KEY = (
    "fact-key:spatialcf/upright-se2/executable-policy-bundle"
)


_EXECUTABLE_POLICY_KEYS = (
    "collision",
    "numeric",
    "objective",
    "preservation",
    "relation:BEHIND",
    "relation:FAR",
    "relation:FRONT",
    "relation:LEFT",
    "relation:NEAR",
    "relation:RIGHT",
    "resource",
    "safety",
    "support",
    "visibility",
)


_EXECUTABLE_POLICY_SCHEMA_BY_KEY = {
    key: f"schema:spatialcf/upright-se2/executable-{key.replace(':', '-').lower()}-policy/1.0"
    for key in _EXECUTABLE_POLICY_KEYS
}


_SEMANTIC_KIND_TO_REF = {
    "COLLISION": UPRIGHT_SE2_COLLISION_PREDICATE_REF,
    "SUPPORT": UPRIGHT_SE2_SUPPORT_PREDICATE_REF,
    "TARGET_RELATION": UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
    "PRESERVATION": UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
    "VISIBILITY": UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
}


_SEMANTIC_KIND_TO_SCHEMA = {
    kind: f"schema:spatialcf/upright-se2/{kind.lower()}-semantics/1.0"
    for kind in _SEMANTIC_KIND_TO_REF
}


_SEMANTIC_BODY_VALUES = {
    "COLLISION": (
        (
            "clearance_measurement_ref",
            "definition:spatialcf/upright-se2/collision-clearance/1.0",
        ),
        (
            "contact_comparator_ref",
            "definition:spatialcf/upright-se2/collision-contact-comparator/1.0",
        ),
        (
            "contact_boundary_policy_ref",
            "definition:spatialcf/upright-se2/collision-closed-contact-boundary/1.0",
        ),
        (
            "evaluation_frame_ref",
            "definition:spatialcf/upright-se2/world-xy/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/collision-evaluation-input/1.0",
        ),
        (
            "collision_evaluator_ref",
            "definition:spatialcf/upright-se2/collision-evaluator/1.0",
        ),
        (
            "obstacle_roster_selector_ref",
            "definition:spatialcf/upright-se2/collision-complete-obstacle-roster/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/collision-source-exactness/1.0",
        ),
        (
            "subject_geometry_selector_ref",
            "definition:spatialcf/upright-se2/collision-subject-geometry/1.0",
        ),
    ),
    "SUPPORT": (
        (
            "clearance_policy_ref",
            "definition:spatialcf/upright-se2/support-clearance/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/support-evaluation-input/1.0",
        ),
        (
            "containment_boundary_policy_ref",
            "definition:spatialcf/upright-se2/support-contained-closed-boundary/1.0",
        ),
        (
            "height_contact_comparator_ref",
            "definition:spatialcf/upright-se2/support-height-contact-comparator/1.0",
        ),
        (
            "same_surface_comparator_ref",
            "definition:spatialcf/upright-se2/support-same-surface-comparator/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/support-source-exactness/1.0",
        ),
        (
            "support_surface_selector_ref",
            "definition:spatialcf/upright-se2/support-required-surface/1.0",
        ),
        (
            "surface_frame_requirement_ref",
            "definition:spatialcf/upright-se2/support-world-plus-z-frame/1.0",
        ),
        (
            "surface_normal_requirement_ref",
            "definition:spatialcf/upright-se2/support-world-plus-z-normal/1.0",
        ),
    ),
    "TARGET_RELATION": (
        (
            "after_state_geometry_selector_ref",
            "definition:spatialcf/upright-se2/relation-after-state-geometry/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/target-relation-evaluation-input/1.0",
        ),
        (
            "relation_boundary_policy_ref",
            "definition:spatialcf/upright-se2/relation-closed-boundary/1.0",
        ),
        (
            "relation_comparator_ref",
            "definition:spatialcf/upright-se2/relation-comparator/1.0",
        ),
        (
            "relation_frame_ref",
            "definition:spatialcf/upright-se2/relation-fixed-camera-frame/1.0",
        ),
        (
            "relation_tolerance_policy_ref",
            "definition:spatialcf/upright-se2/relation-tolerance/1.0",
        ),
        (
            "target_relation_selector_ref",
            "definition:spatialcf/upright-se2/relation-target-selector/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/relation-source-exactness/1.0",
        ),
    ),
    "PRESERVATION": (
        (
            "after_state_selector_ref",
            "definition:spatialcf/upright-se2/preservation-after-state/1.0",
        ),
        (
            "grounded_operand_policy_ref",
            "definition:spatialcf/upright-se2/preservation-grounded-operands/1.0",
        ),
        (
            "before_state_selector_ref",
            "definition:spatialcf/upright-se2/preservation-before-state/1.0",
        ),
        (
            "complete_state_policy_ref",
            "definition:spatialcf/upright-se2/complete-state-delta/1.0",
        ),
        (
            "frozen_leaf_policy_ref",
            "definition:spatialcf/upright-se2/preservation-frozen-leaves/1.0",
        ),
        (
            "preservation_evaluator_ref",
            "definition:spatialcf/upright-se2/preservation-evaluator/1.0",
        ),
        (
            "transition_comparator_ref",
            "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0",
        ),
    ),
    "VISIBILITY": (
        (
            "camera_selector_ref",
            "definition:spatialcf/upright-se2/visibility-fixed-camera-selector/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/visibility-evaluation-input/1.0",
        ),
        (
            "fixed_camera_policy_ref",
            "definition:spatialcf/upright-se2/visibility-fixed-camera-policy/1.0",
        ),
        (
            "metric_definition_ref",
            "definition:spatialcf/upright-se2/visibility-metric/1.0",
        ),
        (
            "occluder_roster_completeness_ref",
            "definition:spatialcf/upright-se2/visibility-complete-occluder-roster/1.0",
        ),
        (
            "threshold_comparator_ref",
            "definition:spatialcf/upright-se2/visibility-threshold-comparator/1.0",
        ),
        (
            "threshold_definition_ref",
            "definition:spatialcf/upright-se2/visibility-threshold/1.0",
        ),
        (
            "visibility_boundary_policy_ref",
            "definition:spatialcf/upright-se2/visibility-closed-boundary/1.0",
        ),
        (
            "visibility_evaluator_ref",
            "definition:spatialcf/upright-se2/visibility-evaluator/1.0",
        ),
    ),
}


_UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-objective/2.0"
)


_UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-term/2.0"
)


_UPRIGHT_SE2_RETAINED_POINT_TERM_ROSTER_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-term-roster/2.0"
)


_UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/exact-rational/1.0"
)


_M2_Q0_SHARED_VALUE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0"
)


_M2_Q0_MAPPING_DEFINITION_ROSTER_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/mapping-definition-roster/3.0"
)


_M2_Q0_SOURCE_PROVENANCE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/source-provenance/3.0"
)


_M2_Q0_SUPPORTED_DOMAIN_REF = (
    "definition:spatialcf/upright-se2/m2-closed-axis-aligned-rect-domain/1.0"
)


_M2_Q0_DOMAIN_SCHEMA_REF = "schema:spatialcf/upright-se2/m2-q0/translation-domain/1.0"


_M2_Q0_DOMAIN_TRANSFORM_REF = (
    "definition:spatialcf/upright-se2/m2-q0/delta-xy-identity/1.0"
)
