"""Public pure upright compilation API with explicit implementation owners."""

from __future__ import annotations

import math

import warnings

from dataclasses import dataclass, fields, is_dataclass

from enum import StrEnum

from fractions import Fraction

from spatialcf.core._internal.kernels.projected_visibility import (
    FixedCardinalProjectionBoxV3,
    FixedCardinalVisibilityPolicyV3,
)

from spatialcf.core._internal.kernels.so2 import (
    ContinuousYawIntervalKindV4,
    SO2AtomicBudgetV2,
    compile_continuous_yaw_lift_v4,
)

from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    FixedCardinalBoxV3,
    FixedCardinalCellPolicyV3,
    FixedCardinalObjectiveTermV3,
    SupportSurfaceV3,
)

from spatialcf.core.problem import UprightCameraContextV2_9

from spatialcf.domain import upright_se2 as upright

from spatialcf.domain.base import (
    CanonicalModel,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    Quaternion,
    RigidTransformV2,
    UncertaintyBudgetV2,
    Vec2,
    Vec3,
)

from spatialcf.domain.compatibility import PlanarTranslateCompilation

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    EditProgram,
    ExtensionFact,
    ExtensionFactBundle,
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalDefinitionEnvelope,
    CanonicalIdValue,
    DefinitionBundle,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    IntegerValue,
    IntervalValue,
    NamedTypedValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueKind,
)

from spatialcf.domain.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
    StateDeltaManifest,
    StateLeafIndex,
    StateVariableRef,
    TypedVariableBound,
)

from spatialcf.domain.outcomes import (
    BackendSelectionRecord,
    CapabilityMismatch,
    ResourceUsage,
    TypedCompilationOutcome,
    _ResourceUsageEntry,
)

from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    GroundedObligation,
    GroundedObligationSet,
    ObservationObligation,
    PredicateAtom,
    PreservationInvariant,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationOwnerBinding,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    ProofPolicy,
    ResourceLimit,
    ResourcePolicy,
    SemanticsProfile,
    SolverBackendDescriptor,
)

from spatialcf.domain.scene import (
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
    CanonicalScene,
    ObjectSupportAssignment,
    PinholeCamera,
    SupportSurfaceFact,
)

from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

from spatialcf.core._internal.upright_se2.arithmetic import (
    _bridge_fraction,
    cardinal_inverse_quarter_turns,
    _rotate_cardinal_xy_components,
    rotate_cardinal_xy,
    _canonical_zero,
    _require_exact_round_trip,
    _dyadic_from_float,
    _dyadic_from_fraction,
    _sorted_bytes,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _CompilerInput,
    _SceneAuthority,
    _registered_profile,
    _validate_problem_and_extract_input,
    _compiler_input_from_scene,
    _input_entity_id,
    _validate_compiler_input_fact_identity,
    _real_field,
    _integer_field,
    _id_field,
    _symbol_field,
    _yaw_argument_field,
    _resolve_scene_authority,
    _validate_required_exact_scene_sources,
    _known_exact_source_values,
    _validate_complete_state_leaves,
    _validate_grounded_semantic_operands,
    _direct_predicate_atom,
    _validate_target_relation_operands,
    _validate_preservation_operands,
    _validate_visibility_operands,
    _require_reference_operand,
    _require_symbol_operand,
    _require_id_operand,
    _expected_state_leaves,
    _closed_entity_index,
    _fact_ids,
    _state_leaf,
    _validate_operational_closure,
    _validate_registry,
    _validate_intervention_authorization,
    _validate_explicit_pose_yaw,
    _resolve_pivot_binding,
    _validate_continuous_request,
    _translation_domain_from_authorization,
    _primary_write_set,
    _derived_write_set,
    _grounded_obligations,
)

from spatialcf.core._internal.upright_se2.compilation import (
    compile_upright_se2,
    _compile_cardinal,
    compile_upright_se2_continuous,
    _compile_continuous,
    _endpoint_construction_recipe,
    _state_footprint,
    _compiled_cell,
)

from spatialcf.core._internal.upright_se2.constants import (
    _INPUT_FAMILY_REF,
    _INPUT_SCHEMA_REF,
    _STATE_FAMILY_REF,
    _STATE_SCHEMA_REF,
    _SCENE_SCHEMA_REF,
    _BACKEND_REF,
    _CONTINUOUS_BACKEND_REF,
    _BACKEND_BUILD_SHA256,
    _CHECKER_BUILD_SHA256,
    _DEPENDENCY_LOCK_SHA256,
    _DEFINITION_CLOSURE_REF,
    _SOLVE_POLICY_REF,
    _DEFINITION_KIND_REF,
    _DERIVED_RULE_REF,
    _CONTINUOUS_UNAVAILABLE_REF,
    _COMPILER_INPUT_FACT_KEY,
    _PIVOT_STATE_HASH_DOMAIN,
    _POSE_STATE_HASH_DOMAIN,
    _UNCHANGED_LEAVES_HASH_DOMAIN,
    _RETAINED_OWNER_BOUND_SCHEMA_REF,
    _REAL_SCHEMA_REF,
    _INTEGER_SCHEMA_REF,
    _ID_SCHEMA_REF,
    _DIGEST_SCHEMA_REF,
    _ENUM_SCHEMA_REF,
    _YAW_ARGUMENT_SCHEMA_REF,
    _DEFINITION_CLOSURE_SCHEMA_REF,
    _SOLVE_POLICY_CLOSURE_SCHEMA_REF,
    _CARDINAL_TURN_FRACTIONS,
    _PRIMARY_ROLES,
    _DERIVED_ROLES,
    _SOURCE_FIELDS,
    _WORLD_XY_FRAME_REF,
    _METRE_UNIT_REF,
    _CLOSED_INTERVAL_TOPOLOGY_REF,
)

from spatialcf.core._internal.upright_se2.evaluation import (
    build_upright_se2_cardinal_evaluation_inputs,
    build_upright_se2_continuous_evaluation_inputs,
)

from spatialcf.core._internal.upright_se2.evaluation_data import (
    UprightSE2CardinalVisibilityEvaluationInput,
    UprightSE2CardinalEvaluationInputs,
    UprightSE2ContinuousVisibilityEvaluationInput,
    UprightSE2ContinuousEvaluationInputs,
)

from spatialcf.core._internal.upright_se2.evaluation_geometry import (
    _bridge_compiled_cell_member,
    _bridge_continuous_compiled_cell_member,
    _bridge_is_proper_exact_dyadic_continuous_descendant,
    _bridge_is_proper_exact_dyadic_descendant,
    _bridge_require_replayed_compilation,
    _bridge_require_replayed_continuous_compilation,
    _bridge_validate_cell_yaw,
    _bridge_object_map,
    _bridge_exact_identity_camera,
    _bridge_cardinal_yaw,
    _bridge_world_box_parts,
    _bridge_fixed_box,
    _bridge_collision_boxes,
    _bridge_subject_role_boxes,
    _bridge_xy_role_signature,
    _bridge_support_role_signature,
    _bridge_validate_subject_role_geometry_closure,
    _bridge_support_surface,
    _bridge_target_and_preservation_rows,
    _bridge_reference_relation_box,
    _bridge_object_pivot_xy,
)

from spatialcf.core._internal.upright_se2.m2 import (
    compile_planar_translate_m2_q0_equivalence,
    _canonical_scene_from_m2_source,
    _m2_yaw_transform_to_rigid,
    _rigid_from_yaw,
    _m2_supported_translation_domain,
    _m2_q0_policy_bundle,
    _m2_q0_policy_definition_ref,
    _m2_q0_policy_owner,
    _m2_q0_policy_payload,
    _m2_q0_real,
    _m2_q0_integer,
    _m2_q0_id,
    _m2_q0_digest,
    _m2_q0_symbol,
    _m2_q0_tuple,
    _m2_q0_record,
    _m2_q0_typed_record,
    _m2_q0_construction,
    _m2_q0_source_leaves,
    _m2_q0_domain_value,
    _m2_q0_int,
    _m2_q0_source_request,
    _m2_q0_scene_object,
    _m2_q0_variable_bounds,
    _m2_q0_closed_interval,
    _m2_q0_typed_reference,
    _m2_q0_target_relation_atom,
    _m2_q0_preservation_atom,
    _m2_q0_visibility_obligation,
    _reseal_q0_construction,
)

from spatialcf.core._internal.upright_se2.materialization import (
    materialize_upright_se2_endpoint,
    materialize_upright_se2_continuous_endpoint,
    _continuous_materialized_program_arguments,
    _continuous_after_state_template,
    _materialized_program,
    _after_state_template,
    _canonical_yaw_after,
    _apply_cardinal_quaternion,
    _derived_after_facts,
    _yaw_then_translate,
)

from spatialcf.core._internal.upright_se2.proof_values import (
    build_upright_se2_retained_owner_evaluation,
    _typed_bound_node,
    _typed_continuous_bound_node,
    _typed_bound_node_with_rational_mode,
    _canonical_proof_row,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _bridge_policy_fields,
    _bridge_policy_real,
    _bridge_policy_integer,
    _bridge_policy_symbol,
    _bridge_policy_id,
    _bridge_resource_cap,
    _bridge_cell_policy,
    _bridge_objective_term,
    _continuous_capability_mismatch,
    _zero_resource_usage,
    _source_requested_gap,
    _solve_policy_bundle,
    _closure_definition_bundle,
    _implementation_registry,
    _backend_descriptor_bundle,
    _solver_config,
    _proof_policy,
    _resource_policy,
    _validate_request_resource_policy,
    _backend_routing_policy,
)

from spatialcf.core._internal.upright_se2.visibility import (
    _bridge_visibility_bound_records,
    _bridge_projection_box,
    _bridge_camera_context,
    _bridge_visibility_inputs,
    _bridge_continuous_visibility_inputs,
)

__all__ = (
    "UprightSE2CardinalEvaluationInputs",
    "UprightSE2CardinalVisibilityEvaluationInput",
    "UprightSE2ContinuousEvaluationInputs",
    "UprightSE2ContinuousVisibilityEvaluationInput",
    "build_upright_se2_cardinal_evaluation_inputs",
    "build_upright_se2_continuous_evaluation_inputs",
    "build_upright_se2_retained_owner_evaluation",
    "cardinal_inverse_quarter_turns",
    "compile_planar_translate_m2_q0_equivalence",
    "compile_upright_se2",
    "compile_upright_se2_continuous",
    "materialize_upright_se2_continuous_endpoint",
    "materialize_upright_se2_endpoint",
    "rotate_cardinal_xy",
)
