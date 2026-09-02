"""Pure compilation of the retained planar-translate v2 compatibility route.

This module creates an M1/v3 compatibility envelope.  It deliberately does not
solve, verify, import adapters, or reinterpret the retained v2 proof material.
"""

from __future__ import annotations

import warnings

from spatialcf.domain.artifacts import StrictConvexCandidateCompilerConfigV2_7
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.compatibility import (
    PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
    PlanarTranslateAlgorithmKernelBinding,
    PlanarTranslateCompatibilityLineage,
    PlanarTranslateCompilation,
    PlanarTranslateFieldMapping,
    PlanarTranslateProfileRegistration,
    PlanarTranslateV2SourceArtifacts,
    _derive_planar_translate_source_fact_bundle,
)
from spatialcf.domain.constraints import CanonicalConstraintSet
from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    SceneStateEnvelope,
)
from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    DefinitionBundle,
    DigestValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)
from spatialcf.domain.objective import ObjectiveSpecV2
from spatialcf.domain.operators import StateLeafIndex, StateVariableRef
from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    ObservationObligation,
    PreservationInvariant,
)
from spatialcf.domain.problem import CanonicalSceneV2_3, SemanticProblemV2_3
from spatialcf.domain.profiles import (
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationOwnerBinding,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    ObjectiveExpression,
    ObjectiveTerm,
    ProofPolicy,
    ResourceLimit,
    ResourcePolicy,
)
from spatialcf.domain.scene import CanonicalScene
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256
from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9

__all__ = ("compile_planar_translate_v2",)

_SOURCE_ENTITY_ID = "entity:spatialcf/planar-translate-source"
_SOURCE_DIGEST_SCHEMA = "schema:spatialcf/planar-translate/source-digest/2.0"
_SOURCE_BINDING_SCHEMA = "schema:spatialcf/planar-translate/source-binding/2.0"
_SOURCE_BINDING_KIND = (
    "definition:spatialcf/planar-translate/source-binding-kind/2.0"
)
_NEUTRAL_SCENE_SCHEMA = "schema:spatialcf/canonical-scene/2.0"
_NEUTRAL_STATE_SCHEMA = "schema:spatialcf/planar-translate/neutral-scene-state/2.0"
_STATE_VARIABLE_SCHEMA = "schema:spatialcf/planar-translate/state-world-xy/2.0"
_STATE_FAMILY = "definition:spatialcf/planar-translate/source-binding/2.0"
_STATE_FIELD_PATH = "field-path:spatialcf-planar-translate-world-xy"
_COMPATIBILITY_HASH_DOMAIN = "spatialcf/counterfactual/planar-translate/compiler/3.0"

_ALGORITHM_KERNEL_BINDINGS = (
    ("algorithm_id", "solver:canonical-branch-and-bound-v2"),
    ("algorithm_version", "algorithm:2.9"),
    ("candidate_config.algorithm_id", "solver:canonical-branch-and-bound-v2"),
    ("candidate_config.algorithm_version", "algorithm:2.7"),
    (
        "candidate_config.so2_kernel_id",
        "geometry-kernel:rational-so2-upright-box-directed-v2",
    ),
    (
        "candidate_config.so2_kernel_version",
        "kernel:2.2-continuous-yaw-upright-box",
    ),
    (
        "candidate_config.obstacle_kernel_id",
        "geometry-kernel:rational-convex-translation-bracket-v2",
    ),
    (
        "candidate_config.obstacle_kernel_version",
        "kernel:2.3-convex-translation-bracket",
    ),
    (
        "candidate_config.partition_kernel_id",
        "geometry-kernel:rational-convex-complement-partition-v2",
    ),
    (
        "candidate_config.partition_kernel_version",
        "kernel:2.4-topology-aware-convex-complement",
    ),
    (
        "candidate_config.intersection_kernel_id",
        "geometry-kernel:rational-strict-convex-intersection-v2",
    ),
    (
        "candidate_config.intersection_kernel_version",
        "kernel:2.5-strict-convex-intersection",
    ),
    (
        "candidate_config.support_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-support-projection-v2",
    ),
    (
        "candidate_config.support_projection_kernel_version",
        "kernel:2.6-exact-horizontal-support-projection",
    ),
    (
        "camera_frame_kernel_id",
        "geometry-kernel:rational-upright-world-to-camera-v2.9",
    ),
    (
        "target_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-directional-relation-v2.9",
    ),
    (
        "visibility_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-upright-camera-visibility-v2.9",
    ),
    (
        "objective_kernel_id",
        "objective-kernel:rational-continuous-yaw-joint-four-term-v2.9",
    ),
)


def compile_planar_translate_v2(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
    registration: PlanarTranslateProfileRegistration,
) -> PlanarTranslateCompilation:
    """Compile exact retained v2 roots into one hash-closed M1 embedding.

    The emitted roots declare only the compatibility mapping and routing
    envelope.  The contained source artifacts remain the sole v2 semantic
    authority for a later single-owner delegation.
    """

    _require_exact_inputs(problem, config, registration)
    _require_strict_round_trip(problem, SemanticProblemV2_3, "problem")
    _require_strict_round_trip(config, ContinuousYawSolverConfigV2_9, "config")
    _require_strict_round_trip(
        registration,
        PlanarTranslateProfileRegistration,
        "registration",
    )
    resource_limit = _lossless_m1_resource_limit(
        config.candidate_config.max_domain_operations
    )

    source_artifacts = PlanarTranslateV2SourceArtifacts.seal(
        problem=problem,
        config=config,
    )
    field_mapping = _build_field_mapping()
    _validate_field_mapping(problem, config, field_mapping)
    if registration.mapping_definition_ref != field_mapping.mapping_definition_ref:
        raise ValueError("registration mapping reference does not match compiler mapping")
    if registration.mapping_definition_sha256 != field_mapping.mapping_definition_sha256:
        raise ValueError("registration mapping digest does not match compiler mapping")

    semantic_problem = _build_semantic_problem(
        problem,
        config,
        registration,
        source_artifacts,
        field_mapping,
    )
    solve_request = _build_solve_request(
        semantic_problem,
        registration,
        field_mapping,
        resource_limit,
    )
    lineage = PlanarTranslateCompatibilityLineage.seal(
        source_artifacts_sha256=source_artifacts.source_artifacts_sha256,
        source_problem_semantic_sha256=source_artifacts.problem_semantic_sha256,
        source_problem_canonical_sha256=source_artifacts.problem_canonical_sha256,
        source_config_domain_sha256=source_artifacts.config_domain_sha256,
        source_config_canonical_sha256=source_artifacts.config_canonical_sha256,
        v3_semantic_problem_sha256=semantic_problem.semantic_problem_sha256,
        v3_solve_request_sha256=solve_request.solve_request_sha256,
        semantics_profile_sha256=registration.semantics_profile.semantics_profile_sha256,
        action_space_profile_sha256=(
            registration.action_space_profile.action_space_profile_sha256
        ),
        mapping_definition_sha256=field_mapping.mapping_definition_sha256,
    )
    _require_distinct_v3_hashes(source_artifacts, semantic_problem, solve_request)
    return PlanarTranslateCompilation.seal(
        source_artifacts=source_artifacts,
        field_mapping=field_mapping,
        registration=registration,
        semantic_problem=semantic_problem,
        solve_request=solve_request,
        lineage=lineage,
    )


def _require_exact_inputs(
    problem: object,
    config: object,
    registration: object,
) -> None:
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(config) is not ContinuousYawSolverConfigV2_9:
        raise TypeError("config must be an exact ContinuousYawSolverConfigV2_9")
    if type(registration) is not PlanarTranslateProfileRegistration:
        raise TypeError("registration must be an exact PlanarTranslateProfileRegistration")


def _require_strict_round_trip(value: object, model_type: type, label: str) -> None:
    """Reject model-construct/model-copy values that bypass their sealed wire."""

    assert type(value) is model_type
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checked = model_type.model_validate(
                value.model_dump(mode="python", round_trip=True),
                strict=True,
            )
    except (TypeError, ValueError, Warning) as error:
        raise ValueError(f"{label} must pass strict canonical validation") from error
    if type(checked) is not model_type or canonical_json_bytes(checked) != canonical_json_bytes(value):
        raise ValueError(f"{label} strict canonical bytes do not round trip")


def _lossless_m1_resource_limit(value: int) -> float:
    try:
        finite_limit = float(value)
    except OverflowError as error:
        raise ValueError(
            "max_domain_operations must be losslessly representable as an M1 finite float"
        ) from error
    if int(finite_limit) != value:
        raise ValueError(
            "max_domain_operations must be losslessly representable as an M1 finite float"
        )
    return finite_limit


def _build_field_mapping() -> PlanarTranslateFieldMapping:
    return PlanarTranslateFieldMapping.seal(
        mapping_definition_ref=PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
        problem_field_names=tuple(SemanticProblemV2_3.model_fields),
        scene_field_names=tuple(CanonicalSceneV2_3.model_fields),
        constraint_field_names=tuple(CanonicalConstraintSet.model_fields),
        objective_field_names=tuple(ObjectiveSpecV2.model_fields),
        numeric_policy_field_names=tuple(NumericPolicyV2.model_fields),
        solver_config_field_names=tuple(ContinuousYawSolverConfigV2_9.model_fields),
        candidate_config_field_names=tuple(
            StrictConvexCandidateCompilerConfigV2_7.model_fields
        ),
        algorithm_kernel_bindings=tuple(
            PlanarTranslateAlgorithmKernelBinding(
                field_name=field_name,
                registered_value=registered_value,
            )
            for field_name, registered_value in _ALGORITHM_KERNEL_BINDINGS
        ),
    )


def _validate_field_mapping(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
    mapping: PlanarTranslateFieldMapping,
) -> None:
    observed_fields = (
        ("problem", tuple(type(problem).model_fields), mapping.problem_field_names),
        ("scene", tuple(type(problem.scene).model_fields), mapping.scene_field_names),
        (
            "constraints",
            tuple(type(problem.constraints).model_fields),
            mapping.constraint_field_names,
        ),
        ("objective", tuple(type(problem.objective).model_fields), mapping.objective_field_names),
        (
            "numeric policy",
            tuple(type(problem.numeric_policy).model_fields),
            mapping.numeric_policy_field_names,
        ),
        (
            "solver config",
            tuple(type(config).model_fields),
            mapping.solver_config_field_names,
        ),
        (
            "candidate config",
            tuple(type(config.candidate_config).model_fields),
            mapping.candidate_config_field_names,
        ),
    )
    for label, observed, expected in observed_fields:
        if observed != expected:
            raise ValueError(f"{label} field coverage does not match the source")

    bindings = (
        ("algorithm_id", config.algorithm_id),
        ("algorithm_version", config.algorithm_version),
        ("candidate_config.algorithm_id", config.candidate_config.algorithm_id),
        ("candidate_config.algorithm_version", config.candidate_config.algorithm_version),
        ("candidate_config.so2_kernel_id", config.candidate_config.so2_kernel_id),
        (
            "candidate_config.so2_kernel_version",
            config.candidate_config.so2_kernel_version,
        ),
        (
            "candidate_config.obstacle_kernel_id",
            config.candidate_config.obstacle_kernel_id,
        ),
        (
            "candidate_config.obstacle_kernel_version",
            config.candidate_config.obstacle_kernel_version,
        ),
        (
            "candidate_config.partition_kernel_id",
            config.candidate_config.partition_kernel_id,
        ),
        (
            "candidate_config.partition_kernel_version",
            config.candidate_config.partition_kernel_version,
        ),
        (
            "candidate_config.intersection_kernel_id",
            config.candidate_config.intersection_kernel_id,
        ),
        (
            "candidate_config.intersection_kernel_version",
            config.candidate_config.intersection_kernel_version,
        ),
        (
            "candidate_config.support_projection_kernel_id",
            config.candidate_config.support_projection_kernel_id,
        ),
        (
            "candidate_config.support_projection_kernel_version",
            config.candidate_config.support_projection_kernel_version,
        ),
        ("camera_frame_kernel_id", config.camera_frame_kernel_id),
        ("target_projection_kernel_id", config.target_projection_kernel_id),
        ("visibility_projection_kernel_id", config.visibility_projection_kernel_id),
        ("objective_kernel_id", config.objective_kernel_id),
    )
    declared = tuple(
        (binding.field_name, binding.registered_value)
        for binding in mapping.algorithm_kernel_bindings
    )
    if bindings != declared:
        raise ValueError("source algorithm and kernel identifiers do not match mapping")


def _build_semantic_problem(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
    registration: PlanarTranslateProfileRegistration,
    source_artifacts: PlanarTranslateV2SourceArtifacts,
    field_mapping: PlanarTranslateFieldMapping,
) -> CounterfactualProblemIR:
    state_leaf = _state_leaf()
    extension_bundle = _derive_planar_translate_source_fact_bundle(
        source_artifacts=source_artifacts,
        field_mapping=field_mapping,
        registration=registration,
    )
    scene_state = SceneStateEnvelope.seal(
        base_scene_schema_ref=_NEUTRAL_SCENE_SCHEMA,
        base_scene_payload=_neutral_scene(source_artifacts.source_artifacts_sha256),
        extension_fact_bundles=(extension_bundle,),
        closed_entity_index=(_SOURCE_ENTITY_ID,),
        canonical_state_leaf_index=StateLeafIndex.seal(leaves=(state_leaf,)),
    )
    definition_bundle = _source_definition_bundle(
        source_artifacts,
        registration,
        field_mapping,
    )
    authorization = InterventionAuthorization.seal(
        editable_entity_ids=(_SOURCE_ENTITY_ID,),
        allowed_operator_refs=registration.action_space_profile.allowed_operator_refs,
        authorized_primary_write_set=(state_leaf,),
        variable_bounds=(),
        maximum_program_steps=1,
        maximum_edited_entities=1,
        required_derived_rule_refs=(),
        complete_state_delta_policy_ref=(
            "definition:spatialcf/planar-translate/complete-delta/2.0"
        ),
    )
    objective_definition_ref = registration.semantics_profile.objective_definition_refs[0]
    objective = ObjectiveExpression.seal(
        aggregation_definition_ref=objective_definition_ref,
        terms=(
            ObjectiveTerm(
                term_id="objective-term:spatialcf/planar-translate/current-v2",
                objective_definition_ref=objective_definition_ref,
                input_selector_definition_ref=PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
                unit_ref="definition:spatialcf/planar-translate/unitless/2.0",
                normalization_definition_ref=(
                    "definition:spatialcf/planar-translate/normalization/2.0"
                ),
            ),
        ),
        deterministic_tie_break_definition_ref=(
            "definition:spatialcf/planar-translate/tie-break-current-v2/2.0"
        ),
    )
    return CounterfactualProblemIR.seal(
        problem_id=(
            "problem:spatialcf/planar-translate/v2-compatibility/"
            f"{source_artifacts.source_artifacts_sha256}"
        ),
        scene_state=scene_state,
        definition_bundle=definition_bundle,
        semantics_profile_ref=registration.semantics_profile.semantics_profile_ref,
        action_space_profile_ref=registration.action_space_profile.action_space_profile_ref,
        intervention_authorization=authorization,
        before_preconditions=(BeforePrecondition(formula=True),),
        after_goal=AfterGoal(formula=True),
        preservation_invariants=(
            PreservationInvariant(
                before_formula=True,
                after_formula=True,
                transition_comparator_ref=(
                    "definition:spatialcf/planar-translate/transition-translate-xy/2.0"
                ),
            ),
        ),
        explicit_observation_obligations=(
            ObservationObligation(
                phase="BEFORE",
                formula=True,
                evidence_policy_ref=(
                    "definition:spatialcf/planar-translate/observation-obligation-policy/2.0"
                ),
            ),
        ),
        objective_expression=objective,
        numeric_semantics_ref=registration.semantics_profile.numeric_semantics_ref,
    )


def _build_solve_request(
    semantic_problem: CounterfactualProblemIR,
    registration: PlanarTranslateProfileRegistration,
    field_mapping: PlanarTranslateFieldMapping,
    resource_limit: float,
) -> CounterfactualSolveRequest:
    backend_owner = registration.backend_owner_ref
    checker_owner = registration.checker_owner_ref
    owner_bindings = tuple(
        sorted(
            (
                ImplementationOwnerBinding(
                    definition_or_capability_ref=registration.compiler_capability_ref,
                    implementation_owner_ref=backend_owner,
                ),
                ImplementationOwnerBinding(
                    definition_or_capability_ref=registration.solver_capability_ref,
                    implementation_owner_ref=backend_owner,
                ),
                ImplementationOwnerBinding(
                    definition_or_capability_ref=registration.checker_capability_ref,
                    implementation_owner_ref=checker_owner,
                ),
            ),
            key=canonical_json_bytes,
        )
    )
    checker_build_hash = canonical_sha256(
        {
            "checker_owner_ref": checker_owner,
            "registration_sha256": registration.registration_sha256,
        },
        domain=f"{_COMPATIBILITY_HASH_DOMAIN}/checker-build",
    )
    owner_builds = tuple(
        sorted(
            (
                (backend_owner, registration.backend_descriptor.implementation_build_sha256),
                (checker_owner, checker_build_hash),
            ),
            key=canonical_json_bytes,
        )
    )
    registry_snapshot = ImplementationRegistrySnapshot.seal(
        definition_and_capability_owner_bindings=owner_bindings,
        implementation_build_hashes=owner_builds,
        dependency_lock_sha256=canonical_sha256(
            {
                "mapping_definition_sha256": field_mapping.mapping_definition_sha256,
                "registration_sha256": registration.registration_sha256,
            },
            domain=f"{_COMPATIBILITY_HASH_DOMAIN}/dependency-lock",
        ),
    )
    policy_bundle = _solve_policy_definition_bundle(field_mapping)
    return CounterfactualSolveRequest.seal(
        semantic_problem=semantic_problem,
        semantic_problem_sha256=semantic_problem.semantic_problem_sha256,
        solve_policy_definition_bundle=policy_bundle,
        implementation_registry_snapshot=registry_snapshot,
        backend_descriptor_bundle=BackendDescriptorBundle.seal(
            backend_descriptors=(registration.backend_descriptor,),
            unavailable_optional_backends=(),
        ),
        solver_config=CounterfactualSolverConfig.seal(
            solver_config_ref="definition:spatialcf/planar-translate/solver-config/2.0",
            compilation_policy_ref="definition:spatialcf/planar-translate/compile/2.0",
            proposal_policy_ref="definition:spatialcf/planar-translate/proposal/2.0",
            objective_bound_policy_ref=(
                "definition:spatialcf/planar-translate/objective-bound/2.0"
            ),
            determinism_policy_ref=(
                "definition:spatialcf/planar-translate/determinism/2.0"
            ),
        ),
        proof_policy=ProofPolicy.seal(
            proof_policy_ref="definition:spatialcf/planar-translate/proof-policy/2.0",
            accepted_claim_definition_refs=(
                registration.action_space_profile.allowed_claim_definition_refs
            ),
            required_checker_capability_refs=(registration.checker_capability_ref,),
            publication_minimum_claim_ref=(
                registration.action_space_profile.allowed_claim_definition_refs[0]
            ),
            permit_noncertified_terminal_records=True,
        ),
        resource_policy=ResourcePolicy.seal(
            resource_policy_ref="definition:spatialcf/planar-translate/resource-policy/2.0",
            limits=(
                ResourceLimit(
                    definition_ref=(
                        registration.backend_descriptor.resource_definition_refs[0]
                    ),
                    finite_limit=resource_limit,
                ),
            ),
            exhaustion_claim_ref=(
                "definition:spatialcf/planar-translate/resource-exhausted/2.0"
            ),
            shared_ledger_policy_ref=(
                "definition:spatialcf/planar-translate/shared-ledger/2.0"
            ),
        ),
        backend_routing_policy=BackendRoutingPolicy.seal(
            routing_policy_ref="definition:spatialcf/planar-translate/routing/2.0",
            capability_filter_definition_ref=(
                "definition:spatialcf/planar-translate/filter/2.0"
            ),
            deterministic_order_definition_ref=(
                "definition:spatialcf/planar-translate/order/2.0"
            ),
            portfolio_composition_definition_ref=(
                "definition:spatialcf/planar-translate/portfolio/2.0"
            ),
            stop_condition_definition_ref=(
                "definition:spatialcf/planar-translate/stop/2.0"
            ),
            resource_partition_definition_ref=(
                "definition:spatialcf/planar-translate/partition/2.0"
            ),
        ),
    )


def _source_definition_bundle(
    source_artifacts: PlanarTranslateV2SourceArtifacts,
    registration: PlanarTranslateProfileRegistration,
    field_mapping: PlanarTranslateFieldMapping,
) -> DefinitionBundle:
    fields = (
        ("action_space_profile_sha256", registration.action_space_profile.action_space_profile_sha256),
        ("mapping_definition_sha256", field_mapping.mapping_definition_sha256),
        ("semantics_profile_sha256", registration.semantics_profile.semantics_profile_sha256),
        ("source_artifacts_sha256", source_artifacts.source_artifacts_sha256),
    )
    record = RecordValue(
        fields=tuple(
            sorted(
                (
                    NamedTypedValue(name=name, value=_typed_digest(digest))
                    for name, digest in fields
                ),
                key=canonical_json_bytes,
            )
        )
    )
    definition = CanonicalDefinitionEnvelope.seal(
        definition_ref=PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
        definition_kind_ref=_SOURCE_BINDING_KIND,
        payload_schema_ref=_SOURCE_BINDING_SCHEMA,
        payload=TypedValue(
            value_schema_ref=_SOURCE_BINDING_SCHEMA,
            payload=record,
        ),
    )
    return DefinitionBundle.seal(definitions=(definition,))


def _solve_policy_definition_bundle(
    field_mapping: PlanarTranslateFieldMapping,
) -> DefinitionBundle:
    definition = CanonicalDefinitionEnvelope.seal(
        definition_ref="definition:spatialcf/planar-translate/solve-policy/2.0",
        definition_kind_ref=_SOURCE_BINDING_KIND,
        payload_schema_ref=_SOURCE_BINDING_SCHEMA,
        payload=TypedValue(
            value_schema_ref=_SOURCE_BINDING_SCHEMA,
            payload=RecordValue(
                fields=(
                    NamedTypedValue(
                        name="mapping_definition_sha256",
                        value=_typed_digest(field_mapping.mapping_definition_sha256),
                    ),
                )
            ),
        ),
    )
    return DefinitionBundle.seal(definitions=(definition,))


def _neutral_scene(source_artifacts_sha256: str) -> CanonicalScene:
    empty_facts = FactSetV2(
        availability=FactAvailabilityV2.KNOWN,
        values=(),
        completeness=FactCompletenessV2.EXACT,
        uncertainty=UncertaintyBudgetV2(),
    )
    return CanonicalScene(
        scene_id=(
            "scene:spatialcf/planar-translate-neutral/" f"{source_artifacts_sha256}"
        ),
        objects=empty_facts,
        geometry_instances=empty_facts,
        collision_bodies=empty_facts,
        workspace_boundaries=empty_facts,
        known_free_spaces=empty_facts,
        support_surfaces=empty_facts,
        cameras=empty_facts,
        baseline_observations=empty_facts,
    )


def _state_leaf() -> StateVariableRef:
    return StateVariableRef(
        state_variable_schema_ref=_STATE_VARIABLE_SCHEMA,
        state_schema_ref=_NEUTRAL_STATE_SCHEMA,
        fact_family_ref=_STATE_FAMILY,
        entity_or_fact_key=_SOURCE_ENTITY_ID,
        field_path_ref=_STATE_FIELD_PATH,
    )


def _typed_digest(digest: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SOURCE_DIGEST_SCHEMA,
        payload=DigestValue(value=digest),
    )


def _require_distinct_v3_hashes(
    source_artifacts: PlanarTranslateV2SourceArtifacts,
    semantic_problem: CounterfactualProblemIR,
    solve_request: CounterfactualSolveRequest,
) -> None:
    source_hashes = {
        source_artifacts.problem_semantic_sha256,
        source_artifacts.problem_canonical_sha256,
        source_artifacts.config_domain_sha256,
        source_artifacts.config_canonical_sha256,
    }
    if semantic_problem.semantic_problem_sha256 in source_hashes:
        raise ValueError("v3 semantic problem digest must differ from every v2 source digest")
    if solve_request.solve_request_sha256 in source_hashes:
        raise ValueError("v3 solve request digest must differ from every v2 source digest")
