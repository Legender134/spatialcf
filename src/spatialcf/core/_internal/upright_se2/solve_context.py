"""Upright compiler solve context; explicit pure implementation owner."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    FixedCardinalCellPolicyV3,
    FixedCardinalObjectiveTermV3,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.compatibility import (
    PlanarTranslateCompilation,
)

from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
)

from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    CanonicalIdValue,
    DefinitionBundle,
    DigestValue,
    EnumSymbolValue,
    FiniteRealValue,
    IntegerValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.outcomes import (
    BackendSelectionRecord,
    CapabilityMismatch,
    ResourceUsage,
    TypedCompilationOutcome,
    _ResourceUsageEntry,
)

from spatialcf.domain.profiles import (
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationOwnerBinding,
    ImplementationRegistrySnapshot,
    ProofPolicy,
    ResourceLimit,
    ResourcePolicy,
    SolverBackendDescriptor,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _bridge_fraction,
    _sorted_bytes,
)

from spatialcf.core._internal.upright_se2.constants import (
    _BACKEND_BUILD_SHA256,
    _BACKEND_REF,
    _CHECKER_BUILD_SHA256,
    _CONTINUOUS_BACKEND_REF,
    _CONTINUOUS_UNAVAILABLE_REF,
    _DEFINITION_KIND_REF,
    _DEPENDENCY_LOCK_SHA256,
    _DIGEST_SCHEMA_REF,
)


def _bridge_policy_fields(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
    policy_key: str,
) -> dict[str, TypedValue]:
    policy = bundle.policy_for(policy_key)
    payload = policy.payload
    if type(payload.payload) is not RecordValue:
        raise ValueError("bridge policy payload must be an exact typed record")
    fields = {field.name: field.value for field in payload.payload.fields}
    if len(fields) != len(payload.payload.fields):
        raise ValueError("bridge policy payload has duplicate fields")
    return fields


def _bridge_policy_real(fields: dict[str, TypedValue], name: str) -> Fraction:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not FiniteRealValue:
        raise ValueError(f"bridge policy field {name!r} must be a finite real")
    return _bridge_fraction(value.payload.value, label=f"policy field {name}")


def _bridge_policy_integer(fields: dict[str, TypedValue], name: str) -> int:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not IntegerValue:
        raise ValueError(f"bridge policy field {name!r} must be an integer")
    return value.payload.value


def _bridge_policy_symbol(fields: dict[str, TypedValue], name: str) -> str:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not EnumSymbolValue:
        raise ValueError(f"bridge policy field {name!r} must be a symbol")
    return value.payload.symbol


def _bridge_policy_id(fields: dict[str, TypedValue], name: str) -> str:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not CanonicalIdValue:
        raise ValueError(f"bridge policy field {name!r} must be an identifier")
    return value.payload.value


def _bridge_resource_cap(bundle: upright.UprightSE2ExecutablePolicyBundle) -> int:
    fields = _bridge_policy_fields(bundle, "resource")
    cap = _bridge_policy_integer(fields, "atomic_step_limit")
    if cap <= 0:
        raise ValueError("cardinal bridge resource cap must be positive")
    if _bridge_policy_symbol(fields, "deterministic_order") != "LOWER_OWNED_XY":
        raise ValueError("cardinal bridge resource policy order is unsupported")
    return cap


def _bridge_cell_policy(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
    *,
    relation: str,
    cell: ClosedXYCellV3,
    resource_cap: int,
) -> FixedCardinalCellPolicyV3:
    collision = _bridge_policy_fields(bundle, "collision")
    support = _bridge_policy_fields(bundle, "support")
    relation_policy = bundle.policy_for(f"relation:{relation}")
    relation_fields = _bridge_policy_fields(bundle, f"relation:{relation}")
    visibility = _bridge_policy_fields(bundle, "visibility")
    safety = _bridge_policy_fields(bundle, "safety")
    objective = upright.decode_upright_se2_objective_policy(bundle)
    if (
        _bridge_policy_symbol(collision, "contact_comparator") != "ALLOW_EQUALITY"
        or _bridge_policy_symbol(collision, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(support, "containment_comparator") != "CONTAINS"
        or _bridge_policy_symbol(support, "containment_boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(relation_fields, "relation_symbol") != relation
        or _bridge_policy_symbol(relation_fields, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(visibility, "comparator") != "GEQ"
        or _bridge_policy_symbol(visibility, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(safety, "safety_penalty_rule")
        != "FROM_CONSTRAINT_SLACK"
    ):
        raise ValueError("cardinal bridge policy symbols are unsupported")
    measurement = _bridge_policy_symbol(relation_fields, "measurement")
    if measurement == "EXTENT_SIGNED_AXIS_GAP":
        retained_measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
    elif measurement == "EXTENT_EUCLIDEAN_SEPARATION":
        retained_measurement = "EXTENT_AWARE_EUCLIDEAN_SEPARATION"
    else:
        raise ValueError("cardinal bridge relation measurement is unsupported")
    threshold = _bridge_policy_real(relation_fields, "threshold")
    tolerance = _bridge_policy_real(relation_fields, "tolerance")
    visibility_threshold = _bridge_policy_real(visibility, "threshold")
    visibility_tolerance = _bridge_policy_real(visibility, "tolerance")
    objective_terms = tuple(
        _bridge_objective_term(term, objective.aggregation_definition_ref)
        for term in objective.terms
    )
    return FixedCardinalCellPolicyV3(
        policy_id=relation_policy.definition_ref,
        policy_version="definition:1",
        relation_threshold=threshold,
        relation_tolerance=tolerance,
        relation_comparator=_bridge_policy_symbol(relation_fields, "comparator"),
        relation_boundary="CLOSED",
        # FROM_CONSTRAINT_SLACK is the fixed unit-scale semantic rule; the
        # request-bound S-term weight remains in the common objective roster.
        safety_penalty_scale=Fraction(1),
        safety_constraint_slack_target=_bridge_policy_real(
            safety,
            "constraint_slack_target",
        ),
        safety_rule="PENALIZE_BELOW_TARGET",
        collision_clearance=_bridge_policy_real(collision, "clearance_m"),
        collision_contact_comparator="GE",
        collision_boundary="CLOSED",
        support_accepted_contact_gap=(
            _bridge_policy_real(support, "contact_gap_lower_m"),
            _bridge_policy_real(support, "contact_gap_upper_m"),
        ),
        support_stability_margin=_bridge_policy_real(
            support,
            "stability_margin_m",
        ),
        support_containment_comparator="GE",
        support_boundary="CLOSED",
        support_frame="WORLD_XY_Z_UP",
        support_normal=(Fraction(), Fraction(), Fraction(1)),
        relation_definition_id="spatial-relation:fixed-cardinal",
        relation_definition_version="definition:1",
        relation_symbol=relation,
        relation_measurement=retained_measurement,
        relation_operand="SUBJECT_COMPOUND_TO_REFERENCE",
        relation_geometry="UPRIGHT_AABB_EXTENTS",
        visibility_cell=cell.canonical_bounds,
        visibility_bounds=(
            max(Fraction(), visibility_threshold - visibility_tolerance),
            min(Fraction(1), visibility_threshold + visibility_tolerance),
        ),
        objective_terms=objective_terms,
        atomic_step_limit=resource_cap,
    )


def _bridge_objective_term(
    term: upright.UprightSE2ObjectiveTermPolicy,
    aggregation: str,
) -> FixedCardinalObjectiveTermV3:
    metric_parts = term.metric_definition_ref.rsplit("/", 1)
    if len(metric_parts) != 2:
        raise ValueError("objective metric definition has no fixed version suffix")
    return FixedCardinalObjectiveTermV3(
        term_id=term.term_id,
        selector=term.input_selector_definition_ref,
        metric_definition_id=metric_parts[0],
        metric_definition_version=metric_parts[1],
        unit=term.unit_ref,
        weight=_bridge_fraction(term.weight, label=f"objective {term.term_id} weight"),
        normalizer=_bridge_fraction(
            term.normalizer,
            label=f"objective {term.term_id} normalizer",
        ),
        aggregation=aggregation,
    )


def _continuous_capability_mismatch(
    solve_request: CounterfactualSolveRequest,
) -> TypedCompilationOutcome:
    descriptor = solve_request.backend_descriptor_bundle.backend_descriptors[0]
    mismatch = CapabilityMismatch(
        backend_ref=descriptor.backend_ref,
        backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        missing_capability_refs=upright.UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS,
        reason_claim_definition_ref=_CONTINUOUS_UNAVAILABLE_REF,
    )
    selection = BackendSelectionRecord.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        implementation_registry_snapshot_sha256=(
            solve_request.implementation_registry_snapshot.implementation_registry_snapshot_sha256
        ),
        backend_descriptor_bundle_sha256=(
            solve_request.backend_descriptor_bundle.backend_descriptor_bundle_sha256
        ),
        backend_routing_policy_sha256=(
            solve_request.backend_routing_policy.backend_routing_policy_sha256
        ),
        ordered_candidate_backend_refs=(descriptor.backend_ref,),
        capability_rows=(mismatch,),
        selection_disposition="NO_SELECTION",
        selection_disposition_claim_ref=_CONTINUOUS_UNAVAILABLE_REF,
        deterministic_selection_reason_ref=_CONTINUOUS_UNAVAILABLE_REF,
    )
    resource_usage = _zero_resource_usage()
    return TypedCompilationOutcome.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        selected_backend_ref=descriptor.backend_ref,
        selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        compilation_reason_claim_definition_ref=_CONTINUOUS_UNAVAILABLE_REF,
        partial_artifact_refs=(),
        resource_usage=resource_usage,
    )


def _zero_resource_usage() -> ResourceUsage:
    return ResourceUsage(
        accounting_claim_definition_ref=(
            "definition:spatialcf/upright-se2/resource-accounting/1.0"
        ),
        entries=(
            _ResourceUsageEntry(
                resource_definition_ref="definition:spatialcf/upright-se2/resource/1.0",
                used=0.0,
            ),
        ),
        exhausted=False,
    )


def _source_requested_gap(
    source_compilation: PlanarTranslateCompilation,
) -> upright.UprightSE2ExactRational:
    """Lift the retained source binary64 gap exactly into the M3 request."""

    source_gap = source_compilation.source_artifacts.config.target_optimality_gap
    numerator, denominator = source_gap.as_integer_ratio()
    return upright.UprightSE2ExactRational(numerator=numerator, denominator=denominator)


def _solve_policy_bundle(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    requested_gap: upright.UprightSE2ExactRational,
) -> DefinitionBundle:
    return upright.build_upright_se2_solve_policy_definition_bundle(
        registration=registration,
        requested_gap=requested_gap,
        objective_bound_policy_ref=_solver_config().objective_bound_policy_ref,
    )


def _closure_definition_bundle(
    *,
    definition_ref: str,
    payload_schema_ref: str,
    registration: upright.UprightSE2ProfileRegistration,
) -> DefinitionBundle:
    return DefinitionBundle.seal(
        definitions=(
            CanonicalDefinitionEnvelope.seal(
                definition_ref=definition_ref,
                definition_kind_ref=_DEFINITION_KIND_REF,
                payload_schema_ref=payload_schema_ref,
                payload=TypedValue(
                    value_schema_ref=payload_schema_ref,
                    payload=RecordValue(
                        fields=(
                            NamedTypedValue(
                                name="profile_registration_sha256",
                                value=TypedValue(
                                    value_schema_ref=_DIGEST_SCHEMA_REF,
                                    payload=DigestValue(
                                        value=registration.profile_registration_sha256
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
            ),
        )
    )


def _implementation_registry(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    continuous: bool = False,
) -> ImplementationRegistrySnapshot:
    _ = registration
    semantic_definition_bindings = tuple(
        ImplementationOwnerBinding(
            definition_or_capability_ref=definition_ref,
            implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
        )
        for definition_ref in (
            *upright.UPRIGHT_SE2_PREDICATE_DEFINITION_REFS,
            upright.UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
            upright.UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF,
        )
    )
    return ImplementationRegistrySnapshot.seal(
        definition_and_capability_owner_bindings=_sorted_bytes(
            *semantic_definition_bindings,
            ImplementationOwnerBinding(
                definition_or_capability_ref=upright.UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
                implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
            *(
                (
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
                    ),
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
                    ),
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
                    ),
                )
                if continuous
                else ()
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
        ),
        implementation_build_hashes=_sorted_bytes(
            (
                upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
                upright.UPRIGHT_SE2_COMPILER_BUILD_SHA256,
            ),
            (upright.UPRIGHT_SE2_BACKEND_OWNER_REF, _BACKEND_BUILD_SHA256),
            (upright.UPRIGHT_SE2_CHECKER_OWNER_REF, _CHECKER_BUILD_SHA256),
        ),
        dependency_lock_sha256=_DEPENDENCY_LOCK_SHA256,
    )


def _backend_descriptor_bundle(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    continuous: bool = False,
) -> BackendDescriptorBundle:
    descriptor = SolverBackendDescriptor.seal(
        backend_ref=_CONTINUOUS_BACKEND_REF if continuous else _BACKEND_REF,
        implementation_build_sha256=_BACKEND_BUILD_SHA256,
        supported_profile_hashes=(
            registration.action_space_profile.action_space_profile_sha256,
        ),
        supported_predicate_capabilities=(
            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        ),
        supported_operator_capabilities=(
            upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF
            if continuous
            else upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
        ),
        supported_objective_capabilities=(
            upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
        ),
        supported_numeric_semantics=(
            registration.semantics_profile.numeric_semantics_ref,
        ),
        emitted_proof_material_definition_refs=(
            upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
            if continuous
            else upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF,
        ),
        compatible_checker_capability_refs=_sorted_bytes(
            (
                upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                if continuous
                else upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            ),
            upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
            upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        ),
        resource_definition_refs=("definition:spatialcf/upright-se2/resource/1.0",),
    )
    return BackendDescriptorBundle.seal(
        backend_descriptors=(descriptor,),
        unavailable_optional_backends=(),
    )


def _solver_config() -> CounterfactualSolverConfig:
    return CounterfactualSolverConfig.seal(
        solver_config_ref="definition:spatialcf/upright-se2/solver-config/1.0",
        compilation_policy_ref="definition:spatialcf/upright-se2/compile/1.0",
        proposal_policy_ref="definition:spatialcf/upright-se2/proposal/1.0",
        objective_bound_policy_ref="definition:spatialcf/upright-se2/objective-bound/1.0",
        determinism_policy_ref="definition:spatialcf/upright-se2/determinism/1.0",
    )


def _proof_policy(*, continuous: bool = False) -> ProofPolicy:
    return ProofPolicy.seal(
        proof_policy_ref="definition:spatialcf/upright-se2/proof-policy/1.0",
        accepted_claim_definition_refs=_sorted_bytes(
            upright.UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF,
            upright.UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
        ),
        required_checker_capability_refs=(
            (
                upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                if continuous
                else upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            ),
        ),
        publication_minimum_claim_ref=(
            "definition:spatialcf/upright-se2/claim-certified-solution/1.0"
        ),
        permit_noncertified_terminal_records=True,
    )


def _resource_policy() -> ResourcePolicy:
    return ResourcePolicy.seal(
        resource_policy_ref="definition:spatialcf/upright-se2/resource-policy/1.0",
        limits=(
            ResourceLimit(
                definition_ref="definition:spatialcf/upright-se2/resource/1.0",
                finite_limit=1.0,
            ),
        ),
        exhaustion_claim_ref="definition:spatialcf/upright-se2/resource-exhausted/1.0",
        shared_ledger_policy_ref="definition:spatialcf/upright-se2/shared-ledger/1.0",
    )


def _validate_request_resource_policy(resource_policy: ResourcePolicy) -> None:
    """Keep resource ownership fixed while leaving its request cap executable."""

    registered = _resource_policy()
    if (
        resource_policy.resource_policy_ref != registered.resource_policy_ref
        or resource_policy.exhaustion_claim_ref != registered.exhaustion_claim_ref
        or resource_policy.shared_ledger_policy_ref
        != registered.shared_ledger_policy_ref
        or tuple(limit.definition_ref for limit in resource_policy.limits)
        != tuple(limit.definition_ref for limit in registered.limits)
    ):
        raise ValueError("resource policy does not match the upright se2 closure")


def _backend_routing_policy() -> BackendRoutingPolicy:
    return BackendRoutingPolicy.seal(
        routing_policy_ref="definition:spatialcf/upright-se2/routing/1.0",
        capability_filter_definition_ref="definition:spatialcf/upright-se2/filter/1.0",
        deterministic_order_definition_ref="definition:spatialcf/upright-se2/order/1.0",
        portfolio_composition_definition_ref="definition:spatialcf/upright-se2/portfolio/1.0",
        stop_condition_definition_ref="definition:spatialcf/upright-se2/stop/1.0",
        resource_partition_definition_ref="definition:spatialcf/upright-se2/partition/1.0",
    )


# Preserve supported public type/function and pickle lookup.
_bridge_policy_fields.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_policy_real.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_policy_integer.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_policy_symbol.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_policy_id.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_resource_cap.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_cell_policy.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_objective_term.__module__ = "spatialcf.core.upright_se2_compiler"
_continuous_capability_mismatch.__module__ = "spatialcf.core.upright_se2_compiler"
_zero_resource_usage.__module__ = "spatialcf.core.upright_se2_compiler"
_source_requested_gap.__module__ = "spatialcf.core.upright_se2_compiler"
_solve_policy_bundle.__module__ = "spatialcf.core.upright_se2_compiler"
_closure_definition_bundle.__module__ = "spatialcf.core.upright_se2_compiler"
_implementation_registry.__module__ = "spatialcf.core.upright_se2_compiler"
_backend_descriptor_bundle.__module__ = "spatialcf.core.upright_se2_compiler"
_solver_config.__module__ = "spatialcf.core.upright_se2_compiler"
_proof_policy.__module__ = "spatialcf.core.upright_se2_compiler"
_resource_policy.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_request_resource_policy.__module__ = "spatialcf.core.upright_se2_compiler"
_backend_routing_policy.__module__ = "spatialcf.core.upright_se2_compiler"
