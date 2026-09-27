"""Static registry routing validation and explicit dependencies."""

from __future__ import annotations

from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
)

from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    canonical_json_bytes,
)

from spatialcf.domain.operators import (
    OperatorDefinition,
)

from spatialcf.domain.outcomes import (
    CapabilityMatch,
    CapabilityMismatch,
)

from spatialcf.domain.predicates import (
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendRoutingPolicy,
    ProofPolicy,
    SemanticsProfile,
    SolverBackendDescriptor,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    SemanticContractError,
    _CLAIM_PROOF_MATERIAL_FIELD,
    _MISMATCH_ORDER,
    _ROLE_CERTIFIED_SOLUTION,
    _ROLE_PROOF_MATERIAL,
    _ROLE_PROVEN_UNSAT,
    _ROLE_ROUTING_MATCH,
    _ROLE_ROUTING_MISMATCH,
    _ROLE_ROUTING_POLICY,
    _ROUTING_MATCH_CLAIM_FIELD,
    _ROUTING_MISMATCH_CLAIM_FIELD,
    _ref_key,
    _self_digest_matches,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_record_reference,
)


def _selection_missing_capability(reason: str, value: str) -> str:
    """Encode one deterministic unsupported requirement for selection replay."""

    digest = canonical_sha256(
        (reason, value),
        domain="spatialcf/counterfactual/backend-capability-mismatch/3.0",
    )
    return f"capability:spatialcf/counterfactual/mismatch/{reason}/{digest}"


def _selection_routing_claim_refs(
    definitions: dict[str, CanonicalDefinitionEnvelope],
    roles: dict[str, str],
    routing_policy: BackendRoutingPolicy,
) -> tuple[str, str]:
    """Resolve typed routing claims while replaying a sealed selection."""

    try:
        routing_definition = definitions[routing_policy.routing_policy_ref]
        match_ref = _definition_record_reference(
            routing_definition,
            _ROUTING_MATCH_CLAIM_FIELD,
        )
        mismatch_ref = _definition_record_reference(
            routing_definition,
            _ROUTING_MISMATCH_CLAIM_FIELD,
        )
    except (DefinitionClosureError, KeyError) as error:
        raise SemanticContractError("routing policy semantics") from error
    if (
        roles.get(routing_policy.routing_policy_ref) != _ROLE_ROUTING_POLICY
        or roles.get(match_ref) != _ROLE_ROUTING_MATCH
        or roles.get(mismatch_ref) != _ROLE_ROUTING_MISMATCH
    ):
        raise SemanticContractError("routing policy semantics")
    return match_ref, mismatch_ref


def _selection_proof_material_refs(
    definitions: dict[str, CanonicalDefinitionEnvelope],
    roles: dict[str, str],
    proof_policy: ProofPolicy,
) -> set[str]:
    """Resolve only proof material bound to the policy's accepted claims."""

    proof_material_refs: set[str] = set()
    for claim_ref in proof_policy.accepted_claim_definition_refs:
        try:
            claim_definition = definitions[claim_ref]
            proof_material_ref = _definition_record_reference(
                claim_definition,
                _CLAIM_PROOF_MATERIAL_FIELD,
            )
        except (DefinitionClosureError, KeyError) as error:
            raise SemanticContractError("proof policy claim metadata") from error
        if (
            roles.get(claim_ref) not in {_ROLE_CERTIFIED_SOLUTION, _ROLE_PROVEN_UNSAT}
            or roles.get(proof_material_ref) != _ROLE_PROOF_MATERIAL
        ):
            raise SemanticContractError("proof policy claim metadata")
        proof_material_refs.add(proof_material_ref)
    return proof_material_refs


def _selection_backend_requirements_for_descriptor_build(
    request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    descriptor: SolverBackendDescriptor,
) -> tuple[set[str], set[str], set[str]]:
    """Partition requirements into this build, another build, and unresolved rows."""

    snapshot = request.implementation_registry_snapshot
    owner_by_capability = {
        binding.definition_or_capability_ref: binding.implementation_owner_ref
        for binding in snapshot.definition_and_capability_owner_bindings
        if binding.definition_or_capability_ref.startswith("capability:")
    }
    build_by_owner = dict(snapshot.implementation_build_hashes)
    matched_for_this_build: set[str] = set()
    other_valid_build: set[str] = set()
    structurally_unresolved: set[str] = set()
    for capability in action_space_profile.backend_capability_requirements:
        owner_ref = owner_by_capability.get(capability)
        if owner_ref is None:
            structurally_unresolved.add(capability)
            continue
        owner_build = build_by_owner.get(owner_ref)
        if owner_build is None:
            structurally_unresolved.add(capability)
        elif owner_build == descriptor.implementation_build_sha256:
            matched_for_this_build.add(capability)
        else:
            other_valid_build.add(capability)
    return (
        matched_for_this_build,
        other_valid_build,
        structurally_unresolved,
    )


def _reconstruct_backend_match(
    request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    semantics_profile: SemanticsProfile,
    predicate_definitions: tuple[PredicateDefinition, ...],
    operator_definitions: tuple[OperatorDefinition, ...],
    descriptor: SolverBackendDescriptor,
    definitions: dict[str, CanonicalDefinitionEnvelope],
    roles: dict[str, str],
) -> CapabilityMatch | CapabilityMismatch:
    """Recompute one submitted routing row without importing the backend owner.

    ``spatialcf.core.backends`` remains the public matcher and protocol owner.
    The registry duplicates this small, pure comparison only to reconstruct the
    candidate rows it must verify in an outcome DAG; it never delegates to, or
    imports, another core module.
    """

    match_claim, mismatch_claim = _selection_routing_claim_refs(
        definitions,
        roles,
        request.backend_routing_policy,
    )
    if not _self_digest_matches(descriptor):
        return CapabilityMismatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            missing_capability_refs=(
                _selection_missing_capability("descriptor", descriptor.backend_ref),
            ),
            reason_claim_definition_ref=mismatch_claim,
        )

    predicate_required = set(action_space_profile.predicate_capability_refs)
    for definition in predicate_definitions:
        predicate_required.add(definition.evaluator_capability_ref)
        predicate_required.add(definition.verifier_capability_ref)
    operator_required = {
        capability
        for definition in operator_definitions
        for capability in (
            definition.compiler_capability_ref,
            definition.verifier_capability_ref,
        )
    }
    missing_by_reason: dict[str, set[str]] = {}
    if action_space_profile.action_space_profile_sha256 not in set(
        descriptor.supported_profile_hashes
    ):
        missing_by_reason["profile"] = {
            _selection_missing_capability(
                "profile",
                action_space_profile.action_space_profile_sha256,
            )
        }
    predicate_missing = predicate_required - set(
        descriptor.supported_predicate_capabilities
    )
    if predicate_missing:
        missing_by_reason["predicate"] = predicate_missing
    operator_missing = operator_required - set(
        descriptor.supported_operator_capabilities
    )
    if operator_missing:
        missing_by_reason["operator"] = operator_missing
    objective_missing = set(action_space_profile.objective_capability_refs) - set(
        descriptor.supported_objective_capabilities
    )
    if objective_missing:
        missing_by_reason["objective"] = objective_missing
    numeric_missing = {
        semantics_profile.numeric_semantics_ref,
        request.semantic_problem.numeric_semantics_ref,
    } - set(descriptor.supported_numeric_semantics)
    if numeric_missing:
        missing_by_reason["numeric"] = {
            _selection_missing_capability("numeric", reference)
            for reference in numeric_missing
        }
    proof_missing = set(
        proof_policy_capabilities
        := request.proof_policy.required_checker_capability_refs
    ) - set(descriptor.compatible_checker_capability_refs)
    proof_material_missing = _selection_proof_material_refs(
        definitions,
        roles,
        request.proof_policy,
    ) - set(descriptor.emitted_proof_material_definition_refs)
    if proof_material_missing:
        proof_missing.update(
            _selection_missing_capability("proof-material", reference)
            for reference in proof_material_missing
        )
    if proof_missing:
        missing_by_reason["proof"] = proof_missing
    resource_missing = {
        limit.definition_ref for limit in request.resource_policy.limits
    } - set(descriptor.resource_definition_refs)
    if resource_missing:
        missing_by_reason["resource"] = {
            _selection_missing_capability("resource", reference)
            for reference in resource_missing
        }
    (
        matched_backend_requirements,
        _other_valid_build_requirements,
        structurally_unresolved_backend_requirements,
    ) = _selection_backend_requirements_for_descriptor_build(
        request,
        action_space_profile,
        descriptor,
    )
    if not matched_backend_requirements:
        missing_by_reason["backend"] = set(
            action_space_profile.backend_capability_requirements
        )
    elif structurally_unresolved_backend_requirements:
        missing_by_reason["backend"] = set(structurally_unresolved_backend_requirements)
    registered = next(
        (
            candidate
            for candidate in request.backend_descriptor_bundle.backend_descriptors
            if candidate.backend_ref == descriptor.backend_ref
        ),
        None,
    )
    if (
        registered is None
        or registered.backend_descriptor_sha256 != descriptor.backend_descriptor_sha256
        or canonical_json_bytes(registered) != canonical_json_bytes(descriptor)
    ):
        missing_by_reason.setdefault("backend", set()).add(
            _selection_missing_capability("backend", descriptor.backend_ref)
        )
    if missing_by_reason:
        return CapabilityMismatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            missing_capability_refs=tuple(
                sorted(
                    {
                        capability
                        for reason in _MISMATCH_ORDER
                        for capability in missing_by_reason.get(reason, set())
                    },
                    key=_ref_key,
                )
            ),
            reason_claim_definition_ref=mismatch_claim,
        )
    matched = (
        matched_backend_requirements
        | predicate_required
        | operator_required
        | set(action_space_profile.objective_capability_refs)
        | set(proof_policy_capabilities)
    )
    return CapabilityMatch(
        backend_ref=descriptor.backend_ref,
        backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        matched_capability_refs=tuple(sorted(matched, key=_ref_key)),
        match_claim_definition_ref=match_claim,
    )


def _reconstructed_backend_rows(
    request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    semantics_profile: SemanticsProfile,
    predicate_definitions: tuple[PredicateDefinition, ...],
    operator_definitions: tuple[OperatorDefinition, ...],
    definitions: dict[str, CanonicalDefinitionEnvelope],
    roles: dict[str, str],
) -> tuple[CapabilityMatch | CapabilityMismatch, ...]:
    """Return the exact candidate universe that an outcome record must carry."""

    rows = tuple(
        _reconstruct_backend_match(
            request,
            action_space_profile,
            semantics_profile,
            predicate_definitions,
            operator_definitions,
            descriptor,
            definitions,
            roles,
        )
        for descriptor in request.backend_descriptor_bundle.backend_descriptors
    )
    if len({row.backend_ref for row in rows}) != len(rows):
        raise SemanticContractError("ordered backend candidates")
    return tuple(sorted(rows, key=lambda row: canonical_json_bytes(row.backend_ref)))


# Keep public class, exception and bound-method lookup stable.
_selection_missing_capability.__module__ = "spatialcf.core.registry"
_selection_routing_claim_refs.__module__ = "spatialcf.core.registry"
_selection_proof_material_refs.__module__ = "spatialcf.core.registry"
_selection_backend_requirements_for_descriptor_build.__module__ = "spatialcf.core.registry"
_reconstruct_backend_match.__module__ = "spatialcf.core.registry"
_reconstructed_backend_rows.__module__ = "spatialcf.core.registry"
