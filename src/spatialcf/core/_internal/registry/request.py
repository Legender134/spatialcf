"""Static registry request validation and explicit dependencies."""

from __future__ import annotations

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    DefinitionBundle,
    ValueSchemaDefinition,
)

from spatialcf.domain.operators import (
    DerivedFactRuleDefinition,
    OperatorDefinition,
    StateVariableDefinition,
)

from spatialcf.domain.predicates import (
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationRegistrySnapshot,
    ProofPolicy,
    ResourcePolicy,
    SemanticsProfile,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    ImplementationResolutionError,
    SemanticContractError,
    _CLAIM_CHECKER_CAPABILITY_FIELD,
    _CLAIM_PROOF_MATERIAL_FIELD,
    _PROOF_CHECKER_CAPABILITY_FIELD,
    _RESOURCE_ACCOUNTING_CLAIM_FIELD,
    _ROLE_CERTIFIED_SOLUTION,
    _ROLE_PROOF_MATERIAL,
    _ROLE_PROVEN_UNSAT,
    _ROLE_RESOURCE_ACCOUNTING,
    _ROLE_RESOURCE_POLICY,
    _ROLE_ROUTING_MATCH,
    _ROLE_ROUTING_MISMATCH,
    _ROLE_ROUTING_POLICY,
    _ROLE_ROUTING_SELECTION_DISPOSITION,
    _ROLE_ROUTING_SELECTION_REASON,
    _ROLE_UNKNOWN,
    _ref_key,
    _self_digest_matches,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_record_reference,
    _definition_roles,
    _definition_root_maps,
    _references,
    _validate_exact_root_definition_closure,
)





class RequestValidation:
    """Stateless validation methods composed by StaticImplementationRegistry."""

    def validate_solve_request(
        self,
        semantic_definition_bundle: DefinitionBundle,
        solve_policy_definition_bundle: DefinitionBundle,
        value_schema_definitions: tuple[ValueSchemaDefinition, ...],
        predicate_definitions: tuple[PredicateDefinition, ...],
        state_variable_definitions: tuple[StateVariableDefinition, ...],
        derived_fact_rule_definitions: tuple[DerivedFactRuleDefinition, ...],
        operator_definitions: tuple[OperatorDefinition, ...],
        implementation_registry_snapshot: ImplementationRegistrySnapshot,
        problem: CounterfactualProblemIR,
        semantics_profile: SemanticsProfile,
        action_space_profile: ActionSpaceProfile,
        scene_state: SceneStateEnvelope,
        request: CounterfactualSolveRequest,
        backend_descriptor_bundle: BackendDescriptorBundle,
        solver_config: CounterfactualSolverConfig,
        proof_policy: ProofPolicy,
        resource_policy: ResourcePolicy,
        backend_routing_policy: BackendRoutingPolicy,
    ) -> CounterfactualSolveRequest:
        """Validate operational policy and static backend snapshot closure."""

        if (
            not isinstance(
                getattr(action_space_profile, "backend_capability_requirements", None),
                tuple,
            )
            or not action_space_profile.backend_capability_requirements
        ):
            raise SemanticContractError("backend capability requirements")
        if (
            not isinstance(
                getattr(backend_descriptor_bundle, "backend_descriptors", None), tuple
            )
            or not backend_descriptor_bundle.backend_descriptors
        ):
            raise SemanticContractError("available backend descriptors")
        if request.implementation_registry_snapshot != implementation_registry_snapshot:
            raise ImplementationResolutionError("registry snapshot")
        self.validate_semantic_problem(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
            value_schema_definitions,
            predicate_definitions,
            state_variable_definitions,
            derived_fact_rule_definitions,
            operator_definitions,
            implementation_registry_snapshot,
            problem,
            semantics_profile,
            action_space_profile,
            scene_state,
        )
        if request.semantic_problem != problem or (
            request.semantic_problem_sha256 != problem.semantic_problem_sha256
        ):
            raise SemanticContractError("solve request semantic problem")
        if request.solve_policy_definition_bundle != solve_policy_definition_bundle:
            raise DefinitionClosureError("solve policy definition bundle")
        if request.backend_descriptor_bundle != backend_descriptor_bundle:
            raise SemanticContractError("backend descriptor bundle")
        if request.solver_config != solver_config:
            raise SemanticContractError("solver config")
        if request.proof_policy != proof_policy:
            raise SemanticContractError("proof policy")
        if request.resource_policy != resource_policy:
            raise SemanticContractError("resource policy")
        if request.backend_routing_policy != backend_routing_policy:
            raise SemanticContractError("backend routing policy")
        for record, reason in (
            (request, "solve request hash"),
            (backend_descriptor_bundle, "backend descriptor bundle hash"),
            (solver_config, "solver config hash"),
            (proof_policy, "proof policy hash"),
            (resource_policy, "resource policy hash"),
            (backend_routing_policy, "backend routing policy hash"),
            (semantics_profile, "semantics profile hash"),
            (action_space_profile, "action profile hash"),
        ):
            if not _self_digest_matches(record):
                raise SemanticContractError(reason)
        _semantic_definitions, _solve_definitions, definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        policy_records = (
            solver_config,
            proof_policy,
            resource_policy,
            backend_routing_policy,
        )
        if (
            not {
                reference
                for record in policy_records
                for reference in _references(record, "definition:")
            }
            <= definitions.keys()
        ):
            raise DefinitionClosureError("solve policy definition closure")
        operational_policy_refs = (
            _references(solver_config, "definition:")
            | _references(resource_policy, "definition:")
            | _references(backend_routing_policy, "definition:")
            | {proof_policy.proof_policy_ref}
        )
        if not operational_policy_refs <= _solve_definitions.keys():
            raise DefinitionClosureError("solve policy definition root")
        if (
            not {
                *proof_policy.accepted_claim_definition_refs,
                proof_policy.publication_minimum_claim_ref,
            }
            <= _semantic_definitions.keys()
        ):
            raise DefinitionClosureError("semantic claim definition root")
        policy_roles = _definition_roles(definitions)
        solve_roles = _definition_roles(_solve_definitions)
        _validate_exact_root_definition_closure(
            _solve_definitions,
            root_records=(
                solver_config,
                proof_policy,
                resource_policy,
                backend_routing_policy,
                backend_descriptor_bundle,
            ),
            record_binding_targets={
                reference
                for reference, role in solve_roles.items()
                if role
                in {
                    _ROLE_ROUTING_SELECTION_DISPOSITION,
                    _ROLE_ROUTING_SELECTION_REASON,
                }
            },
            root_label="solve policy",
        )
        try:
            resource_accounting_ref = _definition_record_reference(
                _solve_definitions[resource_policy.resource_policy_ref],
                _RESOURCE_ACCOUNTING_CLAIM_FIELD,
            )
        except (DefinitionClosureError, KeyError) as error:
            raise DefinitionClosureError("resource policy accounting") from error
        if (
            solve_roles.get(resource_policy.resource_policy_ref)
            != _ROLE_RESOURCE_POLICY
            or solve_roles.get(resource_accounting_ref) != _ROLE_RESOURCE_ACCOUNTING
        ):
            raise DefinitionClosureError("resource policy accounting")
        try:
            routing_definition = _solve_definitions[
                backend_routing_policy.routing_policy_ref
            ]
            routing_match_ref = _definition_record_reference(
                routing_definition,
                "field:routing-match-claim",
            )
            routing_mismatch_ref = _definition_record_reference(
                routing_definition,
                "field:routing-mismatch-claim",
            )
        except (DefinitionClosureError, KeyError) as error:
            raise DefinitionClosureError("routing policy semantics") from error
        if (
            solve_roles.get(backend_routing_policy.routing_policy_ref)
            != _ROLE_ROUTING_POLICY
            or solve_roles.get(routing_match_ref) != _ROLE_ROUTING_MATCH
            or solve_roles.get(routing_mismatch_ref) != _ROLE_ROUTING_MISMATCH
        ):
            raise DefinitionClosureError("routing policy semantics")
        accepted_claim_definition_refs = set(
            proof_policy.accepted_claim_definition_refs
        )
        if (
            proof_policy.publication_minimum_claim_ref
            not in accepted_claim_definition_refs
        ):
            raise SemanticContractError("proof policy publication minimum")
        checker_capabilities = set(proof_policy.required_checker_capability_refs)
        for claim_ref in proof_policy.accepted_claim_definition_refs:
            try:
                claim_definition = _semantic_definitions[claim_ref]
                proof_material_ref = _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                claim_checker_capability = _definition_record_reference(
                    claim_definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
                proof_material_definition = _semantic_definitions[proof_material_ref]
                proof_material_checker_capability = _definition_record_reference(
                    proof_material_definition,
                    _PROOF_CHECKER_CAPABILITY_FIELD,
                )
            except (DefinitionClosureError, KeyError) as error:
                raise DefinitionClosureError("proof policy claim metadata") from error
            if (
                policy_roles.get(proof_material_ref) != _ROLE_PROOF_MATERIAL
                or claim_checker_capability != proof_material_checker_capability
                or claim_checker_capability not in checker_capabilities
                or proof_material_checker_capability not in checker_capabilities
            ):
                raise SemanticContractError("proof policy checker")
        if (
            proof_policy.proof_policy_ref
            != action_space_profile.publication_proof_policy_ref
            or not set(proof_policy.accepted_claim_definition_refs)
            <= set(action_space_profile.allowed_claim_definition_refs)
            or proof_policy.publication_minimum_claim_ref
            not in action_space_profile.allowed_claim_definition_refs
            or any(
                policy_roles.get(reference)
                not in {_ROLE_CERTIFIED_SOLUTION, _ROLE_PROVEN_UNSAT}
                for reference in proof_policy.accepted_claim_definition_refs
            )
            or policy_roles.get(proof_policy.publication_minimum_claim_ref)
            not in {_ROLE_CERTIFIED_SOLUTION, _ROLE_PROVEN_UNSAT}
        ):
            raise SemanticContractError("proof policy action profile")
        capability_owners = self._owners_by_capability()
        backend_capabilities = set(action_space_profile.backend_capability_requirements)
        for capability in backend_capabilities | checker_capabilities:
            if capability not in capability_owners:
                raise ImplementationResolutionError("capability owner")
        self._validate_snapshot(
            implementation_registry_snapshot,
            backend_capabilities | checker_capabilities,
            (
                *predicate_definitions,
                *state_variable_definitions,
                *derived_fact_rule_definitions,
                *operator_definitions,
            ),
        )
        backend_owner_by_capability = {
            capability: capability_owners[capability]
            for capability in backend_capabilities
        }
        available_backend_refs = {
            descriptor.backend_ref
            for descriptor in backend_descriptor_bundle.backend_descriptors
        }
        unavailable_backend_refs = {
            unavailable.backend_ref
            for unavailable in backend_descriptor_bundle.unavailable_optional_backends
        }
        # The domain model normally enforces this too.  Keep the registry
        # boundary defensive because tests and callers can construct frozen
        # models without normal validation.
        if available_backend_refs & unavailable_backend_refs:
            raise DefinitionClosureError("unavailable backend universe")
        descriptor_owner_refs: set[str] = set()
        for descriptor in backend_descriptor_bundle.backend_descriptors:
            if not _self_digest_matches(descriptor):
                raise SemanticContractError("backend descriptor hash")
            if not _references(descriptor, "definition:") <= definitions.keys():
                raise DefinitionClosureError("backend descriptor definition closure")
            descriptor_capabilities = _references(descriptor, "capability:")
            if not descriptor_capabilities <= capability_owners.keys():
                raise ImplementationResolutionError("backend descriptor capability")
            descriptor_owners = tuple(
                sorted(
                    (
                        owner
                        for owner in set(backend_owner_by_capability.values())
                        if owner.implementation_build_sha256
                        == descriptor.implementation_build_sha256
                    ),
                    key=lambda owner: _ref_key(owner.owner_ref),
                )
            )
            # The wire exposes a descriptor build but no owner ref.  Refuse an
            # ambiguous build rather than selecting an arbitrary set member;
            # separate descriptor rows are the explicit portfolio mechanism.
            if len(descriptor_owners) != 1:
                raise ImplementationResolutionError("backend descriptor build")
            descriptor_owner_refs.add(descriptor_owners[0].owner_ref)
        for unavailable in backend_descriptor_bundle.unavailable_optional_backends:
            if policy_roles.get(unavailable.reason_claim_definition_ref) not in {
                _ROLE_UNKNOWN,
                _ROLE_ROUTING_MISMATCH,
            }:
                raise DefinitionClosureError("unavailable backend reason")
        if (
            available_backend_refs
            and not {owner.owner_ref for owner in backend_owner_by_capability.values()}
            <= descriptor_owner_refs
        ):
            raise ImplementationResolutionError("backend descriptor build")
        return request


# Keep public class, exception and bound-method lookup stable.
RequestValidation.validate_solve_request.__module__ = "spatialcf.core.registry"
RequestValidation.validate_solve_request.__qualname__ = "StaticImplementationRegistry.validate_solve_request"
