"""Pure capability matching and protocol declarations for M1 backends.

This module deliberately contains no backend discovery, compilation, solving,
or proof checking.  It works only over records supplied by the caller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spatialcf.domain.counterfactual import CounterfactualSolveRequest
from spatialcf.domain.definitions import (
    HashBoundCanonicalModel,
    ValueKind,
    canonical_json_bytes,
)
from spatialcf.domain.operators import OperatorDefinition
from spatialcf.domain.outcomes import (
    BackendProposal,
    CapabilityMatch,
    CapabilityMismatch,
    TypedCompilationOutcome,
)
from spatialcf.domain.predicates import PredicateDefinition
from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    CounterfactualSolverConfig,
    SemanticsProfile,
    SolverBackendDescriptor,
)
from spatialcf.domain.serialization import canonical_sha256

__all__ = (
    "CompiledProblemProtocol",
    "SolverBackendProtocol",
    "match_backend_capabilities",
    "order_backend_matches",
)

_MISMATCH_ORDER = (
    "profile",
    "predicate",
    "operator",
    "objective",
    "numeric",
    "proof",
    "resource",
    "backend",
)

_PROOF_MATERIAL_ROLE = "definition-kind:proof-material"
_CERTIFIED_SOLUTION_ROLE = "definition-kind:claim-certified-solution"
_PROVEN_UNSAT_ROLE = "definition-kind:claim-proven-unsat"
_ROUTING_POLICY_ROLE = "definition-kind:routing-policy"
_ROUTING_MATCH_ROLE = "definition-kind:routing-match"
_ROUTING_MISMATCH_ROLE = "definition-kind:routing-mismatch"
_RECORD_BINDING_ROLE = "definition-kind:record-binding"
_RECORD_KIND_FIELD = "field:definition-kind"
_RECORD_REFERENCE_FIELD = "field:definition-reference"
_RECORD_BOUND_REFERENCE_FIELD = "field:bound-record-ref"
_RECORD_BOUND_SHA256_FIELD = "field:bound-record-sha256"
_ROUTING_MATCH_CLAIM_FIELD = "field:routing-match-claim"
_ROUTING_MISMATCH_CLAIM_FIELD = "field:routing-mismatch-claim"
_CLAIM_PROOF_MATERIAL_FIELD = "field:claim-proof-material-definition"


@runtime_checkable
class CompiledProblemProtocol(Protocol):
    """The minimal hash-only product passed from compile to solve."""

    @property
    def semantic_problem_sha256(self) -> str: ...

    @property
    def solve_request_sha256(self) -> str: ...

    @property
    def selected_backend_descriptor_sha256(self) -> str: ...

    @property
    def grounded_obligation_set_sha256(self) -> str: ...

    @property
    def compiled_artifact_sha256s(self) -> tuple[str, ...]: ...


@runtime_checkable
class SolverBackendProtocol(Protocol):
    """A declaration-only backend boundary; concrete solving lives elsewhere."""

    def inspect(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> CapabilityMatch | CapabilityMismatch: ...

    def compile(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> CompiledProblemProtocol | TypedCompilationOutcome: ...

    def solve(
        self,
        compiled: CompiledProblemProtocol,
        config: CounterfactualSolverConfig,
    ) -> BackendProposal: ...


def _self_digest_matches(model: HashBoundCanonicalModel) -> bool:
    """Check a submitted hash-bound record without reconstructing it."""

    digest_field = model.SELF_DIGEST_FIELD
    payload = model.model_dump(
        mode="python",
        by_alias=True,
        exclude={digest_field},
        exclude_none=False,
        exclude_defaults=False,
        exclude_unset=False,
        exclude_computed_fields=True,
        round_trip=True,
    )
    return getattr(model, digest_field) == canonical_sha256(
        payload,
        domain=model.HASH_DOMAIN,
    )


def _sorted_capabilities(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=canonical_json_bytes))


def _missing_capability(reason: str, value: str) -> str:
    """Emit a canonical capability row even when the missing thing is a hash."""

    digest = canonical_sha256(
        (reason, value),
        domain="spatialcf/counterfactual/backend-capability-mismatch/3.0",
    )
    return f"capability:spatialcf/counterfactual/mismatch/{reason}/{digest}"


def _required_predicate_capabilities(
    profile: ActionSpaceProfile,
    definitions: tuple[PredicateDefinition, ...],
) -> set[str]:
    required = set(profile.predicate_capability_refs)
    for definition in definitions:
        required.add(definition.evaluator_capability_ref)
        required.add(definition.verifier_capability_ref)
    return required


def _required_operator_capabilities(
    definitions: tuple[OperatorDefinition, ...],
) -> set[str]:
    required: set[str] = set()
    for definition in definitions:
        required.add(definition.compiler_capability_ref)
        required.add(definition.verifier_capability_ref)
    return required


def _definition_record_role(definition) -> str | None:
    """Return a typed-record role, never a semantic inference from an ID."""

    payload = definition.payload.payload
    if payload.kind is not ValueKind.RECORD:
        return None
    fields = {field.name: field.value.payload for field in payload.fields}
    role = fields.get(_RECORD_KIND_FIELD)
    identity = fields.get(_RECORD_REFERENCE_FIELD)
    if (
        role is None
        or role.kind is not ValueKind.ENUM_SYMBOL
        or identity is None
        or identity.kind is not ValueKind.CANONICAL_ID
        or identity.value != definition.definition_ref
    ):
        return None
    return role.symbol


def _definition_record_reference(definition, field_name: str) -> str:
    """Resolve one explicit reference from a closed definition payload."""

    payload = definition.payload.payload
    if payload.kind is not ValueKind.RECORD:
        raise ValueError("definition metadata is not a typed record")
    fields = {field.name: field.value.payload for field in payload.fields}
    value = fields.get(field_name)
    if value is None or value.kind is not ValueKind.CANONICAL_ID:
        raise ValueError("definition metadata reference is missing")
    return value.value


def _definition_record_digest(definition, field_name: str) -> str:
    """Resolve one explicit digest from a typed definition record."""

    payload = definition.payload.payload
    if payload.kind is not ValueKind.RECORD:
        raise ValueError("definition metadata is not a typed record")
    fields = {field.name: field.value.payload for field in payload.fields}
    value = fields.get(field_name)
    if value is None or value.kind is not ValueKind.DIGEST:
        raise ValueError("definition metadata digest is missing")
    return value.value


def _frozen_definitions(
    solve_request: CounterfactualSolveRequest,
) -> dict[str, object]:
    """Combine the two explicit roots without allowing one to override another."""

    definitions: dict[str, object] = {}
    for definition in (
        *solve_request.semantic_problem.definition_bundle.definitions,
        *solve_request.solve_policy_definition_bundle.definitions,
    ):
        existing = definitions.get(definition.definition_ref)
        if existing is not None:
            if canonical_json_bytes(existing) != canonical_json_bytes(definition):
                raise ValueError("frozen definition roots conflict")
            raise ValueError("frozen definition roots overlap")
        definitions[definition.definition_ref] = definition
    return definitions


def _solve_policy_definitions(
    solve_request: CounterfactualSolveRequest,
) -> dict[str, object]:
    """Compatibility name for the request's explicit combined closure."""

    return _frozen_definitions(solve_request)


def _frozen_definition_bindings(
    solve_request: CounterfactualSolveRequest,
) -> dict[str, str]:
    """Return exact record digests declared by the request semantic root."""

    definitions = {
        definition.definition_ref: definition
        for definition in solve_request.semantic_problem.definition_bundle.definitions
    }
    bindings: dict[str, str] = {}
    for definition in definitions.values():
        if _definition_record_role(definition) != _RECORD_BINDING_ROLE:
            continue
        reference = _definition_record_reference(
            definition,
            _RECORD_BOUND_REFERENCE_FIELD,
        )
        digest = _definition_record_digest(
            definition,
            _RECORD_BOUND_SHA256_FIELD,
        )
        if reference in bindings and bindings[reference] != digest:
            raise ValueError("frozen definition bindings conflict")
        bindings[reference] = digest
    return bindings


def _required_proof_material_definitions(
    solve_request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
) -> set[str]:
    definitions = _solve_policy_definitions(solve_request)
    accepted_claims = set(solve_request.proof_policy.accepted_claim_definition_refs)
    if not accepted_claims <= set(action_space_profile.allowed_claim_definition_refs):
        raise ValueError("proof policy claims do not match the frozen profile")
    proof_definitions: set[str] = set()
    for claim_ref in accepted_claims:
        claim = definitions.get(claim_ref)
        if claim is None or _definition_record_role(claim) not in {
            _CERTIFIED_SOLUTION_ROLE,
            _PROVEN_UNSAT_ROLE,
        }:
            raise ValueError("accepted claim does not have typed proof semantics")
        proof_ref = _definition_record_reference(claim, _CLAIM_PROOF_MATERIAL_FIELD)
        proof = definitions.get(proof_ref)
        if proof is None or _definition_record_role(proof) != _PROOF_MATERIAL_ROLE:
            raise ValueError("accepted claim proof material is not closed")
        proof_definitions.add(proof_ref)
    return proof_definitions


def _routing_claim_refs(
    solve_request: CounterfactualSolveRequest,
) -> tuple[str, str]:
    """Bind success/failure claims to the request's frozen routing policy."""

    routing = solve_request.backend_routing_policy
    definitions = _solve_policy_definitions(solve_request)
    definition = definitions.get(routing.routing_policy_ref)
    if (
        definition is None
        or _definition_record_role(definition) != _ROUTING_POLICY_ROLE
    ):
        raise ValueError("routing policy metadata is not closed")
    match_ref = _definition_record_reference(definition, _ROUTING_MATCH_CLAIM_FIELD)
    mismatch_ref = _definition_record_reference(
        definition,
        _ROUTING_MISMATCH_CLAIM_FIELD,
    )
    match_definition = definitions.get(match_ref)
    mismatch_definition = definitions.get(mismatch_ref)
    if (
        match_definition is None
        or mismatch_definition is None
        or _definition_record_role(match_definition) != _ROUTING_MATCH_ROLE
        or _definition_record_role(mismatch_definition) != _ROUTING_MISMATCH_ROLE
    ):
        raise ValueError("routing claim definitions do not have typed routing roles")
    return match_ref, mismatch_ref


def _require_frozen_match_inputs(
    solve_request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    semantics_profile: SemanticsProfile,
    predicate_definitions: tuple[PredicateDefinition, ...],
    operator_definitions: tuple[OperatorDefinition, ...],
) -> None:
    """Bind matcher inputs to the supplied request before comparing support."""

    problem = solve_request.semantic_problem
    if solve_request.semantic_problem_sha256 != problem.semantic_problem_sha256:
        raise ValueError("shared roots do not match the embedded semantic problem")
    records = (
        solve_request,
        problem,
        solve_request.implementation_registry_snapshot,
        solve_request.backend_descriptor_bundle,
        solve_request.backend_routing_policy,
        solve_request.proof_policy,
        solve_request.resource_policy,
        action_space_profile,
        semantics_profile,
        *predicate_definitions,
        *operator_definitions,
    )
    if any(not _self_digest_matches(record) for record in records):
        raise ValueError("frozen matcher input hash")
    if (
        action_space_profile.action_space_profile_ref
        != problem.action_space_profile_ref
        or semantics_profile.semantics_profile_ref != problem.semantics_profile_ref
        or action_space_profile.numeric_semantics_ref != problem.numeric_semantics_ref
        or semantics_profile.numeric_semantics_ref != problem.numeric_semantics_ref
    ):
        raise ValueError("profile roots do not match the frozen semantic problem")
    profile_bindings = _frozen_definition_bindings(solve_request)
    if (
        profile_bindings.get(semantics_profile.semantics_profile_ref)
        != semantics_profile.semantics_profile_sha256
        or profile_bindings.get(action_space_profile.action_space_profile_ref)
        != action_space_profile.action_space_profile_sha256
    ):
        raise ValueError("profile hashes do not match the frozen definition binding")
    if {definition.predicate_ref for definition in predicate_definitions} != set(
        semantics_profile.predicate_definition_refs
    ):
        raise ValueError("predicate definitions do not match the frozen profile")
    if {definition.operator_ref for definition in operator_definitions} != set(
        action_space_profile.allowed_operator_refs
    ):
        raise ValueError("operator definitions do not match the frozen profile")
    definition_refs = set(_frozen_definitions(solve_request))
    if not set(_routing_claim_refs(solve_request)) <= definition_refs:
        raise ValueError("routing definitions do not match the frozen policy")


def _backend_requirements_for_descriptor_build(
    solve_request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    descriptor: SolverBackendDescriptor,
) -> tuple[set[str], set[str], set[str]]:
    """Partition requirements into this build, another build, and unresolved rows."""

    snapshot = solve_request.implementation_registry_snapshot
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


def match_backend_capabilities(
    solve_request: CounterfactualSolveRequest,
    action_space_profile: ActionSpaceProfile,
    semantics_profile: SemanticsProfile,
    predicate_definitions: tuple[PredicateDefinition, ...],
    operator_definitions: tuple[OperatorDefinition, ...],
    descriptor: SolverBackendDescriptor,
) -> CapabilityMatch | CapabilityMismatch:
    """Compare one descriptor against only the supplied frozen records.

    A non-support result is data, not control flow: all mismatch dimensions are
    represented in the returned canonical row in the fixed order above.
    """

    _require_frozen_match_inputs(
        solve_request,
        action_space_profile,
        semantics_profile,
        predicate_definitions,
        operator_definitions,
    )
    match_claim, mismatch_claim = _routing_claim_refs(solve_request)
    if not _self_digest_matches(descriptor):
        return CapabilityMismatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            missing_capability_refs=(
                _missing_capability("descriptor", descriptor.backend_ref),
            ),
            reason_claim_definition_ref=mismatch_claim,
        )

    missing_by_reason: dict[str, set[str]] = {}

    if action_space_profile.action_space_profile_sha256 not in set(
        descriptor.supported_profile_hashes
    ):
        missing_by_reason["profile"] = {
            _missing_capability(
                "profile",
                action_space_profile.action_space_profile_sha256,
            )
        }

    predicate_required = _required_predicate_capabilities(
        action_space_profile,
        predicate_definitions,
    )
    predicate_missing = predicate_required - set(
        descriptor.supported_predicate_capabilities
    )
    if predicate_missing:
        missing_by_reason["predicate"] = set(predicate_missing)

    operator_required = _required_operator_capabilities(operator_definitions)
    operator_missing = operator_required - set(
        descriptor.supported_operator_capabilities
    )
    if operator_missing:
        missing_by_reason["operator"] = set(operator_missing)

    objective_missing = set(action_space_profile.objective_capability_refs) - set(
        descriptor.supported_objective_capabilities
    )
    if objective_missing:
        missing_by_reason["objective"] = set(objective_missing)

    numeric_required = {
        semantics_profile.numeric_semantics_ref,
        solve_request.semantic_problem.numeric_semantics_ref,
    }
    numeric_missing = numeric_required - set(descriptor.supported_numeric_semantics)
    if numeric_missing:
        missing_by_reason["numeric"] = {
            _missing_capability("numeric", reference) for reference in numeric_missing
        }

    proof_missing = set(
        solve_request.proof_policy.required_checker_capability_refs
    ) - set(descriptor.compatible_checker_capability_refs)
    proof_definition_missing = _required_proof_material_definitions(
        solve_request,
        action_space_profile,
    ) - set(descriptor.emitted_proof_material_definition_refs)
    if proof_definition_missing:
        proof_missing |= {
            _missing_capability("proof-material", reference)
            for reference in proof_definition_missing
        }
    if proof_missing:
        missing_by_reason["proof"] = set(proof_missing)

    resource_missing = {
        limit.definition_ref for limit in solve_request.resource_policy.limits
    } - set(descriptor.resource_definition_refs)
    if resource_missing:
        missing_by_reason["resource"] = {
            _missing_capability("resource", reference) for reference in resource_missing
        }

    (
        matched_backend_requirements,
        _other_valid_build_requirements,
        structurally_unresolved_backend_requirements,
    ) = _backend_requirements_for_descriptor_build(
        solve_request,
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
            item
            for item in solve_request.backend_descriptor_bundle.backend_descriptors
            if item.backend_ref == descriptor.backend_ref
        ),
        None,
    )
    if (
        registered is None
        or registered.backend_descriptor_sha256 != descriptor.backend_descriptor_sha256
        or canonical_json_bytes(registered) != canonical_json_bytes(descriptor)
    ):
        missing_by_reason.setdefault("backend", set()).add(
            _missing_capability("backend", descriptor.backend_ref)
        )

    if missing_by_reason:
        return CapabilityMismatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            missing_capability_refs=_sorted_capabilities(
                {
                    capability
                    for reason_name in _MISMATCH_ORDER
                    for capability in missing_by_reason.get(reason_name, set())
                }
            ),
            reason_claim_definition_ref=mismatch_claim,
        )

    matched = (
        matched_backend_requirements
        | predicate_required
        | operator_required
        | set(action_space_profile.objective_capability_refs)
        | set(solve_request.proof_policy.required_checker_capability_refs)
    )
    return CapabilityMatch(
        backend_ref=descriptor.backend_ref,
        backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        matched_capability_refs=_sorted_capabilities(matched),
        match_claim_definition_ref=match_claim,
    )


def order_backend_matches(
    rows: tuple[CapabilityMatch | CapabilityMismatch, ...],
) -> tuple[CapabilityMatch | CapabilityMismatch, ...]:
    """Return the one stable routing order without selecting a backend."""

    backend_refs = tuple(row.backend_ref for row in rows)
    if len(set(backend_refs)) != len(backend_refs):
        raise ValueError("backend matches must not duplicate one backend")
    return tuple(sorted(rows, key=lambda row: canonical_json_bytes(row.backend_ref)))
