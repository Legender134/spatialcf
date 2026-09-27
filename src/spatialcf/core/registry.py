"""Static capability registry composed from explicit stateless validation owners.

The immutable registry resolves only the trusted objects supplied in StaticOwner
rows; it never loads an implementation or executes a backend/checker operation.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter, ValidationError

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    EditProgram,
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    BOOTSTRAP_SCHEMA_SHA256,
    CanonicalDefinitionEnvelope,
    CapabilityRef,
    DefinitionBundle,
    HashBoundCanonicalModel,
    TypedValue,
    ValueKind,
    ValueSchemaDefinition,
    canonical_json_bytes,
)

from spatialcf.domain.operators import (
    DerivedFactRuleDefinition,
    OperatorDefinition,
    StateVariableDefinition,
    StateVariableRef,
    WriteAuthority,
)

from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposal,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendSubmission,
    BackendUnknownEvidence,
    CapabilityMatch,
    CapabilityMismatch,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    CheckedProofOutcome,
    CheckerDisposition,
    NoncertifiedWitnessResult,
    ProofMaterialEnvelope,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    ResourceUsage,
    UnknownResult,
    VerifierDispatchRecord,
)

from spatialcf.domain.predicates import (
    GroundedObligationSet,
    PredicateAtom,
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    OwnerRef,
    ProofPolicy,
    ResourcePolicy,
    SemanticsProfile,
    SolverBackendDescriptor,
)

from spatialcf.domain.scene import CanonicalScene

from spatialcf.domain.serialization import canonical_sha256

from spatialcf.core._internal.registry.contracts import (
    _UNCHANGED_LEAF_DOMAIN,
    _FROZEN_LEAF_PATH_DOMAIN,
    _EXTENSION_FACT_ADDRESS_DOMAIN,
    _ROLE_CERTIFIED_SOLUTION,
    _ROLE_PROVEN_UNSAT,
    _ROLE_NONCERTIFIED_WITNESS,
    _ROLE_UNKNOWN,
    _ROLE_COMPLETE_DOMAIN,
    _ROLE_SOUND_COMPLETE_DOMAIN,
    _ROLE_PROOF_MATERIAL,
    _ROLE_RECORD_BINDING,
    _ROLE_RESOURCE_ACCOUNTING,
    _ROLE_RESOURCE_POLICY,
    _ROLE_ROUTING_POLICY,
    _ROLE_ROUTING_MATCH,
    _ROLE_ROUTING_MISMATCH,
    _ROLE_ROUTING_SELECTION_DISPOSITION,
    _ROLE_ROUTING_SELECTION_REASON,
    _MISMATCH_ORDER,
    _RECORD_KIND_FIELD,
    _RECORD_REFERENCE_FIELD,
    _RECORD_BOUND_REFERENCE_FIELD,
    _RECORD_BOUND_SHA256_FIELD,
    _STATE_SCHEMA_FIELD,
    _STATE_LEAF_SCHEMA_FIELD,
    _STATE_FACT_FAMILY_FIELD,
    _STATE_ENTITY_KEY_FIELD,
    _STATE_FIELD_PATH_FIELD,
    _STATE_VALUE_SCHEMA_FIELD,
    _STATE_FRAME_FIELD,
    _STATE_UNIT_FIELD,
    _STATE_TOPOLOGY_FIELD,
    _PREREQUISITE_FACT_FAMILY_FIELD,
    _OBJECTIVE_INPUT_SELECTOR_FIELD,
    _OBJECTIVE_UNIT_FIELD,
    _OBJECTIVE_NORMALIZATION_FIELD,
    _PROOF_PAYLOAD_SCHEMA_FIELD,
    _PROOF_CHECKER_CAPABILITY_FIELD,
    _CLAIM_PROOF_MATERIAL_FIELD,
    _CLAIM_CHECKER_CAPABILITY_FIELD,
    _COMPLETE_DOMAIN_CLAIM_FIELD,
    _ROUTING_MATCH_CLAIM_FIELD,
    _ROUTING_MISMATCH_CLAIM_FIELD,
    _RESOURCE_ACCOUNTING_CLAIM_FIELD,
    _STATE_METADATA_FIELDS,
    _OWNER_REF_ADAPTER,
    _CAPABILITY_REF_ADAPTER,
    DefinitionClosureError,
    SemanticContractError,
    ImplementationResolutionError,
    _is_sha256,
    _ref_key,
    _state_key,
    _self_digest_matches,
    StaticOwner,
)

from spatialcf.core._internal.registry.definitions import (
    _walk_values,
    _references,
    _schema_dependency_refs,
    _semantic_schema_definition_closure,
    _validate_typed_value,
    _definition_record_fields,
    _definition_record_reference,
    _definition_record_digest,
    _definition_record_role,
    _definition_roles,
    _definition_payload_dependency_refs,
    _root_bootstrap_anchor,
    _has_exact_root_bootstrap_shape,
    _definition_root_maps,
    _definition_record_bindings,
    _validate_exact_root_definition_closure,
    _require_definition_role,
    _require_outcome_definition_role,
    _validate_resource_usage,
    DefinitionsValidation,
)

from spatialcf.core._internal.registry.outcome import (
    validate_submission_contract_structure,
    OutcomeValidation,
)

from spatialcf.core._internal.registry.problem import (
    _formula_predicate_refs,
    _formula_atoms,
    _formula_is_fully_grounded,
    ProblemValidation,
)

from spatialcf.core._internal.registry.program import (
    ProgramValidation,
)

from spatialcf.core._internal.registry.request import (
    RequestValidation,
)

from spatialcf.core._internal.registry.routing import (
    _selection_missing_capability,
    _selection_routing_claim_refs,
    _selection_proof_material_refs,
    _selection_backend_requirements_for_descriptor_build,
    _reconstruct_backend_match,
    _reconstructed_backend_rows,
)

from spatialcf.core._internal.registry.state import (
    _state_definition_metadata,
    _state_definition_semantics,
    _raw_value_matches_schema,
    _validate_state_leaf_value_wires,
    _base_fact_rows,
    _frozen_scalar_values,
    _frozen_leaf_token,
    _extension_fact_address,
    _extension_facts_by_address,
    _state_leaf_owners,
    _declared_extension_leaf_owners,
    _expected_base_leaf_addresses,
    _validated_base_scene_payload,
    _leaf_values,
    _validate_scene_state_envelope,
)

__all__ = (
    "DefinitionClosureError",
    "ImplementationResolutionError",
    "SemanticContractError",
    "StaticImplementationRegistry",
    "StaticOwner",
)


@dataclass(frozen=True)
class StaticImplementationRegistry(
    DefinitionsValidation,
    OutcomeValidation,
    ProblemValidation,
    ProgramValidation,
    RequestValidation,
):
    """A finite immutable map from static capabilities to trusted owners."""

    owners: tuple[StaticOwner, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owners, tuple) or not all(
            isinstance(owner, StaticOwner) for owner in self.owners
        ):
            raise TypeError("static registry owners must be an immutable tuple")
        if not self.owners:
            raise ImplementationResolutionError("static registry must not be empty")
        owner_refs = tuple(owner.owner_ref for owner in self.owners)
        if len(set(owner_refs)) != len(owner_refs):
            raise ImplementationResolutionError("duplicate static owner")
        capabilities = tuple(
            capability for owner in self.owners for capability in owner.capability_refs
        )
        if len(set(capabilities)) != len(capabilities):
            raise ImplementationResolutionError("single-owner capability")

    def _owners_by_ref(self) -> dict[str, StaticOwner]:
        return {owner.owner_ref: owner for owner in self.owners}

    def _owners_by_capability(self) -> dict[str, StaticOwner]:
        return {
            capability: owner
            for owner in self.owners
            for capability in owner.capability_refs
        }
