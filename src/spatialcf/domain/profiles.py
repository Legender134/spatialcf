"""Immutable profile, policy, and availability contracts for M1.

This module closes only local structural invariants.  It intentionally does
not resolve definition references, select a backend, execute routing, track a
resource ledger, or dispatch a checker; those cross-record operations belong
to the later static registry.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self, TypeVar

from pydantic import BeforeValidator, Field, StrictBool, StrictInt, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    NonNegativeFiniteFloat,
    Sha256Digest,
)
from spatialcf.domain.definitions import (
    CapabilityRef,
    DefinitionRef,
    HashBoundCanonicalModel,
    SchemaRef,
    canonical_json_bytes,
)
from spatialcf.domain.operators import StateVariableRef, TypedVariableBound

__all__ = (
    "ActionSpaceProfile",
    "BackendDescriptorBundle",
    "BackendRoutingPolicy",
    "CounterfactualSolverConfig",
    "ImplementationOwnerBinding",
    "ImplementationRegistrySnapshot",
    "InterventionAuthorization",
    "ObjectiveExpression",
    "ObjectiveTerm",
    "ProofPolicy",
    "ResourceLimit",
    "ResourcePolicy",
    "SemanticsProfile",
    "SolverBackendDescriptor",
    "UnavailableBackendRecord",
)

_ValueT = TypeVar("_ValueT")
_PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]


def _require_profile_ref(value: str) -> str:
    if isinstance(value, str):
        profile_prefix = "spatialcf/"
        if not value.startswith(profile_prefix) or "*" in value:
            raise ValueError("profile references must be exact spatialcf versioned IDs")
        profile_path_and_version = value[len(profile_prefix) :]
        if profile_path_and_version.count("@") != 1:
            raise ValueError("profile references must be exact spatialcf versioned IDs")
        profile_path, profile_version = profile_path_and_version.split("@")
        if (
            not profile_path
            or not profile_version
            or any(segment in ("", ".", "..") for segment in profile_path.split("/"))
        ):
            raise ValueError("profile references must be exact spatialcf versioned IDs")
    return value


def _has_exact_prefixed_suffix(value: str, prefixes: tuple[str, ...]) -> bool:
    return "*" not in value and any(
        value.startswith(prefix) and len(value) > len(prefix) for prefix in prefixes
    )


def _require_exact_entity_id(value: str) -> str:
    if isinstance(value, str) and not _has_exact_prefixed_suffix(value, ("entity:",)):
        raise ValueError("editable entities must use exact entity IDs")
    return value


def _require_owner_ref(value: str) -> str:
    if isinstance(value, str) and not _has_exact_prefixed_suffix(value, ("owner:",)):
        raise ValueError("implementation owners must use exact owner references")
    return value


def _require_backend_ref(value: str) -> str:
    if isinstance(value, str) and not _has_exact_prefixed_suffix(value, ("backend:",)):
        raise ValueError("backend references must use exact backend IDs")
    return value


def _require_definition_or_capability_ref(value: str) -> str:
    if isinstance(value, str) and not _has_exact_prefixed_suffix(
        value,
        ("definition:", "capability:"),
    ):
        raise ValueError("owner bindings must name an exact definition or capability")
    return value


ProfileRef = Annotated[CanonicalId, BeforeValidator(_require_profile_ref)]
ExactEntityId = Annotated[CanonicalId, BeforeValidator(_require_exact_entity_id)]
OwnerRef = Annotated[CanonicalId, BeforeValidator(_require_owner_ref)]
BackendRef = Annotated[CanonicalId, BeforeValidator(_require_backend_ref)]
DefinitionOrCapabilityRef = Annotated[
    CanonicalId,
    BeforeValidator(_require_definition_or_capability_ref),
]


def _require_sorted_unique_by_bytes(
    values: tuple[_ValueT, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")


def _state_leaf_key(reference: StateVariableRef) -> bytes:
    return canonical_json_bytes(
        (
            reference.state_schema_ref,
            reference.fact_family_ref,
            reference.entity_or_fact_key,
            reference.field_path_ref,
        )
    )


def _require_sorted_unique_state_refs(
    references: tuple[StateVariableRef, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    _require_sorted_unique_by_bytes(references, label, nonempty=nonempty)
    leaf_keys = tuple(_state_leaf_key(reference) for reference in references)
    if len(set(leaf_keys)) != len(leaf_keys):
        raise ValueError(f"{label} must not claim one leaf through two schemas")
    if any("*" in reference.entity_or_fact_key for reference in references):
        raise ValueError(f"{label} must name exact leaf references")


class SemanticsProfile(HashBoundCanonicalModel):
    """The immutable interpretation closure shared by compatible problems."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/semantics-profile/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantics_profile_sha256"

    semantics_profile_ref: ProfileRef
    accepted_scene_and_fact_schema_refs: tuple[SchemaRef, ...]
    predicate_definition_refs: tuple[DefinitionRef, ...]
    transition_semantics_refs: tuple[DefinitionRef, ...]
    objective_definition_refs: tuple[DefinitionRef, ...]
    numeric_semantics_ref: DefinitionRef
    completeness_policy_ref: DefinitionRef
    uncertainty_policy_ref: DefinitionRef
    derived_fact_rule_refs: tuple[DefinitionRef, ...]
    observation_obligation_policy_ref: DefinitionRef
    semantics_profile_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_reference_sets(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.accepted_scene_and_fact_schema_refs,
            "accepted scene and fact schemas",
        )
        _require_sorted_unique_by_bytes(
            self.predicate_definition_refs,
            "predicate definitions",
        )
        _require_sorted_unique_by_bytes(
            self.transition_semantics_refs,
            "transition semantics",
        )
        _require_sorted_unique_by_bytes(
            self.objective_definition_refs,
            "objective definitions",
        )
        _require_sorted_unique_by_bytes(
            self.derived_fact_rule_refs,
            "derived fact rules",
        )
        return self


class ActionSpaceProfile(HashBoundCanonicalModel):
    """The reusable state/action capability universe without implementation code."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/action-space-profile/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "action_space_profile_sha256"

    action_space_profile_ref: ProfileRef
    accepted_scene_schema_refs: tuple[SchemaRef, ...]
    state_variable_definition_refs: tuple[DefinitionRef, ...]
    allowed_operator_refs: tuple[DefinitionRef, ...]
    mandatory_invariant_template_refs: tuple[DefinitionRef, ...]
    predicate_capability_refs: tuple[CapabilityRef, ...]
    objective_capability_refs: tuple[CapabilityRef, ...]
    numeric_semantics_ref: DefinitionRef
    allowed_claim_definition_refs: tuple[DefinitionRef, ...]
    backend_capability_requirements: tuple[CapabilityRef, ...]
    adapter_capability_requirements: tuple[CapabilityRef, ...]
    publication_proof_policy_ref: DefinitionRef
    action_space_profile_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_capability_sets(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.accepted_scene_schema_refs,
            "accepted scene schemas",
        )
        _require_sorted_unique_by_bytes(
            self.state_variable_definition_refs,
            "state variable definitions",
        )
        _require_sorted_unique_by_bytes(
            self.allowed_operator_refs,
            "allowed operators",
        )
        _require_sorted_unique_by_bytes(
            self.mandatory_invariant_template_refs,
            "mandatory invariant templates",
        )
        _require_sorted_unique_by_bytes(
            self.predicate_capability_refs,
            "predicate capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.objective_capability_refs,
            "objective capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.allowed_claim_definition_refs,
            "allowed claims",
        )
        _require_sorted_unique_by_bytes(
            self.backend_capability_requirements,
            "backend capability requirements",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.adapter_capability_requirements,
            "adapter capability requirements",
        )
        return self


class InterventionAuthorization(HashBoundCanonicalModel):
    """Request-local exact write authority, never a profile-level capability grant."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/intervention-authorization/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "intervention_authorization_sha256"

    editable_entity_ids: tuple[ExactEntityId, ...]
    allowed_operator_refs: tuple[DefinitionRef, ...]
    authorized_primary_write_set: tuple[StateVariableRef, ...]
    variable_bounds: tuple[TypedVariableBound, ...]
    maximum_program_steps: _PositiveStrictInt
    maximum_edited_entities: _PositiveStrictInt
    required_derived_rule_refs: tuple[DefinitionRef, ...]
    complete_state_delta_policy_ref: DefinitionRef
    intervention_authorization_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_exact_authorization(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.editable_entity_ids,
            "editable entity IDs",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.allowed_operator_refs,
            "authorized operators",
            nonempty=True,
        )
        _require_sorted_unique_state_refs(
            self.authorized_primary_write_set,
            "authorized primary write set",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.required_derived_rule_refs,
            "required derived rules",
        )
        _require_sorted_unique_by_bytes(
            self.variable_bounds,
            "variable bounds",
        )
        bound_keys = tuple(
            _state_leaf_key(bound.state_variable_ref) for bound in self.variable_bounds
        )
        if len(set(bound_keys)) != len(bound_keys):
            raise ValueError("variable bounds must not duplicate one state leaf")
        authorized_keys = {
            _state_leaf_key(reference)
            for reference in self.authorized_primary_write_set
        }
        if any(key not in authorized_keys for key in bound_keys):
            raise ValueError("variable bounds must target authorized primary writes")
        if any(
            "*" in bound.state_variable_ref.entity_or_fact_key
            for bound in self.variable_bounds
        ):
            raise ValueError("variable bounds must name exact leaf references")
        return self


class ObjectiveTerm(CanonicalModel):
    """One ordered, typed term whose metric semantics remain definition-bound."""

    term_id: CanonicalId
    objective_definition_ref: DefinitionRef
    input_selector_definition_ref: DefinitionRef
    unit_ref: DefinitionRef
    normalization_definition_ref: DefinitionRef


class ObjectiveExpression(HashBoundCanonicalModel):
    """A deterministic objective expression with preserved declared term order."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/objective-expression/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "objective_expression_sha256"

    aggregation_definition_ref: DefinitionRef
    terms: tuple[ObjectiveTerm, ...]
    deterministic_tie_break_definition_ref: DefinitionRef
    objective_expression_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_ordered_terms(self) -> Self:
        if not self.terms:
            raise ValueError("objective terms must not be empty")
        term_ids = tuple(term.term_id for term in self.terms)
        if len(set(term_ids)) != len(term_ids):
            raise ValueError("objective expression must not contain duplicate term IDs")
        return self


class ProofPolicy(HashBoundCanonicalModel):
    """The immutable claim/checker policy required to publish a terminal record."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/proof-policy/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_policy_sha256"

    proof_policy_ref: DefinitionRef
    accepted_claim_definition_refs: tuple[DefinitionRef, ...]
    required_checker_capability_refs: tuple[CapabilityRef, ...]
    publication_minimum_claim_ref: DefinitionRef
    permit_noncertified_terminal_records: StrictBool
    proof_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_policy_sets(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.accepted_claim_definition_refs,
            "accepted proof claims",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.required_checker_capability_refs,
            "required checker capabilities",
            nonempty=True,
        )
        return self


class ResourceLimit(CanonicalModel):
    """A finite limit whose unit, counting, and overflow semantics are referenced."""

    definition_ref: DefinitionRef
    finite_limit: NonNegativeFiniteFloat


class ResourcePolicy(HashBoundCanonicalModel):
    """The declarative resource closure; it does not contain a runtime ledger."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/resource-policy/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "resource_policy_sha256"

    resource_policy_ref: DefinitionRef
    limits: tuple[ResourceLimit, ...]
    exhaustion_claim_ref: DefinitionRef
    shared_ledger_policy_ref: DefinitionRef
    resource_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_limits(self) -> Self:
        _require_sorted_unique_by_bytes(self.limits, "resource limits")
        limit_refs = tuple(limit.definition_ref for limit in self.limits)
        if len(set(limit_refs)) != len(limit_refs):
            raise ValueError("resource limits must not duplicate one definition")
        return self


class BackendRoutingPolicy(HashBoundCanonicalModel):
    """A versioned routing declaration without executable selection hooks."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/backend-routing-policy/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_routing_policy_sha256"

    routing_policy_ref: DefinitionRef
    capability_filter_definition_ref: DefinitionRef
    deterministic_order_definition_ref: DefinitionRef
    portfolio_composition_definition_ref: DefinitionRef
    stop_condition_definition_ref: DefinitionRef
    resource_partition_definition_ref: DefinitionRef
    backend_routing_policy_sha256: Sha256Digest


class CounterfactualSolverConfig(HashBoundCanonicalModel):
    """The frozen compilation/proposal configuration consumed by a backend."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/solver-config/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "solver_config_sha256"

    solver_config_ref: DefinitionRef
    compilation_policy_ref: DefinitionRef
    proposal_policy_ref: DefinitionRef
    objective_bound_policy_ref: DefinitionRef
    determinism_policy_ref: DefinitionRef
    solver_config_sha256: Sha256Digest


class ImplementationOwnerBinding(CanonicalModel):
    """One committed definition/capability-to-static-owner mapping."""

    definition_or_capability_ref: DefinitionOrCapabilityRef
    implementation_owner_ref: OwnerRef


class ImplementationRegistrySnapshot(HashBoundCanonicalModel):
    """A closed operational owner/build snapshot independent of ambient installs."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/implementation-registry-snapshot/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "implementation_registry_snapshot_sha256"

    definition_and_capability_owner_bindings: tuple[ImplementationOwnerBinding, ...]
    implementation_build_hashes: tuple[tuple[OwnerRef, Sha256Digest], ...]
    dependency_lock_sha256: Sha256Digest
    implementation_registry_snapshot_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_snapshot_closure(self) -> Self:
        bindings = self.definition_and_capability_owner_bindings
        _require_sorted_unique_by_bytes(
            bindings,
            "definition and capability owner bindings",
            nonempty=True,
        )
        binding_refs = tuple(
            binding.definition_or_capability_ref for binding in bindings
        )
        if len(set(binding_refs)) != len(binding_refs):
            raise ValueError(
                "same definition or capability may not bind different owners"
            )

        owner_builds = self.implementation_build_hashes
        encoded_builds = tuple(canonical_json_bytes(build) for build in owner_builds)
        if encoded_builds != tuple(sorted(encoded_builds)):
            raise ValueError("implementation owner builds must be sorted")
        if len(set(encoded_builds)) != len(encoded_builds):
            raise ValueError("duplicate owner-build entry")
        if len({owner for owner, _build_hash in owner_builds}) != len(owner_builds):
            raise ValueError("duplicate implementation owner build")
        build_owners = {owner for owner, _build_hash in owner_builds}
        binding_owners = {binding.implementation_owner_ref for binding in bindings}
        if build_owners != binding_owners:
            raise ValueError("owner bindings and exact owner builds must close")
        return self


class SolverBackendDescriptor(HashBoundCanonicalModel):
    """One available backend's complete, hash-bound capability declaration."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/solver-backend-descriptor/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_descriptor_sha256"

    backend_ref: BackendRef
    implementation_build_sha256: Sha256Digest
    supported_profile_hashes: tuple[Sha256Digest, ...]
    supported_predicate_capabilities: tuple[CapabilityRef, ...]
    supported_operator_capabilities: tuple[CapabilityRef, ...]
    supported_objective_capabilities: tuple[CapabilityRef, ...]
    supported_numeric_semantics: tuple[DefinitionRef, ...]
    emitted_proof_material_definition_refs: tuple[DefinitionRef, ...]
    compatible_checker_capability_refs: tuple[CapabilityRef, ...]
    resource_definition_refs: tuple[DefinitionRef, ...]
    backend_descriptor_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_capability_closure(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.supported_profile_hashes,
            "supported profile hashes",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.supported_predicate_capabilities,
            "supported predicate capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.supported_operator_capabilities,
            "supported operator capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.supported_objective_capabilities,
            "supported objective capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.supported_numeric_semantics,
            "supported numeric semantics",
        )
        _require_sorted_unique_by_bytes(
            self.emitted_proof_material_definition_refs,
            "emitted proof material definitions",
        )
        _require_sorted_unique_by_bytes(
            self.compatible_checker_capability_refs,
            "compatible checker capabilities",
        )
        _require_sorted_unique_by_bytes(
            self.resource_definition_refs,
            "resource definitions",
        )
        return self


class UnavailableBackendRecord(CanonicalModel):
    """One explicitly considered but unavailable optional backend."""

    backend_ref: BackendRef
    reason_claim_definition_ref: DefinitionRef


class BackendDescriptorBundle(HashBoundCanonicalModel):
    """The full canonical backend universe for a single solve request."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/backend-descriptor-bundle/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_descriptor_bundle_sha256"

    backend_descriptors: tuple[SolverBackendDescriptor, ...]
    unavailable_optional_backends: tuple[UnavailableBackendRecord, ...]
    backend_descriptor_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_backend_universe(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.backend_descriptors,
            "available backend descriptors",
            nonempty=True,
        )
        _require_sorted_unique_by_bytes(
            self.unavailable_optional_backends,
            "unavailable backend records",
        )
        available_refs = tuple(
            descriptor.backend_ref for descriptor in self.backend_descriptors
        )
        unavailable_refs = tuple(
            record.backend_ref for record in self.unavailable_optional_backends
        )
        if len(set(available_refs)) != len(available_refs):
            raise ValueError("duplicate backend descriptor")
        if len(set(unavailable_refs)) != len(unavailable_refs):
            raise ValueError("duplicate unavailable backend")
        if set(available_refs) & set(unavailable_refs):
            raise ValueError("backend cannot be both available and unavailable")
        return self
