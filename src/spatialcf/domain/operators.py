"""Closed state-authority and transition-operator contracts for M1.

This module owns only immutable, hash-bound structural contracts.  It does not
apply a state transition, recompute derived facts, resolve references, or load
an implementation; those cross-record concerns belong to the later registry.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Self

from pydantic import BeforeValidator, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel, Sha256Digest
from spatialcf.domain.definitions import (
    CapabilityRef,
    DefinitionRef,
    HashBoundCanonicalModel,
    SchemaRef,
    TypedValue,
    canonical_json_bytes,
)
from spatialcf.domain.predicates import BeforePrecondition, GroundedObligation

__all__ = (
    "DerivedFactRuleDefinition",
    "OperationArgument",
    "OperationInvocation",
    "OperatorDefinition",
    "StateAddressPattern",
    "StateDeltaManifest",
    "StateLeafIndex",
    "StateVariableDefinition",
    "StateVariableRef",
    "TypedVariableBound",
    "WriteAuthority",
)


def _require_field_path_ref(value):
    if isinstance(value, str):
        prefix = "field-path:"
        suffix = value.removeprefix(prefix)
        if not value.startswith(prefix) or not suffix:
            raise ValueError("state field paths must use a field-path reference")
        if any(token in suffix for token in (".", "/", "~", "*")):
            raise ValueError("state field paths must be schema-owned references")
    return value


def _require_parameter_ref(value):
    if isinstance(value, str) and not value.startswith("parameter:"):
        raise ValueError("state address parameters must start with 'parameter:'")
    return value


FieldPathRef = Annotated[CanonicalId, BeforeValidator(_require_field_path_ref)]
ParameterRef = Annotated[CanonicalId, BeforeValidator(_require_parameter_ref)]


class WriteAuthority(StrEnum):
    """The permanent, non-extensible M1 write-authority partition."""

    PRIMARY_WRITABLE = "PRIMARY_WRITABLE"
    DERIVED_ONLY = "DERIVED_ONLY"


class StateVariableDefinition(HashBoundCanonicalModel):
    """One definition-level state-variable shape and its write authority."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/state-variable-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "state_variable_definition_sha256"

    state_variable_ref: DefinitionRef
    state_variable_schema_ref: SchemaRef
    value_schema_ref: SchemaRef
    frame_ref: DefinitionRef
    unit_ref: DefinitionRef
    topology_ref: DefinitionRef
    write_authority: WriteAuthority
    derived_fact_rule_ref: DefinitionRef | None = None
    state_variable_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_write_authority(self) -> Self:
        if self.write_authority is WriteAuthority.PRIMARY_WRITABLE:
            if self.derived_fact_rule_ref is not None:
                raise ValueError(
                    "primary writable state variables must not carry a derived rule"
                )
        elif self.derived_fact_rule_ref is None:
            raise ValueError("derived only state variables require exactly one rule")
        return self


class StateVariableRef(CanonicalModel):
    """One schema-owned addressable leaf, never a free-form selector string."""

    state_variable_schema_ref: SchemaRef
    state_schema_ref: SchemaRef
    fact_family_ref: DefinitionRef
    entity_or_fact_key: CanonicalId
    field_path_ref: FieldPathRef


class StateAddressParameterBinding(CanonicalModel):
    """A typed input used to instantiate one state-variable definition pattern."""

    parameter_ref: ParameterRef
    parameter_schema_ref: SchemaRef


class StateAddressPattern(CanonicalModel):
    """A closed state-variable definition plus its typed instantiation inputs."""

    state_variable_definition_ref: DefinitionRef
    parameter_bindings: tuple[StateAddressParameterBinding, ...] = ()

    @model_validator(mode="after")
    def _validate_parameter_bindings(self) -> Self:
        bindings = self.parameter_bindings
        names = tuple(binding.parameter_ref for binding in bindings)
        if names != tuple(sorted(names, key=canonical_json_bytes)):
            raise ValueError("state address parameter bindings must be sorted")
        if len(set(names)) != len(names):
            raise ValueError("state address parameter bindings must not duplicate")
        return self


class StateLeafIndex(HashBoundCanonicalModel):
    """The complete locally canonical index of all addressable scene leaves."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/state-leaf-index/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "state_leaf_index_sha256"

    leaves: tuple[StateVariableRef, ...]
    state_leaf_index_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_leaves(self) -> Self:
        _require_sorted_unique_state_refs(self.leaves, "state leaf index leaves")
        return self


class DerivedFactRuleDefinition(HashBoundCanonicalModel):
    """A pure, fixed-footprint derived-fact recomputation contract."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/derived-fact-rule-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "derived_fact_rule_definition_sha256"

    derived_fact_rule_ref: DefinitionRef
    fixed_read_set: tuple[StateVariableRef, ...]
    fixed_output_set: tuple[StateVariableRef, ...]
    evaluator_capability_ref: CapabilityRef
    verifier_capability_ref: CapabilityRef
    derived_fact_rule_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_fixed_footprints(self) -> Self:
        _require_sorted_unique_state_refs(
            self.fixed_read_set,
            "derived fact rule fixed read set",
        )
        _require_sorted_unique_state_refs(
            self.fixed_output_set,
            "derived fact rule fixed output set",
            nonempty=True,
        )
        if self.evaluator_capability_ref == self.verifier_capability_ref:
            raise ValueError("derived fact rule capability pair must be distinct")
        return self


class TypedVariableBound(CanonicalModel):
    """A typed bound with explicit value, frame, unit, and topology semantics."""

    state_variable_ref: StateVariableRef
    value_schema_ref: SchemaRef
    typed_domain: TypedValue
    frame_ref: DefinitionRef
    unit_ref: DefinitionRef
    topology_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_typed_domain(self) -> Self:
        if self.typed_domain.value_schema_ref != self.value_schema_ref:
            raise ValueError("typed domain must carry the bound value schema")
        return self


class OperatorDefinition(HashBoundCanonicalModel):
    """The complete local transition meaning of one non-native operator."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/operator-definition/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "definition_sha256"

    operator_ref: DefinitionRef
    parameter_schema_refs: tuple[SchemaRef, ...]
    read_footprint: tuple[StateAddressPattern, ...]
    primary_write_footprint: tuple[StateAddressPattern, ...]
    derived_write_rule_refs: tuple[DefinitionRef, ...]
    required_preconditions: tuple[BeforePrecondition, ...]
    transition_semantics_ref: DefinitionRef
    generated_obligations: tuple[GroundedObligation, ...]
    composability_policy_ref: DefinitionRef
    endpoint_or_path_semantics_ref: DefinitionRef
    compiler_capability_ref: CapabilityRef
    verifier_capability_ref: CapabilityRef
    definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_definition_closure(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.read_footprint,
            "operator read footprint",
        )
        _require_sorted_unique_by_bytes(
            self.primary_write_footprint,
            "operator primary write footprint",
        )
        _require_sorted_unique_by_bytes(
            self.derived_write_rule_refs,
            "operator derived write rules",
        )
        _require_sorted_unique_by_bytes(
            self.required_preconditions,
            "operator required preconditions",
        )
        _require_sorted_unique_by_bytes(
            self.generated_obligations,
            "operator generated obligations",
        )
        if self.compiler_capability_ref == self.verifier_capability_ref:
            raise ValueError("operator capability pair must be distinct")
        return self


class OperationArgument(CanonicalModel):
    """One named typed value in an invocation's closed argument tuple."""

    argument_name: CanonicalId
    value: TypedValue


class OperationInvocation(CanonicalModel):
    """One exact operator reference and its canonical named typed arguments."""

    operator_ref: DefinitionRef
    arguments: tuple[OperationArgument, ...]

    @model_validator(mode="after")
    def _validate_arguments(self) -> Self:
        names = tuple(argument.argument_name for argument in self.arguments)
        if names != tuple(sorted(names, key=canonical_json_bytes)):
            raise ValueError("operation arguments must be sorted")
        if len(set(names)) != len(names):
            raise ValueError("operation arguments must not duplicate")
        return self


class StateDeltaManifest(HashBoundCanonicalModel):
    """The local three-way before/after leaf-partition declaration."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/state-delta-manifest/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "state_delta_manifest_sha256"

    authorized_primary_writes: tuple[StateVariableRef, ...]
    recomputed_derived_writes: tuple[StateVariableRef, ...]
    unchanged_leaves_digest: Sha256Digest
    complete_before_leaf_index_sha256: Sha256Digest
    complete_after_leaf_index_sha256: Sha256Digest
    state_delta_manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_local_partition(self) -> Self:
        _require_sorted_unique_state_refs(
            self.authorized_primary_writes,
            "authorized primary writes",
        )
        _require_sorted_unique_state_refs(
            self.recomputed_derived_writes,
            "recomputed derived writes",
        )
        primary_keys = {
            _state_leaf_key(reference) for reference in self.authorized_primary_writes
        }
        derived_keys = {
            _state_leaf_key(reference) for reference in self.recomputed_derived_writes
        }
        if primary_keys & derived_keys:
            raise ValueError("primary and derived writes must not overlap")
        return self


def _state_leaf_key(reference: StateVariableRef) -> bytes:
    """Return the address identity that cannot be claimed by two schemas."""

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
    if nonempty and not references:
        raise ValueError(f"{label} must not be empty")
    encoded = tuple(canonical_json_bytes(reference) for reference in references)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")
    address_keys = tuple(_state_leaf_key(reference) for reference in references)
    if len(set(address_keys)) != len(address_keys):
        raise ValueError(f"{label} must not have two schemas claiming one leaf")


def _require_sorted_unique_by_bytes(
    values,
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
