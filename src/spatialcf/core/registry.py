"""Static, explicit cross-record validation for M1 counterfactual records.

The registry is deliberately a closed composition-time object.  It resolves
only the direct trusted objects supplied in :class:`StaticOwner` rows and never
loads a module, probes a machine, or performs a backend/checker operation.
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

__all__ = (
    "DefinitionClosureError",
    "ImplementationResolutionError",
    "SemanticContractError",
    "StaticImplementationRegistry",
    "StaticOwner",
)

_UNCHANGED_LEAF_DOMAIN = "spatialcf/counterfactual/unchanged-leaves/3.0"
_FROZEN_LEAF_PATH_DOMAIN = "spatialcf/counterfactual/frozen-scene-leaf-path/3.0"
_EXTENSION_FACT_ADDRESS_DOMAIN = "spatialcf/counterfactual/extension-fact-address/3.0"
_ROLE_CERTIFIED_SOLUTION = "definition-kind:claim-certified-solution"
_ROLE_PROVEN_UNSAT = "definition-kind:claim-proven-unsat"
_ROLE_NONCERTIFIED_WITNESS = "definition-kind:claim-noncertified-witness"
_ROLE_UNKNOWN = "definition-kind:claim-unknown"
_ROLE_COMPLETE_DOMAIN = "definition-kind:complete-domain"
_ROLE_SOUND_COMPLETE_DOMAIN = "definition-kind:sound-complete-domain"
_ROLE_PROOF_MATERIAL = "definition-kind:proof-material"
_ROLE_RECORD_BINDING = "definition-kind:record-binding"
_ROLE_RESOURCE_ACCOUNTING = "definition-kind:resource-accounting"
_ROLE_RESOURCE_POLICY = "definition-kind:resource-policy"
_ROLE_ROUTING_POLICY = "definition-kind:routing-policy"
_ROLE_ROUTING_MATCH = "definition-kind:routing-match"
_ROLE_ROUTING_MISMATCH = "definition-kind:routing-mismatch"
_ROLE_ROUTING_SELECTION_DISPOSITION = "definition-kind:routing-selection-disposition"
_ROLE_ROUTING_SELECTION_REASON = "definition-kind:routing-selection-reason"
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
_RECORD_KIND_FIELD = "field:definition-kind"
_RECORD_REFERENCE_FIELD = "field:definition-reference"
_RECORD_BOUND_REFERENCE_FIELD = "field:bound-record-ref"
_RECORD_BOUND_SHA256_FIELD = "field:bound-record-sha256"
_STATE_SCHEMA_FIELD = "field:state-variable-schema"
_STATE_LEAF_SCHEMA_FIELD = "field:state-leaf-schema"
_STATE_FACT_FAMILY_FIELD = "field:state-fact-family"
_STATE_ENTITY_KEY_FIELD = "field:state-entity-key"
_STATE_FIELD_PATH_FIELD = "field:state-field-path"
_STATE_VALUE_SCHEMA_FIELD = "field:state-value-schema"
_STATE_FRAME_FIELD = "field:state-frame"
_STATE_UNIT_FIELD = "field:state-unit"
_STATE_TOPOLOGY_FIELD = "field:state-topology"
_PREREQUISITE_FACT_FAMILY_FIELD = "field:prerequisite-fact-family"
_OBJECTIVE_INPUT_SELECTOR_FIELD = "field:objective-input-selector"
_OBJECTIVE_UNIT_FIELD = "field:objective-unit"
_OBJECTIVE_NORMALIZATION_FIELD = "field:objective-normalization"
_PROOF_PAYLOAD_SCHEMA_FIELD = "field:proof-payload-schema"
_PROOF_CHECKER_CAPABILITY_FIELD = "field:proof-checker-capability"
_CLAIM_PROOF_MATERIAL_FIELD = "field:claim-proof-material-definition"
_CLAIM_CHECKER_CAPABILITY_FIELD = "field:claim-checker-capability"
_COMPLETE_DOMAIN_CLAIM_FIELD = "field:complete-domain-claim"
_ROUTING_MATCH_CLAIM_FIELD = "field:routing-match-claim"
_ROUTING_MISMATCH_CLAIM_FIELD = "field:routing-mismatch-claim"
_RESOURCE_ACCOUNTING_CLAIM_FIELD = "field:resource-accounting-claim"
_STATE_METADATA_FIELDS = (
    _STATE_SCHEMA_FIELD,
    _STATE_LEAF_SCHEMA_FIELD,
    _STATE_FACT_FAMILY_FIELD,
    _STATE_ENTITY_KEY_FIELD,
    _STATE_FIELD_PATH_FIELD,
    _STATE_VALUE_SCHEMA_FIELD,
)
_OWNER_REF_ADAPTER = TypeAdapter(OwnerRef)
_CAPABILITY_REF_ADAPTER = TypeAdapter(CapabilityRef)


class DefinitionClosureError(ValueError):
    """A definition, schema, canonical-byte, or typed-value closure failed."""


class SemanticContractError(ValueError):
    """A cross-record semantic, policy, program, or outcome contract failed."""


class ImplementationResolutionError(ValueError):
    """A supplied static owner, build, or capability binding did not resolve."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _ref_key(value: object) -> bytes:
    return canonical_json_bytes(value)


def _state_key(value: StateVariableRef) -> bytes:
    return canonical_json_bytes(
        (
            value.state_schema_ref,
            value.fact_family_ref,
            value.entity_or_fact_key,
            value.field_path_ref,
        )
    )


def _self_digest_matches(model: HashBoundCanonicalModel) -> bool:
    field_name = model.SELF_DIGEST_FIELD
    payload = model.model_dump(
        mode="python",
        by_alias=True,
        exclude={field_name},
        exclude_none=False,
        exclude_defaults=False,
        exclude_unset=False,
        exclude_computed_fields=True,
        round_trip=True,
        # Registry validators intentionally inspect already-decoded hostile
        # wires.  Pydantic's serializer warning is not a semantic verdict and
        # must not prevent the closure checks below from rejecting the wire.
        warnings=False,
    )
    return getattr(model, field_name) == canonical_sha256(
        payload,
        domain=model.HASH_DOMAIN,
    )


def _walk_values(value: object):
    """Yield closed values recursively without interpreting an open mapping."""

    yield value
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            yield from _walk_values(getattr(value, field_name))
    elif isinstance(value, tuple | list | frozenset | set):
        for item in value:
            yield from _walk_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)


def _references(value: object, prefix: str) -> set[str]:
    return {
        item
        for item in _walk_values(value)
        if isinstance(item, str) and item.startswith(prefix)
    }


def _schema_dependency_refs(schema: ValueSchemaDefinition) -> tuple[str, ...]:
    """Return every schema edge declared by one immutable value schema."""

    refs = [field.value_schema_ref for field in schema.fields]
    for attribute in (
        "unit_schema_ref",
        "dimension_schema_ref",
        "frame_schema_ref",
        "endpoint_schema_ref",
        "element_schema_ref",
    ):
        reference = getattr(schema, attribute)
        if reference is not None:
            refs.append(reference)
    return tuple(refs)


def _semantic_schema_definition_closure(
    semantic_definitions: dict[str, CanonicalDefinitionEnvelope],
    schemas: dict[str, ValueSchemaDefinition],
    root_records: tuple[object, ...],
) -> set[str]:
    """Resolve schema meaning from semantic wires, never from supplied schemas.

    A :class:`ValueSchemaDefinition` is an implementation-side record, not a
    semantic root.  Its own record-binding envelope therefore cannot establish
    why it is present.  Start at the submitted semantic records, follow only
    non-binding semantic definition records, and then close the resulting
    schema graph through each declared schema dependency.
    """

    roles = _definition_roles(semantic_definitions)
    reachable_definitions: set[str] = set()
    schema_refs: set[str] = set()
    for record in root_records:
        reachable_definitions.update(
            reference
            for reference in _references(record, "definition:")
            if reference in semantic_definitions
        )
        schema_refs.update(_references(record, "schema:"))

    while True:
        before = len(reachable_definitions)
        for reference in tuple(reachable_definitions):
            if roles.get(reference) == _ROLE_RECORD_BINDING:
                continue
            definition = semantic_definitions[reference]
            schema_refs.update(_references(definition, "schema:"))
            reachable_definitions.update(
                dependency
                for dependency in (
                    _definition_payload_dependency_refs(definition)
                    | {definition.definition_kind_ref}
                )
                if dependency in semantic_definitions
            )
        for reference, role in roles.items():
            if role not in {_ROLE_COMPLETE_DOMAIN, _ROLE_SOUND_COMPLETE_DOMAIN}:
                continue
            claim_ref = _definition_record_reference(
                semantic_definitions[reference],
                _COMPLETE_DOMAIN_CLAIM_FIELD,
            )
            if claim_ref in reachable_definitions:
                reachable_definitions.add(reference)
        if len(reachable_definitions) == before:
            break

    expected_schema_refs: set[str] = set()
    pending_schema_refs = list(schema_refs)
    while pending_schema_refs:
        reference = pending_schema_refs.pop()
        if reference in expected_schema_refs:
            continue
        schema = schemas.get(reference)
        if schema is None:
            raise DefinitionClosureError("schema reference")
        expected_schema_refs.add(reference)
        pending_schema_refs.extend(_schema_dependency_refs(schema))
    return expected_schema_refs


def _validate_typed_value(
    value: TypedValue,
    schemas: dict[str, ValueSchemaDefinition],
) -> None:
    """Recursively validate a typed wire node against its exact schema graph.

    Pydantic validates the shape of each union member locally.  This helper is
    deliberately the cross-record half: it resolves every schema edge, rejects
    record additions/omissions, and binds dimensional metadata to the declared
    schema instead of trusting an opaque ``value_schema_ref`` label.
    """

    schema = schemas.get(value.value_schema_ref)
    if schema is None:
        raise DefinitionClosureError("typed value schema")
    payload = value.payload
    if schema.value_kind is not payload.kind:
        raise DefinitionClosureError("typed value kind")

    if payload.kind is ValueKind.ENUM_SYMBOL:
        if payload.symbol not in schema.enum_symbols:
            raise DefinitionClosureError("enum symbol")
        return

    if payload.kind is ValueKind.RECORD:
        declared = {field.field_name: field for field in schema.fields}
        supplied = {field.name: field.value for field in payload.fields}
        if set(supplied) - set(declared):
            raise DefinitionClosureError("record extra field")
        missing = {
            name
            for name, field in declared.items()
            if field.required and name not in supplied
        }
        if missing:
            raise DefinitionClosureError("record required field")
        for name, child in supplied.items():
            if child.value_schema_ref != declared[name].value_schema_ref:
                raise DefinitionClosureError("record field schema")
            _validate_typed_value(child, schemas)
        return

    if (
        payload.kind
        in {
            ValueKind.LENGTH,
            ValueKind.AREA,
            ValueKind.ANGLE,
            ValueKind.TIME,
            ValueKind.PIXEL,
            ValueKind.UNIT_INTERVAL,
        }
        and getattr(payload, "unit_schema_ref", None) != schema.unit_schema_ref
    ):
        raise DefinitionClosureError("typed value unit")

    if payload.kind in {
        ValueKind.POINT_2D,
        ValueKind.POINT_3D,
        ValueKind.VECTOR_2D,
        ValueKind.VECTOR_3D,
        ValueKind.RIGID_POSE,
        ValueKind.CLOSED_BOX,
    }:
        if (
            getattr(payload, "dimension_schema_ref", None)
            != schema.dimension_schema_ref
        ):
            raise DefinitionClosureError("typed value dimension")
        if getattr(payload, "frame_schema_ref", None) != schema.frame_schema_ref:
            raise DefinitionClosureError("typed value frame")

    if payload.kind is ValueKind.INTERVAL:
        if (
            payload.endpoint_schema_ref != schema.endpoint_schema_ref
            or payload.lower_closed != schema.lower_closed
            or payload.upper_closed != schema.upper_closed
        ):
            raise DefinitionClosureError("interval schema")
        endpoint = schemas.get(payload.endpoint_schema_ref)
        if endpoint is None:
            raise DefinitionClosureError("schema reference")
        if (
            endpoint.value_kind is not payload.lower.kind
            or endpoint.value_kind is not payload.upper.kind
        ):
            raise DefinitionClosureError("interval endpoint schema")
        # M1 intentionally represents interval endpoints through the closed
        # ScalarPayload union.  Validate each endpoint as its own typed value
        # so enum membership and any future scalar closure are not bypassed by
        # the enclosing interval record.
        for endpoint_payload in (payload.lower, payload.upper):
            _validate_typed_value(
                TypedValue(
                    value_schema_ref=payload.endpoint_schema_ref,
                    payload=endpoint_payload,
                ),
                schemas,
            )
        return

    if payload.kind in {ValueKind.FINITE_SET, ValueKind.FINITE_ORDERED_TUPLE}:
        if payload.element_schema_ref != schema.element_schema_ref:
            raise DefinitionClosureError("sequence element schema")
        items = (
            payload.elements if payload.kind is ValueKind.FINITE_SET else payload.items
        )
        if schema.min_cardinality is not None and len(items) < schema.min_cardinality:
            raise DefinitionClosureError("sequence cardinality")
        if schema.max_cardinality is not None and len(items) > schema.max_cardinality:
            raise DefinitionClosureError("sequence cardinality")
        for item in items:
            if item.value_schema_ref != schema.element_schema_ref:
                raise DefinitionClosureError("sequence element schema")
            _validate_typed_value(item, schemas)


def _definition_record_fields(
    definition: CanonicalDefinitionEnvelope,
) -> dict[str, object]:
    """Return the scalar payloads of one closed typed definition record."""

    payload = definition.payload.payload
    if payload.kind is not ValueKind.RECORD:
        raise DefinitionClosureError("definition record payload")
    return {field.name: field.value.payload for field in payload.fields}


def _definition_record_reference(
    definition: CanonicalDefinitionEnvelope,
    field_name: str,
) -> str:
    """Resolve one explicit canonical-reference field from a typed record."""

    value = _definition_record_fields(definition).get(field_name)
    if value is None or getattr(value, "kind", None) is not ValueKind.CANONICAL_ID:
        raise DefinitionClosureError("definition record metadata")
    return value.value


def _definition_record_digest(
    definition: CanonicalDefinitionEnvelope,
    field_name: str,
) -> str:
    """Resolve one explicit digest field from a typed definition record."""

    value = _definition_record_fields(definition).get(field_name)
    if value is None or getattr(value, "kind", None) is not ValueKind.DIGEST:
        raise DefinitionClosureError("definition record metadata")
    return value.value


def _definition_record_role(
    definition: CanonicalDefinitionEnvelope,
) -> str:
    """Read an explicit typed definition-record role without parsing its ID."""

    fields = _definition_record_fields(definition)
    role_value = fields.get(_RECORD_KIND_FIELD)
    identity_value = fields.get(_RECORD_REFERENCE_FIELD)
    if (
        role_value is None
        or role_value.kind is not ValueKind.ENUM_SYMBOL
        or identity_value is None
        or identity_value.kind is not ValueKind.CANONICAL_ID
        or identity_value.value != definition.definition_ref
    ):
        raise DefinitionClosureError("definition record identity")
    return role_value.symbol


def _definition_roles(
    definitions: dict[str, CanonicalDefinitionEnvelope],
) -> dict[str, str]:
    return {
        reference: _definition_record_role(definition)
        for reference, definition in definitions.items()
    }


def _definition_payload_dependency_refs(
    definition: CanonicalDefinitionEnvelope,
) -> set[str]:
    """Return payload edges while excluding only its required record identity.

    A typed definition record necessarily names its own envelope in
    ``field:definition-reference``.  That one structural identity field is not
    a dependency.  A self reference anywhere else in the payload remains a
    real cycle edge.
    """

    payload = definition.payload.payload
    if payload.kind is not ValueKind.RECORD:
        return _references(definition.payload, "definition:")
    references: set[str] = set()
    skipped_identity = False
    for field in payload.fields:
        field_payload = field.value.payload
        if (
            not skipped_identity
            and field.name == _RECORD_REFERENCE_FIELD
            and field_payload.kind is ValueKind.CANONICAL_ID
            and field_payload.value == definition.definition_ref
        ):
            skipped_identity = True
            continue
        references.update(_references(field.value, "definition:"))
    return references


def _root_bootstrap_anchor(
    definitions: dict[str, CanonicalDefinitionEnvelope],
) -> str | None:
    """Identify the one root-local self-kind anchor used by typed envelopes."""

    kind_counts: dict[str, int] = {}
    for definition in definitions.values():
        kind_counts[definition.definition_kind_ref] = (
            kind_counts.get(definition.definition_kind_ref, 0) + 1
        )
    highest_count = max(kind_counts.values(), default=0)
    candidates = tuple(
        reference for reference, count in kind_counts.items() if count == highest_count
    )
    if len(candidates) != 1:
        return None
    anchor = candidates[0]
    anchor_definition = definitions.get(anchor)
    if anchor_definition is None or anchor_definition.definition_kind_ref != anchor:
        return None
    return anchor


def _has_exact_root_bootstrap_shape(
    definitions: dict[str, CanonicalDefinitionEnvelope],
    anchor: str | None,
) -> bool:
    """Require one self-kind anchor and no alternate envelope kind edge."""

    return anchor is not None and all(
        definition.definition_kind_ref == anchor for definition in definitions.values()
    )


def _definition_root_maps(
    semantic_definition_bundle: DefinitionBundle,
    solve_policy_definition_bundle: DefinitionBundle,
) -> tuple[
    dict[str, CanonicalDefinitionEnvelope],
    dict[str, CanonicalDefinitionEnvelope],
    dict[str, CanonicalDefinitionEnvelope],
]:
    """Return the two explicit roots and their non-overriding union."""

    semantic = {
        definition.definition_ref: definition
        for definition in semantic_definition_bundle.definitions
    }
    solve = {
        definition.definition_ref: definition
        for definition in solve_policy_definition_bundle.definitions
    }
    if semantic.keys() & solve.keys():
        raise DefinitionClosureError("definition bundle overlap")
    return semantic, solve, semantic | solve


def _definition_record_bindings(
    definitions: dict[str, CanonicalDefinitionEnvelope],
    roles: dict[str, str],
) -> dict[str, str]:
    """Read explicit record identity bindings from the frozen definition DAG."""

    bindings: dict[str, str] = {}
    for reference, role in roles.items():
        if role != _ROLE_RECORD_BINDING:
            continue
        definition = definitions[reference]
        bound_reference = _definition_record_reference(
            definition,
            _RECORD_BOUND_REFERENCE_FIELD,
        )
        bound_digest = _definition_record_digest(
            definition,
            _RECORD_BOUND_SHA256_FIELD,
        )
        if bound_reference in bindings:
            raise DefinitionClosureError("duplicate definition record binding")
        bindings[bound_reference] = bound_digest
    return bindings


def _validate_exact_root_definition_closure(
    definitions: dict[str, CanonicalDefinitionEnvelope],
    *,
    root_records: tuple[object, ...],
    record_binding_targets: set[str],
    root_label: str,
    include_complete_domain_dependents: bool = False,
) -> None:
    """Require a bundle to be exactly reachable from explicit wire records.

    A definition bundle is an input container, not a source of its own
    authority.  The traversal starts at the submitted problem/profile/policy
    records and their supplied specialized records; record-binding envelopes
    become reachable only through the record identity they bind.  In
    particular, an otherwise well-formed generic envelope cannot make itself
    reachable merely by being embedded in the bundle.
    """

    roles = _definition_roles(definitions)
    binding_target_by_envelope = {
        reference: _definition_record_reference(
            definitions[reference], _RECORD_BOUND_REFERENCE_FIELD
        )
        for reference, role in roles.items()
        if role == _ROLE_RECORD_BINDING
    }
    explicit_targets = set(record_binding_targets)
    for record in root_records:
        explicit_targets.update(_references(record, "definition:"))
        explicit_targets.update(_references(record, "schema:"))

    anchor = _root_bootstrap_anchor(definitions)
    reachable = {
        reference for reference in explicit_targets if reference in definitions
    }
    if anchor is not None:
        reachable.add(anchor)

    while True:
        before = len(reachable)
        for reference in tuple(reachable):
            definition = definitions[reference]
            reachable.update(
                dependency
                for dependency in (
                    _definition_payload_dependency_refs(definition)
                    | {definition.definition_kind_ref}
                )
                if dependency in definitions
            )
        reachable.update(
            envelope_ref
            for envelope_ref, target in binding_target_by_envelope.items()
            if target in explicit_targets or target in reachable
        )
        if include_complete_domain_dependents:
            for reference, role in roles.items():
                if role not in {_ROLE_COMPLETE_DOMAIN, _ROLE_SOUND_COMPLETE_DOMAIN}:
                    continue
                claim_ref = _definition_record_reference(
                    definitions[reference], _COMPLETE_DOMAIN_CLAIM_FIELD
                )
                if claim_ref in explicit_targets or claim_ref in reachable:
                    reachable.add(reference)
        if len(reachable) == before:
            break
    if set(definitions) != reachable:
        raise DefinitionClosureError(f"unreachable {root_label} definition")


def _require_definition_role(
    roles: dict[str, str],
    reference: str,
    *permitted: str,
) -> None:
    if roles.get(reference) not in permitted:
        raise DefinitionClosureError("definition record role")


def _require_outcome_definition_role(
    roles: dict[str, str],
    reference: str,
    permitted: tuple[str, ...],
    reason: str,
) -> None:
    """Keep valid-but-inadmissible terminal claims in the outcome error family."""

    if roles.get(reference) not in permitted:
        raise SemanticContractError(reason)


def _validate_resource_usage(
    usage: ResourceUsage,
    resource_policy: ResourcePolicy,
    definitions: dict[str, CanonicalDefinitionEnvelope],
) -> None:
    """Bind every submitted accounting row to the frozen resource policy."""

    policy_definition = definitions.get(resource_policy.resource_policy_ref)
    if (
        policy_definition is None
        or _definition_record_role(policy_definition) != _ROLE_RESOURCE_POLICY
    ):
        raise SemanticContractError("resource usage")
    accounting_ref = _definition_record_reference(
        policy_definition,
        _RESOURCE_ACCOUNTING_CLAIM_FIELD,
    )
    if (
        usage.accounting_claim_definition_ref != accounting_ref
        or _definition_roles(definitions).get(accounting_ref)
        != _ROLE_RESOURCE_ACCOUNTING
        or {entry.resource_definition_ref for entry in usage.entries}
        != {limit.definition_ref for limit in resource_policy.limits}
    ):
        raise SemanticContractError("resource usage")
    limits_by_ref = {
        limit.definition_ref: limit.finite_limit for limit in resource_policy.limits
    }
    if any(
        entry.used > limits_by_ref[entry.resource_definition_ref]
        for entry in usage.entries
    ):
        raise SemanticContractError("resource usage limit")
    exhausted = any(
        entry.used == limits_by_ref[entry.resource_definition_ref]
        for entry in usage.entries
    )
    if usage.exhausted != exhausted:
        raise SemanticContractError("resource usage exhaustion")


def _state_definition_metadata(
    definitions: dict[str, CanonicalDefinitionEnvelope],
) -> dict[str, tuple[str, str, str, str, str, str]]:
    """Resolve schema-owned leaf addresses from typed state definition records."""

    result: dict[str, tuple[str, str, str, str, str]] = {}
    for definition_ref, definition in definitions.items():
        if _definition_record_role(definition) != "definition-kind:state-variable":
            continue
        payload = definition.payload.payload
        assert payload.kind is ValueKind.RECORD
        fields = {field.name: field.value.payload for field in payload.fields}
        values: list[str] = []
        for field_name in _STATE_METADATA_FIELDS:
            field = fields.get(field_name)
            if field is None or field.kind is not ValueKind.CANONICAL_ID:
                # Ordinary state definitions may remain shape-only, but they
                # cannot own a submitted scene leaf until metadata is present.
                break
            values.append(field.value)
        else:
            result[definition_ref] = tuple(values)  # type: ignore[assignment]
    return result


def _state_definition_semantics(
    definitions: dict[str, CanonicalDefinitionEnvelope],
) -> dict[str, tuple[str, str, str]]:
    """Resolve the non-address state meaning carried by its typed record."""

    result: dict[str, tuple[str, str, str]] = {}
    for definition_ref, definition in definitions.items():
        if _definition_record_role(definition) != "definition-kind:state-variable":
            continue
        fields = _definition_record_fields(definition)
        values: list[str] = []
        for field_name in (
            _STATE_FRAME_FIELD,
            _STATE_UNIT_FIELD,
            _STATE_TOPOLOGY_FIELD,
        ):
            field = fields.get(field_name)
            if (
                field is None
                or getattr(field, "kind", None) is not ValueKind.CANONICAL_ID
            ):
                break
            values.append(field.value)
        else:
            result[definition_ref] = tuple(values)  # type: ignore[assignment]
    return result


def _raw_value_matches_schema(value: object, schema: ValueSchemaDefinition) -> bool:
    """Check the frozen raw scene scalar against its declared leaf schema."""

    if value is None:
        return True
    if schema.value_kind is ValueKind.BOOLEAN:
        return isinstance(value, bool)
    if schema.value_kind is ValueKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if schema.value_kind is ValueKind.FINITE_REAL:
        return isinstance(value, float)
    if schema.value_kind is ValueKind.CANONICAL_ID:
        return isinstance(value, str)
    return False


def _validate_state_leaf_value_wires(
    leaves: tuple[StateVariableRef, ...],
    values: dict[bytes, object],
    definitions_by_leaf: dict[bytes, StateVariableDefinition],
    schemas: dict[str, ValueSchemaDefinition],
    actual_extension_addresses: set[tuple[str, str]],
    absent_extension_addresses: set[tuple[str, str]],
) -> None:
    """Validate values by their exact base/extension ownership category.

    Frozen base leaves encode their canonical scene scalars directly.  An
    actually present extension fact, in contrast, must carry a ``TypedValue``
    on each non-presence leaf.  A schema-declared but absent extension address
    is the only place where the explicit ``None`` value sentinel is legal;
    its presence leaf remains a raw boolean.  Keeping this distinction here
    makes the before, semantic, and after closures agree on the same wire.
    """

    for leaf in leaves:
        key = _state_key(leaf)
        state_definition = definitions_by_leaf.get(key)
        if state_definition is None or key not in values:
            raise DefinitionClosureError("state leaf value schema")
        value = values[key]
        owner = (leaf.fact_family_ref, leaf.entity_or_fact_key)
        presence_leaf = leaf.field_path_ref.startswith("field-path:presence-")

        if owner in actual_extension_addresses and not presence_leaf:
            if not isinstance(value, TypedValue):
                raise DefinitionClosureError("state leaf value schema")
            if value.value_schema_ref != state_definition.value_schema_ref:
                raise DefinitionClosureError("state leaf value schema")
            _validate_typed_value(value, schemas)
            continue

        if owner in absent_extension_addresses and not presence_leaf:
            if value is None:
                continue
            raise DefinitionClosureError("state leaf value schema")

        if isinstance(value, TypedValue):
            if value.value_schema_ref != state_definition.value_schema_ref:
                raise DefinitionClosureError("state leaf value schema")
            _validate_typed_value(value, schemas)
            continue

        schema = schemas.get(state_definition.value_schema_ref)
        if schema is None or not _raw_value_matches_schema(value, schema):
            raise DefinitionClosureError("state leaf value schema")


def _base_fact_rows(scene: object) -> tuple[tuple[str, str, object], ...]:
    """Read the finite current scene as data, never through a runtime adapter."""

    families = (
        (
            "definition:spatialcf/counterfactual/base-scene/objects/3.0",
            "objects",
            "object_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/geometry-instances/3.0",
            "geometry_instances",
            "geometry_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/collision-bodies/3.0",
            "collision_bodies",
            "body_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/workspace-boundaries/3.0",
            "workspace_boundaries",
            "fact_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/known-free-spaces/3.0",
            "known_free_spaces",
            "fact_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/support-surfaces/3.0",
            "support_surfaces",
            "surface_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/cameras/3.0",
            "cameras",
            "camera_id",
        ),
        (
            "definition:spatialcf/counterfactual/base-scene/baseline-observations/3.0",
            "baseline_observations",
            "observation_id",
        ),
    )
    rows: list[tuple[str, str, object]] = []
    for family, attribute, identifier in families:
        fact_set = getattr(scene, attribute)
        for values in (fact_set.values, fact_set.inner_values, fact_set.outer_values):
            if values:
                rows.extend(
                    (family, getattr(item, identifier), item) for item in values
                )
    return tuple(rows)


def _frozen_scalar_values(
    value: object,
    path: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, ...], object], ...]:
    """Flatten canonical facts into their actual scalar field leaves."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return tuple(
            item
            for name, child in value.items()
            for item in _frozen_scalar_values(child, (*path, str(name)))
        )
    if isinstance(value, tuple | list):
        return tuple(
            item
            for index, child in enumerate(value)
            for item in _frozen_scalar_values(child, (*path, str(index)))
        )
    if hasattr(value, "value") and type(value).__module__ != "builtins":
        value = value.value
    return ((path, value),)


def _frozen_leaf_token(
    fact_family_ref: str,
    entity_or_fact_key: str,
    path: tuple[str, ...],
) -> str:
    return canonical_sha256(
        (fact_family_ref, entity_or_fact_key, path),
        domain=_FROZEN_LEAF_PATH_DOMAIN,
    )


def _extension_fact_address(
    fact_family_ref: str,
    subject_entity_id: str,
    fact_key: str,
) -> str:
    """Derive the one state address for a full extension ownership triple.

    ``StateVariableRef`` intentionally has one canonical address field rather
    than an open selector map.  Extension facts therefore encode their full
    ``(family, subject, key)`` ownership tuple into that field.  A fact key
    alone is not an ownership identity because the same key may occur for two
    independently editable subjects.
    """

    return "fact-address:spatialcf/counterfactual/" + canonical_sha256(
        (fact_family_ref, subject_entity_id, fact_key),
        domain=_EXTENSION_FACT_ADDRESS_DOMAIN,
    )


def _extension_facts_by_address(
    state: SceneStateEnvelope,
) -> dict[tuple[str, str], object]:
    """Index extension facts by their derived address and reject a hash alias."""

    facts: dict[tuple[str, str], object] = {}
    ownership_by_address: dict[tuple[str, str], tuple[str, str, str]] = {}
    for bundle in state.extension_fact_bundles:
        for fact in bundle.facts:
            address = _extension_fact_address(
                fact.fact_family_ref,
                fact.subject_entity_id,
                fact.fact_key,
            )
            key = (fact.fact_family_ref, address)
            ownership = (
                fact.fact_family_ref,
                fact.subject_entity_id,
                fact.fact_key,
            )
            previous = ownership_by_address.get(key)
            if previous is not None and previous != ownership:
                raise SemanticContractError("extension fact address")
            if previous is not None:
                raise SemanticContractError("extension fact address")
            ownership_by_address[key] = ownership
            facts[key] = fact
    return facts


def _state_leaf_owners(
    leaves: tuple[StateVariableRef, ...],
) -> dict[tuple[str, str], tuple[StateVariableRef, ...]]:
    """Group the frozen state-index leaves by their complete fact address."""

    owners: dict[tuple[str, str], tuple[StateVariableRef, ...]] = {}
    for leaf in leaves:
        address = (leaf.fact_family_ref, leaf.entity_or_fact_key)
        owners[address] = owners.get(address, ()) + (leaf,)
    return owners


def _declared_extension_leaf_owners(
    leaves: tuple[StateVariableRef, ...],
    base_owners: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[StateVariableRef, ...]]:
    """Return schema-declared non-base fact addresses from the frozen index.

    A state schema owns the full extension address universe.  A particular
    scene can legitimately omit one of those facts, in which case its presence
    leaf is false and its value leaf carries the explicit ``None`` absence
    sentinel.  The scene facts may therefore be a strict subset of this map;
    facts outside it are never admitted.
    """

    return {
        address: owned
        for address, owned in _state_leaf_owners(leaves).items()
        if address not in base_owners
    }


def _expected_base_leaf_addresses(scene: object) -> set[tuple[str, str, str]]:
    """Return the finite frozen base-scene leaf address universe.

    Canonical v2 fact identities are closed data.  M1 adds an explicit
    presence and value leaf for each fact; treating the index as a merely
    non-empty collection would permit hidden state to alter a hash partition.
    """

    addresses: set[tuple[str, str, str]] = set()
    for family, identifier, fact in _base_fact_rows(scene):
        membership_token = _frozen_leaf_token(
            family,
            identifier,
            ("membership",),
        )
        addresses.add((family, identifier, f"field-path:presence-{membership_token}"))
        addresses.update(
            (
                family,
                identifier,
                f"field-path:{_frozen_leaf_token(family, identifier, path)}",
            )
            for path, _value in _frozen_scalar_values(fact)
        )
    return addresses


def _validated_base_scene_payload(payload: object) -> CanonicalScene:
    """Return the exact, independently revalidated frozen base-scene contract.

    ``SceneStateEnvelope.model_construct`` can bypass the domain field
    validator.  A lookalike Pydantic model may serialize to the same bytes as
    a scene while carrying a weaker schema, so byte equality alone is not a
    substitute for the exact CanonicalScene type and its full validator.
    """

    if type(payload) is not CanonicalScene:
        raise ValueError("base scene payload must be an exact CanonicalScene")
    round_trip = CanonicalScene.model_validate(
        payload.model_dump(
            mode="python",
            by_alias=True,
            exclude_none=False,
            exclude_defaults=False,
            exclude_unset=False,
            exclude_computed_fields=True,
            round_trip=True,
        ),
        strict=True,
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(round_trip):
        raise ValueError("base scene payload canonical round trip")
    return round_trip


def _leaf_values(
    state: SceneStateEnvelope,
    base_scene: CanonicalScene,
) -> dict[bytes, object]:
    base_values = {
        (family, identifier, _frozen_leaf_token(family, identifier, path)): value
        for family, identifier, fact in _base_fact_rows(base_scene)
        for path, value in _frozen_scalar_values(fact)
    }
    base_owners = {
        (family, identifier)
        for family, identifier, _fact in _base_fact_rows(base_scene)
    }
    extension_values = {
        address: fact.value
        for address, fact in _extension_facts_by_address(state).items()
    }
    values: dict[bytes, object] = {}
    for leaf in state.canonical_state_leaf_index.leaves:
        key = _state_key(leaf)
        owner = (leaf.fact_family_ref, leaf.entity_or_fact_key)
        token = leaf.field_path_ref.removeprefix("field-path:presence-").removeprefix(
            "field-path:"
        )
        present = owner in base_owners or owner in extension_values
        if leaf.field_path_ref.startswith("field-path:presence-"):
            values[key] = present
        elif owner in extension_values:
            values[key] = extension_values[owner]
        elif (
            leaf.fact_family_ref,
            leaf.entity_or_fact_key,
            token,
        ) in base_values:
            values[key] = base_values[
                (leaf.fact_family_ref, leaf.entity_or_fact_key, token)
            ]
        else:
            # ``None`` is an explicit absence payload in the partition, never a
            # silently fabricated presence bit.  Its canonical encoding makes
            # extension removals observable to the delta reconstruction.
            values[key] = None
    return values


def _validate_scene_state_envelope(
    state: SceneStateEnvelope,
    state_variable_definitions: tuple[StateVariableDefinition, ...],
    value_schema_definitions: tuple[ValueSchemaDefinition, ...],
    semantic_definitions: dict[str, CanonicalDefinitionEnvelope],
    *,
    after_state: bool,
) -> tuple[dict[bytes, StateVariableRef], dict[bytes, object], CanonicalScene]:
    """Close one complete scene envelope without treating an index as opaque.

    The problem's frozen before scene and an edit program's after scene use the
    same address universe and typed state definitions.  They differ in values,
    not in whether a nested bundle, base payload, entity index, or leaf index
    is independently hash-bound and schema-owned.
    """

    def fail(detail: str) -> None:
        raise SemanticContractError("complete after state" if after_state else detail)

    if not _self_digest_matches(state):
        fail("scene state hash")
    if not _self_digest_matches(state.canonical_state_leaf_index):
        fail("state leaf index hash")
    if any(not _self_digest_matches(bundle) for bundle in state.extension_fact_bundles):
        fail("extension fact bundle hash")
    try:
        base_scene = _validated_base_scene_payload(state.base_scene_payload)
    except (TypeError, ValidationError, ValueError) as error:
        raise SemanticContractError(
            "complete after state" if after_state else "base scene hash"
        ) from error
    if state.base_scene_sha256 != canonical_sha256(
        base_scene,
        domain="spatialcf/counterfactual/base-scene-payload/3.0",
    ):
        fail("base scene hash")

    expected_entities = {
        identifier for _family, identifier, _value in _base_fact_rows(base_scene)
    } | {
        fact.subject_entity_id
        for bundle in state.extension_fact_bundles
        for fact in bundle.facts
    }
    if set(state.closed_entity_index) != expected_entities:
        fail("complete scene entity index")

    leaves = state.canonical_state_leaf_index.leaves
    leaf_keys = {_state_key(leaf) for leaf in leaves}
    if len(leaf_keys) != len(leaves) or tuple(leaves) != tuple(
        sorted(leaves, key=canonical_json_bytes)
    ):
        fail("exact state leaf index")
    base_owners = {
        (family, identifier)
        for family, identifier, _fact in _base_fact_rows(base_scene)
    }
    extension_facts = _extension_facts_by_address(state)
    base_addresses = {
        (leaf.fact_family_ref, leaf.entity_or_fact_key, leaf.field_path_ref)
        for leaf in leaves
        if (leaf.fact_family_ref, leaf.entity_or_fact_key) in base_owners
    }
    if base_addresses != _expected_base_leaf_addresses(base_scene):
        fail("exact state leaf index")
    by_family_identifier = _state_leaf_owners(leaves)
    for family, identifier, _value in _base_fact_rows(base_scene):
        owned = by_family_identifier.get((family, identifier), ())
        if not any(
            leaf.field_path_ref.startswith("field-path:presence-") for leaf in owned
        ):
            fail("complete state leaf presence")
        if not any(
            not leaf.field_path_ref.startswith("field-path:presence-") for leaf in owned
        ):
            fail("complete state leaf value")

    state_by_schema = {
        definition.state_variable_schema_ref: definition
        for definition in state_variable_definitions
    }
    state_by_ref = {
        definition.state_variable_ref: definition
        for definition in state_variable_definitions
    }
    if len(state_by_schema) != len(state_variable_definitions) or len(
        state_by_ref
    ) != len(state_variable_definitions):
        fail("state leaf definition")
    if any(leaf.state_variable_schema_ref not in state_by_schema for leaf in leaves):
        fail("state leaf definition")
    if not leaf_keys:
        fail("complete state leaf index")

    metadata_by_ref = _state_definition_metadata(semantic_definitions)
    if set(metadata_by_ref) != set(state_by_ref):
        fail("schema-owned field path")
    metadata_by_address: dict[
        tuple[str, str, str, str, str], StateVariableDefinition
    ] = {}
    for definition_ref, state_definition in state_by_ref.items():
        (
            state_variable_schema_ref,
            state_schema_ref,
            fact_family_ref,
            entity_or_fact_key,
            field_path_ref,
            value_schema_ref,
        ) = metadata_by_ref[definition_ref]
        if (
            state_variable_schema_ref != state_definition.state_variable_schema_ref
            or value_schema_ref != state_definition.value_schema_ref
        ):
            fail("schema-owned field path")
        address = (
            state_variable_schema_ref,
            state_schema_ref,
            fact_family_ref,
            entity_or_fact_key,
            field_path_ref,
        )
        if address in metadata_by_address:
            fail("schema-owned field path")
        metadata_by_address[address] = state_definition
    leaf_by_address = {
        (
            leaf.state_variable_schema_ref,
            leaf.state_schema_ref,
            leaf.fact_family_ref,
            leaf.entity_or_fact_key,
            leaf.field_path_ref,
        ): leaf
        for leaf in leaves
    }
    if len(leaf_by_address) != len(leaves) or set(leaf_by_address) != set(
        metadata_by_address
    ):
        fail("schema-owned field path")
    extension_leaf_owners = _declared_extension_leaf_owners(leaves, base_owners)
    if not set(extension_facts) <= set(extension_leaf_owners):
        fail("extension fact address")
    for owned in extension_leaf_owners.values():
        if (
            len(owned) != 2
            or sum(
                leaf.field_path_ref.startswith("field-path:presence-") for leaf in owned
            )
            != 1
        ):
            fail("complete extension state leaf")
    absent_extension_addresses = set(extension_leaf_owners) - set(extension_facts)
    semantics_by_ref = _state_definition_semantics(semantic_definitions)
    if set(semantics_by_ref) != set(state_by_ref):
        fail("state leaf semantics")
    for definition_ref, state_definition in state_by_ref.items():
        if semantics_by_ref[definition_ref] != (
            state_definition.frame_ref,
            state_definition.unit_ref,
            state_definition.topology_ref,
        ):
            fail("state leaf semantics")

    schemas = {schema.value_schema_ref: schema for schema in value_schema_definitions}
    values = _leaf_values(state, base_scene)
    try:
        _validate_state_leaf_value_wires(
            leaves,
            values,
            {
                _state_key(leaf): metadata_by_address[address]
                for address, leaf in leaf_by_address.items()
            },
            schemas,
            set(extension_facts),
            absent_extension_addresses,
        )
    except DefinitionClosureError:
        fail("state leaf value schema")
    return ({_state_key(leaf): leaf for leaf in leaves}, values, base_scene)


def _formula_predicate_refs(value: object) -> set[str]:
    return {
        item.predicate_ref
        for item in _walk_values(value)
        if isinstance(item, PredicateAtom)
    }


def _formula_atoms(value: object) -> tuple[PredicateAtom, ...]:
    """Return every atom in a closed formula tree without reinterpreting IDs."""

    return tuple(
        item for item in _walk_values(value) if isinstance(item, PredicateAtom)
    )


def _formula_is_fully_grounded(formula: object) -> bool:
    """Require a semantic root to carry the finite grounded formula, not a template."""

    if type(formula) is bool:
        return True
    ground = getattr(formula, "ground", None)
    if ground is None:
        return False
    try:
        return canonical_json_bytes(formula) == canonical_json_bytes(ground())
    except ValueError:
        return False


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


@dataclass(frozen=True)
class StaticOwner:
    """One immutable trusted composition row; its object never enters a wire."""

    owner_ref: str
    implementation_build_sha256: str
    capability_refs: tuple[str, ...]
    implementation: object

    def __post_init__(self) -> None:
        try:
            _OWNER_REF_ADAPTER.validate_python(self.owner_ref, strict=True)
        except ValidationError:
            raise ValueError("static owner ref must be canonical")
        if not _is_sha256(self.implementation_build_sha256):
            raise ValueError("static owner build must be an exact sha256")
        if not isinstance(self.capability_refs, tuple):
            raise TypeError("static owner capabilities must be an immutable tuple")
        try:
            validated_capabilities = tuple(
                _CAPABILITY_REF_ADAPTER.validate_python(capability, strict=True)
                for capability in self.capability_refs
            )
        except ValidationError:
            raise ValueError("static owner capabilities must be canonical")
        if any("*" in capability for capability in validated_capabilities):
            raise ValueError("static owner capabilities must be canonical")
        if not validated_capabilities:
            raise ValueError("static owner capabilities must be canonical")
        canonical = tuple(sorted(validated_capabilities, key=_ref_key))
        if validated_capabilities != canonical or len(
            set(validated_capabilities)
        ) != len(validated_capabilities):
            raise ValueError("static owner capabilities must be sorted and unique")
        try:
            hash(self.implementation)
        except TypeError as error:
            raise TypeError("unhashable implementation") from error


@dataclass(frozen=True)
class StaticImplementationRegistry:
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

    def _validate_snapshot(
        self,
        snapshot: ImplementationRegistrySnapshot,
        required_capabilities: set[str],
        specialized_records: tuple[object, ...] = (),
    ) -> None:
        if not _self_digest_matches(snapshot):
            raise ImplementationResolutionError("registry snapshot hash")
        owners = self._owners_by_ref()
        capabilities = self._owners_by_capability()
        build_by_owner = dict(snapshot.implementation_build_hashes)
        binding_by_ref = {
            binding.definition_or_capability_ref: binding.implementation_owner_ref
            for binding in snapshot.definition_and_capability_owner_bindings
        }
        if len(binding_by_ref) != len(
            snapshot.definition_and_capability_owner_bindings
        ):
            raise ImplementationResolutionError("registry snapshot binding")
        if not required_capabilities <= capabilities.keys():
            raise ImplementationResolutionError("capability owner")
        if any(
            not reference.startswith(("capability:", "definition:"))
            for reference in binding_by_ref
        ):
            raise ImplementationResolutionError("registry snapshot binding")
        capability_binding_by_ref = {
            reference: owner_ref
            for reference, owner_ref in binding_by_ref.items()
            if reference.startswith("capability:")
        }
        definition_binding_by_ref = {
            reference: owner_ref
            for reference, owner_ref in binding_by_ref.items()
            if reference.startswith("definition:")
        }
        static_capability_refs = set(capabilities)
        supplied_capability_refs = set(capability_binding_by_ref)
        if supplied_capability_refs - static_capability_refs:
            raise ImplementationResolutionError("capability owner")
        if static_capability_refs - supplied_capability_refs:
            raise ImplementationResolutionError("registry snapshot binding")
        static_owner_refs = set(owners)
        supplied_owner_refs = set(build_by_owner)
        if supplied_owner_refs != static_owner_refs:
            raise ImplementationResolutionError("registry snapshot build")
        for capability, static in capabilities.items():
            if capability_binding_by_ref[capability] != static.owner_ref:
                raise ImplementationResolutionError("capability owner")
        for owner_ref, static in owners.items():
            if build_by_owner[owner_ref] != static.implementation_build_sha256:
                raise ImplementationResolutionError("registry snapshot build")

        definition_capabilities: dict[str, tuple[str, ...]] = {}
        for record in specialized_records:
            if isinstance(record, PredicateDefinition):
                reference = record.predicate_ref
                record_capabilities = (
                    record.evaluator_capability_ref,
                    record.verifier_capability_ref,
                )
            elif isinstance(record, DerivedFactRuleDefinition):
                reference = record.derived_fact_rule_ref
                record_capabilities = (
                    record.evaluator_capability_ref,
                    record.verifier_capability_ref,
                )
            elif isinstance(record, OperatorDefinition):
                reference = record.operator_ref
                record_capabilities = (
                    record.compiler_capability_ref,
                    record.verifier_capability_ref,
                )
            else:
                continue
            previous = definition_capabilities.get(reference)
            if previous is not None and previous != record_capabilities:
                raise ImplementationResolutionError("registry snapshot binding")
            definition_capabilities[reference] = record_capabilities
        for reference, owner_ref in definition_binding_by_ref.items():
            record_capabilities = definition_capabilities.get(reference)
            if not record_capabilities:
                raise ImplementationResolutionError("registry snapshot binding")
            derived_owners = tuple(
                capabilities.get(capability) for capability in record_capabilities
            )
            if (
                any(owner is None for owner in derived_owners)
                or len(
                    {owner.owner_ref for owner in derived_owners if owner is not None}
                )
                != 1
            ):
                raise ImplementationResolutionError("registry snapshot binding")
            derived_owner = next(owner for owner in derived_owners if owner is not None)
            if owner_ref != derived_owner.owner_ref:
                raise ImplementationResolutionError("registry snapshot binding")

    def _validate_definition_hashes(
        self,
        bundle: DefinitionBundle,
    ) -> None:
        for definition in bundle.definitions:
            if not _self_digest_matches(definition):
                raise DefinitionClosureError("definition hash")
        if not _self_digest_matches(bundle):
            raise DefinitionClosureError("definition bundle hash")

    def _validate_definition_cycles(
        self,
        semantic_definitions: dict[str, CanonicalDefinitionEnvelope],
        solve_definitions: dict[str, CanonicalDefinitionEnvelope],
    ) -> None:
        """Reject every definition cycle except one typed anchor per root.

        The public backend matcher owns matching.  The registry owns this
        structural closure and keeps the semantic and operational roots
        separate while constructing the combined dependency graph.
        """

        definitions = semantic_definitions | solve_definitions
        semantic_anchor = _root_bootstrap_anchor(semantic_definitions)
        solve_anchor = _root_bootstrap_anchor(solve_definitions)
        anchors = {semantic_anchor, solve_anchor}
        graph: dict[str, list[str]] = {}
        for reference, definition in definitions.items():
            dependencies = _definition_payload_dependency_refs(definition)
            # An envelope's kind reference is a dependency too.  The only
            # permitted self edge is the unique root-local bootstrap anchor;
            # payload self references are deliberately never discarded here.
            if not (
                definition.definition_kind_ref == reference and reference in anchors
            ):
                dependencies.add(definition.definition_kind_ref)
            graph[reference] = sorted(
                dependencies & definitions.keys(),
                key=_ref_key,
            )
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(reference: str) -> None:
            if reference in visiting:
                start = visiting.index(reference)
                path = visiting[start:] + [reference]
                raise DefinitionClosureError("stable cycle path: " + " -> ".join(path))
            if reference in visited:
                return
            visiting.append(reference)
            for dependency in graph[reference]:
                visit(dependency)
            visiting.pop()
            visited.add(reference)

        for reference in sorted(graph, key=_ref_key):
            visit(reference)
        if not (
            _has_exact_root_bootstrap_shape(semantic_definitions, semantic_anchor)
            and _has_exact_root_bootstrap_shape(solve_definitions, solve_anchor)
        ):
            raise DefinitionClosureError("root bootstrap anchor")

    def validate_definition_closure(
        self,
        semantic_definition_bundle: DefinitionBundle,
        solve_policy_definition_bundle: DefinitionBundle,
        value_schema_definitions: tuple[ValueSchemaDefinition, ...],
        predicate_definitions: tuple[PredicateDefinition, ...],
        state_variable_definitions: tuple[StateVariableDefinition, ...],
        derived_fact_rule_definitions: tuple[DerivedFactRuleDefinition, ...],
        operator_definitions: tuple[OperatorDefinition, ...],
        implementation_registry_snapshot: ImplementationRegistrySnapshot,
    ) -> DefinitionBundle:
        """Validate finite canonical definitions and their static implementation edges."""

        if (
            semantic_definition_bundle.bootstrap_schema_sha256
            != BOOTSTRAP_SCHEMA_SHA256
            or solve_policy_definition_bundle.bootstrap_schema_sha256
            != BOOTSTRAP_SCHEMA_SHA256
        ):
            raise DefinitionClosureError("bootstrap schema")
        self._validate_definition_hashes(semantic_definition_bundle)
        self._validate_definition_hashes(solve_policy_definition_bundle)
        semantic_definitions, solve_definitions, definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        if any(
            not _references(definition, "definition:") <= semantic_definitions.keys()
            for definition in semantic_definitions.values()
        ):
            raise DefinitionClosureError("semantic definition root")
        self._validate_definition_cycles(semantic_definitions, solve_definitions)

        schemas: dict[str, ValueSchemaDefinition] = {}
        for schema in value_schema_definitions:
            if not _self_digest_matches(schema):
                raise DefinitionClosureError("value schema hash")
            previous = schemas.get(schema.value_schema_ref)
            if previous is not None:
                if canonical_json_bytes(previous) != canonical_json_bytes(schema):
                    raise DefinitionClosureError(
                        "same value schema identifier has different bytes"
                    )
                raise DefinitionClosureError("duplicate value schema identifier")
            schemas[schema.value_schema_ref] = schema
        for schema in schemas.values():
            for reference in _schema_dependency_refs(schema):
                if reference not in schemas:
                    raise DefinitionClosureError("schema reference")

        specialized_records = (
            *((record, record.predicate_ref) for record in predicate_definitions),
            *(
                (record, record.state_variable_ref)
                for record in state_variable_definitions
            ),
            *(
                (record, record.derived_fact_rule_ref)
                for record in derived_fact_rule_definitions
            ),
            *((record, record.operator_ref) for record in operator_definitions),
        )
        specialized_by_ref: dict[str, HashBoundCanonicalModel] = {}
        for record, reference in specialized_records:
            if not _self_digest_matches(record):
                raise DefinitionClosureError("specialized definition hash")
            previous = specialized_by_ref.get(reference)
            if previous is not None:
                if canonical_json_bytes(previous) != canonical_json_bytes(record):
                    raise DefinitionClosureError(
                        "same specialized definition identifier has different bytes"
                    )
                raise DefinitionClosureError("duplicate specialized definition")
            specialized_by_ref[reference] = record

        records: tuple[object, ...] = (
            semantic_definition_bundle,
            solve_policy_definition_bundle,
            *predicate_definitions,
            *state_variable_definitions,
            *derived_fact_rule_definitions,
            *operator_definitions,
        )
        for record in records:
            for value in _walk_values(record):
                if isinstance(value, TypedValue):
                    _validate_typed_value(value, schemas)
                if (
                    isinstance(value, str)
                    and value.startswith("schema:")
                    and value not in schemas
                ):
                    raise DefinitionClosureError("typed value schema")

        all_definition_refs = set()
        required_capabilities = set()
        for record in records:
            all_definition_refs.update(_references(record, "definition:"))
            required_capabilities.update(_references(record, "capability:"))
        dangling = all_definition_refs - definitions.keys()
        if dangling:
            raise DefinitionClosureError("dangling definition reference")
        roles = _definition_roles(definitions)
        semantic_roles = _definition_roles(semantic_definitions)
        solve_roles = _definition_roles(solve_definitions)
        if any(role == _ROLE_RECORD_BINDING for role in solve_roles.values()):
            raise DefinitionClosureError("definition record binding")
        record_bindings = _definition_record_bindings(
            semantic_definitions,
            semantic_roles,
        )
        for reference, record in (
            *schemas.items(),
            *specialized_by_ref.items(),
        ):
            if record_bindings.get(reference) != getattr(
                record,
                record.SELF_DIGEST_FIELD,
            ):
                raise DefinitionClosureError("definition record binding")
        for reference, role in roles.items():
            definition = definitions[reference]
            if role == _ROLE_PROOF_MATERIAL:
                payload_schema_ref = _definition_record_reference(
                    definition,
                    _PROOF_PAYLOAD_SCHEMA_FIELD,
                )
                if payload_schema_ref not in schemas:
                    raise DefinitionClosureError("proof material schema")
                _definition_record_reference(
                    definition,
                    _PROOF_CHECKER_CAPABILITY_FIELD,
                )
            elif role in {_ROLE_CERTIFIED_SOLUTION, _ROLE_PROVEN_UNSAT}:
                proof_definition_ref = _definition_record_reference(
                    definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                if (
                    reference not in semantic_definitions
                    or semantic_roles.get(proof_definition_ref) != _ROLE_PROOF_MATERIAL
                ):
                    raise DefinitionClosureError("claim proof material")
                _definition_record_reference(
                    definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
            elif role in {_ROLE_COMPLETE_DOMAIN, _ROLE_SOUND_COMPLETE_DOMAIN}:
                claim_ref = _definition_record_reference(
                    definition,
                    _COMPLETE_DOMAIN_CLAIM_FIELD,
                )
                if (
                    reference not in semantic_definitions
                    or semantic_roles.get(claim_ref) != _ROLE_PROVEN_UNSAT
                ):
                    raise DefinitionClosureError("complete-domain claim")
        # The typed envelope is the semantic source of truth for specialized
        # records.  Matching by a textual reference suffix would make a hostile
        # wire indistinguishable from a legitimate extension.
        for definition in predicate_definitions:
            _require_definition_role(
                semantic_roles,
                definition.predicate_ref,
                "definition-kind:predicate",
            )
        for definition in state_variable_definitions:
            _require_definition_role(
                semantic_roles,
                definition.state_variable_ref,
                "definition-kind:state-variable",
            )
        for definition in derived_fact_rule_definitions:
            _require_definition_role(
                semantic_roles,
                definition.derived_fact_rule_ref,
                "definition-kind:derived-rule",
            )
        for definition in operator_definitions:
            _require_definition_role(
                semantic_roles,
                definition.operator_ref,
                "definition-kind:operator",
            )
        self._validate_snapshot(
            implementation_registry_snapshot,
            required_capabilities,
            (
                *predicate_definitions,
                *state_variable_definitions,
                *derived_fact_rule_definitions,
                *operator_definitions,
            ),
        )
        return semantic_definition_bundle

    def validate_semantic_problem(
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
    ) -> CounterfactualProblemIR:
        """Validate problem/profile/scene/action closure on explicit input records."""

        # Specialized records are not roots merely because a caller supplied
        # them alongside a problem.  The profiles and request authorization
        # choose the finite semantic universe; accepting an extra record here
        # would let an otherwise unreachable envelope and binding make itself
        # reachable in the semantic definition bundle below.
        try:
            supplied_predicate_refs = tuple(
                definition.predicate_ref for definition in predicate_definitions
            )
            supplied_operator_refs = tuple(
                definition.operator_ref for definition in operator_definitions
            )
            supplied_state_refs = tuple(
                definition.state_variable_ref
                for definition in state_variable_definitions
            )
            supplied_derived_rule_refs = tuple(
                definition.derived_fact_rule_ref
                for definition in derived_fact_rule_definitions
            )
        except AttributeError as error:
            raise SemanticContractError("profile definition closure") from error

        expected_predicate_refs = set(semantics_profile.predicate_definition_refs)
        expected_operator_refs = set(action_space_profile.allowed_operator_refs)
        expected_state_refs = set(action_space_profile.state_variable_definition_refs)
        if (
            len(set(supplied_predicate_refs)) != len(supplied_predicate_refs)
            or len(set(supplied_operator_refs)) != len(supplied_operator_refs)
            or len(set(supplied_state_refs)) != len(supplied_state_refs)
            or len(set(supplied_derived_rule_refs)) != len(supplied_derived_rule_refs)
        ):
            raise SemanticContractError("profile definition closure")

        # Preserve the semantic source of a malformed cross-record edge before
        # enforcing exact supplied specialized tuples.  A profile omission is
        # not the same violation as a caller-supplied unreferenced record: the
        # former must identify authorization, operator, or derived-output
        # authority, while the latter remains a closed-root failure below.
        authorization = problem.intervention_authorization
        for leaf in authorization.authorized_primary_write_set:
            matching_definitions = tuple(
                definition
                for definition in state_variable_definitions
                if definition.state_variable_schema_ref
                == leaf.state_variable_schema_ref
            )
            if (
                len(matching_definitions) == 1
                and matching_definitions[0].state_variable_ref
                not in expected_state_refs
            ):
                raise SemanticContractError("authorization profile state")

        supplied_derived_rule_ref_set = set(supplied_derived_rule_refs)
        semantic_derived_rule_refs = set(semantics_profile.derived_fact_rule_refs)
        required_derived_rule_refs = set(authorization.required_derived_rule_refs)
        if (
            not required_derived_rule_refs <= semantic_derived_rule_refs
            or not required_derived_rule_refs <= supplied_derived_rule_ref_set
        ):
            raise SemanticContractError("authorization derived rule")
        operator_derived_rule_refs = {
            rule_ref
            for definition in operator_definitions
            for rule_ref in definition.derived_write_rule_refs
        }
        if not required_derived_rule_refs <= operator_derived_rule_refs:
            raise SemanticContractError("authorization derived rule")
        for definition in operator_definitions:
            if not set(definition.derived_write_rule_refs) <= (
                semantic_derived_rule_refs & supplied_derived_rule_ref_set
            ):
                raise SemanticContractError("operator derived rule")

        for rule in derived_fact_rule_definitions:
            for output in rule.fixed_output_set:
                matching_definitions = tuple(
                    definition
                    for definition in state_variable_definitions
                    if definition.state_variable_schema_ref
                    == output.state_variable_schema_ref
                )
                if len(matching_definitions) != 1:
                    continue
                output_definition = matching_definitions[0]
                if output_definition.write_authority is not WriteAuthority.DERIVED_ONLY:
                    raise SemanticContractError("derived output authority")
                if (
                    output_definition.derived_fact_rule_ref
                    != rule.derived_fact_rule_ref
                ):
                    raise SemanticContractError("derived output owner")
        for definition in state_variable_definitions:
            if (
                definition.derived_fact_rule_ref is not None
                and definition.derived_fact_rule_ref
                not in semantic_derived_rule_refs & supplied_derived_rule_ref_set
            ):
                raise SemanticContractError("derived output owner")

        if (
            set(supplied_predicate_refs) != expected_predicate_refs
            or set(supplied_operator_refs) != expected_operator_refs
            or set(supplied_state_refs) != expected_state_refs
        ):
            raise SemanticContractError("profile definition closure")

        supplied_states_by_ref = {
            definition.state_variable_ref: definition
            for definition in state_variable_definitions
        }
        selected_state_definitions = tuple(
            supplied_states_by_ref[reference] for reference in expected_state_refs
        )
        expected_derived_rule_refs = required_derived_rule_refs | {
            definition.derived_fact_rule_ref
            for definition in selected_state_definitions
            if definition.derived_fact_rule_ref is not None
        }
        if (
            set(supplied_derived_rule_refs) != expected_derived_rule_refs
            or not expected_derived_rule_refs <= semantic_derived_rule_refs
        ):
            raise SemanticContractError("profile definition closure")

        self.validate_definition_closure(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
            value_schema_definitions,
            predicate_definitions,
            state_variable_definitions,
            derived_fact_rule_definitions,
            operator_definitions,
            implementation_registry_snapshot,
        )
        if problem.definition_bundle != semantic_definition_bundle:
            raise SemanticContractError("semantic definition bundle")
        if problem.scene_state != scene_state:
            raise SemanticContractError("scene state")
        if (
            problem.semantics_profile_ref != semantics_profile.semantics_profile_ref
            or problem.action_space_profile_ref
            != action_space_profile.action_space_profile_ref
        ):
            raise SemanticContractError("semantic/action profile")
        profile_definitions, solve_definitions, _definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        profile_bindings = _definition_record_bindings(
            profile_definitions,
            _definition_roles(profile_definitions),
        )
        for reference, digest in (
            (
                semantics_profile.semantics_profile_ref,
                semantics_profile.semantics_profile_sha256,
            ),
            (
                action_space_profile.action_space_profile_ref,
                action_space_profile.action_space_profile_sha256,
            ),
        ):
            if profile_bindings.get(reference) != digest:
                raise DefinitionClosureError("definition record binding")
        if (
            not _references(semantics_profile, "definition:")
            <= profile_definitions.keys()
        ):
            raise DefinitionClosureError("semantic definition root")
        action_profile_refs = _references(action_space_profile, "definition:")
        permitted_operational_profile_ref = {
            action_space_profile.publication_proof_policy_ref
        }
        if (
            not action_profile_refs - permitted_operational_profile_ref
            <= profile_definitions.keys()
            or action_space_profile.publication_proof_policy_ref
            not in solve_definitions
        ):
            raise DefinitionClosureError("semantic definition root")

        # Do not let the caller's schema tuple or the problem's embedded
        # definition bundle establish semantic meaning.  The finite schema
        # universe starts at real problem/profile records and the exact
        # profile-selected specialized records, then closes through semantic
        # definition payloads and declared schema dependencies.
        semantic_root_records: tuple[object, ...] = (
            scene_state,
            problem.intervention_authorization,
            problem.before_preconditions,
            problem.after_goal,
            problem.preservation_invariants,
            problem.explicit_observation_obligations,
            problem.objective_expression,
            problem.numeric_semantics_ref,
            semantics_profile,
            action_space_profile,
            *predicate_definitions,
            *state_variable_definitions,
            *derived_fact_rule_definitions,
            *operator_definitions,
        )
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        expected_schema_refs = _semantic_schema_definition_closure(
            profile_definitions,
            schemas,
            semantic_root_records,
        )
        if set(schemas) != expected_schema_refs:
            raise DefinitionClosureError("semantic schema closure")
        _validate_exact_root_definition_closure(
            profile_definitions,
            root_records=semantic_root_records,
            record_binding_targets={
                problem.numeric_semantics_ref,
                semantics_profile.semantics_profile_ref,
                action_space_profile.action_space_profile_ref,
                *expected_schema_refs,
            },
            root_label="semantic",
            include_complete_domain_dependents=True,
        )
        if (
            problem.numeric_semantics_ref != semantics_profile.numeric_semantics_ref
            or action_space_profile.numeric_semantics_ref
            != semantics_profile.numeric_semantics_ref
        ):
            raise SemanticContractError("profile numeric semantics")
        if (
            scene_state.base_scene_schema_ref
            not in semantics_profile.accepted_scene_and_fact_schema_refs
            or scene_state.base_scene_schema_ref
            not in action_space_profile.accepted_scene_schema_refs
        ):
            raise SemanticContractError("scene schema")

        _, leaf_values, base_scene = _validate_scene_state_envelope(
            scene_state,
            state_variable_definitions,
            value_schema_definitions,
            profile_definitions,
            after_state=False,
        )

        expected_entities = {
            identifier for _family, identifier, _value in _base_fact_rows(base_scene)
        } | {
            fact.subject_entity_id
            for bundle in scene_state.extension_fact_bundles
            for fact in bundle.facts
        }
        if set(scene_state.closed_entity_index) != expected_entities:
            raise SemanticContractError("complete scene entity index")
        leaves = scene_state.canonical_state_leaf_index.leaves
        leaf_keys = {_state_key(leaf) for leaf in leaves}
        base_owners = {
            (family, identifier)
            for family, identifier, _fact in _base_fact_rows(base_scene)
        }
        extension_facts = _extension_facts_by_address(scene_state)
        base_addresses = {
            (leaf.fact_family_ref, leaf.entity_or_fact_key, leaf.field_path_ref)
            for leaf in leaves
            if (leaf.fact_family_ref, leaf.entity_or_fact_key) in base_owners
        }
        if base_addresses != _expected_base_leaf_addresses(base_scene):
            raise SemanticContractError("exact state leaf index")
        by_family_identifier = _state_leaf_owners(leaves)
        for family, identifier, _value in _base_fact_rows(base_scene):
            owned = by_family_identifier.get((family, identifier), ())
            if not any(
                leaf.field_path_ref.startswith("field-path:presence-") for leaf in owned
            ):
                raise SemanticContractError("complete state leaf presence")
            if not any(
                not leaf.field_path_ref.startswith("field-path:presence-")
                for leaf in owned
            ):
                raise SemanticContractError("complete state leaf value")

        state_by_schema = {
            definition.state_variable_schema_ref: definition
            for definition in state_variable_definitions
        }
        state_by_ref = {
            definition.state_variable_ref: definition
            for definition in state_variable_definitions
        }
        if len(state_by_schema) != len(state_variable_definitions) or len(
            state_by_ref
        ) != len(state_variable_definitions):
            raise SemanticContractError("state leaf definition")
        if any(
            leaf.state_variable_schema_ref not in state_by_schema for leaf in leaves
        ):
            raise SemanticContractError("state leaf definition")
        if not leaf_keys:
            raise SemanticContractError("complete state leaf index")

        # A submitted leaf is not owned merely because its schema happens to
        # occur in a state definition.  Its complete address and value schema
        # must come from the explicit typed definition record.  This makes the
        # frozen scene index a closed semantic partition, not an opaque list of
        # presence/value placeholders.
        definitions, _solve_definitions, _all_definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        metadata_by_ref = _state_definition_metadata(definitions)
        if set(metadata_by_ref) != set(state_by_ref):
            raise SemanticContractError("schema-owned field path")
        metadata_by_address: dict[
            tuple[str, str, str, str, str], StateVariableDefinition
        ] = {}
        for definition_ref, state_definition in state_by_ref.items():
            (
                state_variable_schema_ref,
                state_schema_ref,
                fact_family_ref,
                entity_or_fact_key,
                field_path_ref,
                value_schema_ref,
            ) = metadata_by_ref[definition_ref]
            if (
                state_variable_schema_ref != state_definition.state_variable_schema_ref
                or value_schema_ref != state_definition.value_schema_ref
            ):
                raise SemanticContractError("schema-owned field path")
            address = (
                state_variable_schema_ref,
                state_schema_ref,
                fact_family_ref,
                entity_or_fact_key,
                field_path_ref,
            )
            if address in metadata_by_address:
                raise SemanticContractError("schema-owned field path")
            metadata_by_address[address] = state_definition
        leaf_by_address = {
            (
                leaf.state_variable_schema_ref,
                leaf.state_schema_ref,
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
                leaf.field_path_ref,
            ): leaf
            for leaf in leaves
        }
        if set(leaf_by_address) != set(metadata_by_address):
            raise SemanticContractError("schema-owned field path")
        extension_leaf_owners = _declared_extension_leaf_owners(
            leaves,
            base_owners,
        )
        if not set(extension_facts) <= set(extension_leaf_owners):
            raise SemanticContractError("extension fact address")
        for owned in extension_leaf_owners.values():
            if (
                len(owned) != 2
                or sum(
                    leaf.field_path_ref.startswith("field-path:presence-")
                    for leaf in owned
                )
                != 1
            ):
                raise SemanticContractError("complete extension state leaf")
        absent_extension_addresses = set(extension_leaf_owners) - set(extension_facts)
        semantics_by_ref = _state_definition_semantics(definitions)
        if set(semantics_by_ref) != set(state_by_ref):
            raise SemanticContractError("state leaf semantics")
        for definition_ref, state_definition in state_by_ref.items():
            if semantics_by_ref[definition_ref] != (
                state_definition.frame_ref,
                state_definition.unit_ref,
                state_definition.topology_ref,
            ):
                raise SemanticContractError("state leaf semantics")
        semantic_definition_refs = {
            reference
            for record in semantic_root_records
            for reference in _references(record, "definition:")
        } - permitted_operational_profile_ref
        if not semantic_definition_refs <= profile_definitions.keys():
            raise SemanticContractError("semantic definition closure")
        profile_capabilities = _references(action_space_profile, "capability:")
        self._validate_snapshot(
            implementation_registry_snapshot,
            profile_capabilities,
            (
                *predicate_definitions,
                *state_variable_definitions,
                *derived_fact_rule_definitions,
                *operator_definitions,
            ),
        )
        try:
            _validate_state_leaf_value_wires(
                leaves,
                leaf_values,
                {
                    _state_key(leaf): metadata_by_address[address]
                    for address, leaf in leaf_by_address.items()
                },
                schemas,
                set(extension_facts),
                absent_extension_addresses,
            )
        except DefinitionClosureError as error:
            raise SemanticContractError("state leaf value schema") from error

        predicate_by_ref = {
            definition.predicate_ref: definition for definition in predicate_definitions
        }
        derived_rules = {
            definition.derived_fact_rule_ref: definition
            for definition in derived_fact_rule_definitions
        }
        if (
            set(predicate_by_ref) != expected_predicate_refs
            or set(derived_rules) != expected_derived_rule_refs
            or set(state_by_ref) != expected_state_refs
        ):
            raise SemanticContractError("profile definition closure")
        predicate_refs = _formula_predicate_refs(
            (
                problem.before_preconditions,
                problem.after_goal,
                problem.preservation_invariants,
                problem.explicit_observation_obligations,
            )
        )
        if not predicate_refs <= predicate_by_ref.keys():
            raise SemanticContractError("formula predicate closure")
        if not predicate_refs <= set(semantics_profile.predicate_definition_refs):
            raise SemanticContractError("profile predicate closure")
        formula_contexts = (
            *(item.formula for item in problem.before_preconditions),
            problem.after_goal.formula,
            *(
                formula
                for invariant in problem.preservation_invariants
                for formula in (invariant.before_formula, invariant.after_formula)
            ),
            *(item.formula for item in problem.explicit_observation_obligations),
        )
        if not all(_formula_is_fully_grounded(formula) for formula in formula_contexts):
            raise SemanticContractError("formula grounding")
        for atom in _formula_atoms(formula_contexts):
            predicate = predicate_by_ref.get(atom.predicate_ref)
            if (
                predicate is None
                or tuple(operand.value_schema_ref for operand in atom.operands)
                != predicate.operand_schema_refs
            ):
                raise SemanticContractError("formula predicate closure")
            try:
                for operand in atom.operands:
                    _validate_typed_value(operand, schemas)
            except DefinitionClosureError as error:
                raise SemanticContractError("formula predicate closure") from error
        actual_fact_families = {
            family for family, _identifier, _fact in _base_fact_rows(base_scene)
        } | {
            fact.fact_family_ref
            for bundle in scene_state.extension_fact_bundles
            for fact in bundle.facts
        }
        for predicate_ref in predicate_refs:
            predicate = predicate_by_ref[predicate_ref]
            for prerequisite_ref in predicate.observation_prerequisite_template_refs:
                prerequisite = definitions.get(prerequisite_ref)
                if prerequisite is None:
                    raise SemanticContractError("prerequisite closure")
                try:
                    required_fact_family = _definition_record_reference(
                        prerequisite,
                        _PREREQUISITE_FACT_FAMILY_FIELD,
                    )
                except DefinitionClosureError as error:
                    raise SemanticContractError("prerequisite closure") from error
                if required_fact_family not in actual_fact_families:
                    raise SemanticContractError("required fact completeness")

        operators = {
            definition.operator_ref: definition for definition in operator_definitions
        }
        if set(operators) != expected_operator_refs:
            raise SemanticContractError("profile definition closure")
        authorization = problem.intervention_authorization
        if (
            not authorization.allowed_operator_refs
            or not set(authorization.allowed_operator_refs)
            <= set(action_space_profile.allowed_operator_refs)
            or not set(authorization.allowed_operator_refs) <= operators.keys()
        ):
            raise SemanticContractError("authorization")
        if not set(authorization.editable_entity_ids) <= set(
            scene_state.closed_entity_index
        ):
            raise SemanticContractError("authorization")
        authorization_keys = {
            _state_key(leaf) for leaf in authorization.authorized_primary_write_set
        }
        if not authorization_keys <= leaf_keys:
            raise SemanticContractError("authorization")
        leaf_by_key = {_state_key(leaf): leaf for leaf in leaves}
        extension_subjects = {
            address: fact.subject_entity_id
            for address, fact in _extension_facts_by_address(scene_state).items()
        }
        for leaf in authorization.authorized_primary_write_set:
            definition = state_by_schema.get(leaf.state_variable_schema_ref)
            if (
                definition is None
                or definition.write_authority is not WriteAuthority.PRIMARY_WRITABLE
                or extension_subjects.get(
                    (leaf.fact_family_ref, leaf.entity_or_fact_key),
                    leaf.entity_or_fact_key,
                )
                not in authorization.editable_entity_ids
            ):
                raise SemanticContractError("authorization")
        authorized_primary_definition_refs = {
            state_by_schema[leaf.state_variable_schema_ref].state_variable_ref
            for leaf in authorization.authorized_primary_write_set
        }
        profile_state_definition_refs = set(
            action_space_profile.state_variable_definition_refs
        )
        if not authorized_primary_definition_refs <= profile_state_definition_refs:
            raise SemanticContractError("authorization profile state")
        for bound in authorization.variable_bounds:
            bound_leaf = leaf_by_key.get(_state_key(bound.state_variable_ref))
            bound_definition = state_by_schema.get(
                bound.state_variable_ref.state_variable_schema_ref
            )
            if (
                _state_key(bound.state_variable_ref) not in authorization_keys
                or bound_definition is None
                or bound_definition.write_authority
                is not WriteAuthority.PRIMARY_WRITABLE
            ):
                raise SemanticContractError("variable bound authorization")
            if bound_leaf != bound.state_variable_ref or (
                bound.value_schema_ref,
                bound.frame_ref,
                bound.unit_ref,
                bound.topology_ref,
            ) != (
                bound_definition.value_schema_ref,
                bound_definition.frame_ref,
                bound_definition.unit_ref,
                bound_definition.topology_ref,
            ):
                raise SemanticContractError("variable bound semantics")
            try:
                _validate_typed_value(bound.typed_domain, schemas)
            except DefinitionClosureError as error:
                raise SemanticContractError("variable bound semantics") from error
        mandatory_invariant_template_refs = set(
            action_space_profile.mandatory_invariant_template_refs
        )
        if not mandatory_invariant_template_refs <= {
            invariant.transition_comparator_ref
            for invariant in problem.preservation_invariants
        }:
            raise SemanticContractError("mandatory invariant closure")
        required_derived_rule_refs = set(authorization.required_derived_rule_refs)
        if not required_derived_rule_refs <= set(
            semantics_profile.derived_fact_rule_refs
        ) or not required_derived_rule_refs <= {
            rule_ref
            for operator_ref in authorization.allowed_operator_refs
            for rule_ref in operators[operator_ref].derived_write_rule_refs
        }:
            raise SemanticContractError("authorization derived rule")
        for operator_ref in action_space_profile.allowed_operator_refs:
            operator = operators[operator_ref]
            if (
                operator.transition_semantics_ref
                not in semantics_profile.transition_semantics_refs
            ):
                raise SemanticContractError("operator transition semantics")
            for pattern in (
                *operator.read_footprint,
                *operator.primary_write_footprint,
            ):
                state_definition = state_by_ref.get(
                    pattern.state_variable_definition_ref
                )
                if state_definition is None:
                    raise SemanticContractError("operator state footprint")
                if (
                    pattern.state_variable_definition_ref
                    not in profile_state_definition_refs
                ):
                    raise SemanticContractError("operator profile state")
                if any(
                    binding.parameter_schema_ref not in operator.parameter_schema_refs
                    for binding in pattern.parameter_bindings
                ):
                    raise SemanticContractError("operator parameter binding")
            if not set(
                operator.derived_write_rule_refs
            ) <= derived_rules.keys() or not set(
                operator.derived_write_rule_refs
            ) <= set(semantics_profile.derived_fact_rule_refs):
                raise SemanticContractError("operator derived rule")
        allowed_primary_footprints = {
            pattern.state_variable_definition_ref
            for operator_ref in authorization.allowed_operator_refs
            for pattern in operators[operator_ref].primary_write_footprint
        }
        if not authorized_primary_definition_refs <= allowed_primary_footprints:
            raise SemanticContractError("authorization operator footprint")
        output_owners: dict[bytes, str] = {}
        for rule in derived_fact_rule_definitions:
            for read in rule.fixed_read_set:
                read_leaf = leaf_by_key.get(_state_key(read))
                read_definition = state_by_schema.get(read.state_variable_schema_ref)
                if read_leaf != read or read_definition is None:
                    raise SemanticContractError("derived read closure")
            for output in rule.fixed_output_set:
                output_leaf = leaf_by_key.get(_state_key(output))
                output_definition = state_by_schema.get(
                    output.state_variable_schema_ref
                )
                if (
                    output_leaf != output
                    or output_definition is None
                    or output_definition.write_authority
                    is not WriteAuthority.DERIVED_ONLY
                ):
                    raise SemanticContractError("derived output authority")
                if (
                    output_definition.derived_fact_rule_ref
                    != rule.derived_fact_rule_ref
                ):
                    raise SemanticContractError("derived output owner")
                output_key = _state_key(output)
                if output_key in output_owners:
                    raise SemanticContractError("derived output owner")
                output_owners[output_key] = rule.derived_fact_rule_ref
        derived_output_definition_refs = {
            state_by_schema[output.state_variable_schema_ref].state_variable_ref
            for rule in derived_fact_rule_definitions
            for output in rule.fixed_output_set
        }
        derived_only_definition_refs = {
            definition.state_variable_ref
            for definition in state_variable_definitions
            if definition.write_authority is WriteAuthority.DERIVED_ONLY
        }
        profile_primary_footprints = {
            pattern.state_variable_definition_ref
            for operator_ref in action_space_profile.allowed_operator_refs
            for pattern in operators[operator_ref].primary_write_footprint
        }
        if profile_primary_footprints & (
            derived_output_definition_refs | derived_only_definition_refs
        ):
            raise SemanticContractError("primary/derived footprint")
        if not required_derived_rule_refs <= {
            rule.derived_fact_rule_ref for rule in derived_fact_rule_definitions
        }:
            raise SemanticContractError("authorization derived rule")
        objective_refs = {
            term.objective_definition_ref for term in problem.objective_expression.terms
        }
        if not objective_refs <= set(semantics_profile.objective_definition_refs):
            raise SemanticContractError("objective closure")
        for term in problem.objective_expression.terms:
            objective_definition = definitions.get(term.objective_definition_ref)
            if objective_definition is None:
                raise SemanticContractError("objective closure")
            try:
                expected_objective_semantics = (
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_INPUT_SELECTOR_FIELD,
                    ),
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_UNIT_FIELD,
                    ),
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_NORMALIZATION_FIELD,
                    ),
                )
            except DefinitionClosureError as error:
                raise SemanticContractError("objective unit/normalization") from error
            if expected_objective_semantics != (
                term.input_selector_definition_ref,
                term.unit_ref,
                term.normalization_definition_ref,
            ):
                raise SemanticContractError("objective unit/normalization")
        # Hash identity is checked after the explanatory semantic boundary so a
        # malformed authorization/leaf/profile reports that direct violation
        # rather than a stale enclosing root caused by a test-only model copy.
        for record, reason in (
            (problem, "semantic problem hash"),
            (semantics_profile, "semantics profile hash"),
            (action_space_profile, "action profile hash"),
            (scene_state, "scene state hash"),
        ):
            if not _self_digest_matches(record):
                raise SemanticContractError(reason)
        return problem

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

    def validate_edit_program(
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
        program: EditProgram,
        before_state: SceneStateEnvelope,
        after_state: SceneStateEnvelope,
        grounded_obligations: GroundedObligationSet,
        intervention_authorization: InterventionAuthorization,
    ) -> EditProgram:
        """Reconstruct the complete state partition for one explicit edit program."""

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
        if intervention_authorization != problem.intervention_authorization:
            raise SemanticContractError("intervention authorization")
        if before_state != problem.scene_state:
            raise SemanticContractError("program before scene")
        try:
            before_base_scene = _validated_base_scene_payload(
                before_state.base_scene_payload
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise SemanticContractError("program before scene") from error
        if (
            program.semantic_problem_sha256 != problem.semantic_problem_sha256
            or program.action_space_profile_sha256
            != action_space_profile.action_space_profile_sha256
        ):
            raise SemanticContractError("program roots")
        if (
            program.before_state_sha256 != before_state.scene_state_sha256
            or program.after_scene_state != after_state
            or program.after_scene_state_sha256 != after_state.scene_state_sha256
        ):
            raise SemanticContractError("program state")
        if after_state.base_scene_schema_ref != scene_state.base_scene_schema_ref:
            raise SemanticContractError("complete after state")
        if (
            program.grounded_obligation_set_sha256
            != grounded_obligations.grounded_obligation_set_sha256
        ):
            raise SemanticContractError("program obligations")

        definitions, _solve_definitions, _all_definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        _, after_values, after_base_scene = _validate_scene_state_envelope(
            after_state,
            state_variable_definitions,
            value_schema_definitions,
            definitions,
            after_state=True,
        )

        manifest = program.state_delta_manifest
        primary_keys = {_state_key(leaf) for leaf in manifest.authorized_primary_writes}
        derived_keys = {_state_key(leaf) for leaf in manifest.recomputed_derived_writes}
        if primary_keys & derived_keys:
            raise SemanticContractError("primary/derived")
        before_leaves = {
            _state_key(leaf): leaf
            for leaf in before_state.canonical_state_leaf_index.leaves
        }
        after_leaves = {
            _state_key(leaf): leaf
            for leaf in after_state.canonical_state_leaf_index.leaves
        }
        complete_keys = set(before_leaves) | set(after_leaves)
        state_by_schema = {
            definition.state_variable_schema_ref: definition
            for definition in state_variable_definitions
        }
        metadata_by_ref = _state_definition_metadata(definitions)
        expected_after_addresses = {
            (
                metadata[0],
                metadata[1],
                metadata[2],
                metadata[3],
                metadata[4],
            )
            for metadata in metadata_by_ref.values()
        }
        actual_after_addresses = {
            (
                leaf.state_variable_schema_ref,
                leaf.state_schema_ref,
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
                leaf.field_path_ref,
            )
            for leaf in after_leaves.values()
        }
        if actual_after_addresses != expected_after_addresses:
            raise SemanticContractError("complete after state")
        if {
            (leaf.fact_family_ref, leaf.entity_or_fact_key, leaf.field_path_ref)
            for leaf in after_leaves.values()
            if (
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
            )
            in {
                (family, identifier)
                for family, identifier, _fact in _base_fact_rows(after_base_scene)
            }
        } != _expected_base_leaf_addresses(after_base_scene):
            raise SemanticContractError("complete after state")
        after_base_owners = {
            (family, identifier)
            for family, identifier, _fact in _base_fact_rows(after_base_scene)
        }
        after_extension_facts = _extension_facts_by_address(after_state)
        absent_after_extension_addresses = set(
            _declared_extension_leaf_owners(
                tuple(after_leaves.values()),
                after_base_owners,
            )
        ) - set(after_extension_facts)
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        after_definitions_by_leaf: dict[bytes, StateVariableDefinition] = {}
        for leaf in after_leaves.values():
            state_definition = state_by_schema.get(leaf.state_variable_schema_ref)
            if state_definition is None:
                raise SemanticContractError("complete after state")
            after_definitions_by_leaf[_state_key(leaf)] = state_definition
        try:
            _validate_state_leaf_value_wires(
                tuple(after_leaves.values()),
                after_values,
                after_definitions_by_leaf,
                schemas,
                set(after_extension_facts),
                absent_after_extension_addresses,
            )
        except DefinitionClosureError as error:
            raise SemanticContractError("complete after state") from error
        if not (primary_keys | derived_keys) <= complete_keys:
            raise SemanticContractError("complete delta partition")
        authorized_keys = {
            _state_key(leaf)
            for leaf in intervention_authorization.authorized_primary_write_set
        }
        if not primary_keys <= authorized_keys:
            raise SemanticContractError("authorization")
        output_owners: dict[bytes, str] = {}
        for rule in derived_fact_rule_definitions:
            if (
                rule.derived_fact_rule_ref
                not in intervention_authorization.required_derived_rule_refs
            ):
                continue
            for leaf in rule.fixed_output_set:
                key = _state_key(leaf)
                if key in output_owners:
                    raise SemanticContractError("derived output owner")
                output_owners[key] = rule.derived_fact_rule_ref
        rule_outputs = set(output_owners)
        if derived_keys != rule_outputs:
            raise SemanticContractError("derived footprint")
        before_values = _leaf_values(before_state, before_base_scene)
        changed = {
            key
            for key in complete_keys
            if canonical_json_bytes(before_values.get(key))
            != canonical_json_bytes(after_values.get(key))
        }
        if changed != primary_keys | derived_keys:
            raise SemanticContractError("complete delta partition")
        ordered_union_keys = tuple(
            _state_key(leaf) for leaf in before_state.canonical_state_leaf_index.leaves
        ) + tuple(
            key for key in sorted(set(after_leaves) - set(before_leaves), key=_ref_key)
        )
        unchanged_rows = tuple(
            (
                (before_leaves[key] if key in before_leaves else after_leaves[key]),
                before_values.get(key),
            )
            for key in ordered_union_keys
            if key not in changed
            and canonical_json_bytes(before_values.get(key))
            == canonical_json_bytes(after_values.get(key))
        )
        if manifest.unchanged_leaves_digest != canonical_sha256(
            unchanged_rows,
            domain=_UNCHANGED_LEAF_DOMAIN,
        ):
            raise SemanticContractError("unchanged-leaf digest")
        if (
            manifest.complete_before_leaf_index_sha256
            != before_state.canonical_state_leaf_index.state_leaf_index_sha256
            or manifest.complete_after_leaf_index_sha256
            != after_state.canonical_state_leaf_index.state_leaf_index_sha256
        ):
            raise SemanticContractError("complete leaf index")
        operators = {
            definition.operator_ref: definition for definition in operator_definitions
        }
        if len(program.steps) > intervention_authorization.maximum_program_steps:
            raise SemanticContractError("program steps")
        problem_precondition_bytes = {
            canonical_json_bytes(precondition)
            for precondition in problem.before_preconditions
        }
        primary_footprint_definition_refs: set[str] = set()
        derived_rule_refs: set[str] = set()
        for step in program.steps:
            definition = operators.get(step.operator_ref)
            if (
                definition is None
                or step.operator_ref
                not in intervention_authorization.allowed_operator_refs
            ):
                raise SemanticContractError("authorization")
            if (
                not {
                    canonical_json_bytes(precondition)
                    for precondition in definition.required_preconditions
                }
                <= problem_precondition_bytes
            ):
                raise SemanticContractError("operator required precondition")
            if len(step.arguments) != len(definition.parameter_schema_refs):
                raise SemanticContractError("operator parameters")
            if (
                tuple(argument.value.value_schema_ref for argument in step.arguments)
                != definition.parameter_schema_refs
            ):
                raise SemanticContractError("operator parameters")
            for argument in step.arguments:
                try:
                    _validate_typed_value(argument.value, schemas)
                except DefinitionClosureError as error:
                    raise SemanticContractError("operator parameters") from error
            arguments_by_name = {
                argument.argument_name: argument for argument in step.arguments
            }
            for pattern in (
                *definition.read_footprint,
                *definition.primary_write_footprint,
            ):
                for binding in pattern.parameter_bindings:
                    argument = arguments_by_name.get(binding.parameter_ref)
                    if (
                        argument is None
                        or argument.value.value_schema_ref
                        != binding.parameter_schema_ref
                    ):
                        raise SemanticContractError("operator parameter binding")
            primary_footprint_definition_refs.update(
                pattern.state_variable_definition_ref
                for pattern in definition.primary_write_footprint
            )
            derived_rule_refs.update(definition.derived_write_rule_refs)
            if not {
                pattern.state_variable_definition_ref
                for pattern in definition.read_footprint
            } <= {
                state_definition.state_variable_ref
                for state_definition in state_by_schema.values()
            }:
                raise SemanticContractError("operator footprint")
        if (
            not set(intervention_authorization.required_derived_rule_refs)
            <= derived_rule_refs
        ):
            raise SemanticContractError("derived footprint")
        if not derived_rule_refs <= set(
            intervention_authorization.required_derived_rule_refs
        ):
            raise SemanticContractError("authorization derived footprint")
        actual_primary_footprint = {
            key
            for key, leaf in ({**before_leaves, **after_leaves}).items()
            if state_by_schema[leaf.state_variable_schema_ref].state_variable_ref
            in primary_footprint_definition_refs
        }
        if not actual_primary_footprint <= authorized_keys:
            raise SemanticContractError("authorization operator footprint")
        if not primary_keys <= actual_primary_footprint:
            raise SemanticContractError("operator footprint")
        extension_subjects = {
            address: fact.subject_entity_id
            for state in (before_state, after_state)
            for address, fact in _extension_facts_by_address(state).items()
        }
        edited_entities = {
            extension_subjects.get(
                (
                    (before_leaves.get(key) or after_leaves[key]).fact_family_ref,
                    (before_leaves.get(key) or after_leaves[key]).entity_or_fact_key,
                ),
                (before_leaves.get(key) or after_leaves[key]).entity_or_fact_key,
            )
            for key in primary_keys
        }
        if len(edited_entities) > intervention_authorization.maximum_edited_entities:
            raise SemanticContractError("authorization")
        if not _self_digest_matches(grounded_obligations):
            raise SemanticContractError("grounded obligations")
        if not set(grounded_obligations.source_definition_refs) <= definitions.keys():
            raise SemanticContractError("grounded obligations")
        partitions = (
            (grounded_obligations.before_preconditions, problem.before_preconditions),
            (grounded_obligations.after_goals, (problem.after_goal,)),
            (
                grounded_obligations.preservation_invariants,
                problem.preservation_invariants,
            ),
            (
                grounded_obligations.observation_obligations,
                problem.explicit_observation_obligations,
            ),
        )
        if any(
            {canonical_json_bytes(item.context) for item in grounded_partition}
            != {canonical_json_bytes(item) for item in semantic_partition}
            for grounded_partition, semantic_partition in partitions
        ):
            raise SemanticContractError("grounded obligations")
        if any(
            canonical_json_bytes(obligation.context)
            not in {
                canonical_json_bytes(item.context)
                for item in grounded_obligations.after_goals
            }
            for definition in (operators[step.operator_ref] for step in program.steps)
            for obligation in definition.generated_obligations
        ):
            raise SemanticContractError("grounded obligations")
        if not _self_digest_matches(manifest) or not _self_digest_matches(program):
            raise SemanticContractError("program hash")
        return program

    def validate_outcome_contract(
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
        selection: BackendSelectionRecord,
        proposal: BackendProposal | None,
        proof_material: ProofMaterialEnvelope | None,
        checked_proof_outcome: CheckedProofOutcome | None,
        verifier_dispatch_record: VerifierDispatchRecord | None,
        certificate: CertifiedSolutionCertificate | ProvenUnsatCertificate | None,
        result: (
            CertifiedSolutionResult
            | ProvenUnsatResult
            | NoncertifiedWitnessResult
            | UnknownResult
        ),
        program: EditProgram | None,
        grounded_obligations: GroundedObligationSet | None,
    ) -> (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ):
        """Validate the submitted, branch-specific hash DAG without execution.

        The method accepts all four structural terminal envelopes.  A weak
        branch is admissible as a weak branch when policy permits it; only a
        certificate or complete-domain UNSAT branch may claim trusted strength.
        """

        self.validate_solve_request(
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
            request,
            backend_descriptor_bundle,
            solver_config,
            proof_policy,
            resource_policy,
            backend_routing_policy,
        )
        for record in (
            selection,
            proposal,
            proof_material,
            checked_proof_outcome,
            verifier_dispatch_record,
            result,
        ):
            if record is not None and not _self_digest_matches(record):
                raise SemanticContractError("outcome record hash")
        if certificate is not None and not _self_digest_matches(certificate):
            raise SemanticContractError("certificate hash")
        if (
            result.semantic_problem_sha256 != problem.semantic_problem_sha256
            or result.solve_request_sha256 != request.solve_request_sha256
        ):
            raise SemanticContractError("result roots")
        if (
            selection.semantic_problem_sha256 != problem.semantic_problem_sha256
            or selection.solve_request_sha256 != request.solve_request_sha256
            or selection.implementation_registry_snapshot_sha256
            != implementation_registry_snapshot.implementation_registry_snapshot_sha256
            or selection.backend_descriptor_bundle_sha256
            != backend_descriptor_bundle.backend_descriptor_bundle_sha256
            or selection.backend_routing_policy_sha256
            != backend_routing_policy.backend_routing_policy_sha256
        ):
            raise SemanticContractError("selection roots")

        _semantic_definitions, _solve_definitions, definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        roles = _definition_roles(definitions)
        available_backend_refs = {
            descriptor.backend_ref
            for descriptor in backend_descriptor_bundle.backend_descriptors
        }
        unavailable_backend_refs = {
            unavailable.backend_ref
            for unavailable in backend_descriptor_bundle.unavailable_optional_backends
        }
        if not available_backend_refs or (
            available_backend_refs & unavailable_backend_refs
        ):
            # Task 7 has no descriptor digest field for an unavailable row, so
            # a nonempty BackendSelectionRecord cannot safely represent an
            # unavailable-only universe.  It must terminate fail-closed rather
            # than fabricate a CapabilityMismatch.
            raise SemanticContractError("unavailable backend selection")
        if (
            set(selection.ordered_candidate_backend_refs) & unavailable_backend_refs
            or selection.selected_backend_ref in unavailable_backend_refs
        ):
            raise SemanticContractError("unavailable backend selection")
        expected_rows = _reconstructed_backend_rows(
            request,
            action_space_profile,
            semantics_profile,
            predicate_definitions,
            operator_definitions,
            definitions,
            roles,
        )
        if (
            set(selection.ordered_candidate_backend_refs) != available_backend_refs
            or selection.ordered_candidate_backend_refs
            != tuple(row.backend_ref for row in expected_rows)
            or canonical_json_bytes(selection.capability_rows)
            != canonical_json_bytes(expected_rows)
        ):
            raise SemanticContractError("ordered backend candidates")
        if (
            result.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
        ):
            raise SemanticContractError("result roots")
        if (
            roles.get(selection.selection_disposition_claim_ref)
            != _ROLE_ROUTING_SELECTION_DISPOSITION
            or roles.get(selection.deterministic_selection_reason_ref)
            != _ROLE_ROUTING_SELECTION_REASON
        ):
            raise SemanticContractError("selection definition closure")
        descriptor_by_ref = {
            descriptor.backend_ref: descriptor
            for descriptor in backend_descriptor_bundle.backend_descriptors
        }
        matching_rows = tuple(
            row for row in expected_rows if isinstance(row, CapabilityMatch)
        )
        if selection.selection_disposition == "NO_SELECTION":
            if matching_rows:
                raise SemanticContractError("deterministic selected backend")
            if any(
                item is not None
                for item in (
                    proposal,
                    proof_material,
                    checked_proof_outcome,
                    verifier_dispatch_record,
                    certificate,
                    program,
                    grounded_obligations,
                )
            ) or not isinstance(result, UnknownResult):
                raise SemanticContractError("no-selection outcome")
            if (
                result.checked_proof_outcome_sha256 is not None
                or result.verifier_dispatch_record_sha256 is not None
                or result.checker_disposition is not None
            ):
                raise SemanticContractError("no-selection checker")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                result.reason_claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "unknown reason admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or result.reason_claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("unknown claim admissibility")
            _validate_resource_usage(
                result.resource_usage, resource_policy, definitions
            )
            return result
        if selection.selection_disposition != "SELECTED":
            raise SemanticContractError("selection disposition")
        if (
            selection.selected_backend_ref is None
            or selection.selected_backend_descriptor_sha256 is None
            or selection.selected_backend_ref not in descriptor_by_ref
            or descriptor_by_ref[
                selection.selected_backend_ref
            ].backend_descriptor_sha256
            != selection.selected_backend_descriptor_sha256
        ):
            raise SemanticContractError("selected backend descriptor")
        selected_row = next(
            (
                row
                for row in expected_rows
                if row.backend_ref == selection.selected_backend_ref
            ),
            None,
        )
        if not isinstance(selected_row, CapabilityMatch):
            raise SemanticContractError("selected backend capability")
        if not matching_rows or selected_row != matching_rows[0]:
            raise SemanticContractError("deterministic selected backend")
        descriptor = descriptor_by_ref[selection.selected_backend_ref]

        if proposal is None or proof_material is None:
            raise SemanticContractError("selected proposal presence")

        if (
            proposal.semantic_problem_sha256 != problem.semantic_problem_sha256
            or proposal.solve_request_sha256 != request.solve_request_sha256
            or proposal.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or proposal.proposal_backend_ref != selection.selected_backend_ref
            or proposal.proof_material_sha256 != proof_material.proof_material_sha256
        ):
            raise SemanticContractError("proposal roots")
        if (
            proof_material.semantic_problem_sha256 != problem.semantic_problem_sha256
            or proof_material.solve_request_sha256 != request.solve_request_sha256
            or proof_material.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or proof_material.proposal_backend_ref != selection.selected_backend_ref
            or proof_material.proof_material_definition_ref
            not in descriptor.emitted_proof_material_definition_refs
        ):
            raise SemanticContractError("proof material roots")
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        if any(
            value.value_schema_ref != proof_material.payload_schema_ref
            for value in proof_material.typed_payload
        ):
            raise SemanticContractError("proof material schema")
        try:
            for value in proof_material.typed_payload:
                _validate_typed_value(value, schemas)
        except DefinitionClosureError as error:
            raise SemanticContractError("proof material schema") from error
        if proposal.proposal_claim_definition_ref != result.claim_definition_ref:
            raise SemanticContractError("proposal claim")
        if isinstance(result, CertifiedSolutionResult) and (
            program is None
            or proposal.program_sha256 != program.program_sha256
            or proposal.after_scene_state_sha256 != program.after_scene_state_sha256
        ):
            raise SemanticContractError("proposal program")
        capability_owners = self._owners_by_capability()
        proposal_owner = capability_owners.get(proposal.proposal_backend_capability_ref)
        if (
            proposal_owner is None
            or proposal_owner.owner_ref != proposal.proposal_backend_owner_ref
            or proposal_owner.implementation_build_sha256
            != proposal.proposal_backend_build_sha256
            or descriptor.implementation_build_sha256
            != proposal.proposal_backend_build_sha256
            or proposal.proposal_backend_capability_ref
            not in action_space_profile.backend_capability_requirements
        ):
            raise ImplementationResolutionError("proposal backend owner/build")
        result_has_checker = result.checked_proof_outcome_sha256 is not None
        if (checked_proof_outcome is None) != (
            verifier_dispatch_record is None
        ) or result_has_checker != (checked_proof_outcome is not None):
            raise SemanticContractError("checker pair")
        checker_owner: StaticOwner | None = None
        if checked_proof_outcome is not None:
            assert verifier_dispatch_record is not None
            checker_owner = capability_owners.get(
                checked_proof_outcome.checker_capability_ref
            )
            if (
                checker_owner is None
                or checker_owner.implementation_build_sha256
                != checked_proof_outcome.checker_build_sha256
                or checker_owner.owner_ref == proposal_owner.owner_ref
                or checked_proof_outcome.checker_capability_ref
                not in descriptor.compatible_checker_capability_refs
                or checked_proof_outcome.checker_capability_ref
                not in proof_policy.required_checker_capability_refs
            ):
                raise ImplementationResolutionError("checker owner/build")
            if (
                checked_proof_outcome.semantic_problem_sha256
                != problem.semantic_problem_sha256
                or checked_proof_outcome.solve_request_sha256
                != request.solve_request_sha256
                or checked_proof_outcome.backend_selection_record_sha256
                != selection.backend_selection_record_sha256
                or checked_proof_outcome.proof_material_sha256
                != proof_material.proof_material_sha256
                or checked_proof_outcome.checked_claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or result.checked_proof_outcome_sha256
                != checked_proof_outcome.checked_proof_outcome_sha256
                or result.verifier_dispatch_record_sha256
                != verifier_dispatch_record.verifier_dispatch_record_sha256
                or result.checker_disposition
                != checked_proof_outcome.checker_disposition
            ):
                raise SemanticContractError("checked proof roots")
            if (
                verifier_dispatch_record.semantic_problem_sha256
                != problem.semantic_problem_sha256
                or verifier_dispatch_record.solve_request_sha256
                != request.solve_request_sha256
                or verifier_dispatch_record.semantic_definition_bundle_sha256
                != semantic_definition_bundle.definition_bundle_sha256
                or verifier_dispatch_record.solve_policy_definition_bundle_sha256
                != solve_policy_definition_bundle.definition_bundle_sha256
                or verifier_dispatch_record.backend_selection_record_sha256
                != selection.backend_selection_record_sha256
                or verifier_dispatch_record.checked_proof_outcome_sha256
                != checked_proof_outcome.checked_proof_outcome_sha256
                or verifier_dispatch_record.proof_policy_sha256
                != proof_policy.proof_policy_sha256
                or verifier_dispatch_record.proposal_backend_owner_ref
                != proposal.proposal_backend_owner_ref
                or verifier_dispatch_record.proposal_backend_capability_ref
                != proposal.proposal_backend_capability_ref
                or verifier_dispatch_record.proposal_backend_build_sha256
                != proposal.proposal_backend_build_sha256
                or verifier_dispatch_record.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or verifier_dispatch_record.checker_owner_ref != checker_owner.owner_ref
                or verifier_dispatch_record.checker_capability_ref
                != checked_proof_outcome.checker_capability_ref
                or verifier_dispatch_record.checker_build_sha256
                != checked_proof_outcome.checker_build_sha256
            ):
                raise SemanticContractError("verifier dispatch roots")

        for usage in (
            selection.resource_allocation,
            proposal.resource_usage,
            result.resource_usage,
            certificate.resource_usage if certificate is not None else None,
        ):
            if usage is not None:
                _validate_resource_usage(usage, resource_policy, definitions)
        _require_outcome_definition_role(
            roles,
            proof_material.proof_material_definition_ref,
            (_ROLE_PROOF_MATERIAL,),
            "proof material admissibility",
        )
        proof_definition = definitions[proof_material.proof_material_definition_ref]
        if (
            _definition_record_reference(
                proof_definition,
                _PROOF_PAYLOAD_SCHEMA_FIELD,
            )
            != proof_material.payload_schema_ref
        ):
            raise SemanticContractError("proof material schema")
        if (
            checked_proof_outcome is not None
            and _definition_record_reference(
                proof_definition,
                _PROOF_CHECKER_CAPABILITY_FIELD,
            )
            != checked_proof_outcome.checker_capability_ref
        ):
            raise SemanticContractError("proof material checker")

        if isinstance(result, CertifiedSolutionResult):
            if not isinstance(certificate, CertifiedSolutionCertificate):
                raise SemanticContractError("certificate branch")
            if program is None or grounded_obligations is None:
                raise SemanticContractError("certificate program")
            if checked_proof_outcome is None or verifier_dispatch_record is None:
                raise SemanticContractError("certificate checker")
            self.validate_edit_program(
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
                program,
                scene_state,
                program.after_scene_state,
                grounded_obligations,
                problem.intervention_authorization,
            )
            _require_outcome_definition_role(
                roles,
                certificate.claim_definition_ref,
                (_ROLE_CERTIFIED_SOLUTION,),
                "claim admissibility",
            )
            claim_definition = definitions[certificate.claim_definition_ref]
            if (
                _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                != proof_material.proof_material_definition_ref
                or _definition_record_reference(
                    claim_definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
                != checked_proof_outcome.checker_capability_ref
            ):
                raise SemanticContractError("claim admissibility")
            if (
                certificate.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or certificate.proof_material_definition_ref
                != _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
            ):
                raise SemanticContractError("certificate proof material")
            if (
                certificate != result.accepted_certificate
                or certificate.claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref != result.claim_definition_ref
                or certificate.claim_definition_ref
                not in proof_policy.accepted_claim_definition_refs
                or certificate.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or certificate.checker_disposition is not CheckerDisposition.ACCEPTED
                or checked_proof_outcome.checker_disposition
                is not CheckerDisposition.ACCEPTED
                or certificate.program_sha256 != program.program_sha256
                or certificate.after_scene_state_sha256
                != program.after_scene_state_sha256
                or certificate.state_delta_manifest_sha256
                != program.state_delta_manifest.state_delta_manifest_sha256
                or certificate.grounded_obligation_set_sha256
                != grounded_obligations.grounded_obligation_set_sha256
                or result.certificate_sha256 != certificate.certificate_sha256
                or result.program_sha256 != program.program_sha256
                or result.after_scene_state_sha256 != program.after_scene_state_sha256
            ):
                raise SemanticContractError("certificate admissibility")
        elif isinstance(result, ProvenUnsatResult):
            if not isinstance(certificate, ProvenUnsatCertificate):
                raise SemanticContractError("complete-domain certificate")
            if program is not None or grounded_obligations is not None:
                raise SemanticContractError("unsat program presence")
            if checked_proof_outcome is None or verifier_dispatch_record is None:
                raise SemanticContractError("complete-domain checker")
            _require_outcome_definition_role(
                roles,
                certificate.claim_definition_ref,
                (_ROLE_PROVEN_UNSAT,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                certificate.complete_domain_claim_definition_ref,
                (_ROLE_COMPLETE_DOMAIN,),
                "complete-domain coverage",
            )
            _require_outcome_definition_role(
                roles,
                certificate.sound_complete_domain_claim_definition_ref,
                (_ROLE_SOUND_COMPLETE_DOMAIN,),
                "complete-domain coverage",
            )
            claim_definition = definitions[certificate.claim_definition_ref]
            complete_domain_definition = definitions[
                certificate.complete_domain_claim_definition_ref
            ]
            sound_domain_definition = definitions[
                certificate.sound_complete_domain_claim_definition_ref
            ]
            if (
                _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                != proof_material.proof_material_definition_ref
                or _definition_record_reference(
                    claim_definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
                != checked_proof_outcome.checker_capability_ref
            ):
                raise SemanticContractError("claim admissibility")
            if (
                certificate.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or certificate.proof_material_definition_ref
                != _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
            ):
                raise SemanticContractError("certificate proof material")
            if (
                _definition_record_reference(
                    complete_domain_definition,
                    _COMPLETE_DOMAIN_CLAIM_FIELD,
                )
                != certificate.claim_definition_ref
                or _definition_record_reference(
                    sound_domain_definition,
                    _COMPLETE_DOMAIN_CLAIM_FIELD,
                )
                != certificate.claim_definition_ref
            ):
                raise SemanticContractError("complete-domain coverage")
            if (
                certificate != result.accepted_certificate
                or certificate.claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref != result.claim_definition_ref
                or certificate.claim_definition_ref
                not in proof_policy.accepted_claim_definition_refs
                or certificate.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or certificate.checker_disposition is not CheckerDisposition.ACCEPTED
                or checked_proof_outcome.checker_disposition
                is not CheckerDisposition.ACCEPTED
                or result.certificate_sha256 != certificate.certificate_sha256
                or result.complete_domain_coverage_artifact_sha256
                != certificate.complete_domain_coverage_artifact_sha256
            ):
                raise SemanticContractError("complete-domain coverage")
        elif isinstance(result, NoncertifiedWitnessResult):
            if certificate is not None:
                raise SemanticContractError("witness certificate presence")
            if not proof_policy.permit_noncertified_terminal_records:
                raise SemanticContractError("noncertified policy")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_NONCERTIFIED_WITNESS,),
                "claim admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("claim admissibility")
            if result.evidence_claim_definition_ref not in definitions:
                raise SemanticContractError("witness evidence admissibility")
            if result.program_sha256 is None:
                if (
                    program is not None
                    or grounded_obligations is not None
                    or proposal.program_sha256 is not None
                    or proposal.after_scene_state_sha256 is not None
                ):
                    raise SemanticContractError("witness program presence")
            else:
                if program is None or grounded_obligations is None:
                    raise SemanticContractError("witness program presence")
                self.validate_edit_program(
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
                    program,
                    scene_state,
                    program.after_scene_state,
                    grounded_obligations,
                    problem.intervention_authorization,
                )
                if (
                    result.program_sha256 != program.program_sha256
                    or result.after_scene_state_sha256
                    != program.after_scene_state_sha256
                    or proposal.program_sha256 != program.program_sha256
                    or proposal.after_scene_state_sha256
                    != program.after_scene_state_sha256
                ):
                    raise SemanticContractError("witness program")
            if result.checker_disposition is CheckerDisposition.ACCEPTED:
                raise SemanticContractError("weak claim promotion")
            if (
                checked_proof_outcome is not None
                and checked_proof_outcome.checker_disposition
                is CheckerDisposition.ACCEPTED
            ):
                raise SemanticContractError("weak claim promotion")
        elif isinstance(result, UnknownResult):
            if (
                certificate is not None
                or program is not None
                or grounded_obligations is not None
            ):
                raise SemanticContractError("unknown certificate presence")
            if (
                proposal.program_sha256 is not None
                or proposal.after_scene_state_sha256 is not None
            ):
                raise SemanticContractError("unknown proposal program")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                result.reason_claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "unknown reason admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or result.reason_claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("unknown claim admissibility")
            if result.checker_disposition is CheckerDisposition.ACCEPTED:
                raise SemanticContractError("unknown claim promotion")
            if (
                checked_proof_outcome is not None
                and checked_proof_outcome.checker_disposition
                is CheckerDisposition.ACCEPTED
            ):
                raise SemanticContractError("unknown claim promotion")
        else:  # pragma: no cover - discriminated domain union is exhaustive.
            raise SemanticContractError("result branch")

        # A direct result/certificate root must carry every static dependency;
        # this rejects forward, reverse, self, and mixed-branch digest swaps.
        if certificate is not None:
            certificate_fields = (
                ("semantic_problem_sha256", problem.semantic_problem_sha256),
                ("solve_request_sha256", request.solve_request_sha256),
                ("scene_state_sha256", scene_state.scene_state_sha256),
                (
                    "backend_selection_record_sha256",
                    selection.backend_selection_record_sha256,
                ),
                (
                    "checked_proof_outcome_sha256",
                    checked_proof_outcome.checked_proof_outcome_sha256,
                ),
                (
                    "verifier_dispatch_record_sha256",
                    verifier_dispatch_record.verifier_dispatch_record_sha256,
                ),
                (
                    "semantic_definition_bundle_sha256",
                    semantic_definition_bundle.definition_bundle_sha256,
                ),
                (
                    "solve_policy_definition_bundle_sha256",
                    solve_policy_definition_bundle.definition_bundle_sha256,
                ),
                (
                    "semantics_profile_sha256",
                    semantics_profile.semantics_profile_sha256,
                ),
                (
                    "action_space_profile_sha256",
                    action_space_profile.action_space_profile_sha256,
                ),
                (
                    "intervention_authorization_sha256",
                    problem.intervention_authorization.intervention_authorization_sha256,
                ),
                (
                    "objective_expression_sha256",
                    problem.objective_expression.objective_expression_sha256,
                ),
                ("proof_policy_sha256", proof_policy.proof_policy_sha256),
                ("resource_policy_sha256", resource_policy.resource_policy_sha256),
                (
                    "backend_routing_policy_sha256",
                    backend_routing_policy.backend_routing_policy_sha256,
                ),
                ("solver_config_sha256", solver_config.solver_config_sha256),
                (
                    "implementation_registry_snapshot_sha256",
                    implementation_registry_snapshot.implementation_registry_snapshot_sha256,
                ),
                (
                    "backend_descriptor_bundle_sha256",
                    backend_descriptor_bundle.backend_descriptor_bundle_sha256,
                ),
                (
                    "proposal_backend_build_sha256",
                    proposal.proposal_backend_build_sha256,
                ),
                ("checker_build_sha256", checked_proof_outcome.checker_build_sha256),
                ("proof_material_sha256", proof_material.proof_material_sha256),
            )
            if any(
                getattr(certificate, name) != expected
                for name, expected in certificate_fields
            ):
                raise SemanticContractError("certificate roots")
        return result

    def validate_submission_contract(
        self,
        *,
        submission: BackendSubmission,
        outcome_contract_arguments: dict[str, object],
    ) -> (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ):
        """Validate one M3 V2 submission without changing the retained API.

        Proposal wrappers add no semantics: after exact wrapper binding they
        delegate verbatim to :meth:`validate_outcome_contract`.  Program-free
        terminal evidence is checked only for its additive structural roots;
        terminal assembly remains the sole owner of certificates and results.
        """

        if type(submission) is BackendProposalSubmission:
            proposal = outcome_contract_arguments.get("proposal")
            if (
                proposal is None
                or canonical_json_bytes(proposal)
                != canonical_json_bytes(submission.proposal)
                or submission.backend_proposal_sha256
                != submission.proposal.backend_proposal_sha256
            ):
                raise SemanticContractError("proposal submission wrapper")
            return self.validate_outcome_contract(**outcome_contract_arguments)  # type: ignore[arg-type]

        if type(submission) not in (
            BackendCompleteUnsatEvidence,
            BackendUnknownEvidence,
        ):
            raise SemanticContractError("submission branch")
        selection = outcome_contract_arguments.get("selection")
        result = outcome_contract_arguments.get("result")
        if type(selection) is not BackendSelectionRecord:
            raise SemanticContractError("submission selection")
        if type(result) not in (ProvenUnsatResult, UnknownResult):
            raise SemanticContractError("terminal submission result")
        if (
            submission.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or result.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or submission.proof_material_sha256
            != submission.proof_material.proof_material_sha256
            or canonical_json_bytes(submission.resource_usage)
            != canonical_json_bytes(result.resource_usage)
        ):
            raise SemanticContractError("terminal submission roots")
        if type(submission) is BackendCompleteUnsatEvidence:
            if (
                type(result) is not ProvenUnsatResult
                or result.complete_domain_coverage_artifact_sha256
                != submission.complete_domain_coverage_artifact_sha256
            ):
                raise SemanticContractError("complete-domain submission")
        elif type(result) is not UnknownResult:
            raise SemanticContractError("unknown submission")
        return result


def validate_submission_contract_structure(
    *,
    submission: BackendSubmission,
    selection: BackendSelectionRecord,
    checked_proof_outcome: CheckedProofOutcome,
    verifier_dispatch_record: VerifierDispatchRecord,
    result: (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ),
) -> None:
    """Close the M3 submission-to-assembled-record hash DAG without execution.

    This is the dispatch-time half of the additive submission validator.  The
    retained instance method above remains the sole full M1 proposal-contract
    validator; its exact signature and behavior are intentionally unchanged.
    """

    if type(checked_proof_outcome) is not CheckedProofOutcome:
        raise TypeError("submission structure requires a typed checked outcome")
    if type(verifier_dispatch_record) is not VerifierDispatchRecord:
        raise TypeError("submission structure requires a typed checker dispatch")
    if (
        verifier_dispatch_record.checked_proof_outcome_sha256
        != checked_proof_outcome.checked_proof_outcome_sha256
        or verifier_dispatch_record.checker_capability_ref
        != checked_proof_outcome.checker_capability_ref
        or verifier_dispatch_record.checker_build_sha256
        != checked_proof_outcome.checker_build_sha256
    ):
        raise SemanticContractError("checker dispatch does not bind checked outcome")
    if (
        selection.selection_disposition != "SELECTED"
        or result.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or result.checked_proof_outcome_sha256
        != checked_proof_outcome.checked_proof_outcome_sha256
        or result.verifier_dispatch_record_sha256
        != verifier_dispatch_record.verifier_dispatch_record_sha256
        or result.checker_disposition != checked_proof_outcome.checker_disposition
    ):
        raise SemanticContractError("assembled submission roots")
    evidence = (
        submission.proposal
        if type(submission) is BackendProposalSubmission
        else submission
    )
    if (
        evidence.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or evidence.proof_material_sha256 != checked_proof_outcome.proof_material_sha256
        or evidence.proposal_backend_owner_ref
        != verifier_dispatch_record.proposal_backend_owner_ref
        or evidence.proposal_backend_capability_ref
        != verifier_dispatch_record.proposal_backend_capability_ref
        or evidence.proposal_backend_build_sha256
        != verifier_dispatch_record.proposal_backend_build_sha256
    ):
        raise SemanticContractError("assembled submission evidence")
    if type(submission) is BackendCompleteUnsatEvidence and (
        type(result) is not ProvenUnsatResult
        or result.complete_domain_coverage_artifact_sha256
        != submission.complete_domain_coverage_artifact_sha256
    ):
        raise SemanticContractError("assembled complete-domain evidence")
    if type(submission) is BackendUnknownEvidence and type(result) is not UnknownResult:
        raise SemanticContractError("assembled unknown evidence")
    if type(submission) is BackendProposalSubmission:
        if checked_proof_outcome.checker_disposition is CheckerDisposition.ACCEPTED:
            if type(result) is not CertifiedSolutionResult:
                raise SemanticContractError("assembled accepted proposal")
            certificate = result.accepted_certificate
            if (
                result.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
            ):
                raise SemanticContractError("certificate claim does not bind checker")
            if (
                certificate.proof_material_definition_ref
                != evidence.proof_material.proof_material_definition_ref
            ):
                raise SemanticContractError(
                    "certificate proof does not bind submission"
                )
            if (
                result.program_sha256 != evidence.program_sha256
                or result.after_scene_state_sha256 != evidence.after_scene_state_sha256
                or certificate.program_sha256 != evidence.program_sha256
                or certificate.after_scene_state_sha256
                != evidence.after_scene_state_sha256
            ):
                raise SemanticContractError("terminal payload does not bind submission")
        elif type(result) is not NoncertifiedWitnessResult:
            raise SemanticContractError("assembled limited proposal")
