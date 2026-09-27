"""Static registry definitions validation and explicit dependencies."""

from __future__ import annotations

from pydantic import (
    BaseModel,
)

from spatialcf.domain.definitions import (
    BOOTSTRAP_SCHEMA_SHA256,
    CanonicalDefinitionEnvelope,
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
)

from spatialcf.domain.outcomes import (
    ResourceUsage,
)

from spatialcf.domain.predicates import (
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ImplementationRegistrySnapshot,
    ResourcePolicy,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    ImplementationResolutionError,
    SemanticContractError,
    _CLAIM_CHECKER_CAPABILITY_FIELD,
    _CLAIM_PROOF_MATERIAL_FIELD,
    _COMPLETE_DOMAIN_CLAIM_FIELD,
    _PROOF_CHECKER_CAPABILITY_FIELD,
    _PROOF_PAYLOAD_SCHEMA_FIELD,
    _RECORD_BOUND_REFERENCE_FIELD,
    _RECORD_BOUND_SHA256_FIELD,
    _RECORD_KIND_FIELD,
    _RECORD_REFERENCE_FIELD,
    _RESOURCE_ACCOUNTING_CLAIM_FIELD,
    _ROLE_CERTIFIED_SOLUTION,
    _ROLE_COMPLETE_DOMAIN,
    _ROLE_PROOF_MATERIAL,
    _ROLE_PROVEN_UNSAT,
    _ROLE_RECORD_BINDING,
    _ROLE_RESOURCE_ACCOUNTING,
    _ROLE_RESOURCE_POLICY,
    _ROLE_SOUND_COMPLETE_DOMAIN,
    _ref_key,
    _self_digest_matches,
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


class DefinitionsValidation:
    """Stateless validation methods composed by StaticImplementationRegistry."""

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


# Keep public class, exception and bound-method lookup stable.
_walk_values.__module__ = "spatialcf.core.registry"
_references.__module__ = "spatialcf.core.registry"
_schema_dependency_refs.__module__ = "spatialcf.core.registry"
_semantic_schema_definition_closure.__module__ = "spatialcf.core.registry"
_validate_typed_value.__module__ = "spatialcf.core.registry"
_definition_record_fields.__module__ = "spatialcf.core.registry"
_definition_record_reference.__module__ = "spatialcf.core.registry"
_definition_record_digest.__module__ = "spatialcf.core.registry"
_definition_record_role.__module__ = "spatialcf.core.registry"
_definition_roles.__module__ = "spatialcf.core.registry"
_definition_payload_dependency_refs.__module__ = "spatialcf.core.registry"
_root_bootstrap_anchor.__module__ = "spatialcf.core.registry"
_has_exact_root_bootstrap_shape.__module__ = "spatialcf.core.registry"
_definition_root_maps.__module__ = "spatialcf.core.registry"
_definition_record_bindings.__module__ = "spatialcf.core.registry"
_validate_exact_root_definition_closure.__module__ = "spatialcf.core.registry"
_require_definition_role.__module__ = "spatialcf.core.registry"
_require_outcome_definition_role.__module__ = "spatialcf.core.registry"
_validate_resource_usage.__module__ = "spatialcf.core.registry"
DefinitionsValidation._validate_snapshot.__module__ = "spatialcf.core.registry"
DefinitionsValidation._validate_snapshot.__qualname__ = "StaticImplementationRegistry._validate_snapshot"
DefinitionsValidation._validate_definition_hashes.__module__ = "spatialcf.core.registry"
DefinitionsValidation._validate_definition_hashes.__qualname__ = "StaticImplementationRegistry._validate_definition_hashes"
DefinitionsValidation._validate_definition_cycles.__module__ = "spatialcf.core.registry"
DefinitionsValidation._validate_definition_cycles.__qualname__ = "StaticImplementationRegistry._validate_definition_cycles"
DefinitionsValidation.validate_definition_closure.__module__ = "spatialcf.core.registry"
DefinitionsValidation.validate_definition_closure.__qualname__ = "StaticImplementationRegistry.validate_definition_closure"
