"""Static registry state validation and explicit dependencies."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ValidationError,
)

from spatialcf.domain.counterfactual import (
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    TypedValue,
    ValueKind,
    ValueSchemaDefinition,
    canonical_json_bytes,
)

from spatialcf.domain.operators import (
    StateVariableDefinition,
    StateVariableRef,
)

from spatialcf.domain.scene import (
    CanonicalScene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    SemanticContractError,
    _EXTENSION_FACT_ADDRESS_DOMAIN,
    _FROZEN_LEAF_PATH_DOMAIN,
    _STATE_FRAME_FIELD,
    _STATE_METADATA_FIELDS,
    _STATE_TOPOLOGY_FIELD,
    _STATE_UNIT_FIELD,
    _self_digest_matches,
    _state_key,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_record_fields,
    _definition_record_role,
    _validate_typed_value,
)


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


# Keep public class, exception and bound-method lookup stable.
_state_definition_metadata.__module__ = "spatialcf.core.registry"
_state_definition_semantics.__module__ = "spatialcf.core.registry"
_raw_value_matches_schema.__module__ = "spatialcf.core.registry"
_validate_state_leaf_value_wires.__module__ = "spatialcf.core.registry"
_base_fact_rows.__module__ = "spatialcf.core.registry"
_frozen_scalar_values.__module__ = "spatialcf.core.registry"
_frozen_leaf_token.__module__ = "spatialcf.core.registry"
_extension_fact_address.__module__ = "spatialcf.core.registry"
_extension_facts_by_address.__module__ = "spatialcf.core.registry"
_state_leaf_owners.__module__ = "spatialcf.core.registry"
_declared_extension_leaf_owners.__module__ = "spatialcf.core.registry"
_expected_base_leaf_addresses.__module__ = "spatialcf.core.registry"
_validated_base_scene_payload.__module__ = "spatialcf.core.registry"
_leaf_values.__module__ = "spatialcf.core.registry"
_validate_scene_state_envelope.__module__ = "spatialcf.core.registry"
