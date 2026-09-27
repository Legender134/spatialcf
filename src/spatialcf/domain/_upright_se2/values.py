"""Upright profile values contracts and intrinsic operations."""

from __future__ import annotations

from spatialcf.domain.base import (
    CanonicalId,
)

from spatialcf.domain.definitions import (
    CanonicalIdValue,
    CapabilityRef,
    DefinitionRef,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
    UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
    UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
    UPRIGHT_SE2_PREDICATE_DEFINITION_REFS,
    UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
    UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
    _SEMANTIC_BODY_VALUES,
    _SEMANTIC_ID_SCHEMA_REF,
    _SEMANTIC_KIND_TO_REF,
    _SEMANTIC_KIND_TO_SCHEMA,
    _SEMANTIC_REAL_SCHEMA_REF,
    _SEMANTIC_SYMBOL_SCHEMA_REF,
    _SEMANTIC_TUPLE_SCHEMA_REF,
)


def _semantic_definition_ref_for_kind(kind: str) -> DefinitionRef:
    return _SEMANTIC_KIND_TO_REF[kind]


def _semantic_definition_schema_for_kind(kind: str) -> CanonicalId:
    return _SEMANTIC_KIND_TO_SCHEMA[kind]


def _semantic_capabilities_for_definition(
    definition_ref: DefinitionRef,
) -> tuple[CapabilityRef, CapabilityRef]:
    if definition_ref == UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF:
        return (
            UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        )
    if definition_ref in UPRIGHT_SE2_PREDICATE_DEFINITION_REFS:
        return (
            UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
        )
    raise ValueError(
        "semantic owner binding must name a registered semantic definition"
    )


def _semantic_typed_id(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_ID_SCHEMA_REF,
        payload=CanonicalIdValue(value=value),
    )


def _semantic_typed_symbol(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_SYMBOL_SCHEMA_REF,
        payload=EnumSymbolValue(symbol=value),
    )


def _semantic_typed_real(value: float) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_REAL_SCHEMA_REF,
        payload=FiniteRealValue(value=value),
    )


def _semantic_typed_digest(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref="schema:spatialcf/upright-se2/digest/1.0",
        payload=DigestValue(value=value),
    )


def _semantic_typed_tuple(values: tuple[str, ...]) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_TUPLE_SCHEMA_REF,
        payload=FiniteOrderedTupleValue(
            element_schema_ref=_SEMANTIC_ID_SCHEMA_REF,
            items=tuple(_semantic_typed_id(value) for value in values),
        ),
    )


def _semantic_typed_digest_tuple(values: tuple[str, ...]) -> TypedValue:
    return TypedValue(
        value_schema_ref="schema:spatialcf/upright-se2/semantic-digest-tuple/1.0",
        payload=FiniteOrderedTupleValue(
            element_schema_ref="schema:spatialcf/upright-se2/digest/1.0",
            items=tuple(_semantic_typed_digest(value) for value in values),
        ),
    )


def _semantic_typed_record(
    schema_ref: str,
    fields: tuple[tuple[str, TypedValue], ...],
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=canonical_json_bytes,
                )
            )
        ),
    )


def _semantic_definition_body(kind: str) -> TypedValue:
    schema_ref = _semantic_definition_schema_for_kind(kind)
    return _semantic_typed_record(
        schema_ref,
        (
            ("definition_kind", _semantic_typed_symbol(kind)),
            (
                "operand_schema_refs",
                _semantic_typed_tuple(_semantic_operand_schemas(kind)),
            ),
            *(
                (name, _semantic_typed_id(value))
                for name, value in _SEMANTIC_BODY_VALUES[kind]
            ),
        ),
    )


def _semantic_operand_schemas(kind: str) -> tuple[str, ...]:
    if kind == "TARGET_RELATION":
        return (
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/relation-symbol/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    if kind == "PRESERVATION":
        return (
            "schema:spatialcf/upright-se2/entity-ref/1.0",
            "schema:spatialcf/upright-se2/preservation-selector/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    if kind == "VISIBILITY":
        return (
            "schema:spatialcf/upright-se2/camera-ref/1.0",
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/visibility-metric-symbol/1.0",
            "schema:spatialcf/upright-se2/observation-ref/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    return ()


def _materialized_state_record(
    schema_ref: str, fields: tuple[tuple[str, TypedValue], ...]
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


# Keep supported public import and pickle lookup stable.
_semantic_definition_ref_for_kind.__module__ = "spatialcf.domain.upright_se2"
_semantic_definition_schema_for_kind.__module__ = "spatialcf.domain.upright_se2"
_semantic_capabilities_for_definition.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_id.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_symbol.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_real.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_digest.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_tuple.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_digest_tuple.__module__ = "spatialcf.domain.upright_se2"
_semantic_typed_record.__module__ = "spatialcf.domain.upright_se2"
_semantic_definition_body.__module__ = "spatialcf.domain.upright_se2"
_semantic_operand_schemas.__module__ = "spatialcf.domain.upright_se2"
_materialized_state_record.__module__ = "spatialcf.domain.upright_se2"
