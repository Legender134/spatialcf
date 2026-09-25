"""Strict additive wires for direct-GeneralIR ``semantic_place@1``.

Geometry, selection and proof checking deliberately live outside this module.
The interior box is centered in its explicit cavity frame, like UprightBox3DV2.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StrictInt, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel, FiniteFloat, RigidTransformV2, Sha256Digest, Vec3
from spatialcf.domain.counterfactual import EditProgram
from spatialcf.domain.definitions import (
    BooleanValue, CanonicalIdValue, EnumSymbolValue, FiniteOrderedTupleValue, FiniteRealValue, HashBoundCanonicalModel,
    IntegerValue, NamedTypedValue, RecordValue, TypedValue,
    ValueFieldDefinition, ValueKind, ValueSchemaDefinition,
)
from spatialcf.domain.operators import StateVariableRef
from spatialcf.domain.serialization import canonical_sha256
from spatialcf.domain.upright_se2 import ExactDyadic

PROFILE_REF = "spatialcf/semantic_place@1"
PREFIX = "spatialcf/semantic-place/1.0"
HASH_PREFIX = "spatialcf/counterfactual/semantic-place/1.0"


def definition(name: str) -> str:
    return f"definition:{PREFIX}/{name}"


def schema(name: str) -> str:
    return f"schema:{PREFIX}/{name}"


def capability(name: str) -> str:
    return f"capability:{PREFIX}/{name}"


def canonical_numbers(value: object) -> None:
    """Reject negative zero before canonical JSON could erase its sign."""
    if isinstance(value, CanonicalModel):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        for child in value.values():
            canonical_numbers(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            canonical_numbers(child)
    elif isinstance(value, float):
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            raise ValueError("semantic_place numbers must be finite and not negative zero")


def dyadic(value: Fraction) -> ExactDyadic:
    return ExactDyadic(numerator=value.numerator, denominator=value.denominator)


class SemanticPlaceProfileRegistration(CanonicalModel):
    profile_ref: Literal["spatialcf/semantic_place@1"] = PROFILE_REF
    operators: tuple[Literal["PLACE_IN"], Literal["PLACE_ON"]] = ("PLACE_IN", "PLACE_ON")
    derived_rule_refs: tuple[()] = ()
    coordinate_system: Literal["RH_METERS_Z_UP"] = "RH_METERS_Z_UP"
    endpoint_semantics: Literal["FIXED_IDENTITY_CLOSED_CONTACT"] = "FIXED_IDENTITY_CLOSED_CONTACT"


class SemanticPlaceInterval(CanonicalModel):
    lower: FiniteFloat
    upper: FiniteFloat
    closure: Literal["CLOSED"] = "CLOSED"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        canonical_numbers(self)
        if self.lower > self.upper:
            raise ValueError("closed interval must be ordered")
        return self


class SemanticPlaceCavityFact(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/cavity"
    SELF_DIGEST_FIELD: ClassVar[str] = "cavity_sha256"
    cavity_id: CanonicalId
    owner_object_id: CanonicalId
    anchor_from_cavity: RigidTransformV2
    interior_size_m: Vec3
    bottom_support_surface_id: CanonicalId
    shell_body_ids: tuple[CanonicalId, ...]
    unit: Literal["METRE"] = "METRE"
    frame: Literal["OWNER_LOCAL"] = "OWNER_LOCAL"
    cavity_sha256: Sha256Digest

    @model_validator(mode="after")
    def _shape(self) -> Self:
        canonical_numbers(self)
        if any(v <= 0.0 for v in self.interior_size_m.model_dump().values()):
            raise ValueError("cavity interior must have positive volume")
        if not self.shell_body_ids or self.shell_body_ids != tuple(sorted(set(self.shell_body_ids))):
            raise ValueError("shell body roster must be nonempty, sorted and unique")
        return self


class SemanticPlaceCavityInventory(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/cavity-inventory"
    SELF_DIGEST_FIELD: ClassVar[str] = "inventory_sha256"
    cavity_addresses: tuple[CanonicalId, ...]
    completeness: Literal["EXACT"] = "EXACT"
    inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def _roster(self) -> Self:
        if self.cavity_addresses != tuple(sorted(set(self.cavity_addresses))):
            raise ValueError("cavity inventory must contain sorted unique full addresses")
        return self


class SemanticPlaceDomain(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/domain"
    SELF_DIGEST_FIELD: ClassVar[str] = "domain_sha256"
    subject_id: CanonicalId
    operation: Literal["PLACE_ON", "PLACE_IN"]
    x: SemanticPlaceInterval
    y: SemanticPlaceInterval
    z: SemanticPlaceInterval
    target_ids: tuple[CanonicalId, ...]
    support_surface_ids: tuple[CanonicalId, ...]
    authorized_leaves: tuple[StateVariableRef, ...] = Field(min_length=4, max_length=4)
    support_margin_m: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    lateral_margin_m: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    top_margin_m: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    unit: Literal["METRE"] = "METRE"
    frame: Literal["WORLD"] = "WORLD"
    domain_sha256: Sha256Digest

    @model_validator(mode="after")
    def _authority(self) -> Self:
        canonical_numbers(self)
        for ids in (self.target_ids, self.support_surface_ids):
            if not ids or ids != tuple(sorted(set(ids))):
                raise ValueError("authorization must contain sorted unique nonempty IDs")
        if len(set(self.authorized_leaves)) != 4:
            raise ValueError("exactly four distinct scalar leaf addresses are required")
        return self


class SemanticPlaceLimits(CanonicalModel):
    numeric_bits: Annotated[StrictInt, Field(gt=0)] = 4096
    max_targets: Annotated[StrictInt, Field(ge=0)] = 64
    max_strata: Annotated[StrictInt, Field(ge=0)] = 100000
    compile_operations: Annotated[StrictInt, Field(ge=0)] = 1000000
    solve_operations: Annotated[StrictInt, Field(ge=0)] = 1000000
    check_operations: Annotated[StrictInt, Field(ge=0)] = 3000000


class SemanticPlaceExactBox(CanonicalModel):
    lower: tuple[ExactDyadic, ExactDyadic, ExactDyadic]
    upper: tuple[ExactDyadic, ExactDyadic, ExactDyadic]

    @model_validator(mode="after")
    def _positive(self) -> Self:
        if any(a.as_fraction >= b.as_fraction for a, b in zip(self.lower, self.upper, strict=True)):
            raise ValueError("exact box must have positive volume")
        return self


class SemanticPlaceTarget(CanonicalModel):
    target_id: CanonicalId
    support_surface_id: CanonicalId
    height: ExactDyadic
    xy_bounds: tuple[ExactDyadic, ExactDyadic, ExactDyadic, ExactDyadic] | None
    forbidden_open_rectangles: tuple[tuple[ExactDyadic, ExactDyadic, ExactDyadic, ExactDyadic], ...]
    obstacle_geometry_ids: tuple[CanonicalId, ...]
    infeasibility_reason: Literal["NONE", "XY_EMPTY", "Z_BOUNDS", "CAVITY_TOP"]


class SemanticPlaceLedger(CanonicalModel):
    stage: Literal["COMPILE", "SOLVE", "CHECK"]
    operations: Annotated[StrictInt, Field(ge=0)]
    replay_operations: Annotated[StrictInt, Field(ge=0)] = 0
    peak_numeric_bits: Annotated[StrictInt, Field(ge=0)] = 0
    completed_items: tuple[CanonicalId, ...]
    first_unprocessed_item: CanonicalId | None
    reason: Literal["NONE", "RESOURCE_LIMIT", "NUMERIC_GAP", "UNSUPPORTED", "INVALID"] = "NONE"


class SemanticPlaceCompilation(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/compilation"
    SELF_DIGEST_FIELD: ClassVar[str] = "compilation_sha256"
    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    domain: SemanticPlaceDomain
    subject_local_box: SemanticPlaceExactBox
    before_xyz: tuple[ExactDyadic, ExactDyadic, ExactDyadic]
    targets: tuple[SemanticPlaceTarget, ...]
    cavity_facts: tuple[SemanticPlaceCavityFact, ...]
    source_fact_sha256: Sha256Digest
    limits: SemanticPlaceLimits
    ledger: SemanticPlaceLedger
    compilation_sha256: Sha256Digest


class SemanticPlaceStratum(CanonicalModel):
    target_id: CanonicalId
    x: tuple[ExactDyadic, ExactDyadic]
    y: tuple[ExactDyadic, ExactDyadic]
    dimension: Literal[0, 1, 2]
    obstacle_index: StrictInt | None
    minimum_xyz: tuple[ExactDyadic, ExactDyadic, ExactDyadic] | None
    minimum_cost: ExactDyadic | None


class SemanticPlaceCoverage(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/coverage"
    SELF_DIGEST_FIELD: ClassVar[str] = "coverage_sha256"
    target_ids: tuple[CanonicalId, ...]
    strata: tuple[SemanticPlaceStratum, ...]
    winner_target_id: CanonicalId | None
    winner_xyz: tuple[ExactDyadic, ExactDyadic, ExactDyadic] | None
    winner_cost: ExactDyadic | None
    ledger: SemanticPlaceLedger
    coverage_sha256: Sha256Digest


class SemanticPlaceTruthRow(CanonicalModel):
    cavity_id: CanonicalId
    before: bool
    after: bool


class SemanticPlaceProofMaterial(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = HASH_PREFIX + "/proof-material"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_place_proof_sha256"
    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    compilation: SemanticPlaceCompilation | None
    coverage: SemanticPlaceCoverage | None
    program: EditProgram | None
    containment: tuple[SemanticPlaceTruthRow, ...]
    transition_ledger: SemanticPlaceLedger | None
    failure_ledger: SemanticPlaceLedger | None
    semantic_place_proof_sha256: Sha256Digest


def encode_value(value: object) -> TypedValue:
    """Lossless closed typed-value tree, including heterogeneous tuples.

    Tuples use indexed record fields, not an unchecked ANY element schema.
    Each record schema is identified by its complete ordered field schemas.
    Domain intervals and sets have dedicated encoders in static composition.
    """
    if isinstance(value, CanonicalModel):
        value = value.model_dump(mode="python")
    canonical_numbers(value)
    if value is None:
        return TypedValue(value_schema_ref=schema("none"), payload=RecordValue(fields=()))
    if isinstance(value, bool):
        return TypedValue(value_schema_ref=schema("boolean"), payload=BooleanValue(value=value))
    if isinstance(value, int):
        return TypedValue(value_schema_ref=schema("integer"), payload=IntegerValue(value=value))
    if isinstance(value, float):
        return TypedValue(value_schema_ref=schema("real"), payload=FiniteRealValue(value=value))
    if isinstance(value, str):
        return TypedValue(value_schema_ref=schema("id"), payload=CanonicalIdValue(value=value))
    tuple_value = isinstance(value, tuple)
    if tuple_value:
        value = {f"item:{index:08d}": item for index, item in enumerate(value)}
    if isinstance(value, dict):
        fields = tuple(NamedTypedValue(name=name, value=encode_value(child)) for name, child in sorted(value.items()))
        shape = tuple((f.name, f.value.value_schema_ref) for f in fields)
        tag = "tuple" if tuple_value else "record"
        return TypedValue(value_schema_ref=schema(tag + "/" + canonical_sha256(shape, domain=HASH_PREFIX + "/value-shape")), payload=RecordValue(fields=fields))
    raise TypeError("only canonical scalar, record and tuple values may enter a proof")


def decode_value(value: TypedValue) -> object:
    payload = value.payload
    if value.value_schema_ref == schema("none"):
        if payload != RecordValue(fields=()):
            raise ValueError("invalid none wire")
        return None
    if isinstance(payload, RecordValue):
        fields = {field.name: decode_value(field.value) for field in payload.fields}
        if value.value_schema_ref.startswith(schema("tuple/")):
            result = tuple(fields.values())
        else:
            result = fields
    elif isinstance(payload, (BooleanValue, IntegerValue, FiniteRealValue, CanonicalIdValue)):
        result = payload.value
    else:
        raise ValueError("unexpected semantic_place value kind")
    if encode_value(result) != value:
        raise ValueError("typed value schema or tuple fields do not match canonical content")
    return result


def proof_value_schemas() -> tuple[ValueSchemaDefinition, ...]:
    """Finite, closed recursive schema graph for the M5 structural proof wire.

    A tagged record represents exactly one scalar, tuple or record. Generic
    schema validation closes every edge; the proof decoder additionally checks
    tag-specific field presence and the complete SemanticPlaceProofMaterial.
    """
    fields = {'kind':'proof-kind','boolean':'boolean','integer':'integer','real':'real',
              'string':'id','children':'proof-children','names':'proof-names'}
    return (
        ValueSchemaDefinition.seal(value_schema_ref=schema('proof-material'),value_kind=ValueKind.RECORD,
            fields=tuple(ValueFieldDefinition(field_name=k,value_schema_ref=schema(v),required=k=='kind') for k,v in sorted(fields.items()))),
        ValueSchemaDefinition.seal(value_schema_ref=schema('proof-kind'),value_kind=ValueKind.ENUM_SYMBOL,
            enum_symbols=('BOOLEAN','INTEGER','NONE','REAL','RECORD','STRING','TUPLE')),
        ValueSchemaDefinition.seal(value_schema_ref=schema('proof-children'),value_kind=ValueKind.FINITE_ORDERED_TUPLE,element_schema_ref=schema('proof-material')),
        ValueSchemaDefinition.seal(value_schema_ref=schema('proof-names'),value_kind=ValueKind.FINITE_ORDERED_TUPLE,element_schema_ref=schema('id')),
    )


def _proof_node(value) -> TypedValue:
    if isinstance(value, CanonicalModel):
        value=value.model_dump(mode='python')
    fields={}
    if value is None:
        kind='NONE'
    elif isinstance(value,bool):
        kind='BOOLEAN';fields['boolean']=encode_value(value)
    elif isinstance(value,int):
        kind='INTEGER';fields['integer']=encode_value(value)
    elif isinstance(value,float):
        kind='REAL';fields['real']=encode_value(value)
    elif isinstance(value,str):
        kind='STRING';fields['string']=encode_value(value)
    elif isinstance(value,(dict,tuple)):
        kind='RECORD' if isinstance(value,dict) else 'TUPLE'
        if kind=='RECORD':
            value=dict(sorted(value.items()))
            fields['names']=TypedValue(value_schema_ref=schema('proof-names'),payload=FiniteOrderedTupleValue(
                element_schema_ref=schema('id'),items=tuple(encode_value(k) for k in value)))
            value=tuple(value.values())
        fields['children']=TypedValue(value_schema_ref=schema('proof-children'),payload=FiniteOrderedTupleValue(
            element_schema_ref=schema('proof-material'),items=tuple(_proof_node(v) for v in value)))
    else:
        raise TypeError('unsupported proof scalar')
    fields['kind']=TypedValue(value_schema_ref=schema('proof-kind'),payload=EnumSymbolValue(symbol=kind))
    return TypedValue(value_schema_ref=schema('proof-material'),payload=RecordValue(
        fields=tuple(NamedTypedValue(name=k,value=v) for k,v in sorted(fields.items()))))


def encode_proof(proof: SemanticPlaceProofMaterial) -> TypedValue:
    if type(proof) is not SemanticPlaceProofMaterial:
        raise TypeError('expected exact SemanticPlaceProofMaterial')
    return _proof_node(proof)


def _decode_proof_node(node):
    if node.value_schema_ref!=schema('proof-material') or not isinstance(node.payload,RecordValue):
        raise ValueError('wrong proof node schema')
    fields={f.name:f.value for f in node.payload.fields}
    try:
        kind=fields['kind'].payload.symbol
        if kind=='NONE':value=None
        elif kind in ('BOOLEAN','INTEGER','REAL','STRING'):
            value=decode_value(fields[kind.lower()])
        elif kind in ('TUPLE','RECORD'):
            value=tuple(_decode_proof_node(v) for v in fields['children'].payload.items)
            if kind=='RECORD':
                names=tuple(decode_value(v) for v in fields['names'].payload.items)
                if len(set(names))!=len(names):raise ValueError('duplicate proof field')
                value=dict(zip(names,value,strict=True))
        else:raise ValueError('unknown proof node tag')
    except (KeyError,AttributeError,TypeError) as error:
        raise ValueError('invalid tagged proof node') from error
    # One root re-encoding in decode_proof checks all node schemas and fields.
    return value


def decode_proof(value: TypedValue) -> SemanticPlaceProofMaterial:
    """Restore scene enums, then reject every wire-changing coercion."""
    proof=SemanticPlaceProofMaterial.model_validate(_decode_proof_node(value),strict=False)
    if encode_proof(proof)!=value:
        raise ValueError('proof decoding changed canonical typed content')
    return proof
