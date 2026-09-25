"""Strict exact values for the direct General IR rigid SE(3) profile.

This module owns declarative wires. Geometry, search, and proof replay belong
to the M6 core owners.
"""

from __future__ import annotations

import base64
import math
from enum import Enum
from fractions import Fraction
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from spatialcf.domain.base import CanonicalId, CanonicalModel, Sha256Digest
from spatialcf.domain.counterfactual import EditProgram, SceneStateEnvelope
from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalIdValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntegerValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
    ValueFieldDefinition,
    ValueKind,
    ValueSchemaDefinition,
)
from spatialcf.domain.operators import StateDeltaManifest, StateVariableRef
from spatialcf.domain.serialization import canonical_json_bytes

PROFILE_REF = "spatialcf/rigid_se3_multi@1"
PREFIX = "spatialcf/rigid-se3/1.0"
INGRESS_POLICY_REF = f"{PREFIX}/ingress@1"
MAX_INGRESS_INTEGER_BITS = 4096
MAX_INGRESS_NODES = 1_000_000
MAX_INGRESS_COLLECTION_ITEMS = 100_000
MAX_INGRESS_DEPTH = 128
MAX_MODEL_SCALAR_BYTES = 64 * 1024 * 1024
MAX_REQUEST_SCALAR_BYTES = 16 * 1024 * 1024
MAX_PROOF_CHUNKS = 100_000
MAX_PROOF_BYTES = 36_000_000
MAX_COVERAGE_ROWS = 4096
MAX_COVERAGE_BYTES = 8 * 1024 * 1024
MAX_TUPLE_OUTPUT_BOUND_BYTES = 16 * 1024 * 1024
MAX_PROOF_OUTPUT_BOUND_BYTES = 64 * 1024 * 1024
MAX_WITNESS_HINTS = 64
MAX_SWAP_CLAIMS = 64
MAX_LEDGER_EVENTS = 20_000
MAX_COMPILATION_BYTES = 8 * 1024 * 1024
MAX_STEP_TRACE_BYTES = 8 * 1024 * 1024
MAX_PROGRAM_TRACE_BYTES = 16 * 1024 * 1024
INGRESS_POLICY = (
    INGRESS_POLICY_REF, MAX_INGRESS_INTEGER_BITS, MAX_INGRESS_NODES,
    MAX_INGRESS_COLLECTION_ITEMS, MAX_INGRESS_DEPTH, MAX_MODEL_SCALAR_BYTES,
    MAX_REQUEST_SCALAR_BYTES, MAX_PROOF_CHUNKS, MAX_PROOF_BYTES,
    MAX_COVERAGE_ROWS, MAX_COVERAGE_BYTES, MAX_WITNESS_HINTS, MAX_SWAP_CLAIMS,
    MAX_LEDGER_EVENTS, MAX_COMPILATION_BYTES, MAX_STEP_TRACE_BYTES,
    MAX_PROGRAM_TRACE_BYTES, MAX_TUPLE_OUTPUT_BOUND_BYTES,
    MAX_PROOF_OUTPUT_BOUND_BYTES,
)


class M6IngressLimit(RuntimeError):
    """Bounded operational refusal; never a geometric infeasibility result."""


def preflight_m6_ingress(value: object, *, scalar_bytes: int = MAX_MODEL_SCALAR_BYTES) -> tuple[int, int]:
    """Bound M6-owned validation before its exact model predicates execute.

    Count shared objects on every expanded occurrence. An active-path identity
    detects cycles without mistaking repeated immutable submodels for cycles.
    """
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    nodes = 0
    scalar_total = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(item))
            continue
        nodes += 1
        if nodes > MAX_INGRESS_NODES or depth > MAX_INGRESS_DEPTH:
            raise M6IngressLimit("M6 ingress node or depth ceiling exceeded")
        if isinstance(item, int):
            if abs(item).bit_length() > MAX_INGRESS_INTEGER_BITS:
                raise M6IngressLimit("M6 ingress integer width ceiling exceeded")
            scalar_total += len(str(item))
            if scalar_total > scalar_bytes:
                raise M6IngressLimit("M6 ingress scalar byte ceiling exceeded")
            continue
        if isinstance(item, str):
            if len(item) > scalar_bytes:
                raise M6IngressLimit("M6 ingress scalar byte ceiling exceeded")
            size = len(item.encode("utf-8"))
            scalar_total += size
        elif isinstance(item, Enum):
            scalar_total += len(str(item.value).encode("utf-8"))
        elif isinstance(item, bytes):
            scalar_total += len(item)
        if scalar_total > scalar_bytes:
            raise M6IngressLimit("M6 ingress scalar byte ceiling exceeded")
        if isinstance(item, BaseModel):
            children = item.__dict__
        elif isinstance(item, (dict, tuple, list, set, frozenset)):
            children = item
        else:
            continue
        if len(children) > MAX_INGRESS_COLLECTION_ITEMS:
            raise M6IngressLimit("M6 ingress collection ceiling exceeded")
        identity = id(item)
        if identity in active:
            raise M6IngressLimit("M6 ingress cyclic input")
        active.add(identity)
        stack.append((item, depth, True))
        if isinstance(children, dict):
            for key, child in children.items():
                stack.append((child, depth + 1, False))
                stack.append((key, depth + 1, False))
        else:
            for child in children:
                stack.append((child, depth + 1, False))
    return nodes, scalar_total


def m6_output_upper_bound(value: object) -> int:
    """Conservative canonical allocation bound before materializing JSON."""
    nodes, scalar_bytes = preflight_m6_ingress(value)
    return 6 * scalar_bytes + 64 * nodes


class _M6IngressGuard:
    @classmethod
    def _preflight(cls, value: object) -> object:
        key, limit = {
            "RigidSE3Coverage": ("tuples", MAX_COVERAGE_ROWS),
            "RigidSE3ProposalPolicyPayload": ("witness_hints", MAX_WITNESS_HINTS),
            "RigidSE3ProofMaterial": ("swap_evidence", MAX_SWAP_CLAIMS),
            "RigidSE3Ledger": ("events", MAX_LEDGER_EVENTS),
        }.get(cls.__name__, (None, None))
        if key is not None:
            rows = (value.get(key, ()) if isinstance(value, dict)
                    else getattr(value, key, ()))
            if len(rows) > limit:
                raise M6IngressLimit(f"M6 ingress {key} ceiling exceeded")
        output_limit = {
            "RigidSE3Ledger": MAX_COMPILATION_BYTES,
            "RigidSE3Compilation": MAX_COMPILATION_BYTES,
            "RigidSE3StepTrace": MAX_STEP_TRACE_BYTES,
            "RigidSE3ProgramTrace": MAX_PROGRAM_TRACE_BYTES,
            "RigidSE3TupleEvidence": MAX_TUPLE_OUTPUT_BOUND_BYTES,
            "RigidSE3ProofMaterial": MAX_PROOF_OUTPUT_BOUND_BYTES,
        }.get(cls.__name__)
        if output_limit is not None and m6_output_upper_bound(value) > output_limit:
            raise M6IngressLimit(f"M6 ingress {cls.__name__} output ceiling exceeded")
        preflight_m6_ingress(value)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Keep Pydantic's strict JSON-mode tuple/enum semantics. A normal
        # before-validator switches parsed JSON into Python mode and rejects
        # otherwise canonical proof arrays. Guard only the Python entry.
        if issubclass(cls, HashBoundCanonicalModel):
            schema = HashBoundCanonicalModel.__get_pydantic_core_schema__.__func__(
                cls, source_type, handler,
            )
        else:
            schema = handler(source_type)
        original_ref = schema.get("ref")
        if original_ref:
            schema = {**schema, "ref": f"{original_ref}:m6-ingress-inner"}
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=core_schema.no_info_before_validator_function(
                cls._preflight, schema,
            ),
            ref=original_ref,
        )


def _bounded_m6_json(data: str | bytes | bytearray) -> None:
    if len(data) > MAX_MODEL_SCALAR_BYTES:
        raise M6IngressLimit("M6 ingress JSON byte ceiling exceeded")
    if isinstance(data, str) and len(data.encode("utf-8")) > MAX_MODEL_SCALAR_BYTES:
        raise M6IngressLimit("M6 ingress JSON byte ceiling exceeded")
    # Scan the bounded wire without materializing a second JSON tree. Pydantic
    # remains the syntax/canonical validator; this pass only refuses costly
    # structures before its exact and hash-bound model validators run.
    frames: list[list[int | str]] = []
    nodes = 0
    inside_string = False
    escaped = False
    string_bytes = 0
    token_length = 0
    in_number = False
    for character in data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data:
        if inside_string:
            string_bytes += 1
            if string_bytes > MAX_MODEL_SCALAR_BYTES:
                raise M6IngressLimit("M6 ingress JSON scalar ceiling exceeded")
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                inside_string = False
            continue
        if character in "-0123456789":
            if not in_number:
                if len(frames) > MAX_INGRESS_DEPTH:
                    raise M6IngressLimit("M6 ingress JSON depth ceiling exceeded")
                in_number = True
                token_length = 0
                nodes += 1
            if character != "-":
                token_length += 1
            if token_length > 1234:
                raise M6IngressLimit("M6 ingress JSON number width ceiling exceeded")
            continue
        if in_number and character in ".eE+":
            token_length += 1
            if token_length > 1234:
                raise M6IngressLimit("M6 ingress JSON number width ceiling exceeded")
            continue
        in_number = False
        if character == '"':
            if len(frames) > MAX_INGRESS_DEPTH:
                raise M6IngressLimit("M6 ingress JSON depth ceiling exceeded")
            inside_string = True
            string_bytes = 0
            nodes += 1
        elif character in "{[":
            frames.append([character, 0, 0])
            nodes += 1
            if len(frames) - 1 > MAX_INGRESS_DEPTH:
                raise M6IngressLimit("M6 ingress JSON depth ceiling exceeded")
        elif character in "}]":
            if frames:
                _opening, commas, nonempty = frames.pop()
                if nonempty and commas + 1 > MAX_INGRESS_COLLECTION_ITEMS:
                    raise M6IngressLimit("M6 ingress JSON collection ceiling exceeded")
        elif character == "," and frames:
            frames[-1][1] += 1
            if frames[-1][1] + 1 > MAX_INGRESS_COLLECTION_ITEMS:
                raise M6IngressLimit("M6 ingress JSON collection ceiling exceeded")
        elif character not in " \t\r\n:" and character in "tfn":
            if len(frames) > MAX_INGRESS_DEPTH:
                raise M6IngressLimit("M6 ingress JSON depth ceiling exceeded")
            nodes += 1
        if frames and character not in " \t\r\n}]":
            frames[-1][2] = 1
        if nodes > MAX_INGRESS_NODES:
            raise M6IngressLimit("M6 ingress JSON node ceiling exceeded")


class _M6CanonicalModel(_M6IngressGuard, CanonicalModel):
    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):
        _bounded_m6_json(json_data)
        return super().model_validate_json(json_data, *args, **kwargs)


class _M6HashBoundCanonicalModel(_M6IngressGuard, HashBoundCanonicalModel):
    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):
        _bounded_m6_json(json_data)
        return super().model_validate_json(json_data, *args, **kwargs)

    @classmethod
    def seal(cls, **values):
        cls._preflight(values)
        return super().seal(**values)


def schema(name: str) -> str:
    return f"schema:{PREFIX}/{name}"


def proof_value_schemas() -> tuple[ValueSchemaDefinition, ...]:
    """One fixed recursive closure, independent of proof or hint values."""
    fields = {
        "kind": "tree-kind", "boolean": "boolean", "integer": "integer",
        "real": "real",
        "string": "id", "children": "tree-children", "names": "tree-names",
    }
    return (
        ValueSchemaDefinition.seal(
            value_schema_ref=schema("tree"), value_kind=ValueKind.RECORD,
            fields=tuple(ValueFieldDefinition(field_name=name, value_schema_ref=schema(ref),
                                              required=name == "kind") for name, ref in sorted(fields.items())),
        ),
        ValueSchemaDefinition.seal(
            value_schema_ref=schema("tree-kind"), value_kind=ValueKind.ENUM_SYMBOL,
            enum_symbols=("BOOLEAN", "FINITE_REAL", "INTEGER", "NONE", "RECORD", "STRING", "TUPLE"),
        ),
        ValueSchemaDefinition.seal(value_schema_ref=schema("boolean"), value_kind=ValueKind.BOOLEAN),
        ValueSchemaDefinition.seal(value_schema_ref=schema("integer"), value_kind=ValueKind.INTEGER),
        ValueSchemaDefinition.seal(value_schema_ref=schema("real"), value_kind=ValueKind.FINITE_REAL),
        ValueSchemaDefinition.seal(value_schema_ref=schema("id"), value_kind=ValueKind.CANONICAL_ID),
        ValueSchemaDefinition.seal(value_schema_ref=schema("tree-children"),
                                   value_kind=ValueKind.FINITE_ORDERED_TUPLE, element_schema_ref=schema("tree")),
        ValueSchemaDefinition.seal(value_schema_ref=schema("tree-names"),
                                   value_kind=ValueKind.FINITE_ORDERED_TUPLE, element_schema_ref=schema("id")),
        ValueSchemaDefinition.seal(value_schema_ref=schema("proof-chunk"), value_kind=ValueKind.CANONICAL_ID),
        ValueSchemaDefinition.seal(value_schema_ref=schema("proof-bytes"),
                                   value_kind=ValueKind.FINITE_ORDERED_TUPLE,
                                   element_schema_ref=schema("proof-chunk")),
    )


def _typed(name: str, payload: object) -> TypedValue:
    return TypedValue(value_schema_ref=schema(name), payload=payload)


def _encode_tree(value: object, depth: int) -> TypedValue:
    if depth > 128:
        raise ValueError("M6 typed tree exceeds maximum nesting")
    if isinstance(value, CanonicalModel):
        value = value.model_dump(mode="python")
    fields: dict[str, TypedValue] = {}
    if value is None:
        kind = "NONE"
    elif isinstance(value, bool):
        kind = "BOOLEAN"
        fields["boolean"] = _typed("boolean", BooleanValue(value=value))
    elif isinstance(value, int):
        kind = "INTEGER"
        fields["integer"] = _typed("integer", IntegerValue(value=value))
    elif isinstance(value, float):
        if not math.isfinite(value) or (value == 0 and math.copysign(1.0, value) < 0):
            raise ValueError("M6 retained envelope float must be finite and not negative zero")
        kind = "FINITE_REAL"
        fields["real"] = _typed("real", FiniteRealValue(value=value))
    elif isinstance(value, str):
        kind = "STRING"
        fields["string"] = _typed("id", CanonicalIdValue(value=value))
    elif isinstance(value, (tuple, dict)):
        kind = "TUPLE" if isinstance(value, tuple) else "RECORD"
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("M6 typed record keys must be strings")
            names = tuple(sorted(value))
            fields["names"] = _typed(
                "tree-names", FiniteOrderedTupleValue(
                    element_schema_ref=schema("id"),
                    items=tuple(_typed("id", CanonicalIdValue(value=name)) for name in names),
                ),
            )
            children = tuple(value[name] for name in names)
        else:
            children = value
        fields["children"] = _typed(
            "tree-children", FiniteOrderedTupleValue(
                element_schema_ref=schema("tree"),
                items=tuple(_encode_tree(child, depth + 1) for child in children),
            ),
        )
    else:
        raise TypeError("M6 typed tree accepts only scalar, tuple and record values")
    fields["kind"] = _typed("tree-kind", EnumSymbolValue(symbol=kind))
    return _typed("tree", RecordValue(fields=tuple(
        NamedTypedValue(name=name, value=field) for name, field in sorted(fields.items())
    )))


def encode_tree(value: object) -> TypedValue:
    return _encode_tree(value, 0)


def _decode_tree(value: TypedValue, depth: int) -> object:
    if depth > 128 or value.value_schema_ref != schema("tree") or not isinstance(value.payload, RecordValue):
        raise ValueError("invalid M6 typed tree node")
    fields = {field.name: field.value for field in value.payload.fields}
    tag = fields.get("kind")
    if tag is None or tag.value_schema_ref != schema("tree-kind") or not isinstance(tag.payload, EnumSymbolValue):
        raise ValueError("M6 typed tree lacks a valid kind")
    kind = tag.payload.symbol
    expected = {
        "NONE": {"kind"},
        "BOOLEAN": {"kind", "boolean"},
        "INTEGER": {"kind", "integer"},
        "FINITE_REAL": {"kind", "real"},
        "STRING": {"kind", "string"},
        "TUPLE": {"kind", "children"},
        "RECORD": {"kind", "children", "names"},
    }.get(kind)
    if expected is None or set(fields) != expected:
        raise ValueError("M6 typed tree tag and fields disagree")
    if kind in ("BOOLEAN", "FINITE_REAL", "INTEGER", "STRING"):
        name, cls = {
            "BOOLEAN": ("boolean", BooleanValue),
            "FINITE_REAL": ("real", FiniteRealValue),
            "INTEGER": ("integer", IntegerValue),
            "STRING": ("string", CanonicalIdValue),
        }[kind]
        child = fields[name]
        if child.value_schema_ref != schema("id" if name == "string" else name) or not isinstance(child.payload, cls):
            raise ValueError("M6 scalar has wrong typed schema")
        return child.payload.value
    if kind == "NONE":
        return None
    children = fields["children"]
    if children.value_schema_ref != schema("tree-children") or not isinstance(children.payload, FiniteOrderedTupleValue) or children.payload.element_schema_ref != schema("tree"):
        raise ValueError("M6 tree children have wrong typed schema")
    values = tuple(_decode_tree(child, depth + 1) for child in children.payload.items)
    if kind == "TUPLE":
        return values
    names = fields["names"]
    if names.value_schema_ref != schema("tree-names") or not isinstance(names.payload, FiniteOrderedTupleValue) or names.payload.element_schema_ref != schema("id"):
        raise ValueError("M6 tree field names have wrong typed schema")
    keys = tuple(item.payload.value for item in names.payload.items if item.value_schema_ref == schema("id") and isinstance(item.payload, CanonicalIdValue))
    if len(keys) != len(names.payload.items) or keys != tuple(sorted(set(keys))) or len(keys) != len(values):
        raise ValueError("M6 record keys are missing, duplicated or unsorted")
    return dict(zip(keys, values, strict=True))


def decode_tree(value: TypedValue) -> object:
    preflight_m6_ingress(value)
    result = _decode_tree(value, 0)
    if encode_tree(result) != value:
        raise ValueError("M6 typed tree is not canonical")
    return result


class RigidSE3Rational(_M6CanonicalModel):
    """A reduced exact rational; arithmetic is done using Fraction."""

    numerator: StrictInt
    denominator: Annotated[StrictInt, Field(gt=0)]

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def _bounded_width(cls, value):
        if isinstance(value, int) and abs(value).bit_length() > MAX_INGRESS_INTEGER_BITS:
            raise M6IngressLimit("M6 ingress integer width ceiling exceeded")
        return value

    @model_validator(mode="after")
    def _reduced(self) -> Self:
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("rational must be reduced (zero is 0/1)")
        return self

    @property
    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @classmethod
    def from_fraction(cls, value: Fraction) -> Self:
        return cls(numerator=value.numerator, denominator=value.denominator)


class RigidSE3Vector(_M6CanonicalModel):
    x: RigidSE3Rational
    y: RigidSE3Rational
    z: RigidSE3Rational

    @property
    def fractions(self) -> tuple[Fraction, Fraction, Fraction]:
        return (self.x.as_fraction, self.y.as_fraction, self.z.as_fraction)


class RigidSE3Rotation(_M6CanonicalModel):
    """Row-major exact proper SO(3) matrix."""

    rows: tuple[
        tuple[RigidSE3Rational, RigidSE3Rational, RigidSE3Rational],
        tuple[RigidSE3Rational, RigidSE3Rational, RigidSE3Rational],
        tuple[RigidSE3Rational, RigidSE3Rational, RigidSE3Rational],
    ]

    @model_validator(mode="after")
    def _proper(self) -> Self:
        r = tuple(tuple(cell.as_fraction for cell in row) for row in self.rows)
        for i in range(3):
            for j in range(3):
                dot = sum((r[k][i] * r[k][j] for k in range(3)), Fraction())
                if dot != (1 if i == j else 0):
                    raise ValueError("rotation columns must be orthonormal")
        determinant = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        if determinant != 1:
            raise ValueError("rotation determinant must be +1")
        return self


class RigidSE3Pose(_M6CanonicalModel):
    rotation: RigidSE3Rotation
    translation_m: RigidSE3Vector
    frame: Literal["WORLD", "BODY_LOCAL", "JOINT_LOCAL"]
    unit: Literal["METRE"] = "METRE"


def _canonical_unique(values: tuple[CanonicalModel, ...], *, name: str) -> None:
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(set(encoded))):
        raise ValueError(f"{name} values must be sorted and unique")


class RigidSE3RootTranslationDomain(_M6CanonicalModel):
    kind: Literal["FINITE", "CLOSED_RATIONAL_BOX"]
    values: tuple[RigidSE3Vector, ...] = ()
    lower_m: RigidSE3Vector | None = None
    upper_m: RigidSE3Vector | None = None

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if self.kind == "FINITE":
            if not self.values or self.lower_m is not None or self.upper_m is not None:
                raise ValueError("finite translation requires only a nonempty value set")
            _canonical_unique(self.values, name="translation")
        else:
            if self.values or self.lower_m is None or self.upper_m is None:
                raise ValueError("continuous translation requires only closed box endpoints")
            if any(a > b for a, b in zip(self.lower_m.fractions, self.upper_m.fractions, strict=True)):
                raise ValueError("translation box is inverted")
        return self


class RigidSE3RootRotationDomain(_M6CanonicalModel):
    kind: Literal["FINITE", "ALL_SO3"]
    values: tuple[RigidSE3Rotation, ...] = ()

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if self.kind == "FINITE":
            if not self.values:
                raise ValueError("finite rotation domain is empty")
            _canonical_unique(self.values, name="rotation")
        elif self.values:
            raise ValueError("ALL_SO3 cannot carry finite sample values")
        return self


class RigidSE3PrismaticDomain(_M6CanonicalModel):
    kind: Literal["FINITE", "CLOSED_RATIONAL_INTERVAL"]
    values_m: tuple[RigidSE3Rational, ...] = ()
    lower_m: RigidSE3Rational | None = None
    upper_m: RigidSE3Rational | None = None

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if self.kind == "FINITE":
            if not self.values_m or self.lower_m is not None or self.upper_m is not None:
                raise ValueError("finite prismatic domain requires values only")
            if self.values_m != tuple(sorted(set(self.values_m), key=lambda q: q.as_fraction)):
                raise ValueError("prismatic values must be numerically sorted and unique")
        elif (
            self.values_m or self.lower_m is None or self.upper_m is None
            or self.lower_m.as_fraction > self.upper_m.as_fraction
        ):
            raise ValueError("prismatic interval must be closed and ordered")
        return self


class RigidSE3CirclePoint(_M6CanonicalModel):
    cosine: RigidSE3Rational
    sine: RigidSE3Rational

    @model_validator(mode="after")
    def _unit(self) -> Self:
        if self.cosine.as_fraction ** 2 + self.sine.as_fraction ** 2 != 1:
            raise ValueError("revolute state must lie on the exact unit circle")
        return self


class RigidSE3RevoluteDomain(_M6CanonicalModel):
    kind: Literal["FINITE", "FULL_CIRCLE"]
    values: tuple[RigidSE3CirclePoint, ...] = ()

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if self.kind == "FINITE":
            if not self.values:
                raise ValueError("finite revolute domain is empty")
            _canonical_unique(self.values, name="revolute")
        elif self.values:
            raise ValueError("FULL_CIRCLE cannot carry finite samples")
        return self


class RigidSE3Box(_M6CanonicalModel):
    primitive_id: CanonicalId
    local_anchor: RigidSE3Pose
    half_extents_m: RigidSE3Vector

    @model_validator(mode="after")
    def _positive(self) -> Self:
        if self.local_anchor.frame != "BODY_LOCAL":
            raise ValueError("box anchor must be body-local")
        if any(value <= 0 for value in self.half_extents_m.fractions):
            raise ValueError("box must have positive volume")
        return self


class RigidSE3Halfspace(_M6CanonicalModel):
    """Closed world-frame halfspace normal dot point <= upper_m."""

    normal: RigidSE3Vector
    upper_m: RigidSE3Rational

    @model_validator(mode="after")
    def _nonzero(self) -> Self:
        if all(value == 0 for value in self.normal.fractions):
            raise ValueError("region halfspace normal must be nonzero")
        return self


class RigidSE3RegionFact(_M6CanonicalModel):
    """Bounded closed rational region with explicit XYZ box planes."""

    region_id: CanonicalId
    lower_m: RigidSE3Vector
    upper_m: RigidSE3Vector
    halfspaces: tuple[RigidSE3Halfspace, ...]

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        if not self.region_id.startswith("region:"):
            raise ValueError("region fact requires a region ID")
        if any(lower > upper for lower, upper in zip(
            self.lower_m.fractions, self.upper_m.fractions, strict=True,
        )):
            raise ValueError("region XYZ bounds must be ordered")
        _canonical_unique(self.halfspaces, name="region halfspace")
        planes = {(row.normal.fractions, row.upper_m.as_fraction) for row in self.halfspaces}
        for axis, (lower, upper) in enumerate(zip(
            self.lower_m.fractions, self.upper_m.fractions, strict=True,
        )):
            positive = tuple(Fraction(int(index == axis)) for index in range(3))
            negative = tuple(-component for component in positive)
            if (positive, upper) not in planes or (negative, -lower) not in planes:
                raise ValueError("region requires all six exact XYZ bounding planes")
        return self


class RigidSE3BodyFact(_M6CanonicalModel):
    body_id: CanonicalId
    category_id: CanonicalId
    kind: Literal["ROOT", "CHILD"]
    solid: bool
    primitive_ids: tuple[CanonicalId, ...]
    may_move: bool
    may_change_contact: bool

    @model_validator(mode="after")
    def _roster(self) -> Self:
        if not self.body_id.startswith("entity:"):
            raise ValueError("M6 body must be an extension entity")
        if not self.category_id.startswith("category:"):
            raise ValueError("M6 body requires an exact category")
        if self.primitive_ids != tuple(sorted(set(self.primitive_ids))):
            raise ValueError("primitive roster must be sorted and unique")
        if self.solid and not self.primitive_ids:
            raise ValueError("solid body requires collision geometry")
        return self


class RigidSE3JointFact(_M6CanonicalModel):
    joint_id: CanonicalId
    parent_body_id: CanonicalId
    child_body_id: CanonicalId
    kind: Literal["FIXED", "PRISMATIC", "REVOLUTE"]
    parent_attachment: RigidSE3Pose
    child_attachment: RigidSE3Pose
    axis: RigidSE3Vector | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.parent_body_id == self.child_body_id:
            raise ValueError("joint cannot attach a body to itself")
        if self.parent_attachment.frame != "BODY_LOCAL" or self.child_attachment.frame != "JOINT_LOCAL":
            raise ValueError("joint attachment frames are explicit")
        if self.kind == "FIXED":
            if self.axis is not None:
                raise ValueError("fixed joint has no motion axis")
        elif self.axis is None or sum(x * x for x in self.axis.fractions) != 1:
            raise ValueError("moving joint axis must be an exact unit vector")
        return self


class RigidSE3ContactFact(_M6CanonicalModel):
    contact_id: CanonicalId
    first_body_id: CanonicalId
    first_primitive_id: CanonicalId
    first_face: Literal["PX", "NX", "PY", "NY", "PZ", "NZ"]
    second_body_id: CanonicalId
    second_primitive_id: CanonicalId
    second_face: Literal["PX", "NX", "PY", "NY", "PZ", "NZ"]
    release_gap_m: RigidSE3Rational

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.first_body_id == self.second_body_id:
            raise ValueError("contact requires distinct bodies")
        if self.release_gap_m.as_fraction <= 0:
            raise ValueError("release gap must be positive")
        return self


class RigidSE3FactCell(_M6CanonicalModel):
    """Explicit source availability; missing data never becomes a default."""

    availability: Literal["KNOWN_EXACT", "MISSING", "INEXACT"]
    value: TypedValue | None = None
    uncertainty_upper: RigidSE3Rational | None = None

    @model_validator(mode="after")
    def _branch(self) -> Self:
        if self.availability == "KNOWN_EXACT":
            if self.value is None or self.uncertainty_upper is not None:
                raise ValueError("exact fact requires one value and no uncertainty")
        elif self.availability == "MISSING":
            if self.value is not None or self.uncertainty_upper is not None:
                raise ValueError("missing fact cannot invent data")
        elif self.value is not None or self.uncertainty_upper is None or self.uncertainty_upper.as_fraction <= 0:
            raise ValueError("inexact fact requires declared positive uncertainty only")
        return self


class RigidSE3RootState(_M6CanonicalModel):
    body_id: CanonicalId
    world_pose: RigidSE3Pose

    @model_validator(mode="after")
    def _frame(self) -> Self:
        if self.world_pose.frame != "WORLD":
            raise ValueError("root source pose must be world framed")
        return self


class RigidSE3JointState(_M6CanonicalModel):
    joint_id: CanonicalId
    kind: Literal["PRISMATIC", "REVOLUTE"]
    offset_m: RigidSE3Rational | None = None
    circle_point: RigidSE3CirclePoint | None = None

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if (self.kind == "PRISMATIC") != (self.offset_m is not None):
            raise ValueError("prismatic state requires only an offset")
        if (self.kind == "REVOLUTE") != (self.circle_point is not None):
            raise ValueError("revolute state requires only a circle point")
        return self


class RigidSE3ContactState(_M6CanonicalModel):
    contact_id: CanonicalId
    mode: Literal["ENGAGED", "RELEASED"]


class RigidSE3ExactSourceFacts(_M6CanonicalModel):
    """A validated view of separate exact extension facts, never an IR carrier."""

    bodies: tuple[RigidSE3BodyFact, ...]
    boxes: tuple[RigidSE3Box, ...]
    joints: tuple[RigidSE3JointFact, ...]
    contacts: tuple[RigidSE3ContactFact, ...]
    roots: tuple[RigidSE3RootState, ...]
    joint_states: tuple[RigidSE3JointState, ...]
    contact_states: tuple[RigidSE3ContactState, ...]
    regions: tuple[RigidSE3RegionFact, ...] = ()

    @model_validator(mode="after")
    def _closed_graph(self) -> Self:
        for name, rows, key in (
            ("bodies", self.bodies, lambda item: item.body_id),
            ("boxes", self.boxes, lambda item: item.primitive_id),
            ("joints", self.joints, lambda item: item.joint_id),
            ("contacts", self.contacts, lambda item: item.contact_id),
            ("roots", self.roots, lambda item: item.body_id),
            ("joint states", self.joint_states, lambda item: item.joint_id),
            ("contact states", self.contact_states, lambda item: item.contact_id),
            ("regions", self.regions, lambda item: item.region_id),
        ):
            keys = tuple(key(row) for row in rows)
            if keys != tuple(sorted(set(keys))):
                raise ValueError(f"{name} must have sorted unique IDs")
        bodies = {body.body_id: body for body in self.bodies}
        boxes = {box.primitive_id: box for box in self.boxes}
        if not bodies:
            raise ValueError("M6 source must contain at least one body")
        primitive_owners = [primitive for body in self.bodies for primitive in body.primitive_ids]
        if len(primitive_owners) != len(set(primitive_owners)) or set(primitive_owners) != set(boxes):
            raise ValueError("every box must have exactly one body owner")
        roots = {body.body_id for body in self.bodies if body.kind == "ROOT"}
        children = set(bodies) - roots
        if {state.body_id for state in self.roots} != roots:
            raise ValueError("root source poses must cover exactly the roots")
        parent_of: dict[str, str] = {}
        for joint in self.joints:
            if joint.parent_body_id not in bodies or joint.child_body_id not in children:
                raise ValueError("joint references a missing or non-child body")
            if joint.child_body_id in parent_of:
                raise ValueError("child body has multiple parent joints")
            parent_of[joint.child_body_id] = joint.parent_body_id
        if set(parent_of) != children:
            raise ValueError("each child body requires exactly one parent joint")
        for child in children:
            visited: set[str] = set()
            current = child
            while current in parent_of:
                if current in visited:
                    raise ValueError("joint topology contains a cycle")
                visited.add(current)
                current = parent_of[current]
            if current not in roots:
                raise ValueError("joint chain must end at a declared root")
        movable_joints = {joint.joint_id: joint.kind for joint in self.joints if joint.kind != "FIXED"}
        if {state.joint_id: state.kind for state in self.joint_states} != movable_joints:
            raise ValueError("joint state roster or kind disagrees with topology")
        if {state.contact_id for state in self.contact_states} != {fact.contact_id for fact in self.contacts}:
            raise ValueError("contact modes must cover all declared contacts")
        for contact in self.contacts:
            for body_id, primitive_id in (
                (contact.first_body_id, contact.first_primitive_id),
                (contact.second_body_id, contact.second_primitive_id),
            ):
                if body_id not in bodies or primitive_id not in bodies[body_id].primitive_ids:
                    raise ValueError("contact face references a missing body-owned box")
        return self


class RigidSE3Inventory(_M6HashBoundCanonicalModel):
    """Expected complete profile fact addresses, including mutable leaves."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/inventory/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "inventory_sha256"

    body_ids: tuple[CanonicalId, ...]
    primitive_ids: tuple[CanonicalId, ...]
    joint_ids: tuple[CanonicalId, ...]
    contact_ids: tuple[CanonicalId, ...]
    region_ids: tuple[CanonicalId, ...]
    fact_addresses: tuple[CanonicalId, ...]
    inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def _complete_rosters(self) -> Self:
        for name in ("body_ids", "primitive_ids", "joint_ids", "contact_ids", "region_ids", "fact_addresses"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"inventory {name} must be sorted and unique")
        if not self.body_ids or any(not item.startswith("entity:") for item in self.body_ids):
            raise ValueError("inventory requires extension body entities")
        if not self.fact_addresses:
            raise ValueError("inventory must bind the complete fact address roster")
        return self


class RigidSE3SourceFactInput(_M6CanonicalModel):
    """One declared extension address with an explicit availability cell."""

    family: Literal[
        "body", "box", "joint", "contact", "region", "root-x", "root-y", "root-z",
        "root-rotation", "joint-state", "contact-mode",
    ]
    subject_entity_id: CanonicalId
    fact_key: CanonicalId
    cell: RigidSE3FactCell

    @model_validator(mode="after")
    def _subject(self) -> Self:
        if not self.subject_entity_id.startswith("entity:"):
            raise ValueError("source extension fact subject must be a body entity")
        return self


class RigidSE3SourceCells(_M6CanonicalModel):
    """Complete address roster even when individual values are unavailable."""

    inventory: RigidSE3Inventory
    facts: tuple[RigidSE3SourceFactInput, ...]

    @model_validator(mode="after")
    def _roster(self) -> Self:
        keys = tuple((row.family, row.subject_entity_id, row.fact_key) for row in self.facts)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source fact owners must be sorted and unique")
        if any(row.subject_entity_id not in self.inventory.body_ids for row in self.facts):
            raise ValueError("source fact subject is outside inventory")
        by_family = {
            family: tuple(row for row in self.facts if row.family == family)
            for family in ("body", "box", "joint", "contact", "region", "root-x", "root-y",
                           "root-z", "root-rotation", "joint-state", "contact-mode")
        }
        for family, expected in (
            ("body", self.inventory.body_ids),
            ("box", self.inventory.primitive_ids),
            ("joint", self.inventory.joint_ids),
            ("contact", self.inventory.contact_ids),
            ("region", self.inventory.region_ids),
            ("contact-mode", self.inventory.contact_ids),
        ):
            if tuple(sorted(row.fact_key for row in by_family[family])) != expected:
                raise ValueError(f"source {family} cells disagree with inventory")
        for row in by_family["body"]:
            if row.fact_key != row.subject_entity_id:
                raise ValueError("body cell owner must equal its body ID")
        root_ids = tuple(sorted(row.fact_key for row in by_family["root-x"]))
        for family in ("root-y", "root-z", "root-rotation"):
            if tuple(sorted(row.fact_key for row in by_family[family])) != root_ids:
                raise ValueError("root pose cells must cover the same body IDs")
        if not root_ids or any(row.fact_key != row.subject_entity_id or
                               row.fact_key not in self.inventory.body_ids
                               for family in ("root-x", "root-y", "root-z", "root-rotation")
                               for row in by_family[family]):
            raise ValueError("root pose cell owner is not an inventory root")
        exact: dict[tuple[str, str], object] = {}
        classes = {
            "body": RigidSE3BodyFact,
            "box": RigidSE3Box,
            "joint": RigidSE3JointFact,
            "contact": RigidSE3ContactFact,
            "region": RigidSE3RegionFact,
            "root-x": RigidSE3Rational,
            "root-y": RigidSE3Rational,
            "root-z": RigidSE3Rational,
            "root-rotation": RigidSE3Rotation,
        }
        for row in self.facts:
            if row.cell.availability != "KNOWN_EXACT":
                continue
            if row.cell.value is None or row.cell.value.value_schema_ref != schema("tree"):
                raise ValueError("exact source cell must use the M6 tree schema")
            value = decode_tree(row.cell.value)
            if row.family in classes:
                value = classes[row.family].model_validate(value, strict=True)
            elif row.family == "contact-mode":
                if value not in ("ENGAGED", "RELEASED"):
                    raise ValueError("contact mode cell is invalid")
            elif row.family == "joint-state":
                if not isinstance(value, dict):
                    raise ValueError("joint state cell requires an exact coordinate record")
                try:
                    value = RigidSE3Rational.model_validate(value, strict=True)
                except ValueError:
                    value = RigidSE3CirclePoint.model_validate(value, strict=True)
            exact[(row.family, row.fact_key)] = value
            if row.family in ("body", "box", "joint", "contact", "region"):
                identifier = {
                    "body": "body_id", "box": "primitive_id",
                    "joint": "joint_id", "contact": "contact_id",
                    "region": "region_id",
                }[row.family]
                if getattr(value, identifier) != row.fact_key:
                    raise ValueError("exact source fact ID disagrees with inventory key")
            if row.family == "joint" and value.child_body_id != row.subject_entity_id:
                raise ValueError("joint cell must be owned by child body")
            if row.family == "contact" and value.first_body_id != row.subject_entity_id:
                raise ValueError("contact cell must be owned by its first body")
        if all(("body", body_id) in exact for body_id in self.inventory.body_ids):
            known_roots = tuple(sorted(body_id for body_id in self.inventory.body_ids
                                       if exact[("body", body_id)].kind == "ROOT"))
            if known_roots != root_ids:
                raise ValueError("root cell roster disagrees with exact body kinds")
            primitive_ids = tuple(sorted(primitive for body_id in self.inventory.body_ids
                                         for primitive in exact[("body", body_id)].primitive_ids))
            if primitive_ids != self.inventory.primitive_ids:
                raise ValueError("primitive inventory disagrees with exact body rosters")
            for body_id in self.inventory.body_ids:
                body = exact[("body", body_id)]
                if any(("box", primitive_id) in exact and
                       next(row for row in by_family["box"] if row.fact_key == primitive_id).subject_entity_id != body_id
                       for primitive_id in body.primitive_ids):
                    raise ValueError("box cell owner disagrees with exact body")
        for row in by_family["joint-state"]:
            joint = exact.get(("joint", row.fact_key))
            if joint is not None and row.subject_entity_id != joint.child_body_id:
                raise ValueError("joint-state cell must be owned by the child body")
            state = exact.get(("joint-state", row.fact_key))
            if joint is not None and state is not None and (
                (joint.kind == "PRISMATIC" and not isinstance(state, RigidSE3Rational))
                or (joint.kind == "REVOLUTE" and not isinstance(state, RigidSE3CirclePoint))
                or joint.kind == "FIXED"
            ):
                raise ValueError("known joint state kind disagrees with exact joint")
        parent_of: dict[str, str] = {}
        for row in by_family["joint"]:
            joint = exact.get(("joint", row.fact_key))
            if joint is None:
                continue
            if (joint.parent_body_id not in self.inventory.body_ids
                    or joint.child_body_id not in self.inventory.body_ids):
                raise ValueError("known joint references a body outside inventory")
            child = exact.get(("body", joint.child_body_id))
            if child is not None and child.kind != "CHILD":
                raise ValueError("known joint child is declared as a root")
            if joint.child_body_id in parent_of:
                raise ValueError("known child body has multiple parent joints")
            parent_of[joint.child_body_id] = joint.parent_body_id
        for child_id in parent_of:
            visited: set[str] = set()
            current = child_id
            while current in parent_of:
                if current in visited:
                    raise ValueError("known joint topology contains a cycle")
                visited.add(current)
                current = parent_of[current]
        if (len(parent_of) == len(self.inventory.joint_ids)
                and all(("body", body_id) in exact for body_id in self.inventory.body_ids)):
            children = {body_id for body_id in self.inventory.body_ids
                        if exact[("body", body_id)].kind == "CHILD"}
            if set(parent_of) != children:
                raise ValueError("known joint graph does not cover each child exactly once")
        for row in by_family["contact-mode"]:
            contact = exact.get(("contact", row.fact_key))
            if contact is not None and row.subject_entity_id != contact.first_body_id:
                raise ValueError("contact-mode cell must be owned by the first body")
        for row in by_family["contact"]:
            contact = exact.get(("contact", row.fact_key))
            if contact is None:
                continue
            for body_id, primitive_id in (
                (contact.first_body_id, contact.first_primitive_id),
                (contact.second_body_id, contact.second_primitive_id),
            ):
                if body_id not in self.inventory.body_ids or primitive_id not in self.inventory.primitive_ids:
                    raise ValueError("known contact references an item outside inventory")
                body = exact.get(("body", body_id))
                if body is not None and primitive_id not in body.primitive_ids:
                    raise ValueError("known contact face references another body's primitive")
        if all(("joint", joint_id) in exact for joint_id in self.inventory.joint_ids):
            moving_ids = tuple(sorted(joint_id for joint_id in self.inventory.joint_ids
                                      if exact[("joint", joint_id)].kind != "FIXED"))
            if tuple(sorted(row.fact_key for row in by_family["joint-state"])) != moving_ids:
                raise ValueError("joint-state roster disagrees with exact joint kinds")
        if all(row.cell.availability == "KNOWN_EXACT" for row in self.facts):
            def ordered(family: str) -> tuple[object, ...]:
                return tuple(exact[(family, row.fact_key)] for row in sorted(
                    by_family[family], key=lambda item: item.fact_key,
                ))

            roots = tuple(RigidSE3RootState(
                body_id=body_id,
                world_pose=RigidSE3Pose(
                    frame="WORLD",
                    translation_m=RigidSE3Vector(
                        x=exact[("root-x", body_id)],
                        y=exact[("root-y", body_id)],
                        z=exact[("root-z", body_id)],
                    ),
                    rotation=exact[("root-rotation", body_id)],
                ),
            ) for body_id in root_ids)
            joint_states = tuple(RigidSE3JointState(
                joint_id=row.fact_key,
                kind=exact[("joint", row.fact_key)].kind,
                offset_m=exact[("joint-state", row.fact_key)]
                if exact[("joint", row.fact_key)].kind == "PRISMATIC" else None,
                circle_point=exact[("joint-state", row.fact_key)]
                if exact[("joint", row.fact_key)].kind == "REVOLUTE" else None,
            ) for row in sorted(by_family["joint-state"], key=lambda item: item.fact_key))
            RigidSE3ExactSourceFacts(
                bodies=ordered("body"), boxes=ordered("box"),
                joints=ordered("joint"), contacts=ordered("contact"),
                regions=ordered("region"),
                roots=roots, joint_states=joint_states,
                contact_states=tuple(RigidSE3ContactState(
                    contact_id=row.fact_key,
                    mode=exact[("contact-mode", row.fact_key)],
                ) for row in sorted(by_family["contact-mode"], key=lambda item: item.fact_key)),
            )
        return self


class RigidSE3ProfileRegistration(_M6CanonicalModel):
    profile_ref: Literal["spatialcf/rigid_se3_multi@1"] = PROFILE_REF
    operator: Literal["APPLY_EDIT_SET"] = "APPLY_EDIT_SET"
    member_kinds: tuple[
        Literal["SET_CONTACT_MODE"], Literal["SET_JOINT"], Literal["SET_ROOT_SE3"]
    ] = ("SET_CONTACT_MODE", "SET_JOINT", "SET_ROOT_SE3")
    derived_rule_refs: tuple[()] = ()
    coordinate_system: Literal["RH_METERS_Z_UP"] = "RH_METERS_Z_UP"
    endpoint_semantics: Literal["EXACT_PREFIX_STATES"] = "EXACT_PREFIX_STATES"


class RigidSE3RootChoiceDomain(_M6CanonicalModel):
    root_body_id: CanonicalId
    translation: RigidSE3RootTranslationDomain
    rotation: RigidSE3RootRotationDomain

    @model_validator(mode="after")
    def _root(self) -> Self:
        if not self.root_body_id.startswith("entity:"):
            raise ValueError("root choice must name an extension body")
        return self


class RigidSE3JointChoiceDomain(_M6CanonicalModel):
    joint_id: CanonicalId
    kind: Literal["PRISMATIC", "REVOLUTE"]
    prismatic: RigidSE3PrismaticDomain | None = None
    revolute: RigidSE3RevoluteDomain | None = None

    @model_validator(mode="after")
    def _variant(self) -> Self:
        if (self.kind == "PRISMATIC") != (self.prismatic is not None):
            raise ValueError("prismatic joint requires only prismatic choices")
        if (self.kind == "REVOLUTE") != (self.revolute is not None):
            raise ValueError("revolute joint requires only revolute choices")
        return self


class RigidSE3ContactChoiceDomain(_M6CanonicalModel):
    contact_id: CanonicalId
    modes: tuple[Literal["ENGAGED", "RELEASED"], ...]

    @model_validator(mode="after")
    def _modes(self) -> Self:
        if not self.modes or self.modes != tuple(sorted(set(self.modes))):
            raise ValueError("contact modes must be sorted, unique and nonempty")
        return self


class RigidSE3EditMemberTemplate(_M6CanonicalModel):
    kind: Literal["SET_ROOT_SE3", "SET_JOINT", "SET_CONTACT_MODE"]
    target_id: CanonicalId


class RigidSE3EditSetTemplate(_M6CanonicalModel):
    members: tuple[RigidSE3EditMemberTemplate, ...]

    @model_validator(mode="after")
    def _atomic_writes(self) -> Self:
        if not self.members:
            raise ValueError("atomic edit set cannot be empty")
        keys = tuple((member.kind, member.target_id) for member in self.members)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("atomic member targets must be sorted and unique")
        return self


class RigidSE3ProgramSkeleton(_M6CanonicalModel):
    skeleton_id: CanonicalId
    steps: tuple[RigidSE3EditSetTemplate, ...]

    @model_validator(mode="after")
    def _steps(self) -> Self:
        if not self.steps:
            raise ValueError("program skeleton needs at least one atomic set")
        return self


class RigidSE3MemberChoice(_M6CanonicalModel):
    """One absolute member value in an operational witness hint."""

    kind: Literal["SET_ROOT_SE3", "SET_JOINT", "SET_CONTACT_MODE"]
    target_id: CanonicalId
    pose: RigidSE3Pose | None = None
    offset_m: RigidSE3Rational | None = None
    circle_point: RigidSE3CirclePoint | None = None
    mode: Literal["ENGAGED", "RELEASED"] | None = None

    @model_validator(mode="after")
    def _exact_variant(self) -> Self:
        if self.kind == "SET_ROOT_SE3":
            if self.pose is None or self.pose.frame != "WORLD" or any(
                value is not None for value in (self.offset_m, self.circle_point, self.mode)
            ):
                raise ValueError("root choice requires only an absolute world pose")
        elif self.kind == "SET_JOINT":
            if self.pose is not None or self.mode is not None or (
                (self.offset_m is None) == (self.circle_point is None)
            ):
                raise ValueError("joint choice requires one exact joint value")
        elif self.mode is None or any(
            value is not None for value in (self.pose, self.offset_m, self.circle_point)
        ):
            raise ValueError("contact choice requires only an exact mode")
        return self


class RigidSE3StepChoice(_M6CanonicalModel):
    members: tuple[RigidSE3MemberChoice, ...]

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        keys = tuple((member.kind, member.target_id) for member in self.members)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("atomic choice members must be sorted, unique and nonempty")
        return self


class RigidSE3WitnessHint(_M6CanonicalModel):
    skeleton_id: CanonicalId
    steps: tuple[RigidSE3StepChoice, ...]

    @model_validator(mode="after")
    def _program(self) -> Self:
        if not self.steps:
            raise ValueError("witness hint must bind a full nonempty program")
        return self


class RigidSE3ProposalPolicyPayload(_M6CanonicalModel):
    """Operational only; changing hints never changes semantic problem bytes."""

    profile_registration_sha256: Sha256Digest
    witness_hints: tuple[RigidSE3WitnessHint, ...]

    @model_validator(mode="after")
    def _ordered_hints(self) -> Self:
        if len(self.witness_hints) > MAX_WITNESS_HINTS:
            raise M6IngressLimit("M6 ingress witness_hints ceiling exceeded")
        _canonical_unique(self.witness_hints, name="witness hint")
        return self


class RigidSE3ObjectivePolicy(_M6CanonicalModel):
    """Dimensionless rational movement, mode and step cost coefficients."""

    translation_weight: RigidSE3Rational
    rotation_weight: RigidSE3Rational
    prismatic_weight: RigidSE3Rational
    revolute_weight: RigidSE3Rational
    contact_weight: RigidSE3Rational
    step_weight: RigidSE3Rational
    translation_length_m: RigidSE3Rational
    joint_length_m: RigidSE3Rational
    translation_enabled: StrictBool = True
    rotation_enabled: StrictBool = True
    prismatic_enabled: StrictBool = True
    revolute_enabled: StrictBool = True

    @model_validator(mode="after")
    def _weights(self) -> Self:
        if self.translation_length_m.as_fraction <= 0 or self.joint_length_m.as_fraction <= 0:
            raise ValueError("length normalizers must be positive")
        for name in ("translation", "rotation", "prismatic", "revolute", "contact", "step"):
            value = getattr(self, f"{name}_weight").as_fraction
            if value < 0:
                raise ValueError("objective weights must be nonnegative")
            if name in ("translation", "rotation", "prismatic", "revolute") and getattr(self, f"{name}_enabled") and value == 0:
                raise ValueError("enabled exact numeric freedoms require positive weight")
        return self


class RigidSE3Domain(_M6HashBoundCanonicalModel):
    """The source-bound finite program universe or exact continuous variant."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/domain/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "domain_sha256"

    roots: tuple[RigidSE3RootChoiceDomain, ...]
    joints: tuple[RigidSE3JointChoiceDomain, ...]
    contacts: tuple[RigidSE3ContactChoiceDomain, ...]
    program_skeletons: tuple[RigidSE3ProgramSkeleton, ...]
    affected_body_ids: tuple[CanonicalId, ...]
    authorized_primary_leaves: tuple[StateVariableRef, ...]
    domain_sha256: Sha256Digest

    @model_validator(mode="after")
    def _rosters(self) -> Self:
        for name, rows, key in (
            ("roots", self.roots, lambda row: row.root_body_id),
            ("joints", self.joints, lambda row: row.joint_id),
            ("contacts", self.contacts, lambda row: row.contact_id),
            ("programs", self.program_skeletons, lambda row: row.skeleton_id),
        ):
            keys = tuple(key(row) for row in rows)
            if keys != tuple(sorted(set(keys))):
                raise ValueError(f"{name} must have sorted unique IDs")
        if not self.program_skeletons:
            raise ValueError("M6 requires a nonempty explicit program roster")
        if self.affected_body_ids != tuple(sorted(set(self.affected_body_ids))):
            raise ValueError("affected bodies must be sorted and unique")
        if not self.affected_body_ids:
            raise ValueError("M6 requires at least one affected body")
        _canonical_unique(self.authorized_primary_leaves, name="authorized leaf")
        if not self.authorized_primary_leaves:
            raise ValueError("M6 requires exact authorized primary leaves")
        domains = {
            ("SET_ROOT_SE3", row.root_body_id) for row in self.roots
        } | {
            ("SET_JOINT", row.joint_id) for row in self.joints
        } | {
            ("SET_CONTACT_MODE", row.contact_id) for row in self.contacts
        }
        for program in self.program_skeletons:
            for step in program.steps:
                if any((member.kind, member.target_id) not in domains for member in step.members):
                    raise ValueError("edit template lacks a source-bound choice domain")
        return self

    @property
    def finite(self) -> bool:
        return all(row.translation.kind == "FINITE" and row.rotation.kind == "FINITE" for row in self.roots) and all(
            (row.prismatic is not None and row.prismatic.kind == "FINITE")
            or (row.revolute is not None and row.revolute.kind == "FINITE")
            for row in self.joints
        )


class RigidSE3Limits(_M6CanonicalModel):
    """Deterministic resource caps; counters are charged before work."""

    max_entities: Annotated[StrictInt, Field(gt=0)] = 64
    max_primitives: Annotated[StrictInt, Field(ge=0)] = 256
    max_joints: Annotated[StrictInt, Field(ge=0)] = 128
    max_steps: Annotated[StrictInt, Field(gt=0)] = 16
    max_programs: Annotated[StrictInt, Field(gt=0)] = 4096
    max_domain_products: Annotated[StrictInt, Field(gt=0)] = 1000000
    max_evaluated_tuples: Annotated[StrictInt, Field(gt=0)] = 1000000
    max_exact_operations: Annotated[StrictInt, Field(gt=0)] = 10000000
    numeric_bits: Annotated[StrictInt, Field(gt=0, le=MAX_INGRESS_INTEGER_BITS)] = 4096
    compile_operations: Annotated[StrictInt, Field(gt=0)] = 10000000
    solve_operations: Annotated[StrictInt, Field(gt=0)] = 10000000
    check_operations: Annotated[StrictInt, Field(gt=0)] = 30000000


class RigidSE3CountEvent(_M6CanonicalModel):
    """One precharged deterministic unit of profile work."""

    event_id: CanonicalId
    stage: Literal["COMPILE", "SOLVE", "CHECK"]
    kind: Literal[
        "ENTITY", "PRIMITIVE", "JOINT", "PROGRAM", "STEP", "DOMAIN_PRODUCT",
        "EVALUATED_TUPLE", "KINEMATIC", "COLLISION_PAIR", "SAT_AXIS",
        "CONTACT_PAIR", "PREDICATE", "OBJECTIVE_TERM", "EXACT_ARITHMETIC",
    ]
    amount: Annotated[StrictInt, Field(gt=0)]
    cumulative: Annotated[StrictInt, Field(gt=0)]


class RigidSE3Ledger(_M6CanonicalModel):
    """Prefix complete under one stage and one shared cumulative count."""

    stage: Literal["COMPILE", "SOLVE", "CHECK"]
    events: tuple[RigidSE3CountEvent, ...] = ()
    completed_items: tuple[CanonicalId, ...] = ()
    first_unprocessed_item: CanonicalId | None = None
    peak_numeric_bits: Annotated[StrictInt, Field(ge=0)] = 0
    reason: Literal["NONE", "RESOURCE_LIMIT", "NUMERIC_GAP", "UNSUPPORTED"] = "NONE"

    @field_validator("events", mode="before")
    @classmethod
    def _bounded_events(cls, value, info: ValidationInfo):
        if isinstance(value, (list, tuple)) and len(value) > MAX_LEDGER_EVENTS:
            raise M6IngressLimit("M6 ingress events ceiling exceeded")
        # A before-validator receives JSON arrays as Python lists. Convert
        # that one field back to a tuple so strict JSON round-trips retain
        # the canonical tuple contract after the early length check.
        return tuple(value) if info.mode == "json" and isinstance(value, list) else value

    @model_validator(mode="after")
    def _prefix(self) -> Self:
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("resource event IDs must be unique")
        if any(event.stage != self.stage for event in self.events):
            raise ValueError("resource events cannot cross stage ledgers")
        count = 0
        for event in self.events:
            count += event.amount
            if event.cumulative != count:
                raise ValueError("resource event cumulative count is not exact")
        if len(set(self.completed_items)) != len(self.completed_items):
            raise ValueError("completed item IDs must be unique")
        if self.reason == "NONE" and self.first_unprocessed_item is not None:
            raise ValueError("complete ledger cannot name an unprocessed item")
        if self.reason != "NONE" and self.first_unprocessed_item is None:
            raise ValueError("incomplete ledger requires its first unprocessed item")
        return self


class RigidSE3PrefixTruth(_M6CanonicalModel):
    prefix_index: Annotated[StrictInt, Field(ge=0)]
    obligation_id: CanonicalId
    satisfied: StrictBool


class RigidSE3ContactTruth(_M6CanonicalModel):
    contact_id: CanonicalId
    declared_mode: Literal["ENGAGED", "RELEASED"]
    geometrically_satisfied: StrictBool


class RigidSE3StepTrace(_M6CanonicalModel):
    step_index: Annotated[StrictInt, Field(ge=0)]
    choice: RigidSE3StepChoice
    before_state_sha256: Sha256Digest
    after_state_sha256: Sha256Digest
    after_state: SceneStateEnvelope
    step_delta: StateDeltaManifest
    read_leaves: tuple[StateVariableRef, ...]
    primary_write_leaves: tuple[StateVariableRef, ...]
    changed_leaves: tuple[StateVariableRef, ...]
    prefix_truth: tuple[RigidSE3PrefixTruth, ...]
    world_poses: tuple[tuple[CanonicalId, RigidSE3Pose], ...]
    contact_truth: tuple[RigidSE3ContactTruth, ...]

    @model_validator(mode="after")
    def _footprint(self) -> Self:
        if self.after_state.scene_state_sha256 != self.after_state_sha256:
            raise ValueError("step trace must carry its complete after envelope")
        _canonical_unique(self.read_leaves, name="step read leaf")
        _canonical_unique(self.primary_write_leaves, name="step write leaf")
        _canonical_unique(self.changed_leaves, name="step changed leaf")
        if not set(self.changed_leaves) <= set(self.primary_write_leaves):
            raise ValueError("step delta exceeds its primary write footprint")
        if tuple(body_id for body_id, _ in self.world_poses) != self.after_state.closed_entity_index:
            raise ValueError("step trace must carry every body's reconstructed world pose")
        if tuple(row.contact_id for row in self.contact_truth) != tuple(sorted(
            {row.contact_id for row in self.contact_truth}
        )):
            raise ValueError("contact truth table must be sorted and unique")
        return self


class RigidSE3ProgramTrace(_M6HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/program-trace/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "trace_sha256"

    skeleton_id: CanonicalId
    choices: tuple[RigidSE3StepChoice, ...]
    source_state_sha256: Sha256Digest
    after_state_sha256: Sha256Digest
    steps: tuple[RigidSE3StepTrace, ...]
    program: EditProgram
    trace_sha256: Sha256Digest

    @model_validator(mode="after")
    def _chain(self) -> Self:
        if not self.choices or len(self.choices) != len(self.steps):
            raise ValueError("program trace must cover every selected step")
        prior = self.source_state_sha256
        for index, (choice, step) in enumerate(zip(self.choices, self.steps, strict=True)):
            if step.step_index != index or step.choice != choice or step.before_state_sha256 != prior:
                raise ValueError("program trace step chain is discontinuous")
            prior = step.after_state_sha256
        if prior != self.after_state_sha256:
            raise ValueError("program trace endpoint does not match its steps")
        if (self.program.before_state_sha256 != self.source_state_sha256
                or self.program.after_scene_state_sha256 != self.after_state_sha256
                or self.program.after_scene_state != self.steps[-1].after_state
                or len(self.program.steps) != len(self.steps)):
            raise ValueError("program trace disagrees with its complete generic program")
        return self


class RigidSE3FailedPrefix(_M6CanonicalModel):
    """Explicit replay through the first failed source/prefix obligation."""

    source_state_sha256: Sha256Digest
    choices: tuple[RigidSE3StepChoice, ...]
    completed_steps: tuple[RigidSE3StepTrace, ...]
    failure_prefix_index: Annotated[StrictInt, Field(ge=0)]
    failure_obligation_id: CanonicalId
    failure_state: SceneStateEnvelope

    @model_validator(mode="after")
    def _prefix(self) -> Self:
        if len(self.completed_steps) > len(self.choices):
            raise ValueError("failed prefix exceeds the candidate program")
        prior = self.source_state_sha256
        for index, step in enumerate(self.completed_steps):
            if (step.step_index != index or step.choice != self.choices[index]
                    or step.before_state_sha256 != prior):
                raise ValueError("failed prefix state chain is discontinuous")
            prior = step.after_state_sha256
        if (self.failure_prefix_index != len(self.completed_steps)
                or self.failure_state.scene_state_sha256 != prior):
            raise ValueError("failure must bind the first evaluated prefix state")
        return self


class RigidSE3Compilation(_M6HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/compilation/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "compilation_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    source_state_sha256: Sha256Digest
    source_state: SceneStateEnvelope
    inventory_sha256: Sha256Digest
    domain: RigidSE3Domain
    limits: RigidSE3Limits
    ledger: RigidSE3Ledger
    compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _stage(self) -> Self:
        if self.source_state.scene_state_sha256 != self.source_state_sha256:
            raise ValueError("compilation must carry its complete source envelope")
        if self.ledger.stage != "COMPILE":
            raise ValueError("compilation requires a compile-stage ledger")
        return self


class RigidSE3TupleEvidence(_M6CanonicalModel):
    """One canonical Cartesian member; checker regenerates its membership."""

    tuple_id: CanonicalId
    skeleton_id: CanonicalId
    ordinal: Annotated[StrictInt, Field(ge=0)]
    choices: tuple[RigidSE3StepChoice, ...]
    disposition: Literal["FEASIBLE", "INFEASIBLE"]
    failure_prefix_index: Annotated[StrictInt, Field(ge=0)] | None = None
    failure_obligation_id: CanonicalId | None = None
    failed_prefix: RigidSE3FailedPrefix | None = None
    exact_cost: RigidSE3Rational | None = None
    trace: RigidSE3ProgramTrace | None = None

    @model_validator(mode="after")
    def _branch(self) -> Self:
        if self.disposition == "FEASIBLE":
            if (self.exact_cost is None or self.trace is None
                    or self.failure_prefix_index is not None or self.failure_obligation_id is not None
                    or self.failed_prefix is not None):
                raise ValueError("feasible tuple requires exact cost and complete trace only")
        elif (self.failure_prefix_index is None or self.failure_obligation_id is None
              or self.failed_prefix is None or self.exact_cost is not None or self.trace is not None
              or self.failed_prefix.choices != self.choices
              or self.failed_prefix.failure_prefix_index != self.failure_prefix_index
              or self.failed_prefix.failure_obligation_id != self.failure_obligation_id):
            raise ValueError("infeasible tuple requires its first failed prefix trace")
        return self


class RigidSE3Coverage(_M6HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/coverage/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "coverage_sha256"

    domain_sha256: Sha256Digest
    program_products: tuple[tuple[CanonicalId, Annotated[StrictInt, Field(gt=0)]], ...]
    expected_total: Annotated[StrictInt, Field(gt=0)]
    tuples: tuple[RigidSE3TupleEvidence, ...]

    winner_tuple_id: CanonicalId | None
    ledger: RigidSE3Ledger
    coverage_sha256: Sha256Digest

    @model_validator(mode="after")
    def _counts(self) -> Self:
        if len(self.tuples) > MAX_COVERAGE_ROWS:
            raise M6IngressLimit("M6 ingress tuples ceiling exceeded")
        if self.ledger.stage != "SOLVE":
            raise ValueError("coverage requires a solve-stage ledger")
        if len({key for key, _ in self.program_products}) != len(self.program_products):
            raise ValueError("program products must name unique skeletons")
        if sum(count for _, count in self.program_products) != self.expected_total:
            raise ValueError("coverage product sum is inconsistent")
        if len({row.tuple_id for row in self.tuples}) != len(self.tuples):
            raise ValueError("coverage tuple IDs must be unique")
        feasible = {row.tuple_id for row in self.tuples if row.disposition == "FEASIBLE"}
        if self.winner_tuple_id is not None and self.winner_tuple_id not in feasible:
            raise ValueError("coverage winner must be a feasible tuple")
        if self.ledger.reason == "NONE":
            if len(self.tuples) != self.expected_total:
                raise ValueError("complete coverage requires every Cartesian tuple")
            if bool(feasible) != (self.winner_tuple_id is not None):
                raise ValueError("complete coverage winner must match feasible roster")
        elif len(self.tuples) >= self.expected_total:
            raise ValueError("incomplete coverage cannot contain every tuple")
        return self


class RigidSE3SwapEvidence(_M6CanonicalModel):
    kind: Literal["STATE_COMMUTES", "SAFE_ENDPOINT_SWAP"]
    source_trace: RigidSE3ProgramTrace
    swapped_trace: RigidSE3ProgramTrace
    source_trace_sha256: Sha256Digest
    swapped_trace_sha256: Sha256Digest
    adjacent_index: Annotated[StrictInt, Field(ge=0)]
    common_prefix_state_sha256: Sha256Digest
    domain_sha256: Sha256Digest

    @model_validator(mode="after")
    def _bound_adjacent_swap(self) -> Self:
        source = self.source_trace
        swapped = self.swapped_trace
        index = self.adjacent_index
        if (source.trace_sha256 != self.source_trace_sha256
                or swapped.trace_sha256 != self.swapped_trace_sha256):
            raise ValueError("swap evidence must bind both complete trace hashes")
        if (source.source_state_sha256 != swapped.source_state_sha256
                or len(source.steps) != len(swapped.steps)
                or index + 1 >= len(source.steps)):
            raise ValueError("swap requires adjacent steps in one source state")
        expected = list(source.choices)
        expected[index], expected[index + 1] = expected[index + 1], expected[index]
        if tuple(expected) != swapped.choices:
            raise ValueError("swap traces must differ by exactly one adjacent exchange")
        common_source = (
            source.source_state_sha256 if index == 0
            else source.steps[index - 1].after_state_sha256
        )
        common_swapped = (
            swapped.source_state_sha256 if index == 0
            else swapped.steps[index - 1].after_state_sha256
        )
        if (self.common_prefix_state_sha256 != common_source
                or common_source != common_swapped):
            raise ValueError("swap traces must bind the same complete preceding prefix")
        if set(source.steps[index].primary_write_leaves) & set(
            source.steps[index + 1].primary_write_leaves
        ):
            raise ValueError("commuting edits require disjoint primary writes")
        if (source.after_state_sha256 != swapped.after_state_sha256
                or source.steps[-1].after_state != swapped.steps[-1].after_state):
            raise ValueError("commuting edits require equal complete final envelopes")
        return self


class RigidSE3ProofMaterial(_M6HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/rigid-se3/proof-material/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "rigid_se3_proof_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    source_state_sha256: Sha256Digest
    domain_sha256: Sha256Digest
    compilation: RigidSE3Compilation | None
    coverage: RigidSE3Coverage | None
    winner_trace: RigidSE3ProgramTrace | None
    swap_evidence: tuple[RigidSE3SwapEvidence, ...]

    failure_ledger: RigidSE3Ledger | None
    rigid_se3_proof_sha256: Sha256Digest

    @model_validator(mode="after")
    def _branches(self) -> Self:
        if len(self.swap_evidence) > MAX_SWAP_CLAIMS:
            raise M6IngressLimit("M6 ingress swap_evidence ceiling exceeded")
        if self.coverage is not None and self.compilation is None:
            raise ValueError("coverage proof requires bound compilation")
        if self.winner_trace is not None and self.compilation is None:
            raise ValueError("witness trace requires bound compilation")
        if self.failure_ledger is not None and self.failure_ledger.reason == "NONE":
            raise ValueError("failure ledger must name its actual failure")
        return self


def encode_proof_material(proof: RigidSE3ProofMaterial) -> TypedValue:
    """Lossless fixed-schema transport for a complete, hash-bound proof record.

    Nested General IR TypedValue payloads are already validated and can be
    very large. Transport their canonical bytes once, then reparse the full
    record at the independent checker boundary.
    """
    if m6_output_upper_bound(proof) > MAX_PROOF_OUTPUT_BOUND_BYTES:
        raise M6IngressLimit("M6 ingress emitted proof output ceiling exceeded")
    proof = RigidSE3ProofMaterial.model_validate(proof.model_dump(mode="python"), strict=True)
    raw = canonical_json_bytes(proof)
    if len(raw) > MAX_PROOF_BYTES:
        raise M6IngressLimit("M6 ingress emitted proof byte ceiling exceeded")
    payload = base64.urlsafe_b64encode(raw).decode("ascii")
    chunks = tuple(payload[index:index + 480] for index in range(0, len(payload), 480))
    if len(chunks) > MAX_PROOF_CHUNKS:
        raise M6IngressLimit("M6 ingress emitted proof chunk ceiling exceeded")
    return TypedValue(
        value_schema_ref=schema("proof-bytes"),
        payload=FiniteOrderedTupleValue(
            element_schema_ref=schema("proof-chunk"),
            items=tuple(TypedValue(
                value_schema_ref=schema("proof-chunk"),
                payload=CanonicalIdValue(value=chunk),
            ) for chunk in chunks),
        ),
    )


def decode_proof_material(value: TypedValue) -> RigidSE3ProofMaterial:
    """Reject noncanonical chunks or a proof whose internal seals changed."""
    if (value.value_schema_ref != schema("proof-bytes")
            or not isinstance(value.payload, FiniteOrderedTupleValue)
            or value.payload.element_schema_ref != schema("proof-chunk")
            or not value.payload.items):
        raise ValueError("M6 proof payload has the wrong fixed schema")
    if len(value.payload.items) > MAX_PROOF_CHUNKS:
        raise M6IngressLimit("M6 ingress proof chunk ceiling exceeded")
    chunks = []
    encoded_length = 0
    for item in value.payload.items:
        if item.value_schema_ref != schema("proof-chunk") or not isinstance(item.payload, CanonicalIdValue):
            raise ValueError("M6 proof chunk has the wrong fixed schema")
        chunk = item.payload.value
        encoded_length += len(chunk)
        if encoded_length > ((MAX_PROOF_BYTES + 2) // 3) * 4:
            raise M6IngressLimit("M6 ingress proof byte ceiling exceeded")
        chunks.append(chunk)
    if any(len(chunk) != 480 for chunk in chunks[:-1]) or not 0 < len(chunks[-1]) <= 480:
        raise ValueError("M6 proof chunks are not canonical")
    payload = "".join(chunks)
    try:
        raw = base64.b64decode(payload, altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("M6 proof payload is not valid base64") from error
    if len(raw) > MAX_PROOF_BYTES:
        raise M6IngressLimit("M6 ingress proof byte ceiling exceeded")
    if base64.urlsafe_b64encode(raw).decode("ascii") != payload:
        raise ValueError("M6 proof payload has a noncanonical base64 spelling")
    proof = RigidSE3ProofMaterial.model_validate_json(raw, strict=True)
    if canonical_json_bytes(proof) != raw:
        raise ValueError("M6 proof payload is not canonical record bytes")
    return proof
