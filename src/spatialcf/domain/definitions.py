"""Closed M1 definition and typed-value contracts.

This module adds the small, immutable bootstrap vocabulary for the general
counterfactual IR.  It intentionally performs only local structural checks;
cross-record schema and definition resolution belongs to the static registry.
"""

from __future__ import annotations

from copy import copy
from enum import StrEnum
from types import UnionType
from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    StrictBool,
    StrictInt,
    create_model,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import CoreSchema, core_schema

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FiniteFloat,
    Quaternion,
    Sha256Digest,
    Vec2,
    Vec3,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

BOOTSTRAP_SCHEMA_SHA256 = (
    "6778bead7c5de0999af5768d2dd9747e20e08ab37fe110579c286a9e451e4fa4"
)


def _require_definition_ref(value):
    if isinstance(value, str) and not value.startswith("definition:"):
        raise ValueError("definition references must start with 'definition:'")
    return value


def _require_schema_ref(value):
    if isinstance(value, str) and not value.startswith("schema:"):
        raise ValueError("schema references must start with 'schema:'")
    return value


def _require_capability_ref(value):
    if isinstance(value, str) and not value.startswith("capability:"):
        raise ValueError("capability references must start with 'capability:'")
    return value


DefinitionRef = Annotated[CanonicalId, BeforeValidator(_require_definition_ref)]
SchemaRef = Annotated[CanonicalId, BeforeValidator(_require_schema_ref)]
CapabilityRef = Annotated[CanonicalId, BeforeValidator(_require_capability_ref)]
NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]


class ValueKind(StrEnum):
    """The closed M1 bootstrap algebra for canonical typed values."""

    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    FINITE_REAL = "FINITE_REAL"
    CANONICAL_ID = "CANONICAL_ID"
    DIGEST = "DIGEST"
    ENUM_SYMBOL = "ENUM_SYMBOL"
    LENGTH = "LENGTH"
    AREA = "AREA"
    ANGLE = "ANGLE"
    TIME = "TIME"
    PIXEL = "PIXEL"
    UNIT_INTERVAL = "UNIT_INTERVAL"
    ENTITY_REF = "ENTITY_REF"
    OBJECT_REF = "OBJECT_REF"
    GEOMETRY_REF = "GEOMETRY_REF"
    BODY_REF = "BODY_REF"
    SURFACE_REF = "SURFACE_REF"
    CAMERA_REF = "CAMERA_REF"
    FRAME_REF = "FRAME_REF"
    REGION_REF = "REGION_REF"
    PREDICATE_DEFINITION_REF = "PREDICATE_DEFINITION_REF"
    OPERATOR_DEFINITION_REF = "OPERATOR_DEFINITION_REF"
    OBJECTIVE_DEFINITION_REF = "OBJECTIVE_DEFINITION_REF"
    POINT_2D = "POINT_2D"
    POINT_3D = "POINT_3D"
    VECTOR_2D = "VECTOR_2D"
    VECTOR_3D = "VECTOR_3D"
    RIGID_POSE = "RIGID_POSE"
    INTERVAL = "INTERVAL"
    CLOSED_BOX = "CLOSED_BOX"
    FINITE_SET = "FINITE_SET"
    FINITE_ORDERED_TUPLE = "FINITE_ORDERED_TUPLE"
    RECORD = "RECORD"


_DIMENSIONED_KINDS = frozenset(
    (
        ValueKind.LENGTH,
        ValueKind.AREA,
        ValueKind.ANGLE,
        ValueKind.TIME,
        ValueKind.PIXEL,
        ValueKind.UNIT_INTERVAL,
    )
)
_GEOMETRIC_KINDS = frozenset(
    (
        ValueKind.POINT_2D,
        ValueKind.POINT_3D,
        ValueKind.VECTOR_2D,
        ValueKind.VECTOR_3D,
        ValueKind.RIGID_POSE,
        ValueKind.CLOSED_BOX,
    )
)
_SEQUENCE_KINDS = frozenset((ValueKind.FINITE_SET, ValueKind.FINITE_ORDERED_TUPLE))
_AnnotationT = TypeVar("_AnnotationT")


class HashBoundCanonicalModel(CanonicalModel):
    """A strict immutable model whose normal wire carries its own digest."""

    HASH_DOMAIN: ClassVar[str]
    SELF_DIGEST_FIELD: ClassVar[str]

    @staticmethod
    def _replace_self_annotation(
        annotation: _AnnotationT,
        model_type: type[Self],
    ) -> _AnnotationT:
        """Recursively bind typing.Self to the concrete hash-bound model."""

        if annotation is Self:
            return model_type
        origin = get_origin(annotation)
        if origin is None:
            return annotation
        arguments = get_args(annotation)
        rewritten_arguments = tuple(
            HashBoundCanonicalModel._replace_self_annotation(argument, model_type)
            for argument in arguments
        )
        if rewritten_arguments == arguments:
            return annotation
        if origin is Annotated:
            return Annotated[rewritten_arguments[0], *rewritten_arguments[1:]]
        if origin is UnionType or origin is Union:
            union = rewritten_arguments[0]
            for argument in rewritten_arguments[1:]:
                union = union | argument
            return union
        return origin[rewritten_arguments]

    @staticmethod
    def _copy_field_info_with_annotation(
        field_info: FieldInfo,
        annotation: _AnnotationT,
    ) -> FieldInfo:
        """Keep public FieldInfo metadata while changing only its annotation."""

        payload_field_info = copy(field_info)
        payload_field_info.annotation = annotation
        return payload_field_info

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[Self],
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Apply the digest check after every subclass-local validator."""

        schema = handler(source_type)
        original_ref = schema.get("ref")
        if original_ref:
            inner_schema = {**schema, "ref": f"{original_ref}:self-digest-inner"}
            return core_schema.no_info_after_validator_function(
                cls._assert_self_digest_and_return,
                inner_schema,
                ref=original_ref,
            )
        return core_schema.no_info_after_validator_function(
            cls._assert_self_digest_and_return,
            schema,
        )

    @classmethod
    def _assert_self_digest_and_return(cls, model: Self) -> Self:
        field_name = cls.SELF_DIGEST_FIELD
        if field_name not in cls.model_fields:
            raise ValueError("hash-bound model has no declared self digest field")
        payload = model.model_dump(
            mode="python",
            by_alias=True,
            exclude={field_name},
            exclude_none=False,
            exclude_defaults=False,
            exclude_unset=False,
            exclude_computed_fields=True,
            round_trip=True,
        )
        expected = canonical_sha256(payload, domain=cls.HASH_DOMAIN)
        if getattr(model, field_name) != expected:
            raise ValueError("submitted self digest does not match canonical payload")
        return model

    @classmethod
    def seal(cls, **values) -> Self:
        """Strictly validate fields, compute the only excluded digest, and seal."""

        field_name = cls.SELF_DIGEST_FIELD
        if field_name in values:
            raise ValueError("seal() does not accept a caller-supplied self-digest")
        if field_name not in cls.model_fields:
            raise ValueError("hash-bound model has no declared self digest field")

        payload_model = create_model(
            f"_{cls.__name__}SealPayload",
            __base__=CanonicalModel,
            **{
                name: (
                    rewritten_annotation := cls._replace_self_annotation(
                        field.annotation,
                        cls,
                    ),
                    cls._copy_field_info_with_annotation(field, rewritten_annotation),
                )
                for name, field in cls.model_fields.items()
                if name != field_name
            },
        )
        payload = payload_model.model_validate(values, strict=True).model_dump(
            mode="python",
            by_alias=True,
            exclude_none=False,
            exclude_defaults=False,
            exclude_unset=False,
            exclude_computed_fields=True,
            round_trip=True,
        )
        digest = canonical_sha256(payload, domain=cls.HASH_DOMAIN)
        return cls.model_validate(payload | {field_name: digest}, strict=True)


class ValueFieldDefinition(CanonicalModel):
    """One canonical record field declaration without registry resolution."""

    field_name: CanonicalId
    value_schema_ref: SchemaRef
    required: StrictBool = True


class ValueSchemaDefinition(HashBoundCanonicalModel):
    """An immutable local description of one closed typed-value shape."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/value-schema-definition/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "value_schema_definition_sha256"

    value_schema_ref: SchemaRef
    value_kind: ValueKind
    fields: tuple[ValueFieldDefinition, ...] = ()
    enum_symbols: tuple[CanonicalId, ...] = ()
    unit_schema_ref: SchemaRef | None = None
    dimension_schema_ref: SchemaRef | None = None
    frame_schema_ref: SchemaRef | None = None
    endpoint_schema_ref: SchemaRef | None = None
    lower_closed: StrictBool | None = None
    upper_closed: StrictBool | None = None
    element_schema_ref: SchemaRef | None = None
    min_cardinality: NonNegativeStrictInt | None = None
    max_cardinality: NonNegativeStrictInt | None = None
    value_schema_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_local_schema_shape(self) -> Self:
        if self.value_kind in _DIMENSIONED_KINDS:
            if self.unit_schema_ref is None:
                raise ValueError("dimensioned value schemas require a unit schema")
        elif self.unit_schema_ref is not None:
            raise ValueError("only dimensioned value schemas may carry a unit schema")

        if self.value_kind in _GEOMETRIC_KINDS:
            if self.dimension_schema_ref is None or self.frame_schema_ref is None:
                raise ValueError("geometric value schemas require dimension and frame")
        elif self.dimension_schema_ref is not None or self.frame_schema_ref is not None:
            raise ValueError(
                "only geometric value schemas may carry dimension or frame"
            )

        if self.value_kind is ValueKind.INTERVAL:
            if (
                self.endpoint_schema_ref is None
                or self.lower_closed is None
                or self.upper_closed is None
            ):
                raise ValueError("interval schemas require endpoint schema and closure")
        elif (
            self.endpoint_schema_ref is not None
            or self.lower_closed is not None
            or self.upper_closed is not None
        ):
            raise ValueError("only interval schemas may carry endpoint closure")

        if self.value_kind in _SEQUENCE_KINDS:
            if self.element_schema_ref is None:
                raise ValueError("sequence schemas require an element schema")
            if (
                self.max_cardinality is not None
                and self.min_cardinality is not None
                and self.max_cardinality < self.min_cardinality
            ):
                raise ValueError("maximum cardinality must not be below minimum")
        elif (
            self.element_schema_ref is not None
            or self.min_cardinality is not None
            or self.max_cardinality is not None
        ):
            raise ValueError("only sequence schemas may carry cardinality")

        if self.value_kind is ValueKind.ENUM_SYMBOL:
            _require_sorted_unique_ids(self.enum_symbols, "enum symbols", nonempty=True)
        elif self.enum_symbols:
            raise ValueError("only enum schemas may carry enum symbols")

        if self.value_kind is ValueKind.RECORD:
            _require_sorted_unique_field_definitions(self.fields)
        elif self.fields:
            raise ValueError("only record schemas may carry field definitions")
        return self


class BooleanValue(CanonicalModel):
    kind: Literal[ValueKind.BOOLEAN] = ValueKind.BOOLEAN
    value: StrictBool


class IntegerValue(CanonicalModel):
    kind: Literal[ValueKind.INTEGER] = ValueKind.INTEGER
    value: StrictInt


class FiniteRealValue(CanonicalModel):
    kind: Literal[ValueKind.FINITE_REAL] = ValueKind.FINITE_REAL
    value: FiniteFloat


class CanonicalIdValue(CanonicalModel):
    kind: Literal[ValueKind.CANONICAL_ID] = ValueKind.CANONICAL_ID
    value: CanonicalId


class DigestValue(CanonicalModel):
    kind: Literal[ValueKind.DIGEST] = ValueKind.DIGEST
    value: Sha256Digest


class EnumSymbolValue(CanonicalModel):
    kind: Literal[ValueKind.ENUM_SYMBOL] = ValueKind.ENUM_SYMBOL
    symbol: CanonicalId


ScalarPayload: TypeAlias = Annotated[
    BooleanValue
    | IntegerValue
    | FiniteRealValue
    | CanonicalIdValue
    | DigestValue
    | EnumSymbolValue,
    Field(discriminator="kind"),
]


class DimensionedQuantityValue(CanonicalModel):
    kind: Literal[
        ValueKind.LENGTH,
        ValueKind.AREA,
        ValueKind.ANGLE,
        ValueKind.TIME,
        ValueKind.PIXEL,
    ]
    value: FiniteFloat
    unit_schema_ref: SchemaRef


class UnitIntervalValue(CanonicalModel):
    kind: Literal[ValueKind.UNIT_INTERVAL] = ValueKind.UNIT_INTERVAL
    value: FiniteFloat
    unit_schema_ref: SchemaRef

    @model_validator(mode="after")
    def _validate_unit_interval(self) -> Self:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("unit interval values must lie within [0, 1]")
        return self


class ReferenceValue(CanonicalModel):
    kind: Literal[
        ValueKind.ENTITY_REF,
        ValueKind.OBJECT_REF,
        ValueKind.GEOMETRY_REF,
        ValueKind.BODY_REF,
        ValueKind.SURFACE_REF,
        ValueKind.CAMERA_REF,
        ValueKind.FRAME_REF,
        ValueKind.REGION_REF,
        ValueKind.PREDICATE_DEFINITION_REF,
        ValueKind.OPERATOR_DEFINITION_REF,
        ValueKind.OBJECTIVE_DEFINITION_REF,
    ]
    reference: CanonicalId


class Point2DValue(CanonicalModel):
    kind: Literal[ValueKind.POINT_2D] = ValueKind.POINT_2D
    coordinates: Vec2
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef


class Point3DValue(CanonicalModel):
    kind: Literal[ValueKind.POINT_3D] = ValueKind.POINT_3D
    coordinates: Vec3
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef


PointPayload: TypeAlias = Annotated[
    Point2DValue | Point3DValue,
    Field(discriminator="kind"),
]


class Vector2DValue(CanonicalModel):
    kind: Literal[ValueKind.VECTOR_2D] = ValueKind.VECTOR_2D
    coordinates: Vec2
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef


class Vector3DValue(CanonicalModel):
    kind: Literal[ValueKind.VECTOR_3D] = ValueKind.VECTOR_3D
    coordinates: Vec3
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef


class RigidPoseValue(CanonicalModel):
    kind: Literal[ValueKind.RIGID_POSE] = ValueKind.RIGID_POSE
    translation: Vec3
    rotation: Quaternion
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef


class IntervalValue(CanonicalModel):
    kind: Literal[ValueKind.INTERVAL] = ValueKind.INTERVAL
    endpoint_schema_ref: SchemaRef
    lower: ScalarPayload
    upper: ScalarPayload
    lower_closed: StrictBool
    upper_closed: StrictBool

    @model_validator(mode="after")
    def _validate_interval_structure(self) -> Self:
        if self.lower.kind is not self.upper.kind:
            raise ValueError("interval endpoints must share one scalar kind")
        if isinstance(self.lower, (IntegerValue, FiniteRealValue)):
            if self.lower.value > self.upper.value:
                raise ValueError(
                    "interval lower endpoint must not exceed upper endpoint"
                )
            if (
                self.lower.value == self.upper.value
                and not self.lower_closed
                and not self.upper_closed
            ):
                raise ValueError("a degenerate interval must include its endpoint")
        return self


class ClosedBoxValue(CanonicalModel):
    kind: Literal[ValueKind.CLOSED_BOX] = ValueKind.CLOSED_BOX
    minimum: PointPayload
    maximum: PointPayload
    dimension_schema_ref: SchemaRef
    frame_schema_ref: SchemaRef

    @model_validator(mode="after")
    def _validate_closed_box(self) -> Self:
        if self.minimum.kind is not self.maximum.kind:
            raise ValueError("closed box endpoints must share one point dimension")
        minimum = _coordinate_tuple(self.minimum)
        maximum = _coordinate_tuple(self.maximum)
        if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
            raise ValueError("closed box minimum must not exceed maximum")
        return self


class NamedTypedValue(CanonicalModel):
    name: CanonicalId
    value: TypedValue


class FiniteSetValue(CanonicalModel):
    kind: Literal[ValueKind.FINITE_SET] = ValueKind.FINITE_SET
    element_schema_ref: SchemaRef
    elements: tuple[TypedValue, ...]

    @model_validator(mode="after")
    def _canonicalize_elements(self) -> Self:
        encoded = tuple(canonical_json_bytes(element) for element in self.elements)
        if len(set(encoded)) != len(encoded):
            raise ValueError("finite sets must not contain duplicate members")
        ordered = tuple(
            element
            for _, element in sorted(
                zip(encoded, self.elements, strict=True), key=lambda pair: pair[0]
            )
        )
        object.__setattr__(self, "elements", ordered)
        return self


class FiniteOrderedTupleValue(CanonicalModel):
    kind: Literal[ValueKind.FINITE_ORDERED_TUPLE] = ValueKind.FINITE_ORDERED_TUPLE
    element_schema_ref: SchemaRef
    items: tuple[TypedValue, ...]


class RecordValue(CanonicalModel):
    kind: Literal[ValueKind.RECORD] = ValueKind.RECORD
    fields: tuple[NamedTypedValue, ...]

    @model_validator(mode="after")
    def _validate_record_fields(self) -> Self:
        _require_sorted_unique_named_values(self.fields)
        return self


ValuePayload: TypeAlias = Annotated[
    ScalarPayload
    | DimensionedQuantityValue
    | UnitIntervalValue
    | ReferenceValue
    | PointPayload
    | Vector2DValue
    | Vector3DValue
    | RigidPoseValue
    | IntervalValue
    | ClosedBoxValue
    | FiniteSetValue
    | FiniteOrderedTupleValue
    | RecordValue,
    Field(discriminator="kind"),
]


class TypedValue(CanonicalModel):
    """A schema reference plus one node from the closed payload union."""

    value_schema_ref: SchemaRef
    payload: ValuePayload


NamedTypedValue.model_rebuild()
FiniteSetValue.model_rebuild()
FiniteOrderedTupleValue.model_rebuild()
RecordValue.model_rebuild()
TypedValue.model_rebuild()


class CanonicalDefinitionEnvelope(HashBoundCanonicalModel):
    """One immutable definition record with its exact typed payload."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/definition-envelope/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "definition_sha256"

    definition_ref: DefinitionRef
    definition_kind_ref: DefinitionRef
    payload_schema_ref: SchemaRef
    payload: TypedValue
    definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_payload_schema_binding(self) -> Self:
        if self.payload.value_schema_ref != self.payload_schema_ref:
            raise ValueError(
                "definition payload must carry the declared payload schema"
            )
        return self


class DefinitionBundle(HashBoundCanonicalModel):
    """The semantic definition closure with only local identity checks."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/definition-bundle/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "definition_bundle_sha256"

    bootstrap_schema_sha256: Sha256Digest = BOOTSTRAP_SCHEMA_SHA256
    definitions: tuple[CanonicalDefinitionEnvelope, ...]
    definition_bundle_sha256: Sha256Digest

    @model_validator(mode="before")
    @classmethod
    def _reject_bootstrap_replacement(cls, values):
        if (
            isinstance(values, dict)
            and "bootstrap_schema_sha256" in values
            and values["bootstrap_schema_sha256"] != BOOTSTRAP_SCHEMA_SHA256
        ):
            raise ValueError("bootstrap schema replacement is not permitted")
        return values

    @model_validator(mode="after")
    def _validate_definition_order(self) -> Self:
        refs = tuple(definition.definition_ref for definition in self.definitions)
        if refs != tuple(sorted(refs, key=canonical_json_bytes)):
            raise ValueError("definition bundle definitions must be sorted")
        if len(set(refs)) != len(refs):
            raise ValueError("duplicate definition reference in definition bundle")
        return self


def _coordinate_tuple(value):
    if isinstance(value, Point2DValue):
        return (value.coordinates.x, value.coordinates.y)
    return (value.coordinates.x, value.coordinates.y, value.coordinates.z)


def _require_sorted_unique_ids(values, label: str, *, nonempty: bool = False) -> None:
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    if values != tuple(sorted(values, key=canonical_json_bytes)):
        raise ValueError(f"{label} must be sorted")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicate entries")


def _require_sorted_unique_field_definitions(
    fields: tuple[ValueFieldDefinition, ...],
) -> None:
    names = tuple(field.field_name for field in fields)
    if names != tuple(sorted(names, key=canonical_json_bytes)):
        raise ValueError("record field definitions must be sorted")
    if len(set(names)) != len(names):
        raise ValueError("record field definitions must not contain duplicate fields")


def _require_sorted_unique_named_values(fields: tuple[NamedTypedValue, ...]) -> None:
    names = tuple(field.name for field in fields)
    if names != tuple(sorted(names, key=canonical_json_bytes)):
        raise ValueError("record fields must be sorted")
    if len(set(names)) != len(names):
        raise ValueError("record fields must not contain duplicate names")
