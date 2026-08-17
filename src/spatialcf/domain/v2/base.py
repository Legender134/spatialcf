"""Platform-neutral Canonical v2 foundation value objects.

This module deliberately contains no dataset, adapter, solver, or platform imports.
It defines only the strict values shared by the remaining v2 semantic contracts.
"""

from __future__ import annotations

import math
import unicodedata
from enum import StrEnum
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class V2Model(BaseModel):
    """Strict immutable base for every Canonical v2 value object."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        revalidate_instances="always",
    )


FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, gt=0.0),
]
_UnitIntervalFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


def _require_unicode_nfc(value: object) -> object:
    if isinstance(value, str) and value != unicodedata.normalize("NFC", value):
        raise ValueError("canonical IDs must already use Unicode NFC normalization")
    return value


CanonicalId = Annotated[
    str,
    BeforeValidator(_require_unicode_nfc),
    Field(
        strict=True,
        min_length=1,
        max_length=512,
        pattern=r"^[^\s\x00-\x1f\x7f]+$",
    ),
]
Sha256Digest = Annotated[
    str,
    Field(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class SchemaIdentityV2(V2Model):
    """Identity carried by a versioned Canonical v2 root contract."""

    schema_name: CanonicalId
    schema_version: Literal["2.0"] = "2.0"


class FactAvailabilityV2(StrEnum):
    """Whether a source contract supplies a fact family."""

    KNOWN = "KNOWN"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FactCompletenessV2(StrEnum):
    """Soundness/coverage claim attached to known fact values."""

    EXACT = "EXACT"
    INNER_BOUND = "INNER_BOUND"
    OUTER_BOUND = "OUTER_BOUND"
    BRACKETED = "BRACKETED"
    SAMPLED = "SAMPLED"


class Vec2V2(V2Model):
    """Finite two-dimensional vector in the unit declared by its owner."""

    x: FiniteFloat
    y: FiniteFloat


class Vec3V2(V2Model):
    """Finite three-dimensional vector in the unit declared by its owner."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class QuaternionV2(V2Model):
    """Unit quaternion with q/-q normalized to one deterministic sign."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    w: FiniteFloat

    @model_validator(mode="after")
    def validate_and_canonicalize(self) -> Self:
        components = (self.x, self.y, self.z, self.w)
        norm = math.sqrt(sum(component * component for component in components))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("quaternion must have unit length")

        normalized = tuple(component / norm for component in components)
        # q and -q encode the same rotation. Prefer a positive scalar component;
        # for a 180-degree rotation, use x, then y, then z as the stable tie-break.
        sign_probe = next(
            (component for component in (normalized[3], *normalized[:3]) if component),
            0.0,
        )
        if sign_probe < 0.0:
            normalized = tuple(-component for component in normalized)
        normalized = tuple(
            0.0 if component == 0.0 else component for component in normalized
        )

        for field_name, component in zip(("x", "y", "z", "w"), normalized, strict=True):
            object.__setattr__(self, field_name, component)
        return self

    @classmethod
    def identity(cls) -> Self:
        return cls(x=0.0, y=0.0, z=0.0, w=1.0)


class RigidTransformV2(V2Model):
    """Object-local to parent-frame rigid transform."""

    translation: Vec3V2
    rotation: QuaternionV2

    @classmethod
    def identity(cls) -> Self:
        return cls(
            translation=Vec3V2(x=0.0, y=0.0, z=0.0),
            rotation=QuaternionV2.identity(),
        )


class NumericPolicyV2(V2Model):
    """Dimension-specific numeric tolerances; units are never conflated."""

    linear_tolerance_m: NonNegativeFiniteFloat = 0.0
    area_tolerance_m2: NonNegativeFiniteFloat = 0.0
    angular_tolerance_rad: NonNegativeFiniteFloat = 0.0
    pixel_tolerance_px: NonNegativeFiniteFloat = 0.0
    fraction_tolerance: _UnitIntervalFiniteFloat = 0.0


class LinearErrorModelV2(StrEnum):
    """Sound linear error semantics for points and closed solid geometry."""

    EUCLIDEAN_MORPHOLOGICAL_OFFSET_SANDWICH = "EUCLIDEAN_MORPHOLOGICAL_OFFSET_SANDWICH"


class AngularErrorModelV2(StrEnum):
    """Sound angular error semantics on the rotation group."""

    SO3_GEODESIC_BALL_RAD = "SO3_GEODESIC_BALL_RAD"


class ScalarErrorModelV2(StrEnum):
    """Sound error semantics for scalar measurements."""

    SYMMETRIC_CLOSED_INTERVAL = "SYMMETRIC_CLOSED_INTERVAL"


class ErrorCombinationV2(StrEnum):
    """How independent declared error sources compose conservatively."""

    ADDITIVE_WORST_CASE = "ADDITIVE_WORST_CASE"


class UncertaintyCalculusV2(V2Model):
    """Closed, platform-neutral interpretation of every uncertainty budget."""

    linear_error_model: LinearErrorModelV2 = (
        LinearErrorModelV2.EUCLIDEAN_MORPHOLOGICAL_OFFSET_SANDWICH
    )
    angular_error_model: AngularErrorModelV2 = AngularErrorModelV2.SO3_GEODESIC_BALL_RAD
    scalar_error_model: ScalarErrorModelV2 = (
        ScalarErrorModelV2.SYMMETRIC_CLOSED_INTERVAL
    )
    combination: ErrorCombinationV2 = ErrorCombinationV2.ADDITIVE_WORST_CASE


class UncertaintyBudgetV2(V2Model):
    """Source-side errors plus their fixed conservative interpretation."""

    calculus: UncertaintyCalculusV2 = Field(default_factory=UncertaintyCalculusV2)
    source_error: NumericPolicyV2 = Field(default_factory=NumericPolicyV2)
    shape_approximation: NumericPolicyV2 = Field(default_factory=NumericPolicyV2)


FactT = TypeVar("FactT", bound=V2Model)


class FactSetV2(V2Model, Generic[FactT]):
    """Availability-aware unordered facts without empty/missing ambiguity.

    ``KNOWN`` exact/one-sided/sampled sets use ``values``. ``BRACKETED`` facts
    instead carry explicit inner and outer sets. Missing or inapplicable facts
    carry no values, completeness claim, or uncertainty budget.
    """

    availability: FactAvailabilityV2
    values: tuple[FactT, ...] | None = None
    inner_values: tuple[FactT, ...] | None = None
    outer_values: tuple[FactT, ...] | None = None
    completeness: FactCompletenessV2 | None = None
    uncertainty: UncertaintyBudgetV2 | None = None

    @model_validator(mode="after")
    def validate_availability_contract(self) -> Self:
        if self.availability is not FactAvailabilityV2.KNOWN:
            carried = (
                self.values,
                self.inner_values,
                self.outer_values,
                self.completeness,
                self.uncertainty,
            )
            if any(value is not None for value in carried):
                raise ValueError(
                    f"{self.availability.value} facts must not carry values, "
                    "completeness, or uncertainty"
                )
            return self

        completeness = self.completeness or FactCompletenessV2.EXACT
        object.__setattr__(self, "completeness", completeness)
        if self.uncertainty is None:
            raise ValueError("KNOWN facts require explicit uncertainty")

        if completeness is FactCompletenessV2.BRACKETED:
            if self.values is not None:
                raise ValueError("BRACKETED facts must not carry values")
            if self.inner_values is None or self.outer_values is None:
                raise ValueError(
                    "BRACKETED facts require explicit inner_values and outer_values"
                )
            if any(value not in self.outer_values for value in self.inner_values):
                raise ValueError(
                    "BRACKETED inner_values must be contained in outer_values"
                )
            self._canonicalize_fact_values("inner_values", self.inner_values)
            self._canonicalize_fact_values("outer_values", self.outer_values)
            return self

        if self.values is None:
            raise ValueError("KNOWN non-BRACKETED facts require explicit values")
        if self.inner_values is not None or self.outer_values is not None:
            raise ValueError(
                "KNOWN non-BRACKETED facts must not carry inner_values or outer_values"
            )
        self._canonicalize_fact_values("values", self.values)
        return self

    def _canonicalize_fact_values(
        self,
        field_name: Literal["values", "inner_values", "outer_values"],
        values: tuple[FactT, ...],
    ) -> None:
        from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

        if any(not isinstance(value, V2Model) for value in values):
            raise ValueError("FactSetV2 values must be immutable V2Model instances")
        ordered = tuple(sorted(values, key=canonical_json_bytes_v2))
        if any(
            left == right
            for index, left in enumerate(ordered)
            for right in ordered[index + 1 :]
        ):
            raise ValueError(f"{field_name} must not contain duplicate facts")
        object.__setattr__(self, field_name, ordered)
