"""Upright profile yaw contracts and intrinsic operations."""

from __future__ import annotations

import math

from enum import (
    StrEnum,
)

from fractions import (
    Fraction,
)

from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
)

from pydantic import (
    Field,
    StrictInt,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FiniteFloat,
    Quaternion,
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    DefinitionRef,
    HashBoundCanonicalModel,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
    _UPRIGHT_SE2_CARDINAL_QUATERNIONS,
    _UPRIGHT_SE2_YAW_TO_POSE_MAX_ULPS,
    _ValueT,
)


def _require_sorted_unique_by_bytes(values: tuple[_ValueT, ...], label: str) -> None:
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _fraction_from_float(value: float) -> Fraction:
    return Fraction.from_float(value)


class CanonicalSO2Angle(CanonicalModel):
    """One canonical finite binary64 turn value in the half-open SO(2) wire."""

    turns: FiniteFloat

    @model_validator(mode="after")
    def _validate_turns(self) -> Self:
        if _is_negative_zero(self.turns):
            raise ValueError("negative zero turns are not canonical")
        if not -0.5 <= self.turns < 0.5:
            raise ValueError("canonical turns must be in [-0.5, 0.5)")
        if self.turns == 0.0:
            object.__setattr__(self, "turns", 0.0)
        return self


def canonical_yaw_from_upright_quaternion(rotation: Quaternion) -> CanonicalSO2Angle:
    """Derive the registered directed SO(2) view of one yaw-only quaternion.

    The cardinal compiler never uses this conversion for a cardinal transform:
    it is solely the profile-owned consistency view for an already-normalized
    base or endpoint quaternion.  The ``Quaternion`` contract has already
    selected the unique sign representative, so the half-turn seam maps to
    canonical ``-0.5`` rather than admitting a second directed wire value.
    """

    if rotation.x != 0.0 or rotation.y != 0.0:
        raise ValueError("directed yaw quaternion must be yaw-only")
    turns = math.atan2(rotation.z, rotation.w) / math.pi
    if turns == 0.5:
        turns = -0.5
    if not -0.5 <= turns < 0.5:
        raise ValueError("directed yaw quaternion is ambiguous")
    return CanonicalSO2Angle(turns=0.0 if turns == 0.0 else turns)


def validate_directed_yaw_quaternion_consistency(
    explicit_yaw: CanonicalSO2Angle,
    rotation: Quaternion,
) -> None:
    """Enforce the registered binary64 yaw-to-quaternion consistency rule.

    The canonical directed yaw remains primary state.  Its quaternion view is
    recovered with the deterministic ``atan2(z, w) / pi`` rule, then compared
    in the SO(2) quotient using a bounded binary64 reconstruction allowance.
    The allowance covers the final rounding of a normalized stored quaternion;
    it never classifies a non-cardinal angle as cardinal.
    """

    derived_yaw = canonical_yaw_from_upright_quaternion(rotation)
    distance = abs(explicit_yaw.turns - derived_yaw.turns)
    distance = min(distance, 1.0 - distance)
    tolerance = _UPRIGHT_SE2_YAW_TO_POSE_MAX_ULPS * max(
        math.ulp(explicit_yaw.turns),
        math.ulp(derived_yaw.turns),
    )
    if distance > tolerance:
        raise ValueError("pose yaw does not match the frozen directed base pose")


def _compose_upright_quaternion_from_primary_yaw(
    *,
    rotation: Quaternion,
    delta_z: float,
    delta_w: float,
    primary_expected_yaw: CanonicalSO2Angle,
) -> Quaternion:
    """Build one derived yaw view while keeping its canonical yaw primary."""

    composed = Quaternion(
        x=0.0,
        y=0.0,
        z=(
            0.0
            if rotation.z * delta_w + rotation.w * delta_z == 0.0
            else rotation.z * delta_w + rotation.w * delta_z
        ),
        w=(
            0.0
            if rotation.w * delta_w - rotation.z * delta_z == 0.0
            else rotation.w * delta_w - rotation.z * delta_z
        ),
    )
    try:
        validate_directed_yaw_quaternion_consistency(primary_expected_yaw, composed)
    except ValueError:
        if primary_expected_yaw.turns == 0.0:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[0]
        elif primary_expected_yaw.turns == 0.25:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[1]
        elif primary_expected_yaw.turns == -0.5:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[2]
        elif primary_expected_yaw.turns == -0.25:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[3]
        else:
            half_radians = math.pi * primary_expected_yaw.turns
            z = math.sin(half_radians)
            w = math.cos(half_radians)
        reconstructed = Quaternion(x=0.0, y=0.0, z=z, w=w)
        validate_directed_yaw_quaternion_consistency(
            primary_expected_yaw,
            reconstructed,
        )
        return reconstructed
    return composed


class CardinalYaw(CanonicalModel):
    """The exact cardinal yaw wire, never a coerced floating-point turn."""

    q: Annotated[StrictInt, Field(ge=0, le=3)]


class ContinuousYawArc(CanonicalModel):
    """One closed directed SO(2) arc without a wire-level branch-cut split."""

    kind: Literal["ARC"] = "ARC"
    start_angle: CanonicalSO2Angle
    ccw_sweep_turns: FiniteFloat

    @model_validator(mode="after")
    def _validate_sweep(self) -> Self:
        if _is_negative_zero(self.ccw_sweep_turns):
            raise ValueError("negative zero sweep is not canonical")
        if not 0.0 <= self.ccw_sweep_turns < 1.0:
            raise ValueError("ccw sweep turns must be in [0, 1)")
        if self.ccw_sweep_turns == 0.0:
            object.__setattr__(self, "ccw_sweep_turns", 0.0)
        return self

    @property
    def is_closed_point(self) -> bool:
        return self.ccw_sweep_turns == 0.0

    @property
    def crosses_branch_cut(self) -> bool:
        start = _fraction_from_float(self.start_angle.turns)
        upper = start + _fraction_from_float(self.ccw_sweep_turns)
        return start < Fraction(1, 2) < upper

    @property
    def contains_canonical_zero(self) -> bool:
        start = _fraction_from_float(self.start_angle.turns)
        upper = start + _fraction_from_float(self.ccw_sweep_turns)
        return any(
            start <= representative <= upper
            for representative in (Fraction(0), Fraction(1))
        )


class ContinuousYawFullCircle(CanonicalModel):
    """The sole full-SO(2) branch; a sweep of one turn is not an alias."""

    kind: Literal["FULL_CIRCLE"] = "FULL_CIRCLE"


ContinuousYawDomain: TypeAlias = Annotated[
    ContinuousYawArc | ContinuousYawFullCircle,
    Field(discriminator="kind"),
]


class ExactDyadic(CanonicalModel):
    """A normalized exact dyadic rational for a lifted turn endpoint."""

    numerator: StrictInt
    denominator: Annotated[StrictInt, Field(gt=0)]

    @model_validator(mode="after")
    def _validate_normalized_dyadic(self) -> Self:
        if self.denominator & (self.denominator - 1):
            raise ValueError("dyadic denominator must be a power of two")
        if self.numerator == 0 and self.denominator != 1:
            raise ValueError("zero dyadic must use denominator one")
        if self.denominator != 1 and self.numerator % 2 == 0:
            raise ValueError("dyadic numerator and denominator must be normalized")
        return self

    @property
    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class LiftedYawInterval(CanonicalModel):
    """One closed exact-dyadic interval in the fixed ``-0.5`` proof lift."""

    lower: ExactDyadic
    upper: ExactDyadic
    endpoint_closure: Literal["CLOSED"] = "CLOSED"
    seam_ownership: Literal["NONE", "LOWER_OWNS_SEAM", "UPPER_OWNS_ENDPOINT"]

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        if self.lower.as_fraction > self.upper.as_fraction:
            raise ValueError("lifted interval lower endpoint must not exceed upper")
        return self


class ContinuousYawLift(HashBoundCanonicalModel):
    """The one canonical non-wrapping lifted interval closure for a yaw domain."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-yaw-lift/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_yaw_lift_sha256"

    yaw_domain: ContinuousYawDomain
    lift_origin: ExactDyadic
    intervals: tuple[LiftedYawInterval, ...]
    continuous_yaw_lift_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_canonical_lift(self) -> Self:
        if self.lift_origin.as_fraction != Fraction(-1, 2):
            raise ValueError("lift origin must be exactly -0.5 turns")
        if len(self.intervals) != 1:
            raise ValueError("a canonical yaw domain has exactly one lifted interval")
        interval = self.intervals[0]
        if isinstance(self.yaw_domain, ContinuousYawFullCircle):
            if (
                interval.lower.as_fraction,
                interval.upper.as_fraction,
                interval.seam_ownership,
            ) != (Fraction(-1, 2), Fraction(1, 2), "LOWER_OWNS_SEAM"):
                raise ValueError("full-circle lift must use the lower-owned seam")
            return self
        start = _fraction_from_float(self.yaw_domain.start_angle.turns)
        upper = start + _fraction_from_float(self.yaw_domain.ccw_sweep_turns)
        expected_seam = "UPPER_OWNS_ENDPOINT" if upper == Fraction(1, 2) else "NONE"
        if (interval.lower.as_fraction, interval.upper.as_fraction) != (start, upper):
            raise ValueError("arc lift endpoints must be exact dyadic unrolled values")
        if interval.seam_ownership != expected_seam:
            raise ValueError("arc seam ownership is not canonical")
        return self


class PivotMode(StrEnum):
    """The only two endpoint pivot bindings admitted by this profile."""

    OWN = "OWN"
    REFERENCE = "REFERENCE"


class FixedPivotBinding(HashBoundCanonicalModel):
    """A hash-bound frozen subject or named reference object pivot."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/fixed-pivot-binding/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "fixed_pivot_binding_sha256"

    subject_id: CanonicalId
    pivot_mode: PivotMode
    pivot_entity_id: CanonicalId
    pivot_state_sha256: Sha256Digest
    fixed_pivot_binding_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Seal the canonical JSON pivot-mode symbol as its closed enum value."""

        pivot_mode = values.get("pivot_mode")
        if isinstance(pivot_mode, str):
            values = {**values, "pivot_mode": PivotMode(pivot_mode)}
        return super().seal(**values)

    @model_validator(mode="after")
    def _validate_pivot_identity(self) -> Self:
        if self.pivot_mode is PivotMode.OWN and self.pivot_entity_id != self.subject_id:
            raise ValueError("own pivot must be the exact subject entity")
        if (
            self.pivot_mode is PivotMode.REFERENCE
            and self.pivot_entity_id == self.subject_id
        ):
            raise ValueError("reference pivot must not name the subject")
        return self


class ExplicitPoseYawBinding(HashBoundCanonicalModel):
    """The explicit primary yaw fact and its hash-bound derived base-pose view."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/explicit-pose-yaw-binding/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "pose_yaw_binding_sha256"

    entity_id: CanonicalId
    base_pose_sha256: Sha256Digest
    explicit_yaw: CanonicalSO2Angle
    yaw_to_pose_rule_ref: DefinitionRef
    pose_yaw_binding_sha256: Sha256Digest


class CardinalYawAuthorization(HashBoundCanonicalModel):
    """One exact cardinal-yaw authorization without a hidden reference pivot."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/cardinal-yaw-authorization/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "cardinal_yaw_authorization_sha256"

    subject_id: CanonicalId
    operator_ref: DefinitionRef
    pivot_binding: FixedPivotBinding
    yaw: CardinalYaw
    cardinal_yaw_authorization_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_cardinal_authorization(self) -> Self:
        if self.subject_id != self.pivot_binding.subject_id:
            raise ValueError("authorization subject must match fixed pivot subject")
        if self.operator_ref == UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.OWN:
                raise ValueError("own-pivot operator requires an own pivot")
        elif self.operator_ref == UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.REFERENCE:
                raise ValueError("reference-pivot operator requires a reference pivot")
            if self.yaw.q == 0:
                raise ValueError("reference-pivot zero yaw is not canonical")
        else:
            raise ValueError(
                "cardinal authorization requires a registered cardinal operator"
            )
        return self


class ContinuousYawAuthorization(HashBoundCanonicalModel):
    """One continuous-yaw authorization with full-circle and zero rules closed."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-yaw-authorization/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_yaw_authorization_sha256"

    subject_id: CanonicalId
    operator_ref: DefinitionRef
    pivot_binding: FixedPivotBinding
    yaw_domain: ContinuousYawDomain
    continuous_yaw_authorization_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_authorization(self) -> Self:
        if self.subject_id != self.pivot_binding.subject_id:
            raise ValueError("authorization subject must match fixed pivot subject")
        if self.operator_ref == UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.OWN:
                raise ValueError("own-pivot operator requires an own pivot")
        elif self.operator_ref == UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.REFERENCE:
                raise ValueError("reference-pivot operator requires a reference pivot")
            if isinstance(self.yaw_domain, ContinuousYawFullCircle):
                raise ValueError(
                    "full circle is valid only for continuous own-pivot authorization"
                )
            if self.yaw_domain.contains_canonical_zero:
                raise ValueError("reference-pivot arc must exclude canonical zero")
        else:
            raise ValueError(
                "continuous authorization requires a registered continuous operator"
            )
        return self


# Resolve local model forward references before restoring public identities.
CanonicalSO2Angle.model_rebuild()
CardinalYaw.model_rebuild()
ContinuousYawArc.model_rebuild()
ContinuousYawFullCircle.model_rebuild()
ExactDyadic.model_rebuild()
LiftedYawInterval.model_rebuild()
ContinuousYawLift.model_rebuild()
FixedPivotBinding.model_rebuild()
ExplicitPoseYawBinding.model_rebuild()
CardinalYawAuthorization.model_rebuild()
ContinuousYawAuthorization.model_rebuild()


# Keep supported public import and pickle lookup stable.
_require_sorted_unique_by_bytes.__module__ = "spatialcf.domain.upright_se2"
_is_negative_zero.__module__ = "spatialcf.domain.upright_se2"
_fraction_from_float.__module__ = "spatialcf.domain.upright_se2"
CanonicalSO2Angle.__module__ = "spatialcf.domain.upright_se2"
canonical_yaw_from_upright_quaternion.__module__ = "spatialcf.domain.upright_se2"
validate_directed_yaw_quaternion_consistency.__module__ = "spatialcf.domain.upright_se2"
_compose_upright_quaternion_from_primary_yaw.__module__ = "spatialcf.domain.upright_se2"
CardinalYaw.__module__ = "spatialcf.domain.upright_se2"
ContinuousYawArc.__module__ = "spatialcf.domain.upright_se2"
ContinuousYawFullCircle.__module__ = "spatialcf.domain.upright_se2"
ExactDyadic.__module__ = "spatialcf.domain.upright_se2"
LiftedYawInterval.__module__ = "spatialcf.domain.upright_se2"
ContinuousYawLift.__module__ = "spatialcf.domain.upright_se2"
PivotMode.__module__ = "spatialcf.domain.upright_se2"
FixedPivotBinding.__module__ = "spatialcf.domain.upright_se2"
ExplicitPoseYawBinding.__module__ = "spatialcf.domain.upright_se2"
CardinalYawAuthorization.__module__ = "spatialcf.domain.upright_se2"
ContinuousYawAuthorization.__module__ = "spatialcf.domain.upright_se2"
