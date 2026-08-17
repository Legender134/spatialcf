"""Resource-bounded rational foundations for directed SO(2) enclosures."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Self

SO2_INTERVAL_KERNEL_ID_V2 = "geometry-kernel:rational-so2-upright-box-directed-v2"
SO2_INTERVAL_KERNEL_VERSION_V2 = "kernel:2.2-continuous-yaw-upright-box"
SO2_INTERVAL_KERNEL_SOUNDNESS_V2 = "DIRECTED_OUTWARD_BOUNDS"
SO2_INTERVAL_KERNEL_CERTIFIED_OUTWARD_ERROR_M_V2 = 0.0

SO2_INTERVAL_MAX_FRACTION_BITS_V2 = 65_536
SO2_INTERVAL_MAX_MACHIN_TERMS_V2 = 512
SO2_INTERVAL_TAYLOR_TERMS_V2 = 24


class SO2IntervalKindV2(StrEnum):
    EXACT = "EXACT"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class SO2AtomicBudgetExhaustedV2(RuntimeError):
    """The shared directed-kernel ledger has no remaining capacity."""


@dataclass(frozen=True, slots=True)
class SO2AtomicBudgetV2:
    """One mutable-usage ledger shared across all SO(2)/OBB calls."""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def remaining(self) -> int:
        self.validate()
        return self.limit - self.used

    def validate(self) -> None:
        if type(self.limit) is not int:
            raise TypeError("SO2 atomic budget limit must be an exact int")
        if self.limit <= 0:
            raise ValueError("SO2 atomic budget limit must be positive")
        if type(self.used) is not int:
            raise TypeError("SO2 atomic budget used must be an exact int")
        if self.used < 0 or self.used > self.limit:
            raise ValueError("SO2 atomic budget used must lie within its limit")

    def consume(self, amount: int = 1) -> None:
        self.validate()
        if type(amount) is not int or amount < 0:
            raise TypeError("SO2 budget amount must be a non-negative exact int")
        if self.used + amount > self.limit:
            raise SO2AtomicBudgetExhaustedV2
        object.__setattr__(self, "used", self.used + amount)


class _SO2NumericGapV2(ArithmeticError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


@dataclass(frozen=True, slots=True)
class RationalEnclosureV2:
    """A rational enclosure plus its tight outward finite binary64 image."""

    rational_lower: Fraction
    rational_upper: Fraction
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        lower = self.rational_lower
        upper = self.rational_upper
        if type(lower) is not Fraction or type(upper) is not Fraction:
            raise TypeError("rational enclosure endpoints must be exact Fractions")
        _require_fraction_cap(lower)
        _require_fraction_cap(upper)
        if lower > upper:
            raise ValueError("rational enclosure endpoints are reversed")

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        if (
            type(lower_bound) is not float
            or type(upper_bound) is not float
            or not math.isfinite(lower_bound)
            or not math.isfinite(upper_bound)
            or lower_bound > upper_bound
        ):
            raise ValueError("published enclosure must use ordered finite floats")
        lower_bound = 0.0 if lower_bound == 0.0 else lower_bound
        upper_bound = 0.0 if upper_bound == 0.0 else upper_bound
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)

        published_lower = Fraction.from_float(lower_bound)
        published_upper = Fraction.from_float(upper_bound)
        if published_lower > lower or published_upper < upper:
            raise ValueError("published binary64 interval is not outward")
        if published_lower != lower:
            neighbor = math.nextafter(lower_bound, math.inf)
            if not math.isfinite(neighbor) or Fraction.from_float(neighbor) <= lower:
                raise ValueError("published lower endpoint is not tight")
        if published_upper != upper:
            neighbor = math.nextafter(upper_bound, -math.inf)
            if Fraction.from_float(neighbor) >= upper:
                raise ValueError("published upper endpoint is not tight")


@dataclass(frozen=True, slots=True)
class DirectedSinCosBoundsV2:
    """Certified directed enclosures for one exact binary64 yaw."""

    yaw_radians: float
    quadrant_index: int
    reduced_argument: RationalEnclosureV2
    sine: RationalEnclosureV2
    cosine: RationalEnclosureV2
    atomic_steps_used: int

    def __post_init__(self) -> None:
        yaw = self.yaw_radians
        if type(yaw) is not float or not math.isfinite(yaw):
            raise TypeError("yaw_radians must be a finite exact float")
        object.__setattr__(self, "yaw_radians", 0.0 if yaw == 0.0 else yaw)
        if type(self.quadrant_index) is not int or self.quadrant_index not in range(4):
            raise TypeError("quadrant_index must be an exact int in 0..3")
        for field_name in ("reduced_argument", "sine", "cosine"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            object.__setattr__(
                self,
                field_name,
                RationalEnclosureV2(
                    rational_lower=value.rational_lower,
                    rational_upper=value.rational_upper,
                    lower_bound=value.lower_bound,
                    upper_bound=value.upper_bound,
                ),
            )
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("atomic_steps_used must be a positive exact int")
        if not (
            -1 <= self.sine.rational_lower <= self.sine.rational_upper <= 1
            and -1 <= self.cosine.rational_lower <= self.cosine.rational_upper <= 1
        ):
            raise ValueError("trigonometric enclosures must lie within [-1, 1]")


@dataclass(frozen=True, slots=True)
class DirectedSinCosOutcomeV2:
    kind: SO2IntervalKindV2
    bounds: DirectedSinCosBoundsV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not SO2IntervalKindV2:
            raise TypeError("kind must be an SO2IntervalKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is SO2IntervalKindV2.EXACT:
            if type(self.bounds) is not DirectedSinCosBoundsV2:
                raise ValueError("EXACT sin/cos outcome requires bounds")
            checked = DirectedSinCosBoundsV2(
                yaw_radians=self.bounds.yaw_radians,
                quadrant_index=self.bounds.quadrant_index,
                reduced_argument=self.bounds.reduced_argument,
                sine=self.bounds.sine,
                cosine=self.bounds.cosine,
                atomic_steps_used=self.bounds.atomic_steps_used,
            )
            object.__setattr__(self, "bounds", checked)
            if self.finding_codes:
                raise ValueError("EXACT sin/cos outcome cannot carry findings")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError(
                "non-EXACT sin/cos outcome requires findings and no bounds"
            )


@dataclass(slots=True)
class _AlternatingAtanStateV2:
    reciprocal: int
    partial: Fraction
    next_term: Fraction
    included_terms: int

    @classmethod
    def create(cls, reciprocal: int) -> Self:
        return cls(
            reciprocal=reciprocal,
            partial=Fraction(),
            next_term=Fraction(1, reciprocal),
            included_terms=0,
        )

    def extend_one(self, budget: SO2AtomicBudgetV2) -> None:
        budget.consume()
        index = self.included_terms
        term = self.next_term
        self.partial += term
        self.next_term = (
            -term
            * (2 * index + 1)
            / ((2 * index + 3) * self.reciprocal * self.reciprocal)
        )
        self.included_terms += 1

    def enclosure(self) -> tuple[Fraction, Fraction]:
        other = self.partial + self.next_term
        return min(self.partial, other), max(self.partial, other)


def compile_directed_sin_cos_v2(
    yaw_radians: float,
    *,
    max_atomic_steps: int | None = None,
    atomic_budget: SO2AtomicBudgetV2 | None = None,
) -> DirectedSinCosOutcomeV2:
    """Enclose sin/cos for one exact binary64 yaw without trusting libm trig."""

    if type(yaw_radians) is not float or not math.isfinite(yaw_radians):
        return _sin_cos_failure(
            SO2IntervalKindV2.INVALID_INPUT,
            "INVALID_INPUT:YAW_RADIANS",
        )
    budget = _resolve_budget(max_atomic_steps, atomic_budget)
    start_used = budget.used
    yaw_radians = 0.0 if yaw_radians == 0.0 else yaw_radians
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_directed_sin_cos_checked_v2(
                yaw_radians,
                budget,
                start_used,
            )
    except SO2AtomicBudgetExhaustedV2:
        return _sin_cos_failure(
            SO2IntervalKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
        )
    except _SO2NumericGapV2 as error:
        return _sin_cos_failure(SO2IntervalKindV2.NUMERIC_GAP, error.finding_code)
    except (OverflowError, FloatingPointError):
        return _sin_cos_failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SO2_ARITHMETIC",
        )
    except RuntimeWarning:
        return _sin_cos_failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SO2_RUNTIME_WARNING",
        )


def _compile_directed_sin_cos_checked_v2(
    yaw_radians: float,
    budget: SO2AtomicBudgetV2,
    start_used: int,
) -> DirectedSinCosOutcomeV2:
    if yaw_radians == 0.0:
        budget.consume(4)
        exact_zero = _publish_rational_enclosure(Fraction(), Fraction())
        exact_one = _publish_rational_enclosure(Fraction(1), Fraction(1))
        return DirectedSinCosOutcomeV2(
            kind=SO2IntervalKindV2.EXACT,
            bounds=DirectedSinCosBoundsV2(
                yaw_radians=0.0,
                quadrant_index=0,
                reduced_argument=exact_zero,
                sine=exact_zero,
                cosine=exact_one,
                atomic_steps_used=budget.used - start_used,
            ),
        )
    quadrant, reduced = _reduce_yaw_v2(yaw_radians, budget)
    base_sine, base_cosine = _reduced_sin_cos_v2(reduced, budget)
    sine, cosine = _restore_quadrant_v2(
        base_sine,
        base_cosine,
        quadrant,
        negative=yaw_radians < 0.0,
        budget=budget,
    )
    return DirectedSinCosOutcomeV2(
        kind=SO2IntervalKindV2.EXACT,
        bounds=DirectedSinCosBoundsV2(
            yaw_radians=yaw_radians,
            quadrant_index=quadrant,
            reduced_argument=reduced,
            sine=_publish_rational_enclosure(*sine),
            cosine=_publish_rational_enclosure(*cosine),
            atomic_steps_used=budget.used - start_used,
        ),
    )


def _resolve_budget(
    max_atomic_steps: int | None,
    atomic_budget: SO2AtomicBudgetV2 | None,
) -> SO2AtomicBudgetV2:
    if (max_atomic_steps is None) == (atomic_budget is None):
        raise ValueError("provide exactly one SO2 atomic budget source")
    if atomic_budget is not None:
        if type(atomic_budget) is not SO2AtomicBudgetV2:
            raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
        atomic_budget.validate()
        return atomic_budget
    if type(max_atomic_steps) is not int:
        raise TypeError("max_atomic_steps must be an exact int")
    return SO2AtomicBudgetV2(limit=max_atomic_steps)


def _reduce_yaw_v2(
    yaw_radians: float,
    budget: SO2AtomicBudgetV2,
) -> tuple[int, RationalEnclosureV2]:
    exact_yaw = abs(Fraction.from_float(yaw_radians))
    atan5 = _AlternatingAtanStateV2.create(5)
    atan239 = _AlternatingAtanStateV2.create(239)
    for term_count in range(1, SO2_INTERVAL_MAX_MACHIN_TERMS_V2 + 1):
        atan5.extend_one(budget)
        atan239.extend_one(budget)
        if term_count < 8 or term_count % 8:
            continue
        budget.consume(4)
        atan5_lower, atan5_upper = atan5.enclosure()
        atan239_lower, atan239_upper = atan239.enclosure()
        pi_lower = 16 * atan5_lower - 4 * atan239_upper
        pi_upper = 16 * atan5_upper - 4 * atan239_lower
        _require_numeric_fraction_cap(pi_lower, "NUMERIC_GAP:PI_FRACTION_BIT_CAP")
        _require_numeric_fraction_cap(pi_upper, "NUMERIC_GAP:PI_FRACTION_BIT_CAP")
        if pi_lower <= 0 or pi_lower > pi_upper:
            raise RuntimeError("Machin enclosure invariant failed")
        quotient_lower = 2 * exact_yaw / pi_upper
        quotient_upper = 2 * exact_yaw / pi_lower
        nearest_lower = _floor_fraction(quotient_lower + Fraction(1, 2))
        nearest_upper = _floor_fraction(quotient_upper + Fraction(1, 2))
        if nearest_lower != nearest_upper:
            continue
        multiple = nearest_lower
        reduced_lower = exact_yaw - Fraction(multiple, 2) * pi_upper
        reduced_upper = exact_yaw - Fraction(multiple, 2) * pi_lower
        _require_numeric_fraction_cap(
            reduced_lower,
            "NUMERIC_GAP:REDUCED_ARGUMENT_FRACTION_BIT_CAP",
        )
        _require_numeric_fraction_cap(
            reduced_upper,
            "NUMERIC_GAP:REDUCED_ARGUMENT_FRACTION_BIT_CAP",
        )
        budget.consume(4)
        reduced = _publish_rational_enclosure(reduced_lower, reduced_upper)
        projected_lower = Fraction.from_float(reduced.lower_bound)
        projected_upper = Fraction.from_float(reduced.upper_bound)
        if projected_lower < -pi_upper / 4 or projected_upper > pi_upper / 4:
            raise _SO2NumericGapV2("NUMERIC_GAP:REDUCED_ARGUMENT_OUT_OF_RANGE")
        return multiple % 4, reduced
    raise _SO2NumericGapV2("NUMERIC_GAP:RANGE_REDUCTION_AMBIGUOUS")


def _reduced_sin_cos_v2(
    reduced: RationalEnclosureV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    lower = Fraction.from_float(reduced.lower_bound)
    upper = Fraction.from_float(reduced.upper_bound)
    if lower > upper or lower < -1 or upper > 1:
        raise RuntimeError("reduced argument publication invariant failed")
    sin_lower = _sin_point_enclosure_v2(lower, budget)[0]
    sin_upper = _sin_point_enclosure_v2(upper, budget)[1]
    maximum_abs = max(abs(lower), abs(upper))
    minimum_abs = Fraction() if lower <= 0 <= upper else min(abs(lower), abs(upper))
    cos_lower = _cos_point_enclosure_v2(maximum_abs, budget)[0]
    cos_upper = _cos_point_enclosure_v2(minimum_abs, budget)[1]
    return (sin_lower, sin_upper), (cos_lower, cos_upper)


def _sin_point_enclosure_v2(
    value: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    if value < 0:
        lower, upper = _sin_point_enclosure_v2(-value, budget)
        return -upper, -lower
    term = value
    partial = Fraction()
    squared = value * value
    for index in range(SO2_INTERVAL_TAYLOR_TERMS_V2):
        budget.consume()
        partial += term
        term = -term * squared / ((2 * index + 2) * (2 * index + 3))
        _require_numeric_fraction_cap(term, "NUMERIC_GAP:SIN_FRACTION_BIT_CAP")
        _require_numeric_fraction_cap(partial, "NUMERIC_GAP:SIN_FRACTION_BIT_CAP")
    other = partial + term
    _require_numeric_fraction_cap(other, "NUMERIC_GAP:SIN_FRACTION_BIT_CAP")
    return min(partial, other), max(partial, other)


def _cos_point_enclosure_v2(
    value: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    value = abs(value)
    term = Fraction(1)
    partial = Fraction()
    squared = value * value
    for index in range(SO2_INTERVAL_TAYLOR_TERMS_V2):
        budget.consume()
        partial += term
        term = -term * squared / ((2 * index + 1) * (2 * index + 2))
        _require_numeric_fraction_cap(term, "NUMERIC_GAP:COS_FRACTION_BIT_CAP")
        _require_numeric_fraction_cap(partial, "NUMERIC_GAP:COS_FRACTION_BIT_CAP")
    other = partial + term
    _require_numeric_fraction_cap(other, "NUMERIC_GAP:COS_FRACTION_BIT_CAP")
    return min(partial, other), max(partial, other)


def _restore_quadrant_v2(
    sine: tuple[Fraction, Fraction],
    cosine: tuple[Fraction, Fraction],
    quadrant: int,
    *,
    negative: bool,
    budget: SO2AtomicBudgetV2,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    budget.consume(4)
    if quadrant == 0:
        restored_sine, restored_cosine = sine, cosine
    elif quadrant == 1:
        restored_sine, restored_cosine = cosine, _negate_enclosure(sine)
    elif quadrant == 2:
        restored_sine = _negate_enclosure(sine)
        restored_cosine = _negate_enclosure(cosine)
    elif quadrant == 3:
        restored_sine, restored_cosine = _negate_enclosure(cosine), sine
    else:
        raise RuntimeError("quadrant invariant failed")
    if negative:
        restored_sine = _negate_enclosure(restored_sine)
    return _unit_intersection(restored_sine), _unit_intersection(restored_cosine)


def _unit_intersection(
    interval: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    lower = max(Fraction(-1), interval[0])
    upper = min(Fraction(1), interval[1])
    if lower > upper:
        raise RuntimeError("trigonometric unit interval invariant failed")
    return lower, upper


def _negate_enclosure(
    interval: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return -interval[1], -interval[0]


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _publish_rational_enclosure(
    lower: Fraction,
    upper: Fraction,
) -> RationalEnclosureV2:
    _require_numeric_fraction_cap(lower, "NUMERIC_GAP:PUBLICATION_FRACTION_BIT_CAP")
    _require_numeric_fraction_cap(upper, "NUMERIC_GAP:PUBLICATION_FRACTION_BIT_CAP")
    if lower > upper:
        raise RuntimeError("cannot publish a reversed rational enclosure")
    return RationalEnclosureV2(
        rational_lower=lower,
        rational_upper=upper,
        lower_bound=_fraction_floor_binary64(lower),
        upper_bound=_fraction_ceil_binary64(upper),
    )


def _fraction_floor_binary64(value: Fraction) -> float:
    nearest = _nearest_finite_binary64(value)
    if Fraction.from_float(nearest) > value:
        nearest = math.nextafter(nearest, -math.inf)
    if not math.isfinite(nearest):
        raise _SO2NumericGapV2("NUMERIC_GAP:NO_FINITE_BINARY64_FLOOR")
    return 0.0 if nearest == 0.0 else nearest


def _fraction_ceil_binary64(value: Fraction) -> float:
    nearest = _nearest_finite_binary64(value)
    if Fraction.from_float(nearest) < value:
        nearest = math.nextafter(nearest, math.inf)
    if not math.isfinite(nearest):
        raise _SO2NumericGapV2("NUMERIC_GAP:NO_FINITE_BINARY64_CEIL")
    return 0.0 if nearest == 0.0 else nearest


def _nearest_finite_binary64(value: Fraction) -> float:
    try:
        nearest = float(value)
    except (OverflowError, ValueError) as error:
        raise _SO2NumericGapV2("NUMERIC_GAP:NO_FINITE_BINARY64") from error
    if not math.isfinite(nearest):
        raise _SO2NumericGapV2("NUMERIC_GAP:NO_FINITE_BINARY64")
    return nearest


def _require_numeric_fraction_cap(value: Fraction, finding_code: str) -> None:
    if type(value) is not Fraction:
        raise TypeError("numeric value must be an exact Fraction")
    if (
        value.numerator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
        or value.denominator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
    ):
        raise _SO2NumericGapV2(finding_code)


def _sin_cos_failure(
    kind: SO2IntervalKindV2,
    finding_code: str,
) -> DirectedSinCosOutcomeV2:
    return DirectedSinCosOutcomeV2(kind=kind, finding_codes=(finding_code,))


def _require_fraction_cap(value: Fraction) -> None:
    if type(value) is not Fraction:
        raise TypeError("value must be an exact Fraction")
    if (
        value.numerator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
        or value.denominator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
    ):
        raise ValueError("rational value exceeds the SO2 Fraction bit cap")


__all__ = (
    "SO2_INTERVAL_KERNEL_CERTIFIED_OUTWARD_ERROR_M_V2",
    "SO2_INTERVAL_KERNEL_ID_V2",
    "SO2_INTERVAL_KERNEL_SOUNDNESS_V2",
    "SO2_INTERVAL_KERNEL_VERSION_V2",
    "SO2_INTERVAL_MAX_FRACTION_BITS_V2",
    "SO2_INTERVAL_MAX_MACHIN_TERMS_V2",
    "SO2_INTERVAL_TAYLOR_TERMS_V2",
    "DirectedSinCosBoundsV2",
    "DirectedSinCosOutcomeV2",
    "RationalEnclosureV2",
    "SO2AtomicBudgetExhaustedV2",
    "SO2AtomicBudgetV2",
    "SO2IntervalKindV2",
    "compile_directed_sin_cos_v2",
)
