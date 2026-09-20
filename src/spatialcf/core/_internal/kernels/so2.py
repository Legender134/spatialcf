"""Resource-bounded rational foundations for directed SO(2) enclosures."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Self

from spatialcf.domain import upright_se2 as upright

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


class CardinalKernelKindV3(StrEnum):
    """Typed result class for additive exact-cardinal owner seams."""

    EXACT = "EXACT"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class CardinalKernelOutcomeV3:
    kind: CardinalKernelKindV3
    value: int | tuple[Fraction, Fraction] | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CardinalKernelKindV3:
            raise TypeError("cardinal kernel outcome must use CardinalKernelKindV3")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError("cardinal kernel atomic usage must be non-negative")
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"cardinal kernel {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is CardinalKernelKindV3.EXACT:
            if not (
                type(self.value) is int
                or (
                    type(self.value) is tuple
                    and len(self.value) == 2
                    and all(type(component) is Fraction for component in self.value)
                )
            ):
                raise ValueError("exact cardinal outcome must carry an exact value")
            if self.finding_codes or not self.proof_rows:
                raise ValueError(
                    "exact cardinal outcome must have proof and no findings"
                )
            return
        if self.value is not None or not self.finding_codes or not self.proof_rows:
            raise ValueError("non-exact cardinal outcome must have finding and proof")
        expected_prefix = {
            CardinalKernelKindV3.NUMERIC_GAP: "NUMERIC_GAP:",
            CardinalKernelKindV3.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            CardinalKernelKindV3.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("cardinal outcome finding does not match its typed kind")


def _require_cardinal_q_v3(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact cardinal quarter turn")
    if value not in (0, 1, 2, 3):
        raise ValueError(f"{label} must be in 0..3")
    return value


def _require_cardinal_budget_v3(value: object) -> SO2AtomicBudgetV2:
    if type(value) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    value.validate()
    return value


def _cardinal_resource_limit_outcome_v3() -> CardinalKernelOutcomeV3:
    return CardinalKernelOutcomeV3(
        CardinalKernelKindV3.RESOURCE_LIMIT,
        atomic_steps_used=0,
        proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:cap-minus-one",),
        finding_codes=("RESOURCE_LIMIT:SO2_ATOMIC_STEPS",),
    )


def compose_cardinal_quarter_turns_v3(
    left_q: object, right_q: object, *, atomic_budget: SO2AtomicBudgetV2
) -> CardinalKernelOutcomeV3:
    """Compose exact CCW quarter turns on the caller-owned atomic ledger."""
    checked_budget = _require_cardinal_budget_v3(atomic_budget)
    checked_left = _require_cardinal_q_v3(left_q, label="left_q")
    checked_right = _require_cardinal_q_v3(right_q, label="right_q")
    try:
        checked_budget.consume()
    except SO2AtomicBudgetExhaustedV2:
        return _cardinal_resource_limit_outcome_v3()
    return CardinalKernelOutcomeV3(
        CardinalKernelKindV3.EXACT,
        value=(checked_left + checked_right) % 4,
        atomic_steps_used=1,
        proof_rows=(f"CARDINAL_COMPOSE:q={checked_left}+q={checked_right}:mod-4",),
    )


def inverse_cardinal_quarter_turn_v3(
    q: object, *, atomic_budget: SO2AtomicBudgetV2
) -> CardinalKernelOutcomeV3:
    checked_budget = _require_cardinal_budget_v3(atomic_budget)
    checked_q = _require_cardinal_q_v3(q, label="q")
    return _inverse_cardinal_v3(checked_q, checked_budget)


def _inverse_cardinal_v3(
    q: int, atomic_budget: SO2AtomicBudgetV2
) -> CardinalKernelOutcomeV3:
    try:
        atomic_budget.consume()
    except SO2AtomicBudgetExhaustedV2:
        return _cardinal_resource_limit_outcome_v3()
    return CardinalKernelOutcomeV3(
        CardinalKernelKindV3.EXACT,
        value=(-q) % 4,
        atomic_steps_used=1,
        proof_rows=(f"CARDINAL_INVERSE:q={q}:mod-4",),
    )


def rotate_cardinal_fraction_xy_v3(
    x: object, y: object, q: object, *, atomic_budget: SO2AtomicBudgetV2
) -> CardinalKernelOutcomeV3:
    """Apply one exact signed coordinate permutation; no float/trig path exists."""
    checked_budget = _require_cardinal_budget_v3(atomic_budget)
    if type(x) is not Fraction or type(y) is not Fraction:
        raise TypeError("cardinal XY operands must be exact Fractions")
    checked_q = _require_cardinal_q_v3(q, label="q")
    try:
        checked_budget.consume()
    except SO2AtomicBudgetExhaustedV2:
        return _cardinal_resource_limit_outcome_v3()
    return CardinalKernelOutcomeV3(
        CardinalKernelKindV3.EXACT,
        value=((x, y), (-y, x), (-x, -y), (y, -x))[checked_q],
        atomic_steps_used=1,
        proof_rows=(f"CARDINAL_XY:q={checked_q}:exact-signed-permutation",),
    )


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


class ContinuousYawIntervalKindV4(StrEnum):
    """Typed result classes for additive lifted-turn interval owner seams."""

    EXACT = "EXACT"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class ContinuousYawLiftBoundsV4:
    """One canonical closed lifted-turn domain plus its unique coverage root."""

    lift: upright.ContinuousYawLift
    lower: Fraction
    upper: Fraction
    seam_ownership: str
    unrolled_branch_cut: bool
    upper_is_proof_closure_alias: bool
    coverage_sha256: str
    atomic_steps_used: int

    def __post_init__(self) -> None:
        if type(self.lift) is not upright.ContinuousYawLift:
            raise TypeError("lift must be a ContinuousYawLift")
        if type(self.lower) is not Fraction or type(self.upper) is not Fraction:
            raise TypeError("lifted turn bounds must be exact Fractions")
        _require_fraction_cap(self.lower)
        _require_fraction_cap(self.upper)
        if self.lower > self.upper:
            raise ValueError("lifted turn bounds must be ordered")
        interval = self.lift.intervals[0]
        if (self.lower, self.upper, self.seam_ownership) != (
            interval.lower.as_fraction,
            interval.upper.as_fraction,
            interval.seam_ownership,
        ):
            raise ValueError("lift bounds must reproduce the sealed domain lift")
        if type(self.unrolled_branch_cut) is not bool:
            raise TypeError("unrolled_branch_cut must be bool")
        if type(self.upper_is_proof_closure_alias) is not bool:
            raise TypeError("upper_is_proof_closure_alias must be bool")
        if type(self.coverage_sha256) is not str or len(self.coverage_sha256) != 64:
            raise ValueError("coverage_sha256 must be a SHA-256 digest")
        int(self.coverage_sha256, 16)
        if self.coverage_sha256 != self.lift.continuous_yaw_lift_sha256:
            raise ValueError("coverage digest must be the sealed lift digest")
        is_full_circle = type(self.lift.yaw_domain) is upright.ContinuousYawFullCircle
        if is_full_circle:
            if (
                (self.lower, self.upper, self.seam_ownership)
                != (Fraction(-1, 2), Fraction(1, 2), "LOWER_OWNS_SEAM")
                or self.unrolled_branch_cut
                or not self.upper_is_proof_closure_alias
            ):
                raise ValueError("full circle must have one lower-owned seam alias")
        elif self.upper_is_proof_closure_alias:
            raise ValueError("only the full circle may alias its upper endpoint")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("lift atomic usage must be a positive exact int")


@dataclass(frozen=True, slots=True)
class ContinuousYawLiftOutcomeV4:
    """Typed completion of canonical turn-domain lifting."""

    kind: ContinuousYawIntervalKindV4
    bounds: ContinuousYawLiftBoundsV4 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be non-blank strings")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is ContinuousYawIntervalKindV4.EXACT:
            if type(self.bounds) is not ContinuousYawLiftBoundsV4 or self.finding_codes:
                raise ValueError("exact lift outcome requires bounds and no findings")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError("non-exact lift outcome requires findings and no bounds")
        expected_prefix = {
            ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("lift finding does not match its typed outcome")


def _require_continuous_yaw_domain_v4(
    value: object,
) -> upright.ContinuousYawArc | upright.ContinuousYawFullCircle:
    if type(value) not in (upright.ContinuousYawArc, upright.ContinuousYawFullCircle):
        raise TypeError("yaw_domain must be a closed ContinuousYawDomain wire")
    return value


def _exact_dyadic_v4(value: Fraction) -> upright.ExactDyadic:
    _require_numeric_fraction_cap(value, "NUMERIC_GAP:LIFTED_TURN_FRACTION_BIT_CAP")
    denominator = value.denominator
    if denominator & (denominator - 1):
        raise _SO2NumericGapV2("NUMERIC_GAP:LIFTED_TURN_NOT_DYADIC")
    return upright.ExactDyadic(numerator=value.numerator, denominator=denominator)


def compile_continuous_yaw_lift_v4(
    yaw_domain: object,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawLiftOutcomeV4:
    """Lift one closed M3 yaw wire without splitting or rounded turn addition."""

    domain = _require_continuous_yaw_domain_v4(yaw_domain)
    checked_budget = _require_cardinal_budget_v3(atomic_budget)
    required_steps = 2
    if checked_budget.remaining < required_steps:
        return ContinuousYawLiftOutcomeV4(
            ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            finding_codes=("RESOURCE_LIMIT:SO2_ATOMIC_STEPS",),
        )
    start_used = checked_budget.used
    try:
        if type(domain) is upright.ContinuousYawFullCircle:
            lower = Fraction(-1, 2)
            upper = Fraction(1, 2)
            seam_ownership = "LOWER_OWNS_SEAM"
            unrolled_branch_cut = False
            upper_is_proof_closure_alias = True
        else:
            lower = Fraction.from_float(domain.start_angle.turns)
            sweep = Fraction.from_float(domain.ccw_sweep_turns)
            upper = lower + sweep
            _require_numeric_fraction_cap(
                upper,
                "NUMERIC_GAP:LIFTED_TURN_FRACTION_BIT_CAP",
            )
            seam_ownership = (
                "UPPER_OWNS_ENDPOINT" if upper == Fraction(1, 2) else "NONE"
            )
            unrolled_branch_cut = upper > Fraction(1, 2)
            upper_is_proof_closure_alias = False
        checked_budget.consume(required_steps)
        lift = upright.ContinuousYawLift.seal(
            yaw_domain=domain,
            lift_origin=_exact_dyadic_v4(Fraction(-1, 2)),
            intervals=(
                upright.LiftedYawInterval(
                    lower=_exact_dyadic_v4(lower),
                    upper=_exact_dyadic_v4(upper),
                    seam_ownership=seam_ownership,
                ),
            ),
        )
        return ContinuousYawLiftOutcomeV4(
            ContinuousYawIntervalKindV4.EXACT,
            bounds=ContinuousYawLiftBoundsV4(
                lift=lift,
                lower=lower,
                upper=upper,
                seam_ownership=seam_ownership,
                unrolled_branch_cut=unrolled_branch_cut,
                upper_is_proof_closure_alias=upper_is_proof_closure_alias,
                coverage_sha256=lift.continuous_yaw_lift_sha256,
                atomic_steps_used=checked_budget.used - start_used,
            ),
        )
    except SO2AtomicBudgetExhaustedV2:
        raise RuntimeError(
            "preflighted lift budget was unexpectedly exhausted"
        ) from None
    except _SO2NumericGapV2 as error:
        return ContinuousYawLiftOutcomeV4(
            ContinuousYawIntervalKindV4.NUMERIC_GAP,
            finding_codes=(error.finding_code,),
        )


@dataclass(frozen=True, slots=True)
class ContinuousYawSinCosBoundsV4:
    """Outward trigonometric extrema over one sealed lifted-turn interval."""

    lift_bounds: ContinuousYawLiftBoundsV4
    sine: RationalEnclosureV2
    cosine: RationalEnclosureV2
    critical_turns: tuple[Fraction, ...]
    atomic_steps_used: int

    def __post_init__(self) -> None:
        if type(self.lift_bounds) is not ContinuousYawLiftBoundsV4:
            raise TypeError("lift_bounds must be a ContinuousYawLiftBoundsV4")
        for field_name in ("sine", "cosine"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            if not (Fraction(-1) <= value.rational_lower <= value.rational_upper <= 1):
                raise ValueError(
                    f"{field_name} must lie in the trigonometric unit interval"
                )
        if type(self.critical_turns) is not tuple or any(
            type(turn) is not Fraction for turn in self.critical_turns
        ):
            raise TypeError("critical_turns must be exact Fractions")
        if self.critical_turns != tuple(sorted(set(self.critical_turns))):
            raise ValueError("critical_turns must be sorted and unique")
        if any(
            turn < self.lift_bounds.lower or turn > self.lift_bounds.upper
            for turn in self.critical_turns
        ):
            raise ValueError("critical turns must lie in the sealed lifted interval")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("trigonometric atomic usage must be a positive exact int")


@dataclass(frozen=True, slots=True)
class ContinuousYawSinCosOutcomeV4:
    """Typed completion of a whole-cell directed trigonometric enclosure."""

    kind: ContinuousYawIntervalKindV4
    bounds: ContinuousYawSinCosBoundsV4 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be non-blank strings")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is ContinuousYawIntervalKindV4.EXACT:
            if (
                type(self.bounds) is not ContinuousYawSinCosBoundsV4
                or self.finding_codes
            ):
                raise ValueError(
                    "exact sin/cos outcome requires bounds and no findings"
                )
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError(
                "non-exact sin/cos outcome requires findings and no bounds"
            )
        expected_prefix = {
            ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("sin/cos finding does not match its typed outcome")


def compile_lifted_turn_sin_cos_bounds_v4(
    lift_bounds: object,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawSinCosOutcomeV4:
    """Bound sine/cosine on one closed turn cell without sampling or libm trig."""

    if type(lift_bounds) is not ContinuousYawLiftBoundsV4:
        raise TypeError("lift_bounds must be a ContinuousYawLiftBoundsV4")
    budget = _require_cardinal_budget_v3(atomic_budget)
    start_used = budget.used
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            lower_sine, lower_cosine = _turn_point_sin_cos_v4(
                lift_bounds.lower,
                budget,
            )
            upper_sine, upper_cosine = _turn_point_sin_cos_v4(
                lift_bounds.upper,
                budget,
            )
            critical_turns = _turn_quarter_critical_angles_v4(
                lift_bounds.lower,
                lift_bounds.upper,
            )
            budget.consume(len(critical_turns))
            sine_lower = min(lower_sine[0], upper_sine[0])
            sine_upper = max(lower_sine[1], upper_sine[1])
            cosine_lower = min(lower_cosine[0], upper_cosine[0])
            cosine_upper = max(lower_cosine[1], upper_cosine[1])
            for turn in critical_turns:
                quarter = _cardinal_quarter_turn_v4(turn)
                if quarter is None:
                    raise RuntimeError("turn-quarter critical angle was not cardinal")
                exact_sine, exact_cosine = _exact_cardinal_sin_cos_v4(quarter)
                sine_lower = min(sine_lower, exact_sine)
                sine_upper = max(sine_upper, exact_sine)
                cosine_lower = min(cosine_lower, exact_cosine)
                cosine_upper = max(cosine_upper, exact_cosine)
            return ContinuousYawSinCosOutcomeV4(
                ContinuousYawIntervalKindV4.EXACT,
                bounds=ContinuousYawSinCosBoundsV4(
                    lift_bounds=lift_bounds,
                    sine=_publish_rational_enclosure(sine_lower, sine_upper),
                    cosine=_publish_rational_enclosure(cosine_lower, cosine_upper),
                    critical_turns=critical_turns,
                    atomic_steps_used=budget.used - start_used,
                ),
            )
    except SO2AtomicBudgetExhaustedV2:
        return ContinuousYawSinCosOutcomeV4(
            ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            finding_codes=("RESOURCE_LIMIT:SO2_ATOMIC_STEPS",),
        )
    except _SO2NumericGapV2 as error:
        return ContinuousYawSinCosOutcomeV4(
            ContinuousYawIntervalKindV4.NUMERIC_GAP,
            finding_codes=(error.finding_code,),
        )
    except (OverflowError, FloatingPointError):
        return ContinuousYawSinCosOutcomeV4(
            ContinuousYawIntervalKindV4.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:SO2_ARITHMETIC",),
        )
    except RuntimeWarning:
        return ContinuousYawSinCosOutcomeV4(
            ContinuousYawIntervalKindV4.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:SO2_RUNTIME_WARNING",),
        )


def _turn_point_sin_cos_v4(
    turn: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """Use the retained rational Taylor enclosure after exact turn reduction."""

    if type(turn) is not Fraction:
        raise TypeError("turn must be an exact Fraction")
    _require_numeric_fraction_cap(turn, "NUMERIC_GAP:LIFTED_TURN_FRACTION_BIT_CAP")
    budget.consume()
    quarter = _cardinal_quarter_turn_v4(turn)
    if quarter is not None:
        sine, cosine = _exact_cardinal_sin_cos_v4(quarter)
        return (sine, sine), (cosine, cosine)
    nearest_quarter = _floor_fraction(4 * turn + Fraction(1, 2))
    reduced_turn = turn - Fraction(nearest_quarter, 4)
    if not Fraction(-1, 8) <= reduced_turn <= Fraction(1, 8):
        raise RuntimeError("exact turn range reduction invariant failed")
    pi_lower, pi_upper = _machin_pi_enclosure_v4(budget)
    if reduced_turn >= 0:
        reduced_lower = 2 * reduced_turn * pi_lower
        reduced_upper = 2 * reduced_turn * pi_upper
    else:
        reduced_lower = 2 * reduced_turn * pi_upper
        reduced_upper = 2 * reduced_turn * pi_lower
    _require_numeric_fraction_cap(
        reduced_lower,
        "NUMERIC_GAP:TURN_REDUCED_ARGUMENT_FRACTION_BIT_CAP",
    )
    _require_numeric_fraction_cap(
        reduced_upper,
        "NUMERIC_GAP:TURN_REDUCED_ARGUMENT_FRACTION_BIT_CAP",
    )
    reduced = _publish_rational_enclosure(reduced_lower, reduced_upper)
    base_sine, base_cosine = _reduced_sin_cos_v2(reduced, budget)
    return _restore_quadrant_v2(
        base_sine,
        base_cosine,
        nearest_quarter % 4,
        negative=False,
        budget=budget,
    )


def _machin_pi_enclosure_v4(
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    """Return a fixed-order exact Machin enclosure for turn-domain reduction."""

    atan5 = _AlternatingAtanStateV2.create(5)
    atan239 = _AlternatingAtanStateV2.create(239)
    for _ in range(SO2_INTERVAL_TAYLOR_TERMS_V2):
        atan5.extend_one(budget)
        atan239.extend_one(budget)
    atan5_lower, atan5_upper = atan5.enclosure()
    atan239_lower, atan239_upper = atan239.enclosure()
    pi_lower = 16 * atan5_lower - 4 * atan239_upper
    pi_upper = 16 * atan5_upper - 4 * atan239_lower
    _require_numeric_fraction_cap(pi_lower, "NUMERIC_GAP:PI_FRACTION_BIT_CAP")
    _require_numeric_fraction_cap(pi_upper, "NUMERIC_GAP:PI_FRACTION_BIT_CAP")
    if pi_lower <= 0 or pi_lower > pi_upper:
        raise RuntimeError("Machin enclosure invariant failed")
    return pi_lower, pi_upper


def _turn_quarter_critical_angles_v4(
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, ...]:
    """Enumerate all exact n/4 extrema in deterministic lifted-turn order."""

    if type(lower) is not Fraction or type(upper) is not Fraction or lower > upper:
        raise ValueError("lifted turn endpoints must be ordered exact Fractions")
    first = _floor_fraction(4 * lower)
    last = _floor_fraction(4 * upper)
    return tuple(
        Fraction(index, 4)
        for index in range(first, last + 1)
        if lower <= Fraction(index, 4) <= upper
    )


def _cardinal_quarter_turn_v4(turn: Fraction) -> int | None:
    """Return the exact cardinal index when a lifted turn is an n/4 point."""

    quarter_turns = 4 * turn
    if quarter_turns.denominator != 1:
        return None
    return quarter_turns.numerator % 4


def _exact_cardinal_sin_cos_v4(quarter: int) -> tuple[Fraction, Fraction]:
    if quarter not in (0, 1, 2, 3):
        raise ValueError("quarter must be a cardinal index")
    return (
        (Fraction(), Fraction(1)),
        (Fraction(1), Fraction()),
        (Fraction(), Fraction(-1)),
        (Fraction(-1), Fraction()),
    )[quarter]


__all__ = (
    "SO2_INTERVAL_KERNEL_CERTIFIED_OUTWARD_ERROR_M_V2",
    "SO2_INTERVAL_KERNEL_ID_V2",
    "SO2_INTERVAL_KERNEL_SOUNDNESS_V2",
    "SO2_INTERVAL_KERNEL_VERSION_V2",
    "SO2_INTERVAL_MAX_FRACTION_BITS_V2",
    "SO2_INTERVAL_MAX_MACHIN_TERMS_V2",
    "SO2_INTERVAL_TAYLOR_TERMS_V2",
    "CardinalKernelKindV3",
    "CardinalKernelOutcomeV3",
    "ContinuousYawIntervalKindV4",
    "ContinuousYawLiftBoundsV4",
    "ContinuousYawLiftOutcomeV4",
    "ContinuousYawSinCosBoundsV4",
    "ContinuousYawSinCosOutcomeV4",
    "DirectedSinCosBoundsV2",
    "DirectedSinCosOutcomeV2",
    "RationalEnclosureV2",
    "SO2AtomicBudgetExhaustedV2",
    "SO2AtomicBudgetV2",
    "SO2IntervalKindV2",
    "compile_continuous_yaw_lift_v4",
    "compile_directed_sin_cos_v2",
    "compile_lifted_turn_sin_cos_bounds_v4",
    "compose_cardinal_quarter_turns_v3",
    "inverse_cardinal_quarter_turn_v3",
    "rotate_cardinal_fraction_xy_v3",
)
