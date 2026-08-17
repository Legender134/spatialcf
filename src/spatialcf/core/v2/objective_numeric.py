"""Exact, platform-neutral numeric primitives for the Canonical v2 objective.

Every optimization decision is performed with :class:`fractions.Fraction`.
Binary64 values appear only at the publication boundary and are rounded in the
declared outward direction.
"""

from __future__ import annotations

import math
import struct
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TypeVar

from spatialcf.core.v2.rect_kernel import ExactAxisAlignedRectV2
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearNearestKindV2,
    nearest_point_to_origin_rectilinear_v2,
)
from spatialcf.domain.v2.artifacts import RelationDamageBoundV2
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.objective import (
    ConstraintSafetyTargetV2,
    ObjectCameraKeyV2,
    RelationDamageAggregationV2,
    RelationDamageMetricV2,
    RelationDamageTermV2,
    SafetyAggregationKindV2,
    SafetyComponentKindV2,
    SafetyConstraintAggregationV2,
    SafetyMarginTermV2,
    SafetySlackUnitV2,
    TranslationMetricV2,
    TranslationTermV2,
    VisibilityChangeAggregationV2,
    VisibilityChangeMetricV2,
    VisibilityChangeTermV2,
)

OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2 = 4096
OBJECTIVE_NUMERIC_SQRT_ATOMIC_STEPS_V2 = 64

_MAX_FINITE_BINARY64_BITS = 0x7FEF_FFFF_FFFF_FFFF


class ObjectiveNumericKindV2(StrEnum):
    EXACT = "EXACT"
    EMPTY = "EMPTY"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, slots=True)
class ExactPublishedIntervalV2:
    """One exact rational interval and its tight outward binary64 image."""

    exact_lower: Fraction
    exact_upper: Fraction
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if (
            type(self.exact_lower) is not Fraction
            or type(self.exact_upper) is not Fraction
        ):
            raise TypeError("exact interval endpoints must be Fractions")
        if self.exact_lower > self.exact_upper:
            raise ValueError("exact interval lower endpoint exceeds upper endpoint")
        if not _fraction_is_within_cap(self.exact_lower) or not _fraction_is_within_cap(
            self.exact_upper
        ):
            raise ValueError("exact interval exceeds the registered Fraction bit cap")
        if (
            type(self.lower_bound) is not float
            or type(self.upper_bound) is not float
            or not math.isfinite(self.lower_bound)
            or not math.isfinite(self.upper_bound)
            or self.lower_bound > self.upper_bound
        ):
            raise ValueError("published interval must use ordered finite floats")
        lower_exact = Fraction.from_float(self.lower_bound)
        upper_exact = Fraction.from_float(self.upper_bound)
        if lower_exact > self.exact_lower or upper_exact < self.exact_upper:
            raise ValueError("published interval does not enclose the exact interval")
        if lower_exact != self.exact_lower:
            neighbor = math.nextafter(self.lower_bound, math.inf)
            if (
                not math.isfinite(neighbor)
                or Fraction.from_float(neighbor) <= self.exact_lower
            ):
                raise ValueError("published lower bound is not the tight predecessor")
        if upper_exact != self.exact_upper:
            neighbor = math.nextafter(self.upper_bound, -math.inf)
            if Fraction.from_float(neighbor) >= self.exact_upper:
                raise ValueError("published upper bound is not the tight successor")


@dataclass(frozen=True, slots=True)
class ObjectiveIntervalOutcomeV2:
    kind: ObjectiveNumericKindV2
    interval: ExactPublishedIntervalV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ObjectiveNumericKindV2:
            raise TypeError("kind must be an ObjectiveNumericKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            _canonical_findings(self.finding_codes),
        )
        if self.kind is ObjectiveNumericKindV2.EXACT:
            if type(self.interval) is not ExactPublishedIntervalV2:
                raise ValueError("EXACT interval outcome requires an interval")
            checked_interval = ExactPublishedIntervalV2(
                exact_lower=self.interval.exact_lower,
                exact_upper=self.interval.exact_upper,
                lower_bound=self.interval.lower_bound,
                upper_bound=self.interval.upper_bound,
            )
            object.__setattr__(self, "interval", checked_interval)
            if self.finding_codes:
                raise ValueError("EXACT interval outcome cannot carry findings")
            return
        if self.interval is not None:
            raise ValueError("non-EXACT interval outcome cannot carry an interval")
        if not self.finding_codes:
            raise ValueError("non-EXACT interval outcome requires a finding")


@dataclass(frozen=True, slots=True)
class TranslationCellBoundsV2:
    """Exact translation extrema plus their weighted directed publication."""

    nearest_delta_x_m: Fraction
    nearest_delta_y_m: Fraction
    minimum_squared_distance_m2: Fraction
    maximum_squared_distance_m2: Fraction
    weight_over_normalizer: Fraction
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        fractions = (
            self.nearest_delta_x_m,
            self.nearest_delta_y_m,
            self.minimum_squared_distance_m2,
            self.maximum_squared_distance_m2,
            self.weight_over_normalizer,
        )
        if any(type(value) is not Fraction for value in fractions):
            raise TypeError("translation exact values must be Fractions")
        if any(not _fraction_is_within_cap(value) for value in fractions):
            raise ValueError("translation exact value exceeds the Fraction bit cap")
        if (
            self.nearest_delta_x_m**2 + self.nearest_delta_y_m**2
            != self.minimum_squared_distance_m2
        ):
            raise ValueError("nearest witness does not match the minimum distance")
        if (
            self.minimum_squared_distance_m2 < 0
            or self.minimum_squared_distance_m2 > self.maximum_squared_distance_m2
            or self.weight_over_normalizer <= 0
        ):
            raise ValueError("translation exact extrema are invalid")
        if (
            type(self.lower_bound) is not float
            or type(self.upper_bound) is not float
            or not math.isfinite(self.lower_bound)
            or not math.isfinite(self.upper_bound)
            or self.lower_bound < 0
            or self.lower_bound > self.upper_bound
        ):
            raise ValueError("translation loss bounds must be ordered finite floats")
        minimum_loss_squared = (
            self.minimum_squared_distance_m2 * self.weight_over_normalizer**2
        )
        maximum_loss_squared = (
            self.maximum_squared_distance_m2 * self.weight_over_normalizer**2
        )
        if (
            Fraction.from_float(self.lower_bound) ** 2 > minimum_loss_squared
            or Fraction.from_float(self.upper_bound) ** 2 < maximum_loss_squared
        ):
            raise ValueError("translation loss bounds do not enclose the extrema")
        _require_tight_sqrt_bound(
            minimum_loss_squared,
            self.lower_bound,
            lower=True,
        )
        _require_tight_sqrt_bound(
            maximum_loss_squared,
            self.upper_bound,
            lower=False,
        )


@dataclass(frozen=True, slots=True)
class TranslationCellOutcomeV2:
    kind: ObjectiveNumericKindV2
    bounds: TranslationCellBoundsV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ObjectiveNumericKindV2:
            raise TypeError("kind must be an ObjectiveNumericKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            _canonical_findings(self.finding_codes),
        )
        if self.kind is ObjectiveNumericKindV2.EXACT:
            if type(self.bounds) is not TranslationCellBoundsV2:
                raise ValueError("EXACT translation outcome requires bounds")
            checked_bounds = TranslationCellBoundsV2(
                nearest_delta_x_m=self.bounds.nearest_delta_x_m,
                nearest_delta_y_m=self.bounds.nearest_delta_y_m,
                minimum_squared_distance_m2=self.bounds.minimum_squared_distance_m2,
                maximum_squared_distance_m2=self.bounds.maximum_squared_distance_m2,
                weight_over_normalizer=self.bounds.weight_over_normalizer,
                lower_bound=self.bounds.lower_bound,
                upper_bound=self.bounds.upper_bound,
            )
            object.__setattr__(self, "bounds", checked_bounds)
            if self.finding_codes:
                raise ValueError("EXACT translation outcome cannot carry findings")
            return
        if self.bounds is not None:
            raise ValueError("non-EXACT translation outcome cannot carry bounds")
        if self.kind is ObjectiveNumericKindV2.EMPTY:
            if self.finding_codes:
                raise ValueError("EMPTY translation outcome cannot carry findings")
            return
        if not self.finding_codes:
            raise ValueError("non-EXACT translation outcome requires a finding")


@dataclass(frozen=True, slots=True)
class VisibilityMetricIntervalV2:
    key: ObjectCameraKeyV2
    baseline_lower: Fraction
    baseline_upper: Fraction
    candidate_lower: Fraction
    candidate_upper: Fraction

    def __post_init__(self) -> None:
        checked_key = _strict_model(self.key, ObjectCameraKeyV2, "key")
        object.__setattr__(self, "key", checked_key)
        values = (
            self.baseline_lower,
            self.baseline_upper,
            self.candidate_lower,
            self.candidate_upper,
        )
        if any(type(value) is not Fraction for value in values):
            raise TypeError("visibility interval endpoints must be Fractions")
        if not (
            0 <= self.baseline_lower <= self.baseline_upper <= 1
            and 0 <= self.candidate_lower <= self.candidate_upper <= 1
        ):
            raise ValueError("visibility intervals must be ordered within [0, 1]")


@dataclass(frozen=True, slots=True)
class SafetyRawComponentIntervalV2:
    kind: SafetyComponentKindV2
    unit: SafetySlackUnitV2
    raw_lower: Fraction
    raw_upper: Fraction

    def __post_init__(self) -> None:
        if type(self.kind) is not SafetyComponentKindV2:
            raise TypeError("kind must be a SafetyComponentKindV2")
        if type(self.unit) is not SafetySlackUnitV2:
            raise TypeError("unit must be a SafetySlackUnitV2")
        if type(self.raw_lower) is not Fraction or type(self.raw_upper) is not Fraction:
            raise TypeError("raw safety endpoints must be Fractions")
        if self.raw_lower > self.raw_upper:
            raise ValueError("raw safety interval is reversed")


@dataclass(frozen=True, slots=True)
class ConstraintSafetyInputV2:
    constraint_id: str
    components: tuple[SafetyRawComponentIntervalV2, ...]

    def __post_init__(self) -> None:
        if type(self.constraint_id) is not str or not self.constraint_id:
            raise TypeError("constraint_id must be a non-empty exact str")
        object.__setattr__(
            self,
            "components",
            _revalidate_safety_components(self.components),
        )


class _NumericGapV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


ModelT = TypeVar("ModelT", bound=V2Model)


def publish_nonnegative_fraction_interval_v2(
    exact_lower: object,
    exact_upper: object,
) -> ObjectiveIntervalOutcomeV2:
    """Publish a non-negative rational interval with tight outward rounding."""

    try:
        lower, upper = _checked_fraction_interval(
            exact_lower,
            exact_upper,
            nonnegative=True,
        )
        return _publish_signed_fraction_interval_v2(lower, upper)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def publish_fraction_interval_v2(
    exact_lower: object,
    exact_upper: object,
) -> ObjectiveIntervalOutcomeV2:
    """Publish a signed rational interval with tight outward rounding."""

    try:
        lower, upper = _checked_fraction_interval(
            exact_lower,
            exact_upper,
            nonnegative=False,
        )
        return _publish_signed_fraction_interval_v2(lower, upper)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def compile_translation_l2_cell_bounds_v2(
    region: ExactRectilinearRegionV2,
    term: TranslationTermV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> TranslationCellOutcomeV2:
    """Compute weighted L2 minimum/maximum over one exact translation cell."""

    checked_term = _strict_model(term, TranslationTermV2, "term")
    if checked_term.metric is not TranslationMetricV2.EUCLIDEAN_L2_WORLD_XY_METRE:
        return _invalid_translation("INVALID_TRANSLATION_TERM:METRIC")
    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if type(region) is not ExactRectilinearRegionV2:
        raise TypeError("region must be an ExactRectilinearRegionV2")
    if type(region.rectangles) is not tuple:
        raise TypeError("region rectangles must be an exact tuple")
    try:
        budget.consume(len(region.rectangles))
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource_translation()
    try:
        for rectangle in region.rectangles:
            if type(rectangle) is not ExactAxisAlignedRectV2:
                raise TypeError("region rectangles must be exact rectangles")
            bounds = rectangle.bounds
            if bounds is None:
                raise ValueError("region cannot contain an EMPTY rectangle")
            for coordinate in bounds:
                _require_fraction_cap(
                    coordinate,
                    "NUMERIC_GAP:TRANSLATION_FRACTION_BIT_CAP",
                )
    except _NumericGapV2 as error:
        return _numeric_gap_translation(error.finding_code)
    nearest = nearest_point_to_origin_rectilinear_v2(
        region,
        atomic_budget=budget,
    )
    if nearest.kind is RectilinearNearestKindV2.EMPTY:
        return TranslationCellOutcomeV2(kind=ObjectiveNumericKindV2.EMPTY)
    if nearest.kind is RectilinearNearestKindV2.RESOURCE_LIMIT:
        return _resource_translation()
    if nearest.kind is RectilinearNearestKindV2.NUMERIC_GAP:
        return _numeric_gap_translation(*nearest.finding_codes)
    if nearest.nearest_point is None:
        raise ValueError("EXACT nearest outcome omitted its witness")
    try:
        budget.consume(len(region.rectangles))
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource_translation()

    try:
        maximum_squared = max(
            x**2 + y**2
            for rectangle in region.rectangles
            for x in (rectangle.min_x_m, rectangle.max_x_m)
            for y in (rectangle.min_y_m, rectangle.max_y_m)
        )
        minimum_squared = nearest.nearest_point.squared_distance_m2
        factor = Fraction.from_float(checked_term.weight) / Fraction.from_float(
            checked_term.normalizer_m
        )
        for value in (minimum_squared, maximum_squared, factor):
            _require_fraction_cap(value, "NUMERIC_GAP:TRANSLATION_FRACTION_BIT_CAP")
        minimum_weighted_squared = minimum_squared * factor**2
        maximum_weighted_squared = maximum_squared * factor**2
        _require_fraction_cap(
            minimum_weighted_squared,
            "NUMERIC_GAP:TRANSLATION_FRACTION_BIT_CAP",
        )
        _require_fraction_cap(
            maximum_weighted_squared,
            "NUMERIC_GAP:TRANSLATION_FRACTION_BIT_CAP",
        )
        try:
            budget.consume(2 * OBJECTIVE_NUMERIC_SQRT_ATOMIC_STEPS_V2)
        except RectilinearAtomicBudgetExhaustedV2:
            return _resource_translation()
        lower_pair = _directed_sqrt_binary64_bounds(minimum_weighted_squared)
        upper_pair = _directed_sqrt_binary64_bounds(maximum_weighted_squared)
        if lower_pair is None or upper_pair is None:
            raise _NumericGapV2("NUMERIC_GAP:TRANSLATION_NOT_FINITE_BINARY64")
        return TranslationCellOutcomeV2(
            kind=ObjectiveNumericKindV2.EXACT,
            bounds=TranslationCellBoundsV2(
                nearest_delta_x_m=nearest.nearest_point.delta_x_m,
                nearest_delta_y_m=nearest.nearest_point.delta_y_m,
                minimum_squared_distance_m2=minimum_squared,
                maximum_squared_distance_m2=maximum_squared,
                weight_over_normalizer=factor,
                lower_bound=lower_pair[0],
                upper_bound=upper_pair[1],
            ),
        )
    except _NumericGapV2 as error:
        return _numeric_gap_translation(error.finding_code)


def aggregate_relation_damage_bounds_v2(
    term: RelationDamageTermV2,
    vector: tuple[RelationDamageBoundV2, ...],
) -> ObjectiveIntervalOutcomeV2:
    """Aggregate a complete relation-damage vector using exact arithmetic."""

    checked_term = _strict_model(term, RelationDamageTermV2, "term")
    if (
        checked_term.metric
        is not RelationDamageMetricV2.PAIR_AXIS_SATISFIED_LABEL_SET_CHANGED_INDICATOR
        or checked_term.aggregation is not RelationDamageAggregationV2.WEIGHTED_SUM
    ):
        return _invalid_interval("INVALID_RELATION_DAMAGE_TERM:SEMANTICS")
    if type(vector) is not tuple:
        raise TypeError("vector must be an exact tuple")
    checked_vector = tuple(
        _strict_model(item, RelationDamageBoundV2, "relation damage bound")
        for item in vector
    )
    keys = tuple(item.key for item in checked_vector)
    if len(keys) != len(set(keys)):
        return _invalid_interval("INVALID_RELATION_DAMAGE_VECTOR:DUPLICATE_KEY")
    expected = {item.key for item in checked_term.pair_axis_weights}
    provided = set(keys)
    findings = _key_closure_findings(
        expected,
        provided,
        "RELATION_DAMAGE_VECTOR",
    )
    if findings:
        return _invalid_interval(*findings)
    by_key = {item.key: item for item in checked_vector}
    try:
        lower = Fraction()
        upper = Fraction()
        for weighted in checked_term.pair_axis_weights:
            weight = Fraction.from_float(weighted.damage_weight)
            bound = by_key[weighted.key]
            lower += weight * Fraction.from_float(bound.lower_bound)
            upper += weight * Fraction.from_float(bound.upper_bound)
            _require_fraction_cap(lower, "NUMERIC_GAP:RELATION_DAMAGE_FRACTION_BIT_CAP")
            _require_fraction_cap(upper, "NUMERIC_GAP:RELATION_DAMAGE_FRACTION_BIT_CAP")
        factor = Fraction.from_float(checked_term.weight) / Fraction.from_float(
            checked_term.normalizer
        )
        return _publish_signed_fraction_interval_v2(lower * factor, upper * factor)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def absolute_normalized_delta_bounds_v2(
    baseline_lower: object,
    baseline_upper: object,
    candidate_lower: object,
    candidate_upper: object,
) -> ObjectiveIntervalOutcomeV2:
    """Return the exact range of ``abs(baseline - candidate)`` for two intervals."""

    try:
        baseline = _checked_fraction_interval(
            baseline_lower,
            baseline_upper,
            nonnegative=True,
        )
        candidate = _checked_fraction_interval(
            candidate_lower,
            candidate_upper,
            nonnegative=True,
        )
        if baseline[1] > 1 or candidate[1] > 1:
            raise _NumericGapV2("NUMERIC_GAP:NORMALIZED_INTERVAL_OUT_OF_RANGE")
        lower = max(
            Fraction(),
            baseline[0] - candidate[1],
            candidate[0] - baseline[1],
        )
        upper = max(
            baseline[1] - candidate[0],
            candidate[1] - baseline[0],
        )
        return _publish_signed_fraction_interval_v2(lower, upper)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def aggregate_visibility_change_bounds_v2(
    term: VisibilityChangeTermV2,
    intervals: tuple[VisibilityMetricIntervalV2, ...],
) -> ObjectiveIntervalOutcomeV2:
    """Aggregate complete normalized visibility interval pairs."""

    checked_term = _strict_model(term, VisibilityChangeTermV2, "term")
    if (
        checked_term.metric
        is not VisibilityChangeMetricV2.ABSOLUTE_NORMALIZED_METRIC_DELTA
        or checked_term.aggregation is not VisibilityChangeAggregationV2.WEIGHTED_SUM
    ):
        return _invalid_interval("INVALID_VISIBILITY_CHANGE_TERM:SEMANTICS")
    if type(intervals) is not tuple:
        raise TypeError("intervals must be an exact tuple of visibility intervals")
    checked_intervals = tuple(
        _revalidate_visibility_interval(item) for item in intervals
    )
    keys = tuple(item.key for item in checked_intervals)
    if len(keys) != len(set(keys)):
        return _invalid_interval("INVALID_VISIBILITY_INTERVALS:DUPLICATE_KEY")
    expected = {item.key for item in checked_term.object_camera_weights}
    provided = set(keys)
    findings = _key_closure_findings(expected, provided, "VISIBILITY_INTERVALS")
    if findings:
        return _invalid_interval(*findings)
    by_key = {item.key: item for item in checked_intervals}
    try:
        lower = Fraction()
        upper = Fraction()
        for weighted in checked_term.object_camera_weights:
            interval = by_key[weighted.key]
            delta_lower, delta_upper = _absolute_delta_exact(interval)
            weight = Fraction.from_float(weighted.change_weight)
            lower += weight * delta_lower
            upper += weight * delta_upper
            _require_fraction_cap(lower, "NUMERIC_GAP:VISIBILITY_FRACTION_BIT_CAP")
            _require_fraction_cap(upper, "NUMERIC_GAP:VISIBILITY_FRACTION_BIT_CAP")
        factor = Fraction.from_float(checked_term.weight) / Fraction.from_float(
            checked_term.normalizer
        )
        return _publish_signed_fraction_interval_v2(lower * factor, upper * factor)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def compile_constraint_slack_bounds_v2(
    target: ConstraintSafetyTargetV2,
    components: tuple[SafetyRawComponentIntervalV2, ...],
) -> ObjectiveIntervalOutcomeV2:
    """Normalize component margins and take the constraint-wise minimum."""

    checked_target = _strict_model(target, ConstraintSafetyTargetV2, "target")
    if (
        checked_target.component_aggregation
        is not SafetyConstraintAggregationV2.MIN_NORMALIZED_COMPONENT_MARGIN
    ):
        return _invalid_interval("INVALID_SAFETY_TARGET:AGGREGATION")
    checked_components = _revalidate_safety_components(components)
    exact = _constraint_slack_exact(checked_target, checked_components)
    if isinstance(exact, ObjectiveIntervalOutcomeV2):
        return exact
    return _publish_signed_fraction_interval_v2(*exact)


def compile_target_deficit_bounds_v2(
    target: ConstraintSafetyTargetV2,
    constraint_slack: ExactPublishedIntervalV2,
) -> ObjectiveIntervalOutcomeV2:
    """Apply the reverse-monotone target-deficit formula to a slack interval."""

    checked_target = _strict_model(target, ConstraintSafetyTargetV2, "target")
    if (
        checked_target.component_aggregation
        is not SafetyConstraintAggregationV2.MIN_NORMALIZED_COMPONENT_MARGIN
    ):
        return _invalid_interval("INVALID_SAFETY_TARGET:AGGREGATION")
    if type(constraint_slack) is not ExactPublishedIntervalV2:
        raise TypeError("constraint_slack must be an ExactPublishedIntervalV2")
    checked_slack = ExactPublishedIntervalV2(
        exact_lower=constraint_slack.exact_lower,
        exact_upper=constraint_slack.exact_upper,
        lower_bound=constraint_slack.lower_bound,
        upper_bound=constraint_slack.upper_bound,
    )
    try:
        lower, upper = _target_deficit_exact(
            checked_target,
            checked_slack.exact_lower,
            checked_slack.exact_upper,
        )
        return _publish_signed_fraction_interval_v2(lower, upper)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def aggregate_safety_penalty_bounds_v2(
    term: SafetyMarginTermV2,
    constraints: tuple[ConstraintSafetyInputV2, ...],
) -> ObjectiveIntervalOutcomeV2:
    """Aggregate all target deficits with exact normalization and weights."""

    checked_term = _strict_model(term, SafetyMarginTermV2, "term")
    if (
        checked_term.aggregation.kind
        is not SafetyAggregationKindV2.SUM_NORMALIZED_DEFICIT
    ):
        return _invalid_interval("INVALID_SAFETY_TERM:AGGREGATION")
    if any(
        target.component_aggregation
        is not SafetyConstraintAggregationV2.MIN_NORMALIZED_COMPONENT_MARGIN
        for target in checked_term.aggregation.targets
    ):
        return _invalid_interval("INVALID_SAFETY_TARGET:AGGREGATION")
    if type(constraints) is not tuple:
        raise TypeError("constraints must be an exact tuple of safety inputs")
    checked_constraints = tuple(
        _revalidate_constraint_safety(item) for item in constraints
    )
    ids = tuple(item.constraint_id for item in checked_constraints)
    if len(ids) != len(set(ids)):
        return _invalid_interval("INVALID_SAFETY_INPUTS:DUPLICATE_CONSTRAINT")
    expected = {item.constraint_id for item in checked_term.aggregation.targets}
    provided = set(ids)
    findings = _key_closure_findings(expected, provided, "SAFETY_INPUTS")
    if findings:
        return _invalid_interval(*findings)
    by_id = {item.constraint_id: item for item in checked_constraints}
    try:
        lower = Fraction()
        upper = Fraction()
        for target in checked_term.aggregation.targets:
            exact_slack = _constraint_slack_exact(
                target, by_id[target.constraint_id].components
            )
            if isinstance(exact_slack, ObjectiveIntervalOutcomeV2):
                return exact_slack
            deficit_lower, deficit_upper = _target_deficit_exact(
                target,
                exact_slack[0],
                exact_slack[1],
            )
            importance = Fraction.from_float(target.importance)
            lower += importance * deficit_lower
            upper += importance * deficit_upper
            _require_fraction_cap(lower, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
            _require_fraction_cap(upper, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
        factor = Fraction.from_float(checked_term.weight) / Fraction.from_float(
            checked_term.normalizer
        )
        return _publish_signed_fraction_interval_v2(lower * factor, upper * factor)
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def _strict_model(value: object, model_type: type[ModelT], label: str) -> ModelT:
    if type(value) is not model_type:
        raise TypeError(f"{label} must be a {model_type.__name__}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"), strict=True
            )
    except Warning as error:
        raise TypeError(f"{label} failed strict serialization") from error


def _canonical_findings(values: object) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError("finding_codes must be an exact tuple")
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("finding codes must be non-blank exact strings")
    return tuple(sorted(set(values)))


def _revalidate_visibility_interval(value: object) -> VisibilityMetricIntervalV2:
    if type(value) is not VisibilityMetricIntervalV2:
        raise TypeError("visibility interval has the wrong type")
    return VisibilityMetricIntervalV2(
        key=value.key,
        baseline_lower=value.baseline_lower,
        baseline_upper=value.baseline_upper,
        candidate_lower=value.candidate_lower,
        candidate_upper=value.candidate_upper,
    )


def _revalidate_safety_components(
    values: object,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    if type(values) is not tuple:
        raise TypeError("components must be an exact tuple of safety intervals")
    checked: list[SafetyRawComponentIntervalV2] = []
    for value in values:
        if type(value) is not SafetyRawComponentIntervalV2:
            raise TypeError("safety component has the wrong type")
        checked.append(
            SafetyRawComponentIntervalV2(
                kind=value.kind,
                unit=value.unit,
                raw_lower=value.raw_lower,
                raw_upper=value.raw_upper,
            )
        )
    return tuple(checked)


def _revalidate_constraint_safety(value: object) -> ConstraintSafetyInputV2:
    if type(value) is not ConstraintSafetyInputV2:
        raise TypeError("constraint safety input has the wrong type")
    return ConstraintSafetyInputV2(
        constraint_id=value.constraint_id,
        components=_revalidate_safety_components(value.components),
    )


def _checked_fraction_interval(
    lower: object,
    upper: object,
    *,
    nonnegative: bool,
) -> tuple[Fraction, Fraction]:
    if type(lower) is not Fraction or type(upper) is not Fraction:
        raise _NumericGapV2("NUMERIC_GAP:EXPECTED_EXACT_FRACTION")
    _require_fraction_cap(lower, "NUMERIC_GAP:FRACTION_BIT_CAP")
    _require_fraction_cap(upper, "NUMERIC_GAP:FRACTION_BIT_CAP")
    if lower > upper:
        raise _NumericGapV2("NUMERIC_GAP:REVERSED_INTERVAL")
    if nonnegative and lower < 0:
        raise _NumericGapV2("NUMERIC_GAP:NEGATIVE_INTERVAL")
    return lower, upper


def _publish_signed_fraction_interval_v2(
    exact_lower: Fraction,
    exact_upper: Fraction,
) -> ObjectiveIntervalOutcomeV2:
    try:
        lower, upper = _checked_fraction_interval(
            exact_lower,
            exact_upper,
            nonnegative=False,
        )
        return ObjectiveIntervalOutcomeV2(
            kind=ObjectiveNumericKindV2.EXACT,
            interval=ExactPublishedIntervalV2(
                exact_lower=lower,
                exact_upper=upper,
                lower_bound=_fraction_to_float_floor(lower),
                upper_bound=_fraction_to_float_ceil(upper),
            ),
        )
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def _fraction_to_float_floor(value: Fraction) -> float:
    try:
        candidate = float(value)
    except OverflowError as error:
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64") from error
    if not math.isfinite(candidate):
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64")
    if Fraction.from_float(candidate) > value:
        candidate = math.nextafter(candidate, -math.inf)
    if not math.isfinite(candidate):
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64")
    return candidate


def _fraction_to_float_ceil(value: Fraction) -> float:
    try:
        candidate = float(value)
    except OverflowError as error:
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64") from error
    if not math.isfinite(candidate):
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64")
    if Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    if not math.isfinite(candidate):
        raise _NumericGapV2("NUMERIC_GAP:NON_FINITE_BINARY64")
    return candidate


def _fraction_is_within_cap(value: Fraction) -> bool:
    return (
        value.numerator.bit_length() <= OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2
        and value.denominator.bit_length() <= OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2
    )


def _require_fraction_cap(value: Fraction, finding_code: str) -> None:
    if type(value) is not Fraction:
        raise TypeError("internal exact value must be a Fraction")
    if not _fraction_is_within_cap(value):
        raise _NumericGapV2(finding_code)


def _directed_sqrt_binary64_bounds(value: Fraction) -> tuple[float, float] | None:
    if value < 0:
        raise ValueError("squared value must be non-negative")
    low_bits = 0
    high_bits = _MAX_FINITE_BINARY64_BITS
    best_bits = 0
    while low_bits <= high_bits:
        middle_bits = (low_bits + high_bits) // 2
        candidate = _positive_float_from_bits(middle_bits)
        if Fraction.from_float(candidate) ** 2 <= value:
            best_bits = middle_bits
            low_bits = middle_bits + 1
        else:
            high_bits = middle_bits - 1
    lower = _positive_float_from_bits(best_bits)
    if Fraction.from_float(lower) ** 2 == value:
        return lower, lower
    if best_bits == _MAX_FINITE_BINARY64_BITS:
        return None
    return lower, _positive_float_from_bits(best_bits + 1)


def _positive_float_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _require_tight_sqrt_bound(value: Fraction, bound: float, *, lower: bool) -> None:
    exact = Fraction.from_float(bound) ** 2
    if exact == value:
        return
    neighbor = math.nextafter(bound, math.inf if lower else -math.inf)
    neighbor_squared = Fraction.from_float(neighbor) ** 2
    if lower and neighbor_squared <= value:
        raise ValueError("translation lower bound is not tight")
    if not lower and neighbor_squared >= value:
        raise ValueError("translation upper bound is not tight")


def _resolve_atomic_budget(
    max_atomic_cells: int | None,
    atomic_budget: RectilinearAtomicBudgetV2 | None,
) -> RectilinearAtomicBudgetV2:
    if (max_atomic_cells is None) == (atomic_budget is None):
        raise ValueError("provide exactly one of max_atomic_cells or atomic_budget")
    if atomic_budget is not None:
        if type(atomic_budget) is not RectilinearAtomicBudgetV2:
            raise TypeError("atomic_budget must be a RectilinearAtomicBudgetV2")
        atomic_budget.validate()
        return atomic_budget
    if type(max_atomic_cells) is not int:
        raise TypeError("max_atomic_cells must be an exact int")
    return RectilinearAtomicBudgetV2(limit=max_atomic_cells)


def _absolute_delta_exact(
    interval: VisibilityMetricIntervalV2,
) -> tuple[Fraction, Fraction]:
    for value in (
        interval.baseline_lower,
        interval.baseline_upper,
        interval.candidate_lower,
        interval.candidate_upper,
    ):
        _require_fraction_cap(value, "NUMERIC_GAP:VISIBILITY_FRACTION_BIT_CAP")
    return (
        max(
            Fraction(),
            interval.baseline_lower - interval.candidate_upper,
            interval.candidate_lower - interval.baseline_upper,
        ),
        max(
            interval.baseline_upper - interval.candidate_lower,
            interval.candidate_upper - interval.baseline_lower,
        ),
    )


def _constraint_slack_exact(
    target: ConstraintSafetyTargetV2,
    components: tuple[SafetyRawComponentIntervalV2, ...],
) -> tuple[Fraction, Fraction] | ObjectiveIntervalOutcomeV2:
    if type(components) is not tuple or any(
        not isinstance(item, SafetyRawComponentIntervalV2) for item in components
    ):
        raise TypeError("components must be an exact tuple of safety intervals")
    kinds = tuple(item.kind for item in components)
    if len(kinds) != len(set(kinds)):
        return _invalid_interval("INVALID_SAFETY_COMPONENTS:DUPLICATE_KIND")
    expected = {item.kind for item in target.components}
    provided = set(kinds)
    findings = _key_closure_findings(expected, provided, "SAFETY_COMPONENTS")
    if findings:
        return _invalid_interval(*findings)
    by_kind = {item.kind: item for item in components}
    try:
        lowers: list[Fraction] = []
        uppers: list[Fraction] = []
        for specification in target.components:
            raw = by_kind[specification.kind]
            if raw.unit is not specification.unit:
                return _invalid_interval(
                    f"INVALID_SAFETY_COMPONENTS:UNIT:{specification.kind.value}"
                )
            _require_fraction_cap(raw.raw_lower, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
            _require_fraction_cap(raw.raw_upper, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
            normalizer = Fraction.from_float(specification.normalizer)
            lowers.append(raw.raw_lower / normalizer)
            uppers.append(raw.raw_upper / normalizer)
        lower = min(lowers)
        upper = min(uppers)
        _require_fraction_cap(lower, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
        _require_fraction_cap(upper, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
        return lower, upper
    except _NumericGapV2 as error:
        return _numeric_gap_interval(error.finding_code)


def _target_deficit_exact(
    target: ConstraintSafetyTargetV2,
    constraint_slack_lower: Fraction,
    constraint_slack_upper: Fraction,
) -> tuple[Fraction, Fraction]:
    target_slack = Fraction.from_float(target.target_slack)
    lower = max(Fraction(), target_slack - constraint_slack_upper)
    upper = max(Fraction(), target_slack - constraint_slack_lower)
    _require_fraction_cap(lower, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
    _require_fraction_cap(upper, "NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")
    return lower, upper


def _key_closure_findings(
    expected: set[object],
    provided: set[object],
    label: str,
) -> tuple[str, ...]:
    findings: list[str] = []
    if expected - provided:
        findings.append(f"INVALID_{label}:MISSING_KEY")
    if provided - expected:
        findings.append(f"INVALID_{label}:EXTRA_KEY")
    return tuple(findings)


def _numeric_gap_interval(*findings: str) -> ObjectiveIntervalOutcomeV2:
    return ObjectiveIntervalOutcomeV2(
        kind=ObjectiveNumericKindV2.NUMERIC_GAP,
        finding_codes=tuple(findings),
    )


def _invalid_interval(*findings: str) -> ObjectiveIntervalOutcomeV2:
    return ObjectiveIntervalOutcomeV2(
        kind=ObjectiveNumericKindV2.INVALID_INPUT,
        finding_codes=tuple(findings),
    )


def _resource_translation() -> TranslationCellOutcomeV2:
    return TranslationCellOutcomeV2(
        kind=ObjectiveNumericKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )


def _numeric_gap_translation(*findings: str) -> TranslationCellOutcomeV2:
    return TranslationCellOutcomeV2(
        kind=ObjectiveNumericKindV2.NUMERIC_GAP,
        finding_codes=tuple(findings),
    )


def _invalid_translation(*findings: str) -> TranslationCellOutcomeV2:
    return TranslationCellOutcomeV2(
        kind=ObjectiveNumericKindV2.INVALID_INPUT,
        finding_codes=tuple(findings),
    )
