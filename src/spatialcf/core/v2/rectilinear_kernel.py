"""Proof-oriented rational set algebra for finite rectilinear regions.

All arithmetic is over :class:`fractions.Fraction`, and every accepted
rectangle is explicitly in Canonical world-XY translation-delta coordinates.
The public operations return closed outcomes: resource exhaustion and
unrepresentable lower-dimensional projections never carry a partial artifact.
"""

from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.rect_kernel import (
    DirectedRectRoundingV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectKernelProjectionErrorV2,
    RectTopologyV2,
)
from spatialcf.domain.v2.base import Vec2V2
from spatialcf.domain.v2.geometry import (
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
    RingWindingV2,
)

RECTILINEAR_KERNEL_ID_V2 = "geometry-kernel:rational-rectilinear-directed-v2"
RECTILINEAR_KERNEL_VERSION_V2 = "kernel:2.0"
RECTILINEAR_KERNEL_SOUNDNESS_V2 = "DIRECTED_OUTWARD_BOUNDS"
# Exact rational set algebra plus directed binary64 boundaries account for all
# rounding.  Zero denotes no error outside the published bracket, not equality
# between a rational region and either directed projection.
RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M = 0.0


class RectilinearTopologyV2(StrEnum):
    EMPTY = "EMPTY"
    AREA = "AREA"
    DEGENERATE = "DEGENERATE"


class RectilinearOutcomeKindV2(StrEnum):
    EXACT = "EXACT"
    UNKNOWN = "UNKNOWN"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


class RectilinearNearestKindV2(StrEnum):
    """Closed outcome of exact nearest-point optimization."""

    EXACT = "EXACT"
    EMPTY = "EMPTY"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True, init=False)
class ExactRectilinearRegionV2:
    """Canonical union of closed axis-aligned rational rectangles.

    AREA rectangles have pairwise-disjoint planar interiors.  Degenerate line
    and point rectangles are retained only when they contribute set members
    outside the AREA closure.
    """

    topology: RectilinearTopologyV2
    rectangles: tuple[ExactAxisAlignedRectV2, ...]

    def __init__(self) -> None:
        raise TypeError(
            "ExactRectilinearRegionV2 values are created by capped normalization"
        )

    @property
    def area_rectangles(self) -> tuple[ExactAxisAlignedRectV2, ...]:
        return tuple(
            item for item in self.rectangles if item.topology is RectTopologyV2.AREA
        )

    @property
    def degenerate_rectangles(self) -> tuple[ExactAxisAlignedRectV2, ...]:
        return tuple(
            item
            for item in self.rectangles
            if item.topology is RectTopologyV2.DEGENERATE
        )

    def contains_point(self, x_m: Fraction, y_m: Fraction) -> bool:
        if not isinstance(x_m, Fraction) or not isinstance(y_m, Fraction):
            raise TypeError("point coordinates must be Fractions")
        return any(_rect_contains_point(item, x_m, y_m) for item in self.rectangles)


@dataclass(frozen=True, slots=True)
class RectilinearRegionOutcomeV2:
    kind: RectilinearOutcomeKindV2
    region: ExactRectilinearRegionV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RectilinearOutcomeKindV2):
            raise TypeError("kind must be a RectilinearOutcomeKindV2")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is RectilinearOutcomeKindV2.EXACT:
            if self.region is None or self.finding_codes:
                raise ValueError("EXACT region outcome requires only a region")
        elif self.region is not None or not self.finding_codes:
            raise ValueError("non-EXACT region outcome requires findings and no region")


@dataclass(frozen=True, slots=True)
class RectilinearProjectionOutcomeV2:
    kind: RectilinearOutcomeKindV2
    rounding: DirectedRectRoundingV2
    source_topology: RectilinearTopologyV2
    planar_region: PlanarRegionV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RectilinearOutcomeKindV2):
            raise TypeError("kind must be a RectilinearOutcomeKindV2")
        if not isinstance(self.rounding, DirectedRectRoundingV2):
            raise TypeError("rounding must be a DirectedRectRoundingV2")
        if not isinstance(self.source_topology, RectilinearTopologyV2):
            raise TypeError("source_topology must be a RectilinearTopologyV2")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is RectilinearOutcomeKindV2.EXACT:
            if self.finding_codes:
                raise ValueError("EXACT projection cannot carry findings")
        elif self.planar_region is not None or not self.finding_codes:
            raise ValueError(
                "non-EXACT projection requires findings and no planar region"
            )


@dataclass(frozen=True, slots=True)
class ExactRectilinearNearestPointV2:
    """Exact minimizer and a directed binary64 bracket for its L2 norm."""

    delta_x_m: Fraction
    delta_y_m: Fraction
    squared_distance_m2: Fraction
    distance_lower_bound_m: float
    distance_upper_bound_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.delta_x_m, Fraction) or not isinstance(
            self.delta_y_m, Fraction
        ):
            raise TypeError("nearest-point coordinates must be Fractions")
        if not isinstance(self.squared_distance_m2, Fraction):
            raise TypeError("squared distance must be a Fraction")
        expected = self.delta_x_m**2 + self.delta_y_m**2
        if self.squared_distance_m2 != expected:
            raise ValueError("squared distance does not match the exact point")
        lower = self.distance_lower_bound_m
        upper = self.distance_upper_bound_m
        if (
            type(lower) is not float
            or type(upper) is not float
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0.0
            or lower > upper
        ):
            raise ValueError("distance bounds must be ordered finite floats")
        lower_squared = Fraction.from_float(lower) ** 2
        upper_squared = Fraction.from_float(upper) ** 2
        if not lower_squared <= expected <= upper_squared:
            raise ValueError("distance bounds do not enclose the exact norm")
        if lower != upper and math.nextafter(lower, math.inf) != upper:
            raise ValueError("distance bounds must be adjacent binary64 values")


@dataclass(frozen=True, slots=True)
class RectilinearNearestPointOutcomeV2:
    kind: RectilinearNearestKindV2
    nearest_point: ExactRectilinearNearestPointV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RectilinearNearestKindV2):
            raise TypeError("kind must be a RectilinearNearestKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.kind is RectilinearNearestKindV2.EXACT:
            if self.nearest_point is None or self.finding_codes:
                raise ValueError("EXACT nearest outcome requires only a point")
            return
        if self.nearest_point is not None:
            raise ValueError(f"{self.kind.value} outcome cannot carry a point")
        if self.kind is RectilinearNearestKindV2.EMPTY:
            if self.finding_codes:
                raise ValueError("EMPTY nearest outcome cannot carry findings")
            return
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} nearest outcome requires a finding")


class RectilinearAtomicBudgetExhaustedV2(RuntimeError):
    """A shared rectilinear atomic-cell ledger has no remaining capacity."""


class _ResourceLimitV2(RectilinearAtomicBudgetExhaustedV2):
    pass


class _UnsupportedBoundaryV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _UnsupportedLiftV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


@dataclass(frozen=True, slots=True)
class RectilinearAtomicBudgetV2:
    """One immutable-owner, mutable-usage ledger shared across kernel calls."""

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
            raise TypeError("atomic budget limit must be an exact int")
        if self.limit <= 0:
            raise ValueError("atomic budget limit must be positive")
        if type(self.used) is not int:
            raise TypeError("atomic budget used must be an exact int")
        if self.used < 0 or self.used > self.limit:
            raise ValueError("atomic budget used must be within its limit")

    def consume(self, amount: int = 1) -> None:
        self.validate()
        if type(amount) is not int or amount < 0:
            raise TypeError("budget amount must be a non-negative exact int")
        if self.used + amount > self.limit:
            raise _ResourceLimitV2
        object.__setattr__(self, "used", self.used + amount)


def normalize_rectilinear_region_v2(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    try:
        return _exact_region_outcome(_normalize_rectangles(rectangles, budget))
    except _ResourceLimitV2:
        return _resource_region_outcome()


def lift_planar_region_v2(
    region: PlanarRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    """Lift one binary64 Canonical region into exact delta-space set algebra.

    Only finite axis-aligned polygon rings are in the certified subset.  Every
    coordinate is lifted with ``Fraction.from_float`` before grid construction,
    membership classification, or normalization.
    """

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if not isinstance(region, PlanarRegionV2):
        raise TypeError("region must be a PlanarRegionV2")
    checked = PlanarRegionV2.model_validate(
        region.model_dump(mode="python"),
        strict=True,
    )
    try:
        return _exact_region_outcome(_lift_planar_region(checked, budget))
    except _ResourceLimitV2:
        return _resource_region_outcome()
    except _UnsupportedLiftV2 as error:
        return _unknown_region_outcome(error.finding_code)


def union_rectilinear_regions_v2(
    left: ExactRectilinearRegionV2,
    right: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    try:
        _validate_region_operand(left, budget)
        _validate_region_operand(right, budget)
        return _exact_region_outcome(
            _normalize_rectangles(left.rectangles + right.rectangles, budget)
        )
    except _ResourceLimitV2:
        return _resource_region_outcome()


def intersect_rectilinear_regions_v2(
    left: ExactRectilinearRegionV2,
    right: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    try:
        _validate_region_operand(left, budget)
        _validate_region_operand(right, budget)
        budget.consume(len(left.rectangles) * len(right.rectangles))
        intersections = tuple(
            left_rect.intersect(right_rect)
            for left_rect in left.rectangles
            for right_rect in right.rectangles
        )
        return _exact_region_outcome(_normalize_rectangles(intersections, budget))
    except _ResourceLimitV2:
        return _resource_region_outcome()


def difference_rectilinear_region_v2(
    minuend: ExactRectilinearRegionV2,
    subtrahend: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    """Return the exact closed set ``minuend \\ interior(subtrahend)``."""

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    try:
        _validate_region_operand(minuend, budget)
        _validate_region_operand(subtrahend, budget)
        result = _difference_open(minuend, subtrahend, budget)
        return _exact_region_outcome(result)
    except _ResourceLimitV2:
        return _resource_region_outcome()


def clip_rectilinear_region_v2(
    region: ExactRectilinearRegionV2,
    universe: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearRegionOutcomeV2:
    return intersect_rectilinear_regions_v2(
        region,
        universe,
        max_atomic_cells=max_atomic_cells,
        atomic_budget=atomic_budget,
    )


def project_rectilinear_region_v2(
    region: ExactRectilinearRegionV2,
    rounding: DirectedRectRoundingV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearProjectionOutcomeV2:
    if not isinstance(rounding, DirectedRectRoundingV2):
        raise TypeError("rounding must be a DirectedRectRoundingV2")
    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    source_topology = _strict_source_topology(region)
    try:
        _validate_region_operand(region, budget)
        if region.topology is RectilinearTopologyV2.EMPTY:
            return RectilinearProjectionOutcomeV2(
                kind=RectilinearOutcomeKindV2.EXACT,
                rounding=rounding,
                source_topology=region.topology,
            )
        if region.degenerate_rectangles:
            return _unknown_projection(
                rounding,
                region.topology,
                "UNSUPPORTED_RECTILINEAR_PROJECTION:DEGENERATE_FEATURES",
            )

        budget.consume(len(region.area_rectangles))
        projected: list[ExactAxisAlignedRectV2] = []
        for rectangle in region.area_rectangles:
            projection = (
                rectangle.project_inner()
                if rounding is DirectedRectRoundingV2.INNER
                else rectangle.project_outer()
            )
            if projection.topology is RectTopologyV2.AREA:
                projected.append(projection.to_exact_rect())
            elif (
                rounding is DirectedRectRoundingV2.OUTER
                and projection.topology is RectTopologyV2.DEGENERATE
            ):
                return _unknown_projection(
                    rounding,
                    region.topology,
                    "UNSUPPORTED_RECTILINEAR_PROJECTION:OUTER_COLLAPSED",
                )

        normalized = _normalize_rectangles(tuple(projected), budget)
        if normalized.topology is RectilinearTopologyV2.EMPTY:
            return RectilinearProjectionOutcomeV2(
                kind=RectilinearOutcomeKindV2.EXACT,
                rounding=rounding,
                source_topology=region.topology,
            )
        try:
            planar = _polygonize_area(normalized, budget)
        except _UnsupportedBoundaryV2 as error:
            return _unknown_projection(
                rounding,
                region.topology,
                error.finding_code,
            )
        return RectilinearProjectionOutcomeV2(
            kind=RectilinearOutcomeKindV2.EXACT,
            rounding=rounding,
            source_topology=region.topology,
            planar_region=planar,
        )
    except _ResourceLimitV2:
        return _resource_projection(rounding, source_topology)
    except (OverflowError, RectKernelProjectionErrorV2):
        return _unknown_projection(
            rounding,
            source_topology,
            "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_FINITE_BINARY64",
        )


def nearest_point_to_origin_rectilinear_v2(
    region: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> RectilinearNearestPointOutcomeV2:
    """Minimize Euclidean XY translation over one exact closed region.

    Rectangle clamping and squared-distance comparisons are exact rational
    operations.  The final L2 norm is enclosed by adjacent binary64 values via
    an exact bit-pattern bisection; no floating-point value is used to select
    the minimizer.
    """

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if not isinstance(region, ExactRectilinearRegionV2):
        raise TypeError("region must be an ExactRectilinearRegionV2")
    if type(region.rectangles) is not tuple:
        raise TypeError("rectangles must be an exact tuple")
    rectangle_count = len(region.rectangles)
    try:
        # Exact directed sqrt performs a fixed 64-step bit-pattern bisection.
        # Region validation itself owns all shallow and quadratic shell charges
        # so every public operation follows the same accounting contract.
        if rectangle_count:
            budget.consume(64)
        _validate_region_operand(region, budget)
    except _ResourceLimitV2:
        return _resource_nearest_outcome()

    if region.topology is RectilinearTopologyV2.EMPTY:
        return RectilinearNearestPointOutcomeV2(kind=RectilinearNearestKindV2.EMPTY)

    try:
        budget.consume(rectangle_count)
    except _ResourceLimitV2:
        return _resource_nearest_outcome()

    candidates = tuple(_nearest_point_on_rectangle(item) for item in region.rectangles)
    delta_x, delta_y = min(
        candidates,
        key=lambda point: (point[0] ** 2 + point[1] ** 2, point[0], point[1]),
    )
    squared_distance = delta_x**2 + delta_y**2
    distance_bounds = _directed_sqrt_binary64_bounds(squared_distance)
    if distance_bounds is None:
        return RectilinearNearestPointOutcomeV2(
            kind=RectilinearNearestKindV2.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:DISTANCE_NOT_FINITE_BINARY64",),
        )
    lower, upper = distance_bounds
    return RectilinearNearestPointOutcomeV2(
        kind=RectilinearNearestKindV2.EXACT,
        nearest_point=ExactRectilinearNearestPointV2(
            delta_x_m=delta_x,
            delta_y_m=delta_y,
            squared_distance_m2=squared_distance,
            distance_lower_bound_m=lower,
            distance_upper_bound_m=upper,
        ),
    )


def _nearest_point_on_rectangle(
    rectangle: ExactAxisAlignedRectV2,
) -> tuple[Fraction, Fraction]:
    bounds = rectangle.bounds
    if bounds is None:
        raise ValueError(
            "canonical rectilinear regions cannot contain EMPTY rectangles"
        )
    min_x, min_y, max_x, max_y = bounds
    return _clamp_zero(min_x, max_x), _clamp_zero(min_y, max_y)


def _clamp_zero(lower: Fraction, upper: Fraction) -> Fraction:
    if lower > 0:
        return lower
    if upper < 0:
        return upper
    return Fraction(0)


_MAX_FINITE_BINARY64_BITS = 0x7FEF_FFFF_FFFF_FFFF


def _directed_sqrt_binary64_bounds(
    squared_distance: Fraction,
) -> tuple[float, float] | None:
    if squared_distance < 0:
        raise ValueError("squared distance must be non-negative")
    low_bits = 0
    high_bits = _MAX_FINITE_BINARY64_BITS
    best_bits = 0
    while low_bits <= high_bits:
        middle_bits = (low_bits + high_bits) // 2
        value = _positive_float_from_bits(middle_bits)
        if Fraction.from_float(value) ** 2 <= squared_distance:
            best_bits = middle_bits
            low_bits = middle_bits + 1
        else:
            high_bits = middle_bits - 1
    lower = _positive_float_from_bits(best_bits)
    lower_squared = Fraction.from_float(lower) ** 2
    if lower_squared == squared_distance:
        return lower, lower
    if best_bits == _MAX_FINITE_BINARY64_BITS:
        return None
    upper = _positive_float_from_bits(best_bits + 1)
    return lower, upper


def _positive_float_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


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


def _strict_source_topology(region: ExactRectilinearRegionV2) -> RectilinearTopologyV2:
    if not isinstance(region, ExactRectilinearRegionV2):
        raise TypeError("region must be an ExactRectilinearRegionV2")
    if not isinstance(region.topology, RectilinearTopologyV2):
        raise TypeError("region topology is invalid")
    return region.topology


def _validate_region_operand(
    region: ExactRectilinearRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> None:
    if not isinstance(region, ExactRectilinearRegionV2):
        raise TypeError("operand must be an ExactRectilinearRegionV2")
    _validate_region_shell(region, budget)
    canonical = _normalize_rectangles(region.rectangles, budget)
    if canonical != region:
        raise ValueError("rectilinear operand is not canonically normalized")


def _validate_rectangle_tuple(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    budget: RectilinearAtomicBudgetV2,
) -> None:
    if type(rectangles) is not tuple:
        raise TypeError("rectangles must be an exact tuple")
    budget.consume(len(rectangles))
    for rectangle in rectangles:
        if type(rectangle) is not ExactAxisAlignedRectV2:
            raise TypeError("rectangles must contain ExactAxisAlignedRectV2 values")
        # Frozen dataclasses can still be attacked through object.__setattr__.
        # Reconstruct each exact rectangle before reading any geometric field.
        ExactAxisAlignedRectV2(
            coordinate_space=rectangle.coordinate_space,
            topology=rectangle.topology,
            min_x_m=rectangle.min_x_m,
            min_y_m=rectangle.min_y_m,
            max_x_m=rectangle.max_x_m,
            max_y_m=rectangle.max_y_m,
        )
        if (
            rectangle.coordinate_space
            is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        ):
            raise ValueError("rectilinear input must use translation-delta coordinates")


def _normalize_rectangles(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    _validate_rectangle_tuple(rectangles, budget)
    nonempty = tuple(
        rectangle
        for rectangle in rectangles
        if rectangle.topology is not RectTopologyV2.EMPTY
    )
    area = _normalize_area_rectangles(
        tuple(
            rectangle
            for rectangle in nonempty
            if rectangle.topology is RectTopologyV2.AREA
        ),
        budget,
    )
    degenerate = _normalize_degenerate_rectangles(
        tuple(
            rectangle
            for rectangle in nonempty
            if rectangle.topology is RectTopologyV2.DEGENERATE
        ),
        area,
        budget,
    )
    canonical = tuple(sorted((*area, *degenerate), key=_rect_sort_key))
    return _make_exact_region(
        topology=_topology_for_rectangles(canonical),
        rectangles=canonical,
    )


def _normalize_area_rectangles(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    budget: RectilinearAtomicBudgetV2,
) -> tuple[ExactAxisAlignedRectV2, ...]:
    if not rectangles:
        return ()
    xs = tuple(sorted({value for item in rectangles for value in _x_bounds(item)}))
    ys = tuple(sorted({value for item in rectangles for value in _y_bounds(item)}))
    cell_count = max(0, len(xs) - 1) * max(0, len(ys) - 1)
    budget.consume(cell_count)

    row_spans: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
    for y_index in range(len(ys) - 1):
        y0, y1 = ys[y_index], ys[y_index + 1]
        occupied = [
            any(
                _rect_covers_open_cell(item, xs[x_index], y0, xs[x_index + 1], y1)
                for item in rectangles
            )
            for x_index in range(len(xs) - 1)
        ]
        start: int | None = None
        for x_index, is_occupied in enumerate((*occupied, False)):
            if is_occupied and start is None:
                start = x_index
            elif not is_occupied and start is not None:
                row_spans.append((xs[start], y0, xs[x_index], y1))
                start = None

    by_x_span: dict[tuple[Fraction, Fraction], list[tuple[Fraction, Fraction]]] = {}
    for min_x, min_y, max_x, max_y in row_spans:
        by_x_span.setdefault((min_x, max_x), []).append((min_y, max_y))

    merged: list[ExactAxisAlignedRectV2] = []
    for (min_x, max_x), y_intervals in sorted(by_x_span.items()):
        for min_y, max_y in _merge_intervals(tuple(sorted(y_intervals))):
            merged.append(_make_rect(min_x, min_y, max_x, max_y))
    return tuple(sorted(merged, key=_rect_sort_key))


def _normalize_degenerate_rectangles(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    area_rectangles: tuple[ExactAxisAlignedRectV2, ...],
    budget: RectilinearAtomicBudgetV2,
) -> tuple[ExactAxisAlignedRectV2, ...]:
    horizontal: dict[Fraction, list[tuple[Fraction, Fraction]]] = {}
    vertical: dict[Fraction, list[tuple[Fraction, Fraction]]] = {}
    points: set[tuple[Fraction, Fraction]] = set()
    for rectangle in rectangles:
        bounds = rectangle.bounds
        assert bounds is not None
        min_x, min_y, max_x, max_y = bounds
        if min_x == max_x and min_y == max_y:
            points.add((min_x, min_y))
        elif min_y == max_y:
            horizontal.setdefault(min_y, []).append((min_x, max_x))
        else:
            vertical.setdefault(min_x, []).append((min_y, max_y))

    segments: list[ExactAxisAlignedRectV2] = []
    for y, intervals in sorted(horizontal.items()):
        merged = _merge_intervals(tuple(sorted(intervals)))
        covered = _merge_intervals(
            tuple(
                sorted(
                    (bounds[0], bounds[2])
                    for rectangle in area_rectangles
                    if (bounds := rectangle.bounds) is not None
                    and bounds[1] <= y <= bounds[3]
                )
            )
        )
        for start, end in _subtract_covered_intervals(merged, covered):
            if start < end:
                segments.append(_make_rect(start, y, end, y))
            else:
                points.add((start, y))

    for x, intervals in sorted(vertical.items()):
        merged = _merge_intervals(tuple(sorted(intervals)))
        covered = _merge_intervals(
            tuple(
                sorted(
                    (bounds[1], bounds[3])
                    for rectangle in area_rectangles
                    if (bounds := rectangle.bounds) is not None
                    and bounds[0] <= x <= bounds[2]
                )
            )
        )
        for start, end in _subtract_covered_intervals(merged, covered):
            if start < end:
                segments.append(_make_rect(x, start, x, end))
            else:
                points.add((x, start))

    budget.consume(len(segments) + len(points))
    canonical_segments = tuple(sorted(set(segments), key=_rect_sort_key))
    remaining_points = tuple(
        _make_rect(x, y, x, y)
        for x, y in sorted(points)
        if not any(
            _rect_contains_point(item, x, y)
            for item in (*area_rectangles, *canonical_segments)
        )
    )
    return tuple(sorted((*canonical_segments, *remaining_points), key=_rect_sort_key))


def _difference_open(
    minuend: ExactRectilinearRegionV2,
    subtrahend: ExactRectilinearRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    if minuend.topology is RectilinearTopologyV2.EMPTY:
        return _empty_region()
    coordinates = minuend.rectangles + subtrahend.rectangles
    xs = tuple(sorted({value for item in coordinates for value in _x_bounds(item)}))
    ys = tuple(sorted({value for item in coordinates for value in _y_bounds(item)}))
    nx, ny = len(xs), len(ys)
    feature_count = (
        max(0, nx - 1) * max(0, ny - 1)
        + nx * max(0, ny - 1)
        + max(0, nx - 1) * ny
        + nx * ny
    )
    budget.consume(feature_count)

    a_cells = _area_cell_grid(minuend.area_rectangles, xs, ys)
    b_cells = _area_cell_grid(subtrahend.area_rectangles, xs, ys)
    result: list[ExactAxisAlignedRectV2] = []

    for y_index in range(ny - 1):
        for x_index in range(nx - 1):
            if a_cells[y_index][x_index] and not b_cells[y_index][x_index]:
                result.append(
                    _make_rect(
                        xs[x_index],
                        ys[y_index],
                        xs[x_index + 1],
                        ys[y_index + 1],
                    )
                )

    for x_index in range(nx):
        for y_index in range(ny - 1):
            a_member = _vertical_edge_member(
                minuend,
                a_cells,
                xs[x_index],
                ys[y_index],
                ys[y_index + 1],
                x_index,
                y_index,
            )
            b_interior = (
                0 < x_index < nx - 1
                and b_cells[y_index][x_index - 1]
                and b_cells[y_index][x_index]
            )
            if a_member and not b_interior:
                result.append(
                    _make_rect(
                        xs[x_index],
                        ys[y_index],
                        xs[x_index],
                        ys[y_index + 1],
                    )
                )

    for y_index in range(ny):
        for x_index in range(nx - 1):
            a_member = _horizontal_edge_member(
                minuend,
                a_cells,
                ys[y_index],
                xs[x_index],
                xs[x_index + 1],
                x_index,
                y_index,
            )
            b_interior = (
                0 < y_index < ny - 1
                and b_cells[y_index - 1][x_index]
                and b_cells[y_index][x_index]
            )
            if a_member and not b_interior:
                result.append(
                    _make_rect(
                        xs[x_index],
                        ys[y_index],
                        xs[x_index + 1],
                        ys[y_index],
                    )
                )

    for y_index in range(ny):
        for x_index in range(nx):
            a_member = _vertex_member(
                minuend,
                a_cells,
                xs[x_index],
                ys[y_index],
                x_index,
                y_index,
            )
            b_interior = _vertex_is_interior(b_cells, x_index, y_index)
            if a_member and not b_interior:
                result.append(
                    _make_rect(
                        xs[x_index],
                        ys[y_index],
                        xs[x_index],
                        ys[y_index],
                    )
                )
    return _normalize_rectangles(tuple(result), budget)


def _area_cell_grid(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    xs: tuple[Fraction, ...],
    ys: tuple[Fraction, ...],
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(
            any(
                _rect_covers_open_cell(
                    rectangle,
                    xs[x_index],
                    ys[y_index],
                    xs[x_index + 1],
                    ys[y_index + 1],
                )
                for rectangle in rectangles
            )
            for x_index in range(len(xs) - 1)
        )
        for y_index in range(len(ys) - 1)
    )


def _vertical_edge_member(
    region: ExactRectilinearRegionV2,
    cells: tuple[tuple[bool, ...], ...],
    x: Fraction,
    y0: Fraction,
    y1: Fraction,
    x_index: int,
    y_index: int,
) -> bool:
    adjacent = (
        (x_index > 0 and cells[y_index][x_index - 1])
        or (x_index < len(cells[0]) and cells[y_index][x_index])
        if cells and cells[0]
        else False
    )
    return adjacent or any(
        (bounds := item.bounds) is not None
        and bounds[0] == bounds[2] == x
        and bounds[1] <= y0
        and y1 <= bounds[3]
        for item in region.degenerate_rectangles
    )


def _horizontal_edge_member(
    region: ExactRectilinearRegionV2,
    cells: tuple[tuple[bool, ...], ...],
    y: Fraction,
    x0: Fraction,
    x1: Fraction,
    x_index: int,
    y_index: int,
) -> bool:
    width = len(cells[0]) if cells else 0
    adjacent = (
        (y_index > 0 and cells[y_index - 1][x_index])
        or (y_index < len(cells) and width and cells[y_index][x_index])
        if width
        else False
    )
    return adjacent or any(
        (bounds := item.bounds) is not None
        and bounds[1] == bounds[3] == y
        and bounds[0] <= x0
        and x1 <= bounds[2]
        for item in region.degenerate_rectangles
    )


def _vertex_member(
    region: ExactRectilinearRegionV2,
    cells: tuple[tuple[bool, ...], ...],
    x: Fraction,
    y: Fraction,
    x_index: int,
    y_index: int,
) -> bool:
    height = len(cells)
    width = len(cells[0]) if cells else 0
    for dy in (-1, 0):
        for dx in (-1, 0):
            cell_x, cell_y = x_index + dx, y_index + dy
            if 0 <= cell_x < width and 0 <= cell_y < height and cells[cell_y][cell_x]:
                return True
    return any(
        _rect_contains_point(item, x, y) for item in region.degenerate_rectangles
    )


def _vertex_is_interior(
    cells: tuple[tuple[bool, ...], ...],
    x_index: int,
    y_index: int,
) -> bool:
    height = len(cells)
    width = len(cells[0]) if cells else 0
    return all(
        0 <= x_index + dx < width
        and 0 <= y_index + dy < height
        and cells[y_index + dy][x_index + dx]
        for dy in (-1, 0)
        for dx in (-1, 0)
    )


Point = tuple[Fraction, Fraction]
Edge = tuple[Point, Point]


def _polygonize_area(
    region: ExactRectilinearRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> PlanarRegionV2:
    if region.topology is not RectilinearTopologyV2.AREA:
        raise _UnsupportedBoundaryV2(
            "UNSUPPORTED_RECTILINEAR_PROJECTION:DEGENERATE_FEATURES"
        )
    xs = tuple(
        sorted({value for item in region.area_rectangles for value in _x_bounds(item)})
    )
    ys = tuple(
        sorted({value for item in region.area_rectangles for value in _y_bounds(item)})
    )
    cells = _area_cell_grid(region.area_rectangles, xs, ys)
    budget.consume(max(0, len(xs) - 1) * max(0, len(ys) - 1))
    edges: set[Edge] = set()
    for y_index, row in enumerate(cells):
        for x_index, occupied in enumerate(row):
            if not occupied:
                continue
            bottom_left = (xs[x_index], ys[y_index])
            bottom_right = (xs[x_index + 1], ys[y_index])
            top_right = (xs[x_index + 1], ys[y_index + 1])
            top_left = (xs[x_index], ys[y_index + 1])
            for edge in (
                (bottom_left, bottom_right),
                (bottom_right, top_right),
                (top_right, top_left),
                (top_left, bottom_left),
            ):
                reverse = (edge[1], edge[0])
                if reverse in edges:
                    edges.remove(reverse)
                else:
                    edges.add(edge)
    budget.consume(len(edges))
    loops = _trace_boundary_loops(edges)
    positive = tuple(loop for loop in loops if _signed_area(loop) > 0)
    negative = tuple(loop for loop in loops if _signed_area(loop) < 0)
    if not positive or len(positive) + len(negative) != len(loops):
        raise _UnsupportedBoundaryV2(
            "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_MANIFOLD_BOUNDARY"
        )

    holes_by_outer: dict[int, list[tuple[Point, ...]]] = {
        index: [] for index in range(len(positive))
    }
    for hole in negative:
        containers = tuple(
            index
            for index, exterior in enumerate(positive)
            if _point_in_ring(hole[0], exterior)
        )
        if not containers:
            raise _UnsupportedBoundaryV2(
                "UNSUPPORTED_RECTILINEAR_PROJECTION:UNASSIGNED_HOLE"
            )
        owner = min(containers, key=lambda index: abs(_signed_area(positive[index])))
        holes_by_outer[owner].append(hole)

    components = tuple(
        PlanarPolygonComponentV2(
            exterior=_to_planar_ring(exterior, RingWindingV2.COUNTERCLOCKWISE),
            holes=tuple(
                _to_planar_ring(hole, RingWindingV2.CLOCKWISE)
                for hole in holes_by_outer[index]
            ),
        )
        for index, exterior in enumerate(positive)
    )
    try:
        return PlanarRegionV2(components=components)
    except ValueError as error:
        raise _UnsupportedBoundaryV2(
            "UNSUPPORTED_RECTILINEAR_PROJECTION:CANONICAL_ENCODING"
        ) from error


@dataclass(frozen=True, slots=True)
class _ExactPlanarComponentV2:
    exterior: tuple[Point, ...]
    holes: tuple[tuple[Point, ...], ...]


def _lift_planar_region(
    region: PlanarRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    components = tuple(
        _lift_planar_component(component, budget) for component in region.components
    )
    rings = tuple(
        ring
        for component in components
        for ring in (component.exterior, *component.holes)
    )
    xs = tuple(sorted({x for ring in rings for x, _ in ring}))
    ys = tuple(sorted({y for ring in rings for _, y in ring}))
    cell_count = max(0, len(xs) - 1) * max(0, len(ys) - 1)
    edge_count = sum(len(ring) for ring in rings)
    budget.consume(cell_count * max(1, edge_count))

    occupied_rows: list[tuple[bool, ...]] = []
    rectangles: list[ExactAxisAlignedRectV2] = []
    for min_y, max_y in itertools.pairwise(ys):
        row: list[bool] = []
        sample_y = (min_y + max_y) / 2
        for min_x, max_x in itertools.pairwise(xs):
            sample = ((min_x + max_x) / 2, sample_y)
            occupied = any(
                _point_in_component(sample, component) for component in components
            )
            row.append(occupied)
            if occupied:
                rectangles.append(_make_rect(min_x, min_y, max_x, max_y))
        occupied_rows.append(tuple(row))

    occupied_cells = tuple(occupied_rows)
    if not rectangles:
        raise _UnsupportedLiftV2("UNSUPPORTED_RECTILINEAR_LIFT:DEGENERATE_SEMANTICS")
    if _has_non_manifold_grid_vertex(occupied_cells):
        raise _UnsupportedLiftV2("UNSUPPORTED_RECTILINEAR_LIFT:NON_MANIFOLD_BOUNDARY")
    normalized = _normalize_rectangles(tuple(rectangles), budget)
    if normalized.topology is not RectilinearTopologyV2.AREA:
        raise _UnsupportedLiftV2("UNSUPPORTED_RECTILINEAR_LIFT:DEGENERATE_SEMANTICS")
    return normalized


def _lift_planar_component(
    component: PlanarPolygonComponentV2,
    budget: RectilinearAtomicBudgetV2,
) -> _ExactPlanarComponentV2:
    return _ExactPlanarComponentV2(
        exterior=_lift_planar_ring(component.exterior, budget),
        holes=tuple(_lift_planar_ring(hole, budget) for hole in component.holes),
    )


def _lift_planar_ring(
    ring: PlanarRingV2,
    budget: RectilinearAtomicBudgetV2,
) -> tuple[Point, ...]:
    budget.consume(len(ring.vertices))
    vertices = tuple(
        (Fraction.from_float(point.x), Fraction.from_float(point.y))
        for point in ring.vertices
    )
    for left, right in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        same_x = left[0] == right[0]
        same_y = left[1] == right[1]
        if same_x == same_y:
            finding = (
                "UNSUPPORTED_RECTILINEAR_LIFT:DEGENERATE_SEMANTICS"
                if same_x
                else "UNSUPPORTED_RECTILINEAR_LIFT:NON_AXIS_ALIGNED_EDGE"
            )
            raise _UnsupportedLiftV2(finding)
    if _signed_area(vertices) == 0:
        raise _UnsupportedLiftV2("UNSUPPORTED_RECTILINEAR_LIFT:DEGENERATE_SEMANTICS")
    return vertices


def _point_in_component(
    point: Point,
    component: _ExactPlanarComponentV2,
) -> bool:
    return _point_in_ring(point, component.exterior) and not any(
        _point_in_ring(point, hole) for hole in component.holes
    )


def _has_non_manifold_grid_vertex(
    cells: tuple[tuple[bool, ...], ...],
) -> bool:
    height = len(cells)
    width = len(cells[0]) if cells else 0
    for y_index in range(1, height):
        for x_index in range(1, width):
            lower_left = cells[y_index - 1][x_index - 1]
            lower_right = cells[y_index - 1][x_index]
            upper_left = cells[y_index][x_index - 1]
            upper_right = cells[y_index][x_index]
            if (lower_left and upper_right and not lower_right and not upper_left) or (
                lower_right and upper_left and not lower_left and not upper_right
            ):
                return True
    return False


def _trace_boundary_loops(edges: set[Edge]) -> tuple[tuple[Point, ...], ...]:
    outgoing: dict[Point, list[Point]] = {}
    incoming: dict[Point, list[Point]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
        incoming.setdefault(end, []).append(start)
    vertices = set(outgoing) | set(incoming)
    if any(
        len(outgoing.get(vertex, ())) != 1 or len(incoming.get(vertex, ())) != 1
        for vertex in vertices
    ):
        raise _UnsupportedBoundaryV2(
            "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_MANIFOLD_BOUNDARY"
        )

    unused = set(edges)
    loops: list[tuple[Point, ...]] = []
    while unused:
        start_edge = min(unused)
        start = start_edge[0]
        current = start
        vertices_in_loop: list[Point] = []
        seen: set[Point] = set()
        while True:
            if current in seen:
                raise _UnsupportedBoundaryV2(
                    "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_MANIFOLD_BOUNDARY"
                )
            seen.add(current)
            vertices_in_loop.append(current)
            next_point = outgoing[current][0]
            edge = (current, next_point)
            if edge not in unused:
                raise _UnsupportedBoundaryV2(
                    "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_MANIFOLD_BOUNDARY"
                )
            unused.remove(edge)
            current = next_point
            if current == start:
                break
        simplified = _remove_collinear_vertices(tuple(vertices_in_loop))
        if len(simplified) < 4 or _signed_area(simplified) == 0:
            raise _UnsupportedBoundaryV2(
                "UNSUPPORTED_RECTILINEAR_PROJECTION:NON_MANIFOLD_BOUNDARY"
            )
        loops.append(simplified)
    return tuple(sorted(loops, key=lambda loop: (loop[0], len(loop), loop)))


def _remove_collinear_vertices(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
    result = list(vertices)
    changed = True
    while changed and len(result) >= 4:
        changed = False
        kept: list[Point] = []
        for index, current in enumerate(result):
            previous = result[index - 1]
            following = result[(index + 1) % len(result)]
            if (previous[0] == current[0] == following[0]) or (
                previous[1] == current[1] == following[1]
            ):
                changed = True
            else:
                kept.append(current)
        result = kept
    return tuple(result)


def _to_planar_ring(
    vertices: tuple[Point, ...],
    winding: RingWindingV2,
) -> PlanarRingV2:
    return PlanarRingV2(
        winding=winding,
        vertices=tuple(Vec2V2(x=float(x), y=float(y)) for x, y in vertices),
    )


def _point_in_ring(point: Point, ring: tuple[Point, ...]) -> bool:
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
        if (y0 > y) != (y1 > y):
            intersection_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if intersection_x > x:
                inside = not inside
    return inside


def _signed_area(vertices: tuple[Point, ...]) -> Fraction:
    return (
        sum(
            (
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(
                    vertices,
                    vertices[1:] + vertices[:1],
                    strict=True,
                )
            ),
            start=Fraction(),
        )
        / 2
    )


def _merge_intervals(
    intervals: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[tuple[Fraction, Fraction], ...]:
    if not intervals:
        return ()
    merged: list[tuple[Fraction, Fraction]] = []
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return tuple(merged)


def _subtract_covered_intervals(
    intervals: tuple[tuple[Fraction, Fraction], ...],
    covered: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[tuple[Fraction, Fraction], ...]:
    result: list[tuple[Fraction, Fraction]] = []
    for start, end in intervals:
        cursor = start
        for cover_start, cover_end in covered:
            if cover_end < cursor or cover_start > end:
                continue
            if cursor < cover_start:
                result.append((cursor, min(cover_start, end)))
            cursor = max(cursor, cover_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


def _make_rect(
    min_x: Fraction,
    min_y: Fraction,
    max_x: Fraction,
    max_y: Fraction,
) -> ExactAxisAlignedRectV2:
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _rect_sort_key(
    rectangle: ExactAxisAlignedRectV2,
) -> tuple[Fraction, Fraction, Fraction, Fraction, str]:
    bounds = rectangle.bounds
    if bounds is None:
        raise ValueError("EMPTY rectangles have no canonical region sort key")
    return (*bounds, rectangle.topology.value)


def _x_bounds(rectangle: ExactAxisAlignedRectV2) -> tuple[Fraction, Fraction]:
    bounds = rectangle.bounds
    assert bounds is not None
    return bounds[0], bounds[2]


def _y_bounds(rectangle: ExactAxisAlignedRectV2) -> tuple[Fraction, Fraction]:
    bounds = rectangle.bounds
    assert bounds is not None
    return bounds[1], bounds[3]


def _rect_covers_open_cell(
    rectangle: ExactAxisAlignedRectV2,
    min_x: Fraction,
    min_y: Fraction,
    max_x: Fraction,
    max_y: Fraction,
) -> bool:
    bounds = rectangle.bounds
    assert bounds is not None
    return (
        bounds[0] <= min_x
        and max_x <= bounds[2]
        and bounds[1] <= min_y
        and max_y <= bounds[3]
    )


def _rect_contains_point(
    rectangle: ExactAxisAlignedRectV2,
    x: Fraction,
    y: Fraction,
) -> bool:
    bounds = rectangle.bounds
    assert bounds is not None
    return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]


def _area_interiors_overlap(
    left: ExactAxisAlignedRectV2,
    right: ExactAxisAlignedRectV2,
) -> bool:
    left_bounds = left.bounds
    right_bounds = right.bounds
    assert left_bounds is not None and right_bounds is not None
    return max(left_bounds[0], right_bounds[0]) < min(
        left_bounds[2], right_bounds[2]
    ) and max(left_bounds[1], right_bounds[1]) < min(left_bounds[3], right_bounds[3])


def _topology_for_rectangles(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
) -> RectilinearTopologyV2:
    if any(item.topology is RectTopologyV2.AREA for item in rectangles):
        return RectilinearTopologyV2.AREA
    if rectangles:
        return RectilinearTopologyV2.DEGENERATE
    return RectilinearTopologyV2.EMPTY


def _validate_region_shell(
    region: ExactRectilinearRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> None:
    if not isinstance(region.topology, RectilinearTopologyV2):
        raise TypeError("topology must be a RectilinearTopologyV2")
    if type(region.rectangles) is not tuple:
        raise TypeError("rectangles must be an exact tuple")
    # Charge before any scan, sort, set construction, or quadratic pair test.
    # This makes hostile-but-well-typed unsafe copies bounded as well.
    budget.consume(len(region.rectangles))
    if any(not isinstance(item, ExactAxisAlignedRectV2) for item in region.rectangles):
        raise TypeError("rectangles must contain ExactAxisAlignedRectV2 values")
    if any(
        item.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        for item in region.rectangles
    ):
        raise ValueError("rectilinear rectangles must use translation-delta space")
    if any(item.topology is RectTopologyV2.EMPTY for item in region.rectangles):
        raise ValueError("canonical rectilinear regions omit EMPTY rectangles")
    if region.rectangles != tuple(sorted(set(region.rectangles), key=_rect_sort_key)):
        raise ValueError("rectilinear rectangles must be unique and canonically sorted")
    expected = _topology_for_rectangles(region.rectangles)
    if region.topology is not expected:
        raise ValueError(f"rectangles require {expected.value} topology")
    area = region.area_rectangles
    budget.consume(len(area) * (len(area) - 1) // 2)
    for index, left in enumerate(area):
        for right in area[index + 1 :]:
            if _area_interiors_overlap(left, right):
                raise ValueError("AREA rectangle interiors must be pairwise disjoint")


def _make_exact_region(
    *,
    topology: RectilinearTopologyV2,
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
) -> ExactRectilinearRegionV2:
    # All callers construct this tuple from the normalization algorithm itself;
    # strict caller-owned operands are validated (and budgeted) before reaching
    # that algorithm.  Re-running the quadratic shell check here would validate
    # trusted output a second time and double-charge every public operation.
    value = object.__new__(ExactRectilinearRegionV2)
    object.__setattr__(value, "topology", topology)
    object.__setattr__(value, "rectangles", rectangles)
    return value


def _empty_region() -> ExactRectilinearRegionV2:
    return _make_exact_region(
        topology=RectilinearTopologyV2.EMPTY,
        rectangles=(),
    )


def _exact_region_outcome(
    region: ExactRectilinearRegionV2,
) -> RectilinearRegionOutcomeV2:
    return RectilinearRegionOutcomeV2(
        kind=RectilinearOutcomeKindV2.EXACT,
        region=region,
    )


def _resource_region_outcome() -> RectilinearRegionOutcomeV2:
    return RectilinearRegionOutcomeV2(
        kind=RectilinearOutcomeKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )


def _resource_nearest_outcome() -> RectilinearNearestPointOutcomeV2:
    return RectilinearNearestPointOutcomeV2(
        kind=RectilinearNearestKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )


def _unknown_region_outcome(finding: str) -> RectilinearRegionOutcomeV2:
    return RectilinearRegionOutcomeV2(
        kind=RectilinearOutcomeKindV2.UNKNOWN,
        finding_codes=(finding,),
    )


def _unknown_projection(
    rounding: DirectedRectRoundingV2,
    source_topology: RectilinearTopologyV2,
    finding: str,
) -> RectilinearProjectionOutcomeV2:
    return RectilinearProjectionOutcomeV2(
        kind=RectilinearOutcomeKindV2.UNKNOWN,
        rounding=rounding,
        source_topology=source_topology,
        finding_codes=(finding,),
    )


def _resource_projection(
    rounding: DirectedRectRoundingV2,
    source_topology: RectilinearTopologyV2,
) -> RectilinearProjectionOutcomeV2:
    return RectilinearProjectionOutcomeV2(
        kind=RectilinearOutcomeKindV2.RESOURCE_LIMIT,
        rounding=rounding,
        source_topology=source_topology,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )
