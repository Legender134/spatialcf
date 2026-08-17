"""Exact rational foundations for bracketed convex translation obstacles.

The values in this module are immutable proposals, not proof capabilities.
Every public semantic consumer must replay them from raw Canonical inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2 import so2_interval
from spatialcf.core.v2.oriented_upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.geometry import UprightBox3DV2

CONVEX_TRANSLATION_KERNEL_ID_V2 = (
    "geometry-kernel:rational-convex-translation-bracket-v2"
)
CONVEX_TRANSLATION_KERNEL_VERSION_V2 = "kernel:2.3-convex-translation-bracket"
CONVEX_TRANSLATION_MAX_HULL_POINTS_V2 = 64
# A 64-vertex hull can gain four universe corners.  Intersecting that outer
# polygon with the universe rectangle plus eight robust strip planes can then
# have at most 68 + 12 vertices.
CONVEX_TRANSLATION_MAX_POLYGON_VERTICES_V2 = 80

# Sorting at most 64 items takes fewer than n*ceil(log2(n)) comparisons.  The
# fixed charge keeps the implementation independent of Python's sort path.
_HULL_SORT_COMPARISON_BOUND_V2 = 64 * 6


class _InvalidConvexTranslationInputV2(ValueError):
    pass


class ConvexTranslationDomainKindV2(StrEnum):
    BRACKET = "BRACKET"
    IDENTITY = "IDENTITY"
    EMPTY = "EMPTY"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class ConvexForbiddenTopologyV2(StrEnum):
    CLOSED_BRACKET_FOR_OPEN_INTERIOR = "CLOSED_BRACKET_FOR_OPEN_INTERIOR"


@dataclass(frozen=True, slots=True)
class RationalPoint2V2:
    x: Fraction
    y: Fraction

    def __post_init__(self) -> None:
        for field_name in ("x", "y"):
            value = getattr(self, field_name)
            if type(value) is not Fraction:
                raise TypeError(f"{field_name} must be an exact Fraction")
            so2_interval._require_fraction_cap(value)


@dataclass(frozen=True, slots=True)
class RationalConvexPolygonV2:
    """A canonical strictly-convex rational polygon in CCW order."""

    vertices_ccw: tuple[RationalPoint2V2, ...]

    def __post_init__(self) -> None:
        if type(self.vertices_ccw) is not tuple:
            raise TypeError("vertices_ccw must be an exact tuple")
        if len(self.vertices_ccw) < 3:
            raise ValueError("a convex polygon requires at least three vertices")
        if len(self.vertices_ccw) > CONVEX_TRANSLATION_MAX_POLYGON_VERTICES_V2:
            raise ValueError("a convex polygon accepts at most 80 vertices")
        checked = tuple(_copy_point(point) for point in self.vertices_ccw)
        if len({(point.x, point.y) for point in checked}) != len(checked):
            raise ValueError("a convex polygon cannot repeat vertices")
        if checked[0] != min(checked, key=_point_key):
            raise ValueError("the lexicographically least vertex must be first")
        for first, second, third in _cyclic_triples(checked):
            if _cross_no_budget(first, second, third) <= 0:
                raise ValueError(
                    "polygon vertices must be strictly convex, noncollinear, and CCW"
                )
        object.__setattr__(self, "vertices_ccw", checked)


@dataclass(frozen=True, slots=True)
class ConvexObstacleBracketV2:
    inner_forbidden: RationalConvexPolygonV2 | None
    outer_forbidden: RationalConvexPolygonV2
    universe: ExactAxisAlignedRectV2
    topology: ConvexForbiddenTopologyV2
    atomic_steps_used: int

    def __post_init__(self) -> None:
        if self.inner_forbidden is not None:
            if type(self.inner_forbidden) is not RationalConvexPolygonV2:
                raise TypeError(
                    "inner_forbidden must be RationalConvexPolygonV2 or None"
                )
            checked_inner = _copy_polygon(self.inner_forbidden)
        else:
            checked_inner = None
        if type(self.outer_forbidden) is not RationalConvexPolygonV2:
            raise TypeError("outer_forbidden must be RationalConvexPolygonV2")
        checked_outer = _copy_polygon(self.outer_forbidden)
        checked_universe = _copy_universe(self.universe)
        if type(self.topology) is not ConvexForbiddenTopologyV2:
            raise TypeError("topology must be ConvexForbiddenTopologyV2")
        if (
            self.topology
            is not ConvexForbiddenTopologyV2.CLOSED_BRACKET_FOR_OPEN_INTERIOR
        ):
            raise ValueError("unsupported convex forbidden topology")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("atomic_steps_used must be a positive exact int")
        if not all(
            _universe_contains_point(checked_universe, point)
            for point in checked_outer.vertices_ccw
        ):
            raise ValueError("outer forbidden polygon must be within its universe")
        if checked_inner is not None and not all(
            _polygon_contains_point(checked_outer, point)
            for point in checked_inner.vertices_ccw
        ):
            raise ValueError("inner forbidden polygon must be a subset of outer")
        object.__setattr__(self, "inner_forbidden", checked_inner)
        object.__setattr__(self, "outer_forbidden", checked_outer)
        object.__setattr__(self, "universe", checked_universe)


@dataclass(frozen=True, slots=True)
class ConvexTranslationDomainOutcomeV2:
    kind: ConvexTranslationDomainKindV2
    bracket: ConvexObstacleBracketV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ConvexTranslationDomainKindV2:
            raise TypeError("kind must be ConvexTranslationDomainKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ConvexTranslationDomainKindV2.BRACKET:
            if type(self.bracket) is not ConvexObstacleBracketV2:
                raise ValueError("BRACKET outcome requires a convex bracket")
            if findings:
                raise ValueError("BRACKET outcome cannot carry findings")
            object.__setattr__(self, "bracket", _copy_bracket(self.bracket))
            return
        if self.bracket is not None:
            raise ValueError("non-BRACKET outcome cannot carry a bracket")
        if self.kind in (
            ConvexTranslationDomainKindV2.IDENTITY,
            ConvexTranslationDomainKindV2.EMPTY,
        ):
            if findings:
                raise ValueError(
                    "semantic empty/identity outcomes cannot carry findings"
                )
            return
        if not findings:
            raise ValueError("failure outcome requires at least one finding")


def compile_convex_translation_obstacle_v2(
    subject_transform: DirectedYawIntervalTransformV2_2,
    subject_shape: UprightBox3DV2,
    obstacle_transform: DirectedYawIntervalTransformV2_2,
    obstacle_shape: UprightBox3DV2,
    universe: ExactAxisAlignedRectV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> ConvexTranslationDomainOutcomeV2:
    """Compile a closed rational outer bracket for one XY collision obstacle."""

    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    initial_used = atomic_budget.used
    try:
        # This shallow charge happens before any Pydantic or rectangle traversal.
        atomic_budget.consume(5)
    except SO2AtomicBudgetExhaustedV2:
        return _failure(
            ConvexTranslationDomainKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_TRANSLATION_ATOMIC_STEPS",
        )

    try:
        (
            checked_subject_transform,
            checked_subject_shape,
            checked_obstacle_transform,
            checked_obstacle_shape,
            checked_universe,
        ) = _strict_compiler_inputs(
            subject_transform,
            subject_shape,
            obstacle_transform,
            obstacle_shape,
            universe,
        )
    except _InvalidConvexTranslationInputV2:
        return _failure(
            ConvexTranslationDomainKindV2.INVALID_INPUT,
            "INVALID_INPUT:CONVEX_TRANSLATION_INPUT",
        )
    except RuntimeWarning:
        return _failure(
            ConvexTranslationDomainKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_TRANSLATION_INPUT_REVALIDATION",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            z_overlaps = _z_interiors_overlap(
                checked_subject_transform,
                checked_subject_shape,
                checked_obstacle_transform,
                checked_obstacle_shape,
            )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ConvexTranslationDomainKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_TRANSLATION_Z_ARITHMETIC",
        )
    if not z_overlaps:
        return ConvexTranslationDomainOutcomeV2(
            kind=ConvexTranslationDomainKindV2.IDENTITY
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            subject_outcome = compile_oriented_upright_box_bounds_v2(
                checked_subject_transform,
                checked_subject_shape,
                atomic_budget=atomic_budget,
            )
            early = _map_oriented_failure(subject_outcome.kind)
            if early is not None:
                return early
            assert type(subject_outcome.bounds) is OrientedUprightBoxBoundsV2

            obstacle_outcome = compile_oriented_upright_box_bounds_v2(
                checked_obstacle_transform,
                checked_obstacle_shape,
                atomic_budget=atomic_budget,
            )
            early = _map_oriented_failure(obstacle_outcome.kind)
            if early is not None:
                return early
            assert type(obstacle_outcome.bounds) is OrientedUprightBoxBoundsV2

            subject_corners = _directed_world_corner_boxes(
                subject_outcome.bounds,
                atomic_budget,
            )
            obstacle_corners = _directed_world_corner_boxes(
                obstacle_outcome.bounds,
                atomic_budget,
            )
            minkowski_box_corners = _minkowski_difference_box_corners(
                obstacle_corners,
                subject_corners,
                atomic_budget,
            )
            outer = _canonical_convex_hull_v2(
                minkowski_box_corners,
                atomic_budget,
            )
            if outer is None:
                return ConvexTranslationDomainOutcomeV2(
                    kind=ConvexTranslationDomainKindV2.IDENTITY
                )
            clipped = _clip_polygon_to_universe_v2(
                outer,
                checked_universe,
                atomic_budget,
            )
            if clipped is not None:
                inner = _build_strict_inner_forbidden_v2(
                    subject_outcome.bounds,
                    obstacle_outcome.bounds,
                    checked_universe,
                    clipped,
                    atomic_budget,
                )
            else:
                inner = None
            if clipped is not None:
                atomic_budget.consume(_bracket_output_validation_charge(inner, clipped))
                _preflight_bracket_numeric(inner, clipped)
    except SO2AtomicBudgetExhaustedV2:
        return _failure(
            ConvexTranslationDomainKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_TRANSLATION_ATOMIC_STEPS",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ConvexTranslationDomainKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_TRANSLATION_ARITHMETIC",
        )

    if clipped is None:
        return ConvexTranslationDomainOutcomeV2(
            kind=ConvexTranslationDomainKindV2.IDENTITY
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            bracket = ConvexObstacleBracketV2(
                inner_forbidden=inner,
                outer_forbidden=clipped,
                universe=checked_universe,
                topology=(ConvexForbiddenTopologyV2.CLOSED_BRACKET_FOR_OPEN_INTERIOR),
                atomic_steps_used=atomic_budget.used - initial_used,
            )
            return ConvexTranslationDomainOutcomeV2(
                kind=ConvexTranslationDomainKindV2.BRACKET,
                bracket=bracket,
            )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ConvexTranslationDomainKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_TRANSLATION_FINAL_BRACKET",
        )


def _strict_compiler_inputs(
    subject_transform: object,
    subject_shape: object,
    obstacle_transform: object,
    obstacle_shape: object,
    universe: object,
) -> tuple[
    DirectedYawIntervalTransformV2_2,
    UprightBox3DV2,
    DirectedYawIntervalTransformV2_2,
    UprightBox3DV2,
    ExactAxisAlignedRectV2,
]:
    if (
        type(subject_transform) is not DirectedYawIntervalTransformV2_2
        or type(subject_shape) is not UprightBox3DV2
        or type(obstacle_transform) is not DirectedYawIntervalTransformV2_2
        or type(obstacle_shape) is not UprightBox3DV2
    ):
        raise _InvalidConvexTranslationInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_subject_transform = DirectedYawIntervalTransformV2_2.model_validate(
                subject_transform.model_dump(mode="python"),
                strict=True,
            )
            checked_subject_shape = UprightBox3DV2.model_validate(
                subject_shape.model_dump(mode="python"),
                strict=True,
            )
            checked_obstacle_transform = (
                DirectedYawIntervalTransformV2_2.model_validate(
                    obstacle_transform.model_dump(mode="python"),
                    strict=True,
                )
            )
            checked_obstacle_shape = UprightBox3DV2.model_validate(
                obstacle_shape.model_dump(mode="python"),
                strict=True,
            )
    except RuntimeWarning:
        raise
    except (ValidationError, PydanticSerializationError, Warning) as error:
        raise _InvalidConvexTranslationInputV2 from error
    checked_universe = _strict_universe_input(universe)
    return (
        checked_subject_transform,
        checked_subject_shape,
        checked_obstacle_transform,
        checked_obstacle_shape,
        checked_universe,
    )


def _strict_universe_input(universe: object) -> ExactAxisAlignedRectV2:
    if (
        type(universe) is not ExactAxisAlignedRectV2
        or universe.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        or universe.topology is not RectTopologyV2.AREA
    ):
        raise _InvalidConvexTranslationInputV2
    coordinates = (
        universe.min_x_m,
        universe.min_y_m,
        universe.max_x_m,
        universe.max_y_m,
    )
    if any(type(value) is not Fraction for value in coordinates):
        raise _InvalidConvexTranslationInputV2
    min_x, min_y, max_x, max_y = coordinates
    assert type(min_x) is Fraction
    assert type(min_y) is Fraction
    assert type(max_x) is Fraction
    assert type(max_y) is Fraction
    if min_x >= max_x or min_y >= max_y:
        raise _InvalidConvexTranslationInputV2
    try:
        for value in coordinates:
            so2_interval._require_fraction_cap(value)
    except ValueError as error:
        raise _InvalidConvexTranslationInputV2 from error
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _z_interiors_overlap(
    subject_transform: DirectedYawIntervalTransformV2_2,
    subject_shape: UprightBox3DV2,
    obstacle_transform: DirectedYawIntervalTransformV2_2,
    obstacle_shape: UprightBox3DV2,
) -> bool:
    subject_center = Fraction.from_float(subject_transform.translation.z)
    subject_half = Fraction.from_float(subject_shape.size_m.z) / 2
    obstacle_center = Fraction.from_float(obstacle_transform.translation.z)
    obstacle_half = Fraction.from_float(obstacle_shape.size_m.z) / 2
    for value in (subject_center, subject_half, obstacle_center, obstacle_half):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONVEX_TRANSLATION_Z_FRACTION_BIT_CAP",
        )
    return (
        subject_center - subject_half < obstacle_center + obstacle_half
        and obstacle_center - obstacle_half < subject_center + subject_half
    )


def _map_oriented_failure(
    kind: SO2IntervalKindV2,
) -> ConvexTranslationDomainOutcomeV2 | None:
    if kind is SO2IntervalKindV2.EXACT:
        return None
    if kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        return _failure(
            ConvexTranslationDomainKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_TRANSLATION_ATOMIC_STEPS",
        )
    if kind is SO2IntervalKindV2.NUMERIC_GAP:
        return _failure(
            ConvexTranslationDomainKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_TRANSLATION_ORIENTED_BOX",
        )
    if kind is SO2IntervalKindV2.INVALID_INPUT:
        return _failure(
            ConvexTranslationDomainKindV2.INVALID_INPUT,
            "INVALID_INPUT:CONVEX_TRANSLATION_ORIENTED_BOX",
        )
    raise RuntimeError("unhandled oriented-box outcome kind")


_Interval2V2 = tuple[Fraction, Fraction]
_CornerBoxV2 = tuple[_Interval2V2, _Interval2V2]


def _directed_world_corner_boxes(
    box: OrientedUprightBoxBoundsV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[_CornerBoxV2, ...]:
    cosine = (
        box.local_x_axis.x.rational_lower,
        box.local_x_axis.x.rational_upper,
    )
    sine = (
        box.local_x_axis.y.rational_lower,
        box.local_x_axis.y.rational_upper,
    )
    negative_sine = _interval_negate(sine, budget)
    center_x = (box.center_x, box.center_x)
    center_y = (box.center_y, box.center_y)
    corners: list[_CornerBoxV2] = []
    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            local_x = sign_x * box.half_extent_x
            local_y = sign_y * box.half_extent_y
            world_x = _interval_add(
                _interval_add(
                    center_x,
                    _interval_scale_signed(cosine, local_x, budget),
                    budget,
                ),
                _interval_scale_signed(negative_sine, local_y, budget),
                budget,
            )
            world_y = _interval_add(
                _interval_add(
                    center_y,
                    _interval_scale_signed(sine, local_x, budget),
                    budget,
                ),
                _interval_scale_signed(cosine, local_y, budget),
                budget,
            )
            corners.append((world_x, world_y))
    return tuple(corners)


def _minkowski_difference_box_corners(
    obstacle_corners: tuple[_CornerBoxV2, ...],
    subject_corners: tuple[_CornerBoxV2, ...],
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...]:
    if len(obstacle_corners) != 4 or len(subject_corners) != 4:
        raise RuntimeError("upright boxes must each publish four corner boxes")
    points: list[RationalPoint2V2] = []
    for obstacle in obstacle_corners:
        for subject in subject_corners:
            x_interval = _interval_subtract(obstacle[0], subject[0], budget)
            y_interval = _interval_subtract(obstacle[1], subject[1], budget)
            budget.consume(4)
            points.extend(
                RationalPoint2V2(x=x, y=y)
                for x, y in (
                    (x_interval[0], y_interval[0]),
                    (x_interval[1], y_interval[0]),
                    (x_interval[1], y_interval[1]),
                    (x_interval[0], y_interval[1]),
                )
            )
    if len(points) != CONVEX_TRANSLATION_MAX_HULL_POINTS_V2:
        raise RuntimeError("Minkowski corner construction must publish 64 points")
    return tuple(points)


def _interval_negate(
    value: _Interval2V2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    return _checked_interval((-value[1], -value[0]))


def _interval_scale_signed(
    value: _Interval2V2,
    factor: Fraction,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    if factor >= 0:
        return _checked_interval((value[0] * factor, value[1] * factor))
    return _checked_interval((value[1] * factor, value[0] * factor))


def _interval_add(
    left: _Interval2V2,
    right: _Interval2V2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    return _checked_interval((left[0] + right[0], left[1] + right[1]))


def _interval_subtract(
    left: _Interval2V2,
    right: _Interval2V2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    return _checked_interval((left[0] - right[1], left[1] - right[0]))


def _checked_interval(value: _Interval2V2) -> _Interval2V2:
    if value[0] > value[1]:
        raise RuntimeError("interval operation produced reversed endpoints")
    for endpoint in value:
        so2_interval._require_numeric_fraction_cap(
            endpoint,
            "NUMERIC_GAP:CONVEX_TRANSLATION_FRACTION_BIT_CAP",
        )
    return value


def _interval_multiply(
    left: _Interval2V2,
    right: _Interval2V2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return _checked_interval((min(products), max(products)))


def _interval_absolute(
    value: _Interval2V2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    lower, upper = value
    if lower <= 0 <= upper:
        return _checked_interval((Fraction(), max(-lower, upper)))
    return _checked_interval((min(abs(lower), abs(upper)), max(abs(lower), abs(upper))))


def _interval_scale_nonnegative(
    value: _Interval2V2,
    factor: Fraction,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    budget.consume()
    if factor < 0:
        raise RuntimeError("projected-radius scale must be non-negative")
    return _checked_interval((value[0] * factor, value[1] * factor))


@dataclass(frozen=True, slots=True)
class _RobustStripV2:
    normal_x: Fraction
    normal_y: Fraction
    center_projection: Fraction
    direction_error: Fraction
    radius_lower: Fraction
    half_width: Fraction


def _build_strict_inner_forbidden_v2(
    subject: OrientedUprightBoxBoundsV2,
    obstacle: OrientedUprightBoxBoundsV2,
    universe: ExactAxisAlignedRectV2,
    outer: RationalConvexPolygonV2,
    budget: SO2AtomicBudgetV2,
) -> RationalConvexPolygonV2 | None:
    universe_bounds = universe.bounds
    assert universe_bounds is not None
    delta_center_x = obstacle.center_x - subject.center_x
    delta_center_y = obstacle.center_y - subject.center_y
    for value in (delta_center_x, delta_center_y):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONVEX_TRANSLATION_INNER_FRACTION_BIT_CAP",
        )
    vertices: tuple[RationalPoint2V2, ...] | None = (
        RationalPoint2V2(x=universe_bounds[0], y=universe_bounds[1]),
        RationalPoint2V2(x=universe_bounds[2], y=universe_bounds[1]),
        RationalPoint2V2(x=universe_bounds[2], y=universe_bounds[3]),
        RationalPoint2V2(x=universe_bounds[0], y=universe_bounds[3]),
    )
    strips: list[_RobustStripV2] = []
    axes = (
        subject.local_x_axis,
        subject.local_y_axis,
        obstacle.local_x_axis,
        obstacle.local_y_axis,
    )
    for axis in axes:
        axis_x = (axis.x.rational_lower, axis.x.rational_upper)
        axis_y = (axis.y.rational_lower, axis.y.rational_upper)
        budget.consume(8)
        normal_x = (axis_x[0] + axis_x[1]) / 2
        normal_y = (axis_y[0] + axis_y[1]) / 2
        error_x = (axis_x[1] - axis_x[0]) / 2
        error_y = (axis_y[1] - axis_y[0]) / 2
        x_extent = max(
            abs(universe_bounds[0] - delta_center_x),
            abs(universe_bounds[2] - delta_center_x),
        )
        y_extent = max(
            abs(universe_bounds[1] - delta_center_y),
            abs(universe_bounds[3] - delta_center_y),
        )
        direction_error = x_extent * error_x + y_extent * error_y
        subject_radius = _projected_radius_interval(axis_x, axis_y, subject, budget)
        obstacle_radius = _projected_radius_interval(
            axis_x,
            axis_y,
            obstacle,
            budget,
        )
        radius_lower = subject_radius[0] + obstacle_radius[0]
        remaining_width = radius_lower - direction_error
        for value in (
            normal_x,
            normal_y,
            error_x,
            error_y,
            x_extent,
            y_extent,
            direction_error,
            radius_lower,
            remaining_width,
        ):
            so2_interval._require_numeric_fraction_cap(
                value,
                "NUMERIC_GAP:CONVEX_TRANSLATION_INNER_FRACTION_BIT_CAP",
            )
        if remaining_width <= 0:
            return None
        half_width = remaining_width / 2
        center_projection = delta_center_x * normal_x + delta_center_y * normal_y
        for value in (half_width, center_projection):
            so2_interval._require_numeric_fraction_cap(
                value,
                "NUMERIC_GAP:CONVEX_TRANSLATION_INNER_FRACTION_BIT_CAP",
            )
        strip = _RobustStripV2(
            normal_x=normal_x,
            normal_y=normal_y,
            center_projection=center_projection,
            direction_error=direction_error,
            radius_lower=radius_lower,
            half_width=half_width,
        )
        strips.append(strip)
        for sign in (Fraction(1), Fraction(-1)):
            if vertices is None:
                return None
            vertices = _clip_vertices_by_half_plane(
                vertices,
                sign * normal_x,
                sign * normal_y,
                sign * center_projection + half_width,
                budget,
            )

    if vertices is None:
        return None
    vertices = _clip_vertices_to_convex_polygon(vertices, outer, budget)
    if vertices is None:
        return None
    inner = RationalConvexPolygonV2(vertices_ccw=vertices)
    _verify_strict_inner_vertices(
        inner,
        outer,
        tuple(strips),
        delta_center_x,
        delta_center_y,
        budget,
    )
    return inner


def _projected_radius_interval(
    axis_x: _Interval2V2,
    axis_y: _Interval2V2,
    box: OrientedUprightBoxBoundsV2,
    budget: SO2AtomicBudgetV2,
) -> _Interval2V2:
    local_x_dot = _interval_add(
        _interval_multiply(
            axis_x,
            (
                box.local_x_axis.x.rational_lower,
                box.local_x_axis.x.rational_upper,
            ),
            budget,
        ),
        _interval_multiply(
            axis_y,
            (
                box.local_x_axis.y.rational_lower,
                box.local_x_axis.y.rational_upper,
            ),
            budget,
        ),
        budget,
    )
    local_y_dot = _interval_add(
        _interval_multiply(
            axis_x,
            (
                box.local_y_axis.x.rational_lower,
                box.local_y_axis.x.rational_upper,
            ),
            budget,
        ),
        _interval_multiply(
            axis_y,
            (
                box.local_y_axis.y.rational_lower,
                box.local_y_axis.y.rational_upper,
            ),
            budget,
        ),
        budget,
    )
    return _interval_add(
        _interval_scale_nonnegative(
            _interval_absolute(local_x_dot, budget),
            box.half_extent_x,
            budget,
        ),
        _interval_scale_nonnegative(
            _interval_absolute(local_y_dot, budget),
            box.half_extent_y,
            budget,
        ),
        budget,
    )


def _clip_vertices_to_convex_polygon(
    vertices: tuple[RationalPoint2V2, ...],
    container: RationalConvexPolygonV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    clipped: tuple[RationalPoint2V2, ...] | None = vertices
    outer_vertices = container.vertices_ccw
    for index, first in enumerate(outer_vertices):
        if clipped is None:
            return None
        budget.consume()
        second = outer_vertices[(index + 1) % len(outer_vertices)]
        edge_x = second.x - first.x
        edge_y = second.y - first.y
        normal_x = edge_y
        normal_y = -edge_x
        offset = edge_y * first.x - edge_x * first.y
        for value in (edge_x, edge_y, normal_x, normal_y, offset):
            so2_interval._require_numeric_fraction_cap(
                value,
                "NUMERIC_GAP:CONVEX_TRANSLATION_INNER_FRACTION_BIT_CAP",
            )
        clipped = _clip_vertices_by_half_plane(
            clipped,
            normal_x,
            normal_y,
            offset,
            budget,
        )
    return clipped


def _verify_strict_inner_vertices(
    inner: RationalConvexPolygonV2,
    outer: RationalConvexPolygonV2,
    strips: tuple[_RobustStripV2, ...],
    delta_center_x: Fraction,
    delta_center_y: Fraction,
    budget: SO2AtomicBudgetV2,
) -> None:
    for point in inner.vertices_ccw:
        budget.consume(len(strips) + len(outer.vertices_ccw) + 1)
        if not _polygon_contains_point_numeric(outer, point):
            raise RuntimeError("certified inner polygon escaped its outer bracket")
        for strip in strips:
            projection = abs(
                (point.x - delta_center_x) * strip.normal_x
                + (point.y - delta_center_y) * strip.normal_y
            )
            so2_interval._require_numeric_fraction_cap(
                projection,
                "NUMERIC_GAP:CONVEX_TRANSLATION_INNER_FRACTION_BIT_CAP",
            )
            if not projection + strip.direction_error < strip.radius_lower:
                raise RuntimeError("certified inner vertex lost its strict SAT margin")


def _bracket_output_validation_charge(
    inner: RationalConvexPolygonV2 | None,
    outer: RationalConvexPolygonV2,
) -> int:
    """Precharge both direct bracket validation and outcome defensive copy."""

    outer_count = len(outer.vertices_ccw)
    inner_count = 0 if inner is None else len(inner.vertices_ccw)
    one_pass = 8 + 4 * outer_count + inner_count * (outer_count + 4)
    return 2 * one_pass


def _preflight_bracket_numeric(
    inner: RationalConvexPolygonV2 | None,
    outer: RationalConvexPolygonV2,
) -> None:
    _preflight_polygon_numeric(outer.vertices_ccw)
    if inner is None:
        return
    _preflight_polygon_numeric(inner.vertices_ccw)
    if not all(
        _polygon_contains_point_numeric(outer, point) for point in inner.vertices_ccw
    ):
        raise RuntimeError("certified inner polygon escaped its outer bracket")


def _preflight_polygon_numeric(
    vertices: tuple[RationalPoint2V2, ...],
) -> None:
    for first, second, third in _cyclic_triples(vertices):
        if _cross_numeric(first, second, third) <= 0:
            raise RuntimeError("numeric polygon preflight lost strict convexity")


def _clip_polygon_to_universe_v2(
    polygon: RationalConvexPolygonV2,
    universe: ExactAxisAlignedRectV2,
    budget: SO2AtomicBudgetV2,
) -> RationalConvexPolygonV2 | None:
    bounds = universe.bounds
    assert bounds is not None
    vertices: tuple[RationalPoint2V2, ...] | None = polygon.vertices_ccw
    for normal_x, normal_y, offset in (
        (Fraction(-1), Fraction(), -bounds[0]),
        (Fraction(1), Fraction(), bounds[2]),
        (Fraction(), Fraction(-1), -bounds[1]),
        (Fraction(), Fraction(1), bounds[3]),
    ):
        if vertices is None:
            return None
        vertices = _clip_vertices_by_half_plane(
            vertices,
            normal_x,
            normal_y,
            offset,
            budget,
        )
    if vertices is None:
        return None
    return RationalConvexPolygonV2(vertices_ccw=vertices)


def _clip_vertices_by_half_plane(
    vertices: tuple[RationalPoint2V2, ...],
    normal_x: Fraction,
    normal_y: Fraction,
    offset: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    if len(vertices) < 3:
        raise RuntimeError("half-plane clip requires an area polygon")
    output: list[RationalPoint2V2] = []
    previous = vertices[-1]
    previous_value = _half_plane_value(previous, normal_x, normal_y, offset, budget)
    previous_inside = previous_value <= 0
    for current in vertices:
        current_value = _half_plane_value(
            current,
            normal_x,
            normal_y,
            offset,
            budget,
        )
        current_inside = current_value <= 0
        if current_inside:
            if not previous_inside:
                output.append(
                    _segment_half_plane_intersection(
                        previous,
                        current,
                        previous_value,
                        current_value,
                        budget,
                    )
                )
            output.append(current)
        elif previous_inside:
            output.append(
                _segment_half_plane_intersection(
                    previous,
                    current,
                    previous_value,
                    current_value,
                    budget,
                )
            )
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _canonicalize_clipped_vertices(tuple(output), budget)


def _half_plane_value(
    point: RationalPoint2V2,
    normal_x: Fraction,
    normal_y: Fraction,
    offset: Fraction,
    budget: SO2AtomicBudgetV2,
) -> Fraction:
    budget.consume()
    value = point.x * normal_x + point.y * normal_y - offset
    so2_interval._require_numeric_fraction_cap(
        value,
        "NUMERIC_GAP:CONVEX_TRANSLATION_CLIP_FRACTION_BIT_CAP",
    )
    return value


def _segment_half_plane_intersection(
    start: RationalPoint2V2,
    end: RationalPoint2V2,
    start_value: Fraction,
    end_value: Fraction,
    budget: SO2AtomicBudgetV2,
) -> RationalPoint2V2:
    budget.consume()
    denominator = start_value - end_value
    if denominator == 0:
        raise RuntimeError("crossing edge cannot be parallel to its clipping plane")
    parameter = start_value / denominator
    if not 0 <= parameter <= 1:
        raise RuntimeError("clipping intersection parameter escaped its segment")
    x = start.x + parameter * (end.x - start.x)
    y = start.y + parameter * (end.y - start.y)
    for value in (parameter, x, y):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONVEX_TRANSLATION_CLIP_FRACTION_BIT_CAP",
        )
    return RationalPoint2V2(x=x, y=y)


def _canonicalize_clipped_vertices(
    vertices: tuple[RationalPoint2V2, ...],
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    deduplicated: list[RationalPoint2V2] = []
    for point in vertices:
        budget.consume()
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    changed = True
    while changed and len(deduplicated) >= 3:
        changed = False
        kept: list[RationalPoint2V2] = []
        count = len(deduplicated)
        for index, point in enumerate(deduplicated):
            cross = _charged_cross(
                deduplicated[index - 1],
                point,
                deduplicated[(index + 1) % count],
                budget,
            )
            if cross < 0:
                raise RuntimeError("half-plane clipping broke convex winding")
            if cross == 0:
                changed = True
            else:
                kept.append(point)
        deduplicated = kept
    if len(deduplicated) < 3:
        return None
    least = min(
        range(len(deduplicated)), key=lambda index: _point_key(deduplicated[index])
    )
    canonical = tuple(deduplicated[least:] + deduplicated[:least])
    # The strict constructor independently rechecks winding/canonicality.
    _preflight_polygon_numeric(canonical)
    return RationalConvexPolygonV2(vertices_ccw=canonical).vertices_ccw


def _failure(
    kind: ConvexTranslationDomainKindV2,
    finding_code: str,
) -> ConvexTranslationDomainOutcomeV2:
    return ConvexTranslationDomainOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _canonical_convex_hull_v2(
    points: tuple[RationalPoint2V2, ...],
    budget: SO2AtomicBudgetV2,
) -> RationalConvexPolygonV2 | None:
    """Return the canonical strict hull, or ``None`` for lower dimension."""

    if type(points) is not tuple:
        raise TypeError("points must be an exact tuple")
    if len(points) > CONVEX_TRANSLATION_MAX_HULL_POINTS_V2:
        raise ValueError("convex hull accepts at most 64 points")
    if type(budget) is not SO2AtomicBudgetV2:
        raise TypeError("budget must be an SO2AtomicBudgetV2")
    budget.validate()
    budget.consume(len(points) + _HULL_SORT_COMPARISON_BOUND_V2)
    checked = tuple(_copy_point(point) for point in points)
    unique = sorted(set(checked), key=_point_key)
    if len(unique) < 3:
        return None

    lower: list[RationalPoint2V2] = []
    for point in unique:
        while (
            len(lower) >= 2 and _charged_cross(lower[-2], lower[-1], point, budget) <= 0
        ):
            lower.pop()
        lower.append(point)

    upper: list[RationalPoint2V2] = []
    for point in reversed(unique):
        while (
            len(upper) >= 2 and _charged_cross(upper[-2], upper[-1], point, budget) <= 0
        ):
            upper.pop()
        upper.append(point)

    hull = tuple(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return None
    least_index = min(range(len(hull)), key=lambda index: _point_key(hull[index]))
    canonical = hull[least_index:] + hull[:least_index]
    _preflight_polygon_numeric(canonical)
    return RationalConvexPolygonV2(vertices_ccw=canonical)


def _charged_cross(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
    budget: SO2AtomicBudgetV2,
) -> Fraction:
    budget.consume()
    return _cross_numeric(origin, left, right)


def _cross_no_budget(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
) -> Fraction:
    value = _raw_cross(origin, left, right)
    so2_interval._require_fraction_cap(value)
    return value


def _cross_numeric(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
) -> Fraction:
    value = _raw_cross(origin, left, right)
    so2_interval._require_numeric_fraction_cap(
        value,
        "NUMERIC_GAP:CONVEX_TRANSLATION_CROSS_FRACTION_BIT_CAP",
    )
    return value


def _raw_cross(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
) -> Fraction:
    return (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (
        right.x - origin.x
    )


def _cyclic_triples(
    vertices: tuple[RationalPoint2V2, ...],
) -> tuple[tuple[RationalPoint2V2, RationalPoint2V2, RationalPoint2V2], ...]:
    count = len(vertices)
    return tuple(
        (vertices[index], vertices[(index + 1) % count], vertices[(index + 2) % count])
        for index in range(count)
    )


def _point_key(point: RationalPoint2V2) -> tuple[Fraction, Fraction]:
    return point.x, point.y


def _copy_point(point: object) -> RationalPoint2V2:
    if type(point) is not RationalPoint2V2:
        raise TypeError("polygon vertices must be RationalPoint2V2")
    return RationalPoint2V2(x=point.x, y=point.y)


def _copy_polygon(polygon: RationalConvexPolygonV2) -> RationalConvexPolygonV2:
    return RationalConvexPolygonV2(vertices_ccw=polygon.vertices_ccw)


def _copy_universe(universe: object) -> ExactAxisAlignedRectV2:
    if type(universe) is not ExactAxisAlignedRectV2:
        raise TypeError("universe must be an ExactAxisAlignedRectV2")
    if (
        universe.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        or universe.topology is not RectTopologyV2.AREA
        or universe.bounds is None
    ):
        raise ValueError("universe must be an AREA translation-delta rectangle")
    min_x, min_y, max_x, max_y = universe.bounds
    for value in (min_x, min_y, max_x, max_y):
        so2_interval._require_fraction_cap(value)
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _universe_contains_point(
    universe: ExactAxisAlignedRectV2,
    point: RationalPoint2V2,
) -> bool:
    bounds = universe.bounds
    assert bounds is not None
    return bounds[0] <= point.x <= bounds[2] and bounds[1] <= point.y <= bounds[3]


def _polygon_contains_point(
    polygon: RationalConvexPolygonV2,
    point: RationalPoint2V2,
) -> bool:
    vertices = polygon.vertices_ccw
    return all(
        _cross_no_budget(vertices[index], vertices[(index + 1) % len(vertices)], point)
        >= 0
        for index in range(len(vertices))
    )


def _polygon_contains_point_numeric(
    polygon: RationalConvexPolygonV2,
    point: RationalPoint2V2,
) -> bool:
    vertices = polygon.vertices_ccw
    return all(
        _cross_numeric(vertices[index], vertices[(index + 1) % len(vertices)], point)
        >= 0
        for index in range(len(vertices))
    )


def _copy_bracket(bracket: ConvexObstacleBracketV2) -> ConvexObstacleBracketV2:
    return ConvexObstacleBracketV2(
        inner_forbidden=bracket.inner_forbidden,
        outer_forbidden=bracket.outer_forbidden,
        universe=bracket.universe,
        topology=bracket.topology,
        atomic_steps_used=bracket.atomic_steps_used,
    )


__all__ = (
    "CONVEX_TRANSLATION_KERNEL_ID_V2",
    "CONVEX_TRANSLATION_KERNEL_VERSION_V2",
    "CONVEX_TRANSLATION_MAX_HULL_POINTS_V2",
    "CONVEX_TRANSLATION_MAX_POLYGON_VERTICES_V2",
    "ConvexForbiddenTopologyV2",
    "ConvexObstacleBracketV2",
    "ConvexTranslationDomainKindV2",
    "ConvexTranslationDomainOutcomeV2",
    "RationalConvexPolygonV2",
    "RationalPoint2V2",
    "compile_convex_translation_obstacle_v2",
)
