"""Directed rational enclosures for continuously yawed upright boxes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Literal

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2 import so2_interval
from spatialcf.core.v2.so2_interval import (
    DirectedSinCosBoundsV2,
    RationalEnclosureV2,
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
    compile_directed_sin_cos_v2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.geometry import UprightBox3DV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


class _InvalidOrientedBoxInputV2(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OrientedAxisEnclosureV2:
    """One oriented XY unit axis enclosed componentwise."""

    x: RationalEnclosureV2
    y: RationalEnclosureV2

    def __post_init__(self) -> None:
        for field_name in ("x", "y"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            checked = _copy_enclosure(value)
            if checked.rational_lower < -1 or checked.rational_upper > 1:
                raise ValueError("axis components must lie within [-1, 1]")
            object.__setattr__(self, field_name, checked)


@dataclass(frozen=True, slots=True)
class OrientedUprightBoxBoundsV2:
    """Immutable axis and outer-AABB bounds for one upright OBB."""

    transform: DirectedYawIntervalTransformV2_2
    shape: UprightBox3DV2
    local_x_axis: OrientedAxisEnclosureV2
    local_y_axis: OrientedAxisEnclosureV2
    center_x: Fraction
    center_y: Fraction
    center_z: Fraction
    half_extent_x: Fraction
    half_extent_y: Fraction
    half_extent_z: Fraction
    x_radius: RationalEnclosureV2
    y_radius: RationalEnclosureV2
    aabb_x: RationalEnclosureV2
    aabb_y: RationalEnclosureV2
    aabb_z: RationalEnclosureV2
    atomic_steps_used: int

    def __post_init__(self) -> None:
        checked_transform, checked_shape = _strict_snapshot_inputs(
            self.transform,
            self.shape,
        )
        object.__setattr__(self, "transform", checked_transform)
        object.__setattr__(self, "shape", checked_shape)
        for field_name in ("local_x_axis", "local_y_axis"):
            value = getattr(self, field_name)
            if type(value) is not OrientedAxisEnclosureV2:
                raise TypeError(f"{field_name} must be an OrientedAxisEnclosureV2")
            object.__setattr__(
                self,
                field_name,
                OrientedAxisEnclosureV2(x=value.x, y=value.y),
            )
        for field_name in (
            "center_x",
            "center_y",
            "center_z",
            "half_extent_x",
            "half_extent_y",
            "half_extent_z",
        ):
            value = getattr(self, field_name)
            if type(value) is not Fraction:
                raise TypeError(f"{field_name} must be an exact Fraction")
            so2_interval._require_fraction_cap(value)
        if any(
            getattr(self, field_name) <= 0
            for field_name in ("half_extent_x", "half_extent_y", "half_extent_z")
        ):
            raise ValueError("oriented box half extents must be positive")
        expected_center = tuple(
            Fraction.from_float(value)
            for value in (
                self.transform.translation.x,
                self.transform.translation.y,
                self.transform.translation.z,
            )
        )
        if (self.center_x, self.center_y, self.center_z) != expected_center:
            raise ValueError("box center must equal the transform translation")
        expected_half_extents = tuple(
            Fraction.from_float(value) / 2
            for value in (
                self.shape.size_m.x,
                self.shape.size_m.y,
                self.shape.size_m.z,
            )
        )
        if (
            self.half_extent_x,
            self.half_extent_y,
            self.half_extent_z,
        ) != expected_half_extents:
            raise ValueError("box half extents must equal half of shape size")
        for field_name in ("x_radius", "y_radius", "aabb_x", "aabb_y", "aabb_z"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            object.__setattr__(self, field_name, _copy_enclosure(value))
        if self.x_radius.rational_lower < 0 or self.y_radius.rational_lower < 0:
            raise ValueError("AABB radii must be non-negative")
        if not (
            self.local_y_axis.x.rational_lower == -self.local_x_axis.y.rational_upper
            and self.local_y_axis.x.rational_upper
            == -self.local_x_axis.y.rational_lower
            and self.local_y_axis.y == self.local_x_axis.x
        ):
            raise ValueError("local axes must encode (cos,sin) and (-sin,cos)")
        absolute_cosine = _absolute_interval_no_budget(
            self.local_x_axis.x.rational_lower,
            self.local_x_axis.x.rational_upper,
        )
        absolute_sine = _absolute_interval_no_budget(
            self.local_x_axis.y.rational_lower,
            self.local_x_axis.y.rational_upper,
        )
        expected_x_radius = (
            absolute_cosine[0] * self.half_extent_x
            + absolute_sine[0] * self.half_extent_y,
            absolute_cosine[1] * self.half_extent_x
            + absolute_sine[1] * self.half_extent_y,
        )
        expected_y_radius = (
            absolute_sine[0] * self.half_extent_x
            + absolute_cosine[0] * self.half_extent_y,
            absolute_sine[1] * self.half_extent_x
            + absolute_cosine[1] * self.half_extent_y,
        )
        if (
            _enclosure_tuple(self.x_radius) != expected_x_radius
            or _enclosure_tuple(self.y_radius) != expected_y_radius
        ):
            raise ValueError("AABB radius does not match axes and half extents")
        if _enclosure_tuple(self.aabb_x) != (
            self.center_x - self.x_radius.rational_upper,
            self.center_x + self.x_radius.rational_upper,
        ) or _enclosure_tuple(self.aabb_y) != (
            self.center_y - self.y_radius.rational_upper,
            self.center_y + self.y_radius.rational_upper,
        ):
            raise ValueError("XY AABB does not match its center and radius")
        if _enclosure_tuple(self.aabb_z) != (
            self.center_z - self.half_extent_z,
            self.center_z + self.half_extent_z,
        ):
            raise ValueError("Z AABB does not match its center and half extent")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("atomic_steps_used must be a positive exact int")


@dataclass(frozen=True, slots=True)
class OrientedUprightBoxOutcomeV2:
    kind: SO2IntervalKindV2
    bounds: OrientedUprightBoxBoundsV2 | None = None
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
            if type(self.bounds) is not OrientedUprightBoxBoundsV2:
                raise ValueError("EXACT oriented box outcome requires bounds")
            object.__setattr__(self, "bounds", _copy_box_bounds(self.bounds))
            if self.finding_codes:
                raise ValueError("EXACT oriented box outcome cannot carry findings")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError(
                "non-EXACT oriented box outcome requires findings and no bounds"
            )


class OrientedBoxContactKindV2(StrEnum):
    PROVEN_SEPARATED = "PROVEN_SEPARATED"
    PROVEN_CLOSED_INTERSECTION = "PROVEN_CLOSED_INTERSECTION"
    UNKNOWN = "UNKNOWN"


AxisOwnerV2 = Literal["LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y"]


@dataclass(frozen=True, slots=True)
class OrientedAxisGapBoundsV2:
    owner: AxisOwnerV2
    gap: RationalEnclosureV2

    def __post_init__(self) -> None:
        if type(self.owner) is not str or self.owner not in (
            "LEFT_X",
            "LEFT_Y",
            "RIGHT_X",
            "RIGHT_Y",
        ):
            raise ValueError("owner must identify one canonical oriented-box axis")
        if type(self.gap) is not RationalEnclosureV2:
            raise TypeError("gap must be a RationalEnclosureV2")
        object.__setattr__(self, "gap", _copy_enclosure(self.gap))


@dataclass(frozen=True, slots=True)
class OrientedUprightBoxPairBoundsV2:
    left: OrientedUprightBoxBoundsV2
    right: OrientedUprightBoxBoundsV2
    axis_gaps: tuple[OrientedAxisGapBoundsV2, ...]
    z_gap: RationalEnclosureV2
    squared_clearance: RationalEnclosureV2
    contact_kind: OrientedBoxContactKindV2

    def __post_init__(self) -> None:
        if type(self.left) is not OrientedUprightBoxBoundsV2:
            raise TypeError("left must be OrientedUprightBoxBoundsV2")
        if type(self.right) is not OrientedUprightBoxBoundsV2:
            raise TypeError("right must be OrientedUprightBoxBoundsV2")
        object.__setattr__(self, "left", _copy_box_bounds(self.left))
        object.__setattr__(self, "right", _copy_box_bounds(self.right))
        if _box_operand_key(
            self.left.transform,
            self.left.shape,
        ) > _box_operand_key(self.right.transform, self.right.shape):
            raise ValueError("pair operands are not in canonical order")
        if type(self.axis_gaps) is not tuple or len(self.axis_gaps) != 4:
            raise ValueError("axis_gaps must contain four canonical axes")
        expected_owners = ("LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y")
        checked_gaps: list[OrientedAxisGapBoundsV2] = []
        for expected_owner, value in zip(expected_owners, self.axis_gaps, strict=True):
            if type(value) is not OrientedAxisGapBoundsV2:
                raise TypeError("axis_gaps must contain OrientedAxisGapBoundsV2")
            checked = OrientedAxisGapBoundsV2(owner=value.owner, gap=value.gap)
            if checked.owner != expected_owner:
                raise ValueError("axis_gaps are not in canonical owner order")
            checked_gaps.append(checked)
        object.__setattr__(self, "axis_gaps", tuple(checked_gaps))
        for field_name in ("z_gap", "squared_clearance"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            checked = _copy_enclosure(value)
            if checked.rational_lower < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, checked)
        if type(self.contact_kind) is not OrientedBoxContactKindV2:
            raise TypeError("contact_kind must be an OrientedBoxContactKindV2")
        any_separated = self.z_gap.rational_lower > 0 or any(
            value.gap.rational_lower > 0 for value in self.axis_gaps
        )
        proven_intersection = self.z_gap.rational_upper == 0 and all(
            value.gap.rational_upper <= 0 for value in self.axis_gaps
        )
        expected_contact = (
            OrientedBoxContactKindV2.PROVEN_SEPARATED
            if any_separated
            else (
                OrientedBoxContactKindV2.PROVEN_CLOSED_INTERSECTION
                if proven_intersection
                else OrientedBoxContactKindV2.UNKNOWN
            )
        )
        if self.contact_kind is not expected_contact:
            raise ValueError("contact classification does not match published gaps")
        left_z_min = self.left.center_z - self.left.half_extent_z
        left_z_max = self.left.center_z + self.left.half_extent_z
        right_z_min = self.right.center_z - self.right.half_extent_z
        right_z_max = self.right.center_z + self.right.half_extent_z
        expected_z_gap = max(
            Fraction(),
            left_z_min - right_z_max,
            right_z_min - left_z_max,
        )
        if _enclosure_tuple(self.z_gap) != (expected_z_gap, expected_z_gap):
            raise ValueError("Z gap does not match pair geometry")
        if proven_intersection:
            expected_clearance = (Fraction(), Fraction())
        else:
            xy_lower = max(
                Fraction(),
                *(value.gap.rational_lower for value in self.axis_gaps),
            )
            delta_x = self.right.center_x - self.left.center_x
            delta_y = self.right.center_y - self.left.center_y
            delta_z = self.right.center_z - self.left.center_z
            expected_clearance = (
                xy_lower * xy_lower + expected_z_gap * expected_z_gap,
                delta_x * delta_x + delta_y * delta_y + delta_z * delta_z,
            )
        if _enclosure_tuple(self.squared_clearance) != expected_clearance:
            raise ValueError("squared clearance does not match pair gaps and centers")


@dataclass(frozen=True, slots=True)
class OrientedUprightBoxPairOutcomeV2:
    kind: SO2IntervalKindV2
    bounds: OrientedUprightBoxPairBoundsV2 | None = None
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
            if type(self.bounds) is not OrientedUprightBoxPairBoundsV2:
                raise ValueError("EXACT oriented pair outcome requires bounds")
            object.__setattr__(self, "bounds", _copy_pair_bounds(self.bounds))
            if self.finding_codes:
                raise ValueError("EXACT oriented pair outcome cannot carry findings")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError("non-EXACT oriented pair outcome requires findings only")


def compile_oriented_upright_box_bounds_v2(
    transform: DirectedYawIntervalTransformV2_2,
    shape: UprightBox3DV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> OrientedUprightBoxOutcomeV2:
    """Compile one yawed box into directed axes and a conservative outer AABB."""

    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if (
        type(transform) is not DirectedYawIntervalTransformV2_2
        or type(shape) is not UprightBox3DV2
    ):
        return _failure(
            SO2IntervalKindV2.INVALID_INPUT,
            "INVALID_INPUT:ORIENTED_UPRIGHT_BOX_INPUT",
        )
    start_used = atomic_budget.used
    try:
        atomic_budget.consume(2)
        checked_transform, checked_shape = _strict_snapshot_inputs(transform, shape)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_checked_box_bounds_v2(
                checked_transform,
                checked_shape,
                atomic_budget,
                start_used,
            )
    except _InvalidOrientedBoxInputV2:
        return _failure(
            SO2IntervalKindV2.INVALID_INPUT,
            "INVALID_INPUT:ORIENTED_UPRIGHT_BOX_INPUT",
        )
    except SO2AtomicBudgetExhaustedV2:
        return _failure(
            SO2IntervalKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
        )
    except so2_interval._SO2NumericGapV2 as error:
        return _failure(SO2IntervalKindV2.NUMERIC_GAP, error.finding_code)
    except (OverflowError, FloatingPointError):
        return _failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:ORIENTED_UPRIGHT_BOX_ARITHMETIC",
        )
    except RuntimeWarning:
        return _failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:ORIENTED_UPRIGHT_BOX_RUNTIME_WARNING",
        )


def compile_oriented_upright_box_pair_bounds_v2(
    left_transform: DirectedYawIntervalTransformV2_2,
    left_shape: UprightBox3DV2,
    right_transform: DirectedYawIntervalTransformV2_2,
    right_shape: UprightBox3DV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> OrientedUprightBoxPairOutcomeV2:
    """Compile four-axis SAT and conservative clearance for two yawed boxes."""

    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if any(
        (
            type(left_transform) is not DirectedYawIntervalTransformV2_2,
            type(left_shape) is not UprightBox3DV2,
            type(right_transform) is not DirectedYawIntervalTransformV2_2,
            type(right_shape) is not UprightBox3DV2,
        )
    ):
        return _pair_failure(
            SO2IntervalKindV2.INVALID_INPUT,
            "INVALID_INPUT:ORIENTED_UPRIGHT_BOX_PAIR_INPUT",
        )
    try:
        atomic_budget.consume(4)
        checked_left = _strict_snapshot_inputs(left_transform, left_shape)
        checked_right = _strict_snapshot_inputs(right_transform, right_shape)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            atomic_budget.consume(2)
            left_key = _box_operand_key(*checked_left)
            right_key = _box_operand_key(*checked_right)
            if right_key < left_key:
                checked_left, checked_right = checked_right, checked_left
            return _compile_checked_pair_bounds_v2(
                *checked_left,
                *checked_right,
                atomic_budget,
            )
    except _InvalidOrientedBoxInputV2:
        return _pair_failure(
            SO2IntervalKindV2.INVALID_INPUT,
            "INVALID_INPUT:ORIENTED_UPRIGHT_BOX_PAIR_INPUT",
        )
    except SO2AtomicBudgetExhaustedV2:
        return _pair_failure(
            SO2IntervalKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
        )
    except so2_interval._SO2NumericGapV2 as error:
        return _pair_failure(SO2IntervalKindV2.NUMERIC_GAP, error.finding_code)
    except (OverflowError, FloatingPointError):
        return _pair_failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:ORIENTED_UPRIGHT_BOX_PAIR_ARITHMETIC",
        )
    except RuntimeWarning:
        return _pair_failure(
            SO2IntervalKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:ORIENTED_UPRIGHT_BOX_PAIR_RUNTIME_WARNING",
        )


def _compile_checked_pair_bounds_v2(
    left_transform: DirectedYawIntervalTransformV2_2,
    left_shape: UprightBox3DV2,
    right_transform: DirectedYawIntervalTransformV2_2,
    right_shape: UprightBox3DV2,
    budget: SO2AtomicBudgetV2,
) -> OrientedUprightBoxPairOutcomeV2:
    left_outcome = _compile_checked_box_bounds_v2(
        left_transform,
        left_shape,
        budget,
        budget.used,
    )
    if left_outcome.kind is not SO2IntervalKindV2.EXACT:
        return OrientedUprightBoxPairOutcomeV2(
            kind=left_outcome.kind,
            finding_codes=left_outcome.finding_codes,
        )
    right_outcome = _compile_checked_box_bounds_v2(
        right_transform,
        right_shape,
        budget,
        budget.used,
    )
    if right_outcome.kind is not SO2IntervalKindV2.EXACT:
        return OrientedUprightBoxPairOutcomeV2(
            kind=right_outcome.kind,
            finding_codes=right_outcome.finding_codes,
        )
    if (
        type(left_outcome.bounds) is not OrientedUprightBoxBoundsV2
        or type(right_outcome.bounds) is not OrientedUprightBoxBoundsV2
    ):
        raise RuntimeError("EXACT box outcome is missing pair operand bounds")
    left = left_outcome.bounds
    right = right_outcome.bounds
    axis_gaps = (
        _compile_axis_gap("LEFT_X", left.local_x_axis, left, right, budget),
        _compile_axis_gap("LEFT_Y", left.local_y_axis, left, right, budget),
        _compile_axis_gap("RIGHT_X", right.local_x_axis, left, right, budget),
        _compile_axis_gap("RIGHT_Y", right.local_y_axis, left, right, budget),
    )
    left_z_min = left.center_z - left.half_extent_z
    left_z_max = left.center_z + left.half_extent_z
    right_z_min = right.center_z - right.half_extent_z
    right_z_max = right.center_z + right.half_extent_z
    budget.consume()
    exact_z_gap = max(
        Fraction(),
        left_z_min - right_z_max,
        right_z_min - left_z_max,
    )
    z_gap = _publish_interval((exact_z_gap, exact_z_gap), budget)
    any_separated = exact_z_gap > 0 or any(
        value.gap.rational_lower > 0 for value in axis_gaps
    )
    proven_intersection = exact_z_gap == 0 and all(
        value.gap.rational_upper <= 0 for value in axis_gaps
    )
    if any_separated:
        contact_kind = OrientedBoxContactKindV2.PROVEN_SEPARATED
    elif proven_intersection:
        contact_kind = OrientedBoxContactKindV2.PROVEN_CLOSED_INTERSECTION
    else:
        contact_kind = OrientedBoxContactKindV2.UNKNOWN

    budget.consume(4)
    if proven_intersection:
        clearance_lower = Fraction()
        clearance_upper = Fraction()
    else:
        xy_lower = max(
            Fraction(),
            *(value.gap.rational_lower for value in axis_gaps),
        )
        delta_x = right.center_x - left.center_x
        delta_y = right.center_y - left.center_y
        delta_z = right.center_z - left.center_z
        clearance_lower = xy_lower * xy_lower + exact_z_gap * exact_z_gap
        clearance_upper = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        if clearance_lower > clearance_upper:
            raise RuntimeError("squared-clearance enclosure invariant failed")
    squared_clearance = _publish_interval(
        (clearance_lower, clearance_upper),
        budget,
    )
    budget.consume()
    return OrientedUprightBoxPairOutcomeV2(
        kind=SO2IntervalKindV2.EXACT,
        bounds=OrientedUprightBoxPairBoundsV2(
            left=left,
            right=right,
            axis_gaps=axis_gaps,
            z_gap=z_gap,
            squared_clearance=squared_clearance,
            contact_kind=contact_kind,
        ),
    )


def _compile_checked_box_bounds_v2(
    transform: DirectedYawIntervalTransformV2_2,
    shape: UprightBox3DV2,
    budget: SO2AtomicBudgetV2,
    start_used: int,
) -> OrientedUprightBoxOutcomeV2:
    trig = compile_directed_sin_cos_v2(
        transform.yaw_radians,
        atomic_budget=budget,
    )
    if trig.kind is not SO2IntervalKindV2.EXACT:
        return OrientedUprightBoxOutcomeV2(
            kind=trig.kind,
            finding_codes=trig.finding_codes,
        )
    if type(trig.bounds) is not DirectedSinCosBoundsV2:
        raise RuntimeError("EXACT sin/cos outcome is missing bounds")

    center_x = Fraction.from_float(transform.translation.x)
    center_y = Fraction.from_float(transform.translation.y)
    center_z = Fraction.from_float(transform.translation.z)
    half_extent_x = Fraction.from_float(shape.size_m.x) / 2
    half_extent_y = Fraction.from_float(shape.size_m.y) / 2
    half_extent_z = Fraction.from_float(shape.size_m.z) / 2
    for value in (
        center_x,
        center_y,
        center_z,
        half_extent_x,
        half_extent_y,
        half_extent_z,
    ):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:ORIENTED_UPRIGHT_BOX_FRACTION_BIT_CAP",
        )

    cosine = trig.bounds.cosine
    sine = trig.bounds.sine
    negative_sine = _interval_negate(sine, budget)
    absolute_cosine = _interval_absolute(cosine, budget)
    absolute_sine = _interval_absolute(sine, budget)
    x_radius_interval = _interval_add(
        _interval_scale(absolute_cosine, half_extent_x, budget),
        _interval_scale(absolute_sine, half_extent_y, budget),
        budget,
    )
    y_radius_interval = _interval_add(
        _interval_scale(absolute_sine, half_extent_x, budget),
        _interval_scale(absolute_cosine, half_extent_y, budget),
        budget,
    )
    x_radius = _publish_interval(x_radius_interval, budget)
    y_radius = _publish_interval(y_radius_interval, budget)
    local_y_x = _publish_interval(negative_sine, budget)
    aabb_x = _publish_interval(
        (center_x - x_radius_interval[1], center_x + x_radius_interval[1]),
        budget,
    )
    aabb_y = _publish_interval(
        (center_y - y_radius_interval[1], center_y + y_radius_interval[1]),
        budget,
    )
    aabb_z = _publish_interval(
        (center_z - half_extent_z, center_z + half_extent_z),
        budget,
    )
    budget.consume()
    return OrientedUprightBoxOutcomeV2(
        kind=SO2IntervalKindV2.EXACT,
        bounds=OrientedUprightBoxBoundsV2(
            transform=transform,
            shape=shape,
            local_x_axis=OrientedAxisEnclosureV2(x=cosine, y=sine),
            local_y_axis=OrientedAxisEnclosureV2(x=local_y_x, y=cosine),
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            half_extent_x=half_extent_x,
            half_extent_y=half_extent_y,
            half_extent_z=half_extent_z,
            x_radius=x_radius,
            y_radius=y_radius,
            aabb_x=aabb_x,
            aabb_y=aabb_y,
            aabb_z=aabb_z,
            atomic_steps_used=budget.used - start_used,
        ),
    )


def _box_operand_key(
    transform: DirectedYawIntervalTransformV2_2,
    shape: UprightBox3DV2,
) -> bytes:
    return canonical_json_bytes_v2(transform) + b"\0" + canonical_json_bytes_v2(shape)


def _compile_axis_gap(
    owner: AxisOwnerV2,
    axis: OrientedAxisEnclosureV2,
    left: OrientedUprightBoxBoundsV2,
    right: OrientedUprightBoxBoundsV2,
    budget: SO2AtomicBudgetV2,
) -> OrientedAxisGapBoundsV2:
    axis_x = _enclosure_tuple(axis.x)
    axis_y = _enclosure_tuple(axis.y)
    delta_x = right.center_x - left.center_x
    delta_y = right.center_y - left.center_y
    center_projection = _interval_add(
        _interval_scale_signed(axis_x, delta_x, budget),
        _interval_scale_signed(axis_y, delta_y, budget),
        budget,
    )
    absolute_center_projection = _interval_absolute_tuple(center_projection, budget)
    left_radius = _projected_radius(axis_x, axis_y, left, budget)
    right_radius = _projected_radius(axis_x, axis_y, right, budget)
    combined_radius = _interval_add(left_radius, right_radius, budget)
    gap_interval = _interval_subtract(
        absolute_center_projection,
        combined_radius,
        budget,
    )
    return OrientedAxisGapBoundsV2(
        owner=owner,
        gap=_publish_interval(gap_interval, budget),
    )


def _projected_radius(
    axis_x: tuple[Fraction, Fraction],
    axis_y: tuple[Fraction, Fraction],
    box: OrientedUprightBoxBoundsV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    local_x_dot = _interval_add(
        _interval_multiply(
            axis_x,
            _enclosure_tuple(box.local_x_axis.x),
            budget,
        ),
        _interval_multiply(
            axis_y,
            _enclosure_tuple(box.local_x_axis.y),
            budget,
        ),
        budget,
    )
    local_y_dot = _interval_add(
        _interval_multiply(
            axis_x,
            _enclosure_tuple(box.local_y_axis.x),
            budget,
        ),
        _interval_multiply(
            axis_y,
            _enclosure_tuple(box.local_y_axis.y),
            budget,
        ),
        budget,
    )
    return _interval_add(
        _interval_scale(
            _interval_absolute_tuple(local_x_dot, budget),
            box.half_extent_x,
            budget,
        ),
        _interval_scale(
            _interval_absolute_tuple(local_y_dot, budget),
            box.half_extent_y,
            budget,
        ),
        budget,
    )


def _strict_snapshot_inputs(
    transform: object,
    shape: object,
) -> tuple[DirectedYawIntervalTransformV2_2, UprightBox3DV2]:
    if (
        type(transform) is not DirectedYawIntervalTransformV2_2
        or type(shape) is not UprightBox3DV2
    ):
        raise _InvalidOrientedBoxInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_transform = DirectedYawIntervalTransformV2_2.model_validate(
                transform.model_dump(mode="python"),
                strict=True,
            )
            checked_shape = UprightBox3DV2.model_validate(
                shape.model_dump(mode="python"),
                strict=True,
            )
    except (ValidationError, PydanticSerializationError, Warning) as error:
        raise _InvalidOrientedBoxInputV2 from error
    return checked_transform, checked_shape


def _interval_negate(
    value: RationalEnclosureV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    return _checked_interval((-value.rational_upper, -value.rational_lower))


def _interval_absolute(
    value: RationalEnclosureV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    lower = value.rational_lower
    upper = value.rational_upper
    if lower <= 0 <= upper:
        return _checked_interval((Fraction(), max(-lower, upper)))
    return _checked_interval((min(abs(lower), abs(upper)), max(abs(lower), abs(upper))))


def _absolute_interval_no_budget(
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, Fraction]:
    if lower > upper:
        raise RuntimeError("interval absolute received reversed endpoints")
    if lower <= 0 <= upper:
        return _checked_interval((Fraction(), max(-lower, upper)))
    return _checked_interval((min(abs(lower), abs(upper)), max(abs(lower), abs(upper))))


def _interval_absolute_tuple(
    value: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    lower, upper = value
    if lower > upper:
        raise RuntimeError("interval absolute received reversed endpoints")
    if lower <= 0 <= upper:
        return _checked_interval((Fraction(), max(-lower, upper)))
    return _checked_interval((min(abs(lower), abs(upper)), max(abs(lower), abs(upper))))


def _interval_scale(
    value: tuple[Fraction, Fraction],
    factor: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    if factor < 0:
        raise RuntimeError("interval scale factor must be non-negative")
    return _checked_interval((value[0] * factor, value[1] * factor))


def _interval_scale_signed(
    value: tuple[Fraction, Fraction],
    factor: Fraction,
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    if factor >= 0:
        return _checked_interval((value[0] * factor, value[1] * factor))
    return _checked_interval((value[1] * factor, value[0] * factor))


def _interval_multiply(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return _checked_interval((min(products), max(products)))


def _interval_add(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    return _checked_interval((left[0] + right[0], left[1] + right[1]))


def _interval_subtract(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    return _checked_interval((left[0] - right[1], left[1] - right[0]))


def _enclosure_tuple(value: RationalEnclosureV2) -> tuple[Fraction, Fraction]:
    return value.rational_lower, value.rational_upper


def _checked_interval(
    value: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    lower, upper = value
    if lower > upper:
        raise RuntimeError("interval operation produced reversed endpoints")
    for endpoint in value:
        so2_interval._require_numeric_fraction_cap(
            endpoint,
            "NUMERIC_GAP:ORIENTED_INTERVAL_FRACTION_BIT_CAP",
        )
    return value


def _publish_interval(
    value: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> RationalEnclosureV2:
    budget.consume()
    return so2_interval._publish_rational_enclosure(*value)


def _copy_enclosure(value: RationalEnclosureV2) -> RationalEnclosureV2:
    return RationalEnclosureV2(
        rational_lower=value.rational_lower,
        rational_upper=value.rational_upper,
        lower_bound=value.lower_bound,
        upper_bound=value.upper_bound,
    )


def _copy_box_bounds(value: OrientedUprightBoxBoundsV2) -> OrientedUprightBoxBoundsV2:
    return OrientedUprightBoxBoundsV2(
        transform=value.transform,
        shape=value.shape,
        local_x_axis=value.local_x_axis,
        local_y_axis=value.local_y_axis,
        center_x=value.center_x,
        center_y=value.center_y,
        center_z=value.center_z,
        half_extent_x=value.half_extent_x,
        half_extent_y=value.half_extent_y,
        half_extent_z=value.half_extent_z,
        x_radius=value.x_radius,
        y_radius=value.y_radius,
        aabb_x=value.aabb_x,
        aabb_y=value.aabb_y,
        aabb_z=value.aabb_z,
        atomic_steps_used=value.atomic_steps_used,
    )


def _copy_pair_bounds(
    value: OrientedUprightBoxPairBoundsV2,
) -> OrientedUprightBoxPairBoundsV2:
    return OrientedUprightBoxPairBoundsV2(
        left=value.left,
        right=value.right,
        axis_gaps=value.axis_gaps,
        z_gap=value.z_gap,
        squared_clearance=value.squared_clearance,
        contact_kind=value.contact_kind,
    )


def _failure(
    kind: SO2IntervalKindV2,
    finding_code: str,
) -> OrientedUprightBoxOutcomeV2:
    return OrientedUprightBoxOutcomeV2(kind=kind, finding_codes=(finding_code,))


def _pair_failure(
    kind: SO2IntervalKindV2,
    finding_code: str,
) -> OrientedUprightBoxPairOutcomeV2:
    return OrientedUprightBoxPairOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


__all__ = (
    "OrientedAxisEnclosureV2",
    "OrientedAxisGapBoundsV2",
    "OrientedBoxContactKindV2",
    "OrientedUprightBoxBoundsV2",
    "OrientedUprightBoxOutcomeV2",
    "OrientedUprightBoxPairBoundsV2",
    "OrientedUprightBoxPairOutcomeV2",
    "compile_oriented_upright_box_bounds_v2",
    "compile_oriented_upright_box_pair_bounds_v2",
)
