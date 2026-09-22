"""Directed rational enclosures for continuously yawed upright boxes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Literal

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.so2 import (
    CardinalKernelKindV3,
    DirectedSinCosBoundsV2,
    RationalEnclosureV2,
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
    compile_directed_sin_cos_v2,
)
from spatialcf.domain.geometry import DirectedYawIntervalTransformV2_2, UprightBox3DV2
from spatialcf.domain.serialization import canonical_json_bytes


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
    return canonical_json_bytes(transform) + b"\0" + canonical_json_bytes(shape)


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


@dataclass(frozen=True, slots=True)
class ClosedXYCellV3:
    x_lower: Fraction
    x_upper: Fraction
    y_lower: Fraction
    y_upper: Fraction

    def __post_init__(self) -> None:
        if (
            any(
                type(v) is not Fraction
                for v in (self.x_lower, self.x_upper, self.y_lower, self.y_upper)
            )
            or self.x_lower > self.x_upper
            or self.y_lower > self.y_upper
        ):
            raise ValueError("closed XY cell requires ordered exact Fraction bounds")

    @property
    def canonical_bounds(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.x_lower, self.x_upper, self.y_lower, self.y_upper)


@dataclass(frozen=True, slots=True)
class FixedCardinalBoxV3:
    box_id: str
    center_x: Fraction
    center_y: Fraction
    center_z: Fraction
    half_x: Fraction
    half_y: Fraction
    half_z: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.box_id) is not str
            or not self.box_id
            or any(
                type(v) is not Fraction
                for v in (
                    self.center_x,
                    self.center_y,
                    self.center_z,
                    self.half_x,
                    self.half_y,
                    self.half_z,
                )
            )
            or min(self.half_x, self.half_y, self.half_z) < 0
        ):
            raise ValueError("fixed cardinal box must be exact and non-negative")


@dataclass(frozen=True, slots=True)
class SupportSurfaceV3:
    x_lower: Fraction
    x_upper: Fraction
    y_lower: Fraction
    y_upper: Fraction
    z: Fraction

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not Fraction
                for value in (
                    self.x_lower,
                    self.x_upper,
                    self.y_lower,
                    self.y_upper,
                    self.z,
                )
            )
            or self.x_lower > self.x_upper
            or self.y_lower > self.y_upper
        ):
            raise ValueError("support surface must use ordered exact Fraction bounds")


_FIXED_CARDINAL_OBJECTIVE_TERM_ROSTER_V3: dict[str, tuple[str, str, str, str, str]] = {
    "T": (
        "definition:spatialcf/upright-se2/objective-selector-subject-pivot-displacement/1.0",
        "definition:spatialcf/upright-se2/objective-subject-pivot-displacement",
        "1.0",
        "definition:spatialcf/upright-se2/metre/1.0",
        "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
    ),
    "A": (
        "definition:spatialcf/upright-se2/objective-selector-angular-geodesic/1.0",
        "definition:spatialcf/upright-se2/objective-angular-geodesic",
        "1.0",
        "definition:spatialcf/upright-se2/turn/1.0",
        "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
    ),
    "R": (
        "definition:spatialcf/upright-se2/objective-selector-nontarget-relation-damage/1.0",
        "definition:spatialcf/upright-se2/objective-nontarget-relation-damage",
        "1.0",
        "definition:spatialcf/upright-se2/dimensionless/1.0",
        "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
    ),
    "V": (
        "definition:spatialcf/upright-se2/objective-selector-visibility-change/1.0",
        "definition:spatialcf/upright-se2/objective-visibility-change",
        "1.0",
        "definition:spatialcf/upright-se2/dimensionless/1.0",
        "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
    ),
    "S": (
        "definition:spatialcf/upright-se2/objective-selector-safety-margin-penalty/1.0",
        "definition:spatialcf/upright-se2/objective-safety-margin-penalty",
        "1.0",
        "definition:spatialcf/upright-se2/dimensionless/1.0",
        "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
    ),
}


@dataclass(frozen=True, slots=True)
class FixedCardinalObjectiveTermV3:
    """One caller-owned normalized term of the common exact cell objective."""

    term_id: str
    selector: str
    metric_definition_id: str
    metric_definition_version: str
    unit: str
    weight: Fraction
    normalizer: Fraction
    aggregation: str

    def __post_init__(self) -> None:
        expected = (
            _FIXED_CARDINAL_OBJECTIVE_TERM_ROSTER_V3.get(self.term_id)
            if type(self.term_id) is str
            else None
        )
        if (
            type(self.term_id) is not str
            or expected is None
            or any(
                type(value) is not str or not value
                for value in (
                    self.selector,
                    self.metric_definition_id,
                    self.metric_definition_version,
                    self.unit,
                    self.aggregation,
                )
            )
            or type(self.weight) is not Fraction
            or self.weight < 0
            or (self.term_id in ("T", "A") and self.weight <= 0)
            or type(self.normalizer) is not Fraction
            or self.normalizer <= 0
            or (
                self.selector,
                self.metric_definition_id,
                self.metric_definition_version,
                self.unit,
                self.aggregation,
            )
            != expected
        ):
            raise ValueError(
                "objective term must match the closed request-bound roster"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalCellPolicyV3:
    """Complete domain-neutral caller policy for one exact cardinal cell."""

    policy_id: str
    policy_version: str
    relation_threshold: Fraction
    relation_tolerance: Fraction
    relation_comparator: str
    relation_boundary: str
    safety_penalty_scale: Fraction
    safety_constraint_slack_target: Fraction
    safety_rule: str
    collision_clearance: Fraction
    collision_contact_comparator: str
    collision_boundary: str
    support_accepted_contact_gap: tuple[Fraction, Fraction]
    support_stability_margin: Fraction
    support_containment_comparator: str
    support_boundary: str
    support_frame: str
    support_normal: tuple[Fraction, Fraction, Fraction]
    relation_definition_id: str
    relation_definition_version: str
    relation_symbol: str
    relation_measurement: str
    relation_operand: str
    relation_geometry: str
    visibility_cell: tuple[Fraction, Fraction, Fraction, Fraction]
    visibility_bounds: tuple[Fraction, Fraction]
    objective_terms: tuple[FixedCardinalObjectiveTermV3, ...]
    atomic_step_limit: int

    def __post_init__(self) -> None:
        if (
            type(self.policy_id) is not str
            or not self.policy_id
            or type(self.policy_version) is not str
            or not self.policy_version
            or type(self.relation_threshold) is not Fraction
            or self.relation_threshold < 0
            or type(self.relation_tolerance) is not Fraction
            or self.relation_tolerance < 0
            or self.relation_comparator not in ("GE", "LE")
            or self.relation_boundary != "CLOSED"
            or type(self.safety_penalty_scale) is not Fraction
            or self.safety_penalty_scale < 0
            or type(self.safety_constraint_slack_target) is not Fraction
            or self.safety_rule != "PENALIZE_BELOW_TARGET"
            or type(self.collision_clearance) is not Fraction
            or self.collision_clearance < 0
            or self.collision_contact_comparator != "GE"
            or self.collision_boundary != "CLOSED"
            or type(self.support_accepted_contact_gap) is not tuple
            or len(self.support_accepted_contact_gap) != 2
            or any(
                type(value) is not Fraction
                for value in self.support_accepted_contact_gap
            )
            or self.support_accepted_contact_gap[0]
            > self.support_accepted_contact_gap[1]
            or type(self.support_stability_margin) is not Fraction
            or self.support_stability_margin < 0
            or self.support_containment_comparator != "GE"
            or self.support_boundary != "CLOSED"
            or self.support_frame != "WORLD_XY_Z_UP"
            or self.support_normal != (Fraction(), Fraction(), Fraction(1))
            or self.relation_definition_id != "spatial-relation:fixed-cardinal"
            or self.relation_definition_version != "definition:1"
            or self.relation_symbol
            not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR")
            or self.relation_measurement
            not in (
                "EXTENT_AWARE_SIGNED_AXIS_GAP",
                "EXTENT_AWARE_EUCLIDEAN_SEPARATION",
            )
            or self.relation_operand != "SUBJECT_COMPOUND_TO_REFERENCE"
            or self.relation_geometry != "UPRIGHT_AABB_EXTENTS"
            or type(self.visibility_cell) is not tuple
            or len(self.visibility_cell) != 4
            or any(type(value) is not Fraction for value in self.visibility_cell)
            or self.visibility_cell[0] > self.visibility_cell[1]
            or self.visibility_cell[2] > self.visibility_cell[3]
            or type(self.visibility_bounds) is not tuple
            or len(self.visibility_bounds) != 2
            or any(type(value) is not Fraction for value in self.visibility_bounds)
            or not Fraction()
            <= self.visibility_bounds[0]
            <= self.visibility_bounds[1]
            <= Fraction(1)
            or type(self.objective_terms) is not tuple
            or len(self.objective_terms) != 5
            or any(
                type(term) is not FixedCardinalObjectiveTermV3
                for term in self.objective_terms
            )
            or tuple(term.term_id for term in self.objective_terms)
            != ("T", "A", "R", "V", "S")
            or type(self.atomic_step_limit) is not int
            or self.atomic_step_limit <= 0
        ):
            raise ValueError("fixed cardinal cell policy must be a closed caller value")


@dataclass(frozen=True, slots=True)
class FixedCardinalCellBoundsV3:
    cell_id: str
    cell: ClosedXYCellV3
    quarter_turns_ccw: int
    collision_constraint_present: bool
    collision_inner_slack: Fraction
    collision_outer_slack: Fraction
    support_contact_gap: tuple[Fraction, Fraction]
    support_clearance_inner: Fraction
    support_clearance_outer: Fraction
    support_normal: tuple[Fraction, Fraction, Fraction]
    support_frame: str
    closed_containment: bool
    stability_margin_inner: Fraction
    relation_measurement: str
    relation_inner_slack: Fraction
    relation_outer_slack: Fraction
    relation_comparator: str
    relation_threshold: Fraction
    relation_tolerance: Fraction
    relation_boundary: str
    relation_squared_distance_bounds: tuple[Fraction, Fraction] | None
    relation_inner_success: bool
    relation_outer_failure: bool
    common_cell_semantic_objective_terms: tuple[tuple[str, Fraction, Fraction], ...]
    common_cell_objective_terms: tuple[tuple[str, Fraction, Fraction], ...]
    inner_hard_constraint_slack: Fraction
    outer_hard_constraint_slack: Fraction
    safety_penalty_outer: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.cell_id) is not str
            or not self.cell_id
            or type(self.cell) is not ClosedXYCellV3
            or type(self.quarter_turns_ccw) is not int
            or self.quarter_turns_ccw not in (0, 1, 2, 3)
            or type(self.collision_constraint_present) is not bool
            or any(
                type(value) is not Fraction
                for value in (
                    self.collision_inner_slack,
                    self.collision_outer_slack,
                    self.support_clearance_inner,
                    self.support_clearance_outer,
                    self.stability_margin_inner,
                    self.relation_inner_slack,
                    self.relation_outer_slack,
                    self.relation_threshold,
                    self.relation_tolerance,
                    self.inner_hard_constraint_slack,
                    self.outer_hard_constraint_slack,
                    self.safety_penalty_outer,
                )
            )
            or self.inner_hard_constraint_slack > self.outer_hard_constraint_slack
            or self.relation_tolerance < 0
            or self.safety_penalty_outer < 0
            or self.support_frame != "WORLD_XY_Z_UP"
            or self.support_normal != (Fraction(), Fraction(), Fraction(1))
            or self.relation_boundary != "CLOSED"
            or type(self.relation_inner_success) is not bool
            or type(self.relation_outer_failure) is not bool
        ):
            raise ValueError("fixed cardinal cell bounds are not a closed exact record")
        if (
            type(self.support_contact_gap) is not tuple
            or len(self.support_contact_gap) != 2
            or any(type(value) is not Fraction for value in self.support_contact_gap)
            or self.support_contact_gap[0] > self.support_contact_gap[1]
        ):
            raise ValueError("support contact gap must be an ordered exact interval")
        if self.relation_squared_distance_bounds is not None and (
            type(self.relation_squared_distance_bounds) is not tuple
            or len(self.relation_squared_distance_bounds) != 2
            or any(
                type(value) is not Fraction
                for value in self.relation_squared_distance_bounds
            )
            or self.relation_squared_distance_bounds[0]
            > self.relation_squared_distance_bounds[1]
        ):
            raise ValueError(
                "squared distance bounds must be an ordered exact interval"
            )
        for objective_terms in (
            self.common_cell_semantic_objective_terms,
            self.common_cell_objective_terms,
        ):
            if tuple(term[0] for term in objective_terms) != (
                "T",
                "A",
                "R",
                "V",
                "S",
            ):
                raise ValueError(
                    "common-cell objective bounds must contain ordered T/A/R/V/S"
                )
            if any(
                type(value) is not Fraction
                for _, lower, upper in objective_terms
                for value in (lower, upper)
            ) or any(lower > upper for _, lower, upper in objective_terms):
                raise ValueError(
                    "objective term bounds must be ordered exact Fractions"
                )

    @property
    def collision_slack_lower(self) -> Fraction:
        """Backward-compatible lower collision slack view for the additive seam."""

        return self.collision_inner_slack

    @property
    def support_slack_lower(self) -> Fraction:
        """Backward-compatible lower support slack view for the additive seam."""

        return self.support_clearance_inner

    @property
    def relation_slack_lower(self) -> Fraction:
        """Backward-compatible lower relation slack view for the additive seam."""

        return self.relation_inner_slack

    @property
    def safety_penalty_upper(self) -> Fraction:
        """Backward-compatible upper safety penalty view for the additive seam."""

        return self.safety_penalty_outer


@dataclass(frozen=True, slots=True)
class FixedCardinalCellOutcomeV3:
    kind: CardinalKernelKindV3
    bounds: FixedCardinalCellBoundsV3 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CardinalKernelKindV3:
            raise TypeError("fixed cardinal outcome must use CardinalKernelKindV3")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError("fixed cardinal outcome usage must be non-negative")
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"fixed cardinal {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is CardinalKernelKindV3.EXACT:
            if (
                type(self.bounds) is not FixedCardinalCellBoundsV3
                or self.finding_codes
                or not self.proof_rows
            ):
                raise ValueError(
                    "exact fixed cardinal outcome must carry only bounds and proof"
                )
            return
        if self.bounds is not None or not self.finding_codes or not self.proof_rows:
            raise ValueError(
                "non-exact fixed cardinal outcome must carry finding and proof"
            )
        prefix = {
            CardinalKernelKindV3.NUMERIC_GAP: "NUMERIC_GAP:",
            CardinalKernelKindV3.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            CardinalKernelKindV3.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(prefix) for code in self.finding_codes):
            raise ValueError("fixed cardinal finding must match outcome kind")


_CardinalExtentV3 = tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]


def _require_fixed_cardinal_box_v3(value: object, *, label: str) -> FixedCardinalBoxV3:
    if type(value) is not FixedCardinalBoxV3:
        raise TypeError(f"{label} must be a FixedCardinalBoxV3")
    return value


def _require_fixed_cardinal_roster_v3(
    value: object, *, label: str, nonempty: bool
) -> tuple[FixedCardinalBoxV3, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ValueError(
            f"{label} must be an exact {'non-empty ' if nonempty else ''}tuple"
        )
    boxes = tuple(_require_fixed_cardinal_box_v3(box, label=label) for box in value)
    ids = tuple(box.box_id for box in boxes)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ValueError(f"{label} must be sorted and have unique box IDs")
    return boxes


def _fixed_cardinal_extent_v3(
    box: FixedCardinalBoxV3,
    *,
    cell: ClosedXYCellV3,
    q: int,
    pivot_xy: tuple[Fraction, Fraction],
    atomic_budget: SO2AtomicBudgetV2,
) -> _CardinalExtentV3:
    relative_x = box.center_x - pivot_xy[0]
    relative_y = box.center_y - pivot_xy[1]
    rotated = so2_interval.rotate_cardinal_fraction_xy_v3(
        relative_x,
        relative_y,
        q,
        atomic_budget=atomic_budget,
    )
    if (
        rotated.kind is not CardinalKernelKindV3.EXACT
        or type(rotated.value) is not tuple
        or any(type(value) is not Fraction for value in rotated.value)
    ):
        raise RuntimeError(
            "prevalidated cardinal SO2 delegation did not return an exact XY pair"
        )
    rotated_x, rotated_y = rotated.value
    center_x = pivot_xy[0] + rotated_x
    center_y = pivot_xy[1] + rotated_y
    half_x, half_y = (
        (box.half_x, box.half_y) if q % 2 == 0 else (box.half_y, box.half_x)
    )
    return (
        center_x + cell.x_lower - half_x,
        center_x + cell.x_upper + half_x,
        center_y + cell.y_lower - half_y,
        center_y + cell.y_upper + half_y,
        box.center_z - box.half_z,
        box.center_z + box.half_z,
    )


def _compound_extent_v3(extents: tuple[_CardinalExtentV3, ...]) -> _CardinalExtentV3:
    return (
        min(extent[0] for extent in extents),
        max(extent[1] for extent in extents),
        min(extent[2] for extent in extents),
        max(extent[3] for extent in extents),
        min(extent[4] for extent in extents),
        max(extent[5] for extent in extents),
    )


def _pair_clearance_bounds_v3(
    left: _CardinalExtentV3, right: _CardinalExtentV3
) -> tuple[Fraction, Fraction]:
    lower_gaps = (
        right[0] - left[1],
        left[0] - right[1],
        right[2] - left[3],
        left[2] - right[3],
        right[4] - left[5],
        left[4] - right[5],
    )
    upper_gaps = (
        right[1] - left[0],
        left[1] - right[0],
        right[3] - left[2],
        left[3] - right[2],
        right[5] - left[4],
        left[5] - right[4],
    )
    return max(lower_gaps), max(upper_gaps)


def _absolute_fraction_v3(value: Fraction) -> Fraction:
    return value if value >= 0 else -value


def _axis_separation_v3(
    left_lower: Fraction,
    left_upper: Fraction,
    right_lower: Fraction,
    right_upper: Fraction,
) -> Fraction:
    return max(Fraction(), right_lower - left_upper, left_lower - right_upper)


def _squared_extent_separation_v3(
    left: _CardinalExtentV3, right: _CardinalExtentV3
) -> Fraction:
    x = _axis_separation_v3(left[0], left[1], right[0], right[1])
    y = _axis_separation_v3(left[2], left[3], right[2], right[3])
    z = _axis_separation_v3(left[4], left[5], right[4], right[5])
    return x * x + y * y + z * z


def _translate_extent_xy_v3(
    extent: _CardinalExtentV3, x: Fraction, y: Fraction
) -> _CardinalExtentV3:
    return (
        extent[0] + x,
        extent[1] + x,
        extent[2] + y,
        extent[3] + y,
        extent[4],
        extent[5],
    )


def _squared_distance_bounds_over_cell_v3(
    *,
    subject_base_extents: tuple[_CardinalExtentV3, ...],
    subject_cell_extents: tuple[_CardinalExtentV3, ...],
    reference_extent: _CardinalExtentV3,
    cell: ClosedXYCellV3,
) -> tuple[Fraction, Fraction]:
    """Sound exact bounds for compound-box separation on a closed translation cell.

    The lower bound expands each subject extent by the entire cell.  The upper
    bound evaluates all closed-cell corners for every body; taking the minimum
    over bodies remains a sound compound-union upper bound.  Both are squared
    distances, so no irrational square root or sampling is needed.
    """

    lower = min(
        _squared_extent_separation_v3(extent, reference_extent)
        for extent in subject_cell_extents
    )
    corners = (
        (cell.x_lower, cell.y_lower),
        (cell.x_lower, cell.y_upper),
        (cell.x_upper, cell.y_lower),
        (cell.x_upper, cell.y_upper),
    )
    upper = min(
        max(
            _squared_extent_separation_v3(
                _translate_extent_xy_v3(extent, x, y), reference_extent
            )
            for x, y in corners
        )
        for extent in subject_base_extents
    )
    if lower > upper:
        raise RuntimeError("squared distance bounds must be ordered")
    return lower, upper


def _fixed_cardinal_cell_id_v3(cell: ClosedXYCellV3, q: int) -> str:
    values = ",".join(
        f"{value.numerator}/{value.denominator}" for value in cell.canonical_bounds
    )
    return f"cardinal-cell:q={q}:xy={values}"


def evaluate_fixed_cardinal_cell_v3(
    *,
    cell: ClosedXYCellV3,
    quarter_turns_ccw: int,
    subject_boxes: tuple[FixedCardinalBoxV3, ...],
    obstacle_boxes: tuple[FixedCardinalBoxV3, ...],
    support_surface: SupportSurfaceV3,
    relation: str,
    reference_box: FixedCardinalBoxV3,
    near_far_threshold: Fraction,
    policy: FixedCardinalCellPolicyV3,
    atomic_budget: SO2AtomicBudgetV2,
    subject_pivot_xy: tuple[Fraction, Fraction] = (Fraction(), Fraction()),
    objective_subject_pivot_xy: tuple[Fraction, Fraction] | None = None,
) -> FixedCardinalCellOutcomeV3:
    """Bound compound collision/support/relation/safety over one closed cardinal XY cell."""
    if type(cell) is not ClosedXYCellV3:
        raise TypeError("cell must be a ClosedXYCellV3")
    if type(quarter_turns_ccw) is not int:
        raise TypeError("quarter_turns_ccw must be an exact int")
    if quarter_turns_ccw not in (0, 1, 2, 3):
        raise ValueError("quarter_turns_ccw must be in 0..3")
    subjects = _require_fixed_cardinal_roster_v3(
        subject_boxes,
        label="subject_boxes",
        nonempty=True,
    )
    obstacles = _require_fixed_cardinal_roster_v3(
        obstacle_boxes,
        label="obstacle_boxes",
        nonempty=False,
    )
    if type(support_surface) is not SupportSurfaceV3:
        raise TypeError("support_surface must be a SupportSurfaceV3")
    if relation not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR"):
        raise ValueError("relation must be a registered fixed-cardinal relation")
    reference = _require_fixed_cardinal_box_v3(reference_box, label="reference_box")
    if type(near_far_threshold) is not Fraction or near_far_threshold < 0:
        raise ValueError("near_far_threshold must be a non-negative exact Fraction")
    if type(policy) is not FixedCardinalCellPolicyV3:
        raise TypeError("policy must be a FixedCardinalCellPolicyV3")
    if near_far_threshold != policy.relation_threshold:
        raise ValueError("near_far_threshold must match the caller policy")
    if policy.relation_symbol != relation:
        raise ValueError("relation must match the caller-bound registered symbol")
    expected_measurement = (
        "EXTENT_AWARE_EUCLIDEAN_SEPARATION"
        if relation in ("NEAR", "FAR")
        else "EXTENT_AWARE_SIGNED_AXIS_GAP"
    )
    if policy.relation_measurement != expected_measurement:
        raise ValueError("relation measurement must match its registered symbol")
    if policy.visibility_cell != cell.canonical_bounds:
        raise ValueError("visibility bounds must be supplied for this exact cell")
    if (
        type(subject_pivot_xy) is not tuple
        or len(subject_pivot_xy) != 2
        or any(type(value) is not Fraction for value in subject_pivot_xy)
    ):
        raise TypeError("subject_pivot_xy must contain two exact Fractions")
    if objective_subject_pivot_xy is None:
        objective_subject_pivot_xy = subject_pivot_xy
    if (
        type(objective_subject_pivot_xy) is not tuple
        or len(objective_subject_pivot_xy) != 2
        or any(type(value) is not Fraction for value in objective_subject_pivot_xy)
    ):
        raise TypeError("objective_subject_pivot_xy must contain two exact Fractions")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if atomic_budget.limit != policy.atomic_step_limit:
        raise ValueError("atomic budget limit must match the caller policy")
    delegated_rotation_steps = 2 * len(subjects) + len(obstacles) + 2
    required_steps = max(
        4 + len(subjects) + len(obstacles),
        delegated_rotation_steps,
    )
    if atomic_budget.remaining < required_steps:
        return FixedCardinalCellOutcomeV3(
            CardinalKernelKindV3.RESOURCE_LIMIT,
            atomic_steps_used=0,
            proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:cap-minus-one",),
            finding_codes=("RESOURCE_LIMIT:SO2_ATOMIC_STEPS",),
        )

    zero_cell = ClosedXYCellV3(Fraction(), Fraction(), Fraction(), Fraction())
    subject_base_extents = tuple(
        _fixed_cardinal_extent_v3(
            box,
            cell=zero_cell,
            q=quarter_turns_ccw,
            pivot_xy=subject_pivot_xy,
            atomic_budget=atomic_budget,
        )
        for box in subjects
    )
    subject_extents = tuple(
        _fixed_cardinal_extent_v3(
            box,
            cell=cell,
            q=quarter_turns_ccw,
            pivot_xy=subject_pivot_xy,
            atomic_budget=atomic_budget,
        )
        for box in subjects
    )
    obstacle_extents = tuple(
        _fixed_cardinal_extent_v3(
            box,
            cell=zero_cell,
            q=0,
            pivot_xy=(Fraction(), Fraction()),
            atomic_budget=atomic_budget,
        )
        for box in obstacles
    )
    collision_pairs = tuple(
        _pair_clearance_bounds_v3(subject_extent, obstacle_extent)
        for subject_extent in subject_extents
        for obstacle_extent in obstacle_extents
    )
    collision_inner, collision_outer = (
        (
            min(pair[0] for pair in collision_pairs) - policy.collision_clearance,
            min(pair[1] for pair in collision_pairs) - policy.collision_clearance,
        )
        if collision_pairs
        else (Fraction(), Fraction())
    )
    if cell.x_lower == cell.x_upper and cell.y_lower == cell.y_upper:
        collision_outer = collision_inner
    base_compound = _compound_extent_v3(subject_base_extents)
    containment_inner = min(
        base_compound[0] + cell.x_lower - support_surface.x_lower,
        support_surface.x_upper - (base_compound[1] + cell.x_upper),
        base_compound[2] + cell.y_lower - support_surface.y_lower,
        support_surface.y_upper - (base_compound[3] + cell.y_upper),
    )
    containment_outer = min(
        base_compound[0] + cell.x_upper - support_surface.x_lower,
        support_surface.x_upper - (base_compound[1] + cell.x_lower),
        base_compound[2] + cell.y_upper - support_surface.y_lower,
        support_surface.y_upper - (base_compound[3] + cell.y_lower),
    )
    contact_gaps = tuple(extent[4] - support_surface.z for extent in subject_extents)
    contact_gap = (min(contact_gaps), max(contact_gaps))
    accepted_lower, accepted_upper = policy.support_accepted_contact_gap
    contact_inner = min(
        min(contact_gaps) - accepted_lower,
        accepted_upper - max(contact_gaps),
    )
    contact_outer = min(
        max(contact_gaps) - accepted_lower,
        accepted_upper - min(contact_gaps),
    )
    support_inner = min(
        containment_inner - policy.support_stability_margin, contact_inner
    )
    support_outer = min(
        containment_outer - policy.support_stability_margin, contact_outer
    )
    reference_extent = _fixed_cardinal_extent_v3(
        reference,
        cell=zero_cell,
        q=0,
        pivot_xy=(Fraction(), Fraction()),
        atomic_budget=atomic_budget,
    )
    if relation == "LEFT":
        raw_lower = base_compound[1] + cell.x_lower - reference_extent[0]
        raw_upper = base_compound[1] + cell.x_upper - reference_extent[0]
        comparator = "LE"
        measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
        squared_distance_bounds = None
    elif relation == "RIGHT":
        raw_lower = base_compound[0] + cell.x_lower - reference_extent[1]
        raw_upper = base_compound[0] + cell.x_upper - reference_extent[1]
        comparator = "GE"
        measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
        squared_distance_bounds = None
    elif relation == "FRONT":
        raw_lower = base_compound[3] + cell.y_lower - reference_extent[2]
        raw_upper = base_compound[3] + cell.y_upper - reference_extent[2]
        comparator = "LE"
        measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
        squared_distance_bounds = None
    elif relation == "BEHIND":
        raw_lower = base_compound[2] + cell.y_lower - reference_extent[3]
        raw_upper = base_compound[2] + cell.y_upper - reference_extent[3]
        comparator = "GE"
        measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
        squared_distance_bounds = None
    else:
        squared_distance_bounds = _squared_distance_bounds_over_cell_v3(
            subject_base_extents=subject_base_extents,
            subject_cell_extents=subject_extents,
            reference_extent=reference_extent,
            cell=cell,
        )
        squared_lower, squared_upper = squared_distance_bounds
        if relation == "NEAR":
            raw_lower, raw_upper = squared_lower, squared_upper
            threshold = policy.relation_threshold + policy.relation_tolerance
            comparator = "LE"
        else:
            raw_lower, raw_upper = squared_lower, squared_upper
            threshold = max(
                Fraction(), policy.relation_threshold - policy.relation_tolerance
            )
            comparator = "GE"
        threshold_squared = threshold * threshold
        measurement = "EXTENT_AWARE_EUCLIDEAN_SEPARATION"
    if comparator != policy.relation_comparator:
        raise ValueError("relation comparator must match the registered relation")
    if comparator == "LE":
        relation_inner = (
            policy.relation_threshold + policy.relation_tolerance - raw_upper
        )
        relation_outer = (
            policy.relation_threshold + policy.relation_tolerance - raw_lower
        )
        if relation in ("NEAR",):
            relation_inner = threshold_squared - raw_upper
            relation_outer = threshold_squared - raw_lower
    else:
        relation_inner = raw_lower - (
            policy.relation_threshold - policy.relation_tolerance
        )
        relation_outer = raw_upper - (
            policy.relation_threshold - policy.relation_tolerance
        )
        if relation == "FAR":
            relation_inner = raw_lower - threshold_squared
            relation_outer = raw_upper - threshold_squared
    relation_inner_success = relation_inner >= 0
    relation_outer_failure = relation_outer < 0
    hard_inner = min(
        *(
            (collision_inner, support_inner, relation_inner)
            if collision_pairs
            else (support_inner, relation_inner)
        )
    )
    hard_outer = min(
        *(
            (collision_outer, support_outer, relation_outer)
            if collision_pairs
            else (support_outer, relation_outer)
        )
    )
    safety = policy.safety_penalty_scale * max(
        Fraction(), policy.safety_constraint_slack_target - hard_inner
    )
    subject_relative_x = objective_subject_pivot_xy[0] - subject_pivot_xy[0]
    subject_relative_y = objective_subject_pivot_xy[1] - subject_pivot_xy[1]
    rotated_subject_pivot = so2_interval.rotate_cardinal_fraction_xy_v3(
        subject_relative_x,
        subject_relative_y,
        quarter_turns_ccw,
        atomic_budget=atomic_budget,
    )
    if (
        rotated_subject_pivot.kind is not CardinalKernelKindV3.EXACT
        or type(rotated_subject_pivot.value) is not tuple
        or any(type(value) is not Fraction for value in rotated_subject_pivot.value)
    ):
        raise RuntimeError(
            "prevalidated cardinal SO2 delegation did not rotate the subject pivot"
        )
    rotated_pivot_x, rotated_pivot_y = rotated_subject_pivot.value
    orbital_x = subject_pivot_xy[0] + rotated_pivot_x - objective_subject_pivot_xy[0]
    orbital_y = subject_pivot_xy[1] + rotated_pivot_y - objective_subject_pivot_xy[1]
    displacement_x = (orbital_x + cell.x_lower, orbital_x + cell.x_upper)
    displacement_y = (orbital_y + cell.y_lower, orbital_y + cell.y_upper)
    translation_lower = max(
        _axis_separation_v3(
            displacement_x[0], displacement_x[1], Fraction(), Fraction()
        ),
        _axis_separation_v3(
            displacement_y[0], displacement_y[1], Fraction(), Fraction()
        ),
    )
    translation_upper = max(
        _absolute_fraction_v3(displacement_x[0]),
        _absolute_fraction_v3(displacement_x[1]),
    ) + max(
        _absolute_fraction_v3(displacement_y[0]),
        _absolute_fraction_v3(displacement_y[1]),
    )
    remaining_steps = required_steps - delegated_rotation_steps
    if remaining_steps:
        atomic_budget.consume(remaining_steps)
    semantic_objective_terms = (
        ("T", translation_lower, translation_upper),
        (
            "A",
            Fraction(min(quarter_turns_ccw, 4 - quarter_turns_ccw), 4),
            Fraction(min(quarter_turns_ccw, 4 - quarter_turns_ccw), 4),
        ),
        ("R", max(Fraction(), -relation_outer), max(Fraction(), -relation_inner)),
        ("V", policy.visibility_bounds[0], policy.visibility_bounds[1]),
        ("S", max(Fraction(), -hard_outer), safety),
    )
    objective_terms = tuple(
        (
            term_id,
            lower * record.weight / record.normalizer,
            upper * record.weight / record.normalizer,
        )
        for (term_id, lower, upper), record in zip(
            semantic_objective_terms, policy.objective_terms, strict=True
        )
    )
    return FixedCardinalCellOutcomeV3(
        CardinalKernelKindV3.EXACT,
        FixedCardinalCellBoundsV3(
            cell_id=_fixed_cardinal_cell_id_v3(cell, quarter_turns_ccw),
            cell=cell,
            quarter_turns_ccw=quarter_turns_ccw,
            collision_constraint_present=bool(collision_pairs),
            collision_inner_slack=collision_inner,
            collision_outer_slack=collision_outer,
            support_contact_gap=contact_gap,
            support_clearance_inner=support_inner,
            support_clearance_outer=support_outer,
            support_normal=(Fraction(), Fraction(), Fraction(1)),
            support_frame="WORLD_XY_Z_UP",
            closed_containment=True,
            stability_margin_inner=(
                containment_inner - policy.support_stability_margin
            ),
            relation_measurement=measurement,
            relation_inner_slack=relation_inner,
            relation_outer_slack=relation_outer,
            relation_comparator=policy.relation_comparator,
            relation_threshold=policy.relation_threshold,
            relation_tolerance=policy.relation_tolerance,
            relation_boundary=policy.relation_boundary,
            relation_squared_distance_bounds=squared_distance_bounds,
            relation_inner_success=relation_inner_success,
            relation_outer_failure=relation_outer_failure,
            common_cell_semantic_objective_terms=semantic_objective_terms,
            common_cell_objective_terms=objective_terms,
            inner_hard_constraint_slack=hard_inner,
            outer_hard_constraint_slack=hard_outer,
            safety_penalty_outer=safety,
        ),
        atomic_steps_used=required_steps,
        proof_rows=(
            f"CELL:{_fixed_cardinal_cell_id_v3(cell, quarter_turns_ccw)}",
            "SO2:CARDINAL_XY:DELEGATED",
            (
                "COLLISION:COMPLETE_COMPOUND_CLOSED_CONTACT"
                if collision_pairs
                else "COLLISION:UNCONSTRAINED:NO_OBSTACLES"
            ),
            "SUPPORT:WORLD_XY_Z_UP:CLOSED_CONTAINMENT",
            f"RELATION:{relation}:{measurement}:{comparator}",
            f"POLICY:{policy.policy_id}:{policy.policy_version}",
            "OBJECTIVE:COMMON_CELL:T-A-R-V-S",
            "SAFETY:HARD_CONSTRAINT_INNER_OUTER",
        ),
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


@dataclass(frozen=True, slots=True)
class ContinuousYawBoxBoundsV4:
    """Directed axes and outer AABB for every pose in one XY-times-yaw cell."""

    box: FixedCardinalBoxV3
    cell: ClosedXYCellV3
    pivot_xy: tuple[Fraction, Fraction]
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4
    local_x_axis: OrientedAxisEnclosureV2
    local_y_axis: OrientedAxisEnclosureV2
    center_x: RationalEnclosureV2
    center_y: RationalEnclosureV2
    center_z: Fraction
    half_x: Fraction
    half_y: Fraction
    half_z: Fraction
    x_radius: RationalEnclosureV2
    y_radius: RationalEnclosureV2
    aabb_x: RationalEnclosureV2
    aabb_y: RationalEnclosureV2
    aabb_z: RationalEnclosureV2
    atomic_steps_used: int

    def __post_init__(self) -> None:
        if type(self.box) is not FixedCardinalBoxV3:
            raise TypeError("box must be a FixedCardinalBoxV3")
        if type(self.cell) is not ClosedXYCellV3:
            raise TypeError("cell must be a ClosedXYCellV3")
        if (
            type(self.pivot_xy) is not tuple
            or len(self.pivot_xy) != 2
            or any(type(value) is not Fraction for value in self.pivot_xy)
        ):
            raise TypeError("pivot_xy must contain two exact Fractions")
        if type(self.yaw_bounds) is not so2_interval.ContinuousYawSinCosBoundsV4:
            raise TypeError("yaw_bounds must be ContinuousYawSinCosBoundsV4")
        for field_name in ("local_x_axis", "local_y_axis"):
            value = getattr(self, field_name)
            if type(value) is not OrientedAxisEnclosureV2:
                raise TypeError(f"{field_name} must be an OrientedAxisEnclosureV2")
            object.__setattr__(
                self,
                field_name,
                OrientedAxisEnclosureV2(x=value.x, y=value.y),
            )
        if not (
            self.local_x_axis.x == self.yaw_bounds.cosine
            and self.local_x_axis.y == self.yaw_bounds.sine
            and self.local_y_axis.x.rational_lower
            == -self.local_x_axis.y.rational_upper
            and self.local_y_axis.x.rational_upper
            == -self.local_x_axis.y.rational_lower
            and self.local_y_axis.y == self.local_x_axis.x
        ):
            raise ValueError(
                "continuous box axes must reproduce the directed yaw bounds"
            )
        for field_name in (
            "center_x",
            "center_y",
            "x_radius",
            "y_radius",
            "aabb_x",
            "aabb_y",
            "aabb_z",
        ):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            object.__setattr__(self, field_name, _copy_enclosure(value))
        for field_name, expected in (
            ("center_z", self.box.center_z),
            ("half_x", self.box.half_x),
            ("half_y", self.box.half_y),
            ("half_z", self.box.half_z),
        ):
            value = getattr(self, field_name)
            if type(value) is not Fraction:
                raise TypeError(f"{field_name} must be an exact Fraction")
            if value != expected:
                raise ValueError(f"{field_name} must reproduce the fixed source box")
        if self.x_radius.rational_lower < 0 or self.y_radius.rational_lower < 0:
            raise ValueError("continuous AABB radii must be non-negative")
        absolute_cosine = _absolute_interval_no_budget(
            self.local_x_axis.x.rational_lower,
            self.local_x_axis.x.rational_upper,
        )
        absolute_sine = _absolute_interval_no_budget(
            self.local_x_axis.y.rational_lower,
            self.local_x_axis.y.rational_upper,
        )
        if _enclosure_tuple(self.x_radius) != (
            absolute_cosine[0] * self.half_x + absolute_sine[0] * self.half_y,
            absolute_cosine[1] * self.half_x + absolute_sine[1] * self.half_y,
        ) or _enclosure_tuple(self.y_radius) != (
            absolute_sine[0] * self.half_x + absolute_cosine[0] * self.half_y,
            absolute_sine[1] * self.half_x + absolute_cosine[1] * self.half_y,
        ):
            raise ValueError("continuous AABB radii must match axes and source extents")
        if (
            _enclosure_tuple(self.aabb_x)
            != (
                self.center_x.rational_lower - self.x_radius.rational_upper,
                self.center_x.rational_upper + self.x_radius.rational_upper,
            )
            or _enclosure_tuple(self.aabb_y)
            != (
                self.center_y.rational_lower - self.y_radius.rational_upper,
                self.center_y.rational_upper + self.y_radius.rational_upper,
            )
            or _enclosure_tuple(self.aabb_z)
            != (
                self.center_z - self.half_z,
                self.center_z + self.half_z,
            )
        ):
            raise ValueError("continuous AABB must enclose its center and radii")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("continuous box atomic usage must be a positive exact int")


@dataclass(frozen=True, slots=True)
class ContinuousYawBoxOutcomeV4:
    """Typed completion of one continuous-yaw upright-box outer bound."""

    kind: so2_interval.ContinuousYawIntervalKindV4
    bounds: ContinuousYawBoxBoundsV4 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not so2_interval.ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be non-blank strings")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is so2_interval.ContinuousYawIntervalKindV4.EXACT:
            if type(self.bounds) is not ContinuousYawBoxBoundsV4 or self.finding_codes:
                raise ValueError("exact continuous box outcome requires bounds only")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError("non-exact continuous box outcome requires findings only")
        expected_prefix = {
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("continuous box finding does not match its typed outcome")


def compile_continuous_yaw_box_bounds_v4(
    box: object,
    *,
    cell: object,
    pivot_xy: object,
    yaw_bounds: object,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawBoxOutcomeV4:
    """Enclose one upright box over a closed XY cell and closed lifted yaw cell."""

    if type(box) is not FixedCardinalBoxV3:
        raise TypeError("box must be a FixedCardinalBoxV3")
    if type(cell) is not ClosedXYCellV3:
        raise TypeError("cell must be a ClosedXYCellV3")
    if (
        type(pivot_xy) is not tuple
        or len(pivot_xy) != 2
        or any(type(value) is not Fraction for value in pivot_xy)
    ):
        raise TypeError("pivot_xy must contain two exact Fractions")
    if type(yaw_bounds) is not so2_interval.ContinuousYawSinCosBoundsV4:
        raise TypeError("yaw_bounds must be ContinuousYawSinCosBoundsV4")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    start_used = atomic_budget.used
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_continuous_yaw_box_bounds_checked_v4(
                box,
                cell,
                pivot_xy,
                yaw_bounds,
                atomic_budget,
                start_used,
            )
    except SO2AtomicBudgetExhaustedV2:
        return _continuous_box_failure_v4("RESOURCE_LIMIT:SO2_ATOMIC_STEPS")
    except so2_interval._SO2NumericGapV2 as error:
        return _continuous_box_failure_v4(error.finding_code)
    except (OverflowError, FloatingPointError):
        return _continuous_box_failure_v4("NUMERIC_GAP:CONTINUOUS_YAW_BOX_ARITHMETIC")
    except RuntimeWarning:
        return _continuous_box_failure_v4(
            "NUMERIC_GAP:CONTINUOUS_YAW_BOX_RUNTIME_WARNING"
        )


def _compile_continuous_yaw_box_bounds_checked_v4(
    box: FixedCardinalBoxV3,
    cell: ClosedXYCellV3,
    pivot_xy: tuple[Fraction, Fraction],
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    budget: SO2AtomicBudgetV2,
    start_used: int,
) -> ContinuousYawBoxOutcomeV4:
    cosine = _enclosure_tuple(yaw_bounds.cosine)
    sine = _enclosure_tuple(yaw_bounds.sine)
    negative_sine = _interval_negate(yaw_bounds.sine, budget)
    relative_x = box.center_x - pivot_xy[0]
    relative_y = box.center_y - pivot_xy[1]
    for value in (*pivot_xy, relative_x, relative_y, box.center_z):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONTINUOUS_YAW_BOX_FRACTION_BIT_CAP",
        )
    rotated_x = _interval_add(
        _interval_scale_signed(cosine, relative_x, budget),
        _interval_scale_signed(negative_sine, relative_y, budget),
        budget,
    )
    rotated_y = _interval_add(
        _interval_scale_signed(sine, relative_x, budget),
        _interval_scale_signed(cosine, relative_y, budget),
        budget,
    )
    center_x = _interval_add(
        _interval_add(rotated_x, (pivot_xy[0], pivot_xy[0]), budget),
        (cell.x_lower, cell.x_upper),
        budget,
    )
    center_y = _interval_add(
        _interval_add(rotated_y, (pivot_xy[1], pivot_xy[1]), budget),
        (cell.y_lower, cell.y_upper),
        budget,
    )
    absolute_cosine = _interval_absolute(yaw_bounds.cosine, budget)
    absolute_sine = _interval_absolute(yaw_bounds.sine, budget)
    x_radius = _interval_add(
        _interval_scale(absolute_cosine, box.half_x, budget),
        _interval_scale(absolute_sine, box.half_y, budget),
        budget,
    )
    y_radius = _interval_add(
        _interval_scale(absolute_sine, box.half_x, budget),
        _interval_scale(absolute_cosine, box.half_y, budget),
        budget,
    )
    aabb_x = _interval_add(
        center_x,
        (-x_radius[1], x_radius[1]),
        budget,
    )
    aabb_y = _interval_add(
        center_y,
        (-y_radius[1], y_radius[1]),
        budget,
    )
    aabb_z = (box.center_z - box.half_z, box.center_z + box.half_z)
    for interval in (center_x, center_y, x_radius, y_radius, aabb_x, aabb_y, aabb_z):
        _checked_interval(interval)
    return ContinuousYawBoxOutcomeV4(
        so2_interval.ContinuousYawIntervalKindV4.EXACT,
        bounds=ContinuousYawBoxBoundsV4(
            box=box,
            cell=cell,
            pivot_xy=pivot_xy,
            yaw_bounds=yaw_bounds,
            local_x_axis=OrientedAxisEnclosureV2(
                x=yaw_bounds.cosine,
                y=yaw_bounds.sine,
            ),
            local_y_axis=OrientedAxisEnclosureV2(
                x=so2_interval._publish_rational_enclosure(*negative_sine),
                y=yaw_bounds.cosine,
            ),
            center_x=_publish_interval(center_x, budget),
            center_y=_publish_interval(center_y, budget),
            center_z=box.center_z,
            half_x=box.half_x,
            half_y=box.half_y,
            half_z=box.half_z,
            x_radius=_publish_interval(x_radius, budget),
            y_radius=_publish_interval(y_radius, budget),
            aabb_x=_publish_interval(aabb_x, budget),
            aabb_y=_publish_interval(aabb_y, budget),
            aabb_z=_publish_interval(aabb_z, budget),
            atomic_steps_used=budget.used - start_used,
        ),
    )


def _continuous_box_failure_v4(finding_code: str) -> ContinuousYawBoxOutcomeV4:
    if finding_code.startswith("RESOURCE_LIMIT:"):
        kind = so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT
    elif finding_code.startswith("NUMERIC_GAP:"):
        kind = so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP
    else:
        raise ValueError(
            "continuous box failures must be typed resource or numeric gaps"
        )
    return ContinuousYawBoxOutcomeV4(kind, finding_codes=(finding_code,))


class ContinuousYawBoxContactKindV4(StrEnum):
    """Only whole-cell separation is certifiable from outer AABB envelopes."""

    PROVEN_SEPARATED = "PROVEN_SEPARATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ContinuousYawBoxPairBoundsV4:
    """Canonical outer-AABB contact and clearance bounds for two yaw cells."""

    left: ContinuousYawBoxBoundsV4
    right: ContinuousYawBoxBoundsV4
    x_gap: RationalEnclosureV2
    y_gap: RationalEnclosureV2
    z_gap: RationalEnclosureV2
    squared_clearance: RationalEnclosureV2
    contact_kind: ContinuousYawBoxContactKindV4
    atomic_steps_used: int

    def __post_init__(self) -> None:
        if type(self.left) is not ContinuousYawBoxBoundsV4:
            raise TypeError("left must be a ContinuousYawBoxBoundsV4")
        if type(self.right) is not ContinuousYawBoxBoundsV4:
            raise TypeError("right must be a ContinuousYawBoxBoundsV4")
        if _continuous_box_operand_key_v4(self.left) > _continuous_box_operand_key_v4(
            self.right
        ):
            raise ValueError("continuous box pair operands are not canonical")
        for field_name in ("x_gap", "y_gap", "z_gap", "squared_clearance"):
            value = getattr(self, field_name)
            if type(value) is not RationalEnclosureV2:
                raise TypeError(f"{field_name} must be a RationalEnclosureV2")
            checked = _copy_enclosure(value)
            if checked.rational_lower < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, checked)
        expected_x = _continuous_axis_gap_no_budget_v4(
            _enclosure_tuple(self.left.aabb_x),
            _enclosure_tuple(self.right.aabb_x),
        )
        expected_y = _continuous_axis_gap_no_budget_v4(
            _enclosure_tuple(self.left.aabb_y),
            _enclosure_tuple(self.right.aabb_y),
        )
        expected_z = _continuous_axis_gap_no_budget_v4(
            _enclosure_tuple(self.left.aabb_z),
            _enclosure_tuple(self.right.aabb_z),
        )
        if (
            _enclosure_tuple(self.x_gap),
            _enclosure_tuple(self.y_gap),
            _enclosure_tuple(self.z_gap),
        ) != (expected_x, expected_y, expected_z):
            raise ValueError("continuous pair gaps must match the outer AABBs")
        expected_clearance = (
            expected_x[0] * expected_x[0]
            + expected_y[0] * expected_y[0]
            + expected_z[0] * expected_z[0],
            expected_x[1] * expected_x[1]
            + expected_y[1] * expected_y[1]
            + expected_z[1] * expected_z[1],
        )
        if _enclosure_tuple(self.squared_clearance) != expected_clearance:
            raise ValueError("continuous pair clearance must match its axis gaps")
        if type(self.contact_kind) is not ContinuousYawBoxContactKindV4:
            raise TypeError("contact_kind must be a ContinuousYawBoxContactKindV4")
        expected_kind = (
            ContinuousYawBoxContactKindV4.PROVEN_SEPARATED
            if any(value[0] > 0 for value in (expected_x, expected_y, expected_z))
            else ContinuousYawBoxContactKindV4.UNKNOWN
        )
        if self.contact_kind is not expected_kind:
            raise ValueError("continuous pair contact classification is not sound")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError(
                "continuous pair atomic usage must be a positive exact int"
            )


@dataclass(frozen=True, slots=True)
class ContinuousYawBoxPairOutcomeV4:
    """Typed completion of whole-cell outer contact and clearance evaluation."""

    kind: so2_interval.ContinuousYawIntervalKindV4
    bounds: ContinuousYawBoxPairBoundsV4 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not so2_interval.ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be non-blank strings")
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.kind is so2_interval.ContinuousYawIntervalKindV4.EXACT:
            if (
                type(self.bounds) is not ContinuousYawBoxPairBoundsV4
                or self.finding_codes
            ):
                raise ValueError("exact continuous pair outcome requires bounds only")
            return
        if self.bounds is not None or not self.finding_codes:
            raise ValueError("non-exact continuous pair outcome requires findings only")
        expected_prefix = {
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("continuous pair finding does not match its typed outcome")


def evaluate_continuous_yaw_box_pair_bounds_v4(
    left: object,
    right: object,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawBoxPairOutcomeV4:
    """Certify only outer-AABB separation over two complete continuous pose cells."""

    if type(left) is not ContinuousYawBoxBoundsV4:
        raise TypeError("left must be a ContinuousYawBoxBoundsV4")
    if type(right) is not ContinuousYawBoxBoundsV4:
        raise TypeError("right must be a ContinuousYawBoxBoundsV4")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if _continuous_box_operand_key_v4(right) < _continuous_box_operand_key_v4(left):
        left, right = right, left
    start_used = atomic_budget.used
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            x_gap = _continuous_axis_gap_v4(
                _enclosure_tuple(left.aabb_x),
                _enclosure_tuple(right.aabb_x),
                atomic_budget,
            )
            y_gap = _continuous_axis_gap_v4(
                _enclosure_tuple(left.aabb_y),
                _enclosure_tuple(right.aabb_y),
                atomic_budget,
            )
            z_gap = _continuous_axis_gap_v4(
                _enclosure_tuple(left.aabb_z),
                _enclosure_tuple(right.aabb_z),
                atomic_budget,
            )
            squared_clearance = (
                x_gap[0] * x_gap[0] + y_gap[0] * y_gap[0] + z_gap[0] * z_gap[0],
                x_gap[1] * x_gap[1] + y_gap[1] * y_gap[1] + z_gap[1] * z_gap[1],
            )
            _checked_interval(squared_clearance)
            contact_kind = (
                ContinuousYawBoxContactKindV4.PROVEN_SEPARATED
                if any(value[0] > 0 for value in (x_gap, y_gap, z_gap))
                else ContinuousYawBoxContactKindV4.UNKNOWN
            )
            return ContinuousYawBoxPairOutcomeV4(
                so2_interval.ContinuousYawIntervalKindV4.EXACT,
                bounds=ContinuousYawBoxPairBoundsV4(
                    left=left,
                    right=right,
                    x_gap=_publish_interval(x_gap, atomic_budget),
                    y_gap=_publish_interval(y_gap, atomic_budget),
                    z_gap=_publish_interval(z_gap, atomic_budget),
                    squared_clearance=_publish_interval(
                        squared_clearance,
                        atomic_budget,
                    ),
                    contact_kind=contact_kind,
                    atomic_steps_used=atomic_budget.used - start_used,
                ),
            )
    except SO2AtomicBudgetExhaustedV2:
        return _continuous_pair_failure_v4("RESOURCE_LIMIT:SO2_ATOMIC_STEPS")
    except so2_interval._SO2NumericGapV2 as error:
        return _continuous_pair_failure_v4(error.finding_code)
    except (OverflowError, FloatingPointError):
        return _continuous_pair_failure_v4("NUMERIC_GAP:CONTINUOUS_YAW_PAIR_ARITHMETIC")
    except RuntimeWarning:
        return _continuous_pair_failure_v4(
            "NUMERIC_GAP:CONTINUOUS_YAW_PAIR_RUNTIME_WARNING"
        )


def _continuous_box_operand_key_v4(
    value: ContinuousYawBoxBoundsV4,
) -> tuple[object, ...]:
    return (
        value.box.box_id,
        value.box.center_x,
        value.box.center_y,
        value.box.center_z,
        value.box.half_x,
        value.box.half_y,
        value.box.half_z,
        value.cell.canonical_bounds,
        value.pivot_xy,
        value.yaw_bounds.lift_bounds.lower,
        value.yaw_bounds.lift_bounds.upper,
        value.yaw_bounds.lift_bounds.coverage_sha256,
    )


def _continuous_axis_gap_no_budget_v4(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return _checked_interval(
        (
            max(Fraction(), right[0] - left[1], left[0] - right[1]),
            max(Fraction(), right[1] - left[0], left[1] - right[0]),
        )
    )


def _continuous_axis_gap_v4(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
    budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    budget.consume()
    return _continuous_axis_gap_no_budget_v4(left, right)


def _continuous_pair_failure_v4(finding_code: str) -> ContinuousYawBoxPairOutcomeV4:
    if finding_code.startswith("RESOURCE_LIMIT:"):
        kind = so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT
    elif finding_code.startswith("NUMERIC_GAP:"):
        kind = so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP
    else:
        raise ValueError(
            "continuous pair failures must be typed resource or numeric gaps"
        )
    return ContinuousYawBoxPairOutcomeV4(kind, finding_codes=(finding_code,))


@dataclass(frozen=True, slots=True)
class ContinuousYawCellPolicyV4:
    """One request-bound policy bundle for a continuous compound pose cell.

    The retained V3 record remains the sole owner of the request's semantic
    thresholds, safety rule, objective weights, and resource limit.  This
    versioned wrapper deliberately adds no ambient continuous defaults.
    """

    cardinal_policy: FixedCardinalCellPolicyV3

    def __post_init__(self) -> None:
        if type(self.cardinal_policy) is not FixedCardinalCellPolicyV3:
            raise TypeError("cardinal_policy must be a FixedCardinalCellPolicyV3")


@dataclass(frozen=True, slots=True)
class ContinuousYawCellBoundsV4:
    """Directed semantic and objective bounds over one continuous compound cell."""

    cell_id: str
    cell: ClosedXYCellV3
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4
    subject_box_ids: tuple[str, ...]
    obstacle_box_ids: tuple[str, ...]
    collision_constraint_present: bool
    collision_inner_slack: Fraction
    collision_outer_slack: Fraction
    collision_inward_proven: bool
    collision_outer_failure: bool
    support_contact_gap: tuple[Fraction, Fraction]
    support_clearance_inner: Fraction
    support_clearance_outer: Fraction
    support_normal: tuple[Fraction, Fraction, Fraction]
    support_frame: str
    closed_containment: bool
    stability_margin_inner: Fraction
    relation_symbol: str
    relation_measurement: str
    relation_inner_slack: Fraction
    relation_outer_slack: Fraction
    relation_comparator: str
    relation_threshold: Fraction
    relation_tolerance: Fraction
    relation_boundary: str
    relation_squared_distance_bounds: tuple[Fraction, Fraction] | None
    relation_inner_success: bool
    relation_outer_failure: bool
    visibility_inner_fraction: Fraction
    visibility_outer_fraction: Fraction
    visibility_classification: str
    semantic_objective_terms: tuple[tuple[str, Fraction, Fraction], ...]
    weighted_objective_terms: tuple[tuple[str, Fraction, Fraction], ...]
    inner_hard_constraint_slack: Fraction
    outer_hard_constraint_slack: Fraction
    inner_hard_constraint_proven: bool
    outer_hard_constraint_failure: bool
    safety_penalty_outer: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.cell_id) is not str
            or not self.cell_id
            or type(self.cell) is not ClosedXYCellV3
            or type(self.yaw_bounds) is not so2_interval.ContinuousYawSinCosBoundsV4
            or type(self.subject_box_ids) is not tuple
            or not self.subject_box_ids
            or any(
                type(value) is not str or not value for value in self.subject_box_ids
            )
            or self.subject_box_ids != tuple(sorted(set(self.subject_box_ids)))
            or type(self.obstacle_box_ids) is not tuple
            or any(
                type(value) is not str or not value for value in self.obstacle_box_ids
            )
            or self.obstacle_box_ids != tuple(sorted(set(self.obstacle_box_ids)))
            or type(self.collision_constraint_present) is not bool
            or type(self.collision_inward_proven) is not bool
            or type(self.collision_outer_failure) is not bool
            or type(self.closed_containment) is not bool
            or type(self.relation_inner_success) is not bool
            or type(self.relation_outer_failure) is not bool
            or type(self.inner_hard_constraint_proven) is not bool
            or type(self.outer_hard_constraint_failure) is not bool
            or self.support_normal != (Fraction(), Fraction(), Fraction(1))
            or self.support_frame != "WORLD_XY_Z_UP"
            or self.relation_symbol
            not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR")
            or self.relation_measurement
            not in (
                "EXTENT_AWARE_SIGNED_AXIS_GAP",
                "EXTENT_AWARE_EUCLIDEAN_SEPARATION",
            )
            or self.relation_comparator not in ("LE", "GE")
            or self.relation_boundary != "CLOSED"
            or self.visibility_classification not in ("INWARD", "OUTWARD", "UNKNOWN")
        ):
            raise ValueError("continuous yaw cell bounds are not a closed exact record")
        for value in (
            self.collision_inner_slack,
            self.collision_outer_slack,
            self.support_clearance_inner,
            self.support_clearance_outer,
            self.stability_margin_inner,
            self.relation_inner_slack,
            self.relation_outer_slack,
            self.relation_threshold,
            self.relation_tolerance,
            self.visibility_inner_fraction,
            self.visibility_outer_fraction,
            self.inner_hard_constraint_slack,
            self.outer_hard_constraint_slack,
            self.safety_penalty_outer,
        ):
            if type(value) is not Fraction:
                raise TypeError(
                    "continuous yaw semantic bounds must be exact Fractions"
                )
        if (
            self.collision_inner_slack > self.collision_outer_slack
            or self.support_clearance_inner > self.support_clearance_outer
            or self.relation_inner_slack > self.relation_outer_slack
            or self.inner_hard_constraint_slack > self.outer_hard_constraint_slack
            or self.relation_tolerance < 0
            or not Fraction()
            <= self.visibility_inner_fraction
            <= self.visibility_outer_fraction
            <= Fraction(1)
            or self.safety_penalty_outer < 0
        ):
            raise ValueError("continuous yaw semantic intervals must be ordered")
        if (
            type(self.support_contact_gap) is not tuple
            or len(self.support_contact_gap) != 2
            or any(type(value) is not Fraction for value in self.support_contact_gap)
            or self.support_contact_gap[0] > self.support_contact_gap[1]
        ):
            raise ValueError("support contact gap must be an ordered exact interval")
        if self.relation_squared_distance_bounds is not None and (
            type(self.relation_squared_distance_bounds) is not tuple
            or len(self.relation_squared_distance_bounds) != 2
            or any(
                type(value) is not Fraction
                for value in self.relation_squared_distance_bounds
            )
            or self.relation_squared_distance_bounds[0]
            > self.relation_squared_distance_bounds[1]
        ):
            raise ValueError("squared distance bounds must be ordered exact Fractions")
        for terms in (self.semantic_objective_terms, self.weighted_objective_terms):
            if (
                type(terms) is not tuple
                or tuple(term_id for term_id, _, _ in terms)
                != ("T", "A", "R", "V", "S")
                or any(
                    type(value) is not Fraction
                    for _, lower, upper in terms
                    for value in (lower, upper)
                )
                or any(lower > upper for _, lower, upper in terms)
            ):
                raise ValueError("objective terms must be ordered T/A/R/V/S intervals")


@dataclass(frozen=True, slots=True)
class ContinuousYawCellOutcomeV4:
    """Typed completion of continuous compound semantic ownership."""

    kind: so2_interval.ContinuousYawIntervalKindV4
    bounds: ContinuousYawCellBoundsV4 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not so2_interval.ContinuousYawIntervalKindV4:
            raise TypeError("continuous yaw cell outcome must use its V4 kind")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError("continuous yaw cell atomic usage must be non-negative")
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(f"continuous yaw cell {field_name} must be non-blank")
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is so2_interval.ContinuousYawIntervalKindV4.EXACT:
            if (
                type(self.bounds) is not ContinuousYawCellBoundsV4
                or self.finding_codes
                or not self.proof_rows
            ):
                raise ValueError(
                    "exact continuous yaw cell outcome requires bounds and proof"
                )
            return
        if self.bounds is not None or not self.finding_codes or not self.proof_rows:
            raise ValueError(
                "non-exact continuous yaw cell outcome requires typed proof"
            )
        prefix = {
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(prefix) for code in self.finding_codes):
            raise ValueError("continuous yaw cell finding must match its typed outcome")


_ContinuousExtentIntervalsV4 = tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
]


def evaluate_continuous_yaw_cell_v4(
    *,
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    subject_boxes: tuple[ContinuousYawBoxBoundsV4, ...],
    obstacle_boxes: tuple[ContinuousYawBoxBoundsV4, ...],
    support_surface: SupportSurfaceV3,
    relation: str,
    reference_box: ContinuousYawBoxBoundsV4,
    near_far_threshold: Fraction,
    policy: ContinuousYawCellPolicyV4,
    visibility: object,
    atomic_budget: SO2AtomicBudgetV2,
    subject_pivot_xy: tuple[Fraction, Fraction] = (Fraction(), Fraction()),
    objective_subject_pivot_xy: tuple[Fraction, Fraction] | None = None,
) -> ContinuousYawCellOutcomeV4:
    """Evaluate one closed continuous pose cell without reimplementing sibling owners.

    Inputs are already-produced Task 6 box and visibility DTOs.  This owner
    combines their directed intervals; it neither samples yaw nor constructs a
    camera projection.  A nonpoint contact envelope can prove separation or
    failure, but remains explicitly unresolved at touch/overlap ambiguity.
    """

    if type(cell) is not ClosedXYCellV3:
        raise TypeError("cell must be a ClosedXYCellV3")
    if type(yaw_bounds) is not so2_interval.ContinuousYawSinCosBoundsV4:
        raise TypeError("yaw_bounds must be ContinuousYawSinCosBoundsV4")
    subjects = _require_continuous_cell_roster_v4(
        subject_boxes,
        label="subject_boxes",
        nonempty=True,
    )
    obstacles = _require_continuous_cell_roster_v4(
        obstacle_boxes,
        label="obstacle_boxes",
        nonempty=False,
    )
    if type(support_surface) is not SupportSurfaceV3:
        raise TypeError("support_surface must be a SupportSurfaceV3")
    if relation not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR"):
        raise ValueError("relation must be a registered continuous yaw relation")
    if type(reference_box) is not ContinuousYawBoxBoundsV4:
        raise TypeError("reference_box must be a ContinuousYawBoxBoundsV4")
    if type(near_far_threshold) is not Fraction or near_far_threshold < 0:
        raise ValueError("near_far_threshold must be a non-negative exact Fraction")
    if type(policy) is not ContinuousYawCellPolicyV4:
        raise TypeError("policy must be a ContinuousYawCellPolicyV4")
    cardinal_policy = policy.cardinal_policy
    if near_far_threshold != cardinal_policy.relation_threshold:
        raise ValueError("near_far_threshold must match the caller policy")
    if cardinal_policy.relation_symbol != relation:
        raise ValueError("relation must match the caller-bound registered symbol")
    expected_measurement = (
        "EXTENT_AWARE_EUCLIDEAN_SEPARATION"
        if relation in ("NEAR", "FAR")
        else "EXTENT_AWARE_SIGNED_AXIS_GAP"
    )
    if cardinal_policy.relation_measurement != expected_measurement:
        raise ValueError("relation measurement must match its registered symbol")
    if cardinal_policy.visibility_cell != cell.canonical_bounds:
        raise ValueError("visibility policy must be bound to this exact XY cell")
    if (
        type(subject_pivot_xy) is not tuple
        or len(subject_pivot_xy) != 2
        or any(type(value) is not Fraction for value in subject_pivot_xy)
    ):
        raise TypeError("subject_pivot_xy must contain two exact Fractions")
    if objective_subject_pivot_xy is None:
        objective_subject_pivot_xy = subject_pivot_xy
    if (
        type(objective_subject_pivot_xy) is not tuple
        or len(objective_subject_pivot_xy) != 2
        or any(type(value) is not Fraction for value in objective_subject_pivot_xy)
    ):
        raise TypeError("objective_subject_pivot_xy must contain two exact Fractions")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if atomic_budget.limit != cardinal_policy.atomic_step_limit:
        raise ValueError("atomic budget limit must match the caller policy")
    _validate_continuous_cell_geometry_v4(
        cell=cell,
        yaw_bounds=yaw_bounds,
        subjects=subjects,
        obstacles=obstacles,
        reference=reference_box,
        subject_pivot_xy=subject_pivot_xy,
    )
    if not _is_continuous_visibility_outcome_v4(visibility):
        raise TypeError("visibility must be a ContinuousYawVisibilityOutcomeV4")
    visibility_kind = visibility.kind  # type: ignore[union-attr]
    if visibility_kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT:
        return _continuous_cell_from_visibility_nonexact_v4(visibility)
    if not _is_continuous_visibility_bounds_v4(visibility.bounds):  # type: ignore[union-attr]
        raise RuntimeError("exact continuous visibility outcome is missing bounds")
    _validate_continuous_cell_visibility_v4(
        visibility=visibility.bounds,  # type: ignore[union-attr]
        cell=cell,
        yaw_bounds=yaw_bounds,
        subject=subjects[0],
    )
    point_quarter = _continuous_cell_cardinal_point_quarter_v4(cell, yaw_bounds)
    if point_quarter is not None:
        return _delegate_continuous_cell_cardinal_point_v4(
            cell=cell,
            quarter_turns_ccw=point_quarter,
            subjects=subjects,
            obstacles=obstacles,
            support_surface=support_surface,
            relation=relation,
            reference=reference_box,
            near_far_threshold=near_far_threshold,
            policy=cardinal_policy,
            yaw_bounds=yaw_bounds,
            visibility=visibility.bounds,  # type: ignore[union-attr]
            atomic_budget=atomic_budget,
            subject_pivot_xy=subject_pivot_xy,
            objective_subject_pivot_xy=objective_subject_pivot_xy,
        )

    pair_calls = len(subjects) * len(obstacles)
    if relation in ("NEAR", "FAR"):
        pair_calls += len(subjects)
    # Pair-owner evaluation consumes seven retained atomic steps per pair;
    # translation/geodesic interval assembly consumes the remaining fourteen.
    required_steps = 14 + 7 * pair_calls
    if atomic_budget.remaining < required_steps:
        return _continuous_cell_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
            atomic_steps_used=0,
            proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:cap-minus-one",),
        )
    start_used = atomic_budget.used
    try:
        collision_pairs = tuple(
            evaluate_continuous_yaw_box_pair_bounds_v4(
                subject,
                obstacle,
                atomic_budget=atomic_budget,
            )
            for subject in subjects
            for obstacle in obstacles
        )
        if any(
            pair.kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT
            for pair in collision_pairs
        ):
            return _continuous_cell_from_pair_nonexact_v4(
                collision_pairs, start_used, atomic_budget
            )
        relation_pairs = (
            tuple(
                evaluate_continuous_yaw_box_pair_bounds_v4(
                    subject,
                    reference_box,
                    atomic_budget=atomic_budget,
                )
                for subject in subjects
            )
            if relation in ("NEAR", "FAR")
            else ()
        )
        if any(
            pair.kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT
            for pair in relation_pairs
        ):
            return _continuous_cell_from_pair_nonexact_v4(
                relation_pairs, start_used, atomic_budget
            )
        return _evaluate_continuous_yaw_cell_checked_v4(
            cell=cell,
            yaw_bounds=yaw_bounds,
            subjects=subjects,
            obstacles=obstacles,
            support_surface=support_surface,
            relation=relation,
            reference=reference_box,
            policy=cardinal_policy,
            visibility=visibility.bounds,  # type: ignore[union-attr]
            collision_pairs=collision_pairs,
            relation_pairs=relation_pairs,
            atomic_budget=atomic_budget,
            start_used=start_used,
            required_steps=required_steps,
            subject_pivot_xy=subject_pivot_xy,
            objective_subject_pivot_xy=objective_subject_pivot_xy,
        )
    except SO2AtomicBudgetExhaustedV2:
        return _continuous_cell_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
            atomic_steps_used=atomic_budget.used - start_used,
            proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:continuous-compound-owner",),
        )
    except so2_interval._SO2NumericGapV2 as error:
        return _continuous_cell_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP,
            error.finding_code,
            atomic_steps_used=atomic_budget.used - start_used,
            proof_rows=(error.finding_code,),
        )
    except (OverflowError, FloatingPointError):
        return _continuous_cell_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_CELL_ARITHMETIC",
            atomic_steps_used=atomic_budget.used - start_used,
            proof_rows=("NUMERIC_GAP:CONTINUOUS_YAW_CELL_ARITHMETIC",),
        )
    except RuntimeWarning:
        return _continuous_cell_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_CELL_RUNTIME_WARNING",
            atomic_steps_used=atomic_budget.used - start_used,
            proof_rows=("NUMERIC_GAP:CONTINUOUS_YAW_CELL_RUNTIME_WARNING",),
        )


def _require_continuous_cell_roster_v4(
    value: object, *, label: str, nonempty: bool
) -> tuple[ContinuousYawBoxBoundsV4, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ValueError(
            f"{label} must be an exact {'non-empty ' if nonempty else ''}tuple"
        )
    boxes = tuple(
        item
        if type(item) is ContinuousYawBoxBoundsV4
        else (_raise_continuous_cell_box_type_v4(label))
        for item in value
    )
    ids = tuple(box.box.box_id for box in boxes)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ValueError(f"{label} must be sorted and have unique box IDs")
    return boxes


def _raise_continuous_cell_box_type_v4(label: str) -> ContinuousYawBoxBoundsV4:
    raise TypeError(f"{label} must contain ContinuousYawBoxBoundsV4 values")


def _validate_continuous_cell_geometry_v4(
    *,
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    subjects: tuple[ContinuousYawBoxBoundsV4, ...],
    obstacles: tuple[ContinuousYawBoxBoundsV4, ...],
    reference: ContinuousYawBoxBoundsV4,
    subject_pivot_xy: tuple[Fraction, Fraction],
) -> None:
    for subject in subjects:
        if (
            subject.cell != cell
            or subject.yaw_bounds != yaw_bounds
            or subject.pivot_xy != subject_pivot_xy
        ):
            raise ValueError(
                "subject boxes must bind the exact continuous cell and pivot"
            )
    for box in obstacles:
        if not _is_static_world_box_v4(box):
            raise ValueError("obstacle boxes must be static world-frame point boxes")
    if not _is_static_world_box_v4(reference):
        raise ValueError("reference boxes must be static world-frame point boxes")


def _is_static_world_box_v4(value: ContinuousYawBoxBoundsV4) -> bool:
    lift = value.yaw_bounds.lift_bounds
    return (
        value.cell.canonical_bounds == (Fraction(), Fraction(), Fraction(), Fraction())
        and value.pivot_xy == (Fraction(), Fraction())
        and lift.lower == lift.upper == Fraction()
    )


def _continuous_visibility_dto_types_v4() -> tuple[type[object], type[object]]:
    """Load real sibling DTO identities only after this owner has initialized.

    ``projected_visibility`` imports this owner during its own module setup, so
    a module-level import would recreate that cycle.  Evaluation happens after
    both kernel modules are usable; this lazy import preserves the ownership
    boundary while rejecting module/name lookalikes.
    """

    from spatialcf.core._internal.kernels.projected_visibility import (
        ContinuousYawVisibilityBoundsV4,
        ContinuousYawVisibilityOutcomeV4,
    )

    return ContinuousYawVisibilityOutcomeV4, ContinuousYawVisibilityBoundsV4


def _is_continuous_visibility_outcome_v4(value: object) -> bool:
    outcome_type, _ = _continuous_visibility_dto_types_v4()
    return type(value) is outcome_type


def _is_continuous_visibility_bounds_v4(value: object) -> bool:
    _, bounds_type = _continuous_visibility_dto_types_v4()
    return type(value) is bounds_type


def _validate_continuous_cell_visibility_v4(
    *,
    visibility: object,
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    subject: ContinuousYawBoxBoundsV4,
) -> None:
    lift = yaw_bounds.lift_bounds
    if (
        visibility.cell != cell.canonical_bounds
        or visibility.lifted_turn_bounds != (lift.lower, lift.upper)
        or visibility.continuous_yaw_lift_sha256 != lift.coverage_sha256
        or visibility.subject_box_id != subject.box.box_id
        or visibility.subject_box_id not in visibility.occluder_roster
    ):
        raise ValueError("visibility interval must bind the exact continuous pose cell")


def _continuous_cell_cardinal_point_quarter_v4(
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
) -> int | None:
    lift = yaw_bounds.lift_bounds
    if (
        cell.x_lower != cell.x_upper
        or cell.y_lower != cell.y_upper
        or lift.lower != lift.upper
    ):
        return None
    return so2_interval._cardinal_quarter_turn_v4(lift.lower)


def _delegate_continuous_cell_cardinal_point_v4(
    *,
    cell: ClosedXYCellV3,
    quarter_turns_ccw: int,
    subjects: tuple[ContinuousYawBoxBoundsV4, ...],
    obstacles: tuple[ContinuousYawBoxBoundsV4, ...],
    support_surface: SupportSurfaceV3,
    relation: str,
    reference: ContinuousYawBoxBoundsV4,
    near_far_threshold: Fraction,
    policy: FixedCardinalCellPolicyV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    visibility: object,
    atomic_budget: SO2AtomicBudgetV2,
    subject_pivot_xy: tuple[Fraction, Fraction],
    objective_subject_pivot_xy: tuple[Fraction, Fraction],
) -> ContinuousYawCellOutcomeV4:
    retained = evaluate_fixed_cardinal_cell_v3(
        cell=cell,
        quarter_turns_ccw=quarter_turns_ccw,
        subject_boxes=tuple(value.box for value in subjects),
        obstacle_boxes=tuple(value.box for value in obstacles),
        support_surface=support_surface,
        relation=relation,
        reference_box=reference.box,
        near_far_threshold=near_far_threshold,
        policy=policy,
        atomic_budget=atomic_budget,
        subject_pivot_xy=subject_pivot_xy,
        objective_subject_pivot_xy=objective_subject_pivot_xy,
    )
    kind = so2_interval.ContinuousYawIntervalKindV4(retained.kind.value)
    if kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT:
        return ContinuousYawCellOutcomeV4(
            kind,
            atomic_steps_used=retained.atomic_steps_used,
            proof_rows=retained.proof_rows,
            finding_codes=retained.finding_codes,
        )
    if type(retained.bounds) is not FixedCardinalCellBoundsV3:
        raise RuntimeError("exact retained cardinal outcome is missing bounds")
    bounds = retained.bounds
    collision_inward = (
        not bounds.collision_constraint_present or bounds.collision_inner_slack >= 0
    )
    return ContinuousYawCellOutcomeV4(
        kind,
        bounds=ContinuousYawCellBoundsV4(
            cell_id=bounds.cell_id,
            cell=cell,
            yaw_bounds=yaw_bounds,
            subject_box_ids=tuple(value.box.box_id for value in subjects),
            obstacle_box_ids=tuple(value.box.box_id for value in obstacles),
            collision_constraint_present=bounds.collision_constraint_present,
            collision_inner_slack=bounds.collision_inner_slack,
            collision_outer_slack=bounds.collision_outer_slack,
            collision_inward_proven=collision_inward,
            collision_outer_failure=bounds.collision_outer_slack < 0,
            support_contact_gap=bounds.support_contact_gap,
            support_clearance_inner=bounds.support_clearance_inner,
            support_clearance_outer=bounds.support_clearance_outer,
            support_normal=bounds.support_normal,
            support_frame=bounds.support_frame,
            closed_containment=bounds.closed_containment,
            stability_margin_inner=bounds.stability_margin_inner,
            relation_symbol=relation,
            relation_measurement=bounds.relation_measurement,
            relation_inner_slack=bounds.relation_inner_slack,
            relation_outer_slack=bounds.relation_outer_slack,
            relation_comparator=bounds.relation_comparator,
            relation_threshold=bounds.relation_threshold,
            relation_tolerance=bounds.relation_tolerance,
            relation_boundary=bounds.relation_boundary,
            relation_squared_distance_bounds=bounds.relation_squared_distance_bounds,
            relation_inner_success=bounds.relation_inner_success,
            relation_outer_failure=bounds.relation_outer_failure,
            visibility_inner_fraction=visibility.inner_fraction,
            visibility_outer_fraction=visibility.outer_fraction,
            visibility_classification=visibility.classification.value,
            semantic_objective_terms=bounds.common_cell_semantic_objective_terms,
            weighted_objective_terms=bounds.common_cell_objective_terms,
            inner_hard_constraint_slack=bounds.inner_hard_constraint_slack,
            outer_hard_constraint_slack=bounds.outer_hard_constraint_slack,
            inner_hard_constraint_proven=(
                collision_inward
                and bounds.support_clearance_inner >= 0
                and bounds.relation_inner_success
            ),
            outer_hard_constraint_failure=bounds.outer_hard_constraint_slack < 0,
            safety_penalty_outer=bounds.safety_penalty_outer,
        ),
        atomic_steps_used=retained.atomic_steps_used,
        proof_rows=tuple(sorted((*retained.proof_rows, *visibility.proof_rows))),
    )


def _evaluate_continuous_yaw_cell_checked_v4(
    *,
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    subjects: tuple[ContinuousYawBoxBoundsV4, ...],
    obstacles: tuple[ContinuousYawBoxBoundsV4, ...],
    support_surface: SupportSurfaceV3,
    relation: str,
    reference: ContinuousYawBoxBoundsV4,
    policy: FixedCardinalCellPolicyV3,
    visibility: object,
    collision_pairs: tuple[ContinuousYawBoxPairOutcomeV4, ...],
    relation_pairs: tuple[ContinuousYawBoxPairOutcomeV4, ...],
    atomic_budget: SO2AtomicBudgetV2,
    start_used: int,
    required_steps: int,
    subject_pivot_xy: tuple[Fraction, Fraction],
    objective_subject_pivot_xy: tuple[Fraction, Fraction],
) -> ContinuousYawCellOutcomeV4:
    subject_extent = _compound_continuous_extent_intervals_v4(subjects)
    reference_extent = _continuous_extent_intervals_v4(reference)
    if collision_pairs:
        pair_bounds = tuple(
            pair.bounds for pair in collision_pairs if pair.bounds is not None
        )
        signed = tuple(
            _continuous_signed_pair_clearance_v4(left, right)
            for left in subjects
            for right in obstacles
        )
        collision_inner = min(value[0] for value in signed) - policy.collision_clearance
        collision_outer = min(value[1] for value in signed) - policy.collision_clearance
        collision_inward = (
            all(
                pair.contact_kind is ContinuousYawBoxContactKindV4.PROVEN_SEPARATED
                for pair in pair_bounds
            )
            and collision_inner >= 0
        )
        collision_outer_failure = collision_outer < 0
        collision_row = (
            "COLLISION:CONTINUOUS_COMPOUND_PROVEN_SEPARATED"
            if collision_inward
            else "COLLISION:NONPOINT_CONTACT_UNRESOLVED"
        )
    else:
        collision_inner = Fraction()
        collision_outer = Fraction()
        collision_inward = True
        collision_outer_failure = False
        collision_row = "COLLISION:UNCONSTRAINED:NO_OBSTACLES"
    containment_inner, containment_outer = _continuous_support_containment_v4(
        subject_extent,
        support_surface,
    )
    contact_gap = (
        min(value.center_z - value.half_z - support_surface.z for value in subjects),
        max(value.center_z - value.half_z - support_surface.z for value in subjects),
    )
    accepted_lower, accepted_upper = policy.support_accepted_contact_gap
    contact_inner = min(
        contact_gap[0] - accepted_lower,
        accepted_upper - contact_gap[1],
    )
    contact_outer = min(
        contact_gap[1] - accepted_lower,
        accepted_upper - contact_gap[0],
    )
    support_inner = min(
        containment_inner - policy.support_stability_margin,
        contact_inner,
    )
    support_outer = min(
        containment_outer - policy.support_stability_margin,
        contact_outer,
    )
    (
        relation_inner,
        relation_outer,
        relation_measurement,
        relation_squared,
    ) = _continuous_relation_slacks_v4(
        relation=relation,
        subject_extent=subject_extent,
        reference_extent=reference_extent,
        pair_outcomes=relation_pairs,
        policy=policy,
    )
    relation_inner_success = relation_inner >= 0
    relation_outer_failure = relation_outer < 0
    hard_parts_inner = (support_inner, relation_inner)
    hard_parts_outer = (support_outer, relation_outer)
    if collision_pairs:
        hard_parts_inner = (collision_inner, *hard_parts_inner)
        hard_parts_outer = (collision_outer, *hard_parts_outer)
    hard_inner = min(hard_parts_inner)
    hard_outer = min(hard_parts_outer)
    translation = _continuous_translation_objective_v4(
        cell=cell,
        yaw_bounds=yaw_bounds,
        subject_pivot_xy=subject_pivot_xy,
        objective_subject_pivot_xy=objective_subject_pivot_xy,
        atomic_budget=atomic_budget,
    )
    atomic_budget.consume()
    angle = _continuous_shortest_turn_interval_v4(
        yaw_bounds.lift_bounds.lower, yaw_bounds.lift_bounds.upper
    )
    relation_damage = (
        max(Fraction(), -relation_outer),
        max(Fraction(), -relation_inner),
    )
    safety = (
        max(Fraction(), -hard_outer),
        policy.safety_penalty_scale
        * max(Fraction(), policy.safety_constraint_slack_target - hard_inner),
    )
    semantic_terms = (
        ("T", *translation),
        ("A", *angle),
        ("R", *relation_damage),
        ("V", visibility.inner_fraction, visibility.outer_fraction),
        ("S", *safety),
    )
    weighted_terms = tuple(
        (
            term_id,
            lower * term.weight / term.normalizer,
            upper * term.weight / term.normalizer,
        )
        for (term_id, lower, upper), term in zip(
            semantic_terms,
            policy.objective_terms,
            strict=True,
        )
    )
    remaining = required_steps - (atomic_budget.used - start_used)
    if remaining < 0:
        raise RuntimeError("continuous yaw cell atomic preflight undercounted work")
    atomic_budget.consume(remaining)
    proof_rows = tuple(
        sorted(
            {
                f"CELL:{_continuous_cell_id_v4(cell, yaw_bounds)}",
                "SO2:CONTINUOUS_YAW:DELEGATED",
                collision_row,
                "SUPPORT:WORLD_XY_Z_UP:CONTINUOUS_ENVELOPE",
                f"RELATION:{relation}:{relation_measurement}:{policy.relation_comparator}",
                f"POLICY:{policy.policy_id}:{policy.policy_version}",
                "OBJECTIVE:CONTINUOUS_CELL:T-A-R-V-S",
                "SAFETY:HARD_CONSTRAINT_INNER_OUTER",
                *visibility.proof_rows,
            }
        )
    )
    return ContinuousYawCellOutcomeV4(
        so2_interval.ContinuousYawIntervalKindV4.EXACT,
        bounds=ContinuousYawCellBoundsV4(
            cell_id=_continuous_cell_id_v4(cell, yaw_bounds),
            cell=cell,
            yaw_bounds=yaw_bounds,
            subject_box_ids=tuple(value.box.box_id for value in subjects),
            obstacle_box_ids=tuple(value.box.box_id for value in obstacles),
            collision_constraint_present=bool(collision_pairs),
            collision_inner_slack=collision_inner,
            collision_outer_slack=collision_outer,
            collision_inward_proven=collision_inward,
            collision_outer_failure=collision_outer_failure,
            support_contact_gap=contact_gap,
            support_clearance_inner=support_inner,
            support_clearance_outer=support_outer,
            support_normal=(Fraction(), Fraction(), Fraction(1)),
            support_frame="WORLD_XY_Z_UP",
            closed_containment=True,
            stability_margin_inner=containment_inner - policy.support_stability_margin,
            relation_symbol=relation,
            relation_measurement=relation_measurement,
            relation_inner_slack=relation_inner,
            relation_outer_slack=relation_outer,
            relation_comparator=policy.relation_comparator,
            relation_threshold=policy.relation_threshold,
            relation_tolerance=policy.relation_tolerance,
            relation_boundary=policy.relation_boundary,
            relation_squared_distance_bounds=relation_squared,
            relation_inner_success=relation_inner_success,
            relation_outer_failure=relation_outer_failure,
            visibility_inner_fraction=visibility.inner_fraction,
            visibility_outer_fraction=visibility.outer_fraction,
            visibility_classification=visibility.classification.value,
            semantic_objective_terms=semantic_terms,
            weighted_objective_terms=weighted_terms,
            inner_hard_constraint_slack=hard_inner,
            outer_hard_constraint_slack=hard_outer,
            inner_hard_constraint_proven=(
                collision_inward and support_inner >= 0 and relation_inner_success
            ),
            outer_hard_constraint_failure=(
                collision_outer_failure or support_outer < 0 or relation_outer_failure
            ),
            safety_penalty_outer=safety[1],
        ),
        atomic_steps_used=atomic_budget.used - start_used,
        proof_rows=proof_rows,
    )


def _continuous_extent_intervals_v4(
    value: ContinuousYawBoxBoundsV4,
) -> _ContinuousExtentIntervalsV4:
    x_min = (
        value.center_x.rational_lower - value.x_radius.rational_upper,
        value.center_x.rational_upper - value.x_radius.rational_lower,
    )
    x_max = (
        value.center_x.rational_lower + value.x_radius.rational_lower,
        value.center_x.rational_upper + value.x_radius.rational_upper,
    )
    y_min = (
        value.center_y.rational_lower - value.y_radius.rational_upper,
        value.center_y.rational_upper - value.y_radius.rational_lower,
    )
    y_max = (
        value.center_y.rational_lower + value.y_radius.rational_lower,
        value.center_y.rational_upper + value.y_radius.rational_upper,
    )
    z_min = _enclosure_tuple(value.aabb_z)
    z_max = _enclosure_tuple(value.aabb_z)
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def _compound_continuous_extent_intervals_v4(
    values: tuple[ContinuousYawBoxBoundsV4, ...],
) -> _ContinuousExtentIntervalsV4:
    extents = tuple(_continuous_extent_intervals_v4(value) for value in values)
    return tuple(
        (
            min(value[index][0] for value in extents),
            min(value[index][1] for value in extents),
        )
        if index in (0, 2, 4)
        else (
            max(value[index][0] for value in extents),
            max(value[index][1] for value in extents),
        )
        for index in range(6)
    )  # type: ignore[return-value]


def _continuous_signed_pair_clearance_v4(
    left: ContinuousYawBoxBoundsV4,
    right: ContinuousYawBoxBoundsV4,
) -> tuple[Fraction, Fraction]:
    left_extent = _continuous_extent_intervals_v4(left)
    right_extent = _continuous_extent_intervals_v4(right)
    lower = max(
        right_extent[0][0] - left_extent[1][1],
        left_extent[0][0] - right_extent[1][1],
        right_extent[2][0] - left_extent[3][1],
        left_extent[2][0] - right_extent[3][1],
        right_extent[4][0] - left_extent[5][1],
        left_extent[4][0] - right_extent[5][1],
    )
    upper = max(
        right_extent[0][1] - left_extent[1][0],
        left_extent[0][1] - right_extent[1][0],
        right_extent[2][1] - left_extent[3][0],
        left_extent[2][1] - right_extent[3][0],
        right_extent[4][1] - left_extent[5][0],
        left_extent[4][1] - right_extent[5][0],
    )
    return lower, upper


def _continuous_support_containment_v4(
    extent: _ContinuousExtentIntervalsV4,
    surface: SupportSurfaceV3,
) -> tuple[Fraction, Fraction]:
    intervals = (
        (extent[0][0] - surface.x_lower, extent[0][1] - surface.x_lower),
        (surface.x_upper - extent[1][1], surface.x_upper - extent[1][0]),
        (extent[2][0] - surface.y_lower, extent[2][1] - surface.y_lower),
        (surface.y_upper - extent[3][1], surface.y_upper - extent[3][0]),
    )
    return min(value[0] for value in intervals), min(value[1] for value in intervals)


def _continuous_relation_slacks_v4(
    *,
    relation: str,
    subject_extent: _ContinuousExtentIntervalsV4,
    reference_extent: _ContinuousExtentIntervalsV4,
    pair_outcomes: tuple[ContinuousYawBoxPairOutcomeV4, ...],
    policy: FixedCardinalCellPolicyV3,
) -> tuple[Fraction, Fraction, str, tuple[Fraction, Fraction] | None]:
    if relation == "LEFT":
        raw = (
            subject_extent[1][0] - reference_extent[0][1],
            subject_extent[1][1] - reference_extent[0][0],
        )
        threshold = policy.relation_threshold + policy.relation_tolerance
        return (
            threshold - raw[1],
            threshold - raw[0],
            "EXTENT_AWARE_SIGNED_AXIS_GAP",
            None,
        )
    if relation == "RIGHT":
        raw = (
            subject_extent[0][0] - reference_extent[1][1],
            subject_extent[0][1] - reference_extent[1][0],
        )
        threshold = policy.relation_threshold - policy.relation_tolerance
        return (
            raw[0] - threshold,
            raw[1] - threshold,
            "EXTENT_AWARE_SIGNED_AXIS_GAP",
            None,
        )
    if relation == "FRONT":
        raw = (
            subject_extent[3][0] - reference_extent[2][1],
            subject_extent[3][1] - reference_extent[2][0],
        )
        threshold = policy.relation_threshold + policy.relation_tolerance
        return (
            threshold - raw[1],
            threshold - raw[0],
            "EXTENT_AWARE_SIGNED_AXIS_GAP",
            None,
        )
    if relation == "BEHIND":
        raw = (
            subject_extent[2][0] - reference_extent[3][1],
            subject_extent[2][1] - reference_extent[3][0],
        )
        threshold = policy.relation_threshold - policy.relation_tolerance
        return (
            raw[0] - threshold,
            raw[1] - threshold,
            "EXTENT_AWARE_SIGNED_AXIS_GAP",
            None,
        )
    if not pair_outcomes or any(pair.bounds is None for pair in pair_outcomes):
        raise RuntimeError("continuous near/far relation requires exact pair bounds")
    squared = (
        min(
            pair.bounds.squared_clearance.rational_lower
            for pair in pair_outcomes
            if pair.bounds is not None
        ),
        min(
            pair.bounds.squared_clearance.rational_upper
            for pair in pair_outcomes
            if pair.bounds is not None
        ),
    )
    if relation == "NEAR":
        threshold = policy.relation_threshold + policy.relation_tolerance
        threshold_squared = threshold * threshold
        return (
            threshold_squared - squared[1],
            threshold_squared - squared[0],
            "EXTENT_AWARE_EUCLIDEAN_SEPARATION",
            squared,
        )
    threshold = max(Fraction(), policy.relation_threshold - policy.relation_tolerance)
    threshold_squared = threshold * threshold
    return (
        squared[0] - threshold_squared,
        squared[1] - threshold_squared,
        "EXTENT_AWARE_EUCLIDEAN_SEPARATION",
        squared,
    )


def _continuous_translation_objective_v4(
    *,
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
    subject_pivot_xy: tuple[Fraction, Fraction],
    objective_subject_pivot_xy: tuple[Fraction, Fraction],
    atomic_budget: SO2AtomicBudgetV2,
) -> tuple[Fraction, Fraction]:
    relative_x = objective_subject_pivot_xy[0] - subject_pivot_xy[0]
    relative_y = objective_subject_pivot_xy[1] - subject_pivot_xy[1]
    cosine = _enclosure_tuple(yaw_bounds.cosine)
    sine = _enclosure_tuple(yaw_bounds.sine)
    negative_sine = _interval_negate(yaw_bounds.sine, atomic_budget)
    rotated_x = _interval_add(
        _interval_scale_signed(cosine, relative_x, atomic_budget),
        _interval_scale_signed(negative_sine, relative_y, atomic_budget),
        atomic_budget,
    )
    rotated_y = _interval_add(
        _interval_scale_signed(sine, relative_x, atomic_budget),
        _interval_scale_signed(cosine, relative_y, atomic_budget),
        atomic_budget,
    )
    orbital_x = _interval_subtract(
        _interval_add(
            rotated_x, (subject_pivot_xy[0], subject_pivot_xy[0]), atomic_budget
        ),
        (objective_subject_pivot_xy[0], objective_subject_pivot_xy[0]),
        atomic_budget,
    )
    orbital_y = _interval_subtract(
        _interval_add(
            rotated_y, (subject_pivot_xy[1], subject_pivot_xy[1]), atomic_budget
        ),
        (objective_subject_pivot_xy[1], objective_subject_pivot_xy[1]),
        atomic_budget,
    )
    displacement_x = _interval_add(
        orbital_x, (cell.x_lower, cell.x_upper), atomic_budget
    )
    displacement_y = _interval_add(
        orbital_y, (cell.y_lower, cell.y_upper), atomic_budget
    )
    lower = max(
        _axis_separation_v3(
            displacement_x[0], displacement_x[1], Fraction(), Fraction()
        ),
        _axis_separation_v3(
            displacement_y[0], displacement_y[1], Fraction(), Fraction()
        ),
    )
    upper = max(
        _absolute_fraction_v3(displacement_x[0]),
        _absolute_fraction_v3(displacement_x[1]),
    ) + max(
        _absolute_fraction_v3(displacement_y[0]),
        _absolute_fraction_v3(displacement_y[1]),
    )
    return lower, upper


def _continuous_shortest_turn_interval_v4(
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, Fraction]:
    candidates = [lower, upper]
    first_half = so2_interval._floor_fraction(2 * lower)
    last_half = so2_interval._floor_fraction(2 * upper)
    candidates.extend(
        Fraction(index, 2)
        for index in range(first_half, last_half + 1)
        if lower <= Fraction(index, 2) <= upper
    )
    values = tuple(_shortest_turn_distance_v4(value) for value in candidates)
    return min(values), max(values)


def _shortest_turn_distance_v4(value: Fraction) -> Fraction:
    floor = so2_interval._floor_fraction(value)
    return min(value - floor, floor + 1 - value)


def _continuous_cell_id_v4(
    cell: ClosedXYCellV3,
    yaw_bounds: so2_interval.ContinuousYawSinCosBoundsV4,
) -> str:
    xy = ",".join(
        f"{value.numerator}/{value.denominator}" for value in cell.canonical_bounds
    )
    lift = yaw_bounds.lift_bounds
    return (
        "continuous-yaw-cell:"
        f"xy={xy}:turn={lift.lower.numerator}/{lift.lower.denominator}:"
        f"{lift.upper.numerator}/{lift.upper.denominator}:lift={lift.coverage_sha256}"
    )


def _continuous_cell_from_visibility_nonexact_v4(
    visibility: object,
) -> ContinuousYawCellOutcomeV4:
    return ContinuousYawCellOutcomeV4(
        visibility.kind,  # type: ignore[union-attr]
        atomic_steps_used=0,
        proof_rows=visibility.proof_rows,  # type: ignore[union-attr]
        finding_codes=visibility.finding_codes,  # type: ignore[union-attr]
    )


def _continuous_cell_from_pair_nonexact_v4(
    values: tuple[ContinuousYawBoxPairOutcomeV4, ...],
    start_used: int,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawCellOutcomeV4:
    outcome = next(
        value
        for value in values
        if value.kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT
    )
    return _continuous_cell_failure_v4(
        outcome.kind,
        outcome.finding_codes[0],
        atomic_steps_used=atomic_budget.used - start_used,
        proof_rows=("PAIR:CONTINUOUS_NONEXACT:DELEGATED",),
    )


def _continuous_cell_failure_v4(
    kind: so2_interval.ContinuousYawIntervalKindV4,
    finding_code: str,
    *,
    atomic_steps_used: int,
    proof_rows: tuple[str, ...],
) -> ContinuousYawCellOutcomeV4:
    return ContinuousYawCellOutcomeV4(
        kind,
        atomic_steps_used=atomic_steps_used,
        proof_rows=proof_rows,
        finding_codes=(finding_code,),
    )


__all__ = (
    "ClosedXYCellV3",
    "ContinuousYawBoxBoundsV4",
    "ContinuousYawBoxContactKindV4",
    "ContinuousYawBoxOutcomeV4",
    "ContinuousYawBoxPairBoundsV4",
    "ContinuousYawBoxPairOutcomeV4",
    "ContinuousYawCellBoundsV4",
    "ContinuousYawCellOutcomeV4",
    "ContinuousYawCellPolicyV4",
    "FixedCardinalBoxV3",
    "FixedCardinalCellBoundsV3",
    "FixedCardinalCellOutcomeV3",
    "FixedCardinalCellPolicyV3",
    "FixedCardinalObjectiveTermV3",
    "OrientedAxisEnclosureV2",
    "OrientedAxisGapBoundsV2",
    "OrientedBoxContactKindV2",
    "OrientedUprightBoxBoundsV2",
    "OrientedUprightBoxOutcomeV2",
    "OrientedUprightBoxPairBoundsV2",
    "OrientedUprightBoxPairOutcomeV2",
    "SupportSurfaceV3",
    "compile_continuous_yaw_box_bounds_v4",
    "compile_oriented_upright_box_bounds_v2",
    "compile_oriented_upright_box_pair_bounds_v2",
    "evaluate_continuous_yaw_box_pair_bounds_v4",
    "evaluate_continuous_yaw_cell_v4",
    "evaluate_fixed_cardinal_cell_v3",
)
