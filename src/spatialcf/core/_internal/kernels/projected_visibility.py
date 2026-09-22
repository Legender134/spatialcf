"""Exact directed projected-bounding-box visibility bounds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels import upright_box as upright_box_interval
from spatialcf.core._internal.kernels.so2 import (
    CardinalKernelKindV3,
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.core.problem import (
    UprightCameraContextV2_9,
    bound_world_point_in_upright_camera,
)

_Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityPolicyV3:
    """Caller-bound visibility metric and shared-resource values."""

    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    atomic_step_limit: int

    def __post_init__(self) -> None:
        if (
            type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or type(self.metric_threshold) is not Fraction
            or not Fraction() <= self.metric_threshold <= Fraction(1)
            or type(self.metric_tolerance) is not Fraction
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.atomic_step_limit) is not int
            or self.atomic_step_limit <= 0
        ):
            raise ValueError(
                "fixed cardinal visibility policy must be a closed caller value"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalProjectionBoxV3:
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
                type(value) is not Fraction
                for value in (
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
            raise ValueError(
                "fixed cardinal projection box must be exact and non-negative"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityBoundsV3:
    cell: tuple[Fraction, Fraction, Fraction, Fraction]
    camera_context_sha256: str
    projection_convention: str
    occluder_roster: tuple[str, ...]
    subject_as_occluder: bool
    inner_fraction: Fraction
    outer_fraction: Fraction
    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    inner_success: bool
    outer_failure: bool
    gap_unknown: bool
    proof_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not tuple
            or len(self.cell) != 4
            or any(type(value) is not Fraction for value in self.cell)
            or self.cell[0] > self.cell[1]
            or self.cell[2] > self.cell[3]
            or type(self.camera_context_sha256) is not str
            or not self.camera_context_sha256
            or self.projection_convention != "RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9"
            or type(self.occluder_roster) is not tuple
            or not self.occluder_roster
            or any(
                type(value) is not str or not value for value in self.occluder_roster
            )
            or self.occluder_roster != tuple(sorted(set(self.occluder_roster)))
            or type(self.subject_as_occluder) is not bool
            or any(
                type(value) is not Fraction
                for value in (
                    self.inner_fraction,
                    self.outer_fraction,
                    self.metric_threshold,
                    self.metric_tolerance,
                )
            )
            or self.inner_fraction < 0
            or self.outer_fraction > 1
            or self.inner_fraction > self.outer_fraction
            or type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.inner_success) is not bool
            or type(self.outer_failure) is not bool
            or type(self.gap_unknown) is not bool
            or self.inner_success
            is not (
                self.inner_fraction >= self.metric_threshold - self.metric_tolerance
            )
            or self.outer_failure
            is not (self.outer_fraction < self.metric_threshold - self.metric_tolerance)
            or self.gap_unknown
            is not (not self.inner_success and not self.outer_failure)
            or type(self.proof_rows) is not tuple
            or not self.proof_rows
            or any(type(value) is not str or not value for value in self.proof_rows)
            or self.proof_rows != tuple(sorted(set(self.proof_rows)))
        ):
            raise ValueError(
                "fixed cardinal visibility bounds are not a closed exact record"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityOutcomeV3:
    kind: CardinalKernelKindV3
    bounds: FixedCardinalVisibilityBoundsV3 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CardinalKernelKindV3:
            raise TypeError(
                "fixed cardinal visibility outcome must use CardinalKernelKindV3"
            )
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError(
                "fixed cardinal visibility atomic usage must be non-negative"
            )
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"fixed cardinal visibility {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is CardinalKernelKindV3.EXACT:
            if (
                type(self.bounds) is not FixedCardinalVisibilityBoundsV3
                or self.finding_codes
                or not self.proof_rows
                or self.proof_rows != self.bounds.proof_rows
            ):
                raise ValueError(
                    "exact fixed cardinal visibility must carry its closed bounds"
                )
            return
        if self.bounds is not None or not self.proof_rows or not self.finding_codes:
            raise ValueError("non-exact fixed cardinal visibility must carry a finding")
        prefix = {
            CardinalKernelKindV3.NUMERIC_GAP: "NUMERIC_GAP:",
            CardinalKernelKindV3.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            CardinalKernelKindV3.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(prefix) for code in self.finding_codes):
            raise ValueError("visibility finding must match its typed outcome")


@dataclass(frozen=True, slots=True)
class _ProjectedRectangleV3:
    x_lower: Fraction
    x_upper: Fraction
    y_lower: Fraction
    y_upper: Fraction

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not Fraction
                for value in (self.x_lower, self.x_upper, self.y_lower, self.y_upper)
            )
            or self.x_lower > self.x_upper
            or self.y_lower > self.y_upper
        ):
            raise ValueError("projected rectangle must be ordered exact Fractions")

    @property
    def area(self) -> Fraction:
        return (self.x_upper - self.x_lower) * (self.y_upper - self.y_lower)


@dataclass(frozen=True, slots=True)
class _ProjectedBoxV3:
    inner: _ProjectedRectangleV3 | None
    outer: _ProjectedRectangleV3
    depth_lower: Fraction
    depth_upper: Fraction


def evaluate_fixed_cardinal_visibility_v3(
    *,
    context: UprightCameraContextV2_9,
    cell: tuple[Fraction, Fraction, Fraction, Fraction],
    subject: FixedCardinalProjectionBoxV3,
    moving_subject_id: str,
    occluders: tuple[FixedCardinalProjectionBoxV3, ...],
    required_occluder_ids: tuple[str, ...],
    policy: FixedCardinalVisibilityPolicyV3,
    atomic_budget: SO2AtomicBudgetV2,
) -> FixedCardinalVisibilityOutcomeV3:
    """Project exact cardinal box corners via the retained camera owner, never local axes."""
    checked_cell = _require_closed_cell_v3(cell)
    checked_subject = _require_projection_box_v3(subject, label="subject")
    checked_occluders = _require_complete_occluder_roster_v3(
        subject=checked_subject,
        occluders=occluders,
        required_occluder_ids=required_occluder_ids,
    )
    if type(context) is not UprightCameraContextV2_9:
        raise TypeError("context must be an UprightCameraContextV2_9")
    if type(moving_subject_id) is not str or moving_subject_id not in {
        box.box_id for box in checked_occluders
    }:
        raise ValueError("moving_subject_id must identify a complete-roster member")
    if type(policy) is not FixedCardinalVisibilityPolicyV3:
        raise TypeError("policy must be a FixedCardinalVisibilityPolicyV3")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if atomic_budget.limit != policy.atomic_step_limit:
        raise ValueError("atomic budget limit must match the caller policy")
    if (
        policy.metric_definition_id,
        policy.metric_definition_version,
    ) != ("visibility:image-area-fraction", "definition:1"):
        return FixedCardinalVisibilityOutcomeV3(
            CardinalKernelKindV3.UNSUPPORTED,
            atomic_steps_used=0,
            proof_rows=("UNSUPPORTED:VISIBILITY_METRIC:unimplemented-or-unknown",),
            finding_codes=("UNSUPPORTED:VISIBILITY_METRIC",),
        )
    required_steps = 80 * (1 + len(checked_occluders))
    if atomic_budget.remaining < required_steps:
        return _resource_limit_outcome_v3()
    initial_used = atomic_budget.used
    try:
        subject_projection = _project_box_v3(
            context=context,
            cell=(
                checked_cell
                if checked_subject.box_id == moving_subject_id
                else (Fraction(), Fraction(), Fraction(), Fraction())
            ),
            box=checked_subject,
            atomic_budget=atomic_budget,
        )
        if subject_projection is None:
            return _clipped_subject_outcome_v3(
                context=context,
                cell=checked_cell,
                roster=tuple(box.box_id for box in checked_occluders),
                policy=policy,
                atomic_steps_used=atomic_budget.used - initial_used,
            )
        occluder_projections = tuple(
            _project_box_v3(
                context=context,
                cell=(
                    checked_cell
                    if box.box_id == moving_subject_id
                    else (Fraction(), Fraction(), Fraction(), Fraction())
                ),
                box=box,
                atomic_budget=atomic_budget,
            )
            for box in checked_occluders
        )
    except SO2AtomicBudgetExhaustedV2:
        return _resource_limit_outcome_v3()
    except _VisibilityClipGapV3 as error:
        return FixedCardinalVisibilityOutcomeV3(
            CardinalKernelKindV3.NUMERIC_GAP,
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=(f"NUMERIC_GAP:{error.clip}:inner-outer",),
            finding_codes=(f"NUMERIC_GAP:{error.clip}",),
        )
    except ArithmeticError:
        return FixedCardinalVisibilityOutcomeV3(
            CardinalKernelKindV3.NUMERIC_GAP,
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=("NUMERIC_GAP:CAMERA_BOUND:directed-owner",),
            finding_codes=("NUMERIC_GAP:CAMERA_BOUND",),
        )
    if subject_projection.outer.area == 0:
        return FixedCardinalVisibilityOutcomeV3(
            CardinalKernelKindV3.UNSUPPORTED,
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=("UNSUPPORTED:DEGENERATE_PROJECTED_SUBJECT:zero-area",),
            finding_codes=("UNSUPPORTED:DEGENERATE_PROJECTED_SUBJECT",),
        )
    inner_fraction, outer_fraction = _visibility_fraction_bounds_v3(
        subject=subject_projection,
        subject_id=checked_subject.box_id,
        occluder_ids=tuple(box.box_id for box in checked_occluders),
        occluders=occluder_projections,
    )
    proof_rows = tuple(
        sorted(
            (
                f"CAMERA:{context.context_sha256}",
                "COMMON_CELL",
                "CLIP:NEAR_FAR:RETAINED_CONTEXT",
                f"OCCLUDERS:COMPLETE:{','.join(box.box_id for box in checked_occluders)}",
                "PROJECTION:RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
                "SUBJECT_AS_OCCLUDER",
                f"METRIC:{policy.metric_definition_id}:{policy.metric_definition_version}",
                (
                    "VISIBILITY:PROJECTED_AREA_FRACTION:"
                    f"{policy.metric_boundary}_{policy.metric_comparator}:"
                    f"{policy.metric_threshold.numerator}/{policy.metric_threshold.denominator}"
                ),
            )
        )
    )
    bounds = FixedCardinalVisibilityBoundsV3(
        cell=checked_cell,
        camera_context_sha256=context.context_sha256,
        projection_convention="RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
        occluder_roster=tuple(box.box_id for box in checked_occluders),
        subject_as_occluder=True,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        metric_definition_id=policy.metric_definition_id,
        metric_definition_version=policy.metric_definition_version,
        metric_threshold=policy.metric_threshold,
        metric_tolerance=policy.metric_tolerance,
        metric_comparator=policy.metric_comparator,
        metric_boundary=policy.metric_boundary,
        inner_success=inner_fraction
        >= policy.metric_threshold - policy.metric_tolerance,
        outer_failure=outer_fraction
        < policy.metric_threshold - policy.metric_tolerance,
        gap_unknown=(
            inner_fraction
            < policy.metric_threshold - policy.metric_tolerance
            <= outer_fraction
        ),
        proof_rows=proof_rows,
    )
    return FixedCardinalVisibilityOutcomeV3(
        CardinalKernelKindV3.EXACT,
        bounds,
        atomic_steps_used=atomic_budget.used - initial_used,
        proof_rows=proof_rows,
    )


class _VisibilityClipGapV3(ArithmeticError):
    def __init__(self, clip: str) -> None:
        self.clip = clip
        super().__init__(clip)


def _require_closed_cell_v3(
    value: object,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(endpoint) is not Fraction for endpoint in value)
    ):
        raise TypeError("cell must be an exact four-Fraction closed XY cell")
    x_lower, x_upper, y_lower, y_upper = value
    if x_lower > x_upper or y_lower > y_upper:
        raise ValueError("cell must have ordered closed XY bounds")
    return value


def _require_projection_box_v3(
    value: object, *, label: str
) -> FixedCardinalProjectionBoxV3:
    if type(value) is not FixedCardinalProjectionBoxV3:
        raise TypeError(f"{label} must be a FixedCardinalProjectionBoxV3")
    return value


def _require_complete_occluder_roster_v3(
    *,
    subject: FixedCardinalProjectionBoxV3,
    occluders: object,
    required_occluder_ids: object,
) -> tuple[FixedCardinalProjectionBoxV3, ...]:
    if type(occluders) is not tuple:
        raise TypeError("occluders must be an exact tuple")
    checked = tuple(
        _require_projection_box_v3(box, label="occluder") for box in occluders
    )
    roster_ids = tuple(box.box_id for box in checked)
    if roster_ids != tuple(sorted(set(roster_ids))):
        raise ValueError("complete occluder roster must be sorted with unique IDs")
    if (
        type(required_occluder_ids) is not tuple
        or any(
            type(box_id) is not str or not box_id for box_id in required_occluder_ids
        )
        or tuple(required_occluder_ids) != tuple(sorted(set(required_occluder_ids)))
        or tuple(required_occluder_ids) != roster_ids
    ):
        raise ValueError("complete occluder roster must match required ordered IDs")
    if subject.box_id not in roster_ids:
        raise ValueError("complete occluder roster must include the subject")
    roster_subject = checked[roster_ids.index(subject.box_id)]
    if roster_subject != subject:
        raise ValueError("subject-as-occluder must bind the exact subject geometry")
    return checked


def _resource_limit_outcome_v3() -> FixedCardinalVisibilityOutcomeV3:
    return FixedCardinalVisibilityOutcomeV3(
        CardinalKernelKindV3.RESOURCE_LIMIT,
        atomic_steps_used=0,
        proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:cap-minus-one",),
        finding_codes=("RESOURCE_LIMIT:SO2_ATOMIC_STEPS",),
    )


def _clamp_screen_interval_v3(
    value: tuple[Fraction, Fraction], *, limit: int
) -> tuple[Fraction, Fraction]:
    lower = min(Fraction(limit), max(Fraction(), value[0]))
    upper = min(Fraction(limit), max(Fraction(), value[1]))
    if lower > upper:
        raise RuntimeError("screen clipping reversed a projection interval")
    return lower, upper


def _divide_positive_interval_v3(
    numerator: tuple[Fraction, Fraction], denominator: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    if denominator[0] <= 0:
        raise _VisibilityClipGapV3("NEAR_CLIP")
    quotients = tuple(
        numerator_value / denominator_value
        for numerator_value in numerator
        for denominator_value in denominator
    )
    return min(quotients), max(quotients)


def _project_coordinate_v3(
    *,
    focal: Fraction,
    principal: Fraction,
    camera_axis: tuple[Fraction, Fraction],
    depth: tuple[Fraction, Fraction],
    screen_limit: int,
) -> tuple[Fraction, Fraction]:
    scaled_axis = tuple(focal * value for value in camera_axis)
    ratio = _divide_positive_interval_v3(scaled_axis, depth)
    return _clamp_screen_interval_v3(
        (ratio[0] + principal, ratio[1] + principal), limit=screen_limit
    )


def _inner_rectangle_v3(
    horizontal: tuple[tuple[Fraction, Fraction], ...],
    vertical: tuple[tuple[Fraction, Fraction], ...],
) -> _ProjectedRectangleV3 | None:
    # The projected bounding-box minimum is no greater than every upper
    # endpoint, while its maximum is no less than every lower endpoint.
    # These reversed extrema therefore form an inward rectangle for the
    # bounding-box metric (not a claim about the filled object silhouette).
    x_lower = min(value[1] for value in horizontal)
    x_upper = max(value[0] for value in horizontal)
    y_lower = min(value[1] for value in vertical)
    y_upper = max(value[0] for value in vertical)
    if x_lower >= x_upper or y_lower >= y_upper:
        return None
    return _ProjectedRectangleV3(x_lower, x_upper, y_lower, y_upper)


def _outer_rectangle_v3(
    horizontal: tuple[tuple[Fraction, Fraction], ...],
    vertical: tuple[tuple[Fraction, Fraction], ...],
) -> _ProjectedRectangleV3:
    return _ProjectedRectangleV3(
        min(value[0] for value in horizontal),
        max(value[1] for value in horizontal),
        min(value[0] for value in vertical),
        max(value[1] for value in vertical),
    )


def _project_box_v3(
    *,
    context: UprightCameraContextV2_9,
    cell: tuple[Fraction, Fraction, Fraction, Fraction],
    box: FixedCardinalProjectionBoxV3,
    atomic_budget: SO2AtomicBudgetV2,
) -> _ProjectedBoxV3 | None:
    """Use only the retained camera owner to bound every closed-cell box corner."""

    points = tuple(
        bound_world_point_in_upright_camera(
            context,
            world_xyz=(x, y, z),
            delta_x=(cell[0], cell[1]),
            delta_y=(cell[2], cell[3]),
            atomic_budget=atomic_budget,
        )
        for x in (box.center_x - box.half_x, box.center_x + box.half_x)
        for y in (box.center_y - box.half_y, box.center_y + box.half_y)
        for z in (box.center_z - box.half_z, box.center_z + box.half_z)
    )
    depth_lower = min(point.z_camera[0] for point in points)
    depth_upper = max(point.z_camera[1] for point in points)
    if depth_upper <= context.near_clip_m or depth_lower >= context.far_clip_m:
        return None
    if depth_lower <= context.near_clip_m < depth_upper:
        raise _VisibilityClipGapV3("NEAR_CLIP")
    if depth_lower < context.far_clip_m <= depth_upper:
        raise _VisibilityClipGapV3("FAR_CLIP")
    fx, _, cx, _, fy, cy, *_ = context.intrinsics
    horizontal = tuple(
        _project_coordinate_v3(
            focal=fx,
            principal=cx,
            camera_axis=point.x_camera,
            depth=point.z_camera,
            screen_limit=context.width_px,
        )
        for point in points
    )
    vertical = tuple(
        _project_coordinate_v3(
            focal=fy,
            principal=cy,
            camera_axis=point.y_camera,
            depth=point.z_camera,
            screen_limit=context.height_px,
        )
        for point in points
    )
    return _ProjectedBoxV3(
        inner=_inner_rectangle_v3(horizontal, vertical),
        outer=_outer_rectangle_v3(horizontal, vertical),
        depth_lower=depth_lower,
        depth_upper=depth_upper,
    )


def _intersection_area_v3(
    left: _ProjectedRectangleV3 | None, right: _ProjectedRectangleV3 | None
) -> Fraction:
    if left is None or right is None:
        return Fraction()
    width = min(left.x_upper, right.x_upper) - max(left.x_lower, right.x_lower)
    height = min(left.y_upper, right.y_upper) - max(left.y_lower, right.y_lower)
    return max(Fraction(), width) * max(Fraction(), height)


def _visibility_fraction_bounds_v3(
    *,
    subject: _ProjectedBoxV3,
    subject_id: str,
    occluder_ids: tuple[str, ...],
    occluders: tuple[_ProjectedBoxV3 | None, ...],
) -> tuple[Fraction, Fraction]:
    subject_inner_area = Fraction() if subject.inner is None else subject.inner.area
    subject_outer_area = subject.outer.area
    possible_cover = Fraction()
    certain_cover = Fraction()
    for occluder_id, occluder in zip(occluder_ids, occluders, strict=True):
        if occluder_id == subject_id or occluder is None:
            continue
        if occluder.depth_lower <= subject.depth_upper:
            possible_cover += _intersection_area_v3(subject.inner, occluder.outer)
        if occluder.depth_upper <= subject.depth_lower:
            certain_cover = max(
                certain_cover, _intersection_area_v3(subject.outer, occluder.inner)
            )
    inner_visible = max(Fraction(), subject_inner_area - possible_cover)
    inner_fraction = inner_visible / subject_outer_area
    if subject_inner_area == 0:
        return inner_fraction, Fraction(1)
    outer_visible = max(Fraction(), subject_outer_area - certain_cover)
    outer_fraction = min(Fraction(1), outer_visible / subject_inner_area)
    if inner_fraction > outer_fraction:
        raise RuntimeError("visibility fraction enclosure is reversed")
    return inner_fraction, outer_fraction


def _clipped_subject_outcome_v3(
    *,
    context: UprightCameraContextV2_9,
    cell: tuple[Fraction, Fraction, Fraction, Fraction],
    roster: tuple[str, ...],
    policy: FixedCardinalVisibilityPolicyV3,
    atomic_steps_used: int,
) -> FixedCardinalVisibilityOutcomeV3:
    proof_rows = tuple(
        sorted(
            (
                f"CAMERA:{context.context_sha256}",
                "CLIP:OUTSIDE_NEAR_FAR",
                "COMMON_CELL",
                f"OCCLUDERS:COMPLETE:{','.join(roster)}",
                "PROJECTION:RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
                "SUBJECT_AS_OCCLUDER",
                f"METRIC:{policy.metric_definition_id}:{policy.metric_definition_version}",
                (
                    "VISIBILITY:PROJECTED_AREA_FRACTION:"
                    f"{policy.metric_boundary}_{policy.metric_comparator}:"
                    f"{policy.metric_threshold.numerator}/{policy.metric_threshold.denominator}"
                ),
            )
        )
    )
    bounds = FixedCardinalVisibilityBoundsV3(
        cell=cell,
        camera_context_sha256=context.context_sha256,
        projection_convention="RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
        occluder_roster=roster,
        subject_as_occluder=True,
        inner_fraction=Fraction(),
        outer_fraction=Fraction(),
        metric_definition_id=policy.metric_definition_id,
        metric_definition_version=policy.metric_definition_version,
        metric_threshold=policy.metric_threshold,
        metric_tolerance=policy.metric_tolerance,
        metric_comparator=policy.metric_comparator,
        metric_boundary=policy.metric_boundary,
        inner_success=Fraction() >= policy.metric_threshold - policy.metric_tolerance,
        outer_failure=Fraction() < policy.metric_threshold - policy.metric_tolerance,
        gap_unknown=False,
        proof_rows=proof_rows,
    )
    return FixedCardinalVisibilityOutcomeV3(
        CardinalKernelKindV3.EXACT,
        bounds,
        atomic_steps_used=atomic_steps_used,
        proof_rows=proof_rows,
    )


def projected_bounding_box_area_fraction_lower_bound_v2_9(
    *,
    projected_u: tuple[_Interval, ...],
    projected_v: tuple[_Interval, ...],
    image_width_px: int,
    image_height_px: int,
) -> Fraction:
    """Return a sound lower bound for one projected bounding-box area.

    Every directed coordinate ``U_i`` lies in ``[l_i, u_i]``.  Therefore
    ``max(U) >= max(l_i)`` and ``min(U) <= min(u_i)`` for every realization,
    so their difference is bounded below by ``max(l_i) - min(u_i)``.  The
    same independent argument applies to V; both spans are nonnegative.
    """

    _require_intervals(projected_u, "projected_u")
    _require_intervals(projected_v, "projected_v")
    if type(image_width_px) is not int or type(image_height_px) is not int:
        raise TypeError("image dimensions must be exact integers")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image dimensions must be positive")

    width_lower = max(
        Fraction(),
        max(item[0] for item in projected_u) - min(item[1] for item in projected_u),
    )
    height_lower = max(
        Fraction(),
        max(item[0] for item in projected_v) - min(item[1] for item in projected_v),
    )
    return width_lower * height_lower / Fraction(image_width_px * image_height_px)


class ContinuousYawVisibilityClassificationV4(StrEnum):
    """The only whole-pose-cell visibility dispositions this owner can prove."""

    INWARD = "INWARD"
    OUTWARD = "OUTWARD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityBoundsV4:
    """Camera-owner bounds for a full closed XY-times-lifted-yaw pose cell."""

    cell: tuple[Fraction, Fraction, Fraction, Fraction]
    lifted_turn_bounds: tuple[Fraction, Fraction]
    continuous_yaw_lift_sha256: str
    camera_context_sha256: str
    projection_convention: str
    subject_box_id: str
    occluder_roster: tuple[str, ...]
    subject_as_occluder: bool
    inner_fraction: Fraction
    outer_fraction: Fraction
    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    classification: ContinuousYawVisibilityClassificationV4
    outer_subject_rectangle: tuple[Fraction, Fraction, Fraction, Fraction] | None
    subject_depth_bounds: tuple[Fraction, Fraction] | None
    proof_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not tuple
            or len(self.cell) != 4
            or any(type(value) is not Fraction for value in self.cell)
            or self.cell[0] > self.cell[1]
            or self.cell[2] > self.cell[3]
            or type(self.lifted_turn_bounds) is not tuple
            or len(self.lifted_turn_bounds) != 2
            or any(type(value) is not Fraction for value in self.lifted_turn_bounds)
            or self.lifted_turn_bounds[0] > self.lifted_turn_bounds[1]
            or type(self.continuous_yaw_lift_sha256) is not str
            or len(self.continuous_yaw_lift_sha256) != 64
            or type(self.camera_context_sha256) is not str
            or not self.camera_context_sha256
            or self.projection_convention != "RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9"
            or type(self.subject_box_id) is not str
            or not self.subject_box_id
            or type(self.occluder_roster) is not tuple
            or not self.occluder_roster
            or self.occluder_roster != tuple(sorted(set(self.occluder_roster)))
            or self.subject_box_id not in self.occluder_roster
            or type(self.subject_as_occluder) is not bool
            or not self.subject_as_occluder
            or any(
                type(value) is not Fraction
                for value in (
                    self.inner_fraction,
                    self.outer_fraction,
                    self.metric_threshold,
                    self.metric_tolerance,
                )
            )
            or self.inner_fraction < 0
            or self.outer_fraction > 1
            or self.inner_fraction > self.outer_fraction
            or type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.classification) is not ContinuousYawVisibilityClassificationV4
            or type(self.proof_rows) is not tuple
            or not self.proof_rows
            or any(type(value) is not str or not value for value in self.proof_rows)
            or self.proof_rows != tuple(sorted(set(self.proof_rows)))
        ):
            raise ValueError(
                "continuous visibility bounds are not a closed exact record"
            )
        int(self.continuous_yaw_lift_sha256, 16)
        expected_classification = _continuous_visibility_classification_v4(
            self.inner_fraction,
            self.outer_fraction,
            self.metric_threshold,
            self.metric_tolerance,
        )
        if self.classification is not expected_classification:
            raise ValueError(
                "continuous visibility classification does not match bounds"
            )
        for value in (self.outer_subject_rectangle, self.subject_depth_bounds):
            if value is None:
                continue
            if type(value) is not tuple or any(
                type(item) is not Fraction for item in value
            ):
                raise TypeError(
                    "continuous projection endpoints must be exact Fractions"
                )
        if self.outer_subject_rectangle is not None and (
            len(self.outer_subject_rectangle) != 4
            or self.outer_subject_rectangle[0] > self.outer_subject_rectangle[1]
            or self.outer_subject_rectangle[2] > self.outer_subject_rectangle[3]
        ):
            raise ValueError("continuous outer subject rectangle must be ordered")
        if self.subject_depth_bounds is not None and (
            len(self.subject_depth_bounds) != 2
            or self.subject_depth_bounds[0] > self.subject_depth_bounds[1]
        ):
            raise ValueError("continuous subject depth bounds must be ordered")


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityOutcomeV4:
    """Typed completion of one continuous pose-cell camera evaluation."""

    kind: so2_interval.ContinuousYawIntervalKindV4
    bounds: ContinuousYawVisibilityBoundsV4 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not so2_interval.ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError("continuous visibility atomic usage must be non-negative")
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"continuous visibility {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is so2_interval.ContinuousYawIntervalKindV4.EXACT:
            if (
                type(self.bounds) is not ContinuousYawVisibilityBoundsV4
                or self.finding_codes
                or not self.proof_rows
                or self.proof_rows != self.bounds.proof_rows
            ):
                raise ValueError("exact continuous visibility must carry closed bounds")
            return
        if self.bounds is not None or not self.proof_rows or not self.finding_codes:
            raise ValueError("non-exact continuous visibility must carry a finding")
        expected_prefix = {
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("continuous visibility finding must match outcome kind")


def evaluate_continuous_yaw_visibility_v4(
    *,
    context: UprightCameraContextV2_9,
    subject: upright_box_interval.ContinuousYawBoxBoundsV4,
    moving_subject_id: str,
    occluders: tuple[upright_box_interval.ContinuousYawBoxBoundsV4, ...],
    required_occluder_ids: tuple[str, ...],
    policy: FixedCardinalVisibilityPolicyV3,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawVisibilityOutcomeV4:
    """Project a complete continuous pose cell through the retained camera owner.

    Cardinal points retain the v3 path. Other point poses bound the true rotated
    corners for an inward projected-bounding-box visibility bound. Nonpoint
    poses retain the outer-envelope path and explicit visibility ambiguity.
    """

    if type(context) is not UprightCameraContextV2_9:
        raise TypeError("context must be an UprightCameraContextV2_9")
    if type(subject) is not upright_box_interval.ContinuousYawBoxBoundsV4:
        raise TypeError("subject must be a ContinuousYawBoxBoundsV4")
    checked_occluders = _require_complete_continuous_occluder_roster_v4(
        subject=subject,
        occluders=occluders,
        required_occluder_ids=required_occluder_ids,
    )
    if type(moving_subject_id) is not str or moving_subject_id not in {
        value.box.box_id for value in checked_occluders
    }:
        raise ValueError("moving_subject_id must identify a complete-roster member")
    if type(policy) is not FixedCardinalVisibilityPolicyV3:
        raise TypeError("policy must be a FixedCardinalVisibilityPolicyV3")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    if atomic_budget.limit != policy.atomic_step_limit:
        raise ValueError("atomic budget limit must match the caller policy")
    if (
        policy.metric_definition_id,
        policy.metric_definition_version,
    ) != ("visibility:image-area-fraction", "definition:1"):
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED,
            "UNSUPPORTED:VISIBILITY_METRIC",
            atomic_steps_used=0,
            proof_rows=("UNSUPPORTED:VISIBILITY_METRIC:unimplemented-or-unknown",),
        )
    if _all_continuous_boxes_are_cardinal_points_v4((subject, *checked_occluders)):
        return _delegate_cardinal_point_visibility_v4(
            context=context,
            subject=subject,
            moving_subject_id=moving_subject_id,
            occluders=checked_occluders,
            required_occluder_ids=required_occluder_ids,
            policy=policy,
            atomic_budget=atomic_budget,
        )
    point_subject = (
        subject.cell.x_lower == subject.cell.x_upper
        and subject.cell.y_lower == subject.cell.y_upper
        and subject.yaw_bounds.lift_bounds.lower == subject.yaw_bounds.lift_bounds.upper
    )
    required_steps = 80 * (1 + len(checked_occluders)) + (33 if point_subject else 0)
    if atomic_budget.remaining < required_steps:
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
            atomic_steps_used=0,
            proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:cap-minus-one",),
        )
    initial_used = atomic_budget.used
    point_inner_fraction = Fraction()
    try:
        subject_projection = (
            _project_continuous_point_box_v4(
                context=context, box=subject, atomic_budget=atomic_budget,
            )
            if point_subject else _project_box_v3(
                context=context,
                cell=(Fraction(), Fraction(), Fraction(), Fraction()),
                box=_continuous_outer_projection_box_v4(subject),
                atomic_budget=atomic_budget,
            )
        )
        occluder_projections = tuple(
            _project_box_v3(
                context=context,
                cell=(Fraction(), Fraction(), Fraction(), Fraction()),
                box=_continuous_outer_projection_box_v4(value),
                atomic_budget=atomic_budget,
            )
            for value in checked_occluders
        )
        if point_subject and subject_projection is not None and subject_projection.outer.area > 0:
            # Outer occluder envelopes establish possible cover only. Their
            # inner rectangles are not inward bounds for actual rotated boxes.
            inner_area = (
                Fraction() if subject_projection.inner is None
                else subject_projection.inner.area
            )
            possible_cover = sum(
                (_intersection_area_v3(subject_projection.inner, projection.outer)
                 for value, projection in zip(checked_occluders, occluder_projections, strict=True)
                 if value.box.box_id != subject.box.box_id and projection is not None
                 and projection.depth_lower <= subject_projection.depth_upper),
                Fraction(),
            )
            raw_inner_fraction = max(Fraction(), inner_area - possible_cover) / subject_projection.outer.area
            atomic_budget.consume()
            # Directed publication keeps the lower bound conservative and avoids
            # transporting a product of large exact trigonometric denominators.
            point_inner_fraction = Fraction.from_float(
                so2_interval._fraction_floor_binary64(raw_inner_fraction)
            )
    except SO2AtomicBudgetExhaustedV2:
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SO2_ATOMIC_STEPS",
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=("RESOURCE:SO2_ATOMIC_STEPS:continuous-outer-envelope",),
        )
    except _VisibilityClipGapV3 as error:
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP,
            f"NUMERIC_GAP:{error.clip}",
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=(f"NUMERIC_GAP:{error.clip}:continuous-outer-envelope",),
        )
    except ArithmeticError:
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP,
            "NUMERIC_GAP:CAMERA_BOUND",
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=("NUMERIC_GAP:CAMERA_BOUND:directed-owner",),
        )
    if subject_projection is not None and subject_projection.outer.area == 0:
        return _continuous_visibility_failure_v4(
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED,
            "UNSUPPORTED:DEGENERATE_PROJECTED_SUBJECT",
            atomic_steps_used=atomic_budget.used - initial_used,
            proof_rows=("UNSUPPORTED:DEGENERATE_PROJECTED_SUBJECT:zero-area",),
        )
    if subject_projection is None:
        inner_fraction = Fraction()
        outer_fraction = Fraction()
        rectangle = None
        depth = None
        classification_row = "CLASSIFICATION:OUTER_CAMERA_CLIPPED"
    else:
        # The envelope can contain every pose, but does not provide a common
        # visible silhouette or a complete occlusion proof for a nonpoint cell.
        inner_fraction = Fraction()
        outer_fraction = Fraction(1)
        rectangle = (
            subject_projection.outer.x_lower,
            subject_projection.outer.x_upper,
            subject_projection.outer.y_lower,
            subject_projection.outer.y_upper,
        )
        depth = (subject_projection.depth_lower, subject_projection.depth_upper)
        classification_row = "CLASSIFICATION:NONDEGENERATE_OUTER_ENVELOPE_UNKNOWN"
        if point_subject:
            inner_fraction = point_inner_fraction
            classification_row = "CLASSIFICATION:POINT_DIRECTED_CORNER_BOUND:BINARY64_LOWER"
    proof_rows = tuple(
        sorted(
            (
                f"CAMERA:{context.context_sha256}",
                "CONTINUOUS_POSE_CELL:OUTER_ENVELOPE",
                f"OCCLUDERS:COMPLETE:{','.join(value.box.box_id for value in checked_occluders)}",
                "PROJECTION:RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
                "SUBJECT_AS_OCCLUDER",
                f"METRIC:{policy.metric_definition_id}:{policy.metric_definition_version}",
                classification_row,
                f"PROJECTED_OBJECTS:{1 + len(occluder_projections)}",
            )
        )
    )
    bounds = _continuous_visibility_bounds_v4(
        context=context,
        subject=subject,
        occluders=checked_occluders,
        policy=policy,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        rectangle=rectangle,
        depth=depth,
        proof_rows=proof_rows,
    )
    return ContinuousYawVisibilityOutcomeV4(
        so2_interval.ContinuousYawIntervalKindV4.EXACT,
        bounds=bounds,
        atomic_steps_used=atomic_budget.used - initial_used,
        proof_rows=proof_rows,
    )


def _project_continuous_point_box_v4(
    *, context: UprightCameraContextV2_9,
    box: upright_box_interval.ContinuousYawBoxBoundsV4,
    atomic_budget: SO2AtomicBudgetV2,
) -> _ProjectedBoxV3 | None:
    """Bound actual rotated corners, never the corners of an outer world AABB.

    Independent coordinate intervals may overestimate each corner. Reversed
    projected extrema nevertheless enclose a common inner bounding rectangle.
    This proves the retained bounding-box metric, not a filled silhouette.
    """
    def checked(value: _Interval) -> _Interval:
        for endpoint in value:
            so2_interval._require_numeric_fraction_cap(
                endpoint, "NUMERIC_GAP:CONTINUOUS_VISIBILITY_CORNER_FRACTION_BIT_CAP",
            )
        return value

    def scale(value: _Interval, factor: Fraction) -> _Interval:
        atomic_budget.consume()
        checked(value)
        checked((factor, factor))
        products = (value[0] * factor, value[1] * factor)
        return checked((min(products), max(products)))

    def add(left: _Interval, right: _Interval) -> _Interval:
        atomic_budget.consume()
        checked(left)
        checked(right)
        return checked((left[0] + right[0], left[1] + right[1]))

    def endpoints(value: so2_interval.RationalEnclosureV2) -> _Interval:
        return (value.rational_lower, value.rational_upper)

    xy = tuple(
        (
            add(endpoints(box.center_x), add(
                scale(endpoints(box.local_x_axis.x), x),
                scale(endpoints(box.local_y_axis.x), y),
            )),
            add(endpoints(box.center_y), add(
                scale(endpoints(box.local_x_axis.y), x),
                scale(endpoints(box.local_y_axis.y), y),
            )),
        )
        for x in (-box.half_x, box.half_x)
        for y in (-box.half_y, box.half_y)
    )
    points = tuple(
        bound_world_point_in_upright_camera(
            context, world_xyz=(Fraction(), Fraction(), z),
            delta_x=x, delta_y=y, atomic_budget=atomic_budget,
        )
        for x, y in xy for z in (box.center_z - box.half_z, box.center_z + box.half_z)
    )
    depth_lower = min(point.z_camera[0] for point in points)
    depth_upper = max(point.z_camera[1] for point in points)
    if depth_upper <= context.near_clip_m or depth_lower >= context.far_clip_m:
        return None
    if depth_lower <= context.near_clip_m < depth_upper:
        raise _VisibilityClipGapV3("NEAR_CLIP")
    if depth_lower < context.far_clip_m <= depth_upper:
        raise _VisibilityClipGapV3("FAR_CLIP")
    fx, _, cx, _, fy, cy, *_ = context.intrinsics
    horizontal = tuple(
        _project_coordinate_v3(focal=fx, principal=cx, camera_axis=point.x_camera,
                              depth=point.z_camera, screen_limit=context.width_px)
        for point in points
    )
    vertical = tuple(
        _project_coordinate_v3(focal=fy, principal=cy, camera_axis=point.y_camera,
                              depth=point.z_camera, screen_limit=context.height_px)
        for point in points
    )
    return _ProjectedBoxV3(
        inner=_inner_rectangle_v3(horizontal, vertical),
        outer=_outer_rectangle_v3(horizontal, vertical),
        depth_lower=depth_lower, depth_upper=depth_upper,
    )


def _delegate_cardinal_point_visibility_v4(
    *,
    context: UprightCameraContextV2_9,
    subject: upright_box_interval.ContinuousYawBoxBoundsV4,
    moving_subject_id: str,
    occluders: tuple[upright_box_interval.ContinuousYawBoxBoundsV4, ...],
    required_occluder_ids: tuple[str, ...],
    policy: FixedCardinalVisibilityPolicyV3,
    atomic_budget: SO2AtomicBudgetV2,
) -> ContinuousYawVisibilityOutcomeV4:
    """Reuse the Task 3A point camera path exactly at cardinal pose points."""

    retained = evaluate_fixed_cardinal_visibility_v3(
        context=context,
        cell=(Fraction(), Fraction(), Fraction(), Fraction()),
        subject=_continuous_outer_projection_box_v4(subject),
        moving_subject_id=moving_subject_id,
        occluders=tuple(
            _continuous_outer_projection_box_v4(value) for value in occluders
        ),
        required_occluder_ids=required_occluder_ids,
        policy=policy,
        atomic_budget=atomic_budget,
    )
    kind = so2_interval.ContinuousYawIntervalKindV4(retained.kind.value)
    if kind is not so2_interval.ContinuousYawIntervalKindV4.EXACT:
        return ContinuousYawVisibilityOutcomeV4(
            kind,
            atomic_steps_used=retained.atomic_steps_used,
            proof_rows=retained.proof_rows,
            finding_codes=retained.finding_codes,
        )
    if type(retained.bounds) is not FixedCardinalVisibilityBoundsV3:
        raise RuntimeError("exact retained point visibility is missing bounds")
    bounds = _continuous_visibility_bounds_v4(
        context=context,
        subject=subject,
        occluders=occluders,
        policy=policy,
        inner_fraction=retained.bounds.inner_fraction,
        outer_fraction=retained.bounds.outer_fraction,
        rectangle=None,
        depth=None,
        proof_rows=retained.proof_rows,
    )
    return ContinuousYawVisibilityOutcomeV4(
        kind,
        bounds=bounds,
        atomic_steps_used=retained.atomic_steps_used,
        proof_rows=retained.proof_rows,
    )


def _continuous_visibility_bounds_v4(
    *,
    context: UprightCameraContextV2_9,
    subject: upright_box_interval.ContinuousYawBoxBoundsV4,
    occluders: tuple[upright_box_interval.ContinuousYawBoxBoundsV4, ...],
    policy: FixedCardinalVisibilityPolicyV3,
    inner_fraction: Fraction,
    outer_fraction: Fraction,
    rectangle: tuple[Fraction, Fraction, Fraction, Fraction] | None,
    depth: tuple[Fraction, Fraction] | None,
    proof_rows: tuple[str, ...],
) -> ContinuousYawVisibilityBoundsV4:
    lift = subject.yaw_bounds.lift_bounds
    return ContinuousYawVisibilityBoundsV4(
        cell=subject.cell.canonical_bounds,
        lifted_turn_bounds=(lift.lower, lift.upper),
        continuous_yaw_lift_sha256=lift.coverage_sha256,
        camera_context_sha256=context.context_sha256,
        projection_convention="RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9",
        subject_box_id=subject.box.box_id,
        occluder_roster=tuple(value.box.box_id for value in occluders),
        subject_as_occluder=True,
        inner_fraction=inner_fraction,
        outer_fraction=outer_fraction,
        metric_definition_id=policy.metric_definition_id,
        metric_definition_version=policy.metric_definition_version,
        metric_threshold=policy.metric_threshold,
        metric_tolerance=policy.metric_tolerance,
        metric_comparator=policy.metric_comparator,
        metric_boundary=policy.metric_boundary,
        classification=_continuous_visibility_classification_v4(
            inner_fraction,
            outer_fraction,
            policy.metric_threshold,
            policy.metric_tolerance,
        ),
        outer_subject_rectangle=rectangle,
        subject_depth_bounds=depth,
        proof_rows=proof_rows,
    )


def _continuous_visibility_classification_v4(
    inner_fraction: Fraction,
    outer_fraction: Fraction,
    threshold: Fraction,
    tolerance: Fraction,
) -> ContinuousYawVisibilityClassificationV4:
    required = threshold - tolerance
    if inner_fraction >= required:
        return ContinuousYawVisibilityClassificationV4.INWARD
    if outer_fraction < required:
        return ContinuousYawVisibilityClassificationV4.OUTWARD
    return ContinuousYawVisibilityClassificationV4.UNKNOWN


def _require_complete_continuous_occluder_roster_v4(
    *,
    subject: upright_box_interval.ContinuousYawBoxBoundsV4,
    occluders: object,
    required_occluder_ids: object,
) -> tuple[upright_box_interval.ContinuousYawBoxBoundsV4, ...]:
    if type(occluders) is not tuple:
        raise TypeError("occluders must be an exact tuple")
    checked = tuple(
        _require_continuous_box_v4(value, label="occluder") for value in occluders
    )
    roster_ids = tuple(value.box.box_id for value in checked)
    if roster_ids != tuple(sorted(set(roster_ids))):
        raise ValueError("complete occluder roster must be sorted with unique IDs")
    if (
        type(required_occluder_ids) is not tuple
        or any(type(value) is not str or not value for value in required_occluder_ids)
        or tuple(required_occluder_ids) != tuple(sorted(set(required_occluder_ids)))
        or tuple(required_occluder_ids) != roster_ids
    ):
        raise ValueError("complete occluder roster must match required ordered IDs")
    if subject.box.box_id not in roster_ids:
        raise ValueError("complete occluder roster must include the subject")
    if checked[roster_ids.index(subject.box.box_id)] != subject:
        raise ValueError("subject-as-occluder must bind the exact continuous geometry")
    return checked


def _require_continuous_box_v4(
    value: object, *, label: str
) -> upright_box_interval.ContinuousYawBoxBoundsV4:
    if type(value) is not upright_box_interval.ContinuousYawBoxBoundsV4:
        raise TypeError(f"{label} must be a ContinuousYawBoxBoundsV4")
    return value


def _all_continuous_boxes_are_cardinal_points_v4(
    boxes: tuple[upright_box_interval.ContinuousYawBoxBoundsV4, ...],
) -> bool:
    return all(
        value.cell.x_lower == value.cell.x_upper
        and value.cell.y_lower == value.cell.y_upper
        and value.yaw_bounds.lift_bounds.lower == value.yaw_bounds.lift_bounds.upper
        and so2_interval._cardinal_quarter_turn_v4(value.yaw_bounds.lift_bounds.lower)
        is not None
        for value in boxes
    )


def _continuous_outer_projection_box_v4(
    value: upright_box_interval.ContinuousYawBoxBoundsV4,
) -> FixedCardinalProjectionBoxV3:
    x_lower, x_upper = (
        value.aabb_x.rational_lower,
        value.aabb_x.rational_upper,
    )
    y_lower, y_upper = (
        value.aabb_y.rational_lower,
        value.aabb_y.rational_upper,
    )
    z_lower, z_upper = (
        value.aabb_z.rational_lower,
        value.aabb_z.rational_upper,
    )
    return FixedCardinalProjectionBoxV3(
        value.box.box_id,
        (x_lower + x_upper) / 2,
        (y_lower + y_upper) / 2,
        (z_lower + z_upper) / 2,
        (x_upper - x_lower) / 2,
        (y_upper - y_lower) / 2,
        (z_upper - z_lower) / 2,
    )


def _continuous_visibility_failure_v4(
    kind: so2_interval.ContinuousYawIntervalKindV4,
    finding_code: str,
    *,
    atomic_steps_used: int,
    proof_rows: tuple[str, ...],
) -> ContinuousYawVisibilityOutcomeV4:
    return ContinuousYawVisibilityOutcomeV4(
        kind,
        atomic_steps_used=atomic_steps_used,
        proof_rows=proof_rows,
        finding_codes=(finding_code,),
    )


def _require_intervals(value: object, label: str) -> None:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a non-empty exact tuple")
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(endpoint) is not Fraction for endpoint in item)
        ):
            raise TypeError(f"{label} entries must be exact Fraction intervals")
        if item[0] > item[1]:
            raise ValueError(f"{label} interval endpoints are reversed")


__all__ = (
    "ContinuousYawVisibilityBoundsV4",
    "ContinuousYawVisibilityClassificationV4",
    "ContinuousYawVisibilityOutcomeV4",
    "FixedCardinalProjectionBoxV3",
    "FixedCardinalVisibilityBoundsV3",
    "FixedCardinalVisibilityOutcomeV3",
    "FixedCardinalVisibilityPolicyV3",
    "evaluate_continuous_yaw_visibility_v4",
    "evaluate_fixed_cardinal_visibility_v3",
    "projected_bounding_box_area_fraction_lower_bound_v2_9",
)
