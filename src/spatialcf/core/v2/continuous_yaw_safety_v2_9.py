"""Safety-margin bounds for the competition continuous-yaw v2.9 chain."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    compile_upright_camera_context_v2_9,
)
from spatialcf.core.v2.continuous_yaw_relation_damage import (
    ContinuousYawRelationDamageStageV2_9,
    RelationDamageCellV2_9,
    _distance_squared,
    _linear_measurement,
    _relation_context,
)
from spatialcf.core.v2.continuous_yaw_visibility_v2_9 import (
    ContinuousYawVisibilityStageV2_9,
)
from spatialcf.core.v2.convex_translation_domain import RationalPoint2V2
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _require_finding_codes,
)
from spatialcf.core.v2.objective_numeric import (
    ConstraintSafetyInputV2,
    ExactPublishedIntervalV2,
    ObjectiveNumericKindV2,
    SafetyRawComponentIntervalV2,
    _directed_sqrt_binary64_bounds,
    _fraction_to_float_ceil,
    _fraction_to_float_floor,
    aggregate_safety_penalty_bounds_v2,
    compile_constraint_slack_bounds_v2,
)
from spatialcf.core.v2.objective_safety_bounds import ConstraintSlackBoundsV2
from spatialcf.core.v2.oriented_upright_box import (
    compile_oriented_upright_box_bounds_v2,
    compile_oriented_upright_box_pair_bounds_v2,
)
from spatialcf.core.v2.rect_kernel import ExactAxisAlignedRectV2
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
)
from spatialcf.domain.v2.artifacts import ConstraintSlackV2, NonNegativeIntervalV2
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    CollisionClearanceMetricV2,
    MeasurementComparatorV2,
    PositionRegionInterpretationV2,
    RegionAggregationV2,
    RelationAxisV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import GeometryInstanceV2_2
from spatialcf.domain.v2.geometry import UprightBox3DV2
from spatialcf.domain.v2.objective import (
    PairAxisKeyV2,
    SafetyComponentKindV2,
    SafetySlackUnitV2,
)
from spatialcf.domain.v2.scene import (
    GeometryApproximationV2,
    GeometryRoleV2,
    RegionBoundaryPolicyV2,
    UncertaintyBudgetV2,
    Vec3V2,
)

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-safety.v2.9\0"
_IntervalV2 = tuple[Fraction, Fraction]


class ContinuousYawSafetyKindV2_9(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class ContinuousYawSafetyCellV2_9:
    source_cell_id: str
    constraint_bounds: tuple[ConstraintSlackBoundsV2, ...]
    constraint_slacks: tuple[ConstraintSlackV2, ...]
    safety_loss: NonNegativeIntervalV2

    def __post_init__(self) -> None:
        if type(self.source_cell_id) is not str or not self.source_cell_id:
            raise TypeError("source_cell_id must be a non-empty exact string")
        if type(self.constraint_bounds) is not tuple or not self.constraint_bounds:
            raise TypeError("constraint_bounds must be a non-empty exact tuple")
        checked_bounds = tuple(
            ConstraintSlackBoundsV2(
                constraint_id=item.constraint_id,
                raw_components=item.raw_components,
                normalized_slack=item.normalized_slack,
            )
            for item in self.constraint_bounds
        )
        if type(self.constraint_slacks) is not tuple:
            raise TypeError("constraint_slacks must be an exact tuple")
        checked_slacks = tuple(
            ConstraintSlackV2.model_validate(
                item.model_dump(mode="python", warnings="error"), strict=True
            )
            for item in self.constraint_slacks
        )
        bound_ids = tuple(item.constraint_id for item in checked_bounds)
        slack_ids = tuple(item.constraint_id for item in checked_slacks)
        if bound_ids != slack_ids or len(set(bound_ids)) != len(bound_ids):
            raise ValueError("safety cell constraint universe is not closed")
        if type(self.safety_loss) is not NonNegativeIntervalV2:
            raise TypeError("safety_loss must be a NonNegativeIntervalV2")
        checked_loss = NonNegativeIntervalV2.model_validate(
            self.safety_loss.model_dump(mode="python", warnings="error"), strict=True
        )
        object.__setattr__(self, "constraint_bounds", checked_bounds)
        object.__setattr__(self, "constraint_slacks", checked_slacks)
        object.__setattr__(self, "safety_loss", checked_loss)


@dataclass(frozen=True, slots=True)
class ContinuousYawSafetyStageV2_9:
    semantic_problem_sha256: str
    visibility_stage_sha256: str
    relation_stage_sha256: str
    cells: tuple[ContinuousYawSafetyCellV2_9, ...]
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2

    def __post_init__(self) -> None:
        for label, value in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("visibility_stage_sha256", self.visibility_stage_sha256),
            ("relation_stage_sha256", self.relation_stage_sha256),
        ):
            _require_digest(label, value)
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("safety stage requires non-empty cells")
        cells = tuple(_copy_cell(item) for item in self.cells)
        if tuple(item.source_cell_id for item in cells) != tuple(
            sorted(item.source_cell_id for item in cells)
        ):
            raise ValueError("safety cells must be canonically sorted")
        if len({item.source_cell_id for item in cells}) != len(cells):
            raise ValueError("safety cell IDs must be unique")
        usage = _copy_usage(self.resource_usage)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "resource_usage", usage)

    @property
    def stage_sha256(self) -> str:
        digest = hashlib.sha256(_STAGE_HASH_DOMAIN_V2_9)
        for value in (
            self.semantic_problem_sha256,
            self.visibility_stage_sha256,
            self.relation_stage_sha256,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        for cell in self.cells:
            digest.update(cell.source_cell_id.encode())
            for bound in cell.constraint_bounds:
                digest.update(bound.constraint_id.encode())
                for component in bound.raw_components:
                    digest.update(
                        repr(
                            (
                                component.kind.value,
                                component.unit.value,
                                component.raw_lower,
                                component.raw_upper,
                            )
                        ).encode()
                    )
                digest.update(repr(bound.normalized_slack).encode())
            digest.update(repr(cell.safety_loss.model_dump(mode="python")).encode())
        digest.update(repr(self.resource_usage).encode())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousYawSafetyOutcomeV2_9:
    kind: ContinuousYawSafetyKindV2_9
    stage: ContinuousYawSafetyStageV2_9 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawSafetyKindV2_9:
            raise TypeError("kind must be an exact safety kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ContinuousYawSafetyKindV2_9.STAGE:
            if type(self.stage) is not ContinuousYawSafetyStageV2_9:
                raise ValueError("STAGE requires one exact safety stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("safety failure requires findings and no stage")


@dataclass(frozen=True, slots=True)
class ContinuousYawSafetyPointEvaluationV2_9:
    constraint_bounds: tuple[ConstraintSlackBoundsV2, ...]
    constraint_slacks: tuple[ConstraintSlackV2, ...]
    safety_loss: NonNegativeIntervalV2

    def __post_init__(self) -> None:
        cell = ContinuousYawSafetyCellV2_9(
            source_cell_id="point:safety",
            constraint_bounds=self.constraint_bounds,
            constraint_slacks=self.constraint_slacks,
            safety_loss=self.safety_loss,
        )
        object.__setattr__(self, "constraint_bounds", cell.constraint_bounds)
        object.__setattr__(self, "constraint_slacks", cell.constraint_slacks)
        object.__setattr__(self, "safety_loss", cell.safety_loss)


class _UnsupportedSafetyV2_9(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def compile_continuous_yaw_safety_bounds_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    relation_stage: ContinuousYawRelationDamageStageV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawSafetyOutcomeV2_9:
    """Compile sound safety loss and raw slack bounds on every outer cell."""

    try:
        problem, visibility_stage, relation_stage = _strict_inputs(
            problem,
            visibility_stage,
            relation_stage,
            atomic_budget,
            intersection_budget,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            camera = compile_upright_camera_context_v2_9(
                problem,
                atomic_budget=atomic_budget,
                domain_budget=intersection_budget,
            )
            context = _relation_context(problem, camera, intersection_budget)
            cells = tuple(
                _compile_cell(
                    problem,
                    visibility_stage,
                    relation_cell,
                    context,
                    hard_proven=False,
                    atomic_budget=atomic_budget,
                    intersection_budget=intersection_budget,
                )
                for relation_cell in relation_stage.cells
            )
            stage = ContinuousYawSafetyStageV2_9(
                semantic_problem_sha256=problem.semantic_problem_sha256,
                visibility_stage_sha256=visibility_stage.stage_sha256,
                relation_stage_sha256=relation_stage.stage_sha256,
                cells=cells,
                resource_usage=_usage(atomic_budget, intersection_budget),
            )
        return ContinuousYawSafetyOutcomeV2_9(
            kind=ContinuousYawSafetyKindV2_9.STAGE,
            stage=stage,
        )
    except _UnsupportedSafetyV2_9 as error:
        return _failure(
            ContinuousYawSafetyKindV2_9.UNSUPPORTED_MODEL, error.finding_code
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            ContinuousYawSafetyKindV2_9.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_SAFETY_V2_9",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ContinuousYawSafetyKindV2_9.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_SAFETY_V2_9",
        )


def evaluate_continuous_yaw_safety_point_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    relation_stage: ContinuousYawRelationDamageStageV2_9,
    point: RationalPoint2V2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
    cumulative_usage: MultiObstacleStrictConvexCandidateResourceUsageV2 | None = None,
) -> ContinuousYawSafetyPointEvaluationV2_9:
    """Freshly evaluate safety at one edit proven inside the complete hard inner."""

    problem, visibility_stage, relation_stage = _strict_inputs(
        problem,
        visibility_stage,
        relation_stage,
        atomic_budget,
        intersection_budget,
        cumulative_usage=cumulative_usage,
    )
    if type(point) is not RationalPoint2V2:
        raise TypeError("point must be an exact RationalPoint2V2")
    if not visibility_stage.inner_allowed.contains_point(point):
        raise ValueError("safety point is outside the proven hard inner")
    matching = tuple(
        item for item in relation_stage.cells if item.cell.contains_point(point)
    )
    if not matching:
        raise ValueError("safety point escaped the relation outer partition")
    camera = compile_upright_camera_context_v2_9(
        problem,
        atomic_budget=atomic_budget,
        domain_budget=intersection_budget,
    )
    context = _relation_context(problem, camera, intersection_budget)
    bundle = _compile_components(
        problem,
        visibility_stage,
        tuple(item.cell_id for item in matching),
        context,
        (point.x, point.x),
        (point.y, point.y),
        cell=None,
        hard_proven=True,
        atomic_budget=atomic_budget,
        intersection_budget=intersection_budget,
    )
    return ContinuousYawSafetyPointEvaluationV2_9(*bundle)


def _compile_cell(
    problem: SemanticProblemV2_3,
    visibility: ContinuousYawVisibilityStageV2_9,
    relation_cell: RelationDamageCellV2_9,
    context,
    *,
    hard_proven: bool,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawSafetyCellV2_9:
    vertices = relation_cell.cell.closure_polygon.vertices_ccw
    # Position and support stability each enumerate one center plus six
    # pairwise margin crossings per convex edge.
    intersection_budget.consume_domain(2 + 14 * len(vertices))
    dx = min(item.x for item in vertices), max(item.x for item in vertices)
    dy = min(item.y for item in vertices), max(item.y for item in vertices)
    bundle = _compile_components(
        problem,
        visibility,
        (relation_cell.cell_id,),
        context,
        dx,
        dy,
        cell=relation_cell.cell,
        hard_proven=hard_proven,
        atomic_budget=atomic_budget,
        intersection_budget=intersection_budget,
    )
    return ContinuousYawSafetyCellV2_9(relation_cell.cell_id, *bundle)


def _compile_components(
    problem,
    visibility,
    cell_ids,
    context,
    dx,
    dy,
    *,
    cell,
    hard_proven,
    atomic_budget,
    intersection_budget,
):
    targets = problem.objective.safety_margin.aggregation.targets
    intersection_budget.consume_domain(4 + len(targets) + len(cell_ids))
    bounds = []
    inputs = []
    slacks = []
    for target in targets:
        components = _raw_components(
            problem,
            visibility,
            cell_ids,
            context,
            target,
            dx,
            dy,
            cell=cell,
            hard_proven=hard_proven,
            atomic_budget=atomic_budget,
        )
        published = compile_constraint_slack_bounds_v2(target, components)
        interval = _exact_interval(published, "SAFETY_SLACK")
        bounds.append(
            ConstraintSlackBoundsV2(target.constraint_id, components, interval)
        )
        inputs.append(ConstraintSafetyInputV2(target.constraint_id, components))
        slacks.append(
            ConstraintSlackV2(
                constraint_id=target.constraint_id,
                lower_bound=interval.lower_bound,
                upper_bound=interval.upper_bound,
                unit=SafetySlackUnitV2.DIMENSIONLESS,
            )
        )
    loss = _exact_interval(
        aggregate_safety_penalty_bounds_v2(
            problem.objective.safety_margin, tuple(inputs)
        ),
        "SAFETY_AGGREGATION",
    )
    if loss.lower_bound < 0:
        raise RuntimeError("safety aggregation published a negative loss")
    return (
        tuple(bounds),
        tuple(slacks),
        NonNegativeIntervalV2(
            lower_bound=loss.lower_bound,
            upper_bound=loss.upper_bound,
        ),
    )


def _raw_components(
    problem,
    visibility,
    cell_ids,
    context,
    target,
    dx,
    dy,
    *,
    cell,
    hard_proven,
    atomic_budget,
):
    visibility_constraint = next(
        (
            item
            for item in problem.constraints.visibility_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if visibility_constraint is not None:
        return _visibility_components(
            visibility, cell_ids, visibility_constraint, target
        )
    if target.constraint_id == problem.constraints.target_relation.constraint_id:
        return _target_components(
            problem,
            context,
            target,
            dx,
            dy,
            atomic_budget,
            hard_proven=hard_proven,
        )
    if target.constraint_id == problem.constraints.position_domain.constraint_id:
        return _position_components(problem, target, dx, dy, cell=cell)
    collision = next(
        (
            item
            for item in problem.constraints.collision_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if collision is not None:
        return _collision_components(problem, collision, target, dx, dy, atomic_budget)
    support = next(
        (
            item
            for item in problem.constraints.support_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if support is not None:
        return _support_components(
            problem,
            support,
            target,
            dx,
            dy,
            atomic_budget,
            cell=cell,
            hard_proven=hard_proven,
        )
    raise _UnsupportedSafetyV2_9(
        f"UNSUPPORTED_MODEL:SAFETY_CONSTRAINT_V2_9:{target.constraint_id}"
    )


def _position_components(problem, target, dx, dy, *, cell):
    constraint = problem.constraints.position_domain
    if (
        constraint.boundary_policy is not BoundaryPolicyV2.CLOSED
        or constraint.region_interpretation
        is not PositionRegionInterpretationV2.SUBJECT_ANCHOR_LOCUS
        or constraint.workspace_aggregation is not RegionAggregationV2.INTERSECTION
        or len(constraint.workspace_fact_ids) != 1
        or constraint.known_free_space_fact_ids
    ):
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:POSITION_SAFETY_POLICY_V2_9")
    workspace = next(
        item
        for item in problem.scene.workspace_boundaries.values or ()
        if item.fact_id == constraint.workspace_fact_ids[0]
    )
    if (
        workspace.region_approximation is not GeometryApproximationV2.EXACT
        or workspace.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
        or workspace.geometry_uncertainty != UncertaintyBudgetV2()
    ):
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:POSITION_SAFETY_WORKSPACE_V2_9")
    rectangle = ExactAxisAlignedRectV2.from_planar_region(workspace.region_world_xy)
    bounds = rectangle.bounds
    if bounds is None:
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:POSITION_SAFETY_WORKSPACE_V2_9")
    subject = next(
        item
        for item in problem.scene.objects.values or ()
        if item.object_id == constraint.subject_id
    )
    anchor = subject.pose.world_from_object.translation
    clearance = Fraction.from_float(constraint.minimum_boundary_clearance_m)
    permitted = (
        bounds[0] + clearance - Fraction.from_float(anchor.x),
        bounds[1] + clearance - Fraction.from_float(anchor.y),
        bounds[2] - clearance - Fraction.from_float(anchor.x),
        bounds[3] - clearance - Fraction.from_float(anchor.y),
    )
    lower = min(
        dx[0] - permitted[0],
        permitted[2] - dx[1],
        dy[0] - permitted[1],
        permitted[3] - dy[1],
    )
    if dx[0] == dx[1] and dy[0] == dy[1]:
        upper = lower
    elif cell is not None:
        upper = _maximum_rectangular_margin_over_cell(cell, permitted)
    else:
        upper = min(
            (permitted[2] - permitted[0]) / 2,
            (permitted[3] - permitted[1]) / 2,
        )
    return _components_from_values(
        target,
        {
            SafetyComponentKindV2.POSITION_BOUNDARY_CLEARANCE: (
                SafetySlackUnitV2.METRE,
                lower,
                upper,
            )
        },
    )


def _maximum_rectangular_margin_over_cell(
    cell: StrictConvexIntersectionCellV2,
    rectangle: tuple[Fraction, Fraction, Fraction, Fraction],
) -> Fraction:
    """Maximize the minimum signed rectangle-side margin on a convex cell."""

    if type(cell) is not StrictConvexIntersectionCellV2:
        raise TypeError("position safety cell must be an exact convex cell")
    left, bottom, right, top = rectangle
    margins = (
        (Fraction(1), Fraction(), -left),
        (Fraction(-1), Fraction(), right),
        (Fraction(), Fraction(1), -bottom),
        (Fraction(), Fraction(-1), top),
    )
    vertices = cell.closure_polygon.vertices_ccw
    candidates = list(vertices)
    center = RationalPoint2V2((left + right) / 2, (bottom + top) / 2)
    if cell.contains_point(center):
        candidates.append(center)
    for first, second in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        edge_x = second.x - first.x
        edge_y = second.y - first.y
        for index, left_margin in enumerate(margins):
            for right_margin in margins[index + 1 :]:
                difference_x = left_margin[0] - right_margin[0]
                difference_y = left_margin[1] - right_margin[1]
                difference_c = left_margin[2] - right_margin[2]
                denominator = difference_x * edge_x + difference_y * edge_y
                if denominator == 0:
                    continue
                parameter = (
                    -(difference_x * first.x + difference_y * first.y + difference_c)
                    / denominator
                )
                if Fraction() <= parameter <= Fraction(1):
                    candidates.append(
                        RationalPoint2V2(
                            first.x + parameter * edge_x,
                            first.y + parameter * edge_y,
                        )
                    )

    def value(point: RationalPoint2V2) -> Fraction:
        return min(
            coefficient_x * point.x + coefficient_y * point.y + constant
            for coefficient_x, coefficient_y, constant in margins
        )

    return max(value(point) for point in candidates)


def _collision_components(problem, constraint, target, dx, dy, budget):
    if (
        constraint.clearance_metric
        is not CollisionClearanceMetricV2.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
        or constraint.boundary_policy is not BoundaryPolicyV2.CLOSED
        or constraint.minimum_clearance_m != 0.0
        or constraint.support_contact_exceptions
    ):
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:COLLISION_SAFETY_POLICY_V2_9")
    bodies = {
        item.body_id: item for item in problem.scene.collision_bodies.values or ()
    }
    pairs = []
    for subject_body_id in constraint.subject_body_ids:
        for obstacle_body_id in constraint.obstacle_body_ids:
            pairs.append(
                (
                    _collision_operand(problem, bodies[subject_body_id]),
                    _collision_operand(problem, bodies[obstacle_body_id]),
                )
            )
    lower_values = []
    upper_values = []
    singleton = dx[0] == dx[1] and dy[0] == dy[1]
    for subject, obstacle in pairs:
        subject_transform, subject_shape = subject
        obstacle_transform, obstacle_shape = obstacle
        if singleton:
            exact_x = Fraction.from_float(subject_transform.translation.x) + dx[0]
            exact_y = Fraction.from_float(subject_transform.translation.y) + dy[0]
            rounded_x = float(exact_x)
            rounded_y = float(exact_y)
            if not math.isfinite(rounded_x) or not math.isfinite(rounded_y):
                raise ArithmeticError("collision safety point translation overflow")
            moved = DirectedYawIntervalTransformV2_2(
                translation=Vec3V2(
                    x=rounded_x,
                    y=rounded_y,
                    z=subject_transform.translation.z,
                ),
                yaw_radians=subject_transform.yaw_radians,
            )
            outcome = compile_oriented_upright_box_pair_bounds_v2(
                moved,
                subject_shape,
                obstacle_transform,
                obstacle_shape,
                atomic_budget=budget,
            )
            if outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
                raise SO2AtomicBudgetExhaustedV2
            if outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
                raise ArithmeticError("collision safety pair numeric gap")
            if outcome.kind is not SO2IntervalKindV2.EXACT or outcome.bounds is None:
                raise RuntimeError("supported collision pair did not publish bounds")
            squared = outcome.bounds.squared_clearance
            budget.consume(128)
            lower_sqrt = _directed_sqrt_binary64_bounds(squared.rational_lower)
            upper_sqrt = _directed_sqrt_binary64_bounds(squared.rational_upper)
            if lower_sqrt is None or upper_sqrt is None:
                raise ArithmeticError("collision safety sqrt numeric gap")
            rounding_squared = (Fraction.from_float(rounded_x) - exact_x) ** 2 + (
                Fraction.from_float(rounded_y) - exact_y
            ) ** 2
            budget.consume(64)
            rounding_sqrt = _directed_sqrt_binary64_bounds(rounding_squared)
            if rounding_sqrt is None:
                raise ArithmeticError("collision safety rounding enclosure failed")
            rounding_upper = Fraction.from_float(rounding_sqrt[1])
            lower_values.append(
                max(Fraction(), Fraction.from_float(lower_sqrt[0]) - rounding_upper)
            )
            upper_values.append(Fraction.from_float(upper_sqrt[1]) + rounding_upper)
        else:
            lower_values.append(Fraction())
            center_dx = Fraction.from_float(
                subject_transform.translation.x
            ) - Fraction.from_float(obstacle_transform.translation.x)
            center_dy = Fraction.from_float(
                subject_transform.translation.y
            ) - Fraction.from_float(obstacle_transform.translation.y)
            center_dz = Fraction.from_float(
                subject_transform.translation.z
            ) - Fraction.from_float(obstacle_transform.translation.z)
            maximum_squared = (
                max(abs(center_dx + dx[0]), abs(center_dx + dx[1])) ** 2
                + max(abs(center_dy + dy[0]), abs(center_dy + dy[1])) ** 2
                + center_dz**2
            )
            budget.consume(64)
            upper_sqrt = _directed_sqrt_binary64_bounds(maximum_squared)
            if upper_sqrt is None:
                raise ArithmeticError("collision safety cell sqrt numeric gap")
            upper_values.append(Fraction.from_float(upper_sqrt[1]))
    required = Fraction.from_float(constraint.minimum_clearance_m)
    return _components_from_values(
        target,
        {
            SafetyComponentKindV2.COLLISION_CLEARANCE: (
                SafetySlackUnitV2.METRE,
                min(lower_values) - required,
                min(upper_values) - required,
            )
        },
    )


def _collision_operand(problem, body):
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = {
        item.geometry_id: item for item in problem.scene.geometry_instances.values or ()
    }
    if len(body.geometry_instance_ids) != 1:
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:COLLISION_SAFETY_BODY_V2_9")
    geometry = geometries[body.geometry_instance_ids[0]]
    if body.owner_object_id is None:
        if (
            type(geometry) is not GeometryInstanceV2_2
            or geometry.owner_object_id is not None
            or geometry.role is not GeometryRoleV2.COLLISION
            or geometry.approximation is not GeometryApproximationV2.EXACT
            or geometry.uncertainty != UncertaintyBudgetV2()
            or type(geometry.shape) is not UprightBox3DV2
            or type(geometry.anchor_from_geometry)
            is not DirectedYawIntervalTransformV2_2
        ):
            raise _UnsupportedSafetyV2_9(
                "UNSUPPORTED_MODEL:COLLISION_SAFETY_GEOMETRY_V2_9"
            )
        return geometry.anchor_from_geometry, geometry.shape
    object_ = objects[body.owner_object_id]
    if (
        type(geometry) is not GeometryInstanceV2_2
        or geometry.role is not GeometryRoleV2.COLLISION
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or geometry.uncertainty != UncertaintyBudgetV2()
        or type(geometry.shape) is not UprightBox3DV2
        or geometry.anchor_from_geometry.translation != Vec3V2(x=0.0, y=0.0, z=0.0)
        or geometry.anchor_from_geometry.yaw_radians != 0.0
        or type(object_.pose.world_from_object) is not DirectedYawIntervalTransformV2_2
    ):
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:COLLISION_SAFETY_GEOMETRY_V2_9")
    return object_.pose.world_from_object, geometry.shape


def _support_components(
    problem, constraint, target, dx, dy, budget, *, cell, hard_proven
):
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = {
        item.geometry_id: item for item in problem.scene.geometry_instances.values or ()
    }
    surfaces = {
        item.surface_id: item for item in problem.scene.support_surfaces.values or ()
    }
    if len(constraint.subject_contact_geometry_ids) != 1:
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:SUPPORT_SAFETY_GEOMETRY_V2_9")
    subject = objects[constraint.supported_object_id]
    geometry = geometries[constraint.subject_contact_geometry_ids[0]]
    surface = surfaces[constraint.surface_id]
    owner = objects[surface.owner_object_id] if surface.owner_object_id else None
    owner_transform = (
        owner.pose.world_from_object
        if owner is not None
        else DirectedYawIntervalTransformV2_2(
            translation=Vec3V2(x=0.0, y=0.0, z=0.0),
            yaw_radians=0.0,
        )
    )
    if (
        type(geometry) is not GeometryInstanceV2_2
        or geometry.role is not GeometryRoleV2.SUPPORT
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or geometry.uncertainty != UncertaintyBudgetV2()
        or type(geometry.shape) is not UprightBox3DV2
        or geometry.anchor_from_geometry.translation != Vec3V2(x=0.0, y=0.0, z=0.0)
        or geometry.anchor_from_geometry.yaw_radians != 0.0
        or type(subject.pose.world_from_object) is not DirectedYawIntervalTransformV2_2
        or type(owner_transform) is not DirectedYawIntervalTransformV2_2
        or owner_transform.yaw_radians != 0.0
        or surface.region_approximation is not GeometryApproximationV2.EXACT
        or surface.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
        or surface.geometry_uncertainty != UncertaintyBudgetV2()
        or surface.anchor_from_surface.yaw_radians != 0.0
    ):
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:SUPPORT_SAFETY_POLICY_V2_9")
    box_outcome = compile_oriented_upright_box_bounds_v2(
        subject.pose.world_from_object,
        geometry.shape,
        atomic_budget=budget,
    )
    if box_outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        raise SO2AtomicBudgetExhaustedV2
    if box_outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
        raise ArithmeticError("support safety box numeric gap")
    if box_outcome.kind is not SO2IntervalKindV2.EXACT or box_outcome.bounds is None:
        raise RuntimeError("supported support box did not publish bounds")
    box = box_outcome.bounds
    surface_rectangle = ExactAxisAlignedRectV2.from_planar_region(surface.region_uv)
    surface_bounds = surface_rectangle.bounds
    if surface_bounds is None:
        raise _UnsupportedSafetyV2_9("UNSUPPORTED_MODEL:SUPPORT_SAFETY_SURFACE_V2_9")
    owner_t = owner_transform.translation
    surface_t = surface.anchor_from_surface.translation
    inset = Fraction.from_float(constraint.stability_margin_m)
    support_bounds = (
        Fraction.from_float(owner_t.x)
        + Fraction.from_float(surface_t.x)
        + surface_bounds[0]
        + inset,
        Fraction.from_float(owner_t.y)
        + Fraction.from_float(surface_t.y)
        + surface_bounds[1]
        + inset,
        Fraction.from_float(owner_t.x)
        + Fraction.from_float(surface_t.x)
        + surface_bounds[2]
        - inset,
        Fraction.from_float(owner_t.y)
        + Fraction.from_float(surface_t.y)
        + surface_bounds[3]
        - inset,
    )
    center_x = Fraction.from_float(subject.pose.world_from_object.translation.x)
    center_y = Fraction.from_float(subject.pose.world_from_object.translation.y)
    radius_x_lower = box.x_radius.rational_lower
    radius_x_upper = box.x_radius.rational_upper
    radius_y_lower = box.y_radius.rational_lower
    radius_y_upper = box.y_radius.rational_upper
    stability_lower = min(
        center_x + dx[0] - radius_x_upper - support_bounds[0],
        support_bounds[2] - center_x - dx[1] - radius_x_upper,
        center_y + dy[0] - radius_y_upper - support_bounds[1],
        support_bounds[3] - center_y - dy[1] - radius_y_upper,
    )
    if dx[0] == dx[1] and dy[0] == dy[1]:
        stability_upper = min(
            center_x + dx[0] - radius_x_lower - support_bounds[0],
            support_bounds[2] - center_x - dx[0] - radius_x_lower,
            center_y + dy[0] - radius_y_lower - support_bounds[1],
            support_bounds[3] - center_y - dy[0] - radius_y_lower,
        )
    elif cell is not None:
        stability_upper = _maximum_rectangular_margin_over_cell(
            cell,
            (
                support_bounds[0] + radius_x_lower - center_x,
                support_bounds[1] + radius_y_lower - center_y,
                support_bounds[2] - radius_x_lower - center_x,
                support_bounds[3] - radius_y_lower - center_y,
            ),
        )
    else:
        stability_upper = min(
            (support_bounds[2] - support_bounds[0]) / 2 - radius_x_lower,
            (support_bounds[3] - support_bounds[1]) / 2 - radius_y_lower,
        )
    subject_z = Fraction.from_float(subject.pose.world_from_object.translation.z)
    owner_z = Fraction.from_float(owner_t.z)
    surface_z = Fraction.from_float(surface_t.z)
    contact_gap = subject_z - box.half_extent_z - owner_z - surface_z
    contact_area = Fraction.from_float(geometry.shape.size_m.x) * Fraction.from_float(
        geometry.shape.size_m.y
    )
    overlap_lower = contact_area if hard_proven else Fraction()
    values = {
        SafetyComponentKindV2.SUPPORT_CONTACT_GAP_LOWER_MARGIN: (
            SafetySlackUnitV2.METRE,
            contact_gap - Fraction.from_float(constraint.contact_gap_min_m),
            contact_gap - Fraction.from_float(constraint.contact_gap_min_m),
        ),
        SafetyComponentKindV2.SUPPORT_CONTACT_GAP_UPPER_MARGIN: (
            SafetySlackUnitV2.METRE,
            Fraction.from_float(constraint.contact_gap_max_m) - contact_gap,
            Fraction.from_float(constraint.contact_gap_max_m) - contact_gap,
        ),
        SafetyComponentKindV2.SUPPORT_OVERLAP_AREA_MARGIN: (
            SafetySlackUnitV2.SQUARE_METRE,
            overlap_lower - Fraction.from_float(constraint.minimum_overlap_area_m2),
            contact_area - Fraction.from_float(constraint.minimum_overlap_area_m2),
        ),
        SafetyComponentKindV2.SUPPORT_STABILITY_INSET_MARGIN: (
            SafetySlackUnitV2.METRE,
            stability_lower,
            stability_upper,
        ),
    }
    return _components_from_values(target, values)


def _components_from_values(target, values):
    if {item.kind for item in target.components} != set(values):
        raise RuntimeError("safety component universe drifted from its constraint")
    result = []
    for component in target.components:
        unit, lower, upper = values[component.kind]
        if component.unit is not unit or lower > upper:
            raise RuntimeError("safety component interval failed semantic closure")
        # Raw safety expressions can inherit very large rational denominators
        # from directed SO(2) clipping.  Publish each endpoint outward to its
        # wire binary64 enclosure before the shared numeric helper applies its
        # certified Fraction bit cap.  This preserves soundness while avoiding
        # a false numeric gap caused only by an exact intermediate's size.
        published_lower = Fraction.from_float(_fraction_to_float_floor(lower))
        published_upper = Fraction.from_float(_fraction_to_float_ceil(upper))
        result.append(
            SafetyRawComponentIntervalV2(
                kind=component.kind,
                unit=component.unit,
                raw_lower=published_lower,
                raw_upper=published_upper,
            )
        )
    return tuple(result)


def _visibility_components(visibility, cell_ids, constraint, target):
    by_cell = tuple(
        metric for metric in visibility.outer_cell_metrics if metric.cell_id in cell_ids
    )
    expected_objects = set(constraint.query_object_ids)
    if {item.object_id for item in by_cell} != expected_objects:
        raise RuntimeError("visibility safety metrics lost their query closure")
    thresholds = {
        SafetyComponentKindV2.VISIBILITY_VISIBLE_FRACTION_MARGIN: (
            "visible_fraction_lower",
            "visible_fraction_upper",
            Fraction.from_float(constraint.minimum_visible_fraction),
            False,
        ),
        SafetyComponentKindV2.VISIBILITY_IMAGE_AREA_FRACTION_MARGIN: (
            "image_area_fraction_lower",
            "image_area_fraction_upper",
            Fraction.from_float(constraint.minimum_image_area_fraction),
            False,
        ),
        SafetyComponentKindV2.VISIBILITY_TRUNCATED_FRACTION_MARGIN: (
            "truncated_fraction_lower",
            "truncated_fraction_upper",
            Fraction.from_float(constraint.maximum_truncated_fraction),
            True,
        ),
    }
    components = []
    for component in target.components:
        try:
            lower_name, upper_name, threshold, reverse = thresholds[component.kind]
        except KeyError as error:
            raise RuntimeError(
                "visibility safety target has the wrong component"
            ) from error
        per_object = []
        for object_id in sorted(expected_objects):
            values = tuple(item for item in by_cell if item.object_id == object_id)
            lower = min(getattr(item, lower_name) for item in values)
            upper = max(getattr(item, upper_name) for item in values)
            per_object.append(
                (threshold - upper, threshold - lower)
                if reverse
                else (lower - threshold, upper - threshold)
            )
        components.append(
            SafetyRawComponentIntervalV2(
                kind=component.kind,
                unit=component.unit,
                raw_lower=Fraction.from_float(
                    _fraction_to_float_floor(min(item[0] for item in per_object))
                ),
                raw_upper=Fraction.from_float(
                    _fraction_to_float_ceil(min(item[1] for item in per_object))
                ),
            )
        )
    return tuple(components)


def _target_components(
    problem, context, target_spec, dx, dy, budget, *, hard_proven: bool
):
    target = problem.constraints.target_relation
    definition = next(
        item
        for item in problem.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    key = PairAxisKeyV2(
        first_object_id=target.subject_id,
        second_object_id=target.reference_id,
        axis=target.relation_after.axis,
    )
    if key.axis in (RelationAxisV2.HORIZONTAL, RelationAxisV2.DEPTH):
        measurement = _linear_measurement(context, key, dx, dy, budget)
        if key.first_object_id != target.subject_id:
            measurement = -measurement[1], -measurement[0]
    else:
        squared = _distance_squared(context, key, dx, dy, budget)
        budget.consume(128)
        lower = _directed_sqrt_binary64_bounds(squared[0])
        upper = _directed_sqrt_binary64_bounds(squared[1])
        if lower is None or upper is None:
            raise ArithmeticError("target safety distance overflow")
        measurement = Fraction.from_float(lower[0]), Fraction.from_float(upper[1])
    threshold = Fraction.from_float(definition.threshold)
    if definition.comparator is MeasurementComparatorV2.LESS_THAN:
        raw = threshold - measurement[1], threshold - measurement[0]
    elif definition.comparator is MeasurementComparatorV2.GREATER_THAN:
        raw = measurement[0] - threshold, measurement[1] - threshold
    else:  # pragma: no cover - closed enum
        raise RuntimeError("target comparator escaped its closed enum")
    raw = (
        Fraction.from_float(_fraction_to_float_floor(raw[0])),
        Fraction.from_float(_fraction_to_float_ceil(raw[1])),
    )
    if hard_proven:
        raw = max(Fraction(), raw[0]), max(Fraction(), raw[1])
    if len(target_spec.components) != 1 or target_spec.components[0].kind is not (
        SafetyComponentKindV2.TARGET_RELATION_THRESHOLD_MARGIN
    ):
        raise RuntimeError("target safety component universe drifted")
    component = target_spec.components[0]
    return (
        SafetyRawComponentIntervalV2(
            kind=component.kind,
            unit=component.unit,
            raw_lower=raw[0],
            raw_upper=raw[1],
        ),
    )


def _strict_inputs(
    problem, visibility, relation, atomic, domain, *, cumulative_usage=None
):
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(visibility) is not ContinuousYawVisibilityStageV2_9:
        raise TypeError("visibility_stage must be an exact v2.9 stage")
    if type(relation) is not ContinuousYawRelationDamageStageV2_9:
        raise TypeError("relation_stage must be an exact v2.9 stage")
    if type(atomic) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    if type(domain) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    usage = relation.resource_usage if cumulative_usage is None else cumulative_usage
    if type(usage) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("cumulative_usage has the wrong exact type")
    if (
        atomic.used != usage.so2_atomic_steps
        or domain.domain_operations_used != usage.domain_operations
        or domain.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("safety ledgers must continue relation usage")
    domain.consume_domain(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = SemanticProblemV2_3.model_validate(
            problem.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_visibility = ContinuousYawVisibilityStageV2_9(
            semantic_problem_sha256=visibility.semantic_problem_sha256,
            candidate_problem_sha256=visibility.candidate_problem_sha256,
            camera_context_sha256=visibility.camera_context_sha256,
            compiler_config_sha256=visibility.compiler_config_sha256,
            upstream_t15_artifact_sha256=visibility.upstream_t15_artifact_sha256,
            upstream_target_stage_sha256=visibility.upstream_target_stage_sha256,
            subject_id=visibility.subject_id,
            visibility_constraint_id=visibility.visibility_constraint_id,
            inner_allowed=visibility.inner_allowed,
            outer_allowed=visibility.outer_allowed,
            outer_cell_metrics=visibility.outer_cell_metrics,
            baseline_metrics=visibility.baseline_metrics,
            fixed_object_ids=visibility.fixed_object_ids,
            remaining_constraint_ids=visibility.remaining_constraint_ids,
            unsat_prefix_eligible=visibility.unsat_prefix_eligible,
            resource_usage=visibility.resource_usage,
        )
        checked_relation = ContinuousYawRelationDamageStageV2_9(
            semantic_problem_sha256=relation.semantic_problem_sha256,
            candidate_problem_sha256=relation.candidate_problem_sha256,
            camera_context_sha256=relation.camera_context_sha256,
            visibility_stage_sha256=relation.visibility_stage_sha256,
            cells=relation.cells,
            resource_usage=relation.resource_usage,
        )
    if (
        checked_problem.semantic_problem_sha256
        != checked_visibility.semantic_problem_sha256
        or checked_problem.semantic_problem_sha256
        != checked_relation.semantic_problem_sha256
        or checked_relation.visibility_stage_sha256 != checked_visibility.stage_sha256
    ):
        raise ValueError("safety prefix hashes are not closed")
    if tuple(item.cell_id for item in checked_relation.cells) != tuple(
        item.cell_id for item in checked_visibility.outer_allowed.cells
    ):
        raise ValueError("safety relation cells do not close visibility outer")
    return checked_problem, checked_visibility, checked_relation


def _exact_interval(outcome, label: str) -> ExactPublishedIntervalV2:
    if outcome.kind is not ObjectiveNumericKindV2.EXACT or outcome.interval is None:
        if outcome.kind is ObjectiveNumericKindV2.RESOURCE_LIMIT:
            raise StrictConvexIntersectionBudgetExhaustedV2
        if outcome.kind is ObjectiveNumericKindV2.NUMERIC_GAP:
            raise ArithmeticError(f"{label} reached a numeric gap")
        raise RuntimeError(f"{label} failed strict numeric closure")
    return outcome.interval


def _copy_cell(value: ContinuousYawSafetyCellV2_9) -> ContinuousYawSafetyCellV2_9:
    if type(value) is not ContinuousYawSafetyCellV2_9:
        raise TypeError("safety cell has the wrong exact type")
    return ContinuousYawSafetyCellV2_9(
        source_cell_id=value.source_cell_id,
        constraint_bounds=value.constraint_bounds,
        constraint_slacks=value.constraint_slacks,
        safety_loss=value.safety_loss,
    )


def _copy_stage(value: ContinuousYawSafetyStageV2_9) -> ContinuousYawSafetyStageV2_9:
    return ContinuousYawSafetyStageV2_9(
        semantic_problem_sha256=value.semantic_problem_sha256,
        visibility_stage_sha256=value.visibility_stage_sha256,
        relation_stage_sha256=value.relation_stage_sha256,
        cells=value.cells,
        resource_usage=value.resource_usage,
    )


def _copy_usage(value):
    if type(value) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("safety usage has the wrong exact type")
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=value.domain_operations,
        so2_atomic_steps=value.so2_atomic_steps,
        candidate_cells=value.candidate_cells,
    )


def _usage(atomic, domain):
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=domain.domain_operations_used,
        so2_atomic_steps=atomic.used,
        candidate_cells=domain.candidate_cells_used,
    )


def _require_digest(label: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(kind, finding):
    return ContinuousYawSafetyOutcomeV2_9(kind=kind, finding_codes=(finding,))


__all__ = (
    "ContinuousYawSafetyCellV2_9",
    "ContinuousYawSafetyKindV2_9",
    "ContinuousYawSafetyOutcomeV2_9",
    "ContinuousYawSafetyPointEvaluationV2_9",
    "ContinuousYawSafetyStageV2_9",
    "compile_continuous_yaw_safety_bounds_v2_9",
    "evaluate_continuous_yaw_safety_point_v2_9",
)
