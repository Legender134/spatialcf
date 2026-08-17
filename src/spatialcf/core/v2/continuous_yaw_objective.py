"""Private T18 objective bounds for the continuous-yaw candidate chain."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from sys import float_info

from spatialcf.core.v2.continuous_yaw_visibility import (
    _CompleteContinuousYawCandidateStageV2,
    _copy_complete_stage,
)
from spatialcf.core.v2.convex_translation_domain import RationalPoint2V2
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    _artifact_bytes,
    _strict_problem,
)
from spatialcf.core.v2.objective_numeric import (
    ConstraintSafetyInputV2,
    ObjectiveNumericKindV2,
    SafetyRawComponentIntervalV2,
    VisibilityMetricIntervalV2,
    _directed_sqrt_binary64_bounds,
    aggregate_relation_damage_bounds_v2,
    aggregate_safety_penalty_bounds_v2,
    aggregate_visibility_change_bounds_v2,
    compile_constraint_slack_bounds_v2,
)
from spatialcf.core.v2.so2_interval import SO2AtomicBudgetV2
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
)
from spatialcf.domain.v2.artifacts import (
    ConstraintSlackV2,
    NonNegativeIntervalV2,
    ObjectiveTermBoundsV2,
    RelationDamageBoundV2,
)
from spatialcf.domain.v2.base import Vec2V2
from spatialcf.domain.v2.continuous_yaw_candidate import SemanticProblemV2_2
from spatialcf.domain.v2.continuous_yaw_solver import (
    ContinuousYawObjectiveCellV2_8,
    ContinuousYawResourceUsageV2_8,
    ContinuousYawSolverConfigV2_8,
    ContinuousYawWitnessEvaluationV2_8,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.objective import SafetySlackUnitV2

_STAGE_HASH_DOMAIN_V2 = b"spatialcf.continuous-yaw-objective-stage.v2.8\0"
_OUTER_DOMAIN_HASH_V2 = b"spatialcf.continuous-yaw-outer-cell.v2.8\0"
_INNER_DOMAIN_HASH_V2 = b"spatialcf.continuous-yaw-inner-cells.v2.8\0"
_SQRT_COST_V2 = 128


class _UnsupportedContinuousYawObjectiveV2(ValueError):
    pass


class _ContinuousYawObjectiveNumericGapV2(ArithmeticError):
    pass


class _ContinuousYawWitnessNotProvenV2(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ContinuousYawWitnessProposalV2:
    objective_cell_id: str
    edit: CanonicalEditV2

    def __post_init__(self) -> None:
        if type(self.objective_cell_id) is not str or not self.objective_cell_id:
            raise ValueError("objective proposal cell ID must be non-blank")
        if type(self.edit) is not CanonicalEditV2:
            raise TypeError("objective proposal edit has the wrong exact type")
        object.__setattr__(
            self,
            "edit",
            CanonicalEditV2.model_validate(
                self.edit.model_dump(mode="python", warnings="error"), strict=True
            ),
        )


@dataclass(frozen=True, slots=True)
class _ContinuousYawObjectiveStageV2:
    semantic_problem_sha256: str
    solver_config_sha256: str
    candidate_stage_sha256: str
    cells: tuple[ContinuousYawObjectiveCellV2_8, ...]
    proposals: tuple[_ContinuousYawWitnessProposalV2, ...]
    outer_source_cells: tuple[StrictConvexIntersectionCellV2, ...]
    inner_source_cells: tuple[
        tuple[str, tuple[StrictConvexIntersectionCellV2, ...]], ...
    ]
    resource_usage: ContinuousYawResourceUsageV2_8

    def __post_init__(self) -> None:
        for value in (
            self.semantic_problem_sha256,
            self.solver_config_sha256,
            self.candidate_stage_sha256,
        ):
            if type(value) is not str or len(value) != 64:
                raise ValueError("objective stage hashes must be SHA-256 strings")
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("objective stage requires a non-empty exact cell tuple")
        cells = tuple(
            ContinuousYawObjectiveCellV2_8.model_validate(
                item.model_dump(mode="python", warnings="error"), strict=True
            )
            for item in self.cells
        )
        if tuple(item.cell_id for item in cells) != tuple(
            sorted(item.cell_id for item in cells)
        ):
            raise ValueError("objective cells must be canonically ordered")
        if type(self.proposals) is not tuple:
            raise TypeError("objective proposals must be an exact tuple")
        proposals = tuple(
            _ContinuousYawWitnessProposalV2(item.objective_cell_id, item.edit)
            for item in self.proposals
        )
        proposal_ids = tuple(item.objective_cell_id for item in proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("objective stage accepts at most one proposal per cell")
        if not set(proposal_ids) <= {item.cell_id for item in cells}:
            raise ValueError("objective proposal references an unknown cell")
        if type(self.outer_source_cells) is not tuple:
            raise TypeError("outer_source_cells must be an exact tuple")
        if len(self.outer_source_cells) != len(cells):
            raise ValueError("objective cells must preserve every outer source cell")
        if type(self.inner_source_cells) is not tuple:
            raise TypeError("inner_source_cells must be an exact tuple")
        if tuple(item[0] for item in self.inner_source_cells) != tuple(
            cell.cell_id for cell in cells
        ):
            raise ValueError("inner source mapping must close the objective cells")
        if type(self.resource_usage) is not ContinuousYawResourceUsageV2_8:
            raise TypeError("objective resource usage has the wrong exact type")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "proposals", proposals)

    @property
    def global_loss_lower_bound(self) -> float:
        return min(item.term_loss_bounds.total_lower_bound for item in self.cells)

    @property
    def objective_cells_sha256(self) -> str:
        return ContinuousYawObjectiveCellV2_8.sequence_sha256(self.cells)

    @property
    def stage_sha256(self) -> str:
        payload = (
            self.semantic_problem_sha256,
            self.solver_config_sha256,
            self.candidate_stage_sha256,
            self.objective_cells_sha256,
            tuple(
                (item.objective_cell_id, item.edit.edit_sha256)
                for item in self.proposals
            ),
            tuple(
                (name, tuple(cell.cell_id for cell in inner))
                for name, inner in self.inner_source_cells
            ),
            (
                self.resource_usage.domain_operations,
                self.resource_usage.so2_atomic_steps,
                self.resource_usage.candidate_cells,
                self.resource_usage.objective_partition_cells,
            ),
        )
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2 + repr(payload).encode()
        ).hexdigest()

    def cell_by_id(self, cell_id: str) -> ContinuousYawObjectiveCellV2_8:
        matches = tuple(item for item in self.cells if item.cell_id == cell_id)
        if len(matches) != 1:
            raise ValueError("objective cell ID is not closed")
        return matches[0]

    def source_cell_by_objective_id(
        self, cell_id: str
    ) -> StrictConvexIntersectionCellV2:
        index = tuple(item.cell_id for item in self.cells).index(cell_id)
        return self.outer_source_cells[index]

    def inner_cells_by_objective_id(
        self, cell_id: str
    ) -> tuple[StrictConvexIntersectionCellV2, ...]:
        return dict(self.inner_source_cells)[cell_id]


def _compile_continuous_yaw_objective_v2(
    problem: SemanticProblemV2_2,
    config: ContinuousYawSolverConfigV2_8,
    candidate_stage: _CompleteContinuousYawCandidateStageV2,
    *,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> _ContinuousYawObjectiveStageV2:
    checked_problem, checked_config, checked_candidate = _strict_inputs(
        problem, config, candidate_stage, intersection_budget
    )
    outer = checked_candidate.outer_allowed.cells
    if not outer:
        raise ValueError("objective stage cannot compile an empty outer domain")
    if len(outer) > checked_config.max_objective_partition_cells:
        raise StrictConvexIntersectionBudgetExhaustedV2(
            "continuous-yaw objective cell budget exhausted"
        )
    if checked_candidate.remaining_constraint_ids:
        raise _UnsupportedContinuousYawObjectiveV2(
            "objective requires a complete hard-constraint prefix"
        )
    _require_certificate_ready_objective(checked_problem)
    deep_cost = sum(
        20
        + len(cell.half_planes)
        + len(cell.closure_polygon.vertices_ccw)
        + _SQRT_COST_V2
        for cell in outer
    )
    intersection_budget.consume_domain(deep_cost)
    magnitude = _safe_slack_magnitude(checked_problem)
    cells: list[ContinuousYawObjectiveCellV2_8] = []
    proposals: list[_ContinuousYawWitnessProposalV2] = []
    inner_map: list[tuple[str, tuple[StrictConvexIntersectionCellV2, ...]]] = []
    metrics = {item.cell_id: item for item in checked_candidate.outer_cell_metrics}
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        for index, source in enumerate(outer):
            objective_id = f"cell:objective:{index:06d}"
            assigned = tuple(
                item
                for item in checked_candidate.inner_allowed.cells
                if source.contains_point(item.strict_witness)
            )
            term_bounds, slacks = _cell_term_bounds(
                checked_problem,
                source,
                metrics[source.cell_id],
                magnitude,
            )
            cell = ContinuousYawObjectiveCellV2_8(
                cell_id=objective_id,
                outer_domain_sha256=_cell_sha(source, _OUTER_DOMAIN_HASH_V2),
                inner_domain_sha256=(_inner_sha(assigned) if assigned else None),
                term_loss_bounds=term_bounds,
                constraint_slacks=slacks,
            )
            cells.append(cell)
            inner_map.append((objective_id, assigned))
            proposal = _proposal_for_inner(
                checked_problem.semantic_problem_sha256,
                checked_candidate.subject_id,
                objective_id,
                assigned,
                source,
            )
            if proposal is not None:
                proposals.append(proposal)
    usage = ContinuousYawResourceUsageV2_8(
        domain_operations=intersection_budget.domain_operations_used,
        so2_atomic_steps=checked_candidate.resource_usage.so2_atomic_steps,
        candidate_cells=intersection_budget.candidate_cells_used,
        objective_partition_cells=len(cells),
    )
    return _ContinuousYawObjectiveStageV2(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        candidate_stage_sha256=checked_candidate.stage_sha256,
        cells=tuple(cells),
        proposals=tuple(proposals),
        outer_source_cells=outer,
        inner_source_cells=tuple(inner_map),
        resource_usage=usage,
    )


def _evaluate_continuous_yaw_point_v2(
    problem: SemanticProblemV2_2,
    config: ContinuousYawSolverConfigV2_8,
    candidate_stage: _CompleteContinuousYawCandidateStageV2,
    objective_stage: _ContinuousYawObjectiveStageV2,
    edit: CanonicalEditV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawWitnessEvaluationV2_8:
    checked_problem, checked_config, checked_candidate = _strict_inputs(
        problem, config, candidate_stage, intersection_budget
    )
    if type(objective_stage) is not _ContinuousYawObjectiveStageV2:
        raise TypeError("objective_stage has the wrong exact type")
    if type(edit) is not CanonicalEditV2:
        raise TypeError("edit has the wrong exact type")
    checked_edit = CanonicalEditV2.model_validate(
        edit.model_dump(mode="python", warnings="error"), strict=True
    )
    if (
        objective_stage.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
        or objective_stage.solver_config_sha256 != checked_config.config_sha256
        or objective_stage.candidate_stage_sha256 != checked_candidate.stage_sha256
    ):
        raise ValueError("objective point replay prefix is not closed")
    if (
        checked_edit.semantic_problem_sha256 != checked_problem.semantic_problem_sha256
        or checked_edit.subject_id != checked_candidate.subject_id
    ):
        raise _ContinuousYawWitnessNotProvenV2("edit identity is not closed")
    point = RationalPoint2V2(
        Fraction.from_float(checked_edit.translation_xy_m.x),
        Fraction.from_float(checked_edit.translation_xy_m.y),
    )
    if not checked_candidate.inner_allowed.contains_point(point):
        raise _ContinuousYawWitnessNotProvenV2(
            "edit was not proven inside the complete hard inner domain"
        )
    matching_proposal = tuple(
        item for item in objective_stage.proposals if item.edit == checked_edit
    )
    if matching_proposal:
        objective_id = matching_proposal[0].objective_cell_id
    else:
        objective_id = next(
            item.cell_id
            for item, source in zip(
                objective_stage.cells,
                objective_stage.outer_source_cells,
                strict=True,
            )
            if source.contains_point(point)
        )
    source = objective_stage.source_cell_by_objective_id(objective_id)
    point_domain_cost = _point_domain_cost(source)
    intersection_budget.consume_domain(point_domain_cost)
    atomic_budget.consume(_SQRT_COST_V2)
    translation = _point_translation_bounds(checked_problem, point)
    magnitude = _safe_slack_magnitude(checked_problem)
    metric_by_id = {item.cell_id: item for item in checked_candidate.outer_cell_metrics}
    relation = _relation_bounds(checked_problem)
    visibility = _visibility_bounds(checked_problem, metric_by_id[source.cell_id])
    safety, slacks = _point_safety_bounds(
        checked_problem,
        magnitude,
    )
    return ContinuousYawWitnessEvaluationV2_8(
        objective_cell_id=objective_id,
        edit=checked_edit,
        witness_loss_bounds=ObjectiveTermBoundsV2(
            translation_loss=translation,
            relation_damage_loss=relation,
            visibility_change_loss=visibility,
            safety_margin_loss=safety,
        ),
        constraint_slacks=slacks,
    )


def _strict_inputs(problem, config, candidate_stage, intersection_budget):
    if type(config) is not ContinuousYawSolverConfigV2_8:
        raise TypeError("config has the wrong exact type")
    if type(candidate_stage) is not _CompleteContinuousYawCandidateStageV2:
        raise TypeError("candidate_stage has the wrong exact type")
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget has the wrong exact type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = _strict_problem(problem)
        checked_config = ContinuousYawSolverConfigV2_8.model_validate(
            config.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_candidate = _copy_complete_stage(candidate_stage)
    if (
        checked_problem.semantic_problem_sha256
        != checked_candidate.semantic_problem_sha256
    ):
        raise ValueError("candidate problem hash is not closed")
    if checked_config.candidate_config.config_sha256 != (
        checked_candidate.compiler_config_sha256
    ):
        raise ValueError("candidate compiler config hash is not closed")
    return checked_problem, checked_config, checked_candidate


def _require_certificate_ready_objective(problem: SemanticProblemV2_2) -> None:
    objective = problem.objective
    if any(
        value != 0.0
        for value in (
            objective.relation_damage.weight,
            objective.visibility_change.weight,
            objective.safety_margin.weight,
        )
    ):
        raise _UnsupportedContinuousYawObjectiveV2(
            "continuous-yaw certificate MVP requires zero nontranslation weights"
        )
    if any(
        target.target_slack != 0.0
        for target in objective.safety_margin.aggregation.targets
    ):
        raise _UnsupportedContinuousYawObjectiveV2(
            "continuous-yaw certificate MVP requires zero safety targets"
        )


def _cell_term_bounds(problem, cell, metric, magnitude):
    translation = _translation_bounds(problem, cell)
    relation = _relation_bounds(problem)
    visibility = _visibility_bounds(problem, metric)
    safety, slacks = _safety_bounds(problem, magnitude)
    return (
        ObjectiveTermBoundsV2(
            translation_loss=translation,
            relation_damage_loss=relation,
            visibility_change_loss=visibility,
            safety_margin_loss=safety,
        ),
        slacks,
    )


def _translation_bounds(problem, cell):
    minimum_squared, maximum_squared, _ = _cell_distance_extrema(cell)
    factor = Fraction.from_float(
        problem.objective.translation.weight
    ) / Fraction.from_float(problem.objective.translation.normalizer_m)
    lower = _directed_sqrt_binary64_bounds(minimum_squared * factor**2)
    upper = _directed_sqrt_binary64_bounds(maximum_squared * factor**2)
    if lower is None or upper is None:
        raise _ContinuousYawObjectiveNumericGapV2("translation bound overflow")
    return NonNegativeIntervalV2(lower_bound=lower[0], upper_bound=upper[1])


def _point_translation_bounds(problem, point):
    squared = point.x**2 + point.y**2
    factor = Fraction.from_float(
        problem.objective.translation.weight
    ) / Fraction.from_float(problem.objective.translation.normalizer_m)
    pair = _directed_sqrt_binary64_bounds(squared * factor**2)
    if pair is None:
        raise _ContinuousYawObjectiveNumericGapV2("point translation overflow")
    return NonNegativeIntervalV2(lower_bound=pair[0], upper_bound=pair[1])


def _cell_distance_extrema(cell):
    vertices = cell.closure_polygon.vertices_ccw
    maximum = max(point.x**2 + point.y**2 for point in vertices)
    origin = RationalPoint2V2(Fraction(), Fraction())
    if all(
        plane.normal_x * origin.x + plane.normal_y * origin.y <= plane.offset
        for plane in cell.half_planes
    ):
        return Fraction(), maximum, origin
    best: tuple[Fraction, Fraction, Fraction] | None = None
    for first, second in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        dx = second.x - first.x
        dy = second.y - first.y
        denominator = dx**2 + dy**2
        t = -(first.x * dx + first.y * dy) / denominator
        t = min(Fraction(1), max(Fraction(), t))
        x = first.x + t * dx
        y = first.y + t * dy
        candidate = (x**2 + y**2, x, y)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("convex cell lost every edge")
    return best[0], maximum, RationalPoint2V2(best[1], best[2])


def _relation_bounds(problem):
    vector = tuple(
        RelationDamageBoundV2(key=item.key, lower_bound=0.0, upper_bound=1.0)
        for item in problem.objective.relation_damage.pair_axis_weights
    )
    return _nonnegative_interval(
        aggregate_relation_damage_bounds_v2(problem.objective.relation_damage, vector)
    )


def _visibility_bounds(problem, metric):
    observations = {
        (
            item.object_id,
            item.camera_id,
            item.metric_definition_id,
            item.metric_definition_version,
        ): item
        for item in problem.scene.baseline_observations.values or ()
    }
    intervals = []
    for weighted in problem.objective.visibility_change.object_camera_weights:
        key = weighted.key
        observation = observations.get(key.sort_key)
        if observation is None:
            raise _UnsupportedContinuousYawObjectiveV2(
                "visibility objective baseline is incomplete"
            )
        if key.object_id == problem.constraints.allowed_edit.subject_id:
            candidate_lower, candidate_upper = _metric_interval(
                metric, key.metric_definition_id
            )
        else:
            candidate_lower, candidate_upper = Fraction(), Fraction(1)
        intervals.append(
            VisibilityMetricIntervalV2(
                key=key,
                baseline_lower=Fraction.from_float(observation.normalized_lower_bound),
                baseline_upper=Fraction.from_float(observation.normalized_upper_bound),
                candidate_lower=candidate_lower,
                candidate_upper=candidate_upper,
            )
        )
    return _nonnegative_interval(
        aggregate_visibility_change_bounds_v2(
            problem.objective.visibility_change, tuple(intervals)
        )
    )


def _metric_interval(metric, definition_id):
    if definition_id == "visibility:visible-surface-fraction":
        return metric.visible_fraction_lower, metric.visible_fraction_upper
    if definition_id == "visibility:image-area-fraction":
        return metric.image_area_fraction_lower, metric.image_area_fraction_upper
    if definition_id == "visibility:truncated-fraction":
        return metric.truncated_fraction_lower, metric.truncated_fraction_upper
    raise _UnsupportedContinuousYawObjectiveV2(
        "visibility objective metric definition is unsupported"
    )


def _safety_bounds(problem, magnitude):
    constraints = []
    slacks = []
    exact_magnitude = Fraction.from_float(magnitude)
    for target in problem.objective.safety_margin.aggregation.targets:
        components = tuple(
            SafetyRawComponentIntervalV2(
                kind=component.kind,
                unit=component.unit,
                raw_lower=-exact_magnitude * Fraction.from_float(component.normalizer),
                raw_upper=exact_magnitude * Fraction.from_float(component.normalizer),
            )
            for component in target.components
        )
        value = ConstraintSafetyInputV2(target.constraint_id, components)
        constraints.append(value)
        published = compile_constraint_slack_bounds_v2(target, components)
        interval = _exact_interval(published)
        slacks.append(
            ConstraintSlackV2(
                constraint_id=target.constraint_id,
                lower_bound=interval.lower_bound,
                upper_bound=interval.upper_bound,
                unit=SafetySlackUnitV2.DIMENSIONLESS,
            )
        )
    return (
        _nonnegative_interval(
            aggregate_safety_penalty_bounds_v2(
                problem.objective.safety_margin, tuple(constraints)
            )
        ),
        tuple(slacks),
    )


def _point_safety_bounds(problem, magnitude):
    constraints = []
    slacks = []
    exact_magnitude = Fraction.from_float(magnitude)
    for target in problem.objective.safety_margin.aggregation.targets:
        components = tuple(
            SafetyRawComponentIntervalV2(
                kind=component.kind,
                unit=component.unit,
                raw_lower=Fraction(),
                raw_upper=exact_magnitude * Fraction.from_float(component.normalizer),
            )
            for component in target.components
        )
        constraints.append(ConstraintSafetyInputV2(target.constraint_id, components))
        published = compile_constraint_slack_bounds_v2(target, components)
        interval = _exact_interval(published)
        slacks.append(
            ConstraintSlackV2(
                constraint_id=target.constraint_id,
                lower_bound=interval.lower_bound,
                upper_bound=interval.upper_bound,
                unit=SafetySlackUnitV2.DIMENSIONLESS,
            )
        )
    return (
        _nonnegative_interval(
            aggregate_safety_penalty_bounds_v2(
                problem.objective.safety_margin, tuple(constraints)
            )
        ),
        tuple(slacks),
    )


def _nonnegative_interval(outcome):
    interval = _exact_interval(outcome)
    if interval.lower_bound < 0:
        raise RuntimeError(
            "non-negative objective helper published a negative lower bound"
        )
    return NonNegativeIntervalV2(
        lower_bound=interval.lower_bound,
        upper_bound=interval.upper_bound,
    )


def _exact_interval(outcome):
    if outcome.kind is not ObjectiveNumericKindV2.EXACT or outcome.interval is None:
        raise _ContinuousYawObjectiveNumericGapV2(
            ",".join(outcome.finding_codes) or "objective numeric helper failed"
        )
    return outcome.interval


def _safe_slack_magnitude(problem):
    values: list[Fraction] = []

    def visit(value):
        if type(value) is float:
            values.append(abs(Fraction.from_float(value)))
        elif type(value) is int and not isinstance(value, bool):
            values.append(abs(Fraction(value)))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(problem.model_dump(mode="python"))
    base = Fraction(1) + sum(values, Fraction())
    base += sum((1 / item for item in values if item > 0), Fraction())
    exact = (base * (len(values) + 1)) ** 4
    try:
        published = float(exact)
    except OverflowError as error:
        raise _ContinuousYawObjectiveNumericGapV2("safety envelope overflow") from error
    if not math.isfinite(published):
        raise _ContinuousYawObjectiveNumericGapV2("safety envelope overflow")
    if Fraction.from_float(published) < exact:
        published = math.nextafter(published, math.inf)
    if not math.isfinite(published) or published > float_info.max:
        raise _ContinuousYawObjectiveNumericGapV2("safety envelope overflow")
    return published


def _proposal_for_inner(problem_sha, subject_id, objective_id, assigned, outer):
    candidates = []
    for cell in assigned:
        vertices = cell.closure_polygon.vertices_ccw
        centroid = RationalPoint2V2(
            sum((item.x for item in vertices), Fraction()) / len(vertices),
            sum((item.y for item in vertices), Fraction()) / len(vertices),
        )
        for exact in (centroid, cell.strict_witness):
            x = float(exact.x)
            y = float(exact.y)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            binary = RationalPoint2V2(Fraction.from_float(x), Fraction.from_float(y))
            if cell.contains_point(binary) and outer.contains_point(binary):
                candidates.append((binary.x**2 + binary.y**2, binary.x, binary.y, x, y))
    if not candidates:
        return None
    _, _, _, x, y = min(candidates)
    return _ContinuousYawWitnessProposalV2(
        objective_cell_id=objective_id,
        edit=CanonicalEditV2(
            semantic_problem_sha256=problem_sha,
            subject_id=subject_id,
            translation_xy_m=Vec2V2(x=x, y=y),
        ),
    )


def _cell_sha(cell, domain):
    return hashlib.sha256(domain + _artifact_bytes(cell)).hexdigest()


def _inner_sha(cells):
    payload = b"".join(_artifact_bytes(item) + b"\0" for item in cells)
    return hashlib.sha256(_INNER_DOMAIN_HASH_V2 + payload).hexdigest()


def _point_domain_cost(cell):
    return 8 + len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)


__all__ = ()
