"""Joint four-term objective partition for the competition v2.9 chain."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction

from spatialcf.core.v2.continuous_yaw_objective import (
    _cell_distance_extrema,
    _cell_sha,
    _inner_sha,
)
from spatialcf.core.v2.continuous_yaw_relation_damage import (
    ContinuousYawRelationDamageStageV2_9,
    evaluate_relation_damage_point_v2_9,
)
from spatialcf.core.v2.continuous_yaw_safety_v2_9 import (
    ContinuousYawSafetyStageV2_9,
    evaluate_continuous_yaw_safety_point_v2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_objective_v2_9 import (
    ContinuousYawVisibilityObjectiveStageV2_9,
    evaluate_continuous_yaw_visibility_objective_point_v2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_v2_9 import (
    ContinuousYawVisibilityStageV2_9,
)
from spatialcf.core.v2.convex_translation_domain import RationalPoint2V2
from spatialcf.core.v2.objective_numeric import (
    _directed_sqrt_binary64_bounds,
    aggregate_relation_damage_bounds_v2,
)
from spatialcf.core.v2.so2_interval import SO2AtomicBudgetV2
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionKindV2,
    intersect_strict_convex_allowed_complexes_v2,
)
from spatialcf.domain.v2.artifacts import NonNegativeIntervalV2, ObjectiveTermBoundsV2
from spatialcf.domain.v2.base import Vec2V2
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawObjectiveCellV2_9,
    ContinuousYawResourceUsageV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawWitnessEvaluationV2_9,
)
from spatialcf.domain.v2.edit import CanonicalEditV2

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-joint-objective.v2.9\0"
_OUTER_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-outer-cell.v2.9\0"
_INWARD_PROPOSAL_STEPS_V2_9 = 64


class ContinuousYawObjectiveNotProvenV2_9(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ContinuousYawObjectiveProposalV2_9:
    objective_cell_id: str
    edit: CanonicalEditV2

    def __post_init__(self) -> None:
        if type(self.objective_cell_id) is not str or not self.objective_cell_id:
            raise TypeError("objective_cell_id must be a non-empty exact string")
        if type(self.edit) is not CanonicalEditV2:
            raise TypeError("edit must be an exact CanonicalEditV2")
        object.__setattr__(
            self,
            "edit",
            CanonicalEditV2.model_validate(
                self.edit.model_dump(mode="python", warnings="error"), strict=True
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousYawObjectiveStageV2_9:
    semantic_problem_sha256: str
    solver_config_sha256: str
    visibility_stage_sha256: str
    relation_stage_sha256: str
    safety_stage_sha256: str
    visibility_objective_stage_sha256: str
    cells: tuple[ContinuousYawObjectiveCellV2_9, ...]
    proposals: tuple[_ContinuousYawObjectiveProposalV2_9, ...]
    outer_source_cells: tuple[StrictConvexIntersectionCellV2, ...]
    inner_source_cells: tuple[
        tuple[str, tuple[StrictConvexIntersectionCellV2, ...]], ...
    ]
    resource_usage: ContinuousYawResourceUsageV2_9

    def __post_init__(self) -> None:
        for label, value in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("solver_config_sha256", self.solver_config_sha256),
            ("visibility_stage_sha256", self.visibility_stage_sha256),
            ("relation_stage_sha256", self.relation_stage_sha256),
            ("safety_stage_sha256", self.safety_stage_sha256),
            (
                "visibility_objective_stage_sha256",
                self.visibility_objective_stage_sha256,
            ),
        ):
            _require_digest(label, value)
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("objective stage requires non-empty cells")
        cells = tuple(
            ContinuousYawObjectiveCellV2_9.model_validate(
                item.model_dump(mode="python", warnings="error"), strict=True
            )
            for item in self.cells
        )
        if tuple(item.cell_id for item in cells) != tuple(
            sorted(item.cell_id for item in cells)
        ):
            raise ValueError("objective cells must be canonically sorted")
        if type(self.proposals) is not tuple:
            raise TypeError("proposals must be an exact tuple")
        proposals = tuple(
            _ContinuousYawObjectiveProposalV2_9(item.objective_cell_id, item.edit)
            for item in self.proposals
        )
        if len({item.objective_cell_id for item in proposals}) != len(proposals):
            raise ValueError("objective proposals must be unique per cell")
        if type(self.outer_source_cells) is not tuple or len(
            self.outer_source_cells
        ) != len(cells):
            raise ValueError("objective source cells do not close published cells")
        if type(self.inner_source_cells) is not tuple or tuple(
            item[0] for item in self.inner_source_cells
        ) != tuple(item.cell_id for item in cells):
            raise ValueError("objective inner mapping is not closed")
        if type(self.resource_usage) is not ContinuousYawResourceUsageV2_9:
            raise TypeError("resource_usage has the wrong exact type")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "proposals", proposals)

    @property
    def global_loss_lower_bound(self) -> float:
        return min(item.term_loss_bounds.total_lower_bound for item in self.cells)

    @property
    def objective_cells_sha256(self) -> str:
        return ContinuousYawObjectiveCellV2_9.sequence_sha256(self.cells)

    @property
    def stage_sha256(self) -> str:
        payload = (
            self.semantic_problem_sha256,
            self.solver_config_sha256,
            self.visibility_stage_sha256,
            self.relation_stage_sha256,
            self.safety_stage_sha256,
            self.visibility_objective_stage_sha256,
            self.objective_cells_sha256,
            tuple(
                (item.objective_cell_id, item.edit.edit_sha256)
                for item in self.proposals
            ),
            self.resource_usage.model_dump(mode="python"),
        )
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2_9 + repr(payload).encode()
        ).hexdigest()

    def source_cell_by_objective_id(
        self, cell_id: str
    ) -> StrictConvexIntersectionCellV2:
        index = tuple(item.cell_id for item in self.cells).index(cell_id)
        return self.outer_source_cells[index]


def compile_continuous_yaw_objective_v2_9(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    relation_stage: ContinuousYawRelationDamageStageV2_9,
    safety_stage: ContinuousYawSafetyStageV2_9,
    visibility_objective_stage: ContinuousYawVisibilityObjectiveStageV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawObjectiveStageV2_9:
    """Build one real four-term objective enclosure per complete outer cell."""

    checked = _strict_inputs(
        problem,
        config,
        visibility_stage,
        relation_stage,
        safety_stage,
        visibility_objective_stage,
        atomic_budget,
        intersection_budget,
    )
    problem, config, visibility, relation, safety, visible = checked
    if len(relation.cells) > config.max_objective_partition_cells:
        raise StrictConvexIntersectionBudgetExhaustedV2
    intersection_budget.consume_domain(
        len(relation.cells) * len(visibility.inner_allowed.cells)
    )
    assignments = tuple(
        tuple(
            item
            for item in visibility.inner_allowed.cells
            if relation_cell.cell.contains_point(item.strict_witness)
        )
        for relation_cell in relation.cells
    )
    intersection_budget.consume_domain(
        sum(
            24
            + len(item.cell.half_planes)
            + len(item.cell.closure_polygon.vertices_ccw)
            for item in relation.cells
        )
        + sum(
            len(cell.half_planes)
            + len(cell.closure_polygon.vertices_ccw)
            + 2 * (_INWARD_PROPOSAL_STEPS_V2_9 + 3)
            for assigned in assignments
            for cell in assigned
        )
    )
    visible_by_id = {item.source_cell_id: item for item in visible.cells}
    safety_by_id = {item.source_cell_id: item for item in safety.cells}
    cells = []
    proposals = []
    inner_map = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        for index, (relation_cell, initial_assignment) in enumerate(
            zip(relation.cells, assignments, strict=True)
        ):
            source = relation_cell.cell
            objective_id = f"cell:objective:{index:06d}"
            assigned = initial_assignment
            proposal = _nearest_binary64_proposal_v2_9(
                problem.semantic_problem_sha256,
                visibility.subject_id,
                objective_id,
                assigned,
                source,
            )
            if proposal is None:
                assigned = _exact_inner_assignment_fallback_v2_9(
                    visibility.inner_allowed,
                    source,
                    intersection_budget,
                )
                intersection_budget.consume_domain(
                    sum(
                        len(item.half_planes)
                        + len(item.closure_polygon.vertices_ccw)
                        + 2 * (_INWARD_PROPOSAL_STEPS_V2_9 + 3)
                        for item in assigned
                    )
                )
                proposal = _nearest_binary64_proposal_v2_9(
                    problem.semantic_problem_sha256,
                    visibility.subject_id,
                    objective_id,
                    assigned,
                    source,
                )
            translation = _translation_bounds(problem, source)
            relation_loss = _relation_loss(problem, relation_cell.vector)
            visibility_loss = visible_by_id[source.cell_id].loss
            safety_loss = safety_by_id[source.cell_id].safety_loss
            cell = ContinuousYawObjectiveCellV2_9(
                cell_id=objective_id,
                outer_domain_sha256=_cell_sha(source, _OUTER_HASH_DOMAIN_V2_9),
                inner_domain_sha256=_inner_sha(assigned) if assigned else None,
                term_loss_bounds=ObjectiveTermBoundsV2(
                    translation_loss=translation,
                    relation_damage_loss=relation_loss,
                    visibility_change_loss=visibility_loss,
                    safety_margin_loss=safety_loss,
                ),
                constraint_slacks=safety_by_id[source.cell_id].constraint_slacks,
            )
            cells.append(cell)
            inner_map.append((objective_id, assigned))
            if proposal is not None:
                proposals.append(proposal)
    return ContinuousYawObjectiveStageV2_9(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solver_config_sha256=config.config_sha256,
        visibility_stage_sha256=visibility.stage_sha256,
        relation_stage_sha256=relation.stage_sha256,
        safety_stage_sha256=safety.stage_sha256,
        visibility_objective_stage_sha256=visible.stage_sha256,
        cells=tuple(cells),
        proposals=tuple(proposals),
        outer_source_cells=tuple(item.cell for item in relation.cells),
        inner_source_cells=tuple(inner_map),
        resource_usage=_usage(
            atomic_budget, intersection_budget, objective_cells=len(cells)
        ),
    )


def _exact_inner_assignment_fallback_v2_9(
    inner: StrictConvexIntersectionComplexV2,
    outer: StrictConvexIntersectionCellV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[StrictConvexIntersectionCellV2, ...]:
    """Intersect hard-inner cells with one outer cell after witness assignment fails.

    The fast path assigns an inner cell when its existing strict witness lies in
    the outer cell.  That test is sufficient but not necessary for overlap.  A
    failed proposal therefore requires one exact intersection before publishing
    witness-coverage failure.  Keeping this as a fallback preserves every
    already-successful v2.9 proposal and its identity.
    """

    outer_complex = StrictConvexIntersectionComplexV2(
        cells=(outer,),
        universe=inner.universe,
        topology=inner.topology,
    )
    outcome = intersect_strict_convex_allowed_complexes_v2(
        (inner, outer_complex), budget=budget
    )
    if outcome.kind is StrictConvexIntersectionKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is StrictConvexIntersectionKindV2.NUMERIC_GAP:
        raise ArithmeticError("objective inner/outer intersection numeric gap")
    if outcome.kind is StrictConvexIntersectionKindV2.INVALID_INPUT:
        raise RuntimeError("objective compiler produced invalid intersection operands")
    if outcome.complex is None:
        raise RuntimeError("objective inner/outer intersection lost its complex")
    return outcome.complex.cells


def evaluate_continuous_yaw_objective_point_v2_9(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    relation_stage: ContinuousYawRelationDamageStageV2_9,
    safety_stage: ContinuousYawSafetyStageV2_9,
    visibility_objective_stage: ContinuousYawVisibilityObjectiveStageV2_9,
    objective_stage: ContinuousYawObjectiveStageV2_9,
    edit: CanonicalEditV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawWitnessEvaluationV2_9:
    """Fresh point replay for translation, relation, visibility, and safety."""

    problem, config, visibility, relation, safety, _visible = _strict_inputs(
        problem,
        config,
        visibility_stage,
        relation_stage,
        safety_stage,
        visibility_objective_stage,
        atomic_budget,
        intersection_budget,
        expected_objective=objective_stage,
    )
    if type(edit) is not CanonicalEditV2:
        raise TypeError("edit must be an exact CanonicalEditV2")
    edit = CanonicalEditV2.model_validate(
        edit.model_dump(mode="python", warnings="error"), strict=True
    )
    if (
        edit.semantic_problem_sha256 != problem.semantic_problem_sha256
        or edit.subject_id != visibility.subject_id
    ):
        raise ContinuousYawObjectiveNotProvenV2_9("edit identity is not closed")
    point = RationalPoint2V2(
        x=Fraction.from_float(edit.translation_xy_m.x),
        y=Fraction.from_float(edit.translation_xy_m.y),
    )
    if not visibility.inner_allowed.contains_point(point):
        raise ContinuousYawObjectiveNotProvenV2_9("edit lies outside the hard inner")
    matching = tuple(
        item
        for item in objective_stage.outer_source_cells
        if item.contains_point(point)
    )
    if not matching:
        raise RuntimeError("hard-inner point escaped the objective outer")
    proposal = next(
        (item for item in objective_stage.proposals if item.edit == edit), None
    )
    objective_id = (
        proposal.objective_cell_id
        if proposal is not None
        else objective_stage.cells[
            objective_stage.outer_source_cells.index(matching[0])
        ].cell_id
    )
    intersection_budget.consume_domain(
        10 + sum(len(item.half_planes) for item in matching)
    )
    translation = _point_translation(problem, point)
    relation_vector = evaluate_relation_damage_point_v2_9(
        problem,
        visibility,
        point,
        atomic_budget=atomic_budget,
        intersection_budget=intersection_budget,
    )
    relation_loss = _relation_loss(problem, relation_vector)
    visibility_point = evaluate_continuous_yaw_visibility_objective_point_v2_9(
        problem,
        visibility,
        safety,
        point,
        atomic_budget=atomic_budget,
        intersection_budget=intersection_budget,
        cumulative_usage=_base_usage(atomic_budget, intersection_budget),
    )
    safety_point = evaluate_continuous_yaw_safety_point_v2_9(
        problem,
        visibility,
        relation,
        point,
        atomic_budget=atomic_budget,
        intersection_budget=intersection_budget,
        cumulative_usage=_base_usage(atomic_budget, intersection_budget),
    )
    return ContinuousYawWitnessEvaluationV2_9(
        objective_cell_id=objective_id,
        edit=edit,
        witness_loss_bounds=ObjectiveTermBoundsV2(
            translation_loss=translation,
            relation_damage_loss=relation_loss,
            visibility_change_loss=visibility_point.loss,
            safety_margin_loss=safety_point.safety_loss,
        ),
        constraint_slacks=safety_point.constraint_slacks,
    )


def _translation_bounds(problem, cell):
    minimum, maximum, _ = _cell_distance_extrema(cell)
    factor = Fraction.from_float(
        problem.objective.translation.weight
    ) / Fraction.from_float(problem.objective.translation.normalizer_m)
    lower = _directed_sqrt_binary64_bounds(minimum * factor**2)
    upper = _directed_sqrt_binary64_bounds(maximum * factor**2)
    if lower is None or upper is None:
        raise ArithmeticError("translation objective overflow")
    return NonNegativeIntervalV2(lower_bound=lower[0], upper_bound=upper[1])


def _point_translation(problem, point):
    factor = Fraction.from_float(
        problem.objective.translation.weight
    ) / Fraction.from_float(problem.objective.translation.normalizer_m)
    pair = _directed_sqrt_binary64_bounds((point.x**2 + point.y**2) * factor**2)
    if pair is None:
        raise ArithmeticError("point translation objective overflow")
    return NonNegativeIntervalV2(lower_bound=pair[0], upper_bound=pair[1])


def _nearest_binary64_proposal_v2_9(
    problem_sha: str,
    subject_id: str,
    objective_id: str,
    assigned: tuple[StrictConvexIntersectionCellV2, ...],
    outer: StrictConvexIntersectionCellV2,
) -> _ContinuousYawObjectiveProposalV2_9 | None:
    """Choose the nearest proven-feasible binary64 point for one outer cell.

    The exact nearest closure point may lie on a strict boundary or may not be
    representable as binary64.  For each assigned hard-inner cell, walk a
    frozen dyadic sequence from that optimum toward its strict witness, then
    independently recheck the serialized binary64 point against both domains.
    Centroid and strict-witness fallbacks preserve coverage for very narrow
    cells.  The caller precharges every bounded candidate and membership pass.
    """

    candidates: list[tuple[Fraction, Fraction, Fraction, float, float]] = []
    for cell in assigned:
        _, _, nearest = _cell_distance_extrema(cell)
        vertices = cell.closure_polygon.vertices_ccw
        centroid = RationalPoint2V2(
            sum((item.x for item in vertices), Fraction()) / len(vertices),
            sum((item.y for item in vertices), Fraction()) / len(vertices),
        )
        exact_candidates = [nearest]
        for exponent in range(1, _INWARD_PROPOSAL_STEPS_V2_9 + 1):
            fraction = Fraction(1, 1 << exponent)
            exact_candidates.append(
                RationalPoint2V2(
                    x=nearest.x + (cell.strict_witness.x - nearest.x) * fraction,
                    y=nearest.y + (cell.strict_witness.y - nearest.y) * fraction,
                )
            )
        exact_candidates.extend((centroid, cell.strict_witness))
        for exact in exact_candidates:
            try:
                x = float(exact.x)
                y = float(exact.y)
            except OverflowError:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            binary = RationalPoint2V2(
                x=Fraction.from_float(x), y=Fraction.from_float(y)
            )
            if cell.contains_point(binary) and outer.contains_point(binary):
                candidates.append((binary.x**2 + binary.y**2, binary.x, binary.y, x, y))
    if not candidates:
        return None
    _, _, _, x, y = min(candidates)
    return _ContinuousYawObjectiveProposalV2_9(
        objective_cell_id=objective_id,
        edit=CanonicalEditV2(
            semantic_problem_sha256=problem_sha,
            subject_id=subject_id,
            translation_xy_m=Vec2V2(x=x, y=y),
        ),
    )


def _relation_loss(problem, vector):
    outcome = aggregate_relation_damage_bounds_v2(
        problem.objective.relation_damage, tuple(vector)
    )
    if outcome.kind.value == "NUMERIC_GAP":
        raise ArithmeticError("relation objective numeric gap")
    if outcome.kind.value != "EXACT" or outcome.interval is None:
        raise RuntimeError("relation objective failed numeric closure")
    if outcome.interval.lower_bound < 0:
        raise RuntimeError("relation objective published a negative lower")
    return NonNegativeIntervalV2(
        lower_bound=outcome.interval.lower_bound,
        upper_bound=outcome.interval.upper_bound,
    )


def _strict_inputs(
    problem,
    config,
    visibility,
    relation,
    safety,
    visible,
    atomic,
    domain,
    *,
    expected_objective=None,
):
    types = (
        (problem, SemanticProblemV2_3, "problem"),
        (config, ContinuousYawSolverConfigV2_9, "config"),
        (visibility, ContinuousYawVisibilityStageV2_9, "visibility"),
        (relation, ContinuousYawRelationDamageStageV2_9, "relation"),
        (safety, ContinuousYawSafetyStageV2_9, "safety"),
        (visible, ContinuousYawVisibilityObjectiveStageV2_9, "visibility objective"),
        (atomic, SO2AtomicBudgetV2, "atomic budget"),
        (domain, StrictConvexIntersectionBudgetV2, "intersection budget"),
    )
    for value, expected, label in types:
        if type(value) is not expected:
            raise TypeError(f"{label} has the wrong exact type")
    usage = (
        visible.resource_usage
        if expected_objective is None
        else expected_objective.resource_usage
    )
    if (
        atomic.used < usage.so2_atomic_steps
        or domain.domain_operations_used < usage.domain_operations
        or domain.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("objective ledgers rolled back below their fresh prefix")
    if expected_objective is None and (
        atomic.used != usage.so2_atomic_steps
        or domain.domain_operations_used != usage.domain_operations
    ):
        raise ValueError("objective compilation must start at exact prefix usage")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = SemanticProblemV2_3.model_validate(
            problem.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_config = ContinuousYawSolverConfigV2_9.model_validate(
            config.model_dump(mode="python", warnings="error"), strict=True
        )
    if any(
        digest != checked_problem.semantic_problem_sha256
        for digest in (
            visibility.semantic_problem_sha256,
            relation.semantic_problem_sha256,
            safety.semantic_problem_sha256,
            visible.semantic_problem_sha256,
        )
    ):
        raise ValueError("objective problem hashes are not closed")
    if relation.visibility_stage_sha256 != visibility.stage_sha256:
        raise ValueError("relation stage is not visibility-closed")
    if safety.relation_stage_sha256 != relation.stage_sha256:
        raise ValueError("safety stage is not relation-closed")
    if visible.safety_stage_sha256 != safety.stage_sha256:
        raise ValueError("visibility objective is not safety-closed")
    if expected_objective is not None:
        if type(expected_objective) is not ContinuousYawObjectiveStageV2_9:
            raise TypeError("objective_stage has the wrong exact type")
        if (
            expected_objective.semantic_problem_sha256
            != checked_problem.semantic_problem_sha256
            or expected_objective.solver_config_sha256 != checked_config.config_sha256
            or expected_objective.visibility_objective_stage_sha256
            != visible.stage_sha256
        ):
            raise ValueError("objective point prefix hashes are not closed")
    return checked_problem, checked_config, visibility, relation, safety, visible


def _usage(atomic, domain, *, objective_cells):
    return ContinuousYawResourceUsageV2_9(
        domain_operations=domain.domain_operations_used,
        so2_atomic_steps=atomic.used,
        candidate_cells=domain.candidate_cells_used,
        objective_partition_cells=objective_cells,
    )


def _base_usage(atomic, domain):
    from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
        MultiObstacleStrictConvexCandidateResourceUsageV2,
    )

    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=domain.domain_operations_used,
        so2_atomic_steps=atomic.used,
        candidate_cells=domain.candidate_cells_used,
    )


def _require_digest(label, value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = (
    "ContinuousYawObjectiveNotProvenV2_9",
    "ContinuousYawObjectiveStageV2_9",
    "compile_continuous_yaw_objective_v2_9",
    "evaluate_continuous_yaw_objective_point_v2_9",
)
