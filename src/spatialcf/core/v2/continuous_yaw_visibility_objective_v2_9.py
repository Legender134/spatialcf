"""Visibility-change objective bounds for the competition v2.9 chain."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum

from spatialcf.core.v2.continuous_yaw_safety_v2_9 import (
    ContinuousYawSafetyStageV2_9,
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
    ObjectiveNumericKindV2,
    VisibilityMetricIntervalV2,
    aggregate_visibility_change_bounds_v2,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
)
from spatialcf.domain.v2.artifacts import NonNegativeIntervalV2
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-visibility-objective.v2.9\0"


class ContinuousYawVisibilityObjectiveKindV2_9(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityObjectiveCellV2_9:
    source_cell_id: str
    metrics: tuple[VisibilityMetricIntervalV2, ...]
    loss: NonNegativeIntervalV2

    def __post_init__(self) -> None:
        if type(self.source_cell_id) is not str or not self.source_cell_id:
            raise TypeError("source_cell_id must be a non-empty exact string")
        if type(self.metrics) is not tuple or not self.metrics:
            raise TypeError("metrics must be a non-empty exact tuple")
        metrics = tuple(
            VisibilityMetricIntervalV2(
                key=item.key,
                baseline_lower=item.baseline_lower,
                baseline_upper=item.baseline_upper,
                candidate_lower=item.candidate_lower,
                candidate_upper=item.candidate_upper,
            )
            for item in self.metrics
        )
        if tuple(item.key.sort_key for item in metrics) != tuple(
            sorted(item.key.sort_key for item in metrics)
        ):
            raise ValueError("visibility objective keys must be canonically sorted")
        if len({item.key.sort_key for item in metrics}) != len(metrics):
            raise ValueError("visibility objective keys must be unique")
        if type(self.loss) is not NonNegativeIntervalV2:
            raise TypeError("loss must be a NonNegativeIntervalV2")
        loss = NonNegativeIntervalV2.model_validate(
            self.loss.model_dump(mode="python", warnings="error"), strict=True
        )
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "loss", loss)


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityObjectiveStageV2_9:
    semantic_problem_sha256: str
    visibility_stage_sha256: str
    safety_stage_sha256: str
    cells: tuple[ContinuousYawVisibilityObjectiveCellV2_9, ...]
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2

    def __post_init__(self) -> None:
        for label, value in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("visibility_stage_sha256", self.visibility_stage_sha256),
            ("safety_stage_sha256", self.safety_stage_sha256),
        ):
            _require_digest(label, value)
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("visibility objective stage requires non-empty cells")
        cells = tuple(_copy_cell(item) for item in self.cells)
        ids = tuple(item.source_cell_id for item in cells)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("visibility objective cell IDs must be canonical")
        first_keys = tuple(item.key for item in cells[0].metrics)
        if any(
            tuple(item.key for item in cell.metrics) != first_keys for cell in cells
        ):
            raise ValueError("visibility objective key universe changed across cells")
        usage = _copy_usage(self.resource_usage)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "resource_usage", usage)

    @property
    def stage_sha256(self) -> str:
        digest = hashlib.sha256(_STAGE_HASH_DOMAIN_V2_9)
        for value in (
            self.semantic_problem_sha256,
            self.visibility_stage_sha256,
            self.safety_stage_sha256,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        for cell in self.cells:
            digest.update(cell.source_cell_id.encode())
            digest.update(repr(cell.metrics).encode())
            digest.update(repr(cell.loss.model_dump(mode="python")).encode())
        digest.update(repr(self.resource_usage).encode())
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityObjectiveOutcomeV2_9:
    kind: ContinuousYawVisibilityObjectiveKindV2_9
    stage: ContinuousYawVisibilityObjectiveStageV2_9 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawVisibilityObjectiveKindV2_9:
            raise TypeError("kind must be an exact visibility objective kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ContinuousYawVisibilityObjectiveKindV2_9.STAGE:
            if type(self.stage) is not ContinuousYawVisibilityObjectiveStageV2_9:
                raise ValueError("STAGE requires one exact visibility objective stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError(
                "visibility objective failure requires findings and no stage"
            )


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityObjectivePointV2_9:
    covering_cell_ids: tuple[str, ...]
    metrics: tuple[VisibilityMetricIntervalV2, ...]
    loss: NonNegativeIntervalV2

    def __post_init__(self) -> None:
        if (
            type(self.covering_cell_ids) is not tuple
            or not self.covering_cell_ids
            or any(type(item) is not str or not item for item in self.covering_cell_ids)
        ):
            raise TypeError("covering_cell_ids must be a non-empty exact tuple")
        ids = tuple(sorted(set(self.covering_cell_ids)))
        cell = ContinuousYawVisibilityObjectiveCellV2_9(
            source_cell_id=ids[0], metrics=self.metrics, loss=self.loss
        )
        object.__setattr__(self, "covering_cell_ids", ids)
        object.__setattr__(self, "metrics", cell.metrics)
        object.__setattr__(self, "loss", cell.loss)


class _UnsupportedVisibilityObjectiveV2_9(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def compile_continuous_yaw_visibility_objective_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    safety_stage: ContinuousYawSafetyStageV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawVisibilityObjectiveOutcomeV2_9:
    """Aggregate all weighted baseline-to-candidate visibility deltas."""

    try:
        problem, visibility_stage, safety_stage = _strict_inputs(
            problem,
            visibility_stage,
            safety_stage,
            atomic_budget,
            intersection_budget,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            cells = tuple(
                _compile_cell(problem, visibility_stage, item.source_cell_id)
                for item in safety_stage.cells
            )
            intersection_budget.consume_domain(
                len(cells)
                * (4 + len(problem.objective.visibility_change.object_camera_weights))
            )
            stage = ContinuousYawVisibilityObjectiveStageV2_9(
                semantic_problem_sha256=problem.semantic_problem_sha256,
                visibility_stage_sha256=visibility_stage.stage_sha256,
                safety_stage_sha256=safety_stage.stage_sha256,
                cells=cells,
                resource_usage=_usage(atomic_budget, intersection_budget),
            )
        return ContinuousYawVisibilityObjectiveOutcomeV2_9(
            kind=ContinuousYawVisibilityObjectiveKindV2_9.STAGE,
            stage=stage,
        )
    except _UnsupportedVisibilityObjectiveV2_9 as error:
        return _failure(
            ContinuousYawVisibilityObjectiveKindV2_9.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            ContinuousYawVisibilityObjectiveKindV2_9.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_VISIBILITY_OBJECTIVE_V2_9",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ContinuousYawVisibilityObjectiveKindV2_9.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_VISIBILITY_OBJECTIVE_V2_9",
        )


def evaluate_continuous_yaw_visibility_objective_point_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    safety_stage: ContinuousYawSafetyStageV2_9,
    point: RationalPoint2V2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
    cumulative_usage: MultiObstacleStrictConvexCandidateResourceUsageV2 | None = None,
) -> ContinuousYawVisibilityObjectivePointV2_9:
    """Evaluate one point by hulling every closed outer cell that contains it."""

    problem, visibility_stage, _ = _strict_inputs(
        problem,
        visibility_stage,
        safety_stage,
        atomic_budget,
        intersection_budget,
        cumulative_usage=cumulative_usage,
    )
    if type(point) is not RationalPoint2V2:
        raise TypeError("point must be an exact RationalPoint2V2")
    matching = tuple(
        cell
        for cell in visibility_stage.outer_allowed.cells
        if cell.contains_point(point)
    )
    if not matching:
        raise ValueError("visibility objective point escaped the hard outer")
    intersection_budget.consume_domain(
        3 + sum(len(item.half_planes) for item in matching)
    )
    per_cell = tuple(
        _compile_cell(problem, visibility_stage, item.cell_id) for item in matching
    )
    metrics = tuple(
        VisibilityMetricIntervalV2(
            key=source.key,
            baseline_lower=min(item.metrics[index].baseline_lower for item in per_cell),
            baseline_upper=max(item.metrics[index].baseline_upper for item in per_cell),
            candidate_lower=min(
                item.metrics[index].candidate_lower for item in per_cell
            ),
            candidate_upper=max(
                item.metrics[index].candidate_upper for item in per_cell
            ),
        )
        for index, source in enumerate(per_cell[0].metrics)
    )
    return ContinuousYawVisibilityObjectivePointV2_9(
        covering_cell_ids=tuple(item.cell_id for item in matching),
        metrics=metrics,
        loss=_aggregate(problem, metrics),
    )


def _compile_cell(problem, visibility, cell_id):
    baseline = {item.object_id: item for item in visibility.baseline_metrics}
    candidate = {
        item.object_id: item
        for item in visibility.outer_cell_metrics
        if item.cell_id == cell_id
    }
    constraint = problem.constraints.visibility_constraints[0]
    metrics = []
    for weighted in problem.objective.visibility_change.object_camera_weights:
        key = weighted.key
        if key.camera_id != constraint.camera_id:
            raise _UnsupportedVisibilityObjectiveV2_9(
                "UNSUPPORTED_MODEL:VISIBILITY_OBJECTIVE_CAMERA_V2_9"
            )
        try:
            before = baseline[key.object_id]
            after = candidate[key.object_id]
        except KeyError as error:
            raise RuntimeError("visibility objective lost an object metric") from error
        baseline_interval, candidate_interval = _metric_intervals(
            key.metric_definition_id,
            key.metric_definition_version,
            constraint,
            before,
            after,
        )
        metrics.append(
            VisibilityMetricIntervalV2(
                key=key,
                baseline_lower=baseline_interval[0],
                baseline_upper=baseline_interval[1],
                candidate_lower=candidate_interval[0],
                candidate_upper=candidate_interval[1],
            )
        )
    checked = tuple(metrics)
    return ContinuousYawVisibilityObjectiveCellV2_9(
        source_cell_id=cell_id,
        metrics=checked,
        loss=_aggregate(problem, checked),
    )


def _metric_intervals(definition_id, version, constraint, before, after):
    choices = {
        (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
        ): (
            before.visible_fraction,
            (after.visible_fraction_lower, after.visible_fraction_upper),
        ),
        (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
        ): (
            before.image_area_fraction,
            (after.image_area_fraction_lower, after.image_area_fraction_upper),
        ),
        (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
        ): (
            before.truncated_fraction,
            (after.truncated_fraction_lower, after.truncated_fraction_upper),
        ),
    }
    try:
        return choices[(definition_id, version)]
    except KeyError as error:
        raise _UnsupportedVisibilityObjectiveV2_9(
            "UNSUPPORTED_MODEL:VISIBILITY_OBJECTIVE_METRIC_V2_9"
        ) from error


def _aggregate(problem, metrics):
    outcome = aggregate_visibility_change_bounds_v2(
        problem.objective.visibility_change, metrics
    )
    if outcome.kind is ObjectiveNumericKindV2.NUMERIC_GAP:
        raise ArithmeticError("visibility objective numeric gap")
    if outcome.kind is not ObjectiveNumericKindV2.EXACT or outcome.interval is None:
        raise RuntimeError("visibility objective failed strict numeric closure")
    if outcome.interval.lower_bound < 0:
        raise RuntimeError("visibility objective published a negative lower bound")
    return NonNegativeIntervalV2(
        lower_bound=outcome.interval.lower_bound,
        upper_bound=outcome.interval.upper_bound,
    )


def _strict_inputs(
    problem, visibility, safety, atomic, domain, *, cumulative_usage=None
):
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(visibility) is not ContinuousYawVisibilityStageV2_9:
        raise TypeError("visibility_stage must be an exact v2.9 stage")
    if type(safety) is not ContinuousYawSafetyStageV2_9:
        raise TypeError("safety_stage must be an exact v2.9 stage")
    if type(atomic) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    if type(domain) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    usage = safety.resource_usage if cumulative_usage is None else cumulative_usage
    if type(usage) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("cumulative_usage has the wrong exact type")
    if (
        atomic.used != usage.so2_atomic_steps
        or domain.domain_operations_used != usage.domain_operations
        or domain.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("visibility objective ledgers must continue safety usage")
    domain.consume_domain(10)
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
            resource_usage=visibility.resource_usage,
            remaining_constraint_ids=visibility.remaining_constraint_ids,
            unsat_prefix_eligible=visibility.unsat_prefix_eligible,
        )
        checked_safety = ContinuousYawSafetyStageV2_9(
            semantic_problem_sha256=safety.semantic_problem_sha256,
            visibility_stage_sha256=safety.visibility_stage_sha256,
            relation_stage_sha256=safety.relation_stage_sha256,
            cells=safety.cells,
            resource_usage=safety.resource_usage,
        )
    if (
        checked_problem.semantic_problem_sha256
        != checked_visibility.semantic_problem_sha256
        or checked_problem.semantic_problem_sha256
        != checked_safety.semantic_problem_sha256
        or checked_safety.visibility_stage_sha256 != checked_visibility.stage_sha256
    ):
        raise ValueError("visibility objective prefix hashes are not closed")
    if tuple(item.source_cell_id for item in checked_safety.cells) != tuple(
        item.cell_id for item in checked_visibility.outer_allowed.cells
    ):
        raise ValueError("visibility objective cells do not close the hard outer")
    return checked_problem, checked_visibility, checked_safety


def _copy_cell(value):
    if type(value) is not ContinuousYawVisibilityObjectiveCellV2_9:
        raise TypeError("visibility objective cell has the wrong exact type")
    return ContinuousYawVisibilityObjectiveCellV2_9(
        source_cell_id=value.source_cell_id,
        metrics=value.metrics,
        loss=value.loss,
    )


def _copy_stage(value):
    return ContinuousYawVisibilityObjectiveStageV2_9(
        semantic_problem_sha256=value.semantic_problem_sha256,
        visibility_stage_sha256=value.visibility_stage_sha256,
        safety_stage_sha256=value.safety_stage_sha256,
        cells=value.cells,
        resource_usage=value.resource_usage,
    )


def _copy_usage(value):
    if type(value) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("visibility objective usage has the wrong exact type")
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


def _require_digest(label, value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(kind, finding):
    return ContinuousYawVisibilityObjectiveOutcomeV2_9(
        kind=kind, finding_codes=(finding,)
    )


__all__ = (
    "ContinuousYawVisibilityObjectiveCellV2_9",
    "ContinuousYawVisibilityObjectiveKindV2_9",
    "ContinuousYawVisibilityObjectiveOutcomeV2_9",
    "ContinuousYawVisibilityObjectivePointV2_9",
    "ContinuousYawVisibilityObjectiveStageV2_9",
    "compile_continuous_yaw_visibility_objective_v2_9",
    "evaluate_continuous_yaw_visibility_objective_point_v2_9",
)
