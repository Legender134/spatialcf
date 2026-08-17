"""Private T18 certificate assembly for the continuous-yaw solve."""

from __future__ import annotations

import warnings
from fractions import Fraction
from sys import float_info

from spatialcf.core.v2.continuous_yaw_objective import (
    _ContinuousYawObjectiveStageV2,
    _point_domain_cost,
)
from spatialcf.core.v2.continuous_yaw_visibility import (
    _CompleteContinuousYawCandidateStageV2,
)
from spatialcf.domain.v2.certificate import OptimalityClaimV2
from spatialcf.domain.v2.continuous_yaw_candidate import SemanticProblemV2_2
from spatialcf.domain.v2.continuous_yaw_solver import (
    ContinuousYawCandidateRefsV2_8,
    ContinuousYawGlobalOptimalityCertificateV2_8,
    ContinuousYawResourceUsageV2_8,
    ContinuousYawSolverConfigV2_8,
    ContinuousYawWitnessEvaluationV2_8,
    _directed_binary64_gap_ceil,
)


class _CertificateNotProvenV2(ValueError):
    pass


def _assemble_continuous_yaw_certificate_v2(
    problem: SemanticProblemV2_2,
    config: ContinuousYawSolverConfigV2_8,
    candidate_stage: _CompleteContinuousYawCandidateStageV2,
    objective_stage: _ContinuousYawObjectiveStageV2,
    evaluations: tuple[ContinuousYawWitnessEvaluationV2_8, ...],
) -> ContinuousYawGlobalOptimalityCertificateV2_8:
    checked_problem, checked_config = _strict_inputs(problem, config)
    if type(candidate_stage) is not _CompleteContinuousYawCandidateStageV2:
        raise TypeError("candidate_stage has the wrong exact type")
    if type(objective_stage) is not _ContinuousYawObjectiveStageV2:
        raise TypeError("objective_stage has the wrong exact type")
    if type(evaluations) is not tuple or not evaluations:
        raise _CertificateNotProvenV2("certificate requires evaluated proposals")
    checked_evaluations = tuple(
        ContinuousYawWitnessEvaluationV2_8.model_validate(
            item.model_dump(mode="python", warnings="error"), strict=True
        )
        for item in evaluations
    )
    expected_edits = {item.edit.edit_sha256 for item in objective_stage.proposals}
    actual_edits = {item.edit.edit_sha256 for item in checked_evaluations}
    if actual_edits != expected_edits or len(checked_evaluations) != len(
        expected_edits
    ):
        raise _CertificateNotProvenV2(
            "certificate requires every canonical proposal exactly once"
        )
    if (
        candidate_stage.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
        or objective_stage.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
        or objective_stage.solver_config_sha256 != checked_config.config_sha256
        or objective_stage.candidate_stage_sha256 != candidate_stage.stage_sha256
    ):
        raise ValueError("certificate replay prefix is not closed")
    selected = min(checked_evaluations, key=_selection_key)
    lower = objective_stage.global_loss_lower_bound
    upper = selected.witness_loss_bounds.total_upper_bound
    if lower > upper:
        raise RuntimeError("fresh witness upper fell below the global lower")
    exact_gap = Fraction.from_float(upper) - Fraction.from_float(lower)
    if exact_gap > Fraction.from_float(float_info.max):
        raise _CertificateNotProvenV2("directed optimality gap is not finite")
    gap = _directed_binary64_gap_ceil(lower, upper)
    if gap > checked_config.target_optimality_gap:
        raise _CertificateNotProvenV2(
            "directed optimality gap exceeds the configured target"
        )
    if gap == 0.0:
        claim = OptimalityClaimV2.EXACT
        epsilon = None
    else:
        claim = OptimalityClaimV2.EPSILON_OPTIMAL
        epsilon = gap
    final_usage = _final_usage(objective_stage, checked_evaluations)
    if (
        final_usage.domain_operations
        > checked_config.candidate_config.max_domain_operations
        or final_usage.so2_atomic_steps
        > checked_config.candidate_config.max_so2_atomic_steps
        or final_usage.candidate_cells
        > checked_config.candidate_config.max_candidate_cells
        or final_usage.objective_partition_cells
        > checked_config.max_objective_partition_cells
    ):
        raise _CertificateNotProvenV2("final resource usage exceeds solver policy")
    refs = ContinuousYawCandidateRefsV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        t15_candidate_artifact_sha256=candidate_stage.upstream_t15_artifact_sha256,
        target_candidate_stage_sha256=candidate_stage.upstream_target_stage_sha256,
        visibility_candidate_stage_sha256=candidate_stage.stage_sha256,
    )
    return ContinuousYawGlobalOptimalityCertificateV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        candidate_refs_sha256=refs.candidate_refs_sha256,
        objective_cells_sha256=objective_stage.objective_cells_sha256,
        witness_evaluation_sha256=selected.witness_evaluation_sha256,
        edit_sha256=selected.edit.edit_sha256,
        loss_lower_bound=lower,
        loss_upper_bound=upper,
        optimality_gap=gap,
        optimality_claim=claim,
        epsilon=epsilon,
        final_resource_usage=final_usage,
    )


def _strict_inputs(problem, config):
    if type(problem) is not SemanticProblemV2_2:
        raise TypeError("problem has the wrong exact type")
    if type(config) is not ContinuousYawSolverConfigV2_8:
        raise TypeError("config has the wrong exact type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        return (
            SemanticProblemV2_2.model_validate(
                problem.model_dump(mode="python", warnings="error"), strict=True
            ),
            ContinuousYawSolverConfigV2_8.model_validate(
                config.model_dump(mode="python", warnings="error"), strict=True
            ),
        )


def _selection_key(item: ContinuousYawWitnessEvaluationV2_8):
    x = Fraction.from_float(item.edit.translation_xy_m.x)
    y = Fraction.from_float(item.edit.translation_xy_m.y)
    return (
        x**2 + y**2,
        item.witness_loss_bounds.total_upper_bound,
        item.edit.translation_xy_m.x,
        item.edit.translation_xy_m.y,
        item.edit.edit_sha256,
    )


def _final_usage(objective_stage, evaluations):
    domain = objective_stage.resource_usage.domain_operations
    so2 = objective_stage.resource_usage.so2_atomic_steps
    for evaluation in evaluations:
        source = objective_stage.source_cell_by_objective_id(
            evaluation.objective_cell_id
        )
        domain += _point_domain_cost(source)
        so2 += 128
    return ContinuousYawResourceUsageV2_8(
        domain_operations=domain,
        so2_atomic_steps=so2,
        candidate_cells=objective_stage.resource_usage.candidate_cells,
        objective_partition_cells=len(objective_stage.cells),
    )


__all__ = ()
