"""Global certificate assembly for the competition v2.9 solve."""

from __future__ import annotations

from fractions import Fraction
from sys import float_info

from spatialcf.core.v2.continuous_yaw_objective_v2_9 import (
    ContinuousYawObjectiveStageV2_9,
)
from spatialcf.domain.v2.certificate import OptimalityClaimV2
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCandidateRefsV2_9,
    ContinuousYawGlobalOptimalityCertificateV2_9,
    ContinuousYawResourceUsageV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawWitnessEvaluationV2_9,
    _directed_binary64_gap_ceil,
)


class ContinuousYawCertificateNotProvenV2_9(ValueError):
    pass


def assemble_continuous_yaw_certificate_v2_9(
    config: ContinuousYawSolverConfigV2_9,
    refs: ContinuousYawCandidateRefsV2_9,
    objective: ContinuousYawObjectiveStageV2_9,
    evaluations: tuple[ContinuousYawWitnessEvaluationV2_9, ...],
    final_usage: ContinuousYawResourceUsageV2_9,
) -> tuple[
    ContinuousYawGlobalOptimalityCertificateV2_9,
    ContinuousYawWitnessEvaluationV2_9,
]:
    """Select the lowest fresh witness and publish the directed global gap."""

    if type(config) is not ContinuousYawSolverConfigV2_9:
        raise TypeError("config has the wrong exact type")
    if type(refs) is not ContinuousYawCandidateRefsV2_9:
        raise TypeError("refs have the wrong exact type")
    if type(objective) is not ContinuousYawObjectiveStageV2_9:
        raise TypeError("objective has the wrong exact type")
    if type(evaluations) is not tuple or not evaluations:
        raise ContinuousYawCertificateNotProvenV2_9(
            "certificate requires fresh witness evaluations"
        )
    checked = tuple(
        ContinuousYawWitnessEvaluationV2_9.model_validate(
            item.model_dump(mode="python", warnings="error"), strict=True
        )
        for item in evaluations
    )
    expected = {item.edit.edit_sha256 for item in objective.proposals}
    actual = {item.edit.edit_sha256 for item in checked}
    if actual != expected or len(checked) != len(expected):
        raise ContinuousYawCertificateNotProvenV2_9(
            "certificate requires every proposal exactly once"
        )
    selected = min(checked, key=_selection_key)
    lower = objective.global_loss_lower_bound
    upper = selected.witness_loss_bounds.total_upper_bound
    if lower > upper:
        raise RuntimeError("fresh witness upper fell below the global lower")
    exact_gap = Fraction.from_float(upper) - Fraction.from_float(lower)
    if exact_gap > Fraction.from_float(float_info.max):
        raise ContinuousYawCertificateNotProvenV2_9("optimality gap is not finite")
    gap = _directed_binary64_gap_ceil(lower, upper)
    if gap > config.target_optimality_gap:
        raise ContinuousYawCertificateNotProvenV2_9(
            "optimality gap exceeds configured target"
        )
    claim = OptimalityClaimV2.EXACT if gap == 0.0 else OptimalityClaimV2.EPSILON_OPTIMAL
    certificate = ContinuousYawGlobalOptimalityCertificateV2_9(
        semantic_problem_sha256=refs.semantic_problem_sha256,
        candidate_problem_sha256=refs.candidate_problem_sha256,
        solver_config_sha256=config.config_sha256,
        candidate_refs_sha256=refs.candidate_refs_sha256,
        objective_cells_sha256=objective.objective_cells_sha256,
        witness_evaluation_sha256=selected.witness_evaluation_sha256,
        edit_sha256=selected.edit.edit_sha256,
        loss_lower_bound=lower,
        loss_upper_bound=upper,
        optimality_gap=gap,
        optimality_claim=claim,
        epsilon=None if gap == 0.0 else gap,
        final_resource_usage=final_usage,
    )
    return certificate, selected


def _selection_key(item: ContinuousYawWitnessEvaluationV2_9):
    terms = item.witness_loss_bounds
    x = Fraction.from_float(item.edit.translation_xy_m.x)
    y = Fraction.from_float(item.edit.translation_xy_m.y)
    return (
        terms.total_upper_bound,
        terms.translation_loss.upper_bound,
        terms.relation_damage_loss.upper_bound,
        terms.visibility_change_loss.upper_bound,
        terms.safety_margin_loss.upper_bound,
        x**2 + y**2,
        x,
        y,
        item.edit.edit_sha256,
    )


__all__ = (
    "ContinuousYawCertificateNotProvenV2_9",
    "assemble_continuous_yaw_certificate_v2_9",
)
