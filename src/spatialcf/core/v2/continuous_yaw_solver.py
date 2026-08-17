"""Public raw-input minimum-cost solve for Canonical continuous yaw v2.8."""

from __future__ import annotations

import warnings

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_certificate import (
    _assemble_continuous_yaw_certificate_v2,
    _CertificateNotProvenV2,
)
from spatialcf.core.v2.continuous_yaw_objective import (
    _compile_continuous_yaw_objective_v2,
    _ContinuousYawObjectiveNumericGapV2,
    _ContinuousYawWitnessNotProvenV2,
    _evaluate_continuous_yaw_point_v2,
    _UnsupportedContinuousYawObjectiveV2,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _compile_target_aware_candidate_v2,
    _TargetAwareCandidateKindV2,
)
from spatialcf.core.v2.continuous_yaw_visibility import (
    _compile_complete_continuous_yaw_candidate_v2,
    _CompleteContinuousYawCandidateKindV2,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
)
from spatialcf.core.v2.support_strict_convex_candidate_domain import (
    SupportStrictConvexCandidateCompilationKindV2,
    compile_support_strict_convex_candidate_domain_v2_7,
)
from spatialcf.domain.v2.continuous_yaw_candidate import SemanticProblemV2_2
from spatialcf.domain.v2.continuous_yaw_solver import (
    ContinuousYawCandidateRefsV2_8,
    ContinuousYawCertifiedSuccessResultV2_8,
    ContinuousYawMinimumCostSolveOutcomeV2_8,
    ContinuousYawProvenUnsatResultV2_8,
    ContinuousYawResourceUsageV2_8,
    ContinuousYawSolverConfigV2_8,
    ContinuousYawUncertifiedResultV2_8,
)
from spatialcf.domain.v2.result import UncertifiedReasonV2


def solve_continuous_yaw_minimum_cost_v2_8(
    problem: SemanticProblemV2_2,
    config: ContinuousYawSolverConfigV2_8,
) -> ContinuousYawMinimumCostSolveOutcomeV2_8:
    """Run the T15--T18 chain once with one cumulative pair of ledgers."""

    checked = _strict_public_inputs(problem, config)
    if checked is None:
        return ContinuousYawMinimumCostSolveOutcomeV2_8(
            result=None,
            finding_codes=("INVALID_INPUT:CONTINUOUS_YAW_SOLVE",),
        )
    checked_problem, checked_config = checked

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            t15 = compile_support_strict_convex_candidate_domain_v2_7(
                checked_problem, checked_config.candidate_config
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified_without_prefix(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_T15_REPLAY",),
        )
    if t15.kind is not SupportStrictConvexCandidateCompilationKindV2.ARTIFACT:
        return _uncertified_without_prefix(
            checked_problem,
            checked_config,
            _reason_for_kind(t15.kind.value),
            t15.finding_codes,
        )
    if t15.artifact is None:
        raise RuntimeError("T15 ARTIFACT outcome omitted its artifact")
    artifact = t15.artifact
    base_usage = _usage_from_candidate(artifact.resource_usage)
    refs = ContinuousYawCandidateRefsV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        t15_candidate_artifact_sha256=artifact.artifact_sha256,
    )
    if not artifact.allowed_domain_bracket.outer_allowed.cells:
        return _unsat(
            checked_problem,
            checked_config,
            refs,
            "T15",
            artifact.artifact_sha256,
            base_usage,
        )

    atomic_budget = SO2AtomicBudgetV2(
        limit=checked_config.candidate_config.max_so2_atomic_steps,
        used=artifact.resource_usage.so2_atomic_steps,
    )
    intersection_budget = StrictConvexIntersectionBudgetV2(
        max_domain_operations=checked_config.candidate_config.max_domain_operations,
        max_candidate_cells=checked_config.candidate_config.max_candidate_cells,
        domain_operations_used=artifact.resource_usage.domain_operations,
        candidate_cells_used=artifact.resource_usage.candidate_cells,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            target = _compile_target_aware_candidate_v2(
                checked_problem,
                artifact,
                atomic_budget=atomic_budget,
                intersection_budget=intersection_budget,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_TARGET_REPLAY",),
        )
    if target.kind is not _TargetAwareCandidateKindV2.STAGE:
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            _reason_for_kind(target.kind.value),
            target.finding_codes,
        )
    if target.stage is None:
        raise RuntimeError("target STAGE outcome omitted its stage")
    target_stage = target.stage
    refs = ContinuousYawCandidateRefsV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        t15_candidate_artifact_sha256=artifact.artifact_sha256,
        target_candidate_stage_sha256=target_stage.stage_sha256,
    )
    if not target_stage.outer_allowed.cells:
        return _unsat(
            checked_problem,
            checked_config,
            refs,
            "TARGET_RELATION",
            target_stage.stage_sha256,
            _live_usage(intersection_budget, atomic_budget, 0),
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            visibility = _compile_complete_continuous_yaw_candidate_v2(
                checked_problem,
                target_stage,
                atomic_budget=atomic_budget,
                intersection_budget=intersection_budget,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_VISIBILITY_REPLAY",),
        )
    if visibility.kind is not _CompleteContinuousYawCandidateKindV2.STAGE:
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            _reason_for_kind(visibility.kind.value),
            visibility.finding_codes,
        )
    if visibility.stage is None:
        raise RuntimeError("visibility STAGE outcome omitted its stage")
    candidate = visibility.stage
    refs = ContinuousYawCandidateRefsV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        t15_candidate_artifact_sha256=artifact.artifact_sha256,
        target_candidate_stage_sha256=target_stage.stage_sha256,
        visibility_candidate_stage_sha256=candidate.stage_sha256,
    )
    if candidate.unsat_prefix_eligible:
        return _unsat(
            checked_problem,
            checked_config,
            refs,
            "VISIBILITY",
            candidate.stage_sha256,
            _live_usage(intersection_budget, atomic_budget, 0),
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            objective = _compile_continuous_yaw_objective_v2(
                checked_problem,
                checked_config,
                candidate,
                intersection_budget=intersection_budget,
            )
    except _UnsupportedContinuousYawObjectiveV2 as error:
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            (f"UNSUPPORTED_MODEL:CONTINUOUS_YAW_OBJECTIVE:{error}",),
        )
    except (StrictConvexIntersectionBudgetExhaustedV2, SO2AtomicBudgetExhaustedV2):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            ("RESOURCE_LIMIT:CONTINUOUS_YAW_OBJECTIVE",),
        )
    except (_ContinuousYawObjectiveNumericGapV2, ArithmeticError, RuntimeWarning):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            (),
            _live_usage(intersection_budget, atomic_budget, 0),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_OBJECTIVE",),
        )

    proposal_count = len(objective.proposals)
    if proposal_count != len(objective.cells):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            objective.cells,
            objective.resource_usage,
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            ("COMPILATION_INCOMPLETE:CONTINUOUS_YAW_WITNESS_COVERAGE",),
            proposal_count=proposal_count,
        )
    evaluations = []
    for proposal in objective.proposals:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                evaluation = _evaluate_continuous_yaw_point_v2(
                    checked_problem,
                    checked_config,
                    candidate,
                    objective,
                    proposal.edit,
                    atomic_budget=atomic_budget,
                    intersection_budget=intersection_budget,
                )
        except _ContinuousYawWitnessNotProvenV2 as error:
            raise RuntimeError("fresh objective proposal lost feasibility") from error
        except (StrictConvexIntersectionBudgetExhaustedV2, SO2AtomicBudgetExhaustedV2):
            return _uncertified_with_prefix(
                checked_problem,
                checked_config,
                refs,
                objective.cells,
                _live_usage(intersection_budget, atomic_budget, len(objective.cells)),
                UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
                ("RESOURCE_LIMIT:CONTINUOUS_YAW_POINT_EVALUATION",),
                proposal_count=proposal_count,
                evaluated_count=len(evaluations),
            )
        except (_ContinuousYawObjectiveNumericGapV2, ArithmeticError, RuntimeWarning):
            return _uncertified_with_prefix(
                checked_problem,
                checked_config,
                refs,
                objective.cells,
                _live_usage(intersection_budget, atomic_budget, len(objective.cells)),
                UncertifiedReasonV2.NUMERIC_GAP,
                ("NUMERIC_GAP:CONTINUOUS_YAW_POINT_EVALUATION",),
                proposal_count=proposal_count,
                evaluated_count=len(evaluations),
            )
        evaluations.append(evaluation)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            certificate = _assemble_continuous_yaw_certificate_v2(
                checked_problem,
                checked_config,
                candidate,
                objective,
                tuple(evaluations),
            )
    except _CertificateNotProvenV2:
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            objective.cells,
            _live_usage(intersection_budget, atomic_budget, len(objective.cells)),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_OPTIMALITY_GAP",),
            proposal_count=proposal_count,
            evaluated_count=len(evaluations),
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified_with_prefix(
            checked_problem,
            checked_config,
            refs,
            objective.cells,
            _live_usage(intersection_budget, atomic_budget, len(objective.cells)),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_CERTIFICATE",),
            proposal_count=proposal_count,
            evaluated_count=len(evaluations),
        )
    live_final_usage = _live_usage(
        intersection_budget, atomic_budget, len(objective.cells)
    )
    if certificate.final_resource_usage != live_final_usage:
        raise RuntimeError("certificate resource usage drifted from the live ledgers")
    selected = next(
        item
        for item in evaluations
        if item.witness_evaluation_sha256 == certificate.witness_evaluation_sha256
    )
    result = ContinuousYawCertifiedSuccessResultV2_8(
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config=checked_config,
        candidate_refs=refs,
        objective_cells=objective.cells,
        selected_witness=selected,
        global_loss_lower_bound=objective.global_loss_lower_bound,
        witness_loss_bounds=selected.witness_loss_bounds,
        final_resource_usage=live_final_usage,
        certificate=certificate,
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_8(
        result=result,
        cumulative_generation_usage=result.final_resource_usage,
        proposal_count=proposal_count,
        evaluated_proposal_count=len(evaluations),
    )


def _strict_public_inputs(problem, config):
    if type(problem) is not SemanticProblemV2_2 or type(config) is not (
        ContinuousYawSolverConfigV2_8
    ):
        return None
    try:
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
    except (
        ValidationError,
        PydanticSerializationError,
        TypeError,
        ValueError,
        Warning,
    ):
        return None


def _reason_for_kind(kind: str) -> UncertifiedReasonV2:
    if kind == "RESOURCE_LIMIT":
        return UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED
    if kind == "NUMERIC_GAP":
        return UncertifiedReasonV2.NUMERIC_GAP
    if kind == "UNSUPPORTED_MODEL":
        return UncertifiedReasonV2.UNSUPPORTED_MODEL
    return UncertifiedReasonV2.COMPILATION_INCOMPLETE


def _usage_from_candidate(usage) -> ContinuousYawResourceUsageV2_8:
    return ContinuousYawResourceUsageV2_8(
        domain_operations=usage.domain_operations,
        so2_atomic_steps=usage.so2_atomic_steps,
        candidate_cells=usage.candidate_cells,
        objective_partition_cells=0,
    )


def _live_usage(intersection_budget, atomic_budget, objective_cells):
    return ContinuousYawResourceUsageV2_8(
        domain_operations=intersection_budget.domain_operations_used,
        so2_atomic_steps=atomic_budget.used,
        candidate_cells=intersection_budget.candidate_cells_used,
        objective_partition_cells=objective_cells,
    )


def _uncertified_without_prefix(problem, config, reason, findings):
    result = ContinuousYawUncertifiedResultV2_8(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solver_config=config,
        uncertified_reason=reason,
        finding_codes=findings or ("COMPILATION_INCOMPLETE:CONTINUOUS_YAW_T15",),
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_8(
        result=result,
        finding_codes=result.finding_codes,
        cumulative_generation_usage=None,
    )


def _uncertified_with_prefix(
    problem,
    config,
    refs,
    cells,
    usage,
    reason,
    findings,
    *,
    proposal_count=0,
    evaluated_count=0,
):
    result = ContinuousYawUncertifiedResultV2_8(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solver_config=config,
        uncertified_reason=reason,
        candidate_refs=refs,
        objective_cells=cells,
        final_resource_usage=usage,
        finding_codes=findings,
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_8(
        result=result,
        finding_codes=result.finding_codes,
        cumulative_generation_usage=usage,
        proposal_count=proposal_count,
        evaluated_proposal_count=evaluated_count,
    )


def _unsat(problem, config, refs, stage_name, stage_sha, usage):
    result = ContinuousYawProvenUnsatResultV2_8(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solver_config=config,
        candidate_refs=refs,
        empty_outer_stage=stage_name,
        empty_outer_stage_sha256=stage_sha,
        final_resource_usage=usage,
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_8(
        result=result,
        cumulative_generation_usage=usage,
    )


__all__ = ("solve_continuous_yaw_minimum_cost_v2_8",)
