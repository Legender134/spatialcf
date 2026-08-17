"""Public raw-input minimum-cost solve for competition Canonical v2.9."""

from __future__ import annotations

import warnings

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    compile_upright_camera_context_v2_9,
    prepare_camera_independent_candidate_problem_v2_9,
)
from spatialcf.core.v2.continuous_yaw_certificate_v2_9 import (
    ContinuousYawCertificateNotProvenV2_9,
    assemble_continuous_yaw_certificate_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    DirectionalTargetCandidateKindV2_9,
    compile_directional_target_candidate_v2_9,
)
from spatialcf.core.v2.continuous_yaw_objective_v2_9 import (
    ContinuousYawObjectiveNotProvenV2_9,
    compile_continuous_yaw_objective_v2_9,
    evaluate_continuous_yaw_objective_point_v2_9,
)
from spatialcf.core.v2.continuous_yaw_relation_damage import (
    ContinuousYawRelationDamageKindV2_9,
    compile_relation_damage_bounds_v2_9,
)
from spatialcf.core.v2.continuous_yaw_safety_v2_9 import (
    ContinuousYawSafetyKindV2_9,
    compile_continuous_yaw_safety_bounds_v2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_objective_v2_9 import (
    ContinuousYawVisibilityObjectiveKindV2_9,
    compile_continuous_yaw_visibility_objective_v2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_v2_9 import (
    ContinuousYawVisibilityKindV2_9,
    compile_continuous_yaw_visibility_v2_9,
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
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCandidateRefsV2_9,
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawMinimumCostSolveOutcomeV2_9,
    ContinuousYawProvenUnsatResultV2_9,
    ContinuousYawResourceUsageV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawUncertifiedResultV2_9,
)
from spatialcf.domain.v2.result import UncertifiedReasonV2


def solve_continuous_yaw_minimum_cost_v2_9(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
) -> ContinuousYawMinimumCostSolveOutcomeV2_9:
    """Run the complete v2.9 hard-domain, objective, witness, certificate chain."""

    checked = _strict_public_inputs(problem, config)
    if checked is None:
        return ContinuousYawMinimumCostSolveOutcomeV2_9(
            result=None,
            finding_codes=("INVALID_INPUT:CONTINUOUS_YAW_SOLVE_V2_9",),
        )
    problem, config = checked
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            projected = prepare_camera_independent_candidate_problem_v2_9(problem)
            t15 = compile_support_strict_convex_candidate_domain_v2_7(
                projected, config.candidate_config
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            problem,
            projected if "projected" in locals() else None,
            config,
            None,
            (),
            None,
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_T15_REPLAY_V2_9",),
        )
    if t15.kind is not SupportStrictConvexCandidateCompilationKindV2.ARTIFACT:
        return _uncertified(
            problem,
            projected,
            config,
            None,
            (),
            None,
            _reason(t15.kind.value),
            t15.finding_codes,
        )
    if t15.artifact is None:
        raise RuntimeError("T15 ARTIFACT outcome omitted its artifact")
    artifact = t15.artifact
    atomic = SO2AtomicBudgetV2(
        limit=config.candidate_config.max_so2_atomic_steps,
        used=artifact.resource_usage.so2_atomic_steps,
    )
    domain = StrictConvexIntersectionBudgetV2(
        max_domain_operations=config.candidate_config.max_domain_operations,
        max_candidate_cells=config.candidate_config.max_candidate_cells,
        domain_operations_used=artifact.resource_usage.domain_operations,
        candidate_cells_used=artifact.resource_usage.candidate_cells,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            camera = compile_upright_camera_context_v2_9(
                problem, atomic_budget=atomic, domain_budget=domain
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            problem,
            projected,
            config,
            None,
            (),
            _usage(atomic, domain),
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_CAMERA_V2_9",),
        )
    refs = ContinuousYawCandidateRefsV2_9(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        candidate_problem_sha256=projected.semantic_problem_sha256,
        solver_config_sha256=config.config_sha256,
        camera_context_sha256=camera.context_sha256,
        t15_candidate_artifact_sha256=artifact.artifact_sha256,
    )
    if not artifact.allowed_domain_bracket.outer_allowed.cells:
        return _unsat(
            problem,
            projected,
            config,
            refs,
            "T15",
            artifact.artifact_sha256,
            atomic,
            domain,
        )

    try:
        target = compile_directional_target_candidate_v2_9(
            problem,
            projected,
            artifact,
            camera,
            atomic_budget=atomic,
            intersection_budget=domain,
        )
    except (ArithmeticError, RuntimeWarning):
        return _numeric_failure(
            problem, projected, config, refs, atomic, domain, "TARGET"
        )
    if target.kind is not DirectionalTargetCandidateKindV2_9.STAGE:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            target.kind.value,
            target.finding_codes,
            atomic,
            domain,
        )
    if target.stage is None:
        raise RuntimeError("target STAGE outcome omitted its stage")
    target_stage = target.stage
    refs = _refs(refs, target=target_stage.stage_sha256)
    if not target_stage.outer_allowed.cells:
        return _unsat(
            problem,
            projected,
            config,
            refs,
            "TARGET_RELATION",
            target_stage.stage_sha256,
            atomic,
            domain,
        )

    visibility = compile_continuous_yaw_visibility_v2_9(
        problem,
        target_stage,
        camera,
        atomic_budget=atomic,
        intersection_budget=domain,
    )
    if visibility.kind is not ContinuousYawVisibilityKindV2_9.STAGE:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            visibility.kind.value,
            visibility.finding_codes,
            atomic,
            domain,
        )
    if visibility.stage is None:
        raise RuntimeError("visibility STAGE outcome omitted its stage")
    visibility_stage = visibility.stage
    refs = _refs(refs, visibility=visibility_stage.stage_sha256)
    if visibility_stage.unsat_prefix_eligible:
        return _unsat(
            problem,
            projected,
            config,
            refs,
            "VISIBILITY",
            visibility_stage.stage_sha256,
            atomic,
            domain,
        )

    relation = compile_relation_damage_bounds_v2_9(
        problem,
        visibility_stage,
        atomic_budget=atomic,
        intersection_budget=domain,
    )
    if relation.kind is not ContinuousYawRelationDamageKindV2_9.STAGE:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            relation.kind.value,
            relation.finding_codes,
            atomic,
            domain,
        )
    if relation.stage is None:
        raise RuntimeError("relation STAGE outcome omitted its stage")
    safety = compile_continuous_yaw_safety_bounds_v2_9(
        problem,
        visibility_stage,
        relation.stage,
        atomic_budget=atomic,
        intersection_budget=domain,
    )
    if safety.kind is not ContinuousYawSafetyKindV2_9.STAGE:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            safety.kind.value,
            safety.finding_codes,
            atomic,
            domain,
        )
    if safety.stage is None:
        raise RuntimeError("safety STAGE outcome omitted its stage")
    visible = compile_continuous_yaw_visibility_objective_v2_9(
        problem,
        visibility_stage,
        safety.stage,
        atomic_budget=atomic,
        intersection_budget=domain,
    )
    if visible.kind is not ContinuousYawVisibilityObjectiveKindV2_9.STAGE:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            visible.kind.value,
            visible.finding_codes,
            atomic,
            domain,
        )
    if visible.stage is None:
        raise RuntimeError("visibility objective STAGE omitted its stage")
    try:
        objective = compile_continuous_yaw_objective_v2_9(
            problem,
            config,
            visibility_stage,
            relation.stage,
            safety.stage,
            visible.stage,
            atomic_budget=atomic,
            intersection_budget=domain,
        )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _stage_failure(
            problem,
            projected,
            config,
            refs,
            "RESOURCE_LIMIT",
            ("RESOURCE_LIMIT:CONTINUOUS_YAW_OBJECTIVE_V2_9",),
            atomic,
            domain,
        )
    except (ArithmeticError, RuntimeWarning):
        return _numeric_failure(
            problem, projected, config, refs, atomic, domain, "OBJECTIVE"
        )

    proposal_count = len(objective.proposals)
    if proposal_count == 0:
        return _uncertified(
            problem,
            projected,
            config,
            refs,
            objective.cells,
            _usage(atomic, domain, len(objective.cells)),
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            ("COMPILATION_INCOMPLETE:CONTINUOUS_YAW_WITNESS_COVERAGE_V2_9",),
            proposal_count=proposal_count,
        )
    evaluations = []
    for proposal in objective.proposals:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                evaluation = evaluate_continuous_yaw_objective_point_v2_9(
                    problem,
                    config,
                    visibility_stage,
                    relation.stage,
                    safety.stage,
                    visible.stage,
                    objective,
                    proposal.edit,
                    atomic_budget=atomic,
                    intersection_budget=domain,
                )
        except ContinuousYawObjectiveNotProvenV2_9 as error:
            raise RuntimeError("fresh objective proposal lost feasibility") from error
        except (
            StrictConvexIntersectionBudgetExhaustedV2,
            SO2AtomicBudgetExhaustedV2,
        ):
            return _uncertified(
                problem,
                projected,
                config,
                refs,
                objective.cells,
                _usage(atomic, domain, len(objective.cells)),
                UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
                ("RESOURCE_LIMIT:CONTINUOUS_YAW_POINT_V2_9",),
                proposal_count=proposal_count,
                evaluated_count=len(evaluations),
            )
        except (ArithmeticError, RuntimeWarning):
            return _uncertified(
                problem,
                projected,
                config,
                refs,
                objective.cells,
                _usage(atomic, domain, len(objective.cells)),
                UncertifiedReasonV2.NUMERIC_GAP,
                ("NUMERIC_GAP:CONTINUOUS_YAW_POINT_V2_9",),
                proposal_count=proposal_count,
                evaluated_count=len(evaluations),
            )
        evaluations.append(evaluation)
    final_usage = _usage(atomic, domain, len(objective.cells))
    try:
        certificate, selected = assemble_continuous_yaw_certificate_v2_9(
            config, refs, objective, tuple(evaluations), final_usage
        )
    except ContinuousYawCertificateNotProvenV2_9:
        return _uncertified(
            problem,
            projected,
            config,
            refs,
            objective.cells,
            final_usage,
            UncertifiedReasonV2.NUMERIC_GAP,
            ("NUMERIC_GAP:CONTINUOUS_YAW_OPTIMALITY_GAP_V2_9",),
            proposal_count=proposal_count,
            evaluated_count=len(evaluations),
        )
    result = ContinuousYawCertifiedSuccessResultV2_9(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        candidate_problem_sha256=projected.semantic_problem_sha256,
        solver_config=config,
        candidate_refs=refs,
        objective_cells=objective.cells,
        selected_witness=selected,
        global_loss_lower_bound=objective.global_loss_lower_bound,
        witness_loss_bounds=selected.witness_loss_bounds,
        final_resource_usage=final_usage,
        certificate=certificate,
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_9(
        result=result,
        cumulative_generation_usage=final_usage,
        proposal_count=proposal_count,
        evaluated_proposal_count=len(evaluations),
    )


class ContinuousYawMinimumCostSolverV2_9:
    @staticmethod
    def solve(problem, config):
        return solve_continuous_yaw_minimum_cost_v2_9(problem, config)


def _strict_public_inputs(problem, config):
    if (
        type(problem) is not SemanticProblemV2_3
        or type(config) is not ContinuousYawSolverConfigV2_9
    ):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return (
                SemanticProblemV2_3.model_validate(
                    problem.model_dump(mode="python", warnings="error"), strict=True
                ),
                ContinuousYawSolverConfigV2_9.model_validate(
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


def _refs(refs, *, target=None, visibility=None):
    return ContinuousYawCandidateRefsV2_9(
        semantic_problem_sha256=refs.semantic_problem_sha256,
        candidate_problem_sha256=refs.candidate_problem_sha256,
        solver_config_sha256=refs.solver_config_sha256,
        camera_context_sha256=refs.camera_context_sha256,
        t15_candidate_artifact_sha256=refs.t15_candidate_artifact_sha256,
        target_candidate_stage_sha256=target or refs.target_candidate_stage_sha256,
        visibility_candidate_stage_sha256=visibility
        or refs.visibility_candidate_stage_sha256,
    )


def _usage(atomic, domain, objective_cells=0):
    return ContinuousYawResourceUsageV2_9(
        domain_operations=domain.domain_operations_used,
        so2_atomic_steps=atomic.used,
        candidate_cells=domain.candidate_cells_used,
        objective_partition_cells=objective_cells,
    )


def _unsat(problem, projected, config, refs, stage, digest, atomic, domain):
    usage = _usage(atomic, domain)
    result = ContinuousYawProvenUnsatResultV2_9(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        candidate_problem_sha256=projected.semantic_problem_sha256,
        solver_config=config,
        candidate_refs=refs,
        empty_outer_stage=stage,
        empty_outer_stage_sha256=digest,
        final_resource_usage=usage,
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_9(
        result=result, cumulative_generation_usage=usage
    )


def _uncertified(
    problem,
    projected,
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
    if projected is None:
        return ContinuousYawMinimumCostSolveOutcomeV2_9(
            result=None, finding_codes=tuple(findings)
        )
    result = ContinuousYawUncertifiedResultV2_9(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        candidate_problem_sha256=projected.semantic_problem_sha256,
        solver_config=config,
        uncertified_reason=reason,
        candidate_refs=refs,
        objective_cells=tuple(cells),
        final_resource_usage=usage,
        finding_codes=tuple(findings),
    )
    return ContinuousYawMinimumCostSolveOutcomeV2_9(
        result=result,
        finding_codes=result.finding_codes,
        cumulative_generation_usage=usage,
        proposal_count=proposal_count,
        evaluated_proposal_count=evaluated_count,
    )


def _reason(kind):
    return {
        "RESOURCE_LIMIT": UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
        "NUMERIC_GAP": UncertifiedReasonV2.NUMERIC_GAP,
        "UNSUPPORTED_MODEL": UncertifiedReasonV2.UNSUPPORTED_MODEL,
    }.get(kind, UncertifiedReasonV2.COMPILATION_INCOMPLETE)


def _stage_failure(problem, projected, config, refs, kind, findings, atomic, domain):
    return _uncertified(
        problem,
        projected,
        config,
        refs,
        (),
        _usage(atomic, domain),
        _reason(kind),
        findings,
    )


def _numeric_failure(problem, projected, config, refs, atomic, domain, label):
    return _uncertified(
        problem,
        projected,
        config,
        refs,
        (),
        _usage(atomic, domain),
        UncertifiedReasonV2.NUMERIC_GAP,
        (f"NUMERIC_GAP:CONTINUOUS_YAW_{label}_V2_9",),
    )


__all__ = (
    "ContinuousYawMinimumCostSolverV2_9",
    "solve_continuous_yaw_minimum_cost_v2_9",
)
