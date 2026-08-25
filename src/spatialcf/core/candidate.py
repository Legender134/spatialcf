"""Public current candidate-domain compilation boundary."""

from __future__ import annotations

import warnings

from spatialcf.core._internal.compilation.support import (
    SupportStrictConvexCandidateCompilationKindV2,
    SupportStrictConvexCandidateCompilationOutcomeV2,
    compile_support_strict_convex_candidate_domain_v2_7,
)
from spatialcf.core.problem import (
    prepare_camera_independent_candidate_problem,
)
from spatialcf.domain.constraints import (
    AllowedPositionDomainConstraint,
    CollisionConstraint,
    SupportConstraint,
    TargetRelationConstraint,
    VisibilityConstraint,
)
from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9


def compile_candidate_domain(
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    """Compile the current camera-independent T15 candidate prefix."""

    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(config) is not ContinuousYawSolverConfigV2_9:
        raise TypeError("config must be an exact ContinuousYawSolverConfigV2_9")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = SemanticProblemV2_3.model_validate(
            problem.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_config = ContinuousYawSolverConfigV2_9.model_validate(
            config.model_dump(mode="python", warnings="error"), strict=True
        )
        projected = prepare_camera_independent_candidate_problem(checked_problem)
        outcome = compile_support_strict_convex_candidate_domain_v2_7(
            projected, checked_config.candidate_config
        )
    if outcome.kind is SupportStrictConvexCandidateCompilationKindV2.ARTIFACT:
        if outcome.artifact is None:
            raise RuntimeError("T15 ARTIFACT outcome omitted its artifact")
        _require_current_family_partition(
            checked_problem,
            outcome.artifact.ordered_constraint_ids,
            outcome.artifact.remaining_constraint_ids,
        )
    return outcome


def _require_current_family_partition(
    checked_problem: SemanticProblemV2_3,
    ordered_constraint_ids: tuple[str, ...],
    remaining_constraint_ids: tuple[str, ...],
) -> None:
    constraints = checked_problem.constraints
    if (
        type(constraints.position_domain) is not AllowedPositionDomainConstraint
        or any(
            type(constraint) is not CollisionConstraint
            for constraint in constraints.collision_constraints
        )
        or any(
            type(constraint) is not SupportConstraint
            for constraint in constraints.support_constraints
        )
        or type(constraints.target_relation) is not TargetRelationConstraint
        or any(
            type(constraint) is not VisibilityConstraint
            for constraint in constraints.visibility_constraints
        )
    ):
        raise RuntimeError("current candidate family order changed")

    expected_ordered = (
        constraints.position_domain.constraint_id,
        *(item.constraint_id for item in constraints.collision_constraints),
        *(item.constraint_id for item in constraints.support_constraints),
    )
    expected_remaining = tuple(
        sorted(
            (
                constraints.target_relation.constraint_id,
                *(item.constraint_id for item in constraints.visibility_constraints),
            )
        )
    )
    if (
        ordered_constraint_ids != expected_ordered
        or remaining_constraint_ids != expected_remaining
    ):
        raise RuntimeError("current candidate family order changed")


__all__ = ["compile_candidate_domain"]
