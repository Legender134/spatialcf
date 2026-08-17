"""Pure-core solve entry points for the Exact Cardinal Canonical 2.1 subset."""

from __future__ import annotations

import warnings
from typing import TypeVar

from spatialcf.core.v2.certificate_builder import _ExactCardinalSelectionFrameV2
from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    CandidateDomainVariableV2,
    CompilationResourceUsageV2,
    ObjectivePartitionArtifactV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.certificate import (
    GlobalOptimalityCertificateV2,
    ProvenUnsatCertificateV2,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    UncertifiedResultV2,
)

ModelT = TypeVar("ModelT", bound=V2Model)


def solve_canonical_minimum_cost_v2_1(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve one exact-cardinal problem through the common engine."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        solve_registered_capability_v2,
    )

    return solve_registered_capability_v2(
        SolveCapabilityKeyV2.V2_1,
        problem,
        config,
    )


def _solve_canonical_minimum_cost_v2_1_in_selection_frame(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Compatibility seam using the common engine with an exact frame."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        _solve_registered_capability_in_selection_frame_v2,
    )

    return _solve_registered_capability_in_selection_frame_v2(
        SolveCapabilityKeyV2.V2_1,
        problem,
        config,
        selection_frame,
    )


class CanonicalMinimumCostSolverV2_1:
    """Stateless object wrapper for the Canonical 2.1 solve."""

    def solve(
        self,
        problem: SemanticProblemV2_1,
        config: CoreSolverConfigV2,
    ) -> CanonicalMinimumCostSolveOutcomeV2:
        return solve_canonical_minimum_cost_v2_1(problem, config)


def _rebind_outcome(
    replay: CanonicalMinimumCostSolveOutcomeV2,
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    preprocessing_domain_operations: int,
    *,
    candidate_variable_override: CandidateDomainVariableV2 | None = None,
    cumulative_domain_operations_increment: int | None = None,
) -> CanonicalMinimumCostSolveOutcomeV2:
    if (
        type(preprocessing_domain_operations) is not int
        or preprocessing_domain_operations < 0
    ):
        raise TypeError("preprocessing domain operations must be non-negative")
    cumulative_increment = (
        preprocessing_domain_operations
        if cumulative_domain_operations_increment is None
        else cumulative_domain_operations_increment
    )
    if type(cumulative_increment) is not int or cumulative_increment < 0:
        raise TypeError("cumulative domain operations increment must be non-negative")
    old_result = replay.result
    if old_result is None:
        raise RuntimeError("cannot rebind a missing replay result")
    problem_hash = problem.semantic_problem_sha256
    config_hash = config.core_solver_config_sha256
    old_candidate = getattr(old_result, "candidate_domain", None)
    candidate = (
        None
        if old_candidate is None
        else _strict_internal(
            old_candidate.model_copy(
                update={
                    "semantic_problem_sha256": problem_hash,
                    "core_solver_config_sha256": config_hash,
                    "candidate_variable": (
                        old_candidate.candidate_variable
                        if candidate_variable_override is None
                        else candidate_variable_override
                    ),
                    "resource_usage": _bump_usage(
                        old_candidate.resource_usage,
                        preprocessing_domain_operations,
                    ),
                }
            ),
            CandidateDomainArtifactV2,
        )
    )
    old_relation = getattr(old_result, "relation_cost_partition", None)
    relation = None
    if old_relation is not None:
        if candidate is None:
            raise RuntimeError("relation replay omitted its candidate")
        relation = _strict_internal(
            old_relation.model_copy(
                update={
                    "semantic_problem_sha256": problem_hash,
                    "core_solver_config_sha256": config_hash,
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    ),
                }
            ),
            RelationCostPartitionV2,
        )
    old_objective = getattr(old_result, "objective_partition", None)
    objective = None
    if old_objective is not None:
        if candidate is None or relation is None:
            raise RuntimeError("objective replay omitted an upstream artifact")
        objective = _strict_internal(
            old_objective.model_copy(
                update={
                    "semantic_problem_sha256": problem_hash,
                    "core_solver_config_sha256": config_hash,
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    ),
                    "relation_cost_partition_sha256": (
                        relation.relation_cost_partition_sha256
                    ),
                }
            ),
            ObjectivePartitionArtifactV2,
        )

    if type(old_result) is CertifiedSuccessResultV2:
        if candidate is None or relation is None or objective is None:
            raise RuntimeError("success replay omitted its artifact chain")
        edit = _strict_internal(
            old_result.edit.model_copy(
                update={"semantic_problem_sha256": problem_hash}
            ),
            CanonicalEditV2,
        )
        certificate = _strict_internal(
            old_result.certificate.model_copy(
                update={
                    "semantic_problem_sha256": problem_hash,
                    "core_solver_config_sha256": config_hash,
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    ),
                    "relation_cost_partition_sha256": (
                        relation.relation_cost_partition_sha256
                    ),
                    "objective_partition_artifact_sha256": (
                        objective.objective_partition_artifact_sha256
                    ),
                    "edit_sha256": edit.edit_sha256,
                }
            ),
            GlobalOptimalityCertificateV2,
        )
        result = CertifiedSuccessResultV2(
            semantic_problem_sha256=problem_hash,
            core_solver_config=config,
            candidate_domain=candidate,
            relation_cost_partition=relation,
            objective_partition=objective,
            edit=edit,
            global_loss_lower_bound=old_result.global_loss_lower_bound,
            witness_loss_bounds=old_result.witness_loss_bounds,
            certificate=certificate,
        )
    elif type(old_result) is ProvenUnsatResultV2:
        if candidate is None:
            raise RuntimeError("UNSAT replay omitted its candidate")
        certificate = _strict_internal(
            old_result.certificate.model_copy(
                update={
                    "semantic_problem_sha256": problem_hash,
                    "core_solver_config_sha256": config_hash,
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    ),
                }
            ),
            ProvenUnsatCertificateV2,
        )
        result = ProvenUnsatResultV2(
            semantic_problem_sha256=problem_hash,
            core_solver_config=config,
            candidate_domain=candidate,
            certificate=certificate,
        )
    elif type(old_result) is UncertifiedResultV2:
        result = UncertifiedResultV2(
            semantic_problem_sha256=problem_hash,
            core_solver_config=config,
            uncertified_reason=old_result.uncertified_reason,
            candidate_domain=candidate,
            relation_cost_partition=relation,
            objective_partition=objective,
        )
    else:
        raise TypeError("normalized replay returned an invalid result type")

    usage = (
        None
        if replay.cumulative_generation_usage is None
        else _bump_usage(
            replay.cumulative_generation_usage,
            cumulative_increment,
        )
    )
    return CanonicalMinimumCostSolveOutcomeV2(
        result=result,
        finding_codes=replay.finding_codes,
        cumulative_generation_usage=usage,
        proposal_count=replay.proposal_count,
        evaluated_proposal_count=replay.evaluated_proposal_count,
    )


def _bump_usage(
    usage: CompilationResourceUsageV2,
    preprocessing_domain_operations: int,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=usage.domain_operations + preprocessing_domain_operations,
        partition_cells=usage.partition_cells,
        refinement_steps=usage.refinement_steps,
    )


def _strict_internal(value: object, model_type: type[ModelT]) -> ModelT:
    if type(value) is not model_type:
        raise TypeError("rebound internal model has the wrong exact type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        return model_type.model_validate(
            value.model_dump(mode="python"),
            strict=True,
        )


__all__ = (
    "CanonicalMinimumCostSolverV2_1",
    "solve_canonical_minimum_cost_v2_1",
)
