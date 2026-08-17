"""Pure-core orchestration for the Canonical v2 minimum-cost solve.

The solver owns one fresh artifact chain and one cumulative pair of geometry
ledgers.  Certificate assembly consumes that same chain through private core
seams; submitted artifacts and public replay entry points are never inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TypeVar

from pydantic import ValidationError

from spatialcf.core.v2._internal.boundary import (
    InvalidCallerInputV2 as _InvalidInputV2,
)
from spatialcf.core.v2._internal.boundary import (
    NumericBoundaryGapV2 as _NumericInputV2,
)
from spatialcf.core.v2._internal.boundary import strict_input_model_v2
from spatialcf.core.v2.candidate_domain import (
    CandidateDomainCompilationOutcomeV2,
    CandidateDomainCompilerV2,
)
from spatialcf.core.v2.certificate_builder import (
    CertificateBuildKindV2,
    GlobalOptimalityBuildKindV2,
    _build_proven_unsat_from_fresh_candidate_v2,
    _build_selected_global_from_fresh_replay_in_frame_v2,
    _ExactCardinalSelectionFrameV2,
    _GlobalReplayBundleV2,
    _SelectedGlobalBuildOutcomeV2,
)
from spatialcf.core.v2.edit_feasibility import _EditFeasibilityDomainBudgetV2
from spatialcf.core.v2.objective_partition import (
    ObjectivePartitionCompilationKindV2,
    ObjectivePartitionCompilationOutcomeV2,
    _compile_verified_partition,
)
from spatialcf.core.v2.objective_partition import (
    _CompilationIncompleteV2 as _ObjectiveCompilationIncompleteV2,
)
from spatialcf.core.v2.objective_partition import (
    _DomainOperationBudgetV2 as _ObjectiveDomainOperationBudgetV2,
)
from spatialcf.core.v2.objective_partition import (
    _NumericGapV2 as _ObjectiveNumericGapV2,
)
from spatialcf.core.v2.objective_partition import (
    _ResourceLimitV2 as _ObjectiveResourceLimitV2,
)
from spatialcf.core.v2.objective_partition import (
    _UnsupportedV2 as _ObjectiveUnsupportedV2,
)
from spatialcf.core.v2.point_objective import (
    _PointObjectiveReplayContextV2,
    _require_candidate_closure,
    _require_objective_closure,
    _require_relation_closure,
    _require_relation_usage,
)
from spatialcf.core.v2.rectilinear_kernel import (
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
)
from spatialcf.core.v2.relation_cost_partition import (
    RelationCostPartitionCompilationKindV2,
    RelationCostPartitionCompilationOutcomeV2,
    compile_relation_cost_partition_v2,
)
from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    ObjectivePartitionArtifactV2,
    RegionBoundStatusV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    UncertifiedReasonV2,
    UncertifiedResultV2,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


@dataclass(frozen=True, slots=True)
class CanonicalMinimumCostSolveOutcomeV2:
    """Closed orchestration outcome and its exact cumulative work counters.

    ``result`` is absent only when either raw input could not be strictly
    reconstructed, because no honest semantic problem/config hash exists in
    that case.  Every normal failure over valid inputs carries a canonical
    ``UncertifiedResultV2`` with the longest hash-closed artifact prefix.

    The counters are actual telemetry only on the value returned by this
    module's fresh ``solve`` replay.  This publicly constructible dataclass is
    neither a proof token nor a capability: its success telemetry cannot be
    authenticated from a submitted result alone.  Any consumer making a
    semantic or resource claim must replay the raw problem and configuration.
    The same-replay UNSAT path is the narrower exception whose usage is
    structurally closed exactly to the candidate artifact.
    """

    result: CertifiedSuccessResultV2 | ProvenUnsatResultV2 | UncertifiedResultV2 | None
    finding_codes: tuple[str, ...] = ()
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None
    proposal_count: int = 0
    evaluated_proposal_count: int = 0

    def __post_init__(self) -> None:
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        for name in ("proposal_count", "evaluated_proposal_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be a non-negative exact int")
        if self.evaluated_proposal_count > self.proposal_count:
            raise ValueError("evaluated proposal count cannot exceed proposal count")

        result = self.result
        if result is not None:
            result_type = type(result)
            if result_type not in (
                CertifiedSuccessResultV2,
                ProvenUnsatResultV2,
                UncertifiedResultV2,
            ):
                raise TypeError("result must be an exact Canonical v2 solve result")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    result = result_type.model_validate(
                        result.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError("result must pass strict validation") from error

        usage = self.cumulative_generation_usage
        if usage is not None:
            if type(usage) is not CompilationResourceUsageV2:
                raise TypeError(
                    "cumulative_generation_usage must be CompilationResourceUsageV2"
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    usage = CompilationResourceUsageV2.model_validate(
                        usage.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "cumulative_generation_usage must pass strict validation"
                ) from error

        object.__setattr__(self, "finding_codes", findings)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "cumulative_generation_usage", usage)

        if result is None:
            if not findings:
                raise ValueError("missing result requires an input finding")
            if usage is not None:
                raise ValueError("missing result cannot carry trusted generation usage")
            if self.proposal_count or self.evaluated_proposal_count:
                raise ValueError("missing result cannot carry proposal progress")
            return

        config = result.core_solver_config
        if usage is not None and (
            usage.domain_operations > config.max_domain_operations
            or usage.partition_cells > config.max_partition_cells
            or usage.refinement_steps > config.max_refinement_steps
        ):
            raise ValueError("cumulative generation usage exceeds configured limits")
        candidate = getattr(result, "candidate_domain", None)
        if candidate is not None:
            if usage is None:
                raise ValueError("an artifact-bearing result requires cumulative usage")
            base = candidate.resource_usage
            if (
                usage.domain_operations < base.domain_operations
                or usage.partition_cells < base.partition_cells
                or usage.refinement_steps != base.refinement_steps
            ):
                raise ValueError(
                    "cumulative generation usage rolls back candidate work"
                )
        elif usage is not None:
            raise ValueError("a result without a candidate cannot carry trusted usage")

        if type(result) is CertifiedSuccessResultV2:
            if findings:
                raise ValueError("certified success cannot carry failure findings")
            if usage is None:
                raise ValueError(
                    "certified success requires cumulative generation usage"
                )
            if self.proposal_count != len(result.objective_partition.cells):
                raise ValueError(
                    "certified success proposal count must equal objective cell count"
                )
            if self.evaluated_proposal_count != self.proposal_count:
                raise ValueError("certified success must evaluate every proposal")
            return
        if type(result) is ProvenUnsatResultV2:
            if findings:
                raise ValueError("proven unsat cannot carry failure findings")
            if usage is None:
                raise ValueError("proven unsat requires cumulative generation usage")
            if self.proposal_count or self.evaluated_proposal_count:
                raise ValueError("proven unsat cannot carry proposals")
            if usage != result.candidate_domain.resource_usage:
                raise ValueError("proven unsat usage must equal fresh candidate usage")
            return
        if not findings:
            raise ValueError("uncertified result requires at least one finding")
        objective = result.objective_partition
        if objective is None:
            if self.proposal_count or self.evaluated_proposal_count:
                raise ValueError(
                    "uncertified result without objective cannot carry proposals"
                )
        elif self.proposal_count > len(objective.cells):
            raise ValueError(
                "uncertified proposal count cannot exceed objective cell count"
            )


ModelT = TypeVar("ModelT", bound=V2Model)


def solve_canonical_minimum_cost_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve one raw Canonical problem with a single fresh pure-core replay."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        solve_registered_capability_v2,
    )

    return solve_registered_capability_v2(
        SolveCapabilityKeyV2.V2_0,
        problem,
        config,
    )


def _solve_canonical_minimum_cost_in_selection_frame_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve once while ordering exact proposals in a private semantic frame."""

    if type(selection_frame) is not _ExactCardinalSelectionFrameV2:
        raise TypeError("selection_frame has the wrong exact type")

    if type(problem) is SemanticProblemV2_1:
        return CanonicalMinimumCostSolveOutcomeV2(
            result=None,
            finding_codes=("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION",),
        )

    try:
        checked_problem = _strict_input_model(
            problem,
            SemanticProblemV2,
            "SEMANTIC_PROBLEM",
        )
        checked_config = _strict_input_model(
            config,
            CoreSolverConfigV2,
            "CORE_SOLVER_CONFIG",
        )
    except (_InvalidInputV2, _NumericInputV2) as error:
        return CanonicalMinimumCostSolveOutcomeV2(
            result=None,
            finding_codes=(error.finding_code,),
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate_outcome = CandidateDomainCompilerV2().compile(
                checked_problem,
                checked_config,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN",
        )
    if type(candidate_outcome) is not CandidateDomainCompilationOutcomeV2:
        raise TypeError("candidate compiler returned an invalid internal outcome")

    candidate = candidate_outcome.candidate_domain
    if candidate is None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            candidate_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(candidate_outcome.finding_codes or ("CANDIDATE_DOMAIN_NOT_BUILT",)),
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate = _strict_internal_model(
                candidate,
                CandidateDomainArtifactV2,
                "CANDIDATE_DOMAIN",
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN_REVALIDATION",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _require_candidate_closure(checked_problem, checked_config, candidate)
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_REFERENCE_CLOSURE",
        )
    if candidate_outcome.uncertified_reason is not None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            candidate_outcome.uncertified_reason,
            *(candidate_outcome.finding_codes or ("CANDIDATE_DOMAIN_UNCERTIFIED",)),
            candidate=candidate,
            usage=candidate.resource_usage,
        )

    if candidate.hard_domain.outer_bound.status is RegionBoundStatusV2.EMPTY:
        return _solve_empty_candidate(checked_problem, checked_config, candidate)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            relation_outcome = compile_relation_cost_partition_v2(
                checked_problem,
                checked_config,
                candidate,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_COST_PARTITION",
            candidate=candidate,
            usage=candidate.resource_usage,
        )
    if type(relation_outcome) is not RelationCostPartitionCompilationOutcomeV2:
        raise TypeError("relation compiler returned an invalid internal outcome")
    relation_usage = relation_outcome.cumulative_resource_usage
    if relation_outcome.kind is not RelationCostPartitionCompilationKindV2.PARTITION:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            relation_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(relation_outcome.finding_codes or ("RELATION_PARTITION_NOT_BUILT",)),
            candidate=candidate,
            usage=relation_usage or candidate.resource_usage,
        )
    relation = relation_outcome.relation_cost_partition
    if relation is None or relation_usage is None:
        raise RuntimeError("relation PARTITION omitted artifact or usage")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            relation = _strict_internal_model(
                relation,
                RelationCostPartitionV2,
                "RELATION_COST_PARTITION",
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_COST_PARTITION_REVALIDATION",
            candidate=candidate,
            usage=relation_usage,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _require_relation_usage(checked_config, candidate, relation_usage)
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_RESOURCE_USAGE_CLOSURE",
            candidate=candidate,
            usage=relation_usage,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _require_relation_closure(
                checked_problem,
                checked_config,
                candidate,
                relation,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_REFERENCE_CLOSURE",
            candidate=candidate,
            usage=relation_usage,
        )

    objective_domain_budget = _ObjectiveDomainOperationBudgetV2(
        limit=checked_config.max_domain_operations,
        base_used=relation_usage.domain_operations,
    )
    atomic_budget = RectilinearAtomicBudgetV2(
        limit=checked_config.max_partition_cells,
        used=relation_usage.partition_cells,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            objective_outcome = _compile_verified_partition(
                checked_problem,
                checked_config,
                candidate,
                relation,
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            )
    except (RectilinearAtomicBudgetExhaustedV2, _ObjectiveResourceLimitV2) as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            str(error) or "RESOURCE_LIMIT:OBJECTIVE_PARTITION",
            candidate=candidate,
            relation=relation,
            usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveNumericGapV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            *error.finding_codes,
            candidate=candidate,
            relation=relation,
            usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveUnsupportedV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            candidate=candidate,
            relation=relation,
            usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveCompilationIncompleteV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *error.finding_codes,
            candidate=candidate,
            relation=relation,
            usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:OBJECTIVE_PARTITION",
            candidate=candidate,
            relation=relation,
            usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )

    if type(objective_outcome) is not ObjectivePartitionCompilationOutcomeV2:
        raise TypeError("objective compiler returned an invalid internal outcome")
    if objective_outcome.kind is not ObjectivePartitionCompilationKindV2.PARTITION:
        raise RuntimeError("private objective compiler returned a non-partition")
    objective = objective_outcome.objective_partition
    objective_usage = objective_outcome.cumulative_resource_usage
    if objective is None or objective_usage is None:
        raise RuntimeError("objective PARTITION omitted artifact or usage")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_objective = _strict_internal_model(
                objective,
                ObjectivePartitionArtifactV2,
                "OBJECTIVE_PARTITION",
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:OBJECTIVE_PARTITION_REVALIDATION",
            candidate=candidate,
            relation=relation,
            usage=objective_usage,
        )
    if checked_objective != objective:
        raise RuntimeError("fresh objective changed under strict reconstruction")
    expected_objective_usage = _objective_usage(
        relation_usage,
        objective_domain_budget,
        atomic_budget,
    )
    if objective_usage != expected_objective_usage:
        raise RuntimeError("objective cumulative ledger drift")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _require_objective_closure(
                checked_problem,
                checked_config,
                candidate,
                relation,
                objective,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:OBJECTIVE_REFERENCE_CLOSURE",
            candidate=candidate,
            relation=relation,
            usage=objective_usage,
        )

    context = _PointObjectiveReplayContextV2(
        problem=checked_problem,
        config=checked_config,
        candidate=candidate,
        relation=relation,
        objective=objective,
        domain_budget=_EditFeasibilityDomainBudgetV2(
            limit=checked_config.max_domain_operations,
            used=objective_usage.domain_operations,
        ),
        atomic_budget=atomic_budget,
        replay_usage=objective_usage,
    )
    bundle = _GlobalReplayBundleV2(
        problem=checked_problem,
        config=checked_config,
        candidate=candidate,
        relation=relation,
        objective_outcome=objective_outcome,
        point_context=context,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        selected = _build_selected_global_from_fresh_replay_in_frame_v2(
            bundle,
            selection_frame,
        )
    if type(selected) is not _SelectedGlobalBuildOutcomeV2:
        raise TypeError("selected global assembler returned an invalid outcome")
    global_outcome = selected.outcome
    if global_outcome.kind is GlobalOptimalityBuildKindV2.CERTIFIED_SUCCESS:
        result = global_outcome.certified_success_result
        if result is None:
            raise RuntimeError("global success omitted its solve result")
        return CanonicalMinimumCostSolveOutcomeV2(
            result=result,
            cumulative_generation_usage=global_outcome.cumulative_generation_usage,
            proposal_count=selected.proposal_count,
            evaluated_proposal_count=selected.evaluated_proposal_count,
        )
    reason = global_outcome.uncertified_reason
    if global_outcome.kind is GlobalOptimalityBuildKindV2.NOT_PROVEN:
        reason = UncertifiedReasonV2.COMPILATION_INCOMPLETE
    return _valid_uncertified(
        checked_problem,
        checked_config,
        reason or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
        *(global_outcome.finding_codes or ("GLOBAL_OPTIMALITY_NOT_CERTIFIED",)),
        candidate=candidate,
        relation=relation,
        objective=objective,
        usage=global_outcome.cumulative_generation_usage or _context_usage(context),
        proposal_count=selected.proposal_count,
        evaluated_proposal_count=selected.evaluated_proposal_count,
    )


class CanonicalMinimumCostSolverV2:
    """Stateless object wrapper for pure-core pipeline composition."""

    def solve(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
    ) -> CanonicalMinimumCostSolveOutcomeV2:
        return solve_canonical_minimum_cost_v2(problem, config)


def _solve_empty_candidate(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            outcome = _build_proven_unsat_from_fresh_candidate_v2(
                problem,
                config,
                candidate,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            problem,
            config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:PROVEN_UNSAT_ASSEMBLY",
            candidate=candidate,
            usage=candidate.resource_usage,
        )
    if outcome.kind is CertificateBuildKindV2.PROVEN_UNSAT:
        result = outcome.proven_unsat_result
        if result is None:
            raise RuntimeError("PROVEN_UNSAT builder omitted its solve result")
        return CanonicalMinimumCostSolveOutcomeV2(
            result=result,
            cumulative_generation_usage=outcome.verification_resource_usage,
        )
    return _valid_uncertified(
        problem,
        config,
        outcome.uncertified_reason or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
        *(outcome.finding_codes or ("PROVEN_UNSAT_NOT_CERTIFIED",)),
        candidate=candidate,
        usage=outcome.verification_resource_usage or candidate.resource_usage,
    )


def _strict_input_model(
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    return strict_input_model_v2(value, model_type, label)


def _strict_internal_model(
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    if type(value) is not model_type:
        raise TypeError(f"fresh {label} has the wrong internal type")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise ArithmeticError(
            f"numeric failure reconstructing fresh {label}"
        ) from error


def _objective_usage(
    base: CompilationResourceUsageV2,
    domain_budget: _ObjectiveDomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=base.domain_operations + domain_budget.used,
        partition_cells=atomic_budget.used,
        refinement_steps=base.refinement_steps,
    )


def _context_usage(
    context: _PointObjectiveReplayContextV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=context.domain_budget.used,
        partition_cells=context.atomic_budget.used,
        refinement_steps=context.replay_usage.refinement_steps,
    )


def _valid_uncertified(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    candidate: CandidateDomainArtifactV2 | None = None,
    relation: RelationCostPartitionV2 | None = None,
    objective: ObjectivePartitionArtifactV2 | None = None,
    usage: CompilationResourceUsageV2 | None = None,
    proposal_count: int = 0,
    evaluated_proposal_count: int = 0,
) -> CanonicalMinimumCostSolveOutcomeV2:
    if objective is not None and relation is None:
        raise RuntimeError("objective prefix requires a relation prefix")
    if relation is not None and candidate is None:
        raise RuntimeError("relation prefix requires a candidate prefix")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        result = UncertifiedResultV2(
            semantic_problem_sha256=problem.semantic_problem_sha256,
            core_solver_config=config,
            uncertified_reason=reason,
            candidate_domain=candidate,
            relation_cost_partition=relation,
            objective_partition=objective,
        )
        result = UncertifiedResultV2.model_validate(
            result.model_dump(mode="python"),
            strict=True,
        )
        encoded = canonical_json_bytes_v2(result)
        restored = UncertifiedResultV2.model_validate_json(encoded, strict=True)
    if (
        restored != result
        or canonical_json_bytes_v2(restored) != encoded
        or restored.solve_result_sha256 != result.solve_result_sha256
    ):
        raise RuntimeError("UNCERTIFIED canonical round-trip closure drift")
    return CanonicalMinimumCostSolveOutcomeV2(
        result=result,
        finding_codes=tuple(finding_codes),
        cumulative_generation_usage=usage,
        proposal_count=proposal_count,
        evaluated_proposal_count=evaluated_proposal_count,
    )
