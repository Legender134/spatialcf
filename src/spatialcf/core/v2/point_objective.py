"""Fresh-replay objective bounds for one concrete Canonical edit.

The public entry accepts only the semantic problem, solver configuration, and
edit.  Candidate, relation, and objective artifacts are regenerated inside the
pure core.  A bounded point loss is published only after the independent
positive inner-domain feasibility proof succeeds; objective proposals and
submitted artifacts are never accepted as proof inputs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2.candidate_domain import (
    CandidateDomainCompilationOutcomeV2,
    CandidateDomainCompilerV2,
)
from spatialcf.core.v2.edit_feasibility import (
    CanonicalEditFeasibilityKindV2,
    CanonicalEditFeasibilityVerificationOutcomeV2,
    _contains_point_with_budget,
    _DomainOperationBudgetExhaustedV2,
    _EditFeasibilityDomainBudgetV2,
    _verify_replayed_candidate_edit_membership_v2,
)
from spatialcf.core.v2.objective_numeric import (
    ConstraintSafetyInputV2,
    ObjectiveIntervalOutcomeV2,
    ObjectiveNumericKindV2,
    TranslationCellOutcomeV2,
    VisibilityMetricIntervalV2,
    aggregate_relation_damage_bounds_v2,
    aggregate_safety_penalty_bounds_v2,
    aggregate_visibility_change_bounds_v2,
    compile_translation_l2_cell_bounds_v2,
)
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
from spatialcf.core.v2.objective_safety_bounds import (
    ObjectiveSafetyBoundsKindV2,
    ObjectiveSafetyBoundsOutcomeV2,
    compile_objective_safety_bounds_v2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    lift_planar_region_v2,
    normalize_rectilinear_region_v2,
)
from spatialcf.core.v2.relation_cost_partition import (
    RelationCostPartitionCompilationKindV2,
    RelationCostPartitionCompilationOutcomeV2,
    compile_relation_cost_partition_v2,
)
from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    ConstraintSlackV2,
    NonNegativeIntervalV2,
    ObjectivePartitionArtifactV2,
    ObjectivePartitionCellV2,
    ObjectiveTermBoundsV2,
    RegionBoundStatusV2,
    RelationCostCellV2,
    RelationCostPartitionV2,
    RelationDamageBoundV2,
)
from spatialcf.domain.v2.base import (
    CanonicalId,
    FactAvailabilityV2,
    FactCompletenessV2,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.objective import ObjectCameraKeyV2, SafetySlackUnitV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import CoreSolverConfigV2, UncertifiedReasonV2
from spatialcf.domain.v2.scene import BaselineObservationV2


class PointObjectiveEvaluationKindV2(StrEnum):
    BOUNDED_FEASIBLE = "BOUNDED_FEASIBLE"
    NOT_PROVEN = "NOT_PROVEN"
    UNCERTIFIED = "UNCERTIFIED"


_SHA256_ADAPTER = TypeAdapter(Sha256Digest)
_CANONICAL_ID_ADAPTER = TypeAdapter(CanonicalId)


@dataclass(frozen=True, slots=True)
class PointObjectiveEvaluationOutcomeV2:
    """Closed result for one independently replayed feasible point."""

    kind: PointObjectiveEvaluationKindV2
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    candidate_domain_artifact_sha256: Sha256Digest | None = None
    relation_cost_partition_sha256: Sha256Digest | None = None
    objective_partition_artifact_sha256: Sha256Digest | None = None
    canonical_edit_sha256: Sha256Digest | None = None
    witness_loss_bounds: ObjectiveTermBoundsV2 | None = None
    relation_damage_vector: tuple[RelationDamageBoundV2, ...] = ()
    constraint_slacks: tuple[ConstraintSlackV2, ...] = ()
    covering_objective_cell_ids: tuple[CanonicalId, ...] = ()
    covering_relation_cell_ids: tuple[CanonicalId, ...] = ()
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not PointObjectiveEvaluationKindV2:
            raise TypeError("kind must be a PointObjectiveEvaluationKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be exact non-blank strings")
        if type(self.relation_damage_vector) is not tuple or any(
            type(item) is not RelationDamageBoundV2
            for item in self.relation_damage_vector
        ):
            raise TypeError("relation_damage_vector must be an exact bound tuple")
        if type(self.constraint_slacks) is not tuple or any(
            type(item) is not ConstraintSlackV2 for item in self.constraint_slacks
        ):
            raise TypeError("constraint_slacks must be an exact slack tuple")

        findings = tuple(sorted(set(self.finding_codes)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                vector = tuple(
                    RelationDamageBoundV2.model_validate(
                        item.model_dump(mode="python"),
                        strict=True,
                    )
                    for item in self.relation_damage_vector
                )
        except (ValidationError, TypeError, ValueError, Warning) as error:
            raise TypeError(
                "relation_damage_vector must pass strict validation"
            ) from error
        vector = tuple(sorted(vector, key=lambda item: item.key.sort_key))
        if len({item.key for item in vector}) != len(vector):
            raise ValueError("point relation vector keys must be unique")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                slacks = tuple(
                    ConstraintSlackV2.model_validate(
                        item.model_dump(mode="python"),
                        strict=True,
                    )
                    for item in self.constraint_slacks
                )
        except (ValidationError, TypeError, ValueError, Warning) as error:
            raise TypeError("constraint_slacks must pass strict validation") from error
        slacks = tuple(sorted(slacks, key=lambda item: item.constraint_id))
        if len({item.constraint_id for item in slacks}) != len(slacks):
            raise ValueError("point constraint slack IDs must be unique")
        objective_ids = _strict_ids(
            self.covering_objective_cell_ids,
            "covering_objective_cell_ids",
        )
        relation_ids = _strict_ids(
            self.covering_relation_cell_ids,
            "covering_relation_cell_ids",
        )
        loss = self.witness_loss_bounds
        if loss is not None:
            if type(loss) is not ObjectiveTermBoundsV2:
                raise TypeError("witness_loss_bounds must be ObjectiveTermBoundsV2")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    loss = ObjectiveTermBoundsV2.model_validate(
                        loss.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "witness_loss_bounds must pass strict validation"
                ) from error
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
        references: list[str | None] = []
        for field_name in (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "candidate_domain_artifact_sha256",
            "relation_cost_partition_sha256",
            "objective_partition_artifact_sha256",
            "canonical_edit_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = _SHA256_ADAPTER.validate_python(value, strict=True)
                object.__setattr__(self, field_name, value)
            references.append(value)

        object.__setattr__(self, "finding_codes", findings)
        object.__setattr__(self, "relation_damage_vector", vector)
        object.__setattr__(self, "constraint_slacks", slacks)
        object.__setattr__(self, "covering_objective_cell_ids", objective_ids)
        object.__setattr__(self, "covering_relation_cell_ids", relation_ids)
        object.__setattr__(self, "witness_loss_bounds", loss)
        object.__setattr__(self, "cumulative_generation_usage", usage)

        if self.kind is PointObjectiveEvaluationKindV2.BOUNDED_FEASIBLE:
            if any(value is None for value in references):
                raise ValueError("BOUNDED_FEASIBLE requires all replay hashes")
            if (
                loss is None
                or not vector
                or not slacks
                or not objective_ids
                or not relation_ids
                or usage is None
            ):
                raise ValueError("BOUNDED_FEASIBLE requires complete point bounds")
            if self.uncertified_reason is not None or findings:
                raise ValueError("BOUNDED_FEASIBLE cannot carry failure diagnostics")
            return

        if any(value is not None for value in references):
            raise ValueError("non-success outcomes cannot carry replay hashes")
        if loss is not None or vector or slacks or objective_ids or relation_ids:
            raise ValueError("non-success outcomes cannot carry partial point bounds")
        if not findings:
            raise ValueError(f"{self.kind.value} requires at least one finding")
        if self.kind is PointObjectiveEvaluationKindV2.NOT_PROVEN:
            if self.uncertified_reason is not None:
                raise ValueError("NOT_PROVEN cannot carry an uncertified reason")
            return
        if type(self.uncertified_reason) is not UncertifiedReasonV2:
            raise TypeError("UNCERTIFIED requires an UncertifiedReasonV2")


PointObjectiveOutcomeV2 = PointObjectiveEvaluationOutcomeV2


@dataclass(frozen=True, slots=True)
class _PointObjectiveReplayContextV2:
    problem: SemanticProblemV2
    config: CoreSolverConfigV2
    candidate: CandidateDomainArtifactV2
    relation: RelationCostPartitionV2
    objective: ObjectivePartitionArtifactV2
    domain_budget: _EditFeasibilityDomainBudgetV2
    atomic_budget: RectilinearAtomicBudgetV2
    replay_usage: CompilationResourceUsageV2


class _PointResourceLimitV2(RuntimeError):
    pass


class _PointNumericGapV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _PointUnsupportedV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _PointIncompleteV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _InvalidInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _NumericInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


ModelT = TypeVar("ModelT", bound=V2Model)


def evaluate_point_objective_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    edit: CanonicalEditV2,
) -> PointObjectiveEvaluationOutcomeV2:
    """Freshly compile and bound the objective at one positively proven edit."""

    try:
        checked_problem = _strict_model(problem, SemanticProblemV2, "SEMANTIC_PROBLEM")
        checked_config = _strict_model(config, CoreSolverConfigV2, "CORE_SOLVER_CONFIG")
        checked_edit = _strict_model(edit, CanonicalEditV2, "CANONICAL_EDIT")
    except _NumericInputV2 as error:
        return _uncertified(UncertifiedReasonV2.NUMERIC_GAP, error.finding_code)
    except _InvalidInputV2 as error:
        return _uncertified(UncertifiedReasonV2.UNSUPPORTED_MODEL, error.finding_code)

    reference_findings = _edit_reference_findings(checked_problem, checked_edit)
    if reference_findings:
        return _not_proven(*reference_findings)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate_outcome = CandidateDomainCompilerV2().compile(
                checked_problem,
                checked_config,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_CANDIDATE_REPLAY",
        )
    if type(candidate_outcome) is not CandidateDomainCompilationOutcomeV2:
        raise TypeError("candidate compiler returned an invalid internal outcome")
    candidate = candidate_outcome.candidate_domain
    if candidate is None:
        return _uncertified(
            candidate_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(candidate_outcome.finding_codes or ("POINT_REPLAY_NO_CANDIDATE",)),
        )
    _require_candidate_closure(checked_problem, checked_config, candidate)
    if candidate_outcome.uncertified_reason is not None:
        return _uncertified(
            candidate_outcome.uncertified_reason,
            *(candidate_outcome.finding_codes or ("POINT_CANDIDATE_UNCERTIFIED",)),
            cumulative_generation_usage=candidate.resource_usage,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            relation_outcome = compile_relation_cost_partition_v2(
                checked_problem,
                checked_config,
                candidate,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_RELATION_REPLAY",
            cumulative_generation_usage=candidate.resource_usage,
        )
    if type(relation_outcome) is not RelationCostPartitionCompilationOutcomeV2:
        raise TypeError("relation compiler returned an invalid internal outcome")
    relation_usage = relation_outcome.cumulative_resource_usage
    if relation_outcome.kind is not RelationCostPartitionCompilationKindV2.PARTITION:
        return _uncertified(
            relation_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(relation_outcome.finding_codes or ("POINT_REPLAY_NO_RELATION",)),
            cumulative_generation_usage=relation_usage or candidate.resource_usage,
        )
    relation = relation_outcome.relation_cost_partition
    if relation is None or relation_usage is None:
        raise RuntimeError("PARTITION relation replay omitted artifact or usage")
    _require_relation_usage(checked_config, candidate, relation_usage)
    _require_relation_closure(
        checked_problem,
        checked_config,
        candidate,
        relation,
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
        return _uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            str(error) or "RESOURCE_LIMIT:max_partition_cells",
            cumulative_generation_usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveNumericGapV2 as error:
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            *error.finding_codes,
            cumulative_generation_usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveUnsupportedV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            cumulative_generation_usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveCompilationIncompleteV2 as error:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *error.finding_codes,
            cumulative_generation_usage=_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_OBJECTIVE_REPLAY",
            cumulative_generation_usage=_objective_usage(
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
        raise RuntimeError("PARTITION objective replay omitted artifact or usage")
    if (
        relation_usage.domain_operations + objective_domain_budget.used
        != objective_usage.domain_operations
    ):
        raise RuntimeError("objective domain ledger drift")
    if atomic_budget.used != objective_usage.partition_cells:
        raise RuntimeError("objective atomic ledger drift")
    if objective_usage.refinement_steps != relation_usage.refinement_steps:
        raise RuntimeError("objective refinement ledger drift")
    _require_objective_closure(
        checked_problem,
        checked_config,
        candidate,
        relation,
        objective,
    )
    domain_budget = _EditFeasibilityDomainBudgetV2(
        limit=checked_config.max_domain_operations,
        used=objective_usage.domain_operations,
    )
    context = _PointObjectiveReplayContextV2(
        problem=checked_problem,
        config=checked_config,
        candidate=candidate,
        relation=relation,
        objective=objective,
        domain_budget=domain_budget,
        atomic_budget=atomic_budget,
        replay_usage=objective_usage,
    )
    return _evaluate_point_objective_from_replay_v2(context, checked_edit)


class PointObjectiveEvaluatorV2:
    """Stateless wrapper for the public fresh-replay evaluation."""

    def evaluate(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        edit: CanonicalEditV2,
    ) -> PointObjectiveEvaluationOutcomeV2:
        return evaluate_point_objective_v2(problem, config, edit)


def _evaluate_point_objective_from_replay_v2(
    context: _PointObjectiveReplayContextV2,
    edit: CanonicalEditV2,
) -> PointObjectiveEvaluationOutcomeV2:
    """Evaluate one edit while continuing caller-owned replay ledgers.

    The context is intentionally reusable for a sequence of different edits.
    Its artifact prefix and refinement count stay frozen while the two mutable
    budget objects advance.  Every membership proof receives the usage derived
    from those current ledgers, so a later point can neither reset nor replay an
    earlier budget prefix.
    """

    try:
        checked_edit = _strict_replay_edit(edit)
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_EDIT_REVALIDATION",
            cumulative_generation_usage=_context_usage(context),
        )
    _validate_replay_context(context, checked_edit)
    current_usage = _context_usage(context)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            feasibility = _verify_replayed_candidate_edit_membership_v2(
                context.problem,
                checked_edit,
                context.candidate,
                context.domain_budget,
                context.atomic_budget,
                cumulative_resource_usage=current_usage,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_FEASIBILITY",
            cumulative_generation_usage=_context_usage(context),
        )
    if type(feasibility) is not CanonicalEditFeasibilityVerificationOutcomeV2:
        raise TypeError("feasibility helper returned an invalid internal outcome")
    feasibility_usage = feasibility.verification_resource_usage
    _require_feasibility_usage(context, feasibility_usage)
    if feasibility.kind is CanonicalEditFeasibilityKindV2.NOT_PROVEN:
        return _not_proven(
            *(feasibility.finding_codes or ("POINT_FEASIBILITY_NOT_PROVEN",)),
            cumulative_generation_usage=feasibility_usage,
        )
    if feasibility.kind is CanonicalEditFeasibilityKindV2.UNCERTIFIED:
        return _uncertified(
            feasibility.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(feasibility.finding_codes or ("POINT_FEASIBILITY_UNCERTIFIED",)),
            cumulative_generation_usage=feasibility_usage,
        )
    if feasibility_usage is None:
        raise RuntimeError("positive feasibility proof omitted cumulative usage")
    _require_feasibility_closure(context, checked_edit, feasibility)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_point_bounds(context, checked_edit)
    except (
        RectilinearAtomicBudgetExhaustedV2,
        _DomainOperationBudgetExhaustedV2,
        _PointResourceLimitV2,
    ) as error:
        return _uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            str(error) or "RESOURCE_LIMIT:POINT_OBJECTIVE",
            cumulative_generation_usage=_context_usage(context),
        )
    except _PointNumericGapV2 as error:
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            *error.finding_codes,
            cumulative_generation_usage=_context_usage(context),
        )
    except _PointUnsupportedV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            cumulative_generation_usage=_context_usage(context),
        )
    except _PointIncompleteV2 as error:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *error.finding_codes,
            cumulative_generation_usage=_context_usage(context),
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:POINT_OBJECTIVE_EVALUATION",
            cumulative_generation_usage=_context_usage(context),
        )


def _compile_point_bounds(
    context: _PointObjectiveReplayContextV2,
    edit: CanonicalEditV2,
) -> PointObjectiveEvaluationOutcomeV2:
    _reserve_point_domain_work(context)
    point = _point_region(edit, context.atomic_budget)
    x = Fraction.from_float(edit.translation_xy_m.x)
    y = Fraction.from_float(edit.translation_xy_m.y)

    relation_cells = _covering_cells(
        context.relation.cells,
        x,
        y,
        context.atomic_budget,
    )
    objective_cells = _covering_cells(
        context.objective.cells,
        x,
        y,
        context.atomic_budget,
    )
    if not relation_cells or not objective_cells:
        raise _PointIncompleteV2("COMPILATION_INCOMPLETE:POINT_CELL_COVERAGE")
    relation_ids = tuple(sorted(cell.cell_id for cell in relation_cells))
    objective_ids = tuple(sorted(cell.cell_id for cell in objective_cells))
    objective_parent_ids = tuple(
        sorted({cell.parent_relation_cell_id for cell in objective_cells})
    )
    if objective_parent_ids != relation_ids:
        raise RuntimeError("point relation/objective covering-cell mapping drift")

    relation_vector = _damage_vector_hull(
        context.problem,
        relation_cells,
    )
    objective_vector = _damage_vector_hull(
        context.problem,
        objective_cells,
    )
    if relation_vector != objective_vector:
        raise RuntimeError("point relation vector differs across fresh artifacts")

    translation = _require_translation(
        compile_translation_l2_cell_bounds_v2(
            point,
            context.problem.objective.translation,
            atomic_budget=context.atomic_budget,
        )
    )
    relation_loss = _require_interval(
        aggregate_relation_damage_bounds_v2(
            context.problem.objective.relation_damage,
            relation_vector,
        ),
        "POINT_RELATION_DAMAGE",
    )

    visibility_intervals = _visibility_intervals(
        context.problem,
    )
    visibility_loss = _require_interval(
        aggregate_visibility_change_bounds_v2(
            context.problem.objective.visibility_change,
            visibility_intervals,
        ),
        "POINT_VISIBILITY_CHANGE",
    )

    safety = _require_safety(
        compile_objective_safety_bounds_v2(
            context.problem,
            point,
            atomic_budget=context.atomic_budget,
        )
    )
    safety_inputs = tuple(
        ConstraintSafetyInputV2(
            constraint_id=item.constraint_id,
            components=item.raw_components,
        )
        for item in safety.constraint_bounds
    )
    safety_loss = _require_interval(
        aggregate_safety_penalty_bounds_v2(
            context.problem.objective.safety_margin,
            safety_inputs,
        ),
        "POINT_SAFETY_MARGIN",
    )
    slacks = tuple(
        ConstraintSlackV2(
            constraint_id=item.constraint_id,
            lower_bound=item.normalized_slack.lower_bound,
            upper_bound=item.normalized_slack.upper_bound,
            unit=SafetySlackUnitV2.DIMENSIONLESS,
        )
        for item in safety.constraint_bounds
    )
    _require_point_slacks_enclosed(
        objective_cells,
        slacks,
    )
    loss = _build_point_term_bounds(
        NonNegativeIntervalV2(
            lower_bound=translation.bounds.lower_bound,
            upper_bound=translation.bounds.upper_bound,
        ),
        _nonnegative_interval(relation_loss),
        _nonnegative_interval(visibility_loss),
        _nonnegative_interval(safety_loss),
    )
    usage = _context_usage(context)
    return PointObjectiveEvaluationOutcomeV2(
        kind=PointObjectiveEvaluationKindV2.BOUNDED_FEASIBLE,
        semantic_problem_sha256=context.problem.semantic_problem_sha256,
        core_solver_config_sha256=context.config.core_solver_config_sha256,
        candidate_domain_artifact_sha256=(
            context.candidate.candidate_domain_artifact_sha256
        ),
        relation_cost_partition_sha256=(
            context.relation.relation_cost_partition_sha256
        ),
        objective_partition_artifact_sha256=(
            context.objective.objective_partition_artifact_sha256
        ),
        canonical_edit_sha256=edit.edit_sha256,
        witness_loss_bounds=loss,
        relation_damage_vector=relation_vector,
        constraint_slacks=slacks,
        covering_objective_cell_ids=objective_ids,
        covering_relation_cell_ids=relation_ids,
        cumulative_generation_usage=usage,
    )


def _strict_model(
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    if type(value) is not model_type:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}:TYPE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"), strict=True
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise _NumericInputV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}") from error


def _strict_ids(values: object, label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    try:
        checked = tuple(
            _CANONICAL_ID_ADAPTER.validate_python(value, strict=True)
            for value in values
        )
    except ValidationError as error:
        raise TypeError(f"{label} contains an invalid CanonicalId") from error
    if len(checked) != len(set(checked)):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(checked))


def _edit_reference_findings(
    problem: SemanticProblemV2,
    edit: CanonicalEditV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if edit.semantic_problem_sha256 != problem.semantic_problem_sha256:
        findings.append("EDIT_REFERENCE_MISMATCH:SEMANTIC_PROBLEM_HASH")
    if edit.subject_id != problem.constraints.allowed_edit.subject_id:
        findings.append("EDIT_REFERENCE_MISMATCH:SUBJECT_ID")
    return tuple(findings)


def _require_candidate_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> None:
    if candidate.semantic_problem_sha256 != problem.semantic_problem_sha256:
        raise RuntimeError("fresh candidate problem hash is not closed")
    if candidate.core_solver_config_sha256 != config.core_solver_config_sha256:
        raise RuntimeError("fresh candidate config hash is not closed")
    if candidate.candidate_variable.subject_id != (
        problem.constraints.allowed_edit.subject_id
    ):
        raise RuntimeError("fresh candidate subject is not closed")


def _require_relation_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionV2,
) -> None:
    expected = (
        problem.semantic_problem_sha256,
        config.core_solver_config_sha256,
        candidate.candidate_domain_artifact_sha256,
        problem.objective.objective_spec_sha256,
    )
    actual = (
        relation.semantic_problem_sha256,
        relation.core_solver_config_sha256,
        relation.candidate_domain_artifact_sha256,
        relation.objective_spec_sha256,
    )
    if actual != expected:
        raise RuntimeError("fresh relation replay reference closure drift")


def _require_relation_usage(
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    usage: CompilationResourceUsageV2,
) -> None:
    candidate_usage = candidate.resource_usage
    if (
        usage.domain_operations < candidate_usage.domain_operations
        or usage.partition_cells < candidate_usage.partition_cells
    ):
        raise RuntimeError("fresh relation replay rolled back a resource ledger")
    if usage.refinement_steps != candidate_usage.refinement_steps:
        raise RuntimeError("fresh relation replay changed the refinement ledger")
    if (
        usage.domain_operations > config.max_domain_operations
        or usage.partition_cells > config.max_partition_cells
    ):
        raise RuntimeError("fresh relation replay exceeded a configured ledger")


def _require_objective_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionV2,
    objective: ObjectivePartitionArtifactV2,
) -> None:
    expected = (
        problem.semantic_problem_sha256,
        config.core_solver_config_sha256,
        candidate.candidate_domain_artifact_sha256,
        relation.relation_cost_partition_sha256,
        problem.objective.objective_spec_sha256,
    )
    actual = (
        objective.semantic_problem_sha256,
        objective.core_solver_config_sha256,
        objective.candidate_domain_artifact_sha256,
        objective.relation_cost_partition_sha256,
        objective.objective_spec_sha256,
    )
    if actual != expected:
        raise RuntimeError("fresh objective replay reference closure drift")


def _strict_replay_edit(edit: object) -> CanonicalEditV2:
    if type(edit) is not CanonicalEditV2:
        raise TypeError("edit must be an exact CanonicalEditV2")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return CanonicalEditV2.model_validate(
                edit.model_dump(mode="python"),
                strict=True,
            )
    except (ValidationError, TypeError, ValueError) as error:
        raise TypeError("edit must pass strict validation") from error


def _validate_replay_context(
    context: _PointObjectiveReplayContextV2,
    edit: CanonicalEditV2,
) -> None:
    if type(context) is not _PointObjectiveReplayContextV2:
        raise TypeError("context must be a _PointObjectiveReplayContextV2")
    if type(context.domain_budget) is not _EditFeasibilityDomainBudgetV2:
        raise TypeError("context domain budget has the wrong type")
    if type(context.atomic_budget) is not RectilinearAtomicBudgetV2:
        raise TypeError("context atomic budget has the wrong type")
    context.domain_budget.validate()
    context.atomic_budget.validate()
    if context.domain_budget.limit != context.config.max_domain_operations:
        raise ValueError("context domain budget/config limit mismatch")
    if context.atomic_budget.limit != context.config.max_partition_cells:
        raise ValueError("context atomic budget/config limit mismatch")
    if type(context.replay_usage) is not CompilationResourceUsageV2:
        raise TypeError("context replay usage has the wrong type")
    if context.replay_usage.domain_operations > context.domain_budget.used:
        raise ValueError("context domain ledger rolled back below replay")
    if context.replay_usage.partition_cells > context.atomic_budget.used:
        raise ValueError("context atomic ledger rolled back below replay")
    if (
        context.replay_usage.refinement_steps
        != context.candidate.resource_usage.refinement_steps
    ):
        raise ValueError("context refinement ledger drift")
    _require_candidate_closure(context.problem, context.config, context.candidate)
    _require_relation_closure(
        context.problem,
        context.config,
        context.candidate,
        context.relation,
    )
    _require_objective_closure(
        context.problem,
        context.config,
        context.candidate,
        context.relation,
        context.objective,
    )
    if _edit_reference_findings(context.problem, edit):
        raise ValueError("context edit references are not closed")


def _require_feasibility_closure(
    context: _PointObjectiveReplayContextV2,
    edit: CanonicalEditV2,
    outcome: object,
) -> None:
    expected = (
        context.problem.semantic_problem_sha256,
        context.config.core_solver_config_sha256,
        context.candidate.candidate_domain_artifact_sha256,
        edit.edit_sha256,
    )
    actual = (
        outcome.semantic_problem_sha256,
        outcome.core_solver_config_sha256,
        outcome.candidate_domain_artifact_sha256,
        outcome.canonical_edit_sha256,
    )
    if actual != expected:
        raise RuntimeError("positive feasibility proof reference closure drift")


def _require_feasibility_usage(
    context: _PointObjectiveReplayContextV2,
    usage: CompilationResourceUsageV2 | None,
) -> None:
    if usage is None:
        raise RuntimeError("feasibility helper omitted cumulative usage")
    expected = _context_usage(context)
    if usage != expected:
        raise RuntimeError("feasibility helper cumulative ledger drift")


def _point_region(
    edit: CanonicalEditV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    x = Fraction.from_float(edit.translation_xy_m.x)
    y = Fraction.from_float(edit.translation_xy_m.y)
    rectangle = ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=x,
        min_y_m=y,
        max_x_m=x,
        max_y_m=y,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )
    return _require_region(
        normalize_rectilinear_region_v2(
            (rectangle,),
            atomic_budget=atomic_budget,
        )
    )


def _reserve_point_domain_work(context: _PointObjectiveReplayContextV2) -> None:
    """Atomically reserve every logical deep pass for one point evaluation."""

    relation_cells = len(context.relation.cells)
    objective_cells = len(context.objective.cells)
    relation_keys = len(context.problem.objective.relation_damage.pair_axis_weights)
    safety_targets = len(context.problem.objective.safety_margin.aggregation.targets)
    visibility_values = context.problem.scene.baseline_observations.values
    visibility_facts = len(visibility_values) if visibility_values is not None else 0
    visibility_weights = len(
        context.problem.objective.visibility_change.object_camera_weights
    )
    units = (
        1  # exact singleton-region construction
        + relation_cells
        + objective_cells  # closed-cell coverage passes
        + objective_cells  # objective-to-relation parent mapping
        + 2
        * (relation_cells + objective_cells)
        * relation_keys  # vector maps plus per-key interval hulls
        + relation_keys  # relation weighted aggregation
        + visibility_facts
        + 2 * visibility_weights  # interval construction plus aggregation
        + 2 * safety_targets  # singleton safety compilation plus aggregation
        + objective_cells * safety_targets  # all covering-cell enclosure checks
        + relation_keys
        + safety_targets
        + objective_cells
        + relation_cells  # canonical success publication
    )
    context.domain_budget.consume(units)


CellV2 = RelationCostCellV2 | ObjectivePartitionCellV2


def _covering_cells(
    cells: tuple[CellV2, ...],
    x: Fraction,
    y: Fraction,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> tuple[CellV2, ...]:
    covering: list[CellV2] = []
    for cell in cells:
        outer = cell.domain.outer_bound
        if outer.status is RegionBoundStatusV2.EMPTY:
            continue
        if outer.status is not RegionBoundStatusV2.NON_EMPTY or outer.region is None:
            raise _PointIncompleteV2("COMPILATION_INCOMPLETE:POINT_CELL_OUTER_BOUND")
        lifted = _require_region(
            lift_planar_region_v2(
                outer.region,
                atomic_budget=atomic_budget,
            )
        )
        if _contains_point_with_budget(lifted, x, y, atomic_budget):
            covering.append(cell)
    return tuple(covering)


def _damage_vector_hull(
    problem: SemanticProblemV2,
    cells: tuple[CellV2, ...],
) -> tuple[RelationDamageBoundV2, ...]:
    keys = tuple(
        item.key for item in problem.objective.relation_damage.pair_axis_weights
    )
    by_cell: list[dict[object, RelationDamageBoundV2]] = []
    for cell in cells:
        mapping = {item.key: item for item in cell.relation_damage_vector}
        if set(mapping) != set(keys):
            raise RuntimeError("fresh point relation vector key closure drift")
        by_cell.append(mapping)
    return tuple(
        RelationDamageBoundV2(
            key=key,
            lower_bound=min(mapping[key].lower_bound for mapping in by_cell),
            upper_bound=max(mapping[key].upper_bound for mapping in by_cell),
        )
        for key in keys
    )


def _visibility_intervals(
    problem: SemanticProblemV2,
) -> tuple[VisibilityMetricIntervalV2, ...]:
    facts = problem.scene.baseline_observations
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.values is None
    ):
        raise _PointIncompleteV2("COMPILATION_INCOMPLETE:POINT_VISIBILITY_BASELINE")
    observations: dict[tuple[str, str, str, str], BaselineObservationV2] = {}
    for observation in facts.values:
        key = (
            observation.object_id,
            observation.camera_id,
            observation.metric_definition_id,
            observation.metric_definition_version,
        )
        if key in observations:
            raise _PointUnsupportedV2("INVALID_POINT_VISIBILITY:DUPLICATE_BASELINE")
        observations[key] = observation
    weights = problem.objective.visibility_change.object_camera_weights
    intervals: list[VisibilityMetricIntervalV2] = []
    for weighted in weights:
        key = weighted.key
        semantic_key = (
            key.object_id,
            key.camera_id,
            key.metric_definition_id,
            key.metric_definition_version,
        )
        observation = observations.get(semantic_key)
        if observation is None:
            raise _PointIncompleteV2(
                "COMPILATION_INCOMPLETE:POINT_VISIBILITY_BASELINE_KEY"
            )
        intervals.append(
            VisibilityMetricIntervalV2(
                key=ObjectCameraKeyV2.model_validate(
                    key.model_dump(mode="python"),
                    strict=True,
                ),
                baseline_lower=Fraction.from_float(observation.normalized_lower_bound),
                baseline_upper=Fraction.from_float(observation.normalized_upper_bound),
                candidate_lower=Fraction(),
                candidate_upper=Fraction(1),
            )
        )
    return tuple(intervals)


def _require_point_slacks_enclosed(
    cells: tuple[ObjectivePartitionCellV2, ...],
    slacks: tuple[ConstraintSlackV2, ...],
) -> None:
    point_by_id = {item.constraint_id: item for item in slacks}
    for cell in cells:
        cell_by_id = {item.constraint_id: item for item in cell.constraint_slacks}
        if set(cell_by_id) != set(point_by_id):
            raise RuntimeError("fresh objective slack key closure drift")
        if any(
            cell_by_id[key].lower_bound > point.lower_bound
            or cell_by_id[key].upper_bound < point.upper_bound
            for key, point in point_by_id.items()
        ):
            raise RuntimeError("point slack escaped a covering objective cell bound")


def _require_region(outcome: RectilinearRegionOutcomeV2) -> ExactRectilinearRegionV2:
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _PointResourceLimitV2("RESOURCE_LIMIT:max_partition_cells")
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise _PointUnsupportedV2(
            *(outcome.finding_codes or ("UNSUPPORTED_POINT_RECTILINEAR_REGION",))
        )
    return outcome.region


def _require_translation(outcome: TranslationCellOutcomeV2) -> TranslationCellOutcomeV2:
    if outcome.kind is ObjectiveNumericKindV2.EXACT and outcome.bounds is not None:
        return outcome
    _raise_numeric(outcome.kind, outcome.finding_codes, "POINT_TRANSLATION")
    raise AssertionError("unreachable")


def _require_interval(
    outcome: ObjectiveIntervalOutcomeV2,
    label: str,
) -> ObjectiveIntervalOutcomeV2:
    if outcome.kind is ObjectiveNumericKindV2.EXACT and outcome.interval is not None:
        return outcome
    _raise_numeric(outcome.kind, outcome.finding_codes, label)
    raise AssertionError("unreachable")


def _raise_numeric(
    kind: ObjectiveNumericKindV2,
    findings: tuple[str, ...],
    label: str,
) -> None:
    codes = findings or (f"{label}:{kind.value}",)
    if kind is ObjectiveNumericKindV2.NUMERIC_GAP:
        raise _PointNumericGapV2(*codes)
    if kind is ObjectiveNumericKindV2.RESOURCE_LIMIT:
        raise _PointResourceLimitV2(*codes)
    if kind is ObjectiveNumericKindV2.EMPTY:
        raise _PointIncompleteV2(*codes)
    if kind is ObjectiveNumericKindV2.INVALID_INPUT:
        raise _PointUnsupportedV2(*codes)
    raise RuntimeError(f"unknown point numeric outcome: {kind!r}")


def _require_safety(
    outcome: ObjectiveSafetyBoundsOutcomeV2,
) -> ObjectiveSafetyBoundsOutcomeV2:
    if outcome.kind is ObjectiveSafetyBoundsKindV2.EXACT:
        return outcome
    findings = outcome.finding_codes or (f"POINT_SAFETY:{outcome.kind.value}",)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.NUMERIC_GAP:
        raise _PointNumericGapV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.RESOURCE:
        raise _PointResourceLimitV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.EMPTY:
        raise _PointIncompleteV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.UNSUPPORTED:
        raise _PointUnsupportedV2(*findings)
    raise RuntimeError(f"unknown point safety outcome: {outcome.kind!r}")


def _nonnegative_interval(
    outcome: ObjectiveIntervalOutcomeV2,
) -> NonNegativeIntervalV2:
    if outcome.interval is None:
        raise RuntimeError("EXACT point interval omitted its value")
    if outcome.interval.lower_bound < 0.0:
        raise RuntimeError("non-negative point loss published a negative bound")
    return NonNegativeIntervalV2(
        lower_bound=outcome.interval.lower_bound,
        upper_bound=outcome.interval.upper_bound,
    )


def _build_point_term_bounds(
    translation: NonNegativeIntervalV2,
    relation: NonNegativeIntervalV2,
    visibility: NonNegativeIntervalV2,
    safety: NonNegativeIntervalV2,
) -> ObjectiveTermBoundsV2:
    intervals = (translation, relation, visibility, safety)
    if not _point_directed_total_is_finite(
        tuple(item.lower_bound for item in intervals),
        upward=False,
    ) or not _point_directed_total_is_finite(
        tuple(item.upper_bound for item in intervals),
        upward=True,
    ):
        raise _PointNumericGapV2("NUMERIC_GAP:POINT_OBJECTIVE_TERM_TOTAL")
    return ObjectiveTermBoundsV2(
        translation_loss=translation,
        relation_damage_loss=relation,
        visibility_change_loss=visibility,
        safety_margin_loss=safety,
    )


def _point_directed_total_is_finite(
    values: tuple[float, ...],
    *,
    upward: bool,
) -> bool:
    exact = sum((Fraction.from_float(value) for value in values), Fraction())
    try:
        published = float(exact)
    except OverflowError:
        return False
    if not math.isfinite(published):
        return False
    published_exact = Fraction.from_float(published)
    if upward and published_exact < exact:
        published = math.nextafter(published, math.inf)
    elif not upward and published_exact > exact:
        published = math.nextafter(published, -math.inf)
    return math.isfinite(published)


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


def _uncertified(
    reason: UncertifiedReasonV2,
    *findings: str,
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None,
) -> PointObjectiveEvaluationOutcomeV2:
    return PointObjectiveEvaluationOutcomeV2(
        kind=PointObjectiveEvaluationKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=tuple(findings),
        cumulative_generation_usage=cumulative_generation_usage,
    )


def _not_proven(
    *findings: str,
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None,
) -> PointObjectiveEvaluationOutcomeV2:
    return PointObjectiveEvaluationOutcomeV2(
        kind=PointObjectiveEvaluationKindV2.NOT_PROVEN,
        finding_codes=tuple(findings),
        cumulative_generation_usage=cumulative_generation_usage,
    )
