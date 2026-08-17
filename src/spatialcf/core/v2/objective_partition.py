"""Verified, platform-neutral compilation of Canonical v2 objective bounds.

The compiler accepts published upstream artifacts only as submissions.  It
first freezes all four inputs, asks the independent replay verifier to resolve
the complete candidate/relation chain, and uses the byte-equal strict snapshots
as the data source for objective construction.  No unverified resource counter,
relation vector, or domain ledger is used to make an objective claim.

Objective cells are a one-to-one lift of verified relation cells.  Each cell
contains conservative four-term loss intervals and complete safety slack
intervals.  A nearest inner-domain edit may be returned as a proposal, but it
is deliberately not labelled feasible, optimal, or assigned a loss; those are
later verifier/certificate responsibilities.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import ClassVar, TypeVar

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2._internal.resources.domain_operations import (
    BaseUsageDomainOperationBudgetV2,
)
from spatialcf.core.v2.artifact_verifier import (
    CoreArtifactVerificationKindV2,
    verify_core_artifacts_v2,
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
from spatialcf.core.v2.objective_safety_bounds import (
    ObjectiveSafetyBoundsKindV2,
    ObjectiveSafetyBoundsOutcomeV2,
    compile_objective_safety_bounds_v2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    intersect_rectilinear_regions_v2,
    lift_planar_region_v2,
    normalize_rectilinear_region_v2,
)
from spatialcf.core.v2.relation_cost_partition import (
    RelationCostPartitionCompilationKindV2,
    RelationCostPartitionCompilationOutcomeV2,
)
from spatialcf.domain.v2.artifacts import (
    ArtifactCoverageV2,
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    ConstraintSlackV2,
    NonNegativeIntervalV2,
    ObjectivePartitionArtifactV2,
    ObjectivePartitionCellV2,
    ObjectiveTermBoundsV2,
    PlanarRegionBoundV2,
    RegionBoundStatusV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import (
    CanonicalId,
    FactAvailabilityV2,
    FactCompletenessV2,
    Sha256Digest,
    V2Model,
    Vec2V2,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.objective import ObjectCameraKeyV2, SafetySlackUnitV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import CoreSolverConfigV2, UncertifiedReasonV2
from spatialcf.domain.v2.scene import BaselineObservationV2


class ObjectivePartitionCompilationKindV2(StrEnum):
    PARTITION = "PARTITION"
    UNCERTIFIED = "UNCERTIFIED"


_CANONICAL_ID_ADAPTER = TypeAdapter(CanonicalId)
_SHA256_DIGEST_ADAPTER = TypeAdapter(Sha256Digest)
# Cell construction, artifact construction, explicit strict reconstruction,
# and outcome reconstruction each traverse the published vectors once.
_OBJECTIVE_PUBLICATION_VALIDATION_PASSES_V2 = 4


@dataclass(frozen=True, slots=True)
class ObjectiveWitnessProposalV2:
    """One unscored, unclaimed edit proposed from an exact inner-domain point."""

    objective_cell_id: CanonicalId
    parent_relation_cell_id: CanonicalId
    edit: CanonicalEditV2

    def __post_init__(self) -> None:
        try:
            objective_cell_id = _CANONICAL_ID_ADAPTER.validate_python(
                self.objective_cell_id,
                strict=True,
            )
            parent_relation_cell_id = _CANONICAL_ID_ADAPTER.validate_python(
                self.parent_relation_cell_id,
                strict=True,
            )
        except ValidationError as error:
            raise TypeError("proposal cell IDs must be CanonicalId values") from error
        if type(self.edit) is not CanonicalEditV2:
            raise TypeError("proposal edit must be a CanonicalEditV2")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                edit = CanonicalEditV2.model_validate(
                    self.edit.model_dump(mode="python"),
                    strict=True,
                )
        except (ValidationError, TypeError, ValueError, Warning) as error:
            raise TypeError("proposal edit must pass strict validation") from error
        object.__setattr__(self, "objective_cell_id", objective_cell_id)
        object.__setattr__(
            self,
            "parent_relation_cell_id",
            parent_relation_cell_id,
        )
        object.__setattr__(self, "edit", edit)


@dataclass(frozen=True, slots=True)
class ObjectivePartitionCompilationOutcomeV2:
    """Closed objective-compilation result with one cumulative core ledger."""

    kind: ObjectivePartitionCompilationKindV2
    objective_partition: ObjectivePartitionArtifactV2 | None = None
    witness_proposals: tuple[ObjectiveWitnessProposalV2, ...] = ()
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()
    cumulative_resource_usage: CompilationResourceUsageV2 | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ObjectivePartitionCompilationKindV2:
            raise TypeError("kind must be an ObjectivePartitionCompilationKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be an exact tuple of non-blank strings")
        if type(self.witness_proposals) is not tuple or any(
            type(item) is not ObjectiveWitnessProposalV2
            for item in self.witness_proposals
        ):
            raise TypeError("witness_proposals must be an exact proposal tuple")
        findings = tuple(sorted(set(self.finding_codes)))
        proposals = tuple(
            ObjectiveWitnessProposalV2(
                objective_cell_id=item.objective_cell_id,
                parent_relation_cell_id=item.parent_relation_cell_id,
                edit=item.edit,
            )
            for item in self.witness_proposals
        )
        proposal_ids = tuple(item.objective_cell_id for item in proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("witness proposals must have unique objective cell IDs")
        proposals = tuple(sorted(proposals, key=lambda item: item.objective_cell_id))

        artifact = self.objective_partition
        if artifact is not None:
            if type(artifact) is not ObjectivePartitionArtifactV2:
                raise TypeError(
                    "objective_partition must be an ObjectivePartitionArtifactV2"
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    artifact = ObjectivePartitionArtifactV2.model_validate(
                        artifact.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "objective_partition must pass strict validation"
                ) from error
        usage = self.cumulative_resource_usage
        if usage is not None:
            if type(usage) is not CompilationResourceUsageV2:
                raise TypeError(
                    "cumulative_resource_usage must be a CompilationResourceUsageV2"
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
                    "cumulative_resource_usage must pass strict validation"
                ) from error

        object.__setattr__(self, "finding_codes", findings)
        object.__setattr__(self, "witness_proposals", proposals)
        object.__setattr__(self, "objective_partition", artifact)
        object.__setattr__(self, "cumulative_resource_usage", usage)

        if self.kind is ObjectivePartitionCompilationKindV2.PARTITION:
            if artifact is None or usage is None:
                raise ValueError("PARTITION requires an artifact and cumulative usage")
            if self.uncertified_reason is not None or findings:
                raise ValueError("PARTITION cannot carry failure diagnostics")
            artifact_ids = {cell.cell_id for cell in artifact.cells}
            relation_by_objective = {
                cell.cell_id: cell.parent_relation_cell_id for cell in artifact.cells
            }
            if not set(proposal_ids) <= artifact_ids:
                raise ValueError("proposal references an unknown objective cell")
            if any(
                relation_by_objective[item.objective_cell_id]
                != item.parent_relation_cell_id
                for item in proposals
            ):
                raise ValueError(
                    "proposal parent relation cell reference is not closed"
                )
            return

        if artifact is not None or proposals:
            raise ValueError("UNCERTIFIED cannot carry partial artifacts or proposals")
        if type(self.uncertified_reason) is not UncertifiedReasonV2:
            raise TypeError("UNCERTIFIED requires an UncertifiedReasonV2")
        if not findings:
            raise ValueError("UNCERTIFIED requires at least one finding")


ObjectivePartitionCompileOutcomeV2 = ObjectivePartitionCompilationOutcomeV2


class _ResourceLimitV2(RuntimeError):
    pass


@dataclass(slots=True)
class _DomainOperationBudgetV2(BaseUsageDomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = _ResourceLimitV2
    _exhaustion_error_message: ClassVar[str] = "RESOURCE_LIMIT:max_domain_operations"


class _UnsupportedV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _NumericGapV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _CompilationIncompleteV2(RuntimeError):
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


def compile_objective_partition_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionCompilationOutcomeV2,
) -> ObjectivePartitionCompilationOutcomeV2:
    """Compile objective bounds only after complete independent artifact replay."""

    try:
        checked_problem = _strict_model(problem, SemanticProblemV2, "SEMANTIC_PROBLEM")
        checked_config = _strict_model(config, CoreSolverConfigV2, "CORE_SOLVER_CONFIG")
        checked_candidate = _strict_model(
            candidate,
            CandidateDomainArtifactV2,
            "CANDIDATE_DOMAIN",
        )
        checked_relation = _strict_relation_outcome(relation)
    except _NumericInputV2 as error:
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            error.finding_code,
        )
    except _InvalidInputV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            verification = verify_core_artifacts_v2(
                checked_problem,
                checked_config,
                checked_candidate,
                checked_relation,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:OBJECTIVE_ARTIFACT_VERIFICATION",
        )

    if verification.kind is not CoreArtifactVerificationKindV2.VERIFIED:
        reason = (
            verification.uncertified_reason
            if verification.kind is CoreArtifactVerificationKindV2.UNCERTIFIED
            else UncertifiedReasonV2.COMPILATION_INCOMPLETE
        )
        return _uncertified(
            reason or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(verification.finding_codes or ("OBJECTIVE_UPSTREAM_NOT_VERIFIED",)),
            cumulative_resource_usage=verification.verification_resource_usage,
        )
    base_usage = verification.verification_resource_usage
    if base_usage is None:  # pragma: no cover - verifier outcome invariant
        raise RuntimeError("VERIFIED upstream outcome omitted resource usage")

    # The verifier proved the strict snapshots byte-for-byte identical to its
    # one candidate/relation replay.  Replaying yet again here would perform
    # unledgered work; the verified snapshots are therefore the closed source.
    if (
        checked_relation.kind is not RelationCostPartitionCompilationKindV2.PARTITION
        or checked_relation.relation_cost_partition is None
    ):
        raise RuntimeError("VERIFIED relation snapshot omitted its partition")
    verified_relation = checked_relation.relation_cost_partition
    _require_verified_replay_closure(
        checked_problem,
        checked_config,
        checked_candidate,
        verified_relation,
        verification.semantic_problem_sha256,
        verification.core_solver_config_sha256,
        verification.candidate_domain_artifact_sha256,
        verification.relation_cost_partition_sha256,
    )

    domain_budget = _DomainOperationBudgetV2(
        limit=checked_config.max_domain_operations,
        base_used=base_usage.domain_operations,
    )
    atomic_budget = RectilinearAtomicBudgetV2(
        limit=checked_config.max_partition_cells,
        used=base_usage.partition_cells,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_verified_partition(
                checked_problem,
                checked_config,
                checked_candidate,
                verified_relation,
                base_usage,
                domain_budget,
                atomic_budget,
            )
    except (RectilinearAtomicBudgetExhaustedV2, _ResourceLimitV2) as error:
        finding = str(error) or "RESOURCE_LIMIT:max_partition_cells"
        return _uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            finding,
            cumulative_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )
    except _NumericGapV2 as error:
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            *error.finding_codes,
            cumulative_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )
    except _UnsupportedV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            cumulative_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )
    except _CompilationIncompleteV2 as error:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *error.finding_codes,
            cumulative_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:OBJECTIVE_PARTITION",
            cumulative_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )


class ObjectivePartitionCompilerV2:
    """Stateless object wrapper for pipeline composition."""

    def compile(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        candidate: CandidateDomainArtifactV2,
        relation: RelationCostPartitionCompilationOutcomeV2,
    ) -> ObjectivePartitionCompilationOutcomeV2:
        return compile_objective_partition_v2(problem, config, candidate, relation)


def _compile_verified_partition(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionV2,
    base_usage: CompilationResourceUsageV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ObjectivePartitionCompilationOutcomeV2:
    if (
        relation.cell_outer_union_coverage
        is not ArtifactCoverageV2.EXACT_OUTER_COVERAGE
    ):
        raise _CompilationIncompleteV2(
            "COMPILATION_INCOMPLETE:RELATION_CELL_OUTER_COVERAGE"
        )
    if not relation.cells:
        raise _CompilationIncompleteV2("COMPILATION_INCOMPLETE:NO_RELATION_CELLS")

    safety_target_count = len(problem.objective.safety_margin.aggregation.targets)
    # Reserve every O(cells * (keys + targets)) publication pass before any
    # nested cell/vector construction.  A cap miss therefore performs no
    # partial publication work and emits no partial artifact.
    publication_units = _OBJECTIVE_PUBLICATION_VALIDATION_PASSES_V2 * sum(
        1 + len(cell.relation_damage_vector) + safety_target_count
        for cell in relation.cells
    )
    domain_budget.consume(publication_units)

    candidate_outer = _lift_bound(
        candidate.hard_domain.outer_bound,
        domain_budget,
        atomic_budget,
    )
    exact_cells: list[
        tuple[ExactRectilinearRegionV2 | None, ExactRectilinearRegionV2]
    ] = []
    # This is a separate pass over cells for coverage/topology compilation.
    domain_budget.consume(len(relation.cells))
    for relation_cell in relation.cells:
        atomic_budget.consume(1)
        inner = _lift_optional_inner_bound(
            relation_cell.domain.inner_bound,
            domain_budget,
            atomic_budget,
        )
        outer = _lift_bound(
            relation_cell.domain.outer_bound,
            domain_budget,
            atomic_budget,
        )
        if inner is not None:
            domain_budget.consume()
            intersection = _require_exact_region(
                intersect_rectilinear_regions_v2(
                    inner,
                    outer,
                    atomic_budget=atomic_budget,
                )
            )
            if intersection != inner:
                raise _UnsupportedV2("INVALID_RELATION_CELL:INNER_NOT_SUBSET_OUTER")
        exact_cells.append((inner, outer))

    domain_budget.consume()
    outer_union = _require_exact_region(
        normalize_rectilinear_region_v2(
            tuple(
                rectangle for _, outer in exact_cells for rectangle in outer.rectangles
            ),
            atomic_budget=atomic_budget,
        )
    )
    if outer_union != candidate_outer:
        raise _CompilationIncompleteV2(
            "COMPILATION_INCOMPLETE:RELATION_CELL_OUTER_UNION"
        )

    visibility_intervals = _visibility_intervals(problem, domain_budget)
    domain_budget.consume(len(visibility_intervals))
    visibility_loss = _require_exact_interval(
        aggregate_visibility_change_bounds_v2(
            problem.objective.visibility_change,
            visibility_intervals,
        ),
        "VISIBILITY_CHANGE",
    )

    cells: list[ObjectivePartitionCellV2] = []
    proposals: list[ObjectiveWitnessProposalV2] = []
    domain_budget.consume(len(relation.cells))
    for index, (relation_cell, exact_pair) in enumerate(
        zip(relation.cells, exact_cells, strict=True)
    ):
        inner, outer = exact_pair
        translation = _require_exact_translation(
            compile_translation_l2_cell_bounds_v2(
                outer,
                problem.objective.translation,
                atomic_budget=atomic_budget,
            )
        )
        domain_budget.consume(len(relation_cell.relation_damage_vector))
        relation_loss = _require_exact_interval(
            aggregate_relation_damage_bounds_v2(
                problem.objective.relation_damage,
                relation_cell.relation_damage_vector,
            ),
            "RELATION_DAMAGE",
        )
        domain_budget.consume(safety_target_count)
        safety = _require_exact_safety(
            compile_objective_safety_bounds_v2(
                problem,
                outer,
                atomic_budget=atomic_budget,
            )
        )
        domain_budget.consume(safety_target_count)
        safety_loss = _require_exact_interval(
            aggregate_safety_penalty_bounds_v2(
                problem.objective.safety_margin,
                tuple(
                    ConstraintSafetyInputV2(
                        constraint_id=item.constraint_id,
                        components=item.raw_components,
                    )
                    for item in safety.constraint_bounds
                ),
            ),
            "SAFETY_MARGIN",
        )

        objective_cell_id = f"cell:objective:{index:06d}"
        term_loss_bounds = _build_term_bounds(
            NonNegativeIntervalV2(
                lower_bound=translation.bounds.lower_bound,
                upper_bound=translation.bounds.upper_bound,
            ),
            _published_nonnegative(relation_loss),
            _published_nonnegative(visibility_loss),
            _published_nonnegative(safety_loss),
        )
        cells.append(
            ObjectivePartitionCellV2(
                cell_id=objective_cell_id,
                parent_relation_cell_id=relation_cell.cell_id,
                domain=relation_cell.domain,
                relation_damage_vector=relation_cell.relation_damage_vector,
                term_loss_bounds=term_loss_bounds,
                constraint_slacks=tuple(
                    ConstraintSlackV2(
                        constraint_id=item.constraint_id,
                        lower_bound=item.normalized_slack.lower_bound,
                        upper_bound=item.normalized_slack.upper_bound,
                        unit=SafetySlackUnitV2.DIMENSIONLESS,
                    )
                    for item in safety.constraint_bounds
                ),
            )
        )

        if inner is not None:
            proposal = _inner_nearest_proposal(
                problem,
                objective_cell_id,
                relation_cell.cell_id,
                inner,
                atomic_budget,
            )
            if proposal is not None:
                proposals.append(proposal)

    artifact = ObjectivePartitionArtifactV2(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        core_solver_config_sha256=config.core_solver_config_sha256,
        candidate_domain_artifact_sha256=candidate.candidate_domain_artifact_sha256,
        relation_cost_partition_sha256=relation.relation_cost_partition_sha256,
        objective_spec_sha256=problem.objective.objective_spec_sha256,
        cells=tuple(cells),
        cell_outer_union_coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )
    checked_artifact = ObjectivePartitionArtifactV2.model_validate(
        artifact.model_dump(mode="python"),
        strict=True,
    )
    # Reassert the one-to-one lift after schema canonicalization.
    expected_mapping = tuple(
        (
            f"cell:objective:{index:06d}",
            cell.cell_id,
            cell.domain,
            cell.relation_damage_vector,
        )
        for index, cell in enumerate(relation.cells)
    )
    actual_mapping = tuple(
        (
            cell.cell_id,
            cell.parent_relation_cell_id,
            cell.domain,
            cell.relation_damage_vector,
        )
        for cell in checked_artifact.cells
    )
    if actual_mapping != expected_mapping:
        raise RuntimeError("objective/relation cell mapping invariant drift")
    return ObjectivePartitionCompilationOutcomeV2(
        kind=ObjectivePartitionCompilationKindV2.PARTITION,
        objective_partition=checked_artifact,
        witness_proposals=tuple(proposals),
        cumulative_resource_usage=_cumulative_usage(
            base_usage,
            domain_budget,
            atomic_budget,
        ),
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
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise _NumericInputV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}") from error


def _strict_relation_outcome(
    value: object,
) -> RelationCostPartitionCompilationOutcomeV2:
    label = "RELATION_COST_PARTITION_OUTCOME"
    if type(value) is not RelationCostPartitionCompilationOutcomeV2:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}:TYPE")
    partition = value.relation_cost_partition
    usage = value.cumulative_resource_usage
    if partition is not None and type(partition) is not RelationCostPartitionV2:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    if usage is not None and type(usage) is not CompilationResourceUsageV2:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    if type(value.finding_codes) is not tuple or any(
        type(code) is not str or not code.strip() for code in value.finding_codes
    ):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if partition is not None:
                partition = RelationCostPartitionV2.model_validate(
                    partition.model_dump(mode="python"),
                    strict=True,
                )
            if usage is not None:
                usage = CompilationResourceUsageV2.model_validate(
                    usage.model_dump(mode="python"),
                    strict=True,
                )
            return RelationCostPartitionCompilationOutcomeV2(
                kind=value.kind,
                relation_cost_partition=partition,
                uncertified_reason=value.uncertified_reason,
                finding_codes=tuple(value.finding_codes),
                cumulative_resource_usage=usage,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise _NumericInputV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}") from error


def _require_verified_replay_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionV2,
    problem_hash: str | None,
    config_hash: str | None,
    candidate_hash: str | None,
    relation_hash: str | None,
) -> None:
    expected = (
        problem.semantic_problem_sha256,
        config.core_solver_config_sha256,
        candidate.candidate_domain_artifact_sha256,
        relation.relation_cost_partition_sha256,
    )
    observed = (problem_hash, config_hash, candidate_hash, relation_hash)
    if observed != expected:
        raise RuntimeError("verified replay hash closure drift")
    for value in expected:
        _SHA256_DIGEST_ADAPTER.validate_python(value, strict=True)


def _lift_bound(
    bound: PlanarRegionBoundV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    domain_budget.consume()
    if bound.status is RegionBoundStatusV2.EMPTY:
        return _require_exact_region(
            normalize_rectilinear_region_v2((), atomic_budget=atomic_budget)
        )
    if bound.status is not RegionBoundStatusV2.NON_EMPTY or bound.region is None:
        raise _CompilationIncompleteV2("COMPILATION_INCOMPLETE:UNAVAILABLE_CELL_BOUND")
    return _require_exact_region(
        lift_planar_region_v2(bound.region, atomic_budget=atomic_budget)
    )


def _lift_optional_inner_bound(
    bound: PlanarRegionBoundV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | None:
    domain_budget.consume()
    if bound.status is RegionBoundStatusV2.UNAVAILABLE:
        return None
    if bound.status is RegionBoundStatusV2.EMPTY:
        return _require_exact_region(
            normalize_rectilinear_region_v2((), atomic_budget=atomic_budget)
        )
    if bound.status is not RegionBoundStatusV2.NON_EMPTY or bound.region is None:
        raise _CompilationIncompleteV2("COMPILATION_INCOMPLETE:INVALID_INNER_BOUND")
    return _require_exact_region(
        lift_planar_region_v2(bound.region, atomic_budget=atomic_budget)
    )


def _require_exact_region(
    outcome: RectilinearRegionOutcomeV2,
) -> ExactRectilinearRegionV2:
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _ResourceLimitV2("RESOURCE_LIMIT:max_partition_cells")
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise _UnsupportedV2(*(outcome.finding_codes or ("UNSUPPORTED_CELL_REGION",)))
    return outcome.region


def _visibility_intervals(
    problem: SemanticProblemV2,
    domain_budget: _DomainOperationBudgetV2,
) -> tuple[VisibilityMetricIntervalV2, ...]:
    facts = problem.scene.baseline_observations
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.values is None
    ):
        raise _CompilationIncompleteV2(
            "COMPILATION_INCOMPLETE:VISIBILITY_BASELINE_NOT_EXACT"
        )
    domain_budget.consume(len(facts.values))
    observations: dict[tuple[str, str, str, str], BaselineObservationV2] = {}
    for observation in facts.values:
        key = (
            observation.object_id,
            observation.camera_id,
            observation.metric_definition_id,
            observation.metric_definition_version,
        )
        if key in observations:
            raise _UnsupportedV2("INVALID_VISIBILITY_BASELINE:DUPLICATE_KEY")
        observations[key] = observation

    intervals: list[VisibilityMetricIntervalV2] = []
    domain_budget.consume(
        len(problem.objective.visibility_change.object_camera_weights)
    )
    for weighted in problem.objective.visibility_change.object_camera_weights:
        key = weighted.key
        semantic_key = (
            key.object_id,
            key.camera_id,
            key.metric_definition_id,
            key.metric_definition_version,
        )
        observation = observations.get(semantic_key)
        if observation is None:
            raise _CompilationIncompleteV2(
                "COMPILATION_INCOMPLETE:VISIBILITY_BASELINE_KEY"
            )
        intervals.append(
            VisibilityMetricIntervalV2(
                key=ObjectCameraKeyV2.model_validate(
                    key.model_dump(mode="python"),
                    strict=True,
                ),
                baseline_lower=Fraction.from_float(observation.normalized_lower_bound),
                baseline_upper=Fraction.from_float(observation.normalized_upper_bound),
                # Candidate visibility is deliberately not guessed from a
                # platform renderer.  [0, 1] is the sound semantic range.
                candidate_lower=Fraction(),
                candidate_upper=Fraction(1),
            )
        )
    return tuple(intervals)


def _require_exact_translation(
    outcome: TranslationCellOutcomeV2,
) -> TranslationCellOutcomeV2:
    if outcome.kind is ObjectiveNumericKindV2.EXACT and outcome.bounds is not None:
        return outcome
    _raise_numeric_outcome(
        outcome.kind,
        outcome.finding_codes,
        "TRANSLATION",
    )
    raise AssertionError("unreachable")


def _require_exact_interval(
    outcome: ObjectiveIntervalOutcomeV2,
    label: str,
) -> ObjectiveIntervalOutcomeV2:
    if outcome.kind is ObjectiveNumericKindV2.EXACT and outcome.interval is not None:
        return outcome
    _raise_numeric_outcome(outcome.kind, outcome.finding_codes, label)
    raise AssertionError("unreachable")


def _raise_numeric_outcome(
    kind: ObjectiveNumericKindV2,
    findings: tuple[str, ...],
    label: str,
) -> None:
    codes = findings or (f"OBJECTIVE_{label}:{kind.value}",)
    if kind is ObjectiveNumericKindV2.NUMERIC_GAP:
        raise _NumericGapV2(*codes)
    if kind is ObjectiveNumericKindV2.RESOURCE_LIMIT:
        raise _ResourceLimitV2(*codes)
    if kind is ObjectiveNumericKindV2.EMPTY:
        raise _CompilationIncompleteV2(*codes)
    if kind is ObjectiveNumericKindV2.INVALID_INPUT:
        raise _UnsupportedV2(*codes)
    raise RuntimeError(f"unknown objective numeric outcome: {kind!r}")


def _require_exact_safety(
    outcome: ObjectiveSafetyBoundsOutcomeV2,
) -> ObjectiveSafetyBoundsOutcomeV2:
    if outcome.kind is ObjectiveSafetyBoundsKindV2.EXACT:
        return outcome
    findings = outcome.finding_codes or (f"OBJECTIVE_SAFETY:{outcome.kind.value}",)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.NUMERIC_GAP:
        raise _NumericGapV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.RESOURCE:
        raise _ResourceLimitV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.EMPTY:
        raise _CompilationIncompleteV2(*findings)
    if outcome.kind is ObjectiveSafetyBoundsKindV2.UNSUPPORTED:
        raise _UnsupportedV2(*findings)
    raise RuntimeError(f"unknown objective safety outcome: {outcome.kind!r}")


def _published_nonnegative(
    outcome: ObjectiveIntervalOutcomeV2,
) -> NonNegativeIntervalV2:
    interval = outcome.interval
    if interval is None:  # pragma: no cover - exact-outcome invariant
        raise RuntimeError("EXACT objective interval omitted its value")
    if interval.lower_bound < 0.0:
        raise RuntimeError("non-negative objective term published a negative bound")
    return NonNegativeIntervalV2(
        lower_bound=interval.lower_bound,
        upper_bound=interval.upper_bound,
    )


def _build_term_bounds(
    translation: NonNegativeIntervalV2,
    relation: NonNegativeIntervalV2,
    visibility: NonNegativeIntervalV2,
    safety: NonNegativeIntervalV2,
) -> ObjectiveTermBoundsV2:
    intervals = (translation, relation, visibility, safety)
    if not _directed_total_is_finite(
        tuple(item.lower_bound for item in intervals),
        upward=False,
    ) or not _directed_total_is_finite(
        tuple(item.upper_bound for item in intervals),
        upward=True,
    ):
        raise _NumericGapV2("NUMERIC_GAP:OBJECTIVE_TERM_TOTAL")
    return ObjectiveTermBoundsV2(
        translation_loss=translation,
        relation_damage_loss=relation,
        visibility_change_loss=visibility,
        safety_margin_loss=safety,
    )


def _directed_total_is_finite(
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


def _inner_nearest_proposal(
    problem: SemanticProblemV2,
    objective_cell_id: str,
    parent_relation_cell_id: str,
    inner: ExactRectilinearRegionV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ObjectiveWitnessProposalV2 | None:
    if not inner.rectangles:
        return None
    nearest = _require_exact_translation(
        compile_translation_l2_cell_bounds_v2(
            inner,
            problem.objective.translation,
            atomic_budget=atomic_budget,
        )
    )
    bounds = nearest.bounds
    if bounds is None:  # pragma: no cover - exact-outcome invariant
        raise RuntimeError("EXACT inner translation omitted its witness")
    delta_x = _exact_binary64(bounds.nearest_delta_x_m)
    delta_y = _exact_binary64(bounds.nearest_delta_y_m)
    if delta_x is None or delta_y is None:
        return None
    # Membership is a new linear pass, distinct from the nearest-point pass.
    atomic_budget.consume(len(inner.rectangles))
    if not inner.contains_point(
        Fraction.from_float(delta_x),
        Fraction.from_float(delta_y),
    ):
        raise RuntimeError("inner nearest proposal escaped its exact cell")
    return ObjectiveWitnessProposalV2(
        objective_cell_id=objective_cell_id,
        parent_relation_cell_id=parent_relation_cell_id,
        edit=CanonicalEditV2(
            semantic_problem_sha256=problem.semantic_problem_sha256,
            subject_id=problem.constraints.allowed_edit.subject_id,
            translation_xy_m=Vec2V2(x=delta_x, y=delta_y),
        ),
    )


def _exact_binary64(value: Fraction) -> float | None:
    try:
        published = float(value)
    except OverflowError:
        return None
    if not math.isfinite(published) or Fraction.from_float(published) != value:
        return None
    return 0.0 if published == 0.0 else published


def _cumulative_usage(
    base_usage: CompilationResourceUsageV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=base_usage.domain_operations + domain_budget.used,
        partition_cells=atomic_budget.used,
        refinement_steps=base_usage.refinement_steps,
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *findings: str,
    cumulative_resource_usage: CompilationResourceUsageV2 | None = None,
) -> ObjectivePartitionCompilationOutcomeV2:
    return ObjectivePartitionCompilationOutcomeV2(
        kind=ObjectivePartitionCompilationKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=tuple(findings),
        cumulative_resource_usage=cumulative_resource_usage,
    )
