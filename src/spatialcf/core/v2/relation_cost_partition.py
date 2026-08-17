"""Proof-oriented single-cell relation-damage bounds.

The emitted artifact contains bounds only.  An independent verifier must
resolve every hash, repeat the exact domain lifts, and recompute all relation
measurements before using it in a certificate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import ClassVar

from pydantic import ValidationError

from spatialcf.core.v2._internal.resources.domain_operations import (
    RemainingDomainOperationBudgetV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M,
    RECTILINEAR_KERNEL_ID_V2,
    RECTILINEAR_KERNEL_VERSION_V2,
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    intersect_rectilinear_regions_v2,
    lift_planar_region_v2,
    normalize_rectilinear_region_v2,
)
from spatialcf.core.v2.visibility_domain import (
    VisibilityDomainKindV2,
    compile_visibility_domain_v2,
)
from spatialcf.domain.v2.artifacts import (
    ArtifactCoverageV2,
    CandidateCompilationCoverageV2,
    CandidateConstraintKindV2,
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    ConstraintCompilationDispositionV2,
    DomainCompletenessV2,
    PlanarRegionBoundV2,
    RegionBoundStatusV2,
    RelationCostCellV2,
    RelationCostPartitionV2,
    RelationDamageBoundV2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    QuaternionV2,
    RigidTransformV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    MeasurementComparatorV2,
    MeasurementOperandOrderV2,
    RelationAxisV2,
    RelationDefinitionV2,
    RelationMeasurementV2,
    RelationRepresentativePointV2,
    RelationV2,
    VisibilityConstraintV2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.objective import PairAxisKeyV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    DirectedOutwardGeometryKernelSpecV2,
    UncertifiedReasonV2,
)
from spatialcf.domain.v2.scene import (
    CameraAxesV2,
    CameraDepthConventionV2,
    CameraDistortionModelV2,
    CameraMatrixLayoutV2,
    CameraPixelConventionV2,
    CanonicalObjectV2,
    PinholeCameraV2,
)

RELATION_COST_PARTITION_ALGORITHM_ID_V2 = "solver:canonical-branch-and-bound-v2"
RELATION_COST_PARTITION_ALGORITHM_VERSION_V2 = "algorithm:2.0"
RELATION_COST_PARTITION_DOMAIN_OPERATIONS_V2 = 5

_MAX_CERTIFIED_FRACTION_BITS = 4096


class RelationCostPartitionCompilationKindV2(StrEnum):
    PARTITION = "PARTITION"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class RelationCostPartitionCompilationOutcomeV2:
    """Closed compilation result with an auditable cumulative resource ledger.

    A successful partition always carries usage.  An uncertified result carries
    usage once all three inputs have passed strict reconstruction; ``None``
    means validation failed before the candidate ledger could be trusted.
    """

    kind: RelationCostPartitionCompilationKindV2
    relation_cost_partition: RelationCostPartitionV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()
    cumulative_resource_usage: CompilationResourceUsageV2 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RelationCostPartitionCompilationKindV2):
            raise TypeError("kind must be a RelationCostPartitionCompilationKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.cumulative_resource_usage is not None and not isinstance(
            self.cumulative_resource_usage,
            CompilationResourceUsageV2,
        ):
            raise TypeError(
                "cumulative_resource_usage must be a CompilationResourceUsageV2"
            )
        if self.kind is RelationCostPartitionCompilationKindV2.PARTITION:
            if not isinstance(self.relation_cost_partition, RelationCostPartitionV2):
                raise ValueError("PARTITION requires a relation-cost partition")
            if self.uncertified_reason is not None or self.finding_codes:
                raise ValueError("PARTITION cannot carry uncertified diagnostics")
            if self.cumulative_resource_usage is None:
                raise ValueError("PARTITION requires cumulative resource usage")
            return
        if self.relation_cost_partition is not None:
            raise ValueError("UNCERTIFIED cannot carry a partition")
        if not isinstance(self.uncertified_reason, UncertifiedReasonV2):
            raise TypeError("UNCERTIFIED requires an uncertified reason")
        if not self.finding_codes:
            raise ValueError("UNCERTIFIED requires a finding")


RelationCostPartitionCompileOutcomeV2 = RelationCostPartitionCompilationOutcomeV2


class _ResourceLimitV2(RuntimeError):
    pass


@dataclass(slots=True)
class _DomainOperationBudgetV2(RemainingDomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = _ResourceLimitV2
    _exhaustion_error_message: ClassVar[str] = "RESOURCE_LIMIT:max_domain_operations"


@dataclass(frozen=True, slots=True)
class _ExactContextV2:
    problem: SemanticProblemV2
    candidate: CandidateDomainArtifactV2
    outer_domain: ExactRectilinearRegionV2
    subject_id: str
    objects: dict[str, CanonicalObjectV2]
    relation_geometries: dict[str, GeometryInstanceV2]
    camera: PinholeCameraV2
    visibility_proofs: _VisibilityProofStateV2


@dataclass(slots=True)
class _VisibilityProofStateV2:
    domain_budget: _DomainOperationBudgetV2
    atomic_budget: RectilinearAtomicBudgetV2
    after_cache: dict[
        tuple[str, ExactRectilinearRegionV2],
        bool,
    ]


class _UnsupportedDomainV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _InexactRelationMeasurementV2(RuntimeError):
    """Expected certified-subset miss, distinct from an invariant failure."""


def compile_relation_cost_partition_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> RelationCostPartitionCompilationOutcomeV2:
    """Compile one conservative relation-cost cell over a COMPLETE hard domain."""

    checked_problem = _strict_problem(problem)
    if isinstance(checked_problem, RelationCostPartitionCompilationOutcomeV2):
        return checked_problem
    checked_config = _strict_config(config)
    if isinstance(checked_config, RelationCostPartitionCompilationOutcomeV2):
        return checked_config
    checked_candidate = _strict_candidate(candidate)
    if isinstance(checked_candidate, RelationCostPartitionCompilationOutcomeV2):
        return checked_candidate

    base_usage = checked_candidate.resource_usage

    registry_finding = _registry_finding(checked_config)
    if registry_finding is not None:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
            cumulative_resource_usage=base_usage,
        )

    closure_findings = _candidate_closure_findings(
        checked_problem,
        checked_config,
        checked_candidate,
    )
    if closure_findings:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *closure_findings,
            cumulative_resource_usage=base_usage,
        )

    missing = tuple(
        code
        for code in checked_problem.certification_blockers
        if code.startswith("MISSING_FACT:")
    )
    if missing:
        return _uncertified(
            UncertifiedReasonV2.MISSING_FACT,
            *missing,
            cumulative_resource_usage=base_usage,
        )

    resource_finding = _resource_preflight(checked_config, checked_candidate)
    if resource_finding is not None:
        return _uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            resource_finding,
            cumulative_resource_usage=base_usage,
        )

    remaining_operations = (
        checked_config.max_domain_operations
        - checked_candidate.resource_usage.domain_operations
    )
    budget = _DomainOperationBudgetV2(remaining=remaining_operations)
    atomic_budget = RectilinearAtomicBudgetV2(
        limit=checked_config.max_partition_cells,
        used=checked_candidate.resource_usage.partition_cells,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _compile_checked_partition(
                checked_problem,
                checked_config,
                checked_candidate,
                budget,
                atomic_budget,
            )
    except (_ResourceLimitV2, RectilinearAtomicBudgetExhaustedV2) as error:
        finding = str(error) or "RESOURCE_LIMIT:max_partition_cells"
        return _uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            finding,
            cumulative_resource_usage=_cumulative_usage(
                checked_candidate,
                budget,
                atomic_budget,
            ),
        )
    except _UnsupportedDomainV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            cumulative_resource_usage=_cumulative_usage(
                checked_candidate,
                budget,
                atomic_budget,
            ),
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_COST_PARTITION",
            cumulative_resource_usage=_cumulative_usage(
                checked_candidate,
                budget,
                atomic_budget,
            ),
        )


def _compile_checked_partition(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> RelationCostPartitionCompilationOutcomeV2:
    atomic_budget.consume(1)
    exact_inner = _lift_bound(
        candidate.hard_domain.inner_bound,
        budget,
        atomic_budget,
    )
    exact_outer = _lift_bound(
        candidate.hard_domain.outer_bound,
        budget,
        atomic_budget,
    )
    _verify_inner_subset(exact_inner, exact_outer, budget, atomic_budget)

    vector = _relation_damage_vector(
        problem,
        candidate,
        exact_outer,
        budget,
        atomic_budget,
    )
    partition = RelationCostPartitionV2(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        core_solver_config_sha256=config.core_solver_config_sha256,
        candidate_domain_artifact_sha256=(candidate.candidate_domain_artifact_sha256),
        objective_spec_sha256=problem.objective.objective_spec_sha256,
        cells=(
            RelationCostCellV2(
                cell_id="cell:relation-cost:000000",
                domain=candidate.hard_domain,
                relation_damage_vector=vector,
            ),
        ),
        cell_outer_union_coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )
    checked_partition = RelationCostPartitionV2.model_validate(
        partition.model_dump(mode="python"),
        strict=True,
    )
    _verify_partition_outer_union(
        checked_partition,
        candidate,
        exact_outer,
        budget,
        atomic_budget,
    )
    if budget.used < RELATION_COST_PARTITION_DOMAIN_OPERATIONS_V2:
        raise RuntimeError("relation partition domain-operation accounting drift")
    return RelationCostPartitionCompilationOutcomeV2(
        kind=RelationCostPartitionCompilationKindV2.PARTITION,
        relation_cost_partition=checked_partition,
        cumulative_resource_usage=_cumulative_usage(
            candidate,
            budget,
            atomic_budget,
        ),
    )


class RelationCostPartitionCompilerV2:
    """Stateless object wrapper for pipeline composition."""

    def compile(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        candidate: CandidateDomainArtifactV2,
    ) -> RelationCostPartitionCompilationOutcomeV2:
        return compile_relation_cost_partition_v2(problem, config, candidate)


def _strict_problem(
    value: object,
) -> SemanticProblemV2 | RelationCostPartitionCompilationOutcomeV2:
    if not isinstance(value, SemanticProblemV2):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:SEMANTIC_PROBLEM:TYPE",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return SemanticProblemV2.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:SEMANTIC_PROBLEM_REVALIDATION",
        )
    except (ValidationError, TypeError, ValueError, Warning):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:SEMANTIC_PROBLEM",
        )


def _strict_config(
    value: object,
) -> CoreSolverConfigV2 | RelationCostPartitionCompilationOutcomeV2:
    if not isinstance(value, CoreSolverConfigV2):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:CORE_SOLVER_CONFIG:TYPE",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return CoreSolverConfigV2.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CORE_SOLVER_CONFIG_REVALIDATION",
        )
    except (ValidationError, TypeError, ValueError, Warning):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:CORE_SOLVER_CONFIG",
        )


def _strict_candidate(
    value: object,
) -> CandidateDomainArtifactV2 | RelationCostPartitionCompilationOutcomeV2:
    if not isinstance(value, CandidateDomainArtifactV2):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:CANDIDATE_DOMAIN:TYPE",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return CandidateDomainArtifactV2.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN_REVALIDATION",
        )
    except (ValidationError, TypeError, ValueError, Warning):
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "INVALID_INPUT:CANDIDATE_DOMAIN",
        )


def _registry_finding(config: CoreSolverConfigV2) -> str | None:
    if (
        config.algorithm_id != RELATION_COST_PARTITION_ALGORITHM_ID_V2
        or config.algorithm_version != RELATION_COST_PARTITION_ALGORITHM_VERSION_V2
    ):
        return (
            f"UNREGISTERED_ALGORITHM:{config.algorithm_id}@{config.algorithm_version}"
        )
    kernel = config.geometry_kernel
    registered = (
        isinstance(kernel, DirectedOutwardGeometryKernelSpecV2)
        and kernel.kernel_id == RECTILINEAR_KERNEL_ID_V2
        and kernel.kernel_version == RECTILINEAR_KERNEL_VERSION_V2
        and kernel.certified_outward_error_m
        == RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M
    )
    if registered:
        return None
    error = getattr(kernel, "certified_outward_error_m", "NONE")
    return (
        f"UNREGISTERED_GEOMETRY_KERNEL:{kernel.kernel_id}"
        f"@{kernel.kernel_version}:{kernel.soundness.value}:{error}"
    )


def _candidate_closure_findings(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if candidate.semantic_problem_sha256 != problem.semantic_problem_sha256:
        findings.append("HASH_MISMATCH:SEMANTIC_PROBLEM")
    if candidate.core_solver_config_sha256 != config.core_solver_config_sha256:
        findings.append("HASH_MISMATCH:CORE_SOLVER_CONFIG")
    if candidate.compilation_coverage is not CandidateCompilationCoverageV2.COMPLETE:
        findings.append(
            "COMPILATION_INCOMPLETE:CANDIDATE_DOMAIN:"
            f"{candidate.compilation_coverage.value}"
        )
    expected = _ordered_constraints(problem)
    if candidate.ordered_constraint_ids != tuple(item[0] for item in expected):
        findings.append("COMPILATION_INCOMPLETE:ORDERED_CONSTRAINT_IDS")
    if (
        tuple(
            (step.constraint_id, step.constraint_kind)
            for step in candidate.shrink_ledger
        )
        != expected
    ):
        findings.append("COMPILATION_INCOMPLETE:SHRINK_LEDGER")
    if any(
        step.disposition is not ConstraintCompilationDispositionV2.APPLIED
        for step in candidate.shrink_ledger
    ):
        findings.append("COMPILATION_INCOMPLETE:NON_APPLIED_STEP")
    hard = candidate.hard_domain
    if (
        hard.completeness
        not in {DomainCompletenessV2.EXACT, DomainCompletenessV2.BRACKETED}
        or hard.coverage is not ArtifactCoverageV2.EXACT_OUTER_COVERAGE
        or hard.inner_bound.status is RegionBoundStatusV2.UNAVAILABLE
    ):
        findings.append("COMPILATION_INCOMPLETE:HARD_DOMAIN_BRACKET")
    if hard.outer_bound.status is not RegionBoundStatusV2.NON_EMPTY:
        findings.append("COMPILATION_INCOMPLETE:EMPTY_OR_UNAVAILABLE_HARD_OUTER")
    subject_id = problem.constraints.allowed_edit.subject_id
    if candidate.candidate_variable.subject_id != subject_id:
        findings.append("CANDIDATE_VARIABLE_MISMATCH:SUBJECT")
    else:
        subject = next(
            (
                item
                for item in problem.scene.objects.values or ()
                if item.object_id == subject_id
            ),
            None,
        )
        if subject is not None:
            translation = subject.pose.world_from_object.translation
            anchor = candidate.candidate_variable.baseline_anchor_world_xy_m
            if (anchor.x, anchor.y) != (translation.x, translation.y):
                findings.append("CANDIDATE_VARIABLE_MISMATCH:BASELINE_ANCHOR")
    return tuple(sorted(set(findings)))


def _ordered_constraints(
    problem: SemanticProblemV2,
) -> tuple[tuple[str, CandidateConstraintKindV2], ...]:
    constraints = problem.constraints
    return (
        (
            constraints.position_domain.constraint_id,
            CandidateConstraintKindV2.POSITION_DOMAIN,
        ),
        *tuple(
            (item.constraint_id, CandidateConstraintKindV2.SUPPORT)
            for item in constraints.support_constraints
        ),
        *tuple(
            (item.constraint_id, CandidateConstraintKindV2.COLLISION)
            for item in constraints.collision_constraints
        ),
        *tuple(
            (item.constraint_id, CandidateConstraintKindV2.VISIBILITY)
            for item in constraints.visibility_constraints
        ),
        (
            constraints.target_relation.constraint_id,
            CandidateConstraintKindV2.TARGET_RELATION,
        ),
    )


def _resource_preflight(
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> str | None:
    usage = candidate.resource_usage
    if (
        usage.domain_operations + RELATION_COST_PARTITION_DOMAIN_OPERATIONS_V2
        > config.max_domain_operations
    ):
        return "RESOURCE_LIMIT:max_domain_operations"
    if usage.partition_cells + 1 > config.max_partition_cells:
        return "RESOURCE_LIMIT:max_partition_cells"
    if usage.refinement_steps > config.max_refinement_steps:
        return "RESOURCE_LIMIT:max_refinement_steps"
    return None


def _lift_bound(
    bound: PlanarRegionBoundV2,
    budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    budget.consume()
    if bound.status is RegionBoundStatusV2.EMPTY:
        return _require_exact_region(
            normalize_rectilinear_region_v2(
                (),
                atomic_budget=atomic_budget,
            )
        )
    if bound.status is not RegionBoundStatusV2.NON_EMPTY or bound.region is None:
        raise _UnsupportedDomainV2("UNSUPPORTED_HARD_DOMAIN:UNAVAILABLE_BOUND")
    return _require_exact_region(
        lift_planar_region_v2(
            bound.region,
            atomic_budget=atomic_budget,
        )
    )


def _verify_inner_subset(
    inner: ExactRectilinearRegionV2,
    outer: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> None:
    budget.consume()
    intersection = _require_exact_region(
        intersect_rectilinear_regions_v2(
            inner,
            outer,
            atomic_budget=atomic_budget,
        )
    )
    if intersection != inner:
        raise _UnsupportedDomainV2("INVALID_HARD_DOMAIN:INNER_NOT_SUBSET_OUTER")


def _verify_partition_outer_union(
    partition: RelationCostPartitionV2,
    candidate: CandidateDomainArtifactV2,
    expected_outer: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> None:
    if len(partition.cells) != 1 or partition.cells[0].domain != candidate.hard_domain:
        raise _UnsupportedDomainV2("INVALID_RELATION_PARTITION:CELL_DOMAIN")
    independently_lifted: list[ExactRectilinearRegionV2] = []
    for cell in partition.cells:
        independently_lifted.append(
            _lift_bound(cell.domain.outer_bound, budget, atomic_budget)
        )
    budget.consume()
    union = _require_exact_region(
        normalize_rectilinear_region_v2(
            tuple(
                rectangle
                for region in independently_lifted
                for rectangle in region.rectangles
            ),
            atomic_budget=atomic_budget,
        )
    )
    if union != expected_outer:
        raise _UnsupportedDomainV2("INVALID_RELATION_PARTITION:OUTER_UNION")


def _require_exact_region(
    outcome: RectilinearRegionOutcomeV2,
) -> ExactRectilinearRegionV2:
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _ResourceLimitV2("RESOURCE_LIMIT:max_partition_cells")
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise _UnsupportedDomainV2(*outcome.finding_codes)
    return outcome.region


def _relation_damage_vector(
    problem: SemanticProblemV2,
    candidate: CandidateDomainArtifactV2,
    outer_domain: ExactRectilinearRegionV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> tuple[RelationDamageBoundV2, ...]:
    keys = tuple(
        sorted(
            (item.key for item in problem.objective.relation_damage.pair_axis_weights),
            key=lambda item: item.sort_key,
        )
    )
    context = _exact_context(
        problem,
        candidate,
        outer_domain,
        domain_budget,
        atomic_budget,
    )
    bounds: list[RelationDamageBoundV2] = []
    for key in keys:
        indicator = _exact_damage_indicator(context, key) if context else None
        lower, upper = (indicator, indicator) if indicator is not None else (0.0, 1.0)
        bounds.append(
            RelationDamageBoundV2(
                key=key,
                lower_bound=lower,
                upper_bound=upper,
            )
        )
    return tuple(bounds)


def _exact_context(
    problem: SemanticProblemV2,
    candidate: CandidateDomainArtifactV2,
    outer_domain: ExactRectilinearRegionV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> _ExactContextV2 | None:
    if not _numeric_policy_is_zero(problem.numeric_policy):
        return None
    for facts in (
        problem.scene.objects,
        problem.scene.geometry_instances,
        problem.scene.cameras,
    ):
        if not _facts_are_exact_zero(facts):
            return None
    camera_id = problem.objective.relation_damage.evaluation_camera_id
    camera = next(
        (
            item
            for item in problem.scene.cameras.values or ()
            if item.camera_id == camera_id
        ),
        None,
    )
    if camera is None or not _camera_is_exact_identity(camera):
        return None
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries: dict[str, GeometryInstanceV2] = {}
    for object_id in objects:
        selected = tuple(
            item
            for item in problem.scene.geometry_instances.values or ()
            if item.owner_object_id == object_id
            and item.role is GeometryRoleV2.RELATION
        )
        if len(selected) == 1:
            geometries[object_id] = selected[0]
    return _ExactContextV2(
        problem=problem,
        candidate=candidate,
        outer_domain=outer_domain,
        subject_id=problem.constraints.allowed_edit.subject_id,
        objects=objects,
        relation_geometries=geometries,
        camera=camera,
        visibility_proofs=_VisibilityProofStateV2(
            domain_budget=domain_budget,
            atomic_budget=atomic_budget,
            after_cache={},
        ),
    )


def _exact_damage_indicator(
    context: _ExactContextV2,
    key: PairAxisKeyV2,
) -> float | None:
    first = context.objects.get(key.first_object_id)
    second = context.objects.get(key.second_object_id)
    first_geometry = context.relation_geometries.get(key.first_object_id)
    second_geometry = context.relation_geometries.get(key.second_object_id)
    if (
        first is None
        or second is None
        or first_geometry is None
        or second_geometry is None
        or not _object_geometry_is_exact_identity(first, first_geometry)
        or not _object_geometry_is_exact_identity(second, second_geometry)
    ):
        return None
    definitions = _axis_definitions(context.problem, key.axis)
    if definitions is None or not _visibility_gate_holds(
        context,
        key,
        definitions,
    ):
        return None

    try:
        if key.axis is RelationAxisV2.HORIZONTAL:
            baseline, candidate_interval = _horizontal_intervals(
                context,
                key,
                first,
                first_geometry,
                second,
                second_geometry,
            )
            baseline_labels = _linear_label_set(definitions, baseline, baseline)
            candidate_labels = _linear_label_set(
                definitions,
                candidate_interval[0],
                candidate_interval[1],
            )
        elif key.axis is RelationAxisV2.DEPTH:
            baseline = (
                _centroid(first, first_geometry)[2]
                - _centroid(second, second_geometry)[2]
            )
            _require_fraction_size(baseline)
            baseline_labels = _linear_label_set(definitions, baseline, baseline)
            candidate_labels = baseline_labels
        else:
            baseline_squared, candidate_squared = _distance_squared_intervals(
                context,
                first,
                first_geometry,
                second,
                second_geometry,
            )
            baseline_labels = _distance_label_set(
                definitions,
                baseline_squared,
                baseline_squared,
            )
            candidate_labels = _distance_label_set(
                definitions,
                candidate_squared[0],
                candidate_squared[1],
            )
        if baseline_labels is None or candidate_labels is None:
            return None
        return 0.0 if baseline_labels == candidate_labels else 1.0
    except (_InexactRelationMeasurementV2, ArithmeticError):
        return None


def _axis_definitions(
    problem: SemanticProblemV2,
    axis: RelationAxisV2,
) -> tuple[RelationDefinitionV2, ...] | None:
    definitions = tuple(
        item
        for item in problem.relation_semantics.definitions
        if item.relation.axis is axis
    )
    expected_measurement = {
        RelationAxisV2.HORIZONTAL: RelationMeasurementV2.PROJECTED_CENTER_DELTA_X,
        RelationAxisV2.DEPTH: RelationMeasurementV2.CAMERA_DEPTH_DELTA,
        RelationAxisV2.DISTANCE: RelationMeasurementV2.SHAPE_GAP_XY,
    }[axis]
    if len(definitions) != 2 or any(
        item.measurement is not expected_measurement
        or item.operand_order is not MeasurementOperandOrderV2.FIRST_MINUS_SECOND
        or item.boundary_policy is not BoundaryPolicyV2.CLOSED
        or item.tolerance != 0.0
        for item in definitions
    ):
        return None
    if axis is not RelationAxisV2.DISTANCE and any(
        item.representative_point
        is not RelationRepresentativePointV2.RELATION_GEOMETRY_VOLUME_CENTROID
        for item in definitions
    ):
        return None
    if axis is RelationAxisV2.DISTANCE and any(
        item.representative_point is not None for item in definitions
    ):
        return None
    return definitions


def _visibility_gate_holds(
    context: _ExactContextV2,
    key: PairAxisKeyV2,
    definitions: tuple[RelationDefinitionV2, ...],
) -> bool:
    if not any(item.requires_both_visible for item in definitions):
        return True
    return all(
        _endpoint_visibility_is_proven(context, object_id)
        for object_id in (key.first_object_id, key.second_object_id)
    )


def _endpoint_visibility_is_proven(
    context: _ExactContextV2,
    object_id: str,
) -> bool:
    for constraint in _qualifying_visibility_constraints(context, object_id):
        if not _baseline_visibility_worst_case_passes(
            context.problem,
            constraint,
            object_id,
        ):
            continue
        if _after_visibility_is_proven(context, constraint):
            return True
    return False


def _qualifying_visibility_constraints(
    context: _ExactContextV2,
    object_id: str,
) -> tuple[VisibilityConstraintV2, ...]:
    return tuple(
        sorted(
            (
                item
                for item in context.problem.constraints.visibility_constraints
                if item.camera_id == context.camera.camera_id
                and object_id in item.query_object_ids
                and item.minimum_visible_fraction > 0.0
                and item.minimum_image_area_fraction > 0.0
            ),
            key=lambda item: item.constraint_id,
        )
    )


def _baseline_visibility_worst_case_passes(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraintV2,
    object_id: str,
) -> bool:
    facts = problem.scene.baseline_observations
    if (
        not _facts_are_exact_zero(facts)
        or FactCompletenessV2.EXACT not in constraint.accepted_baseline_completeness
        or constraint.threshold_boundary_policy is not BoundaryPolicyV2.CLOSED
    ):
        return False
    observations = {
        (
            item.object_id,
            item.camera_id,
            item.metric_definition_id,
            item.metric_definition_version,
        ): item
        for item in facts.values or ()
    }
    specifications = (
        (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            constraint.minimum_visible_fraction,
            True,
        ),
        (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            constraint.minimum_image_area_fraction,
            True,
        ),
        (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            constraint.maximum_truncated_fraction,
            False,
        ),
    )
    for definition_id, version, threshold, is_minimum in specifications:
        observation = observations.get(
            (object_id, constraint.camera_id, definition_id, version)
        )
        if observation is None:
            return False
        lower = Fraction.from_float(observation.normalized_lower_bound)
        upper = Fraction.from_float(observation.normalized_upper_bound)
        exact_threshold = Fraction.from_float(threshold)
        if is_minimum:
            if lower < exact_threshold:
                return False
        elif upper > exact_threshold:
            return False
    return True


def _after_visibility_is_proven(
    context: _ExactContextV2,
    constraint: VisibilityConstraintV2,
) -> bool:
    state = context.visibility_proofs
    cache_key = (constraint.constraint_id, context.outer_domain)
    if cache_key in state.after_cache:
        return state.after_cache[cache_key]

    state.domain_budget.consume()
    outcome = compile_visibility_domain_v2(
        context.problem,
        constraint,
        context.outer_domain,
        atomic_budget=state.atomic_budget,
    )
    if outcome.kind is VisibilityDomainKindV2.RESOURCE_LIMIT:
        finding = (
            outcome.finding_codes[0]
            if outcome.finding_codes
            else "RESOURCE_LIMIT:max_partition_cells"
        )
        raise _ResourceLimitV2(finding)
    if outcome.kind is VisibilityDomainKindV2.IDENTITY:
        proven = True
    elif outcome.kind is VisibilityDomainKindV2.BRACKET:
        if outcome.inner_allowed_delta is None:
            raise ValueError("visibility BRACKET omitted its inner domain")
        state.domain_budget.consume()
        intersection = _require_exact_region(
            intersect_rectilinear_regions_v2(
                context.outer_domain,
                outcome.inner_allowed_delta,
                atomic_budget=state.atomic_budget,
            )
        )
        proven = intersection == context.outer_domain
    else:
        proven = False
    state.after_cache[cache_key] = proven
    return proven


def _horizontal_intervals(
    context: _ExactContextV2,
    key: PairAxisKeyV2,
    first: CanonicalObjectV2,
    first_geometry: GeometryInstanceV2,
    second: CanonicalObjectV2,
    second_geometry: GeometryInstanceV2,
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    first_center = _centroid(first, first_geometry)
    second_center = _centroid(second, second_geometry)
    if first_center[2] <= 0 or second_center[2] <= 0:
        raise _InexactRelationMeasurementV2("projected centroid depth must be positive")
    intrinsics = context.camera.intrinsics_row_major
    fx = Fraction.from_float(intrinsics[0])
    cx = Fraction.from_float(intrinsics[2])
    baseline = (
        fx * first_center[0] / first_center[2]
        + cx
        - fx * second_center[0] / second_center[2]
        - cx
    )
    if key.first_object_id == context.subject_id:
        coefficient = fx / first_center[2]
    elif key.second_object_id == context.subject_id:
        coefficient = -fx / second_center[2]
    else:
        coefficient = Fraction()
    values = tuple(
        baseline + coefficient * x
        for rectangle in context.outer_domain.rectangles
        for x in (rectangle.min_x_m, rectangle.max_x_m)
    )
    if not values:
        raise _InexactRelationMeasurementV2("candidate outer domain is empty")
    lower, upper = min(values), max(values)
    for value in (baseline, coefficient, lower, upper):
        _require_fraction_size(value)
    return baseline, (lower, upper)


def _distance_squared_intervals(
    context: _ExactContextV2,
    first: CanonicalObjectV2,
    first_geometry: GeometryInstanceV2,
    second: CanonicalObjectV2,
    second_geometry: GeometryInstanceV2,
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    if first.object_id == context.subject_id:
        moving_object, moving_geometry = first, first_geometry
        fixed_object, fixed_geometry = second, second_geometry
    elif second.object_id == context.subject_id:
        moving_object, moving_geometry = second, second_geometry
        fixed_object, fixed_geometry = first, first_geometry
    else:
        first_bounds = _world_xy_bounds(first, first_geometry)
        second_bounds = _world_xy_bounds(second, second_geometry)
        baseline_squared = _axis_aligned_gap_squared(first_bounds, second_bounds)
        _require_fraction_size(baseline_squared)
        return baseline_squared, (baseline_squared, baseline_squared)
    moving_bounds = _world_xy_bounds(moving_object, moving_geometry)
    fixed_bounds = _world_xy_bounds(fixed_object, fixed_geometry)
    qx = (
        fixed_bounds[0] - moving_bounds[2],
        fixed_bounds[2] - moving_bounds[0],
    )
    qy = (
        fixed_bounds[1] - moving_bounds[3],
        fixed_bounds[3] - moving_bounds[1],
    )
    baseline_squared = _point_interval_distance(Fraction(), *qx) ** 2 + (
        _point_interval_distance(Fraction(), *qy) ** 2
    )
    minima: list[Fraction] = []
    maxima: list[Fraction] = []
    for rectangle in context.outer_domain.rectangles:
        min_x = _interval_interval_distance(
            rectangle.min_x_m,
            rectangle.max_x_m,
            *qx,
        )
        min_y = _interval_interval_distance(
            rectangle.min_y_m,
            rectangle.max_y_m,
            *qy,
        )
        max_x = max(
            _point_interval_distance(rectangle.min_x_m, *qx),
            _point_interval_distance(rectangle.max_x_m, *qx),
        )
        max_y = max(
            _point_interval_distance(rectangle.min_y_m, *qy),
            _point_interval_distance(rectangle.max_y_m, *qy),
        )
        minima.append(min_x**2 + min_y**2)
        maxima.append(max_x**2 + max_y**2)
    if not minima:
        raise _InexactRelationMeasurementV2("candidate outer domain is empty")
    candidate = (min(minima), max(maxima))
    for value in (baseline_squared, *candidate):
        _require_fraction_size(value)
    return baseline_squared, candidate


def _axis_aligned_gap_squared(
    first: tuple[Fraction, Fraction, Fraction, Fraction],
    second: tuple[Fraction, Fraction, Fraction, Fraction],
) -> Fraction:
    gap_x = _interval_interval_distance(first[0], first[2], second[0], second[2])
    gap_y = _interval_interval_distance(first[1], first[3], second[1], second[3])
    return gap_x**2 + gap_y**2


def _linear_label_set(
    definitions: tuple[RelationDefinitionV2, ...],
    lower: Fraction,
    upper: Fraction,
) -> frozenset[RelationV2] | None:
    labels: set[RelationV2] = set()
    for definition in definitions:
        threshold = Fraction.from_float(definition.threshold)
        _require_fraction_size(threshold)
        if definition.comparator is MeasurementComparatorV2.LESS_THAN:
            always_true = upper <= threshold
            always_false = lower > threshold
        else:
            always_true = lower >= threshold
            always_false = upper < threshold
        if not always_true and not always_false:
            return None
        if always_true:
            labels.add(definition.relation)
    return frozenset(labels)


def _distance_label_set(
    definitions: tuple[RelationDefinitionV2, ...],
    lower_squared: Fraction,
    upper_squared: Fraction,
) -> frozenset[RelationV2] | None:
    labels: set[RelationV2] = set()
    for definition in definitions:
        threshold = Fraction.from_float(definition.threshold)
        _require_fraction_size(threshold)
        if definition.comparator is MeasurementComparatorV2.LESS_THAN:
            if threshold < 0:
                always_true, always_false = False, True
            else:
                squared = threshold**2
                always_true = upper_squared <= squared
                always_false = lower_squared > squared
        else:
            if threshold <= 0:
                always_true, always_false = True, False
            else:
                squared = threshold**2
                always_true = lower_squared >= squared
                always_false = upper_squared < squared
        if not always_true and not always_false:
            return None
        if always_true:
            labels.add(definition.relation)
    return frozenset(labels)


def _centroid(
    object_: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> tuple[Fraction, Fraction, Fraction]:
    object_translation = object_.pose.world_from_object.translation
    geometry_translation = geometry.anchor_from_geometry.translation
    values = (
        Fraction.from_float(object_translation.x)
        + Fraction.from_float(geometry_translation.x),
        Fraction.from_float(object_translation.y)
        + Fraction.from_float(geometry_translation.y),
        Fraction.from_float(object_translation.z)
        + Fraction.from_float(geometry_translation.z),
    )
    for value in values:
        _require_fraction_size(value)
    return values


def _world_xy_bounds(
    object_: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if not isinstance(geometry.shape, UprightBox3DV2):
        raise TypeError("relation geometry must be an upright box")
    center = _centroid(object_, geometry)
    half_x = Fraction.from_float(geometry.shape.size_m.x) / 2
    half_y = Fraction.from_float(geometry.shape.size_m.y) / 2
    values = (
        center[0] - half_x,
        center[1] - half_y,
        center[0] + half_x,
        center[1] + half_y,
    )
    for value in values:
        _require_fraction_size(value)
    return values


def _point_interval_distance(
    point: Fraction, lower: Fraction, upper: Fraction
) -> Fraction:
    if point < lower:
        return lower - point
    if point > upper:
        return point - upper
    return Fraction()


def _interval_interval_distance(
    first_lower: Fraction,
    first_upper: Fraction,
    second_lower: Fraction,
    second_upper: Fraction,
) -> Fraction:
    if first_upper < second_lower:
        return second_lower - first_upper
    if first_lower > second_upper:
        return first_lower - second_upper
    return Fraction()


def _object_geometry_is_exact_identity(
    object_: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> bool:
    return (
        geometry.owner_object_id == object_.object_id
        and geometry.role is GeometryRoleV2.RELATION
        and geometry.approximation is GeometryApproximationV2.EXACT
        and _uncertainty_is_zero(geometry.uncertainty)
        and isinstance(geometry.shape, UprightBox3DV2)
        and _is_identity_rotation(object_.pose.world_from_object.rotation)
        and _is_identity_rotation(geometry.anchor_from_geometry.rotation)
    )


def _camera_is_exact_identity(camera: PinholeCameraV2) -> bool:
    intrinsics = camera.intrinsics_row_major
    return (
        _is_identity_transform(camera.world_to_camera)
        and camera.distortion_model is CameraDistortionModelV2.NONE
        and camera.brown_conrady_coefficients is None
        and _uncertainty_is_zero(camera.calibration_uncertainty)
        and intrinsics[1] == 0.0
        and intrinsics[3] == 0.0
        and intrinsics[6:] == (0.0, 0.0, 1.0)
        and camera.matrix_layout is CameraMatrixLayoutV2.ROW_MAJOR
        and camera.camera_axes is CameraAxesV2.X_RIGHT_Y_DOWN_Z_FORWARD
        and camera.pixel_convention is CameraPixelConventionV2.CENTER_AT_HALF
        and camera.depth_convention is CameraDepthConventionV2.POSITIVE_Z_FORWARD
    )


def _facts_are_exact_zero(facts: FactSetV2) -> bool:
    return (
        facts.availability is FactAvailabilityV2.KNOWN
        and facts.completeness is FactCompletenessV2.EXACT
        and facts.values is not None
        and facts.uncertainty is not None
        and _uncertainty_is_zero(facts.uncertainty)
    )


def _is_identity_rotation(rotation: QuaternionV2) -> bool:
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _is_identity_transform(transform: RigidTransformV2) -> bool:
    translation = transform.translation
    return (translation.x, translation.y, translation.z) == (
        0.0,
        0.0,
        0.0,
    ) and _is_identity_rotation(transform.rotation)


def _numeric_policy_is_zero(policy: NumericPolicyV2) -> bool:
    return all(
        value == 0.0
        for value in (
            policy.linear_tolerance_m,
            policy.area_tolerance_m2,
            policy.angular_tolerance_rad,
            policy.pixel_tolerance_px,
            policy.fraction_tolerance,
        )
    )


def _uncertainty_is_zero(uncertainty: UncertaintyBudgetV2) -> bool:
    return _numeric_policy_is_zero(
        uncertainty.source_error
    ) and _numeric_policy_is_zero(uncertainty.shape_approximation)


def _require_fraction_size(value: Fraction) -> None:
    if (
        value.numerator.bit_length() > _MAX_CERTIFIED_FRACTION_BITS
        or value.denominator.bit_length() > _MAX_CERTIFIED_FRACTION_BITS
    ):
        raise OverflowError("certified Fraction exceeds the registered bit limit")


def _cumulative_usage(
    candidate: CandidateDomainArtifactV2,
    domain_budget: _DomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=(
            candidate.resource_usage.domain_operations + domain_budget.used
        ),
        partition_cells=atomic_budget.used,
        refinement_steps=candidate.resource_usage.refinement_steps,
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *findings: str,
    cumulative_resource_usage: CompilationResourceUsageV2 | None = None,
) -> RelationCostPartitionCompilationOutcomeV2:
    return RelationCostPartitionCompilationOutcomeV2(
        kind=RelationCostPartitionCompilationKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=tuple(findings),
        cumulative_resource_usage=cumulative_resource_usage,
    )
