"""Shared current candidate cells; exact migrated body."""

from __future__ import annotations

"""Deterministic Canonical v2 candidate-domain compilation.

The certified subset is intentionally narrow: exact, closed, axis-aligned
facts are compiled with the registered directed rational rectilinear kernel.
Every constraint outside that subset is recorded as unknown and therefore
preserves the complete outer bound while clearing the inner bound.
"""


import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

from pydantic import ValidationError

from spatialcf.core._internal.compilation.collision import (
    CollisionDomainKindV2,
    compile_collision_domain_v2,
)
from spatialcf.core._internal.compilation.support import (
    SupportDomainKindV2,
    compile_support_domain_v2,
)
from spatialcf.core._internal.compilation.target import (
    TargetRelationDomainKindV2,
    compile_target_relation_domain_v2,
)
from spatialcf.core._internal.compilation.visibility import (
    VisibilityDomainKindV2,
    compile_visibility_domain_v2,
)
from spatialcf.core._internal.kernels.rect import (
    AxisMarginXYV2,
    DirectedRectRoundingV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectKernelProjectionErrorV2,
    RectTopologyV2,
    UnsupportedRectRegionErrorV2,
    WorldPointXYV2,
)
from spatialcf.core._internal.kernels.rectilinear import (
    RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M,
    RECTILINEAR_KERNEL_ID_V2,
    RECTILINEAR_KERNEL_VERSION_V2,
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    intersect_rectilinear_regions_v2,
    lift_planar_region_v2,
    normalize_rectilinear_region_v2,
    project_rectilinear_region_v2,
)
from spatialcf.core._internal.resources import (
    DomainOperationBudgetV2,
)
from spatialcf.domain.artifacts import (
    ArtifactCoverageV2,
    CandidateCompilationCoverageV2,
    CandidateConstraintKindV2,
    CandidateDomainArtifactV2,
    CandidateDomainVariableV2,
    CompilationResourceUsageV2,
    ConstraintCompilationDispositionV2,
    ConstraintDomainShrinkStepV2,
    DomainCompletenessV2,
    PlanarDomainBoundsV2,
    PlanarRegionBoundV2,
    RegionBoundStatusV2,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec2,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    PositionRegionInterpretation,
    RegionAggregation,
)
from spatialcf.domain.geometry import (
    ExtrudedPlanarPolygonV2,
    GeometryApproximationV2,
    PlanarRegionV2,
    UprightBox3DV2,
)
from spatialcf.domain.problem import SemanticProblemV2
from spatialcf.domain.result import (
    CoreSolverConfigV2,
    DirectedOutwardGeometryKernelSpecV2,
    UncertifiedReasonV2,
)
from spatialcf.domain.scene import (
    CanonicalObject,
    RegionBoundaryPolicy,
)

from spatialcf.core._internal.compilation.universe import (
    _ResourceLimitErrorV2,
    _DomainOperationBudgetV2,
    _SearchUniverseV2,
    _SearchUniverseFailureV2,
    _compile_search_universe,
    _search_failure,
    _planar_outer_margin,
    _search_anchor_margin,
    _occupancy_outer_radius,
    _shape_local_l1_radius,
    _baseline_anchor,
    _object_anchor_world_xy,
    _exact_nonempty_domain,
    _require_bound,
)

CANDIDATE_DOMAIN_ALGORITHM_ID_V2 = "solver:canonical-branch-and-bound-v2"
CANDIDATE_DOMAIN_ALGORITHM_VERSION_V2 = "algorithm:2.0"


@dataclass(frozen=True, slots=True)
class CandidateDomainCompilationOutcomeV2:
    """Closed outcome of candidate-domain compilation, not a solve result."""

    candidate_domain: CandidateDomainArtifactV2 | None
    uncertified_reason: UncertifiedReasonV2 | None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "finding_codes", tuple(sorted(set(self.finding_codes)))
        )
        if self.candidate_domain is None and self.uncertified_reason is None:
            raise ValueError("missing candidate domain requires an uncertified reason")


CandidateDomainCompileOutcomeV2 = CandidateDomainCompilationOutcomeV2


class _PartitionResourceLimitErrorV2(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PositionCompilationV2:
    exact_domain: ExactRectilinearRegionV2 | None
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExactDomainBracketV2:
    inner: ExactRectilinearRegionV2
    outer: ExactRectilinearRegionV2


@dataclass(frozen=True, slots=True)
class _ExactConstraintCompilationV2:
    exact_domain: _ExactDomainBracketV2 | None
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IntegratedConstraintV2:
    constraint_domain: PlanarDomainBoundsV2 | None
    output_domain: PlanarDomainBoundsV2
    exact_output: _ExactDomainBracketV2
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PublishedDomainAttemptV2:
    domain: PlanarDomainBoundsV2 | None
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        if (self.domain is None) == (self.failure_kind is None):
            raise ValueError(
                "published domain attempt requires exactly one result or failure"
            )


class CandidateDomainCompilerV2:
    """Compile the sound rectilinear subset into an auditable shrink ledger."""

    def compile(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
    ) -> CandidateDomainCompilationOutcomeV2:
        try:
            problem = _strict_problem(problem)
        except (ArithmeticError, RuntimeWarning):
            return _numeric_gap_outcome("SEMANTIC_PROBLEM_REVALIDATION")
        try:
            config = _strict_config(config)
        except (ArithmeticError, RuntimeWarning):
            return _numeric_gap_outcome("CORE_SOLVER_CONFIG_REVALIDATION")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                registry_finding = _registry_finding(config)
                if registry_finding is not None:
                    return CandidateDomainCompilationOutcomeV2(
                        candidate_domain=None,
                        uncertified_reason=UncertifiedReasonV2.UNSUPPORTED_MODEL,
                        finding_codes=(registry_finding,),
                    )

                budget = _DomainOperationBudgetV2(config.max_domain_operations)
                try:
                    search_result = _compile_search_universe(problem, budget)
                    if isinstance(search_result, _SearchUniverseFailureV2):
                        return CandidateDomainCompilationOutcomeV2(
                            candidate_domain=None,
                            uncertified_reason=search_result.uncertified_reason,
                            finding_codes=search_result.finding_codes,
                        )
                    if config.max_partition_cells < 1:
                        raise _PartitionResourceLimitErrorV2
                    partition_budget = RectilinearAtomicBudgetV2(
                        limit=config.max_partition_cells
                    )
                    return _compile_artifact(
                        problem,
                        config,
                        search_result,
                        budget,
                        partition_budget,
                    )
                except _ResourceLimitErrorV2:
                    return CandidateDomainCompilationOutcomeV2(
                        candidate_domain=None,
                        uncertified_reason=(
                            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED
                        ),
                        finding_codes=("RESOURCE_LIMIT:max_domain_operations",),
                    )
                except (
                    _PartitionResourceLimitErrorV2,
                    RectilinearAtomicBudgetExhaustedV2,
                ):
                    return CandidateDomainCompilationOutcomeV2(
                        candidate_domain=None,
                        uncertified_reason=(
                            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED
                        ),
                        finding_codes=("RESOURCE_LIMIT:max_partition_cells",),
                    )
        except (ArithmeticError, RuntimeWarning):
            return _numeric_gap_outcome("CANDIDATE_DOMAIN_COMPILATION")
        except Warning as warning:
            raise _validation_error_from_warning(
                "CandidateDomainCompilerV2",
                problem,
                warning,
            ) from warning


def _strict_problem(problem: SemanticProblemV2) -> SemanticProblemV2:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return SemanticProblemV2.model_validate(
                problem.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        raise
    except Warning as warning:
        raise _validation_error_from_warning(
            "SemanticProblemV2",
            problem,
            warning,
        ) from warning


def _strict_config(config: CoreSolverConfigV2) -> CoreSolverConfigV2:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return CoreSolverConfigV2.model_validate(
                config.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        raise
    except Warning as warning:
        raise _validation_error_from_warning(
            "CoreSolverConfigV2",
            config,
            warning,
        ) from warning


def _validation_error_from_warning(
    title: str,
    value: object,
    warning: Warning,
) -> ValidationError:
    return ValidationError.from_exception_data(
        title,
        [
            {
                "type": "value_error",
                "loc": ("input",),
                "input": value,
                "ctx": {"error": ValueError(str(warning))},
            },
        ],
    )


def _numeric_gap_outcome(stage: str) -> CandidateDomainCompilationOutcomeV2:
    return CandidateDomainCompilationOutcomeV2(
        candidate_domain=None,
        uncertified_reason=UncertifiedReasonV2.NUMERIC_GAP,
        finding_codes=(f"NUMERIC_GAP:{stage}",),
    )


def _registry_finding(config: CoreSolverConfigV2) -> str | None:
    if (
        config.algorithm_id != CANDIDATE_DOMAIN_ALGORITHM_ID_V2
        or config.algorithm_version != CANDIDATE_DOMAIN_ALGORITHM_VERSION_V2
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


def _compile_artifact(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    universe: _SearchUniverseV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> CandidateDomainCompilationOutcomeV2:
    ordered = _ordered_constraints(problem)
    empty_exact = _normalize_exact_rectangles((), budget, partition_budget)
    exact_universe = _lift_published_domain(
        universe.domain,
        empty_exact,
        budget,
        partition_budget,
    )
    if exact_universe is None:
        return CandidateDomainCompilationOutcomeV2(
            candidate_domain=None,
            uncertified_reason=UncertifiedReasonV2.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:SEARCH_UNIVERSE_RECTILINEAR_LIFT",),
        )

    steps: list[ConstraintDomainShrinkStepV2] = []
    findings: list[str] = []
    current = universe.domain
    current_exact = exact_universe
    had_unknown = False

    for index, (constraint_id, kind) in enumerate(ordered):
        input_domain = current
        if kind is CandidateConstraintKindV2.POSITION_DOMAIN:
            position = _compile_position(
                problem,
                universe,
                exact_universe,
                budget,
                partition_budget,
            )
            compilation = _ExactConstraintCompilationV2(
                exact_domain=(
                    _ExactDomainBracketV2(
                        inner=position.exact_domain,
                        outer=position.exact_domain,
                    )
                    if position.exact_domain is not None
                    else None
                ),
                finding_codes=position.finding_codes,
            )
        elif kind is CandidateConstraintKindV2.SUPPORT:
            compilation = _compile_support(
                problem,
                constraint_id,
                exact_universe,
                empty_exact,
                budget,
                partition_budget,
            )
        elif kind is CandidateConstraintKindV2.COLLISION:
            compilation = _compile_collision(
                problem,
                constraint_id,
                exact_universe.outer,
                empty_exact,
                budget,
                partition_budget,
            )
        elif kind is CandidateConstraintKindV2.TARGET_RELATION:
            compilation = _compile_target_relation(
                problem,
                exact_universe.outer,
                empty_exact,
                budget,
                partition_budget,
            )
        elif kind is CandidateConstraintKindV2.VISIBILITY:
            compilation = _compile_visibility(
                problem,
                constraint_id,
                exact_universe.outer,
                empty_exact,
                budget,
                partition_budget,
            )

        integrated = _integrate_exact_constraint(
            constraint_id=constraint_id,
            kind=kind,
            compilation=compilation,
            current_exact=current_exact,
            current_published=current,
            empty_exact=empty_exact,
            budget=budget,
            partition_budget=partition_budget,
        )
        steps.append(
            _step(
                index=index,
                constraint_id=constraint_id,
                kind=kind,
                input_domain=input_domain,
                output_domain=integrated.output_domain,
                constraint_domain=integrated.constraint_domain,
            )
        )
        current = integrated.output_domain
        current_exact = integrated.exact_output
        findings.extend(integrated.finding_codes)
        if integrated.constraint_domain is None:
            had_unknown = True
        if current.outer_bound.status is RegionBoundStatusV2.EMPTY and not had_unknown:
            artifact = _artifact(
                problem,
                config,
                universe.domain,
                ordered,
                tuple(steps),
                CandidateCompilationCoverageV2.EMPTY_OUTER_PREFIX,
                current,
                budget,
                partition_budget,
            )
            return CandidateDomainCompilationOutcomeV2(
                candidate_domain=artifact,
                uncertified_reason=None,
                finding_codes=tuple(findings),
            )

    semantic_findings = tuple(problem.certification_blockers)
    findings.extend(semantic_findings)
    if not had_unknown and not semantic_findings:
        coverage = CandidateCompilationCoverageV2.COMPLETE
        reason = None
    elif any(code.startswith("MISSING_FACT:") for code in findings):
        coverage = CandidateCompilationCoverageV2.PARTIAL
        reason = UncertifiedReasonV2.MISSING_FACT
    else:
        coverage = CandidateCompilationCoverageV2.PARTIAL
        reason = UncertifiedReasonV2.COMPILATION_INCOMPLETE
    artifact = _artifact(
        problem,
        config,
        universe.domain,
        ordered,
        tuple(steps),
        coverage,
        current,
        budget,
        partition_budget,
    )
    return CandidateDomainCompilationOutcomeV2(
        candidate_domain=artifact,
        uncertified_reason=reason,
        finding_codes=tuple(findings),
    )


def _compile_support(
    problem: SemanticProblemV2,
    constraint_id: str,
    search_universe: _ExactDomainBracketV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _ExactConstraintCompilationV2:
    budget.consume()
    outcome = compile_support_domain_v2(problem, constraint_id)
    if outcome.kind is SupportDomainKindV2.UNKNOWN:
        return _ExactConstraintCompilationV2(
            exact_domain=None,
            finding_codes=outcome.finding_codes,
        )
    if outcome.kind is SupportDomainKindV2.IDENTITY:
        return _ExactConstraintCompilationV2(
            exact_domain=search_universe,
            finding_codes=(),
        )
    if outcome.kind is SupportDomainKindV2.EMPTY:
        return _ExactConstraintCompilationV2(
            exact_domain=_ExactDomainBracketV2(
                inner=empty_exact,
                outer=empty_exact,
            ),
            finding_codes=outcome.finding_codes,
        )

    rect = outcome.delta_locus
    assert rect is not None
    exact = _normalize_exact_rectangles((rect,), budget, partition_budget)
    clipped = _intersect_exact_regions(
        exact,
        search_universe.outer,
        budget,
        partition_budget,
    )
    return _ExactConstraintCompilationV2(
        exact_domain=_ExactDomainBracketV2(inner=clipped, outer=clipped),
        finding_codes=(),
    )


def _compile_collision(
    problem: SemanticProblemV2,
    constraint_id: str,
    search_universe: ExactRectilinearRegionV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _ExactConstraintCompilationV2:
    budget.consume()
    outcome = compile_collision_domain_v2(
        problem,
        constraint_id,
        search_universe,
        atomic_budget=partition_budget,
    )
    if outcome.kind is CollisionDomainKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is CollisionDomainKindV2.UNKNOWN:
        return _ExactConstraintCompilationV2(
            exact_domain=None,
            finding_codes=outcome.finding_codes,
        )
    if outcome.kind is CollisionDomainKindV2.IDENTITY:
        exact = _ExactDomainBracketV2(
            inner=search_universe,
            outer=search_universe,
        )
    elif outcome.kind is CollisionDomainKindV2.EMPTY:
        exact = _ExactDomainBracketV2(inner=empty_exact, outer=empty_exact)
    else:
        assert outcome.inner_allowed_delta is not None
        assert outcome.outer_allowed_delta is not None
        exact = _ExactDomainBracketV2(
            inner=outcome.inner_allowed_delta,
            outer=outcome.outer_allowed_delta,
        )
    return _ExactConstraintCompilationV2(
        exact_domain=exact,
        finding_codes=outcome.finding_codes,
    )


def _compile_target_relation(
    problem: SemanticProblemV2,
    search_universe: ExactRectilinearRegionV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _ExactConstraintCompilationV2:
    budget.consume()
    outcome = compile_target_relation_domain_v2(
        problem,
        search_universe,
        atomic_budget=partition_budget,
    )
    if outcome.kind is TargetRelationDomainKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is TargetRelationDomainKindV2.UNKNOWN:
        return _ExactConstraintCompilationV2(
            exact_domain=None,
            finding_codes=outcome.finding_codes,
        )
    if outcome.kind is TargetRelationDomainKindV2.IDENTITY:
        exact = _ExactDomainBracketV2(
            inner=search_universe,
            outer=search_universe,
        )
    elif outcome.kind is TargetRelationDomainKindV2.EMPTY:
        exact = _ExactDomainBracketV2(inner=empty_exact, outer=empty_exact)
    else:
        assert outcome.inner_allowed_delta is not None
        assert outcome.outer_allowed_delta is not None
        exact = _ExactDomainBracketV2(
            inner=outcome.inner_allowed_delta,
            outer=outcome.outer_allowed_delta,
        )
    return _ExactConstraintCompilationV2(
        exact_domain=exact,
        finding_codes=outcome.finding_codes,
    )


def _compile_visibility(
    problem: SemanticProblemV2,
    constraint_id: str,
    search_universe: ExactRectilinearRegionV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _ExactConstraintCompilationV2:
    budget.consume()
    outcome = compile_visibility_domain_v2(
        problem,
        constraint_id,
        search_universe,
        atomic_budget=partition_budget,
    )
    if outcome.kind is VisibilityDomainKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is VisibilityDomainKindV2.UNKNOWN:
        return _ExactConstraintCompilationV2(
            exact_domain=None,
            finding_codes=outcome.finding_codes,
        )
    if outcome.kind is VisibilityDomainKindV2.IDENTITY:
        exact = _ExactDomainBracketV2(
            inner=search_universe,
            outer=search_universe,
        )
    elif outcome.kind is VisibilityDomainKindV2.EMPTY:
        exact = _ExactDomainBracketV2(inner=empty_exact, outer=empty_exact)
    else:
        assert outcome.inner_allowed_delta is not None
        assert outcome.outer_allowed_delta is not None
        exact = _ExactDomainBracketV2(
            inner=outcome.inner_allowed_delta,
            outer=outcome.outer_allowed_delta,
        )
    return _ExactConstraintCompilationV2(
        exact_domain=exact,
        finding_codes=outcome.finding_codes,
    )


def _normalize_exact_rectangles(
    rectangles: tuple[ExactAxisAlignedRectV2, ...],
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    budget.consume()
    outcome = normalize_rectilinear_region_v2(
        rectangles,
        atomic_budget=partition_budget,
    )
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise AssertionError("rectilinear normalization has no unsupported branch")
    return outcome.region


def _intersect_exact_regions(
    left: ExactRectilinearRegionV2,
    right: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2:
    budget.consume()
    outcome = intersect_rectilinear_regions_v2(
        left,
        right,
        atomic_budget=partition_budget,
    )
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise AssertionError(
            "exact rectilinear intersection unexpectedly became unknown"
        )
    return outcome.region


def _lift_published_domain(
    domain: PlanarDomainBoundsV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _ExactDomainBracketV2 | None:
    checked = PlanarDomainBoundsV2.model_validate(
        domain.model_dump(mode="python"),
        strict=True,
    )
    if (
        checked.inner_bound.status is RegionBoundStatusV2.UNAVAILABLE
        or checked.outer_bound.status is RegionBoundStatusV2.UNAVAILABLE
    ):
        return None
    if checked.inner_bound == checked.outer_bound:
        exact = _lift_published_bound(
            checked.inner_bound,
            empty_exact,
            budget,
            partition_budget,
        )
        if exact is None:
            return None
        return _ExactDomainBracketV2(inner=exact, outer=exact)

    inner = _lift_published_bound(
        checked.inner_bound,
        empty_exact,
        budget,
        partition_budget,
    )
    outer = _lift_published_bound(
        checked.outer_bound,
        empty_exact,
        budget,
        partition_budget,
    )
    if inner is None or outer is None:
        return None
    return _ExactDomainBracketV2(inner=inner, outer=outer)


def _lift_published_bound(
    bound: PlanarRegionBoundV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | None:
    if bound.status is RegionBoundStatusV2.EMPTY:
        return empty_exact
    if bound.status is not RegionBoundStatusV2.NON_EMPTY or bound.region is None:
        return None
    budget.consume()
    outcome = lift_planar_region_v2(
        bound.region,
        atomic_budget=partition_budget,
    )
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT:
        return None
    assert outcome.region is not None
    return outcome.region


def _publish_exact_domain(
    exact: _ExactDomainBracketV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _PublishedDomainAttemptV2:
    outer, outer_failure = _project_exact_bound(
        exact.outer,
        DirectedRectRoundingV2.OUTER,
        budget,
        partition_budget,
    )
    if outer is None:
        assert outer_failure is not None
        return _PublishedDomainAttemptV2(
            domain=None,
            failure_kind=outer_failure,
        )

    inner, _ = _project_exact_bound(
        exact.inner,
        DirectedRectRoundingV2.INNER,
        budget,
        partition_budget,
    )
    assert inner is not None
    completeness = (
        DomainCompletenessV2.EXACT if inner == outer else DomainCompletenessV2.BRACKETED
    )
    return _PublishedDomainAttemptV2(
        domain=PlanarDomainBoundsV2(
            inner_bound=inner,
            outer_bound=outer,
            completeness=completeness,
            coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
        )
    )


def _project_exact_bound(
    exact: ExactRectilinearRegionV2,
    rounding: DirectedRectRoundingV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> tuple[PlanarRegionBoundV2 | None, str | None]:
    budget.consume()
    outcome = project_rectilinear_region_v2(
        exact,
        rounding,
        atomic_budget=partition_budget,
    )
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        raise _PartitionResourceLimitErrorV2
    if outcome.kind is RectilinearOutcomeKindV2.UNKNOWN:
        if rounding is DirectedRectRoundingV2.INNER:
            return PlanarRegionBoundV2.empty(), None
        if any("NON_FINITE_BINARY64" in code for code in outcome.finding_codes):
            return None, "NUMERIC_GAP"
        if any(
            token in code
            for code in outcome.finding_codes
            for token in ("DEGENERATE_FEATURES", "OUTER_COLLAPSED")
        ):
            return None, "DEGENERATE"
        return None, "UNSUPPORTED"

    if outcome.planar_region is None:
        return PlanarRegionBoundV2.empty(), None
    return PlanarRegionBoundV2.non_empty(outcome.planar_region), None


def _integrate_exact_constraint(
    *,
    constraint_id: str,
    kind: CandidateConstraintKindV2,
    compilation: _ExactConstraintCompilationV2,
    current_exact: _ExactDomainBracketV2,
    current_published: PlanarDomainBoundsV2,
    empty_exact: ExactRectilinearRegionV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _IntegratedConstraintV2:
    if compilation.exact_domain is None:
        return _unknown_integration(
            current_exact=current_exact,
            current_published=current_published,
            empty_exact=empty_exact,
            finding_codes=compilation.finding_codes,
        )

    published_constraint = _publish_exact_domain(
        compilation.exact_domain,
        budget,
        partition_budget,
    )
    if published_constraint.domain is None:
        assert published_constraint.failure_kind is not None
        return _unknown_integration(
            current_exact=current_exact,
            current_published=current_published,
            empty_exact=empty_exact,
            finding_codes=(
                *compilation.finding_codes,
                _publication_failure_finding(
                    kind,
                    constraint_id,
                    published_constraint.failure_kind,
                    output=False,
                ),
            ),
        )

    # Equal inward/outward publication proves that every boundary is exactly
    # representable in binary64.  In that case the already-validated rational
    # domain is also the exact lift of the published domain, so lifting it a
    # second time would only repeat the same grid classification and charge.
    lifted_constraint = (
        compilation.exact_domain
        if published_constraint.domain.completeness is DomainCompletenessV2.EXACT
        else _lift_published_domain(
            published_constraint.domain,
            empty_exact,
            budget,
            partition_budget,
        )
    )
    if lifted_constraint is None:
        return _unknown_integration(
            current_exact=current_exact,
            current_published=current_published,
            empty_exact=empty_exact,
            finding_codes=(
                *compilation.finding_codes,
                f"COMPILATION_INCOMPLETE:{kind.value}_DOMAIN_LIFT:{constraint_id}",
            ),
        )

    exact_output = _ExactDomainBracketV2(
        inner=_intersect_exact_regions(
            current_exact.inner,
            lifted_constraint.inner,
            budget,
            partition_budget,
        ),
        outer=_intersect_exact_regions(
            current_exact.outer,
            lifted_constraint.outer,
            budget,
            partition_budget,
        ),
    )
    published_output = _publish_exact_domain(
        exact_output,
        budget,
        partition_budget,
    )
    if published_output.domain is None:
        assert published_output.failure_kind is not None
        return _unknown_integration(
            current_exact=current_exact,
            current_published=current_published,
            empty_exact=empty_exact,
            finding_codes=(
                *compilation.finding_codes,
                _publication_failure_finding(
                    kind,
                    constraint_id,
                    published_output.failure_kind,
                    output=True,
                ),
            ),
        )

    lifted_output = (
        exact_output
        if published_output.domain.completeness is DomainCompletenessV2.EXACT
        else _lift_published_domain(
            published_output.domain,
            empty_exact,
            budget,
            partition_budget,
        )
    )
    if lifted_output is None:
        return _unknown_integration(
            current_exact=current_exact,
            current_published=current_published,
            empty_exact=empty_exact,
            finding_codes=(
                *compilation.finding_codes,
                f"COMPILATION_INCOMPLETE:{kind.value}_OUTPUT_LIFT:{constraint_id}",
            ),
        )
    return _IntegratedConstraintV2(
        constraint_domain=published_constraint.domain,
        output_domain=published_output.domain,
        exact_output=lifted_output,
        finding_codes=compilation.finding_codes,
    )


def _unknown_integration(
    *,
    current_exact: _ExactDomainBracketV2,
    current_published: PlanarDomainBoundsV2,
    empty_exact: ExactRectilinearRegionV2,
    finding_codes: tuple[str, ...],
) -> _IntegratedConstraintV2:
    return _IntegratedConstraintV2(
        constraint_domain=None,
        output_domain=_unknown_as_universe(current_published),
        exact_output=_ExactDomainBracketV2(
            inner=empty_exact,
            outer=current_exact.outer,
        ),
        finding_codes=tuple(sorted(set(finding_codes))),
    )


def _publication_failure_finding(
    kind: CandidateConstraintKindV2,
    constraint_id: str,
    failure_kind: str,
    *,
    output: bool,
) -> str:
    if failure_kind == "NUMERIC_GAP":
        return f"NUMERIC_GAP:{kind.value}_PROJECTION:{constraint_id}"
    if failure_kind == "DEGENERATE":
        if kind is CandidateConstraintKindV2.POSITION_DOMAIN:
            return "COMPILATION_INCOMPLETE:POSITION_DOMAIN:DEGENERATE"
        if kind is CandidateConstraintKindV2.SUPPORT:
            label = "INTERSECTION_DEGENERATE" if output else "DEGENERATE"
            return f"COMPILATION_INCOMPLETE:SUPPORT_DOMAIN_{label}:{constraint_id}"
        return (
            f"COMPILATION_INCOMPLETE:{kind.value}_DOMAIN_DEGENERATE_OUTER:"
            f"{constraint_id}"
        )
    phase = "OUTPUT" if output else "DOMAIN"
    return f"COMPILATION_INCOMPLETE:{kind.value}_{phase}_PROJECTION:{constraint_id}"


def _compile_position(
    problem: SemanticProblemV2,
    universe: _SearchUniverseV2,
    exact_universe: _ExactDomainBracketV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> _PositionCompilationV2:
    findings = _position_support_findings(problem)
    if findings:
        return _PositionCompilationV2(exact_domain=None, finding_codes=findings)

    position = problem.constraints.position_domain
    rect = universe.world_rects[0]
    free_ids = position.known_free_space_fact_ids
    if free_ids:
        free_by_id = {
            item.fact_id: item for item in problem.scene.known_free_spaces.values or ()
        }
        free_fact = free_by_id[free_ids[0]]
        try:
            budget.consume()
            free_rect = ExactAxisAlignedRectV2.from_planar_region(
                free_fact.region_world_xy,
                coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
            )
        except UnsupportedRectRegionErrorV2:
            return _PositionCompilationV2(
                exact_domain=None,
                finding_codes=("UNSUPPORTED_POSITION_DOMAIN:FREE_SPACE_REGION",),
            )
        budget.consume()
        rect = rect.intersect(free_rect)

    clearance = position.minimum_boundary_clearance_m
    if clearance:
        budget.consume()
        rect = rect.erode_axis(
            AxisMarginXYV2.from_binary64(x_m=clearance, y_m=clearance)
        )
    anchor = _baseline_anchor(problem)
    budget.consume()
    delta_rect = rect.relative_to(
        WorldPointXYV2.from_binary64(x_m=anchor.x, y_m=anchor.y)
    )
    exact = _normalize_exact_rectangles((delta_rect,), budget, partition_budget)
    clipped = _intersect_exact_regions(
        exact,
        exact_universe.outer,
        budget,
        partition_budget,
    )
    return _PositionCompilationV2(exact_domain=clipped, finding_codes=())


def _position_support_findings(problem: SemanticProblemV2) -> tuple[str, ...]:
    position = problem.constraints.position_domain
    findings: list[str] = []
    if len(position.workspace_fact_ids) != 1:
        findings.append("UNSUPPORTED_POSITION_DOMAIN:MULTIPLE_WORKSPACES")
    if position.workspace_aggregation is not RegionAggregation.INTERSECTION:
        findings.append("UNSUPPORTED_POSITION_DOMAIN:WORKSPACE_AGGREGATION")
    if (
        position.region_interpretation
        is not PositionRegionInterpretation.SUBJECT_ANCHOR_LOCUS
    ):
        findings.append(
            f"UNSUPPORTED_POSITION_DOMAIN:{position.region_interpretation.value}"
        )
    if position.boundary_policy is not BoundaryPolicy.CLOSED:
        findings.append("UNSUPPORTED_POSITION_DOMAIN:BOUNDARY")
    if FactCompletenessV2.EXACT not in position.required_completeness:
        findings.append("UNSUPPORTED_POSITION_DOMAIN:COMPLETENESS_POLICY")
    if len(position.known_free_space_fact_ids) > 1:
        findings.append("UNSUPPORTED_POSITION_DOMAIN:MULTIPLE_FREE_SPACES")
    if not _position_numeric_policy_is_zero(problem.numeric_policy):
        findings.append("UNSUPPORTED_POSITION_DOMAIN:NUMERIC_POLICY")

    findings.extend(
        _fact_family_support_findings(
            problem.scene.workspace_boundaries,
            position.workspace_fact_ids,
            "WORKSPACE",
        )
    )
    if position.known_free_space_fact_ids:
        findings.extend(
            _fact_family_support_findings(
                problem.scene.known_free_spaces,
                position.known_free_space_fact_ids,
                "FREE_SPACE",
            )
        )
    return tuple(sorted(set(findings)))


def _fact_family_support_findings(
    facts: FactSetV2,
    required_ids: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    findings: list[str] = []
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return (f"UNSUPPORTED_POSITION_DOMAIN:{label}_AVAILABILITY",)
    if facts.completeness is not FactCompletenessV2.EXACT:
        findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_COMPLETENESS")
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_FACT_UNCERTAINTY")
    id_field = "fact_id"
    by_id = {getattr(item, id_field): item for item in facts.values or ()}
    for fact_id in required_ids:
        item = by_id.get(fact_id)
        if item is None:
            findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_REFERENCE")
            continue
        if item.region_approximation is not GeometryApproximationV2.EXACT:
            findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_APPROXIMATION")
        if item.boundary_policy is not RegionBoundaryPolicy.CLOSED:
            findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_BOUNDARY")
        if not _uncertainty_is_zero(item.geometry_uncertainty):
            findings.append(f"UNSUPPORTED_POSITION_DOMAIN:{label}_UNCERTAINTY")
    return tuple(findings)


def _position_numeric_policy_is_zero(policy: NumericPolicyV2) -> bool:
    return policy.linear_tolerance_m == 0.0 and policy.area_tolerance_m2 == 0.0


def _uncertainty_is_zero(uncertainty: UncertaintyBudgetV2) -> bool:
    return _numeric_policy_is_zero(
        uncertainty.source_error
    ) and _numeric_policy_is_zero(uncertainty.shape_approximation)


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
            for item in sorted(
                constraints.support_constraints,
                key=lambda item: item.constraint_id,
            )
        ),
        *tuple(
            (item.constraint_id, CandidateConstraintKindV2.COLLISION)
            for item in sorted(
                constraints.collision_constraints,
                key=lambda item: item.constraint_id,
            )
        ),
        *tuple(
            (item.constraint_id, CandidateConstraintKindV2.VISIBILITY)
            for item in sorted(
                constraints.visibility_constraints,
                key=lambda item: item.constraint_id,
            )
        ),
        (
            constraints.target_relation.constraint_id,
            CandidateConstraintKindV2.TARGET_RELATION,
        ),
    )


def _step(
    *,
    index: int,
    constraint_id: str,
    kind: CandidateConstraintKindV2,
    input_domain: PlanarDomainBoundsV2,
    output_domain: PlanarDomainBoundsV2,
    constraint_domain: PlanarDomainBoundsV2 | None = None,
) -> ConstraintDomainShrinkStepV2:
    disposition = (
        ConstraintCompilationDispositionV2.APPLIED
        if constraint_domain is not None
        else ConstraintCompilationDispositionV2.UNKNOWN_AS_UNIVERSE
    )
    return ConstraintDomainShrinkStepV2(
        step_index=index,
        constraint_id=constraint_id,
        constraint_kind=kind,
        disposition=disposition,
        constraint_domain=constraint_domain,
        input_domain=input_domain,
        output_domain=output_domain,
    )


def _artifact(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    search_universe: PlanarDomainBoundsV2,
    ordered: tuple[tuple[str, CandidateConstraintKindV2], ...],
    steps: tuple[ConstraintDomainShrinkStepV2, ...],
    coverage: CandidateCompilationCoverageV2,
    hard_domain: PlanarDomainBoundsV2,
    budget: _DomainOperationBudgetV2,
    partition_budget: RectilinearAtomicBudgetV2,
) -> CandidateDomainArtifactV2:
    # One final cell accounts for the emitted candidate hard-domain artifact;
    # downstream partition stages resume this same cumulative ledger value.
    partition_budget.consume()
    anchor = _baseline_anchor(problem)
    return CandidateDomainArtifactV2(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        core_solver_config_sha256=config.core_solver_config_sha256,
        candidate_variable=CandidateDomainVariableV2(
            subject_id=problem.constraints.allowed_edit.subject_id,
            baseline_anchor_world_xy_m=anchor,
        ),
        search_universe=search_universe,
        ordered_constraint_ids=tuple(item[0] for item in ordered),
        compilation_coverage=coverage,
        compiled_prefix_length=len(steps),
        resource_usage=CompilationResourceUsageV2(
            domain_operations=budget.used,
            partition_cells=partition_budget.used,
            refinement_steps=0,
        ),
        hard_domain=hard_domain,
        shrink_ledger=steps,
    )


def _domain_from_exact_rect(
    rect: ExactAxisAlignedRectV2,
) -> PlanarDomainBoundsV2 | None:
    if rect.topology is RectTopologyV2.EMPTY:
        return _exact_empty_domain()
    if rect.topology is RectTopologyV2.DEGENERATE:
        return None
    inner_projection = rect.project_inner()
    outer_projection = rect.project_outer()
    if outer_projection.topology is RectTopologyV2.DEGENERATE:
        return None
    if outer_projection.topology is RectTopologyV2.EMPTY:
        return _exact_empty_domain()
    outer = PlanarRegionBoundV2.non_empty(outer_projection.to_planar_region())
    if inner_projection.topology is RectTopologyV2.AREA:
        inner = PlanarRegionBoundV2.non_empty(inner_projection.to_planar_region())
    else:
        inner = PlanarRegionBoundV2.empty()
    completeness = (
        DomainCompletenessV2.EXACT if inner == outer else DomainCompletenessV2.BRACKETED
    )
    return PlanarDomainBoundsV2(
        inner_bound=inner,
        outer_bound=outer,
        completeness=completeness,
        coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )

def _intersect_published_domains(
    left: PlanarDomainBoundsV2,
    right: PlanarDomainBoundsV2,
) -> PlanarDomainBoundsV2 | None:
    inner = _intersect_published_bounds(
        left.inner_bound,
        right.inner_bound,
        project_inner=True,
    )
    outer = _intersect_published_bounds(
        left.outer_bound,
        right.outer_bound,
        project_inner=False,
    )
    if inner is None or outer is None:
        return None
    completeness = (
        DomainCompletenessV2.EXACT if inner == outer else DomainCompletenessV2.BRACKETED
    )
    return PlanarDomainBoundsV2(
        inner_bound=inner,
        outer_bound=outer,
        completeness=completeness,
        coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )


def _intersect_published_bounds(
    left: PlanarRegionBoundV2,
    right: PlanarRegionBoundV2,
    *,
    project_inner: bool,
) -> PlanarRegionBoundV2 | None:
    coordinate_space = RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
    if (
        left.status is RegionBoundStatusV2.EMPTY
        or right.status is RegionBoundStatusV2.EMPTY
    ):
        return PlanarRegionBoundV2.empty()
    if (
        left.status is not RegionBoundStatusV2.NON_EMPTY
        or right.status is not RegionBoundStatusV2.NON_EMPTY
        or left.region is None
        or right.region is None
    ):
        raise ValueError("published candidate bounds must be representable")
    left_rect = ExactAxisAlignedRectV2.from_planar_region(
        left.region,
        coordinate_space=coordinate_space,
    )
    right_rect = ExactAxisAlignedRectV2.from_planar_region(
        right.region,
        coordinate_space=coordinate_space,
    )
    intersection = left_rect.intersect(right_rect)
    if intersection.topology is RectTopologyV2.EMPTY:
        return PlanarRegionBoundV2.empty()
    if intersection.topology is RectTopologyV2.DEGENERATE:
        return None
    projection = (
        intersection.project_inner() if project_inner else intersection.project_outer()
    )
    if projection.topology is not RectTopologyV2.AREA:
        return None
    return PlanarRegionBoundV2.non_empty(projection.to_planar_region())


def _exact_empty_domain() -> PlanarDomainBoundsV2:
    bound = PlanarRegionBoundV2.empty()
    return PlanarDomainBoundsV2(
        inner_bound=bound,
        outer_bound=bound,
        completeness=DomainCompletenessV2.EXACT,
        coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )


def _unknown_as_universe(
    domain: PlanarDomainBoundsV2,
) -> PlanarDomainBoundsV2:
    completeness = (
        DomainCompletenessV2.EXACT
        if domain.outer_bound.status is RegionBoundStatusV2.EMPTY
        else DomainCompletenessV2.BRACKETED
    )
    return PlanarDomainBoundsV2(
        inner_bound=PlanarRegionBoundV2.empty(),
        outer_bound=domain.outer_bound,
        completeness=completeness,
        coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )
