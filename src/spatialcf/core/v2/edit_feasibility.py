"""Pure-core replay proof that one concrete Canonical edit is feasible.

The public entry point deliberately accepts no submitted candidate artifact or
verification token.  It recompiles the hard domain from the frozen semantic
problem and solver configuration, then proves only positive membership in the
freshly compiled inner bound.  Failure to prove membership is never relabelled
as infeasibility.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import ClassVar, TypeVar

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2._internal.resources.domain_operations import (
    ValidatedLiveDomainOperationBudgetV2,
)
from spatialcf.core.v2.candidate_domain import (
    CandidateDomainCompilationOutcomeV2,
    CandidateDomainCompilerV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    lift_planar_region_v2,
)
from spatialcf.domain.v2.artifacts import (
    CandidateCompilationCoverageV2,
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    DomainCompletenessV2,
    RegionBoundStatusV2,
)
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import CoreSolverConfigV2, UncertifiedReasonV2

_SHA256_DIGEST_ADAPTER = TypeAdapter(Sha256Digest)


class CanonicalEditFeasibilityKindV2(StrEnum):
    """Closed positive-proof result; there is intentionally no INFEASIBLE kind."""

    VERIFIED_FEASIBLE = "VERIFIED_FEASIBLE"
    NOT_PROVEN = "NOT_PROVEN"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class CanonicalEditFeasibilityVerificationOutcomeV2:
    """Result of an independent concrete-edit replay.

    Hashes are present only for a successful positive proof.  They make the
    proof references explicit, but this value is not an unforgeable token and
    downstream certificate code must perform its own replay.
    """

    kind: CanonicalEditFeasibilityKindV2
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    candidate_domain_artifact_sha256: Sha256Digest | None = None
    canonical_edit_sha256: Sha256Digest | None = None
    verification_resource_usage: CompilationResourceUsageV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CanonicalEditFeasibilityKindV2):
            raise TypeError("kind must be a CanonicalEditFeasibilityKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be an exact tuple of non-blank strings")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )

        usage = self.verification_resource_usage
        if usage is not None:
            if not isinstance(usage, CompilationResourceUsageV2):
                raise TypeError(
                    "verification_resource_usage must be a CompilationResourceUsageV2"
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
                    "verification_resource_usage must pass strict validation"
                ) from error
            object.__setattr__(self, "verification_resource_usage", usage)

        for field_name in (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "candidate_domain_artifact_sha256",
            "canonical_edit_sha256",
        ):
            reference = getattr(self, field_name)
            if reference is None:
                continue
            try:
                reference = _SHA256_DIGEST_ADAPTER.validate_python(
                    reference,
                    strict=True,
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise ValueError(f"{field_name} must be a Sha256Digest") from error
            object.__setattr__(self, field_name, reference)

        references = (
            self.semantic_problem_sha256,
            self.core_solver_config_sha256,
            self.candidate_domain_artifact_sha256,
            self.canonical_edit_sha256,
        )
        if self.kind is CanonicalEditFeasibilityKindV2.VERIFIED_FEASIBLE:
            if any(reference is None for reference in references):
                raise ValueError(
                    "VERIFIED_FEASIBLE requires all four verified references"
                )
            if self.verification_resource_usage is None:
                raise ValueError(
                    "VERIFIED_FEASIBLE requires verification resource usage"
                )
            if self.uncertified_reason is not None or self.finding_codes:
                raise ValueError("VERIFIED_FEASIBLE cannot carry failure diagnostics")
            return

        if any(reference is not None for reference in references):
            raise ValueError(
                "non-VERIFIED_FEASIBLE outcomes cannot carry verified references"
            )
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} requires at least one finding")
        if self.kind is CanonicalEditFeasibilityKindV2.NOT_PROVEN:
            if self.uncertified_reason is not None:
                raise ValueError("NOT_PROVEN cannot carry an uncertified reason")
            return
        if not isinstance(self.uncertified_reason, UncertifiedReasonV2):
            raise TypeError("UNCERTIFIED requires an uncertified reason")


CanonicalEditFeasibilityOutcomeV2 = CanonicalEditFeasibilityVerificationOutcomeV2


class _InvalidInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _NumericInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _DomainOperationBudgetExhaustedV2(RuntimeError):
    """The shared concrete-edit domain-operation ledger is exhausted."""


@dataclass(slots=True)
class _EditFeasibilityDomainBudgetV2(ValidatedLiveDomainOperationBudgetV2):
    """Mutable usage ledger shared by concrete-edit membership checks."""

    _exhaustion_error_type: ClassVar[type[RuntimeError]] = (
        _DomainOperationBudgetExhaustedV2
    )


ModelT = TypeVar("ModelT", bound=V2Model)


def verify_canonical_edit_feasibility_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    edit: CanonicalEditV2,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    """Recompile the hard domain and prove ``edit`` lies in its inner bound."""

    try:
        checked_problem = _strict_model(
            problem,
            SemanticProblemV2,
            label="SEMANTIC_PROBLEM",
        )
        checked_config = _strict_model(
            config,
            CoreSolverConfigV2,
            label="CORE_SOLVER_CONFIG",
        )
        checked_edit = _strict_model(
            edit,
            CanonicalEditV2,
            label="CANONICAL_EDIT",
        )
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

    reference_findings: list[str] = []
    if checked_edit.semantic_problem_sha256 != (
        checked_problem.semantic_problem_sha256
    ):
        reference_findings.append("EDIT_REFERENCE_MISMATCH:SEMANTIC_PROBLEM_HASH")
    if checked_edit.subject_id != checked_problem.constraints.allowed_edit.subject_id:
        reference_findings.append("EDIT_REFERENCE_MISMATCH:SUBJECT_ID")
    if reference_findings:
        return _not_proven(*reference_findings)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = CandidateDomainCompilerV2().compile(
                checked_problem,
                checked_config,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN_REPLAY",
        )

    if not isinstance(replay, CandidateDomainCompilationOutcomeV2):
        raise TypeError(
            "CandidateDomainCompilerV2 returned an invalid internal outcome"
        )
    candidate = replay.candidate_domain
    if candidate is None:
        return _uncertified(
            replay.uncertified_reason or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(replay.finding_codes or ("REPLAY_NO_CANDIDATE_DOMAIN",)),
        )

    replay_usage = candidate.resource_usage
    _require_replay_reference_closure(
        checked_problem,
        checked_config,
        candidate.semantic_problem_sha256,
        candidate.core_solver_config_sha256,
        candidate.candidate_variable.subject_id,
    )
    if replay.uncertified_reason is not None:
        return _uncertified(
            replay.uncertified_reason,
            *(replay.finding_codes or ("REPLAY_CANDIDATE_UNCERTIFIED",)),
            verification_resource_usage=replay_usage,
        )
    atomic_budget = RectilinearAtomicBudgetV2(
        limit=checked_config.max_partition_cells,
        used=replay_usage.partition_cells,
    )
    domain_budget = _EditFeasibilityDomainBudgetV2(
        limit=checked_config.max_domain_operations,
        used=replay_usage.domain_operations,
    )
    return _verify_replayed_candidate_edit_membership_v2(
        checked_problem,
        checked_edit,
        candidate,
        domain_budget,
        atomic_budget,
        cumulative_resource_usage=replay_usage,
    )


def _verify_replayed_candidate_edit_membership_v2(
    problem: SemanticProblemV2,
    edit: CanonicalEditV2,
    candidate: CandidateDomainArtifactV2,
    domain_budget: _EditFeasibilityDomainBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
    *,
    cumulative_resource_usage: CompilationResourceUsageV2 | None = None,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    """Verify one edit against a caller-owned fresh replay and shared ledger.

    This module-private seam exists for a solver that has already performed a
    strict candidate replay and owns one global budget across many proposals.
    It neither recompiles nor resets that ledger.  It is not a trusted public
    shortcut: callers outside that pipeline must use the public replay entry
    point above.

    Candidate compilation owns its prefix of the domain-operation ledger.  A
    concrete inner lift/membership stage reserves one further domain operation
    and consumes its exact rectilinear work from the caller-owned atomic ledger.
    Refinement usage is carried forward unchanged from the supplied cumulative
    usage (or the candidate replay when omitted).
    """

    if not isinstance(problem, SemanticProblemV2):
        raise TypeError("problem must be a strict SemanticProblemV2")
    if not isinstance(edit, CanonicalEditV2):
        raise TypeError("edit must be a strict CanonicalEditV2")
    if not isinstance(candidate, CandidateDomainArtifactV2):
        raise TypeError("candidate must be a fresh CandidateDomainArtifactV2")
    if type(domain_budget) is not _EditFeasibilityDomainBudgetV2:
        raise TypeError("domain_budget must be an _EditFeasibilityDomainBudgetV2")
    if type(atomic_budget) is not RectilinearAtomicBudgetV2:
        raise TypeError("atomic_budget must be a RectilinearAtomicBudgetV2")
    domain_budget.validate()
    atomic_budget.validate()
    if domain_budget.used < candidate.resource_usage.domain_operations:
        raise ValueError("shared domain ledger predates the candidate replay")
    if atomic_budget.used < candidate.resource_usage.partition_cells:
        raise ValueError("shared atomic budget predates the candidate replay")

    if cumulative_resource_usage is None:
        base_usage = CompilationResourceUsageV2(
            domain_operations=domain_budget.used,
            partition_cells=atomic_budget.used,
            refinement_steps=candidate.resource_usage.refinement_steps,
        )
    else:
        base_usage = cumulative_resource_usage
    if type(base_usage) is not CompilationResourceUsageV2:
        raise TypeError(
            "cumulative_resource_usage must be a CompilationResourceUsageV2"
        )
    candidate_usage = candidate.resource_usage
    if (
        base_usage.domain_operations < candidate_usage.domain_operations
        or base_usage.partition_cells < candidate_usage.partition_cells
        or base_usage.refinement_steps < candidate_usage.refinement_steps
    ):
        raise ValueError("cumulative resource usage rollback below candidate usage")
    if base_usage.domain_operations != domain_budget.used:
        raise ValueError("cumulative usage must equal the shared domain ledger")
    if base_usage.partition_cells != atomic_budget.used:
        raise ValueError("cumulative usage must equal the shared atomic ledger")

    reference_findings: list[str] = []
    if edit.semantic_problem_sha256 != problem.semantic_problem_sha256:
        reference_findings.append("EDIT_REFERENCE_MISMATCH:SEMANTIC_PROBLEM_HASH")
    if edit.subject_id != problem.constraints.allowed_edit.subject_id:
        reference_findings.append("EDIT_REFERENCE_MISMATCH:SUBJECT_ID")
    if reference_findings:
        return _not_proven(
            *reference_findings,
            verification_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )
    if candidate.semantic_problem_sha256 != problem.semantic_problem_sha256:
        raise RuntimeError("candidate replay problem hash is not closed")
    if candidate.candidate_variable.subject_id != (
        problem.constraints.allowed_edit.subject_id
    ):
        raise RuntimeError("candidate replay subject is not closed")

    current_usage = _cumulative_usage(base_usage, domain_budget, atomic_budget)
    if candidate.compilation_coverage is not CandidateCompilationCoverageV2.COMPLETE:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            "COMPILATION_INCOMPLETE:CANDIDATE_DOMAIN_NOT_COMPLETE",
            verification_resource_usage=current_usage,
        )
    hard_domain = candidate.hard_domain
    if hard_domain.completeness not in {
        DomainCompletenessV2.EXACT,
        DomainCompletenessV2.BRACKETED,
    }:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            "COMPILATION_INCOMPLETE:HARD_INNER_BOUND_UNKNOWN",
            verification_resource_usage=current_usage,
        )
    inner_bound = hard_domain.inner_bound
    if inner_bound.status is RegionBoundStatusV2.UNAVAILABLE:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            "COMPILATION_INCOMPLETE:HARD_INNER_BOUND_UNKNOWN",
            verification_resource_usage=current_usage,
        )
    if inner_bound.status is RegionBoundStatusV2.EMPTY:
        return _not_proven(
            "EDIT_NOT_IN_PROVEN_HARD_INNER",
            verification_resource_usage=current_usage,
        )
    if inner_bound.region is None:  # pragma: no cover - schema invariant
        raise RuntimeError("NON_EMPTY hard inner bound has no region")

    try:
        # Reserve the whole semantic membership stage before beginning the
        # potentially multi-pass rectilinear lift.
        domain_budget.consume()
    except _DomainOperationBudgetExhaustedV2:
        return _domain_resource_uncertified(
            _cumulative_usage(base_usage, domain_budget, atomic_budget),
            domain_budget.limit,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            lifted = lift_planar_region_v2(
                inner_bound.region,
                atomic_budget=atomic_budget,
            )
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource_uncertified(
            _cumulative_usage(base_usage, domain_budget, atomic_budget),
            atomic_budget.limit,
        )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:HARD_INNER_LIFT",
            verification_resource_usage=_cumulative_usage(
                base_usage,
                domain_budget,
                atomic_budget,
            ),
        )

    current_usage = _cumulative_usage(base_usage, domain_budget, atomic_budget)
    if lifted.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource_uncertified(current_usage, atomic_budget.limit)
    if lifted.kind is RectilinearOutcomeKindV2.UNKNOWN:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(lifted.finding_codes or ("COMPILATION_INCOMPLETE:HARD_INNER_LIFT",)),
            verification_resource_usage=current_usage,
        )
    if lifted.region is None:  # pragma: no cover - outcome invariant
        raise RuntimeError("EXACT hard inner lift has no exact region")

    point_x = Fraction.from_float(edit.translation_xy_m.x)
    point_y = Fraction.from_float(edit.translation_xy_m.y)
    try:
        contained = _contains_point_with_budget(
            lifted.region,
            point_x,
            point_y,
            atomic_budget,
        )
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource_uncertified(
            _cumulative_usage(base_usage, domain_budget, atomic_budget),
            atomic_budget.limit,
        )

    final_usage = _cumulative_usage(base_usage, domain_budget, atomic_budget)
    if not contained:
        return _not_proven(
            "EDIT_NOT_IN_PROVEN_HARD_INNER",
            verification_resource_usage=final_usage,
        )
    return CanonicalEditFeasibilityVerificationOutcomeV2(
        kind=CanonicalEditFeasibilityKindV2.VERIFIED_FEASIBLE,
        semantic_problem_sha256=problem.semantic_problem_sha256,
        core_solver_config_sha256=candidate.core_solver_config_sha256,
        candidate_domain_artifact_sha256=(candidate.candidate_domain_artifact_sha256),
        canonical_edit_sha256=edit.edit_sha256,
        verification_resource_usage=final_usage,
    )


class CanonicalEditFeasibilityVerifierV2:
    """Stateless wrapper for pure-core pipeline composition."""

    def verify(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        edit: CanonicalEditV2,
    ) -> CanonicalEditFeasibilityVerificationOutcomeV2:
        return verify_canonical_edit_feasibility_v2(problem, config, edit)


def _strict_model(
    value: object,
    model_type: type[ModelT],
    *,
    label: str,
) -> ModelT:
    if not isinstance(value, model_type):
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


def _require_replay_reference_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate_problem_hash: str,
    candidate_config_hash: str,
    candidate_subject_id: str,
) -> None:
    """Treat a public compiler violating its own closure as an internal error."""

    if candidate_problem_hash != problem.semantic_problem_sha256:
        raise RuntimeError("candidate replay problem hash is not closed")
    if candidate_config_hash != config.core_solver_config_sha256:
        raise RuntimeError("candidate replay config hash is not closed")
    if candidate_subject_id != problem.constraints.allowed_edit.subject_id:
        raise RuntimeError("candidate replay subject is not closed")


def _contains_point_with_budget(
    region: ExactRectilinearRegionV2,
    x_m: Fraction,
    y_m: Fraction,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> bool:
    """Check every canonical rectangle, charging the shared ledger exactly once."""

    contained = False
    for rectangle in region.rectangles:
        atomic_budget.consume()
        bounds = rectangle.bounds
        if bounds is None:  # pragma: no cover - normalized region invariant
            raise RuntimeError("normalized rectilinear region contains an empty cell")
        inside_rectangle = (
            bounds[0] <= x_m <= bounds[2] and bounds[1] <= y_m <= bounds[3]
        )
        contained = contained or inside_rectangle
    return contained


def _cumulative_usage(
    base_usage: CompilationResourceUsageV2,
    domain_budget: _EditFeasibilityDomainBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=domain_budget.used,
        partition_cells=atomic_budget.used,
        refinement_steps=base_usage.refinement_steps,
    )


def _domain_resource_uncertified(
    replay_usage: CompilationResourceUsageV2,
    limit: int,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    return _uncertified(
        UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
        f"RESOURCE_LIMIT:max_domain_operations:{limit}",
        verification_resource_usage=replay_usage,
    )


def _resource_uncertified(
    replay_usage: CompilationResourceUsageV2,
    limit: int,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    # ``limit`` is included to make cap/cap-1 diagnostics self-contained while
    # the cumulative usage records how far the all-or-nothing operation got.
    return _uncertified(
        UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
        f"RESOURCE_LIMIT:max_partition_cells:{limit}",
        verification_resource_usage=replay_usage,
    )


def _not_proven(
    *finding_codes: str,
    verification_resource_usage: CompilationResourceUsageV2 | None = None,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    return CanonicalEditFeasibilityVerificationOutcomeV2(
        kind=CanonicalEditFeasibilityKindV2.NOT_PROVEN,
        finding_codes=finding_codes,
        verification_resource_usage=verification_resource_usage,
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    verification_resource_usage: CompilationResourceUsageV2 | None = None,
) -> CanonicalEditFeasibilityVerificationOutcomeV2:
    return CanonicalEditFeasibilityVerificationOutcomeV2(
        kind=CanonicalEditFeasibilityKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=finding_codes,
        verification_resource_usage=verification_resource_usage,
    )
