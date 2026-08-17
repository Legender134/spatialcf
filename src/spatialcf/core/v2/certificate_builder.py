"""Pure-core certificate builders backed by fresh semantic replay.

Both public entry points accept only raw semantic inputs.  They rebuild every
artifact used by the claim and never consume a submitted artifact, point
outcome, certificate, or previously issued verification token.
"""

from __future__ import annotations

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
from spatialcf.core.v2.candidate_domain import (
    _registry_finding as _candidate_registry_finding,
)
from spatialcf.core.v2.edit_feasibility import (
    _DomainOperationBudgetExhaustedV2,
    _EditFeasibilityDomainBudgetV2,
)
from spatialcf.core.v2.objective_partition import (
    ObjectivePartitionCompilationKindV2,
    ObjectivePartitionCompilationOutcomeV2,
    ObjectiveWitnessProposalV2,
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
    PointObjectiveEvaluationKindV2,
    PointObjectiveEvaluationOutcomeV2,
    _context_usage,
    _evaluate_point_objective_from_replay_v2,
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
    ConstraintCompilationDispositionV2,
    DomainCompletenessV2,
    ObjectivePartitionCellV2,
    ObjectiveTermBoundsV2,
    RegionBoundStatusV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.certificate import (
    EmptyOuterProofMethodV2,
    GlobalOptimalityCertificateV2,
    OptimalityClaimV2,
    ProvenUnsatCertificateV2,
    _directed_binary64_gap_ceil,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    UncertifiedReasonV2,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_SHA256_DIGEST_ADAPTER = TypeAdapter(Sha256Digest)


class CertificateBuildKindV2(StrEnum):
    """Closed result kinds currently implemented by this builder module."""

    PROVEN_UNSAT = "PROVEN_UNSAT"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class CertificateBuildOutcomeV2:
    """Closed result of a fresh certificate-building replay.

    Verified hashes exist only alongside the strictly reconstructed result.
    Like every in-process value, this outcome is not an unforgeable token;
    consumers that make a semantic claim must independently replay the raw
    inputs.
    """

    kind: CertificateBuildKindV2
    proven_unsat_result: ProvenUnsatResultV2 | None = None
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    candidate_domain_artifact_sha256: Sha256Digest | None = None
    certificate_sha256: Sha256Digest | None = None
    solve_result_sha256: Sha256Digest | None = None
    verification_resource_usage: CompilationResourceUsageV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CertificateBuildKindV2:
            raise TypeError("kind must be a CertificateBuildKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be an exact tuple of non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))

        result = self.proven_unsat_result
        if result is not None:
            if type(result) is not ProvenUnsatResultV2:
                raise TypeError("proven_unsat_result must be a ProvenUnsatResultV2")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    result = ProvenUnsatResultV2.model_validate(
                        result.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "proven_unsat_result must pass strict validation"
                ) from error

        usage = self.verification_resource_usage
        if usage is not None:
            if type(usage) is not CompilationResourceUsageV2:
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

        for field_name in (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "candidate_domain_artifact_sha256",
            "certificate_sha256",
            "solve_result_sha256",
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

        object.__setattr__(self, "finding_codes", findings)
        object.__setattr__(self, "proven_unsat_result", result)
        object.__setattr__(self, "verification_resource_usage", usage)

        references = (
            self.semantic_problem_sha256,
            self.core_solver_config_sha256,
            self.candidate_domain_artifact_sha256,
            self.certificate_sha256,
            self.solve_result_sha256,
        )
        if self.kind is CertificateBuildKindV2.PROVEN_UNSAT:
            if (
                result is None
                or usage is None
                or any(item is None for item in references)
            ):
                raise ValueError(
                    "PROVEN_UNSAT requires a result, usage, and all verified references"
                )
            if self.uncertified_reason is not None or findings:
                raise ValueError("PROVEN_UNSAT cannot carry failure diagnostics")
            expected_references = (
                result.semantic_problem_sha256,
                result.core_solver_config.core_solver_config_sha256,
                result.candidate_domain.candidate_domain_artifact_sha256,
                result.certificate.certificate_sha256,
                result.solve_result_sha256,
            )
            if references != expected_references:
                raise ValueError("PROVEN_UNSAT verified references are not closed")
            if usage != result.candidate_domain.resource_usage:
                raise ValueError("PROVEN_UNSAT usage is not closed to fresh replay")
            return

        if result is not None or any(item is not None for item in references):
            raise ValueError("UNCERTIFIED cannot carry a result or verified references")
        if type(self.uncertified_reason) is not UncertifiedReasonV2:
            raise TypeError("UNCERTIFIED requires an UncertifiedReasonV2")
        if not findings:
            raise ValueError("UNCERTIFIED requires at least one finding")


ProvenUnsatBuildOutcomeV2 = CertificateBuildOutcomeV2


class GlobalOptimalityBuildKindV2(StrEnum):
    """Closed outcomes of the fresh global-optimality replay."""

    CERTIFIED_SUCCESS = "CERTIFIED_SUCCESS"
    NOT_PROVEN = "NOT_PROVEN"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class GlobalOptimalityBuildOutcomeV2:
    """Auditable value returned by the global builder, never a proof token.

    A consumer making a semantic claim must replay the raw problem,
    configuration, and edit again.  The references below only close a
    successful value to the strictly reconstructed result built by this call.
    """

    kind: GlobalOptimalityBuildKindV2
    certified_success_result: CertifiedSuccessResultV2 | None = None
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    candidate_domain_artifact_sha256: Sha256Digest | None = None
    relation_cost_partition_sha256: Sha256Digest | None = None
    objective_partition_artifact_sha256: Sha256Digest | None = None
    canonical_edit_sha256: Sha256Digest | None = None
    certificate_sha256: Sha256Digest | None = None
    solve_result_sha256: Sha256Digest | None = None
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not GlobalOptimalityBuildKindV2:
            raise TypeError("kind must be a GlobalOptimalityBuildKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))

        result = self.certified_success_result
        if result is not None:
            if type(result) is not CertifiedSuccessResultV2:
                raise TypeError(
                    "certified_success_result must be a CertifiedSuccessResultV2"
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    result = CertifiedSuccessResultV2.model_validate(
                        result.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "certified_success_result must pass strict validation"
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

        reference_names = (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "candidate_domain_artifact_sha256",
            "relation_cost_partition_sha256",
            "objective_partition_artifact_sha256",
            "canonical_edit_sha256",
            "certificate_sha256",
            "solve_result_sha256",
        )
        for field_name in reference_names:
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

        object.__setattr__(self, "finding_codes", findings)
        object.__setattr__(self, "certified_success_result", result)
        object.__setattr__(self, "cumulative_generation_usage", usage)
        references = tuple(getattr(self, name) for name in reference_names)

        if self.kind is GlobalOptimalityBuildKindV2.CERTIFIED_SUCCESS:
            if (
                result is None
                or usage is None
                or any(item is None for item in references)
            ):
                raise ValueError(
                    "CERTIFIED_SUCCESS requires a result, usage, and every reference"
                )
            if self.uncertified_reason is not None or findings:
                raise ValueError("CERTIFIED_SUCCESS cannot carry failure diagnostics")
            expected = (
                result.semantic_problem_sha256,
                result.core_solver_config.core_solver_config_sha256,
                result.candidate_domain.candidate_domain_artifact_sha256,
                result.relation_cost_partition.relation_cost_partition_sha256,
                result.objective_partition.objective_partition_artifact_sha256,
                result.edit.edit_sha256,
                result.certificate.certificate_sha256,
                result.solve_result_sha256,
            )
            if references != expected:
                raise ValueError("CERTIFIED_SUCCESS verified references are not closed")
            base = result.candidate_domain.resource_usage
            limits = result.core_solver_config
            if (
                usage.domain_operations < base.domain_operations
                or usage.partition_cells < base.partition_cells
                or usage.refinement_steps != base.refinement_steps
                or usage.domain_operations > limits.max_domain_operations
                or usage.partition_cells > limits.max_partition_cells
            ):
                raise ValueError(
                    "CERTIFIED_SUCCESS cumulative usage is not closed to its replay"
                )
            return

        if result is not None or any(item is not None for item in references):
            raise ValueError("non-success outcomes cannot carry verified payload")
        if not findings:
            raise ValueError(f"{self.kind.value} requires at least one finding")
        if self.kind is GlobalOptimalityBuildKindV2.NOT_PROVEN:
            if self.uncertified_reason is not None:
                raise ValueError("NOT_PROVEN cannot carry an uncertified reason")
            return
        if type(self.uncertified_reason) is not UncertifiedReasonV2:
            raise TypeError("UNCERTIFIED requires an UncertifiedReasonV2")


GlobalOptimalityCertificateBuildOutcomeV2 = GlobalOptimalityBuildOutcomeV2


class _InvalidInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _NumericInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


ModelT = TypeVar("ModelT", bound=V2Model)


def build_proven_unsat_result_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
) -> CertificateBuildOutcomeV2:
    """Build PROVEN_UNSAT only from a fresh complete empty-outer replay."""

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
    except _NumericInputV2 as error:
        return _uncertified(UncertifiedReasonV2.NUMERIC_GAP, error.finding_code)
    except _InvalidInputV2 as error:
        return _uncertified(UncertifiedReasonV2.UNSUPPORTED_MODEL, error.finding_code)

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

    if type(replay) is not CandidateDomainCompilationOutcomeV2:
        raise TypeError(
            "CandidateDomainCompilerV2 returned an invalid internal outcome"
        )
    candidate = replay.candidate_domain
    if candidate is None:
        return _uncertified(
            replay.uncertified_reason or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(replay.finding_codes or ("REPLAY_NO_CANDIDATE_DOMAIN",)),
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate = CandidateDomainArtifactV2.model_validate(
                candidate.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN_REVALIDATION",
        )

    usage = candidate.resource_usage
    _require_candidate_reference_closure(checked_problem, checked_config, candidate)
    if replay.uncertified_reason is not None:
        return _uncertified(
            replay.uncertified_reason,
            *(replay.finding_codes or ("REPLAY_CANDIDATE_UNCERTIFIED",)),
            verification_resource_usage=usage,
        )

    return _build_proven_unsat_from_fresh_candidate_v2(
        checked_problem,
        checked_config,
        candidate,
    )


def _build_proven_unsat_from_fresh_candidate_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> CertificateBuildOutcomeV2:
    """Assemble UNSAT from a caller-owned fresh candidate without replay.

    This private seam is not a verification token.  Its caller must have just
    produced ``candidate`` from the same raw inputs; strict reconstruction and
    reference closure here turn any violated internal promise into an
    invariant error rather than silently accepting a submitted artifact.
    """

    checked_problem = _strict_fresh_internal_model(
        problem,
        SemanticProblemV2,
        "SEMANTIC_PROBLEM",
    )
    checked_config = _strict_fresh_internal_model(
        config,
        CoreSolverConfigV2,
        "CORE_SOLVER_CONFIG",
    )
    checked_candidate = _strict_fresh_internal_model(
        candidate,
        CandidateDomainArtifactV2,
        "CANDIDATE_DOMAIN",
    )
    _require_candidate_reference_closure(
        checked_problem,
        checked_config,
        checked_candidate,
    )
    usage = checked_candidate.resource_usage
    if (
        usage.domain_operations > checked_config.max_domain_operations
        or usage.partition_cells > checked_config.max_partition_cells
        or usage.refinement_steps > checked_config.max_refinement_steps
    ):
        raise RuntimeError("fresh UNSAT candidate exceeded its configured resource cap")
    registry_finding = _candidate_registry_finding(checked_config)
    if registry_finding is not None:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
            verification_resource_usage=usage,
        )
    unsat_findings = _unsat_eligibility_findings(checked_candidate)
    if unsat_findings:
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *unsat_findings,
            verification_resource_usage=usage,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            certificate = ProvenUnsatCertificateV2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                core_solver_config_sha256=(checked_config.core_solver_config_sha256),
                candidate_domain_artifact_sha256=(
                    checked_candidate.candidate_domain_artifact_sha256
                ),
                empty_outer_proof_method=(
                    EmptyOuterProofMethodV2.CERTIFIED_EMPTY_OUTER_DOMAIN
                ),
            )
            certificate = ProvenUnsatCertificateV2.model_validate(
                certificate.model_dump(mode="python"),
                strict=True,
            )
            result = ProvenUnsatResultV2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                core_solver_config=checked_config,
                candidate_domain=checked_candidate,
                certificate=certificate,
            )
            result = ProvenUnsatResultV2.model_validate(
                result.model_dump(mode="python"),
                strict=True,
            )
            _require_canonical_round_trip(certificate, result)
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:PROVEN_UNSAT_CONSTRUCTION",
            verification_resource_usage=usage,
        )

    return CertificateBuildOutcomeV2(
        kind=CertificateBuildKindV2.PROVEN_UNSAT,
        proven_unsat_result=result,
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        core_solver_config_sha256=checked_config.core_solver_config_sha256,
        candidate_domain_artifact_sha256=(
            checked_candidate.candidate_domain_artifact_sha256
        ),
        certificate_sha256=certificate.certificate_sha256,
        solve_result_sha256=result.solve_result_sha256,
        verification_resource_usage=usage,
    )


class ProvenUnsatCertificateBuilderV2:
    """Stateless wrapper for pure-core pipeline composition."""

    def build(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
    ) -> CertificateBuildOutcomeV2:
        return build_proven_unsat_result_v2(problem, config)


@dataclass(frozen=True, slots=True)
class _GlobalReplayBundleV2:
    """Fresh artifacts and the two mutable ledgers owned by one replay."""

    problem: SemanticProblemV2
    config: CoreSolverConfigV2
    candidate: CandidateDomainArtifactV2
    relation: RelationCostPartitionV2
    objective_outcome: ObjectivePartitionCompilationOutcomeV2
    point_context: _PointObjectiveReplayContextV2


@dataclass(frozen=True, slots=True)
class _ExactCardinalSelectionFrameV2:
    """Exact map from normalized edit XY into the semantic tie-break frame."""

    normalized_to_semantic_quarter_turns_ccw: int = 0

    def __post_init__(self) -> None:
        quarter_turns = self.normalized_to_semantic_quarter_turns_ccw
        if type(quarter_turns) is not int or not 0 <= quarter_turns <= 3:
            raise TypeError("selection frame quarter-turn must be an exact int in 0..3")

    def semantic_xy(
        self,
        x: Fraction,
        y: Fraction,
    ) -> tuple[Fraction, Fraction]:
        if type(x) is not Fraction or type(y) is not Fraction:
            raise TypeError("selection frame coordinates must be exact Fractions")
        quarter_turns = self.normalized_to_semantic_quarter_turns_ccw
        if quarter_turns == 0:
            return x, y
        if quarter_turns == 1:
            return -y, x
        if quarter_turns == 2:
            return -x, -y
        return y, -x


@dataclass(frozen=True, slots=True)
class _EvaluatedGlobalProposalV2:
    proposal: ObjectiveWitnessProposalV2
    point: PointObjectiveEvaluationOutcomeV2
    singleton_values: tuple[Fraction, Fraction, Fraction, Fraction] | None
    singleton_total: Fraction | None


@dataclass(frozen=True, slots=True)
class _SelectedGlobalBuildOutcomeV2:
    """Private counted result for a solver-owned fresh replay bundle."""

    outcome: GlobalOptimalityBuildOutcomeV2
    proposal_count: int
    evaluated_proposal_count: int

    def __post_init__(self) -> None:
        if type(self.outcome) is not GlobalOptimalityBuildOutcomeV2:
            raise TypeError("outcome must be a GlobalOptimalityBuildOutcomeV2")
        if type(self.proposal_count) is not int or self.proposal_count < 0:
            raise TypeError("proposal_count must be a non-negative exact int")
        if (
            type(self.evaluated_proposal_count) is not int
            or self.evaluated_proposal_count < 0
            or self.evaluated_proposal_count > self.proposal_count
        ):
            raise ValueError("evaluated_proposal_count must lie within proposal_count")
        if (
            self.outcome.kind is GlobalOptimalityBuildKindV2.CERTIFIED_SUCCESS
            and self.evaluated_proposal_count != self.proposal_count
        ):
            raise ValueError("successful selected build must evaluate every proposal")


def build_global_optimality_result_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    edit: CanonicalEditV2,
) -> GlobalOptimalityBuildOutcomeV2:
    """Build a certified global result from one complete fresh core replay.

    This conservative first implementation accepts only exact, non-empty
    objective cells whose non-translation loss intervals are singletons.  It
    evaluates exactly one canonical inner proposal per cell with shared ledgers.
    """

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
        return _global_uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            error.finding_code,
        )
    except _InvalidInputV2 as error:
        return _global_uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )

    edit_findings = _global_edit_reference_findings(checked_problem, checked_edit)
    if edit_findings:
        return _global_not_proven(*edit_findings)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate_outcome = CandidateDomainCompilerV2().compile(
                checked_problem,
                checked_config,
            )
    except (ArithmeticError, RuntimeWarning):
        return _global_uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:GLOBAL_CANDIDATE_REPLAY",
        )
    if type(candidate_outcome) is not CandidateDomainCompilationOutcomeV2:
        raise TypeError("candidate compiler returned an invalid internal outcome")
    candidate = candidate_outcome.candidate_domain
    if candidate is None:
        return _global_uncertified(
            candidate_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(candidate_outcome.finding_codes or ("GLOBAL_REPLAY_NO_CANDIDATE",)),
        )
    if type(candidate) is not CandidateDomainArtifactV2:
        raise TypeError("candidate compiler returned an invalid artifact")
    _require_candidate_closure(checked_problem, checked_config, candidate)
    if candidate_outcome.uncertified_reason is not None:
        return _global_uncertified(
            candidate_outcome.uncertified_reason,
            *(candidate_outcome.finding_codes or ("GLOBAL_CANDIDATE_UNCERTIFIED",)),
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
        return _global_uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:GLOBAL_RELATION_REPLAY",
            cumulative_generation_usage=candidate.resource_usage,
        )
    if type(relation_outcome) is not RelationCostPartitionCompilationOutcomeV2:
        raise TypeError("relation compiler returned an invalid internal outcome")
    relation_usage = relation_outcome.cumulative_resource_usage
    if relation_outcome.kind is not RelationCostPartitionCompilationKindV2.PARTITION:
        return _global_uncertified(
            relation_outcome.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(relation_outcome.finding_codes or ("GLOBAL_REPLAY_NO_RELATION",)),
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
        return _global_uncertified(
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            str(error) or "RESOURCE_LIMIT:max_partition_cells",
            cumulative_generation_usage=_global_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveNumericGapV2 as error:
        return _global_uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            *error.finding_codes,
            cumulative_generation_usage=_global_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveUnsupportedV2 as error:
        return _global_uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *error.finding_codes,
            cumulative_generation_usage=_global_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except _ObjectiveCompilationIncompleteV2 as error:
        return _global_uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *error.finding_codes,
            cumulative_generation_usage=_global_objective_usage(
                relation_usage,
                objective_domain_budget,
                atomic_budget,
            ),
        )
    except (ArithmeticError, RuntimeWarning):
        return _global_uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:GLOBAL_OBJECTIVE_REPLAY",
            cumulative_generation_usage=_global_objective_usage(
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
        raise RuntimeError("global objective domain ledger drift")
    if atomic_budget.used != objective_usage.partition_cells:
        raise RuntimeError("global objective atomic ledger drift")
    if objective_usage.refinement_steps != relation_usage.refinement_steps:
        raise RuntimeError("global objective refinement ledger drift")
    _require_objective_closure(
        checked_problem,
        checked_config,
        candidate,
        relation,
        objective,
    )

    point_context = _PointObjectiveReplayContextV2(
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
    return _build_global_from_fresh_replay_v2(
        _GlobalReplayBundleV2(
            problem=checked_problem,
            config=checked_config,
            candidate=candidate,
            relation=relation,
            objective_outcome=objective_outcome,
            point_context=point_context,
        ),
        checked_edit,
    )


class GlobalOptimalityCertificateBuilderV2:
    """Stateless wrapper for the raw-input global certificate builder."""

    def build(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        edit: CanonicalEditV2,
    ) -> GlobalOptimalityBuildOutcomeV2:
        return build_global_optimality_result_v2(problem, config, edit)


def _build_global_from_fresh_replay_v2(
    bundle: _GlobalReplayBundleV2,
    requested_edit: CanonicalEditV2,
) -> GlobalOptimalityBuildOutcomeV2:
    """Build for a caller edit while preserving the historical private API."""

    return _build_global_from_fresh_replay_counted_v2(
        bundle,
        requested_edit,
        _ExactCardinalSelectionFrameV2(),
    ).outcome


def _build_selected_global_from_fresh_replay_v2(
    bundle: _GlobalReplayBundleV2,
) -> _SelectedGlobalBuildOutcomeV2:
    """Auto-select and assemble from one solver-owned fresh replay bundle."""

    return _build_selected_global_from_fresh_replay_in_frame_v2(
        bundle,
        _ExactCardinalSelectionFrameV2(),
    )


def _build_selected_global_from_fresh_replay_in_frame_v2(
    bundle: _GlobalReplayBundleV2,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> _SelectedGlobalBuildOutcomeV2:
    """Auto-select in an exact caller-owned semantic coordinate frame."""

    if type(selection_frame) is not _ExactCardinalSelectionFrameV2:
        raise TypeError("selection_frame has the wrong exact type")
    return _build_global_from_fresh_replay_counted_v2(
        bundle,
        None,
        selection_frame,
    )


def _build_global_from_fresh_replay_counted_v2(
    bundle: _GlobalReplayBundleV2,
    requested_edit: CanonicalEditV2 | None,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> _SelectedGlobalBuildOutcomeV2:
    """Assemble from caller-owned *fresh* replay state and shared ledgers.

    This private seam is for a future solver that already owns the same fresh
    prefix.  It is not a capability or proof token and must never be exposed as
    accepting submitted artifacts.
    """

    _validate_global_replay_bundle_prefix(bundle, requested_edit)
    objective = bundle.point_context.objective
    proposals = bundle.objective_outcome.witness_proposals
    proposal_count = len(proposals)
    evaluated_proposal_count = 0

    def finish(
        outcome: GlobalOptimalityBuildOutcomeV2,
    ) -> _SelectedGlobalBuildOutcomeV2:
        return _SelectedGlobalBuildOutcomeV2(
            outcome=outcome,
            proposal_count=proposal_count,
            evaluated_proposal_count=evaluated_proposal_count,
        )

    try:
        _reserve_global_builder_work(bundle.point_context, proposal_count)
    except _DomainOperationBudgetExhaustedV2:
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
                "RESOURCE_LIMIT:max_domain_operations",
                cumulative_generation_usage=_context_usage(bundle.point_context),
            )
        )
    _validate_global_replay_bundle_closure(bundle, requested_edit)

    eligibility_findings = _global_eligibility_findings(bundle, proposals)
    if eligibility_findings:
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.COMPILATION_INCOMPLETE,
                *eligibility_findings,
                cumulative_generation_usage=_context_usage(bundle.point_context),
            )
        )

    global_lower = _global_loss_lower_bound_v2(objective.cells)
    evaluated: list[_EvaluatedGlobalProposalV2] = []
    for proposal in proposals:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                point = _evaluate_point_objective_from_replay_v2(
                    bundle.point_context,
                    proposal.edit,
                )
        except (ArithmeticError, RuntimeWarning):
            return finish(
                _global_uncertified(
                    UncertifiedReasonV2.NUMERIC_GAP,
                    "NUMERIC_GAP:GLOBAL_POINT_EVALUATION",
                    cumulative_generation_usage=_context_usage(bundle.point_context),
                )
            )
        evaluated_proposal_count += 1
        if type(point) is not PointObjectiveEvaluationOutcomeV2:
            raise TypeError("point evaluator returned an invalid internal outcome")
        if point.kind is PointObjectiveEvaluationKindV2.NOT_PROVEN:
            findings = point.finding_codes or ("GLOBAL_PROPOSAL_NOT_PROVEN",)
            raise RuntimeError(
                "fresh objective proposal lost inner feasibility: " + "|".join(findings)
            )
        if point.kind is PointObjectiveEvaluationKindV2.UNCERTIFIED:
            return finish(
                _global_uncertified(
                    point.uncertified_reason
                    or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
                    *(point.finding_codes or ("GLOBAL_PROPOSAL_UNCERTIFIED",)),
                    cumulative_generation_usage=_context_usage(bundle.point_context),
                )
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Warning)
                _require_global_point_closure(bundle, proposal, point)
                bounds = point.witness_loss_bounds
                if bounds is None:  # pragma: no cover - point outcome invariant
                    raise RuntimeError(
                        "bounded point outcome omitted witness loss bounds"
                    )
                singleton_values = _singleton_term_values(bounds)
                evaluated.append(
                    _EvaluatedGlobalProposalV2(
                        proposal=proposal,
                        point=point,
                        singleton_values=singleton_values,
                        singleton_total=(
                            sum(singleton_values, Fraction())
                            if singleton_values is not None
                            else None
                        ),
                    )
                )
        except (ArithmeticError, RuntimeWarning):
            return finish(
                _global_uncertified(
                    UncertifiedReasonV2.NUMERIC_GAP,
                    "NUMERIC_GAP:GLOBAL_POINT_POSTPROCESS",
                    cumulative_generation_usage=_context_usage(bundle.point_context),
                )
            )

    final_usage = _context_usage(bundle.point_context)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            selected = _select_certifiable_global_winner(
                tuple(evaluated),
                selection_frame=selection_frame,
            )
    except (ArithmeticError, RuntimeWarning):
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.NUMERIC_GAP,
                "NUMERIC_GAP:GLOBAL_SELECTION",
                cumulative_generation_usage=final_usage,
            )
        )
    if selected is None:
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.COMPILATION_INCOMPLETE,
                "TIE_BREAK_UNRESOLVED",
                cumulative_generation_usage=final_usage,
            )
        )
    selected_edit = selected.proposal.edit
    if requested_edit is not None and canonical_json_bytes_v2(
        requested_edit
    ) != canonical_json_bytes_v2(selected_edit):
        return finish(
            _global_not_proven(
                "SELECTED_EDIT_MISMATCH",
                cumulative_generation_usage=final_usage,
            )
        )
    result_edit = selected_edit if requested_edit is None else requested_edit

    witness = selected.point.witness_loss_bounds
    if witness is None:  # pragma: no cover - point outcome invariant
        raise RuntimeError("selected point omitted witness loss bounds")
    witness_upper = witness.total_upper_bound
    if global_lower > witness_upper:
        raise RuntimeError("fresh global lower bound exceeds its feasible witness")
    try:
        gap = _directed_binary64_gap_ceil(global_lower, witness_upper)
    except (ArithmeticError, RuntimeWarning):
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.NUMERIC_GAP,
                "NUMERIC_GAP:GLOBAL_OPTIMALITY_GAP",
                cumulative_generation_usage=final_usage,
            )
        )
    if gap > bundle.config.target_optimality_gap:
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.NUMERIC_GAP,
                "NUMERIC_GAP:TARGET_OPTIMALITY_GAP_EXCEEDED",
                cumulative_generation_usage=final_usage,
            )
        )
    claim = OptimalityClaimV2.EXACT if gap == 0.0 else OptimalityClaimV2.EPSILON_OPTIMAL
    epsilon = None if claim is OptimalityClaimV2.EXACT else gap

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            certificate = GlobalOptimalityCertificateV2(
                semantic_problem_sha256=bundle.problem.semantic_problem_sha256,
                core_solver_config_sha256=bundle.config.core_solver_config_sha256,
                candidate_domain_artifact_sha256=(
                    bundle.candidate.candidate_domain_artifact_sha256
                ),
                relation_cost_partition_sha256=(
                    bundle.relation.relation_cost_partition_sha256
                ),
                objective_partition_artifact_sha256=(
                    objective.objective_partition_artifact_sha256
                ),
                edit_sha256=result_edit.edit_sha256,
                loss_lower_bound=global_lower,
                loss_upper_bound=witness_upper,
                optimality_gap=gap,
                optimality_claim=claim,
                epsilon=epsilon,
            )
            certificate = GlobalOptimalityCertificateV2.model_validate(
                certificate.model_dump(mode="python"),
                strict=True,
            )
            result = CertifiedSuccessResultV2(
                semantic_problem_sha256=bundle.problem.semantic_problem_sha256,
                core_solver_config=bundle.config,
                candidate_domain=bundle.candidate,
                relation_cost_partition=bundle.relation,
                objective_partition=objective,
                edit=result_edit,
                global_loss_lower_bound=global_lower,
                witness_loss_bounds=witness,
                certificate=certificate,
            )
            result = CertifiedSuccessResultV2.model_validate(
                result.model_dump(mode="python"),
                strict=True,
            )
            _require_global_canonical_round_trip(certificate, result)
    except (ArithmeticError, RuntimeWarning):
        return finish(
            _global_uncertified(
                UncertifiedReasonV2.NUMERIC_GAP,
                "NUMERIC_GAP:GLOBAL_RESULT_CONSTRUCTION",
                cumulative_generation_usage=final_usage,
            )
        )
    if result_edit.subject_id != bundle.candidate.candidate_variable.subject_id:
        raise RuntimeError("final edit subject is not closed to the candidate")

    return finish(
        GlobalOptimalityBuildOutcomeV2(
            kind=GlobalOptimalityBuildKindV2.CERTIFIED_SUCCESS,
            certified_success_result=result,
            semantic_problem_sha256=bundle.problem.semantic_problem_sha256,
            core_solver_config_sha256=bundle.config.core_solver_config_sha256,
            candidate_domain_artifact_sha256=(
                bundle.candidate.candidate_domain_artifact_sha256
            ),
            relation_cost_partition_sha256=(
                bundle.relation.relation_cost_partition_sha256
            ),
            objective_partition_artifact_sha256=(
                objective.objective_partition_artifact_sha256
            ),
            canonical_edit_sha256=result_edit.edit_sha256,
            certificate_sha256=certificate.certificate_sha256,
            solve_result_sha256=result.solve_result_sha256,
            cumulative_generation_usage=final_usage,
        )
    )


def _strict_model(
    value: object,
    model_type: type[ModelT],
    *,
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


def _strict_fresh_internal_model(
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    if type(value) is not model_type:
        raise TypeError(f"fresh {label} has the wrong exact type")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise TypeError(f"fresh {label} failed strict reconstruction") from error


def _validate_global_replay_bundle_prefix(
    bundle: object,
    requested_edit: object,
) -> None:
    if type(bundle) is not _GlobalReplayBundleV2:
        raise TypeError("bundle must be a _GlobalReplayBundleV2")
    if requested_edit is not None and type(requested_edit) is not CanonicalEditV2:
        raise TypeError("requested_edit must be an exact CanonicalEditV2")
    if type(bundle.objective_outcome) is not ObjectivePartitionCompilationOutcomeV2:
        raise TypeError("bundle objective outcome has the wrong type")
    if type(bundle.point_context) is not _PointObjectiveReplayContextV2:
        raise TypeError("bundle point context has the wrong type")
    objective = bundle.objective_outcome.objective_partition
    usage = bundle.objective_outcome.cumulative_resource_usage
    if (
        bundle.objective_outcome.kind
        is not ObjectivePartitionCompilationKindV2.PARTITION
        or objective is None
        or usage is None
    ):
        raise ValueError("fresh global bundle requires a complete objective partition")
    context = bundle.point_context
    if (
        context.problem is not bundle.problem
        or context.config is not bundle.config
        or context.candidate is not bundle.candidate
        or context.relation is not bundle.relation
        or context.objective is not objective
        or context.replay_usage != usage
    ):
        raise ValueError("fresh global bundle/context identity is not closed")
    if (
        context.domain_budget.used != usage.domain_operations
        or context.atomic_budget.used != usage.partition_cells
        or usage.refinement_steps != bundle.candidate.resource_usage.refinement_steps
    ):
        raise ValueError("fresh global bundle ledger prefix is not closed")
    if (
        context.domain_budget.limit != bundle.config.max_domain_operations
        or context.atomic_budget.limit != bundle.config.max_partition_cells
    ):
        raise ValueError("fresh global bundle ledger limits are not closed")


def _validate_global_replay_bundle_closure(
    bundle: _GlobalReplayBundleV2,
    requested_edit: CanonicalEditV2 | None,
) -> None:
    objective = bundle.objective_outcome.objective_partition
    usage = bundle.objective_outcome.cumulative_resource_usage
    if objective is None or usage is None:  # pragma: no cover - prefix invariant
        raise RuntimeError("fresh global bundle lost its objective prefix")
    _require_candidate_closure(bundle.problem, bundle.config, bundle.candidate)
    _require_relation_usage(bundle.config, bundle.candidate, usage)
    _require_relation_closure(
        bundle.problem,
        bundle.config,
        bundle.candidate,
        bundle.relation,
    )
    _require_objective_closure(
        bundle.problem,
        bundle.config,
        bundle.candidate,
        bundle.relation,
        objective,
    )
    if requested_edit is not None and _global_edit_reference_findings(
        bundle.problem,
        requested_edit,
    ):
        raise ValueError("fresh global bundle edit references are not closed")


def _reserve_global_builder_work(
    context: _PointObjectiveReplayContextV2,
    proposal_count: int,
) -> None:
    """Atomically reserve all builder-owned deep passes before inspection.

    Point evaluation reserves its own work separately on the same ledger.  The
    formula below covers structural eligibility, cell/term/proposal closure,
    cell-total lower-bound reduction, exact tie tuples, every ordered winner
    comparison, and final result assembly.  A cap miss therefore exposes no
    inspected-cell or evaluated-proposal prefix.
    """

    if type(proposal_count) is not int or proposal_count < 0:
        raise TypeError("proposal_count must be a non-negative exact int")
    candidate_steps = len(context.candidate.shrink_ledger)
    relation_cells = len(context.relation.cells)
    objective_cells = len(context.objective.cells)
    ordered_comparisons = proposal_count * max(0, proposal_count - 1)
    units = (
        8
        + candidate_steps
        + 2 * relation_cells
        + 8 * objective_cells
        + 8 * proposal_count
        + ordered_comparisons
    )
    context.domain_budget.consume(units)


def _global_eligibility_findings(
    bundle: _GlobalReplayBundleV2,
    proposals: tuple[ObjectiveWitnessProposalV2, ...],
) -> tuple[str, ...]:
    findings: list[str] = []
    if not bundle.candidate.is_global_verification_eligible:
        findings.append("GLOBAL_CANDIDATE_NOT_ELIGIBLE")
    if not bundle.relation.is_global_verification_eligible:
        findings.append("GLOBAL_RELATION_NOT_ELIGIBLE")
    objective = bundle.point_context.objective
    if not objective.is_global_verification_eligible:
        findings.append("GLOBAL_OBJECTIVE_NOT_ELIGIBLE")
    if not objective.cells:
        findings.append("GLOBAL_OBJECTIVE_HAS_NO_CELLS")

    cell_ids: list[str] = []
    relation_by_cell: dict[str, str] = {}
    for cell in objective.cells:
        cell_ids.append(cell.cell_id)
        relation_by_cell[cell.cell_id] = cell.parent_relation_cell_id
        domain = cell.domain
        if domain.completeness is not DomainCompletenessV2.EXACT:
            findings.append(f"GLOBAL_CELL_DOMAIN_NOT_EXACT:{cell.cell_id}")
        if (
            domain.inner_bound.status is not RegionBoundStatusV2.NON_EMPTY
            or domain.outer_bound.status is not RegionBoundStatusV2.NON_EMPTY
            or domain.inner_bound != domain.outer_bound
        ):
            findings.append(f"GLOBAL_CELL_DOMAIN_NOT_EXACT_NONEMPTY:{cell.cell_id}")
        bounds = cell.term_loss_bounds
        if any(
            interval.lower_bound != interval.upper_bound
            for interval in (
                bounds.relation_damage_loss,
                bounds.visibility_change_loss,
                bounds.safety_margin_loss,
            )
        ):
            findings.append(f"GLOBAL_CELL_NONTRANSLATION_NOT_CONSTANT:{cell.cell_id}")

    proposal_ids = tuple(item.objective_cell_id for item in proposals)
    if len(proposal_ids) != len(cell_ids) or set(proposal_ids) != set(cell_ids):
        findings.append("GLOBAL_PROPOSAL_COVERAGE_NOT_EXACT")
    for proposal in proposals:
        expected_parent = relation_by_cell.get(proposal.objective_cell_id)
        if expected_parent != proposal.parent_relation_cell_id:
            raise RuntimeError("fresh objective proposal parent reference drift")
        if (
            proposal.edit.semantic_problem_sha256
            != bundle.problem.semantic_problem_sha256
        ):
            raise RuntimeError("fresh objective proposal problem hash drift")
        if proposal.edit.subject_id != bundle.candidate.candidate_variable.subject_id:
            raise RuntimeError("fresh objective proposal subject drift")
    return tuple(sorted(set(findings)))


def _global_loss_lower_bound_v2(
    cells: tuple[ObjectivePartitionCellV2, ...],
) -> float:
    """Reduce complete cell totals; never mix component minima across cells."""

    if (
        type(cells) is not tuple
        or not cells
        or any(type(cell) is not ObjectivePartitionCellV2 for cell in cells)
    ):
        raise ValueError("global loss lower bound requires objective cells")
    return min(cell.term_loss_bounds.total_lower_bound for cell in cells)


def _require_global_point_closure(
    bundle: _GlobalReplayBundleV2,
    proposal: ObjectiveWitnessProposalV2,
    point: PointObjectiveEvaluationOutcomeV2,
) -> None:
    if point.kind is not PointObjectiveEvaluationKindV2.BOUNDED_FEASIBLE:
        raise RuntimeError("global point closure requires a bounded feasible outcome")
    expected = (
        bundle.problem.semantic_problem_sha256,
        bundle.config.core_solver_config_sha256,
        bundle.candidate.candidate_domain_artifact_sha256,
        bundle.relation.relation_cost_partition_sha256,
        bundle.point_context.objective.objective_partition_artifact_sha256,
        proposal.edit.edit_sha256,
    )
    actual = (
        point.semantic_problem_sha256,
        point.core_solver_config_sha256,
        point.candidate_domain_artifact_sha256,
        point.relation_cost_partition_sha256,
        point.objective_partition_artifact_sha256,
        point.canonical_edit_sha256,
    )
    if actual != expected:
        raise RuntimeError("fresh global point reference closure drift")
    if point.cumulative_generation_usage != _context_usage(bundle.point_context):
        raise RuntimeError("fresh global point cumulative ledger drift")
    if proposal.objective_cell_id not in point.covering_objective_cell_ids:
        raise RuntimeError("fresh proposal escaped its objective cell")
    if proposal.parent_relation_cell_id not in point.covering_relation_cell_ids:
        raise RuntimeError("fresh proposal escaped its relation cell")


def _singleton_term_values(
    bounds: ObjectiveTermBoundsV2,
) -> tuple[Fraction, Fraction, Fraction, Fraction] | None:
    intervals = (
        bounds.translation_loss,
        bounds.relation_damage_loss,
        bounds.visibility_change_loss,
        bounds.safety_margin_loss,
    )
    if any(item.lower_bound != item.upper_bound for item in intervals):
        return None
    return tuple(Fraction.from_float(item.lower_bound) for item in intervals)  # type: ignore[return-value]


def _provably_precedes_global(
    left: _EvaluatedGlobalProposalV2,
    right: _EvaluatedGlobalProposalV2,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> bool:
    left_bounds = left.point.witness_loss_bounds
    right_bounds = right.point.witness_loss_bounds
    if left_bounds is None or right_bounds is None:
        raise RuntimeError("evaluated global proposal omitted point bounds")
    if left_bounds.total_upper_bound < right_bounds.total_lower_bound:
        return True
    left_nontranslation = _singleton_nontranslation_values(left_bounds)
    right_nontranslation = _singleton_nontranslation_values(right_bounds)
    if left_nontranslation is not None and left_nontranslation == right_nontranslation:
        left_edit = left.proposal.edit.translation_xy_m
        right_edit = right.proposal.edit.translation_xy_m
        left_x, left_y = selection_frame.semantic_xy(
            Fraction.from_float(left_edit.x),
            Fraction.from_float(left_edit.y),
        )
        right_x, right_y = selection_frame.semantic_xy(
            Fraction.from_float(right_edit.x),
            Fraction.from_float(right_edit.y),
        )
        left_squared = left_x**2 + left_y**2
        right_squared = right_x**2 + right_y**2
        if left_squared != right_squared:
            return left_squared < right_squared
        return (left_x, left_y) < (right_x, right_y)
    if (
        left.singleton_values is None
        or right.singleton_values is None
        or left.singleton_total is None
        or right.singleton_total is None
    ):
        return False
    if left.singleton_total != right.singleton_total:
        return left.singleton_total < right.singleton_total
    left_edit = left.proposal.edit.translation_xy_m
    right_edit = right.proposal.edit.translation_xy_m
    left_x, left_y = selection_frame.semantic_xy(
        Fraction.from_float(left_edit.x),
        Fraction.from_float(left_edit.y),
    )
    right_x, right_y = selection_frame.semantic_xy(
        Fraction.from_float(right_edit.x),
        Fraction.from_float(right_edit.y),
    )
    left_key = (
        *left.singleton_values,
        left_x,
        left_y,
    )
    right_key = (
        *right.singleton_values,
        right_x,
        right_y,
    )
    return left_key < right_key


def _singleton_nontranslation_values(
    bounds: ObjectiveTermBoundsV2,
) -> tuple[Fraction, Fraction, Fraction] | None:
    intervals = (
        bounds.relation_damage_loss,
        bounds.visibility_change_loss,
        bounds.safety_margin_loss,
    )
    if any(item.lower_bound != item.upper_bound for item in intervals):
        return None
    return tuple(Fraction.from_float(item.lower_bound) for item in intervals)  # type: ignore[return-value]


def _global_point_semantic_signature(
    point: PointObjectiveEvaluationOutcomeV2,
) -> tuple[object, ...]:
    return (
        point.semantic_problem_sha256,
        point.core_solver_config_sha256,
        point.candidate_domain_artifact_sha256,
        point.relation_cost_partition_sha256,
        point.objective_partition_artifact_sha256,
        point.canonical_edit_sha256,
        point.witness_loss_bounds,
        point.relation_damage_vector,
        point.constraint_slacks,
        point.covering_objective_cell_ids,
        point.covering_relation_cell_ids,
    )


def _select_certifiable_global_winner(
    proposals: tuple[_EvaluatedGlobalProposalV2, ...],
    *,
    selection_frame: _ExactCardinalSelectionFrameV2 | None = None,
) -> _EvaluatedGlobalProposalV2 | None:
    if selection_frame is None:
        selection_frame = _ExactCardinalSelectionFrameV2()
    elif type(selection_frame) is not _ExactCardinalSelectionFrameV2:
        raise TypeError("selection_frame has the wrong exact type")
    groups: dict[tuple[str, bytes], list[_EvaluatedGlobalProposalV2]] = {}
    for proposal in proposals:
        edit = proposal.proposal.edit
        groups.setdefault(
            (edit.edit_sha256, canonical_json_bytes_v2(edit)),
            [],
        ).append(proposal)

    representatives: list[_EvaluatedGlobalProposalV2] = []
    for group in groups.values():
        signature = _global_point_semantic_signature(group[0].point)
        if any(
            _global_point_semantic_signature(item.point) != signature
            for item in group[1:]
        ):
            raise RuntimeError(
                "same fresh proposal edit produced different semantic point evidence"
            )
        representatives.append(
            min(
                group,
                key=lambda item: (
                    item.proposal.objective_cell_id,
                    item.proposal.parent_relation_cell_id,
                ),
            )
        )

    winners = tuple(
        left
        for left in representatives
        if all(
            left is right or _provably_precedes_global(left, right, selection_frame)
            for right in representatives
        )
    )
    return winners[0] if len(winners) == 1 else None


def _global_edit_reference_findings(
    problem: SemanticProblemV2,
    edit: CanonicalEditV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if edit.semantic_problem_sha256 != problem.semantic_problem_sha256:
        findings.append("EDIT_REFERENCE_MISMATCH:SEMANTIC_PROBLEM_HASH")
    if edit.subject_id != problem.constraints.allowed_edit.subject_id:
        findings.append("EDIT_REFERENCE_MISMATCH:SUBJECT_ID")
    return tuple(findings)


def _global_objective_usage(
    base: CompilationResourceUsageV2,
    domain_budget: _ObjectiveDomainOperationBudgetV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> CompilationResourceUsageV2:
    return CompilationResourceUsageV2(
        domain_operations=base.domain_operations + domain_budget.used,
        partition_cells=atomic_budget.used,
        refinement_steps=base.refinement_steps,
    )


def _require_candidate_reference_closure(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
) -> None:
    if candidate.semantic_problem_sha256 != problem.semantic_problem_sha256:
        raise RuntimeError("candidate replay problem hash is not closed")
    if candidate.core_solver_config_sha256 != config.core_solver_config_sha256:
        raise RuntimeError("candidate replay config hash is not closed")
    if candidate.candidate_variable.subject_id != (
        problem.constraints.allowed_edit.subject_id
    ):
        raise RuntimeError("candidate replay subject is not closed")


def _unsat_eligibility_findings(
    candidate: CandidateDomainArtifactV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if candidate.hard_domain.outer_bound.status is not RegionBoundStatusV2.EMPTY:
        findings.append("UNSAT_NOT_PROVEN:NON_EMPTY_HARD_OUTER")
    if any(
        step.disposition is not ConstraintCompilationDispositionV2.APPLIED
        for step in candidate.shrink_ledger
    ):
        findings.append("UNSAT_NOT_PROVEN:UNKNOWN_BEFORE_EMPTY")
    if not candidate.is_unsat_verification_eligible and not findings:
        findings.append("UNSAT_NOT_PROVEN:CANDIDATE_NOT_ELIGIBLE")
    return tuple(sorted(set(findings)))


def _require_canonical_round_trip(
    certificate: ProvenUnsatCertificateV2,
    result: ProvenUnsatResultV2,
) -> None:
    certificate_bytes = canonical_json_bytes_v2(certificate)
    result_bytes = canonical_json_bytes_v2(result)
    restored_certificate = ProvenUnsatCertificateV2.model_validate_json(
        certificate_bytes,
        strict=True,
    )
    restored_result = ProvenUnsatResultV2.model_validate_json(
        result_bytes,
        strict=True,
    )
    if (
        restored_certificate != certificate
        or restored_result != result
        or canonical_json_bytes_v2(restored_certificate) != certificate_bytes
        or canonical_json_bytes_v2(restored_result) != result_bytes
        or restored_certificate.certificate_sha256 != certificate.certificate_sha256
        or restored_result.solve_result_sha256 != result.solve_result_sha256
        or restored_result.certificate != restored_certificate
    ):
        raise RuntimeError("PROVEN_UNSAT canonical round-trip closure drift")


def _require_global_canonical_round_trip(
    certificate: GlobalOptimalityCertificateV2,
    result: CertifiedSuccessResultV2,
) -> None:
    certificate_bytes = canonical_json_bytes_v2(certificate)
    result_bytes = canonical_json_bytes_v2(result)
    restored_certificate = GlobalOptimalityCertificateV2.model_validate_json(
        certificate_bytes,
        strict=True,
    )
    restored_result = CertifiedSuccessResultV2.model_validate_json(
        result_bytes,
        strict=True,
    )
    if (
        restored_certificate != certificate
        or restored_result != result
        or canonical_json_bytes_v2(restored_certificate) != certificate_bytes
        or canonical_json_bytes_v2(restored_result) != result_bytes
        or restored_certificate.certificate_sha256 != certificate.certificate_sha256
        or restored_result.solve_result_sha256 != result.solve_result_sha256
        or restored_result.certificate != restored_certificate
        or restored_result.edit.subject_id
        != restored_result.candidate_domain.candidate_variable.subject_id
    ):
        raise RuntimeError("CERTIFIED_SUCCESS canonical round-trip closure drift")


def _global_uncertified(
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None,
) -> GlobalOptimalityBuildOutcomeV2:
    return GlobalOptimalityBuildOutcomeV2(
        kind=GlobalOptimalityBuildKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=tuple(finding_codes),
        cumulative_generation_usage=cumulative_generation_usage,
    )


def _global_not_proven(
    *finding_codes: str,
    cumulative_generation_usage: CompilationResourceUsageV2 | None = None,
) -> GlobalOptimalityBuildOutcomeV2:
    return GlobalOptimalityBuildOutcomeV2(
        kind=GlobalOptimalityBuildKindV2.NOT_PROVEN,
        finding_codes=tuple(finding_codes),
        cumulative_generation_usage=cumulative_generation_usage,
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    verification_resource_usage: CompilationResourceUsageV2 | None = None,
) -> CertificateBuildOutcomeV2:
    return CertificateBuildOutcomeV2(
        kind=CertificateBuildKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=finding_codes,
        verification_resource_usage=verification_resource_usage,
    )
