"""Strict domain-only wire contracts for the continuous-yaw solver.

The records in this module are structural data, not replay capabilities.  A
consumer must pass a submitted result to the pure-core v2.8 verifier before
trusting its stage references, resource telemetry, witness, or certificate.
"""

from __future__ import annotations

import math
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from spatialcf.domain.v2.artifacts import (
    ConstraintSlackV2,
    ObjectiveTermBoundsV2,
)
from spatialcf.domain.v2.base import (
    CanonicalId,
    NonNegativeFiniteFloat,
    PositiveFiniteFloat,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.certificate import OptimalityClaimV2
from spatialcf.domain.v2.continuous_yaw_candidate import (
    SchemaIdentityV2_2,
    StrictConvexCandidateCompilerConfigV2_7,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.result import UncertifiedReasonV2
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_CONFIG_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-solver-config.v2.8"
_CANDIDATE_REFS_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-candidate-refs.v2.8"
_OBJECTIVE_CELLS_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-objective-cells.v2.8"
_WITNESS_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-witness-evaluation.v2.8"
_CERTIFICATE_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-certificate.v2.8"
_SOLVE_RESULT_HASH_DOMAIN_V2_8 = "spatialcf.continuous-yaw-solve-result.v2.8"
_MAX_DETERMINISTIC_LIMIT_V2_8 = 2**63 - 1

PositiveDeterministicLimitV2_8 = Annotated[
    int,
    Field(strict=True, ge=1, le=_MAX_DETERMINISTIC_LIMIT_V2_8),
]
NonNegativeDeterministicLimitV2_8 = Annotated[
    int,
    Field(strict=True, ge=0, le=_MAX_DETERMINISTIC_LIMIT_V2_8),
]


class ContinuousYawSolveStatusV2(StrEnum):
    CERTIFIED_SUCCESS = "CERTIFIED_SUCCESS"
    PROVEN_UNSAT = "PROVEN_UNSAT"
    UNCERTIFIED = "UNCERTIFIED"


class ContinuousYawSolveVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


def _canonical_findings(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or not value.strip() for value in values
    ):
        raise TypeError("finding_codes must be exact non-blank strings")
    return tuple(sorted(set(values)))


def _directed_binary64_gap_ceil(lower_bound: float, upper_bound: float) -> float:
    exact = Fraction.from_float(upper_bound) - Fraction.from_float(lower_bound)
    try:
        published = float(exact)
    except OverflowError as error:
        raise ValueError("optimality gap must be finite binary64") from error
    if not math.isfinite(published):
        raise ValueError("optimality gap must be finite binary64")
    if Fraction.from_float(published) < exact:
        published = math.nextafter(published, math.inf)
    if not math.isfinite(published):
        raise ValueError("optimality gap must be finite binary64")
    return published


class ContinuousYawResourceUsageV2_8(V2Model):
    """One cumulative generation ledger for the complete v2.8 solve."""

    domain_operations: NonNegativeDeterministicLimitV2_8
    so2_atomic_steps: NonNegativeDeterministicLimitV2_8
    candidate_cells: NonNegativeDeterministicLimitV2_8
    objective_partition_cells: NonNegativeDeterministicLimitV2_8
    branch_nodes: Literal[0] = 0
    refinement_steps: Literal[0] = 0


class ContinuousYawSolverConfigV2_8(V2Model):
    """Closed public policy for the complete continuous-yaw solve."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(
            schema_name="continuous-yaw-solver-config"
        )
    )
    algorithm_id: Literal["solver:canonical-branch-and-bound-v2"] = (
        "solver:canonical-branch-and-bound-v2"
    )
    algorithm_version: Literal["algorithm:2.8"] = "algorithm:2.8"
    candidate_config: StrictConvexCandidateCompilerConfigV2_7
    target_projection_kernel_id: Literal[
        "geometry-kernel:rational-continuous-yaw-shape-gap-v2"
    ] = "geometry-kernel:rational-continuous-yaw-shape-gap-v2"
    visibility_projection_kernel_id: Literal[
        "geometry-kernel:rational-continuous-yaw-fixed-camera-visibility-v2"
    ] = "geometry-kernel:rational-continuous-yaw-fixed-camera-visibility-v2"
    objective_kernel_id: Literal[
        "objective-kernel:rational-continuous-yaw-cell-bounds-v2"
    ] = "objective-kernel:rational-continuous-yaw-cell-bounds-v2"
    max_objective_partition_cells: PositiveDeterministicLimitV2_8
    max_branch_nodes: Literal[0] = 0
    max_refinement_steps: Literal[0] = 0
    target_optimality_gap: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = SchemaIdentityV2_2(schema_name="continuous-yaw-solver-config")
        if self.schema_identity != expected:
            raise ValueError("continuous-yaw solver config identity must be fixed")
        return self

    @property
    def config_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_CONFIG_HASH_DOMAIN_V2_8)


class ContinuousYawCandidateRefsV2_8(V2Model):
    """Hash chain for the private T15, target, and visibility stages."""

    semantic_problem_sha256: Sha256Digest
    solver_config_sha256: Sha256Digest
    t15_candidate_artifact_sha256: Sha256Digest
    target_candidate_stage_sha256: Sha256Digest | None = None
    visibility_candidate_stage_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_prefix(self) -> Self:
        if (
            self.visibility_candidate_stage_sha256 is not None
            and self.target_candidate_stage_sha256 is None
        ):
            raise ValueError("visibility stage requires a target stage reference")
        return self

    @property
    def candidate_refs_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_CANDIDATE_REFS_HASH_DOMAIN_V2_8)


class ContinuousYawObjectiveCellV2_8(V2Model):
    """One objective enclosure over a private strict-convex candidate cell."""

    _SEQUENCE_HASH_DOMAIN: ClassVar[str] = _OBJECTIVE_CELLS_HASH_DOMAIN_V2_8

    cell_id: CanonicalId
    outer_domain_sha256: Sha256Digest
    inner_domain_sha256: Sha256Digest | None = None
    term_loss_bounds: ObjectiveTermBoundsV2
    constraint_slacks: tuple[ConstraintSlackV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_slacks(self) -> Self:
        slack_ids = tuple(slack.constraint_id for slack in self.constraint_slacks)
        if len(slack_ids) != len(set(slack_ids)):
            raise ValueError("objective cell constraint slacks must be unique")
        object.__setattr__(
            self,
            "constraint_slacks",
            tuple(sorted(self.constraint_slacks, key=lambda item: item.constraint_id)),
        )
        return self

    @classmethod
    def sequence_sha256(
        cls,
        cells: tuple[ContinuousYawObjectiveCellV2_8, ...],
    ) -> Sha256Digest:
        if type(cells) is not tuple:
            raise TypeError("objective cells must be an exact tuple")
        checked = tuple(
            cls.model_validate(cell.model_dump(mode="python"), strict=True)
            for cell in cells
        )
        ordered = tuple(sorted(checked, key=lambda cell: cell.cell_id))
        cell_ids = tuple(cell.cell_id for cell in ordered)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("objective cell IDs must be unique")
        return canonical_sha256_v2(ordered, domain=cls._SEQUENCE_HASH_DOMAIN)


class ContinuousYawWitnessEvaluationV2_8(V2Model):
    """Fresh point loss and slack enclosure for one concrete edit."""

    objective_cell_id: CanonicalId
    edit: CanonicalEditV2
    witness_loss_bounds: ObjectiveTermBoundsV2
    constraint_slacks: tuple[ConstraintSlackV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_slacks(self) -> Self:
        slack_ids = tuple(slack.constraint_id for slack in self.constraint_slacks)
        if len(slack_ids) != len(set(slack_ids)):
            raise ValueError("witness constraint slacks must be unique")
        object.__setattr__(
            self,
            "constraint_slacks",
            tuple(sorted(self.constraint_slacks, key=lambda item: item.constraint_id)),
        )
        return self

    @property
    def witness_evaluation_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_WITNESS_HASH_DOMAIN_V2_8)


class ContinuousYawGlobalOptimalityCertificateV2_8(V2Model):
    """Directed scalar optimum claim over the complete v2.8 candidate chain."""

    semantic_problem_sha256: Sha256Digest
    solver_config_sha256: Sha256Digest
    candidate_refs_sha256: Sha256Digest
    objective_cells_sha256: Sha256Digest
    witness_evaluation_sha256: Sha256Digest
    edit_sha256: Sha256Digest
    loss_lower_bound: NonNegativeFiniteFloat
    loss_upper_bound: NonNegativeFiniteFloat
    optimality_gap: NonNegativeFiniteFloat
    optimality_claim: OptimalityClaimV2
    epsilon: PositiveFiniteFloat | None = None
    final_resource_usage: ContinuousYawResourceUsageV2_8

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if self.loss_lower_bound > self.loss_upper_bound:
            raise ValueError("loss lower bound cannot exceed loss upper bound")
        expected_gap = _directed_binary64_gap_ceil(
            self.loss_lower_bound,
            self.loss_upper_bound,
        )
        if self.optimality_gap != expected_gap:
            raise ValueError("optimality gap must be the directed exact difference")
        if self.optimality_claim is OptimalityClaimV2.EXACT:
            if self.optimality_gap != 0.0 or self.epsilon is not None:
                raise ValueError("EXACT requires zero gap and no epsilon")
            return self
        if self.epsilon is None or self.optimality_gap > self.epsilon:
            raise ValueError("EPSILON_OPTIMAL requires epsilon covering the gap")
        return self

    @property
    def certificate_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_CERTIFICATE_HASH_DOMAIN_V2_8)


def _canonical_objective_cells(
    cells: tuple[ContinuousYawObjectiveCellV2_8, ...],
) -> tuple[ContinuousYawObjectiveCellV2_8, ...]:
    ordered = tuple(sorted(cells, key=lambda cell: cell.cell_id))
    cell_ids = tuple(cell.cell_id for cell in ordered)
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("objective cell IDs must be unique")
    return ordered


def _validate_usage_limits(
    usage: ContinuousYawResourceUsageV2_8,
    config: ContinuousYawSolverConfigV2_8,
) -> None:
    candidate_config = config.candidate_config
    if usage.domain_operations > candidate_config.max_domain_operations:
        raise ValueError("resource usage exceeds the domain-operation limit")
    if usage.so2_atomic_steps > candidate_config.max_so2_atomic_steps:
        raise ValueError("resource usage exceeds the SO(2) atomic-step limit")
    if usage.candidate_cells > candidate_config.max_candidate_cells:
        raise ValueError("resource usage exceeds the candidate-cell limit")
    if usage.objective_partition_cells > config.max_objective_partition_cells:
        raise ValueError("resource usage exceeds the objective-cell limit")


class _ContinuousYawSolveResultBaseV2_8(V2Model):
    semantic_problem_sha256: Sha256Digest
    solver_config: ContinuousYawSolverConfigV2_8

    @property
    def solve_result_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_SOLVE_RESULT_HASH_DOMAIN_V2_8)


class ContinuousYawCertifiedSuccessResultV2_8(_ContinuousYawSolveResultBaseV2_8):
    status: Literal[ContinuousYawSolveStatusV2.CERTIFIED_SUCCESS] = (
        ContinuousYawSolveStatusV2.CERTIFIED_SUCCESS
    )
    candidate_refs: ContinuousYawCandidateRefsV2_8
    objective_cells: tuple[ContinuousYawObjectiveCellV2_8, ...] = Field(min_length=1)
    selected_witness: ContinuousYawWitnessEvaluationV2_8
    global_loss_lower_bound: NonNegativeFiniteFloat
    witness_loss_bounds: ObjectiveTermBoundsV2
    final_resource_usage: ContinuousYawResourceUsageV2_8
    certificate: ContinuousYawGlobalOptimalityCertificateV2_8

    @model_validator(mode="after")
    def validate_success(self) -> Self:
        refs = self.candidate_refs
        config_hash = self.solver_config.config_sha256
        if refs.semantic_problem_sha256 != self.semantic_problem_sha256:
            raise ValueError("candidate refs problem hash is not closed")
        if refs.solver_config_sha256 != config_hash:
            raise ValueError("candidate refs config hash is not closed")
        if (
            refs.target_candidate_stage_sha256 is None
            or refs.visibility_candidate_stage_sha256 is None
        ):
            raise ValueError("success requires the complete candidate stage chain")
        cells = _canonical_objective_cells(self.objective_cells)
        object.__setattr__(self, "objective_cells", cells)
        selected_cells = tuple(
            cell
            for cell in cells
            if cell.cell_id == self.selected_witness.objective_cell_id
        )
        if len(selected_cells) != 1 or selected_cells[0].inner_domain_sha256 is None:
            raise ValueError("selected witness requires one objective inner cell")
        if self.selected_witness.edit.semantic_problem_sha256 != (
            self.semantic_problem_sha256
        ):
            raise ValueError("selected edit problem hash is not closed")
        if self.selected_witness.witness_loss_bounds != self.witness_loss_bounds:
            raise ValueError("selected witness loss bounds are not closed")
        if self.global_loss_lower_bound > self.witness_loss_bounds.total_upper_bound:
            raise ValueError("global lower bound exceeds witness upper bound")
        _validate_usage_limits(self.final_resource_usage, self.solver_config)
        if self.final_resource_usage.objective_partition_cells != len(cells):
            raise ValueError("objective-cell usage must equal published cells")

        certificate = self.certificate
        expected = (
            self.semantic_problem_sha256,
            config_hash,
            refs.candidate_refs_sha256,
            ContinuousYawObjectiveCellV2_8.sequence_sha256(cells),
            self.selected_witness.witness_evaluation_sha256,
            self.selected_witness.edit.edit_sha256,
        )
        actual = (
            certificate.semantic_problem_sha256,
            certificate.solver_config_sha256,
            certificate.candidate_refs_sha256,
            certificate.objective_cells_sha256,
            certificate.witness_evaluation_sha256,
            certificate.edit_sha256,
        )
        if actual != expected:
            raise ValueError("success certificate references are not closed")
        if certificate.loss_lower_bound != self.global_loss_lower_bound:
            raise ValueError("certificate lower bound is not closed")
        if certificate.loss_upper_bound != self.witness_loss_bounds.total_upper_bound:
            raise ValueError("certificate upper bound is not closed")
        if certificate.final_resource_usage != self.final_resource_usage:
            raise ValueError("certificate resource usage is not closed")
        if certificate.optimality_gap > self.solver_config.target_optimality_gap:
            raise ValueError("certificate exceeds the configured optimality gap")
        return self


class ContinuousYawProvenUnsatResultV2_8(_ContinuousYawSolveResultBaseV2_8):
    status: Literal[ContinuousYawSolveStatusV2.PROVEN_UNSAT] = (
        ContinuousYawSolveStatusV2.PROVEN_UNSAT
    )
    candidate_refs: ContinuousYawCandidateRefsV2_8
    empty_outer_stage: Literal["T15", "TARGET_RELATION", "VISIBILITY"]
    empty_outer_stage_sha256: Sha256Digest
    final_resource_usage: ContinuousYawResourceUsageV2_8

    @model_validator(mode="after")
    def validate_unsat(self) -> Self:
        refs = self.candidate_refs
        if refs.semantic_problem_sha256 != self.semantic_problem_sha256:
            raise ValueError("candidate refs problem hash is not closed")
        if refs.solver_config_sha256 != self.solver_config.config_sha256:
            raise ValueError("candidate refs config hash is not closed")
        expected_stage_sha = {
            "T15": refs.t15_candidate_artifact_sha256,
            "TARGET_RELATION": refs.target_candidate_stage_sha256,
            "VISIBILITY": refs.visibility_candidate_stage_sha256,
        }[self.empty_outer_stage]
        if (
            expected_stage_sha is None
            or self.empty_outer_stage_sha256 != expected_stage_sha
        ):
            raise ValueError("empty-outer stage hash is not closed")
        if self.final_resource_usage.objective_partition_cells != 0:
            raise ValueError("UNSAT cannot publish objective cells")
        _validate_usage_limits(self.final_resource_usage, self.solver_config)
        return self


class ContinuousYawUncertifiedResultV2_8(_ContinuousYawSolveResultBaseV2_8):
    status: Literal[ContinuousYawSolveStatusV2.UNCERTIFIED] = (
        ContinuousYawSolveStatusV2.UNCERTIFIED
    )
    uncertified_reason: UncertifiedReasonV2
    candidate_refs: ContinuousYawCandidateRefsV2_8 | None = None
    objective_cells: tuple[ContinuousYawObjectiveCellV2_8, ...] = ()
    final_resource_usage: ContinuousYawResourceUsageV2_8 | None = None
    finding_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uncertified(self) -> Self:
        findings = _canonical_findings(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        refs = self.candidate_refs
        if refs is not None and (
            refs.semantic_problem_sha256 != self.semantic_problem_sha256
            or refs.solver_config_sha256 != self.solver_config.config_sha256
        ):
            raise ValueError("uncertified candidate refs are not closed")
        cells = _canonical_objective_cells(self.objective_cells)
        object.__setattr__(self, "objective_cells", cells)
        if cells and refs is None:
            raise ValueError("objective cells require candidate stage references")
        usage = self.final_resource_usage
        if usage is not None:
            _validate_usage_limits(usage, self.solver_config)
            if usage.objective_partition_cells < len(cells):
                raise ValueError("resource usage undercounts objective cells")
        elif refs is not None or cells:
            raise ValueError("artifact-bearing uncertified result requires usage")
        return self


ContinuousYawSolveResultV2_8: TypeAlias = Annotated[
    ContinuousYawCertifiedSuccessResultV2_8
    | ContinuousYawProvenUnsatResultV2_8
    | ContinuousYawUncertifiedResultV2_8,
    Field(discriminator="status"),
]


class ContinuousYawMinimumCostSolveOutcomeV2_8(V2Model):
    """Fresh-solve result plus exact non-capability generation telemetry."""

    result: ContinuousYawSolveResultV2_8 | None
    finding_codes: tuple[str, ...] = ()
    cumulative_generation_usage: ContinuousYawResourceUsageV2_8 | None = None
    proposal_count: NonNegativeDeterministicLimitV2_8 = 0
    evaluated_proposal_count: NonNegativeDeterministicLimitV2_8 = 0

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        findings = _canonical_findings(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.evaluated_proposal_count > self.proposal_count:
            raise ValueError("evaluated proposal count cannot exceed proposal count")
        result = self.result
        usage = self.cumulative_generation_usage
        if result is None:
            if not findings:
                raise ValueError("missing result requires an input finding")
            if (
                usage is not None
                or self.proposal_count
                or self.evaluated_proposal_count
            ):
                raise ValueError("missing result cannot carry generation telemetry")
            return self
        if usage != result.final_resource_usage:
            raise ValueError("outcome usage must equal result usage")
        if type(result) is ContinuousYawCertifiedSuccessResultV2_8:
            if findings:
                raise ValueError("certified success cannot carry findings")
            if self.proposal_count != len(result.objective_cells):
                raise ValueError("success proposal count must equal objective cells")
            if self.evaluated_proposal_count != self.proposal_count:
                raise ValueError("success must evaluate every proposal")
            return self
        if type(result) is ContinuousYawProvenUnsatResultV2_8:
            if findings or self.proposal_count or self.evaluated_proposal_count:
                raise ValueError("proven UNSAT cannot carry findings or proposals")
            return self
        if findings != result.finding_codes:
            raise ValueError("uncertified findings must equal result findings")
        if self.proposal_count > len(result.objective_cells):
            raise ValueError("proposal count exceeds published objective cells")
        return self


class ContinuousYawSolveVerificationOutcomeV2_8(V2Model):
    """Fresh replay comparison outcome; verified refs exist only on a match."""

    kind: ContinuousYawSolveVerificationKindV2
    semantic_problem_sha256: Sha256Digest | None = None
    solver_config_sha256: Sha256Digest | None = None
    submitted_solve_result_sha256: Sha256Digest | None = None
    verified_status: ContinuousYawSolveStatusV2 | None = None
    replay_generation_usage: ContinuousYawResourceUsageV2_8 | None = None
    proposal_count: NonNegativeDeterministicLimitV2_8 = 0
    evaluated_proposal_count: NonNegativeDeterministicLimitV2_8 = 0
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_verification(self) -> Self:
        findings = _canonical_findings(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.evaluated_proposal_count > self.proposal_count:
            raise ValueError("evaluated proposal count cannot exceed proposal count")
        if self.replay_generation_usage is None and (
            self.proposal_count or self.evaluated_proposal_count
        ):
            raise ValueError("proposal telemetry requires replay usage")
        refs = (
            self.semantic_problem_sha256,
            self.solver_config_sha256,
            self.submitted_solve_result_sha256,
        )
        if self.kind is ContinuousYawSolveVerificationKindV2.VERIFIED:
            if any(value is None for value in refs):
                raise ValueError("VERIFIED requires all replay references")
            if self.verified_status is None:
                raise ValueError("VERIFIED requires the matched solve status")
            if self.uncertified_reason is not None or findings:
                raise ValueError("VERIFIED cannot carry failure diagnostics")
            if self.verified_status is ContinuousYawSolveStatusV2.CERTIFIED_SUCCESS:
                if (
                    self.replay_generation_usage is None
                    or self.proposal_count < 1
                    or self.evaluated_proposal_count != self.proposal_count
                ):
                    raise ValueError(
                        "verified success requires full proposal telemetry"
                    )
            elif self.verified_status is ContinuousYawSolveStatusV2.PROVEN_UNSAT and (
                self.replay_generation_usage is None
                or self.proposal_count
                or self.evaluated_proposal_count
            ):
                raise ValueError("verified UNSAT requires zero-proposal telemetry")
            return self
        if any(value is not None for value in refs) or self.verified_status is not None:
            raise ValueError("failure verification cannot carry verified refs")
        if not findings:
            raise ValueError("failure verification requires at least one finding")
        if self.kind is ContinuousYawSolveVerificationKindV2.MISMATCH:
            if self.uncertified_reason is not None:
                raise ValueError("MISMATCH cannot carry an uncertified reason")
            return self
        if self.uncertified_reason is None:
            raise ValueError("UNCERTIFIED verification requires a reason")
        return self


__all__ = (
    "ContinuousYawCandidateRefsV2_8",
    "ContinuousYawCertifiedSuccessResultV2_8",
    "ContinuousYawGlobalOptimalityCertificateV2_8",
    "ContinuousYawMinimumCostSolveOutcomeV2_8",
    "ContinuousYawObjectiveCellV2_8",
    "ContinuousYawProvenUnsatResultV2_8",
    "ContinuousYawResourceUsageV2_8",
    "ContinuousYawSolveResultV2_8",
    "ContinuousYawSolveStatusV2",
    "ContinuousYawSolveVerificationKindV2",
    "ContinuousYawSolveVerificationOutcomeV2_8",
    "ContinuousYawSolverConfigV2_8",
    "ContinuousYawUncertifiedResultV2_8",
    "ContinuousYawWitnessEvaluationV2_8",
)
