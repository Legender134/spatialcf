"""Canonical v2 deterministic core configuration and solve-result contracts.

The result state machine closes all in-record hash references.  Its structural
eligibility checks are necessary inputs to certification, not a replacement
for the independent pure-core verifier that recomputes region unions, witness
feasibility, objective bounds, and empty-outer proofs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BeforeValidator, Field, model_validator

from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    ObjectivePartitionArtifactV2,
    ObjectiveTermBoundsV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import (
    CanonicalId,
    NonNegativeFiniteFloat,
    SchemaIdentityV2,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.certificate import (
    GlobalOptimalityCertificateV2,
    ProvenUnsatCertificateV2,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_CORE_SOLVER_CONFIG_HASH_DOMAIN = "core-solver-config-v2"
_SOLVE_RESULT_HASH_DOMAIN = "canonical-solve-result-v2"
_MAX_DETERMINISTIC_LIMIT = 2**63 - 1

DeterministicLimitV2 = Annotated[
    int,
    Field(strict=True, ge=1, le=_MAX_DETERMINISTIC_LIMIT),
]
NonNegativeDeterministicLimitV2 = Annotated[
    int,
    Field(strict=True, ge=0, le=_MAX_DETERMINISTIC_LIMIT),
]


class GeometryKernelSoundnessV2(StrEnum):
    """Proof capability of the platform-neutral geometry implementation."""

    EXACT = "EXACT"
    DIRECTED_OUTWARD_BOUNDS = "DIRECTED_OUTWARD_BOUNDS"
    UNVERIFIED_BINARY64 = "UNVERIFIED_BINARY64"


def _reject_boolean_exact_error(value: object) -> object:
    if type(value) is bool:
        raise ValueError("geometry kernel error must be a JSON number, not boolean")
    return value


ExactZeroErrorV2 = Annotated[
    Literal[0.0],
    BeforeValidator(_reject_boolean_exact_error),
]


class _GeometryKernelSpecBaseV2(V2Model):
    """Shared identity fields for a versioned numerical geometry kernel."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="geometry-kernel-spec")
    )
    kernel_id: CanonicalId
    kernel_version: CanonicalId

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="geometry-kernel-spec"):
            raise ValueError("geometry kernel schema identity must be fixed")
        return self


class ExactGeometryKernelSpecV2(_GeometryKernelSpecBaseV2):
    """Kernel whose supported operations are independently exact."""

    soundness: Literal[GeometryKernelSoundnessV2.EXACT] = (
        GeometryKernelSoundnessV2.EXACT
    )
    certified_outward_error_m: ExactZeroErrorV2


class DirectedOutwardGeometryKernelSpecV2(_GeometryKernelSpecBaseV2):
    """Kernel with a sound declared outward linear error bound."""

    soundness: Literal[GeometryKernelSoundnessV2.DIRECTED_OUTWARD_BOUNDS] = (
        GeometryKernelSoundnessV2.DIRECTED_OUTWARD_BOUNDS
    )
    certified_outward_error_m: NonNegativeFiniteFloat


class UnverifiedBinary64GeometryKernelSpecV2(_GeometryKernelSpecBaseV2):
    """Diagnostic binary64 kernel that is ineligible for semantic proofs."""

    soundness: Literal[GeometryKernelSoundnessV2.UNVERIFIED_BINARY64] = (
        GeometryKernelSoundnessV2.UNVERIFIED_BINARY64
    )


GeometryKernelSpecV2: TypeAlias = Annotated[
    ExactGeometryKernelSpecV2
    | DirectedOutwardGeometryKernelSpecV2
    | UnverifiedBinary64GeometryKernelSpecV2,
    Field(discriminator="soundness"),
]


class CoreSolverConfigV2(V2Model):
    """Finite deterministic work limits and algorithm identity for core solve.

    A zero branch-node limit means no branch expansion beyond an analytic root
    solve; a zero refinement-step limit disables refinement.  Domain operations
    and partition cells remain positive because every emitted solve artifact
    requires at least one of each.
    """

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="core-solver-config")
    )
    algorithm_id: CanonicalId
    algorithm_version: CanonicalId
    max_domain_operations: DeterministicLimitV2
    max_partition_cells: DeterministicLimitV2
    max_branch_nodes: NonNegativeDeterministicLimitV2
    max_refinement_steps: NonNegativeDeterministicLimitV2
    target_optimality_gap: NonNegativeFiniteFloat
    geometry_kernel: GeometryKernelSpecV2

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="core-solver-config"):
            raise ValueError("core solver config schema identity must be fixed")
        return self

    @property
    def core_solver_config_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_CORE_SOLVER_CONFIG_HASH_DOMAIN,
        )


class SolveStatusV2(StrEnum):
    CERTIFIED_SUCCESS = "CERTIFIED_SUCCESS"
    PROVEN_UNSAT = "PROVEN_UNSAT"
    UNCERTIFIED = "UNCERTIFIED"


class UncertifiedReasonV2(StrEnum):
    BOUNDED_SEARCH_EXHAUSTED = "BOUNDED_SEARCH_EXHAUSTED"
    NUMERIC_GAP = "NUMERIC_GAP"
    MISSING_FACT = "MISSING_FACT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    COMPILATION_INCOMPLETE = "COMPILATION_INCOMPLETE"


class _CanonicalSolveResultBaseV2(V2Model):
    """Shared identity, config, hash, and reference-closure helpers."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="canonical-solve-result")
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config: CoreSolverConfigV2

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(
            schema_name="canonical-solve-result"
        ):
            raise ValueError("canonical solve result schema identity must be fixed")
        return self

    def _validate_artifact_references(
        self,
        candidate: CandidateDomainArtifactV2 | None,
        relation: RelationCostPartitionV2 | None,
        objective: ObjectivePartitionArtifactV2 | None,
    ) -> None:
        config_hash = self.core_solver_config.core_solver_config_sha256
        if candidate is not None:
            if candidate.semantic_problem_sha256 != self.semantic_problem_sha256:
                raise ValueError("candidate domain problem hash is not closed")
            if candidate.core_solver_config_sha256 != config_hash:
                raise ValueError("candidate domain config hash is not closed")

        if relation is not None:
            if candidate is None:
                raise ValueError("relation partition requires a candidate domain")
            if relation.semantic_problem_sha256 != self.semantic_problem_sha256:
                raise ValueError("relation partition problem hash is not closed")
            if relation.core_solver_config_sha256 != config_hash:
                raise ValueError("relation partition config hash is not closed")
            if relation.candidate_domain_artifact_sha256 != (
                candidate.candidate_domain_artifact_sha256
            ):
                raise ValueError("relation partition candidate hash is not closed")

        if objective is not None:
            if candidate is None:
                raise ValueError("objective partition requires a candidate domain")
            if relation is None:
                raise ValueError("objective partition requires a relation partition")
            if objective.semantic_problem_sha256 != self.semantic_problem_sha256:
                raise ValueError("objective partition problem hash is not closed")
            if objective.core_solver_config_sha256 != config_hash:
                raise ValueError("objective partition config hash is not closed")
            if objective.candidate_domain_artifact_sha256 != (
                candidate.candidate_domain_artifact_sha256
            ):
                raise ValueError("objective partition candidate hash is not closed")
            if objective.relation_cost_partition_sha256 != (
                relation.relation_cost_partition_sha256
            ):
                raise ValueError("objective partition relation hash is not closed")
            if objective.objective_spec_sha256 != relation.objective_spec_sha256:
                raise ValueError("partition objective spec hashes are not closed")

    def _validate_certificate_common_references(
        self,
        certificate: GlobalOptimalityCertificateV2 | ProvenUnsatCertificateV2,
        candidate: CandidateDomainArtifactV2,
    ) -> None:
        if certificate.semantic_problem_sha256 != self.semantic_problem_sha256:
            raise ValueError("certificate problem hash is not closed")
        if certificate.core_solver_config_sha256 != (
            self.core_solver_config.core_solver_config_sha256
        ):
            raise ValueError("certificate config hash is not closed")
        if certificate.candidate_domain_artifact_sha256 != (
            candidate.candidate_domain_artifact_sha256
        ):
            raise ValueError("certificate candidate hash is not closed")

    def _validate_certifiable_geometry_kernel(self) -> None:
        if (
            self.core_solver_config.geometry_kernel.soundness
            is GeometryKernelSoundnessV2.UNVERIFIED_BINARY64
        ):
            raise ValueError(
                "unverified geometry kernel cannot support a semantic certificate"
            )

    @property
    def solve_result_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_SOLVE_RESULT_HASH_DOMAIN)


class CertifiedSuccessResultV2(_CanonicalSolveResultBaseV2):
    """Certified feasible edit with a complete global-optimality proof chain.

    ``global_loss_lower_bound`` is a scalar bound over the complete outer
    partition. ``witness_loss_bounds`` describes the independently evaluated
    concrete edit.  They are deliberately separate: component-wise minima
    taken across different cells are not a valid decomposition of the global
    scalar lower bound.
    """

    status: Literal[SolveStatusV2.CERTIFIED_SUCCESS] = SolveStatusV2.CERTIFIED_SUCCESS
    candidate_domain: CandidateDomainArtifactV2
    relation_cost_partition: RelationCostPartitionV2
    objective_partition: ObjectivePartitionArtifactV2
    edit: CanonicalEditV2
    global_loss_lower_bound: NonNegativeFiniteFloat
    witness_loss_bounds: ObjectiveTermBoundsV2
    certificate: GlobalOptimalityCertificateV2

    @model_validator(mode="after")
    def validate_success(self) -> Self:
        self._validate_certifiable_geometry_kernel()
        if (
            self.certificate.optimality_gap
            > self.core_solver_config.target_optimality_gap
            or (
                self.certificate.epsilon is not None
                and self.certificate.epsilon
                > self.core_solver_config.target_optimality_gap
            )
        ):
            raise ValueError(
                "CERTIFIED_SUCCESS certificate must meet the configured "
                "target optimality gap"
            )
        if not all(
            (
                self.candidate_domain.is_global_verification_eligible,
                self.relation_cost_partition.is_global_verification_eligible,
                self.objective_partition.is_global_verification_eligible,
            )
        ):
            raise ValueError(
                "CERTIFIED_SUCCESS artifacts must be global verification eligible"
            )

        self._validate_artifact_references(
            self.candidate_domain,
            self.relation_cost_partition,
            self.objective_partition,
        )
        if self.edit.semantic_problem_sha256 != self.semantic_problem_sha256:
            raise ValueError("edit problem hash is not closed")
        if self.edit.subject_id != self.candidate_domain.candidate_variable.subject_id:
            raise ValueError("edit subject is not closed to the candidate variable")
        self._validate_certificate_common_references(
            self.certificate,
            self.candidate_domain,
        )
        if self.certificate.relation_cost_partition_sha256 != (
            self.relation_cost_partition.relation_cost_partition_sha256
        ):
            raise ValueError("certificate relation partition hash is not closed")
        if self.certificate.objective_partition_artifact_sha256 != (
            self.objective_partition.objective_partition_artifact_sha256
        ):
            raise ValueError("certificate objective partition hash is not closed")
        if self.certificate.edit_sha256 != self.edit.edit_sha256:
            raise ValueError("certificate edit hash is not closed")
        if self.certificate.loss_lower_bound != self.global_loss_lower_bound:
            raise ValueError(
                "certificate lower bound and result global lower bound are not closed"
            )
        if (
            self.certificate.loss_upper_bound
            != self.witness_loss_bounds.total_upper_bound
        ):
            raise ValueError(
                "certificate upper bound and witness loss bounds are not closed"
            )
        return self


class ProvenUnsatResultV2(_CanonicalSolveResultBaseV2):
    """Certified empty hard-domain outer bound with no edit or pseudo-loss."""

    status: Literal[SolveStatusV2.PROVEN_UNSAT] = SolveStatusV2.PROVEN_UNSAT
    candidate_domain: CandidateDomainArtifactV2
    certificate: ProvenUnsatCertificateV2

    @model_validator(mode="after")
    def validate_unsat(self) -> Self:
        self._validate_certifiable_geometry_kernel()
        if not self.candidate_domain.is_unsat_verification_eligible:
            raise ValueError(
                "PROVEN_UNSAT requires a complete certified empty outer domain"
            )
        self._validate_artifact_references(self.candidate_domain, None, None)
        self._validate_certificate_common_references(
            self.certificate,
            self.candidate_domain,
        )
        return self


class UncertifiedResultV2(_CanonicalSolveResultBaseV2):
    """Closed non-certificate outcome that may retain hash-closed partial work."""

    status: Literal[SolveStatusV2.UNCERTIFIED] = SolveStatusV2.UNCERTIFIED
    uncertified_reason: UncertifiedReasonV2
    candidate_domain: CandidateDomainArtifactV2 | None = None
    relation_cost_partition: RelationCostPartitionV2 | None = None
    objective_partition: ObjectivePartitionArtifactV2 | None = None

    @model_validator(mode="after")
    def validate_uncertified(self) -> Self:
        self._validate_artifact_references(
            self.candidate_domain,
            self.relation_cost_partition,
            self.objective_partition,
        )
        return self


CanonicalSolveResultV2: TypeAlias = Annotated[
    CertifiedSuccessResultV2 | ProvenUnsatResultV2 | UncertifiedResultV2,
    Field(discriminator="status"),
]
