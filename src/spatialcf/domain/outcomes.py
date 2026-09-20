"""Typed counterfactual outcomes, certificates, and the one-way M1 hash DAG.

The contracts here deliberately stop at local structural validation.  They bind
every later record to the two roots and reject malformed local dependency
shapes, but they do not resolve definitions, select an implementation, execute
a backend, or certify a proposal.  Those cross-record checks remain Task 8
registry work.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self, TypeAlias, TypeVar

from pydantic import Field, StrictBool, model_validator

from spatialcf.domain.base import CanonicalModel, NonNegativeFiniteFloat, Sha256Digest
from spatialcf.domain.definitions import (
    CapabilityRef,
    DefinitionRef,
    HashBoundCanonicalModel,
    SchemaRef,
    TypedValue,
)
from spatialcf.domain.profiles import BackendRef, OwnerRef
from spatialcf.domain.serialization import canonical_json_bytes

__all__ = (
    "BackendCompleteUnsatEvidence",
    "BackendProposal",
    "BackendProposalSubmission",
    "BackendSelectionRecord",
    "BackendSubmission",
    "BackendTerminalEvidence",
    "BackendUnknownEvidence",
    "CapabilityMatch",
    "CapabilityMismatch",
    "CertifiedSolutionCertificate",
    "CertifiedSolutionResult",
    "CheckedProofOutcome",
    "CheckerDisposition",
    "CounterfactualCertificate",
    "CounterfactualSolveResult",
    "NoncertifiedWitnessResult",
    "ProofMaterialEnvelope",
    "ProvenUnsatCertificate",
    "ProvenUnsatResult",
    "ResourceUsage",
    "TypedCompilationOutcome",
    "UnknownResult",
    "VerifierDispatchRecord",
)

_ValueT = TypeVar("_ValueT")


def _require_sorted_unique_by_bytes(
    values: tuple[_ValueT, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")


class _ArtifactReference(CanonicalModel):
    """One typed, content-addressed artifact without an independent identity."""

    artifact_schema_ref: SchemaRef
    artifact_sha256: Sha256Digest


class _ResourceUsageEntry(CanonicalModel):
    """One deterministic resource counter closed by its definition reference."""

    resource_definition_ref: DefinitionRef
    used: NonNegativeFiniteFloat


class ResourceUsage(CanonicalModel):
    """A closed embedded resource row; parent records bind it into their digest."""

    accounting_claim_definition_ref: DefinitionRef
    entries: tuple[_ResourceUsageEntry, ...]
    exhausted: StrictBool

    @model_validator(mode="after")
    def _validate_entries(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.entries,
            "resource usage entries",
            nonempty=True,
        )
        refs = tuple(entry.resource_definition_ref for entry in self.entries)
        if len(set(refs)) != len(refs):
            raise ValueError("resource usage must not duplicate one resource")
        return self


class CapabilityMatch(CanonicalModel):
    """A locally closed successful capability inspection row."""

    kind: Literal["MATCH"] = "MATCH"
    backend_ref: BackendRef
    backend_descriptor_sha256: Sha256Digest
    matched_capability_refs: tuple[CapabilityRef, ...]
    match_claim_definition_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_capabilities(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.matched_capability_refs,
            "matched capabilities",
            nonempty=True,
        )
        return self


class CapabilityMismatch(CanonicalModel):
    """A locally closed typed non-support row rather than an exception."""

    kind: Literal["MISMATCH"] = "MISMATCH"
    backend_ref: BackendRef
    backend_descriptor_sha256: Sha256Digest
    missing_capability_refs: tuple[CapabilityRef, ...]
    reason_claim_definition_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_capabilities(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.missing_capability_refs,
            "missing capabilities",
            nonempty=True,
        )
        return self


_CapabilityRow: TypeAlias = Annotated[
    CapabilityMatch | CapabilityMismatch,
    Field(discriminator="kind"),
]


class BackendSelectionRecord(HashBoundCanonicalModel):
    """The ordered frozen backend routing record before any proposal is trusted."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/backend-selection-record/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_selection_record_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    implementation_registry_snapshot_sha256: Sha256Digest
    backend_descriptor_bundle_sha256: Sha256Digest
    backend_routing_policy_sha256: Sha256Digest
    ordered_candidate_backend_refs: tuple[BackendRef, ...]
    capability_rows: tuple[_CapabilityRow, ...]
    selection_disposition: Literal["SELECTED", "NO_SELECTION"]
    selection_disposition_claim_ref: DefinitionRef
    selected_backend_ref: BackendRef | None = None
    selected_backend_descriptor_sha256: Sha256Digest | None = None
    resource_allocation: ResourceUsage | None = None
    deterministic_selection_reason_ref: DefinitionRef
    backend_selection_record_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_local_selection(self) -> Self:
        candidates = self.ordered_candidate_backend_refs
        if not candidates:
            raise ValueError("selection candidates must not be empty")
        if len(set(candidates)) != len(candidates):
            raise ValueError("selection candidates must not contain duplicate backends")
        row_backends = tuple(row.backend_ref for row in self.capability_rows)
        if row_backends != candidates:
            raise ValueError("selection candidates must have one local row each")

        selected_values = (
            self.selected_backend_ref,
            self.selected_backend_descriptor_sha256,
            self.resource_allocation,
        )
        if self.selection_disposition == "SELECTED":
            if any(value is None for value in selected_values):
                raise ValueError(
                    "selected disposition requires selected backend fields"
                )
            assert self.selected_backend_ref is not None
            assert self.selected_backend_descriptor_sha256 is not None
            matching_rows = tuple(
                row
                for row in self.capability_rows
                if row.backend_ref == self.selected_backend_ref
            )
            if len(matching_rows) != 1 or not isinstance(
                matching_rows[0], CapabilityMatch
            ):
                raise ValueError("selected backend must have one capability match")
            if (
                matching_rows[0].backend_descriptor_sha256
                != self.selected_backend_descriptor_sha256
            ):
                raise ValueError("selected backend descriptor does not match local row")
        elif any(value is not None for value in selected_values):
            raise ValueError("no-selection disposition forbids selected backend fields")
        return self


class TypedCompilationOutcome(HashBoundCanonicalModel):
    """An explicitly untrusted compilation-stage terminal record."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/typed-compilation-outcome/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "typed_compilation_outcome_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    selected_backend_ref: BackendRef
    selected_backend_descriptor_sha256: Sha256Digest
    compilation_reason_claim_definition_ref: DefinitionRef
    partial_artifact_refs: tuple[_ArtifactReference, ...]
    resource_usage: ResourceUsage
    typed_compilation_outcome_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_partial_artifacts(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.partial_artifact_refs,
            "compilation partial artifacts",
        )
        return self


class ProofMaterialEnvelope(HashBoundCanonicalModel):
    """A typed proof-material payload that remains untrusted until checked."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/proof-material-envelope/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_material_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    proposal_backend_ref: BackendRef
    proof_material_definition_ref: DefinitionRef
    payload_schema_ref: SchemaRef
    typed_payload: tuple[TypedValue, ...]
    artifact_refs: tuple[_ArtifactReference, ...]
    proof_material_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_payload_closure(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.artifact_refs,
            "proof material artifacts",
        )
        return self


class BackendProposal(HashBoundCanonicalModel):
    """An untrusted backend proposal that cannot name a certified result."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/backend-proposal/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_proposal_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    proposal_backend_ref: BackendRef
    proposal_backend_owner_ref: OwnerRef
    proposal_backend_capability_ref: CapabilityRef
    proposal_backend_build_sha256: Sha256Digest
    proposal_claim_definition_ref: DefinitionRef
    proof_material: ProofMaterialEnvelope
    proof_material_sha256: Sha256Digest
    program_sha256: Sha256Digest | None = None
    after_scene_state_sha256: Sha256Digest | None = None
    witness_artifact_refs: tuple[_ArtifactReference, ...] = ()
    model_artifact_refs: tuple[_ArtifactReference, ...] = ()
    partial_artifact_refs: tuple[_ArtifactReference, ...] = ()
    objective_lower_bound: NonNegativeFiniteFloat
    objective_upper_bound: NonNegativeFiniteFloat
    resource_usage: ResourceUsage
    backend_proposal_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_untrusted_proposal(self) -> Self:
        if self.proof_material_sha256 != self.proof_material.proof_material_sha256:
            raise ValueError("proposal proof material digest does not match envelope")
        if (self.program_sha256 is None) != (self.after_scene_state_sha256 is None):
            raise ValueError(
                "proposal program and after-state hashes must appear together"
            )
        if self.objective_lower_bound > self.objective_upper_bound:
            raise ValueError(
                "proposal objective lower bound must not exceed upper bound"
            )
        for label, artifacts in (
            ("proposal witness artifacts", self.witness_artifact_refs),
            ("proposal model artifacts", self.model_artifact_refs),
            ("proposal partial artifacts", self.partial_artifact_refs),
        ):
            _require_sorted_unique_by_bytes(artifacts, label)
        return self


class BackendProposalSubmission(HashBoundCanonicalModel):
    """The additive submission-v2 wrapper for one retained finite-bound proposal."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/backend-proposal-submission/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_proposal_submission_sha256"

    submission_kind: Literal["PROPOSAL"] = "PROPOSAL"
    proposal: BackendProposal
    backend_proposal_sha256: Sha256Digest
    backend_proposal_submission_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_retained_proposal(self) -> Self:
        if self.backend_proposal_sha256 != self.proposal.backend_proposal_sha256:
            raise ValueError("submission proposal digest does not match retained proposal")
        return self


class _BackendTerminalEvidenceBase(HashBoundCanonicalModel):
    """Shared pre-check terminal evidence fields with no objective/program payload."""

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    proposal_backend_ref: BackendRef
    proposal_backend_owner_ref: OwnerRef
    proposal_backend_capability_ref: CapabilityRef
    proposal_backend_build_sha256: Sha256Digest
    proof_material: ProofMaterialEnvelope
    proof_material_sha256: Sha256Digest
    resource_usage: ResourceUsage
    partial_artifact_refs: tuple[_ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def _validate_terminal_evidence_closure(self) -> Self:
        material = self.proof_material
        if self.proof_material_sha256 != material.proof_material_sha256:
            raise ValueError("terminal proof material digest does not match envelope")
        if (
            self.semantic_problem_sha256,
            self.solve_request_sha256,
            self.backend_selection_record_sha256,
            self.proposal_backend_ref,
        ) != (
            material.semantic_problem_sha256,
            material.solve_request_sha256,
            material.backend_selection_record_sha256,
            material.proposal_backend_ref,
        ):
            raise ValueError("terminal evidence roots do not match proof material")
        _require_sorted_unique_by_bytes(
            self.partial_artifact_refs,
            "terminal partial artifacts",
        )
        return self


class BackendCompleteUnsatEvidence(_BackendTerminalEvidenceBase):
    """Untrusted complete-domain empty evidence without fabricated bounds."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/backend-complete-unsat-evidence/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_complete_unsat_evidence_sha256"

    submission_kind: Literal["COMPLETE_DOMAIN_UNSAT_EVIDENCE"] = (
        "COMPLETE_DOMAIN_UNSAT_EVIDENCE"
    )
    complete_domain_claim_definition_ref: DefinitionRef
    authorized_domain_sha256: Sha256Digest
    complete_domain_coverage_artifact_sha256: Sha256Digest
    backend_complete_unsat_evidence_sha256: Sha256Digest


class BackendUnknownEvidence(_BackendTerminalEvidenceBase):
    """Untrusted selected-backend unknown evidence without fabricated bounds."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/backend-unknown-evidence/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_unknown_evidence_sha256"

    submission_kind: Literal["UNKNOWN_EVIDENCE"] = "UNKNOWN_EVIDENCE"
    reason_claim_definition_ref: DefinitionRef
    backend_unknown_evidence_sha256: Sha256Digest


BackendTerminalEvidence: TypeAlias = Annotated[
    BackendCompleteUnsatEvidence | BackendUnknownEvidence,
    Field(discriminator="submission_kind"),
]

BackendSubmission: TypeAlias = Annotated[
    BackendProposalSubmission | BackendCompleteUnsatEvidence | BackendUnknownEvidence,
    Field(discriminator="submission_kind"),
]


class CheckerDisposition(StrEnum):
    """The complete structural disposition vocabulary for one trusted checker."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    LIMITED = "LIMITED"


class CheckedProofOutcome(HashBoundCanonicalModel):
    """The checked claim before verifier dispatch or certificate assembly."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/checked-proof-outcome/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "checked_proof_outcome_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    proof_material_sha256: Sha256Digest
    checker_capability_ref: CapabilityRef
    checker_build_sha256: Sha256Digest
    checker_disposition: CheckerDisposition
    checked_claim_definition_ref: DefinitionRef
    checked_fact_refs: tuple[_ArtifactReference, ...]
    checked_proof_outcome_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_checked_facts(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.checked_fact_refs,
            "checked proof facts",
            nonempty=True,
        )
        return self


class VerifierDispatchRecord(HashBoundCanonicalModel):
    """A dispatch record that keeps proposal and trusted checker identities apart."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/verifier-dispatch-record/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "verifier_dispatch_record_sha256"

    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    semantic_definition_bundle_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    proof_policy_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    proposal_backend_owner_ref: OwnerRef
    proposal_backend_capability_ref: CapabilityRef
    proposal_backend_build_sha256: Sha256Digest
    proof_material_definition_ref: DefinitionRef
    checker_owner_ref: OwnerRef
    checker_capability_ref: CapabilityRef
    checker_build_sha256: Sha256Digest
    checked_proof_outcome_sha256: Sha256Digest
    verifier_dispatch_record_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_independent_checker(self) -> Self:
        if self.proposal_backend_owner_ref == self.checker_owner_ref:
            raise ValueError("proposal backend and checker owners must be distinct")
        if self.proposal_backend_capability_ref == self.checker_capability_ref:
            raise ValueError(
                "proposal backend and checker capabilities must be distinct"
            )
        return self


class _CertificateBase(HashBoundCanonicalModel):
    """Shared exact certificate fields; subclasses supply the discriminated branch."""

    SELF_DIGEST_FIELD: ClassVar[str] = "certificate_sha256"

    certificate_kind: str
    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    scene_state_sha256: Sha256Digest
    backend_selection_record_sha256: Sha256Digest
    checked_proof_outcome_sha256: Sha256Digest
    verifier_dispatch_record_sha256: Sha256Digest
    semantic_definition_bundle_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    semantics_profile_sha256: Sha256Digest
    action_space_profile_sha256: Sha256Digest
    intervention_authorization_sha256: Sha256Digest
    objective_expression_sha256: Sha256Digest
    proof_policy_sha256: Sha256Digest
    resource_policy_sha256: Sha256Digest
    backend_routing_policy_sha256: Sha256Digest
    solver_config_sha256: Sha256Digest
    implementation_registry_snapshot_sha256: Sha256Digest
    backend_descriptor_bundle_sha256: Sha256Digest
    proposal_backend_build_sha256: Sha256Digest
    checker_build_sha256: Sha256Digest
    claim_definition_ref: DefinitionRef
    proof_material_definition_ref: DefinitionRef
    proof_material_sha256: Sha256Digest
    checker_disposition: Literal[CheckerDisposition.ACCEPTED]
    resource_usage: ResourceUsage
    certificate_sha256: Sha256Digest


class CertifiedSolutionCertificate(_CertificateBase):
    """A trusted solution certificate with a complete program/state transition."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/certified-solution-certificate/3.0"
    )

    certificate_kind: Literal["CERTIFIED_SOLUTION"] = "CERTIFIED_SOLUTION"
    program_sha256: Sha256Digest
    after_scene_state_sha256: Sha256Digest
    state_delta_manifest_sha256: Sha256Digest
    grounded_obligation_set_sha256: Sha256Digest


class ProvenUnsatCertificate(_CertificateBase):
    """A trusted complete-domain UNSAT certificate with no program payload."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/proven-unsat-certificate/3.0"

    certificate_kind: Literal["PROVEN_UNSAT"] = "PROVEN_UNSAT"
    authorized_domain_sha256: Sha256Digest
    complete_domain_coverage_artifact_sha256: Sha256Digest
    complete_domain_claim_definition_ref: DefinitionRef
    sound_complete_domain_claim_definition_ref: DefinitionRef


CounterfactualCertificate: TypeAlias = Annotated[
    CertifiedSolutionCertificate | ProvenUnsatCertificate,
    Field(discriminator="certificate_kind"),
]


class _ResultBase(HashBoundCanonicalModel):
    """Shared direct roots and checker-pair structure for all four outcomes."""

    SELF_DIGEST_FIELD: ClassVar[str] = "solve_result_sha256"

    structural_outcome_class: str
    semantic_problem_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    claim_definition_ref: DefinitionRef
    backend_selection_record_sha256: Sha256Digest
    checked_proof_outcome_sha256: Sha256Digest | None = None
    verifier_dispatch_record_sha256: Sha256Digest | None = None
    checker_disposition: CheckerDisposition | None = None
    resource_usage: ResourceUsage
    solve_result_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_checker_pair(self) -> Self:
        has_checked = self.checked_proof_outcome_sha256 is not None
        has_dispatch = self.verifier_dispatch_record_sha256 is not None
        if has_checked != has_dispatch:
            raise ValueError("checker pair must be both present or both absent")
        if has_checked and self.checker_disposition is None:
            raise ValueError("checker pair requires a checker disposition")
        if not has_checked and self.checker_disposition is not None:
            raise ValueError("checker disposition requires the checker pair")
        return self


def _validate_certificate_result_binding(
    result: _ResultBase,
    certificate: _CertificateBase,
) -> None:
    for field_name in (
        "semantic_problem_sha256",
        "solve_request_sha256",
        "claim_definition_ref",
        "backend_selection_record_sha256",
        "checked_proof_outcome_sha256",
        "verifier_dispatch_record_sha256",
    ):
        if getattr(result, field_name) != getattr(certificate, field_name):
            raise ValueError("result direct root does not match accepted certificate")
    if result.checker_disposition is not CheckerDisposition.ACCEPTED:
        raise ValueError("certified results require ACCEPTED checker disposition")


class CertifiedSolutionResult(_ResultBase):
    """A certified result whose direct certificate digest matches its payload."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/certified-solution-result/3.0"
    )

    structural_outcome_class: Literal["CERTIFIED_SOLUTION"] = "CERTIFIED_SOLUTION"
    checker_disposition: Literal[CheckerDisposition.ACCEPTED]
    accepted_certificate: CertifiedSolutionCertificate
    certificate_sha256: Sha256Digest
    program_sha256: Sha256Digest
    after_scene_state_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_solution_certificate(self) -> Self:
        if self.certificate_sha256 != self.accepted_certificate.certificate_sha256:
            raise ValueError("certificate digest does not match accepted certificate")
        _validate_certificate_result_binding(self, self.accepted_certificate)
        if self.program_sha256 != self.accepted_certificate.program_sha256:
            raise ValueError("solution program digest does not match certificate")
        if (
            self.after_scene_state_sha256
            != self.accepted_certificate.after_scene_state_sha256
        ):
            raise ValueError("solution after-state digest does not match certificate")
        return self


class ProvenUnsatResult(_ResultBase):
    """A certified complete-domain negative result without any edit or witness."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/proven-unsat-result/3.0"

    structural_outcome_class: Literal["PROVEN_UNSAT"] = "PROVEN_UNSAT"
    checker_disposition: Literal[CheckerDisposition.ACCEPTED]
    accepted_certificate: ProvenUnsatCertificate
    certificate_sha256: Sha256Digest
    complete_domain_coverage_artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_unsat_certificate(self) -> Self:
        if self.certificate_sha256 != self.accepted_certificate.certificate_sha256:
            raise ValueError("certificate digest does not match accepted certificate")
        _validate_certificate_result_binding(self, self.accepted_certificate)
        if (
            self.complete_domain_coverage_artifact_sha256
            != self.accepted_certificate.complete_domain_coverage_artifact_sha256
        ):
            raise ValueError(
                "complete-domain coverage digest does not match certificate"
            )
        return self


class NoncertifiedWitnessResult(_ResultBase):
    """A typed diagnostic witness that cannot contain a trusted certificate."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/noncertified-witness-result/3.0"
    )

    structural_outcome_class: Literal["NONCERTIFIED_WITNESS"] = "NONCERTIFIED_WITNESS"
    evidence_claim_definition_ref: DefinitionRef
    program_sha256: Sha256Digest | None = None
    after_scene_state_sha256: Sha256Digest | None = None
    witness_artifact_refs: tuple[_ArtifactReference, ...] = ()
    model_artifact_refs: tuple[_ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def _validate_noncertified_witness(self) -> Self:
        if self.checker_disposition is CheckerDisposition.ACCEPTED:
            raise ValueError(
                "noncertified results may not carry ACCEPTED checker disposition"
            )
        if (self.program_sha256 is None) != (self.after_scene_state_sha256 is None):
            raise ValueError(
                "witness program and after-state hashes must appear together"
            )
        _require_sorted_unique_by_bytes(
            self.witness_artifact_refs,
            "witness artifacts",
        )
        _require_sorted_unique_by_bytes(
            self.model_artifact_refs,
            "model artifacts",
        )
        if (
            self.program_sha256 is None
            and not self.witness_artifact_refs
            and not self.model_artifact_refs
        ):
            raise ValueError(
                "noncertified result requires a witness, program, or model"
            )
        return self


class UnknownResult(_ResultBase):
    """A hash-closed terminal unknown with typed reason and optional artifacts."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/unknown-result/3.0"

    structural_outcome_class: Literal["UNKNOWN"] = "UNKNOWN"
    reason_claim_definition_ref: DefinitionRef
    partial_artifact_refs: tuple[_ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def _validate_unknown_result(self) -> Self:
        if self.checker_disposition is CheckerDisposition.ACCEPTED:
            raise ValueError(
                "unknown results may not carry ACCEPTED checker disposition"
            )
        _require_sorted_unique_by_bytes(
            self.partial_artifact_refs,
            "unknown partial artifacts",
        )
        return self


CounterfactualSolveResult: TypeAlias = Annotated[
    CertifiedSolutionResult
    | ProvenUnsatResult
    | NoncertifiedWitnessResult
    | UnknownResult,
    Field(discriminator="structural_outcome_class"),
]
