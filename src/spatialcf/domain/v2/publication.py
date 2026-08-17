"""Typed terminal publication records outside Canonical v2 semantics.

Publication binds a resolved pure-core result to provenance and one final audit.
It never changes the core status.  In particular, platform/write-back failures
cannot be relabelled as semantic UNSAT, and an audit failure preserves the
certified core result that it audited.

Records embed the complete production ``ObjectiveSpecV2`` and close its digest
to the core result.  An independent publication verifier must still resolve
``semantic_problem_sha256`` and check that its actual objective is identical;
record construction is structural validation, not a substitute for resolution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    SchemaIdentityV2,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.evidence import (
    AuditStatusV2,
    EvidenceEnvelopeV2,
    FinalAuditOutcomeV2,
    MappingProofKindV2,
    MappingProofStatusV2,
    MappingProofV2,
    PreSemanticEvidenceEnvelopeV2,
)
from spatialcf.domain.v2.objective import ObjectiveModeV2, ObjectiveSpecV2
from spatialcf.domain.v2.result import (
    CanonicalSolveResultV2,
    CertifiedSuccessResultV2,
    ProvenUnsatResultV2,
    SolveStatusV2,
    UncertifiedResultV2,
)
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_PAIR_HASH_DOMAIN = "spatialcf.pair-record.v2"
_REJECTION_HASH_DOMAIN = "spatialcf.rejection-record.v2"
_FINAL_AUDIT_HASH_DOMAIN = "spatialcf.final-audit-outcome.v2"
_MANIFEST_HASH_DOMAIN = "spatialcf.publication-manifest.v2"

FindingIdV2 = Annotated[
    str,
    Field(strict=True, pattern=r"^CV2-[0-9]{3}$"),
]


class PublicationRecordKindV2(StrEnum):
    PAIR = "PAIR"
    REJECTION = "REJECTION"


class RejectionStageV2(StrEnum):
    SOURCE_ADAPTER = "SOURCE_ADAPTER"
    SEMANTIC_VALIDATION = "SEMANTIC_VALIDATION"
    CORE_SOLVE = "CORE_SOLVE"
    WRITE_BACK = "WRITE_BACK"
    FINAL_AUDIT = "FINAL_AUDIT"
    PUBLICATION = "PUBLICATION"


class FindingClassificationV2(StrEnum):
    GENERAL_CONSTRAINT = "GENERAL_CONSTRAINT"
    ADAPTER_MAPPING = "ADAPTER_MAPPING"
    PLATFORM_EXECUTION_NOISE = "PLATFORM_EXECUTION_NOISE"


class RejectionCodeV2(StrEnum):
    SOURCE_ADAPTER_REJECTED = "SOURCE_ADAPTER_REJECTED"
    SEMANTIC_VALIDATION_REJECTED = "SEMANTIC_VALIDATION_REJECTED"
    CORE_PROVEN_UNSAT = "CORE_PROVEN_UNSAT"
    CORE_UNCERTIFIED = "CORE_UNCERTIFIED"
    WRITE_BACK_REJECTED = "WRITE_BACK_REJECTED"
    FINAL_AUDIT_FAILED = "FINAL_AUDIT_FAILED"
    PUBLICATION_REJECTED = "PUBLICATION_REJECTED"


class RejectionOwnerV2(StrEnum):
    SOURCE_ADAPTER = "SOURCE_ADAPTER"
    CANONICAL_CONTRACT = "CANONICAL_CONTRACT"
    CORE_SOLVER = "CORE_SOLVER"
    WRITE_BACK_ADAPTER = "WRITE_BACK_ADAPTER"
    PLATFORM_RUNTIME = "PLATFORM_RUNTIME"
    PUBLICATION = "PUBLICATION"


_FINDING_CLASSIFICATIONS = {
    "CV2-001": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-002": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-003": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-004": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-005": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-006": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-007": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-008": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-009": FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
    "CV2-010": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-011": FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
    "CV2-012": FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
    "CV2-013": FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
    "CV2-014": FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
    "CV2-015": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-016": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-017": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-018": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-019": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-020": FindingClassificationV2.ADAPTER_MAPPING,
    "CV2-021": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-022": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-023": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-024": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-025": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-026": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-027": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-028": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-029": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-030": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-031": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-032": FindingClassificationV2.GENERAL_CONSTRAINT,
    "CV2-033": FindingClassificationV2.GENERAL_CONSTRAINT,
}

_STAGE_CODES = {
    RejectionStageV2.SOURCE_ADAPTER: frozenset(
        {RejectionCodeV2.SOURCE_ADAPTER_REJECTED}
    ),
    RejectionStageV2.SEMANTIC_VALIDATION: frozenset(
        {RejectionCodeV2.SEMANTIC_VALIDATION_REJECTED}
    ),
    RejectionStageV2.CORE_SOLVE: frozenset(
        {RejectionCodeV2.CORE_PROVEN_UNSAT, RejectionCodeV2.CORE_UNCERTIFIED}
    ),
    RejectionStageV2.WRITE_BACK: frozenset({RejectionCodeV2.WRITE_BACK_REJECTED}),
    RejectionStageV2.FINAL_AUDIT: frozenset({RejectionCodeV2.FINAL_AUDIT_FAILED}),
    RejectionStageV2.PUBLICATION: frozenset({RejectionCodeV2.PUBLICATION_REJECTED}),
}

_STAGE_CLASSIFICATIONS = {
    RejectionStageV2.SOURCE_ADAPTER: frozenset(
        {FindingClassificationV2.ADAPTER_MAPPING}
    ),
    RejectionStageV2.SEMANTIC_VALIDATION: frozenset(
        {
            FindingClassificationV2.GENERAL_CONSTRAINT,
            FindingClassificationV2.ADAPTER_MAPPING,
        }
    ),
    RejectionStageV2.CORE_SOLVE: frozenset(
        {FindingClassificationV2.GENERAL_CONSTRAINT}
    ),
    RejectionStageV2.WRITE_BACK: frozenset(
        {
            FindingClassificationV2.ADAPTER_MAPPING,
            FindingClassificationV2.PLATFORM_EXECUTION_NOISE,
        }
    ),
    RejectionStageV2.FINAL_AUDIT: frozenset(FindingClassificationV2),
    RejectionStageV2.PUBLICATION: frozenset(
        {FindingClassificationV2.GENERAL_CONSTRAINT}
    ),
}


class FindingReferenceV2(V2Model):
    """Stable inventory finding and closed rejection code, never free text."""

    finding_id: FindingIdV2
    classification: FindingClassificationV2
    code: RejectionCodeV2

    @model_validator(mode="after")
    def validate_inventory_classification(self) -> Self:
        expected = _FINDING_CLASSIFICATIONS.get(self.finding_id)
        if expected is None:
            raise ValueError("finding_id is absent from the frozen CV2 inventory")
        if self.classification is not expected:
            raise ValueError(
                "finding classification does not match the frozen CV2 inventory"
            )
        return self


class PairRecordV2(V2Model):
    """Publishable production pair with a certified result and passing audit."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="pair-record")
    )
    record_kind: Literal[PublicationRecordKindV2.PAIR] = PublicationRecordKindV2.PAIR
    request_id: CanonicalId
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    objective_spec: ObjectiveSpecV2
    solve_result: CanonicalSolveResultV2
    solve_result_sha256: Sha256Digest
    edit_sha256: Sha256Digest
    certificate_sha256: Sha256Digest
    evidence_envelope: EvidenceEnvelopeV2
    evidence_envelope_sha256: Sha256Digest
    final_audit_id: CanonicalId
    final_audit_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="pair-record"):
            raise ValueError("pair-record schema identity must be fixed")
        if self.solve_result.status is not SolveStatusV2.CERTIFIED_SUCCESS:
            raise ValueError(
                "PairRecordV2 requires a CERTIFIED_SUCCESS core solve result"
            )
        assert isinstance(self.solve_result, CertifiedSuccessResultV2)
        _validate_success_reference_closure(
            semantic_problem_sha256=self.semantic_problem_sha256,
            core_solver_config_sha256=self.core_solver_config_sha256,
            objective_spec=self.objective_spec,
            solve_result=self.solve_result,
            solve_result_sha256=self.solve_result_sha256,
            edit_sha256=self.edit_sha256,
            certificate_sha256=self.certificate_sha256,
            evidence_envelope=self.evidence_envelope,
            evidence_envelope_sha256=self.evidence_envelope_sha256,
        )
        _validate_verified_write_back(
            self.evidence_envelope,
            edit_sha256=self.edit_sha256,
            subject_id=self.solve_result.edit.subject_id,
        )
        _validate_selected_audit(
            evidence_envelope=self.evidence_envelope,
            final_audit_id=self.final_audit_id,
            final_audit_sha256=self.final_audit_sha256,
            edit_sha256=self.edit_sha256,
            required_status=AuditStatusV2.PASS,
        )
        return self

    @property
    def final_audit(self) -> FinalAuditOutcomeV2:
        return self.evidence_envelope.final_audits[0]

    @property
    def pair_record_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_PAIR_HASH_DOMAIN)


class RejectionRecordV2(V2Model):
    """Terminal structured rejection that preserves the stage's true state."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="rejection-record")
    )
    record_kind: Literal[PublicationRecordKindV2.REJECTION] = (
        PublicationRecordKindV2.REJECTION
    )
    request_id: CanonicalId
    stage: RejectionStageV2
    findings: tuple[FindingReferenceV2, ...] = Field(min_length=1)
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    objective_spec: ObjectiveSpecV2 | None = None
    pre_semantic_evidence: PreSemanticEvidenceEnvelopeV2 | None = None
    pre_semantic_evidence_sha256: Sha256Digest | None = None
    solve_result: CanonicalSolveResultV2 | None = None
    solve_result_sha256: Sha256Digest | None = None
    edit_sha256: Sha256Digest | None = None
    certificate_sha256: Sha256Digest | None = None
    evidence_envelope: EvidenceEnvelopeV2 | None = None
    evidence_envelope_sha256: Sha256Digest | None = None
    final_audit_id: CanonicalId | None = None
    final_audit_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="rejection-record"):
            raise ValueError("rejection-record schema identity must be fixed")
        self._canonicalize_and_validate_findings()

        if self.stage in {
            RejectionStageV2.SOURCE_ADAPTER,
            RejectionStageV2.SEMANTIC_VALIDATION,
        }:
            self._validate_pre_core()
        elif self.stage is RejectionStageV2.CORE_SOLVE:
            self._validate_core_solve()
        elif self.stage is RejectionStageV2.WRITE_BACK:
            self._validate_write_back()
        elif self.stage is RejectionStageV2.FINAL_AUDIT:
            self._validate_post_success_audit(AuditStatusV2.FAIL)
        else:
            self._validate_post_success_audit(AuditStatusV2.PASS)
        return self

    def _canonicalize_and_validate_findings(self) -> None:
        finding_ids = tuple(finding.finding_id for finding in self.findings)
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("rejection finding IDs must be unique")
        for finding in self.findings:
            if finding.code not in _STAGE_CODES[self.stage]:
                raise ValueError("rejection code is incompatible with rejection stage")
            if finding.classification not in _STAGE_CLASSIFICATIONS[self.stage]:
                raise ValueError(
                    "finding classification is incompatible with rejection stage"
                )
        object.__setattr__(
            self,
            "findings",
            tuple(sorted(self.findings, key=lambda finding: finding.finding_id)),
        )

    def _validate_pre_core(self) -> None:
        carried = (
            self.semantic_problem_sha256,
            self.core_solver_config_sha256,
            self.objective_spec,
            self.solve_result,
            self.solve_result_sha256,
            self.edit_sha256,
            self.certificate_sha256,
            self.evidence_envelope,
            self.evidence_envelope_sha256,
            self.final_audit_id,
            self.final_audit_sha256,
        )
        if any(value is not None for value in carried):
            raise ValueError(
                "SOURCE_ADAPTER/SEMANTIC_VALIDATION pre-core rejection must not "
                "fabricate core or publication artifacts"
            )
        if (
            self.pre_semantic_evidence is None
            or self.pre_semantic_evidence_sha256 is None
        ):
            raise ValueError("pre-core rejection requires pre-semantic evidence")
        if self.pre_semantic_evidence_sha256 != (
            self.pre_semantic_evidence.pre_semantic_evidence_sha256
        ):
            raise ValueError("pre-semantic evidence hash is not closed")
        if self.stage is RejectionStageV2.SEMANTIC_VALIDATION and not any(
            proof.status
            in {
                MappingProofStatusV2.REJECTED,
                MappingProofStatusV2.INCOMPLETE,
            }
            for proof in self.pre_semantic_evidence.mapping_proofs
        ):
            raise ValueError(
                "SEMANTIC_VALIDATION requires a non-VERIFIED mapping proof"
            )

    def _validate_core_solve(self) -> None:
        self._forbid_pre_semantic_evidence()
        if self.solve_result is None or self.solve_result.status not in {
            SolveStatusV2.PROVEN_UNSAT,
            SolveStatusV2.UNCERTIFIED,
        }:
            raise ValueError(
                "CORE_SOLVE rejection requires PROVEN_UNSAT or UNCERTIFIED result"
            )
        assert isinstance(
            self.solve_result,
            ProvenUnsatResultV2 | UncertifiedResultV2,
        )
        _validate_common_result_and_evidence_closure(
            semantic_problem_sha256=self.semantic_problem_sha256,
            core_solver_config_sha256=self.core_solver_config_sha256,
            solve_result=self.solve_result,
            solve_result_sha256=self.solve_result_sha256,
            evidence_envelope=self.evidence_envelope,
            evidence_envelope_sha256=self.evidence_envelope_sha256,
        )
        if self.evidence_envelope is None:
            raise AssertionError("validated closure requires evidence")
        if any(
            proof.kind is MappingProofKindV2.WRITE_BACK
            for proof in self.evidence_envelope.mapping_proofs
        ):
            raise ValueError(
                "CORE_SOLVE rejection must not contain future WRITE_BACK evidence"
            )
        if self.evidence_envelope.final_audits:
            raise ValueError("CORE_SOLVE rejection must precede final audit")
        if any(
            value is not None
            for value in (
                self.objective_spec,
                self.edit_sha256,
                self.final_audit_id,
                self.final_audit_sha256,
            )
        ):
            raise ValueError(
                "CORE_SOLVE rejection must not carry production edit or audit fields"
            )

        expected_certificate = (
            self.solve_result.certificate.certificate_sha256
            if isinstance(self.solve_result, ProvenUnsatResultV2)
            else None
        )
        if self.certificate_sha256 != expected_certificate:
            raise ValueError(
                "PROVEN_UNSAT certificate hash is not closed, or UNCERTIFIED "
                "carries a semantic certificate"
            )
        expected_code = (
            RejectionCodeV2.CORE_PROVEN_UNSAT
            if isinstance(self.solve_result, ProvenUnsatResultV2)
            else RejectionCodeV2.CORE_UNCERTIFIED
        )
        if any(finding.code is not expected_code for finding in self.findings):
            raise ValueError("CORE_SOLVE rejection code does not match core status")

    def _validate_write_back(self) -> None:
        self._forbid_pre_semantic_evidence()
        result = self._require_success_closure()
        if self.evidence_envelope is None:
            raise AssertionError("validated closure requires evidence")
        if self.evidence_envelope.final_audits or any(
            value is not None
            for value in (self.final_audit_id, self.final_audit_sha256)
        ):
            raise ValueError("WRITE_BACK rejection must precede final audit")
        if self.edit_sha256 is None:
            raise AssertionError("validated success closure requires edit hash")
        _validate_failed_write_back(
            self.evidence_envelope,
            edit_sha256=self.edit_sha256,
            subject_id=result.edit.subject_id,
        )
        if result.status.value != "CERTIFIED_SUCCESS":
            raise AssertionError("success closure returned a non-success result")

    def _validate_post_success_audit(
        self,
        required_status: AuditStatusV2,
    ) -> None:
        self._forbid_pre_semantic_evidence()
        result = self._require_success_closure()
        if self.evidence_envelope is None or self.edit_sha256 is None:
            raise AssertionError("validated success closure requires evidence and edit")
        if self.final_audit_id is None or self.final_audit_sha256 is None:
            raise ValueError(f"{self.stage.value} requires a selected final audit")
        _validate_verified_write_back(
            self.evidence_envelope,
            edit_sha256=self.edit_sha256,
            subject_id=result.edit.subject_id,
        )
        _validate_selected_audit(
            evidence_envelope=self.evidence_envelope,
            final_audit_id=self.final_audit_id,
            final_audit_sha256=self.final_audit_sha256,
            edit_sha256=self.edit_sha256,
            required_status=required_status,
        )

    def _require_success_closure(self) -> CertifiedSuccessResultV2:
        if (
            self.solve_result is None
            or self.solve_result.status is not SolveStatusV2.CERTIFIED_SUCCESS
        ):
            raise ValueError(
                f"{self.stage.value} rejection must preserve the core "
                "CERTIFIED_SUCCESS result; it cannot relabel PROVEN_UNSAT"
            )
        assert isinstance(self.solve_result, CertifiedSuccessResultV2)
        required = (
            self.semantic_problem_sha256,
            self.core_solver_config_sha256,
            self.objective_spec,
            self.solve_result_sha256,
            self.edit_sha256,
            self.certificate_sha256,
            self.evidence_envelope,
            self.evidence_envelope_sha256,
        )
        if any(value is None for value in required):
            raise ValueError(
                f"{self.stage.value} rejection requires complete success hash closure"
            )
        assert self.semantic_problem_sha256 is not None
        assert self.core_solver_config_sha256 is not None
        assert self.objective_spec is not None
        assert self.solve_result_sha256 is not None
        assert self.edit_sha256 is not None
        assert self.certificate_sha256 is not None
        assert self.evidence_envelope is not None
        assert self.evidence_envelope_sha256 is not None
        _validate_success_reference_closure(
            semantic_problem_sha256=self.semantic_problem_sha256,
            core_solver_config_sha256=self.core_solver_config_sha256,
            objective_spec=self.objective_spec,
            solve_result=self.solve_result,
            solve_result_sha256=self.solve_result_sha256,
            edit_sha256=self.edit_sha256,
            certificate_sha256=self.certificate_sha256,
            evidence_envelope=self.evidence_envelope,
            evidence_envelope_sha256=self.evidence_envelope_sha256,
        )
        return self.solve_result

    def _forbid_pre_semantic_evidence(self) -> None:
        if (
            self.pre_semantic_evidence is not None
            or self.pre_semantic_evidence_sha256 is not None
        ):
            raise ValueError(
                f"{self.stage.value} must use post-semantic EvidenceEnvelopeV2"
            )

    @property
    def owners(self) -> tuple[RejectionOwnerV2, ...]:
        return tuple(
            sorted(
                {
                    _derive_owner(self.stage, finding.classification)
                    for finding in self.findings
                },
                key=lambda owner: owner.value,
            )
        )

    @property
    def final_audit(self) -> FinalAuditOutcomeV2 | None:
        if self.final_audit_id is None or self.evidence_envelope is None:
            return None
        return next(
            (
                audit
                for audit in self.evidence_envelope.final_audits
                if audit.audit_id == self.final_audit_id
            ),
            None,
        )

    @property
    def rejection_record_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_REJECTION_HASH_DOMAIN)


TerminalRecordV2: TypeAlias = Annotated[
    PairRecordV2 | RejectionRecordV2,
    Field(discriminator="record_kind"),
]


class PublicationManifestV2(V2Model):
    """Canonical terminal set with exactly one outcome per request."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="publication-manifest")
    )
    records: tuple[TerminalRecordV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_and_validate_manifest(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="publication-manifest"):
            raise ValueError("publication-manifest schema identity must be fixed")
        request_ids = tuple(record.request_id for record in self.records)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError(
                "publication terminal request_id values must be globally unique"
            )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda record: record.request_id)),
        )
        return self

    @property
    def publication_manifest_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_MANIFEST_HASH_DOMAIN)


def _validate_success_reference_closure(
    *,
    semantic_problem_sha256: Sha256Digest,
    core_solver_config_sha256: Sha256Digest,
    objective_spec: ObjectiveSpecV2,
    solve_result: CertifiedSuccessResultV2,
    solve_result_sha256: Sha256Digest,
    edit_sha256: Sha256Digest,
    certificate_sha256: Sha256Digest,
    evidence_envelope: EvidenceEnvelopeV2,
    evidence_envelope_sha256: Sha256Digest,
) -> None:
    _validate_common_result_and_evidence_closure(
        semantic_problem_sha256=semantic_problem_sha256,
        core_solver_config_sha256=core_solver_config_sha256,
        solve_result=solve_result,
        solve_result_sha256=solve_result_sha256,
        evidence_envelope=evidence_envelope,
        evidence_envelope_sha256=evidence_envelope_sha256,
    )
    if objective_spec.mode is not ObjectiveModeV2.PRODUCTION:
        raise ValueError("publication requires a PRODUCTION ObjectiveSpecV2")
    if objective_spec.objective_spec_sha256 != (
        solve_result.objective_partition.objective_spec_sha256
    ):
        raise ValueError("production objective hash is not closed to solve result")
    if edit_sha256 != solve_result.edit.edit_sha256:
        raise ValueError("Canonical Edit hash is not closed to solve result")
    if certificate_sha256 != solve_result.certificate.certificate_sha256:
        raise ValueError("semantic certificate hash is not closed to solve result")
    if solve_result.edit.subject_id not in {
        binding.canonical_object_id
        for binding in evidence_envelope.native_object_bindings
    }:
        raise ValueError(
            "Canonical Edit subject requires a verified native object binding"
        )


def _validate_common_result_and_evidence_closure(
    *,
    semantic_problem_sha256: Sha256Digest | None,
    core_solver_config_sha256: Sha256Digest | None,
    solve_result: CanonicalSolveResultV2,
    solve_result_sha256: Sha256Digest | None,
    evidence_envelope: EvidenceEnvelopeV2 | None,
    evidence_envelope_sha256: Sha256Digest | None,
) -> None:
    required = (
        semantic_problem_sha256,
        core_solver_config_sha256,
        solve_result_sha256,
        evidence_envelope,
        evidence_envelope_sha256,
    )
    if any(value is None for value in required):
        raise ValueError(
            "result/evidence hash closure requires every digest and object"
        )
    assert semantic_problem_sha256 is not None
    assert core_solver_config_sha256 is not None
    assert solve_result_sha256 is not None
    assert evidence_envelope is not None
    assert evidence_envelope_sha256 is not None
    if semantic_problem_sha256 != solve_result.semantic_problem_sha256:
        raise ValueError("semantic problem hash is not closed to solve result")
    if semantic_problem_sha256 != evidence_envelope.semantic_problem_sha256:
        raise ValueError("semantic problem hash is not closed to evidence envelope")
    if core_solver_config_sha256 != (
        solve_result.core_solver_config.core_solver_config_sha256
    ):
        raise ValueError("core solver config hash is not closed to solve result")
    if solve_result_sha256 != solve_result.solve_result_sha256:
        raise ValueError("solve result hash is not closed")
    if evidence_envelope_sha256 != evidence_envelope.evidence_envelope_sha256:
        raise ValueError("evidence envelope hash is not closed")


def _validate_selected_audit(
    *,
    evidence_envelope: EvidenceEnvelopeV2,
    final_audit_id: CanonicalId,
    final_audit_sha256: Sha256Digest,
    edit_sha256: Sha256Digest,
    required_status: AuditStatusV2,
) -> None:
    if len(evidence_envelope.final_audits) != 1:
        raise ValueError("publication requires exactly one unambiguous final audit")
    audit = evidence_envelope.final_audits[0]
    if audit.audit_id != final_audit_id:
        raise ValueError("selected final audit ID is not closed")
    if audit.status is not required_status:
        raise ValueError(
            f"publication final audit must have {required_status.value} status"
        )
    if audit.canonical_edit_sha256 != edit_sha256:
        raise ValueError("final audit Canonical Edit hash is not closed")
    if final_audit_sha256 != canonical_sha256_v2(
        audit,
        domain=_FINAL_AUDIT_HASH_DOMAIN,
    ):
        raise ValueError("final audit hash is not closed")


def _validate_verified_write_back(
    evidence_envelope: EvidenceEnvelopeV2,
    *,
    edit_sha256: Sha256Digest,
    subject_id: CanonicalId,
) -> None:
    matching = _matching_write_back_proofs(
        evidence_envelope,
        edit_sha256=edit_sha256,
        subject_id=subject_id,
    )
    if not any(proof.status is MappingProofStatusV2.VERIFIED for proof in matching):
        raise ValueError(
            "post-write-back publication requires a matching VERIFIED WRITE_BACK proof"
        )
    if any(
        proof.status
        in {
            MappingProofStatusV2.REJECTED,
            MappingProofStatusV2.INCOMPLETE,
        }
        for proof in matching
    ):
        raise ValueError(
            "matching edit has a conflicting non-VERIFIED WRITE_BACK proof"
        )


def _validate_failed_write_back(
    evidence_envelope: EvidenceEnvelopeV2,
    *,
    edit_sha256: Sha256Digest,
    subject_id: CanonicalId,
) -> None:
    matching = _matching_write_back_proofs(
        evidence_envelope,
        edit_sha256=edit_sha256,
        subject_id=subject_id,
    )
    if not any(
        proof.status
        in {
            MappingProofStatusV2.REJECTED,
            MappingProofStatusV2.INCOMPLETE,
        }
        for proof in matching
    ):
        raise ValueError(
            "WRITE_BACK rejection requires a matching non-VERIFIED write-back proof"
        )
    if any(proof.status is MappingProofStatusV2.VERIFIED for proof in matching):
        raise ValueError(
            "WRITE_BACK rejection conflicts with a matching VERIFIED write-back proof"
        )


def _matching_write_back_proofs(
    evidence_envelope: EvidenceEnvelopeV2,
    *,
    edit_sha256: Sha256Digest,
    subject_id: CanonicalId,
) -> tuple[MappingProofV2, ...]:
    subject_binding = next(
        (
            binding
            for binding in evidence_envelope.native_object_bindings
            if binding.canonical_object_id == subject_id
        ),
        None,
    )
    if subject_binding is None:
        return ()
    return tuple(
        proof
        for proof in evidence_envelope.mapping_proofs
        if proof.kind is MappingProofKindV2.WRITE_BACK
        and proof.canonical_edit_sha256 == edit_sha256
        and subject_id in proof.canonical_ids
        and subject_binding.native_object_locator in proof.native_locators
    )


def _derive_owner(
    stage: RejectionStageV2,
    classification: FindingClassificationV2,
) -> RejectionOwnerV2:
    if stage is RejectionStageV2.PUBLICATION:
        return RejectionOwnerV2.PUBLICATION
    if classification is FindingClassificationV2.PLATFORM_EXECUTION_NOISE:
        return RejectionOwnerV2.PLATFORM_RUNTIME
    if classification is FindingClassificationV2.ADAPTER_MAPPING:
        if stage in {
            RejectionStageV2.WRITE_BACK,
            RejectionStageV2.FINAL_AUDIT,
        }:
            return RejectionOwnerV2.WRITE_BACK_ADAPTER
        return RejectionOwnerV2.SOURCE_ADAPTER
    if stage is RejectionStageV2.CORE_SOLVE:
        return RejectionOwnerV2.CORE_SOLVER
    return RejectionOwnerV2.CANONICAL_CONTRACT
