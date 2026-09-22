"""Strict immutable domain contracts for M4 semantic contrast datasets."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, ClassVar, Literal, Self, TypeAlias, TypeVar

from pydantic import Field, StrictBool, StrictInt, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    NonNegativeFiniteFloat,
    Sha256Digest,
)
from spatialcf.domain.counterfactual import CounterfactualSolveRequest, SceneStateEnvelope
from spatialcf.domain.constraints import Relation
from spatialcf.domain.definitions import DefinitionRef, HashBoundCanonicalModel
from spatialcf.domain.lineage import (
    CounterfactualLineage,
    NativeNotRequested,
    RuntimeProvenance,
    SemanticObjectReference,
)
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.domain.upright_se2 import UprightSE2ExactRational

__all__ = (
    "ArtifactInventoryEntry",
    "BackendProfile",
    "CandidateRecordInput",
    "CatalogSplit",
    "CounterfactualLineage",
    "DatasetManifest",
    "FrozenCandidateRecord",
    "FrozenSourceRecord",
    "NativeExecutionStatus",
    "NoncertifiedWitnessTerminal",
    "PairContent",
    "PolicyEligibilityEvidence",
    "PolicyRejectedTerminal",
    "ProvenUnsatTerminal",
    "PublicationPolicy",
    "PublishedPairTerminal",
    "RecordEnvelope",
    "SemanticContrastCatalog",
    "SemanticContrastCatalogInput",
    "SemanticContrastEvidence",
    "SemanticContrastReport",
    "SemanticObjectReference",
    "SourceByteReference",
    "SourceFileManifest",
    "SourceRecordIdentity",
    "SourceRecordInput",
    "TERMINAL_REASON_PRECEDENCE",
    "TerminalLedger",
    "TerminalRecord",
    "TerminalReasonStage",
    "TerminalStatus",
    "UnknownTerminal",
)

_ValueT = TypeVar("_ValueT")


def _require_sorted_unique(values: tuple[_ValueT, ...], label: str) -> None:
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted by canonical bytes")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicates")


def _validate_relative_path(value: str, label: str) -> None:
    if value.startswith("/") or PureWindowsPath(value).drive or "\\" in value or "//" in value:
        raise ValueError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} contains an unsafe or aliased component")


class BackendProfile(StrEnum):
    CARDINAL = "UPRIGHT_SE2_CARDINAL"
    CONTINUOUS = "UPRIGHT_SE2_CONTINUOUS"


class CatalogSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class TerminalStatus(StrEnum):
    PUBLISHED_PAIR = "PUBLISHED_PAIR"
    PROVEN_UNSAT = "PROVEN_UNSAT"
    UNKNOWN = "UNKNOWN"
    NONCERTIFIED_WITNESS = "NONCERTIFIED_WITNESS"
    POLICY_REJECTED = "POLICY_REJECTED"


class TerminalReasonStage(StrEnum):
    STRUCTURAL_VALIDATION = "STRUCTURAL_VALIDATION"
    PROFILE_SOURCE_VALIDATION = "PROFILE_SOURCE_VALIDATION"
    PUBLICATION_POLICY = "PUBLICATION_POLICY"
    BACKEND_SELECTION = "BACKEND_SELECTION"
    COMPILE_SOLVE_ASSEMBLY = "COMPILE_SOLVE_ASSEMBLY"


TERMINAL_REASON_PRECEDENCE = (
    TerminalReasonStage.STRUCTURAL_VALIDATION,
    TerminalReasonStage.PROFILE_SOURCE_VALIDATION,
    TerminalReasonStage.PUBLICATION_POLICY,
    TerminalReasonStage.BACKEND_SELECTION,
    TerminalReasonStage.COMPILE_SOLVE_ASSEMBLY,
)


class NativeExecutionStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"


class SourceRecordIdentity(CanonicalModel):
    kind: Literal["SOURCE_RECORD_IDENTITY"] = "SOURCE_RECORD_IDENTITY"
    version: Literal["1"] = "1"
    dataset_id: CanonicalId
    revision_id: CanonicalId
    record_id: CanonicalId


class SourceByteReference(CanonicalModel):
    kind: Literal["SOURCE_BYTE_REFERENCE"] = "SOURCE_BYTE_REFERENCE"
    version: Literal["1"] = "1"
    relative_path: CanonicalId
    byte_length: StrictInt
    byte_sha256: Sha256Digest

    @model_validator(mode="after")
    def _valid_source_bytes(self) -> Self:
        _validate_relative_path(self.relative_path, "source byte path")
        if self.byte_length <= 0:
            raise ValueError("source byte length must be positive")
        return self


class SourceFileManifest(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/source-files/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "source_files_manifest_sha256"

    kind: Literal["SEMANTIC_CONTRAST_SOURCE_FILES"] = (
        "SEMANTIC_CONTRAST_SOURCE_FILES"
    )
    version: Literal["1"] = "1"
    files: tuple[SourceByteReference, ...]
    source_files_manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _valid_files(self) -> Self:
        if not self.files:
            raise ValueError("source file manifest must not be empty")
        _require_sorted_unique(self.files, "source files")
        paths = tuple(item.relative_path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("source files must not contain path aliases")
        return self


class SourceRecordInput(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/source-record-input/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "source_record_input_sha256"

    kind: Literal["SEMANTIC_CONTRAST_SOURCE_RECORD_INPUT"] = (
        "SEMANTIC_CONTRAST_SOURCE_RECORD_INPUT"
    )
    version: Literal["1"] = "1"
    identity: SourceRecordIdentity
    source_record_bytes: SourceByteReference
    source_files: SourceFileManifest
    scene_state: SceneStateEnvelope
    scene_state_sha256: Sha256Digest
    source_group: CanonicalId
    source_record_input_sha256: Sha256Digest

    @model_validator(mode="after")
    def _bind_complete_scene(self) -> Self:
        if self.scene_state_sha256 != self.scene_state.scene_state_sha256:
            raise ValueError("source record does not bind its complete scene state")
        if self.source_group != self.scene_state.base_scene_payload.scene_id:
            raise ValueError("source group must equal the base scene id")
        paths = {item.relative_path for item in self.source_files.files}
        if self.source_record_bytes.relative_path in paths:
            raise ValueError("source record bytes and source files must be distinct")
        return self


class CandidateRecordInput(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/candidate-input/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "candidate_record_input_sha256"

    kind: Literal["SEMANTIC_CONTRAST_CANDIDATE_INPUT"] = (
        "SEMANTIC_CONTRAST_CANDIDATE_INPUT"
    )
    version: Literal["1"] = "1"
    candidate_id: CanonicalId
    source_identity: SourceRecordIdentity
    solve_request: CounterfactualSolveRequest
    solve_request_sha256: Sha256Digest
    backend_profile: BackendProfile
    candidate_record_input_sha256: Sha256Digest

    @model_validator(mode="after")
    def _bind_request(self) -> Self:
        if self.solve_request_sha256 != self.solve_request.solve_request_sha256:
            raise ValueError("candidate does not bind its complete solve request")
        return self


class PublicationPolicy(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/publication-policy/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "publication_policy_sha256"

    kind: Literal["SEMANTIC_CONTRAST_PUBLICATION_POLICY"] = (
        "SEMANTIC_CONTRAST_PUBLICATION_POLICY"
    )
    version: Literal["1"] = "1"
    dataset_kind: Literal["spatialcf/semantic-contrast-dataset@1"] = (
        "spatialcf/semantic-contrast-dataset@1"
    )
    publish_eligible_certified_members: Literal[True] = True
    require_opposite_target_relation: Literal[True] = True
    allow_backfill: Literal[False] = False
    native_execution: Literal[NativeExecutionStatus.NOT_REQUESTED] = (
        NativeExecutionStatus.NOT_REQUESTED
    )
    maximum_catalog_size: StrictInt
    eligibility_claim_definition_ref: DefinitionRef
    reason_precedence: tuple[TerminalReasonStage, ...] = TERMINAL_REASON_PRECEDENCE
    publication_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _positive_limit(self) -> Self:
        if self.maximum_catalog_size <= 0:
            raise ValueError("maximum catalog size must be positive")
        if self.reason_precedence != TERMINAL_REASON_PRECEDENCE:
            raise ValueError("terminal reason precedence is frozen for M4")
        return self


class SemanticContrastCatalogInput(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/catalog-input/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "catalog_input_sha256"

    kind: Literal["SEMANTIC_CONTRAST_CATALOG_INPUT"] = (
        "SEMANTIC_CONTRAST_CATALOG_INPUT"
    )
    version: Literal["1"] = "1"
    profile_version: Literal["spatialcf/semantic-contrast@1"] = (
        "spatialcf/semantic-contrast@1"
    )
    provenance_scope: Literal["LOCAL_SOURCE_SNAPSHOT"] = "LOCAL_SOURCE_SNAPSHOT"
    sources: tuple[SourceRecordInput, ...]
    candidates: tuple[CandidateRecordInput, ...]
    publication_policy: PublicationPolicy
    catalog_input_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Normalize the two unordered transport arrays before self-hashing."""

        if "sources" in values:
            values["sources"] = tuple(
                sorted(
                    values["sources"],
                    key=lambda item: canonical_json_bytes(item.identity),
                )
            )
        if "candidates" in values:
            values["candidates"] = tuple(
                sorted(
                    values["candidates"],
                    key=lambda item: (
                        canonical_json_bytes(item.source_identity),
                        canonical_json_bytes(item.candidate_id),
                    ),
                )
            )
        return super().seal(**values)

    @model_validator(mode="after")
    def _validate_local_catalog(self) -> Self:
        identities = tuple(canonical_json_bytes(item.identity) for item in self.sources)
        if identities != tuple(sorted(identities)):
            raise ValueError("catalog sources must be in canonical identity order")
        if len(identities) != len(set(identities)):
            raise ValueError("catalog source identities must be unique")
        source_record_paths = tuple(
            item.source_record_bytes.relative_path for item in self.sources
        )
        if len(source_record_paths) != len(set(source_record_paths)):
            raise ValueError("catalog source-record paths must be unique")
        file_identities: dict[str, tuple[int, str]] = {}
        for source in self.sources:
            for file_ref in source.source_files.files:
                if file_ref.relative_path in source_record_paths:
                    raise ValueError(
                        "source-file paths must not alias source-record paths"
                    )
                identity = (file_ref.byte_length, file_ref.byte_sha256)
                previous = file_identities.setdefault(file_ref.relative_path, identity)
                if previous != identity:
                    raise ValueError("one source-file path has conflicting byte identity")
        source_by_identity = {
            canonical_json_bytes(item.identity): item for item in self.sources
        }
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        candidate_order = tuple(
            (
                canonical_json_bytes(item.source_identity),
                canonical_json_bytes(item.candidate_id),
            )
            for item in self.candidates
        )
        if candidate_order != tuple(sorted(candidate_order)):
            raise ValueError("catalog candidates must be in canonical source/id order")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("catalog candidate ids must be unique")
        tuples: set[tuple[bytes, str, BackendProfile]] = set()
        for candidate in self.candidates:
            source = source_by_identity.get(canonical_json_bytes(candidate.source_identity))
            if source is None:
                raise ValueError("candidate references an unknown source record")
            request_scene = candidate.solve_request.semantic_problem.scene_state
            if canonical_json_bytes(request_scene) != canonical_json_bytes(source.scene_state):
                raise ValueError("candidate request before state differs from its source")
            key = (
                canonical_json_bytes(candidate.source_identity),
                candidate.solve_request_sha256,
                candidate.backend_profile,
            )
            if key in tuples:
                raise ValueError("duplicate source/request/backend candidate tuple")
            tuples.add(key)
        if len(self.candidates) > self.publication_policy.maximum_catalog_size:
            raise ValueError("catalog exceeds its declared maximum size")
        return self


class PolicyEligibilityEvidence(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/policy-evidence/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "policy_evidence_sha256"

    kind: Literal["SEMANTIC_CONTRAST_POLICY_EVIDENCE"] = (
        "SEMANTIC_CONTRAST_POLICY_EVIDENCE"
    )
    version: Literal["1"] = "1"
    eligible: StrictBool
    clause_definition_ref: DefinitionRef
    solve_request_sha256: Sha256Digest
    source_record_input_sha256: Sha256Digest
    publication_policy_sha256: Sha256Digest
    policy_evidence_sha256: Sha256Digest


class FrozenSourceRecord(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/frozen-source/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "frozen_source_record_sha256"

    kind: Literal["SEMANTIC_CONTRAST_FROZEN_SOURCE"] = (
        "SEMANTIC_CONTRAST_FROZEN_SOURCE"
    )
    version: Literal["1"] = "1"
    source: SemanticObjectReference[Literal["SourceRecordInput"]]
    identity: SourceRecordIdentity
    source_group: CanonicalId
    scene_state_sha256: Sha256Digest
    split: CatalogSplit
    frozen_source_record_sha256: Sha256Digest


class FrozenCandidateRecord(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/frozen-candidate/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "frozen_candidate_record_sha256"

    kind: Literal["SEMANTIC_CONTRAST_FROZEN_CANDIDATE"] = (
        "SEMANTIC_CONTRAST_FROZEN_CANDIDATE"
    )
    version: Literal["1"] = "1"
    candidate: SemanticObjectReference[Literal["CandidateRecordInput"]]
    candidate_id: CanonicalId
    source_identity: SourceRecordIdentity
    source_record_input_sha256: Sha256Digest
    solve_request_sha256: Sha256Digest
    backend_profile: BackendProfile
    source_group: CanonicalId
    split: CatalogSplit
    policy_evidence: PolicyEligibilityEvidence
    frozen_candidate_record_sha256: Sha256Digest

    @model_validator(mode="after")
    def _bind_candidate_decisions(self) -> Self:
        if self.policy_evidence.solve_request_sha256 != self.solve_request_sha256:
            raise ValueError("policy evidence does not bind the candidate request")
        if self.policy_evidence.source_record_input_sha256 != (
            self.source_record_input_sha256
        ):
            raise ValueError("policy evidence does not bind the candidate source")
        return self


class SemanticContrastCatalog(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/frozen-catalog/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_contrast_catalog_sha256"

    kind: Literal["SEMANTIC_CONTRAST_CATALOG"] = "SEMANTIC_CONTRAST_CATALOG"
    version: Literal["1"] = "1"
    normalized_input_byte_length: StrictInt
    normalized_input_byte_sha256: Sha256Digest
    catalog_input: SemanticObjectReference[Literal["SemanticContrastCatalogInput"]]
    sources: tuple[FrozenSourceRecord, ...]
    candidates: tuple[FrozenCandidateRecord, ...]
    publication_policy: SemanticObjectReference[Literal["PublicationPolicy"]]
    runtime_provenance: SemanticObjectReference[Literal["RuntimeProvenance"]]
    semantic_contrast_catalog_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_frozen_catalog(self) -> Self:
        if self.normalized_input_byte_length <= 0:
            raise ValueError("normalized catalog input must not be empty")
        source_order = tuple(
            canonical_json_bytes(item.identity) for item in self.sources
        )
        if source_order != tuple(sorted(source_order)) or len(source_order) != len(
            set(source_order)
        ):
            raise ValueError("frozen sources must be in unique canonical identity order")
        candidate_order = tuple(
            (
                canonical_json_bytes(item.source_identity),
                canonical_json_bytes(item.candidate_id),
            )
            for item in self.candidates
        )
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_order != tuple(sorted(candidate_order)) or len(candidate_order) != len(
            set(candidate_order)
        ):
            raise ValueError("frozen candidates must be in unique canonical order")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("frozen candidate ids must be unique")
        source_map = {
            canonical_json_bytes(item.identity): item for item in self.sources
        }
        group_splits: dict[str, CatalogSplit] = {}
        for frozen in self.sources:
            previous = group_splits.setdefault(frozen.source_group, frozen.split)
            if previous != frozen.split:
                raise ValueError("source group aliases cannot cross splits")
        for item in self.candidates:
            source = source_map.get(canonical_json_bytes(item.source_identity))
            if source is None:
                raise ValueError("frozen candidate source is missing")
            if (
                item.source_record_input_sha256 != source.source.semantic_sha256
                or item.source_group != source.source_group
                or item.split != source.split
            ):
                raise ValueError("candidate source group or split differs from source")
            evidence = item.policy_evidence
            if (
                evidence.source_record_input_sha256
                != source.source.semantic_sha256
                or evidence.publication_policy_sha256
                != self.publication_policy.semantic_sha256
            ):
                raise ValueError("candidate policy evidence binds the wrong roots")
        return self


class SemanticContrastEvidence(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/semantic-evidence/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_contrast_evidence_sha256"

    kind: Literal["SEMANTIC_CONTRAST_EVIDENCE"] = "SEMANTIC_CONTRAST_EVIDENCE"
    version: Literal["1"] = "1"
    subject_id: CanonicalId
    reference_id: CanonicalId
    before_relation: Relation
    after_relation: Relation
    accepted_claim_definition_ref: DefinitionRef
    objective_lower_bound: NonNegativeFiniteFloat
    objective_upper_bound: NonNegativeFiniteFloat
    requested_gap: UprightSE2ExactRational
    proof_material_sha256: Sha256Digest
    semantic_contrast_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def _opposing_labels(self) -> Self:
        if self.after_relation != self.before_relation.opposite:
            raise ValueError("semantic pair labels must be exact opposites")
        if self.objective_lower_bound > self.objective_upper_bound:
            raise ValueError("objective lower bound exceeds upper bound")
        if self.requested_gap.as_fraction < 0:
            raise ValueError("requested gap must be non-negative")
        # Claim strength and interval evidence are decided by the retained checker.
        # Never reinterpret its exact rational policy using float subtraction here.
        return self


class PairContent(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/pair-content/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "pair_content_sha256"

    kind: Literal["SEMANTIC_CONTRAST_PAIR_CONTENT"] = (
        "SEMANTIC_CONTRAST_PAIR_CONTENT"
    )
    version: Literal["1"] = "1"
    candidate_id: CanonicalId
    source_identity: SourceRecordIdentity
    source_group: CanonicalId
    split: CatalogSplit
    before_scene_state: SemanticObjectReference[Literal["SceneStateEnvelope"]]
    after_scene_state: SemanticObjectReference[Literal["SceneStateEnvelope"]]
    solve_request: SemanticObjectReference[Literal["CounterfactualSolveRequest"]]
    program: SemanticObjectReference[Literal["EditProgram"]]
    grounded_obligations: SemanticObjectReference[Literal["GroundedObligationSet"]]
    result: SemanticObjectReference[Literal["CertifiedSolutionResult"]]
    certificate: SemanticObjectReference[Literal["CertifiedSolutionCertificate"]]
    semantic_evidence: SemanticContrastEvidence
    pair_content_sha256: Sha256Digest


class RecordEnvelope(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/record-envelope/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "record_envelope_sha256"

    kind: Literal["SEMANTIC_CONTRAST_RECORD_ENVELOPE"] = (
        "SEMANTIC_CONTRAST_RECORD_ENVELOPE"
    )
    version: Literal["1"] = "1"
    content: SemanticObjectReference[Literal["PairContent"]]
    pair_content_sha256: Sha256Digest
    lineage: SemanticObjectReference[Literal["CounterfactualLineage"]]
    counterfactual_lineage_sha256: Sha256Digest
    record_envelope_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_envelope(self) -> Self:
        if self.pair_content_sha256 != self.content.semantic_sha256:
            raise ValueError("record envelope does not bind its pair content")
        if self.counterfactual_lineage_sha256 != (
            self.lineage.semantic_sha256
        ):
            raise ValueError("record envelope does not bind its lineage")
        return self


class PublishedPairTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-published/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_record_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL"] = "SEMANTIC_CONTRAST_TERMINAL"
    version: Literal["1"] = "1"
    status: Literal[TerminalStatus.PUBLISHED_PAIR] = TerminalStatus.PUBLISHED_PAIR
    candidate_id: CanonicalId
    solve_request_sha256: Sha256Digest
    source_record_input_sha256: Sha256Digest
    record_envelope_sha256: Sha256Digest
    pair_content_sha256: Sha256Digest
    counterfactual_lineage_sha256: Sha256Digest
    result: SemanticObjectReference[Literal["CertifiedSolutionResult"]]
    certificate: SemanticObjectReference[Literal["CertifiedSolutionCertificate"]]
    program: SemanticObjectReference[Literal["EditProgram"]]
    grounded_obligations: SemanticObjectReference[Literal["GroundedObligationSet"]]
    terminal_record_sha256: Sha256Digest


class ProvenUnsatTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-unsat/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_record_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL"] = "SEMANTIC_CONTRAST_TERMINAL"
    version: Literal["1"] = "1"
    status: Literal[TerminalStatus.PROVEN_UNSAT] = TerminalStatus.PROVEN_UNSAT
    candidate_id: CanonicalId
    source_record_input_sha256: Sha256Digest
    selection: SemanticObjectReference[Literal["BackendSelectionRecord"]]
    submission: SemanticObjectReference[Literal["BackendCompleteUnsatEvidence"]]
    checked_outcome: SemanticObjectReference[Literal["CheckedProofOutcome"]]
    dispatch: SemanticObjectReference[Literal["VerifierDispatchRecord"]]
    result: SemanticObjectReference[Literal["ProvenUnsatResult"]]
    certificate: SemanticObjectReference[Literal["ProvenUnsatCertificate"]]
    terminal_record_sha256: Sha256Digest


class UnknownTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-unknown/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_record_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL"] = "SEMANTIC_CONTRAST_TERMINAL"
    version: Literal["1"] = "1"
    status: Literal[TerminalStatus.UNKNOWN] = TerminalStatus.UNKNOWN
    candidate_id: CanonicalId
    source_record_input_sha256: Sha256Digest
    selection_disposition: Literal["SELECTED", "NO_SELECTION"]
    selection: SemanticObjectReference[Literal["BackendSelectionRecord"]]
    submission: SemanticObjectReference[Literal["BackendUnknownEvidence"]] | None = None
    checked_outcome: SemanticObjectReference[Literal["CheckedProofOutcome"]] | None = None
    dispatch: SemanticObjectReference[Literal["VerifierDispatchRecord"]] | None = None
    result: SemanticObjectReference[Literal["UnknownResult"]]
    terminal_record_sha256: Sha256Digest

    @model_validator(mode="after")
    def _unknown_evidence(self) -> Self:
        checked_pair = self.checked_outcome is not None or self.dispatch is not None
        if (self.checked_outcome is None) != (self.dispatch is None):
            raise ValueError("UNKNOWN checked outcome and dispatch must appear together")
        if self.selection_disposition == "NO_SELECTION":
            if self.submission is not None or checked_pair:
                raise ValueError("NO_SELECTION UNKNOWN forbids submitted/checker evidence")
        elif self.submission is None:
            raise ValueError("selected UNKNOWN requires backend unknown evidence")
        elif not checked_pair:
            raise ValueError("selected UNKNOWN requires fresh checker evidence")
        return self


class NoncertifiedWitnessTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-witness/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_record_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL"] = "SEMANTIC_CONTRAST_TERMINAL"
    version: Literal["1"] = "1"
    status: Literal[TerminalStatus.NONCERTIFIED_WITNESS] = (
        TerminalStatus.NONCERTIFIED_WITNESS
    )
    candidate_id: CanonicalId
    source_record_input_sha256: Sha256Digest
    selection: SemanticObjectReference[Literal["BackendSelectionRecord"]]
    submission: SemanticObjectReference[Literal["BackendProposalSubmission"]]
    checked_outcome: SemanticObjectReference[Literal["CheckedProofOutcome"]]
    dispatch: SemanticObjectReference[Literal["VerifierDispatchRecord"]]
    result: SemanticObjectReference[Literal["NoncertifiedWitnessResult"]]
    terminal_record_sha256: Sha256Digest


class PolicyRejectedTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-policy/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_record_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL"] = "SEMANTIC_CONTRAST_TERMINAL"
    version: Literal["1"] = "1"
    status: Literal[TerminalStatus.POLICY_REJECTED] = TerminalStatus.POLICY_REJECTED
    candidate_id: CanonicalId
    source_record_input_sha256: Sha256Digest
    solve_request: SemanticObjectReference[Literal["CounterfactualSolveRequest"]]
    publication_policy_sha256: Sha256Digest
    policy_evidence: PolicyEligibilityEvidence
    terminal_record_sha256: Sha256Digest

    @model_validator(mode="after")
    def _policy_rejection(self) -> Self:
        if self.policy_evidence.eligible:
            raise ValueError("POLICY_REJECTED requires ineligible policy evidence")
        if self.policy_evidence.solve_request_sha256 != self.solve_request.semantic_sha256:
            raise ValueError("policy rejection binds the wrong request")
        if self.policy_evidence.source_record_input_sha256 != self.source_record_input_sha256:
            raise ValueError("policy rejection binds the wrong source")
        if self.policy_evidence.publication_policy_sha256 != self.publication_policy_sha256:
            raise ValueError("policy rejection binds the wrong policy")
        return self


TerminalRecord: TypeAlias = Annotated[
    PublishedPairTerminal
    | ProvenUnsatTerminal
    | UnknownTerminal
    | NoncertifiedWitnessTerminal
    | PolicyRejectedTerminal,
    Field(discriminator="status"),
]


class TerminalLedger(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/terminal-ledger/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_ledger_sha256"
    kind: Literal["SEMANTIC_CONTRAST_TERMINAL_LEDGER"] = (
        "SEMANTIC_CONTRAST_TERMINAL_LEDGER"
    )
    version: Literal["1"] = "1"
    semantic_contrast_catalog_sha256: Sha256Digest
    ordered_candidate_ids: tuple[CanonicalId, ...]
    terminals: tuple[TerminalRecord, ...]
    terminal_ledger_sha256: Sha256Digest

    @model_validator(mode="after")
    def _one_ordered_terminal(self) -> Self:
        if len(set(self.ordered_candidate_ids)) != len(self.ordered_candidate_ids):
            raise ValueError("terminal ledger candidate ids must be unique")
        if tuple(item.candidate_id for item in self.terminals) != self.ordered_candidate_ids:
            raise ValueError("terminal order must equal frozen catalog order")
        return self


class SemanticContrastReport(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/report/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_contrast_report_sha256"
    kind: Literal["SEMANTIC_CONTRAST_REPORT"] = "SEMANTIC_CONTRAST_REPORT"
    version: Literal["1"] = "1"
    native_execution: NativeNotRequested = Field(default_factory=NativeNotRequested)
    dataset_kind: Literal["spatialcf/semantic-contrast-dataset@1"] = (
        "spatialcf/semantic-contrast-dataset@1"
    )
    semantic_contrast_catalog_sha256: Sha256Digest
    terminal_ledger_sha256: Sha256Digest
    ordered_record_envelope_sha256: tuple[Sha256Digest, ...]
    candidate_count: StrictInt
    pair_count: StrictInt
    proven_unsat_count: StrictInt
    unknown_count: StrictInt
    noncertified_witness_count: StrictInt
    policy_rejected_count: StrictInt
    semantic_contrast_report_sha256: Sha256Digest

    @model_validator(mode="after")
    def _counts_close(self) -> Self:
        counts = (
            self.pair_count,
            self.proven_unsat_count,
            self.unknown_count,
            self.noncertified_witness_count,
            self.policy_rejected_count,
        )
        if self.candidate_count < 0 or any(value < 0 for value in counts):
            raise ValueError("report counts must be non-negative")
        if sum(counts) != self.candidate_count:
            raise ValueError("report terminal counts must cover the catalog")
        if len(self.ordered_record_envelope_sha256) != self.pair_count:
            raise ValueError("report record roots must equal the published pair count")
        if len(set(self.ordered_record_envelope_sha256)) != len(
            self.ordered_record_envelope_sha256
        ):
            raise ValueError("report record roots must be unique")
        return self


class ArtifactInventoryEntry(CanonicalModel):
    kind: Literal["ARTIFACT_INVENTORY_ENTRY"] = "ARTIFACT_INVENTORY_ENTRY"
    version: Literal["1"] = "1"
    relative_path: CanonicalId
    payload_kind: Literal["TYPED_JSON", "SOURCE_BYTES"]
    typed_object_kind: CanonicalId | None = None
    typed_object_sha256: Sha256Digest | None = None
    byte_length: StrictInt
    byte_sha256: Sha256Digest

    @model_validator(mode="after")
    def _inventory_identity(self) -> Self:
        _validate_relative_path(self.relative_path, "artifact inventory path")
        if self.relative_path == "manifest.json":
            raise ValueError("manifest inventory must exclude the manifest itself")
        if self.byte_length < 0:
            raise ValueError("artifact byte length must be non-negative")
        semantic_pair = self.typed_object_kind is not None or self.typed_object_sha256 is not None
        if (self.typed_object_kind is None) != (self.typed_object_sha256 is None):
            raise ValueError("typed object kind and semantic hash must appear together")
        if self.payload_kind == "TYPED_JSON" and not semantic_pair:
            raise ValueError("typed JSON inventory requires a semantic identity")
        if self.payload_kind == "SOURCE_BYTES" and semantic_pair:
            raise ValueError("source bytes cannot masquerade as a typed object")
        return self


class DatasetManifest(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/dataset-manifest/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "dataset_manifest_sha256"
    kind: Literal["SEMANTIC_CONTRAST_DATASET_MANIFEST"] = (
        "SEMANTIC_CONTRAST_DATASET_MANIFEST"
    )
    version: Literal["1"] = "1"
    native_execution: NativeNotRequested = Field(default_factory=NativeNotRequested)
    dataset_kind: Literal["spatialcf/semantic-contrast-dataset@1"] = (
        "spatialcf/semantic-contrast-dataset@1"
    )
    semantic_contrast_catalog_sha256: Sha256Digest
    terminal_ledger_sha256: Sha256Digest
    semantic_contrast_report_sha256: Sha256Digest
    runtime_provenance_sha256: Sha256Digest
    ordered_record_envelope_sha256: tuple[Sha256Digest, ...]
    inventory: tuple[ArtifactInventoryEntry, ...]
    dataset_manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_manifest_root(self) -> Self:
        _require_sorted_unique(self.inventory, "artifact inventory")
        paths = tuple(item.relative_path for item in self.inventory)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact inventory paths must be unique")
        required = {"catalog.json", "terminals.json", "report.json"}
        if not required.issubset(paths):
            raise ValueError("manifest inventory omits a required dataset root")
        if len(set(self.ordered_record_envelope_sha256)) != len(
            self.ordered_record_envelope_sha256
        ):
            raise ValueError("manifest record envelope roots must be unique")
        return self
