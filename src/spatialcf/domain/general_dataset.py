"""Versioned CPU snapshot and dataset contracts for M5/M6 capabilities."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    NonNegativeFiniteFloat,
    Sha256Digest,
)
from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
    EditProgram,
    SceneStateEnvelope,
)
from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.outcomes import (
    BackendSelectionRecord,
    BackendSubmission,
    CheckedProofOutcome,
    CounterfactualCertificate,
    CounterfactualSolveResult,
    VerifierDispatchRecord,
)
from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    PreservationInvariant,
    GroundedObligationSet,
)
from spatialcf.domain.rigid_se3 import (
    RigidSE3Domain,
    RigidSE3ExactSourceFacts,
    RigidSE3Limits,
    RigidSE3ObjectivePolicy,
    RigidSE3SourceCells,
    RigidSE3WitnessHint,
)
from spatialcf.domain.scene import CanonicalScene
from spatialcf.domain.semantic_place import (
    SemanticPlaceCavityFact,
    SemanticPlaceInterval,
    SemanticPlaceLimits,
)

_PREFIX = "spatialcf/general-dataset/1.0"
Profile = Literal["spatialcf/semantic_place@1", "spatialcf/rigid_se3_multi@1"]
Split = Literal["train", "dev", "test"]


class _Snapshot(HashBoundCanonicalModel):
    SELF_DIGEST_FIELD: ClassVar[str] = "source_snapshot_sha256"
    source_id: CanonicalId
    dataset_id: CanonicalId
    revision_id: CanonicalId
    scene: CanonicalScene
    source_snapshot_sha256: Sha256Digest


class PlacementSnapshot(_Snapshot):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/placement-snapshot"
    kind: Literal["PLACEMENT_SNAPSHOT"] = "PLACEMENT_SNAPSHOT"
    cavities: tuple[SemanticPlaceCavityFact, ...] = ()


class RigidSnapshot(_Snapshot):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/rigid-snapshot"
    kind: Literal["RIGID_SNAPSHOT"] = "RIGID_SNAPSHOT"
    facts: RigidSE3ExactSourceFacts | RigidSE3SourceCells


Snapshot = Annotated[PlacementSnapshot | RigidSnapshot, Field(discriminator="kind")]


class _Task(HashBoundCanonicalModel):
    SELF_DIGEST_FIELD: ClassVar[str] = "task_sha256"
    candidate_id: CanonicalId
    source_id: CanonicalId
    backend_enabled: StrictBool = True
    task_sha256: Sha256Digest


class PlacementTask(_Task):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/placement-task"
    kind: Literal["PLACEMENT_TASK"] = "PLACEMENT_TASK"
    subject_id: CanonicalId
    operation: Literal["PLACE_ON", "PLACE_IN"]
    target_ids: tuple[CanonicalId, ...] = Field(min_length=1)
    x: SemanticPlaceInterval
    y: SemanticPlaceInterval
    z: SemanticPlaceInterval
    support_margin_m: NonNegativeFiniteFloat = 0.0
    lateral_margin_m: NonNegativeFiniteFloat = 0.0
    top_margin_m: NonNegativeFiniteFloat = 0.0
    limits: SemanticPlaceLimits = SemanticPlaceLimits()


class RigidTask(_Task):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/rigid-task"
    kind: Literal["RIGID_TASK"] = "RIGID_TASK"
    domain: RigidSE3Domain
    objective: RigidSE3ObjectivePolicy
    after_goal: AfterGoal
    before_preconditions: tuple[BeforePrecondition, ...] = ()
    preservation_invariants: tuple[PreservationInvariant, ...] = ()
    witness_hints: tuple[RigidSE3WitnessHint, ...] = ()
    limits: RigidSE3Limits = RigidSE3Limits()

    @model_validator(mode="after")
    def _nonconstant_goal(self) -> Self:
        if isinstance(self.after_goal.formula, bool):
            raise ValueError("rigid dataset tasks require a nonconstant goal")
        return self


Task = Annotated[PlacementTask | RigidTask, Field(discriminator="kind")]


class GeneralPublicationPolicy(CanonicalModel):
    version: Literal["1"] = "1"
    publish_all_certified: Literal[True] = True
    require_goal_transition: Literal[True] = True
    allow_backfill: Literal[False] = False
    native_execution: Literal["NOT_REQUESTED"] = "NOT_REQUESTED"
    maximum_tasks: Annotated[StrictInt, Field(gt=0, le=10_000)] = 1000


class GeneralDatasetInput(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/input"
    SELF_DIGEST_FIELD: ClassVar[str] = "input_sha256"
    kind: Literal["GENERAL_DATASET_INPUT"] = "GENERAL_DATASET_INPUT"
    version: Literal["1"] = "1"
    sources: tuple[Snapshot, ...]
    tasks: tuple[Task, ...]
    policy: GeneralPublicationPolicy = GeneralPublicationPolicy()
    input_sha256: Sha256Digest

    @model_validator(mode="after")
    def _closed_catalog(self) -> Self:
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("duplicate source identity")
        if len({task.candidate_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate candidate identity")
        if len(self.tasks) > self.policy.maximum_tasks:
            raise ValueError("catalog exceeds maximum_tasks")
        for task in self.tasks:
            source = sources.get(task.source_id)
            if source is None:
                raise ValueError("task source is missing")
            if isinstance(task, PlacementTask) != isinstance(source, PlacementSnapshot):
                raise ValueError("task and source profile differ")
        if set(sources) != {task.source_id for task in self.tasks}:
            raise ValueError("catalog contains unreferenced sources")
        return self


class GeneralFrozenCandidate(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/candidate"
    SELF_DIGEST_FIELD: ClassVar[str] = "candidate_sha256"
    candidate_id: CanonicalId
    source_id: CanonicalId
    source_snapshot_sha256: Sha256Digest
    task_sha256: Sha256Digest
    profile: Profile
    split: Split
    request: CounterfactualSolveRequest
    selection: BackendSelectionRecord
    candidate_sha256: Sha256Digest

    @model_validator(mode="after")
    def _bind_selection(self) -> Self:
        if self.request.semantic_problem.action_space_profile_ref != self.profile:
            raise ValueError("candidate profile does not bind request")
        if self.selection.solve_request_sha256 != self.request.solve_request_sha256:
            raise ValueError("candidate selection does not bind request")
        return self


class GeneralFrozenCatalog(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/catalog"
    SELF_DIGEST_FIELD: ClassVar[str] = "catalog_sha256"
    kind: Literal["GENERAL_DATASET_CATALOG"] = "GENERAL_DATASET_CATALOG"
    version: Literal["1"] = "1"
    input_sha256: Sha256Digest
    runtime_provenance_sha256: Sha256Digest
    candidates: tuple[GeneralFrozenCandidate, ...]
    catalog_sha256: Sha256Digest

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        if len({row.candidate_id for row in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate frozen candidate")
        return self


class RetainedAssembly(CanonicalModel):
    """All six assembler outputs, retained without another proof interpretation."""

    checked_proof_outcome: CheckedProofOutcome | None
    verifier_dispatch_record: VerifierDispatchRecord | None
    certificate: CounterfactualCertificate | None
    result: CounterfactualSolveResult
    program: EditProgram | None
    grounded_obligations: GroundedObligationSet | None


TerminalStatus = Literal[
    "PUBLISHED_PAIR", "PROVEN_UNSAT", "UNKNOWN", "NONCERTIFIED_WITNESS"
]


class GeneralTerminal(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/terminal"
    SELF_DIGEST_FIELD: ClassVar[str] = "terminal_sha256"
    candidate_id: CanonicalId
    candidate_sha256: Sha256Digest
    status: TerminalStatus
    selection: BackendSelectionRecord
    submission: BackendSubmission | None
    assembly: RetainedAssembly
    record_sha256: Sha256Digest | None
    terminal_sha256: Sha256Digest

    @model_validator(mode="after")
    def _terminal_shape(self) -> Self:
        expected = self.assembly.result.structural_outcome_class
        if expected == "CERTIFIED_SOLUTION":
            expected = "PUBLISHED_PAIR"
        if self.status != expected:
            raise ValueError("terminal status differs from result")
        if (self.status == "PUBLISHED_PAIR") != (self.record_sha256 is not None):
            raise ValueError("only certified terminals have records")
        if (self.selection.selection_disposition == "SELECTED") != (
            self.submission is not None
        ):
            raise ValueError("selection and submission presence differ")
        return self


class GeneralTerminalLedger(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/terminals"
    SELF_DIGEST_FIELD: ClassVar[str] = "ledger_sha256"
    catalog_sha256: Sha256Digest
    terminals: tuple[GeneralTerminal, ...]
    ledger_sha256: Sha256Digest


class GeneralPairContent(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/pair-content"
    SELF_DIGEST_FIELD: ClassVar[str] = "content_sha256"
    candidate_id: CanonicalId
    split: Split
    before: SceneStateEnvelope
    after: SceneStateEnvelope
    program: EditProgram
    goal: AfterGoal
    before_goal: Literal[False] = False
    after_goal: Literal[True] = True
    accepted_claim_definition_ref: CanonicalId
    result_sha256: Sha256Digest
    certificate_sha256: Sha256Digest
    objective_lower_bound: NonNegativeFiniteFloat
    objective_upper_bound: NonNegativeFiniteFloat
    content_sha256: Sha256Digest


class GeneralPairLineage(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/pair-lineage"
    SELF_DIGEST_FIELD: ClassVar[str] = "lineage_sha256"
    source_snapshot_sha256: Sha256Digest
    task_sha256: Sha256Digest
    candidate_sha256: Sha256Digest
    request_sha256: Sha256Digest
    selection_sha256: Sha256Digest
    submission_sha256: Sha256Digest
    checked_outcome_sha256: Sha256Digest
    dispatch_sha256: Sha256Digest
    certificate_sha256: Sha256Digest
    result_sha256: Sha256Digest
    program_sha256: Sha256Digest
    obligations_sha256: Sha256Digest
    runtime_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    content_sha256: Sha256Digest
    lineage_sha256: Sha256Digest


class GeneralPairRecord(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/record"
    SELF_DIGEST_FIELD: ClassVar[str] = "record_sha256"
    content: GeneralPairContent
    lineage: GeneralPairLineage
    record_sha256: Sha256Digest


class GeneralRecords(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/records"
    SELF_DIGEST_FIELD: ClassVar[str] = "records_sha256"
    records: tuple[GeneralPairRecord, ...]
    records_sha256: Sha256Digest


class GeneralDatasetReport(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/report"
    SELF_DIGEST_FIELD: ClassVar[str] = "report_sha256"
    catalog_sha256: Sha256Digest
    ledger_sha256: Sha256Digest
    candidate_count: Annotated[StrictInt, Field(ge=0)]
    pair_count: Annotated[StrictInt, Field(ge=0)]
    proven_unsat_count: Annotated[StrictInt, Field(ge=0)]
    unknown_count: Annotated[StrictInt, Field(ge=0)]
    noncertified_witness_count: Annotated[StrictInt, Field(ge=0)]
    ordered_record_sha256: tuple[Sha256Digest, ...]
    native_execution: Literal["NOT_REQUESTED"] = "NOT_REQUESTED"
    report_sha256: Sha256Digest


class GeneralFileIdentity(CanonicalModel):
    name: Literal[
        "input.json",
        "catalog.json",
        "terminals.json",
        "records.json",
        "report.json",
        "provenance.json",
    ]
    byte_length: Annotated[StrictInt, Field(ge=0, le=256 * 1024 * 1024)]
    byte_sha256: Sha256Digest


class GeneralDatasetManifest(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = _PREFIX + "/manifest"
    SELF_DIGEST_FIELD: ClassVar[str] = "manifest_sha256"
    kind: Literal["spatialcf/general-counterfactual-dataset@1"] = (
        "spatialcf/general-counterfactual-dataset@1"
    )
    inventory: tuple[GeneralFileIdentity, ...]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _exact_inventory(self) -> Self:
        names = tuple(row.name for row in self.inventory)
        if names != (
            "catalog.json",
            "input.json",
            "provenance.json",
            "records.json",
            "report.json",
            "terminals.json",
        ):
            raise ValueError("manifest requires the exact ordered file inventory")
        return self
