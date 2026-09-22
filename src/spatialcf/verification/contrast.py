"""Descriptor-bound structural verification for M4 semantic datasets.

This module is deliberately domain-only.  It validates bytes, typed identities,
the closed reference graph, and cross-object bindings.  Fresh compilation and
checker replay remain owned by the generation workflow.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from spatialcf.domain.base import CanonicalModel
from spatialcf.domain.contrast import (
    CandidateRecordInput,
    DatasetManifest,
    PairContent,
    NoncertifiedWitnessTerminal,
    PolicyRejectedTerminal,
    PublicationPolicy,
    PublishedPairTerminal,
    ProvenUnsatTerminal,
    RecordEnvelope,
    SemanticContrastCatalog,
    SemanticContrastCatalogInput,
    SemanticContrastReport,
    SourceRecordInput,
    TerminalLedger,
    TerminalStatus,
    UnknownTerminal,
)
from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
    EditProgram,
    SceneStateEnvelope,
)
from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.lineage import (
    CounterfactualLineage,
    RuntimeProvenance,
    SemanticObjectReference,
    SourceLineageIdentity,
)
from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendUnknownEvidence,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    CheckedProofOutcome,
    NoncertifiedWitnessResult,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    UnknownResult,
    VerifierDispatchRecord,
)
from spatialcf.domain.predicates import GroundedObligationSet
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    bound_child_directory,
    read_regular_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
)

__all__ = ("SemanticContrastBundle", "read_semantic_contrast_bundle")

_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_ENTRIES = 100_000

_OBJECT_TYPES: Mapping[str, type[HashBoundCanonicalModel]] = {
    value.__name__: value
    for value in (
        BackendCompleteUnsatEvidence,
        BackendProposalSubmission,
        BackendSelectionRecord,
        BackendUnknownEvidence,
        CandidateRecordInput,
        CertifiedSolutionCertificate,
        CertifiedSolutionResult,
        CheckedProofOutcome,
        CounterfactualLineage,
        CounterfactualSolveRequest,
        EditProgram,
        GroundedObligationSet,
        NoncertifiedWitnessResult,
        PairContent,
        ProvenUnsatCertificate,
        ProvenUnsatResult,
        PublicationPolicy,
        RuntimeProvenance,
        SceneStateEnvelope,
        SemanticContrastCatalogInput,
        SourceRecordInput,
        UnknownResult,
        VerifierDispatchRecord,
    )
}


@dataclass(frozen=True, slots=True)
class SemanticContrastBundle:
    manifest: DatasetManifest
    catalog: SemanticContrastCatalog
    ledger: TerminalLedger
    report: SemanticContrastReport
    records: tuple[RecordEnvelope, ...]
    objects: Mapping[str, HashBoundCanonicalModel]
    source_payloads: Mapping[str, bytes]
    payloads: Mapping[str, bytes]

    def resolve(self, reference: SemanticObjectReference) -> HashBoundCanonicalModel:
        try:
            value = self.objects[reference.semantic_sha256]
        except KeyError as error:
            raise ValueError(
                "semantic dataset contains a dangling reference"
            ) from error
        if type(value).__name__ != reference.expected_kind:
            raise ValueError("semantic dataset reference has the wrong object kind")
        return value


def _parse_exact(payload: bytes, model_type: type[HashBoundCanonicalModel]):
    try:
        value = model_type.model_validate_json(payload, strict=False)
    except Exception as error:
        raise ValueError(f"invalid {model_type.__name__} JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical {model_type.__name__} bytes")
    return model_type.model_validate(
        value.model_dump(mode="python", warnings="error", round_trip=True),
        strict=True,
    )


def _semantic_sha(value: HashBoundCanonicalModel) -> str:
    return getattr(value, value.SELF_DIGEST_FIELD)


def _references(value: object) -> tuple[SemanticObjectReference, ...]:
    found: list[SemanticObjectReference] = []

    def visit(item: object) -> None:
        if isinstance(item, SemanticObjectReference):
            found.append(item)
        elif isinstance(item, CanonicalModel):
            for field_name in type(item).model_fields:
                visit(getattr(item, field_name))
        elif isinstance(item, tuple):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def _require_regular(entries: Mapping[str, os.stat_result], label: str) -> None:
    identities: set[tuple[int, int]] = set()
    for name, result in entries.items():
        if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
            raise ValueError(f"semantic {label} entry must be one regular file: {name}")
        identity = (result.st_dev, result.st_ino)
        if identity in identities:
            raise ValueError(f"semantic {label} contains a hardlink alias")
        identities.add(identity)


def _validate_solve_roots(bundle, request, carrier) -> None:
    """Resolve transport bindings only; checker semantics require fresh replay."""
    resolve = bundle.resolve
    selection = resolve(carrier.selection)
    result = resolve(carrier.result)
    roots = request.semantic_problem_sha256, request.solve_request_sha256
    for value in (selection, result):
        if (value.semantic_problem_sha256, value.solve_request_sha256) != roots:
            raise ValueError("terminal evidence binds a different request")
    selection_sha = selection.backend_selection_record_sha256
    if result.backend_selection_record_sha256 != selection_sha:
        raise ValueError("terminal result binds a different selection")
    if isinstance(carrier, UnknownTerminal):
        if carrier.selection_disposition != selection.selection_disposition:
            raise ValueError("UNKNOWN selection disposition differs from selection")
        if carrier.selection_disposition == "NO_SELECTION":
            if result.checked_proof_outcome_sha256 is not None:
                raise ValueError("NO_SELECTION result carries checker evidence")
            return
    if selection.selection_disposition != "SELECTED":
        raise ValueError("submitted terminal has no selected backend")
    submission = resolve(carrier.submission)
    evidence = submission.proposal if type(submission) is BackendProposalSubmission else submission
    checked, dispatch = resolve(carrier.checked_outcome), resolve(carrier.dispatch)
    for value in (evidence, checked, dispatch):
        if (
            (value.semantic_problem_sha256, value.solve_request_sha256) != roots
            or value.backend_selection_record_sha256 != selection_sha
        ):
            raise ValueError("terminal submission/checker request or selection disagrees")
    if (
        checked.proof_material_sha256 != evidence.proof_material_sha256
        or result.checked_proof_outcome_sha256 != checked.checked_proof_outcome_sha256
        or dispatch.checked_proof_outcome_sha256 != checked.checked_proof_outcome_sha256
        or result.verifier_dispatch_record_sha256 != dispatch.verifier_dispatch_record_sha256
        or result.checker_disposition != checked.checker_disposition
        or result.resource_usage != evidence.resource_usage
    ):
        raise ValueError("terminal proof, checker, dispatch or resource roots disagree")
    if isinstance(carrier, UnknownTerminal) and (
        result.reason_claim_definition_ref != submission.reason_claim_definition_ref
    ):
        raise ValueError("UNKNOWN result and submission reasons disagree")
    if isinstance(carrier, (CounterfactualLineage, ProvenUnsatTerminal)):
        certificate = resolve(carrier.certificate)
        if (
            (certificate.semantic_problem_sha256, certificate.solve_request_sha256) != roots
            or certificate.backend_selection_record_sha256 != selection_sha
            or certificate.checked_proof_outcome_sha256 != checked.checked_proof_outcome_sha256
            or certificate.verifier_dispatch_record_sha256 != dispatch.verifier_dispatch_record_sha256
            or certificate.proof_material_sha256 != evidence.proof_material_sha256
            or certificate.scene_state_sha256 != request.semantic_problem.scene_state.scene_state_sha256
            or result.certificate_sha256 != certificate.certificate_sha256
            or result.accepted_certificate != certificate
        ):
            raise ValueError("terminal certificate binds different solve evidence")


def _validate_cross_object(bundle: SemanticContrastBundle) -> None:
    resolve = bundle.resolve
    catalog_input = resolve(bundle.catalog.catalog_input)
    policy = resolve(bundle.catalog.publication_policy)
    runtime = resolve(bundle.catalog.runtime_provenance)
    if not isinstance(catalog_input, SemanticContrastCatalogInput):
        raise ValueError("catalog input reference has the wrong payload")
    if not isinstance(policy, PublicationPolicy) or not isinstance(
        runtime, RuntimeProvenance
    ):
        raise ValueError("catalog policy or runtime reference has the wrong payload")
    normalized = canonical_json_bytes(catalog_input)
    if (
        bundle.catalog.normalized_input_byte_length != len(normalized)
        or bundle.catalog.normalized_input_byte_sha256
        != hashlib.sha256(normalized).hexdigest()
        or policy != catalog_input.publication_policy
    ):
        raise ValueError("frozen catalog does not bind its normalized input")
    source_inputs = {
        _semantic_sha(item): item for item in catalog_input.sources
    }
    candidate_inputs = {
        _semantic_sha(item): item for item in catalog_input.candidates
    }
    if (
        tuple(item.source.semantic_sha256 for item in bundle.catalog.sources) != tuple(source_inputs)
        or tuple(item.candidate.semantic_sha256 for item in bundle.catalog.candidates) != tuple(candidate_inputs)
    ):
        raise ValueError("frozen source or candidate universe differs from catalog input")
    for frozen in bundle.catalog.sources:
        source = resolve(frozen.source)
        if source_inputs.get(frozen.source.semantic_sha256) != source:
            raise ValueError("frozen source differs from normalized input")
        assert isinstance(source, SourceRecordInput)
        if (
            source.identity != frozen.identity
            or source.source_group != frozen.source_group
            or source.scene_state_sha256 != frozen.scene_state_sha256
        ):
            raise ValueError("frozen source fields disagree with source object")
        for reference in (source.source_record_bytes, *source.source_files.files):
            if len(bundle.source_payloads[reference.byte_sha256]) != reference.byte_length:
                raise ValueError("source reference byte length differs from retained source")
    for frozen in bundle.catalog.candidates:
        candidate = resolve(frozen.candidate)
        if candidate_inputs.get(frozen.candidate.semantic_sha256) != candidate:
            raise ValueError("frozen candidate differs from normalized input")
        assert isinstance(candidate, CandidateRecordInput)
        if (
            candidate.candidate_id != frozen.candidate_id
            or candidate.source_identity != frozen.source_identity
            or candidate.solve_request_sha256 != frozen.solve_request_sha256
            or candidate.backend_profile != frozen.backend_profile
        ):
            raise ValueError("frozen candidate fields disagree with candidate object")
    if bundle.ledger.semantic_contrast_catalog_sha256 != _semantic_sha(bundle.catalog):
        raise ValueError("terminal ledger binds the wrong catalog")
    if (
        bundle.report.semantic_contrast_catalog_sha256 != _semantic_sha(bundle.catalog)
        or bundle.report.terminal_ledger_sha256 != _semantic_sha(bundle.ledger)
        or bundle.manifest.semantic_contrast_catalog_sha256
        != _semantic_sha(bundle.catalog)
        or bundle.manifest.terminal_ledger_sha256 != _semantic_sha(bundle.ledger)
        or bundle.manifest.semantic_contrast_report_sha256
        != _semantic_sha(bundle.report)
        or bundle.manifest.runtime_provenance_sha256 != _semantic_sha(runtime)
    ):
        raise ValueError("semantic roots do not bind one dataset")
    record_roots = tuple(_semantic_sha(item) for item in bundle.records)
    if record_roots != bundle.report.ordered_record_envelope_sha256 or record_roots != (
        bundle.manifest.ordered_record_envelope_sha256
    ):
        raise ValueError("semantic record order differs between roots")
    records = {_semantic_sha(item): item for item in bundle.records}
    frozen_by_id = {item.candidate_id: item for item in bundle.catalog.candidates}
    if bundle.ledger.ordered_candidate_ids != tuple(frozen_by_id):
        raise ValueError("terminal universe or order differs from frozen catalog")
    published_roots = tuple(
        item.record_envelope_sha256 for item in bundle.ledger.terminals
        if type(item) is PublishedPairTerminal
    )
    if published_roots != record_roots:
        raise ValueError("published terminal records differ from ordered dataset records")
    counts = Counter(item.status for item in bundle.ledger.terminals)
    if (
        bundle.report.candidate_count != len(frozen_by_id)
        or bundle.report.pair_count != counts[TerminalStatus.PUBLISHED_PAIR]
        or bundle.report.proven_unsat_count != counts[TerminalStatus.PROVEN_UNSAT]
        or bundle.report.unknown_count != counts[TerminalStatus.UNKNOWN]
        or bundle.report.noncertified_witness_count != counts[TerminalStatus.NONCERTIFIED_WITNESS]
        or bundle.report.policy_rejected_count != counts[TerminalStatus.POLICY_REJECTED]
    ):
        raise ValueError("report counts differ from terminal ledger")
    for terminal in bundle.ledger.terminals:
        frozen = frozen_by_id[terminal.candidate_id]
        candidate = resolve(frozen.candidate)
        request = candidate.solve_request
        source = source_inputs[frozen.source_record_input_sha256]
        if terminal.source_record_input_sha256 != source.source_record_input_sha256:
            raise ValueError("terminal binds a different source")
        if isinstance(terminal, PolicyRejectedTerminal):
            if (
                resolve(terminal.solve_request) != request
                or terminal.policy_evidence != frozen.policy_evidence
                or terminal.publication_policy_sha256 != policy.publication_policy_sha256
            ):
                raise ValueError("policy terminal does not bind the frozen decision")
            continue
        if not frozen.policy_evidence.eligible:
            raise ValueError("ineligible candidate carries a submitted terminal")
        if isinstance(terminal, (UnknownTerminal, ProvenUnsatTerminal, NoncertifiedWitnessTerminal)):
            _validate_solve_roots(bundle, request, terminal)
        if isinstance(terminal, PublishedPairTerminal):
            record = records.get(terminal.record_envelope_sha256)
            if record is None:
                raise ValueError("published terminal has no record envelope")
            content = resolve(record.content)
            lineage = resolve(record.lineage)
            if not isinstance(content, PairContent) or not isinstance(
                lineage, CounterfactualLineage
            ):
                raise ValueError("published record has the wrong content objects")
            expected_source = SourceLineageIdentity.seal(
                source_dataset_id=source.identity.dataset_id,
                source_revision_id=source.identity.revision_id,
                source_record_id=source.identity.record_id,
                source_record_byte_sha256=source.source_record_bytes.byte_sha256,
                source_files_manifest_sha256=source.source_files.source_files_manifest_sha256,
                source_group=source.source_group,
                scene_state_sha256=source.scene_state_sha256,
            )
            if (
                content.candidate_id != terminal.candidate_id
                or content.source_identity != source.identity
                or content.source_group != frozen.source_group
                or content.split != frozen.split
                or lineage.source != expected_source
                or lineage.publication_policy_sha256 != policy.publication_policy_sha256
                or lineage.runtime_provenance != bundle.catalog.runtime_provenance
                or resolve(content.solve_request) != request
                or lineage.solve_request != content.solve_request
                or terminal.solve_request_sha256 != request.solve_request_sha256
            ):
                raise ValueError("published source, request, split or provenance roots disagree")
            _validate_solve_roots(bundle, request, lineage)
            if (
                content.pair_content_sha256 != terminal.pair_content_sha256
                or lineage.pair_content_sha256 != content.pair_content_sha256
                or lineage.counterfactual_lineage_sha256
                != terminal.counterfactual_lineage_sha256
                or content.result != terminal.result
                or content.certificate != terminal.certificate
                or content.program != terminal.program
                or content.grounded_obligations != terminal.grounded_obligations
                or lineage.result != terminal.result
                or lineage.certificate != terminal.certificate
                or lineage.program != terminal.program
                or lineage.grounded_obligations != terminal.grounded_obligations
            ):
                raise ValueError(
                    "published terminal, content and lineage roots disagree"
                )
            request = resolve(content.solve_request)
            program = resolve(content.program)
            result = resolve(content.result)
            certificate = resolve(content.certificate)
            obligations = resolve(content.grounded_obligations)
            before = resolve(content.before_scene_state)
            after = resolve(content.after_scene_state)
            if not all(
                (
                    isinstance(request, CounterfactualSolveRequest),
                    isinstance(program, EditProgram),
                    isinstance(result, CertifiedSolutionResult),
                    isinstance(certificate, CertifiedSolutionCertificate),
                    isinstance(obligations, GroundedObligationSet),
                    isinstance(before, SceneStateEnvelope),
                    isinstance(after, SceneStateEnvelope),
                )
            ):
                raise ValueError("published pair resolves to the wrong object classes")
            if (
                before != request.semantic_problem.scene_state
                or after != program.after_scene_state
                or program.before_state_sha256 != before.scene_state_sha256
                or result.program_sha256 != program.program_sha256
                or result.certificate_sha256 != certificate.certificate_sha256
                or certificate.solve_request_sha256 != request.solve_request_sha256
                or obligations.grounded_obligation_set_sha256
                != program.grounded_obligation_set_sha256
                or certificate.grounded_obligation_set_sha256 != program.grounded_obligation_set_sha256
                or certificate.program_sha256 != program.program_sha256
                or certificate.after_scene_state_sha256 != after.scene_state_sha256
                or certificate.state_delta_manifest_sha256 != program.state_delta_manifest.state_delta_manifest_sha256
                or program.semantic_problem_sha256 != request.semantic_problem_sha256
                or result.solve_request_sha256 != request.solve_request_sha256
                or result.after_scene_state_sha256 != after.scene_state_sha256
                or content.semantic_evidence.accepted_claim_definition_ref
                != result.claim_definition_ref
                or content.semantic_evidence.proof_material_sha256
                != certificate.proof_material_sha256
            ):
                raise ValueError(
                    "published pair contains inconsistent semantic bindings"
                )


@contextmanager
def read_semantic_contrast_bundle(
    root: Path, *, expected_identity: tuple[int, int] | None = None
) -> Iterator[SemanticContrastBundle]:
    """Retain and validate one exact semantic dataset through caller replay."""

    root = Path(os.path.abspath(root))
    with ExitStack() as stack:
        descriptor = stack.enter_context(
            bound_absolute_directory(root, expected_identity=expected_identity)
        )
        top = snapshot_exact_directory(
            descriptor,
            regular_names={
                "manifest.json",
                "catalog.json",
                "terminals.json",
                "report.json",
            },
            directory_names={"objects", "sources", "records"},
        )
        directories = {
            name: stack.enter_context(bound_child_directory(descriptor, name))
            for name in ("objects", "sources", "records")
        }
        child_entries = {
            name: scan_directory(child, maximum_entries=_MAX_ENTRIES)
            for name, child in directories.items()
        }
        for name, entries in child_entries.items():
            _require_regular(entries, name)
        payloads: dict[str, bytes] = {}

        def read(
            parent: int,
            relative: str,
            name: str,
            expected: os.stat_result,
            maximum: int,
        ) -> bytes:
            payload = read_regular_at(parent, name, maximum, expected_stat=expected)
            payloads[relative] = payload
            return payload

        roots = {
            name: read(descriptor, name, name, top[name], _MAX_JSON_BYTES)
            for name in (
                "manifest.json",
                "catalog.json",
                "terminals.json",
                "report.json",
            )
        }
        manifest = _parse_exact(roots["manifest.json"], DatasetManifest)
        catalog = _parse_exact(roots["catalog.json"], SemanticContrastCatalog)
        ledger = _parse_exact(roots["terminals.json"], TerminalLedger)
        report = _parse_exact(roots["report.json"], SemanticContrastReport)
        inventory = {item.relative_path: item for item in manifest.inventory}
        expected_paths = set(inventory)
        actual_paths = {"catalog.json", "terminals.json", "report.json"}
        actual_paths.update(
            f"{directory}/{name}"
            for directory, entries in child_entries.items()
            for name in entries
        )
        if expected_paths != actual_paths:
            raise ValueError("semantic artifact inventory differs from filesystem")
        models: dict[str, HashBoundCanonicalModel] = {}
        records: list[RecordEnvelope] = []
        record_names: dict[str, str] = {}
        sources: dict[str, bytes] = {}
        for relative in sorted(actual_paths):
            inventory_entry = inventory[relative]
            if relative in roots:
                payload = roots[relative]
                model = {
                    "catalog.json": catalog,
                    "terminals.json": ledger,
                    "report.json": report,
                }[relative]
            else:
                directory, name = relative.split("/", 1)
                if directory == "objects":
                    match = re.fullmatch(r"([0-9a-f]{64})\.json", name)
                elif directory == "records":
                    match = re.fullmatch(r"([0-9a-f]{64})\.json", name)
                else:
                    match = re.fullmatch(r"([0-9a-f]{64})\.bin", name)
                if match is None:
                    raise ValueError(
                        "semantic artifact filename is not content addressed"
                    )
                payload = read(
                    directories[directory],
                    relative,
                    name,
                    child_entries[directory][name],
                    (
                        _MAX_SOURCE_BYTES
                        if directory == "sources"
                        else _MAX_JSON_BYTES
                    ),
                )
                if directory == "sources":
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != match.group(1):
                        raise ValueError(
                            "semantic source filename differs from byte hash"
                        )
                    sources[digest] = payload
                    model = None
                else:
                    model_type = _OBJECT_TYPES.get(
                        inventory_entry.typed_object_kind or ""
                    )
                    if directory == "records":
                        model_type = RecordEnvelope
                    if model_type is None:
                        raise ValueError(
                            "semantic inventory names an unknown object class"
                        )
                    model = _parse_exact(payload, model_type)
                    if (
                        directory == "objects"
                        and _semantic_sha(model) != match.group(1)
                    ):
                        raise ValueError(
                            "semantic object filename differs from typed hash"
                        )
                    if directory == "records":
                        records.append(model)
                        record_names[_semantic_sha(model)] = name
            if (
                len(payload) != inventory_entry.byte_length
                or hashlib.sha256(payload).hexdigest()
                != inventory_entry.byte_sha256
            ):
                raise ValueError("semantic inventory byte identity mismatch")
            if model is not None:
                if (
                    inventory_entry.payload_kind != "TYPED_JSON"
                    or inventory_entry.typed_object_kind != type(model).__name__
                    or inventory_entry.typed_object_sha256 != _semantic_sha(model)
                ):
                    raise ValueError("semantic inventory typed identity mismatch")
                if relative.startswith("objects/"):
                    digest = _semantic_sha(model)
                    if digest in models:
                        raise ValueError("semantic object identity is duplicated")
                    models[digest] = model
            elif inventory_entry.payload_kind != "SOURCE_BYTES":
                raise ValueError("semantic source payload has the wrong inventory kind")
        records.sort(
            key=lambda item: manifest.ordered_record_envelope_sha256.index(
                _semantic_sha(item)
            )
        )
        bundle = SemanticContrastBundle(
            manifest=manifest,
            catalog=catalog,
            ledger=ledger,
            report=report,
            records=tuple(records),
            objects=MappingProxyType(models),
            source_payloads=MappingProxyType(sources),
            payloads=MappingProxyType(payloads),
        )
        for record in records:
            content = bundle.resolve(record.content)
            assert isinstance(content, PairContent)
            expected_name = (
                hashlib.sha256(canonical_json_bytes(content.candidate_id)).hexdigest()
                + ".json"
            )
            if record_names[_semantic_sha(record)] != expected_name:
                raise ValueError(
                    "semantic record filename differs from candidate identity"
                )
        roots_for_graph: tuple[object, ...] = (catalog, ledger, *records)
        reachable: set[str] = set()
        active: set[str] = set()

        def walk(value: object) -> None:
            for reference in _references(value):
                digest = reference.semantic_sha256
                target = bundle.resolve(reference)
                if digest in active:
                    raise ValueError("semantic object graph contains a cycle")
                if digest not in reachable:
                    active.add(digest)
                    walk(target)
                    active.remove(digest)
                    reachable.add(digest)

        for root_value in roots_for_graph:
            walk(root_value)
        if reachable != set(models):
            raise ValueError("semantic object store contains unreachable objects")
        required_sources: set[str] = set()
        for value in models.values():
            if isinstance(value, SourceRecordInput):
                required_sources.add(value.source_record_bytes.byte_sha256)
                required_sources.update(
                    item.byte_sha256 for item in value.source_files.files
                )
        if required_sources != set(sources):
            raise ValueError("semantic source store is incomplete or contains extras")
        _validate_cross_object(bundle)
        def revalidate() -> None:
            observed = scan_directory(descriptor, maximum_entries=len(top) + 1)
            if set(observed) != set(top):
                raise RuntimeError("semantic root file set changed during replay")
            revalidate_entries(descriptor, top)
            for name, entries in child_entries.items():
                observed = scan_directory(directories[name], maximum_entries=len(entries) + 1)
                if set(observed) != set(entries):
                    raise RuntimeError("semantic child file set changed during replay")
                revalidate_entries(directories[name], entries)

        revalidate()
        yield bundle
        revalidate()
