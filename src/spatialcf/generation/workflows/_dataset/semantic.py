"""Generate and replay semantic contrast datasets from retained proofs."""

from __future__ import annotations

import hashlib

import os

from collections import (
    Counter,
)

from collections.abc import (
    Callable,
    Mapping,
)

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Literal,
)

from spatialcf.core.outcome_assembler import (
    AssembledCounterfactualOutcome,
    assemble_counterfactual_outcome,
    assemble_no_selection_unknown,
)

from spatialcf.core.upright_se2_backend import (
    UprightSE2CardinalBackend,
    UprightSE2ContinuousBackend,
)

from spatialcf.domain import (
    contrast as semantic,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.definitions import (
    HashBoundCanonicalModel,
)

from spatialcf.domain.lineage import (
    NativeNotRequested,
    RuntimeProvenance,
    SemanticObjectReference,
    SourceLineageIdentity,
)

from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposalSubmission,
    BackendSubmission,
    BackendUnknownEvidence,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    NoncertifiedWitnessResult,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    UnknownResult,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.generation.planning.campaign import (
    _SemanticCatalogPlan,
    _bound_semantic_catalog,
    _derive_semantic_catalog,
    _semantic_target_labels,
)

from spatialcf.generation.workflows.contracts import (
    _FreshTransitions,
)

from spatialcf.verification.filesystem import (
    RenameLocation,
    open_native_output_parent,
)

from spatialcf.verification.contrast import (
    SemanticContrastBundle,
    read_semantic_contrast_bundle,
)

from spatialcf.generation._internal.artifact_runtime import (
    _semantic_publication_error,
    _semantic_runtime_provenance,
)


def _semantic_ref(value: HashBoundCanonicalModel):
    """Reference an owned payload; its CAS serialization validates it strictly.

    Revalidating the complete proof for each reference duplicates the payload
    validation at serialization and the independent staged read/replay.  The
    public reference factory retains its validation for external callers.
    """
    if not isinstance(value, HashBoundCanonicalModel):
        raise TypeError("semantic references require a hash-bound payload")
    expected_kind = type(value).__name__
    return SemanticObjectReference[Literal[expected_kind]](
        expected_kind=expected_kind,
        semantic_sha256=_semantic_sha(value),
    )


def _semantic_sha(value: HashBoundCanonicalModel) -> str:
    return getattr(value, value.SELF_DIGEST_FIELD)


@dataclass(frozen=True, slots=True)
class _DerivedSemanticBundle:
    manifest: semantic.DatasetManifest
    report: semantic.SemanticContrastReport
    payloads: Mapping[str, bytes]


def _semantic_backend(profile: semantic.BackendProfile):
    if profile is semantic.BackendProfile.CARDINAL:
        return UprightSE2CardinalBackend()
    if profile is semantic.BackendProfile.CONTINUOUS:
        return UprightSE2ContinuousBackend()
    raise ValueError("semantic candidate selects an unsupported backend profile")


def _add_semantic_object(
    objects: dict[str, HashBoundCanonicalModel], value: HashBoundCanonicalModel
) -> None:
    digest = _semantic_sha(value)
    previous = objects.setdefault(digest, value)
    if previous is value:
        # Every stored payload is strictly serialized below before staging.
        return
    if type(previous) is not type(value) or canonical_json_bytes(
        previous
    ) != canonical_json_bytes(value):
        raise ValueError("semantic object digest has conflicting typed payloads")


def _semantic_terminal(
    *,
    frozen: semantic.FrozenCandidateRecord,
    source: semantic.SourceRecordInput,
    candidate: semantic.CandidateRecordInput,
    selection,
    submission: BackendSubmission | None,
    assembled: AssembledCounterfactualOutcome | None,
    policy: semantic.PublicationPolicy,
    runtime: RuntimeProvenance,
    objects: dict[str, HashBoundCanonicalModel],
) -> tuple[semantic.TerminalRecord, semantic.RecordEnvelope | None]:
    request = candidate.solve_request
    if not frozen.policy_evidence.eligible:
        _add_semantic_object(objects, request)
        terminal = semantic.PolicyRejectedTerminal.seal(
            candidate_id=candidate.candidate_id,
            source_record_input_sha256=source.source_record_input_sha256,
            solve_request=_semantic_ref(request),
            publication_policy_sha256=policy.publication_policy_sha256,
            policy_evidence=frozen.policy_evidence,
        )
        return terminal, None
    if selection is None or assembled is None:
        raise RuntimeError("eligible semantic candidate lacks owned assembly")
    _add_semantic_object(objects, selection)
    _add_semantic_object(objects, assembled.result)
    result = assembled.result
    if type(result) is CertifiedSolutionResult:
        if (
            type(submission) is not BackendProposalSubmission
            or type(assembled.certificate) is not CertifiedSolutionCertificate
            or assembled.checked_proof_outcome is None
            or assembled.verifier_dispatch_record is None
            or assembled.program is None
            or assembled.grounded_obligations is None
        ):
            raise RuntimeError(
                "certified semantic result lacks complete owned evidence"
            )
        for value in (
            request,
            submission,
            assembled.checked_proof_outcome,
            assembled.verifier_dispatch_record,
            assembled.certificate,
            assembled.program,
            assembled.grounded_obligations,
            request.semantic_problem.scene_state,
            assembled.program.after_scene_state,
        ):
            _add_semantic_object(objects, value)
        before_pair, _after_pair, before_relation, after_relation = (
            _semantic_target_labels(request)
        )
        solve_policy = upright.decode_upright_se2_solve_policy_definition_payload(
            request.solve_policy_definition_bundle
        )
        evidence = semantic.SemanticContrastEvidence.seal(
            subject_id=before_pair[0],
            reference_id=before_pair[1],
            before_relation=before_relation,
            after_relation=after_relation,
            accepted_claim_definition_ref=result.claim_definition_ref,
            objective_lower_bound=submission.proposal.objective_lower_bound,
            objective_upper_bound=submission.proposal.objective_upper_bound,
            requested_gap=solve_policy.requested_gap,
            proof_material_sha256=assembled.certificate.proof_material_sha256,
        )
        content = semantic.PairContent.seal(
            candidate_id=candidate.candidate_id,
            source_identity=source.identity,
            source_group=source.source_group,
            split=frozen.split,
            before_scene_state=_semantic_ref(request.semantic_problem.scene_state),
            after_scene_state=_semantic_ref(assembled.program.after_scene_state),
            solve_request=_semantic_ref(request),
            program=_semantic_ref(assembled.program),
            grounded_obligations=_semantic_ref(assembled.grounded_obligations),
            result=_semantic_ref(result),
            certificate=_semantic_ref(assembled.certificate),
            semantic_evidence=evidence,
        )
        _add_semantic_object(objects, content)
        source_lineage = SourceLineageIdentity.seal(
            source_dataset_id=source.identity.dataset_id,
            source_revision_id=source.identity.revision_id,
            source_record_id=source.identity.record_id,
            source_record_byte_sha256=source.source_record_bytes.byte_sha256,
            source_files_manifest_sha256=(
                source.source_files.source_files_manifest_sha256
            ),
            source_group=source.source_group,
            scene_state_sha256=source.scene_state_sha256,
        )
        lineage = semantic.CounterfactualLineage.seal(
            pair_content_sha256=content.pair_content_sha256,
            source=source_lineage,
            solve_request=_semantic_ref(request),
            selection=_semantic_ref(selection),
            submission=_semantic_ref(submission),
            checked_outcome=_semantic_ref(assembled.checked_proof_outcome),
            dispatch=_semantic_ref(assembled.verifier_dispatch_record),
            certificate=_semantic_ref(assembled.certificate),
            result=_semantic_ref(result),
            program=_semantic_ref(assembled.program),
            grounded_obligations=_semantic_ref(assembled.grounded_obligations),
            publication_policy_sha256=policy.publication_policy_sha256,
            runtime_provenance=_semantic_ref(runtime),
            native_execution=NativeNotRequested(),
        )
        _add_semantic_object(objects, lineage)
        record = semantic.RecordEnvelope.seal(
            content=_semantic_ref(content),
            pair_content_sha256=content.pair_content_sha256,
            lineage=_semantic_ref(lineage),
            counterfactual_lineage_sha256=lineage.counterfactual_lineage_sha256,
        )
        terminal = semantic.PublishedPairTerminal.seal(
            candidate_id=candidate.candidate_id,
            solve_request_sha256=request.solve_request_sha256,
            source_record_input_sha256=source.source_record_input_sha256,
            record_envelope_sha256=record.record_envelope_sha256,
            pair_content_sha256=content.pair_content_sha256,
            counterfactual_lineage_sha256=lineage.counterfactual_lineage_sha256,
            result=_semantic_ref(result),
            certificate=_semantic_ref(assembled.certificate),
            program=_semantic_ref(assembled.program),
            grounded_obligations=_semantic_ref(assembled.grounded_obligations),
        )
        return terminal, record
    if assembled.checked_proof_outcome is not None:
        _add_semantic_object(objects, assembled.checked_proof_outcome)
    if assembled.verifier_dispatch_record is not None:
        _add_semantic_object(objects, assembled.verifier_dispatch_record)
    if submission is not None:
        _add_semantic_object(objects, submission)
    if type(result) is ProvenUnsatResult:
        if (
            submission is None
            or type(assembled.certificate) is not ProvenUnsatCertificate
            or assembled.checked_proof_outcome is None
            or assembled.verifier_dispatch_record is None
        ):
            raise RuntimeError("UNSAT semantic result lacks complete owned evidence")
        _add_semantic_object(objects, assembled.certificate)
        return semantic.ProvenUnsatTerminal.seal(
            candidate_id=candidate.candidate_id,
            source_record_input_sha256=source.source_record_input_sha256,
            selection=_semantic_ref(selection),
            submission=_semantic_ref(submission),
            checked_outcome=_semantic_ref(assembled.checked_proof_outcome),
            dispatch=_semantic_ref(assembled.verifier_dispatch_record),
            result=_semantic_ref(result),
            certificate=_semantic_ref(assembled.certificate),
        ), None
    if type(result) is NoncertifiedWitnessResult:
        if (
            submission is None
            or assembled.checked_proof_outcome is None
            or assembled.verifier_dispatch_record is None
        ):
            raise RuntimeError("witness semantic result lacks checker evidence")
        return semantic.NoncertifiedWitnessTerminal.seal(
            candidate_id=candidate.candidate_id,
            source_record_input_sha256=source.source_record_input_sha256,
            selection=_semantic_ref(selection),
            submission=_semantic_ref(submission),
            checked_outcome=_semantic_ref(assembled.checked_proof_outcome),
            dispatch=_semantic_ref(assembled.verifier_dispatch_record),
            result=_semantic_ref(result),
        ), None
    if type(result) is UnknownResult:
        selected = selection.selection_disposition == "SELECTED"
        return semantic.UnknownTerminal.seal(
            candidate_id=candidate.candidate_id,
            source_record_input_sha256=source.source_record_input_sha256,
            selection_disposition=("SELECTED" if selected else "NO_SELECTION"),
            selection=_semantic_ref(selection),
            submission=_semantic_ref(submission) if submission is not None else None,
            checked_outcome=(
                _semantic_ref(assembled.checked_proof_outcome)
                if assembled.checked_proof_outcome is not None
                else None
            ),
            dispatch=(
                _semantic_ref(assembled.verifier_dispatch_record)
                if assembled.verifier_dispatch_record is not None
                else None
            ),
            result=_semantic_ref(result),
        ), None
    raise TypeError("semantic workflow received an unsupported assembled result")


def _semantic_inventory_entry(path: str, payload: bytes, value=None):
    if value is None:
        return semantic.ArtifactInventoryEntry(
            relative_path=path,
            payload_kind="SOURCE_BYTES",
            byte_length=len(payload),
            byte_sha256=hashlib.sha256(payload).hexdigest(),
        )
    return semantic.ArtifactInventoryEntry(
        relative_path=path,
        payload_kind="TYPED_JSON",
        typed_object_kind=type(value).__name__,
        typed_object_sha256=_semantic_sha(value),
        byte_length=len(payload),
        byte_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _derive_semantic_bundle(
    plan: _SemanticCatalogPlan,
    runtime: RuntimeProvenance,
    submission_provider: Callable[
        [object, object, object, semantic.CandidateRecordInput], BackendSubmission
    ],
) -> _DerivedSemanticBundle:
    transitions = _FreshTransitions(profile="semantic")
    members = {}
    for frozen in plan.catalog.candidates:
        members[frozen.candidate_id] = transitions.register_semantic(
            frozen.candidate_id, frozen.candidate.semantic_sha256
        )
    transitions.begin_semantic(tuple(members))
    sources = {
        canonical_json_bytes(item.identity): item for item in plan.catalog_input.sources
    }
    candidates = {item.candidate_id: item for item in plan.catalog_input.candidates}
    objects: dict[str, HashBoundCanonicalModel] = {}
    for value in (
        plan.catalog_input,
        plan.catalog_input.publication_policy,
        runtime,
        *plan.catalog_input.sources,
        *plan.catalog_input.candidates,
    ):
        _add_semantic_object(objects, value)
    records: list[semantic.RecordEnvelope] = []
    for frozen, compilation in zip(
        plan.catalog.candidates, plan.compilations, strict=True
    ):
        candidate = candidates[frozen.candidate_id]
        source = sources[canonical_json_bytes(candidate.source_identity)]
        selection = submission = assembled = None
        if frozen.policy_evidence.eligible:
            backend = _semantic_backend(frozen.backend_profile)
            selection = backend.select(candidate.solve_request)
            if selection.selection_disposition == "NO_SELECTION":
                assembled = assemble_no_selection_unknown(
                    solve_request=candidate.solve_request,
                    selection=selection,
                )
            else:
                expected_type = (
                    upright.UprightSE2Compilation
                    if frozen.backend_profile is semantic.BackendProfile.CARDINAL
                    else upright.UprightSE2ContinuousCompilation
                )
                if type(compilation) is not expected_type:
                    raise RuntimeError(
                        "selected semantic candidate has no exact compilation"
                    )
                submission = submission_provider(
                    backend, compilation, selection, candidate
                )
                assembled = assemble_counterfactual_outcome(
                    solve_request=candidate.solve_request,
                    selection=selection,
                    compilation=compilation,
                    submission=submission,
                )
        terminal, record = _semantic_terminal(
            frozen=frozen,
            source=source,
            candidate=candidate,
            selection=selection,
            submission=submission,
            assembled=assembled,
            policy=plan.catalog_input.publication_policy,
            runtime=runtime,
            objects=objects,
        )
        transitions.terminal_semantic(members[candidate.candidate_id], terminal)
        if record is not None:
            records.append(record)
    ordered_ids = tuple(item.candidate_id for item in plan.catalog.candidates)
    terminals = tuple(
        item.value for item in transitions.validate_semantic_closure(ordered_ids)
    )
    ledger = semantic.TerminalLedger.seal(
        semantic_contrast_catalog_sha256=plan.catalog.semantic_contrast_catalog_sha256,
        ordered_candidate_ids=ordered_ids,
        terminals=terminals,
    )
    counts = Counter(item.status for item in terminals)
    record_roots = tuple(item.record_envelope_sha256 for item in records)
    report = semantic.SemanticContrastReport.seal(
        semantic_contrast_catalog_sha256=plan.catalog.semantic_contrast_catalog_sha256,
        terminal_ledger_sha256=ledger.terminal_ledger_sha256,
        ordered_record_envelope_sha256=record_roots,
        candidate_count=len(terminals),
        pair_count=counts[semantic.TerminalStatus.PUBLISHED_PAIR],
        proven_unsat_count=counts[semantic.TerminalStatus.PROVEN_UNSAT],
        unknown_count=counts[semantic.TerminalStatus.UNKNOWN],
        noncertified_witness_count=counts[semantic.TerminalStatus.NONCERTIFIED_WITNESS],
        policy_rejected_count=counts[semantic.TerminalStatus.POLICY_REJECTED],
    )
    payloads: dict[str, bytes] = {
        "catalog.json": canonical_json_bytes(plan.catalog),
        "terminals.json": canonical_json_bytes(ledger),
        "report.json": canonical_json_bytes(report),
    }
    values: dict[str, HashBoundCanonicalModel] = {
        "catalog.json": plan.catalog,
        "terminals.json": ledger,
        "report.json": report,
    }
    for digest, value in objects.items():
        path = f"objects/{digest}.json"
        payloads[path] = canonical_json_bytes(value)
        values[path] = value
    for record in records:
        content = objects[record.content.semantic_sha256]
        assert isinstance(content, semantic.PairContent)
        digest = hashlib.sha256(canonical_json_bytes(content.candidate_id)).hexdigest()
        path = f"records/{digest}.json"
        payloads[path] = canonical_json_bytes(record)
        values[path] = record
    for digest, payload in plan.source_payloads:
        payloads[f"sources/{digest}.bin"] = payload
    inventory = []
    for path, payload in payloads.items():
        inventory.append(_semantic_inventory_entry(path, payload, values.get(path)))
    manifest = semantic.DatasetManifest.seal(
        semantic_contrast_catalog_sha256=plan.catalog.semantic_contrast_catalog_sha256,
        terminal_ledger_sha256=ledger.terminal_ledger_sha256,
        semantic_contrast_report_sha256=report.semantic_contrast_report_sha256,
        runtime_provenance_sha256=runtime.runtime_provenance_sha256,
        ordered_record_envelope_sha256=record_roots,
        inventory=tuple(sorted(inventory, key=canonical_json_bytes)),
    )
    payloads["manifest.json"] = canonical_json_bytes(manifest)
    return _DerivedSemanticBundle(manifest=manifest, report=report, payloads=payloads)


def _generation_submission(backend, compilation, _selection, candidate):
    return backend.solve_submission(compilation, candidate.solve_request.solver_config)


def _retained_submission_provider(bundle: SemanticContrastBundle):
    terminals = {item.candidate_id: item for item in bundle.ledger.terminals}
    records = {item.record_envelope_sha256: item for item in bundle.records}

    def provide(_backend, _compilation, _selection, candidate):
        terminal = terminals[candidate.candidate_id]
        if isinstance(terminal, semantic.PublishedPairTerminal):
            record = records[terminal.record_envelope_sha256]
            lineage = bundle.resolve(record.lineage)
            assert isinstance(lineage, semantic.CounterfactualLineage)
            value = bundle.resolve(lineage.submission)
        else:
            reference = getattr(terminal, "submission", None)
            if reference is None:
                raise ValueError(
                    "selected semantic terminal has no retained submission"
                )
            value = bundle.resolve(reference)
        if type(value) not in {
            BackendProposalSubmission,
            BackendCompleteUnsatEvidence,
            BackendUnknownEvidence,
        }:
            raise TypeError("retained semantic submission has the wrong exact type")
        return value

    return provide


def _fresh_replay_semantic_bundle(
    bundle: SemanticContrastBundle,
) -> semantic.SemanticContrastReport:
    runtime = _semantic_runtime_provenance()
    retained_runtime = bundle.resolve(bundle.catalog.runtime_provenance)
    if retained_runtime != runtime:
        raise ValueError(
            "semantic dataset runtime provenance differs from this runtime"
        )
    catalog_input = bundle.resolve(bundle.catalog.catalog_input)
    if not isinstance(catalog_input, semantic.SemanticContrastCatalogInput):
        raise ValueError("semantic dataset catalog input has the wrong type")
    catalog, compilations = _derive_semantic_catalog(catalog_input, runtime)
    if canonical_json_bytes(catalog) != canonical_json_bytes(bundle.catalog):
        raise ValueError("semantic dataset frozen catalog differs on fresh derivation")
    plan = _SemanticCatalogPlan(
        catalog=catalog,
        catalog_input=catalog_input,
        normalized_input=canonical_json_bytes(catalog_input),
        source_payloads=tuple(sorted(bundle.source_payloads.items())),
        compilations=compilations,
    )
    expected = _derive_semantic_bundle(
        plan,
        runtime,
        _retained_submission_provider(bundle),
    )
    if dict(expected.payloads) != dict(bundle.payloads):
        raise ValueError("semantic dataset differs from fresh reconstructed bytes")
    return expected.report


def _write_semantic_bundle(transaction, payloads: Mapping[str, bytes]) -> None:
    for directory in ("objects", "sources", "records"):
        transaction.mkdir(directory)
    for path in sorted(payloads):
        transaction.write(path, payloads[path])


def generate_semantic_contrast_dataset(
    catalog: Path, output: Path
) -> semantic.SemanticContrastReport:
    """Generate, replay, and atomically publish one immutable M4 dataset."""

    if not isinstance(catalog, Path) or not isinstance(output, Path):
        raise TypeError("semantic generation requires Path arguments")
    runtime = _semantic_runtime_provenance()
    output = Path(os.path.abspath(output))
    parent = open_native_output_parent(output)
    transaction = None
    publication_attempted = False
    completed_report = None
    try:
        parent.ensure_absent(parent.output_name)
        transaction = parent.create_staging(label="semantic")
        with transaction:
            try:
                with _bound_semantic_catalog(catalog, runtime) as (
                    plan,
                    revalidate_source,
                ):
                    derived = _derive_semantic_bundle(
                        plan, runtime, _generation_submission
                    )
                    _write_semantic_bundle(transaction, derived.payloads)
                    staging = output.parent / transaction.name
                    with read_semantic_contrast_bundle(
                        staging, expected_identity=transaction.identity
                    ) as retained:
                        replayed = _fresh_replay_semantic_bundle(retained)
                        if replayed != derived.report:
                            raise RuntimeError(
                                "semantic staged replay changed the report"
                            )
                        revalidate_source()
                        transaction.fsync()
                        seal = transaction.seal()
                        transaction.validate_seal(seal)
                        revalidate_source()
                    transaction.validate_seal(seal)
                    revalidate_source()
                    publication_attempted = True
                    transaction.publish()
                    transaction.validate_location(RenameLocation.OUTPUT)
                    transaction.validate_seal(seal)
                    with read_semantic_contrast_bundle(
                        output, expected_identity=transaction.identity
                    ) as published:
                        if published.report != derived.report:
                            raise RuntimeError(
                                "semantic published observation changed report"
                            )
                    completed_report = derived.report
            except BaseException as error:  # noqa: BLE001
                if publication_attempted:
                    _semantic_publication_error(output, transaction, error)
                raise
    except BaseException as active_error:
        try:
            # Transaction.__exit__ may fail after the inner publication handler.
            # Reconcile while the parent binding is still retained, even when
            # the transaction descriptor itself has already been closed.
            if publication_attempted and transaction is not None:
                _semantic_publication_error(output, transaction, active_error)
            raise
        except BaseException as classified_error:
            try:
                parent.close()
            except BaseException as close_error:  # noqa: BLE001
                classified_error.add_note(str(close_error))
            raise
    try:
        parent.close()
    except BaseException as error:
        _semantic_publication_error(output, transaction, error)
    if completed_report is None:
        raise RuntimeError("semantic dataset generation did not produce a report")
    return completed_report


def verify_semantic_contrast_dataset(root: Path) -> semantic.SemanticContrastReport:
    """Structurally validate and freshly replay one immutable M4 dataset."""

    if not isinstance(root, Path):
        raise TypeError("semantic verification requires a Path")
    with read_semantic_contrast_bundle(Path(os.path.abspath(root))) as bundle:
        return _fresh_replay_semantic_bundle(bundle)


# Preserve supported public names and pickle lookup.
_semantic_ref.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_sha.__module__ = "spatialcf.generation.workflows.dataset"
_DerivedSemanticBundle.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_backend.__module__ = "spatialcf.generation.workflows.dataset"
_add_semantic_object.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_terminal.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_inventory_entry.__module__ = "spatialcf.generation.workflows.dataset"
_derive_semantic_bundle.__module__ = "spatialcf.generation.workflows.dataset"
_generation_submission.__module__ = "spatialcf.generation.workflows.dataset"
_retained_submission_provider.__module__ = "spatialcf.generation.workflows.dataset"
_fresh_replay_semantic_bundle.__module__ = "spatialcf.generation.workflows.dataset"
_write_semantic_bundle.__module__ = "spatialcf.generation.workflows.dataset"
generate_semantic_contrast_dataset.__module__ = "spatialcf.generation.workflows.dataset"
verify_semantic_contrast_dataset.__module__ = "spatialcf.generation.workflows.dataset"
