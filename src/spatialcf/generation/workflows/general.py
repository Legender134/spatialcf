"""M5/M6 dataset generation and fresh proof replay through existing owners."""

from collections import Counter
from dataclasses import fields
import hashlib
import os
from pathlib import Path

from spatialcf.core.outcome_assembler import (
    assemble_counterfactual_outcome,
    assemble_no_selection_unknown,
)
from spatialcf.domain import general_dataset as models
from spatialcf.domain.lineage import ModuleFileIdentity, RuntimeProvenance
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256
from spatialcf.generation.planning.general import backend_for, build_general_catalog
from spatialcf.generation._internal.artifact_runtime import (
    _semantic_publication_error,
    _semantic_retained_bytes,
    _semantic_runtime_provenance,
)
from spatialcf.verification.filesystem import open_native_output_parent, RenameLocation
from spatialcf.verification.general_dataset import (
    MAX_FILE_BYTES,
    MAX_INPUT_BYTES,
    read_general_bundle,
    read_general_input,
)

# Additive literal closure. M4 continues using its own unchanged closure.
_GENERAL_RUNTIME_EXTRA = (
    "spatialcf/core/_internal/compilation/candidate_cells.py",
    "spatialcf/core/_internal/kernels/rigid_se3.py",
    "spatialcf/core/_internal/kernels/semantic_place.py",
    "spatialcf/core/backends.py",
    "spatialcf/core/rigid_se3_backend.py",
    "spatialcf/core/rigid_se3_compiler.py",
    "spatialcf/core/rigid_se3_verification.py",
    "spatialcf/core/semantic_place_backend.py",
    "spatialcf/core/semantic_place_compiler.py",
    "spatialcf/core/semantic_place_verification.py",
    "spatialcf/domain/general_dataset.py",
    "spatialcf/domain/rigid_se3.py",
    "spatialcf/domain/semantic_place.py",
    "spatialcf/generation/general.py",
    "spatialcf/generation/planning/general.py",
    "spatialcf/generation/workflows/general.py",
    "spatialcf/verification/general_dataset.py",
)


def general_runtime_provenance():
    base = _semantic_runtime_provenance()
    modules = {row.package_relative_path: row for row in base.module_files}
    root = Path(__file__).parents[3]
    for name in _GENERAL_RUNTIME_EXTRA:
        raw = _semantic_retained_bytes(root / name)
        modules[name] = ModuleFileIdentity(
            package_relative_path=name,
            byte_length=len(raw),
            byte_sha256=hashlib.sha256(raw).hexdigest(),
        )
    return RuntimeProvenance.seal(
        interpreter=base.interpreter,
        dependencies=base.dependencies,
        lock_metadata=base.lock_metadata,
        module_files=tuple(
            modules[name] for name in sorted(modules, key=canonical_json_bytes)
        ),
    )


def _sha(value):
    return getattr(value, value.SELF_DIGEST_FIELD)


def _pair(candidate, submission, assembly, plan, runtime):
    program = assembly.program
    if program is None or submission is None:
        raise ValueError("certified result lacks retained program or submission")
    content = models.GeneralPairContent.seal(
        candidate_id=candidate.candidate_id,
        split=candidate.split,
        before=candidate.request.semantic_problem.scene_state,
        after=program.after_scene_state,
        program=program,
        goal=candidate.request.semantic_problem.after_goal,
        accepted_claim_definition_ref=assembly.result.claim_definition_ref,
        result_sha256=_sha(assembly.result),
        certificate_sha256=_sha(assembly.certificate),
        objective_lower_bound=submission.proposal.objective_lower_bound,
        objective_upper_bound=submission.proposal.objective_upper_bound,
    )
    lineage = models.GeneralPairLineage.seal(
        source_snapshot_sha256=candidate.source_snapshot_sha256,
        task_sha256=candidate.task_sha256,
        candidate_sha256=candidate.candidate_sha256,
        request_sha256=_sha(candidate.request),
        selection_sha256=_sha(candidate.selection),
        submission_sha256=_sha(submission),
        checked_outcome_sha256=_sha(assembly.checked_proof_outcome),
        dispatch_sha256=_sha(assembly.verifier_dispatch_record),
        certificate_sha256=_sha(assembly.certificate),
        result_sha256=_sha(assembly.result),
        program_sha256=_sha(program),
        obligations_sha256=_sha(assembly.grounded_obligations),
        runtime_sha256=_sha(runtime),
        policy_sha256=canonical_sha256(
            plan.input.policy, domain="spatialcf/general-dataset/1.0/policy"
        ),
        content_sha256=_sha(content),
    )
    return models.GeneralPairRecord.seal(content=content, lineage=lineage)


def _derive(plan, runtime, retained=None):
    terminals, records = [], []
    if retained is not None and tuple(row.candidate_id for row in retained) != tuple(
        row.candidate_id for row in plan.catalog.candidates
    ):
        raise ValueError("retained terminal universe differs from frozen catalog")
    for index, (candidate, compilation) in enumerate(
        zip(plan.catalog.candidates, plan.compilations, strict=True)
    ):
        request, selection = candidate.request, candidate.selection
        if selection.selection_disposition == "NO_SELECTION":
            submission = None
            outcome = assemble_no_selection_unknown(
                solve_request=request, selection=selection
            )
        else:
            submission = (
                retained[index].submission
                if retained is not None
                else backend_for(candidate.profile).solve_submission(
                    compilation, request.solver_config
                )
            )
            if submission is None:
                raise ValueError("selected terminal lacks submission")
            outcome = assemble_counterfactual_outcome(
                solve_request=request,
                selection=selection,
                compilation=compilation,
                submission=submission,
            )
        assembly = models.RetainedAssembly(
            **{field.name: getattr(outcome, field.name) for field in fields(outcome)}
        )
        status = assembly.result.structural_outcome_class
        record = None
        if status == "CERTIFIED_SOLUTION":
            status = "PUBLISHED_PAIR"
            record = _pair(candidate, submission, assembly, plan, runtime)
            records.append(record)
        terminals.append(
            models.GeneralTerminal.seal(
                candidate_id=candidate.candidate_id,
                candidate_sha256=_sha(candidate),
                status=status,
                selection=selection,
                submission=submission,
                assembly=assembly,
                record_sha256=_sha(record) if record else None,
            )
        )
    ledger = models.GeneralTerminalLedger.seal(
        catalog_sha256=_sha(plan.catalog), terminals=tuple(terminals)
    )
    counts = Counter(row.status for row in terminals)
    report = models.GeneralDatasetReport.seal(
        catalog_sha256=_sha(plan.catalog),
        ledger_sha256=_sha(ledger),
        candidate_count=len(terminals),
        pair_count=counts["PUBLISHED_PAIR"],
        proven_unsat_count=counts["PROVEN_UNSAT"],
        unknown_count=counts["UNKNOWN"],
        noncertified_witness_count=counts["NONCERTIFIED_WITNESS"],
        ordered_record_sha256=tuple(_sha(record) for record in records),
    )
    values = {
        "input.json": plan.input,
        "catalog.json": plan.catalog,
        "terminals.json": ledger,
        "records.json": models.GeneralRecords.seal(records=tuple(records)),
        "report.json": report,
        "provenance.json": runtime,
    }
    payloads = {name: canonical_json_bytes(value) for name, value in values.items()}
    for name, raw in payloads.items():
        if len(raw) > (MAX_INPUT_BYTES if name == "input.json" else MAX_FILE_BYTES):
            raise ValueError("general dataset exceeds transport limit")
    manifest = models.GeneralDatasetManifest.seal(
        inventory=tuple(
            models.GeneralFileIdentity(
                name=name,
                byte_length=len(payloads[name]),
                byte_sha256=hashlib.sha256(payloads[name]).hexdigest(),
            )
            for name in sorted(payloads)
        )
    )
    payloads["manifest.json"] = canonical_json_bytes(manifest)
    return payloads, report


def _replay(bundle):
    runtime = general_runtime_provenance()
    if runtime != bundle.provenance:
        raise ValueError("general dataset runtime provenance differs from this runtime")
    plan = build_general_catalog(bundle.input, runtime_sha256=_sha(runtime))
    if canonical_json_bytes(plan.catalog) != bundle.payloads["catalog.json"]:
        raise ValueError("general frozen catalog differs on fresh derivation")
    payloads, report = _derive(plan, runtime, bundle.ledger.terminals)
    if payloads != dict(bundle.payloads):
        raise ValueError("general dataset differs from fresh reconstructed bytes")
    return report


def verify_general_dataset(root: Path):
    if not isinstance(root, Path):
        raise TypeError("general verification requires a Path")
    with read_general_bundle(root) as bundle:
        return _replay(bundle)


def inspect_general_dataset(root: Path):
    """Inspection returns a freshly verified report, never a persisted flag."""
    return verify_general_dataset(root)


def generate_general_dataset(catalog: Path, output: Path):
    if not isinstance(catalog, Path) or not isinstance(output, Path):
        raise TypeError("general generation requires Path arguments")
    runtime = general_runtime_provenance()
    output = Path(os.path.abspath(output))
    parent = open_native_output_parent(output)
    transaction = None
    publication_attempted = False
    completed = None
    try:
        parent.ensure_absent(parent.output_name)
        transaction = parent.create_staging(label="general")
        with transaction:
            try:
                with read_general_input(catalog) as (value, revalidate_source):
                    plan = build_general_catalog(value, runtime_sha256=_sha(runtime))
                    payloads, report = _derive(plan, runtime)
                    for name in sorted(payloads):
                        transaction.write(name, payloads[name])
                    with read_general_bundle(
                        output.parent / transaction.name,
                        expected_identity=transaction.identity,
                    ) as bundle:
                        if _replay(bundle) != report:
                            raise RuntimeError("staged general report changed")
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
                    with read_general_bundle(
                        output, expected_identity=transaction.identity
                    ) as bundle:
                        if dict(bundle.payloads) != payloads:
                            raise RuntimeError("published general bundle changed")
                    completed = report
            except BaseException as error:
                if publication_attempted:
                    _semantic_publication_error(output, transaction, error)
                raise
    except BaseException as error:
        try:
            if publication_attempted and transaction is not None:
                _semantic_publication_error(output, transaction, error)
            raise
        except BaseException as classified:
            try:
                parent.close()
            except BaseException as close_error:
                classified.add_note(str(close_error))
            raise
    try:
        parent.close()
    except BaseException as error:
        _semantic_publication_error(output, transaction, error)
    if completed is None:
        raise RuntimeError("general generation did not produce a report")
    return completed
