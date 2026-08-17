"""Current partial-batch execution and immutable publication."""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import AI2ThorAdapter
from spatialcf.domain.enums import Relation
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2
from spatialcf.generation._internal.assets import (
    AssetBundle,
    _read_bundle_metadata_fd,
    _verify_bundle_payloads_fd,
    asset_bundle_sha256_from_checked,
    publish_asset_bundle,
)
from spatialcf.generation._internal.execution.audit import EndpointAuditRejected
from spatialcf.generation._internal.execution.correspondence import (
    RequestLineage,
    legacy_sha256,
    request_binding_sha256,
)
from spatialcf.generation._internal.execution.run import (
    AuditExecution,
    execute_audit,
)
from spatialcf.generation._internal.planning.campaign import (
    BatchManifest,
    BatchRequest,
)
from spatialcf.generation.errors import require_wire_version
from spatialcf.verification.filesystem import (
    BindingStatus,
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

_MANIFEST_VERSION = "competition-native-batch-manifest:2.9.1"
_SUMMARY_VERSION = "competition-native-batch-summary:2.9.1"
_PORTABLE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ROOT_FILES = {
    "request-manifest.json",
    "outcomes.jsonl",
    "summary.json",
    "checksums.sha256",
}
_CASE_FILES = {
    "before-rgb.png",
    "before-depth.npy",
    "before-instance.png",
    "before-pointcloud.ply",
    "after-rgb.png",
    "after-depth.npy",
    "after-instance.png",
    "after-pointcloud.ply",
    "bundle.json",
    "checksums.sha256",
}
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_ASSET_BYTES = 512 * 1024 * 1024
_MAX_REJECTION_REASONS = 32
_MAX_REJECTION_REASON_CHARS = 4096
_BATCH_VERIFICATION_CAPABILITY = object()


class BatchAttempt(V2Model):
    attempt_index: int = Field(strict=True, ge=0)
    request_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    outcome: Literal["accepted", "rejected"]
    stage: str | None = Field(default=None, strict=True, min_length=1, max_length=128)
    reasons: tuple[str, ...] = ()
    case_path: str | None = None
    case_tree_sha256: Sha256Digest | None = None
    native_audit_run_sha256: Sha256Digest | None = None
    native_asset_bundle_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if len(self.reasons) > _MAX_REJECTION_REASONS or any(
            len(reason) > _MAX_REJECTION_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("native batch rejection reasons exceed the frozen limits")
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            type(reason) is not str or not reason for reason in self.reasons
        ):
            raise ValueError("native batch reasons must be non-empty and canonical")
        if self.outcome == "accepted":
            if (
                self.stage is not None
                or self.reasons
                or self.case_path != f"accepted/{self.request_id}"
                or self.case_tree_sha256 is None
                or self.native_audit_run_sha256 is None
                or self.native_asset_bundle_sha256 is None
            ):
                raise ValueError("accepted native batch outcome is not closed")
        elif (
            type(self.stage) is not str
            or not self.stage
            or not self.reasons
            or self.case_path is not None
            or self.case_tree_sha256 is not None
            or self.native_audit_run_sha256 is not None
            or self.native_asset_bundle_sha256 is not None
        ):
            raise ValueError("rejected native batch outcome is not closed")
        return self


class BatchStageCount(V2Model):
    stage: str = Field(strict=True, min_length=1, max_length=128)
    count: int = Field(strict=True, gt=0)


class BatchSummary(V2Model):
    """Closed current summary for one partial frozen-slot shard."""

    summary_version: Literal["competition-native-batch-summary:2.9.1"] = (
        "competition-native-batch-summary:2.9.1"
    )
    claim_scope: Literal[
        "DECLARED_RELATION_SUBSET_ONE_CPU_EDIT_PER_ACCEPTED_REQUEST"
    ] = "DECLARED_RELATION_SUBSET_ONE_CPU_EDIT_PER_ACCEPTED_REQUEST"
    evidence_eligible: Literal[False] = False
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    request_count: int = Field(strict=True, ge=1, le=6)
    accepted_count: int = Field(strict=True, ge=0, le=6)
    rejected_count: int = Field(strict=True, ge=0, le=6)
    accepted_relations: tuple[Relation, ...]
    rejected_relations: tuple[Relation, ...]
    rejected_by_stage: tuple[BatchStageCount, ...]
    request_manifest_sha256: Sha256Digest
    outcomes_sha256: Sha256Digest
    declared_relations: tuple[Relation, ...]

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        accepted = tuple(sorted(self.accepted_relations, key=lambda item: item.value))
        rejected = tuple(sorted(self.rejected_relations, key=lambda item: item.value))
        stages = tuple(sorted(self.rejected_by_stage, key=lambda item: item.stage))
        declared = tuple(sorted(self.declared_relations, key=lambda item: item.value))
        if accepted != self.accepted_relations or len(set(accepted)) != len(accepted):
            raise ValueError("accepted native batch relations are not canonical")
        if rejected != self.rejected_relations or len(set(rejected)) != len(rejected):
            raise ValueError("rejected native batch relations are not canonical")
        if set(accepted).intersection(rejected):
            raise ValueError("native partial batch relation outcomes overlap")
        if (
            declared != self.declared_relations
            or len(set(declared)) != len(declared)
            or set(accepted).union(rejected) != set(declared)
            or self.request_count != len(declared)
        ):
            raise ValueError("native partial batch declared relations are not closed")
        if self.accepted_count != len(accepted) or self.rejected_count != len(rejected):
            raise ValueError("native batch summary counts are not closed")
        if self.accepted_count + self.rejected_count != self.request_count:
            raise ValueError("native batch summary total is not closed")
        if stages != self.rejected_by_stage or len(
            {item.stage for item in stages}
        ) != len(stages):
            raise ValueError("native batch stage counts are not canonical")
        if sum(item.count for item in stages) != self.rejected_count:
            raise ValueError("native batch stage counts do not close rejections")
        return self


@dataclass(frozen=True, slots=True)
class RetainedBatchVerification:
    _capability: object = field(repr=False, compare=False)
    root_identity: DirectoryIdentity
    summary: BatchSummary
    checksum_payload: bytes = field(repr=False)
    _root_entries: tuple[tuple[str, os.stat_result], ...] = field(
        repr=False,
        compare=False,
    )


def _normalize_request_lineage(
    manifest: BatchManifest,
    request_lineage: Mapping[str, RequestLineage],
) -> dict[str, RequestLineage]:
    if not isinstance(request_lineage, Mapping):
        raise TypeError("native batch request lineage must be a mapping")
    copied = dict(request_lineage)
    if any(type(request_id) is not str for request_id in copied):
        raise TypeError("native batch request lineage keys must be exact strings")
    request_ids = tuple(request.request_id for request in manifest.requests)
    if len(copied) != len(request_ids) or set(copied) != set(request_ids):
        raise ValueError("native batch request lineage must exactly cover the manifest")
    normalized: dict[str, RequestLineage] = {}
    for request in manifest.requests:
        entry = copied[request.request_id]
        if type(entry) is not RequestLineage:
            raise TypeError("native batch request lineage values must be exact")
        checked = RequestLineage.model_validate(
            entry.model_dump(mode="python", warnings="error"), strict=True
        )
        if (
            checked.batch_request_sha256 != request_binding_sha256(request)
            or checked.endpoint_plan.endpoint_workspace != request.endpoint_workspace
        ):
            raise ValueError("native batch request lineage binding mismatch")
        normalized[request.request_id] = checked
    return normalized


def load_batch_manifest(path: Path) -> BatchManifest:
    if not isinstance(path, Path):
        raise TypeError("batch manifest path must be a Path")
    absolute = Path(os.path.abspath(path))
    with bound_absolute_directory(absolute.parent) as descriptor:
        item = os.stat(absolute.name, dir_fd=descriptor, follow_symlinks=False)
        payload = read_regular_at(
            descriptor,
            absolute.name,
            _MAX_METADATA_BYTES,
            expected_stat=item,
        )
        revalidate_entries(descriptor, {absolute.name: item})
    require_wire_version(
        payload,
        artifact_kind="batch manifest",
        field="manifest_version",
        expected=_MANIFEST_VERSION,
    )
    manifest = BatchManifest.model_validate_json(payload, strict=True)
    if payload != canonical_json_bytes_v2(manifest) + b"\n":
        raise ValueError("native batch manifest is not canonical")
    return manifest


def execute_batch(
    manifest: BatchManifest,
    output_root: Path,
    *,
    request_lineage: Mapping[str, RequestLineage],
    expected_parent_identity: DirectoryIdentity | None = None,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
    runner: Callable[..., AuditExecution] = execute_audit,
) -> BatchSummary:
    """Execute every declared request once and atomically publish all outcomes."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(manifest) is not BatchManifest:
            raise TypeError("manifest must be exact BatchManifest")
        checked = BatchManifest.model_validate(
            manifest.model_dump(mode="python", warnings="error"), strict=True
        )
        lineage = _normalize_request_lineage(checked, request_lineage)
        manifest_payload = canonical_json_bytes_v2(checked) + b"\n"
        if len(manifest_payload) > _MAX_METADATA_BYTES:
            raise ValueError("native batch request manifest exceeds byte limit")
        if not isinstance(output_root, Path):
            raise TypeError("output_root must be a Path")
        output = Path(os.path.abspath(output_root))
        return _execute_batch_transaction(
            checked,
            output,
            lineage,
            expected_parent_identity=expected_parent_identity,
            adapter_factory=adapter_factory,
            runner=runner,
        )


def _execute_batch_transaction(
    manifest: BatchManifest,
    output: Path,
    request_lineage: dict[str, RequestLineage],
    *,
    expected_parent_identity: DirectoryIdentity | None,
    adapter_factory,
    runner,
) -> BatchSummary:
    parent = open_native_output_parent(
        output, expected_parent_identity=expected_parent_identity
    )
    completed_summary: BatchSummary | None = None
    transaction = None
    publication_completed = False
    try:
        parent.ensure_absent(parent.output_name)
        try:
            with parent.create_staging() as transaction:
                try:
                    completed_summary = _populate_batch_transaction(
                        transaction,
                        manifest,
                        output,
                        request_lineage,
                        adapter_factory=adapter_factory,
                        runner=runner,
                    )
                    parent.validate()
                    publication_completed = True
                except BaseException as error:
                    _rollback_batch_transaction(transaction, output, error)
                    raise
        except CompetitionNativePublicationError:
            raise
        except BaseException as error:
            if publication_completed and transaction is not None:
                _raise_transaction_exit_publication_error(
                    parent, transaction, output, error
                )
            raise
    except BaseException as error:
        try:
            parent.close()
        except BaseException as close_error:  # noqa: BLE001
            error.add_note(str(close_error))
        raise
    try:
        parent.close()
    except BaseException as close_error:
        if publication_completed:
            raise CompetitionNativePublicationError(
                output,
                published=True,
                recovery_name=output.name,
                detail="native batch published but retained parent close failed",
            ) from close_error
        raise
    if completed_summary is None:
        raise RuntimeError("native batch publication returned no summary")
    return completed_summary


def _populate_batch_transaction(
    transaction,
    manifest: BatchManifest,
    output: Path,
    request_lineage: dict[str, RequestLineage],
    *,
    adapter_factory,
    runner,
) -> BatchSummary:
    transaction.mkdir("accepted")
    attempts: list[BatchAttempt] = []
    for index, request in enumerate(manifest.requests):
        lineage = request_lineage[request.request_id]
        try:
            adapter_context = adapter_factory(
                [request.scene_id],
                width=manifest.width,
                height=manifest.height,
                seed=manifest.seed,
            )
            with adapter_context as adapter:
                execution = runner(adapter, request, request_lineage=lineage)
        except EndpointAuditRejected as error:
            attempts.append(
                _rejected_attempt(index, request, error.stage, error.reasons)
            )
            continue
        if type(execution) is not AuditExecution:
            raise TypeError("native batch runner returned an invalid execution")
        _require_execution_request_binding(manifest, request, execution)
        _require_request_lineage(request, execution, lineage)
        relative_case = f"accepted/{request.request_id}"
        case_root = output.parent / transaction.name / relative_case
        with bound_child_directory(transaction.descriptor, "accepted") as accepted_fd:
            bundle = publish_asset_bundle(
                execution,
                case_root,
                expected_parent_identity=directory_identity_fd(accepted_fd),
            )
            transaction.adopt_exact_tree(relative_case, regular_paths=_CASE_FILES)
            with bound_child_directory(accepted_fd, request.request_id) as case_fd:
                case_tree_sha256 = _case_tree_digest_fd(case_fd)
        attempts.append(
            BatchAttempt(
                attempt_index=index,
                request_id=request.request_id,
                outcome="accepted",
                case_path=relative_case,
                case_tree_sha256=case_tree_sha256,
                native_audit_run_sha256=(
                    execution.run.competition_native_audit_run_sha256
                ),
                native_asset_bundle_sha256=bundle.asset_bundle_sha256,
            )
        )
    manifest_payload = canonical_json_bytes_v2(manifest) + b"\n"
    outcomes_payload = b"".join(
        canonical_json_bytes_v2(item) + b"\n" for item in attempts
    )
    summary = _summary(manifest, tuple(attempts), manifest_payload, outcomes_payload)
    metadata = {
        "request-manifest.json": manifest_payload,
        "outcomes.jsonl": outcomes_payload,
        "summary.json": canonical_json_bytes_v2(summary) + b"\n",
    }
    if any(len(payload) > _MAX_METADATA_BYTES for payload in metadata.values()):
        raise ValueError("native batch root metadata exceeds byte limit")
    for name, payload in metadata.items():
        transaction.write(name, payload)
    checksum_payload = _root_checksum_payload_fd(transaction.descriptor)
    if len(checksum_payload) > _MAX_METADATA_BYTES:
        raise ValueError("native batch root checksum metadata exceeds byte limit")
    transaction.write("checksums.sha256", checksum_payload)
    transaction.fsync()
    seal = transaction.seal()
    verified = _verify_batch_fd(transaction.descriptor, request_lineage=request_lineage)
    if verified != summary:
        raise RuntimeError("native batch staging verification changed")
    transaction.validate_seal(seal)
    transaction.publish()
    transaction.validate_location(RenameLocation.OUTPUT)
    transaction.validate_seal(seal)
    with bound_child_directory(
        transaction.parent.parent_descriptor,
        transaction.parent.output_name,
    ) as final_descriptor:
        if directory_identity_fd(final_descriptor) != transaction.identity:
            raise RuntimeError("native batch final output identity changed")
        final_summary = _verify_batch_fd(
            final_descriptor,
            request_lineage=request_lineage,
        )
        if canonical_json_bytes_v2(final_summary) != canonical_json_bytes_v2(summary):
            raise RuntimeError("native batch final verification changed")
    transaction.validate_location(RenameLocation.OUTPUT)
    transaction.validate_seal(seal)
    return final_summary


def _rollback_batch_transaction(
    transaction, output: Path, active_error: object
) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native batch failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native batch failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if reconciliation.output is BindingStatus.FOREIGN:
            try:
                transaction.cleanup()
            except BaseException as cleanup_error:
                raise CompetitionNativePublicationError(
                    output,
                    published=None,
                    recovery_name=transaction.recovery_name,
                    detail=(
                        "native batch final binding was replaced and owned cleanup "
                        "failed"
                    ),
                ) from cleanup_error
            raise CompetitionNativePublicationError(
                output,
                published=None,
                recovery_name=None,
                detail="native batch final binding was replaced by a foreign entry",
            ) from active_error
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="native batch publication failed before commit",
            ) from active_error
        return
    try:
        transaction.rollback()
    except CompetitionNativePublicationError:
        raise
    except BaseException as rollback_error:
        try:
            final = transaction.reconcile()
        except BaseException:  # noqa: BLE001
            published = None
        else:
            published = (
                True
                if final.location is RenameLocation.OUTPUT
                else False
                if final.location is RenameLocation.SOURCE
                else None
            )
        raise CompetitionNativePublicationError(
            output,
            published=published,
            recovery_name=transaction.recovery_name,
            detail="native batch publication rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="native batch publication failed and was rolled back",
        ) from active_error


def _raise_transaction_exit_publication_error(
    parent, transaction, output: Path, active_error: object
) -> None:
    try:
        reconciliation = reconcile_owned_rename_at(
            parent.parent_descriptor,
            transaction.name,
            parent.output_name,
            transaction.identity,
        )
    except BaseException as reconciliation_error:  # noqa: BLE001
        active_error.add_note(str(reconciliation_error))
        published = None
        recovery_name = transaction.recovery_name
    else:
        published = (
            True
            if reconciliation.location is RenameLocation.OUTPUT
            else False
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
        recovery_name = (
            parent.output_name
            if reconciliation.location is RenameLocation.OUTPUT
            else transaction.name
            if reconciliation.location is RenameLocation.SOURCE
            else transaction.recovery_name
        )
    raise CompetitionNativePublicationError(
        output,
        published=published,
        recovery_name=recovery_name,
        detail="native batch transaction close failed after publication",
    ) from active_error


def verify_batch(
    root: Path,
    *,
    request_lineage: Mapping[str, RequestLineage],
    expected_root_identity: DirectoryIdentity | None = None,
) -> BatchSummary:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        with bound_absolute_directory(
            Path(os.path.abspath(root)), expected_identity=expected_root_identity
        ) as descriptor:
            return _verify_batch_fd(descriptor, request_lineage=request_lineage)


def prepare_batch_verification(
    root_descriptor: int,
    *,
    request_lineage: Mapping[str, RequestLineage],
) -> RetainedBatchVerification:
    if type(root_descriptor) is not int:
        raise TypeError("batch verification descriptor must be an exact integer")
    root_entries = snapshot_exact_directory(
        root_descriptor,
        regular_names=_ROOT_FILES,
        directory_names={"accepted"},
    )
    root_identity = directory_identity_fd(root_descriptor)
    checksum_payload = _sealed_batch_tree_checksum_payload_fd(root_descriptor)
    summary = _verify_batch_fd(root_descriptor, request_lineage=request_lineage)
    revalidate_entries(root_descriptor, root_entries)
    if _sealed_batch_tree_checksum_payload_fd(root_descriptor) != checksum_payload:
        raise ValueError("native batch tree changed during semantic verification")
    return RetainedBatchVerification(
        _capability=_BATCH_VERIFICATION_CAPABILITY,
        root_identity=root_identity,
        summary=summary,
        checksum_payload=checksum_payload,
        _root_entries=tuple(sorted(root_entries.items())),
    )


def revalidate_batch_verification(
    root_descriptor: int, retained: RetainedBatchVerification
) -> BatchSummary:
    if type(root_descriptor) is not int:
        raise TypeError("batch verification descriptor must be an exact integer")
    if (
        type(retained) is not RetainedBatchVerification
        or retained._capability is not _BATCH_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained batch verification capability must be exact")
    if directory_identity_fd(root_descriptor) != retained.root_identity:
        raise ValueError("retained native batch root identity changed")
    revalidate_entries(root_descriptor, dict(retained._root_entries))
    if (
        _sealed_batch_tree_checksum_payload_fd(root_descriptor)
        != retained.checksum_payload
    ):
        raise ValueError("retained native batch tree changed")
    if type(retained.summary) is not BatchSummary:
        raise TypeError("retained native batch summary must be exact")
    return BatchSummary.model_validate(
        retained.summary.model_dump(mode="python", warnings="error"), strict=True
    )


def _verify_batch_fd(
    root_descriptor: int,
    *,
    request_lineage: Mapping[str, RequestLineage],
) -> BatchSummary:
    root_entries = snapshot_exact_directory(
        root_descriptor,
        regular_names=_ROOT_FILES,
        directory_names={"accepted"},
    )
    manifest_payload = read_regular_at(
        root_descriptor,
        "request-manifest.json",
        _MAX_METADATA_BYTES,
        expected_stat=root_entries["request-manifest.json"],
    )
    require_wire_version(
        manifest_payload,
        artifact_kind="batch manifest",
        field="manifest_version",
        expected=_MANIFEST_VERSION,
    )
    manifest = BatchManifest.model_validate_json(manifest_payload, strict=True)
    if manifest_payload != canonical_json_bytes_v2(manifest) + b"\n":
        raise ValueError("native batch request manifest is not canonical")
    lineage = _normalize_request_lineage(manifest, request_lineage)
    outcomes_payload = read_regular_at(
        root_descriptor,
        "outcomes.jsonl",
        _MAX_METADATA_BYTES,
        expected_stat=root_entries["outcomes.jsonl"],
    )
    attempts = _parse_attempts(outcomes_payload)
    if len(attempts) != len(manifest.requests):
        raise ValueError("native batch outcomes do not cover the request prefix")
    for index, (request, attempt) in enumerate(
        zip(manifest.requests, attempts, strict=True)
    ):
        if attempt.attempt_index != index or attempt.request_id != request.request_id:
            raise ValueError("native batch outcome order changed")
    summary_payload = read_regular_at(
        root_descriptor,
        "summary.json",
        _MAX_METADATA_BYTES,
        expected_stat=root_entries["summary.json"],
    )
    require_wire_version(
        summary_payload,
        artifact_kind="batch summary",
        field="summary_version",
        expected=_SUMMARY_VERSION,
    )
    summary = BatchSummary.model_validate_json(summary_payload, strict=True)
    if summary_payload != canonical_json_bytes_v2(summary) + b"\n":
        raise ValueError("native batch summary is not canonical")
    expected_summary = _summary(manifest, attempts, manifest_payload, outcomes_payload)
    if summary != expected_summary:
        raise ValueError("native batch summary mismatch")
    ledger = _parse_root_checksums(
        read_regular_at(
            root_descriptor,
            "checksums.sha256",
            _MAX_METADATA_BYTES,
            expected_stat=root_entries["checksums.sha256"],
        )
    )
    accepted = tuple(item for item in attempts if item.outcome == "accepted")
    expected_ledger_names = _ROOT_FILES - {"checksums.sha256"} | {
        f"accepted/{item.request_id}/{name}"
        for item in accepted
        for name in _CASE_FILES
    }
    if set(ledger) != expected_ledger_names:
        raise ValueError("native batch checksum file set mismatch")
    for name, payload in {
        "request-manifest.json": manifest_payload,
        "outcomes.jsonl": outcomes_payload,
        "summary.json": summary_payload,
    }.items():
        if hashlib.sha256(payload).hexdigest() != ledger[name]:
            raise ValueError(f"native batch checksum mismatch: {name}")
    with bound_child_directory(root_descriptor, "accepted") as accepted_fd:
        accepted_entries = snapshot_exact_directory(
            accepted_fd,
            regular_names=set(),
            directory_names={item.request_id for item in accepted},
        )
        for request, attempt in zip(manifest.requests, attempts, strict=True):
            if attempt.outcome != "accepted":
                continue
            with bound_child_directory(accepted_fd, request.request_id) as case_fd:
                bundle, bundle_payload, case_entries = _read_bundle_metadata_fd(case_fd)
                _require_request_binding(manifest, request, attempt, bundle)
                _require_request_lineage(
                    request, bundle.native_audit_run, lineage[request.request_id]
                )
                prefix = f"accepted/{request.request_id}/"
                expected_case_digests = {
                    name: ledger[prefix + name] for name in _CASE_FILES
                }
                checked_bundle, case_digests = _verify_bundle_payloads_fd(
                    case_fd,
                    bundle,
                    bundle_payload,
                    case_entries,
                    bundle.native_audit_run,
                    expected_digests=expected_case_digests,
                )
                if checked_bundle != bundle or case_digests != expected_case_digests:
                    raise RuntimeError("native batch accepted replay changed")
                if (
                    _case_tree_digest_from_ledger(ledger, prefix)
                    != attempt.case_tree_sha256
                ):
                    raise ValueError("native batch accepted tree digest mismatch")
                revalidate_entries(case_fd, case_entries)
        revalidate_entries(accepted_fd, accepted_entries)
    revalidate_entries(root_descriptor, root_entries)
    return summary


def _require_request_binding(
    manifest: BatchManifest,
    request: BatchRequest,
    attempt: BatchAttempt,
    bundle: AssetBundle,
) -> None:
    run = bundle.native_audit_run
    if (
        run.case_id != request.case_id
        or run.source_scene.scene_id != request.scene_id
        or run.source_scene.object_by_id(run.intervention.subject_id).name
        != request.subject_name
        or run.source_scene.object_by_id(run.intervention.reference_id).name
        != request.reference_name
        or run.intervention.relation_before is not request.relation_before
        or run.intervention.relation_after is not request.relation_after
        or run.endpoint_workspace != request.endpoint_workspace
        or run.solver_config != request.solver_config
        or run.native_render_width_px != manifest.width
        or run.native_render_height_px != manifest.height
        or run.native_seed != manifest.seed
        or run.max_settlement_steps != request.max_settlement_steps
        or run.competition_native_audit_run_sha256 != attempt.native_audit_run_sha256
        or asset_bundle_sha256_from_checked(bundle)
        != attempt.native_asset_bundle_sha256
    ):
        raise ValueError("native batch accepted request binding mismatch")


def _require_execution_request_binding(
    manifest: BatchManifest,
    request: BatchRequest,
    execution: AuditExecution,
) -> None:
    run = execution.run
    if (
        run.case_id != request.case_id
        or run.source_scene.scene_id != request.scene_id
        or run.source_scene.object_by_id(run.intervention.subject_id).name
        != request.subject_name
        or run.source_scene.object_by_id(run.intervention.reference_id).name
        != request.reference_name
        or run.intervention.relation_before is not request.relation_before
        or run.intervention.relation_after is not request.relation_after
        or run.endpoint_workspace != request.endpoint_workspace
        or run.solver_config != request.solver_config
        or run.native_render_width_px != manifest.width
        or run.native_render_height_px != manifest.height
        or run.native_seed != manifest.seed
        or run.max_settlement_steps != request.max_settlement_steps
    ):
        raise ValueError("native batch runner returned a different request")


def _require_request_lineage(
    request: BatchRequest,
    execution_or_run,
    lineage: RequestLineage,
) -> None:
    run = (
        execution_or_run.run
        if type(execution_or_run) is AuditExecution
        else execution_or_run
    )
    audit = run.native_audit
    binding = run.proxy_bundle.binding
    if (
        lineage.batch_request_sha256 != request_binding_sha256(request)
        or run.source_capture != lineage.source_capture
        or run.surface_evidence != lineage.surface_evidence
        or legacy_sha256(run.source_scene) != lineage.frozen_source_scene_sha256
        or binding.source_capture_sha256 != lineage.source_capture_sha256
        or binding.runtime_identity_sha256 != lineage.runtime_identity_sha256
        or binding.placement_sha256 != lineage.placement_sha256
        or binding.surface_evidence_sha256 != lineage.surface_evidence_sha256
        or binding.subject_surface_evidence_sha256
        != lineage.subject_surface_evidence_sha256
        or binding.patch_index != lineage.patch_index
        or binding.patch_sha256 != lineage.patch_sha256
        or binding.spawn_map_source_sha256 != lineage.spawn_map_source_sha256
        or run.proxy_bundle.semantic_problem.semantic_problem_sha256
        != lineage.semantic_problem_sha256
        or run.proxy_bundle.proxy_bundle_sha256 != lineage.proxy_bundle_sha256
        or run.endpoint_plan_sha256 != lineage.endpoint_plan_sha256
        or run.solve_result.solve_result_sha256 != lineage.solve_result_sha256
        or run.camera_evidence != lineage.camera_evidence
        or run.camera_policy != lineage.camera_policy
        or run.runtime_pose_policy != lineage.runtime_pose_policy
        or binding.runtime_collision_delegated_native_object_ids
        != lineage.runtime_collision_delegated_native_object_ids
        or binding.visibility_semantics_id != lineage.visibility_semantics_id
        or binding.image_area_metric_definition_id
        != lineage.image_area_metric_definition_id
        or binding.image_area_metric_definition_version
        != lineage.image_area_metric_definition_version
        or binding.image_area_metric_formula != lineage.image_area_metric_formula
        or audit.source_capture_sha256 != lineage.source_capture_sha256
        or audit.runtime_identity_sha256 != lineage.runtime_identity_sha256
        or audit.placement_sha256 != lineage.placement_sha256
        or audit.surface_evidence_sha256 != lineage.surface_evidence_sha256
        or audit.subject_surface_evidence_sha256
        != lineage.subject_surface_evidence_sha256
        or audit.patch_index != lineage.patch_index
        or audit.patch_sha256 != lineage.patch_sha256
        or audit.spawn_map_source_sha256 != lineage.spawn_map_source_sha256
        or audit.endpoint_plan_sha256 != lineage.endpoint_plan_sha256
        or audit.camera_evidence != lineage.camera_evidence
        or audit.camera_policy != lineage.camera_policy
        or audit.runtime_pose_policy != lineage.runtime_pose_policy
        or audit.runtime_collision_delegated_native_object_ids
        != lineage.runtime_collision_delegated_native_object_ids
        or audit.visibility_semantics_id != lineage.visibility_semantics_id
        or audit.image_area_metric_definition_id
        != lineage.image_area_metric_definition_id
        or audit.image_area_metric_definition_version
        != lineage.image_area_metric_definition_version
        or audit.image_area_metric_formula != lineage.image_area_metric_formula
    ):
        raise ValueError("native batch request lineage mismatch")


def _summary(
    manifest: BatchManifest,
    attempts: tuple[BatchAttempt, ...],
    manifest_payload: bytes,
    outcomes_payload: bytes,
) -> BatchSummary:
    by_id = {item.request_id: item for item in manifest.requests}
    accepted = tuple(
        sorted(
            (
                by_id[item.request_id].relation_before
                for item in attempts
                if item.outcome == "accepted"
            ),
            key=lambda item: item.value,
        )
    )
    rejected = tuple(
        sorted(
            (
                by_id[item.request_id].relation_before
                for item in attempts
                if item.outcome == "rejected"
            ),
            key=lambda item: item.value,
        )
    )
    counts = Counter(item.stage for item in attempts if item.outcome == "rejected")
    return BatchSummary(
        batch_id=manifest.batch_id,
        request_count=len(manifest.requests),
        declared_relations=tuple(
            sorted(
                (item.relation_before for item in manifest.requests),
                key=lambda item: item.value,
            )
        ),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_relations=accepted,
        rejected_relations=rejected,
        rejected_by_stage=tuple(
            BatchStageCount(stage=stage, count=count)
            for stage, count in sorted(counts.items())
        ),
        request_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        outcomes_sha256=hashlib.sha256(outcomes_payload).hexdigest(),
    )


def _rejected_attempt(
    index: int,
    request: BatchRequest,
    stage: str,
    reasons: tuple[str, ...],
) -> BatchAttempt:
    return BatchAttempt(
        attempt_index=index,
        request_id=request.request_id,
        outcome="rejected",
        stage=stage,
        reasons=tuple(sorted(set(reasons))),
    )


def _parse_attempts(payload: bytes) -> tuple[BatchAttempt, ...]:
    if b"\r" in payload or (payload and not payload.endswith(b"\n")):
        raise ValueError("native batch outcomes encoding changed")
    attempts = []
    for line in payload.splitlines():
        item = BatchAttempt.model_validate_json(line, strict=True)
        if line != canonical_json_bytes_v2(item):
            raise ValueError("native batch outcome is not canonical")
        attempts.append(item)
    return tuple(attempts)


def _root_checksum_payload_fd(root_descriptor: int) -> bytes:
    try:
        checksum_stat = os.stat(
            "checksums.sha256", dir_fd=root_descriptor, follow_symlinks=False
        )
    except FileNotFoundError:
        checksum_stat = None
    regular_names = _ROOT_FILES - {"checksums.sha256"}
    if checksum_stat is not None:
        regular_names = _ROOT_FILES
    root_entries = snapshot_exact_directory(
        root_descriptor,
        regular_names=regular_names,
        directory_names={"accepted"},
    )
    digests = {
        name: hashlib.sha256(
            read_regular_at(
                root_descriptor,
                name,
                _MAX_METADATA_BYTES,
                expected_stat=root_entries[name],
            )
        ).hexdigest()
        for name in _ROOT_FILES - {"checksums.sha256"}
    }
    attempts = _parse_attempts(
        read_regular_at(
            root_descriptor,
            "outcomes.jsonl",
            _MAX_METADATA_BYTES,
            expected_stat=root_entries["outcomes.jsonl"],
        )
    )
    accepted_ids = {item.request_id for item in attempts if item.outcome == "accepted"}
    with bound_child_directory(root_descriptor, "accepted") as accepted_fd:
        accepted_entries = snapshot_exact_directory(
            accepted_fd, regular_names=set(), directory_names=accepted_ids
        )
        for request_id in sorted(accepted_entries):
            if _PORTABLE_COMPONENT.fullmatch(request_id) is None:
                raise ValueError("native batch accepted directory name changed")
            with bound_child_directory(accepted_fd, request_id) as case_fd:
                case_entries = snapshot_exact_directory(
                    case_fd, regular_names=_CASE_FILES
                )
                for name, item in case_entries.items():
                    limit = (
                        _MAX_METADATA_BYTES
                        if name.endswith((".json", ".sha256"))
                        else _MAX_ASSET_BYTES
                    )
                    digests[f"accepted/{request_id}/{name}"] = hashlib.sha256(
                        read_regular_at(case_fd, name, limit, expected_stat=item)
                    ).hexdigest()
                revalidate_entries(case_fd, case_entries)
        revalidate_entries(accepted_fd, accepted_entries)
    revalidate_entries(root_descriptor, root_entries)
    return "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(digests.items())
    ).encode("ascii")


def _sealed_batch_tree_checksum_payload_fd(root_descriptor: int) -> bytes:
    entries = snapshot_exact_directory(
        root_descriptor,
        regular_names=_ROOT_FILES,
        directory_names={"accepted"},
    )
    published = read_regular_at(
        root_descriptor,
        "checksums.sha256",
        _MAX_METADATA_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    recomputed = _root_checksum_payload_fd(root_descriptor)
    if published != recomputed:
        raise ValueError("native batch checksum payload changed")
    revalidate_entries(root_descriptor, entries)
    return recomputed


def _parse_root_checksums(payload: bytes) -> dict[str, str]:
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError("native batch checksum encoding changed")
    entries: dict[str, str] = {}
    for line in payload.decode("ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or _PORTABLE_COMPONENT.fullmatch(relative.split("/")[-1]) is None
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or relative in entries
        ):
            raise ValueError("native batch checksum syntax changed")
        entries[relative] = digest
    if tuple(entries) != tuple(sorted(entries)):
        raise ValueError("native batch checksums are not canonical")
    return entries


def _case_tree_digest_fd(descriptor: int) -> str:
    entries = snapshot_exact_directory(descriptor, regular_names=_CASE_FILES)
    ledger = {
        name: hashlib.sha256(
            read_regular_at(
                descriptor,
                name,
                _MAX_METADATA_BYTES
                if name.endswith((".json", ".sha256"))
                else _MAX_ASSET_BYTES,
                expected_stat=entries[name],
            )
        ).hexdigest()
        for name in _CASE_FILES
    }
    revalidate_entries(descriptor, entries)
    return _case_tree_digest_from_ledger(ledger, "")


def _case_tree_digest_from_ledger(ledger: dict[str, str], prefix: str) -> str:
    inventory = "".join(
        f"{digest}  {relative.removeprefix(prefix)}\n"
        for relative, digest in sorted(ledger.items())
        if relative.startswith(prefix)
    ).encode("ascii")
    return hashlib.sha256(inventory).hexdigest()


__all__ = (
    "BatchAttempt",
    "BatchManifest",
    "BatchRequest",
    "BatchStageCount",
    "BatchSummary",
    "RetainedBatchVerification",
    "execute_batch",
    "load_batch_manifest",
    "prepare_batch_verification",
    "revalidate_batch_verification",
    "verify_batch",
)
