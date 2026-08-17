"""Descriptor-safe storage for the current capture roster."""

from __future__ import annotations

import hashlib
import os
import stat
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from spatialcf.domain.v2.serialization import canonical_json_bytes_v2
from spatialcf.generation._internal.evidence.camera import (
    SourceCameraEvidence,
)
from spatialcf.generation._internal.evidence.reachability import (
    CandidateTargetReachability,
)
from spatialcf.generation._internal.evidence.surface import (
    SourceSurfaceEvidence,
)
from spatialcf.generation.capture.compiler import compile_roster
from spatialcf.generation.capture.models import (
    CompetitionNativeCandidateInventoryV2_9,
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativeObjectInventoryV2_9,
    CompetitionNativeRosterRejectionV2_9,
    CompetitionNativeSourceCaptureOutcomeV2_9,
    RosterCompilation,
    RosterPolicy,
    RosterSummary,
)
from spatialcf.generation.errors import require_wire_version
from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
)

ROSTER_FILES = frozenset(
    {
        "policy.json",
        "scene-inventory.jsonl",
        "object-inventory.jsonl",
        "candidate-inventory.jsonl",
        "request-manifest.json",
        "rejections.jsonl",
        "summary.json",
        "surface-evidence.jsonl",
        "camera-evidence.jsonl",
        "target-reachability.jsonl",
        "checksums.sha256",
    }
)
_GLOBAL_FILE_MAX_BYTES = 1024 * 1024 * 1024
_POLICY_SOURCE_MAX_BYTES = 8 * 1024
_POLICY_FIXED_MAX_BYTES = 32 * 1024
_POLICY_MAX_BYTES = _POLICY_FIXED_MAX_BYTES + 512 * _POLICY_SOURCE_MAX_BYTES
_SOURCE_INVENTORY_RECORD_MAX_BYTES = 8 * 1024 * 1024
_SURFACE_EVIDENCE_RECORD_MAX_BYTES = 2 * 1024 * 1024
_CAMERA_EVIDENCE_RECORD_MAX_BYTES = 256 * 1024
_TARGET_REACHABILITY_RECORD_MAX_BYTES = 32 * 1024
_OBJECT_INVENTORY_RECORD_MAX_BYTES = 32 * 1024
_CANDIDATE_INVENTORY_RECORD_MAX_BYTES = 40 * 1024
_REJECTION_RECORD_MAX_BYTES = 24 * 1024
_SUMMARY_MAX_BYTES = 64 * 1024
_CHECKSUM_MAX_BYTES = 4 * 1024
_REQUEST_MANIFEST_REQUEST_MAX_BYTES = 32 * 1024
_REQUEST_MANIFEST_FIXED_MAX_BYTES = 16 * 1024
_ROSTER_VERIFICATION_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class RetainedRosterVerification:
    """One roster compilation bound to its original directory snapshot."""

    _capability: object = field(repr=False, compare=False)
    root_identity: DirectoryIdentity
    compilation: RosterCompilation
    _entries: tuple[tuple[str, os.stat_result], ...] = field(
        repr=False,
        compare=False,
    )


def _request_manifest_max_bytes(policy: RosterPolicy) -> int:
    return (
        _REQUEST_MANIFEST_FIXED_MAX_BYTES
        + policy.max_requests_total * _REQUEST_MANIFEST_REQUEST_MAX_BYTES
    )


def _json_line(value: object) -> bytes:
    return canonical_json_bytes_v2(value) + b"\n"


def _publication_payloads(compilation: RosterCompilation) -> dict[str, bytes]:
    payloads = {
        "policy.json": _json_line(compilation.policy),
        "scene-inventory.jsonl": b"".join(
            _json_line(item) for item in compilation.scene_inventory
        ),
        "object-inventory.jsonl": b"".join(
            _json_line(item) for item in compilation.object_inventory
        ),
        "candidate-inventory.jsonl": b"".join(
            _json_line(item) for item in compilation.candidate_inventory
        ),
        "request-manifest.json": _json_line(compilation.request_manifest),
        "rejections.jsonl": b"".join(
            _json_line(item) for item in compilation.rejections
        ),
        "summary.json": _json_line(compilation.summary),
        "surface-evidence.jsonl": b"".join(
            _json_line(item) for item in compilation.surface_evidence
        ),
        "camera-evidence.jsonl": b"".join(
            _json_line(item) for item in compilation.camera_evidence
        ),
        "target-reachability.jsonl": b"".join(
            _json_line(item) for item in compilation.target_reachability
        ),
    }
    payloads["checksums.sha256"] = b"".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
        for name in sorted(payloads)
    )
    return payloads


def _rollback_roster_transaction(transaction, output: Path, active_error) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="candidate roster failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="candidate roster failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="candidate roster publication failed before commit",
            ) from active_error
        return
    try:
        transaction.rollback()
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
            detail="candidate roster rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="candidate roster publication failed and was rolled back",
        ) from active_error


def _raise_roster_transaction_exit_error(
    parent,
    transaction,
    output: Path,
    active_error,
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
        detail="candidate roster transaction close failed after publication",
    ) from active_error


def publish_roster(
    compilation: RosterCompilation,
    output_root: Path,
) -> RosterSummary:
    """Atomically publish a fresh-verified, no-overwrite current roster."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(compilation) is not RosterCompilation:
            raise TypeError("candidate roster compilation must be exact")
        if not isinstance(output_root, Path):
            raise TypeError("candidate roster output_root must be a Path")
        checked = RosterCompilation.model_validate(
            compilation.model_dump(mode="python", warnings="error"),
            strict=True,
        )
        recomputed = compile_roster(
            checked.policy,
            checked.scene_inventory,
            checked.surface_evidence,
            checked.camera_evidence,
        )
        if checked != recomputed:
            raise ValueError("candidate roster compilation is not freshly reproducible")
        payloads = _publication_payloads(checked)
        output = Path(os.path.abspath(output_root))
        parent = open_native_output_parent(output)
        transaction = None
        publication_completed = False
        try:
            parent.ensure_absent(parent.output_name)
            try:
                with parent.create_staging() as transaction:
                    try:
                        for name in sorted(payloads):
                            transaction.write(name, payloads[name])
                        transaction.fsync()
                        seal = transaction.seal()
                        verified = _verify_roster_tree_fd(transaction.descriptor)
                        if verified != checked:
                            raise RuntimeError(
                                "sealed candidate roster verification changed"
                            )
                        transaction.validate_seal(seal)
                        transaction.publish()
                        transaction.validate_location(RenameLocation.OUTPUT)
                        transaction.validate_seal(seal)
                        parent.validate()
                        publication_completed = True
                    except BaseException as error:
                        _rollback_roster_transaction(transaction, output, error)
                        raise
            except CompetitionNativePublicationError:
                raise
            except BaseException as error:
                if publication_completed and transaction is not None:
                    _raise_roster_transaction_exit_error(
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
            published: bool | None = True
            recovery_name: str | None = output.name
            if transaction is not None:
                try:
                    reconciliation = reconcile_owned_rename_at(
                        parent.parent_descriptor,
                        transaction.name,
                        parent.output_name,
                        transaction.identity,
                    )
                except BaseException as reconciliation_error:  # noqa: BLE001
                    close_error.add_note(str(reconciliation_error))
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
                detail="candidate roster published but retained parent close failed",
            ) from close_error
        return checked.summary


def _read_canonical_json_fd(
    descriptor: int, name: str, model_type, entries, limit: int
):
    payload = read_regular_at(descriptor, name, limit, expected_stat=entries[name])
    if not payload.endswith(b"\n"):
        raise ValueError(f"candidate roster {name} requires canonical LF")
    value = model_type.model_validate_json(payload, strict=True)
    if payload != _json_line(value):
        raise ValueError(f"candidate roster {name} is not canonical")
    return value, hashlib.sha256(payload).hexdigest()


def _stat_fingerprint_v2_9(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _read_canonical_jsonl_fd(
    descriptor: int,
    name: str,
    model_type,
    entries,
    *,
    maximum_records: int,
    record_max_bytes: int,
) -> tuple[tuple[object, ...], str]:
    maximum_file_bytes = min(
        _GLOBAL_FILE_MAX_BYTES,
        maximum_records * record_max_bytes,
    )
    expected = entries[name]
    if expected.st_size > maximum_file_bytes:
        raise ValueError(f"candidate roster {name} exceeds policy-derived byte cap")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_descriptor = os.open(name, flags, dir_fd=descriptor)
    values: list[object] = []
    digest = hashlib.sha256()
    buffer = b""
    total = 0
    try:
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_fingerprint_v2_9(before) != _stat_fingerprint_v2_9(expected)
        ):
            raise ValueError(f"candidate roster input must be regular: {name}")
        while True:
            chunk = os.read(file_descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_file_bytes:
                raise ValueError(f"candidate roster {name} exceeds byte cap")
            digest.update(chunk)
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    raise ValueError(
                        f"candidate roster {name} contains an empty record"
                    )
                if len(line) + 1 > record_max_bytes:
                    raise ValueError(f"candidate roster {name} record exceeds byte cap")
                if len(values) >= maximum_records:
                    raise ValueError(f"candidate roster {name} exceeds record cap")
                value = model_type.model_validate_json(line, strict=True)
                if line + b"\n" != _json_line(value):
                    raise ValueError(f"candidate roster {name} record is not canonical")
                values.append(value)
            if len(buffer) + 1 > record_max_bytes:
                raise ValueError(f"candidate roster {name} record exceeds byte cap")
        after = os.fstat(file_descriptor)
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            buffer
            or total != after.st_size
            or _stat_fingerprint_v2_9(before) != _stat_fingerprint_v2_9(after)
            or _stat_fingerprint_v2_9(after) != _stat_fingerprint_v2_9(current)
        ):
            raise ValueError(f"candidate roster input changed while read: {name}")
    finally:
        os.close(file_descriptor)
    return tuple(values), digest.hexdigest()


def _verify_checksum_payload(payload: bytes, first_seven: dict[str, str]) -> None:
    expected_names = set(first_seven)
    seen: set[str] = set()
    if not payload.endswith(b"\n"):
        raise ValueError("candidate roster checksum ledger requires final LF")
    for line in payload.decode("ascii").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("candidate roster checksum ledger syntax is invalid")
        digest, name = line[:64], line[66:]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or name not in expected_names
            or name in seen
        ):
            raise ValueError("candidate roster checksum ledger entries are invalid")
        seen.add(name)
        if digest != first_seven[name]:
            raise ValueError("candidate roster checksum digest mismatch")
    if seen != expected_names:
        raise ValueError("candidate roster checksum file set mismatch")
    expected = b"".join(
        f"{first_seven[name]}  {name}\n".encode("ascii") for name in sorted(first_seven)
    )
    if payload != expected:
        raise ValueError("candidate roster checksum ledger is not canonical")


def _read_policy_fd(descriptor: int, entries) -> tuple[RosterPolicy, str]:
    payload = read_regular_at(
        descriptor,
        "policy.json",
        _POLICY_MAX_BYTES,
        expected_stat=entries["policy.json"],
    )
    require_wire_version(
        payload,
        artifact_kind="candidate roster policy",
        field="policy_version",
        expected="competition-native-candidate-roster-policy:2.9.4",
    )
    policy = RosterPolicy.model_validate_json(payload, strict=True)
    if payload != _json_line(policy):
        raise ValueError("candidate roster policy.json is not canonical")
    return policy, hashlib.sha256(payload).hexdigest()


def _verify_roster_tree_fd(descriptor: int) -> RosterCompilation:
    """Verify local file integrity and freshly recompile every derived row."""

    initial_entries = scan_directory(descriptor, maximum_entries=len(ROSTER_FILES))
    if "policy.json" not in initial_entries:
        raise ValueError("candidate roster policy.json is missing")
    _read_policy_fd(descriptor, initial_entries)
    entries = snapshot_exact_directory(descriptor, regular_names=ROSTER_FILES)
    policy, policy_digest = _read_policy_fd(descriptor, entries)
    scene_inventory, scene_digest = _read_canonical_jsonl_fd(
        descriptor,
        "scene-inventory.jsonl",
        CompetitionNativeSourceCaptureOutcomeV2_9,
        entries,
        maximum_records=policy.max_scenes,
        record_max_bytes=_SOURCE_INVENTORY_RECORD_MAX_BYTES,
    )
    object_limit = policy.max_scenes * policy.max_objects_per_scene
    object_inventory, object_digest = _read_canonical_jsonl_fd(
        descriptor,
        "object-inventory.jsonl",
        CompetitionNativeObjectInventoryV2_9,
        entries,
        maximum_records=object_limit,
        record_max_bytes=_OBJECT_INVENTORY_RECORD_MAX_BYTES,
    )
    candidate_inventory, candidate_digest = _read_canonical_jsonl_fd(
        descriptor,
        "candidate-inventory.jsonl",
        CompetitionNativeCandidateInventoryV2_9,
        entries,
        maximum_records=policy.max_candidates_total,
        record_max_bytes=_CANDIDATE_INVENTORY_RECORD_MAX_BYTES,
    )
    manifest, manifest_digest = _read_canonical_json_fd(
        descriptor,
        "request-manifest.json",
        CompetitionNativeCandidateRosterManifestV2_9,
        entries,
        _request_manifest_max_bytes(policy),
    )
    rejection_limit = policy.max_scenes + object_limit * 2 + policy.max_candidates_total
    rejections, rejection_digest = _read_canonical_jsonl_fd(
        descriptor,
        "rejections.jsonl",
        CompetitionNativeRosterRejectionV2_9,
        entries,
        maximum_records=rejection_limit,
        record_max_bytes=_REJECTION_RECORD_MAX_BYTES,
    )
    summary, summary_digest = _read_canonical_json_fd(
        descriptor,
        "summary.json",
        RosterSummary,
        entries,
        _SUMMARY_MAX_BYTES,
    )
    surface_evidence, surface_digest = _read_canonical_jsonl_fd(
        descriptor,
        "surface-evidence.jsonl",
        SourceSurfaceEvidence,
        entries,
        maximum_records=policy.max_scenes,
        record_max_bytes=_SURFACE_EVIDENCE_RECORD_MAX_BYTES,
    )
    camera_evidence, camera_digest = _read_canonical_jsonl_fd(
        descriptor,
        "camera-evidence.jsonl",
        SourceCameraEvidence,
        entries,
        maximum_records=policy.max_scenes,
        record_max_bytes=_CAMERA_EVIDENCE_RECORD_MAX_BYTES,
    )
    target_reachability, target_digest = _read_canonical_jsonl_fd(
        descriptor,
        "target-reachability.jsonl",
        CandidateTargetReachability,
        entries,
        maximum_records=policy.max_candidates_total,
        record_max_bytes=_TARGET_REACHABILITY_RECORD_MAX_BYTES,
    )
    digests = {
        "policy.json": policy_digest,
        "scene-inventory.jsonl": scene_digest,
        "object-inventory.jsonl": object_digest,
        "candidate-inventory.jsonl": candidate_digest,
        "request-manifest.json": manifest_digest,
        "rejections.jsonl": rejection_digest,
        "summary.json": summary_digest,
        "surface-evidence.jsonl": surface_digest,
        "camera-evidence.jsonl": camera_digest,
        "target-reachability.jsonl": target_digest,
    }
    checksum_payload = read_regular_at(
        descriptor,
        "checksums.sha256",
        _CHECKSUM_MAX_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    _verify_checksum_payload(checksum_payload, digests)
    recomputed = compile_roster(
        policy,
        tuple(scene_inventory),
        tuple(surface_evidence),
        tuple(camera_evidence),
    )
    if (
        recomputed.object_inventory != object_inventory
        or recomputed.candidate_inventory != candidate_inventory
        or recomputed.request_manifest != manifest
        or recomputed.rejections != rejections
        or recomputed.summary != summary
        or recomputed.target_reachability != target_reachability
    ):
        raise ValueError("candidate roster persisted compilation changed on replay")
    revalidate_entries(descriptor, entries)
    return recomputed


def prepare_roster_verification(
    root_descriptor: int,
) -> RetainedRosterVerification:
    """Verify a current roster under an already-retained descriptor."""

    if type(root_descriptor) is not int:
        raise TypeError("roster descriptor must be an exact integer")
    entries = snapshot_exact_directory(root_descriptor, regular_names=ROSTER_FILES)
    identity = directory_identity_fd(root_descriptor)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        compilation = _verify_roster_tree_fd(root_descriptor)
    revalidate_entries(root_descriptor, entries)
    return RetainedRosterVerification(
        _capability=_ROSTER_VERIFICATION_CAPABILITY,
        root_identity=identity,
        compilation=compilation,
        _entries=tuple(sorted(entries.items())),
    )


def revalidate_roster_verification(
    root_descriptor: int,
    retained: RetainedRosterVerification,
) -> RosterCompilation:
    """Revalidate a prepared roster without establishing a new baseline."""

    if type(root_descriptor) is not int:
        raise TypeError("roster descriptor must be an exact integer")
    if (
        type(retained) is not RetainedRosterVerification
        or retained._capability is not _ROSTER_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained roster verification must be exact")
    if directory_identity_fd(root_descriptor) != retained.root_identity:
        raise ValueError("retained roster root identity changed")
    revalidate_entries(root_descriptor, dict(retained._entries))
    return RosterCompilation.model_validate(
        retained.compilation.model_dump(mode="python", warnings="error"),
        strict=True,
    )


def load_roster(root: Path) -> RosterCompilation:
    """Fresh-read and recompile one complete current candidate roster."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if not isinstance(root, Path):
            raise TypeError("candidate roster root must be a Path")
        with bound_absolute_directory(root) as descriptor:
            return _verify_roster_tree_fd(descriptor)


def verify_roster(root: Path) -> RosterSummary:
    return load_roster(root).summary


__all__ = (
    "ROSTER_FILES",
    "RetainedRosterVerification",
    "load_roster",
    "prepare_roster_verification",
    "publish_roster",
    "revalidate_roster_verification",
    "verify_roster",
)
