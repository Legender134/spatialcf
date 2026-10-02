"""Explicit source campaign storage owner."""

from __future__ import annotations

import hashlib

import json

import os

import warnings

from pathlib import (
    Path,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.generation.errors import (
    require_wire_version,
)
from spatialcf.generation.planning.native_versions import require_planning_wire

from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    RenameLocation,
    bound_absolute_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

from spatialcf.generation.planning.campaign_contracts import (
    RetainedSourcePlan,
    RetainedSourcePlanVerification,
    SourcePlan,
    SourcePolicy,
    _FILES,
    _MAX_CHECKSUM_BYTES,
    _MAX_PLAN_BYTES,
    _MAX_POLICY_BYTES,
    _SOURCE_PLAN_CAPABILITY,
    _SOURCE_PLAN_VERIFICATION_CAPABILITY,
    _SOURCE_PLAN_VERSION,
    _SOURCE_POLICY_VERSION,
)


def _parse_source_plan(payload: bytes) -> SourcePlan:
    """Parse only the current plan after the caller's version preflight."""

    require_planning_wire(
        payload,
        artifact_kind="source plan",
        field="plan_version",
        expected=_SOURCE_PLAN_VERSION,
    )
    plan = SourcePlan.model_validate_json(payload, strict=True)
    if type(plan) is not SourcePlan:
        raise TypeError("source plan must be exact SourcePlan")
    if (
        payload
        != canonical_json_bytes(plan.model_dump(mode="json", warnings="error")) + b"\n"
    ):
        raise ValueError("native source plan is not canonical")
    return plan


def _load_source_policy_fd(
    descriptor: int,
    name: str,
    expected_stat: os.stat_result,
) -> SourcePolicy:
    payload = read_regular_at(
        descriptor, name, _MAX_POLICY_BYTES, expected_stat=expected_stat
    )
    require_planning_wire(
        payload,
        artifact_kind="source policy",
        field="policy_version",
        expected=_SOURCE_POLICY_VERSION,
    )
    policy = SourcePolicy.model_validate_json(payload, strict=True)
    if type(policy) is not SourcePolicy:
        raise TypeError("source policy must be exact SourcePolicy")
    if payload != canonical_json_bytes(policy) + b"\n":
        raise ValueError("native source policy is not canonical")
    return policy


def load_source_policy(path: Path) -> SourcePolicy:
    if not isinstance(path, Path):
        raise TypeError("source policy path must be a Path")
    absolute = Path(os.path.abspath(path))
    with bound_absolute_directory(absolute.parent) as descriptor:
        item = os.stat(absolute.name, dir_fd=descriptor, follow_symlinks=False)
        policy = _load_source_policy_fd(descriptor, absolute.name, item)
        revalidate_entries(descriptor, {absolute.name: item})
        return policy


def _load_source_plan_fd(descriptor: int) -> SourcePlan:
    entries = snapshot_exact_directory(descriptor, regular_names=_FILES)
    payload = read_regular_at(
        descriptor,
        "plan.json",
        _MAX_PLAN_BYTES,
        expected_stat=entries["plan.json"],
    )
    require_planning_wire(
        payload,
        artifact_kind="source plan",
        field="plan_version",
        expected=_SOURCE_PLAN_VERSION,
    )
    plan = _parse_source_plan(payload)
    checksum = read_regular_at(
        descriptor,
        "checksums.sha256",
        _MAX_CHECKSUM_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    expected = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
    if checksum != expected:
        raise ValueError("native source plan checksum mismatch")
    revalidate_entries(descriptor, entries)
    return plan


def prepare_source_plan_verification(
    root_descriptor: int,
) -> RetainedSourcePlanVerification:
    """Verify a current source plan under an already-retained descriptor."""

    if type(root_descriptor) is not int:
        raise TypeError("source plan descriptor must be an exact integer")
    entries = snapshot_exact_directory(root_descriptor, regular_names=_FILES)
    identity = directory_identity_fd(root_descriptor)
    plan = _load_source_plan_fd(root_descriptor)
    revalidate_entries(root_descriptor, entries)
    return RetainedSourcePlanVerification(
        _capability=_SOURCE_PLAN_VERIFICATION_CAPABILITY,
        root_identity=identity,
        plan=plan,
        _entries=tuple(sorted(entries.items())),
    )


def revalidate_source_plan_verification(
    root_descriptor: int,
    retained: RetainedSourcePlanVerification,
) -> SourcePlan:
    """Revalidate a prepared source plan without a new snapshot baseline."""

    if type(root_descriptor) is not int:
        raise TypeError("source plan descriptor must be an exact integer")
    if (
        type(retained) is not RetainedSourcePlanVerification
        or retained._capability is not _SOURCE_PLAN_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained source plan verification must be exact")
    if directory_identity_fd(root_descriptor) != retained.root_identity:
        raise ValueError("retained source plan root identity changed")
    revalidate_entries(root_descriptor, dict(retained._entries))
    return SourcePlan.model_validate(
        retained.plan.model_dump(mode="python", warnings="error"),
        strict=True,
    )


def load_source_plan(root: Path) -> SourcePlan:
    if not isinstance(root, Path):
        raise TypeError("source plan root must be a Path")
    with bound_absolute_directory(root) as descriptor:
        return _load_source_plan_fd(descriptor)


def _retain_checked_source_plan(plan: SourcePlan) -> RetainedSourcePlan:
    if type(plan) is not SourcePlan:
        raise TypeError("retained source plan payload must be exact SourcePlan")
    payload = (
        canonical_json_bytes(plan.model_dump(mode="json", warnings="error")) + b"\n"
    )
    return RetainedSourcePlan(
        _capability=_SOURCE_PLAN_CAPABILITY,
        plan=plan,
        plan_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_source_plan_retained(root: Path) -> RetainedSourcePlan:
    if not isinstance(root, Path):
        raise TypeError("source plan root must be a Path")
    with bound_absolute_directory(root) as descriptor:
        return _retain_checked_source_plan(_load_source_plan_fd(descriptor))


def _rollback_transaction(
    transaction: object, output: Path, active_error: object
) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native source plan failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native source plan failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="native source plan publication failed before commit",
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
            detail="native source plan publication rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="native source plan publication failed and was rolled back",
        ) from active_error


def _raise_transaction_exit_error(
    parent: object,
    transaction: object,
    output: Path,
    active_error: BaseException,
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
        detail="native source plan transaction close failed after publication",
    ) from active_error


def publish_source_plan(plan: SourcePlan, output_root: Path) -> SourcePlan:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(plan) is not SourcePlan:
            raise TypeError("native source plan must be exact SourcePlan")
        if not isinstance(output_root, Path):
            raise TypeError("native source plan output_root must be a Path")
        checked = SourcePlan.model_validate(
            plan.model_dump(mode="python", warnings="error"), strict=True
        )
        payload = canonical_json_bytes(checked) + b"\n"
        if len(payload) > _MAX_PLAN_BYTES:
            raise ValueError("native source plan exceeds byte limit")
        checksum = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
        output = Path(os.path.abspath(output_root))
        parent = open_native_output_parent(output)
        transaction = None
        publication_completed = False
        try:
            parent.ensure_absent(parent.output_name)
            try:
                with parent.create_staging(label="plan") as transaction:
                    try:
                        transaction.write("plan.json", payload)
                        transaction.write("checksums.sha256", checksum)
                        transaction.fsync()
                        seal = transaction.seal()
                        verified = _load_source_plan_fd(transaction.descriptor)
                        if verified != checked:
                            raise RuntimeError(
                                "sealed native source plan verification changed"
                            )
                        transaction.validate_seal(seal)
                        transaction.publish()
                        transaction.validate_location(RenameLocation.OUTPUT)
                        transaction.validate_seal(seal)
                        parent.validate()
                        publication_completed = True
                    except BaseException as error:
                        _rollback_transaction(transaction, output, error)
                        raise
            except CompetitionNativePublicationError:
                raise
            except BaseException as error:
                if publication_completed and transaction is not None:
                    _raise_transaction_exit_error(parent, transaction, output, error)
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
                detail="native source plan published but retained parent close failed",
            ) from close_error
        return checked


# Preserve supported public names and pickle lookup.
_parse_source_plan.__module__ = "spatialcf.generation.planning.campaign"
_load_source_policy_fd.__module__ = "spatialcf.generation.planning.campaign"
load_source_policy.__module__ = "spatialcf.generation.planning.campaign"
_load_source_plan_fd.__module__ = "spatialcf.generation.planning.campaign"
prepare_source_plan_verification.__module__ = "spatialcf.generation.planning.campaign"
revalidate_source_plan_verification.__module__ = "spatialcf.generation.planning.campaign"
load_source_plan.__module__ = "spatialcf.generation.planning.campaign"
_retain_checked_source_plan.__module__ = "spatialcf.generation.planning.campaign"
load_source_plan_retained.__module__ = "spatialcf.generation.planning.campaign"
_rollback_transaction.__module__ = "spatialcf.generation.planning.campaign"
_raise_transaction_exit_error.__module__ = "spatialcf.generation.planning.campaign"
publish_source_plan.__module__ = "spatialcf.generation.planning.campaign"
