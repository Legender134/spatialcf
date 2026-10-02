"""Authenticate retained dataset artifacts and reconstruct verified indexes."""

from __future__ import annotations

import hashlib

import os

from collections.abc import (
    Mapping,
)

from contextlib import (
    ExitStack,
)

from pathlib import (
    Path,
    PurePosixPath,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

import spatialcf.generation.capture as capture
import spatialcf.generation.execution as execution
import spatialcf.generation.planning as planning
import spatialcf.generation.publication as publication

from spatialcf.generation.dataset_models import (
    DatasetManifest,
    DatasetRecord,
    GenerationReport,
    _canonical_model_bytes,
    _safe_relative_path,
)

from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    read_regular_at,
    revalidate_entries,
    snapshot_exact_directory,
)

from spatialcf.generation.workflows._dataset.contracts import (
    _BUNDLE_FILES,
    _MAX_ASSET_BYTES,
    _MAX_METADATA_BYTES,
    _PUBLIC_DIRECTORIES,
    _PUBLIC_FILES,
    _RetainedBatchState,
    _RetainedCampaignState,
    _STATE_DIRECTORIES,
    _VerifiedAttempt,
)

from spatialcf.generation.workflows._dataset.derivation import (
    _dataset_tree_sha256,
    _derive_dataset,
)

from spatialcf.generation.workflows._dataset.state import (
    _capture_plan,
    _config_from_capture_plan,
    _existing_names,
)


def _parse_attempts(payload: bytes) -> tuple[execution.BatchAttempt, ...]:
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError("dataset batch outcomes require canonical LF")
    attempts = tuple(
        execution.BatchAttempt.model_validate_json(line, strict=True)
        for line in payload.splitlines()
    )
    canonical = b"".join(_canonical_model_bytes(item) + b"\n" for item in attempts)
    if canonical != payload:
        raise ValueError("dataset batch outcomes are not canonical")
    return attempts


def _verified_bundle_at_path(source_root: Path) -> publication.AssetBundle:
    loaded = publication.load_asset_bundle(source_root)
    return publication.verify_asset_bundle(source_root, loaded.native_audit_run)


def _verified_bundle_fd(descriptor: int) -> publication.AssetBundle:
    entries = snapshot_exact_directory(descriptor, regular_names=_BUNDLE_FILES)
    payload = read_regular_at(
        descriptor,
        "bundle.json",
        _MAX_METADATA_BYTES,
        expected_stat=entries["bundle.json"],
    )
    bundle = publication.AssetBundle.model_validate_json(payload, strict=True)
    if payload != canonical_json_bytes(bundle) + b"\n":
        raise ValueError("dataset accepted bundle metadata is not canonical")
    checked = publication.verify_asset_bundle_fd(
        descriptor,
        bundle.native_audit_run,
    )
    revalidate_entries(descriptor, entries)
    return checked


def _path_attempts(
    plan: planning.SourcePlan,
    batches_root: Path,
) -> dict[str, _VerifiedAttempt]:
    attempts: dict[str, _VerifiedAttempt] = {}
    for batch in plan.batches:
        batch_root = batches_root / batch.batch_id
        with bound_absolute_directory(batch_root) as descriptor:
            entries = _existing_names(descriptor, maximum=5)
            item = entries.get("outcomes.jsonl")
            if item is None:
                raise ValueError("dataset batch outcomes are absent")
            payload = read_regular_at(
                descriptor,
                "outcomes.jsonl",
                _MAX_METADATA_BYTES,
                expected_stat=item,
            )
            parsed = _parse_attempts(payload)
            if tuple(item.request_id for item in parsed) != tuple(
                item.request_id for item in batch.requests
            ):
                raise ValueError("dataset batch outcome membership changed")
            for attempt in parsed:
                if attempt.request_id in attempts:
                    raise ValueError("dataset batch request outcome is duplicated")
                source_root = (
                    batch_root / attempt.case_path
                    if attempt.outcome == "accepted" and attempt.case_path is not None
                    else None
                )
                bundle = (
                    _verified_bundle_at_path(source_root)
                    if source_root is not None
                    else None
                )
                attempts[attempt.request_id] = _VerifiedAttempt(
                    attempt=attempt,
                    bundle=bundle,
                    source_root=source_root,
                )
            revalidate_entries(descriptor, entries)
    return attempts


def _descriptor_attempts(
    plan: planning.SourcePlan,
    batch_descriptors: Mapping[str, int],
) -> dict[str, _VerifiedAttempt]:
    attempts: dict[str, _VerifiedAttempt] = {}
    for batch in plan.batches:
        descriptor = batch_descriptors[batch.batch_id]
        item = os.stat("outcomes.jsonl", dir_fd=descriptor, follow_symlinks=False)
        payload = read_regular_at(
            descriptor,
            "outcomes.jsonl",
            _MAX_METADATA_BYTES,
            expected_stat=item,
        )
        parsed = _parse_attempts(payload)
        if tuple(row.request_id for row in parsed) != tuple(
            request.request_id for request in batch.requests
        ):
            raise ValueError("dataset batch outcome membership changed")
        accepted = tuple(row for row in parsed if row.outcome == "accepted")
        with bound_child_directory(descriptor, "accepted") as accepted_fd:
            accepted_entries = snapshot_exact_directory(
                accepted_fd,
                regular_names=set(),
                directory_names={row.request_id for row in accepted},
            )
            for attempt in parsed:
                if attempt.request_id in attempts:
                    raise ValueError("dataset batch request outcome is duplicated")
                bundle = None
                if attempt.outcome == "accepted":
                    with bound_child_directory(
                        accepted_fd,
                        attempt.request_id,
                    ) as case_fd:
                        bundle = _verified_bundle_fd(case_fd)
                attempts[attempt.request_id] = _VerifiedAttempt(
                    attempt=attempt,
                    bundle=bundle,
                    source_root=None,
                )
            revalidate_entries(accepted_fd, accepted_entries)
        revalidate_entries(descriptor, {"outcomes.jsonl": item})
    return attempts


def _parse_records(payload: bytes) -> tuple[DatasetRecord, ...]:
    if payload and not payload.endswith(b"\n"):
        raise ValueError("dataset records require canonical LF")
    records = tuple(
        DatasetRecord.model_validate_json(line, strict=True)
        for line in payload.splitlines()
    )
    if len({item.request_id for item in records}) != len(records):
        raise ValueError("dataset record request IDs are duplicated")
    canonical = b"".join(_canonical_model_bytes(item) + b"\n" for item in records)
    if canonical != payload:
        raise ValueError("dataset records are not canonical")
    return records


def _parse_checksum_ledger(payload: bytes) -> dict[str, str]:
    if payload and not payload.endswith(b"\n"):
        raise ValueError("dataset checksum ledger requires canonical LF")
    ledger: dict[str, str] = {}
    for line in payload.splitlines():
        try:
            digest, encoded_name = line.split(b"  ", 1)
            name = encoded_name.decode("ascii")
            digest_text = digest.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("dataset checksum ledger row is malformed") from error
        _safe_relative_path(name)
        if (
            len(digest_text) != 64
            or any(item not in "0123456789abcdef" for item in digest_text)
            or name in ledger
        ):
            raise ValueError("dataset checksum ledger row is invalid")
        ledger[name] = digest_text
    if tuple(ledger) != tuple(sorted(ledger)):
        raise ValueError("dataset checksum ledger is not canonical")
    return ledger


def _verify_public_index_fd(descriptor: int, *, with_state: bool):
    directories = {"assets"} | ({".spatialcf"} if with_state else set())
    entries = snapshot_exact_directory(
        descriptor,
        regular_names=_PUBLIC_FILES,
        directory_names=directories,
    )
    metadata = {
        name: read_regular_at(
            descriptor,
            name,
            _MAX_METADATA_BYTES,
            expected_stat=entries[name],
        )
        for name in _PUBLIC_FILES
    }
    manifest = DatasetManifest.model_validate_json(
        metadata["manifest.json"], strict=True
    )
    if metadata["manifest.json"] != _canonical_model_bytes(manifest) + b"\n":
        raise ValueError("dataset manifest is not canonical")
    report = GenerationReport.model_validate_json(metadata["report.json"], strict=True)
    if metadata["report.json"] != _canonical_model_bytes(report) + b"\n":
        raise ValueError("dataset report is not canonical")
    records = _parse_records(metadata["records.jsonl"])
    if (
        manifest.record_count != len(records)
        or manifest.records_sha256
        != hashlib.sha256(metadata["records.jsonl"]).hexdigest()
        or manifest.report_sha256 != hashlib.sha256(metadata["report.json"]).hexdigest()
        or manifest.asset_bundle_paths != tuple(item.bundle_path for item in records)
        or report.accepted_request_count != len(records)
    ):
        raise ValueError("dataset record/report manifest closure changed")
    public_payloads = {
        name: payload
        for name, payload in metadata.items()
        if name != "checksums.sha256"
    }
    record_by_path = {item.bundle_path: item for item in records}
    with bound_child_directory(descriptor, "assets") as assets_fd:
        asset_entries = snapshot_exact_directory(
            assets_fd,
            regular_names=set(),
            directory_names={PurePosixPath(item).name for item in record_by_path},
        )
        for bundle_path, record in record_by_path.items():
            digest = PurePosixPath(bundle_path).name
            with bound_child_directory(assets_fd, digest) as bundle_fd:
                bundle_entries = snapshot_exact_directory(
                    bundle_fd,
                    regular_names=_BUNDLE_FILES,
                )
                bundle_payload = read_regular_at(
                    bundle_fd,
                    "bundle.json",
                    _MAX_METADATA_BYTES,
                    expected_stat=bundle_entries["bundle.json"],
                )
                bundle = publication.AssetBundle.model_validate_json(
                    bundle_payload, strict=True
                )
                checked = publication.verify_asset_bundle_fd(
                    bundle_fd,
                    bundle.native_audit_run,
                )
                before = tuple(
                    f"{bundle_path}/{item.relative_path}"
                    for item in checked.assets
                    if item.phase is publication.AssetPhase.BEFORE
                )
                after = tuple(
                    f"{bundle_path}/{item.relative_path}"
                    for item in checked.assets
                    if item.phase is publication.AssetPhase.AFTER
                )
                if (
                    checked.asset_bundle_sha256 != record.bundle_sha256
                    or record.bundle_sha256 != digest
                    or record.before_assets != before
                    or record.after_assets != after
                ):
                    raise ValueError("dataset public asset binding changed")
                for name in sorted(_BUNDLE_FILES):
                    public_payloads[f"{bundle_path}/{name}"] = read_regular_at(
                        bundle_fd,
                        name,
                        _MAX_ASSET_BYTES,
                        expected_stat=bundle_entries[name],
                    )
                revalidate_entries(bundle_fd, bundle_entries)
        revalidate_entries(assets_fd, asset_entries)
    ledger = _parse_checksum_ledger(metadata["checksums.sha256"])
    expected_ledger = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in public_payloads.items()
    }
    if ledger != dict(sorted(expected_ledger.items())):
        raise ValueError("dataset checksum ledger mismatch")
    expected_tree = _dataset_tree_sha256(
        manifest.config_sha256,
        manifest.records_sha256,
        records,
    )
    if (
        manifest.dataset_tree_sha256 != expected_tree
        or report.dataset_tree_sha256 != expected_tree
    ):
        raise ValueError("dataset tree digest changed")
    revalidate_entries(descriptor, entries)
    return manifest, report, records


def _verify_source_campaign_fd(
    plan: planning.SourcePlan,
    descriptor: int,
    stack: ExitStack,
) -> tuple[
    execution.SourceExecutionSummary,
    dict[str, _VerifiedAttempt],
    _RetainedCampaignState,
]:
    entries = snapshot_exact_directory(
        descriptor,
        regular_names=set(),
        directory_names={batch.batch_id for batch in plan.batches},
    )
    retained_batches: list[_RetainedBatchState] = []
    batch_descriptors: dict[str, int] = {}
    summaries: dict[str, execution.BatchSummary] = {}
    for batch in plan.batches:
        batch_fd = stack.enter_context(
            bound_child_directory(descriptor, batch.batch_id)
        )
        retained = execution.prepare_source_batch_verification(
            plan,
            batch,
            batch_fd,
        )
        summary = retained.summary
        batch_descriptors[batch.batch_id] = batch_fd
        summaries[batch.batch_id] = summary
        retained_batches.append(
            _RetainedBatchState(
                batch_id=batch.batch_id,
                descriptor=batch_fd,
                verification=retained,
                summary=summary,
            )
        )
    attempts = _descriptor_attempts(plan, batch_descriptors)
    summary = execution.summarize_verified_source_campaign(plan, summaries)
    return (
        summary,
        attempts,
        _RetainedCampaignState(
            entries=entries,
            batches=tuple(retained_batches),
        ),
    )


def _revalidate_source_campaign_fd(
    descriptor: int,
    plan: planning.SourcePlan,
    retained: _RetainedCampaignState,
) -> None:
    for state in retained.batches:
        summary = execution.revalidate_source_batch_verification(
            state.descriptor,
            state.verification,
        )
        if summary != state.summary:
            raise ValueError("dataset retained batch summary changed")
    revalidate_entries(descriptor, retained.entries)


def _verify_dataset_fd(
    descriptor: int,
) -> tuple[GenerationReport, tuple[DatasetRecord, ...]]:
    root_entries = snapshot_exact_directory(
        descriptor,
        regular_names=_PUBLIC_FILES,
        directory_names=_PUBLIC_DIRECTORIES,
    )
    with ExitStack() as stack:
        state_fd = stack.enter_context(bound_child_directory(descriptor, ".spatialcf"))
        state_entries = snapshot_exact_directory(
            state_fd,
            regular_names=set(),
            directory_names=set(_STATE_DIRECTORIES),
        )
        stage_descriptors = {
            name: stack.enter_context(bound_child_directory(state_fd, name))
            for name in _STATE_DIRECTORIES
        }
        for name, stage_fd in stage_descriptors.items():
            expected = state_entries[name]
            if directory_identity_fd(stage_fd) != (expected.st_dev, expected.st_ino):
                raise RuntimeError("dataset retained stage identity changed")

        manifest, report, records = _verify_public_index_fd(
            descriptor,
            with_state=True,
        )

        capture_fd = stage_descriptors["capture-plan"]
        capture_verification = capture.prepare_capture_plan_verification(capture_fd)
        capture_plan = capture_verification.plan
        config = _config_from_capture_plan(capture_plan)
        if _capture_plan(config) != capture_plan:
            raise ValueError("dataset capture plan no longer matches its config")

        roster_fd = stage_descriptors["roster"]
        roster_verification = capture.prepare_roster_verification(roster_fd)
        compilation = roster_verification.compilation

        source_plan_fd = stage_descriptors["source-plan"]
        source_plan_verification = planning.prepare_source_plan_verification(
            source_plan_fd
        )
        source_plan = source_plan_verification.plan
        expected_policy = planning.build_default_source_policy(
            compilation,
            **({"candidate_strategy": source_plan.source_policy.endpoint_candidate_strategy}
               if source_plan.source_policy.policy_version == "competition-native-source-policy:2.9.14" else {}),
        )
        if (
            source_plan.source_policy != expected_policy
            or source_plan.roster_manifest != compilation.request_manifest
        ):
            raise ValueError("dataset source plan binding changed")

        batches_fd = stage_descriptors["batches"]
        execution_summary, attempts, retained_campaign = _verify_source_campaign_fd(
            source_plan,
            batches_fd,
            stack,
        )
        (
            expected_records,
            expected_records_payload,
            expected_report,
            expected_report_payload,
            expected_manifest,
            _bundles,
        ) = _derive_dataset(
            config,
            compilation,
            source_plan,
            execution_summary,
            attempts,
        )
        if (
            records != expected_records
            or manifest != expected_manifest
            or report != expected_report
            or hashlib.sha256(expected_records_payload).hexdigest()
            != manifest.records_sha256
            or hashlib.sha256(expected_report_payload).hexdigest()
            != manifest.report_sha256
        ):
            raise ValueError("dataset public index differs from verified stages")

        final_manifest, final_report, final_records = _verify_public_index_fd(
            descriptor,
            with_state=True,
        )
        if (
            final_manifest != manifest
            or final_report != report
            or final_records != records
        ):
            raise ValueError("dataset public index changed during verification")

        _revalidate_source_campaign_fd(
            batches_fd,
            source_plan,
            retained_campaign,
        )
        if (
            capture.revalidate_capture_plan_verification(
                capture_fd,
                capture_verification,
            )
            != capture_plan
            or capture.revalidate_roster_verification(
                roster_fd,
                roster_verification,
            )
            != compilation
            or planning.revalidate_source_plan_verification(
                source_plan_fd,
                source_plan_verification,
            )
            != source_plan
        ):
            raise ValueError("dataset retained stage semantics changed")
        for name, stage_fd in stage_descriptors.items():
            expected = state_entries[name]
            if directory_identity_fd(stage_fd) != (expected.st_dev, expected.st_ino):
                raise RuntimeError("dataset retained stage identity changed")
        revalidate_entries(state_fd, state_entries)
        revalidate_entries(descriptor, root_entries)
        return expected_report, expected_records


# Preserve supported public names and pickle lookup.
_parse_attempts.__module__ = "spatialcf.generation.workflows.dataset"
_verified_bundle_at_path.__module__ = "spatialcf.generation.workflows.dataset"
_verified_bundle_fd.__module__ = "spatialcf.generation.workflows.dataset"
_path_attempts.__module__ = "spatialcf.generation.workflows.dataset"
_descriptor_attempts.__module__ = "spatialcf.generation.workflows.dataset"
_parse_records.__module__ = "spatialcf.generation.workflows.dataset"
_parse_checksum_ledger.__module__ = "spatialcf.generation.workflows.dataset"
_verify_public_index_fd.__module__ = "spatialcf.generation.workflows.dataset"
_verify_source_campaign_fd.__module__ = "spatialcf.generation.workflows.dataset"
_revalidate_source_campaign_fd.__module__ = "spatialcf.generation.workflows.dataset"
_verify_dataset_fd.__module__ = "spatialcf.generation.workflows.dataset"
