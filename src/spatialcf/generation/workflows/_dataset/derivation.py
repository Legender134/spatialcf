"""Derive canonical dataset records from verified batch artifacts."""

from __future__ import annotations

import hashlib

from collections import (
    Counter,
)

from collections.abc import (
    Mapping,
)

from pathlib import (
    Path,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

import spatialcf.generation.execution as execution
import spatialcf.generation.planning as planning
import spatialcf.generation.publication as publication

from spatialcf.generation.config import (
    GenerationConfig,
)

from spatialcf.generation.dataset_models import (
    DatasetManifest,
    DatasetRecord,
    GenerationReport,
    _canonical_model_bytes,
)

from spatialcf.generation.workflows.contracts import (
    ExecutionRejection,
    PlanningRejection,
    VerificationRejection,
    VerifiedExample,
    _FreshTransitions,
)

from spatialcf.generation.workflows._dataset.contracts import (
    _DATASET_TREE_HASH_DOMAIN,
    _VerifiedAttempt,
)

from spatialcf.generation.workflows._dataset.state import (
    _config_sha256,
)


def _terminal_key(
    prefix: str, reasons: tuple[str, ...], stage: str | None = None
) -> str:
    joined = "|".join(reasons)
    return f"{prefix}:{joined}" if stage is None else f"{prefix}:{stage}:{joined}"


def _dataset_tree_sha256(
    config_sha256: Sha256Digest,
    records_sha256: Sha256Digest,
    records: tuple[DatasetRecord, ...],
) -> Sha256Digest:
    return canonical_sha256(
        {
            "asset_bundles": tuple(
                (item.bundle_path, item.bundle_sha256) for item in records
            ),
            "config_sha256": config_sha256,
            "records_sha256": records_sha256,
        },
        domain=_DATASET_TREE_HASH_DOMAIN,
    )


def _derive_dataset(
    config: GenerationConfig,
    compilation,
    plan: planning.SourcePlan,
    execution_summary: execution.SourceExecutionSummary,
    attempts: Mapping[str, _VerifiedAttempt],
    *,
    fresh_transitions: _FreshTransitions | None = None,
):
    remaining = dict(attempts)
    request_ids = tuple(item.request_id for item in plan.request_outcomes)
    if fresh_transitions is not None:
        fresh_transitions._validate_terminal_closure(request_ids)
    records: list[DatasetRecord] = []
    bundles: dict[str, tuple[publication.AssetBundle, Path]] = {}
    seen_bundle_paths: set[str] = set()
    reasons: Counter[str] = Counter()
    for outcome in compilation.scene_inventory:
        if outcome.status == "rejected":
            reasons[_terminal_key("capture", outcome.reasons)] += 1
    planned_count = 0
    execution_rejected = 0
    for outcome in plan.request_outcomes:
        terminal = (
            fresh_transitions._terminal_for(outcome.request_id)
            if fresh_transitions is not None
            else None
        )
        if outcome.status == "rejected":
            if fresh_transitions is not None:
                request = fresh_transitions._request(outcome.request_id)
                if (
                    type(terminal) is not PlanningRejection
                    or terminal.request is not request
                    or terminal.reason != "|".join(outcome.reasons)
                ):
                    raise ValueError("dataset planning terminal binding changed")
            reasons[_terminal_key("planning", outcome.reasons)] += 1
            continue
        planned_count += 1
        try:
            verified = remaining.pop(outcome.request_id)
        except KeyError as error:
            raise ValueError(
                "dataset planned request has no terminal outcome"
            ) from error
        attempt = verified.attempt
        if attempt.outcome == "rejected":
            if fresh_transitions is not None:
                request = fresh_transitions._request(outcome.request_id)
                reason = "|".join(attempt.reasons)
                if attempt.stage == "native_verification":
                    if (
                        type(terminal) is not VerificationRejection
                        or terminal.execution.plan.request is not request
                        or terminal.reason != reason
                    ):
                        raise ValueError(
                            "dataset verification terminal binding changed"
                        )
                elif (
                    type(terminal) is not ExecutionRejection
                    or terminal.request is not request
                    or terminal.reason != reason
                    or (
                        terminal.plan is not None
                        and terminal.plan.request is not request
                    )
                ):
                    raise ValueError("dataset execution terminal binding changed")
            execution_rejected += 1
            reasons[_terminal_key("execution", attempt.reasons, attempt.stage)] += 1
            continue
        if attempt.case_path != f"accepted/{outcome.request_id}":
            raise ValueError("dataset accepted outcome path changed")
        bundle = verified.bundle
        if bundle is None:
            raise ValueError("dataset accepted outcome has no verified asset bundle")
        if (
            attempt.native_asset_bundle_sha256 != bundle.asset_bundle_sha256
            or bundle.native_audit_run.intervention.subject_id != outcome.subject_id
            or bundle.native_audit_run.intervention.reference_id != outcome.reference_id
        ):
            raise ValueError("dataset accepted asset binding changed")
        if fresh_transitions is not None:
            if type(terminal) is not VerifiedExample:
                raise ValueError("dataset accepted terminal is not verified")
            request = terminal.execution.plan.request
            intervention = request.intervention
            audit_edit = bundle.native_audit_run.solve_result.selected_witness.edit
            if (
                request.source.request.scene_id != outcome.scene_id
                or intervention.subject_id != outcome.subject_id
                or intervention.reference_id != outcome.reference_id
                or intervention.relation_before is not outcome.relation_before
                or intervention.relation_after is not outcome.relation_before.opposite
                or terminal.execution.readback.application.edit
                != terminal.execution.plan.edit
                or bundle.native_audit_run.intervention != intervention
                or audit_edit.semantic_problem_sha256
                != terminal.execution.plan.edit.semantic_problem_sha256
                or audit_edit.translation_xy_m
                != terminal.execution.plan.edit.translation_xy_m
                or audit_edit.subject_id
                != f"object:{terminal.execution.plan.edit.subject_id}"
                or fresh_transitions.require_verified(outcome.request_id)
                is not terminal
            ):
                raise ValueError("dataset accepted terminal binding changed")
        bundle_path = f"assets/{bundle.asset_bundle_sha256}"
        before_assets = tuple(
            f"{bundle_path}/{item.relative_path}"
            for item in bundle.assets
            if item.phase is publication.AssetPhase.BEFORE
        )
        after_assets = tuple(
            f"{bundle_path}/{item.relative_path}"
            for item in bundle.assets
            if item.phase is publication.AssetPhase.AFTER
        )
        record = DatasetRecord(
            request_id=outcome.request_id,
            scene_id=outcome.scene_id,
            subject_id=outcome.subject_id,
            reference_id=outcome.reference_id,
            relation_before=outcome.relation_before,
            relation_after=outcome.relation_before.opposite,
            bundle_path=bundle_path,
            bundle_sha256=bundle.asset_bundle_sha256,
            before_assets=before_assets,
            after_assets=after_assets,
        )
        if bundle_path in seen_bundle_paths:
            raise ValueError("dataset accepted bundle identity is duplicated")
        seen_bundle_paths.add(bundle_path)
        if verified.source_root is not None:
            bundles[bundle_path] = (bundle, verified.source_root)
        records.append(record)
    if remaining:
        raise ValueError("dataset batch outcomes escape the frozen request roster")
    records_tuple = tuple(records)
    records_payload = b"".join(
        _canonical_model_bytes(item) + b"\n" for item in records_tuple
    )
    records_sha256 = hashlib.sha256(records_payload).hexdigest()
    config_sha256 = _config_sha256(config)
    tree_sha256 = _dataset_tree_sha256(
        config_sha256,
        records_sha256,
        records_tuple,
    )
    report = GenerationReport(
        source_count=compilation.summary.source_count,
        source_capture_rejected_count=compilation.summary.rejected_source_count,
        frozen_request_count=len(plan.request_outcomes),
        planned_request_count=planned_count,
        planning_rejected_request_count=len(plan.request_outcomes) - planned_count,
        accepted_request_count=len(records_tuple),
        execution_rejected_request_count=execution_rejected,
        terminal_reasons=dict(sorted(reasons.items())),
        dataset_tree_sha256=tree_sha256,
    )
    if (
        execution_summary.endpoint_planned_request_count != planned_count
        or execution_summary.accepted_request_count != len(records_tuple)
        or execution_summary.native_rejected_request_count != execution_rejected
    ):
        raise ValueError("dataset execution summary counts changed")
    report_payload = _canonical_model_bytes(report) + b"\n"
    manifest = DatasetManifest(
        config_sha256=config_sha256,
        record_count=len(records_tuple),
        records_sha256=records_sha256,
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
        asset_bundle_paths=tuple(item.bundle_path for item in records_tuple),
        dataset_tree_sha256=tree_sha256,
    )
    return (
        records_tuple,
        records_payload,
        report,
        report_payload,
        manifest,
        bundles,
    )


# Preserve supported public names and pickle lookup.
_terminal_key.__module__ = "spatialcf.generation.workflows.dataset"
_dataset_tree_sha256.__module__ = "spatialcf.generation.workflows.dataset"
_derive_dataset.__module__ = "spatialcf.generation.workflows.dataset"
