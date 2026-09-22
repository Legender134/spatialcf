"""Protocol-only dataset transaction and publication workflow."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from spatialcf.adapters.base import EnvironmentAdapter as AI2ThorAdapter
from spatialcf.adapters.base import SettledReadback
from spatialcf.core.outcome_assembler import (
    AssembledCounterfactualOutcome,
    assemble_counterfactual_outcome,
    assemble_no_selection_unknown,
)
from spatialcf.core.upright_se2_backend import (
    UprightSE2CardinalBackend,
    UprightSE2ContinuousBackend,
)
from spatialcf.domain import contrast as semantic
from spatialcf.domain import upright_se2 as upright
from spatialcf.domain.base import Sha256Digest
from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.lineage import (
    DependencyInventoryEntry,
    InterpreterIdentity,
    LockMetadata,
    ModuleFileIdentity,
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
from spatialcf.domain.request import InterventionSpec
from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)
from spatialcf.generation import capture, execution, planning, publication
from spatialcf.generation.config import GenerationConfig, load_generation_config
from spatialcf.generation.dataset import (
    DatasetManifest,
    DatasetRecord,
    GenerationReport,
    _canonical_model_bytes,
    _safe_relative_path,
)
from spatialcf.generation.execution.audit import (
    EndpointAuditRejected,
    _execute_audit_with_transitions,
)
from spatialcf.generation.planning.campaign import (
    _SemanticCatalogPlan,
    _bound_semantic_catalog,
    _derive_semantic_catalog,
    _semantic_target_labels,
)
from spatialcf.generation.workflows.capture import (
    _capture_and_publish_dataset_with_transitions,
)
from spatialcf.generation.workflows.contracts import (
    CounterfactualRequest,
    ExecutedEdit,
    ExecutionRejection,
    PlanningRejection,
    VerificationRejection,
    VerifiedExample,
    _FreshTransitions,
)
from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_directory,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
    sync_directory_fd,
    write_regular_sync_at,
)
from spatialcf.verification.contrast import (
    SemanticContrastBundle,
    read_semantic_contrast_bundle,
)

# Literal import closure of the semantic entry points, including package imports.
# Namespace packages have no source file; tests and private release tools are excluded.
_SEMANTIC_RUNTIME_MODULE_CLOSURE = (
    "spatialcf/__init__.py",
    "spatialcf/adapters/__init__.py",
    "spatialcf/adapters/ai2thor/__init__.py",
    "spatialcf/adapters/ai2thor/adapter.py",
    "spatialcf/adapters/ai2thor/camera.py",
    "spatialcf/adapters/ai2thor/capture.py",
    "spatialcf/adapters/ai2thor/conversion.py",
    "spatialcf/adapters/ai2thor/execution.py",
    "spatialcf/adapters/ai2thor/models.py",
    "spatialcf/adapters/ai2thor/support.py",
    "spatialcf/adapters/ai2thor/validation.py",
    "spatialcf/adapters/base.py",
    "spatialcf/composition.py",
    "spatialcf/core/__init__.py",
    "spatialcf/core/_internal/compilation/__init__.py",
    "spatialcf/core/_internal/compilation/collision.py",
    "spatialcf/core/_internal/compilation/support.py",
    "spatialcf/core/_internal/compilation/target.py",
    "spatialcf/core/_internal/compilation/visibility.py",
    "spatialcf/core/_internal/kernels/__init__.py",
    "spatialcf/core/_internal/kernels/convex_partition.py",
    "spatialcf/core/_internal/kernels/convex_translation.py",
    "spatialcf/core/_internal/kernels/projected_visibility.py",
    "spatialcf/core/_internal/kernels/rect.py",
    "spatialcf/core/_internal/kernels/rectilinear.py",
    "spatialcf/core/_internal/kernels/so2.py",
    "spatialcf/core/_internal/kernels/strict_convex.py",
    "spatialcf/core/_internal/kernels/upright_box.py",
    "spatialcf/core/_internal/objective/__init__.py",
    "spatialcf/core/_internal/objective/base.py",
    "spatialcf/core/_internal/objective/numeric.py",
    "spatialcf/core/_internal/objective/relation.py",
    "spatialcf/core/_internal/objective/relation_damage.py",
    "spatialcf/core/_internal/objective/relation_partition.py",
    "spatialcf/core/_internal/objective/safety.py",
    "spatialcf/core/_internal/objective/safety_bounds.py",
    "spatialcf/core/_internal/objective/visibility.py",
    "spatialcf/core/_internal/objective/visibility_objective.py",
    "spatialcf/core/_internal/resources.py",
    "spatialcf/core/certificate.py",
    "spatialcf/core/feasibility.py",
    "spatialcf/core/objective.py",
    "spatialcf/core/outcome_assembler.py",
    "spatialcf/core/problem.py",
    "spatialcf/core/registry.py",
    "spatialcf/core/solver.py",
    "spatialcf/core/upright_se2_backend.py",
    "spatialcf/core/upright_se2_compiler.py",
    "spatialcf/core/upright_se2_verification.py",
    "spatialcf/core/verification.py",
    "spatialcf/domain/__init__.py",
    "spatialcf/domain/artifacts.py",
    "spatialcf/domain/base.py",
    "spatialcf/domain/certificate.py",
    "spatialcf/domain/compatibility.py",
    "spatialcf/domain/constraints.py",
    "spatialcf/domain/contrast.py",
    "spatialcf/domain/counterfactual.py",
    "spatialcf/domain/definitions.py",
    "spatialcf/domain/edit.py",
    "spatialcf/domain/geometry.py",
    "spatialcf/domain/lineage.py",
    "spatialcf/domain/objective.py",
    "spatialcf/domain/operators.py",
    "spatialcf/domain/outcomes.py",
    "spatialcf/domain/predicates.py",
    "spatialcf/domain/problem.py",
    "spatialcf/domain/profiles.py",
    "spatialcf/domain/request.py",
    "spatialcf/domain/result.py",
    "spatialcf/domain/scene.py",
    "spatialcf/domain/serialization.py",
    "spatialcf/domain/solver.py",
    "spatialcf/domain/source.py",
    "spatialcf/domain/upright_se2.py",
    "spatialcf/generation/__init__.py",
    "spatialcf/generation/_internal/__init__.py",
    "spatialcf/generation/_internal/canonical_json.py",
    "spatialcf/generation/_internal/source_manifest.py",
    "spatialcf/generation/capture/__init__.py",
    "spatialcf/generation/capture/compiler.py",
    "spatialcf/generation/capture/models.py",
    "spatialcf/generation/capture/plan.py",
    "spatialcf/generation/capture/reachability.py",
    "spatialcf/generation/capture/source.py",
    "spatialcf/generation/capture/storage.py",
    "spatialcf/generation/capture/visual_evidence.py",
    "spatialcf/generation/config.py",
    "spatialcf/generation/contrast.py",
    "spatialcf/generation/dataset.py",
    "spatialcf/generation/errors.py",
    "spatialcf/generation/execution/__init__.py",
    "spatialcf/generation/execution/audit.py",
    "spatialcf/generation/execution/batch.py",
    "spatialcf/generation/execution/campaign.py",
    "spatialcf/generation/execution/correspondence.py",
    "spatialcf/generation/planning/__init__.py",
    "spatialcf/generation/planning/campaign.py",
    "spatialcf/generation/planning/endpoint.py",
    "spatialcf/generation/planning/models.py",
    "spatialcf/generation/planning/problem.py",
    "spatialcf/generation/planning/view_guard.py",
    "spatialcf/generation/publication/__init__.py",
    "spatialcf/generation/publication/assets.py",
    "spatialcf/generation/workflows/__init__.py",
    "spatialcf/generation/workflows/capture.py",
    "spatialcf/generation/workflows/contracts.py",
    "spatialcf/generation/workflows/dataset.py",
    "spatialcf/generation/workflows/execution.py",
    "spatialcf/geometry/__init__.py",
    "spatialcf/geometry/obb.py",
    "spatialcf/geometry/regions.py",
    "spatialcf/geometry/transforms.py",
    "spatialcf/relations/__init__.py",
    "spatialcf/relations/engine.py",
    "spatialcf/verification/__init__.py",
    "spatialcf/verification/artifacts.py",
    "spatialcf/verification/contrast.py",
    "spatialcf/verification/dataset.py",
    "spatialcf/verification/filesystem.py",
    "spatialcf/verification/integrity.py",
    "spatialcf/verification/profile.py",
    "spatialcf/verification/provenance.py",
    "spatialcf/verification/split.py",
    "spatialcf/verification/verifier.py",
)

_CONFIG_HASH_DOMAIN = "spatialcf.generation-config.v1"
_DATASET_TREE_HASH_DOMAIN = "spatialcf.dataset-tree.v1"
_PUBLIC_FILES = frozenset(
    {"manifest.json", "records.jsonl", "report.json", "checksums.sha256"}
)
_PUBLIC_DIRECTORIES = frozenset({"assets", ".spatialcf"})
_STATE_DIRECTORIES = ("capture-plan", "roster", "source-plan", "batches")
_BUNDLE_FILES = frozenset(
    {
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
)
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_ASSET_BYTES = 512 * 1024 * 1024


class SourcePlanIncompleteError(RuntimeError):
    """The persisted current source plan cannot enter native execution."""


def _require_complete_source_plan(plan: planning.SourcePlan) -> None:
    if type(plan) is not planning.SourcePlan:
        raise TypeError("dataset source plan must be exact")
    request_ids = tuple(item.request_id for item in plan.roster_manifest.requests)
    outcome_ids = tuple(item.request_id for item in plan.request_outcomes)
    if outcome_ids != request_ids or len(set(outcome_ids)) != len(outcome_ids):
        raise SourcePlanIncompleteError(
            "dataset source plan is incomplete:planned=0:"
            f"rejected={len(request_ids)}:"
            "reasons=source_plan:request_outcomes_not_closed"
        )
    rejected = tuple(
        item for item in plan.request_outcomes if item.status != "planned"
    )
    if rejected:
        reasons = tuple(sorted({reason for item in rejected for reason in item.reasons}))
        raise SourcePlanIncompleteError(
            "dataset source plan is incomplete:"
            f"planned={len(plan.request_outcomes) - len(rejected)}:"
            f"rejected={len(rejected)}:reasons={','.join(reasons)}"
        )


def _config_sha256(config: GenerationConfig) -> Sha256Digest:
    return canonical_sha256(
        config.model_dump(mode="json", warnings="error"),
        domain=_CONFIG_HASH_DOMAIN,
    )


def _capture_plan(config: GenerationConfig) -> capture.CapturePlan:
    return capture.build_legacy_capture_plan(
        config.scene_names,
        assigned_split=config.split,
        campaign_id=config.campaign_id,
        seed=config.seed,
        width=config.width,
        height=config.height,
        max_requests_total=config.max_requests,
    )


def _checked_config(config: GenerationConfig | Path) -> GenerationConfig:
    if type(config) is GenerationConfig:
        return GenerationConfig.model_validate(
            config.model_dump(mode="python"), strict=True
        )
    if isinstance(config, Path):
        return load_generation_config(config)
    raise TypeError("config must be an exact GenerationConfig or Path")


def _absolute_output(output: Path) -> Path:
    if not isinstance(output, Path):
        raise TypeError("dataset output must be a Path")
    absolute = Path(os.path.abspath(output))
    if absolute == absolute.parent:
        raise ValueError("dataset output may not be filesystem root")
    return absolute


def _initialize_dataset_root(
    output: Path,
    expected_plan: capture.CapturePlan,
) -> None:
    with open_native_output_parent(output) as parent:
        parent.ensure_absent(parent.output_name)
        with parent.create_staging(label="dataset") as transaction:
            transaction.mkdir(".spatialcf")
            plan_root = output.parent / transaction.name / ".spatialcf" / "capture-plan"
            capture.publish_capture_plan(expected_plan, plan_root)
            transaction.adopt_exact_tree(
                ".spatialcf/capture-plan",
                regular_paths={"plan.json", "checksums.sha256"},
            )
            transaction.fsync()
            seal = transaction.seal()
            if capture.load_capture_plan(plan_root) != expected_plan:
                raise RuntimeError("dataset capture-plan staging verification changed")
            transaction.validate_seal(seal)
            transaction.publish()
            try:
                final = capture.load_capture_plan(
                    output / ".spatialcf" / "capture-plan"
                )
                if final != expected_plan:
                    raise RuntimeError(
                        "dataset capture-plan final verification changed"
                    )
                transaction.validate_location(RenameLocation.OUTPUT)
                transaction.validate_seal(seal)
            except BaseException:
                transaction.rollback()
                raise


def _existing_names(descriptor: int, *, maximum: int) -> dict[str, os.stat_result]:
    return scan_directory(descriptor, maximum_entries=maximum)


def _validate_generation_root(root: Path) -> None:
    allowed = set(_PUBLIC_FILES) | set(_PUBLIC_DIRECTORIES)
    with bound_absolute_directory(root) as descriptor:
        entries = _existing_names(descriptor, maximum=len(allowed))
        if ".spatialcf" not in entries or not set(entries) <= allowed:
            raise ValueError("dataset root file set is not resumable")
        for name, item in entries.items():
            if name in _PUBLIC_FILES:
                if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                    raise ValueError("dataset public metadata must be regular")
            elif not stat.S_ISDIR(item.st_mode):
                raise ValueError("dataset public child must be a real directory")
        with bound_child_directory(descriptor, ".spatialcf") as state_fd:
            state = _existing_names(state_fd, maximum=len(_STATE_DIRECTORIES))
            names = set(state)
            if "capture-plan" not in names or not names <= set(_STATE_DIRECTORIES):
                raise ValueError("dataset resumable stage set is invalid")
            if "source-plan" in names and "roster" not in names:
                raise ValueError("dataset source plan has no roster stage")
            if "batches" in names and "source-plan" not in names:
                raise ValueError("dataset batches have no source plan stage")
            if any(not stat.S_ISDIR(item.st_mode) for item in state.values()):
                raise ValueError("dataset stage root must be a real directory")
            revalidate_entries(state_fd, state)
        revalidate_entries(descriptor, entries)


def _load_or_build_roster(
    plan: capture.CapturePlan,
    root: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
    fresh_transitions: _FreshTransitions | None = None,
):
    if root.exists():
        summary = capture.verify_roster(root)
    else:
        if fresh_transitions is None:
            summary = capture.capture_and_publish_dataset(
                plan,
                root,
                adapter_factory=adapter_factory,
            )
        else:
            summary = _capture_and_publish_dataset_with_transitions(
                plan,
                root,
                adapter_factory=adapter_factory,
                dataset_loader=capture.load_prior_dataset,
                fresh_transitions=fresh_transitions,
            )
    compilation = capture.load_roster(root)
    if compilation.summary != summary or compilation.policy != plan.roster_policy:
        raise ValueError("dataset roster stage identity differs from config")
    return compilation


def _load_or_build_source_plan(
    compilation,
    root: Path,
    *,
    fresh_transitions: _FreshTransitions | None = None,
) -> planning.SourcePlan:
    expected_policy = planning.build_default_source_policy(compilation)
    if root.exists():
        plan = planning.load_source_plan(root)
    else:
        planned = _plan_source_campaign_with_transitions(
            compilation,
            expected_policy,
            fresh_transitions=fresh_transitions,
        )
        planning.publish_source_plan(planned, root)
        plan = planning.load_source_plan(root)
    if (
        plan.source_policy != expected_policy
        or plan.roster_manifest != compilation.request_manifest
    ):
        raise ValueError("dataset source-plan stage identity differs from config")
    return plan


def _plan_source_campaign_with_transitions(
    compilation,
    policy,
    *,
    fresh_transitions: _FreshTransitions | None,
) -> planning.SourcePlan:
    plan = planning.plan_source_campaign(compilation, policy)
    if fresh_transitions is None:
        return plan
    for request, outcome in zip(
        plan.roster_manifest.requests, plan.request_outcomes, strict=True
    ):
        intervention = InterventionSpec(
            subject_id=request.subject_id,
            reference_id=request.reference_id,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            camera_id=request.camera_id,
        )
        value = fresh_transitions.request(
            request.request_id, request.source_id, intervention
        )
        if type(value) is not CounterfactualRequest:
            raise TypeError("planning transition request must be exact")
        if outcome.status == "rejected":
            fresh_transitions.reject_planning(
                request.request_id, "|".join(outcome.reasons)
            )
    return plan


def _ensure_batches_root(root: Path) -> None:
    with bound_absolute_directory(root.parent) as descriptor:
        try:
            item = os.stat(root.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(root.name, mode=0o700, dir_fd=descriptor)
            sync_directory_fd(descriptor)
            item = os.stat(root.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(item.st_mode):
            raise ValueError("dataset batches stage must be a real directory")
        revalidate_entries(descriptor, {root.name: item})


def _run_batches(
    plan: planning.SourcePlan,
    root: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
    fresh_transitions: _FreshTransitions | None = None,
) -> execution.SourceExecutionSummary:
    _require_complete_source_plan(plan)
    _ensure_batches_root(root)

    class _FreshAuditRunner:
        def __call__(self, adapter, request, *, request_lineage):
            try:
                return _execute_audit_with_transitions(
                    adapter,
                    request,
                    request_lineage=request_lineage,
                    fresh_transitions=fresh_transitions,
                )
            except EndpointAuditRejected as error:
                if error.stage == "native_verification":
                    readback = error.readback
                    if type(readback) is not SettledReadback:
                        raise TypeError(
                            "native verification rejection must retain exact readback"
                        )
                    executed = fresh_transitions.executed(
                        request.request_id,
                        readback.application.edit,
                        readback,
                    )
                    if type(executed) is not ExecutedEdit:
                        raise TypeError(
                            "verification transition execution must be exact"
                        )
                    rejection = fresh_transitions.reject_verification(
                        request.request_id, "|".join(error.reasons)
                    )
                    if type(rejection) is not VerificationRejection:
                        raise TypeError(
                            "verification rejection transition must be exact"
                        )
                else:
                    rejection = fresh_transitions.reject_execution(
                        request.request_id, "|".join(error.reasons)
                    )
                    if type(rejection) is not ExecutionRejection:
                        raise TypeError("execution rejection transition must be exact")
                raise

        def require_verified(self, request_id: str) -> None:
            fresh_transitions.require_verified(request_id)

    runner = _FreshAuditRunner() if fresh_transitions is not None else None

    def execute_current_batch(
        manifest,
        output,
        *,
        expected_parent_identity,
        request_lineage,
    ):
        arguments = {
            "expected_parent_identity": expected_parent_identity,
            "request_lineage": request_lineage,
            "adapter_factory": adapter_factory,
        }
        if runner is not None:
            arguments["runner"] = runner
        return execution.execute_batch(
            manifest,
            output,
            **arguments,
        )

    return execution.run_source_campaign(
        plan,
        root,
        execute=True,
        batch_executor=execute_current_batch,
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


@dataclass(frozen=True, slots=True)
class _VerifiedAttempt:
    attempt: execution.BatchAttempt
    bundle: publication.AssetBundle | None
    source_root: Path | None


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


def _checksum_payload(payloads: Mapping[str, bytes]) -> bytes:
    return b"".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode("ascii")
        for name, payload in sorted(payloads.items())
    )


def _stage_public_index(
    transaction,
    records_payload: bytes,
    report_payload: bytes,
    manifest: DatasetManifest,
    bundles: Mapping[str, tuple[publication.AssetBundle, Path]],
) -> dict[str, bytes]:
    transaction.mkdir("assets")
    payloads: dict[str, bytes] = {}
    for bundle_path, (bundle, source_root) in bundles.items():
        transaction.mkdir(bundle_path)
        with bound_absolute_directory(source_root) as descriptor:
            entries = snapshot_exact_directory(
                descriptor,
                regular_names=_BUNDLE_FILES,
            )
            for name in sorted(_BUNDLE_FILES):
                payload = read_regular_at(
                    descriptor,
                    name,
                    _MAX_ASSET_BYTES,
                    expected_stat=entries[name],
                )
                relative = f"{bundle_path}/{name}"
                transaction.write(relative, payload)
                payloads[relative] = payload
            revalidate_entries(descriptor, entries)
        with (
            bound_child_directory(transaction.descriptor, "assets") as assets_fd,
            bound_child_directory(assets_fd, bundle.asset_bundle_sha256) as bundle_fd,
        ):
            if (
                publication.verify_asset_bundle_fd(
                    bundle_fd,
                    bundle.native_audit_run,
                )
                != bundle
            ):
                raise RuntimeError("dataset staged asset verification changed")
    metadata = {
        "manifest.json": _canonical_model_bytes(manifest) + b"\n",
        "records.jsonl": records_payload,
        "report.json": report_payload,
    }
    for name, payload in metadata.items():
        transaction.write(name, payload)
        payloads[name] = payload
    checksum = _checksum_payload(payloads)
    transaction.write("checksums.sha256", checksum)
    payloads["checksums.sha256"] = checksum
    return payloads


def _read_or_write_exact(
    descriptor: int,
    name: str,
    payload: bytes,
    created: dict[Path, tuple[int, int]],
    relative: Path,
) -> None:
    try:
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        written = write_regular_sync_at(descriptor, name, payload)
        created[relative] = (written.st_dev, written.st_ino)
        return
    observed = read_regular_at(
        descriptor,
        name,
        max(1, len(payload)),
        expected_stat=item,
    )
    if observed != payload:
        raise FileExistsError(f"dataset public entry differs: {relative.as_posix()}")


def _ensure_exact_directory(
    descriptor: int,
    name: str,
    created: dict[Path, tuple[int, int]],
    directories: dict[Path, tuple[int, int]],
    relative: Path,
) -> None:
    try:
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        os.mkdir(name, mode=0o700, dir_fd=descriptor)
        sync_directory_fd(descriptor)
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        created[relative] = (item.st_dev, item.st_ino)
    if not stat.S_ISDIR(item.st_mode):
        raise ValueError(f"dataset public directory is unsafe: {relative.as_posix()}")
    identity = (item.st_dev, item.st_ino)
    previous = directories.setdefault(relative, identity)
    if previous != identity:
        raise RuntimeError("dataset public directory identity changed")


def _copy_staged_public_index(
    root_descriptor: int,
    transaction,
    payloads: Mapping[str, bytes],
    created: dict[Path, tuple[int, int]],
    directories: dict[Path, tuple[int, int]],
) -> None:
    _ensure_exact_directory(
        root_descriptor,
        "assets",
        created,
        directories,
        Path("assets"),
    )
    with (
        bound_child_directory(root_descriptor, "assets") as final_assets_fd,
        bound_child_directory(transaction.descriptor, "assets") as staged_fd,
    ):
        staged_assets = _existing_names(
            staged_fd,
            maximum=max(1, len(payloads)),
        )
        for digest in sorted(staged_assets):
            _ensure_exact_directory(
                final_assets_fd,
                digest,
                created,
                directories,
                Path("assets") / digest,
            )
            with (
                bound_child_directory(staged_fd, digest) as source_fd,
                bound_child_directory(final_assets_fd, digest) as target_fd,
            ):
                source_entries = snapshot_exact_directory(
                    source_fd,
                    regular_names=_BUNDLE_FILES,
                )
                for name in sorted(_BUNDLE_FILES):
                    payload = read_regular_at(
                        source_fd,
                        name,
                        _MAX_ASSET_BYTES,
                        expected_stat=source_entries[name],
                    )
                    _read_or_write_exact(
                        target_fd,
                        name,
                        payload,
                        created,
                        Path("assets") / digest / name,
                    )
                revalidate_entries(source_fd, source_entries)
    for name in ("records.jsonl", "report.json", "checksums.sha256"):
        _read_or_write_exact(
            root_descriptor,
            name,
            payloads[name],
            created,
            Path(name),
        )
    _read_or_write_exact(
        root_descriptor,
        "manifest.json",
        payloads["manifest.json"],
        created,
        Path("manifest.json"),
    )
    sync_directory_fd(root_descriptor)


@contextmanager
def _bound_rollback_parent(
    root_descriptor: int,
    relative: Path,
    directories: Mapping[Path, tuple[int, int]],
) -> Iterator[int]:
    with ExitStack() as stack:
        descriptor = root_descriptor
        opened: list[tuple[int, str, tuple[int, int]]] = []
        prefix = Path()
        for component in relative.parts:
            prefix /= component
            expected = directories.get(prefix)
            if expected is None:
                raise RuntimeError("dataset rollback parent ownership is unknown")
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(before.st_mode)
                or (
                    before.st_dev,
                    before.st_ino,
                )
                != expected
            ):
                raise RuntimeError("dataset rollback parent identity changed")
            parent_descriptor = descriptor
            descriptor = stack.enter_context(
                open_directory(component, dir_fd=parent_descriptor)
            )
            if directory_identity_fd(descriptor) != expected:
                raise RuntimeError("dataset rollback parent binding changed")
            opened.append((parent_descriptor, component, expected))
        yield descriptor
        for parent_descriptor, component, expected in reversed(opened):
            current = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or (
                    current.st_dev,
                    current.st_ino,
                )
                != expected
            ):
                raise RuntimeError("dataset rollback parent binding changed")


def _rollback_created_fd(
    root_descriptor: int,
    created: Mapping[Path, tuple[int, int]],
    directories: Mapping[Path, tuple[int, int]],
) -> None:
    errors: list[BaseException] = []
    for relative, identity in sorted(
        created.items(), key=lambda item: len(item[0].parts), reverse=True
    ):
        try:
            with _bound_rollback_parent(
                root_descriptor,
                relative.parent,
                directories,
            ) as parent_fd:
                item = os.stat(
                    relative.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (item.st_dev, item.st_ino) != identity:
                    raise RuntimeError("dataset rollback entry identity changed")
                if stat.S_ISDIR(item.st_mode):
                    os.rmdir(relative.name, dir_fd=parent_fd)
                elif stat.S_ISREG(item.st_mode):
                    os.unlink(relative.name, dir_fd=parent_fd)
                else:
                    raise RuntimeError("dataset rollback entry type changed")
                sync_directory_fd(parent_fd)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
    if errors:
        raise RuntimeError("dataset public index rollback was incomplete") from errors[
            0
        ]


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


@dataclass(frozen=True, slots=True)
class _RetainedBatchState:
    batch_id: str
    descriptor: int
    verification: execution.RetainedSourceBatchVerification
    summary: execution.BatchSummary


@dataclass(frozen=True, slots=True)
class _RetainedCampaignState:
    entries: Mapping[str, os.stat_result]
    batches: tuple[_RetainedBatchState, ...]


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


def _publication_root_binding(
    parent,
    root_name: str,
    root_identity: tuple[int, int],
) -> bool | None:
    try:
        parent.validate()
    except BaseException:  # noqa: BLE001
        return None
    try:
        current = os.stat(
            root_name,
            dir_fd=parent.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except BaseException:  # noqa: BLE001
        return None
    if (
        stat.S_ISDIR(current.st_mode)
        and (
            current.st_dev,
            current.st_ino,
        )
        == root_identity
    ):
        return True
    return None


def _raise_publication_failure(
    root: Path,
    parent,
    root_descriptor: int,
    root_identity: tuple[int, int],
    created: Mapping[Path, tuple[int, int]],
    directories: Mapping[Path, tuple[int, int]],
    active_error: BaseException,
) -> None:
    rollback_error: BaseException | None = None
    try:
        _rollback_created_fd(root_descriptor, created, directories)
    except BaseException as error:  # noqa: BLE001
        rollback_error = error
        active_error.add_note(str(error))
    binding = _publication_root_binding(parent, root.name, root_identity)
    if rollback_error is None and binding is True:
        raise active_error
    if rollback_error is None and binding is False:
        raise CompetitionNativePublicationError(
            root,
            published=False,
            recovery_name=None,
            detail="dataset root moved during public index publication and was rolled back",
        ) from active_error
    detail = (
        "dataset public index rollback could not prove complete cleanup"
        if rollback_error is not None
        else "dataset root binding became foreign during public index publication"
    )
    raise CompetitionNativePublicationError(
        root,
        published=None,
        recovery_name=None,
        detail=detail,
    ) from active_error


def _publish_dataset_index(
    root: Path,
    records_payload: bytes,
    report_payload: bytes,
    manifest: DatasetManifest,
    bundles: Mapping[str, tuple[publication.AssetBundle, Path]],
) -> GenerationReport:
    stage_target = root.parent / f"{root.name}-dataset-index"
    created: dict[Path, tuple[int, int]] = {}
    directories: dict[Path, tuple[int, int]] = {}
    with (
        open_native_output_parent(stage_target) as parent,
        parent.create_staging(label="dataset-index") as transaction,
    ):
        payloads = _stage_public_index(
            transaction,
            records_payload,
            report_payload,
            manifest,
            bundles,
        )
        transaction.fsync()
        seal = transaction.seal()
        staged_manifest, staged_report, _ = _verify_public_index_fd(
            transaction.descriptor,
            with_state=False,
        )
        if staged_manifest != manifest:
            raise RuntimeError("dataset index staging verification changed")
        transaction.validate_seal(seal)
        parent.validate()
        root_entry = os.stat(
            root.name,
            dir_fd=parent.parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(root_entry.st_mode):
            raise ValueError("dataset publication root must be a real directory")
        with open_directory(
            root.name,
            dir_fd=parent.parent_descriptor,
        ) as root_descriptor:
            root_identity = directory_identity_fd(root_descriptor)
            if root_identity != (root_entry.st_dev, root_entry.st_ino):
                raise RuntimeError("dataset publication root binding changed")
            try:
                _copy_staged_public_index(
                    root_descriptor,
                    transaction,
                    payloads,
                    created,
                    directories,
                )
                final_report, _ = _verify_dataset_fd(root_descriptor)
                if final_report != staged_report:
                    raise RuntimeError("dataset index final verification changed")
                transaction.validate_seal(seal)
                if (
                    _publication_root_binding(parent, root.name, root_identity)
                    is not True
                ):
                    raise RuntimeError("dataset publication root binding changed")
                return final_report
            except BaseException as error:  # noqa: BLE001
                _raise_publication_failure(
                    root,
                    parent,
                    root_descriptor,
                    root_identity,
                    created,
                    directories,
                    error,
                )
                raise AssertionError("unreachable")


def _config_from_capture_plan(plan: capture.CapturePlan) -> GenerationConfig:
    scene_names: list[str] = []
    for locator in plan.source_locators:
        if locator.kind != "legacy-ai2thor":
            raise ValueError("public dataset contains a non-AI2-THOR legacy source")
        scene_names.append(locator.scene_name)
    return GenerationConfig(
        adapter="ai2thor",
        scene_names=tuple(scene_names),
        split=plan.assigned_split,
        campaign_id=plan.roster_policy.campaign_id,
        seed=plan.roster_policy.seed,
        width=plan.roster_policy.width,
        height=plan.roster_policy.height,
        max_requests=plan.roster_policy.max_requests_total,
    )


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
        expected_policy = planning.build_default_source_policy(compilation)
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


def generate_dataset(
    config: GenerationConfig | Path,
    output: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
) -> GenerationReport:
    """Generate or resume one immutable dataset and publish its stable index."""

    checked = _checked_config(config)
    root = _absolute_output(output)
    expected_capture = _capture_plan(checked)
    fresh_transitions = _FreshTransitions() if not root.exists() else None
    if not root.exists():
        _initialize_dataset_root(root, expected_capture)
    _validate_generation_root(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    loaded_capture = capture.load_capture_plan(root / ".spatialcf" / "capture-plan")
    if loaded_capture != expected_capture:
        raise ValueError("dataset capture plan identity differs from config")
    roster = _load_or_build_roster(
        loaded_capture,
        root / ".spatialcf" / "roster",
        adapter_factory=adapter_factory,
        fresh_transitions=fresh_transitions,
    )
    source_plan = _load_or_build_source_plan(
        roster,
        root / ".spatialcf" / "source-plan",
        fresh_transitions=fresh_transitions,
    )
    _require_complete_source_plan(source_plan)
    execution_summary = _run_batches(
        source_plan,
        root / ".spatialcf" / "batches",
        adapter_factory=adapter_factory,
        fresh_transitions=fresh_transitions,
    )
    attempts = _path_attempts(
        source_plan,
        root / ".spatialcf" / "batches",
    )
    (
        _records,
        records_payload,
        _report,
        report_payload,
        manifest,
        bundles,
    ) = _derive_dataset(
        checked,
        roster,
        source_plan,
        execution_summary,
        attempts,
        fresh_transitions=fresh_transitions,
    )
    return _publish_dataset_index(
        root,
        records_payload,
        report_payload,
        manifest,
        bundles,
    )


def _semantic_retained_bytes(path: Path) -> bytes:
    with bound_absolute_directory(path.parent) as descriptor:
        before = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        payload = read_regular_at(
            descriptor,
            path.name,
            max(1, before.st_size),
            expected_stat=before,
        )
        revalidate_entries(descriptor, {path.name: before})
        return payload


def _semantic_runtime_provenance() -> RuntimeProvenance:
    """Derive the uncached runtime identity used by generation and replay."""

    module_root = Path(__file__).parents[3]
    modules = []
    for relative in _SEMANTIC_RUNTIME_MODULE_CLOSURE:
        payload = _semantic_retained_bytes(module_root / relative)
        modules.append(
            ModuleFileIdentity(
                package_relative_path=relative,
                byte_length=len(payload),
                byte_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    dependencies: dict[str, DependencyInventoryEntry] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise ValueError("installed dependency has no distribution name")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name == "spatialcf":
            continue
        candidates = [
            item
            for item in distribution.files or ()
            if item.name in {"METADATA", "PKG-INFO"}
            and len(item.parts) == 2
            and item.parent.name.endswith((".dist-info", ".egg-info"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "installed dependency metadata identity is unavailable or ambiguous"
            )
        payload = _semantic_retained_bytes(
            Path(distribution.locate_file(candidates[0]))
        )
        value = DependencyInventoryEntry(
            distribution_name=name,
            distribution_version=distribution.version,
            metadata_byte_length=len(payload),
            metadata_byte_sha256=hashlib.sha256(payload).hexdigest(),
        )
        previous = dependencies.setdefault(name, value)
        if previous != value:
            raise ValueError("installed dependency identity is ambiguous")
    return RuntimeProvenance.seal(
        interpreter=InterpreterIdentity(
            implementation=(
                sys.implementation.name.capitalize().replace("Cpython", "CPython")
            ),
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            micro=sys.version_info.micro,
        ),
        dependencies=tuple(
            sorted(
                dependencies.values(),
                key=lambda item: canonical_json_bytes(item.distribution_name),
            )
        ),
        module_files=tuple(modules),
        lock_metadata=LockMetadata(
            availability="UNAVAILABLE",
            scope="INSTALLED_PACKAGE_METADATA",
        ),
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


def _semantic_publication_error(output: Path, transaction, error: BaseException):
    if isinstance(error, CompetitionNativePublicationError):
        raise error
    try:
        transaction.parent.validate()
        reconciliation = (
            reconcile_owned_rename_at(
                transaction.parent.parent_descriptor,
                transaction.name,
                transaction.parent.output_name,
                transaction.identity,
            )
            if transaction.closed else transaction.reconcile()
        )
        transaction.parent.validate()
    except BaseException as reconcile_error:  # noqa: BLE001
        error.add_note(str(reconcile_error))
        published = None
        recovery_name = None
    else:
        published = (
            True
            if reconciliation.location is RenameLocation.OUTPUT
            else False
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
        recovery_name = (
            transaction.parent.output_name
            if reconciliation.location is RenameLocation.OUTPUT
            else transaction.name
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
    raise CompetitionNativePublicationError(
        output,
        published=published,
        recovery_name=recovery_name,
        detail="semantic dataset publication failed after a publication attempt",
    ) from error


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


__all__ = (
    "generate_dataset",
    "generate_semantic_contrast_dataset",
    "verify_semantic_contrast_dataset",
)
