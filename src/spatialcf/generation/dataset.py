"""Stable dataset-generation and fresh-verification facade."""

from __future__ import annotations

import hashlib
import os
import stat
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spatialcf.adapters.ai2thor import AI2ThorAdapter
from spatialcf.domain.enums import Relation
from spatialcf.domain.v2.base import Sha256Digest
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation import capture, execution, planning, publication
from spatialcf.generation.config import GenerationConfig, load_generation_config
from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_directory,
    open_native_output_parent,
    read_regular_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
    sync_directory_fd,
    write_regular_sync_at,
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


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return canonical_json_bytes_v2(model.model_dump(mode="json", warnings="error"))


def _safe_relative_path(value: str, *, parts: int | None = None) -> PurePosixPath:
    if type(value) is not str:
        raise TypeError("dataset relative paths must be exact strings")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(item in {"", ".", ".."} for item in path.parts)
        or "\\" in value
        or "\x00" in value
        or (parts is not None and len(path.parts) != parts)
    ):
        raise ValueError("dataset relative path is unsafe")
    return path


class DatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    record_version: Literal[1] = 1
    request_id: str
    scene_id: str
    subject_id: str
    reference_id: str
    relation_before: Relation
    relation_after: Relation
    bundle_path: str
    bundle_sha256: Sha256Digest
    before_assets: tuple[str, ...]
    after_assets: tuple[str, ...]

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        bundle = _safe_relative_path(self.bundle_path, parts=2)
        if bundle.parts != ("assets", self.bundle_sha256):
            raise ValueError("dataset bundle path is not content addressed")
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("dataset record relation does not flip to its opposite")
        expected_prefix = self.bundle_path + "/"
        assets = (*self.before_assets, *self.after_assets)
        if (
            not self.before_assets
            or not self.after_assets
            or len(set(assets)) != len(assets)
            or any(
                len(_safe_relative_path(item).parts) != 3
                or not item.startswith(expected_prefix)
                for item in assets
            )
        ):
            raise ValueError("dataset record asset paths are not closed")
        return self


class GenerationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    report_version: Literal[1] = 1
    source_count: int = Field(ge=0)
    source_capture_rejected_count: int = Field(ge=0)
    frozen_request_count: int = Field(ge=0)
    planned_request_count: int = Field(ge=0)
    planning_rejected_request_count: int = Field(ge=0)
    accepted_request_count: int = Field(ge=0)
    execution_rejected_request_count: int = Field(ge=0)
    terminal_reasons: dict[str, int]
    dataset_tree_sha256: Sha256Digest

    @field_validator("terminal_reasons")
    @classmethod
    def validate_terminal_reasons(cls, value: dict[str, int]) -> dict[str, int]:
        if type(value) is not dict or any(
            type(key) is not str or not key or type(count) is not int or count <= 0
            for key, count in value.items()
        ):
            raise ValueError("terminal reasons must be positive exact counts")
        if tuple(value) != tuple(sorted(value)):
            raise ValueError("terminal reasons are not canonical")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.source_capture_rejected_count > self.source_count:
            raise ValueError("source capture rejection count exceeds source count")
        if self.frozen_request_count != (
            self.planned_request_count + self.planning_rejected_request_count
        ):
            raise ValueError("frozen request counts do not close")
        if self.planned_request_count != (
            self.accepted_request_count + self.execution_rejected_request_count
        ):
            raise ValueError("planned request counts do not close")
        if sum(self.terminal_reasons.values()) != (
            self.source_capture_rejected_count
            + self.planning_rejected_request_count
            + self.execution_rejected_request_count
        ):
            raise ValueError("terminal reason counts do not close")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    manifest_version: Literal[1] = 1
    config_sha256: Sha256Digest
    record_count: int = Field(ge=0)
    records_sha256: Sha256Digest
    report_sha256: Sha256Digest
    asset_bundle_paths: tuple[str, ...]
    dataset_tree_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_asset_roster(self) -> Self:
        if len(self.asset_bundle_paths) != self.record_count or len(
            set(self.asset_bundle_paths)
        ) != len(self.asset_bundle_paths):
            raise ValueError("dataset manifest asset roster does not close")
        for path in self.asset_bundle_paths:
            parsed = _safe_relative_path(path, parts=2)
            if parsed.parts[0] != "assets":
                raise ValueError("dataset manifest asset path escapes assets")
        return self


def _config_sha256(config: GenerationConfig) -> Sha256Digest:
    return canonical_sha256_v2(
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
):
    if root.exists():
        summary = capture.verify_roster(root)
    else:
        summary = capture.capture_and_publish_dataset(
            plan,
            root,
            adapter_factory=adapter_factory,
        )
    compilation = capture.load_roster(root)
    if compilation.summary != summary or compilation.policy != plan.roster_policy:
        raise ValueError("dataset roster stage identity differs from config")
    return compilation


def _load_or_build_source_plan(compilation, root: Path) -> planning.SourcePlan:
    expected_policy = planning.build_default_source_policy(compilation)
    if root.exists():
        plan = planning.load_source_plan(root)
    else:
        planned = planning.plan_source_campaign(compilation, expected_policy)
        planning.publish_source_plan(planned, root)
        plan = planning.load_source_plan(root)
    if (
        plan.source_policy != expected_policy
        or plan.roster_manifest != compilation.request_manifest
    ):
        raise ValueError("dataset source-plan stage identity differs from config")
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
) -> execution.SourceExecutionSummary:
    _ensure_batches_root(root)

    def execute_current_batch(
        manifest,
        output,
        *,
        expected_parent_identity,
        request_lineage,
    ):
        return execution.execute_batch(
            manifest,
            output,
            expected_parent_identity=expected_parent_identity,
            request_lineage=request_lineage,
            adapter_factory=adapter_factory,
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
    if payload != canonical_json_bytes_v2(bundle) + b"\n":
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
    return canonical_sha256_v2(
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
):
    remaining = dict(attempts)
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
        if outcome.status == "rejected":
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
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
) -> GenerationReport:
    """Generate or resume one immutable dataset and publish its stable index."""

    checked = _checked_config(config)
    root = _absolute_output(output)
    expected_capture = _capture_plan(checked)
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
    )
    source_plan = _load_or_build_source_plan(
        roster,
        root / ".spatialcf" / "source-plan",
    )
    execution_summary = _run_batches(
        source_plan,
        root / ".spatialcf" / "batches",
        adapter_factory=adapter_factory,
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
    )
    return _publish_dataset_index(
        root,
        records_payload,
        report_payload,
        manifest,
        bundles,
    )


def verify_dataset(root: Path) -> GenerationReport:
    """Freshly verify public metadata, assets, stages, and all count closure."""

    absolute = _absolute_output(root)
    with bound_absolute_directory(absolute) as descriptor:
        report, _ = _verify_dataset_fd(descriptor)
        return report


def read_dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    """Read the exact canonical JSONL roster under a retained root descriptor."""

    absolute = _absolute_output(root)
    with bound_absolute_directory(absolute) as descriptor:
        entries = snapshot_exact_directory(
            descriptor,
            regular_names=_PUBLIC_FILES,
            directory_names=_PUBLIC_DIRECTORIES,
        )
        payload = read_regular_at(
            descriptor,
            "records.jsonl",
            _MAX_METADATA_BYTES,
            expected_stat=entries["records.jsonl"],
        )
        records = _parse_records(payload)
        revalidate_entries(descriptor, entries)
        return records


def inspect_dataset(root: Path) -> dict[str, object]:
    """Return a compact summary only after full fresh verification."""

    absolute = _absolute_output(root)
    with bound_absolute_directory(absolute) as descriptor:
        report, records = _verify_dataset_fd(descriptor)
    relation_counts = Counter(item.relation_after.value for item in records)
    return {
        "dataset_tree_sha256": report.dataset_tree_sha256,
        "record_count": len(records),
        "relation_counts": dict(sorted(relation_counts.items())),
        "report": report.model_dump(mode="json"),
    }


__all__ = (
    "DatasetManifest",
    "DatasetRecord",
    "GenerationReport",
    "generate_dataset",
    "inspect_dataset",
    "read_dataset_records",
    "verify_dataset",
)
