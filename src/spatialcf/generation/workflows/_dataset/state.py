"""Validate generation configuration and bind persistent workflow state."""

from __future__ import annotations

import os

import stat

from pathlib import (
    Path,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

import spatialcf.generation.capture as capture
import spatialcf.generation.planning as planning

from spatialcf.generation.config import (
    GenerationConfig,
    load_generation_config,
)

from spatialcf.verification.filesystem import (
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    open_native_output_parent,
    revalidate_entries,
    scan_directory,
    sync_directory_fd,
)

from spatialcf.generation.workflows._dataset.contracts import (
    SourcePlanIncompleteError,
    _CONFIG_HASH_DOMAIN,
    _PUBLIC_DIRECTORIES,
    _PUBLIC_FILES,
    _STATE_DIRECTORIES,
)


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


# Preserve supported public names and pickle lookup.
_require_complete_source_plan.__module__ = "spatialcf.generation.workflows.dataset"
_config_sha256.__module__ = "spatialcf.generation.workflows.dataset"
_capture_plan.__module__ = "spatialcf.generation.workflows.dataset"
_checked_config.__module__ = "spatialcf.generation.workflows.dataset"
_absolute_output.__module__ = "spatialcf.generation.workflows.dataset"
_initialize_dataset_root.__module__ = "spatialcf.generation.workflows.dataset"
_existing_names.__module__ = "spatialcf.generation.workflows.dataset"
_validate_generation_root.__module__ = "spatialcf.generation.workflows.dataset"
_ensure_batches_root.__module__ = "spatialcf.generation.workflows.dataset"
_config_from_capture_plan.__module__ = "spatialcf.generation.workflows.dataset"
