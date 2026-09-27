"""Immutable internal records shared by dataset workflow stages."""

from __future__ import annotations

import os

from collections.abc import (
    Mapping,
)

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

import spatialcf.generation.execution as execution
import spatialcf.generation.publication as publication


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


@dataclass(frozen=True, slots=True)
class _VerifiedAttempt:
    attempt: execution.BatchAttempt
    bundle: publication.AssetBundle | None
    source_root: Path | None


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


# Preserve supported public names and pickle lookup.
SourcePlanIncompleteError.__module__ = "spatialcf.generation.workflows.dataset"
_VerifiedAttempt.__module__ = "spatialcf.generation.workflows.dataset"
_RetainedBatchState.__module__ = "spatialcf.generation.workflows.dataset"
_RetainedCampaignState.__module__ = "spatialcf.generation.workflows.dataset"
