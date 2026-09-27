"""Stable dataset-generation and fresh-verification facade."""

from __future__ import annotations

from spatialcf.generation.dataset_models import (
    _canonical_model_bytes,
    _safe_relative_path,
    DatasetRecord,
    GenerationReport,
    DatasetManifest,
)

from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spatialcf.composition import DEFAULT_ENVIRONMENT_ADAPTER_FACTORY as AI2ThorAdapter
from spatialcf.domain.base import Sha256Digest
from spatialcf.domain.request import Relation
from spatialcf.domain.serialization import (
    canonical_json_bytes,
)
from spatialcf.generation.config import GenerationConfig












from spatialcf.generation.workflows import dataset as dataset_workflow


def generate_dataset(
    config: GenerationConfig | Path,
    output: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
) -> GenerationReport:
    return dataset_workflow.generate_dataset(
        config, output, adapter_factory=adapter_factory
    )


def verify_dataset(root: Path) -> GenerationReport:
    """Freshly verify public metadata, assets, stages, and all count closure."""

    absolute = dataset_workflow._absolute_output(root)
    with dataset_workflow.bound_absolute_directory(absolute) as descriptor:
        report, _ = dataset_workflow._verify_dataset_fd(descriptor)
        return report


def read_dataset_records(root: Path) -> tuple[DatasetRecord, ...]:
    """Read the exact canonical JSONL roster under a retained root descriptor."""

    absolute = dataset_workflow._absolute_output(root)
    with dataset_workflow.bound_absolute_directory(absolute) as descriptor:
        entries = dataset_workflow.snapshot_exact_directory(
            descriptor,
            regular_names=dataset_workflow._PUBLIC_FILES,
            directory_names=dataset_workflow._PUBLIC_DIRECTORIES,
        )
        payload = dataset_workflow.read_regular_at(
            descriptor,
            "records.jsonl",
            dataset_workflow._MAX_METADATA_BYTES,
            expected_stat=entries["records.jsonl"],
        )
        records = dataset_workflow._parse_records(payload)
        dataset_workflow.revalidate_entries(descriptor, entries)
        return records


def inspect_dataset(root: Path) -> dict[str, object]:
    """Return a compact summary only after full fresh verification."""

    absolute = dataset_workflow._absolute_output(root)
    with dataset_workflow.bound_absolute_directory(absolute) as descriptor:
        report, records = dataset_workflow._verify_dataset_fd(descriptor)
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
