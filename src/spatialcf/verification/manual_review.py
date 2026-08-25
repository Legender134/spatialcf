"""Deterministic manual-review queues and strict annotation gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from spatialcf.domain.request import QualityTier, Relation
from spatialcf.verification.dataset import _read_dataset

_QUEUE_SCHEMA_VERSION = 3
_IMMUTABLE_FIELDS = (
    "pair_id",
    "request_id",
    "relation_after",
    "quality",
    "generator",
    "rgb_before_path",
    "rgb_after_path",
    "topdown_path",
    "expected_before",
    "expected_after",
)


class ManualReviewRow(BaseModel):
    """One sampled pair plus one unambiguous reviewer's annotations."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pair_id: str
    request_id: str
    relation_after: Relation
    quality: QualityTier
    generator: str
    rgb_before_path: str
    rgb_after_path: str
    topdown_path: str
    expected_before: Relation
    expected_after: Relation
    observed_before: Relation | None = None
    observed_after: Relation | None = None
    collision_free: bool | None = None
    single_object_edit: bool | None = None
    approved: bool | None = None
    reviewer_id: str | None = None
    reviewer_notes: str = ""

    @field_validator(
        "pair_id",
        "request_id",
        "generator",
        "rgb_before_path",
        "rgb_after_path",
        "topdown_path",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("required review text must be non-empty")
        return value

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value or not value.strip()):
            raise ValueError("reviewer_id must be non-empty when provided")
        return value

    @field_validator("reviewer_notes")
    @classmethod
    def notes_are_text(cls, value: str) -> str:
        return value

    @field_validator("rgb_before_path", "rgb_after_path", "topdown_path")
    @classmethod
    def relative_posix_path(cls, value: str) -> str:
        if (
            "\\" in value
            or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ValueError("review artifact paths must be relative POSIX paths")
        return value


class _ReviewIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    pair_id: str
    request_id: str
    relation_after: Relation
    quality: QualityTier
    generator: str
    rgb_before_path: str
    rgb_after_path: str
    topdown_path: str
    expected_before: Relation
    expected_after: Relation


class _ReviewQueueManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    count: int
    dataset_checksums_sha256: str
    rows: tuple[_ReviewIdentity, ...]
    schema_version: int

    @field_validator("count")
    @classmethod
    def positive_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("count must be positive")
        return value

    @field_validator("dataset_checksums_sha256")
    @classmethod
    def checksum_digest(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("dataset checksum digest must be lowercase SHA-256")
        return value

    @field_validator("schema_version")
    @classmethod
    def supported_schema(cls, value: int) -> int:
        if value != _QUEUE_SCHEMA_VERSION:
            raise ValueError("unsupported review queue schema")
        return value

    @model_validator(mode="after")
    def count_matches_rows(self) -> _ReviewQueueManifest:
        if len(self.rows) != self.count:
            raise ValueError("count does not match review identity rows")
        pair_ids = [row.pair_id for row in self.rows]
        request_ids = [row.request_id for row in self.rows]
        if len(set(pair_ids)) != len(pair_ids) or len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError("review identity rows must be unique")
        return self


def _manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.queue.json")


def _canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _row_json(row: ManualReviewRow) -> bytes:
    return _canonical_json(row.model_dump(mode="json"))


def _identity(row: ManualReviewRow) -> dict[str, Any]:
    payload = row.model_dump(mode="json")
    return {field: payload[field] for field in _IMMUTABLE_FIELDS}


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return float(value)


def sample_review(
    root: Path,
    output: Path,
    count: int = 50,
    *,
    seed: int = 20260723,
) -> None:
    """Write a seed-stable, relation/generator/quality-stratified PURE test queue."""
    count = _validate_positive_integer("count", count)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    output = Path(output)
    manifest_path = _manifest_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)

    dataset = _read_dataset(root)
    eligible = [
        record
        for record in dataset.pairs
        if record.split == "test" and record.quality is QualityTier.PURE
    ]
    if len(eligible) < count:
        raise ValueError(
            f"eligible PURE test pairs={len(eligible)} below count={count}"
        )

    buckets: dict[tuple[str, str, str], list[Any]] = {}
    for record in eligible:
        key = (
            record.relation_after.value,
            record.quality.value,
            record.generator,
        )
        buckets.setdefault(key, []).append(record)
    for key, records in buckets.items():
        records.sort(
            key=lambda record: (
                hashlib.sha256(
                    f"{seed}:{record.request_id}:{record.pair_id}".encode()
                ).hexdigest(),
                record.pair_id,
            )
        )

    selected = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("review sampler exhausted eligible rows unexpectedly")

    rows = [
        ManualReviewRow(
            pair_id=record.pair_id,
            request_id=record.request_id,
            relation_after=record.relation_after,
            quality=record.quality,
            generator=record.generator,
            rgb_before_path=record.rgb_before_path,
            rgb_after_path=record.rgb_after_path,
            topdown_path=record.topdown_path,
            expected_before=record.answer_before,
            expected_after=record.answer_after,
        )
        for record in selected
    ]
    queue_payload = b"".join(_row_json(row) for row in rows)
    manifest = {
        "count": len(rows),
        "dataset_checksums_sha256": hashlib.sha256(
            (dataset.root / "checksums.sha256").read_bytes()
        ).hexdigest(),
        "rows": [_identity(row) for row in rows],
        "schema_version": _QUEUE_SCHEMA_VERSION,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output, queue_payload)
    try:
        _write_exclusive(
            manifest_path,
            _canonical_json(manifest, pretty=True),
        )
    except BaseException:
        output.unlink()
        raise


def _read_manifest(path: Path) -> _ReviewQueueManifest:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"review queue manifest is missing or unsafe: {path}")
    payload = path.read_bytes()
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError("review queue manifest must use canonical LF JSON")
    try:
        manifest = _ReviewQueueManifest.model_validate_json(payload)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as error:
        raise ValueError("review queue manifest schema is invalid") from error
    if payload != _canonical_json(manifest.model_dump(mode="json"), pretty=True):
        raise ValueError("review queue manifest schema is invalid")
    return manifest


def _read_rows(path: Path) -> list[ManualReviewRow]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("review queue must be an ordinary file")
    payload = path.read_bytes()
    if b"\r" in payload or (payload and not payload.endswith(b"\n")):
        raise ValueError("review queue must use canonical LF JSONL")
    rows: list[ManualReviewRow] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"invalid review row at line {line_number}: blank row")
        try:
            row = ManualReviewRow.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid review row at line {line_number}: {error}"
            ) from error
        if line != _row_json(row).rstrip(b"\n"):
            raise ValueError(
                f"invalid review row at line {line_number}: row is not canonical JSON"
            )
        rows.append(row)
    return rows


def validate_review(
    path: Path,
    minimum_approved: int = 50,
    *,
    dataset_root: Path | None = None,
    minimum_relation_agreement: float = 1.0,
    minimum_collision_free_rate: float = 1.0,
    minimum_single_object_edit_rate: float = 1.0,
    minimum_approval_rate: float = 0.0,
) -> dict[str, Any]:
    """Validate exact queue membership, annotations, agreement, and gates."""
    minimum_approved = _validate_positive_integer(
        "minimum_approved",
        minimum_approved,
    )
    minimum_relation_agreement = _validate_rate(
        "minimum_relation_agreement",
        minimum_relation_agreement,
    )
    minimum_collision_free_rate = _validate_rate(
        "minimum_collision_free_rate",
        minimum_collision_free_rate,
    )
    minimum_single_object_edit_rate = _validate_rate(
        "minimum_single_object_edit_rate",
        minimum_single_object_edit_rate,
    )
    minimum_approval_rate = _validate_rate(
        "minimum_approval_rate",
        minimum_approval_rate,
    )
    path = Path(path)
    if dataset_root is None:
        raise ValueError(
            "dataset_root is required explicitly for portable review validation"
        )
    manifest = _read_manifest(_manifest_path(path))
    rows = _read_rows(path)

    try:
        source_dataset = _read_dataset(Path(dataset_root))
    except (OSError, ValueError) as error:
        raise ValueError(f"source dataset validation failed: {error}") from error
    source_checksum = hashlib.sha256(
        (source_dataset.root / "checksums.sha256").read_bytes()
    ).hexdigest()
    if source_checksum != manifest.dataset_checksums_sha256:
        raise ValueError(
            "source dataset checksum differs from the sampled review queue"
        )

    expected_rows = manifest.rows
    expected_ids = [row.pair_id for row in expected_rows]
    actual_ids = [row.pair_id for row in rows]
    if (
        len(rows) != manifest.count
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected_ids)
    ):
        raise ValueError(
            "review queue membership differs from sampled queue; "
            f"expected={len(expected_ids)} actual={len(actual_ids)}"
        )
    expected_by_id = {row.pair_id: row.model_dump(mode="json") for row in expected_rows}
    source_by_id = {record.pair_id: record for record in source_dataset.pairs}
    for identity in expected_rows:
        source = source_by_id.get(identity.pair_id)
        if source is None or source.request_id != identity.request_id:
            raise ValueError(
                "review queue membership is absent from the source dataset: "
                f"{identity.pair_id}"
            )
        source_identity = {
            "pair_id": source.pair_id,
            "request_id": source.request_id,
            "relation_after": source.relation_after.value,
            "quality": source.quality.value,
            "generator": source.generator,
            "rgb_before_path": source.rgb_before_path,
            "rgb_after_path": source.rgb_after_path,
            "topdown_path": source.topdown_path,
            "expected_before": source.answer_before.value,
            "expected_after": source.answer_after.value,
        }
        if source_identity != identity.model_dump(mode="json"):
            raise ValueError(
                f"review queue identity differs from source dataset for "
                f"{identity.pair_id}"
            )
    for row in rows:
        if _identity(row) != expected_by_id[row.pair_id]:
            raise ValueError(f"review queue row metadata changed for {row.pair_id}")

    incomplete = [
        row.pair_id
        for row in rows
        if (
            row.observed_before is None
            or row.observed_after is None
            or row.collision_free is None
            or row.single_object_edit is None
            or row.approved is None
            or row.reviewer_id is None
        )
    ]
    if incomplete:
        raise ValueError(f"incomplete review rows: {incomplete[:5]}")

    reviewed = len(rows)
    approved = sum(
        row.approved is True
        and row.observed_before is row.expected_before
        and row.observed_after is row.expected_after
        and row.collision_free is True
        and row.single_object_edit is True
        for row in rows
    )
    relation_agreement = (
        sum(
            row.observed_before is row.expected_before
            and row.observed_after is row.expected_after
            for row in rows
        )
        / reviewed
    )
    collision_free_rate = sum(row.collision_free is True for row in rows) / reviewed
    single_object_edit_rate = (
        sum(row.single_object_edit is True for row in rows) / reviewed
    )
    approval_rate = approved / reviewed
    if approved < minimum_approved:
        raise ValueError(f"approved={approved} below minimum={minimum_approved}")
    for name, actual, threshold in (
        (
            "relation_agreement",
            relation_agreement,
            minimum_relation_agreement,
        ),
        (
            "collision_free_rate",
            collision_free_rate,
            minimum_collision_free_rate,
        ),
        (
            "single_object_edit_rate",
            single_object_edit_rate,
            minimum_single_object_edit_rate,
        ),
        ("approval_rate", approval_rate, minimum_approval_rate),
    ):
        if actual < threshold:
            raise ValueError(f"{name}={actual:.6f} below minimum={threshold:.6f}")
    return {
        "approved": approved,
        "approval_rate": approval_rate,
        "collision_free_rate": collision_free_rate,
        "relation_agreement": relation_agreement,
        "reviewed": reviewed,
        "reviewers": sorted({row.reviewer_id for row in rows}),
        "single_object_edit_rate": single_object_edit_rate,
    }
