"""Immutable dataset records and intrinsic validation, below workflow entry points."""

from __future__ import annotations

from pathlib import (
    PurePosixPath,
)

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.request import (
    Relation,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json", warnings="error"))


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


DatasetRecord.model_rebuild()
GenerationReport.model_rebuild()
DatasetManifest.model_rebuild()


# Preserve supported public names and pickle lookup.
_canonical_model_bytes.__module__ = "spatialcf.generation.dataset"
_safe_relative_path.__module__ = "spatialcf.generation.dataset"
DatasetRecord.__module__ = "spatialcf.generation.dataset"
GenerationReport.__module__ = "spatialcf.generation.dataset"
DatasetManifest.__module__ = "spatialcf.generation.dataset"
