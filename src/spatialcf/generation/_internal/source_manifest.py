"""Current frozen source-prefix manifest contract and loader."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spatialcf.generation._internal.canonical_json import canonical_json_bytes
from spatialcf.solver.certified_models import CertifiedSolverConfig

_PORTABLE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SolverConfig(_FrozenModel):
    optimality_tolerance: Annotated[float, Field(strict=True, allow_inf_nan=False)] = (
        1e-6
    )
    numeric_tolerance: Annotated[float, Field(strict=True, allow_inf_nan=False)] = 1e-9
    target_interior_margin: Annotated[
        float, Field(strict=True, allow_inf_nan=False)
    ] = 5e-7
    initial_disk_segments: int = Field(default=128, strict=True)
    max_disk_segments: int = Field(default=8192, strict=True)
    timeout_seconds: None = None

    @model_validator(mode="after")
    def require_frozen_defaults(self) -> Self:
        expected = CertifiedSolverConfig()
        if self.model_dump() != {
            "optimality_tolerance": expected.optimality_tolerance,
            "numeric_tolerance": expected.numeric_tolerance,
            "target_interior_margin": expected.target_interior_margin,
            "initial_disk_segments": expected.initial_disk_segments,
            "max_disk_segments": expected.max_disk_segments,
            "timeout_seconds": expected.timeout_seconds,
        }:
            raise ValueError("final pilot requires the frozen default solver config")
        return self

    def to_solver_config(self) -> CertifiedSolverConfig:
        return CertifiedSolverConfig(**self.model_dump())


class LegacySource(_FrozenModel):
    kind: Literal["legacy-ai2thor"]
    scene_name: str = Field(min_length=1)


class ProceduralSource(_FrozenModel):
    kind: Literal["procedural"]
    dataset_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["train", "val", "test"]
    index: int = Field(ge=0, strict=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_alias: str = Field(min_length=1)
    loader_id: str = Field(min_length=1)
    loader_version: str = Field(min_length=1)


Source = Annotated[LegacySource | ProceduralSource, Field(discriminator="kind")]


def _source_scene_id(source: Source) -> str:
    if isinstance(source, LegacySource):
        return source.scene_name
    return source.scene_alias


def _source_locator(source: Source) -> tuple[object, ...]:
    if isinstance(source, LegacySource):
        return (source.kind, source.scene_name)
    return (
        source.kind,
        source.dataset_id,
        source.revision,
        source.split,
        source.index,
    )


class SourcePlanEntry(_FrozenModel):
    """One exact source in the prefix that must receive a planning outcome."""

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    scene_id: str = Field(min_length=1)
    source: Source

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        if _PORTABLE_COMPONENT.fullmatch(self.source_id) is None:
            raise ValueError("source_id is not a portable component")
        if _source_scene_id(self.source) != self.scene_id:
            raise ValueError("source scene_id does not match its source identity")
        return self


class SourcePlanManifest(_FrozenModel):
    """Canonical source prefix and frozen request-enumeration policy."""

    schema_version: Literal["certified-ai2thor-source-plan-manifest-v1"]
    plan_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    batch_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    seed: int = Field(strict=True)
    camera_policy: Literal["all-observed-source-cameras-v1"]
    use_navigation_feasibility: bool = Field(strict=True)
    solver_config: SolverConfig
    sources: tuple[SourcePlanEntry, ...]

    @model_validator(mode="after")
    def validate_source_prefix(self) -> Self:
        for label, value in (
            ("plan_version", self.plan_version),
            ("batch_version", self.batch_version),
        ):
            if _PORTABLE_COMPONENT.fullmatch(value) is None:
                raise ValueError(f"{label} is not a portable component")
        source_ids = tuple(item.source_id for item in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_id values must be unique")
        if source_ids != tuple(sorted(source_ids)):
            raise ValueError("sources must use canonical source-id order")
        scene_ids = tuple(item.scene_id for item in self.sources)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError("source scene_id values must be unique")
        locators = tuple(_source_locator(item.source) for item in self.sources)
        if len(set(locators)) != len(locators):
            raise ValueError("source locators must be unique")
        return self


def load_source_plan_manifest(path: Path) -> SourcePlanManifest:
    """Strictly parse one canonical source-plan manifest."""

    raw = Path(path).read_bytes()
    try:
        json.loads(raw)
        manifest = SourcePlanManifest.model_validate_json(raw, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source-plan manifest must be valid UTF-8 JSON") from error
    if raw != canonical_json_bytes(manifest.model_dump(mode="json"), pretty=True):
        raise ValueError("source-plan manifest must use canonical pretty JSON")
    return manifest


# Stored Pydantic type names from the frozen wire family remain available privately.
CertifiedAI2ThorSourcePlanManifest = SourcePlanManifest
CertifiedAI2ThorSourcePlanEntry = SourcePlanEntry
LegacyAI2ThorSource = LegacySource
PilotSolverConfig = SolverConfig
ProcTHORSource = ProceduralSource

__all__ = (
    "LegacySource",
    "ProceduralSource",
    "SolverConfig",
    "SourcePlanEntry",
    "SourcePlanManifest",
    "load_source_plan_manifest",
)
