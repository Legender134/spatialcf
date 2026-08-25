"""Immutable source identities and the current source-plan manifest."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.base import CanonicalModel

_PORTABLE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class SolverConfig(CanonicalModel):
    """Frozen solver settings carried by the current source-plan wire."""

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
        if self.model_dump(mode="python") != {
            "optimality_tolerance": 1e-6,
            "numeric_tolerance": 1e-9,
            "target_interior_margin": 5e-7,
            "initial_disk_segments": 128,
            "max_disk_segments": 8192,
            "timeout_seconds": None,
        }:
            raise ValueError("final pilot requires the frozen default solver config")
        return self


class LegacyAI2ThorSource(CanonicalModel):
    """One current AI2-THOR built-in scene source."""

    kind: Literal["legacy-ai2thor"]
    scene_name: str = Field(min_length=1)


class ProceduralSource(CanonicalModel):
    """One content-bound current ProcTHOR source."""

    kind: Literal["procedural"]
    dataset_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["train", "val", "test"]
    index: int = Field(ge=0, strict=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_alias: str = Field(min_length=1)
    loader_id: str = Field(min_length=1)
    loader_version: str = Field(min_length=1)


SourceLocator = Annotated[
    LegacyAI2ThorSource | ProceduralSource,
    Field(discriminator="kind"),
]


def _source_scene_id(source: SourceLocator) -> str:
    if isinstance(source, LegacyAI2ThorSource):
        return source.scene_name
    return source.scene_alias


def _source_identity(source: SourceLocator) -> tuple[object, ...]:
    if isinstance(source, LegacyAI2ThorSource):
        return (source.kind, source.scene_name)
    return (
        source.kind,
        source.dataset_id,
        source.revision,
        source.split,
        source.index,
    )


class SourcePlanEntry(CanonicalModel):
    """One exact source in the prefix that must receive a terminal outcome."""

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    scene_id: str = Field(min_length=1)
    source: SourceLocator

    @model_validator(mode="after")
    def validate_source_identity(self) -> Self:
        if _PORTABLE_COMPONENT.fullmatch(self.source_id) is None:
            raise ValueError("source_id is not a portable component")
        if _source_scene_id(self.source) != self.scene_id:
            raise ValueError("source scene_id does not match its source identity")
        return self


class SourcePlanManifest(CanonicalModel):
    """Canonical current source prefix and request-enumeration policy."""

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
        locators = tuple(_source_identity(item.source) for item in self.sources)
        if len(set(locators)) != len(locators):
            raise ValueError("source locators must be unique")
        return self


__all__ = (
    "LegacyAI2ThorSource",
    "ProceduralSource",
    "SolverConfig",
    "SourceLocator",
    "SourcePlanEntry",
    "SourcePlanManifest",
)
