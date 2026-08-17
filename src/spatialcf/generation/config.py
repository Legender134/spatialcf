"""Strict, version-free configuration for dataset generation."""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    read_regular_at,
    revalidate_entries,
)

_MAX_CONFIG_BYTES = 1024 * 1024


class GenerationConfig(BaseModel):
    """The complete supported public dataset-generation configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    config_version: Literal[1] = 1
    adapter: Literal["ai2thor"] = "ai2thor"
    scene_names: tuple[str, ...] = Field(min_length=1, max_length=32)
    split: Literal["train", "validation", "test"] = "train"
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    seed: int = 20260723
    width: int = Field(default=640, ge=1, le=4096)
    height: int = Field(default=480, ge=1, le=4096)
    max_requests: int = Field(default=60, ge=1, le=1000)

    @model_validator(mode="after")
    def canonicalize_scene_names(self) -> Self:
        if any(type(item) is not str or not item for item in self.scene_names):
            raise ValueError("scene_names must contain non-empty exact strings")
        object.__setattr__(self, "scene_names", tuple(sorted(set(self.scene_names))))
        return self


def load_generation_config(path: Path) -> GenerationConfig:
    """Read one regular TOML file and return the exact strict public model."""

    if not isinstance(path, Path):
        raise TypeError("generation config path must be a Path")
    absolute = Path(os.path.abspath(path))
    if absolute == absolute.parent or absolute.name in {"", ".", ".."}:
        raise ValueError("generation config path is unsafe")
    with bound_absolute_directory(absolute.parent) as descriptor:
        entry = os.stat(absolute.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise ValueError("generation config input must be regular")
        payload = read_regular_at(
            descriptor,
            absolute.name,
            _MAX_CONFIG_BYTES,
            expected_stat=entry,
        )
        revalidate_entries(descriptor, {absolute.name: entry})
    try:
        parsed = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("generation config must be valid UTF-8 TOML") from error
    if type(parsed) is not dict:
        raise ValueError("generation config must be one TOML table")
    data = dict(parsed)
    scenes = data.get("scene_names")
    if type(scenes) is list:
        if any(type(item) is not str for item in scenes):
            raise TypeError("generation config scene_names must be strings")
        data["scene_names"] = tuple(scenes)
    return GenerationConfig.model_validate(data, strict=True)


__all__ = ("GenerationConfig", "load_generation_config")
