"""Canonical evidence needed to replay official dataset generation."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from spatialcf.domain.enums import QualityTier, SolverStatus
from spatialcf.domain.models import InterventionSpec, Vec3
from spatialcf.solver.objective import ObjectiveBreakdown
from spatialcf.solver.search import SolveResult


ATTESTATION_SCHEMA_VERSION = 3
DATASET_MANIFEST_SCHEMA_VERSION = 4
LEGACY_ATTESTED_MANIFEST_SCHEMA_VERSION = 3
ATTESTED_MANIFEST_SCHEMA_VERSIONS = frozenset({
    LEGACY_ATTESTED_MANIFEST_SCHEMA_VERSION,
    DATASET_MANIFEST_SCHEMA_VERSION,
})
GENERATOR_VERSION = "0.1.0"
DATASET_SEED = 20260723


class _FrozenEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _provenance_path(value: str) -> str:
    if (
        not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or value.split("/", 1)[0] != "provenance"
    ):
        raise ValueError("provenance path must be a safe relative POSIX path")
    return value


class SourceSceneEvidence(_FrozenEvidence):
    scene_id: str
    path: str
    sha256: str

    @field_validator("scene_id")
    @classmethod
    def required_scene_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source scene_id must be non-empty")
        return value

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _provenance_path(value)

    @field_validator("sha256")
    @classmethod
    def checksum(cls, value: str) -> str:
        if (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("source scene checksum must be lowercase SHA-256")
        return value


class GenerationProvenance(_FrozenEvidence):
    schema_version: int
    adapter_backend: Literal["json", "ai2thor"]
    adapter_implementation: str
    adapter_config: dict[str, Any]
    generator: Literal["spatialcf", "random", "target-only"]
    generator_implementation: str
    generator_version: str
    generator_config: dict[str, Any]
    dataset_seed: int
    requested_pairs: int | None
    attempt_limit: int | None
    attempted_requests: int
    source_corpus_sha256: str
    source_scenes: tuple[SourceSceneEvidence, ...]

    @field_validator(
        "adapter_implementation",
        "generator_implementation",
        "generator_version",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("provenance implementation fields must be non-empty")
        return value

    @field_validator("schema_version")
    @classmethod
    def supported_schema(cls, value: int) -> int:
        if value != ATTESTATION_SCHEMA_VERSION:
            raise ValueError("unsupported generation attestation schema")
        return value

    @field_validator("dataset_seed")
    @classmethod
    def deterministic_seed(cls, value: int) -> int:
        if value != DATASET_SEED:
            raise ValueError("generation attestation has the wrong dataset seed")
        return value

    @field_validator("generator_version")
    @classmethod
    def supported_generator_version(cls, value: str) -> str:
        if value != GENERATOR_VERSION:
            raise ValueError("unsupported official generator version")
        return value

    @field_validator("attempted_requests")
    @classmethod
    def nonnegative_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("attempted_requests must be non-negative")
        return value

    @model_validator(mode="after")
    def exact_generation_budget(self) -> "GenerationProvenance":
        for name, value in (
            ("requested_pairs", self.requested_pairs),
            ("attempt_limit", self.attempt_limit),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"{name} must be null or a positive exact integer"
                )
        if (self.requested_pairs is None) == (self.attempt_limit is None):
            raise ValueError(
                "exactly one of requested_pairs and attempt_limit is required"
            )
        if (
            self.attempt_limit is not None
            and self.attempted_requests != self.attempt_limit
        ):
            raise ValueError(
                "attempt_limit requires a complete exact attempt prefix"
            )
        return self

    @field_validator("source_corpus_sha256")
    @classmethod
    def corpus_checksum(cls, value: str) -> str:
        if (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("source corpus digest must be lowercase SHA-256")
        return value

    @field_validator("adapter_config")
    @classmethod
    def exact_adapter_config(
        cls,
        value: dict[str, Any],
        info: Any,
    ) -> dict[str, Any]:
        backend = info.data.get("adapter_backend")
        if backend == "json":
            if value != {"mode": "embedded-canonical-scenes"}:
                raise ValueError("JSON adapter config is not the official config")
        elif backend == "ai2thor":
            if set(value) != {"scene_names", "width", "height", "seed"}:
                raise ValueError("AI2-THOR adapter config has unexpected fields")
            names = value["scene_names"]
            if (
                type(names) is not list
                or not names
                or any(type(name) is not str or not name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError("AI2-THOR scene_names must be a unique list")
            for field in ("width", "height", "seed"):
                if type(value[field]) is not int:
                    raise ValueError(
                        f"AI2-THOR adapter {field} must be an exact integer"
                    )
            if value["width"] <= 0 or value["height"] <= 0:
                raise ValueError("AI2-THOR dimensions must be positive")
            if value["seed"] != DATASET_SEED:
                raise ValueError("AI2-THOR adapter seed is not official")
        return value

    @field_validator("generator_config")
    @classmethod
    def exact_generator_config(
        cls,
        value: dict[str, Any],
        info: Any,
    ) -> dict[str, Any]:
        generator = info.data.get("generator")
        if generator == "spatialcf":
            expected_keys = {
                "seed",
                "grid_step",
                "refine_steps",
                "max_candidates",
                "timeout_seconds",
            }
            if set(value) != expected_keys:
                raise ValueError("spatial generator config has unexpected fields")
            if type(value["seed"]) is not int or type(value["max_candidates"]) is not int:
                raise ValueError("spatial integer config fields must be exact integers")
            if type(value["grid_step"]) is not float:
                raise ValueError("spatial grid_step must be an exact float")
            if (
                type(value["refine_steps"]) is not list
                or any(type(item) is not float for item in value["refine_steps"])
            ):
                raise ValueError("spatial refine_steps must be a JSON float list")
            if value["timeout_seconds"] is not None:
                raise ValueError("spatial timeout_seconds must be null")
        elif generator in {"random", "target-only"}:
            if set(value) != {"max_candidates", "seed"}:
                raise ValueError("baseline generator config has unexpected fields")
            if (
                type(value["max_candidates"]) is not int
                or type(value["seed"]) is not int
            ):
                raise ValueError("baseline config values must be exact integers")
        return value


class ObjectiveEvidence(_FrozenEvidence):
    normalized_translation: float
    leakage: float
    visibility_change: float
    inverse_safety_margin: float
    total: float

    @classmethod
    def from_score(cls, score: ObjectiveBreakdown) -> "ObjectiveEvidence":
        return cls(
            normalized_translation=score.normalized_translation,
            leakage=score.leakage,
            visibility_change=score.visibility_change,
            inverse_safety_margin=score.inverse_safety_margin,
            total=score.total,
        )


class GeneratorResultEvidence(_FrozenEvidence):
    status: SolverStatus
    subject_position: Vec3 | None
    score: ObjectiveEvidence | None
    quality: QualityTier
    evaluated_candidates: int
    reason: str | None

    @classmethod
    def from_result(cls, result: SolveResult) -> "GeneratorResultEvidence":
        return cls(
            status=result.status,
            subject_position=result.subject_position,
            score=(
                ObjectiveEvidence.from_score(result.score)
                if result.score is not None
                else None
            ),
            quality=result.quality,
            evaluated_candidates=result.evaluated_candidates,
            reason=result.reason,
        )


class AttemptEvidence(_FrozenEvidence):
    attempt_index: int
    request_id: str
    scene_id: str
    spec: InterventionSpec
    holdout_tags: frozenset[str]
    generator_result: GeneratorResultEvidence
    outcome: Literal["pair", "failure"]
    outcome_id: str

    @field_validator("attempt_index")
    @classmethod
    def positive_index(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("attempt_index must be positive")
        return value

    @field_validator("request_id", "scene_id", "outcome_id")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("attempt identity fields must be non-empty")
        return value

    @field_serializer("holdout_tags", when_used="json")
    def serialize_holdout_tags(self, value: frozenset[str]) -> list[str]:
        return sorted(value)
