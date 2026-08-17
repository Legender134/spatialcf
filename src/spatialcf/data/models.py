"""Strict, immutable records for published counterfactual datasets."""

import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus

_DATASET_SEED = 20260723
_HOLDOUT_TAGS = frozenset({"unseen_scene", "unseen_category", "unseen_combination"})


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _validate_relative_path(value: str) -> str:
    """Keep artifact references inside the immutable dataset directory."""
    if not value or "\\" in value:
        raise ValueError("artifact path must be a non-empty relative POSIX path")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("artifact path must be a non-empty relative POSIX path")
    if value.split("/", 1)[0] not in {"assets", "scenes", "relations", "topdown"}:
        raise ValueError("artifact path must use an approved artifact prefix")
    if value in {"pairs.jsonl", "failures.jsonl", "manifest.json", "checksums.sha256"}:
        raise ValueError("artifact path may not alias dataset metadata")
    return value


class PairRecord(_FrozenRecord):
    """An accepted, independently verified counterfactual pair."""

    pair_id: str
    request_id: str
    scene_id: str
    split: Literal["train", "dev", "test"]
    holdout_tags: frozenset[str]
    source: str
    seed: int
    generator: str
    subject_id: str
    subject_category: str
    reference_id: str
    reference_category: str
    camera_id: str
    relation_before: Relation
    relation_after: Relation
    question: str
    answer_before: Relation
    answer_after: Relation
    scene_before_path: str
    scene_after_path: str
    rgb_before_path: str
    rgb_after_path: str
    depth_before_path: str
    depth_after_path: str
    instance_before_path: str
    instance_after_path: str
    pointcloud_before_path: str
    pointcloud_after_path: str
    topdown_path: str
    relation_graph_before_path: str
    relation_graph_after_path: str
    relation_diff: tuple[str, ...]
    normalized_edit_distance: float
    leakage_score: float
    visibility_change: float
    inverse_safety_margin: float
    solver_status: SolverStatus
    evaluated_candidates: int
    quality_flags: tuple[str, ...]
    quality: QualityTier
    generator_version: str

    @classmethod
    def artifact_path_fields(cls) -> tuple[str, ...]:
        return tuple(name for name in cls.model_fields if name.endswith("_path"))

    @field_validator(
        "pair_id", "request_id", "scene_id", "source", "generator", "subject_id",
        "subject_category", "reference_id", "reference_category", "camera_id", "question",
        "generator_version",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("required text fields must be non-empty and non-whitespace")
        return value

    @field_validator(
        "scene_before_path",
        "scene_after_path",
        "rgb_before_path",
        "rgb_after_path",
        "depth_before_path",
        "depth_after_path",
        "instance_before_path",
        "instance_after_path",
        "pointcloud_before_path",
        "pointcloud_after_path",
        "topdown_path",
        "relation_graph_before_path",
        "relation_graph_after_path",
    )
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact path must be non-empty")
        return _validate_relative_path(value)

    @field_validator(
        "normalized_edit_distance",
        "leakage_score",
        "visibility_change",
        "inverse_safety_margin",
    )
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("scores must be finite and non-negative")
        return value

    @field_validator("normalized_edit_distance", "leakage_score")
    @classmethod
    def validate_normalized_score(cls, value: float) -> float:
        if value > 1.0:
            raise ValueError("normalized scores must not exceed one")
        return value

    @field_serializer("holdout_tags", when_used="json")
    def serialize_holdout_tags(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_accepted_pair(self) -> "PairRecord":
        if any(not value for value in (self.pair_id, self.request_id, self.scene_id)):
            raise ValueError("pair_id, request_id, and scene_id must be non-empty")
        if self.seed != _DATASET_SEED:
            raise ValueError(f"seed must be the deterministic dataset seed {_DATASET_SEED}")
        if self.subject_id == self.reference_id:
            raise ValueError("subject_id and reference_id must differ")
        if self.relation_before.opposite is not self.relation_after:
            raise ValueError("accepted pairs must use an opposite relation flip")
        if (
            self.answer_before is not self.relation_before
            or self.answer_after is not self.relation_after
        ):
            raise ValueError("answers must preserve independently verified relations")
        if self.solver_status is not SolverStatus.SUCCESS:
            raise ValueError("accepted pairs require independent verifier status SUCCESS")
        if self.quality is QualityTier.REJECTED:
            raise ValueError("accepted pairs cannot have REJECTED quality")
        if self.quality is QualityTier.PURE:
            if self.leakage_score != 0.0 or self.quality_flags != ("PURE",):
                raise ValueError("PURE pairs require zero leakage and a PURE quality flag")
        if self.quality is QualityTier.LOW_LEAKAGE:
            if self.leakage_score <= 0.0 or self.quality_flags != ("LOW_LEAKAGE",):
                raise ValueError(
                    "LOW_LEAKAGE pairs require positive leakage and a LOW_LEAKAGE quality flag"
                )
        if not self.holdout_tags.issubset(_HOLDOUT_TAGS):
            raise ValueError("unknown holdout tag")
        if self.split == "test":
            if "unseen_scene" not in self.holdout_tags:
                raise ValueError("test pairs require the unseen_scene holdout tag")
            if self.quality is not QualityTier.PURE:
                raise ValueError("test split must remain PURE")
        elif self.holdout_tags:
            raise ValueError("holdout tags are permitted only on the test split")
        if self.evaluated_candidates < 0:
            raise ValueError("evaluated_candidates must be non-negative")
        return self


class FailureRecord(_FrozenRecord):
    """Append-only evidence for a request that did not become accepted data."""

    failure_id: str
    request_id: str
    scene_id: str
    subject_id: str
    reference_id: str
    relation_before: Relation
    relation_after: Relation
    generator: str
    generator_version: str
    seed: int
    status: SolverStatus
    reason: str
    evaluated_candidates: int

    @field_validator(
        "failure_id",
        "request_id",
        "scene_id",
        "subject_id",
        "reference_id",
        "generator",
        "generator_version",
        "reason",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("required text fields must be non-empty and non-whitespace")
        return value

    @model_validator(mode="after")
    def validate_failure(self) -> "FailureRecord":
        if any(
            not value
            for value in (
                self.failure_id,
                self.request_id,
                self.scene_id,
                self.reason,
            )
        ):
            raise ValueError(
                "failure_id, request_id, scene_id, and reason must be non-empty"
            )
        if self.seed != _DATASET_SEED:
            raise ValueError(f"seed must be the deterministic dataset seed {_DATASET_SEED}")
        if self.subject_id == self.reference_id:
            raise ValueError("subject_id and reference_id must differ")
        if self.relation_before.opposite is not self.relation_after:
            raise ValueError("failures must retain an opposite relation flip")
        if self.status is SolverStatus.SUCCESS:
            raise ValueError("failure status must not be SUCCESS")
        if self.evaluated_candidates < 0:
            raise ValueError("evaluated_candidates must be non-negative")
        return self
