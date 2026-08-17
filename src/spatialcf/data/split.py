"""Deterministic, scene-isolated split and holdout selection."""

import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

_SEED = 20260723
_HOLDOUT_TAGS = frozenset({"unseen_scene", "unseen_category", "unseen_combination"})


class SplitAssignment(BaseModel):
    """A leak-safe split decision attached to a published pair."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    split: Literal["train", "dev", "test"]
    holdout_tags: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def validate_isolation(self) -> "SplitAssignment":
        if not self.holdout_tags.issubset(_HOLDOUT_TAGS):
            raise ValueError("unknown holdout tag")
        if self.split != "test" and self.holdout_tags:
            raise ValueError("holdout-tagged examples are restricted to the test split")
        if self.split == "test" and "unseen_scene" not in self.holdout_tags:
            raise ValueError("test assignments require unseen_scene isolation")
        return self


def _split_for_bucket(bucket: int) -> Literal["train", "dev", "test"]:
    if not 0 <= bucket < 100:
        raise ValueError("bucket must be in [0, 100)")
    return "train" if bucket < 60 else "dev" if bucket < 80 else "test"


def assign_split(scene_id: str, variant_id: str | None = None) -> Literal["train", "dev", "test"]:
    """Assign scenes, never pair variants, by a stable SHA-256 bucket."""
    del variant_id
    if not scene_id or not scene_id.strip():
        raise ValueError("scene_id must be non-empty")
    bucket = int(hashlib.sha256(f"{_SEED}:{scene_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
    return _split_for_bucket(bucket)


def select_holdouts(
    eligible_categories: list[str], combinations: list[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Choose fixed fractions from canonicalized eligible values."""
    rng = random.Random(_SEED)
    if any(not item or not item.strip() for item in eligible_categories + combinations):
        raise ValueError("holdout IDs must be non-empty")
    categories = sorted(set(eligible_categories))
    combos = sorted(set(combinations))
    category_count = max(1, round(len(categories) * 0.10)) if categories else 0
    combo_count = max(1, round(len(combos) * 0.20)) if combos else 0
    return (
        frozenset(rng.sample(categories, category_count)),
        frozenset(rng.sample(combos, combo_count)),
    )
