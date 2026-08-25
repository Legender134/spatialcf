"""Canonical platform-neutral edit contract."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    SchemaIdentityV2,
    Sha256Digest,
    Vec2,
)
from spatialcf.domain.serialization import canonical_sha256

_CANONICAL_EDIT_HASH_DOMAIN = "canonical-edit-v2"


class CanonicalEdit(CanonicalModel):
    """The sole permitted edit: translate one object frame in world X/Y."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="canonical-edit")
    )
    semantic_problem_sha256: Sha256Digest
    subject_id: CanonicalId
    translation_xy_m: Vec2

    @model_validator(mode="after")
    def validate_and_canonicalize_edit(self) -> Self:
        if self.schema_identity.schema_name != "canonical-edit":
            raise ValueError("canonical edit schema identity must be fixed")
        translation = Vec2(
            x=0.0 if self.translation_xy_m.x == 0.0 else self.translation_xy_m.x,
            y=0.0 if self.translation_xy_m.y == 0.0 else self.translation_xy_m.y,
        )
        object.__setattr__(self, "translation_xy_m", translation)
        return self

    @property
    def edit_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_CANONICAL_EDIT_HASH_DOMAIN)
