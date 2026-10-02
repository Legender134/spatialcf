"""A declared source request selector, fixed before capture and solving."""

from typing import Literal, Self
from pydantic import Field, model_validator
from spatialcf.domain.base import CanonicalModel

LEGACY_ROSTER_POLICY = "competition-native-candidate-roster-policy:2.9.4"
SELECTED_ROSTER_POLICY = "competition-native-candidate-roster-policy:2.9.5"


class FixedRequest(CanonicalModel):
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    subject_name: str = Field(strict=True, min_length=1, max_length=512)
    reference_name: str = Field(strict=True, min_length=1, max_length=512)
    relation_before: Literal["left", "right", "front", "behind", "near", "far"]

    @model_validator(mode="after")
    def distinct_objects(self) -> Self:
        if self.subject_name == self.reference_name:
            raise ValueError("fixed request objects must be distinct")
        return self

    def matches(self, candidate) -> bool:
        return (candidate.scene_id, candidate.subject_name, candidate.reference_name,
                candidate.relation_before) == (
                    self.scene_id, self.subject_name, self.reference_name, self.relation_before)
