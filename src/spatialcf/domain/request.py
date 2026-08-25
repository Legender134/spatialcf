"""Version-free counterfactual request values and relation result enums."""

from enum import StrEnum

from pydantic import model_validator

from spatialcf.domain.scene import FrozenModel


class RelationAxis(StrEnum):
    HORIZONTAL = "horizontal"
    DEPTH = "depth"
    DISTANCE = "distance"


class Relation(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"

    @property
    def axis(self) -> RelationAxis:
        return {
            Relation.LEFT: RelationAxis.HORIZONTAL,
            Relation.RIGHT: RelationAxis.HORIZONTAL,
            Relation.FRONT: RelationAxis.DEPTH,
            Relation.BEHIND: RelationAxis.DEPTH,
            Relation.NEAR: RelationAxis.DISTANCE,
            Relation.FAR: RelationAxis.DISTANCE,
        }[self]

    @property
    def converse(self) -> "Relation":
        return {
            Relation.LEFT: Relation.RIGHT,
            Relation.RIGHT: Relation.LEFT,
            Relation.FRONT: Relation.BEHIND,
            Relation.BEHIND: Relation.FRONT,
            Relation.NEAR: Relation.NEAR,
            Relation.FAR: Relation.FAR,
        }[self]

    @property
    def opposite(self) -> "Relation":
        return {
            Relation.LEFT: Relation.RIGHT,
            Relation.RIGHT: Relation.LEFT,
            Relation.FRONT: Relation.BEHIND,
            Relation.BEHIND: Relation.FRONT,
            Relation.NEAR: Relation.FAR,
            Relation.FAR: Relation.NEAR,
        }[self]

    @property
    def inverse(self) -> "Relation":
        """Compatibility alias for the relation converse."""
        return self.converse


class SolverStatus(StrEnum):
    SUCCESS = "SUCCESS"
    UNSATISFIABLE = "UNSATISFIABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_SCENE = "INVALID_SCENE"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTIFIED = "UNCERTIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_VISIBLE = "NOT_VISIBLE"


class QualityTier(StrEnum):
    PURE = "PURE"
    LOW_LEAKAGE = "LOW_LEAKAGE"
    REJECTED = "REJECTED"


class InterventionSpec(FrozenModel):
    subject_id: str
    reference_id: str
    relation_before: Relation
    relation_after: Relation
    camera_id: str

    @model_validator(mode="after")
    def validate_flip(self) -> "InterventionSpec":
        if self.subject_id == self.reference_id:
            raise ValueError("subject and reference must differ")
        if self.relation_before == self.relation_after:
            raise ValueError("counterfactual must change the relation")
        if self.relation_before.axis is not self.relation_after.axis:
            raise ValueError("counterfactual relations must share one relation axis")
        if self.relation_before.opposite != self.relation_after:
            raise ValueError("MVP supports opposite relation flips only")
        return self
