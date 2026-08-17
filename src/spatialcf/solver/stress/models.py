"""Frozen data contracts shared by solver-independent stress modules."""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import InterventionSpec, Scene, Vec2

StressProfileName: TypeAlias = Literal["quick", "deep"]
StressDirection: TypeAlias = Literal["lr", "fb", "nf"]
StressFamily: TypeAlias = Literal[
    "target_boundary",
    "obstacle_corner",
    "support_boundary",
    "preservation_intersection",
    "tied_optimum",
    "room_bounds",
    "support_locus",
    "obstacle_coverage",
    "relation_upper_bound",
]

SAT_STRESS_FAMILIES = frozenset(
    {
        "target_boundary",
        "obstacle_corner",
        "support_boundary",
        "preservation_intersection",
        "tied_optimum",
    }
)
UNSAT_STRESS_FAMILIES = frozenset(
    {
        "room_bounds",
        "support_locus",
        "obstacle_coverage",
        "relation_upper_bound",
    }
)
_DIRECTION_FLIPS: dict[StressDirection, tuple[Relation, Relation]] = {
    "lr": (Relation.LEFT, Relation.RIGHT),
    "fb": (Relation.FRONT, Relation.BEHIND),
    "nf": (Relation.NEAR, Relation.FAR),
}


class _FrozenStressModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StressTransform(_FrozenStressModel):
    """An identity or oracle-preserving whole-scene XY transformation."""

    translation_xy: Vec2 = Vec2(x=0.0, y=0.0)
    mirror: Literal["none", "camera_horizontal", "camera_depth"] = "none"
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    base_case_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @property
    def transformed(self) -> bool:
        return (
            self.translation_xy != Vec2(x=0.0, y=0.0)
            or self.mirror != "none"
            or self.rotation_degrees != 0
        )


class StressSlot(_FrozenStressModel):
    """One closed schedule position before case construction and ID sorting."""

    seed: int = Field(ge=2026080200, le=2026080209, strict=True)
    direction: StressDirection
    raw_slot: int = Field(ge=0, le=99, strict=True)
    family: StressFamily
    expected_outcome: Literal["SAT", "UNSAT"]
    transformed: bool

    @model_validator(mode="after")
    def validate_outcome_family_and_slot(self) -> StressSlot:
        if self.expected_outcome == "SAT":
            if self.family not in SAT_STRESS_FAMILIES or self.raw_slot >= 60:
                raise ValueError("SAT stress slot has an invalid family or raw slot")
        elif self.family not in UNSAT_STRESS_FAMILIES or self.raw_slot < 60:
            raise ValueError("UNSAT stress slot has an invalid family or raw slot")
        return self

    @property
    def key(self) -> str:
        return f"stress-v1/{self.seed}/{self.direction}/{self.raw_slot:03d}"


class SatStressOracle(_FrozenStressModel):
    expected_outcome: Literal["SAT"] = "SAT"
    proof_kind: StressFamily
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    exact_infimum_points: tuple[Vec2, ...] = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sat_oracle(self) -> SatStressOracle:
        if self.proof_kind not in SAT_STRESS_FAMILIES:
            raise ValueError("SAT oracle proof_kind must be a SAT stress family")
        if not all(
            math.isfinite(value)
            for point in self.exact_infimum_points
            for value in (point.x, point.y)
        ):
            raise ValueError("SAT oracle witness points must be finite")
        return self


class UnsatStressOracle(_FrozenStressModel):
    expected_outcome: Literal["UNSAT"] = "UNSAT"
    proof_kind: StressFamily
    maximum_possible_value_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_value_m: float = Field(allow_inf_nan=False, strict=True)
    expected_reason: str = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unsat_oracle(self) -> UnsatStressOracle:
        if self.proof_kind not in UNSAT_STRESS_FAMILIES:
            raise ValueError("UNSAT oracle proof_kind must be an UNSAT stress family")
        if self.maximum_possible_value_m >= self.required_value_m:
            raise ValueError("UNSAT oracle must have a strict reachable-value bound")
        return self


StressOracle = Annotated[
    SatStressOracle | UnsatStressOracle,
    Field(discriminator="expected_outcome"),
]


class StressOracleResult(_FrozenStressModel):
    expected_outcome: Literal["SAT", "UNSAT"]
    exact_infimum_m: float | None
    exact_infimum_points: tuple[Vec2, ...]
    maximum_possible_value_m: float | None
    required_value_m: float | None


def expected_oracle_result(oracle: StressOracle) -> StressOracleResult:
    """Project stored declarations to fields independently recomputed later."""
    if isinstance(oracle, SatStressOracle):
        return StressOracleResult(
            expected_outcome="SAT",
            exact_infimum_m=oracle.exact_infimum_m,
            exact_infimum_points=oracle.exact_infimum_points,
            maximum_possible_value_m=None,
            required_value_m=None,
        )
    return StressOracleResult(
        expected_outcome="UNSAT",
        exact_infimum_m=None,
        exact_infimum_points=(),
        maximum_possible_value_m=oracle.maximum_possible_value_m,
        required_value_m=oracle.required_value_m,
    )


class StressCaseDraft(_FrozenStressModel):
    seed: int = Field(ge=2026080200, le=2026080209, strict=True)
    direction: StressDirection
    raw_slot: int = Field(ge=0, le=99, strict=True)
    family: StressFamily
    transform: StressTransform = StressTransform()
    scene: Scene
    intervention: InterventionSpec
    oracle: StressOracle

    @model_validator(mode="after")
    def validate_draft_oracle(self) -> StressCaseDraft:
        is_sat = isinstance(self.oracle, SatStressOracle)
        if (is_sat and self.family not in SAT_STRESS_FAMILIES) or (
            not is_sat and self.family not in UNSAT_STRESS_FAMILIES
        ):
            raise ValueError("stress case family must match its oracle outcome")
        if (self.intervention.relation_before, self.intervention.relation_after) != (
            _DIRECTION_FLIPS[self.direction]
        ):
            raise ValueError("stress case direction must match its intervention flip")
        return self

    @property
    def expected_outcome(self) -> Literal["SAT", "UNSAT"]:
        return self.oracle.expected_outcome


class StressCase(StressCaseDraft):
    case_id: str = Field(pattern=r"^stress-\d{10}-(?:lr|fb|nf)-\d{3}$")

    @model_validator(mode="after")
    def validate_scene_identity(self) -> StressCase:
        if self.scene.scene_id != self.case_id:
            raise ValueError("stress case scene_id must equal case_id")
        _, encoded_seed, encoded_direction, _ = self.case_id.split("-")
        if int(encoded_seed) != self.seed or encoded_direction != self.direction:
            raise ValueError(
                "stress case_id seed and direction must match the stress case"
            )
        return self

    def as_draft(self) -> StressCaseDraft:
        return StressCaseDraft.model_validate(
            self.model_dump(mode="python", exclude={"case_id"})
        )
