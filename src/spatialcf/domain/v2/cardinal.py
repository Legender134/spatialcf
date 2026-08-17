"""Exact cardinal-yaw Canonical 2.1 semantic contracts.

The wire deliberately does not contain a quaternion. A quarter turn is an
exact integer in ``{0, 1, 2, 3}``, measured counter-clockwise about positive Z.
All operations use only rational addition, sign changes, and coordinate swaps.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FactSetV2,
    UncertaintyBudgetV2,
    V2Model,
    Vec3V2,
)
from spatialcf.domain.v2.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryRoleV2,
    GeometryShapeV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import (
    BaselineObservationV2,
    CanonicalObjectV2,
    CanonicalSceneV2,
    KnownFreeSpaceFactV2,
    ObjectPoseV2,
    PinholeCameraV2,
    SupportSurfaceFactV2,
    WorkspaceBoundaryFactV2,
)
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_PROBLEM_HASH_DOMAIN_V2_1 = "spatialcf.semantic-problem.v2.1"


class SchemaIdentityV2_1(V2Model):
    """Identity carried only by Canonical 2.1 root contracts."""

    schema_name: CanonicalId
    schema_version: Literal["2.1"] = "2.1"


def _require_exact_quarter_turn(value: object) -> object:
    if type(value) is not int or value not in range(4):
        raise ValueError("quarter_turns_ccw must be an exact integer from 0 through 3")
    return value


ExactQuarterTurnsV2 = Annotated[
    Literal[0, 1, 2, 3],
    BeforeValidator(_require_exact_quarter_turn),
]


class ExactCardinalYawTransformV2(V2Model):
    """Exact local-to-parent Z-cardinal transform in metres."""

    kind: Literal["EXACT_CARDINAL_YAW"] = "EXACT_CARDINAL_YAW"
    translation: Vec3V2
    quarter_turns_ccw: ExactQuarterTurnsV2

    @model_validator(mode="after")
    def canonicalize_signed_zero(self) -> Self:
        translation = self.translation
        object.__setattr__(
            self,
            "translation",
            Vec3V2(
                x=0.0 if translation.x == 0.0 else translation.x,
                y=0.0 if translation.y == 0.0 else translation.y,
                z=0.0 if translation.z == 0.0 else translation.z,
            ),
        )
        return self

    @classmethod
    def identity(cls) -> Self:
        return cls(
            translation=Vec3V2(x=0.0, y=0.0, z=0.0),
            quarter_turns_ccw=0,
        )


ExactRationalPoint3V2 = tuple[Fraction, Fraction, Fraction]


def _rotate_exact_cardinal_v2(
    quarter_turns_ccw: int,
    point: ExactRationalPoint3V2,
) -> ExactRationalPoint3V2:
    x, y, z = point
    if quarter_turns_ccw == 0:
        return x, y, z
    if quarter_turns_ccw == 1:
        return -y, x, z
    if quarter_turns_ccw == 2:
        return -x, -y, z
    return y, -x, z


def apply_exact_cardinal_transform_v2(
    transform: ExactCardinalYawTransformV2,
    point: ExactRationalPoint3V2,
) -> ExactRationalPoint3V2:
    """Apply ``p_parent = t + R_q p_local`` exactly."""

    rotated = _rotate_exact_cardinal_v2(transform.quarter_turns_ccw, point)
    translation = transform.translation
    return (
        Fraction.from_float(translation.x) + rotated[0],
        Fraction.from_float(translation.y) + rotated[1],
        Fraction.from_float(translation.z) + rotated[2],
    )


def _exact_binary64(value: Fraction, field_name: str) -> float:
    try:
        candidate = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{field_name} is not a finite exact binary64 value"
        ) from error
    if not math.isfinite(candidate) or Fraction.from_float(candidate) != value:
        raise ValueError(f"{field_name} is not an exact binary64 value")
    return 0.0 if candidate == 0.0 else candidate


def compose_exact_cardinal_transforms_v2(
    parent_from_intermediate: ExactCardinalYawTransformV2,
    intermediate_from_local: ExactCardinalYawTransformV2,
) -> ExactCardinalYawTransformV2:
    """Compose two exact transforms, rejecting an unrepresentable wire result."""

    right_translation = intermediate_from_local.translation
    transformed_translation = apply_exact_cardinal_transform_v2(
        parent_from_intermediate,
        (
            Fraction.from_float(right_translation.x),
            Fraction.from_float(right_translation.y),
            Fraction.from_float(right_translation.z),
        ),
    )
    return ExactCardinalYawTransformV2(
        translation=Vec3V2(
            x=_exact_binary64(transformed_translation[0], "translation.x"),
            y=_exact_binary64(transformed_translation[1], "translation.y"),
            z=_exact_binary64(transformed_translation[2], "translation.z"),
        ),
        quarter_turns_ccw=(
            parent_from_intermediate.quarter_turns_ccw
            + intermediate_from_local.quarter_turns_ccw
        )
        % 4,
    )


def invert_exact_cardinal_transform_v2(
    parent_from_local: ExactCardinalYawTransformV2,
) -> ExactCardinalYawTransformV2:
    """Return the exact local-from-parent inverse."""

    inverse_quarter_turns = (-parent_from_local.quarter_turns_ccw) % 4
    translation = parent_from_local.translation
    inverse_translation = _rotate_exact_cardinal_v2(
        inverse_quarter_turns,
        (
            -Fraction.from_float(translation.x),
            -Fraction.from_float(translation.y),
            -Fraction.from_float(translation.z),
        ),
    )
    return ExactCardinalYawTransformV2(
        translation=Vec3V2(
            x=_exact_binary64(inverse_translation[0], "translation.x"),
            y=_exact_binary64(inverse_translation[1], "translation.y"),
            z=_exact_binary64(inverse_translation[2], "translation.z"),
        ),
        quarter_turns_ccw=inverse_quarter_turns,
    )


class ObjectPoseV2_1(ObjectPoseV2):
    world_from_object: ExactCardinalYawTransformV2


class CanonicalObjectV2_1(CanonicalObjectV2):
    pose: ObjectPoseV2_1


class GeometryInstanceV2_1(V2Model):
    geometry_id: CanonicalId
    owner_object_id: CanonicalId | None
    role: GeometryRoleV2
    anchor_from_geometry: ExactCardinalYawTransformV2
    approximation: GeometryApproximationV2
    uncertainty: UncertaintyBudgetV2
    shape: GeometryShapeV2


class SupportSurfaceFactV2_1(SupportSurfaceFactV2):
    anchor_from_surface: ExactCardinalYawTransformV2

    @model_validator(mode="after")
    def require_exact_positive_z_normal(self) -> Self:
        normal = self.normal_in_anchor
        if (normal.x, normal.y, normal.z) != (0.0, 0.0, 1.0):
            raise ValueError("cardinal support surface normal must be exact +Z")
        return self


class PinholeCameraV2_1(PinholeCameraV2):
    world_to_camera: ExactCardinalYawTransformV2


class CanonicalSceneV2_1(CanonicalSceneV2):
    """Canonical scene root whose transform-bearing facts use the 2.1 wire."""

    schema_identity: SchemaIdentityV2_1 = Field(
        default_factory=lambda: SchemaIdentityV2_1(schema_name="canonical-scene")
    )
    objects: FactSetV2[CanonicalObjectV2_1]
    geometry_instances: FactSetV2[GeometryInstanceV2_1]
    collision_bodies: FactSetV2[CollisionBodyFactV2]
    workspace_boundaries: FactSetV2[WorkspaceBoundaryFactV2]
    known_free_spaces: FactSetV2[KnownFreeSpaceFactV2]
    support_surfaces: FactSetV2[SupportSurfaceFactV2_1]
    cameras: FactSetV2[PinholeCameraV2_1]
    baseline_observations: FactSetV2[BaselineObservationV2]

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_1:
        return SchemaIdentityV2_1(schema_name="canonical-scene")


class SemanticProblemV2_1(SemanticProblemV2):
    """Semantic problem root with an independently domain-separated 2.1 hash."""

    schema_identity: SchemaIdentityV2_1 = Field(
        default_factory=lambda: SchemaIdentityV2_1(schema_name="semantic-problem")
    )
    scene: CanonicalSceneV2_1

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_1:
        return SchemaIdentityV2_1(schema_name="semantic-problem")

    @property
    def semantic_problem_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_PROBLEM_HASH_DOMAIN_V2_1)


__all__ = (
    "CanonicalObjectV2_1",
    "CanonicalSceneV2_1",
    "ExactCardinalYawTransformV2",
    "GeometryInstanceV2_1",
    "ObjectPoseV2_1",
    "PinholeCameraV2_1",
    "SchemaIdentityV2_1",
    "SemanticProblemV2_1",
    "SupportSurfaceFactV2_1",
    "apply_exact_cardinal_transform_v2",
    "compose_exact_cardinal_transforms_v2",
    "invert_exact_cardinal_transform_v2",
)
