"""Versioned continuous-yaw Scene roots and candidate compiler config.

This module is a domain-only wire boundary. It deliberately imports no core,
adapter, evidence, publication, or platform module.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FactSetV2,
    UncertaintyBudgetV2,
    V2Model,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
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

_PROBLEM_HASH_DOMAIN_V2_2 = "spatialcf.semantic-problem.v2.2"
_CONFIG_HASH_DOMAIN_V2_5 = "spatialcf.strict-convex-candidate-config.v2.5"
_CONFIG_HASH_DOMAIN_V2_6 = "spatialcf.strict-convex-candidate-config.v2.6"
_CONFIG_HASH_DOMAIN_V2_7 = "spatialcf.strict-convex-candidate-config.v2.7"
_MAX_DETERMINISTIC_LIMIT_V2_5 = 2**63 - 1
_DeterministicLimitV2_5 = Annotated[
    int,
    Field(strict=True, ge=1, le=_MAX_DETERMINISTIC_LIMIT_V2_5),
]


class SchemaIdentityV2_2(V2Model):
    """Identity carried only by Canonical 2.2 root contracts."""

    schema_name: CanonicalId
    schema_version: Literal["2.2"] = "2.2"


class ObjectPoseV2_2(ObjectPoseV2):
    world_from_object: DirectedYawIntervalTransformV2_2


class CanonicalObjectV2_2(CanonicalObjectV2):
    pose: ObjectPoseV2_2


class GeometryInstanceV2_2(V2Model):
    geometry_id: CanonicalId
    owner_object_id: CanonicalId | None
    role: GeometryRoleV2
    anchor_from_geometry: DirectedYawIntervalTransformV2_2
    approximation: GeometryApproximationV2
    uncertainty: UncertaintyBudgetV2
    shape: GeometryShapeV2


class SupportSurfaceFactV2_2(SupportSurfaceFactV2):
    anchor_from_surface: DirectedYawIntervalTransformV2_2


class PinholeCameraV2_2(PinholeCameraV2):
    world_to_camera: DirectedYawIntervalTransformV2_2


class CanonicalSceneV2_2(CanonicalSceneV2):
    """Canonical Scene whose transform-bearing facts use directed yaw."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(schema_name="canonical-scene")
    )
    objects: FactSetV2[CanonicalObjectV2_2]
    geometry_instances: FactSetV2[GeometryInstanceV2_2]
    collision_bodies: FactSetV2[CollisionBodyFactV2]
    workspace_boundaries: FactSetV2[WorkspaceBoundaryFactV2]
    known_free_spaces: FactSetV2[KnownFreeSpaceFactV2]
    support_surfaces: FactSetV2[SupportSurfaceFactV2_2]
    cameras: FactSetV2[PinholeCameraV2_2]
    baseline_observations: FactSetV2[BaselineObservationV2]

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_2:
        return SchemaIdentityV2_2(schema_name="canonical-scene")


class SemanticProblemV2_2(SemanticProblemV2):
    """Semantic Problem with a domain-separated Canonical 2.2 hash."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(schema_name="semantic-problem")
    )
    scene: CanonicalSceneV2_2

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_2:
        return SchemaIdentityV2_2(schema_name="semantic-problem")

    @property
    def semantic_problem_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_PROBLEM_HASH_DOMAIN_V2_2)


class StrictConvexCandidateCompilerConfigV2_5(V2Model):
    """Standalone bounded config for the strict-convex candidate stage."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
    )
    algorithm_id: Literal["solver:canonical-branch-and-bound-v2"] = (
        "solver:canonical-branch-and-bound-v2"
    )
    algorithm_version: Literal["algorithm:2.5"] = "algorithm:2.5"
    so2_kernel_id: Literal["geometry-kernel:rational-so2-upright-box-directed-v2"] = (
        "geometry-kernel:rational-so2-upright-box-directed-v2"
    )
    so2_kernel_version: Literal["kernel:2.2-continuous-yaw-upright-box"] = (
        "kernel:2.2-continuous-yaw-upright-box"
    )
    obstacle_kernel_id: Literal[
        "geometry-kernel:rational-convex-translation-bracket-v2"
    ] = "geometry-kernel:rational-convex-translation-bracket-v2"
    obstacle_kernel_version: Literal["kernel:2.3-convex-translation-bracket"] = (
        "kernel:2.3-convex-translation-bracket"
    )
    partition_kernel_id: Literal[
        "geometry-kernel:rational-convex-complement-partition-v2"
    ] = "geometry-kernel:rational-convex-complement-partition-v2"
    partition_kernel_version: Literal["kernel:2.4-topology-aware-convex-complement"] = (
        "kernel:2.4-topology-aware-convex-complement"
    )
    max_domain_operations: _DeterministicLimitV2_5
    max_so2_atomic_steps: _DeterministicLimitV2_5

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        expected = SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
        if self.schema_identity != expected:
            raise ValueError("strict-convex candidate config identity must be fixed")
        return self

    @property
    def config_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_CONFIG_HASH_DOMAIN_V2_5)


class StrictConvexCandidateCompilerConfigV2_6(V2Model):
    """Standalone bounded config for multi-obstacle strict intersection."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
    )
    algorithm_id: Literal["solver:canonical-branch-and-bound-v2"] = (
        "solver:canonical-branch-and-bound-v2"
    )
    algorithm_version: Literal["algorithm:2.6"] = "algorithm:2.6"
    so2_kernel_id: Literal["geometry-kernel:rational-so2-upright-box-directed-v2"] = (
        "geometry-kernel:rational-so2-upright-box-directed-v2"
    )
    so2_kernel_version: Literal["kernel:2.2-continuous-yaw-upright-box"] = (
        "kernel:2.2-continuous-yaw-upright-box"
    )
    obstacle_kernel_id: Literal[
        "geometry-kernel:rational-convex-translation-bracket-v2"
    ] = "geometry-kernel:rational-convex-translation-bracket-v2"
    obstacle_kernel_version: Literal["kernel:2.3-convex-translation-bracket"] = (
        "kernel:2.3-convex-translation-bracket"
    )
    partition_kernel_id: Literal[
        "geometry-kernel:rational-convex-complement-partition-v2"
    ] = "geometry-kernel:rational-convex-complement-partition-v2"
    partition_kernel_version: Literal["kernel:2.4-topology-aware-convex-complement"] = (
        "kernel:2.4-topology-aware-convex-complement"
    )
    intersection_kernel_id: Literal[
        "geometry-kernel:rational-strict-convex-intersection-v2"
    ] = "geometry-kernel:rational-strict-convex-intersection-v2"
    intersection_kernel_version: Literal["kernel:2.5-strict-convex-intersection"] = (
        "kernel:2.5-strict-convex-intersection"
    )
    max_domain_operations: _DeterministicLimitV2_5
    max_so2_atomic_steps: _DeterministicLimitV2_5
    max_candidate_cells: _DeterministicLimitV2_5

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        expected = SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
        if self.schema_identity != expected:
            raise ValueError("strict-convex candidate config identity must be fixed")
        return self

    @property
    def config_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_CONFIG_HASH_DOMAIN_V2_6)


class StrictConvexCandidateCompilerConfigV2_7(V2Model):
    """Standalone bounded config for exact horizontal support projection."""

    schema_identity: SchemaIdentityV2_2 = Field(
        default_factory=lambda: SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
    )
    algorithm_id: Literal["solver:canonical-branch-and-bound-v2"] = (
        "solver:canonical-branch-and-bound-v2"
    )
    algorithm_version: Literal["algorithm:2.7"] = "algorithm:2.7"
    so2_kernel_id: Literal["geometry-kernel:rational-so2-upright-box-directed-v2"] = (
        "geometry-kernel:rational-so2-upright-box-directed-v2"
    )
    so2_kernel_version: Literal["kernel:2.2-continuous-yaw-upright-box"] = (
        "kernel:2.2-continuous-yaw-upright-box"
    )
    obstacle_kernel_id: Literal[
        "geometry-kernel:rational-convex-translation-bracket-v2"
    ] = "geometry-kernel:rational-convex-translation-bracket-v2"
    obstacle_kernel_version: Literal["kernel:2.3-convex-translation-bracket"] = (
        "kernel:2.3-convex-translation-bracket"
    )
    partition_kernel_id: Literal[
        "geometry-kernel:rational-convex-complement-partition-v2"
    ] = "geometry-kernel:rational-convex-complement-partition-v2"
    partition_kernel_version: Literal["kernel:2.4-topology-aware-convex-complement"] = (
        "kernel:2.4-topology-aware-convex-complement"
    )
    intersection_kernel_id: Literal[
        "geometry-kernel:rational-strict-convex-intersection-v2"
    ] = "geometry-kernel:rational-strict-convex-intersection-v2"
    intersection_kernel_version: Literal["kernel:2.5-strict-convex-intersection"] = (
        "kernel:2.5-strict-convex-intersection"
    )
    support_projection_kernel_id: Literal[
        "geometry-kernel:rational-continuous-yaw-support-projection-v2"
    ] = "geometry-kernel:rational-continuous-yaw-support-projection-v2"
    support_projection_kernel_version: Literal[
        "kernel:2.6-exact-horizontal-support-projection"
    ] = "kernel:2.6-exact-horizontal-support-projection"
    max_domain_operations: _DeterministicLimitV2_5
    max_so2_atomic_steps: _DeterministicLimitV2_5
    max_candidate_cells: _DeterministicLimitV2_5

    @model_validator(mode="after")
    def validate_schema_identity(self) -> Self:
        expected = SchemaIdentityV2_2(
            schema_name="strict-convex-candidate-compiler-config"
        )
        if self.schema_identity != expected:
            raise ValueError("strict-convex candidate config identity must be fixed")
        return self

    @property
    def config_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_CONFIG_HASH_DOMAIN_V2_7)


__all__ = (
    "CanonicalObjectV2_2",
    "CanonicalSceneV2_2",
    "GeometryInstanceV2_2",
    "ObjectPoseV2_2",
    "PinholeCameraV2_2",
    "SchemaIdentityV2_2",
    "SemanticProblemV2_2",
    "StrictConvexCandidateCompilerConfigV2_5",
    "StrictConvexCandidateCompilerConfigV2_6",
    "StrictConvexCandidateCompilerConfigV2_7",
    "SupportSurfaceFactV2_2",
)
