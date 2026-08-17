"""Canonical 2.3 roots for one exact upright arbitrary-azimuth camera.

This module is a domain-only wire boundary.  It deliberately contains no
projection math, solver logic, adapter inference, evidence, or platform code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spatialcf.domain.v2.base import (
    CanonicalId,
    FactSetV2,
    FiniteFloat,
    V2Model,
    Vec3V2,
)
from spatialcf.domain.v2.continuous_yaw_candidate import (
    CanonicalSceneV2_2,
    SemanticProblemV2_2,
)
from spatialcf.domain.v2.scene import PinholeCameraV2
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_PROBLEM_HASH_DOMAIN_V2_3 = "spatialcf.semantic-problem.v2.3"


class SchemaIdentityV2_3(V2Model):
    """Identity carried only by Canonical 2.3 root contracts."""

    schema_name: CanonicalId
    schema_version: Literal["2.3"] = "2.3"


class UprightWorldToCameraTransformV2_3(V2Model):
    """Fixed upright world-to-camera basis with arbitrary horizontal azimuth."""

    kind: Literal["UPRIGHT_WORLD_TO_CAMERA"] = "UPRIGHT_WORLD_TO_CAMERA"
    azimuth_radians: FiniteFloat
    translation: Vec3V2


class PinholeCameraV2_3(PinholeCameraV2):
    """Pinhole camera whose frame uses the Canonical 2.3 upright basis."""

    world_to_camera: UprightWorldToCameraTransformV2_3


class CanonicalSceneV2_3(CanonicalSceneV2_2):
    """Canonical 2.3 Scene with an explicit upright camera frame."""

    schema_identity: SchemaIdentityV2_3 = Field(
        default_factory=lambda: SchemaIdentityV2_3(schema_name="canonical-scene")
    )
    cameras: FactSetV2[PinholeCameraV2_3]

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_3:
        return SchemaIdentityV2_3(schema_name="canonical-scene")


class SemanticProblemV2_3(SemanticProblemV2_2):
    """Semantic Problem with a domain-separated Canonical 2.3 hash."""

    schema_identity: SchemaIdentityV2_3 = Field(
        default_factory=lambda: SchemaIdentityV2_3(schema_name="semantic-problem")
    )
    scene: CanonicalSceneV2_3

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2_3:
        return SchemaIdentityV2_3(schema_name="semantic-problem")

    @property
    def semantic_problem_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_PROBLEM_HASH_DOMAIN_V2_3)


__all__ = (
    "CanonicalSceneV2_3",
    "PinholeCameraV2_3",
    "SchemaIdentityV2_3",
    "SemanticProblemV2_3",
    "UprightWorldToCameraTransformV2_3",
)
