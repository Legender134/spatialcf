"""Strict value model returned by Canonical 2.1 scene-fact adapters."""

from __future__ import annotations

from spatialcf.adapters.canonical_v2_models import CanonicalSceneAdaptationV2
from spatialcf.domain.v2.cardinal import CanonicalSceneV2_1


class CanonicalSceneAdaptationV2_1(CanonicalSceneAdaptationV2):
    """Closed Canonical 2.1 scene, provenance, and native object bindings."""

    scene: CanonicalSceneV2_1

    @classmethod
    def _scene_model_type(cls) -> type[CanonicalSceneV2_1]:
        return CanonicalSceneV2_1
