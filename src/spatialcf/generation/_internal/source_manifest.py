"""Current source-plan parsing and file IO."""

from __future__ import annotations

import json
from pathlib import Path

import spatialcf.domain.source as source_domain
from spatialcf.generation._internal.canonical_json import canonical_json_bytes


def load_source_plan_manifest(path: Path) -> source_domain.SourcePlanManifest:
    """Strictly parse one canonical source-plan manifest."""

    raw = Path(path).read_bytes()
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source-plan manifest must be valid UTF-8 JSON") from error
    if type(decoded) is not dict:
        raise ValueError("source-plan manifest must be a JSON object")
    observed_version = decoded.get("schema_version")
    expected_version = "certified-ai2thor-source-plan-manifest-v1"
    if observed_version != expected_version:
        from spatialcf.generation.errors import UnsupportedArtifactVersion

        raise UnsupportedArtifactVersion(
            "source-plan manifest",
            expected=expected_version,
            observed=observed_version,
        )
    manifest = source_domain.SourcePlanManifest.model_validate_json(raw, strict=True)
    if raw != canonical_json_bytes(manifest.model_dump(mode="json"), pretty=True):
        raise ValueError("source-plan manifest must use canonical pretty JSON")
    return manifest


__all__ = ("load_source_plan_manifest",)
