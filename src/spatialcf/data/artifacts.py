"""Shared canonical serializers for published pair evidence."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any

from shapely.geometry import mapping

from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.geometry.obb import obb_footprint
from spatialcf.solver.feasible import FeasibleRegionBuilder


def canonical_value(value: Any) -> Any:
    """Convert model values into a deterministic JSON-compatible tree."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (canonical_value(item) for item in value),
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON with the repository's immutable canonical encoding."""
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (
        json.dumps(canonical_value(value), **options) + "\n"
    ).encode("utf-8")


def topdown_payload(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
) -> dict[str, Any]:
    """Build the canonical, independently reproducible top-down evidence."""
    subject_before = before.object_by_id(spec.subject_id)
    subject_after = after.object_by_id(spec.subject_id)
    return {
        "camera_id": spec.camera_id,
        "feasible_region": mapping(
            FeasibleRegionBuilder().build(before, spec)
        ),
        "movement_path": [
            [subject_before.position.x, subject_before.position.y],
            [subject_after.position.x, subject_after.position.y],
        ],
        "objects": [
            {
                "center": [obj.position.x, obj.position.y],
                "object_id": obj.object_id,
                "polygon": [
                    list(point)
                    for point in obb_footprint(obj.obb).exterior.coords
                ],
            }
            for obj in sorted(
                after.objects,
                key=lambda item: item.object_id,
            )
        ],
        "reference_id": spec.reference_id,
        "relation_after": spec.relation_after,
        "relation_before": spec.relation_before,
        "room_polygon": [
            [point.x, point.y] for point in before.room_polygon_xy
        ],
        "subject_after": [
            subject_after.position.x,
            subject_after.position.y,
        ],
        "subject_before": [
            subject_before.position.x,
            subject_before.position.y,
        ],
        "subject_id": spec.subject_id,
    }
