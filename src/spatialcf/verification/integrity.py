"""Shared canonical serializers for published pair evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel
from shapely.geometry import Polygon, mapping

from spatialcf.core.feasibility import FeasibleRegionBuilder
from spatialcf.domain.request import InterventionSpec, Relation
from spatialcf.domain.scene import Scene
from spatialcf.geometry.obb import (
    OBB_INTERSECTION_Z_OVERLAP_TOLERANCE,
    obb_footprint,
    obb_z_overlap_depth,
)
from spatialcf.relations.engine import RelationEngine

ObjectiveValues = tuple[float, float, float, float, float]

_COMPETITION_NATIVE_OBSERVATION_DOMAIN = (
    b"spatialcf.competition-native-observation.v2.9\0"
)


def competition_legacy_sha256(value: BaseModel) -> str:
    """Hash one legacy model under the frozen finite-number convention."""

    if not isinstance(value, BaseModel):
        raise TypeError("competition legacy digest requires a Pydantic model")
    payload = json.dumps(
        _stable_competition_legacy_value(value.model_dump(mode="json")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_competition_legacy_value(value: object) -> object:
    if isinstance(value, Enum):
        return _stable_competition_legacy_value(value.value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("competition legacy digest requires finite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("competition legacy digest requires string mapping keys")
        return {
            key: _stable_competition_legacy_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_competition_legacy_value(item) for item in value]
    if isinstance(value, AbstractSet):
        normalized = [_stable_competition_legacy_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    raise TypeError(
        f"unsupported competition legacy digest value {type(value).__name__!r}"
    )


def competition_native_observation_payload_sha256(
    *,
    scene: BaseModel,
    rgb_png: bytes,
    depth_npy: bytes,
    instance_png: bytes,
    pointcloud_ply: bytes,
    instance_pixel_counts: Mapping[str, int],
    is_scene_at_rest: bool,
) -> str:
    """Hash a shared observation payload without importing an adapter type."""

    counts = dict(instance_pixel_counts)
    if any(
        type(key) is not str or type(value) is not int for key, value in counts.items()
    ):
        raise TypeError("native observation pixel counts must be exact")
    assets = {
        "depth_npy_sha256": hashlib.sha256(depth_npy).hexdigest(),
        "instance_png_sha256": hashlib.sha256(instance_png).hexdigest(),
        "pointcloud_ply_sha256": hashlib.sha256(pointcloud_ply).hexdigest(),
        "rgb_png_sha256": hashlib.sha256(rgb_png).hexdigest(),
    }
    payload = json.dumps(
        {
            "assets": assets,
            "instance_pixel_counts": dict(sorted(counts.items())),
            "is_scene_at_rest": is_scene_at_rest,
            "scene_sha256": competition_legacy_sha256(scene),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(_COMPETITION_NATIVE_OBSERVATION_DOMAIN + payload).hexdigest()


def canonical_value(value: Any) -> Any:
    """Convert model values into a deterministic JSON-compatible tree."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in sorted(value.items())}
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
    return (json.dumps(canonical_value(value), **options) + "\n").encode("utf-8")


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
        "feasible_region": mapping(FeasibleRegionBuilder().build(before, spec)),
        "movement_path": [
            [subject_before.position.x, subject_before.position.y],
            [subject_after.position.x, subject_after.position.y],
        ],
        "objects": [
            {
                "center": [obj.position.x, obj.position.y],
                "object_id": obj.object_id,
                "polygon": [
                    list(point) for point in obb_footprint(obj.obb).exterior.coords
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
        "room_polygon": [[point.x, point.y] for point in before.room_polygon_xy],
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


def validate_generation_budget(
    requested_pairs: int | None,
    attempt_limit: int | None,
    *,
    attempted_requests: int | None = None,
) -> Literal["accepted_pairs", "attempt_prefix"]:
    """Validate the mutually exclusive authenticated selection modes."""
    for name, value in (
        ("requested_pairs/limit", requested_pairs),
        ("attempt_limit", attempt_limit),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be null or a positive exact integer")
    if (requested_pairs is None) == (attempt_limit is None):
        raise ValueError(
            "exactly one of requested_pairs/limit and attempt_limit is required"
        )
    if attempted_requests is not None and (
        type(attempted_requests) is not int or attempted_requests < 0
    ):
        raise ValueError("attempted_requests must be a non-negative exact integer")
    if (
        attempt_limit is not None
        and attempted_requests is not None
        and attempted_requests != attempt_limit
    ):
        raise ValueError("attempt_limit requires a complete exact attempt prefix")
    return "attempt_prefix" if attempt_limit is not None else "accepted_pairs"


def source_corpus_digest(scenes: tuple[Scene, ...] | list[Scene]) -> str:
    """Hash a sorted canonical scene corpus without invoking a generator."""
    lines = []
    for scene in sorted(scenes, key=lambda item: item.scene_id):
        payload = canonical_json_bytes(scene.model_dump(mode="json"), pretty=True)
        lines.append(f"{scene.scene_id}  {hashlib.sha256(payload).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def calculate_visibility_change(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
) -> float:
    """Return the largest normalized rendered-view change for the query pair."""

    def relative(old: float, new: float) -> float:
        return abs(old - new) / max(abs(old), 1e-6)

    deltas: list[float] = []
    for object_id in (spec.subject_id, spec.reference_id):
        old = before.object_by_id(object_id).views[spec.camera_id]
        new = after.object_by_id(object_id).views[spec.camera_id]
        deltas.extend(
            (
                relative(old.visible_fraction, new.visible_fraction),
                relative(old.image_area_fraction, new.image_area_fraction),
                relative(old.truncated_fraction, new.truncated_fraction),
            )
        )
    return min(1.0, max(deltas))


def calculate_inverse_safety_margin(
    scene: Scene,
    spec: InterventionSpec,
    engine: RelationEngine,
) -> float:
    """Penalize candidates with limited geometric or target-relation slack."""
    subject = scene.object_by_id(spec.subject_id)
    footprint = obb_footprint(subject.obb)
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    clearances = [footprint.distance(room.boundary)]
    clearances.extend(
        footprint.distance(obb_footprint(obj.obb))
        for obj in scene.objects
        if obj.object_id not in {spec.subject_id, subject.support_object_id}
        and obb_z_overlap_depth(subject.obb, obj.obb)
        > OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
    )
    for obstacle in scene.collision_obstacles:
        conservative_obb = obstacle.conservative_obb()
        if (
            obb_z_overlap_depth(subject.obb, conservative_obb)
            > OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
        ):
            clearances.append(footprint.distance(obb_footprint(conservative_obb)))
    xs = [point.x for point in scene.room_polygon_xy]
    ys = [point.y for point in scene.room_polygon_xy]
    room_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    normalized_clearance = min(clearances) / max(0.02 * room_diagonal, 0.10)

    target = engine.observe(
        scene,
        spec.subject_id,
        spec.reference_id,
        spec.relation_after,
        spec.camera_id,
    )
    camera = scene.camera_by_id(spec.camera_id)
    relation_scale = {
        Relation.LEFT: camera.width * engine.LEFT_RIGHT_FRACTION,
        Relation.RIGHT: camera.width * engine.LEFT_RIGHT_FRACTION,
        Relation.FRONT: engine.FRONT_BEHIND_METERS,
        Relation.BEHIND: engine.FRONT_BEHIND_METERS,
        Relation.NEAR: engine.NEAR_METERS,
        Relation.FAR: engine.FAR_METERS,
    }[spec.relation_after]
    normalized_relation_margin = target.margin / max(relation_scale, 1e-9)
    safety = min(normalized_clearance, normalized_relation_margin)
    return 1.0 / (1.0 + max(0.0, safety))


def calculate_weighted_objective(
    normalized_translation: float,
    leakage: float,
    visibility_change: float,
    inverse_safety_margin: float,
    *,
    translation_weight: float,
    relation_damage_weight: float,
    visibility_change_weight: float,
    inverse_safety_margin_weight: float,
) -> ObjectiveValues:
    """Apply explicit normalized objective weight scalars."""
    total = (
        translation_weight * normalized_translation
        + relation_damage_weight * leakage
        + visibility_change_weight * visibility_change
        + inverse_safety_margin_weight * inverse_safety_margin
    )
    return (
        normalized_translation,
        leakage,
        visibility_change,
        inverse_safety_margin,
        total,
    )


def calculate_candidate_objective(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
    leakage_count: int,
    engine: RelationEngine,
    *,
    translation_weight: float,
    relation_damage_weight: float,
    visibility_change_weight: float,
    inverse_safety_margin_weight: float,
) -> ObjectiveValues:
    """Compute objective terms using the original scene as normalization base."""
    old = before.object_by_id(spec.subject_id).position
    new = after.object_by_id(spec.subject_id).position
    xs = [point.x for point in before.room_polygon_xy]
    ys = [point.y for point in before.room_polygon_xy]
    room_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    non_target_pair_count = max(
        1,
        len(before.objects) * (len(before.objects) - 1) // 2 - 1,
    )
    return calculate_weighted_objective(
        normalized_translation=math.hypot(new.x - old.x, new.y - old.y)
        / max(room_diagonal, 1e-9),
        leakage=min(1.0, leakage_count / non_target_pair_count),
        visibility_change=calculate_visibility_change(before, after, spec),
        inverse_safety_margin=calculate_inverse_safety_margin(after, spec, engine),
        translation_weight=translation_weight,
        relation_damage_weight=relation_damage_weight,
        visibility_change_weight=visibility_change_weight,
        inverse_safety_margin_weight=inverse_safety_margin_weight,
    )


def calculate_minimum_cost_candidate_objective(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
    relation_damage_count: int,
    engine: RelationEngine,
    *,
    translation_weight: float,
    relation_damage_weight: float,
    visibility_change_weight: float,
    inverse_safety_margin_weight: float,
) -> ObjectiveValues:
    """Score a hard-valid candidate using unordered pair-axis damage."""
    soft_relation_axis_count = max(1, 3 * (len(before.objects) - 1) - 1)
    old = before.object_by_id(spec.subject_id).position
    new = after.object_by_id(spec.subject_id).position
    xs = [point.x for point in before.room_polygon_xy]
    ys = [point.y for point in before.room_polygon_xy]
    room_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return calculate_weighted_objective(
        normalized_translation=math.hypot(new.x - old.x, new.y - old.y)
        / max(room_diagonal, 1e-9),
        leakage=min(1.0, relation_damage_count / soft_relation_axis_count),
        visibility_change=calculate_visibility_change(before, after, spec),
        inverse_safety_margin=calculate_inverse_safety_margin(after, spec, engine),
        translation_weight=translation_weight,
        relation_damage_weight=relation_damage_weight,
        visibility_change_weight=visibility_change_weight,
        inverse_safety_margin_weight=inverse_safety_margin_weight,
    )
