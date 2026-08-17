"""Normalized objective terms for verified spatial counterfactuals."""

import math
from dataclasses import dataclass

from shapely.geometry import Polygon

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.geometry.obb import (
    OBB_INTERSECTION_Z_OVERLAP_TOLERANCE,
    obb_footprint,
    obb_z_overlap_depth,
)
from spatialcf.relations.engine import RelationEngine


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """The normalized, weighted cost of one accepted intervention."""

    normalized_translation: float
    leakage: float
    visibility_change: float
    inverse_safety_margin: float
    total: float

    @property
    def relation_damage(self) -> float:
        """Normalized collateral relation cost (legacy name: leakage)."""
        return self.leakage


@dataclass(frozen=True)
class ObjectiveWeights:
    """Explicit weights for the fixed-camera minimum-total-cost objective."""

    translation: float = 1.0
    relation_damage: float = 5.0
    visibility_change: float = 2.0
    inverse_safety_margin: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.translation,
            self.relation_damage,
            self.visibility_change,
            self.inverse_safety_margin,
        )
        if any(type(value) is not float for value in values):
            raise ValueError("objective weights must be exact floats")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("objective weights must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one objective weight must be positive")


DEFAULT_OBJECTIVE_WEIGHTS = ObjectiveWeights()


def visibility_change(before: Scene, after: Scene, spec: InterventionSpec) -> float:
    """Return the largest normalized rendered-view change for the query pair."""
    def relative(old: float, new: float) -> float:
        return abs(old - new) / max(abs(old), 1e-6)

    deltas: list[float] = []
    for object_id in (spec.subject_id, spec.reference_id):
        old = before.object_by_id(object_id).views[spec.camera_id]
        new = after.object_by_id(object_id).views[spec.camera_id]
        deltas.extend((
            relative(old.visible_fraction, new.visible_fraction),
            relative(old.image_area_fraction, new.image_area_fraction),
            relative(old.truncated_fraction, new.truncated_fraction),
        ))
    return min(1.0, max(deltas))


def inverse_safety_margin(
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
            clearances.append(
                footprint.distance(obb_footprint(conservative_obb))
            )
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


def objective_score(
    normalized_translation: float,
    leakage: float,
    visibility_change: float,
    inverse_safety_margin: float,
    *,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> ObjectiveBreakdown:
    """Apply the approved normalized objective weights."""
    total = (
        weights.translation * normalized_translation
        + weights.relation_damage * leakage
        + weights.visibility_change * visibility_change
        + weights.inverse_safety_margin * inverse_safety_margin
    )
    return ObjectiveBreakdown(
        normalized_translation,
        leakage,
        visibility_change,
        inverse_safety_margin,
        total,
    )


def score_candidate(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
    leakage_count: int,
    engine: RelationEngine,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> ObjectiveBreakdown:
    """Compute objective terms using the original scene as the normalization base."""
    old = before.object_by_id(spec.subject_id).position
    new = after.object_by_id(spec.subject_id).position
    xs = [point.x for point in before.room_polygon_xy]
    ys = [point.y for point in before.room_polygon_xy]
    room_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    non_target_pair_count = max(
        1,
        len(before.objects) * (len(before.objects) - 1) // 2 - 1,
    )
    return objective_score(
        normalized_translation=math.hypot(new.x - old.x, new.y - old.y)
        / max(room_diagonal, 1e-9),
        leakage=min(1.0, leakage_count / non_target_pair_count),
        visibility_change=visibility_change(before, after, spec),
        inverse_safety_margin=inverse_safety_margin(after, spec, engine),
        weights=weights,
    )


def score_minimum_cost_candidate(
    before: Scene,
    after: Scene,
    spec: InterventionSpec,
    relation_damage_count: int,
    engine: RelationEngine,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> ObjectiveBreakdown:
    """Score a hard-valid candidate using unordered pair-axis damage.

    The requested target axis is excluded from the denominator because changing
    it is the intervention itself. Reciprocal directed labels are represented by
    one unordered pair-axis slot and therefore cannot be double-counted.
    """
    # Only the subject moves in the canonical problem, so normalizing by every
    # stationary-stationary pair would make identical damage artificially cheap
    # in larger scenes.  There are three axes for each subject/other pair, minus
    # the requested target axis whose change is mandatory rather than damage.
    soft_relation_axis_count = max(1, 3 * (len(before.objects) - 1) - 1)
    old = before.object_by_id(spec.subject_id).position
    new = after.object_by_id(spec.subject_id).position
    xs = [point.x for point in before.room_polygon_xy]
    ys = [point.y for point in before.room_polygon_xy]
    room_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return objective_score(
        normalized_translation=math.hypot(new.x - old.x, new.y - old.y)
        / max(room_diagonal, 1e-9),
        leakage=min(1.0, relation_damage_count / soft_relation_axis_count),
        visibility_change=visibility_change(before, after, spec),
        inverse_safety_margin=inverse_safety_margin(after, spec, engine),
        weights=weights,
    )
