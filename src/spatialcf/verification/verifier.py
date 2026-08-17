"""Acceptance checks for counterfactual scene interventions."""

import math
from dataclasses import dataclass

from shapely.geometry import Point, Polygon

from spatialcf.domain.enums import QualityTier, Relation, RelationAxis, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene, SceneObject
from spatialcf.geometry.obb import inside_room, obb_footprint, obbs_intersect_3d
from spatialcf.geometry.regions import subject_position_region_geometry
from spatialcf.relations.engine import RelationEngine


@dataclass(frozen=True)
class VerificationResult:
    """The independently recomputed outcome of one intervention."""

    status: SolverStatus
    quality: QualityTier
    leakage_count: int
    changed_relations: tuple[str, ...]
    errors: tuple[str, ...]
    relation_damage_count: int = 0
    relation_damage_items: tuple[str, ...] = ()


class Verifier:
    """Verify hard scene invariants, target relations, and graph leakage."""

    TRANSLATION_TOLERANCE = 1e-9

    def __init__(self, engine: RelationEngine | None = None) -> None:
        self.engine = engine or RelationEngine()

    def verify(
        self,
        before: Scene,
        after: Scene,
        spec: InterventionSpec,
    ) -> VerificationResult:
        """Accept or reject an intervention by recomputing every constraint."""
        return self._verify(
            before,
            after,
            spec,
            allow_relation_damage=False,
            runtime_collision_delegated_object_ids=frozenset(),
            runtime_pose_subject_object_id=None,
        )

    def verify_minimum_cost(
        self,
        before: Scene,
        after: Scene,
        spec: InterventionSpec,
    ) -> VerificationResult:
        """Verify hard constraints while measuring collateral relation damage."""
        return self._verify(
            before,
            after,
            spec,
            allow_relation_damage=True,
            runtime_collision_delegated_object_ids=frozenset(),
            runtime_pose_subject_object_id=None,
        )

    def verify_minimum_cost_with_runtime_collision_authority(
        self,
        before: Scene,
        after: Scene,
        spec: InterventionSpec,
        *,
        runtime_collision_delegated_object_ids: tuple[str, ...],
    ) -> VerificationResult:
        """Trust native physics only for a canonical, explicitly named roster."""

        delegated = runtime_collision_delegated_object_ids
        if type(delegated) is not tuple or any(
            type(object_id) is not str or not object_id for object_id in delegated
        ):
            raise TypeError("runtime collision authority must be exact object IDs")
        if delegated != tuple(sorted(set(delegated))):
            raise ValueError("runtime collision authority must be canonical")
        before_ids = {item.object_id for item in before.objects}
        after_ids = {item.object_id for item in after.objects}
        if not set(delegated) <= before_ids & after_ids:
            raise ValueError("runtime collision authority references an unknown object")
        return self._verify(
            before,
            after,
            spec,
            allow_relation_damage=True,
            runtime_collision_delegated_object_ids=frozenset(delegated),
            runtime_pose_subject_object_id=None,
        )

    def verify_minimum_cost_with_runtime_pose_authority(
        self,
        before: Scene,
        after: Scene,
        spec: InterventionSpec,
        *,
        runtime_collision_delegated_object_ids: tuple[str, ...],
        runtime_pose_subject_object_id: str,
    ) -> VerificationResult:
        """Verify final observed geometry for one bounded native subject pose."""

        delegated = runtime_collision_delegated_object_ids
        if type(delegated) is not tuple or any(
            type(object_id) is not str or not object_id for object_id in delegated
        ):
            raise TypeError("runtime collision authority must be exact object IDs")
        if delegated != tuple(sorted(set(delegated))):
            raise ValueError("runtime collision authority must be canonical")
        if (
            type(runtime_pose_subject_object_id) is not str
            or not runtime_pose_subject_object_id
            or runtime_pose_subject_object_id != spec.subject_id
        ):
            raise ValueError("runtime pose authority must name the intervention subject")
        before_ids = {item.object_id for item in before.objects}
        after_ids = {item.object_id for item in after.objects}
        if not set(delegated) <= before_ids & after_ids:
            raise ValueError("runtime collision authority references an unknown object")
        return self._verify(
            before,
            after,
            spec,
            allow_relation_damage=True,
            runtime_collision_delegated_object_ids=frozenset(delegated),
            runtime_pose_subject_object_id=runtime_pose_subject_object_id,
        )

    def _verify(
        self,
        before: Scene,
        after: Scene,
        spec: InterventionSpec,
        *,
        allow_relation_damage: bool,
        runtime_collision_delegated_object_ids: frozenset[str],
        runtime_pose_subject_object_id: str | None,
    ) -> VerificationResult:
        errors: list[str] = []
        for field in (
            "scene_id",
            "source",
            "coordinate_system",
            "generation_seed",
            "pinned_object_ids",
        ):
            if getattr(before, field) != getattr(after, field):
                errors.append(f"{field}_changed")
        if before.room_polygon_xy != after.room_polygon_xy:
            errors.append("room_changed")
        if before.cameras != after.cameras:
            errors.append("camera_changed")
        if before.collision_obstacles != after.collision_obstacles:
            errors.append("collision_obstacles_changed")
        if before.subject_position_regions != after.subject_position_regions:
            errors.append("subject_position_regions_changed")

        before_ids = {obj.object_id for obj in before.objects}
        after_ids = {obj.object_id for obj in after.objects}
        if before_ids != after_ids:
            return self._invalid("object_set_changed")

        try:
            subject_before = before.object_by_id(spec.subject_id)
            subject_after = after.object_by_id(spec.subject_id)
            reference_before = before.object_by_id(spec.reference_id)
            reference_after = after.object_by_id(spec.reference_id)
            before.camera_by_id(spec.camera_id)
            after.camera_by_id(spec.camera_id)
        except KeyError:
            return self._invalid("unknown_spec_target")

        if (
            not subject_before.request_eligible
            or not reference_before.request_eligible
        ):
            errors.append("request_endpoint_ineligible")
        if (
            subject_after.request_eligible
            != subject_before.request_eligible
        ):
            errors.append("subject_request_eligibility_changed")
        if (
            reference_after.request_eligible
            != reference_before.request_eligible
        ):
            errors.append("reference_request_eligibility_changed")
        if not subject_before.movable or spec.subject_id in before.pinned_object_ids:
            errors.append("subject_not_movable")
        if before.children_by_support().get(spec.subject_id):
            errors.append("subject_has_supported_objects")
        if subject_after.name != subject_before.name:
            errors.append("subject_name_changed")
        if subject_after.category != subject_before.category:
            errors.append("subject_category_changed")
        if subject_after.movable != subject_before.movable:
            errors.append("subject_movable_changed")
        runtime_pose_authorized = runtime_pose_subject_object_id == spec.subject_id
        if not runtime_pose_authorized:
            if (
                subject_after.position.z != subject_before.position.z
                or subject_after.obb.center.z != subject_before.obb.center.z
            ):
                errors.append("subject_z_changed")
            if (
                subject_after.rotation != subject_before.rotation
                or subject_after.obb.rotation != subject_before.obb.rotation
            ):
                errors.append("subject_rotation_changed")
            if subject_after.obb.extent != subject_before.obb.extent:
                errors.append("subject_extent_changed")
        if subject_after.support_object_id != subject_before.support_object_id:
            errors.append("support_assignment_changed")
        position_dx = subject_after.position.x - subject_before.position.x
        position_dy = subject_after.position.y - subject_before.position.y
        obb_dx = subject_after.obb.center.x - subject_before.obb.center.x
        obb_dy = subject_after.obb.center.y - subject_before.obb.center.y
        if not runtime_pose_authorized and not (
            math.isclose(
                position_dx,
                obb_dx,
                rel_tol=0.0,
                abs_tol=self.TRANSLATION_TOLERANCE,
            )
            and math.isclose(
                position_dy,
                obb_dy,
                rel_tol=0.0,
                abs_tol=self.TRANSLATION_TOLERANCE,
            )
        ):
            errors.append("subject_translation_mismatch")

        stationary_views_unchanged = True
        for obj in before.objects:
            if obj.object_id == spec.subject_id:
                continue
            after_object = after.object_by_id(obj.object_id)
            if obj.views != after_object.views:
                stationary_views_unchanged = False
            if not self._same_structure(obj, after_object):
                errors.append(f"other_object_changed:{obj.object_id}")

        room = Polygon([(point.x, point.y) for point in before.room_polygon_xy])
        if not inside_room(subject_after.obb, room):
            errors.append("outside_room")
        subject_position = Point(subject_after.position.x, subject_after.position.y)
        for position_region in before.subject_position_regions:
            if position_region.subject_object_id != spec.subject_id:
                continue
            if not subject_position_region_geometry(position_region).covers(
                subject_position
            ):
                errors.append(
                    f"outside_subject_position_region:{position_region.region_id}"
                )

        support_id = subject_before.support_object_id
        if support_id is not None:
            try:
                support = after.object_by_id(support_id)
            except KeyError:
                errors.append("support_missing")
            else:
                if not obb_footprint(support.obb).buffer(1e-6).covers(
                    obb_footprint(subject_after.obb)
                ):
                    errors.append("support_invalid")

        for obj in after.objects:
            if (
                obj.object_id not in {spec.subject_id, support_id}
                and obj.object_id not in runtime_collision_delegated_object_ids
                and obbs_intersect_3d(subject_after.obb, obj.obb)
            ):
                errors.append(f"collision:{obj.object_id}")
        for obstacle in after.collision_obstacles:
            if obbs_intersect_3d(subject_after.obb, obstacle.obb):
                errors.append(f"collision_obstacle:{obstacle.obstacle_id}")

        if errors:
            return VerificationResult(
                SolverStatus.INVALID_SCENE,
                QualityTier.REJECTED,
                0,
                (),
                tuple(sorted(errors)),
            )

        source = self.engine.observe(
            before,
            spec.subject_id,
            spec.reference_id,
            spec.relation_before,
            spec.camera_id,
        )
        target = self.engine.observe(
            after,
            spec.subject_id,
            spec.reference_id,
            spec.relation_after,
            spec.camera_id,
        )
        old_after = self.engine.observe(
            after,
            spec.subject_id,
            spec.reference_id,
            spec.relation_before,
            spec.camera_id,
        )
        if source.status is SolverStatus.NOT_VISIBLE or target.status is SolverStatus.NOT_VISIBLE:
            return VerificationResult(
                SolverStatus.NOT_VISIBLE,
                QualityTier.REJECTED,
                0,
                (),
                ("not_visible",),
            )
        if not source.satisfied:
            return VerificationResult(
                SolverStatus.AMBIGUOUS,
                QualityTier.REJECTED,
                0,
                (),
                ("source_relation_not_satisfied",),
            )
        if not target.satisfied or old_after.satisfied:
            return VerificationResult(
                SolverStatus.AMBIGUOUS,
                QualityTier.REJECTED,
                0,
                (),
                ("target_not_satisfied",),
            )

        forward_before = self.engine.pair_labels(
            before,
            spec.subject_id,
            spec.reference_id,
            spec.camera_id,
        )
        forward_after = self.engine.pair_labels(
            after,
            spec.subject_id,
            spec.reference_id,
            spec.camera_id,
        )
        reverse_before = self.engine.pair_labels(
            before,
            spec.reference_id,
            spec.subject_id,
            spec.camera_id,
        )
        reverse_after = self.engine.pair_labels(
            after,
            spec.reference_id,
            spec.subject_id,
            spec.camera_id,
        )

        if (
            reverse_before != self._converses(forward_before)
            or reverse_after != self._converses(forward_after)
        ):
            return self._rejected("reverse_relation_inconsistent")

        target_collateral_axes: set[RelationAxis] = set()
        for axis in RelationAxis:
            if axis is spec.relation_before.axis:
                continue
            if self._axis_labels(forward_before, axis) != self._axis_labels(
                forward_after, axis
            ):
                if not allow_relation_damage:
                    return self._rejected("target_pair_collateral_change")
                target_collateral_axes.add(axis)

        changed: list[str] = []
        leaked_pairs: set[tuple[str, str]] = set()
        target_pair = frozenset({spec.subject_id, spec.reference_id})
        objects = sorted(before.objects, key=lambda obj: obj.object_id)
        directed_pairs = (
            (
                (first, second)
                for first in objects
                for second in objects
                if first.object_id != second.object_id
                and spec.subject_id in {first.object_id, second.object_id}
            )
            if stationary_views_unchanged
            else (
                (first, second)
                for first in objects
                for second in objects
                if first.object_id != second.object_id
            )
        )
        for first, second in directed_pairs:
            old = self.engine.pair_labels(
                before, first.object_id, second.object_id, spec.camera_id
            )
            new = self.engine.pair_labels(
                after, first.object_id, second.object_id, spec.camera_id
            )
            for relation in sorted(old - new, key=lambda item: item.value):
                changed.append(f"-{first.object_id}:{relation.value}:{second.object_id}")
            for relation in sorted(new - old, key=lambda item: item.value):
                changed.append(f"+{first.object_id}:{relation.value}:{second.object_id}")
            if (
                old != new
                and frozenset({first.object_id, second.object_id}) != target_pair
            ):
                leaked_pairs.add(tuple(sorted((first.object_id, second.object_id))))

        leakage = len(leaked_pairs)
        if leakage and not allow_relation_damage:
            return VerificationResult(
                SolverStatus.UNSATISFIABLE,
                QualityTier.REJECTED,
                leakage,
                tuple(sorted(changed)),
                ("non_target_relation_changed",),
            )

        relation_damage_items: set[str] = {
            f"{min(spec.subject_id, spec.reference_id)}|"
            f"{max(spec.subject_id, spec.reference_id)}:{axis.value}"
            for axis in target_collateral_axes
        }
        unordered_pairs = (
            (
                (first, second)
                for index, first in enumerate(objects)
                for second in objects[index + 1:]
                if spec.subject_id in {first.object_id, second.object_id}
            )
            if stationary_views_unchanged
            else (
                (first, second)
                for index, first in enumerate(objects)
                for second in objects[index + 1:]
            )
        )
        for first, second in unordered_pairs:
            pair = frozenset({first.object_id, second.object_id})
            old = self.engine.pair_labels(
                before, first.object_id, second.object_id, spec.camera_id
            )
            new = self.engine.pair_labels(
                after, first.object_id, second.object_id, spec.camera_id
            )
            for axis in RelationAxis:
                if pair == target_pair and axis is spec.relation_before.axis:
                    continue
                if self._axis_labels(old, axis) != self._axis_labels(new, axis):
                    relation_damage_items.add(
                        f"{first.object_id}|{second.object_id}:{axis.value}"
                    )

        relation_damage = len(relation_damage_items)
        return VerificationResult(
            SolverStatus.SUCCESS,
            QualityTier.LOW_LEAKAGE if relation_damage else QualityTier.PURE,
            leakage,
            tuple(sorted(changed)),
            (),
            relation_damage,
            tuple(sorted(relation_damage_items)),
        )

    @staticmethod
    def _axis_labels(
        labels: frozenset[Relation],
        axis: RelationAxis,
    ) -> frozenset[Relation]:
        return frozenset(relation for relation in labels if relation.axis is axis)

    @staticmethod
    def _converses(labels: frozenset[Relation]) -> frozenset[Relation]:
        return frozenset(relation.converse for relation in labels)

    @staticmethod
    def _rejected(error: str) -> VerificationResult:
        return VerificationResult(
            SolverStatus.UNSATISFIABLE,
            QualityTier.REJECTED,
            0,
            (),
            (error,),
        )

    @staticmethod
    def _same_structure(before: SceneObject, after: SceneObject) -> bool:
        return before.model_copy(update={"views": after.views}) == after

    @staticmethod
    def _invalid(error: str) -> VerificationResult:
        return VerificationResult(
            SolverStatus.INVALID_SCENE,
            QualityTier.REJECTED,
            0,
            (),
            (error,),
        )
