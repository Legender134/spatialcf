import math
from dataclasses import dataclass

from spatialcf.domain.enums import Relation, SolverStatus
from spatialcf.domain.models import ObjectView, Scene
from spatialcf.geometry.obb import ground_gap


@dataclass(frozen=True)
class RelationResult:
    relation: Relation
    satisfied: bool
    margin: float
    status: SolverStatus


class RelationEngine:
    COMPARISON_TOLERANCE = 1e-9
    LEFT_RIGHT_FRACTION = 0.05
    FRONT_BEHIND_METERS = 0.20
    NEAR_METERS = 0.50
    FAR_METERS = 1.50
    MIN_VISIBLE_FRACTION = 0.20
    MIN_IMAGE_AREA_FRACTION = 0.0025
    MAX_TRUNCATED_FRACTION = 0.50

    def _view(
        self,
        scene: Scene,
        object_id: str,
        camera_id: str,
    ) -> ObjectView | None:
        obj = scene.object_by_id(object_id)
        return obj.views.get(camera_id)

    def _visible(self, view: ObjectView) -> bool:
        return (
            view.visible_fraction >= self.MIN_VISIBLE_FRACTION
            and view.image_area_fraction >= self.MIN_IMAGE_AREA_FRACTION
            and view.truncated_fraction <= self.MAX_TRUNCATED_FRACTION
        )

    def observe(
        self,
        scene: Scene,
        subject_id: str,
        reference_id: str,
        relation: Relation,
        camera_id: str,
    ) -> RelationResult:
        subject = scene.object_by_id(subject_id)
        reference = scene.object_by_id(reference_id)
        subject_view = self._view(scene, subject_id, camera_id)
        reference_view = self._view(scene, reference_id, camera_id)
        if (
            subject_view is None
            or reference_view is None
            or not self._visible(subject_view)
            or not self._visible(reference_view)
        ):
            return RelationResult(relation, False, 0.0, SolverStatus.NOT_VISIBLE)
        if relation in {Relation.LEFT, Relation.RIGHT, Relation.FRONT, Relation.BEHIND}:
            if relation in {Relation.LEFT, Relation.RIGHT}:
                threshold = scene.camera_by_id(camera_id).width * self.LEFT_RIGHT_FRACTION
                delta = reference_view.bbox.center_x - subject_view.bbox.center_x
                signed = delta if relation is Relation.LEFT else -delta
            else:
                delta = reference_view.camera_depth - subject_view.camera_depth
                signed = delta if relation is Relation.FRONT else -delta
                threshold = self.FRONT_BEHIND_METERS
            distance = abs(delta)
            at_threshold = math.isclose(
                distance,
                threshold,
                rel_tol=0.0,
                abs_tol=self.COMPARISON_TOLERANCE,
            )
            if distance < threshold and not at_threshold:
                return RelationResult(relation, False, distance - threshold, SolverStatus.AMBIGUOUS)
            return RelationResult(
                relation=relation,
                satisfied=signed >= threshold or (at_threshold and signed > 0.0),
                margin=0.0 if at_threshold else distance - threshold,
                status=SolverStatus.SUCCESS,
            )
        gap = ground_gap(subject.obb, reference.obb)
        if self.NEAR_METERS < gap < self.FAR_METERS:
            return RelationResult(relation, False, 0.0, SolverStatus.AMBIGUOUS)
        satisfied = gap <= self.NEAR_METERS if relation is Relation.NEAR else gap >= self.FAR_METERS
        margin = self.NEAR_METERS - gap if relation is Relation.NEAR else gap - self.FAR_METERS
        return RelationResult(relation, satisfied, abs(margin), SolverStatus.SUCCESS)

    def pair_labels(
        self,
        scene: Scene,
        first_id: str,
        second_id: str,
        camera_id: str,
    ) -> frozenset[Relation]:
        return frozenset(
            relation
            for relation in Relation
            if self.observe(scene, first_id, second_id, relation, camera_id).satisfied
        )

    def graph(self, scene: Scene, camera_id: str) -> dict[str, list[str]]:
        objects = sorted(scene.objects, key=lambda obj: obj.object_id)
        graph = {}
        for subject in objects:
            for reference in objects:
                if subject.object_id == reference.object_id:
                    continue
                graph[f"{subject.object_id}:{reference.object_id}"] = sorted(
                    relation.value
                    for relation in self.pair_labels(
                        scene, subject.object_id, reference.object_id, camera_id
                    )
                )
        return graph
