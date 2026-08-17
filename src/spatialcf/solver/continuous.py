"""Certified continuous XY counterfactual solving by distance-bound closure."""

import math
import time
from dataclasses import dataclass
from itertools import pairwise

from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus
from spatialcf.domain.models import (
    InterventionSpec,
    ObjectView,
    Scene,
    SceneObject,
    Vec3,
)
from spatialcf.geometry.obb import obb_footprint
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.analytic_motion import (
    AnalyticMotionModel,
    CandidateProjectionError,
)
from spatialcf.solver.certified_constraints import CertifiedConstraintBuilder
from spatialcf.solver.certified_models import (
    SUPPORTED_DIRECTIONS,
    CertifiedSolverConfig,
    CertifiedSolveResult,
    OptimalityCertificate,
    expected_target_diff,
)
from spatialcf.solver.feasible import _configuration_obstacle
from spatialcf.solver.objective import ObjectiveBreakdown, score_candidate
from spatialcf.verification.verifier import VerificationResult, Verifier

_ATOMIC_GEOMETRY_TYPES = (Polygon, LineString, LinearRing, Point)
_STRICT_PRESERVATION_ERRORS = frozenset(
    {"target_pair_collateral_change", "non_target_relation_changed"}
)


def _atomic_components(region: BaseGeometry) -> tuple[BaseGeometry, ...]:
    """Return every non-empty polygon, line, or point component."""
    if region.is_empty:
        return ()
    if isinstance(region, _ATOMIC_GEOMETRY_TYPES):
        return (region,)
    return tuple(
        component for part in region.geoms for component in _atomic_components(part)
    )


@dataclass(frozen=True)
class _NearestPointCandidate:
    point: Point
    component: BaseGeometry
    distance: float


def _nearest_on_segment(
    origin: Point,
    start: tuple[float, float],
    end: tuple[float, float],
) -> Point:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared_span = dx * dx + dy * dy
    if squared_span == 0.0:
        return Point(start)
    fraction = ((origin.x - start[0]) * dx + (origin.y - start[1]) * dy) / squared_span
    fraction = min(1.0, max(0.0, fraction))
    return Point(start[0] + fraction * dx, start[1] + fraction * dy)


def _segment_candidates(
    line: LineString | LinearRing,
    origin: Point,
) -> tuple[Point, ...]:
    coordinates = tuple((float(row[0]), float(row[1])) for row in line.coords)
    return tuple(
        _nearest_on_segment(origin, start, end) for start, end in pairwise(coordinates)
    )


def _component_nearest_candidates(
    component: BaseGeometry,
    origin: Point,
) -> tuple[Point, ...]:
    if isinstance(component, Point):
        return (component,)
    if isinstance(component, Polygon):
        if component.covers(origin):
            return (origin,)
        candidates = list(_segment_candidates(component.exterior, origin))
        for ring in component.interiors:
            candidates.extend(_segment_candidates(ring, origin))
    elif isinstance(component, (LineString, LinearRing)):
        candidates = list(_segment_candidates(component, origin))
    else:
        raise TypeError(f"unsupported atomic geometry: {component.geom_type}")
    if not candidates:
        return ()
    distances = tuple(origin.distance(candidate) for candidate in candidates)
    minimum = min(distances)
    return tuple(
        candidate
        for candidate, distance in zip(candidates, distances, strict=True)
        if math.isclose(
            distance,
            minimum,
            rel_tol=0.0,
            abs_tol=max(math.ulp(distance), math.ulp(minimum)),
        )
    )


def _nearest_candidate_entries(
    region: BaseGeometry,
    origin: Point,
) -> tuple[_NearestPointCandidate, ...]:
    ranked = [
        _NearestPointCandidate(
            point=point,
            component=component,
            distance=origin.distance(point),
        )
        for component in sorted(_atomic_components(region), key=lambda item: item.wkb)
        for point in _component_nearest_candidates(component, origin)
    ]
    unique: list[_NearestPointCandidate] = []
    seen_coordinates: set[tuple[float, float]] = set()
    for candidate in sorted(
        ranked,
        key=lambda item: (
            item.distance,
            item.point.x,
            item.point.y,
            item.component.wkb,
        ),
    ):
        coordinates = (candidate.point.x, candidate.point.y)
        if coordinates in seen_coordinates:
            continue
        seen_coordinates.add(coordinates)
        unique.append(candidate)
    return tuple(unique)


def _nearest_candidates(
    region: BaseGeometry,
    origin: Point,
    numeric_tolerance: float = 1e-9,
) -> tuple[Point, ...]:
    """Return every deterministic tied nearest coordinate per component."""
    del numeric_tolerance
    return tuple(
        candidate.point for candidate in _nearest_candidate_entries(region, origin)
    )


def _interpolate(start: Point, end: Point, fraction: float) -> Point:
    return Point(
        start.x + fraction * (end.x - start.x),
        start.y + fraction * (end.y - start.y),
    )


def _polygon_component(
    region: BaseGeometry,
    point: Point,
    numeric_tolerance: float,
) -> Polygon | None:
    components = tuple(
        component
        for component in _atomic_components(region)
        if isinstance(component, Polygon)
    )
    exact = tuple(component for component in components if component.covers(point))
    if exact:
        return min(exact, key=lambda component: component.wkb)
    nearby = tuple(
        component
        for component in components
        if component.distance(point) <= numeric_tolerance
    )
    if not nearby:
        return None
    return min(
        nearby,
        key=lambda component: (component.distance(point), component.wkb),
    )


@dataclass(frozen=True)
class _VerifiedCandidate:
    distance: float
    point: Point
    position: Vec3
    score: ObjectiveBreakdown
    verification: VerificationResult

    @property
    def key(self) -> tuple[float, float, float]:
        return (self.distance, self.point.x, self.point.y)


class CertifiedSpatialCFSolver:
    """Close analytic inner/outer distance bounds around a strict solution."""

    def __init__(self, config: CertifiedSolverConfig | None = None) -> None:
        self.config = config if config is not None else CertifiedSolverConfig()

    @staticmethod
    def _failure(
        status: SolverStatus,
        evaluated_candidates: int,
        reason: str,
    ) -> CertifiedSolveResult:
        return CertifiedSolveResult(
            status=status,
            subject_position=None,
            score=None,
            quality=QualityTier.REJECTED,
            evaluated_candidates=evaluated_candidates,
            reason=reason,
            certificate=None,
        )

    @staticmethod
    def _object_geometry_values(obj: SceneObject) -> tuple[float, ...]:
        values = (
            obj.position.x,
            obj.position.y,
            obj.position.z,
            obj.rotation.x,
            obj.rotation.y,
            obj.rotation.z,
            obj.rotation.w,
            obj.obb.center.x,
            obj.obb.center.y,
            obj.obb.center.z,
            obj.obb.extent.x,
            obj.obb.extent.y,
            obj.obb.extent.z,
            obj.obb.rotation.x,
            obj.obb.rotation.y,
            obj.obb.rotation.z,
            obj.obb.rotation.w,
        )
        view_values = tuple(
            value
            for camera_id in sorted(obj.views)
            for value in (
                obj.views[camera_id].bbox.xmin,
                obj.views[camera_id].bbox.ymin,
                obj.views[camera_id].bbox.xmax,
                obj.views[camera_id].bbox.ymax,
                obj.views[camera_id].camera_depth,
                obj.views[camera_id].visible_fraction,
                obj.views[camera_id].image_area_fraction,
                obj.views[camera_id].truncated_fraction,
            )
        )
        return values + view_values

    @classmethod
    def _scene_geometry_is_finite(cls, scene: Scene) -> bool:
        values = [
            value for point in scene.room_polygon_xy for value in (point.x, point.y)
        ]
        for camera in scene.cameras:
            values.extend(camera.intrinsics)
            values.extend(camera.world_to_camera)
        for obj in scene.objects:
            values.extend(cls._object_geometry_values(obj))
        for obstacle in scene.collision_obstacles:
            obb = obstacle.obb
            values.extend(
                (
                    obstacle.clearance_m,
                    obb.center.x,
                    obb.center.y,
                    obb.center.z,
                    obb.extent.x,
                    obb.extent.y,
                    obb.extent.z,
                    obb.rotation.x,
                    obb.rotation.y,
                    obb.rotation.z,
                    obb.rotation.w,
                )
            )
        return all(math.isfinite(float(value)) for value in values)

    @staticmethod
    def _scene_graph_failure_reason(scene: Scene) -> str | None:
        object_ids = tuple(obj.object_id for obj in scene.objects)
        if len(set(object_ids)) != len(object_ids):
            return "duplicate_object_ids"

        camera_ids = tuple(camera.camera_id for camera in scene.cameras)
        if len(set(camera_ids)) != len(camera_ids):
            return "duplicate_camera_ids"

        object_id_set = set(object_ids)
        obstacle_ids = tuple(
            obstacle.obstacle_id for obstacle in scene.collision_obstacles
        )
        if len(set(obstacle_ids)) != len(obstacle_ids):
            return "duplicate_collision_obstacle_ids"
        if object_id_set.intersection(obstacle_ids):
            return "collision_obstacle_id_conflict"
        camera_id_set = set(camera_ids)
        if not scene.pinned_object_ids.issubset(object_id_set):
            return "pinned_object_missing"

        support_parents: dict[str, str | None] = {}
        for obj in sorted(scene.objects, key=lambda item: item.object_id):
            for camera_id, view in sorted(obj.views.items()):
                if camera_id not in camera_id_set:
                    return "view_camera_missing"
                if view.camera_id != camera_id:
                    return "view_camera_mismatch"
            support_id = obj.support_object_id
            if support_id is not None and support_id not in object_id_set:
                return "support_object_missing"
            support_parents[obj.object_id] = support_id

        for object_id in sorted(object_ids):
            trail: set[str] = set()
            current: str | None = object_id
            while current is not None:
                if current in trail:
                    return "invalid_support_graph"
                trail.add(current)
                current = support_parents[current]
        return None

    @staticmethod
    def _view_geometry_is_valid(view: ObjectView) -> bool:
        bbox = view.bbox
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
        center_x = (bbox.xmin + bbox.xmax) / 2.0
        center_y = (bbox.ymin + bbox.ymax) / 2.0
        return (
            all(
                math.isfinite(value)
                for value in (width, height, bbox.area, center_x, center_y)
            )
            and width > 0.0
            and height > 0.0
            and bbox.area > 0.0
            and view.camera_depth > 0.0
        )

    def _validate_scene(
        self,
        scene: Scene,
        spec: InterventionSpec,
    ) -> CertifiedSolveResult | None:
        graph_failure = self._scene_graph_failure_reason(scene)
        if graph_failure is not None:
            return self._failure(SolverStatus.INVALID_SCENE, 0, graph_failure)

        try:
            subject = scene.object_by_id(spec.subject_id)
        except KeyError:
            return self._failure(SolverStatus.INVALID_SCENE, 0, "unknown_subject")
        try:
            reference = scene.object_by_id(spec.reference_id)
        except KeyError:
            return self._failure(SolverStatus.INVALID_SCENE, 0, "unknown_reference")
        try:
            scene.camera_by_id(spec.camera_id)
        except KeyError:
            return self._failure(SolverStatus.INVALID_SCENE, 0, "unknown_camera")

        if not subject.request_eligible or not reference.request_eligible:
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "request_endpoint_ineligible",
            )
        if not subject.movable or spec.subject_id in scene.pinned_object_ids:
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "subject_not_movable",
            )
        if scene.children_by_support().get(spec.subject_id):
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "subject_has_supported_objects",
            )
        if not self._scene_geometry_is_finite(scene):
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "non_finite_geometry",
            )
        room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
        if not math.isfinite(room.area):
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "non_finite_geometry",
            )
        if not room.is_valid or room.is_empty or room.area <= 0.0:
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "invalid_room_geometry",
            )
        if any(
            extent <= 0.0
            for obj in scene.objects
            for extent in (obj.obb.extent.x, obj.obb.extent.y, obj.obb.extent.z)
        ) or any(
            extent <= 0.0
            for obstacle in scene.collision_obstacles
            for extent in (
                obstacle.obb.extent.x,
                obstacle.obb.extent.y,
                obstacle.obb.extent.z,
            )
        ):
            return self._failure(
                SolverStatus.INVALID_SCENE,
                0,
                "invalid_object_geometry",
            )

        query_endpoint_ids = {spec.subject_id, spec.reference_id}
        for obj in sorted(scene.objects, key=lambda item: item.object_id):
            for camera_id, view in sorted(obj.views.items()):
                if not self._view_geometry_is_valid(view):
                    reason = (
                        "invalid_query_view_geometry"
                        if obj.object_id in query_endpoint_ids
                        and camera_id == spec.camera_id
                        else "invalid_object_view_geometry"
                    )
                    return self._failure(
                        SolverStatus.INVALID_SCENE,
                        0,
                        reason,
                    )

        motion_model = AnalyticMotionModel()
        for camera_id in sorted(subject.views):
            try:
                motion_model.calibration(scene, subject.object_id, camera_id)
            except CandidateProjectionError:
                reason = (
                    "invalid_query_view_geometry"
                    if camera_id == spec.camera_id
                    else "invalid_subject_view_calibration"
                )
                return self._failure(
                    SolverStatus.INVALID_SCENE,
                    0,
                    reason,
                )

        source = RelationEngine().observe(
            scene,
            spec.subject_id,
            spec.reference_id,
            spec.relation_before,
            spec.camera_id,
        )
        if source.status is SolverStatus.NOT_VISIBLE:
            return self._failure(
                SolverStatus.NOT_VISIBLE,
                0,
                "source_not_visible",
            )
        if not source.satisfied:
            return self._failure(
                SolverStatus.AMBIGUOUS,
                0,
                "source_relation_not_satisfied",
            )
        return None

    @staticmethod
    def _deadline(config: CertifiedSolverConfig) -> float | None:
        if config.timeout_seconds is None:
            return None
        return time.monotonic() + config.timeout_seconds

    @staticmethod
    def _timed_out(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline

    @staticmethod
    def _evaluate_candidate(
        scene: Scene,
        spec: InterventionSpec,
        point: Point,
        origin: Point,
    ) -> tuple[_VerifiedCandidate | None, VerificationResult]:
        after = AnalyticMotionModel().with_object_xy(
            scene,
            spec.subject_id,
            point.x,
            point.y,
        )
        verifier = Verifier()
        verification = verifier.verify(scene, after, spec)
        if not (
            verification.status is SolverStatus.SUCCESS
            and verification.quality is QualityTier.PURE
            and verification.leakage_count == 0
            and verification.changed_relations == expected_target_diff(spec)
        ):
            return None, verification
        score = score_candidate(
            scene,
            after,
            spec,
            verification.leakage_count,
            verifier.engine,
        )
        return (
            _VerifiedCandidate(
                distance=origin.distance(point),
                point=point,
                position=after.object_by_id(spec.subject_id).position,
                score=score,
                verification=verification,
            ),
            verification,
        )

    @staticmethod
    def _is_strict_preservation_failure(
        verification: VerificationResult,
    ) -> bool:
        return (
            verification.status is not SolverStatus.SUCCESS
            and bool(verification.errors)
            and set(verification.errors).issubset(_STRICT_PRESERVATION_ERRORS)
        )

    @staticmethod
    def _is_target_boundary_failure(
        verification: VerificationResult,
    ) -> bool:
        return (
            verification.status is not SolverStatus.SUCCESS
            and verification.errors == ("target_not_satisfied",)
        )

    def _next_disk_segments(
        self,
        current: int,
        radial_geometry_error: float,
    ) -> int:
        """Skip disk resolutions that cannot leave a stable gap budget."""
        next_segments = current * 2
        stable_radial_budget = max(
            self.config.numeric_tolerance,
            (
                self.config.optimality_tolerance
                - 2.0 * self.config.numeric_tolerance
            )
            / 4.0,
        )
        while (
            next_segments < self.config.max_disk_segments
            and radial_geometry_error * (current / next_segments) ** 2
            > stable_radial_budget
        ):
            next_segments *= 2
        return next_segments

    def _strict_inward_candidate(
        self,
        scene: Scene,
        spec: InterventionSpec,
        boundary: Point,
        origin: Point,
        component: Polygon,
        deadline: float | None,
    ) -> tuple[_VerifiedCandidate | None, int, bool]:
        representative = component.representative_point()
        guard = max(
            self.config.numeric_tolerance,
            math.ulp(boundary.x),
            math.ulp(boundary.y),
        )
        span = boundary.distance(representative)
        if span == 0.0:
            return None, 0, False
        low = min(1.0, guard / span)
        high = 1.0
        evaluated = 0

        if self._timed_out(deadline):
            return None, evaluated, True
        evaluated += 1
        try:
            high_candidate, _ = self._evaluate_candidate(
                scene,
                spec,
                _interpolate(boundary, representative, high),
                origin,
            )
        except CandidateProjectionError:
            return None, evaluated, False
        if self._timed_out(deadline):
            return None, evaluated, True
        if high_candidate is None:
            return None, evaluated, False

        for _ in range(64):
            middle = (low + high) / 2.0
            if self._timed_out(deadline):
                return None, evaluated, True
            evaluated += 1
            try:
                candidate, _ = self._evaluate_candidate(
                    scene,
                    spec,
                    _interpolate(boundary, representative, middle),
                    origin,
                )
            except CandidateProjectionError:
                return None, evaluated, False
            if self._timed_out(deadline):
                return None, evaluated, True
            if candidate is not None:
                high = middle
            else:
                low = middle

        evaluated += 1
        try:
            candidate, _ = self._evaluate_candidate(
                scene,
                spec,
                _interpolate(boundary, representative, high),
                origin,
            )
        except CandidateProjectionError:
            return None, evaluated, False
        return candidate, evaluated, self._timed_out(deadline)

    def _verified_segment_boundary_candidate(
        self,
        scene: Scene,
        spec: InterventionSpec,
        boundary: Point,
        interior: Point,
        origin: Point,
        deadline: float | None,
    ) -> tuple[_VerifiedCandidate | None, int, bool]:
        """Find the first verified point between an outer and inner candidate."""
        low = 0.0
        high = 1.0
        evaluated = 0
        for _ in range(64):
            if self._timed_out(deadline):
                return None, evaluated, True
            middle = (low + high) / 2.0
            evaluated += 1
            try:
                candidate, _ = self._evaluate_candidate(
                    scene,
                    spec,
                    _interpolate(boundary, interior, middle),
                    origin,
                )
            except CandidateProjectionError:
                return None, evaluated, False
            if candidate is None:
                low = middle
            else:
                high = middle

        if self._timed_out(deadline):
            return None, evaluated, True
        evaluated += 1
        try:
            candidate, _ = self._evaluate_candidate(
                scene,
                spec,
                _interpolate(boundary, interior, high),
                origin,
            )
        except CandidateProjectionError:
            return None, evaluated, False
        return candidate, evaluated, self._timed_out(deadline)

    def _guarded_target_boundary_candidate(
        self,
        scene: Scene,
        spec: InterventionSpec,
        boundary: Point,
        interior: Point,
        origin: Point,
        deadline: float | None,
    ) -> tuple[_VerifiedCandidate | None, int, bool]:
        """Keep a replay guard without exceeding the optimality tolerance."""
        if self._timed_out(deadline):
            return None, 0, True
        span = boundary.distance(interior)
        if span == 0.0:
            return None, 0, False
        guarded_distance = min(
            span,
            max(
                self.config.numeric_tolerance,
                self.config.optimality_tolerance
                - 2.0 * self.config.numeric_tolerance,
            ),
        )
        point = _interpolate(boundary, interior, guarded_distance / span)
        try:
            candidate, _ = self._evaluate_candidate(
                scene,
                spec,
                point,
                origin,
            )
        except CandidateProjectionError:
            return None, 1, False
        if candidate is not None and not self._candidate_is_collision_free(
            scene,
            spec,
            point,
        ):
            candidate = None
        return candidate, 1, self._timed_out(deadline)

    def _candidate_is_collision_free(
        self,
        scene: Scene,
        spec: InterventionSpec,
        point: Point,
    ) -> bool:
        after = AnalyticMotionModel().with_object_xy(
            scene,
            spec.subject_id,
            point.x,
            point.y,
        )
        subject = after.object_by_id(spec.subject_id)
        subject_bottom = subject.obb.center.z - subject.obb.extent.z / 2.0
        subject_top = subject.obb.center.z + subject.obb.extent.z / 2.0
        subject_footprint = obb_footprint(subject.obb)
        for stationary in after.objects:
            if stationary.object_id in {
                spec.subject_id,
                subject.support_object_id,
            }:
                continue
            stationary_bottom = (
                stationary.obb.center.z - stationary.obb.extent.z / 2.0
            )
            stationary_top = (
                stationary.obb.center.z + stationary.obb.extent.z / 2.0
            )
            vertical_overlap = min(subject_top, stationary_top) - max(
                subject_bottom,
                stationary_bottom,
            )
            if (
                vertical_overlap > self.config.numeric_tolerance
                and subject_footprint.intersection(
                    obb_footprint(stationary.obb)
                ).area
                > self.config.numeric_tolerance
            ):
                return False
        return True

    def _direct_far_candidate(
        self,
        scene: Scene,
        spec: InterventionSpec,
        origin: Point,
    ) -> Point | None:
        """Return the exact unconstrained FAR-boundary projection when defined."""
        if spec.relation_after is not Relation.FAR:
            return None
        subject = scene.object_by_id(spec.subject_id)
        reference = scene.object_by_id(spec.reference_id)
        configuration = self._target_pair_configuration(subject, reference)
        distance = origin.distance(configuration)
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        contact = nearest_points(configuration, origin)[0]
        radius = RelationEngine.FAR_METERS + self.config.numeric_tolerance
        scale = radius / distance
        point = Point(
            contact.x + (origin.x - contact.x) * scale,
            contact.y + (origin.y - contact.y) * scale,
        )
        if not all(math.isfinite(value) for value in (point.x, point.y)):
            raise ArithmeticError("non-finite direct FAR candidate")
        return point

    @staticmethod
    def _target_pair_configuration(
        subject: SceneObject,
        reference: SceneObject,
    ) -> Polygon:
        relative_vertices = [
            (float(x) - subject.position.x, float(y) - subject.position.y)
            for x, y in tuple(obb_footprint(subject.obb).exterior.coords)[:-1]
        ]
        return _configuration_obstacle(
            obb_footprint(reference.obb),
            relative_vertices,
        )

    @staticmethod
    def _target_boundary_coefficients(
        scene: Scene,
        spec: InterventionSpec,
    ) -> tuple[float, float, float] | None:
        subject = scene.object_by_id(spec.subject_id)
        reference = scene.object_by_id(spec.reference_id)
        calibration = AnalyticMotionModel().calibration(
            scene,
            subject.object_id,
            spec.camera_id,
        )
        reference_view = reference.views.get(spec.camera_id)
        if reference_view is None:
            return None
        if spec.relation_after is Relation.RIGHT:
            camera = calibration.camera
            target = (
                reference_view.bbox.center_x
                + camera.width * RelationEngine.LEFT_RIGHT_FRACTION
            )
            fx, _, cx, _, _, _, _, _, _ = camera.intrinsics
            x_nx, x_ny, x_constant = calibration.camera_x_coefficients
            d_nx, d_ny, d_constant = calibration.depth_coefficients
            scale = cx + calibration.horizontal_residual - target
            return (
                fx * x_nx + scale * d_nx,
                fx * x_ny + scale * d_ny,
                fx * x_constant + scale * d_constant,
            )
        if spec.relation_after is Relation.BEHIND:
            d_nx, d_ny, d_constant = calibration.depth_coefficients
            target = reference_view.camera_depth + RelationEngine.FRONT_BEHIND_METERS
            return (
                d_nx,
                d_ny,
                d_constant + calibration.depth_residual - target,
            )
        return None

    def _preserved_far_boundary_candidates(
        self,
        scene: Scene,
        spec: InterventionSpec,
        approximate: Point,
    ) -> tuple[Point, ...]:
        """Intersect the exact target line with a locally active FAR arc."""
        labels = RelationEngine().pair_labels(
            scene,
            spec.subject_id,
            spec.reference_id,
            spec.camera_id,
        )
        coefficients = self._target_boundary_coefficients(scene, spec)
        if Relation.FAR not in labels or coefficients is None:
            return ()
        subject = scene.object_by_id(spec.subject_id)
        reference = scene.object_by_id(spec.reference_id)
        configuration = self._target_pair_configuration(subject, reference)
        contact = nearest_points(configuration, approximate)[0]
        nx, ny, constant = coefficients
        norm_squared = nx * nx + ny * ny
        if not math.isfinite(norm_squared) or norm_squared <= 0.0:
            raise ArithmeticError("invalid target boundary normal")
        signed = (nx * contact.x + ny * contact.y + constant) / norm_squared
        foot_x = contact.x - nx * signed
        foot_y = contact.y - ny * signed
        center_distance = abs(nx * contact.x + ny * contact.y + constant) / math.sqrt(
            norm_squared
        )
        radius = RelationEngine.FAR_METERS + self.config.numeric_tolerance
        if center_distance > radius:
            return ()
        offset = math.sqrt(max(0.0, radius * radius - center_distance**2))
        norm = math.sqrt(norm_squared)
        tangent_x = -ny / norm
        tangent_y = nx / norm
        points = (
            Point(
                foot_x - offset * tangent_x,
                foot_y - offset * tangent_y,
            ),
            Point(
                foot_x + offset * tangent_x,
                foot_y + offset * tangent_y,
            ),
        )
        if not all(
            math.isfinite(value) for point in points for value in (point.x, point.y)
        ):
            raise ArithmeticError("non-finite preserved FAR candidate")
        return tuple(sorted(points, key=lambda point: (point.x, point.y)))

    def solve(
        self,
        scene: Scene,
        spec: InterventionSpec,
    ) -> CertifiedSolveResult:
        """Return a strict candidate only after its distance bounds close."""
        deadline = self._deadline(self.config)
        if (
            spec.relation_before,
            spec.relation_after,
        ) not in SUPPORTED_DIRECTIONS:
            return self._failure(
                SolverStatus.UNSUPPORTED,
                0,
                "unsupported_direction",
            )
        if self._timed_out(deadline):
            return self._failure(SolverStatus.TIMEOUT, 0, "timeout")
        invalid = self._validate_scene(scene, spec)
        if invalid is not None:
            return invalid
        if self._timed_out(deadline):
            return self._failure(SolverStatus.TIMEOUT, 0, "timeout")

        subject = scene.object_by_id(spec.subject_id)
        origin = Point(subject.position.x, subject.position.y)
        best: _VerifiedCandidate | None = None
        evaluated = 0
        segments = self.config.initial_disk_segments

        try:
            direct_far = self._direct_far_candidate(scene, spec, origin)
            if direct_far is not None:
                best, _ = self._evaluate_candidate(
                    scene,
                    spec,
                    direct_far,
                    origin,
                )
                evaluated += 1
        except CandidateProjectionError:
            best = None
        except (ArithmeticError, GEOSException):
            return self._failure(
                SolverStatus.UNCERTIFIED,
                evaluated,
                "numeric_geometry_failure",
            )
        if self._timed_out(deadline):
            return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")

        while segments <= self.config.max_disk_segments:
            if self._timed_out(deadline):
                return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")
            try:
                bracket = CertifiedConstraintBuilder().build(
                    scene,
                    spec,
                    disk_segments=segments,
                    numeric_tolerance=self.config.numeric_tolerance,
                    target_interior_margin=self.config.target_interior_margin,
                )
            except (ArithmeticError, GEOSException):
                return self._failure(
                    SolverStatus.UNCERTIFIED,
                    evaluated,
                    "numeric_geometry_failure",
                )
            if self._timed_out(deadline):
                return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")
            if bracket.outer.is_empty:
                return self._failure(
                    SolverStatus.UNSATISFIABLE,
                    evaluated,
                    "empty_outer_region",
                )

            try:
                lower = origin.distance(bracket.outer)
                if not math.isfinite(lower):
                    raise ArithmeticError("non-finite lower distance bound")
                nearest_candidates = _nearest_candidate_entries(
                    bracket.inner,
                    origin,
                )
            except (ArithmeticError, GEOSException):
                return self._failure(
                    SolverStatus.UNCERTIFIED,
                    evaluated,
                    "numeric_geometry_failure",
                )
            verified_inner: list[_VerifiedCandidate] = []
            for nearest in nearest_candidates:
                point = nearest.point
                if self._timed_out(deadline):
                    return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")
                try:
                    if not all(math.isfinite(value) for value in (point.x, point.y)):
                        raise ArithmeticError("non-finite nearest candidate")
                    candidate, verification = self._evaluate_candidate(
                        scene, spec, point, origin
                    )
                except CandidateProjectionError:
                    candidate = None
                    verification = None
                except (ArithmeticError, GEOSException):
                    return self._failure(
                        SolverStatus.UNCERTIFIED,
                        evaluated,
                        "numeric_geometry_failure",
                    )
                evaluated += 1
                if self._timed_out(deadline):
                    return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")
                if (
                    candidate is None
                    and verification is not None
                    and self._is_strict_preservation_failure(verification)
                ):
                    component = (
                        nearest.component
                        if isinstance(nearest.component, Polygon)
                        else None
                    )
                    if component is not None:
                        try:
                            candidate, nudge_evaluated, timed_out = (
                                self._strict_inward_candidate(
                                    scene,
                                    spec,
                                    point,
                                    origin,
                                    component,
                                    deadline,
                                )
                            )
                        except (ArithmeticError, GEOSException):
                            return self._failure(
                                SolverStatus.UNCERTIFIED,
                                evaluated,
                                "numeric_geometry_failure",
                            )
                        evaluated += nudge_evaluated
                        if timed_out:
                            return self._failure(
                                SolverStatus.TIMEOUT,
                                evaluated,
                                "timeout",
                            )
                if candidate is not None and (best is None or candidate.key < best.key):
                    best = candidate
                if candidate is not None:
                    verified_inner.append(candidate)

            try:
                outer_nearest = tuple(
                    candidate
                    for candidate in _nearest_candidate_entries(
                        bracket.outer,
                        origin,
                    )
                    if candidate.distance <= lower + self.config.numeric_tolerance
                )
            except (ArithmeticError, GEOSException):
                return self._failure(
                    SolverStatus.UNCERTIFIED,
                    evaluated,
                    "numeric_geometry_failure",
                )
            for outer in outer_nearest:
                if self._timed_out(deadline):
                    return self._failure(
                        SolverStatus.TIMEOUT,
                        evaluated,
                        "timeout",
                    )
                try:
                    outer_candidate, outer_verification = self._evaluate_candidate(
                        scene,
                        spec,
                        outer.point,
                        origin,
                    )
                except CandidateProjectionError:
                    outer_candidate = None
                    outer_verification = None
                except (ArithmeticError, GEOSException):
                    return self._failure(
                        SolverStatus.UNCERTIFIED,
                        evaluated,
                        "numeric_geometry_failure",
                    )
                evaluated += 1
                if outer_candidate is not None:
                    if best is None or outer_candidate.key < best.key:
                        best = outer_candidate
                    continue
                try:
                    exact_candidates = self._preserved_far_boundary_candidates(
                        scene,
                        spec,
                        outer.point,
                    )
                    for point in exact_candidates:
                        exact_candidate, _ = self._evaluate_candidate(
                            scene,
                            spec,
                            point,
                            origin,
                        )
                        evaluated += 1
                        if exact_candidate is not None and (
                            best is None or exact_candidate.key < best.key
                        ):
                            best = exact_candidate
                except CandidateProjectionError:
                    pass
                except (ArithmeticError, GEOSException):
                    return self._failure(
                        SolverStatus.UNCERTIFIED,
                        evaluated,
                        "numeric_geometry_failure",
                    )
                if (
                    outer_verification is None
                    or not verified_inner
                ):
                    continue
                strict_preservation_failure = self._is_strict_preservation_failure(
                    outer_verification
                )
                target_boundary_failure = self._is_target_boundary_failure(
                    outer_verification
                )
                current_gap = (
                    best.distance - lower + 2 * self.config.numeric_tolerance
                    if best is not None
                    else math.inf
                )
                if not strict_preservation_failure and not (
                    target_boundary_failure
                    and current_gap <= self.config.optimality_tolerance
                ):
                    continue
                interior = min(
                    verified_inner,
                    key=lambda candidate: (
                        outer.point.distance(candidate.point),
                        candidate.key,
                    ),
                )
                try:
                    refinement = (
                        self._guarded_target_boundary_candidate
                        if target_boundary_failure
                        else self._verified_segment_boundary_candidate
                    )
                    refined, refinement_evaluated, timed_out = refinement(
                        scene,
                        spec,
                        outer.point,
                        interior.point,
                        origin,
                        deadline,
                    )
                except (ArithmeticError, GEOSException):
                    return self._failure(
                        SolverStatus.UNCERTIFIED,
                        evaluated,
                        "numeric_geometry_failure",
                    )
                evaluated += refinement_evaluated
                if timed_out:
                    return self._failure(
                        SolverStatus.TIMEOUT,
                        evaluated,
                        "timeout",
                    )
                if refined is not None and (best is None or refined.key < best.key):
                    best = refined

            if self._timed_out(deadline):
                return self._failure(SolverStatus.TIMEOUT, evaluated, "timeout")

            if best is not None:
                gap = best.distance - lower + 2 * self.config.numeric_tolerance
                if gap <= self.config.optimality_tolerance:
                    try:
                        outer_points = tuple(
                            candidate.point
                            for candidate in _nearest_candidate_entries(
                                bracket.outer,
                                origin,
                            )
                            if candidate.distance
                            <= lower + self.config.numeric_tolerance
                        )
                    except (ArithmeticError, GEOSException):
                        return self._failure(
                            SolverStatus.UNCERTIFIED,
                            evaluated,
                            "numeric_geometry_failure",
                        )
                    strict_failures: list[bool] = []
                    for point in outer_points:
                        if self._timed_out(deadline):
                            return self._failure(
                                SolverStatus.TIMEOUT,
                                evaluated,
                                "timeout",
                            )
                        try:
                            outer_candidate, outer_verification = (
                                self._evaluate_candidate(
                                    scene,
                                    spec,
                                    point,
                                    origin,
                                )
                            )
                        except CandidateProjectionError:
                            strict_failures.append(False)
                        except (ArithmeticError, GEOSException):
                            return self._failure(
                                SolverStatus.UNCERTIFIED,
                                evaluated,
                                "numeric_geometry_failure",
                            )
                        else:
                            strict_failures.append(
                                outer_candidate is None
                                and self._is_strict_preservation_failure(
                                    outer_verification
                                )
                            )
                        evaluated += 1
                        if self._timed_out(deadline):
                            return self._failure(
                                SolverStatus.TIMEOUT,
                                evaluated,
                                "timeout",
                            )
                    infimum_only = bool(strict_failures) and all(strict_failures)
                    certified_lower = min(lower, best.distance)
                    certificate = OptimalityCertificate.create(
                        distance_lower_bound=certified_lower,
                        distance_upper_bound=best.distance,
                        radial_geometry_error=bracket.radial_geometry_error,
                        numeric_error_bound=self.config.numeric_tolerance,
                        disk_segments=segments,
                        infimum_only=infimum_only,
                    )
                    return CertifiedSolveResult.success(
                        subject_position=best.position,
                        score=best.score,
                        quality=best.verification.quality,
                        evaluated_candidates=evaluated,
                        certificate=certificate,
                        tolerance=self.config.optimality_tolerance,
                        leakage_count=best.verification.leakage_count,
                        relation_diff=best.verification.changed_relations,
                        spec=spec,
                    )
            segments = (
                self._next_disk_segments(
                    segments,
                    bracket.radial_geometry_error,
                )
                if best is not None
                else segments * 2
            )

        return self._failure(
            SolverStatus.UNCERTIFIED,
            evaluated,
            (
                "optimality_gap_not_closed"
                if best is not None
                else "no_verified_inner_candidate"
            ),
        )
