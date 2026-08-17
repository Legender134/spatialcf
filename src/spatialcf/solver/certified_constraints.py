"""Complete conservative constraints for certified analytic XY motion."""

import math

import shapely
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from spatialcf.domain.enums import Relation, RelationAxis
from spatialcf.domain.models import InterventionSpec, ObjectView, Scene, SceneObject
from spatialcf.geometry.obb import (
    OBB_INTERSECTION_Z_OVERLAP_TOLERANCE,
    obb_footprint,
    obb_z_overlap_depth,
)
from spatialcf.geometry.regions import subject_position_region_geometry
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.analytic_motion import AnalyticMotionModel, ProjectionCalibration
from spatialcf.solver.certified_geometry import (
    complement_bracket,
    far_bracket,
    intersect_brackets,
    near_bracket,
    union_brackets,
)
from spatialcf.solver.certified_models import (
    CertifiedGeometryError,
    ConstraintBracketSummary,
    ConstraintBuildDiagnostics,
    ConstraintDiagnosticStep,
    ConstraintRegionSummary,
    FeasibleRegionBracket,
)
from spatialcf.solver.feasible import (
    _configuration_obstacle,
    _relative_vertices,
    clip_half_plane,
)

_AXIS_RELATIONS = {
    RelationAxis.HORIZONTAL: (Relation.LEFT, Relation.RIGHT),
    RelationAxis.DEPTH: (Relation.FRONT, Relation.BEHIND),
    RelationAxis.DISTANCE: (Relation.NEAR, Relation.FAR),
}


def _normalize_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Return deterministic valid finite geometry without dropping dimensions."""
    coordinates = shapely.get_coordinates(geometry)
    if any(not math.isfinite(float(value)) for row in coordinates for value in row):
        raise ArithmeticError("GEOS overlay produced non-finite geometry")
    if not geometry.is_valid:
        raise ArithmeticError("GEOS overlay produced invalid geometry")
    geometry = shapely.normalize(geometry)
    coordinates = shapely.get_coordinates(geometry)
    if any(not math.isfinite(float(value)) for row in coordinates for value in row):
        raise ArithmeticError("GEOS overlay produced non-finite geometry")
    if not geometry.is_valid:
        raise ArithmeticError("GEOS normalization produced invalid geometry")
    return geometry


def _make_bracket(
    *,
    inner: BaseGeometry,
    outer: BaseGeometry,
    radial_geometry_error: float,
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    return FeasibleRegionBracket.create(
        inner=_normalize_geometry(inner),
        outer=_normalize_geometry(outer),
        radial_geometry_error=radial_geometry_error,
        disk_segments=disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _exact_bracket(
    geometry: BaseGeometry,
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    geometry = _normalize_geometry(geometry)
    return _make_bracket(
        inner=geometry,
        outer=geometry,
        radial_geometry_error=0.0,
        disk_segments=disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _empty_bracket(
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    return _exact_bracket(Polygon(), disk_segments, numeric_tolerance)


def _intersect(
    left: FeasibleRegionBracket,
    right: FeasibleRegionBracket,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    result = intersect_brackets(left, right)
    return _make_bracket(
        inner=result.inner,
        outer=result.outer,
        radial_geometry_error=result.radial_geometry_error,
        disk_segments=result.disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _union(
    left: FeasibleRegionBracket,
    right: FeasibleRegionBracket,
    universe: BaseGeometry,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    result = union_brackets(left, right, universe)
    return _make_bracket(
        inner=result.inner,
        outer=result.outer,
        radial_geometry_error=result.radial_geometry_error,
        disk_segments=result.disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _complement(
    value: FeasibleRegionBracket,
    universe: BaseGeometry,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    result = complement_bracket(value, universe)
    return _make_bracket(
        inner=result.inner,
        outer=result.outer,
        radial_geometry_error=result.radial_geometry_error,
        disk_segments=result.disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _center_locus(
    container: BaseGeometry,
    relative_vertices: list[tuple[float, float]],
) -> BaseGeometry:
    """Intersect translated containers while retaining line and point results."""
    if not relative_vertices:
        return _normalize_geometry(container)
    first_x, first_y = relative_vertices[0]
    locus = _normalize_geometry(
        translate(container, xoff=-first_x, yoff=-first_y)
    )
    for x, y in relative_vertices[1:]:
        locus = _normalize_geometry(
            locus.intersection(translate(container, xoff=-x, yoff=-y))
        )
        if locus.is_empty:
            break
    return locus


def _half_plane_region(
    universe: BaseGeometry,
    nx: float,
    ny: float,
    constant: float,
    *,
    keep_greater: bool,
) -> BaseGeometry:
    """Intersect ``universe`` with one inclusive affine half-plane."""
    if universe.is_empty:
        return _normalize_geometry(universe)
    if not all(math.isfinite(value) for value in (nx, ny, constant)):
        raise CertifiedGeometryError("half-plane coefficients must be finite")
    min_x, min_y, max_x, max_y = universe.bounds
    if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
        raise ArithmeticError("half-plane universe has non-finite bounds")
    corners = (
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    )
    values = tuple(nx * x + ny * y + constant for x, y in corners)
    oriented = values if keep_greater else tuple(-value for value in values)
    if min(oriented) >= 0.0:
        return _normalize_geometry(universe)
    if max(oriented) < 0.0:
        return Polygon()
    span = max(max_x - min_x, max_y - min_y, 1.0)
    mask_box = box(min_x - span, min_y - span, max_x + span, max_y + span)
    mask = clip_half_plane(mask_box, nx, ny, -constant, keep_greater)
    return _normalize_geometry(universe.intersection(mask))


def _linear_bracket(
    universe: BaseGeometry,
    inner_coefficients: tuple[float, float, float],
    outer_coefficients: tuple[float, float, float],
    *,
    keep_greater: bool,
    include_inner_in_outer: bool = False,
    disk_segments: int,
    numeric_tolerance: float,
) -> FeasibleRegionBracket:
    inner = _half_plane_region(
        universe, *inner_coefficients, keep_greater=keep_greater
    )
    outer = _half_plane_region(
        universe, *outer_coefficients, keep_greater=keep_greater
    )
    if include_inner_in_outer:
        outer = _normalize_geometry(outer.union(inner))
    return _make_bracket(
        inner=inner,
        outer=outer,
        radial_geometry_error=0.0,
        disk_segments=disk_segments,
        numeric_tolerance=numeric_tolerance,
    )


def _guard_inner_coefficients(
    coefficients: tuple[float, float, float],
    *,
    keep_greater: bool,
    margin: float,
) -> tuple[float, float, float]:
    """Move one half-plane boundary ``margin`` metres into its kept side."""
    if margin == 0.0:
        return coefficients
    nx, ny, constant = coefficients
    norm = math.hypot(nx, ny)
    if not math.isfinite(norm) or norm == 0.0:
        raise CertifiedGeometryError(
            "target half-plane must have a finite non-zero candidate-space normal"
        )
    guarded_constant = constant + (-norm * margin if keep_greater else norm * margin)
    return nx, ny, guarded_constant


def _exclude_interior(geometry: BaseGeometry, obstacle: BaseGeometry) -> BaseGeometry:
    """Remove positive-area collision states but restore allowed contact."""
    return _normalize_geometry(
        geometry.difference(obstacle).union(geometry.intersection(obstacle.boundary))
    )


def _component_count(geometry: BaseGeometry) -> int:
    if geometry.is_empty:
        return 0
    parts = getattr(geometry, "geoms", None)
    if parts is None:
        return 1
    return sum(_component_count(part) for part in parts)


def _region_summary(geometry: BaseGeometry) -> ConstraintRegionSummary:
    is_empty = bool(geometry.is_empty)
    return ConstraintRegionSummary(
        area_m2=float(geometry.area),
        component_count=_component_count(geometry),
        dimension=-1 if is_empty else int(shapely.get_dimensions(geometry)),
        is_empty=is_empty,
    )


def _bracket_summary(region: FeasibleRegionBracket) -> ConstraintBracketSummary:
    return ConstraintBracketSummary(
        inner=_region_summary(region.inner),
        outer=_region_summary(region.outer),
    )


def _record_diagnostic_step(
    steps: list[ConstraintDiagnosticStep] | None,
    *,
    stage: str,
    constraint_id: str,
    applied: bool,
    outer_was_empty: bool,
    region: FeasibleRegionBracket,
) -> None:
    if steps is None:
        return
    steps.append(
        ConstraintDiagnosticStep(
            index=len(steps),
            stage=stage,
            constraint_id=constraint_id,
            applied=applied,
            emptied_outer=(
                applied and not outer_was_empty and bool(region.outer.is_empty)
            ),
            inner=_region_summary(region.inner),
            outer=_region_summary(region.outer),
        )
    )


class CertifiedConstraintBuilder:
    """Compile the target and complete original relation graph into a bracket."""

    def __init__(
        self,
        *,
        relation_engine: RelationEngine | None = None,
        motion_model: AnalyticMotionModel | None = None,
    ) -> None:
        self.engine = relation_engine or RelationEngine()
        self.motion_model = motion_model or AnalyticMotionModel()

    def build(
        self,
        scene: Scene,
        spec: InterventionSpec,
        disk_segments: int,
        numeric_tolerance: float,
        target_interior_margin: float = 0.0,
    ) -> FeasibleRegionBracket:
        return self._build(
            scene,
            spec,
            disk_segments,
            numeric_tolerance,
            target_interior_margin,
            diagnostic_steps=None,
            diagnostic_regions=None,
        )

    def build_with_diagnostics(
        self,
        scene: Scene,
        spec: InterventionSpec,
        disk_segments: int,
        numeric_tolerance: float,
        target_interior_margin: float = 0.0,
    ) -> tuple[FeasibleRegionBracket, ConstraintBuildDiagnostics]:
        """Build the exact bracket and report each canonical constraint prefix.

        Diagnostics observe the same private construction path used by
        :meth:`build`; they neither add nor remove a feasibility constraint.
        """
        steps: list[ConstraintDiagnosticStep] = []
        regions: dict[str, FeasibleRegionBracket] = {}
        bracket = self._build(
            scene,
            spec,
            disk_segments,
            numeric_tolerance,
            target_interior_margin,
            diagnostic_steps=steps,
            diagnostic_regions=regions,
        )
        first_empty = next(
            (step.index for step in steps if step.emptied_outer),
            None,
        )
        structural = regions["structural"]
        target_with_structural = _intersect(
            structural,
            regions["target_constraint"],
            numeric_tolerance,
        )
        if structural.outer.is_empty:
            outer_empty_group = "structural"
        elif target_with_structural.outer.is_empty:
            outer_empty_group = "target_relation"
        elif bracket.outer.is_empty:
            outer_empty_group = "preserved_relation"
        else:
            outer_empty_group = None
        return bracket, ConstraintBuildDiagnostics(
            steps=tuple(steps),
            first_outer_empty_step_index=first_empty,
            structural=_bracket_summary(structural),
            target_with_structural=_bracket_summary(target_with_structural),
            final=_bracket_summary(bracket),
            outer_empty_group=outer_empty_group,
        )

    def _build(
        self,
        scene: Scene,
        spec: InterventionSpec,
        disk_segments: int,
        numeric_tolerance: float,
        target_interior_margin: float,
        *,
        diagnostic_steps: list[ConstraintDiagnosticStep] | None,
        diagnostic_regions: dict[str, FeasibleRegionBracket] | None,
    ) -> FeasibleRegionBracket:
        self._validate_inputs(
            scene,
            spec,
            disk_segments,
            numeric_tolerance,
            target_interior_margin,
        )
        subject = scene.object_by_id(spec.subject_id)
        relative_vertices = _relative_vertices(subject)
        room = self._room(scene)
        room_locus = _center_locus(room, relative_vertices)
        room_is_convex = room.equals(room.convex_hull)
        region = _make_bracket(
            inner=room_locus if room_is_convex else Polygon(),
            outer=room_locus,
            radial_geometry_error=0.0,
            disk_segments=disk_segments,
            numeric_tolerance=numeric_tolerance,
        )
        universe = region.outer
        _record_diagnostic_step(
            diagnostic_steps,
            stage="room_boundary",
            constraint_id="room",
            applied=True,
            outer_was_empty=False,
            region=region,
        )
        for position_region in sorted(
            scene.subject_position_regions,
            key=lambda item: item.region_id,
        ):
            if position_region.subject_object_id != subject.object_id:
                continue
            allowed = subject_position_region_geometry(position_region)
            outer_was_empty = bool(region.outer.is_empty)
            region = _intersect(
                region,
                _exact_bracket(allowed, disk_segments, numeric_tolerance),
                numeric_tolerance,
            )
            _record_diagnostic_step(
                diagnostic_steps,
                stage="subject_position_region",
                constraint_id=f"position-region:{position_region.region_id}",
                applied=True,
                outer_was_empty=outer_was_empty,
                region=region,
            )

        support_id = subject.support_object_id
        if support_id is not None:
            support = scene.object_by_id(support_id)
            support_locus = _center_locus(
                obb_footprint(support.obb),
                relative_vertices,
            )
            outer_was_empty = bool(region.outer.is_empty)
            region = _intersect(
                region,
                _exact_bracket(support_locus, disk_segments, numeric_tolerance),
                numeric_tolerance,
            )
            _record_diagnostic_step(
                diagnostic_steps,
                stage="support_locus",
                constraint_id=f"support:{support_id}",
                applied=True,
                outer_was_empty=outer_was_empty,
                region=region,
            )

        for obstacle in sorted(scene.objects, key=lambda obj: obj.object_id):
            if obstacle.object_id in {subject.object_id, support_id}:
                continue
            if (
                obb_z_overlap_depth(subject.obb, obstacle.obb)
                <= OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
            ):
                continue
            configuration = _configuration_obstacle(
                obb_footprint(obstacle.obb), relative_vertices
            )
            outer_was_empty = bool(region.outer.is_empty)
            region = _make_bracket(
                inner=_exclude_interior(region.inner, configuration),
                outer=_exclude_interior(region.outer, configuration),
                radial_geometry_error=region.radial_geometry_error,
                disk_segments=region.disk_segments,
                numeric_tolerance=numeric_tolerance,
            )
            _record_diagnostic_step(
                diagnostic_steps,
                stage="object_collision",
                constraint_id=f"object-collision:{obstacle.object_id}",
                applied=True,
                outer_was_empty=outer_was_empty,
                region=region,
            )

        for obstacle in sorted(
            scene.collision_obstacles,
            key=lambda item: item.obstacle_id,
        ):
            conservative_obb = obstacle.conservative_obb()
            if (
                obb_z_overlap_depth(subject.obb, conservative_obb)
                <= OBB_INTERSECTION_Z_OVERLAP_TOLERANCE
            ):
                continue
            configuration = _configuration_obstacle(
                obb_footprint(conservative_obb),
                relative_vertices,
            )
            outer_was_empty = bool(region.outer.is_empty)
            region = _make_bracket(
                inner=_exclude_interior(region.inner, configuration),
                outer=_exclude_interior(region.outer, configuration),
                radial_geometry_error=region.radial_geometry_error,
                disk_segments=region.disk_segments,
                numeric_tolerance=numeric_tolerance,
            )
            _record_diagnostic_step(
                diagnostic_steps,
                stage="collision_obstacle",
                constraint_id=f"collision-obstacle:{obstacle.obstacle_id}",
                applied=True,
                outer_was_empty=outer_was_empty,
                region=region,
            )

        visibility = self._visibility_bracket(
            scene,
            spec,
            universe,
            disk_segments,
            numeric_tolerance,
        )
        outer_was_empty = bool(region.outer.is_empty)
        region = _intersect(region, visibility, numeric_tolerance)
        _record_diagnostic_step(
            diagnostic_steps,
            stage="visibility",
            constraint_id=f"visibility:{spec.camera_id}",
            applied=True,
            outer_was_empty=outer_was_empty,
            region=region,
        )
        if diagnostic_regions is not None:
            diagnostic_regions["structural"] = region

        for stationary in sorted(scene.objects, key=lambda obj: obj.object_id):
            if stationary.object_id == subject.object_id:
                continue
            labels = self.engine.pair_labels(
                scene,
                subject.object_id,
                stationary.object_id,
                spec.camera_id,
            )
            permanently_invisible = self._pair_is_permanently_invisible(
                subject, stationary, spec.camera_id
            )
            for axis in RelationAxis:
                if stationary.object_id == spec.reference_id and axis is spec.relation_after.axis:
                    constraint = self._relation_bracket(
                        scene,
                        spec,
                        stationary,
                        spec.relation_after,
                        relative_vertices,
                        universe,
                        disk_segments,
                        numeric_tolerance,
                        target_interior_margin,
                    )
                    stage = "target_relation"
                    constraint_id = (
                        f"target:{subject.object_id}:"
                        f"{spec.relation_after.value}:{stationary.object_id}"
                    )
                    if diagnostic_regions is not None:
                        diagnostic_regions["target_constraint"] = constraint
                else:
                    constraint = self._preserved_axis_bracket(
                        scene,
                        spec,
                        stationary,
                        axis,
                        labels,
                        permanently_invisible,
                        relative_vertices,
                        universe,
                        disk_segments,
                        numeric_tolerance,
                    )
                    stage = "preserved_relation"
                    axis_labels = ",".join(
                        relation.value
                        for relation in _AXIS_RELATIONS[axis]
                        if relation in labels
                    )
                    constraint_id = (
                        f"preserve:{subject.object_id}:{stationary.object_id}:"
                        f"{axis.value}:{axis_labels or 'none'}"
                    )
                outer_was_empty = bool(region.outer.is_empty)
                if constraint is not None:
                    region = _intersect(region, constraint, numeric_tolerance)
                _record_diagnostic_step(
                    diagnostic_steps,
                    stage=stage,
                    constraint_id=constraint_id,
                    applied=constraint is not None,
                    outer_was_empty=outer_was_empty,
                    region=region,
                )
        return region

    @staticmethod
    def _validate_inputs(
        scene: Scene,
        spec: InterventionSpec,
        disk_segments: int,
        numeric_tolerance: float,
        target_interior_margin: float,
    ) -> None:
        if (
            type(disk_segments) is not int
            or disk_segments < 4
            or disk_segments & (disk_segments - 1)
        ):
            raise ValueError("disk_segments must be a power of two at least four")
        if (
            isinstance(numeric_tolerance, bool)
            or not isinstance(numeric_tolerance, (int, float))
            or not math.isfinite(float(numeric_tolerance))
            or numeric_tolerance <= 0.0
        ):
            raise ValueError("numeric_tolerance must be finite and positive")
        if (
            isinstance(target_interior_margin, bool)
            or not isinstance(target_interior_margin, (int, float))
            or not math.isfinite(float(target_interior_margin))
            or target_interior_margin < 0.0
        ):
            raise ValueError(
                "target_interior_margin must be finite and non-negative"
            )
        subject = scene.object_by_id(spec.subject_id)
        scene.object_by_id(spec.reference_id)
        scene.camera_by_id(spec.camera_id)
        if scene.children_by_support().get(subject.object_id):
            raise ValueError("subject has supported objects")
        if subject.support_object_id is not None:
            scene.object_by_id(subject.support_object_id)

    @staticmethod
    def _room(scene: Scene) -> Polygon:
        room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
        coordinates = [
            value
            for point in scene.room_polygon_xy
            for value in (point.x, point.y)
        ]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("room polygon coordinates must be finite")
        if not room.is_valid or room.is_empty or room.area <= 0.0:
            raise ValueError("room polygon must be valid with positive area")
        return shapely.normalize(room)

    def _visibility_bracket(
        self,
        scene: Scene,
        spec: InterventionSpec,
        universe: BaseGeometry,
        disk_segments: int,
        numeric_tolerance: float,
    ) -> FeasibleRegionBracket:
        subject = scene.object_by_id(spec.subject_id)
        reference = scene.object_by_id(spec.reference_id)
        subject_view = subject.views.get(spec.camera_id)
        reference_view = reference.views.get(spec.camera_id)
        if subject_view is None:
            raise ValueError("subject has no view for camera")
        if (
            subject_view.visible_fraction < self.engine.MIN_VISIBLE_FRACTION
            or subject_view.image_area_fraction < self.engine.MIN_IMAGE_AREA_FRACTION
            or reference_view is None
            or not self._fixed_view_is_visible(reference_view)
        ):
            return _empty_bracket(disk_segments, numeric_tolerance)

        calibration = self.motion_model.calibration(
            scene, subject.object_id, spec.camera_id
        )
        raw_nx, raw_ny, raw_constant = calibration.depth_coefficients
        calibrated_constant = raw_constant + calibration.depth_residual
        outer = _half_plane_region(
            universe, raw_nx, raw_ny, raw_constant, keep_greater=True
        )
        outer = _half_plane_region(
            outer, raw_nx, raw_ny, calibrated_constant, keep_greater=True
        )
        inner = _half_plane_region(
            universe,
            raw_nx,
            raw_ny,
            raw_constant - numeric_tolerance,
            keep_greater=True,
        )
        inner = _half_plane_region(
            inner,
            raw_nx,
            raw_ny,
            calibrated_constant - numeric_tolerance,
            keep_greater=True,
        )
        inner = self._full_bbox_inside_image(inner, calibration, subject_view)
        return _make_bracket(
            inner=inner,
            outer=outer,
            radial_geometry_error=0.0,
            disk_segments=disk_segments,
            numeric_tolerance=numeric_tolerance,
        )

    @staticmethod
    def _full_bbox_inside_image(
        universe: BaseGeometry,
        calibration: ProjectionCalibration,
        subject_view: ObjectView,
    ) -> BaseGeometry:
        camera = calibration.camera
        fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
        x_nx, x_ny, x_constant = calibration.camera_x_coefficients
        y_nx, y_ny, y_constant = calibration.camera_y_coefficients
        d_nx, d_ny, d_constant = calibration.depth_coefficients
        half_width = (subject_view.bbox.xmax - subject_view.bbox.xmin) / 2.0
        half_height = (subject_view.bbox.ymax - subject_view.bbox.ymin) / 2.0
        if half_width <= 0.0 or half_height <= 0.0:
            raise ValueError("subject projected bbox must have positive area")

        constraints = (
            (
                fx * x_nx
                + (cx + calibration.horizontal_residual - half_width) * d_nx,
                fx * x_ny
                + (cx + calibration.horizontal_residual - half_width) * d_ny,
                fx * x_constant
                + (cx + calibration.horizontal_residual - half_width) * d_constant,
                True,
            ),
            (
                fx * x_nx
                + (
                    cx
                    + calibration.horizontal_residual
                    - (camera.width - half_width)
                )
                * d_nx,
                fx * x_ny
                + (
                    cx
                    + calibration.horizontal_residual
                    - (camera.width - half_width)
                )
                * d_ny,
                fx * x_constant
                + (
                    cx
                    + calibration.horizontal_residual
                    - (camera.width - half_width)
                )
                * d_constant,
                False,
            ),
            (
                -fy * y_nx
                + (cy + calibration.vertical_residual - half_height) * d_nx,
                -fy * y_ny
                + (cy + calibration.vertical_residual - half_height) * d_ny,
                -fy * y_constant
                + (cy + calibration.vertical_residual - half_height) * d_constant,
                True,
            ),
            (
                -fy * y_nx
                + (
                    cy
                    + calibration.vertical_residual
                    - (camera.height - half_height)
                )
                * d_nx,
                -fy * y_ny
                + (
                    cy
                    + calibration.vertical_residual
                    - (camera.height - half_height)
                )
                * d_ny,
                -fy * y_constant
                + (
                    cy
                    + calibration.vertical_residual
                    - (camera.height - half_height)
                )
                * d_constant,
                False,
            ),
        )
        result = universe
        for nx, ny, constant, keep_greater in constraints:
            result = _half_plane_region(
                result, nx, ny, constant, keep_greater=keep_greater
            )
        return result

    def _preserved_axis_bracket(
        self,
        scene: Scene,
        spec: InterventionSpec,
        stationary: SceneObject,
        axis: RelationAxis,
        labels: frozenset[Relation],
        permanently_invisible: bool,
        relative_vertices: list[tuple[float, float]],
        universe: BaseGeometry,
        disk_segments: int,
        numeric_tolerance: float,
    ) -> FeasibleRegionBracket | None:
        axis_labels = tuple(
            relation for relation in _AXIS_RELATIONS[axis] if relation in labels
        )
        if len(axis_labels) == 1:
            return self._relation_bracket(
                scene,
                spec,
                stationary,
                axis_labels[0],
                relative_vertices,
                universe,
                disk_segments,
                numeric_tolerance,
            )
        if len(axis_labels) > 1:
            raise ValueError("original relation axis has contradictory labels")
        if permanently_invisible:
            return None
        first_relation, second_relation = _AXIS_RELATIONS[axis]
        first = self._relation_bracket(
            scene,
            spec,
            stationary,
            first_relation,
            relative_vertices,
            universe,
            disk_segments,
            numeric_tolerance,
        )
        second = self._relation_bracket(
            scene,
            spec,
            stationary,
            second_relation,
            relative_vertices,
            universe,
            disk_segments,
            numeric_tolerance,
        )
        excluded = _union(first, second, universe, numeric_tolerance)
        dead_zone = _complement(
            excluded,
            universe,
            numeric_tolerance,
        )
        guarded_first = _normalize_geometry(
            first.outer.buffer(numeric_tolerance).intersection(universe)
        )
        guarded_second = _normalize_geometry(
            second.outer.buffer(numeric_tolerance).intersection(universe)
        )
        guarded_excluded = _normalize_geometry(
            guarded_first.union(guarded_second)
        )
        return _make_bracket(
            inner=_normalize_geometry(universe.difference(guarded_excluded)),
            outer=dead_zone.outer,
            radial_geometry_error=dead_zone.radial_geometry_error,
            disk_segments=dead_zone.disk_segments,
            numeric_tolerance=numeric_tolerance,
        )

    def _relation_bracket(
        self,
        scene: Scene,
        spec: InterventionSpec,
        stationary: SceneObject,
        relation: Relation,
        relative_vertices: list[tuple[float, float]],
        universe: BaseGeometry,
        disk_segments: int,
        numeric_tolerance: float,
        inner_margin: float = 0.0,
    ) -> FeasibleRegionBracket:
        subject = scene.object_by_id(spec.subject_id)
        if relation.axis in {RelationAxis.HORIZONTAL, RelationAxis.DEPTH}:
            reference_view = stationary.views.get(spec.camera_id)
            if reference_view is None:
                return _empty_bracket(disk_segments, numeric_tolerance)
            calibration = self.motion_model.calibration(
                scene, subject.object_id, spec.camera_id
            )
            if relation.axis is RelationAxis.HORIZONTAL:
                camera = calibration.camera
                nominal_target = reference_view.bbox.center_x + (
                    camera.width * self.engine.LEFT_RIGHT_FRACTION
                    if relation is Relation.RIGHT
                    else -camera.width * self.engine.LEFT_RIGHT_FRACTION
                )
                fx, _, cx, _, _, _, _, _, _ = camera.intrinsics
                x_nx, x_ny, x_constant = calibration.camera_x_coefficients
                d_nx, d_ny, d_constant = calibration.depth_coefficients

                def coefficients(target: float) -> tuple[float, float, float]:
                    scale = cx + calibration.horizontal_residual - target
                    return (
                        fx * x_nx + scale * d_nx,
                        fx * x_ny + scale * d_ny,
                        fx * x_constant + scale * d_constant,
                    )

                outer_target = nominal_target + (
                    -self.engine.COMPARISON_TOLERANCE
                    if relation is Relation.RIGHT
                    else self.engine.COMPARISON_TOLERANCE
                )
                inner_coefficients = _guard_inner_coefficients(
                    coefficients(nominal_target),
                    keep_greater=relation is Relation.RIGHT,
                    margin=inner_margin,
                )
                return _linear_bracket(
                    universe,
                    inner_coefficients,
                    coefficients(outer_target),
                    keep_greater=relation is Relation.RIGHT,
                    include_inner_in_outer=True,
                    disk_segments=disk_segments,
                    numeric_tolerance=numeric_tolerance,
                )
            d_nx, d_ny, d_constant = calibration.depth_coefficients
            calibrated_constant = d_constant + calibration.depth_residual
            nominal_target = reference_view.camera_depth + (
                self.engine.FRONT_BEHIND_METERS
                if relation is Relation.BEHIND
                else -self.engine.FRONT_BEHIND_METERS
            )
            outer_target = nominal_target + (
                -self.engine.COMPARISON_TOLERANCE
                if relation is Relation.BEHIND
                else self.engine.COMPARISON_TOLERANCE
            )
            keep_greater = relation is Relation.BEHIND
            inner_coefficients = _guard_inner_coefficients(
                (d_nx, d_ny, calibrated_constant - nominal_target),
                keep_greater=keep_greater,
                margin=inner_margin,
            )
            return _linear_bracket(
                universe,
                inner_coefficients,
                (d_nx, d_ny, calibrated_constant - outer_target),
                keep_greater=keep_greater,
                disk_segments=disk_segments,
                numeric_tolerance=numeric_tolerance,
            )

        configuration = _configuration_obstacle(
            obb_footprint(stationary.obb), relative_vertices
        )
        if relation is Relation.NEAR:
            result = near_bracket(
                configuration,
                self.engine.NEAR_METERS,
                disk_segments,
                numeric_tolerance,
            )
        else:
            result = far_bracket(
                configuration,
                self.engine.FAR_METERS,
                disk_segments,
                numeric_tolerance,
                universe,
            )
            if inner_margin > 0.0:
                guarded = far_bracket(
                    configuration,
                    self.engine.FAR_METERS + inner_margin,
                    disk_segments,
                    numeric_tolerance,
                    universe,
                )
                return _make_bracket(
                    inner=guarded.inner,
                    outer=result.outer,
                    radial_geometry_error=max(
                        guarded.radial_geometry_error,
                        result.radial_geometry_error,
                    ),
                    disk_segments=result.disk_segments,
                    numeric_tolerance=numeric_tolerance,
                )
        return _make_bracket(
            inner=result.inner,
            outer=result.outer,
            radial_geometry_error=result.radial_geometry_error,
            disk_segments=result.disk_segments,
            numeric_tolerance=numeric_tolerance,
        )

    def _pair_is_permanently_invisible(
        self,
        subject: SceneObject,
        stationary: SceneObject,
        camera_id: str,
    ) -> bool:
        subject_view = subject.views.get(camera_id)
        stationary_view = stationary.views.get(camera_id)
        return (
            subject_view is None
            or subject_view.visible_fraction < self.engine.MIN_VISIBLE_FRACTION
            or subject_view.image_area_fraction < self.engine.MIN_IMAGE_AREA_FRACTION
            or stationary_view is None
            or not self._fixed_view_is_visible(stationary_view)
        )

    def _fixed_view_is_visible(self, view: ObjectView) -> bool:
        return (
            view.visible_fraction >= self.engine.MIN_VISIBLE_FRACTION
            and view.image_area_fraction >= self.engine.MIN_IMAGE_AREA_FRACTION
            and view.truncated_fraction <= self.engine.MAX_TRUNCATED_FRACTION
        )
