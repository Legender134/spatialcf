"""Canonical scene construction and fail-closed before-scene validation."""

from __future__ import annotations

import math
from collections.abc import Iterable

from shapely.geometry import Polygon

from spatialcf.domain.enums import SolverStatus
from spatialcf.domain.models import (
    OBB,
    BBox2D,
    Camera,
    ObjectView,
    Quaternion,
    Scene,
    SceneObject,
    Vec2,
    Vec3,
)
from spatialcf.geometry.obb import ground_gap, obb_footprint
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.stress.models import (
    StressCase,
    StressCaseDraft,
    StressOracleResult,
    expected_oracle_result,
)
from spatialcf.solver.stress.oracles import (
    UnattainedOracleInfimumError,
    recompute_stress_oracle,
)

_SOURCE = "core-solver-stress-v1"
_CAMERA_ID = "camera"
_BBOX_HALF_SIZE = 10.0
_GEOMETRY_TOLERANCE = 1e-9
_PROJECTION_TOLERANCE = 1e-12


class StressCaseError(ValueError):
    """A generated stress case cannot support a validation claim."""


def yaw_quaternion(degrees: int) -> Quaternion:
    radians = math.radians(degrees) / 2.0
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(radians),
        w=math.cos(radians),
    )


def make_camera(
    *,
    focal_px: float,
    depth_offset: float = 0.0,
    camera_id: str = _CAMERA_ID,
    camera_height: float = 1.5,
) -> Camera:
    """Build the frozen affine Z-up camera used by direct stress families."""
    values = (focal_px, depth_offset, camera_height)
    if focal_px <= 0.0 or not all(math.isfinite(value) for value in values):
        raise StressCaseError("invalid_camera")
    return Camera(
        camera_id=camera_id,
        width=640,
        height=480,
        intrinsics=(
            float(focal_px),
            0.0,
            320.0,
            0.0,
            float(focal_px),
            240.0,
            0.0,
            0.0,
            1.0,
        ),
        world_to_camera=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -float(camera_height),
            0.0,
            1.0,
            0.0,
            float(depth_offset),
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def _camera_coordinate(camera: Camera, row: int, point: Vec3) -> float:
    start = row * 4
    matrix = camera.world_to_camera
    return (
        matrix[start] * point.x
        + matrix[start + 1] * point.y
        + matrix[start + 2] * point.z
        + matrix[start + 3]
    )


def _initial_view(camera: Camera, position: Vec3) -> ObjectView:
    camera_x = _camera_coordinate(camera, 0, position)
    camera_y = _camera_coordinate(camera, 1, position)
    depth = _camera_coordinate(camera, 2, position)
    if (
        not all(math.isfinite(value) for value in (camera_x, camera_y, depth))
        or depth <= 0.0
    ):
        raise StressCaseError("invalid_initial_projection")
    fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
    center_x = fx * camera_x / depth + cx
    center_y = cy - fy * camera_y / depth
    bbox = BBox2D(
        xmin=center_x - _BBOX_HALF_SIZE,
        ymin=center_y - _BBOX_HALF_SIZE,
        xmax=center_x + _BBOX_HALF_SIZE,
        ymax=center_y + _BBOX_HALF_SIZE,
    )
    if not (
        0.0 <= bbox.xmin <= bbox.xmax <= camera.width
        and 0.0 <= bbox.ymin <= bbox.ymax <= camera.height
    ):
        raise StressCaseError("initial_bbox_outside_image")
    return ObjectView(
        camera_id=camera.camera_id,
        bbox=bbox,
        camera_depth=depth,
        visible_fraction=0.90,
        image_area_fraction=0.02,
        truncated_fraction=0.0,
    )


def make_object(
    object_id: str,
    *,
    position: Vec3,
    extent: Vec3,
    yaw_degrees: int,
    movable: bool,
    camera: Camera,
    category: str | None = None,
    support_object_id: str | None = None,
    request_eligible: bool = True,
) -> SceneObject:
    """Build one canonical OBB object and its direct initial projection."""
    numeric = (*position.model_dump().values(), *extent.model_dump().values())
    if (
        not object_id
        or not all(math.isfinite(float(value)) for value in numeric)
        or min(extent.x, extent.y, extent.z) <= 0.0
    ):
        raise StressCaseError("invalid_object")
    rotation = yaw_quaternion(yaw_degrees)
    return SceneObject(
        object_id=object_id,
        name=object_id,
        category=category or object_id,
        movable=movable,
        request_eligible=request_eligible,
        position=position,
        rotation=rotation,
        obb=OBB(center=position, extent=extent, rotation=rotation),
        support_object_id=support_object_id,
        views={camera.camera_id: _initial_view(camera, position)},
    )


def make_scene(
    *,
    scene_id: str,
    generation_seed: int,
    room_bounds: tuple[float, float, float, float],
    camera: Camera,
    objects: Iterable[SceneObject],
    source: str = _SOURCE,
) -> Scene:
    """Build a rectangular Canonical Scene without binding a final case ID."""
    min_x, min_y, max_x, max_y = room_bounds
    if (
        not all(math.isfinite(value) for value in room_bounds)
        or min_x >= max_x
        or min_y >= max_y
    ):
        raise StressCaseError("invalid_room")
    return Scene(
        scene_id=scene_id,
        source=source,
        room_polygon_xy=(
            Vec2(x=min_x, y=min_y),
            Vec2(x=max_x, y=min_y),
            Vec2(x=max_x, y=max_y),
            Vec2(x=min_x, y=max_y),
        ),
        cameras=(camera,),
        objects=tuple(objects),
        pinned_object_ids=frozenset(),
        generation_seed=generation_seed,
    )


def _require_finite_tree(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise StressCaseError("non_finite")
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_tree(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _require_finite_tree(item)


def _require_scene_references(case: StressCaseDraft) -> None:
    scene = case.scene
    object_ids = [obj.object_id for obj in scene.objects]
    camera_ids = [camera.camera_id for camera in scene.cameras]
    if len(object_ids) != len(set(object_ids)) or len(camera_ids) != len(
        set(camera_ids)
    ):
        raise StressCaseError("scene_references")
    objects = set(object_ids)
    cameras = set(camera_ids)
    spec = case.intervention
    if (
        spec.subject_id == spec.reference_id
        or spec.subject_id not in objects
        or spec.reference_id not in objects
        or spec.camera_id not in cameras
    ):
        raise StressCaseError("scene_references")
    subject = scene.object_by_id(spec.subject_id)
    if not subject.movable or spec.subject_id in scene.pinned_object_ids:
        raise StressCaseError("scene_references")
    if scene.children_by_support().get(spec.subject_id):
        raise StressCaseError("scene_references")
    if any(obj.movable for obj in scene.objects if obj.object_id != spec.subject_id):
        raise StressCaseError("scene_references")
    for obj in scene.objects:
        if obj.support_object_id is not None and (
            obj.support_object_id not in objects
            or obj.support_object_id == obj.object_id
        ):
            raise StressCaseError("scene_references")
        if any(
            key != view.camera_id or key not in cameras
            for key, view in obj.views.items()
        ):
            raise StressCaseError("scene_references")
    for object_id in (spec.subject_id, spec.reference_id):
        if spec.camera_id not in scene.object_by_id(object_id).views:
            raise StressCaseError("scene_references")


def _require_inside_room(scene: Scene) -> None:
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    if not room.is_valid or room.is_empty:
        raise StressCaseError("outside_room")
    for obj in scene.objects:
        if not room.buffer(_GEOMETRY_TOLERANCE).covers(obb_footprint(obj.obb)):
            raise StressCaseError("outside_room")


def _require_initial_boxes_inside_image(scene: Scene) -> None:
    for obj in scene.objects:
        for camera_id, view in obj.views.items():
            camera = scene.camera_by_id(camera_id)
            bbox = view.bbox
            if not (
                0.0 <= bbox.xmin <= bbox.xmax <= camera.width
                and 0.0 <= bbox.ymin <= bbox.ymax <= camera.height
            ):
                raise StressCaseError("initial_bbox_outside_image")


def _require_canonical_initial_observations(scene: Scene) -> None:
    for camera in scene.cameras:
        if camera.width != 640 or camera.height != 480:
            raise StressCaseError("initial_observation_mismatch")
    for obj in scene.objects:
        for camera_id, view in obj.views.items():
            camera = scene.camera_by_id(camera_id)
            camera_x = _camera_coordinate(camera, 0, obj.position)
            camera_y = _camera_coordinate(camera, 1, obj.position)
            depth = _camera_coordinate(camera, 2, obj.position)
            if depth <= 0.0:
                raise StressCaseError("initial_observation_mismatch")
            fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
            projected_x = fx * camera_x / depth + cx
            projected_y = cy - fy * camera_y / depth
            bbox = view.bbox
            observed_y = (bbox.ymin + bbox.ymax) / 2.0
            comparisons = (
                (bbox.xmax - bbox.xmin, 2.0 * _BBOX_HALF_SIZE),
                (bbox.ymax - bbox.ymin, 2.0 * _BBOX_HALF_SIZE),
                (bbox.center_x, projected_x),
                (observed_y, projected_y),
                (view.camera_depth, depth),
                (view.visible_fraction, 0.90),
                (view.image_area_fraction, 0.02),
                (view.truncated_fraction, 0.0),
            )
            if not all(
                math.isclose(
                    observed,
                    expected,
                    rel_tol=0.0,
                    abs_tol=_PROJECTION_TOLERANCE,
                )
                for observed, expected in comparisons
            ):
                raise StressCaseError("initial_observation_mismatch")


def _require_support_contact(scene: Scene) -> None:
    for obj in scene.objects:
        if obj.support_object_id is None:
            continue
        support = scene.object_by_id(obj.support_object_id)
        if (
            not obb_footprint(support.obb)
            .buffer(_GEOMETRY_TOLERANCE)
            .covers(obb_footprint(obj.obb))
        ):
            raise StressCaseError("support_overhang")
        object_bottom = obj.obb.center.z - obj.obb.extent.z / 2.0
        support_top = support.obb.center.z + support.obb.extent.z / 2.0
        if not math.isclose(
            object_bottom,
            support_top,
            rel_tol=0.0,
            abs_tol=_GEOMETRY_TOLERANCE,
        ):
            raise StressCaseError("support_separation")


def _require_no_3d_intersection(scene: Scene, minimum_gap: float) -> None:
    for index, first in enumerate(scene.objects):
        for second in scene.objects[index + 1 :]:
            if (
                first.support_object_id == second.object_id
                or second.support_object_id == first.object_id
            ):
                continue
            first_bottom = first.obb.center.z - first.obb.extent.z / 2.0
            first_top = first.obb.center.z + first.obb.extent.z / 2.0
            second_bottom = second.obb.center.z - second.obb.extent.z / 2.0
            second_top = second.obb.center.z + second.obb.extent.z / 2.0
            z_overlap = min(first_top, second_top) - max(first_bottom, second_bottom)
            if z_overlap > _GEOMETRY_TOLERANCE and ground_gap(first.obb, second.obb) < (
                minimum_gap - _GEOMETRY_TOLERANCE
            ):
                raise StressCaseError("object_intersection")


def _require_source_relation(case: StressCaseDraft) -> None:
    spec = case.intervention
    observed = RelationEngine().observe(
        case.scene,
        spec.subject_id,
        spec.reference_id,
        spec.relation_before,
        spec.camera_id,
    )
    if observed.status is not SolverStatus.SUCCESS or not observed.satisfied:
        raise StressCaseError("source_unsatisfied")


def _within_coordinate_ulps(first: Vec2, second: Vec2) -> bool:
    scale = max(
        1.0,
        abs(first.x),
        abs(first.y),
        abs(second.x),
        abs(second.y),
    )
    tolerance = 64.0 * math.ulp(scale)
    return abs(first.x - second.x) <= tolerance and abs(first.y - second.y) <= tolerance


def _transformed_oracle_matches(
    expected: StressOracleResult,
    recomputed: StressOracleResult,
) -> bool:
    if (
        expected.expected_outcome != recomputed.expected_outcome
        or expected.exact_infimum_m != recomputed.exact_infimum_m
        or expected.maximum_possible_value_m != recomputed.maximum_possible_value_m
        or expected.required_value_m != recomputed.required_value_m
        or len(expected.exact_infimum_points) != len(recomputed.exact_infimum_points)
    ):
        return False
    candidate_indices = tuple(
        tuple(
            recomputed_index
            for recomputed_index, recomputed_point in enumerate(
                recomputed.exact_infimum_points
            )
            if _within_coordinate_ulps(expected_point, recomputed_point)
        )
        for expected_point in expected.exact_infimum_points
    )
    matched_expected: list[int | None] = [None] * len(recomputed.exact_infimum_points)

    def augment(expected_index: int, visited: set[int]) -> bool:
        for recomputed_index in candidate_indices[expected_index]:
            if recomputed_index in visited:
                continue
            visited.add(recomputed_index)
            previous = matched_expected[recomputed_index]
            if previous is None or augment(previous, visited):
                matched_expected[recomputed_index] = expected_index
                return True
        return False

    for expected_index in sorted(
        range(len(candidate_indices)),
        key=lambda index: (len(candidate_indices[index]), index),
    ):
        if not augment(expected_index, set()):
            return False
    return True


def validate_before_draft(case: StressCaseDraft) -> None:
    """Validate geometry and oracle without requiring final public identity."""
    _require_finite_tree(case.model_dump(mode="json"))
    _require_scene_references(case)
    _require_initial_boxes_inside_image(case.scene)
    _require_inside_room(case.scene)
    _require_support_contact(case.scene)
    _require_no_3d_intersection(case.scene, minimum_gap=0.05)
    _require_canonical_initial_observations(case.scene)
    _require_source_relation(case)
    try:
        recomputed = recompute_stress_oracle(case)
    except UnattainedOracleInfimumError as error:
        raise StressCaseError("unattained_oracle_infimum") from error
    expected = expected_oracle_result(case.oracle)
    oracle_matches = (
        _transformed_oracle_matches(expected, recomputed)
        if case.transform.transformed
        else recomputed == expected
    )
    if not oracle_matches:
        raise StressCaseError("oracle_mismatch")


def validate_before_case(case: StressCase) -> None:
    """Add final case/scene identity binding to the draft validity contract."""
    if case.scene.scene_id != case.case_id:
        raise StressCaseError("scene_id_mismatch")
    validate_before_draft(case.as_draft())
