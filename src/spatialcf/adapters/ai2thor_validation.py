"""Shared validation primitives for canonical AI2-THOR observations.

This adapter-layer module owns the native tolerance and normalization contract.
Pipeline code may orchestrate and publish the resulting evidence, but both the
pipeline and grounded candidate executor use this single implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from shapely.geometry import Polygon

from spatialcf.adapters.ai2thor import (
    AI2ThorFloorEnvelope,
    AI2ThorObservation,
    AI2ThorPoseApplication,
    AI2ThorRuntimeIdentity,
    AI2ThorSceneSettlement,
)
from spatialcf.domain.models import InterventionSpec, Quaternion, Scene, Vec2, Vec3
from spatialcf.geometry.obb import obb_footprint
from spatialcf.relations.engine import RelationEngine


@dataclass(frozen=True)
class CertifiedAI2ThorPilotTolerances:
    """Frozen tolerances shared by certified and grounded native execution."""

    floor_clearance_m: float = 0.02
    native_collision_clearance_m: float = 0.02
    commanded_pose_residual_m: float = 1e-5
    stationary_geometry_residual_m: float = 1e-5
    rotation_residual_degrees: float = 1e-4
    camera_intrinsics_residual: float = 1e-8
    camera_extrinsics_residual: float = 1e-5
    vertical_contact_residual_m: float = 1e-2
    overlap_xy_area_tolerance: float = 1e-9
    overlap_z_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


AI2ThorValidationTolerances = CertifiedAI2ThorPilotTolerances


class CertifiedAI2ThorCaseRejected(RuntimeError):
    """A native validation case failed without repairing its request."""

    def __init__(self, stage: str, reasons: tuple[str, ...]) -> None:
        if not stage or not reasons:
            raise ValueError("a rejected case requires a stage and reasons")
        self.stage = stage
        self.reasons = reasons
        super().__init__(f"{stage}: {', '.join(reasons)}")


AI2ThorValidationRejected = CertifiedAI2ThorCaseRejected


def floor_polygon(envelope: AI2ThorFloorEnvelope) -> Polygon:
    polygon = Polygon([(point.x, point.y) for point in envelope.polygon_xy])
    if not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0:
        raise AI2ThorValidationRejected("input", ("invalid_floor_envelope",))
    return polygon


def is_finite_non_negative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def position_distance(left: Vec3, right: Vec3) -> float:
    return math.dist(
        (left.x, left.y, left.z),
        (right.x, right.y, right.z),
    )


def scene_without_views(scene: Scene) -> Scene:
    return scene.model_copy(
        update={
            "objects": tuple(
                obj.model_copy(update={"views": {}}) for obj in scene.objects
            )
        }
    )


def floor_envelope_contract_errors(
    scene: Scene,
    envelope: AI2ThorFloorEnvelope,
    limits: AI2ThorValidationTolerances,
    *,
    runtime_identity: AI2ThorRuntimeIdentity | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    if envelope.scene_id != scene.scene_id:
        errors.append("floor_envelope_scene_mismatch")
    if not envelope.floor_object_id or not envelope.floor_name:
        errors.append("floor_envelope_identity_invalid")
    if not is_finite_non_negative(envelope.clearance_m):
        errors.append("floor_envelope_clearance_invalid")
        return tuple(errors)
    clearance = float(envelope.clearance_m)
    if clearance != limits.floor_clearance_m:
        errors.append("floor_envelope_clearance_mismatch")

    bounds = envelope.native_aabb
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    if quaternion_residual_degrees(bounds.rotation, identity) > 1e-12:
        errors.append("floor_envelope_aabb_rotation_invalid")
    expected_top = bounds.center.z + bounds.extent.z / 2.0
    if (
        not math.isfinite(envelope.floor_top_z)
        or abs(envelope.floor_top_z - expected_top) > 1e-12
    ):
        errors.append("floor_envelope_top_mismatch")
    native_bounds = (
        bounds.center.x - bounds.extent.x / 2.0,
        bounds.center.y - bounds.extent.y / 2.0,
        bounds.center.x + bounds.extent.x / 2.0,
        bounds.center.y + bounds.extent.y / 2.0,
    )
    is_procedural = (
        runtime_identity is not None
        and runtime_identity.native_scene_name == "Procedural"
    )
    effective_bounds = native_bounds
    if is_procedural:
        source_bounds = runtime_identity.source_floor_xz_bounds
        if source_bounds is None or any(
            not math.isclose(source, native, rel_tol=0.0, abs_tol=1e-5)
            for source, native in zip(source_bounds, native_bounds, strict=True)
        ):
            errors.append("procedural_floor_source_mismatch")
        elif runtime_identity.source_scene_alias != scene.scene_id:
            errors.append("procedural_floor_source_scene_mismatch")
        else:
            effective_bounds = (
                max(source_bounds[0], native_bounds[0]),
                max(source_bounds[1], native_bounds[1]),
                min(source_bounds[2], native_bounds[2]),
                min(source_bounds[3], native_bounds[3]),
            )
    if is_procedural:
        minimum_x = effective_bounds[0] + clearance
        minimum_y = effective_bounds[1] + clearance
        maximum_x = effective_bounds[2] - clearance
        maximum_y = effective_bounds[3] - clearance
    else:
        # Preserve the frozen legacy contract's operation order.  Rebuilding
        # the bounds as min/max first can change a coordinate by one ULP.
        half_x = bounds.extent.x / 2.0 - clearance
        half_y = bounds.extent.y / 2.0 - clearance
        minimum_x = bounds.center.x - half_x
        minimum_y = bounds.center.y - half_y
        maximum_x = bounds.center.x + half_x
        maximum_y = bounds.center.y + half_y
    if maximum_x <= minimum_x or maximum_y <= minimum_y:
        errors.append("floor_envelope_empty")
    else:
        expected_polygon = (
            Vec2(x=minimum_x, y=minimum_y),
            Vec2(x=maximum_x, y=minimum_y),
            Vec2(x=maximum_x, y=maximum_y),
            Vec2(x=minimum_x, y=maximum_y),
        )
        if len(envelope.polygon_xy) != len(expected_polygon) or any(
            not math.isclose(actual.x, expected.x, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(actual.y, expected.y, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                envelope.polygon_xy,
                expected_polygon,
                strict=True,
            )
        ):
            errors.append("floor_envelope_polygon_mismatch")
    if scene.room_polygon_xy != envelope.polygon_xy:
        errors.append("floor_envelope_room_mismatch")
    return tuple(sorted(set(errors)))


def settlement_contract_errors(
    scene_id: str,
    settlement: AI2ThorSceneSettlement,
) -> tuple[str, ...]:
    errors: list[str] = []
    if settlement.observed_scene.scene_id != scene_id:
        errors.append("settlement_scene_mismatch")
    if settlement.observation.scene != settlement.observed_scene:
        errors.append("settlement_observation_scene_mismatch")
    if settlement.observation.is_scene_at_rest is not True:
        errors.append("settlement_scene_not_at_rest")
    if type(settlement.pass_steps) is not int or settlement.pass_steps < 0:
        errors.append("settlement_pass_steps_invalid")
    return tuple(errors)


def pose_application_contract_errors(
    expected_after: Scene,
    spec: InterventionSpec,
    application: AI2ThorPoseApplication,
) -> tuple[tuple[str, ...], float]:
    errors: list[str] = []
    if scene_without_views(application.commanded_scene) != scene_without_views(
        expected_after
    ):
        errors.append("commanded_scene_mismatch")

    expected_position = expected_after.object_by_id(spec.subject_id).position
    try:
        observed_position = application.observed_scene.object_by_id(
            spec.subject_id
        ).position
    except KeyError:
        errors.append("observed_subject_missing")
        return tuple(sorted(set(errors))), math.inf
    if position_distance(application.commanded_position, expected_position) > 1e-12:
        errors.append("commanded_position_mismatch")
    if position_distance(application.observed_position, observed_position) > 1e-12:
        errors.append("observed_position_mismatch")

    recomputed_residual = position_distance(
        application.commanded_position,
        application.observed_position,
    )
    if not is_finite_non_negative(application.position_residual_m):
        errors.append("reported_pose_residual_invalid")
    elif abs(application.position_residual_m - recomputed_residual) > 1e-12:
        errors.append("reported_pose_residual_mismatch")
    return tuple(sorted(set(errors))), recomputed_residual


def observation_contract_errors(
    observation: AI2ThorObservation,
    camera_id: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        camera = observation.scene.camera_by_id(camera_id)
    except KeyError:
        return ("observation_camera_missing",)

    assets = (
        ("rgb", observation.rgb_png, observation.rgb_png_sha256),
        ("depth", observation.depth_npy, observation.depth_npy_sha256),
        (
            "instance",
            observation.instance_png,
            observation.instance_png_sha256,
        ),
        (
            "pointcloud",
            observation.pointcloud_ply,
            observation.pointcloud_ply_sha256,
        ),
    )
    for name, payload, expected_digest in assets:
        if sha256(payload).hexdigest() != expected_digest:
            errors.append(f"observation_{name}_digest_mismatch")

    for name, payload in (
        ("rgb", observation.rgb_png),
        ("instance", observation.instance_png),
    ):
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.size != (camera.width, camera.height):
                    errors.append(f"observation_{name}_dimensions_mismatch")
                if image.mode != "RGB":
                    errors.append(f"observation_{name}_dtype_mismatch")
        except (OSError, UnidentifiedImageError, ValueError):
            errors.append(f"observation_{name}_invalid")

    try:
        depth = np.load(BytesIO(observation.depth_npy), allow_pickle=False)
        if depth.shape != (camera.height, camera.width):
            errors.append("observation_depth_dimensions_mismatch")
        if depth.dtype != np.float32:
            errors.append("observation_depth_dtype_mismatch")
        if not np.any(np.isfinite(depth) & (depth > 0.0)):
            errors.append("observation_depth_has_no_positive_sample")
    except (OSError, ValueError, TypeError):
        errors.append("observation_depth_invalid")

    try:
        pointcloud = observation.pointcloud_ply.decode("ascii")
        lines = pointcloud.splitlines()
        if (
            len(lines) < 4
            or lines[0] != "ply"
            or lines[1] != "format ascii 1.0"
            or "end_header" not in lines
        ):
            raise ValueError
        vertex_lines = [line for line in lines if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError
        vertex_count = int(vertex_lines[0].removeprefix("element vertex "))
        header_end = lines.index("end_header")
        vertices = lines[header_end + 1 :]
        if vertex_count <= 0 or len(vertices) != vertex_count:
            raise ValueError
        for vertex in vertices:
            values = vertex.split()
            if len(values) != 6:
                raise ValueError
            coordinates = tuple(float(value) for value in values[:3])
            colors = tuple(int(value) for value in values[3:])
            if not all(math.isfinite(value) for value in coordinates) or not all(
                0 <= value <= 255 for value in colors
            ):
                raise ValueError
    except (UnicodeDecodeError, ValueError):
        errors.append("observation_pointcloud_invalid")

    expected_ids = {obj.object_id for obj in observation.scene.objects}
    if set(observation.instance_pixel_counts) != expected_ids:
        errors.append("observation_instance_count_ids_mismatch")
    max_pixels = camera.width * camera.height
    if any(
        type(count) is not int or count < 0 or count > max_pixels
        for count in observation.instance_pixel_counts.values()
    ):
        errors.append("observation_instance_count_invalid")
    return tuple(sorted(set(errors)))


def vertical_contact_residual(
    scene: Scene,
    spec: InterventionSpec,
    envelope: AI2ThorFloorEnvelope,
) -> tuple[float, bool]:
    """Measure contact against AI2-THOR's declared receptacle boundary."""
    subject = scene.object_by_id(spec.subject_id)
    subject_bottom = subject.obb.center.z - subject.obb.extent.z / 2.0
    if subject.support_object_id is None:
        return abs(subject_bottom - envelope.floor_top_z), True
    support = scene.object_by_id(subject.support_object_id)
    support_bottom = support.obb.center.z - support.obb.extent.z / 2.0
    support_top = support.obb.center.z + support.obb.extent.z / 2.0
    footprint_supported = obb_footprint(support.obb).buffer(1e-9).covers(
        obb_footprint(subject.obb)
    )
    return min(
        abs(subject_bottom - support_bottom),
        abs(subject_bottom - support_top),
    ), footprint_supported


def coordinate_residual(left: Vec3, right: Vec3) -> float:
    return max(
        abs(left.x - right.x),
        abs(left.y - right.y),
        abs(left.z - right.z),
    )


def quaternion_residual_degrees(
    left: Quaternion,
    right: Quaternion,
) -> float:
    left_values = (left.x, left.y, left.z, left.w)
    right_values = (right.x, right.y, right.z, right.w)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if (
        not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
        or left_norm <= 0.0
        or right_norm <= 0.0
    ):
        return math.inf
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            )
        )
        / (left_norm * right_norm)
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def camera_residual_errors(
    before: Scene,
    after: Scene,
    limits: AI2ThorValidationTolerances,
) -> tuple[str, ...]:
    if len(before.cameras) != len(after.cameras):
        return ("camera_set_changed",)
    errors: list[str] = []
    after_by_id = {camera.camera_id: camera for camera in after.cameras}
    if len(after_by_id) != len(after.cameras):
        return ("camera_set_changed",)
    for expected in before.cameras:
        observed = after_by_id.get(expected.camera_id)
        if observed is None:
            errors.append("camera_set_changed")
            continue
        if expected.width != observed.width or expected.height != observed.height:
            errors.append(f"camera_dimensions_changed:{expected.camera_id}")
        if max(
            abs(left - right)
            for left, right in zip(
                expected.intrinsics,
                observed.intrinsics,
                strict=True,
            )
        ) > limits.camera_intrinsics_residual:
            errors.append(f"camera_intrinsics_changed:{expected.camera_id}")
        if max(
            abs(left - right)
            for left, right in zip(
                expected.world_to_camera,
                observed.world_to_camera,
                strict=True,
            )
        ) > limits.camera_extrinsics_residual:
            errors.append(f"camera_extrinsics_changed:{expected.camera_id}")
    return tuple(errors)


def relation_graph_diff(
    before: Scene,
    observed: Scene,
    spec: InterventionSpec,
) -> tuple[tuple[str, ...], int]:
    """Return the complete directed relation diff and leaked pair count."""
    engine = RelationEngine()
    changed: list[str] = []
    leaked_pairs: set[tuple[str, str]] = set()
    target_pair = frozenset((spec.subject_id, spec.reference_id))
    objects = sorted(before.objects, key=lambda obj: obj.object_id)
    for first in objects:
        for second in objects:
            if first.object_id == second.object_id:
                continue
            old = engine.pair_labels(
                before,
                first.object_id,
                second.object_id,
                spec.camera_id,
            )
            new = engine.pair_labels(
                observed,
                first.object_id,
                second.object_id,
                spec.camera_id,
            )
            for relation in sorted(old - new, key=lambda item: item.value):
                changed.append(
                    f"-{first.object_id}:{relation.value}:{second.object_id}"
                )
            for relation in sorted(new - old, key=lambda item: item.value):
                changed.append(
                    f"+{first.object_id}:{relation.value}:{second.object_id}"
                )
            if (
                old != new
                and frozenset((first.object_id, second.object_id)) != target_pair
            ):
                leaked_pairs.add(tuple(sorted((first.object_id, second.object_id))))
    return tuple(sorted(changed)), len(leaked_pairs)


def normalized_native_after(
    before: Scene,
    observed: Scene,
    spec: InterventionSpec,
    limits: AI2ThorValidationTolerances,
) -> tuple[Scene | None, tuple[str, ...], float, float]:
    """Validate raw drift, then normalize only for the core verifier."""
    errors: list[str] = []
    maximum_geometry_residual = 0.0
    maximum_rotation_residual = 0.0
    for field in (
        "scene_id",
        "source",
        "coordinate_system",
        "generation_seed",
        "pinned_object_ids",
        "room_polygon_xy",
    ):
        if getattr(before, field) != getattr(observed, field):
            errors.append(f"native_{field}_changed")
    errors.extend(camera_residual_errors(before, observed, limits))

    before_ids = {obj.object_id for obj in before.objects}
    observed_ids = {obj.object_id for obj in observed.objects}
    if before_ids != observed_ids:
        errors.append("native_object_set_changed")
        return (
            None,
            tuple(sorted(set(errors))),
            maximum_geometry_residual,
            maximum_rotation_residual,
        )

    normalized_objects = []
    for expected in before.objects:
        current = observed.object_by_id(expected.object_id)
        for field in (
            "name",
            "category",
            "movable",
            "request_eligible",
            "support_object_id",
        ):
            if getattr(expected, field) != getattr(current, field):
                errors.append(
                    f"native_object_field_changed:{expected.object_id}:{field}"
                )

        geometry_residual = max(
            coordinate_residual(expected.position, current.position),
            coordinate_residual(expected.obb.center, current.obb.center),
            coordinate_residual(expected.obb.extent, current.obb.extent),
        )
        rotation_residual = max(
            quaternion_residual_degrees(expected.rotation, current.rotation),
            quaternion_residual_degrees(expected.obb.rotation, current.obb.rotation),
        )
        maximum_rotation_residual = max(
            maximum_rotation_residual,
            rotation_residual,
        )

        if expected.object_id == spec.subject_id:
            subject_geometry_residual = max(
                abs(expected.position.z - current.position.z),
                abs(expected.obb.center.z - current.obb.center.z),
                coordinate_residual(expected.obb.extent, current.obb.extent),
                abs(
                    (current.position.x - expected.position.x)
                    - (current.obb.center.x - expected.obb.center.x)
                ),
                abs(
                    (current.position.y - expected.position.y)
                    - (current.obb.center.y - expected.obb.center.y)
                ),
            )
            maximum_geometry_residual = max(
                maximum_geometry_residual,
                subject_geometry_residual,
            )
            if subject_geometry_residual > limits.stationary_geometry_residual_m:
                errors.append("native_subject_geometry_residual_exceeded")
            if rotation_residual > limits.rotation_residual_degrees:
                errors.append("native_subject_rotation_residual_exceeded")
            delta_x = current.position.x - expected.position.x
            delta_y = current.position.y - expected.position.y
            normalized_objects.append(
                expected.model_copy(
                    update={
                        "position": Vec3(
                            x=current.position.x,
                            y=current.position.y,
                            z=expected.position.z,
                        ),
                        "obb": expected.obb.model_copy(
                            update={
                                "center": Vec3(
                                    x=expected.obb.center.x + delta_x,
                                    y=expected.obb.center.y + delta_y,
                                    z=expected.obb.center.z,
                                )
                            }
                        ),
                        "views": current.views,
                    }
                )
            )
        else:
            maximum_geometry_residual = max(
                maximum_geometry_residual,
                geometry_residual,
            )
            if geometry_residual > limits.stationary_geometry_residual_m:
                errors.append(
                    f"native_stationary_geometry_residual_exceeded:{expected.object_id}"
                )
            if rotation_residual > limits.rotation_residual_degrees:
                errors.append(
                    f"native_stationary_rotation_residual_exceeded:{expected.object_id}"
                )
            normalized_objects.append(
                expected.model_copy(update={"views": current.views})
            )

    if errors:
        return (
            None,
            tuple(sorted(set(errors))),
            maximum_geometry_residual,
            maximum_rotation_residual,
        )
    return (
        before.model_copy(update={"objects": tuple(normalized_objects)}),
        (),
        maximum_geometry_residual,
        maximum_rotation_residual,
    )


__all__ = (
    "AI2ThorValidationRejected",
    "AI2ThorValidationTolerances",
    "CertifiedAI2ThorCaseRejected",
    "CertifiedAI2ThorPilotTolerances",
    "camera_residual_errors",
    "coordinate_residual",
    "floor_envelope_contract_errors",
    "floor_polygon",
    "is_finite_non_negative",
    "normalized_native_after",
    "observation_contract_errors",
    "pose_application_contract_errors",
    "position_distance",
    "quaternion_residual_degrees",
    "relation_graph_diff",
    "scene_without_views",
    "settlement_contract_errors",
    "vertical_contact_residual",
)
