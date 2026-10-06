"""Pure legacy-scene conversion to the canonical proxy problem."""

from __future__ import annotations

import json
import math
from inspect import get_annotations as _get_annotations
from typing import Literal, get_type_hints as _get_type_hints

from pydantic import Field, ValidationError

from spatialcf.domain.artifacts import StrictConvexCandidateCompilerConfigV2_7
from spatialcf.domain.base import CanonicalModel, UncertaintyBudgetV2
from spatialcf.domain.constraints import Relation
from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.request import InterventionSpec
from spatialcf.domain.scene import Scene
from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9

_ZERO_UNCERTAINTY = UncertaintyBudgetV2().model_dump(mode="json")

_VISIBILITY_DEFINITIONS = (
    {
        "metric_definition_id": "visibility:visible-surface-fraction",
        "metric_definition_version": "definition:1",
        "kind": "VISIBLE_FRACTION",
        "formula": "VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA",
        "area_measure": "CONTINUOUS_PIXEL_PLANE_AREA",
        "depth_policy": "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES",
    },
    {
        "metric_definition_id": "visibility:image-area-fraction",
        "metric_definition_version": "definition:1",
        "kind": "IMAGE_AREA_FRACTION",
        "formula": "VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA",
        "area_measure": "CONTINUOUS_PIXEL_PLANE_AREA",
        "depth_policy": "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES",
    },
    {
        "metric_definition_id": "visibility:truncated-fraction",
        "metric_definition_version": "definition:1",
        "kind": "TRUNCATED_FRACTION",
        "formula": "ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA",
        "area_measure": "CONTINUOUS_PIXEL_PLANE_AREA",
        "depth_policy": "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES",
    },
)

_BBOX_IMAGE_AREA_METRIC_ID = (
    "visibility:visible-clipped-projected-bounding-box-area-fraction"
)

_BBOX_IMAGE_AREA_METRIC_VERSION = "definition:2"

_BBOX_VISIBILITY_SEMANTICS_ID = "visibility-semantics:analytic-bbox-v1"

_BBOX_VISIBILITY_DEFINITIONS = (
    _VISIBILITY_DEFINITIONS[0],
    {
        "metric_definition_id": _BBOX_IMAGE_AREA_METRIC_ID,
        "metric_definition_version": _BBOX_IMAGE_AREA_METRIC_VERSION,
        "kind": "IMAGE_AREA_FRACTION",
        "formula": "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA",
        "area_measure": "CONTINUOUS_PIXEL_PLANE_AREA",
        "depth_policy": "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES",
    },
    _VISIBILITY_DEFINITIONS[2],
)


class _ProxyConversionError(ValueError):
    """One legacy case cannot be translated without inventing a fact."""

    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code


class _ProxyChallenge(CanonicalModel):
    case_id: str
    direction: str
    archetype: str
    expected_outcome: Literal["SAT", "UNSAT"]
    scene: Scene
    intervention: InterventionSpec
    scene_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intervention_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expectation_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _default_solver_config_base() -> ContinuousYawSolverConfigV2_9:
    """Return the frozen, bounded CPU policy used by challenge replay."""

    return ContinuousYawSolverConfigV2_9(
        candidate_config=StrictConvexCandidateCompilerConfigV2_7(
            max_domain_operations=2_000_000,
            max_so2_atomic_steps=2_000_000,
            max_candidate_cells=50_000,
        ),
        max_objective_partition_cells=100_000,
        target_optimality_gap=0.011,
    )


def _convert_bbox_proxy_scene(
    case: _ProxyChallenge,
) -> SemanticProblemV2_3:
    """Translate one case using the explicit projected bounding-box metric."""

    return _convert_proxy_scene(
        case,
        visibility_semantics_id=_BBOX_VISIBILITY_SEMANTICS_ID,
        visibility_definitions=_BBOX_VISIBILITY_DEFINITIONS,
        image_area_metric_definition_id=_BBOX_IMAGE_AREA_METRIC_ID,
        image_area_metric_definition_version=_BBOX_IMAGE_AREA_METRIC_VERSION,
    )


def _convert_target_proxy_scene(case: _ProxyChallenge) -> SemanticProblemV2_3:
    """Translate the target-proposal scene with its frozen analytic metric."""

    return _convert_proxy_scene(
        case,
        visibility_semantics_id="visibility-semantics:analytic-v1",
        visibility_definitions=_VISIBILITY_DEFINITIONS,
        image_area_metric_definition_id="visibility:image-area-fraction",
        image_area_metric_definition_version="definition:1",
    )


def _convert_proxy_scene(
    case: _ProxyChallenge,
    *,
    visibility_semantics_id: str,
    visibility_definitions: tuple[dict[str, str], ...],
    image_area_metric_definition_id: str,
    image_area_metric_definition_version: str,
) -> SemanticProblemV2_3:

    if type(case) is not _ProxyChallenge:
        raise TypeError("case must be an exact _ProxyChallenge")
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    camera = scene.camera_by_id(spec.camera_id)
    subject_id = _object_id(subject.object_id)
    reference_id = _object_id(spec.reference_id)
    camera_id = _camera_id(camera.camera_id)
    yaw_by_native_id = {
        item.object_id: _legacy_yaw(item.rotation, item.object_id)
        for item in scene.objects
    }
    camera_azimuth, camera_translation = _legacy_camera(camera.world_to_camera)
    workspace = _subject_anchor_workspace(
        scene, subject, yaw_by_native_id[subject.object_id]
    )

    objects = []
    geometries = []
    bodies = []
    for item in scene.objects:
        canonical_id = _object_id(item.object_id)
        support_surface_id = (
            _surface_id(item.support_object_id)
            if item.object_id == subject.object_id
            else None
        )
        objects.append(
            {
                "object_id": canonical_id,
                "category_id": _category_id(item.category),
                "movable": item.object_id == subject.object_id,
                "pose": {
                    "anchor_kind": "OBJECT_PIVOT",
                    "world_from_object": _yaw_transform(
                        item.position.x,
                        item.position.y,
                        item.position.z,
                        yaw_by_native_id[item.object_id],
                    ),
                },
                "support_assignment": (
                    {"availability": "KNOWN", "surface_id": support_surface_id}
                    if support_surface_id is not None
                    else {"availability": "NOT_APPLICABLE"}
                ),
            }
        )
        # The objective is closed over every subject/fixed-object pair.  That
        # requires RELATION geometry and a positive visibility gate for every
        # object, even when a legacy fixture omitted that object's baseline
        # view.  We preserve such omissions below (no fabricated observation),
        # allowing the fresh compiler to return a typed MISSING_FACT outcome.
        roles = ["COLLISION", "RELATION", "VISUAL"]
        if item.object_id == subject.object_id:
            roles.append("SUPPORT")
        for role in roles:
            geometries.append(_geometry(item, canonical_id, role))
        bodies.append(
            {
                "body_id": _body_id(item.object_id),
                "owner_object_id": canonical_id,
                "composition": "CLOSED_SOLID_UNION",
                "geometry_instance_ids": (_geometry_id(item.object_id, "collision"),),
            }
        )

    support_surface, support_owner_body = _support_surface(scene, subject)
    if support_owner_body is not None and support_owner_body not in {
        item["body_id"] for item in bodies
    }:
        floor_geometry, floor_body = _floor_body(scene, subject)
        geometries.append(floor_geometry)
        bodies.append(floor_body)

    # Canonical collision coverage is a semantic closure over every fixed
    # body, even when an implementation can later prove a pair Z-separated.
    collision_obstacles = [
        _body_id(item.object_id)
        for item in scene.objects
        if item.object_id != subject.object_id
    ]
    if support_owner_body not in collision_obstacles:
        collision_obstacles.append(support_owner_body)

    viewed_native_ids = tuple(
        sorted(
            item.object_id for item in scene.objects if camera.camera_id in item.views
        )
    )
    if (
        spec.subject_id not in viewed_native_ids
        or spec.reference_id not in viewed_native_ids
    ):
        raise _ProxyConversionError("MISSING_FACT:TARGET_BASELINE_VISIBILITY")
    observations = []
    for native_id in viewed_native_ids:
        view = scene.object_by_id(native_id).views[camera.camera_id]
        for metric_id, metric_version, value in (
            (
                "visibility:visible-surface-fraction",
                "definition:1",
                view.visible_fraction,
            ),
            (
                image_area_metric_definition_id,
                image_area_metric_definition_version,
                view.image_area_fraction,
            ),
            (
                "visibility:truncated-fraction",
                "definition:1",
                view.truncated_fraction,
            ),
        ):
            observations.append(
                {
                    "observation_id": f"observation:{native_id}:{metric_id.rsplit(':', 1)[-1]}",
                    "object_id": _object_id(native_id),
                    "camera_id": camera_id,
                    "metric_definition_id": metric_id,
                    "metric_definition_version": metric_version,
                    "normalized_value": value,
                    "normalized_lower_bound": value,
                    "normalized_upper_bound": value,
                }
            )

    target_before = Relation(spec.relation_before.value.upper())
    target_after = Relation(spec.relation_after.value.upper())
    target_axis = target_after.axis.value
    pair_weights = []
    for native_id in sorted(item.object_id for item in scene.objects):
        if native_id == subject.object_id:
            continue
        for axis in ("HORIZONTAL", "DEPTH", "DISTANCE"):
            if native_id == spec.reference_id and axis == target_axis:
                continue
            pair_weights.append(
                {
                    "key": {
                        "first_object_id": subject_id,
                        "second_object_id": _object_id(native_id),
                        "axis": axis,
                    },
                    "damage_weight": 1.0,
                }
            )

    constraint_ids = (
        "constraint:collision",
        "constraint:position-domain",
        "constraint:support",
        "constraint:target-relation",
        "constraint:visibility",
    )
    payload = {
        "schema_identity": {
            "schema_name": "semantic-problem",
            "schema_version": "2.3",
        },
        "scene": {
            "schema_identity": {
                "schema_name": "canonical-scene",
                "schema_version": "2.3",
            },
            "scene_id": f"scene:{case.case_id}",
            "coordinate_system": "RH_METERS_Z_UP",
            "objects": _known_facts(objects),
            "geometry_instances": _known_facts(geometries),
            "collision_bodies": _known_facts(bodies),
            "workspace_boundaries": _known_facts(
                [
                    {
                        "fact_id": "workspace:subject-anchor-locus",
                        "region_world_xy": _rectangle_region(*workspace),
                        "boundary_policy": "CLOSED",
                        "region_approximation": "EXACT",
                        "geometry_uncertainty": _ZERO_UNCERTAINTY,
                    }
                ]
            ),
            "known_free_spaces": {"availability": "NOT_APPLICABLE"},
            "support_surfaces": _known_facts([support_surface]),
            "cameras": _known_facts(
                [
                    {
                        "camera_id": camera_id,
                        "width_px": camera.width,
                        "height_px": camera.height,
                        "intrinsics_row_major": camera.intrinsics,
                        "world_to_camera": {
                            "kind": "UPRIGHT_WORLD_TO_CAMERA",
                            "azimuth_radians": camera_azimuth,
                            "translation": camera_translation,
                        },
                        "matrix_layout": "ROW_MAJOR",
                        "camera_axes": "X_RIGHT_Y_DOWN_Z_FORWARD",
                        "pixel_convention": "CENTER_AT_HALF",
                        "depth_convention": "POSITIVE_Z_FORWARD",
                        "near_clip_m": 0.01,
                        "far_clip_m": 1000.0,
                        "distortion_model": "NONE",
                        "brown_conrady_coefficients": None,
                        "calibration_uncertainty": _ZERO_UNCERTAINTY,
                    }
                ]
            ),
            # A legacy case may carry exact baselines for only a subset of
            # objects.  Preserve that as a sound INNER_BOUND fact set; do not
            # invent normalized observations for the missing objects.
            "baseline_observations": _known_facts(
                observations,
                completeness=(
                    "EXACT"
                    if len(viewed_native_ids) == len(scene.objects)
                    else "INNER_BOUND"
                ),
            ),
        },
        "constraints": {
            "schema_identity": {
                "schema_name": "canonical-constraint-set",
                "schema_version": "2.0",
            },
            "constraint_set_id": "constraint-set:competition-challenge-v2.9",
            "allowed_edit": {
                "constraint_id": "constraint:allowed-edit",
                "subject_id": subject_id,
                "translation_axes": ("X", "Y"),
                "immutable_fields": (
                    "SUBJECT_Z",
                    "SUBJECT_ROTATION",
                    "OTHER_OBJECTS",
                    "CAMERAS",
                ),
            },
            "position_domain": {
                "constraint_id": "constraint:position-domain",
                "subject_id": subject_id,
                "workspace_fact_ids": ("workspace:subject-anchor-locus",),
                "workspace_aggregation": "INTERSECTION",
                "known_free_space_fact_ids": (),
                "known_free_space_aggregation": None,
                "region_interpretation": "SUBJECT_ANCHOR_LOCUS",
                "subject_occupancy_body_ids": (),
                "subject_occupancy_aggregation": None,
                "boundary_policy": "CLOSED",
                "required_completeness": ("EXACT",),
                "minimum_boundary_clearance_m": 0.0,
            },
            "collision_constraints": (
                {
                    "constraint_id": "constraint:collision",
                    "subject_body_ids": (_body_id(subject.object_id),),
                    "obstacle_body_ids": tuple(sorted(collision_obstacles)),
                    "clearance_metric": (
                        "SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE"
                    ),
                    "boundary_policy": "CLOSED",
                    "minimum_clearance_m": 0.0,
                    "support_contact_exceptions": (),
                },
            ),
            "support_constraints": (
                {
                    "constraint_id": "constraint:support",
                    "supported_object_id": subject_id,
                    "surface_id": support_surface["surface_id"],
                    "subject_contact_geometry_ids": (
                        _geometry_id(subject.object_id, "support"),
                    ),
                    "contact_feature": "LOWEST_FACE_ALONG_SURFACE_NORMAL",
                    "contact_aggregation": "UNION_ALL_SELECTED_FEATURES",
                    "contact_gap_min_m": 0.0,
                    "contact_gap_max_m": 0.0,
                    "overlap_metric": ("PROJECTED_CONTACT_UNION_INTERSECTION_AREA"),
                    "minimum_overlap_area_m2": 0.0,
                    "stability_metric": (
                        "FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET"
                    ),
                    "stability_margin_m": 0.0,
                    "boundary_policy": "CLOSED",
                    "assignment_policy": "EXACT_SURFACE",
                },
            ),
            "visibility_constraints": (
                {
                    "constraint_id": "constraint:visibility",
                    "visibility_semantics_id": visibility_semantics_id,
                    "camera_id": camera_id,
                    "query_object_ids": tuple(
                        _object_id(item.object_id)
                        for item in sorted(
                            scene.objects, key=lambda value: value.object_id
                        )
                    ),
                    "occluder_geometry_ids": (),
                    "visible_fraction_metric_definition_id": (
                        "visibility:visible-surface-fraction"
                    ),
                    "visible_fraction_metric_definition_version": "definition:1",
                    "image_area_metric_definition_id": (
                        image_area_metric_definition_id
                    ),
                    "image_area_metric_definition_version": (
                        image_area_metric_definition_version
                    ),
                    "truncated_fraction_metric_definition_id": (
                        "visibility:truncated-fraction"
                    ),
                    "truncated_fraction_metric_definition_version": "definition:1",
                    "mask_policy": "FULL_OBJECT",
                    "occluder_soundness_policy": "EXACT_OR_OUTER_SHAPE_BOUND",
                    "minimum_visible_fraction": 0.20,
                    "minimum_image_area_fraction": 0.0025,
                    "maximum_truncated_fraction": 0.50,
                    "threshold_boundary_policy": "CLOSED",
                    "accepted_baseline_completeness": ("EXACT",),
                },
            ),
            "target_relation": {
                "constraint_id": "constraint:target-relation",
                "subject_id": subject_id,
                "reference_id": reference_id,
                "camera_id": camera_id,
                "relation_before": target_before.value,
                "relation_after": target_after.value,
                "semantics_id": "relation-semantics:competition-v1",
            },
        },
        "relation_semantics": _relation_semantics(camera.width),
        "visibility_semantics": {
            "schema_identity": {
                "schema_name": "visibility-semantics",
                "schema_version": "2.0",
            },
            "semantics_id": visibility_semantics_id,
            "definitions": visibility_definitions,
        },
        "objective": _objective(
            camera_id=camera_id,
            subject_id=subject_id,
            pair_weights=pair_weights,
            target_axis=target_axis,
            constraint_ids=constraint_ids,
        ),
        "numeric_policy": {
            "linear_tolerance_m": 0.0,
            "area_tolerance_m2": 0.0,
            "angular_tolerance_rad": 0.0,
            "pixel_tolerance_px": 0.0,
            "fraction_tolerance": 0.0,
        },
    }
    try:
        return SemanticProblemV2_3.model_validate_json(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            strict=True,
        )
    except ValidationError as error:
        raise _ProxyConversionError(
            "INVALID_INPUT:LEGACY_CHALLENGE_CANONICALIZATION"
        ) from error


def _object_id(native_id: str) -> str:
    return f"object:{native_id}"


def _body_id(native_id: str) -> str:
    return f"body:{native_id}"


def _geometry_id(native_id: str, role: str) -> str:
    return f"geometry:{native_id}:{role.lower()}"


def _surface_id(native_id: str | None) -> str:
    return "surface:floor" if native_id is None else f"surface:{native_id}:top"


def _camera_id(native_id: str) -> str:
    return f"camera:{native_id}"


def _category_id(category: str) -> str:
    safe = "-".join(category.strip().lower().replace("_", "-").split())
    if not safe:
        raise _ProxyConversionError("MISSING_FACT:OBJECT_CATEGORY")
    return f"category:{safe}"


def _legacy_yaw(quaternion, object_id: str) -> float:
    if quaternion.x != 0.0 or quaternion.y != 0.0:
        raise _ProxyConversionError(f"UNSUPPORTED_MODEL:OBJECT_PITCH_ROLL:{object_id}")
    norm = math.hypot(quaternion.z, quaternion.w)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise _ProxyConversionError(f"INVALID_INPUT:OBJECT_QUATERNION:{object_id}")
    return 0.0 if quaternion.z == 0.0 else 2.0 * math.atan2(quaternion.z, quaternion.w)


def _legacy_camera(matrix: tuple[float, ...]) -> tuple[float, dict[str, float]]:
    cosine, negative_sine = matrix[0], matrix[1]
    sine, second_cosine = matrix[8], matrix[9]
    if math.isclose(math.hypot(sine, second_cosine), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise _ProxyConversionError("MISSING_FACT:COMPLETE_UPRIGHT_CAMERA_DEPTH_BASIS")
    expected = (
        (matrix[2], 0.0),
        (matrix[4], 0.0),
        (matrix[5], 0.0),
        (matrix[6], 1.0),
        (matrix[10], 0.0),
        (matrix[12], 0.0),
        (matrix[13], 0.0),
        (matrix[14], 0.0),
        (matrix[15], 1.0),
        (negative_sine, -sine),
        (cosine, second_cosine),
    )
    if any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9)
        for actual, wanted in expected
    ) or not math.isclose(math.hypot(sine, cosine), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise _ProxyConversionError("UNSUPPORTED_MODEL:CAMERA_NOT_EXACT_UPRIGHT")
    angle = 0.0 if sine == 0.0 else math.atan2(sine, cosine)
    # Legacy scenes used an image-Y-up camera row. Canonical v2 freezes
    # X-right/Y-down/Z-forward, so the vertical row and its translation are
    # negated while the horizontal/depth plane is preserved exactly.
    return angle, {"x": matrix[3], "y": -matrix[7], "z": matrix[11]}


def _yaw_transform(x: float, y: float, z: float, yaw: float) -> dict[str, object]:
    return {
        "kind": "DIRECTED_YAW_INTERVAL",
        "translation": {"x": x, "y": y, "z": z},
        "yaw_radians": yaw,
    }


def _geometry(item, canonical_id: str, role: str) -> dict[str, object]:
    return {
        "geometry_id": _geometry_id(item.object_id, role),
        "owner_object_id": canonical_id,
        "role": role,
        "anchor_from_geometry": _yaw_transform(0.0, 0.0, 0.0, 0.0),
        "approximation": "EXACT",
        "uncertainty": _ZERO_UNCERTAINTY,
        "shape": {
            "shape_type": "UPRIGHT_BOX_3D",
            "origin_convention": "CENTERED_AT_GEOMETRY_FRAME",
            "size_m": item.obb.extent.model_dump(mode="json"),
        },
    }


def _subject_anchor_workspace(
    scene: Scene, subject, yaw: float
) -> tuple[float, float, float, float]:
    points = tuple((item.x, item.y) for item in scene.room_polygon_xy)
    xs = sorted({item[0] for item in points})
    ys = sorted({item[1] for item in points})
    if len(points) != 4 or len(xs) != 2 or len(ys) != 2:
        raise _ProxyConversionError("UNSUPPORTED_MODEL:NON_RECTANGULAR_ROOM")
    half_x = subject.obb.extent.x / 2.0
    half_y = subject.obb.extent.y / 2.0
    radius_x = math.nextafter(
        abs(math.cos(yaw)) * half_x + abs(math.sin(yaw)) * half_y,
        math.inf,
    )
    radius_y = math.nextafter(
        abs(math.sin(yaw)) * half_x + abs(math.cos(yaw)) * half_y,
        math.inf,
    )
    bounds = (
        math.nextafter(xs[0] + radius_x, math.inf),
        math.nextafter(ys[0] + radius_y, math.inf),
        math.nextafter(xs[1] - radius_x, -math.inf),
        math.nextafter(ys[1] - radius_y, -math.inf),
    )
    if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
        raise _ProxyConversionError("UNSAT:EMPTY_SUBJECT_ANCHOR_LOCUS")
    return bounds


def _rectangle_region(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> dict[str, object]:
    return {
        "components": (
            {
                "exterior": {
                    "winding": "COUNTERCLOCKWISE",
                    "vertices": (
                        {"x": min_x, "y": min_y},
                        {"x": max_x, "y": min_y},
                        {"x": max_x, "y": max_y},
                        {"x": min_x, "y": max_y},
                    ),
                },
                "holes": (),
            },
        )
    }


def _known_facts(
    values: list[dict[str, object]], *, completeness: str = "EXACT"
) -> dict[str, object]:
    return {
        "availability": "KNOWN",
        "values": tuple(values),
        "inner_values": None,
        "outer_values": None,
        "completeness": completeness,
        "uncertainty": _ZERO_UNCERTAINTY,
    }


def _support_surface(scene: Scene, subject) -> tuple[dict[str, object], str]:
    support_native_id = subject.support_object_id
    if support_native_id is None:
        min_x = min(point.x for point in scene.room_polygon_xy)
        max_x = max(point.x for point in scene.room_polygon_xy)
        min_y = min(point.y for point in scene.room_polygon_xy)
        max_y = max(point.y for point in scene.room_polygon_xy)
        return (
            {
                "surface_id": "surface:floor",
                "owner_object_id": None,
                "supporting_body_id": "body:floor",
                "anchor_from_surface": _yaw_transform(0.0, 0.0, 0.0, 0.0),
                "normal_in_anchor": {"x": 0.0, "y": 0.0, "z": 1.0},
                "region_uv": _rectangle_region(min_x, min_y, max_x, max_y),
                "region_approximation": "EXACT",
                "boundary_policy": "CLOSED",
                "geometry_uncertainty": _ZERO_UNCERTAINTY,
            },
            "body:floor",
        )
    owner = scene.object_by_id(support_native_id)
    if _legacy_yaw(owner.rotation, owner.object_id) != 0.0:
        raise _ProxyConversionError("UNSUPPORTED_MODEL:ROTATED_SUPPORT_SURFACE")
    half_x = owner.obb.extent.x / 2.0
    half_y = owner.obb.extent.y / 2.0
    return (
        {
            "surface_id": _surface_id(owner.object_id),
            "owner_object_id": _object_id(owner.object_id),
            "supporting_body_id": _body_id(owner.object_id),
            "anchor_from_surface": _yaw_transform(
                0.0, 0.0, owner.obb.extent.z / 2.0, 0.0
            ),
            "normal_in_anchor": {"x": 0.0, "y": 0.0, "z": 1.0},
            "region_uv": _rectangle_region(-half_x, -half_y, half_x, half_y),
            "region_approximation": "EXACT",
            "boundary_policy": "CLOSED",
            "geometry_uncertainty": _ZERO_UNCERTAINTY,
        },
        _body_id(owner.object_id),
    )


def _floor_body(scene: Scene, subject) -> tuple[dict[str, object], dict[str, object]]:
    min_x = min(point.x for point in scene.room_polygon_xy)
    max_x = max(point.x for point in scene.room_polygon_xy)
    min_y = min(point.y for point in scene.room_polygon_xy)
    max_y = max(point.y for point in scene.room_polygon_xy)
    depth = max(subject.obb.extent.z, 0.1)
    geometry = {
        "geometry_id": "geometry:floor:collision",
        "owner_object_id": None,
        "role": "COLLISION",
        "anchor_from_geometry": _yaw_transform(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            -depth / 2.0,
            0.0,
        ),
        "approximation": "EXACT",
        "uncertainty": _ZERO_UNCERTAINTY,
        "shape": {
            "shape_type": "UPRIGHT_BOX_3D",
            "origin_convention": "CENTERED_AT_GEOMETRY_FRAME",
            "size_m": {"x": max_x - min_x, "y": max_y - min_y, "z": depth},
        },
    }
    body = {
        "body_id": "body:floor",
        "owner_object_id": None,
        "composition": "CLOSED_SOLID_UNION",
        "geometry_instance_ids": ("geometry:floor:collision",),
    }
    return geometry, body


def _relation_semantics(width_px: int) -> dict[str, object]:
    threshold_x = width_px * 0.05
    definitions = []
    for relation, measurement, comparator, threshold, unit, point in (
        (
            "LEFT",
            "PROJECTED_CENTER_DELTA_X",
            "LESS_THAN",
            -threshold_x,
            "PIXEL",
            "RELATION_GEOMETRY_VOLUME_CENTROID",
        ),
        (
            "RIGHT",
            "PROJECTED_CENTER_DELTA_X",
            "GREATER_THAN",
            threshold_x,
            "PIXEL",
            "RELATION_GEOMETRY_VOLUME_CENTROID",
        ),
        (
            "FRONT",
            "CAMERA_DEPTH_DELTA",
            "LESS_THAN",
            -0.20,
            "METRE",
            "RELATION_GEOMETRY_VOLUME_CENTROID",
        ),
        (
            "BEHIND",
            "CAMERA_DEPTH_DELTA",
            "GREATER_THAN",
            0.20,
            "METRE",
            "RELATION_GEOMETRY_VOLUME_CENTROID",
        ),
        ("NEAR", "SHAPE_GAP_XY", "LESS_THAN", 0.50, "METRE", None),
        ("FAR", "SHAPE_GAP_XY", "GREATER_THAN", 1.50, "METRE", None),
    ):
        definitions.append(
            {
                "relation": relation,
                "measurement": measurement,
                "comparator": comparator,
                "threshold": threshold,
                "unit": unit,
                "representative_point": point,
                "operand_order": "FIRST_MINUS_SECOND",
                "boundary_policy": "CLOSED",
                "tolerance": 0.0,
                "tolerance_policy": ("SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET"),
                "requires_both_visible": True,
            }
        )
    return {
        "schema_identity": {
            "schema_name": "relation-semantics",
            "schema_version": "2.0",
        },
        "semantics_id": "relation-semantics:competition-v1",
        "definitions": tuple(definitions),
    }


def _objective(
    *,
    camera_id: str,
    subject_id: str,
    pair_weights: list[dict[str, object]],
    target_axis: str,
    constraint_ids: tuple[str, ...],
) -> dict[str, object]:
    target_unit = "PIXEL" if target_axis == "HORIZONTAL" else "METRE"
    component_by_constraint = {
        "constraint:collision": (("COLLISION_CLEARANCE", "METRE"),),
        "constraint:position-domain": (("POSITION_BOUNDARY_CLEARANCE", "METRE"),),
        "constraint:support": (
            ("SUPPORT_CONTACT_GAP_LOWER_MARGIN", "METRE"),
            ("SUPPORT_CONTACT_GAP_UPPER_MARGIN", "METRE"),
            ("SUPPORT_OVERLAP_AREA_MARGIN", "SQUARE_METRE"),
            ("SUPPORT_STABILITY_INSET_MARGIN", "METRE"),
        ),
        "constraint:target-relation": (
            ("TARGET_RELATION_THRESHOLD_MARGIN", target_unit),
        ),
        "constraint:visibility": (
            ("VISIBILITY_VISIBLE_FRACTION_MARGIN", "FRACTION"),
            ("VISIBILITY_IMAGE_AREA_FRACTION_MARGIN", "FRACTION"),
            ("VISIBILITY_TRUNCATED_FRACTION_MARGIN", "FRACTION"),
        ),
    }

    targets = tuple(
        {
            "constraint_id": constraint_id,
            "target_slack": 0.0,
            "component_aggregation": "MIN_NORMALIZED_COMPONENT_MARGIN",
            "components": tuple(
                {"kind": kind, "unit": unit, "normalizer": 1.0}
                for kind, unit in component_by_constraint[constraint_id]
            ),
            "importance": 1.0,
        }
        for constraint_id in constraint_ids
    )
    return {
        "schema_identity": {
            "schema_name": "objective-spec",
            "schema_version": "2.0",
        },
        "objective_id": "objective:competition-challenge-v2.9",
        "mode": "PRODUCTION",
        "translation": {
            "weight": 1.0,
            "normalizer_m": 1.0,
            "metric": "EUCLIDEAN_L2_WORLD_XY_METRE",
        },
        "relation_damage": {
            "weight": 0.25,
            "normalizer": 1.0,
            "metric": "PAIR_AXIS_SATISFIED_LABEL_SET_CHANGED_INDICATOR",
            "aggregation": "WEIGHTED_SUM",
            "evaluation_camera_id": camera_id,
            "pair_axis_weights": tuple(pair_weights),
        },
        "visibility_change": {
            "weight": 0.50,
            "normalizer": 1.0,
            "metric": "ABSOLUTE_NORMALIZED_METRIC_DELTA",
            "aggregation": "WEIGHTED_SUM",
            "object_camera_weights": (
                {
                    "key": {
                        "object_id": subject_id,
                        "camera_id": camera_id,
                        "metric_definition_id": ("visibility:visible-surface-fraction"),
                        "metric_definition_version": "definition:1",
                    },
                    "change_weight": 1.0,
                },
            ),
        },
        "safety_margin": {
            "weight": 0.75,
            "normalizer": 1.0,
            "aggregation": {
                "kind": "SUM_NORMALIZED_DEFICIT",
                "targets": targets,
            },
        },
        "tie_break": (
            "TRANSLATION",
            "RELATION_DAMAGE",
            "VISIBILITY_CHANGE",
            "SAFETY_PENALTY",
            "DELTA_X",
            "DELTA_Y",
        ),
    }


_ProxyChallenge.model_rebuild()
_ProxyChallenge.__annotations__ = {
    name: hint
    for name, hint in _get_type_hints(_ProxyChallenge, include_extras=True).items()
    if name in _get_annotations(_ProxyChallenge)
}

# Preserve existing import and pickle identities.
_ProxyConversionError.__module__ = "spatialcf.generation.planning.problem"
_ProxyChallenge.__module__ = "spatialcf.generation.planning.problem"
_default_solver_config_base.__module__ = "spatialcf.generation.planning.problem"
_convert_bbox_proxy_scene.__module__ = "spatialcf.generation.planning.problem"
_convert_target_proxy_scene.__module__ = "spatialcf.generation.planning.problem"
_convert_proxy_scene.__module__ = "spatialcf.generation.planning.problem"
_object_id.__module__ = "spatialcf.generation.planning.problem"
_body_id.__module__ = "spatialcf.generation.planning.problem"
_geometry_id.__module__ = "spatialcf.generation.planning.problem"
_surface_id.__module__ = "spatialcf.generation.planning.problem"
_camera_id.__module__ = "spatialcf.generation.planning.problem"
_category_id.__module__ = "spatialcf.generation.planning.problem"
_legacy_yaw.__module__ = "spatialcf.generation.planning.problem"
_legacy_camera.__module__ = "spatialcf.generation.planning.problem"
_yaw_transform.__module__ = "spatialcf.generation.planning.problem"
_geometry.__module__ = "spatialcf.generation.planning.problem"
_subject_anchor_workspace.__module__ = "spatialcf.generation.planning.problem"
_rectangle_region.__module__ = "spatialcf.generation.planning.problem"
_known_facts.__module__ = "spatialcf.generation.planning.problem"
_support_surface.__module__ = "spatialcf.generation.planning.problem"
_floor_body.__module__ = "spatialcf.generation.planning.problem"
_relation_semantics.__module__ = "spatialcf.generation.planning.problem"
_objective.__module__ = "spatialcf.generation.planning.problem"
