"""Current source-only proxy preparation and projection authority."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from spatialcf.adapters.ai2thor import (
    AI2ThorReceptacleSurfacePatch,
    _receptacle_scene_sha256,
)
from spatialcf.domain.models import OBB, InterventionSpec, Scene
from spatialcf.domain.v2.base import FactCompletenessV2, UncertaintyBudgetV2, V2Model
from spatialcf.domain.v2.constraints import RelationV2
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import (
    StrictConvexCandidateCompilerConfigV2_7,
)
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawSolverConfigV2_9,
)
from spatialcf.generation._internal.evidence.surface import (
    ReceptacleSurfacePatch,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    verify_source_surface_evidence,
)
from spatialcf.generation._internal.planning.models import (
    _PROXY_POLICY_SHA256,
    CollisionDelegation,
    EndpointWorkspace,
    ProxyBinding,
    ProxyBundle,
    SubjectPlacementFact,
)
from spatialcf.generation.capture.models import CompetitionNativeSourceCaptureV2_9

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
_NON_SUPPORT_COLLISION_MARGIN_M = 0.01
_TARGET_PROPOSAL_POLICY_SHA256 = hashlib.sha256(
    b"spatialcf.competition-native-scene-proxy.v2.9.1\0"
    b"obb-frame;explicit-endpoint-only-workspace;l1-radius-prune;"
    b"subject-bottom-support-plane;support-solid-top-clamp;"
    b"fixed-non-support-obb-margin-1cm;existing-clearance-max"
).hexdigest()


class _ProxyConversionError(ValueError):
    """One legacy case cannot be translated without inventing a fact."""

    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code


class _ProxyChallenge(V2Model):
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

    target_before = RelationV2(spec.relation_before.value.upper())
    target_after = RelationV2(spec.relation_after.value.upper())
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


@dataclass(frozen=True, slots=True)
class _PreparedProxy:
    scene: Scene
    intervention: InterventionSpec
    case_id: str
    base_problem: SemanticProblemV2_3
    legacy_scene_sha256: str
    intervention_sha256: str
    collision_proxies: tuple[tuple[str, OBB], ...]
    minimum_margin_native_object_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PreparedCurrentProxy:
    base: _PreparedProxy
    source_capture: CompetitionNativeSourceCaptureV2_9
    source_surface_evidence: SourceSurfaceEvidence
    subject_surface_evidence: SubjectSurfaceEvidence


def default_solver_config() -> ContinuousYawSolverConfigV2_9:
    """Return the bounded CPU policy for non-evidence native proxy solves.

    Native proxy cells retain conservative non-target relation-damage brackets
    that can contribute up to 0.75 plus directed-rounding residue to the
    certified objective gap.  The looser target changes only the accepted
    epsilon-optimality claim; every hard constraint and every shared resource
    cap remains identical to challenge replay.
    """

    payload = _default_solver_config_base().model_dump(mode="python", warnings="error")
    payload["target_optimality_gap"] = 1.0
    return ContinuousYawSolverConfigV2_9.model_validate(payload, strict=True)


def default_planning_workspace(
    scene: Scene,
    subject_object_id: str,
) -> EndpointWorkspace:
    """Return the exact rectangular subject-anchor locus used by conversion."""

    checked_scene = _strict_legacy(scene, Scene, "scene")
    if type(subject_object_id) is not str or not subject_object_id:
        raise TypeError("subject_object_id must be a non-empty exact string")
    subject = checked_scene.object_by_id(subject_object_id)
    points = tuple((item.x, item.y) for item in checked_scene.room_polygon_xy)
    xs = sorted({item[0] for item in points})
    ys = sorted({item[1] for item in points})
    if len(points) != 4 or len(xs) != 2 or len(ys) != 2:
        raise ValueError("native planning requires one rectangular room polygon")
    yaw = _obb_yaw(subject.obb)
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
    return EndpointWorkspace(
        min_x_m=math.nextafter(xs[0] + radius_x, math.inf),
        min_y_m=math.nextafter(ys[0] + radius_y, math.inf),
        max_x_m=math.nextafter(xs[1] - radius_x, -math.inf),
        max_y_m=math.nextafter(ys[1] - radius_y, -math.inf),
    )


def _verified_proxy_inputs(
    scene: Scene,
    intervention: InterventionSpec,
    source_capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    *,
    case_id: str,
) -> tuple[
    Scene,
    InterventionSpec,
    CompetitionNativeSourceCaptureV2_9,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
]:
    checked_capture = _strict_v2(
        source_capture,
        CompetitionNativeSourceCaptureV2_9,
        "source_capture",
    )
    verified_source_evidence = verify_source_surface_evidence(
        checked_capture,
        source_surface_evidence,
    )
    checked_evidence = _strict_v2(
        subject_surface_evidence,
        SubjectSurfaceEvidence,
        "subject_surface_evidence",
    )
    checked_scene = _strict_legacy(scene, Scene, "scene")
    checked_intervention = _strict_legacy(
        intervention,
        InterventionSpec,
        "intervention",
    )
    if type(case_id) is not str or not case_id:
        raise TypeError("case_id must be a non-empty exact string")
    source_subjects = tuple(
        item
        for item in verified_source_evidence.subjects
        if item.subject_object_id == checked_intervention.subject_id
    )
    if len(source_subjects) != 1 or source_subjects[0] != checked_evidence:
        raise ValueError(
            "subject surface evidence does not bind verified capture facts"
        )
    if checked_scene != checked_capture.scene:
        raise ValueError("native proxy scene does not bind verified capture facts")
    if (
        checked_evidence.subject_object_id != checked_intervention.subject_id
        or checked_evidence.support_object_id
        != checked_scene.object_by_id(checked_intervention.subject_id).support_object_id
        or checked_evidence.subject_object_id
        not in {item.object_id for item in checked_scene.objects}
        or checked_evidence.support_object_id
        not in {item.object_id for item in checked_scene.objects}
    ):
        raise ValueError("subject surface evidence does not bind the native proxy")
    native_patches = tuple(
        AI2ThorReceptacleSurfacePatch(
            x_min=item.x_min,
            x_max=item.x_max,
            native_y=item.native_y,
            z_min=item.z_min,
            z_max=item.z_max,
        )
        for item in checked_evidence.patches
    )
    if checked_evidence.scene_sha256 != _receptacle_scene_sha256(
        checked_scene,
        native_patches,
    ):
        raise ValueError("subject surface evidence does not bind source geometry")
    return (
        checked_scene,
        checked_intervention,
        checked_capture,
        verified_source_evidence,
        checked_evidence,
    )


def _prepare_proxy(
    scene: Scene,
    intervention: InterventionSpec,
    source_capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    *,
    case_id: str,
) -> _PreparedCurrentProxy:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        (
            checked_scene,
            checked_intervention,
            checked_capture,
            verified_source_evidence,
            checked_evidence,
        ) = _verified_proxy_inputs(
            scene,
            intervention,
            source_capture,
            source_surface_evidence,
            subject_surface_evidence,
            case_id=case_id,
        )
        base = _prepare_base_proxy(
            checked_scene,
            checked_intervention,
            case_id=case_id,
            bbox_visibility=True,
        )
    return _PreparedCurrentProxy(
        base=base,
        source_capture=checked_capture,
        source_surface_evidence=verified_source_evidence,
        subject_surface_evidence=checked_evidence,
    )


def _project_patch_payload(
    prepared: _PreparedProxy,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    endpoint_workspace: EndpointWorkspace,
    *,
    patch_index: int,
) -> tuple[SemanticProblemV2_3, dict[str, object]]:
    """Return a version-neutral patch projection without public artifacts."""

    if type(prepared) is not _PreparedProxy:
        raise TypeError("prepared native proxy context must be exact")
    if type(patch_index) is not int or patch_index < 0:
        raise TypeError("patch_index must be a non-negative exact integer")
    if patch_index >= len(subject_surface_evidence.patches):
        raise ValueError("patch index escaped subject surface evidence")
    selected_patch = subject_surface_evidence.patches[patch_index]
    base_problem, base_payload = _project_base_payload(
        prepared,
        endpoint_workspace,
    )
    problem = _project_direct_support_patch(
        base_problem,
        selected_patch,
        subject_object_id=_object_id(prepared.intervention.subject_id),
    )
    return problem, {
        **base_payload,
        "binding_version": "competition-native-proxy-binding:2.9.4",
        "proxy_scope": (
            "CAPTURE_PATCH_CPU_NONDELEGATED_NATIVE_AUDIT_AND_BBOX_VISIBILITY"
        ),
        "support_plane_policy": (
            "SUBJECT_BOTTOM_PLANE_SINGLE_NATIVE_PATCH_AND_SUPPORT_SOLID_TOP_CLAMP"
        ),
        "proxy_policy_sha256": _PROXY_POLICY_SHA256,
        "semantic_problem_sha256": problem.semantic_problem_sha256,
        "source_capture_sha256": subject_surface_evidence.source_capture_sha256,
        "runtime_identity_sha256": subject_surface_evidence.runtime_identity_sha256,
        "scene_sha256": subject_surface_evidence.scene_sha256,
        "positions_sha256": subject_surface_evidence.positions_sha256,
        "spawn_map_source_sha256": (subject_surface_evidence.spawn_map_source_sha256),
        "placement_sha256": subject_surface_evidence.placement_sha256,
        "surface_evidence_sha256": source_surface_evidence.surface_evidence_sha256,
        "subject_surface_evidence_sha256": (
            subject_surface_evidence.subject_surface_evidence_sha256
        ),
        "patch_index": patch_index,
        "patch_sha256": selected_patch.patch_sha256,
        "selected_patch": selected_patch,
    }


def _delegate_runtime_collisions(
    semantic_problem: SemanticProblemV2_3,
    binding_payload: dict[str, object],
    prepared: _PreparedProxy,
) -> tuple[SemanticProblemV2_3, dict[str, object]]:
    included_roster = _binding_roster(
        binding_payload,
        "included_collision_native_object_ids",
    )
    excluded_roster = _binding_roster(
        binding_payload,
        "excluded_collision_native_object_ids",
    )
    clearance_roster = _binding_roster(
        binding_payload,
        "clearance_overlay_native_object_ids",
    )
    minimum_margin_roster = _binding_roster(
        binding_payload,
        "minimum_margin_native_object_ids",
    )
    source_candidates = _runtime_collision_delegation_candidates(prepared)
    delegated = tuple(sorted(source_candidates & set(included_roster)))
    problem = _without_delegated_collision_bodies(
        semantic_problem,
        delegated,
    )
    included = tuple(item for item in included_roster if item not in delegated)
    fixed_ids = {item.object_id for item in prepared.scene.objects} - {
        prepared.intervention.subject_id
    }
    if (set(included) | set(excluded_roster) | set(delegated)) != fixed_ids:
        raise RuntimeError("runtime collision authority partition is not closed")
    return problem, {
        **binding_payload,
        "included_collision_native_object_ids": included,
        "clearance_overlay_native_object_ids": tuple(
            item for item in clearance_roster if item not in delegated
        ),
        "minimum_margin_native_object_ids": tuple(
            item for item in minimum_margin_roster if item not in delegated
        ),
        "runtime_collision_delegated_native_object_ids": delegated,
    }


def _binding_roster(
    payload: dict[str, object],
    field_name: str,
) -> tuple[str, ...]:
    values = payload.get(field_name)
    if type(values) is not tuple or any(type(item) is not str for item in values):
        raise RuntimeError(f"internal proxy payload {field_name} is not exact")
    return values


def _project_proxy(
    prepared: _PreparedCurrentProxy,
    endpoint_workspace: EndpointWorkspace,
    *,
    patch_index: int,
) -> ProxyBundle:
    """Project one runtime-delegated patch with bbox visibility semantics."""

    if type(prepared) is not _PreparedCurrentProxy:
        raise TypeError("prepared bbox proxy context must be exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        problem, payload = _project_patch_payload(
            prepared.base,
            prepared.source_surface_evidence,
            prepared.subject_surface_evidence,
            endpoint_workspace,
            patch_index=patch_index,
        )
        problem, payload = _delegate_runtime_collisions(
            problem,
            payload,
            prepared.base,
        )
        binding = ProxyBinding(
            **{
                **payload,
                "binding_version": "competition-native-proxy-binding:2.9.4",
                "proxy_scope": (
                    "CAPTURE_PATCH_CPU_NONDELEGATED_NATIVE_AUDIT_AND_BBOX_VISIBILITY"
                ),
                "proxy_policy_sha256": _PROXY_POLICY_SHA256,
                "semantic_problem_sha256": problem.semantic_problem_sha256,
            }
        )
        return ProxyBundle(
            semantic_problem=problem,
            binding=binding,
        )


def build_proxy_bundle(
    scene: Scene,
    intervention: InterventionSpec,
    *,
    workspace: EndpointWorkspace,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    placement: SubjectPlacementFact,
    collision_delegation: CollisionDelegation,
) -> ProxyBundle:
    """Build the current patch proxy without running a platform."""

    if type(placement) is not SubjectPlacementFact:
        raise TypeError("proxy placement must be exact")
    if type(collision_delegation) is not CollisionDelegation:
        raise TypeError("proxy collision delegation must be exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_placement = _strict_v2(
            placement,
            SubjectPlacementFact,
            "placement",
        )
        prepared = _prepare_proxy(
            scene,
            intervention,
            checked_placement.source_capture,
            source_surface_evidence,
            subject_surface_evidence,
            case_id=collision_delegation.case_id,
        )
        subject_placement = checked_placement.subject_placement
        checked_evidence = prepared.subject_surface_evidence
        if (
            checked_placement.source_capture != prepared.source_capture
            or subject_placement.object_id != prepared.base.intervention.subject_id
            or subject_placement.support_object_id != checked_evidence.support_object_id
            or subject_placement.placement_sha256 != checked_evidence.placement_sha256
        ):
            raise ValueError("proxy placement does not bind prepared evidence")
        return _project_proxy(
            prepared,
            workspace,
            patch_index=collision_delegation.patch_index,
        )


def build_target_proposal_problem(
    scene: Scene,
    intervention: InterventionSpec,
    workspace: EndpointWorkspace,
    *,
    case_id: str,
) -> SemanticProblemV2_3:
    """Build reachability's fixed-margin target-only semantic problem."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        prepared = _prepare_base_proxy(
            scene,
            intervention,
            case_id=case_id,
            bbox_visibility=False,
        )
        problem, _ = _project_base_payload(prepared, workspace)
        return problem


def _project_proxy_problem(
    prepared: _PreparedProxy,
    workspace: EndpointWorkspace,
) -> SemanticProblemV2_3:
    """Project a prepared request without constructing a wire artifact."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        problem, _ = _project_base_payload(prepared, workspace)
        return problem


def _runtime_collision_delegation_candidates(
    prepared: _PreparedProxy,
) -> frozenset[str]:
    if type(prepared) is not _PreparedProxy:
        raise TypeError("prepared native proxy context must be exact")
    subject = prepared.scene.object_by_id(prepared.intervention.subject_id)
    support_id = subject.support_object_id
    result = set()
    for native_id, obstacle in prepared.collision_proxies:
        if native_id in {subject.object_id, support_id}:
            continue
        if _conservative_source_overlap(subject.obb, obstacle):
            result.add(native_id)
    return frozenset(result)


def _conservative_source_overlap(subject: OBB, obstacle: OBB) -> bool:
    subject_lower_z, subject_upper_z = _z_interval(subject)
    obstacle_lower_z, obstacle_upper_z = _z_interval(obstacle)
    if obstacle_upper_z <= subject_lower_z or subject_upper_z <= obstacle_lower_z:
        return False
    combined_radius = _l1_horizontal_radius(subject) + _l1_horizontal_radius(obstacle)
    return (
        abs(
            Fraction.from_float(subject.center.x)
            - Fraction.from_float(obstacle.center.x)
        )
        < combined_radius
        and abs(
            Fraction.from_float(subject.center.y)
            - Fraction.from_float(obstacle.center.y)
        )
        < combined_radius
    )


def _without_delegated_collision_bodies(
    problem: SemanticProblemV2_3,
    delegated_native_ids: tuple[str, ...],
) -> SemanticProblemV2_3:
    if not delegated_native_ids:
        return problem
    delegated_body_ids = {_body_id(item) for item in delegated_native_ids}
    payload = problem.model_dump(mode="python", warnings="error")
    bodies = payload["scene"]["collision_bodies"]["values"]
    delegated_geometry_ids = {
        geometry_id
        for body in bodies
        if body["body_id"] in delegated_body_ids
        for geometry_id in body["geometry_instance_ids"]
    }
    if {body["body_id"] for body in bodies} & delegated_body_ids != (
        delegated_body_ids
    ):
        raise ValueError("delegated collision body is absent from the CPU proxy")
    payload["scene"]["collision_bodies"]["values"] = tuple(
        body for body in bodies if body["body_id"] not in delegated_body_ids
    )
    payload["scene"]["geometry_instances"]["values"] = tuple(
        geometry
        for geometry in payload["scene"]["geometry_instances"]["values"]
        if geometry["geometry_id"] not in delegated_geometry_ids
    )
    collision = payload["constraints"]["collision_constraints"][0]
    collision["obstacle_body_ids"] = tuple(
        body_id
        for body_id in collision["obstacle_body_ids"]
        if body_id not in delegated_body_ids
    )
    return SemanticProblemV2_3.model_validate(payload, strict=True)


def _prepare_base_proxy(
    scene: Scene,
    intervention: InterventionSpec,
    *,
    case_id: str,
    bbox_visibility: bool = False,
) -> _PreparedProxy:
    checked_scene = _strict_legacy(scene, Scene, "scene")
    checked_intervention = _strict_legacy(
        intervention,
        InterventionSpec,
        "intervention",
    )
    if not isinstance(case_id, str):
        raise TypeError("case_id must be an exact str")
    if type(bbox_visibility) is not bool:
        raise TypeError("bbox_visibility must be an exact bool")

    subject = checked_scene.object_by_id(checked_intervention.subject_id)
    if not subject.movable:
        raise ValueError("native subject must be movable")
    checked_scene.object_by_id(checked_intervention.reference_id)
    checked_scene.camera_by_id(checked_intervention.camera_id)
    if checked_intervention.camera_id not in subject.views:
        raise ValueError("subject requires an exact baseline view")
    reference = checked_scene.object_by_id(checked_intervention.reference_id)
    if checked_intervention.camera_id not in reference.views:
        raise ValueError("reference requires an exact baseline view")

    proxy_scene = checked_scene.model_copy(
        update={
            "objects": tuple(
                item.model_copy(
                    update={
                        "movable": item.object_id == subject.object_id,
                        "position": item.obb.center,
                        "rotation": item.obb.rotation,
                    }
                )
                for item in checked_scene.objects
            )
        }
    )
    proxy_scene = Scene.model_validate(
        proxy_scene.model_dump(mode="python"),
        strict=True,
    )
    legacy_sha = _legacy_sha256(checked_scene)
    intervention_sha = _legacy_sha256(checked_intervention)
    transient = _ProxyChallenge(
        case_id=case_id,
        direction=(
            f"{checked_intervention.relation_before.value}_to_"
            f"{checked_intervention.relation_after.value}"
        ),
        archetype="native-scene-proxy",
        expected_outcome="SAT",
        scene=proxy_scene,
        intervention=checked_intervention,
        scene_source_sha256=legacy_sha,
        intervention_source_sha256=intervention_sha,
        expectation_source_sha256=(
            _PROXY_POLICY_SHA256 if bbox_visibility else _TARGET_PROPOSAL_POLICY_SHA256
        ),
    )
    base_problem = (
        _convert_bbox_proxy_scene(transient)
        if bbox_visibility
        else _convert_target_proxy_scene(transient)
    )
    collision_proxy_by_native_id, minimum_margin_ids = _fixed_margin_collision_proxies(
        checked_scene,
        subject_object_id=subject.object_id,
        support_object_id=subject.support_object_id,
    )
    return _PreparedProxy(
        scene=checked_scene,
        intervention=checked_intervention,
        case_id=case_id,
        base_problem=base_problem,
        legacy_scene_sha256=legacy_sha,
        intervention_sha256=intervention_sha,
        collision_proxies=tuple(sorted(collision_proxy_by_native_id.items())),
        minimum_margin_native_object_ids=frozenset(minimum_margin_ids),
    )


def _project_base_payload(
    prepared: _PreparedProxy,
    endpoint_workspace: EndpointWorkspace,
) -> tuple[SemanticProblemV2_3, dict[str, object]]:
    """Return a version-neutral base projection without public artifacts."""

    if type(prepared) is not _PreparedProxy:
        raise TypeError("prepared native proxy context must be exact")
    checked_workspace = _strict_v2(
        endpoint_workspace,
        EndpointWorkspace,
        "endpoint_workspace",
    )
    checked_scene = prepared.scene
    checked_intervention = prepared.intervention
    subject = checked_scene.object_by_id(checked_intervention.subject_id)
    reference = checked_scene.object_by_id(checked_intervention.reference_id)
    _require_workspace_subset(prepared.base_problem, checked_workspace)
    collision_proxy_by_native_id = dict(prepared.collision_proxies)
    included, excluded = _partition_fixed_obstacles(
        checked_scene,
        subject.object_id,
        subject.obb,
        checked_workspace,
        collision_proxy_by_native_id,
    )
    if subject.support_object_id is not None:
        included.add(subject.support_object_id)
        excluded.discard(subject.support_object_id)

    problem = _project_problem(
        prepared.base_problem,
        checked_scene,
        checked_workspace,
        included,
        collision_proxy_by_native_id,
    )
    fixed_ids = {item.object_id for item in checked_scene.objects} - {subject.object_id}
    if included | excluded != fixed_ids or included & excluded:
        raise RuntimeError("native obstacle partition is not closed")
    overlay_ids = tuple(
        sorted(
            (
                {
                    item.source_object_id
                    for item in checked_scene.collision_obstacles
                    if item.source_object_id != subject.support_object_id
                }
            )
            & included
        )
    )
    return problem, {
        "case_id": prepared.case_id,
        "native_scene_id": checked_scene.scene_id,
        "native_camera_id": checked_intervention.camera_id,
        "subject_native_object_id": subject.object_id,
        "reference_native_object_id": reference.object_id,
        "legacy_scene_sha256": prepared.legacy_scene_sha256,
        "intervention_sha256": prepared.intervention_sha256,
        "semantic_problem_sha256": problem.semantic_problem_sha256,
        "endpoint_workspace": checked_workspace,
        "included_collision_native_object_ids": tuple(sorted(included)),
        "excluded_collision_native_object_ids": tuple(sorted(excluded)),
        "clearance_overlay_native_object_ids": overlay_ids,
        "minimum_margin_native_object_ids": tuple(
            sorted(prepared.minimum_margin_native_object_ids & included)
        ),
    }


def _project_direct_support_patch(
    problem: SemanticProblemV2_3,
    patch: ReceptacleSurfacePatch,
    *,
    subject_object_id: str,
) -> SemanticProblemV2_3:
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("patch projection problem must be exact")
    checked_patch = _strict_v2(
        patch,
        ReceptacleSurfacePatch,
        "selected_patch",
    )
    payload = problem.model_dump(mode="python")
    surfaces = payload["scene"]["support_surfaces"]["values"]
    constraints = payload["constraints"]["support_constraints"]
    matching_constraints = tuple(
        item for item in constraints if item["supported_object_id"] == subject_object_id
    )
    if len(matching_constraints) != 1:
        raise ValueError("patch-bound proxy requires one direct support constraint")
    surface_id = matching_constraints[0]["surface_id"]
    matching_surfaces = tuple(
        item for item in surfaces if item["surface_id"] == surface_id
    )
    if len(matching_surfaces) != 1:
        raise ValueError("patch-bound proxy requires one direct support surface")
    surface = matching_surfaces[0]
    world_x, world_y, world_yaw = _support_surface_world_pose_payload(
        payload,
        surface,
    )
    vertices = surface["region_uv"]["components"][0]["exterior"]["vertices"]
    world_coordinates = (
        (checked_patch.x_min, checked_patch.z_min),
        (checked_patch.x_max, checked_patch.z_min),
        (checked_patch.x_max, checked_patch.z_max),
        (checked_patch.x_min, checked_patch.z_max),
    )
    cosine = math.cos(world_yaw)
    sine = math.sin(world_yaw)
    coordinates = tuple(
        (
            cosine * (x - world_x) + sine * (y - world_y),
            -sine * (x - world_x) + cosine * (y - world_y),
        )
        for x, y in world_coordinates
    )
    surface["region_uv"]["components"][0]["exterior"]["vertices"] = tuple(
        {**vertex, "x": coordinate[0], "y": coordinate[1]}
        for vertex, coordinate in zip(vertices, coordinates, strict=True)
    )
    return SemanticProblemV2_3.model_validate(payload, strict=True)


def _support_surface_world_pose_payload(
    problem_payload: dict,
    surface_payload: dict,
) -> tuple[float, float, float]:
    anchor = surface_payload["anchor_from_surface"]
    local = anchor["translation"]
    owner_id = surface_payload["owner_object_id"]
    if owner_id is None:
        return local["x"], local["y"], anchor["yaw_radians"]
    owners = tuple(
        item
        for item in problem_payload["scene"]["objects"]["values"]
        if item["object_id"] == owner_id
    )
    if len(owners) != 1:
        raise ValueError("direct support owner binding is not closed")
    owner = owners[0]["pose"]["world_from_object"]
    owner_translation = owner["translation"]
    owner_yaw = owner["yaw_radians"]
    cosine = math.cos(owner_yaw)
    sine = math.sin(owner_yaw)
    return (
        owner_translation["x"] + cosine * local["x"] - sine * local["y"],
        owner_translation["y"] + sine * local["x"] + cosine * local["y"],
        owner_yaw + anchor["yaw_radians"],
    )


def _strict_legacy(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _strict_v2(value, expected_type, label: str):
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    return expected_type.model_validate(value.model_dump(mode="python"), strict=True)


def _fixed_margin_collision_proxies(
    scene: Scene,
    *,
    subject_object_id: str,
    support_object_id: str | None,
) -> tuple[dict[str, OBB], set[str]]:
    """Return max(existing, 1 cm) proxies without inflating direct support."""

    objects = {item.object_id: item for item in scene.objects}
    overlays = {}
    for obstacle in scene.collision_obstacles:
        if obstacle.source_object_id not in objects:
            raise ValueError("collision overlay references an unknown native object")
        if obstacle.source_object_id in overlays:
            raise ValueError("collision overlays must be unique by native object")
        overlays[obstacle.source_object_id] = obstacle

    result: dict[str, OBB] = {}
    minimum_margin_ids: set[str] = set()
    for object_id, item in objects.items():
        if object_id == subject_object_id:
            continue
        overlay = overlays.get(object_id)
        if object_id == support_object_id:
            continue
        base = item.obb if overlay is None else overlay.obb
        existing = 0.0 if overlay is None else overlay.clearance_m
        proxy = _inflate_obb(base, max(existing, _NON_SUPPORT_COLLISION_MARGIN_M))
        _require_valid_proxy_obb(proxy)
        result[object_id] = proxy
        minimum_margin_ids.add(object_id)
    return result, minimum_margin_ids


def _inflate_obb(obb: OBB, margin_m: float) -> OBB:
    diameter = 2.0 * margin_m
    return obb.model_copy(
        update={
            "extent": obb.extent.model_copy(
                update={
                    "x": obb.extent.x + diameter,
                    "y": obb.extent.y + diameter,
                    "z": obb.extent.z + diameter,
                }
            )
        }
    )


def _partition_fixed_obstacles(
    scene: Scene,
    subject_id: str,
    subject_obb: OBB,
    workspace: EndpointWorkspace,
    collision_proxies: dict[str, OBB],
) -> tuple[set[str], set[str]]:
    included: set[str] = set()
    excluded: set[str] = set()
    for item in scene.objects:
        if item.object_id == subject_id:
            continue
        proxy = collision_proxies.get(item.object_id, item.obb)
        target = (
            excluded if _proven_disjoint(subject_obb, proxy, workspace) else included
        )
        target.add(item.object_id)
    return included, excluded


def _proven_disjoint(
    subject: OBB,
    obstacle: OBB,
    workspace: EndpointWorkspace,
) -> bool:
    subject_lower_z, subject_upper_z = _z_interval(subject)
    obstacle_lower_z, obstacle_upper_z = _z_interval(obstacle)
    if obstacle_upper_z <= subject_lower_z or subject_upper_z <= obstacle_lower_z:
        return True

    # hx + hy encloses the projection of any upright yawed rectangle on both
    # world axes.  It is deliberately looser than sqrt(hx^2 + hy^2), but its
    # exact rational proof needs no trigonometric or binary64 tolerance.
    subject_radius = _l1_horizontal_radius(subject)
    obstacle_radius = _l1_horizontal_radius(obstacle)
    obstacle_x = Fraction.from_float(obstacle.center.x)
    obstacle_y = Fraction.from_float(obstacle.center.y)
    min_x = Fraction.from_float(workspace.min_x_m) - subject_radius
    max_x = Fraction.from_float(workspace.max_x_m) + subject_radius
    min_y = Fraction.from_float(workspace.min_y_m) - subject_radius
    max_y = Fraction.from_float(workspace.max_y_m) + subject_radius
    return (
        obstacle_x + obstacle_radius <= min_x
        or max_x <= obstacle_x - obstacle_radius
        or obstacle_y + obstacle_radius <= min_y
        or max_y <= obstacle_y - obstacle_radius
    )


def _z_interval(obb: OBB) -> tuple[Fraction, Fraction]:
    center = Fraction.from_float(obb.center.z)
    half = Fraction.from_float(obb.extent.z) / 2
    return center - half, center + half


def _l1_horizontal_radius(obb: OBB) -> Fraction:
    return (Fraction.from_float(obb.extent.x) + Fraction.from_float(obb.extent.y)) / 2


def _require_workspace_subset(
    problem: SemanticProblemV2_3,
    workspace: EndpointWorkspace,
) -> None:
    component = problem.scene.workspace_boundaries.values[0].region_world_xy.components[
        0
    ]
    vertices = component.exterior.vertices
    allowed = (
        min(item.x for item in vertices),
        min(item.y for item in vertices),
        max(item.x for item in vertices),
        max(item.y for item in vertices),
    )
    requested = (
        workspace.min_x_m,
        workspace.min_y_m,
        workspace.max_x_m,
        workspace.max_y_m,
    )
    if any(
        Fraction.from_float(actual) < Fraction.from_float(limit)
        for actual, limit in zip(requested[:2], allowed[:2], strict=True)
    ) or any(
        Fraction.from_float(actual) > Fraction.from_float(limit)
        for actual, limit in zip(requested[2:], allowed[2:], strict=True)
    ):
        raise ValueError("endpoint workspace exceeds the subject anchor locus")


def _project_problem(
    base_problem: SemanticProblemV2_3,
    scene: Scene,
    workspace: EndpointWorkspace,
    included: set[str],
    collision_proxies: dict[str, OBB],
) -> SemanticProblemV2_3:
    payload = base_problem.model_dump(mode="python")
    spec = base_problem.constraints.target_relation
    subject_native_id = spec.subject_id.removeprefix("object:")
    kept_object_ids = {spec.subject_id, spec.reference_id}
    all_objects = {
        item["object_id"]: item for item in payload["scene"]["objects"]["values"]
    }
    all_bodies = {
        item["body_id"]: item for item in payload["scene"]["collision_bodies"]["values"]
    }
    support_body_ids = {
        item["supporting_body_id"]
        for item in payload["scene"]["support_surfaces"]["values"]
    }
    support_geometry_ids = {
        geometry_id
        for body_id in support_body_ids
        for geometry_id in all_bodies[body_id]["geometry_instance_ids"]
    }
    payload["scene"]["objects"]["values"] = tuple(
        all_objects[item] for item in sorted(kept_object_ids)
    )

    geometries = []
    for geometry in payload["scene"]["geometry_instances"]["values"]:
        owner = geometry["owner_object_id"]
        if owner is None and geometry["geometry_id"] in support_geometry_ids:
            geometries.append(geometry)
            continue
        if owner == spec.subject_id:
            geometries.append(geometry)
            continue
        if owner == spec.reference_id and geometry["role"].value in {
            "RELATION",
            "VISUAL",
        }:
            geometries.append(geometry)
            continue
        native_id = owner.removeprefix("object:") if owner is not None else None
        if native_id not in included or geometry["role"].value != "COLLISION":
            continue
        proxy = collision_proxies.get(native_id, scene.object_by_id(native_id).obb)
        fixed = dict(geometry)
        fixed["owner_object_id"] = None
        fixed["anchor_from_geometry"] = _obb_transform(proxy)
        fixed["shape"] = {
            **fixed["shape"],
            "size_m": proxy.extent.model_dump(mode="python"),
        }
        geometries.append(fixed)
    payload["scene"]["geometry_instances"]["values"] = tuple(geometries)

    kept_body_ids = (
        {_body_id(subject_native_id)}
        | {_body_id(native_id) for native_id in included}
        | support_body_ids
    )
    bodies = []
    for body in payload["scene"]["collision_bodies"]["values"]:
        if body["body_id"] not in kept_body_ids:
            continue
        if body["owner_object_id"] == spec.subject_id:
            bodies.append(body)
        else:
            bodies.append({**body, "owner_object_id": None})
    payload["scene"]["collision_bodies"]["values"] = tuple(bodies)

    _set_workspace(payload, workspace)
    _set_support_proxy(
        payload,
        scene.object_by_id(subject_native_id),
        all_objects,
        kept_object_ids,
    )
    payload["scene"]["baseline_observations"]["values"] = tuple(
        item
        for item in payload["scene"]["baseline_observations"]["values"]
        if item["object_id"] in kept_object_ids
    )
    payload["scene"]["baseline_observations"]["completeness"] = FactCompletenessV2.EXACT
    payload["constraints"]["collision_constraints"][0]["obstacle_body_ids"] = tuple(
        sorted(kept_body_ids - {_body_id(subject_native_id)})
    )
    payload["constraints"]["visibility_constraints"][0]["query_object_ids"] = tuple(
        sorted(kept_object_ids)
    )
    payload["objective"]["relation_damage"]["pair_axis_weights"] = tuple(
        item
        for item in payload["objective"]["relation_damage"]["pair_axis_weights"]
        if {
            item["key"]["first_object_id"],
            item["key"]["second_object_id"],
        }
        == kept_object_ids
    )
    return SemanticProblemV2_3.model_validate(payload, strict=True)


def _set_workspace(payload: dict, workspace: EndpointWorkspace) -> None:
    vertices = payload["scene"]["workspace_boundaries"]["values"][0]["region_world_xy"][
        "components"
    ][0]["exterior"]["vertices"]
    coordinates = (
        (workspace.min_x_m, workspace.min_y_m),
        (workspace.max_x_m, workspace.min_y_m),
        (workspace.max_x_m, workspace.max_y_m),
        (workspace.min_x_m, workspace.max_y_m),
    )
    payload["scene"]["workspace_boundaries"]["values"][0]["region_world_xy"][
        "components"
    ][0]["exterior"]["vertices"] = tuple(
        {**vertex, "x": coordinate[0], "y": coordinate[1]}
        for vertex, coordinate in zip(vertices, coordinates, strict=True)
    )


def _set_support_proxy(
    payload: dict,
    subject,
    all_objects: dict[str, dict],
    kept_object_ids: set[str],
) -> None:
    surface = payload["scene"]["support_surfaces"]["values"][0]
    exact_contact_z = Fraction.from_float(subject.obb.center.z) - (
        Fraction.from_float(subject.obb.extent.z) / 2
    )
    _clamp_support_collision_top(
        payload,
        surface["supporting_body_id"],
        exact_contact_z,
    )
    desired_z = float(exact_contact_z)
    exact_gap = exact_contact_z - Fraction.from_float(desired_z)
    gap_lower, gap_upper = _directed_fraction_bounds(exact_gap)
    support_constraint = payload["constraints"]["support_constraints"][0]
    support_constraint["contact_gap_min_m"] = gap_lower
    support_constraint["contact_gap_max_m"] = gap_upper
    owner_id = surface["owner_object_id"]
    anchor = surface["anchor_from_surface"]
    if owner_id is None:
        surface["anchor_from_surface"] = {
            **anchor,
            "translation": {**anchor["translation"], "z": desired_z},
        }
        return
    owner_pose = all_objects[owner_id]["pose"]["world_from_object"]
    if owner_id in kept_object_ids:
        surface["anchor_from_surface"] = {
            **anchor,
            "translation": {
                **anchor["translation"],
                "z": desired_z - owner_pose["translation"]["z"],
            },
        }
        return
    angle = owner_pose["yaw_radians"]
    local = anchor["translation"]
    parent = owner_pose["translation"]
    surface["owner_object_id"] = None
    surface["anchor_from_surface"] = {
        "kind": "DIRECTED_YAW_INTERVAL",
        "translation": {
            "x": parent["x"]
            + math.cos(angle) * local["x"]
            - math.sin(angle) * local["y"],
            "y": parent["y"]
            + math.sin(angle) * local["x"]
            + math.cos(angle) * local["y"],
            "z": desired_z,
        },
        "yaw_radians": angle + anchor["yaw_radians"],
    }


def _clamp_support_collision_top(
    payload: dict,
    supporting_body_id: str,
    exact_contact_z: Fraction,
) -> None:
    """Keep the proxy support solid at or below its exact contact plane.

    AI2-THOR receptacle OBBs are visual/collision envelopes rather than exact
    contact surfaces.  Their top can sit slightly above an object's observed
    resting bottom, making one source snapshot claim both support and solid
    penetration.  The native proxy already replaces the receptacle surface by
    the subject-bottom plane; its collision solid must use that same boundary.
    The center is rounded downward so the closed solid can touch but never
    cross the exact rational plane.
    """

    bodies = {
        item["body_id"]: item for item in payload["scene"]["collision_bodies"]["values"]
    }
    try:
        geometry_ids = set(bodies[supporting_body_id]["geometry_instance_ids"])
    except KeyError as error:
        raise RuntimeError(
            "support proxy lost its supporting collision body"
        ) from error
    if not geometry_ids:
        raise RuntimeError("support proxy collision body must not be empty")

    matched: set[str] = set()
    for geometry in payload["scene"]["geometry_instances"]["values"]:
        geometry_id = geometry["geometry_id"]
        if geometry_id not in geometry_ids:
            continue
        matched.add(geometry_id)
        try:
            size_z = geometry["shape"]["size_m"]["z"]
            anchor = geometry["anchor_from_geometry"]
            center_z = anchor["translation"]["z"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "support collision proxy must be one anchored upright box"
            ) from error
        half_extent_z = Fraction.from_float(size_z) / 2
        current_top = Fraction.from_float(center_z) + half_extent_z
        if current_top <= exact_contact_z:
            continue
        clamped_center = _fraction_floor_binary64(exact_contact_z - half_extent_z)
        geometry["anchor_from_geometry"] = {
            **anchor,
            "translation": {
                **anchor["translation"],
                "z": clamped_center,
            },
        }
        if Fraction.from_float(clamped_center) + half_extent_z > exact_contact_z:
            raise RuntimeError("support collision top clamp rounded inward")
    if matched != geometry_ids:
        raise RuntimeError("support collision body geometry closure is incomplete")


def _fraction_floor_binary64(value: Fraction) -> float:
    try:
        published = float(value)
    except OverflowError as error:
        raise ArithmeticError("support collision center is not finite") from error
    if not math.isfinite(published):
        raise ArithmeticError("support collision center is not finite")
    if Fraction.from_float(published) > value:
        published = math.nextafter(published, -math.inf)
    if not math.isfinite(published) or Fraction.from_float(published) > value:
        raise ArithmeticError("support collision center cannot be rounded downward")
    return published


def _directed_fraction_bounds(value: Fraction) -> tuple[float, float]:
    published = float(value)
    if not math.isfinite(published):
        raise ArithmeticError("native support residual is not finite")
    exact = Fraction.from_float(published)
    lower = math.nextafter(published, -math.inf) if exact > value else published
    upper = math.nextafter(published, math.inf) if exact < value else published
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ArithmeticError("native support residual cannot be enclosed")
    return lower, upper


def _obb_transform(obb: OBB) -> dict[str, object]:
    _require_valid_proxy_obb(obb)
    yaw = _obb_yaw(obb)
    return {
        "kind": "DIRECTED_YAW_INTERVAL",
        "translation": obb.center.model_dump(mode="python"),
        "yaw_radians": yaw,
    }


def _obb_yaw(obb: OBB) -> float:
    quaternion = obb.rotation
    if quaternion.x != 0.0 or quaternion.y != 0.0:
        raise ValueError("native collision proxy must be exact upright yaw")
    norm = math.hypot(quaternion.z, quaternion.w)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("native collision proxy quaternion must be unit length")
    return 0.0 if quaternion.z == 0.0 else 2.0 * math.atan2(quaternion.z, quaternion.w)


def _require_valid_proxy_obb(obb: OBB) -> None:
    values = (
        obb.center.x,
        obb.center.y,
        obb.center.z,
        obb.extent.x,
        obb.extent.y,
        obb.extent.z,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("native collision proxy must be finite")
    if min(obb.extent.x, obb.extent.y, obb.extent.z) <= 0.0:
        raise ValueError("native collision proxy extents must be positive")
    _obb_yaw(obb)


def _legacy_sha256(value: BaseModel) -> str:
    """Hash one legacy model after applying the v2 finite-number convention."""

    if not isinstance(value, BaseModel):
        raise TypeError("competition legacy digest requires a Pydantic model")
    payload = json.dumps(
        _stable_value(value.model_dump(mode="json")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_value(value):
    if isinstance(value, Enum):
        return _stable_value(value.value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("competition legacy digest requires finite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("competition legacy digest requires string mapping keys")
        return {key: _stable_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item) for item in value]
    if isinstance(value, AbstractSet):
        normalized = [_stable_value(item) for item in value]
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


__all__ = (
    "build_proxy_bundle",
    "build_target_proposal_problem",
    "default_planning_workspace",
    "default_solver_config",
)
