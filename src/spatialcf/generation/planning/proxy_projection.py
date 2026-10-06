"""Prepared proxy projection, support patches and collision authority bindings."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

from spatialcf.domain.base import FactCompletenessV2
from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.scene import OBB, Scene
from spatialcf.generation.capture.models import (
    ReceptacleSurfacePatch,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
)
from spatialcf.generation.planning.models import (
    _PROXY_POLICY_SHA256,
    EndpointWorkspace,
    ProxyBinding,
    ProxyBundle,
)
from spatialcf.generation.planning.proxy_conversion import (
    _body_id as _body_id,
    _object_id as _object_id,
)
from spatialcf.generation.planning.proxy_geometry import (
    _clamp_support_collision_top,
    _conservative_source_overlap,
    _directed_fraction_bounds,
    _obb_transform,
    _partition_fixed_obstacles,
    _require_workspace_subset,
    _strict_v2,
)
from spatialcf.generation.planning.proxy_preparation import (
    _PreparedCurrentProxy,
    _PreparedProxy,
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
        if native_id in {
            subject.object_id,
            support_id,
            prepared.intervention.reference_id,
        }:
            continue
        if _conservative_source_overlap(subject.obb, obstacle):
            result.add(native_id)
    return frozenset(result)


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


_project_patch_payload.__module__ = "spatialcf.generation.planning.problem"
_delegate_runtime_collisions.__module__ = "spatialcf.generation.planning.problem"
_binding_roster.__module__ = "spatialcf.generation.planning.problem"
_project_proxy.__module__ = "spatialcf.generation.planning.problem"
_project_proxy_problem.__module__ = "spatialcf.generation.planning.problem"
_runtime_collision_delegation_candidates.__module__ = (
    "spatialcf.generation.planning.problem"
)
_without_delegated_collision_bodies.__module__ = "spatialcf.generation.planning.problem"
_project_base_payload.__module__ = "spatialcf.generation.planning.problem"
_project_direct_support_patch.__module__ = "spatialcf.generation.planning.problem"
_support_surface_world_pose_payload.__module__ = "spatialcf.generation.planning.problem"
_project_problem.__module__ = "spatialcf.generation.planning.problem"
_set_workspace.__module__ = "spatialcf.generation.planning.problem"
_set_support_proxy.__module__ = "spatialcf.generation.planning.problem"
