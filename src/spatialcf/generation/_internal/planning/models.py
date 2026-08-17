"""Flat models for the current source-only endpoint planner."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import model_validator

from spatialcf.domain.v2.base import CanonicalId, FiniteFloat, Sha256Digest, V2Model
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.serialization import canonical_sha256_v2
from spatialcf.generation._internal.evidence.surface import ReceptacleSurfacePatch
from spatialcf.generation.capture.models import (
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportKindV2_9,
)

_BINDING_HASH_DOMAIN = "spatialcf.competition-native-proxy-binding.v2.9.4"
_BUNDLE_HASH_DOMAIN = "spatialcf.competition-native-proxy-bundle.v2.9.4"
_ENDPOINT_PLAN_HASH_DOMAIN = "spatialcf.competition-native-endpoint-plan.v2.9.4"
_NON_SUPPORT_COLLISION_MARGIN_M = 0.01
_PROXY_POLICY_SHA256 = hashlib.sha256(
    b"spatialcf.competition-native-scene-proxy.v2.9.4\0"
    b"proxy-v2.9.3;visibility-semantics:analytic-bbox-v1;"
    b"visibility:visible-clipped-projected-bounding-box-area-fraction;"
    b"definition:2;"
    b"VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"
).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


class EndpointWorkspace(V2Model):
    """Closed absolute world-XY endpoint policy for one native request."""

    min_x_m: FiniteFloat
    min_y_m: FiniteFloat
    max_x_m: FiniteFloat
    max_y_m: FiniteFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_x_m >= self.max_x_m or self.min_y_m >= self.max_y_m:
            raise ValueError("endpoint workspace must have positive area")
        return self


class SubjectPlacementFact(V2Model):
    """One exact captured receptacle placement supplied without an adapter."""

    placement_version: Literal["competition-native-endpoint-placement:2.9.2"] = (
        "competition-native-endpoint-placement:2.9.2"
    )
    source_capture: CompetitionNativeSourceCaptureV2_9
    subject_placement: CompetitionNativeSubjectPlacementFactV2_9

    @model_validator(mode="after")
    def validate_placement(self) -> Self:
        if type(self.source_capture) is not CompetitionNativeSourceCaptureV2_9:
            raise TypeError("endpoint placement source_capture must be exact")
        if (
            type(self.subject_placement)
            is not CompetitionNativeSubjectPlacementFactV2_9
        ):
            raise TypeError("endpoint placement subject_placement must be exact")
        capture = self.source_capture
        placement = self.subject_placement
        matching_placements = tuple(
            item
            for item in capture.placement_facts
            if item.object_id == placement.object_id
        )
        if len(matching_placements) != 1 or matching_placements[0] != placement:
            raise ValueError(
                "endpoint subject placement does not bind the source capture"
            )
        if (
            placement.availability
            is not CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
            or placement.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or placement.support_object_id is None
            or placement.floor_object_id is not None
            or not placement.native_positions
        ):
            raise ValueError("endpoint subject placement is not executable")
        matching_subjects = tuple(
            item
            for item in capture.scene.objects
            if item.object_id == placement.object_id
        )
        matching_supports = tuple(
            item
            for item in capture.scene.objects
            if item.object_id == placement.support_object_id
        )
        matching_support_facts = tuple(
            item
            for item in capture.support_facts
            if item.object_id == placement.object_id
        )
        if (
            len(matching_subjects) != 1
            or len(matching_supports) != 1
            or len(matching_support_facts) != 1
            or matching_subjects[0].support_object_id != placement.support_object_id
            or matching_support_facts[0].support_kind
            is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or matching_support_facts[0].support_object_id
            != placement.support_object_id
        ):
            raise ValueError("endpoint subject/support binding is not closed")
        return self


@dataclass(frozen=True, slots=True)
class CollisionDelegation:
    """Request-local inputs that select one patch collision-authority partition."""

    case_id: str
    patch_index: int

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise TypeError(
                "collision delegation case_id must be a non-empty exact str"
            )
        if type(self.patch_index) is not int or self.patch_index < 0:
            raise TypeError("collision delegation patch_index must be non-negative")


class ProxyBinding(V2Model):
    """Current bbox proxy lineage with an explicit native collision authority."""

    binding_version: Literal["competition-native-proxy-binding:2.9.4"] = (
        "competition-native-proxy-binding:2.9.4"
    )
    proxy_scope: Literal[
        "CAPTURE_PATCH_CPU_NONDELEGATED_NATIVE_AUDIT_AND_BBOX_VISIBILITY"
    ] = "CAPTURE_PATCH_CPU_NONDELEGATED_NATIVE_AUDIT_AND_BBOX_VISIBILITY"
    obstacle_filter: Literal["EXACT_Z_AND_L1_XY_OUTER_DISJOINTNESS"] = (
        "EXACT_Z_AND_L1_XY_OUTER_DISJOINTNESS"
    )
    support_plane_policy: Literal[
        "SUBJECT_BOTTOM_PLANE_SINGLE_NATIVE_PATCH_AND_SUPPORT_SOLID_TOP_CLAMP"
    ] = "SUBJECT_BOTTOM_PLANE_SINGLE_NATIVE_PATCH_AND_SUPPORT_SOLID_TOP_CLAMP"
    native_audit_status: Literal["NOT_PERFORMED"] = "NOT_PERFORMED"
    evidence_eligible: Literal[False] = False
    case_id: CanonicalId
    native_scene_id: CanonicalId
    native_camera_id: CanonicalId
    subject_native_object_id: CanonicalId
    reference_native_object_id: CanonicalId
    legacy_scene_sha256: Sha256Digest
    intervention_sha256: Sha256Digest
    proxy_policy_sha256: Sha256Digest = _PROXY_POLICY_SHA256
    semantic_problem_sha256: Sha256Digest
    endpoint_workspace: EndpointWorkspace
    included_collision_native_object_ids: tuple[CanonicalId, ...]
    excluded_collision_native_object_ids: tuple[CanonicalId, ...]
    clearance_overlay_native_object_ids: tuple[CanonicalId, ...]
    non_support_collision_margin_m: Literal[0.01] = _NON_SUPPORT_COLLISION_MARGIN_M
    minimum_margin_native_object_ids: tuple[CanonicalId, ...]
    source_capture_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    scene_sha256: Sha256Digest
    positions_sha256: Sha256Digest
    spawn_map_source_sha256: Sha256Digest
    placement_sha256: Sha256Digest
    surface_evidence_sha256: Sha256Digest
    subject_surface_evidence_sha256: Sha256Digest
    patch_index: int
    patch_sha256: Sha256Digest
    selected_patch: ReceptacleSurfacePatch
    runtime_collision_delegated_native_object_ids: tuple[CanonicalId, ...]
    visibility_semantics_id: Literal["visibility-semantics:analytic-bbox-v1"] = (
        "visibility-semantics:analytic-bbox-v1"
    )
    image_area_metric_definition_id: Literal[
        "visibility:visible-clipped-projected-bounding-box-area-fraction"
    ] = "visibility:visible-clipped-projected-bounding-box-area-fraction"
    image_area_metric_definition_version: Literal["definition:2"] = "definition:2"
    image_area_metric_formula: Literal[
        "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"
    ] = "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"

    @model_validator(mode="after")
    def validate_rosters(self) -> Self:
        if self.proxy_policy_sha256 != _PROXY_POLICY_SHA256:
            raise ValueError("native proxy policy digest mismatch")
        for field_name in (
            "included_collision_native_object_ids",
            "excluded_collision_native_object_ids",
            "clearance_overlay_native_object_ids",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and canonical")
        included = set(self.included_collision_native_object_ids)
        excluded = set(self.excluded_collision_native_object_ids)
        if included & excluded:
            raise ValueError("included and excluded collision rosters must be disjoint")
        if not set(self.clearance_overlay_native_object_ids) <= included:
            raise ValueError("clearance overlays must belong to included obstacles")
        if self.subject_native_object_id in included | excluded:
            raise ValueError("the subject is not a fixed collision obstacle")
        return self

    @model_validator(mode="after")
    def validate_minimum_margin_roster(self) -> Self:
        included = set(self.included_collision_native_object_ids)
        minimum_margin = self.minimum_margin_native_object_ids
        if minimum_margin != tuple(sorted(set(minimum_margin))):
            raise ValueError("minimum margin object IDs must be unique and canonical")
        if not set(minimum_margin) <= included:
            raise ValueError("minimum margin objects must be included obstacles")
        if self.subject_native_object_id in minimum_margin:
            raise ValueError("the subject cannot receive a fixed-obstacle margin")
        return self

    @model_validator(mode="after")
    def validate_patch_lineage(self) -> Self:
        if type(self.patch_index) is not int or self.patch_index < 0:
            raise ValueError("native proxy patch index must be non-negative and exact")
        if (
            self.selected_patch.patch_index != self.patch_index
            or self.selected_patch.patch_sha256 != self.patch_sha256
        ):
            raise ValueError("native proxy selected patch lineage mismatch")
        patch = self.selected_patch
        workspace = self.endpoint_workspace
        if (
            workspace.min_x_m < patch.x_min
            or workspace.max_x_m > patch.x_max
            or workspace.min_y_m < patch.z_min
            or workspace.max_y_m > patch.z_max
        ):
            raise ValueError("endpoint workspace exceeds the selected support patch")
        return self

    @model_validator(mode="after")
    def validate_runtime_collision_delegation(self) -> Self:
        included = set(self.included_collision_native_object_ids)
        excluded = set(self.excluded_collision_native_object_ids)
        delegated = self.runtime_collision_delegated_native_object_ids
        if delegated != tuple(sorted(set(delegated))):
            raise ValueError("runtime collision delegation must be canonical")
        delegated_set = set(delegated)
        if delegated_set & (included | excluded):
            raise ValueError("runtime collision delegation must be a disjoint roster")
        if self.subject_native_object_id in delegated_set:
            raise ValueError("the subject cannot be a delegated collision obstacle")
        return self

    @property
    def proxy_binding_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_BINDING_HASH_DOMAIN)


class ProxyBundle(V2Model):
    """One current Canonical problem plus its non-capability native lineage."""

    bundle_version: Literal["competition-native-proxy-bundle:2.9.4"] = (
        "competition-native-proxy-bundle:2.9.4"
    )
    semantic_problem: SemanticProblemV2_3
    binding: ProxyBinding

    @model_validator(mode="after")
    def validate_closure(self) -> Self:
        if self.binding.semantic_problem_sha256 != (
            self.semantic_problem.semantic_problem_sha256
        ):
            raise ValueError("native proxy binding does not close the problem hash")
        target = self.semantic_problem.constraints.target_relation
        if target.subject_id != _object_id(self.binding.subject_native_object_id):
            raise ValueError("native proxy subject binding mismatch")
        if target.reference_id != _object_id(self.binding.reference_native_object_id):
            raise ValueError("native proxy reference binding mismatch")
        if target.camera_id != _camera_id(self.binding.native_camera_id):
            raise ValueError("native proxy camera binding mismatch")
        return self

    @model_validator(mode="after")
    def validate_patch_surface(self) -> Self:
        surface = _direct_support_surface(
            self.semantic_problem,
            _object_id(self.binding.subject_native_object_id),
        )
        vertices = surface.region_uv.components[0].exterior.vertices
        if tuple((item.x, item.y) for item in vertices) != (
            _support_surface_local_patch_coordinates(
                self.semantic_problem,
                surface,
                self.binding.selected_patch,
            )
        ):
            raise ValueError("patch-bound proxy support rectangle mismatch")
        return self

    @model_validator(mode="after")
    def validate_collision_authority_partition(self) -> Self:
        obstacle_body_ids = set(
            self.semantic_problem.constraints.collision_constraints[0].obstacle_body_ids
        )
        included_body_ids = {
            _body_id(item) for item in self.binding.included_collision_native_object_ids
        }
        delegated_body_ids = {
            _body_id(item)
            for item in self.binding.runtime_collision_delegated_native_object_ids
        }
        if obstacle_body_ids != included_body_ids:
            raise ValueError("CPU collision bodies do not close the included roster")
        if obstacle_body_ids & delegated_body_ids:
            raise ValueError("delegated collision bodies escaped into the CPU roster")
        return self

    @model_validator(mode="after")
    def validate_bbox_visibility(self) -> Self:
        semantics = self.semantic_problem.visibility_semantics
        constraint = self.semantic_problem.constraints.visibility_constraints[0]
        if (
            semantics.semantics_id != self.binding.visibility_semantics_id
            or constraint.visibility_semantics_id
            != self.binding.visibility_semantics_id
            or constraint.image_area_metric_definition_id
            != self.binding.image_area_metric_definition_id
            or constraint.image_area_metric_definition_version
            != self.binding.image_area_metric_definition_version
        ):
            raise ValueError("bbox visibility metric lineage mismatch")
        definitions = tuple(
            definition
            for definition in semantics.definitions
            if definition.reference
            == (
                self.binding.image_area_metric_definition_id,
                self.binding.image_area_metric_definition_version,
            )
        )
        if (
            len(definitions) != 1
            or definitions[0].formula.value != self.binding.image_area_metric_formula
        ):
            raise ValueError("bbox visibility formula lineage mismatch")
        return self

    @property
    def proxy_bundle_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_BUNDLE_HASH_DOMAIN)


class EndpointPlan(V2Model):
    """One current solver-certified endpoint wholly owned by a source patch."""

    plan_version: Literal["competition-native-endpoint-plan:2.9.4"] = (
        "competition-native-endpoint-plan:2.9.4"
    )
    candidate_strategy: Literal[
        "stratified_directional_receptacle_patch_relation_ranked_"
        "runtime_collision_delegated_bbox_visibility"
    ] = (
        "stratified_directional_receptacle_patch_relation_ranked_"
        "runtime_collision_delegated_bbox_visibility"
    )
    planning_workspace: EndpointWorkspace
    endpoint_workspace: EndpointWorkspace
    candidate_index: int
    attempted_workspace_count: int
    candidate_point_count: int
    patch_index: int
    patch_sha256: str
    source_capture_sha256: str
    placement_sha256: str
    surface_evidence_sha256: str
    subject_surface_evidence_sha256: str
    proxy_bundle_sha256: str
    solve_result_sha256: str
    runtime_collision_delegated_native_object_ids: tuple[CanonicalId, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if type(self.planning_workspace) is not EndpointWorkspace:
            raise TypeError("planning_workspace must be exact")
        if type(self.endpoint_workspace) is not EndpointWorkspace:
            raise TypeError("endpoint_workspace must be exact")
        for label, value in (
            ("candidate_index", self.candidate_index),
            ("attempted_workspace_count", self.attempted_workspace_count),
            ("candidate_point_count", self.candidate_point_count),
            ("patch_index", self.patch_index),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(f"{label} must be a non-negative exact integer")
        if self.attempted_workspace_count < 1 or self.candidate_point_count < 1:
            raise ValueError("endpoint plan must record non-empty bounded work")
        if self.candidate_index >= self.candidate_point_count:
            raise ValueError("endpoint plan candidate index escaped its roster")
        for label, value in (
            ("patch", self.patch_sha256),
            ("source capture", self.source_capture_sha256),
            ("placement", self.placement_sha256),
            ("surface evidence", self.surface_evidence_sha256),
            ("subject surface evidence", self.subject_surface_evidence_sha256),
            ("proxy bundle", self.proxy_bundle_sha256),
            ("solve result", self.solve_result_sha256),
        ):
            _require_sha256(value, f"endpoint plan {label}")
        return self

    @model_validator(mode="after")
    def validate_runtime_collision_delegation(self) -> Self:
        delegated = self.runtime_collision_delegated_native_object_ids
        if delegated != tuple(sorted(set(delegated))):
            raise ValueError("endpoint runtime collision delegation must be canonical")
        return self

    @property
    def endpoint_plan_sha256(self) -> str:
        return canonical_sha256_v2(self, domain=_ENDPOINT_PLAN_HASH_DOMAIN)


def _direct_support_surface(problem: SemanticProblemV2_3, subject_object_id: str):
    constraints = tuple(
        item
        for item in problem.constraints.support_constraints
        if item.supported_object_id == subject_object_id
    )
    if len(constraints) != 1:
        raise ValueError("patch-bound proxy requires one direct support constraint")
    surfaces = tuple(
        item
        for item in problem.scene.support_surfaces.values
        if item.surface_id == constraints[0].surface_id
    )
    if len(surfaces) != 1:
        raise ValueError("patch-bound proxy requires one direct support surface")
    return surfaces[0]


def _support_surface_world_pose(
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


def _support_surface_local_patch_coordinates(
    problem: SemanticProblemV2_3,
    surface,
    patch: ReceptacleSurfacePatch,
) -> tuple[tuple[float, float], ...]:
    payload = problem.model_dump(mode="python")
    surface_payload = next(
        item
        for item in payload["scene"]["support_surfaces"]["values"]
        if item["surface_id"] == surface.surface_id
    )
    world_x, world_y, world_yaw = _support_surface_world_pose(
        payload,
        surface_payload,
    )
    cosine = math.cos(world_yaw)
    sine = math.sin(world_yaw)
    return tuple(
        (
            cosine * (x - world_x) + sine * (y - world_y),
            -sine * (x - world_x) + cosine * (y - world_y),
        )
        for x, y in (
            (patch.x_min, patch.z_min),
            (patch.x_max, patch.z_min),
            (patch.x_max, patch.z_max),
            (patch.x_min, patch.z_max),
        )
    )


def _object_id(native_id: str) -> str:
    return f"object:{native_id}"


def _body_id(native_id: str) -> str:
    return f"body:{native_id}"


def _camera_id(native_id: str) -> str:
    return f"camera:{native_id}"


__all__ = (
    "CollisionDelegation",
    "EndpointPlan",
    "EndpointWorkspace",
    "ProxyBinding",
    "ProxyBundle",
    "SubjectPlacementFact",
)
