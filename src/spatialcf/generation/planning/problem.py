"""Current source-only proxy preparation and projection authority."""

from __future__ import annotations

import warnings

from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.request import InterventionSpec
from spatialcf.domain.scene import Scene
from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9
from spatialcf.generation.capture.models import (
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
)
from spatialcf.generation.planning.models import (
    CollisionDelegation,
    EndpointWorkspace,
    ProxyBundle,
    SubjectPlacementFact,
)
from spatialcf.generation.planning.proxy_conversion import (
    _BBOX_IMAGE_AREA_METRIC_ID as _BBOX_IMAGE_AREA_METRIC_ID,
    _BBOX_IMAGE_AREA_METRIC_VERSION as _BBOX_IMAGE_AREA_METRIC_VERSION,
    _BBOX_VISIBILITY_DEFINITIONS as _BBOX_VISIBILITY_DEFINITIONS,
    _BBOX_VISIBILITY_SEMANTICS_ID as _BBOX_VISIBILITY_SEMANTICS_ID,
    _VISIBILITY_DEFINITIONS as _VISIBILITY_DEFINITIONS,
    _ZERO_UNCERTAINTY as _ZERO_UNCERTAINTY,
    _body_id as _body_id,
    _camera_id as _camera_id,
    _category_id as _category_id,
    _convert_bbox_proxy_scene as _convert_bbox_proxy_scene,
    _convert_proxy_scene as _convert_proxy_scene,
    _convert_target_proxy_scene as _convert_target_proxy_scene,
    _default_solver_config_base as _default_solver_config_base,
    _floor_body as _floor_body,
    _geometry as _geometry,
    _geometry_id as _geometry_id,
    _known_facts as _known_facts,
    _legacy_camera as _legacy_camera,
    _legacy_yaw as _legacy_yaw,
    _object_id as _object_id,
    _objective as _objective,
    _ProxyChallenge as _ProxyChallenge,
    _ProxyConversionError as _ProxyConversionError,
    _rectangle_region as _rectangle_region,
    _relation_semantics as _relation_semantics,
    _subject_anchor_workspace as _subject_anchor_workspace,
    _support_surface as _support_surface,
    _surface_id as _surface_id,
    _yaw_transform as _yaw_transform,
)


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


__all__ = (
    "build_proxy_bundle",
    "build_target_proposal_problem",
    "default_planning_workspace",
    "default_solver_config",
)

# Supported historical imports resolve to the actual implementation objects.
from spatialcf.generation.planning.proxy_geometry import (
    _NON_SUPPORT_COLLISION_MARGIN_M as _NON_SUPPORT_COLLISION_MARGIN_M,
    _clamp_support_collision_top as _clamp_support_collision_top,
    _conservative_source_overlap as _conservative_source_overlap,
    _directed_fraction_bounds as _directed_fraction_bounds,
    _fixed_margin_collision_proxies as _fixed_margin_collision_proxies,
    _fraction_floor_binary64 as _fraction_floor_binary64,
    _inflate_obb as _inflate_obb,
    _l1_horizontal_radius as _l1_horizontal_radius,
    _obb_transform as _obb_transform,
    _obb_yaw as _obb_yaw,
    _partition_fixed_obstacles as _partition_fixed_obstacles,
    _proven_disjoint as _proven_disjoint,
    _require_valid_proxy_obb as _require_valid_proxy_obb,
    _require_workspace_subset as _require_workspace_subset,
    _strict_legacy as _strict_legacy,
    _strict_v2 as _strict_v2,
    _z_interval as _z_interval,
    default_planning_workspace as default_planning_workspace,
)
from spatialcf.generation.planning.proxy_preparation import (
    _TARGET_PROPOSAL_POLICY_SHA256 as _TARGET_PROPOSAL_POLICY_SHA256,
    _legacy_sha256 as _legacy_sha256,
    _prepare_base_proxy as _prepare_base_proxy,
    _prepare_proxy as _prepare_proxy,
    _PreparedCurrentProxy as _PreparedCurrentProxy,
    _PreparedProxy as _PreparedProxy,
    _verified_proxy_inputs as _verified_proxy_inputs,
)
from spatialcf.generation.planning.proxy_projection import (
    _binding_roster as _binding_roster,
    _delegate_runtime_collisions as _delegate_runtime_collisions,
    _project_base_payload as _project_base_payload,
    _project_direct_support_patch as _project_direct_support_patch,
    _project_patch_payload as _project_patch_payload,
    _project_problem as _project_problem,
    _project_proxy as _project_proxy,
    _project_proxy_problem as _project_proxy_problem,
    _runtime_collision_delegation_candidates as _runtime_collision_delegation_candidates,
    _set_support_proxy as _set_support_proxy,
    _set_workspace as _set_workspace,
    _support_surface_world_pose_payload as _support_surface_world_pose_payload,
    _without_delegated_collision_bodies as _without_delegated_collision_bodies,
)
