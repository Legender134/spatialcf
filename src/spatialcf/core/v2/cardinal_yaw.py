"""Exact Cardinal 2.1 capability checks and normalization to Canonical 2.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from spatialcf.core.v2._internal.resources.domain_operations import (
    DomainOperationBudgetV2,
)
from spatialcf.core.v2.candidate_domain import (
    CANDIDATE_DOMAIN_ALGORITHM_ID_V2,
    CANDIDATE_DOMAIN_ALGORITHM_VERSION_V2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M,
    RECTILINEAR_KERNEL_ID_V2,
    RECTILINEAR_KERNEL_VERSION_V2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactSetV2,
    NumericPolicyV2,
    RigidTransformV2,
    UncertaintyBudgetV2,
    Vec2V2,
    Vec3V2,
)
from spatialcf.domain.v2.cardinal import (
    CanonicalObjectV2_1,
    CanonicalSceneV2_1,
    GeometryInstanceV2_1,
    PinholeCameraV2_1,
    SemanticProblemV2_1,
    SupportSurfaceFactV2_1,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    DirectedOutwardGeometryKernelSpecV2,
)
from spatialcf.domain.v2.scene import (
    CameraDistortionModelV2,
    CanonicalObjectV2,
    CanonicalSceneV2,
    ObjectPoseV2,
    PinholeCameraV2,
    RegionBoundaryPolicyV2,
    SupportSurfaceFactV2,
)

CARDINAL_ALGORITHM_VERSION_V2_1 = "algorithm:2.1"
CARDINAL_KERNEL_VERSION_V2_1 = "kernel:2.1-cardinal-yaw"


class CardinalResourceLimitV2(RuntimeError):
    pass


class CardinalUnsupportedModelV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


@dataclass(slots=True)
class CardinalDomainBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = CardinalResourceLimitV2


@dataclass(frozen=True, slots=True)
class PreparedCardinalProblemV2_1:
    normalized_problem: SemanticProblemV2
    internal_config: CoreSolverConfigV2
    preprocessing_domain_operations: int


def reserve_cardinal_problem_structure_v2_1(
    problem: SemanticProblemV2_1,
    budget: CardinalDomainBudgetV2,
) -> None:
    """Reserve the frozen structural pass before deep strict reconstruction."""

    budget.consume()
    scene = problem.scene
    if type(scene) is not CanonicalSceneV2_1:
        raise TypeError("cardinal problem scene has the wrong exact type")
    for family_name in (
        "objects",
        "geometry_instances",
        "collision_bodies",
        "workspace_boundaries",
        "known_free_spaces",
        "support_surfaces",
        "cameras",
        "baseline_observations",
    ):
        budget.consume()
        facts = getattr(scene, family_name)
        if not isinstance(facts, FactSetV2):
            raise TypeError(f"{family_name} has the wrong fact-set type")
        branches = tuple(
            values
            for values in (facts.values, facts.inner_values, facts.outer_values)
            if values is not None
        )
        budget.consume(sum(len(values) for values in branches))
        if family_name != "support_surfaces":
            continue
        for values in branches:
            for surface in values:
                if type(surface) is not SupportSurfaceFactV2_1:
                    raise TypeError("support surface has the wrong exact type")
                components = surface.region_uv.components
                budget.consume(len(components))
                for component in components:
                    budget.consume(len(component.exterior.vertices))
                    for hole in component.holes:
                        budget.consume(1 + len(hole.vertices))


def registry_finding_v2_1(config: CoreSolverConfigV2) -> str | None:
    if (
        config.algorithm_id != CANDIDATE_DOMAIN_ALGORITHM_ID_V2
        or config.algorithm_version != CARDINAL_ALGORITHM_VERSION_V2_1
    ):
        return (
            f"UNREGISTERED_ALGORITHM:{config.algorithm_id}@{config.algorithm_version}"
        )
    kernel = config.geometry_kernel
    registered = (
        type(kernel) is DirectedOutwardGeometryKernelSpecV2
        and kernel.kernel_id == RECTILINEAR_KERNEL_ID_V2
        and kernel.kernel_version == CARDINAL_KERNEL_VERSION_V2_1
        and kernel.certified_outward_error_m
        == RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M
    )
    if registered:
        return None
    error = getattr(kernel, "certified_outward_error_m", "NONE")
    return (
        f"UNREGISTERED_GEOMETRY_KERNEL:{kernel.kernel_id}"
        f"@{kernel.kernel_version}:{kernel.soundness.value}:{error}"
    )


def prepare_cardinal_problem_v2_1(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    preprocessing_domain_operations: int,
) -> PreparedCardinalProblemV2_1:
    if config.max_branch_nodes != 0 or config.max_refinement_steps != 0:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:BRANCH_OR_REFINEMENT")
    if not _zero_numeric_policy(problem.numeric_policy):
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:NONZERO_UNCERTAINTY")
    normalized_scene = _normalize_scene(problem.scene)
    normalized_problem = SemanticProblemV2(
        scene=normalized_scene,
        constraints=problem.constraints,
        relation_semantics=problem.relation_semantics,
        visibility_semantics=problem.visibility_semantics,
        objective=problem.objective,
        numeric_policy=problem.numeric_policy,
    )
    remaining = config.max_domain_operations - preprocessing_domain_operations
    if remaining < 1:
        raise CardinalResourceLimitV2
    payload = config.model_dump(mode="python", warnings="error")
    payload["algorithm_version"] = CANDIDATE_DOMAIN_ALGORITHM_VERSION_V2
    payload["max_domain_operations"] = remaining
    kernel = dict(payload["geometry_kernel"])
    kernel["kernel_version"] = RECTILINEAR_KERNEL_VERSION_V2
    payload["geometry_kernel"] = kernel
    return PreparedCardinalProblemV2_1(
        normalized_problem=normalized_problem,
        internal_config=CoreSolverConfigV2.model_validate(payload, strict=True),
        preprocessing_domain_operations=preprocessing_domain_operations,
    )


def _normalize_scene(scene: CanonicalSceneV2_1) -> CanonicalSceneV2:
    for family_name in (
        "objects",
        "geometry_instances",
        "collision_bodies",
        "workspace_boundaries",
        "known_free_spaces",
        "support_surfaces",
        "cameras",
        "baseline_observations",
    ):
        _require_zero_fact_uncertainty(getattr(scene, family_name))

    objects = _map_fact_set(
        scene.objects,
        CanonicalObjectV2,
        _normalize_object,
    )
    object_turns = {
        item.object_id: item.pose.world_from_object.quarter_turns_ccw
        for item in scene.objects.values or ()
    }
    geometries = _map_fact_set(
        scene.geometry_instances,
        GeometryInstanceV2,
        lambda item: _normalize_geometry(item, object_turns),
    )
    surfaces = _map_fact_set(
        scene.support_surfaces,
        SupportSurfaceFactV2,
        lambda item: _normalize_support_surface(item, object_turns),
    )
    cameras = _map_fact_set(scene.cameras, PinholeCameraV2, _normalize_camera)

    for item in _fact_items(scene.workspace_boundaries):
        if item.region_approximation is not GeometryApproximationV2.EXACT:
            raise CardinalUnsupportedModelV2(
                "UNSUPPORTED_MODEL:NONEXACT_RECTILINEAR_FACT"
            )
    for facts in (scene.workspace_boundaries, scene.known_free_spaces):
        for item in _fact_items(facts):
            if item.boundary_policy is not RegionBoundaryPolicyV2.CLOSED:
                raise CardinalUnsupportedModelV2(
                    "UNSUPPORTED_MODEL:NONCLOSED_RECTILINEAR_FACT"
                )
            _require_zero_budget(item.geometry_uncertainty)

    return CanonicalSceneV2(
        scene_id=scene.scene_id,
        coordinate_system=scene.coordinate_system,
        objects=objects,
        geometry_instances=geometries,
        collision_bodies=scene.collision_bodies,
        workspace_boundaries=scene.workspace_boundaries,
        known_free_spaces=scene.known_free_spaces,
        support_surfaces=surfaces,
        cameras=cameras,
        baseline_observations=scene.baseline_observations,
    )


def _map_fact_set(
    facts: FactSetV2,
    output_type: type,
    mapper: object,
) -> FactSetV2:
    map_item = mapper
    payload: dict[str, object] = {
        "availability": facts.availability,
        "completeness": facts.completeness,
        "uncertainty": facts.uncertainty,
    }
    for field_name in ("values", "inner_values", "outer_values"):
        values = getattr(facts, field_name)
        payload[field_name] = (
            None if values is None else tuple(map_item(item) for item in values)
        )
    return FactSetV2[output_type].model_validate(payload, strict=True)


def _fact_items(facts: FactSetV2) -> tuple[object, ...]:
    return tuple(
        item
        for values in (facts.values, facts.inner_values, facts.outer_values)
        if values is not None
        for item in values
    )


def _normalize_object(item: CanonicalObjectV2_1) -> CanonicalObjectV2:
    if type(item) is not CanonicalObjectV2_1:
        raise TypeError("object fact has the wrong exact type")
    transform = item.pose.world_from_object
    return CanonicalObjectV2(
        object_id=item.object_id,
        category_id=item.category_id,
        movable=item.movable,
        pose=ObjectPoseV2(
            world_from_object=_identity_rotation_transform(transform.translation)
        ),
        support_assignment=item.support_assignment,
    )


def _normalize_geometry(
    item: GeometryInstanceV2_1,
    object_turns: dict[str, int],
) -> GeometryInstanceV2:
    if type(item) is not GeometryInstanceV2_1:
        raise TypeError("geometry fact has the wrong exact type")
    if item.role is GeometryRoleV2.OCCLUDER:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:OCCLUDER_GEOMETRY")
    if type(item.shape) is not UprightBox3DV2:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:NONBOX_GEOMETRY")
    if item.approximation is not GeometryApproximationV2.EXACT:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:GEOMETRY_APPROXIMATION")
    _require_zero_budget(item.uncertainty)
    owner_turns = 0
    if item.owner_object_id is not None:
        owner_turns = object_turns[item.owner_object_id]
    anchor = item.anchor_from_geometry
    translation = (
        anchor.translation
        if item.owner_object_id is None
        else _rotate_vec3(owner_turns, anchor.translation)
    )
    total_turns = (owner_turns + anchor.quarter_turns_ccw) % 4
    size = item.shape.size_m
    normalized_size = Vec3V2(x=size.y, y=size.x, z=size.z) if total_turns % 2 else size
    return GeometryInstanceV2(
        geometry_id=item.geometry_id,
        owner_object_id=item.owner_object_id,
        role=item.role,
        anchor_from_geometry=_identity_rotation_transform(translation),
        approximation=item.approximation,
        uncertainty=item.uncertainty,
        shape=UprightBox3DV2(size_m=normalized_size),
    )


def _normalize_support_surface(
    item: SupportSurfaceFactV2_1,
    object_turns: dict[str, int],
) -> SupportSurfaceFactV2:
    if type(item) is not SupportSurfaceFactV2_1:
        raise TypeError("support surface fact has the wrong exact type")
    if item.region_approximation is not GeometryApproximationV2.EXACT:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:SUPPORT_APPROXIMATION")
    if item.boundary_policy is not RegionBoundaryPolicyV2.CLOSED:
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:SUPPORT_BOUNDARY")
    _require_zero_budget(item.geometry_uncertainty)
    owner_turns = 0
    if item.owner_object_id is not None:
        owner_turns = object_turns[item.owner_object_id]
    anchor = item.anchor_from_surface
    translation = (
        anchor.translation
        if item.owner_object_id is None
        else _rotate_vec3(owner_turns, anchor.translation)
    )
    total_turns = (owner_turns + anchor.quarter_turns_ccw) % 4
    return SupportSurfaceFactV2(
        surface_id=item.surface_id,
        owner_object_id=item.owner_object_id,
        supporting_body_id=item.supporting_body_id,
        anchor_from_surface=_identity_rotation_transform(translation),
        normal_in_anchor=item.normal_in_anchor,
        region_uv=_rotate_region(total_turns, item.region_uv),
        region_approximation=item.region_approximation,
        boundary_policy=item.boundary_policy,
        geometry_uncertainty=item.geometry_uncertainty,
    )


def _normalize_camera(item: PinholeCameraV2_1) -> PinholeCameraV2:
    if type(item) is not PinholeCameraV2_1:
        raise TypeError("camera fact has the wrong exact type")
    transform = item.world_to_camera
    if transform.quarter_turns_ccw != 0 or transform.translation != Vec3V2(
        x=0.0, y=0.0, z=0.0
    ):
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:CAMERA_NOT_IDENTITY")
    if (
        item.distortion_model is not CameraDistortionModelV2.NONE
        or item.brown_conrady_coefficients is not None
    ):
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:CAMERA_DISTORTION")
    _require_zero_budget(item.calibration_uncertainty)
    payload = item.model_dump(mode="python", warnings="error")
    payload["world_to_camera"] = RigidTransformV2.identity().model_dump(
        mode="python", warnings="error"
    )
    return PinholeCameraV2.model_validate(payload, strict=True)


def _rotate_region(quarter_turns: int, region: PlanarRegionV2) -> PlanarRegionV2:
    return PlanarRegionV2(
        components=tuple(
            PlanarPolygonComponentV2(
                exterior=_rotate_ring(quarter_turns, component.exterior),
                holes=tuple(
                    _rotate_ring(quarter_turns, hole) for hole in component.holes
                ),
            )
            for component in region.components
        )
    )


def _rotate_ring(quarter_turns: int, ring: PlanarRingV2) -> PlanarRingV2:
    return PlanarRingV2(
        winding=ring.winding,
        vertices=tuple(
            Vec2V2(x=x, y=y)
            for point in ring.vertices
            for x, y in (_rotate_xy(quarter_turns, point.x, point.y),)
        ),
    )


def _rotate_vec3(quarter_turns: int, value: Vec3V2) -> Vec3V2:
    x, y = _rotate_xy(quarter_turns, value.x, value.y)
    return Vec3V2(x=x, y=y, z=value.z)


def _rotate_xy(quarter_turns: int, x: float, y: float) -> tuple[float, float]:
    if quarter_turns == 0:
        result = x, y
    elif quarter_turns == 1:
        result = -y, x
    elif quarter_turns == 2:
        result = -x, -y
    else:
        result = y, -x
    return tuple(0.0 if value == 0.0 else value for value in result)


def _identity_rotation_transform(translation: Vec3V2) -> RigidTransformV2:
    return RigidTransformV2(
        translation=translation,
        rotation=RigidTransformV2.identity().rotation,
    )


def _require_zero_fact_uncertainty(facts: FactSetV2) -> None:
    if facts.availability is FactAvailabilityV2.KNOWN:
        if facts.uncertainty is None:
            raise TypeError("KNOWN facts omitted their uncertainty budget")
        _require_zero_budget(facts.uncertainty)


def _require_zero_budget(budget: UncertaintyBudgetV2) -> None:
    if not (
        _zero_numeric_policy(budget.source_error)
        and _zero_numeric_policy(budget.shape_approximation)
    ):
        raise CardinalUnsupportedModelV2("UNSUPPORTED_MODEL:NONZERO_UNCERTAINTY")


def _zero_numeric_policy(policy: NumericPolicyV2) -> bool:
    return all(value == 0.0 for value in policy.model_dump(mode="python").values())


__all__ = (
    "CARDINAL_ALGORITHM_VERSION_V2_1",
    "CARDINAL_KERNEL_VERSION_V2_1",
    "CardinalDomainBudgetV2",
    "CardinalResourceLimitV2",
    "CardinalUnsupportedModelV2",
    "PreparedCardinalProblemV2_1",
    "prepare_cardinal_problem_v2_1",
    "registry_finding_v2_1",
    "reserve_cardinal_problem_structure_v2_1",
)
