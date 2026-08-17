"""Exact camera-translation capability and preprocessing budget."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, TypeVar

from spatialcf.core.v2._internal.resources.domain_operations import (
    DomainOperationBudgetV2,
)
from spatialcf.core.v2.candidate_domain import CANDIDATE_DOMAIN_ALGORITHM_ID_V2
from spatialcf.core.v2.cardinal_yaw import CARDINAL_KERNEL_VERSION_V2_1
from spatialcf.core.v2.rectilinear_kernel import (
    RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M,
    RECTILINEAR_KERNEL_ID_V2,
)
from spatialcf.core.v2.zero_distortion import (
    ZERO_DISTORTION_ALGORITHM_VERSION_V2_2,
    reserve_zero_distortion_problem_structure_v2_2,
)
from spatialcf.domain.v2.artifacts import CandidateDomainVariableV2
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    Vec2V2,
    Vec3V2,
)
from spatialcf.domain.v2.cardinal import (
    CanonicalObjectV2_1,
    CanonicalSceneV2_1,
    ExactCardinalYawTransformV2,
    GeometryInstanceV2_1,
    ObjectPoseV2_1,
    PinholeCameraV2_1,
    SemanticProblemV2_1,
    SupportSurfaceFactV2_1,
)
from spatialcf.domain.v2.geometry import (
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
)
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    DirectedOutwardGeometryKernelSpecV2,
)
from spatialcf.domain.v2.scene import (
    KnownFreeSpaceFactV2,
    WorkspaceBoundaryFactV2,
)

FactItemT = TypeVar("FactItemT")

CAMERA_TRANSLATION_ALGORITHM_VERSION_V2_3 = "algorithm:2.3"


class CameraTranslationResourceLimitV2(RuntimeError):
    pass


class CameraTranslationUnsupportedModelV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class CameraTranslationNumericGapV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


@dataclass(slots=True)
class CameraTranslationDomainBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = (
        CameraTranslationResourceLimitV2
    )


@dataclass(frozen=True, slots=True)
class PreparedCameraTranslationProblemV2_3:
    normalized_problem: SemanticProblemV2_1
    internal_config: CoreSolverConfigV2
    preprocessing_domain_operations: int


def registry_finding_v2_3(config: CoreSolverConfigV2) -> str | None:
    if (
        config.algorithm_id != CANDIDATE_DOMAIN_ALGORITHM_ID_V2
        or config.algorithm_version != CAMERA_TRANSLATION_ALGORITHM_VERSION_V2_3
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


def reserve_camera_translation_problem_structure_v2_3(
    problem: SemanticProblemV2_1,
    budget: CameraTranslationDomainBudgetV2,
) -> None:
    """Reserve all frozen rebasing loops before inspecting coordinates."""

    reserve_zero_distortion_problem_structure_v2_2(problem, budget)
    scene = problem.scene
    if type(scene) is not CanonicalSceneV2_1:
        raise TypeError("camera-translation scene has the wrong exact type")

    budget.consume()
    references = _camera_reference_ids(problem)
    budget.consume(len(references))

    for family_name in (
        "objects",
        "geometry_instances",
        "support_surfaces",
        "cameras",
    ):
        facts = getattr(scene, family_name)
        if not isinstance(facts, FactSetV2):
            raise TypeError(f"{family_name} has the wrong fact-set type")
        budget.consume(len(_fact_items(facts)))

    for family_name in ("workspace_boundaries", "known_free_spaces"):
        facts = getattr(scene, family_name)
        if not isinstance(facts, FactSetV2):
            raise TypeError(f"{family_name} has the wrong fact-set type")
        for item in _fact_items(facts):
            budget.consume()
            components = item.region_world_xy.components
            budget.consume(len(components))
            for component in components:
                budget.consume(1 + len(component.exterior.vertices))
                budget.consume(len(component.holes))
                for hole in component.holes:
                    budget.consume(len(hole.vertices))

    budget.consume(2)


def prepare_camera_translation_problem_v2_3(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    preprocessing_domain_operations: int,
) -> PreparedCameraTranslationProblemV2_3:
    scene = problem.scene
    camera = _single_evaluation_camera(problem)
    transform = camera.world_to_camera
    if transform.quarter_turns_ccw != 0:
        raise CameraTranslationUnsupportedModelV2("UNSUPPORTED_MODEL:CAMERA_ROTATION")
    shift = transform.translation

    objects = _map_fact_set(
        scene.objects,
        CanonicalObjectV2_1,
        lambda item: _shift_object(item, shift),
    )
    geometries = _map_fact_set(
        scene.geometry_instances,
        GeometryInstanceV2_1,
        lambda item: _shift_geometry(item, shift),
    )
    surfaces = _map_fact_set(
        scene.support_surfaces,
        SupportSurfaceFactV2_1,
        lambda item: _shift_surface(item, shift),
    )
    workspaces = _map_fact_set(
        scene.workspace_boundaries,
        WorkspaceBoundaryFactV2,
        lambda item: _shift_world_region_fact(item, shift),
    )
    free_spaces = _map_fact_set(
        scene.known_free_spaces,
        KnownFreeSpaceFactV2,
        lambda item: _shift_world_region_fact(item, shift),
    )
    cameras = _map_fact_set(
        scene.cameras,
        PinholeCameraV2_1,
        _zero_camera_translation,
    )
    normalized_scene = CanonicalSceneV2_1.model_validate(
        scene.model_copy(
            update={
                "objects": objects,
                "geometry_instances": geometries,
                "workspace_boundaries": workspaces,
                "known_free_spaces": free_spaces,
                "support_surfaces": surfaces,
                "cameras": cameras,
            }
        ).model_dump(mode="python", warnings="error"),
        strict=True,
    )
    normalized_problem = SemanticProblemV2_1.model_validate(
        problem.model_copy(update={"scene": normalized_scene}).model_dump(
            mode="python",
            warnings="error",
        ),
        strict=True,
    )

    remaining = config.max_domain_operations - preprocessing_domain_operations
    if remaining < 1:
        raise CameraTranslationResourceLimitV2
    config_payload = config.model_dump(mode="python", warnings="error")
    config_payload["algorithm_version"] = ZERO_DISTORTION_ALGORITHM_VERSION_V2_2
    config_payload["max_domain_operations"] = remaining
    return PreparedCameraTranslationProblemV2_3(
        normalized_problem=normalized_problem,
        internal_config=CoreSolverConfigV2.model_validate(
            config_payload,
            strict=True,
        ),
        preprocessing_domain_operations=preprocessing_domain_operations,
    )


def _candidate_variable_for_original_problem_v2_3(
    problem: SemanticProblemV2_1,
) -> CandidateDomainVariableV2:
    """Return the edit variable in the original world coordinate frame."""

    subject_id = problem.constraints.allowed_edit.subject_id
    objects = problem.scene.objects.values
    if objects is None:
        raise TypeError("camera-translation problem omitted exact object facts")
    subject = next((item for item in objects if item.object_id == subject_id), None)
    if type(subject) is not CanonicalObjectV2_1:
        raise TypeError("camera-translation subject has the wrong exact type")
    translation = subject.pose.world_from_object.translation
    return CandidateDomainVariableV2(
        subject_id=subject_id,
        baseline_anchor_world_xy_m=Vec2V2(
            x=translation.x,
            y=translation.y,
        ),
    )


def _single_evaluation_camera(problem: SemanticProblemV2_1) -> PinholeCameraV2_1:
    facts = problem.scene.cameras
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.values is None
        or len(facts.values) != 1
        or facts.inner_values is not None
        or facts.outer_values is not None
    ):
        raise CameraTranslationUnsupportedModelV2(
            "UNSUPPORTED_MODEL:SINGLE_EXACT_CAMERA_REQUIRED"
        )
    camera = facts.values[0]
    if type(camera) is not PinholeCameraV2_1:
        raise TypeError("camera fact has the wrong exact type")
    if _camera_reference_ids(problem) != {camera.camera_id}:
        raise CameraTranslationUnsupportedModelV2(
            "UNSUPPORTED_MODEL:CAMERA_REFERENCE_MISMATCH"
        )
    return camera


def _map_fact_set(
    facts: FactSetV2,
    output_type: type[FactItemT],
    mapper: Callable[[object], FactItemT],
) -> FactSetV2[FactItemT]:
    payload: dict[str, object] = {
        "availability": facts.availability,
        "completeness": facts.completeness,
        "uncertainty": facts.uncertainty,
    }
    for field_name in ("values", "inner_values", "outer_values"):
        values = getattr(facts, field_name)
        payload[field_name] = (
            None if values is None else tuple(mapper(item) for item in values)
        )
    return FactSetV2[output_type].model_validate(payload, strict=True)


def _shift_object(
    item: CanonicalObjectV2_1,
    shift: Vec3V2,
) -> CanonicalObjectV2_1:
    if type(item) is not CanonicalObjectV2_1:
        raise TypeError("object fact has the wrong exact type")
    pose = ObjectPoseV2_1(
        world_from_object=_shift_transform(
            item.pose.world_from_object,
            shift,
            f"object:{item.object_id}",
        )
    )
    return CanonicalObjectV2_1.model_validate(
        item.model_copy(update={"pose": pose}).model_dump(
            mode="python",
            warnings="error",
        ),
        strict=True,
    )


def _shift_geometry(
    item: GeometryInstanceV2_1,
    shift: Vec3V2,
) -> GeometryInstanceV2_1:
    if type(item) is not GeometryInstanceV2_1:
        raise TypeError("geometry fact has the wrong exact type")
    if item.owner_object_id is not None:
        return item
    transform = _shift_transform(
        item.anchor_from_geometry,
        shift,
        f"geometry:{item.geometry_id}",
    )
    return GeometryInstanceV2_1.model_validate(
        item.model_copy(update={"anchor_from_geometry": transform}).model_dump(
            mode="python",
            warnings="error",
        ),
        strict=True,
    )


def _shift_surface(
    item: SupportSurfaceFactV2_1,
    shift: Vec3V2,
) -> SupportSurfaceFactV2_1:
    if type(item) is not SupportSurfaceFactV2_1:
        raise TypeError("support surface fact has the wrong exact type")
    if item.owner_object_id is not None:
        return item
    transform = _shift_transform(
        item.anchor_from_surface,
        shift,
        f"support:{item.surface_id}",
    )
    return SupportSurfaceFactV2_1.model_validate(
        item.model_copy(update={"anchor_from_surface": transform}).model_dump(
            mode="python",
            warnings="error",
        ),
        strict=True,
    )


def _shift_world_region_fact(
    item: WorkspaceBoundaryFactV2 | KnownFreeSpaceFactV2,
    shift: Vec3V2,
) -> WorkspaceBoundaryFactV2 | KnownFreeSpaceFactV2:
    if type(item) not in (WorkspaceBoundaryFactV2, KnownFreeSpaceFactV2):
        raise TypeError("world-region fact has the wrong exact type")
    region = _shift_region(item.region_world_xy, shift, item.fact_id)
    return type(item).model_validate(
        item.model_copy(update={"region_world_xy": region}).model_dump(
            mode="python",
            warnings="error",
        ),
        strict=True,
    )


def _shift_region(
    region: PlanarRegionV2,
    shift: Vec3V2,
    field_name: str,
) -> PlanarRegionV2:
    return PlanarRegionV2(
        components=tuple(
            PlanarPolygonComponentV2(
                exterior=_shift_ring(
                    component.exterior,
                    shift,
                    f"{field_name}:exterior",
                ),
                holes=tuple(
                    _shift_ring(hole, shift, f"{field_name}:hole")
                    for hole in component.holes
                ),
            )
            for component in region.components
        )
    )


def _shift_ring(
    ring: PlanarRingV2,
    shift: Vec3V2,
    field_name: str,
) -> PlanarRingV2:
    return PlanarRingV2(
        winding=ring.winding,
        vertices=tuple(
            Vec2V2(
                x=_exact_add_binary64(point.x, shift.x, f"{field_name}:x"),
                y=_exact_add_binary64(point.y, shift.y, f"{field_name}:y"),
            )
            for point in ring.vertices
        ),
    )


def _shift_transform(
    transform: ExactCardinalYawTransformV2,
    shift: Vec3V2,
    field_name: str,
) -> ExactCardinalYawTransformV2:
    translation = transform.translation
    return ExactCardinalYawTransformV2(
        translation=Vec3V2(
            x=_exact_add_binary64(translation.x, shift.x, f"{field_name}:x"),
            y=_exact_add_binary64(translation.y, shift.y, f"{field_name}:y"),
            z=_exact_add_binary64(translation.z, shift.z, f"{field_name}:z"),
        ),
        quarter_turns_ccw=transform.quarter_turns_ccw,
    )


def _zero_camera_translation(camera: PinholeCameraV2_1) -> PinholeCameraV2_1:
    if type(camera) is not PinholeCameraV2_1:
        raise TypeError("camera fact has the wrong exact type")
    transform = camera.world_to_camera
    if transform.quarter_turns_ccw != 0:
        raise CameraTranslationUnsupportedModelV2("UNSUPPORTED_MODEL:CAMERA_ROTATION")
    return PinholeCameraV2_1.model_validate(
        camera.model_copy(
            update={"world_to_camera": ExactCardinalYawTransformV2.identity()}
        ).model_dump(mode="python", warnings="error"),
        strict=True,
    )


def _exact_add_binary64(left: float, right: float, field_name: str) -> float:
    exact = Fraction.from_float(left) + Fraction.from_float(right)
    try:
        candidate = float(exact)
    except (OverflowError, ValueError) as error:
        raise CameraTranslationNumericGapV2(
            f"NUMERIC_GAP:CAMERA_TRANSLATION:{field_name}"
        ) from error
    if not math.isfinite(candidate) or Fraction.from_float(candidate) != exact:
        raise CameraTranslationNumericGapV2(
            f"NUMERIC_GAP:CAMERA_TRANSLATION:{field_name}"
        )
    return 0.0 if candidate == 0.0 else candidate


def _fact_items(facts: FactSetV2) -> tuple[object, ...]:
    return tuple(
        item
        for values in (facts.values, facts.inner_values, facts.outer_values)
        if values is not None
        for item in values
    )


def _camera_reference_ids(problem: SemanticProblemV2_1) -> set[str]:
    constraints = problem.constraints
    objective = problem.objective
    return {
        constraints.target_relation.camera_id,
        objective.relation_damage.evaluation_camera_id,
        *(item.camera_id for item in constraints.visibility_constraints),
        *(
            item.key.camera_id
            for item in objective.visibility_change.object_camera_weights
        ),
    }


__all__ = (
    "CAMERA_TRANSLATION_ALGORITHM_VERSION_V2_3",
    "CameraTranslationDomainBudgetV2",
    "CameraTranslationNumericGapV2",
    "CameraTranslationResourceLimitV2",
    "CameraTranslationUnsupportedModelV2",
    "PreparedCameraTranslationProblemV2_3",
    "prepare_camera_translation_problem_v2_3",
    "registry_finding_v2_3",
    "reserve_camera_translation_problem_structure_v2_3",
)
