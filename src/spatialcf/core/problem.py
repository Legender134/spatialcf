"""Current camera-normalized semantic-problem compilation owner."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from fractions import Fraction

from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.so2 import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
    compile_directed_sin_cos_v2,
)
from spatialcf.core._internal.kernels.strict_convex import (
    StrictConvexIntersectionBudgetV2,
)
from spatialcf.domain.artifacts import SemanticProblemV2_2
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.geometry import DirectedYawIntervalTransformV2_2
from spatialcf.domain.problem import (
    PinholeCameraV2_3,
    SemanticProblemV2_3,
)
from spatialcf.domain.scene import (
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
)

from spatialcf.core._internal.kernels.camera import (
    _CAMERA_CONTEXT_HASH_DOMAIN_V2_9,
    _IntervalV2,
    UprightCameraPointBoundsV2_9,
    UprightCameraContextV2_9,
    bound_world_point_in_upright_camera,
    _require_fraction,
    _require_interval,
    _multiply,
    _add,
    _subtract,
    _fraction_text,
)

_CAMERA_PREPARATION_DOMAIN_COST_V2_9 = 16


def prepare_camera_independent_candidate_problem(
    problem: SemanticProblemV2_3,
) -> SemanticProblemV2_2:
    """Project only the camera wire for the camera-independent T15 compiler."""

    checked = _strict_problem(problem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        payload = checked.model_dump(mode="python", warnings="error")
        payload["schema_identity"] = {
            "schema_name": "semantic-problem",
            "schema_version": "2.2",
        }
        scene = payload["scene"]
        scene["schema_identity"] = {
            "schema_name": "canonical-scene",
            "schema_version": "2.2",
        }
        cameras = scene["cameras"]["values"]
        if type(cameras) is not tuple:
            raise RuntimeError("strict camera fact set lost its exact tuple")
        projected_cameras = []
        for camera in cameras:
            if type(camera) is not dict:
                raise RuntimeError("strict camera dump lost its mapping")
            projected = dict(camera)
            transform = camera["world_to_camera"]
            if type(transform) is not dict:
                raise RuntimeError("strict camera transform lost its mapping")
            projected["world_to_camera"] = DirectedYawIntervalTransformV2_2(
                translation=transform["translation"],
                yaw_radians=0.0,
            ).model_dump(mode="python")
            projected_cameras.append(projected)
        scene["cameras"]["values"] = tuple(projected_cameras)
        projected_problem = SemanticProblemV2_2.model_validate(payload, strict=True)
        _require_non_camera_projection_closure(checked, projected_problem)
        return projected_problem


def compile_upright_camera_context(
    problem: SemanticProblemV2_3,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    domain_budget: StrictConvexIntersectionBudgetV2,
) -> UprightCameraContextV2_9:
    """Compile one exact directed upright-camera context on shared ledgers."""

    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    if type(domain_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError(
            "domain_budget must be an exact StrictConvexIntersectionBudgetV2"
        )
    atomic_budget.validate()
    domain_budget.consume_domain(_CAMERA_PREPARATION_DOMAIN_COST_V2_9)
    checked = _strict_problem(problem)
    camera = _evaluation_camera(checked)
    transform = camera.world_to_camera
    outcome = compile_directed_sin_cos_v2(
        0.0 if transform.azimuth_radians == 0.0 else transform.azimuth_radians,
        atomic_budget=atomic_budget,
    )
    if outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        raise SO2AtomicBudgetExhaustedV2
    if outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
        raise ArithmeticError("directed upright-camera trigonometry failed")
    if outcome.kind is not SO2IntervalKindV2.EXACT or outcome.bounds is None:
        raise RuntimeError("supported upright camera did not produce sin/cos bounds")
    bounds = outcome.bounds
    return UprightCameraContextV2_9(
        camera_id=camera.camera_id,
        width_px=camera.width_px,
        height_px=camera.height_px,
        intrinsics=tuple(
            Fraction.from_float(value) for value in camera.intrinsics_row_major
        ),
        near_clip_m=Fraction.from_float(camera.near_clip_m),
        far_clip_m=Fraction.from_float(camera.far_clip_m),
        translation_xyz=tuple(
            Fraction.from_float(value)
            for value in (
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            )
        ),
        sine=(bounds.sine.rational_lower, bounds.sine.rational_upper),
        cosine=(bounds.cosine.rational_lower, bounds.cosine.rational_upper),
    )


def _strict_problem(problem: SemanticProblemV2_3) -> SemanticProblemV2_3:
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        payload = problem.model_dump(mode="python", warnings="error")
        return SemanticProblemV2_3.model_validate(payload, strict=True)


def _evaluation_camera(problem: SemanticProblemV2_3) -> PinholeCameraV2_3:
    facts = problem.scene.cameras
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.uncertainty != UncertaintyBudgetV2()
        or type(facts.values) is not tuple
        or len(facts.values) != 1
    ):
        raise ValueError("v2.9 requires one exact camera fact")
    camera = facts.values[0]
    if type(camera) is not PinholeCameraV2_3:
        raise RuntimeError("Canonical 2.3 camera fact lost its exact type")
    camera_ids = {
        problem.objective.relation_damage.evaluation_camera_id,
        *(
            item.key.camera_id
            for item in problem.objective.visibility_change.object_camera_weights
        ),
        *(item.camera_id for item in problem.constraints.visibility_constraints),
    }
    if camera_ids != {camera.camera_id}:
        raise ValueError("v2.9 camera references must select one evaluation camera")
    intrinsics = camera.intrinsics_row_major
    if (
        camera.distortion_model is not CameraDistortionModel.NONE
        or camera.brown_conrady_coefficients is not None
        or camera.calibration_uncertainty != UncertaintyBudgetV2()
        or camera.matrix_layout is not CameraMatrixLayout.ROW_MAJOR
        or camera.camera_axes is not CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConvention.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConvention.POSITIVE_Z_FORWARD
        or intrinsics[1] != 0.0
        or intrinsics[3] != 0.0
        or intrinsics[6:] != (0.0, 0.0, 1.0)
    ):
        raise ValueError("v2.9 camera lies outside the exact upright subset")
    return camera


def _require_non_camera_projection_closure(
    original: SemanticProblemV2_3,
    projected: SemanticProblemV2_2,
) -> None:
    left = original.model_dump(mode="python", warnings="error")
    right = projected.model_dump(mode="python", warnings="error")
    left.pop("schema_identity")
    right.pop("schema_identity")
    left_scene = left["scene"]
    right_scene = right["scene"]
    left_scene.pop("schema_identity")
    right_scene.pop("schema_identity")
    left_scene.pop("cameras")
    right_scene.pop("cameras")
    if left != right:
        raise RuntimeError("camera-independent candidate projection changed semantics")


__all__ = (
    "UprightCameraContextV2_9",
    "UprightCameraPointBoundsV2_9",
    "bound_world_point_in_upright_camera",
    "compile_upright_camera_context",
    "prepare_camera_independent_candidate_problem",
)
