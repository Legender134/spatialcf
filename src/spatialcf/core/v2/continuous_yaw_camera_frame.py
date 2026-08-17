"""Directed upright-camera frame and private T15 problem projection for v2.9."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from fractions import Fraction

from spatialcf.core.v2 import so2_interval
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
    compile_directed_sin_cos_v2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetV2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_camera import (
    PinholeCameraV2_3,
    SemanticProblemV2_3,
)
from spatialcf.domain.v2.continuous_yaw_candidate import SemanticProblemV2_2
from spatialcf.domain.v2.scene import (
    CameraAxesV2,
    CameraDepthConventionV2,
    CameraDistortionModelV2,
    CameraMatrixLayoutV2,
    CameraPixelConventionV2,
)

_CAMERA_CONTEXT_HASH_DOMAIN_V2_9 = b"spatialcf.upright-camera-context.v2.9\0"
_CAMERA_PREPARATION_DOMAIN_COST_V2_9 = 16
_IntervalV2 = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class UprightCameraPointBoundsV2_9:
    x_camera: _IntervalV2
    y_camera: _IntervalV2
    z_camera: _IntervalV2
    positive_depth: bool

    def __post_init__(self) -> None:
        for interval in (self.x_camera, self.y_camera, self.z_camera):
            _require_interval(interval)
        if type(self.positive_depth) is not bool:
            raise TypeError("positive_depth must be an exact bool")
        if self.positive_depth is not (self.z_camera[0] > 0):
            raise ValueError("positive_depth does not match the directed Z bound")


@dataclass(frozen=True, slots=True)
class UprightCameraContextV2_9:
    camera_id: str
    width_px: int
    height_px: int
    intrinsics: tuple[Fraction, ...]
    near_clip_m: Fraction
    far_clip_m: Fraction
    translation_xyz: tuple[Fraction, Fraction, Fraction]
    sine: _IntervalV2
    cosine: _IntervalV2

    def __post_init__(self) -> None:
        if type(self.camera_id) is not str or not self.camera_id.strip():
            raise TypeError("camera_id must be an exact non-blank string")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise TypeError("width_px must be a positive exact int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise TypeError("height_px must be a positive exact int")
        if type(self.intrinsics) is not tuple or len(self.intrinsics) != 9:
            raise TypeError("intrinsics must be an exact nine-value tuple")
        for value in (
            *self.intrinsics,
            self.near_clip_m,
            self.far_clip_m,
            *self.translation_xyz,
        ):
            _require_fraction(value)
        if self.near_clip_m <= 0 or self.far_clip_m <= self.near_clip_m:
            raise ValueError("camera clip bounds are reversed")
        _require_interval(self.sine)
        _require_interval(self.cosine)
        if self.sine[0] < -1 or self.sine[1] > 1:
            raise ValueError("sine interval escaped [-1, 1]")
        if self.cosine[0] < -1 or self.cosine[1] > 1:
            raise ValueError("cosine interval escaped [-1, 1]")

    @property
    def context_sha256(self) -> str:
        values = (
            self.camera_id,
            str(self.width_px),
            str(self.height_px),
            *(_fraction_text(value) for value in self.intrinsics),
            _fraction_text(self.near_clip_m),
            _fraction_text(self.far_clip_m),
            *(_fraction_text(value) for value in self.translation_xyz),
            *(_fraction_text(value) for value in self.sine),
            *(_fraction_text(value) for value in self.cosine),
        )
        payload = "\0".join(values).encode("utf-8")
        return hashlib.sha256(_CAMERA_CONTEXT_HASH_DOMAIN_V2_9 + payload).hexdigest()


def prepare_camera_independent_candidate_problem_v2_9(
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


def compile_upright_camera_context_v2_9(
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


def bound_world_point_in_upright_camera_v2_9(
    context: UprightCameraContextV2_9,
    *,
    world_xyz: tuple[Fraction, Fraction, Fraction],
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    atomic_budget: SO2AtomicBudgetV2,
) -> UprightCameraPointBoundsV2_9:
    """Bound one world point plus an XY edit in the compiled camera frame."""

    if type(context) is not UprightCameraContextV2_9:
        raise TypeError("context must be an exact UprightCameraContextV2_9")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(world_xyz) is not tuple or len(world_xyz) != 3:
        raise TypeError("world_xyz must be an exact three-Fraction tuple")
    for value in world_xyz:
        _require_fraction(value)
    _require_interval(delta_x)
    _require_interval(delta_y)

    world_x = _add((world_xyz[0], world_xyz[0]), delta_x, atomic_budget)
    world_y = _add((world_xyz[1], world_xyz[1]), delta_y, atomic_budget)
    tx, ty, tz = context.translation_xyz
    x_camera = _add(
        _subtract(
            _multiply(context.cosine, world_x, atomic_budget),
            _multiply(context.sine, world_y, atomic_budget),
            atomic_budget,
        ),
        (tx, tx),
        atomic_budget,
    )
    y_exact = -world_xyz[2] + ty
    _require_fraction(y_exact)
    z_camera = _add(
        _add(
            _multiply(context.sine, world_x, atomic_budget),
            _multiply(context.cosine, world_y, atomic_budget),
            atomic_budget,
        ),
        (tz, tz),
        atomic_budget,
    )
    return UprightCameraPointBoundsV2_9(
        x_camera=x_camera,
        y_camera=(y_exact, y_exact),
        z_camera=z_camera,
        positive_depth=z_camera[0] > 0,
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
        camera.distortion_model is not CameraDistortionModelV2.NONE
        or camera.brown_conrady_coefficients is not None
        or camera.calibration_uncertainty != UncertaintyBudgetV2()
        or camera.matrix_layout is not CameraMatrixLayoutV2.ROW_MAJOR
        or camera.camera_axes is not CameraAxesV2.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConventionV2.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConventionV2.POSITIVE_Z_FORWARD
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


def _require_fraction(value: Fraction) -> None:
    if type(value) is not Fraction:
        raise TypeError("directed camera values must be exact Fractions")
    so2_interval._require_numeric_fraction_cap(
        value, "NUMERIC_GAP:UPRIGHT_CAMERA_FRACTION_BIT_CAP"
    )


def _require_interval(value: _IntervalV2) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("directed camera intervals must be exact pairs")
    for endpoint in value:
        _require_fraction(endpoint)
    if value[0] > value[1]:
        raise ValueError("directed camera interval endpoints are reversed")


def _multiply(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    result = (min(products), max(products))
    _require_interval(result)
    return result


def _add(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    result = (left[0] + right[0], left[1] + right[1])
    _require_interval(result)
    return result


def _subtract(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    result = (left[0] - right[1], left[1] - right[0])
    _require_interval(result)
    return result


def _fraction_text(value: Fraction) -> str:
    if value.numerator.bit_length() <= 2048 and value.denominator.bit_length() <= 2048:
        return f"{value.numerator}/{value.denominator}"
    numerator_sign = "-" if value.numerator < 0 else "+"
    return (
        "signed-hex-v1:"
        f"{numerator_sign}{format(abs(value.numerator), 'x')}"
        f"/+{format(value.denominator, 'x')}"
    )


__all__ = (
    "UprightCameraContextV2_9",
    "UprightCameraPointBoundsV2_9",
    "bound_world_point_in_upright_camera_v2_9",
    "compile_upright_camera_context_v2_9",
    "prepare_camera_independent_candidate_problem_v2_9",
)
