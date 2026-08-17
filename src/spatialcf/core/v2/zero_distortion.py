"""Exact identity-projection capability for zero Brown-Conrady distortion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from spatialcf.core.v2._internal.resources.domain_operations import (
    DomainOperationBudgetV2,
)
from spatialcf.core.v2.candidate_domain import CANDIDATE_DOMAIN_ALGORITHM_ID_V2
from spatialcf.core.v2.cardinal_yaw import (
    CARDINAL_ALGORITHM_VERSION_V2_1,
    CARDINAL_KERNEL_VERSION_V2_1,
    reserve_cardinal_problem_structure_v2_1,
)
from spatialcf.core.v2.rectilinear_kernel import (
    RECTILINEAR_KERNEL_CERTIFIED_OUTWARD_ERROR_M,
    RECTILINEAR_KERNEL_ID_V2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactSetV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.cardinal import (
    CanonicalSceneV2_1,
    PinholeCameraV2_1,
    SemanticProblemV2_1,
)
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    DirectedOutwardGeometryKernelSpecV2,
)
from spatialcf.domain.v2.scene import (
    BrownConradyCoefficientsV2,
    CameraDistortionModelV2,
)

ZERO_DISTORTION_ALGORITHM_VERSION_V2_2 = "algorithm:2.2"


class ZeroDistortionResourceLimitV2(RuntimeError):
    pass


class ZeroDistortionUnsupportedModelV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


@dataclass(slots=True)
class ZeroDistortionDomainBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = ZeroDistortionResourceLimitV2


@dataclass(frozen=True, slots=True)
class PreparedZeroDistortionProblemV2_2:
    normalized_problem: SemanticProblemV2_1
    internal_config: CoreSolverConfigV2
    preprocessing_domain_operations: int


def reserve_zero_distortion_problem_structure_v2_2(
    problem: SemanticProblemV2_1,
    budget: ZeroDistortionDomainBudgetV2,
) -> None:
    """Reserve the frozen camera pass before inspecting any camera deeply."""

    reserve_cardinal_problem_structure_v2_1(problem, budget)
    scene = problem.scene
    if type(scene) is not CanonicalSceneV2_1:
        raise TypeError("zero-distortion problem scene has the wrong exact type")
    cameras = scene.cameras
    if not isinstance(cameras, FactSetV2):
        raise TypeError("camera facts have the wrong exact type")
    branches = tuple(
        values
        for values in (cameras.values, cameras.inner_values, cameras.outer_values)
        if values is not None
    )
    budget.consume(1 + sum(len(values) for values in branches))


def prepare_zero_distortion_problem_v2_2(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    preprocessing_domain_operations: int,
) -> PreparedZeroDistortionProblemV2_2:
    cameras = _normalize_camera_facts(problem.scene.cameras)
    normalized_scene = CanonicalSceneV2_1.model_validate(
        problem.scene.model_copy(update={"cameras": cameras}).model_dump(
            mode="python",
            warnings="error",
        ),
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
        raise ZeroDistortionResourceLimitV2
    payload = config.model_dump(mode="python", warnings="error")
    payload["algorithm_version"] = CARDINAL_ALGORITHM_VERSION_V2_1
    payload["max_domain_operations"] = remaining
    return PreparedZeroDistortionProblemV2_2(
        normalized_problem=normalized_problem,
        internal_config=CoreSolverConfigV2.model_validate(payload, strict=True),
        preprocessing_domain_operations=preprocessing_domain_operations,
    )


def registry_finding_v2_2(config: CoreSolverConfigV2) -> str | None:
    if (
        config.algorithm_id != CANDIDATE_DOMAIN_ALGORITHM_ID_V2
        or config.algorithm_version != ZERO_DISTORTION_ALGORITHM_VERSION_V2_2
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


def camera_has_exact_identity_projection_v2(camera: PinholeCameraV2_1) -> bool:
    """Return whether distortion is exactly identity with zero uncertainty."""

    if type(camera) is not PinholeCameraV2_1:
        return False
    coefficients = camera.brown_conrady_coefficients
    if camera.distortion_model is CameraDistortionModelV2.NONE:
        distortion_is_identity = coefficients is None
    else:
        distortion_is_identity = (
            camera.distortion_model is CameraDistortionModelV2.BROWN_CONRADY
            and type(coefficients) is BrownConradyCoefficientsV2
            and all(
                value == 0.0
                for value in (
                    coefficients.k1,
                    coefficients.k2,
                    coefficients.p1,
                    coefficients.p2,
                    coefficients.k3,
                )
            )
        )
    return distortion_is_identity and _zero_uncertainty(camera.calibration_uncertainty)


def _zero_uncertainty(budget: UncertaintyBudgetV2) -> bool:
    return _zero_policy(budget.source_error) and _zero_policy(
        budget.shape_approximation
    )


def _zero_policy(policy: NumericPolicyV2) -> bool:
    return all(value == 0.0 for value in policy.model_dump(mode="python").values())


def _normalize_camera_facts(
    facts: FactSetV2[PinholeCameraV2_1],
) -> FactSetV2[PinholeCameraV2_1]:
    if facts.availability is FactAvailabilityV2.KNOWN:
        uncertainty = facts.uncertainty
        if uncertainty is None or not _zero_uncertainty(uncertainty):
            raise ZeroDistortionUnsupportedModelV2(
                "UNSUPPORTED_MODEL:NONZERO_CAMERA_FACT_UNCERTAINTY"
            )
    payload: dict[str, object] = {
        "availability": facts.availability,
        "completeness": facts.completeness,
        "uncertainty": facts.uncertainty,
    }
    for field_name in ("values", "inner_values", "outer_values"):
        values = getattr(facts, field_name)
        payload[field_name] = (
            None
            if values is None
            else tuple(_normalize_camera(item) for item in values)
        )
    return FactSetV2[PinholeCameraV2_1].model_validate(payload, strict=True)


def _normalize_camera(camera: PinholeCameraV2_1) -> PinholeCameraV2_1:
    if type(camera) is not PinholeCameraV2_1:
        raise TypeError("camera fact has the wrong exact type")
    if not _zero_uncertainty(camera.calibration_uncertainty):
        raise ZeroDistortionUnsupportedModelV2(
            "UNSUPPORTED_MODEL:NONZERO_CAMERA_CALIBRATION_UNCERTAINTY"
        )
    coefficients = camera.brown_conrady_coefficients
    if (
        camera.distortion_model is CameraDistortionModelV2.BROWN_CONRADY
        and type(coefficients) is BrownConradyCoefficientsV2
        and any(
            value != 0.0
            for value in (
                coefficients.k1,
                coefficients.k2,
                coefficients.p1,
                coefficients.p2,
                coefficients.k3,
            )
        )
    ):
        raise ZeroDistortionUnsupportedModelV2(
            "UNSUPPORTED_MODEL:NONZERO_BROWN_CONRADY"
        )
    if not camera_has_exact_identity_projection_v2(camera):
        raise ZeroDistortionUnsupportedModelV2(
            "UNSUPPORTED_MODEL:NONIDENTITY_CAMERA_PROJECTION"
        )
    if camera.distortion_model is CameraDistortionModelV2.NONE:
        return camera
    payload = camera.model_dump(mode="python", warnings="error")
    payload["distortion_model"] = CameraDistortionModelV2.NONE
    payload["brown_conrady_coefficients"] = None
    return PinholeCameraV2_1.model_validate(payload, strict=True)


__all__ = (
    "ZERO_DISTORTION_ALGORITHM_VERSION_V2_2",
    "PreparedZeroDistortionProblemV2_2",
    "ZeroDistortionDomainBudgetV2",
    "ZeroDistortionResourceLimitV2",
    "ZeroDistortionUnsupportedModelV2",
    "camera_has_exact_identity_projection_v2",
    "prepare_zero_distortion_problem_v2_2",
    "registry_finding_v2_2",
    "reserve_zero_distortion_problem_structure_v2_2",
)
