"""Immutable private registries for Canonical v2 orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from spatialcf.core.v2._internal.orchestration.capabilities import (
    AvailableCompilerPrefixV2,
    CompilerPrefixKeyV2,
    DirectSolveCapabilityV2,
    ReservedCompilerPrefixV2,
    SolveCapabilityKeyV2,
    SolveCapabilityV2,
    SolveStageKeyV2,
)
from spatialcf.core.v2.continuous_yaw_solver import (
    solve_continuous_yaw_minimum_cost_v2_8,
)
from spatialcf.core.v2.continuous_yaw_solver_v2_9 import (
    solve_continuous_yaw_minimum_cost_v2_9,
)
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateCompilationOutcomeV2,
    compile_multi_obstacle_strict_convex_candidate_domain_v2_6,
)
from spatialcf.core.v2.strict_convex_candidate_domain import (
    StrictConvexCandidateCompilationOutcomeV2,
    compile_strict_convex_candidate_domain_v2_5,
)
from spatialcf.core.v2.support_strict_convex_candidate_domain import (
    SupportStrictConvexCandidateCompilationOutcomeV2,
    compile_support_strict_convex_candidate_domain_v2_7,
)
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import (
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_5,
    StrictConvexCandidateCompilerConfigV2_6,
    StrictConvexCandidateCompilerConfigV2_7,
)
from spatialcf.domain.v2.continuous_yaw_solver import ContinuousYawSolverConfigV2_8
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawSolverConfigV2_9,
)
from spatialcf.domain.v2.problem import SemanticProblemV2

_ALGORITHM_ID_V2 = "solver:canonical-branch-and-bound-v2"


def _freeze_solve_registry_v2(
    entries: tuple[SolveCapabilityV2 | DirectSolveCapabilityV2, ...],
) -> Mapping[SolveCapabilityKeyV2, SolveCapabilityV2 | DirectSolveCapabilityV2]:
    if type(entries) is not tuple:
        raise TypeError("solve capability entries must be an exact tuple")
    result: dict[SolveCapabilityKeyV2, SolveCapabilityV2 | DirectSolveCapabilityV2] = {}
    for entry in entries:
        if type(entry) not in (SolveCapabilityV2, DirectSolveCapabilityV2):
            raise TypeError("registry entries must be exact SolveCapabilityV2")
        key = entry.algorithm_version
        if key in result:
            raise ValueError(f"duplicate solve capability: {key.value}")
        result[key] = entry
    return MappingProxyType(result)


SOLVE_CAPABILITIES_V2 = _freeze_solve_registry_v2(
    (
        SolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_0,
            problem_type=SemanticProblemV2,
            stage_keys=(),
        ),
        DirectSolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_8,
            problem_type=SemanticProblemV2_2,
            config_type=ContinuousYawSolverConfigV2_8,
            solver=solve_continuous_yaw_minimum_cost_v2_8,
        ),
        DirectSolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_9,
            problem_type=SemanticProblemV2_3,
            config_type=ContinuousYawSolverConfigV2_9,
            solver=solve_continuous_yaw_minimum_cost_v2_9,
        ),
        SolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_1,
            problem_type=SemanticProblemV2_1,
            stage_keys=(SolveStageKeyV2.CARDINAL_YAW,),
        ),
        SolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_2,
            problem_type=SemanticProblemV2_1,
            stage_keys=(
                SolveStageKeyV2.ZERO_DISTORTION,
                SolveStageKeyV2.CARDINAL_YAW,
            ),
        ),
        SolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_3,
            problem_type=SemanticProblemV2_1,
            stage_keys=(
                SolveStageKeyV2.CAMERA_TRANSLATION,
                SolveStageKeyV2.ZERO_DISTORTION,
                SolveStageKeyV2.CARDINAL_YAW,
            ),
        ),
        SolveCapabilityV2(
            algorithm_id=_ALGORITHM_ID_V2,
            algorithm_version=SolveCapabilityKeyV2.V2_4,
            problem_type=SemanticProblemV2_1,
            stage_keys=(
                SolveStageKeyV2.CAMERA_CARDINAL_REBASE,
                SolveStageKeyV2.CAMERA_TRANSLATION,
                SolveStageKeyV2.ZERO_DISTORTION,
                SolveStageKeyV2.CARDINAL_YAW,
            ),
        ),
    )
)

CompilerPrefixCapabilityV2 = AvailableCompilerPrefixV2 | ReservedCompilerPrefixV2


def _freeze_compiler_prefix_registry_v2(
    entries: tuple[CompilerPrefixCapabilityV2, ...],
) -> Mapping[CompilerPrefixKeyV2, CompilerPrefixCapabilityV2]:
    if type(entries) is not tuple:
        raise TypeError("compiler prefix entries must be an exact tuple")
    result: dict[CompilerPrefixKeyV2, CompilerPrefixCapabilityV2] = {}
    for entry in entries:
        if type(entry) not in (AvailableCompilerPrefixV2, ReservedCompilerPrefixV2):
            raise TypeError("compiler prefix entry has the wrong exact type")
        key = entry.algorithm_version
        if key in result:
            raise ValueError(f"duplicate compiler prefix: {key.value}")
        result[key] = entry
    return MappingProxyType(result)


COMPILER_PREFIX_CAPABILITIES_V2 = _freeze_compiler_prefix_registry_v2(
    (
        AvailableCompilerPrefixV2(
            algorithm_version=CompilerPrefixKeyV2.V2_5,
            problem_type=SemanticProblemV2_2,
            config_type=StrictConvexCandidateCompilerConfigV2_5,
            compiler=compile_strict_convex_candidate_domain_v2_5,
        ),
        AvailableCompilerPrefixV2(
            algorithm_version=CompilerPrefixKeyV2.V2_6,
            problem_type=SemanticProblemV2_2,
            config_type=StrictConvexCandidateCompilerConfigV2_6,
            compiler=compile_multi_obstacle_strict_convex_candidate_domain_v2_6,
        ),
        AvailableCompilerPrefixV2(
            algorithm_version=CompilerPrefixKeyV2.V2_7,
            problem_type=SemanticProblemV2_2,
            config_type=StrictConvexCandidateCompilerConfigV2_7,
            compiler=compile_support_strict_convex_candidate_domain_v2_7,
        ),
    )
)


def resolve_solve_capability_v2(
    key: SolveCapabilityKeyV2,
) -> SolveCapabilityV2 | DirectSolveCapabilityV2:
    """Resolve one exact internal capability without a fallback."""

    if type(key) is not SolveCapabilityKeyV2:
        raise TypeError("key must be an exact SolveCapabilityKeyV2")
    try:
        return SOLVE_CAPABILITIES_V2[key]
    except KeyError as error:
        raise KeyError(f"unknown solve capability: {key.value}") from error


def resolve_compiler_prefix_v2(
    key: CompilerPrefixKeyV2,
) -> CompilerPrefixCapabilityV2:
    """Resolve one exact compiler-prefix capability without fallback."""

    if type(key) is not CompilerPrefixKeyV2:
        raise TypeError("key must be an exact CompilerPrefixKeyV2")
    try:
        return COMPILER_PREFIX_CAPABILITIES_V2[key]
    except KeyError as error:
        raise KeyError(f"unknown compiler prefix: {key.value}") from error


def dispatch_compiler_prefix_v2(
    key: CompilerPrefixKeyV2,
    problem: object,
    config: object,
) -> (
    StrictConvexCandidateCompilationOutcomeV2
    | MultiObstacleStrictConvexCandidateCompilationOutcomeV2
    | SupportStrictConvexCandidateCompilationOutcomeV2
):
    """Run an available compiler prefix; reserved routes cannot dispatch."""

    capability = resolve_compiler_prefix_v2(key)
    if type(capability) is ReservedCompilerPrefixV2:
        raise RuntimeError(f"reserved compiler prefix: {capability.algorithm_version}")
    if type(capability) is not AvailableCompilerPrefixV2:
        raise TypeError("compiler prefix registry returned an invalid entry")
    if type(problem) is not capability.problem_type:
        raise TypeError("problem has the wrong exact type")
    if type(config) is not capability.config_type:
        raise TypeError("config has the wrong exact type")
    outcome = capability.compiler(problem, config)
    expected_outcome_type = {
        CompilerPrefixKeyV2.V2_5: StrictConvexCandidateCompilationOutcomeV2,
        CompilerPrefixKeyV2.V2_6: MultiObstacleStrictConvexCandidateCompilationOutcomeV2,
        CompilerPrefixKeyV2.V2_7: SupportStrictConvexCandidateCompilationOutcomeV2,
    }[key]
    if type(outcome) is not expected_outcome_type:
        raise TypeError("compiler prefix returned an invalid outcome")
    return outcome


__all__ = (
    "COMPILER_PREFIX_CAPABILITIES_V2",
    "SOLVE_CAPABILITIES_V2",
    "dispatch_compiler_prefix_v2",
    "resolve_compiler_prefix_v2",
    "resolve_solve_capability_v2",
)
