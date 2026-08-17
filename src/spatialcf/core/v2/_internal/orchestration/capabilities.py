"""Frozen private values describing Canonical v2 solve capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import (
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_5,
    StrictConvexCandidateCompilerConfigV2_6,
    StrictConvexCandidateCompilerConfigV2_7,
)
from spatialcf.domain.v2.continuous_yaw_solver import (
    ContinuousYawMinimumCostSolveOutcomeV2_8,
    ContinuousYawSolverConfigV2_8,
)
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawMinimumCostSolveOutcomeV2_9,
    ContinuousYawSolverConfigV2_9,
)


class SolveCapabilityKeyV2(StrEnum):
    """Closed identities authorized to run the complete solve pipeline."""

    V2_0 = "algorithm:2.0"
    V2_1 = "algorithm:2.1"
    V2_2 = "algorithm:2.2"
    V2_3 = "algorithm:2.3"
    V2_4 = "algorithm:2.4"
    V2_8 = "algorithm:2.8"
    V2_9 = "algorithm:2.9"


class SolveStageKeyV2(StrEnum):
    """Closed normalization stages composed ahead of the base solve."""

    CARDINAL_YAW = "cardinal-yaw"
    ZERO_DISTORTION = "zero-distortion"
    CAMERA_TRANSLATION = "camera-translation"
    CAMERA_CARDINAL_REBASE = "camera-cardinal-rebase"


class CompilerPrefixKeyV2(StrEnum):
    """Closed identities for incomplete candidate compiler prefixes."""

    V2_5 = "algorithm:2.5"
    V2_6 = "algorithm:2.6"
    V2_7 = "algorithm:2.7"


@dataclass(frozen=True, slots=True)
class SolveCapabilityV2:
    """One immutable, module-owned complete solve capability."""

    algorithm_id: str
    algorithm_version: SolveCapabilityKeyV2
    problem_type: type[V2Model]
    stage_keys: tuple[SolveStageKeyV2, ...]

    def __post_init__(self) -> None:
        if type(self.algorithm_id) is not str or not self.algorithm_id.strip():
            raise TypeError("algorithm_id must be an exact non-blank string")
        if type(self.algorithm_version) is not SolveCapabilityKeyV2:
            raise TypeError("algorithm_version has the wrong exact type")
        if self.algorithm_version in (
            SolveCapabilityKeyV2.V2_8,
            SolveCapabilityKeyV2.V2_9,
        ):
            raise ValueError(
                f"{self.algorithm_version.value} requires DirectSolveCapabilityV2"
            )
        if not isinstance(self.problem_type, type) or not issubclass(
            self.problem_type,
            V2Model,
        ):
            raise TypeError("problem_type must be a V2Model type")
        if type(self.stage_keys) is not tuple or any(
            type(item) is not SolveStageKeyV2 for item in self.stage_keys
        ):
            raise TypeError("stage_keys must be an exact tuple of SolveStageKeyV2")
        if len(set(self.stage_keys)) != len(self.stage_keys):
            raise ValueError("stage keys must be unique")


@dataclass(frozen=True, slots=True)
class DirectSolveCapabilityV2:
    """One raw-input complete solver that does not use normalization stages."""

    algorithm_id: str
    algorithm_version: SolveCapabilityKeyV2
    problem_type: type[SemanticProblemV2_2 | SemanticProblemV2_3]
    config_type: type[ContinuousYawSolverConfigV2_8 | ContinuousYawSolverConfigV2_9]
    solver: (
        Callable[
            [SemanticProblemV2_2, ContinuousYawSolverConfigV2_8],
            ContinuousYawMinimumCostSolveOutcomeV2_8,
        ]
        | Callable[
            [SemanticProblemV2_3, ContinuousYawSolverConfigV2_9],
            ContinuousYawMinimumCostSolveOutcomeV2_9,
        ]
    )

    def __post_init__(self) -> None:
        if type(self.algorithm_id) is not str or not self.algorithm_id.strip():
            raise TypeError("algorithm_id must be an exact non-blank string")
        expected_types = {
            SolveCapabilityKeyV2.V2_8: (
                SemanticProblemV2_2,
                ContinuousYawSolverConfigV2_8,
            ),
            SolveCapabilityKeyV2.V2_9: (
                SemanticProblemV2_3,
                ContinuousYawSolverConfigV2_9,
            ),
        }
        if self.algorithm_version not in expected_types:
            raise TypeError("direct solver version is not registered")
        expected_problem, expected_config = expected_types[self.algorithm_version]
        if self.problem_type is not expected_problem:
            raise TypeError("direct solver problem_type is not exact")
        if self.config_type is not expected_config:
            raise TypeError("direct solver config_type is not exact")
        if not callable(self.solver):
            raise TypeError("direct solver must be callable")


@dataclass(frozen=True, slots=True)
class AvailableCompilerPrefixV2:
    """One exact compiler-only capability with no certificate authority."""

    algorithm_version: CompilerPrefixKeyV2
    problem_type: type[SemanticProblemV2_2]
    config_type: type[
        StrictConvexCandidateCompilerConfigV2_5
        | StrictConvexCandidateCompilerConfigV2_6
        | StrictConvexCandidateCompilerConfigV2_7
    ]
    compiler: Callable[..., object]

    def __post_init__(self) -> None:
        if type(self.algorithm_version) is not CompilerPrefixKeyV2:
            raise TypeError(
                "available prefix algorithm_version has the wrong exact type"
            )
        if self.problem_type is not SemanticProblemV2_2:
            raise TypeError("available prefix problem_type is not exact")
        expected_config_type = {
            CompilerPrefixKeyV2.V2_5: StrictConvexCandidateCompilerConfigV2_5,
            CompilerPrefixKeyV2.V2_6: StrictConvexCandidateCompilerConfigV2_6,
            CompilerPrefixKeyV2.V2_7: StrictConvexCandidateCompilerConfigV2_7,
        }[self.algorithm_version]
        if self.config_type is not expected_config_type:
            raise TypeError("available prefix config_type is not exact")
        if not callable(self.compiler):
            raise TypeError("available prefix compiler must be callable")


@dataclass(frozen=True, slots=True)
class ReservedCompilerPrefixV2:
    """A frozen route identity that deliberately carries no compiler."""

    algorithm_version: CompilerPrefixKeyV2
    route: str

    def __post_init__(self) -> None:
        if self.algorithm_version is not CompilerPrefixKeyV2.V2_7:
            raise TypeError("reserved prefix must be exact algorithm:2.7")
        if type(self.route) is not str or not self.route.strip():
            raise TypeError("reserved prefix route must be an exact non-blank string")


__all__ = (
    "AvailableCompilerPrefixV2",
    "CompilerPrefixKeyV2",
    "DirectSolveCapabilityV2",
    "ReservedCompilerPrefixV2",
    "SolveCapabilityKeyV2",
    "SolveCapabilityV2",
    "SolveStageKeyV2",
)
