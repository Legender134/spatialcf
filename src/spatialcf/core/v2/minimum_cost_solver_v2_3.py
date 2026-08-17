"""Pure-core solve entry points for the exact camera-translation 2.3 capability."""

from __future__ import annotations

from spatialcf.core.v2.certificate_builder import _ExactCardinalSelectionFrameV2
from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.result import CoreSolverConfigV2


def solve_canonical_minimum_cost_v2_3(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve one exact translated-camera problem through the common engine."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        solve_registered_capability_v2,
    )

    return solve_registered_capability_v2(
        SolveCapabilityKeyV2.V2_3,
        problem,
        config,
    )


def _solve_canonical_minimum_cost_v2_3_in_selection_frame(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Compatibility seam using the common engine with an exact frame."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        _solve_registered_capability_in_selection_frame_v2,
    )

    return _solve_registered_capability_in_selection_frame_v2(
        SolveCapabilityKeyV2.V2_3,
        problem,
        config,
        selection_frame,
    )


class CanonicalMinimumCostSolverV2_3:
    """Stateless wrapper for the exact camera-translation 2.3 solve."""

    def solve(
        self,
        problem: SemanticProblemV2_1,
        config: CoreSolverConfigV2,
    ) -> CanonicalMinimumCostSolveOutcomeV2:
        return solve_canonical_minimum_cost_v2_3(problem, config)


__all__ = (
    "CanonicalMinimumCostSolverV2_3",
    "solve_canonical_minimum_cost_v2_3",
)
