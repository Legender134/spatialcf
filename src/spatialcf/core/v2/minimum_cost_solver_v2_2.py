"""Pure-core solve entry points for the zero-distortion 2.2 capability."""

from __future__ import annotations

from spatialcf.core.v2.certificate_builder import _ExactCardinalSelectionFrameV2
from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.result import CoreSolverConfigV2


def solve_canonical_minimum_cost_v2_2(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve an exact identity-projection 2.1 problem under algorithm 2.2."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        solve_registered_capability_v2,
    )

    return solve_registered_capability_v2(
        SolveCapabilityKeyV2.V2_2,
        problem,
        config,
    )


def _solve_canonical_minimum_cost_v2_2_in_selection_frame(
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
        SolveCapabilityKeyV2.V2_2,
        problem,
        config,
        selection_frame,
    )


class CanonicalMinimumCostSolverV2_2:
    """Stateless wrapper for the zero-distortion 2.2 solve."""

    def solve(
        self,
        problem: SemanticProblemV2_1,
        config: CoreSolverConfigV2,
    ) -> CanonicalMinimumCostSolveOutcomeV2:
        return solve_canonical_minimum_cost_v2_2(problem, config)


__all__ = (
    "CanonicalMinimumCostSolverV2_2",
    "solve_canonical_minimum_cost_v2_2",
)
