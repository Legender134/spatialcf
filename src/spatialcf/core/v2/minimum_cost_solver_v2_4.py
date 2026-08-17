"""Pure-core solve for an exact translated cardinal-yaw camera."""

from __future__ import annotations

from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.result import CoreSolverConfigV2


def solve_canonical_minimum_cost_v2_4(
    problem: SemanticProblemV2_1,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Solve through one exact cardinal rebase and one private 2.3 replay."""

    from spatialcf.core.v2._internal.orchestration.capabilities import (
        SolveCapabilityKeyV2,
    )
    from spatialcf.core.v2._internal.orchestration.solve import (
        solve_registered_capability_v2,
    )

    return solve_registered_capability_v2(
        SolveCapabilityKeyV2.V2_4,
        problem,
        config,
    )


class CanonicalMinimumCostSolverV2_4:
    """Stateless wrapper for exact cardinal-camera rebasing."""

    def solve(
        self,
        problem: SemanticProblemV2_1,
        config: CoreSolverConfigV2,
    ) -> CanonicalMinimumCostSolveOutcomeV2:
        return solve_canonical_minimum_cost_v2_4(problem, config)


__all__ = (
    "CanonicalMinimumCostSolverV2_4",
    "solve_canonical_minimum_cost_v2_4",
)
