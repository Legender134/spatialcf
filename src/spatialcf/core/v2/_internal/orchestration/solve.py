"""Private complete-solve dispatch for registered Canonical v2 capabilities."""

from __future__ import annotations

import warnings

import spatialcf.core.v2.minimum_cost_solver as _base_solver
from spatialcf.core.v2._internal.orchestration import stages as _stages
from spatialcf.core.v2._internal.orchestration.capabilities import (
    DirectSolveCapabilityV2,
    SolveCapabilityKeyV2,
    SolveStageKeyV2,
)
from spatialcf.core.v2._internal.orchestration.registry import (
    resolve_solve_capability_v2,
)
from spatialcf.core.v2.certificate_builder import _ExactCardinalSelectionFrameV2
from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import CoreSolverConfigV2


def solve_registered_capability_v2(
    key: SolveCapabilityKeyV2,
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Run one exact registered complete-solve capability.

    The registry remains closed throughout migration; no compiler-only prefix
    is accepted here.
    """

    return _solve_registered_capability_in_selection_frame_v2(
        key,
        problem,
        config,
        _ExactCardinalSelectionFrameV2(),
    )


def _solve_registered_capability_in_selection_frame_v2(
    key: SolveCapabilityKeyV2,
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Run one registered capability in an exact private selection frame."""

    if type(key) is not SolveCapabilityKeyV2:
        raise TypeError("key must be an exact SolveCapabilityKeyV2")
    if type(selection_frame) is not _ExactCardinalSelectionFrameV2:
        raise TypeError("selection_frame has the wrong exact type")
    capability = resolve_solve_capability_v2(key)
    if type(capability) is DirectSolveCapabilityV2:
        raise RuntimeError(
            "direct solve capabilities must use their typed public entrypoint"
        )
    migrated_stages = {
        SolveStageKeyV2.CARDINAL_YAW,
        SolveStageKeyV2.ZERO_DISTORTION,
        SolveStageKeyV2.CAMERA_TRANSLATION,
        SolveStageKeyV2.CAMERA_CARDINAL_REBASE,
    }
    if any(stage_key not in migrated_stages for stage_key in capability.stage_keys):
        raise NotImplementedError(
            f"registered staged solve has not been migrated: {key.value}"
        )
    return _solve_stage_chain_v2(
        capability.stage_keys,
        0,
        problem,
        config,
        selection_frame,
    )


def _solve_stage_chain_v2(
    stage_keys: tuple[SolveStageKeyV2, ...],
    index: int,
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    if index == len(stage_keys):
        outcome = _base_solver._solve_canonical_minimum_cost_in_selection_frame_v2(
            problem,  # type: ignore[arg-type]
            config,  # type: ignore[arg-type]
            selection_frame,
        )
        if type(outcome) is not CanonicalMinimumCostSolveOutcomeV2:
            raise TypeError("base solve pipeline returned an invalid internal outcome")
        return outcome

    prepared_or_outcome = _stages.prepare_solve_stage_v2(
        stage_keys[index],
        problem,
        config,
        selection_frame,
    )
    if type(prepared_or_outcome) is CanonicalMinimumCostSolveOutcomeV2:
        return prepared_or_outcome
    if type(prepared_or_outcome) is not _stages.PreparedSolveStageV2:
        raise TypeError("solve stage returned an invalid internal preparation")
    prepared = prepared_or_outcome

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = _solve_prepared_child_v2(
                stage_keys,
                index + 1,
                prepared,
            )
    except (ArithmeticError, RuntimeWarning):
        return _stages.child_numeric_outcome_v2(prepared)
    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("child solve pipeline returned an invalid internal outcome")
    if replay.result is None:
        return _stages.child_missing_outcome_v2(prepared, replay)
    return _stages.rebind_solve_stage_v2(prepared, replay)


def _solve_prepared_child_v2(
    stage_keys: tuple[SolveStageKeyV2, ...],
    index: int,
    prepared: _stages.PreparedSolveStageV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Execute one prepared stage's remaining private child chain."""

    if type(prepared) is not _stages.PreparedSolveStageV2:
        raise TypeError("prepared stage has the wrong exact type")
    return _solve_stage_chain_v2(
        stage_keys,
        index,
        prepared.normalized_problem,
        prepared.internal_config,
        prepared.child_selection_frame,
    )


__all__ = ("solve_registered_capability_v2",)
