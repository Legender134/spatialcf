"""Trusted expected-config replay verifier for continuous-yaw v2.8 results."""

from __future__ import annotations

import warnings

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_solver import (
    solve_continuous_yaw_minimum_cost_v2_8,
)
from spatialcf.domain.v2.continuous_yaw_candidate import SemanticProblemV2_2
from spatialcf.domain.v2.continuous_yaw_solver import (
    ContinuousYawCertifiedSuccessResultV2_8,
    ContinuousYawProvenUnsatResultV2_8,
    ContinuousYawSolverConfigV2_8,
    ContinuousYawSolveResultV2_8,
    ContinuousYawSolveVerificationKindV2,
    ContinuousYawSolveVerificationOutcomeV2_8,
    ContinuousYawUncertifiedResultV2_8,
)
from spatialcf.domain.v2.result import UncertifiedReasonV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_RESULT_ADAPTER = TypeAdapter(ContinuousYawSolveResultV2_8)


def verify_continuous_yaw_solve_result_v2_8(
    problem: SemanticProblemV2_2,
    expected_config: ContinuousYawSolverConfigV2_8,
    submitted_result: ContinuousYawSolveResultV2_8,
) -> ContinuousYawSolveVerificationOutcomeV2_8:
    """Replay exactly once after gating the submitter's embedded config."""

    checked = _strict_inputs(problem, expected_config, submitted_result)
    if checked is None:
        return _mismatch("MISMATCH:CONTINUOUS_YAW_SUBMITTED_INPUT")
    checked_problem, checked_config, checked_submitted = checked
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if (
                checked_submitted.semantic_problem_sha256
                != checked_problem.semantic_problem_sha256
            ):
                return _mismatch("MISMATCH:CONTINUOUS_YAW_PROBLEM_HASH")
            if (
                checked_submitted.solver_config.config_sha256
                != checked_config.config_sha256
                or canonical_json_bytes_v2(checked_submitted.solver_config)
                != canonical_json_bytes_v2(checked_config)
            ):
                return _mismatch("MISMATCH:CONTINUOUS_YAW_EXPECTED_CONFIG")
    except (ArithmeticError, RuntimeWarning):
        return ContinuousYawSolveVerificationOutcomeV2_8(
            kind=ContinuousYawSolveVerificationKindV2.UNCERTIFIED,
            uncertified_reason=UncertifiedReasonV2.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:CONTINUOUS_YAW_VERIFICATION_GATE",),
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = solve_continuous_yaw_minimum_cost_v2_8(
                checked_problem, checked_config
            )
    except (ArithmeticError, RuntimeWarning):
        return ContinuousYawSolveVerificationOutcomeV2_8(
            kind=ContinuousYawSolveVerificationKindV2.UNCERTIFIED,
            uncertified_reason=UncertifiedReasonV2.NUMERIC_GAP,
            finding_codes=("NUMERIC_GAP:CONTINUOUS_YAW_SOLVE_REPLAY",),
        )
    if replay.result is None:
        return ContinuousYawSolveVerificationOutcomeV2_8(
            kind=ContinuousYawSolveVerificationKindV2.UNCERTIFIED,
            uncertified_reason=UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            finding_codes=("COMPILATION_INCOMPLETE:CONTINUOUS_YAW_SOLVE_REPLAY",),
        )
    fresh = replay.result
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        exact_match = (
            type(fresh) is type(checked_submitted)
            and fresh == checked_submitted
            and canonical_json_bytes_v2(fresh)
            == canonical_json_bytes_v2(checked_submitted)
            and fresh.solve_result_sha256 == checked_submitted.solve_result_sha256
        )
    if not exact_match:
        return ContinuousYawSolveVerificationOutcomeV2_8(
            kind=ContinuousYawSolveVerificationKindV2.MISMATCH,
            replay_generation_usage=replay.cumulative_generation_usage,
            proposal_count=replay.proposal_count,
            evaluated_proposal_count=replay.evaluated_proposal_count,
            finding_codes=("MISMATCH:CONTINUOUS_YAW_SOLVE_RESULT",),
        )
    return ContinuousYawSolveVerificationOutcomeV2_8(
        kind=ContinuousYawSolveVerificationKindV2.VERIFIED,
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        solver_config_sha256=checked_config.config_sha256,
        submitted_solve_result_sha256=checked_submitted.solve_result_sha256,
        verified_status=checked_submitted.status,
        replay_generation_usage=replay.cumulative_generation_usage,
        proposal_count=replay.proposal_count,
        evaluated_proposal_count=replay.evaluated_proposal_count,
    )


def _strict_inputs(problem, config, submitted):
    if type(problem) is not SemanticProblemV2_2 or type(config) is not (
        ContinuousYawSolverConfigV2_8
    ):
        return None
    if type(submitted) not in (
        ContinuousYawCertifiedSuccessResultV2_8,
        ContinuousYawProvenUnsatResultV2_8,
        ContinuousYawUncertifiedResultV2_8,
    ):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_problem = SemanticProblemV2_2.model_validate(
                problem.model_dump(mode="python", warnings="error"), strict=True
            )
            checked_config = ContinuousYawSolverConfigV2_8.model_validate(
                config.model_dump(mode="python", warnings="error"), strict=True
            )
            checked_submitted = _RESULT_ADAPTER.validate_python(
                submitted.model_dump(mode="python", warnings="error"), strict=True
            )
        return checked_problem, checked_config, checked_submitted
    except (
        AttributeError,
        ValidationError,
        PydanticSerializationError,
        TypeError,
        ValueError,
        Warning,
    ):
        return None


def _mismatch(finding):
    return ContinuousYawSolveVerificationOutcomeV2_8(
        kind=ContinuousYawSolveVerificationKindV2.MISMATCH,
        finding_codes=(finding,),
    )


__all__ = ("verify_continuous_yaw_solve_result_v2_8",)
