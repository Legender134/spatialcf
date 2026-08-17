"""Trusted expected-config replay verifier for competition v2.9 results."""

from __future__ import annotations

import warnings

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_solver_v2_9 import (
    solve_continuous_yaw_minimum_cost_v2_9,
)
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawProvenUnsatResultV2_9,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawSolveResultV2_9,
    ContinuousYawSolveVerificationKindV2,
    ContinuousYawSolveVerificationOutcomeV2_9,
    ContinuousYawUncertifiedResultV2_9,
)
from spatialcf.domain.v2.result import UncertifiedReasonV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_RESULT_ADAPTER = TypeAdapter(ContinuousYawSolveResultV2_9)


def verify_continuous_yaw_solve_result_v2_9(
    problem: SemanticProblemV2_3,
    expected_config: ContinuousYawSolverConfigV2_9,
    submitted_result: ContinuousYawSolveResultV2_9,
) -> ContinuousYawSolveVerificationOutcomeV2_9:
    """Gate trusted input policy, then replay the complete solver exactly once."""

    checked = _strict_inputs(problem, expected_config, submitted_result)
    if checked is None:
        return _mismatch("MISMATCH:CONTINUOUS_YAW_SUBMITTED_INPUT_V2_9")
    problem, config, submitted = checked
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if submitted.semantic_problem_sha256 != problem.semantic_problem_sha256:
                return _mismatch("MISMATCH:CONTINUOUS_YAW_PROBLEM_HASH_V2_9")
            if (
                submitted.solver_config.config_sha256 != config.config_sha256
                or canonical_json_bytes_v2(submitted.solver_config)
                != canonical_json_bytes_v2(config)
            ):
                return _mismatch("MISMATCH:CONTINUOUS_YAW_EXPECTED_CONFIG_V2_9")
    except (ArithmeticError, RuntimeWarning):
        return _uncertified("NUMERIC_GAP:CONTINUOUS_YAW_VERIFICATION_GATE_V2_9")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = solve_continuous_yaw_minimum_cost_v2_9(problem, config)
    except (ArithmeticError, RuntimeWarning):
        return _uncertified("NUMERIC_GAP:CONTINUOUS_YAW_SOLVE_REPLAY_V2_9")
    if replay.result is None:
        return ContinuousYawSolveVerificationOutcomeV2_9(
            kind=ContinuousYawSolveVerificationKindV2.UNCERTIFIED,
            uncertified_reason=UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            finding_codes=("COMPILATION_INCOMPLETE:CONTINUOUS_YAW_SOLVE_REPLAY_V2_9",),
        )
    fresh = replay.result
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        exact = (
            type(fresh) is type(submitted)
            and fresh == submitted
            and canonical_json_bytes_v2(fresh) == canonical_json_bytes_v2(submitted)
            and fresh.solve_result_sha256 == submitted.solve_result_sha256
        )
    if not exact:
        return ContinuousYawSolveVerificationOutcomeV2_9(
            kind=ContinuousYawSolveVerificationKindV2.MISMATCH,
            replay_generation_usage=replay.cumulative_generation_usage,
            proposal_count=replay.proposal_count,
            evaluated_proposal_count=replay.evaluated_proposal_count,
            finding_codes=("MISMATCH:CONTINUOUS_YAW_SOLVE_RESULT_V2_9",),
        )
    return ContinuousYawSolveVerificationOutcomeV2_9(
        kind=ContinuousYawSolveVerificationKindV2.VERIFIED,
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solver_config_sha256=config.config_sha256,
        submitted_solve_result_sha256=submitted.solve_result_sha256,
        verified_status=submitted.status,
        replay_generation_usage=replay.cumulative_generation_usage,
        proposal_count=replay.proposal_count,
        evaluated_proposal_count=replay.evaluated_proposal_count,
    )


class ContinuousYawSolveVerifierV2_9:
    @staticmethod
    def verify(problem, expected_config, submitted_result):
        return verify_continuous_yaw_solve_result_v2_9(
            problem, expected_config, submitted_result
        )


def _strict_inputs(problem, config, submitted):
    if type(problem) is not SemanticProblemV2_3 or type(config) is not (
        ContinuousYawSolverConfigV2_9
    ):
        return None
    if type(submitted) not in (
        ContinuousYawCertifiedSuccessResultV2_9,
        ContinuousYawProvenUnsatResultV2_9,
        ContinuousYawUncertifiedResultV2_9,
    ):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return (
                SemanticProblemV2_3.model_validate(
                    problem.model_dump(mode="python", warnings="error"), strict=True
                ),
                ContinuousYawSolverConfigV2_9.model_validate(
                    config.model_dump(mode="python", warnings="error"), strict=True
                ),
                _RESULT_ADAPTER.validate_python(
                    submitted.model_dump(mode="python", warnings="error"), strict=True
                ),
            )
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
    return ContinuousYawSolveVerificationOutcomeV2_9(
        kind=ContinuousYawSolveVerificationKindV2.MISMATCH,
        finding_codes=(finding,),
    )


def _uncertified(finding):
    return ContinuousYawSolveVerificationOutcomeV2_9(
        kind=ContinuousYawSolveVerificationKindV2.UNCERTIFIED,
        uncertified_reason=UncertifiedReasonV2.NUMERIC_GAP,
        finding_codes=(finding,),
    )


__all__ = (
    "ContinuousYawSolveVerifierV2_9",
    "verify_continuous_yaw_solve_result_v2_9",
)
