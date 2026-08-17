"""Common private replay engine for Canonical v2 solve verifiers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

from spatialcf.core.v2._internal.boundary import (
    InvalidCallerInputV2,
    NumericBoundaryGapV2,
    strict_fresh_solve_result_v2,
    strict_input_model_v2,
    strict_submitted_solve_result_v2,
)
from spatialcf.core.v2.minimum_cost_solver import (
    CanonicalMinimumCostSolveOutcomeV2,
)
from spatialcf.core.v2.solve_verifier import (
    CanonicalSolveVerificationKindV2,
    CanonicalSolveVerificationOutcomeV2,
)
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    UncertifiedReasonV2,
)


@dataclass(frozen=True, slots=True)
class SolveReplayBindingsV2:
    """Fixed private dependencies selected by one compatibility wrapper."""

    problem_type: type[V2Model]
    solve: Callable[[V2Model, CoreSolverConfigV2], CanonicalMinimumCostSolveOutcomeV2]
    canonical_json_bytes: Callable[[object], bytes]
    exact_schema_identity_type: type[V2Model] | None = None
    exact_schema_version: str | None = None
    schema_mismatch_finding: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem_type, type) or not issubclass(
            self.problem_type, V2Model
        ):
            raise TypeError("problem_type must be a V2Model type")
        if not callable(self.solve):
            raise TypeError("solve must be callable")
        if not callable(self.canonical_json_bytes):
            raise TypeError("canonical_json_bytes must be callable")

        schema_values = (
            self.exact_schema_identity_type,
            self.exact_schema_version,
            self.schema_mismatch_finding,
        )
        if all(value is None for value in schema_values):
            return
        if any(value is None for value in schema_values):
            raise ValueError("exact schema precheck fields must be supplied together")
        if not isinstance(self.exact_schema_identity_type, type) or not issubclass(
            self.exact_schema_identity_type, V2Model
        ):
            raise TypeError("exact_schema_identity_type must be a V2Model type")
        if (
            type(self.exact_schema_version) is not str
            or not self.exact_schema_version.strip()
        ):
            raise TypeError("exact_schema_version must be an exact non-blank string")
        if (
            type(self.schema_mismatch_finding) is not str
            or not self.schema_mismatch_finding.strip()
        ):
            raise TypeError("schema_mismatch_finding must be an exact non-blank string")


def verify_solve_replay_v2(
    problem: object,
    expected_config: object,
    submitted_result: object,
    *,
    bindings: SolveReplayBindingsV2,
) -> CanonicalSolveVerificationOutcomeV2:
    """Replay one fixed solver and compare the complete canonical result."""

    if type(bindings) is not SolveReplayBindingsV2:
        raise TypeError("bindings must be an exact SolveReplayBindingsV2")
    schema_mismatch = bindings.schema_mismatch_finding
    if schema_mismatch is not None and (
        type(problem) is not bindings.problem_type
        or type(getattr(problem, "schema_identity", None))
        is not bindings.exact_schema_identity_type
        or getattr(problem.schema_identity, "schema_version", None)  # type: ignore[union-attr]
        != bindings.exact_schema_version
    ):
        return _uncertified(UncertifiedReasonV2.UNSUPPORTED_MODEL, schema_mismatch)

    try:
        checked_problem = strict_input_model_v2(
            problem,
            bindings.problem_type,
            "SEMANTIC_PROBLEM",
        )
        checked_config = strict_input_model_v2(
            expected_config,
            CoreSolverConfigV2,
            "EXPECTED_CORE_SOLVER_CONFIG",
        )
    except NumericBoundaryGapV2 as error:
        return _uncertified(UncertifiedReasonV2.NUMERIC_GAP, error.finding_code)
    except InvalidCallerInputV2 as error:
        return _uncertified(UncertifiedReasonV2.UNSUPPORTED_MODEL, error.finding_code)

    try:
        checked_submitted = strict_submitted_solve_result_v2(submitted_result)
    except (InvalidCallerInputV2, NumericBoundaryGapV2) as error:
        return _mismatch(error.finding_code)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            problem_sha256 = checked_problem.semantic_problem_sha256
            submitted_problem_sha256 = checked_submitted.semantic_problem_sha256
            submitted_config = checked_submitted.core_solver_config
            submitted_config_sha256 = submitted_config.core_solver_config_sha256
            expected_config_sha256 = checked_config.core_solver_config_sha256
            submitted_config_bytes = bindings.canonical_json_bytes(submitted_config)
            expected_config_bytes = bindings.canonical_json_bytes(checked_config)
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONFIG_CANONICALIZATION",
        )
    if submitted_problem_sha256 != problem_sha256:
        return _mismatch("SUBMITTED_REFERENCE_MISMATCH:SEMANTIC_PROBLEM")
    if (
        submitted_config_sha256 != expected_config_sha256
        or submitted_config_bytes != expected_config_bytes
    ):
        return _mismatch("SUBMITTED_REFERENCE_MISMATCH:EXPECTED_CONFIG")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = bindings.solve(checked_problem, checked_config)
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:MINIMUM_SOLVE_REPLAY",
        )
    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("minimum solver returned an invalid internal outcome")
    if replay.result is None:
        reason = (
            UncertifiedReasonV2.NUMERIC_GAP
            if any(code.startswith("NUMERIC_GAP:") for code in replay.finding_codes)
            else UncertifiedReasonV2.COMPILATION_INCOMPLETE
        )
        return _uncertified(
            reason,
            *(replay.finding_codes or ("SOLVE_REPLAY_HAS_NO_RESULT",)),
            replay=replay,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            fresh_result = strict_fresh_solve_result_v2(replay.result)
            submitted_bytes = bindings.canonical_json_bytes(checked_submitted)
            fresh_bytes = bindings.canonical_json_bytes(fresh_result)
            fresh_sha256 = fresh_result.solve_result_sha256
            submitted_sha256 = checked_submitted.solve_result_sha256
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:SOLVE_RESULT_REVALIDATION",
            replay=replay,
        )

    findings: list[str] = []
    if type(fresh_result) is not type(checked_submitted):
        findings.append("SOLVE_RESULT_MISMATCH:TYPE")
    if fresh_result.status is not checked_submitted.status:
        findings.append("SOLVE_RESULT_MISMATCH:STATUS")
    if fresh_result != checked_submitted:
        findings.append("SOLVE_RESULT_MISMATCH:MODEL")
    if fresh_bytes != submitted_bytes:
        findings.append("SOLVE_RESULT_MISMATCH:CANONICAL_BYTES")
    if fresh_sha256 != submitted_sha256:
        findings.append("SOLVE_RESULT_MISMATCH:SHA256")
    if findings:
        return _mismatch(*findings, replay=replay)

    return CanonicalSolveVerificationOutcomeV2(
        kind=CanonicalSolveVerificationKindV2.VERIFIED,
        semantic_problem_sha256=problem_sha256,
        core_solver_config_sha256=expected_config_sha256,
        submitted_solve_result_sha256=submitted_sha256,
        verified_status=checked_submitted.status,
        replay_generation_usage=replay.cumulative_generation_usage,
        proposal_count=replay.proposal_count,
        evaluated_proposal_count=replay.evaluated_proposal_count,
    )


def _telemetry(replay: CanonicalMinimumCostSolveOutcomeV2 | None) -> dict[str, object]:
    if replay is None:
        return {}
    return {
        "replay_generation_usage": replay.cumulative_generation_usage,
        "proposal_count": replay.proposal_count,
        "evaluated_proposal_count": replay.evaluated_proposal_count,
    }


def _mismatch(
    *finding_codes: str,
    replay: CanonicalMinimumCostSolveOutcomeV2 | None = None,
) -> CanonicalSolveVerificationOutcomeV2:
    return CanonicalSolveVerificationOutcomeV2(
        kind=CanonicalSolveVerificationKindV2.MISMATCH,
        finding_codes=tuple(finding_codes),
        **_telemetry(replay),
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    replay: CanonicalMinimumCostSolveOutcomeV2 | None = None,
) -> CanonicalSolveVerificationOutcomeV2:
    return CanonicalSolveVerificationOutcomeV2(
        kind=CanonicalSolveVerificationKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=tuple(finding_codes),
        **_telemetry(replay),
    )
