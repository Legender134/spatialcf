"""Independent pure-core replay verification for Canonical v2 solve results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2.minimum_cost_solver import (
    CanonicalMinimumCostSolveOutcomeV2 as CanonicalMinimumCostSolveOutcomeV2,  # noqa: PLC0414
)
from spatialcf.core.v2.minimum_cost_solver import solve_canonical_minimum_cost_v2
from spatialcf.domain.v2.artifacts import CompilationResourceUsageV2
from spatialcf.domain.v2.base import Sha256Digest
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    SolveStatusV2,
    UncertifiedReasonV2,
    UncertifiedResultV2,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_SHA256_ADAPTER = TypeAdapter(Sha256Digest)


class CanonicalSolveVerificationKindV2(StrEnum):
    """Closed states of independent deterministic replay verification."""

    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class CanonicalSolveVerificationOutcomeV2:
    """Closed verifier outcome.

    Verified references exist only after a fresh solve matches the submitted
    result byte-for-byte. ``replay_generation_usage`` is telemetry from this
    call, not a transferable proof token; a consumer must replay raw inputs.
    """

    kind: CanonicalSolveVerificationKindV2
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    submitted_solve_result_sha256: Sha256Digest | None = None
    verified_status: SolveStatusV2 | None = None
    replay_generation_usage: CompilationResourceUsageV2 | None = None
    proposal_count: int = 0
    evaluated_proposal_count: int = 0
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CanonicalSolveVerificationKindV2:
            raise TypeError("kind must be CanonicalSolveVerificationKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)

        for name in ("proposal_count", "evaluated_proposal_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be a non-negative exact int")
        if self.evaluated_proposal_count > self.proposal_count:
            raise ValueError("evaluated proposal count cannot exceed proposal count")

        usage = self.replay_generation_usage
        if usage is not None:
            if type(usage) is not CompilationResourceUsageV2:
                raise TypeError("replay_generation_usage has the wrong exact type")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    usage = CompilationResourceUsageV2.model_validate(
                        usage.model_dump(mode="python"), strict=True
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "replay_generation_usage must validate strictly"
                ) from error
            object.__setattr__(self, "replay_generation_usage", usage)
        if usage is None and (self.proposal_count or self.evaluated_proposal_count):
            raise ValueError("proposal telemetry requires replay generation usage")

        reference_names = (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "submitted_solve_result_sha256",
        )
        for name in reference_names:
            value = getattr(self, name)
            if value is None:
                continue
            try:
                value = _SHA256_ADAPTER.validate_python(value, strict=True)
            except (ValidationError, TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a Sha256Digest") from error
            object.__setattr__(self, name, value)
        references = tuple(getattr(self, name) for name in reference_names)

        if self.kind is CanonicalSolveVerificationKindV2.VERIFIED:
            if any(value is None for value in references):
                raise ValueError("VERIFIED requires all verified references")
            if type(self.verified_status) is not SolveStatusV2:
                raise TypeError("VERIFIED requires an exact SolveStatusV2")
            if self.uncertified_reason is not None or findings:
                raise ValueError("VERIFIED cannot carry failure diagnostics")
            if self.verified_status is SolveStatusV2.CERTIFIED_SUCCESS and (
                usage is None
                or self.proposal_count < 1
                or self.evaluated_proposal_count != self.proposal_count
            ):
                raise ValueError(
                    "verified success requires complete fresh proposal telemetry"
                )
            if self.verified_status is SolveStatusV2.PROVEN_UNSAT and (
                usage is None
                or self.proposal_count != 0
                or self.evaluated_proposal_count != 0
            ):
                raise ValueError(
                    "verified unsat requires zero-proposal fresh telemetry"
                )
            return

        if (
            any(value is not None for value in references)
            or self.verified_status is not None
        ):
            raise ValueError("failure outcomes cannot carry verified references")
        if not findings:
            raise ValueError(f"{self.kind.value} requires at least one finding")
        if self.kind is CanonicalSolveVerificationKindV2.MISMATCH:
            if self.uncertified_reason is not None:
                raise ValueError("MISMATCH cannot carry an uncertified reason")
            return
        if type(self.uncertified_reason) is not UncertifiedReasonV2:
            raise TypeError("UNCERTIFIED requires an UncertifiedReasonV2")


def verify_canonical_solve_result_v2(
    problem: SemanticProblemV2,
    expected_config: CoreSolverConfigV2,
    submitted_result: CertifiedSuccessResultV2
    | ProvenUnsatResultV2
    | UncertifiedResultV2,
) -> CanonicalSolveVerificationOutcomeV2:
    """Replay with trusted policy and compare the complete canonical result."""

    from spatialcf.core.v2._internal.certification.solve_replay import (
        SolveReplayBindingsV2,
        verify_solve_replay_v2,
    )

    return verify_solve_replay_v2(
        problem,
        expected_config,
        submitted_result,
        bindings=SolveReplayBindingsV2(
            problem_type=SemanticProblemV2,
            solve=solve_canonical_minimum_cost_v2,
            canonical_json_bytes=canonical_json_bytes_v2,
        ),
    )


class CanonicalSolveResultVerifierV2:
    """Stateless object wrapper for independent result replay."""

    def verify(
        self,
        problem: SemanticProblemV2,
        expected_config: CoreSolverConfigV2,
        submitted_result: CertifiedSuccessResultV2
        | ProvenUnsatResultV2
        | UncertifiedResultV2,
    ) -> CanonicalSolveVerificationOutcomeV2:
        return verify_canonical_solve_result_v2(
            problem, expected_config, submitted_result
        )
