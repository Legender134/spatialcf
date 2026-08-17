"""Independent replay verification for Exact Cardinal Canonical 2.1 results."""

from __future__ import annotations

from spatialcf.core.v2._internal.certification.solve_replay import (
    SolveReplayBindingsV2,
    verify_solve_replay_v2,
)
from spatialcf.core.v2.minimum_cost_solver_v2_1 import (
    solve_canonical_minimum_cost_v2_1,
)
from spatialcf.core.v2.solve_verifier import CanonicalSolveVerificationOutcomeV2
from spatialcf.domain.v2.cardinal import SchemaIdentityV2_1, SemanticProblemV2_1
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    UncertifiedResultV2,
)
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


def verify_canonical_solve_result_v2_1(
    problem: SemanticProblemV2_1,
    expected_config: CoreSolverConfigV2,
    submitted_result: CertifiedSuccessResultV2
    | ProvenUnsatResultV2
    | UncertifiedResultV2,
) -> CanonicalSolveVerificationOutcomeV2:
    """Replay with trusted 2.1 inputs and compare complete canonical bytes."""

    return verify_solve_replay_v2(
        problem,
        expected_config,
        submitted_result,
        bindings=SolveReplayBindingsV2(
            problem_type=SemanticProblemV2_1,
            solve=solve_canonical_minimum_cost_v2_1,
            canonical_json_bytes=canonical_json_bytes_v2,
            exact_schema_identity_type=SchemaIdentityV2_1,
            exact_schema_version="2.1",
            schema_mismatch_finding=("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION"),
        ),
    )


class CanonicalSolveResultVerifierV2_1:
    """Stateless object wrapper for independent Cardinal 2.1 replay."""

    def verify(
        self,
        problem: SemanticProblemV2_1,
        expected_config: CoreSolverConfigV2,
        submitted_result: CertifiedSuccessResultV2
        | ProvenUnsatResultV2
        | UncertifiedResultV2,
    ) -> CanonicalSolveVerificationOutcomeV2:
        return verify_canonical_solve_result_v2_1(
            problem,
            expected_config,
            submitted_result,
        )


__all__ = (
    "CanonicalSolveResultVerifierV2_1",
    "verify_canonical_solve_result_v2_1",
)
