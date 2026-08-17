"""Deterministic replay verification for candidate and relation artifacts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2.candidate_domain import CandidateDomainCompilerV2
from spatialcf.core.v2.relation_cost_partition import (
    RelationCostPartitionCompilationKindV2,
    RelationCostPartitionCompilationOutcomeV2,
    compile_relation_cost_partition_v2,
)
from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    CompilationResourceUsageV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import CoreSolverConfigV2, UncertifiedReasonV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_SHA256_DIGEST_ADAPTER = TypeAdapter(Sha256Digest)


class CoreArtifactVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class CoreArtifactVerificationOutcomeV2:
    """Closed replay result; verified references exist only after exact replay.

    ``verification_resource_usage`` records the deterministic public compiler
    replay under the original solver configuration.  It is not added to the
    submitted compilation ledger and is not an unforgeable capability.
    """

    kind: CoreArtifactVerificationKindV2
    semantic_problem_sha256: Sha256Digest | None = None
    core_solver_config_sha256: Sha256Digest | None = None
    candidate_domain_artifact_sha256: Sha256Digest | None = None
    relation_cost_partition_sha256: Sha256Digest | None = None
    verification_resource_usage: CompilationResourceUsageV2 | None = None
    uncertified_reason: UncertifiedReasonV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CoreArtifactVerificationKindV2):
            raise TypeError("kind must be a CoreArtifactVerificationKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be an exact tuple of non-blank strings")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        usage = self.verification_resource_usage
        if usage is not None:
            if not isinstance(usage, CompilationResourceUsageV2):
                raise TypeError(
                    "verification_resource_usage must be a CompilationResourceUsageV2"
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Warning)
                    usage = CompilationResourceUsageV2.model_validate(
                        usage.model_dump(mode="python"),
                        strict=True,
                    )
            except (ValidationError, TypeError, ValueError, Warning) as error:
                raise TypeError(
                    "verification_resource_usage must pass strict validation"
                ) from error
            object.__setattr__(self, "verification_resource_usage", usage)

        for field_name in (
            "semantic_problem_sha256",
            "core_solver_config_sha256",
            "candidate_domain_artifact_sha256",
            "relation_cost_partition_sha256",
        ):
            reference = getattr(self, field_name)
            if reference is None:
                continue
            try:
                reference = _SHA256_DIGEST_ADAPTER.validate_python(
                    reference,
                    strict=True,
                )
            except (ValidationError, TypeError, ValueError) as error:
                raise ValueError(f"{field_name} must be a Sha256Digest") from error
            object.__setattr__(self, field_name, reference)

        references = (
            self.semantic_problem_sha256,
            self.core_solver_config_sha256,
            self.candidate_domain_artifact_sha256,
            self.relation_cost_partition_sha256,
        )
        if self.kind is CoreArtifactVerificationKindV2.VERIFIED:
            if any(value is None for value in references):
                raise ValueError("VERIFIED requires all four verified references")
            if self.verification_resource_usage is None:
                raise ValueError("VERIFIED requires verification resource usage")
            if self.uncertified_reason is not None or self.finding_codes:
                raise ValueError("VERIFIED cannot carry failure diagnostics")
            return

        if any(value is not None for value in references):
            raise ValueError("non-VERIFIED outcomes cannot carry verified references")
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} requires at least one finding")
        if self.kind is CoreArtifactVerificationKindV2.MISMATCH:
            if self.uncertified_reason is not None:
                raise ValueError("MISMATCH cannot carry an uncertified reason")
            return
        if not isinstance(self.uncertified_reason, UncertifiedReasonV2):
            raise TypeError("UNCERTIFIED requires an uncertified reason")


CoreArtifactVerifyOutcomeV2 = CoreArtifactVerificationOutcomeV2


class _InvalidInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _NumericInputV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


ModelT = TypeVar("ModelT", bound=V2Model)


def verify_core_artifacts_v2(
    problem: SemanticProblemV2,
    config: CoreSolverConfigV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionCompilationOutcomeV2,
) -> CoreArtifactVerificationOutcomeV2:
    """Replay both public compilers and compare their complete frozen outputs."""

    try:
        checked_problem = _strict_model(
            problem,
            SemanticProblemV2,
            label="SEMANTIC_PROBLEM",
        )
        checked_config = _strict_model(
            config,
            CoreSolverConfigV2,
            label="CORE_SOLVER_CONFIG",
        )
        checked_candidate = _strict_model(
            candidate,
            CandidateDomainArtifactV2,
            label="CANDIDATE_DOMAIN",
        )
        checked_relation = _strict_relation_outcome(relation)
    except _NumericInputV2 as error:
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            error.finding_code,
        )
    except _InvalidInputV2 as error:
        return _uncertified(
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate_replay = CandidateDomainCompilerV2().compile(
                checked_problem,
                checked_config,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CANDIDATE_DOMAIN_REPLAY",
        )

    expected_candidate = candidate_replay.candidate_domain
    if expected_candidate is None:
        return _uncertified(
            candidate_replay.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(candidate_replay.finding_codes or ("REPLAY_NO_CANDIDATE_DOMAIN",)),
        )

    candidate_mismatches = _artifact_mismatches(
        checked_candidate,
        expected_candidate,
        label="CANDIDATE_DOMAIN",
        actual_hash=checked_candidate.candidate_domain_artifact_sha256,
        expected_hash=expected_candidate.candidate_domain_artifact_sha256,
    )
    if candidate_mismatches:
        return _mismatch(
            *candidate_mismatches,
            verification_resource_usage=expected_candidate.resource_usage,
        )
    if candidate_replay.uncertified_reason is not None:
        return _uncertified(
            candidate_replay.uncertified_reason,
            *(candidate_replay.finding_codes or ("REPLAY_CANDIDATE_UNCERTIFIED",)),
            verification_resource_usage=expected_candidate.resource_usage,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            relation_replay = compile_relation_cost_partition_v2(
                checked_problem,
                checked_config,
                expected_candidate,
            )
    except (ArithmeticError, RuntimeWarning):
        return _uncertified(
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:RELATION_COST_PARTITION_REPLAY",
            verification_resource_usage=expected_candidate.resource_usage,
        )

    verification_usage = (
        relation_replay.cumulative_resource_usage or expected_candidate.resource_usage
    )
    if relation_replay.kind is not RelationCostPartitionCompilationKindV2.PARTITION:
        return _uncertified(
            relation_replay.uncertified_reason
            or UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            *(relation_replay.finding_codes or ("REPLAY_NO_RELATION_COST_PARTITION",)),
            verification_resource_usage=verification_usage,
        )

    expected_partition = relation_replay.relation_cost_partition
    if expected_partition is None:  # pragma: no cover - outcome invariant
        return _uncertified(
            UncertifiedReasonV2.COMPILATION_INCOMPLETE,
            "REPLAY_NO_RELATION_COST_PARTITION",
            verification_resource_usage=verification_usage,
        )
    if checked_relation.kind is not RelationCostPartitionCompilationKindV2.PARTITION:
        return _mismatch(
            "ARTIFACT_MISMATCH:RELATION_COST_PARTITION:OUTCOME_KIND",
            verification_resource_usage=verification_usage,
        )

    actual_partition = checked_relation.relation_cost_partition
    if actual_partition is None:  # pragma: no cover - outcome invariant
        return _mismatch(
            "ARTIFACT_MISMATCH:RELATION_COST_PARTITION:MISSING",
            verification_resource_usage=verification_usage,
        )
    relation_mismatches = list(
        _artifact_mismatches(
            actual_partition,
            expected_partition,
            label="RELATION_COST_PARTITION",
            actual_hash=actual_partition.relation_cost_partition_sha256,
            expected_hash=expected_partition.relation_cost_partition_sha256,
        )
    )
    if (
        checked_relation.cumulative_resource_usage
        != relation_replay.cumulative_resource_usage
    ):
        relation_mismatches.append(
            "ARTIFACT_MISMATCH:RELATION_COST_PARTITION:CUMULATIVE_RESOURCE_USAGE"
        )
    if relation_mismatches:
        return _mismatch(
            *relation_mismatches,
            verification_resource_usage=verification_usage,
        )

    return CoreArtifactVerificationOutcomeV2(
        kind=CoreArtifactVerificationKindV2.VERIFIED,
        semantic_problem_sha256=checked_problem.semantic_problem_sha256,
        core_solver_config_sha256=checked_config.core_solver_config_sha256,
        candidate_domain_artifact_sha256=(
            expected_candidate.candidate_domain_artifact_sha256
        ),
        relation_cost_partition_sha256=(
            expected_partition.relation_cost_partition_sha256
        ),
        verification_resource_usage=verification_usage,
    )


class CoreArtifactVerifierV2:
    """Stateless wrapper for pipeline composition."""

    def verify(
        self,
        problem: SemanticProblemV2,
        config: CoreSolverConfigV2,
        candidate: CandidateDomainArtifactV2,
        relation: RelationCostPartitionCompilationOutcomeV2,
    ) -> CoreArtifactVerificationOutcomeV2:
        return verify_core_artifacts_v2(problem, config, candidate, relation)


def _strict_model(
    value: object,
    model_type: type[ModelT],
    *,
    label: str,
) -> ModelT:
    if not isinstance(value, model_type):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}:TYPE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise _NumericInputV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}") from error


def _strict_relation_outcome(
    value: object,
) -> RelationCostPartitionCompilationOutcomeV2:
    label = "RELATION_COST_PARTITION_OUTCOME"
    if not isinstance(value, RelationCostPartitionCompilationOutcomeV2):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}:TYPE")
    partition = value.relation_cost_partition
    if partition is not None and not isinstance(partition, RelationCostPartitionV2):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    usage = value.cumulative_resource_usage
    if usage is not None and not isinstance(usage, CompilationResourceUsageV2):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    if type(value.finding_codes) is not tuple or any(
        type(code) is not str or not code.strip() for code in value.finding_codes
    ):
        raise _InvalidInputV2(f"INVALID_INPUT:{label}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if partition is not None:
                partition = RelationCostPartitionV2.model_validate(
                    partition.model_dump(mode="python"),
                    strict=True,
                )
            if usage is not None:
                usage = CompilationResourceUsageV2.model_validate(
                    usage.model_dump(mode="python"),
                    strict=True,
                )
            return RelationCostPartitionCompilationOutcomeV2(
                kind=value.kind,
                relation_cost_partition=partition,
                uncertified_reason=value.uncertified_reason,
                finding_codes=tuple(value.finding_codes),
                cumulative_resource_usage=usage,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise _NumericInputV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise _InvalidInputV2(f"INVALID_INPUT:{label}") from error


def _artifact_mismatches(
    actual: V2Model,
    expected: V2Model,
    *,
    label: str,
    actual_hash: str,
    expected_hash: str,
) -> tuple[str, ...]:
    findings: list[str] = []
    if actual != expected:
        findings.append(f"ARTIFACT_MISMATCH:{label}:MODEL")
    if canonical_json_bytes_v2(actual) != canonical_json_bytes_v2(expected):
        findings.append(f"ARTIFACT_MISMATCH:{label}:CANONICAL_BYTES")
    if actual_hash != expected_hash:
        findings.append(f"ARTIFACT_MISMATCH:{label}:HASH")
    return tuple(findings)


def _mismatch(
    *finding_codes: str,
    verification_resource_usage: CompilationResourceUsageV2 | None,
) -> CoreArtifactVerificationOutcomeV2:
    return CoreArtifactVerificationOutcomeV2(
        kind=CoreArtifactVerificationKindV2.MISMATCH,
        finding_codes=finding_codes,
        verification_resource_usage=verification_resource_usage,
    )


def _uncertified(
    reason: UncertifiedReasonV2,
    *finding_codes: str,
    verification_resource_usage: CompilationResourceUsageV2 | None = None,
) -> CoreArtifactVerificationOutcomeV2:
    return CoreArtifactVerificationOutcomeV2(
        kind=CoreArtifactVerificationKindV2.UNCERTIFIED,
        uncertified_reason=reason,
        finding_codes=finding_codes,
        verification_resource_usage=verification_resource_usage,
    )
