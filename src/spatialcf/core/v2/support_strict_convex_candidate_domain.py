"""Raw support-aware strict-convex candidate compilation for Canonical v2.2."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_support_projection import (
    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2,
    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2,
    ContinuousYawSupportProjectionBracketV2,
    ContinuousYawSupportProjectionKindV2,
    compile_exact_horizontal_support_projection_v2,
)
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateCompilationKindV2,
    MultiObstacleStrictConvexCandidateDomainArtifactV2_2,
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _copy_universe,
    _precharge_problem_structure,
    _require_finding_codes,
    _require_id_tuple,
    _strict_problem,
    compile_multi_obstacle_strict_convex_candidate_domain_v2_6,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core.v2.so2_interval import SO2AtomicBudgetV2
from spatialcf.core.v2.strict_convex_candidate_domain import (
    _InvalidInputV2 as _LegacyInvalidInputV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionKindV2,
    StrictConvexIntersectionOutcomeV2,
    intersect_strict_convex_allowed_complexes_v2,
)
from spatialcf.domain.v2.continuous_yaw_candidate import (
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_6,
    StrictConvexCandidateCompilerConfigV2_7,
)

_ARTIFACT_HASH_DOMAIN_V2_2 = (
    b"spatialcf.support-strict-convex-candidate-artifact.v2.2\0"
)
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTERSECTION_KERNEL_ID = "geometry-kernel:rational-strict-convex-intersection-v2"
_INTERSECTION_KERNEL_VERSION = "kernel:2.5-strict-convex-intersection"


class SupportStrictConvexCandidateCompilationKindV2(StrEnum):
    ARTIFACT = "ARTIFACT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class SupportStrictConvexCandidateVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class SupportStrictConvexAllowedBracketV2:
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    intersection_kernel_id: str
    intersection_kernel_version: str
    support_projection_kernel_id: str
    support_projection_kernel_version: str
    so2_atomic_steps_used: int

    def __post_init__(self) -> None:
        checked_inner = _copy_intersection_complex(self.inner_allowed)
        checked_outer = _copy_intersection_complex(self.outer_allowed)
        if checked_inner.universe != checked_outer.universe:
            raise ValueError("support-aware bracket requires one exact universe")
        if self.intersection_kernel_id != _INTERSECTION_KERNEL_ID:
            raise ValueError("unexpected intersection kernel ID")
        if self.intersection_kernel_version != _INTERSECTION_KERNEL_VERSION:
            raise ValueError("unexpected intersection kernel version")
        if self.support_projection_kernel_id != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2
        ):
            raise ValueError("unexpected support projection kernel ID")
        if self.support_projection_kernel_version != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2
        ):
            raise ValueError("unexpected support projection kernel version")
        if (
            type(self.so2_atomic_steps_used) is not int
            or self.so2_atomic_steps_used <= 0
        ):
            raise ValueError("so2_atomic_steps_used must be a positive exact int")
        if not all(
            checked_outer.contains_point(cell.strict_witness)
            for cell in checked_inner.cells
        ):
            raise ValueError("support-aware inner witness escaped the outer domain")
        object.__setattr__(self, "inner_allowed", checked_inner)
        object.__setattr__(self, "outer_allowed", checked_outer)


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateDomainArtifactV2_2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    upstream_t14_artifact_sha256: str
    subject_id: str
    search_universe: ExactAxisAlignedRectV2
    ordered_constraint_ids: tuple[str, ...]
    ordered_obstacle_body_ids: tuple[str, ...]
    support_constraint_id: str
    surface_id: str
    contact_geometry_id: str
    allowed_domain_bracket: SupportStrictConvexAllowedBracketV2
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t14_artifact_sha256", self.upstream_t14_artifact_sha256),
        ):
            if type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label, value in (
            ("subject_id", self.subject_id),
            ("support_constraint_id", self.support_constraint_id),
            ("surface_id", self.surface_id),
            ("contact_geometry_id", self.contact_geometry_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        checked_universe = _copy_universe(self.search_universe)
        if (
            checked_universe.topology is not RectTopologyV2.AREA
            or checked_universe.coordinate_space
            is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        ):
            raise ValueError("search universe must be an AREA translation-delta rect")
        compiled_ids = _require_id_tuple(
            self.ordered_constraint_ids,
            label="ordered_constraint_ids",
            nonempty=True,
            sorted_required=False,
        )
        obstacle_ids = _require_id_tuple(
            self.ordered_obstacle_body_ids,
            label="ordered_obstacle_body_ids",
            nonempty=True,
            sorted_required=True,
        )
        remaining_ids = _require_id_tuple(
            self.remaining_constraint_ids,
            label="remaining_constraint_ids",
            nonempty=False,
            sorted_required=True,
        )
        if self.support_constraint_id not in compiled_ids:
            raise ValueError("support constraint must be included in compiled IDs")
        if set(compiled_ids) & set(remaining_ids):
            raise ValueError("compiled and remaining constraint IDs must be disjoint")
        if type(self.allowed_domain_bracket) is not SupportStrictConvexAllowedBracketV2:
            raise TypeError("allowed_domain_bracket has the wrong exact type")
        checked_bracket = _copy_bracket(self.allowed_domain_bracket)
        if (
            checked_bracket.inner_allowed.universe != checked_universe
            or checked_bracket.outer_allowed.universe != checked_universe
        ):
            raise ValueError("allowed bracket must use the search universe")
        if (
            type(self.resource_usage)
            is not MultiObstacleStrictConvexCandidateResourceUsageV2
        ):
            raise TypeError("resource_usage has the wrong exact type")
        usage = MultiObstacleStrictConvexCandidateResourceUsageV2(
            domain_operations=self.resource_usage.domain_operations,
            so2_atomic_steps=self.resource_usage.so2_atomic_steps,
            candidate_cells=self.resource_usage.candidate_cells,
        )
        if usage.so2_atomic_steps != checked_bracket.so2_atomic_steps_used:
            raise ValueError("SO(2) usage must equal bracket cumulative usage")
        published_cells = len(checked_bracket.inner_allowed.cells) + len(
            checked_bracket.outer_allowed.cells
        )
        if usage.candidate_cells < published_cells:
            raise ValueError("cumulative candidate usage cannot undercount final cells")
        object.__setattr__(self, "search_universe", checked_universe)
        object.__setattr__(self, "ordered_constraint_ids", compiled_ids)
        object.__setattr__(self, "ordered_obstacle_body_ids", obstacle_ids)
        object.__setattr__(self, "remaining_constraint_ids", remaining_ids)
        object.__setattr__(self, "allowed_domain_bracket", checked_bracket)
        object.__setattr__(self, "resource_usage", usage)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            _ARTIFACT_HASH_DOMAIN_V2_2 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateCompilationOutcomeV2:
    kind: SupportStrictConvexCandidateCompilationKindV2
    artifact: SupportStrictConvexCandidateDomainArtifactV2_2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not SupportStrictConvexCandidateCompilationKindV2:
            raise TypeError("kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is SupportStrictConvexCandidateCompilationKindV2.ARTIFACT:
            if (
                type(self.artifact)
                is not SupportStrictConvexCandidateDomainArtifactV2_2
            ):
                raise ValueError("ARTIFACT outcome requires an exact artifact")
            if findings:
                raise ValueError("ARTIFACT outcome cannot carry findings")
            object.__setattr__(self, "artifact", _copy_artifact(self.artifact))
            return
        if self.artifact is not None or not findings:
            raise ValueError("failure requires findings and no artifact")


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateVerificationOutcomeV2:
    kind: SupportStrictConvexCandidateVerificationKindV2
    semantic_problem_sha256: str | None = None
    compiler_config_sha256: str | None = None
    artifact_sha256: str | None = None
    verification_resource_usage: (
        MultiObstacleStrictConvexCandidateResourceUsageV2 | None
    ) = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not SupportStrictConvexCandidateVerificationKindV2:
            raise TypeError("verification kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        refs = (
            self.semantic_problem_sha256,
            self.compiler_config_sha256,
            self.artifact_sha256,
        )
        if self.kind is SupportStrictConvexCandidateVerificationKindV2.VERIFIED:
            if any(
                type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None
                for digest in refs
            ):
                raise ValueError("VERIFIED outcome requires three SHA-256 references")
            if findings:
                raise ValueError("VERIFIED outcome cannot carry findings")
            if (
                type(self.verification_resource_usage)
                is not MultiObstacleStrictConvexCandidateResourceUsageV2
            ):
                raise ValueError("VERIFIED outcome requires replay resource usage")
        else:
            if any(digest is not None for digest in refs):
                raise ValueError("failure verification outcome cannot carry references")
            if not findings:
                raise ValueError("failure verification outcome requires findings")
        if self.verification_resource_usage is not None:
            if (
                type(self.verification_resource_usage)
                is not MultiObstacleStrictConvexCandidateResourceUsageV2
            ):
                raise TypeError("verification resource usage has the wrong exact type")
            usage = self.verification_resource_usage
            object.__setattr__(
                self,
                "verification_resource_usage",
                MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=usage.domain_operations,
                    so2_atomic_steps=usage.so2_atomic_steps,
                    candidate_cells=usage.candidate_cells,
                ),
            )


class SupportStrictConvexCandidateDomainCompilerV2_7:
    def compile(
        self,
        problem: SemanticProblemV2_2,
        config: StrictConvexCandidateCompilerConfigV2_7,
    ) -> SupportStrictConvexCandidateCompilationOutcomeV2:
        return compile_support_strict_convex_candidate_domain_v2_7(problem, config)


class _InvalidInputV2(ValueError):
    pass


def compile_support_strict_convex_candidate_domain_v2_7(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_7,
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    """Fresh-compile T14 plus one exact horizontal SUPPORT predicate."""

    try:
        checked_config = _strict_config(config)
    except _InvalidInputV2:
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:SUPPORT_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_CONFIG_REVALIDATION",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            upstream = compile_multi_obstacle_strict_convex_candidate_domain_v2_6(
                problem, _t14_config(checked_config)
            )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:UPSTREAM_MULTI_OBSTACLE_REPLAY",
        )
    if (
        upstream.kind
        is not MultiObstacleStrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(upstream.artifact)
        is not MultiObstacleStrictConvexCandidateDomainArtifactV2_2
    ):
        return _from_upstream_failure(upstream.kind, upstream.finding_codes)
    upstream_artifact = upstream.artifact
    budget = StrictConvexIntersectionBudgetV2(
        max_domain_operations=checked_config.max_domain_operations,
        max_candidate_cells=checked_config.max_candidate_cells,
        domain_operations_used=upstream_artifact.resource_usage.domain_operations,
        candidate_cells_used=upstream_artifact.resource_usage.candidate_cells,
    )
    atomic_budget = SO2AtomicBudgetV2(
        limit=checked_config.max_so2_atomic_steps,
        used=upstream_artifact.resource_usage.so2_atomic_steps,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _precharge_problem_structure(problem, budget)  # type: ignore[arg-type]
            checked_problem = _strict_problem(problem)
            if (
                checked_problem.semantic_problem_sha256
                != upstream_artifact.semantic_problem_sha256
            ):
                raise _InvalidInputV2
            support = compile_exact_horizontal_support_projection_v2(
                checked_problem,
                "constraint:support",
                upstream_artifact.search_universe,
                atomic_budget=atomic_budget,
                intersection_budget=budget,
            )
            if support.kind is ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT:
                return _resource_failure()
            if support.kind is ContinuousYawSupportProjectionKindV2.NUMERIC_GAP:
                return SupportStrictConvexCandidateCompilationOutcomeV2(
                    kind=SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
                    finding_codes=support.finding_codes,
                )
            if support.kind is ContinuousYawSupportProjectionKindV2.INVALID_INPUT:
                raise RuntimeError("strictly checked support input became invalid")
            if support.kind is ContinuousYawSupportProjectionKindV2.UNSUPPORTED_MODEL:
                return SupportStrictConvexCandidateCompilationOutcomeV2(
                    kind=SupportStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL,
                    finding_codes=support.finding_codes,
                )
            if (
                support.kind is not ContinuousYawSupportProjectionKindV2.BRACKET
                or type(support.bracket) is not ContinuousYawSupportProjectionBracketV2
            ):
                raise RuntimeError("malformed support projection outcome")
            support_bracket = support.bracket
            inner = _require_intersection(
                intersect_strict_convex_allowed_complexes_v2(
                    (
                        upstream_artifact.allowed_domain_bracket.inner_allowed,
                        support_bracket.inner_allowed,
                    ),
                    budget=budget,
                )
            )
            outer = _require_intersection(
                intersect_strict_convex_allowed_complexes_v2(
                    (
                        upstream_artifact.allowed_domain_bracket.outer_allowed,
                        support_bracket.outer_allowed,
                    ),
                    budget=budget,
                )
            )
            remaining = tuple(
                item
                for item in upstream_artifact.remaining_constraint_ids
                if item != support_bracket.support_constraint_id
            )
            if len(remaining) + 1 != len(upstream_artifact.remaining_constraint_ids):
                raise RuntimeError("T14 remaining IDs lost the support constraint")
            budget.consume_domain(
                24
                + len(remaining)
                + len(upstream_artifact.ordered_obstacle_body_ids)
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (inner, outer)
                    for cell in complex_.cells
                )
            )
            bracket = SupportStrictConvexAllowedBracketV2(
                inner_allowed=inner,
                outer_allowed=outer,
                intersection_kernel_id=checked_config.intersection_kernel_id,
                intersection_kernel_version=checked_config.intersection_kernel_version,
                support_projection_kernel_id=(
                    checked_config.support_projection_kernel_id
                ),
                support_projection_kernel_version=(
                    checked_config.support_projection_kernel_version
                ),
                so2_atomic_steps_used=atomic_budget.used,
            )
            artifact = SupportStrictConvexCandidateDomainArtifactV2_2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_config.config_sha256,
                upstream_t14_artifact_sha256=upstream_artifact.artifact_sha256,
                subject_id=upstream_artifact.subject_id,
                search_universe=upstream_artifact.search_universe,
                ordered_constraint_ids=(
                    *upstream_artifact.ordered_constraint_ids,
                    support_bracket.support_constraint_id,
                ),
                ordered_obstacle_body_ids=(upstream_artifact.ordered_obstacle_body_ids),
                support_constraint_id=support_bracket.support_constraint_id,
                surface_id=support_bracket.surface_id,
                contact_geometry_id=support_bracket.contact_geometry_id,
                allowed_domain_bracket=bracket,
                resource_usage=MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=budget.domain_operations_used,
                    so2_atomic_steps=atomic_budget.used,
                    candidate_cells=budget.candidate_cells_used,
                ),
                remaining_constraint_ids=remaining,
            )
            return SupportStrictConvexCandidateCompilationOutcomeV2(
                kind=SupportStrictConvexCandidateCompilationKindV2.ARTIFACT,
                artifact=artifact,
            )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _resource_failure()
    except (_InvalidInputV2, _LegacyInvalidInputV2):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:SUPPORT_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_STRICT_CONVEX_COMPILATION",
        )


def verify_support_strict_convex_candidate_domain_v2_7(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_7,
    submitted_artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
) -> SupportStrictConvexCandidateVerificationOutcomeV2:
    """Fresh replay raw inputs and compare the entire submitted T15 artifact."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_submitted = _copy_artifact(submitted_artifact)
    except (ArithmeticError, RuntimeWarning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUBMITTED_SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )
    except (AttributeError, TypeError, ValueError, Warning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("INVALID_INPUT:SUBMITTED_SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = compile_support_strict_convex_candidate_domain_v2_7(
                problem, config
            )
    except (ArithmeticError, RuntimeWarning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUPPORT_STRICT_CONVEX_REPLAY",),
        )
    if (
        replay.kind is not SupportStrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(replay.artifact) is not SupportStrictConvexCandidateDomainArtifactV2_2
    ):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=replay.finding_codes,
        )
    fresh = replay.artifact
    usage = fresh.resource_usage
    if (
        checked_submitted != fresh
        or _artifact_bytes(checked_submitted) != _artifact_bytes(fresh)  # type: ignore[arg-type]
        or checked_submitted.artifact_sha256 != fresh.artifact_sha256
    ):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.MISMATCH,
            verification_resource_usage=usage,
            finding_codes=("MISMATCH:SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )
    return SupportStrictConvexCandidateVerificationOutcomeV2(
        kind=SupportStrictConvexCandidateVerificationKindV2.VERIFIED,
        semantic_problem_sha256=fresh.semantic_problem_sha256,
        compiler_config_sha256=fresh.compiler_config_sha256,
        artifact_sha256=fresh.artifact_sha256,
        verification_resource_usage=usage,
    )


def _t14_config(
    config: StrictConvexCandidateCompilerConfigV2_7,
) -> StrictConvexCandidateCompilerConfigV2_6:
    return StrictConvexCandidateCompilerConfigV2_6(
        max_domain_operations=config.max_domain_operations,
        max_so2_atomic_steps=config.max_so2_atomic_steps,
        max_candidate_cells=config.max_candidate_cells,
    )


def _require_intersection(
    outcome: StrictConvexIntersectionOutcomeV2,
) -> StrictConvexIntersectionComplexV2:
    if outcome.kind is StrictConvexIntersectionKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is StrictConvexIntersectionKindV2.NUMERIC_GAP:
        raise ArithmeticError("strict-convex support intersection numeric gap")
    if outcome.kind is StrictConvexIntersectionKindV2.INVALID_INPUT:
        raise RuntimeError("compiler produced invalid support intersection operands")
    if (
        outcome.kind is not StrictConvexIntersectionKindV2.COMPLEX
        or type(outcome.complex) is not StrictConvexIntersectionComplexV2
    ):
        raise RuntimeError("malformed support intersection outcome")
    return outcome.complex


def _strict_config(value: object) -> StrictConvexCandidateCompilerConfigV2_7:
    if type(value) is not StrictConvexCandidateCompilerConfigV2_7:
        raise _InvalidInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return StrictConvexCandidateCompilerConfigV2_7.model_validate(
                value.model_dump(mode="python", warnings="error"), strict=True
            )
    except (ArithmeticError, RuntimeWarning):
        raise
    except (
        AttributeError,
        PydanticSerializationError,
        TypeError,
        ValidationError,
        ValueError,
        Warning,
    ) as error:
        raise _InvalidInputV2 from error


def _from_upstream_failure(
    kind: MultiObstacleStrictConvexCandidateCompilationKindV2,
    finding_codes: tuple[str, ...],
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    mapped = {
        MultiObstacleStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL: (
            SupportStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP: (
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT: (
            SupportStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.INVALID_INPUT: (
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT
        ),
    }.get(kind)
    if mapped is None:
        raise RuntimeError("malformed T14 compiler outcome")
    return SupportStrictConvexCandidateCompilationOutcomeV2(
        kind=mapped,
        finding_codes=finding_codes,
    )


def _copy_bracket(
    value: SupportStrictConvexAllowedBracketV2,
) -> SupportStrictConvexAllowedBracketV2:
    return SupportStrictConvexAllowedBracketV2(
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        intersection_kernel_id=value.intersection_kernel_id,
        intersection_kernel_version=value.intersection_kernel_version,
        support_projection_kernel_id=value.support_projection_kernel_id,
        support_projection_kernel_version=value.support_projection_kernel_version,
        so2_atomic_steps_used=value.so2_atomic_steps_used,
    )


def _copy_artifact(
    value: SupportStrictConvexCandidateDomainArtifactV2_2,
) -> SupportStrictConvexCandidateDomainArtifactV2_2:
    return SupportStrictConvexCandidateDomainArtifactV2_2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        upstream_t14_artifact_sha256=value.upstream_t14_artifact_sha256,
        subject_id=value.subject_id,
        search_universe=value.search_universe,
        ordered_constraint_ids=value.ordered_constraint_ids,
        ordered_obstacle_body_ids=value.ordered_obstacle_body_ids,
        support_constraint_id=value.support_constraint_id,
        surface_id=value.surface_id,
        contact_geometry_id=value.contact_geometry_id,
        allowed_domain_bracket=value.allowed_domain_bracket,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
    )


def _failure(
    kind: SupportStrictConvexCandidateCompilationKindV2,
    finding_code: str,
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    return SupportStrictConvexCandidateCompilationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _resource_failure() -> SupportStrictConvexCandidateCompilationOutcomeV2:
    return _failure(
        SupportStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
        "RESOURCE_LIMIT:SUPPORT_STRICT_CONVEX_CANDIDATE",
    )


__all__ = (
    "SupportStrictConvexAllowedBracketV2",
    "SupportStrictConvexCandidateCompilationKindV2",
    "SupportStrictConvexCandidateCompilationOutcomeV2",
    "SupportStrictConvexCandidateDomainArtifactV2_2",
    "SupportStrictConvexCandidateDomainCompilerV2_7",
    "SupportStrictConvexCandidateVerificationKindV2",
    "SupportStrictConvexCandidateVerificationOutcomeV2",
    "compile_support_strict_convex_candidate_domain_v2_7",
    "verify_support_strict_convex_candidate_domain_v2_7",
)
