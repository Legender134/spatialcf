"""Raw multi-obstacle strict-convex candidate compilation for Canonical v2.2."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from fractions import Fraction
from typing import Any

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.convex_translation_partition import (
    ConvexAllowedTranslationBracketV2,
    ConvexAllowedTranslationKindV2,
    compile_convex_allowed_translation_v2,
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
from spatialcf.core.v2.strict_convex_candidate_domain import (
    _precharge_problem_structure,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionKindV2,
    intersect_strict_convex_allowed_complexes_v2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec3V2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    CollisionClearanceMetricV2,
    PositionRegionInterpretationV2,
    RegionAggregationV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_candidate import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_6,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

_ARTIFACT_HASH_DOMAIN_V2_2 = (
    b"spatialcf.multi-obstacle-strict-convex-candidate-artifact.v2.2\0"
)
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTERSECTION_KERNEL_ID = "geometry-kernel:rational-strict-convex-intersection-v2"
_INTERSECTION_KERNEL_VERSION = "kernel:2.5-strict-convex-intersection"


class MultiObstacleStrictConvexCandidateCompilationKindV2(StrEnum):
    ARTIFACT = "ARTIFACT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class MultiObstacleStrictConvexCandidateVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class MultiObstacleStrictConvexCandidateResourceUsageV2:
    domain_operations: int
    so2_atomic_steps: int
    candidate_cells: int

    def __post_init__(self) -> None:
        if type(self.domain_operations) is not int or self.domain_operations < 0:
            raise ValueError("domain_operations must be a non-negative exact int")
        if type(self.so2_atomic_steps) is not int or self.so2_atomic_steps <= 0:
            raise ValueError("so2_atomic_steps must be a positive exact int")
        if type(self.candidate_cells) is not int or self.candidate_cells < 0:
            raise ValueError("candidate_cells must be a non-negative exact int")


@dataclass(frozen=True, slots=True)
class MultiObstacleStrictConvexAllowedBracketV2:
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    intersection_kernel_id: str
    intersection_kernel_version: str
    so2_atomic_steps_used: int

    def __post_init__(self) -> None:
        checked_inner = _copy_intersection_complex(self.inner_allowed)
        checked_outer = _copy_intersection_complex(self.outer_allowed)
        if checked_inner.universe != checked_outer.universe:
            raise ValueError("inner and outer intersections require one universe")
        if self.intersection_kernel_id != _INTERSECTION_KERNEL_ID:
            raise ValueError("unexpected intersection kernel ID")
        if self.intersection_kernel_version != _INTERSECTION_KERNEL_VERSION:
            raise ValueError("unexpected intersection kernel version")
        if (
            type(self.so2_atomic_steps_used) is not int
            or self.so2_atomic_steps_used <= 0
        ):
            raise ValueError("so2_atomic_steps_used must be a positive exact int")
        if not all(
            checked_outer.contains_point(cell.strict_witness)
            for cell in checked_inner.cells
        ):
            raise ValueError("inner intersection witness escaped outer intersection")
        object.__setattr__(self, "inner_allowed", checked_inner)
        object.__setattr__(self, "outer_allowed", checked_outer)


@dataclass(frozen=True, slots=True)
class MultiObstacleStrictConvexCandidateDomainArtifactV2_2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    subject_id: str
    search_universe: ExactAxisAlignedRectV2
    ordered_constraint_ids: tuple[str, ...]
    ordered_obstacle_body_ids: tuple[str, ...]
    allowed_domain_bracket: MultiObstacleStrictConvexAllowedBracketV2
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
        ):
            if type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if type(self.subject_id) is not str or not self.subject_id.strip():
            raise ValueError("subject_id must be a non-blank exact string")
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
        if set(compiled_ids) & set(remaining_ids):
            raise ValueError("compiled and remaining constraint IDs must be disjoint")
        if (
            type(self.allowed_domain_bracket)
            is not MultiObstacleStrictConvexAllowedBracketV2
        ):
            raise TypeError("allowed_domain_bracket has the wrong exact type")
        checked_bracket = _copy_bracket(self.allowed_domain_bracket)
        if (
            checked_bracket.inner_allowed.universe != checked_universe
            or checked_bracket.outer_allowed.universe != checked_universe
        ):
            raise ValueError("allowed bracket must use the exact search universe")
        if (
            type(self.resource_usage)
            is not MultiObstacleStrictConvexCandidateResourceUsageV2
        ):
            raise TypeError("resource_usage has the wrong exact type")
        checked_usage = MultiObstacleStrictConvexCandidateResourceUsageV2(
            domain_operations=self.resource_usage.domain_operations,
            so2_atomic_steps=self.resource_usage.so2_atomic_steps,
            candidate_cells=self.resource_usage.candidate_cells,
        )
        if checked_usage.so2_atomic_steps != checked_bracket.so2_atomic_steps_used:
            raise ValueError("SO(2) usage must equal the aggregate bracket usage")
        if checked_usage.candidate_cells != (
            len(checked_bracket.inner_allowed.cells)
            + len(checked_bracket.outer_allowed.cells)
        ):
            raise ValueError("candidate-cell usage must equal all published cells")
        object.__setattr__(self, "search_universe", checked_universe)
        object.__setattr__(self, "ordered_constraint_ids", compiled_ids)
        object.__setattr__(self, "ordered_obstacle_body_ids", obstacle_ids)
        object.__setattr__(self, "remaining_constraint_ids", remaining_ids)
        object.__setattr__(self, "allowed_domain_bracket", checked_bracket)
        object.__setattr__(self, "resource_usage", checked_usage)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            _ARTIFACT_HASH_DOMAIN_V2_2 + _artifact_bytes(self)
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MultiObstacleStrictConvexCandidateCompilationOutcomeV2:
    kind: MultiObstacleStrictConvexCandidateCompilationKindV2
    artifact: MultiObstacleStrictConvexCandidateDomainArtifactV2_2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not MultiObstacleStrictConvexCandidateCompilationKindV2:
            raise TypeError("kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is MultiObstacleStrictConvexCandidateCompilationKindV2.ARTIFACT:
            if (
                type(self.artifact)
                is not MultiObstacleStrictConvexCandidateDomainArtifactV2_2
            ):
                raise ValueError("ARTIFACT outcome requires an exact artifact")
            if findings:
                raise ValueError("ARTIFACT outcome cannot carry findings")
            object.__setattr__(self, "artifact", _copy_artifact(self.artifact))
            return
        if self.artifact is not None:
            raise ValueError("failure outcome cannot carry an artifact")
        if not findings:
            raise ValueError("failure outcome requires at least one finding")


@dataclass(frozen=True, slots=True)
class MultiObstacleStrictConvexCandidateVerificationOutcomeV2:
    kind: MultiObstacleStrictConvexCandidateVerificationKindV2
    semantic_problem_sha256: str | None = None
    compiler_config_sha256: str | None = None
    artifact_sha256: str | None = None
    verification_resource_usage: (
        MultiObstacleStrictConvexCandidateResourceUsageV2 | None
    ) = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not MultiObstacleStrictConvexCandidateVerificationKindV2:
            raise TypeError("verification kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        refs = (
            self.semantic_problem_sha256,
            self.compiler_config_sha256,
            self.artifact_sha256,
        )
        if self.kind is MultiObstacleStrictConvexCandidateVerificationKindV2.VERIFIED:
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
            object.__setattr__(
                self,
                "verification_resource_usage",
                MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=(
                        self.verification_resource_usage.domain_operations
                    ),
                    so2_atomic_steps=self.verification_resource_usage.so2_atomic_steps,
                    candidate_cells=self.verification_resource_usage.candidate_cells,
                ),
            )


class MultiObstacleStrictConvexCandidateDomainCompilerV2_6:
    def compile(
        self,
        problem: SemanticProblemV2_2,
        config: StrictConvexCandidateCompilerConfigV2_6,
    ) -> MultiObstacleStrictConvexCandidateCompilationOutcomeV2:
        return compile_multi_obstacle_strict_convex_candidate_domain_v2_6(
            problem, config
        )


class _InvalidInputV2(ValueError):
    pass


class _UnsupportedModelV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code


def compile_multi_obstacle_strict_convex_candidate_domain_v2_6(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_6,
) -> MultiObstacleStrictConvexCandidateCompilationOutcomeV2:
    """Fresh-compile the bounded multi-obstacle collision prefix."""

    try:
        checked_config = _strict_config(config)
    except _InvalidInputV2:
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:MULTI_OBSTACLE_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:MULTI_OBSTACLE_REVALIDATION",
        )

    budget = StrictConvexIntersectionBudgetV2(
        max_domain_operations=checked_config.max_domain_operations,
        max_candidate_cells=checked_config.max_candidate_cells,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _precharge_problem_structure(problem, budget)  # type: ignore[arg-type]
            checked_problem = _strict_problem(problem)
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _resource_failure()
    except (_InvalidInputV2, _LegacyInvalidInputV2):
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:MULTI_OBSTACLE_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:MULTI_OBSTACLE_REVALIDATION",
        )

    atomic_budget = SO2AtomicBudgetV2(limit=checked_config.max_so2_atomic_steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            from spatialcf.core.v2.candidate_domain import (
                _compile_search_universe,
                _SearchUniverseFailureV2,
            )

            search = _compile_search_universe(checked_problem, budget)  # type: ignore[arg-type]
            if isinstance(search, _SearchUniverseFailureV2):
                kind = (
                    MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP
                    if search.uncertified_reason.value == "NUMERIC_GAP"
                    else MultiObstacleStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL
                )
                return MultiObstacleStrictConvexCandidateCompilationOutcomeV2(
                    kind=kind,
                    finding_codes=search.finding_codes,
                )
            subject, obstacles = _extract_supported_pairs(checked_problem, budget)
            inner_complexes = []
            outer_complexes = []
            for _, obstacle_transform, obstacle_shape in obstacles:
                allowed = compile_convex_allowed_translation_v2(
                    subject[0],
                    subject[1],
                    obstacle_transform,
                    obstacle_shape,
                    search.delta_rect,
                    atomic_budget=atomic_budget,
                )
                if allowed.kind is ConvexAllowedTranslationKindV2.RESOURCE_LIMIT:
                    return _failure(
                        MultiObstacleStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
                        "RESOURCE_LIMIT:max_so2_atomic_steps",
                    )
                if allowed.kind is ConvexAllowedTranslationKindV2.NUMERIC_GAP:
                    return MultiObstacleStrictConvexCandidateCompilationOutcomeV2(
                        kind=MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
                        finding_codes=allowed.finding_codes,
                    )
                if allowed.kind is ConvexAllowedTranslationKindV2.INVALID_INPUT:
                    raise RuntimeError("supported obstacle produced invalid T12 input")
                if (
                    allowed.kind is not ConvexAllowedTranslationKindV2.BRACKET
                    or type(allowed.bracket) is not ConvexAllowedTranslationBracketV2
                ):
                    raise RuntimeError("malformed T12 allowed-domain outcome")
                inner_complexes.append(allowed.bracket.inner_allowed)
                outer_complexes.append(allowed.bracket.outer_allowed)

            inner = intersect_strict_convex_allowed_complexes_v2(
                tuple(inner_complexes), budget=budget
            )
            outer = intersect_strict_convex_allowed_complexes_v2(
                tuple(outer_complexes), budget=budget
            )
            checked_inner = _require_intersection_success(inner)
            checked_outer = _require_intersection_success(outer)

            constraints = checked_problem.constraints
            remaining_ids = tuple(
                sorted(
                    (
                        *(
                            item.constraint_id
                            for item in constraints.support_constraints
                        ),
                        *(
                            item.constraint_id
                            for item in constraints.visibility_constraints
                        ),
                        constraints.target_relation.constraint_id,
                    )
                )
            )
            budget.consume_domain(
                12
                + len(remaining_ids)
                + len(obstacles)
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (checked_inner, checked_outer)
                    for cell in complex_.cells
                )
            )
            bracket = MultiObstacleStrictConvexAllowedBracketV2(
                inner_allowed=checked_inner,
                outer_allowed=checked_outer,
                intersection_kernel_id=checked_config.intersection_kernel_id,
                intersection_kernel_version=checked_config.intersection_kernel_version,
                so2_atomic_steps_used=atomic_budget.used,
            )
            artifact = MultiObstacleStrictConvexCandidateDomainArtifactV2_2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_config.config_sha256,
                subject_id=constraints.allowed_edit.subject_id,
                search_universe=search.delta_rect,
                ordered_constraint_ids=(
                    constraints.position_domain.constraint_id,
                    constraints.collision_constraints[0].constraint_id,
                ),
                ordered_obstacle_body_ids=tuple(item[0] for item in obstacles),
                allowed_domain_bracket=bracket,
                resource_usage=MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=budget.domain_operations_used,
                    so2_atomic_steps=atomic_budget.used,
                    candidate_cells=budget.candidate_cells_used,
                ),
                remaining_constraint_ids=remaining_ids,
            )
            return MultiObstacleStrictConvexCandidateCompilationOutcomeV2(
                kind=MultiObstacleStrictConvexCandidateCompilationKindV2.ARTIFACT,
                artifact=artifact,
            )
    except _UnsupportedModelV2 as error:
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _resource_failure()
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:MULTI_OBSTACLE_COMPILATION",
        )


def verify_multi_obstacle_strict_convex_candidate_domain_v2_6(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_6,
    submitted_artifact: MultiObstacleStrictConvexCandidateDomainArtifactV2_2,
) -> MultiObstacleStrictConvexCandidateVerificationOutcomeV2:
    """Fresh replay raw inputs and compare the entire submitted T14 artifact."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_submitted = _copy_artifact(submitted_artifact)
    except (ArithmeticError, RuntimeWarning):
        return MultiObstacleStrictConvexCandidateVerificationOutcomeV2(
            kind=MultiObstacleStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUBMITTED_MULTI_OBSTACLE_ARTIFACT",),
        )
    except (AttributeError, TypeError, ValueError, Warning):
        return MultiObstacleStrictConvexCandidateVerificationOutcomeV2(
            kind=MultiObstacleStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("INVALID_INPUT:SUBMITTED_MULTI_OBSTACLE_ARTIFACT",),
        )

    replay = compile_multi_obstacle_strict_convex_candidate_domain_v2_6(problem, config)
    if (
        replay.kind is not MultiObstacleStrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(replay.artifact)
        is not MultiObstacleStrictConvexCandidateDomainArtifactV2_2
    ):
        return MultiObstacleStrictConvexCandidateVerificationOutcomeV2(
            kind=MultiObstacleStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=replay.finding_codes,
        )
    fresh = replay.artifact
    usage = fresh.resource_usage
    if (
        checked_submitted != fresh
        or _artifact_bytes(checked_submitted) != _artifact_bytes(fresh)
        or checked_submitted.artifact_sha256 != fresh.artifact_sha256
    ):
        return MultiObstacleStrictConvexCandidateVerificationOutcomeV2(
            kind=MultiObstacleStrictConvexCandidateVerificationKindV2.MISMATCH,
            verification_resource_usage=usage,
            finding_codes=("MISMATCH:MULTI_OBSTACLE_STRICT_CONVEX_ARTIFACT",),
        )
    return MultiObstacleStrictConvexCandidateVerificationOutcomeV2(
        kind=MultiObstacleStrictConvexCandidateVerificationKindV2.VERIFIED,
        semantic_problem_sha256=fresh.semantic_problem_sha256,
        compiler_config_sha256=fresh.compiler_config_sha256,
        artifact_sha256=fresh.artifact_sha256,
        verification_resource_usage=usage,
    )


def _extract_supported_pairs(
    problem: SemanticProblemV2_2,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[
    tuple[DirectedYawIntervalTransformV2_2, UprightBox3DV2],
    tuple[tuple[str, DirectedYawIntervalTransformV2_2, UprightBox3DV2], ...],
]:
    constraints = problem.constraints
    position = constraints.position_domain
    if (
        position.region_interpretation
        is not PositionRegionInterpretationV2.SUBJECT_ANCHOR_LOCUS
        or position.workspace_aggregation is not RegionAggregationV2.INTERSECTION
        or position.boundary_policy is not BoundaryPolicyV2.CLOSED
        or position.known_free_space_fact_ids
        or len(position.workspace_fact_ids) != 1
        or position.minimum_boundary_clearance_m != 0.0
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:POSITION_DOMAIN_SUBSET")
    workspace_values = _exact_fact_values(
        problem.scene.workspace_boundaries, "WORKSPACE_BOUNDARIES", budget
    )
    if (
        len(workspace_values) != 1
        or workspace_values[0].fact_id != position.workspace_fact_ids[0]
        or workspace_values[0].region_approximation is not GeometryApproximationV2.EXACT
        or workspace_values[0].geometry_uncertainty != UncertaintyBudgetV2()
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:POSITION_WORKSPACE_SUBSET")
    if len(constraints.collision_constraints) != 1:
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_CONSTRAINT_CARDINALITY")
    collision = constraints.collision_constraints[0]
    if len(collision.subject_body_ids) != 1 or not collision.obstacle_body_ids:
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_PAIR_CARDINALITY")
    if tuple(sorted(set(collision.obstacle_body_ids))) != collision.obstacle_body_ids:
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_OBSTACLE_IDENTITIES")
    if (
        collision.clearance_metric
        is not CollisionClearanceMetricV2.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
        or collision.boundary_policy is not BoundaryPolicyV2.CLOSED
        or collision.minimum_clearance_m != 0.0
        or collision.support_contact_exceptions
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_POLICY")
    if problem.numeric_policy != NumericPolicyV2():
        raise _UnsupportedModelV2("UNSUPPORTED:NUMERIC_POLICY")

    bodies = _exact_fact_values(
        problem.scene.collision_bodies, "COLLISION_BODIES", budget
    )
    geometries = _exact_fact_values(
        problem.scene.geometry_instances, "GEOMETRY_INSTANCES", budget
    )
    objects = _exact_fact_values(problem.scene.objects, "OBJECTS", budget)
    body_by_id = {item.body_id: item for item in bodies}
    geometry_by_id = {item.geometry_id: item for item in geometries}
    object_by_id = {item.object_id: item for item in objects}
    budget.consume_domain(len(bodies) + len(geometries) + len(objects))

    subject_id = constraints.allowed_edit.subject_id
    try:
        subject_body = body_by_id[collision.subject_body_ids[0]]
        subject_object = object_by_id[subject_id]
    except KeyError as error:
        raise RuntimeError("semantic graph lost the collision subject") from error
    if (
        subject_body.owner_object_id != subject_id
        or len(subject_body.geometry_instance_ids) != 1
        or not subject_object.movable
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_BODY_SUBSET")
    try:
        subject_geometry = geometry_by_id[subject_body.geometry_instance_ids[0]]
    except KeyError as error:
        raise RuntimeError("semantic graph lost subject collision geometry") from error
    subject_shape = _require_collision_geometry(subject_geometry, subject_id)
    subject_transform = subject_object.pose.world_from_object
    if type(subject_transform) is not DirectedYawIntervalTransformV2_2:
        raise RuntimeError("v2.2 subject pose lost its directed-yaw transform")

    obstacles = []
    for body_id in collision.obstacle_body_ids:
        try:
            body = body_by_id[body_id]
            owner_id = body.owner_object_id
        except KeyError as error:
            raise RuntimeError("semantic graph lost an obstacle reference") from error
        if owner_id == subject_id or len(body.geometry_instance_ids) != 1:
            raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_BODY_SUBSET")
        try:
            obstacle_geometry = geometry_by_id[body.geometry_instance_ids[0]]
        except KeyError as error:
            raise RuntimeError(
                "semantic graph lost obstacle collision geometry"
            ) from error
        if owner_id is None:
            obstacle_shape, obstacle_transform = (
                _require_environment_collision_geometry(obstacle_geometry)
            )
        else:
            try:
                obstacle_object = object_by_id[owner_id]
            except KeyError as error:
                raise RuntimeError("semantic graph lost an obstacle owner") from error
            if obstacle_object.movable:
                raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_BODY_SUBSET")
            obstacle_shape = _require_collision_geometry(obstacle_geometry, owner_id)
            obstacle_transform = obstacle_object.pose.world_from_object
        if type(obstacle_transform) is not DirectedYawIntervalTransformV2_2:
            raise RuntimeError("v2.2 obstacle pose lost its directed-yaw transform")
        # T12 publishes the exact full universe when immutable subject and
        # obstacle Z interiors are disjoint (including closed-face contact).
        # Keep the pair in the replay ledger rather than rejecting it or
        # silently dropping its semantic identity.
        budget.consume_domain(4)
        obstacles.append((body_id, obstacle_transform, obstacle_shape))
    return (subject_transform, subject_shape), tuple(obstacles)


def _require_collision_geometry(geometry: Any, owner_id: str) -> UprightBox3DV2:
    if (
        type(geometry) is not GeometryInstanceV2_2
        or geometry.owner_object_id != owner_id
        or geometry.role is not GeometryRoleV2.COLLISION
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or geometry.uncertainty != UncertaintyBudgetV2()
        or type(geometry.shape) is not UprightBox3DV2
        or not _is_identity_anchor(geometry.anchor_from_geometry)
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_GEOMETRY_SUBSET")
    return geometry.shape


def _require_environment_collision_geometry(
    geometry: Any,
) -> tuple[UprightBox3DV2, DirectedYawIntervalTransformV2_2]:
    """Close one ownerless world-frame collision box for the T12 kernel."""

    if (
        type(geometry) is not GeometryInstanceV2_2
        or geometry.owner_object_id is not None
        or geometry.role is not GeometryRoleV2.COLLISION
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or geometry.uncertainty != UncertaintyBudgetV2()
        or type(geometry.shape) is not UprightBox3DV2
        or type(geometry.anchor_from_geometry) is not DirectedYawIntervalTransformV2_2
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_GEOMETRY_SUBSET")
    return geometry.shape, geometry.anchor_from_geometry


def _exact_fact_values(
    facts: Any,
    label: str,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[Any, ...]:
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.uncertainty != UncertaintyBudgetV2()
        or type(facts.values) is not tuple
    ):
        raise _UnsupportedModelV2(f"UNSUPPORTED:{label}_FACT_SET")
    budget.consume_domain(len(facts.values) + 1)
    return facts.values


def _is_identity_anchor(transform: DirectedYawIntervalTransformV2_2) -> bool:
    return (
        type(transform) is DirectedYawIntervalTransformV2_2
        and transform.translation == Vec3V2(x=0.0, y=0.0, z=0.0)
        and transform.yaw_radians == 0.0
    )


def _require_intersection_success(
    outcome: Any,
) -> StrictConvexIntersectionComplexV2:
    if outcome.kind is StrictConvexIntersectionKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is StrictConvexIntersectionKindV2.NUMERIC_GAP:
        raise ArithmeticError("strict-convex intersection numeric gap")
    if outcome.kind is StrictConvexIntersectionKindV2.INVALID_INPUT:
        raise RuntimeError("compiler produced invalid strict-convex operands")
    if (
        outcome.kind is not StrictConvexIntersectionKindV2.COMPLEX
        or type(outcome.complex) is not StrictConvexIntersectionComplexV2
    ):
        raise RuntimeError("malformed strict-convex intersection outcome")
    return outcome.complex


def _strict_config(value: object) -> StrictConvexCandidateCompilerConfigV2_6:
    if type(value) is not StrictConvexCandidateCompilerConfigV2_6:
        raise _InvalidInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            payload = value.model_dump(mode="python", warnings="error")
            return StrictConvexCandidateCompilerConfigV2_6.model_validate(
                payload, strict=True
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


def _strict_problem(value: object) -> SemanticProblemV2_2:
    if type(value) is not SemanticProblemV2_2:
        raise _InvalidInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            payload = value.model_dump(mode="python", warnings="error")
            return SemanticProblemV2_2.model_validate(payload, strict=True)
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


def _require_id_tuple(
    value: object,
    *,
    label: str,
    nonempty: bool,
    sorted_required: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ValueError(f"{label} must be an exact tuple with valid cardinality")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-blank exact strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must contain unique IDs")
    if sorted_required and tuple(sorted(value)) != value:
        raise ValueError(f"{label} must be canonically sorted")
    return value


def _require_finding_codes(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise ValueError("finding_codes must be exact non-blank strings")
    return tuple(sorted(set(value)))


def _copy_universe(value: ExactAxisAlignedRectV2) -> ExactAxisAlignedRectV2:
    if type(value) is not ExactAxisAlignedRectV2:
        raise TypeError("search_universe has the wrong exact type")
    bounds = value.bounds
    if bounds is None:
        raise ValueError("search universe cannot be empty")
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=bounds[0],
        min_y_m=bounds[1],
        max_x_m=bounds[2],
        max_y_m=bounds[3],
        coordinate_space=value.coordinate_space,
    )


def _copy_intersection_complex(
    value: StrictConvexIntersectionComplexV2,
) -> StrictConvexIntersectionComplexV2:
    if type(value) is not StrictConvexIntersectionComplexV2:
        raise TypeError("intersection complex has the wrong exact type")
    return StrictConvexIntersectionComplexV2(
        cells=value.cells,
        universe=value.universe,
        topology=value.topology,
    )


def _copy_bracket(
    value: MultiObstacleStrictConvexAllowedBracketV2,
) -> MultiObstacleStrictConvexAllowedBracketV2:
    return MultiObstacleStrictConvexAllowedBracketV2(
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        intersection_kernel_id=value.intersection_kernel_id,
        intersection_kernel_version=value.intersection_kernel_version,
        so2_atomic_steps_used=value.so2_atomic_steps_used,
    )


def _copy_artifact(
    value: MultiObstacleStrictConvexCandidateDomainArtifactV2_2,
) -> MultiObstacleStrictConvexCandidateDomainArtifactV2_2:
    return MultiObstacleStrictConvexCandidateDomainArtifactV2_2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        subject_id=value.subject_id,
        search_universe=value.search_universe,
        ordered_constraint_ids=value.ordered_constraint_ids,
        ordered_obstacle_body_ids=value.ordered_obstacle_body_ids,
        allowed_domain_bracket=value.allowed_domain_bracket,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
    )


def _artifact_bytes(
    value: MultiObstacleStrictConvexCandidateDomainArtifactV2_2,
) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "denominator": _canonical_integer(value.denominator),
            "numerator": _canonical_integer(value.numerator),
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported artifact hash value: {type(value).__name__}")


def _canonical_integer(value: int) -> int | dict[str, str]:
    """Serialize huge exact integers without Python's decimal-digit limit.

    Ordinary values deliberately retain the historical JSON number encoding,
    preserving every existing artifact byte and hash.  Very large directed
    rational coefficients use a frozen signed hexadecimal representation;
    hexadecimal conversion is linear and is not governed by
    ``sys.int_max_str_digits``.
    """

    if type(value) is not int:
        raise TypeError("canonical artifact integer must have exact int type")
    # Frozen v2.6/v2.7 artifacts reach 8,583 bits and must retain their exact
    # historical decimal JSON bytes.  Twelve thousand bits remain safely
    # below CPython's default 4,300-decimal-digit conversion boundary, while
    # the larger v2.9 projection coefficients use the limit-independent form.
    if value.bit_length() <= 12_000:
        return value
    sign = "-" if value < 0 else "+"
    return {"encoding": "signed-hex-v1", "value": sign + format(abs(value), "x")}


def _failure(
    kind: MultiObstacleStrictConvexCandidateCompilationKindV2,
    finding_code: str,
) -> MultiObstacleStrictConvexCandidateCompilationOutcomeV2:
    return MultiObstacleStrictConvexCandidateCompilationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _resource_failure() -> MultiObstacleStrictConvexCandidateCompilationOutcomeV2:
    return _failure(
        MultiObstacleStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
        "RESOURCE_LIMIT:MULTI_OBSTACLE_STRICT_CONVEX_CANDIDATE",
    )


__all__ = (
    "MultiObstacleStrictConvexAllowedBracketV2",
    "MultiObstacleStrictConvexCandidateCompilationKindV2",
    "MultiObstacleStrictConvexCandidateCompilationOutcomeV2",
    "MultiObstacleStrictConvexCandidateDomainArtifactV2_2",
    "MultiObstacleStrictConvexCandidateDomainCompilerV2_6",
    "MultiObstacleStrictConvexCandidateResourceUsageV2",
    "MultiObstacleStrictConvexCandidateVerificationKindV2",
    "MultiObstacleStrictConvexCandidateVerificationOutcomeV2",
    "compile_multi_obstacle_strict_convex_candidate_domain_v2_6",
    "verify_multi_obstacle_strict_convex_candidate_domain_v2_6",
)
