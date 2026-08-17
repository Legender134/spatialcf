"""Frozen candidate-stage values for strict convex continuous-yaw domains.

These values are immutable proposals, not proof capabilities. A semantic
consumer must fresh replay the raw problem and compiler config.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from fractions import Fraction
from typing import Any, ClassVar

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2._internal.resources.domain_operations import (
    DomainOperationBudgetV2,
)
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
    StrictConvexCandidateCompilerConfigV2_5,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

_ARTIFACT_HASH_DOMAIN_V2_2 = b"spatialcf.strict-convex-candidate-artifact.v2.2\0"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class StrictConvexCandidateCompilationKindV2(StrEnum):
    ARTIFACT = "ARTIFACT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class StrictConvexCandidateVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class StrictConvexCandidateResourceUsageV2:
    domain_operations: int
    so2_atomic_steps: int

    def __post_init__(self) -> None:
        if type(self.domain_operations) is not int or self.domain_operations < 0:
            raise ValueError("domain_operations must be a non-negative exact int")
        if type(self.so2_atomic_steps) is not int or self.so2_atomic_steps <= 0:
            raise ValueError("so2_atomic_steps must be a positive exact int")


@dataclass(frozen=True, slots=True)
class StrictConvexCandidateDomainArtifactV2_2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    subject_id: str
    search_universe: ExactAxisAlignedRectV2
    ordered_constraint_ids: tuple[str, ...]
    allowed_domain_bracket: ConvexAllowedTranslationBracketV2
    resource_usage: StrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
        ):
            if type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.subject_id) is not str or not self.subject_id.strip():
            raise ValueError("subject_id must be a non-blank exact string")
        checked_universe = _copy_universe(self.search_universe)
        if (
            checked_universe.topology is not RectTopologyV2.AREA
            or checked_universe.coordinate_space
            is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        ):
            raise ValueError("search universe must be an AREA translation-delta rect")
        if type(self.ordered_constraint_ids) is not tuple or not (
            self.ordered_constraint_ids
        ):
            raise ValueError("ordered_constraint_ids must be a non-empty exact tuple")
        if any(
            type(item) is not str or not item.strip()
            for item in self.ordered_constraint_ids
        ):
            raise ValueError("constraint IDs must be non-blank exact strings")
        if len(set(self.ordered_constraint_ids)) != len(self.ordered_constraint_ids):
            raise ValueError("constraint IDs must be unique")
        if type(self.remaining_constraint_ids) is not tuple or any(
            type(item) is not str or not item.strip()
            for item in self.remaining_constraint_ids
        ):
            raise ValueError(
                "remaining_constraint_ids must be an exact tuple of non-blank strings"
            )
        if len(set(self.remaining_constraint_ids)) != len(
            self.remaining_constraint_ids
        ):
            raise ValueError("remaining constraint IDs must be unique")
        if (
            tuple(sorted(self.remaining_constraint_ids))
            != self.remaining_constraint_ids
        ):
            raise ValueError("remaining constraint IDs must be canonically ordered")
        if set(self.ordered_constraint_ids) & set(self.remaining_constraint_ids):
            raise ValueError("compiled and remaining constraint IDs must be disjoint")
        checked_bracket = _copy_bracket(self.allowed_domain_bracket)
        if (
            checked_bracket.inner_allowed.universe != checked_universe
            or checked_bracket.outer_allowed.universe != checked_universe
        ):
            raise ValueError("allowed bracket must use the exact search universe")
        if type(self.resource_usage) is not StrictConvexCandidateResourceUsageV2:
            raise TypeError("resource_usage has the wrong exact type")
        checked_usage = StrictConvexCandidateResourceUsageV2(
            self.resource_usage.domain_operations,
            self.resource_usage.so2_atomic_steps,
        )
        if checked_usage.so2_atomic_steps != checked_bracket.atomic_steps_used:
            raise ValueError("SO(2) resource usage must equal the bracket replay usage")
        object.__setattr__(self, "search_universe", checked_universe)
        object.__setattr__(self, "allowed_domain_bracket", checked_bracket)
        object.__setattr__(self, "resource_usage", checked_usage)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            _ARTIFACT_HASH_DOMAIN_V2_2 + _artifact_bytes(self)
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class StrictConvexCandidateCompilationOutcomeV2:
    kind: StrictConvexCandidateCompilationKindV2
    artifact: StrictConvexCandidateDomainArtifactV2_2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not StrictConvexCandidateCompilationKindV2:
            raise TypeError("kind has the wrong exact type")
        if type(self.finding_codes) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is StrictConvexCandidateCompilationKindV2.ARTIFACT:
            if type(self.artifact) is not StrictConvexCandidateDomainArtifactV2_2:
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
class StrictConvexCandidateVerificationOutcomeV2:
    kind: StrictConvexCandidateVerificationKindV2
    semantic_problem_sha256: str | None = None
    compiler_config_sha256: str | None = None
    artifact_sha256: str | None = None
    verification_resource_usage: StrictConvexCandidateResourceUsageV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not StrictConvexCandidateVerificationKindV2:
            raise TypeError("verification kind has the wrong exact type")
        if type(self.finding_codes) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        refs = (
            self.semantic_problem_sha256,
            self.compiler_config_sha256,
            self.artifact_sha256,
        )
        if self.kind is StrictConvexCandidateVerificationKindV2.VERIFIED:
            if any(
                type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None
                for value in refs
            ):
                raise ValueError("VERIFIED requires three exact SHA-256 references")
            if (
                type(self.verification_resource_usage)
                is not StrictConvexCandidateResourceUsageV2
            ):
                raise ValueError("VERIFIED requires exact replay resource usage")
            object.__setattr__(
                self,
                "verification_resource_usage",
                StrictConvexCandidateResourceUsageV2(
                    self.verification_resource_usage.domain_operations,
                    self.verification_resource_usage.so2_atomic_steps,
                ),
            )
            if findings:
                raise ValueError("VERIFIED cannot carry findings")
            return
        if any(value is not None for value in refs):
            raise ValueError("non-VERIFIED outcomes cannot carry verified references")
        if not findings:
            raise ValueError("non-VERIFIED outcomes require findings")
        if self.verification_resource_usage is not None:
            if (
                type(self.verification_resource_usage)
                is not StrictConvexCandidateResourceUsageV2
            ):
                raise TypeError("verification_resource_usage has the wrong exact type")
            object.__setattr__(
                self,
                "verification_resource_usage",
                StrictConvexCandidateResourceUsageV2(
                    self.verification_resource_usage.domain_operations,
                    self.verification_resource_usage.so2_atomic_steps,
                ),
            )


class StrictConvexCandidateDomainCompilerV2_5:
    """Compile the bounded one-pair continuous-yaw collision prefix."""

    def compile(
        self,
        problem: SemanticProblemV2_2,
        config: StrictConvexCandidateCompilerConfigV2_5,
    ) -> StrictConvexCandidateCompilationOutcomeV2:
        return compile_strict_convex_candidate_domain_v2_5(problem, config)


class _InvalidInputV2(ValueError):
    pass


class _UnsupportedModelV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _ResourceLimitErrorV2(RuntimeError):
    pass


@dataclass(slots=True)
class _DomainOperationBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = _ResourceLimitErrorV2


def compile_strict_convex_candidate_domain_v2_5(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_5,
) -> StrictConvexCandidateCompilationOutcomeV2:
    """Fresh-compile a collision prefix; remaining hard constraints stay explicit."""

    try:
        checked_config = _strict_config(config)
    except _InvalidInputV2:
        return _failure(
            StrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:STRICT_CONVEX_CANDIDATE_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_CANDIDATE_REVALIDATION",
        )

    budget = _DomainOperationBudgetV2(checked_config.max_domain_operations)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _precharge_problem_structure(problem, budget)
            checked_problem = _strict_problem(problem)
    except _ResourceLimitErrorV2:
        return _failure(
            StrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:max_domain_operations",
        )
    except _InvalidInputV2:
        return _failure(
            StrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:STRICT_CONVEX_CANDIDATE_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_CANDIDATE_REVALIDATION",
        )
    atomic_budget = SO2AtomicBudgetV2(limit=checked_config.max_so2_atomic_steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            from spatialcf.core.v2.candidate_domain import (
                _compile_search_universe,
                _SearchUniverseFailureV2,
            )

            search = _compile_search_universe(checked_problem, budget)
            if isinstance(search, _SearchUniverseFailureV2):
                kind = (
                    StrictConvexCandidateCompilationKindV2.NUMERIC_GAP
                    if search.uncertified_reason.value == "NUMERIC_GAP"
                    else StrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL
                )
                return StrictConvexCandidateCompilationOutcomeV2(
                    kind=kind,
                    finding_codes=search.finding_codes,
                )
            subject_transform, subject_shape, obstacle_transform, obstacle_shape = (
                _extract_supported_pair(checked_problem, budget)
            )
            allowed = compile_convex_allowed_translation_v2(
                subject_transform,
                subject_shape,
                obstacle_transform,
                obstacle_shape,
                search.delta_rect,
                atomic_budget=atomic_budget,
            )
            if allowed.kind is ConvexAllowedTranslationKindV2.RESOURCE_LIMIT:
                return _failure(
                    StrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
                    "RESOURCE_LIMIT:max_so2_atomic_steps",
                )
            if allowed.kind is ConvexAllowedTranslationKindV2.NUMERIC_GAP:
                return StrictConvexCandidateCompilationOutcomeV2(
                    kind=StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
                    finding_codes=allowed.finding_codes,
                )
            if allowed.kind is ConvexAllowedTranslationKindV2.INVALID_INPUT:
                raise RuntimeError("strict supported pair produced invalid T12 input")
            if (
                allowed.kind is not ConvexAllowedTranslationKindV2.BRACKET
                or type(allowed.bracket) is not ConvexAllowedTranslationBracketV2
            ):
                raise RuntimeError("malformed T12 allowed-domain outcome")
            constraints = checked_problem.constraints
            budget.consume(
                len(constraints.support_constraints)
                + len(constraints.visibility_constraints)
                + 1
            )
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
            artifact = StrictConvexCandidateDomainArtifactV2_2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_config.config_sha256,
                subject_id=constraints.allowed_edit.subject_id,
                search_universe=search.delta_rect,
                ordered_constraint_ids=(
                    constraints.position_domain.constraint_id,
                    constraints.collision_constraints[0].constraint_id,
                ),
                allowed_domain_bracket=allowed.bracket,
                resource_usage=StrictConvexCandidateResourceUsageV2(
                    domain_operations=budget.used,
                    so2_atomic_steps=atomic_budget.used,
                ),
                remaining_constraint_ids=remaining_ids,
            )
            return StrictConvexCandidateCompilationOutcomeV2(
                kind=StrictConvexCandidateCompilationKindV2.ARTIFACT,
                artifact=artifact,
            )
    except _UnsupportedModelV2 as error:
        return _failure(
            StrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except _ResourceLimitErrorV2:
        return _failure(
            StrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:max_domain_operations",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_CANDIDATE_COMPILATION",
        )


def verify_strict_convex_candidate_domain_v2_5(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_5,
    submitted_artifact: StrictConvexCandidateDomainArtifactV2_2,
) -> StrictConvexCandidateVerificationOutcomeV2:
    """Fresh replay raw inputs and compare the entire submitted prefix artifact."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_submitted = _copy_artifact(submitted_artifact)
    except (ArithmeticError, RuntimeWarning):
        return StrictConvexCandidateVerificationOutcomeV2(
            kind=StrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUBMITTED_CANDIDATE_ARTIFACT",),
        )
    except (AttributeError, TypeError, ValueError, Warning):
        return StrictConvexCandidateVerificationOutcomeV2(
            kind=StrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("INVALID_INPUT:SUBMITTED_CANDIDATE_ARTIFACT",),
        )

    replay = compile_strict_convex_candidate_domain_v2_5(problem, config)
    if (
        replay.kind is not StrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(replay.artifact) is not StrictConvexCandidateDomainArtifactV2_2
    ):
        return StrictConvexCandidateVerificationOutcomeV2(
            kind=StrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=replay.finding_codes,
        )
    fresh = replay.artifact
    usage = fresh.resource_usage
    if (
        checked_submitted != fresh
        or _artifact_bytes(checked_submitted) != _artifact_bytes(fresh)
        or checked_submitted.artifact_sha256 != fresh.artifact_sha256
    ):
        return StrictConvexCandidateVerificationOutcomeV2(
            kind=StrictConvexCandidateVerificationKindV2.MISMATCH,
            verification_resource_usage=usage,
            finding_codes=("MISMATCH:STRICT_CONVEX_CANDIDATE_ARTIFACT",),
        )
    return StrictConvexCandidateVerificationOutcomeV2(
        kind=StrictConvexCandidateVerificationKindV2.VERIFIED,
        semantic_problem_sha256=fresh.semantic_problem_sha256,
        compiler_config_sha256=fresh.compiler_config_sha256,
        artifact_sha256=fresh.artifact_sha256,
        verification_resource_usage=usage,
    )


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


def _precharge_problem_structure(
    value: object,
    budget: _DomainOperationBudgetV2,
) -> None:
    if type(value) is not SemanticProblemV2_2:
        raise _InvalidInputV2
    try:
        scene = value.scene
        constraints = value.constraints
        objective = value.objective
        relation_semantics = value.relation_semantics
        visibility_semantics = value.visibility_semantics
        families = (
            scene.objects,
            scene.geometry_instances,
            scene.collision_bodies,
            scene.workspace_boundaries,
            scene.known_free_spaces,
            scene.support_surfaces,
            scene.cameras,
            scene.baseline_observations,
        )
        family_values = tuple(
            values
            for facts in families
            for values in (facts.values, facts.inner_values, facts.outer_values)
            if values is not None
        )
        if any(type(values) is not tuple for values in family_values):
            raise _InvalidInputV2
        budget.consume(
            1
            + sum(len(values) for values in family_values)
            + len(constraints.collision_constraints)
            + len(constraints.support_constraints)
            + len(constraints.visibility_constraints)
            + len(relation_semantics.definitions)
            + len(visibility_semantics.definitions)
            + len(objective.relation_damage.pair_axis_weights)
            + len(objective.visibility_change.object_camera_weights)
            + len(objective.safety_margin.aggregation.targets)
        )
        for facts in (scene.workspace_boundaries, scene.known_free_spaces):
            for item in facts.values or ():
                _precharge_region(item.region_world_xy, budget)
        for item in scene.support_surfaces.values or ():
            _precharge_region(item.region_uv, budget)
        for item in scene.geometry_instances.values or ():
            shape = item.shape
            if hasattr(shape, "footprint"):
                budget.consume()
                _precharge_component(shape.footprint, budget)
    except _ResourceLimitErrorV2:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise _InvalidInputV2 from error


def _precharge_region(value: Any, budget: _DomainOperationBudgetV2) -> None:
    components = value.components
    if type(components) is not tuple:
        raise _InvalidInputV2
    budget.consume(len(components) + 1)
    for component in components:
        _precharge_component(component, budget)


def _precharge_component(value: Any, budget: _DomainOperationBudgetV2) -> None:
    holes = value.holes
    if type(holes) is not tuple:
        raise _InvalidInputV2
    budget.consume(len(value.exterior.vertices) + len(holes) + 1)
    for hole in holes:
        budget.consume(len(hole.vertices) + 1)


def _strict_config(value: object) -> StrictConvexCandidateCompilerConfigV2_5:
    if type(value) is not StrictConvexCandidateCompilerConfigV2_5:
        raise _InvalidInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            payload = value.model_dump(mode="python", warnings="error")
            return StrictConvexCandidateCompilerConfigV2_5.model_validate(
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


def _extract_supported_pair(
    problem: SemanticProblemV2_2,
    budget: _DomainOperationBudgetV2,
) -> tuple[
    DirectedYawIntervalTransformV2_2,
    UprightBox3DV2,
    DirectedYawIntervalTransformV2_2,
    UprightBox3DV2,
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
    if len(collision.subject_body_ids) != 1 or len(collision.obstacle_body_ids) != 1:
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_PAIR_CARDINALITY")
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
    budget.consume(len(bodies) + len(geometries) + len(objects))

    subject_id = constraints.allowed_edit.subject_id
    subject_body = body_by_id[collision.subject_body_ids[0]]
    obstacle_body = body_by_id[collision.obstacle_body_ids[0]]
    if (
        subject_body.owner_object_id != subject_id
        or obstacle_body.owner_object_id is None
        or obstacle_body.owner_object_id == subject_id
        or len(subject_body.geometry_instance_ids) != 1
        or len(obstacle_body.geometry_instance_ids) != 1
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:COLLISION_BODY_SUBSET")
    try:
        subject_object = object_by_id[subject_id]
        obstacle_object = object_by_id[obstacle_body.owner_object_id]
        subject_geometry = geometry_by_id[subject_body.geometry_instance_ids[0]]
        obstacle_geometry = geometry_by_id[obstacle_body.geometry_instance_ids[0]]
    except KeyError as error:
        raise RuntimeError(
            "strict semantic graph lost a collision reference"
        ) from error
    budget.consume(4)
    if not subject_object.movable or obstacle_object.movable:
        raise _UnsupportedModelV2("UNSUPPORTED:FIXED_OBSTACLE_SUBSET")
    for geometry, owner_id in (
        (subject_geometry, subject_id),
        (obstacle_geometry, obstacle_body.owner_object_id),
    ):
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
    subject_transform = subject_object.pose.world_from_object
    obstacle_transform = obstacle_object.pose.world_from_object
    if (
        type(subject_transform) is not DirectedYawIntervalTransformV2_2
        or type(obstacle_transform) is not DirectedYawIntervalTransformV2_2
    ):
        raise RuntimeError("v2.2 object pose lost its directed-yaw transform")
    subject_shape = subject_geometry.shape
    obstacle_shape = obstacle_geometry.shape
    assert type(subject_shape) is UprightBox3DV2
    assert type(obstacle_shape) is UprightBox3DV2
    if not _z_interiors_overlap(
        subject_transform, subject_shape, obstacle_transform, obstacle_shape
    ):
        raise _UnsupportedModelV2("UNSUPPORTED:Z_SEPARATED_COLLISION_PAIR")
    return subject_transform, subject_shape, obstacle_transform, obstacle_shape


def _exact_fact_values(
    facts: Any,
    label: str,
    budget: _DomainOperationBudgetV2,
) -> tuple[Any, ...]:
    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.uncertainty != UncertaintyBudgetV2()
        or type(facts.values) is not tuple
    ):
        raise _UnsupportedModelV2(f"UNSUPPORTED:{label}_FACT_SET")
    budget.consume(len(facts.values) + 1)
    return facts.values


def _is_identity_anchor(transform: DirectedYawIntervalTransformV2_2) -> bool:
    return (
        type(transform) is DirectedYawIntervalTransformV2_2
        and transform.translation == Vec3V2(x=0.0, y=0.0, z=0.0)
        and transform.yaw_radians == 0.0
    )


def _z_interiors_overlap(
    subject_transform: DirectedYawIntervalTransformV2_2,
    subject_shape: UprightBox3DV2,
    obstacle_transform: DirectedYawIntervalTransformV2_2,
    obstacle_shape: UprightBox3DV2,
) -> bool:
    subject_z = Fraction.from_float(subject_transform.translation.z)
    obstacle_z = Fraction.from_float(obstacle_transform.translation.z)
    subject_half = Fraction.from_float(subject_shape.size_m.z) / 2
    obstacle_half = Fraction.from_float(obstacle_shape.size_m.z) / 2
    return max(subject_z - subject_half, obstacle_z - obstacle_half) < min(
        subject_z + subject_half, obstacle_z + obstacle_half
    )


def _failure(
    kind: StrictConvexCandidateCompilationKindV2,
    finding_code: str,
) -> StrictConvexCandidateCompilationOutcomeV2:
    return StrictConvexCandidateCompilationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


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


def _copy_bracket(
    value: ConvexAllowedTranslationBracketV2,
) -> ConvexAllowedTranslationBracketV2:
    if type(value) is not ConvexAllowedTranslationBracketV2:
        raise TypeError("allowed_domain_bracket has the wrong exact type")
    return ConvexAllowedTranslationBracketV2(
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        obstacle_kernel_id=value.obstacle_kernel_id,
        obstacle_kernel_version=value.obstacle_kernel_version,
        partition_kernel_id=value.partition_kernel_id,
        partition_kernel_version=value.partition_kernel_version,
        atomic_steps_used=value.atomic_steps_used,
    )


def _copy_artifact(
    value: StrictConvexCandidateDomainArtifactV2_2,
) -> StrictConvexCandidateDomainArtifactV2_2:
    return StrictConvexCandidateDomainArtifactV2_2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        subject_id=value.subject_id,
        search_universe=value.search_universe,
        ordered_constraint_ids=value.ordered_constraint_ids,
        allowed_domain_bracket=value.allowed_domain_bracket,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
    )


def _artifact_bytes(value: StrictConvexCandidateDomainArtifactV2_2) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"denominator": value.denominator, "numerator": value.numerator}
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


__all__ = (
    "StrictConvexCandidateCompilationKindV2",
    "StrictConvexCandidateCompilationOutcomeV2",
    "StrictConvexCandidateDomainArtifactV2_2",
    "StrictConvexCandidateDomainCompilerV2_5",
    "StrictConvexCandidateResourceUsageV2",
    "StrictConvexCandidateVerificationKindV2",
    "StrictConvexCandidateVerificationOutcomeV2",
    "compile_strict_convex_candidate_domain_v2_5",
    "verify_strict_convex_candidate_domain_v2_5",
)
