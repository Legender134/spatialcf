"""Directed six-way target compilation for the competition v2.9 solver."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    UprightCameraContextV2_9,
    bound_world_point_in_upright_camera_v2_9,
    prepare_camera_independent_candidate_problem_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _compile_target_aware_candidate_v2,
    _TargetAwareCandidateKindV2,
)
from spatialcf.core.v2.convex_translation_domain import (
    RationalConvexPolygonV2,
)
from spatialcf.core.v2.convex_translation_partition import (
    RationalHalfPlane2V2,
    RationalHalfPlaneRelationV2,
    _canonical_half_plane_v2,
)
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _require_finding_codes,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionTopologyV2,
    _clip_universe_by_planes,
    _find_strict_witness,
    _universe_planes,
)
from spatialcf.core.v2.support_strict_convex_candidate_domain import (
    SupportStrictConvexCandidateDomainArtifactV2_2,
)
from spatialcf.core.v2.support_strict_convex_candidate_domain import (
    _copy_artifact as _copy_t15_artifact,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    MeasurementComparatorV2,
    RelationMeasurementV2,
    RelationV2,
)
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.directional-target-stage.v2.9\0"
_IntervalV2 = tuple[Fraction, Fraction]


class DirectionalTargetCandidateKindV2_9(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, slots=True)
class DirectionalTargetCandidateStageV2_9:
    semantic_problem_sha256: str
    candidate_problem_sha256: str
    camera_context_sha256: str
    compiler_config_sha256: str
    upstream_t15_artifact_sha256: str
    subject_id: str
    target_constraint_id: str
    target_relation: RelationV2
    target_threshold: Fraction
    requires_both_visible: bool
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("candidate_problem_sha256", self.candidate_problem_sha256),
            ("camera_context_sha256", self.camera_context_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t15_artifact_sha256", self.upstream_t15_artifact_sha256),
        ):
            _require_digest(label, digest)
        for label, value in (
            ("subject_id", self.subject_id),
            ("target_constraint_id", self.target_constraint_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        if type(self.target_relation) is not RelationV2:
            raise TypeError("target_relation must be an exact RelationV2")
        if type(self.target_threshold) is not Fraction:
            raise TypeError("target_threshold must be an exact Fraction")
        if type(self.requires_both_visible) is not bool:
            raise TypeError("requires_both_visible must be an exact bool")
        inner = _copy_intersection_complex(self.inner_allowed)
        outer = _copy_intersection_complex(self.outer_allowed)
        if inner.universe != outer.universe:
            raise ValueError("directional target requires one exact universe")
        if not all(outer.contains_point(cell.strict_witness) for cell in inner.cells):
            raise ValueError("directional target inner witness escaped its outer")
        usage = _copy_usage(self.resource_usage)
        remaining = _canonical_ids(self.remaining_constraint_ids)
        if self.target_constraint_id in remaining:
            raise ValueError("compiled target remained in the constraint suffix")
        object.__setattr__(self, "inner_allowed", inner)
        object.__setattr__(self, "outer_allowed", outer)
        object.__setattr__(self, "resource_usage", usage)
        object.__setattr__(self, "remaining_constraint_ids", remaining)

    @property
    def stage_sha256(self) -> str:
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2_9 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DirectionalTargetCandidateOutcomeV2_9:
    kind: DirectionalTargetCandidateKindV2_9
    stage: DirectionalTargetCandidateStageV2_9 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not DirectionalTargetCandidateKindV2_9:
            raise TypeError("kind must be an exact directional target kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is DirectionalTargetCandidateKindV2_9.STAGE:
            if type(self.stage) is not DirectionalTargetCandidateStageV2_9:
                raise ValueError("STAGE requires one exact directional stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("directional target failure requires findings only")


@dataclass(frozen=True, slots=True)
class _AffineIntervalV2:
    x: _IntervalV2
    y: _IntervalV2
    constant: _IntervalV2

    def __post_init__(self) -> None:
        for value in (self.x, self.y, self.constant):
            _require_interval(value)


class _UnsupportedDirectionalTargetV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def compile_directional_target_candidate_v2_9(
    problem: SemanticProblemV2_3,
    projected_problem: SemanticProblemV2_2,
    t15_artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
    camera: UprightCameraContextV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> DirectionalTargetCandidateOutcomeV2_9:
    """Compile all six target directions on one continued v2.9 ledger."""

    try:
        checked = _strict_inputs(
            problem,
            projected_problem,
            t15_artifact,
            camera,
            atomic_budget,
            intersection_budget,
        )
        original, projected, artifact, context = checked
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            target, definition, subject, reference = _extract_target(
                original, projected, context, intersection_budget
            )
            if target.relation_after in (RelationV2.NEAR, RelationV2.FAR):
                inner, outer = _compile_shape_gap_on_continued_ledgers(
                    projected,
                    artifact,
                    atomic_budget,
                    intersection_budget,
                )
            else:
                affine = _directional_affine(
                    target.relation_after,
                    definition.measurement,
                    Fraction.from_float(definition.threshold),
                    subject,
                    reference,
                    context,
                    artifact,
                    atomic_budget,
                )
                inner = _clip_bracket_complex(
                    artifact.allowed_domain_bracket.inner_allowed,
                    affine,
                    definition.comparator,
                    Fraction.from_float(definition.threshold)
                    if definition.measurement
                    is RelationMeasurementV2.CAMERA_DEPTH_DELTA
                    else Fraction(),
                    inner=True,
                    atomic_budget=atomic_budget,
                    domain_budget=intersection_budget,
                )
                outer = _clip_bracket_complex(
                    artifact.allowed_domain_bracket.outer_allowed,
                    affine,
                    definition.comparator,
                    Fraction.from_float(definition.threshold)
                    if definition.measurement
                    is RelationMeasurementV2.CAMERA_DEPTH_DELTA
                    else Fraction(),
                    inner=False,
                    atomic_budget=atomic_budget,
                    domain_budget=intersection_budget,
                )
            remaining = tuple(
                item
                for item in artifact.remaining_constraint_ids
                if item != target.constraint_id
            )
            intersection_budget.consume_domain(
                12
                + len(remaining)
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (inner, outer)
                    for cell in complex_.cells
                )
            )
            stage = DirectionalTargetCandidateStageV2_9(
                semantic_problem_sha256=original.semantic_problem_sha256,
                candidate_problem_sha256=projected.semantic_problem_sha256,
                camera_context_sha256=context.context_sha256,
                compiler_config_sha256=artifact.compiler_config_sha256,
                upstream_t15_artifact_sha256=artifact.artifact_sha256,
                subject_id=artifact.subject_id,
                target_constraint_id=target.constraint_id,
                target_relation=target.relation_after,
                target_threshold=Fraction.from_float(definition.threshold),
                requires_both_visible=definition.requires_both_visible,
                inner_allowed=inner,
                outer_allowed=outer,
                resource_usage=_usage(atomic_budget, intersection_budget),
                remaining_constraint_ids=remaining,
            )
        return DirectionalTargetCandidateOutcomeV2_9(
            kind=DirectionalTargetCandidateKindV2_9.STAGE,
            stage=stage,
        )
    except _UnsupportedDirectionalTargetV2 as error:
        return _failure(
            DirectionalTargetCandidateKindV2_9.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            DirectionalTargetCandidateKindV2_9.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_DIRECTIONAL_TARGET",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            DirectionalTargetCandidateKindV2_9.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_DIRECTIONAL_TARGET",
        )


def _strict_inputs(
    problem: object,
    projected_problem: object,
    artifact: object,
    camera: object,
    atomic_budget: object,
    intersection_budget: object,
) -> tuple[
    SemanticProblemV2_3,
    SemanticProblemV2_2,
    SupportStrictConvexCandidateDomainArtifactV2_2,
    UprightCameraContextV2_9,
]:
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(projected_problem) is not SemanticProblemV2_2:
        raise TypeError("projected_problem must be an exact SemanticProblemV2_2")
    if type(artifact) is not SupportStrictConvexCandidateDomainArtifactV2_2:
        raise TypeError("t15_artifact must be the exact T15 artifact type")
    if type(camera) is not UprightCameraContextV2_9:
        raise TypeError("camera must be an exact UprightCameraContextV2_9")
    intersection_budget.consume_domain(16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        original = SemanticProblemV2_3.model_validate(
            problem.model_dump(mode="python", warnings="error"), strict=True
        )
        projected = SemanticProblemV2_2.model_validate(
            projected_problem.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_artifact = _copy_t15_artifact(artifact)
        context = UprightCameraContextV2_9(
            camera_id=camera.camera_id,
            width_px=camera.width_px,
            height_px=camera.height_px,
            intrinsics=camera.intrinsics,
            near_clip_m=camera.near_clip_m,
            far_clip_m=camera.far_clip_m,
            translation_xyz=camera.translation_xyz,
            sine=camera.sine,
            cosine=camera.cosine,
        )
        replayed = prepare_camera_independent_candidate_problem_v2_9(original)
    if replayed != projected:
        raise ValueError("projected candidate problem does not match the original")
    if checked_artifact.semantic_problem_sha256 != projected.semantic_problem_sha256:
        raise ValueError("T15 artifact does not close the projected problem")
    usage = checked_artifact.resource_usage
    if (
        atomic_budget.used < usage.so2_atomic_steps
        or intersection_budget.domain_operations_used < usage.domain_operations
        or intersection_budget.candidate_cells_used < usage.candidate_cells
    ):
        raise ValueError("live directional ledgers rolled back below T15 usage")
    if original.constraints.target_relation.constraint_id not in (
        checked_artifact.remaining_constraint_ids
    ):
        raise ValueError("T15 artifact does not expose the target relation")
    return original, projected, checked_artifact, context


def _extract_target(
    original: SemanticProblemV2_3,
    projected: SemanticProblemV2_2,
    camera: UprightCameraContextV2_9,
    budget: StrictConvexIntersectionBudgetV2,
):
    target = original.constraints.target_relation
    definition = next(
        item
        for item in original.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    expected_measurement = {
        RelationV2.LEFT: RelationMeasurementV2.PROJECTED_CENTER_DELTA_X,
        RelationV2.RIGHT: RelationMeasurementV2.PROJECTED_CENTER_DELTA_X,
        RelationV2.FRONT: RelationMeasurementV2.CAMERA_DEPTH_DELTA,
        RelationV2.BEHIND: RelationMeasurementV2.CAMERA_DEPTH_DELTA,
        RelationV2.NEAR: RelationMeasurementV2.SHAPE_GAP_XY,
        RelationV2.FAR: RelationMeasurementV2.SHAPE_GAP_XY,
    }[target.relation_after]
    expected_comparator = (
        MeasurementComparatorV2.LESS_THAN
        if target.relation_after in (RelationV2.LEFT, RelationV2.FRONT, RelationV2.NEAR)
        else MeasurementComparatorV2.GREATER_THAN
    )
    if (
        definition.measurement is not expected_measurement
        or definition.comparator is not expected_comparator
        or definition.boundary_policy is not BoundaryPolicyV2.CLOSED
        or definition.tolerance != 0.0
        or original.numeric_policy != NumericPolicyV2()
        or target.camera_id != camera.camera_id
    ):
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_POLICY"
        )
    for label, facts in (
        ("OBJECTS", projected.scene.objects),
        ("GEOMETRIES", projected.scene.geometry_instances),
    ):
        if (
            facts.availability is not FactAvailabilityV2.KNOWN
            or facts.completeness is not FactCompletenessV2.EXACT
            or facts.uncertainty != UncertaintyBudgetV2()
            or type(facts.values) is not tuple
        ):
            raise _UnsupportedDirectionalTargetV2(
                f"UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_{label}"
            )
    objects = {item.object_id: item for item in projected.scene.objects.values or ()}
    geometries = tuple(
        item
        for item in projected.scene.geometry_instances.values or ()
        if item.role is GeometryRoleV2.RELATION
        and item.owner_object_id in (target.subject_id, target.reference_id)
    )
    budget.consume_domain(len(objects) + len(geometries) + 4)
    try:
        subject_object = objects[target.subject_id]
        reference_object = objects[target.reference_id]
    except KeyError as error:
        raise RuntimeError(
            "directional target semantic graph lost an object"
        ) from error
    by_owner: dict[str, list[GeometryInstanceV2_2]] = {}
    for geometry in geometries:
        by_owner.setdefault(geometry.owner_object_id or "", []).append(geometry)
    if any(
        len(by_owner.get(owner_id, ())) != 1
        for owner_id in (target.subject_id, target.reference_id)
    ):
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_GEOMETRY_CARDINALITY"
        )
    centroids = []
    for item, geometry in (
        (subject_object, by_owner[target.subject_id][0]),
        (reference_object, by_owner[target.reference_id][0]),
    ):
        anchor = geometry.anchor_from_geometry
        if (
            type(geometry) is not GeometryInstanceV2_2
            or geometry.approximation is not GeometryApproximationV2.EXACT
            or geometry.uncertainty != UncertaintyBudgetV2()
            or type(geometry.shape) is not UprightBox3DV2
            or (anchor.translation.x, anchor.translation.y) != (0.0, 0.0)
        ):
            raise _UnsupportedDirectionalTargetV2(
                "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_GEOMETRY"
            )
        centroids.append(
            (
                Fraction.from_float(item.pose.world_from_object.translation.x),
                Fraction.from_float(item.pose.world_from_object.translation.y),
                Fraction.from_float(
                    item.pose.world_from_object.translation.z + anchor.translation.z
                ),
            )
        )
    if not subject_object.movable or reference_object.movable:
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_REFERENCE"
        )
    return target, definition, centroids[0], centroids[1]


def _directional_affine(
    relation: RelationV2,
    measurement: RelationMeasurementV2,
    threshold: Fraction,
    subject: tuple[Fraction, Fraction, Fraction],
    reference: tuple[Fraction, Fraction, Fraction],
    camera: UprightCameraContextV2_9,
    artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
    budget: SO2AtomicBudgetV2,
) -> _AffineIntervalV2:
    universe_bounds = artifact.search_universe.bounds
    if universe_bounds is None:
        raise RuntimeError("T15 search universe cannot be empty")
    if measurement is RelationMeasurementV2.CAMERA_DEPTH_DELTA:
        # Camera depth is affine over world XY for every upright azimuth.  It
        # remains meaningful outside the positive-depth half-space; the
        # following visibility stage independently enforces near/far clipping
        # for relations whose policy requires both endpoints visible.
        base_x = subject[0] - reference[0]
        base_y = subject[1] - reference[1]
        constant = _add(
            _scale(camera.sine, base_x, budget),
            _scale(camera.cosine, base_y, budget),
            budget,
        )
        return _AffineIntervalV2(
            x=camera.sine,
            y=camera.cosine,
            constant=constant,
        )
    subject_bounds = bound_world_point_in_upright_camera_v2_9(
        camera,
        world_xyz=subject,
        delta_x=(universe_bounds[0], universe_bounds[2]),
        delta_y=(universe_bounds[1], universe_bounds[3]),
        atomic_budget=budget,
    )
    reference_bounds = bound_world_point_in_upright_camera_v2_9(
        camera,
        world_xyz=reference,
        delta_x=(Fraction(), Fraction()),
        delta_y=(Fraction(), Fraction()),
        atomic_budget=budget,
    )
    if not subject_bounds.positive_depth or not reference_bounds.positive_depth:
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_DEPTH"
        )
    if measurement is not RelationMeasurementV2.PROJECTED_CENTER_DELTA_X:
        raise RuntimeError("directional target escaped its measurement partition")
    fx = camera.intrinsics[0]
    if fx <= 0:
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_INTRINSICS"
        )
    reference_ratio = _divide_positive(
        reference_bounds.x_camera, reference_bounds.z_camera, budget
    )
    k = _add(
        reference_ratio,
        (threshold / fx, threshold / fx),
        budget,
    )
    x_coefficient = _subtract(
        camera.cosine,
        _multiply(k, camera.sine, budget),
        budget,
    )
    y_coefficient = _subtract(
        _negate(camera.sine, budget),
        _multiply(k, camera.cosine, budget),
        budget,
    )
    subject_x0 = _add(
        _subtract(
            _scale(camera.cosine, subject[0], budget),
            _scale(camera.sine, subject[1], budget),
            budget,
        ),
        (camera.translation_xyz[0], camera.translation_xyz[0]),
        budget,
    )
    subject_z0 = _add(
        _add(
            _scale(camera.sine, subject[0], budget),
            _scale(camera.cosine, subject[1], budget),
            budget,
        ),
        (camera.translation_xyz[2], camera.translation_xyz[2]),
        budget,
    )
    constant = _subtract(subject_x0, _multiply(k, subject_z0, budget), budget)
    if relation not in (RelationV2.LEFT, RelationV2.RIGHT):
        raise RuntimeError("horizontal affine received a non-horizontal relation")
    return _AffineIntervalV2(
        x=x_coefficient,
        y=y_coefficient,
        constant=constant,
    )


def _clip_bracket_complex(
    upstream: StrictConvexIntersectionComplexV2,
    affine: _AffineIntervalV2,
    comparator: MeasurementComparatorV2,
    threshold: Fraction,
    *,
    inner: bool,
    atomic_budget: SO2AtomicBudgetV2,
    domain_budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    universe = upstream.universe
    sign_cells = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    cells: list[StrictConvexIntersectionCellV2] = []
    for upstream_cell in upstream.cells:
        for x_nonnegative, y_nonnegative in sign_cells:
            sign_planes = (
                _canonical_half_plane_v2(
                    Fraction(-1) if x_nonnegative else Fraction(1),
                    Fraction(),
                    Fraction(),
                    RationalHalfPlaneRelationV2.LE
                    if x_nonnegative
                    else RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
                _canonical_half_plane_v2(
                    Fraction(),
                    Fraction(-1) if y_nonnegative else Fraction(1),
                    Fraction(),
                    RationalHalfPlaneRelationV2.LE
                    if y_nonnegative
                    else RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
            )
            lower_x, upper_x = _coefficient_for_sign(affine.x, x_nonnegative)
            lower_y, upper_y = _coefficient_for_sign(affine.y, y_nonnegative)
            if comparator is MeasurementComparatorV2.LESS_THAN:
                coefficients = (
                    (upper_x, upper_y, affine.constant[1])
                    if inner
                    else (lower_x, lower_y, affine.constant[0])
                )
                target_plane = _canonical_half_plane_v2(
                    coefficients[0],
                    coefficients[1],
                    threshold - coefficients[2],
                    RationalHalfPlaneRelationV2.LE,
                    atomic_budget,
                )
            elif comparator is MeasurementComparatorV2.GREATER_THAN:
                coefficients = (
                    (lower_x, lower_y, affine.constant[0])
                    if inner
                    else (upper_x, upper_y, affine.constant[1])
                )
                target_plane = _canonical_half_plane_v2(
                    -coefficients[0],
                    -coefficients[1],
                    coefficients[2] - threshold,
                    RationalHalfPlaneRelationV2.LE,
                    atomic_budget,
                )
            else:
                raise RuntimeError("unsupported relation comparator escaped extraction")
            semantic = _canonical_planes(
                (*upstream_cell.half_planes[4:], *sign_planes, target_plane)
            )
            closure = _clip_universe_by_planes(universe, semantic, domain_budget)
            if closure is None:
                continue
            half_planes = _universe_planes(universe) + semantic
            witness = _find_strict_witness(closure, half_planes, domain_budget)
            if witness is None:
                continue
            domain_budget.consume_candidate_cells()
            cells.append(
                StrictConvexIntersectionCellV2(
                    cell_id=(
                        f"cell:directional-target:{'inner' if inner else 'outer'}:"
                        f"{len(cells):06d}"
                    ),
                    half_planes=half_planes,
                    closure_polygon=RationalConvexPolygonV2(vertices_ccw=closure),
                    strict_witness=witness,
                )
            )
    return StrictConvexIntersectionComplexV2(
        cells=tuple(cells),
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _compile_shape_gap_on_continued_ledgers(
    problem: SemanticProblemV2_2,
    artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
    atomic_budget: SO2AtomicBudgetV2,
    domain_budget: StrictConvexIntersectionBudgetV2,
) -> tuple[StrictConvexIntersectionComplexV2, StrictConvexIntersectionComplexV2]:
    base = artifact.resource_usage
    shadow_atomic = SO2AtomicBudgetV2(
        limit=base.so2_atomic_steps + (atomic_budget.limit - atomic_budget.used),
        used=base.so2_atomic_steps,
    )
    shadow_domain = StrictConvexIntersectionBudgetV2(
        max_domain_operations=(
            base.domain_operations
            + (
                domain_budget.max_domain_operations
                - domain_budget.domain_operations_used
            )
        ),
        max_candidate_cells=(
            base.candidate_cells
            + (domain_budget.max_candidate_cells - domain_budget.candidate_cells_used)
        ),
        domain_operations_used=base.domain_operations,
        candidate_cells_used=base.candidate_cells,
    )
    outcome = _compile_target_aware_candidate_v2(
        problem,
        artifact,
        atomic_budget=shadow_atomic,
        intersection_budget=shadow_domain,
    )
    atomic_delta = shadow_atomic.used - base.so2_atomic_steps
    domain_delta = shadow_domain.domain_operations_used - base.domain_operations
    cell_delta = shadow_domain.candidate_cells_used - base.candidate_cells
    atomic_budget.consume(atomic_delta)
    domain_budget.consume_domain(domain_delta)
    domain_budget.consume_candidate_cells(cell_delta)
    if outcome.kind is _TargetAwareCandidateKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is _TargetAwareCandidateKindV2.NUMERIC_GAP:
        raise ArithmeticError("shape-gap target replay reached a numeric gap")
    if outcome.kind is _TargetAwareCandidateKindV2.UNSUPPORTED_MODEL:
        raise _UnsupportedDirectionalTargetV2(outcome.finding_codes[0])
    if outcome.kind is not _TargetAwareCandidateKindV2.STAGE or outcome.stage is None:
        raise RuntimeError("shape-gap target replay returned an invalid outcome")
    return outcome.stage.inner_allowed, outcome.stage.outer_allowed


def _coefficient_for_sign(
    interval: _IntervalV2, nonnegative: bool
) -> tuple[Fraction, Fraction]:
    return interval if nonnegative else (interval[1], interval[0])


def _canonical_planes(
    planes: tuple[RationalHalfPlane2V2, ...],
) -> tuple[RationalHalfPlane2V2, ...]:
    unique = set(planes)
    return tuple(
        sorted(
            unique,
            key=lambda plane: (
                plane.normal_x,
                plane.normal_y,
                plane.offset,
                0 if plane.relation is RationalHalfPlaneRelationV2.LT else 1,
            ),
        )
    )


def _scale(
    interval: _IntervalV2, scalar: Fraction, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    return _multiply(interval, (scalar, scalar), budget)


def _multiply(
    left: _IntervalV2, right: _IntervalV2, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    products = tuple(a * b for a in left for b in right)
    result = (min(products), max(products))
    _require_interval(result)
    return result


def _add(
    left: _IntervalV2, right: _IntervalV2, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    result = (left[0] + right[0], left[1] + right[1])
    _require_interval(result)
    return result


def _subtract(
    left: _IntervalV2, right: _IntervalV2, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    result = (left[0] - right[1], left[1] - right[0])
    _require_interval(result)
    return result


def _negate(interval: _IntervalV2, budget: SO2AtomicBudgetV2) -> _IntervalV2:
    budget.consume()
    result = (-interval[1], -interval[0])
    _require_interval(result)
    return result


def _divide_positive(
    numerator: _IntervalV2,
    denominator: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    if denominator[0] <= 0:
        raise _UnsupportedDirectionalTargetV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_DIRECTIONAL_TARGET_DEPTH"
        )
    budget.consume()
    reciprocal = (Fraction(1, 1) / denominator[1], Fraction(1, 1) / denominator[0])
    _require_interval(reciprocal)
    return _multiply(numerator, reciprocal, budget)


def _require_interval(value: _IntervalV2) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("directed relation interval must be an exact pair")
    if any(type(item) is not Fraction for item in value):
        raise TypeError("directed relation interval endpoints must be Fractions")
    if value[0] > value[1]:
        raise ValueError("directed relation interval endpoints are reversed")


def _usage(
    atomic_budget: SO2AtomicBudgetV2,
    domain_budget: StrictConvexIntersectionBudgetV2,
) -> MultiObstacleStrictConvexCandidateResourceUsageV2:
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=domain_budget.domain_operations_used,
        so2_atomic_steps=atomic_budget.used,
        candidate_cells=domain_budget.candidate_cells_used,
    )


def _copy_usage(
    usage: MultiObstacleStrictConvexCandidateResourceUsageV2,
) -> MultiObstacleStrictConvexCandidateResourceUsageV2:
    if type(usage) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("resource_usage has the wrong exact type")
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=usage.domain_operations,
        so2_atomic_steps=usage.so2_atomic_steps,
        candidate_cells=usage.candidate_cells,
    )


def _copy_stage(
    stage: DirectionalTargetCandidateStageV2_9,
) -> DirectionalTargetCandidateStageV2_9:
    return DirectionalTargetCandidateStageV2_9(
        semantic_problem_sha256=stage.semantic_problem_sha256,
        candidate_problem_sha256=stage.candidate_problem_sha256,
        camera_context_sha256=stage.camera_context_sha256,
        compiler_config_sha256=stage.compiler_config_sha256,
        upstream_t15_artifact_sha256=stage.upstream_t15_artifact_sha256,
        subject_id=stage.subject_id,
        target_constraint_id=stage.target_constraint_id,
        target_relation=stage.target_relation,
        target_threshold=stage.target_threshold,
        requires_both_visible=stage.requires_both_visible,
        inner_allowed=stage.inner_allowed,
        outer_allowed=stage.outer_allowed,
        resource_usage=stage.resource_usage,
        remaining_constraint_ids=stage.remaining_constraint_ids,
    )


def _canonical_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise ValueError("remaining constraint IDs must be exact non-blank strings")
    if len(value) != len(set(value)):
        raise ValueError("remaining constraint IDs must be unique")
    return tuple(sorted(value))


def _require_digest(label: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(
    kind: DirectionalTargetCandidateKindV2_9, finding: str
) -> DirectionalTargetCandidateOutcomeV2_9:
    return DirectionalTargetCandidateOutcomeV2_9(
        kind=kind,
        finding_codes=(finding,),
    )


__all__ = (
    "DirectionalTargetCandidateKindV2_9",
    "DirectionalTargetCandidateOutcomeV2_9",
    "DirectionalTargetCandidateStageV2_9",
    "compile_directional_target_candidate_v2_9",
)
