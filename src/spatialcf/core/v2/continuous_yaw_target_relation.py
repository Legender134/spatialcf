"""Private T16 target-relation projection over the T15 strict-convex prefix."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.convex_translation_domain import (
    ConvexTranslationDomainKindV2,
    RationalConvexPolygonV2,
    RationalPoint2V2,
    _canonical_convex_hull_v2,
    _clip_polygon_to_universe_v2,
    compile_convex_translation_obstacle_v2,
)
from spatialcf.core.v2.convex_translation_partition import (
    ConvexAllowedCellComplexV2,
    RationalHalfPlane2V2,
    RationalHalfPlaneRelationV2,
    _canonical_half_plane_v2,
    _compile_polygon_complement_v2,
)
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _require_finding_codes,
    _strict_problem,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
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
    StrictConvexIntersectionKindV2,
    StrictConvexIntersectionTopologyV2,
    _clip_universe_by_planes,
    _find_strict_witness,
    _universe_planes,
    intersect_strict_convex_allowed_complexes_v2,
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
    Vec3V2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    MeasurementComparatorV2,
    RelationMeasurementV2,
    RelationV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_candidate import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

CONTINUOUS_YAW_TARGET_RELATION_KERNEL_ID_V2 = (
    "geometry-kernel:rational-continuous-yaw-shape-gap-v2"
)
_STAGE_HASH_DOMAIN_V2 = b"spatialcf.continuous-yaw-target-stage.v2.8\0"


class _TargetAwareCandidateKindV2(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, slots=True)
class _TargetAwareCandidateStageV2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    upstream_t15_artifact_sha256: str
    subject_id: str
    target_constraint_id: str
    target_relation: RelationV2
    target_threshold_m: Fraction
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t15_artifact_sha256", self.upstream_t15_artifact_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label, value in (
            ("subject_id", self.subject_id),
            ("target_constraint_id", self.target_constraint_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        if self.target_relation not in (RelationV2.NEAR, RelationV2.FAR):
            raise ValueError("target stage accepts only NEAR or FAR")
        if type(self.target_threshold_m) is not Fraction or self.target_threshold_m < 0:
            raise ValueError("target threshold must be a non-negative Fraction")
        inner = _copy_intersection_complex(self.inner_allowed)
        outer = _copy_intersection_complex(self.outer_allowed)
        if inner.universe != outer.universe:
            raise ValueError("target inner and outer require one exact universe")
        if not all(outer.contains_point(cell.strict_witness) for cell in inner.cells):
            raise ValueError("target inner witness escaped the target outer domain")
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
        remaining = _canonical_ids(self.remaining_constraint_ids)
        if self.target_constraint_id in remaining:
            raise ValueError("compiled target cannot remain in the constraint suffix")
        object.__setattr__(self, "inner_allowed", inner)
        object.__setattr__(self, "outer_allowed", outer)
        object.__setattr__(self, "resource_usage", usage)
        object.__setattr__(self, "remaining_constraint_ids", remaining)

    @property
    def stage_sha256(self) -> str:
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _TargetAwareCandidateOutcomeV2:
    kind: _TargetAwareCandidateKindV2
    stage: _TargetAwareCandidateStageV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not _TargetAwareCandidateKindV2:
            raise TypeError("kind must be an exact target-aware kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is _TargetAwareCandidateKindV2.STAGE:
            if type(self.stage) is not _TargetAwareCandidateStageV2 or findings:
                raise ValueError("STAGE requires one exact stage and no findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("target failure requires findings and no stage")


class _UnsupportedTargetRelationV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def _compile_target_aware_candidate_v2(
    problem: SemanticProblemV2_2,
    t15_artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> _TargetAwareCandidateOutcomeV2:
    """Compile one exact SHAPE_GAP_XY target on continued T15 ledgers."""

    try:
        checked_problem, checked_artifact = _strict_inputs(
            problem,
            t15_artifact,
            atomic_budget,
            intersection_budget,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            target, relation, threshold, subject, reference = _extract_target(
                checked_problem,
                intersection_budget,
            )
            expanded_universe = _expanded_universe(
                checked_artifact.search_universe,
                threshold,
                atomic_budget,
            )
            overlap = compile_convex_translation_obstacle_v2(
                subject[0],
                subject[1],
                reference[0],
                reference[1],
                expanded_universe,
                atomic_budget=atomic_budget,
            )
            if overlap.kind is ConvexTranslationDomainKindV2.RESOURCE_LIMIT:
                raise SO2AtomicBudgetExhaustedV2
            if overlap.kind is ConvexTranslationDomainKindV2.NUMERIC_GAP:
                raise ArithmeticError("directed relation overlap numeric gap")
            if overlap.kind is ConvexTranslationDomainKindV2.INVALID_INPUT:
                raise RuntimeError("supported target produced invalid overlap input")
            if overlap.kind is ConvexTranslationDomainKindV2.IDENTITY:
                # The collision bracket is compiled over a universe expanded by
                # the exact target threshold.  If the two XY footprints still
                # cannot overlap there, no point in the original universe can
                # satisfy the corresponding NEAR predicate after that offset.
                inner_near_polygon = None
                outer_near_polygon = None
            elif (
                overlap.kind is ConvexTranslationDomainKindV2.BRACKET
                and overlap.bracket is not None
            ):
                inner_near_polygon = _offset_and_clip_polygon(
                    overlap.bracket.inner_forbidden,
                    threshold,
                    square=False,
                    universe=checked_artifact.search_universe,
                    atomic_budget=atomic_budget,
                )
                outer_near_polygon = _offset_and_clip_polygon(
                    overlap.bracket.outer_forbidden,
                    threshold,
                    square=True,
                    universe=checked_artifact.search_universe,
                    atomic_budget=atomic_budget,
                )
            else:
                raise RuntimeError("relation boxes must produce an overlap bracket")
            target_inner, target_outer = _target_complexes(
                relation,
                inner_near_polygon,
                outer_near_polygon,
                checked_artifact.search_universe,
                atomic_budget,
                intersection_budget,
            )
            inner = _intersect(
                checked_artifact.allowed_domain_bracket.inner_allowed,
                target_inner,
                intersection_budget,
            )
            outer = _intersect(
                checked_artifact.allowed_domain_bracket.outer_allowed,
                target_outer,
                intersection_budget,
            )
            remaining = tuple(
                item
                for item in checked_artifact.remaining_constraint_ids
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
            stage = _TargetAwareCandidateStageV2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_artifact.compiler_config_sha256,
                upstream_t15_artifact_sha256=checked_artifact.artifact_sha256,
                subject_id=checked_artifact.subject_id,
                target_constraint_id=target.constraint_id,
                target_relation=relation,
                target_threshold_m=threshold,
                inner_allowed=inner,
                outer_allowed=outer,
                resource_usage=MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=intersection_budget.domain_operations_used,
                    so2_atomic_steps=atomic_budget.used,
                    candidate_cells=intersection_budget.candidate_cells_used,
                ),
                remaining_constraint_ids=remaining,
            )
        return _TargetAwareCandidateOutcomeV2(
            kind=_TargetAwareCandidateKindV2.STAGE,
            stage=stage,
        )
    except _UnsupportedTargetRelationV2 as error:
        return _failure(
            _TargetAwareCandidateKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _failure(
            _TargetAwareCandidateKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_TARGET_RELATION",
        )
    except SO2AtomicBudgetExhaustedV2:
        return _failure(
            _TargetAwareCandidateKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_TARGET_RELATION",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            _TargetAwareCandidateKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_TARGET_RELATION",
        )


def _strict_inputs(
    problem: object,
    artifact: object,
    atomic_budget: object,
    intersection_budget: object,
) -> tuple[SemanticProblemV2_2, SupportStrictConvexCandidateDomainArtifactV2_2]:
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be StrictConvexIntersectionBudgetV2")
    if type(artifact) is not SupportStrictConvexCandidateDomainArtifactV2_2:
        raise TypeError("t15_artifact must be the exact T15 artifact type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = _strict_problem(problem)
        checked_artifact = _copy_t15_artifact(artifact)
    if (
        checked_artifact.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
    ):
        raise ValueError("T15 artifact problem hash is not closed")
    usage = checked_artifact.resource_usage
    if (
        atomic_budget.used != usage.so2_atomic_steps
        or intersection_budget.domain_operations_used != usage.domain_operations
        or intersection_budget.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("live target ledgers must continue exact T15 usage")
    target_id = checked_problem.constraints.target_relation.constraint_id
    if target_id not in checked_artifact.remaining_constraint_ids:
        raise ValueError("T15 artifact does not expose the target relation")
    return checked_problem, checked_artifact


def _extract_target(
    problem: SemanticProblemV2_2,
    budget: StrictConvexIntersectionBudgetV2,
):
    target = problem.constraints.target_relation
    definition = next(
        item
        for item in problem.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    expected_comparator = {
        RelationV2.NEAR: MeasurementComparatorV2.LESS_THAN,
        RelationV2.FAR: MeasurementComparatorV2.GREATER_THAN,
    }.get(target.relation_after)
    if (
        expected_comparator is None
        or definition.measurement is not RelationMeasurementV2.SHAPE_GAP_XY
        or definition.comparator is not expected_comparator
        or definition.boundary_policy is not BoundaryPolicyV2.CLOSED
        or definition.tolerance != 0.0
        or problem.numeric_policy != NumericPolicyV2()
    ):
        raise _UnsupportedTargetRelationV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_RELATION_POLICY"
        )
    threshold = Fraction.from_float(definition.threshold)
    if threshold < 0:
        raise _UnsupportedTargetRelationV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_RELATION_THRESHOLD"
        )
    for label, facts in (
        ("OBJECTS", problem.scene.objects),
        ("GEOMETRIES", problem.scene.geometry_instances),
    ):
        if (
            facts.availability is not FactAvailabilityV2.KNOWN
            or facts.completeness is not FactCompletenessV2.EXACT
            or facts.uncertainty != UncertaintyBudgetV2()
            or type(facts.values) is not tuple
        ):
            raise _UnsupportedTargetRelationV2(
                f"UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_{label}"
            )
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = tuple(
        item
        for item in problem.scene.geometry_instances.values or ()
        if item.role is GeometryRoleV2.RELATION
        and item.owner_object_id in (target.subject_id, target.reference_id)
    )
    budget.consume_domain(len(objects) + len(geometries) + 4)
    try:
        subject_object = objects[target.subject_id]
        reference_object = objects[target.reference_id]
    except KeyError as error:
        raise RuntimeError("target semantic graph lost an object") from error
    if not subject_object.movable or reference_object.movable:
        raise _UnsupportedTargetRelationV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_REFERENCE"
        )
    by_owner: dict[str, list[GeometryInstanceV2_2]] = {}
    for geometry in geometries:
        by_owner.setdefault(geometry.owner_object_id or "", []).append(geometry)
    if any(
        len(by_owner.get(owner_id, ())) != 1
        for owner_id in (target.subject_id, target.reference_id)
    ):
        raise _UnsupportedTargetRelationV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_GEOMETRY_CARDINALITY"
        )
    subject_geometry = by_owner[target.subject_id][0]
    reference_geometry = by_owner[target.reference_id][0]
    for geometry in (subject_geometry, reference_geometry):
        anchor = geometry.anchor_from_geometry
        if (
            type(geometry) is not GeometryInstanceV2_2
            or geometry.approximation is not GeometryApproximationV2.EXACT
            or geometry.uncertainty != UncertaintyBudgetV2()
            or type(geometry.shape) is not UprightBox3DV2
            or anchor.yaw_radians != 0.0
            or (anchor.translation.x, anchor.translation.y) != (0.0, 0.0)
        ):
            raise _UnsupportedTargetRelationV2(
                "UNSUPPORTED_MODEL:CONTINUOUS_YAW_TARGET_GEOMETRY"
            )
    subject_transform = _xy_relation_transform(subject_object.pose.world_from_object)
    reference_transform = _xy_relation_transform(
        reference_object.pose.world_from_object
    )
    return (
        target,
        target.relation_after,
        threshold,
        (subject_transform, subject_geometry.shape),
        (reference_transform, reference_geometry.shape),
    )


def _xy_relation_transform(
    transform: DirectedYawIntervalTransformV2_2,
) -> DirectedYawIntervalTransformV2_2:
    if type(transform) is not DirectedYawIntervalTransformV2_2:
        raise RuntimeError("v2.2 object lost its directed-yaw transform")
    return DirectedYawIntervalTransformV2_2(
        translation=Vec3V2(
            x=transform.translation.x,
            y=transform.translation.y,
            z=0.0,
        ),
        yaw_radians=transform.yaw_radians,
    )


def _expanded_universe(
    universe: ExactAxisAlignedRectV2,
    threshold: Fraction,
    budget: SO2AtomicBudgetV2,
) -> ExactAxisAlignedRectV2:
    budget.consume(8)
    bounds = universe.bounds
    if bounds is None:
        raise RuntimeError("T15 search universe cannot be empty")
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=bounds[0] - threshold,
        min_y_m=bounds[1] - threshold,
        max_x_m=bounds[2] + threshold,
        max_y_m=bounds[3] + threshold,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _offset_and_clip_polygon(
    polygon: RationalConvexPolygonV2 | None,
    threshold: Fraction,
    *,
    square: bool,
    universe: ExactAxisAlignedRectV2,
    atomic_budget: SO2AtomicBudgetV2,
) -> RationalConvexPolygonV2 | None:
    if polygon is None:
        return None
    offsets = (
        (
            (threshold, threshold),
            (-threshold, threshold),
            (-threshold, -threshold),
            (threshold, -threshold),
        )
        if square
        else (
            (threshold, Fraction()),
            (Fraction(), threshold),
            (-threshold, Fraction()),
            (Fraction(), -threshold),
        )
    )
    atomic_budget.consume(2 + len(polygon.vertices_ccw) * len(offsets))
    points = tuple(
        RationalPoint2V2(x=point.x + offset_x, y=point.y + offset_y)
        for point in polygon.vertices_ccw
        for offset_x, offset_y in offsets
    )
    hull = _canonical_convex_hull_v2(points, atomic_budget)
    if hull is None:
        return None
    return _clip_polygon_to_universe_v2(hull, universe, atomic_budget)


def _target_complexes(
    relation: RelationV2,
    inner_near: RationalConvexPolygonV2 | None,
    outer_near: RationalConvexPolygonV2 | None,
    universe: ExactAxisAlignedRectV2,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> tuple[
    ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2,
    ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2,
]:
    if relation is RelationV2.NEAR:
        return (
            _inside_complex(inner_near, universe, atomic_budget, intersection_budget),
            _inside_complex(outer_near, universe, atomic_budget, intersection_budget),
        )
    if relation is not RelationV2.FAR:
        raise RuntimeError("unsupported target relation escaped extraction")
    if outer_near is None:
        raw_target_inner = _identity_complex(universe, intersection_budget)
    else:
        raw_target_inner = _compile_polygon_complement_v2(
            outer_near,
            universe,
            atomic_budget,
        )
    # The strict-cell complement intentionally assigns shared polygon edges to
    # outer cells, but its two-dimensional representation cannot retain a
    # boundary-only component when that edge also lies on the search-universe
    # boundary.  FAR inner is an under-approximation, so excluding all universe
    # boundary points is sound and keeps it nested in the inclusive FAR outer.
    target_inner = _intersect(
        _strict_universe_interior_complex(
            universe,
            atomic_budget,
            intersection_budget,
        ),
        raw_target_inner,
        intersection_budget,
    )
    target_outer = _inclusive_complement_complex(
        inner_near,
        universe,
        atomic_budget,
        intersection_budget,
    )
    return target_inner, target_outer


def _inside_complex(
    polygon: RationalConvexPolygonV2 | None,
    universe: ExactAxisAlignedRectV2,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    if polygon is None:
        return _empty_complex(universe)
    semantic = tuple(
        sorted(
            _polygon_inside_planes(
                polygon, RationalHalfPlaneRelationV2.LE, atomic_budget
            ),
            key=_plane_key,
        )
    )
    closure_vertices = _clip_universe_by_planes(
        universe,
        semantic,
        intersection_budget,
    )
    if closure_vertices is None:
        return _empty_complex(universe)
    half_planes = _universe_planes(universe) + semantic
    witness = _find_strict_witness(
        closure_vertices,
        half_planes,
        intersection_budget,
    )
    if witness is None:
        return _empty_complex(universe)
    intersection_budget.consume_candidate_cells()
    cell = StrictConvexIntersectionCellV2(
        cell_id="cell:target-relation:inside",
        half_planes=half_planes,
        closure_polygon=RationalConvexPolygonV2(vertices_ccw=closure_vertices),
        strict_witness=witness,
    )
    return StrictConvexIntersectionComplexV2(
        cells=(cell,),
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _inclusive_complement_complex(
    polygon: RationalConvexPolygonV2 | None,
    universe: ExactAxisAlignedRectV2,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    if polygon is None:
        return _identity_complex(universe, intersection_budget)
    inside_le = _polygon_inside_planes(
        polygon,
        RationalHalfPlaneRelationV2.LE,
        atomic_budget,
    )
    inside_lt = _polygon_inside_planes(
        polygon,
        RationalHalfPlaneRelationV2.LT,
        atomic_budget,
    )
    cells = []
    for index, inside in enumerate(inside_le):
        outside = _canonical_half_plane_v2(
            -inside.normal_x,
            -inside.normal_y,
            -inside.offset,
            RationalHalfPlaneRelationV2.LE,
            atomic_budget,
        )
        semantic = tuple(sorted((*inside_lt[:index], outside), key=_plane_key))
        closure_vertices = _clip_universe_by_planes(
            universe,
            semantic,
            intersection_budget,
        )
        if closure_vertices is None:
            continue
        half_planes = _universe_planes(universe) + semantic
        witness = _find_strict_witness(
            closure_vertices,
            half_planes,
            intersection_budget,
        )
        if witness is None:
            continue
        intersection_budget.consume_candidate_cells()
        cells.append(
            StrictConvexIntersectionCellV2(
                cell_id=f"cell:target-relation:inclusive-complement:{index:06d}",
                half_planes=half_planes,
                closure_polygon=RationalConvexPolygonV2(vertices_ccw=closure_vertices),
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


def _strict_universe_interior_complex(
    universe: ExactAxisAlignedRectV2,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    bounds = universe.bounds
    if bounds is None:
        raise RuntimeError("target universe cannot be empty")
    semantic = tuple(
        sorted(
            (
                _canonical_half_plane_v2(
                    Fraction(-1),
                    Fraction(),
                    -bounds[0],
                    RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
                _canonical_half_plane_v2(
                    Fraction(1),
                    Fraction(),
                    bounds[2],
                    RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
                _canonical_half_plane_v2(
                    Fraction(),
                    Fraction(-1),
                    -bounds[1],
                    RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
                _canonical_half_plane_v2(
                    Fraction(),
                    Fraction(1),
                    bounds[3],
                    RationalHalfPlaneRelationV2.LT,
                    atomic_budget,
                ),
            ),
            key=_plane_key,
        )
    )
    closure = _clip_universe_by_planes(
        universe,
        semantic,
        intersection_budget,
    )
    if closure is None:
        return _empty_complex(universe)
    half_planes = _universe_planes(universe) + semantic
    witness = _find_strict_witness(closure, half_planes, intersection_budget)
    if witness is None:
        return _empty_complex(universe)
    intersection_budget.consume_candidate_cells()
    return StrictConvexIntersectionComplexV2(
        cells=(
            StrictConvexIntersectionCellV2(
                cell_id="cell:target-relation:strict-universe-interior",
                half_planes=half_planes,
                closure_polygon=RationalConvexPolygonV2(vertices_ccw=closure),
                strict_witness=witness,
            ),
        ),
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _polygon_inside_planes(
    polygon: RationalConvexPolygonV2,
    relation: RationalHalfPlaneRelationV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalHalfPlane2V2, ...]:
    vertices = polygon.vertices_ccw
    return tuple(
        _canonical_half_plane_v2(
            second.y - first.y,
            first.x - second.x,
            (second.y - first.y) * first.x + (first.x - second.x) * first.y,
            relation,
            budget,
        )
        for index, first in enumerate(vertices)
        for second in (vertices[(index + 1) % len(vertices)],)
    )


def _identity_complex(
    universe: ExactAxisAlignedRectV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    bounds = universe.bounds
    if bounds is None:
        raise RuntimeError("target universe cannot be empty")
    vertices = (
        RationalPoint2V2(x=bounds[0], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[3]),
        RationalPoint2V2(x=bounds[0], y=bounds[3]),
    )
    half_planes = _universe_planes(universe)
    budget.consume_domain(12)
    budget.consume_candidate_cells()
    return StrictConvexIntersectionComplexV2(
        cells=(
            StrictConvexIntersectionCellV2(
                cell_id="cell:target-relation:universe",
                half_planes=half_planes,
                closure_polygon=RationalConvexPolygonV2(vertices_ccw=vertices),
                strict_witness=RationalPoint2V2(
                    x=(bounds[0] + bounds[2]) / 2,
                    y=(bounds[1] + bounds[3]) / 2,
                ),
            ),
        ),
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _empty_complex(
    universe: ExactAxisAlignedRectV2,
) -> StrictConvexIntersectionComplexV2:
    return StrictConvexIntersectionComplexV2(
        cells=(),
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _intersect(
    upstream: StrictConvexIntersectionComplexV2,
    target: ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    outcome = intersect_strict_convex_allowed_complexes_v2(
        (upstream, target),
        budget=budget,
    )
    if outcome.kind is StrictConvexIntersectionKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is StrictConvexIntersectionKindV2.NUMERIC_GAP:
        raise ArithmeticError("target strict intersection numeric gap")
    if outcome.kind is StrictConvexIntersectionKindV2.INVALID_INPUT:
        raise RuntimeError("target compiler produced invalid intersection operands")
    if outcome.complex is None:
        raise RuntimeError("target intersection lost its complex")
    return outcome.complex


def _plane_key(
    plane: RationalHalfPlane2V2,
) -> tuple[Fraction, Fraction, Fraction, int]:
    return (
        plane.normal_x,
        plane.normal_y,
        plane.offset,
        0 if plane.relation is RationalHalfPlaneRelationV2.LT else 1,
    )


def _canonical_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise ValueError("remaining constraint IDs must be exact non-blank strings")
    if len(value) != len(set(value)):
        raise ValueError("remaining constraint IDs must be unique")
    return tuple(sorted(value))


def _copy_stage(value: _TargetAwareCandidateStageV2) -> _TargetAwareCandidateStageV2:
    return _TargetAwareCandidateStageV2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        upstream_t15_artifact_sha256=value.upstream_t15_artifact_sha256,
        subject_id=value.subject_id,
        target_constraint_id=value.target_constraint_id,
        target_relation=value.target_relation,
        target_threshold_m=value.target_threshold_m,
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
    )


def _failure(
    kind: _TargetAwareCandidateKindV2,
    finding_code: str,
) -> _TargetAwareCandidateOutcomeV2:
    return _TargetAwareCandidateOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


__all__ = (
    "CONTINUOUS_YAW_TARGET_RELATION_KERNEL_ID_V2",
    "_TargetAwareCandidateKindV2",
    "_TargetAwareCandidateOutcomeV2",
    "_TargetAwareCandidateStageV2",
    "_compile_target_aware_candidate_v2",
)
