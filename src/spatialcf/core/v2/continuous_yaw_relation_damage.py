"""Non-target relation-damage bounds for the competition v2.9 chain."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    UprightCameraContextV2_9,
    bound_world_point_in_upright_camera_v2_9,
    compile_upright_camera_context_v2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_v2_9 import (
    ContinuousYawVisibilityStageV2_9,
    VisibilityCellMetricBoundsV2_9,
)
from spatialcf.core.v2.continuous_yaw_visibility_v2_9 import (
    _copy_stage as _copy_visibility_stage,
)
from spatialcf.core.v2.convex_translation_domain import RationalPoint2V2
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _require_finding_codes,
)
from spatialcf.core.v2.oriented_upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.core.v2.relation_cost_partition import (
    _axis_definitions,
    _distance_label_set,
    _linear_label_set,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
)
from spatialcf.domain.v2.artifacts import RelationDamageBoundV2
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec3V2,
)
from spatialcf.domain.v2.constraints import RelationAxisV2, RelationDefinitionV2
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_candidate import (
    CanonicalObjectV2_2,
    GeometryInstanceV2_2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.objective import PairAxisKeyV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-relation-damage.v2.9\0"
_IntervalV2 = tuple[Fraction, Fraction]


class ContinuousYawRelationDamageKindV2_9(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class RelationDamageCellV2_9:
    cell: StrictConvexIntersectionCellV2
    vector: tuple[RelationDamageBoundV2, ...]

    def __post_init__(self) -> None:
        if type(self.cell) is not StrictConvexIntersectionCellV2:
            raise TypeError("relation-damage cell must carry an exact source cell")
        checked_cell = StrictConvexIntersectionCellV2(
            cell_id=self.cell.cell_id,
            half_planes=self.cell.half_planes,
            closure_polygon=self.cell.closure_polygon,
            strict_witness=self.cell.strict_witness,
        )
        if type(self.vector) is not tuple or not self.vector:
            raise ValueError("relation-damage vector must be a non-empty exact tuple")
        vector = tuple(
            RelationDamageBoundV2.model_validate(
                item.model_dump(mode="python", warnings="error"), strict=True
            )
            for item in self.vector
        )
        if vector != tuple(sorted(vector, key=lambda item: item.key.sort_key)):
            raise ValueError("relation-damage vector must be canonically sorted")
        if len({item.key.sort_key for item in vector}) != len(vector):
            raise ValueError("relation-damage vector keys must be unique")
        object.__setattr__(self, "cell", checked_cell)
        object.__setattr__(self, "vector", vector)

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id


@dataclass(frozen=True, slots=True)
class ContinuousYawRelationDamageStageV2_9:
    semantic_problem_sha256: str
    candidate_problem_sha256: str
    camera_context_sha256: str
    visibility_stage_sha256: str
    cells: tuple[RelationDamageCellV2_9, ...]
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2

    def __post_init__(self) -> None:
        for label, value in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("candidate_problem_sha256", self.candidate_problem_sha256),
            ("camera_context_sha256", self.camera_context_sha256),
            ("visibility_stage_sha256", self.visibility_stage_sha256),
        ):
            _require_digest(label, value)
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("relation-damage stage requires non-empty cells")
        cells = tuple(_copy_cell(item) for item in self.cells)
        if cells != tuple(sorted(cells, key=lambda item: item.cell_id)):
            raise ValueError("relation-damage cells must be canonically sorted")
        key_universe = tuple(item.key for item in cells[0].vector)
        if any(
            tuple(item.key for item in cell.vector) != key_universe for cell in cells
        ):
            raise ValueError("relation-damage key universe changed across cells")
        usage = _copy_usage(self.resource_usage)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "resource_usage", usage)

    @property
    def stage_sha256(self) -> str:
        digest = hashlib.sha256(_STAGE_HASH_DOMAIN_V2_9)
        for value in (
            self.semantic_problem_sha256,
            self.candidate_problem_sha256,
            self.camera_context_sha256,
            self.visibility_stage_sha256,
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        for cell in self.cells:
            digest.update(_artifact_bytes(cell.cell))  # type: ignore[arg-type]
            for bound in cell.vector:
                digest.update(canonical_json_bytes_v2(bound))
        digest.update(_artifact_bytes(self.resource_usage))  # type: ignore[arg-type]
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousYawRelationDamageOutcomeV2_9:
    kind: ContinuousYawRelationDamageKindV2_9
    stage: ContinuousYawRelationDamageStageV2_9 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawRelationDamageKindV2_9:
            raise TypeError("kind must be an exact relation-damage kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ContinuousYawRelationDamageKindV2_9.STAGE:
            if type(self.stage) is not ContinuousYawRelationDamageStageV2_9:
                raise ValueError("STAGE requires one exact relation-damage stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("relation-damage failure requires findings and no stage")


@dataclass(frozen=True, slots=True)
class _RelationContextV2_9:
    problem: SemanticProblemV2_3
    camera: UprightCameraContextV2_9
    subject_id: str
    objects: dict[str, CanonicalObjectV2_2]
    geometries: dict[str, GeometryInstanceV2_2]


class _UnsupportedRelationDamageV2_9(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def compile_relation_damage_bounds_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawRelationDamageOutcomeV2_9:
    """Bound every non-target pair-axis damage indicator on each outer cell."""

    try:
        checked_problem, checked_visibility = _strict_inputs(
            problem, visibility_stage, atomic_budget, intersection_budget
        )
        if not checked_visibility.outer_allowed.cells:
            raise _UnsupportedRelationDamageV2_9(
                "UNSUPPORTED_MODEL:RELATION_DAMAGE_EMPTY_OUTER_V2_9"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            camera = compile_upright_camera_context_v2_9(
                checked_problem,
                atomic_budget=atomic_budget,
                domain_budget=intersection_budget,
            )
            if camera.context_sha256 != checked_visibility.camera_context_sha256:
                raise ValueError("visibility stage camera context is not replay-closed")
            context = _relation_context(checked_problem, camera, intersection_budget)
            keys = tuple(
                sorted(
                    (
                        item.key
                        for item in checked_problem.objective.relation_damage.pair_axis_weights
                    ),
                    key=lambda item: item.sort_key,
                )
            )
            intersection_budget.consume_domain(
                12
                + len(keys) * len(checked_visibility.outer_allowed.cells)
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for cell in checked_visibility.outer_allowed.cells
                )
            )
            cells = tuple(
                RelationDamageCellV2_9(
                    cell=cell,
                    vector=tuple(
                        _bound_key(
                            context,
                            checked_visibility,
                            cell,
                            key,
                            atomic_budget,
                        )
                        for key in keys
                    ),
                )
                for cell in checked_visibility.outer_allowed.cells
            )
            stage = ContinuousYawRelationDamageStageV2_9(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                candidate_problem_sha256=checked_visibility.candidate_problem_sha256,
                camera_context_sha256=camera.context_sha256,
                visibility_stage_sha256=checked_visibility.stage_sha256,
                cells=cells,
                resource_usage=_usage(atomic_budget, intersection_budget),
            )
        return ContinuousYawRelationDamageOutcomeV2_9(
            kind=ContinuousYawRelationDamageKindV2_9.STAGE,
            stage=stage,
        )
    except _UnsupportedRelationDamageV2_9 as error:
        return _failure(
            ContinuousYawRelationDamageKindV2_9.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            ContinuousYawRelationDamageKindV2_9.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_RELATION_DAMAGE_V2_9",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ContinuousYawRelationDamageKindV2_9.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_RELATION_DAMAGE_V2_9",
        )


def evaluate_relation_damage_point_v2_9(
    problem: SemanticProblemV2_3,
    visibility_stage: ContinuousYawVisibilityStageV2_9,
    point: RationalPoint2V2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> tuple[RelationDamageBoundV2, ...]:
    """Freshly recompute every damage key at one exact translation edit."""

    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(visibility_stage) is not ContinuousYawVisibilityStageV2_9:
        raise TypeError("visibility_stage must be the exact v2.9 stage")
    if type(point) is not RationalPoint2V2:
        raise TypeError("point must be an exact RationalPoint2V2")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    checked_problem = SemanticProblemV2_3.model_validate(
        problem.model_dump(mode="python", warnings="error"), strict=True
    )
    checked_visibility = _copy_visibility_stage(visibility_stage)
    if checked_visibility.semantic_problem_sha256 != (
        checked_problem.semantic_problem_sha256
    ):
        raise ValueError("point relation replay problem is not closed")
    if point.x == 0 and point.y == 0:
        intersection_budget.consume_domain()
        return tuple(
            RelationDamageBoundV2(key=item.key, lower_bound=0.0, upper_bound=0.0)
            for item in sorted(
                checked_problem.objective.relation_damage.pair_axis_weights,
                key=lambda item: item.key.sort_key,
            )
        )
    camera = compile_upright_camera_context_v2_9(
        checked_problem,
        atomic_budget=atomic_budget,
        domain_budget=intersection_budget,
    )
    context = _relation_context(checked_problem, camera, intersection_budget)
    matching = tuple(
        cell
        for cell in checked_visibility.outer_allowed.cells
        if cell.contains_point(point)
    )
    if not matching:
        raise ValueError("point relation replay lies outside the visibility outer")
    return tuple(
        _hull_bounds(
            item.key,
            tuple(
                _bound_key_for_delta(
                    context,
                    checked_visibility,
                    cell.cell_id,
                    item.key,
                    (point.x, point.x),
                    (point.y, point.y),
                    atomic_budget,
                )
                for cell in matching
            ),
        )
        for item in sorted(
            checked_problem.objective.relation_damage.pair_axis_weights,
            key=lambda weighted: weighted.key.sort_key,
        )
    )


def _hull_bounds(
    key: PairAxisKeyV2,
    values: tuple[RelationDamageBoundV2, ...],
) -> RelationDamageBoundV2:
    if not values or any(item.key != key for item in values):
        raise RuntimeError("point relation hull lost its key closure")
    return RelationDamageBoundV2(
        key=key,
        lower_bound=min(item.lower_bound for item in values),
        upper_bound=max(item.upper_bound for item in values),
    )


def _strict_inputs(
    problem: object,
    visibility_stage: object,
    atomic_budget: object,
    domain_budget: object,
) -> tuple[SemanticProblemV2_3, ContinuousYawVisibilityStageV2_9]:
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(visibility_stage) is not ContinuousYawVisibilityStageV2_9:
        raise TypeError("visibility_stage must be the exact v2.9 stage")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    if type(domain_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    atomic_budget.validate()
    usage = visibility_stage.resource_usage
    if (
        atomic_budget.used != usage.so2_atomic_steps
        or domain_budget.domain_operations_used != usage.domain_operations
        or domain_budget.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("relation-damage ledgers must continue visibility usage")
    domain_budget.consume_domain(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = SemanticProblemV2_3.model_validate(
            problem.model_dump(mode="python", warnings="error"), strict=True
        )
        checked_visibility = _copy_visibility_stage(visibility_stage)
    if checked_visibility.semantic_problem_sha256 != (
        checked_problem.semantic_problem_sha256
    ):
        raise ValueError("visibility stage does not close the relation problem")
    if checked_visibility.remaining_constraint_ids:
        raise ValueError("relation damage requires a complete hard prefix")
    return checked_problem, checked_visibility


def _relation_context(
    problem: SemanticProblemV2_3,
    camera: UprightCameraContextV2_9,
    budget: StrictConvexIntersectionBudgetV2,
) -> _RelationContextV2_9:
    if problem.numeric_policy != NumericPolicyV2():
        raise _UnsupportedRelationDamageV2_9(
            "UNSUPPORTED_MODEL:RELATION_DAMAGE_NUMERIC_POLICY_V2_9"
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
            raise _UnsupportedRelationDamageV2_9(
                f"UNSUPPORTED_MODEL:RELATION_DAMAGE_{label}_V2_9"
            )
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = {}
    for object_id in objects:
        selected = tuple(
            item
            for item in problem.scene.geometry_instances.values or ()
            if item.owner_object_id == object_id
            and item.role is GeometryRoleV2.RELATION
        )
        if len(selected) != 1:
            raise _UnsupportedRelationDamageV2_9(
                "UNSUPPORTED_MODEL:RELATION_DAMAGE_GEOMETRY_CARDINALITY_V2_9"
            )
        geometry = selected[0]
        if (
            type(geometry) is not GeometryInstanceV2_2
            or geometry.approximation is not GeometryApproximationV2.EXACT
            or geometry.uncertainty != UncertaintyBudgetV2()
            or type(geometry.shape) is not UprightBox3DV2
            or (
                geometry.anchor_from_geometry.translation.x,
                geometry.anchor_from_geometry.translation.y,
            )
            != (0.0, 0.0)
        ):
            raise _UnsupportedRelationDamageV2_9(
                "UNSUPPORTED_MODEL:RELATION_DAMAGE_GEOMETRY_V2_9"
            )
        geometries[object_id] = geometry
    budget.consume_domain(len(objects) + len(geometries) + 4)
    return _RelationContextV2_9(
        problem=problem,
        camera=camera,
        subject_id=problem.constraints.allowed_edit.subject_id,
        objects=objects,
        geometries=geometries,
    )


def _bound_key(
    context: _RelationContextV2_9,
    visibility: ContinuousYawVisibilityStageV2_9,
    cell: StrictConvexIntersectionCellV2,
    key: PairAxisKeyV2,
    budget: SO2AtomicBudgetV2,
) -> RelationDamageBoundV2:
    vertices = cell.closure_polygon.vertices_ccw
    if key.axis in (RelationAxisV2.HORIZONTAL, RelationAxisV2.DEPTH):
        definitions = _axis_definitions(context.problem, key.axis)
        if definitions is None or not _visibility_gate(
            context.problem, visibility, cell.cell_id, key, definitions
        ):
            return RelationDamageBoundV2(key=key, lower_bound=0.0, upper_bound=1.0)
        _require_key_objects(context, key)
        baseline = _linear_measurement(
            context,
            key,
            (Fraction(), Fraction()),
            (Fraction(), Fraction()),
            budget,
        )
        candidate = _linear_measurement_over_convex_cell(context, key, vertices, budget)
        return _linear_damage_bound(key, definitions, baseline, candidate)
    return _bound_key_for_delta(
        context,
        visibility,
        cell.cell_id,
        key,
        (min(item.x for item in vertices), max(item.x for item in vertices)),
        (min(item.y for item in vertices), max(item.y for item in vertices)),
        budget,
    )


def _bound_key_for_delta(
    context: _RelationContextV2_9,
    visibility: ContinuousYawVisibilityStageV2_9,
    cell_id: str,
    key: PairAxisKeyV2,
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> RelationDamageBoundV2:
    definitions = _axis_definitions(context.problem, key.axis)
    if definitions is None or not _visibility_gate(
        context.problem, visibility, cell_id, key, definitions
    ):
        return RelationDamageBoundV2(key=key, lower_bound=0.0, upper_bound=1.0)
    _require_key_objects(context, key)
    if key.axis in (RelationAxisV2.HORIZONTAL, RelationAxisV2.DEPTH):
        baseline = _linear_measurement(
            context, key, (Fraction(), Fraction()), (Fraction(), Fraction()), budget
        )
        candidate = _linear_measurement(context, key, delta_x, delta_y, budget)
        return _linear_damage_bound(key, definitions, baseline, candidate)
    else:
        baseline = _distance_squared(
            context, key, (Fraction(), Fraction()), (Fraction(), Fraction()), budget
        )
        candidate = _distance_squared(context, key, delta_x, delta_y, budget)
        baseline_labels = _distance_label_set(definitions, *baseline)
        candidate_labels = _distance_label_set(definitions, *candidate)
    if baseline_labels is None or candidate_labels is None:
        return RelationDamageBoundV2(key=key, lower_bound=0.0, upper_bound=1.0)
    indicator = 0.0 if baseline_labels == candidate_labels else 1.0
    return RelationDamageBoundV2(key=key, lower_bound=indicator, upper_bound=indicator)


def _linear_measurement_over_convex_cell(
    context: _RelationContextV2_9,
    key: PairAxisKeyV2,
    vertices: tuple[RationalPoint2V2, ...],
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    """Enclose a linear or positive linear-fractional relation on one cell.

    Camera depth is affine in the edit. Projected horizontal position is a
    linear-fractional function whose denominator is proven positive by
    ``_linear_measurement``. Both attain their extrema over a compact convex
    polygon at a vertex.  Hulling the directed vertex evaluations therefore
    avoids the nonexistent corners introduced by an axis-aligned bounding box
    without trusting nearest-rounded trigonometry.
    """

    if not vertices:
        raise RuntimeError("relation cell cannot have an empty closure polygon")
    values = tuple(
        _linear_measurement(
            context,
            key,
            (vertex.x, vertex.x),
            (vertex.y, vertex.y),
            budget,
        )
        for vertex in vertices
    )
    return min(value[0] for value in values), max(value[1] for value in values)


def _linear_damage_bound(
    key: PairAxisKeyV2,
    definitions: tuple[RelationDefinitionV2, ...],
    baseline: _IntervalV2,
    candidate: _IntervalV2,
) -> RelationDamageBoundV2:
    baseline_labels = _linear_label_set(definitions, *baseline)
    candidate_labels = _linear_label_set(definitions, *candidate)
    if baseline_labels is None or candidate_labels is None:
        return RelationDamageBoundV2(key=key, lower_bound=0.0, upper_bound=1.0)
    indicator = 0.0 if baseline_labels == candidate_labels else 1.0
    return RelationDamageBoundV2(key=key, lower_bound=indicator, upper_bound=indicator)


def _require_key_objects(context: _RelationContextV2_9, key: PairAxisKeyV2) -> None:
    if key.first_object_id not in context.objects or key.second_object_id not in (
        context.objects
    ):
        raise RuntimeError("relation objective key escaped the object universe")


def _linear_measurement(
    context: _RelationContextV2_9,
    key: PairAxisKeyV2,
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    values = []
    for object_id in (key.first_object_id, key.second_object_id):
        object_ = context.objects[object_id]
        geometry = context.geometries[object_id]
        centroid = _centroid(object_, geometry)
        moving = object_id == context.subject_id
        point = bound_world_point_in_upright_camera_v2_9(
            context.camera,
            world_xyz=centroid,
            delta_x=delta_x if moving else (Fraction(), Fraction()),
            delta_y=delta_y if moving else (Fraction(), Fraction()),
            atomic_budget=budget,
        )
        if key.axis is RelationAxisV2.HORIZONTAL and not point.positive_depth:
            raise _UnsupportedRelationDamageV2_9(
                "UNSUPPORTED_MODEL:RELATION_DAMAGE_DEPTH_V2_9"
            )
        if key.axis is RelationAxisV2.HORIZONTAL:
            projected = _add_exact(
                _scale_exact(
                    _divide_positive(point.x_camera, point.z_camera, budget),
                    context.camera.intrinsics[0],
                    budget,
                ),
                context.camera.intrinsics[2],
                budget,
            )
            values.append(projected)
        else:
            values.append(point.z_camera)
    return _subtract(values[0], values[1], budget)


def _distance_squared(
    context: _RelationContextV2_9,
    key: PairAxisKeyV2,
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    boxes = []
    centers = []
    for object_id in (key.first_object_id, key.second_object_id):
        object_ = context.objects[object_id]
        geometry = context.geometries[object_id]
        box = _oriented_relation_box(object_, geometry, budget)
        moving = object_id == context.subject_id
        dx = delta_x if moving else (Fraction(), Fraction())
        dy = delta_y if moving else (Fraction(), Fraction())
        boxes.append(
            (
                box.aabb_x.rational_lower + dx[0],
                box.aabb_y.rational_lower + dy[0],
                box.aabb_x.rational_upper + dx[1],
                box.aabb_y.rational_upper + dy[1],
            )
        )
        centers.append(
            (
                (box.center_x + dx[0], box.center_x + dx[1]),
                (box.center_y + dy[0], box.center_y + dy[1]),
            )
        )
    lower_x = _interval_gap(boxes[0][0], boxes[0][2], boxes[1][0], boxes[1][2])
    lower_y = _interval_gap(boxes[0][1], boxes[0][3], boxes[1][1], boxes[1][3])
    center_dx = _subtract(centers[0][0], centers[1][0], budget)
    center_dy = _subtract(centers[0][1], centers[1][1], budget)
    upper_x = max(abs(center_dx[0]), abs(center_dx[1]))
    upper_y = max(abs(center_dy[0]), abs(center_dy[1]))
    budget.consume(4)
    return lower_x**2 + lower_y**2, upper_x**2 + upper_y**2


def _oriented_relation_box(
    object_: CanonicalObjectV2_2,
    geometry: GeometryInstanceV2_2,
    budget: SO2AtomicBudgetV2,
) -> OrientedUprightBoxBoundsV2:
    transform = object_.pose.world_from_object
    shifted = DirectedYawIntervalTransformV2_2(
        translation=Vec3V2(
            x=transform.translation.x,
            y=transform.translation.y,
            z=transform.translation.z + geometry.anchor_from_geometry.translation.z,
        ),
        yaw_radians=transform.yaw_radians,
    )
    assert type(geometry.shape) is UprightBox3DV2
    outcome = compile_oriented_upright_box_bounds_v2(
        shifted, geometry.shape, atomic_budget=budget
    )
    if outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        raise SO2AtomicBudgetExhaustedV2
    if outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
        raise ArithmeticError("relation box reached a numeric gap")
    if outcome.kind is not SO2IntervalKindV2.EXACT or outcome.bounds is None:
        raise RuntimeError("supported relation box did not publish bounds")
    return outcome.bounds


def _visibility_gate(
    problem: SemanticProblemV2_3,
    visibility: ContinuousYawVisibilityStageV2_9,
    cell_id: str,
    key: PairAxisKeyV2,
    definitions: tuple[object, ...],
) -> bool:
    if not any(item.requires_both_visible for item in definitions):
        return True
    constraint = problem.constraints.visibility_constraints[0]
    metrics = {
        (item.cell_id, item.object_id): item for item in visibility.outer_cell_metrics
    }
    baseline = {item.object_id: item for item in visibility.baseline_metrics}
    for object_id in (key.first_object_id, key.second_object_id):
        after = metrics.get((cell_id, object_id))
        before = baseline.get(object_id)
        if after is None or before is None:
            return False
        if not _metric_passes(after, constraint):
            return False
    return True


def _metric_passes(metric: VisibilityCellMetricBoundsV2_9, constraint: object) -> bool:
    return (
        metric.visible_fraction_lower
        >= Fraction.from_float(constraint.minimum_visible_fraction)
        and metric.image_area_fraction_lower
        >= Fraction.from_float(constraint.minimum_image_area_fraction)
        and metric.truncated_fraction_upper
        <= Fraction.from_float(constraint.maximum_truncated_fraction)
    )


def _centroid(
    object_: CanonicalObjectV2_2, geometry: GeometryInstanceV2_2
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        Fraction.from_float(object_.pose.world_from_object.translation.x),
        Fraction.from_float(object_.pose.world_from_object.translation.y),
        Fraction.from_float(
            object_.pose.world_from_object.translation.z
            + geometry.anchor_from_geometry.translation.z
        ),
    )


def _interval_gap(a0: Fraction, a1: Fraction, b0: Fraction, b1: Fraction) -> Fraction:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return Fraction()


def _divide_positive(
    numerator: _IntervalV2,
    denominator: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    if denominator[0] <= 0:
        raise _UnsupportedRelationDamageV2_9(
            "UNSUPPORTED_MODEL:RELATION_DAMAGE_DEPTH_V2_9"
        )
    budget.consume(2)
    reciprocal = (Fraction(1) / denominator[1], Fraction(1) / denominator[0])
    products = tuple(a * b for a in numerator for b in reciprocal)
    return min(products), max(products)


def _scale_exact(
    interval: _IntervalV2, scalar: Fraction, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    values = interval[0] * scalar, interval[1] * scalar
    return min(values), max(values)


def _add_exact(
    interval: _IntervalV2, scalar: Fraction, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    return interval[0] + scalar, interval[1] + scalar


def _subtract(
    left: _IntervalV2, right: _IntervalV2, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    return left[0] - right[1], left[1] - right[0]


def _copy_cell(value: RelationDamageCellV2_9) -> RelationDamageCellV2_9:
    if type(value) is not RelationDamageCellV2_9:
        raise TypeError("relation-damage cell has the wrong exact type")
    return RelationDamageCellV2_9(cell=value.cell, vector=value.vector)


def _copy_stage(
    value: ContinuousYawRelationDamageStageV2_9,
) -> ContinuousYawRelationDamageStageV2_9:
    return ContinuousYawRelationDamageStageV2_9(
        semantic_problem_sha256=value.semantic_problem_sha256,
        candidate_problem_sha256=value.candidate_problem_sha256,
        camera_context_sha256=value.camera_context_sha256,
        visibility_stage_sha256=value.visibility_stage_sha256,
        cells=value.cells,
        resource_usage=value.resource_usage,
    )


def _copy_usage(
    value: MultiObstacleStrictConvexCandidateResourceUsageV2,
) -> MultiObstacleStrictConvexCandidateResourceUsageV2:
    if type(value) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("relation-damage usage has the wrong exact type")
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=value.domain_operations,
        so2_atomic_steps=value.so2_atomic_steps,
        candidate_cells=value.candidate_cells,
    )


def _usage(
    atomic: SO2AtomicBudgetV2, domain: StrictConvexIntersectionBudgetV2
) -> MultiObstacleStrictConvexCandidateResourceUsageV2:
    return MultiObstacleStrictConvexCandidateResourceUsageV2(
        domain_operations=domain.domain_operations_used,
        so2_atomic_steps=atomic.used,
        candidate_cells=domain.candidate_cells_used,
    )


def _require_digest(label: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(
    kind: ContinuousYawRelationDamageKindV2_9, finding: str
) -> ContinuousYawRelationDamageOutcomeV2_9:
    return ContinuousYawRelationDamageOutcomeV2_9(kind=kind, finding_codes=(finding,))


__all__ = (
    "ContinuousYawRelationDamageKindV2_9",
    "ContinuousYawRelationDamageOutcomeV2_9",
    "ContinuousYawRelationDamageStageV2_9",
    "RelationDamageCellV2_9",
    "compile_relation_damage_bounds_v2_9",
    "evaluate_relation_damage_point_v2_9",
)
