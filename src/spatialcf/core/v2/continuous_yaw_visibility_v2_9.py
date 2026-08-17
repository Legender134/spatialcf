"""Camera-correct hard visibility and metric bounds for solver v2.9."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    UprightCameraContextV2_9,
    bound_world_point_in_upright_camera_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    DirectionalTargetCandidateStageV2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _copy_stage as _copy_target_stage,
)
from spatialcf.core.v2.convex_translation_domain import (
    _directed_world_corner_boxes,
)
from spatialcf.core.v2.multi_obstacle_strict_convex_candidate_domain import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _require_finding_codes,
)
from spatialcf.core.v2.oriented_upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.core.v2.projected_bounding_box_visibility import (
    projected_bounding_box_area_fraction_lower_bound_v2_9,
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
    StrictConvexIntersectionComplexV2,
)
from spatialcf.core.v2.visibility_domain import _visibility_semantics_findings
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec3V2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    OccluderSoundnessPolicyV2,
    VisibilityConstraintV2,
    VisibilityMaskPolicyV2,
    VisibilityMetricFormulaV2,
    VisibilityMetricKindV2,
)
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
from spatialcf.domain.v2.scene import BaselineObservationV2

_STAGE_HASH_DOMAIN_V2_9 = b"spatialcf.continuous-yaw-visibility-stage.v2.9\0"
_IntervalV2 = tuple[Fraction, Fraction]


class ContinuousYawVisibilityKindV2_9(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class _ProjectionClassV2_9(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class VisibilityCellMetricBoundsV2_9:
    cell_id: str
    object_id: str
    visible_fraction_lower: Fraction
    visible_fraction_upper: Fraction
    image_area_fraction_lower: Fraction
    image_area_fraction_upper: Fraction
    truncated_fraction_lower: Fraction
    truncated_fraction_upper: Fraction

    def __post_init__(self) -> None:
        for label, value in (("cell_id", self.cell_id), ("object_id", self.object_id)):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        _require_metric_intervals(self)


@dataclass(frozen=True, slots=True)
class VisibilityBaselineMetricBoundsV2_9:
    object_id: str
    visible_fraction: _IntervalV2
    image_area_fraction: _IntervalV2
    truncated_fraction: _IntervalV2

    def __post_init__(self) -> None:
        if type(self.object_id) is not str or not self.object_id.strip():
            raise ValueError("object_id must be a non-blank exact string")
        for interval in (
            self.visible_fraction,
            self.image_area_fraction,
            self.truncated_fraction,
        ):
            _require_unit_interval(interval)


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityStageV2_9:
    semantic_problem_sha256: str
    candidate_problem_sha256: str
    camera_context_sha256: str
    compiler_config_sha256: str
    upstream_t15_artifact_sha256: str
    upstream_target_stage_sha256: str
    subject_id: str
    visibility_constraint_id: str
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    outer_cell_metrics: tuple[VisibilityCellMetricBoundsV2_9, ...]
    baseline_metrics: tuple[VisibilityBaselineMetricBoundsV2_9, ...]
    fixed_object_ids: tuple[str, ...]
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]
    unsat_prefix_eligible: bool

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("candidate_problem_sha256", self.candidate_problem_sha256),
            ("camera_context_sha256", self.camera_context_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t15_artifact_sha256", self.upstream_t15_artifact_sha256),
            ("upstream_target_stage_sha256", self.upstream_target_stage_sha256),
        ):
            _require_digest(label, digest)
        for label, value in (
            ("subject_id", self.subject_id),
            ("visibility_constraint_id", self.visibility_constraint_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        inner = _copy_intersection_complex(self.inner_allowed)
        outer = _copy_intersection_complex(self.outer_allowed)
        if inner.universe != outer.universe:
            raise ValueError("visibility inner and outer require one exact universe")
        if not all(outer.contains_point(cell.strict_witness) for cell in inner.cells):
            raise ValueError("visibility inner witness escaped its outer domain")
        metrics = tuple(_copy_cell_metric(item) for item in self.outer_cell_metrics)
        if metrics != tuple(
            sorted(metrics, key=lambda item: (item.cell_id, item.object_id))
        ):
            raise ValueError("visibility cell metrics must be canonically sorted")
        baseline = tuple(_copy_baseline_metric(item) for item in self.baseline_metrics)
        if baseline != tuple(sorted(baseline, key=lambda item: item.object_id)):
            raise ValueError("baseline metrics must be canonically sorted")
        query_ids = tuple(item.object_id for item in baseline)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("baseline metrics must have unique object IDs")
        expected_keys = {
            (cell.cell_id, object_id) for cell in outer.cells for object_id in query_ids
        }
        if {(item.cell_id, item.object_id) for item in metrics} != expected_keys:
            raise ValueError("visibility metrics do not close outer cells and queries")
        fixed = _canonical_ids(self.fixed_object_ids)
        if self.subject_id in fixed or not set(fixed) <= set(query_ids):
            raise ValueError("fixed visibility IDs do not close the query universe")
        usage = _copy_usage(self.resource_usage)
        remaining = _canonical_ids(self.remaining_constraint_ids)
        if self.visibility_constraint_id in remaining:
            raise ValueError("compiled visibility remained in the suffix")
        if type(self.unsat_prefix_eligible) is not bool:
            raise TypeError("unsat_prefix_eligible must be an exact bool")
        if self.unsat_prefix_eligible is not (not outer.cells and not remaining):
            raise ValueError("visibility UNSAT eligibility is not structurally closed")
        object.__setattr__(self, "inner_allowed", inner)
        object.__setattr__(self, "outer_allowed", outer)
        object.__setattr__(self, "outer_cell_metrics", metrics)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "fixed_object_ids", fixed)
        object.__setattr__(self, "resource_usage", usage)
        object.__setattr__(self, "remaining_constraint_ids", remaining)

    @property
    def stage_sha256(self) -> str:
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2_9 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityOutcomeV2_9:
    kind: ContinuousYawVisibilityKindV2_9
    stage: ContinuousYawVisibilityStageV2_9 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawVisibilityKindV2_9:
            raise TypeError("kind must be an exact visibility kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ContinuousYawVisibilityKindV2_9.STAGE:
            if type(self.stage) is not ContinuousYawVisibilityStageV2_9:
                raise ValueError("STAGE requires one exact visibility stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("visibility failure requires findings and no stage")


@dataclass(frozen=True, slots=True)
class _ProjectionGeometryV2_9:
    object_id: str
    moving: bool
    box: OrientedUprightBoxBoundsV2
    corner_boxes: tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...
    ]
    world_z: _IntervalV2
    inscribed_diameter_xy: Fraction
    height: Fraction


@dataclass(frozen=True, slots=True)
class _MetricValuesV2_9:
    visible_fraction: _IntervalV2
    image_area_fraction: _IntervalV2
    truncated_fraction: _IntervalV2


@dataclass(frozen=True, slots=True)
class _ObjectClassificationV2_9:
    kind: _ProjectionClassV2_9
    metrics: _MetricValuesV2_9 | None


class _UnsupportedVisibilityV2_9(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _InvalidVisibilityInputV2_9(ValueError):
    pass


def compile_continuous_yaw_visibility_v2_9(
    problem: SemanticProblemV2_3,
    target_stage: DirectionalTargetCandidateStageV2_9,
    camera: UprightCameraContextV2_9,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawVisibilityOutcomeV2_9:
    """Prove all visibility queries against the same directed camera context."""

    try:
        checked_problem, checked_target, checked_camera = _strict_inputs(
            problem,
            target_stage,
            camera,
            atomic_budget,
            intersection_budget,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            constraint, geometries, baseline, image_area_formula = (
                _extract_visibility_subset(
                    checked_problem,
                    checked_target,
                    checked_camera,
                    intersection_budget,
                )
            )
            all_cells = checked_target.inner_allowed.cells + (
                checked_target.outer_allowed.cells
            )
            intersection_budget.consume_domain(
                12
                + len(geometries) * 8
                + sum(
                    24 + len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for cell in all_cells
                )
            )
            projection_by_id = {
                item[0].object_id: _compile_projection_geometry(
                    item[0], item[1], atomic_budget
                )
                for item in geometries
            }
            fixed = tuple(
                sorted(
                    object_id
                    for object_id, geometry in projection_by_id.items()
                    if not geometry.moving
                )
            )
            baseline_by_id = {item.object_id: item for item in baseline}
            fixed_results = {
                object_id: _reconcile_fixed_metrics(
                    _classify_object(
                        geometry,
                        (Fraction(), Fraction()),
                        (Fraction(), Fraction()),
                        checked_camera,
                        constraint,
                        image_area_formula,
                        atomic_budget,
                    ),
                    baseline_by_id[object_id],
                )
                for object_id, geometry in projection_by_id.items()
                if not geometry.moving
            }
            inner_cells = tuple(
                cell
                for cell in checked_target.inner_allowed.cells
                if _classify_cell(
                    cell,
                    projection_by_id,
                    fixed_results,
                    checked_camera,
                    constraint,
                    image_area_formula,
                    atomic_budget,
                )[0]
                is _ProjectionClassV2_9.PASS
            )
            outer_results = tuple(
                (
                    cell,
                    _classify_cell(
                        cell,
                        projection_by_id,
                        fixed_results,
                        checked_camera,
                        constraint,
                        image_area_formula,
                        atomic_budget,
                    ),
                )
                for cell in checked_target.outer_allowed.cells
            )
            outer_cells = tuple(
                cell
                for cell, (kind, _) in outer_results
                if kind is not _ProjectionClassV2_9.FAIL
            )
            metrics = tuple(
                VisibilityCellMetricBoundsV2_9(
                    cell_id=cell.cell_id,
                    object_id=object_id,
                    visible_fraction_lower=values.visible_fraction[0],
                    visible_fraction_upper=values.visible_fraction[1],
                    image_area_fraction_lower=values.image_area_fraction[0],
                    image_area_fraction_upper=values.image_area_fraction[1],
                    truncated_fraction_lower=values.truncated_fraction[0],
                    truncated_fraction_upper=values.truncated_fraction[1],
                )
                for cell, (kind, by_object) in outer_results
                if kind is not _ProjectionClassV2_9.FAIL
                for object_id, values in sorted(by_object.items())
            )
            intersection_budget.consume_candidate_cells(
                len(inner_cells) + len(outer_cells)
            )
            inner = _filtered_complex(checked_target.inner_allowed, inner_cells)
            outer = _filtered_complex(checked_target.outer_allowed, outer_cells)
            remaining = tuple(
                item
                for item in checked_target.remaining_constraint_ids
                if item != constraint.constraint_id
            )
            intersection_budget.consume_domain(
                18
                + len(metrics) * 8
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (inner, outer)
                    for cell in complex_.cells
                )
            )
            stage = ContinuousYawVisibilityStageV2_9(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                candidate_problem_sha256=checked_target.candidate_problem_sha256,
                camera_context_sha256=checked_camera.context_sha256,
                compiler_config_sha256=checked_target.compiler_config_sha256,
                upstream_t15_artifact_sha256=(
                    checked_target.upstream_t15_artifact_sha256
                ),
                upstream_target_stage_sha256=checked_target.stage_sha256,
                subject_id=checked_target.subject_id,
                visibility_constraint_id=constraint.constraint_id,
                inner_allowed=inner,
                outer_allowed=outer,
                outer_cell_metrics=metrics,
                baseline_metrics=baseline,
                fixed_object_ids=fixed,
                resource_usage=_usage(atomic_budget, intersection_budget),
                remaining_constraint_ids=remaining,
                unsat_prefix_eligible=not outer.cells and not remaining,
            )
        return ContinuousYawVisibilityOutcomeV2_9(
            kind=ContinuousYawVisibilityKindV2_9.STAGE,
            stage=stage,
        )
    except _InvalidVisibilityInputV2_9:
        return _failure(
            ContinuousYawVisibilityKindV2_9.INVALID_INPUT,
            "INVALID_INPUT:CONTINUOUS_YAW_VISIBILITY_V2_9",
        )
    except _UnsupportedVisibilityV2_9 as error:
        return _failure(
            ContinuousYawVisibilityKindV2_9.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            ContinuousYawVisibilityKindV2_9.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_VISIBILITY_V2_9",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ContinuousYawVisibilityKindV2_9.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_VISIBILITY_V2_9",
        )


def _strict_inputs(
    problem: object,
    target_stage: object,
    camera: object,
    atomic_budget: object,
    intersection_budget: object,
) -> tuple[
    SemanticProblemV2_3,
    DirectionalTargetCandidateStageV2_9,
    UprightCameraContextV2_9,
]:
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be an exact strict budget")
    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("problem must be an exact SemanticProblemV2_3")
    if type(target_stage) is not DirectionalTargetCandidateStageV2_9:
        raise TypeError("target_stage must be the exact v2.9 directional stage")
    if type(camera) is not UprightCameraContextV2_9:
        raise TypeError("camera must be the exact v2.9 camera context")
    raw_usage = target_stage.resource_usage
    if (
        atomic_budget.used != raw_usage.so2_atomic_steps
        or intersection_budget.domain_operations_used != raw_usage.domain_operations
        or intersection_budget.candidate_cells_used != raw_usage.candidate_cells
    ):
        raise ValueError("live visibility ledgers must continue exact target usage")
    intersection_budget.consume_domain(12)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_problem = SemanticProblemV2_3.model_validate(
                problem.model_dump(mode="python", warnings="error"), strict=True
            )
            checked_target = _copy_target_stage(target_stage)
            checked_camera = UprightCameraContextV2_9(
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
    except (ValidationError, PydanticSerializationError) as error:
        raise _InvalidVisibilityInputV2_9 from error
    if (
        checked_target.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
    ):
        raise ValueError("directional target does not close the visibility problem")
    if checked_target.camera_context_sha256 != checked_camera.context_sha256:
        raise ValueError("directional target does not close the camera context")
    visibility_ids = tuple(
        item.constraint_id
        for item in checked_problem.constraints.visibility_constraints
    )
    if checked_target.remaining_constraint_ids != visibility_ids:
        raise ValueError(
            "target suffix must contain exactly the visibility constraints"
        )
    return checked_problem, checked_target, checked_camera


def _extract_visibility_subset(
    problem: SemanticProblemV2_3,
    target_stage: DirectionalTargetCandidateStageV2_9,
    camera: UprightCameraContextV2_9,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[
    VisibilityConstraintV2,
    tuple[tuple[CanonicalObjectV2_2, GeometryInstanceV2_2], ...],
    tuple[VisibilityBaselineMetricBoundsV2_9, ...],
    VisibilityMetricFormulaV2,
]:
    if (
        len(problem.constraints.visibility_constraints) != 1
        or problem.numeric_policy != NumericPolicyV2()
    ):
        raise _UnsupportedVisibilityV2_9(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_POLICY_V2_9"
        )
    constraint = problem.constraints.visibility_constraints[0]
    target = problem.constraints.target_relation
    if (
        constraint.constraint_id != target_stage.remaining_constraint_ids[0]
        or constraint.camera_id != camera.camera_id
        or constraint.threshold_boundary_policy is not BoundaryPolicyV2.CLOSED
        or constraint.mask_policy is not VisibilityMaskPolicyV2.FULL_OBJECT
        or constraint.occluder_soundness_policy
        is not OccluderSoundnessPolicyV2.EXACT_OR_OUTER_SHAPE_BOUND
        or constraint.occluder_geometry_ids
        or constraint.accepted_baseline_completeness != (FactCompletenessV2.EXACT,)
        or _visibility_semantics_findings(
            problem,
            constraint,
            supported_image_area_formulas=(
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA,
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA,
            ),
        )
    ):
        raise _UnsupportedVisibilityV2_9(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_POLICY_V2_9"
        )
    if target_stage.requires_both_visible and (
        not {target.subject_id, target.reference_id} <= set(constraint.query_object_ids)
        or constraint.minimum_visible_fraction <= 0.0
        or constraint.minimum_image_area_fraction <= 0.0
    ):
        raise _UnsupportedVisibilityV2_9(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_TARGET_GATE_V2_9"
        )
    objects = _exact_values(problem.scene.objects, "OBJECTS", budget)
    geometries = _exact_values(
        problem.scene.geometry_instances, "GEOMETRY_INSTANCES", budget
    )
    if any(item.role is GeometryRoleV2.OCCLUDER for item in geometries):
        raise _UnsupportedVisibilityV2_9(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_OCCLUDERS_V2_9"
        )
    object_by_id = {item.object_id: item for item in objects}
    pairs = []
    for object_id in constraint.query_object_ids:
        try:
            object_ = object_by_id[object_id]
        except KeyError as error:
            raise RuntimeError("visibility graph lost a query object") from error
        visual = tuple(
            item
            for item in geometries
            if item.owner_object_id == object_id and item.role is GeometryRoleV2.VISUAL
        )
        if len(visual) != 1:
            raise _UnsupportedVisibilityV2_9(
                "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISUAL_GEOMETRY_CARDINALITY_V2_9"
            )
        geometry = visual[0]
        anchor = geometry.anchor_from_geometry
        if (
            type(object_) is not CanonicalObjectV2_2
            or type(geometry) is not GeometryInstanceV2_2
            or geometry.approximation is not GeometryApproximationV2.EXACT
            or geometry.uncertainty != UncertaintyBudgetV2()
            or type(geometry.shape) is not UprightBox3DV2
            or (anchor.translation.x, anchor.translation.y) != (0.0, 0.0)
        ):
            raise _UnsupportedVisibilityV2_9(
                "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISUAL_GEOMETRY_V2_9"
            )
        if object_.movable is not (object_id == target_stage.subject_id):
            raise _UnsupportedVisibilityV2_9(
                "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_MOVEMENT_V2_9"
            )
        pairs.append((object_, geometry))
    baseline = _baseline_bounds(problem, constraint, budget)
    image_area_formula = next(
        definition.formula
        for definition in problem.visibility_semantics.definitions
        if definition.kind is VisibilityMetricKindV2.IMAGE_AREA_FRACTION
    )
    return constraint, tuple(pairs), baseline, image_area_formula


def _exact_values(
    facts: object, label: str, budget: StrictConvexIntersectionBudgetV2
) -> tuple[object, ...]:
    if (
        getattr(facts, "availability", None) is not FactAvailabilityV2.KNOWN
        or getattr(facts, "completeness", None) is not FactCompletenessV2.EXACT
        or getattr(facts, "uncertainty", None) != UncertaintyBudgetV2()
        or type(getattr(facts, "values", None)) is not tuple
        or getattr(facts, "inner_values", None) is not None
        or getattr(facts, "outer_values", None) is not None
    ):
        raise _UnsupportedVisibilityV2_9(
            f"UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_{label}_V2_9"
        )
    values = facts.values
    budget.consume_domain(len(values) + 1)
    return values


def _baseline_bounds(
    problem: SemanticProblemV2_3,
    constraint: VisibilityConstraintV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[VisibilityBaselineMetricBoundsV2_9, ...]:
    observations = _exact_values(
        problem.scene.baseline_observations, "BASELINE_OBSERVATIONS", budget
    )
    by_key = {
        (
            item.object_id,
            item.camera_id,
            item.metric_definition_id,
            item.metric_definition_version,
        ): item
        for item in observations
        if type(item) is BaselineObservationV2
    }
    specs = (
        (
            "visible",
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            Fraction.from_float(constraint.minimum_visible_fraction),
            True,
        ),
        (
            "area",
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            Fraction.from_float(constraint.minimum_image_area_fraction),
            True,
        ),
        (
            "truncated",
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            Fraction.from_float(constraint.maximum_truncated_fraction),
            False,
        ),
    )
    budget.consume_domain(len(constraint.query_object_ids) * len(specs) + 2)
    result = []
    for object_id in constraint.query_object_ids:
        values: dict[str, _IntervalV2] = {}
        for label, definition_id, version, threshold, minimum in specs:
            observation = by_key.get(
                (object_id, constraint.camera_id, definition_id, version)
            )
            if observation is None:
                raise _UnsupportedVisibilityV2_9(
                    "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_BASELINE_V2_9"
                )
            interval = (
                Fraction.from_float(observation.normalized_lower_bound),
                Fraction.from_float(observation.normalized_upper_bound),
            )
            _require_unit_interval(interval)
            if not (interval[0] >= threshold if minimum else interval[1] <= threshold):
                raise _UnsupportedVisibilityV2_9(
                    "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_BASELINE_V2_9"
                )
            values[label] = interval
        result.append(
            VisibilityBaselineMetricBoundsV2_9(
                object_id=object_id,
                visible_fraction=values["visible"],
                image_area_fraction=values["area"],
                truncated_fraction=values["truncated"],
            )
        )
    return tuple(sorted(result, key=lambda item: item.object_id))


def _compile_projection_geometry(
    object_: CanonicalObjectV2_2,
    visual: GeometryInstanceV2_2,
    budget: SO2AtomicBudgetV2,
) -> _ProjectionGeometryV2_9:
    transform = object_.pose.world_from_object
    if type(transform) is not DirectedYawIntervalTransformV2_2:
        raise RuntimeError("continuous-yaw object lost its directed transform")
    anchor = visual.anchor_from_geometry
    shifted = DirectedYawIntervalTransformV2_2(
        translation=Vec3V2(
            x=transform.translation.x,
            y=transform.translation.y,
            z=transform.translation.z + anchor.translation.z,
        ),
        yaw_radians=transform.yaw_radians,
    )
    assert type(visual.shape) is UprightBox3DV2
    outcome = compile_oriented_upright_box_bounds_v2(
        shifted, visual.shape, atomic_budget=budget
    )
    if outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        raise SO2AtomicBudgetExhaustedV2
    if outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
        raise ArithmeticError("visibility object box reached a numeric gap")
    if outcome.kind is not SO2IntervalKindV2.EXACT or outcome.bounds is None:
        raise RuntimeError("supported visual box did not publish directed bounds")
    box = outcome.bounds
    return _ProjectionGeometryV2_9(
        object_id=object_.object_id,
        moving=object_.movable,
        box=box,
        corner_boxes=_directed_world_corner_boxes(box, budget),
        world_z=(box.center_z - box.half_extent_z, box.center_z + box.half_extent_z),
        inscribed_diameter_xy=2 * min(box.half_extent_x, box.half_extent_y),
        height=2 * box.half_extent_z,
    )


def _classify_cell(
    cell: StrictConvexIntersectionCellV2,
    geometries: dict[str, _ProjectionGeometryV2_9],
    fixed_results: dict[str, _ObjectClassificationV2_9],
    camera: UprightCameraContextV2_9,
    constraint: VisibilityConstraintV2,
    image_area_formula: VisibilityMetricFormulaV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[_ProjectionClassV2_9, dict[str, _MetricValuesV2_9]]:
    vertices = cell.closure_polygon.vertices_ccw
    delta_x = (min(item.x for item in vertices), max(item.x for item in vertices))
    delta_y = (min(item.y for item in vertices), max(item.y for item in vertices))
    results = {
        object_id: (
            _classify_object(
                geometry,
                delta_x,
                delta_y,
                camera,
                constraint,
                image_area_formula,
                budget,
            )
            if geometry.moving
            else fixed_results[object_id]
        )
        for object_id, geometry in geometries.items()
    }
    if any(item.kind is _ProjectionClassV2_9.FAIL for item in results.values()):
        return _ProjectionClassV2_9.FAIL, {}
    metrics = {}
    for object_id, result in results.items():
        if result.metrics is None:
            raise RuntimeError("kept visibility query lost its metric bounds")
        metrics[object_id] = result.metrics
    kind = (
        _ProjectionClassV2_9.PASS
        if all(item.kind is _ProjectionClassV2_9.PASS for item in results.values())
        else _ProjectionClassV2_9.AMBIGUOUS
    )
    return kind, metrics


def _classify_object(
    geometry: _ProjectionGeometryV2_9,
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    camera: UprightCameraContextV2_9,
    constraint: VisibilityConstraintV2,
    image_area_formula: VisibilityMetricFormulaV2,
    budget: SO2AtomicBudgetV2,
) -> _ObjectClassificationV2_9:
    projected_u: list[_IntervalV2] = []
    projected_v: list[_IntervalV2] = []
    depths: list[_IntervalV2] = []
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    for corner_x, corner_y in geometry.corner_boxes:
        corner_projected_u: tuple[list[_IntervalV2], list[_IntervalV2]] = ([], [])
        corner_projected_v: tuple[list[_IntervalV2], list[_IntervalV2]] = ([], [])
        for world_x in corner_x:
            for world_y in corner_y:
                for z_index, world_z in enumerate(geometry.world_z):
                    point = bound_world_point_in_upright_camera_v2_9(
                        camera,
                        world_xyz=(world_x, world_y, world_z),
                        delta_x=delta_x,
                        delta_y=delta_y,
                        atomic_budget=budget,
                    )
                    depths.append(point.z_camera)
                    if point.z_camera[0] <= 0:
                        continue
                    corner_projected_u[z_index].append(
                        _add_exact(
                            _scale_exact(
                                _divide_positive(
                                    point.x_camera, point.z_camera, budget
                                ),
                                fx,
                                budget,
                            ),
                            cx,
                            budget,
                        )
                    )
                    corner_projected_v[z_index].append(
                        _add_exact(
                            _scale_exact(
                                _divide_positive(
                                    point.y_camera, point.z_camera, budget
                                ),
                                fy,
                                budget,
                            ),
                            cy,
                            budget,
                        )
                    )
        for values_u, values_v in zip(
            corner_projected_u,
            corner_projected_v,
            strict=True,
        ):
            if values_u:
                projected_u.append(
                    _enclose_projected_corner_identity_v2_9(tuple(values_u))
                )
            if values_v:
                projected_v.append(
                    _enclose_projected_corner_identity_v2_9(tuple(values_v))
                )
    if not depths:
        raise RuntimeError("upright box did not expose camera-space corners")
    z_lower = min(item[0] for item in depths)
    z_upper = max(item[1] for item in depths)
    required = (
        constraint.minimum_visible_fraction > 0.0
        or constraint.minimum_image_area_fraction > 0.0
        or constraint.maximum_truncated_fraction < 1.0
    )
    if (
        z_upper < camera.near_clip_m or z_lower > camera.far_clip_m or z_upper <= 0
    ) and required:
        return _ObjectClassificationV2_9(_ProjectionClassV2_9.FAIL, None)
    if z_lower <= 0 or not projected_u or not projected_v:
        return _ambiguous_classification()
    min_u = min(item[0] for item in projected_u)
    max_u = max(item[1] for item in projected_u)
    min_v = min(item[0] for item in projected_v)
    max_v = max(item[1] for item in projected_v)
    left = Fraction(1, 2)
    right = Fraction(camera.width_px) - left
    top = Fraction(1, 2)
    bottom = Fraction(camera.height_px) - top
    outside = max_u <= left or min_u >= right or max_v <= top or min_v >= bottom
    if outside and required:
        return _ObjectClassificationV2_9(_ProjectionClassV2_9.FAIL, None)
    fully_contained = (
        z_lower >= camera.near_clip_m
        and z_upper <= camera.far_clip_m
        and min_u >= left
        and max_u <= right
        and min_v >= top
        and max_v <= bottom
    )
    center = bound_world_point_in_upright_camera_v2_9(
        camera,
        world_xyz=(geometry.box.center_x, geometry.box.center_y, geometry.box.center_z),
        delta_x=delta_x,
        delta_y=delta_y,
        atomic_budget=budget,
    )
    if not center.positive_depth:
        return _ambiguous_classification()
    budget.consume(6)
    area_lower = _image_area_fraction_lower_bound_v2_9(
        image_area_formula,
        projected_u=tuple(projected_u),
        projected_v=tuple(projected_v),
        focal_x_px=fx,
        focal_y_px=fy,
        inscribed_diameter_xy=geometry.inscribed_diameter_xy,
        height=geometry.height,
        center_depth_upper=center.z_camera[1],
        image_width_px=camera.width_px,
        image_height_px=camera.height_px,
    )
    metrics = _MetricValuesV2_9(
        visible_fraction=(Fraction(1), Fraction(1)),
        image_area_fraction=(area_lower, Fraction(1)),
        truncated_fraction=(Fraction(), Fraction()),
    )
    thresholds_pass = (
        metrics.visible_fraction[0]
        >= Fraction.from_float(constraint.minimum_visible_fraction)
        and metrics.image_area_fraction[0]
        >= Fraction.from_float(constraint.minimum_image_area_fraction)
        and metrics.truncated_fraction[1]
        <= Fraction.from_float(constraint.maximum_truncated_fraction)
    )
    if fully_contained and thresholds_pass:
        return _ObjectClassificationV2_9(_ProjectionClassV2_9.PASS, metrics)
    return _ambiguous_classification()


def _image_area_fraction_lower_bound_v2_9(
    formula: VisibilityMetricFormulaV2,
    *,
    projected_u: tuple[_IntervalV2, ...],
    projected_v: tuple[_IntervalV2, ...],
    focal_x_px: Fraction,
    focal_y_px: Fraction,
    inscribed_diameter_xy: Fraction,
    height: Fraction,
    center_depth_upper: Fraction,
    image_width_px: int,
    image_height_px: int,
) -> Fraction:
    if (
        formula
        is VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA
    ):
        return min(
            Fraction(1),
            focal_x_px
            * focal_y_px
            * inscribed_diameter_xy
            * height
            / (
                center_depth_upper
                * center_depth_upper
                * Fraction(image_width_px)
                * Fraction(image_height_px)
            ),
        )
    if (
        formula
        is VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA
    ):
        return min(
            Fraction(1),
            projected_bounding_box_area_fraction_lower_bound_v2_9(
                projected_u=projected_u,
                projected_v=projected_v,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
            ),
        )
    raise RuntimeError("unsupported image-area visibility formula reached projection")


def _enclose_projected_corner_identity_v2_9(
    values: tuple[_IntervalV2, ...],
) -> _IntervalV2:
    if not values:
        raise RuntimeError("projected corner identity lost every enclosure")
    return min(item[0] for item in values), max(item[1] for item in values)


def _ambiguous_classification() -> _ObjectClassificationV2_9:
    return _ObjectClassificationV2_9(
        _ProjectionClassV2_9.AMBIGUOUS,
        _MetricValuesV2_9(
            visible_fraction=(Fraction(), Fraction(1)),
            image_area_fraction=(Fraction(), Fraction(1)),
            truncated_fraction=(Fraction(), Fraction(1)),
        ),
    )


def _reconcile_fixed_metrics(
    result: _ObjectClassificationV2_9,
    baseline: VisibilityBaselineMetricBoundsV2_9,
) -> _ObjectClassificationV2_9:
    """Close fixed-object metrics with the accepted exact baseline evidence.

    A fixed object, fixed camera, and empty occluder roster cannot change after
    the subject translation.  The source contract already checks that every
    baseline interval passes the visibility thresholds.  Therefore an
    ambiguous conservative box projection can be closed by that observation;
    only a proven geometric FAIL remains a failure.
    """

    if result.kind is _ProjectionClassV2_9.FAIL:
        return result
    if result.kind is _ProjectionClassV2_9.AMBIGUOUS:
        return _ObjectClassificationV2_9(
            kind=_ProjectionClassV2_9.PASS,
            metrics=_MetricValuesV2_9(
                visible_fraction=baseline.visible_fraction,
                image_area_fraction=baseline.image_area_fraction,
                truncated_fraction=baseline.truncated_fraction,
            ),
        )
    if result.metrics is None:
        raise RuntimeError("passing fixed projection lost its metric bounds")
    metrics = result.metrics
    return _ObjectClassificationV2_9(
        kind=_ProjectionClassV2_9.PASS,
        metrics=_MetricValuesV2_9(
            visible_fraction=_interval_hull(
                metrics.visible_fraction, baseline.visible_fraction
            ),
            image_area_fraction=_interval_hull(
                metrics.image_area_fraction, baseline.image_area_fraction
            ),
            truncated_fraction=_interval_hull(
                metrics.truncated_fraction, baseline.truncated_fraction
            ),
        ),
    )


def _interval_hull(left: _IntervalV2, right: _IntervalV2) -> _IntervalV2:
    _require_unit_interval(left)
    _require_unit_interval(right)
    return min(left[0], right[0]), max(left[1], right[1])


def _divide_positive(
    numerator: _IntervalV2,
    denominator: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    if denominator[0] <= 0:
        raise ArithmeticError("visibility projection denominator is not positive")
    budget.consume(2)
    reciprocal = (Fraction(1) / denominator[1], Fraction(1) / denominator[0])
    products = tuple(left * right for left in numerator for right in reciprocal)
    return min(products), max(products)


def _scale_exact(
    interval: _IntervalV2, scalar: Fraction, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    values = (interval[0] * scalar, interval[1] * scalar)
    return (min(values), max(values))


def _add_exact(
    interval: _IntervalV2, scalar: Fraction, budget: SO2AtomicBudgetV2
) -> _IntervalV2:
    budget.consume()
    return interval[0] + scalar, interval[1] + scalar


def _filtered_complex(
    source: StrictConvexIntersectionComplexV2,
    cells: tuple[StrictConvexIntersectionCellV2, ...],
) -> StrictConvexIntersectionComplexV2:
    return StrictConvexIntersectionComplexV2(
        cells=cells, universe=source.universe, topology=source.topology
    )


def _require_metric_intervals(value: VisibilityCellMetricBoundsV2_9) -> None:
    for lower, upper in (
        (value.visible_fraction_lower, value.visible_fraction_upper),
        (value.image_area_fraction_lower, value.image_area_fraction_upper),
        (value.truncated_fraction_lower, value.truncated_fraction_upper),
    ):
        _require_unit_interval((lower, upper))


def _require_unit_interval(interval: _IntervalV2) -> None:
    if type(interval) is not tuple or len(interval) != 2:
        raise TypeError("visibility intervals must be exact pairs")
    if any(type(item) is not Fraction for item in interval):
        raise TypeError("visibility interval endpoints must be exact Fractions")
    if interval[0] < 0 or interval[0] > interval[1] or interval[1] > 1:
        raise ValueError("visibility interval must lie in [0, 1]")


def _copy_cell_metric(
    value: VisibilityCellMetricBoundsV2_9,
) -> VisibilityCellMetricBoundsV2_9:
    if type(value) is not VisibilityCellMetricBoundsV2_9:
        raise TypeError("visibility cell metric has the wrong exact type")
    return VisibilityCellMetricBoundsV2_9(
        cell_id=value.cell_id,
        object_id=value.object_id,
        visible_fraction_lower=value.visible_fraction_lower,
        visible_fraction_upper=value.visible_fraction_upper,
        image_area_fraction_lower=value.image_area_fraction_lower,
        image_area_fraction_upper=value.image_area_fraction_upper,
        truncated_fraction_lower=value.truncated_fraction_lower,
        truncated_fraction_upper=value.truncated_fraction_upper,
    )


def _copy_baseline_metric(
    value: VisibilityBaselineMetricBoundsV2_9,
) -> VisibilityBaselineMetricBoundsV2_9:
    if type(value) is not VisibilityBaselineMetricBoundsV2_9:
        raise TypeError("visibility baseline metric has the wrong exact type")
    return VisibilityBaselineMetricBoundsV2_9(
        object_id=value.object_id,
        visible_fraction=value.visible_fraction,
        image_area_fraction=value.image_area_fraction,
        truncated_fraction=value.truncated_fraction,
    )


def _copy_usage(
    value: MultiObstacleStrictConvexCandidateResourceUsageV2,
) -> MultiObstacleStrictConvexCandidateResourceUsageV2:
    if type(value) is not MultiObstacleStrictConvexCandidateResourceUsageV2:
        raise TypeError("visibility usage has the wrong exact type")
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


def _copy_stage(
    value: ContinuousYawVisibilityStageV2_9,
) -> ContinuousYawVisibilityStageV2_9:
    return ContinuousYawVisibilityStageV2_9(
        semantic_problem_sha256=value.semantic_problem_sha256,
        candidate_problem_sha256=value.candidate_problem_sha256,
        camera_context_sha256=value.camera_context_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        upstream_t15_artifact_sha256=value.upstream_t15_artifact_sha256,
        upstream_target_stage_sha256=value.upstream_target_stage_sha256,
        subject_id=value.subject_id,
        visibility_constraint_id=value.visibility_constraint_id,
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        outer_cell_metrics=value.outer_cell_metrics,
        baseline_metrics=value.baseline_metrics,
        fixed_object_ids=value.fixed_object_ids,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
        unsat_prefix_eligible=value.unsat_prefix_eligible,
    )


def _canonical_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise ValueError("visibility IDs must be exact non-blank strings")
    if len(value) != len(set(value)):
        raise ValueError("visibility IDs must be unique")
    return tuple(sorted(value))


def _require_digest(label: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(
    kind: ContinuousYawVisibilityKindV2_9, finding: str
) -> ContinuousYawVisibilityOutcomeV2_9:
    return ContinuousYawVisibilityOutcomeV2_9(kind=kind, finding_codes=(finding,))


__all__ = (
    "ContinuousYawVisibilityKindV2_9",
    "ContinuousYawVisibilityOutcomeV2_9",
    "ContinuousYawVisibilityStageV2_9",
    "VisibilityBaselineMetricBoundsV2_9",
    "VisibilityCellMetricBoundsV2_9",
    "compile_continuous_yaw_visibility_v2_9",
)
