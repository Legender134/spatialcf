"""Exact rational safety-slack bounds over one rectilinear delta cell."""

from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from spatialcf.core.v2.objective_numeric import (
    OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2,
    ExactPublishedIntervalV2,
    ObjectiveNumericKindV2,
    SafetyRawComponentIntervalV2,
    compile_constraint_slack_bounds_v2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
    UnsupportedRectRegionErrorV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearTopologyV2,
    union_rectilinear_regions_v2,
)
from spatialcf.domain.v2.base import (
    CanonicalId,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    QuaternionV2,
    UncertaintyBudgetV2,
    V2Model,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    CollisionClearanceMetricV2,
    CollisionConstraintV2,
    MeasurementComparatorV2,
    PositionRegionInterpretationV2,
    RegionAggregationV2,
    RelationMeasurementV2,
    RelationV2,
    SupportConstraintV2,
    VisibilityConstraintV2,
)
from spatialcf.domain.v2.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.objective import (
    ConstraintSafetyTargetV2,
    SafetyComponentKindV2,
    SafetySlackUnitV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import (
    CanonicalObjectV2,
    RegionBoundaryPolicyV2,
    SupportSurfaceFactV2,
)

_MAX_FINITE_BINARY64_BITS = 0x7FEF_FFFF_FFFF_FFFF
_DIRECTED_SQRT_ATOMIC_STEPS_V2 = 64
_CANONICAL_ID_ADAPTER_V2 = TypeAdapter(CanonicalId)


class ObjectiveSafetyBoundsKindV2(StrEnum):
    EXACT = "EXACT"
    UNSUPPORTED = "UNSUPPORTED"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE = "RESOURCE"
    EMPTY = "EMPTY"


@dataclass(frozen=True, slots=True)
class ConstraintSlackBoundsV2:
    """Auditable raw component bounds and their normalized scalar minimum."""

    constraint_id: str
    raw_components: tuple[SafetyRawComponentIntervalV2, ...]
    normalized_slack: ExactPublishedIntervalV2

    def __post_init__(self) -> None:
        try:
            checked_id = _CANONICAL_ID_ADAPTER_V2.validate_python(
                self.constraint_id,
                strict=True,
            )
        except ValidationError as error:
            raise TypeError("constraint_id must be a CanonicalId") from error
        if type(self.raw_components) is not tuple or not self.raw_components:
            raise TypeError("raw_components must be a non-empty exact tuple")
        checked_components: list[SafetyRawComponentIntervalV2] = []
        for component in self.raw_components:
            if type(component) is not SafetyRawComponentIntervalV2:
                raise TypeError("raw component has the wrong type")
            checked_components.append(
                SafetyRawComponentIntervalV2(
                    kind=component.kind,
                    unit=component.unit,
                    raw_lower=component.raw_lower,
                    raw_upper=component.raw_upper,
                )
            )
        kinds = tuple(item.kind for item in checked_components)
        if len(kinds) != len(set(kinds)):
            raise ValueError("raw component kinds must be unique")
        if type(self.normalized_slack) is not ExactPublishedIntervalV2:
            raise TypeError("normalized_slack must be an ExactPublishedIntervalV2")
        checked_slack = ExactPublishedIntervalV2(
            exact_lower=self.normalized_slack.exact_lower,
            exact_upper=self.normalized_slack.exact_upper,
            lower_bound=self.normalized_slack.lower_bound,
            upper_bound=self.normalized_slack.upper_bound,
        )
        object.__setattr__(self, "constraint_id", checked_id)
        object.__setattr__(self, "raw_components", tuple(checked_components))
        object.__setattr__(self, "normalized_slack", checked_slack)


@dataclass(frozen=True, slots=True)
class ObjectiveSafetyBoundsOutcomeV2:
    kind: ObjectiveSafetyBoundsKindV2
    constraint_bounds: tuple[ConstraintSlackBoundsV2, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ObjectiveSafetyBoundsKindV2:
            raise TypeError("kind must be an ObjectiveSafetyBoundsKindV2")
        if type(self.constraint_bounds) is not tuple or any(
            type(item) is not ConstraintSlackBoundsV2 for item in self.constraint_bounds
        ):
            raise TypeError("constraint_bounds must be an exact tuple")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise TypeError("finding_codes must be exact non-blank strings")
        checked_bounds = tuple(
            ConstraintSlackBoundsV2(
                constraint_id=item.constraint_id,
                raw_components=item.raw_components,
                normalized_slack=item.normalized_slack,
            )
            for item in self.constraint_bounds
        )
        ids = tuple(item.constraint_id for item in checked_bounds)
        if len(ids) != len(set(ids)):
            raise ValueError("constraint bounds must have unique IDs")
        object.__setattr__(
            self,
            "constraint_bounds",
            tuple(sorted(checked_bounds, key=lambda item: item.constraint_id)),
        )
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.kind is ObjectiveSafetyBoundsKindV2.EXACT:
            if not self.constraint_bounds or self.finding_codes:
                raise ValueError("EXACT requires only non-empty constraint bounds")
            return
        if self.constraint_bounds:
            raise ValueError("non-EXACT outcomes cannot carry partial bounds")
        if self.kind is ObjectiveSafetyBoundsKindV2.EMPTY:
            if self.finding_codes:
                raise ValueError("EMPTY cannot carry findings")
            return
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} requires a finding")


ObjectiveSafetyBoundsCompileOutcomeV2 = ObjectiveSafetyBoundsOutcomeV2


@dataclass(frozen=True, slots=True)
class _WorldBoxV2:
    min_x: Fraction
    min_y: Fraction
    min_z: Fraction
    max_x: Fraction
    max_y: Fraction
    max_z: Fraction


class _UnsupportedV2(RuntimeError):
    def __init__(self, *finding_codes: str) -> None:
        self.finding_codes = tuple(sorted(set(finding_codes)))
        super().__init__("|".join(self.finding_codes))


class _NumericGapV2(RuntimeError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


class _EmptySemanticV2(RuntimeError):
    pass


ModelT = TypeVar("ModelT", bound=V2Model)


def compile_objective_safety_bounds_v2(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> ObjectiveSafetyBoundsOutcomeV2:
    """Compute every normalized hard-constraint slack over an exact cell."""

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    checked_problem = _strict_problem(problem)
    if isinstance(checked_problem, ObjectiveSafetyBoundsOutcomeV2):
        return checked_problem
    checked_cell = _strict_cell(cell, budget)
    if isinstance(checked_cell, ObjectiveSafetyBoundsOutcomeV2):
        return checked_cell
    validation = union_rectilinear_regions_v2(
        checked_cell,
        checked_cell,
        atomic_budget=budget,
    )
    if validation.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource(*validation.finding_codes)
    if validation.kind is not RectilinearOutcomeKindV2.EXACT:
        return _numeric(*validation.finding_codes)
    if validation.region is None:  # pragma: no cover - nested outcome invariant
        raise ValueError("EXACT cell validation omitted its region")
    checked_cell = validation.region
    if checked_cell.topology is RectilinearTopologyV2.EMPTY:
        return ObjectiveSafetyBoundsOutcomeV2(kind=ObjectiveSafetyBoundsKindV2.EMPTY)
    if not _numeric_policy_is_zero(checked_problem.numeric_policy):
        return _unsupported("UNSUPPORTED_SAFETY_BOUNDS:NUMERIC_POLICY")

    targets = checked_problem.objective.safety_margin.aggregation.targets
    compiled: list[ConstraintSlackBoundsV2] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            for target in targets:
                budget.consume(len(checked_cell.rectangles))
                components = _raw_components_for_target(
                    checked_problem,
                    checked_cell,
                    target,
                    budget,
                )
                for component in components:
                    _require_fraction_cap(component.raw_lower)
                    _require_fraction_cap(component.raw_upper)
                normalized = compile_constraint_slack_bounds_v2(target, components)
                if normalized.kind is ObjectiveNumericKindV2.NUMERIC_GAP:
                    raise _NumericGapV2(
                        normalized.finding_codes[0]
                        if normalized.finding_codes
                        else "NUMERIC_GAP:SAFETY_SLACK"
                    )
                if (
                    normalized.kind is not ObjectiveNumericKindV2.EXACT
                    or normalized.interval is None
                ):
                    raise ValueError(
                        "strict safety target and raw components failed closure"
                    )
                compiled.append(
                    ConstraintSlackBoundsV2(
                        constraint_id=target.constraint_id,
                        raw_components=components,
                        normalized_slack=normalized.interval,
                    )
                )
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource("RESOURCE_LIMIT:OBJECTIVE_SAFETY_ATOMIC_CELLS")
    except _UnsupportedV2 as error:
        return _unsupported(*error.finding_codes)
    except _NumericGapV2 as error:
        return _numeric(error.finding_code)
    except RuntimeWarning:
        return _numeric("NUMERIC_GAP:OBJECTIVE_SAFETY_BOUNDS")
    except _EmptySemanticV2:
        return ObjectiveSafetyBoundsOutcomeV2(kind=ObjectiveSafetyBoundsKindV2.EMPTY)

    expected_ids = tuple(target.constraint_id for target in targets)
    actual_ids = tuple(item.constraint_id for item in compiled)
    if actual_ids != expected_ids:
        raise ValueError("safety constraint compilation order drift")
    return ObjectiveSafetyBoundsOutcomeV2(
        kind=ObjectiveSafetyBoundsKindV2.EXACT,
        constraint_bounds=tuple(compiled),
    )


def _strict_problem(
    value: object,
) -> SemanticProblemV2 | ObjectiveSafetyBoundsOutcomeV2:
    if not isinstance(value, SemanticProblemV2):
        return _unsupported("INVALID_INPUT:SEMANTIC_PROBLEM:TYPE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return SemanticProblemV2.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning):
        return _numeric("NUMERIC_GAP:SEMANTIC_PROBLEM_REVALIDATION")
    except (ValidationError, TypeError, ValueError, Warning):
        return _unsupported("INVALID_INPUT:SEMANTIC_PROBLEM")


def _strict_cell(
    value: object,
    budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | ObjectiveSafetyBoundsOutcomeV2:
    if not isinstance(value, ExactRectilinearRegionV2):
        return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:TYPE")
    if not isinstance(value.topology, RectilinearTopologyV2):
        return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:TOPOLOGY")
    if type(value.rectangles) is not tuple:
        return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:RECTANGLES")
    try:
        # Freeze the cost of this independent strict pass before inspecting a
        # single nested rectangle or coordinate.  The kernel validation below
        # owns its separate canonical-shell charges, so this is not a proxy for
        # or a replacement of those operations.
        budget.consume(len(value.rectangles))
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource("RESOURCE_LIMIT:OBJECTIVE_CELL_STRICT_PASS")
    expected = RectilinearTopologyV2.EMPTY
    try:
        for rectangle in value.rectangles:
            if type(rectangle) is not ExactAxisAlignedRectV2:
                return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:RECTANGLE")
            checked = ExactAxisAlignedRectV2(
                coordinate_space=rectangle.coordinate_space,
                topology=rectangle.topology,
                min_x_m=rectangle.min_x_m,
                min_y_m=rectangle.min_y_m,
                max_x_m=rectangle.max_x_m,
                max_y_m=rectangle.max_y_m,
            )
            if (
                checked.coordinate_space
                is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
            ):
                return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:COORDINATE_SPACE")
            if checked.topology is RectTopologyV2.AREA:
                expected = RectilinearTopologyV2.AREA
            elif (
                checked.topology is RectTopologyV2.DEGENERATE
                and expected is RectilinearTopologyV2.EMPTY
            ):
                expected = RectilinearTopologyV2.DEGENERATE
            for coordinate in checked.bounds or ():
                _require_fraction_cap(coordinate)
    except _NumericGapV2 as error:
        return _numeric(error.finding_code)
    except (TypeError, ValueError):
        return _unsupported("INVALID_INPUT:OBJECTIVE_CELL")
    if value.topology is not expected:
        return _unsupported("INVALID_INPUT:OBJECTIVE_CELL:TOPOLOGY_MISMATCH")
    return value


def _raw_components_for_target(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
    target: ConstraintSafetyTargetV2,
    budget: RectilinearAtomicBudgetV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    constraints = problem.constraints
    if target.constraint_id == constraints.position_domain.constraint_id:
        return _position_components(problem, cell)
    collision = next(
        (
            item
            for item in constraints.collision_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if collision is not None:
        return _collision_components(problem, cell, collision, budget)
    support = next(
        (
            item
            for item in constraints.support_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if support is not None:
        return _support_components(problem, cell, support)
    visibility = next(
        (
            item
            for item in constraints.visibility_constraints
            if item.constraint_id == target.constraint_id
        ),
        None,
    )
    if visibility is not None:
        return _visibility_components(visibility)
    if target.constraint_id == constraints.target_relation.constraint_id:
        return _target_relation_components(problem, cell, budget)
    raise _UnsupportedV2(
        f"UNSUPPORTED_SAFETY_BOUNDS:UNKNOWN_CONSTRAINT:{target.constraint_id}"
    )


def _position_components(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    constraint = problem.constraints.position_domain
    findings: list[str] = []
    if constraint.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append("UNSUPPORTED_POSITION_SLACK:BOUNDARY_POLICY")
    if (
        constraint.region_interpretation
        is not PositionRegionInterpretationV2.SUBJECT_ANCHOR_LOCUS
    ):
        findings.append("UNSUPPORTED_POSITION_SLACK:REGION_INTERPRETATION")
    if (
        len(constraint.workspace_fact_ids) != 1
        or constraint.workspace_aggregation is not RegionAggregationV2.INTERSECTION
    ):
        findings.append("UNSUPPORTED_POSITION_SLACK:WORKSPACE_SELECTION")
    if len(constraint.known_free_space_fact_ids) > 1 or (
        constraint.known_free_space_fact_ids
        and constraint.known_free_space_aggregation
        is not RegionAggregationV2.INTERSECTION
    ):
        findings.append("UNSUPPORTED_POSITION_SLACK:FREE_SPACE_SELECTION")
    findings.extend(
        _family_findings("POSITION_WORKSPACE", problem.scene.workspace_boundaries)
    )
    if constraint.known_free_space_fact_ids:
        findings.extend(
            _family_findings("POSITION_FREE_SPACE", problem.scene.known_free_spaces)
        )
    if findings:
        raise _UnsupportedV2(*findings)

    workspace = {
        item.fact_id: item for item in problem.scene.workspace_boundaries.values or ()
    }[constraint.workspace_fact_ids[0]]
    if (
        workspace.region_approximation is not GeometryApproximationV2.EXACT
        or workspace.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
        or not _uncertainty_is_zero(workspace.geometry_uncertainty)
    ):
        raise _UnsupportedV2("UNSUPPORTED_POSITION_SLACK:WORKSPACE_FACT")
    permitted = _planar_rect_bounds(
        workspace.region_world_xy,
        RectCoordinateSpaceV2.WORLD_XY_M,
        "POSITION_WORKSPACE",
    )
    if constraint.known_free_space_fact_ids:
        free = {
            item.fact_id: item for item in problem.scene.known_free_spaces.values or ()
        }[constraint.known_free_space_fact_ids[0]]
        if (
            free.region_approximation is not GeometryApproximationV2.EXACT
            or free.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
            or not _uncertainty_is_zero(free.geometry_uncertainty)
        ):
            raise _UnsupportedV2("UNSUPPORTED_POSITION_SLACK:FREE_SPACE_FACT")
        free_bounds = _planar_rect_bounds(
            free.region_world_xy,
            RectCoordinateSpaceV2.WORLD_XY_M,
            "POSITION_FREE_SPACE",
        )
        permitted = (
            max(permitted[0], free_bounds[0]),
            max(permitted[1], free_bounds[1]),
            min(permitted[2], free_bounds[2]),
            min(permitted[3], free_bounds[3]),
        )
    clearance = Fraction.from_float(constraint.minimum_boundary_clearance_m)
    permitted = (
        permitted[0] + clearance,
        permitted[1] + clearance,
        permitted[2] - clearance,
        permitted[3] - clearance,
    )
    if permitted[0] > permitted[2] or permitted[1] > permitted[3]:
        raise _EmptySemanticV2
    subject_id = problem.constraints.allowed_edit.subject_id
    subject = _objects(problem)[subject_id]
    anchor = subject.pose.world_from_object.translation
    anchor_x = Fraction.from_float(anchor.x)
    anchor_y = Fraction.from_float(anchor.y)
    permitted = (
        permitted[0] - anchor_x,
        permitted[1] - anchor_y,
        permitted[2] - anchor_x,
        permitted[3] - anchor_y,
    )

    minima: list[Fraction] = []
    maxima: list[Fraction] = []
    for rectangle in cell.rectangles:
        bounds = _bounds(rectangle)
        if not (
            permitted[0] <= bounds[0] <= bounds[2] <= permitted[2]
            and permitted[1] <= bounds[1] <= bounds[3] <= permitted[3]
        ):
            raise _UnsupportedV2("UNSUPPORTED_POSITION_SLACK:CELL_OUTSIDE_LOCUS")
        minima.append(
            min(
                bounds[0] - permitted[0],
                permitted[2] - bounds[2],
                bounds[1] - permitted[1],
                permitted[3] - bounds[3],
            )
        )
        center_x = (permitted[0] + permitted[2]) / 2
        center_y = (permitted[1] + permitted[3]) / 2
        best_x = min(max(center_x, bounds[0]), bounds[2])
        best_y = min(max(center_y, bounds[1]), bounds[3])
        maxima.append(
            min(
                best_x - permitted[0],
                permitted[2] - best_x,
                best_y - permitted[1],
                permitted[3] - best_y,
            )
        )
    return (
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.POSITION_BOUNDARY_CLEARANCE,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=min(minima),
            raw_upper=max(maxima),
        ),
    )


def _collision_components(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
    constraint: CollisionConstraintV2,
    budget: RectilinearAtomicBudgetV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    findings: list[str] = []
    if (
        constraint.clearance_metric
        is not CollisionClearanceMetricV2.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
    ):
        findings.append("UNSUPPORTED_COLLISION_SLACK:CLEARANCE_METRIC")
    if constraint.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append("UNSUPPORTED_COLLISION_SLACK:BOUNDARY_POLICY")
    if constraint.support_contact_exceptions:
        findings.append("UNSUPPORTED_COLLISION_SLACK:SUPPORT_EXCEPTIONS")
    for label, facts in (
        ("COLLISION_OBJECTS", problem.scene.objects),
        ("COLLISION_BODIES", problem.scene.collision_bodies),
        ("COLLISION_GEOMETRIES", problem.scene.geometry_instances),
    ):
        findings.extend(_family_findings(label, facts))
    if findings:
        raise _UnsupportedV2(*findings)

    bodies = {
        item.body_id: item for item in problem.scene.collision_bodies.values or ()
    }
    subject_id = problem.constraints.allowed_edit.subject_id
    required = Fraction.from_float(constraint.minimum_clearance_m)
    pair_count = len(constraint.subject_body_ids) * len(constraint.obstacle_body_ids)
    # The caller already charged one rectangle visit for this target.  Precharge
    # every additional body-pair visit and both fixed 64-step directed square
    # roots per pair before entering the geometric loop.
    budget.consume(
        (pair_count - 1) * len(cell.rectangles)
        + pair_count * 2 * _DIRECTED_SQRT_ATOMIC_STEPS_V2
    )
    pair_lowers: list[Fraction] = []
    pair_uppers: list[Fraction] = []
    for subject_body_id in constraint.subject_body_ids:
        subject_body = bodies[subject_body_id]
        if subject_body.owner_object_id != subject_id:
            raise _UnsupportedV2("UNSUPPORTED_COLLISION_SLACK:SUBJECT_BODY_OWNER")
        subject_box = _single_body_box(problem, subject_body)
        for obstacle_body_id in constraint.obstacle_body_ids:
            obstacle_body = bodies[obstacle_body_id]
            if obstacle_body.owner_object_id == subject_id:
                raise _UnsupportedV2("UNSUPPORTED_COLLISION_SLACK:MOVING_OBSTACLE")
            obstacle_box = _single_body_box(problem, obstacle_body)
            pair_lower, pair_upper = _collision_pair_clearance_bounds(
                cell,
                subject_box,
                obstacle_box,
                constraint.constraint_id,
            )
            pair_lowers.append(pair_lower - required)
            pair_uppers.append(pair_upper - required)
    return (
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.COLLISION_CLEARANCE,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=min(pair_lowers),
            raw_upper=min(pair_uppers),
        ),
    )


def _support_components(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
    constraint: SupportConstraintV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    findings: list[str] = []
    if constraint.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append("UNSUPPORTED_SUPPORT_SLACK:BOUNDARY_POLICY")
    if len(constraint.subject_contact_geometry_ids) != 1:
        findings.append("UNSUPPORTED_SUPPORT_SLACK:CONTACT_CARDINALITY")
    for label, facts in (
        ("SUPPORT_OBJECTS", problem.scene.objects),
        ("SUPPORT_GEOMETRIES", problem.scene.geometry_instances),
        ("SUPPORT_SURFACES", problem.scene.support_surfaces),
    ):
        findings.extend(_family_findings(label, facts))
    if findings:
        raise _UnsupportedV2(*findings)
    objects = _objects(problem)
    geometries = _geometries(problem)
    surfaces = {
        item.surface_id: item for item in problem.scene.support_surfaces.values or ()
    }
    supported = objects[constraint.supported_object_id]
    geometry = geometries[constraint.subject_contact_geometry_ids[0]]
    surface = surfaces[constraint.surface_id]
    if (
        geometry.owner_object_id != supported.object_id
        or geometry.role is not GeometryRoleV2.SUPPORT
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or not isinstance(geometry.shape, UprightBox3DV2)
        or not _uncertainty_is_zero(geometry.uncertainty)
        or not _identity_rotation(supported.pose.world_from_object.rotation)
        or not _identity_rotation(geometry.anchor_from_geometry.rotation)
    ):
        raise _UnsupportedV2("UNSUPPORTED_SUPPORT_SLACK:CONTACT_GEOMETRY")
    if (
        surface.region_approximation is not GeometryApproximationV2.EXACT
        or surface.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
        or not _uncertainty_is_zero(surface.geometry_uncertainty)
        or not _identity_rotation(surface.anchor_from_surface.rotation)
        or (
            surface.normal_in_anchor.x,
            surface.normal_in_anchor.y,
            surface.normal_in_anchor.z,
        )
        != (0.0, 0.0, 1.0)
    ):
        raise _UnsupportedV2("UNSUPPORTED_SUPPORT_SLACK:SURFACE")
    if surface.owner_object_id is not None and not _identity_rotation(
        objects[surface.owner_object_id].pose.world_from_object.rotation
    ):
        raise _UnsupportedV2("UNSUPPORTED_SUPPORT_SLACK:SURFACE_OWNER_ROTATION")

    contact, contact_z = _support_contact_world(supported, geometry)
    support, support_z = _support_surface_world(surface, objects)
    subject_id = problem.constraints.allowed_edit.subject_id
    coefficient = int(supported.object_id == subject_id) - int(
        surface.owner_object_id == subject_id
    )
    if coefficient not in {-1, 0, 1}:  # pragma: no cover - boolean difference
        raise ValueError("relative edit coefficient escaped its closed range")

    overlap_minima: list[Fraction] = []
    overlap_maxima: list[Fraction] = []
    stability_minima: list[Fraction] = []
    stability_maxima: list[Fraction] = []
    inset = Fraction.from_float(constraint.stability_margin_m)
    for rectangle in cell.rectangles:
        bounds = _bounds(rectangle)
        tx = _scaled_interval(bounds[0], bounds[2], coefficient)
        ty = _scaled_interval(bounds[1], bounds[3], coefficient)
        wx = _overlap_width_extrema(contact[0], contact[2], support[0], support[2], tx)
        wy = _overlap_width_extrema(contact[1], contact[3], support[1], support[3], ty)
        overlap_minima.append(wx[0] * wy[0])
        overlap_maxima.append(wx[1] * wy[1])
        x_side_min = min(
            contact[0] + tx[0] - support[0],
            support[2] - contact[2] - tx[1],
        )
        y_side_min = min(
            contact[1] + ty[0] - support[1],
            support[3] - contact[3] - ty[1],
        )
        stability_minima.append(min(x_side_min, y_side_min) - inset)
        best_tx = min(
            max(
                (support[0] + support[2] - contact[0] - contact[2]) / 2,
                tx[0],
            ),
            tx[1],
        )
        best_ty = min(
            max(
                (support[1] + support[3] - contact[1] - contact[3]) / 2,
                ty[0],
            ),
            ty[1],
        )
        stability_maxima.append(
            min(
                contact[0] + best_tx - support[0],
                support[2] - contact[2] - best_tx,
                contact[1] + best_ty - support[1],
                support[3] - contact[3] - best_ty,
            )
            - inset
        )
    gap = contact_z - support_z
    minimum_area = Fraction.from_float(constraint.minimum_overlap_area_m2)
    return (
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.SUPPORT_CONTACT_GAP_LOWER_MARGIN,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=gap - Fraction.from_float(constraint.contact_gap_min_m),
            raw_upper=gap - Fraction.from_float(constraint.contact_gap_min_m),
        ),
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.SUPPORT_CONTACT_GAP_UPPER_MARGIN,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=Fraction.from_float(constraint.contact_gap_max_m) - gap,
            raw_upper=Fraction.from_float(constraint.contact_gap_max_m) - gap,
        ),
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.SUPPORT_OVERLAP_AREA_MARGIN,
            unit=SafetySlackUnitV2.SQUARE_METRE,
            raw_lower=min(overlap_minima) - minimum_area,
            raw_upper=max(overlap_maxima) - minimum_area,
        ),
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.SUPPORT_STABILITY_INSET_MARGIN,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=min(stability_minima),
            raw_upper=max(stability_maxima),
        ),
    )


def _visibility_components(
    constraint: VisibilityConstraintV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    visible = Fraction.from_float(constraint.minimum_visible_fraction)
    area = Fraction.from_float(constraint.minimum_image_area_fraction)
    truncated = Fraction.from_float(constraint.maximum_truncated_fraction)
    return (
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.VISIBILITY_VISIBLE_FRACTION_MARGIN,
            unit=SafetySlackUnitV2.FRACTION,
            raw_lower=-visible,
            raw_upper=1 - visible,
        ),
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.VISIBILITY_IMAGE_AREA_FRACTION_MARGIN,
            unit=SafetySlackUnitV2.FRACTION,
            raw_lower=-area,
            raw_upper=1 - area,
        ),
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.VISIBILITY_TRUNCATED_FRACTION_MARGIN,
            unit=SafetySlackUnitV2.FRACTION,
            raw_lower=truncated - 1,
            raw_upper=truncated,
        ),
    )


def _target_relation_components(
    problem: SemanticProblemV2,
    cell: ExactRectilinearRegionV2,
    budget: RectilinearAtomicBudgetV2,
) -> tuple[SafetyRawComponentIntervalV2, ...]:
    target = problem.constraints.target_relation
    definition = next(
        item
        for item in problem.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    findings: list[str] = []
    if target.relation_after not in {RelationV2.NEAR, RelationV2.FAR}:
        findings.append("UNSUPPORTED_TARGET_SLACK:RELATION")
    if definition.measurement is not RelationMeasurementV2.SHAPE_GAP_XY:
        findings.append("UNSUPPORTED_TARGET_SLACK:MEASUREMENT")
    if definition.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append("UNSUPPORTED_TARGET_SLACK:BOUNDARY_POLICY")
    if definition.tolerance != 0.0:
        findings.append("UNSUPPORTED_TARGET_SLACK:TOLERANCE")
    for label, facts in (
        ("TARGET_OBJECTS", problem.scene.objects),
        ("TARGET_GEOMETRIES", problem.scene.geometry_instances),
    ):
        findings.extend(_family_findings(label, facts))
    if findings:
        raise _UnsupportedV2(*findings)
    objects = _objects(problem)
    geometries = tuple(problem.scene.geometry_instances.values or ())
    boxes: list[_WorldBoxV2] = []
    for object_id in (target.subject_id, target.reference_id):
        selected = tuple(
            item
            for item in geometries
            if item.owner_object_id == object_id
            and item.role is GeometryRoleV2.RELATION
        )
        if len(selected) != 1:
            raise _UnsupportedV2(
                f"UNSUPPORTED_TARGET_SLACK:GEOMETRY_CARDINALITY:{object_id}"
            )
        geometry = selected[0]
        object_ = objects[object_id]
        if (
            geometry.approximation is not GeometryApproximationV2.EXACT
            or not isinstance(geometry.shape, UprightBox3DV2)
            or not _uncertainty_is_zero(geometry.uncertainty)
            or not _identity_rotation(object_.pose.world_from_object.rotation)
            or not _identity_rotation(geometry.anchor_from_geometry.rotation)
        ):
            raise _UnsupportedV2(
                f"UNSUPPORTED_TARGET_SLACK:GEOMETRY:{geometry.geometry_id}"
            )
        boxes.append(_world_box(object_, geometry))
    subject_box, reference_box = boxes
    edited_subject_id = problem.constraints.allowed_edit.subject_id
    coefficient = int(target.subject_id == edited_subject_id) - int(
        target.reference_id == edited_subject_id
    )
    qx = (
        reference_box.min_x - subject_box.max_x,
        reference_box.max_x - subject_box.min_x,
    )
    qy = (
        reference_box.min_y - subject_box.max_y,
        reference_box.max_y - subject_box.min_y,
    )
    budget.consume(2 * _DIRECTED_SQRT_ATOMIC_STEPS_V2)
    squared_lower, squared_upper = _gap_squared_extrema(
        cell,
        qx,
        qy,
        Fraction(),
        coefficient=coefficient,
    )
    measurement_lower, _ = _sqrt_fraction_bounds(squared_lower)
    _, measurement_upper = _sqrt_fraction_bounds(squared_upper)
    threshold = Fraction.from_float(definition.threshold)
    if definition.comparator is MeasurementComparatorV2.LESS_THAN:
        raw_lower = threshold - measurement_upper
        raw_upper = threshold - measurement_lower
    else:
        raw_lower = measurement_lower - threshold
        raw_upper = measurement_upper - threshold
    return (
        SafetyRawComponentIntervalV2(
            kind=SafetyComponentKindV2.TARGET_RELATION_THRESHOLD_MARGIN,
            unit=SafetySlackUnitV2.METRE,
            raw_lower=raw_lower,
            raw_upper=raw_upper,
        ),
    )


def _collision_pair_clearance_bounds(
    cell: ExactRectilinearRegionV2,
    subject_box: _WorldBoxV2,
    obstacle_box: _WorldBoxV2,
    constraint_id: str,
) -> tuple[Fraction, Fraction]:
    qx = (
        obstacle_box.min_x - subject_box.max_x,
        obstacle_box.max_x - subject_box.min_x,
    )
    qy = (
        obstacle_box.min_y - subject_box.max_y,
        obstacle_box.max_y - subject_box.min_y,
    )
    z_gap = _interval_distance(
        subject_box.min_z,
        subject_box.max_z,
        obstacle_box.min_z,
        obstacle_box.max_z,
    )
    z_interiors_overlap = (
        subject_box.min_z < obstacle_box.max_z
        and obstacle_box.min_z < subject_box.max_z
    )
    if z_interiors_overlap and any(
        bounds[2] > qx[0]
        and bounds[0] < qx[1]
        and bounds[3] > qy[0]
        and bounds[1] < qy[1]
        for bounds in (_bounds(rectangle) for rectangle in cell.rectangles)
    ):
        raise _UnsupportedV2(
            f"UNSUPPORTED_COLLISION_SLACK:INTERIOR_DISJOINT:{constraint_id}"
        )
    squared_lower, squared_upper = _gap_squared_extrema(cell, qx, qy, z_gap)
    distance_lower, _ = _sqrt_fraction_bounds(squared_lower)
    _, distance_upper = _sqrt_fraction_bounds(squared_upper)
    return distance_lower, distance_upper


def _single_body_box(
    problem: SemanticProblemV2,
    body: CollisionBodyFactV2,
) -> _WorldBoxV2:
    if len(body.geometry_instance_ids) != 1:
        raise _UnsupportedV2(
            f"UNSUPPORTED_COLLISION_SLACK:GEOMETRY_CARDINALITY:{body.body_id}"
        )
    geometry = _geometries(problem)[body.geometry_instance_ids[0]]
    if (
        geometry.role is not GeometryRoleV2.COLLISION
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or not isinstance(geometry.shape, UprightBox3DV2)
        or not _uncertainty_is_zero(geometry.uncertainty)
        or not _identity_rotation(geometry.anchor_from_geometry.rotation)
    ):
        raise _UnsupportedV2(
            f"UNSUPPORTED_COLLISION_SLACK:GEOMETRY:{geometry.geometry_id}"
        )
    owner = None
    if body.owner_object_id is not None:
        owner = _objects(problem)[body.owner_object_id]
        if not _identity_rotation(owner.pose.world_from_object.rotation):
            raise _UnsupportedV2(
                f"UNSUPPORTED_COLLISION_SLACK:OWNER_ROTATION:{owner.object_id}"
            )
    return _world_box(owner, geometry)


def _world_box(
    owner: CanonicalObjectV2 | None,
    geometry: GeometryInstanceV2,
) -> _WorldBoxV2:
    if not isinstance(geometry.shape, UprightBox3DV2):
        raise TypeError("world box requires an UprightBox3DV2")
    owner_translation = (Fraction(), Fraction(), Fraction())
    if owner is not None:
        translation = owner.pose.world_from_object.translation
        owner_translation = tuple(
            Fraction.from_float(value)
            for value in (translation.x, translation.y, translation.z)
        )
    translation = geometry.anchor_from_geometry.translation
    center = tuple(
        owner_translation[index] + Fraction.from_float(value)
        for index, value in enumerate((translation.x, translation.y, translation.z))
    )
    half = tuple(
        Fraction.from_float(value) / 2
        for value in (
            geometry.shape.size_m.x,
            geometry.shape.size_m.y,
            geometry.shape.size_m.z,
        )
    )
    return _WorldBoxV2(
        min_x=center[0] - half[0],
        min_y=center[1] - half[1],
        min_z=center[2] - half[2],
        max_x=center[0] + half[0],
        max_y=center[1] + half[1],
        max_z=center[2] + half[2],
    )


def _support_contact_world(
    owner: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], Fraction]:
    box = _world_box(owner, geometry)
    return (box.min_x, box.min_y, box.max_x, box.max_y), box.min_z


def _support_surface_world(
    surface: SupportSurfaceFactV2,
    objects: dict[str, CanonicalObjectV2],
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], Fraction]:
    local = _planar_rect_bounds(
        surface.region_uv,
        RectCoordinateSpaceV2.WORLD_XY_M,
        "SUPPORT_SURFACE",
    )
    owner_translation = (Fraction(), Fraction(), Fraction())
    if surface.owner_object_id is not None:
        translation = objects[
            surface.owner_object_id
        ].pose.world_from_object.translation
        owner_translation = tuple(
            Fraction.from_float(value)
            for value in (translation.x, translation.y, translation.z)
        )
    translation = surface.anchor_from_surface.translation
    dx = owner_translation[0] + Fraction.from_float(translation.x)
    dy = owner_translation[1] + Fraction.from_float(translation.y)
    z = owner_translation[2] + Fraction.from_float(translation.z)
    return (
        (local[0] + dx, local[1] + dy, local[2] + dx, local[3] + dy),
        z,
    )


def _gap_squared_extrema(
    cell: ExactRectilinearRegionV2,
    qx: tuple[Fraction, Fraction],
    qy: tuple[Fraction, Fraction],
    z_gap: Fraction,
    *,
    coefficient: int = 1,
) -> tuple[Fraction, Fraction]:
    minima: list[Fraction] = []
    maxima: list[Fraction] = []
    for rectangle in cell.rectangles:
        bounds = _bounds(rectangle)
        relative_x = _scaled_interval(bounds[0], bounds[2], coefficient)
        relative_y = _scaled_interval(bounds[1], bounds[3], coefficient)
        min_x = _interval_distance(*relative_x, qx[0], qx[1])
        min_y = _interval_distance(*relative_y, qy[0], qy[1])
        max_x = max(
            _point_interval_distance(relative_x[0], *qx),
            _point_interval_distance(relative_x[1], *qx),
        )
        max_y = max(
            _point_interval_distance(relative_y[0], *qy),
            _point_interval_distance(relative_y[1], *qy),
        )
        minima.append(min_x**2 + min_y**2 + z_gap**2)
        maxima.append(max_x**2 + max_y**2 + z_gap**2)
    return min(minima), max(maxima)


def _overlap_width_extrema(
    contact_lower: Fraction,
    contact_upper: Fraction,
    surface_lower: Fraction,
    surface_upper: Fraction,
    translation: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    candidates = {translation[0], translation[1]}
    for breakpoint in (
        surface_lower - contact_upper,
        surface_lower - contact_lower,
        surface_upper - contact_upper,
        surface_upper - contact_lower,
    ):
        if translation[0] <= breakpoint <= translation[1]:
            candidates.add(breakpoint)
    widths = tuple(
        max(
            Fraction(),
            min(contact_upper + value, surface_upper)
            - max(contact_lower + value, surface_lower),
        )
        for value in candidates
    )
    return min(widths), max(widths)


def _scaled_interval(
    lower: Fraction,
    upper: Fraction,
    coefficient: int,
) -> tuple[Fraction, Fraction]:
    if coefficient == 1:
        return lower, upper
    if coefficient == -1:
        return -upper, -lower
    return Fraction(), Fraction()


def _interval_distance(
    left_lower: Fraction,
    left_upper: Fraction,
    right_lower: Fraction,
    right_upper: Fraction,
) -> Fraction:
    if left_upper < right_lower:
        return right_lower - left_upper
    if left_lower > right_upper:
        return left_lower - right_upper
    return Fraction()


def _point_interval_distance(
    point: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> Fraction:
    if point < lower:
        return lower - point
    if point > upper:
        return point - upper
    return Fraction()


def _sqrt_fraction_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    if value < 0:
        raise ValueError("squared distance must be non-negative")
    _require_fraction_cap(value)
    low_bits = 0
    high_bits = _MAX_FINITE_BINARY64_BITS
    best_bits = 0
    while low_bits <= high_bits:
        middle = (low_bits + high_bits) // 2
        candidate = _positive_float_from_bits(middle)
        if Fraction.from_float(candidate) ** 2 <= value:
            best_bits = middle
            low_bits = middle + 1
        else:
            high_bits = middle - 1
    lower_float = _positive_float_from_bits(best_bits)
    lower = Fraction.from_float(lower_float)
    if lower**2 == value:
        return lower, lower
    if best_bits == _MAX_FINITE_BINARY64_BITS:
        raise _NumericGapV2("NUMERIC_GAP:SAFETY_SQRT_NOT_FINITE")
    upper = Fraction.from_float(_positive_float_from_bits(best_bits + 1))
    return lower, upper


def _positive_float_from_bits(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _planar_rect_bounds(
    region: object,
    coordinate_space: RectCoordinateSpaceV2,
    label: str,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    try:
        rectangle = ExactAxisAlignedRectV2.from_planar_region(
            region,
            coordinate_space=coordinate_space,
        )
    except (UnsupportedRectRegionErrorV2, TypeError, ValueError) as error:
        raise _UnsupportedV2(
            f"UNSUPPORTED_SAFETY_BOUNDS:NON_RECTANGLE:{label}"
        ) from error
    if rectangle.topology is not RectTopologyV2.AREA:
        raise _UnsupportedV2(f"UNSUPPORTED_SAFETY_BOUNDS:NON_AREA:{label}")
    return _bounds(rectangle)


def _bounds(
    rectangle: ExactAxisAlignedRectV2,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    bounds = rectangle.bounds
    if bounds is None:
        raise ValueError("objective cell contains an EMPTY rectangle")
    return bounds


def _family_findings(label: str, facts: FactSetV2) -> tuple[str, ...]:
    if facts.availability is FactAvailabilityV2.MISSING:
        return (f"MISSING_FACT:{label}",)
    findings: list[str] = []
    if facts.availability is not FactAvailabilityV2.KNOWN:
        findings.append(f"UNSUPPORTED_SAFETY_BOUNDS:{label}_AVAILABILITY")
    if facts.completeness is not FactCompletenessV2.EXACT:
        findings.append(f"UNSUPPORTED_SAFETY_BOUNDS:{label}_COMPLETENESS")
    if facts.values is None:
        findings.append(f"MISSING_FACT:{label}_VALUES")
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_SAFETY_BOUNDS:{label}_UNCERTAINTY")
    return tuple(findings)


def _objects(problem: SemanticProblemV2) -> dict[str, CanonicalObjectV2]:
    return {item.object_id: item for item in problem.scene.objects.values or ()}


def _geometries(problem: SemanticProblemV2) -> dict[str, GeometryInstanceV2]:
    return {
        item.geometry_id: item for item in problem.scene.geometry_instances.values or ()
    }


def _identity_rotation(rotation: QuaternionV2) -> bool:
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _numeric_policy_is_zero(policy: NumericPolicyV2) -> bool:
    return all(
        value == 0.0
        for value in (
            policy.linear_tolerance_m,
            policy.area_tolerance_m2,
            policy.angular_tolerance_rad,
            policy.pixel_tolerance_px,
            policy.fraction_tolerance,
        )
    )


def _uncertainty_is_zero(uncertainty: UncertaintyBudgetV2) -> bool:
    return _numeric_policy_is_zero(
        uncertainty.source_error
    ) and _numeric_policy_is_zero(uncertainty.shape_approximation)


def _require_fraction_cap(value: Fraction) -> None:
    if (
        type(value) is not Fraction
        or value.numerator.bit_length() > OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2
        or value.denominator.bit_length() > OBJECTIVE_NUMERIC_MAX_FRACTION_BITS_V2
    ):
        raise _NumericGapV2("NUMERIC_GAP:SAFETY_FRACTION_BIT_CAP")


def _resolve_atomic_budget(
    max_atomic_cells: int | None,
    atomic_budget: RectilinearAtomicBudgetV2 | None,
) -> RectilinearAtomicBudgetV2:
    if (max_atomic_cells is None) == (atomic_budget is None):
        raise ValueError("provide exactly one of max_atomic_cells or atomic_budget")
    if atomic_budget is not None:
        if type(atomic_budget) is not RectilinearAtomicBudgetV2:
            raise TypeError("atomic_budget must be a RectilinearAtomicBudgetV2")
        atomic_budget.validate()
        return atomic_budget
    if type(max_atomic_cells) is not int:
        raise TypeError("max_atomic_cells must be an exact int")
    return RectilinearAtomicBudgetV2(limit=max_atomic_cells)


def _unsupported(*findings: str) -> ObjectiveSafetyBoundsOutcomeV2:
    return ObjectiveSafetyBoundsOutcomeV2(
        kind=ObjectiveSafetyBoundsKindV2.UNSUPPORTED,
        finding_codes=tuple(findings),
    )


def _numeric(*findings: str) -> ObjectiveSafetyBoundsOutcomeV2:
    return ObjectiveSafetyBoundsOutcomeV2(
        kind=ObjectiveSafetyBoundsKindV2.NUMERIC_GAP,
        finding_codes=tuple(findings),
    )


def _resource(*findings: str) -> ObjectiveSafetyBoundsOutcomeV2:
    return ObjectiveSafetyBoundsOutcomeV2(
        kind=ObjectiveSafetyBoundsKindV2.RESOURCE,
        finding_codes=tuple(
            findings or ("RESOURCE_LIMIT:OBJECTIVE_SAFETY_ATOMIC_CELLS",)
        ),
    )
