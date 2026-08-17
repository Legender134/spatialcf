"""Directed strict-convex SUPPORT projection for one continuously yawed box."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import gcd, lcm

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core.v2 import so2_interval
from spatialcf.core.v2.convex_translation_domain import (
    RationalConvexPolygonV2,
    RationalPoint2V2,
)
from spatialcf.core.v2.convex_translation_partition import (
    RationalHalfPlane2V2,
    RationalHalfPlaneRelationV2,
)
from spatialcf.core.v2.oriented_upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
    UnsupportedRectRegionErrorV2,
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
    StrictConvexIntersectionTopologyV2,
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
    SupportAssignmentPolicyV2,
    SupportContactAggregationV2,
    SupportContactFeatureV2,
    SupportOverlapMetricV2,
    SupportStabilityMetricV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.continuous_yaw_candidate import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
    SupportSurfaceFactV2_2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.scene import RegionBoundaryPolicyV2

CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2 = (
    "geometry-kernel:rational-continuous-yaw-support-projection-v2"
)
CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2 = (
    "kernel:2.6-exact-horizontal-support-projection"
)


class ContinuousYawSupportProjectionKindV2(StrEnum):
    BRACKET = "BRACKET"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class _UnsupportedSupportProjectionV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code


class _InvalidSupportProjectionInputV2(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ContinuousYawSupportProjectionBracketV2:
    support_constraint_id: str
    surface_id: str
    contact_geometry_id: str
    support_projection_kernel_id: str
    support_projection_kernel_version: str
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    inner_bounds: tuple[Fraction, Fraction, Fraction, Fraction]
    outer_bounds: tuple[Fraction, Fraction, Fraction, Fraction]
    so2_atomic_steps_used: int
    domain_operations_used: int
    candidate_cells_used: int

    def __post_init__(self) -> None:
        for field_name in (
            "support_constraint_id",
            "surface_id",
            "contact_geometry_id",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be an exact non-blank string")
        if self.support_projection_kernel_id != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2
        ):
            raise ValueError("unexpected support projection kernel ID")
        if self.support_projection_kernel_version != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2
        ):
            raise ValueError("unexpected support projection kernel version")
        for field_name in ("inner_allowed", "outer_allowed"):
            value = getattr(self, field_name)
            if type(value) is not StrictConvexIntersectionComplexV2:
                raise TypeError(f"{field_name} must be a strict complex")
            object.__setattr__(
                self,
                field_name,
                StrictConvexIntersectionComplexV2(
                    cells=value.cells,
                    universe=value.universe,
                    topology=value.topology,
                ),
            )
        if self.inner_allowed.universe != self.outer_allowed.universe:
            raise ValueError("support bracket requires one exact universe")
        inner = _require_bounds_tuple(self.inner_bounds, label="inner_bounds")
        outer = _require_bounds_tuple(self.outer_bounds, label="outer_bounds")
        if not (
            outer[0] <= inner[0] <= inner[2] <= outer[2]
            and outer[1] <= inner[1] <= inner[3] <= outer[3]
        ):
            raise ValueError("support inner bounds must be contained in outer bounds")
        object.__setattr__(self, "inner_bounds", inner)
        object.__setattr__(self, "outer_bounds", outer)
        for field_name in (
            "so2_atomic_steps_used",
            "domain_operations_used",
            "candidate_cells_used",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive exact int")


@dataclass(frozen=True, slots=True)
class ContinuousYawSupportProjectionOutcomeV2:
    kind: ContinuousYawSupportProjectionKindV2
    bracket: ContinuousYawSupportProjectionBracketV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ContinuousYawSupportProjectionKindV2:
            raise TypeError("kind must be ContinuousYawSupportProjectionKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ContinuousYawSupportProjectionKindV2.BRACKET:
            if type(self.bracket) is not ContinuousYawSupportProjectionBracketV2:
                raise ValueError("BRACKET requires a support projection bracket")
            if findings:
                raise ValueError("BRACKET cannot carry findings")
            object.__setattr__(self, "bracket", _copy_bracket(self.bracket))
            return
        if self.bracket is not None or not findings:
            raise ValueError("failure requires findings and no bracket")


def compile_exact_horizontal_support_projection_v2(
    problem: SemanticProblemV2_2,
    support_constraint_id: str,
    universe: ExactAxisAlignedRectV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> ContinuousYawSupportProjectionOutcomeV2:
    """Compile one fixed-owner horizontal SUPPORT predicate into a bracket."""

    try:
        _require_budgets(atomic_budget, intersection_budget)
        start_so2 = atomic_budget.used
        start_domain = intersection_budget.domain_operations_used
        start_cells = intersection_budget.candidate_cells_used
        checked_problem, checked_id, checked_universe = _strict_inputs(
            problem,
            support_constraint_id,
            universe,
            intersection_budget,
        )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _failure(
            ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SUPPORT_PROJECTION",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_PROJECTION_REVALIDATION",
        )
    except _InvalidSupportProjectionInputV2:
        return _failure(
            ContinuousYawSupportProjectionKindV2.INVALID_INPUT,
            "INVALID_INPUT:SUPPORT_PROJECTION",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            constraint, subject, geometry, surface, owner_transform = (
                _extract_supported_subset(
                    checked_problem,
                    checked_id,
                    intersection_budget,
                )
            )
            box_outcome = compile_oriented_upright_box_bounds_v2(
                subject.pose.world_from_object,
                geometry.shape,
                atomic_budget=atomic_budget,
            )
            if box_outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
                return _failure(
                    ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
                    "RESOURCE_LIMIT:SUPPORT_PROJECTION_SO2",
                )
            if box_outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
                return _failure(
                    ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
                    *box_outcome.finding_codes,
                )
            if box_outcome.kind is not SO2IntervalKindV2.EXACT:
                raise RuntimeError("supported support box failed strict compilation")
            if type(box_outcome.bounds) is not OrientedUprightBoxBoundsV2:
                raise RuntimeError("EXACT oriented support box is missing bounds")
            inner_bounds, outer_bounds = _support_bounds(
                constraint,
                subject.pose.world_from_object,
                geometry,
                surface,
                owner_transform,
                box_outcome.bounds,
                intersection_budget,
            )
            inner = _complex_from_bounds(
                inner_bounds,
                checked_universe,
                cell_id="cell:support-projection:inner",
                budget=intersection_budget,
            )
            outer = _complex_from_bounds(
                outer_bounds,
                checked_universe,
                cell_id="cell:support-projection:outer",
                budget=intersection_budget,
            )
            if not inner.cells or not outer.cells:
                raise _UnsupportedSupportProjectionV2(
                    "UNSUPPORTED_MODEL:SUPPORT_PROJECTION_NON_AREA_LOCUS"
                )
            intersection_budget.consume_domain(
                8 + len(inner.cells[0].half_planes) + len(outer.cells[0].half_planes)
            )
            bracket = ContinuousYawSupportProjectionBracketV2(
                support_constraint_id=constraint.constraint_id,
                surface_id=surface.surface_id,
                contact_geometry_id=geometry.geometry_id,
                support_projection_kernel_id=(
                    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2
                ),
                support_projection_kernel_version=(
                    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2
                ),
                inner_allowed=inner,
                outer_allowed=outer,
                inner_bounds=inner_bounds,
                outer_bounds=outer_bounds,
                so2_atomic_steps_used=atomic_budget.used - start_so2,
                domain_operations_used=(
                    intersection_budget.domain_operations_used - start_domain
                ),
                candidate_cells_used=(
                    intersection_budget.candidate_cells_used - start_cells
                ),
            )
            return ContinuousYawSupportProjectionOutcomeV2(
                kind=ContinuousYawSupportProjectionKindV2.BRACKET,
                bracket=bracket,
            )
    except _UnsupportedSupportProjectionV2 as error:
        return _failure(
            ContinuousYawSupportProjectionKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _failure(
            ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SUPPORT_PROJECTION",
        )
    except ArithmeticError:
        return _failure(
            ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_PROJECTION_ARITHMETIC",
        )
    except RuntimeWarning:
        return _failure(
            ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_PROJECTION_RUNTIME_WARNING",
        )


def _require_budgets(
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> None:
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be StrictConvexIntersectionBudgetV2")
    intersection_budget.consume_domain(0)
    intersection_budget.consume_candidate_cells(0)


def _strict_inputs(
    problem: object,
    support_constraint_id: object,
    universe: object,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[SemanticProblemV2_2, str, ExactAxisAlignedRectV2]:
    if type(problem) is not SemanticProblemV2_2:
        raise _InvalidSupportProjectionInputV2
    if type(support_constraint_id) is not str or not support_constraint_id.strip():
        raise _InvalidSupportProjectionInputV2
    if type(universe) is not ExactAxisAlignedRectV2:
        raise _InvalidSupportProjectionInputV2
    budget.consume_domain(3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_problem = SemanticProblemV2_2.model_validate(
                problem.model_dump(mode="python", warnings="error"), strict=True
            )
    except (ArithmeticError, RuntimeWarning):
        raise
    except (ValidationError, PydanticSerializationError, Warning) as error:
        raise _InvalidSupportProjectionInputV2 from error
    checked_universe = ExactAxisAlignedRectV2(
        coordinate_space=universe.coordinate_space,
        topology=universe.topology,
        min_x_m=universe.min_x_m,
        min_y_m=universe.min_y_m,
        max_x_m=universe.max_x_m,
        max_y_m=universe.max_y_m,
    )
    if (
        checked_universe.coordinate_space
        is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        or checked_universe.topology is not RectTopologyV2.AREA
    ):
        raise _InvalidSupportProjectionInputV2
    return checked_problem, support_constraint_id, checked_universe


def _extract_supported_subset(
    problem: SemanticProblemV2_2,
    constraint_id: str,
    budget: StrictConvexIntersectionBudgetV2,
):
    constraints = problem.constraints.support_constraints
    if len(constraints) != 1 or constraints[0].constraint_id != constraint_id:
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_CONSTRAINT_CARDINALITY"
        )
    constraint = constraints[0]
    if (
        constraint.supported_object_id != problem.constraints.allowed_edit.subject_id
        or len(constraint.subject_contact_geometry_ids) != 1
        or constraint.contact_feature
        is not SupportContactFeatureV2.LOWEST_FACE_ALONG_SURFACE_NORMAL
        or constraint.contact_aggregation
        is not SupportContactAggregationV2.UNION_ALL_SELECTED_FEATURES
        or constraint.overlap_metric
        is not SupportOverlapMetricV2.PROJECTED_CONTACT_UNION_INTERSECTION_AREA
        or constraint.stability_metric
        is not SupportStabilityMetricV2.FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET
        or constraint.boundary_policy is not BoundaryPolicyV2.CLOSED
        or constraint.assignment_policy is not SupportAssignmentPolicyV2.EXACT_SURFACE
        or problem.numeric_policy != NumericPolicyV2()
    ):
        raise _UnsupportedSupportProjectionV2("UNSUPPORTED_MODEL:SUPPORT_POLICY")
    for label, facts in (
        ("OBJECTS", problem.scene.objects),
        ("GEOMETRIES", problem.scene.geometry_instances),
        ("BODIES", problem.scene.collision_bodies),
        ("SURFACES", problem.scene.support_surfaces),
    ):
        if (
            facts.availability is not FactAvailabilityV2.KNOWN
            or facts.completeness is not FactCompletenessV2.EXACT
            or facts.uncertainty != UncertaintyBudgetV2()
        ):
            raise _UnsupportedSupportProjectionV2(
                f"UNSUPPORTED_MODEL:SUPPORT_{label}_FACTS"
            )
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = {
        item.geometry_id: item for item in problem.scene.geometry_instances.values or ()
    }
    surfaces = {
        item.surface_id: item for item in problem.scene.support_surfaces.values or ()
    }
    bodies = {
        item.body_id: item for item in problem.scene.collision_bodies.values or ()
    }
    budget.consume_domain(
        len(objects) + len(geometries) + len(surfaces) + len(bodies) + 4
    )
    try:
        subject = objects[constraint.supported_object_id]
        geometry = geometries[constraint.subject_contact_geometry_ids[0]]
        surface = surfaces[constraint.surface_id]
        body = bodies[surface.supporting_body_id]
    except KeyError as error:
        raise RuntimeError(
            "support semantic graph lost a canonical reference"
        ) from error
    collision = problem.constraints.collision_constraints[0]
    if not subject.movable or body.body_id not in collision.obstacle_body_ids:
        raise _UnsupportedSupportProjectionV2("UNSUPPORTED_MODEL:SUPPORT_OWNER_SUBSET")
    if surface.owner_object_id is None:
        if body.owner_object_id is not None:
            raise _UnsupportedSupportProjectionV2(
                "UNSUPPORTED_MODEL:SUPPORT_OWNER_SUBSET"
            )
        owner_transform = DirectedYawIntervalTransformV2_2(
            translation=Vec3V2(x=0.0, y=0.0, z=0.0),
            yaw_radians=0.0,
        )
    else:
        try:
            owner = objects[surface.owner_object_id]
        except KeyError as error:
            raise RuntimeError(
                "support semantic graph lost its owner object"
            ) from error
        if owner.movable or body.owner_object_id != owner.object_id:
            raise _UnsupportedSupportProjectionV2(
                "UNSUPPORTED_MODEL:SUPPORT_OWNER_SUBSET"
            )
        owner_transform = owner.pose.world_from_object
    if (
        type(geometry) is not GeometryInstanceV2_2
        or geometry.owner_object_id != subject.object_id
        or geometry.role is not GeometryRoleV2.SUPPORT
        or geometry.approximation is not GeometryApproximationV2.EXACT
        or geometry.uncertainty != UncertaintyBudgetV2()
        or type(geometry.shape) is not UprightBox3DV2
        or not _identity_transform(
            geometry.anchor_from_geometry, require_zero_translation=True
        )
    ):
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_CONTACT_GEOMETRY"
        )
    if (
        type(surface) is not SupportSurfaceFactV2_2
        or surface.region_approximation is not GeometryApproximationV2.EXACT
        or surface.boundary_policy is not RegionBoundaryPolicyV2.CLOSED
        or surface.geometry_uncertainty != UncertaintyBudgetV2()
        or (
            surface.normal_in_anchor.x,
            surface.normal_in_anchor.y,
            surface.normal_in_anchor.z,
        )
        != (0.0, 0.0, 1.0)
        or not _identity_transform(surface.anchor_from_surface)
        or not _identity_transform(owner_transform)
    ):
        raise _UnsupportedSupportProjectionV2("UNSUPPORTED_MODEL:SUPPORT_SURFACE")
    try:
        ExactAxisAlignedRectV2.from_planar_region(surface.region_uv)
    except UnsupportedRectRegionErrorV2 as error:
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_SURFACE_RECTANGLE"
        ) from error
    return constraint, subject, geometry, surface, owner_transform


def _identity_transform(
    transform: DirectedYawIntervalTransformV2_2,
    *,
    require_zero_translation: bool = False,
) -> bool:
    if type(transform) is not DirectedYawIntervalTransformV2_2:
        return False
    if transform.yaw_radians != 0.0:
        return False
    return not require_zero_translation or (
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ) == (0.0, 0.0, 0.0)


def _support_bounds(
    constraint,
    subject_transform: DirectedYawIntervalTransformV2_2,
    geometry: GeometryInstanceV2_2,
    surface: SupportSurfaceFactV2_2,
    owner_transform: DirectedYawIntervalTransformV2_2,
    box: OrientedUprightBoxBoundsV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[
    tuple[Fraction, Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction, Fraction],
]:
    budget.consume_domain(24)
    surface_rect = ExactAxisAlignedRectV2.from_planar_region(surface.region_uv)
    surface_bounds = surface_rect.bounds
    assert surface_bounds is not None
    owner_x = Fraction.from_float(owner_transform.translation.x)
    owner_y = Fraction.from_float(owner_transform.translation.y)
    owner_z = Fraction.from_float(owner_transform.translation.z)
    surface_x = Fraction.from_float(surface.anchor_from_surface.translation.x)
    surface_y = Fraction.from_float(surface.anchor_from_surface.translation.y)
    surface_z = Fraction.from_float(surface.anchor_from_surface.translation.z)
    center_x = Fraction.from_float(subject_transform.translation.x)
    center_y = Fraction.from_float(subject_transform.translation.y)
    center_z = Fraction.from_float(subject_transform.translation.z)
    contact_z = center_z - box.half_extent_z
    gap = contact_z - (owner_z + surface_z)
    if not (
        Fraction.from_float(constraint.contact_gap_min_m)
        <= gap
        <= Fraction.from_float(constraint.contact_gap_max_m)
    ):
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_CONTACT_GAP_EMPTY"
        )
    contact_area = Fraction.from_float(geometry.shape.size_m.x) * Fraction.from_float(
        geometry.shape.size_m.y
    )
    if contact_area < Fraction.from_float(constraint.minimum_overlap_area_m2):
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_CONTACT_AREA_EMPTY"
        )
    margin = Fraction.from_float(constraint.stability_margin_m)
    inset = (
        owner_x + surface_x + surface_bounds[0] + margin,
        owner_y + surface_y + surface_bounds[1] + margin,
        owner_x + surface_x + surface_bounds[2] - margin,
        owner_y + surface_y + surface_bounds[3] - margin,
    )
    inner = (
        inset[0] - center_x + box.x_radius.rational_upper,
        inset[1] - center_y + box.y_radius.rational_upper,
        inset[2] - center_x - box.x_radius.rational_upper,
        inset[3] - center_y - box.y_radius.rational_upper,
    )
    outer = (
        inset[0] - center_x + box.x_radius.rational_lower,
        inset[1] - center_y + box.y_radius.rational_lower,
        inset[2] - center_x - box.x_radius.rational_lower,
        inset[3] - center_y - box.y_radius.rational_lower,
    )
    for value in (*inner, *outer):
        so2_interval._require_numeric_fraction_cap(
            value, "NUMERIC_GAP:SUPPORT_PROJECTION_FRACTION_BIT_CAP"
        )
    return inner, outer


def _complex_from_bounds(
    bounds: tuple[Fraction, Fraction, Fraction, Fraction],
    universe: ExactAxisAlignedRectV2,
    *,
    cell_id: str,
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionComplexV2:
    budget.consume_domain(16)
    rectangle = ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=bounds[0],
        min_y_m=bounds[1],
        max_x_m=bounds[2],
        max_y_m=bounds[3],
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    ).intersect(universe)
    if rectangle.topology is RectTopologyV2.EMPTY:
        return _strict_complex((), universe)
    if rectangle.topology is RectTopologyV2.DEGENERATE:
        raise _UnsupportedSupportProjectionV2(
            "UNSUPPORTED_MODEL:SUPPORT_PROJECTION_DEGENERATE_LOCUS"
        )
    clipped = rectangle.bounds
    universe_bounds = universe.bounds
    assert clipped is not None and universe_bounds is not None
    semantic_planes = tuple(
        sorted(
            (
                _canonical_plane(Fraction(-1), Fraction(), -bounds[0]),
                _canonical_plane(Fraction(1), Fraction(), bounds[2]),
                _canonical_plane(Fraction(), Fraction(-1), -bounds[1]),
                _canonical_plane(Fraction(), Fraction(1), bounds[3]),
            ),
            key=lambda plane: (
                plane.normal_x,
                plane.normal_y,
                plane.offset,
                1,
            ),
        )
    )
    universe_planes = (
        _canonical_plane(Fraction(-1), Fraction(), -universe_bounds[0]),
        _canonical_plane(Fraction(1), Fraction(), universe_bounds[2]),
        _canonical_plane(Fraction(), Fraction(-1), -universe_bounds[1]),
        _canonical_plane(Fraction(), Fraction(1), universe_bounds[3]),
    )
    closure = RationalConvexPolygonV2(
        vertices_ccw=(
            RationalPoint2V2(x=clipped[0], y=clipped[1]),
            RationalPoint2V2(x=clipped[2], y=clipped[1]),
            RationalPoint2V2(x=clipped[2], y=clipped[3]),
            RationalPoint2V2(x=clipped[0], y=clipped[3]),
        )
    )
    witness = RationalPoint2V2(
        x=(clipped[0] + clipped[2]) / 2,
        y=(clipped[1] + clipped[3]) / 2,
    )
    budget.consume_candidate_cells()
    return _strict_complex(
        (
            StrictConvexIntersectionCellV2(
                cell_id=cell_id,
                half_planes=universe_planes + semantic_planes,
                closure_polygon=closure,
                strict_witness=witness,
            ),
        ),
        universe,
    )


def _strict_complex(
    cells: tuple[StrictConvexIntersectionCellV2, ...],
    universe: ExactAxisAlignedRectV2,
) -> StrictConvexIntersectionComplexV2:
    return StrictConvexIntersectionComplexV2(
        cells=cells,
        universe=universe,
        topology=(
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ),
    )


def _canonical_plane(
    normal_x: Fraction,
    normal_y: Fraction,
    offset: Fraction,
) -> RationalHalfPlane2V2:
    for value in (normal_x, normal_y, offset):
        so2_interval._require_numeric_fraction_cap(
            value, "NUMERIC_GAP:SUPPORT_PROJECTION_HALF_PLANE_BIT_CAP"
        )
    denominator = lcm(normal_x.denominator, normal_y.denominator, offset.denominator)
    values = (
        normal_x.numerator * (denominator // normal_x.denominator),
        normal_y.numerator * (denominator // normal_y.denominator),
        offset.numerator * (denominator // offset.denominator),
    )
    divisor = gcd(gcd(abs(values[0]), abs(values[1])), abs(values[2])) or 1
    return RationalHalfPlane2V2(
        normal_x=Fraction(values[0] // divisor),
        normal_y=Fraction(values[1] // divisor),
        offset=Fraction(values[2] // divisor),
        relation=RationalHalfPlaneRelationV2.LE,
    )


def _require_bounds_tuple(
    value: object, *, label: str
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} must be an exact four-Fraction tuple")
    if any(type(item) is not Fraction for item in value):
        raise TypeError(f"{label} must contain exact Fractions")
    checked = value
    if checked[0] > checked[2] or checked[1] > checked[3]:
        raise ValueError(f"{label} must be ordered")
    return checked


def _copy_bracket(
    value: ContinuousYawSupportProjectionBracketV2,
) -> ContinuousYawSupportProjectionBracketV2:
    return ContinuousYawSupportProjectionBracketV2(
        support_constraint_id=value.support_constraint_id,
        surface_id=value.surface_id,
        contact_geometry_id=value.contact_geometry_id,
        support_projection_kernel_id=value.support_projection_kernel_id,
        support_projection_kernel_version=value.support_projection_kernel_version,
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        inner_bounds=value.inner_bounds,
        outer_bounds=value.outer_bounds,
        so2_atomic_steps_used=value.so2_atomic_steps_used,
        domain_operations_used=value.domain_operations_used,
        candidate_cells_used=value.candidate_cells_used,
    )


def _failure(
    kind: ContinuousYawSupportProjectionKindV2,
    *finding_codes: str,
) -> ContinuousYawSupportProjectionOutcomeV2:
    return ContinuousYawSupportProjectionOutcomeV2(
        kind=kind,
        finding_codes=tuple(finding_codes),
    )


__all__ = (
    "CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2",
    "CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2",
    "ContinuousYawSupportProjectionBracketV2",
    "ContinuousYawSupportProjectionKindV2",
    "ContinuousYawSupportProjectionOutcomeV2",
    "compile_exact_horizontal_support_projection_v2",
)
