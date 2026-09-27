"""Exact search-universe preparation below candidate and collision compilation."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar
from spatialcf.core._internal.kernels.rect import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectKernelProjectionErrorV2,
    RectTopologyV2,
    UnsupportedRectRegionErrorV2,
    WorldPointXYV2,
)
from spatialcf.core._internal.resources import DomainOperationBudgetV2
from spatialcf.domain.artifacts import (
    ArtifactCoverageV2,
    DomainCompletenessV2,
    PlanarDomainBoundsV2,
    PlanarRegionBoundV2,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    UncertaintyBudgetV2,
    Vec2,
)
from spatialcf.domain.constraints import PositionRegionInterpretation
from spatialcf.domain.geometry import (
    ExtrudedPlanarPolygonV2,
    GeometryApproximationV2,
    PlanarRegionV2,
    UprightBox3DV2,
)
from spatialcf.domain.problem import SemanticProblemV2
from spatialcf.domain.result import UncertifiedReasonV2
from spatialcf.domain.scene import CanonicalObject


class _ResourceLimitErrorV2(RuntimeError):
    pass


@dataclass(slots=True)
class _DomainOperationBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = _ResourceLimitErrorV2


@dataclass(frozen=True, slots=True)
class _SearchUniverseV2:
    world_rects: tuple[ExactAxisAlignedRectV2, ...]
    delta_rect: ExactAxisAlignedRectV2
    domain: PlanarDomainBoundsV2


@dataclass(frozen=True, slots=True)
class _SearchUniverseFailureV2:
    uncertified_reason: UncertifiedReasonV2
    finding_codes: tuple[str, ...]


def _compile_search_universe(
    problem: SemanticProblemV2,
    budget: _DomainOperationBudgetV2,
) -> _SearchUniverseV2 | _SearchUniverseFailureV2:
    position = problem.constraints.position_domain
    facts = problem.scene.workspace_boundaries
    if facts.availability is FactAvailabilityV2.MISSING:
        return _search_failure(
            problem,
            UncertifiedReasonV2.MISSING_FACT,
            *(
                f"MISSING_FACT:WORKSPACE_AVAILABILITY:{fact_id}"
                for fact_id in position.workspace_fact_ids
            ),
        )
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_AVAILABILITY:"
            f"{facts.availability.value}",
        )
    if facts.completeness is not FactCompletenessV2.EXACT:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_COMPLETENESS:"
            f"{facts.completeness.value}",
        )
    by_id = {item.fact_id: item for item in facts.values or ()}
    selected = tuple(by_id.get(fact_id) for fact_id in position.workspace_fact_ids)
    missing_references = tuple(
        fact_id
        for fact_id, item in zip(position.workspace_fact_ids, selected, strict=True)
        if item is None
    )
    if not selected or missing_references:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *(
                f"UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_REFERENCE:{fact_id}"
                for fact_id in missing_references
            ),
        )
    workspace_facts = tuple(item for item in selected if item is not None)
    unsupported_approximations = tuple(
        item
        for item in workspace_facts
        if item.region_approximation
        not in {GeometryApproximationV2.EXACT, GeometryApproximationV2.OUTER}
    )
    if unsupported_approximations:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            *(
                "UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_APPROXIMATION:"
                f"{item.fact_id}:{item.region_approximation.value}"
                for item in unsupported_approximations
            ),
        )
    if facts.uncertainty is None:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_UNCERTAINTY:FACT_SET",
        )
    family_margin = _planar_outer_margin(facts.uncertainty)
    if family_margin is None:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_UNCERTAINTY:FACT_SET",
        )

    world_rects: list[ExactAxisAlignedRectV2] = []
    for item in workspace_facts:
        try:
            budget.consume()
            rect = ExactAxisAlignedRectV2.from_planar_region(
                item.region_world_xy,
                coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
            )
        except UnsupportedRectRegionErrorV2:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                f"UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_REGION:{item.fact_id}",
            )
        item_margin = _planar_outer_margin(item.geometry_uncertainty)
        if item_margin is None:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                f"UNSUPPORTED_SEARCH_UNIVERSE:WORKSPACE_UNCERTAINTY:{item.fact_id}",
            )
        total_margin = family_margin + item_margin
        if total_margin:
            budget.consume()
            rect = rect.dilate_axis(AxisMarginXYV2(x_m=total_margin, y_m=total_margin))
        world_rects.append(rect)

    budget.consume()
    world_envelope = ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min(_require_bound(rect.min_x_m) for rect in world_rects),
        min_y_m=min(_require_bound(rect.min_y_m) for rect in world_rects),
        max_x_m=max(_require_bound(rect.max_x_m) for rect in world_rects),
        max_y_m=max(_require_bound(rect.max_y_m) for rect in world_rects),
        coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
    )
    anchor_margin = _search_anchor_margin(problem)
    if isinstance(anchor_margin, _SearchUniverseFailureV2):
        return anchor_margin
    if anchor_margin:
        budget.consume()
        world_envelope = world_envelope.dilate_axis(
            AxisMarginXYV2(x_m=anchor_margin, y_m=anchor_margin)
        )
    anchor = _baseline_anchor(problem)
    budget.consume()
    delta_rect = world_envelope.relative_to(
        WorldPointXYV2.from_binary64(x_m=anchor.x, y_m=anchor.y)
    )
    try:
        budget.consume()
        outer_projection = delta_rect.project_outer()
        if outer_projection.topology is not RectTopologyV2.AREA:
            return _search_failure(
                problem,
                UncertifiedReasonV2.NUMERIC_GAP,
                "UNSUPPORTED_SEARCH_UNIVERSE:PROJECTION",
            )
        outer_region = outer_projection.to_planar_region()
    except RectKernelProjectionErrorV2:
        return _search_failure(
            problem,
            UncertifiedReasonV2.NUMERIC_GAP,
            "UNSUPPORTED_SEARCH_UNIVERSE:PROJECTION",
        )
    universe_domain = _exact_nonempty_domain(outer_region)
    return _SearchUniverseV2(
        world_rects=tuple(world_rects),
        delta_rect=delta_rect,
        domain=universe_domain,
    )


def _search_failure(
    problem: SemanticProblemV2,
    reason: UncertifiedReasonV2,
    *finding_codes: str,
) -> _SearchUniverseFailureV2:
    missing = tuple(
        sorted(
            code
            for code in problem.certification_blockers
            if code.startswith("MISSING_FACT:")
        )
    )
    if missing:
        reason = UncertifiedReasonV2.MISSING_FACT
    return _SearchUniverseFailureV2(
        uncertified_reason=reason,
        finding_codes=tuple(sorted({*finding_codes, *missing})),
    )


def _planar_outer_margin(
    uncertainty: UncertaintyBudgetV2,
) -> Fraction | None:
    policies = (uncertainty.source_error, uncertainty.shape_approximation)
    if any(
        value != 0.0
        for policy in policies
        for value in (
            policy.area_tolerance_m2,
            policy.angular_tolerance_rad,
            policy.pixel_tolerance_px,
            policy.fraction_tolerance,
        )
    ):
        return None
    return sum(
        (Fraction.from_float(policy.linear_tolerance_m) for policy in policies),
        start=Fraction(),
    )


def _search_anchor_margin(
    problem: SemanticProblemV2,
) -> Fraction | _SearchUniverseFailureV2:
    margin = Fraction.from_float(problem.numeric_policy.linear_tolerance_m)
    position = problem.constraints.position_domain
    if (
        position.region_interpretation
        is PositionRegionInterpretation.SUBJECT_ANCHOR_LOCUS
    ):
        return margin
    occupancy_radius = _occupancy_outer_radius(problem)
    if isinstance(occupancy_radius, _SearchUniverseFailureV2):
        return occupancy_radius
    return margin + occupancy_radius


def _occupancy_outer_radius(
    problem: SemanticProblemV2,
) -> Fraction | _SearchUniverseFailureV2:
    position = problem.constraints.position_domain
    bodies = problem.scene.collision_bodies
    geometries = problem.scene.geometry_instances
    subject_id = problem.constraints.allowed_edit.subject_id
    subject = next(
        item
        for item in problem.scene.objects.values or ()
        if item.object_id == subject_id
    )
    subject_rotation = subject.pose.world_from_object.rotation
    if (
        subject_rotation.x,
        subject_rotation.y,
        subject_rotation.z,
        subject_rotation.w,
    ) != (0.0, 0.0, 0.0, 1.0):
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            f"UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_ROTATION:{subject_id}",
        )
    if bodies.availability is FactAvailabilityV2.MISSING:
        return _search_failure(
            problem,
            UncertifiedReasonV2.MISSING_FACT,
            *(
                f"MISSING_FACT:OCCUPANCY_BODY_FACTS:{body_id}"
                for body_id in position.subject_occupancy_body_ids
            ),
        )
    if bodies.availability is not FactAvailabilityV2.KNOWN:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_BODY_FACTS:"
            f"{bodies.availability.value}",
        )
    if bodies.completeness is not FactCompletenessV2.EXACT:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_BODY_FACTS:"
            f"{bodies.completeness.value}",
        )

    body_by_id = {item.body_id: item for item in bodies.values or ()}
    selected_geometry_ids: list[str] = []
    for body_id in position.subject_occupancy_body_ids:
        body = body_by_id.get(body_id)
        if body is None:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                f"UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_BODY_REFERENCE:{body_id}",
            )
        selected_geometry_ids.extend(body.geometry_instance_ids)
    if not selected_geometry_ids:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_BODY_REFERENCE:EMPTY",
        )

    if geometries.availability is FactAvailabilityV2.MISSING:
        return _search_failure(
            problem,
            UncertifiedReasonV2.MISSING_FACT,
            *(
                f"MISSING_FACT:OCCUPANCY_GEOMETRY_FACTS:{geometry_id}"
                for geometry_id in selected_geometry_ids
            ),
        )
    if geometries.availability is not FactAvailabilityV2.KNOWN:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_FACTS:"
            f"{geometries.availability.value}",
        )
    if geometries.completeness is not FactCompletenessV2.EXACT:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_FACTS:"
            f"{geometries.completeness.value}",
        )
    if geometries.uncertainty is None:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_UNCERTAINTY:FACT_SET",
        )
    family_linear = _planar_outer_margin(geometries.uncertainty)
    if family_linear is None:
        return _search_failure(
            problem,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_UNCERTAINTY:FACT_SET",
        )

    geometry_by_id = {item.geometry_id: item for item in geometries.values or ()}
    radii: list[Fraction] = []
    for geometry_id in selected_geometry_ids:
        geometry = geometry_by_id.get(geometry_id)
        if geometry is None:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_REFERENCE:"
                f"{geometry_id}",
            )
        if geometry.approximation not in {
            GeometryApproximationV2.EXACT,
            GeometryApproximationV2.OUTER,
        }:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_APPROXIMATION:"
                f"{geometry_id}:{geometry.approximation.value}",
            )
        rotation = geometry.anchor_from_geometry.rotation
        # A future wider subset must derive a directed Fraction bracket for
        # the complete stored quaternion transform.  Binary64 "unit" checks
        # alone are insufficient when very large offsets amplify their error.
        if (
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ) != (0.0, 0.0, 0.0, 1.0):
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                f"UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_ROTATION:{geometry_id}",
            )
        item_linear = _planar_outer_margin(geometry.uncertainty)
        if item_linear is None:
            return _search_failure(
                problem,
                UncertifiedReasonV2.UNSUPPORTED_MODEL,
                "UNSUPPORTED_SEARCH_UNIVERSE:OCCUPANCY_GEOMETRY_UNCERTAINTY:"
                f"{geometry_id}",
            )
        translation = geometry.anchor_from_geometry.translation
        anchor_l1 = abs(Fraction.from_float(translation.x)) + abs(
            Fraction.from_float(translation.y)
        )
        shape_l1 = _shape_local_l1_radius(geometry.shape)
        radii.append(anchor_l1 + shape_l1 + family_linear + item_linear)
    return max(radii)


def _shape_local_l1_radius(
    shape: UprightBox3DV2 | ExtrudedPlanarPolygonV2,
) -> Fraction:
    if isinstance(shape, UprightBox3DV2):
        return (
            abs(Fraction.from_float(shape.size_m.x))
            + abs(Fraction.from_float(shape.size_m.y))
        ) / 2
    return max(
        abs(Fraction.from_float(point.x)) + abs(Fraction.from_float(point.y))
        for point in shape.footprint.exterior.vertices
    )


def _baseline_anchor(problem: SemanticProblemV2) -> Vec2:
    subject_id = problem.constraints.allowed_edit.subject_id
    subject = next(
        item
        for item in problem.scene.objects.values or ()
        if item.object_id == subject_id
    )
    return _object_anchor_world_xy(subject)


def _object_anchor_world_xy(subject: CanonicalObject) -> Vec2:
    translation = subject.pose.world_from_object.translation
    return Vec2(x=translation.x, y=translation.y)


def _exact_nonempty_domain(region: PlanarRegionV2) -> PlanarDomainBoundsV2:
    bound = PlanarRegionBoundV2.non_empty(region)
    return PlanarDomainBoundsV2(
        inner_bound=bound,
        outer_bound=bound,
        completeness=DomainCompletenessV2.EXACT,
        coverage=ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
    )


def _require_bound(value: Fraction | None) -> Fraction:
    if value is None:
        raise ValueError("area rectangle unexpectedly lacks a finite bound")
    return value


# Preserve the supported original pickle/import lookup.
_ResourceLimitErrorV2.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_DomainOperationBudgetV2.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_SearchUniverseV2.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_SearchUniverseFailureV2.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_compile_search_universe.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_search_failure.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_planar_outer_margin.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_search_anchor_margin.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_occupancy_outer_radius.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_shape_local_l1_radius.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_baseline_anchor.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_object_anchor_world_xy.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_exact_nonempty_domain.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
_require_bound.__module__ = "spatialcf.core._internal.compilation.candidate_cells"
