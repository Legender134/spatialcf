"""Sound collision-domain compilation for the exact Canonical v2 box subset.

The only variable is the subject's world-XY translation delta.  Exact rational
configuration obstacles are subtracted from a caller-supplied finite search
universe while retaining their contact boundary.  Positive Euclidean
clearance is published as a conservative inner/outer rectilinear bracket.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.rect_kernel import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearTopologyV2,
    difference_rectilinear_region_v2,
    intersect_rectilinear_regions_v2,
    normalize_rectilinear_region_v2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    QuaternionV2,
    RigidTransformV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    CollisionClearanceMetricV2,
    CollisionConstraintV2,
)
from spatialcf.domain.v2.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import CanonicalObjectV2


class CollisionDomainKindV2(StrEnum):
    """Mathematical effect of one collision predicate on finite edit deltas."""

    REGION_BRACKET = "REGION_BRACKET"
    IDENTITY = "IDENTITY"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class CollisionDomainCompilationOutcomeV2:
    """Closed result of compiling one collision predicate.

    ``IDENTITY`` means every point in the supplied search universe is allowed.
    A ``REGION_BRACKET`` always satisfies ``inner_allowed_delta`` subset
    ``outer_allowed_delta``; both are already clipped to that universe.
    """

    kind: CollisionDomainKindV2
    inner_allowed_delta: ExactRectilinearRegionV2 | None = None
    outer_allowed_delta: ExactRectilinearRegionV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CollisionDomainKindV2):
            raise TypeError("kind must be a CollisionDomainKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(item) is not str for item in self.finding_codes
        ):
            raise TypeError("finding_codes must be an exact tuple of strings")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )

        if self.kind is CollisionDomainKindV2.REGION_BRACKET:
            if self.inner_allowed_delta is None or self.outer_allowed_delta is None:
                raise ValueError("REGION_BRACKET requires both inner and outer regions")
            _validate_delta_region_shell(self.inner_allowed_delta)
            _validate_delta_region_shell(self.outer_allowed_delta)
            if self.outer_allowed_delta.topology is RectilinearTopologyV2.EMPTY:
                raise ValueError("an empty outer region must use EMPTY")
            if not _rectilinear_subset(
                self.inner_allowed_delta,
                self.outer_allowed_delta,
            ):
                raise ValueError("inner allowed delta must be a subset of outer")
            if self.finding_codes:
                raise ValueError("REGION_BRACKET cannot carry findings")
            return

        if self.inner_allowed_delta is not None or self.outer_allowed_delta is not None:
            raise ValueError(f"{self.kind.value} must not carry allowed regions")
        if self.kind is CollisionDomainKindV2.IDENTITY:
            if self.finding_codes:
                raise ValueError("IDENTITY cannot carry findings")
            return
        if self.kind is CollisionDomainKindV2.EMPTY:
            if not self.finding_codes or any(
                not item.startswith("EXACT_EMPTY:") for item in self.finding_codes
            ):
                raise ValueError("EMPTY requires an exact cause finding")
            return
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} requires a finding")


CollisionDomainOutcomeV2 = CollisionDomainCompilationOutcomeV2


@dataclass(frozen=True, slots=True)
class _WorldBoxV2:
    min_x_m: Fraction
    min_y_m: Fraction
    min_z_m: Fraction
    max_x_m: Fraction
    max_y_m: Fraction
    max_z_m: Fraction


def compile_collision_domain_v2(
    problem: SemanticProblemV2,
    constraint: CollisionConstraintV2 | str,
    search_universe: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> CollisionDomainCompilationOutcomeV2:
    """Compile one exact box-union collision predicate over XY translations.

    The semantic root, an optional caller-supplied constraint object, and the
    exact search universe are independently reconstructed or normalized before
    they can influence a certified result.
    """

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if not isinstance(problem, SemanticProblemV2):
        raise TypeError("problem must be a SemanticProblemV2")
    checked_problem = SemanticProblemV2.model_validate(
        problem.model_dump(mode="python"),
        strict=True,
    )
    selected = _resolve_constraint(checked_problem, constraint)
    checked_universe = _revalidate_search_universe(
        search_universe,
        atomic_budget=budget,
    )
    if isinstance(checked_universe, CollisionDomainCompilationOutcomeV2):
        return checked_universe
    if checked_universe.topology is RectilinearTopologyV2.EMPTY:
        constraint_id = (
            _requested_constraint_id(constraint)
            if isinstance(selected, CollisionDomainCompilationOutcomeV2)
            else selected.constraint_id
        )
        return _empty(f"EXACT_EMPTY:SEARCH_UNIVERSE:{constraint_id}")
    if isinstance(selected, CollisionDomainCompilationOutcomeV2):
        return selected

    findings = _supported_subset_findings(checked_problem, selected)
    if findings:
        return _unknown(*findings)

    bodies = {
        item.body_id: item
        for item in checked_problem.scene.collision_bodies.values or ()
    }
    geometries = {
        item.geometry_id: item
        for item in checked_problem.scene.geometry_instances.values or ()
    }
    objects = {
        item.object_id: item for item in checked_problem.scene.objects.values or ()
    }
    subject_boxes = _boxes_for_bodies(
        tuple(bodies[body_id] for body_id in selected.subject_body_ids),
        geometries,
        objects,
    )
    obstacle_boxes = _boxes_for_bodies(
        tuple(bodies[body_id] for body_id in selected.obstacle_body_ids),
        geometries,
        objects,
    )
    pair_count = len(subject_boxes) * len(obstacle_boxes)
    try:
        budget.consume(pair_count)
    except RectilinearAtomicBudgetExhaustedV2:
        return _resource_limit()

    clearance_m = Fraction.from_float(selected.minimum_clearance_m)
    outer_forbidden: list[ExactAxisAlignedRectV2] = []
    inner_forbidden: list[ExactAxisAlignedRectV2] = []
    for subject_box in subject_boxes:
        for obstacle_box in obstacle_boxes:
            q0 = _pair_configuration_obstacle(
                subject_box,
                obstacle_box,
                clearance_m=clearance_m,
            )
            if q0 is None:
                continue
            outer_forbidden.append(q0)
            inner_forbidden.append(
                q0
                if clearance_m == 0
                else q0.dilate_axis(AxisMarginXYV2(x_m=clearance_m, y_m=clearance_m))
            )

    if not outer_forbidden:
        return CollisionDomainCompilationOutcomeV2(kind=CollisionDomainKindV2.IDENTITY)

    outer_forbidden_region = normalize_rectilinear_region_v2(
        tuple(outer_forbidden),
        atomic_budget=budget,
    )
    if outer_forbidden_region.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource_limit()
    assert outer_forbidden_region.region is not None

    if clearance_m == 0:
        inner_forbidden_region = outer_forbidden_region
    else:
        inner_forbidden_region = normalize_rectilinear_region_v2(
            tuple(inner_forbidden),
            atomic_budget=budget,
        )
        if inner_forbidden_region.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
            return _resource_limit()
    assert inner_forbidden_region.region is not None

    outer_allowed = difference_rectilinear_region_v2(
        checked_universe,
        outer_forbidden_region.region,
        atomic_budget=budget,
    )
    if outer_allowed.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource_limit()
    if clearance_m == 0:
        inner_allowed = outer_allowed
    else:
        inner_allowed = difference_rectilinear_region_v2(
            checked_universe,
            inner_forbidden_region.region,
            atomic_budget=budget,
        )
        if inner_allowed.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
            return _resource_limit()
    assert outer_allowed.region is not None and inner_allowed.region is not None

    if outer_allowed.region.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:COLLISION_DOMAIN:{selected.constraint_id}")
    return CollisionDomainCompilationOutcomeV2(
        kind=CollisionDomainKindV2.REGION_BRACKET,
        inner_allowed_delta=inner_allowed.region,
        outer_allowed_delta=outer_allowed.region,
    )


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


def _resolve_constraint(
    problem: SemanticProblemV2,
    requested: CollisionConstraintV2 | str,
) -> CollisionConstraintV2 | CollisionDomainCompilationOutcomeV2:
    if isinstance(requested, CollisionConstraintV2):
        checked = CollisionConstraintV2.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a CollisionConstraintV2 or exact str ID")

    registered = next(
        (
            item
            for item in problem.constraints.collision_constraints
            if item.constraint_id == constraint_id
        ),
        None,
    )
    if registered is None:
        return _unknown(f"UNKNOWN_COLLISION_CONSTRAINT:{constraint_id}")
    if checked is not None and checked != registered:
        return _unknown(f"COLLISION_CONSTRAINT_MISMATCH:{constraint_id}")
    return registered


def _requested_constraint_id(requested: CollisionConstraintV2 | str) -> str:
    if isinstance(requested, CollisionConstraintV2):
        return requested.constraint_id
    if type(requested) is str:
        return requested
    raise TypeError("constraint must be a CollisionConstraintV2 or exact str ID")


def _revalidate_search_universe(
    search_universe: ExactRectilinearRegionV2,
    *,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | CollisionDomainCompilationOutcomeV2:
    if not isinstance(search_universe, ExactRectilinearRegionV2):
        raise TypeError("search_universe must be an ExactRectilinearRegionV2")
    checked = intersect_rectilinear_regions_v2(
        search_universe,
        search_universe,
        atomic_budget=atomic_budget,
    )
    if checked.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource_limit()
    assert checked.region is not None
    return checked.region


def _supported_subset_findings(
    problem: SemanticProblemV2,
    constraint: CollisionConstraintV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if (
        constraint.clearance_metric
        is not CollisionClearanceMetricV2.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
    ):
        findings.append(
            f"UNSUPPORTED_COLLISION_DOMAIN:CLEARANCE_METRIC:{constraint.constraint_id}"
        )
    if constraint.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append(
            f"UNSUPPORTED_COLLISION_DOMAIN:BOUNDARY_POLICY:{constraint.constraint_id}"
        )
    if constraint.support_contact_exceptions:
        findings.append(
            "UNSUPPORTED_COLLISION_DOMAIN:SUPPORT_CONTACT_EXCEPTIONS:"
            f"{constraint.constraint_id}"
        )
    if not _numeric_policy_is_zero(problem.numeric_policy):
        findings.append("UNSUPPORTED_COLLISION_DOMAIN:NUMERIC_POLICY")

    for label, facts in (
        ("OBJECTS", problem.scene.objects),
        ("COLLISION_BODIES", problem.scene.collision_bodies),
        ("GEOMETRY_INSTANCES", problem.scene.geometry_instances),
    ):
        findings.extend(_fact_family_findings(label, facts))

    if not _all_exact_families(problem):
        return tuple(sorted(set(findings)))

    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    bodies = {
        item.body_id: item for item in problem.scene.collision_bodies.values or ()
    }
    geometries = {
        item.geometry_id: item for item in problem.scene.geometry_instances.values or ()
    }
    relevant_body_ids = tuple(
        sorted((*constraint.subject_body_ids, *constraint.obstacle_body_ids))
    )
    relevant_geometry_ids: set[str] = set()
    relevant_owner_ids: set[str] = set()
    for body_id in relevant_body_ids:
        body = bodies.get(body_id)
        if body is None:
            findings.append(f"MISSING_FACT:COLLISION_BODY:{body_id}")
            continue
        if body.composition != "CLOSED_SOLID_UNION":
            findings.append(f"UNSUPPORTED_COLLISION_DOMAIN:BODY_COMPOSITION:{body_id}")
        relevant_geometry_ids.update(body.geometry_instance_ids)
        if body.owner_object_id is not None:
            relevant_owner_ids.add(body.owner_object_id)
            owner = objects.get(body.owner_object_id)
            if owner is None:
                findings.append(
                    f"MISSING_FACT:COLLISION_BODY_OWNER:{body.owner_object_id}"
                )

    for owner_id in sorted(relevant_owner_ids):
        owner = objects.get(owner_id)
        if owner is not None and not _has_exact_identity_rotation(
            owner.pose.world_from_object
        ):
            findings.append(
                "UNSUPPORTED_COLLISION_DOMAIN:NON_IDENTITY_ROTATION:"
                f"OBJECT_POSE:{owner_id}"
            )

    for geometry_id in sorted(relevant_geometry_ids):
        geometry = geometries.get(geometry_id)
        if geometry is None:
            findings.append(f"MISSING_FACT:COLLISION_GEOMETRY:{geometry_id}")
            continue
        if geometry.role is not GeometryRoleV2.COLLISION:
            findings.append(f"UNSUPPORTED_COLLISION_DOMAIN:GEOMETRY_ROLE:{geometry_id}")
        if geometry.approximation is not GeometryApproximationV2.EXACT:
            findings.append(
                "UNSUPPORTED_COLLISION_DOMAIN:GEOMETRY_APPROXIMATION:"
                f"{geometry_id}:{geometry.approximation.value}"
            )
        if not isinstance(geometry.shape, UprightBox3DV2):
            findings.append(
                f"UNSUPPORTED_COLLISION_DOMAIN:GEOMETRY_SHAPE:{geometry_id}"
            )
        if not _uncertainty_is_zero(geometry.uncertainty):
            findings.append(
                f"UNSUPPORTED_COLLISION_DOMAIN:GEOMETRY_ITEM_UNCERTAINTY:{geometry_id}"
            )
        if not _has_exact_identity_rotation(geometry.anchor_from_geometry):
            findings.append(
                "UNSUPPORTED_COLLISION_DOMAIN:NON_IDENTITY_ROTATION:"
                f"GEOMETRY_ANCHOR:{geometry_id}"
            )
    return tuple(sorted(set(findings)))


def _all_exact_families(problem: SemanticProblemV2) -> bool:
    return all(
        facts.availability is FactAvailabilityV2.KNOWN
        and facts.completeness is FactCompletenessV2.EXACT
        and facts.uncertainty is not None
        and _uncertainty_is_zero(facts.uncertainty)
        for facts in (
            problem.scene.objects,
            problem.scene.collision_bodies,
            problem.scene.geometry_instances,
        )
    )


def _fact_family_findings(label: str, facts: FactSetV2) -> tuple[str, ...]:
    if facts.availability is FactAvailabilityV2.MISSING:
        return (f"MISSING_FACT:{label}",)
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return (f"UNSUPPORTED_COLLISION_DOMAIN:{label}_AVAILABILITY",)
    findings: list[str] = []
    if facts.completeness is not FactCompletenessV2.EXACT:
        completeness = (
            facts.completeness.value if facts.completeness is not None else "NONE"
        )
        findings.append(
            f"UNSUPPORTED_COLLISION_DOMAIN:{label}_COMPLETENESS:{completeness}"
        )
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_COLLISION_DOMAIN:{label}_FACT_UNCERTAINTY")
    return tuple(findings)


def _has_exact_identity_rotation(transform: RigidTransformV2) -> bool:
    rotation: QuaternionV2 = transform.rotation
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _boxes_for_bodies(
    bodies: tuple[CollisionBodyFactV2, ...],
    geometries: dict[str, GeometryInstanceV2],
    objects: dict[str, CanonicalObjectV2],
) -> tuple[_WorldBoxV2, ...]:
    boxes: list[_WorldBoxV2] = []
    for body in sorted(bodies, key=lambda item: item.body_id):
        owner_translation = (Fraction(), Fraction(), Fraction())
        if body.owner_object_id is not None:
            translation = objects[
                body.owner_object_id
            ].pose.world_from_object.translation
            owner_translation = tuple(
                Fraction.from_float(value)
                for value in (translation.x, translation.y, translation.z)
            )
        for geometry_id in sorted(body.geometry_instance_ids):
            geometry = geometries[geometry_id]
            assert isinstance(geometry.shape, UprightBox3DV2)
            geometry_translation = geometry.anchor_from_geometry.translation
            center = tuple(
                owner_translation[index] + Fraction.from_float(value)
                for index, value in enumerate(
                    (
                        geometry_translation.x,
                        geometry_translation.y,
                        geometry_translation.z,
                    )
                )
            )
            half_size = tuple(
                Fraction.from_float(value) / 2
                for value in (
                    geometry.shape.size_m.x,
                    geometry.shape.size_m.y,
                    geometry.shape.size_m.z,
                )
            )
            boxes.append(
                _WorldBoxV2(
                    min_x_m=center[0] - half_size[0],
                    min_y_m=center[1] - half_size[1],
                    min_z_m=center[2] - half_size[2],
                    max_x_m=center[0] + half_size[0],
                    max_y_m=center[1] + half_size[1],
                    max_z_m=center[2] + half_size[2],
                )
            )
    return tuple(boxes)


def _pair_configuration_obstacle(
    subject: _WorldBoxV2,
    obstacle: _WorldBoxV2,
    *,
    clearance_m: Fraction,
) -> ExactAxisAlignedRectV2 | None:
    if clearance_m == 0:
        z_interiors_overlap = (
            subject.min_z_m < obstacle.max_z_m and obstacle.min_z_m < subject.max_z_m
        )
        if not z_interiors_overlap:
            return None
    else:
        vertical_gap_m = max(
            obstacle.min_z_m - subject.max_z_m,
            subject.min_z_m - obstacle.max_z_m,
            Fraction(),
        )
        if vertical_gap_m >= clearance_m:
            return None

    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=obstacle.min_x_m - subject.max_x_m,
        min_y_m=obstacle.min_y_m - subject.max_y_m,
        max_x_m=obstacle.max_x_m - subject.min_x_m,
        max_y_m=obstacle.max_y_m - subject.min_y_m,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


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


def _validate_delta_region_shell(region: ExactRectilinearRegionV2) -> None:
    if not isinstance(region, ExactRectilinearRegionV2):
        raise TypeError("allowed delta must be an ExactRectilinearRegionV2")
    if not isinstance(region.topology, RectilinearTopologyV2):
        raise TypeError("allowed delta topology is invalid")
    if type(region.rectangles) is not tuple:
        raise TypeError("allowed delta rectangles must be an exact tuple")
    expected_topology = RectilinearTopologyV2.EMPTY
    for rectangle in region.rectangles:
        if type(rectangle) is not ExactAxisAlignedRectV2:
            raise TypeError("allowed delta rectangles have an invalid value")
        checked = ExactAxisAlignedRectV2(
            coordinate_space=rectangle.coordinate_space,
            topology=rectangle.topology,
            min_x_m=rectangle.min_x_m,
            min_y_m=rectangle.min_y_m,
            max_x_m=rectangle.max_x_m,
            max_y_m=rectangle.max_y_m,
        )
        if checked.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M:
            raise ValueError("allowed delta must use translation-delta coordinates")
        if checked.topology is RectTopologyV2.AREA:
            expected_topology = RectilinearTopologyV2.AREA
        elif (
            checked.topology is RectTopologyV2.DEGENERATE
            and expected_topology is RectilinearTopologyV2.EMPTY
        ):
            expected_topology = RectilinearTopologyV2.DEGENERATE
    if region.topology is not expected_topology:
        raise ValueError("allowed delta topology does not match its rectangles")


def _rectilinear_subset(
    inner: ExactRectilinearRegionV2,
    outer: ExactRectilinearRegionV2,
) -> bool:
    """Decide inclusion on the exact axis-aligned arrangement atoms."""

    if inner.topology is RectilinearTopologyV2.EMPTY:
        return True
    coordinates = inner.rectangles + outer.rectangles
    xs = tuple(
        sorted(
            {
                value
                for rectangle in coordinates
                for value in (rectangle.min_x_m, rectangle.max_x_m)
                if value is not None
            }
        )
    )
    ys = tuple(
        sorted(
            {
                value
                for rectangle in coordinates
                for value in (rectangle.min_y_m, rectangle.max_y_m)
                if value is not None
            }
        )
    )
    x_samples = tuple(
        sorted(set(xs) | {(left + right) / 2 for left, right in itertools.pairwise(xs)})
    )
    y_samples = tuple(
        sorted(
            set(ys) | {(lower + upper) / 2 for lower, upper in itertools.pairwise(ys)}
        )
    )
    return all(
        not inner.contains_point(x_m, y_m) or outer.contains_point(x_m, y_m)
        for x_m in x_samples
        for y_m in y_samples
    )


def _unknown(*findings: str) -> CollisionDomainCompilationOutcomeV2:
    return CollisionDomainCompilationOutcomeV2(
        kind=CollisionDomainKindV2.UNKNOWN,
        finding_codes=tuple(findings),
    )


def _resource_limit() -> CollisionDomainCompilationOutcomeV2:
    return CollisionDomainCompilationOutcomeV2(
        kind=CollisionDomainKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:COLLISION_DOMAIN_ATOMIC_CELLS",),
    )


def _empty(finding: str) -> CollisionDomainCompilationOutcomeV2:
    return CollisionDomainCompilationOutcomeV2(
        kind=CollisionDomainKindV2.EMPTY,
        finding_codes=(finding,),
    )
