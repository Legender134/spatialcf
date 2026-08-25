# ruff: noqa: F811, I001
"""Private current candidate-family compilers; exact migrated bodies."""

from __future__ import annotations

# Migrated from collision_domain.py.
"""Sound collision-domain compilation for the exact Canonical v2 box subset.

The only variable is the subject's world-XY translation delta.  Exact rational
configuration obstacles are subtracted from a caller-supplied finite search
universe while retaining their contact boundary.  Positive Euclidean
clearance is published as a conservative inner/outer rectilinear bracket.
"""


import itertools
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels.rect import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core._internal.kernels.rectilinear import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetExhaustedV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearTopologyV2,
    difference_rectilinear_region_v2,
    intersect_rectilinear_regions_v2,
    normalize_rectilinear_region_v2,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    Quaternion,
    RigidTransformV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    CollisionClearanceMetric,
    CollisionConstraint,
)
from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.problem import SemanticProblemV2
from spatialcf.domain.scene import CanonicalObject


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
    constraint: CollisionConstraint | str,
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
    requested: CollisionConstraint | str,
) -> CollisionConstraint | CollisionDomainCompilationOutcomeV2:
    if isinstance(requested, CollisionConstraint):
        checked = CollisionConstraint.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a CollisionConstraint or exact str ID")

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


def _requested_constraint_id(requested: CollisionConstraint | str) -> str:
    if isinstance(requested, CollisionConstraint):
        return requested.constraint_id
    if type(requested) is str:
        return requested
    raise TypeError("constraint must be a CollisionConstraint or exact str ID")


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
    constraint: CollisionConstraint,
) -> tuple[str, ...]:
    findings: list[str] = []
    if (
        constraint.clearance_metric
        is not CollisionClearanceMetric.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
    ):
        findings.append(
            f"UNSUPPORTED_COLLISION_DOMAIN:CLEARANCE_METRIC:{constraint.constraint_id}"
        )
    if constraint.boundary_policy is not BoundaryPolicy.CLOSED:
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
    rotation: Quaternion = transform.rotation
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _boxes_for_bodies(
    bodies: tuple[CollisionBodyFactV2, ...],
    geometries: dict[str, GeometryInstanceV2],
    objects: dict[str, CanonicalObject],
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


# Migrated from strict_convex_candidate_domain.py.
"""Frozen candidate-stage values for strict convex continuous-yaw domains.

These values are immutable proposals, not proof capabilities. A semantic
consumer must fresh replay the raw problem and compiler config.
"""


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

from spatialcf.core._internal.kernels.convex_partition import (
    ConvexAllowedTranslationBracketV2,
    ConvexAllowedTranslationKindV2,
    compile_convex_allowed_translation_v2,
)
from spatialcf.core._internal.kernels.rect import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core._internal.kernels.so2 import SO2AtomicBudgetV2
from spatialcf.core._internal.resources import (
    DomainOperationBudgetV2,
)
from spatialcf.domain.artifacts import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_5,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec3,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    CollisionClearanceMetric,
    PositionRegionInterpretation,
    RegionAggregation,
)
from spatialcf.domain.geometry import (
    DirectedYawIntervalTransformV2_2,
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

_strict_artifact_hash_domain_v2_2 = b"spatialcf.strict-convex-candidate-artifact.v2.2\0"
_strict_digest_pattern = re.compile(r"[0-9a-f]{64}")


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
            if (
                type(digest) is not str
                or _strict_digest_pattern.fullmatch(digest) is None
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.subject_id) is not str or not self.subject_id.strip():
            raise ValueError("subject_id must be a non-blank exact string")
        checked_universe = _strict_copy_universe(self.search_universe)
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
        checked_bracket = _strict_copy_bracket(self.allowed_domain_bracket)
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
            _strict_artifact_hash_domain_v2_2 + _strict_artifact_bytes(self)
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
            object.__setattr__(self, "artifact", _strict_copy_artifact(self.artifact))
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
                type(value) is not str
                or _strict_digest_pattern.fullmatch(value) is None
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


class _StrictInvalidInputV2(ValueError):
    pass


class _StrictUnsupportedModelV2(ValueError):
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
        checked_config = _strict_legacy_config(config)
    except _StrictInvalidInputV2:
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:STRICT_CONVEX_CANDIDATE_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_CANDIDATE_REVALIDATION",
        )

    budget = _DomainOperationBudgetV2(checked_config.max_domain_operations)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _strict_precharge_problem_structure(problem, budget)
            checked_problem = _strict_legacy_problem(problem)
    except _ResourceLimitErrorV2:
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:max_domain_operations",
        )
    except _StrictInvalidInputV2:
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:STRICT_CONVEX_CANDIDATE_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_CANDIDATE_REVALIDATION",
        )
    atomic_budget = SO2AtomicBudgetV2(limit=checked_config.max_so2_atomic_steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            from spatialcf.core._internal.compilation.candidate_cells import (
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
                return _strict_failure(
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
    except _StrictUnsupportedModelV2 as error:
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except _ResourceLimitErrorV2:
        return _strict_failure(
            StrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:max_domain_operations",
        )
    except (ArithmeticError, RuntimeWarning):
        return _strict_failure(
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
            checked_submitted = _strict_copy_artifact(submitted_artifact)
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
        or _strict_artifact_bytes(checked_submitted) != _strict_artifact_bytes(fresh)
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


def _strict_legacy_problem(value: object) -> SemanticProblemV2_2:
    if type(value) is not SemanticProblemV2_2:
        raise _StrictInvalidInputV2
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
        raise _StrictInvalidInputV2 from error


def _strict_precharge_problem_structure(
    value: object,
    budget: _DomainOperationBudgetV2,
) -> None:
    if type(value) is not SemanticProblemV2_2:
        raise _StrictInvalidInputV2
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
            raise _StrictInvalidInputV2
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
        raise _StrictInvalidInputV2 from error


def _precharge_region(value: Any, budget: _DomainOperationBudgetV2) -> None:
    components = value.components
    if type(components) is not tuple:
        raise _StrictInvalidInputV2
    budget.consume(len(components) + 1)
    for component in components:
        _precharge_component(component, budget)


def _precharge_component(value: Any, budget: _DomainOperationBudgetV2) -> None:
    holes = value.holes
    if type(holes) is not tuple:
        raise _StrictInvalidInputV2
    budget.consume(len(value.exterior.vertices) + len(holes) + 1)
    for hole in holes:
        budget.consume(len(hole.vertices) + 1)


def _strict_legacy_config(value: object) -> StrictConvexCandidateCompilerConfigV2_5:
    if type(value) is not StrictConvexCandidateCompilerConfigV2_5:
        raise _StrictInvalidInputV2
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
        raise _StrictInvalidInputV2 from error


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
        is not PositionRegionInterpretation.SUBJECT_ANCHOR_LOCUS
        or position.workspace_aggregation is not RegionAggregation.INTERSECTION
        or position.boundary_policy is not BoundaryPolicy.CLOSED
        or position.known_free_space_fact_ids
        or len(position.workspace_fact_ids) != 1
        or position.minimum_boundary_clearance_m != 0.0
    ):
        raise _StrictUnsupportedModelV2("UNSUPPORTED:POSITION_DOMAIN_SUBSET")
    workspace_values = _strict_exact_fact_values(
        problem.scene.workspace_boundaries, "WORKSPACE_BOUNDARIES", budget
    )
    if (
        len(workspace_values) != 1
        or workspace_values[0].fact_id != position.workspace_fact_ids[0]
        or workspace_values[0].region_approximation is not GeometryApproximationV2.EXACT
        or workspace_values[0].geometry_uncertainty != UncertaintyBudgetV2()
    ):
        raise _StrictUnsupportedModelV2("UNSUPPORTED:POSITION_WORKSPACE_SUBSET")
    if len(constraints.collision_constraints) != 1:
        raise _StrictUnsupportedModelV2("UNSUPPORTED:COLLISION_CONSTRAINT_CARDINALITY")
    collision = constraints.collision_constraints[0]
    if len(collision.subject_body_ids) != 1 or len(collision.obstacle_body_ids) != 1:
        raise _StrictUnsupportedModelV2("UNSUPPORTED:COLLISION_PAIR_CARDINALITY")
    if (
        collision.clearance_metric
        is not CollisionClearanceMetric.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
        or collision.boundary_policy is not BoundaryPolicy.CLOSED
        or collision.minimum_clearance_m != 0.0
        or collision.support_contact_exceptions
    ):
        raise _StrictUnsupportedModelV2("UNSUPPORTED:COLLISION_POLICY")
    if problem.numeric_policy != NumericPolicyV2():
        raise _StrictUnsupportedModelV2("UNSUPPORTED:NUMERIC_POLICY")

    bodies = _strict_exact_fact_values(
        problem.scene.collision_bodies, "COLLISION_BODIES", budget
    )
    geometries = _strict_exact_fact_values(
        problem.scene.geometry_instances, "GEOMETRY_INSTANCES", budget
    )
    objects = _strict_exact_fact_values(problem.scene.objects, "OBJECTS", budget)
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
        raise _StrictUnsupportedModelV2("UNSUPPORTED:COLLISION_BODY_SUBSET")
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
        raise _StrictUnsupportedModelV2("UNSUPPORTED:FIXED_OBSTACLE_SUBSET")
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
            or not _strict_is_identity_anchor(geometry.anchor_from_geometry)
        ):
            raise _StrictUnsupportedModelV2("UNSUPPORTED:COLLISION_GEOMETRY_SUBSET")
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
        raise _StrictUnsupportedModelV2("UNSUPPORTED:Z_SEPARATED_COLLISION_PAIR")
    return subject_transform, subject_shape, obstacle_transform, obstacle_shape


def _strict_exact_fact_values(
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
        raise _StrictUnsupportedModelV2(f"UNSUPPORTED:{label}_FACT_SET")
    budget.consume(len(facts.values) + 1)
    return facts.values


def _strict_is_identity_anchor(transform: DirectedYawIntervalTransformV2_2) -> bool:
    return (
        type(transform) is DirectedYawIntervalTransformV2_2
        and transform.translation == Vec3(x=0.0, y=0.0, z=0.0)
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


def _strict_failure(
    kind: StrictConvexCandidateCompilationKindV2,
    finding_code: str,
) -> StrictConvexCandidateCompilationOutcomeV2:
    return StrictConvexCandidateCompilationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _strict_copy_universe(value: ExactAxisAlignedRectV2) -> ExactAxisAlignedRectV2:
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


def _strict_copy_bracket(
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


def _strict_copy_artifact(
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


def _strict_artifact_bytes(value: StrictConvexCandidateDomainArtifactV2_2) -> bytes:
    return json.dumps(
        _strict_canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_canonical_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"denominator": value.denominator, "numerator": value.numerator}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _strict_canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_strict_canonical_value(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"unsupported artifact hash value: {type(value).__name__}")


_strict_exports = (
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

# Migrated from multi_obstacle_strict_convex_candidate_domain.py.
"""Raw multi-obstacle strict-convex candidate compilation for Canonical v2.2."""


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

from spatialcf.core._internal.kernels.convex_partition import (
    ConvexAllowedTranslationBracketV2,
    ConvexAllowedTranslationKindV2,
    compile_convex_allowed_translation_v2,
)
from spatialcf.core._internal.kernels.rect import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core._internal.kernels.so2 import SO2AtomicBudgetV2
from spatialcf.core._internal.kernels.strict_convex import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionKindV2,
    intersect_strict_convex_allowed_complexes_v2,
)
from spatialcf.core._internal.compilation.collision import (
    _StrictInvalidInputV2 as _LegacyInvalidInputV2,
)
from spatialcf.core._internal.compilation.collision import (
    _strict_precharge_problem_structure as _precharge_problem_structure,
)
from spatialcf.domain.artifacts import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_6,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
    Vec3,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    CollisionClearanceMetric,
    PositionRegionInterpretation,
    RegionAggregation,
)
from spatialcf.domain.geometry import (
    DirectedYawIntervalTransformV2_2,
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
            from spatialcf.core._internal.compilation.candidate_cells import (
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
        is not PositionRegionInterpretation.SUBJECT_ANCHOR_LOCUS
        or position.workspace_aggregation is not RegionAggregation.INTERSECTION
        or position.boundary_policy is not BoundaryPolicy.CLOSED
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
        is not CollisionClearanceMetric.SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE
        or collision.boundary_policy is not BoundaryPolicy.CLOSED
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
        and transform.translation == Vec3(x=0.0, y=0.0, z=0.0)
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
