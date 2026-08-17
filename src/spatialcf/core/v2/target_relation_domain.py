"""Sound target-relation domain compilation for exact rectangular NEAR/FAR.

The exact Euclidean rounded-rectangle locus is bracketed by rational
axis-aligned regions.  No camera callback or source-specific predicate is
consulted: unsupported semantics remain explicit ``UNKNOWN`` outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.rect_kernel import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    RectilinearTopologyV2,
    difference_rectilinear_region_v2,
    intersect_rectilinear_regions_v2,
    normalize_rectilinear_region_v2,
    union_rectilinear_regions_v2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    QuaternionV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    RelationMeasurementV2,
    RelationV2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import CanonicalObjectV2


class TargetRelationDomainKindV2(StrEnum):
    BRACKET = "BRACKET"
    IDENTITY = "IDENTITY"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class TargetRelationDomainOutcomeV2:
    """Inner/outer allowed-delta bracket or one closed terminal disposition."""

    kind: TargetRelationDomainKindV2
    inner_allowed_delta: ExactRectilinearRegionV2 | None = None
    outer_allowed_delta: ExactRectilinearRegionV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TargetRelationDomainKindV2):
            raise TypeError("kind must be a TargetRelationDomainKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.kind is TargetRelationDomainKindV2.BRACKET:
            if self.inner_allowed_delta is None or self.outer_allowed_delta is None:
                raise ValueError(
                    "BRACKET requires both inner and outer allowed domains"
                )
            if self.finding_codes:
                raise ValueError("BRACKET cannot carry findings")
            return
        if self.inner_allowed_delta is not None or self.outer_allowed_delta is not None:
            raise ValueError(f"{self.kind.value} must not carry partial domains")
        if (
            self.kind
            in {
                TargetRelationDomainKindV2.UNKNOWN,
                TargetRelationDomainKindV2.RESOURCE_LIMIT,
            }
            and not self.finding_codes
        ):
            raise ValueError(f"{self.kind.value} requires a finding")


TargetRelationDomainCompilationOutcomeV2 = TargetRelationDomainOutcomeV2


def compile_target_relation_domain_v2(
    problem: SemanticProblemV2,
    universe: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> TargetRelationDomainOutcomeV2:
    """Compile the target's after-relation over one finite delta universe."""

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if not isinstance(problem, SemanticProblemV2):
        raise TypeError("problem must be a SemanticProblemV2")
    if not isinstance(universe, ExactRectilinearRegionV2):
        raise TypeError("universe must be an ExactRectilinearRegionV2")

    checked_problem = SemanticProblemV2.model_validate(
        problem.model_dump(mode="python"),
        strict=True,
    )

    validation = union_rectilinear_regions_v2(
        universe,
        universe,
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(validation)
    if failure is not None:
        return failure
    checked_universe = _exact_region(validation)

    findings = _supported_subset_findings(checked_problem)
    if findings:
        return _unknown(*findings)

    target = checked_problem.constraints.target_relation
    definition = next(
        item
        for item in checked_problem.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    threshold = Fraction.from_float(definition.threshold)

    if target.relation_after is RelationV2.NEAR and threshold < 0:
        return _empty(f"EXACT_EMPTY:NEGATIVE_NEAR_THRESHOLD:{target.constraint_id}")
    if target.relation_after is RelationV2.FAR and threshold <= 0:
        return TargetRelationDomainOutcomeV2(kind=TargetRelationDomainKindV2.IDENTITY)
    if checked_universe.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:TARGET_RELATION_DOMAIN:{target.constraint_id}")

    q0_rectangle = _overlap_delta_rectangle(checked_problem)
    q0_outcome = normalize_rectilinear_region_v2(
        (q0_rectangle,),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(q0_outcome)
    if failure is not None:
        return failure
    q0 = _exact_region(q0_outcome)

    if target.relation_after is RelationV2.NEAR:
        return _compile_near(
            target.constraint_id,
            checked_universe,
            q0_rectangle,
            q0,
            threshold,
            budget,
        )
    return _compile_far(
        target.constraint_id,
        checked_universe,
        q0_rectangle,
        q0,
        threshold,
        budget,
    )


def _compile_near(
    constraint_id: str,
    universe: ExactRectilinearRegionV2,
    q0_rectangle: ExactAxisAlignedRectV2,
    q0: ExactRectilinearRegionV2,
    threshold: Fraction,
    budget: RectilinearAtomicBudgetV2,
) -> TargetRelationDomainOutcomeV2:
    if threshold == 0:
        clipped = intersect_rectilinear_regions_v2(
            universe,
            q0,
            atomic_budget=budget,
        )
        failure = _maybe_nested_failure(clipped)
        if failure is not None:
            return failure
        exact = _exact_region(clipped)
        if exact.topology is RectilinearTopologyV2.EMPTY:
            return _empty(f"EXACT_EMPTY:TARGET_RELATION_DOMAIN:{constraint_id}")
        return _bracket(exact, exact)

    inner_rectangle = q0_rectangle.dilate_axis(
        AxisMarginXYV2(x_m=threshold / 2, y_m=threshold / 2)
    )
    outer_rectangle = q0_rectangle.dilate_axis(
        AxisMarginXYV2(x_m=threshold, y_m=threshold)
    )
    inner_shape_outcome = normalize_rectilinear_region_v2(
        (inner_rectangle,),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(inner_shape_outcome)
    if failure is not None:
        return failure
    outer_shape_outcome = normalize_rectilinear_region_v2(
        (outer_rectangle,),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(outer_shape_outcome)
    if failure is not None:
        return failure

    inner_outcome = intersect_rectilinear_regions_v2(
        universe,
        _exact_region(inner_shape_outcome),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(inner_outcome)
    if failure is not None:
        return failure
    outer_outcome = intersect_rectilinear_regions_v2(
        universe,
        _exact_region(outer_shape_outcome),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(outer_outcome)
    if failure is not None:
        return failure
    inner = _exact_region(inner_outcome)
    outer = _exact_region(outer_outcome)
    if outer.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:TARGET_RELATION_DOMAIN:{constraint_id}")
    return _bracket(inner, outer)


def _compile_far(
    constraint_id: str,
    universe: ExactRectilinearRegionV2,
    q0_rectangle: ExactAxisAlignedRectV2,
    q0: ExactRectilinearRegionV2,
    threshold: Fraction,
    budget: RectilinearAtomicBudgetV2,
) -> TargetRelationDomainOutcomeV2:
    dilated_rectangle = q0_rectangle.dilate_axis(
        AxisMarginXYV2(x_m=threshold, y_m=threshold)
    )
    dilated_outcome = normalize_rectilinear_region_v2(
        (dilated_rectangle,),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(dilated_outcome)
    if failure is not None:
        return failure
    inner_outcome = difference_rectilinear_region_v2(
        universe,
        _exact_region(dilated_outcome),
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(inner_outcome)
    if failure is not None:
        return failure
    outer_outcome = difference_rectilinear_region_v2(
        universe,
        q0,
        atomic_budget=budget,
    )
    failure = _maybe_nested_failure(outer_outcome)
    if failure is not None:
        return failure
    inner = _exact_region(inner_outcome)
    outer = _exact_region(outer_outcome)
    if outer.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:TARGET_RELATION_DOMAIN:{constraint_id}")
    return _bracket(inner, outer)


def _supported_subset_findings(problem: SemanticProblemV2) -> tuple[str, ...]:
    target = problem.constraints.target_relation
    definition = next(
        item
        for item in problem.relation_semantics.definitions
        if item.relation is target.relation_after
    )
    findings: list[str] = []
    if target.relation_after not in {RelationV2.NEAR, RelationV2.FAR}:
        findings.append(
            f"UNSUPPORTED_TARGET_RELATION:AFTER_RELATION:{target.relation_after.value}"
        )
    if definition.measurement is not RelationMeasurementV2.SHAPE_GAP_XY:
        findings.append(
            f"UNSUPPORTED_TARGET_RELATION:MEASUREMENT:{definition.measurement.value}"
        )
    if definition.tolerance != 0.0:
        findings.append("UNSUPPORTED_TARGET_RELATION:TOLERANCE")
    if definition.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append(
            "UNSUPPORTED_TARGET_RELATION:BOUNDARY_POLICY:"
            f"{definition.boundary_policy.value}"
        )
    if not _numeric_policy_is_zero(problem.numeric_policy):
        findings.append("UNSUPPORTED_TARGET_RELATION:NUMERIC_POLICY")

    object_facts = problem.scene.objects
    geometry_facts = problem.scene.geometry_instances
    findings.extend(_family_findings("OBJECT", object_facts))
    findings.extend(_family_findings("GEOMETRY", geometry_facts))

    objects = {item.object_id: item for item in object_facts.values or ()}
    for object_id in (target.subject_id, target.reference_id):
        object_ = objects.get(object_id)
        if object_ is not None and not _is_identity_rotation(
            object_.pose.world_from_object.rotation
        ):
            findings.append(
                f"UNSUPPORTED_TARGET_RELATION:NON_IDENTITY_ROTATION:"
                f"OBJECT_POSE:{object_id}"
            )

    if (
        geometry_facts.availability is FactAvailabilityV2.KNOWN
        and geometry_facts.completeness is FactCompletenessV2.EXACT
    ):
        geometries = geometry_facts.values or ()
        for object_id in (target.subject_id, target.reference_id):
            selected = tuple(
                item
                for item in geometries
                if item.owner_object_id == object_id
                and item.role is GeometryRoleV2.RELATION
            )
            if len(selected) != 1:
                findings.append(
                    f"UNSUPPORTED_TARGET_RELATION:GEOMETRY_CARDINALITY:"
                    f"{object_id}:{len(selected)}"
                )
                continue
            geometry = selected[0]
            if geometry.approximation is not GeometryApproximationV2.EXACT:
                findings.append(
                    f"UNSUPPORTED_TARGET_RELATION:GEOMETRY_APPROXIMATION:"
                    f"{geometry.geometry_id}:{geometry.approximation.value}"
                )
            if not _uncertainty_is_zero(geometry.uncertainty):
                findings.append(
                    f"UNSUPPORTED_TARGET_RELATION:GEOMETRY_ITEM_UNCERTAINTY:"
                    f"{geometry.geometry_id}"
                )
            if not isinstance(geometry.shape, UprightBox3DV2):
                findings.append(
                    f"UNSUPPORTED_TARGET_RELATION:GEOMETRY_SHAPE:"
                    f"{geometry.geometry_id}:{geometry.shape.shape_type}"
                )
            if not _is_identity_rotation(geometry.anchor_from_geometry.rotation):
                findings.append(
                    f"UNSUPPORTED_TARGET_RELATION:NON_IDENTITY_ROTATION:"
                    f"{geometry.geometry_id}"
                )
    return tuple(sorted(set(findings)))


def _family_findings(label: str, facts: FactSetV2) -> tuple[str, ...]:
    if facts.availability is FactAvailabilityV2.MISSING:
        return (f"MISSING_FACT:{label}",)
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return (
            f"UNSUPPORTED_TARGET_RELATION:{label}_AVAILABILITY:{facts.availability.value}",
        )
    findings: list[str] = []
    if facts.completeness is not FactCompletenessV2.EXACT:
        value = facts.completeness.value if facts.completeness is not None else "NONE"
        findings.append(f"UNSUPPORTED_TARGET_RELATION:{label}_COMPLETENESS:{value}")
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_TARGET_RELATION:{label}_FACT_UNCERTAINTY")
    return tuple(findings)


def _overlap_delta_rectangle(
    problem: SemanticProblemV2,
) -> ExactAxisAlignedRectV2:
    target = problem.constraints.target_relation
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    geometries = problem.scene.geometry_instances.values or ()
    subject_geometry = next(
        item
        for item in geometries
        if item.owner_object_id == target.subject_id
        and item.role is GeometryRoleV2.RELATION
    )
    reference_geometry = next(
        item
        for item in geometries
        if item.owner_object_id == target.reference_id
        and item.role is GeometryRoleV2.RELATION
    )
    subject_bounds = _world_box_bounds(objects[target.subject_id], subject_geometry)
    reference_bounds = _world_box_bounds(
        objects[target.reference_id], reference_geometry
    )
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=reference_bounds[0] - subject_bounds[2],
        min_y_m=reference_bounds[1] - subject_bounds[3],
        max_x_m=reference_bounds[2] - subject_bounds[0],
        max_y_m=reference_bounds[3] - subject_bounds[1],
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _world_box_bounds(
    object_: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    if not isinstance(geometry.shape, UprightBox3DV2):
        raise TypeError("certified relation geometry must be an UprightBox3DV2")
    center_x = Fraction.from_float(
        object_.pose.world_from_object.translation.x
    ) + Fraction.from_float(geometry.anchor_from_geometry.translation.x)
    center_y = Fraction.from_float(
        object_.pose.world_from_object.translation.y
    ) + Fraction.from_float(geometry.anchor_from_geometry.translation.y)
    half_x = Fraction.from_float(geometry.shape.size_m.x) / 2
    half_y = Fraction.from_float(geometry.shape.size_m.y) / 2
    return (
        center_x - half_x,
        center_y - half_y,
        center_x + half_x,
        center_y + half_y,
    )


def _is_identity_rotation(rotation: QuaternionV2) -> bool:
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


def _exact_region(outcome: RectilinearRegionOutcomeV2) -> ExactRectilinearRegionV2:
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise RuntimeError("nested rectilinear outcome is not exact")
    return outcome.region


def _maybe_nested_failure(
    outcome: RectilinearRegionOutcomeV2,
) -> TargetRelationDomainOutcomeV2 | None:
    if outcome.kind is RectilinearOutcomeKindV2.EXACT:
        return None
    return _nested_failure(outcome)


def _nested_failure(
    outcome: RectilinearRegionOutcomeV2,
) -> TargetRelationDomainOutcomeV2:
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource()
    return _unknown(*outcome.finding_codes)


def _bracket(
    inner: ExactRectilinearRegionV2,
    outer: ExactRectilinearRegionV2,
) -> TargetRelationDomainOutcomeV2:
    return TargetRelationDomainOutcomeV2(
        kind=TargetRelationDomainKindV2.BRACKET,
        inner_allowed_delta=inner,
        outer_allowed_delta=outer,
    )


def _empty(finding: str) -> TargetRelationDomainOutcomeV2:
    return TargetRelationDomainOutcomeV2(
        kind=TargetRelationDomainKindV2.EMPTY,
        finding_codes=(finding,),
    )


def _unknown(*findings: str) -> TargetRelationDomainOutcomeV2:
    return TargetRelationDomainOutcomeV2(
        kind=TargetRelationDomainKindV2.UNKNOWN,
        finding_codes=tuple(findings),
    )


def _resource() -> TargetRelationDomainOutcomeV2:
    return TargetRelationDomainOutcomeV2(
        kind=TargetRelationDomainKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )
