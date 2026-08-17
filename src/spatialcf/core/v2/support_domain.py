"""Sound SUPPORT-domain compilation for the exact rectangular Canonical v2 subset.

The compiler works entirely in Canonical world coordinates.  It derives a
domain over the sole allowed edit variable, world-XY translation delta, from
the relative motion of the supported object and the support-surface owner.
Anything outside the deliberately small exact subset is ``UNKNOWN`` rather
than being approximated into a false ``EMPTY`` or certified restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.rect_kernel import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
    TranslationDeltaXYV2,
    UnsupportedRectRegionErrorV2,
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
    SupportConstraintV2,
)
from spatialcf.domain.v2.geometry import (
    ExtrudedPlanarPolygonV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    PlanarRegionV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import (
    CanonicalObjectV2,
    RegionBoundaryPolicyV2,
    SupportSurfaceFactV2,
)


class SupportDomainKindV2(StrEnum):
    """Mathematical effect of one SUPPORT predicate on edit deltas."""

    RECT_DELTA_LOCUS = "RECT_DELTA_LOCUS"
    IDENTITY = "IDENTITY"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SupportDomainCompilationOutcomeV2:
    """Closed, deterministic result of compiling one support constraint.

    ``IDENTITY`` means the exact predicate is constant true over every XY edit
    delta; it does not mean that only the zero delta is allowed.
    """

    kind: SupportDomainKindV2
    delta_locus: ExactAxisAlignedRectV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SupportDomainKindV2):
            raise TypeError("kind must be a SupportDomainKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.kind is SupportDomainKindV2.RECT_DELTA_LOCUS:
            if self.delta_locus is None:
                raise ValueError("RECT_DELTA_LOCUS requires a rectangle")
            if (
                self.delta_locus.coordinate_space
                is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
                or self.delta_locus.topology is not RectTopologyV2.AREA
            ):
                raise ValueError(
                    "support delta locus must be a positive-area translation rectangle"
                )
            if self.finding_codes:
                raise ValueError("a compiled rectangular locus cannot carry findings")
            return
        if self.delta_locus is not None:
            raise ValueError(f"{self.kind.value} must not carry a delta rectangle")
        if self.kind is SupportDomainKindV2.UNKNOWN and not self.finding_codes:
            raise ValueError("UNKNOWN support outcome requires a finding")


SupportDomainOutcomeV2 = SupportDomainCompilationOutcomeV2


@dataclass(frozen=True, slots=True)
class _WorldContactV2:
    rectangle: ExactAxisAlignedRectV2
    plane_z_m: Fraction


def compile_support_domain_v2(
    problem: SemanticProblemV2,
    constraint: SupportConstraintV2 | str,
) -> SupportDomainCompilationOutcomeV2:
    """Compile one exact rectangular SUPPORT predicate into delta coordinates.

    The root and optional constraint instance are reconstructed with strict
    validation before any certified geometric conclusion.
    """

    if not isinstance(problem, SemanticProblemV2):
        raise TypeError("problem must be a SemanticProblemV2")

    checked_problem = SemanticProblemV2.model_validate(
        problem.model_dump(mode="python"),
        strict=True,
    )
    selected = _resolve_constraint(checked_problem, constraint)
    if isinstance(selected, SupportDomainCompilationOutcomeV2):
        return selected

    findings = list(_supported_subset_findings(checked_problem, selected))
    if findings:
        return _unknown(*findings)

    objects = {
        item.object_id: item for item in checked_problem.scene.objects.values or ()
    }
    geometries = {
        item.geometry_id: item
        for item in checked_problem.scene.geometry_instances.values or ()
    }
    surfaces = {
        item.surface_id: item
        for item in checked_problem.scene.support_surfaces.values or ()
    }
    supported_object = objects[selected.supported_object_id]
    geometry = geometries[selected.subject_contact_geometry_ids[0]]
    surface = surfaces[selected.surface_id]

    transform_findings = _transform_findings(
        supported_object,
        geometry,
        surface,
        objects,
    )
    if transform_findings:
        return _unknown(*transform_findings)

    try:
        contact = _contact_in_world(supported_object, geometry)
        support = _surface_in_world(surface, objects)
    except UnsupportedRectRegionErrorV2:
        return _unknown(
            f"UNSUPPORTED_SUPPORT_DOMAIN:NON_RECTANGULAR_GEOMETRY:"
            f"{selected.constraint_id}"
        )

    gap_m = contact.plane_z_m - support.plane_z_m
    gap_min_m = Fraction.from_float(selected.contact_gap_min_m)
    gap_max_m = Fraction.from_float(selected.contact_gap_max_m)
    if gap_m < gap_min_m or gap_m > gap_max_m:
        return _empty(f"EXACT_EMPTY:CONTACT_GAP:{selected.constraint_id}")

    contact_bounds = contact.rectangle.bounds
    assert contact_bounds is not None
    contact_area_m2 = (contact_bounds[2] - contact_bounds[0]) * (
        contact_bounds[3] - contact_bounds[1]
    )
    if contact_area_m2 < Fraction.from_float(selected.minimum_overlap_area_m2):
        return _empty(f"EXACT_EMPTY:OVERLAP_AREA:{selected.constraint_id}")

    inset = support.rectangle.erode_axis(
        AxisMarginXYV2.isotropic_from_binary64(selected.stability_margin_m)
    )
    if inset.topology is RectTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:STABILITY_INSET:{selected.constraint_id}")
    if inset.topology is RectTopologyV2.DEGENERATE:
        return _unknown(
            f"UNSUPPORTED_SUPPORT_DOMAIN:DEGENERATE_SURFACE_INSET:"
            f"{selected.constraint_id}"
        )

    relative_edit_coefficient = _relative_edit_coefficient(
        checked_problem,
        supported_object,
        surface,
    )
    if relative_edit_coefficient == 0:
        if inset.contains(contact.rectangle):
            return SupportDomainCompilationOutcomeV2(kind=SupportDomainKindV2.IDENTITY)
        return _empty(f"EXACT_EMPTY:STABILITY_CONTAINMENT:{selected.constraint_id}")

    relative_locus = _containment_translation_locus(contact.rectangle, inset)
    if relative_locus.topology is RectTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:STABILITY_CONTAINMENT:{selected.constraint_id}")
    if relative_locus.topology is RectTopologyV2.DEGENERATE:
        return _unknown(
            f"UNSUPPORTED_SUPPORT_DOMAIN:DEGENERATE_DELTA_LOCUS:"
            f"{selected.constraint_id}"
        )

    delta_locus = (
        relative_locus
        if relative_edit_coefficient == 1
        else _negate_delta_rectangle(relative_locus)
    )
    return SupportDomainCompilationOutcomeV2(
        kind=SupportDomainKindV2.RECT_DELTA_LOCUS,
        delta_locus=delta_locus,
    )


def _resolve_constraint(
    problem: SemanticProblemV2,
    requested: SupportConstraintV2 | str,
) -> SupportConstraintV2 | SupportDomainCompilationOutcomeV2:
    if isinstance(requested, SupportConstraintV2):
        checked = SupportConstraintV2.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a SupportConstraintV2 or exact str ID")

    registered = next(
        (
            item
            for item in problem.constraints.support_constraints
            if item.constraint_id == constraint_id
        ),
        None,
    )
    if registered is None:
        return _unknown(f"UNKNOWN_SUPPORT_CONSTRAINT:{constraint_id}")
    if checked is not None and checked != registered:
        return _unknown(f"SUPPORT_CONSTRAINT_MISMATCH:{constraint_id}")
    return registered


def _supported_subset_findings(
    problem: SemanticProblemV2,
    constraint: SupportConstraintV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if constraint.boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append(
            f"UNSUPPORTED_SUPPORT_DOMAIN:BOUNDARY_POLICY:{constraint.constraint_id}"
        )
    if not _numeric_policy_is_zero(problem.numeric_policy):
        findings.append("UNSUPPORTED_SUPPORT_DOMAIN:NUMERIC_POLICY")

    for label, facts in (
        ("OBJECTS", problem.scene.objects),
        ("GEOMETRY_INSTANCES", problem.scene.geometry_instances),
        ("SUPPORT_SURFACES", problem.scene.support_surfaces),
    ):
        findings.extend(_fact_family_findings(label, facts))

    if len(constraint.subject_contact_geometry_ids) != 1:
        findings.append(
            f"UNSUPPORTED_SUPPORT_DOMAIN:CONTACT_UNION_CARDINALITY:"
            f"{constraint.constraint_id}:{len(constraint.subject_contact_geometry_ids)}"
        )

    geometry_facts = problem.scene.geometry_instances
    if (
        geometry_facts.availability is FactAvailabilityV2.KNOWN
        and geometry_facts.completeness is FactCompletenessV2.EXACT
    ):
        geometries = {item.geometry_id: item for item in geometry_facts.values or ()}
        for geometry_id in constraint.subject_contact_geometry_ids:
            geometry = geometries.get(geometry_id)
            if geometry is None:
                findings.append(f"MISSING_FACT:SUPPORT_GEOMETRY:{geometry_id}")
                continue
            if geometry.approximation is not GeometryApproximationV2.EXACT:
                findings.append(
                    "UNSUPPORTED_SUPPORT_DOMAIN:CONTACT_APPROXIMATION:"
                    f"{geometry_id}:{geometry.approximation.value}"
                )
            if not _uncertainty_is_zero(geometry.uncertainty):
                findings.append(
                    f"UNSUPPORTED_SUPPORT_DOMAIN:CONTACT_ITEM_UNCERTAINTY:{geometry_id}"
                )

    surface_facts = problem.scene.support_surfaces
    if (
        surface_facts.availability is FactAvailabilityV2.KNOWN
        and surface_facts.completeness is FactCompletenessV2.EXACT
    ):
        surfaces = {item.surface_id: item for item in surface_facts.values or ()}
        surface = surfaces.get(constraint.surface_id)
        if surface is None:
            findings.append(f"MISSING_FACT:SUPPORT_SURFACE:{constraint.surface_id}")
            return tuple(sorted(set(findings)))
        if surface.region_approximation is not GeometryApproximationV2.EXACT:
            findings.append(
                "UNSUPPORTED_SUPPORT_DOMAIN:SURFACE_APPROXIMATION:"
                f"{surface.surface_id}:{surface.region_approximation.value}"
            )
        if surface.boundary_policy is not RegionBoundaryPolicyV2.CLOSED:
            findings.append(
                f"UNSUPPORTED_SUPPORT_DOMAIN:SURFACE_BOUNDARY:{surface.surface_id}"
            )
        if not _uncertainty_is_zero(surface.geometry_uncertainty):
            findings.append(
                f"UNSUPPORTED_SUPPORT_DOMAIN:SURFACE_ITEM_UNCERTAINTY:"
                f"{surface.surface_id}"
            )
        if (
            surface.normal_in_anchor.x,
            surface.normal_in_anchor.y,
            surface.normal_in_anchor.z,
        ) != (
            0.0,
            0.0,
            1.0,
        ):
            findings.append(
                f"UNSUPPORTED_SUPPORT_DOMAIN:NON_HORIZONTAL_SURFACE:"
                f"{surface.surface_id}"
            )
    return tuple(sorted(set(findings)))


def _fact_family_findings(label: str, facts: FactSetV2) -> tuple[str, ...]:
    if facts.availability is FactAvailabilityV2.MISSING:
        return (f"MISSING_FACT:{label}",)
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return (f"UNSUPPORTED_SUPPORT_DOMAIN:{label}_AVAILABILITY",)
    findings: list[str] = []
    if facts.completeness is not FactCompletenessV2.EXACT:
        value = facts.completeness.value if facts.completeness is not None else "NONE"
        findings.append(f"UNSUPPORTED_SUPPORT_DOMAIN:{label}_COMPLETENESS:{value}")
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_SUPPORT_DOMAIN:{label}_FACT_UNCERTAINTY")
    return tuple(findings)


def _transform_findings(
    supported_object: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
    surface: SupportSurfaceFactV2,
    objects: dict[str, CanonicalObjectV2],
) -> tuple[str, ...]:
    findings: list[str] = []
    transforms = (
        (
            f"OBJECT_POSE:{supported_object.object_id}",
            supported_object.pose.world_from_object,
        ),
        (geometry.geometry_id, geometry.anchor_from_geometry),
        (f"SURFACE_FRAME:{surface.surface_id}", surface.anchor_from_surface),
    )
    for label, transform in transforms:
        if not _has_exact_identity_rotation(transform):
            findings.append(f"UNSUPPORTED_SUPPORT_DOMAIN:NON_IDENTITY_ROTATION:{label}")
    if surface.owner_object_id is not None:
        owner = objects[surface.owner_object_id]
        if not _has_exact_identity_rotation(owner.pose.world_from_object):
            findings.append(
                "UNSUPPORTED_SUPPORT_DOMAIN:NON_IDENTITY_ROTATION:"
                f"SURFACE_OWNER_POSE:{owner.object_id}"
            )
    return tuple(sorted(set(findings)))


def _has_exact_identity_rotation(transform: RigidTransformV2) -> bool:
    rotation: QuaternionV2 = transform.rotation
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _contact_in_world(
    supported_object: CanonicalObjectV2,
    geometry: GeometryInstanceV2,
) -> _WorldContactV2:
    if isinstance(geometry.shape, UprightBox3DV2):
        half_x = Fraction.from_float(geometry.shape.size_m.x) / 2
        half_y = Fraction.from_float(geometry.shape.size_m.y) / 2
        local = ExactAxisAlignedRectV2.from_fraction_bounds(
            min_x_m=-half_x,
            min_y_m=-half_y,
            max_x_m=half_x,
            max_y_m=half_y,
            coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
        )
        local_z = -Fraction.from_float(geometry.shape.size_m.z) / 2
    elif isinstance(geometry.shape, ExtrudedPlanarPolygonV2):
        local = ExactAxisAlignedRectV2.from_planar_region(
            # A component is wrapped back into the canonical region expected by
            # the exact parser; no floating-point geometric operation is used.
            PlanarRegionV2(components=(geometry.shape.footprint,))
        )
        local_z = Fraction.from_float(geometry.shape.lower_z_m)
    else:  # pragma: no cover - closed discriminated union, retained defensively
        raise UnsupportedRectRegionErrorV2("unsupported support geometry shape")

    object_transform = supported_object.pose.world_from_object
    geometry_transform = geometry.anchor_from_geometry
    dx = Fraction.from_float(object_transform.translation.x) + Fraction.from_float(
        geometry_transform.translation.x
    )
    dy = Fraction.from_float(object_transform.translation.y) + Fraction.from_float(
        geometry_transform.translation.y
    )
    plane_z = (
        Fraction.from_float(object_transform.translation.z)
        + Fraction.from_float(geometry_transform.translation.z)
        + local_z
    )
    return _WorldContactV2(
        rectangle=local.translate(TranslationDeltaXYV2(dx_m=dx, dy_m=dy)),
        plane_z_m=plane_z,
    )


def _surface_in_world(
    surface: SupportSurfaceFactV2,
    objects: dict[str, CanonicalObjectV2],
) -> _WorldContactV2:
    local = ExactAxisAlignedRectV2.from_planar_region(surface.region_uv)
    owner_translation = (Fraction(), Fraction(), Fraction())
    if surface.owner_object_id is not None:
        translation = objects[
            surface.owner_object_id
        ].pose.world_from_object.translation
        owner_translation = tuple(
            Fraction.from_float(value)
            for value in (translation.x, translation.y, translation.z)
        )
    surface_translation = surface.anchor_from_surface.translation
    dx = owner_translation[0] + Fraction.from_float(surface_translation.x)
    dy = owner_translation[1] + Fraction.from_float(surface_translation.y)
    plane_z = owner_translation[2] + Fraction.from_float(surface_translation.z)
    return _WorldContactV2(
        rectangle=local.translate(TranslationDeltaXYV2(dx_m=dx, dy_m=dy)),
        plane_z_m=plane_z,
    )


def _relative_edit_coefficient(
    problem: SemanticProblemV2,
    supported_object: CanonicalObjectV2,
    surface: SupportSurfaceFactV2,
) -> int:
    subject_id = problem.constraints.allowed_edit.subject_id
    object_coefficient = int(supported_object.object_id == subject_id)
    surface_coefficient = int(surface.owner_object_id == subject_id)
    return object_coefficient - surface_coefficient


def _containment_translation_locus(
    contact: ExactAxisAlignedRectV2,
    surface_inset: ExactAxisAlignedRectV2,
) -> ExactAxisAlignedRectV2:
    contact_bounds = contact.bounds
    surface_bounds = surface_inset.bounds
    assert contact_bounds is not None and surface_bounds is not None
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=surface_bounds[0] - contact_bounds[0],
        min_y_m=surface_bounds[1] - contact_bounds[1],
        max_x_m=surface_bounds[2] - contact_bounds[2],
        max_y_m=surface_bounds[3] - contact_bounds[3],
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _negate_delta_rectangle(
    rectangle: ExactAxisAlignedRectV2,
) -> ExactAxisAlignedRectV2:
    bounds = rectangle.bounds
    assert bounds is not None
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=-bounds[2],
        min_y_m=-bounds[3],
        max_x_m=-bounds[0],
        max_y_m=-bounds[1],
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


def _unknown(*findings: str) -> SupportDomainCompilationOutcomeV2:
    return SupportDomainCompilationOutcomeV2(
        kind=SupportDomainKindV2.UNKNOWN,
        finding_codes=tuple(findings),
    )


def _empty(finding: str) -> SupportDomainCompilationOutcomeV2:
    return SupportDomainCompilationOutcomeV2(
        kind=SupportDomainKindV2.EMPTY,
        finding_codes=(finding,),
    )
