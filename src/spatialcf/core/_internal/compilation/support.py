# ruff: noqa: F811, I001
"""Private current candidate-family compilers; exact migrated bodies."""

from __future__ import annotations

# Migrated from support_domain.py.
"""Sound SUPPORT-domain compilation for the exact rectangular Canonical v2 subset.

The compiler works entirely in Canonical world coordinates.  It derives a
domain over the sole allowed edit variable, world-XY translation delta, from
the relative motion of the supported object and the support-surface owner.
Anything outside the deliberately small exact subset is ``UNKNOWN`` rather
than being approximated into a false ``EMPTY`` or certified restriction.
"""


from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels.rect import (
    AxisMarginXYV2,
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
    TranslationDeltaXYV2,
    UnsupportedRectRegionErrorV2,
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
    SupportConstraint,
)
from spatialcf.domain.geometry import (
    ExtrudedPlanarPolygonV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    PlanarRegionV2,
    UprightBox3DV2,
)
from spatialcf.domain.problem import SemanticProblemV2
from spatialcf.domain.scene import (
    CanonicalObject,
    RegionBoundaryPolicy,
    SupportSurfaceFact,
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
    constraint: SupportConstraint | str,
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
    requested: SupportConstraint | str,
) -> SupportConstraint | SupportDomainCompilationOutcomeV2:
    if isinstance(requested, SupportConstraint):
        checked = SupportConstraint.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a SupportConstraint or exact str ID")

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
    constraint: SupportConstraint,
) -> tuple[str, ...]:
    findings: list[str] = []
    if constraint.boundary_policy is not BoundaryPolicy.CLOSED:
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
        if surface.boundary_policy is not RegionBoundaryPolicy.CLOSED:
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
    supported_object: CanonicalObject,
    geometry: GeometryInstanceV2,
    surface: SupportSurfaceFact,
    objects: dict[str, CanonicalObject],
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
    rotation: Quaternion = transform.rotation
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _contact_in_world(
    supported_object: CanonicalObject,
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
    surface: SupportSurfaceFact,
    objects: dict[str, CanonicalObject],
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
    supported_object: CanonicalObject,
    surface: SupportSurfaceFact,
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


# Migrated from continuous_yaw_support_projection.py.
"""Directed strict-convex SUPPORT projection for one continuously yawed box."""


import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import gcd, lcm

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.convex_partition import (
    RationalHalfPlane2V2,
    RationalHalfPlaneRelationV2,
)
from spatialcf.core._internal.kernels.convex_translation import (
    RationalConvexPolygonV2,
    RationalPoint2V2,
)
from spatialcf.core._internal.kernels.rect import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
    UnsupportedRectRegionErrorV2,
)
from spatialcf.core._internal.kernels.so2 import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
)
from spatialcf.core._internal.kernels.strict_convex import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
    StrictConvexIntersectionComplexV2,
    StrictConvexIntersectionTopologyV2,
)
from spatialcf.core._internal.kernels.upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.domain.artifacts import (
    GeometryInstanceV2_2,
    SemanticProblemV2_2,
    SupportSurfaceFactV2_2,
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
    SupportAssignmentPolicy,
    SupportContactAggregation,
    SupportContactFeature,
    SupportOverlapMetric,
    SupportStabilityMetric,
)
from spatialcf.domain.geometry import (
    DirectedYawIntervalTransformV2_2,
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.scene import RegionBoundaryPolicy

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
            object.__setattr__(self, "bracket", _projection_copy_bracket(self.bracket))
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
        return _projection_failure(
            ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SUPPORT_PROJECTION",
        )
    except (ArithmeticError, RuntimeWarning):
        return _projection_failure(
            ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_PROJECTION_REVALIDATION",
        )
    except _InvalidSupportProjectionInputV2:
        return _projection_failure(
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
                return _projection_failure(
                    ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
                    "RESOURCE_LIMIT:SUPPORT_PROJECTION_SO2",
                )
            if box_outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
                return _projection_failure(
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
        return _projection_failure(
            ContinuousYawSupportProjectionKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (SO2AtomicBudgetExhaustedV2, StrictConvexIntersectionBudgetExhaustedV2):
        return _projection_failure(
            ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:SUPPORT_PROJECTION",
        )
    except ArithmeticError:
        return _projection_failure(
            ContinuousYawSupportProjectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_PROJECTION_ARITHMETIC",
        )
    except RuntimeWarning:
        return _projection_failure(
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
        is not SupportContactFeature.LOWEST_FACE_ALONG_SURFACE_NORMAL
        or constraint.contact_aggregation
        is not SupportContactAggregation.UNION_ALL_SELECTED_FEATURES
        or constraint.overlap_metric
        is not SupportOverlapMetric.PROJECTED_CONTACT_UNION_INTERSECTION_AREA
        or constraint.stability_metric
        is not SupportStabilityMetric.FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET
        or constraint.boundary_policy is not BoundaryPolicy.CLOSED
        or constraint.assignment_policy is not SupportAssignmentPolicy.EXACT_SURFACE
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
            translation=Vec3(x=0.0, y=0.0, z=0.0),
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
        or surface.boundary_policy is not RegionBoundaryPolicy.CLOSED
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


def _projection_copy_bracket(
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


def _projection_failure(
    kind: ContinuousYawSupportProjectionKindV2,
    *finding_codes: str,
) -> ContinuousYawSupportProjectionOutcomeV2:
    return ContinuousYawSupportProjectionOutcomeV2(
        kind=kind,
        finding_codes=tuple(finding_codes),
    )


_projection_exports = (
    "CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2",
    "CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2",
    "ContinuousYawSupportProjectionBracketV2",
    "ContinuousYawSupportProjectionKindV2",
    "ContinuousYawSupportProjectionOutcomeV2",
    "compile_exact_horizontal_support_projection_v2",
)

# Migrated from support_strict_convex_candidate_domain.py.
"""Raw support-aware strict-convex candidate compilation for Canonical v2.2."""


import hashlib
import re
import warnings
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

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
    StrictConvexIntersectionOutcomeV2,
    intersect_strict_convex_allowed_complexes_v2,
)
from spatialcf.core._internal.compilation.support import (
    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2,
    CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2,
    ContinuousYawSupportProjectionBracketV2,
    ContinuousYawSupportProjectionKindV2,
    compile_exact_horizontal_support_projection_v2,
)
from spatialcf.core._internal.compilation.collision import (
    MultiObstacleStrictConvexCandidateCompilationKindV2,
    MultiObstacleStrictConvexCandidateDomainArtifactV2_2,
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _copy_universe,
    _precharge_problem_structure,
    _require_finding_codes,
    _require_id_tuple,
    _strict_problem,
    compile_multi_obstacle_strict_convex_candidate_domain_v2_6,
)
from spatialcf.core._internal.compilation.collision import (
    _StrictInvalidInputV2 as _LegacyInvalidInputV2,
)
from spatialcf.domain.artifacts import (
    SemanticProblemV2_2,
    StrictConvexCandidateCompilerConfigV2_6,
    StrictConvexCandidateCompilerConfigV2_7,
)

_ARTIFACT_HASH_DOMAIN_V2_2 = (
    b"spatialcf.support-strict-convex-candidate-artifact.v2.2\0"
)
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTERSECTION_KERNEL_ID = "geometry-kernel:rational-strict-convex-intersection-v2"
_INTERSECTION_KERNEL_VERSION = "kernel:2.5-strict-convex-intersection"


class SupportStrictConvexCandidateCompilationKindV2(StrEnum):
    ARTIFACT = "ARTIFACT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class SupportStrictConvexCandidateVerificationKindV2(StrEnum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    UNCERTIFIED = "UNCERTIFIED"


@dataclass(frozen=True, slots=True)
class SupportStrictConvexAllowedBracketV2:
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    intersection_kernel_id: str
    intersection_kernel_version: str
    support_projection_kernel_id: str
    support_projection_kernel_version: str
    so2_atomic_steps_used: int

    def __post_init__(self) -> None:
        checked_inner = _copy_intersection_complex(self.inner_allowed)
        checked_outer = _copy_intersection_complex(self.outer_allowed)
        if checked_inner.universe != checked_outer.universe:
            raise ValueError("support-aware bracket requires one exact universe")
        if self.intersection_kernel_id != _INTERSECTION_KERNEL_ID:
            raise ValueError("unexpected intersection kernel ID")
        if self.intersection_kernel_version != _INTERSECTION_KERNEL_VERSION:
            raise ValueError("unexpected intersection kernel version")
        if self.support_projection_kernel_id != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_ID_V2
        ):
            raise ValueError("unexpected support projection kernel ID")
        if self.support_projection_kernel_version != (
            CONTINUOUS_YAW_SUPPORT_PROJECTION_KERNEL_VERSION_V2
        ):
            raise ValueError("unexpected support projection kernel version")
        if (
            type(self.so2_atomic_steps_used) is not int
            or self.so2_atomic_steps_used <= 0
        ):
            raise ValueError("so2_atomic_steps_used must be a positive exact int")
        if not all(
            checked_outer.contains_point(cell.strict_witness)
            for cell in checked_inner.cells
        ):
            raise ValueError("support-aware inner witness escaped the outer domain")
        object.__setattr__(self, "inner_allowed", checked_inner)
        object.__setattr__(self, "outer_allowed", checked_outer)


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateDomainArtifactV2_2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    upstream_t14_artifact_sha256: str
    subject_id: str
    search_universe: ExactAxisAlignedRectV2
    ordered_constraint_ids: tuple[str, ...]
    ordered_obstacle_body_ids: tuple[str, ...]
    support_constraint_id: str
    surface_id: str
    contact_geometry_id: str
    allowed_domain_bracket: SupportStrictConvexAllowedBracketV2
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t14_artifact_sha256", self.upstream_t14_artifact_sha256),
        ):
            if type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label, value in (
            ("subject_id", self.subject_id),
            ("support_constraint_id", self.support_constraint_id),
            ("surface_id", self.surface_id),
            ("contact_geometry_id", self.contact_geometry_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
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
        if self.support_constraint_id not in compiled_ids:
            raise ValueError("support constraint must be included in compiled IDs")
        if set(compiled_ids) & set(remaining_ids):
            raise ValueError("compiled and remaining constraint IDs must be disjoint")
        if type(self.allowed_domain_bracket) is not SupportStrictConvexAllowedBracketV2:
            raise TypeError("allowed_domain_bracket has the wrong exact type")
        checked_bracket = _copy_bracket(self.allowed_domain_bracket)
        if (
            checked_bracket.inner_allowed.universe != checked_universe
            or checked_bracket.outer_allowed.universe != checked_universe
        ):
            raise ValueError("allowed bracket must use the search universe")
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
        if usage.so2_atomic_steps != checked_bracket.so2_atomic_steps_used:
            raise ValueError("SO(2) usage must equal bracket cumulative usage")
        published_cells = len(checked_bracket.inner_allowed.cells) + len(
            checked_bracket.outer_allowed.cells
        )
        if usage.candidate_cells < published_cells:
            raise ValueError("cumulative candidate usage cannot undercount final cells")
        object.__setattr__(self, "search_universe", checked_universe)
        object.__setattr__(self, "ordered_constraint_ids", compiled_ids)
        object.__setattr__(self, "ordered_obstacle_body_ids", obstacle_ids)
        object.__setattr__(self, "remaining_constraint_ids", remaining_ids)
        object.__setattr__(self, "allowed_domain_bracket", checked_bracket)
        object.__setattr__(self, "resource_usage", usage)

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            _ARTIFACT_HASH_DOMAIN_V2_2 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateCompilationOutcomeV2:
    kind: SupportStrictConvexCandidateCompilationKindV2
    artifact: SupportStrictConvexCandidateDomainArtifactV2_2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not SupportStrictConvexCandidateCompilationKindV2:
            raise TypeError("kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is SupportStrictConvexCandidateCompilationKindV2.ARTIFACT:
            if (
                type(self.artifact)
                is not SupportStrictConvexCandidateDomainArtifactV2_2
            ):
                raise ValueError("ARTIFACT outcome requires an exact artifact")
            if findings:
                raise ValueError("ARTIFACT outcome cannot carry findings")
            object.__setattr__(self, "artifact", _copy_artifact(self.artifact))
            return
        if self.artifact is not None or not findings:
            raise ValueError("failure requires findings and no artifact")


@dataclass(frozen=True, slots=True)
class SupportStrictConvexCandidateVerificationOutcomeV2:
    kind: SupportStrictConvexCandidateVerificationKindV2
    semantic_problem_sha256: str | None = None
    compiler_config_sha256: str | None = None
    artifact_sha256: str | None = None
    verification_resource_usage: (
        MultiObstacleStrictConvexCandidateResourceUsageV2 | None
    ) = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not SupportStrictConvexCandidateVerificationKindV2:
            raise TypeError("verification kind has the wrong exact type")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        refs = (
            self.semantic_problem_sha256,
            self.compiler_config_sha256,
            self.artifact_sha256,
        )
        if self.kind is SupportStrictConvexCandidateVerificationKindV2.VERIFIED:
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
            usage = self.verification_resource_usage
            object.__setattr__(
                self,
                "verification_resource_usage",
                MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=usage.domain_operations,
                    so2_atomic_steps=usage.so2_atomic_steps,
                    candidate_cells=usage.candidate_cells,
                ),
            )


class SupportStrictConvexCandidateDomainCompilerV2_7:
    def compile(
        self,
        problem: SemanticProblemV2_2,
        config: StrictConvexCandidateCompilerConfigV2_7,
    ) -> SupportStrictConvexCandidateCompilationOutcomeV2:
        return compile_support_strict_convex_candidate_domain_v2_7(problem, config)


class _InvalidInputV2(ValueError):
    pass


def compile_support_strict_convex_candidate_domain_v2_7(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_7,
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    """Fresh-compile T14 plus one exact horizontal SUPPORT predicate."""

    try:
        checked_config = _strict_config(config)
    except _InvalidInputV2:
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:SUPPORT_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_CONFIG_REVALIDATION",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            upstream = compile_multi_obstacle_strict_convex_candidate_domain_v2_6(
                problem, _t14_config(checked_config)
            )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:UPSTREAM_MULTI_OBSTACLE_REPLAY",
        )
    if (
        upstream.kind
        is not MultiObstacleStrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(upstream.artifact)
        is not MultiObstacleStrictConvexCandidateDomainArtifactV2_2
    ):
        return _from_upstream_failure(upstream.kind, upstream.finding_codes)
    upstream_artifact = upstream.artifact
    budget = StrictConvexIntersectionBudgetV2(
        max_domain_operations=checked_config.max_domain_operations,
        max_candidate_cells=checked_config.max_candidate_cells,
        domain_operations_used=upstream_artifact.resource_usage.domain_operations,
        candidate_cells_used=upstream_artifact.resource_usage.candidate_cells,
    )
    atomic_budget = SO2AtomicBudgetV2(
        limit=checked_config.max_so2_atomic_steps,
        used=upstream_artifact.resource_usage.so2_atomic_steps,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            _precharge_problem_structure(problem, budget)  # type: ignore[arg-type]
            checked_problem = _strict_problem(problem)
            if (
                checked_problem.semantic_problem_sha256
                != upstream_artifact.semantic_problem_sha256
            ):
                raise _InvalidInputV2
            support = compile_exact_horizontal_support_projection_v2(
                checked_problem,
                "constraint:support",
                upstream_artifact.search_universe,
                atomic_budget=atomic_budget,
                intersection_budget=budget,
            )
            if support.kind is ContinuousYawSupportProjectionKindV2.RESOURCE_LIMIT:
                return _resource_failure()
            if support.kind is ContinuousYawSupportProjectionKindV2.NUMERIC_GAP:
                return SupportStrictConvexCandidateCompilationOutcomeV2(
                    kind=SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
                    finding_codes=support.finding_codes,
                )
            if support.kind is ContinuousYawSupportProjectionKindV2.INVALID_INPUT:
                raise RuntimeError("strictly checked support input became invalid")
            if support.kind is ContinuousYawSupportProjectionKindV2.UNSUPPORTED_MODEL:
                return SupportStrictConvexCandidateCompilationOutcomeV2(
                    kind=SupportStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL,
                    finding_codes=support.finding_codes,
                )
            if (
                support.kind is not ContinuousYawSupportProjectionKindV2.BRACKET
                or type(support.bracket) is not ContinuousYawSupportProjectionBracketV2
            ):
                raise RuntimeError("malformed support projection outcome")
            support_bracket = support.bracket
            inner = _require_intersection(
                intersect_strict_convex_allowed_complexes_v2(
                    (
                        upstream_artifact.allowed_domain_bracket.inner_allowed,
                        support_bracket.inner_allowed,
                    ),
                    budget=budget,
                )
            )
            outer = _require_intersection(
                intersect_strict_convex_allowed_complexes_v2(
                    (
                        upstream_artifact.allowed_domain_bracket.outer_allowed,
                        support_bracket.outer_allowed,
                    ),
                    budget=budget,
                )
            )
            remaining = tuple(
                item
                for item in upstream_artifact.remaining_constraint_ids
                if item != support_bracket.support_constraint_id
            )
            if len(remaining) + 1 != len(upstream_artifact.remaining_constraint_ids):
                raise RuntimeError("T14 remaining IDs lost the support constraint")
            budget.consume_domain(
                24
                + len(remaining)
                + len(upstream_artifact.ordered_obstacle_body_ids)
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (inner, outer)
                    for cell in complex_.cells
                )
            )
            bracket = SupportStrictConvexAllowedBracketV2(
                inner_allowed=inner,
                outer_allowed=outer,
                intersection_kernel_id=checked_config.intersection_kernel_id,
                intersection_kernel_version=checked_config.intersection_kernel_version,
                support_projection_kernel_id=(
                    checked_config.support_projection_kernel_id
                ),
                support_projection_kernel_version=(
                    checked_config.support_projection_kernel_version
                ),
                so2_atomic_steps_used=atomic_budget.used,
            )
            artifact = SupportStrictConvexCandidateDomainArtifactV2_2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_config.config_sha256,
                upstream_t14_artifact_sha256=upstream_artifact.artifact_sha256,
                subject_id=upstream_artifact.subject_id,
                search_universe=upstream_artifact.search_universe,
                ordered_constraint_ids=(
                    *upstream_artifact.ordered_constraint_ids,
                    support_bracket.support_constraint_id,
                ),
                ordered_obstacle_body_ids=(upstream_artifact.ordered_obstacle_body_ids),
                support_constraint_id=support_bracket.support_constraint_id,
                surface_id=support_bracket.surface_id,
                contact_geometry_id=support_bracket.contact_geometry_id,
                allowed_domain_bracket=bracket,
                resource_usage=MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=budget.domain_operations_used,
                    so2_atomic_steps=atomic_budget.used,
                    candidate_cells=budget.candidate_cells_used,
                ),
                remaining_constraint_ids=remaining,
            )
            return SupportStrictConvexCandidateCompilationOutcomeV2(
                kind=SupportStrictConvexCandidateCompilationKindV2.ARTIFACT,
                artifact=artifact,
            )
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _resource_failure()
    except (_InvalidInputV2, _LegacyInvalidInputV2):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT,
            "INVALID_INPUT:SUPPORT_STRICT_CONVEX_INPUT",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:SUPPORT_STRICT_CONVEX_COMPILATION",
        )


def verify_support_strict_convex_candidate_domain_v2_7(
    problem: SemanticProblemV2_2,
    config: StrictConvexCandidateCompilerConfigV2_7,
    submitted_artifact: SupportStrictConvexCandidateDomainArtifactV2_2,
) -> SupportStrictConvexCandidateVerificationOutcomeV2:
    """Fresh replay raw inputs and compare the entire submitted T15 artifact."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_submitted = _copy_artifact(submitted_artifact)
    except (ArithmeticError, RuntimeWarning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUBMITTED_SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )
    except (AttributeError, TypeError, ValueError, Warning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("INVALID_INPUT:SUBMITTED_SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            replay = compile_support_strict_convex_candidate_domain_v2_7(
                problem, config
            )
    except (ArithmeticError, RuntimeWarning):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=("NUMERIC_GAP:SUPPORT_STRICT_CONVEX_REPLAY",),
        )
    if (
        replay.kind is not SupportStrictConvexCandidateCompilationKindV2.ARTIFACT
        or type(replay.artifact) is not SupportStrictConvexCandidateDomainArtifactV2_2
    ):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.UNCERTIFIED,
            finding_codes=replay.finding_codes,
        )
    fresh = replay.artifact
    usage = fresh.resource_usage
    if (
        checked_submitted != fresh
        or _artifact_bytes(checked_submitted) != _artifact_bytes(fresh)  # type: ignore[arg-type]
        or checked_submitted.artifact_sha256 != fresh.artifact_sha256
    ):
        return SupportStrictConvexCandidateVerificationOutcomeV2(
            kind=SupportStrictConvexCandidateVerificationKindV2.MISMATCH,
            verification_resource_usage=usage,
            finding_codes=("MISMATCH:SUPPORT_STRICT_CONVEX_ARTIFACT",),
        )
    return SupportStrictConvexCandidateVerificationOutcomeV2(
        kind=SupportStrictConvexCandidateVerificationKindV2.VERIFIED,
        semantic_problem_sha256=fresh.semantic_problem_sha256,
        compiler_config_sha256=fresh.compiler_config_sha256,
        artifact_sha256=fresh.artifact_sha256,
        verification_resource_usage=usage,
    )


def _t14_config(
    config: StrictConvexCandidateCompilerConfigV2_7,
) -> StrictConvexCandidateCompilerConfigV2_6:
    return StrictConvexCandidateCompilerConfigV2_6(
        max_domain_operations=config.max_domain_operations,
        max_so2_atomic_steps=config.max_so2_atomic_steps,
        max_candidate_cells=config.max_candidate_cells,
    )


def _require_intersection(
    outcome: StrictConvexIntersectionOutcomeV2,
) -> StrictConvexIntersectionComplexV2:
    if outcome.kind is StrictConvexIntersectionKindV2.RESOURCE_LIMIT:
        raise StrictConvexIntersectionBudgetExhaustedV2
    if outcome.kind is StrictConvexIntersectionKindV2.NUMERIC_GAP:
        raise ArithmeticError("strict-convex support intersection numeric gap")
    if outcome.kind is StrictConvexIntersectionKindV2.INVALID_INPUT:
        raise RuntimeError("compiler produced invalid support intersection operands")
    if (
        outcome.kind is not StrictConvexIntersectionKindV2.COMPLEX
        or type(outcome.complex) is not StrictConvexIntersectionComplexV2
    ):
        raise RuntimeError("malformed support intersection outcome")
    return outcome.complex


def _strict_config(value: object) -> StrictConvexCandidateCompilerConfigV2_7:
    if type(value) is not StrictConvexCandidateCompilerConfigV2_7:
        raise _InvalidInputV2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return StrictConvexCandidateCompilerConfigV2_7.model_validate(
                value.model_dump(mode="python", warnings="error"), strict=True
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


def _from_upstream_failure(
    kind: MultiObstacleStrictConvexCandidateCompilationKindV2,
    finding_codes: tuple[str, ...],
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    mapped = {
        MultiObstacleStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL: (
            SupportStrictConvexCandidateCompilationKindV2.UNSUPPORTED_MODEL
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.NUMERIC_GAP: (
            SupportStrictConvexCandidateCompilationKindV2.NUMERIC_GAP
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT: (
            SupportStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT
        ),
        MultiObstacleStrictConvexCandidateCompilationKindV2.INVALID_INPUT: (
            SupportStrictConvexCandidateCompilationKindV2.INVALID_INPUT
        ),
    }.get(kind)
    if mapped is None:
        raise RuntimeError("malformed T14 compiler outcome")
    return SupportStrictConvexCandidateCompilationOutcomeV2(
        kind=mapped,
        finding_codes=finding_codes,
    )


def _copy_bracket(
    value: SupportStrictConvexAllowedBracketV2,
) -> SupportStrictConvexAllowedBracketV2:
    return SupportStrictConvexAllowedBracketV2(
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        intersection_kernel_id=value.intersection_kernel_id,
        intersection_kernel_version=value.intersection_kernel_version,
        support_projection_kernel_id=value.support_projection_kernel_id,
        support_projection_kernel_version=value.support_projection_kernel_version,
        so2_atomic_steps_used=value.so2_atomic_steps_used,
    )


def _copy_artifact(
    value: SupportStrictConvexCandidateDomainArtifactV2_2,
) -> SupportStrictConvexCandidateDomainArtifactV2_2:
    return SupportStrictConvexCandidateDomainArtifactV2_2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        upstream_t14_artifact_sha256=value.upstream_t14_artifact_sha256,
        subject_id=value.subject_id,
        search_universe=value.search_universe,
        ordered_constraint_ids=value.ordered_constraint_ids,
        ordered_obstacle_body_ids=value.ordered_obstacle_body_ids,
        support_constraint_id=value.support_constraint_id,
        surface_id=value.surface_id,
        contact_geometry_id=value.contact_geometry_id,
        allowed_domain_bracket=value.allowed_domain_bracket,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
    )


def _failure(
    kind: SupportStrictConvexCandidateCompilationKindV2,
    finding_code: str,
) -> SupportStrictConvexCandidateCompilationOutcomeV2:
    return SupportStrictConvexCandidateCompilationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _resource_failure() -> SupportStrictConvexCandidateCompilationOutcomeV2:
    return _failure(
        SupportStrictConvexCandidateCompilationKindV2.RESOURCE_LIMIT,
        "RESOURCE_LIMIT:SUPPORT_STRICT_CONVEX_CANDIDATE",
    )


__all__ = (
    "SupportStrictConvexAllowedBracketV2",
    "SupportStrictConvexCandidateCompilationKindV2",
    "SupportStrictConvexCandidateCompilationOutcomeV2",
    "SupportStrictConvexCandidateDomainArtifactV2_2",
    "SupportStrictConvexCandidateDomainCompilerV2_7",
    "SupportStrictConvexCandidateVerificationKindV2",
    "SupportStrictConvexCandidateVerificationOutcomeV2",
    "compile_support_strict_convex_candidate_domain_v2_7",
    "verify_support_strict_convex_candidate_domain_v2_7",
)
