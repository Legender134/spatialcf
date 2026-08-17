"""Immutable public contracts for certified continuous solver results."""

import math
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus
from spatialcf.domain.models import InterventionSpec, Vec3
from spatialcf.solver.objective import ObjectiveBreakdown

SUPPORTED_DIRECTIONS = frozenset({
    (Relation.LEFT, Relation.RIGHT),
    (Relation.FRONT, Relation.BEHIND),
    (Relation.NEAR, Relation.FAR),
})

_MAX_CERTIFIED_OPTIMALITY_TOLERANCE = 1e-6


class CertifiedGeometryError(ArithmeticError, ValueError):
    """A numerical geometry result cannot support a certified proof."""


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_power_of_two_at_least_four(value: object) -> bool:
    return type(value) is int and value >= 4 and value & (value - 1) == 0


@dataclass(frozen=True)
class ConstraintRegionSummary:
    """Serialization-safe summary of one inner or outer feasible geometry."""

    area_m2: float
    component_count: int
    dimension: int
    is_empty: bool

    def __post_init__(self) -> None:
        if not _is_finite_number(self.area_m2) or self.area_m2 < 0.0:
            raise ValueError("constraint region area must be finite and non-negative")
        if type(self.component_count) is not int or self.component_count < 0:
            raise ValueError("constraint region component count must be non-negative")
        if type(self.dimension) is not int or self.dimension not in {-1, 0, 1, 2}:
            raise ValueError("constraint region dimension must be -1, 0, 1, or 2")
        if type(self.is_empty) is not bool:
            raise ValueError("constraint region empty flag must be a boolean")
        if self.is_empty != (self.component_count == 0):
            raise ValueError("empty constraint regions must have zero components")
        if self.is_empty != (self.dimension == -1):
            raise ValueError("empty constraint regions must have dimension -1")


@dataclass(frozen=True)
class ConstraintBracketSummary:
    """Serialization-safe summaries for both conservative bracket sides."""

    inner: ConstraintRegionSummary
    outer: ConstraintRegionSummary

    def __post_init__(self) -> None:
        if (
            type(self.inner) is not ConstraintRegionSummary
            or type(self.outer) is not ConstraintRegionSummary
        ):
            raise ValueError("constraint bracket summaries must be exact summaries")


@dataclass(frozen=True)
class ConstraintDiagnosticStep:
    """One deterministic constraint application in canonical builder order."""

    index: int
    stage: str
    constraint_id: str
    applied: bool
    emptied_outer: bool
    inner: ConstraintRegionSummary
    outer: ConstraintRegionSummary

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("constraint diagnostic index must be non-negative")
        if type(self.stage) is not str or not self.stage:
            raise ValueError("constraint diagnostic stage must be non-empty")
        if type(self.constraint_id) is not str or not self.constraint_id:
            raise ValueError("constraint diagnostic ID must be non-empty")
        if type(self.applied) is not bool or type(self.emptied_outer) is not bool:
            raise ValueError("constraint diagnostic flags must be booleans")
        if self.emptied_outer and (not self.applied or not self.outer.is_empty):
            raise ValueError("only an applied empty constraint can empty the outer region")


@dataclass(frozen=True)
class ConstraintBuildDiagnostics:
    """Ordered evidence from the exact certified constraint construction path.

    ``first_outer_empty_step`` is deliberately order-dependent.  It identifies
    the first empty prefix in the builder's canonical order; it is not a
    minimal unsatisfiable core.
    """

    steps: tuple[ConstraintDiagnosticStep, ...]
    first_outer_empty_step_index: int | None
    structural: ConstraintBracketSummary
    target_with_structural: ConstraintBracketSummary
    final: ConstraintBracketSummary
    outer_empty_group: str | None

    def __post_init__(self) -> None:
        if (
            type(self.steps) is not tuple
            or not self.steps
            or any(type(step) is not ConstraintDiagnosticStep for step in self.steps)
        ):
            raise ValueError("constraint diagnostics must contain at least one step")
        if tuple(step.index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("constraint diagnostic indices must be contiguous")
        emptied = tuple(step.index for step in self.steps if step.emptied_outer)
        expected = emptied[0] if emptied else None
        if self.first_outer_empty_step_index is not None and (
            type(self.first_outer_empty_step_index) is not int
            or not 0 <= self.first_outer_empty_step_index < len(self.steps)
        ):
            raise ValueError("first outer-empty diagnostic index is invalid")
        if self.first_outer_empty_step_index != expected:
            raise ValueError("first outer-empty diagnostic index is inconsistent")
        if any(
            type(summary) is not ConstraintBracketSummary
            for summary in (
                self.structural,
                self.target_with_structural,
                self.final,
            )
        ):
            raise ValueError("constraint group summaries must be exact summaries")
        if self.final != ConstraintBracketSummary(
            inner=self.steps[-1].inner,
            outer=self.steps[-1].outer,
        ):
            raise ValueError("final constraint group summary is inconsistent")
        if self.structural.outer.is_empty:
            expected_group = "structural"
        elif self.target_with_structural.outer.is_empty:
            expected_group = "target_relation"
        elif self.final.outer.is_empty:
            expected_group = "preserved_relation"
        else:
            expected_group = None
        if self.outer_empty_group != expected_group:
            raise ValueError("outer-empty constraint group is inconsistent")

    @property
    def first_outer_empty_step(self) -> ConstraintDiagnosticStep | None:
        if self.first_outer_empty_step_index is None:
            return None
        return self.steps[self.first_outer_empty_step_index]


def expected_target_diff(spec: InterventionSpec) -> tuple[str, ...]:
    """Return the complete canonical target-pair relation change tuple."""
    return tuple(sorted((
        f"-{spec.subject_id}:{spec.relation_before.value}:{spec.reference_id}",
        f"+{spec.subject_id}:{spec.relation_after.value}:{spec.reference_id}",
        (
            f"-{spec.reference_id}:{spec.relation_before.converse.value}:"
            f"{spec.subject_id}"
        ),
        (
            f"+{spec.reference_id}:{spec.relation_after.converse.value}:"
            f"{spec.subject_id}"
        ),
    )))


@dataclass(frozen=True)
class CertifiedSolverConfig:
    optimality_tolerance: float = 1e-6
    numeric_tolerance: float = 1e-9
    target_interior_margin: float = 5e-7
    initial_disk_segments: int = 128
    max_disk_segments: int = 8192
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _is_finite_number(self.optimality_tolerance) or self.optimality_tolerance <= 0:
            raise ValueError("optimality_tolerance must be finite and positive")
        if self.optimality_tolerance > _MAX_CERTIFIED_OPTIMALITY_TOLERANCE:
            raise ValueError("optimality_tolerance must be at most 1e-6 metres")
        if not _is_finite_number(self.numeric_tolerance) or self.numeric_tolerance <= 0:
            raise ValueError("numeric_tolerance must be finite and positive")
        if self.numeric_tolerance >= self.optimality_tolerance:
            raise ValueError("numeric_tolerance must be below optimality_tolerance")
        if (
            not _is_finite_number(self.target_interior_margin)
            or self.target_interior_margin < 0
        ):
            raise ValueError("target_interior_margin must be finite and non-negative")
        if (
            self.target_interior_margin + 2 * self.numeric_tolerance
            > self.optimality_tolerance
        ):
            raise ValueError(
                "target_interior_margin and numeric guard exceed optimality tolerance"
            )
        if not _is_power_of_two_at_least_four(self.initial_disk_segments):
            raise ValueError("initial_disk_segments must be a power of two at least four")
        if not _is_power_of_two_at_least_four(self.max_disk_segments):
            raise ValueError("max_disk_segments must be a power of two at least four")
        if self.initial_disk_segments > self.max_disk_segments:
            raise ValueError("disk segment limits must be ordered")
        if self.timeout_seconds is not None and (
            not _is_finite_number(self.timeout_seconds) or self.timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be None or finite and non-negative")


@dataclass(frozen=True)
class OptimalityCertificate:
    distance_lower_bound: float
    distance_upper_bound: float
    optimality_gap: float
    radial_geometry_error: float
    numeric_error_bound: float
    disk_segments: int
    infimum_only: bool

    @classmethod
    def create(
        cls,
        *,
        distance_lower_bound: float,
        distance_upper_bound: float,
        radial_geometry_error: float,
        numeric_error_bound: float,
        disk_segments: int,
        infimum_only: bool,
    ) -> "OptimalityCertificate":
        if (
            not _is_finite_number(distance_lower_bound)
            or not _is_finite_number(distance_upper_bound)
            or distance_lower_bound < 0
            or distance_upper_bound < distance_lower_bound
        ):
            raise ValueError("certificate distance bounds are invalid")
        if (
            not _is_finite_number(radial_geometry_error)
            or radial_geometry_error < 0
        ):
            raise ValueError("radial_geometry_error must be finite and non-negative")
        if not _is_finite_number(numeric_error_bound) or numeric_error_bound < 0:
            raise ValueError("numeric_error_bound must be finite and non-negative")
        if not _is_power_of_two_at_least_four(disk_segments):
            raise ValueError("disk_segments must be a power of two at least four")
        gap = distance_upper_bound - distance_lower_bound + 2 * numeric_error_bound
        return cls(
            distance_lower_bound,
            distance_upper_bound,
            gap,
            radial_geometry_error,
            numeric_error_bound,
            disk_segments,
            infimum_only,
        )


@dataclass(frozen=True)
class FeasibleRegionBracket:
    inner: BaseGeometry
    outer: BaseGeometry
    radial_geometry_error: float
    disk_segments: int

    @classmethod
    def create(
        cls,
        *,
        inner: BaseGeometry,
        outer: BaseGeometry,
        radial_geometry_error: float,
        disk_segments: int,
        numeric_tolerance: float,
    ) -> "FeasibleRegionBracket":
        if (
            not _is_finite_number(radial_geometry_error)
            or radial_geometry_error < 0
        ):
            raise ValueError("radial_geometry_error must be finite and non-negative")
        if not inner.is_empty and not outer.buffer(numeric_tolerance).covers(inner):
            raise CertifiedGeometryError("outer region must cover inner region")
        return cls(inner, outer, radial_geometry_error, disk_segments)


@dataclass(frozen=True)
class CertifiedSolveResult:
    status: SolverStatus
    subject_position: Vec3 | None
    score: ObjectiveBreakdown | None
    quality: QualityTier
    evaluated_candidates: int
    reason: str | None
    certificate: OptimalityCertificate | None

    @classmethod
    def success(
        cls,
        *,
        subject_position: Vec3 | None,
        score: ObjectiveBreakdown | None,
        quality: QualityTier,
        evaluated_candidates: int,
        certificate: OptimalityCertificate | None,
        tolerance: float,
        leakage_count: int,
        relation_diff: tuple[str, ...],
        spec: InterventionSpec,
    ) -> "CertifiedSolveResult":
        if (
            subject_position is None
            or score is None
            or quality is not QualityTier.PURE
            or certificate is None
            or not _is_finite_number(tolerance)
            or tolerance <= 0.0
            or tolerance > _MAX_CERTIFIED_OPTIMALITY_TOLERANCE
            or certificate.optimality_gap > tolerance
            or certificate.optimality_gap > _MAX_CERTIFIED_OPTIMALITY_TOLERANCE
            or score.leakage != 0.0
            or type(leakage_count) is not int
            or leakage_count != 0
            or type(relation_diff) is not tuple
            or len(relation_diff) != 4
            or any(type(change) is not str for change in relation_diff)
            or relation_diff != expected_target_diff(spec)
        ):
            return cls(
                SolverStatus.INVALID_SCENE,
                None,
                None,
                QualityTier.REJECTED,
                evaluated_candidates,
                "invalid_success_contract",
                None,
            )
        return cls(
            SolverStatus.SUCCESS,
            subject_position,
            score,
            quality,
            evaluated_candidates,
            None,
            certificate,
        )
