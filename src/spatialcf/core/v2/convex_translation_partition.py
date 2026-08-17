"""Topology-aware exact complements for rational convex translation obstacles.

The immutable values in this module are proposals, not proof capabilities.
Public semantic consumers must recompile them from raw Canonical inputs.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import gcd, lcm

from spatialcf.core.v2 import convex_translation_domain, so2_interval
from spatialcf.core.v2.convex_translation_domain import (
    CONVEX_TRANSLATION_KERNEL_ID_V2,
    CONVEX_TRANSLATION_KERNEL_VERSION_V2,
    CONVEX_TRANSLATION_MAX_POLYGON_VERTICES_V2,
    ConvexObstacleBracketV2,
    ConvexTranslationDomainKindV2,
    RationalConvexPolygonV2,
    RationalPoint2V2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.domain.v2.continuous_yaw import DirectedYawIntervalTransformV2_2
from spatialcf.domain.v2.geometry import UprightBox3DV2

CONVEX_ALLOWED_PARTITION_KERNEL_ID_V2 = (
    "geometry-kernel:rational-convex-complement-partition-v2"
)
CONVEX_ALLOWED_PARTITION_KERNEL_VERSION_V2 = (
    "kernel:2.4-topology-aware-convex-complement"
)
CONVEX_ALLOWED_PARTITION_MAX_CELLS_V2 = CONVEX_TRANSLATION_MAX_POLYGON_VERTICES_V2
_MAX_CELL_HALF_PLANES_V2 = 4 + CONVEX_ALLOWED_PARTITION_MAX_CELLS_V2


class RationalHalfPlaneRelationV2(StrEnum):
    LE = "LE"
    LT = "LT"


class ConvexAllowedTopologyV2(StrEnum):
    FIRST_VIOLATED_EDGE_STRICT_PARTITION = "FIRST_VIOLATED_EDGE_STRICT_PARTITION"


class ConvexAllowedTranslationKindV2(StrEnum):
    BRACKET = "BRACKET"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, slots=True)
class RationalHalfPlane2V2:
    """Canonical exact half-plane ``normal dot point relation offset``."""

    normal_x: Fraction
    normal_y: Fraction
    offset: Fraction
    relation: RationalHalfPlaneRelationV2

    def __post_init__(self) -> None:
        values = (self.normal_x, self.normal_y, self.offset)
        if any(type(value) is not Fraction for value in values):
            raise TypeError("half-plane coefficients must be exact Fractions")
        if type(self.relation) is not RationalHalfPlaneRelationV2:
            raise TypeError("relation must be RationalHalfPlaneRelationV2")
        for value in values:
            so2_interval._require_fraction_cap(value)
        if self.normal_x == 0 and self.normal_y == 0:
            raise ValueError("half-plane normal cannot be zero")
        if any(value.denominator != 1 for value in values):
            raise ValueError("half-plane coefficients must clear all denominators")
        coefficient_gcd = gcd(
            gcd(abs(self.normal_x.numerator), abs(self.normal_y.numerator)),
            abs(self.offset.numerator),
        )
        if coefficient_gcd != 1:
            raise ValueError("half-plane coefficients must have positive gcd one")

    def contains_point(self, point: RationalPoint2V2) -> bool:
        checked = _copy_point(point)
        value = self.normal_x * checked.x + self.normal_y * checked.y
        so2_interval._require_fraction_cap(value)
        if self.relation is RationalHalfPlaneRelationV2.LE:
            return value <= self.offset
        return value < self.offset


@dataclass(frozen=True, slots=True)
class RationalConvexAllowedCellV2:
    cell_id: str
    half_planes: tuple[RationalHalfPlane2V2, ...]
    closure_polygon: RationalConvexPolygonV2
    strict_witness: RationalPoint2V2

    def __post_init__(self) -> None:
        if type(self.cell_id) is not str or not self.cell_id.strip():
            raise ValueError("cell_id must be a non-blank exact string")
        if (
            self.cell_id != "cell:convex-complement:universe"
            and re.fullmatch(r"cell:convex-complement:0[0-7][0-9]", self.cell_id)
            is None
        ):
            raise ValueError("complement cell_id must encode edge 000 through 079")
        if type(self.half_planes) is not tuple:
            raise TypeError("half_planes must be an exact tuple")
        if not 4 <= len(self.half_planes) <= _MAX_CELL_HALF_PLANES_V2:
            raise ValueError("cell must carry four universe planes and bounded edges")
        checked_planes = tuple(_copy_half_plane(plane) for plane in self.half_planes)
        is_universe_cell = self.cell_id == "cell:convex-complement:universe"
        strict_indices = tuple(
            index
            for index, plane in enumerate(checked_planes)
            if plane.relation is RationalHalfPlaneRelationV2.LT
        )
        if is_universe_cell:
            if strict_indices:
                raise ValueError("full-universe cell cannot carry a strict plane")
        elif strict_indices != (len(checked_planes) - 1,):
            raise ValueError("complement cell requires one final strict plane")
        checked_closure = _copy_polygon(self.closure_polygon)
        checked_witness = _copy_point(self.strict_witness)
        if not all(
            _contains_closed_relaxation(plane, vertex)
            for plane in checked_planes
            for vertex in checked_closure.vertices_ccw
        ):
            raise ValueError("closure polygon escaped a closed half-plane relaxation")
        if not all(plane.contains_point(checked_witness) for plane in checked_planes):
            raise ValueError("strict witness must satisfy every semantic half-plane")
        object.__setattr__(self, "half_planes", checked_planes)
        object.__setattr__(self, "closure_polygon", checked_closure)
        object.__setattr__(self, "strict_witness", checked_witness)

    def contains_point(self, point: RationalPoint2V2) -> bool:
        checked = _copy_point(point)
        return all(plane.contains_point(checked) for plane in self.half_planes)


@dataclass(frozen=True, slots=True)
class ConvexAllowedCellComplexV2:
    cells: tuple[RationalConvexAllowedCellV2, ...]
    universe: ExactAxisAlignedRectV2
    topology: ConvexAllowedTopologyV2

    def __post_init__(self) -> None:
        if type(self.cells) is not tuple:
            raise TypeError("cells must be an exact tuple")
        if len(self.cells) > CONVEX_ALLOWED_PARTITION_MAX_CELLS_V2:
            raise ValueError("allowed complex accepts at most 80 cells")
        if type(self.topology) is not ConvexAllowedTopologyV2:
            raise TypeError("topology must be ConvexAllowedTopologyV2")
        if self.topology is not (
            ConvexAllowedTopologyV2.FIRST_VIOLATED_EDGE_STRICT_PARTITION
        ):
            raise ValueError("unsupported allowed-complex topology")
        checked_universe = _copy_universe(self.universe)
        checked_cells = tuple(_copy_cell(cell) for cell in self.cells)
        cell_ids = tuple(cell.cell_id for cell in checked_cells)
        if cell_ids != tuple(sorted(cell_ids)) or len(set(cell_ids)) != len(cell_ids):
            raise ValueError("cell IDs must be unique and canonically sorted")
        universe_cells = tuple(
            cell
            for cell in checked_cells
            if cell.cell_id == "cell:convex-complement:universe"
        )
        if universe_cells and checked_cells != universe_cells:
            raise ValueError("full-universe cell cannot mix with complement cells")
        expected_universe_planes = _universe_planes_unbudgeted(checked_universe)
        for cell in checked_cells:
            if cell.half_planes[:4] != expected_universe_planes:
                raise ValueError("cell universe half-planes do not match the complex")
            if not all(
                _universe_contains_point(checked_universe, point)
                for point in cell.closure_polygon.vertices_ccw
            ):
                raise ValueError("cell closure escaped its translation universe")
        object.__setattr__(self, "cells", checked_cells)
        object.__setattr__(self, "universe", checked_universe)

    def contains_point(self, point: RationalPoint2V2) -> bool:
        checked = _copy_point(point)
        return any(cell.contains_point(checked) for cell in self.cells)


@dataclass(frozen=True, slots=True)
class ConvexAllowedTranslationBracketV2:
    inner_allowed: ConvexAllowedCellComplexV2
    outer_allowed: ConvexAllowedCellComplexV2
    obstacle_kernel_id: str
    obstacle_kernel_version: str
    partition_kernel_id: str
    partition_kernel_version: str
    atomic_steps_used: int

    def __post_init__(self) -> None:
        checked_inner = _copy_complex(self.inner_allowed)
        checked_outer = _copy_complex(self.outer_allowed)
        if checked_inner.universe != checked_outer.universe:
            raise ValueError("inner and outer allowed complexes require one universe")
        if self.obstacle_kernel_id != CONVEX_TRANSLATION_KERNEL_ID_V2:
            raise ValueError("unexpected obstacle kernel ID")
        if self.obstacle_kernel_version != CONVEX_TRANSLATION_KERNEL_VERSION_V2:
            raise ValueError("unexpected obstacle kernel version")
        if self.partition_kernel_id != CONVEX_ALLOWED_PARTITION_KERNEL_ID_V2:
            raise ValueError("unexpected partition kernel ID")
        if self.partition_kernel_version != CONVEX_ALLOWED_PARTITION_KERNEL_VERSION_V2:
            raise ValueError("unexpected partition kernel version")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used <= 0:
            raise ValueError("atomic_steps_used must be a positive exact int")
        if not all(
            checked_outer.contains_point(cell.strict_witness)
            for cell in checked_inner.cells
        ):
            raise ValueError("inner allowed witness escaped outer allowed complex")
        object.__setattr__(self, "inner_allowed", checked_inner)
        object.__setattr__(self, "outer_allowed", checked_outer)


@dataclass(frozen=True, slots=True)
class ConvexAllowedTranslationOutcomeV2:
    kind: ConvexAllowedTranslationKindV2
    bracket: ConvexAllowedTranslationBracketV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not ConvexAllowedTranslationKindV2:
            raise TypeError("kind must be ConvexAllowedTranslationKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is ConvexAllowedTranslationKindV2.BRACKET:
            if type(self.bracket) is not ConvexAllowedTranslationBracketV2:
                raise ValueError("BRACKET outcome requires an allowed bracket")
            if findings:
                raise ValueError("BRACKET outcome cannot carry findings")
            object.__setattr__(self, "bracket", _copy_allowed_bracket(self.bracket))
            return
        if self.bracket is not None:
            raise ValueError("failure outcome cannot carry an allowed bracket")
        if not findings:
            raise ValueError("failure outcome requires at least one finding")


def compile_convex_allowed_translation_v2(
    subject_transform: DirectedYawIntervalTransformV2_2,
    subject_shape: UprightBox3DV2,
    obstacle_transform: DirectedYawIntervalTransformV2_2,
    obstacle_shape: UprightBox3DV2,
    universe: ExactAxisAlignedRectV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
) -> ConvexAllowedTranslationOutcomeV2:
    """Replay T11 once and complement its directed forbidden bracket."""

    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an SO2AtomicBudgetV2")
    atomic_budget.validate()
    initial_used = atomic_budget.used
    try:
        atomic_budget.consume()
    except SO2AtomicBudgetExhaustedV2:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_ALLOWED_TRANSLATION_ATOMIC_STEPS",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            checked_universe = _copy_universe(universe)
    except RuntimeWarning:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_ALLOWED_TRANSLATION_UNIVERSE",
        )
    except (TypeError, ValueError, Warning):
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.INVALID_INPUT,
            "INVALID_INPUT:CONVEX_ALLOWED_TRANSLATION_UNIVERSE",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            obstacle_outcome = (
                convex_translation_domain.compile_convex_translation_obstacle_v2(
                    subject_transform,
                    subject_shape,
                    obstacle_transform,
                    obstacle_shape,
                    checked_universe,
                    atomic_budget=atomic_budget,
                )
            )
    except (ArithmeticError, RuntimeWarning):
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_ALLOWED_TRANSLATION_OBSTACLE_REPLAY",
        )
    if obstacle_outcome.kind is ConvexTranslationDomainKindV2.RESOURCE_LIMIT:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_ALLOWED_TRANSLATION_ATOMIC_STEPS",
        )
    if obstacle_outcome.kind is ConvexTranslationDomainKindV2.NUMERIC_GAP:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_ALLOWED_TRANSLATION_OBSTACLE_REPLAY",
        )
    if obstacle_outcome.kind is ConvexTranslationDomainKindV2.INVALID_INPUT:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.INVALID_INPUT,
            "INVALID_INPUT:CONVEX_ALLOWED_TRANSLATION_OBSTACLE_REPLAY",
        )
    if obstacle_outcome.kind is ConvexTranslationDomainKindV2.EMPTY:
        raise RuntimeError("T11 cannot return EMPTY for a one-pair obstacle")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if obstacle_outcome.kind is ConvexTranslationDomainKindV2.IDENTITY:
                inner_allowed = _full_universe_complex(checked_universe, atomic_budget)
                outer_allowed = inner_allowed
            else:
                if (
                    obstacle_outcome.kind is not ConvexTranslationDomainKindV2.BRACKET
                    or type(obstacle_outcome.bracket) is not ConvexObstacleBracketV2
                ):
                    raise RuntimeError("malformed T11 obstacle replay outcome")
                obstacle = obstacle_outcome.bracket
                inner_allowed = _compile_polygon_complement_v2(
                    obstacle.outer_forbidden,
                    obstacle.universe,
                    atomic_budget,
                )
                if obstacle.inner_forbidden is None:
                    outer_allowed = _full_universe_complex(
                        obstacle.universe,
                        atomic_budget,
                    )
                else:
                    outer_allowed = _compile_polygon_complement_v2(
                        obstacle.inner_forbidden,
                        obstacle.universe,
                        atomic_budget,
                    )
            atomic_budget.consume(
                2 * _bracket_validation_charge(inner_allowed, outer_allowed)
            )
            bracket = ConvexAllowedTranslationBracketV2(
                inner_allowed=inner_allowed,
                outer_allowed=outer_allowed,
                obstacle_kernel_id=CONVEX_TRANSLATION_KERNEL_ID_V2,
                obstacle_kernel_version=CONVEX_TRANSLATION_KERNEL_VERSION_V2,
                partition_kernel_id=CONVEX_ALLOWED_PARTITION_KERNEL_ID_V2,
                partition_kernel_version=CONVEX_ALLOWED_PARTITION_KERNEL_VERSION_V2,
                atomic_steps_used=atomic_budget.used - initial_used,
            )
            return ConvexAllowedTranslationOutcomeV2(
                kind=ConvexAllowedTranslationKindV2.BRACKET,
                bracket=bracket,
            )
    except SO2AtomicBudgetExhaustedV2:
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONVEX_ALLOWED_TRANSLATION_ATOMIC_STEPS",
        )
    except (ArithmeticError, RuntimeWarning):
        return _allowed_failure(
            ConvexAllowedTranslationKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONVEX_ALLOWED_TRANSLATION_PARTITION",
        )


def _canonical_half_plane_v2(
    normal_x: Fraction,
    normal_y: Fraction,
    offset: Fraction,
    relation: RationalHalfPlaneRelationV2,
    budget: SO2AtomicBudgetV2,
) -> RationalHalfPlane2V2:
    if type(budget) is not SO2AtomicBudgetV2:
        raise TypeError("budget must be an SO2AtomicBudgetV2")
    budget.validate()
    budget.consume(7)
    values = (normal_x, normal_y, offset)
    if any(type(value) is not Fraction for value in values):
        raise TypeError("half-plane coefficients must be exact Fractions")
    if type(relation) is not RationalHalfPlaneRelationV2:
        raise TypeError("relation must be RationalHalfPlaneRelationV2")
    return _canonical_half_plane_unbudgeted(normal_x, normal_y, offset, relation)


def _canonical_half_plane_unbudgeted(
    normal_x: Fraction,
    normal_y: Fraction,
    offset: Fraction,
    relation: RationalHalfPlaneRelationV2,
) -> RationalHalfPlane2V2:
    for value in (normal_x, normal_y, offset):
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONVEX_COMPLEMENT_HALF_PLANE_FRACTION_BIT_CAP",
        )
    common_denominator = lcm(
        normal_x.denominator,
        normal_y.denominator,
        offset.denominator,
    )
    integers = (
        normal_x.numerator * (common_denominator // normal_x.denominator),
        normal_y.numerator * (common_denominator // normal_y.denominator),
        offset.numerator * (common_denominator // offset.denominator),
    )
    if integers[0] == 0 and integers[1] == 0:
        raise ValueError("half-plane normal cannot be zero")
    coefficient_gcd = gcd(gcd(abs(integers[0]), abs(integers[1])), abs(integers[2]))
    if coefficient_gcd == 0:
        coefficient_gcd = 1
    canonical = tuple(Fraction(value // coefficient_gcd) for value in integers)
    for value in canonical:
        so2_interval._require_numeric_fraction_cap(
            value,
            "NUMERIC_GAP:CONVEX_COMPLEMENT_HALF_PLANE_FRACTION_BIT_CAP",
        )
    return RationalHalfPlane2V2(
        normal_x=canonical[0],
        normal_y=canonical[1],
        offset=canonical[2],
        relation=relation,
    )


def _compile_polygon_complement_v2(
    polygon: RationalConvexPolygonV2,
    universe: ExactAxisAlignedRectV2,
    budget: SO2AtomicBudgetV2,
) -> ConvexAllowedCellComplexV2:
    """Compile ``universe \\ polygon`` by the first violated polygon edge."""

    if type(budget) is not SO2AtomicBudgetV2:
        raise TypeError("budget must be an SO2AtomicBudgetV2")
    budget.validate()
    if type(polygon) is not RationalConvexPolygonV2:
        raise TypeError("polygon must be RationalConvexPolygonV2")
    budget.consume(3 + len(polygon.vertices_ccw))
    checked_polygon = _copy_polygon(polygon)
    checked_universe = _copy_universe(universe)
    universe_planes = _universe_planes(checked_universe, budget)
    prefix_vertices: tuple[RationalPoint2V2, ...] | None = _universe_vertices(
        checked_universe
    )
    prior_inside: list[RationalHalfPlane2V2] = []
    cells: list[RationalConvexAllowedCellV2] = []
    vertices = checked_polygon.vertices_ccw
    for edge_index, first in enumerate(vertices):
        if prefix_vertices is None:
            break
        budget.consume(5)
        second = vertices[(edge_index + 1) % len(vertices)]
        edge_x = second.x - first.x
        edge_y = second.y - first.y
        inside = _canonical_half_plane_v2(
            edge_y,
            -edge_x,
            edge_y * first.x - edge_x * first.y,
            RationalHalfPlaneRelationV2.LE,
            budget,
        )
        outside = _canonical_half_plane_v2(
            -edge_y,
            edge_x,
            -edge_y * first.x + edge_x * first.y,
            RationalHalfPlaneRelationV2.LT,
            budget,
        )
        closure_vertices = _clip_vertices_by_half_plane(
            prefix_vertices,
            outside,
            budget,
        )
        if closure_vertices is not None:
            half_planes = universe_planes + tuple(prior_inside) + (outside,)
            witness = _find_strict_witness(closure_vertices, half_planes, budget)
            if witness is not None:
                # Reserve the initial strict polygon reconstruction separately
                # from the later cell defensive-copy validation.
                budget.consume(4 + len(closure_vertices))
                closure = RationalConvexPolygonV2(vertices_ccw=closure_vertices)
                budget.consume(_cell_validation_charge(closure, half_planes))
                cells.append(
                    RationalConvexAllowedCellV2(
                        cell_id=f"cell:convex-complement:{edge_index:03d}",
                        half_planes=half_planes,
                        closure_polygon=closure,
                        strict_witness=witness,
                    )
                )
        prefix_vertices = _clip_vertices_by_half_plane(
            prefix_vertices,
            inside,
            budget,
        )
        prior_inside.append(inside)
    budget.consume(_complex_validation_charge(tuple(cells)))
    return ConvexAllowedCellComplexV2(
        cells=tuple(cells),
        universe=checked_universe,
        topology=ConvexAllowedTopologyV2.FIRST_VIOLATED_EDGE_STRICT_PARTITION,
    )


def _universe_planes(
    universe: ExactAxisAlignedRectV2,
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalHalfPlane2V2, ...]:
    bounds = universe.bounds
    assert bounds is not None
    return (
        _canonical_half_plane_v2(
            Fraction(-1),
            Fraction(),
            -bounds[0],
            RationalHalfPlaneRelationV2.LE,
            budget,
        ),
        _canonical_half_plane_v2(
            Fraction(1),
            Fraction(),
            bounds[2],
            RationalHalfPlaneRelationV2.LE,
            budget,
        ),
        _canonical_half_plane_v2(
            Fraction(),
            Fraction(-1),
            -bounds[1],
            RationalHalfPlaneRelationV2.LE,
            budget,
        ),
        _canonical_half_plane_v2(
            Fraction(),
            Fraction(1),
            bounds[3],
            RationalHalfPlaneRelationV2.LE,
            budget,
        ),
    )


def _universe_planes_unbudgeted(
    universe: ExactAxisAlignedRectV2,
) -> tuple[RationalHalfPlane2V2, ...]:
    bounds = universe.bounds
    assert bounds is not None
    return (
        _canonical_half_plane_unbudgeted(
            Fraction(-1), Fraction(), -bounds[0], RationalHalfPlaneRelationV2.LE
        ),
        _canonical_half_plane_unbudgeted(
            Fraction(1), Fraction(), bounds[2], RationalHalfPlaneRelationV2.LE
        ),
        _canonical_half_plane_unbudgeted(
            Fraction(), Fraction(-1), -bounds[1], RationalHalfPlaneRelationV2.LE
        ),
        _canonical_half_plane_unbudgeted(
            Fraction(), Fraction(1), bounds[3], RationalHalfPlaneRelationV2.LE
        ),
    )


def _universe_vertices(
    universe: ExactAxisAlignedRectV2,
) -> tuple[RationalPoint2V2, ...]:
    bounds = universe.bounds
    assert bounds is not None
    return (
        RationalPoint2V2(x=bounds[0], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[3]),
        RationalPoint2V2(x=bounds[0], y=bounds[3]),
    )


def _clip_vertices_by_half_plane(
    vertices: tuple[RationalPoint2V2, ...],
    plane: RationalHalfPlane2V2,
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    if len(vertices) < 3:
        raise RuntimeError("half-plane clip requires an area polygon")
    output: list[RationalPoint2V2] = []
    previous = vertices[-1]
    previous_value = _plane_value(previous, plane, budget)
    previous_inside = previous_value <= 0
    for current in vertices:
        current_value = _plane_value(current, plane, budget)
        current_inside = current_value <= 0
        if current_inside:
            if not previous_inside:
                output.append(
                    _segment_plane_intersection(
                        previous, current, previous_value, current_value, budget
                    )
                )
            output.append(current)
        elif previous_inside:
            output.append(
                _segment_plane_intersection(
                    previous, current, previous_value, current_value, budget
                )
            )
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _canonicalize_vertices(tuple(output), budget)


def _plane_value(
    point: RationalPoint2V2,
    plane: RationalHalfPlane2V2,
    budget: SO2AtomicBudgetV2,
) -> Fraction:
    budget.consume()
    value = plane.normal_x * point.x + plane.normal_y * point.y - plane.offset
    so2_interval._require_numeric_fraction_cap(
        value,
        "NUMERIC_GAP:CONVEX_COMPLEMENT_CLIP_FRACTION_BIT_CAP",
    )
    return value


def _segment_plane_intersection(
    start: RationalPoint2V2,
    end: RationalPoint2V2,
    start_value: Fraction,
    end_value: Fraction,
    budget: SO2AtomicBudgetV2,
) -> RationalPoint2V2:
    budget.consume()
    denominator = start_value - end_value
    if denominator == 0:
        raise RuntimeError("crossing edge cannot be parallel to clipping plane")
    parameter = start_value / denominator
    if not 0 <= parameter <= 1:
        raise RuntimeError("clipping intersection escaped its source segment")
    point = RationalPoint2V2(
        x=start.x + parameter * (end.x - start.x),
        y=start.y + parameter * (end.y - start.y),
    )
    return point


def _canonicalize_vertices(
    vertices: tuple[RationalPoint2V2, ...],
    budget: SO2AtomicBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    deduplicated: list[RationalPoint2V2] = []
    for point in vertices:
        budget.consume()
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    changed = True
    while changed and len(deduplicated) >= 3:
        changed = False
        kept: list[RationalPoint2V2] = []
        count = len(deduplicated)
        for index, point in enumerate(deduplicated):
            budget.consume()
            cross = _cross(
                deduplicated[index - 1],
                point,
                deduplicated[(index + 1) % count],
            )
            if cross < 0:
                raise RuntimeError("half-plane clipping broke convex winding")
            if cross == 0:
                changed = True
            else:
                kept.append(point)
        deduplicated = kept
    if len(deduplicated) < 3:
        return None
    least = min(
        range(len(deduplicated)),
        key=lambda index: (deduplicated[index].x, deduplicated[index].y),
    )
    return tuple(deduplicated[least:] + deduplicated[:least])


def _find_strict_witness(
    closure_vertices: tuple[RationalPoint2V2, ...],
    half_planes: tuple[RationalHalfPlane2V2, ...],
    budget: SO2AtomicBudgetV2,
) -> RationalPoint2V2 | None:
    # Reserve candidate construction before midpoint/centroid Fraction work.
    budget.consume(3 * len(closure_vertices) + 2)
    candidates: list[RationalPoint2V2] = list(closure_vertices)
    for index, point in enumerate(closure_vertices):
        other = closure_vertices[(index + 1) % len(closure_vertices)]
        candidates.append(
            RationalPoint2V2(x=(point.x + other.x) / 2, y=(point.y + other.y) / 2)
        )
    candidates.append(
        RationalPoint2V2(
            x=sum((point.x for point in closure_vertices), Fraction())
            / len(closure_vertices),
            y=sum((point.y for point in closure_vertices), Fraction())
            / len(closure_vertices),
        )
    )
    budget.consume(len(candidates) * len(half_planes))
    for candidate in candidates:
        if all(plane.contains_point(candidate) for plane in half_planes):
            return candidate
    return None


def _contains_closed_relaxation(
    plane: RationalHalfPlane2V2,
    point: RationalPoint2V2,
) -> bool:
    value = plane.normal_x * point.x + plane.normal_y * point.y
    so2_interval._require_fraction_cap(value)
    return value <= plane.offset


def _cross(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
) -> Fraction:
    value = (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (
        right.x - origin.x
    )
    so2_interval._require_numeric_fraction_cap(
        value,
        "NUMERIC_GAP:CONVEX_COMPLEMENT_CROSS_FRACTION_BIT_CAP",
    )
    return value


def _cell_validation_charge(
    closure: RationalConvexPolygonV2,
    half_planes: tuple[RationalHalfPlane2V2, ...],
) -> int:
    return 4 + len(half_planes) * (len(closure.vertices_ccw) + 1)


def _complex_validation_charge(
    cells: tuple[RationalConvexAllowedCellV2, ...],
) -> int:
    return 5 + sum(
        12
        + len(cell.closure_polygon.vertices_ccw)
        + len(cell.half_planes) * (len(cell.closure_polygon.vertices_ccw) + 1)
        for cell in cells
    )


def _full_universe_complex(
    universe: ExactAxisAlignedRectV2,
    budget: SO2AtomicBudgetV2,
) -> ConvexAllowedCellComplexV2:
    checked_universe = _copy_universe(universe)
    planes = _universe_planes(checked_universe, budget)
    # Precharge the rectangle polygon plus exact center construction.
    budget.consume(10)
    closure = RationalConvexPolygonV2(vertices_ccw=_universe_vertices(checked_universe))
    bounds = checked_universe.bounds
    assert bounds is not None
    witness = RationalPoint2V2(
        x=(bounds[0] + bounds[2]) / 2,
        y=(bounds[1] + bounds[3]) / 2,
    )
    budget.consume(_cell_validation_charge(closure, planes))
    cell = RationalConvexAllowedCellV2(
        cell_id="cell:convex-complement:universe",
        half_planes=planes,
        closure_polygon=closure,
        strict_witness=witness,
    )
    budget.consume(_complex_validation_charge((cell,)))
    return ConvexAllowedCellComplexV2(
        cells=(cell,),
        universe=checked_universe,
        topology=ConvexAllowedTopologyV2.FIRST_VIOLATED_EDGE_STRICT_PARTITION,
    )


def _bracket_validation_charge(
    inner: ConvexAllowedCellComplexV2,
    outer: ConvexAllowedCellComplexV2,
) -> int:
    outer_membership_cost = sum(len(cell.half_planes) for cell in outer.cells)
    return (
        8
        + _complex_validation_charge(inner.cells)
        + _complex_validation_charge(outer.cells)
        + len(inner.cells) * max(1, outer_membership_cost)
    )


def _allowed_failure(
    kind: ConvexAllowedTranslationKindV2,
    finding_code: str,
) -> ConvexAllowedTranslationOutcomeV2:
    return ConvexAllowedTranslationOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _copy_point(point: object) -> RationalPoint2V2:
    if type(point) is not RationalPoint2V2:
        raise TypeError("point must be RationalPoint2V2")
    return RationalPoint2V2(x=point.x, y=point.y)


def _copy_polygon(polygon: object) -> RationalConvexPolygonV2:
    if type(polygon) is not RationalConvexPolygonV2:
        raise TypeError("polygon must be RationalConvexPolygonV2")
    return RationalConvexPolygonV2(vertices_ccw=polygon.vertices_ccw)


def _copy_half_plane(plane: object) -> RationalHalfPlane2V2:
    if type(plane) is not RationalHalfPlane2V2:
        raise TypeError("half-plane must be RationalHalfPlane2V2")
    return RationalHalfPlane2V2(
        normal_x=plane.normal_x,
        normal_y=plane.normal_y,
        offset=plane.offset,
        relation=plane.relation,
    )


def _copy_cell(cell: object) -> RationalConvexAllowedCellV2:
    if type(cell) is not RationalConvexAllowedCellV2:
        raise TypeError("cell must be RationalConvexAllowedCellV2")
    return RationalConvexAllowedCellV2(
        cell_id=cell.cell_id,
        half_planes=cell.half_planes,
        closure_polygon=cell.closure_polygon,
        strict_witness=cell.strict_witness,
    )


def _copy_complex(complex_: object) -> ConvexAllowedCellComplexV2:
    if type(complex_) is not ConvexAllowedCellComplexV2:
        raise TypeError("complex must be ConvexAllowedCellComplexV2")
    return ConvexAllowedCellComplexV2(
        cells=complex_.cells,
        universe=complex_.universe,
        topology=complex_.topology,
    )


def _copy_allowed_bracket(
    bracket: object,
) -> ConvexAllowedTranslationBracketV2:
    if type(bracket) is not ConvexAllowedTranslationBracketV2:
        raise TypeError("bracket must be ConvexAllowedTranslationBracketV2")
    return ConvexAllowedTranslationBracketV2(
        inner_allowed=bracket.inner_allowed,
        outer_allowed=bracket.outer_allowed,
        obstacle_kernel_id=bracket.obstacle_kernel_id,
        obstacle_kernel_version=bracket.obstacle_kernel_version,
        partition_kernel_id=bracket.partition_kernel_id,
        partition_kernel_version=bracket.partition_kernel_version,
        atomic_steps_used=bracket.atomic_steps_used,
    )


def _copy_universe(universe: object) -> ExactAxisAlignedRectV2:
    if type(universe) is not ExactAxisAlignedRectV2:
        raise TypeError("universe must be ExactAxisAlignedRectV2")
    if (
        universe.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M
        or universe.topology is not RectTopologyV2.AREA
        or universe.bounds is None
    ):
        raise ValueError("universe must be an AREA translation-delta rectangle")
    min_x, min_y, max_x, max_y = universe.bounds
    for value in (min_x, min_y, max_x, max_y):
        so2_interval._require_fraction_cap(value)
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def _universe_contains_point(
    universe: ExactAxisAlignedRectV2,
    point: RationalPoint2V2,
) -> bool:
    bounds = universe.bounds
    assert bounds is not None
    return bounds[0] <= point.x <= bounds[2] and bounds[1] <= point.y <= bounds[3]


__all__ = (
    "CONVEX_ALLOWED_PARTITION_KERNEL_ID_V2",
    "CONVEX_ALLOWED_PARTITION_KERNEL_VERSION_V2",
    "CONVEX_ALLOWED_PARTITION_MAX_CELLS_V2",
    "ConvexAllowedCellComplexV2",
    "ConvexAllowedTopologyV2",
    "ConvexAllowedTranslationBracketV2",
    "ConvexAllowedTranslationKindV2",
    "ConvexAllowedTranslationOutcomeV2",
    "RationalConvexAllowedCellV2",
    "RationalHalfPlane2V2",
    "RationalHalfPlaneRelationV2",
    "compile_convex_allowed_translation_v2",
)
