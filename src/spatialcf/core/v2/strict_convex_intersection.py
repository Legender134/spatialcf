"""Exact immutable values for strict-convex cell-complex intersection.

Values in this module are structural records, not verification capabilities.
Every public constructor rebuilds nested rational values defensively.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import product
from math import prod

from spatialcf.core.v2.convex_translation_domain import (
    RationalConvexPolygonV2,
    RationalPoint2V2,
)
from spatialcf.core.v2.convex_translation_partition import (
    ConvexAllowedCellComplexV2,
    RationalConvexAllowedCellV2,
    RationalHalfPlane2V2,
    RationalHalfPlaneRelationV2,
    _universe_planes_unbudgeted,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
    RectTopologyV2,
)
from spatialcf.core.v2.so2_interval import SO2_INTERVAL_MAX_FRACTION_BITS_V2

_MAX_DETERMINISTIC_LIMIT_V2 = 2**63 - 1


class StrictConvexIntersectionKindV2(StrEnum):
    COMPLEX = "COMPLEX"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class StrictConvexIntersectionTopologyV2(StrEnum):
    DISTRIBUTIVE_STRICT_CELL_INTERSECTION = "DISTRIBUTIVE_STRICT_CELL_INTERSECTION"


class StrictConvexIntersectionBudgetExhaustedV2(RuntimeError):
    """Raised before a bounded intersection counter would exceed its cap."""


class _StrictConvexIntersectionNumericGapV2(ArithmeticError):
    def __init__(self, finding_code: str) -> None:
        super().__init__(finding_code)
        self.finding_code = finding_code


@dataclass(slots=True)
class StrictConvexIntersectionBudgetV2:
    max_domain_operations: int
    max_candidate_cells: int
    domain_operations_used: int = 0
    candidate_cells_used: int = 0

    def __post_init__(self) -> None:
        self.max_domain_operations = _require_positive_limit(
            self.max_domain_operations, label="max_domain_operations"
        )
        self.max_candidate_cells = _require_positive_limit(
            self.max_candidate_cells, label="max_candidate_cells"
        )
        self.domain_operations_used = _require_initial_counter(
            self.domain_operations_used,
            maximum=self.max_domain_operations,
            label="domain_operations_used",
        )
        self.candidate_cells_used = _require_initial_counter(
            self.candidate_cells_used,
            maximum=self.max_candidate_cells,
            label="candidate_cells_used",
        )

    def consume_domain(self, amount: int = 1) -> None:
        amount = _require_consume_amount(amount)
        updated = self.domain_operations_used + amount
        if updated > self.max_domain_operations:
            raise StrictConvexIntersectionBudgetExhaustedV2(
                "strict-convex intersection domain-operation budget exhausted"
            )
        self.domain_operations_used = updated

    def consume_candidate_cells(self, amount: int = 1) -> None:
        amount = _require_consume_amount(amount)
        updated = self.candidate_cells_used + amount
        if updated > self.max_candidate_cells:
            raise StrictConvexIntersectionBudgetExhaustedV2(
                "strict-convex intersection candidate-cell budget exhausted"
            )
        self.candidate_cells_used = updated

    @property
    def limit(self) -> int:
        """Compatibility view for shared domain-operation consumers."""

        return self.max_domain_operations

    @property
    def used(self) -> int:
        """Compatibility view for shared domain-operation consumers."""

        return self.domain_operations_used

    def consume(self, amount: int = 1) -> None:
        """Charge work through the shared domain-operation interface."""

        self.consume_domain(amount)


@dataclass(frozen=True, slots=True)
class StrictConvexIntersectionCellV2:
    cell_id: str
    half_planes: tuple[RationalHalfPlane2V2, ...]
    closure_polygon: RationalConvexPolygonV2
    strict_witness: RationalPoint2V2

    def __post_init__(self) -> None:
        if type(self.cell_id) is not str or not self.cell_id.strip():
            raise ValueError("cell_id must be a non-blank exact string")
        if type(self.half_planes) is not tuple:
            raise TypeError("half_planes must be an exact tuple")
        if len(self.half_planes) < 4:
            raise ValueError("cell must carry four exact universe planes")
        checked_planes = tuple(_copy_half_plane(plane) for plane in self.half_planes)
        inferred_universe = _universe_from_planes(checked_planes[:4])
        semantic_planes = checked_planes[4:]
        if self.cell_id.startswith("cell:convex-complement:"):
            strict_indices = tuple(
                index
                for index, plane in enumerate(semantic_planes)
                if plane.relation is RationalHalfPlaneRelationV2.LT
            )
            if self.cell_id == "cell:convex-complement:universe":
                if semantic_planes:
                    raise ValueError(
                        "legacy universe cell cannot carry semantic planes"
                    )
            elif strict_indices != (len(semantic_planes) - 1,):
                raise ValueError(
                    "legacy complement cell requires one final strict plane"
                )
        else:
            if semantic_planes != tuple(
                sorted(semantic_planes, key=_semantic_plane_key)
            ):
                raise ValueError("semantic half-planes must be canonically sorted")
            coefficient_keys = tuple(
                _plane_coefficient_key(plane) for plane in semantic_planes
            )
            if len(set(coefficient_keys)) != len(coefficient_keys):
                raise ValueError("semantic half-planes cannot carry duplicate geometry")

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
        if not all(
            _universe_contains_point(inferred_universe, vertex)
            for vertex in checked_closure.vertices_ccw
        ):
            raise ValueError("closure polygon escaped its inferred universe")
        object.__setattr__(self, "half_planes", checked_planes)
        object.__setattr__(self, "closure_polygon", checked_closure)
        object.__setattr__(self, "strict_witness", checked_witness)

    def contains_point(self, point: RationalPoint2V2) -> bool:
        checked = _copy_point(point)
        return all(plane.contains_point(checked) for plane in self.half_planes)


@dataclass(frozen=True, slots=True)
class StrictConvexIntersectionComplexV2:
    cells: tuple[StrictConvexIntersectionCellV2, ...]
    universe: ExactAxisAlignedRectV2
    topology: StrictConvexIntersectionTopologyV2

    def __post_init__(self) -> None:
        if type(self.cells) is not tuple:
            raise TypeError("cells must be an exact tuple")
        if type(self.topology) is not StrictConvexIntersectionTopologyV2:
            raise TypeError("topology must be StrictConvexIntersectionTopologyV2")
        if self.topology is not (
            StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
        ):
            raise ValueError("unsupported strict-convex intersection topology")
        checked_universe = _copy_area_universe(self.universe)
        checked_cells = tuple(_copy_cell(cell) for cell in self.cells)
        cell_ids = tuple(cell.cell_id for cell in checked_cells)
        if cell_ids != tuple(sorted(cell_ids)) or len(set(cell_ids)) != len(cell_ids):
            raise ValueError("cell IDs must be unique and canonically sorted")
        expected_planes = _universe_planes(checked_universe)
        if any(cell.half_planes[:4] != expected_planes for cell in checked_cells):
            raise ValueError("cell universe half-planes do not match the complex")
        object.__setattr__(self, "cells", checked_cells)
        object.__setattr__(self, "universe", checked_universe)

    def contains_point(self, point: RationalPoint2V2) -> bool:
        checked = _copy_point(point)
        return any(cell.contains_point(checked) for cell in self.cells)


@dataclass(frozen=True, slots=True)
class StrictConvexIntersectionOutcomeV2:
    kind: StrictConvexIntersectionKindV2
    complex: StrictConvexIntersectionComplexV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not StrictConvexIntersectionKindV2:
            raise TypeError("kind must be StrictConvexIntersectionKindV2")
        if type(self.finding_codes) is not tuple or any(
            type(code) is not str or not code.strip() for code in self.finding_codes
        ):
            raise ValueError("finding_codes must be exact non-blank strings")
        findings = tuple(sorted(set(self.finding_codes)))
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is StrictConvexIntersectionKindV2.COMPLEX:
            if type(self.complex) is not StrictConvexIntersectionComplexV2:
                raise ValueError("COMPLEX outcome requires an intersection complex")
            if findings:
                raise ValueError("COMPLEX outcome cannot carry findings")
            object.__setattr__(self, "complex", _copy_complex(self.complex))
            return
        if self.complex is not None:
            raise ValueError("failure outcome cannot carry an intersection complex")
        if not findings:
            raise ValueError("failure outcome requires at least one finding")


def intersect_strict_convex_allowed_complexes_v2(
    complexes: tuple[
        ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2, ...
    ],
    *,
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionOutcomeV2:
    """Intersect a finite tuple of exact legacy or strict cell complexes."""

    try:
        checked_budget = _require_live_budget(budget)
        checked_complexes = _precharge_and_rebuild_inputs(complexes, checked_budget)
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _intersection_failure(
            StrictConvexIntersectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:STRICT_CONVEX_INTERSECTION",
        )
    except (TypeError, ValueError):
        return _intersection_failure(
            StrictConvexIntersectionKindV2.INVALID_INPUT,
            "INVALID_INPUT:STRICT_CONVEX_INTERSECTION",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return _intersect_checked_complexes(checked_complexes, checked_budget)
    except StrictConvexIntersectionBudgetExhaustedV2:
        return _intersection_failure(
            StrictConvexIntersectionKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:STRICT_CONVEX_INTERSECTION",
        )
    except _StrictConvexIntersectionNumericGapV2 as error:
        return _intersection_failure(
            StrictConvexIntersectionKindV2.NUMERIC_GAP,
            error.finding_code,
        )
    except ArithmeticError:
        return _intersection_failure(
            StrictConvexIntersectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_INTERSECTION_ARITHMETIC",
        )
    except RuntimeWarning:
        return _intersection_failure(
            StrictConvexIntersectionKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:STRICT_CONVEX_INTERSECTION_RUNTIME_WARNING",
        )


def _require_live_budget(
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionBudgetV2:
    if type(budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("budget must be StrictConvexIntersectionBudgetV2")
    _require_positive_limit(budget.max_domain_operations, label="max_domain_operations")
    _require_positive_limit(budget.max_candidate_cells, label="max_candidate_cells")
    _require_initial_counter(
        budget.domain_operations_used,
        maximum=budget.max_domain_operations,
        label="domain_operations_used",
    )
    _require_initial_counter(
        budget.candidate_cells_used,
        maximum=budget.max_candidate_cells,
        label="candidate_cells_used",
    )
    return budget


def _precharge_and_rebuild_inputs(
    complexes: object,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2, ...]:
    if type(complexes) is not tuple or not complexes:
        raise ValueError("complexes must be a non-empty exact tuple")
    budget.consume_domain(1 + len(complexes))
    rebuilt: list[ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2] = []
    for complex_ in complexes:
        if type(complex_) not in (
            ConvexAllowedCellComplexV2,
            StrictConvexIntersectionComplexV2,
        ):
            raise TypeError("every operand must be a supported exact complex")
        if type(complex_.cells) is not tuple:
            raise TypeError("operand cells must be an exact tuple")
        budget.consume_domain(1 + len(complex_.cells))
        expected_cell_type = (
            RationalConvexAllowedCellV2
            if type(complex_) is ConvexAllowedCellComplexV2
            else StrictConvexIntersectionCellV2
        )
        for cell in complex_.cells:
            if type(cell) is not expected_cell_type:
                raise TypeError("operand cell has the wrong exact type")
            if type(cell.half_planes) is not tuple:
                raise TypeError("operand half-planes must be an exact tuple")
            if type(cell.closure_polygon) is not RationalConvexPolygonV2:
                raise TypeError("operand closure must be RationalConvexPolygonV2")
            if type(cell.closure_polygon.vertices_ccw) is not tuple:
                raise TypeError("operand vertices must be an exact tuple")
            budget.consume_domain(
                1 + len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
            )
        if type(complex_) is ConvexAllowedCellComplexV2:
            rebuilt.append(
                ConvexAllowedCellComplexV2(
                    cells=complex_.cells,
                    universe=complex_.universe,
                    topology=complex_.topology,
                )
            )
        else:
            rebuilt.append(_copy_complex(complex_))
    checked = tuple(rebuilt)
    first_universe = checked[0].universe
    if any(complex_.universe != first_universe for complex_ in checked[1:]):
        raise ValueError("all operand complexes must share one exact universe")
    budget.consume_domain(prod(len(complex_.cells) for complex_ in checked))
    return checked


def _intersect_checked_complexes(
    complexes: tuple[
        ConvexAllowedCellComplexV2 | StrictConvexIntersectionComplexV2, ...
    ],
    budget: StrictConvexIntersectionBudgetV2,
) -> StrictConvexIntersectionOutcomeV2:
    if len(complexes) == 1:
        cells: list[StrictConvexIntersectionCellV2] = []
        for source in complexes[0].cells:
            budget.consume_domain(
                1 + len(source.half_planes) + len(source.closure_polygon.vertices_ccw)
            )
            budget.consume_candidate_cells()
            cells.append(
                StrictConvexIntersectionCellV2(
                    cell_id=source.cell_id,
                    half_planes=source.half_planes,
                    closure_polygon=source.closure_polygon,
                    strict_witness=source.strict_witness,
                )
            )
        return _successful_intersection(tuple(cells), complexes[0].universe)

    universe = complexes[0].universe
    universe_planes = _universe_planes(universe)
    proposals: dict[
        tuple[tuple[Fraction, Fraction, Fraction, str], ...],
        tuple[
            tuple[RationalHalfPlane2V2, ...],
            RationalConvexPolygonV2,
            RationalPoint2V2,
        ],
    ] = {}
    for source_cells in product(*(complex_.cells for complex_ in complexes)):
        budget.consume_domain(
            1
            + sum(
                len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                for cell in source_cells
            )
        )
        semantic_planes = _merge_semantic_planes(source_cells, budget)
        half_planes = universe_planes + semantic_planes
        closure_vertices = _clip_universe_by_planes(universe, semantic_planes, budget)
        if closure_vertices is None:
            continue
        witness = _find_strict_witness(closure_vertices, half_planes, budget)
        if witness is None:
            continue
        key = tuple(
            (
                plane.normal_x,
                plane.normal_y,
                plane.offset,
                plane.relation.value,
            )
            for plane in semantic_planes
        )
        budget.consume_domain()
        proposals.setdefault(
            key,
            (
                half_planes,
                RationalConvexPolygonV2(vertices_ccw=closure_vertices),
                witness,
            ),
        )

    cells = []
    for index, key in enumerate(sorted(proposals)):
        half_planes, closure, witness = proposals[key]
        budget.consume_domain()
        budget.consume_candidate_cells()
        cells.append(
            StrictConvexIntersectionCellV2(
                cell_id=f"cell:strict-convex-intersection:{index:06d}",
                half_planes=half_planes,
                closure_polygon=closure,
                strict_witness=witness,
            )
        )
    return _successful_intersection(tuple(cells), universe)


def _merge_semantic_planes(
    cells: tuple[RationalConvexAllowedCellV2 | StrictConvexIntersectionCellV2, ...],
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[RationalHalfPlane2V2, ...]:
    merged: dict[tuple[Fraction, Fraction, Fraction], RationalHalfPlaneRelationV2] = {}
    for cell in cells:
        for source in cell.half_planes[4:]:
            budget.consume_domain()
            plane = _copy_half_plane(source)
            key = _plane_coefficient_key(plane)
            prior = merged.get(key)
            if prior is None or plane.relation is RationalHalfPlaneRelationV2.LT:
                merged[key] = plane.relation
    return tuple(
        RationalHalfPlane2V2(
            normal_x=normal_x,
            normal_y=normal_y,
            offset=offset,
            relation=relation,
        )
        for (normal_x, normal_y, offset), relation in sorted(
            merged.items(),
            key=lambda item: (
                *item[0],
                0 if item[1] is RationalHalfPlaneRelationV2.LT else 1,
            ),
        )
    )


def _clip_universe_by_planes(
    universe: ExactAxisAlignedRectV2,
    planes: tuple[RationalHalfPlane2V2, ...],
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    bounds = universe.bounds
    assert bounds is not None
    vertices: tuple[RationalPoint2V2, ...] | None = (
        RationalPoint2V2(x=bounds[0], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[1]),
        RationalPoint2V2(x=bounds[2], y=bounds[3]),
        RationalPoint2V2(x=bounds[0], y=bounds[3]),
    )
    for plane in planes:
        if vertices is None:
            return None
        budget.consume_domain(2 * len(vertices) + 1)
        output: list[RationalPoint2V2] = []
        previous = vertices[-1]
        previous_value = _numeric_plane_value(previous, plane)
        previous_inside = previous_value <= 0
        for current in vertices:
            current_value = _numeric_plane_value(current, plane)
            current_inside = current_value <= 0
            if current_inside:
                if not previous_inside:
                    output.append(
                        _segment_plane_intersection(
                            previous,
                            current,
                            previous_value,
                            current_value,
                        )
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _segment_plane_intersection(
                        previous,
                        current,
                        previous_value,
                        current_value,
                    )
                )
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        vertices = _canonicalize_vertices(tuple(output), budget)
    return vertices


def _numeric_plane_value(
    point: RationalPoint2V2, plane: RationalHalfPlane2V2
) -> Fraction:
    return _require_numeric_fraction_cap(
        plane.normal_x * point.x + plane.normal_y * point.y - plane.offset
    )


def _segment_plane_intersection(
    start: RationalPoint2V2,
    end: RationalPoint2V2,
    start_value: Fraction,
    end_value: Fraction,
) -> RationalPoint2V2:
    denominator = _require_numeric_fraction_cap(start_value - end_value)
    if denominator == 0:
        raise RuntimeError("crossing edge cannot be parallel to clipping plane")
    parameter = _require_numeric_fraction_cap(start_value / denominator)
    if not 0 <= parameter <= 1:
        raise RuntimeError("clipping intersection escaped its source segment")
    return RationalPoint2V2(
        x=_require_numeric_fraction_cap(start.x + parameter * (end.x - start.x)),
        y=_require_numeric_fraction_cap(start.y + parameter * (end.y - start.y)),
    )


def _canonicalize_vertices(
    vertices: tuple[RationalPoint2V2, ...],
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[RationalPoint2V2, ...] | None:
    budget.consume_domain(1 + len(vertices))
    deduplicated: list[RationalPoint2V2] = []
    for point in vertices:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    changed = True
    while changed and len(deduplicated) >= 3:
        budget.consume_domain(len(deduplicated))
        changed = False
        kept: list[RationalPoint2V2] = []
        count = len(deduplicated)
        for index, point in enumerate(deduplicated):
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


def _cross(
    origin: RationalPoint2V2,
    left: RationalPoint2V2,
    right: RationalPoint2V2,
) -> Fraction:
    return _require_numeric_fraction_cap(
        (left.x - origin.x) * (right.y - origin.y)
        - (left.y - origin.y) * (right.x - origin.x)
    )


def _find_strict_witness(
    closure_vertices: tuple[RationalPoint2V2, ...],
    half_planes: tuple[RationalHalfPlane2V2, ...],
    budget: StrictConvexIntersectionBudgetV2,
) -> RationalPoint2V2 | None:
    candidates: list[RationalPoint2V2] = list(closure_vertices)
    for index, point in enumerate(closure_vertices):
        other = closure_vertices[(index + 1) % len(closure_vertices)]
        candidates.append(
            RationalPoint2V2(
                x=_require_numeric_fraction_cap((point.x + other.x) / 2),
                y=_require_numeric_fraction_cap((point.y + other.y) / 2),
            )
        )
    candidates.append(
        RationalPoint2V2(
            x=_require_numeric_fraction_cap(
                sum((point.x for point in closure_vertices), Fraction())
                / len(closure_vertices)
            ),
            y=_require_numeric_fraction_cap(
                sum((point.y for point in closure_vertices), Fraction())
                / len(closure_vertices)
            ),
        )
    )
    budget.consume_domain(
        3 * len(closure_vertices) + 2 + len(candidates) * len(half_planes)
    )
    for candidate in candidates:
        if all(_semantic_contains(plane, candidate) for plane in half_planes):
            return candidate
    return None


def _semantic_contains(plane: RationalHalfPlane2V2, point: RationalPoint2V2) -> bool:
    value = _require_numeric_fraction_cap(
        plane.normal_x * point.x + plane.normal_y * point.y
    )
    if plane.relation is RationalHalfPlaneRelationV2.LE:
        return value <= plane.offset
    return value < plane.offset


def _require_numeric_fraction_cap(value: Fraction) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError("numeric intersection values must be exact Fractions")
    if (
        value.numerator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
        or value.denominator.bit_length() > SO2_INTERVAL_MAX_FRACTION_BITS_V2
    ):
        raise _StrictConvexIntersectionNumericGapV2(
            "NUMERIC_GAP:STRICT_CONVEX_INTERSECTION_FRACTION_BIT_CAP"
        )
    return value


def _successful_intersection(
    cells: tuple[StrictConvexIntersectionCellV2, ...],
    universe: ExactAxisAlignedRectV2,
) -> StrictConvexIntersectionOutcomeV2:
    return StrictConvexIntersectionOutcomeV2(
        kind=StrictConvexIntersectionKindV2.COMPLEX,
        complex=StrictConvexIntersectionComplexV2(
            cells=cells,
            universe=universe,
            topology=(
                StrictConvexIntersectionTopologyV2.DISTRIBUTIVE_STRICT_CELL_INTERSECTION
            ),
        ),
    )


def _intersection_failure(
    kind: StrictConvexIntersectionKindV2, finding_code: str
) -> StrictConvexIntersectionOutcomeV2:
    return StrictConvexIntersectionOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


def _require_positive_limit(value: int, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int")
    if not 1 <= value <= _MAX_DETERMINISTIC_LIMIT_V2:
        raise ValueError(f"{label} must be in [1, 2**63 - 1]")
    return value


def _require_initial_counter(value: int, *, maximum: int, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int")
    if not 0 <= value <= maximum:
        raise ValueError(f"{label} must be within its configured limit")
    return value


def _require_consume_amount(value: int) -> int:
    if type(value) is not int:
        raise TypeError("budget amount must be an exact int")
    if value < 0:
        raise ValueError("budget amount cannot be negative")
    return value


def _copy_point(point: RationalPoint2V2) -> RationalPoint2V2:
    if type(point) is not RationalPoint2V2:
        raise TypeError("point must be RationalPoint2V2")
    return RationalPoint2V2(x=point.x, y=point.y)


def _copy_polygon(polygon: RationalConvexPolygonV2) -> RationalConvexPolygonV2:
    if type(polygon) is not RationalConvexPolygonV2:
        raise TypeError("closure_polygon must be RationalConvexPolygonV2")
    if type(polygon.vertices_ccw) is not tuple:
        raise TypeError("polygon vertices must be an exact tuple")
    return RationalConvexPolygonV2(
        vertices_ccw=tuple(_copy_point(point) for point in polygon.vertices_ccw)
    )


def _copy_half_plane(plane: RationalHalfPlane2V2) -> RationalHalfPlane2V2:
    if type(plane) is not RationalHalfPlane2V2:
        raise TypeError("half-plane must be RationalHalfPlane2V2")
    return RationalHalfPlane2V2(
        normal_x=plane.normal_x,
        normal_y=plane.normal_y,
        offset=plane.offset,
        relation=plane.relation,
    )


def _copy_area_universe(universe: ExactAxisAlignedRectV2) -> ExactAxisAlignedRectV2:
    if type(universe) is not ExactAxisAlignedRectV2:
        raise TypeError("universe must be ExactAxisAlignedRectV2")
    checked = ExactAxisAlignedRectV2(
        coordinate_space=universe.coordinate_space,
        topology=universe.topology,
        min_x_m=universe.min_x_m,
        min_y_m=universe.min_y_m,
        max_x_m=universe.max_x_m,
        max_y_m=universe.max_y_m,
    )
    if checked.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M:
        raise ValueError("intersection universe must use translation-delta XY")
    if checked.topology is not RectTopologyV2.AREA:
        raise ValueError("intersection universe must be a non-empty area rectangle")
    return checked


def _copy_cell(cell: StrictConvexIntersectionCellV2) -> StrictConvexIntersectionCellV2:
    if type(cell) is not StrictConvexIntersectionCellV2:
        raise TypeError("cell must be StrictConvexIntersectionCellV2")
    return StrictConvexIntersectionCellV2(
        cell_id=cell.cell_id,
        half_planes=cell.half_planes,
        closure_polygon=cell.closure_polygon,
        strict_witness=cell.strict_witness,
    )


def _copy_complex(
    complex_: StrictConvexIntersectionComplexV2,
) -> StrictConvexIntersectionComplexV2:
    if type(complex_) is not StrictConvexIntersectionComplexV2:
        raise TypeError("complex must be StrictConvexIntersectionComplexV2")
    return StrictConvexIntersectionComplexV2(
        cells=complex_.cells,
        universe=complex_.universe,
        topology=complex_.topology,
    )


def _plane_coefficient_key(
    plane: RationalHalfPlane2V2,
) -> tuple[Fraction, Fraction, Fraction]:
    return (plane.normal_x, plane.normal_y, plane.offset)


def _semantic_plane_key(
    plane: RationalHalfPlane2V2,
) -> tuple[Fraction, Fraction, Fraction, int]:
    relation_order = 0 if plane.relation is RationalHalfPlaneRelationV2.LT else 1
    return (*_plane_coefficient_key(plane), relation_order)


def _contains_closed_relaxation(
    plane: RationalHalfPlane2V2, point: RationalPoint2V2
) -> bool:
    return plane.normal_x * point.x + plane.normal_y * point.y <= plane.offset


def _universe_from_planes(
    planes: tuple[RationalHalfPlane2V2, ...],
) -> ExactAxisAlignedRectV2:
    if len(planes) != 4:
        raise ValueError("exactly four universe planes are required")
    if (
        planes[0].normal_x >= 0
        or planes[0].normal_y != 0
        or planes[1].normal_x <= 0
        or planes[1].normal_y != 0
        or planes[2].normal_x != 0
        or planes[2].normal_y >= 0
        or planes[3].normal_x != 0
        or planes[3].normal_y <= 0
        or any(plane.relation is not RationalHalfPlaneRelationV2.LE for plane in planes)
    ):
        raise ValueError("first four planes must be the exact universe planes")
    return _copy_area_universe(
        ExactAxisAlignedRectV2.from_fraction_bounds(
            min_x_m=_require_numeric_fraction_cap(
                planes[0].offset / planes[0].normal_x
            ),
            min_y_m=_require_numeric_fraction_cap(
                planes[2].offset / planes[2].normal_y
            ),
            max_x_m=_require_numeric_fraction_cap(
                planes[1].offset / planes[1].normal_x
            ),
            max_y_m=_require_numeric_fraction_cap(
                planes[3].offset / planes[3].normal_y
            ),
            coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
        )
    )


def _universe_planes(
    universe: ExactAxisAlignedRectV2,
) -> tuple[RationalHalfPlane2V2, ...]:
    return _universe_planes_unbudgeted(universe)


def _universe_contains_point(
    universe: ExactAxisAlignedRectV2, point: RationalPoint2V2
) -> bool:
    bounds = universe.bounds
    assert bounds is not None
    return bounds[0] <= point.x <= bounds[2] and bounds[1] <= point.y <= bounds[3]


__all__ = (
    "StrictConvexIntersectionBudgetExhaustedV2",
    "StrictConvexIntersectionBudgetV2",
    "StrictConvexIntersectionCellV2",
    "StrictConvexIntersectionComplexV2",
    "StrictConvexIntersectionKindV2",
    "StrictConvexIntersectionOutcomeV2",
    "StrictConvexIntersectionTopologyV2",
    "intersect_strict_convex_allowed_complexes_v2",
)
