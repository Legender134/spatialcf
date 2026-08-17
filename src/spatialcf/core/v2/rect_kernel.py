"""Certified rational kernel for the Canonical v2 rectangle subset.

Every accepted binary64 coordinate is lifted with :class:`fractions.Fraction`
before arithmetic.  Consequently intersection, translation, and axis offsets
are exact over the input bit patterns.  Canonical ``PlanarRegionV2`` values are
binary64, so publication uses explicit inward or outward directed rounding;
the kernel is therefore registered as ``DIRECTED_OUTWARD_BOUNDS`` rather than
claiming that the published floating-point polygon is exact.

This module deliberately supports only one hole-free, non-degenerate,
axis-aligned rectangle component.  Unsupported geometry is rejected instead
of being delegated to an uncertified general-purpose geometry library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Self

from spatialcf.domain.v2.base import Vec2V2
from spatialcf.domain.v2.geometry import (
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
    RingWindingV2,
)

RECT_KERNEL_ID_V2 = "geometry-kernel:rational-axis-aligned-rect-directed-v2"
RECT_KERNEL_VERSION_V2 = "kernel:2.0"
RECT_KERNEL_SOUNDNESS_V2 = "DIRECTED_OUTWARD_BOUNDS"
# The directed float boundary already accounts for its rounding. Zero means
# no additional numerical error remains outside that published bracket; it
# does not claim equality between the rational set and its float encoding.
RECT_KERNEL_CERTIFIED_OUTWARD_ERROR_M = 0.0


class RectKernelErrorV2(ValueError):
    """Base error for the certified rectangle subset."""


class UnsupportedRectRegionErrorV2(RectKernelErrorV2):
    """A Canonical planar region is outside the certified rectangle subset."""


class RectCoordinateSpaceMismatchErrorV2(RectKernelErrorV2):
    """An operation attempted to conflate world coordinates and edit deltas."""


class RectKernelProjectionErrorV2(RectKernelErrorV2):
    """A rational result cannot be represented by the requested float output."""


class RectTopologyV2(StrEnum):
    """Set topology retained by exact rectangle operations."""

    AREA = "AREA"
    EMPTY = "EMPTY"
    DEGENERATE = "DEGENERATE"


class RectCoordinateSpaceV2(StrEnum):
    """Non-interchangeable meanings of rectangle coordinates."""

    WORLD_XY_M = "WORLD_XY_M"
    TRANSLATION_DELTA_XY_M = "TRANSLATION_DELTA_XY_M"


class DirectedRectRoundingV2(StrEnum):
    """Whether a binary64 projection is a subset or superset."""

    INNER = "INNER"
    OUTER = "OUTER"


def _require_fraction(value: object, *, label: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{label} must be a Fraction")
    return value


def _fraction_from_binary64(value: float, *, label: str) -> Fraction:
    if type(value) is not float:
        raise TypeError(f"{label} must be a binary64 float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return Fraction.from_float(value)


@dataclass(frozen=True, slots=True)
class TranslationDeltaXYV2:
    """An exact world-XY displacement, never an absolute point."""

    dx_m: Fraction
    dy_m: Fraction

    def __post_init__(self) -> None:
        _require_fraction(self.dx_m, label="dx_m")
        _require_fraction(self.dy_m, label="dy_m")

    @classmethod
    def from_binary64(cls, *, dx_m: float, dy_m: float) -> Self:
        return cls(
            dx_m=_fraction_from_binary64(dx_m, label="dx_m"),
            dy_m=_fraction_from_binary64(dy_m, label="dy_m"),
        )


@dataclass(frozen=True, slots=True)
class WorldPointXYV2:
    """An exact absolute point in Canonical world XY."""

    x_m: Fraction
    y_m: Fraction

    def __post_init__(self) -> None:
        _require_fraction(self.x_m, label="x_m")
        _require_fraction(self.y_m, label="y_m")

    @classmethod
    def from_binary64(cls, *, x_m: float, y_m: float) -> Self:
        return cls(
            x_m=_fraction_from_binary64(x_m, label="x_m"),
            y_m=_fraction_from_binary64(y_m, label="y_m"),
        )


@dataclass(frozen=True, slots=True)
class AxisMarginXYV2:
    """Exact non-negative per-axis morphological offsets."""

    x_m: Fraction
    y_m: Fraction

    def __post_init__(self) -> None:
        x_m = _require_fraction(self.x_m, label="x_m")
        y_m = _require_fraction(self.y_m, label="y_m")
        if x_m < 0 or y_m < 0:
            raise ValueError("axis margins must be non-negative")

    @classmethod
    def from_binary64(cls, *, x_m: float, y_m: float) -> Self:
        return cls(
            x_m=_fraction_from_binary64(x_m, label="x_m"),
            y_m=_fraction_from_binary64(y_m, label="y_m"),
        )

    @classmethod
    def isotropic_from_binary64(cls, margin_m: float) -> Self:
        margin = _fraction_from_binary64(margin_m, label="margin_m")
        return cls(x_m=margin, y_m=margin)


@dataclass(frozen=True, slots=True)
class ExactAxisAlignedRectV2:
    """An immutable rectangle over exact rationals.

    ``EMPTY`` carries no arbitrary coordinates.  ``DEGENERATE`` carries a
    non-empty line or point so a touching intersection can never be mistaken
    for a proof that the set is empty.
    """

    coordinate_space: RectCoordinateSpaceV2
    topology: RectTopologyV2
    min_x_m: Fraction | None = None
    min_y_m: Fraction | None = None
    max_x_m: Fraction | None = None
    max_y_m: Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_space, RectCoordinateSpaceV2):
            raise TypeError("coordinate_space must be a RectCoordinateSpaceV2")
        if not isinstance(self.topology, RectTopologyV2):
            raise TypeError("topology must be a RectTopologyV2")
        coordinates = (self.min_x_m, self.min_y_m, self.max_x_m, self.max_y_m)
        if self.topology is RectTopologyV2.EMPTY:
            if any(value is not None for value in coordinates):
                raise ValueError("EMPTY rectangle must not carry coordinates")
            return
        if any(value is None for value in coordinates):
            raise ValueError("non-empty rectangle requires four bounds")

        min_x = _require_fraction(self.min_x_m, label="min_x_m")
        min_y = _require_fraction(self.min_y_m, label="min_y_m")
        max_x = _require_fraction(self.max_x_m, label="max_x_m")
        max_y = _require_fraction(self.max_y_m, label="max_y_m")
        if min_x > max_x or min_y > max_y:
            raise ValueError("non-empty rectangle bounds must be ordered")
        expected = (
            RectTopologyV2.AREA
            if min_x < max_x and min_y < max_y
            else RectTopologyV2.DEGENERATE
        )
        if self.topology is not expected:
            raise ValueError(f"bounds require {expected.value} topology")

    @classmethod
    def empty(cls, coordinate_space: RectCoordinateSpaceV2) -> Self:
        return cls(
            coordinate_space=coordinate_space,
            topology=RectTopologyV2.EMPTY,
        )

    @classmethod
    def from_fraction_bounds(
        cls,
        *,
        min_x_m: Fraction,
        min_y_m: Fraction,
        max_x_m: Fraction,
        max_y_m: Fraction,
        coordinate_space: RectCoordinateSpaceV2,
    ) -> Self:
        min_x = _require_fraction(min_x_m, label="min_x_m")
        min_y = _require_fraction(min_y_m, label="min_y_m")
        max_x = _require_fraction(max_x_m, label="max_x_m")
        max_y = _require_fraction(max_y_m, label="max_y_m")
        if min_x > max_x or min_y > max_y:
            return cls.empty(coordinate_space)
        topology = (
            RectTopologyV2.AREA
            if min_x < max_x and min_y < max_y
            else RectTopologyV2.DEGENERATE
        )
        return cls(
            coordinate_space=coordinate_space,
            topology=topology,
            min_x_m=min_x,
            min_y_m=min_y,
            max_x_m=max_x,
            max_y_m=max_y,
        )

    @classmethod
    def from_binary64_bounds(
        cls,
        *,
        min_x_m: float,
        min_y_m: float,
        max_x_m: float,
        max_y_m: float,
        coordinate_space: RectCoordinateSpaceV2,
    ) -> Self:
        return cls.from_fraction_bounds(
            min_x_m=_fraction_from_binary64(min_x_m, label="min_x_m"),
            min_y_m=_fraction_from_binary64(min_y_m, label="min_y_m"),
            max_x_m=_fraction_from_binary64(max_x_m, label="max_x_m"),
            max_y_m=_fraction_from_binary64(max_y_m, label="max_y_m"),
            coordinate_space=coordinate_space,
        )

    @classmethod
    def from_planar_region(
        cls,
        region: PlanarRegionV2,
        *,
        coordinate_space: RectCoordinateSpaceV2 = (RectCoordinateSpaceV2.WORLD_XY_M),
    ) -> Self:
        """Lift one canonical rectangle into exact binary64 rationals."""

        if not isinstance(region, PlanarRegionV2):
            raise TypeError("region must be a PlanarRegionV2")
        if len(region.components) != 1:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle requires exactly one component"
            )
        component = region.components[0]
        if component.holes:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle does not support holes"
            )
        vertices = component.exterior.vertices
        if len(vertices) != 4:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle requires exactly four non-redundant vertices"
            )

        rational_vertices: list[tuple[Fraction, Fraction]] = []
        for index, point in enumerate(vertices):
            try:
                x = _fraction_from_binary64(point.x, label=f"vertex[{index}].x")
                y = _fraction_from_binary64(point.y, label=f"vertex[{index}].y")
            except (TypeError, ValueError) as error:
                raise UnsupportedRectRegionErrorV2(
                    "rectangle vertices must be finite binary64 coordinates"
                ) from error
            rational_vertices.append((x, y))

        if len(set(rational_vertices)) != 4:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle vertices must be distinct"
            )
        xs = sorted({x for x, _ in rational_vertices})
        ys = sorted({y for _, y in rational_vertices})
        if len(xs) != 2 or len(ys) != 2 or xs[0] >= xs[1] or ys[0] >= ys[1]:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle must have positive area and two axis bounds"
            )
        corners = {
            (xs[0], ys[0]),
            (xs[0], ys[1]),
            (xs[1], ys[0]),
            (xs[1], ys[1]),
        }
        if set(rational_vertices) != corners:
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle edges must be axis aligned"
            )
        closed = rational_vertices[1:] + rational_vertices[:1]
        if any(
            (left[0] == right[0]) is (left[1] == right[1])
            for left, right in zip(rational_vertices, closed, strict=True)
        ):
            raise UnsupportedRectRegionErrorV2(
                "certified rectangle ring must follow four axis-aligned edges"
            )
        return cls.from_fraction_bounds(
            min_x_m=xs[0],
            min_y_m=ys[0],
            max_x_m=xs[1],
            max_y_m=ys[1],
            coordinate_space=coordinate_space,
        )

    @property
    def bounds(self) -> tuple[Fraction, Fraction, Fraction, Fraction] | None:
        if self.topology is RectTopologyV2.EMPTY:
            return None
        assert self.min_x_m is not None
        assert self.min_y_m is not None
        assert self.max_x_m is not None
        assert self.max_y_m is not None
        return (self.min_x_m, self.min_y_m, self.max_x_m, self.max_y_m)

    @property
    def width_m(self) -> Fraction | None:
        if self.bounds is None:
            return None
        return self.bounds[2] - self.bounds[0]

    @property
    def height_m(self) -> Fraction | None:
        if self.bounds is None:
            return None
        return self.bounds[3] - self.bounds[1]

    def _require_same_space(self, other: ExactAxisAlignedRectV2) -> None:
        if self.coordinate_space is not other.coordinate_space:
            raise RectCoordinateSpaceMismatchErrorV2(
                "rectangle coordinate spaces must match"
            )

    def intersect(self, other: ExactAxisAlignedRectV2) -> Self:
        if not isinstance(other, ExactAxisAlignedRectV2):
            raise TypeError("other must be an ExactAxisAlignedRectV2")
        self._require_same_space(other)
        if self.topology is RectTopologyV2.EMPTY:
            return self
        if other.topology is RectTopologyV2.EMPTY:
            return other  # type: ignore[return-value]
        left = self.bounds
        right = other.bounds
        assert left is not None and right is not None
        return type(self).from_fraction_bounds(
            min_x_m=max(left[0], right[0]),
            min_y_m=max(left[1], right[1]),
            max_x_m=min(left[2], right[2]),
            max_y_m=min(left[3], right[3]),
            coordinate_space=self.coordinate_space,
        )

    def contains(self, other: ExactAxisAlignedRectV2) -> bool:
        if not isinstance(other, ExactAxisAlignedRectV2):
            raise TypeError("other must be an ExactAxisAlignedRectV2")
        self._require_same_space(other)
        if other.topology is RectTopologyV2.EMPTY:
            return True
        if self.topology is RectTopologyV2.EMPTY:
            return False
        left = self.bounds
        right = other.bounds
        assert left is not None and right is not None
        return (
            left[0] <= right[0]
            and left[1] <= right[1]
            and right[2] <= left[2]
            and right[3] <= left[3]
        )

    def translate(self, delta: TranslationDeltaXYV2) -> Self:
        """Translate while preserving the rectangle's coordinate meaning."""

        if not isinstance(delta, TranslationDeltaXYV2):
            raise TypeError("delta must be a TranslationDeltaXYV2")
        if self.topology is RectTopologyV2.EMPTY:
            return self
        bounds = self.bounds
        assert bounds is not None
        return type(self).from_fraction_bounds(
            min_x_m=bounds[0] + delta.dx_m,
            min_y_m=bounds[1] + delta.dy_m,
            max_x_m=bounds[2] + delta.dx_m,
            max_y_m=bounds[3] + delta.dy_m,
            coordinate_space=self.coordinate_space,
        )

    def place_at(self, anchor: WorldPointXYV2) -> Self:
        """Place an anchor-relative delta rectangle at an absolute world point."""

        if not isinstance(anchor, WorldPointXYV2):
            raise TypeError("anchor must be a WorldPointXYV2")
        if self.coordinate_space is not RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M:
            raise RectCoordinateSpaceMismatchErrorV2(
                "place_at requires a translation-delta rectangle"
            )
        if self.topology is RectTopologyV2.EMPTY:
            return type(self).empty(RectCoordinateSpaceV2.WORLD_XY_M)
        bounds = self.bounds
        assert bounds is not None
        return type(self).from_fraction_bounds(
            min_x_m=bounds[0] + anchor.x_m,
            min_y_m=bounds[1] + anchor.y_m,
            max_x_m=bounds[2] + anchor.x_m,
            max_y_m=bounds[3] + anchor.y_m,
            coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M,
        )

    def relative_to(self, origin: WorldPointXYV2) -> Self:
        """Convert absolute world bounds to candidate translation deltas."""

        if not isinstance(origin, WorldPointXYV2):
            raise TypeError("origin must be a WorldPointXYV2")
        if self.coordinate_space is not RectCoordinateSpaceV2.WORLD_XY_M:
            raise RectCoordinateSpaceMismatchErrorV2(
                "relative_to requires a world-coordinate rectangle"
            )
        if self.topology is RectTopologyV2.EMPTY:
            return type(self).empty(RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M)
        bounds = self.bounds
        assert bounds is not None
        return type(self).from_fraction_bounds(
            min_x_m=bounds[0] - origin.x_m,
            min_y_m=bounds[1] - origin.y_m,
            max_x_m=bounds[2] - origin.x_m,
            max_y_m=bounds[3] - origin.y_m,
            coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
        )

    def erode_axis(self, margin: AxisMarginXYV2) -> Self:
        if not isinstance(margin, AxisMarginXYV2):
            raise TypeError("margin must be an AxisMarginXYV2")
        if self.topology is RectTopologyV2.EMPTY:
            return self
        bounds = self.bounds
        assert bounds is not None
        return type(self).from_fraction_bounds(
            min_x_m=bounds[0] + margin.x_m,
            min_y_m=bounds[1] + margin.y_m,
            max_x_m=bounds[2] - margin.x_m,
            max_y_m=bounds[3] - margin.y_m,
            coordinate_space=self.coordinate_space,
        )

    def dilate_axis(self, margin: AxisMarginXYV2) -> Self:
        if not isinstance(margin, AxisMarginXYV2):
            raise TypeError("margin must be an AxisMarginXYV2")
        if self.topology is RectTopologyV2.EMPTY:
            return self
        bounds = self.bounds
        assert bounds is not None
        return type(self).from_fraction_bounds(
            min_x_m=bounds[0] - margin.x_m,
            min_y_m=bounds[1] - margin.y_m,
            max_x_m=bounds[2] + margin.x_m,
            max_y_m=bounds[3] + margin.y_m,
            coordinate_space=self.coordinate_space,
        )

    def project_inner(self) -> FloatRectProjectionV2:
        return self._project(DirectedRectRoundingV2.INNER)

    def project_outer(self) -> FloatRectProjectionV2:
        return self._project(DirectedRectRoundingV2.OUTER)

    def _project(
        self,
        rounding: DirectedRectRoundingV2,
    ) -> FloatRectProjectionV2:
        if self.topology is RectTopologyV2.EMPTY:
            return FloatRectProjectionV2.empty(
                coordinate_space=self.coordinate_space,
                rounding=rounding,
            )
        bounds = self.bounds
        assert bounds is not None
        if rounding is DirectedRectRoundingV2.INNER:
            min_x = _fraction_to_float_ceil(bounds[0])
            min_y = _fraction_to_float_ceil(bounds[1])
            max_x = _fraction_to_float_floor(bounds[2])
            max_y = _fraction_to_float_floor(bounds[3])
        else:
            min_x = _fraction_to_float_floor(bounds[0])
            min_y = _fraction_to_float_floor(bounds[1])
            max_x = _fraction_to_float_ceil(bounds[2])
            max_y = _fraction_to_float_ceil(bounds[3])
        return FloatRectProjectionV2.from_binary64_bounds(
            min_x_m=min_x,
            min_y_m=min_y,
            max_x_m=max_x,
            max_y_m=max_y,
            coordinate_space=self.coordinate_space,
            rounding=rounding,
        )


@dataclass(frozen=True, slots=True)
class FloatRectProjectionV2:
    """A directed binary64 projection with topology kept explicit."""

    coordinate_space: RectCoordinateSpaceV2
    rounding: DirectedRectRoundingV2
    topology: RectTopologyV2
    min_x_m: float | None = None
    min_y_m: float | None = None
    max_x_m: float | None = None
    max_y_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_space, RectCoordinateSpaceV2):
            raise TypeError("coordinate_space must be a RectCoordinateSpaceV2")
        if not isinstance(self.rounding, DirectedRectRoundingV2):
            raise TypeError("rounding must be a DirectedRectRoundingV2")
        if not isinstance(self.topology, RectTopologyV2):
            raise TypeError("topology must be a RectTopologyV2")
        coordinates = (self.min_x_m, self.min_y_m, self.max_x_m, self.max_y_m)
        if self.topology is RectTopologyV2.EMPTY:
            if any(value is not None for value in coordinates):
                raise ValueError("EMPTY projection must not carry coordinates")
            return
        if any(value is None for value in coordinates):
            raise ValueError("non-empty projection requires four bounds")
        for label, value in zip(
            ("min_x_m", "min_y_m", "max_x_m", "max_y_m"),
            coordinates,
            strict=True,
        ):
            _fraction_from_binary64(value, label=label)  # type: ignore[arg-type]
        assert self.min_x_m is not None
        assert self.min_y_m is not None
        assert self.max_x_m is not None
        assert self.max_y_m is not None
        if self.min_x_m > self.max_x_m or self.min_y_m > self.max_y_m:
            raise ValueError("non-empty projection bounds must be ordered")
        expected = (
            RectTopologyV2.AREA
            if self.min_x_m < self.max_x_m and self.min_y_m < self.max_y_m
            else RectTopologyV2.DEGENERATE
        )
        if self.topology is not expected:
            raise ValueError(f"projection bounds require {expected.value} topology")

    @classmethod
    def empty(
        cls,
        *,
        coordinate_space: RectCoordinateSpaceV2,
        rounding: DirectedRectRoundingV2,
    ) -> Self:
        return cls(
            coordinate_space=coordinate_space,
            rounding=rounding,
            topology=RectTopologyV2.EMPTY,
        )

    @classmethod
    def from_binary64_bounds(
        cls,
        *,
        min_x_m: float,
        min_y_m: float,
        max_x_m: float,
        max_y_m: float,
        coordinate_space: RectCoordinateSpaceV2,
        rounding: DirectedRectRoundingV2,
    ) -> Self:
        values = (min_x_m, min_y_m, max_x_m, max_y_m)
        for label, value in zip(
            ("min_x_m", "min_y_m", "max_x_m", "max_y_m"),
            values,
            strict=True,
        ):
            _fraction_from_binary64(value, label=label)
        if min_x_m > max_x_m or min_y_m > max_y_m:
            return cls.empty(
                coordinate_space=coordinate_space,
                rounding=rounding,
            )
        topology = (
            RectTopologyV2.AREA
            if min_x_m < max_x_m and min_y_m < max_y_m
            else RectTopologyV2.DEGENERATE
        )
        return cls(
            coordinate_space=coordinate_space,
            rounding=rounding,
            topology=topology,
            min_x_m=0.0 if min_x_m == 0.0 else min_x_m,
            min_y_m=0.0 if min_y_m == 0.0 else min_y_m,
            max_x_m=0.0 if max_x_m == 0.0 else max_x_m,
            max_y_m=0.0 if max_y_m == 0.0 else max_y_m,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        if self.topology is RectTopologyV2.EMPTY:
            return None
        assert self.min_x_m is not None
        assert self.min_y_m is not None
        assert self.max_x_m is not None
        assert self.max_y_m is not None
        return (self.min_x_m, self.min_y_m, self.max_x_m, self.max_y_m)

    def to_exact_rect(self) -> ExactAxisAlignedRectV2:
        if self.topology is RectTopologyV2.EMPTY:
            return ExactAxisAlignedRectV2.empty(self.coordinate_space)
        bounds = self.bounds
        assert bounds is not None
        return ExactAxisAlignedRectV2.from_fraction_bounds(
            min_x_m=Fraction.from_float(bounds[0]),
            min_y_m=Fraction.from_float(bounds[1]),
            max_x_m=Fraction.from_float(bounds[2]),
            max_y_m=Fraction.from_float(bounds[3]),
            coordinate_space=self.coordinate_space,
        )

    def to_planar_region(self) -> PlanarRegionV2:
        """Encode an area projection; lower-dimensional sets stay explicit."""

        if self.topology is not RectTopologyV2.AREA:
            raise RectKernelProjectionErrorV2(
                "only an AREA projection can become a canonical PlanarRegionV2"
            )
        bounds = self.bounds
        assert bounds is not None
        min_x, min_y, max_x, max_y = bounds
        return PlanarRegionV2(
            components=(
                PlanarPolygonComponentV2(
                    exterior=PlanarRingV2(
                        winding=RingWindingV2.COUNTERCLOCKWISE,
                        vertices=(
                            Vec2V2(x=min_x, y=min_y),
                            Vec2V2(x=max_x, y=min_y),
                            Vec2V2(x=max_x, y=max_y),
                            Vec2V2(x=min_x, y=max_y),
                        ),
                    )
                ),
            )
        )


def _nearest_finite_float(value: Fraction) -> float:
    try:
        nearest = float(value)
    except OverflowError as error:
        raise RectKernelProjectionErrorV2(
            "rational boundary has no finite binary64 projection"
        ) from error
    if not math.isfinite(nearest):
        raise RectKernelProjectionErrorV2(
            "rational boundary has no finite binary64 projection"
        )
    return nearest


def _finite_neighbor(value: float, direction: float) -> float:
    neighbor = math.nextafter(value, direction)
    if not math.isfinite(neighbor):
        raise RectKernelProjectionErrorV2(
            "directed boundary has no finite binary64 projection"
        )
    return neighbor


def _fraction_to_float_floor(value: Fraction) -> float:
    nearest = _nearest_finite_float(value)
    if Fraction.from_float(nearest) > value:
        nearest = _finite_neighbor(nearest, -math.inf)
    return 0.0 if nearest == 0.0 else nearest


def _fraction_to_float_ceil(value: Fraction) -> float:
    nearest = _nearest_finite_float(value)
    if Fraction.from_float(nearest) < value:
        nearest = _finite_neighbor(nearest, math.inf)
    return 0.0 if nearest == 0.0 else nearest


__all__ = [
    "RECT_KERNEL_CERTIFIED_OUTWARD_ERROR_M",
    "RECT_KERNEL_ID_V2",
    "RECT_KERNEL_SOUNDNESS_V2",
    "RECT_KERNEL_VERSION_V2",
    "AxisMarginXYV2",
    "DirectedRectRoundingV2",
    "ExactAxisAlignedRectV2",
    "FloatRectProjectionV2",
    "RectCoordinateSpaceMismatchErrorV2",
    "RectCoordinateSpaceV2",
    "RectKernelErrorV2",
    "RectKernelProjectionErrorV2",
    "RectTopologyV2",
    "TranslationDeltaXYV2",
    "UnsupportedRectRegionErrorV2",
    "WorldPointXYV2",
]
