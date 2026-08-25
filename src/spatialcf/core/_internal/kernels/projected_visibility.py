"""Exact directed projected-bounding-box visibility bounds."""

from __future__ import annotations

from fractions import Fraction

_Interval = tuple[Fraction, Fraction]


def projected_bounding_box_area_fraction_lower_bound_v2_9(
    *,
    projected_u: tuple[_Interval, ...],
    projected_v: tuple[_Interval, ...],
    image_width_px: int,
    image_height_px: int,
) -> Fraction:
    """Return a sound lower bound for one projected bounding-box area.

    Every directed coordinate ``U_i`` lies in ``[l_i, u_i]``.  Therefore
    ``max(U) >= max(l_i)`` and ``min(U) <= min(u_i)`` for every realization,
    so their difference is bounded below by ``max(l_i) - min(u_i)``.  The
    same independent argument applies to V; both spans are nonnegative.
    """

    _require_intervals(projected_u, "projected_u")
    _require_intervals(projected_v, "projected_v")
    if type(image_width_px) is not int or type(image_height_px) is not int:
        raise TypeError("image dimensions must be exact integers")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image dimensions must be positive")

    width_lower = max(
        Fraction(),
        max(item[0] for item in projected_u) - min(item[1] for item in projected_u),
    )
    height_lower = max(
        Fraction(),
        max(item[0] for item in projected_v) - min(item[1] for item in projected_v),
    )
    return width_lower * height_lower / Fraction(image_width_px * image_height_px)


def _require_intervals(value: object, label: str) -> None:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a non-empty exact tuple")
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(endpoint) is not Fraction for endpoint in item)
        ):
            raise TypeError(f"{label} entries must be exact Fraction intervals")
        if item[0] > item[1]:
            raise ValueError(f"{label} interval endpoints are reversed")


__all__ = ("projected_bounding_box_area_fraction_lower_bound_v2_9",)
