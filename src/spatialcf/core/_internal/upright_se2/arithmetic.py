"""Upright compiler arithmetic; explicit pure implementation owner."""

from __future__ import annotations

import math

import warnings

from fractions import (
    Fraction,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    Vec2,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)


def _bridge_fraction(value: float, *, label: str) -> Fraction:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be one finite exact binary64 source value")
    return Fraction.from_float(value)


def cardinal_inverse_quarter_turns(q: int) -> int:
    """Return the exact inverse in the four-element cardinal rotation group."""

    return (-upright.CardinalYaw(q=q).q) % 4


def _rotate_cardinal_xy_components(
    x: float | Fraction,
    y: float | Fraction,
    quarter_turns_ccw: int,
) -> tuple[float | Fraction, float | Fraction]:
    """Share the compiler's cardinal permutation with the public and exact paths."""

    quarter_turns_ccw = upright.CardinalYaw(q=quarter_turns_ccw).q
    if quarter_turns_ccw == 0:
        return x, y
    if quarter_turns_ccw == 1:
        return -y, x
    if quarter_turns_ccw == 2:
        return -x, -y
    if quarter_turns_ccw == 3:
        return y, -x
    raise RuntimeError("validated cardinal quarter turns escaped 0..3")


def rotate_cardinal_xy(x: float, y: float, q: int) -> tuple[float, float]:
    """Apply an exact cardinal signed-coordinate permutation to ``(x, y)``."""

    point = Vec2(x=x, y=y)
    rotated_x, rotated_y = _rotate_cardinal_xy_components(point.x, point.y, q)
    return (_canonical_zero(rotated_x), _canonical_zero(rotated_y))


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _require_exact_round_trip(value: object, model_type: type, label: str) -> None:
    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checked = model_type.model_validate(
                value.model_dump(mode="python", round_trip=True),
                strict=True,
            )
    except (TypeError, ValueError, Warning) as error:
        raise ValueError(f"{label} must pass strict canonical validation") from error
    if type(checked) is not model_type or canonical_json_bytes(
        checked
    ) != canonical_json_bytes(value):
        raise ValueError(f"{label} strict canonical bytes do not round trip")


def _dyadic_from_float(value: float) -> upright.ExactDyadic:
    return _dyadic_from_fraction(Fraction.from_float(value))


def _dyadic_from_fraction(value: Fraction) -> upright.ExactDyadic:
    return upright.ExactDyadic(
        numerator=value.numerator,
        denominator=value.denominator,
    )


def _sorted_bytes(*values):
    return tuple(sorted(values, key=canonical_json_bytes))


# Preserve supported public type/function and pickle lookup.
_bridge_fraction.__module__ = "spatialcf.core.upright_se2_compiler"
cardinal_inverse_quarter_turns.__module__ = "spatialcf.core.upright_se2_compiler"
_rotate_cardinal_xy_components.__module__ = "spatialcf.core.upright_se2_compiler"
rotate_cardinal_xy.__module__ = "spatialcf.core.upright_se2_compiler"
_canonical_zero.__module__ = "spatialcf.core.upright_se2_compiler"
_require_exact_round_trip.__module__ = "spatialcf.core.upright_se2_compiler"
_dyadic_from_float.__module__ = "spatialcf.core.upright_se2_compiler"
_dyadic_from_fraction.__module__ = "spatialcf.core.upright_se2_compiler"
_sorted_bytes.__module__ = "spatialcf.core.upright_se2_compiler"
