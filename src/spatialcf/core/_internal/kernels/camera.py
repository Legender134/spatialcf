"""Exact camera context, directed point projection and intrinsic validation."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.so2 import SO2AtomicBudgetV2


_CAMERA_CONTEXT_HASH_DOMAIN_V2_9 = b"spatialcf.upright-camera-context.v2.9\0"


_IntervalV2 = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class UprightCameraPointBoundsV2_9:
    x_camera: _IntervalV2
    y_camera: _IntervalV2
    z_camera: _IntervalV2
    positive_depth: bool

    def __post_init__(self) -> None:
        for interval in (self.x_camera, self.y_camera, self.z_camera):
            _require_interval(interval)
        if type(self.positive_depth) is not bool:
            raise TypeError("positive_depth must be an exact bool")
        if self.positive_depth is not (self.z_camera[0] > 0):
            raise ValueError("positive_depth does not match the directed Z bound")


@dataclass(frozen=True, slots=True)
class UprightCameraContextV2_9:
    camera_id: str
    width_px: int
    height_px: int
    intrinsics: tuple[Fraction, ...]
    near_clip_m: Fraction
    far_clip_m: Fraction
    translation_xyz: tuple[Fraction, Fraction, Fraction]
    sine: _IntervalV2
    cosine: _IntervalV2

    def __post_init__(self) -> None:
        if type(self.camera_id) is not str or not self.camera_id.strip():
            raise TypeError("camera_id must be an exact non-blank string")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise TypeError("width_px must be a positive exact int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise TypeError("height_px must be a positive exact int")
        if type(self.intrinsics) is not tuple or len(self.intrinsics) != 9:
            raise TypeError("intrinsics must be an exact nine-value tuple")
        for value in (
            *self.intrinsics,
            self.near_clip_m,
            self.far_clip_m,
            *self.translation_xyz,
        ):
            _require_fraction(value)
        if self.near_clip_m <= 0 or self.far_clip_m <= self.near_clip_m:
            raise ValueError("camera clip bounds are reversed")
        _require_interval(self.sine)
        _require_interval(self.cosine)
        if self.sine[0] < -1 or self.sine[1] > 1:
            raise ValueError("sine interval escaped [-1, 1]")
        if self.cosine[0] < -1 or self.cosine[1] > 1:
            raise ValueError("cosine interval escaped [-1, 1]")

    @property
    def context_sha256(self) -> str:
        values = (
            self.camera_id,
            str(self.width_px),
            str(self.height_px),
            *(_fraction_text(value) for value in self.intrinsics),
            _fraction_text(self.near_clip_m),
            _fraction_text(self.far_clip_m),
            *(_fraction_text(value) for value in self.translation_xyz),
            *(_fraction_text(value) for value in self.sine),
            *(_fraction_text(value) for value in self.cosine),
        )
        payload = "\0".join(values).encode("utf-8")
        return hashlib.sha256(_CAMERA_CONTEXT_HASH_DOMAIN_V2_9 + payload).hexdigest()


def bound_world_point_in_upright_camera(
    context: UprightCameraContextV2_9,
    *,
    world_xyz: tuple[Fraction, Fraction, Fraction],
    delta_x: _IntervalV2,
    delta_y: _IntervalV2,
    atomic_budget: SO2AtomicBudgetV2,
) -> UprightCameraPointBoundsV2_9:
    """Bound one world point plus an XY edit in the compiled camera frame."""

    if type(context) is not UprightCameraContextV2_9:
        raise TypeError("context must be an exact UprightCameraContextV2_9")
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be an exact SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(world_xyz) is not tuple or len(world_xyz) != 3:
        raise TypeError("world_xyz must be an exact three-Fraction tuple")
    for value in world_xyz:
        _require_fraction(value)
    _require_interval(delta_x)
    _require_interval(delta_y)

    world_x = _add((world_xyz[0], world_xyz[0]), delta_x, atomic_budget)
    world_y = _add((world_xyz[1], world_xyz[1]), delta_y, atomic_budget)
    tx, ty, tz = context.translation_xyz
    x_camera = _add(
        _subtract(
            _multiply(context.cosine, world_x, atomic_budget),
            _multiply(context.sine, world_y, atomic_budget),
            atomic_budget,
        ),
        (tx, tx),
        atomic_budget,
    )
    y_exact = -world_xyz[2] + ty
    _require_fraction(y_exact)
    z_camera = _add(
        _add(
            _multiply(context.sine, world_x, atomic_budget),
            _multiply(context.cosine, world_y, atomic_budget),
            atomic_budget,
        ),
        (tz, tz),
        atomic_budget,
    )
    return UprightCameraPointBoundsV2_9(
        x_camera=x_camera,
        y_camera=(y_exact, y_exact),
        z_camera=z_camera,
        positive_depth=z_camera[0] > 0,
    )


def _require_fraction(value: Fraction) -> None:
    if type(value) is not Fraction:
        raise TypeError("directed camera values must be exact Fractions")
    so2_interval._require_numeric_fraction_cap(
        value, "NUMERIC_GAP:UPRIGHT_CAMERA_FRACTION_BIT_CAP"
    )


def _require_interval(value: _IntervalV2) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("directed camera intervals must be exact pairs")
    for endpoint in value:
        _require_fraction(endpoint)
    if value[0] > value[1]:
        raise ValueError("directed camera interval endpoints are reversed")


def _multiply(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    result = (min(products), max(products))
    _require_interval(result)
    return result


def _add(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    result = (left[0] + right[0], left[1] + right[1])
    _require_interval(result)
    return result


def _subtract(
    left: _IntervalV2,
    right: _IntervalV2,
    budget: SO2AtomicBudgetV2,
) -> _IntervalV2:
    budget.consume()
    result = (left[0] - right[1], left[1] - right[0])
    _require_interval(result)
    return result


def _fraction_text(value: Fraction) -> str:
    if value.numerator.bit_length() <= 2048 and value.denominator.bit_length() <= 2048:
        return f"{value.numerator}/{value.denominator}"
    numerator_sign = "-" if value.numerator < 0 else "+"
    return (
        "signed-hex-v1:"
        f"{numerator_sign}{format(abs(value.numerator), 'x')}"
        f"/+{format(value.denominator, 'x')}"
    )


# Preserve the supported original pickle/import lookup.
UprightCameraPointBoundsV2_9.__module__ = "spatialcf.core.problem"
UprightCameraContextV2_9.__module__ = "spatialcf.core.problem"
bound_world_point_in_upright_camera.__module__ = "spatialcf.core.problem"
_require_fraction.__module__ = "spatialcf.core.problem"
_require_interval.__module__ = "spatialcf.core.problem"
_multiply.__module__ = "spatialcf.core.problem"
_add.__module__ = "spatialcf.core.problem"
_subtract.__module__ = "spatialcf.core.problem"
_fraction_text.__module__ = "spatialcf.core.problem"
