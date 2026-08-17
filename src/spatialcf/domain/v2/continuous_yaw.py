"""Standalone Canonical wire for directed continuous-yaw proof kernels.

This module intentionally defines no Scene or SemanticProblem root.  A finite
binary64 yaw is an exact input value whose irrational trigonometric image must
be enclosed by a registered directed kernel before any consumer can use it.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from spatialcf.domain.v2.base import FiniteFloat, V2Model, Vec3V2


class DirectedYawIntervalTransformV2_2(V2Model):
    """Finite local-to-parent translation plus binary64 horizontal yaw."""

    kind: Literal["DIRECTED_YAW_INTERVAL"] = "DIRECTED_YAW_INTERVAL"
    translation: Vec3V2
    yaw_radians: FiniteFloat

    @model_validator(mode="after")
    def canonicalize_signed_zero(self) -> Self:
        translation = self.translation
        object.__setattr__(
            self,
            "translation",
            Vec3V2(
                x=0.0 if translation.x == 0.0 else translation.x,
                y=0.0 if translation.y == 0.0 else translation.y,
                z=0.0 if translation.z == 0.0 else translation.z,
            ),
        )
        object.__setattr__(
            self,
            "yaw_radians",
            0.0 if self.yaw_radians == 0.0 else self.yaw_radians,
        )
        return self


__all__ = ("DirectedYawIntervalTransformV2_2",)
