"""Visibility proof data shared by projection and upright-box evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.so2 import CardinalKernelKindV3


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityPolicyV3:
    """Caller-bound visibility metric and shared-resource values."""

    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    atomic_step_limit: int

    def __post_init__(self) -> None:
        if (
            type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or type(self.metric_threshold) is not Fraction
            or not Fraction() <= self.metric_threshold <= Fraction(1)
            or type(self.metric_tolerance) is not Fraction
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.atomic_step_limit) is not int
            or self.atomic_step_limit <= 0
        ):
            raise ValueError(
                "fixed cardinal visibility policy must be a closed caller value"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalProjectionBoxV3:
    box_id: str
    center_x: Fraction
    center_y: Fraction
    center_z: Fraction
    half_x: Fraction
    half_y: Fraction
    half_z: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.box_id) is not str
            or not self.box_id
            or any(
                type(value) is not Fraction
                for value in (
                    self.center_x,
                    self.center_y,
                    self.center_z,
                    self.half_x,
                    self.half_y,
                    self.half_z,
                )
            )
            or min(self.half_x, self.half_y, self.half_z) < 0
        ):
            raise ValueError(
                "fixed cardinal projection box must be exact and non-negative"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityBoundsV3:
    cell: tuple[Fraction, Fraction, Fraction, Fraction]
    camera_context_sha256: str
    projection_convention: str
    occluder_roster: tuple[str, ...]
    subject_as_occluder: bool
    inner_fraction: Fraction
    outer_fraction: Fraction
    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    inner_success: bool
    outer_failure: bool
    gap_unknown: bool
    proof_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not tuple
            or len(self.cell) != 4
            or any(type(value) is not Fraction for value in self.cell)
            or self.cell[0] > self.cell[1]
            or self.cell[2] > self.cell[3]
            or type(self.camera_context_sha256) is not str
            or not self.camera_context_sha256
            or self.projection_convention != "RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9"
            or type(self.occluder_roster) is not tuple
            or not self.occluder_roster
            or any(
                type(value) is not str or not value for value in self.occluder_roster
            )
            or self.occluder_roster != tuple(sorted(set(self.occluder_roster)))
            or type(self.subject_as_occluder) is not bool
            or any(
                type(value) is not Fraction
                for value in (
                    self.inner_fraction,
                    self.outer_fraction,
                    self.metric_threshold,
                    self.metric_tolerance,
                )
            )
            or self.inner_fraction < 0
            or self.outer_fraction > 1
            or self.inner_fraction > self.outer_fraction
            or type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.inner_success) is not bool
            or type(self.outer_failure) is not bool
            or type(self.gap_unknown) is not bool
            or self.inner_success
            is not (
                self.inner_fraction >= self.metric_threshold - self.metric_tolerance
            )
            or self.outer_failure
            is not (self.outer_fraction < self.metric_threshold - self.metric_tolerance)
            or self.gap_unknown
            is not (not self.inner_success and not self.outer_failure)
            or type(self.proof_rows) is not tuple
            or not self.proof_rows
            or any(type(value) is not str or not value for value in self.proof_rows)
            or self.proof_rows != tuple(sorted(set(self.proof_rows)))
        ):
            raise ValueError(
                "fixed cardinal visibility bounds are not a closed exact record"
            )


@dataclass(frozen=True, slots=True)
class FixedCardinalVisibilityOutcomeV3:
    kind: CardinalKernelKindV3
    bounds: FixedCardinalVisibilityBoundsV3 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not CardinalKernelKindV3:
            raise TypeError(
                "fixed cardinal visibility outcome must use CardinalKernelKindV3"
            )
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError(
                "fixed cardinal visibility atomic usage must be non-negative"
            )
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"fixed cardinal visibility {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is CardinalKernelKindV3.EXACT:
            if (
                type(self.bounds) is not FixedCardinalVisibilityBoundsV3
                or self.finding_codes
                or not self.proof_rows
                or self.proof_rows != self.bounds.proof_rows
            ):
                raise ValueError(
                    "exact fixed cardinal visibility must carry its closed bounds"
                )
            return
        if self.bounds is not None or not self.proof_rows or not self.finding_codes:
            raise ValueError("non-exact fixed cardinal visibility must carry a finding")
        prefix = {
            CardinalKernelKindV3.NUMERIC_GAP: "NUMERIC_GAP:",
            CardinalKernelKindV3.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            CardinalKernelKindV3.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(prefix) for code in self.finding_codes):
            raise ValueError("visibility finding must match its typed outcome")


class ContinuousYawVisibilityClassificationV4(StrEnum):
    """The only whole-pose-cell visibility dispositions this owner can prove."""

    INWARD = "INWARD"
    OUTWARD = "OUTWARD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityBoundsV4:
    """Camera-owner bounds for a full closed XY-times-lifted-yaw pose cell."""

    cell: tuple[Fraction, Fraction, Fraction, Fraction]
    lifted_turn_bounds: tuple[Fraction, Fraction]
    continuous_yaw_lift_sha256: str
    camera_context_sha256: str
    projection_convention: str
    subject_box_id: str
    occluder_roster: tuple[str, ...]
    subject_as_occluder: bool
    inner_fraction: Fraction
    outer_fraction: Fraction
    metric_definition_id: str
    metric_definition_version: str
    metric_threshold: Fraction
    metric_tolerance: Fraction
    metric_comparator: str
    metric_boundary: str
    classification: ContinuousYawVisibilityClassificationV4
    outer_subject_rectangle: tuple[Fraction, Fraction, Fraction, Fraction] | None
    subject_depth_bounds: tuple[Fraction, Fraction] | None
    proof_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not tuple
            or len(self.cell) != 4
            or any(type(value) is not Fraction for value in self.cell)
            or self.cell[0] > self.cell[1]
            or self.cell[2] > self.cell[3]
            or type(self.lifted_turn_bounds) is not tuple
            or len(self.lifted_turn_bounds) != 2
            or any(type(value) is not Fraction for value in self.lifted_turn_bounds)
            or self.lifted_turn_bounds[0] > self.lifted_turn_bounds[1]
            or type(self.continuous_yaw_lift_sha256) is not str
            or len(self.continuous_yaw_lift_sha256) != 64
            or type(self.camera_context_sha256) is not str
            or not self.camera_context_sha256
            or self.projection_convention != "RETAINED_UPRIGHT_CAMERA_CONTEXT_V2_9"
            or type(self.subject_box_id) is not str
            or not self.subject_box_id
            or type(self.occluder_roster) is not tuple
            or not self.occluder_roster
            or self.occluder_roster != tuple(sorted(set(self.occluder_roster)))
            or self.subject_box_id not in self.occluder_roster
            or type(self.subject_as_occluder) is not bool
            or not self.subject_as_occluder
            or any(
                type(value) is not Fraction
                for value in (
                    self.inner_fraction,
                    self.outer_fraction,
                    self.metric_threshold,
                    self.metric_tolerance,
                )
            )
            or self.inner_fraction < 0
            or self.outer_fraction > 1
            or self.inner_fraction > self.outer_fraction
            or type(self.metric_definition_id) is not str
            or not self.metric_definition_id
            or type(self.metric_definition_version) is not str
            or not self.metric_definition_version
            or self.metric_tolerance < 0
            or self.metric_comparator != "GEQ"
            or self.metric_boundary != "CLOSED"
            or type(self.classification) is not ContinuousYawVisibilityClassificationV4
            or type(self.proof_rows) is not tuple
            or not self.proof_rows
            or any(type(value) is not str or not value for value in self.proof_rows)
            or self.proof_rows != tuple(sorted(set(self.proof_rows)))
        ):
            raise ValueError(
                "continuous visibility bounds are not a closed exact record"
            )
        int(self.continuous_yaw_lift_sha256, 16)
        expected_classification = _continuous_visibility_classification_v4(
            self.inner_fraction,
            self.outer_fraction,
            self.metric_threshold,
            self.metric_tolerance,
        )
        if self.classification is not expected_classification:
            raise ValueError(
                "continuous visibility classification does not match bounds"
            )
        for value in (self.outer_subject_rectangle, self.subject_depth_bounds):
            if value is None:
                continue
            if type(value) is not tuple or any(
                type(item) is not Fraction for item in value
            ):
                raise TypeError(
                    "continuous projection endpoints must be exact Fractions"
                )
        if self.outer_subject_rectangle is not None and (
            len(self.outer_subject_rectangle) != 4
            or self.outer_subject_rectangle[0] > self.outer_subject_rectangle[1]
            or self.outer_subject_rectangle[2] > self.outer_subject_rectangle[3]
        ):
            raise ValueError("continuous outer subject rectangle must be ordered")
        if self.subject_depth_bounds is not None and (
            len(self.subject_depth_bounds) != 2
            or self.subject_depth_bounds[0] > self.subject_depth_bounds[1]
        ):
            raise ValueError("continuous subject depth bounds must be ordered")


@dataclass(frozen=True, slots=True)
class ContinuousYawVisibilityOutcomeV4:
    """Typed completion of one continuous pose-cell camera evaluation."""

    kind: so2_interval.ContinuousYawIntervalKindV4
    bounds: ContinuousYawVisibilityBoundsV4 | None = None
    atomic_steps_used: int = 0
    proof_rows: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not so2_interval.ContinuousYawIntervalKindV4:
            raise TypeError("kind must be a ContinuousYawIntervalKindV4")
        if type(self.atomic_steps_used) is not int or self.atomic_steps_used < 0:
            raise ValueError("continuous visibility atomic usage must be non-negative")
        for field_name in ("proof_rows", "finding_codes"):
            values = getattr(self, field_name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(
                    f"continuous visibility {field_name} must be non-blank strings"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if self.kind is so2_interval.ContinuousYawIntervalKindV4.EXACT:
            if (
                type(self.bounds) is not ContinuousYawVisibilityBoundsV4
                or self.finding_codes
                or not self.proof_rows
                or self.proof_rows != self.bounds.proof_rows
            ):
                raise ValueError("exact continuous visibility must carry closed bounds")
            return
        if self.bounds is not None or not self.proof_rows or not self.finding_codes:
            raise ValueError("non-exact continuous visibility must carry a finding")
        expected_prefix = {
            so2_interval.ContinuousYawIntervalKindV4.NUMERIC_GAP: "NUMERIC_GAP:",
            so2_interval.ContinuousYawIntervalKindV4.RESOURCE_LIMIT: "RESOURCE_LIMIT:",
            so2_interval.ContinuousYawIntervalKindV4.UNSUPPORTED: "UNSUPPORTED:",
        }[self.kind]
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("continuous visibility finding must match outcome kind")


def _continuous_visibility_classification_v4(
    inner_fraction: Fraction,
    outer_fraction: Fraction,
    threshold: Fraction,
    tolerance: Fraction,
) -> ContinuousYawVisibilityClassificationV4:
    required = threshold - tolerance
    if inner_fraction >= required:
        return ContinuousYawVisibilityClassificationV4.INWARD
    if outer_fraction < required:
        return ContinuousYawVisibilityClassificationV4.OUTWARD
    return ContinuousYawVisibilityClassificationV4.UNKNOWN


# Preserve the supported original pickle/import lookup.
FixedCardinalVisibilityPolicyV3.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
FixedCardinalProjectionBoxV3.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
FixedCardinalVisibilityBoundsV3.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
FixedCardinalVisibilityOutcomeV3.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
ContinuousYawVisibilityClassificationV4.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
ContinuousYawVisibilityBoundsV4.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
ContinuousYawVisibilityOutcomeV4.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
_continuous_visibility_classification_v4.__module__ = "spatialcf.core._internal.kernels.projected_visibility"
