"""Upright compiler evaluation data; explicit pure implementation owner."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)

from fractions import (
    Fraction,
)

from spatialcf.core._internal.kernels.projected_visibility import (
    FixedCardinalProjectionBoxV3,
    FixedCardinalVisibilityPolicyV3,
)

from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    FixedCardinalBoxV3,
    FixedCardinalCellPolicyV3,
    SupportSurfaceV3,
)

from spatialcf.core.problem import (
    UprightCameraContextV2_9,
)

from spatialcf.domain.predicates import (
    PredicateAtom,
    PreservationInvariant,
)

from spatialcf.domain.scene import (
    ObjectSupportAssignment,
    SupportSurfaceFact,
)


@dataclass(frozen=True, slots=True)
class UprightSE2CardinalVisibilityEvaluationInput:
    """One exact retained projected-visibility invocation input.

    The bridge supplies this immutable record to both future proposal and
    fresh-checker owners.  It contains no budget instance or result: callers
    create their own shared retained-kernel ledger before invoking an owner.
    """

    observation_id: str
    context: UprightCameraContextV2_9
    cell: tuple[Fraction, Fraction, Fraction, Fraction]
    subject: FixedCardinalProjectionBoxV3
    moving_subject_id: str
    occluders: tuple[FixedCardinalProjectionBoxV3, ...]
    required_occluder_ids: tuple[str, ...]
    policy: FixedCardinalVisibilityPolicyV3

    def __post_init__(self) -> None:
        if (
            type(self.observation_id) is not str
            or not self.observation_id
            or type(self.context) is not UprightCameraContextV2_9
            or type(self.cell) is not tuple
            or len(self.cell) != 4
            or any(type(value) is not Fraction for value in self.cell)
            or self.cell[0] > self.cell[1]
            or self.cell[2] > self.cell[3]
            or type(self.subject) is not FixedCardinalProjectionBoxV3
            or type(self.moving_subject_id) is not str
            or not self.moving_subject_id
            or type(self.occluders) is not tuple
            or not self.occluders
            or any(
                type(box) is not FixedCardinalProjectionBoxV3 for box in self.occluders
            )
            or tuple(box.box_id for box in self.occluders)
            != tuple(sorted(box.box_id for box in self.occluders))
            or type(self.required_occluder_ids) is not tuple
            or self.required_occluder_ids != tuple(box.box_id for box in self.occluders)
            or self.subject.box_id not in self.required_occluder_ids
            or self.moving_subject_id not in self.required_occluder_ids
            or type(self.policy) is not FixedCardinalVisibilityPolicyV3
        ):
            raise ValueError(
                "cardinal visibility bridge input must be complete and exact"
            )


@dataclass(frozen=True, slots=True)
class UprightSE2CardinalEvaluationInputs:
    """The complete immutable retained-kernel input bundle for one M3 cell."""

    solve_request_sha256: str
    semantic_closure_sha256: str
    policy_bundle_sha256: str
    upright_se2_compilation_sha256: str
    compiled_cell_sha256: str
    cell: ClosedXYCellV3
    quarter_turns_ccw: int
    subject_boxes: tuple[FixedCardinalBoxV3, ...]
    obstacle_boxes: tuple[FixedCardinalBoxV3, ...]
    support_assignment: ObjectSupportAssignment
    support_surface_fact: SupportSurfaceFact
    support_surface: SupportSurfaceV3
    target_before_relation: PredicateAtom
    target_after_relation: PredicateAtom
    preservation_invariants: tuple[PreservationInvariant, ...]
    relation: str
    reference_box: FixedCardinalBoxV3
    near_far_threshold: Fraction
    cell_policy: FixedCardinalCellPolicyV3
    subject_pivot_xy: tuple[Fraction, Fraction]
    objective_subject_pivot_xy: tuple[Fraction, Fraction]
    visibility_inputs: tuple[UprightSE2CardinalVisibilityEvaluationInput, ...]
    resource_atomic_step_limit: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str or not value
                for value in (
                    self.solve_request_sha256,
                    self.semantic_closure_sha256,
                    self.policy_bundle_sha256,
                    self.upright_se2_compilation_sha256,
                    self.compiled_cell_sha256,
                )
            )
            or type(self.cell) is not ClosedXYCellV3
            or type(self.quarter_turns_ccw) is not int
            or self.quarter_turns_ccw not in (0, 1, 2, 3)
            or type(self.subject_boxes) is not tuple
            or not self.subject_boxes
            or any(type(box) is not FixedCardinalBoxV3 for box in self.subject_boxes)
            or tuple(box.box_id for box in self.subject_boxes)
            != tuple(sorted(box.box_id for box in self.subject_boxes))
            or type(self.obstacle_boxes) is not tuple
            or any(type(box) is not FixedCardinalBoxV3 for box in self.obstacle_boxes)
            or tuple(box.box_id for box in self.obstacle_boxes)
            != tuple(sorted(box.box_id for box in self.obstacle_boxes))
            or type(self.support_assignment) is not ObjectSupportAssignment
            or type(self.support_surface_fact) is not SupportSurfaceFact
            or type(self.support_surface) is not SupportSurfaceV3
            or type(self.target_before_relation) is not PredicateAtom
            or type(self.target_after_relation) is not PredicateAtom
            or type(self.preservation_invariants) is not tuple
            or not self.preservation_invariants
            or any(
                type(item) is not PreservationInvariant
                for item in self.preservation_invariants
            )
            or self.relation not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR")
            or type(self.reference_box) is not FixedCardinalBoxV3
            or type(self.near_far_threshold) is not Fraction
            or self.near_far_threshold < 0
            or type(self.cell_policy) is not FixedCardinalCellPolicyV3
            or self.cell_policy.visibility_cell != self.cell.canonical_bounds
            or self.cell_policy.relation_symbol != self.relation
            or self.cell_policy.relation_threshold != self.near_far_threshold
            or type(self.subject_pivot_xy) is not tuple
            or len(self.subject_pivot_xy) != 2
            or any(type(value) is not Fraction for value in self.subject_pivot_xy)
            or type(self.objective_subject_pivot_xy) is not tuple
            or len(self.objective_subject_pivot_xy) != 2
            or any(
                type(value) is not Fraction for value in self.objective_subject_pivot_xy
            )
            or type(self.visibility_inputs) is not tuple
            or not self.visibility_inputs
            or any(
                type(item) is not UprightSE2CardinalVisibilityEvaluationInput
                for item in self.visibility_inputs
            )
            or tuple(item.observation_id for item in self.visibility_inputs)
            != tuple(sorted(item.observation_id for item in self.visibility_inputs))
            or type(self.resource_atomic_step_limit) is not int
            or self.resource_atomic_step_limit <= 0
            or self.cell_policy.atomic_step_limit != self.resource_atomic_step_limit
            or any(
                item.policy.atomic_step_limit != self.resource_atomic_step_limit
                for item in self.visibility_inputs
            )
        ):
            raise ValueError(
                "cardinal evaluation bridge inputs must be complete and exact"
            )


@dataclass(frozen=True, slots=True)
class UprightSE2ContinuousVisibilityEvaluationInput:
    """One source-bound V4 visibility invocation without a kernel result.

    The continuous backend may turn these fixed source boxes into Task 6 V4
    pose-cell bounds, but it never reconstructs source facts or policy values.
    Exactly one observation is accepted by the current V4 compound-owner seam;
    a broader observation conjunction must be added by that semantic owner,
    not silently aggregated by proposal orchestration.
    """

    observation_id: str
    source_visual_box_id: str
    context: UprightCameraContextV2_9
    subject: FixedCardinalBoxV3
    moving_subject_id: str
    occluders: tuple[FixedCardinalBoxV3, ...]
    required_occluder_ids: tuple[str, ...]
    policy: FixedCardinalVisibilityPolicyV3

    def __post_init__(self) -> None:
        if (
            type(self.observation_id) is not str
            or not self.observation_id
            or type(self.source_visual_box_id) is not str
            or not self.source_visual_box_id
            or type(self.context) is not UprightCameraContextV2_9
            or type(self.subject) is not FixedCardinalBoxV3
            or type(self.moving_subject_id) is not str
            or not self.moving_subject_id
            or type(self.occluders) is not tuple
            or not self.occluders
            or any(type(box) is not FixedCardinalBoxV3 for box in self.occluders)
            or tuple(box.box_id for box in self.occluders)
            != tuple(sorted(box.box_id for box in self.occluders))
            or type(self.required_occluder_ids) is not tuple
            or self.required_occluder_ids != tuple(box.box_id for box in self.occluders)
            or self.subject.box_id not in self.required_occluder_ids
            or self.moving_subject_id not in self.required_occluder_ids
            or type(self.policy) is not FixedCardinalVisibilityPolicyV3
        ):
            raise ValueError(
                "continuous visibility bridge input must be complete and exact"
            )


@dataclass(frozen=True, slots=True)
class UprightSE2ContinuousEvaluationInputs:
    """The compiler-owned source/policy bridge for one lifted SE(2) cell."""

    solve_request_sha256: str
    semantic_closure_sha256: str
    policy_bundle_sha256: str
    continuous_upright_se2_compilation_sha256: str
    compiled_cell_sha256: str
    cell: ClosedXYCellV3
    subject_boxes: tuple[FixedCardinalBoxV3, ...]
    obstacle_boxes: tuple[FixedCardinalBoxV3, ...]
    support_assignment: ObjectSupportAssignment
    support_surface_fact: SupportSurfaceFact
    support_surface: SupportSurfaceV3
    target_before_relation: PredicateAtom
    target_after_relation: PredicateAtom
    preservation_invariants: tuple[PreservationInvariant, ...]
    relation: str
    reference_box: FixedCardinalBoxV3
    near_far_threshold: Fraction
    cell_policy: FixedCardinalCellPolicyV3
    subject_pivot_xy: tuple[Fraction, Fraction]
    objective_subject_pivot_xy: tuple[Fraction, Fraction]
    visibility_inputs: tuple[UprightSE2ContinuousVisibilityEvaluationInput, ...]
    resource_atomic_step_limit: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str or not value
                for value in (
                    self.solve_request_sha256,
                    self.semantic_closure_sha256,
                    self.policy_bundle_sha256,
                    self.continuous_upright_se2_compilation_sha256,
                    self.compiled_cell_sha256,
                )
            )
            or type(self.cell) is not ClosedXYCellV3
            or type(self.subject_boxes) is not tuple
            or not self.subject_boxes
            or any(type(box) is not FixedCardinalBoxV3 for box in self.subject_boxes)
            or tuple(box.box_id for box in self.subject_boxes)
            != tuple(sorted(box.box_id for box in self.subject_boxes))
            or type(self.obstacle_boxes) is not tuple
            or any(type(box) is not FixedCardinalBoxV3 for box in self.obstacle_boxes)
            or tuple(box.box_id for box in self.obstacle_boxes)
            != tuple(sorted(box.box_id for box in self.obstacle_boxes))
            or type(self.support_assignment) is not ObjectSupportAssignment
            or type(self.support_surface_fact) is not SupportSurfaceFact
            or type(self.support_surface) is not SupportSurfaceV3
            or type(self.target_before_relation) is not PredicateAtom
            or type(self.target_after_relation) is not PredicateAtom
            or type(self.preservation_invariants) is not tuple
            or not self.preservation_invariants
            or any(
                type(item) is not PreservationInvariant
                for item in self.preservation_invariants
            )
            or self.relation not in ("LEFT", "RIGHT", "FRONT", "BEHIND", "NEAR", "FAR")
            or type(self.reference_box) is not FixedCardinalBoxV3
            or type(self.near_far_threshold) is not Fraction
            or self.near_far_threshold < 0
            or type(self.cell_policy) is not FixedCardinalCellPolicyV3
            or self.cell_policy.visibility_cell != self.cell.canonical_bounds
            or self.cell_policy.relation_symbol != self.relation
            or self.cell_policy.relation_threshold != self.near_far_threshold
            or type(self.subject_pivot_xy) is not tuple
            or len(self.subject_pivot_xy) != 2
            or any(type(value) is not Fraction for value in self.subject_pivot_xy)
            or type(self.objective_subject_pivot_xy) is not tuple
            or len(self.objective_subject_pivot_xy) != 2
            or any(
                type(value) is not Fraction for value in self.objective_subject_pivot_xy
            )
            or type(self.visibility_inputs) is not tuple
            or len(self.visibility_inputs) != 1
            or any(
                type(item) is not UprightSE2ContinuousVisibilityEvaluationInput
                for item in self.visibility_inputs
            )
            or type(self.resource_atomic_step_limit) is not int
            or self.resource_atomic_step_limit <= 0
            or self.cell_policy.atomic_step_limit != self.resource_atomic_step_limit
            or any(
                item.policy.atomic_step_limit != self.resource_atomic_step_limit
                for item in self.visibility_inputs
            )
        ):
            raise ValueError(
                "continuous evaluation bridge inputs must be complete and exact"
            )


# Preserve supported public type/function and pickle lookup.
UprightSE2CardinalVisibilityEvaluationInput.__module__ = "spatialcf.core.upright_se2_compiler"
UprightSE2CardinalEvaluationInputs.__module__ = "spatialcf.core.upright_se2_compiler"
UprightSE2ContinuousVisibilityEvaluationInput.__module__ = "spatialcf.core.upright_se2_compiler"
UprightSE2ContinuousEvaluationInputs.__module__ = "spatialcf.core.upright_se2_compiler"
