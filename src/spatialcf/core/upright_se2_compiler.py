"""Pure exact-cardinal compilation for the Upright SE(2) M3 profile.

This module validates the closed request/profile subset into a domain-level
compilation.  It deliberately does not select an endpoint, execute a backend,
or invoke a verifier.  A separate pure materializer derives an explicit
endpoint only after a caller supplies an authorized world-XY value.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels.projected_visibility import (
    FixedCardinalProjectionBoxV3,
    FixedCardinalVisibilityPolicyV3,
)
from spatialcf.core._internal.kernels.so2 import (
    ContinuousYawIntervalKindV4,
    SO2AtomicBudgetV2,
    compile_continuous_yaw_lift_v4,
)
from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    FixedCardinalBoxV3,
    FixedCardinalCellPolicyV3,
    FixedCardinalObjectiveTermV3,
    SupportSurfaceV3,
)
from spatialcf.core.problem import UprightCameraContextV2_9
from spatialcf.domain import upright_se2 as upright
from spatialcf.domain.base import (
    CanonicalModel,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    Quaternion,
    RigidTransformV2,
    UncertaintyBudgetV2,
    Vec2,
    Vec3,
)
from spatialcf.domain.compatibility import PlanarTranslateCompilation
from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    EditProgram,
    ExtensionFact,
    ExtensionFactBundle,
    SceneStateEnvelope,
)
from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalDefinitionEnvelope,
    CanonicalIdValue,
    DefinitionBundle,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    IntegerValue,
    IntervalValue,
    NamedTypedValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueKind,
)
from spatialcf.domain.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
    StateDeltaManifest,
    StateLeafIndex,
    StateVariableRef,
    TypedVariableBound,
)
from spatialcf.domain.outcomes import (
    BackendSelectionRecord,
    CapabilityMismatch,
    ResourceUsage,
    TypedCompilationOutcome,
    _ResourceUsageEntry,
)
from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    GroundedObligation,
    GroundedObligationSet,
    ObservationObligation,
    PredicateAtom,
    PreservationInvariant,
)
from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationOwnerBinding,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    ProofPolicy,
    ResourceLimit,
    ResourcePolicy,
    SemanticsProfile,
    SolverBackendDescriptor,
)
from spatialcf.domain.scene import (
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
    CanonicalScene,
    ObjectSupportAssignment,
    PinholeCamera,
    SupportSurfaceFact,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = (
    "UprightSE2CardinalEvaluationInputs",
    "UprightSE2CardinalVisibilityEvaluationInput",
    "UprightSE2ContinuousEvaluationInputs",
    "UprightSE2ContinuousVisibilityEvaluationInput",
    "build_upright_se2_cardinal_evaluation_inputs",
    "build_upright_se2_continuous_evaluation_inputs",
    "build_upright_se2_retained_owner_evaluation",
    "cardinal_inverse_quarter_turns",
    "compile_planar_translate_m2_q0_equivalence",
    "compile_upright_se2",
    "compile_upright_se2_continuous",
    "materialize_upright_se2_continuous_endpoint",
    "materialize_upright_se2_endpoint",
    "rotate_cardinal_xy",
)


_INPUT_FAMILY_REF = "definition:spatialcf/upright-se2/compiler-input/1.0"
_INPUT_SCHEMA_REF = "schema:spatialcf/upright-se2/compiler-input/1.0"
_STATE_FAMILY_REF = "definition:spatialcf/upright-se2/state/1.0"
_STATE_SCHEMA_REF = "schema:spatialcf/upright-se2/state/1.0"
_SCENE_SCHEMA_REF = "schema:spatialcf/canonical-scene/2.3"
_BACKEND_REF = "backend:spatialcf/upright-se2/cardinal"
_CONTINUOUS_BACKEND_REF = "backend:spatialcf/upright-se2/continuous"
_BACKEND_BUILD_SHA256 = "b" * 64
_CHECKER_BUILD_SHA256 = "a" * 64
_DEPENDENCY_LOCK_SHA256 = "d" * 64
_DEFINITION_CLOSURE_REF = "definition:spatialcf/upright-se2/definition-closure/1.0"
_SOLVE_POLICY_REF = "definition:spatialcf/upright-se2/solve-policy/1.0"
_DEFINITION_KIND_REF = "definition:spatialcf/upright-se2/definition-kind/1.0"
_DERIVED_RULE_REF = "definition:spatialcf/upright-se2/derived-pose-and-facts/1.0"
_CONTINUOUS_UNAVAILABLE_REF = (
    "definition:spatialcf/upright-se2/continuous-capability-unavailable/1.0"
)
_COMPILER_INPUT_FACT_KEY = "fact-key:spatialcf/upright-se2/compiler-input"
_PIVOT_STATE_HASH_DOMAIN = "spatialcf/counterfactual/upright-se2/pivot-state/3.0"
_POSE_STATE_HASH_DOMAIN = "spatialcf/counterfactual/upright-se2/base-pose/3.0"
_UNCHANGED_LEAVES_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/unchanged-leaves/3.0"
)
_RETAINED_OWNER_BOUND_SCHEMA_REF = upright.UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF

_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/finite-real/1.0"
_INTEGER_SCHEMA_REF = "schema:spatialcf/upright-se2/integer/1.0"
_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/canonical-id/1.0"
_DIGEST_SCHEMA_REF = "schema:spatialcf/upright-se2/digest/1.0"
_ENUM_SCHEMA_REF = "schema:spatialcf/upright-se2/enum-symbol/1.0"
_YAW_ARGUMENT_SCHEMA_REF = "schema:spatialcf/upright-se2/yaw-argument/1.0"
_DEFINITION_CLOSURE_SCHEMA_REF = "schema:spatialcf/upright-se2/definition-closure/1.0"
_SOLVE_POLICY_CLOSURE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/solve-policy-closure/1.0"
)

_CARDINAL_TURN_FRACTIONS = {
    0: Fraction(0),
    1: Fraction(1, 4),
    2: Fraction(-1, 2),
    3: Fraction(-1, 4),
}
_PRIMARY_ROLES = (
    "subject-world-x",
    "subject-world-y",
    "subject-explicit-yaw",
)
_DERIVED_ROLES = (
    "subject-derived-canonical-pose",
    "subject-derived-collision",
    "subject-derived-support",
    "subject-derived-relation",
    "subject-derived-visibility",
)
_SOURCE_FIELDS = (
    ("objects", "object_id"),
    ("geometry-instances", "geometry_id"),
    ("collision-bodies", "body_id"),
    ("workspace-boundaries", "fact_id"),
    ("known-free-spaces", "fact_id"),
    ("support-surfaces", "surface_id"),
    ("cameras", "camera_id"),
    ("baseline-observations", "observation_id"),
)
_WORLD_XY_FRAME_REF = "definition:spatialcf/upright-se2/world-xy/1.0"
_METRE_UNIT_REF = "definition:spatialcf/upright-se2/metre/1.0"
_CLOSED_INTERVAL_TOPOLOGY_REF = "definition:spatialcf/upright-se2/closed-interval/1.0"


@dataclass(frozen=True)
class _CompilerInput:
    """The fully typed, profile-owned compiler-input fact payload."""

    operation_kind: str
    operator_ref: str
    subject_id: str
    reference_id: str
    subject_yaw_turns: float
    yaw_argument: upright.CardinalYaw | upright.ContinuousYawDomain


@dataclass(frozen=True)
class _SceneAuthority:
    """Named, scene-owned object pivots and complete source state."""

    scene: CanonicalScene
    subject: object
    reference: object


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


def build_upright_se2_retained_owner_evaluation(
    *,
    compiled_cell: upright.UprightSE2CompiledCell,
    owner_ref: str,
    evaluator_capability_ref: str,
    outcome_kind: upright.UprightSE2RetainedOwnerOutcomeKind,
    raw_proof_rows: tuple[str, ...],
    raw_findings: tuple[str, ...],
    atomic_steps: int,
    resource_delta: ResourceUsage,
    label: str,
    exact_bound_value: object | None = None,
    additional_exact_bounds: tuple[TypedValue, ...] = (),
    continuous_exact_rationals: bool = False,
) -> upright.UprightSE2RetainedOwnerEvaluation:
    """Serialize one already-produced retained-owner DTO into proof rows.

    This intentionally performs no scene reconstruction, kernel invocation,
    search, checker dispatch, certificate work, or terminal assembly.  Both
    proposal and fresh-checker owners feed their retained DTO fields through
    this one canonical proof-row transport seam.
    """

    if (
        type(compiled_cell) is not upright.UprightSE2CompiledCell
        or type(owner_ref) is not str
        or not owner_ref
        or type(evaluator_capability_ref) is not str
        or not evaluator_capability_ref
        or type(outcome_kind) is not upright.UprightSE2RetainedOwnerOutcomeKind
        or type(raw_proof_rows) is not tuple
        or any(type(row) is not str for row in raw_proof_rows)
        or type(raw_findings) is not tuple
        or any(type(finding) is not str for finding in raw_findings)
        or type(atomic_steps) is not int
        or atomic_steps < 0
        or type(resource_delta) is not ResourceUsage
        or type(label) is not str
        or not label
        or type(additional_exact_bounds) is not tuple
        or any(type(value) is not TypedValue for value in additional_exact_bounds)
        or type(continuous_exact_rationals) is not bool
    ):
        raise TypeError("retained owner proof transport inputs must be exact")
    if outcome_kind is upright.UprightSE2RetainedOwnerOutcomeKind.EXACT:
        if exact_bound_value is None:
            raise ValueError("exact retained owner outcome requires bounds")
        bound_node = (
            _typed_continuous_bound_node
            if continuous_exact_rationals
            else _typed_bound_node
        )
        exact_bounds = tuple(
            sorted(
                (bound_node(exact_bound_value), *additional_exact_bounds),
                key=canonical_json_bytes,
            )
        )
        finding_codes: tuple[str, ...] = ()
    else:
        exact_bounds = ()
        digest = canonical_sha256(
            (label, outcome_kind.value, raw_findings),
            domain="spatialcf/counterfactual/upright-se2/proof-finding/3.0",
        )
        finding_codes = (
            "finding:spatialcf/upright-se2/proof-transport/"
            + f"{outcome_kind.value.lower()}/{digest}",
        )
    proof_rows = tuple(
        sorted(
            {
                _canonical_proof_row(label, row)
                for row in (*raw_proof_rows, f"proof:spatialcf/upright-se2/{label}")
            },
            key=canonical_json_bytes,
        )
    )
    return upright.UprightSE2RetainedOwnerEvaluation.seal(
        compiled_cell=compiled_cell,
        owner_ref=owner_ref,
        evaluator_capability_ref=evaluator_capability_ref,
        outcome_kind=outcome_kind,
        exact_bounds=exact_bounds,
        finding_codes=finding_codes,
        proof_rows=proof_rows,
        resource_delta=resource_delta,
    )


def _typed_bound_node(value: object) -> TypedValue:
    """Encode retained-owner bounds without interpreting their geometry."""

    return _typed_bound_node_with_rational_mode(value, structural_rationals=False)


def _typed_continuous_bound_node(value: object) -> TypedValue:
    """Encode continuous retained bounds without length-limiting exact rationals."""

    return _typed_bound_node_with_rational_mode(value, structural_rationals=True)


def _typed_bound_node_with_rational_mode(
    value: object,
    *,
    structural_rationals: bool,
) -> TypedValue:
    """Transport retained DTOs while preserving the cardinal scalar wire by default."""

    if isinstance(value, StrEnum):
        payload = CanonicalIdValue(value=value.value)
    elif type(value) is Fraction:
        fraction_id = f"exact-rational:{value.numerator}/{value.denominator}"
        if structural_rationals and len(fraction_id) > 512:
            payload = RecordValue(
                fields=tuple(
                    sorted(
                        (
                            NamedTypedValue(
                                name="denominator",
                                value=TypedValue(
                                    value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
                                    payload=IntegerValue(value=value.denominator),
                                ),
                            ),
                            NamedTypedValue(
                                name="numerator",
                                value=TypedValue(
                                    value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
                                    payload=IntegerValue(value=value.numerator),
                                ),
                            ),
                        ),
                        key=lambda field: canonical_json_bytes(field.name),
                    )
                )
            )
        else:
            payload = CanonicalIdValue(value=fraction_id)
    elif type(value) is bool:
        payload = BooleanValue(value=value)
    elif type(value) is int:
        payload = IntegerValue(value=value)
    elif type(value) is float:
        payload = FiniteRealValue(value=0.0 if value == 0.0 else value)
    elif isinstance(value, str):
        payload = CanonicalIdValue(value=value)
    elif value is None:
        payload = EnumSymbolValue(symbol="NULL")
    elif type(value) in (tuple, list):
        payload = FiniteOrderedTupleValue(
            element_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
            items=tuple(
                _typed_bound_node_with_rational_mode(
                    item, structural_rationals=structural_rationals
                )
                for item in value
            ),
        )
    elif type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError("retained bound records require string field names")
        payload = RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=name,
                            value=_typed_bound_node_with_rational_mode(
                                item, structural_rationals=structural_rationals
                            ),
                        )
                        for name, item in value.items()
                    ),
                    key=canonical_json_bytes,
                )
            )
        )
    elif isinstance(value, CanonicalModel):
        return _typed_bound_node_with_rational_mode(
            value.model_dump(mode="python", round_trip=True),
            structural_rationals=structural_rationals,
        )
    elif is_dataclass(value):
        payload = RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=field.name,
                            value=_typed_bound_node_with_rational_mode(
                                getattr(value, field.name),
                                structural_rationals=structural_rationals,
                            ),
                        )
                        for field in fields(value)
                    ),
                    key=canonical_json_bytes,
                )
            )
        )
    else:
        raise TypeError(f"unsupported retained bound value {type(value).__name__}")
    return TypedValue(
        value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF, payload=payload
    )


def _canonical_proof_row(label: str, raw: str) -> str:
    if raw and not any(character.isspace() for character in raw):
        return raw
    digest = canonical_sha256(
        (label, raw),
        domain="spatialcf/counterfactual/upright-se2/proof-row/3.0",
    )
    return f"proof:spatialcf/upright-se2/{label}/{digest}"


def compile_upright_se2(
    solve_request: CounterfactualSolveRequest,
) -> (
    upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation
    | TypedCompilationOutcome
):
    """Compile one closed M3 request without solving or checking it.

    Cardinal compilation preserves its retained wire exactly.  A registered
    continuous request dispatches to the additive continuous compiler sibling;
    malformed or incompatible wires raise before an outcome can be built.
    """

    _require_exact_round_trip(
        solve_request, CounterfactualSolveRequest, "solve request"
    )
    registration = _registered_profile()
    compiler_input, authority, executable_policy_bundle = (
        _validate_problem_and_extract_input(
            solve_request,
            registration,
        )
    )
    _validate_operational_closure(
        solve_request,
        registration,
        operation_kind=compiler_input.operation_kind,
    )
    translation_domain = _validate_intervention_authorization(
        solve_request,
        compiler_input,
        authority,
    )
    base_yaw = _validate_explicit_pose_yaw(compiler_input, solve_request, authority)
    pivot_binding = _resolve_pivot_binding(compiler_input, solve_request, authority)

    if compiler_input.operation_kind == "CONTINUOUS":
        return _compile_continuous(
            solve_request,
            registration=registration,
            compiler_input=compiler_input,
            authority=authority,
            executable_policy_bundle=executable_policy_bundle,
            translation_domain=translation_domain,
            base_yaw=base_yaw,
            pivot_binding=pivot_binding,
        )

    if type(compiler_input.yaw_argument) is not upright.CardinalYaw:
        raise ValueError("cardinal request must carry a cardinal yaw argument")
    authorization = upright.CardinalYawAuthorization.seal(
        subject_id=compiler_input.subject_id,
        operator_ref=compiler_input.operator_ref,
        pivot_binding=pivot_binding,
        yaw=compiler_input.yaw_argument,
    )
    return _compile_cardinal(
        solve_request=solve_request,
        registration=registration,
        compiler_input=compiler_input,
        authority=authority,
        base_yaw=base_yaw,
        translation_domain=translation_domain,
        authorization=authorization,
        executable_policy_bundle=executable_policy_bundle,
    )


def build_upright_se2_cardinal_evaluation_inputs(
    compilation: upright.UprightSE2Compilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> UprightSE2CardinalEvaluationInputs:
    """Construct one sealed cell's retained-owner inputs without evaluating it.

    This is intentionally the sole public scene/policy-to-kernel bridge for
    the cardinal M3 route.  It replays the compiler's source closure before
    translating its exact scene facts to the retained DTOs, but it does not
    create a resource ledger, call a retained owner, or select a point.
    """

    _require_exact_round_trip(
        compilation,
        upright.UprightSE2Compilation,
        "upright se2 compilation",
    )
    _require_exact_round_trip(
        compiled_cell,
        upright.UprightSE2CompiledCell,
        "compiled cell",
    )
    member = _bridge_compiled_cell_member(compilation, compiled_cell)
    replayed = compile_upright_se2(compilation.source_solve_request)
    if type(replayed) is not upright.UprightSE2Compilation:
        raise ValueError("cardinal bridge source did not replay to a compilation")
    _bridge_require_replayed_compilation(compilation, replayed)

    registration = _registered_profile()
    compiler_input, authority, policy_bundle = _validate_problem_and_extract_input(
        compilation.source_solve_request,
        registration,
    )
    _validate_operational_closure(compilation.source_solve_request, registration)
    translation_domain = _validate_intervention_authorization(
        compilation.source_solve_request,
        compiler_input,
        authority,
    )
    _validate_explicit_pose_yaw(
        compiler_input,
        compilation.source_solve_request,
        authority,
    )
    pivot_binding = _resolve_pivot_binding(
        compiler_input,
        compilation.source_solve_request,
        authority,
    )
    if (
        compiler_input.operation_kind != "CARDINAL"
        or type(compiler_input.yaw_argument) is not upright.CardinalYaw
    ):
        raise ValueError("cardinal bridge requires one registered cardinal source")
    authorization = upright.CardinalYawAuthorization.seal(
        subject_id=compiler_input.subject_id,
        operator_ref=compiler_input.operator_ref,
        pivot_binding=pivot_binding,
        yaw=compiler_input.yaw_argument,
    )
    if (
        canonical_json_bytes(compilation.operation.authorization)
        != canonical_json_bytes(authorization)
        or compilation.operation.translation_domain != translation_domain
        or compilation.operation.pivot_binding != pivot_binding
    ):
        raise ValueError("cardinal bridge compilation operation does not close")
    _bridge_validate_cell_yaw(member, compilation.operation.quarter_turns_ccw)

    scene = authority.scene
    camera = _bridge_exact_identity_camera(scene)
    objects = _bridge_object_map(scene)
    subject = objects.get(compiler_input.subject_id)
    reference = objects.get(compiler_input.reference_id)
    if subject is None or reference is None:
        raise ValueError("cardinal bridge scene authority is incomplete")
    subject_boxes, obstacle_boxes = _bridge_collision_boxes(
        scene,
        subject_id=compiler_input.subject_id,
        support_surface_id=subject.support_assignment.surface_id,
        objects=objects,
    )
    _bridge_validate_subject_role_geometry_closure(
        scene,
        subject_id=compiler_input.subject_id,
        subject_boxes=subject_boxes,
        objects=objects,
    )
    support_surface_fact = upright.validate_required_upright_support_surface(
        scene,
        compiler_input.subject_id,
    )
    support_surface = _bridge_support_surface(support_surface_fact, objects)
    target_before_relation, target_after_relation, relation = (
        _bridge_target_and_preservation_rows(
            compilation.source_solve_request.semantic_problem,
            subject_id=compiler_input.subject_id,
            reference_id=compiler_input.reference_id,
        )
    )
    reference_box = _bridge_reference_relation_box(
        scene,
        reference_id=compiler_input.reference_id,
        objects=objects,
    )
    cell = ClosedXYCellV3(
        member.x_lower.as_fraction,
        member.x_upper.as_fraction,
        member.y_lower.as_fraction,
        member.y_upper.as_fraction,
    )
    resource_cap = _bridge_resource_cap(policy_bundle)
    cell_policy = _bridge_cell_policy(
        policy_bundle,
        relation=relation,
        cell=cell,
        resource_cap=resource_cap,
    )
    subject_pivot_xy = _bridge_object_pivot_xy(
        objects,
        compilation.operation.pivot_binding.pivot_entity_id,
    )
    objective_subject_pivot_xy = _bridge_object_pivot_xy(
        objects,
        compiler_input.subject_id,
    )
    visibility_inputs = _bridge_visibility_inputs(
        scene,
        camera=camera,
        objects=objects,
        subject_id=compiler_input.subject_id,
        quarter_turns_ccw=compilation.operation.quarter_turns_ccw,
        pivot_xy=subject_pivot_xy,
        cell=cell,
        policy_bundle=policy_bundle,
        resource_cap=resource_cap,
    )
    return UprightSE2CardinalEvaluationInputs(
        solve_request_sha256=compilation.solve_request_sha256,
        semantic_closure_sha256=compilation.semantic_closure.semantic_closure_sha256,
        policy_bundle_sha256=compilation.semantic_closure.policy_bundle_sha256,
        upright_se2_compilation_sha256=compilation.upright_se2_compilation_sha256,
        compiled_cell_sha256=member.compiled_cell_sha256,
        cell=cell,
        quarter_turns_ccw=compilation.operation.quarter_turns_ccw,
        subject_boxes=subject_boxes,
        obstacle_boxes=obstacle_boxes,
        support_assignment=subject.support_assignment,
        support_surface_fact=support_surface_fact,
        support_surface=support_surface,
        target_before_relation=target_before_relation,
        target_after_relation=target_after_relation,
        preservation_invariants=(
            compilation.source_solve_request.semantic_problem.preservation_invariants
        ),
        relation=relation,
        reference_box=reference_box,
        near_far_threshold=cell_policy.relation_threshold,
        cell_policy=cell_policy,
        subject_pivot_xy=subject_pivot_xy,
        objective_subject_pivot_xy=objective_subject_pivot_xy,
        visibility_inputs=visibility_inputs,
        resource_atomic_step_limit=resource_cap,
    )


def build_upright_se2_continuous_evaluation_inputs(
    compilation: upright.UprightSE2ContinuousCompilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> UprightSE2ContinuousEvaluationInputs:
    """Rebuild one authorized lifted cell's V4 owner inputs without evaluation.

    The bridge is the only continuous route from immutable source facts and
    request-bound policy to retained-kernel DTOs.  It deliberately performs no
    yaw enclosure, geometry evaluation, proposal search, proof checking, or
    endpoint materialization.
    """

    _require_exact_round_trip(
        compilation,
        upright.UprightSE2ContinuousCompilation,
        "continuous upright se2 compilation",
    )
    _require_exact_round_trip(
        compiled_cell,
        upright.UprightSE2CompiledCell,
        "continuous compiled cell",
    )
    member = _bridge_continuous_compiled_cell_member(compilation, compiled_cell)
    replayed = compile_upright_se2(compilation.source_solve_request)
    if type(replayed) is not upright.UprightSE2ContinuousCompilation:
        raise ValueError("continuous bridge source did not replay to a compilation")
    _bridge_require_replayed_continuous_compilation(compilation, replayed)

    registration = _registered_profile()
    compiler_input, authority, policy_bundle = _validate_problem_and_extract_input(
        compilation.source_solve_request,
        registration,
    )
    _validate_operational_closure(
        compilation.source_solve_request,
        registration,
        operation_kind="CONTINUOUS",
    )
    translation_domain = _validate_intervention_authorization(
        compilation.source_solve_request,
        compiler_input,
        authority,
    )
    _validate_explicit_pose_yaw(
        compiler_input,
        compilation.source_solve_request,
        authority,
    )
    pivot_binding = _resolve_pivot_binding(
        compiler_input,
        compilation.source_solve_request,
        authority,
    )
    if not isinstance(
        compiler_input.yaw_argument,
        (upright.ContinuousYawArc, upright.ContinuousYawFullCircle),
    ):
        raise ValueError(  # noqa: TRY004 - preserves the public invalid-domain contract.
            "continuous bridge requires one registered source yaw domain"
        )
    authorization = upright.ContinuousYawAuthorization.seal(
        subject_id=compiler_input.subject_id,
        operator_ref=compiler_input.operator_ref,
        pivot_binding=pivot_binding,
        yaw_domain=compiler_input.yaw_argument,
    )
    if (
        canonical_json_bytes(compilation.operation.authorization)
        != canonical_json_bytes(authorization)
        or compilation.operation.translation_domain != translation_domain
        or compilation.operation.pivot_binding != pivot_binding
    ):
        raise ValueError("continuous bridge compilation operation does not close")

    scene = authority.scene
    camera = _bridge_exact_identity_camera(scene)
    objects = _bridge_object_map(scene)
    subject = objects.get(compiler_input.subject_id)
    reference = objects.get(compiler_input.reference_id)
    if subject is None or reference is None:
        raise ValueError("continuous bridge scene authority is incomplete")
    subject_boxes, obstacle_boxes = _bridge_collision_boxes(
        scene,
        subject_id=compiler_input.subject_id,
        support_surface_id=subject.support_assignment.surface_id,
        objects=objects,
    )
    _bridge_validate_subject_role_geometry_closure(
        scene,
        subject_id=compiler_input.subject_id,
        subject_boxes=subject_boxes,
        objects=objects,
    )
    support_surface_fact = upright.validate_required_upright_support_surface(
        scene,
        compiler_input.subject_id,
    )
    support_surface = _bridge_support_surface(support_surface_fact, objects)
    target_before_relation, target_after_relation, relation = (
        _bridge_target_and_preservation_rows(
            compilation.source_solve_request.semantic_problem,
            subject_id=compiler_input.subject_id,
            reference_id=compiler_input.reference_id,
        )
    )
    reference_box = _bridge_reference_relation_box(
        scene,
        reference_id=compiler_input.reference_id,
        objects=objects,
    )
    cell = ClosedXYCellV3(
        member.x_lower.as_fraction,
        member.x_upper.as_fraction,
        member.y_lower.as_fraction,
        member.y_upper.as_fraction,
    )
    resource_cap = _bridge_resource_cap(policy_bundle)
    cell_policy = _bridge_cell_policy(
        policy_bundle,
        relation=relation,
        cell=cell,
        resource_cap=resource_cap,
    )
    subject_pivot_xy = _bridge_object_pivot_xy(
        objects,
        compilation.operation.pivot_binding.pivot_entity_id,
    )
    objective_subject_pivot_xy = _bridge_object_pivot_xy(
        objects,
        compiler_input.subject_id,
    )
    visibility_inputs = _bridge_continuous_visibility_inputs(
        scene,
        camera=camera,
        objects=objects,
        subject_id=compiler_input.subject_id,
        visibility_subject_box_id=subject_boxes[0].box_id,
        policy_bundle=policy_bundle,
        resource_cap=resource_cap,
    )
    return UprightSE2ContinuousEvaluationInputs(
        solve_request_sha256=compilation.solve_request_sha256,
        semantic_closure_sha256=compilation.semantic_closure.semantic_closure_sha256,
        policy_bundle_sha256=compilation.semantic_closure.policy_bundle_sha256,
        continuous_upright_se2_compilation_sha256=(
            compilation.continuous_upright_se2_compilation_sha256
        ),
        compiled_cell_sha256=member.compiled_cell_sha256,
        cell=cell,
        subject_boxes=subject_boxes,
        obstacle_boxes=obstacle_boxes,
        support_assignment=subject.support_assignment,
        support_surface_fact=support_surface_fact,
        support_surface=support_surface,
        target_before_relation=target_before_relation,
        target_after_relation=target_after_relation,
        preservation_invariants=(
            compilation.source_solve_request.semantic_problem.preservation_invariants
        ),
        relation=relation,
        reference_box=reference_box,
        near_far_threshold=cell_policy.relation_threshold,
        cell_policy=cell_policy,
        subject_pivot_xy=subject_pivot_xy,
        objective_subject_pivot_xy=objective_subject_pivot_xy,
        visibility_inputs=visibility_inputs,
        resource_atomic_step_limit=resource_cap,
    )


def _bridge_compiled_cell_member(
    compilation: upright.UprightSE2Compilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Return one authorized root or its proper exact-dyadic descendant.

    The compilation roster is the sole request-authorized root set.  A backend
    may name deterministic refinement cells, but cannot alter a root's
    authorization, cardinal yaw, domain, or identity by doing so.
    """

    matching = tuple(
        cell
        for cell in compilation.compiled_cells
        if canonical_json_bytes(cell) == canonical_json_bytes(compiled_cell)
    )
    if len(matching) == 1:
        return matching[0]
    if matching:
        raise ValueError("compiled cell matches more than one compilation root")

    roots = tuple(
        root
        for root in compilation.compiled_cells
        if _bridge_is_proper_exact_dyadic_descendant(compiled_cell, root)
    )
    if len(roots) != 1:
        raise ValueError(
            "compiled cell must be one exact compilation root or its unique exact-dyadic descendant"
        )
    return compiled_cell


def _bridge_continuous_compiled_cell_member(
    compilation: upright.UprightSE2ContinuousCompilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Accept only the sole root or one exact lifted `(x,y,u)` descendant."""

    root = compilation.compiled_cells[0]
    if canonical_json_bytes(root) == canonical_json_bytes(compiled_cell):
        return root
    if not _bridge_is_proper_exact_dyadic_continuous_descendant(compiled_cell, root):
        raise ValueError(
            "continuous compiled cell must be its root or a unique exact-dyadic lifted descendant"
        )
    return compiled_cell


def _bridge_is_proper_exact_dyadic_continuous_descendant(
    cell: upright.UprightSE2CompiledCell,
    root: upright.UprightSE2CompiledCell,
) -> bool:
    """Require a strict closed descendant without changing root authorization."""

    if (
        not cell.cell_id.startswith(f"{root.cell_id}/")
        or cell.authorization_sha256 != root.authorization_sha256
        or cell.x_lower.as_fraction < root.x_lower.as_fraction
        or cell.x_upper.as_fraction > root.x_upper.as_fraction
        or cell.y_lower.as_fraction < root.y_lower.as_fraction
        or cell.y_upper.as_fraction > root.y_upper.as_fraction
        or cell.yaw_interval.lower.as_fraction < root.yaw_interval.lower.as_fraction
        or cell.yaw_interval.upper.as_fraction > root.yaw_interval.upper.as_fraction
    ):
        return False
    return (
        cell.x_lower != root.x_lower
        or cell.x_upper != root.x_upper
        or cell.y_lower != root.y_lower
        or cell.y_upper != root.y_upper
        or cell.yaw_interval.lower != root.yaw_interval.lower
        or cell.yaw_interval.upper != root.yaw_interval.upper
    )


def _bridge_is_proper_exact_dyadic_descendant(
    cell: upright.UprightSE2CompiledCell,
    root: upright.UprightSE2CompiledCell,
) -> bool:
    """Require a strict, yaw-bound XY refinement of exactly one root."""

    if (
        not cell.cell_id.startswith(f"{root.cell_id}/")
        or cell.authorization_sha256 != root.authorization_sha256
        or cell.yaw_interval != root.yaw_interval
        or cell.x_lower.as_fraction < root.x_lower.as_fraction
        or cell.x_upper.as_fraction > root.x_upper.as_fraction
        or cell.y_lower.as_fraction < root.y_lower.as_fraction
        or cell.y_upper.as_fraction > root.y_upper.as_fraction
    ):
        return False
    return (
        cell.x_lower != root.x_lower
        or cell.x_upper != root.x_upper
        or cell.y_lower != root.y_lower
        or cell.y_upper != root.y_upper
    )


def _bridge_require_replayed_compilation(
    compilation: upright.UprightSE2Compilation,
    replayed: upright.UprightSE2Compilation,
) -> None:
    """Require source replay to reproduce every normal compiler-owned root.

    A retained M2 q=0 construction adds one provenance record after normal M3
    compilation, so that optional construction root is deliberately checked by
    the source model itself rather than compared to a direct replay.
    """

    for field_name in (
        "solve_request_sha256",
        "source_solve_request",
        "closure",
        "operation",
        "endpoint_construction_recipe",
        "state_footprint",
        "grounded_obligations",
        "semantic_closure",
        "compiled_cells",
    ):
        if canonical_json_bytes(
            getattr(compilation, field_name)
        ) != canonical_json_bytes(getattr(replayed, field_name)):
            raise ValueError(
                "cardinal bridge compilation root or policy closure does not replay"
            )


def _bridge_require_replayed_continuous_compilation(
    compilation: upright.UprightSE2ContinuousCompilation,
    replayed: upright.UprightSE2ContinuousCompilation,
) -> None:
    """Require source replay to reproduce the additive continuous root exactly."""

    for field_name in (
        "solve_request_sha256",
        "source_solve_request",
        "closure",
        "operation",
        "endpoint_construction_recipe",
        "state_footprint",
        "grounded_obligations",
        "semantic_closure",
        "compiled_cells",
        "continuous_yaw_lift",
    ):
        if canonical_json_bytes(
            getattr(compilation, field_name)
        ) != canonical_json_bytes(getattr(replayed, field_name)):
            raise ValueError(
                "continuous bridge compilation root or policy closure does not replay"
            )


def _bridge_validate_cell_yaw(
    cell: upright.UprightSE2CompiledCell,
    quarter_turns_ccw: int,
) -> None:
    expected = _CARDINAL_TURN_FRACTIONS.get(quarter_turns_ccw)
    if expected is None or (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        cell.yaw_interval.seam_ownership,
    ) != (expected, expected, "NONE"):
        raise ValueError("compiled cell yaw interval does not bind the operation")


def _bridge_object_map(scene: CanonicalScene) -> dict[str, object]:
    objects = _known_exact_source_values(scene.objects, "objects")
    result = {object_.object_id: object_ for object_ in objects}
    if len(result) != len(objects):
        raise ValueError("cardinal bridge object roster is not unique")
    return result


def _bridge_exact_identity_camera(scene: CanonicalScene) -> PinholeCamera:
    """Return the sole retained-core camera subset whose XY frame is world XY."""

    cameras = _known_exact_source_values(scene.cameras, "cameras")
    if scene.cameras.uncertainty != UncertaintyBudgetV2() or len(cameras) != 1:
        raise ValueError("cardinal bridge requires one exact fixed camera")
    camera = cameras[0]
    if type(camera) is not PinholeCamera:
        raise ValueError("cardinal bridge camera must use the exact pinhole model")
    if (
        camera.distortion_model is not CameraDistortionModel.NONE
        or camera.brown_conrady_coefficients is not None
    ):
        raise ValueError("cardinal bridge camera distortion must be NONE")
    if camera.calibration_uncertainty != UncertaintyBudgetV2():
        raise ValueError("cardinal bridge camera calibration must be exact")
    if (
        camera.matrix_layout is not CameraMatrixLayout.ROW_MAJOR
        or camera.camera_axes is not CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConvention.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConvention.POSITIVE_Z_FORWARD
    ):
        raise ValueError("cardinal bridge camera convention is unsupported")
    intrinsics = camera.intrinsics_row_major
    if (
        intrinsics[1] != 0.0
        or intrinsics[3] != 0.0
        or intrinsics[6:] != (0.0, 0.0, 1.0)
    ):
        raise ValueError("cardinal bridge camera intrinsics are noncanonical")
    rotation = camera.world_to_camera.rotation
    if type(rotation) is not Quaternion or (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("cardinal bridge camera must use identity orientation")
    return camera


def _bridge_cardinal_yaw(rotation: Quaternion, *, label: str) -> int:
    """Decode only a source yaw already representable by the fixed DTOs."""

    if type(rotation) is not Quaternion or rotation.x != 0.0 or rotation.y != 0.0:
        raise ValueError(f"{label} must be a cardinal upright rotation")
    if rotation.z == 0.0:
        return 0
    if rotation.z == rotation.w:
        return 1
    if rotation.w == 0.0:
        return 2
    if rotation.z == -rotation.w:
        return 3
    raise ValueError(f"{label} is unsupported by the fixed cardinal DTO bridge")


def _bridge_fraction(value: float, *, label: str) -> Fraction:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be one finite exact binary64 source value")
    return Fraction.from_float(value)


def _bridge_world_box_parts(
    geometry: object,
    objects: dict[str, object],
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, int]:
    """Translate one exact upright source box to fixed world-box components."""

    if (
        type(getattr(geometry, "shape", None)) is not UprightBox3DV2
        or getattr(geometry, "approximation", None) is not GeometryApproximationV2.EXACT
    ):
        raise ValueError("geometry is unsupported by the exact fixed-box bridge")
    anchor = geometry.anchor_from_geometry
    anchor_q = _bridge_cardinal_yaw(
        anchor.rotation,
        label=f"geometry {geometry.geometry_id} anchor rotation",
    )
    owner_id = geometry.owner_object_id
    if owner_id is None:
        owner_q = 0
        owner_x = Fraction()
        owner_y = Fraction()
        owner_z = Fraction()
    else:
        owner = objects.get(owner_id)
        if owner is None:
            raise ValueError("geometry owner is absent from the exact object roster")
        owner_pose = owner.pose.world_from_object
        owner_q = _bridge_cardinal_yaw(
            owner_pose.rotation,
            label=f"geometry {geometry.geometry_id} owner rotation",
        )
        owner_x = _bridge_fraction(
            owner_pose.translation.x,
            label=f"geometry {geometry.geometry_id} owner x",
        )
        owner_y = _bridge_fraction(
            owner_pose.translation.y,
            label=f"geometry {geometry.geometry_id} owner y",
        )
        owner_z = _bridge_fraction(
            owner_pose.translation.z,
            label=f"geometry {geometry.geometry_id} owner z",
        )
    anchor_x = _bridge_fraction(
        anchor.translation.x,
        label=f"geometry {geometry.geometry_id} anchor x",
    )
    anchor_y = _bridge_fraction(
        anchor.translation.y,
        label=f"geometry {geometry.geometry_id} anchor y",
    )
    relative_x, relative_y = _rotate_cardinal_xy_components(anchor_x, anchor_y, owner_q)
    total_q = (owner_q + anchor_q) % 4
    half_x = (
        _bridge_fraction(
            geometry.shape.size_m.x,
            label=f"geometry {geometry.geometry_id} size x",
        )
        / 2
    )
    half_y = (
        _bridge_fraction(
            geometry.shape.size_m.y,
            label=f"geometry {geometry.geometry_id} size y",
        )
        / 2
    )
    if total_q % 2:
        half_x, half_y = half_y, half_x
    return (
        owner_x + relative_x,
        owner_y + relative_y,
        owner_z
        + _bridge_fraction(
            anchor.translation.z,
            label=f"geometry {geometry.geometry_id} anchor z",
        ),
        half_x,
        half_y,
        _bridge_fraction(
            geometry.shape.size_m.z,
            label=f"geometry {geometry.geometry_id} size z",
        )
        / 2,
        total_q,
    )


def _bridge_fixed_box(
    geometry: object,
    objects: dict[str, object],
) -> FixedCardinalBoxV3:
    center_x, center_y, center_z, half_x, half_y, half_z, _ = _bridge_world_box_parts(
        geometry,
        objects,
    )
    return FixedCardinalBoxV3(
        box_id=geometry.geometry_id,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
    )


def _bridge_collision_boxes(
    scene: CanonicalScene,
    *,
    subject_id: str,
    support_surface_id: str | None,
    objects: dict[str, object],
) -> tuple[tuple[FixedCardinalBoxV3, ...], tuple[FixedCardinalBoxV3, ...]]:
    if support_surface_id is None:
        raise ValueError("cardinal bridge subject support assignment is incomplete")
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    geometry_by_id = {geometry.geometry_id: geometry for geometry in geometries}
    if len(geometry_by_id) != len(geometries):
        raise ValueError("cardinal bridge geometry roster is not unique")
    bodies = _known_exact_source_values(scene.collision_bodies, "collision bodies")
    support_sources = tuple(
        surface
        for surface in _known_exact_source_values(
            scene.support_surfaces, "support surfaces"
        )
        if surface.surface_id == support_surface_id
    )
    if len(support_sources) != 1:
        raise ValueError("cardinal bridge support surface is absent or duplicated")
    supporting_body_id = support_sources[0].supporting_body_id
    subject_geometry_ids: list[str] = []
    obstacle_geometry_ids: list[str] = []
    referenced_geometry_ids: list[str] = []
    for body in bodies:
        ids = body.geometry_instance_ids
        if not ids:
            raise ValueError("cardinal bridge collision body must name geometry")
        for geometry_id in ids:
            geometry = geometry_by_id.get(geometry_id)
            if geometry is None or geometry.role is not GeometryRoleV2.COLLISION:
                raise ValueError("collision body must bind exact collision geometry")
            if geometry.owner_object_id != body.owner_object_id:
                raise ValueError("collision body and geometry owners must agree")
            referenced_geometry_ids.append(geometry_id)
        if body.owner_object_id == subject_id:
            subject_geometry_ids.extend(ids)
        elif body.body_id != supporting_body_id:
            obstacle_geometry_ids.extend(ids)
    all_collision_ids = tuple(
        geometry.geometry_id
        for geometry in geometries
        if geometry.role is GeometryRoleV2.COLLISION
    )
    if tuple(sorted(referenced_geometry_ids)) != tuple(sorted(all_collision_ids)):
        raise ValueError("collision bodies do not cover the complete collision roster")
    if (
        not subject_geometry_ids
        or len(set(subject_geometry_ids)) != len(subject_geometry_ids)
        or len(set(obstacle_geometry_ids)) != len(obstacle_geometry_ids)
    ):
        raise ValueError("collision body roster is incomplete or duplicates a geometry")
    subject_boxes = tuple(
        sorted(
            (
                _bridge_fixed_box(geometry_by_id[geometry_id], objects)
                for geometry_id in subject_geometry_ids
            ),
            key=lambda box: box.box_id,
        )
    )
    obstacle_boxes = tuple(
        sorted(
            (
                _bridge_fixed_box(geometry_by_id[geometry_id], objects)
                for geometry_id in obstacle_geometry_ids
            ),
            key=lambda box: box.box_id,
        )
    )
    return subject_boxes, obstacle_boxes


def _bridge_subject_role_boxes(
    scene: CanonicalScene,
    *,
    subject_id: str,
    role: GeometryRoleV2,
    objects: dict[str, object],
) -> tuple[FixedCardinalBoxV3, ...]:
    """Translate every exact subject geometry for one semantic role to fixed boxes."""

    geometries = _known_exact_source_values(
        scene.geometry_instances,
        "geometry instances",
    )
    candidates = tuple(
        geometry
        for geometry in geometries
        if geometry.owner_object_id == subject_id and geometry.role is role
    )
    if not candidates:
        raise ValueError(f"subject {role.value.lower()} geometry is absent")
    return tuple(
        sorted(
            (_bridge_fixed_box(geometry, objects) for geometry in candidates),
            key=lambda box: box.box_id,
        )
    )


def _bridge_xy_role_signature(
    boxes: tuple[FixedCardinalBoxV3, ...],
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
    """Return the complete per-box XY inputs the retained relation owner consumes."""

    return tuple(
        sorted((box.center_x, box.center_y, box.half_x, box.half_y) for box in boxes)
    )


def _bridge_support_role_signature(
    boxes: tuple[FixedCardinalBoxV3, ...],
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction], ...]:
    """Return the complete per-box support footprint/contact inputs the owner uses."""

    return tuple(
        sorted(
            (
                box.center_x,
                box.center_y,
                box.center_z,
                box.half_x,
                box.half_y,
                box.half_z,
            )
            for box in boxes
        )
    )


def _bridge_validate_subject_role_geometry_closure(
    scene: CanonicalScene,
    *,
    subject_id: str,
    subject_boxes: tuple[FixedCardinalBoxV3, ...],
    objects: dict[str, object],
) -> None:
    """Fail closed unless role-specific subject geometry preserves owner inputs."""

    relation_boxes = _bridge_subject_role_boxes(
        scene,
        subject_id=subject_id,
        role=GeometryRoleV2.RELATION,
        objects=objects,
    )
    if _bridge_xy_role_signature(relation_boxes) != _bridge_xy_role_signature(
        subject_boxes
    ):
        raise ValueError(
            "subject relation geometry must be XY-congruent with collision boxes"
        )
    support_boxes = _bridge_subject_role_boxes(
        scene,
        subject_id=subject_id,
        role=GeometryRoleV2.SUPPORT,
        objects=objects,
    )
    if _bridge_support_role_signature(support_boxes) != _bridge_support_role_signature(
        subject_boxes
    ):
        raise ValueError(
            "subject support geometry must be contact-congruent with collision boxes"
        )


def _bridge_support_surface(
    surface: SupportSurfaceFact,
    objects: dict[str, object],
) -> SupportSurfaceV3:
    if (
        surface.region_approximation is not GeometryApproximationV2.EXACT
        or surface.boundary_policy.value != "CLOSED"
        or (
            surface.normal_in_anchor.x,
            surface.normal_in_anchor.y,
            surface.normal_in_anchor.z,
        )
        != (0.0, 0.0, 1.0)
        or len(surface.region_uv.components) != 1
        or surface.region_uv.components[0].holes
    ):
        raise ValueError("support surface is unsupported by the exact fixed-box bridge")
    anchor = surface.anchor_from_surface
    anchor_q = _bridge_cardinal_yaw(
        anchor.rotation,
        label=f"support surface {surface.surface_id} anchor rotation",
    )
    if surface.owner_object_id is None:
        owner_q = 0
        owner_x = Fraction()
        owner_y = Fraction()
        owner_z = Fraction()
    else:
        owner = objects.get(surface.owner_object_id)
        if owner is None:
            raise ValueError("support surface owner is absent from the object roster")
        pose = owner.pose.world_from_object
        owner_q = _bridge_cardinal_yaw(
            pose.rotation,
            label=f"support surface {surface.surface_id} owner rotation",
        )
        owner_x = _bridge_fraction(pose.translation.x, label="support owner x")
        owner_y = _bridge_fraction(pose.translation.y, label="support owner y")
        owner_z = _bridge_fraction(pose.translation.z, label="support owner z")
    anchor_x = _bridge_fraction(anchor.translation.x, label="support anchor x")
    anchor_y = _bridge_fraction(anchor.translation.y, label="support anchor y")
    translated_x, translated_y = _rotate_cardinal_xy_components(
        anchor_x, anchor_y, owner_q
    )
    origin_x = owner_x + translated_x
    origin_y = owner_y + translated_y
    total_q = (owner_q + anchor_q) % 4
    vertices = surface.region_uv.components[0].exterior.vertices
    if len(vertices) != 4:
        raise ValueError("support surface must be one exact axis-aligned rectangle")
    world_vertices = tuple(
        (
            origin_x
            + _rotate_cardinal_xy_components(
                _bridge_fraction(vertex.x, label="support vertex x"),
                _bridge_fraction(vertex.y, label="support vertex y"),
                total_q,
            )[0],
            origin_y
            + _rotate_cardinal_xy_components(
                _bridge_fraction(vertex.x, label="support vertex x"),
                _bridge_fraction(vertex.y, label="support vertex y"),
                total_q,
            )[1],
        )
        for vertex in vertices
    )
    xs = tuple(sorted({vertex[0] for vertex in world_vertices}))
    ys = tuple(sorted({vertex[1] for vertex in world_vertices}))
    if (
        len(xs) != 2
        or len(ys) != 2
        or set(world_vertices) != {(x, y) for x in xs for y in ys}
    ):
        raise ValueError("support surface must remain an exact world-XY rectangle")
    return SupportSurfaceV3(
        x_lower=xs[0],
        x_upper=xs[1],
        y_lower=ys[0],
        y_upper=ys[1],
        z=owner_z + _bridge_fraction(anchor.translation.z, label="support anchor z"),
    )


def _bridge_target_and_preservation_rows(
    problem: CounterfactualProblemIR,
    *,
    subject_id: str,
    reference_id: str,
) -> tuple[PredicateAtom, PredicateAtom, str]:
    if len(problem.before_preconditions) != 1:
        raise ValueError("cardinal bridge requires one target before relation")
    before = _direct_predicate_atom(problem.before_preconditions[0].formula)
    _validate_target_relation_operands(
        before,
        subject_id=subject_id,
        reference_id=reference_id,
        phase="BEFORE",
    )
    after = _direct_predicate_atom(problem.after_goal.formula)
    after_relation = _validate_target_relation_operands(
        after,
        subject_id=subject_id,
        reference_id=reference_id,
        phase="AFTER",
    )
    if not after_relation.startswith("relation:"):
        raise ValueError("target after relation must be one registered relation")
    if len(problem.preservation_invariants) != 1:
        raise ValueError("cardinal bridge requires one preservation relation row")
    preservation = problem.preservation_invariants[0]
    for phase, formula in (
        ("BEFORE", preservation.before_formula),
        ("AFTER", preservation.after_formula),
    ):
        _validate_preservation_operands(
            _direct_predicate_atom(formula),
            subject_id,
            phase,
        )
    return before, after, after_relation.removeprefix("relation:")


def _bridge_reference_relation_box(
    scene: CanonicalScene,
    *,
    reference_id: str,
    objects: dict[str, object],
) -> FixedCardinalBoxV3:
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    candidates = tuple(
        geometry
        for geometry in geometries
        if (
            geometry.owner_object_id == reference_id
            and geometry.role is GeometryRoleV2.RELATION
        )
    )
    if len(candidates) != 1:
        raise ValueError("target reference must bind one exact relation geometry")
    return _bridge_fixed_box(candidates[0], objects)


def _bridge_policy_fields(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
    policy_key: str,
) -> dict[str, TypedValue]:
    policy = bundle.policy_for(policy_key)
    payload = policy.payload
    if type(payload.payload) is not RecordValue:
        raise ValueError("bridge policy payload must be an exact typed record")
    fields = {field.name: field.value for field in payload.payload.fields}
    if len(fields) != len(payload.payload.fields):
        raise ValueError("bridge policy payload has duplicate fields")
    return fields


def _bridge_policy_real(fields: dict[str, TypedValue], name: str) -> Fraction:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not FiniteRealValue:
        raise ValueError(f"bridge policy field {name!r} must be a finite real")
    return _bridge_fraction(value.payload.value, label=f"policy field {name}")


def _bridge_policy_integer(fields: dict[str, TypedValue], name: str) -> int:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not IntegerValue:
        raise ValueError(f"bridge policy field {name!r} must be an integer")
    return value.payload.value


def _bridge_policy_symbol(fields: dict[str, TypedValue], name: str) -> str:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not EnumSymbolValue:
        raise ValueError(f"bridge policy field {name!r} must be a symbol")
    return value.payload.symbol


def _bridge_policy_id(fields: dict[str, TypedValue], name: str) -> str:
    value = fields.get(name)
    if type(value) is not TypedValue or type(value.payload) is not CanonicalIdValue:
        raise ValueError(f"bridge policy field {name!r} must be an identifier")
    return value.payload.value


def _bridge_resource_cap(bundle: upright.UprightSE2ExecutablePolicyBundle) -> int:
    fields = _bridge_policy_fields(bundle, "resource")
    cap = _bridge_policy_integer(fields, "atomic_step_limit")
    if cap <= 0:
        raise ValueError("cardinal bridge resource cap must be positive")
    if _bridge_policy_symbol(fields, "deterministic_order") != "LOWER_OWNED_XY":
        raise ValueError("cardinal bridge resource policy order is unsupported")
    return cap


def _bridge_cell_policy(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
    *,
    relation: str,
    cell: ClosedXYCellV3,
    resource_cap: int,
) -> FixedCardinalCellPolicyV3:
    collision = _bridge_policy_fields(bundle, "collision")
    support = _bridge_policy_fields(bundle, "support")
    relation_policy = bundle.policy_for(f"relation:{relation}")
    relation_fields = _bridge_policy_fields(bundle, f"relation:{relation}")
    visibility = _bridge_policy_fields(bundle, "visibility")
    safety = _bridge_policy_fields(bundle, "safety")
    objective = upright.decode_upright_se2_objective_policy(bundle)
    if (
        _bridge_policy_symbol(collision, "contact_comparator") != "ALLOW_EQUALITY"
        or _bridge_policy_symbol(collision, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(support, "containment_comparator") != "CONTAINS"
        or _bridge_policy_symbol(support, "containment_boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(relation_fields, "relation_symbol") != relation
        or _bridge_policy_symbol(relation_fields, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(visibility, "comparator") != "GEQ"
        or _bridge_policy_symbol(visibility, "boundary_policy") != "CLOSED"
        or _bridge_policy_symbol(safety, "safety_penalty_rule")
        != "FROM_CONSTRAINT_SLACK"
    ):
        raise ValueError("cardinal bridge policy symbols are unsupported")
    measurement = _bridge_policy_symbol(relation_fields, "measurement")
    if measurement == "EXTENT_SIGNED_AXIS_GAP":
        retained_measurement = "EXTENT_AWARE_SIGNED_AXIS_GAP"
    elif measurement == "EXTENT_EUCLIDEAN_SEPARATION":
        retained_measurement = "EXTENT_AWARE_EUCLIDEAN_SEPARATION"
    else:
        raise ValueError("cardinal bridge relation measurement is unsupported")
    threshold = _bridge_policy_real(relation_fields, "threshold")
    tolerance = _bridge_policy_real(relation_fields, "tolerance")
    visibility_threshold = _bridge_policy_real(visibility, "threshold")
    visibility_tolerance = _bridge_policy_real(visibility, "tolerance")
    objective_terms = tuple(
        _bridge_objective_term(term, objective.aggregation_definition_ref)
        for term in objective.terms
    )
    return FixedCardinalCellPolicyV3(
        policy_id=relation_policy.definition_ref,
        policy_version="definition:1",
        relation_threshold=threshold,
        relation_tolerance=tolerance,
        relation_comparator=_bridge_policy_symbol(relation_fields, "comparator"),
        relation_boundary="CLOSED",
        # FROM_CONSTRAINT_SLACK is the fixed unit-scale semantic rule; the
        # request-bound S-term weight remains in the common objective roster.
        safety_penalty_scale=Fraction(1),
        safety_constraint_slack_target=_bridge_policy_real(
            safety,
            "constraint_slack_target",
        ),
        safety_rule="PENALIZE_BELOW_TARGET",
        collision_clearance=_bridge_policy_real(collision, "clearance_m"),
        collision_contact_comparator="GE",
        collision_boundary="CLOSED",
        support_accepted_contact_gap=(
            _bridge_policy_real(support, "contact_gap_lower_m"),
            _bridge_policy_real(support, "contact_gap_upper_m"),
        ),
        support_stability_margin=_bridge_policy_real(
            support,
            "stability_margin_m",
        ),
        support_containment_comparator="GE",
        support_boundary="CLOSED",
        support_frame="WORLD_XY_Z_UP",
        support_normal=(Fraction(), Fraction(), Fraction(1)),
        relation_definition_id="spatial-relation:fixed-cardinal",
        relation_definition_version="definition:1",
        relation_symbol=relation,
        relation_measurement=retained_measurement,
        relation_operand="SUBJECT_COMPOUND_TO_REFERENCE",
        relation_geometry="UPRIGHT_AABB_EXTENTS",
        visibility_cell=cell.canonical_bounds,
        visibility_bounds=(
            max(Fraction(), visibility_threshold - visibility_tolerance),
            min(Fraction(1), visibility_threshold + visibility_tolerance),
        ),
        objective_terms=objective_terms,
        atomic_step_limit=resource_cap,
    )


def _bridge_objective_term(
    term: upright.UprightSE2ObjectiveTermPolicy,
    aggregation: str,
) -> FixedCardinalObjectiveTermV3:
    metric_parts = term.metric_definition_ref.rsplit("/", 1)
    if len(metric_parts) != 2:
        raise ValueError("objective metric definition has no fixed version suffix")
    return FixedCardinalObjectiveTermV3(
        term_id=term.term_id,
        selector=term.input_selector_definition_ref,
        metric_definition_id=metric_parts[0],
        metric_definition_version=metric_parts[1],
        unit=term.unit_ref,
        weight=_bridge_fraction(term.weight, label=f"objective {term.term_id} weight"),
        normalizer=_bridge_fraction(
            term.normalizer,
            label=f"objective {term.term_id} normalizer",
        ),
        aggregation=aggregation,
    )


def _bridge_object_pivot_xy(
    objects: dict[str, object],
    object_id: str,
) -> tuple[Fraction, Fraction]:
    object_ = objects.get(object_id)
    if object_ is None:
        raise ValueError("cardinal bridge pivot object is absent from the scene")
    pose = object_.pose.world_from_object
    _bridge_cardinal_yaw(pose.rotation, label=f"pivot object {object_id} rotation")
    return (
        _bridge_fraction(pose.translation.x, label=f"pivot object {object_id} x"),
        _bridge_fraction(pose.translation.y, label=f"pivot object {object_id} y"),
    )


def _bridge_visibility_bound_records(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
) -> tuple[dict[str, TypedValue], ...]:
    fields = _bridge_policy_fields(bundle, "visibility")
    value = fields.get("observation_bounds")
    if (
        type(value) is not TypedValue
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bounds/1.0"
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
    ):
        raise ValueError(
            "cardinal bridge visibility roster is not an exact typed tuple"
        )
    records: list[dict[str, TypedValue]] = []
    expected_names = {
        "observation_id",
        "camera_id",
        "object_id",
        "metric_definition_id",
        "metric_definition_version",
        "comparator",
        "boundary_policy",
        "threshold",
        "tolerance",
    }
    for item in value.payload.items:
        if (
            item.value_schema_ref
            != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
            or type(item.payload) is not RecordValue
        ):
            raise ValueError("cardinal bridge visibility row has the wrong type")
        record = {field.name: field.value for field in item.payload.fields}
        if len(record) != len(item.payload.fields) or set(record) != expected_names:
            raise ValueError("cardinal bridge visibility row is incomplete")
        records.append(record)
    identifiers = tuple(
        _bridge_policy_id(record, "observation_id") for record in records
    )
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or identifiers != tuple(sorted(identifiers, key=canonical_json_bytes))
    ):
        raise ValueError("cardinal bridge visibility rows are not canonical")
    return tuple(records)


def _bridge_projection_box(
    geometry: object,
    objects: dict[str, object],
    *,
    subject_id: str,
    quarter_turns_ccw: int,
    pivot_xy: tuple[Fraction, Fraction],
) -> FixedCardinalProjectionBoxV3:
    center_x, center_y, center_z, half_x, half_y, half_z, _ = _bridge_world_box_parts(
        geometry,
        objects,
    )
    if geometry.owner_object_id == subject_id:
        relative_x = center_x - pivot_xy[0]
        relative_y = center_y - pivot_xy[1]
        rotated_x, rotated_y = _rotate_cardinal_xy_components(
            relative_x,
            relative_y,
            quarter_turns_ccw,
        )
        center_x = pivot_xy[0] + rotated_x
        center_y = pivot_xy[1] + rotated_y
        if quarter_turns_ccw % 2:
            half_x, half_y = half_y, half_x
    return FixedCardinalProjectionBoxV3(
        box_id=geometry.geometry_id,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
    )


def _bridge_camera_context(camera: object) -> UprightCameraContextV2_9:
    rotation = camera.world_to_camera.rotation
    if type(rotation) is not Quaternion or (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("cardinal bridge camera must use identity orientation")
    return UprightCameraContextV2_9(
        camera_id=camera.camera_id,
        width_px=camera.width_px,
        height_px=camera.height_px,
        intrinsics=tuple(
            _bridge_fraction(value, label=f"camera {camera.camera_id} intrinsic")
            for value in camera.intrinsics_row_major
        ),
        near_clip_m=_bridge_fraction(
            camera.near_clip_m,
            label=f"camera {camera.camera_id} near clip",
        ),
        far_clip_m=_bridge_fraction(
            camera.far_clip_m,
            label=f"camera {camera.camera_id} far clip",
        ),
        translation_xyz=tuple(
            _bridge_fraction(value, label=f"camera {camera.camera_id} translation")
            for value in (
                camera.world_to_camera.translation.x,
                camera.world_to_camera.translation.y,
                camera.world_to_camera.translation.z,
            )
        ),
        sine=(Fraction(), Fraction()),
        cosine=(Fraction(1), Fraction(1)),
    )


def _bridge_visibility_inputs(
    scene: CanonicalScene,
    *,
    camera: PinholeCamera,
    objects: dict[str, object],
    subject_id: str,
    quarter_turns_ccw: int,
    pivot_xy: tuple[Fraction, Fraction],
    cell: ClosedXYCellV3,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    resource_cap: int,
) -> tuple[UprightSE2CardinalVisibilityEvaluationInput, ...]:
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    visual_by_object: dict[str, object] = {}
    for geometry in geometries:
        if geometry.role is not GeometryRoleV2.VISUAL:
            continue
        if (
            geometry.owner_object_id is None
            or geometry.owner_object_id in visual_by_object
        ):
            raise ValueError("visual geometry roster must have one owner-bound box")
        visual_by_object[geometry.owner_object_id] = geometry
    moving_geometry = visual_by_object.get(subject_id)
    if moving_geometry is None:
        raise ValueError("cardinal bridge subject has no exact visual geometry")
    projection_by_object = {
        object_id: _bridge_projection_box(
            geometry,
            objects,
            subject_id=subject_id,
            quarter_turns_ccw=quarter_turns_ccw,
            pivot_xy=pivot_xy,
        )
        for object_id, geometry in visual_by_object.items()
    }
    occluders = tuple(sorted(projection_by_object.values(), key=lambda box: box.box_id))
    required_occluder_ids = tuple(box.box_id for box in occluders)
    if len(set(required_occluder_ids)) != len(required_occluder_ids):
        raise ValueError("cardinal bridge visual box IDs are not unique")
    observations = _known_exact_source_values(
        scene.baseline_observations,
        "baseline observations",
    )
    observation_by_id = {
        observation.observation_id: observation for observation in observations
    }
    if len(observation_by_id) != len(observations):
        raise ValueError("cardinal bridge observation roster is not unique")
    bound_records = _bridge_visibility_bound_records(policy_bundle)
    inputs: list[UprightSE2CardinalVisibilityEvaluationInput] = []
    for bound in bound_records:
        observation_id = _bridge_policy_id(bound, "observation_id")
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError("visibility policy row does not bind a source observation")
        metric = (
            _bridge_policy_symbol(bound, "metric_definition_id"),
            _bridge_policy_id(bound, "metric_definition_version"),
        )
        if metric != ("visibility:image-area-fraction", "definition:1"):
            raise ValueError("unsupported registered visibility metric family")
        if (
            observation.camera_id != camera.camera_id
            or observation.camera_id != _bridge_policy_id(bound, "camera_id")
            or observation.object_id != _bridge_policy_id(bound, "object_id")
            or (observation.metric_definition_id, observation.metric_definition_version)
            != metric
            or _bridge_policy_symbol(bound, "comparator") != "GEQ"
            or _bridge_policy_symbol(bound, "boundary_policy") != "CLOSED"
        ):
            raise ValueError(
                "visibility policy row does not close its source observation"
            )
        observed_box = projection_by_object.get(observation.object_id)
        if observed_box is None:
            raise ValueError("visibility source geometry is incomplete")
        inputs.append(
            UprightSE2CardinalVisibilityEvaluationInput(
                observation_id=observation_id,
                context=_bridge_camera_context(camera),
                cell=cell.canonical_bounds,
                subject=observed_box,
                moving_subject_id=moving_geometry.geometry_id,
                occluders=occluders,
                required_occluder_ids=required_occluder_ids,
                policy=FixedCardinalVisibilityPolicyV3(
                    metric_definition_id=metric[0],
                    metric_definition_version=metric[1],
                    metric_threshold=_bridge_policy_real(bound, "threshold"),
                    metric_tolerance=_bridge_policy_real(bound, "tolerance"),
                    metric_comparator="GEQ",
                    metric_boundary="CLOSED",
                    atomic_step_limit=resource_cap,
                ),
            )
        )
    if tuple(sorted(observation_by_id, key=canonical_json_bytes)) != tuple(
        _bridge_policy_id(record, "observation_id") for record in bound_records
    ):
        raise ValueError("visibility policy does not cover the complete source roster")
    return tuple(inputs)


def _bridge_continuous_visibility_inputs(
    scene: CanonicalScene,
    *,
    camera: PinholeCamera,
    objects: dict[str, object],
    subject_id: str,
    visibility_subject_box_id: str,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    resource_cap: int,
) -> tuple[UprightSE2ContinuousVisibilityEvaluationInput, ...]:
    """Bind the one V4 continuous visibility obligation from exact source facts.

    Task 6.1's compound cell seam accepts one caller-supplied visibility DTO
    whose subject ID must equal its primary collision body.  The compiler keeps
    the source visual geometry values and records its original ID, while using
    that collision ID solely as the retained DTO join key.  This is a typed
    source-to-owner mapping, never a backend geometry rewrite.
    """

    geometries = _known_exact_source_values(
        scene.geometry_instances,
        "geometry instances",
    )
    visual_by_object: dict[str, object] = {}
    for geometry in geometries:
        if geometry.role is not GeometryRoleV2.VISUAL:
            continue
        if (
            geometry.owner_object_id is None
            or geometry.owner_object_id in visual_by_object
        ):
            raise ValueError("visual geometry roster must have one owner-bound box")
        visual_by_object[geometry.owner_object_id] = geometry
    moving_geometry = visual_by_object.get(subject_id)
    if moving_geometry is None:
        raise ValueError("continuous bridge subject has no exact visual geometry")
    source_subject = _bridge_fixed_box(moving_geometry, objects)
    subject = FixedCardinalBoxV3(
        box_id=visibility_subject_box_id,
        center_x=source_subject.center_x,
        center_y=source_subject.center_y,
        center_z=source_subject.center_z,
        half_x=source_subject.half_x,
        half_y=source_subject.half_y,
        half_z=source_subject.half_z,
    )
    projection_by_object = {
        object_id: (
            subject if object_id == subject_id else _bridge_fixed_box(geometry, objects)
        )
        for object_id, geometry in visual_by_object.items()
    }
    occluders = tuple(sorted(projection_by_object.values(), key=lambda box: box.box_id))
    required_occluder_ids = tuple(box.box_id for box in occluders)
    if len(set(required_occluder_ids)) != len(required_occluder_ids):
        raise ValueError("continuous bridge visual box IDs are not unique")
    observations = _known_exact_source_values(
        scene.baseline_observations,
        "baseline observations",
    )
    observation_by_id = {
        observation.observation_id: observation for observation in observations
    }
    if len(observation_by_id) != len(observations):
        raise ValueError("continuous bridge observation roster is not unique")
    bound_records = _bridge_visibility_bound_records(policy_bundle)
    if len(bound_records) != 1:
        raise ValueError(
            "unsupported continuous V4 visibility conjunction requires one observation"
        )
    bound = bound_records[0]
    observation_id = _bridge_policy_id(bound, "observation_id")
    observation = observation_by_id.get(observation_id)
    if observation is None:
        raise ValueError("visibility policy row does not bind a source observation")
    metric = (
        _bridge_policy_symbol(bound, "metric_definition_id"),
        _bridge_policy_id(bound, "metric_definition_version"),
    )
    if metric != ("visibility:image-area-fraction", "definition:1"):
        raise ValueError("unsupported registered continuous visibility metric family")
    if (
        observation.camera_id != camera.camera_id
        or observation.camera_id != _bridge_policy_id(bound, "camera_id")
        or observation.object_id != subject_id
        or observation.object_id != _bridge_policy_id(bound, "object_id")
        or (observation.metric_definition_id, observation.metric_definition_version)
        != metric
        or _bridge_policy_symbol(bound, "comparator") != "GEQ"
        or _bridge_policy_symbol(bound, "boundary_policy") != "CLOSED"
    ):
        raise ValueError(
            "unsupported continuous visibility policy does not bind the moving subject"
        )
    return (
        UprightSE2ContinuousVisibilityEvaluationInput(
            observation_id=observation_id,
            source_visual_box_id=source_subject.box_id,
            context=_bridge_camera_context(camera),
            subject=subject,
            moving_subject_id=visibility_subject_box_id,
            occluders=occluders,
            required_occluder_ids=required_occluder_ids,
            policy=FixedCardinalVisibilityPolicyV3(
                metric_definition_id=metric[0],
                metric_definition_version=metric[1],
                metric_threshold=_bridge_policy_real(bound, "threshold"),
                metric_tolerance=_bridge_policy_real(bound, "tolerance"),
                metric_comparator="GEQ",
                metric_boundary="CLOSED",
                atomic_step_limit=resource_cap,
            ),
        ),
    )


def compile_planar_translate_m2_q0_equivalence(
    source_compilation: PlanarTranslateCompilation,
) -> upright.UprightSE2Compilation:
    """Compile one exact retained M2 root into the narrow q=0 construction.

    This is not a delegation, proof checker, endpoint materializer, or a
    cross-policy solver claim.  It first proves that the retained source has
    the one supported closed rectangular world-XY translation domain, then
    builds an ordinary domain-level M3 request whose extra construction root
    binds the retained bytes and the three disjoint relation-row kinds.
    """

    if type(source_compilation) is not PlanarTranslateCompilation:
        raise TypeError(
            "source compilation must be an exact PlanarTranslateCompilation"
        )
    _require_exact_round_trip(
        source_compilation,
        PlanarTranslateCompilation,
        "source compilation",
    )
    source_problem = source_compilation.source_artifacts.problem
    source_scene = _canonical_scene_from_m2_source(source_problem.scene)
    subject_id = source_problem.constraints.allowed_edit.subject_id
    reference_id = source_problem.constraints.target_relation.reference_id
    domain = _m2_supported_translation_domain(source_problem)
    registration = _registered_profile()
    resource_policy = _resource_policy()
    policy_bundle = _m2_q0_policy_bundle(registration, resource_policy, source_scene)
    construction = _m2_q0_construction(
        source_compilation=source_compilation,
        domain=domain,
        policy_bundle=policy_bundle,
    )
    request = _m2_q0_source_request(
        source_problem=source_problem,
        source_scene=source_scene,
        subject_id=subject_id,
        reference_id=reference_id,
        domain=domain,
        registration=registration,
        policy_bundle=policy_bundle,
        construction=construction,
        requested_gap=_source_requested_gap(source_compilation),
    )
    compiled = compile_upright_se2(request)
    if type(compiled) is not upright.UprightSE2Compilation:
        raise ValueError("supported q=0 source must compile to a cardinal M3 domain")
    return _reseal_q0_construction(compiled, construction)


def _canonical_scene_from_m2_source(source_scene: object) -> CanonicalScene:
    """Convert only the retained source's explicit upright scene wire.

    This conversion is target construction data, not an M2/M3 scene equality
    claim.  The unchanged source model remains embedded in the construction
    record, while the target wire uses the M3 rigid-transform representation.
    """

    if not hasattr(source_scene, "model_dump"):
        raise TypeError("source compilation must carry a canonical M2 scene")
    raw = source_scene.model_dump(mode="python", round_trip=True)
    raw["schema_identity"] = {
        "schema_name": "canonical-scene",
        "schema_version": "2.0",
    }
    for object_ in raw["objects"]["values"] or ():
        object_["pose"]["world_from_object"] = _m2_yaw_transform_to_rigid(
            object_["pose"]["world_from_object"]
        )
    for geometry in raw["geometry_instances"]["values"] or ():
        geometry["anchor_from_geometry"] = _m2_yaw_transform_to_rigid(
            geometry["anchor_from_geometry"]
        )
    for surface in raw["support_surfaces"]["values"] or ():
        surface["anchor_from_surface"] = _m2_yaw_transform_to_rigid(
            surface["anchor_from_surface"]
        )
    for camera in raw["cameras"]["values"] or ():
        world_to_camera = camera["world_to_camera"]
        if world_to_camera.get("kind") != "UPRIGHT_WORLD_TO_CAMERA":
            raise ValueError("source camera must use the retained upright camera wire")
        camera["world_to_camera"] = _rigid_from_yaw(
            world_to_camera["translation"],
            world_to_camera["azimuth_radians"],
        )
    try:
        return CanonicalScene.model_validate(raw, strict=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "source scene has no supported exact upright M3 construction"
        ) from error


def _m2_yaw_transform_to_rigid(value: dict[str, object]) -> dict[str, object]:
    if value.get("kind") != "DIRECTED_YAW_INTERVAL":
        raise ValueError("source transform must use the retained directed-yaw wire")
    return _rigid_from_yaw(value["translation"], value["yaw_radians"])


def _rigid_from_yaw(
    translation: object,
    yaw_radians: object,
) -> dict[str, object]:
    if type(yaw_radians) is not float:
        raise ValueError("source yaw must be a finite exact binary64 value")
    return {
        "translation": translation,
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": _canonical_zero(math.sin(yaw_radians / 2.0)),
            "w": _canonical_zero(math.cos(yaw_radians / 2.0)),
        },
    }


def _m2_supported_translation_domain(
    source_problem: object,
) -> upright.UprightSE2TranslationDomain:
    """Return the sole closed axis-aligned M2 anchor domain as XY deltas."""

    constraints = source_problem.constraints
    position = constraints.position_domain
    if (
        position.region_interpretation.value != "SUBJECT_ANCHOR_LOCUS"
        or position.workspace_aggregation.value != "INTERSECTION"
        or position.boundary_policy.value != "CLOSED"
        or position.known_free_space_fact_ids
        or position.subject_occupancy_body_ids
        or position.minimum_boundary_clearance_m != 0.0
        or len(position.workspace_fact_ids) != 1
        or tuple(item.value for item in position.required_completeness) != ("EXACT",)
    ):
        raise ValueError(
            "source M2 domain has no supported total world-XY representation"
        )
    workspace = tuple(
        item
        for item in source_problem.scene.workspace_boundaries.values or ()
        if item.fact_id == position.workspace_fact_ids[0]
    )
    if len(workspace) != 1:
        raise ValueError("source M2 domain must bind one exact workspace fact")
    region = workspace[0].region_world_xy
    if len(region.components) != 1 or region.components[0].holes:
        raise ValueError("source M2 domain must be one closed axis-aligned rectangle")
    vertices = region.components[0].exterior.vertices
    xs = tuple(sorted({point.x for point in vertices}))
    ys = tuple(sorted({point.y for point in vertices}))
    if (
        len(vertices) != 4
        or len(xs) != 2
        or len(ys) != 2
        or {(point.x, point.y) for point in vertices}
        != {(x, y) for x in xs for y in ys}
    ):
        raise ValueError("source M2 domain must be one closed axis-aligned rectangle")
    subject = tuple(
        item
        for item in source_problem.scene.objects.values or ()
        if item.object_id == constraints.allowed_edit.subject_id
    )
    if len(subject) != 1:
        raise ValueError("source M2 domain must bind one source subject pose")
    before = subject[0].pose.world_from_object.translation
    return upright.UprightSE2TranslationDomain(
        x_lower=_dyadic_from_float(_canonical_zero(xs[0] - before.x)),
        x_upper=_dyadic_from_float(_canonical_zero(xs[1] - before.x)),
        y_lower=_dyadic_from_float(_canonical_zero(ys[0] - before.y)),
        y_upper=_dyadic_from_float(_canonical_zero(ys[1] - before.y)),
    )


def _m2_q0_policy_bundle(
    registration: upright.UprightSE2ProfileRegistration,
    resource_policy: ResourcePolicy,
    source_scene: CanonicalScene,
) -> upright.UprightSE2ExecutablePolicyBundle:
    """Build the explicit target M3 policy; no source policy is consumed."""

    keys = (
        "collision",
        "numeric",
        "objective",
        "preservation",
        "relation:BEHIND",
        "relation:FAR",
        "relation:FRONT",
        "relation:LEFT",
        "relation:NEAR",
        "relation:RIGHT",
        "resource",
        "safety",
        "support",
        "visibility",
    )
    objective_policy = upright.build_upright_se2_q0_target_objective_policy()
    return upright.build_upright_se2_executable_policy_bundle(
        profile_registration_sha256=registration.profile_registration_sha256,
        policies=tuple(
            upright.UprightSE2ExecutablePolicyValue.seal(
                policy_key=key,
                policy_family_ref="definition:spatialcf/upright-se2/executable-policy/1.0",
                definition_ref=_m2_q0_policy_definition_ref(key),
                payload_schema_ref=(
                    "schema:spatialcf/upright-se2/executable-"
                    f"{key.replace(':', '-').lower()}-policy/1.0"
                ),
                owner_binding=_m2_q0_policy_owner(_m2_q0_policy_definition_ref(key)),
                payload=_m2_q0_policy_payload(
                    key, resource_policy, source_scene, objective_policy
                ),
            )
            for key in keys
        ),
    )


def _m2_q0_policy_definition_ref(policy_key: str) -> str:
    if policy_key == "collision":
        return upright.UPRIGHT_SE2_COLLISION_PREDICATE_REF
    if policy_key == "support":
        return upright.UPRIGHT_SE2_SUPPORT_PREDICATE_REF
    if policy_key.startswith("relation:"):
        return upright.UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
    if policy_key == "visibility":
        return upright.UPRIGHT_SE2_VISIBILITY_PREDICATE_REF
    if policy_key == "preservation":
        return upright.UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
    return upright.UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF


def _m2_q0_policy_owner(definition_ref: str) -> upright.UprightSE2SemanticOwnerBinding:
    if definition_ref == upright.UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF:
        evaluator, verifier = (
            upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
            upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        )
    else:
        evaluator, verifier = (
            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
        )
    return upright.UprightSE2SemanticOwnerBinding(
        definition_ref=definition_ref,
        evaluator_capability_ref=evaluator,
        verifier_capability_ref=verifier,
        evaluator_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        verifier_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
        evaluator_build_sha256=_BACKEND_BUILD_SHA256,
        verifier_build_sha256=_CHECKER_BUILD_SHA256,
    )


def _m2_q0_policy_payload(
    policy_key: str,
    resource_policy: ResourcePolicy,
    source_scene: CanonicalScene,
    objective_policy: upright.UprightSE2FiveTermObjectivePolicy,
) -> TypedValue:
    """Encode target-only request policy values in their registered schemas."""

    if policy_key == "collision":
        fields = (
            ("boundary_policy", _m2_q0_symbol("CLOSED")),
            ("clearance_m", _m2_q0_real(0.0)),
            ("contact_comparator", _m2_q0_symbol("ALLOW_EQUALITY")),
            ("obstacle_selector", _m2_q0_id("selector:complete-obstacle-roster")),
            (
                "subject_geometry_selector",
                _m2_q0_id("selector:compound-subject-geometry"),
            ),
        )
    elif policy_key == "support":
        fields = (
            ("contact_gap_lower_m", _m2_q0_real(0.0)),
            ("contact_gap_upper_m", _m2_q0_real(0.0)),
            ("containment_boundary_policy", _m2_q0_symbol("CLOSED")),
            ("containment_comparator", _m2_q0_symbol("CONTAINS")),
            ("normal_selector", _m2_q0_id("selector:world-positive-z")),
            ("stability_margin_m", _m2_q0_real(0.0)),
            ("support_frame_selector", _m2_q0_id("selector:world-xy")),
            (
                "support_surface_selector",
                _m2_q0_id("selector:assigned-support-surface"),
            ),
        )
    elif policy_key.startswith("relation:"):
        relation = policy_key.removeprefix("relation:")
        fields = (
            ("boundary_policy", _m2_q0_symbol("CLOSED")),
            (
                "comparator",
                _m2_q0_symbol("LE" if relation in {"LEFT", "FRONT", "NEAR"} else "GE"),
            ),
            ("fixed_camera_selector", _m2_q0_id("selector:fixed-camera")),
            ("frame_selector", _m2_q0_id("selector:world-xy")),
            (
                "measurement",
                _m2_q0_symbol(
                    "EXTENT_EUCLIDEAN_SEPARATION"
                    if relation in {"NEAR", "FAR"}
                    else "EXTENT_SIGNED_AXIS_GAP"
                ),
            ),
            ("operand_order", _m2_q0_symbol("SUBJECT_THEN_REFERENCE")),
            ("relation_symbol", _m2_q0_symbol(relation)),
            ("representative_geometry", _m2_q0_symbol("COMPOUND_BODY")),
            ("threshold", _m2_q0_real(0.25 if relation == "LEFT" else 1.0)),
            ("tolerance", _m2_q0_real(0.0)),
            ("visibility_gate", _m2_q0_symbol("NONE")),
        )
    elif policy_key == "visibility":
        fields = (
            ("boundary_policy", _m2_q0_symbol("CLOSED")),
            ("camera_projection_convention", _m2_q0_symbol("UPRIGHT_CAMERA_V2_9")),
            ("comparator", _m2_q0_symbol("GEQ")),
            ("depth_policy", _m2_q0_symbol("NEAR_CLIPPED")),
            ("mask_policy", _m2_q0_symbol("COMPLETE_MASK")),
            (
                "observation_bounds",
                _m2_q0_tuple(
                    "visibility-observation-bounds",
                    tuple(
                        _m2_q0_typed_record(
                            "schema:spatialcf/upright-se2/visibility-observation-bound/1.0",
                            (
                                (
                                    "observation_id",
                                    _m2_q0_id(observation.observation_id),
                                ),
                                ("camera_id", _m2_q0_id(observation.camera_id)),
                                ("object_id", _m2_q0_id(observation.object_id)),
                                (
                                    "metric_definition_id",
                                    _m2_q0_symbol(observation.metric_definition_id),
                                ),
                                (
                                    "metric_definition_version",
                                    _m2_q0_id(observation.metric_definition_version),
                                ),
                                ("comparator", _m2_q0_symbol("GEQ")),
                                ("boundary_policy", _m2_q0_symbol("CLOSED")),
                                ("threshold", _m2_q0_real(0.5)),
                                ("tolerance", _m2_q0_real(0.0)),
                            ),
                        )
                        for observation in source_scene.baseline_observations.values
                        or ()
                    ),
                    element_schema_ref=(
                        "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
                    ),
                ),
            ),
            ("occluder_policy", _m2_q0_symbol("COMPLETE_ROSTER")),
            ("projected_area_metric", _m2_q0_symbol("PROJECTED_AREA")),
            ("subject_as_occluder", _m2_q0_symbol("INCLUDED")),
            ("threshold", _m2_q0_real(0.5)),
            ("tolerance", _m2_q0_real(0.0)),
        )
    elif policy_key == "numeric":
        fields = (
            ("dyadic_refinement_policy", _m2_q0_symbol("EXACT_DYADIC")),
            ("exact_number_representation", _m2_q0_symbol("BINARY64_BITS")),
            (
                "numeric_semantics_ref",
                _m2_q0_id("definition:spatialcf/upright-se2/numeric-semantics/1.0"),
            ),
            ("tolerance_m", _m2_q0_real(0.0)),
        )
    elif policy_key == "objective":
        fields = (
            ("aggregation_rule", _m2_q0_symbol("WEIGHTED_NORMALIZED_SUM")),
            ("comparison_rule", _m2_q0_symbol("INTERVAL_LEXICOGRAPHIC")),
            (
                "objective_policy_sha256",
                _m2_q0_digest(objective_policy.five_term_objective_policy_sha256),
            ),
            (
                "terms",
                _m2_q0_tuple(
                    "objective-term-values",
                    tuple(
                        _m2_q0_typed_record(
                            "schema:spatialcf/upright-se2/objective-term-value/1.0",
                            (
                                ("term_id", _m2_q0_symbol(term.term_id)),
                                (
                                    "objective_definition_ref",
                                    _m2_q0_id(term.objective_definition_ref),
                                ),
                                (
                                    "metric_definition_ref",
                                    _m2_q0_id(term.metric_definition_ref),
                                ),
                                (
                                    "input_selector_definition_ref",
                                    _m2_q0_id(term.input_selector_definition_ref),
                                ),
                                ("unit_ref", _m2_q0_id(term.unit_ref)),
                                (
                                    "normalization_definition_ref",
                                    _m2_q0_id(term.normalization_definition_ref),
                                ),
                                (
                                    "normalizer_unit_ref",
                                    _m2_q0_id(term.normalizer_unit_ref),
                                ),
                                ("weight", _m2_q0_real(term.weight)),
                                ("normalizer", _m2_q0_real(term.normalizer)),
                            ),
                        )
                        for term in objective_policy.terms
                    ),
                    element_schema_ref=(
                        "schema:spatialcf/upright-se2/objective-term-value/1.0"
                    ),
                ),
            ),
            ("tie_break_rule", _m2_q0_symbol("T_R_V_S_A")),
        )
    elif policy_key == "safety":
        fields = (
            ("constraint_slack_target", _m2_q0_real(0.0)),
            ("hard_constraint_selector", _m2_q0_id("selector:all-hard-constraints")),
            ("safety_penalty_rule", _m2_q0_symbol("FROM_CONSTRAINT_SLACK")),
        )
    elif policy_key == "resource":
        fields = (
            (
                "atomic_step_limit",
                _m2_q0_integer(int(resource_policy.limits[0].finite_limit)),
            ),
            ("deterministic_order", _m2_q0_symbol("LOWER_OWNED_XY")),
            (
                "limits",
                _m2_q0_tuple(
                    "resource-limit-values",
                    tuple(
                        _m2_q0_id(limit.definition_ref)
                        for limit in resource_policy.limits
                    ),
                ),
            ),
            (
                "resource_policy_sha256",
                _m2_q0_digest(resource_policy.resource_policy_sha256),
            ),
            (
                "shared_ledger_policy_ref",
                _m2_q0_id(resource_policy.shared_ledger_policy_ref),
            ),
        )
    elif policy_key == "preservation":
        fields = (
            ("frozen_observation_policy", _m2_q0_symbol("COMPLETE_GROUNDED")),
            ("grounded_invariant_selector", _m2_q0_id("selector:grounded-invariants")),
            ("state_selector", _m2_q0_id("selector:frozen-nonprimary-state")),
        )
    else:
        raise ValueError("q=0 target policy key is unknown")
    return _m2_q0_record(policy_key.replace(":", "-"), fields)


def _m2_q0_real(value: float) -> TypedValue:
    return TypedValue(
        value_schema_ref=_REAL_SCHEMA_REF, payload=FiniteRealValue(value=value)
    )


def _m2_q0_integer(value: int) -> TypedValue:
    return TypedValue(
        value_schema_ref=_INTEGER_SCHEMA_REF, payload=IntegerValue(value=value)
    )


def _m2_q0_id(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_ID_SCHEMA_REF, payload=CanonicalIdValue(value=value)
    )


def _m2_q0_digest(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_DIGEST_SCHEMA_REF, payload=DigestValue(value=value)
    )


def _m2_q0_symbol(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_ENUM_SCHEMA_REF, payload=EnumSymbolValue(symbol=value)
    )


def _m2_q0_tuple(
    name: str,
    values: tuple[TypedValue, ...],
    *,
    element_schema_ref: str = "schema:spatialcf/upright-se2/policy-item/1.0",
) -> TypedValue:
    return TypedValue(
        value_schema_ref=f"schema:spatialcf/upright-se2/{name}/1.0",
        payload=FiniteOrderedTupleValue(
            element_schema_ref=element_schema_ref,
            items=values,
        ),
    )


def _m2_q0_record(name: str, fields: tuple[tuple[str, TypedValue], ...]) -> TypedValue:
    return _m2_q0_typed_record(
        f"schema:spatialcf/upright-se2/executable-{name.lower()}-policy/1.0",
        fields,
    )


def _m2_q0_typed_record(
    schema_ref: str,
    fields: tuple[tuple[str, TypedValue], ...],
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(name=field_name, value=value)
                        for field_name, value in fields
                    ),
                    key=canonical_json_bytes,
                )
            )
        ),
    )


def _m2_q0_construction(
    *,
    source_compilation: PlanarTranslateCompilation,
    domain: upright.UprightSE2TranslationDomain,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
) -> upright.UprightSE2M2Q0Construction:
    """Bind provenance, the one domain equality, and heterogeneous policy rows."""

    domain_value = _m2_q0_domain_value(domain)
    domain_digest = canonical_sha256(
        domain_value,
        domain="spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0",
    )
    definitions: list[upright.UprightSE2M2Q0MappingDefinition] = []
    rows: list[upright.UprightSE2M2Q0MappingRow] = []

    def add_row(
        *,
        selector: str,
        kind: str,
        source_value: object,
        target_selector: str | None = None,
        target_value: object | None = None,
        reason: str | None = None,
        transform: str | None = None,
        shared_value: TypedValue | None = None,
    ) -> None:
        definition = upright.UprightSE2M2Q0MappingDefinition.seal(
            mapping_definition_ref=(
                "definition:spatialcf/upright-se2/m2-q0/mapping/"
                f"{selector.removeprefix('source:').replace('/', '-')}/1.0"
            ),
            row_kind=kind,
            source_selector=selector,
            target_selector=target_selector,
            source_value_schema_ref=(
                "schema:spatialcf/upright-se2/m2-q0/translation-domain/1.0"
                if kind == "EQUALITY"
                else "schema:spatialcf/upright-se2/m2-q0/source-leaf/1.0"
            ),
            target_value_schema_ref=(
                None
                if target_selector is None
                else (
                    "schema:spatialcf/upright-se2/m2-q0/translation-domain/1.0"
                    if kind == "EQUALITY"
                    else "schema:spatialcf/upright-se2/m2-q0/target-leaf/1.0"
                )
            ),
            source_unit_ref=(
                "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                if kind == "EQUALITY"
                else None
            ),
            target_unit_ref=(
                "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                if kind == "EQUALITY"
                else None
            ),
            transform_ref=transform,
            non_equivalence_reason_ref=reason,
            mapping_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            mapping_version="mapping-version:spatialcf/upright-se2/m2-q0/1",
            accepted_source_domain_ref=(
                "definition:spatialcf/upright-se2/m2-closed-axis-aligned-rect-domain/1.0"
            ),
        )
        definitions.append(definition)
        source_digest = canonical_sha256(
            source_value,
            domain="spatialcf/counterfactual/upright-se2/m2-q0/source-leaf/3.0",
        )
        target_digest = (
            None
            if target_value is None
            else canonical_sha256(
                target_value,
                domain="spatialcf/counterfactual/upright-se2/m2-q0/target-leaf/3.0",
            )
        )
        if kind == "EQUALITY":
            source_digest = domain_digest
            target_digest = domain_digest
        rows.append(
            upright.UprightSE2M2Q0MappingRow.seal(
                mapping_definition=definition,
                row_kind=kind,
                source_value_sha256=source_digest,
                target_value_sha256=target_digest,
                shared_value=shared_value,
                shared_value_sha256=(
                    None
                    if shared_value is None
                    else canonical_sha256(
                        shared_value,
                        domain="spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0",
                    )
                ),
                non_equivalence_reason_ref=reason,
            )
        )

    source_problem = source_compilation.source_artifacts.problem
    add_row(
        selector="source:constraints/position-domain",
        kind="EQUALITY",
        source_value=domain_value,
        target_selector="target:operation/translation-domain",
        target_value=domain_value,
        transform="definition:spatialcf/upright-se2/m2-q0/delta-xy-identity/1.0",
        shared_value=domain_value,
    )
    for selector, value in _m2_q0_source_leaves(source_compilation):
        add_row(selector=selector, kind="SOURCE_CONTEXT", source_value=value)
    for selector, source_value, target_selector, target_value, reason in (
        (
            "source:policy/objective",
            source_problem.objective,
            "target:policy/objective",
            policy_bundle.policy_for("objective").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-objective/1.0",
        ),
        (
            "source:policy/relation",
            source_problem.relation_semantics,
            "target:policy/relation",
            tuple(
                policy.payload
                for policy in policy_bundle.policies
                if policy.policy_key.startswith("relation:")
            ),
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-relation/1.0",
        ),
        (
            "source:policy/visibility",
            source_problem.visibility_semantics,
            "target:policy/visibility",
            policy_bundle.policy_for("visibility").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-visibility/1.0",
        ),
        (
            "source:policy/support",
            source_problem.constraints.support_constraints,
            "target:policy/support",
            policy_bundle.policy_for("support").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-support/1.0",
        ),
        (
            "source:policy/numeric",
            source_problem.numeric_policy,
            "target:policy/numeric",
            policy_bundle.policy_for("numeric").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-numeric/1.0",
        ),
        (
            "source:policy/resource",
            source_compilation.source_artifacts.config,
            "target:policy/resource",
            policy_bundle.policy_for("resource").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-resource/1.0",
        ),
        (
            "source:policy/tie-break",
            source_problem.objective.tie_break,
            "target:policy/tie-break",
            policy_bundle.policy_for("objective").payload,
            "definition:spatialcf/upright-se2/m2-q0/heterogeneous-tie-break/1.0",
        ),
    ):
        add_row(
            selector=selector,
            kind="NON_EQUIVALENCE",
            source_value=source_value,
            target_selector=target_selector,
            target_value=target_value,
            reason=reason,
        )
    definitions = tuple(sorted(definitions, key=canonical_json_bytes))
    rows = tuple(
        sorted(rows, key=lambda row: canonical_json_bytes(row.mapping_definition))
    )
    source_free_rows = (
        upright.UprightSE2M2Q0SourceFreeConstructionRow.seal(
            construction_selector="construction:cardinal-own-pivot-q",
            target_value=_m2_q0_int(0),
            target_value_sha256=canonical_sha256(
                _m2_q0_int(0),
                domain="spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0",
            ),
            construction_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            construction_version="construction-version:spatialcf/upright-se2/m2-q0/1",
        ),
        upright.UprightSE2M2Q0SourceFreeConstructionRow.seal(
            construction_selector="construction:target-tie-break",
            target_value=_m2_q0_symbol("T_R_V_S_A"),
            target_value_sha256=canonical_sha256(
                _m2_q0_symbol("T_R_V_S_A"),
                domain="spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0",
            ),
            construction_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            construction_version="construction-version:spatialcf/upright-se2/m2-q0/1",
        ),
    )
    return upright.UprightSE2M2Q0Construction.seal(
        source_compilation=source_compilation,
        source_compilation_sha256=source_compilation.compilation_sha256,
        source_provenance_sha256=canonical_sha256(
            source_compilation,
            domain="spatialcf/counterfactual/upright-se2/m2-q0/source-provenance/3.0",
        ),
        authorized_domain=domain,
        authorized_domain_sha256=canonical_sha256(
            domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        ),
        mapping_definitions=definitions,
        mapping_definition_roster_sha256=canonical_sha256(
            tuple(item.mapping_definition_sha256 for item in definitions),
            domain="spatialcf/counterfactual/upright-se2/m2-q0/mapping-definition-roster/3.0",
        ),
        rows=rows,
        source_free_construction_rows=source_free_rows,
        frozen_m3_policy_bundle=policy_bundle,
        frozen_m3_policy_bundle_sha256=policy_bundle.policy_bundle_sha256,
    )


def _m2_q0_source_leaves(
    source_compilation: PlanarTranslateCompilation,
) -> tuple[tuple[str, object], ...]:
    """Enumerate every retained source leaf in canonical source-tree order."""

    leaves: list[tuple[str, object]] = []

    def visit(value: object, selector: str) -> None:
        if type(value) is dict:
            if not value:
                leaves.append((selector, value))
                return
            for key in sorted(value, key=canonical_json_bytes):
                visit(value[key], f"{selector}/{key}")
            return
        if type(value) in (list, tuple):
            if not value:
                leaves.append((selector, value))
                return
            for index, item in enumerate(value):
                visit(item, f"{selector}/{index}")
            return
        leaves.append((selector, value))

    visit(
        source_compilation.model_dump(mode="python", round_trip=True),
        "source:compilation",
    )
    return tuple(leaves)


def _m2_q0_domain_value(domain: upright.UprightSE2TranslationDomain) -> TypedValue:
    return _m2_q0_record(
        "m2-q0-domain",
        (
            ("x_lower", _m2_q0_real(float(domain.x_lower.as_fraction))),
            ("x_upper", _m2_q0_real(float(domain.x_upper.as_fraction))),
            ("y_lower", _m2_q0_real(float(domain.y_lower.as_fraction))),
            ("y_upper", _m2_q0_real(float(domain.y_upper.as_fraction))),
        ),
    )


def _m2_q0_int(value: int) -> TypedValue:
    return TypedValue(
        value_schema_ref=_INTEGER_SCHEMA_REF, payload=IntegerValue(value=value)
    )


def _m2_q0_source_request(
    *,
    source_problem: object,
    source_scene: CanonicalScene,
    subject_id: str,
    reference_id: str,
    domain: upright.UprightSE2TranslationDomain,
    registration: upright.UprightSE2ProfileRegistration,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    construction: upright.UprightSE2M2Q0Construction,
    requested_gap: upright.UprightSE2ExactRational,
) -> CounterfactualSolveRequest:
    """Build the fixed M3 q=0 source root without selecting an endpoint.

    The retained M2 root supplies only scene authority, the supported delta
    domain, and verbatim provenance.  This constructor deliberately owns the
    distinct M3 policy, grounded M3 obligations, and source-free q=0 constant;
    none of those target values are presented as M2 mappings.
    """

    subject = _m2_q0_scene_object(source_scene, subject_id, "subject")
    _m2_q0_scene_object(source_scene, reference_id, "reference")
    source_relation = source_problem.constraints.target_relation
    before_relation = f"relation:{source_relation.relation_before.value}"
    after_relation = f"relation:{source_relation.relation_after.value}"
    if before_relation == after_relation:
        raise ValueError(
            "source M2 relation must specify distinct before and after goals"
        )

    compiler_fact = ExtensionFact(
        fact_family_ref=_INPUT_FAMILY_REF,
        subject_entity_id=_input_entity_id(subject_id),
        fact_key=_COMPILER_INPUT_FACT_KEY,
        value=_m2_q0_typed_record(
            _INPUT_SCHEMA_REF,
            (
                ("operation_kind", _m2_q0_symbol("CARDINAL")),
                (
                    "operator_ref",
                    _m2_q0_id(upright.UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF),
                ),
                ("reference_id", _m2_q0_id(reference_id)),
                ("subject_id", _m2_q0_id(subject_id)),
                (
                    "subject_yaw_turns",
                    _m2_q0_real(
                        upright.canonical_yaw_from_upright_quaternion(
                            subject.pose.world_from_object.rotation
                        ).turns
                    ),
                ),
                (
                    "yaw_argument",
                    _m2_q0_typed_record(
                        _YAW_ARGUMENT_SCHEMA_REF,
                        (
                            ("kind", _m2_q0_symbol("CARDINAL")),
                            ("quarter_turns_ccw", _m2_q0_int(0)),
                        ),
                    ),
                ),
            ),
        ),
    )
    policy_fact = ExtensionFact(
        fact_family_ref="definition:spatialcf/upright-se2/executable-policy/1.0",
        subject_entity_id="entity:upright-se2-policy",
        fact_key="fact-key:spatialcf/upright-se2/executable-policy-bundle",
        value=upright.executable_policy_bundle_to_typed_value(policy_bundle),
    )
    construction_fact = ExtensionFact(
        fact_family_ref="definition:spatialcf/upright-se2/m2-q0-construction/1.0",
        subject_entity_id="entity:upright-se2-policy",
        fact_key="fact-key:spatialcf/upright-se2/m2-q0-construction",
        value=_m2_q0_digest(construction.m2_q0_construction_sha256),
    )
    scene_state = SceneStateEnvelope.seal(
        base_scene_schema_ref=_SCENE_SCHEMA_REF,
        base_scene_payload=source_scene,
        extension_fact_bundles=(
            ExtensionFactBundle.seal(
                facts=_sorted_bytes(compiler_fact, policy_fact, construction_fact)
            ),
        ),
        closed_entity_index=_closed_entity_index(source_scene, subject_id),
        canonical_state_leaf_index=StateLeafIndex.seal(
            leaves=_expected_state_leaves(source_scene, subject_id)
        ),
    )
    authorization = InterventionAuthorization.seal(
        editable_entity_ids=(_input_entity_id(subject_id),),
        allowed_operator_refs=(upright.UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,),
        authorized_primary_write_set=_primary_write_set(subject_id),
        variable_bounds=_m2_q0_variable_bounds(subject_id, domain),
        maximum_program_steps=1,
        maximum_edited_entities=1,
        required_derived_rule_refs=(_DERIVED_RULE_REF,),
        complete_state_delta_policy_ref=(
            "definition:spatialcf/upright-se2/complete-state-delta/1.0"
        ),
    )
    semantic_problem = CounterfactualProblemIR.seal(
        problem_id=(
            "problem:spatialcf/upright-se2/m2-q0/"
            f"{construction.source_compilation_sha256}"
        ),
        scene_state=scene_state,
        definition_bundle=upright.build_upright_se2_semantic_definition_bundle(
            registration,
            objective_policy=upright.decode_upright_se2_objective_policy(policy_bundle),
        ),
        semantics_profile_ref=upright.UPRIGHT_SE2_SEMANTICS_PROFILE_REF,
        action_space_profile_ref=upright.UPRIGHT_SE2_PROFILE_REF,
        intervention_authorization=authorization,
        before_preconditions=(
            BeforePrecondition(
                formula=_m2_q0_target_relation_atom(
                    subject_id=subject_id,
                    reference_id=reference_id,
                    relation=before_relation,
                    phase="BEFORE",
                )
            ),
        ),
        after_goal=AfterGoal(
            formula=_m2_q0_target_relation_atom(
                subject_id=subject_id,
                reference_id=reference_id,
                relation=after_relation,
                phase="AFTER",
            )
        ),
        preservation_invariants=(
            PreservationInvariant(
                before_formula=_m2_q0_preservation_atom(subject_id, "BEFORE"),
                after_formula=_m2_q0_preservation_atom(subject_id, "AFTER"),
                transition_comparator_ref=(
                    "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0"
                ),
            ),
        ),
        explicit_observation_obligations=_sorted_bytes(
            *(
                _m2_q0_visibility_obligation(observation)
                for observation in source_scene.baseline_observations.values or ()
            )
        ),
        objective_expression=upright.build_upright_se2_objective_expression(),
        numeric_semantics_ref=registration.semantics_profile.numeric_semantics_ref,
    )
    return CounterfactualSolveRequest.seal(
        semantic_problem=semantic_problem,
        semantic_problem_sha256=semantic_problem.semantic_problem_sha256,
        solve_policy_definition_bundle=_solve_policy_bundle(
            registration, requested_gap=requested_gap
        ),
        implementation_registry_snapshot=_implementation_registry(registration),
        backend_descriptor_bundle=_backend_descriptor_bundle(registration),
        solver_config=_solver_config(),
        proof_policy=_proof_policy(),
        resource_policy=_resource_policy(),
        backend_routing_policy=_backend_routing_policy(),
    )


def _m2_q0_scene_object(scene: CanonicalScene, object_id: str, label: str) -> object:
    matches = tuple(
        object_
        for object_ in scene.objects.values or ()
        if object_.object_id == object_id
    )
    if len(matches) != 1:
        raise ValueError(f"source M2 {label} must name one exact scene object")
    return matches[0]


def _m2_q0_variable_bounds(
    subject_id: str,
    domain: upright.UprightSE2TranslationDomain,
) -> tuple[TypedVariableBound, ...]:
    return _sorted_bytes(
        TypedVariableBound(
            state_variable_ref=_state_leaf(subject_id, "subject-world-x"),
            value_schema_ref=_REAL_SCHEMA_REF,
            typed_domain=_m2_q0_closed_interval(domain.x_lower, domain.x_upper),
            frame_ref=_WORLD_XY_FRAME_REF,
            unit_ref=_METRE_UNIT_REF,
            topology_ref=_CLOSED_INTERVAL_TOPOLOGY_REF,
        ),
        TypedVariableBound(
            state_variable_ref=_state_leaf(subject_id, "subject-world-y"),
            value_schema_ref=_REAL_SCHEMA_REF,
            typed_domain=_m2_q0_closed_interval(domain.y_lower, domain.y_upper),
            frame_ref=_WORLD_XY_FRAME_REF,
            unit_ref=_METRE_UNIT_REF,
            topology_ref=_CLOSED_INTERVAL_TOPOLOGY_REF,
        ),
    )


def _m2_q0_closed_interval(
    lower: upright.ExactDyadic,
    upper: upright.ExactDyadic,
) -> TypedValue:
    return TypedValue(
        value_schema_ref=_REAL_SCHEMA_REF,
        payload=IntervalValue(
            endpoint_schema_ref=_REAL_SCHEMA_REF,
            lower=FiniteRealValue(value=float(lower.as_fraction)),
            upper=FiniteRealValue(value=float(upper.as_fraction)),
            lower_closed=True,
            upper_closed=True,
        ),
    )


def _m2_q0_typed_reference(
    schema_ref: str, kind: ValueKind, reference: str
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=ReferenceValue(kind=kind, reference=reference),
    )


def _m2_q0_target_relation_atom(
    *,
    subject_id: str,
    reference_id: str,
    relation: str,
    phase: str,
) -> PredicateAtom:
    return PredicateAtom(
        predicate_ref=upright.UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
        operands=(
            _m2_q0_typed_reference(
                "schema:spatialcf/upright-se2/object-ref/1.0",
                ValueKind.OBJECT_REF,
                subject_id,
            ),
            _m2_q0_typed_reference(
                "schema:spatialcf/upright-se2/object-ref/1.0",
                ValueKind.OBJECT_REF,
                reference_id,
            ),
            TypedValue(
                value_schema_ref="schema:spatialcf/upright-se2/relation-symbol/1.0",
                payload=EnumSymbolValue(symbol=relation),
            ),
            TypedValue(
                value_schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
                payload=EnumSymbolValue(symbol=f"phase:{phase}"),
            ),
        ),
    )


def _m2_q0_preservation_atom(subject_id: str, phase: str) -> PredicateAtom:
    return PredicateAtom(
        predicate_ref=upright.UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
        operands=(
            _m2_q0_typed_reference(
                "schema:spatialcf/upright-se2/entity-ref/1.0",
                ValueKind.ENTITY_REF,
                _input_entity_id(subject_id),
            ),
            TypedValue(
                value_schema_ref="schema:spatialcf/upright-se2/preservation-selector/1.0",
                payload=EnumSymbolValue(symbol="preservation:FROZEN_NONPRIMARY_LEAVES"),
            ),
            TypedValue(
                value_schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
                payload=EnumSymbolValue(symbol=f"phase:{phase}"),
            ),
        ),
    )


def _m2_q0_visibility_obligation(observation: object) -> ObservationObligation:
    return ObservationObligation(
        phase="AFTER",
        formula=PredicateAtom(
            predicate_ref=upright.UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
            operands=(
                _m2_q0_typed_reference(
                    "schema:spatialcf/upright-se2/camera-ref/1.0",
                    ValueKind.CAMERA_REF,
                    observation.camera_id,
                ),
                _m2_q0_typed_reference(
                    "schema:spatialcf/upright-se2/object-ref/1.0",
                    ValueKind.OBJECT_REF,
                    observation.object_id,
                ),
                TypedValue(
                    value_schema_ref=(
                        "schema:spatialcf/upright-se2/visibility-metric-symbol/1.0"
                    ),
                    payload=EnumSymbolValue(symbol=observation.metric_definition_id),
                ),
                TypedValue(
                    value_schema_ref="schema:spatialcf/upright-se2/observation-ref/1.0",
                    payload=CanonicalIdValue(value=observation.observation_id),
                ),
                TypedValue(
                    value_schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
                    payload=EnumSymbolValue(symbol="phase:AFTER"),
                ),
            ),
        ),
        evidence_policy_ref="definition:spatialcf/upright-se2/visibility-evidence-policy/1.0",
    )


def _reseal_q0_construction(
    compilation: upright.UprightSE2Compilation,
    construction: upright.UprightSE2M2Q0Construction,
) -> upright.UprightSE2Compilation:
    values = compilation.model_dump(mode="python", round_trip=True)
    values.pop("upright_se2_compilation_sha256")
    values["m2_q0_construction"] = construction
    return upright.UprightSE2Compilation.seal(**values)


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


def _registered_profile() -> upright.UprightSE2ProfileRegistration:
    semantics_profile = SemanticsProfile.seal(
        semantics_profile_ref=upright.UPRIGHT_SE2_SEMANTICS_PROFILE_REF,
        accepted_scene_and_fact_schema_refs=(_SCENE_SCHEMA_REF,),
        predicate_definition_refs=upright.UPRIGHT_SE2_PREDICATE_DEFINITION_REFS,
        transition_semantics_refs=upright.UPRIGHT_SE2_OPERATOR_REFS,
        objective_definition_refs=(upright.UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,),
        numeric_semantics_ref="definition:spatialcf/upright-se2/numeric-semantics/1.0",
        completeness_policy_ref="definition:spatialcf/upright-se2/completeness-policy/1.0",
        uncertainty_policy_ref="definition:spatialcf/upright-se2/uncertainty-policy/1.0",
        derived_fact_rule_refs=(_DERIVED_RULE_REF,),
        observation_obligation_policy_ref=(
            "definition:spatialcf/upright-se2/observation-policy/1.0"
        ),
    )
    action_space_profile = ActionSpaceProfile.seal(
        action_space_profile_ref=upright.UPRIGHT_SE2_PROFILE_REF,
        accepted_scene_schema_refs=(_SCENE_SCHEMA_REF,),
        state_variable_definition_refs=_sorted_bytes(
            "definition:spatialcf/upright-se2/state-world-xy/1.0",
            "definition:spatialcf/upright-se2/state-directed-yaw/1.0",
        ),
        allowed_operator_refs=upright.UPRIGHT_SE2_OPERATOR_REFS,
        mandatory_invariant_template_refs=(
            "definition:spatialcf/upright-se2/invariants-frozen-state/1.0",
        ),
        predicate_capability_refs=upright.UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS,
        objective_capability_refs=upright.UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS,
        numeric_semantics_ref="definition:spatialcf/upright-se2/numeric-semantics/1.0",
        allowed_claim_definition_refs=_sorted_bytes(
            "definition:spatialcf/upright-se2/claim-certified-solution/1.0",
            upright.UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
            "definition:spatialcf/upright-se2/claim-proven-unsat/1.0",
            "definition:spatialcf/upright-se2/claim-unknown/1.0",
        ),
        backend_capability_requirements=upright.UPRIGHT_SE2_STAGED_CAPABILITY_REFS,
        adapter_capability_requirements=(),
        publication_proof_policy_ref=(
            "definition:spatialcf/upright-se2/publication-proof-policy/1.0"
        ),
    )
    return upright.UprightSE2ProfileRegistration.seal(
        semantics_profile=semantics_profile,
        action_space_profile=action_space_profile,
        backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        checker_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
        profile_capability_ref=upright.UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
        cardinal_compiler_capability_ref=upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
        cardinal_backend_capability_ref=upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
        cardinal_checker_capability_ref=upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        continuous_compiler_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
        continuous_backend_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
        continuous_checker_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
    )


def _validate_problem_and_extract_input(
    solve_request: CounterfactualSolveRequest,
    registration: upright.UprightSE2ProfileRegistration,
) -> tuple[_CompilerInput, _SceneAuthority, upright.UprightSE2ExecutablePolicyBundle]:
    problem = solve_request.semantic_problem
    if problem.semantics_profile_ref != upright.UPRIGHT_SE2_SEMANTICS_PROFILE_REF:
        raise ValueError("semantics profile reference is not the upright se2 profile")
    if problem.action_space_profile_ref != upright.UPRIGHT_SE2_PROFILE_REF:
        raise ValueError(
            "action-space profile reference is not the upright se2 profile"
        )
    if (
        problem.numeric_semantics_ref
        != registration.semantics_profile.numeric_semantics_ref
    ):
        raise ValueError("numeric semantics do not match the upright se2 profile")
    if canonical_json_bytes(problem.objective_expression) != canonical_json_bytes(
        upright.build_upright_se2_objective_expression()
    ):
        raise ValueError(
            "objective expression does not match the fixed five-term upright policy"
        )

    scene_state = problem.scene_state
    if scene_state.base_scene_schema_ref != _SCENE_SCHEMA_REF:
        raise ValueError("base scene schema does not match the upright se2 profile")
    compiler_input = _compiler_input_from_scene(solve_request)
    executable_policy_bundle = (
        upright.request_bound_executable_policy_bundle_from_problem(problem)
    )
    objective_policy = upright.decode_upright_se2_objective_policy(
        executable_policy_bundle
    )
    if canonical_json_bytes(problem.definition_bundle) != canonical_json_bytes(
        upright.build_upright_se2_semantic_definition_bundle(
            registration, objective_policy=objective_policy
        )
    ):
        raise ValueError(
            "semantic definition bundle does not match the request-bound upright closure"
        )
    if (
        executable_policy_bundle.profile_registration_sha256
        != registration.profile_registration_sha256
    ):
        raise ValueError("executable policy bundle does not bind the upright profile")
    authority = _resolve_scene_authority(scene_state.base_scene_payload, compiler_input)
    if scene_state.closed_entity_index != _closed_entity_index(
        authority.scene,
        authority.subject.object_id,
    ):
        raise ValueError("closed entity index does not match the upright se2 profile")
    _validate_complete_state_leaves(
        scene_state.canonical_state_leaf_index.leaves,
        authority.scene,
        compiler_input.subject_id,
    )
    _validate_grounded_semantic_operands(problem, authority, compiler_input)
    upright.validate_upright_se2_executable_policy_visibility_binding(
        executable_policy_bundle, problem
    )
    return compiler_input, authority, executable_policy_bundle


def _compiler_input_from_scene(
    solve_request: CounterfactualSolveRequest,
) -> _CompilerInput:
    scene = solve_request.semantic_problem.scene_state
    bundles = scene.extension_fact_bundles
    facts = tuple(fact for bundle in bundles for fact in bundle.facts)
    compiler_facts = tuple(
        fact
        for fact in facts
        if fact.fact_family_ref == _INPUT_FAMILY_REF
        and fact.fact_key == _COMPILER_INPUT_FACT_KEY
    )
    construction_facts = tuple(
        fact
        for fact in facts
        if fact.fact_family_ref
        == "definition:spatialcf/upright-se2/m2-q0-construction/1.0"
    )
    if (
        len(compiler_facts) != 1
        or len(construction_facts) > 1
        or len(facts) != 2 + len(construction_facts)
    ):
        raise ValueError(
            "upright se2 compilation requires one compiler input, one executable policy, and at most one construction root"
        )
    fact = compiler_facts[0]
    if (
        fact.value.value_schema_ref != _INPUT_SCHEMA_REF
        or type(fact.value.payload) is not RecordValue
    ):
        raise ValueError("compiler input must use the fixed upright se2 record schema")
    fields = fact.value.payload.fields
    expected_names = _sorted_bytes(
        "operation_kind",
        "operator_ref",
        "reference_id",
        "subject_id",
        "subject_yaw_turns",
        "yaw_argument",
    )
    if tuple(field.name for field in fields) != expected_names:
        raise ValueError(
            "compiler input fields do not match the fixed upright se2 schema"
        )
    values = {field.name: field.value for field in fields}
    compiler_input = _CompilerInput(
        operation_kind=_symbol_field(values, "operation_kind"),
        operator_ref=_id_field(values, "operator_ref"),
        subject_id=_id_field(values, "subject_id"),
        reference_id=_id_field(values, "reference_id"),
        subject_yaw_turns=_real_field(values, "subject_yaw_turns"),
        yaw_argument=_yaw_argument_field(values["yaw_argument"]),
    )
    _validate_compiler_input_fact_identity(fact, compiler_input.subject_id)
    return compiler_input


def _input_entity_id(object_id: str) -> str:
    """Return the M1 authorization owner corresponding to one scene object."""

    return f"entity:{object_id}"


def _validate_compiler_input_fact_identity(
    fact: ExtensionFact,
    subject_id: str,
) -> None:
    if (
        fact.fact_family_ref != _INPUT_FAMILY_REF
        or fact.subject_entity_id != _input_entity_id(subject_id)
        or fact.fact_key != _COMPILER_INPUT_FACT_KEY
    ):
        raise ValueError(
            "compiler input fact identity does not match the upright se2 profile"
        )


def _real_field(values: dict[str, TypedValue], name: str) -> float:
    value = values[name]
    if (
        value.value_schema_ref != _REAL_SCHEMA_REF
        or type(value.payload) is not FiniteRealValue
    ):
        raise ValueError(f"compiler input field {name!r} must be a finite real")
    return value.payload.value


def _integer_field(values: dict[str, TypedValue], name: str) -> int:
    value = values[name]
    if (
        value.value_schema_ref != _INTEGER_SCHEMA_REF
        or type(value.payload) is not IntegerValue
    ):
        raise ValueError(f"compiler input field {name!r} must be an integer")
    return value.payload.value


def _id_field(values: dict[str, TypedValue], name: str) -> str:
    value = values[name]
    if (
        value.value_schema_ref != _ID_SCHEMA_REF
        or type(value.payload) is not CanonicalIdValue
    ):
        raise ValueError(f"compiler input field {name!r} must be a canonical ID")
    return value.payload.value


def _symbol_field(values: dict[str, TypedValue], name: str) -> str:
    value = values[name]
    if (
        value.value_schema_ref != _ENUM_SCHEMA_REF
        or type(value.payload) is not EnumSymbolValue
    ):
        raise ValueError(f"compiler input field {name!r} must be an enum symbol")
    return value.payload.symbol


def _yaw_argument_field(
    value: TypedValue,
) -> upright.CardinalYaw | upright.ContinuousYawDomain:
    if (
        value.value_schema_ref != _YAW_ARGUMENT_SCHEMA_REF
        or type(value.payload) is not RecordValue
    ):
        raise ValueError("yaw argument must use the fixed upright se2 record schema")
    fields = value.payload.fields
    values = {field.name: field.value for field in fields}
    if "kind" not in values:
        raise ValueError("yaw argument must declare one wire branch")
    kind = _symbol_field(values, "kind")
    if kind == "CARDINAL":
        if tuple(field.name for field in fields) != _sorted_bytes(
            "kind", "quarter_turns_ccw"
        ):
            raise ValueError("cardinal yaw argument fields are not canonical")
        return upright.CardinalYaw(q=_integer_field(values, "quarter_turns_ccw"))
    if kind == "ARC":
        if tuple(field.name for field in fields) != _sorted_bytes(
            "ccw_sweep_turns", "kind", "start_turns"
        ):
            raise ValueError("arc yaw argument fields are not canonical")
        return upright.ContinuousYawArc(
            start_angle=upright.CanonicalSO2Angle(
                turns=_real_field(values, "start_turns")
            ),
            ccw_sweep_turns=_real_field(values, "ccw_sweep_turns"),
        )
    if kind == "FULL_CIRCLE":
        if tuple(field.name for field in fields) != _sorted_bytes("kind"):
            raise ValueError("full-circle yaw argument must not carry arc fields")
        return upright.ContinuousYawFullCircle()
    raise ValueError("yaw argument must be CARDINAL, ARC, or FULL_CIRCLE")


def _resolve_scene_authority(
    scene: CanonicalScene,
    compiler_input: _CompilerInput,
) -> _SceneAuthority:
    objects = {object_.object_id: object_ for object_ in scene.objects.values or ()}
    subject = objects.get(compiler_input.subject_id)
    if subject is None:
        raise ValueError("subject object must name one Canonical Scene object")
    reference = objects.get(compiler_input.reference_id)
    if reference is None:
        raise ValueError("reference object must name one Canonical Scene object")
    if subject.object_id == reference.object_id:
        raise ValueError("reference object must not name the subject object")
    if not subject.movable:
        raise ValueError("subject object must be movable")
    if subject.pose.anchor_kind != "OBJECT_PIVOT":
        raise ValueError("subject object must expose an OBJECT_PIVOT pose")
    if reference.pose.anchor_kind != "OBJECT_PIVOT":
        raise ValueError("reference object must expose an OBJECT_PIVOT pose")
    _validate_required_exact_scene_sources(scene, subject)
    return _SceneAuthority(scene=scene, subject=subject, reference=reference)


def _validate_required_exact_scene_sources(
    scene: CanonicalScene, subject: object
) -> None:
    """Reject incomplete sources before materializing closed derived inputs."""

    for field_name, label in (
        ("objects", "objects"),
        ("geometry_instances", "geometry instances"),
        ("collision_bodies", "collision bodies"),
        ("support_surfaces", "support surfaces"),
        ("cameras", "cameras"),
        ("baseline_observations", "baseline observations"),
    ):
        _known_exact_source_values(getattr(scene, field_name), label)

    assignment = subject.support_assignment
    if (
        assignment.availability is not FactAvailabilityV2.KNOWN
        or assignment.surface_id is None
    ):
        raise ValueError(
            "subject support assignment must be KNOWN with a support surface"
        )
    upright.validate_required_upright_support_surface(scene, subject.object_id)


def _known_exact_source_values(
    facts: FactSetV2,
    label: str,
) -> tuple[object, ...]:
    """Return the sole complete branch accepted by this compilation profile."""

    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.values is None
        or facts.inner_values is not None
        or facts.outer_values is not None
    ):
        raise ValueError(f"{label} must be a KNOWN EXACT fact set")
    return facts.values


def _validate_complete_state_leaves(
    leaves: tuple[StateVariableRef, ...],
    scene: CanonicalScene,
    subject_id: str,
) -> None:
    if leaves != _expected_state_leaves(scene, subject_id):
        raise ValueError(
            "complete state leaf index does not match the upright se2 profile"
        )


def _validate_grounded_semantic_operands(
    problem: CounterfactualProblemIR,
    authority: _SceneAuthority,
    compiler_input: _CompilerInput,
) -> None:
    """Resolve every M3 goal/preservation/visibility operand against scene authority."""

    if (
        len(problem.before_preconditions) != 1
        or type(problem.before_preconditions[0]) is not BeforePrecondition
    ):
        raise ValueError(
            "semantic closure requires one grounded target before-precondition"
        )
    before_atom = _direct_predicate_atom(problem.before_preconditions[0].formula)
    if before_atom.predicate_ref != upright.UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF:
        raise ValueError(
            "semantic closure before-precondition must use the target relation"
        )
    before_relation = _validate_target_relation_operands(
        before_atom,
        subject_id=authority.subject.object_id,
        reference_id=authority.reference.object_id,
        phase="BEFORE",
    )

    if type(problem.after_goal) is not AfterGoal:
        raise ValueError("semantic closure after-goal must use the target relation")
    after_atom = _direct_predicate_atom(problem.after_goal.formula)
    if after_atom.predicate_ref != upright.UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF:
        raise ValueError("semantic closure after-goal must use the target relation")
    after_relation = _validate_target_relation_operands(
        after_atom,
        subject_id=authority.subject.object_id,
        reference_id=authority.reference.object_id,
        phase="AFTER",
    )
    if before_relation == after_relation:
        raise ValueError(
            "semantic target relation must name distinct before and after relations"
        )

    if (
        len(problem.preservation_invariants) != 1
        or type(problem.preservation_invariants[0]) is not PreservationInvariant
    ):
        raise ValueError(
            "semantic closure requires one grounded preservation invariant"
        )
    preservation = problem.preservation_invariants[0]
    if preservation.transition_comparator_ref != (
        "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0"
    ):
        raise ValueError(
            "preservation transition comparator does not match the upright policy"
        )
    for phase, formula in (
        ("BEFORE", preservation.before_formula),
        ("AFTER", preservation.after_formula),
    ):
        atom = _direct_predicate_atom(formula)
        if atom.predicate_ref != upright.UPRIGHT_SE2_PRESERVATION_PREDICATE_REF:
            raise ValueError(
                "preservation invariant must use the upright preservation predicate"
            )
        _validate_preservation_operands(atom, compiler_input.subject_id, phase)

    expected_observations = _known_exact_source_values(
        authority.scene.baseline_observations,
        "baseline observations",
    )
    if len(problem.explicit_observation_obligations) != len(expected_observations):
        raise ValueError(
            "semantic closure must ground every exact visibility observation"
        )
    actual_by_observation_id: dict[str, ObservationObligation] = {}
    for obligation in problem.explicit_observation_obligations:
        if type(obligation) is not ObservationObligation or obligation.phase != "AFTER":
            raise ValueError(
                "semantic closure visibility obligations must be AFTER observations"
            )
        if obligation.evidence_policy_ref != (
            "definition:spatialcf/upright-se2/visibility-evidence-policy/1.0"
        ):
            raise ValueError(
                "visibility evidence policy does not match the upright policy"
            )
        atom = _direct_predicate_atom(obligation.formula)
        if atom.predicate_ref != upright.UPRIGHT_SE2_VISIBILITY_PREDICATE_REF:
            raise ValueError(
                "visibility obligation must use the upright visibility predicate"
            )
        observation_id = _validate_visibility_operands(atom, authority.scene)
        if observation_id in actual_by_observation_id:
            raise ValueError(
                "semantic closure visibility obligations must not duplicate observations"
            )
        actual_by_observation_id[observation_id] = obligation
    expected_by_observation_id = {
        observation.observation_id: observation for observation in expected_observations
    }
    if set(actual_by_observation_id) != set(expected_by_observation_id):
        raise ValueError(
            "semantic closure visibility obligations must cover the exact observation roster"
        )


def _direct_predicate_atom(formula: object) -> PredicateAtom:
    if type(formula) is not PredicateAtom:
        raise ValueError(
            "semantic obligations must use direct grounded predicate atoms"
        )
    if not formula.operands:
        raise ValueError("semantic predicate atoms must carry explicit operands")
    return formula


def _validate_target_relation_operands(
    atom: PredicateAtom,
    *,
    subject_id: str,
    reference_id: str,
    phase: str,
) -> str:
    if len(atom.operands) != 4:
        raise ValueError("target relation predicate requires four explicit operands")
    _require_reference_operand(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
        reference=subject_id,
    )
    _require_reference_operand(
        atom.operands[1],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
        reference=reference_id,
    )
    relation = _require_symbol_operand(
        atom.operands[2],
        schema_ref="schema:spatialcf/upright-se2/relation-symbol/1.0",
    )
    if relation not in {
        "relation:LEFT",
        "relation:RIGHT",
        "relation:FRONT",
        "relation:BEHIND",
        "relation:NEAR",
        "relation:FAR",
    }:
        raise ValueError("target relation operand must name one registered relation")
    if (
        _require_symbol_operand(
            atom.operands[3],
            schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != f"phase:{phase}"
    ):
        raise ValueError("target relation phase operand does not match its obligation")
    return relation


def _validate_preservation_operands(
    atom: PredicateAtom,
    subject_id: str,
    phase: str,
) -> None:
    if len(atom.operands) != 3:
        raise ValueError("preservation predicate requires three explicit operands")
    _require_reference_operand(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/entity-ref/1.0",
        kind=ValueKind.ENTITY_REF,
        reference=f"entity:{subject_id}",
    )
    if (
        _require_symbol_operand(
            atom.operands[1],
            schema_ref="schema:spatialcf/upright-se2/preservation-selector/1.0",
        )
        != "preservation:FROZEN_NONPRIMARY_LEAVES"
    ):
        raise ValueError("preservation selector must name the frozen non-primary state")
    if (
        _require_symbol_operand(
            atom.operands[2],
            schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != f"phase:{phase}"
    ):
        raise ValueError("preservation phase operand does not match its formula")


def _validate_visibility_operands(atom: PredicateAtom, scene: CanonicalScene) -> str:
    if len(atom.operands) != 5:
        raise ValueError("visibility predicate requires five explicit operands")
    camera_id = _require_reference_operand(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/camera-ref/1.0",
        kind=ValueKind.CAMERA_REF,
    )
    object_id = _require_reference_operand(
        atom.operands[1],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
    )
    metric_id = _require_symbol_operand(
        atom.operands[2],
        schema_ref="schema:spatialcf/upright-se2/visibility-metric-symbol/1.0",
    )
    observation_id = _require_id_operand(
        atom.operands[3],
        schema_ref="schema:spatialcf/upright-se2/observation-ref/1.0",
    )
    if (
        _require_symbol_operand(
            atom.operands[4],
            schema_ref="schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != "phase:AFTER"
    ):
        raise ValueError("visibility phase operand must be AFTER")
    observations = _known_exact_source_values(
        scene.baseline_observations,
        "baseline observations",
    )
    matching = tuple(
        observation
        for observation in observations
        if observation.observation_id == observation_id
    )
    if len(matching) != 1:
        raise ValueError("visibility operand must name one exact baseline observation")
    observation = matching[0]
    if (
        camera_id != observation.camera_id
        or object_id != observation.object_id
        or metric_id != observation.metric_definition_id
    ):
        raise ValueError(
            "visibility operands do not bind their exact baseline observation"
        )
    return observation_id


def _require_reference_operand(
    value: TypedValue,
    *,
    schema_ref: str,
    kind: ValueKind,
    reference: str | None = None,
) -> str:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not ReferenceValue
        or value.payload.kind is not kind
    ):
        raise ValueError(
            "semantic reference operand has the wrong schema or reference kind"
        )
    if reference is not None and value.payload.reference != reference:
        raise ValueError(
            "semantic reference operand does not bind the named scene authority"
        )
    return value.payload.reference


def _require_symbol_operand(value: TypedValue, *, schema_ref: str) -> str:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not EnumSymbolValue
    ):
        raise ValueError("semantic symbol operand has the wrong schema")
    return value.payload.symbol


def _require_id_operand(value: TypedValue, *, schema_ref: str) -> str:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not CanonicalIdValue
    ):
        raise ValueError("semantic identifier operand has the wrong schema")
    return value.payload.value


def _expected_state_leaves(
    scene: CanonicalScene,
    subject_id: str,
) -> tuple[StateVariableRef, ...]:
    rows: list[tuple[str, str]] = [
        *((subject_id, role) for role in (*_PRIMARY_ROLES, "subject-world-z")),
        *((subject_id, role) for role in _DERIVED_ROLES),
    ]
    for family_name, identifier in _SOURCE_FIELDS:
        role = f"frozen-source-{family_name}"
        rows.extend(
            (entity_id, role)
            for entity_id in _fact_ids(scene, family_name.replace("-", "_"), identifier)
        )
        rows.append((scene.scene_id, f"frozen-source-{family_name}-set"))
    return _sorted_bytes(*(_state_leaf(entity_id, role) for entity_id, role in rows))


def _closed_entity_index(
    scene: CanonicalScene,
    subject_id: str,
) -> tuple[str, ...]:
    return _sorted_bytes(
        scene.scene_id,
        _input_entity_id(subject_id),
        "entity:upright-se2-policy",
        *(
            entity_id
            for family_name, identifier in _SOURCE_FIELDS
            for entity_id in _fact_ids(scene, family_name.replace("-", "_"), identifier)
        ),
    )


def _fact_ids(
    scene: CanonicalScene, field_name: str, identifier: str
) -> tuple[str, ...]:
    facts = getattr(scene, field_name)
    return tuple(
        sorted(
            {
                getattr(value, identifier)
                for values in (facts.values, facts.inner_values, facts.outer_values)
                if values is not None
                for value in values
            }
        )
    )


def _state_leaf(entity_id: str, role: str) -> StateVariableRef:
    return StateVariableRef(
        state_variable_schema_ref=f"schema:spatialcf/upright-se2/{role}/1.0",
        state_schema_ref=_STATE_SCHEMA_REF,
        fact_family_ref=_STATE_FAMILY_REF,
        entity_or_fact_key=entity_id,
        field_path_ref=f"field-path:upright-se2-{role}",
    )


def _validate_operational_closure(
    solve_request: CounterfactualSolveRequest,
    registration: upright.UprightSE2ProfileRegistration,
    *,
    operation_kind: str = "CARDINAL",
) -> None:
    solve_policy = upright.decode_upright_se2_solve_policy_definition_payload(
        solve_request.solve_policy_definition_bundle
    )
    if (
        solve_policy.profile_registration_sha256
        != registration.profile_registration_sha256
        or solve_policy.objective_bound_policy_ref
        != solve_request.solver_config.objective_bound_policy_ref
        or solve_policy.exact_global_claim_definition_ref
        != upright.UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF
        or solve_policy.finite_gap_claim_definition_ref
        != upright.UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF
    ):
        raise ValueError(
            "solve policy definition bundle does not match the upright se2 closure"
        )
    if operation_kind not in {"CARDINAL", "CONTINUOUS"}:
        raise ValueError("operational closure operation kind is unknown")
    continuous = operation_kind == "CONTINUOUS"
    _validate_registry(
        solve_request.implementation_registry_snapshot,
        registration,
        continuous=continuous,
    )
    if canonical_json_bytes(
        solve_request.backend_descriptor_bundle
    ) != canonical_json_bytes(
        _backend_descriptor_bundle(registration, continuous=continuous)
    ):
        raise ValueError(
            "backend descriptor bundle does not match the upright se2 closure"
        )
    if canonical_json_bytes(solve_request.solver_config) != canonical_json_bytes(
        _solver_config()
    ):
        raise ValueError("solver config does not match the upright se2 closure")
    if canonical_json_bytes(solve_request.proof_policy) != canonical_json_bytes(
        _proof_policy(continuous=continuous)
    ):
        raise ValueError("proof policy does not match the upright se2 closure")
    _validate_request_resource_policy(solve_request.resource_policy)
    if canonical_json_bytes(
        solve_request.backend_routing_policy
    ) != canonical_json_bytes(_backend_routing_policy()):
        raise ValueError(
            "backend routing policy does not match the upright se2 closure"
        )


def _validate_registry(
    registry: ImplementationRegistrySnapshot,
    registration: upright.UprightSE2ProfileRegistration,
    *,
    continuous: bool = False,
) -> None:
    build_by_owner = dict(registry.implementation_build_hashes)
    if (
        build_by_owner.get(upright.UPRIGHT_SE2_COMPILER_OWNER_REF)
        != upright.UPRIGHT_SE2_COMPILER_BUILD_SHA256
    ):
        raise ValueError("compiler build does not match the fixed upright se2 compiler")
    if canonical_json_bytes(registry) != canonical_json_bytes(
        _implementation_registry(registration, continuous=continuous)
    ):
        raise ValueError(
            "implementation registry does not match the upright se2 closure"
        )


def _validate_intervention_authorization(
    solve_request: CounterfactualSolveRequest,
    compiler_input: _CompilerInput,
    authority: _SceneAuthority,
) -> upright.UprightSE2TranslationDomain:
    authorization = solve_request.semantic_problem.intervention_authorization
    if authorization.editable_entity_ids != (
        _input_entity_id(authority.subject.object_id),
    ):
        raise ValueError("upright se2 authorization must permit exactly the subject")
    if authorization.allowed_operator_refs != (compiler_input.operator_ref,):
        raise ValueError(
            "upright se2 authorization must permit exactly the requested operator"
        )
    if authorization.authorized_primary_write_set != _primary_write_set(
        compiler_input.subject_id
    ):
        raise ValueError("upright se2 authorization must contain only subject X/Y/yaw")
    if (
        authorization.maximum_program_steps != 1
        or authorization.maximum_edited_entities != 1
    ):
        raise ValueError(
            "upright se2 authorization requires one step and one edited entity"
        )
    if authorization.required_derived_rule_refs != (_DERIVED_RULE_REF,):
        raise ValueError("upright se2 authorization must require the derived fact rule")
    if authorization.complete_state_delta_policy_ref != (
        "definition:spatialcf/upright-se2/complete-state-delta/1.0"
    ):
        raise ValueError("upright se2 authorization must use the complete state policy")
    if compiler_input.operation_kind == "CARDINAL":
        if compiler_input.operator_ref not in (
            upright.UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
            upright.UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
        ):
            raise ValueError(
                "cardinal compiler input must name a registered cardinal operator"
            )
        if type(compiler_input.yaw_argument) is not upright.CardinalYaw:
            raise ValueError("cardinal requests must carry one cardinal yaw argument")
    elif compiler_input.operation_kind == "CONTINUOUS":
        if compiler_input.operator_ref not in (
            upright.UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
            upright.UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
        ):
            raise ValueError(
                "continuous compiler input must name a registered continuous operator"
            )
        if not isinstance(
            compiler_input.yaw_argument,
            (upright.ContinuousYawArc, upright.ContinuousYawFullCircle),
        ):
            raise ValueError("continuous requests must carry one continuous yaw domain")
    else:
        raise ValueError("compiler input operation kind must be CARDINAL or CONTINUOUS")
    return _translation_domain_from_authorization(
        authorization.variable_bounds,
        authority.subject.object_id,
    )


def _validate_explicit_pose_yaw(
    compiler_input: _CompilerInput,
    solve_request: CounterfactualSolveRequest,
    authority: _SceneAuthority,
) -> upright.CanonicalSO2Angle:
    explicit_yaw = upright.CanonicalSO2Angle(turns=compiler_input.subject_yaw_turns)
    rotation = authority.subject.pose.world_from_object.rotation
    upright.validate_directed_yaw_quaternion_consistency(explicit_yaw, rotation)
    upright.ExplicitPoseYawBinding.seal(
        entity_id=compiler_input.subject_id,
        base_pose_sha256=canonical_sha256(
            {
                "scene_state_sha256": solve_request.semantic_problem.scene_state.scene_state_sha256,
                "subject_object_id": compiler_input.subject_id,
                "object_pivot_pose": authority.subject.pose,
            },
            domain=_POSE_STATE_HASH_DOMAIN,
        ),
        explicit_yaw=explicit_yaw,
        yaw_to_pose_rule_ref=upright.UPRIGHT_SE2_YAW_TO_POSE_RULE_REF,
    )
    return explicit_yaw


def _resolve_pivot_binding(
    compiler_input: _CompilerInput,
    solve_request: CounterfactualSolveRequest,
    authority: _SceneAuthority,
) -> upright.FixedPivotBinding:
    if compiler_input.operator_ref in (
        upright.UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
        upright.UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
    ):
        pivot_mode = upright.PivotMode.OWN
        pivot_entity_id = compiler_input.subject_id
        pivot_pose = authority.subject.pose
    else:
        pivot_mode = upright.PivotMode.REFERENCE
        pivot_entity_id = compiler_input.reference_id
        pivot_pose = authority.reference.pose
    return upright.FixedPivotBinding.seal(
        subject_id=compiler_input.subject_id,
        pivot_mode=pivot_mode,
        pivot_entity_id=pivot_entity_id,
        pivot_state_sha256=canonical_sha256(
            {
                "scene_state_sha256": solve_request.semantic_problem.scene_state.scene_state_sha256,
                "pivot_entity_id": pivot_entity_id,
                "object_pivot_pose": pivot_pose,
            },
            domain=_PIVOT_STATE_HASH_DOMAIN,
        ),
    )


def _validate_continuous_request(
    compiler_input: _CompilerInput,
    pivot_binding: upright.FixedPivotBinding,
) -> None:
    yaw_domain = compiler_input.yaw_argument
    if not isinstance(
        yaw_domain,
        (upright.ContinuousYawArc, upright.ContinuousYawFullCircle),
    ):
        raise TypeError("continuous request must carry a continuous yaw domain")
    upright.ContinuousYawAuthorization.seal(
        subject_id=compiler_input.subject_id,
        operator_ref=compiler_input.operator_ref,
        pivot_binding=pivot_binding,
        yaw_domain=yaw_domain,
    )


def _translation_domain_from_authorization(
    bounds: tuple[object, ...],
    subject_id: str,
) -> upright.UprightSE2TranslationDomain:
    expected = {
        "subject-world-x": _state_leaf(subject_id, "subject-world-x"),
        "subject-world-y": _state_leaf(subject_id, "subject-world-y"),
    }
    if len(bounds) != len(expected):
        raise ValueError(
            "upright se2 authorization requires complete world-XY variable bounds"
        )
    resolved: dict[str, tuple[upright.ExactDyadic, upright.ExactDyadic]] = {}
    for bound in bounds:
        role = next(
            (
                candidate
                for candidate, state_leaf in expected.items()
                if bound.state_variable_ref == state_leaf
            ),
            None,
        )
        if role is None:
            raise ValueError(
                "world-XY variable bounds must target the subject X/Y leaves"
            )
        if (
            bound.value_schema_ref,
            bound.frame_ref,
            bound.unit_ref,
            bound.topology_ref,
        ) != (
            _REAL_SCHEMA_REF,
            _WORLD_XY_FRAME_REF,
            _METRE_UNIT_REF,
            _CLOSED_INTERVAL_TOPOLOGY_REF,
        ):
            raise ValueError(
                "world-XY variable bounds do not use the authorized type semantics"
            )
        domain = bound.typed_domain
        if (
            domain.value_schema_ref != _REAL_SCHEMA_REF
            or type(domain.payload) is not IntervalValue
            or domain.payload.endpoint_schema_ref != _REAL_SCHEMA_REF
            or type(domain.payload.lower) is not FiniteRealValue
            or type(domain.payload.upper) is not FiniteRealValue
            or not domain.payload.lower_closed
            or not domain.payload.upper_closed
        ):
            raise ValueError(
                "world-XY variable bounds must use closed finite-real intervals"
            )
        resolved[role] = (
            _dyadic_from_float(domain.payload.lower.value),
            _dyadic_from_float(domain.payload.upper.value),
        )
    if set(resolved) != set(expected):
        raise ValueError("world-XY variable bounds must be complete")
    return upright.UprightSE2TranslationDomain(
        x_lower=resolved["subject-world-x"][0],
        x_upper=resolved["subject-world-x"][1],
        y_lower=resolved["subject-world-y"][0],
        y_upper=resolved["subject-world-y"][1],
    )


def _compile_cardinal(
    *,
    solve_request: CounterfactualSolveRequest,
    registration: upright.UprightSE2ProfileRegistration,
    compiler_input: _CompilerInput,
    authority: _SceneAuthority,
    base_yaw: upright.CanonicalSO2Angle,
    translation_domain: upright.UprightSE2TranslationDomain,
    authorization: upright.CardinalYawAuthorization,
    executable_policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
) -> upright.UprightSE2Compilation:
    operation = upright.UprightSE2CardinalOperation(
        authorization=authorization,
        translation_domain=translation_domain,
        inverse_quarter_turns_ccw=cardinal_inverse_quarter_turns(authorization.yaw.q),
        maximum_program_steps=1,
        maximum_edited_entities=1,
    )
    endpoint_recipe = _endpoint_construction_recipe(
        solve_request=solve_request,
        authority=authority,
        operation=operation,
        base_yaw=base_yaw,
    )
    footprint = _state_footprint(solve_request, compiler_input.subject_id)
    grounded_obligations = _grounded_obligations(solve_request)
    semantic_closure = upright.build_upright_se2_semantic_closure(
        profile_registration=registration,
        semantic_problem=solve_request.semantic_problem,
        definition_bundle=solve_request.semantic_problem.definition_bundle,
        grounded_obligations=grounded_obligations,
        objective_expression=solve_request.semantic_problem.objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=solve_request.resource_policy,
    )
    closure = upright.UprightSE2CompilerClosure.seal(
        profile_registration_sha256=registration.profile_registration_sha256,
        definition_bundle_sha256=solve_request.semantic_problem.definition_bundle.definition_bundle_sha256,
        solve_policy_definition_bundle_sha256=(
            solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        ),
        semantic_closure_sha256=semantic_closure.semantic_closure_sha256,
        policy_bundle_sha256=semantic_closure.policy_bundle_sha256,
        resource_policy_sha256=solve_request.resource_policy.resource_policy_sha256,
        compiler_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
        compiler_build_sha256=upright.UPRIGHT_SE2_COMPILER_BUILD_SHA256,
    )
    cell = _compiled_cell(operation)
    return upright.UprightSE2Compilation.seal(
        solve_request_sha256=solve_request.solve_request_sha256,
        source_solve_request=solve_request,
        closure=closure,
        operation=operation,
        endpoint_construction_recipe=endpoint_recipe,
        state_footprint=footprint,
        grounded_obligations=grounded_obligations,
        semantic_closure=semantic_closure,
        compiled_cells=(cell,),
    )


def compile_upright_se2_continuous(
    solve_request: CounterfactualSolveRequest,
) -> upright.UprightSE2ContinuousCompilation:
    """Compile a registered continuous request through the sole M3 compiler.

    The public sibling intentionally rejects cardinal requests rather than
    converting them or widening the retained cardinal compilation API.
    """

    compiled = compile_upright_se2(solve_request)
    if type(compiled) is not upright.UprightSE2ContinuousCompilation:
        raise ValueError("continuous compiler requires one continuous M3 request")
    return compiled


def _compile_continuous(
    solve_request: CounterfactualSolveRequest,
    *,
    registration: upright.UprightSE2ProfileRegistration,
    compiler_input: _CompilerInput,
    authority: _SceneAuthority,
    executable_policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    translation_domain: upright.UprightSE2TranslationDomain,
    base_yaw: upright.CanonicalSO2Angle,
    pivot_binding: upright.FixedPivotBinding,
) -> upright.UprightSE2ContinuousCompilation:
    """Construct continuous roots only; no search, geometry, or checking."""

    if not isinstance(
        compiler_input.yaw_argument,
        (upright.ContinuousYawArc, upright.ContinuousYawFullCircle),
    ):
        raise ValueError(  # noqa: TRY004 - preserves the public invalid-domain contract.
            "continuous compilation requires a continuous yaw domain"
        )
    authorization = upright.ContinuousYawAuthorization.seal(
        subject_id=compiler_input.subject_id,
        operator_ref=compiler_input.operator_ref,
        pivot_binding=pivot_binding,
        yaw_domain=compiler_input.yaw_argument,
    )
    operation = upright.UprightSE2ContinuousOperation(
        authorization=authorization,
        translation_domain=translation_domain,
        maximum_program_steps=1,
        maximum_edited_entities=1,
    )
    budget = SO2AtomicBudgetV2(limit=_bridge_resource_cap(executable_policy_bundle))
    lift_outcome = compile_continuous_yaw_lift_v4(
        operation.yaw_domain,
        atomic_budget=budget,
    )
    if lift_outcome.kind is not ContinuousYawIntervalKindV4.EXACT:
        # This is an input/interval closure failure, not a backend conclusion.
        raise ValueError(
            "continuous compiler could not construct exact lifted yaw root: "
            + ",".join(lift_outcome.finding_codes)
        )
    assert lift_outcome.bounds is not None
    lift = lift_outcome.bounds.lift
    endpoint_recipe = upright.UprightSE2ContinuousEndpointConstructionRecipe.seal(
        source_scene_state_sha256=(
            solve_request.semantic_problem.scene_state.scene_state_sha256
        ),
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=authority.subject.object_id,
        reference_id=authority.reference.object_id,
        pivot_binding=operation.pivot_binding,
        yaw_domain=operation.yaw_domain,
        translation_domain=operation.translation_domain,
        subject_before_pose=authority.subject.pose.world_from_object,
        reference_before_pose=authority.reference.pose.world_from_object,
        subject_yaw_turns=base_yaw,
        evaluation_scene=authority.scene,
    )
    footprint = _state_footprint(solve_request, compiler_input.subject_id)
    grounded_obligations = _grounded_obligations(solve_request)
    semantic_closure = upright.build_upright_se2_continuous_semantic_closure(
        profile_registration=registration,
        semantic_problem=solve_request.semantic_problem,
        definition_bundle=solve_request.semantic_problem.definition_bundle,
        grounded_obligations=grounded_obligations,
        objective_expression=solve_request.semantic_problem.objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=solve_request.resource_policy,
    )
    closure = upright.UprightSE2CompilerClosure.seal(
        profile_registration_sha256=registration.profile_registration_sha256,
        definition_bundle_sha256=(
            solve_request.semantic_problem.definition_bundle.definition_bundle_sha256
        ),
        solve_policy_definition_bundle_sha256=(
            solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        ),
        semantic_closure_sha256=semantic_closure.semantic_closure_sha256,
        policy_bundle_sha256=semantic_closure.policy_bundle_sha256,
        resource_policy_sha256=solve_request.resource_policy.resource_policy_sha256,
        compiler_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
        compiler_build_sha256=upright.UPRIGHT_SE2_COMPILER_BUILD_SHA256,
    )
    interval = lift.intervals[0]
    cell = upright.UprightSE2CompiledCell.seal(
        cell_id=(
            "cell:spatialcf/upright-se2/continuous/"
            f"{operation.authorization_sha256}/{lift.continuous_yaw_lift_sha256}"
        ),
        authorization_sha256=operation.authorization_sha256,
        x_lower=operation.translation_domain.x_lower,
        x_upper=operation.translation_domain.x_upper,
        y_lower=operation.translation_domain.y_lower,
        y_upper=operation.translation_domain.y_upper,
        yaw_interval=interval,
    )
    return upright.UprightSE2ContinuousCompilation.seal(
        solve_request_sha256=solve_request.solve_request_sha256,
        source_solve_request=solve_request,
        closure=closure,
        operation=operation,
        endpoint_construction_recipe=endpoint_recipe,
        state_footprint=footprint,
        grounded_obligations=grounded_obligations,
        semantic_closure=semantic_closure,
        compiled_cells=(cell,),
        continuous_yaw_lift=lift,
    )


def _endpoint_construction_recipe(
    *,
    solve_request: CounterfactualSolveRequest,
    authority: _SceneAuthority,
    operation: upright.UprightSE2CardinalOperation,
    base_yaw: upright.CanonicalSO2Angle,
) -> upright.UprightSE2EndpointConstructionRecipe:
    """Bind source-only data required by later endpoint materialization."""

    return upright.UprightSE2EndpointConstructionRecipe.seal(
        source_scene_state_sha256=(
            solve_request.semantic_problem.scene_state.scene_state_sha256
        ),
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=authority.subject.object_id,
        reference_id=authority.reference.object_id,
        pivot_binding=operation.pivot_binding,
        quarter_turns_ccw=operation.quarter_turns_ccw,
        translation_domain=operation.translation_domain,
        subject_before_pose=authority.subject.pose.world_from_object,
        reference_before_pose=authority.reference.pose.world_from_object,
        subject_yaw_turns=base_yaw,
        evaluation_scene=authority.scene,
    )


def materialize_upright_se2_endpoint(
    compilation: upright.UprightSE2Compilation,
    translation_xy_m: Vec2,
) -> upright.UprightSE2MaterializedEndpoint:
    """Purely materialize one authorized endpoint from a domain compilation.

    The caller selects ``translation_xy_m`` only here.  This function has no
    solver/backend/checker dependency and returns no feasibility or execution
    result.
    """

    if type(compilation) is not upright.UprightSE2Compilation:
        raise TypeError("materialization requires an UprightSE2Compilation")
    if type(translation_xy_m) is not Vec2:
        raise TypeError("materialization requires an exact Vec2 translation")
    recipe = compilation.endpoint_construction_recipe
    after_state = _after_state_template(recipe, translation_xy_m)
    program = _materialized_program(compilation, after_state, translation_xy_m)
    return upright.UprightSE2MaterializedEndpoint.seal(
        upright_se2_compilation_sha256=compilation.upright_se2_compilation_sha256,
        compilation=compilation,
        endpoint_construction_recipe=recipe,
        translation_xy_m=translation_xy_m,
        after_state=after_state,
        program=program,
    )


def materialize_upright_se2_continuous_endpoint(
    compilation: upright.UprightSE2ContinuousCompilation,
    translation_xy_m: Vec2,
    selected_lifted_yaw: upright.ExactDyadic,
) -> upright.UprightSE2ContinuousMaterializedEndpoint:
    """Materialize one authorized continuous endpoint without evaluating it.

    The selected point remains untrusted until the Task 8 checker performs a
    fresh replay.  This compiler seam owns only source-bound endpoint/program
    construction; it does not call a geometry kernel, search, or checker.
    """

    if type(compilation) is not upright.UprightSE2ContinuousCompilation:
        raise TypeError("continuous materialization requires continuous compilation")
    if type(translation_xy_m) is not Vec2:
        raise TypeError("continuous materialization requires an exact Vec2")
    if type(selected_lifted_yaw) is not upright.ExactDyadic:
        raise TypeError("continuous materialization requires an exact dyadic yaw")
    recipe = compilation.endpoint_construction_recipe
    selected = selected_lifted_yaw.as_fraction
    root = compilation.compiled_cells[0].yaw_interval
    if not root.lower.as_fraction <= selected <= root.upper.as_fraction:
        raise ValueError("continuous endpoint yaw is outside compiled root")
    if (
        isinstance(recipe.yaw_domain, upright.ContinuousYawFullCircle)
        and selected == root.upper.as_fraction
    ):
        raise ValueError("continuous endpoint rejects the upper full-circle seam alias")
    x = Fraction.from_float(translation_xy_m.x)
    y = Fraction.from_float(translation_xy_m.y)
    if not (
        recipe.translation_domain.x_lower.as_fraction
        <= x
        <= recipe.translation_domain.x_upper.as_fraction
        and recipe.translation_domain.y_lower.as_fraction
        <= y
        <= recipe.translation_domain.y_upper.as_fraction
    ):
        raise ValueError("continuous endpoint translation is outside compiled root")
    after_state = _continuous_after_state_template(
        recipe,
        translation_xy_m,
        selected_lifted_yaw,
    )
    source_problem = compilation.source_solve_request.semantic_problem
    after_scene = upright._materialized_expected_after_scene_state(
        compilation, after_state
    )
    program = EditProgram.seal(
        program_id="program:spatialcf/upright-se2/continuous-materialized-endpoint",
        semantic_problem_sha256=source_problem.semantic_problem_sha256,
        action_space_profile_sha256=(
            compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
        ),
        steps=(
            OperationInvocation(
                operator_ref=compilation.operation.authorization.operator_ref,
                arguments=_continuous_materialized_program_arguments(
                    compilation,
                    translation_xy_m,
                    selected_lifted_yaw,
                ),
            ),
        ),
        before_state_sha256=source_problem.scene_state.scene_state_sha256,
        after_scene_state=after_scene,
        after_scene_state_sha256=after_scene.scene_state_sha256,
        state_delta_manifest=compilation.state_footprint.state_delta_manifest,
        grounded_obligation_set_sha256=(
            compilation.grounded_obligations.grounded_obligation_set_sha256
        ),
    )
    return upright.UprightSE2ContinuousMaterializedEndpoint.seal(
        continuous_upright_se2_compilation_sha256=(
            compilation.continuous_upright_se2_compilation_sha256
        ),
        compilation=compilation,
        endpoint_construction_recipe=recipe,
        translation_xy_m=translation_xy_m,
        selected_lifted_yaw=selected_lifted_yaw,
        after_state=after_state,
        program=program,
    )


def _continuous_materialized_program_arguments(
    compilation: upright.UprightSE2ContinuousCompilation,
    translation_xy_m: Vec2,
    selected_lifted_yaw: upright.ExactDyadic,
) -> tuple[OperationArgument, ...]:
    """Use the domain's shared source-bound continuous invocation wire."""

    return upright._continuous_materialized_program_arguments(
        compilation,
        translation_xy_m,
        selected_lifted_yaw,
    )


def _continuous_after_state_template(
    recipe: upright.UprightSE2ContinuousEndpointConstructionRecipe,
    translation_xy_m: Vec2,
    selected_lifted_yaw: upright.ExactDyadic,
) -> upright.UprightSE2AfterStateTemplate:
    """Derive a continuous selected pose from a compiler-bound recipe only."""

    subject_after, yaw_after, after_pose, reference_pivot = (
        upright._continuous_materialized_expected_pose(
            recipe,
            translation_xy_m,
            selected_lifted_yaw,
        )
    )
    subject_pose = recipe.subject_before_pose
    return upright.UprightSE2AfterStateTemplate.seal(
        subject_id=recipe.subject_id,
        subject_before_pose=subject_pose,
        evaluation_scene=recipe.evaluation_scene,
        subject_pivot_xy_m=subject_after,
        subject_pivot_z_m=after_pose.translation.z,
        subject_pose=after_pose,
        subject_yaw_turns=yaw_after,
        reference_pivot_xy_m=reference_pivot,
        collision_facts=_derived_after_facts(
            recipe.evaluation_scene, recipe.subject_id, after_pose, "COLLISION"
        ),
        support_facts=_derived_after_facts(
            recipe.evaluation_scene, recipe.subject_id, after_pose, "SUPPORT"
        ),
        relation_facts=_derived_after_facts(
            recipe.evaluation_scene, recipe.subject_id, after_pose, "RELATION"
        ),
        visibility_facts=_derived_after_facts(
            recipe.evaluation_scene, recipe.subject_id, after_pose, "VISIBILITY"
        ),
    )


def _materialized_program(
    compilation: upright.UprightSE2Compilation,
    after_state: upright.UprightSE2AfterStateTemplate,
    translation_xy_m: Vec2,
) -> EditProgram:
    """Build the retained generic program from the sealed compiler roots only."""

    source_problem = compilation.source_solve_request.semantic_problem
    before_state = source_problem.scene_state
    complete_after_state = upright._materialized_expected_after_scene_state(
        compilation, after_state
    )
    return EditProgram.seal(
        program_id="program:spatialcf/upright-se2/materialized-endpoint",
        semantic_problem_sha256=source_problem.semantic_problem_sha256,
        action_space_profile_sha256=(
            compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
        ),
        steps=(
            OperationInvocation(
                operator_ref=compilation.operation.authorization.operator_ref,
                arguments=upright._materialized_program_arguments(
                    compilation, translation_xy_m
                ),
            ),
        ),
        before_state_sha256=before_state.scene_state_sha256,
        after_scene_state=complete_after_state,
        after_scene_state_sha256=complete_after_state.scene_state_sha256,
        state_delta_manifest=compilation.state_footprint.state_delta_manifest,
        grounded_obligation_set_sha256=(
            compilation.grounded_obligations.grounded_obligation_set_sha256
        ),
    )


def _after_state_template(
    recipe: upright.UprightSE2EndpointConstructionRecipe,
    translation_xy_m: Vec2,
) -> upright.UprightSE2AfterStateTemplate:
    """Derive an endpoint template only from a sealed recipe and input point."""

    subject_pose = recipe.subject_before_pose
    reference_pose = recipe.reference_before_pose
    subject = Vec2(x=subject_pose.translation.x, y=subject_pose.translation.y)
    reference = Vec2(x=reference_pose.translation.x, y=reference_pose.translation.y)
    pivot = (
        subject
        if recipe.pivot_binding.pivot_mode is upright.PivotMode.OWN
        else reference
    )
    subject_after = _yaw_then_translate(
        subject,
        pivot,
        translation_xy_m,
        recipe.quarter_turns_ccw,
    )
    yaw_after = _canonical_yaw_after(recipe.subject_yaw_turns, recipe.quarter_turns_ccw)
    after_rotation = _apply_cardinal_quaternion(
        subject_pose.rotation,
        recipe.quarter_turns_ccw,
        primary_expected_yaw=yaw_after,
    )
    after_pose = RigidTransformV2(
        translation=Vec3(
            x=subject_after.x,
            y=subject_after.y,
            z=_canonical_zero(subject_pose.translation.z),
        ),
        rotation=after_rotation,
    )
    return upright.UprightSE2AfterStateTemplate.seal(
        subject_id=recipe.subject_id,
        subject_before_pose=subject_pose,
        evaluation_scene=recipe.evaluation_scene,
        subject_pivot_xy_m=subject_after,
        subject_pivot_z_m=_canonical_zero(subject_pose.translation.z),
        subject_pose=after_pose,
        subject_yaw_turns=yaw_after,
        reference_pivot_xy_m=reference,
        collision_facts=_derived_after_facts(
            recipe.evaluation_scene,
            recipe.subject_id,
            after_pose,
            "COLLISION",
        ),
        support_facts=_derived_after_facts(
            recipe.evaluation_scene,
            recipe.subject_id,
            after_pose,
            "SUPPORT",
        ),
        relation_facts=_derived_after_facts(
            recipe.evaluation_scene,
            recipe.subject_id,
            after_pose,
            "RELATION",
        ),
        visibility_facts=_derived_after_facts(
            recipe.evaluation_scene,
            recipe.subject_id,
            after_pose,
            "VISIBILITY",
        ),
    )


def _canonical_yaw_after(
    yaw_before: upright.CanonicalSO2Angle,
    q: int,
) -> upright.CanonicalSO2Angle:
    turns = Fraction.from_float(yaw_before.turns) + _CARDINAL_TURN_FRACTIONS[q]
    while turns < Fraction(-1, 2):
        turns += 1
    while turns >= Fraction(1, 2):
        turns -= 1
    return upright.CanonicalSO2Angle(turns=_canonical_zero(float(turns)))


def _apply_cardinal_quaternion(
    rotation: Quaternion,
    q: int,
    *,
    primary_expected_yaw: upright.CanonicalSO2Angle,
) -> Quaternion:
    return upright._bound_cardinal_quaternion(
        rotation,
        q,
        primary_expected_yaw=primary_expected_yaw,
    )


def _derived_after_facts(
    scene: CanonicalScene,
    subject_id: str,
    after_pose: RigidTransformV2,
    fact_kind: str,
) -> tuple[upright.UprightSE2DerivedAfterFact, ...]:
    if fact_kind == "COLLISION":
        sources = _known_exact_source_values(scene.collision_bodies, "collision bodies")
        identifier = "body_id"
    elif fact_kind == "SUPPORT":
        subject = next(
            object_
            for object_ in _known_exact_source_values(scene.objects, "objects")
            if object_.object_id == subject_id
        )
        support_id = subject.support_assignment.surface_id
        sources = tuple(
            surface
            for surface in _known_exact_source_values(
                scene.support_surfaces,
                "support surfaces",
            )
            if surface.surface_id == support_id
        )
        identifier = "surface_id"
    elif fact_kind == "RELATION":
        sources = tuple(
            geometry
            for geometry in _known_exact_source_values(
                scene.geometry_instances,
                "geometry instances",
            )
            if geometry.role.value == "RELATION"
        )
        identifier = "geometry_id"
    elif fact_kind == "VISIBILITY":
        sources = _known_exact_source_values(
            scene.baseline_observations,
            "baseline observations",
        )
        identifier = "observation_id"
    else:
        raise ValueError("derived fact kind is not supported")
    return _sorted_bytes(
        *(
            upright.UprightSE2DerivedAfterFact(
                fact_kind=fact_kind,
                source_fact_id=getattr(source, identifier),
                source_fact_sha256=canonical_sha256(
                    source,
                    domain=upright.UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN,
                ),
                source_fact=source,
                after_subject_pose=after_pose,
            )
            for source in sources
        )
    )


def _yaw_then_translate(
    point: Vec2,
    pivot: Vec2,
    translation: Vec2,
    q: int,
) -> Vec2:
    rotated_x, rotated_y = rotate_cardinal_xy(point.x - pivot.x, point.y - pivot.y, q)
    return Vec2(
        x=_canonical_zero(pivot.x + rotated_x + translation.x),
        y=_canonical_zero(pivot.y + rotated_y + translation.y),
    )


def _state_footprint(
    solve_request: CounterfactualSolveRequest,
    subject_id: str,
) -> upright.UprightSE2StateFootprint:
    leaves = (
        solve_request.semantic_problem.scene_state.canonical_state_leaf_index.leaves
    )
    primary = _primary_write_set(subject_id)
    derived = _derived_write_set(subject_id)
    primary_bytes = {canonical_json_bytes(leaf) for leaf in primary}
    derived_bytes = {canonical_json_bytes(leaf) for leaf in derived}
    frozen = tuple(
        leaf
        for leaf in leaves
        if canonical_json_bytes(leaf) not in primary_bytes | derived_bytes
    )
    if len(primary) + len(derived) + len(frozen) != len(leaves):
        raise ValueError("complete state leaf partition is not disjoint")
    leaf_index_sha256 = solve_request.semantic_problem.scene_state.canonical_state_leaf_index.state_leaf_index_sha256
    manifest = StateDeltaManifest.seal(
        authorized_primary_writes=primary,
        recomputed_derived_writes=derived,
        unchanged_leaves_digest=canonical_sha256(
            frozen,
            domain=_UNCHANGED_LEAVES_HASH_DOMAIN,
        ),
        complete_before_leaf_index_sha256=leaf_index_sha256,
        complete_after_leaf_index_sha256=leaf_index_sha256,
    )
    return upright.UprightSE2StateFootprint.seal(
        state_delta_manifest=manifest,
        frozen_leaf_refs=frozen,
    )


def _primary_write_set(subject_id: str) -> tuple[StateVariableRef, ...]:
    return _sorted_bytes(*(_state_leaf(subject_id, role) for role in _PRIMARY_ROLES))


def _derived_write_set(subject_id: str) -> tuple[StateVariableRef, ...]:
    return _sorted_bytes(*(_state_leaf(subject_id, role) for role in _DERIVED_ROLES))


def _compiled_cell(
    operation: upright.UprightSE2CardinalOperation,
) -> upright.UprightSE2CompiledCell:
    yaw = _CARDINAL_TURN_FRACTIONS[operation.quarter_turns_ccw]
    return upright.UprightSE2CompiledCell.seal(
        cell_id=(
            f"cell:spatialcf/upright-se2/cardinal/{operation.authorization_sha256}"
        ),
        authorization_sha256=operation.authorization_sha256,
        x_lower=operation.translation_domain.x_lower,
        x_upper=operation.translation_domain.x_upper,
        y_lower=operation.translation_domain.y_lower,
        y_upper=operation.translation_domain.y_upper,
        yaw_interval=upright.LiftedYawInterval(
            lower=_dyadic_from_fraction(yaw),
            upper=_dyadic_from_fraction(yaw),
            seam_ownership="NONE",
        ),
    )


def _dyadic_from_float(value: float) -> upright.ExactDyadic:
    return _dyadic_from_fraction(Fraction.from_float(value))


def _dyadic_from_fraction(value: Fraction) -> upright.ExactDyadic:
    return upright.ExactDyadic(
        numerator=value.numerator,
        denominator=value.denominator,
    )


def _grounded_obligations(
    solve_request: CounterfactualSolveRequest,
) -> GroundedObligationSet:
    problem = solve_request.semantic_problem
    source_definition_refs = tuple(
        definition.definition_ref
        for definition in problem.definition_bundle.definitions
    )
    return GroundedObligationSet.seal(
        before_preconditions=tuple(
            GroundedObligation(
                context=context,
                source_definition_refs=source_definition_refs,
            )
            for context in problem.before_preconditions
        ),
        after_goals=(
            GroundedObligation(
                context=problem.after_goal,
                source_definition_refs=source_definition_refs,
            ),
        ),
        preservation_invariants=tuple(
            GroundedObligation(
                context=context,
                source_definition_refs=source_definition_refs,
            )
            for context in problem.preservation_invariants
        ),
        observation_obligations=tuple(
            GroundedObligation(
                context=context,
                source_definition_refs=source_definition_refs,
            )
            for context in problem.explicit_observation_obligations
        ),
        grounding_entity_sets=(),
    )


def _continuous_capability_mismatch(
    solve_request: CounterfactualSolveRequest,
) -> TypedCompilationOutcome:
    descriptor = solve_request.backend_descriptor_bundle.backend_descriptors[0]
    mismatch = CapabilityMismatch(
        backend_ref=descriptor.backend_ref,
        backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        missing_capability_refs=upright.UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS,
        reason_claim_definition_ref=_CONTINUOUS_UNAVAILABLE_REF,
    )
    selection = BackendSelectionRecord.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        implementation_registry_snapshot_sha256=(
            solve_request.implementation_registry_snapshot.implementation_registry_snapshot_sha256
        ),
        backend_descriptor_bundle_sha256=(
            solve_request.backend_descriptor_bundle.backend_descriptor_bundle_sha256
        ),
        backend_routing_policy_sha256=(
            solve_request.backend_routing_policy.backend_routing_policy_sha256
        ),
        ordered_candidate_backend_refs=(descriptor.backend_ref,),
        capability_rows=(mismatch,),
        selection_disposition="NO_SELECTION",
        selection_disposition_claim_ref=_CONTINUOUS_UNAVAILABLE_REF,
        deterministic_selection_reason_ref=_CONTINUOUS_UNAVAILABLE_REF,
    )
    resource_usage = _zero_resource_usage()
    return TypedCompilationOutcome.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        selected_backend_ref=descriptor.backend_ref,
        selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        compilation_reason_claim_definition_ref=_CONTINUOUS_UNAVAILABLE_REF,
        partial_artifact_refs=(),
        resource_usage=resource_usage,
    )


def _zero_resource_usage() -> ResourceUsage:
    return ResourceUsage(
        accounting_claim_definition_ref=(
            "definition:spatialcf/upright-se2/resource-accounting/1.0"
        ),
        entries=(
            _ResourceUsageEntry(
                resource_definition_ref="definition:spatialcf/upright-se2/resource/1.0",
                used=0.0,
            ),
        ),
        exhausted=False,
    )


def _source_requested_gap(
    source_compilation: PlanarTranslateCompilation,
) -> upright.UprightSE2ExactRational:
    """Lift the retained source binary64 gap exactly into the M3 request."""

    source_gap = source_compilation.source_artifacts.config.target_optimality_gap
    numerator, denominator = source_gap.as_integer_ratio()
    return upright.UprightSE2ExactRational(numerator=numerator, denominator=denominator)


def _solve_policy_bundle(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    requested_gap: upright.UprightSE2ExactRational,
) -> DefinitionBundle:
    return upright.build_upright_se2_solve_policy_definition_bundle(
        registration=registration,
        requested_gap=requested_gap,
        objective_bound_policy_ref=_solver_config().objective_bound_policy_ref,
    )


def _closure_definition_bundle(
    *,
    definition_ref: str,
    payload_schema_ref: str,
    registration: upright.UprightSE2ProfileRegistration,
) -> DefinitionBundle:
    return DefinitionBundle.seal(
        definitions=(
            CanonicalDefinitionEnvelope.seal(
                definition_ref=definition_ref,
                definition_kind_ref=_DEFINITION_KIND_REF,
                payload_schema_ref=payload_schema_ref,
                payload=TypedValue(
                    value_schema_ref=payload_schema_ref,
                    payload=RecordValue(
                        fields=(
                            NamedTypedValue(
                                name="profile_registration_sha256",
                                value=TypedValue(
                                    value_schema_ref=_DIGEST_SCHEMA_REF,
                                    payload=DigestValue(
                                        value=registration.profile_registration_sha256
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
            ),
        )
    )


def _implementation_registry(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    continuous: bool = False,
) -> ImplementationRegistrySnapshot:
    _ = registration
    semantic_definition_bindings = tuple(
        ImplementationOwnerBinding(
            definition_or_capability_ref=definition_ref,
            implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
        )
        for definition_ref in (
            *upright.UPRIGHT_SE2_PREDICATE_DEFINITION_REFS,
            upright.UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
            upright.UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF,
        )
    )
    return ImplementationRegistrySnapshot.seal(
        definition_and_capability_owner_bindings=_sorted_bytes(
            *semantic_definition_bindings,
            ImplementationOwnerBinding(
                definition_or_capability_ref=upright.UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
                implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
            *(
                (
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
                    ),
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
                    ),
                    ImplementationOwnerBinding(
                        definition_or_capability_ref=(
                            upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                        ),
                        implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
                    ),
                )
                if continuous
                else ()
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            ),
            ImplementationOwnerBinding(
                definition_or_capability_ref=(
                    upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF
                ),
                implementation_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
            ),
        ),
        implementation_build_hashes=_sorted_bytes(
            (
                upright.UPRIGHT_SE2_COMPILER_OWNER_REF,
                upright.UPRIGHT_SE2_COMPILER_BUILD_SHA256,
            ),
            (upright.UPRIGHT_SE2_BACKEND_OWNER_REF, _BACKEND_BUILD_SHA256),
            (upright.UPRIGHT_SE2_CHECKER_OWNER_REF, _CHECKER_BUILD_SHA256),
        ),
        dependency_lock_sha256=_DEPENDENCY_LOCK_SHA256,
    )


def _backend_descriptor_bundle(
    registration: upright.UprightSE2ProfileRegistration,
    *,
    continuous: bool = False,
) -> BackendDescriptorBundle:
    descriptor = SolverBackendDescriptor.seal(
        backend_ref=_CONTINUOUS_BACKEND_REF if continuous else _BACKEND_REF,
        implementation_build_sha256=_BACKEND_BUILD_SHA256,
        supported_profile_hashes=(
            registration.action_space_profile.action_space_profile_sha256,
        ),
        supported_predicate_capabilities=(
            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        ),
        supported_operator_capabilities=(
            upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF
            if continuous
            else upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
        ),
        supported_objective_capabilities=(
            upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
        ),
        supported_numeric_semantics=(
            registration.semantics_profile.numeric_semantics_ref,
        ),
        emitted_proof_material_definition_refs=(
            upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
            if continuous
            else upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF,
        ),
        compatible_checker_capability_refs=_sorted_bytes(
            (
                upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                if continuous
                else upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            ),
            upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
            upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        ),
        resource_definition_refs=("definition:spatialcf/upright-se2/resource/1.0",),
    )
    return BackendDescriptorBundle.seal(
        backend_descriptors=(descriptor,),
        unavailable_optional_backends=(),
    )


def _solver_config() -> CounterfactualSolverConfig:
    return CounterfactualSolverConfig.seal(
        solver_config_ref="definition:spatialcf/upright-se2/solver-config/1.0",
        compilation_policy_ref="definition:spatialcf/upright-se2/compile/1.0",
        proposal_policy_ref="definition:spatialcf/upright-se2/proposal/1.0",
        objective_bound_policy_ref="definition:spatialcf/upright-se2/objective-bound/1.0",
        determinism_policy_ref="definition:spatialcf/upright-se2/determinism/1.0",
    )


def _proof_policy(*, continuous: bool = False) -> ProofPolicy:
    return ProofPolicy.seal(
        proof_policy_ref="definition:spatialcf/upright-se2/proof-policy/1.0",
        accepted_claim_definition_refs=_sorted_bytes(
            upright.UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF,
            upright.UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
        ),
        required_checker_capability_refs=(
            (
                upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
                if continuous
                else upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            ),
        ),
        publication_minimum_claim_ref=(
            "definition:spatialcf/upright-se2/claim-certified-solution/1.0"
        ),
        permit_noncertified_terminal_records=True,
    )


def _resource_policy() -> ResourcePolicy:
    return ResourcePolicy.seal(
        resource_policy_ref="definition:spatialcf/upright-se2/resource-policy/1.0",
        limits=(
            ResourceLimit(
                definition_ref="definition:spatialcf/upright-se2/resource/1.0",
                finite_limit=1.0,
            ),
        ),
        exhaustion_claim_ref="definition:spatialcf/upright-se2/resource-exhausted/1.0",
        shared_ledger_policy_ref="definition:spatialcf/upright-se2/shared-ledger/1.0",
    )


def _validate_request_resource_policy(resource_policy: ResourcePolicy) -> None:
    """Keep resource ownership fixed while leaving its request cap executable."""

    registered = _resource_policy()
    if (
        resource_policy.resource_policy_ref != registered.resource_policy_ref
        or resource_policy.exhaustion_claim_ref != registered.exhaustion_claim_ref
        or resource_policy.shared_ledger_policy_ref
        != registered.shared_ledger_policy_ref
        or tuple(limit.definition_ref for limit in resource_policy.limits)
        != tuple(limit.definition_ref for limit in registered.limits)
    ):
        raise ValueError("resource policy does not match the upright se2 closure")


def _backend_routing_policy() -> BackendRoutingPolicy:
    return BackendRoutingPolicy.seal(
        routing_policy_ref="definition:spatialcf/upright-se2/routing/1.0",
        capability_filter_definition_ref="definition:spatialcf/upright-se2/filter/1.0",
        deterministic_order_definition_ref="definition:spatialcf/upright-se2/order/1.0",
        portfolio_composition_definition_ref="definition:spatialcf/upright-se2/portfolio/1.0",
        stop_condition_definition_ref="definition:spatialcf/upright-se2/stop/1.0",
        resource_partition_definition_ref="definition:spatialcf/upright-se2/partition/1.0",
    )


def _sorted_bytes(*values):
    return tuple(sorted(values, key=canonical_json_bytes))
