"""Immutable Task 1 contracts for the ``spatialcf/upright_se2@1`` profile.

This module owns only declarative profile, authorization, and proof-wire data.
It does not compile a scene, transform geometry, invoke a kernel, select a
backend, or solve a counterfactual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, ClassVar, Literal, Self, TypeAlias, TypeVar

from pydantic import Field, StrictBool, StrictInt, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    FiniteFloat,
    Quaternion,
    RigidTransformV2,
    Sha256Digest,
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
    CapabilityRef,
    DefinitionBundle,
    DefinitionRef,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntegerValue,
    IntervalValue,
    NamedTypedValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueKind,
)
from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryInstanceV2,
    GeometryRoleV2,
)
from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
    StateDeltaManifest,
    StateVariableRef,
)
from spatialcf.domain.outcomes import ResourceUsage
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
    ObjectiveExpression,
    ObjectiveTerm,
    OwnerRef,
    ProofPolicy,
    ResourcePolicy,
    SemanticsProfile,
)
from spatialcf.domain.scene import (
    BaselineObservation,
    CanonicalScene,
    ObjectPose,
    SupportSurfaceFact,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = (
    "UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF",
    "UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR",
    "UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF",
    "UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN",
    "UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF",
    "UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR",
    "UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF",
    "UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF",
    "UPRIGHT_SE2_YAW_TO_POSE_RULE_REF",
    "CanonicalSO2Angle",
    "CardinalYaw",
    "CardinalYawAuthorization",
    "ContinuousYawArc",
    "ContinuousYawAuthorization",
    "ContinuousYawDomain",
    "ContinuousYawFullCircle",
    "ContinuousYawLift",
    "ExactDyadic",
    "ExplicitPoseYawBinding",
    "FixedPivotBinding",
    "LiftedYawInterval",
    "PivotMode",
    "UprightSE2AfterStateTemplate",
    "UprightSE2BackendAvailability",
    "UprightSE2CardinalOperation",
    "UprightSE2CardinalProofTuple",
    "UprightSE2CheckerReplayPolicy",
    "UprightSE2Compilation",
    "UprightSE2CompiledCell",
    "UprightSE2CompilerClosure",
    "UprightSE2ContinuousCompilation",
    "UprightSE2ContinuousEndpointConstructionRecipe",
    "UprightSE2ContinuousMaterializedEndpoint",
    "UprightSE2ContinuousOperation",
    "UprightSE2ContinuousProofMaterial",
    "UprightSE2ContinuousProofTuple",
    "UprightSE2ContinuousProposalCandidate",
    "UprightSE2ContinuousSemanticClosure",
    "UprightSE2CoverageArtifact",
    "UprightSE2DerivedAfterFact",
    "UprightSE2EndpointConstructionRecipe",
    "UprightSE2ExactRational",
    "UprightSE2ExecutablePolicyBundle",
    "UprightSE2ExecutablePolicyValue",
    "UprightSE2FiveTermObjectivePolicy",
    "UprightSE2M2Q0Construction",
    "UprightSE2M2Q0MappingDefinition",
    "UprightSE2M2Q0MappingRow",
    "UprightSE2M2Q0SourceFreeConstructionRow",
    "UprightSE2MaterializedEndpoint",
    "UprightSE2ObjectiveTermPolicy",
    "UprightSE2ProfileRegistration",
    "UprightSE2ProofCellEvaluation",
    "UprightSE2ProofFrontierRow",
    "UprightSE2ProofLeafDisposition",
    "UprightSE2ProofMaterial",
    "UprightSE2ProofPruneDecision",
    "UprightSE2ProofResourceLedger",
    "UprightSE2ProofStageDelta",
    "UprightSE2ProposalCandidate",
    "UprightSE2ProposalPointEvaluation",
    "UprightSE2ProposalPointObjective",
    "UprightSE2ProposalPointTerm",
    "UprightSE2RetainedOwnerEvaluation",
    "UprightSE2RetainedOwnerOutcomeKind",
    "UprightSE2SemanticClosure",
    "UprightSE2SemanticDefinition",
    "UprightSE2SemanticOwnerBinding",
    "UprightSE2SolvePolicyDefinitionPayload",
    "UprightSE2StateFootprint",
    "UprightSE2TranslationDomain",
    "UprightSE2VerificationBundle",
    "build_upright_se2_checker_replay_policy",
    "build_upright_se2_continuous_semantic_closure",
    "build_upright_se2_executable_policy_bundle",
    "build_upright_se2_objective_expression",
    "build_upright_se2_q0_target_objective_policy",
    "build_upright_se2_semantic_closure",
    "build_upright_se2_semantic_definition_bundle",
    "build_upright_se2_solve_policy_definition_bundle",
    "canonical_yaw_from_upright_quaternion",
    "decode_upright_se2_continuous_proof_material",
    "decode_upright_se2_objective_policy",
    "decode_upright_se2_proof_material",
    "decode_upright_se2_solve_policy_definition_payload",
    "encode_upright_se2_continuous_proof_material",
    "encode_upright_se2_proof_material",
    "request_bound_executable_policy_bundle_from_problem",
    "validate_directed_yaw_quaternion_consistency",
    "validate_required_upright_support_surface",
    "validate_upright_se2_executable_policy_visibility_binding",
)


UPRIGHT_SE2_PROFILE_REF = "spatialcf/upright_se2@1"
UPRIGHT_SE2_SEMANTICS_PROFILE_REF = "spatialcf/upright_se2/semantics@1"
UPRIGHT_SE2_YAW_TO_POSE_RULE_REF = "definition:spatialcf/upright-se2/yaw-to-pose/1.0"
UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/derived-source-fact/3.0"
)
UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/proof-material/1.0"
)
UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/cardinal-proof-material/1.0"
)
UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR = "UPRIGHT_SE2_CARDINAL_PROOF_MATERIAL_V1"
UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/continuous-proof-material/1.0"
)
UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/continuous-proof-material/1.0"
)
UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR = (
    "UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_V1"
)
UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/solve-policy/1.0"
)
UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/solve-policy-closure/1.0"
)
UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/claim-certified-solution/1.0"
)
UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/claim-finite-gap-solution/1.0"
)
_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/pivot-state/3.0"
)
_UPRIGHT_SE2_UNCHANGED_LEAVES_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/unchanged-leaves/3.0"
)
_UPRIGHT_SE2_YAW_TO_POSE_MAX_ULPS = 8
_UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF = (
    "definition:spatialcf/upright-se2/compiler-input/1.0"
)
_UPRIGHT_SE2_COMPILER_INPUT_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/compiler-input/1.0"
)
_UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY = "fact-key:spatialcf/upright-se2/compiler-input"
_UPRIGHT_SE2_STATE_FAMILY_REF = "definition:spatialcf/upright-se2/state/1.0"
_UPRIGHT_SE2_STATE_SCHEMA_REF = "schema:spatialcf/upright-se2/state/1.0"
_UPRIGHT_SE2_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/finite-real/1.0"
_UPRIGHT_SE2_INTEGER_SCHEMA_REF = "schema:spatialcf/upright-se2/integer/1.0"
_UPRIGHT_SE2_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/canonical-id/1.0"
_UPRIGHT_SE2_DIGEST_SCHEMA_REF = "schema:spatialcf/upright-se2/digest/1.0"
_UPRIGHT_SE2_ENUM_SCHEMA_REF = "schema:spatialcf/upright-se2/enum-symbol/1.0"
_UPRIGHT_SE2_YAW_ARGUMENT_SCHEMA_REF = "schema:spatialcf/upright-se2/yaw-argument/1.0"
_UPRIGHT_SE2_WORLD_XY_FRAME_REF = "definition:spatialcf/upright-se2/world-xy/1.0"
_UPRIGHT_SE2_METRE_UNIT_REF = "definition:spatialcf/upright-se2/metre/1.0"
_UPRIGHT_SE2_CLOSED_INTERVAL_TOPOLOGY_REF = (
    "definition:spatialcf/upright-se2/closed-interval/1.0"
)
_UPRIGHT_SE2_DERIVED_RULE_REF = (
    "definition:spatialcf/upright-se2/derived-pose-and-facts/1.0"
)
_UPRIGHT_SE2_PRIMARY_ROLES = (
    "subject-world-x",
    "subject-world-y",
    "subject-explicit-yaw",
)
_UPRIGHT_SE2_DERIVED_ROLES = (
    "subject-derived-canonical-pose",
    "subject-derived-collision",
    "subject-derived-support",
    "subject-derived-relation",
    "subject-derived-visibility",
)
_UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS = {
    0: Fraction(0),
    1: Fraction(1, 4),
    2: Fraction(-1, 2),
    3: Fraction(-1, 4),
}
_UPRIGHT_SE2_CARDINAL_QUATERNIONS = {
    0: (0.0, 1.0),
    1: (0.7071067811865476, 0.7071067811865476),
    2: (1.0, 0.0),
    3: (-0.7071067811865476, 0.7071067811865476),
}

UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/cardinal-own-pivot/1.0"
)
UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/cardinal-reference-pivot/1.0"
)
UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/continuous-own-pivot/1.0"
)
UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF = (
    "definition:spatialcf/upright-se2/continuous-reference-pivot/1.0"
)

UPRIGHT_SE2_OPERATOR_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
        ),
        key=canonical_json_bytes,
    )
)

UPRIGHT_SE2_PROFILE_CAPABILITY_REF = "capability:spatialcf/upright-se2/profile/1"
UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/compiler/1"
)
UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/backend/1"
)
UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/cardinal/checker/1"
)
UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/compiler/1"
)
UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/backend/1"
)
UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/continuous/checker/1"
)
UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)
UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)
UPRIGHT_SE2_STAGED_CAPABILITY_REFS = tuple(
    sorted(
        (
            *UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
            *UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS,
        ),
        key=canonical_json_bytes,
    )
)
UPRIGHT_SE2_BACKEND_OWNER_REF = "owner:spatialcf/upright-se2/backend"
UPRIGHT_SE2_CHECKER_OWNER_REF = "owner:spatialcf/upright-se2/checker"
UPRIGHT_SE2_COMPILER_OWNER_REF = "owner:spatialcf/upright-se2/compiler"
UPRIGHT_SE2_COMPILER_BUILD_SHA256 = "c" * 64
UPRIGHT_SE2_CHECKER_BUILD_SHA256 = "a" * 64

UPRIGHT_SE2_COLLISION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/collision-clearance-contact/1.0"
)
UPRIGHT_SE2_SUPPORT_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/same-surface-support/1.0"
)
UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/target-relation/1.0"
)
UPRIGHT_SE2_PRESERVATION_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/preservation/1.0"
)
UPRIGHT_SE2_VISIBILITY_PREDICATE_REF = (
    "definition:spatialcf/upright-se2/fixed-camera-visibility/1.0"
)
UPRIGHT_SE2_PREDICATE_DEFINITION_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_COLLISION_PREDICATE_REF,
            UPRIGHT_SE2_SUPPORT_PREDICATE_REF,
            UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
            UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
            UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
        ),
        key=canonical_json_bytes,
    )
)
UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/objective-five-term/1.0"
)
UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-predicate-evaluator/1"
)
UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-predicate-verifier/1"
)
UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-objective-evaluator/1"
)
UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF = (
    "capability:spatialcf/upright-se2/semantic-objective-verifier/1"
)
UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)
UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS = tuple(
    sorted(
        (
            UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        ),
        key=canonical_json_bytes,
    )
)

_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True)
class _UprightSE2SourceInput:
    """The exact cardinal compiler-input wire carried by a bound solve request."""

    operator_ref: DefinitionRef
    subject_id: CanonicalId
    reference_id: CanonicalId
    subject_yaw_turns: CanonicalSO2Angle
    quarter_turns_ccw: int


@dataclass(frozen=True)
class _UprightSE2ContinuousSourceInput:
    """The exact continuous compiler-input wire bound by the additive closure."""

    operator_ref: DefinitionRef
    subject_id: CanonicalId
    reference_id: CanonicalId
    subject_yaw_turns: CanonicalSO2Angle
    yaw_domain: ContinuousYawDomain


def _require_sorted_unique_by_bytes(values: tuple[_ValueT, ...], label: str) -> None:
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _fraction_from_float(value: float) -> Fraction:
    return Fraction.from_float(value)


def _m2_q0_source_leaf_values(
    source_compilation: PlanarTranslateCompilation,
) -> dict[str, object]:
    """Return every retained M2 leaf under its canonical provenance selector."""

    values: dict[str, object] = {}

    def visit(value: object, selector: str) -> None:
        if type(value) is dict:
            if not value:
                values[selector] = value
                return
            for key in sorted(value, key=canonical_json_bytes):
                visit(value[key], f"{selector}/{key}")
            return
        if type(value) in (list, tuple):
            if not value:
                values[selector] = value
                return
            for index, item in enumerate(value):
                visit(item, f"{selector}/{index}")
            return
        values[selector] = value

    visit(
        source_compilation.model_dump(mode="python", round_trip=True),
        "source:compilation",
    )
    return values


def _m2_q0_source_leaf_selectors(
    source_compilation: PlanarTranslateCompilation,
) -> frozenset[str]:
    """Return every retained M2 leaf selector in canonical source-tree order."""

    return frozenset(_m2_q0_source_leaf_values(source_compilation))


class CanonicalSO2Angle(CanonicalModel):
    """One canonical finite binary64 turn value in the half-open SO(2) wire."""

    turns: FiniteFloat

    @model_validator(mode="after")
    def _validate_turns(self) -> Self:
        if _is_negative_zero(self.turns):
            raise ValueError("negative zero turns are not canonical")
        if not -0.5 <= self.turns < 0.5:
            raise ValueError("canonical turns must be in [-0.5, 0.5)")
        if self.turns == 0.0:
            object.__setattr__(self, "turns", 0.0)
        return self


def canonical_yaw_from_upright_quaternion(rotation: Quaternion) -> CanonicalSO2Angle:
    """Derive the registered directed SO(2) view of one yaw-only quaternion.

    The cardinal compiler never uses this conversion for a cardinal transform:
    it is solely the profile-owned consistency view for an already-normalized
    base or endpoint quaternion.  The ``Quaternion`` contract has already
    selected the unique sign representative, so the half-turn seam maps to
    canonical ``-0.5`` rather than admitting a second directed wire value.
    """

    if rotation.x != 0.0 or rotation.y != 0.0:
        raise ValueError("directed yaw quaternion must be yaw-only")
    turns = math.atan2(rotation.z, rotation.w) / math.pi
    if turns == 0.5:
        turns = -0.5
    if not -0.5 <= turns < 0.5:
        raise ValueError("directed yaw quaternion is ambiguous")
    return CanonicalSO2Angle(turns=0.0 if turns == 0.0 else turns)


def validate_directed_yaw_quaternion_consistency(
    explicit_yaw: CanonicalSO2Angle,
    rotation: Quaternion,
) -> None:
    """Enforce the registered binary64 yaw-to-quaternion consistency rule.

    The canonical directed yaw remains primary state.  Its quaternion view is
    recovered with the deterministic ``atan2(z, w) / pi`` rule, then compared
    in the SO(2) quotient using a bounded binary64 reconstruction allowance.
    The allowance covers the final rounding of a normalized stored quaternion;
    it never classifies a non-cardinal angle as cardinal.
    """

    derived_yaw = canonical_yaw_from_upright_quaternion(rotation)
    distance = abs(explicit_yaw.turns - derived_yaw.turns)
    distance = min(distance, 1.0 - distance)
    tolerance = _UPRIGHT_SE2_YAW_TO_POSE_MAX_ULPS * max(
        math.ulp(explicit_yaw.turns),
        math.ulp(derived_yaw.turns),
    )
    if distance > tolerance:
        raise ValueError("pose yaw does not match the frozen directed base pose")


def _compose_upright_quaternion_from_primary_yaw(
    *,
    rotation: Quaternion,
    delta_z: float,
    delta_w: float,
    primary_expected_yaw: CanonicalSO2Angle,
) -> Quaternion:
    """Build one derived yaw view while keeping its canonical yaw primary."""

    composed = Quaternion(
        x=0.0,
        y=0.0,
        z=(
            0.0
            if rotation.z * delta_w + rotation.w * delta_z == 0.0
            else rotation.z * delta_w + rotation.w * delta_z
        ),
        w=(
            0.0
            if rotation.w * delta_w - rotation.z * delta_z == 0.0
            else rotation.w * delta_w - rotation.z * delta_z
        ),
    )
    try:
        validate_directed_yaw_quaternion_consistency(primary_expected_yaw, composed)
    except ValueError:
        if primary_expected_yaw.turns == 0.0:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[0]
        elif primary_expected_yaw.turns == 0.25:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[1]
        elif primary_expected_yaw.turns == -0.5:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[2]
        elif primary_expected_yaw.turns == -0.25:
            z, w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[3]
        else:
            half_radians = math.pi * primary_expected_yaw.turns
            z = math.sin(half_radians)
            w = math.cos(half_radians)
        reconstructed = Quaternion(x=0.0, y=0.0, z=z, w=w)
        validate_directed_yaw_quaternion_consistency(
            primary_expected_yaw,
            reconstructed,
        )
        return reconstructed
    return composed


class CardinalYaw(CanonicalModel):
    """The exact cardinal yaw wire, never a coerced floating-point turn."""

    q: Annotated[StrictInt, Field(ge=0, le=3)]


class ContinuousYawArc(CanonicalModel):
    """One closed directed SO(2) arc without a wire-level branch-cut split."""

    kind: Literal["ARC"] = "ARC"
    start_angle: CanonicalSO2Angle
    ccw_sweep_turns: FiniteFloat

    @model_validator(mode="after")
    def _validate_sweep(self) -> Self:
        if _is_negative_zero(self.ccw_sweep_turns):
            raise ValueError("negative zero sweep is not canonical")
        if not 0.0 <= self.ccw_sweep_turns < 1.0:
            raise ValueError("ccw sweep turns must be in [0, 1)")
        if self.ccw_sweep_turns == 0.0:
            object.__setattr__(self, "ccw_sweep_turns", 0.0)
        return self

    @property
    def is_closed_point(self) -> bool:
        return self.ccw_sweep_turns == 0.0

    @property
    def crosses_branch_cut(self) -> bool:
        start = _fraction_from_float(self.start_angle.turns)
        upper = start + _fraction_from_float(self.ccw_sweep_turns)
        return start < Fraction(1, 2) < upper

    @property
    def contains_canonical_zero(self) -> bool:
        start = _fraction_from_float(self.start_angle.turns)
        upper = start + _fraction_from_float(self.ccw_sweep_turns)
        return any(
            start <= representative <= upper
            for representative in (Fraction(0), Fraction(1))
        )


class ContinuousYawFullCircle(CanonicalModel):
    """The sole full-SO(2) branch; a sweep of one turn is not an alias."""

    kind: Literal["FULL_CIRCLE"] = "FULL_CIRCLE"


ContinuousYawDomain: TypeAlias = Annotated[
    ContinuousYawArc | ContinuousYawFullCircle,
    Field(discriminator="kind"),
]


class ExactDyadic(CanonicalModel):
    """A normalized exact dyadic rational for a lifted turn endpoint."""

    numerator: StrictInt
    denominator: Annotated[StrictInt, Field(gt=0)]

    @model_validator(mode="after")
    def _validate_normalized_dyadic(self) -> Self:
        if self.denominator & (self.denominator - 1):
            raise ValueError("dyadic denominator must be a power of two")
        if self.numerator == 0 and self.denominator != 1:
            raise ValueError("zero dyadic must use denominator one")
        if self.denominator != 1 and self.numerator % 2 == 0:
            raise ValueError("dyadic numerator and denominator must be normalized")
        return self

    @property
    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


class LiftedYawInterval(CanonicalModel):
    """One closed exact-dyadic interval in the fixed ``-0.5`` proof lift."""

    lower: ExactDyadic
    upper: ExactDyadic
    endpoint_closure: Literal["CLOSED"] = "CLOSED"
    seam_ownership: Literal["NONE", "LOWER_OWNS_SEAM", "UPPER_OWNS_ENDPOINT"]

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        if self.lower.as_fraction > self.upper.as_fraction:
            raise ValueError("lifted interval lower endpoint must not exceed upper")
        return self


class ContinuousYawLift(HashBoundCanonicalModel):
    """The one canonical non-wrapping lifted interval closure for a yaw domain."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-yaw-lift/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_yaw_lift_sha256"

    yaw_domain: ContinuousYawDomain
    lift_origin: ExactDyadic
    intervals: tuple[LiftedYawInterval, ...]
    continuous_yaw_lift_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_canonical_lift(self) -> Self:
        if self.lift_origin.as_fraction != Fraction(-1, 2):
            raise ValueError("lift origin must be exactly -0.5 turns")
        if len(self.intervals) != 1:
            raise ValueError("a canonical yaw domain has exactly one lifted interval")
        interval = self.intervals[0]
        if isinstance(self.yaw_domain, ContinuousYawFullCircle):
            if (
                interval.lower.as_fraction,
                interval.upper.as_fraction,
                interval.seam_ownership,
            ) != (Fraction(-1, 2), Fraction(1, 2), "LOWER_OWNS_SEAM"):
                raise ValueError("full-circle lift must use the lower-owned seam")
            return self
        start = _fraction_from_float(self.yaw_domain.start_angle.turns)
        upper = start + _fraction_from_float(self.yaw_domain.ccw_sweep_turns)
        expected_seam = "UPPER_OWNS_ENDPOINT" if upper == Fraction(1, 2) else "NONE"
        if (interval.lower.as_fraction, interval.upper.as_fraction) != (start, upper):
            raise ValueError("arc lift endpoints must be exact dyadic unrolled values")
        if interval.seam_ownership != expected_seam:
            raise ValueError("arc seam ownership is not canonical")
        return self


class PivotMode(StrEnum):
    """The only two endpoint pivot bindings admitted by this profile."""

    OWN = "OWN"
    REFERENCE = "REFERENCE"


class FixedPivotBinding(HashBoundCanonicalModel):
    """A hash-bound frozen subject or named reference object pivot."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/fixed-pivot-binding/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "fixed_pivot_binding_sha256"

    subject_id: CanonicalId
    pivot_mode: PivotMode
    pivot_entity_id: CanonicalId
    pivot_state_sha256: Sha256Digest
    fixed_pivot_binding_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Seal the canonical JSON pivot-mode symbol as its closed enum value."""

        pivot_mode = values.get("pivot_mode")
        if isinstance(pivot_mode, str):
            values = {**values, "pivot_mode": PivotMode(pivot_mode)}
        return super().seal(**values)

    @model_validator(mode="after")
    def _validate_pivot_identity(self) -> Self:
        if self.pivot_mode is PivotMode.OWN and self.pivot_entity_id != self.subject_id:
            raise ValueError("own pivot must be the exact subject entity")
        if (
            self.pivot_mode is PivotMode.REFERENCE
            and self.pivot_entity_id == self.subject_id
        ):
            raise ValueError("reference pivot must not name the subject")
        return self


class ExplicitPoseYawBinding(HashBoundCanonicalModel):
    """The explicit primary yaw fact and its hash-bound derived base-pose view."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/explicit-pose-yaw-binding/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "pose_yaw_binding_sha256"

    entity_id: CanonicalId
    base_pose_sha256: Sha256Digest
    explicit_yaw: CanonicalSO2Angle
    yaw_to_pose_rule_ref: DefinitionRef
    pose_yaw_binding_sha256: Sha256Digest


class CardinalYawAuthorization(HashBoundCanonicalModel):
    """One exact cardinal-yaw authorization without a hidden reference pivot."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/cardinal-yaw-authorization/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "cardinal_yaw_authorization_sha256"

    subject_id: CanonicalId
    operator_ref: DefinitionRef
    pivot_binding: FixedPivotBinding
    yaw: CardinalYaw
    cardinal_yaw_authorization_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_cardinal_authorization(self) -> Self:
        if self.subject_id != self.pivot_binding.subject_id:
            raise ValueError("authorization subject must match fixed pivot subject")
        if self.operator_ref == UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.OWN:
                raise ValueError("own-pivot operator requires an own pivot")
        elif self.operator_ref == UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.REFERENCE:
                raise ValueError("reference-pivot operator requires a reference pivot")
            if self.yaw.q == 0:
                raise ValueError("reference-pivot zero yaw is not canonical")
        else:
            raise ValueError(
                "cardinal authorization requires a registered cardinal operator"
            )
        return self


class ContinuousYawAuthorization(HashBoundCanonicalModel):
    """One continuous-yaw authorization with full-circle and zero rules closed."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-yaw-authorization/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_yaw_authorization_sha256"

    subject_id: CanonicalId
    operator_ref: DefinitionRef
    pivot_binding: FixedPivotBinding
    yaw_domain: ContinuousYawDomain
    continuous_yaw_authorization_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_authorization(self) -> Self:
        if self.subject_id != self.pivot_binding.subject_id:
            raise ValueError("authorization subject must match fixed pivot subject")
        if self.operator_ref == UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.OWN:
                raise ValueError("own-pivot operator requires an own pivot")
        elif self.operator_ref == UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF:
            if self.pivot_binding.pivot_mode is not PivotMode.REFERENCE:
                raise ValueError("reference-pivot operator requires a reference pivot")
            if isinstance(self.yaw_domain, ContinuousYawFullCircle):
                raise ValueError(
                    "full circle is valid only for continuous own-pivot authorization"
                )
            if self.yaw_domain.contains_canonical_zero:
                raise ValueError("reference-pivot arc must exclude canonical zero")
        else:
            raise ValueError(
                "continuous authorization requires a registered continuous operator"
            )
        return self


class UprightSE2ProfileRegistration(HashBoundCanonicalModel):
    """One immutable semantic/action profile plus staged static capabilities."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/profile-registration/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "profile_registration_sha256"

    semantics_profile: SemanticsProfile
    action_space_profile: ActionSpaceProfile
    backend_owner_ref: OwnerRef
    checker_owner_ref: OwnerRef
    profile_capability_ref: CapabilityRef
    cardinal_compiler_capability_ref: CapabilityRef
    cardinal_backend_capability_ref: CapabilityRef
    cardinal_checker_capability_ref: CapabilityRef
    continuous_compiler_capability_ref: CapabilityRef
    continuous_backend_capability_ref: CapabilityRef
    continuous_checker_capability_ref: CapabilityRef
    profile_registration_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_immutable_profile(self) -> Self:
        if (
            self.semantics_profile.semantics_profile_ref
            != UPRIGHT_SE2_SEMANTICS_PROFILE_REF
        ):
            raise ValueError("semantics profile reference must be fixed")
        if (
            self.action_space_profile.action_space_profile_ref
            != UPRIGHT_SE2_PROFILE_REF
        ):
            raise ValueError("action-space profile reference must be fixed")
        if (
            self.semantics_profile.transition_semantics_refs
            != UPRIGHT_SE2_OPERATOR_REFS
        ):
            raise ValueError(
                "semantics profile must name exactly four upright operators"
            )
        if self.action_space_profile.allowed_operator_refs != UPRIGHT_SE2_OPERATOR_REFS:
            raise ValueError(
                "action-space profile must name exactly four upright operators"
            )
        if (
            self.action_space_profile.backend_capability_requirements
            != UPRIGHT_SE2_STAGED_CAPABILITY_REFS
        ):
            raise ValueError(
                "action-space profile must retain both staged capability sets"
            )
        if self.backend_owner_ref != UPRIGHT_SE2_BACKEND_OWNER_REF:
            raise ValueError("backend owner reference must be fixed")
        if self.checker_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF:
            raise ValueError("checker owner reference must be fixed")
        if self.backend_owner_ref == self.checker_owner_ref:
            raise ValueError("backend and checker owners must be distinct")
        expected_capabilities = (
            UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
        )
        actual_capabilities = (
            self.profile_capability_ref,
            self.cardinal_compiler_capability_ref,
            self.cardinal_backend_capability_ref,
            self.cardinal_checker_capability_ref,
            self.continuous_compiler_capability_ref,
            self.continuous_backend_capability_ref,
            self.continuous_checker_capability_ref,
        )
        if actual_capabilities != expected_capabilities:
            raise ValueError("profile capability references must be fixed")
        return self


class UprightSE2SemanticOwnerBinding(CanonicalModel):
    """The static evaluator/verifier ownership for one semantic definition."""

    definition_ref: DefinitionRef
    evaluator_capability_ref: CapabilityRef
    verifier_capability_ref: CapabilityRef
    evaluator_owner_ref: OwnerRef
    verifier_owner_ref: OwnerRef
    evaluator_build_sha256: Sha256Digest
    verifier_build_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_owner_binding(self) -> Self:
        if self.evaluator_owner_ref != UPRIGHT_SE2_BACKEND_OWNER_REF:
            raise ValueError("semantic evaluator owner must be the upright backend")
        if self.verifier_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF:
            raise ValueError("semantic verifier owner must be the upright checker")
        if self.evaluator_build_sha256 != "b" * 64:
            raise ValueError("semantic evaluator build must be fixed")
        if self.verifier_build_sha256 != "a" * 64:
            raise ValueError("semantic verifier build must be fixed")
        expected_capabilities = _semantic_capabilities_for_definition(
            self.definition_ref
        )
        if (
            self.evaluator_capability_ref,
            self.verifier_capability_ref,
        ) != expected_capabilities:
            raise ValueError("semantic evaluator/verifier capabilities must be fixed")
        return self


class UprightSE2SemanticDefinition(HashBoundCanonicalModel):
    """One typed, versioned predicate body with its executable ownership."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/semantic-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_definition_sha256"

    definition_ref: DefinitionRef
    definition_kind: Literal[
        "COLLISION",
        "SUPPORT",
        "TARGET_RELATION",
        "PRESERVATION",
        "VISIBILITY",
    ]
    payload_schema_ref: CanonicalId
    semantic_body: TypedValue
    owner_binding: UprightSE2SemanticOwnerBinding
    semantic_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_concrete_semantic_body(self) -> Self:
        expected_ref = _semantic_definition_ref_for_kind(self.definition_kind)
        if self.definition_ref != expected_ref:
            raise ValueError(
                "semantic definition reference does not match its body kind"
            )
        expected_schema = _semantic_definition_schema_for_kind(self.definition_kind)
        if self.payload_schema_ref != expected_schema:
            raise ValueError("semantic definition schema does not match its body kind")
        if self.owner_binding.definition_ref != self.definition_ref:
            raise ValueError("semantic owner binding must bind its semantic definition")
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.definition_ref)
        ):
            raise ValueError(
                "semantic definition owner binding does not match the frozen policy"
            )
        expected_body = _semantic_definition_body(self.definition_kind)
        if canonical_json_bytes(self.semantic_body) != canonical_json_bytes(
            expected_body
        ):
            raise ValueError(
                "semantic definition body does not match the frozen upright policy"
            )
        return self


class UprightSE2ObjectiveTermPolicy(CanonicalModel):
    """One concrete term of the fixed five-term M3 objective."""

    term_id: Literal["T", "A", "R", "V", "S"]
    objective_definition_ref: DefinitionRef
    metric_definition_ref: DefinitionRef
    input_selector_definition_ref: DefinitionRef
    unit_ref: DefinitionRef
    normalization_definition_ref: DefinitionRef
    normalizer_unit_ref: DefinitionRef
    weight: FiniteFloat
    normalizer: FiniteFloat

    @model_validator(mode="after")
    def _validate_term_policy(self) -> Self:
        expected = _objective_term_policy_values(self.term_id)
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    "objective term policy does not match the frozen upright policy"
                )
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("objective weights must be finite and non-negative")
        if self.term_id in ("T", "A") and self.weight <= 0.0:
            raise ValueError(
                "translation and angular objective weights must be positive"
            )
        if not math.isfinite(self.normalizer) or self.normalizer <= 0.0:
            raise ValueError(
                "objective normalizers must be finite and strictly positive"
            )
        if self.normalizer_unit_ref != self.unit_ref:
            raise ValueError("objective normalizer units must match term units")
        return self


class UprightSE2FiveTermObjectivePolicy(HashBoundCanonicalModel):
    """The complete non-ambient objective, comparison, tie, and prune policy."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/five-term-objective-policy/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "five_term_objective_policy_sha256"

    objective_definition_ref: DefinitionRef
    aggregation_definition_ref: DefinitionRef
    safety_penalty_definition_ref: DefinitionRef
    strict_interval_comparator_ref: DefinitionRef
    exact_equality_definition_ref: DefinitionRef
    directed_gap_subtraction_definition_ref: DefinitionRef
    exact_prune_comparator_ref: DefinitionRef
    gap_prune_comparator_ref: DefinitionRef
    interval_boundary_policy_ref: DefinitionRef
    deterministic_tie_break_definition_ref: DefinitionRef
    terms: tuple[UprightSE2ObjectiveTermPolicy, ...]
    owner_binding: UprightSE2SemanticOwnerBinding
    five_term_objective_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_objective_policy(self) -> Self:
        expected_refs = _objective_policy_refs()
        for name, expected_value in expected_refs.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    "objective policy reference does not match the frozen upright policy"
                )
        if tuple(term.term_id for term in self.terms) != ("T", "A", "R", "V", "S"):
            raise ValueError("objective policy must contain ordered T/A/R/V/S terms")
        if self.owner_binding.definition_ref != self.objective_definition_ref:
            raise ValueError(
                "objective owner binding must bind the objective definition"
            )
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.objective_definition_ref)
        ):
            raise ValueError("objective owner binding does not match the frozen policy")
        return self

    @property
    def weights_by_term(self) -> dict[str, float]:
        """Return the explicit weight mapping for later pure evaluators."""

        return {term.term_id: term.weight for term in self.terms}


class UprightSE2ExecutablePolicyValue(HashBoundCanonicalModel):
    """One request-owned executable policy value for a registered M3 family.

    The profile owns the permitted families, schemas, and evaluator bindings.
    This model deliberately owns none of the numeric or selector values: those
    are carried by the request-specific ``payload`` and sealed with it.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/executable-policy-value/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "executable_policy_value_sha256"

    policy_key: CanonicalId
    policy_family_ref: DefinitionRef
    definition_ref: DefinitionRef
    payload_schema_ref: CanonicalId
    owner_binding: UprightSE2SemanticOwnerBinding
    payload: TypedValue
    executable_policy_value_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_registered_request_policy(self) -> Self:
        expected = _executable_policy_spec(self.policy_key)
        if (
            self.policy_family_ref,
            self.definition_ref,
            self.payload_schema_ref,
        ) != (
            expected["policy_family_ref"],
            expected["definition_ref"],
            expected["payload_schema_ref"],
        ):
            raise ValueError("executable policy does not match a registered family")
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.definition_ref)
        ):
            raise ValueError("executable policy has an unowned evaluator binding")
        _validate_executable_policy_payload(self.policy_key, self.payload)
        return self


class UprightSE2ExecutablePolicyBundle(HashBoundCanonicalModel):
    """The complete non-ambient executable-policy bundle bound by one request."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/executable-policy-bundle/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "policy_bundle_sha256"

    profile_registration_sha256: Sha256Digest
    policies: tuple[UprightSE2ExecutablePolicyValue, ...]
    policy_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_request_policy_bundle(self) -> Self:
        keys = tuple(policy.policy_key for policy in self.policies)
        if keys != _EXECUTABLE_POLICY_KEYS:
            raise ValueError(
                "executable policy bundle must contain every policy exactly once"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("executable policy bundle must not duplicate one policy")
        return self

    def policy_for(self, policy_key: str) -> UprightSE2ExecutablePolicyValue:
        """Return one policy only after the complete roster has been sealed."""

        for policy in self.policies:
            if policy.policy_key == policy_key:
                return policy
        raise ValueError("executable policy bundle is missing the requested policy")


class UprightSE2SemanticClosure(HashBoundCanonicalModel):
    """The typed, hash-bound input closure consumed by every future M3 evaluator."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/semantic-closure/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_closure_sha256"

    profile_registration_sha256: Sha256Digest
    profile_registration: UprightSE2ProfileRegistration
    semantic_problem: CounterfactualProblemIR
    definition_bundle: DefinitionBundle
    predicate_definitions: tuple[UprightSE2SemanticDefinition, ...]
    objective_policy: UprightSE2FiveTermObjectivePolicy
    evaluator_bindings: tuple[UprightSE2SemanticOwnerBinding, ...]
    grounded_obligations: GroundedObligationSet
    objective_expression: ObjectiveExpression
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle
    resource_policy: ResourcePolicy
    semantic_closure_sha256: Sha256Digest

    @property
    def definition_bundle_sha256(self) -> Sha256Digest:
        """Expose the nested definition-bundle digest as a typed closure field."""

        return self.definition_bundle.definition_bundle_sha256

    @property
    def grounded_obligation_set_sha256(self) -> Sha256Digest:
        """Expose the nested grounded-obligation digest for evaluator replay."""

        return self.grounded_obligations.grounded_obligation_set_sha256

    @property
    def objective_expression_sha256(self) -> Sha256Digest:
        """Expose the nested objective-expression digest for evaluator replay."""

        return self.objective_expression.objective_expression_sha256

    @model_validator(mode="after")
    def _validate_closed_semantics(self) -> Self:
        if (
            self.profile_registration.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "semantic closure profile digest does not bind its profile"
            )
        _validate_task2_semantic_profile(self.profile_registration)
        if (
            self.semantic_problem.semantics_profile_ref
            != self.profile_registration.semantics_profile.semantics_profile_ref
            or self.semantic_problem.action_space_profile_ref
            != self.profile_registration.action_space_profile.action_space_profile_ref
        ):
            raise ValueError(
                "semantic closure source problem does not bind its profile"
            )
        expected_definitions = _upright_semantic_definitions()
        if tuple(
            canonical_json_bytes(value) for value in self.predicate_definitions
        ) != tuple(canonical_json_bytes(value) for value in expected_definitions):
            raise ValueError(
                "semantic closure must contain every exact predicate definition"
            )
        expected_policy = decode_upright_se2_objective_policy(
            self.executable_policy_bundle
        )
        if canonical_json_bytes(self.objective_policy) != canonical_json_bytes(
            expected_policy
        ):
            raise ValueError(
                "semantic closure must contain the exact five-term objective policy"
            )
        expected_bindings = _upright_semantic_owner_bindings()
        if self.evaluator_bindings != expected_bindings:
            raise ValueError(
                "semantic closure evaluator/verifier bindings do not close"
            )
        expected_bundle = _semantic_definition_bundle_for_digest(
            self.profile_registration_sha256, expected_policy
        )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            expected_bundle
        ):
            raise ValueError("semantic closure definition envelopes do not close")
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            build_upright_se2_objective_expression()
        ):
            raise ValueError("semantic closure objective expression does not close")
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            self.semantic_problem.definition_bundle
        ):
            raise ValueError(
                "semantic closure must bind the source problem definition bundle"
            )
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            self.semantic_problem.objective_expression
        ):
            raise ValueError(
                "semantic closure must bind the source problem objective expression"
            )
        expected_obligations = _bound_grounded_obligations(self.semantic_problem)
        if canonical_json_bytes(self.grounded_obligations) != canonical_json_bytes(
            expected_obligations
        ):
            raise ValueError(
                "semantic closure grounded obligations must bind the complete source roster"
            )
        if (
            self.executable_policy_bundle.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "executable policy bundle does not bind the semantic profile"
            )
        _validate_executable_policy_resource_binding(
            self.executable_policy_bundle,
            self.resource_policy,
        )
        validate_upright_se2_executable_policy_visibility_binding(
            self.executable_policy_bundle,
            self.semantic_problem,
        )
        return self

    @property
    def policy_bundle_sha256(self) -> Sha256Digest:
        """Expose the request-bound executable root to compiler consumers."""

        return self.executable_policy_bundle.policy_bundle_sha256


class UprightSE2ContinuousSemanticClosure(HashBoundCanonicalModel):
    """The additive continuous-yaw semantic closure.

    It intentionally has a distinct hash domain and validation entrypoint.  In
    particular, the frozen cardinal closure is never widened to accept an ARC
    or FULL_CIRCLE compiler-input wire.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-semantic-closure/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_closure_sha256"

    profile_registration_sha256: Sha256Digest
    profile_registration: UprightSE2ProfileRegistration
    semantic_problem: CounterfactualProblemIR
    definition_bundle: DefinitionBundle
    predicate_definitions: tuple[UprightSE2SemanticDefinition, ...]
    objective_policy: UprightSE2FiveTermObjectivePolicy
    evaluator_bindings: tuple[UprightSE2SemanticOwnerBinding, ...]
    grounded_obligations: GroundedObligationSet
    objective_expression: ObjectiveExpression
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle
    resource_policy: ResourcePolicy
    semantic_closure_sha256: Sha256Digest

    @property
    def definition_bundle_sha256(self) -> Sha256Digest:
        return self.definition_bundle.definition_bundle_sha256

    @property
    def grounded_obligation_set_sha256(self) -> Sha256Digest:
        return self.grounded_obligations.grounded_obligation_set_sha256

    @property
    def objective_expression_sha256(self) -> Sha256Digest:
        return self.objective_expression.objective_expression_sha256

    @model_validator(mode="after")
    def _validate_closed_semantics(self) -> Self:
        if (
            self.profile_registration.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "continuous semantic closure profile digest does not bind its profile"
            )
        _validate_task2_semantic_profile(self.profile_registration)
        if (
            self.semantic_problem.semantics_profile_ref
            != self.profile_registration.semantics_profile.semantics_profile_ref
            or self.semantic_problem.action_space_profile_ref
            != self.profile_registration.action_space_profile.action_space_profile_ref
        ):
            raise ValueError(
                "continuous semantic closure source problem does not bind its profile"
            )
        expected_definitions = _upright_semantic_definitions()
        if tuple(
            canonical_json_bytes(value) for value in self.predicate_definitions
        ) != tuple(canonical_json_bytes(value) for value in expected_definitions):
            raise ValueError(
                "continuous semantic closure must contain every exact predicate definition"
            )
        expected_policy = decode_upright_se2_objective_policy(
            self.executable_policy_bundle
        )
        if canonical_json_bytes(self.objective_policy) != canonical_json_bytes(
            expected_policy
        ):
            raise ValueError(
                "continuous semantic closure must contain the exact five-term objective policy"
            )
        if self.evaluator_bindings != _upright_semantic_owner_bindings():
            raise ValueError(
                "continuous semantic closure evaluator/verifier bindings do not close"
            )
        expected_bundle = _semantic_definition_bundle_for_digest(
            self.profile_registration_sha256, expected_policy
        )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            expected_bundle
        ):
            raise ValueError(
                "continuous semantic closure definition envelopes do not close"
            )
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            build_upright_se2_objective_expression()
        ):
            raise ValueError(
                "continuous semantic closure objective expression does not close"
            )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            self.semantic_problem.definition_bundle
        ) or canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            self.semantic_problem.objective_expression
        ):
            raise ValueError(
                "continuous semantic closure must bind the source semantic roots"
            )
        expected_obligations = _bound_continuous_grounded_obligations(
            self.semantic_problem
        )
        if canonical_json_bytes(self.grounded_obligations) != canonical_json_bytes(
            expected_obligations
        ):
            raise ValueError(
                "continuous semantic closure grounded obligations must bind the complete source roster"
            )
        if (
            self.executable_policy_bundle.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "continuous executable policy bundle does not bind the semantic profile"
            )
        _validate_executable_policy_resource_binding(
            self.executable_policy_bundle,
            self.resource_policy,
        )
        validate_upright_se2_executable_policy_visibility_binding(
            self.executable_policy_bundle,
            self.semantic_problem,
        )
        return self

    @property
    def policy_bundle_sha256(self) -> Sha256Digest:
        return self.executable_policy_bundle.policy_bundle_sha256


_SEMANTIC_ID_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-id/1.0"
_SEMANTIC_SYMBOL_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-symbol/1.0"
_SEMANTIC_REAL_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-real/1.0"
_SEMANTIC_TUPLE_SCHEMA_REF = "schema:spatialcf/upright-se2/semantic-tuple/1.0"
UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/semantic-definition-closure/1.0"
)
_SEMANTIC_DEFINITION_KIND_REF = (
    "definition:spatialcf/upright-se2/semantic-definition-body/1.0"
)
_OBJECTIVE_DEFINITION_KIND_REF = (
    "definition:spatialcf/upright-se2/five-term-objective-body/1.0"
)
_SEMANTIC_CLOSURE_KIND_REF = (
    "definition:spatialcf/upright-se2/semantic-closure-body/1.0"
)
_SEMANTIC_CLOSURE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/semantic-definition-closure/1.0"
)
_OBJECTIVE_POLICY_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/five-term-objective-policy/1.0"
)

_EXECUTABLE_POLICY_FAMILY_REF = "definition:spatialcf/upright-se2/executable-policy/1.0"
_EXECUTABLE_POLICY_WIRE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/executable-policy-wire/1.0"
)
_EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/executable-policy-bundle/1.0"
)
_EXECUTABLE_POLICY_BUNDLE_FACT_KEY = (
    "fact-key:spatialcf/upright-se2/executable-policy-bundle"
)
_EXECUTABLE_POLICY_KEYS = (
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
_EXECUTABLE_POLICY_SCHEMA_BY_KEY = {
    key: f"schema:spatialcf/upright-se2/executable-{key.replace(':', '-').lower()}-policy/1.0"
    for key in _EXECUTABLE_POLICY_KEYS
}

_SEMANTIC_KIND_TO_REF = {
    "COLLISION": UPRIGHT_SE2_COLLISION_PREDICATE_REF,
    "SUPPORT": UPRIGHT_SE2_SUPPORT_PREDICATE_REF,
    "TARGET_RELATION": UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
    "PRESERVATION": UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
    "VISIBILITY": UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
}
_SEMANTIC_KIND_TO_SCHEMA = {
    kind: f"schema:spatialcf/upright-se2/{kind.lower()}-semantics/1.0"
    for kind in _SEMANTIC_KIND_TO_REF
}
_SEMANTIC_BODY_VALUES = {
    "COLLISION": (
        (
            "clearance_measurement_ref",
            "definition:spatialcf/upright-se2/collision-clearance/1.0",
        ),
        (
            "contact_comparator_ref",
            "definition:spatialcf/upright-se2/collision-contact-comparator/1.0",
        ),
        (
            "contact_boundary_policy_ref",
            "definition:spatialcf/upright-se2/collision-closed-contact-boundary/1.0",
        ),
        (
            "evaluation_frame_ref",
            "definition:spatialcf/upright-se2/world-xy/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/collision-evaluation-input/1.0",
        ),
        (
            "collision_evaluator_ref",
            "definition:spatialcf/upright-se2/collision-evaluator/1.0",
        ),
        (
            "obstacle_roster_selector_ref",
            "definition:spatialcf/upright-se2/collision-complete-obstacle-roster/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/collision-source-exactness/1.0",
        ),
        (
            "subject_geometry_selector_ref",
            "definition:spatialcf/upright-se2/collision-subject-geometry/1.0",
        ),
    ),
    "SUPPORT": (
        (
            "clearance_policy_ref",
            "definition:spatialcf/upright-se2/support-clearance/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/support-evaluation-input/1.0",
        ),
        (
            "containment_boundary_policy_ref",
            "definition:spatialcf/upright-se2/support-contained-closed-boundary/1.0",
        ),
        (
            "height_contact_comparator_ref",
            "definition:spatialcf/upright-se2/support-height-contact-comparator/1.0",
        ),
        (
            "same_surface_comparator_ref",
            "definition:spatialcf/upright-se2/support-same-surface-comparator/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/support-source-exactness/1.0",
        ),
        (
            "support_surface_selector_ref",
            "definition:spatialcf/upright-se2/support-required-surface/1.0",
        ),
        (
            "surface_frame_requirement_ref",
            "definition:spatialcf/upright-se2/support-world-plus-z-frame/1.0",
        ),
        (
            "surface_normal_requirement_ref",
            "definition:spatialcf/upright-se2/support-world-plus-z-normal/1.0",
        ),
    ),
    "TARGET_RELATION": (
        (
            "after_state_geometry_selector_ref",
            "definition:spatialcf/upright-se2/relation-after-state-geometry/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/target-relation-evaluation-input/1.0",
        ),
        (
            "relation_boundary_policy_ref",
            "definition:spatialcf/upright-se2/relation-closed-boundary/1.0",
        ),
        (
            "relation_comparator_ref",
            "definition:spatialcf/upright-se2/relation-comparator/1.0",
        ),
        (
            "relation_frame_ref",
            "definition:spatialcf/upright-se2/relation-fixed-camera-frame/1.0",
        ),
        (
            "relation_tolerance_policy_ref",
            "definition:spatialcf/upright-se2/relation-tolerance/1.0",
        ),
        (
            "target_relation_selector_ref",
            "definition:spatialcf/upright-se2/relation-target-selector/1.0",
        ),
        (
            "source_fact_completeness_ref",
            "definition:spatialcf/upright-se2/relation-source-exactness/1.0",
        ),
    ),
    "PRESERVATION": (
        (
            "after_state_selector_ref",
            "definition:spatialcf/upright-se2/preservation-after-state/1.0",
        ),
        (
            "grounded_operand_policy_ref",
            "definition:spatialcf/upright-se2/preservation-grounded-operands/1.0",
        ),
        (
            "before_state_selector_ref",
            "definition:spatialcf/upright-se2/preservation-before-state/1.0",
        ),
        (
            "complete_state_policy_ref",
            "definition:spatialcf/upright-se2/complete-state-delta/1.0",
        ),
        (
            "frozen_leaf_policy_ref",
            "definition:spatialcf/upright-se2/preservation-frozen-leaves/1.0",
        ),
        (
            "preservation_evaluator_ref",
            "definition:spatialcf/upright-se2/preservation-evaluator/1.0",
        ),
        (
            "transition_comparator_ref",
            "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0",
        ),
    ),
    "VISIBILITY": (
        (
            "camera_selector_ref",
            "definition:spatialcf/upright-se2/visibility-fixed-camera-selector/1.0",
        ),
        (
            "evaluation_input_schema_ref",
            "schema:spatialcf/upright-se2/visibility-evaluation-input/1.0",
        ),
        (
            "fixed_camera_policy_ref",
            "definition:spatialcf/upright-se2/visibility-fixed-camera-policy/1.0",
        ),
        (
            "metric_definition_ref",
            "definition:spatialcf/upright-se2/visibility-metric/1.0",
        ),
        (
            "occluder_roster_completeness_ref",
            "definition:spatialcf/upright-se2/visibility-complete-occluder-roster/1.0",
        ),
        (
            "threshold_comparator_ref",
            "definition:spatialcf/upright-se2/visibility-threshold-comparator/1.0",
        ),
        (
            "threshold_definition_ref",
            "definition:spatialcf/upright-se2/visibility-threshold/1.0",
        ),
        (
            "visibility_boundary_policy_ref",
            "definition:spatialcf/upright-se2/visibility-closed-boundary/1.0",
        ),
        (
            "visibility_evaluator_ref",
            "definition:spatialcf/upright-se2/visibility-evaluator/1.0",
        ),
    ),
}


def _semantic_definition_ref_for_kind(kind: str) -> DefinitionRef:
    return _SEMANTIC_KIND_TO_REF[kind]


def _semantic_definition_schema_for_kind(kind: str) -> CanonicalId:
    return _SEMANTIC_KIND_TO_SCHEMA[kind]


def _semantic_capabilities_for_definition(
    definition_ref: DefinitionRef,
) -> tuple[CapabilityRef, CapabilityRef]:
    if definition_ref == UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF:
        return (
            UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
        )
    if definition_ref in UPRIGHT_SE2_PREDICATE_DEFINITION_REFS:
        return (
            UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
        )
    raise ValueError(
        "semantic owner binding must name a registered semantic definition"
    )


def _semantic_typed_id(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_ID_SCHEMA_REF,
        payload=CanonicalIdValue(value=value),
    )


def _semantic_typed_symbol(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_SYMBOL_SCHEMA_REF,
        payload=EnumSymbolValue(symbol=value),
    )


def _semantic_typed_real(value: float) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_REAL_SCHEMA_REF,
        payload=FiniteRealValue(value=value),
    )


def _semantic_typed_digest(value: str) -> TypedValue:
    return TypedValue(
        value_schema_ref="schema:spatialcf/upright-se2/digest/1.0",
        payload=DigestValue(value=value),
    )


def _semantic_typed_tuple(values: tuple[str, ...]) -> TypedValue:
    return TypedValue(
        value_schema_ref=_SEMANTIC_TUPLE_SCHEMA_REF,
        payload=FiniteOrderedTupleValue(
            element_schema_ref=_SEMANTIC_ID_SCHEMA_REF,
            items=tuple(_semantic_typed_id(value) for value in values),
        ),
    )


def _semantic_typed_digest_tuple(values: tuple[str, ...]) -> TypedValue:
    return TypedValue(
        value_schema_ref="schema:spatialcf/upright-se2/semantic-digest-tuple/1.0",
        payload=FiniteOrderedTupleValue(
            element_schema_ref="schema:spatialcf/upright-se2/digest/1.0",
            items=tuple(_semantic_typed_digest(value) for value in values),
        ),
    )


def _semantic_typed_record(
    schema_ref: str,
    fields: tuple[tuple[str, TypedValue], ...],
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=canonical_json_bytes,
                )
            )
        ),
    )


def _semantic_definition_body(kind: str) -> TypedValue:
    schema_ref = _semantic_definition_schema_for_kind(kind)
    return _semantic_typed_record(
        schema_ref,
        (
            ("definition_kind", _semantic_typed_symbol(kind)),
            (
                "operand_schema_refs",
                _semantic_typed_tuple(_semantic_operand_schemas(kind)),
            ),
            *(
                (name, _semantic_typed_id(value))
                for name, value in _SEMANTIC_BODY_VALUES[kind]
            ),
        ),
    )


def _semantic_operand_schemas(kind: str) -> tuple[str, ...]:
    if kind == "TARGET_RELATION":
        return (
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/relation-symbol/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    if kind == "PRESERVATION":
        return (
            "schema:spatialcf/upright-se2/entity-ref/1.0",
            "schema:spatialcf/upright-se2/preservation-selector/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    if kind == "VISIBILITY":
        return (
            "schema:spatialcf/upright-se2/camera-ref/1.0",
            "schema:spatialcf/upright-se2/object-ref/1.0",
            "schema:spatialcf/upright-se2/visibility-metric-symbol/1.0",
            "schema:spatialcf/upright-se2/observation-ref/1.0",
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
    return ()


def _semantic_owner_binding(
    definition_ref: DefinitionRef,
) -> UprightSE2SemanticOwnerBinding:
    evaluator_capability_ref, verifier_capability_ref = (
        _semantic_capabilities_for_definition(definition_ref)
    )
    return UprightSE2SemanticOwnerBinding(
        definition_ref=definition_ref,
        evaluator_capability_ref=evaluator_capability_ref,
        verifier_capability_ref=verifier_capability_ref,
        evaluator_owner_ref=UPRIGHT_SE2_BACKEND_OWNER_REF,
        verifier_owner_ref=UPRIGHT_SE2_CHECKER_OWNER_REF,
        evaluator_build_sha256="b" * 64,
        verifier_build_sha256="a" * 64,
    )


def _executable_policy_spec(policy_key: str) -> dict[str, str]:
    """Return profile-owned family metadata for one request-owned policy key."""

    if policy_key not in _EXECUTABLE_POLICY_SCHEMA_BY_KEY:
        raise ValueError("executable policy key is unknown")
    if policy_key == "collision":
        definition_ref = UPRIGHT_SE2_COLLISION_PREDICATE_REF
    elif policy_key == "support":
        definition_ref = UPRIGHT_SE2_SUPPORT_PREDICATE_REF
    elif policy_key.startswith("relation:"):
        definition_ref = UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
    elif policy_key == "visibility":
        definition_ref = UPRIGHT_SE2_VISIBILITY_PREDICATE_REF
    elif policy_key == "preservation":
        definition_ref = UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
    else:
        # Numeric, resource, and safety determine objective/search accounting;
        # their evaluator owner is the registered objective authority.
        definition_ref = UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF
    return {
        "policy_family_ref": _EXECUTABLE_POLICY_FAMILY_REF,
        "definition_ref": definition_ref,
        "payload_schema_ref": _EXECUTABLE_POLICY_SCHEMA_BY_KEY[policy_key],
    }


def _policy_payload_fields(
    policy_key: str,
    payload: TypedValue,
) -> dict[str, TypedValue]:
    expected_schema = _executable_policy_spec(policy_key)["payload_schema_ref"]
    if (
        payload.value_schema_ref != expected_schema
        or type(payload.payload) is not RecordValue
    ):
        raise ValueError(
            "executable policy payload must use its registered record schema"
        )
    fields = {field.name: field.value for field in payload.payload.fields}
    if len(fields) != len(payload.payload.fields):
        raise ValueError("executable policy payload contains duplicate fields")
    return fields


def _require_policy_fields(
    policy_key: str,
    fields: dict[str, TypedValue],
    expected: tuple[str, ...],
) -> None:
    if set(fields) != set(expected):
        raise ValueError(
            f"executable policy {policy_key!r} fields are incomplete or unknown"
        )


def _policy_real(fields: dict[str, TypedValue], name: str) -> float:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (_SEMANTIC_REAL_SCHEMA_REF, _UPRIGHT_SE2_REAL_SCHEMA_REF)
        or type(value.payload) is not FiniteRealValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a finite real")
    return value.payload.value


def _policy_integer(fields: dict[str, TypedValue], name: str) -> int:
    value = fields[name]
    if (
        value.value_schema_ref != "schema:spatialcf/upright-se2/integer/1.0"
        or type(value.payload) is not IntegerValue
    ):
        raise ValueError(f"executable policy field {name!r} must be an integer")
    return value.payload.value


def _policy_symbol(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (
            _SEMANTIC_SYMBOL_SCHEMA_REF,
            "schema:spatialcf/upright-se2/enum-symbol/1.0",
        )
        or type(value.payload) is not EnumSymbolValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a symbol")
    return value.payload.symbol


def _policy_id(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (_SEMANTIC_ID_SCHEMA_REF, _UPRIGHT_SE2_ID_SCHEMA_REF)
        or type(value.payload) is not CanonicalIdValue
    ):
        raise ValueError(f"executable policy field {name!r} must be an identifier")
    return value.payload.value


def _policy_digest(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref != "schema:spatialcf/upright-se2/digest/1.0"
        or type(value.payload) is not DigestValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a digest")
    return value.payload.value


def _policy_tuple(fields: dict[str, TypedValue], name: str) -> FiniteOrderedTupleValue:
    value = fields[name]
    if type(value.payload) is not FiniteOrderedTupleValue:
        raise ValueError(f"executable policy field {name!r} must be an ordered tuple")
    return value.payload


def _policy_id_tuple(fields: dict[str, TypedValue], name: str) -> tuple[str, ...]:
    value = fields[name]
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/resource-limit-values/1.0"
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/policy-item/1.0"
    ):
        raise ValueError(
            f"executable policy field {name!r} must be an exact identifier tuple"
        )
    identifiers: list[str] = []
    for item in value.payload.items:
        if (
            item.value_schema_ref
            not in (_SEMANTIC_ID_SCHEMA_REF, _UPRIGHT_SE2_ID_SCHEMA_REF)
            or type(item.payload) is not CanonicalIdValue
        ):
            raise ValueError(
                f"executable policy field {name!r} must be an exact identifier tuple"
            )
        identifiers.append(item.payload.value)
    return tuple(identifiers)


def _visibility_observation_bound_fields(value: TypedValue) -> dict[str, TypedValue]:
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
        or type(value.payload) is not RecordValue
    ):
        raise ValueError(
            "executable visibility observation bounds must use typed bound records"
        )
    fields = {field.name: field.value for field in value.payload.fields}
    if len(fields) != len(value.payload.fields) or set(fields) != {
        "observation_id",
        "camera_id",
        "object_id",
        "metric_definition_id",
        "metric_definition_version",
        "comparator",
        "boundary_policy",
        "threshold",
        "tolerance",
    }:
        raise ValueError(
            "executable visibility observation bound fields are incomplete or unknown"
        )
    return fields


def _validate_visibility_observation_bounds(fields: dict[str, TypedValue]) -> None:
    value = fields["observation_bounds"]
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bounds/1.0"
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
    ):
        raise ValueError(
            "executable visibility policy must bind an exact typed observation roster"
        )
    observation_ids: list[str] = []
    for item in value.payload.items:
        bound = _visibility_observation_bound_fields(item)
        observation_ids.append(_policy_id(bound, "observation_id"))
        _policy_id(bound, "camera_id")
        _policy_id(bound, "object_id")
        metric_pair = (
            _policy_symbol(bound, "metric_definition_id"),
            _policy_id(bound, "metric_definition_version"),
        )
        if metric_pair not in {
            ("visibility:image-area-fraction", "definition:1"),
            ("visibility:truncated-fraction", "definition:1"),
            ("visibility:visible-surface-fraction", "definition:1"),
        }:
            raise ValueError("executable visibility observation metric is unsupported")
        for name, expected in (
            ("comparator", _policy_symbol(fields, "comparator")),
            ("boundary_policy", _policy_symbol(fields, "boundary_policy")),
        ):
            if _policy_symbol(bound, name) != expected:
                raise ValueError(
                    f"executable visibility observation bound {name!r} is inconsistent"
                )
        for name in ("threshold", "tolerance"):
            bound_value = _policy_real(bound, name)
            if not 0.0 <= bound_value <= 1.0:
                raise ValueError(
                    f"executable visibility observation bound {name} must be in [0, 1]"
                )
    if not observation_ids or len(set(observation_ids)) != len(observation_ids):
        raise ValueError(
            "executable visibility observation bounds must be nonempty and unique"
        )
    if tuple(observation_ids) != tuple(
        sorted(observation_ids, key=canonical_json_bytes)
    ):
        raise ValueError("executable visibility observation bounds must be ordered")


def _decode_upright_se2_objective_policy_payload(
    payload: TypedValue,
) -> UprightSE2FiveTermObjectivePolicy:
    fields = _policy_payload_fields("objective", payload)
    _require_policy_fields(
        "objective",
        fields,
        (
            "aggregation_rule",
            "comparison_rule",
            "objective_policy_sha256",
            "terms",
            "tie_break_rule",
        ),
    )
    _require_policy_values(
        fields,
        {
            "aggregation_rule": "WEIGHTED_NORMALIZED_SUM",
            "comparison_rule": "INTERVAL_LEXICOGRAPHIC",
            "tie_break_rule": "T_R_V_S_A",
        },
    )
    terms_value = fields["terms"]
    if (
        terms_value.value_schema_ref
        != "schema:spatialcf/upright-se2/objective-term-values/1.0"
        or type(terms_value.payload) is not FiniteOrderedTupleValue
        or terms_value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/objective-term-value/1.0"
    ):
        raise ValueError(
            "executable objective policy terms must use the exact typed roster"
        )
    terms: list[UprightSE2ObjectiveTermPolicy] = []
    for item in terms_value.payload.items:
        if (
            item.value_schema_ref
            != "schema:spatialcf/upright-se2/objective-term-value/1.0"
            or type(item.payload) is not RecordValue
        ):
            raise ValueError(
                "executable objective policy term must use its typed record"
            )
        term_fields = {field.name: field.value for field in item.payload.fields}
        if len(term_fields) != len(item.payload.fields) or set(term_fields) != {
            "term_id",
            "objective_definition_ref",
            "metric_definition_ref",
            "input_selector_definition_ref",
            "unit_ref",
            "normalization_definition_ref",
            "normalizer_unit_ref",
            "weight",
            "normalizer",
        }:
            raise ValueError(
                "executable objective policy term fields are incomplete or unknown"
            )
        terms.append(
            UprightSE2ObjectiveTermPolicy(
                term_id=_policy_symbol(term_fields, "term_id"),
                objective_definition_ref=_policy_id(
                    term_fields, "objective_definition_ref"
                ),
                metric_definition_ref=_policy_id(term_fields, "metric_definition_ref"),
                input_selector_definition_ref=_policy_id(
                    term_fields, "input_selector_definition_ref"
                ),
                unit_ref=_policy_id(term_fields, "unit_ref"),
                normalization_definition_ref=_policy_id(
                    term_fields, "normalization_definition_ref"
                ),
                normalizer_unit_ref=_policy_id(term_fields, "normalizer_unit_ref"),
                weight=_policy_real(term_fields, "weight"),
                normalizer=_policy_real(term_fields, "normalizer"),
            )
        )
    policy = UprightSE2FiveTermObjectivePolicy.seal(
        **_objective_policy_refs(),
        terms=tuple(terms),
        owner_binding=_semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
    )
    if _policy_digest(fields, "objective_policy_sha256") != (
        policy.five_term_objective_policy_sha256
    ):
        raise ValueError("executable objective policy self digest does not match")
    return policy


def _validate_executable_policy_payload(policy_key: str, payload: TypedValue) -> None:
    """Validate every executable family without substituting a value or default."""

    fields = _policy_payload_fields(policy_key, payload)
    if policy_key == "collision":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "clearance_m",
                "contact_comparator",
                "obstacle_selector",
                "subject_geometry_selector",
            ),
        )
        if _policy_real(fields, "clearance_m") < 0.0:
            raise ValueError(
                "executable collision policy clearance must be non-negative"
            )
        _require_policy_values(
            fields,
            {
                "contact_comparator": "ALLOW_EQUALITY",
                "boundary_policy": "CLOSED",
            },
        )
        _require_policy_ids(
            fields,
            {
                "obstacle_selector": "selector:complete-obstacle-roster",
                "subject_geometry_selector": "selector:compound-subject-geometry",
            },
        )
        return
    if policy_key == "support":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "contact_gap_lower_m",
                "contact_gap_upper_m",
                "containment_boundary_policy",
                "containment_comparator",
                "normal_selector",
                "stability_margin_m",
                "support_frame_selector",
                "support_surface_selector",
            ),
        )
        if _policy_real(fields, "contact_gap_lower_m") > _policy_real(
            fields, "contact_gap_upper_m"
        ):
            raise ValueError(
                "executable support policy contact-gap interval is reversed"
            )
        if _policy_real(fields, "stability_margin_m") < 0.0:
            raise ValueError(
                "executable support policy stability margin must be non-negative"
            )
        _require_policy_ids(
            fields,
            {
                "normal_selector": "selector:world-positive-z",
                "support_frame_selector": "selector:world-xy",
                "support_surface_selector": "selector:assigned-support-surface",
            },
        )
        _require_policy_values(
            fields,
            {
                "containment_boundary_policy": "CLOSED",
                "containment_comparator": "CONTAINS",
            },
        )
        return
    if policy_key.startswith("relation:"):
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "comparator",
                "fixed_camera_selector",
                "frame_selector",
                "measurement",
                "operand_order",
                "relation_symbol",
                "representative_geometry",
                "threshold",
                "tolerance",
                "visibility_gate",
            ),
        )
        if _policy_symbol(fields, "relation_symbol") != policy_key.removeprefix(
            "relation:"
        ):
            raise ValueError("executable relation policy has the wrong relation symbol")
        relation = policy_key.removeprefix("relation:")
        _require_policy_values(
            fields,
            {
                "boundary_policy": "CLOSED",
                "comparator": "LE" if relation in {"LEFT", "FRONT", "NEAR"} else "GE",
                "measurement": (
                    "EXTENT_EUCLIDEAN_SEPARATION"
                    if relation in {"NEAR", "FAR"}
                    else "EXTENT_SIGNED_AXIS_GAP"
                ),
                "operand_order": "SUBJECT_THEN_REFERENCE",
                "representative_geometry": "COMPOUND_BODY",
                "visibility_gate": "NONE",
            },
        )
        _require_policy_ids(
            fields,
            {
                "fixed_camera_selector": "selector:fixed-camera",
                "frame_selector": "selector:world-xy",
            },
        )
        if _policy_real(fields, "threshold") < 0.0:
            raise ValueError(
                "executable relation policy threshold must be non-negative"
            )
        if _policy_real(fields, "tolerance") < 0.0:
            raise ValueError(
                "executable relation policy tolerance must be non-negative"
            )
        return
    if policy_key == "visibility":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "camera_projection_convention",
                "comparator",
                "depth_policy",
                "mask_policy",
                "observation_bounds",
                "occluder_policy",
                "projected_area_metric",
                "subject_as_occluder",
                "threshold",
                "tolerance",
            ),
        )
        expected_symbols = {
            "boundary_policy": "CLOSED",
            "camera_projection_convention": "UPRIGHT_CAMERA_V2_9",
            "comparator": "GEQ",
            "depth_policy": "NEAR_CLIPPED",
            "mask_policy": "COMPLETE_MASK",
            "occluder_policy": "COMPLETE_ROSTER",
            "projected_area_metric": "PROJECTED_AREA",
            "subject_as_occluder": "INCLUDED",
        }
        for name, expected in expected_symbols.items():
            if _policy_symbol(fields, name) != expected:
                raise ValueError(
                    f"executable visibility policy field {name!r} is unsupported"
                )
        _validate_visibility_observation_bounds(fields)
        if not _policy_tuple(fields, "observation_bounds").items:
            raise ValueError(
                "executable visibility policy must bind observation bounds"
            )
        for name in ("threshold", "tolerance"):
            value = _policy_real(fields, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"executable visibility policy {name} must be in [0, 1]"
                )
        return
    if policy_key == "numeric":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "dyadic_refinement_policy",
                "exact_number_representation",
                "numeric_semantics_ref",
                "tolerance_m",
            ),
        )
        _require_policy_ids(
            fields,
            {
                "numeric_semantics_ref": "definition:spatialcf/upright-se2/numeric-semantics/1.0"
            },
        )
        if _policy_real(fields, "tolerance_m") < 0.0:
            raise ValueError("executable numeric policy tolerance must be non-negative")
        _require_policy_values(
            fields,
            {
                "exact_number_representation": "BINARY64_BITS",
                "dyadic_refinement_policy": "EXACT_DYADIC",
            },
        )
        return
    if policy_key == "objective":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "aggregation_rule",
                "comparison_rule",
                "objective_policy_sha256",
                "terms",
                "tie_break_rule",
            ),
        )
        _decode_upright_se2_objective_policy_payload(payload)
        return
    if policy_key == "safety":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "constraint_slack_target",
                "hard_constraint_selector",
                "safety_penalty_rule",
            ),
        )
        if _policy_real(fields, "constraint_slack_target") < 0.0:
            raise ValueError(
                "executable safety policy slack target must be non-negative"
            )
        _require_policy_ids(
            fields,
            {"hard_constraint_selector": "selector:all-hard-constraints"},
        )
        _require_policy_values(
            fields,
            {"safety_penalty_rule": "FROM_CONSTRAINT_SLACK"},
        )
        return
    if policy_key == "resource":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "atomic_step_limit",
                "deterministic_order",
                "limits",
                "resource_policy_sha256",
                "shared_ledger_policy_ref",
            ),
        )
        atomic_step_limit = _policy_integer(fields, "atomic_step_limit")
        if atomic_step_limit < 0:
            raise ValueError(
                "executable resource policy atomic step limit must be non-negative"
            )
        if not _policy_id_tuple(fields, "limits"):
            raise ValueError("executable resource policy must bind every request limit")
        _policy_digest(fields, "resource_policy_sha256")
        _policy_id(fields, "shared_ledger_policy_ref")
        _require_policy_values(
            fields,
            {"deterministic_order": "LOWER_OWNED_XY"},
        )
        return
    if policy_key == "preservation":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "frozen_observation_policy",
                "grounded_invariant_selector",
                "state_selector",
            ),
        )
        _require_policy_values(
            fields,
            {"frozen_observation_policy": "COMPLETE_GROUNDED"},
        )
        _require_policy_ids(
            fields,
            {
                "grounded_invariant_selector": "selector:grounded-invariants",
                "state_selector": "selector:frozen-nonprimary-state",
            },
        )
        return
    raise ValueError("executable policy key is unknown")


def _require_policy_values(
    fields: dict[str, TypedValue], expected: dict[str, str]
) -> None:
    for name, expected_value in expected.items():
        if _policy_symbol(fields, name) != expected_value:
            raise ValueError(f"executable policy field {name!r} is unsupported")


def _require_policy_ids(
    fields: dict[str, TypedValue], expected: dict[str, str]
) -> None:
    for name, expected_value in expected.items():
        if _policy_id(fields, name) != expected_value:
            raise ValueError(f"executable policy field {name!r} is unsupported")


def _executable_policy_wire(policy: UprightSE2ExecutablePolicyValue) -> TypedValue:
    return _semantic_typed_record(
        _EXECUTABLE_POLICY_WIRE_SCHEMA_REF,
        (
            ("definition_ref", _semantic_typed_id(policy.definition_ref)),
            ("payload", policy.payload),
            ("payload_schema_ref", _semantic_typed_id(policy.payload_schema_ref)),
            ("policy_family_ref", _semantic_typed_id(policy.policy_family_ref)),
            ("policy_key", _semantic_typed_id(policy.policy_key)),
        ),
    )


def executable_policy_bundle_to_typed_value(
    bundle: UprightSE2ExecutablePolicyBundle,
) -> TypedValue:
    """Encode a sealed bundle into the semantic-problem extension-fact wire."""

    return _semantic_typed_record(
        _EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF,
        (
            (
                "policies",
                TypedValue(
                    value_schema_ref=(
                        "schema:spatialcf/upright-se2/executable-policy-wire-tuple/1.0"
                    ),
                    payload=FiniteOrderedTupleValue(
                        element_schema_ref=_EXECUTABLE_POLICY_WIRE_SCHEMA_REF,
                        items=tuple(
                            _executable_policy_wire(policy)
                            for policy in bundle.policies
                        ),
                    ),
                ),
            ),
            (
                "profile_registration_sha256",
                _semantic_typed_digest(bundle.profile_registration_sha256),
            ),
        ),
    )


def _record_field_map(value: TypedValue, schema_ref: str) -> dict[str, TypedValue]:
    if value.value_schema_ref != schema_ref or type(value.payload) is not RecordValue:
        raise ValueError("executable policy wire has the wrong record schema")
    return {field.name: field.value for field in value.payload.fields}


def _wire_policy_value(value: TypedValue) -> UprightSE2ExecutablePolicyValue:
    fields = _record_field_map(value, _EXECUTABLE_POLICY_WIRE_SCHEMA_REF)
    if set(fields) != {
        "definition_ref",
        "payload",
        "payload_schema_ref",
        "policy_family_ref",
        "policy_key",
    }:
        raise ValueError("executable policy wire fields are incomplete or unknown")
    policy_key = _policy_id(fields, "policy_key")
    definition_ref = _policy_id(fields, "definition_ref")
    return UprightSE2ExecutablePolicyValue.seal(
        policy_key=policy_key,
        policy_family_ref=_policy_id(fields, "policy_family_ref"),
        definition_ref=definition_ref,
        payload_schema_ref=_policy_id(fields, "payload_schema_ref"),
        owner_binding=_semantic_owner_binding(definition_ref),
        payload=fields["payload"],
    )


def request_bound_executable_policy_bundle_from_problem(
    semantic_problem: CounterfactualProblemIR,
) -> UprightSE2ExecutablePolicyBundle:
    """Recover the request-owned bundle from the only permitted semantic wire."""

    facts = tuple(
        fact
        for bundle in semantic_problem.scene_state.extension_fact_bundles
        for fact in bundle.facts
        if fact.fact_family_ref == _EXECUTABLE_POLICY_FAMILY_REF
    )
    if len(facts) != 1:
        raise ValueError("executable policy bundle must be present exactly once")
    fact = facts[0]
    if (
        fact.fact_key != _EXECUTABLE_POLICY_BUNDLE_FACT_KEY
        or fact.subject_entity_id != "entity:upright-se2-policy"
    ):
        raise ValueError("executable policy bundle has an unbound extension-fact owner")
    fields = _record_field_map(fact.value, _EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF)
    if set(fields) != {"policies", "profile_registration_sha256"}:
        raise ValueError(
            "executable policy bundle wire fields are incomplete or unknown"
        )
    profile_digest = _policy_digest(fields, "profile_registration_sha256")
    policies_value = fields["policies"]
    if (
        policies_value.value_schema_ref
        != "schema:spatialcf/upright-se2/executable-policy-wire-tuple/1.0"
        or type(policies_value.payload) is not FiniteOrderedTupleValue
        or policies_value.payload.element_schema_ref
        != _EXECUTABLE_POLICY_WIRE_SCHEMA_REF
    ):
        raise ValueError(
            "executable policy bundle must carry the registered policy tuple"
        )
    return UprightSE2ExecutablePolicyBundle.seal(
        profile_registration_sha256=profile_digest,
        policies=tuple(
            _wire_policy_value(value) for value in policies_value.payload.items
        ),
    )


def build_upright_se2_executable_policy_bundle(
    *,
    profile_registration_sha256: Sha256Digest,
    policies: tuple[UprightSE2ExecutablePolicyValue, ...],
) -> UprightSE2ExecutablePolicyBundle:
    """Seal supplied policy values without selecting any ambient values."""

    return UprightSE2ExecutablePolicyBundle.seal(
        profile_registration_sha256=profile_registration_sha256,
        policies=policies,
    )


def decode_upright_se2_objective_policy(
    bundle: UprightSE2ExecutablePolicyBundle,
) -> UprightSE2FiveTermObjectivePolicy:
    """Decode the one request-bound T/A/R/V/S evaluator policy from its wire."""

    return _decode_upright_se2_objective_policy_payload(
        bundle.policy_for("objective").payload
    )


def _validate_executable_policy_resource_binding(
    bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> None:
    resource_fields = _policy_payload_fields(
        "resource", bundle.policy_for("resource").payload
    )
    if len(resource_policy.limits) != 1:
        raise ValueError("resource policy must bind one registered limit")
    limit = resource_policy.limits[0]
    if (
        not math.isfinite(limit.finite_limit)
        or limit.finite_limit < 0.0
        or not limit.finite_limit.is_integer()
    ):
        raise ValueError("resource policy finite limit must be a non-negative integer")
    if _policy_id_tuple(resource_fields, "limits") != (limit.definition_ref,):
        raise ValueError(
            "executable resource policy limits do not bind the source request"
        )
    if _policy_integer(resource_fields, "atomic_step_limit") != int(limit.finite_limit):
        raise ValueError(
            "executable resource policy atomic step limit does not bind the source request"
        )
    if (
        _policy_digest(resource_fields, "resource_policy_sha256")
        != resource_policy.resource_policy_sha256
    ):
        raise ValueError(
            "executable policy resource root does not bind the source request"
        )
    if (
        _policy_id(resource_fields, "shared_ledger_policy_ref")
        != resource_policy.shared_ledger_policy_ref
    ):
        raise ValueError(
            "executable policy shared ledger does not bind the source request"
        )


def validate_upright_se2_executable_policy_visibility_binding(
    bundle: UprightSE2ExecutablePolicyBundle,
    problem: CounterfactualProblemIR,
) -> None:
    """Bind every request visibility bound to one exact grounded observation."""

    fields = _policy_payload_fields(
        "visibility", bundle.policy_for("visibility").payload
    )
    _validate_visibility_observation_bounds(fields)
    value = fields["observation_bounds"]
    assert type(value.payload) is FiniteOrderedTupleValue
    scene = problem.scene_state.base_scene_payload
    observations = _exact_source_values(
        scene.baseline_observations, "baseline observations"
    )
    expected = {observation.observation_id: observation for observation in observations}
    if len(problem.explicit_observation_obligations) != len(expected):
        raise ValueError(
            "semantic visibility obligations must cover every exact observation"
        )
    obligation_ids: set[str] = set()
    for obligation in problem.explicit_observation_obligations:
        if type(obligation) is not ObservationObligation:
            raise ValueError(
                "semantic visibility obligations must be exact observations"
            )
        obligation_id = _bound_visibility_observation_id(obligation, scene)
        if obligation_id in obligation_ids:
            raise ValueError(
                "semantic visibility obligations must not duplicate observations"
            )
        obligation_ids.add(obligation_id)
    if obligation_ids != set(expected):
        raise ValueError("semantic visibility obligations must match the exact roster")
    bound_ids: list[str] = []
    for item in value.payload.items:
        bound = _visibility_observation_bound_fields(item)
        observation_id = _policy_id(bound, "observation_id")
        source = expected.get(observation_id)
        if source is None:
            raise ValueError("executable visibility observation bound is unknown")
        if (
            _policy_id(bound, "camera_id") != source.camera_id
            or _policy_id(bound, "object_id") != source.object_id
            or _policy_symbol(bound, "metric_definition_id")
            != source.metric_definition_id
            or _policy_id(bound, "metric_definition_version")
            != source.metric_definition_version
        ):
            raise ValueError(
                "executable visibility observation bound does not match its obligation"
            )
        if (
            _policy_symbol(bound, "comparator") != _policy_symbol(fields, "comparator")
            or _policy_symbol(bound, "boundary_policy")
            != _policy_symbol(fields, "boundary_policy")
            or _policy_real(bound, "threshold") != _policy_real(fields, "threshold")
            or _policy_real(bound, "tolerance") != _policy_real(fields, "tolerance")
        ):
            raise ValueError(
                "executable visibility observation bound does not match its policy"
            )
        bound_ids.append(observation_id)
    if tuple(bound_ids) != tuple(
        observation.observation_id for observation in observations
    ):
        raise ValueError(
            "executable visibility observation bounds must match the exact ordered roster"
        )


def _upright_semantic_definitions() -> tuple[UprightSE2SemanticDefinition, ...]:
    kinds = tuple(
        sorted(
            _SEMANTIC_KIND_TO_REF,
            key=lambda kind: canonical_json_bytes(_SEMANTIC_KIND_TO_REF[kind]),
        )
    )
    return tuple(
        UprightSE2SemanticDefinition.seal(
            definition_ref=_semantic_definition_ref_for_kind(kind),
            definition_kind=kind,
            payload_schema_ref=_semantic_definition_schema_for_kind(kind),
            semantic_body=_semantic_definition_body(kind),
            owner_binding=_semantic_owner_binding(
                _semantic_definition_ref_for_kind(kind)
            ),
        )
        for kind in kinds
    )


def _objective_term_policy_values(term_id: str) -> dict[str, object]:
    values = {
        "T": (
            "definition:spatialcf/upright-se2/objective-subject-pivot-displacement/1.0",
            "definition:spatialcf/upright-se2/objective-selector-subject-pivot-displacement/1.0",
            "definition:spatialcf/upright-se2/metre/1.0",
            "definition:spatialcf/upright-se2/normalizer-subject-pivot-displacement/1.0",
        ),
        "A": (
            "definition:spatialcf/upright-se2/objective-angular-geodesic/1.0",
            "definition:spatialcf/upright-se2/objective-selector-angular-geodesic/1.0",
            "definition:spatialcf/upright-se2/turn/1.0",
            "definition:spatialcf/upright-se2/normalizer-angular-geodesic/1.0",
        ),
        "R": (
            "definition:spatialcf/upright-se2/objective-nontarget-relation-damage/1.0",
            "definition:spatialcf/upright-se2/objective-selector-nontarget-relation-damage/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-nontarget-relation-damage/1.0",
        ),
        "V": (
            "definition:spatialcf/upright-se2/objective-visibility-change/1.0",
            "definition:spatialcf/upright-se2/objective-selector-visibility-change/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-visibility-change/1.0",
        ),
        "S": (
            "definition:spatialcf/upright-se2/objective-safety-margin-penalty/1.0",
            "definition:spatialcf/upright-se2/objective-selector-safety-margin-penalty/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-safety-margin-penalty/1.0",
        ),
    }[term_id]
    return {
        "objective_definition_ref": UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
        "metric_definition_ref": values[0],
        "input_selector_definition_ref": values[1],
        "unit_ref": values[2],
        "normalization_definition_ref": values[3],
        "normalizer_unit_ref": values[2],
    }


def _objective_policy_refs() -> dict[str, str]:
    return {
        "objective_definition_ref": UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
        "aggregation_definition_ref": "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
        "safety_penalty_definition_ref": "definition:spatialcf/upright-se2/objective-safety-penalty-from-hard-constraints/1.0",
        "strict_interval_comparator_ref": "definition:spatialcf/upright-se2/objective-strict-upper-lower-comparator/1.0",
        "exact_equality_definition_ref": "definition:spatialcf/upright-se2/objective-degenerate-exact-equality/1.0",
        "directed_gap_subtraction_definition_ref": "definition:spatialcf/upright-se2/objective-directed-gap-subtraction/1.0",
        "exact_prune_comparator_ref": "definition:spatialcf/upright-se2/objective-exact-strict-prune/1.0",
        "gap_prune_comparator_ref": "definition:spatialcf/upright-se2/objective-gap-closed-prune/1.0",
        "interval_boundary_policy_ref": "definition:spatialcf/upright-se2/objective-interval-boundary-policy/1.0",
        "deterministic_tie_break_definition_ref": "definition:spatialcf/upright-se2/objective-tie-break-T-R-V-S-A/1.0",
    }


def build_upright_se2_q0_target_objective_policy() -> UprightSE2FiveTermObjectivePolicy:
    """Build the explicit, source-independent unit target policy used by q=0."""

    return UprightSE2FiveTermObjectivePolicy.seal(
        **_objective_policy_refs(),
        terms=tuple(
            UprightSE2ObjectiveTermPolicy(
                term_id=term_id,
                **_objective_term_policy_values(term_id),
                weight=1.0,
                normalizer=1.0,
            )
            for term_id in ("T", "A", "R", "V", "S")
        ),
        owner_binding=_semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
    )


def build_upright_se2_objective_expression() -> ObjectiveExpression:
    """Return the fixed structural expression, without a numeric policy fallback."""

    return ObjectiveExpression.seal(
        aggregation_definition_ref=_objective_policy_refs()[
            "aggregation_definition_ref"
        ],
        terms=tuple(
            ObjectiveTerm(
                term_id=term_id,
                objective_definition_ref=UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
                input_selector_definition_ref=_objective_term_policy_values(term_id)[
                    "input_selector_definition_ref"
                ],
                unit_ref=_objective_term_policy_values(term_id)["unit_ref"],
                normalization_definition_ref=_objective_term_policy_values(term_id)[
                    "normalization_definition_ref"
                ],
            )
            for term_id in ("T", "A", "R", "V", "S")
        ),
        deterministic_tie_break_definition_ref=(
            _objective_policy_refs()["deterministic_tie_break_definition_ref"]
        ),
    )


def _semantic_owner_binding_payload(
    binding: UprightSE2SemanticOwnerBinding,
) -> TypedValue:
    return _semantic_typed_record(
        "schema:spatialcf/upright-se2/semantic-owner-binding/1.0",
        (
            ("definition_ref", _semantic_typed_id(binding.definition_ref)),
            (
                "evaluator_build_sha256",
                _semantic_typed_digest(binding.evaluator_build_sha256),
            ),
            (
                "evaluator_capability_ref",
                _semantic_typed_id(binding.evaluator_capability_ref),
            ),
            ("evaluator_owner_ref", _semantic_typed_id(binding.evaluator_owner_ref)),
            (
                "verifier_build_sha256",
                _semantic_typed_digest(binding.verifier_build_sha256),
            ),
            (
                "verifier_capability_ref",
                _semantic_typed_id(binding.verifier_capability_ref),
            ),
            ("verifier_owner_ref", _semantic_typed_id(binding.verifier_owner_ref)),
        ),
    )


def _semantic_definition_envelope(
    definition: UprightSE2SemanticDefinition,
) -> CanonicalDefinitionEnvelope:
    return CanonicalDefinitionEnvelope.seal(
        definition_ref=definition.definition_ref,
        definition_kind_ref=_SEMANTIC_DEFINITION_KIND_REF,
        payload_schema_ref=definition.payload_schema_ref,
        payload=_semantic_typed_record(
            definition.payload_schema_ref,
            (
                ("definition_kind", _semantic_typed_symbol(definition.definition_kind)),
                ("definition_ref", _semantic_typed_id(definition.definition_ref)),
                (
                    "owner_binding",
                    _semantic_owner_binding_payload(definition.owner_binding),
                ),
                ("semantic_body", definition.semantic_body),
                (
                    "semantic_definition_sha256",
                    _semantic_typed_digest(definition.semantic_definition_sha256),
                ),
            ),
        ),
    )


def _objective_term_payload(term: UprightSE2ObjectiveTermPolicy) -> TypedValue:
    return _semantic_typed_record(
        "schema:spatialcf/upright-se2/objective-term-policy/1.0",
        (
            (
                "input_selector_definition_ref",
                _semantic_typed_id(term.input_selector_definition_ref),
            ),
            ("metric_definition_ref", _semantic_typed_id(term.metric_definition_ref)),
            (
                "normalization_definition_ref",
                _semantic_typed_id(term.normalization_definition_ref),
            ),
            ("normalizer", _semantic_typed_real(term.normalizer)),
            ("normalizer_unit_ref", _semantic_typed_id(term.normalizer_unit_ref)),
            (
                "objective_definition_ref",
                _semantic_typed_id(term.objective_definition_ref),
            ),
            ("term_id", _semantic_typed_symbol(term.term_id)),
            ("unit_ref", _semantic_typed_id(term.unit_ref)),
            ("weight", _semantic_typed_real(term.weight)),
        ),
    )


def _objective_policy_envelope(
    policy: UprightSE2FiveTermObjectivePolicy,
) -> CanonicalDefinitionEnvelope:
    return CanonicalDefinitionEnvelope.seal(
        definition_ref=policy.objective_definition_ref,
        definition_kind_ref=_OBJECTIVE_DEFINITION_KIND_REF,
        payload_schema_ref=_OBJECTIVE_POLICY_SCHEMA_REF,
        payload=_semantic_typed_record(
            _OBJECTIVE_POLICY_SCHEMA_REF,
            (
                (
                    "aggregation_definition_ref",
                    _semantic_typed_id(policy.aggregation_definition_ref),
                ),
                (
                    "deterministic_tie_break_definition_ref",
                    _semantic_typed_id(policy.deterministic_tie_break_definition_ref),
                ),
                (
                    "directed_gap_subtraction_definition_ref",
                    _semantic_typed_id(policy.directed_gap_subtraction_definition_ref),
                ),
                (
                    "exact_equality_definition_ref",
                    _semantic_typed_id(policy.exact_equality_definition_ref),
                ),
                (
                    "exact_prune_comparator_ref",
                    _semantic_typed_id(policy.exact_prune_comparator_ref),
                ),
                (
                    "five_term_objective_policy_sha256",
                    _semantic_typed_digest(policy.five_term_objective_policy_sha256),
                ),
                (
                    "gap_prune_comparator_ref",
                    _semantic_typed_id(policy.gap_prune_comparator_ref),
                ),
                (
                    "interval_boundary_policy_ref",
                    _semantic_typed_id(policy.interval_boundary_policy_ref),
                ),
                (
                    "objective_definition_ref",
                    _semantic_typed_id(policy.objective_definition_ref),
                ),
                (
                    "owner_binding",
                    _semantic_owner_binding_payload(policy.owner_binding),
                ),
                (
                    "safety_penalty_definition_ref",
                    _semantic_typed_id(policy.safety_penalty_definition_ref),
                ),
                (
                    "strict_interval_comparator_ref",
                    _semantic_typed_id(policy.strict_interval_comparator_ref),
                ),
                (
                    "terms",
                    TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/objective-term-policy-list/1.0",
                        payload=FiniteOrderedTupleValue(
                            element_schema_ref="schema:spatialcf/upright-se2/objective-term-policy/1.0",
                            items=tuple(
                                _objective_term_payload(term) for term in policy.terms
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _semantic_definition_bundle_for_digest(
    profile_registration_sha256: Sha256Digest,
    objective_policy: UprightSE2FiveTermObjectivePolicy,
) -> DefinitionBundle:
    definitions = _upright_semantic_definitions()
    policy = objective_policy
    predicate_envelopes = tuple(
        _semantic_definition_envelope(definition) for definition in definitions
    )
    objective_envelope = _objective_policy_envelope(policy)
    closure_envelope = CanonicalDefinitionEnvelope.seal(
        definition_ref=UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF,
        definition_kind_ref=_SEMANTIC_CLOSURE_KIND_REF,
        payload_schema_ref=_SEMANTIC_CLOSURE_SCHEMA_REF,
        payload=_semantic_typed_record(
            _SEMANTIC_CLOSURE_SCHEMA_REF,
            (
                (
                    "definition_content_sha256s",
                    _semantic_typed_digest_tuple(
                        tuple(
                            definition.semantic_definition_sha256
                            for definition in definitions
                        )
                        + (policy.five_term_objective_policy_sha256,)
                    ),
                ),
                (
                    "objective_definition_ref",
                    _semantic_typed_id(policy.objective_definition_ref),
                ),
                (
                    "profile_registration_sha256",
                    _semantic_typed_digest(profile_registration_sha256),
                ),
                (
                    "semantic_definition_refs",
                    _semantic_typed_tuple(
                        tuple(definition.definition_ref for definition in definitions)
                        + (policy.objective_definition_ref,)
                    ),
                ),
            ),
        ),
    )
    return DefinitionBundle.seal(
        definitions=tuple(
            sorted(
                (*predicate_envelopes, objective_envelope, closure_envelope),
                key=lambda definition: canonical_json_bytes(definition.definition_ref),
            )
        )
    )


def _validate_task2_semantic_profile(
    registration: UprightSE2ProfileRegistration,
) -> None:
    """Require the complete Task 2 closure only where semantic evaluation begins.

    The Task 1 profile record remains a structural, capability-staged wire so
    its frozen profile tests can still construct the pre-evaluation form.  The
    compiler cannot reach a semantic bundle without this stricter M3 gate.
    """

    if (
        registration.semantics_profile.predicate_definition_refs
        != UPRIGHT_SE2_PREDICATE_DEFINITION_REFS
    ):
        raise ValueError(
            "semantics profile must name the complete upright predicate closure"
        )
    if registration.semantics_profile.objective_definition_refs != (
        UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
    ):
        raise ValueError("semantics profile must name the five-term objective policy")
    if (
        registration.action_space_profile.predicate_capability_refs
        != UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS
    ):
        raise ValueError(
            "action-space profile must name the semantic predicate capabilities"
        )
    if (
        registration.action_space_profile.objective_capability_refs
        != UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS
    ):
        raise ValueError(
            "action-space profile must name the semantic objective capabilities"
        )


def build_upright_se2_semantic_definition_bundle(
    registration: UprightSE2ProfileRegistration,
    *,
    objective_policy: UprightSE2FiveTermObjectivePolicy,
) -> DefinitionBundle:
    """Build the only profile/definition bundle accepted by the M3 compiler."""

    _validate_task2_semantic_profile(registration)
    return _semantic_definition_bundle_for_digest(
        registration.profile_registration_sha256, objective_policy
    )


def _upright_semantic_owner_bindings() -> tuple[UprightSE2SemanticOwnerBinding, ...]:
    return tuple(
        sorted(
            (
                *(
                    _semantic_owner_binding(definition_ref)
                    for definition_ref in UPRIGHT_SE2_PREDICATE_DEFINITION_REFS
                ),
                _semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
            ),
            key=canonical_json_bytes,
        )
    )


def _predicate_atom_from_context(context: object) -> PredicateAtom:
    if isinstance(context, (BeforePrecondition, AfterGoal, ObservationObligation)):
        formula = context.formula
        if type(formula) is PredicateAtom:
            return formula
    raise ValueError("semantic obligations must use direct grounded predicate atoms")


def _validate_grounded_semantic_shape(
    grounded_obligations: GroundedObligationSet,
) -> None:
    if len(grounded_obligations.before_preconditions) != 1:
        raise ValueError(
            "semantic closure requires one grounded target before-precondition"
        )
    if len(grounded_obligations.after_goals) != 1:
        raise ValueError("semantic closure requires one grounded target after-goal")
    if len(grounded_obligations.preservation_invariants) != 1:
        raise ValueError(
            "semantic closure requires one grounded preservation invariant"
        )
    if not grounded_obligations.observation_obligations:
        raise ValueError("semantic closure requires grounded visibility obligations")
    before = _predicate_atom_from_context(
        grounded_obligations.before_preconditions[0].context
    )
    after = _predicate_atom_from_context(grounded_obligations.after_goals[0].context)
    if (
        before.predicate_ref != UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
        or after.predicate_ref != UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
        or not before.operands
        or not after.operands
    ):
        raise ValueError(
            "semantic closure target relation obligations must be grounded"
        )
    preservation_context = grounded_obligations.preservation_invariants[0].context
    if type(preservation_context) is not PreservationInvariant:
        raise ValueError(
            "semantic closure preservation must use a transition invariant"
        )
    if (
        type(preservation_context.before_formula) is not PredicateAtom
        or type(preservation_context.after_formula) is not PredicateAtom
        or preservation_context.before_formula.predicate_ref
        != UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
        or preservation_context.after_formula.predicate_ref
        != UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
        or not preservation_context.before_formula.operands
        or not preservation_context.after_formula.operands
        or preservation_context.transition_comparator_ref
        != "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0"
    ):
        raise ValueError("semantic closure preservation invariant must be grounded")
    for obligation in grounded_obligations.observation_obligations:
        context = obligation.context
        if type(context) is not ObservationObligation:
            raise ValueError(
                "semantic closure observations must use observation obligations"
            )
        atom = _predicate_atom_from_context(context)
        if (
            atom.predicate_ref != UPRIGHT_SE2_VISIBILITY_PREDICATE_REF
            or not atom.operands
            or context.evidence_policy_ref
            != "definition:spatialcf/upright-se2/visibility-evidence-policy/1.0"
        ):
            raise ValueError("semantic closure visibility obligations must be grounded")


def _bound_source_input(problem: CounterfactualProblemIR) -> _UprightSE2SourceInput:
    """Decode the sole cardinal compiler-input fact from a bound semantic root."""

    bundles = problem.scene_state.extension_fact_bundles
    facts = tuple(fact for bundle in bundles for fact in bundle.facts)
    compiler_facts = tuple(
        fact
        for fact in facts
        if (
            fact.fact_family_ref == _UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF
            and fact.fact_key == _UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY
        )
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
            "semantic closure source problem must contain one compiler input, one executable policy, and at most one construction root"
        )
    request_bound_executable_policy_bundle_from_problem(problem)
    fact = compiler_facts[0]
    if (
        fact.fact_family_ref != _UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF
        or fact.fact_key != _UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY
        or fact.value.value_schema_ref != _UPRIGHT_SE2_COMPILER_INPUT_SCHEMA_REF
        or type(fact.value.payload) is not RecordValue
    ):
        raise ValueError(
            "semantic closure source compiler input has the wrong identity"
        )
    fields = fact.value.payload.fields
    expected_names = tuple(
        sorted(
            (
                "operation_kind",
                "operator_ref",
                "reference_id",
                "subject_id",
                "subject_yaw_turns",
                "yaw_argument",
            ),
            key=canonical_json_bytes,
        )
    )
    if tuple(field.name for field in fields) != expected_names:
        raise ValueError(
            "semantic closure source compiler input fields are not canonical"
        )
    values = {field.name: field.value for field in fields}
    if (
        _bound_symbol(values["operation_kind"], _UPRIGHT_SE2_ENUM_SCHEMA_REF)
        != "CARDINAL"
    ):
        raise ValueError("semantic closure source input must be a cardinal operation")
    subject_id = _bound_id(values["subject_id"], _UPRIGHT_SE2_ID_SCHEMA_REF)
    if fact.subject_entity_id != f"entity:{subject_id}":
        raise ValueError(
            "semantic closure source compiler input subject is not scene-owned"
        )
    yaw_argument = values["yaw_argument"]
    if (
        yaw_argument.value_schema_ref != _UPRIGHT_SE2_YAW_ARGUMENT_SCHEMA_REF
        or type(yaw_argument.payload) is not RecordValue
    ):
        raise ValueError("semantic closure source yaw argument has the wrong schema")
    yaw_fields = yaw_argument.payload.fields
    if tuple(field.name for field in yaw_fields) != tuple(
        sorted(("kind", "quarter_turns_ccw"), key=canonical_json_bytes)
    ):
        raise ValueError(
            "semantic closure source cardinal yaw fields are not canonical"
        )
    yaw_values = {field.name: field.value for field in yaw_fields}
    if _bound_symbol(yaw_values["kind"], _UPRIGHT_SE2_ENUM_SCHEMA_REF) != "CARDINAL":
        raise ValueError("semantic closure source yaw must be cardinal")
    return _UprightSE2SourceInput(
        operator_ref=_bound_id(values["operator_ref"], _UPRIGHT_SE2_ID_SCHEMA_REF),
        subject_id=subject_id,
        reference_id=_bound_id(values["reference_id"], _UPRIGHT_SE2_ID_SCHEMA_REF),
        subject_yaw_turns=CanonicalSO2Angle(
            turns=_bound_real(values["subject_yaw_turns"])
        ),
        quarter_turns_ccw=CardinalYaw(
            q=_bound_integer(yaw_values["quarter_turns_ccw"])
        ).q,
    )


def _bound_continuous_source_input(
    problem: CounterfactualProblemIR,
) -> _UprightSE2ContinuousSourceInput:
    """Decode the continuous wire without changing the cardinal decoder."""

    bundles = problem.scene_state.extension_fact_bundles
    facts = tuple(fact for bundle in bundles for fact in bundle.facts)
    compiler_facts = tuple(
        fact
        for fact in facts
        if (
            fact.fact_family_ref == _UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF
            and fact.fact_key == _UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY
        )
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
            "continuous semantic closure source problem must contain one compiler input, one executable policy, and at most one construction root"
        )
    request_bound_executable_policy_bundle_from_problem(problem)
    fact = compiler_facts[0]
    if (
        fact.fact_family_ref != _UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF
        or fact.fact_key != _UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY
        or fact.value.value_schema_ref != _UPRIGHT_SE2_COMPILER_INPUT_SCHEMA_REF
        or type(fact.value.payload) is not RecordValue
    ):
        raise ValueError(
            "continuous semantic closure source compiler input has the wrong identity"
        )
    fields = fact.value.payload.fields
    expected_names = tuple(
        sorted(
            (
                "operation_kind",
                "operator_ref",
                "reference_id",
                "subject_id",
                "subject_yaw_turns",
                "yaw_argument",
            ),
            key=canonical_json_bytes,
        )
    )
    if tuple(field.name for field in fields) != expected_names:
        raise ValueError(
            "continuous semantic closure source compiler input fields are not canonical"
        )
    values = {field.name: field.value for field in fields}
    if (
        _bound_symbol(values["operation_kind"], _UPRIGHT_SE2_ENUM_SCHEMA_REF)
        != "CONTINUOUS"
    ):
        raise ValueError(
            "continuous semantic closure source input must be a continuous operation"
        )
    subject_id = _bound_id(values["subject_id"], _UPRIGHT_SE2_ID_SCHEMA_REF)
    if fact.subject_entity_id != f"entity:{subject_id}":
        raise ValueError(
            "continuous semantic closure source compiler input subject is not scene-owned"
        )
    yaw_argument = values["yaw_argument"]
    if (
        yaw_argument.value_schema_ref != _UPRIGHT_SE2_YAW_ARGUMENT_SCHEMA_REF
        or type(yaw_argument.payload) is not RecordValue
    ):
        raise ValueError(
            "continuous semantic closure source yaw argument has the wrong schema"
        )
    yaw_fields = yaw_argument.payload.fields
    yaw_values = {field.name: field.value for field in yaw_fields}
    if "kind" not in yaw_values:
        raise ValueError("continuous semantic closure source yaw must declare a kind")
    kind = _bound_symbol(yaw_values["kind"], _UPRIGHT_SE2_ENUM_SCHEMA_REF)
    if kind == "ARC":
        if tuple(field.name for field in yaw_fields) != tuple(
            sorted(("ccw_sweep_turns", "kind", "start_turns"), key=canonical_json_bytes)
        ):
            raise ValueError(
                "continuous semantic closure arc yaw fields are not canonical"
            )
        yaw_domain: ContinuousYawDomain = ContinuousYawArc(
            start_angle=CanonicalSO2Angle(turns=_bound_real(yaw_values["start_turns"])),
            ccw_sweep_turns=_bound_real(yaw_values["ccw_sweep_turns"]),
        )
    elif kind == "FULL_CIRCLE":
        if tuple(field.name for field in yaw_fields) != ("kind",):
            raise ValueError(
                "continuous semantic closure full-circle yaw must not carry arc fields"
            )
        yaw_domain = ContinuousYawFullCircle()
    else:
        raise ValueError("continuous semantic closure yaw must be ARC or FULL_CIRCLE")
    return _UprightSE2ContinuousSourceInput(
        operator_ref=_bound_id(values["operator_ref"], _UPRIGHT_SE2_ID_SCHEMA_REF),
        subject_id=subject_id,
        reference_id=_bound_id(values["reference_id"], _UPRIGHT_SE2_ID_SCHEMA_REF),
        subject_yaw_turns=CanonicalSO2Angle(
            turns=_bound_real(values["subject_yaw_turns"])
        ),
        yaw_domain=yaw_domain,
    )


def _bound_real(value: TypedValue) -> float:
    if (
        value.value_schema_ref != _UPRIGHT_SE2_REAL_SCHEMA_REF
        or type(value.payload) is not FiniteRealValue
    ):
        raise ValueError("semantic closure source input must use finite real operands")
    return value.payload.value


def _bound_integer(value: TypedValue) -> int:
    if (
        value.value_schema_ref != _UPRIGHT_SE2_INTEGER_SCHEMA_REF
        or type(value.payload) is not IntegerValue
    ):
        raise ValueError("semantic closure source input must use integer operands")
    return value.payload.value


def _bound_id(value: TypedValue, schema_ref: str) -> CanonicalId:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not CanonicalIdValue
    ):
        raise ValueError("semantic closure source input must use canonical ID operands")
    return value.payload.value


def _bound_symbol(value: TypedValue, schema_ref: str) -> str:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not EnumSymbolValue
    ):
        raise ValueError("semantic closure source input must use symbol operands")
    return value.payload.symbol


def _bound_scene_object(scene: CanonicalScene, object_id: CanonicalId) -> object:
    objects = _exact_source_values(scene.objects, "objects")
    matching = tuple(object_ for object_ in objects if object_.object_id == object_id)
    if len(matching) != 1:
        raise ValueError(
            "semantic closure source input must name one exact scene object"
        )
    return matching[0]


def _bound_predicate_atom(formula: object) -> PredicateAtom:
    if type(formula) is not PredicateAtom or not formula.operands:
        raise ValueError(
            "semantic closure source obligations must use direct predicate atoms"
        )
    return formula


def _bound_reference(
    value: TypedValue,
    *,
    schema_ref: str,
    kind: ValueKind,
    expected: str | None = None,
) -> str:
    if (
        value.value_schema_ref != schema_ref
        or type(value.payload) is not ReferenceValue
        or value.payload.kind is not kind
    ):
        raise ValueError(
            "semantic closure source obligation has the wrong reference operand"
        )
    if expected is not None and value.payload.reference != expected:
        raise ValueError(
            "semantic closure source obligation does not bind request authority"
        )
    return value.payload.reference


def _validate_bound_target_atom(
    atom: PredicateAtom,
    *,
    subject_id: CanonicalId,
    reference_id: CanonicalId,
    phase: str,
) -> str:
    if (
        atom.predicate_ref != UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
        or len(atom.operands) != 4
    ):
        raise ValueError(
            "semantic closure source target relation has the wrong predicate"
        )
    _bound_reference(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
        expected=subject_id,
    )
    _bound_reference(
        atom.operands[1],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
        expected=reference_id,
    )
    relation = _bound_symbol(
        atom.operands[2],
        "schema:spatialcf/upright-se2/relation-symbol/1.0",
    )
    if relation not in {
        "relation:LEFT",
        "relation:RIGHT",
        "relation:FRONT",
        "relation:BEHIND",
        "relation:NEAR",
        "relation:FAR",
    }:
        raise ValueError("semantic closure source target relation is not registered")
    if (
        _bound_symbol(
            atom.operands[3],
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != f"phase:{phase}"
    ):
        raise ValueError("semantic closure source target relation has the wrong phase")
    return relation


def _validate_bound_preservation_atom(
    atom: PredicateAtom,
    *,
    subject_id: CanonicalId,
    phase: str,
) -> None:
    if (
        atom.predicate_ref != UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
        or len(atom.operands) != 3
    ):
        raise ValueError("semantic closure source preservation has the wrong predicate")
    _bound_reference(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/entity-ref/1.0",
        kind=ValueKind.ENTITY_REF,
        expected=f"entity:{subject_id}",
    )
    if (
        _bound_symbol(
            atom.operands[1],
            "schema:spatialcf/upright-se2/preservation-selector/1.0",
        )
        != "preservation:FROZEN_NONPRIMARY_LEAVES"
    ):
        raise ValueError(
            "semantic closure source preservation selector is not frozen-state"
        )
    if (
        _bound_symbol(
            atom.operands[2],
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != f"phase:{phase}"
    ):
        raise ValueError("semantic closure source preservation has the wrong phase")


def _bound_visibility_observation_id(
    obligation: ObservationObligation,
    scene: CanonicalScene,
) -> CanonicalId:
    if obligation.phase != "AFTER" or obligation.evidence_policy_ref != (
        "definition:spatialcf/upright-se2/visibility-evidence-policy/1.0"
    ):
        raise ValueError(
            "semantic closure source visibility policy is not fixed-camera AFTER"
        )
    atom = _bound_predicate_atom(obligation.formula)
    if (
        atom.predicate_ref != UPRIGHT_SE2_VISIBILITY_PREDICATE_REF
        or len(atom.operands) != 5
    ):
        raise ValueError("semantic closure source visibility has the wrong predicate")
    camera_id = _bound_reference(
        atom.operands[0],
        schema_ref="schema:spatialcf/upright-se2/camera-ref/1.0",
        kind=ValueKind.CAMERA_REF,
    )
    object_id = _bound_reference(
        atom.operands[1],
        schema_ref="schema:spatialcf/upright-se2/object-ref/1.0",
        kind=ValueKind.OBJECT_REF,
    )
    metric_definition_id = _bound_symbol(
        atom.operands[2],
        "schema:spatialcf/upright-se2/visibility-metric-symbol/1.0",
    )
    observation_id = _bound_id(
        atom.operands[3],
        "schema:spatialcf/upright-se2/observation-ref/1.0",
    )
    if (
        _bound_symbol(
            atom.operands[4],
            "schema:spatialcf/upright-se2/phase-symbol/1.0",
        )
        != "phase:AFTER"
    ):
        raise ValueError("semantic closure source visibility has the wrong phase")
    observations = _exact_source_values(
        scene.baseline_observations, "baseline observations"
    )
    matching = tuple(
        observation
        for observation in observations
        if observation.observation_id == observation_id
    )
    if len(matching) != 1:
        raise ValueError(
            "semantic closure source visibility names no exact observation"
        )
    observation = matching[0]
    if (
        camera_id != observation.camera_id
        or object_id != observation.object_id
        or metric_definition_id != observation.metric_definition_id
    ):
        raise ValueError(
            "semantic closure source visibility does not bind its exact observation"
        )
    return observation_id


def _validate_bound_semantic_problem(problem: CounterfactualProblemIR) -> None:
    """Close target, preservation, and fixed-camera obligations over the source scene."""

    source_input = _bound_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    if subject.object_id == reference.object_id or not subject.movable:
        raise ValueError(
            "semantic closure source input must name one movable subject and reference"
        )
    if (
        subject.pose.anchor_kind != "OBJECT_PIVOT"
        or reference.pose.anchor_kind != "OBJECT_PIVOT"
    ):
        raise ValueError("semantic closure source pivots must be exact object pivots")
    if (
        len(problem.before_preconditions) != 1
        or type(problem.before_preconditions[0]) is not BeforePrecondition
    ):
        raise ValueError(
            "semantic closure source problem requires one target before-precondition"
        )
    before_relation = _validate_bound_target_atom(
        _bound_predicate_atom(problem.before_preconditions[0].formula),
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        phase="BEFORE",
    )
    if type(problem.after_goal) is not AfterGoal:
        raise ValueError(
            "semantic closure source problem requires one target after-goal"
        )
    after_relation = _validate_bound_target_atom(
        _bound_predicate_atom(problem.after_goal.formula),
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        phase="AFTER",
    )
    if before_relation == after_relation:
        raise ValueError("semantic closure source target relations must differ")
    if (
        len(problem.preservation_invariants) != 1
        or type(problem.preservation_invariants[0]) is not PreservationInvariant
    ):
        raise ValueError(
            "semantic closure source problem requires one preservation invariant"
        )
    preservation = problem.preservation_invariants[0]
    if preservation.transition_comparator_ref != (
        "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0"
    ):
        raise ValueError("semantic closure source preservation comparator is not fixed")
    _validate_bound_preservation_atom(
        _bound_predicate_atom(preservation.before_formula),
        subject_id=source_input.subject_id,
        phase="BEFORE",
    )
    _validate_bound_preservation_atom(
        _bound_predicate_atom(preservation.after_formula),
        subject_id=source_input.subject_id,
        phase="AFTER",
    )
    expected_observations = _exact_source_values(
        scene.baseline_observations, "baseline observations"
    )
    if len(problem.explicit_observation_obligations) != len(expected_observations):
        raise ValueError(
            "semantic closure source visibility must cover every exact observation"
        )
    actual_observation_ids: set[CanonicalId] = set()
    for obligation in problem.explicit_observation_obligations:
        if type(obligation) is not ObservationObligation:
            raise ValueError(
                "semantic closure source visibility must use observation obligations"
            )
        observation_id = _bound_visibility_observation_id(obligation, scene)
        if observation_id in actual_observation_ids:
            raise ValueError(
                "semantic closure source visibility must not duplicate observations"
            )
        actual_observation_ids.add(observation_id)
    expected_observation_ids = {
        observation.observation_id for observation in expected_observations
    }
    if actual_observation_ids != expected_observation_ids:
        raise ValueError(
            "semantic closure source visibility roster must match the exact scene"
        )


def _bound_grounded_obligations(
    problem: CounterfactualProblemIR,
) -> GroundedObligationSet:
    """Derive the only grounded obligation roster from a sealed source problem."""

    _validate_bound_semantic_problem(problem)
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


def _validate_bound_continuous_semantic_problem(
    problem: CounterfactualProblemIR,
) -> None:
    """Close the retained semantic roster over one continuous compiler input."""

    source_input = _bound_continuous_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    if subject.object_id == reference.object_id or not subject.movable:
        raise ValueError(
            "continuous semantic closure source input must name one movable subject and reference"
        )
    if (
        subject.pose.anchor_kind != "OBJECT_PIVOT"
        or reference.pose.anchor_kind != "OBJECT_PIVOT"
    ):
        raise ValueError(
            "continuous semantic closure source pivots must be exact object pivots"
        )
    if (
        len(problem.before_preconditions) != 1
        or type(problem.before_preconditions[0]) is not BeforePrecondition
    ):
        raise ValueError(
            "continuous semantic closure source problem requires one target before-precondition"
        )
    before_relation = _validate_bound_target_atom(
        _bound_predicate_atom(problem.before_preconditions[0].formula),
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        phase="BEFORE",
    )
    if type(problem.after_goal) is not AfterGoal:
        raise ValueError(
            "continuous semantic closure source problem requires one target after-goal"
        )
    after_relation = _validate_bound_target_atom(
        _bound_predicate_atom(problem.after_goal.formula),
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        phase="AFTER",
    )
    if before_relation == after_relation:
        raise ValueError(
            "continuous semantic closure source target relations must differ"
        )
    if (
        len(problem.preservation_invariants) != 1
        or type(problem.preservation_invariants[0]) is not PreservationInvariant
    ):
        raise ValueError(
            "continuous semantic closure source problem requires one preservation invariant"
        )
    preservation = problem.preservation_invariants[0]
    if preservation.transition_comparator_ref != (
        "definition:spatialcf/upright-se2/preservation-transition-comparator/1.0"
    ):
        raise ValueError(
            "continuous semantic closure source preservation comparator is not fixed"
        )
    _validate_bound_preservation_atom(
        _bound_predicate_atom(preservation.before_formula),
        subject_id=source_input.subject_id,
        phase="BEFORE",
    )
    _validate_bound_preservation_atom(
        _bound_predicate_atom(preservation.after_formula),
        subject_id=source_input.subject_id,
        phase="AFTER",
    )
    expected_observations = _exact_source_values(
        scene.baseline_observations, "baseline observations"
    )
    if len(problem.explicit_observation_obligations) != len(expected_observations):
        raise ValueError(
            "continuous semantic closure source visibility must cover every exact observation"
        )
    actual_observation_ids: set[CanonicalId] = set()
    for obligation in problem.explicit_observation_obligations:
        if type(obligation) is not ObservationObligation:
            raise ValueError(
                "continuous semantic closure source visibility must use observation obligations"
            )
        observation_id = _bound_visibility_observation_id(obligation, scene)
        if observation_id in actual_observation_ids:
            raise ValueError(
                "continuous semantic closure source visibility must not duplicate observations"
            )
        actual_observation_ids.add(observation_id)
    expected_observation_ids = {
        observation.observation_id for observation in expected_observations
    }
    if actual_observation_ids != expected_observation_ids:
        raise ValueError(
            "continuous semantic closure visibility roster must match the exact scene"
        )


def _bound_continuous_grounded_obligations(
    problem: CounterfactualProblemIR,
) -> GroundedObligationSet:
    """Derive the continuous closure's source-bound obligation roster."""

    _validate_bound_continuous_semantic_problem(problem)
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


def build_upright_se2_semantic_closure(
    *,
    profile_registration: UprightSE2ProfileRegistration,
    semantic_problem: CounterfactualProblemIR,
    definition_bundle: DefinitionBundle,
    grounded_obligations: GroundedObligationSet,
    objective_expression: ObjectiveExpression,
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> UprightSE2SemanticClosure:
    """Seal all definitions, operands, objective policy, and owner builds together."""

    objective_policy = decode_upright_se2_objective_policy(executable_policy_bundle)
    return UprightSE2SemanticClosure.seal(
        profile_registration_sha256=profile_registration.profile_registration_sha256,
        profile_registration=profile_registration,
        semantic_problem=semantic_problem,
        definition_bundle=definition_bundle,
        predicate_definitions=_upright_semantic_definitions(),
        objective_policy=objective_policy,
        evaluator_bindings=_upright_semantic_owner_bindings(),
        grounded_obligations=grounded_obligations,
        objective_expression=objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=resource_policy,
    )


def build_upright_se2_continuous_semantic_closure(
    *,
    profile_registration: UprightSE2ProfileRegistration,
    semantic_problem: CounterfactualProblemIR,
    definition_bundle: DefinitionBundle,
    grounded_obligations: GroundedObligationSet,
    objective_expression: ObjectiveExpression,
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> UprightSE2ContinuousSemanticClosure:
    """Seal continuous-yaw semantics without widening the cardinal closure."""

    objective_policy = decode_upright_se2_objective_policy(executable_policy_bundle)
    return UprightSE2ContinuousSemanticClosure.seal(
        profile_registration_sha256=profile_registration.profile_registration_sha256,
        profile_registration=profile_registration,
        semantic_problem=semantic_problem,
        definition_bundle=definition_bundle,
        predicate_definitions=_upright_semantic_definitions(),
        objective_policy=objective_policy,
        evaluator_bindings=_upright_semantic_owner_bindings(),
        grounded_obligations=grounded_obligations,
        objective_expression=objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=resource_policy,
    )


class UprightSE2BackendAvailability(HashBoundCanonicalModel):
    """Operational availability is separate from the immutable profile hash."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/backend-availability/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_availability_sha256"

    profile_registration_sha256: Sha256Digest
    available_capability_refs: tuple[CapabilityRef, ...]
    backend_availability_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_staged_availability(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.available_capability_refs,
            "available staged capabilities",
        )
        if self.available_capability_refs not in (
            UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
            UPRIGHT_SE2_STAGED_CAPABILITY_REFS,
        ):
            raise ValueError(
                "availability must be cardinal-only or cardinal-plus-continuous"
            )
        return self


class UprightSE2CompiledCell(HashBoundCanonicalModel):
    """A declarative exact-dyadic M3 cell, not a compiled solver implementation."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/compiled-cell/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "compiled_cell_sha256"

    cell_id: CanonicalId
    authorization_sha256: Sha256Digest
    x_lower: ExactDyadic
    x_upper: ExactDyadic
    y_lower: ExactDyadic
    y_upper: ExactDyadic
    yaw_interval: LiftedYawInterval
    compiled_cell_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_cell_bounds(self) -> Self:
        if not self.cell_id.startswith("cell:"):
            raise ValueError("compiled cell IDs must use the cell namespace")
        if self.x_lower.as_fraction > self.x_upper.as_fraction:
            raise ValueError("compiled cell x bounds must be ordered")
        if self.y_lower.as_fraction > self.y_upper.as_fraction:
            raise ValueError("compiled cell y bounds must be ordered")
        return self


def _lifted_cell_order_key(
    cell: UprightSE2CompiledCell,
) -> tuple[Fraction, Fraction, bytes]:
    """Return the frozen lifted-yaw order with a canonical cell-ID tie-break."""

    return (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        canonical_json_bytes(cell.cell_id),
    )


class UprightSE2CoverageArtifact(HashBoundCanonicalModel):
    """Canonical roster/coverage data that remains untrusted until checker replay."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/coverage-artifact/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "coverage_artifact_sha256"

    authorization_sha256: Sha256Digest
    cells: tuple[UprightSE2CompiledCell, ...]
    unresolved_cell_sha256s: tuple[Sha256Digest, ...]
    coverage_artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_coverage_roster(self) -> Self:
        if not self.cells:
            raise ValueError("coverage artifact must name at least one compiled cell")
        if self.cells != tuple(sorted(self.cells, key=_lifted_cell_order_key)):
            raise ValueError("coverage cells must use ascending exact lifted yaw order")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("coverage artifact must not duplicate a cell ID")
        if self.unresolved_cell_sha256s != tuple(
            sorted(set(self.unresolved_cell_sha256s))
        ):
            raise ValueError("unresolved cell digests must be sorted and unique")
        if any(
            cell.authorization_sha256 != self.authorization_sha256
            for cell in self.cells
        ):
            raise ValueError("coverage cells must bind the same authorization")
        return self


class UprightSE2ProofLeafDisposition(StrEnum):
    """The closed leaf status a later checker must replay without relabelling."""

    INWARD_FEASIBLE = "INWARD_FEASIBLE"
    OUTWARD_INFEASIBLE = "OUTWARD_INFEASIBLE"
    PRUNED = "PRUNED"
    UNRESOLVED = "UNRESOLVED"


class UprightSE2RetainedOwnerOutcomeKind(StrEnum):
    """The finite retained-owner result alphabet carried by one proof row."""

    EXACT = "EXACT"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE = "INCOMPLETE"
    FINITE_MISS = "FINITE_MISS"


def _proof_usage_total(usages: tuple[ResourceUsage, ...]) -> ResourceUsage:
    """Derive one canonical shared-ledger total from non-reset stage usages."""

    if not usages:
        raise ValueError("proof resource aggregation requires at least one usage")
    accounting_refs = {usage.accounting_claim_definition_ref for usage in usages}
    if len(accounting_refs) != 1:
        raise ValueError("proof resource usages must use one accounting claim")
    totals: dict[str, float] = {}
    for usage in usages:
        for entry in usage.entries:
            totals[entry.resource_definition_ref] = (
                totals.get(entry.resource_definition_ref, 0.0) + entry.used
            )
    entry_values = tuple(
        {
            "resource_definition_ref": resource_definition_ref,
            "used": used,
        }
        for resource_definition_ref, used in sorted(
            totals.items(), key=lambda item: canonical_json_bytes(item[0])
        )
    )
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": next(iter(accounting_refs)),
            "entries": entry_values,
            "exhausted": any(usage.exhausted for usage in usages),
        }
    )


class UprightSE2CardinalProofTuple(HashBoundCanonicalModel):
    """The one request-authorized operator/reference/pivot/q proof tuple."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/cardinal-proof-tuple/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "cardinal_proof_tuple_sha256"

    authorization: CardinalYawAuthorization
    reference_id: CanonicalId
    translation_domain: UprightSE2TranslationDomain
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    cardinal_proof_tuple_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_tuple_cells(self) -> Self:
        if not self.compiled_cells:
            raise ValueError("proof tuple must retain at least one compiled cell")
        if tuple(cell.authorization_sha256 for cell in self.compiled_cells) != (
            self.authorization.cardinal_yaw_authorization_sha256,
        ) * len(self.compiled_cells):
            raise ValueError("proof tuple cells must bind its authorization")
        if self.compiled_cells != tuple(
            sorted(self.compiled_cells, key=_lifted_cell_order_key)
        ):
            raise ValueError("proof tuple cells must use canonical lifted-yaw order")
        if len({cell.cell_id for cell in self.compiled_cells}) != len(
            self.compiled_cells
        ):
            raise ValueError("proof tuple must not duplicate a compiled cell")
        return self


class UprightSE2RetainedOwnerEvaluation(HashBoundCanonicalModel):
    """One exact retained-owner outcome, bounds, findings, proof rows, and delta."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/retained-owner-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "retained_owner_evaluation_sha256"

    compiled_cell: UprightSE2CompiledCell
    owner_ref: OwnerRef
    evaluator_capability_ref: CapabilityRef
    outcome_kind: UprightSE2RetainedOwnerOutcomeKind
    exact_bounds: tuple[TypedValue, ...]
    finding_codes: tuple[CanonicalId, ...]
    proof_rows: tuple[CanonicalId, ...]
    resource_delta: ResourceUsage
    retained_owner_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_retained_owner_outcome(self) -> Self:
        _require_sorted_unique_by_bytes(self.exact_bounds, "proof exact bounds")
        _require_sorted_unique_by_bytes(self.finding_codes, "proof finding codes")
        _require_sorted_unique_by_bytes(self.proof_rows, "proof rows")
        if not self.proof_rows:
            raise ValueError("retained owner evaluation must retain proof rows")
        if any(type(bound.payload) is DigestValue for bound in self.exact_bounds):
            raise ValueError("proof exact bounds must not be digest-only")
        if self.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT:
            if not self.exact_bounds or self.finding_codes:
                raise ValueError(
                    "exact retained owner evaluations require bounds and no finding"
                )
            return self
        if self.exact_bounds or not self.finding_codes:
            raise ValueError(
                "nonexact retained owner evaluations require findings and no exact bounds"
            )
        expected_prefix = f"finding:spatialcf/upright-se2/proof-transport/{self.outcome_kind.value.lower()}"
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("retained owner finding must match its typed outcome")
        return self


class UprightSE2ProofCellEvaluation(HashBoundCanonicalModel):
    """Every retained-owner result for one exact cell and optional leaf status."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-cell-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_cell_evaluation_sha256"

    compiled_cell: UprightSE2CompiledCell
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...]
    leaf_disposition: UprightSE2ProofLeafDisposition | None
    complete_domain_empty: StrictBool | None
    proof_cell_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_leaf_disposition(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.owner_evaluations,
            "proof cell retained owner evaluations",
        )
        if not self.owner_evaluations:
            raise ValueError("proof cell must retain owner evaluations")
        if any(
            canonical_json_bytes(evaluation.compiled_cell)
            != canonical_json_bytes(self.compiled_cell)
            for evaluation in self.owner_evaluations
        ):
            raise ValueError("proof owner evaluation must bind its exact common cell")
        all_exact = all(
            evaluation.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT
            for evaluation in self.owner_evaluations
        )
        any_nonexact = not all_exact
        if self.leaf_disposition is None:
            if self.complete_domain_empty is not None:
                raise ValueError(
                    "internal proof cells must not claim complete-domain empty"
                )
            return self
        if self.complete_domain_empty is None:
            raise ValueError("proof leaf disposition requires an explicit empty claim")
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE:
            if not all_exact or self.complete_domain_empty:
                raise ValueError(
                    "inward-feasible proof cells require exact nonempty evidence"
                )
            return self
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE:
            if not all_exact or not self.complete_domain_empty:
                raise ValueError(
                    "complete-domain empty proof cells require exact outward-infeasible evidence"
                )
            return self
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED:
            if self.complete_domain_empty:
                raise ValueError(
                    "pruned proof cells cannot claim complete-domain empty"
                )
            return self
        if self.complete_domain_empty or not any_nonexact:
            raise ValueError(
                "unresolved proof cells require nonexact evidence and no complete-domain empty claim"
            )
        return self


class UprightSE2ExactRational(CanonicalModel):
    """One normalized exact rational retained in profile proof transport."""

    numerator: StrictInt
    denominator: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def _validate_normalized_rational(self) -> Self:
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("exact rational values must be normalized")
        return self

    @property
    def as_fraction(self) -> Fraction:
        """Return the sole exact arithmetic representation for profile checks."""

        return Fraction(self.numerator, self.denominator)


class UprightSE2SolvePolicyDefinitionPayload(CanonicalModel):
    """Request-owned finite-gap and claim-strength policy for M3 replay."""

    profile_registration_sha256: Sha256Digest
    requested_gap: UprightSE2ExactRational
    objective_bound_policy_ref: DefinitionRef
    exact_global_claim_definition_ref: DefinitionRef
    finite_gap_claim_definition_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_requested_gap_and_claims(self) -> Self:
        if self.requested_gap.as_fraction < 0:
            raise ValueError("requested gap must be non-negative")
        if (
            self.exact_global_claim_definition_ref
            == self.finite_gap_claim_definition_ref
        ):
            raise ValueError("exact and finite-gap claims must be distinct")
        return self


def _solve_policy_rational_value(value: UprightSE2ExactRational) -> TypedValue:
    return _materialized_state_record(
        _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
        (
            (
                "denominator",
                TypedValue(
                    value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                    payload=IntegerValue(value=value.denominator),
                ),
            ),
            (
                "numerator",
                TypedValue(
                    value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                    payload=IntegerValue(value=value.numerator),
                ),
            ),
        ),
    )


def build_upright_se2_solve_policy_definition_bundle(
    registration: UprightSE2ProfileRegistration,
    *,
    requested_gap: UprightSE2ExactRational,
    objective_bound_policy_ref: DefinitionRef,
    exact_global_claim_definition_ref: DefinitionRef = UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF,
    finite_gap_claim_definition_ref: DefinitionRef = UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
) -> DefinitionBundle:
    """Seal the request's exact gap and claim policy into its definition closure."""

    if type(registration) is not UprightSE2ProfileRegistration:
        raise TypeError("solve policy requires an exact upright se2 registration")
    if type(requested_gap) is not UprightSE2ExactRational:
        raise TypeError("solve policy requested gap requires an exact rational")
    payload = UprightSE2SolvePolicyDefinitionPayload(
        profile_registration_sha256=registration.profile_registration_sha256,
        requested_gap=requested_gap,
        objective_bound_policy_ref=objective_bound_policy_ref,
        exact_global_claim_definition_ref=exact_global_claim_definition_ref,
        finite_gap_claim_definition_ref=finite_gap_claim_definition_ref,
    )
    fields = (
        (
            "exact_global_claim_definition_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(
                    value=payload.exact_global_claim_definition_ref
                ),
            ),
        ),
        (
            "finite_gap_claim_definition_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(value=payload.finite_gap_claim_definition_ref),
            ),
        ),
        (
            "objective_bound_policy_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(value=payload.objective_bound_policy_ref),
            ),
        ),
        (
            "profile_registration_sha256",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_DIGEST_SCHEMA_REF,
                payload=DigestValue(value=payload.profile_registration_sha256),
            ),
        ),
        ("requested_gap", _solve_policy_rational_value(payload.requested_gap)),
    )
    return DefinitionBundle.seal(
        definitions=(
            CanonicalDefinitionEnvelope.seal(
                definition_ref=UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF,
                definition_kind_ref="definition:spatialcf/upright-se2/definition-kind/1.0",
                payload_schema_ref=UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF,
                payload=_materialized_state_record(
                    UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF, fields
                ),
            ),
        )
    )


def decode_upright_se2_solve_policy_definition_payload(
    bundle: DefinitionBundle,
) -> UprightSE2SolvePolicyDefinitionPayload:
    """Decode only the registered canonical request-owned solve-policy payload."""

    if type(bundle) is not DefinitionBundle or len(bundle.definitions) != 1:
        raise ValueError("solve policy bundle must contain one registered definition")
    definition = bundle.definitions[0]
    if (
        definition.definition_ref != UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF
        or definition.payload_schema_ref != UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF
        or definition.payload.value_schema_ref
        != UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF
        or type(definition.payload.payload) is not RecordValue
    ):
        raise ValueError("solve policy bundle has the wrong registered payload")
    fields = {field.name: field.value for field in definition.payload.payload.fields}
    if set(fields) != {
        "exact_global_claim_definition_ref",
        "finite_gap_claim_definition_ref",
        "objective_bound_policy_ref",
        "profile_registration_sha256",
        "requested_gap",
    }:
        raise ValueError("solve policy payload fields are not closed")
    profile = fields["profile_registration_sha256"]
    requested_gap = fields["requested_gap"]
    if (
        type(profile.payload) is not DigestValue
        or type(requested_gap.payload) is not RecordValue
    ):
        raise ValueError("solve policy payload has invalid scalar forms")
    gap_fields = {field.name: field.value for field in requested_gap.payload.fields}
    if set(gap_fields) != {"numerator", "denominator"} or any(
        type(gap_fields[name].payload) is not IntegerValue
        for name in ("numerator", "denominator")
    ):
        raise ValueError("solve policy requested gap is not an exact rational")

    def canonical_id(name: str) -> str:
        value = fields[name]
        if type(value.payload) is not CanonicalIdValue:
            raise ValueError("solve policy reference is not canonical")
        return value.payload.value

    return UprightSE2SolvePolicyDefinitionPayload(
        profile_registration_sha256=profile.payload.value,
        requested_gap=UprightSE2ExactRational(
            numerator=gap_fields["numerator"].payload.value,
            denominator=gap_fields["denominator"].payload.value,
        ),
        objective_bound_policy_ref=canonical_id("objective_bound_policy_ref"),
        exact_global_claim_definition_ref=canonical_id(
            "exact_global_claim_definition_ref"
        ),
        finite_gap_claim_definition_ref=canonical_id("finite_gap_claim_definition_ref"),
    )


class UprightSE2ProposalPointTerm(CanonicalModel):
    """One ordered exact retained semantic T/A/R/V/S point-term interval."""

    term_id: Literal["T", "A", "R", "V", "S"]
    lower: UprightSE2ExactRational
    upper: UprightSE2ExactRational

    @model_validator(mode="after")
    def _validate_ordered_interval(self) -> Self:
        if self.lower.as_fraction > self.upper.as_fraction:
            raise ValueError("point term interval bounds must be ordered")
        return self


class UprightSE2ProposalPointObjective(CanonicalModel):
    """The complete interval point objective copied from retained owner output."""

    terms: tuple[UprightSE2ProposalPointTerm, ...]
    total_lower: UprightSE2ExactRational
    total_upper: UprightSE2ExactRational

    @model_validator(mode="after")
    def _validate_complete_interval_point_objective(self) -> Self:
        if tuple(term.term_id for term in self.terms) != ("T", "A", "R", "V", "S"):
            raise ValueError("point objective must contain ordered T/A/R/V/S terms")
        if self.total_lower.as_fraction > self.total_upper.as_fraction:
            raise ValueError("point objective total interval bounds must be ordered")
        return self


_UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-objective/2.0"
)
_UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-term/2.0"
)
_UPRIGHT_SE2_RETAINED_POINT_TERM_ROSTER_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/retained-point-term-roster/2.0"
)
_UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF = (
    "schema:spatialcf/upright-se2/exact-rational/1.0"
)


def _retained_point_objective_value(
    objective: UprightSE2ProposalPointObjective,
) -> TypedValue:
    """Encode the exact owner output transported into a proposal point record."""

    def rational(value: UprightSE2ExactRational) -> TypedValue:
        return _materialized_state_record(
            _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
            (
                (
                    "numerator",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                        payload=IntegerValue(value=value.numerator),
                    ),
                ),
                (
                    "denominator",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                        payload=IntegerValue(value=value.denominator),
                    ),
                ),
            ),
        )

    terms = tuple(
        _materialized_state_record(
            _UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF,
            (
                (
                    "term_id",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_ENUM_SCHEMA_REF,
                        payload=EnumSymbolValue(symbol=term.term_id),
                    ),
                ),
                ("lower", rational(term.lower)),
                ("upper", rational(term.upper)),
            ),
        )
        for term in objective.terms
    )
    return _materialized_state_record(
        _UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF,
        (
            (
                "terms",
                TypedValue(
                    value_schema_ref=(
                        _UPRIGHT_SE2_RETAINED_POINT_TERM_ROSTER_SCHEMA_REF
                    ),
                    payload=FiniteOrderedTupleValue(
                        element_schema_ref=_UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF,
                        items=terms,
                    ),
                ),
            ),
            ("total_lower", rational(objective.total_lower)),
            ("total_upper", rational(objective.total_upper)),
        ),
    )


def _retained_point_owner_objective(
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...],
) -> TypedValue:
    """Require exactly one retained exact-bound payload for the point objective."""

    matches = tuple(
        bound
        for evaluation in owner_evaluations
        for bound in evaluation.exact_bounds
        if bound.value_schema_ref == _UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF
    )
    if len(matches) != 1:
        raise ValueError(
            "proposal point evaluation requires one retained point owner objective"
        )
    return matches[0]


class UprightSE2ProposalPointEvaluation(HashBoundCanonicalModel):
    """Exact retained point-cell output transported without invoking a kernel."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proposal-point-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proposal_point_evaluation_sha256"

    point_cell_evaluation: UprightSE2ProofCellEvaluation
    point_objective: UprightSE2ProposalPointObjective
    proposal_point_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_exact_point_transport(self) -> Self:
        cell_evaluation = self.point_cell_evaluation
        cell = cell_evaluation.compiled_cell
        if (
            cell_evaluation.leaf_disposition is None
            and cell_evaluation.complete_domain_empty is not None
        ):
            raise ValueError("proposal point evaluation must retain an internal cell")
        if cell_evaluation.leaf_disposition is not None and (
            cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
            or cell_evaluation.complete_domain_empty is not False
        ):
            raise ValueError(
                "proposal point evaluation may reuse only an inward-feasible final cell"
            )
        if cell.x_lower != cell.x_upper or cell.y_lower != cell.y_upper:
            raise ValueError(
                "proposal point evaluation requires a degenerate point cell"
            )
        if any(
            evaluation.outcome_kind is not UprightSE2RetainedOwnerOutcomeKind.EXACT
            for evaluation in cell_evaluation.owner_evaluations
        ):
            raise ValueError(
                "proposal point evaluation requires exact retained owner evidence"
            )
        retained_objective = _retained_point_owner_objective(
            cell_evaluation.owner_evaluations
        )
        if canonical_json_bytes(retained_objective) != canonical_json_bytes(
            _retained_point_objective_value(self.point_objective)
        ):
            raise ValueError(
                "proposal point objective must byte-bind retained point owner output"
            )
        return self


class UprightSE2ProofPruneDecision(HashBoundCanonicalModel):
    """One inspectable deterministic prune decision for a retained cell row."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-prune-decision/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_prune_decision_sha256"

    cell_evaluation: UprightSE2ProofCellEvaluation
    prune_reason_codes: tuple[CanonicalId, ...]
    proof_prune_decision_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_prune_decision(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.prune_reason_codes,
            "proof prune reason codes",
        )
        if (
            self.cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.PRUNED
            or not self.prune_reason_codes
        ):
            raise ValueError("proof prune decision must retain one pruned cell reason")
        return self


class UprightSE2ProofFrontierRow(HashBoundCanonicalModel):
    """One final unresolved frontier row with the full nonexact cell evidence."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-frontier-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_frontier_row_sha256"

    cell_evaluation: UprightSE2ProofCellEvaluation
    frontier_reason_codes: tuple[CanonicalId, ...]
    proof_frontier_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_frontier_row(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.frontier_reason_codes,
            "proof frontier reason codes",
        )
        if (
            self.cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.UNRESOLVED
            or not self.frontier_reason_codes
        ):
            raise ValueError(
                "proof frontier row must retain one unresolved cell reason"
            )
        return self


class UprightSE2ProofStageDelta(HashBoundCanonicalModel):
    """One shared-ledger stage that retains its full owner rows and resource delta."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-stage-delta/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_stage_delta_sha256"

    stage_ref: CanonicalId
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...]
    resource_delta: ResourceUsage
    proof_stage_delta_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_stage_delta(self) -> Self:
        if not self.stage_ref.startswith("stage:"):
            raise ValueError("proof stage references must use the stage namespace")
        _require_sorted_unique_by_bytes(
            self.owner_evaluations,
            "proof stage owner evaluations",
        )
        if not self.owner_evaluations:
            raise ValueError("proof stage must retain at least one owner evaluation")
        expected = _proof_usage_total(
            tuple(evaluation.resource_delta for evaluation in self.owner_evaluations)
        )
        if canonical_json_bytes(self.resource_delta) != canonical_json_bytes(expected):
            raise ValueError("proof stage resource delta does not match owner rows")
        return self


class UprightSE2ProofResourceLedger(HashBoundCanonicalModel):
    """The single canonical total resource ledger carried by the proof payload."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-resource-ledger/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_resource_ledger_sha256"

    stage_deltas: tuple[UprightSE2ProofStageDelta, ...]
    canonical_total_resource_usage: ResourceUsage
    proof_resource_ledger_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_shared_ledger(self) -> Self:
        _require_sorted_unique_by_bytes(self.stage_deltas, "proof stage deltas")
        if not self.stage_deltas:
            raise ValueError("proof resource ledger must retain stage deltas")
        if len({stage.stage_ref for stage in self.stage_deltas}) != len(
            self.stage_deltas
        ):
            raise ValueError("proof stage references must be unique")
        expected = _proof_usage_total(
            tuple(stage.resource_delta for stage in self.stage_deltas)
        )
        if canonical_json_bytes(
            self.canonical_total_resource_usage
        ) != canonical_json_bytes(expected):
            raise ValueError(
                "proof resource ledger canonical total does not match stages"
            )
        return self


def _proof_cell_is_within_root(
    cell: UprightSE2CompiledCell,
    root: UprightSE2CompiledCell,
) -> bool:
    """Return whether one proof row is a yaw-bound exact-dyadic root descendant."""

    return (
        cell.authorization_sha256 == root.authorization_sha256
        and cell.yaw_interval == root.yaw_interval
        and root.x_lower.as_fraction <= cell.x_lower.as_fraction
        and cell.x_upper.as_fraction <= root.x_upper.as_fraction
        and root.y_lower.as_fraction <= cell.y_lower.as_fraction
        and cell.y_upper.as_fraction <= root.y_upper.as_fraction
    )


def _proof_root_for_cell(
    cell: UprightSE2CompiledCell,
    roots: tuple[UprightSE2CompiledCell, ...],
) -> UprightSE2CompiledCell:
    """Resolve exactly one compilation root for a retained proof cell."""

    matches = tuple(root for root in roots if _proof_cell_is_within_root(cell, root))
    if len(matches) != 1:
        raise ValueError(
            "proof cells must bind one authorization/yaw compilation root domain"
        )
    return matches[0]


def _proof_validate_leaf_partition(
    roots: tuple[UprightSE2CompiledCell, ...],
    leaves: tuple[UprightSE2CompiledCell, ...],
) -> None:
    """Require final leaves to partition roots with lower-owned split seams.

    Retained-owner geometry remains closed.  Proof coverage instead assigns an
    internal X/Y seam to its lower child, while retaining the root's outer
    upper boundary.  A degenerate axis has one owned coordinate and cannot be
    widened or subdivided into synthetic area.
    """

    leaves_by_root: dict[bytes, list[UprightSE2CompiledCell]] = {
        canonical_json_bytes(root): [] for root in roots
    }
    for leaf in leaves:
        root = _proof_root_for_cell(leaf, roots)
        if (
            root.x_lower.as_fraction < root.x_upper.as_fraction
            and leaf.x_lower.as_fraction >= leaf.x_upper.as_fraction
        ) or (
            root.y_lower.as_fraction < root.y_upper.as_fraction
            and leaf.y_lower.as_fraction >= leaf.y_upper.as_fraction
        ):
            raise ValueError(
                "proof final leaves must not collapse a nondegenerate root axis"
            )
        leaves_by_root[canonical_json_bytes(root)].append(leaf)

    for root in roots:
        root_key = canonical_json_bytes(root)
        root_leaves = tuple(leaves_by_root[root_key])
        if not root_leaves:
            raise ValueError("proof coverage leaves must cover every compilation root")
        x_coordinates = tuple(
            sorted(
                {
                    root.x_lower.as_fraction,
                    root.x_upper.as_fraction,
                    *(
                        coordinate
                        for leaf in root_leaves
                        for coordinate in (
                            leaf.x_lower.as_fraction,
                            leaf.x_upper.as_fraction,
                        )
                    ),
                }
            )
        )
        y_coordinates = tuple(
            sorted(
                {
                    root.y_lower.as_fraction,
                    root.y_upper.as_fraction,
                    *(
                        coordinate
                        for leaf in root_leaves
                        for coordinate in (
                            leaf.y_lower.as_fraction,
                            leaf.y_upper.as_fraction,
                        )
                    ),
                }
            )
        )
        x_segments = (
            ((root.x_lower.as_fraction, root.x_upper.as_fraction),)
            if root.x_lower.as_fraction == root.x_upper.as_fraction
            else tuple(pairwise(x_coordinates))
        )
        y_segments = (
            ((root.y_lower.as_fraction, root.y_upper.as_fraction),)
            if root.y_lower.as_fraction == root.y_upper.as_fraction
            else tuple(pairwise(y_coordinates))
        )
        for x_lower, x_upper in x_segments:
            for y_lower, y_upper in y_segments:
                covering = tuple(
                    leaf
                    for leaf in root_leaves
                    if (
                        leaf.x_lower.as_fraction <= x_lower
                        and x_upper <= leaf.x_upper.as_fraction
                        and leaf.y_lower.as_fraction <= y_lower
                        and y_upper <= leaf.y_upper.as_fraction
                    )
                )
                if len(covering) != 1:
                    raise ValueError(
                        "proof coverage leaves must form one exact-dyadic partition"
                    )
        for x_coordinate in x_coordinates:
            for y_coordinate in y_coordinates:
                owning = tuple(
                    leaf
                    for leaf in root_leaves
                    if _proof_leaf_owns_coordinate(
                        lower=leaf.x_lower.as_fraction,
                        upper=leaf.x_upper.as_fraction,
                        coordinate=x_coordinate,
                        root_lower=root.x_lower.as_fraction,
                    )
                    and _proof_leaf_owns_coordinate(
                        lower=leaf.y_lower.as_fraction,
                        upper=leaf.y_upper.as_fraction,
                        coordinate=y_coordinate,
                        root_lower=root.y_lower.as_fraction,
                    )
                )
                if len(owning) != 1:
                    raise ValueError(
                        "proof coverage leaves must assign each split seam to one lower owner"
                    )


def _proof_leaf_owns_coordinate(
    *,
    lower: Fraction,
    upper: Fraction,
    coordinate: Fraction,
    root_lower: Fraction,
) -> bool:
    """Apply proof-only lower ownership while retaining the root outer lower."""

    return (coordinate == root_lower and lower == root_lower) or (
        lower < coordinate <= upper
    )


class UprightSE2ProofMaterial(HashBoundCanonicalModel):
    """Full hash-bound M3 cardinal proof material prior to independent checking."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-material/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_material_sha256"

    solve_request_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2Compilation
    cardinal_tuple_roster: tuple[UprightSE2CardinalProofTuple, ...]
    coverage_artifact: UprightSE2CoverageArtifact
    compiled_cell_sha256s: tuple[Sha256Digest, ...]
    evaluated_cells: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_order: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_candidates: tuple[UprightSE2ProposalCandidate, ...] = ()
    prune_decisions: tuple[UprightSE2ProofPruneDecision, ...]
    unresolved_frontier: tuple[UprightSE2ProofFrontierRow, ...]
    resource_ledger: UprightSE2ProofResourceLedger
    proof_material_sha256: Sha256Digest

    @property
    def total_resource_usage(self) -> ResourceUsage:
        """Expose the exact inner total callers must copy into their outer record."""

        return self.resource_ledger.canonical_total_resource_usage

    @model_validator(mode="after")
    def _validate_proof_material(self) -> Self:
        compilation = self.compilation
        if (
            self.solve_request_sha256 != compilation.solve_request_sha256
            or self.semantic_closure_sha256
            != compilation.semantic_closure.semantic_closure_sha256
            or self.upright_se2_compilation_sha256
            != compilation.upright_se2_compilation_sha256
        ):
            raise ValueError("proof material roots do not bind its compilation")
        source_input = _bound_source_input(
            compilation.source_solve_request.semantic_problem
        )
        if len(self.cardinal_tuple_roster) != 1:
            raise ValueError(
                "proof tuple roster must contain one request-authorized tuple"
            )
        proof_tuple = self.cardinal_tuple_roster[0]
        if (
            canonical_json_bytes(proof_tuple.authorization)
            != canonical_json_bytes(compilation.operation.authorization)
            or proof_tuple.reference_id != source_input.reference_id
            or proof_tuple.translation_domain
            != compilation.operation.translation_domain
            or tuple(canonical_json_bytes(cell) for cell in proof_tuple.compiled_cells)
            != tuple(canonical_json_bytes(cell) for cell in compilation.compiled_cells)
            or proof_tuple.authorization.operator_ref != source_input.operator_ref
            or proof_tuple.authorization.yaw.q != source_input.quarter_turns_ccw
        ):
            raise ValueError(
                "proof tuple roster does not bind the source authorization"
            )
        root_cells = proof_tuple.compiled_cells
        if any(
            (
                root.x_lower,
                root.x_upper,
                root.y_lower,
                root.y_upper,
            )
            != (
                proof_tuple.translation_domain.x_lower,
                proof_tuple.translation_domain.x_upper,
                proof_tuple.translation_domain.y_lower,
                proof_tuple.translation_domain.y_upper,
            )
            for root in root_cells
        ):
            raise ValueError(
                "proof tuple roots must bind the complete world-XY translation domain"
            )
        if (
            self.coverage_artifact.authorization_sha256
            != compilation.operation.authorization_sha256
        ):
            raise ValueError(
                "proof coverage does not bind the compilation authorization"
            )
        _proof_validate_leaf_partition(root_cells, self.coverage_artifact.cells)
        expected_cell_sha256s = tuple(
            cell.compiled_cell_sha256 for cell in self.coverage_artifact.cells
        )
        if self.compiled_cell_sha256s != expected_cell_sha256s:
            raise ValueError(
                "proof material must bind the complete canonical cell roster"
            )
        if self.evaluated_cells != tuple(
            sorted(
                self.evaluated_cells,
                key=lambda row: _lifted_cell_order_key(row.compiled_cell),
            )
        ):
            raise ValueError(
                "proof material evaluated cell roster must use deterministic order"
            )
        evaluated_cell_ids = tuple(
            row.compiled_cell.cell_id for row in self.evaluated_cells
        )
        if len(set(evaluated_cell_ids)) != len(evaluated_cell_ids):
            raise ValueError(
                "proof material evaluated cells must not duplicate a cell ID"
            )
        if len(
            {canonical_json_bytes(row.compiled_cell) for row in self.evaluated_cells}
        ) != len(self.evaluated_cells):
            raise ValueError("proof material evaluated cells must not duplicate a cell")
        for row in self.evaluated_cells:
            _proof_root_for_cell(row.compiled_cell, root_cells)
        final_leaf_rows = tuple(
            row for row in self.evaluated_cells if row.leaf_disposition is not None
        )
        if tuple(
            canonical_json_bytes(row.compiled_cell) for row in final_leaf_rows
        ) != tuple(canonical_json_bytes(cell) for cell in self.coverage_artifact.cells):
            raise ValueError(
                "proof evaluated cell roster has an incomplete, duplicated, or unordered final leaf roster"
            )

        expected_proposals = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        )
        if tuple(canonical_json_bytes(row) for row in self.proposal_order) != tuple(
            canonical_json_bytes(row) for row in expected_proposals
        ):
            raise ValueError("proof proposal order must be complete and deterministic")
        expected_proposal_cells = tuple(
            canonical_json_bytes(row) for row in expected_proposals
        )
        candidate_cells = tuple(
            canonical_json_bytes(candidate.final_inward_cell)
            for candidate in self.proposal_candidates
        )
        if len(set(candidate_cells)) != len(candidate_cells) or set(
            candidate_cells
        ) != set(expected_proposal_cells):
            raise ValueError("proof proposal candidates must cover final inward cells")
        if self.proposal_candidates != tuple(
            sorted(
                self.proposal_candidates,
                key=lambda candidate: candidate.canonical_order_key,
            )
        ):
            raise ValueError(
                "proof proposal candidates must use canonical witness order"
            )
        for candidate in self.proposal_candidates:
            point_evaluation = candidate.point_evaluation.point_cell_evaluation
            point_cell = point_evaluation.compiled_cell
            if not any(
                canonical_json_bytes(row) == canonical_json_bytes(point_evaluation)
                for row in self.evaluated_cells
            ):
                raise ValueError(
                    "proposal point evaluation must be present in evaluated cells"
                )
            point_root = _proof_root_for_cell(point_cell, root_cells)
            reuses_final_row = canonical_json_bytes(
                point_evaluation
            ) == canonical_json_bytes(candidate.final_inward_cell)
            if reuses_final_row:
                if (
                    canonical_json_bytes(point_cell) != canonical_json_bytes(point_root)
                    or point_cell.x_lower != point_cell.x_upper
                    or point_cell.y_lower != point_cell.y_upper
                ):
                    raise ValueError(
                        "proposal point may reuse only its byte-identical fully degenerate root final row"
                    )
            else:
                if point_evaluation.leaf_disposition is not None:
                    raise ValueError(
                        "nondegenerate proposal points require a distinct internal exact point row"
                    )
                final_cell = candidate.final_inward_cell.compiled_cell
                for lower, upper, coordinate in (
                    (
                        final_cell.x_lower.as_fraction,
                        final_cell.x_upper.as_fraction,
                        point_cell.x_lower.as_fraction,
                    ),
                    (
                        final_cell.y_lower.as_fraction,
                        final_cell.y_upper.as_fraction,
                        point_cell.y_lower.as_fraction,
                    ),
                ):
                    if (lower < upper and not lower < coordinate < upper) or (
                        lower == upper and coordinate != lower
                    ):
                        raise ValueError(
                            "nondegenerate proposal points require a strict-interior exact descendant"
                        )
            owning_leaves = tuple(
                row
                for row in final_leaf_rows
                if _proof_cell_is_within_root(point_cell, row.compiled_cell)
                and _proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.x_lower.as_fraction,
                    upper=row.compiled_cell.x_upper.as_fraction,
                    coordinate=point_cell.x_lower.as_fraction,
                    root_lower=point_root.x_lower.as_fraction,
                )
                and _proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.y_lower.as_fraction,
                    upper=row.compiled_cell.y_upper.as_fraction,
                    coordinate=point_cell.y_lower.as_fraction,
                    root_lower=point_root.y_lower.as_fraction,
                )
            )
            if len(owning_leaves) != 1 or canonical_json_bytes(
                owning_leaves[0]
            ) != canonical_json_bytes(candidate.final_inward_cell):
                raise ValueError(
                    "proposal point cell must be uniquely owned by its final inward leaf"
                )
            policy_terms = compilation.semantic_closure.objective_policy.terms
            if tuple(term.term_id for term in candidate.point_objective.terms) != tuple(
                term.term_id for term in policy_terms
            ):
                raise ValueError(
                    "proposal point objective terms do not bind the request policy"
                )
            expected_lower = sum(
                (
                    point_term.lower.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            expected_upper = sum(
                (
                    point_term.upper.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            if (
                candidate.point_objective.total_lower.as_fraction != expected_lower
                or candidate.point_objective.total_upper.as_fraction != expected_upper
            ):
                raise ValueError(
                    "proposal point objective total does not bind the request policy"
                )
            if (
                candidate.materialized_endpoint.upright_se2_compilation_sha256
                != compilation.upright_se2_compilation_sha256
                or candidate.program.semantic_problem_sha256
                != compilation.source_solve_request.semantic_problem_sha256
                or candidate.program.action_space_profile_sha256
                != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
                or candidate.program.before_state_sha256
                != compilation.source_solve_request.semantic_problem.scene_state.scene_state_sha256
                or candidate.program.grounded_obligation_set_sha256
                != compilation.grounded_obligations.grounded_obligation_set_sha256
                or candidate.program.state_delta_manifest
                != compilation.state_footprint.state_delta_manifest
            ):
                raise ValueError(
                    "proposal candidate program does not bind compilation roots"
                )
        expected_pruned = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED
        )
        if len(self.prune_decisions) != len(expected_pruned) or any(
            canonical_json_bytes(decision.cell_evaluation)
            != canonical_json_bytes(expected)
            for decision, expected in zip(
                self.prune_decisions, expected_pruned, strict=True
            )
        ):
            raise ValueError("proof prune decisions must be complete and deterministic")
        expected_unresolved = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED
        )
        if len(self.unresolved_frontier) != len(expected_unresolved) or any(
            canonical_json_bytes(row.cell_evaluation) != canonical_json_bytes(expected)
            for row, expected in zip(
                self.unresolved_frontier,
                expected_unresolved,
                strict=True,
            )
        ):
            raise ValueError(
                "proof unresolved frontier must be complete and deterministic"
            )
        if self.coverage_artifact.unresolved_cell_sha256s != tuple(
            sorted(
                row.compiled_cell.compiled_cell_sha256 for row in expected_unresolved
            )
        ):
            raise ValueError("proof coverage unresolved roster does not match frontier")

        expected_owner_evaluations = tuple(
            sorted(
                (
                    owner_evaluation
                    for row in self.evaluated_cells
                    for owner_evaluation in row.owner_evaluations
                ),
                key=canonical_json_bytes,
            )
        )
        staged_owner_evaluations = tuple(
            owner_evaluation
            for stage in self.resource_ledger.stage_deltas
            for owner_evaluation in stage.owner_evaluations
        )
        if tuple(sorted(staged_owner_evaluations, key=canonical_json_bytes)) != (
            expected_owner_evaluations
        ):
            raise ValueError(
                "proof shared ledger must account for every owner evaluation"
            )
        return self


def _proof_wire_record(
    fields: tuple[tuple[CanonicalId, TypedValue], ...],
) -> TypedValue:
    """Build one canonical record using the sole public proof payload schema."""

    return TypedValue(
        value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


def _proof_wire_encode_node(value: object) -> TypedValue:
    """Encode every canonical proof leaf structurally under one schema identity."""

    if isinstance(value, StrEnum):
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=CanonicalIdValue(value=value.value),
        )
    if value is None:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=EnumSymbolValue(symbol="NULL"),
        )
    if type(value) is bool:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=BooleanValue(value=value),
        )
    if type(value) is int:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=IntegerValue(value=value),
        )
    if type(value) is float:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteRealValue(value=value),
        )
    if type(value) is str:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=CanonicalIdValue(value=value),
        )
    if type(value) in (tuple, list):
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteOrderedTupleValue(
                element_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                items=tuple(_proof_wire_encode_node(item) for item in value),
            ),
        )
    if type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError(
                "proof payload records require canonical string field names"
            )
        return _proof_wire_record(
            tuple((name, _proof_wire_encode_node(item)) for name, item in value.items())
        )
    raise TypeError(f"proof payload cannot encode {type(value).__name__}")


def _proof_wire_decode_node(value: TypedValue) -> object:
    """Decode one structural proof node while rejecting other schema families."""

    if value.value_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
        raise ValueError("proof payload node has the wrong schema")
    payload = value.payload
    if type(payload) is CanonicalIdValue:
        return payload.value
    if type(payload) is BooleanValue:
        return payload.value
    if type(payload) is IntegerValue:
        return payload.value
    if type(payload) is FiniteRealValue:
        return payload.value
    if type(payload) is EnumSymbolValue:
        if payload.symbol != "NULL":
            raise ValueError("proof payload has an unknown scalar discriminator")
        return None
    if type(payload) is FiniteOrderedTupleValue:
        if payload.element_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
            raise ValueError("proof payload tuple has the wrong element schema")
        return tuple(_proof_wire_decode_node(item) for item in payload.items)
    if type(payload) is RecordValue:
        return {
            field.name: _proof_wire_decode_node(field.value) for field in payload.fields
        }
    raise ValueError("proof payload must be structural rather than a digest-only value")


def encode_upright_se2_proof_material(material: UprightSE2ProofMaterial) -> TypedValue:
    """Encode one sealed full cardinal proof into the sole M3 typed payload wire."""

    if type(material) is not UprightSE2ProofMaterial:
        raise TypeError("proof material codec requires UprightSE2ProofMaterial")
    checked = UprightSE2ProofMaterial.model_validate(
        material.model_dump(mode="python", round_trip=True),
        strict=True,
    )
    return _proof_wire_record(
        (
            (
                "definition_ref",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=CanonicalIdValue(
                        value=UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
                    ),
                ),
            ),
            (
                "discriminator",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=EnumSymbolValue(
                        symbol=UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR
                    ),
                ),
            ),
            (
                "proof_material",
                _proof_wire_encode_node(
                    checked.model_dump(mode="python", round_trip=True)
                ),
            ),
        )
    )


def decode_upright_se2_proof_material(value: TypedValue) -> UprightSE2ProofMaterial:
    """Decode the one full canonical proof payload and reject all substitutions."""

    if value.value_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
        raise ValueError("proof payload has the wrong schema")
    if type(value.payload) is not RecordValue:
        raise ValueError("proof payload must use its canonical record")
    fields = {field.name: field.value for field in value.payload.fields}
    if set(fields) != {"definition_ref", "discriminator", "proof_material"}:
        raise ValueError("proof payload record has missing or unknown fields")
    definition = fields["definition_ref"]
    if (
        type(definition.payload) is not CanonicalIdValue
        or definition.payload.value != UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
    ):
        raise ValueError("proof payload has the wrong definition identity")
    discriminator = fields["discriminator"]
    if (
        type(discriminator.payload) is not EnumSymbolValue
        or discriminator.payload.symbol != UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR
    ):
        raise ValueError("proof payload has an unknown discriminator")
    raw = _proof_wire_decode_node(fields["proof_material"])
    if type(raw) is not dict:
        raise ValueError("proof payload material must be a structural record")
    try:
        # The wire carries enum members as canonical scalar symbols; strict
        # reconstruction would incorrectly demand Python enum instances.  The
        # exact re-encode comparison below still rejects every coercive or
        # non-canonical representation.
        material = UprightSE2ProofMaterial.model_validate(raw, strict=False)
    except Exception as error:
        raise ValueError(
            "proof payload does not decode to sealed proof material"
        ) from error
    if canonical_json_bytes(
        encode_upright_se2_proof_material(material)
    ) != canonical_json_bytes(value):
        raise ValueError("proof payload is not the canonical complete representation")
    return material


def _continuous_proof_wire_record(
    fields: tuple[tuple[CanonicalId, TypedValue], ...],
) -> TypedValue:
    """Encode an additive continuous payload without touching cardinal codec."""

    # ``encode_upright_se2_continuous_proof_material`` has already rebuilt its
    # source material through the exact sealed model.  Constructing this
    # recursive, structural wire from those checked scalar/tuple/record values
    # must not repeatedly re-run Pydantic's recursive payload-union validation:
    # a real proposal embeds its compiler-materialized endpoint and otherwise
    # expands that same checked tree at every nested TypedValue boundary.
    #
    # These constructions retain the schema tags and canonical named-field
    # ordering explicitly.  They do not accept caller-provided wire nodes, and
    # the public decoder still reconstructs and re-encodes through the sealed
    # continuous proof model before accepting any payload.
    return TypedValue.model_construct(
        value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=RecordValue.model_construct(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue.model_construct(name=name, value=value)
                        for name, value in fields
                    ),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


def _continuous_proof_wire_encode_node(value: object) -> TypedValue:
    """Structurally encode only the additive continuous proof schema."""

    if isinstance(value, StrEnum):
        payload = CanonicalIdValue.model_construct(value=value.value)
    elif value is None:
        payload = EnumSymbolValue.model_construct(symbol="NULL")
    elif type(value) is bool:
        payload = BooleanValue.model_construct(value=value)
    elif type(value) is int:
        payload = IntegerValue.model_construct(value=value)
    elif type(value) is float:
        payload = FiniteRealValue.model_construct(value=value)
    elif type(value) is str:
        payload = CanonicalIdValue.model_construct(value=value)
    elif type(value) in (tuple, list):
        return TypedValue.model_construct(
            value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteOrderedTupleValue.model_construct(
                element_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                items=tuple(_continuous_proof_wire_encode_node(item) for item in value),
            ),
        )
    elif type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError("continuous proof payload records require string names")
        return _continuous_proof_wire_record(
            tuple(
                (name, _continuous_proof_wire_encode_node(item))
                for name, item in value.items()
            )
        )
    else:
        raise TypeError(
            f"continuous proof payload cannot encode {type(value).__name__}"
        )
    return TypedValue.model_construct(
        value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=payload,
    )


def _continuous_proof_wire_decode_node(value: TypedValue) -> object:
    if (
        value.value_schema_ref
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
    ):
        raise ValueError("continuous proof payload node has the wrong schema")
    payload = value.payload
    if type(payload) is CanonicalIdValue:
        return payload.value
    if type(payload) is BooleanValue:
        return payload.value
    if type(payload) is IntegerValue:
        return payload.value
    if type(payload) is FiniteRealValue:
        return payload.value
    if type(payload) is EnumSymbolValue:
        if payload.symbol != "NULL":
            raise ValueError("continuous proof payload has an unknown scalar")
        return None
    if type(payload) is FiniteOrderedTupleValue:
        if (
            payload.element_schema_ref
            != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
        ):
            raise ValueError("continuous proof payload tuple has wrong schema")
        return tuple(_continuous_proof_wire_decode_node(item) for item in payload.items)
    if type(payload) is RecordValue:
        return {
            field.name: _continuous_proof_wire_decode_node(field.value)
            for field in payload.fields
        }
    raise ValueError("continuous proof payload must be structural")


def encode_upright_se2_continuous_proof_material(
    material: UprightSE2ContinuousProofMaterial,
) -> TypedValue:
    """Encode the independent continuous proof payload wire."""

    if type(material) is not UprightSE2ContinuousProofMaterial:
        raise TypeError("continuous proof codec requires continuous proof material")
    checked = UprightSE2ContinuousProofMaterial.model_validate(
        material.model_dump(mode="python", round_trip=True), strict=True
    )
    return _continuous_proof_wire_record(
        (
            (
                "definition_ref",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=CanonicalIdValue(
                        value=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
                    ),
                ),
            ),
            (
                "discriminator",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=EnumSymbolValue(
                        symbol=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR
                    ),
                ),
            ),
            (
                "proof_material",
                _continuous_proof_wire_encode_node(
                    checked.model_dump(mode="python", round_trip=True)
                ),
            ),
        )
    )


def decode_upright_se2_continuous_proof_material(
    value: TypedValue,
) -> UprightSE2ContinuousProofMaterial:
    """Decode only the additive continuous proof discriminator/schema pair."""

    if (
        value.value_schema_ref
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
    ):
        raise ValueError("continuous proof payload has the wrong schema")
    if type(value.payload) is not RecordValue:
        raise ValueError("continuous proof payload must use a record")
    fields = {field.name: field.value for field in value.payload.fields}
    if set(fields) != {"definition_ref", "discriminator", "proof_material"}:
        raise ValueError("continuous proof payload has missing or unknown fields")
    if (
        type(fields["definition_ref"].payload) is not CanonicalIdValue
        or fields["definition_ref"].payload.value
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
    ):
        raise ValueError("continuous proof payload has wrong definition identity")
    if (
        type(fields["discriminator"].payload) is not EnumSymbolValue
        or fields["discriminator"].payload.symbol
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR
    ):
        raise ValueError("continuous proof payload has wrong discriminator")
    raw = _continuous_proof_wire_decode_node(fields["proof_material"])
    if type(raw) is not dict:
        raise ValueError("continuous proof material must be a structural record")
    try:
        material = UprightSE2ContinuousProofMaterial.model_validate(raw, strict=False)
    except Exception as error:
        raise ValueError(
            "continuous proof payload does not decode to sealed material"
        ) from error
    if canonical_json_bytes(
        encode_upright_se2_continuous_proof_material(material)
    ) != canonical_json_bytes(value):
        raise ValueError("continuous proof payload is not canonical")
    return material


class UprightSE2CheckerReplayPolicy(HashBoundCanonicalModel):
    """The explicit fresh-policy root consumed by the cardinal checker only."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/checker-replay-policy/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "checker_replay_policy_sha256"

    proof_policy_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    requested_gap: UprightSE2ExactRational
    objective_bound_policy_ref: DefinitionRef
    exact_global_claim_definition_ref: DefinitionRef
    finite_gap_claim_definition_ref: DefinitionRef
    checker_owner_ref: OwnerRef
    checker_capability_ref: CapabilityRef
    checker_build_sha256: Sha256Digest
    checker_replay_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_checker_identity(self) -> Self:
        if (
            self.checker_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF
            or self.checker_capability_ref
            != UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            or self.checker_build_sha256 != UPRIGHT_SE2_CHECKER_BUILD_SHA256
        ):
            raise ValueError("checker replay policy must bind the registered checker")
        return self


def build_upright_se2_checker_replay_policy(
    proof_policy: ProofPolicy,
    solve_policy_definition_bundle: DefinitionBundle,
) -> UprightSE2CheckerReplayPolicy:
    """Bind one fresh M1 proof policy to the static M3 checker identity."""

    if type(proof_policy) is not ProofPolicy:
        raise TypeError("checker replay policy requires an exact ProofPolicy")
    solve_policy = decode_upright_se2_solve_policy_definition_payload(
        solve_policy_definition_bundle
    )
    return UprightSE2CheckerReplayPolicy.seal(
        proof_policy_sha256=proof_policy.proof_policy_sha256,
        solve_policy_definition_bundle_sha256=(
            solve_policy_definition_bundle.definition_bundle_sha256
        ),
        requested_gap=solve_policy.requested_gap,
        objective_bound_policy_ref=solve_policy.objective_bound_policy_ref,
        exact_global_claim_definition_ref=(
            solve_policy.exact_global_claim_definition_ref
        ),
        finite_gap_claim_definition_ref=solve_policy.finite_gap_claim_definition_ref,
        checker_owner_ref=UPRIGHT_SE2_CHECKER_OWNER_REF,
        checker_capability_ref=UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        checker_build_sha256=UPRIGHT_SE2_CHECKER_BUILD_SHA256,
    )


class UprightSE2VerificationBundle(HashBoundCanonicalModel):
    """A closed profile/availability/pose/proof bundle for later checker replay."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/verification-bundle/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "verification_bundle_sha256"

    profile_registration_sha256: Sha256Digest
    backend_availability_sha256: Sha256Digest
    pose_yaw_bindings: tuple[ExplicitPoseYawBinding, ...]
    proof_material: UprightSE2ProofMaterial
    verification_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_pose_binding_identity(self) -> Self:
        _require_sorted_unique_by_bytes(self.pose_yaw_bindings, "pose/yaw bindings")
        entity_ids = tuple(binding.entity_id for binding in self.pose_yaw_bindings)
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(
                "verification bundle must reject same entity ID with different bytes"
            )
        return self


class UprightSE2CompilerClosure(HashBoundCanonicalModel):
    """The static profile/definition/owner closure consumed by the pure compiler."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/compiler-closure/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "compiler_closure_sha256"

    profile_registration_sha256: Sha256Digest
    definition_bundle_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    policy_bundle_sha256: Sha256Digest
    resource_policy_sha256: Sha256Digest
    compiler_owner_ref: OwnerRef
    compiler_build_sha256: Sha256Digest
    compiler_closure_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_compiler_owner(self) -> Self:
        if self.compiler_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("compiler owner reference must be fixed")
        if self.compiler_build_sha256 != UPRIGHT_SE2_COMPILER_BUILD_SHA256:
            raise ValueError("compiler build digest must be fixed")
        return self


class UprightSE2TranslationDomain(CanonicalModel):
    """The complete exact world-XY search domain for one cardinal tuple."""

    x_lower: ExactDyadic
    x_upper: ExactDyadic
    y_lower: ExactDyadic
    y_upper: ExactDyadic

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.x_lower.as_fraction > self.x_upper.as_fraction:
            raise ValueError("translation x lower bound must not exceed upper")
        if self.y_lower.as_fraction > self.y_upper.as_fraction:
            raise ValueError("translation y lower bound must not exceed upper")
        return self


class UprightSE2CardinalOperation(CanonicalModel):
    """One resolved cardinal *domain* transition before endpoint selection.

    The operation deliberately carries the complete authorized world-XY
    domain, rather than a selected translation.  A later, explicit endpoint
    materializer is the only place that may combine this operation with a
    concrete world-XY value.
    """

    authorization: CardinalYawAuthorization
    translation_domain: UprightSE2TranslationDomain
    inverse_quarter_turns_ccw: Annotated[StrictInt, Field(ge=0, le=3)]
    maximum_program_steps: Annotated[StrictInt, Field(gt=0)]
    maximum_edited_entities: Annotated[StrictInt, Field(gt=0)]

    @property
    def subject_id(self) -> CanonicalId:
        return self.authorization.subject_id

    @property
    def pivot_binding(self) -> FixedPivotBinding:
        return self.authorization.pivot_binding

    @property
    def quarter_turns_ccw(self) -> int:
        return self.authorization.yaw.q

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.authorization.cardinal_yaw_authorization_sha256

    @model_validator(mode="after")
    def _validate_inverse(self) -> Self:
        if self.inverse_quarter_turns_ccw != (-self.authorization.yaw.q) % 4:
            raise ValueError(
                "cardinal inverse must match the exact quarter-turn inverse"
            )
        if self.maximum_program_steps != 1:
            raise ValueError("upright se2 compilation permits exactly one program step")
        if self.maximum_edited_entities != 1:
            raise ValueError(
                "upright se2 compilation permits exactly one edited entity"
            )
        return self


class UprightSE2ContinuousOperation(CanonicalModel):
    """One resolved continuous domain transition before endpoint selection.

    This is intentionally a sibling of ``UprightSE2CardinalOperation``.  It
    does not widen that frozen cardinal record, so existing cardinal canonical
    bytes, discriminators, and hash domains remain unchanged.
    """

    authorization: ContinuousYawAuthorization
    translation_domain: UprightSE2TranslationDomain
    maximum_program_steps: Annotated[StrictInt, Field(gt=0)]
    maximum_edited_entities: Annotated[StrictInt, Field(gt=0)]

    @property
    def subject_id(self) -> CanonicalId:
        return self.authorization.subject_id

    @property
    def pivot_binding(self) -> FixedPivotBinding:
        return self.authorization.pivot_binding

    @property
    def yaw_domain(self) -> ContinuousYawDomain:
        return self.authorization.yaw_domain

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.authorization.continuous_yaw_authorization_sha256

    @model_validator(mode="after")
    def _validate_limits(self) -> Self:
        if self.maximum_program_steps != 1:
            raise ValueError("continuous upright se2 permits exactly one program step")
        if self.maximum_edited_entities != 1:
            raise ValueError("continuous upright se2 permits exactly one edited entity")
        return self


class UprightSE2StateFootprint(HashBoundCanonicalModel):
    """The complete primary/derived/frozen state partition for one transition."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/state-footprint/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "state_footprint_sha256"

    state_delta_manifest: StateDeltaManifest
    frozen_leaf_refs: tuple[StateVariableRef, ...]
    state_footprint_sha256: Sha256Digest

    @property
    def complete_before_leaf_index_sha256(self) -> Sha256Digest:
        return self.state_delta_manifest.complete_before_leaf_index_sha256

    @property
    def complete_after_leaf_index_sha256(self) -> Sha256Digest:
        return self.state_delta_manifest.complete_after_leaf_index_sha256

    @model_validator(mode="after")
    def _validate_frozen_partition(self) -> Self:
        _require_sorted_unique_by_bytes(self.frozen_leaf_refs, "frozen state leaves")
        primary = set(self.state_delta_manifest.authorized_primary_writes)
        derived = set(self.state_delta_manifest.recomputed_derived_writes)
        if primary & set(self.frozen_leaf_refs) or derived & set(self.frozen_leaf_refs):
            raise ValueError("frozen leaves must not overlap primary or derived writes")
        return self


_DerivedSourceFact: TypeAlias = (
    CollisionBodyFactV2 | SupportSurfaceFact | GeometryInstanceV2 | BaselineObservation
)


def _derived_source_identifier(source_fact: _DerivedSourceFact) -> CanonicalId:
    if isinstance(source_fact, CollisionBodyFactV2):
        return source_fact.body_id
    if isinstance(source_fact, SupportSurfaceFact):
        return source_fact.surface_id
    if isinstance(source_fact, GeometryInstanceV2):
        return source_fact.geometry_id
    return source_fact.observation_id


def _exact_source_values(
    facts: FactSetV2,
    label: str,
) -> tuple[CanonicalModel, ...]:
    """Return only the profile-authorized complete source-fact branch."""

    if (
        facts.availability is not FactAvailabilityV2.KNOWN
        or facts.completeness is not FactCompletenessV2.EXACT
        or facts.values is None
        or facts.inner_values is not None
        or facts.outer_values is not None
    ):
        raise ValueError(f"{label} must be a KNOWN EXACT fact set")
    return facts.values


def validate_required_upright_support_surface(
    scene: CanonicalScene,
    subject_id: CanonicalId,
) -> SupportSurfaceFact:
    """Resolve the subject's frozen support under the upright world-+Z rule."""

    subject = next(
        (
            object_
            for object_ in _exact_source_values(scene.objects, "objects")
            if object_.object_id == subject_id
        ),
        None,
    )
    if (
        subject is None
        or subject.support_assignment.availability is not FactAvailabilityV2.KNOWN
        or subject.support_assignment.surface_id is None
    ):
        raise ValueError(
            "subject support assignment must be KNOWN with a support surface"
        )
    sources = tuple(
        surface
        for surface in _exact_source_values(scene.support_surfaces, "support surfaces")
        if surface.surface_id == subject.support_assignment.surface_id
    )
    if len(sources) != 1:
        raise ValueError(
            "subject support assignment must bind one exact support surface"
        )

    surface = sources[0]
    normal = surface.normal_in_anchor
    if (normal.x, normal.y, normal.z) != (0.0, 0.0, 1.0):
        raise ValueError(
            "required support surface must have an exact horizontal world +Z normal"
        )
    surface_rotation = surface.anchor_from_surface.rotation
    if surface_rotation.x != 0.0 or surface_rotation.y != 0.0:
        raise ValueError(
            "required support surface frame must preserve exact horizontal world +Z"
        )
    if surface.owner_object_id is not None:
        owner = next(
            (
                object_
                for object_ in _exact_source_values(scene.objects, "objects")
                if object_.object_id == surface.owner_object_id
            ),
            None,
        )
        if owner is None:
            raise ValueError("required support surface owner must be a scene object")
        owner_rotation = owner.pose.world_from_object.rotation
        if owner_rotation.x != 0.0 or owner_rotation.y != 0.0:
            raise ValueError(
                "required support surface owner pose must preserve exact horizontal world +Z"
            )
    return surface


def _expected_derived_sources(
    scene: CanonicalScene,
    subject_id: CanonicalId,
    fact_kind: Literal["COLLISION", "SUPPORT", "RELATION", "VISIBILITY"],
) -> tuple[_DerivedSourceFact, ...]:
    if fact_kind == "COLLISION":
        return tuple(_exact_source_values(scene.collision_bodies, "collision bodies"))
    if fact_kind == "SUPPORT":
        return (validate_required_upright_support_surface(scene, subject_id),)
    if fact_kind == "RELATION":
        return tuple(
            geometry
            for geometry in _exact_source_values(
                scene.geometry_instances,
                "geometry instances",
            )
            if geometry.role is GeometryRoleV2.RELATION
        )
    return tuple(
        _exact_source_values(scene.baseline_observations, "baseline observations")
    )


class UprightSE2DerivedAfterFact(CanonicalModel):
    """One concrete frozen-source evaluation input at the canonical after pose."""

    fact_kind: Literal["COLLISION", "SUPPORT", "RELATION", "VISIBILITY"]
    source_fact_id: CanonicalId
    source_fact_sha256: Sha256Digest
    source_fact: _DerivedSourceFact
    after_subject_pose: RigidTransformV2

    @model_validator(mode="after")
    def _validate_concrete_source_fact(self) -> Self:
        source_type = {
            "COLLISION": CollisionBodyFactV2,
            "SUPPORT": SupportSurfaceFact,
            "RELATION": GeometryInstanceV2,
            "VISIBILITY": BaselineObservation,
        }[self.fact_kind]
        if not isinstance(self.source_fact, source_type):
            raise TypeError("derived fact kind does not match its concrete source fact")
        if self.fact_kind == "RELATION" and (
            self.source_fact.role is not GeometryRoleV2.RELATION
        ):
            raise ValueError("relation derived fact must use relation-role geometry")
        if _derived_source_identifier(self.source_fact) != self.source_fact_id:
            raise ValueError(
                "derived fact source ID does not match its concrete source fact"
            )
        if (
            canonical_sha256(
                self.source_fact,
                domain=UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN,
            )
            != self.source_fact_sha256
        ):
            raise ValueError(
                "derived fact source digest does not bind its concrete source fact"
            )
        return self


class UprightSE2AfterStateTemplate(HashBoundCanonicalModel):
    """A complete endpoint pose template without solver or checker conclusions."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/after-state-template/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "after_state_template_sha256"

    subject_id: CanonicalId
    subject_before_pose: RigidTransformV2
    evaluation_scene: CanonicalScene
    subject_pivot_xy_m: Vec2
    subject_pivot_z_m: FiniteFloat
    subject_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    reference_pivot_xy_m: Vec2
    collision_facts: tuple[UprightSE2DerivedAfterFact, ...]
    support_facts: tuple[UprightSE2DerivedAfterFact, ...]
    relation_facts: tuple[UprightSE2DerivedAfterFact, ...]
    visibility_facts: tuple[UprightSE2DerivedAfterFact, ...]
    after_state_template_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_derived_views(self) -> Self:
        subject = next(
            (
                object_
                for object_ in self.evaluation_scene.objects.values or ()
                if object_.object_id == self.subject_id
            ),
            None,
        )
        if subject is None:
            raise ValueError("after-state evaluation scene must contain the subject")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError(
                "after-state subject before pose must bind the evaluation scene"
            )
        translation = self.subject_pose.translation
        if (translation.x, translation.y, translation.z) != (
            self.subject_pivot_xy_m.x,
            self.subject_pivot_xy_m.y,
            self.subject_pivot_z_m,
        ):
            raise ValueError("canonical subject pose must match the after pivot")
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns,
            self.subject_pose.rotation,
        )
        _exact_source_values(self.evaluation_scene.cameras, "cameras")
        for label, facts in (
            ("collision facts", self.collision_facts),
            ("support facts", self.support_facts),
            ("relation facts", self.relation_facts),
            ("visibility facts", self.visibility_facts),
        ):
            _require_sorted_unique_by_bytes(facts, label)
            if any(fact.after_subject_pose != self.subject_pose for fact in facts):
                raise ValueError(
                    "derived fact after poses must match the canonical subject pose"
                )
        for fact_kind, facts in (
            ("COLLISION", self.collision_facts),
            ("SUPPORT", self.support_facts),
            ("RELATION", self.relation_facts),
            ("VISIBILITY", self.visibility_facts),
        ):
            if any(fact.fact_kind != fact_kind for fact in facts):
                raise ValueError("after-state derived fact family is inconsistent")
            expected_sources = _expected_derived_sources(
                self.evaluation_scene,
                self.subject_id,
                fact_kind,
            )
            expected_by_id = {
                _derived_source_identifier(source): source
                for source in expected_sources
            }
            actual_by_id = {fact.source_fact_id: fact.source_fact for fact in facts}
            if len(actual_by_id) != len(facts) or set(actual_by_id) != set(
                expected_by_id
            ):
                raise ValueError(
                    "after-state derived facts must cover the complete affected set"
                )
            if any(
                canonical_json_bytes(actual_by_id[source_id])
                != canonical_json_bytes(source)
                for source_id, source in expected_by_id.items()
            ):
                raise ValueError(
                    "after-state derived facts must bind evaluation-scene sources"
                )
        return self


class UprightSE2EndpointConstructionRecipe(HashBoundCanonicalModel):
    """The complete source-bound recipe for a later selected endpoint.

    This is intentionally not an endpoint and does not carry a translation.
    It records every source value needed by the pure materializer so a
    compilation remains a statement about an authorized domain only.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/endpoint-construction-recipe/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "endpoint_construction_recipe_sha256"

    source_scene_state_sha256: Sha256Digest
    operation_authorization_sha256: Sha256Digest
    subject_id: CanonicalId
    reference_id: CanonicalId
    pivot_binding: FixedPivotBinding
    quarter_turns_ccw: Annotated[StrictInt, Field(ge=0, le=3)]
    translation_domain: UprightSE2TranslationDomain
    subject_before_pose: RigidTransformV2
    reference_before_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    evaluation_scene: CanonicalScene
    endpoint_construction_recipe_sha256: Sha256Digest

    @property
    def authorization_sha256(self) -> Sha256Digest:
        """Expose the bound operation authorization without duplicating it."""

        return self.operation_authorization_sha256

    @property
    def translation_domain_sha256(self) -> Sha256Digest:
        """Return the explicit canonical root of the authorized XY domain."""

        return canonical_sha256(
            self.translation_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        )

    @model_validator(mode="after")
    def _validate_source_bound_recipe(self) -> Self:
        subject = _bound_scene_object(self.evaluation_scene, self.subject_id)
        reference = _bound_scene_object(self.evaluation_scene, self.reference_id)
        if self.subject_id == self.reference_id:
            raise ValueError("endpoint recipe subject and reference must differ")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError(
                "endpoint recipe subject pose must bind its evaluation scene"
            )
        if reference.pose.world_from_object != self.reference_before_pose:
            raise ValueError(
                "endpoint recipe reference pose must bind its evaluation scene"
            )
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns,
            self.subject_before_pose.rotation,
        )
        expected_pivot_id = (
            self.subject_id
            if self.pivot_binding.pivot_mode is PivotMode.OWN
            else self.reference_id
        )
        if self.pivot_binding.pivot_entity_id != expected_pivot_id:
            raise ValueError(
                "endpoint recipe pivot must bind its selected scene object"
            )
        expected_state_sha256 = canonical_sha256(
            {
                "scene_state_sha256": self.source_scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": (
                    subject.pose
                    if self.pivot_binding.pivot_mode is PivotMode.OWN
                    else reference.pose
                ),
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        )
        if self.pivot_binding.pivot_state_sha256 != expected_state_sha256:
            raise ValueError(
                "endpoint recipe pivot digest must bind the source scene state"
            )
        return self


class UprightSE2ContinuousEndpointConstructionRecipe(HashBoundCanonicalModel):
    """Source-bound continuous endpoint recipe, separate from cardinal bytes."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-endpoint-construction-recipe/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_endpoint_construction_recipe_sha256"

    source_scene_state_sha256: Sha256Digest
    operation_authorization_sha256: Sha256Digest
    subject_id: CanonicalId
    reference_id: CanonicalId
    pivot_binding: FixedPivotBinding
    yaw_domain: ContinuousYawDomain
    translation_domain: UprightSE2TranslationDomain
    subject_before_pose: RigidTransformV2
    reference_before_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    evaluation_scene: CanonicalScene
    continuous_endpoint_construction_recipe_sha256: Sha256Digest

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.operation_authorization_sha256

    @property
    def translation_domain_sha256(self) -> Sha256Digest:
        return canonical_sha256(
            self.translation_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        )

    @model_validator(mode="after")
    def _validate_source_bound_recipe(self) -> Self:
        subject = _bound_scene_object(self.evaluation_scene, self.subject_id)
        reference = _bound_scene_object(self.evaluation_scene, self.reference_id)
        if self.subject_id == self.reference_id:
            raise ValueError("continuous endpoint subject and reference must differ")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError("continuous endpoint subject pose must bind its scene")
        if reference.pose.world_from_object != self.reference_before_pose:
            raise ValueError("continuous endpoint reference pose must bind its scene")
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns, self.subject_before_pose.rotation
        )
        expected_pivot_id = (
            self.subject_id
            if self.pivot_binding.pivot_mode is PivotMode.OWN
            else self.reference_id
        )
        if self.pivot_binding.pivot_entity_id != expected_pivot_id:
            raise ValueError("continuous endpoint pivot must bind its scene object")
        expected_state_sha256 = canonical_sha256(
            {
                "scene_state_sha256": self.source_scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": (
                    subject.pose
                    if self.pivot_binding.pivot_mode is PivotMode.OWN
                    else reference.pose
                ),
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        )
        if self.pivot_binding.pivot_state_sha256 != expected_state_sha256:
            raise ValueError("continuous endpoint pivot digest does not bind source")
        return self


def _continuous_canonical_yaw_after(
    yaw_before: CanonicalSO2Angle,
    selected_lifted_yaw: ExactDyadic,
) -> CanonicalSO2Angle:
    """Apply an authorized lifted delta through the directed-yaw owner."""

    turns = Fraction.from_float(yaw_before.turns) + selected_lifted_yaw.as_fraction
    while turns < Fraction(-1, 2):
        turns += 1
    while turns >= Fraction(1, 2):
        turns -= 1
    return CanonicalSO2Angle(turns=0.0 if turns == 0 else float(turns))


def _continuous_materialized_expected_pose(
    recipe: UprightSE2ContinuousEndpointConstructionRecipe,
    translation_xy_m: Vec2,
    selected_lifted_yaw: ExactDyadic,
) -> tuple[Vec2, CanonicalSO2Angle, RigidTransformV2, Vec2]:
    """Replay one continuous endpoint pose from its source-bound recipe."""

    selected = selected_lifted_yaw.as_fraction
    yaw_after = _continuous_canonical_yaw_after(
        recipe.subject_yaw_turns,
        selected_lifted_yaw,
    )
    radians = 2.0 * math.pi * float(selected)
    cos_yaw = math.cos(radians)
    sin_yaw = math.sin(radians)
    subject_pose = recipe.subject_before_pose
    reference_pose = recipe.reference_before_pose
    pivot_pose = (
        subject_pose
        if recipe.pivot_binding.pivot_mode is PivotMode.OWN
        else reference_pose
    )
    rel_x = subject_pose.translation.x - pivot_pose.translation.x
    rel_y = subject_pose.translation.y - pivot_pose.translation.y
    subject_after = Vec2(
        x=(
            0.0
            if pivot_pose.translation.x
            + cos_yaw * rel_x
            - sin_yaw * rel_y
            + translation_xy_m.x
            == 0.0
            else pivot_pose.translation.x
            + cos_yaw * rel_x
            - sin_yaw * rel_y
            + translation_xy_m.x
        ),
        y=(
            0.0
            if pivot_pose.translation.y
            + sin_yaw * rel_x
            + cos_yaw * rel_y
            + translation_xy_m.y
            == 0.0
            else pivot_pose.translation.y
            + sin_yaw * rel_x
            + cos_yaw * rel_y
            + translation_xy_m.y
        ),
    )
    half_radians = math.pi * float(selected)
    delta_z = math.sin(half_radians)
    delta_w = math.cos(half_radians)
    after_rotation = _compose_upright_quaternion_from_primary_yaw(
        rotation=subject_pose.rotation,
        delta_z=delta_z,
        delta_w=delta_w,
        primary_expected_yaw=yaw_after,
    )
    return (
        subject_after,
        yaw_after,
        RigidTransformV2(
            translation=Vec3(
                x=subject_after.x,
                y=subject_after.y,
                z=(
                    0.0
                    if subject_pose.translation.z == 0.0
                    else subject_pose.translation.z
                ),
            ),
            rotation=after_rotation,
        ),
        Vec2(
            x=reference_pose.translation.x,
            y=reference_pose.translation.y,
        ),
    )


class UprightSE2MaterializedEndpoint(HashBoundCanonicalModel):
    """One explicit selected endpoint, separate from domain-level compilation."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/materialized-endpoint/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "materialized_endpoint_sha256"

    upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2Compilation
    endpoint_construction_recipe: UprightSE2EndpointConstructionRecipe
    translation_xy_m: Vec2
    after_state: UprightSE2AfterStateTemplate
    program: EditProgram
    materialized_endpoint_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_endpoint_from_recipe(self) -> Self:
        compilation = self.compilation
        recipe = self.endpoint_construction_recipe
        if (
            self.upright_se2_compilation_sha256
            != compilation.upright_se2_compilation_sha256
            or recipe != compilation.endpoint_construction_recipe
        ):
            raise ValueError(
                "materialized endpoint must bind its exact compilation recipe"
            )
        x = _fraction_from_float(self.translation_xy_m.x)
        y = _fraction_from_float(self.translation_xy_m.y)
        if (
            not recipe.translation_domain.x_lower.as_fraction
            <= x
            <= recipe.translation_domain.x_upper.as_fraction
        ):
            raise ValueError(
                "materialized endpoint translation x is outside its domain"
            )
        if (
            not recipe.translation_domain.y_lower.as_fraction
            <= y
            <= recipe.translation_domain.y_upper.as_fraction
        ):
            raise ValueError(
                "materialized endpoint translation y is outside its domain"
            )
        pivot_pose = (
            recipe.subject_before_pose
            if recipe.pivot_binding.pivot_mode is PivotMode.OWN
            else recipe.reference_before_pose
        )
        rotated_x, rotated_y = _bound_rotate_cardinal_xy(
            recipe.subject_before_pose.translation.x - pivot_pose.translation.x,
            recipe.subject_before_pose.translation.y - pivot_pose.translation.y,
            recipe.quarter_turns_ccw,
        )
        expected_xy = Vec2(
            x=0.0
            if pivot_pose.translation.x + rotated_x + self.translation_xy_m.x == 0.0
            else pivot_pose.translation.x + rotated_x + self.translation_xy_m.x,
            y=0.0
            if pivot_pose.translation.y + rotated_y + self.translation_xy_m.y == 0.0
            else pivot_pose.translation.y + rotated_y + self.translation_xy_m.y,
        )
        expected_yaw = _bound_cardinal_after_yaw(
            recipe.subject_yaw_turns,
            recipe.quarter_turns_ccw,
        )
        expected_pose = RigidTransformV2(
            translation=Vec3(
                x=expected_xy.x,
                y=expected_xy.y,
                z=(
                    0.0
                    if recipe.subject_before_pose.translation.z == 0.0
                    else recipe.subject_before_pose.translation.z
                ),
            ),
            rotation=_bound_cardinal_quaternion(
                recipe.subject_before_pose.rotation,
                recipe.quarter_turns_ccw,
                primary_expected_yaw=expected_yaw,
            ),
        )
        if (
            self.after_state.evaluation_scene != recipe.evaluation_scene
            or self.after_state.subject_id != recipe.subject_id
            or self.after_state.subject_before_pose != recipe.subject_before_pose
            or self.after_state.reference_pivot_xy_m
            != Vec2(
                x=recipe.reference_before_pose.translation.x,
                y=recipe.reference_before_pose.translation.y,
            )
            or self.after_state.subject_pivot_xy_m != expected_xy
            or self.after_state.subject_yaw_turns != expected_yaw
            or self.after_state.subject_pose != expected_pose
        ):
            raise ValueError(
                "materialized endpoint after state must derive from its recipe"
            )
        _validate_materialized_endpoint_program(self)
        return self

    @property
    def program_sha256(self) -> Sha256Digest:
        return self.program.program_sha256

    @property
    def after_scene_state_sha256(self) -> Sha256Digest:
        return self.program.after_scene_state_sha256


def _materialized_program_arguments(
    compilation: UprightSE2Compilation, translation_xy_m: Vec2
) -> tuple[OperationArgument, ...]:
    """Derive the sole generic invocation payload from sealed endpoint inputs."""

    return tuple(
        sorted(
            (
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/pivot-entity-id",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/entity-id/1.0",
                        payload=CanonicalIdValue(
                            value=compilation.operation.pivot_binding.pivot_entity_id
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/quarter-turns-ccw",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/cardinal-yaw/1.0",
                        payload=IntegerValue(
                            value=compilation.operation.quarter_turns_ccw
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/subject-id",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/entity-id/1.0",
                        payload=CanonicalIdValue(
                            value=compilation.endpoint_construction_recipe.subject_id
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/translation-x-m",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/metre/1.0",
                        payload=FiniteRealValue(value=translation_xy_m.x),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/translation-y-m",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/metre/1.0",
                        payload=FiniteRealValue(value=translation_xy_m.y),
                    ),
                ),
            ),
            key=lambda argument: canonical_json_bytes(argument.argument_name),
        )
    )


def _materialized_derived_bundle(
    after_state: UprightSE2AfterStateTemplate,
) -> ExtensionFactBundle:
    """Carry each authorized primary/recomputed state value structurally."""

    role_values = {
        "subject-world-x": _materialized_state_real(
            after_state.subject_pose.translation.x
        ),
        "subject-world-y": _materialized_state_real(
            after_state.subject_pose.translation.y
        ),
        "subject-explicit-yaw": _materialized_state_real(
            after_state.subject_yaw_turns.turns
        ),
        "subject-derived-canonical-pose": _materialized_pose_value(
            after_state.subject_pose
        ),
        "subject-derived-collision": _materialized_derived_fact_roster_value(
            after_state.collision_facts
        ),
        "subject-derived-support": _materialized_derived_fact_roster_value(
            after_state.support_facts
        ),
        "subject-derived-relation": _materialized_derived_fact_roster_value(
            after_state.relation_facts
        ),
        "subject-derived-visibility": _materialized_derived_fact_roster_value(
            after_state.visibility_facts
        ),
    }
    expected_roles = _UPRIGHT_SE2_PRIMARY_ROLES + _UPRIGHT_SE2_DERIVED_ROLES
    if tuple(role_values) != expected_roles:
        raise AssertionError("materialized role value roster drift")

    return ExtensionFactBundle.seal(
        facts=tuple(
            sorted(
                (
                    ExtensionFact(
                        fact_family_ref=_UPRIGHT_SE2_STATE_FAMILY_REF,
                        subject_entity_id=after_state.subject_id,
                        fact_key=f"fact-key:spatialcf/upright-se2/{role}",
                        value=role_values[role],
                    )
                    for role in expected_roles
                ),
                key=canonical_json_bytes,
            )
        )
    )


def _materialized_state_real(value: float) -> TypedValue:
    return TypedValue(
        value_schema_ref=_UPRIGHT_SE2_REAL_SCHEMA_REF,
        payload=FiniteRealValue(value=value),
    )


def _materialized_state_record(
    schema_ref: str, fields: tuple[tuple[str, TypedValue], ...]
) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema_ref,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


def _materialized_pose_value(pose: RigidTransformV2) -> TypedValue:
    return _materialized_state_record(
        "schema:spatialcf/upright-se2/subject-derived-canonical-pose/1.0",
        (
            ("translation_x_m", _materialized_state_real(pose.translation.x)),
            ("translation_y_m", _materialized_state_real(pose.translation.y)),
            ("translation_z_m", _materialized_state_real(pose.translation.z)),
            ("rotation_x", _materialized_state_real(pose.rotation.x)),
            ("rotation_y", _materialized_state_real(pose.rotation.y)),
            ("rotation_z", _materialized_state_real(pose.rotation.z)),
            ("rotation_w", _materialized_state_real(pose.rotation.w)),
        ),
    )


def _materialized_derived_fact_value(fact: UprightSE2DerivedAfterFact) -> TypedValue:
    return _materialized_state_record(
        "schema:spatialcf/upright-se2/derived-after-fact/1.0",
        (
            (
                "fact_kind",
                TypedValue(
                    value_schema_ref="schema:spatialcf/upright-se2/enum-symbol/1.0",
                    payload=EnumSymbolValue(symbol=fact.fact_kind),
                ),
            ),
            (
                "source_fact_id",
                TypedValue(
                    value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                    payload=CanonicalIdValue(value=fact.source_fact_id),
                ),
            ),
            (
                "source_fact_sha256",
                TypedValue(
                    value_schema_ref="schema:spatialcf/upright-se2/digest/1.0",
                    payload=DigestValue(value=fact.source_fact_sha256),
                ),
            ),
            ("after_subject_pose", _materialized_pose_value(fact.after_subject_pose)),
        ),
    )


def _materialized_derived_fact_roster_value(
    facts: tuple[UprightSE2DerivedAfterFact, ...],
) -> TypedValue:
    return TypedValue(
        value_schema_ref="schema:spatialcf/upright-se2/derived-after-fact-roster/1.0",
        payload=FiniteOrderedTupleValue(
            element_schema_ref="schema:spatialcf/upright-se2/derived-after-fact/1.0",
            items=tuple(_materialized_derived_fact_value(fact) for fact in facts),
        ),
    )


def _materialized_expected_after_scene_state(
    compilation: UprightSE2Compilation | UprightSE2ContinuousCompilation,
    after_state: UprightSE2AfterStateTemplate,
) -> SceneStateEnvelope:
    """Reconstruct the one complete source-preserving after scene exactly."""

    before_state = compilation.source_solve_request.semantic_problem.scene_state
    source_scene = before_state.base_scene_payload
    objects = source_scene.objects.values
    if objects is None:
        raise ValueError("materialized program source scene must contain exact objects")
    replaced = 0
    after_objects = []
    for object_ in objects:
        if object_.object_id == after_state.subject_id:
            replaced += 1
            after_objects.append(
                object_.model_copy(
                    update={
                        "pose": ObjectPose(world_from_object=after_state.subject_pose)
                    }
                )
            )
        else:
            after_objects.append(object_)
    if replaced != 1:
        raise ValueError("materialized program must replace exactly one subject pose")
    after_scene = source_scene.model_copy(
        update={
            "objects": source_scene.objects.model_copy(
                update={"values": tuple(after_objects)}
            )
        }
    )
    return SceneStateEnvelope.seal(
        base_scene_schema_ref=before_state.base_scene_schema_ref,
        base_scene_payload=after_scene,
        extension_fact_bundles=tuple(
            sorted(
                (
                    *before_state.extension_fact_bundles,
                    _materialized_derived_bundle(after_state),
                ),
                key=canonical_json_bytes,
            )
        ),
        closed_entity_index=before_state.closed_entity_index,
        canonical_state_leaf_index=before_state.canonical_state_leaf_index,
    )


def _validate_materialized_endpoint_program(
    endpoint: UprightSE2MaterializedEndpoint,
) -> None:
    """Close retained program fields over one sealed compilation and endpoint."""

    compilation = endpoint.compilation
    source_problem = compilation.source_solve_request.semantic_problem
    before_state = source_problem.scene_state
    program = endpoint.program
    if (
        program.semantic_problem_sha256 != source_problem.semantic_problem_sha256
        or program.action_space_profile_sha256
        != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
        or program.before_state_sha256 != before_state.scene_state_sha256
        or program.grounded_obligation_set_sha256
        != compilation.grounded_obligations.grounded_obligation_set_sha256
        or program.state_delta_manifest
        != compilation.state_footprint.state_delta_manifest
    ):
        raise ValueError("materialized endpoint program roots do not bind compilation")
    expected_step = OperationInvocation(
        operator_ref=compilation.operation.authorization.operator_ref,
        arguments=_materialized_program_arguments(
            compilation, endpoint.translation_xy_m
        ),
    )
    if program.steps != (expected_step,):
        raise ValueError(
            "materialized endpoint program must contain exactly one bound invocation"
        )
    expected_after = _materialized_expected_after_scene_state(
        compilation, endpoint.after_state
    )
    if (
        program.after_scene_state_sha256 != expected_after.scene_state_sha256
        or canonical_json_bytes(program.after_scene_state)
        != canonical_json_bytes(expected_after)
    ):
        raise ValueError(
            "materialized endpoint program must equal the complete expected after scene"
        )


class UprightSE2ProposalCandidate(HashBoundCanonicalModel):
    """One compiler-materialized inward witness, ordered independently of cells."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proposal-candidate/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proposal_candidate_sha256"

    final_inward_cell: UprightSE2ProofCellEvaluation
    selected_translation_xy_m: Vec2
    point_evaluation: UprightSE2ProposalPointEvaluation
    point_objective: UprightSE2ProposalPointObjective
    materialized_endpoint: UprightSE2MaterializedEndpoint
    program: EditProgram
    proposal_candidate_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_candidate(self) -> Self:
        if (
            self.final_inward_cell.leaf_disposition
            is not UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        ):
            raise ValueError(
                "proposal candidates require one final inward-feasible cell"
            )
        if canonical_json_bytes(self.point_objective) != canonical_json_bytes(
            self.point_evaluation.point_objective
        ):
            raise ValueError(
                "proposal candidate point objective must match retained point evaluation"
            )
        if (
            self.selected_translation_xy_m
            != self.materialized_endpoint.translation_xy_m
        ):
            raise ValueError(
                "proposal candidate point must match its materialized endpoint"
            )
        cell = self.final_inward_cell.compiled_cell
        point_x = _fraction_from_float(self.selected_translation_xy_m.x)
        point_y = _fraction_from_float(self.selected_translation_xy_m.y)
        if not (
            cell.x_lower.as_fraction <= point_x <= cell.x_upper.as_fraction
            and cell.y_lower.as_fraction <= point_y <= cell.y_upper.as_fraction
        ):
            raise ValueError("proposal candidate point is outside its final cell")
        point_cell = self.point_evaluation.point_cell_evaluation.compiled_cell
        if (
            point_cell.x_lower.as_fraction != point_x
            or point_cell.x_upper.as_fraction != point_x
            or point_cell.y_lower.as_fraction != point_y
            or point_cell.y_upper.as_fraction != point_y
        ):
            raise ValueError(
                "proposal candidate point must match its retained point cell"
            )
        if canonical_json_bytes(self.program) != canonical_json_bytes(
            self.materialized_endpoint.program
        ):
            raise ValueError(
                "proposal candidate program must match its materialized endpoint"
            )
        return self

    @property
    def canonical_order_key(self) -> tuple[Fraction, Fraction, bytes, bytes]:
        """Return the frozen witness key, deliberately excluding traversal cell IDs."""

        return (
            self.point_objective.total_upper.as_fraction,
            self.point_objective.total_lower.as_fraction,
            canonical_json_bytes(self.point_objective.terms),
            canonical_json_bytes(self.program),
        )


_M2_Q0_SHARED_VALUE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/shared-value/3.0"
)
_M2_Q0_MAPPING_DEFINITION_ROSTER_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/mapping-definition-roster/3.0"
)
_M2_Q0_SOURCE_PROVENANCE_HASH_DOMAIN = (
    "spatialcf/counterfactual/upright-se2/m2-q0/source-provenance/3.0"
)
_M2_Q0_SUPPORTED_DOMAIN_REF = (
    "definition:spatialcf/upright-se2/m2-closed-axis-aligned-rect-domain/1.0"
)
_M2_Q0_DOMAIN_SCHEMA_REF = "schema:spatialcf/upright-se2/m2-q0/translation-domain/1.0"
_M2_Q0_DOMAIN_TRANSFORM_REF = (
    "definition:spatialcf/upright-se2/m2-q0/delta-xy-identity/1.0"
)


def _m2_q0_dyadic_from_float(value: float) -> ExactDyadic:
    fraction = Fraction.from_float(0.0 if value == 0.0 else value)
    return ExactDyadic(numerator=fraction.numerator, denominator=fraction.denominator)


def _m2_q0_authorized_domain_from_source(
    source_compilation: PlanarTranslateCompilation,
) -> UprightSE2TranslationDomain:
    """Reconstruct the sole supported M2 world-XY delta domain from provenance."""

    source_problem = source_compilation.source_artifacts.problem
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
            "q=0 construction source has no supported total world-XY representation"
        )
    workspace = tuple(
        item
        for item in source_problem.scene.workspace_boundaries.values or ()
        if item.fact_id == position.workspace_fact_ids[0]
    )
    if len(workspace) != 1:
        raise ValueError("q=0 construction source must bind one exact workspace fact")
    region = workspace[0].region_world_xy
    if len(region.components) != 1 or region.components[0].holes:
        raise ValueError(
            "q=0 construction source must be one closed axis-aligned rectangle"
        )
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
        raise ValueError(
            "q=0 construction source must be one closed axis-aligned rectangle"
        )
    subject = tuple(
        item
        for item in source_problem.scene.objects.values or ()
        if item.object_id == constraints.allowed_edit.subject_id
    )
    if len(subject) != 1:
        raise ValueError("q=0 construction source must bind one source subject pose")
    before = subject[0].pose.world_from_object.translation
    return UprightSE2TranslationDomain(
        x_lower=_m2_q0_dyadic_from_float(xs[0] - before.x),
        x_upper=_m2_q0_dyadic_from_float(xs[1] - before.x),
        y_lower=_m2_q0_dyadic_from_float(ys[0] - before.y),
        y_upper=_m2_q0_dyadic_from_float(ys[1] - before.y),
    )


def _m2_q0_domain_typed_value(domain: UprightSE2TranslationDomain) -> TypedValue:
    return TypedValue(
        value_schema_ref=(
            "schema:spatialcf/upright-se2/executable-m2-q0-domain-policy/1.0"
        ),
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=name,
                            value=TypedValue(
                                value_schema_ref=_UPRIGHT_SE2_REAL_SCHEMA_REF,
                                payload=FiniteRealValue(value=float(value.as_fraction)),
                            ),
                        )
                        for name, value in (
                            ("x_lower", domain.x_lower),
                            ("x_upper", domain.x_upper),
                            ("y_lower", domain.y_lower),
                            ("y_upper", domain.y_upper),
                        )
                    ),
                    key=canonical_json_bytes,
                )
            )
        ),
    )


def _m2_q0_non_equivalence_values(
    source_compilation: PlanarTranslateCompilation,
    policy_bundle: UprightSE2ExecutablePolicyBundle,
) -> dict[str, tuple[object, object]]:
    """Return the exact M2 and construction-bound M3 roots for each mismatch."""

    source_problem = source_compilation.source_artifacts.problem
    return {
        "source:policy/objective": (
            source_problem.objective,
            policy_bundle.policy_for("objective").payload,
        ),
        "source:policy/relation": (
            source_problem.relation_semantics,
            tuple(
                policy.payload
                for policy in policy_bundle.policies
                if policy.policy_key.startswith("relation:")
            ),
        ),
        "source:policy/visibility": (
            source_problem.visibility_semantics,
            policy_bundle.policy_for("visibility").payload,
        ),
        "source:policy/support": (
            source_problem.constraints.support_constraints,
            policy_bundle.policy_for("support").payload,
        ),
        "source:policy/numeric": (
            source_problem.numeric_policy,
            policy_bundle.policy_for("numeric").payload,
        ),
        "source:policy/resource": (
            source_compilation.source_artifacts.config,
            policy_bundle.policy_for("resource").payload,
        ),
        "source:policy/tie-break": (
            source_problem.objective.tie_break,
            policy_bundle.policy_for("objective").payload,
        ),
    }


class UprightSE2M2Q0MappingDefinition(HashBoundCanonicalModel):
    """One versioned source-to-M3 construction relation definition.

    This record deliberately describes a relation, never an evaluator.  It is
    carried as compiler-owned evidence for the narrow q=0 construction route
    and cannot alter the Task 1 profile registration or executable M3 policy.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/mapping-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "mapping_definition_sha256"

    mapping_definition_ref: DefinitionRef
    row_kind: Literal["EQUALITY", "SOURCE_CONTEXT", "NON_EQUIVALENCE"]
    source_selector: CanonicalId
    target_selector: CanonicalId | None
    source_value_schema_ref: CanonicalId
    target_value_schema_ref: CanonicalId | None
    source_unit_ref: CanonicalId | None
    target_unit_ref: CanonicalId | None
    transform_ref: DefinitionRef | None
    non_equivalence_reason_ref: DefinitionRef | None
    mapping_owner_ref: OwnerRef
    mapping_version: CanonicalId
    accepted_source_domain_ref: DefinitionRef
    mapping_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_relation_shape(self) -> Self:
        if self.mapping_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("q=0 mapping definitions must use the compiler owner")
        if self.mapping_version != "mapping-version:spatialcf/upright-se2/m2-q0/1":
            raise ValueError("q=0 mapping definitions must use the fixed version")
        expected_mapping_ref = (
            "definition:spatialcf/upright-se2/m2-q0/mapping/"
            f"{self.source_selector.removeprefix('source:').replace('/', '-')}/1.0"
        )
        if self.mapping_definition_ref != expected_mapping_ref:
            raise ValueError(
                "q=0 mapping definition reference must bind its source selector"
            )
        if self.accepted_source_domain_ref != _M2_Q0_SUPPORTED_DOMAIN_REF:
            raise ValueError(
                "q=0 mapping definitions must name the sole supported domain"
            )
        if self.row_kind == "EQUALITY":
            if (
                self.target_selector is None
                or self.target_value_schema_ref is None
                or self.transform_ref is None
                or self.non_equivalence_reason_ref is not None
            ):
                raise ValueError(
                    "equality mapping definition must declare one exact transform"
                )
            if (
                self.source_selector != "source:constraints/position-domain"
                or self.target_selector != "target:operation/translation-domain"
                or self.source_value_schema_ref != _M2_Q0_DOMAIN_SCHEMA_REF
                or self.target_value_schema_ref != _M2_Q0_DOMAIN_SCHEMA_REF
                or self.source_unit_ref
                != "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                or self.target_unit_ref
                != "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                or self.transform_ref != _M2_Q0_DOMAIN_TRANSFORM_REF
            ):
                raise ValueError(
                    "q=0 equality must be the exact world-XY domain mapping"
                )
        elif self.row_kind == "SOURCE_CONTEXT":
            if any(
                value is not None
                for value in (
                    self.target_selector,
                    self.target_value_schema_ref,
                    self.target_unit_ref,
                    self.transform_ref,
                    self.non_equivalence_reason_ref,
                )
            ):
                raise ValueError(
                    "source-context mapping definition must not name an M3 target"
                )
            if (
                not self.source_selector.startswith("source:compilation/")
                or self.source_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/source-leaf/1.0"
                or self.source_unit_ref is not None
            ):
                raise ValueError(
                    "source-context mapping definition must bind one typed source leaf"
                )
        else:
            if (
                self.target_selector is None
                or self.target_value_schema_ref is None
                or self.non_equivalence_reason_ref is None
                or self.transform_ref is not None
            ):
                raise ValueError(
                    "non-equivalence mapping definition must name target and reason"
                )
            expected_non_equivalence = {
                "source:policy/objective": (
                    "target:policy/objective",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-objective/1.0",
                ),
                "source:policy/relation": (
                    "target:policy/relation",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-relation/1.0",
                ),
                "source:policy/visibility": (
                    "target:policy/visibility",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-visibility/1.0",
                ),
                "source:policy/support": (
                    "target:policy/support",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-support/1.0",
                ),
                "source:policy/numeric": (
                    "target:policy/numeric",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-numeric/1.0",
                ),
                "source:policy/resource": (
                    "target:policy/resource",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-resource/1.0",
                ),
                "source:policy/tie-break": (
                    "target:policy/tie-break",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-tie-break/1.0",
                ),
            }
            if (
                self.source_selector not in expected_non_equivalence
                or self.target_selector
                != expected_non_equivalence[self.source_selector][0]
                or self.non_equivalence_reason_ref
                != expected_non_equivalence[self.source_selector][1]
                or self.source_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/source-leaf/1.0"
                or self.target_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/target-leaf/1.0"
                or self.source_unit_ref is not None
                or self.target_unit_ref is not None
            ):
                raise ValueError(
                    "q=0 non-equivalence must bind its fixed heterogeneous policy"
                )
        return self


class UprightSE2M2Q0MappingRow(HashBoundCanonicalModel):
    """One fully bound q=0 relation row over exact source provenance."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/mapping-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "mapping_row_sha256"

    mapping_definition: UprightSE2M2Q0MappingDefinition
    row_kind: Literal["EQUALITY", "SOURCE_CONTEXT", "NON_EQUIVALENCE"]
    source_value_sha256: Sha256Digest
    target_value_sha256: Sha256Digest | None
    shared_value: TypedValue | None
    shared_value_sha256: Sha256Digest | None
    non_equivalence_reason_ref: DefinitionRef | None
    mapping_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_bound_relation(self) -> Self:
        if self.row_kind != self.mapping_definition.row_kind:
            raise ValueError("q=0 mapping row kind must match its definition")
        if self.row_kind == "EQUALITY":
            if (
                self.target_value_sha256 != self.source_value_sha256
                or self.shared_value is None
                or self.shared_value_sha256 is None
                or self.source_value_sha256 != self.shared_value_sha256
                or self.non_equivalence_reason_ref is not None
            ):
                raise ValueError(
                    "equality mapping row must bind one shared normalized value"
                )
            if self.shared_value_sha256 != canonical_sha256(
                self.shared_value,
                domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
            ):
                raise ValueError("equality mapping row shared-value digest is wrong")
        elif self.row_kind == "SOURCE_CONTEXT":
            if any(
                value is not None
                for value in (
                    self.target_value_sha256,
                    self.shared_value,
                    self.shared_value_sha256,
                    self.non_equivalence_reason_ref,
                )
            ):
                raise ValueError("source-context mapping row must retain source only")
        else:
            if (
                self.target_value_sha256 is None
                or self.shared_value is not None
                or self.shared_value_sha256 is not None
                or self.non_equivalence_reason_ref
                != self.mapping_definition.non_equivalence_reason_ref
            ):
                raise ValueError(
                    "non-equivalence mapping row must bind target and reason"
                )
        return self


class UprightSE2M2Q0SourceFreeConstructionRow(HashBoundCanonicalModel):
    """One target-only angular or ordering constant, never a source mapping."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/source-free-construction-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "source_free_construction_row_sha256"

    construction_selector: CanonicalId
    target_value: TypedValue
    target_value_sha256: Sha256Digest
    construction_owner_ref: OwnerRef
    construction_version: CanonicalId
    source_free_construction_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_source_free_value(self) -> Self:
        if self.construction_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("source-free construction row must use the compiler owner")
        if (
            self.construction_version
            != "construction-version:spatialcf/upright-se2/m2-q0/1"
        ):
            raise ValueError("source-free construction row must use the fixed version")
        if self.target_value_sha256 != canonical_sha256(
            self.target_value,
            domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
        ):
            raise ValueError("source-free construction value digest is wrong")
        return self


class UprightSE2M2Q0Construction(HashBoundCanonicalModel):
    """The narrow domain-only bridge from one exact retained M2 compilation.

    The full source compilation is embedded verbatim as provenance.  The sole
    equality claim is the declared supported world-XY domain transform; every
    heterogeneous policy root remains source context or explicit
    non-equivalence evidence.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/construction/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "m2_q0_construction_sha256"

    source_compilation: PlanarTranslateCompilation
    source_compilation_sha256: Sha256Digest
    source_provenance_sha256: Sha256Digest
    authorized_domain: UprightSE2TranslationDomain
    authorized_domain_sha256: Sha256Digest
    construction_q: Literal[0] = 0
    mapping_definitions: tuple[UprightSE2M2Q0MappingDefinition, ...]
    mapping_definition_roster_sha256: Sha256Digest
    rows: tuple[UprightSE2M2Q0MappingRow, ...]
    source_free_construction_rows: tuple[UprightSE2M2Q0SourceFreeConstructionRow, ...]
    frozen_m3_policy_bundle: UprightSE2ExecutablePolicyBundle
    frozen_m3_policy_bundle_sha256: Sha256Digest
    m2_q0_construction_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_closed_construction(self) -> Self:
        if self.source_compilation_sha256 != self.source_compilation.compilation_sha256:
            raise ValueError("q=0 construction must bind its exact source compilation")
        if self.source_provenance_sha256 != canonical_sha256(
            self.source_compilation,
            domain=_M2_Q0_SOURCE_PROVENANCE_HASH_DOMAIN,
        ):
            raise ValueError("q=0 construction source provenance digest is wrong")
        if self.authorized_domain_sha256 != canonical_sha256(
            self.authorized_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        ):
            raise ValueError("q=0 construction authorized-domain digest is wrong")
        if (
            self.frozen_m3_policy_bundle_sha256
            != self.frozen_m3_policy_bundle.policy_bundle_sha256
        ):
            raise ValueError("q=0 construction must bind its executable M3 policy")
        expected_domain = _m2_q0_authorized_domain_from_source(self.source_compilation)
        if self.authorized_domain != expected_domain:
            raise ValueError(
                "q=0 construction authorized domain must derive from source provenance"
            )
        definitions = tuple(
            sorted(
                self.mapping_definitions, key=lambda item: canonical_json_bytes(item)
            )
        )
        if self.mapping_definitions != definitions:
            raise ValueError("q=0 mapping definitions must use canonical source order")
        if self.mapping_definition_roster_sha256 != canonical_sha256(
            tuple(item.mapping_definition_sha256 for item in definitions),
            domain=_M2_Q0_MAPPING_DEFINITION_ROSTER_HASH_DOMAIN,
        ):
            raise ValueError("q=0 mapping-definition roster digest is wrong")
        definition_bytes = {canonical_json_bytes(item) for item in definitions}
        if any(
            canonical_json_bytes(row.mapping_definition) not in definition_bytes
            for row in self.rows
        ):
            raise ValueError("q=0 mapping rows must use the closed definition roster")
        if (
            len(self.rows) != len(definitions)
            or {canonical_json_bytes(row.mapping_definition) for row in self.rows}
            != definition_bytes
        ):
            raise ValueError(
                "q=0 mapping rows must bind every closed definition exactly once"
            )
        if self.rows != tuple(sorted(self.rows, key=canonical_json_bytes)):
            raise ValueError("q=0 mapping rows must use canonical source order")
        selectors = tuple(row.mapping_definition.source_selector for row in self.rows)
        if len(selectors) != len(set(selectors)):
            raise ValueError(
                "q=0 source selectors must be exhaustive without duplicates"
            )
        source_context_selectors = {
            row.mapping_definition.source_selector
            for row in self.rows
            if row.row_kind == "SOURCE_CONTEXT"
        }
        if source_context_selectors != _m2_q0_source_leaf_selectors(
            self.source_compilation
        ):
            raise ValueError(
                "q=0 construction must retain every source leaf exactly once"
            )
        equality_rows = tuple(row for row in self.rows if row.row_kind == "EQUALITY")
        if len(equality_rows) != 1:
            raise ValueError("q=0 construction must bind one supported-domain equality")
        equality = equality_rows[0]
        expected_domain_value = _m2_q0_domain_typed_value(expected_domain)
        expected_domain_digest = canonical_sha256(
            expected_domain_value,
            domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
        )
        if (
            equality.shared_value != expected_domain_value
            or equality.source_value_sha256 != expected_domain_digest
            or equality.target_value_sha256 != expected_domain_digest
            or equality.shared_value_sha256 != expected_domain_digest
        ):
            raise ValueError(
                "q=0 construction authorized-domain equality must bind source and target"
            )
        source_leaf_values = _m2_q0_source_leaf_values(self.source_compilation)
        for row in self.rows:
            if row.row_kind != "SOURCE_CONTEXT":
                continue
            selector = row.mapping_definition.source_selector
            if row.source_value_sha256 != canonical_sha256(
                source_leaf_values[selector],
                domain="spatialcf/counterfactual/upright-se2/m2-q0/source-leaf/3.0",
            ):
                raise ValueError(
                    "q=0 source-context mapping digest must bind source provenance"
                )
        expected_non_equivalence = _m2_q0_non_equivalence_values(
            self.source_compilation,
            self.frozen_m3_policy_bundle,
        )
        non_equivalence_rows = {
            row.mapping_definition.source_selector: row
            for row in self.rows
            if row.row_kind == "NON_EQUIVALENCE"
        }
        if set(non_equivalence_rows) != set(expected_non_equivalence):
            raise ValueError(
                "q=0 construction must bind every heterogeneous policy root"
            )
        for selector, (source_value, target_value) in expected_non_equivalence.items():
            row = non_equivalence_rows[selector]
            if row.source_value_sha256 != canonical_sha256(
                source_value,
                domain="spatialcf/counterfactual/upright-se2/m2-q0/source-leaf/3.0",
            ) or row.target_value_sha256 != canonical_sha256(
                target_value,
                domain="spatialcf/counterfactual/upright-se2/m2-q0/target-leaf/3.0",
            ):
                raise ValueError(
                    "q=0 non-equivalence mapping digest must bind source and M3 policy"
                )
        if not any(row.row_kind == "SOURCE_CONTEXT" for row in self.rows):
            raise ValueError("q=0 construction must retain source provenance context")
        if not self.source_free_construction_rows:
            raise ValueError("q=0 construction must bind source-free angular constants")
        if self.source_free_construction_rows != tuple(
            sorted(self.source_free_construction_rows, key=canonical_json_bytes)
        ):
            raise ValueError(
                "q=0 source-free construction rows must use canonical order"
            )
        source_free_by_selector = {
            row.construction_selector: row for row in self.source_free_construction_rows
        }
        if set(source_free_by_selector) != {
            "construction:cardinal-own-pivot-q",
            "construction:target-tie-break",
        }:
            raise ValueError(
                "q=0 construction must bind its exact source-free constants"
            )
        q_row = source_free_by_selector["construction:cardinal-own-pivot-q"]
        tie_break_row = source_free_by_selector["construction:target-tie-break"]
        if (
            q_row.target_value.value_schema_ref
            != "schema:spatialcf/upright-se2/integer/1.0"
            or type(q_row.target_value.payload) is not IntegerValue
            or q_row.target_value.payload.value != 0
            or tie_break_row.target_value.value_schema_ref
            != "schema:spatialcf/upright-se2/enum-symbol/1.0"
            or type(tie_break_row.target_value.payload) is not EnumSymbolValue
            or tie_break_row.target_value.payload.symbol != "T_R_V_S_A"
        ):
            raise ValueError(
                "q=0 source-free constants must be fixed construction values"
            )
        return self


class UprightSE2Compilation(HashBoundCanonicalModel):
    """A deterministic cardinal compilation, never a solve/check/certificate result."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/upright-se2/compilation/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "upright_se2_compilation_sha256"

    solve_request_sha256: Sha256Digest
    source_solve_request: CounterfactualSolveRequest
    closure: UprightSE2CompilerClosure
    operation: UprightSE2CardinalOperation
    endpoint_construction_recipe: UprightSE2EndpointConstructionRecipe
    state_footprint: UprightSE2StateFootprint
    grounded_obligations: GroundedObligationSet
    semantic_closure: UprightSE2SemanticClosure
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    m2_q0_construction: UprightSE2M2Q0Construction | None = None
    upright_se2_compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_compilation_cells(self) -> Self:
        if self.source_solve_request.solve_request_sha256 != self.solve_request_sha256:
            raise ValueError(
                "compilation solve request digest does not bind its source request"
            )
        if (
            self.source_solve_request.semantic_problem_sha256
            != self.semantic_closure.semantic_problem.semantic_problem_sha256
            or canonical_json_bytes(self.source_solve_request.semantic_problem)
            != canonical_json_bytes(self.semantic_closure.semantic_problem)
        ):
            raise ValueError(
                "compilation semantic closure does not bind the source request"
            )
        if (
            self.closure.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
        ):
            raise ValueError(
                "compiler and semantic closures must bind the same profile digest"
            )
        if (
            self.semantic_closure.profile_registration.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
        ):
            raise ValueError(
                "semantic closure profile does not bind its profile digest"
            )
        if (
            self.endpoint_construction_recipe.operation_authorization_sha256
            != self.operation.authorization_sha256
            or self.endpoint_construction_recipe.translation_domain
            != self.operation.translation_domain
        ):
            raise ValueError(
                "compilation endpoint recipe must bind its operation domain"
            )
        if (
            self.semantic_closure.grounded_obligation_set_sha256
            != self.grounded_obligations.grounded_obligation_set_sha256
        ):
            raise ValueError(
                "semantic closure must bind the compiled grounded obligations"
            )
        if (
            self.semantic_closure.semantic_closure_sha256
            != self.closure.semantic_closure_sha256
        ):
            raise ValueError("compiler closure must bind the semantic closure")
        if (
            self.semantic_closure.policy_bundle_sha256
            != self.closure.policy_bundle_sha256
        ):
            raise ValueError("compiler closure must bind the executable policy bundle")
        if (
            self.semantic_closure.resource_policy.resource_policy_sha256
            != self.closure.resource_policy_sha256
        ):
            raise ValueError("compiler closure must bind the request resource policy")
        if (
            self.semantic_closure.definition_bundle_sha256
            != self.closure.definition_bundle_sha256
        ):
            raise ValueError(
                "compiler closure must bind the semantic definition bundle"
            )
        if not self.compiled_cells:
            raise ValueError(
                "upright se2 compilation must include one or more canonical cells"
            )
        if tuple(cell.authorization_sha256 for cell in self.compiled_cells) != (
            self.operation.authorization_sha256,
        ):
            raise ValueError(
                "compiled cells must bind the resolved cardinal authorization"
            )
        if self.m2_q0_construction is not None:
            construction = self.m2_q0_construction
            if (
                self.operation.quarter_turns_ccw != 0
                or self.operation.pivot_binding.pivot_mode is not PivotMode.OWN
                or self.operation.translation_domain != construction.authorized_domain
                or self.semantic_closure.policy_bundle_sha256
                != construction.frozen_m3_policy_bundle_sha256
            ):
                raise ValueError(
                    "q=0 construction must bind the domain-only M3 operation"
                )
            construction_facts = tuple(
                fact
                for bundle in self.source_solve_request.semantic_problem.scene_state.extension_fact_bundles
                for fact in bundle.facts
                if fact.fact_family_ref
                == "definition:spatialcf/upright-se2/m2-q0-construction/1.0"
            )
            if len(construction_facts) != 1:
                raise ValueError(
                    "q=0 construction root must be present in the source request"
                )
            fact = construction_facts[0]
            if (
                fact.fact_key != "fact-key:spatialcf/upright-se2/m2-q0-construction"
                or type(fact.value.payload) is not DigestValue
                or fact.value.payload.value != construction.m2_q0_construction_sha256
            ):
                raise ValueError(
                    "q=0 construction request root does not bind construction"
                )
        _validate_compilation_source_binding(self)
        return self


class UprightSE2ContinuousCompilation(HashBoundCanonicalModel):
    """A deterministic continuous compilation, never a checked result.

    This additive model has its own hash domain and carries the canonical
    lifted-yaw roots.  Cardinal compilation remains closed over its existing
    type and wire.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-compilation/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_upright_se2_compilation_sha256"

    solve_request_sha256: Sha256Digest
    source_solve_request: CounterfactualSolveRequest
    closure: UprightSE2CompilerClosure
    operation: UprightSE2ContinuousOperation
    endpoint_construction_recipe: UprightSE2ContinuousEndpointConstructionRecipe
    state_footprint: UprightSE2StateFootprint
    grounded_obligations: GroundedObligationSet
    semantic_closure: UprightSE2ContinuousSemanticClosure
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    continuous_yaw_lift: ContinuousYawLift
    continuous_upright_se2_compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_compilation(self) -> Self:
        if self.source_solve_request.solve_request_sha256 != self.solve_request_sha256:
            raise ValueError("continuous compilation must bind its source request")
        if (
            self.closure.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
            or self.closure.semantic_closure_sha256
            != self.semantic_closure.semantic_closure_sha256
            or self.closure.policy_bundle_sha256
            != self.semantic_closure.policy_bundle_sha256
            or self.closure.resource_policy_sha256
            != self.semantic_closure.resource_policy.resource_policy_sha256
        ):
            raise ValueError("continuous compiler closure does not bind semantics")
        if (
            self.endpoint_construction_recipe.operation_authorization_sha256
            != self.operation.authorization_sha256
            or self.endpoint_construction_recipe.translation_domain
            != self.operation.translation_domain
            or self.endpoint_construction_recipe.yaw_domain != self.operation.yaw_domain
        ):
            raise ValueError("continuous endpoint recipe does not bind operation")
        if self.continuous_yaw_lift.yaw_domain != self.operation.yaw_domain:
            raise ValueError("continuous lift does not bind operation yaw domain")
        if len(self.compiled_cells) != 1:
            raise ValueError("continuous compilation requires one canonical lift root")
        cell = self.compiled_cells[0]
        interval = self.continuous_yaw_lift.intervals[0]
        if (
            cell.authorization_sha256 != self.operation.authorization_sha256
            or cell.x_lower != self.operation.translation_domain.x_lower
            or cell.x_upper != self.operation.translation_domain.x_upper
            or cell.y_lower != self.operation.translation_domain.y_lower
            or cell.y_upper != self.operation.translation_domain.y_upper
            or cell.yaw_interval != interval
        ):
            raise ValueError("continuous compilation root does not bind operation")
        _validate_continuous_compilation_source_binding(self)
        return self


class UprightSE2ContinuousMaterializedEndpoint(HashBoundCanonicalModel):
    """One continuous endpoint selected by the compiler-owned materializer."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-materialized-endpoint/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_materialized_endpoint_sha256"

    continuous_upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2ContinuousCompilation
    endpoint_construction_recipe: UprightSE2ContinuousEndpointConstructionRecipe
    translation_xy_m: Vec2
    selected_lifted_yaw: ExactDyadic
    after_state: UprightSE2AfterStateTemplate
    program: EditProgram
    continuous_materialized_endpoint_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_endpoint_from_recipe(self) -> Self:
        compilation = self.compilation
        recipe = self.endpoint_construction_recipe
        if (
            self.continuous_upright_se2_compilation_sha256
            != compilation.continuous_upright_se2_compilation_sha256
            or recipe != compilation.endpoint_construction_recipe
        ):
            raise ValueError("continuous endpoint must bind its compilation recipe")
        x = _fraction_from_float(self.translation_xy_m.x)
        y = _fraction_from_float(self.translation_xy_m.y)
        yaw = self.selected_lifted_yaw.as_fraction
        root = compilation.compiled_cells[0]
        if not (
            recipe.translation_domain.x_lower.as_fraction
            <= x
            <= recipe.translation_domain.x_upper.as_fraction
            and recipe.translation_domain.y_lower.as_fraction
            <= y
            <= recipe.translation_domain.y_upper.as_fraction
            and root.yaw_interval.lower.as_fraction
            <= yaw
            <= root.yaw_interval.upper.as_fraction
        ):
            raise ValueError("continuous endpoint point is outside its authorized root")
        if (
            isinstance(recipe.yaw_domain, ContinuousYawFullCircle)
            and yaw == root.yaw_interval.upper.as_fraction
        ):
            raise ValueError(
                "continuous endpoint rejects the upper full-circle seam alias"
            )
        expected_xy, expected_yaw, expected_pose, expected_reference = (
            _continuous_materialized_expected_pose(
                recipe,
                self.translation_xy_m,
                self.selected_lifted_yaw,
            )
        )
        if (
            self.after_state.evaluation_scene != recipe.evaluation_scene
            or self.after_state.subject_id != recipe.subject_id
            or self.after_state.subject_before_pose != recipe.subject_before_pose
            or self.after_state.reference_pivot_xy_m != expected_reference
            or self.after_state.subject_pivot_xy_m != expected_xy
            or self.after_state.subject_pivot_z_m != expected_pose.translation.z
            or self.after_state.subject_yaw_turns != expected_yaw
            or self.after_state.subject_pose != expected_pose
        ):
            raise ValueError(
                "continuous endpoint after state must derive from its recipe"
            )
        _validate_continuous_materialized_endpoint_program(self)
        return self


def _continuous_materialized_program_arguments(
    compilation: UprightSE2ContinuousCompilation,
    translation_xy_m: Vec2,
    selected_lifted_yaw: ExactDyadic,
) -> tuple[OperationArgument, ...]:
    """Derive the sole continuous invocation from sealed endpoint inputs."""

    return tuple(
        sorted(
            (
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/pivot-entity-id",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/entity-id/1.0",
                        payload=CanonicalIdValue(
                            value=compilation.operation.pivot_binding.pivot_entity_id
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/selected-lifted-yaw-turn",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/exact-dyadic-turn/1.0",
                        payload=CanonicalIdValue(
                            value=(
                                "exact-dyadic:"
                                f"{selected_lifted_yaw.numerator}/"
                                f"{selected_lifted_yaw.denominator}"
                            ),
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/subject-id",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/entity-id/1.0",
                        payload=CanonicalIdValue(
                            value=compilation.endpoint_construction_recipe.subject_id
                        ),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/translation-x-m",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/metre/1.0",
                        payload=FiniteRealValue(value=translation_xy_m.x),
                    ),
                ),
                OperationArgument(
                    argument_name="argument:spatialcf/upright-se2/translation-y-m",
                    value=TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/metre/1.0",
                        payload=FiniteRealValue(value=translation_xy_m.y),
                    ),
                ),
            ),
            key=lambda argument: canonical_json_bytes(argument.argument_name),
        )
    )


def _validate_continuous_materialized_endpoint_program(
    endpoint: UprightSE2ContinuousMaterializedEndpoint,
) -> None:
    """Close one continuous program over its sealed source and endpoint."""

    compilation = endpoint.compilation
    source_problem = compilation.source_solve_request.semantic_problem
    before_state = source_problem.scene_state
    program = endpoint.program
    if (
        program.semantic_problem_sha256 != source_problem.semantic_problem_sha256
        or program.action_space_profile_sha256
        != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
        or program.before_state_sha256 != before_state.scene_state_sha256
        or program.grounded_obligation_set_sha256
        != compilation.grounded_obligations.grounded_obligation_set_sha256
        or program.state_delta_manifest
        != compilation.state_footprint.state_delta_manifest
    ):
        raise ValueError("continuous endpoint program roots do not bind compilation")
    expected_step = OperationInvocation(
        operator_ref=compilation.operation.authorization.operator_ref,
        arguments=_continuous_materialized_program_arguments(
            compilation,
            endpoint.translation_xy_m,
            endpoint.selected_lifted_yaw,
        ),
    )
    if program.steps != (expected_step,):
        raise ValueError(
            "continuous endpoint program must contain exactly one bound invocation"
        )
    expected_after = _materialized_expected_after_scene_state(
        compilation,
        endpoint.after_state,
    )
    if (
        program.after_scene_state_sha256 != expected_after.scene_state_sha256
        or canonical_json_bytes(program.after_scene_state)
        != canonical_json_bytes(expected_after)
    ):
        raise ValueError(
            "continuous endpoint program must equal the complete expected after scene"
        )


class UprightSE2ContinuousProofTuple(HashBoundCanonicalModel):
    """The one request-authorized continuous operator/reference/lift tuple.

    Continuous search may subdivide this root, but it must never add another
    operator, pivot, reference, translation domain, or lifted yaw domain.  The
    record intentionally mirrors the cardinal tuple without widening it.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-proof-tuple/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_proof_tuple_sha256"

    authorization: ContinuousYawAuthorization
    reference_id: CanonicalId
    translation_domain: UprightSE2TranslationDomain
    continuous_yaw_lift: ContinuousYawLift
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    continuous_proof_tuple_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_tuple(self) -> Self:
        if len(self.compiled_cells) != 1:
            raise ValueError("continuous proof tuple requires one lift root")
        root = self.compiled_cells[0]
        interval = self.continuous_yaw_lift.intervals[0]
        if (
            root.authorization_sha256
            != self.authorization.continuous_yaw_authorization_sha256
            or root.x_lower != self.translation_domain.x_lower
            or root.x_upper != self.translation_domain.x_upper
            or root.y_lower != self.translation_domain.y_lower
            or root.y_upper != self.translation_domain.y_upper
            or root.yaw_interval != interval
            or self.continuous_yaw_lift.yaw_domain != self.authorization.yaw_domain
        ):
            raise ValueError("continuous proof tuple root does not bind its authority")
        return self


class UprightSE2ContinuousProposalCandidate(HashBoundCanonicalModel):
    """One compiler-materialized continuous inward witness.

    It transports retained point evidence only.  The backend may choose the
    candidate, but it cannot construct the endpoint, program, objective, or
    certificate represented here.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-proposal-candidate/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_proposal_candidate_sha256"

    final_inward_cell: UprightSE2ProofCellEvaluation
    selected_translation_xy_m: Vec2
    selected_lifted_yaw: ExactDyadic
    point_evaluation: UprightSE2ProposalPointEvaluation
    point_objective: UprightSE2ProposalPointObjective
    materialized_endpoint: UprightSE2ContinuousMaterializedEndpoint
    program: EditProgram
    continuous_proposal_candidate_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_candidate(self) -> Self:
        if self.final_inward_cell.leaf_disposition not in (
            UprightSE2ProofLeafDisposition.INWARD_FEASIBLE,
            UprightSE2ProofLeafDisposition.UNRESOLVED,
        ):
            raise ValueError(
                "continuous proposal candidates require one final inward or unresolved cell"
            )
        if canonical_json_bytes(self.point_objective) != canonical_json_bytes(
            self.point_evaluation.point_objective
        ):
            raise ValueError("continuous candidate objective must bind point evidence")
        if (
            self.selected_translation_xy_m
            != self.materialized_endpoint.translation_xy_m
            or self.selected_lifted_yaw
            != self.materialized_endpoint.selected_lifted_yaw
            or canonical_json_bytes(self.program)
            != canonical_json_bytes(self.materialized_endpoint.program)
        ):
            raise ValueError("continuous candidate must bind its materialized endpoint")
        cell = self.final_inward_cell.compiled_cell
        point_x = _fraction_from_float(self.selected_translation_xy_m.x)
        point_y = _fraction_from_float(self.selected_translation_xy_m.y)
        point_yaw = self.selected_lifted_yaw.as_fraction
        if not (
            cell.x_lower.as_fraction <= point_x <= cell.x_upper.as_fraction
            and cell.y_lower.as_fraction <= point_y <= cell.y_upper.as_fraction
            and cell.yaw_interval.lower.as_fraction
            <= point_yaw
            <= cell.yaw_interval.upper.as_fraction
        ):
            raise ValueError("continuous candidate point is outside its final cell")
        point_cell = self.point_evaluation.point_cell_evaluation.compiled_cell
        if (
            point_cell.x_lower.as_fraction != point_x
            or point_cell.x_upper.as_fraction != point_x
            or point_cell.y_lower.as_fraction != point_y
            or point_cell.y_upper.as_fraction != point_y
            or point_cell.yaw_interval.lower.as_fraction != point_yaw
            or point_cell.yaw_interval.upper.as_fraction != point_yaw
        ):
            raise ValueError(
                "continuous candidate point must bind its exact point cell"
            )
        return self

    @property
    def canonical_order_key(self) -> tuple[Fraction, Fraction, bytes, bytes]:
        """Use the frozen objective/program witness key, never traversal IDs."""

        return (
            self.point_objective.total_upper.as_fraction,
            self.point_objective.total_lower.as_fraction,
            canonical_json_bytes(self.point_objective.terms),
            canonical_json_bytes(self.program),
        )


def _continuous_proof_cell_within_root(
    cell: UprightSE2CompiledCell,
    root: UprightSE2CompiledCell,
) -> bool:
    """Return whether a proof cell is an exact-dyadic `(x, y, u)` descendant."""

    return (
        cell.authorization_sha256 == root.authorization_sha256
        and root.x_lower.as_fraction <= cell.x_lower.as_fraction
        and cell.x_upper.as_fraction <= root.x_upper.as_fraction
        and root.y_lower.as_fraction <= cell.y_lower.as_fraction
        and cell.y_upper.as_fraction <= root.y_upper.as_fraction
        and root.yaw_interval.lower.as_fraction <= cell.yaw_interval.lower.as_fraction
        and cell.yaw_interval.upper.as_fraction <= root.yaw_interval.upper.as_fraction
    )


def _continuous_proof_leaf_owns_coordinate(
    *,
    lower: Fraction,
    upper: Fraction,
    coordinate: Fraction,
    root_lower: Fraction,
    root_upper: Fraction,
    full_circle_alias: bool,
) -> bool:
    """Apply lower-owned seams, including the full-circle upper closure alias."""

    if full_circle_alias and coordinate == root_upper:
        return lower == root_lower
    return (coordinate == root_lower and lower == root_lower) or (
        lower < coordinate <= upper
    )


def _typed_continuous_bound_proves_inward_feasibility(value: TypedValue) -> bool:
    """Read one complete retained V4 feasibility record from a structural bound."""

    payload = value.payload
    if type(payload) is not RecordValue:
        return False
    fields = {field.name: field.value for field in payload.fields}
    inner_hard_constraint_proven = fields.get("inner_hard_constraint_proven")
    relation_inner_success = fields.get("relation_inner_success")
    classification = fields.get("visibility_classification")
    return (
        inner_hard_constraint_proven is not None
        and type(inner_hard_constraint_proven.payload) is BooleanValue
        and inner_hard_constraint_proven.payload.value is True
        and relation_inner_success is not None
        and type(relation_inner_success.payload) is BooleanValue
        and relation_inner_success.payload.value is True
        and classification is not None
        and type(classification.payload) is CanonicalIdValue
        and classification.payload.value == "INWARD"
    )


def _continuous_point_retains_inward_evidence(
    point_row: UprightSE2ProofCellEvaluation,
) -> bool:
    """Bind feasibility to one exact V4 compound row, not objective transport."""

    return any(
        evaluation.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT
        and "proof:spatialcf/upright-se2/continuous-compound-cell"
        in evaluation.proof_rows
        and any(
            _typed_continuous_bound_proves_inward_feasibility(bound)
            for bound in evaluation.exact_bounds
        )
        for evaluation in point_row.owner_evaluations
    )


def _continuous_proof_validate_leaf_partition(
    root: UprightSE2CompiledCell,
    leaves: tuple[UprightSE2CompiledCell, ...],
    *,
    full_circle_alias: bool,
) -> None:
    """Require one exact-dyadic, lower-owned `(x, y, u)` leaf partition."""

    if not leaves or any(
        not _continuous_proof_cell_within_root(leaf, root) for leaf in leaves
    ):
        raise ValueError("continuous proof leaves must cover one exact root")
    axes = (
        ("x", root.x_lower.as_fraction, root.x_upper.as_fraction),
        ("y", root.y_lower.as_fraction, root.y_upper.as_fraction),
        ("u", root.yaw_interval.lower.as_fraction, root.yaw_interval.upper.as_fraction),
    )
    leaf_bounds = {
        "x": tuple(
            (leaf.x_lower.as_fraction, leaf.x_upper.as_fraction) for leaf in leaves
        ),
        "y": tuple(
            (leaf.y_lower.as_fraction, leaf.y_upper.as_fraction) for leaf in leaves
        ),
        "u": tuple(
            (
                leaf.yaw_interval.lower.as_fraction,
                leaf.yaw_interval.upper.as_fraction,
            )
            for leaf in leaves
        ),
    }
    coordinates: dict[str, tuple[Fraction, ...]] = {}
    segments: dict[str, tuple[tuple[Fraction, Fraction], ...]] = {}
    for axis, root_lower, root_upper in axes:
        values = tuple(
            sorted(
                {
                    root_lower,
                    root_upper,
                    *(value for pair in leaf_bounds[axis] for value in pair),
                }
            )
        )
        coordinates[axis] = values
        segments[axis] = (
            ((root_lower, root_upper),)
            if root_lower == root_upper
            else tuple(pairwise(values))
        )
    for x_lower, x_upper in segments["x"]:
        for y_lower, y_upper in segments["y"]:
            for u_lower, u_upper in segments["u"]:
                covering = tuple(
                    leaf
                    for leaf in leaves
                    if (
                        leaf.x_lower.as_fraction <= x_lower
                        and x_upper <= leaf.x_upper.as_fraction
                        and leaf.y_lower.as_fraction <= y_lower
                        and y_upper <= leaf.y_upper.as_fraction
                        and leaf.yaw_interval.lower.as_fraction <= u_lower
                        and u_upper <= leaf.yaw_interval.upper.as_fraction
                    )
                )
                if len(covering) != 1:
                    raise ValueError(
                        "continuous proof leaves must form one exact-dyadic partition"
                    )
    for x in coordinates["x"]:
        for y in coordinates["y"]:
            for u in coordinates["u"]:
                owners = tuple(
                    leaf
                    for leaf in leaves
                    if _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.x_lower.as_fraction,
                        upper=leaf.x_upper.as_fraction,
                        coordinate=x,
                        root_lower=root.x_lower.as_fraction,
                        root_upper=root.x_upper.as_fraction,
                        full_circle_alias=False,
                    )
                    and _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.y_lower.as_fraction,
                        upper=leaf.y_upper.as_fraction,
                        coordinate=y,
                        root_lower=root.y_lower.as_fraction,
                        root_upper=root.y_upper.as_fraction,
                        full_circle_alias=False,
                    )
                    and _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.yaw_interval.lower.as_fraction,
                        upper=leaf.yaw_interval.upper.as_fraction,
                        coordinate=u,
                        root_lower=root.yaw_interval.lower.as_fraction,
                        root_upper=root.yaw_interval.upper.as_fraction,
                        full_circle_alias=full_circle_alias,
                    )
                )
                if len(owners) != 1:
                    raise ValueError(
                        "continuous proof leaves must assign every exact seam once"
                    )


class UprightSE2ContinuousProofMaterial(HashBoundCanonicalModel):
    """Complete untrusted continuous branch-and-bound evidence for Task 8.

    This is deliberately a profile proof transport only: fresh geometry replay,
    proof acceptance, certificates, and terminal result assembly remain outside
    this domain record and outside the proposal backend.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-proof-material/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_proof_material_sha256"

    solve_request_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    continuous_upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2ContinuousCompilation
    continuous_tuple_roster: tuple[UprightSE2ContinuousProofTuple, ...]
    coverage_artifact: UprightSE2CoverageArtifact
    compiled_cell_sha256s: tuple[Sha256Digest, ...]
    evaluated_cells: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_order: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_candidates: tuple[UprightSE2ContinuousProposalCandidate, ...] = ()
    incumbent_candidate_sha256: Sha256Digest | None
    prune_decisions: tuple[UprightSE2ProofPruneDecision, ...]
    unresolved_frontier: tuple[UprightSE2ProofFrontierRow, ...]
    resource_ledger: UprightSE2ProofResourceLedger
    continuous_proof_material_sha256: Sha256Digest

    @property
    def total_resource_usage(self) -> ResourceUsage:
        """Expose the one shared ledger total for the outer submission."""

        return self.resource_ledger.canonical_total_resource_usage

    @model_validator(mode="after")
    def _validate_continuous_proof_material(self) -> Self:
        compilation = self.compilation
        if (
            self.solve_request_sha256 != compilation.solve_request_sha256
            or self.semantic_closure_sha256
            != compilation.semantic_closure.semantic_closure_sha256
            or self.continuous_upright_se2_compilation_sha256
            != compilation.continuous_upright_se2_compilation_sha256
        ):
            raise ValueError("continuous proof material roots do not bind compilation")
        if len(self.continuous_tuple_roster) != 1:
            raise ValueError("continuous proof tuple roster must contain one tuple")
        proof_tuple = self.continuous_tuple_roster[0]
        source_input = _bound_continuous_source_input(
            compilation.source_solve_request.semantic_problem
        )
        if (
            canonical_json_bytes(proof_tuple.authorization)
            != canonical_json_bytes(compilation.operation.authorization)
            or proof_tuple.reference_id != source_input.reference_id
            or proof_tuple.translation_domain
            != compilation.operation.translation_domain
            or proof_tuple.continuous_yaw_lift != compilation.continuous_yaw_lift
            or tuple(canonical_json_bytes(cell) for cell in proof_tuple.compiled_cells)
            != tuple(canonical_json_bytes(cell) for cell in compilation.compiled_cells)
            or proof_tuple.authorization.operator_ref != source_input.operator_ref
        ):
            raise ValueError("continuous proof tuple does not bind the source root")
        root = proof_tuple.compiled_cells[0]
        if (
            self.coverage_artifact.authorization_sha256
            != compilation.operation.authorization_sha256
        ):
            raise ValueError("continuous proof coverage does not bind authorization")
        _continuous_proof_validate_leaf_partition(
            root,
            self.coverage_artifact.cells,
            full_circle_alias=(
                type(compilation.operation.yaw_domain) is ContinuousYawFullCircle
            ),
        )
        expected_cell_sha256s = tuple(
            cell.compiled_cell_sha256 for cell in self.coverage_artifact.cells
        )
        if self.compiled_cell_sha256s != expected_cell_sha256s:
            raise ValueError("continuous proof must bind the canonical leaf roster")
        if self.evaluated_cells != tuple(
            sorted(
                self.evaluated_cells,
                key=lambda row: _lifted_cell_order_key(row.compiled_cell),
            )
        ):
            raise ValueError("continuous evaluated cells must use deterministic order")
        if len({row.compiled_cell.cell_id for row in self.evaluated_cells}) != len(
            self.evaluated_cells
        ) or len(
            {canonical_json_bytes(row.compiled_cell) for row in self.evaluated_cells}
        ) != len(self.evaluated_cells):
            raise ValueError("continuous proof rows must not duplicate cells")
        if any(
            not _continuous_proof_cell_within_root(row.compiled_cell, root)
            for row in self.evaluated_cells
        ):
            raise ValueError("continuous proof rows must be root descendants")
        final_rows = tuple(
            row for row in self.evaluated_cells if row.leaf_disposition is not None
        )
        if tuple(
            canonical_json_bytes(row.compiled_cell) for row in final_rows
        ) != tuple(canonical_json_bytes(cell) for cell in self.coverage_artifact.cells):
            raise ValueError("continuous final rows must exactly bind coverage leaves")
        expected_proposals = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        )
        if tuple(canonical_json_bytes(row) for row in self.proposal_order) != tuple(
            canonical_json_bytes(row) for row in expected_proposals
        ):
            raise ValueError("continuous proposal order must be complete and stable")
        candidate_cells = tuple(
            canonical_json_bytes(candidate.final_inward_cell)
            for candidate in self.proposal_candidates
        )
        expected_inward_candidate_cells = tuple(
            canonical_json_bytes(row) for row in expected_proposals
        )
        final_rows_by_bytes = {canonical_json_bytes(row): row for row in final_rows}
        if len(set(candidate_cells)) != len(candidate_cells) or not set(
            expected_inward_candidate_cells
        ).issubset(candidate_cells):
            raise ValueError(
                "continuous candidates must cover inward leaves exactly once"
            )
        if any(
            candidate_cell not in final_rows_by_bytes
            for candidate_cell in candidate_cells
        ):
            raise ValueError("continuous candidate parent must be a final proof leaf")
        if self.proposal_candidates != tuple(
            sorted(
                self.proposal_candidates,
                key=lambda candidate: candidate.canonical_order_key,
            )
        ):
            raise ValueError("continuous proposal candidates use the wrong order")
        expected_incumbent = (
            None
            if not self.proposal_candidates
            else self.proposal_candidates[0].continuous_proposal_candidate_sha256
        )
        if self.incumbent_candidate_sha256 != expected_incumbent:
            raise ValueError("continuous incumbent must be the canonical witness")
        for candidate in self.proposal_candidates:
            point_row = candidate.point_evaluation.point_cell_evaluation
            parent = candidate.final_inward_cell
            if parent.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED:
                if not any(
                    canonical_json_bytes(frontier.cell_evaluation)
                    == canonical_json_bytes(parent)
                    for frontier in self.unresolved_frontier
                ):
                    raise ValueError(
                        "continuous unresolved candidate parent must be in frontier"
                    )
                if point_row.leaf_disposition is not None:
                    raise ValueError(
                        "continuous unresolved candidates require an internal point row"
                    )
                if point_row.compiled_cell.cell_id != (
                    f"{parent.compiled_cell.cell_id}/proposal-point"
                ):
                    raise ValueError(
                        "continuous unresolved candidate point must bind its parent identity"
                    )
                if not _continuous_point_retains_inward_evidence(point_row):
                    raise ValueError(
                        "continuous unresolved candidate requires inward feasibility evidence"
                    )
            if not any(
                canonical_json_bytes(row) == canonical_json_bytes(point_row)
                for row in self.evaluated_cells
            ):
                raise ValueError("continuous candidate point row is absent from proof")
            point_cell = point_row.compiled_cell
            selected_u = candidate.selected_lifted_yaw.as_fraction
            if (
                point_cell.yaw_interval.lower.as_fraction != selected_u
                or point_cell.yaw_interval.upper.as_fraction != selected_u
            ):
                raise ValueError("continuous candidate point yaw is not degenerate")
            if (
                type(compilation.operation.yaw_domain) is ContinuousYawFullCircle
                and selected_u == root.yaw_interval.upper.as_fraction
            ):
                raise ValueError("full-circle upper closure alias cannot be a witness")
            final_cell = candidate.final_inward_cell.compiled_cell
            for lower, upper, coordinate in (
                (
                    final_cell.x_lower.as_fraction,
                    final_cell.x_upper.as_fraction,
                    point_cell.x_lower.as_fraction,
                ),
                (
                    final_cell.y_lower.as_fraction,
                    final_cell.y_upper.as_fraction,
                    point_cell.y_lower.as_fraction,
                ),
                (
                    final_cell.yaw_interval.lower.as_fraction,
                    final_cell.yaw_interval.upper.as_fraction,
                    selected_u,
                ),
            ):
                if (lower < upper and not lower < coordinate < upper) or (
                    lower == upper and coordinate != lower
                ):
                    raise ValueError(
                        "continuous proposal points require strict interior descendants"
                    )
            owners = tuple(
                row
                for row in final_rows
                if _continuous_proof_cell_within_root(point_cell, row.compiled_cell)
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.x_lower.as_fraction,
                    upper=row.compiled_cell.x_upper.as_fraction,
                    coordinate=point_cell.x_lower.as_fraction,
                    root_lower=root.x_lower.as_fraction,
                    root_upper=root.x_upper.as_fraction,
                    full_circle_alias=False,
                )
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.y_lower.as_fraction,
                    upper=row.compiled_cell.y_upper.as_fraction,
                    coordinate=point_cell.y_lower.as_fraction,
                    root_lower=root.y_lower.as_fraction,
                    root_upper=root.y_upper.as_fraction,
                    full_circle_alias=False,
                )
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.yaw_interval.lower.as_fraction,
                    upper=row.compiled_cell.yaw_interval.upper.as_fraction,
                    coordinate=selected_u,
                    root_lower=root.yaw_interval.lower.as_fraction,
                    root_upper=root.yaw_interval.upper.as_fraction,
                    full_circle_alias=(
                        type(compilation.operation.yaw_domain)
                        is ContinuousYawFullCircle
                    ),
                )
            )
            if len(owners) != 1 or canonical_json_bytes(
                owners[0]
            ) != canonical_json_bytes(candidate.final_inward_cell):
                raise ValueError("continuous candidate point lacks one owning leaf")
            policy_terms = compilation.semantic_closure.objective_policy.terms
            if tuple(term.term_id for term in candidate.point_objective.terms) != tuple(
                term.term_id for term in policy_terms
            ):
                raise ValueError("continuous candidate terms do not bind policy")
            expected_lower = sum(
                (
                    point_term.lower.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            expected_upper = sum(
                (
                    point_term.upper.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            if (
                candidate.point_objective.total_lower.as_fraction != expected_lower
                or candidate.point_objective.total_upper.as_fraction != expected_upper
            ):
                raise ValueError("continuous candidate total does not bind policy")
            if (
                candidate.materialized_endpoint.continuous_upright_se2_compilation_sha256
                != compilation.continuous_upright_se2_compilation_sha256
                or candidate.program.semantic_problem_sha256
                != compilation.source_solve_request.semantic_problem_sha256
                or candidate.program.action_space_profile_sha256
                != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
                or candidate.program.before_state_sha256
                != compilation.source_solve_request.semantic_problem.scene_state.scene_state_sha256
                or candidate.program.grounded_obligation_set_sha256
                != compilation.grounded_obligations.grounded_obligation_set_sha256
                or candidate.program.state_delta_manifest
                != compilation.state_footprint.state_delta_manifest
            ):
                raise ValueError("continuous candidate program does not bind roots")
        expected_pruned = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED
        )
        if len(self.prune_decisions) != len(expected_pruned) or any(
            canonical_json_bytes(decision.cell_evaluation)
            != canonical_json_bytes(expected)
            for decision, expected in zip(
                self.prune_decisions, expected_pruned, strict=True
            )
        ):
            raise ValueError("continuous prune decisions must be complete")
        expected_unresolved = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED
        )
        if len(self.unresolved_frontier) != len(expected_unresolved) or any(
            canonical_json_bytes(frontier.cell_evaluation)
            != canonical_json_bytes(expected)
            for frontier, expected in zip(
                self.unresolved_frontier, expected_unresolved, strict=True
            )
        ):
            raise ValueError("continuous frontier must be complete")
        if self.coverage_artifact.unresolved_cell_sha256s != tuple(
            sorted(
                row.compiled_cell.compiled_cell_sha256 for row in expected_unresolved
            )
        ):
            raise ValueError("continuous coverage frontier does not bind leaves")
        expected_owner_evaluations = tuple(
            sorted(
                (
                    evaluation
                    for row in self.evaluated_cells
                    for evaluation in row.owner_evaluations
                ),
                key=canonical_json_bytes,
            )
        )
        staged_owner_evaluations = tuple(
            evaluation
            for stage in self.resource_ledger.stage_deltas
            for evaluation in stage.owner_evaluations
        )
        if tuple(sorted(staged_owner_evaluations, key=canonical_json_bytes)) != (
            expected_owner_evaluations
        ):
            raise ValueError("continuous shared ledger must account for every owner")
        return self


UprightSE2MaterializedEndpoint.model_rebuild()
UprightSE2ProofMaterial.model_rebuild()
UprightSE2ContinuousMaterializedEndpoint.model_rebuild()
UprightSE2ContinuousProofMaterial.model_rebuild()


def _bound_state_leaf(subject_id: CanonicalId, role: str) -> StateVariableRef:
    return StateVariableRef(
        state_variable_schema_ref=f"schema:spatialcf/upright-se2/{role}/1.0",
        state_schema_ref=_UPRIGHT_SE2_STATE_SCHEMA_REF,
        fact_family_ref=_UPRIGHT_SE2_STATE_FAMILY_REF,
        entity_or_fact_key=subject_id,
        field_path_ref=f"field-path:upright-se2-{role}",
    )


def _bound_translation_domain(
    problem: CounterfactualProblemIR,
    subject_id: CanonicalId,
) -> UprightSE2TranslationDomain:
    expected = {
        "subject-world-x": _bound_state_leaf(subject_id, "subject-world-x"),
        "subject-world-y": _bound_state_leaf(subject_id, "subject-world-y"),
    }
    bounds = problem.intervention_authorization.variable_bounds
    if len(bounds) != len(expected):
        raise ValueError("source authorization must bind complete world-XY bounds")
    resolved: dict[str, tuple[ExactDyadic, ExactDyadic]] = {}
    for bound in bounds:
        role = next(
            (
                candidate
                for candidate, state_leaf in expected.items()
                if bound.state_variable_ref == state_leaf
            ),
            None,
        )
        if role is None or role in resolved:
            raise ValueError("source authorization bounds must target subject world XY")
        if (
            bound.value_schema_ref,
            bound.frame_ref,
            bound.unit_ref,
            bound.topology_ref,
        ) != (
            _UPRIGHT_SE2_REAL_SCHEMA_REF,
            _UPRIGHT_SE2_WORLD_XY_FRAME_REF,
            _UPRIGHT_SE2_METRE_UNIT_REF,
            _UPRIGHT_SE2_CLOSED_INTERVAL_TOPOLOGY_REF,
        ):
            raise ValueError("source authorization bounds use the wrong type semantics")
        domain = bound.typed_domain
        if (
            domain.value_schema_ref != _UPRIGHT_SE2_REAL_SCHEMA_REF
            or type(domain.payload) is not IntervalValue
            or domain.payload.endpoint_schema_ref != _UPRIGHT_SE2_REAL_SCHEMA_REF
            or type(domain.payload.lower) is not FiniteRealValue
            or type(domain.payload.upper) is not FiniteRealValue
            or not domain.payload.lower_closed
            or not domain.payload.upper_closed
        ):
            raise ValueError(
                "source authorization bounds must be closed finite-real intervals"
            )
        lower = ExactDyadic(
            numerator=Fraction.from_float(domain.payload.lower.value).numerator,
            denominator=Fraction.from_float(domain.payload.lower.value).denominator,
        )
        upper = ExactDyadic(
            numerator=Fraction.from_float(domain.payload.upper.value).numerator,
            denominator=Fraction.from_float(domain.payload.upper.value).denominator,
        )
        resolved[role] = (lower, upper)
    if set(resolved) != set(expected):
        raise ValueError("source authorization must provide both world-XY bounds")
    return UprightSE2TranslationDomain(
        x_lower=resolved["subject-world-x"][0],
        x_upper=resolved["subject-world-x"][1],
        y_lower=resolved["subject-world-y"][0],
        y_upper=resolved["subject-world-y"][1],
    )


def _bound_cardinal_after_yaw(
    yaw_before: CanonicalSO2Angle,
    q: int,
) -> CanonicalSO2Angle:
    turns = (
        Fraction.from_float(yaw_before.turns) + _UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS[q]
    )
    while turns < Fraction(-1, 2):
        turns += 1
    while turns >= Fraction(1, 2):
        turns -= 1
    return CanonicalSO2Angle(turns=0.0 if turns == 0 else float(turns))


def _bound_rotate_cardinal_xy(x: float, y: float, q: int) -> tuple[float, float]:
    if q == 0:
        return (0.0 if x == 0.0 else x, 0.0 if y == 0.0 else y)
    if q == 1:
        return (0.0 if y == 0.0 else -y, 0.0 if x == 0.0 else x)
    if q == 2:
        return (0.0 if x == 0.0 else -x, 0.0 if y == 0.0 else -y)
    return (0.0 if y == 0.0 else y, 0.0 if x == 0.0 else -x)


def _bound_cardinal_quaternion(
    rotation: Quaternion,
    q: int,
    *,
    primary_expected_yaw: CanonicalSO2Angle,
) -> Quaternion:
    delta_z, delta_w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[q]
    return _compose_upright_quaternion_from_primary_yaw(
        rotation=rotation,
        delta_z=delta_z,
        delta_w=delta_w,
        primary_expected_yaw=primary_expected_yaw,
    )


def _bound_cardinal_compiled_cell(
    operation: UprightSE2CardinalOperation,
) -> UprightSE2CompiledCell:
    """Derive the sole exact cardinal cell from an already bound operation."""

    yaw = _UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS[operation.quarter_turns_ccw]
    endpoint = ExactDyadic(numerator=yaw.numerator, denominator=yaw.denominator)
    return UprightSE2CompiledCell.seal(
        cell_id=(
            f"cell:spatialcf/upright-se2/cardinal/{operation.authorization_sha256}"
        ),
        authorization_sha256=operation.authorization_sha256,
        x_lower=operation.translation_domain.x_lower,
        x_upper=operation.translation_domain.x_upper,
        y_lower=operation.translation_domain.y_lower,
        y_upper=operation.translation_domain.y_upper,
        yaw_interval=LiftedYawInterval(
            lower=endpoint,
            upper=endpoint,
            seam_ownership="NONE",
        ),
    )


def _validate_compilation_source_binding(compilation: UprightSE2Compilation) -> None:
    """Recompute the Task 2 transition and closure inputs from the bound request."""

    source_request = compilation.source_solve_request
    problem = source_request.semantic_problem
    source_input = _bound_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    authorization = problem.intervention_authorization
    if (
        authorization.editable_entity_ids != (f"entity:{source_input.subject_id}",)
        or authorization.allowed_operator_refs != (source_input.operator_ref,)
        or authorization.authorized_primary_write_set
        != tuple(
            sorted(
                (
                    _bound_state_leaf(source_input.subject_id, role)
                    for role in _UPRIGHT_SE2_PRIMARY_ROLES
                ),
                key=canonical_json_bytes,
            )
        )
        or authorization.maximum_program_steps != 1
        or authorization.maximum_edited_entities != 1
        or authorization.required_derived_rule_refs != (_UPRIGHT_SE2_DERIVED_RULE_REF,)
        or authorization.complete_state_delta_policy_ref
        != "definition:spatialcf/upright-se2/complete-state-delta/1.0"
    ):
        raise ValueError("compilation source authorization does not close")
    operation = compilation.operation
    expected_pivot_mode = (
        PivotMode.OWN
        if source_input.operator_ref == UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF
        else PivotMode.REFERENCE
    )
    expected_pivot_id = (
        source_input.subject_id
        if expected_pivot_mode is PivotMode.OWN
        else source_input.reference_id
    )
    expected_pivot_state_sha256 = canonical_sha256(
        {
            "scene_state_sha256": problem.scene_state.scene_state_sha256,
            "pivot_entity_id": expected_pivot_id,
            "object_pivot_pose": (
                subject.pose if expected_pivot_mode is PivotMode.OWN else reference.pose
            ),
        },
        domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
    )
    if (
        source_input.operator_ref
        not in (
            UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
        )
        or source_input.subject_id == source_input.reference_id
        or operation.subject_id != source_input.subject_id
        or operation.authorization.operator_ref != source_input.operator_ref
        or operation.quarter_turns_ccw != source_input.quarter_turns_ccw
        or operation.pivot_binding.pivot_mode is not expected_pivot_mode
        or operation.pivot_binding.pivot_entity_id != expected_pivot_id
        or operation.pivot_binding.pivot_state_sha256 != expected_pivot_state_sha256
        or operation.translation_domain
        != _bound_translation_domain(problem, source_input.subject_id)
    ):
        raise ValueError("compilation operation does not bind the source request")
    subject_pose = subject.pose.world_from_object
    reference_pose = reference.pose.world_from_object
    validate_directed_yaw_quaternion_consistency(
        source_input.subject_yaw_turns,
        subject_pose.rotation,
    )
    expected_recipe = UprightSE2EndpointConstructionRecipe.seal(
        source_scene_state_sha256=problem.scene_state.scene_state_sha256,
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        pivot_binding=operation.pivot_binding,
        quarter_turns_ccw=source_input.quarter_turns_ccw,
        translation_domain=operation.translation_domain,
        subject_before_pose=subject_pose,
        reference_before_pose=reference_pose,
        subject_yaw_turns=source_input.subject_yaw_turns,
        evaluation_scene=scene,
    )
    if canonical_json_bytes(
        compilation.endpoint_construction_recipe
    ) != canonical_json_bytes(expected_recipe):
        raise ValueError(
            "compilation endpoint recipe does not derive from the source request"
        )
    leaf_index = problem.scene_state.canonical_state_leaf_index
    footprint = compilation.state_footprint
    expected_primary_writes = tuple(
        sorted(
            (
                _bound_state_leaf(source_input.subject_id, role)
                for role in _UPRIGHT_SE2_PRIMARY_ROLES
            ),
            key=canonical_json_bytes,
        )
    )
    expected_derived_writes = tuple(
        sorted(
            (
                _bound_state_leaf(source_input.subject_id, role)
                for role in _UPRIGHT_SE2_DERIVED_ROLES
            ),
            key=canonical_json_bytes,
        )
    )
    expected_written_bytes = {
        *(canonical_json_bytes(leaf) for leaf in expected_primary_writes),
        *(canonical_json_bytes(leaf) for leaf in expected_derived_writes),
    }
    expected_frozen_leaf_refs = tuple(
        leaf
        for leaf in leaf_index.leaves
        if canonical_json_bytes(leaf) not in expected_written_bytes
    )
    manifest = footprint.state_delta_manifest
    partition = tuple(
        sorted(
            (
                *footprint.state_delta_manifest.authorized_primary_writes,
                *footprint.state_delta_manifest.recomputed_derived_writes,
                *footprint.frozen_leaf_refs,
            ),
            key=canonical_json_bytes,
        )
    )
    if (
        partition != leaf_index.leaves
        or manifest.authorized_primary_writes != expected_primary_writes
        or manifest.recomputed_derived_writes != expected_derived_writes
        or footprint.frozen_leaf_refs != expected_frozen_leaf_refs
        or manifest.unchanged_leaves_digest
        != canonical_sha256(
            expected_frozen_leaf_refs,
            domain=_UPRIGHT_SE2_UNCHANGED_LEAVES_HASH_DOMAIN,
        )
        or footprint.complete_before_leaf_index_sha256
        != leaf_index.state_leaf_index_sha256
        or footprint.complete_after_leaf_index_sha256
        != leaf_index.state_leaf_index_sha256
    ):
        raise ValueError(
            "compilation state footprint does not bind the source leaf index"
        )
    expected_cell = _bound_cardinal_compiled_cell(operation)
    if len(compilation.compiled_cells) != 1 or canonical_json_bytes(
        compilation.compiled_cells[0]
    ) != canonical_json_bytes(expected_cell):
        raise ValueError("compiled cells do not derive from the source-bound operation")
    if (
        compilation.closure.definition_bundle_sha256
        != problem.definition_bundle.definition_bundle_sha256
        or compilation.closure.solve_policy_definition_bundle_sha256
        != source_request.solve_policy_definition_bundle.definition_bundle_sha256
        or dict(
            source_request.implementation_registry_snapshot.implementation_build_hashes
        ).get(UPRIGHT_SE2_COMPILER_OWNER_REF)
        != compilation.closure.compiler_build_sha256
    ):
        raise ValueError("compiler closure does not bind the source request")


def _validate_continuous_compilation_source_binding(
    compilation: UprightSE2ContinuousCompilation,
) -> None:
    """Replay continuous operation and recipe roots from the bound request."""

    source_request = compilation.source_solve_request
    problem = source_request.semantic_problem
    source_input = _bound_continuous_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    expected_pivot_mode = (
        PivotMode.OWN
        if source_input.operator_ref == UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF
        else PivotMode.REFERENCE
    )
    expected_pivot_id = (
        source_input.subject_id
        if expected_pivot_mode is PivotMode.OWN
        else source_input.reference_id
    )
    expected_pivot_pose = (
        subject.pose if expected_pivot_mode is PivotMode.OWN else reference.pose
    )
    expected_pivot = FixedPivotBinding.seal(
        subject_id=source_input.subject_id,
        pivot_mode=expected_pivot_mode,
        pivot_entity_id=expected_pivot_id,
        pivot_state_sha256=canonical_sha256(
            {
                "scene_state_sha256": problem.scene_state.scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": expected_pivot_pose,
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        ),
    )
    expected_authorization = ContinuousYawAuthorization.seal(
        subject_id=source_input.subject_id,
        operator_ref=source_input.operator_ref,
        pivot_binding=expected_pivot,
        yaw_domain=source_input.yaw_domain,
    )
    operation = compilation.operation
    if (
        source_input.operator_ref
        not in (
            UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
        )
        or source_input.subject_id == source_input.reference_id
        or operation.subject_id != source_input.subject_id
        or canonical_json_bytes(operation.authorization)
        != canonical_json_bytes(expected_authorization)
        or operation.pivot_binding != expected_pivot
        or operation.yaw_domain != source_input.yaw_domain
        or operation.translation_domain
        != _bound_translation_domain(problem, source_input.subject_id)
    ):
        raise ValueError(
            "continuous compilation operation does not bind source request"
        )
    subject_pose = subject.pose.world_from_object
    reference_pose = reference.pose.world_from_object
    validate_directed_yaw_quaternion_consistency(
        source_input.subject_yaw_turns,
        subject_pose.rotation,
    )
    expected_recipe = UprightSE2ContinuousEndpointConstructionRecipe.seal(
        source_scene_state_sha256=problem.scene_state.scene_state_sha256,
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        pivot_binding=expected_pivot,
        yaw_domain=source_input.yaw_domain,
        translation_domain=operation.translation_domain,
        subject_before_pose=subject_pose,
        reference_before_pose=reference_pose,
        subject_yaw_turns=source_input.subject_yaw_turns,
        evaluation_scene=scene,
    )
    if canonical_json_bytes(
        compilation.endpoint_construction_recipe
    ) != canonical_json_bytes(expected_recipe):
        raise ValueError(
            "continuous compilation endpoint recipe does not derive from source request"
        )


# ``UprightSE2ProofMaterial`` is deliberately declared before the compilation it
# transports so the existing verification bundle keeps its public placement.
# Resolve that forward reference only after the complete source-bound compiler
# record is available.
UprightSE2CardinalProofTuple.model_rebuild()
UprightSE2ProofMaterial.model_rebuild()
UprightSE2VerificationBundle.model_rebuild()
