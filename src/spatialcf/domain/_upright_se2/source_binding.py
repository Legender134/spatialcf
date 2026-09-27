"""Upright profile source binding contracts and intrinsic operations."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)

from typing import (
    TypeAlias,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
)

from spatialcf.domain.definitions import (
    CanonicalIdValue,
    DefinitionRef,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    IntegerValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueKind,
)

from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryInstanceV2,
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

from spatialcf.domain.scene import (
    BaselineObservation,
    CanonicalScene,
    SupportSurfaceFact,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
    UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
    UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
    _UPRIGHT_SE2_COMPILER_INPUT_FACT_KEY,
    _UPRIGHT_SE2_COMPILER_INPUT_FAMILY_REF,
    _UPRIGHT_SE2_COMPILER_INPUT_SCHEMA_REF,
    _UPRIGHT_SE2_ENUM_SCHEMA_REF,
    _UPRIGHT_SE2_ID_SCHEMA_REF,
    _UPRIGHT_SE2_INTEGER_SCHEMA_REF,
    _UPRIGHT_SE2_REAL_SCHEMA_REF,
    _UPRIGHT_SE2_YAW_ARGUMENT_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.policies import (
    UprightSE2ExecutablePolicyBundle,
    _policy_id,
    _policy_payload_fields,
    _policy_real,
    _policy_symbol,
    _validate_visibility_observation_bounds,
    _visibility_observation_bound_fields,
    request_bound_executable_policy_bundle_from_problem,
)

from spatialcf.domain._upright_se2.yaw import (
    CanonicalSO2Angle,
    CardinalYaw,
    ContinuousYawArc,
    ContinuousYawDomain,
    ContinuousYawFullCircle,
)


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


# Keep supported public import and pickle lookup stable.
_UprightSE2SourceInput.__module__ = "spatialcf.domain.upright_se2"
_UprightSE2ContinuousSourceInput.__module__ = "spatialcf.domain.upright_se2"
validate_upright_se2_executable_policy_visibility_binding.__module__ = "spatialcf.domain.upright_se2"
_predicate_atom_from_context.__module__ = "spatialcf.domain.upright_se2"
_validate_grounded_semantic_shape.__module__ = "spatialcf.domain.upright_se2"
_bound_source_input.__module__ = "spatialcf.domain.upright_se2"
_bound_continuous_source_input.__module__ = "spatialcf.domain.upright_se2"
_bound_real.__module__ = "spatialcf.domain.upright_se2"
_bound_integer.__module__ = "spatialcf.domain.upright_se2"
_bound_id.__module__ = "spatialcf.domain.upright_se2"
_bound_symbol.__module__ = "spatialcf.domain.upright_se2"
_bound_scene_object.__module__ = "spatialcf.domain.upright_se2"
_bound_predicate_atom.__module__ = "spatialcf.domain.upright_se2"
_bound_reference.__module__ = "spatialcf.domain.upright_se2"
_validate_bound_target_atom.__module__ = "spatialcf.domain.upright_se2"
_validate_bound_preservation_atom.__module__ = "spatialcf.domain.upright_se2"
_bound_visibility_observation_id.__module__ = "spatialcf.domain.upright_se2"
_validate_bound_semantic_problem.__module__ = "spatialcf.domain.upright_se2"
_bound_grounded_obligations.__module__ = "spatialcf.domain.upright_se2"
_validate_bound_continuous_semantic_problem.__module__ = "spatialcf.domain.upright_se2"
_bound_continuous_grounded_obligations.__module__ = "spatialcf.domain.upright_se2"
_derived_source_identifier.__module__ = "spatialcf.domain.upright_se2"
_exact_source_values.__module__ = "spatialcf.domain.upright_se2"
