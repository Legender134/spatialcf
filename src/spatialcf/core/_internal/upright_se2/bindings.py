"""Upright compiler bindings; explicit pure implementation owner."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    ExtensionFact,
)

from spatialcf.domain.definitions import (
    CanonicalIdValue,
    EnumSymbolValue,
    FiniteRealValue,
    IntegerValue,
    IntervalValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueKind,
)

from spatialcf.domain.operators import (
    StateVariableRef,
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
    ImplementationRegistrySnapshot,
    SemanticsProfile,
)

from spatialcf.domain.scene import (
    CanonicalScene,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _dyadic_from_float,
    _sorted_bytes,
)

from spatialcf.core._internal.upright_se2.constants import (
    _CLOSED_INTERVAL_TOPOLOGY_REF,
    _COMPILER_INPUT_FACT_KEY,
    _DERIVED_ROLES,
    _DERIVED_RULE_REF,
    _ENUM_SCHEMA_REF,
    _ID_SCHEMA_REF,
    _INPUT_FAMILY_REF,
    _INPUT_SCHEMA_REF,
    _INTEGER_SCHEMA_REF,
    _METRE_UNIT_REF,
    _PIVOT_STATE_HASH_DOMAIN,
    _POSE_STATE_HASH_DOMAIN,
    _PRIMARY_ROLES,
    _REAL_SCHEMA_REF,
    _SCENE_SCHEMA_REF,
    _SOURCE_FIELDS,
    _STATE_FAMILY_REF,
    _STATE_SCHEMA_REF,
    _WORLD_XY_FRAME_REF,
    _YAW_ARGUMENT_SCHEMA_REF,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _backend_descriptor_bundle,
    _backend_routing_policy,
    _implementation_registry,
    _proof_policy,
    _solver_config,
    _validate_request_resource_policy,
)


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


def _primary_write_set(subject_id: str) -> tuple[StateVariableRef, ...]:
    return _sorted_bytes(*(_state_leaf(subject_id, role) for role in _PRIMARY_ROLES))


def _derived_write_set(subject_id: str) -> tuple[StateVariableRef, ...]:
    return _sorted_bytes(*(_state_leaf(subject_id, role) for role in _DERIVED_ROLES))


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


# Preserve supported public type/function and pickle lookup.
_CompilerInput.__module__ = "spatialcf.core.upright_se2_compiler"
_SceneAuthority.__module__ = "spatialcf.core.upright_se2_compiler"
_registered_profile.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_problem_and_extract_input.__module__ = "spatialcf.core.upright_se2_compiler"
_compiler_input_from_scene.__module__ = "spatialcf.core.upright_se2_compiler"
_input_entity_id.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_compiler_input_fact_identity.__module__ = "spatialcf.core.upright_se2_compiler"
_real_field.__module__ = "spatialcf.core.upright_se2_compiler"
_integer_field.__module__ = "spatialcf.core.upright_se2_compiler"
_id_field.__module__ = "spatialcf.core.upright_se2_compiler"
_symbol_field.__module__ = "spatialcf.core.upright_se2_compiler"
_yaw_argument_field.__module__ = "spatialcf.core.upright_se2_compiler"
_resolve_scene_authority.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_required_exact_scene_sources.__module__ = "spatialcf.core.upright_se2_compiler"
_known_exact_source_values.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_complete_state_leaves.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_grounded_semantic_operands.__module__ = "spatialcf.core.upright_se2_compiler"
_direct_predicate_atom.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_target_relation_operands.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_preservation_operands.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_visibility_operands.__module__ = "spatialcf.core.upright_se2_compiler"
_require_reference_operand.__module__ = "spatialcf.core.upright_se2_compiler"
_require_symbol_operand.__module__ = "spatialcf.core.upright_se2_compiler"
_require_id_operand.__module__ = "spatialcf.core.upright_se2_compiler"
_expected_state_leaves.__module__ = "spatialcf.core.upright_se2_compiler"
_closed_entity_index.__module__ = "spatialcf.core.upright_se2_compiler"
_fact_ids.__module__ = "spatialcf.core.upright_se2_compiler"
_state_leaf.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_operational_closure.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_registry.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_intervention_authorization.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_explicit_pose_yaw.__module__ = "spatialcf.core.upright_se2_compiler"
_resolve_pivot_binding.__module__ = "spatialcf.core.upright_se2_compiler"
_validate_continuous_request.__module__ = "spatialcf.core.upright_se2_compiler"
_translation_domain_from_authorization.__module__ = "spatialcf.core.upright_se2_compiler"
_primary_write_set.__module__ = "spatialcf.core.upright_se2_compiler"
_derived_write_set.__module__ = "spatialcf.core.upright_se2_compiler"
_grounded_obligations.__module__ = "spatialcf.core.upright_se2_compiler"
