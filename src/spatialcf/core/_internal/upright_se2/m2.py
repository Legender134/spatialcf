"""Upright compiler m2; explicit pure implementation owner."""

from __future__ import annotations

import math

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.compatibility import (
    PlanarTranslateCompilation,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    ExtensionFact,
    ExtensionFactBundle,
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    CanonicalIdValue,
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

from spatialcf.domain.operators import (
    StateLeafIndex,
    TypedVariableBound,
)

from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    ObservationObligation,
    PredicateAtom,
    PreservationInvariant,
)

from spatialcf.domain.profiles import (
    InterventionAuthorization,
    ResourcePolicy,
)

from spatialcf.domain.scene import (
    CanonicalScene,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _canonical_zero,
    _dyadic_from_float,
    _require_exact_round_trip,
    _sorted_bytes,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _closed_entity_index,
    _expected_state_leaves,
    _input_entity_id,
    _primary_write_set,
    _registered_profile,
    _state_leaf,
)

from spatialcf.core._internal.upright_se2.compilation import (
    compile_upright_se2,
)

from spatialcf.core._internal.upright_se2.constants import (
    _BACKEND_BUILD_SHA256,
    _CHECKER_BUILD_SHA256,
    _CLOSED_INTERVAL_TOPOLOGY_REF,
    _COMPILER_INPUT_FACT_KEY,
    _DERIVED_RULE_REF,
    _DIGEST_SCHEMA_REF,
    _ENUM_SCHEMA_REF,
    _ID_SCHEMA_REF,
    _INPUT_FAMILY_REF,
    _INPUT_SCHEMA_REF,
    _INTEGER_SCHEMA_REF,
    _METRE_UNIT_REF,
    _REAL_SCHEMA_REF,
    _SCENE_SCHEMA_REF,
    _WORLD_XY_FRAME_REF,
    _YAW_ARGUMENT_SCHEMA_REF,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _backend_descriptor_bundle,
    _backend_routing_policy,
    _implementation_registry,
    _proof_policy,
    _resource_policy,
    _solve_policy_bundle,
    _solver_config,
    _source_requested_gap,
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


# Preserve supported public type/function and pickle lookup.
compile_planar_translate_m2_q0_equivalence.__module__ = "spatialcf.core.upright_se2_compiler"
_canonical_scene_from_m2_source.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_yaw_transform_to_rigid.__module__ = "spatialcf.core.upright_se2_compiler"
_rigid_from_yaw.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_supported_translation_domain.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_policy_bundle.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_policy_definition_ref.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_policy_owner.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_policy_payload.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_real.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_integer.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_id.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_digest.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_symbol.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_tuple.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_record.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_typed_record.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_construction.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_source_leaves.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_domain_value.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_int.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_source_request.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_scene_object.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_variable_bounds.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_closed_interval.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_typed_reference.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_target_relation_atom.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_preservation_atom.__module__ = "spatialcf.core.upright_se2_compiler"
_m2_q0_visibility_obligation.__module__ = "spatialcf.core.upright_se2_compiler"
_reseal_q0_construction.__module__ = "spatialcf.core.upright_se2_compiler"
