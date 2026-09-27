"""Upright profile endpoints contracts and intrinsic operations."""

from __future__ import annotations

import math

from fractions import (
    Fraction,
)

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    RigidTransformV2,
    Sha256Digest,
    Vec2,
    Vec3,
)

from spatialcf.domain.counterfactual import (
    EditProgram,
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
    HashBoundCanonicalModel,
    IntegerValue,
    TypedValue,
)

from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
)

from spatialcf.domain.scene import (
    ObjectPose,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.cells import (
    UprightSE2ProofLeafDisposition,
)

from spatialcf.domain._upright_se2.compilation import (
    UprightSE2Compilation,
    UprightSE2ContinuousCompilation,
    _bound_cardinal_after_yaw,
    _bound_cardinal_quaternion,
    _bound_rotate_cardinal_xy,
)

from spatialcf.domain._upright_se2.constants import (
    _UPRIGHT_SE2_DERIVED_ROLES,
    _UPRIGHT_SE2_ID_SCHEMA_REF,
    _UPRIGHT_SE2_PRIMARY_ROLES,
    _UPRIGHT_SE2_REAL_SCHEMA_REF,
    _UPRIGHT_SE2_STATE_FAMILY_REF,
)

from spatialcf.domain._upright_se2.proof_evaluation import (
    UprightSE2ProofCellEvaluation,
    UprightSE2ProposalPointEvaluation,
    UprightSE2ProposalPointObjective,
)

from spatialcf.domain._upright_se2.state import (
    UprightSE2AfterStateTemplate,
    UprightSE2ContinuousEndpointConstructionRecipe,
    UprightSE2DerivedAfterFact,
    UprightSE2EndpointConstructionRecipe,
)

from spatialcf.domain._upright_se2.values import (
    _materialized_state_record,
)

from spatialcf.domain._upright_se2.yaw import (
    CanonicalSO2Angle,
    ContinuousYawFullCircle,
    ExactDyadic,
    PivotMode,
    _compose_upright_quaternion_from_primary_yaw,
    _fraction_from_float,
)


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


# Resolve local model forward references before restoring public identities.
UprightSE2MaterializedEndpoint.model_rebuild()
UprightSE2ProposalCandidate.model_rebuild()
UprightSE2ContinuousMaterializedEndpoint.model_rebuild()
UprightSE2ContinuousProposalCandidate.model_rebuild()


# Keep supported public import and pickle lookup stable.
_continuous_canonical_yaw_after.__module__ = "spatialcf.domain.upright_se2"
_continuous_materialized_expected_pose.__module__ = "spatialcf.domain.upright_se2"
UprightSE2MaterializedEndpoint.__module__ = "spatialcf.domain.upright_se2"
_materialized_program_arguments.__module__ = "spatialcf.domain.upright_se2"
_materialized_derived_bundle.__module__ = "spatialcf.domain.upright_se2"
_materialized_state_real.__module__ = "spatialcf.domain.upright_se2"
_materialized_pose_value.__module__ = "spatialcf.domain.upright_se2"
_materialized_derived_fact_value.__module__ = "spatialcf.domain.upright_se2"
_materialized_derived_fact_roster_value.__module__ = "spatialcf.domain.upright_se2"
_materialized_expected_after_scene_state.__module__ = "spatialcf.domain.upright_se2"
_validate_materialized_endpoint_program.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProposalCandidate.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousMaterializedEndpoint.__module__ = "spatialcf.domain.upright_se2"
_continuous_materialized_program_arguments.__module__ = "spatialcf.domain.upright_se2"
_validate_continuous_materialized_endpoint_program.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousProposalCandidate.__module__ = "spatialcf.domain.upright_se2"
