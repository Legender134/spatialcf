"""Upright compiler materialization; explicit pure implementation owner."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    Quaternion,
    RigidTransformV2,
    Vec2,
    Vec3,
)

from spatialcf.domain.counterfactual import (
    EditProgram,
)

from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
)

from spatialcf.domain.scene import (
    CanonicalScene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _canonical_zero,
    _sorted_bytes,
    rotate_cardinal_xy,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _known_exact_source_values,
)

from spatialcf.core._internal.upright_se2.constants import (
    _CARDINAL_TURN_FRACTIONS,
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


# Preserve supported public type/function and pickle lookup.
materialize_upright_se2_endpoint.__module__ = "spatialcf.core.upright_se2_compiler"
materialize_upright_se2_continuous_endpoint.__module__ = "spatialcf.core.upright_se2_compiler"
_continuous_materialized_program_arguments.__module__ = "spatialcf.core.upright_se2_compiler"
_continuous_after_state_template.__module__ = "spatialcf.core.upright_se2_compiler"
_materialized_program.__module__ = "spatialcf.core.upright_se2_compiler"
_after_state_template.__module__ = "spatialcf.core.upright_se2_compiler"
_canonical_yaw_after.__module__ = "spatialcf.core.upright_se2_compiler"
_apply_cardinal_quaternion.__module__ = "spatialcf.core.upright_se2_compiler"
_derived_after_facts.__module__ = "spatialcf.core.upright_se2_compiler"
_yaw_then_translate.__module__ = "spatialcf.core.upright_se2_compiler"
