"""Upright compiler compilation; explicit pure implementation owner."""

from __future__ import annotations

from spatialcf.core._internal.kernels.so2 import (
    ContinuousYawIntervalKindV4,
    SO2AtomicBudgetV2,
    compile_continuous_yaw_lift_v4,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
)

from spatialcf.domain.operators import (
    StateDeltaManifest,
)

from spatialcf.domain.outcomes import (
    TypedCompilationOutcome,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _dyadic_from_fraction,
    _require_exact_round_trip,
    cardinal_inverse_quarter_turns,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _CompilerInput,
    _SceneAuthority,
    _derived_write_set,
    _grounded_obligations,
    _primary_write_set,
    _registered_profile,
    _resolve_pivot_binding,
    _validate_explicit_pose_yaw,
    _validate_intervention_authorization,
    _validate_operational_closure,
    _validate_problem_and_extract_input,
)

from spatialcf.core._internal.upright_se2.constants import (
    _CARDINAL_TURN_FRACTIONS,
    _UNCHANGED_LEAVES_HASH_DOMAIN,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _bridge_resource_cap,
)


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


# Preserve supported public type/function and pickle lookup.
compile_upright_se2.__module__ = "spatialcf.core.upright_se2_compiler"
_compile_cardinal.__module__ = "spatialcf.core.upright_se2_compiler"
compile_upright_se2_continuous.__module__ = "spatialcf.core.upright_se2_compiler"
_compile_continuous.__module__ = "spatialcf.core.upright_se2_compiler"
_endpoint_construction_recipe.__module__ = "spatialcf.core.upright_se2_compiler"
_state_footprint.__module__ = "spatialcf.core.upright_se2_compiler"
_compiled_cell.__module__ = "spatialcf.core.upright_se2_compiler"
