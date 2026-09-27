"""Upright compiler evaluation; explicit pure implementation owner."""

from __future__ import annotations

from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _require_exact_round_trip,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _registered_profile,
    _resolve_pivot_binding,
    _validate_explicit_pose_yaw,
    _validate_intervention_authorization,
    _validate_operational_closure,
    _validate_problem_and_extract_input,
)

from spatialcf.core._internal.upright_se2.compilation import (
    compile_upright_se2,
)

from spatialcf.core._internal.upright_se2.evaluation_data import (
    UprightSE2CardinalEvaluationInputs,
    UprightSE2ContinuousEvaluationInputs,
)

from spatialcf.core._internal.upright_se2.evaluation_geometry import (
    _bridge_collision_boxes,
    _bridge_compiled_cell_member,
    _bridge_continuous_compiled_cell_member,
    _bridge_exact_identity_camera,
    _bridge_object_map,
    _bridge_object_pivot_xy,
    _bridge_reference_relation_box,
    _bridge_require_replayed_compilation,
    _bridge_require_replayed_continuous_compilation,
    _bridge_support_surface,
    _bridge_target_and_preservation_rows,
    _bridge_validate_cell_yaw,
    _bridge_validate_subject_role_geometry_closure,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _bridge_cell_policy,
    _bridge_resource_cap,
)

from spatialcf.core._internal.upright_se2.visibility import (
    _bridge_continuous_visibility_inputs,
    _bridge_visibility_inputs,
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


# Preserve supported public type/function and pickle lookup.
build_upright_se2_cardinal_evaluation_inputs.__module__ = "spatialcf.core.upright_se2_compiler"
build_upright_se2_continuous_evaluation_inputs.__module__ = "spatialcf.core.upright_se2_compiler"
