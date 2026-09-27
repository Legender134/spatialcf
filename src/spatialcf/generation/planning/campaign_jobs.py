"""Explicit source campaign jobs owner."""

from __future__ import annotations

import hashlib

import warnings

from collections import (
    Counter,
)

from collections.abc import (
    Callable,
)

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from dataclasses import (
    dataclass,
)

from multiprocessing import (
    get_context,
)

from multiprocessing.connection import (
    Connection,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.request import (
    InterventionSpec,
    Relation,
)

from spatialcf.domain.scene import (
    Scene,
)

from spatialcf.generation.capture.compiler import (
    compile_roster,
    verify_competition_native_camera_evidence_capture_v2_9_3,
)

from spatialcf.generation.capture.models import (
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportKindV2_9,
    RosterCompilation,
    SourceCameraEvidence,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    verify_source_surface_evidence,
)

from spatialcf.generation.planning.endpoint import (
    EndpointPlanRejected,
    endpoint_workspace_within_position_region,
    plan_endpoint,
)

from spatialcf.generation.planning.models import (
    CollisionDelegation,
    EndpointPlan,
    EndpointWorkspace,
    SubjectPlacementFact,
)

from spatialcf.generation.planning.problem import (
    build_proxy_bundle,
    default_planning_workspace,
    default_solver_config,
)

from spatialcf.generation.planning.view_guard import (
    evaluate_source_view_guard,
)

from spatialcf.generation.planning.campaign_contracts import (
    BatchManifest,
    BatchRequest,
    RuntimePosePolicy,
    SourcePlan,
    SourcePolicy,
    SourceRequestOutcome,
    SourceSlotEntry,
    SourceSlotOutcome,
    _MAX_REASON_CHARS,
    _RELATIONS,
    _SCREENING_DOMAIN_OPERATIONS,
    _accepted_source_capture_roster_sha256,
    _case_id,
    _fresh_solve_result,
    _legacy_sha256,
    _manifest_file_sha256,
    _runtime_identity_sha256,
    _target_reachability_ledger_sha256,
    _workspace_is_subset,
)


def build_default_source_policy(compilation: RosterCompilation) -> SourcePolicy:
    if type(compilation) is not RosterCompilation:
        raise TypeError("candidate roster compilation must be exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked = RosterCompilation.model_validate(
            compilation.model_dump(mode="python", warnings="error"), strict=True
        )
        accepted_captures = tuple(
            item.capture
            for item in checked.scene_inventory
            if item.status == "accepted" and item.capture is not None
        )
        return SourcePolicy(
            campaign_id=checked.policy.campaign_id,
            roster_manifest_sha256=checked.request_manifest.manifest_sha256,
            width=checked.policy.width,
            height=checked.policy.height,
            seed=checked.policy.seed,
            candidate_wall_time_seconds=60,
            solver_config=default_solver_config(),
            camera_policy=checked.policy.camera_policy,
            accepted_source_capture_roster_sha256=(
                _accepted_source_capture_roster_sha256(accepted_captures)
            ),
            runtime_pose_policy=RuntimePosePolicy(),
            roster_policy_sha256=checked.policy.policy_sha256,
            target_reachability_ledger_sha256=(
                _target_reachability_ledger_sha256(checked.target_reachability)
            ),
        )


def _endpoint_rejection_reason(reason: str) -> str:
    if reason in {
        "endpoint_plan:SOURCE_VIEW_MISSING",
        "endpoint_plan:SOURCE_VIEW_UNCERTIFIED",
    }:
        return reason
    candidate = f"source_plan:endpoint:{reason}"
    if len(candidate) <= _MAX_REASON_CHARS:
        return candidate
    digest = hashlib.sha256(reason.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"source_plan:endpoint_reason_sha256:{digest}"


def _endpoint_rejection_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted({_endpoint_rejection_reason(item) for item in reasons}))
    if len(normalized) <= 32:
        return normalized
    payload = b"\0".join(item.encode("utf-8") for item in normalized)
    return (
        f"source_plan:endpoint_reasons_sha256:{hashlib.sha256(payload).hexdigest()}",
    )


def _rejected_outcome(
    request: CompetitionNativeSelectedRequestV2_9,
    slot: int,
    case_id: str,
    reasons: str | tuple[str, ...],
    *,
    capture: CompetitionNativeSourceCaptureV2_9,
    placement: CompetitionNativeSubjectPlacementFactV2_9,
) -> SourceRequestOutcome:
    if type(reasons) is str:
        reasons = (reasons,)
    return SourceRequestOutcome(
        request_id=request.request_id,
        candidate_id=request.candidate_id,
        selection_index=request.selection_index,
        relation_before=request.relation_before,
        relation_slot_index=slot,
        status="rejected",
        source_id=request.source_id,
        source_locator_sha256=request.source_locator_sha256,
        source_capture_sha256=request.source_capture_sha256,
        source_scene_sha256=_legacy_sha256(capture.scene),
        runtime_identity_sha256=_runtime_identity_sha256(capture),
        scene_id=request.scene_id,
        subject_id=request.subject_id,
        subject_name=request.subject_name,
        reference_id=request.reference_id,
        reference_name=request.reference_name,
        support_kind=request.support_kind,
        placement_sha256=placement.placement_sha256,
        case_id=case_id,
        planning_workspace=None,
        endpoint_workspace=None,
        endpoint_candidate_index=None,
        attempted_workspace_count=None,
        candidate_point_count=None,
        solve_result_sha256=None,
        selected_edit_sha256=None,
        source_view_fact_sha256=None,
        source_view_guard=None,
        reasons=tuple(sorted(set(reasons))),
    )


def _endpoint_policy_controls(policy: SourcePolicy) -> tuple[int, int, int]:
    """Freeze the current campaign-to-endpoint optional policy seam."""

    if type(policy) is not SourcePolicy:
        raise TypeError("native source policy must be exact")
    return (
        policy.max_endpoint_candidate_points,
        policy.max_included_collision_obstacles,
        _SCREENING_DOMAIN_OPERATIONS,
    )


def _preflight_request(
    policy: SourcePolicy,
    request: CompetitionNativeSelectedRequestV2_9,
    capture: CompetitionNativeSourceCaptureV2_9,
    placement: CompetitionNativeSubjectPlacementFactV2_9,
    case_id: str,
) -> str | None:
    _endpoint_policy_controls(policy)
    if capture.source_view_fact is None:
        return "endpoint_plan:SOURCE_VIEW_MISSING"
    runtime = capture.runtime_identity
    if runtime.native_scene_name == "Procedural":
        return "source_plan:procedural_native_audit_unsupported"
    if not capture.is_scene_at_rest:
        return "source_plan:source_scene_not_at_rest"
    if (runtime.width, runtime.height, runtime.seed) != (
        policy.width,
        policy.height,
        policy.seed,
    ):
        return "source_plan:render_policy_mismatch"
    if placement.support_kind is not request.support_kind:
        return "source_plan:placement_lineage_mismatch"
    if request.support_kind is CompetitionNativeSupportKindV2_9.FLOOR:
        return "source_plan:floor_native_audit_unsupported"
    subject = capture.scene.object_by_id(request.subject_id)
    if (
        request.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
        or placement.availability
        is not CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        or placement.support_object_id is None
        or not placement.native_positions
        or subject.support_object_id != placement.support_object_id
    ):
        return "source_plan:placement_not_executable"
    region = placement.position_region
    if (
        region is None
        or not region.components
        or region.subject_object_id != request.subject_id
        or region.source_kind != "ai2thor-receptacle-trigger-grid-v1"
    ):
        return "source_plan:receptacle_position_region_missing"
    try:
        BatchRequest(
            request_id=request.request_id,
            case_id=case_id,
            scene_id=request.scene_id,
            subject_name=request.subject_name,
            reference_name=request.reference_name,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            endpoint_workspace=EndpointWorkspace(
                min_x_m=0.0, min_y_m=0.0, max_x_m=1.0, max_y_m=1.0
            ),
            solver_config=policy.solver_config,
            max_settlement_steps=policy.max_settlement_steps,
        )
    except (TypeError, ValueError):
        return "source_plan:batch_contract_unsupported"
    return None


@dataclass(frozen=True, slots=True)
class _PlanningJob:
    manifest_index: int
    request: CompetitionNativeSelectedRequestV2_9
    slot: int
    case_id: str
    capture: CompetitionNativeSourceCaptureV2_9
    placement_fact: CompetitionNativeSubjectPlacementFactV2_9
    source_surface_evidence: SourceSurfaceEvidence
    subject_surface_evidence: SubjectSurfaceEvidence
    scene: Scene
    intervention: InterventionSpec
    workspace: EndpointWorkspace
    placement: SubjectPlacementFact
    camera_evidence_sha256: Sha256Digest
    max_candidate_points: int
    max_included_collision_obstacles: int
    screening_domain_operations: int


def _prepare_job(
    policy: SourcePolicy,
    request: CompetitionNativeSelectedRequestV2_9,
    *,
    manifest_index: int,
    slot: int,
    capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    camera_evidence_sha256: Sha256Digest,
) -> _PlanningJob | SourceRequestOutcome:
    case_id = _case_id(policy, request)
    placements = tuple(
        item for item in capture.placement_facts if item.object_id == request.subject_id
    )
    if len(placements) != 1:
        raise ValueError("fresh roster selected a non-unique placement fact")
    placement_fact = placements[0]
    if capture.source_view_fact is None:
        return _rejected_outcome(
            request,
            slot,
            case_id,
            "endpoint_plan:SOURCE_VIEW_MISSING",
            capture=capture,
            placement=placement_fact,
        )
    verified_evidence = verify_source_surface_evidence(capture, source_surface_evidence)
    subjects = tuple(
        item
        for item in verified_evidence.subjects
        if item.subject_object_id == request.subject_id
    )
    if len(subjects) != 1:
        return _rejected_outcome(
            request,
            slot,
            case_id,
            "source_plan:subject_surface_evidence_missing",
            capture=capture,
            placement=placement_fact,
        )
    reason = _preflight_request(policy, request, capture, placement_fact, case_id)
    if reason is not None:
        return _rejected_outcome(
            request,
            slot,
            case_id,
            reason,
            capture=capture,
            placement=placement_fact,
        )
    try:
        intervention = InterventionSpec(
            subject_id=request.subject_id,
            reference_id=request.reference_id,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            camera_id=request.camera_id,
        )
        workspace = default_planning_workspace(capture.scene, request.subject_id)
        placement = SubjectPlacementFact(
            source_capture=capture, subject_placement=placement_fact
        )
    except Exception as error:  # noqa: BLE001
        return _rejected_outcome(
            request,
            slot,
            case_id,
            f"source_plan:workspace:{type(error).__name__}",
            capture=capture,
            placement=placement_fact,
        )
    (
        max_candidate_points,
        max_included_collision_obstacles,
        screening_domain_operations,
    ) = _endpoint_policy_controls(policy)
    return _PlanningJob(
        manifest_index=manifest_index,
        request=request,
        slot=slot,
        case_id=case_id,
        capture=capture,
        placement_fact=placement_fact,
        source_surface_evidence=verified_evidence,
        subject_surface_evidence=subjects[0],
        scene=capture.scene,
        intervention=intervention,
        workspace=workspace,
        placement=placement,
        camera_evidence_sha256=camera_evidence_sha256,
        max_candidate_points=max_candidate_points,
        max_included_collision_obstacles=max_included_collision_obstacles,
        screening_domain_operations=screening_domain_operations,
    )


def _endpoint_worker(connection: Connection, arguments: tuple[object, ...]) -> None:
    try:
        if len(arguments) != 12:
            connection.send(("error", "InvalidWorkerArguments"))
            return
        (
            scene,
            intervention,
            workspace,
            config,
            source_surface_evidence,
            subject_surface_evidence,
            source_view_fact,
            placement,
            case_id,
            max_candidate_points,
            max_included_collision_obstacles,
            screening_domain_operations,
        ) = arguments
        try:
            result = plan_endpoint(
                scene,
                intervention,
                workspace,
                config,
                source_surface_evidence,
                subject_surface_evidence,
                source_view_fact,
                placement=placement,
                case_id=case_id,
                max_candidate_points=max_candidate_points,
                max_included_collision_obstacles=(max_included_collision_obstacles),
                screening_domain_operations=screening_domain_operations,
            )
        except EndpointPlanRejected as error:
            connection.send(("rejected", error.reasons))
        except Exception as error:  # noqa: BLE001
            connection.send(("error", type(error).__name__))
        else:
            connection.send(("ok", result))
    finally:
        connection.close()


def _bounded_default_endpoint_plan(
    policy: SourcePolicy, job: _PlanningJob
) -> EndpointPlan:
    arguments = (
        job.scene,
        job.intervention,
        job.workspace,
        policy.solver_config,
        job.source_surface_evidence,
        job.subject_surface_evidence,
        job.capture.source_view_fact,
        job.placement,
        job.case_id,
        job.max_candidate_points,
        job.max_included_collision_obstacles,
        job.screening_domain_operations,
    )
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_endpoint_worker, args=(sender, arguments), daemon=False
    )
    try:
        try:
            process.start()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise EndpointPlanRejected(
                (f"endpoint_plan:worker_start_{type(error).__name__.lower()}",)
            ) from error
        finally:
            sender.close()
        if not receiver.poll(policy.candidate_wall_time_seconds):
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            raise EndpointPlanRejected(("endpoint_plan:candidate_wall_time_exceeded",))
        try:
            payload = receiver.recv()
        except EOFError as error:
            raise EndpointPlanRejected(("endpoint_plan:worker_eof",)) from error
    finally:
        receiver.close()
        if process.pid is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        process.close()
    if type(payload) is not tuple or len(payload) < 2 or type(payload[0]) is not str:
        raise EndpointPlanRejected(("endpoint_plan:worker_invalid_payload",))
    if payload[0] == "ok" and len(payload) == 2:
        if type(payload[1]) is not EndpointPlan:
            raise EndpointPlanRejected(("endpoint_plan:worker_invalid_result",))
        return payload[1]
    if payload[0] == "rejected" and len(payload) == 2:
        reasons = payload[1]
        if (
            type(reasons) is not tuple
            or not reasons
            or any(type(item) is not str or not item for item in reasons)
        ):
            raise EndpointPlanRejected(("endpoint_plan:worker_invalid_rejection",))
        raise EndpointPlanRejected(reasons)
    raise EndpointPlanRejected(("endpoint_plan:worker_internal_error",))


def _run_job(
    policy: SourcePolicy,
    endpoint_planner: Callable[..., object],
    job: _PlanningJob,
) -> SourceRequestOutcome:
    request = job.request
    default_planner = endpoint_planner is plan_endpoint
    try:
        endpoint = (
            _bounded_default_endpoint_plan(policy, job)
            if default_planner
            else endpoint_planner(
                job.scene,
                job.intervention,
                job.workspace,
                policy.solver_config,
                job.source_surface_evidence,
                job.subject_surface_evidence,
                job.capture.source_view_fact,
                placement=job.placement,
                case_id=job.case_id,
                max_candidate_points=job.max_candidate_points,
                max_included_collision_obstacles=(job.max_included_collision_obstacles),
                screening_domain_operations=job.screening_domain_operations,
            )
        )
    except EndpointPlanRejected as error:
        reasons = error.reasons
        if default_planner and any(
            item.startswith("endpoint_plan:worker_") for item in reasons
        ):
            reasons = ("endpoint_plan:worker_internal_error",)
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            _endpoint_rejection_reasons(reasons),
            capture=job.capture,
            placement=job.placement_fact,
        )
    except Exception as error:  # noqa: BLE001
        reason = (
            _endpoint_rejection_reasons(("endpoint_plan:worker_internal_error",))
            if default_planner
            else f"source_plan:endpoint_exception:{type(error).__name__}"
        )
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            reason,
            capture=job.capture,
            placement=job.placement_fact,
        )
    if type(endpoint) is not EndpointPlan:
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_invalid_result",
            capture=job.capture,
            placement=job.placement_fact,
        )
    subject_evidence = job.subject_surface_evidence
    if (
        endpoint.planning_workspace != job.workspace
        or not _workspace_is_subset(endpoint.endpoint_workspace, job.workspace)
        or endpoint.candidate_point_count > policy.max_endpoint_candidate_points
        or endpoint.attempted_workspace_count > 4 * endpoint.candidate_point_count
        or endpoint.source_capture_sha256 != job.capture.source_capture_sha256
        or job.capture.source_view_fact is None
        or endpoint.source_view_fact_sha256
        != job.capture.source_view_fact.source_view_fact_sha256
        or endpoint.semantic_problem_sha256
        != endpoint.source_view_guard.semantic_problem_sha256
        or endpoint.selected_edit_sha256 != endpoint.source_view_guard.edit_sha256
        or endpoint.placement_sha256 != job.placement_fact.placement_sha256
        or endpoint.surface_evidence_sha256
        != job.source_surface_evidence.surface_evidence_sha256
        or endpoint.subject_surface_evidence_sha256
        != subject_evidence.subject_surface_evidence_sha256
        or endpoint.patch_index >= len(subject_evidence.patches)
        or endpoint.patch_sha256
        != subject_evidence.patches[endpoint.patch_index].patch_sha256
        or endpoint.runtime_collision_delegated_native_object_ids
        != tuple(sorted(endpoint.runtime_collision_delegated_native_object_ids))
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_binding_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    region = job.placement_fact.position_region
    if region is None or not endpoint_workspace_within_position_region(
        job.scene.object_by_id(request.subject_id),
        region,
        endpoint.endpoint_workspace,
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_outside_receptacle_position_region",
            capture=job.capture,
            placement=job.placement_fact,
        )
    try:
        proxy = build_proxy_bundle(
            job.scene,
            job.intervention,
            workspace=endpoint.endpoint_workspace,
            source_surface_evidence=job.source_surface_evidence,
            subject_surface_evidence=subject_evidence,
            placement=job.placement,
            collision_delegation=CollisionDelegation(
                case_id=job.case_id, patch_index=endpoint.patch_index
            ),
        )
    except Exception as error:  # noqa: BLE001
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            f"source_plan:proxy_replay:{type(error).__name__}",
            capture=job.capture,
            placement=job.placement_fact,
        )
    if (
        endpoint.proxy_bundle_sha256 != proxy.proxy_bundle_sha256
        or endpoint.runtime_collision_delegated_native_object_ids
        != proxy.binding.runtime_collision_delegated_native_object_ids
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_proxy_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    fresh_result = _fresh_solve_result(
        proxy, policy.solver_config, endpoint.solve_result_sha256
    )
    if fresh_result is None:
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_solve_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    if (
        endpoint.semantic_problem_sha256
        != proxy.semantic_problem.semantic_problem_sha256
        or endpoint.selected_edit_sha256
        != fresh_result.selected_witness.edit.edit_sha256
    ):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_identity_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    replayed_guard = evaluate_source_view_guard(
        job.scene,
        job.intervention,
        job.capture.source_view_fact,
        fresh_result.selected_witness.edit,
        semantic_problem_sha256=proxy.semantic_problem.semantic_problem_sha256,
        solve_result_sha256=fresh_result.solve_result_sha256,
    )
    if replayed_guard != endpoint.source_view_guard:
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:endpoint_guard_mismatch",
            capture=job.capture,
            placement=job.placement_fact,
        )
    try:
        BatchRequest(
            request_id=request.request_id,
            case_id=job.case_id,
            scene_id=request.scene_id,
            subject_name=request.subject_name,
            reference_name=request.reference_name,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            endpoint_workspace=endpoint.endpoint_workspace,
            solver_config=policy.solver_config,
            max_settlement_steps=policy.max_settlement_steps,
        )
    except (TypeError, ValueError):
        return _rejected_outcome(
            request,
            job.slot,
            job.case_id,
            "source_plan:batch_contract_unsupported",
            capture=job.capture,
            placement=job.placement_fact,
        )
    return SourceRequestOutcome(
        request_id=request.request_id,
        candidate_id=request.candidate_id,
        selection_index=request.selection_index,
        relation_before=request.relation_before,
        relation_slot_index=job.slot,
        status="planned",
        source_id=request.source_id,
        source_locator_sha256=request.source_locator_sha256,
        source_capture_sha256=request.source_capture_sha256,
        source_scene_sha256=_legacy_sha256(job.scene),
        runtime_identity_sha256=_runtime_identity_sha256(job.capture),
        scene_id=request.scene_id,
        subject_id=request.subject_id,
        subject_name=request.subject_name,
        reference_id=request.reference_id,
        reference_name=request.reference_name,
        support_kind=request.support_kind,
        placement_sha256=job.placement_fact.placement_sha256,
        case_id=job.case_id,
        planning_workspace=endpoint.planning_workspace,
        endpoint_workspace=endpoint.endpoint_workspace,
        endpoint_candidate_index=endpoint.candidate_index,
        attempted_workspace_count=endpoint.attempted_workspace_count,
        candidate_point_count=endpoint.candidate_point_count,
        solve_result_sha256=endpoint.solve_result_sha256,
        selected_edit_sha256=endpoint.selected_edit_sha256,
        source_view_fact_sha256=endpoint.source_view_fact_sha256,
        source_view_guard=endpoint.source_view_guard,
        reasons=(),
        surface_evidence_sha256=(job.source_surface_evidence.surface_evidence_sha256),
        subject_surface_evidence_sha256=(
            subject_evidence.subject_surface_evidence_sha256
        ),
        patch_index=endpoint.patch_index,
        patch_sha256=endpoint.patch_sha256,
        semantic_problem_sha256=proxy.semantic_problem.semantic_problem_sha256,
        proxy_bundle_sha256=proxy.proxy_bundle_sha256,
        endpoint_plan_sha256=endpoint.endpoint_plan_sha256,
        camera_evidence_sha256=job.camera_evidence_sha256,
        runtime_collision_delegated_native_object_ids=(
            endpoint.runtime_collision_delegated_native_object_ids
        ),
    )


def _plan_parallel_outcomes(
    policy: SourcePolicy,
    requests: tuple[CompetitionNativeSelectedRequestV2_9, ...],
    request_slots: dict[str, int],
    captures: dict[str, CompetitionNativeSourceCaptureV2_9],
    surfaces: dict[str, SourceSurfaceEvidence],
    cameras: dict[str, SourceCameraEvidence],
    endpoint_planner: Callable[..., object],
) -> tuple[SourceRequestOutcome, ...]:
    ordered: list[SourceRequestOutcome | None] = [None] * len(requests)
    jobs: list[_PlanningJob] = []
    for index, request in enumerate(requests):
        prepared = _prepare_job(
            policy,
            request,
            manifest_index=index,
            slot=request_slots[request.request_id],
            capture=captures[request.source_id],
            source_surface_evidence=surfaces[request.source_id],
            camera_evidence_sha256=cameras[request.source_id].camera_evidence_sha256,
        )
        if type(prepared) is SourceRequestOutcome:
            ordered[index] = prepared
        elif type(prepared) is _PlanningJob:
            jobs.append(prepared)
        else:
            raise RuntimeError("endpoint preflight returned invalid result")
    with ThreadPoolExecutor(
        max_workers=policy.max_parallel_endpoint_workers,
        thread_name_prefix="spatialcf-endpoint-current",
    ) as executor:
        futures = {
            executor.submit(_run_job, policy, endpoint_planner, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            if ordered[job.manifest_index] is not None:
                raise RuntimeError("endpoint outcome index was reused")
            ordered[job.manifest_index] = future.result()
    if any(item is None for item in ordered):
        raise RuntimeError("endpoint outcomes do not close the manifest")
    return tuple(item for item in ordered if item is not None)


def plan_source_campaign(
    compilation: RosterCompilation,
    policy: SourcePolicy,
    *,
    endpoint_planner: Callable[..., object] = plan_endpoint,
) -> SourcePlan:
    """Plan each frozen request once, without backfilling rejected slots."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(compilation) is not RosterCompilation:
            raise TypeError("candidate roster compilation must be exact")
        if type(policy) is not SourcePolicy:
            raise TypeError("native source policy must be exact")
        checked = RosterCompilation.model_validate(
            compilation.model_dump(mode="python", warnings="error"), strict=True
        )
        recomputed = compile_roster(
            checked.policy,
            checked.scene_inventory,
            checked.surface_evidence,
            checked.camera_evidence,
        )
        if checked != recomputed:
            raise ValueError("candidate roster compilation is not freshly reproducible")
        policy = SourcePolicy.model_validate(
            policy.model_dump(mode="python", warnings="error"), strict=True
        )
        manifest = checked.request_manifest
        if (
            policy.roster_manifest_sha256 != manifest.manifest_sha256
            or policy.campaign_id != manifest.campaign_id
        ):
            raise ValueError("native source policy roster manifest digest mismatch")
        if (policy.width, policy.height, policy.seed) != (
            checked.policy.width,
            checked.policy.height,
            checked.policy.seed,
        ):
            raise ValueError("native source policy render does not match roster")
        if (
            policy.roster_policy_sha256 != checked.policy.policy_sha256
            or policy.target_reachability_ledger_sha256
            != _target_reachability_ledger_sha256(checked.target_reachability)
        ):
            raise ValueError("target-reachability source policy roster binding changed")

        relation_ordinals: Counter[Relation] = Counter()
        request_slots: dict[str, int] = {}
        relation_members: dict[tuple[Relation, int], str] = {}
        for request in manifest.requests:
            slot = relation_ordinals[request.relation_before]
            relation_ordinals[request.relation_before] += 1
            request_slots[request.request_id] = slot
            relation_members[(request.relation_before, slot)] = request.request_id
        slot_count = max(relation_ordinals.values(), default=0)
        frozen_entries = tuple(
            tuple(
                SourceSlotEntry(
                    relation=relation,
                    request_id=relation_members.get((relation, slot)),
                )
                for relation in _RELATIONS
            )
            for slot in range(slot_count)
        )

        selected_source_ids = {item.source_id for item in manifest.requests}
        source_captures = tuple(
            item.capture
            for item in checked.scene_inventory
            if item.status == "accepted" and item.capture is not None
        )
        captures = {item.source.source_id: item for item in source_captures}
        if not selected_source_ids.issubset(captures):
            raise ValueError("selected request has no accepted frozen capture")
        if policy.accepted_source_capture_roster_sha256 != (
            _accepted_source_capture_roster_sha256(source_captures)
        ):
            raise ValueError("camera-bound accepted capture roster digest mismatch")
        surfaces = {item.source_id: item for item in checked.surface_evidence}
        cameras = {
            item.source_id: verify_competition_native_camera_evidence_capture_v2_9_3(
                captures[item.source_id], item, policy.camera_policy
            )
            for item in checked.camera_evidence
            if item.source_id in captures
        }
        if set(surfaces) != set(captures):
            raise ValueError("surface evidence does not cover accepted sources")
        if set(cameras) != set(captures):
            raise ValueError("camera evidence does not cover accepted sources")

        outcomes = _plan_parallel_outcomes(
            policy,
            manifest.requests,
            request_slots,
            captures,
            surfaces,
            cameras,
            endpoint_planner,
        )
        outcome_by_id = {item.request_id: item for item in outcomes}
        request_by_id = {item.request_id: item for item in manifest.requests}
        slots: list[SourceSlotOutcome] = []
        batches: list[BatchManifest] = []
        for slot_index, entries in enumerate(frozen_entries):
            blockers = tuple(
                sorted(
                    f"source_plan:missing_relation:{entry.relation.value}"
                    if entry.request_id is None
                    else f"source_plan:request_rejected:{entry.request_id}"
                    for entry in entries
                    if entry.request_id is None
                    or outcome_by_id[entry.request_id].status != "planned"
                )
            )
            slots.append(
                SourceSlotOutcome(
                    slot_index=slot_index,
                    status="incomplete" if blockers else "ready",
                    entries=entries,
                    blockers=blockers,
                    batch_id=None if blockers else f"b{slot_index:04d}",
                )
            )
            planned_request_ids = tuple(
                sorted(
                    entry.request_id
                    for entry in entries
                    if entry.request_id is not None
                    and outcome_by_id[entry.request_id].status == "planned"
                )
            )
            if not planned_request_ids:
                continue
            batches.append(
                BatchManifest(
                    batch_id=f"b{slot_index:04d}",
                    width=policy.width,
                    height=policy.height,
                    seed=policy.seed,
                    requests=tuple(
                        BatchRequest(
                            request_id=request_id,
                            case_id=outcome_by_id[request_id].case_id,
                            scene_id=request_by_id[request_id].scene_id,
                            subject_name=request_by_id[request_id].subject_name,
                            reference_name=request_by_id[request_id].reference_name,
                            relation_before=request_by_id[request_id].relation_before,
                            relation_after=request_by_id[request_id].relation_after,
                            endpoint_workspace=outcome_by_id[
                                request_id
                            ].endpoint_workspace,
                            solver_config=policy.solver_config,
                            max_settlement_steps=policy.max_settlement_steps,
                        )
                        for request_id in planned_request_ids
                    ),
                )
            )
        return SourcePlan(
            source_policy=policy,
            source_policy_sha256=policy.competition_native_source_policy_sha256,
            roster_manifest=manifest,
            roster_manifest_sha256=manifest.manifest_sha256,
            roster_manifest_file_sha256=_manifest_file_sha256(manifest),
            source_captures=source_captures,
            request_outcomes=outcomes,
            slots=tuple(slots),
            batches=tuple(batches),
            surface_evidence=tuple(
                surfaces[item.source.source_id] for item in source_captures
            ),
            camera_evidence=tuple(
                cameras[item.source.source_id] for item in source_captures
            ),
            target_reachability=checked.target_reachability,
        )


# Preserve supported public names and pickle lookup.
build_default_source_policy.__module__ = "spatialcf.generation.planning.campaign"
_endpoint_rejection_reason.__module__ = "spatialcf.generation.planning.campaign"
_endpoint_rejection_reasons.__module__ = "spatialcf.generation.planning.campaign"
_rejected_outcome.__module__ = "spatialcf.generation.planning.campaign"
_endpoint_policy_controls.__module__ = "spatialcf.generation.planning.campaign"
_preflight_request.__module__ = "spatialcf.generation.planning.campaign"
_PlanningJob.__module__ = "spatialcf.generation.planning.campaign"
_prepare_job.__module__ = "spatialcf.generation.planning.campaign"
_endpoint_worker.__module__ = "spatialcf.generation.planning.campaign"
_bounded_default_endpoint_plan.__module__ = "spatialcf.generation.planning.campaign"
_run_job.__module__ = "spatialcf.generation.planning.campaign"
_plan_parallel_outcomes.__module__ = "spatialcf.generation.planning.campaign"
plan_source_campaign.__module__ = "spatialcf.generation.planning.campaign"
