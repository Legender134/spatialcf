"""Coordinate capture, planning, batch execution and dataset publication."""

from __future__ import annotations

from collections.abc import (
    Callable,
)

from pathlib import (
    Path,
)

from spatialcf.adapters.base import (
    EnvironmentAdapter as AI2ThorAdapter,
)

from spatialcf.adapters.base import (
    SettledReadback,
)

from spatialcf.domain.request import (
    InterventionSpec,
)

import spatialcf.generation.capture as capture
import spatialcf.generation.execution as execution
import spatialcf.generation.planning as planning

from spatialcf.generation.config import (
    GenerationConfig,
)

from spatialcf.generation.dataset_models import (
    GenerationReport,
)

from spatialcf.generation.execution.audit import (
    EndpointAuditRejected,
    _execute_audit_with_transitions,
)

from spatialcf.generation.workflows.capture import (
    _capture_and_publish_dataset_with_transitions,
)

from spatialcf.generation.workflows.contracts import (
    CounterfactualRequest,
    ExecutedEdit,
    ExecutionRejection,
    VerificationRejection,
    _FreshTransitions,
)

from spatialcf.generation.workflows._dataset.artifacts import (
    _path_attempts,
)

from spatialcf.generation.workflows._dataset.derivation import (
    _derive_dataset,
)

from spatialcf.generation.workflows._dataset.publication import (
    _publish_dataset_index,
)

from spatialcf.generation.workflows._dataset.state import (
    _absolute_output,
    _capture_plan,
    _checked_config,
    _ensure_batches_root,
    _initialize_dataset_root,
    _require_complete_source_plan,
    _validate_generation_root,
)


def _load_or_build_roster(
    plan: capture.CapturePlan,
    root: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
    fresh_transitions: _FreshTransitions | None = None,
):
    if root.exists():
        summary = capture.verify_roster(root)
    else:
        if fresh_transitions is None:
            summary = capture.capture_and_publish_dataset(
                plan,
                root,
                adapter_factory=adapter_factory,
            )
        else:
            summary = _capture_and_publish_dataset_with_transitions(
                plan,
                root,
                adapter_factory=adapter_factory,
                dataset_loader=capture.load_prior_dataset,
                fresh_transitions=fresh_transitions,
            )
    compilation = capture.load_roster(root)
    if compilation.summary != summary or compilation.policy != plan.roster_policy:
        raise ValueError("dataset roster stage identity differs from config")
    return compilation


def _load_or_build_source_plan(
    compilation,
    root: Path,
    *,
    fresh_transitions: _FreshTransitions | None = None,
) -> planning.SourcePlan:
    expected_policy = planning.build_default_source_policy(compilation)
    if root.exists():
        plan = planning.load_source_plan(root)
        if plan.source_policy.policy_version == "competition-native-source-policy:2.9.14":
            expected_policy = planning.build_default_source_policy(
                compilation, candidate_strategy=plan.source_policy.endpoint_candidate_strategy,
            )
    else:
        planned = _plan_source_campaign_with_transitions(
            compilation,
            expected_policy,
            fresh_transitions=fresh_transitions,
        )
        planning.publish_source_plan(planned, root)
        plan = planning.load_source_plan(root)
    if (
        plan.source_policy != expected_policy
        or plan.roster_manifest != compilation.request_manifest
    ):
        raise ValueError("dataset source-plan stage identity differs from config")
    return plan


def _plan_source_campaign_with_transitions(
    compilation,
    policy,
    *,
    fresh_transitions: _FreshTransitions | None,
) -> planning.SourcePlan:
    plan = planning.plan_source_campaign(compilation, policy)
    if fresh_transitions is None:
        return plan
    for request, outcome in zip(
        plan.roster_manifest.requests, plan.request_outcomes, strict=True
    ):
        intervention = InterventionSpec(
            subject_id=request.subject_id,
            reference_id=request.reference_id,
            relation_before=request.relation_before,
            relation_after=request.relation_after,
            camera_id=request.camera_id,
        )
        value = fresh_transitions.request(
            request.request_id, request.source_id, intervention
        )
        if type(value) is not CounterfactualRequest:
            raise TypeError("planning transition request must be exact")
        if outcome.status == "rejected":
            fresh_transitions.reject_planning(
                request.request_id, "|".join(outcome.reasons)
            )
    return plan


def _run_batches(
    plan: planning.SourcePlan,
    root: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
    fresh_transitions: _FreshTransitions | None = None,
) -> execution.SourceExecutionSummary:
    _require_complete_source_plan(plan)
    _ensure_batches_root(root)

    class _FreshAuditRunner:
        def __call__(self, adapter, request, *, request_lineage):
            try:
                return _execute_audit_with_transitions(
                    adapter,
                    request,
                    request_lineage=request_lineage,
                    fresh_transitions=fresh_transitions,
                )
            except EndpointAuditRejected as error:
                if error.stage == "native_verification":
                    readback = error.readback
                    if type(readback) is not SettledReadback:
                        raise TypeError(
                            "native verification rejection must retain exact readback"
                        )
                    executed = fresh_transitions.executed(
                        request.request_id,
                        readback.application.edit,
                        readback,
                    )
                    if type(executed) is not ExecutedEdit:
                        raise TypeError(
                            "verification transition execution must be exact"
                        )
                    rejection = fresh_transitions.reject_verification(
                        request.request_id, "|".join(error.reasons)
                    )
                    if type(rejection) is not VerificationRejection:
                        raise TypeError(
                            "verification rejection transition must be exact"
                        )
                else:
                    rejection = fresh_transitions.reject_execution(
                        request.request_id, "|".join(error.reasons)
                    )
                    if type(rejection) is not ExecutionRejection:
                        raise TypeError("execution rejection transition must be exact")
                raise

        def require_verified(self, request_id: str) -> None:
            fresh_transitions.require_verified(request_id)

    runner = _FreshAuditRunner() if fresh_transitions is not None else None

    def execute_current_batch(
        manifest,
        output,
        *,
        expected_parent_identity,
        request_lineage,
    ):
        arguments = {
            "expected_parent_identity": expected_parent_identity,
            "request_lineage": request_lineage,
            "adapter_factory": adapter_factory,
        }
        if runner is not None:
            arguments["runner"] = runner
        return execution.execute_batch(
            manifest,
            output,
            **arguments,
        )

    return execution.run_source_campaign(
        plan,
        root,
        execute=True,
        batch_executor=execute_current_batch,
    )


def generate_dataset(
    config: GenerationConfig | Path,
    output: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter],
) -> GenerationReport:
    """Generate or resume one immutable dataset and publish its stable index."""

    checked = _checked_config(config)
    root = _absolute_output(output)
    expected_capture = _capture_plan(checked)
    fresh_transitions = _FreshTransitions() if not root.exists() else None
    if not root.exists():
        _initialize_dataset_root(root, expected_capture)
    _validate_generation_root(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    loaded_capture = capture.load_capture_plan(root / ".spatialcf" / "capture-plan")
    if loaded_capture != expected_capture:
        raise ValueError("dataset capture plan identity differs from config")
    roster = _load_or_build_roster(
        loaded_capture,
        root / ".spatialcf" / "roster",
        adapter_factory=adapter_factory,
        fresh_transitions=fresh_transitions,
    )
    source_plan = _load_or_build_source_plan(
        roster,
        root / ".spatialcf" / "source-plan",
        fresh_transitions=fresh_transitions,
    )
    _require_complete_source_plan(source_plan)
    execution_summary = _run_batches(
        source_plan,
        root / ".spatialcf" / "batches",
        adapter_factory=adapter_factory,
        fresh_transitions=fresh_transitions,
    )
    attempts = _path_attempts(
        source_plan,
        root / ".spatialcf" / "batches",
    )
    (
        _records,
        records_payload,
        _report,
        report_payload,
        manifest,
        bundles,
    ) = _derive_dataset(
        checked,
        roster,
        source_plan,
        execution_summary,
        attempts,
        fresh_transitions=fresh_transitions,
    )
    return _publish_dataset_index(
        root,
        records_payload,
        report_payload,
        manifest,
        bundles,
    )


# Preserve supported public names and pickle lookup.
_load_or_build_roster.__module__ = "spatialcf.generation.workflows.dataset"
_load_or_build_source_plan.__module__ = "spatialcf.generation.workflows.dataset"
_plan_source_campaign_with_transitions.__module__ = "spatialcf.generation.workflows.dataset"
_run_batches.__module__ = "spatialcf.generation.workflows.dataset"
generate_dataset.__module__ = "spatialcf.generation.workflows.dataset"
