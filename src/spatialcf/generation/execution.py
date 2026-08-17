"""Current native-execution authority aliases and operations."""

from __future__ import annotations

from spatialcf.adapters.ai2thor import AI2ThorAdapter
from spatialcf.domain.models import Scene
from spatialcf.generation._internal.execution import (
    AuditExecution,
    AuditRun,
    EndpointAudit,
    RequestLineage,
)
from spatialcf.generation._internal.execution import (
    execute_audit as _execute_current_audit,
)
from spatialcf.generation._internal.execution import (
    verify_audit_run as _verify_current_audit_run,
)
from spatialcf.generation._internal.execution.batch import (
    BatchAttempt,
    BatchManifest,
    BatchStageCount,
    BatchSummary,
    RetainedBatchVerification,
    execute_batch,
    load_batch_manifest,
    prepare_batch_verification,
    revalidate_batch_verification,
    verify_batch,
)
from spatialcf.generation._internal.execution.campaign import (
    RetainedSourceBatchVerification,
    SourceExecutionSummary,
    prepare_source_batch_verification,
    revalidate_source_batch_verification,
    run_planned_requests,
    run_source_campaign,
    summarize_verified_source_campaign,
    verify_source_campaign,
)
from spatialcf.generation._internal.planning.campaign import BatchRequest
from spatialcf.generation._internal.planning.models import EndpointPlan, ProxyBundle


def execute_audit(
    adapter: AI2ThorAdapter,
    request: BatchRequest,
    *,
    request_lineage: RequestLineage,
) -> AuditExecution:
    return _execute_current_audit(
        adapter,
        request,
        request_lineage=request_lineage,
    )


def verify_audit_run(run: AuditRun, observed_after_scene: Scene) -> AuditRun:
    return _verify_current_audit_run(run, observed_after_scene)


__all__ = (
    "AuditExecution",
    "AuditRun",
    "BatchAttempt",
    "BatchManifest",
    "BatchRequest",
    "BatchStageCount",
    "BatchSummary",
    "EndpointAudit",
    "EndpointPlan",
    "ProxyBundle",
    "RequestLineage",
    "RetainedBatchVerification",
    "RetainedSourceBatchVerification",
    "SourceExecutionSummary",
    "execute_audit",
    "execute_batch",
    "load_batch_manifest",
    "prepare_batch_verification",
    "prepare_source_batch_verification",
    "revalidate_batch_verification",
    "revalidate_source_batch_verification",
    "run_planned_requests",
    "run_source_campaign",
    "summarize_verified_source_campaign",
    "verify_audit_run",
    "verify_batch",
    "verify_source_campaign",
)
