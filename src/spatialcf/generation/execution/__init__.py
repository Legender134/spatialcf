"""Final public execution package. Lazy exports avoid loading unrelated execution owners."""

from importlib import import_module

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

_EXPORTS = {
    "AI2ThorAdapter": ("spatialcf.adapters.base", "EnvironmentAdapter"),
    "AuditExecution": ("spatialcf.generation.execution.audit", "AuditExecution"),
    "AuditRun": ("spatialcf.generation.execution.audit", "AuditRun"),
    "EndpointAudit": ("spatialcf.generation.execution.audit", "EndpointAudit"),
    "execute_audit": ("spatialcf.generation.execution.audit", "execute_audit"),
    "verify_audit_run": ("spatialcf.generation.execution.audit", "verify_audit_run"),
    "execute_batch": ("spatialcf.generation.execution.batch", "execute_batch"),
    "RetainedSourceBatchVerification": (
        "spatialcf.generation.execution.campaign",
        "RetainedSourceBatchVerification",
    ),
    "SourceExecutionSummary": (
        "spatialcf.generation.execution.campaign",
        "SourceExecutionSummary",
    ),
    "prepare_source_batch_verification": (
        "spatialcf.generation.execution.campaign",
        "prepare_source_batch_verification",
    ),
    "revalidate_source_batch_verification": (
        "spatialcf.generation.execution.campaign",
        "revalidate_source_batch_verification",
    ),
    "run_planned_requests": (
        "spatialcf.generation.execution.campaign",
        "run_planned_requests",
    ),
    "run_source_campaign": (
        "spatialcf.generation.execution.campaign",
        "run_source_campaign",
    ),
    "summarize_verified_source_campaign": (
        "spatialcf.generation.execution.campaign",
        "summarize_verified_source_campaign",
    ),
    "verify_source_campaign": (
        "spatialcf.generation.execution.campaign",
        "verify_source_campaign",
    ),
    "RequestLineage": (
        "spatialcf.generation.execution.correspondence",
        "RequestLineage",
    ),
    "BatchManifest": ("spatialcf.generation.planning.campaign", "BatchManifest"),
    "BatchRequest": ("spatialcf.generation.planning.campaign", "BatchRequest"),
    "EndpointPlan": ("spatialcf.generation.planning.models", "EndpointPlan"),
    "ProxyBundle": ("spatialcf.generation.planning.models", "ProxyBundle"),
    "BatchAttempt": ("spatialcf.generation.workflows.execution", "BatchAttempt"),
    "BatchStageCount": ("spatialcf.generation.workflows.execution", "BatchStageCount"),
    "BatchSummary": ("spatialcf.generation.workflows.execution", "BatchSummary"),
    "RetainedBatchVerification": (
        "spatialcf.generation.workflows.execution",
        "RetainedBatchVerification",
    ),
    "load_batch_manifest": (
        "spatialcf.generation.workflows.execution",
        "load_batch_manifest",
    ),
    "prepare_batch_verification": (
        "spatialcf.generation.workflows.execution",
        "prepare_batch_verification",
    ),
    "revalidate_batch_verification": (
        "spatialcf.generation.workflows.execution",
        "revalidate_batch_verification",
    ),
    "verify_batch": ("spatialcf.generation.workflows.execution", "verify_batch"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
