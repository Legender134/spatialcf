"""Schema-2 composition wrapper for the reusable batch workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from spatialcf.composition import DEFAULT_ENVIRONMENT_ADAPTER_FACTORY as AI2ThorAdapter
from spatialcf.generation.execution.audit import AuditExecution, execute_audit
from spatialcf.generation.execution.correspondence import RequestLineage
from spatialcf.generation.planning.campaign import BatchManifest
from spatialcf.generation.workflows import execution as execution_workflow
from spatialcf.generation.workflows.execution import BatchSummary
from spatialcf.verification.filesystem import DirectoryIdentity


def execute_batch(
    manifest: BatchManifest,
    output_root: Path,
    *,
    request_lineage: Mapping[str, RequestLineage],
    expected_parent_identity: DirectoryIdentity | None = None,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
    runner: Callable[..., AuditExecution] = execute_audit,
) -> BatchSummary:
    return execution_workflow.execute_batch(
        manifest,
        output_root,
        request_lineage=request_lineage,
        expected_parent_identity=expected_parent_identity,
        adapter_factory=adapter_factory,
        runner=runner,
    )


__all__ = ("execute_batch",)
