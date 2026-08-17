"""Current dataset-capture authority and operations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from spatialcf.adapters.ai2thor import AI2ThorAdapter
from spatialcf.generation._internal.source_manifest import (
    SourcePlanManifest,
    load_source_plan_manifest,
)
from spatialcf.generation.capture.models import RosterCompilation, RosterSummary
from spatialcf.generation.capture.plan import (
    CapturePlan,
    RetainedCapturePlanVerification,
    build_capture_plan,
    build_legacy_capture_plan,
    load_capture_plan,
    prepare_capture_plan_verification,
    publish_capture_plan,
    revalidate_capture_plan_verification,
)
from spatialcf.generation.capture.source import (
    capture_dataset,
    load_prior_dataset,
)
from spatialcf.generation.capture.storage import (
    RetainedRosterVerification,
    load_roster,
    prepare_roster_verification,
    publish_roster,
    revalidate_roster_verification,
    verify_roster,
)


def load_source_manifest(path: Path) -> SourcePlanManifest:
    if not isinstance(path, Path):
        raise TypeError("source manifest path must be a Path")
    return load_source_plan_manifest(path)


def capture_and_publish_dataset(
    plan: CapturePlan,
    roster_root: Path,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
    dataset_loader: Callable[[str, str], object] = load_prior_dataset,
) -> RosterSummary:
    compilation = capture_dataset(
        plan,
        adapter_factory=adapter_factory,
        dataset_loader=dataset_loader,
    )
    return publish_roster(compilation, roster_root)


__all__ = (
    "CapturePlan",
    "RetainedCapturePlanVerification",
    "RetainedRosterVerification",
    "RosterCompilation",
    "RosterSummary",
    "build_capture_plan",
    "build_legacy_capture_plan",
    "capture_and_publish_dataset",
    "capture_dataset",
    "load_capture_plan",
    "load_roster",
    "load_source_manifest",
    "prepare_capture_plan_verification",
    "prepare_roster_verification",
    "publish_capture_plan",
    "revalidate_capture_plan_verification",
    "revalidate_roster_verification",
    "verify_roster",
)
