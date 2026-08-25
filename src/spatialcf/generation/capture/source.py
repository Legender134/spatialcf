"""Schema-2 facade for protocol-only dataset capture."""

from __future__ import annotations

from collections.abc import Callable

from spatialcf.composition import DEFAULT_ENVIRONMENT_ADAPTER_FACTORY as AI2ThorAdapter
from spatialcf.generation.capture.models import RosterCompilation
from spatialcf.generation.capture.plan import CapturePlan
from spatialcf.generation.workflows import capture as capture_workflow


def load_prior_dataset(name: str, revision: str) -> object:
    return capture_workflow.load_prior_dataset(name, revision)


def capture_dataset(
    plan: CapturePlan,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
    dataset_loader: Callable[[str, str], object] = load_prior_dataset,
) -> RosterCompilation:
    return capture_workflow.capture_dataset(
        plan, adapter_factory=adapter_factory, dataset_loader=dataset_loader
    )


__all__ = ("capture_dataset", "load_prior_dataset")
