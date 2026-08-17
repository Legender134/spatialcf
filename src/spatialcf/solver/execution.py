"""Platform-neutral execution oracle for grounded counterfactual search.

The core solver only speaks canonical :class:`~spatialcf.domain.models.Scene`
objects.  Simulator actions, physics engines, renderer events, and platform
specific tolerances belong in implementations of :class:`CandidateExecutor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from spatialcf.domain.models import InterventionSpec, Scene

EvidenceT_co = TypeVar("EvidenceT_co", covariant=True)


class CandidateExecutionStatus(StrEnum):
    """Outcome of attempting one command against an external execution model."""

    OBSERVED = "OBSERVED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ExecutionResidual:
    """One platform-measured residual and its frozen acceptance limit."""

    name: str
    value: float
    limit: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("residual name must be a non-empty exact string")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("residual value must be an exact finite float")
        if self.value < 0.0:
            raise ValueError("residual value must be non-negative")
        if type(self.limit) is not float or not math.isfinite(self.limit):
            raise ValueError("residual limit must be an exact finite float")
        if self.limit < 0.0:
            raise ValueError("residual limit must be non-negative")

    @property
    def within_limit(self) -> bool:
        return self.value <= self.limit


@dataclass(frozen=True)
class CandidateExecution(Generic[EvidenceT_co]):
    """A canonical observation, a candidate rejection, or an executor failure.

    ``REJECTED`` means this particular command is known to be infeasible and
    search may continue.  ``ERROR`` means execution state or evidence integrity
    is no longer trustworthy; a solver must stop without claiming UNSAT.
    """

    status: CandidateExecutionStatus
    commanded_scene: Scene
    observed_before_scene: Scene | None
    observed_scene: Scene | None
    residuals: tuple[ExecutionResidual, ...]
    errors: tuple[str, ...]
    evidence: EvidenceT_co | None

    def __post_init__(self) -> None:
        if type(self.status) is not CandidateExecutionStatus:
            raise ValueError("execution status must be CandidateExecutionStatus")
        if type(self.commanded_scene) is not Scene:
            raise ValueError("commanded_scene must be a canonical Scene")
        if (
            self.observed_before_scene is not None
            and type(self.observed_before_scene) is not Scene
        ):
            raise ValueError("observed_before_scene must be a canonical Scene")
        if self.observed_scene is not None and type(self.observed_scene) is not Scene:
            raise ValueError("observed_scene must be a canonical Scene")
        if type(self.residuals) is not tuple or any(
            type(residual) is not ExecutionResidual for residual in self.residuals
        ):
            raise ValueError("residuals must be a tuple of ExecutionResidual")
        if type(self.errors) is not tuple or any(
            type(error) is not str or not error for error in self.errors
        ):
            raise ValueError("errors must be a tuple of non-empty exact strings")
        if len(set(self.errors)) != len(self.errors) or tuple(sorted(self.errors)) != self.errors:
            raise ValueError("execution errors must be sorted and unique")
        if self.status is CandidateExecutionStatus.OBSERVED:
            if self.observed_before_scene is None:
                raise ValueError("OBSERVED execution requires an observed_before_scene")
            if self.observed_scene is None:
                raise ValueError("OBSERVED execution requires an observed_scene")
            if self.errors:
                raise ValueError("OBSERVED execution cannot contain errors")
            if any(not residual.within_limit for residual in self.residuals):
                raise ValueError("OBSERVED execution contains a failed residual")
        elif self.status is CandidateExecutionStatus.REJECTED:
            if self.observed_before_scene is None:
                raise ValueError("REJECTED execution requires an observed_before_scene")
            if not self.errors:
                raise ValueError("REJECTED execution requires errors")
        else:
            if self.observed_before_scene is not None or self.observed_scene is not None:
                raise ValueError("ERROR execution cannot contain observed scenes")
            if not self.errors:
                raise ValueError("ERROR execution requires errors")

    @classmethod
    def observed(
        cls,
        commanded_scene: Scene,
        *,
        observed_before_scene: Scene,
        observed_scene: Scene,
        residuals: tuple[ExecutionResidual, ...] = (),
        evidence: EvidenceT_co | None = None,
    ) -> CandidateExecution[EvidenceT_co]:
        return cls(
            CandidateExecutionStatus.OBSERVED,
            commanded_scene,
            observed_before_scene,
            observed_scene,
            residuals,
            (),
            evidence,
        )

    @classmethod
    def rejected(
        cls,
        commanded_scene: Scene,
        errors: tuple[str, ...],
        *,
        observed_before_scene: Scene,
        observed_scene: Scene | None = None,
        residuals: tuple[ExecutionResidual, ...] = (),
        evidence: EvidenceT_co | None = None,
    ) -> CandidateExecution[EvidenceT_co]:
        return cls(
            CandidateExecutionStatus.REJECTED,
            commanded_scene,
            observed_before_scene,
            observed_scene,
            residuals,
            tuple(sorted(set(errors))),
            evidence,
        )

    @classmethod
    def error(
        cls,
        commanded_scene: Scene,
        errors: tuple[str, ...],
        *,
        evidence: EvidenceT_co | None = None,
    ) -> CandidateExecution[EvidenceT_co]:
        return cls(
            CandidateExecutionStatus.ERROR,
            commanded_scene,
            None,
            None,
            (),
            tuple(sorted(set(errors))),
            evidence,
        )


@runtime_checkable
class CandidateExecutor(Protocol[EvidenceT_co]):
    """Execute each command from the same source state and fixed camera.

    Implementations must isolate candidates from one another.  ``OBSERVED`` and
    ``REJECTED`` results carry the same episode's canonical source observation;
    an ``OBSERVED`` after-scene is mapped to the canonical IR only after every
    platform-specific residual has been checked.  The core independently checks
    that the episode represents the nominal source semantics, then reruns its
    verifier and objective on the observed before/after pair.
    """

    def execute_candidate(
        self,
        before: Scene,
        commanded: Scene,
        spec: InterventionSpec,
    ) -> CandidateExecution[EvidenceT_co]:
        ...
