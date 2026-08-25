"""Unique generation transition values between protocol workflow stages."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcf.adapters.base import CapturedSource, SettledReadback
from spatialcf.domain.edit import CanonicalEdit
from spatialcf.domain.request import InterventionSpec


@dataclass(frozen=True)
class CounterfactualRequest:
    source: CapturedSource
    intervention: InterventionSpec

    def __post_init__(self) -> None:
        if type(self.source) is not CapturedSource:
            raise TypeError("counterfactual request source must be exact")
        if type(self.intervention) is not InterventionSpec:
            raise TypeError("counterfactual request intervention must be exact")


@dataclass(frozen=True)
class CertifiedPlan:
    request: CounterfactualRequest
    edit: CanonicalEdit


@dataclass(frozen=True)
class PlanningRejection:
    request: CounterfactualRequest
    reason: str


@dataclass(frozen=True)
class ExecutedEdit:
    plan: CertifiedPlan
    readback: SettledReadback

    def __post_init__(self) -> None:
        if type(self.plan) is not CertifiedPlan:
            raise TypeError("executed edit plan must be exact")
        if type(self.readback) is not SettledReadback:
            raise TypeError("executed edit readback must be exact")
        if (
            self.readback.application.intervention != self.plan.request.intervention
            or self.readback.application.edit != self.plan.edit
        ):
            raise ValueError("executed edit does not bind its certified plan")


@dataclass(frozen=True)
class ExecutionRejection:
    request: CounterfactualRequest
    reason: str
    plan: CertifiedPlan | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not CounterfactualRequest:
            raise TypeError("execution rejection request must be exact")
        if self.plan is not None and (
            type(self.plan) is not CertifiedPlan
            or self.plan.request is not self.request
        ):
            raise ValueError("execution rejection plan does not bind its request")


@dataclass(frozen=True)
class VerifiedExample:
    execution: ExecutedEdit


@dataclass(frozen=True)
class VerificationRejection:
    execution: ExecutedEdit
    reason: str


class _FreshTransitions:
    """Explicit, process-local transition carrier for one fresh dataset root."""

    def __init__(self) -> None:
        self._sources: dict[str, CapturedSource] = {}
        self._requests: dict[str, CounterfactualRequest] = {}
        self._plans: dict[str, CertifiedPlan] = {}
        self._executed: dict[str, ExecutedEdit] = {}
        self._verified: dict[str, VerifiedExample] = {}
        self._terminals: dict[
            str,
            PlanningRejection
            | ExecutionRejection
            | VerificationRejection
            | VerifiedExample,
        ] = {}

    def capture(self, source_id: str, source: CapturedSource) -> None:
        if type(source_id) is not str or type(source) is not CapturedSource:
            raise TypeError("fresh capture carrier requires exact source values")
        if source_id in self._sources:
            raise ValueError("fresh capture carrier reused a source")
        self._sources[source_id] = source

    def request(
        self, request_id: str, source_id: str, intervention: InterventionSpec
    ) -> CounterfactualRequest:
        if request_id in self._requests or request_id in self._terminals:
            raise ValueError("fresh transition request was reused")
        try:
            source = self._sources[source_id]
        except KeyError as error:
            raise ValueError(
                "fresh transition request has no captured source"
            ) from error
        value = CounterfactualRequest(source=source, intervention=intervention)
        self._requests[request_id] = value
        return value

    def reject_planning(self, request_id: str, reason: str) -> PlanningRejection:
        value = PlanningRejection(request=self._request(request_id), reason=reason)
        self._terminal_once(request_id, value)
        return value

    def executed(
        self, request_id: str, edit: CanonicalEdit, readback: SettledReadback
    ) -> ExecutedEdit:
        request = self._request(request_id)
        if request_id in self._plans or request_id in self._terminals:
            raise ValueError("fresh transition plan was reused")
        plan = CertifiedPlan(request=request, edit=edit)
        value = ExecutedEdit(plan=plan, readback=readback)
        self._plans[request_id] = plan
        self._executed[request_id] = value
        return value

    def reject_execution(self, request_id: str, reason: str) -> ExecutionRejection:
        request = self._request(request_id)
        value = ExecutionRejection(
            request=request, plan=self._plans.get(request_id), reason=reason
        )
        self._terminal_once(request_id, value)
        return value

    def verify(self, request_id: str) -> VerifiedExample:
        if request_id in self._terminals or request_id in self._verified:
            raise ValueError("fresh transition verification was reused")
        try:
            execution = self._executed[request_id]
        except KeyError as error:
            raise ValueError("verification has no executed edit") from error
        value = VerifiedExample(execution=execution)
        self._verified[request_id] = value
        self._terminal_once(request_id, value)
        return value

    def reject_verification(
        self, request_id: str, reason: str
    ) -> VerificationRejection:
        try:
            execution = self._executed[request_id]
        except KeyError as error:
            raise ValueError("verification rejection has no executed edit") from error
        value = VerificationRejection(execution=execution, reason=reason)
        self._terminal_once(request_id, value)
        return value

    def require_verified(self, request_id: str) -> VerifiedExample:
        try:
            value = self._verified[request_id]
        except KeyError as error:
            raise ValueError(
                "accepted publication bypassed verified example"
            ) from error
        if self._terminal_for(request_id) is not value:
            raise ValueError("accepted publication bypassed verified example")
        return value

    def _terminal_for(
        self, request_id: str
    ) -> (
        PlanningRejection | ExecutionRejection | VerificationRejection | VerifiedExample
    ):
        try:
            return self._terminals[request_id]
        except KeyError as error:
            raise ValueError("fresh transition has no terminal") from error

    def _validate_terminal_closure(
        self, request_ids: tuple[str, ...]
    ) -> tuple[
        PlanningRejection
        | ExecutionRejection
        | VerificationRejection
        | VerifiedExample,
        ...,
    ]:
        if type(request_ids) is not tuple or any(
            type(request_id) is not str for request_id in request_ids
        ):
            raise TypeError("fresh transition terminal closure requires exact IDs")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("fresh transition terminal closure has duplicate requests")
        if set(request_ids) != set(self._terminals):
            raise ValueError(
                "fresh transition terminal closure does not cover requests"
            )
        return tuple(self._terminal_for(request_id) for request_id in request_ids)

    def _request(self, request_id: str) -> CounterfactualRequest:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise ValueError("fresh transition has no request") from error

    def _terminal_once(
        self,
        request_id: str,
        value: PlanningRejection
        | ExecutionRejection
        | VerificationRejection
        | VerifiedExample,
    ) -> None:
        if type(value) not in {
            PlanningRejection,
            ExecutionRejection,
            VerificationRejection,
            VerifiedExample,
        }:
            raise TypeError("fresh transition terminal must be exact")
        if request_id in self._terminals:
            raise ValueError("fresh transition has duplicate terminal")
        self._terminals[request_id] = value


__all__ = (
    "CertifiedPlan",
    "CounterfactualRequest",
    "ExecutedEdit",
    "ExecutionRejection",
    "PlanningRejection",
    "VerificationRejection",
    "VerifiedExample",
)
