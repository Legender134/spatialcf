"""Unique generation transition values between protocol workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spatialcf.adapters.base import CapturedSource, SettledReadback
from spatialcf.domain.edit import CanonicalEdit
from spatialcf.domain.contrast import (
    NoncertifiedWitnessTerminal, PolicyRejectedTerminal, ProvenUnsatTerminal,
    PublishedPairTerminal, UnknownTerminal,
)
from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.request import InterventionSpec

_TransitionProfile = Literal["v2", "semantic"]


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


@dataclass(frozen=True, slots=True, eq=False)
class _SemanticMember:
    """Invocation-local identity for one frozen M4 catalog member."""

    candidate_id: str
    candidate_sha256: str


@dataclass(frozen=True, slots=True)
class _SemanticTerminal:
    """Invocation-local carrier for one exact M4 terminal object."""

    member: _SemanticMember
    value: HashBoundCanonicalModel


class _FreshTransitions:
    """Explicit, process-local transition carrier for one fresh dataset root."""

    def __init__(self, *, profile: _TransitionProfile = "v2") -> None:
        if profile not in {"v2", "semantic"}:
            raise ValueError("fresh transition profile is unsupported")
        self._profile = profile
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
        self._semantic_members: dict[str, _SemanticMember] = {}
        self._semantic_terminals: dict[str, _SemanticTerminal] = {}
        self._semantic_carriers: set[int] = set()
        self._semantic_started = False

    def register_semantic(
        self, candidate_id: str, candidate_sha256: str
    ) -> _SemanticMember:
        """Register the complete M4 universe before any member is solved."""

        self._require_profile("semantic")
        if self._semantic_started:
            raise ValueError("semantic registration is closed after processing begins")
        if type(candidate_id) is not str or type(candidate_sha256) is not str:
            raise TypeError("semantic transition registration requires exact strings")
        if len(candidate_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in candidate_sha256
        ):
            raise ValueError("semantic transition candidate digest is invalid")
        if candidate_id in self._semantic_members:
            raise ValueError("fresh semantic transition reused a catalog member")
        value = _SemanticMember(candidate_id, candidate_sha256)
        self._semantic_members[candidate_id] = value
        return value

    def begin_semantic(self, candidate_ids: tuple[str, ...]) -> None:
        self._require_profile("semantic")
        if self._semantic_started or candidate_ids != tuple(self._semantic_members):
            raise ValueError("semantic processing requires the exact registered universe once")
        self._semantic_started = True

    def terminal_semantic(
        self, member: _SemanticMember, value: HashBoundCanonicalModel
    ) -> _SemanticTerminal:
        """Bind one and only one immutable terminal to a registered member."""

        self._require_profile("semantic")
        if not self._semantic_started:
            raise ValueError("semantic processing has not begun")
        if type(member) is not _SemanticMember or self._semantic_members.get(member.candidate_id) is not member:
            raise ValueError("semantic terminal has no invocation-local member")
        candidate_id = member.candidate_id
        if candidate_id in self._semantic_terminals:
            raise ValueError("fresh semantic transition has duplicate terminal")
        if type(value) not in {
            PublishedPairTerminal, ProvenUnsatTerminal, UnknownTerminal,
            NoncertifiedWitnessTerminal, PolicyRejectedTerminal,
        }:
            raise TypeError("fresh semantic terminal requires an exact terminal type")
        if value.candidate_id != candidate_id:
            raise ValueError("semantic terminal binds the wrong candidate")
        if id(value) in self._semantic_carriers:
            raise ValueError("fresh semantic transition reused a terminal carrier")
        type(value).model_validate(
            value.model_dump(mode="python", warnings="error", round_trip=True),
            strict=True,
        )
        terminal = _SemanticTerminal(member, value)
        self._semantic_terminals[candidate_id] = terminal
        self._semantic_carriers.add(id(value))
        return terminal

    def validate_semantic_closure(
        self, candidate_ids: tuple[str, ...]
    ) -> tuple[_SemanticTerminal, ...]:
        """Return terminals in frozen catalog order after exact coverage checks."""

        self._require_profile("semantic")
        if not self._semantic_started:
            raise ValueError("semantic processing has not begun")
        if type(candidate_ids) is not tuple or any(
            type(candidate_id) is not str for candidate_id in candidate_ids
        ):
            raise TypeError("semantic terminal closure requires exact IDs")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("semantic terminal closure has duplicate candidates")
        if candidate_ids != tuple(self._semantic_members):
            raise ValueError(
                "semantic terminal closure differs from registered universe"
            )
        if set(candidate_ids) != set(self._semantic_terminals):
            raise ValueError("semantic terminal closure does not cover catalog")
        return tuple(self._semantic_terminals[item] for item in candidate_ids)

    def _require_profile(self, expected: _TransitionProfile) -> None:
        if self._profile != expected:
            raise RuntimeError("fresh transition operation used the wrong profile")

    def capture(self, source_id: str, source: CapturedSource) -> None:
        self._require_profile("v2")
        if type(source_id) is not str or type(source) is not CapturedSource:
            raise TypeError("fresh capture carrier requires exact source values")
        if source_id in self._sources:
            raise ValueError("fresh capture carrier reused a source")
        self._sources[source_id] = source

    def request(
        self, request_id: str, source_id: str, intervention: InterventionSpec
    ) -> CounterfactualRequest:
        self._require_profile("v2")
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
        self._require_profile("v2")
        value = PlanningRejection(request=self._request(request_id), reason=reason)
        self._terminal_once(request_id, value)
        return value

    def executed(
        self, request_id: str, edit: CanonicalEdit, readback: SettledReadback
    ) -> ExecutedEdit:
        self._require_profile("v2")
        request = self._request(request_id)
        if request_id in self._plans or request_id in self._terminals:
            raise ValueError("fresh transition plan was reused")
        plan = CertifiedPlan(request=request, edit=edit)
        value = ExecutedEdit(plan=plan, readback=readback)
        self._plans[request_id] = plan
        self._executed[request_id] = value
        return value

    def reject_execution(self, request_id: str, reason: str) -> ExecutionRejection:
        self._require_profile("v2")
        request = self._request(request_id)
        value = ExecutionRejection(
            request=request, plan=self._plans.get(request_id), reason=reason
        )
        self._terminal_once(request_id, value)
        return value

    def verify(self, request_id: str) -> VerifiedExample:
        self._require_profile("v2")
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
        self._require_profile("v2")
        try:
            execution = self._executed[request_id]
        except KeyError as error:
            raise ValueError("verification rejection has no executed edit") from error
        value = VerificationRejection(execution=execution, reason=reason)
        self._terminal_once(request_id, value)
        return value

    def require_verified(self, request_id: str) -> VerifiedExample:
        self._require_profile("v2")
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
        self._require_profile("v2")
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
