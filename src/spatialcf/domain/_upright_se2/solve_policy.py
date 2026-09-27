"""Upright profile solve policy contracts and intrinsic operations."""

from __future__ import annotations

import math

from fractions import (
    Fraction,
)

from typing import (
    Annotated,
    ClassVar,
    Self,
)

from pydantic import (
    Field,
    StrictInt,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    CanonicalIdValue,
    CapabilityRef,
    DefinitionBundle,
    DefinitionRef,
    DigestValue,
    HashBoundCanonicalModel,
    IntegerValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.profiles import (
    OwnerRef,
    ProofPolicy,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
    UPRIGHT_SE2_CHECKER_BUILD_SHA256,
    UPRIGHT_SE2_CHECKER_OWNER_REF,
    UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF,
    UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
    UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF,
    UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF,
    _UPRIGHT_SE2_DIGEST_SCHEMA_REF,
    _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
    _UPRIGHT_SE2_ID_SCHEMA_REF,
    _UPRIGHT_SE2_INTEGER_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.registration import (
    UprightSE2ProfileRegistration,
)

from spatialcf.domain._upright_se2.values import (
    _materialized_state_record,
)


class UprightSE2ExactRational(CanonicalModel):
    """One normalized exact rational retained in profile proof transport."""

    numerator: StrictInt
    denominator: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def _validate_normalized_rational(self) -> Self:
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("exact rational values must be normalized")
        return self

    @property
    def as_fraction(self) -> Fraction:
        """Return the sole exact arithmetic representation for profile checks."""

        return Fraction(self.numerator, self.denominator)


class UprightSE2SolvePolicyDefinitionPayload(CanonicalModel):
    """Request-owned finite-gap and claim-strength policy for M3 replay."""

    profile_registration_sha256: Sha256Digest
    requested_gap: UprightSE2ExactRational
    objective_bound_policy_ref: DefinitionRef
    exact_global_claim_definition_ref: DefinitionRef
    finite_gap_claim_definition_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_requested_gap_and_claims(self) -> Self:
        if self.requested_gap.as_fraction < 0:
            raise ValueError("requested gap must be non-negative")
        if (
            self.exact_global_claim_definition_ref
            == self.finite_gap_claim_definition_ref
        ):
            raise ValueError("exact and finite-gap claims must be distinct")
        return self


def _solve_policy_rational_value(value: UprightSE2ExactRational) -> TypedValue:
    return _materialized_state_record(
        _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
        (
            (
                "denominator",
                TypedValue(
                    value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                    payload=IntegerValue(value=value.denominator),
                ),
            ),
            (
                "numerator",
                TypedValue(
                    value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                    payload=IntegerValue(value=value.numerator),
                ),
            ),
        ),
    )


def build_upright_se2_solve_policy_definition_bundle(
    registration: UprightSE2ProfileRegistration,
    *,
    requested_gap: UprightSE2ExactRational,
    objective_bound_policy_ref: DefinitionRef,
    exact_global_claim_definition_ref: DefinitionRef = UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF,
    finite_gap_claim_definition_ref: DefinitionRef = UPRIGHT_SE2_FINITE_GAP_CLAIM_DEFINITION_REF,
) -> DefinitionBundle:
    """Seal the request's exact gap and claim policy into its definition closure."""

    if type(registration) is not UprightSE2ProfileRegistration:
        raise TypeError("solve policy requires an exact upright se2 registration")
    if type(requested_gap) is not UprightSE2ExactRational:
        raise TypeError("solve policy requested gap requires an exact rational")
    payload = UprightSE2SolvePolicyDefinitionPayload(
        profile_registration_sha256=registration.profile_registration_sha256,
        requested_gap=requested_gap,
        objective_bound_policy_ref=objective_bound_policy_ref,
        exact_global_claim_definition_ref=exact_global_claim_definition_ref,
        finite_gap_claim_definition_ref=finite_gap_claim_definition_ref,
    )
    fields = (
        (
            "exact_global_claim_definition_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(
                    value=payload.exact_global_claim_definition_ref
                ),
            ),
        ),
        (
            "finite_gap_claim_definition_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(value=payload.finite_gap_claim_definition_ref),
            ),
        ),
        (
            "objective_bound_policy_ref",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_ID_SCHEMA_REF,
                payload=CanonicalIdValue(value=payload.objective_bound_policy_ref),
            ),
        ),
        (
            "profile_registration_sha256",
            TypedValue(
                value_schema_ref=_UPRIGHT_SE2_DIGEST_SCHEMA_REF,
                payload=DigestValue(value=payload.profile_registration_sha256),
            ),
        ),
        ("requested_gap", _solve_policy_rational_value(payload.requested_gap)),
    )
    return DefinitionBundle.seal(
        definitions=(
            CanonicalDefinitionEnvelope.seal(
                definition_ref=UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF,
                definition_kind_ref="definition:spatialcf/upright-se2/definition-kind/1.0",
                payload_schema_ref=UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF,
                payload=_materialized_state_record(
                    UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF, fields
                ),
            ),
        )
    )


def decode_upright_se2_solve_policy_definition_payload(
    bundle: DefinitionBundle,
) -> UprightSE2SolvePolicyDefinitionPayload:
    """Decode only the registered canonical request-owned solve-policy payload."""

    if type(bundle) is not DefinitionBundle or len(bundle.definitions) != 1:
        raise ValueError("solve policy bundle must contain one registered definition")
    definition = bundle.definitions[0]
    if (
        definition.definition_ref != UPRIGHT_SE2_SOLVE_POLICY_DEFINITION_REF
        or definition.payload_schema_ref != UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF
        or definition.payload.value_schema_ref
        != UPRIGHT_SE2_SOLVE_POLICY_PAYLOAD_SCHEMA_REF
        or type(definition.payload.payload) is not RecordValue
    ):
        raise ValueError("solve policy bundle has the wrong registered payload")
    fields = {field.name: field.value for field in definition.payload.payload.fields}
    if set(fields) != {
        "exact_global_claim_definition_ref",
        "finite_gap_claim_definition_ref",
        "objective_bound_policy_ref",
        "profile_registration_sha256",
        "requested_gap",
    }:
        raise ValueError("solve policy payload fields are not closed")
    profile = fields["profile_registration_sha256"]
    requested_gap = fields["requested_gap"]
    if (
        type(profile.payload) is not DigestValue
        or type(requested_gap.payload) is not RecordValue
    ):
        raise ValueError("solve policy payload has invalid scalar forms")
    gap_fields = {field.name: field.value for field in requested_gap.payload.fields}
    if set(gap_fields) != {"numerator", "denominator"} or any(
        type(gap_fields[name].payload) is not IntegerValue
        for name in ("numerator", "denominator")
    ):
        raise ValueError("solve policy requested gap is not an exact rational")

    def canonical_id(name: str) -> str:
        value = fields[name]
        if type(value.payload) is not CanonicalIdValue:
            raise ValueError("solve policy reference is not canonical")
        return value.payload.value

    return UprightSE2SolvePolicyDefinitionPayload(
        profile_registration_sha256=profile.payload.value,
        requested_gap=UprightSE2ExactRational(
            numerator=gap_fields["numerator"].payload.value,
            denominator=gap_fields["denominator"].payload.value,
        ),
        objective_bound_policy_ref=canonical_id("objective_bound_policy_ref"),
        exact_global_claim_definition_ref=canonical_id(
            "exact_global_claim_definition_ref"
        ),
        finite_gap_claim_definition_ref=canonical_id("finite_gap_claim_definition_ref"),
    )


class UprightSE2CheckerReplayPolicy(HashBoundCanonicalModel):
    """The explicit fresh-policy root consumed by the cardinal checker only."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/checker-replay-policy/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "checker_replay_policy_sha256"

    proof_policy_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    requested_gap: UprightSE2ExactRational
    objective_bound_policy_ref: DefinitionRef
    exact_global_claim_definition_ref: DefinitionRef
    finite_gap_claim_definition_ref: DefinitionRef
    checker_owner_ref: OwnerRef
    checker_capability_ref: CapabilityRef
    checker_build_sha256: Sha256Digest
    checker_replay_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_checker_identity(self) -> Self:
        if (
            self.checker_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF
            or self.checker_capability_ref
            != UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
            or self.checker_build_sha256 != UPRIGHT_SE2_CHECKER_BUILD_SHA256
        ):
            raise ValueError("checker replay policy must bind the registered checker")
        return self


def build_upright_se2_checker_replay_policy(
    proof_policy: ProofPolicy,
    solve_policy_definition_bundle: DefinitionBundle,
) -> UprightSE2CheckerReplayPolicy:
    """Bind one fresh M1 proof policy to the static M3 checker identity."""

    if type(proof_policy) is not ProofPolicy:
        raise TypeError("checker replay policy requires an exact ProofPolicy")
    solve_policy = decode_upright_se2_solve_policy_definition_payload(
        solve_policy_definition_bundle
    )
    return UprightSE2CheckerReplayPolicy.seal(
        proof_policy_sha256=proof_policy.proof_policy_sha256,
        solve_policy_definition_bundle_sha256=(
            solve_policy_definition_bundle.definition_bundle_sha256
        ),
        requested_gap=solve_policy.requested_gap,
        objective_bound_policy_ref=solve_policy.objective_bound_policy_ref,
        exact_global_claim_definition_ref=(
            solve_policy.exact_global_claim_definition_ref
        ),
        finite_gap_claim_definition_ref=solve_policy.finite_gap_claim_definition_ref,
        checker_owner_ref=UPRIGHT_SE2_CHECKER_OWNER_REF,
        checker_capability_ref=UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        checker_build_sha256=UPRIGHT_SE2_CHECKER_BUILD_SHA256,
    )


# Resolve local model forward references before restoring public identities.
UprightSE2ExactRational.model_rebuild()
UprightSE2SolvePolicyDefinitionPayload.model_rebuild()
UprightSE2CheckerReplayPolicy.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2ExactRational.__module__ = "spatialcf.domain.upright_se2"
UprightSE2SolvePolicyDefinitionPayload.__module__ = "spatialcf.domain.upright_se2"
_solve_policy_rational_value.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_solve_policy_definition_bundle.__module__ = "spatialcf.domain.upright_se2"
decode_upright_se2_solve_policy_definition_payload.__module__ = "spatialcf.domain.upright_se2"
UprightSE2CheckerReplayPolicy.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_checker_replay_policy.__module__ = "spatialcf.domain.upright_se2"
