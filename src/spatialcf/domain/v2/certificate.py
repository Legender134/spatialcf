"""Platform-neutral semantic certificate contracts for Canonical v2.

These immutable models bind the inputs and outputs of an independent pure-core
verification.  Constructing a model does not itself prove region coverage,
feasibility, loss bounds, or emptiness; the certificate verifier must resolve
the referenced hashes and recompute those claims.
"""

from __future__ import annotations

import math
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    NonNegativeFiniteFloat,
    PositiveFiniteFloat,
    SchemaIdentityV2,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_GLOBAL_CERTIFICATE_HASH_DOMAIN = "global-optimality-certificate-v2"
_UNSAT_CERTIFICATE_HASH_DOMAIN = "proven-unsat-certificate-v2"


class CertificateKindV2(StrEnum):
    GLOBAL_OPTIMUM = "GLOBAL_OPTIMUM"
    PROVEN_UNSAT = "PROVEN_UNSAT"


class OptimalityClaimV2(StrEnum):
    EXACT = "EXACT"
    EPSILON_OPTIMAL = "EPSILON_OPTIMAL"


class EmptyOuterProofMethodV2(StrEnum):
    """Closed pure-core method for a proven empty hard-domain outer bound."""

    CERTIFIED_EMPTY_OUTER_DOMAIN = "CERTIFIED_EMPTY_OUTER_DOMAIN"


def _directed_binary64_gap_ceil(lower_bound: float, upper_bound: float) -> float:
    exact_gap = Fraction.from_float(upper_bound) - Fraction.from_float(lower_bound)
    try:
        published = float(exact_gap)
    except OverflowError as error:
        raise ValueError("optimality gap must be finite binary64") from error
    if not math.isfinite(published):
        raise ValueError("optimality gap must be finite binary64")
    if Fraction.from_float(published) < exact_gap:
        published = math.nextafter(published, math.inf)
    if not math.isfinite(published):
        raise ValueError("optimality gap must be finite binary64")
    return published


class GlobalOptimalityCertificateV2(V2Model):
    """Hash-closed total-loss optimality claim emitted by the core verifier.

    ``optimality_gap`` is the tight binary64 upper bound on the exact difference
    between the two published binary64 loss bounds. Emitters must not use a
    nearest-rounded subtraction or a human-decimal approximation.
    """

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(
            schema_name="global-optimality-certificate"
        )
    )
    certificate_kind: Literal[CertificateKindV2.GLOBAL_OPTIMUM] = (
        CertificateKindV2.GLOBAL_OPTIMUM
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    candidate_domain_artifact_sha256: Sha256Digest
    relation_cost_partition_sha256: Sha256Digest
    objective_partition_artifact_sha256: Sha256Digest
    edit_sha256: Sha256Digest
    loss_lower_bound: NonNegativeFiniteFloat
    loss_upper_bound: NonNegativeFiniteFloat
    optimality_gap: NonNegativeFiniteFloat
    optimality_claim: OptimalityClaimV2
    epsilon: PositiveFiniteFloat | None = None

    @model_validator(mode="after")
    def validate_certificate(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(
            schema_name="global-optimality-certificate"
        ):
            raise ValueError("global certificate schema identity must be fixed")
        if self.loss_lower_bound > self.loss_upper_bound:
            raise ValueError("loss lower_bound must not exceed loss upper_bound")
        expected_gap = _directed_binary64_gap_ceil(
            self.loss_lower_bound,
            self.loss_upper_bound,
        )
        if self.optimality_gap != expected_gap:
            raise ValueError(
                "optimality gap must equal loss_upper_bound - loss_lower_bound"
            )

        if self.optimality_claim is OptimalityClaimV2.EXACT:
            if self.optimality_gap != 0.0 or self.epsilon is not None:
                raise ValueError(
                    "EXACT claim requires equal loss bounds, zero gap, and no epsilon"
                )
            return self

        if self.epsilon is None:
            raise ValueError("EPSILON_OPTIMAL claim requires an explicit epsilon")
        if self.optimality_gap > self.epsilon:
            raise ValueError("optimality gap must not exceed epsilon")
        return self

    @property
    def certificate_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_GLOBAL_CERTIFICATE_HASH_DOMAIN,
        )


class ProvenUnsatCertificateV2(V2Model):
    """Hash-closed claim that the complete hard-domain outer bound is empty."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="proven-unsat-certificate")
    )
    certificate_kind: Literal[CertificateKindV2.PROVEN_UNSAT] = (
        CertificateKindV2.PROVEN_UNSAT
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    candidate_domain_artifact_sha256: Sha256Digest
    empty_outer_proof_method: EmptyOuterProofMethodV2

    @model_validator(mode="after")
    def validate_certificate(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(
            schema_name="proven-unsat-certificate"
        ):
            raise ValueError("UNSAT certificate schema identity must be fixed")
        return self

    @property
    def certificate_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_UNSAT_CERTIFICATE_HASH_DOMAIN,
        )


SemanticCertificateV2: TypeAlias = Annotated[
    GlobalOptimalityCertificateV2 | ProvenUnsatCertificateV2,
    Field(discriminator="certificate_kind"),
]
