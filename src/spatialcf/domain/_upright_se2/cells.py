"""Upright profile cells contracts and intrinsic operations."""

from __future__ import annotations

from enum import (
    StrEnum,
)

from fractions import (
    Fraction,
)

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    CapabilityRef,
    HashBoundCanonicalModel,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
    UPRIGHT_SE2_STAGED_CAPABILITY_REFS,
)

from spatialcf.domain._upright_se2.state import (
    UprightSE2TranslationDomain,
)

from spatialcf.domain._upright_se2.yaw import (
    CardinalYawAuthorization,
    ContinuousYawAuthorization,
    ContinuousYawLift,
    ExactDyadic,
    LiftedYawInterval,
    _require_sorted_unique_by_bytes,
)


class UprightSE2BackendAvailability(HashBoundCanonicalModel):
    """Operational availability is separate from the immutable profile hash."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/backend-availability/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "backend_availability_sha256"

    profile_registration_sha256: Sha256Digest
    available_capability_refs: tuple[CapabilityRef, ...]
    backend_availability_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_staged_availability(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.available_capability_refs,
            "available staged capabilities",
        )
        if self.available_capability_refs not in (
            UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
            UPRIGHT_SE2_STAGED_CAPABILITY_REFS,
        ):
            raise ValueError(
                "availability must be cardinal-only or cardinal-plus-continuous"
            )
        return self


class UprightSE2CompiledCell(HashBoundCanonicalModel):
    """A declarative exact-dyadic M3 cell, not a compiled solver implementation."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/compiled-cell/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "compiled_cell_sha256"

    cell_id: CanonicalId
    authorization_sha256: Sha256Digest
    x_lower: ExactDyadic
    x_upper: ExactDyadic
    y_lower: ExactDyadic
    y_upper: ExactDyadic
    yaw_interval: LiftedYawInterval
    compiled_cell_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_cell_bounds(self) -> Self:
        if not self.cell_id.startswith("cell:"):
            raise ValueError("compiled cell IDs must use the cell namespace")
        if self.x_lower.as_fraction > self.x_upper.as_fraction:
            raise ValueError("compiled cell x bounds must be ordered")
        if self.y_lower.as_fraction > self.y_upper.as_fraction:
            raise ValueError("compiled cell y bounds must be ordered")
        return self


def _lifted_cell_order_key(
    cell: UprightSE2CompiledCell,
) -> tuple[Fraction, Fraction, bytes]:
    """Return the frozen lifted-yaw order with a canonical cell-ID tie-break."""

    return (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        canonical_json_bytes(cell.cell_id),
    )


class UprightSE2CoverageArtifact(HashBoundCanonicalModel):
    """Canonical roster/coverage data that remains untrusted until checker replay."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/coverage-artifact/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "coverage_artifact_sha256"

    authorization_sha256: Sha256Digest
    cells: tuple[UprightSE2CompiledCell, ...]
    unresolved_cell_sha256s: tuple[Sha256Digest, ...]
    coverage_artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_coverage_roster(self) -> Self:
        if not self.cells:
            raise ValueError("coverage artifact must name at least one compiled cell")
        if self.cells != tuple(sorted(self.cells, key=_lifted_cell_order_key)):
            raise ValueError("coverage cells must use ascending exact lifted yaw order")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("coverage artifact must not duplicate a cell ID")
        if self.unresolved_cell_sha256s != tuple(
            sorted(set(self.unresolved_cell_sha256s))
        ):
            raise ValueError("unresolved cell digests must be sorted and unique")
        if any(
            cell.authorization_sha256 != self.authorization_sha256
            for cell in self.cells
        ):
            raise ValueError("coverage cells must bind the same authorization")
        return self


class UprightSE2ProofLeafDisposition(StrEnum):
    """The closed leaf status a later checker must replay without relabelling."""

    INWARD_FEASIBLE = "INWARD_FEASIBLE"
    OUTWARD_INFEASIBLE = "OUTWARD_INFEASIBLE"
    PRUNED = "PRUNED"
    UNRESOLVED = "UNRESOLVED"


class UprightSE2RetainedOwnerOutcomeKind(StrEnum):
    """The finite retained-owner result alphabet carried by one proof row."""

    EXACT = "EXACT"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"
    INCOMPLETE = "INCOMPLETE"
    FINITE_MISS = "FINITE_MISS"


class UprightSE2CardinalProofTuple(HashBoundCanonicalModel):
    """The one request-authorized operator/reference/pivot/q proof tuple."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/cardinal-proof-tuple/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "cardinal_proof_tuple_sha256"

    authorization: CardinalYawAuthorization
    reference_id: CanonicalId
    translation_domain: UprightSE2TranslationDomain
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    cardinal_proof_tuple_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_tuple_cells(self) -> Self:
        if not self.compiled_cells:
            raise ValueError("proof tuple must retain at least one compiled cell")
        if tuple(cell.authorization_sha256 for cell in self.compiled_cells) != (
            self.authorization.cardinal_yaw_authorization_sha256,
        ) * len(self.compiled_cells):
            raise ValueError("proof tuple cells must bind its authorization")
        if self.compiled_cells != tuple(
            sorted(self.compiled_cells, key=_lifted_cell_order_key)
        ):
            raise ValueError("proof tuple cells must use canonical lifted-yaw order")
        if len({cell.cell_id for cell in self.compiled_cells}) != len(
            self.compiled_cells
        ):
            raise ValueError("proof tuple must not duplicate a compiled cell")
        return self


class UprightSE2ContinuousProofTuple(HashBoundCanonicalModel):
    """The one request-authorized continuous operator/reference/lift tuple.

    Continuous search may subdivide this root, but it must never add another
    operator, pivot, reference, translation domain, or lifted yaw domain.  The
    record intentionally mirrors the cardinal tuple without widening it.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-proof-tuple/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_proof_tuple_sha256"

    authorization: ContinuousYawAuthorization
    reference_id: CanonicalId
    translation_domain: UprightSE2TranslationDomain
    continuous_yaw_lift: ContinuousYawLift
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    continuous_proof_tuple_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_tuple(self) -> Self:
        if len(self.compiled_cells) != 1:
            raise ValueError("continuous proof tuple requires one lift root")
        root = self.compiled_cells[0]
        interval = self.continuous_yaw_lift.intervals[0]
        if (
            root.authorization_sha256
            != self.authorization.continuous_yaw_authorization_sha256
            or root.x_lower != self.translation_domain.x_lower
            or root.x_upper != self.translation_domain.x_upper
            or root.y_lower != self.translation_domain.y_lower
            or root.y_upper != self.translation_domain.y_upper
            or root.yaw_interval != interval
            or self.continuous_yaw_lift.yaw_domain != self.authorization.yaw_domain
        ):
            raise ValueError("continuous proof tuple root does not bind its authority")
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2BackendAvailability.model_rebuild()
UprightSE2CompiledCell.model_rebuild()
UprightSE2CoverageArtifact.model_rebuild()
UprightSE2CardinalProofTuple.model_rebuild()
UprightSE2ContinuousProofTuple.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2BackendAvailability.__module__ = "spatialcf.domain.upright_se2"
UprightSE2CompiledCell.__module__ = "spatialcf.domain.upright_se2"
_lifted_cell_order_key.__module__ = "spatialcf.domain.upright_se2"
UprightSE2CoverageArtifact.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofLeafDisposition.__module__ = "spatialcf.domain.upright_se2"
UprightSE2RetainedOwnerOutcomeKind.__module__ = "spatialcf.domain.upright_se2"
UprightSE2CardinalProofTuple.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousProofTuple.__module__ = "spatialcf.domain.upright_se2"
