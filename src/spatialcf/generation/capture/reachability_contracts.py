"""Reachability evidence contracts; independent of capture and planning orchestration."""

from __future__ import annotations

from collections.abc import (
    Iterable,
)

from enum import (
    StrEnum,
)

from fractions import (
    Fraction,
)

from typing import (
    Literal,
    Self,
)

from pydantic import (
    Field,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.request import (
    Relation,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)


_REACHABLE_POSITIONS_HASH_DOMAIN_V2_9_4 = (
    "spatialcf.competition-native-reachable-native-positions.v2.9.4"
)


_TARGET_REACHABILITY_HASH_DOMAIN_V2_9_4 = (
    "spatialcf.competition-native-candidate-target-reachability.v2.9.4"
)


class CompetitionNativeTargetReachabilityStatusV2_9_4(StrEnum):
    REACHABLE = "REACHABLE_NATIVE_TARGET"
    UNREACHABLE = "NO_REACHABLE_NATIVE_TARGET"


def _canonical_coordinate_payload_v2_9_4(
    coordinates: Iterable[tuple[Fraction, Fraction]],
) -> tuple[tuple[str, str], ...]:
    checked = tuple(coordinates)
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not Fraction
        or type(item[1]) is not Fraction
        for item in checked
    ):
        raise TypeError("reachable native coordinates must contain exact fractions")
    return tuple(sorted({(str(x), str(y)) for x, y in checked}))


def competition_native_reachable_native_positions_sha256_v2_9_4(
    coordinates: Iterable[tuple[Fraction, Fraction]],
) -> Sha256Digest:
    """Hash one canonical set of target-reachable native XY deltas."""

    return canonical_sha256(
        _canonical_coordinate_payload_v2_9_4(coordinates),
        domain=_REACHABLE_POSITIONS_HASH_DOMAIN_V2_9_4,
    )


class CompetitionNativeCandidateTargetReachabilityV2_9_4(CanonicalModel):
    reachability_version: Literal[
        "competition-native-candidate-target-reachability:2.9.4"
    ] = "competition-native-candidate-target-reachability:2.9.4"
    authority: Literal[
        "SOURCE_ONLY_CANONICAL_TARGET_INNER_NOT_SOLVER_OR_NATIVE_OUTCOME"
    ] = "SOURCE_ONLY_CANONICAL_TARGET_INNER_NOT_SOLVER_OR_NATIVE_OUTCOME"
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_capture_sha256: Sha256Digest
    subject_id: str = Field(strict=True, min_length=1, max_length=512)
    reference_id: str = Field(strict=True, min_length=1, max_length=512)
    relation_before: Relation
    relation_after: Relation
    placement_sha256: Sha256Digest
    surface_evidence_sha256: Sha256Digest
    subject_surface_evidence_sha256: Sha256Digest
    camera_evidence_sha256: Sha256Digest
    native_position_count: int = Field(strict=True, ge=1, le=10_000)
    reachable_native_position_count: int = Field(strict=True, ge=0, le=10_000)
    reachable_native_positions_sha256: Sha256Digest
    status: CompetitionNativeTargetReachabilityStatusV2_9_4
    target_reachability_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_reachability(self) -> Self:
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("target reachability relation pair is not opposite")
        if self.reachable_native_position_count > self.native_position_count:
            raise ValueError("reachable native position count exceeds source count")
        reachable = self.reachable_native_position_count > 0
        if reachable != (
            self.status is CompetitionNativeTargetReachabilityStatusV2_9_4.REACHABLE
        ):
            raise ValueError("target reachability terminal status is not closed")
        payload = self.model_dump(
            mode="python",
            exclude={"target_reachability_sha256"},
            warnings="error",
        )
        if self.target_reachability_sha256 != canonical_sha256(
            payload,
            domain=_TARGET_REACHABILITY_HASH_DOMAIN_V2_9_4,
        ):
            raise ValueError("target reachability digest mismatch")
        return self


CandidateTargetReachability = CompetitionNativeCandidateTargetReachabilityV2_9_4


TargetReachabilityStatus = CompetitionNativeTargetReachabilityStatusV2_9_4


# Resolve model annotations before restoring the public type identity.
CompetitionNativeCandidateTargetReachabilityV2_9_4.model_rebuild()


# Preserve supported public names and pickle lookup.
CompetitionNativeTargetReachabilityStatusV2_9_4.__module__ = "spatialcf.generation.capture.reachability"
_canonical_coordinate_payload_v2_9_4.__module__ = "spatialcf.generation.capture.reachability"
competition_native_reachable_native_positions_sha256_v2_9_4.__module__ = "spatialcf.generation.capture.reachability"
CompetitionNativeCandidateTargetReachabilityV2_9_4.__module__ = "spatialcf.generation.capture.reachability"
