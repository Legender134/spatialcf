"""Capture contracts: exact contracts and pure derivation."""

from __future__ import annotations

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

from spatialcf.domain.scene import (
    OBB,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from enum import (
    StrEnum,
)

from spatialcf.domain.scene import (
    SubjectPositionRegion,
    Vec2,
)

from spatialcf.generation.capture._models.camera_contracts import (
    CameraPolicy,
)

from spatialcf.generation.capture._models.constants import (
    DatasetSplitV2_9,
    _CURRENT_POLICY_HASH_DOMAIN,
    _MAX_CANDIDATES_TOTAL,
    _MAX_NATIVE_POSITIONS,
    _MAX_OBJECTS_PER_SCENE,
    _MAX_POLICY_SOURCES,
    _MAX_POLYGON_VERTICES,
    _MAX_REASONS,
    _MAX_REASON_CHARS,
    _MAX_REQUESTS_TOTAL,
    _MAX_TEXT_CHARS,
    _PLACEMENT_HASH_DOMAIN,
)


class CompetitionNativeSupportKindV2_9(StrEnum):
    FLOOR = "FLOOR"
    RECEPTACLE = "RECEPTACLE"
    UNKNOWN = "UNKNOWN"
    MULTIPLE_AMBIGUOUS = "MULTIPLE_AMBIGUOUS"
    CYCLIC = "CYCLIC"


class CompetitionNativePlacementAvailabilityV2_9(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    KNOWN_RECEPTACLE_SPAWN = "KNOWN_RECEPTACLE_SPAWN"
    KNOWN_FLOOR_INNER_REGION = "KNOWN_FLOOR_INNER_REGION"
    MISSING = "MISSING"


class CompetitionNativeSubjectStateV2_9(StrEnum):
    ELIGIBLE_RECEPTACLE_DOMAIN = "ELIGIBLE_RECEPTACLE_DOMAIN"
    ELIGIBLE_FLOOR_INNER_DOMAIN = "ELIGIBLE_FLOOR_INNER_DOMAIN"
    REJECTED_NOT_MOVABLE = "REJECTED_NOT_MOVABLE"
    REJECTED_NOT_REQUEST_ELIGIBLE = "REJECTED_NOT_REQUEST_ELIGIBLE"
    REJECTED_PINNED = "REJECTED_PINNED"
    REJECTED_HAS_DEPENDENT_CHILD = "REJECTED_HAS_DEPENDENT_CHILD"
    REJECTED_UNKNOWN_SUPPORT = "REJECTED_UNKNOWN_SUPPORT"
    REJECTED_MISSING_PLACEMENT_DOMAIN = "REJECTED_MISSING_PLACEMENT_DOMAIN"
    REJECTED_INVALID_SOURCE_FACT = "REJECTED_INVALID_SOURCE_FACT"


class CompetitionNativeCandidateStateV2_9(StrEnum):
    SELECTED = "SELECTED"
    REJECTED_SUBJECT = "REJECTED_SUBJECT"
    REJECTED_REFERENCE = "REJECTED_REFERENCE"
    REJECTED_DEPENDENCY = "REJECTED_DEPENDENCY"
    REJECTED_NOT_VISIBLE = "REJECTED_NOT_VISIBLE"
    REJECTED_AMBIGUOUS = "REJECTED_AMBIGUOUS"
    REJECTED_RELATION_NOT_OBSERVED = "REJECTED_RELATION_NOT_OBSERVED"
    REJECTED_POLICY_CAP = "REJECTED_POLICY_CAP"
    REJECTED_TARGET_UNREACHABLE = "REJECTED_TARGET_UNREACHABLE"


class CompetitionNativeSourceRefV2_9(CanonicalModel):
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    split: DatasetSplitV2_9
    source_locator_sha256: Sha256Digest = Field(
        description=(
            "Digest of the frozen locator record; this is distinct from a "
            "procedural runtime's source content digest."
        )
    )


class RosterPolicy(CanonicalModel):
    """The only supported candidate-roster policy."""

    policy_version: Literal["competition-native-candidate-roster-policy:2.9.4"] = (
        "competition-native-candidate-roster-policy:2.9.4"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    seed: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    width: int = Field(strict=True, gt=0, le=4096)
    height: int = Field(strict=True, gt=0, le=4096)
    max_scenes: int = Field(strict=True, gt=0, le=_MAX_POLICY_SOURCES)
    max_objects_per_scene: int = Field(strict=True, gt=0, le=_MAX_OBJECTS_PER_SCENE)
    max_candidates_total: int = Field(strict=True, gt=0, le=_MAX_CANDIDATES_TOTAL)
    max_requests_total: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_scene: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_subject: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    max_requests_per_category_pair: int = Field(
        strict=True, gt=0, le=_MAX_REQUESTS_TOTAL
    )
    target_requests_per_relation: int = Field(strict=True, gt=0, le=_MAX_REQUESTS_TOTAL)
    sources: tuple[CompetitionNativeSourceRefV2_9, ...] = Field(
        min_length=1, max_length=_MAX_POLICY_SOURCES
    )
    require_scene_unique_referents: Literal[True] = True
    require_receptacle_surface_evidence: Literal[True] = True
    require_source_camera_evidence: Literal[True] = True
    camera_policy: CameraPolicy
    require_receptacle_native_target_reachability: Literal[True] = True

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.width * self.height > 4_194_304:
            raise ValueError("candidate roster render dimensions exceed pixel limit")
        source_ids = tuple(item.source_id for item in self.sources)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError("candidate roster sources must use canonical unique order")
        if len(self.sources) > self.max_scenes:
            raise ValueError("candidate roster source count exceeds max_scenes")
        return self

    @property
    def policy_sha256(self) -> Sha256Digest:
        return canonical_sha256(self, domain=_CURRENT_POLICY_HASH_DOMAIN)


CompetitionNativeCandidateRosterPolicyV2_9_4 = RosterPolicy


class CompetitionNativeRuntimeIdentityV2_9(CanonicalModel):
    ai2thor_version: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    unity_commit_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_scene_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_TEXT_CHARS
    )
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    seed: int = Field(strict=True)
    render_depth_image: bool
    render_instance_segmentation: bool
    grid_size_m: float
    snap_to_grid: bool
    rotate_step_degrees: int = Field(strict=True, gt=0)
    coordinate_transform_version: str = Field(
        strict=True, min_length=1, max_length=_MAX_TEXT_CHARS
    )
    source_dataset_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_revision: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_split: Literal["train", "val", "validation", "test"] | None
    source_index: int | None = Field(default=None, ge=0)
    source_sha256: Sha256Digest | None = Field(
        description=(
            "Digest of actual procedural source content, independent of the "
            "frozen source locator digest."
        )
    )
    source_scene_alias: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_loader_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_loader_version: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_room_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    source_floor_xz_bounds: tuple[float, float, float, float] | None
    teleport_vertical_guard_m: float


def validate_competition_native_runtime_source_lineage_v2_9(
    source: CompetitionNativeSourceRefV2_9,
    runtime: CompetitionNativeRuntimeIdentityV2_9,
) -> None:
    """Bind legacy/procedural runtime identity to one frozen source locator."""

    runtime_split = (
        "validation" if runtime.source_split == "val" else runtime.source_split
    )
    if runtime_split is not None and runtime_split != source.split:
        raise ValueError("source capture runtime split does not match frozen source")
    if runtime.native_scene_name == "Procedural":
        if (
            runtime.source_scene_alias != source.scene_id
            or runtime.source_sha256 is None
            or runtime.source_split is None
        ):
            raise ValueError(
                "procedural runtime scene alias does not match frozen source"
            )
    elif runtime.native_scene_name not in {
        source.scene_id,
        f"{source.scene_id}_physics",
    }:
        raise ValueError("legacy runtime scene name does not match frozen source")


class CompetitionNativeSupportFactV2_9(CanonicalModel):
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    raw_parent_object_ids: tuple[str, ...] = Field(max_length=_MAX_OBJECTS_PER_SCENE)
    structural_parent_object_ids: tuple[str, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    domain_parent_object_ids: tuple[str, ...] = Field(max_length=_MAX_OBJECTS_PER_SCENE)
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None
    floor_object_id: str | None

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        for values in (
            self.raw_parent_object_ids,
            self.structural_parent_object_ids,
            self.domain_parent_object_ids,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("captured support parent IDs are not canonical")
            if any(not value or len(value) > _MAX_TEXT_CHARS for value in values):
                raise ValueError("captured support parent ID exceeds persisted limit")
        if any(
            value is not None and len(value) > _MAX_TEXT_CHARS
            for value in (self.support_object_id, self.floor_object_id)
        ):
            raise ValueError("captured support target exceeds persisted limit")
        raw = set(self.raw_parent_object_ids)
        structural = set(self.structural_parent_object_ids)
        domain = set(self.domain_parent_object_ids)
        if not structural.issubset(raw) or not domain.issubset(raw):
            raise ValueError("captured support partitions escape raw parents")
        if structural.intersection(domain):
            raise ValueError("captured support partitions overlap")
        kind = self.support_kind
        if kind is CompetitionNativeSupportKindV2_9.RECEPTACLE:
            valid = (
                len(domain) == 1
                and not structural
                and self.support_object_id == self.domain_parent_object_ids[0]
                and self.floor_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.FLOOR:
            valid = (
                len(structural) == 1
                and not domain
                and self.floor_object_id == self.structural_parent_object_ids[0]
                and self.support_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.UNKNOWN:
            valid = (
                not domain
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        elif kind is CompetitionNativeSupportKindV2_9.MULTIPLE_AMBIGUOUS:
            valid = (
                len(structural) + len(domain) > 1
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        else:
            valid = (
                bool(domain)
                and self.support_object_id is None
                and self.floor_object_id is None
            )
        if not valid:
            raise ValueError("captured support fact is not closed")
        return self


class CompetitionNativePositionV2_9(CanonicalModel):
    x: float
    y: float
    z: float


class CompetitionNativeFloorEnvelopeV2_9(CanonicalModel):
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    floor_object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    floor_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    native_aabb: OBB
    floor_top_z: float
    clearance_m: float = Field(ge=0.0)
    polygon_xy: tuple[Vec2, ...] = Field(min_length=3, max_length=_MAX_POLYGON_VERTICES)

    @model_validator(mode="after")
    def validate_polygon(self) -> Self:
        coordinates = tuple((point.x, point.y) for point in self.polygon_xy)
        twice_area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(coordinates, coordinates[1:] + coordinates[:1])
        )
        if len(set(coordinates)) < 3 or twice_area == 0.0:
            raise ValueError("floor envelope polygon is degenerate")
        return self


def _placement_payload(
    *,
    object_id: str,
    availability: CompetitionNativePlacementAvailabilityV2_9,
    support_kind: CompetitionNativeSupportKindV2_9,
    support_object_id: str | None,
    floor_object_id: str | None,
    native_positions: tuple[CompetitionNativePositionV2_9, ...],
    position_region: SubjectPositionRegion | None,
    reasons: tuple[str, ...],
) -> dict[str, object]:
    return {
        "availability": availability.value,
        "floor_object_id": floor_object_id,
        "native_positions": tuple(
            item.model_dump(mode="json") for item in native_positions
        ),
        "object_id": object_id,
        "position_region": (
            None if position_region is None else position_region.model_dump(mode="json")
        ),
        "reasons": reasons,
        "support_kind": support_kind.value,
        "support_object_id": support_object_id,
    }


class CompetitionNativeSubjectPlacementFactV2_9(CanonicalModel):
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    availability: CompetitionNativePlacementAvailabilityV2_9
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    floor_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    native_positions: tuple[CompetitionNativePositionV2_9, ...] = Field(
        max_length=_MAX_NATIVE_POSITIONS
    )
    position_region: SubjectPositionRegion | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)
    placement_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_placement(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("placement reasons must be non-empty and canonical")
        region = self.position_region
        if region is not None and (
            len(region.region_id) > _MAX_TEXT_CHARS
            or len(region.subject_object_id) > _MAX_TEXT_CHARS
            or len(region.components) > _MAX_POLYGON_VERTICES
            or any(
                len(component.exterior) > _MAX_POLYGON_VERTICES
                or len(component.holes) > _MAX_POLYGON_VERTICES
                or any(len(hole) > _MAX_POLYGON_VERTICES for hole in component.holes)
                for component in region.components
            )
        ):
            raise ValueError("placement region exceeds persisted nested limit")
        position_keys = tuple(
            (item.x, item.z, item.y) for item in self.native_positions
        )
        if position_keys != tuple(sorted(set(position_keys))):
            raise ValueError("placement native positions must be unique and canonical")
        if (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.NOT_APPLICABLE
        ):
            valid = (
                not self.native_positions
                and self.position_region is None
                and not self.reasons
            )
        elif (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        ):
            valid = (
                self.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and bool(self.native_positions)
                and (
                    self.position_region is None
                    or (
                        self.position_region.subject_object_id == self.object_id
                        and self.position_region.source_kind
                        == "ai2thor-receptacle-trigger-grid-v1"
                        and bool(self.position_region.components)
                    )
                )
                and not self.reasons
            )
        elif (
            self.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
        ):
            valid = (
                self.support_kind is CompetitionNativeSupportKindV2_9.FLOOR
                and not self.native_positions
                and self.position_region is not None
                and bool(self.position_region.components)
                and not self.reasons
            )
        else:
            valid = (
                not self.native_positions
                and self.position_region is None
                and bool(self.reasons)
            )
        if not valid:
            raise ValueError("subject placement fact is not closed")
        expected = canonical_sha256(
            _placement_payload(
                object_id=self.object_id,
                availability=self.availability,
                support_kind=self.support_kind,
                support_object_id=self.support_object_id,
                floor_object_id=self.floor_object_id,
                native_positions=self.native_positions,
                position_region=self.position_region,
                reasons=self.reasons,
            ),
            domain=_PLACEMENT_HASH_DOMAIN,
        )
        if self.placement_sha256 != expected:
            raise ValueError("subject placement digest mismatch")
        return self


def build_competition_native_subject_placement_fact_v2_9(
    *,
    object_id: str,
    availability: CompetitionNativePlacementAvailabilityV2_9,
    support_kind: CompetitionNativeSupportKindV2_9,
    support_object_id: str | None,
    floor_object_id: str | None,
    native_positions: tuple[CompetitionNativePositionV2_9, ...] = (),
    position_region: SubjectPositionRegion | None = None,
    reasons: tuple[str, ...] = (),
) -> CompetitionNativeSubjectPlacementFactV2_9:
    reasons = tuple(sorted(set(reasons)))
    native_positions = tuple(
        sorted(native_positions, key=lambda item: (item.x, item.z, item.y))
    )
    payload = _placement_payload(
        object_id=object_id,
        availability=availability,
        support_kind=support_kind,
        support_object_id=support_object_id,
        floor_object_id=floor_object_id,
        native_positions=native_positions,
        position_region=position_region,
        reasons=reasons,
    )
    return CompetitionNativeSubjectPlacementFactV2_9(
        object_id=object_id,
        availability=availability,
        support_kind=support_kind,
        support_object_id=support_object_id,
        floor_object_id=floor_object_id,
        native_positions=native_positions,
        position_region=position_region,
        reasons=reasons,
        placement_sha256=canonical_sha256(payload, domain=_PLACEMENT_HASH_DOMAIN),
    )


# Resolve model annotations before restoring the public type identity.
CompetitionNativeSourceRefV2_9.model_rebuild()
RosterPolicy.model_rebuild()
CompetitionNativeRuntimeIdentityV2_9.model_rebuild()
CompetitionNativeSupportFactV2_9.model_rebuild()
CompetitionNativePositionV2_9.model_rebuild()
CompetitionNativeFloorEnvelopeV2_9.model_rebuild()
CompetitionNativeSubjectPlacementFactV2_9.model_rebuild()


# Preserve supported public names and pickle lookup.
CompetitionNativeSupportKindV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativePlacementAvailabilityV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSubjectStateV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCandidateStateV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSourceRefV2_9.__module__ = "spatialcf.generation.capture.models"
RosterPolicy.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeRuntimeIdentityV2_9.__module__ = "spatialcf.generation.capture.models"
validate_competition_native_runtime_source_lineage_v2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSupportFactV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativePositionV2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeFloorEnvelopeV2_9.__module__ = "spatialcf.generation.capture.models"
_placement_payload.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSubjectPlacementFactV2_9.__module__ = "spatialcf.generation.capture.models"
build_competition_native_subject_placement_fact_v2_9.__module__ = "spatialcf.generation.capture.models"
