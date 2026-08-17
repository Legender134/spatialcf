"""Flat current capture and roster contracts.

These models preserve the current wire literals and hash domains without
inheriting historical policy or compilation versions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import OBB, Scene, SubjectPositionRegion, Vec2
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation._internal.evidence.camera import (
    CameraPolicy,
    SourceCameraEvidence,
)
from spatialcf.generation._internal.evidence.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)
from spatialcf.generation._internal.evidence.surface import (
    SourceSurfaceEvidence,
)

DatasetSplitV2_9 = Literal["train", "validation", "test"]

_POLICY_HASH_DOMAIN_V2_9_4 = (
    "spatialcf.competition-native-candidate-roster-policy.v2.9.4"
)
_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"
_SOURCE_CAPTURE_HASH_DOMAIN = "spatialcf.competition-native-source-capture.v2.9"
_MANIFEST_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-manifest.v2.9"
_SUMMARY_HASH_DOMAIN = "spatialcf.competition-native-candidate-roster-summary.v2.9"
_PLACEMENT_HASH_DOMAIN = "spatialcf.competition-native-placement-fact.v2.9"
_MAX_TEXT_CHARS = 512
_MAX_REASON_CHARS = 256
_MAX_REASONS = 8
_MAX_POLICY_SOURCES = 512
_MAX_OBJECTS_PER_SCENE = 96
_MAX_CANDIDATES_TOTAL = 40_000
_MAX_REQUESTS_TOTAL = 1_000
_MAX_NATIVE_POSITIONS = 10_000
_MAX_CAMERAS_PER_SCENE = 64
_MAX_OBSTACLES_PER_SCENE = 2_048
_MAX_REGIONS_PER_SCENE = 2_048
_MAX_POLYGON_VERTICES = 4_096
_SOURCE_CAPTURE_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024 - 16 * 1024
_MAX_PERSISTED_REQUEST_TEXT_CHARS = _MAX_TEXT_CHARS


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


class CompetitionNativeSourceRefV2_9(V2Model):
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    split: DatasetSplitV2_9
    source_locator_sha256: Sha256Digest = Field(
        description=(
            "Digest of the frozen locator record; this is distinct from a "
            "procedural runtime's source content digest."
        )
    )


class RosterPolicy(V2Model):
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
        return canonical_sha256_v2(self, domain=_POLICY_HASH_DOMAIN_V2_9_4)


CompetitionNativeCandidateRosterPolicyV2_9_4 = RosterPolicy


class CompetitionNativeRuntimeIdentityV2_9(V2Model):
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


class CompetitionNativeSupportFactV2_9(V2Model):
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


class CompetitionNativePositionV2_9(V2Model):
    x: float
    y: float
    z: float


class CompetitionNativeFloorEnvelopeV2_9(V2Model):
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


class CompetitionNativeSubjectPlacementFactV2_9(V2Model):
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
        expected = canonical_sha256_v2(
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
        placement_sha256=canonical_sha256_v2(payload, domain=_PLACEMENT_HASH_DOMAIN),
    )


def _capture_payload(
    *,
    source: CompetitionNativeSourceRefV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    scene: Scene,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
    settlement_pass_steps: int,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...],
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
) -> dict[str, object]:
    return {
        "depth_npy_sha256": depth_npy_sha256,
        "floor_envelope": None
        if floor_envelope is None
        else floor_envelope.model_dump(mode="json"),
        "instance_png_sha256": instance_png_sha256,
        "is_scene_at_rest": is_scene_at_rest,
        "placement_facts": tuple(
            item.model_dump(mode="json") for item in placement_facts
        ),
        "pointcloud_ply_sha256": pointcloud_ply_sha256,
        "reachable_positions": tuple(
            item.model_dump(mode="json") for item in reachable_positions
        ),
        "rgb_png_sha256": rgb_png_sha256,
        "runtime_identity": runtime_identity.model_dump(mode="json"),
        "scene": scene.model_dump(mode="json", warnings="error"),
        "settlement_pass_steps": settlement_pass_steps,
        "source": source.model_dump(mode="json"),
        "support_facts": tuple(item.model_dump(mode="json") for item in support_facts),
    }


def normalize_competition_native_source_scene_v2_9(scene: Scene) -> Scene:
    """Reject duplicate source rosters and return their one canonical ordering."""

    if type(scene) is not Scene:
        raise TypeError("captured source scene must be an exact Scene")
    if len(scene.scene_id) > _MAX_TEXT_CHARS or len(scene.source) > _MAX_TEXT_CHARS:
        raise ValueError("captured source scene identity exceeds persisted limit")
    if (
        len(scene.objects) > _MAX_OBJECTS_PER_SCENE
        or len(scene.cameras) > _MAX_CAMERAS_PER_SCENE
        or len(scene.collision_obstacles) > _MAX_OBSTACLES_PER_SCENE
        or len(scene.subject_position_regions) > _MAX_REGIONS_PER_SCENE
        or len(scene.room_polygon_xy) > _MAX_POLYGON_VERTICES
    ):
        raise ValueError("captured source scene nested roster exceeds persisted limit")
    for item in scene.objects:
        if (
            any(
                len(value) > _MAX_TEXT_CHARS
                for value in (item.object_id, item.name, item.category)
            )
            or len(item.views) > _MAX_CAMERAS_PER_SCENE
            or any(
                len(camera_id) > _MAX_TEXT_CHARS
                or len(view.camera_id) > _MAX_TEXT_CHARS
                for camera_id, view in item.views.items()
            )
            or (
                item.support_object_id is not None
                and len(item.support_object_id) > _MAX_TEXT_CHARS
            )
        ):
            raise ValueError("captured source object exceeds persisted limit")
    if (
        any(len(item.camera_id) > _MAX_TEXT_CHARS for item in scene.cameras)
        or any(
            len(item.obstacle_id) > _MAX_TEXT_CHARS
            or len(item.source_object_id) > _MAX_TEXT_CHARS
            for item in scene.collision_obstacles
        )
        or any(
            len(item.region_id) > _MAX_TEXT_CHARS
            or len(item.subject_object_id) > _MAX_TEXT_CHARS
            or len(item.components) > _MAX_POLYGON_VERTICES
            or any(
                len(component.exterior) > _MAX_POLYGON_VERTICES
                or len(component.holes) > _MAX_POLYGON_VERTICES
                or any(len(hole) > _MAX_POLYGON_VERTICES for hole in component.holes)
                for component in item.components
            )
            for item in scene.subject_position_regions
        )
        or len(scene.pinned_object_ids) > _MAX_OBJECTS_PER_SCENE
        or any(
            len(object_id) > _MAX_TEXT_CHARS for object_id in scene.pinned_object_ids
        )
    ):
        raise ValueError("captured source scene strings exceed persisted limit")
    unique_rosters = (
        (tuple(item.object_id for item in scene.objects), "object IDs"),
        (tuple(item.name for item in scene.objects), "object names"),
        (tuple(item.camera_id for item in scene.cameras), "camera IDs"),
        (
            tuple(item.obstacle_id for item in scene.collision_obstacles),
            "collision obstacle IDs",
        ),
        (
            tuple(item.region_id for item in scene.subject_position_regions),
            "subject position region IDs",
        ),
    )
    for values, label in unique_rosters:
        if len(values) != len(set(values)):
            raise ValueError(f"captured source scene contains duplicate {label}")
    normalized = scene.model_copy(
        update={
            "cameras": tuple(sorted(scene.cameras, key=lambda item: item.camera_id)),
            "objects": tuple(sorted(scene.objects, key=lambda item: item.object_id)),
            "collision_obstacles": tuple(
                sorted(scene.collision_obstacles, key=lambda item: item.obstacle_id)
            ),
            "subject_position_regions": tuple(
                sorted(
                    scene.subject_position_regions,
                    key=lambda item: item.region_id,
                )
            ),
        }
    )
    return Scene.model_validate(normalized.model_dump(mode="python"), strict=True)


def _point_in_polygon_or_boundary(point: Vec2, polygon: tuple[Vec2, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        cross = (point.y - previous.y) * (current.x - previous.x) - (
            point.x - previous.x
        ) * (current.y - previous.y)
        if (
            abs(cross) <= 1e-9
            and min(previous.x, current.x) - 1e-9
            <= point.x
            <= max(previous.x, current.x) + 1e-9
            and min(previous.y, current.y) - 1e-9
            <= point.y
            <= max(previous.y, current.y) + 1e-9
        ):
            return True
        if (current.y > point.y) != (previous.y > point.y):
            intersection_x = previous.x + (
                (point.y - previous.y)
                * (current.x - previous.x)
                / (current.y - previous.y)
            )
            if point.x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _validate_floor_envelope_binding(
    scene: Scene,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
) -> None:
    floor_facts = tuple(
        item
        for item in support_facts
        if item.support_kind is CompetitionNativeSupportKindV2_9.FLOOR
    )
    known_floor_ids = {
        item.floor_object_id for item in floor_facts if item.floor_object_id is not None
    }
    known_floor_placements = tuple(
        item
        for item in placement_facts
        if item.availability
        is CompetitionNativePlacementAvailabilityV2_9.KNOWN_FLOOR_INNER_REGION
    )
    if floor_envelope is None:
        if known_floor_placements:
            raise ValueError("known floor placement has no captured floor envelope")
        return
    native_aabb = floor_envelope.native_aabb
    floor_bounds = (
        native_aabb.center.x - native_aabb.extent.x / 2.0,
        native_aabb.center.y - native_aabb.extent.y / 2.0,
        native_aabb.center.x + native_aabb.extent.x / 2.0,
        native_aabb.center.y + native_aabb.extent.y / 2.0,
    )
    if runtime_identity.source_floor_xz_bounds is not None:
        source_bounds = runtime_identity.source_floor_xz_bounds
        floor_bounds = (
            max(source_bounds[0], floor_bounds[0]),
            max(source_bounds[1], floor_bounds[1]),
            min(source_bounds[2], floor_bounds[2]),
            min(source_bounds[3], floor_bounds[3]),
        )
    clearance = floor_envelope.clearance_m
    expected_polygon = (
        Vec2(x=floor_bounds[0] + clearance, y=floor_bounds[1] + clearance),
        Vec2(x=floor_bounds[2] - clearance, y=floor_bounds[1] + clearance),
        Vec2(x=floor_bounds[2] - clearance, y=floor_bounds[3] - clearance),
        Vec2(x=floor_bounds[0] + clearance, y=floor_bounds[3] - clearance),
    )
    if (
        floor_envelope.scene_id != scene.scene_id
        or known_floor_ids != {floor_envelope.floor_object_id}
        or not floor_facts
        or floor_envelope.polygon_xy != expected_polygon
        or floor_envelope.floor_top_z
        != native_aabb.center.z + native_aabb.extent.z / 2.0
        or any(
            not _point_in_polygon_or_boundary(point, scene.room_polygon_xy)
            for point in floor_envelope.polygon_xy
        )
        or floor_envelope.native_aabb.extent.x <= 0.0
        or floor_envelope.native_aabb.extent.y <= 0.0
        or floor_envelope.native_aabb.extent.z <= 0.0
    ):
        raise ValueError("captured floor envelope does not bind the settled scene")
    floor_fact_by_object = {item.object_id: item for item in floor_facts}
    if any(
        item.object_id not in floor_fact_by_object
        or item.floor_object_id != floor_envelope.floor_object_id
        or item.position_region is None
        or item.position_region.subject_object_id != item.object_id
        or any(
            not _point_in_polygon_or_boundary(point, floor_envelope.polygon_xy)
            for component in item.position_region.components
            for ring in (component.exterior, *component.holes)
            for point in ring
        )
        for item in known_floor_placements
    ):
        raise ValueError("captured floor placement does not bind its support fact")


def validate_competition_native_floor_envelope_v2_9(
    scene: Scene,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    *,
    expected_clearance_m: float,
) -> None:
    """Bind an adapter floor envelope to this scene, support roster and call."""

    if floor_envelope.clearance_m != expected_clearance_m:
        raise ValueError("captured floor envelope clearance does not match request")
    _validate_floor_envelope_binding(
        scene, support_facts, floor_envelope, (), runtime_identity
    )


class CompetitionNativeSourceCaptureV2_9(V2Model):
    source: CompetitionNativeSourceRefV2_9
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9
    scene: Scene
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    pointcloud_ply_sha256: Sha256Digest
    is_scene_at_rest: bool
    settlement_pass_steps: int = Field(strict=True, ge=0)
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...] = Field(
        max_length=_MAX_NATIVE_POSITIONS
    )
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...] = Field(
        max_length=_MAX_OBJECTS_PER_SCENE
    )
    source_capture_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        if self.scene.scene_id != self.source.scene_id:
            raise ValueError("source capture scene identity mismatch")
        if (self.runtime_identity.width, self.runtime_identity.height) == (0, 0):
            raise ValueError("source capture runtime dimensions are invalid")
        validate_competition_native_runtime_source_lineage_v2_9(
            self.source, self.runtime_identity
        )
        normalized_scene = normalize_competition_native_source_scene_v2_9(self.scene)
        if normalized_scene != self.scene:
            raise ValueError("captured source scene ordering is not canonical")
        object_ids = tuple(item.object_id for item in self.scene.objects)
        support_ids = tuple(item.object_id for item in self.support_facts)
        placement_ids = tuple(item.object_id for item in self.placement_facts)
        if support_ids != tuple(sorted(object_ids)) or placement_ids != tuple(
            sorted(object_ids)
        ):
            raise ValueError("source capture object fact rosters are not closed")
        object_id_set = set(object_ids)
        if any(
            parent_id not in object_id_set
            for fact in self.support_facts
            for parent_id in fact.domain_parent_object_ids
        ):
            raise ValueError("source capture domain parent has no stable object")
        support_by_id = {item.object_id: item for item in self.support_facts}
        if any(
            item.scene_id != self.scene.scene_id
            or item.object_name != self.scene.object_by_id(item.object_id).name
            for item in self.support_facts
        ):
            raise ValueError("source capture support facts do not bind the scene")
        if any(
            placement.support_kind
            is not support_by_id[placement.object_id].support_kind
            or placement.support_object_id
            != support_by_id[placement.object_id].support_object_id
            or placement.floor_object_id
            != support_by_id[placement.object_id].floor_object_id
            for placement in self.placement_facts
        ):
            raise ValueError("source capture placement facts do not bind support facts")
        _validate_floor_envelope_binding(
            self.scene,
            self.support_facts,
            self.floor_envelope,
            self.placement_facts,
            self.runtime_identity,
        )
        position_keys = tuple(
            (item.x, item.z, item.y) for item in self.reachable_positions
        )
        if position_keys != tuple(sorted(set(position_keys))):
            raise ValueError("captured reachable positions are not canonical")
        capture_payload = _capture_payload(
            source=self.source,
            runtime_identity=self.runtime_identity,
            scene=self.scene,
            rgb_png_sha256=self.rgb_png_sha256,
            depth_npy_sha256=self.depth_npy_sha256,
            instance_png_sha256=self.instance_png_sha256,
            pointcloud_ply_sha256=self.pointcloud_ply_sha256,
            is_scene_at_rest=self.is_scene_at_rest,
            settlement_pass_steps=self.settlement_pass_steps,
            support_facts=self.support_facts,
            floor_envelope=self.floor_envelope,
            reachable_positions=self.reachable_positions,
            placement_facts=self.placement_facts,
        )
        if (
            len(canonical_json_bytes_v2(capture_payload))
            > _SOURCE_CAPTURE_PAYLOAD_MAX_BYTES
        ):
            raise ValueError("source capture exceeds persisted record byte limit")
        expected = canonical_sha256_v2(
            capture_payload,
            domain=_SOURCE_CAPTURE_HASH_DOMAIN,
        )
        if self.source_capture_sha256 != expected:
            raise ValueError("source capture digest mismatch")
        return self


def build_competition_native_source_capture_v2_9(
    *,
    source: CompetitionNativeSourceRefV2_9,
    runtime_identity: CompetitionNativeRuntimeIdentityV2_9,
    scene: Scene,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
    settlement_pass_steps: int,
    support_facts: tuple[CompetitionNativeSupportFactV2_9, ...],
    floor_envelope: CompetitionNativeFloorEnvelopeV2_9 | None,
    reachable_positions: tuple[CompetitionNativePositionV2_9, ...],
    placement_facts: tuple[CompetitionNativeSubjectPlacementFactV2_9, ...],
) -> CompetitionNativeSourceCaptureV2_9:
    normalized_scene = normalize_competition_native_source_scene_v2_9(scene)
    support_facts = tuple(sorted(support_facts, key=lambda item: item.object_id))
    reachable_positions = tuple(
        sorted(reachable_positions, key=lambda item: (item.x, item.z, item.y))
    )
    placement_facts = tuple(sorted(placement_facts, key=lambda item: item.object_id))
    payload = _capture_payload(
        source=source,
        runtime_identity=runtime_identity,
        scene=normalized_scene,
        rgb_png_sha256=rgb_png_sha256,
        depth_npy_sha256=depth_npy_sha256,
        instance_png_sha256=instance_png_sha256,
        pointcloud_ply_sha256=pointcloud_ply_sha256,
        is_scene_at_rest=is_scene_at_rest,
        settlement_pass_steps=settlement_pass_steps,
        support_facts=support_facts,
        floor_envelope=floor_envelope,
        reachable_positions=reachable_positions,
        placement_facts=placement_facts,
    )
    return CompetitionNativeSourceCaptureV2_9(
        source=source,
        runtime_identity=runtime_identity,
        scene=normalized_scene,
        rgb_png_sha256=rgb_png_sha256,
        depth_npy_sha256=depth_npy_sha256,
        instance_png_sha256=instance_png_sha256,
        pointcloud_ply_sha256=pointcloud_ply_sha256,
        is_scene_at_rest=is_scene_at_rest,
        settlement_pass_steps=settlement_pass_steps,
        support_facts=support_facts,
        floor_envelope=floor_envelope,
        reachable_positions=reachable_positions,
        placement_facts=placement_facts,
        source_capture_sha256=canonical_sha256_v2(
            payload, domain=_SOURCE_CAPTURE_HASH_DOMAIN
        ),
    )


class CompetitionNativeSourceCaptureOutcomeV2_9(V2Model):
    source: CompetitionNativeSourceRefV2_9
    status: Literal["accepted", "rejected"]
    capture: CompetitionNativeSourceCaptureV2_9 | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("source outcome reasons must be canonical")
        if self.status == "accepted":
            if (
                self.capture is None
                or self.reasons
                or self.capture.source != self.source
            ):
                raise ValueError("accepted source outcome is not closed")
        elif self.capture is not None or not self.reasons:
            raise ValueError("rejected source outcome is not closed")
        return self


class CompetitionNativeObjectInventoryV2_9(V2Model):
    inventory_id: str = Field(pattern=r"^object-[0-9a-f]{64}$")
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    object_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_name: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    category: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    subject_state: CompetitionNativeSubjectStateV2_9
    reference_eligible: bool
    support_kind: CompetitionNativeSupportKindV2_9
    support_object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    placement_sha256: Sha256Digest
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("object inventory reasons must be canonical")
        eligible = self.subject_state in {
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_RECEPTACLE_DOMAIN,
            CompetitionNativeSubjectStateV2_9.ELIGIBLE_FLOOR_INNER_DOMAIN,
        }
        if eligible == bool(self.reasons):
            raise ValueError("object subject terminal state is not closed")
        return self


class CompetitionNativeCandidateInventoryV2_9(V2Model):
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    source_capture_sha256: Sha256Digest
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    state: CompetitionNativeCandidateStateV2_9
    selection_index: int | None = Field(default=None, strict=True, ge=0)
    reasons: tuple[str, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        if self.subject_id == self.reference_id:
            raise ValueError("candidate subject and reference must differ")
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("candidate reasons must be canonical")
        if self.state is CompetitionNativeCandidateStateV2_9.SELECTED:
            if self.selection_index is None or self.reasons:
                raise ValueError("selected candidate is not closed")
        elif self.selection_index is not None or not self.reasons:
            raise ValueError("rejected candidate is not closed")
        return self


class CompetitionNativeSelectedRequestV2_9(V2Model):
    request_id: str = Field(pattern=r"^request-[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    selection_index: int = Field(strict=True, ge=0)
    source_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    source_locator_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    scene_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    split: DatasetSplitV2_9
    subject_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    subject_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_id: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_name: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    reference_category: str = Field(
        strict=True, min_length=1, max_length=_MAX_PERSISTED_REQUEST_TEXT_CHARS
    )
    relation_before: Relation
    relation_after: Relation
    support_kind: CompetitionNativeSupportKindV2_9
    camera_id: Literal["main"] = "main"

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("selected request relation_after must be opposite")
        return self


class CompetitionNativeCandidateRosterManifestV2_9(V2Model):
    manifest_version: Literal["competition-native-candidate-roster-manifest:2.9"] = (
        "competition-native-candidate-roster-manifest:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    requests: tuple[CompetitionNativeSelectedRequestV2_9, ...] = Field(
        max_length=_MAX_REQUESTS_TOTAL
    )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        indices = tuple(item.selection_index for item in self.requests)
        if indices != tuple(range(len(self.requests))):
            raise ValueError("request manifest selection indices are not contiguous")
        request_ids = tuple(item.request_id for item in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request manifest request IDs are not unique")
        return self

    @property
    def manifest_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_MANIFEST_HASH_DOMAIN)


class CompetitionNativeRosterRejectionV2_9(V2Model):
    rejection_id: str = Field(pattern=r"^rejection-[0-9a-f]{64}$")
    stage: Literal["source", "object_subject", "object_reference", "candidate"]
    source_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    scene_id: str = Field(strict=True, min_length=1, max_length=_MAX_TEXT_CHARS)
    object_id: str | None = Field(default=None, max_length=_MAX_TEXT_CHARS)
    candidate_id: str | None
    reasons: tuple[str, ...] = Field(max_length=_MAX_REASONS)

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.reasons != tuple(sorted(set(self.reasons))) or any(
            not reason or len(reason) > _MAX_REASON_CHARS for reason in self.reasons
        ):
            raise ValueError("roster rejection reasons must be canonical")
        if self.stage == "source":
            valid = self.object_id is None and self.candidate_id is None
        elif self.stage in {"object_subject", "object_reference"}:
            valid = self.object_id is not None and self.candidate_id is None
        else:
            valid = self.object_id is None and self.candidate_id is not None
        if not valid:
            raise ValueError("roster rejection identity is not closed")
        return self


class CompetitionNativeStageCountV2_9(V2Model):
    stage: str = Field(strict=True, min_length=1, max_length=128)
    count: int = Field(strict=True, gt=0)


class CompetitionNativeCandidateStateCountV2_9(V2Model):
    state: CompetitionNativeCandidateStateV2_9
    count: int = Field(strict=True, gt=0)


class CompetitionNativeRelationCountV2_9(V2Model):
    relation: Relation
    count: int = Field(strict=True, gt=0)


class RosterSummary(V2Model):
    summary_version: Literal["competition-native-candidate-roster-summary:2.9"] = (
        "competition-native-candidate-roster-summary:2.9"
    )
    evidence_eligible: Literal[False] = False
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    policy_sha256: Sha256Digest
    manifest_sha256: Sha256Digest
    source_count: int = Field(strict=True, ge=0)
    accepted_source_count: int = Field(strict=True, ge=0)
    rejected_source_count: int = Field(strict=True, ge=0)
    object_count: int = Field(strict=True, ge=0)
    eligible_subject_count: int = Field(strict=True, ge=0)
    rejected_subject_count: int = Field(strict=True, ge=0)
    candidate_count: int = Field(strict=True, ge=0)
    selected_request_count: int = Field(strict=True, ge=0)
    rejected_candidate_count: int = Field(strict=True, ge=0)
    candidate_state_counts: tuple[CompetitionNativeCandidateStateCountV2_9, ...] = (
        Field(max_length=len(CompetitionNativeCandidateStateV2_9))
    )
    relation_selected_counts: tuple[CompetitionNativeRelationCountV2_9, ...] = Field(
        max_length=len(Relation)
    )
    rejection_stage_counts: tuple[CompetitionNativeStageCountV2_9, ...] = Field(
        max_length=4
    )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.source_count != self.accepted_source_count + self.rejected_source_count:
            raise ValueError("summary source counts do not close")
        if (
            self.object_count
            != self.eligible_subject_count + self.rejected_subject_count
        ):
            raise ValueError("summary object counts do not close")
        if (
            self.candidate_count
            != self.selected_request_count + self.rejected_candidate_count
        ):
            raise ValueError("summary candidate counts do not close")
        if (
            sum(item.count for item in self.candidate_state_counts)
            != self.candidate_count
        ):
            raise ValueError("summary candidate state counts do not close")
        if (
            sum(item.count for item in self.relation_selected_counts)
            != self.selected_request_count
        ):
            raise ValueError("summary relation counts do not close")
        for counts, key in (
            (self.candidate_state_counts, lambda item: item.state.value),
            (self.relation_selected_counts, lambda item: item.relation.value),
            (self.rejection_stage_counts, lambda item: item.stage),
        ):
            keys = tuple(key(item) for item in counts)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise ValueError("summary count keys are not canonical")
        return self

    @property
    def summary_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_SUMMARY_HASH_DOMAIN)


CompetitionNativeCandidateRosterSummaryV2_9 = RosterSummary


class RosterCompilation(V2Model):
    """The only supported complete roster compilation."""

    policy: RosterPolicy
    scene_inventory: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    object_inventory: tuple[CompetitionNativeObjectInventoryV2_9, ...] = Field(
        max_length=_MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
    )
    candidate_inventory: tuple[CompetitionNativeCandidateInventoryV2_9, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )
    request_manifest: CompetitionNativeCandidateRosterManifestV2_9
    rejections: tuple[CompetitionNativeRosterRejectionV2_9, ...] = Field(
        max_length=(
            _MAX_POLICY_SOURCES
            + 2 * _MAX_POLICY_SOURCES * _MAX_OBJECTS_PER_SCENE
            + _MAX_CANDIDATES_TOTAL
        )
    )
    summary: RosterSummary
    surface_evidence: tuple[SourceSurfaceEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    camera_evidence: tuple[SourceCameraEvidence, ...] = Field(
        max_length=_MAX_POLICY_SOURCES
    )
    target_reachability: tuple[CandidateTargetReachability, ...] = Field(
        max_length=_MAX_CANDIDATES_TOTAL
    )

    @model_validator(mode="after")
    def validate_compilation(self) -> Self:
        if self.summary.selected_request_count != len(self.request_manifest.requests):
            raise ValueError("compilation request count does not close")
        if self.summary.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("compilation policy digest mismatch")
        if self.summary.manifest_sha256 != self.request_manifest.manifest_sha256:
            raise ValueError("compilation manifest digest mismatch")
        return self

    @model_validator(mode="after")
    def validate_surface_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.surface_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("surface evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.surface_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_capture_sha256 != capture.source_capture_sha256
            ):
                raise ValueError("surface evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_camera_evidence(self) -> Self:
        accepted = tuple(
            item for item in self.scene_inventory if item.status == "accepted"
        )
        if tuple(item.source_id for item in self.camera_evidence) != tuple(
            item.source.source_id for item in accepted
        ):
            raise ValueError("camera evidence does not exactly cover accepted sources")
        for evidence, record in zip(self.camera_evidence, accepted, strict=True):
            capture = record.capture
            if (
                capture is None
                or evidence.scene_id != record.source.scene_id
                or evidence.source_locator_sha256 != record.source.source_locator_sha256
                or evidence.source_capture_sha256 != capture.source_capture_sha256
                or evidence.camera != capture.scene.camera_by_id("main")
                or evidence.rgb_png_sha256 != capture.rgb_png_sha256
                or evidence.depth_npy_sha256 != capture.depth_npy_sha256
                or evidence.instance_png_sha256 != capture.instance_png_sha256
                or evidence.pointcloud_ply_sha256 != capture.pointcloud_ply_sha256
                or evidence.is_scene_at_rest is not capture.is_scene_at_rest
            ):
                raise ValueError("camera evidence does not bind its source capture")
        return self

    @model_validator(mode="after")
    def validate_target_reachability(self) -> Self:
        rows = self.target_reachability
        row_ids = tuple(item.candidate_id for item in rows)
        if row_ids != tuple(sorted(set(row_ids))):
            raise ValueError("target reachability rows are not canonical")
        candidates = {item.candidate_id: item for item in self.candidate_inventory}
        expected_ids = tuple(
            sorted(
                item.candidate_id
                for item in self.candidate_inventory
                if item.support_kind is CompetitionNativeSupportKindV2_9.RECEPTACLE
                and item.state
                in {
                    CompetitionNativeCandidateStateV2_9.SELECTED,
                    CompetitionNativeCandidateStateV2_9.REJECTED_POLICY_CAP,
                    CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE,
                }
            )
        )
        if row_ids != expected_ids:
            raise ValueError(
                "target reachability does not exactly cover receptacle selection inputs"
            )
        capture_by_source = {
            item.source.source_id: item.capture
            for item in self.scene_inventory
            if item.capture is not None
        }
        surface_by_source = {item.source_id: item for item in self.surface_evidence}
        camera_by_source = {item.source_id: item for item in self.camera_evidence}
        for row in rows:
            candidate = candidates[row.candidate_id]
            capture = capture_by_source.get(candidate.source_id)
            surface = surface_by_source.get(candidate.source_id)
            camera = camera_by_source.get(candidate.source_id)
            if capture is None or surface is None or camera is None:
                raise ValueError("target reachability source evidence is absent")
            placement = next(
                (
                    item
                    for item in capture.placement_facts
                    if item.object_id == candidate.subject_id
                ),
                None,
            )
            subject_surface = next(
                (
                    item
                    for item in surface.subjects
                    if item.subject_object_id == candidate.subject_id
                ),
                None,
            )
            unreachable = (
                candidate.state
                is CompetitionNativeCandidateStateV2_9.REJECTED_TARGET_UNREACHABLE
            )
            if (
                placement is None
                or subject_surface is None
                or row.source_id != candidate.source_id
                or row.scene_id != candidate.scene_id
                or row.source_capture_sha256 != candidate.source_capture_sha256
                or row.subject_id != candidate.subject_id
                or row.reference_id != candidate.reference_id
                or row.relation_before is not candidate.relation_before
                or row.placement_sha256 != placement.placement_sha256
                or row.surface_evidence_sha256 != surface.surface_evidence_sha256
                or row.subject_surface_evidence_sha256
                != subject_surface.subject_surface_evidence_sha256
                or row.camera_evidence_sha256 != camera.camera_evidence_sha256
                or unreachable != (row.status is TargetReachabilityStatus.UNREACHABLE)
            ):
                raise ValueError(
                    "target reachability row does not bind candidate facts"
                )
        return self


CompetitionNativeCandidateRosterCompilationV2_9_4 = RosterCompilation


__all__ = (
    "CompetitionNativeCandidateInventoryV2_9",
    "CompetitionNativeCandidateRosterCompilationV2_9_4",
    "CompetitionNativeCandidateRosterManifestV2_9",
    "CompetitionNativeCandidateRosterPolicyV2_9_4",
    "CompetitionNativeCandidateRosterSummaryV2_9",
    "CompetitionNativeCandidateStateCountV2_9",
    "CompetitionNativeCandidateStateV2_9",
    "CompetitionNativeFloorEnvelopeV2_9",
    "CompetitionNativeObjectInventoryV2_9",
    "CompetitionNativePlacementAvailabilityV2_9",
    "CompetitionNativePositionV2_9",
    "CompetitionNativeRelationCountV2_9",
    "CompetitionNativeRosterRejectionV2_9",
    "CompetitionNativeRuntimeIdentityV2_9",
    "CompetitionNativeSelectedRequestV2_9",
    "CompetitionNativeSourceCaptureOutcomeV2_9",
    "CompetitionNativeSourceCaptureV2_9",
    "CompetitionNativeSourceRefV2_9",
    "CompetitionNativeStageCountV2_9",
    "CompetitionNativeSubjectPlacementFactV2_9",
    "CompetitionNativeSubjectStateV2_9",
    "CompetitionNativeSupportFactV2_9",
    "CompetitionNativeSupportKindV2_9",
    "DatasetSplitV2_9",
    "RosterCompilation",
    "RosterPolicy",
    "RosterSummary",
    "build_competition_native_source_capture_v2_9",
    "build_competition_native_subject_placement_fact_v2_9",
    "normalize_competition_native_source_scene_v2_9",
    "validate_competition_native_floor_envelope_v2_9",
    "validate_competition_native_runtime_source_lineage_v2_9",
)
