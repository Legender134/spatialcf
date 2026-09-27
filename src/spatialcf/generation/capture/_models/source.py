"""Capture source: exact contracts and pure derivation."""

from __future__ import annotations

import math

from typing import (
    Literal,
    Self,
)

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.scene import (
    Scene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.domain.base import CanonicalId

from spatialcf.domain.scene import (
    BBox2D,
    Vec2,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.generation.capture._models.constants import (
    _MAX_CAMERAS_PER_SCENE,
    _MAX_NATIVE_POSITIONS,
    _MAX_OBJECTS_PER_SCENE,
    _MAX_OBSTACLES_PER_SCENE,
    _MAX_POLYGON_VERTICES,
    _MAX_REGIONS_PER_SCENE,
    _MAX_TEXT_CHARS,
    _SOURCE_CAPTURE_HASH_DOMAIN,
    _SOURCE_CAPTURE_PAYLOAD_MAX_BYTES,
    _SOURCE_VIEW_BINDING_HASH_DOMAIN,
    _SOURCE_VIEW_FACT_HASH_DOMAIN,
    _SOURCE_VIEW_SAMPLING_POLICY_SHA256,
)

from spatialcf.generation.capture._models.contracts import (
    CompetitionNativeFloorEnvelopeV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativePositionV2_9,
    CompetitionNativeRuntimeIdentityV2_9,
    CompetitionNativeSourceRefV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportFactV2_9,
    CompetitionNativeSupportKindV2_9,
    validate_competition_native_runtime_source_lineage_v2_9,
)


class SourceViewObjectSamples(CanonicalModel):
    object_id: CanonicalId
    source_mask_bbox: BBox2D
    source_mask_pixel_count: int = Field(strict=True, gt=0)
    sample_rows: tuple[int, ...]
    sample_columns: tuple[int, ...]
    sample_weights: tuple[int, ...]
    local_x_quantized: tuple[int, ...]
    local_y_quantized: tuple[int, ...]
    local_z_quantized: tuple[int, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        arrays = (
            self.sample_rows,
            self.sample_columns,
            self.sample_weights,
            self.local_x_quantized,
            self.local_y_quantized,
            self.local_z_quantized,
        )
        if not self.sample_rows or len({len(item) for item in arrays}) != 1:
            raise ValueError("source-view sample arrays must be equal and nonempty")
        if any(type(value) is not int for array in arrays for value in array):
            raise TypeError("source-view samples must contain exact integers")
        if any(value < 0 for value in (*self.sample_rows, *self.sample_columns)):
            raise ValueError("source-view sample pixels must be in bounds")
        keys = tuple(
            (row // 2, column // 2)
            for row, column in zip(
                self.sample_rows, self.sample_columns, strict=True
            )
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source-view sample tiles must be canonical")
        if any(weight < 1 or weight > 4 for weight in self.sample_weights):
            raise ValueError("source-view sample weights must be in [1, 4]")
        if sum(self.sample_weights) != self.source_mask_pixel_count:
            raise ValueError("source-view sample weights do not close mask count")
        bbox_values = (
            self.source_mask_bbox.xmin,
            self.source_mask_bbox.ymin,
            self.source_mask_bbox.xmax,
            self.source_mask_bbox.ymax,
        )
        if any(
            not math.isfinite(value) or value < 0 or not float(value).is_integer()
            for value in bbox_values
        ):
            raise ValueError("source-view mask bbox must be an integer envelope")
        xmin, ymin, xmax, ymax = (int(value) for value in bbox_values)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("source-view mask bbox must be nonempty and half-open")
        if any(
            row < ymin or row >= ymax or column < xmin or column >= xmax
            for row, column in zip(self.sample_rows, self.sample_columns, strict=True)
        ):
            raise ValueError("source-view sample pixels must lie inside the mask bbox")
        if self.source_mask_pixel_count > (xmax - xmin) * (ymax - ymin):
            raise ValueError("source-view mask count exceeds the mask bbox area")
        for row, column, weight in zip(
            self.sample_rows,
            self.sample_columns,
            self.sample_weights,
            strict=True,
        ):
            tile_ymin = (row // 2) * 2
            tile_xmin = (column // 2) * 2
            tile_area = max(0, min(ymax, tile_ymin + 2) - max(ymin, tile_ymin)) * max(
                0, min(xmax, tile_xmin + 2) - max(xmin, tile_xmin)
            )
            if weight > tile_area:
                raise ValueError("source-view sample weight exceeds its mask tile")
        return self


class SourceViewFact(CanonicalModel):
    fact_version: Literal["competition-native-source-view-fact:2.9.5"]
    source_id: CanonicalId
    scene_id: CanonicalId
    source_locator_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    scene_sha256: Sha256Digest
    camera_sha256: Sha256Digest
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    sampling_policy_sha256: Sha256Digest
    objects: tuple[SourceViewObjectSamples, ...]
    source_view_fact_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        object_ids = tuple(item.object_id for item in self.objects)
        if not object_ids or object_ids != tuple(sorted(set(object_ids))):
            raise ValueError("source-view object rows must be canonical")
        if sum(len(item.sample_rows) for item in self.objects) > 76_800:
            raise ValueError("source-view fact exceeds the sample cap")
        if self.sampling_policy_sha256 != _SOURCE_VIEW_SAMPLING_POLICY_SHA256:
            raise ValueError("source-view sampling policy changed")
        payload = self.model_dump(
            mode="python", exclude={"source_view_fact_sha256"}
        )
        expected = canonical_sha256(
            payload, domain=_SOURCE_VIEW_FACT_HASH_DOMAIN
        )
        if self.source_view_fact_sha256 != expected:
            raise ValueError("source-view fact digest mismatch")
        return self


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
    source_view_fact: SourceViewFact | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
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
    if source_view_fact is not None:
        payload["source_view_fact"] = source_view_fact.model_dump(mode="json")
    return payload


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


class CompetitionNativeSourceCaptureV2_9(CanonicalModel):
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
    source_view_fact: SourceViewFact | None = None
    source_capture_sha256: Sha256Digest

    @model_serializer(mode="wrap")
    def serialize_optional_source_view_fact(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if self.source_view_fact is None:
            payload.pop("source_view_fact", None)
        return payload

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
        if self.source_view_fact is not None:
            fact = self.source_view_fact
            camera = self.scene.camera_by_id("main")
            if (
                fact.source_id != self.source.source_id
                or fact.scene_id != self.scene.scene_id
                or fact.source_locator_sha256 != self.source.source_locator_sha256
                or fact.runtime_identity_sha256
                != canonical_sha256(
                    self.runtime_identity,
                    domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN,
                )
                or fact.scene_sha256
                != canonical_sha256(
                    self.scene, domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN
                )
                or fact.camera_sha256
                != canonical_sha256(
                    camera, domain=_SOURCE_VIEW_BINDING_HASH_DOMAIN
                )
                or fact.rgb_png_sha256 != self.rgb_png_sha256
                or fact.depth_npy_sha256 != self.depth_npy_sha256
                or fact.instance_png_sha256 != self.instance_png_sha256
            ):
                raise ValueError("source-view fact does not bind captured evidence")
            if any(
                view is not None and view.camera_id != "main"
                for item in self.scene.objects
                for view in (item.views.get("main"),)
            ):
                raise ValueError("source-view fact main view camera identity changed")
            expected_object_ids = tuple(
                sorted(
                    item.object_id
                    for item in self.scene.objects
                    if (view := item.views.get("main")) is not None
                    and view.visible_fraction > 0.0
                )
            )
            fact_object_ids = tuple(item.object_id for item in fact.objects)
            if fact_object_ids != expected_object_ids:
                raise ValueError("source-view fact object roster does not bind main views")
            for samples in fact.objects:
                bbox = samples.source_mask_bbox
                if bbox.xmax > camera.width or bbox.ymax > camera.height:
                    raise ValueError("source-view fact bbox exceeds main camera bounds")
                if any(
                    row >= camera.height or column >= camera.width
                    for row, column in zip(
                        samples.sample_rows,
                        samples.sample_columns,
                        strict=True,
                    )
                ):
                    raise ValueError("source-view fact samples exceed main camera bounds")
                view = self.scene.object_by_id(samples.object_id).views["main"]
                if any(
                    not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                    for actual, expected in zip(
                        (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax),
                        (
                            view.bbox.xmin,
                            view.bbox.ymin,
                            view.bbox.xmax,
                            view.bbox.ymax,
                        ),
                        strict=True,
                    )
                ):
                    raise ValueError("source-view fact bbox does not bind its main view")
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
            source_view_fact=self.source_view_fact,
        )
        if (
            len(canonical_json_bytes(capture_payload))
            > _SOURCE_CAPTURE_PAYLOAD_MAX_BYTES
        ):
            raise ValueError("source capture exceeds persisted record byte limit")
        expected = canonical_sha256(
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
    source_view_fact: SourceViewFact | None = None,
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
        source_view_fact=source_view_fact,
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
        source_view_fact=source_view_fact,
        source_capture_sha256=canonical_sha256(
            payload, domain=_SOURCE_CAPTURE_HASH_DOMAIN
        ),
    )


# Resolve model annotations before restoring the public type identity.
SourceViewObjectSamples.model_rebuild()
SourceViewFact.model_rebuild()
CompetitionNativeSourceCaptureV2_9.model_rebuild()


# Preserve supported public names and pickle lookup.
SourceViewObjectSamples.__module__ = "spatialcf.generation.capture.models"
SourceViewFact.__module__ = "spatialcf.generation.capture.models"
_capture_payload.__module__ = "spatialcf.generation.capture.models"
normalize_competition_native_source_scene_v2_9.__module__ = "spatialcf.generation.capture.models"
_point_in_polygon_or_boundary.__module__ = "spatialcf.generation.capture.models"
_validate_floor_envelope_binding.__module__ = "spatialcf.generation.capture.models"
validate_competition_native_floor_envelope_v2_9.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSourceCaptureV2_9.__module__ = "spatialcf.generation.capture.models"
build_competition_native_source_capture_v2_9.__module__ = "spatialcf.generation.capture.models"
