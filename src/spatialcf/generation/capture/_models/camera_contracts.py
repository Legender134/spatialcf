"""Capture camera contracts: exact contracts and pure derivation."""

from __future__ import annotations

import math

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
    FiniteFloat,
    Sha256Digest,
)

from spatialcf.domain.scene import (
    Camera,
    Scene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.generation.capture._models.camera_geometry import (
    _expected_camera_world_to_camera,
)

from spatialcf.generation.capture._models.constants import (
    _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
    _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
    _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
    _EVIDENCE_HASH_DOMAIN,
    _GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _LEGACY_POSE_POLICY_VERSION,
    _MAX_ANGLE_RESIDUAL_DEGREES,
    _MAX_POSE_BANK_MEMBERS,
    _MAX_POSITION_RESIDUAL_M,
    _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _PLACEMENT_ROSTER_HASH_DOMAIN,
    _POLICY_HASH_DOMAIN,
    _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
    _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
)


def _editable_pose_policy_versions() -> frozenset[str]:
    return frozenset(
        {
            _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
            _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
            _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
            _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
            _GRID_MARGIN_EDITABLE_POLICY_VERSION,
            _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
            _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
        }
    )


class CompetitionNativeCameraPoseV2_9_3(CanonicalModel):
    """One exact native TeleportFull pose on the canonical evidence wire."""

    pose_version: Literal["competition-native-camera-pose:2.9.3"] = (
        "competition-native-camera-pose:2.9.3"
    )
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    yaw_degrees: FiniteFloat
    horizon_degrees: FiniteFloat
    standing: bool


def _policy_payload(
    pose_policy_version: str = _LEGACY_POSE_POLICY_VERSION,
) -> dict[str, object]:
    if pose_policy_version not in {
        _LEGACY_POSE_POLICY_VERSION,
        _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
        _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
        _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
        _GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    }:
        raise ValueError("camera evidence pose policy version is unsupported")
    return {
        "camera_id": "main",
        "maximum_pose_bank_count": 256,
        "maximum_truncated_fraction": 0.5,
        "minimum_image_area_fraction": 0.0025,
        "minimum_visible_fraction": 0.2,
        "policy_version": "competition-native-camera-selection-policy:2.9.3",
        "pose_policy_version": pose_policy_version,
    }


class CompetitionNativeCameraPolicyV2_9_3(CanonicalModel):
    """Frozen literal source-camera selection policy and its own digest."""

    policy_version: Literal["competition-native-camera-selection-policy:2.9.3"] = (
        "competition-native-camera-selection-policy:2.9.3"
    )
    pose_policy_version: Literal[
        "deterministic-pair-camera-tier-1:1",
        "deterministic-pair-camera-tier-1-solver-upright:2",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain:3",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.2m:4",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.21m:5",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.21m-reset-per-pose:6",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m:7",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m-physics-paused:8",
        "deterministic-pair-camera-tier-1-solver-upright-edit-domain-movable-clearance-0.25m-physics-paused-final-settle:9",
    ] = _LEGACY_POSE_POLICY_VERSION
    camera_id: Literal["main"] = "main"
    maximum_pose_bank_count: Literal[256] = 256
    minimum_visible_fraction: Literal[0.2] = 0.2
    minimum_image_area_fraction: Literal[0.0025] = 0.0025
    maximum_truncated_fraction: Literal[0.5] = 0.5
    policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_policy_digest(self) -> Self:
        expected = canonical_sha256(
            _policy_payload(self.pose_policy_version),
            domain=_POLICY_HASH_DOMAIN,
        )
        if self.policy_sha256 != expected:
            raise ValueError("camera evidence policy digest mismatch")
        return self


class CompetitionNativeCameraScoreV2_9_3(CanonicalModel):
    """The two literal source-only counts used by camera selection."""

    score_version: Literal["competition-native-camera-score:2.9.3"] = (
        "competition-native-camera-score:2.9.3"
    )
    movable_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)
    all_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_score_counts(self) -> Self:
        if (
            self.movable_scene_unique_category_qualifying_count
            > self.all_scene_unique_category_qualifying_count
        ):
            raise ValueError("camera evidence movable score exceeds total score")
        return self


class CompetitionNativeCameraPlacementPositionV2_9_4(CanonicalModel):
    """One exact native subject anchor considered by camera selection."""

    position_version: Literal["competition-native-camera-placement-position:2.9.4"] = (
        "competition-native-camera-placement-position:2.9.4"
    )
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class CompetitionNativeCameraPlacementRosterEntryV2_9_4(CanonicalModel):
    """One source subject and its complete canonical native placement roster."""

    entry_version: Literal["competition-native-camera-placement-roster-entry:2.9.4"] = (
        "competition-native-camera-placement-roster-entry:2.9.4"
    )
    subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    positions: tuple[CompetitionNativeCameraPlacementPositionV2_9_4, ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_positions(self) -> Self:
        ordered = tuple(
            sorted(
                set(self.positions),
                key=lambda item: (item.x, item.z, item.y),
            )
        )
        if self.positions != ordered:
            raise ValueError("camera placement positions must be unique and canonical")
        return self


class CompetitionNativeCameraScoreV2_9_4(CanonicalModel):
    """Source visibility plus the native edit domain visible from one pose."""

    score_version: Literal["competition-native-camera-score:2.9.4"] = (
        "competition-native-camera-score:2.9.4"
    )
    placement_roster_sha256: Sha256Digest
    visible_native_placement_subject_count: int = Field(strict=True, ge=0)
    visible_native_placement_count: int = Field(strict=True, ge=0)
    movable_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)
    all_scene_unique_category_qualifying_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_score_counts(self) -> Self:
        if (
            self.movable_scene_unique_category_qualifying_count
            > self.all_scene_unique_category_qualifying_count
            or self.visible_native_placement_subject_count
            > self.movable_scene_unique_category_qualifying_count
            or self.visible_native_placement_subject_count
            > self.visible_native_placement_count
        ):
            raise ValueError("editable camera score counts are inconsistent")
        return self


CompetitionNativeCameraScoreFamilyV2_9_3 = (
    CompetitionNativeCameraScoreV2_9_3 | CompetitionNativeCameraScoreV2_9_4
)


def _strict_placement_roster_v2_9_4(
    placement_roster: object,
) -> tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...]:
    if type(placement_roster) is not tuple or any(
        type(item) is not CompetitionNativeCameraPlacementRosterEntryV2_9_4
        for item in placement_roster
    ):
        raise TypeError("camera placement roster must be an exact entry tuple")
    checked = tuple(
        CompetitionNativeCameraPlacementRosterEntryV2_9_4.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in placement_roster
    )
    if not checked:
        raise ValueError("camera placement roster must not be empty")
    if tuple(
        sorted(checked, key=lambda item: item.subject_object_id)
    ) != checked or len({item.subject_object_id for item in checked}) != len(checked):
        raise ValueError("camera placement roster must be unique and canonical")
    return checked


def competition_native_camera_placement_roster_sha256_v2_9_4(
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
) -> Sha256Digest:
    """Hash one exact source placement roster in its independent domain."""

    checked = _strict_placement_roster_v2_9_4(placement_roster)
    payload = {
        "placement_roster_version": "competition-native-camera-placement-roster:2.9.4",
        "subjects": tuple(item.model_dump(mode="json") for item in checked),
    }
    return canonical_sha256(payload, domain=_PLACEMENT_ROSTER_HASH_DOMAIN)


def select_competition_native_camera_score_index_v2_9_4(
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_4, ...],
) -> int:
    """Select edit-domain coverage, then source visibility, then bank index."""

    if (
        type(pose_scores) is not tuple
        or not pose_scores
        or any(
            type(item) is not CompetitionNativeCameraScoreV2_9_4 for item in pose_scores
        )
    ):
        raise TypeError("editable camera score ledger must be an exact nonempty tuple")
    checked = tuple(
        CompetitionNativeCameraScoreV2_9_4.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in pose_scores
    )
    if (
        len(checked) > _MAX_POSE_BANK_MEMBERS
        or len({item.placement_roster_sha256 for item in checked}) != 1
    ):
        raise ValueError("editable camera score ledger is not source-aligned")
    return min(
        range(len(checked)),
        key=lambda index: (
            -checked[index].visible_native_placement_subject_count,
            -checked[index].visible_native_placement_count,
            -checked[index].movable_scene_unique_category_qualifying_count,
            -checked[index].all_scene_unique_category_qualifying_count,
            index,
        ),
    )


def _strict_score_ledger(
    pose_scores: object,
) -> tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...]:
    if type(pose_scores) is not tuple:
        raise TypeError("camera evidence score ledger must be an exact score tuple")
    if not pose_scores or len(pose_scores) > _MAX_POSE_BANK_MEMBERS:
        raise ValueError(
            "camera evidence score ledger must contain 1 through 256 scores"
        )
    score_type: type[CompetitionNativeCameraScoreFamilyV2_9_3]
    if all(type(item) is CompetitionNativeCameraScoreV2_9_3 for item in pose_scores):
        score_type = CompetitionNativeCameraScoreV2_9_3
    elif all(type(item) is CompetitionNativeCameraScoreV2_9_4 for item in pose_scores):
        score_type = CompetitionNativeCameraScoreV2_9_4
    else:
        raise TypeError("camera evidence score ledger mixes score versions")
    return tuple(
        score_type.model_validate(item.model_dump(mode="python"), strict=True)
        for item in pose_scores
    )


def select_competition_native_camera_score_index_v2_9_3(
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_3, ...],
) -> int:
    """Select the literal score argmax, breaking complete ties by bank index."""

    checked = _strict_score_ledger(pose_scores)
    if any(type(item) is not CompetitionNativeCameraScoreV2_9_3 for item in checked):
        raise TypeError("legacy camera selector requires 2.9.3 scores")
    return min(
        range(len(checked)),
        key=lambda index: (
            -checked[index].movable_scene_unique_category_qualifying_count,
            -checked[index].all_scene_unique_category_qualifying_count,
            index,
        ),
    )


def _select_competition_native_camera_score_index(
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...],
) -> int:
    checked = _strict_score_ledger(pose_scores)
    if type(checked[0]) is CompetitionNativeCameraScoreV2_9_3:
        return select_competition_native_camera_score_index_v2_9_3(checked)  # type: ignore[arg-type]
    return select_competition_native_camera_score_index_v2_9_4(checked)  # type: ignore[arg-type]


def verify_competition_native_camera_observation_binding_v2_9_3(
    requested_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_native_camera_position: tuple[float, float, float],
    camera: Camera,
) -> None:
    """Close persisted native observation fields to one requested main Camera."""

    if (
        type(requested_pose) is not CompetitionNativeCameraPoseV2_9_3
        or type(observed_pose) is not CompetitionNativeCameraPoseV2_9_3
    ):
        raise TypeError(
            "camera evidence camera observation binding poses must be exact"
        )
    if (
        type(observed_native_camera_position) is not tuple
        or len(observed_native_camera_position) != 3
        or any(type(item) is not float for item in observed_native_camera_position)
    ):
        raise TypeError(
            "camera evidence camera observation binding native position must be exact"
        )
    if type(camera) is not Camera or camera.camera_id != "main":
        raise TypeError(
            "camera evidence camera observation binding Camera must be exact main"
        )
    position_residual_m = math.dist(
        (requested_pose.x, requested_pose.y, requested_pose.z),
        (observed_pose.x, observed_pose.y, observed_pose.z),
    )
    yaw_residual_degrees = abs(
        (observed_pose.yaw_degrees - requested_pose.yaw_degrees + 180.0) % 360.0 - 180.0
    )
    horizon_residual_degrees = abs(
        observed_pose.horizon_degrees - requested_pose.horizon_degrees
    )
    if (
        position_residual_m > _MAX_POSITION_RESIDUAL_M
        or yaw_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or horizon_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or observed_pose.standing is not requested_pose.standing
    ):
        raise ValueError(
            "camera evidence camera observation binding does not close requested pose"
        )
    expected_world_to_camera = _expected_camera_world_to_camera(
        observed_native_camera_position,
        yaw_degrees=observed_pose.yaw_degrees,
        horizon_degrees=observed_pose.horizon_degrees,
    )
    if camera.world_to_camera != expected_world_to_camera:
        raise ValueError(
            "camera evidence camera observation binding does not close main Camera"
        )


def _evidence_payload(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    policy_sha256: str,
    pose_bank_sha256: str,
    pose_bank_count: int,
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...],
    selected_pose_index: int,
    requested_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_pose: CompetitionNativeCameraPoseV2_9_3,
    observed_native_camera_position: tuple[float, float, float],
    camera: Camera,
    score: CompetitionNativeCameraScoreFamilyV2_9_3,
    rgb_png_sha256: str,
    depth_npy_sha256: str,
    instance_png_sha256: str,
    pointcloud_ply_sha256: str,
    is_scene_at_rest: bool,
) -> dict[str, object]:
    return {
        "camera": camera.model_dump(mode="json"),
        "depth_npy_sha256": depth_npy_sha256,
        "evidence_version": "competition-native-source-camera-evidence:2.9.3",
        "instance_png_sha256": instance_png_sha256,
        "is_scene_at_rest": is_scene_at_rest,
        "observed_native_camera_position": observed_native_camera_position,
        "observed_pose": observed_pose.model_dump(mode="json"),
        "pointcloud_ply_sha256": pointcloud_ply_sha256,
        "policy_sha256": policy_sha256,
        "pose_bank_count": pose_bank_count,
        "pose_bank_sha256": pose_bank_sha256,
        "pose_scores": tuple(item.model_dump(mode="json") for item in pose_scores),
        "requested_pose": requested_pose.model_dump(mode="json"),
        "rgb_png_sha256": rgb_png_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "scene_id": scene_id,
        "score": score.model_dump(mode="json"),
        "selected_pose_index": selected_pose_index,
        "source_capture_sha256": source_capture_sha256,
        "source_id": source_id,
        "source_locator_sha256": source_locator_sha256,
    }


class CompetitionNativeSourceCameraEvidenceV2_9_3(CanonicalModel):
    """One selected source camera with complete immutable capture lineage."""

    evidence_version: Literal["competition-native-source-camera-evidence:2.9.3"] = (
        "competition-native-source-camera-evidence:2.9.3"
    )
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_locator_sha256: Sha256Digest
    runtime_identity_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    policy_sha256: Sha256Digest
    pose_bank_sha256: Sha256Digest
    pose_bank_count: int = Field(strict=True, ge=1, le=256)
    pose_scores: tuple[CompetitionNativeCameraScoreFamilyV2_9_3, ...] = Field(
        min_length=1, max_length=256
    )
    selected_pose_index: int = Field(strict=True, ge=0, le=255)
    requested_pose: CompetitionNativeCameraPoseV2_9_3
    observed_pose: CompetitionNativeCameraPoseV2_9_3
    observed_native_camera_position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    camera: Camera
    score: CompetitionNativeCameraScoreFamilyV2_9_3
    rgb_png_sha256: Sha256Digest
    depth_npy_sha256: Sha256Digest
    instance_png_sha256: Sha256Digest
    pointcloud_ply_sha256: Sha256Digest
    is_scene_at_rest: bool
    camera_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        checked_scores = _strict_score_ledger(self.pose_scores)
        if len(checked_scores) != self.pose_bank_count:
            raise ValueError("camera evidence score ledger is not bank-aligned")
        if self.selected_pose_index >= self.pose_bank_count:
            raise ValueError("camera evidence selected index is outside pose bank")
        if (
            self.selected_pose_index
            != _select_competition_native_camera_score_index(checked_scores)
            or self.score != checked_scores[self.selected_pose_index]
        ):
            raise ValueError("camera evidence selected index is not the literal argmax")
        if self.camera.camera_id != "main":
            raise ValueError("camera evidence must persist the main camera")
        verify_competition_native_camera_observation_binding_v2_9_3(
            self.requested_pose,
            self.observed_pose,
            self.observed_native_camera_position,
            self.camera,
        )
        expected_policies = {
            canonical_sha256(
                _policy_payload(version),
                domain=_POLICY_HASH_DOMAIN,
            )
            for version in (
                _LEGACY_POSE_POLICY_VERSION,
                _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
                _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
                _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
                _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
                _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
                _GRID_MARGIN_EDITABLE_POLICY_VERSION,
                _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
                _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
            )
        }
        if self.policy_sha256 not in expected_policies:
            raise ValueError("camera evidence policy digest mismatch")
        editable_policy_sha256s = {
            canonical_sha256(
                _policy_payload(version),
                domain=_POLICY_HASH_DOMAIN,
            )
            for version in _editable_pose_policy_versions()
        }
        if (type(checked_scores[0]) is CompetitionNativeCameraScoreV2_9_4) != (
            self.policy_sha256 in editable_policy_sha256s
        ):
            raise ValueError("camera evidence score version does not bind policy")
        expected = canonical_sha256(
            _evidence_payload(
                source_id=self.source_id,
                scene_id=self.scene_id,
                source_locator_sha256=self.source_locator_sha256,
                runtime_identity_sha256=self.runtime_identity_sha256,
                source_capture_sha256=self.source_capture_sha256,
                policy_sha256=self.policy_sha256,
                pose_bank_sha256=self.pose_bank_sha256,
                pose_bank_count=self.pose_bank_count,
                pose_scores=checked_scores,
                selected_pose_index=self.selected_pose_index,
                requested_pose=self.requested_pose,
                observed_pose=self.observed_pose,
                observed_native_camera_position=self.observed_native_camera_position,
                camera=self.camera,
                score=self.score,
                rgb_png_sha256=self.rgb_png_sha256,
                depth_npy_sha256=self.depth_npy_sha256,
                instance_png_sha256=self.instance_png_sha256,
                pointcloud_ply_sha256=self.pointcloud_ply_sha256,
                is_scene_at_rest=self.is_scene_at_rest,
            ),
            domain=_EVIDENCE_HASH_DOMAIN,
        )
        if self.camera_evidence_sha256 != expected:
            raise ValueError("camera evidence digest mismatch")
        return self


def _strict_scene(scene: object) -> Scene:
    if type(scene) is not Scene:
        raise TypeError("camera evidence scene must be an exact Scene")
    return Scene.model_validate(scene.model_dump(mode="python"), strict=True)


CameraPolicy = CompetitionNativeCameraPolicyV2_9_3


SourceCameraEvidence = CompetitionNativeSourceCameraEvidenceV2_9_3


# Resolve model annotations before restoring the public type identity.
CompetitionNativeCameraPoseV2_9_3.model_rebuild()
CompetitionNativeCameraPolicyV2_9_3.model_rebuild()
CompetitionNativeCameraScoreV2_9_3.model_rebuild()
CompetitionNativeCameraPlacementPositionV2_9_4.model_rebuild()
CompetitionNativeCameraPlacementRosterEntryV2_9_4.model_rebuild()
CompetitionNativeCameraScoreV2_9_4.model_rebuild()
CompetitionNativeSourceCameraEvidenceV2_9_3.model_rebuild()


# Preserve supported public names and pickle lookup.
_editable_pose_policy_versions.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraPoseV2_9_3.__module__ = "spatialcf.generation.capture.models"
_policy_payload.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraPolicyV2_9_3.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraScoreV2_9_3.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraPlacementPositionV2_9_4.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraPlacementRosterEntryV2_9_4.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeCameraScoreV2_9_4.__module__ = "spatialcf.generation.capture.models"
_strict_placement_roster_v2_9_4.__module__ = "spatialcf.generation.capture.models"
competition_native_camera_placement_roster_sha256_v2_9_4.__module__ = "spatialcf.generation.capture.models"
select_competition_native_camera_score_index_v2_9_4.__module__ = "spatialcf.generation.capture.models"
_strict_score_ledger.__module__ = "spatialcf.generation.capture.models"
select_competition_native_camera_score_index_v2_9_3.__module__ = "spatialcf.generation.capture.models"
_select_competition_native_camera_score_index.__module__ = "spatialcf.generation.capture.models"
verify_competition_native_camera_observation_binding_v2_9_3.__module__ = "spatialcf.generation.capture.models"
_evidence_payload.__module__ = "spatialcf.generation.capture.models"
CompetitionNativeSourceCameraEvidenceV2_9_3.__module__ = "spatialcf.generation.capture.models"
_strict_scene.__module__ = "spatialcf.generation.capture.models"
