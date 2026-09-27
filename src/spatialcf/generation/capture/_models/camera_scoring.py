"""Capture camera scoring: exact contracts and pure derivation."""

from __future__ import annotations

import math

from dataclasses import (
    asdict,
)

from spatialcf.adapters.base import (
    AdapterCameraApplication,
    AdapterObservation,
    AdapterPose,
    AdapterPosition,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.scene import (
    OBB,
    Camera,
    Scene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.relations.engine import (
    RelationEngine,
)

from spatialcf.generation.capture._models.camera_contracts import (
    CompetitionNativeCameraPlacementPositionV2_9_4,
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraPolicyV2_9_3,
    CompetitionNativeCameraPoseV2_9_3,
    CompetitionNativeCameraScoreV2_9_3,
    CompetitionNativeCameraScoreV2_9_4,
    _editable_pose_policy_versions,
    _policy_payload,
    _strict_placement_roster_v2_9_4,
    _strict_scene,
    competition_native_camera_placement_roster_sha256_v2_9_4,
)

from spatialcf.generation.capture._models.camera_geometry import (
    _CameraConversionError,
    _legacy_camera,
    deterministic_pair_camera_poses,
    filter_competition_native_camera_positions_v2_9_5,
    filter_competition_native_camera_positions_v2_9_6,
    filter_competition_native_camera_positions_v2_9_7,
    filter_competition_native_camera_positions_v2_9_8,
)

from spatialcf.generation.capture._models.constants import (
    _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION,
    _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
    _EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION,
    _GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _LEGACY_POSE_POLICY_VERSION,
    _MAX_ANGLE_RESIDUAL_DEGREES,
    _MAX_POSE_BANK_MEMBERS,
    _MAX_POSITION_RESIDUAL_M,
    _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _POLICY_HASH_DOMAIN,
    _POSE_BANK_HASH_DOMAIN,
    _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
    _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
)


def _strict_native_pose(pose: object, *, label: str) -> AdapterPose:
    if type(pose) is not AdapterPose:
        raise TypeError(f"{label} must be an exact AdapterPose")
    if type(pose.position) is not AdapterPosition:
        raise TypeError(f"{label} position must be exact")
    return AdapterPose(
        position=AdapterPosition(**asdict(pose.position)),
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _wire_pose(pose: AdapterPose) -> CompetitionNativeCameraPoseV2_9_3:
    return CompetitionNativeCameraPoseV2_9_3(
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        yaw_degrees=pose.yaw_degrees,
        horizon_degrees=pose.horizon_degrees,
        standing=pose.standing,
    )


def _pose_key(
    pose: CompetitionNativeCameraPoseV2_9_3,
) -> tuple[float, float, float, float, float, bool]:
    return (
        pose.x,
        pose.y,
        pose.z,
        pose.yaw_degrees,
        pose.horizon_degrees,
        pose.standing,
    )


def build_competition_native_camera_pose_bank_v2_9_3(
    scene: Scene,
    pairs: tuple[tuple[str, str], ...],
    reachable_positions: tuple[AdapterPosition, ...],
    fallback_pose: AdapterPose,
    *,
    policy: CompetitionNativeCameraPolicyV2_9_3 | None = None,
) -> tuple[CompetitionNativeCameraPoseV2_9_3, ...]:
    """Build the bounded, permutation-invariant Tier-1 source pose bank."""

    checked_scene = _strict_scene(scene)
    if type(pairs) is not tuple or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not str or not item for item in pair)
        for pair in pairs
    ):
        raise TypeError("camera evidence pairs must be an exact tuple of string pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("camera evidence pairs must be unique")
    if (
        type(reachable_positions) is not tuple
        or not reachable_positions
        or any(type(item) is not AdapterPosition for item in reachable_positions)
    ):
        raise TypeError("reachable positions must be a non-empty exact tuple")
    checked_positions = tuple(
        AdapterPosition(**asdict(item)) for item in reachable_positions
    )
    if len(set(checked_positions)) != len(checked_positions):
        raise ValueError("reachable positions must be unique")
    checked_fallback = _strict_native_pose(fallback_pose, label="fallback pose")
    checked_policy = (
        build_competition_native_camera_policy_v2_9_3()
        if policy is None
        else _strict_policy(policy)
    )
    solver_upright = checked_policy.pose_policy_version in {
        _SOLVER_UPRIGHT_POSE_POLICY_VERSION,
        *_editable_pose_policy_versions(),
    }
    if (
        checked_policy.pose_policy_version
        == _COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION
    ):
        checked_positions = filter_competition_native_camera_positions_v2_9_5(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no collision-safe reachable positions"
            )
    elif checked_policy.pose_policy_version in {
        _CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION,
        _RESET_PER_POSE_EDITABLE_POLICY_VERSION,
    }:
        checked_positions = filter_competition_native_camera_positions_v2_9_6(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no contact-margin-safe reachable positions"
            )
    elif checked_policy.pose_policy_version in {
        _GRID_MARGIN_EDITABLE_POLICY_VERSION,
        _PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION,
    }:
        checked_positions = filter_competition_native_camera_positions_v2_9_7(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no grid-margin-safe reachable positions"
            )
    elif (
        checked_policy.pose_policy_version
        == _SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION
    ):
        checked_positions = filter_competition_native_camera_positions_v2_9_8(
            checked_scene,
            checked_positions,
        )
        if pairs and not checked_positions:
            raise ValueError(
                "camera evidence has no quantization-safe reachable positions"
            )

    poses: set[CompetitionNativeCameraPoseV2_9_3] = set()
    for subject_object_id, support_object_id in pairs:
        subject = checked_scene.object_by_id(subject_object_id)
        checked_scene.object_by_id(support_object_id)
        if (
            not subject.movable
            or subject.support_object_id != support_object_id
            or subject_object_id == support_object_id
        ):
            raise ValueError("camera evidence pair does not bind movable support")
        generated = deterministic_pair_camera_poses(
            checked_scene,
            subject_object_id,
            support_object_id,
            checked_positions,
        )
        poses.update(
            _wire_pose(item)
            for item in generated
            if not solver_upright or item.horizon_degrees == 0.0
        )
        if len(poses) > _MAX_POSE_BANK_MEMBERS:
            raise ValueError("camera evidence pose bank exceeds 256 members")

    if not poses:
        if solver_upright:
            checked_fallback = AdapterPose(
                position=checked_fallback.position,
                yaw_degrees=checked_fallback.yaw_degrees,
                horizon_degrees=0.0,
                standing=checked_fallback.standing,
            )
        poses.add(_wire_pose(checked_fallback))
    return tuple(sorted(poses, key=_pose_key))


def build_competition_native_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the one frozen 2.9.3 source-camera selection policy."""

    payload = _policy_payload(_LEGACY_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_solver_upright_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the bounded camera policy supported by the certified solver."""

    payload = _policy_payload(_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_editable_camera_policy_v2_9_3() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the upright policy that ranks source-native edit coverage."""

    payload = _policy_payload(_EDITABLE_SOLVER_UPRIGHT_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_collision_safe_editable_camera_policy_v2_9_5() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the source-only edit policy with fixed 0.2m movable clearance."""

    payload = _policy_payload(_COLLISION_SAFE_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_contact_margin_editable_camera_policy_v2_9_6() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the edit policy with 0.2m agent plus fixed 1cm margin."""

    payload = _policy_payload(_CONTACT_MARGIN_EDITABLE_POSE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_reset_per_pose_editable_camera_policy_v2_9_7() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the contact-safe policy that resets the source per pose."""

    payload = _policy_payload(_RESET_PER_POSE_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_grid_margin_editable_camera_policy_v2_9_8() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the sequential source policy with one-grid movable margin."""

    payload = _policy_payload(_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_paused_camera_policy_v2_9_9() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the one-grid policy whose observation bank pauses physics."""

    payload = _policy_payload(_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def build_competition_native_settled_camera_policy_v2_9_10() -> (
    CompetitionNativeCameraPolicyV2_9_3
):
    """Return the paused-bank policy that freezes only after final settlement."""

    payload = _policy_payload(_SETTLED_PAUSED_GRID_MARGIN_EDITABLE_POLICY_VERSION)
    return CompetitionNativeCameraPolicyV2_9_3(
        **payload,
        policy_sha256=canonical_sha256(payload, domain=_POLICY_HASH_DOMAIN),
    )


def _strict_pose_bank(
    pose_bank: object,
) -> tuple[CompetitionNativeCameraPoseV2_9_3, ...]:
    if type(pose_bank) is not tuple or any(
        type(item) is not CompetitionNativeCameraPoseV2_9_3 for item in pose_bank
    ):
        raise TypeError("camera evidence pose bank must be an exact pose tuple")
    checked = tuple(
        CompetitionNativeCameraPoseV2_9_3.model_validate(
            item.model_dump(mode="python"), strict=True
        )
        for item in pose_bank
    )
    if not checked or len(checked) > _MAX_POSE_BANK_MEMBERS:
        raise ValueError("camera evidence pose bank must contain 1 through 256 poses")
    if tuple(sorted(set(checked), key=_pose_key)) != checked:
        raise ValueError("camera evidence pose bank is not unique and canonical")
    return checked


def competition_native_camera_pose_bank_sha256_v2_9_3(
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
) -> Sha256Digest:
    """Hash one exact canonical pose bank in its independent domain."""

    checked = _strict_pose_bank(pose_bank)
    payload = {
        "pose_bank_version": "competition-native-camera-pose-bank:2.9.3",
        "poses": tuple(item.model_dump(mode="json") for item in checked),
    }
    return canonical_sha256(payload, domain=_POSE_BANK_HASH_DOMAIN)


def score_competition_native_camera_scene_v2_9_3(
    scene: Scene,
) -> CompetitionNativeCameraScoreV2_9_3:
    """Count qualifying objects whose category occurs once in the scene."""

    checked = _strict_scene(scene)
    checked.camera_by_id("main")
    category_counts: dict[str, int] = {}
    for item in checked.objects:
        category_counts[item.category] = category_counts.get(item.category, 0) + 1

    qualifying = []
    for item in checked.objects:
        view = item.views.get("main")
        if (
            category_counts[item.category] == 1
            and item.request_eligible
            and view is not None
            and view.visible_fraction >= RelationEngine.MIN_VISIBLE_FRACTION
            and view.image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION
            and view.truncated_fraction <= RelationEngine.MAX_TRUNCATED_FRACTION
        ):
            qualifying.append(item)
    return CompetitionNativeCameraScoreV2_9_3(
        movable_scene_unique_category_qualifying_count=sum(
            item.movable for item in qualifying
        ),
        all_scene_unique_category_qualifying_count=len(qualifying),
    )


def _rotation_matrix_values(obb: OBB) -> tuple[tuple[float, float, float], ...]:
    rotation = obb.rotation
    norm = math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera placement OBB rotation is invalid")
    x, y, z, w = (
        rotation.x / norm,
        rotation.y / norm,
        rotation.z / norm,
        rotation.w / norm,
    )
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _translated_obb_fully_visible_v2_9_4(
    scene: Scene,
    subject_object_id: str,
    position: CompetitionNativeCameraPlacementPositionV2_9_4,
) -> bool:
    subject = scene.object_by_id(subject_object_id)
    camera = scene.camera_by_id("main")
    delta = (
        position.x - subject.position.x,
        position.z - subject.position.y,
        position.y - subject.position.z,
    )
    rotation = _rotation_matrix_values(subject.obb)
    center = (
        subject.obb.center.x,
        subject.obb.center.y,
        subject.obb.center.z,
    )
    matrix = camera.world_to_camera
    fx, fy = camera.intrinsics[0], camera.intrinsics[4]
    cx, cy = camera.intrinsics[2], camera.intrinsics[5]
    projected: list[tuple[float, float]] = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                local = (
                    x_sign * subject.obb.extent.x / 2.0,
                    y_sign * subject.obb.extent.y / 2.0,
                    z_sign * subject.obb.extent.z / 2.0,
                )
                world = tuple(
                    center[axis]
                    + delta[axis]
                    + sum(rotation[axis][inner] * local[inner] for inner in range(3))
                    for axis in range(3)
                )
                homogeneous = tuple(
                    sum(matrix[row * 4 + column] * world[column] for column in range(3))
                    + matrix[row * 4 + 3]
                    for row in range(4)
                )
                if homogeneous[3] == 0.0:
                    raise ValueError("camera placement projected a point to infinity")
                camera_x = homogeneous[0] / homogeneous[3]
                camera_y = homogeneous[1] / homogeneous[3]
                camera_z = homogeneous[2] / homogeneous[3]
                if camera_z <= 1e-12:
                    return False
                projected.append(
                    (
                        fx * camera_x / camera_z + cx,
                        cy - fy * camera_y / camera_z,
                    )
                )
    min_x = min(item[0] for item in projected)
    max_x = max(item[0] for item in projected)
    min_y = min(item[1] for item in projected)
    max_y = max(item[1] for item in projected)
    if not (
        min_x >= 0.5
        and max_x <= camera.width - 0.5
        and min_y >= 0.5
        and max_y <= camera.height - 0.5
    ):
        return False
    image_area_fraction = (
        (max_x - min_x) * (max_y - min_y) / (camera.width * camera.height)
    )
    return image_area_fraction >= RelationEngine.MIN_IMAGE_AREA_FRACTION


def score_competition_native_editable_camera_scene_v2_9_4(
    scene: Scene,
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
) -> CompetitionNativeCameraScoreV2_9_4:
    """Score one frozen scene by complete visible native edit coverage."""

    observed = _strict_scene(scene)
    base = score_competition_native_camera_scene_v2_9_3(observed)
    roster = _strict_placement_roster_v2_9_4(placement_roster)
    category_counts: dict[str, int] = {}
    for item in observed.objects:
        category_counts[item.category] = category_counts.get(item.category, 0) + 1

    visible_subjects = 0
    visible_positions = 0
    for entry in roster:
        subject = observed.object_by_id(entry.subject_object_id)
        observed.object_by_id(entry.support_object_id)
        view = subject.views.get("main")
        if (
            not subject.movable
            or not subject.request_eligible
            or subject.support_object_id != entry.support_object_id
            or category_counts[subject.category] != 1
            or view is None
            or view.visible_fraction < RelationEngine.MIN_VISIBLE_FRACTION
            or view.image_area_fraction < RelationEngine.MIN_IMAGE_AREA_FRACTION
            or view.truncated_fraction > RelationEngine.MAX_TRUNCATED_FRACTION
        ):
            continue
        count = sum(
            _translated_obb_fully_visible_v2_9_4(
                observed,
                entry.subject_object_id,
                position,
            )
            for position in entry.positions
        )
        visible_positions += count
        visible_subjects += count > 0
    return CompetitionNativeCameraScoreV2_9_4(
        placement_roster_sha256=(
            competition_native_camera_placement_roster_sha256_v2_9_4(roster)
        ),
        visible_native_placement_subject_count=visible_subjects,
        visible_native_placement_count=visible_positions,
        movable_scene_unique_category_qualifying_count=(
            base.movable_scene_unique_category_qualifying_count
        ),
        all_scene_unique_category_qualifying_count=(
            base.all_scene_unique_category_qualifying_count
        ),
    )


def score_competition_native_editable_camera_application_v2_9_4(
    *,
    source_scene: Scene,
    pose: CompetitionNativeCameraPoseV2_9_3,
    application: AdapterCameraApplication,
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
    policy: CompetitionNativeCameraPolicyV2_9_3,
) -> CompetitionNativeCameraScoreV2_9_4:
    """Close and score one source observation by native edit coverage."""

    checked_policy = _strict_policy(policy)
    if checked_policy.pose_policy_version not in _editable_pose_policy_versions():
        raise ValueError("editable camera score requires the edit-domain policy")
    score_competition_native_source_camera_application_v2_9_3(
        source_scene=source_scene,
        pose=pose,
        application=application,
        policy=checked_policy,
    )
    checked_application = _strict_application(application)
    return score_competition_native_editable_camera_scene_v2_9_4(
        checked_application.observed_scene,
        placement_roster,
    )


def _strict_application(value: object) -> AdapterCameraApplication:
    if type(value) is not AdapterCameraApplication:
        raise TypeError("camera evidence application must be exact")
    requested_pose = _strict_native_pose(value.requested_pose, label="requested pose")
    observed_pose = _strict_native_pose(value.observed_pose, label="observed pose")
    if type(value.observed_camera_position) is not AdapterPosition:
        raise TypeError("camera evidence observed camera position must be exact")
    observed_position = AdapterPosition(**asdict(value.observed_camera_position))
    observed_scene = _strict_scene(value.observed_scene)
    if type(value.observation) is not AdapterObservation:
        raise TypeError("camera evidence observation must be exact")
    observation = value.observation
    if type(observation.scene) is not Scene:
        raise TypeError("camera evidence observation scene must be exact")
    if any(
        type(blob) is not bytes
        for blob in (
            observation.rgb_png,
            observation.depth_npy,
            observation.instance_png,
            observation.pointcloud_ply,
        )
    ):
        raise TypeError("camera evidence observation assets must be exact bytes")
    if type(observation.is_settled) is not bool or any(
        type(key) is not str or type(count) is not int or count < 0
        for key, count in observation.instance_pixel_counts
    ):
        raise TypeError("camera evidence observation metadata must be exact")
    rebuilt_observation = AdapterObservation.create(
        scene=_strict_scene(observation.scene),
        rgb_png=observation.rgb_png,
        depth_npy=observation.depth_npy,
        instance_png=observation.instance_png,
        pointcloud_ply=observation.pointcloud_ply,
        instance_pixel_counts=observation.instance_pixel_counts,
        is_settled=observation.is_settled,
        instance_colors=observation.instance_colors,
        instance_evidence_provenance=observation.instance_evidence_provenance,
    )
    if (
        rebuilt_observation != observation
        or rebuilt_observation.scene != observed_scene
    ):
        raise ValueError("camera evidence observation binding or asset hash changed")
    residuals = (
        value.position_residual_m,
        value.yaw_residual_degrees,
        value.horizon_residual_degrees,
    )
    if any(type(item) is not float or not math.isfinite(item) for item in residuals):
        raise TypeError(
            "camera evidence application residuals must be exact finite floats"
        )
    expected_residuals = (
        math.dist(
            (
                requested_pose.position.x,
                requested_pose.position.y,
                requested_pose.position.z,
            ),
            (
                observed_pose.position.x,
                observed_pose.position.y,
                observed_pose.position.z,
            ),
        ),
        abs(
            (observed_pose.yaw_degrees - requested_pose.yaw_degrees + 180.0) % 360.0
            - 180.0
        ),
        abs(observed_pose.horizon_degrees - requested_pose.horizon_degrees),
    )
    if any(
        not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=4.0 * max(math.ulp(actual), math.ulp(expected)),
        )
        for actual, expected in zip(residuals, expected_residuals, strict=True)
    ):
        raise ValueError("camera evidence application residual changed")
    if (
        value.position_residual_m > _MAX_POSITION_RESIDUAL_M
        or value.yaw_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
        or value.horizon_residual_degrees > _MAX_ANGLE_RESIDUAL_DEGREES
    ):
        raise ValueError("camera evidence application pose drift exceeds tolerance")
    if observed_pose.standing is not requested_pose.standing:
        raise ValueError("camera evidence application standing state changed")
    if not rebuilt_observation.is_settled:
        raise ValueError("camera evidence application is not at rest")
    return AdapterCameraApplication(
        source=value.source,
        binding=value.binding,
        requested_pose=requested_pose,
        observed_pose=observed_pose,
        observed_camera_position=observed_position,
        observed_scene=observed_scene,
        observation=rebuilt_observation,
        position_residual_m=value.position_residual_m,
        yaw_residual_degrees=value.yaw_residual_degrees,
        horizon_residual_degrees=value.horizon_residual_degrees,
    )


def _validate_application_source_closure(
    source_scene: Scene,
    application: AdapterCameraApplication,
) -> None:
    observed = application.observed_scene
    if (
        observed.scene_id != source_scene.scene_id
        or observed.source != source_scene.source
        or observed.coordinate_system != source_scene.coordinate_system
        or observed.room_polygon_xy != source_scene.room_polygon_xy
        or observed.collision_obstacles != source_scene.collision_obstacles
        or observed.subject_position_regions != source_scene.subject_position_regions
        or observed.pinned_object_ids != source_scene.pinned_object_ids
        or observed.generation_seed != source_scene.generation_seed
    ):
        raise ValueError("camera evidence application source root changed")

    source_object_ids = tuple(item.object_id for item in source_scene.objects)
    observed_object_ids = tuple(item.object_id for item in observed.objects)
    if source_object_ids != observed_object_ids:
        raise ValueError("camera evidence application source object roster changed")
    for source_object, observed_object in zip(
        source_scene.objects, observed.objects, strict=True
    ):
        if (
            observed_object.object_id != source_object.object_id
            or observed_object.name != source_object.name
            or observed_object.category != source_object.category
            or observed_object.movable is not source_object.movable
            or observed_object.request_eligible is not source_object.request_eligible
            or observed_object.support_object_id != source_object.support_object_id
            or observed_object.position != source_object.position
            or observed_object.rotation != source_object.rotation
            or observed_object.obb != source_object.obb
        ):
            raise ValueError("camera evidence application source object changed")

    source_camera_ids = tuple(item.camera_id for item in source_scene.cameras)
    observed_camera_ids = tuple(item.camera_id for item in observed.cameras)
    if source_camera_ids != observed_camera_ids:
        raise ValueError("camera evidence application source camera roster changed")
    for source_camera, observed_camera in zip(
        source_scene.cameras, observed.cameras, strict=True
    ):
        if (
            observed_camera.camera_id != source_camera.camera_id
            or observed_camera.width != source_camera.width
            or observed_camera.height != source_camera.height
            or observed_camera.intrinsics != source_camera.intrinsics
            or (
                source_camera.camera_id != "main"
                and observed_camera.world_to_camera != source_camera.world_to_camera
            )
        ):
            raise ValueError("camera evidence application source camera changed")


def score_competition_native_source_camera_application_v2_9_3(
    *,
    source_scene: Scene,
    pose: CompetitionNativeCameraPoseV2_9_3,
    application: AdapterCameraApplication,
    policy: CompetitionNativeCameraPolicyV2_9_3 | None = None,
) -> CompetitionNativeCameraScoreV2_9_3:
    """Close and score one bank-aligned application without retaining siblings."""

    source = _strict_scene(source_scene)
    source.camera_by_id("main")
    if type(pose) is not CompetitionNativeCameraPoseV2_9_3:
        raise TypeError("camera evidence pose must be exact")
    checked_pose = CompetitionNativeCameraPoseV2_9_3.model_validate(
        pose.model_dump(mode="python"), strict=True
    )
    checked_application = _strict_application(application)
    if _wire_pose(checked_application.requested_pose) != checked_pose:
        raise ValueError("camera evidence application is not bank-aligned")
    if checked_application.observed_scene.scene_id != source.scene_id:
        raise ValueError("camera evidence application scene identity mismatch")
    _validate_application_source_closure(source, checked_application)
    observed_camera = checked_application.observed_scene.camera_by_id("main")
    if policy is not None:
        verify_competition_native_solver_camera_binding_v2_9_3(
            policy,
            checked_pose,
            observed_camera,
        )
    return score_competition_native_camera_scene_v2_9_3(
        checked_application.observed_scene
    )


def _strict_policy(
    policy: object,
) -> CompetitionNativeCameraPolicyV2_9_3:
    if type(policy) is not CompetitionNativeCameraPolicyV2_9_3:
        raise TypeError("camera evidence policy must be exact")
    return CompetitionNativeCameraPolicyV2_9_3.model_validate(
        policy.model_dump(mode="python"), strict=True
    )


def verify_competition_native_solver_camera_binding_v2_9_3(
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose: CompetitionNativeCameraPoseV2_9_3,
    camera: Camera,
) -> None:
    """Require exact upright camera semantics only for the solver policy."""

    checked_policy = _strict_policy(policy)
    if type(pose) is not CompetitionNativeCameraPoseV2_9_3:
        raise TypeError("solver camera pose must be exact")
    checked_pose = CompetitionNativeCameraPoseV2_9_3.model_validate(
        pose.model_dump(mode="python"), strict=True
    )
    if type(camera) is not Camera or camera.camera_id != "main":
        raise TypeError("solver camera must be exact main")
    checked_camera = Camera.model_validate(
        camera.model_dump(mode="python"), strict=True
    )
    if checked_policy.pose_policy_version == _LEGACY_POSE_POLICY_VERSION:
        return
    if checked_pose.horizon_degrees != 0.0:
        raise ValueError("solver-upright camera pose must use horizon zero")
    try:
        _legacy_camera(checked_camera.world_to_camera)
    except _CameraConversionError as error:
        raise ValueError("solver-upright camera matrix is unsupported") from error


build_settled_camera_policy = build_competition_native_settled_camera_policy_v2_9_10


# Preserve supported public names and pickle lookup.
_strict_native_pose.__module__ = "spatialcf.generation.capture.models"
_wire_pose.__module__ = "spatialcf.generation.capture.models"
_pose_key.__module__ = "spatialcf.generation.capture.models"
build_competition_native_camera_pose_bank_v2_9_3.__module__ = "spatialcf.generation.capture.models"
build_competition_native_camera_policy_v2_9_3.__module__ = "spatialcf.generation.capture.models"
build_competition_native_solver_upright_camera_policy_v2_9_3.__module__ = "spatialcf.generation.capture.models"
build_competition_native_editable_camera_policy_v2_9_3.__module__ = "spatialcf.generation.capture.models"
build_competition_native_collision_safe_editable_camera_policy_v2_9_5.__module__ = "spatialcf.generation.capture.models"
build_competition_native_contact_margin_editable_camera_policy_v2_9_6.__module__ = "spatialcf.generation.capture.models"
build_competition_native_reset_per_pose_editable_camera_policy_v2_9_7.__module__ = "spatialcf.generation.capture.models"
build_competition_native_grid_margin_editable_camera_policy_v2_9_8.__module__ = "spatialcf.generation.capture.models"
build_competition_native_paused_camera_policy_v2_9_9.__module__ = "spatialcf.generation.capture.models"
build_competition_native_settled_camera_policy_v2_9_10.__module__ = "spatialcf.generation.capture.models"
_strict_pose_bank.__module__ = "spatialcf.generation.capture.models"
competition_native_camera_pose_bank_sha256_v2_9_3.__module__ = "spatialcf.generation.capture.models"
score_competition_native_camera_scene_v2_9_3.__module__ = "spatialcf.generation.capture.models"
_rotation_matrix_values.__module__ = "spatialcf.generation.capture.models"
_translated_obb_fully_visible_v2_9_4.__module__ = "spatialcf.generation.capture.models"
score_competition_native_editable_camera_scene_v2_9_4.__module__ = "spatialcf.generation.capture.models"
score_competition_native_editable_camera_application_v2_9_4.__module__ = "spatialcf.generation.capture.models"
_strict_application.__module__ = "spatialcf.generation.capture.models"
_validate_application_source_closure.__module__ = "spatialcf.generation.capture.models"
score_competition_native_source_camera_application_v2_9_3.__module__ = "spatialcf.generation.capture.models"
_strict_policy.__module__ = "spatialcf.generation.capture.models"
verify_competition_native_solver_camera_binding_v2_9_3.__module__ = "spatialcf.generation.capture.models"
