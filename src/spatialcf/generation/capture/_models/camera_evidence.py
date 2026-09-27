"""Capture camera evidence: exact contracts and pure derivation."""

from __future__ import annotations

from spatialcf.adapters.base import (
    AdapterCameraApplication,
)

from spatialcf.domain.scene import (
    Camera,
    Scene,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.generation.capture._models.camera_contracts import (
    CompetitionNativeCameraPlacementRosterEntryV2_9_4,
    CompetitionNativeCameraPolicyV2_9_3,
    CompetitionNativeCameraPoseV2_9_3,
    CompetitionNativeCameraScoreV2_9_3,
    CompetitionNativeCameraScoreV2_9_4,
    CompetitionNativeSourceCameraEvidenceV2_9_3,
    _editable_pose_policy_versions,
    _evidence_payload,
    _strict_placement_roster_v2_9_4,
    _strict_scene,
    _strict_score_ledger,
    select_competition_native_camera_score_index_v2_9_3,
    select_competition_native_camera_score_index_v2_9_4,
)

from spatialcf.generation.capture._models.camera_scoring import (
    _strict_application,
    _strict_policy,
    _strict_pose_bank,
    _validate_application_source_closure,
    _wire_pose,
    competition_native_camera_pose_bank_sha256_v2_9_3,
    score_competition_native_camera_scene_v2_9_3,
    score_competition_native_editable_camera_application_v2_9_4,
    score_competition_native_source_camera_application_v2_9_3,
    verify_competition_native_solver_camera_binding_v2_9_3,
)

from spatialcf.generation.capture._models.constants import (
    _EVIDENCE_HASH_DOMAIN,
)


def build_competition_native_source_camera_evidence_v2_9_3(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_3, ...],
    selected_application: AdapterCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Build evidence from a complete light ledger and one replayed winner."""

    checked_source_scene = _strict_scene(source_scene)
    if checked_source_scene.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity mismatch")
    checked_source_scene.camera_by_id("main")
    checked_policy = _strict_policy(policy)
    checked_bank = _strict_pose_bank(pose_bank)
    checked_scores = _strict_score_ledger(pose_scores)
    if len(checked_scores) != len(checked_bank):
        raise ValueError("camera evidence score ledger is not bank-aligned")
    selected_index = select_competition_native_camera_score_index_v2_9_3(checked_scores)
    selected_pose = checked_bank[selected_index]
    selected = _strict_application(selected_application)
    selected_score = score_competition_native_source_camera_application_v2_9_3(
        source_scene=checked_source_scene,
        pose=selected_pose,
        application=selected,
    )
    if selected_score != checked_scores[selected_index]:
        raise ValueError("camera evidence replay score differs from frozen score")

    observed_pose = _wire_pose(selected.observed_pose)
    observed_position = (
        selected.observed_camera_position.x,
        selected.observed_camera_position.y,
        selected.observed_camera_position.z,
    )
    camera = selected.observed_scene.camera_by_id("main")
    observation = selected.observation
    pose_bank_sha256 = competition_native_camera_pose_bank_sha256_v2_9_3(checked_bank)
    payload = _evidence_payload(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        policy_sha256=checked_policy.policy_sha256,
        pose_bank_sha256=pose_bank_sha256,
        pose_bank_count=len(checked_bank),
        pose_scores=checked_scores,
        selected_pose_index=selected_index,
        requested_pose=selected_pose,
        observed_pose=observed_pose,
        observed_native_camera_position=observed_position,
        camera=camera,
        score=selected_score,
        rgb_png_sha256=observation.rgb_png_sha256,
        depth_npy_sha256=observation.depth_npy_sha256,
        instance_png_sha256=observation.instance_png_sha256,
        pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
        is_scene_at_rest=observation.is_settled,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256(payload, domain=_EVIDENCE_HASH_DOMAIN),
    )


def build_competition_native_source_camera_evidence_v2_9_4(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    pose_scores: tuple[CompetitionNativeCameraScoreV2_9_4, ...],
    placement_roster: tuple[CompetitionNativeCameraPlacementRosterEntryV2_9_4, ...],
    selected_application: AdapterCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Build source evidence whose winner maximizes visible edit coverage."""

    checked_source_scene = _strict_scene(source_scene)
    if checked_source_scene.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity mismatch")
    checked_source_scene.camera_by_id("main")
    checked_policy = _strict_policy(policy)
    if checked_policy.pose_policy_version not in _editable_pose_policy_versions():
        raise ValueError("editable camera evidence requires the edit-domain policy")
    checked_bank = _strict_pose_bank(pose_bank)
    checked_scores = _strict_score_ledger(pose_scores)
    if any(
        type(item) is not CompetitionNativeCameraScoreV2_9_4 for item in checked_scores
    ):
        raise TypeError("editable camera evidence requires 2.9.4 scores")
    if len(checked_scores) != len(checked_bank):
        raise ValueError("camera evidence score ledger is not bank-aligned")
    checked_roster = _strict_placement_roster_v2_9_4(placement_roster)
    selected_index = select_competition_native_camera_score_index_v2_9_4(
        checked_scores  # type: ignore[arg-type]
    )
    selected_pose = checked_bank[selected_index]
    selected = _strict_application(selected_application)
    selected_score = score_competition_native_editable_camera_application_v2_9_4(
        source_scene=checked_source_scene,
        pose=selected_pose,
        application=selected,
        placement_roster=checked_roster,
        policy=checked_policy,
    )
    if selected_score != checked_scores[selected_index]:
        raise ValueError("camera evidence replay score differs from frozen score")

    observed_pose = _wire_pose(selected.observed_pose)
    observed_position = (
        selected.observed_camera_position.x,
        selected.observed_camera_position.y,
        selected.observed_camera_position.z,
    )
    camera = selected.observed_scene.camera_by_id("main")
    observation = selected.observation
    payload = _evidence_payload(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        policy_sha256=checked_policy.policy_sha256,
        pose_bank_sha256=competition_native_camera_pose_bank_sha256_v2_9_3(
            checked_bank
        ),
        pose_bank_count=len(checked_bank),
        pose_scores=checked_scores,
        selected_pose_index=selected_index,
        requested_pose=selected_pose,
        observed_pose=observed_pose,
        observed_native_camera_position=observed_position,
        camera=camera,
        score=selected_score,
        rgb_png_sha256=observation.rgb_png_sha256,
        depth_npy_sha256=observation.depth_npy_sha256,
        instance_png_sha256=observation.instance_png_sha256,
        pointcloud_ply_sha256=observation.pointcloud_ply_sha256,
        is_scene_at_rest=observation.is_settled,
    )
    return CompetitionNativeSourceCameraEvidenceV2_9_3(
        **payload,
        camera_evidence_sha256=canonical_sha256(payload, domain=_EVIDENCE_HASH_DOMAIN),
    )


def select_competition_native_source_camera_evidence_v2_9_3(
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    source_scene: Scene,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    applications: tuple[AdapterCameraApplication, ...],
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Select one complete source observation using literal scene scores only."""

    checked_source_scene = _strict_scene(source_scene)
    checked_bank = _strict_pose_bank(pose_bank)
    if type(applications) is not tuple or any(
        type(item) is not AdapterCameraApplication for item in applications
    ):
        raise TypeError("camera evidence applications must be an exact tuple")
    if len(applications) != len(checked_bank):
        raise ValueError("camera evidence applications are not bank-aligned")
    checked_applications = tuple(_strict_application(item) for item in applications)
    scores: list[CompetitionNativeCameraScoreV2_9_3] = []
    for index, (pose, application) in enumerate(
        zip(checked_bank, checked_applications, strict=True)
    ):
        try:
            score = score_competition_native_source_camera_application_v2_9_3(
                source_scene=checked_source_scene,
                pose=pose,
                application=application,
                policy=policy,
            )
        except ValueError as error:
            if "bank-aligned" in str(error):
                raise ValueError(
                    f"camera evidence application {index} is not bank-aligned"
                ) from error
            raise
        scores.append(score)

    pose_scores = tuple(scores)
    selected_index = select_competition_native_camera_score_index_v2_9_3(pose_scores)
    return build_competition_native_source_camera_evidence_v2_9_3(
        source_id=source_id,
        scene_id=scene_id,
        source_locator_sha256=source_locator_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        source_capture_sha256=source_capture_sha256,
        source_scene=checked_source_scene,
        policy=policy,
        pose_bank=checked_bank,
        pose_scores=pose_scores,
        selected_application=checked_applications[selected_index],
    )


def verify_competition_native_source_camera_evidence_v2_9_3(
    evidence: CompetitionNativeSourceCameraEvidenceV2_9_3,
    *,
    source_id: str,
    scene_id: str,
    source_locator_sha256: str,
    runtime_identity_sha256: str,
    source_capture_sha256: str,
    policy: CompetitionNativeCameraPolicyV2_9_3,
    pose_bank: tuple[CompetitionNativeCameraPoseV2_9_3, ...],
    source_scene: Scene,
    selected_scene: Scene,
    selected_camera: Camera,
    selected_application: AdapterCameraApplication,
) -> CompetitionNativeSourceCameraEvidenceV2_9_3:
    """Close persisted evidence against independently supplied capture bindings."""

    if type(evidence) is not CompetitionNativeSourceCameraEvidenceV2_9_3:
        raise TypeError("camera evidence must be exact")
    checked = CompetitionNativeSourceCameraEvidenceV2_9_3.model_validate(
        evidence.model_dump(mode="python"), strict=True
    )
    checked_policy = _strict_policy(policy)
    checked_bank = _strict_pose_bank(pose_bank)
    source = _strict_scene(source_scene)
    scene = _strict_scene(selected_scene)
    if type(selected_camera) is not Camera:
        raise TypeError("camera evidence selected camera must be exact")
    camera = Camera.model_validate(
        selected_camera.model_dump(mode="python"), strict=True
    )
    application = _strict_application(selected_application)
    if source.scene_id != scene_id:
        raise ValueError("camera evidence source scene identity does not close")
    _validate_application_source_closure(source, application)
    expected_lineage = (
        source_id,
        scene_id,
        source_locator_sha256,
        runtime_identity_sha256,
        source_capture_sha256,
    )
    if expected_lineage != (
        checked.source_id,
        checked.scene_id,
        checked.source_locator_sha256,
        checked.runtime_identity_sha256,
        checked.source_capture_sha256,
    ):
        raise ValueError("camera evidence lineage does not close")
    if (
        checked.policy_sha256 != checked_policy.policy_sha256
        or checked.pose_bank_sha256
        != competition_native_camera_pose_bank_sha256_v2_9_3(checked_bank)
        or checked.pose_bank_count != len(checked_bank)
        or checked.selected_pose_index >= len(checked_bank)
        or checked_bank[checked.selected_pose_index] != checked.requested_pose
        or checked.requested_pose != _wire_pose(application.requested_pose)
    ):
        raise ValueError("camera evidence policy or pose bank does not close")
    if (
        scene.scene_id != scene_id
        or application.observed_scene != scene
        or application.observation.scene != scene
        or scene.camera_by_id("main") != camera
        or checked.camera != camera
        or checked.observed_pose != _wire_pose(application.observed_pose)
        or checked.observed_native_camera_position
        != (
            application.observed_camera_position.x,
            application.observed_camera_position.y,
            application.observed_camera_position.z,
        )
        or checked.score != score_competition_native_camera_scene_v2_9_3(scene)
        or checked.rgb_png_sha256 != application.observation.rgb_png_sha256
        or checked.depth_npy_sha256 != application.observation.depth_npy_sha256
        or checked.instance_png_sha256 != application.observation.instance_png_sha256
        or checked.pointcloud_ply_sha256
        != application.observation.pointcloud_ply_sha256
        or checked.is_scene_at_rest is not application.observation.is_settled
    ):
        raise ValueError("camera evidence selected capture does not close")
    verify_competition_native_solver_camera_binding_v2_9_3(
        checked_policy,
        checked.requested_pose,
        camera,
    )
    return checked


verify_source_camera_evidence = verify_competition_native_source_camera_evidence_v2_9_3


# Preserve supported public names and pickle lookup.
build_competition_native_source_camera_evidence_v2_9_3.__module__ = "spatialcf.generation.capture.models"
build_competition_native_source_camera_evidence_v2_9_4.__module__ = "spatialcf.generation.capture.models"
select_competition_native_source_camera_evidence_v2_9_3.__module__ = "spatialcf.generation.capture.models"
verify_competition_native_source_camera_evidence_v2_9_3.__module__ = "spatialcf.generation.capture.models"
