"""Strict source authentication and immutable proxy preparation."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from inspect import get_annotations as _get_annotations
from typing import get_type_hints as _get_type_hints

from pydantic import BaseModel

from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.request import InterventionSpec
from spatialcf.domain.scene import OBB, Scene
from spatialcf.generation.capture.models import (
    CompetitionNativeSourceCaptureV2_9,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    verify_source_surface_evidence,
)
from spatialcf.generation.planning.models import _PROXY_POLICY_SHA256
from spatialcf.generation.planning.proxy_conversion import (
    _convert_bbox_proxy_scene as _convert_bbox_proxy_scene,
    _convert_target_proxy_scene as _convert_target_proxy_scene,
    _ProxyChallenge as _ProxyChallenge,
)
from spatialcf.generation.planning.proxy_geometry import (
    _fixed_margin_collision_proxies,
    _strict_legacy,
    _strict_v2,
)
from spatialcf.verification.integrity import competition_legacy_sha256

_TARGET_PROPOSAL_POLICY_SHA256 = hashlib.sha256(
    b"spatialcf.competition-native-scene-proxy.v2.9.1\0"
    b"obb-frame;explicit-endpoint-only-workspace;l1-radius-prune;"
    b"subject-bottom-support-plane;support-solid-top-clamp;"
    b"fixed-non-support-obb-margin-1cm;existing-clearance-max"
).hexdigest()


@dataclass(frozen=True, slots=True)
class _PreparedProxy:
    scene: Scene
    intervention: InterventionSpec
    case_id: str
    base_problem: SemanticProblemV2_3
    legacy_scene_sha256: str
    intervention_sha256: str
    collision_proxies: tuple[tuple[str, OBB], ...]
    minimum_margin_native_object_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PreparedCurrentProxy:
    base: _PreparedProxy
    source_capture: CompetitionNativeSourceCaptureV2_9
    source_surface_evidence: SourceSurfaceEvidence
    subject_surface_evidence: SubjectSurfaceEvidence


def _verified_proxy_inputs(
    scene: Scene,
    intervention: InterventionSpec,
    source_capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    *,
    case_id: str,
) -> tuple[
    Scene,
    InterventionSpec,
    CompetitionNativeSourceCaptureV2_9,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
]:
    checked_capture = _strict_v2(
        source_capture,
        CompetitionNativeSourceCaptureV2_9,
        "source_capture",
    )
    verified_source_evidence = verify_source_surface_evidence(
        checked_capture,
        source_surface_evidence,
    )
    checked_evidence = _strict_v2(
        subject_surface_evidence,
        SubjectSurfaceEvidence,
        "subject_surface_evidence",
    )
    checked_scene = _strict_legacy(scene, Scene, "scene")
    checked_intervention = _strict_legacy(
        intervention,
        InterventionSpec,
        "intervention",
    )
    if type(case_id) is not str or not case_id:
        raise TypeError("case_id must be a non-empty exact string")
    source_subjects = tuple(
        item
        for item in verified_source_evidence.subjects
        if item.subject_object_id == checked_intervention.subject_id
    )
    if len(source_subjects) != 1 or source_subjects[0] != checked_evidence:
        raise ValueError(
            "subject surface evidence does not bind verified capture facts"
        )
    if checked_scene != checked_capture.scene:
        raise ValueError("native proxy scene does not bind verified capture facts")
    if (
        checked_evidence.subject_object_id != checked_intervention.subject_id
        or checked_evidence.support_object_id
        != checked_scene.object_by_id(checked_intervention.subject_id).support_object_id
        or checked_evidence.subject_object_id
        not in {item.object_id for item in checked_scene.objects}
        or checked_evidence.support_object_id
        not in {item.object_id for item in checked_scene.objects}
    ):
        raise ValueError("subject surface evidence does not bind the native proxy")
    return (
        checked_scene,
        checked_intervention,
        checked_capture,
        verified_source_evidence,
        checked_evidence,
    )


def _prepare_proxy(
    scene: Scene,
    intervention: InterventionSpec,
    source_capture: CompetitionNativeSourceCaptureV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    *,
    case_id: str,
) -> _PreparedCurrentProxy:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        (
            checked_scene,
            checked_intervention,
            checked_capture,
            verified_source_evidence,
            checked_evidence,
        ) = _verified_proxy_inputs(
            scene,
            intervention,
            source_capture,
            source_surface_evidence,
            subject_surface_evidence,
            case_id=case_id,
        )
        base = _prepare_base_proxy(
            checked_scene,
            checked_intervention,
            case_id=case_id,
            bbox_visibility=True,
        )
    return _PreparedCurrentProxy(
        base=base,
        source_capture=checked_capture,
        source_surface_evidence=verified_source_evidence,
        subject_surface_evidence=checked_evidence,
    )


def _prepare_base_proxy(
    scene: Scene,
    intervention: InterventionSpec,
    *,
    case_id: str,
    bbox_visibility: bool = False,
) -> _PreparedProxy:
    checked_scene = _strict_legacy(scene, Scene, "scene")
    checked_intervention = _strict_legacy(
        intervention,
        InterventionSpec,
        "intervention",
    )
    if not isinstance(case_id, str):
        raise TypeError("case_id must be an exact str")
    if type(bbox_visibility) is not bool:
        raise TypeError("bbox_visibility must be an exact bool")

    subject = checked_scene.object_by_id(checked_intervention.subject_id)
    if not subject.movable:
        raise ValueError("native subject must be movable")
    checked_scene.object_by_id(checked_intervention.reference_id)
    checked_scene.camera_by_id(checked_intervention.camera_id)
    if checked_intervention.camera_id not in subject.views:
        raise ValueError("subject requires an exact baseline view")
    reference = checked_scene.object_by_id(checked_intervention.reference_id)
    if checked_intervention.camera_id not in reference.views:
        raise ValueError("reference requires an exact baseline view")

    proxy_scene = checked_scene.model_copy(
        update={
            "objects": tuple(
                item.model_copy(
                    update={
                        "movable": item.object_id == subject.object_id,
                        "position": item.obb.center,
                        "rotation": item.obb.rotation,
                    }
                )
                for item in checked_scene.objects
            )
        }
    )
    proxy_scene = Scene.model_validate(
        proxy_scene.model_dump(mode="python"),
        strict=True,
    )
    legacy_sha = _legacy_sha256(checked_scene)
    intervention_sha = _legacy_sha256(checked_intervention)
    transient = _ProxyChallenge(
        case_id=case_id,
        direction=(
            f"{checked_intervention.relation_before.value}_to_"
            f"{checked_intervention.relation_after.value}"
        ),
        archetype="native-scene-proxy",
        expected_outcome="SAT",
        scene=proxy_scene,
        intervention=checked_intervention,
        scene_source_sha256=legacy_sha,
        intervention_source_sha256=intervention_sha,
        expectation_source_sha256=(
            _PROXY_POLICY_SHA256 if bbox_visibility else _TARGET_PROPOSAL_POLICY_SHA256
        ),
    )
    base_problem = (
        _convert_bbox_proxy_scene(transient)
        if bbox_visibility
        else _convert_target_proxy_scene(transient)
    )
    collision_proxy_by_native_id, minimum_margin_ids = _fixed_margin_collision_proxies(
        checked_scene,
        subject_object_id=subject.object_id,
        support_object_id=subject.support_object_id,
    )
    return _PreparedProxy(
        scene=checked_scene,
        intervention=checked_intervention,
        case_id=case_id,
        base_problem=base_problem,
        legacy_scene_sha256=legacy_sha,
        intervention_sha256=intervention_sha,
        collision_proxies=tuple(sorted(collision_proxy_by_native_id.items())),
        minimum_margin_native_object_ids=frozenset(minimum_margin_ids),
    )


def _legacy_sha256(value: BaseModel) -> str:
    """Hash one legacy model after applying the v2 finite-number convention."""
    return competition_legacy_sha256(value)


# Resolve declared annotations before restoring legacy type identity.
for _owned_type in (
    _PreparedProxy,
    _PreparedCurrentProxy,
):
    _owned_type.__annotations__ = {
        key: hint
        for key, hint in _get_type_hints(_owned_type, include_extras=True).items()
        if key in _get_annotations(_owned_type)
    }
del _owned_type
_PreparedProxy.__module__ = "spatialcf.generation.planning.problem"
_PreparedCurrentProxy.__module__ = "spatialcf.generation.planning.problem"
_verified_proxy_inputs.__module__ = "spatialcf.generation.planning.problem"
_prepare_proxy.__module__ = "spatialcf.generation.planning.problem"
_prepare_base_proxy.__module__ = "spatialcf.generation.planning.problem"
_legacy_sha256.__module__ = "spatialcf.generation.planning.problem"
