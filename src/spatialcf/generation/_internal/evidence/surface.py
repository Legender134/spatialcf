"""Capture-bound receptacle surface evidence for the current roster."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import (
    AI2ThorNativePosition,
    AI2ThorReceptacleSpawnMap,
    AI2ThorReceptacleSurfacePatch,
    AI2ThorRuntimeIdentity,
    build_receptacle_support_position_region,
)
from spatialcf.domain.v2.base import FiniteFloat, Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import canonical_sha256_v2

if TYPE_CHECKING:
    from spatialcf.generation.capture.models import (
        CompetitionNativeSourceCaptureV2_9,
    )

_PATCH_HASH_DOMAIN = "spatialcf.competition-native-receptacle-surface-patch.v2.9.2"
_SUBJECT_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-subject-surface-evidence.v2.9.2"
)
_SOURCE_EVIDENCE_HASH_DOMAIN = (
    "spatialcf.competition-native-source-surface-evidence.v2.9.2"
)
_RUNTIME_IDENTITY_HASH_DOMAIN = "spatialcf.competition-native-runtime-identity.v2.9.2"


def _patch_payload(
    *,
    patch_index: int,
    x_min: float,
    x_max: float,
    native_y: float,
    z_min: float,
    z_max: float,
) -> dict[str, object]:
    return {
        "native_y": native_y,
        "patch_index": patch_index,
        "x_max": x_max,
        "x_min": x_min,
        "z_max": z_max,
        "z_min": z_min,
    }


class CompetitionNativeReceptacleSurfacePatchV2_9_2(V2Model):
    """One ordered raw 21x21 trigger-grid patch and its independent digest."""

    patch_index: int = Field(strict=True, ge=0)
    x_min: FiniteFloat
    x_max: FiniteFloat
    native_y: FiniteFloat
    z_min: FiniteFloat
    z_max: FiniteFloat
    patch_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if self.x_min >= self.x_max or self.z_min >= self.z_max:
            raise ValueError("surface evidence patch must have positive area")
        expected = canonical_sha256_v2(
            _patch_payload(
                patch_index=self.patch_index,
                x_min=self.x_min,
                x_max=self.x_max,
                native_y=self.native_y,
                z_min=self.z_min,
                z_max=self.z_max,
            ),
            domain=_PATCH_HASH_DOMAIN,
        )
        if self.patch_sha256 != expected:
            raise ValueError("surface evidence patch digest mismatch")
        return self


def _subject_payload(
    *,
    subject_object_id: str,
    support_object_id: str,
    native_subject_object_id: str,
    native_support_object_id: str,
    runtime_identity_sha256: str,
    scene_sha256: str,
    positions_sha256: str,
    spawn_map_source_sha256: str,
    placement_sha256: str,
    source_capture_sha256: str,
    patches: tuple[CompetitionNativeReceptacleSurfacePatchV2_9_2, ...],
) -> dict[str, object]:
    return {
        "native_support_object_id": native_support_object_id,
        "native_subject_object_id": native_subject_object_id,
        "patches": tuple(item.model_dump(mode="json") for item in patches),
        "placement_sha256": placement_sha256,
        "positions_sha256": positions_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "scene_sha256": scene_sha256,
        "source_capture_sha256": source_capture_sha256,
        "spawn_map_source_sha256": spawn_map_source_sha256,
        "subject_object_id": subject_object_id,
        "support_object_id": support_object_id,
    }


class CompetitionNativeSubjectSurfaceEvidenceV2_9_2(V2Model):
    """All source-capture bindings for one receptacle-supported subject."""

    subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    native_subject_object_id: str = Field(strict=True, min_length=1, max_length=512)
    native_support_object_id: str = Field(strict=True, min_length=1, max_length=512)
    runtime_identity_sha256: Sha256Digest
    scene_sha256: Sha256Digest
    positions_sha256: Sha256Digest
    spawn_map_source_sha256: Sha256Digest
    placement_sha256: Sha256Digest
    source_capture_sha256: Sha256Digest
    patches: tuple[CompetitionNativeReceptacleSurfacePatchV2_9_2, ...] = Field(
        min_length=1,
        max_length=256,
    )
    subject_surface_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_subject_evidence(self) -> Self:
        if self.subject_object_id == self.support_object_id:
            raise ValueError("surface evidence subject and support must differ")
        if tuple(item.patch_index for item in self.patches) != tuple(
            range(len(self.patches))
        ):
            raise ValueError("surface evidence patch indexes are not canonical")
        patch_keys = tuple(
            (
                item.native_y,
                item.x_min,
                item.z_min,
                item.x_max,
                item.z_max,
            )
            for item in self.patches
        )
        if patch_keys != tuple(sorted(set(patch_keys))):
            raise ValueError("surface evidence patches are not unique and canonical")
        expected = canonical_sha256_v2(
            _subject_payload(
                subject_object_id=self.subject_object_id,
                support_object_id=self.support_object_id,
                native_subject_object_id=self.native_subject_object_id,
                native_support_object_id=self.native_support_object_id,
                runtime_identity_sha256=self.runtime_identity_sha256,
                scene_sha256=self.scene_sha256,
                positions_sha256=self.positions_sha256,
                spawn_map_source_sha256=self.spawn_map_source_sha256,
                placement_sha256=self.placement_sha256,
                source_capture_sha256=self.source_capture_sha256,
                patches=self.patches,
            ),
            domain=_SUBJECT_EVIDENCE_HASH_DOMAIN,
        )
        if self.subject_surface_evidence_sha256 != expected:
            raise ValueError("subject surface evidence digest mismatch")
        return self


def _source_payload(
    *,
    source_id: str,
    scene_id: str,
    source_capture_sha256: str,
    subjects: tuple[CompetitionNativeSubjectSurfaceEvidenceV2_9_2, ...],
) -> dict[str, object]:
    return {
        "evidence_version": "competition-native-source-surface-evidence:2.9.2",
        "scene_id": scene_id,
        "source_capture_sha256": source_capture_sha256,
        "source_id": source_id,
        "subjects": tuple(item.model_dump(mode="json") for item in subjects),
    }


class CompetitionNativeSourceSurfaceEvidenceV2_9_2(V2Model):
    """One sibling surface-evidence row for an unchanged accepted capture."""

    evidence_version: Literal["competition-native-source-surface-evidence:2.9.2"] = (
        "competition-native-source-surface-evidence:2.9.2"
    )
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_capture_sha256: Sha256Digest
    subjects: tuple[CompetitionNativeSubjectSurfaceEvidenceV2_9_2, ...] = Field(
        max_length=96
    )
    surface_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_source_evidence(self) -> Self:
        subject_ids = tuple(item.subject_object_id for item in self.subjects)
        if subject_ids != tuple(sorted(set(subject_ids))):
            raise ValueError("source surface evidence subjects are not canonical")
        if any(
            item.source_capture_sha256 != self.source_capture_sha256
            for item in self.subjects
        ):
            raise ValueError("source surface evidence capture lineage mismatch")
        expected = canonical_sha256_v2(
            _source_payload(
                source_id=self.source_id,
                scene_id=self.scene_id,
                source_capture_sha256=self.source_capture_sha256,
                subjects=self.subjects,
            ),
            domain=_SOURCE_EVIDENCE_HASH_DOMAIN,
        )
        if self.surface_evidence_sha256 != expected:
            raise ValueError("source surface evidence digest mismatch")
        return self


def _strict_spawn_map(value: object) -> AI2ThorReceptacleSpawnMap:
    if type(value) is not AI2ThorReceptacleSpawnMap:
        raise TypeError("surface evidence spawn map must be exact")
    if type(value.runtime_identity) is not AI2ThorRuntimeIdentity:
        raise TypeError("surface evidence runtime identity must be exact")
    if type(value.surface_patches) is not tuple or any(
        type(item) is not AI2ThorReceptacleSurfacePatch
        for item in value.surface_patches
    ):
        raise TypeError("surface evidence patches must be an exact tuple")
    return AI2ThorReceptacleSpawnMap(
        scene_id=value.scene_id,
        subject_object_id=value.subject_object_id,
        support_object_id=value.support_object_id,
        native_subject_object_id=value.native_subject_object_id,
        native_support_object_id=value.native_support_object_id,
        runtime_identity=AI2ThorRuntimeIdentity(**asdict(value.runtime_identity)),
        positions=tuple(value.positions),
        positions_sha256=value.positions_sha256,
        scene_sha256=value.scene_sha256,
        source_sha256=value.source_sha256,
        surface_patches=tuple(
            AI2ThorReceptacleSurfacePatch(**asdict(item))
            for item in value.surface_patches
        ),
    )


def _build_patch(
    patch_index: int,
    patch: AI2ThorReceptacleSurfacePatch,
) -> CompetitionNativeReceptacleSurfacePatchV2_9_2:
    payload = _patch_payload(
        patch_index=patch_index,
        x_min=patch.x_min,
        x_max=patch.x_max,
        native_y=patch.native_y,
        z_min=patch.z_min,
        z_max=patch.z_max,
    )
    return CompetitionNativeReceptacleSurfacePatchV2_9_2(
        **payload,
        patch_sha256=canonical_sha256_v2(payload, domain=_PATCH_HASH_DOMAIN),
    )


def verify_competition_native_source_surface_evidence_v2_9_2(
    capture: CompetitionNativeSourceCaptureV2_9,
    evidence: CompetitionNativeSourceSurfaceEvidenceV2_9_2,
) -> CompetitionNativeSourceSurfaceEvidenceV2_9_2:
    """Freshly replay every persisted patch binding against one capture."""

    from spatialcf.generation.capture.models import (
        CompetitionNativePlacementAvailabilityV2_9,
        CompetitionNativeRuntimeIdentityV2_9,
        CompetitionNativeSourceCaptureV2_9,
        CompetitionNativeSupportKindV2_9,
    )

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("surface evidence capture must be exact")
    if type(evidence) is not CompetitionNativeSourceSurfaceEvidenceV2_9_2:
        raise TypeError("surface evidence must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"),
        strict=True,
    )
    checked_evidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2.model_validate(
        evidence.model_dump(mode="python"),
        strict=True,
    )
    if (
        checked_evidence.source_id != checked_capture.source.source_id
        or checked_evidence.scene_id != checked_capture.scene.scene_id
        or checked_evidence.source_capture_sha256
        != checked_capture.source_capture_sha256
    ):
        raise ValueError("surface evidence does not bind source capture")

    placements = {item.object_id: item for item in checked_capture.placement_facts}
    supports = {item.object_id: item for item in checked_capture.support_facts}
    expected_subject_ids = tuple(
        sorted(
            item.object_id
            for item in checked_capture.placement_facts
            if item.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        )
    )
    if tuple(item.subject_object_id for item in checked_evidence.subjects) != (
        expected_subject_ids
    ):
        raise ValueError("surface evidence does not close capture placements")

    runtime = AI2ThorRuntimeIdentity(
        **checked_capture.runtime_identity.model_dump(mode="python")
    )
    runtime_digest = canonical_sha256_v2(
        CompetitionNativeRuntimeIdentityV2_9(**asdict(runtime)),
        domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
    )
    for subject in checked_evidence.subjects:
        placement = placements[subject.subject_object_id]
        support = supports[subject.subject_object_id]
        support_object_id = support.support_object_id
        if (
            support.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or support_object_id is None
            or subject.support_object_id != support_object_id
            or subject.native_subject_object_id != support.native_object_id
            or subject.native_support_object_id
            != supports[support_object_id].native_object_id
            or subject.runtime_identity_sha256 != runtime_digest
            or subject.placement_sha256 != placement.placement_sha256
            or subject.source_capture_sha256 != checked_capture.source_capture_sha256
            or placement.position_region is None
        ):
            raise ValueError("surface evidence subject does not bind capture facts")
        spawn_map = AI2ThorReceptacleSpawnMap(
            scene_id=checked_capture.scene.scene_id,
            subject_object_id=subject.subject_object_id,
            support_object_id=subject.support_object_id,
            native_subject_object_id=subject.native_subject_object_id,
            native_support_object_id=subject.native_support_object_id,
            runtime_identity=runtime,
            positions=tuple(
                AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
                for item in placement.native_positions
            ),
            positions_sha256=subject.positions_sha256,
            scene_sha256=subject.scene_sha256,
            source_sha256=subject.spawn_map_source_sha256,
            surface_patches=tuple(
                AI2ThorReceptacleSurfacePatch(
                    x_min=item.x_min,
                    x_max=item.x_max,
                    native_y=item.native_y,
                    z_min=item.z_min,
                    z_max=item.z_max,
                )
                for item in subject.patches
            ),
        )
        if (
            build_receptacle_support_position_region(
                checked_capture.scene,
                spawn_map,
            )
            != placement.position_region
        ):
            raise ValueError("surface evidence placement region changed on replay")
    return checked_evidence


def build_competition_native_source_surface_evidence_v2_9_2(
    capture: CompetitionNativeSourceCaptureV2_9,
    spawn_maps: tuple[AI2ThorReceptacleSpawnMap, ...],
) -> CompetitionNativeSourceSurfaceEvidenceV2_9_2:
    """Close raw patch ownership against one unchanged source capture."""

    from spatialcf.generation.capture.models import (
        CompetitionNativePlacementAvailabilityV2_9,
        CompetitionNativeRuntimeIdentityV2_9,
        CompetitionNativeSourceCaptureV2_9,
        CompetitionNativeSupportKindV2_9,
    )

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("surface evidence capture must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"),
        strict=True,
    )
    if type(spawn_maps) is not tuple:
        raise TypeError("surface evidence spawn_maps must be an exact tuple")
    checked_maps = tuple(
        sorted(
            (_strict_spawn_map(item) for item in spawn_maps),
            key=lambda item: item.subject_object_id,
        )
    )
    map_subject_ids = tuple(item.subject_object_id for item in checked_maps)
    if len(map_subject_ids) != len(set(map_subject_ids)):
        raise ValueError("surface evidence spawn-map subjects must be unique")

    placement_by_id = {item.object_id: item for item in checked_capture.placement_facts}
    support_by_id = {item.object_id: item for item in checked_capture.support_facts}
    expected_subject_ids = tuple(
        sorted(
            item.object_id
            for item in checked_capture.placement_facts
            if item.availability
            is CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        )
    )
    if map_subject_ids != expected_subject_ids:
        raise ValueError("surface evidence spawn maps do not close capture placements")

    subjects: list[CompetitionNativeSubjectSurfaceEvidenceV2_9_2] = []
    for spawn_map in checked_maps:
        placement = placement_by_id[spawn_map.subject_object_id]
        support = support_by_id[spawn_map.subject_object_id]
        support_object_id = support.support_object_id
        if (
            support.support_kind is not CompetitionNativeSupportKindV2_9.RECEPTACLE
            or support_object_id is None
            or spawn_map.scene_id != checked_capture.scene.scene_id
            or spawn_map.support_object_id != support_object_id
            or spawn_map.native_subject_object_id != support.native_object_id
            or spawn_map.native_support_object_id
            != support_by_id[support_object_id].native_object_id
            or not spawn_map.surface_patches
            or placement.position_region is None
            or build_receptacle_support_position_region(
                checked_capture.scene,
                spawn_map,
            )
            != placement.position_region
        ):
            raise ValueError("surface evidence spawn map does not bind capture facts")
        expected_runtime = CompetitionNativeRuntimeIdentityV2_9(
            **asdict(spawn_map.runtime_identity)
        )
        if expected_runtime != checked_capture.runtime_identity:
            raise ValueError("surface evidence runtime does not bind source capture")
        patches = tuple(
            _build_patch(index, patch)
            for index, patch in enumerate(spawn_map.surface_patches)
        )
        runtime_identity_sha256 = canonical_sha256_v2(
            checked_capture.runtime_identity,
            domain=_RUNTIME_IDENTITY_HASH_DOMAIN,
        )
        payload = _subject_payload(
            subject_object_id=spawn_map.subject_object_id,
            support_object_id=spawn_map.support_object_id,
            native_subject_object_id=spawn_map.native_subject_object_id,
            native_support_object_id=spawn_map.native_support_object_id,
            runtime_identity_sha256=runtime_identity_sha256,
            scene_sha256=spawn_map.scene_sha256,
            positions_sha256=spawn_map.positions_sha256,
            spawn_map_source_sha256=spawn_map.source_sha256,
            placement_sha256=placement.placement_sha256,
            source_capture_sha256=checked_capture.source_capture_sha256,
            patches=patches,
        )
        subjects.append(
            CompetitionNativeSubjectSurfaceEvidenceV2_9_2(
                **payload,
                subject_surface_evidence_sha256=canonical_sha256_v2(
                    payload,
                    domain=_SUBJECT_EVIDENCE_HASH_DOMAIN,
                ),
            )
        )

    subject_tuple = tuple(subjects)
    source_payload = _source_payload(
        source_id=checked_capture.source.source_id,
        scene_id=checked_capture.scene.scene_id,
        source_capture_sha256=checked_capture.source_capture_sha256,
        subjects=subject_tuple,
    )
    evidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2(
        source_id=checked_capture.source.source_id,
        scene_id=checked_capture.scene.scene_id,
        source_capture_sha256=checked_capture.source_capture_sha256,
        subjects=subject_tuple,
        surface_evidence_sha256=canonical_sha256_v2(
            source_payload,
            domain=_SOURCE_EVIDENCE_HASH_DOMAIN,
        ),
    )
    return verify_competition_native_source_surface_evidence_v2_9_2(
        checked_capture,
        evidence,
    )


ReceptacleSurfacePatch = CompetitionNativeReceptacleSurfacePatchV2_9_2
SourceSurfaceEvidence = CompetitionNativeSourceSurfaceEvidenceV2_9_2
SubjectSurfaceEvidence = CompetitionNativeSubjectSurfaceEvidenceV2_9_2
build_source_surface_evidence = build_competition_native_source_surface_evidence_v2_9_2
verify_source_surface_evidence = (
    verify_competition_native_source_surface_evidence_v2_9_2
)

__all__ = (
    "ReceptacleSurfacePatch",
    "SourceSurfaceEvidence",
    "SubjectSurfaceEvidence",
    "build_source_surface_evidence",
    "verify_source_surface_evidence",
)
