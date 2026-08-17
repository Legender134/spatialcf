"""Current eight-role native asset publication and fresh verification."""

from __future__ import annotations

import hashlib
import os
import warnings
from enum import StrEnum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import Field, model_validator

from spatialcf.adapters.ai2thor import AI2ThorObservation
from spatialcf.adapters.ai2thor_validation import observation_contract_errors
from spatialcf.domain.models import Scene
from spatialcf.domain.v2.base import CanonicalId, Sha256Digest, V2Model
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation._internal.execution.audit import observation_sha256
from spatialcf.generation._internal.execution.correspondence import legacy_sha256
from spatialcf.generation._internal.execution.run import (
    AuditExecution,
    AuditRun,
    verify_audit_run,
)
from spatialcf.generation.errors import require_exact_type, require_wire_version
from spatialcf.verification.filesystem import (
    BindingStatus,
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

_BUNDLE_VERSION = "competition-native-asset-bundle:2.9.7"
_BUNDLE_HASH_DOMAIN = "spatialcf.competition-native-asset-bundle.v2.9.7"
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_ASSET_BYTES = 512 * 1024 * 1024
_MAX_AGGREGATE_ASSET_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_DIMENSION_PX = 4096
_MAX_IMAGE_PIXELS = 4_194_304


class AssetPhase(StrEnum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"


class AssetKind(StrEnum):
    RGB_PNG = "RGB_PNG"
    DEPTH_NPY = "DEPTH_NPY"
    INSTANCE_PNG = "INSTANCE_PNG"
    POINTCLOUD_PLY = "POINTCLOUD_PLY"


_ASSET_FILENAMES = {
    (AssetPhase.BEFORE, AssetKind.RGB_PNG): "before-rgb.png",
    (AssetPhase.BEFORE, AssetKind.DEPTH_NPY): "before-depth.npy",
    (AssetPhase.BEFORE, AssetKind.INSTANCE_PNG): "before-instance.png",
    (AssetPhase.BEFORE, AssetKind.POINTCLOUD_PLY): "before-pointcloud.ply",
    (AssetPhase.AFTER, AssetKind.RGB_PNG): "after-rgb.png",
    (AssetPhase.AFTER, AssetKind.DEPTH_NPY): "after-depth.npy",
    (AssetPhase.AFTER, AssetKind.INSTANCE_PNG): "after-instance.png",
    (AssetPhase.AFTER, AssetKind.POINTCLOUD_PLY): "after-pointcloud.ply",
}
_ASSET_ORDER = {key: index for index, key in enumerate(_ASSET_FILENAMES)}
_EXPECTED_SUFFIXES = {
    AssetKind.RGB_PNG: ".png",
    AssetKind.DEPTH_NPY: ".npy",
    AssetKind.INSTANCE_PNG: ".png",
    AssetKind.POINTCLOUD_PLY: ".ply",
}


class ReturnedAssetRef(V2Model):
    phase: AssetPhase
    kind: AssetKind
    relative_path: str = Field(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    byte_length: int = Field(strict=True, gt=0)
    content_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.name != self.relative_path
            or path.suffix.lower() != _EXPECTED_SUFFIXES[self.kind]
        ):
            raise ValueError("returned asset must use one local kind-matched filename")
        return self


class InstancePixelCount(V2Model):
    object_id: CanonicalId
    pixel_count: int = Field(strict=True, ge=0)


class AssetBundle(V2Model):
    """The single current asset bundle wire."""

    bundle_version: Literal["competition-native-asset-bundle:2.9.7"] = (
        "competition-native-asset-bundle:2.9.7"
    )
    verification_scope: Literal[
        "FRESH_SELECTED_CAMERA_ASSETS_BBOX_PATCH_SOLVER_NATIVE_COLLISION_"
        "AND_BOUNDED_SUBJECT_POSE_AUTHORITY"
    ] = (
        "FRESH_SELECTED_CAMERA_ASSETS_BBOX_PATCH_SOLVER_NATIVE_COLLISION_"
        "AND_BOUNDED_SUBJECT_POSE_AUTHORITY"
    )
    evidence_eligible: Literal[False] = False
    native_audit_run: AuditRun
    native_audit_run_sha256: Sha256Digest
    observed_after_scene: Scene
    observed_after_scene_sha256: Sha256Digest
    before_instance_pixel_counts: tuple[InstancePixelCount, ...]
    after_instance_pixel_counts: tuple[InstancePixelCount, ...]
    assets: tuple[ReturnedAssetRef, ...]

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        run = AuditRun.model_validate(self.native_audit_run, strict=True)
        if self.native_audit_run_sha256 != run.competition_native_audit_run_sha256:
            raise ValueError("native asset run hash is not closed")
        if self.observed_after_scene_sha256 != legacy_sha256(self.observed_after_scene):
            raise ValueError("native asset after scene hash is not closed")
        if self.observed_after_scene_sha256 != run.native_audit.observed_scene_sha256:
            raise ValueError("native asset after scene does not belong to the audit")
        before = _canonical_counts(self.before_instance_pixel_counts)
        after = _canonical_counts(self.after_instance_pixel_counts)
        expected_ids = tuple(
            sorted(item.object_id for item in run.source_scene.objects)
        )
        if tuple(item.object_id for item in before) != expected_ids:
            raise ValueError("native before pixel-count roster is not closed")
        if tuple(item.object_id for item in after) != expected_ids:
            raise ValueError("native after pixel-count roster is not closed")
        object.__setattr__(self, "before_instance_pixel_counts", before)
        object.__setattr__(self, "after_instance_pixel_counts", after)
        assets = tuple(
            ReturnedAssetRef.model_validate(item, strict=True) for item in self.assets
        )
        keys = tuple((item.phase, item.kind) for item in assets)
        if len(assets) != len(_ASSET_FILENAMES) or set(keys) != set(_ASSET_FILENAMES):
            raise ValueError("native asset bundle requires all eight roles")
        if any(
            item.relative_path != _ASSET_FILENAMES[(item.phase, item.kind)]
            for item in assets
        ):
            raise ValueError("native asset filenames are frozen by role")
        object.__setattr__(
            self,
            "assets",
            tuple(
                sorted(assets, key=lambda item: _ASSET_ORDER[(item.phase, item.kind)])
            ),
        )
        return self

    @property
    def competition_native_asset_bundle_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_BUNDLE_HASH_DOMAIN)

    @property
    def asset_bundle_sha256(self) -> Sha256Digest:
        return self.competition_native_asset_bundle_sha256


def asset_bundle_sha256_from_checked(bundle: AssetBundle) -> Sha256Digest:
    if type(bundle) is not AssetBundle:
        raise TypeError("checked native asset bundle must be exact")
    return canonical_sha256_v2(
        bundle.model_dump(mode="json", warnings="error"),
        domain=_BUNDLE_HASH_DOMAIN,
    )


def publish_asset_bundle(
    execution: AuditExecution,
    output_root: Path,
    *,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> AssetBundle:
    """Atomically publish the eight bytes payloads retained by one audit."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked = require_exact_type(execution, AuditExecution, label="audit execution")
        if not isinstance(output_root, Path):
            raise TypeError("output_root must be a Path")
        output = Path(os.path.abspath(output_root))
        payloads = _observation_payloads(
            checked.before_observation, checked.after_observation
        )
        _require_payload_limits(payloads)
        _require_observation(checked.before_observation, "before")
        _require_observation(checked.fresh_before_observation, "fresh before")
        _require_observation(checked.after_observation, "after")
        bundle = AssetBundle(
            native_audit_run=checked.run,
            native_audit_run_sha256=(checked.run.competition_native_audit_run_sha256),
            observed_after_scene=checked.after_observation.scene,
            observed_after_scene_sha256=legacy_sha256(checked.after_observation.scene),
            before_instance_pixel_counts=_pixel_counts(checked.before_observation),
            after_instance_pixel_counts=_pixel_counts(checked.after_observation),
            assets=tuple(
                ReturnedAssetRef(
                    phase=phase,
                    kind=kind,
                    relative_path=_ASSET_FILENAMES[(phase, kind)],
                    byte_length=len(payload),
                    content_sha256=hashlib.sha256(payload).hexdigest(),
                )
                for (phase, kind), payload in payloads.items()
            ),
        )
        bundle_payload = canonical_json_bytes_v2(bundle) + b"\n"
        if len(bundle_payload) > _MAX_METADATA_BYTES:
            raise ValueError("native asset bundle metadata exceeds byte limit")
        checksum_payload = _checksum_payload(bundle, bundle_payload)
        if len(checksum_payload) > _MAX_METADATA_BYTES:
            raise ValueError("native asset checksum metadata exceeds byte limit")
        return _publish_bundle_transaction(
            output,
            bundle,
            payloads,
            bundle_payload,
            checksum_payload,
            checked.run,
            expected_parent_identity=expected_parent_identity,
        )


def _publish_bundle_transaction(
    output: Path,
    bundle: AssetBundle,
    payloads: dict[tuple[AssetPhase, AssetKind], bytes],
    bundle_payload: bytes,
    checksum_payload: bytes,
    expected_run: AuditRun,
    *,
    expected_parent_identity: DirectoryIdentity | None,
) -> AssetBundle:
    parent = open_native_output_parent(
        output, expected_parent_identity=expected_parent_identity
    )
    transaction = None
    publication_completed = False
    completed_bundle: AssetBundle | None = None
    try:
        parent.ensure_absent(parent.output_name)
        try:
            with parent.create_staging() as transaction:
                try:
                    for key, payload in payloads.items():
                        transaction.write(_ASSET_FILENAMES[key], payload)
                    transaction.write("bundle.json", bundle_payload)
                    transaction.write("checksums.sha256", checksum_payload)
                    transaction.fsync()
                    seal = transaction.seal()
                    if (
                        _verify_bundle_tree_fd(transaction.descriptor, expected_run)
                        != bundle
                    ):
                        raise RuntimeError("native asset sealed verification changed")
                    transaction.validate_seal(seal)
                    transaction.publish()
                    transaction.validate_location(RenameLocation.OUTPUT)
                    transaction.validate_seal(seal)
                    with bound_child_directory(
                        parent.parent_descriptor, parent.output_name
                    ) as final_descriptor:
                        if (
                            directory_identity_fd(final_descriptor)
                            != transaction.identity
                        ):
                            raise RuntimeError(
                                "native asset final output identity changed"
                            )
                        verified = _verify_bundle_tree_fd(
                            final_descriptor, expected_run
                        )
                        if canonical_json_bytes_v2(verified) != canonical_json_bytes_v2(
                            bundle
                        ):
                            raise RuntimeError(
                                "native asset final verification changed"
                            )
                        completed_bundle = verified
                    transaction.validate_location(RenameLocation.OUTPUT)
                    transaction.validate_seal(seal)
                    parent.validate()
                    publication_completed = True
                except BaseException as error:
                    _rollback_asset_transaction(transaction, output, error)
                    raise
        except CompetitionNativePublicationError:
            raise
        except BaseException as error:
            if publication_completed and transaction is not None:
                _raise_asset_transaction_exit_error(parent, transaction, output, error)
            raise
    except BaseException as error:
        try:
            parent.close()
        except BaseException as close_error:  # noqa: BLE001
            error.add_note(str(close_error))
        raise
    try:
        parent.close()
    except BaseException as close_error:
        published = True
        recovery_name: str | None = output.name
        if transaction is not None:
            try:
                reconciliation = reconcile_owned_rename_at(
                    parent.parent_descriptor,
                    transaction.name,
                    parent.output_name,
                    transaction.identity,
                )
            except BaseException as reconciliation_error:  # noqa: BLE001
                close_error.add_note(str(reconciliation_error))
            else:
                published = (
                    True
                    if reconciliation.location is RenameLocation.OUTPUT
                    else False
                    if reconciliation.location is RenameLocation.SOURCE
                    else None
                )
                recovery_name = (
                    parent.output_name
                    if reconciliation.location is RenameLocation.OUTPUT
                    else transaction.name
                    if reconciliation.location is RenameLocation.SOURCE
                    else transaction.recovery_name
                )
        raise CompetitionNativePublicationError(
            output,
            published=published,
            recovery_name=recovery_name,
            detail="competition native asset published but retained parent close failed",
        ) from close_error
    if completed_bundle is None:
        raise RuntimeError("native asset publication returned no verified bundle")
    return completed_bundle


def _rollback_asset_transaction(
    transaction, output: Path, active_error: object
) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native asset failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="native asset failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if reconciliation.output is BindingStatus.FOREIGN:
            try:
                transaction.cleanup()
            except BaseException as cleanup_error:
                raise CompetitionNativePublicationError(
                    output,
                    published=None,
                    recovery_name=transaction.recovery_name,
                    detail=(
                        "native asset final binding was replaced and owned cleanup "
                        "failed"
                    ),
                ) from cleanup_error
            raise CompetitionNativePublicationError(
                output,
                published=None,
                recovery_name=None,
                detail="native asset final binding was replaced by a foreign entry",
            ) from active_error
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="native asset publication failed before commit",
            ) from active_error
        return
    try:
        transaction.rollback()
    except CompetitionNativePublicationError as rollback_error:
        try:
            final = transaction.reconcile()
        except BaseException:  # noqa: BLE001
            published = rollback_error.published
        else:
            published = (
                True
                if final.location is RenameLocation.OUTPUT
                else False
                if final.location is RenameLocation.SOURCE
                else None
            )
        raise CompetitionNativePublicationError(
            output,
            published=published,
            recovery_name=transaction.recovery_name,
            detail="competition native asset rollback did not fully clean recovery",
        ) from rollback_error
    except BaseException as rollback_error:
        try:
            final = transaction.reconcile()
        except BaseException:  # noqa: BLE001
            published = None
        else:
            published = (
                True
                if final.location is RenameLocation.OUTPUT
                else False
                if final.location is RenameLocation.SOURCE
                else None
            )
        raise CompetitionNativePublicationError(
            output,
            published=published,
            recovery_name=transaction.recovery_name,
            detail="competition native asset rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="native asset publication failed and was rolled back",
        ) from active_error


def _raise_asset_transaction_exit_error(parent, transaction, output, error) -> None:
    try:
        reconciliation = reconcile_owned_rename_at(
            parent.parent_descriptor,
            transaction.name,
            parent.output_name,
            transaction.identity,
        )
    except BaseException as reconciliation_error:  # noqa: BLE001
        error.add_note(str(reconciliation_error))
        published = None
        recovery_name = transaction.recovery_name
    else:
        published = (
            True
            if reconciliation.location is RenameLocation.OUTPUT
            else False
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
        recovery_name = (
            parent.output_name
            if reconciliation.location is RenameLocation.OUTPUT
            else transaction.name
            if reconciliation.location is RenameLocation.SOURCE
            else transaction.recovery_name
        )
    raise CompetitionNativePublicationError(
        output,
        published=published,
        recovery_name=recovery_name,
        detail="native asset transaction close failed after publication",
    ) from error


def load_asset_bundle(root: Path) -> AssetBundle:
    if not isinstance(root, Path):
        raise TypeError("asset bundle root must be a Path")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        with bound_absolute_directory(root) as descriptor:
            bundle, _, entries = _read_bundle_metadata_fd(descriptor)
            revalidate_entries(descriptor, entries)
            return bundle


def verify_asset_bundle(root: Path, expected_run: AuditRun) -> AssetBundle:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    run = require_exact_type(expected_run, AuditRun, label="audit run")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        with bound_absolute_directory(root) as descriptor:
            return _verify_bundle_tree_fd(descriptor, run)


def verify_asset_bundle_fd(descriptor: int, expected_run: AuditRun) -> AssetBundle:
    run = require_exact_type(expected_run, AuditRun, label="audit run")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(descriptor) is not int:
            raise TypeError("descriptor must be an exact integer")
        if descriptor < 0:
            raise ValueError("descriptor must be non-negative")
        return _verify_bundle_tree_fd(descriptor, run)


def _verify_bundle_tree_fd(descriptor: int, expected_run: AuditRun) -> AssetBundle:
    bundle, _ = _verify_bundle_tree_fd_with_digests(descriptor, expected_run)
    return bundle


def _verify_bundle_tree_fd_with_digests(
    descriptor: int,
    expected_run: AuditRun,
) -> tuple[AssetBundle, dict[str, str]]:
    bundle, bundle_payload, entries = _read_bundle_metadata_fd(descriptor)
    if canonical_json_bytes_v2(
        bundle.native_audit_run.model_dump(mode="json", warnings="error")
    ) != canonical_json_bytes_v2(
        expected_run.model_dump(mode="json", warnings="error")
    ):
        raise ValueError("native asset bundle targets a different audit run")
    return _verify_bundle_payloads_fd(
        descriptor, bundle, bundle_payload, entries, expected_run
    )


def _read_bundle_metadata_fd(
    descriptor: int,
) -> tuple[AssetBundle, bytes, dict[str, os.stat_result]]:
    expected_names = {"bundle.json", "checksums.sha256", *_ASSET_FILENAMES.values()}
    entries = snapshot_exact_directory(descriptor, regular_names=expected_names)
    bundle_payload = read_regular_at(
        descriptor,
        "bundle.json",
        _MAX_METADATA_BYTES,
        expected_stat=entries["bundle.json"],
    )
    if not bundle_payload.endswith(b"\n"):
        raise ValueError("native asset bundle metadata requires canonical LF")
    require_wire_version(
        bundle_payload,
        artifact_kind="asset bundle",
        field="bundle_version",
        expected=_BUNDLE_VERSION,
    )
    bundle = AssetBundle.model_validate_json(bundle_payload, strict=True)
    if bundle_payload != canonical_json_bytes_v2(bundle) + b"\n":
        raise ValueError("native asset bundle metadata is not canonical")
    return bundle, bundle_payload, entries


def _verify_bundle_payloads_fd(
    descriptor: int,
    bundle: AssetBundle,
    bundle_payload: bytes,
    entries: dict[str, os.stat_result],
    expected_run: AuditRun,
    *,
    expected_digests: dict[str, str] | None = None,
) -> tuple[AssetBundle, dict[str, str]]:
    by_key = {(item.phase, item.kind): item for item in bundle.assets}
    payloads: dict[tuple[AssetPhase, AssetKind], bytes] = {}
    aggregate = 0
    for key, filename in _ASSET_FILENAMES.items():
        reference = by_key[key]
        aggregate += reference.byte_length
        if aggregate > _MAX_AGGREGATE_ASSET_BYTES:
            raise ValueError("native asset aggregate byte limit exceeded")
        payload = read_regular_at(
            descriptor,
            filename,
            min(_MAX_ASSET_BYTES, reference.byte_length),
            expected_stat=entries[filename],
        )
        if (len(payload), hashlib.sha256(payload).hexdigest()) != (
            reference.byte_length,
            reference.content_sha256,
        ):
            raise ValueError(f"native asset content mismatch: {filename}")
        payloads[key] = payload
    checksum_payload = read_regular_at(
        descriptor,
        "checksums.sha256",
        _MAX_METADATA_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    if checksum_payload != _checksum_payload(bundle, bundle_payload):
        raise ValueError("native asset checksum ledger mismatch")
    digests = {
        "bundle.json": hashlib.sha256(bundle_payload).hexdigest(),
        "checksums.sha256": hashlib.sha256(checksum_payload).hexdigest(),
        **{
            _ASSET_FILENAMES[key]: hashlib.sha256(payload).hexdigest()
            for key, payload in payloads.items()
        },
    }
    if expected_digests is not None and digests != expected_digests:
        raise ValueError("native asset outer checksum ledger mismatch")
    verify_audit_run(expected_run, bundle.observed_after_scene)
    before = _observation_from_bundle(bundle, AssetPhase.BEFORE, payloads)
    fresh_before = _observation_from_bundle(
        bundle,
        AssetPhase.BEFORE,
        payloads,
        scene=expected_run.fresh_source_scene,
    )
    after = _observation_from_bundle(bundle, AssetPhase.AFTER, payloads)
    _require_observation(before, "before")
    _require_observation(fresh_before, "fresh before")
    _require_observation(after, "after")
    if observation_sha256(before) != expected_run.before_observation_sha256:
        raise ValueError("native before observation digest mismatch")
    if observation_sha256(fresh_before) != expected_run.fresh_before_observation_sha256:
        raise ValueError("native fresh before observation digest mismatch")
    if observation_sha256(after) != expected_run.native_audit.after_observation_sha256:
        raise ValueError("native after observation digest mismatch")
    revalidate_entries(descriptor, entries)
    return bundle, digests


def _observation_from_bundle(
    bundle: AssetBundle,
    phase: AssetPhase,
    payloads: dict[tuple[AssetPhase, AssetKind], bytes],
    *,
    scene: Scene | None = None,
) -> AI2ThorObservation:
    selected_scene = (
        scene
        if scene is not None
        else bundle.native_audit_run.source_scene
        if phase is AssetPhase.BEFORE
        else bundle.observed_after_scene
    )
    counts = (
        bundle.before_instance_pixel_counts
        if phase is AssetPhase.BEFORE
        else bundle.after_instance_pixel_counts
    )
    return AI2ThorObservation.create(
        scene=selected_scene,
        rgb_png=payloads[(phase, AssetKind.RGB_PNG)],
        depth_npy=payloads[(phase, AssetKind.DEPTH_NPY)],
        instance_png=payloads[(phase, AssetKind.INSTANCE_PNG)],
        pointcloud_ply=payloads[(phase, AssetKind.POINTCLOUD_PLY)],
        instance_pixel_counts={item.object_id: item.pixel_count for item in counts},
        is_scene_at_rest=True,
    )


def _observation_payloads(
    before: AI2ThorObservation, after: AI2ThorObservation
) -> dict[tuple[AssetPhase, AssetKind], bytes]:
    return {
        (AssetPhase.BEFORE, AssetKind.RGB_PNG): before.rgb_png,
        (AssetPhase.BEFORE, AssetKind.DEPTH_NPY): before.depth_npy,
        (AssetPhase.BEFORE, AssetKind.INSTANCE_PNG): before.instance_png,
        (AssetPhase.BEFORE, AssetKind.POINTCLOUD_PLY): before.pointcloud_ply,
        (AssetPhase.AFTER, AssetKind.RGB_PNG): after.rgb_png,
        (AssetPhase.AFTER, AssetKind.DEPTH_NPY): after.depth_npy,
        (AssetPhase.AFTER, AssetKind.INSTANCE_PNG): after.instance_png,
        (AssetPhase.AFTER, AssetKind.POINTCLOUD_PLY): after.pointcloud_ply,
    }


def _pixel_counts(observation: AI2ThorObservation) -> tuple[InstancePixelCount, ...]:
    return tuple(
        InstancePixelCount(object_id=object_id, pixel_count=count)
        for object_id, count in sorted(observation.instance_pixel_counts.items())
    )


def _canonical_counts(values) -> tuple[InstancePixelCount, ...]:
    checked = tuple(
        InstancePixelCount.model_validate(item, strict=True) for item in values
    )
    ordered = tuple(sorted(checked, key=lambda item: item.object_id))
    if len({item.object_id for item in ordered}) != len(ordered):
        raise ValueError("native pixel-count object IDs must be unique")
    return ordered


def _require_observation(observation: AI2ThorObservation, label: str) -> None:
    if type(observation) is not AI2ThorObservation:
        raise TypeError(f"{label} observation must be exact")
    try:
        camera = observation.scene.camera_by_id("main")
    except KeyError as error:
        raise ValueError(
            f"invalid {label} native observation: missing camera"
        ) from error
    _require_image_shape(camera.width, camera.height)
    _require_png_header(
        observation.rgb_png,
        width=camera.width,
        height=camera.height,
        label="rgb",
    )
    _require_png_header(
        observation.instance_png,
        width=camera.width,
        height=camera.height,
        label="instance",
    )
    _require_depth_header(
        observation.depth_npy, width=camera.width, height=camera.height
    )
    errors = observation_contract_errors(observation, "main")
    if errors:
        raise ValueError(f"invalid {label} native observation: {';'.join(errors)}")
    if observation.is_scene_at_rest is not True:
        raise ValueError(f"{label} native observation is not at rest")


def _require_payload_limits(
    payloads: dict[tuple[AssetPhase, AssetKind], bytes],
) -> None:
    aggregate = 0
    for payload in payloads.values():
        if type(payload) is not bytes or not payload:
            raise ValueError("native asset payloads must be non-empty exact bytes")
        if len(payload) > _MAX_ASSET_BYTES:
            raise ValueError("native asset byte limit exceeded")
        aggregate += len(payload)
        if aggregate > _MAX_AGGREGATE_ASSET_BYTES:
            raise ValueError("native asset aggregate byte limit exceeded")


def _require_image_shape(width: int, height: int) -> None:
    if (
        type(width) is not int
        or type(height) is not int
        or width < 1
        or height < 1
        or width > _MAX_IMAGE_DIMENSION_PX
        or height > _MAX_IMAGE_DIMENSION_PX
        or width * height > _MAX_IMAGE_PIXELS
    ):
        raise ValueError("native observation image dimensions exceed the frozen limit")


def _require_depth_header(payload: bytes, *, width: int, height: int) -> None:
    stream = BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY header version")
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("native observation depth header is invalid") from error
    if (
        shape != (height, width)
        or fortran_order is not False
        or dtype != np.dtype(np.float32)
    ):
        raise ValueError("native observation depth header does not match the camera")


def _require_png_header(
    payload: bytes,
    *,
    width: int,
    height: int,
    label: str,
) -> None:
    try:
        with Image.open(BytesIO(payload)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (width, height)
                or image.width * image.height > _MAX_IMAGE_PIXELS
            ):
                raise ValueError
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(
            f"native {label} PNG header does not match the camera"
        ) from error


def _checksum_payload(bundle: AssetBundle, bundle_payload: bytes) -> bytes:
    entries = {"bundle.json": hashlib.sha256(bundle_payload).hexdigest()}
    entries.update({item.relative_path: item.content_sha256 for item in bundle.assets})
    return "".join(
        f"{digest}  {name}\n" for name, digest in sorted(entries.items())
    ).encode("ascii")


__all__ = (
    "AssetBundle",
    "AssetKind",
    "AssetPhase",
    "InstancePixelCount",
    "ReturnedAssetRef",
    "load_asset_bundle",
    "publish_asset_bundle",
    "verify_asset_bundle",
    "verify_asset_bundle_fd",
)
