"""Flat current capture-plan contract and source locators."""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    NonNegativeFiniteFloat,
    PositiveFiniteFloat,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)
from spatialcf.generation._internal.evidence.camera import (
    build_settled_camera_policy,
)
from spatialcf.generation._internal.source_manifest import (
    LegacySource,
    ProceduralSource,
    SolverConfig,
    SourcePlanEntry,
    SourcePlanManifest,
)
from spatialcf.generation.capture.models import (
    CompetitionNativeSourceRefV2_9,
    DatasetSplitV2_9,
    RosterPolicy,
)
from spatialcf.generation.errors import require_wire_version
from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

_LOCATOR_DOMAIN = "spatialcf.competition-native-dataset-source-locator.v2.9"
_SOURCE_MANIFEST_DOMAIN = "spatialcf.competition-native-source-manifest.v2.9"
_PLAN_DOMAIN_V2_9_3 = "spatialcf.competition-native-dataset-capture-plan.v2.9.3"
_MAX_SOURCES = 32
_MAX_OBJECTS_PER_SCENE = 96
_MAX_CANDIDATES = 40_000
_PLAN_FILES = frozenset({"plan.json", "checksums.sha256"})
_MAX_PLAN_BYTES = 64 * 1024 * 1024
_MAX_CHECKSUM_BYTES = 1024
_CAPTURE_PLAN_VERIFICATION_CAPABILITY = object()


class CompetitionNativeLegacyDatasetLocatorV2_9(V2Model):
    """One complete iTHOR source locator, independent of runtime results."""

    locator_version: Literal["competition-native-dataset-source-locator:2.9"] = (
        "competition-native-dataset-source-locator:2.9"
    )
    kind: Literal["legacy-ai2thor"] = "legacy-ai2thor"
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_name: str = Field(strict=True, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        if self.scene_id != self.scene_name:
            raise ValueError("legacy dataset locator scene identity mismatch")
        return self

    @property
    def locator_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_LOCATOR_DOMAIN)


class CompetitionNativeProceduralDatasetLocatorV2_9(V2Model):
    """One complete ProcTHOR locator with frozen source-content identity."""

    locator_version: Literal["competition-native-dataset-source-locator:2.9"] = (
        "competition-native-dataset-source-locator:2.9"
    )
    kind: Literal["procedural"] = "procedural"
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    dataset_id: Literal["allenai/procthor-10k"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["train", "val", "test"]
    index: int = Field(strict=True, ge=0)
    source_sha256: Sha256Digest
    scene_alias: str = Field(strict=True, min_length=1, max_length=512)
    loader_id: Literal["prior"]
    loader_version: Literal["1.0.3"]

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        if self.scene_id != self.scene_alias:
            raise ValueError("procedural dataset locator scene identity mismatch")
        return self

    @property
    def locator_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_LOCATOR_DOMAIN)


CompetitionNativeDatasetSourceLocatorV2_9 = Annotated[
    CompetitionNativeLegacyDatasetLocatorV2_9
    | CompetitionNativeProceduralDatasetLocatorV2_9,
    Field(discriminator="kind"),
]


class CaptureSettings(V2Model):
    """Read-only runtime settings shared by every source in the prefix."""

    settings_version: Literal["competition-native-dataset-capture-settings:2.9"] = (
        "competition-native-dataset-capture-settings:2.9"
    )
    max_settlement_steps: int = Field(default=60, strict=True, gt=0, le=600)
    floor_clearance_m: NonNegativeFiniteFloat = 0.1
    navigation_agent_radius_m: PositiveFiniteFloat = 0.2
    navigation_clearance_m: NonNegativeFiniteFloat = 0.0

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        if (
            self.floor_clearance_m > 2.0
            or self.navigation_agent_radius_m > 2.0
            or self.navigation_clearance_m > 2.0
        ):
            raise ValueError("dataset capture geometry setting exceeds frozen limit")
        return self


def _source_manifest_sha256(
    manifest: SourcePlanManifest,
) -> Sha256Digest:
    return canonical_sha256_v2(
        manifest.model_dump(mode="json"),
        domain=_SOURCE_MANIFEST_DOMAIN,
    )


def _source_locators(
    manifest: SourcePlanManifest,
) -> tuple[CompetitionNativeDatasetSourceLocatorV2_9, ...]:
    locators: list[CompetitionNativeDatasetSourceLocatorV2_9] = []
    for entry in manifest.sources:
        source = entry.source
        if type(source) is LegacySource:
            locators.append(
                CompetitionNativeLegacyDatasetLocatorV2_9(
                    source_id=entry.source_id,
                    scene_id=entry.scene_id,
                    scene_name=source.scene_name,
                )
            )
        elif type(source) is ProceduralSource:
            locators.append(
                CompetitionNativeProceduralDatasetLocatorV2_9(
                    source_id=entry.source_id,
                    scene_id=entry.scene_id,
                    dataset_id=source.dataset_id,
                    revision=source.revision,
                    split=source.split,
                    index=source.index,
                    source_sha256=source.source_sha256,
                    scene_alias=source.scene_alias,
                    loader_id=source.loader_id,
                    loader_version=source.loader_version,
                )
            )
        else:
            raise TypeError("dataset source manifest contains an unsupported source")
    return tuple(locators)


def _public_source_split(value: str) -> DatasetSplitV2_9:
    if value == "val":
        return "validation"
    if value == "train":
        return "train"
    if value == "test":
        return "test"
    raise ValueError("unsupported ProcTHOR source split")


def _source_refs(
    locators: tuple[CompetitionNativeDatasetSourceLocatorV2_9, ...],
    assigned_split: DatasetSplitV2_9,
) -> tuple[CompetitionNativeSourceRefV2_9, ...]:
    refs = []
    for locator in locators:
        if (
            isinstance(locator, CompetitionNativeProceduralDatasetLocatorV2_9)
            and _public_source_split(locator.split) != assigned_split
        ):
            raise ValueError("procedural source split differs from assigned split")
        refs.append(
            CompetitionNativeSourceRefV2_9(
                source_id=locator.source_id,
                scene_id=locator.scene_id,
                split=assigned_split,
                source_locator_sha256=locator.locator_sha256,
            )
        )
    return tuple(refs)


CompetitionNativeDatasetCaptureSettingsV2_9 = CaptureSettings


class CapturePlan(V2Model):
    """The only supported manifest-to-roster capture plan."""

    plan_version: Literal["competition-native-dataset-capture-plan:2.9.3"] = (
        "competition-native-dataset-capture-plan:2.9.3"
    )
    evidence_eligible: Literal[False] = False
    source_manifest: SourcePlanManifest
    source_manifest_sha256: Sha256Digest
    assigned_split: DatasetSplitV2_9
    source_locators: tuple[CompetitionNativeDatasetSourceLocatorV2_9, ...] = Field(
        min_length=1,
        max_length=_MAX_SOURCES,
    )
    capture_settings: CaptureSettings
    roster_policy: RosterPolicy

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if type(self.source_manifest) is not SourcePlanManifest:
            raise TypeError("dataset capture source manifest must be exact")
        manifest_locators = _source_locators(self.source_manifest)
        try:
            source_start = manifest_locators.index(self.source_locators[0])
        except ValueError as error:
            raise ValueError(
                "dataset capture locators escape the source manifest"
            ) from error
        source_stop = source_start + len(self.source_locators)
        if manifest_locators[source_start:source_stop] != self.source_locators:
            raise ValueError("dataset capture locators are not one contiguous slice")
        expected_refs = _source_refs(self.source_locators, self.assigned_split)
        if (
            self.source_manifest_sha256 != _source_manifest_sha256(self.source_manifest)
            or self.roster_policy.sources != expected_refs
            or self.roster_policy.max_scenes != len(expected_refs)
            or self.roster_policy.width != self.source_manifest.width
            or self.roster_policy.height != self.source_manifest.height
            or self.roster_policy.seed != self.source_manifest.seed
        ):
            raise ValueError("dataset capture plan source binding mismatch")
        if type(self.roster_policy) is not RosterPolicy:
            raise TypeError("target-reachability dataset capture policy must be exact")
        return self

    @property
    def plan_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_PLAN_DOMAIN_V2_9_3)


CompetitionNativeDatasetCapturePlanV2_9_3 = CapturePlan


@dataclass(frozen=True, slots=True)
class RetainedCapturePlanVerification:
    """One capture plan verified against a retained directory snapshot."""

    _capability: object = field(repr=False, compare=False)
    root_identity: DirectoryIdentity
    plan: CapturePlan
    _entries: tuple[tuple[str, os.stat_result], ...] = field(
        repr=False,
        compare=False,
    )


def _parse_capture_plan(payload: bytes) -> CapturePlan:
    plan = CapturePlan.model_validate_json(payload, strict=True)
    if payload != canonical_json_bytes_v2(plan) + b"\n":
        raise ValueError("dataset capture plan is not canonical")
    return plan


def _load_plan_fd(descriptor: int) -> CapturePlan:
    entries = snapshot_exact_directory(descriptor, regular_names=_PLAN_FILES)
    payload = read_regular_at(
        descriptor,
        "plan.json",
        _MAX_PLAN_BYTES,
        expected_stat=entries["plan.json"],
    )
    require_wire_version(
        payload,
        artifact_kind="dataset capture plan",
        field="plan_version",
        expected="competition-native-dataset-capture-plan:2.9.3",
    )
    plan = _parse_capture_plan(payload)
    checksum = read_regular_at(
        descriptor,
        "checksums.sha256",
        _MAX_CHECKSUM_BYTES,
        expected_stat=entries["checksums.sha256"],
    )
    expected = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
    if checksum != expected:
        raise ValueError("dataset capture plan checksum mismatch")
    revalidate_entries(descriptor, entries)
    return plan


def prepare_capture_plan_verification(
    root_descriptor: int,
) -> RetainedCapturePlanVerification:
    """Verify a current capture plan under an already-retained descriptor."""

    if type(root_descriptor) is not int:
        raise TypeError("capture plan descriptor must be an exact integer")
    entries = snapshot_exact_directory(root_descriptor, regular_names=_PLAN_FILES)
    identity = directory_identity_fd(root_descriptor)
    plan = _load_plan_fd(root_descriptor)
    revalidate_entries(root_descriptor, entries)
    return RetainedCapturePlanVerification(
        _capability=_CAPTURE_PLAN_VERIFICATION_CAPABILITY,
        root_identity=identity,
        plan=plan,
        _entries=tuple(sorted(entries.items())),
    )


def revalidate_capture_plan_verification(
    root_descriptor: int,
    retained: RetainedCapturePlanVerification,
) -> CapturePlan:
    """Revalidate a prepared capture plan without establishing a new baseline."""

    if type(root_descriptor) is not int:
        raise TypeError("capture plan descriptor must be an exact integer")
    if (
        type(retained) is not RetainedCapturePlanVerification
        or retained._capability is not _CAPTURE_PLAN_VERIFICATION_CAPABILITY
    ):
        raise TypeError("retained capture plan verification must be exact")
    if directory_identity_fd(root_descriptor) != retained.root_identity:
        raise ValueError("retained capture plan root identity changed")
    revalidate_entries(root_descriptor, dict(retained._entries))
    return CapturePlan.model_validate(
        retained.plan.model_dump(mode="python", warnings="error"),
        strict=True,
    )


def load_capture_plan(root: Path) -> CapturePlan:
    """Fresh-read the exact current two-file capture plan."""

    if not isinstance(root, Path):
        raise TypeError("dataset capture plan root must be a Path")
    with bound_absolute_directory(root) as descriptor:
        return _load_plan_fd(descriptor)


def _rollback_plan_transaction(transaction, output: Path, active_error) -> None:
    try:
        reconciliation = transaction.reconcile()
    except BaseException as reconciliation_error:
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="dataset capture plan failure could not reconcile publication state",
        ) from reconciliation_error
    if reconciliation.location is RenameLocation.UNKNOWN:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise active_error
        raise CompetitionNativePublicationError(
            output,
            published=None,
            recovery_name=transaction.recovery_name,
            detail="dataset capture plan failure left an unknown publication state",
        ) from active_error
    if reconciliation.location is RenameLocation.SOURCE:
        if isinstance(active_error, CompetitionNativePublicationError):
            raise CompetitionNativePublicationError(
                output,
                published=False,
                recovery_name=transaction.recovery_name,
                detail="dataset capture plan publication failed before commit",
            ) from active_error
        return
    try:
        transaction.rollback()
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
            detail="dataset capture plan publication rollback failed",
        ) from rollback_error
    if isinstance(active_error, CompetitionNativePublicationError):
        raise CompetitionNativePublicationError(
            output,
            published=False,
            recovery_name=transaction.recovery_name,
            detail="dataset capture plan publication failed and was rolled back",
        ) from active_error


def _raise_plan_transaction_exit_error(
    parent,
    transaction,
    output: Path,
    active_error,
) -> None:
    try:
        reconciliation = reconcile_owned_rename_at(
            parent.parent_descriptor,
            transaction.name,
            parent.output_name,
            transaction.identity,
        )
    except BaseException as reconciliation_error:  # noqa: BLE001
        active_error.add_note(str(reconciliation_error))
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
        detail="dataset capture plan transaction close failed after publication",
    ) from active_error


def publish_capture_plan(plan: CapturePlan, output_root: Path) -> CapturePlan:
    """Atomically publish one absent, fresh-verified current capture plan."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        if type(plan) is not CapturePlan:
            raise TypeError("dataset capture plan must be exact")
        if not isinstance(output_root, Path):
            raise TypeError("dataset capture plan output_root must be a Path")
        checked = CapturePlan.model_validate(
            plan.model_dump(mode="python", warnings="error"),
            strict=True,
        )
        payload = canonical_json_bytes_v2(checked) + b"\n"
        if len(payload) > _MAX_PLAN_BYTES:
            raise ValueError("dataset capture plan exceeds byte limit")
        checksum = f"{hashlib.sha256(payload).hexdigest()}  plan.json\n".encode("ascii")
        output = Path(os.path.abspath(output_root))
        parent = open_native_output_parent(output)
        transaction = None
        publication_completed = False
        try:
            parent.ensure_absent(parent.output_name)
            try:
                with parent.create_staging(label="dataset-plan") as transaction:
                    try:
                        transaction.write("plan.json", payload)
                        transaction.write("checksums.sha256", checksum)
                        transaction.fsync()
                        seal = transaction.seal()
                        if _load_plan_fd(transaction.descriptor) != checked:
                            raise RuntimeError(
                                "sealed dataset capture plan verification changed"
                            )
                        transaction.validate_seal(seal)
                        transaction.publish()
                        transaction.validate_location(RenameLocation.OUTPUT)
                        transaction.validate_seal(seal)
                        parent.validate()
                        publication_completed = True
                    except BaseException as error:
                        _rollback_plan_transaction(transaction, output, error)
                        raise
            except CompetitionNativePublicationError:
                raise
            except BaseException as error:
                if publication_completed and transaction is not None:
                    _raise_plan_transaction_exit_error(
                        parent, transaction, output, error
                    )
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
            published: bool | None = True
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
                detail="dataset capture plan published but parent close failed",
            ) from close_error
        return checked


def build_capture_plan(
    source_manifest: SourcePlanManifest,
    *,
    assigned_split: DatasetSplitV2_9,
    campaign_id: str,
    max_objects_per_scene: int = _MAX_OBJECTS_PER_SCENE,
    max_candidates_total: int = _MAX_CANDIDATES,
    max_requests_total: int = 60,
    max_requests_per_scene: int = 60,
    max_requests_per_subject: int = 18,
    max_requests_per_category_pair: int = 12,
    target_requests_per_relation: int = 10,
    capture_settings: CaptureSettings | None = None,
    source_start: int = 0,
    source_stop: int | None = None,
) -> CapturePlan:
    """Build the current plan directly, without an earlier-version plan."""

    if type(source_manifest) is not SourcePlanManifest:
        raise TypeError("dataset capture source manifest must be exact")
    checked_manifest = SourcePlanManifest.model_validate(
        source_manifest.model_dump(mode="python"),
        strict=True,
    )
    if type(source_start) is not int or (
        source_stop is not None and type(source_stop) is not int
    ):
        raise TypeError("dataset source slice bounds must be exact integers")
    resolved_stop = (
        len(checked_manifest.sources) if source_stop is None else source_stop
    )
    if not 0 <= source_start < resolved_stop <= len(checked_manifest.sources):
        raise ValueError("dataset source slice must be non-empty and in bounds")
    if resolved_stop - source_start > _MAX_SOURCES:
        raise ValueError("dataset source slice exceeds the bounded shard size")
    locators = _source_locators(checked_manifest)[source_start:resolved_stop]
    refs = _source_refs(locators, assigned_split)
    settings = (
        CaptureSettings()
        if capture_settings is None
        else CaptureSettings.model_validate(
            capture_settings.model_dump(mode="python", warnings="error"),
            strict=True,
        )
    )
    policy = RosterPolicy(
        campaign_id=campaign_id,
        seed=checked_manifest.seed,
        width=checked_manifest.width,
        height=checked_manifest.height,
        max_scenes=len(refs),
        max_objects_per_scene=max_objects_per_scene,
        max_candidates_total=max_candidates_total,
        max_requests_total=max_requests_total,
        max_requests_per_scene=max_requests_per_scene,
        max_requests_per_subject=max_requests_per_subject,
        max_requests_per_category_pair=max_requests_per_category_pair,
        target_requests_per_relation=target_requests_per_relation,
        sources=refs,
        camera_policy=build_settled_camera_policy(),
    )
    return CapturePlan(
        source_manifest=checked_manifest,
        source_manifest_sha256=_source_manifest_sha256(checked_manifest),
        assigned_split=assigned_split,
        source_locators=locators,
        capture_settings=settings,
        roster_policy=policy,
    )


def build_legacy_capture_plan(
    scene_names: tuple[str, ...],
    *,
    assigned_split: DatasetSplitV2_9,
    campaign_id: str,
    seed: int,
    width: int,
    height: int,
    max_requests_total: int,
) -> CapturePlan:
    """Build the current capture plan for canonical legacy AI2-THOR scenes."""

    if type(scene_names) is not tuple or any(
        type(scene_name) is not str or not scene_name for scene_name in scene_names
    ):
        raise TypeError("legacy capture scene_names must be an exact string tuple")
    if not scene_names or scene_names != tuple(sorted(set(scene_names))):
        raise ValueError("legacy capture scene_names must be canonical and non-empty")
    manifest = SourcePlanManifest(
        schema_version="certified-ai2thor-source-plan-manifest-v1",
        plan_version="spatialcf-public-dataset-v1",
        batch_version="spatialcf-public-dataset-v1",
        width=width,
        height=height,
        seed=seed,
        camera_policy="all-observed-source-cameras-v1",
        use_navigation_feasibility=True,
        solver_config=SolverConfig(),
        sources=tuple(
            SourcePlanEntry(
                source_id=f"source-{index:04d}",
                scene_id=scene_name,
                source=LegacySource(kind="legacy-ai2thor", scene_name=scene_name),
            )
            for index, scene_name in enumerate(scene_names)
        ),
    )
    return build_capture_plan(
        manifest,
        assigned_split=assigned_split,
        campaign_id=campaign_id,
        max_requests_total=max_requests_total,
    )


__all__ = (
    "CapturePlan",
    "CaptureSettings",
    "CompetitionNativeDatasetCapturePlanV2_9_3",
    "CompetitionNativeDatasetCaptureSettingsV2_9",
    "CompetitionNativeDatasetSourceLocatorV2_9",
    "CompetitionNativeLegacyDatasetLocatorV2_9",
    "CompetitionNativeProceduralDatasetLocatorV2_9",
    "RetainedCapturePlanVerification",
    "build_capture_plan",
    "build_legacy_capture_plan",
    "load_capture_plan",
    "prepare_capture_plan_verification",
    "publish_capture_plan",
    "revalidate_capture_plan_verification",
)
