"""Publish descriptor-bound dataset indexes and reconcile rollback."""

from __future__ import annotations

import hashlib

import os

import stat

from collections.abc import (
    Iterator,
    Mapping,
)

from contextlib import (
    ExitStack,
    contextmanager,
)

from pathlib import (
    Path,
)

import spatialcf.generation.publication as publication

from spatialcf.generation.dataset_models import (
    DatasetManifest,
    GenerationReport,
    _canonical_model_bytes,
)

from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_directory,
    open_native_output_parent,
    read_regular_at,
    revalidate_entries,
    snapshot_exact_directory,
    sync_directory_fd,
    write_regular_sync_at,
)

from spatialcf.generation.workflows._dataset.artifacts import (
    _verify_dataset_fd,
    _verify_public_index_fd,
)

from spatialcf.generation.workflows._dataset.contracts import (
    _BUNDLE_FILES,
    _MAX_ASSET_BYTES,
)

from spatialcf.generation.workflows._dataset.state import (
    _existing_names,
)


def _checksum_payload(payloads: Mapping[str, bytes]) -> bytes:
    return b"".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode("ascii")
        for name, payload in sorted(payloads.items())
    )


def _stage_public_index(
    transaction,
    records_payload: bytes,
    report_payload: bytes,
    manifest: DatasetManifest,
    bundles: Mapping[str, tuple[publication.AssetBundle, Path]],
) -> dict[str, bytes]:
    transaction.mkdir("assets")
    payloads: dict[str, bytes] = {}
    for bundle_path, (bundle, source_root) in bundles.items():
        transaction.mkdir(bundle_path)
        with bound_absolute_directory(source_root) as descriptor:
            entries = snapshot_exact_directory(
                descriptor,
                regular_names=_BUNDLE_FILES,
            )
            for name in sorted(_BUNDLE_FILES):
                payload = read_regular_at(
                    descriptor,
                    name,
                    _MAX_ASSET_BYTES,
                    expected_stat=entries[name],
                )
                relative = f"{bundle_path}/{name}"
                transaction.write(relative, payload)
                payloads[relative] = payload
            revalidate_entries(descriptor, entries)
        with (
            bound_child_directory(transaction.descriptor, "assets") as assets_fd,
            bound_child_directory(assets_fd, bundle.asset_bundle_sha256) as bundle_fd,
        ):
            if (
                publication.verify_asset_bundle_fd(
                    bundle_fd,
                    bundle.native_audit_run,
                )
                != bundle
            ):
                raise RuntimeError("dataset staged asset verification changed")
    metadata = {
        "manifest.json": _canonical_model_bytes(manifest) + b"\n",
        "records.jsonl": records_payload,
        "report.json": report_payload,
    }
    for name, payload in metadata.items():
        transaction.write(name, payload)
        payloads[name] = payload
    checksum = _checksum_payload(payloads)
    transaction.write("checksums.sha256", checksum)
    payloads["checksums.sha256"] = checksum
    return payloads


def _read_or_write_exact(
    descriptor: int,
    name: str,
    payload: bytes,
    created: dict[Path, tuple[int, int]],
    relative: Path,
) -> None:
    try:
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        written = write_regular_sync_at(descriptor, name, payload)
        created[relative] = (written.st_dev, written.st_ino)
        return
    observed = read_regular_at(
        descriptor,
        name,
        max(1, len(payload)),
        expected_stat=item,
    )
    if observed != payload:
        raise FileExistsError(f"dataset public entry differs: {relative.as_posix()}")


def _ensure_exact_directory(
    descriptor: int,
    name: str,
    created: dict[Path, tuple[int, int]],
    directories: dict[Path, tuple[int, int]],
    relative: Path,
) -> None:
    try:
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        os.mkdir(name, mode=0o700, dir_fd=descriptor)
        sync_directory_fd(descriptor)
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        created[relative] = (item.st_dev, item.st_ino)
    if not stat.S_ISDIR(item.st_mode):
        raise ValueError(f"dataset public directory is unsafe: {relative.as_posix()}")
    identity = (item.st_dev, item.st_ino)
    previous = directories.setdefault(relative, identity)
    if previous != identity:
        raise RuntimeError("dataset public directory identity changed")


def _copy_staged_public_index(
    root_descriptor: int,
    transaction,
    payloads: Mapping[str, bytes],
    created: dict[Path, tuple[int, int]],
    directories: dict[Path, tuple[int, int]],
) -> None:
    _ensure_exact_directory(
        root_descriptor,
        "assets",
        created,
        directories,
        Path("assets"),
    )
    with (
        bound_child_directory(root_descriptor, "assets") as final_assets_fd,
        bound_child_directory(transaction.descriptor, "assets") as staged_fd,
    ):
        staged_assets = _existing_names(
            staged_fd,
            maximum=max(1, len(payloads)),
        )
        for digest in sorted(staged_assets):
            _ensure_exact_directory(
                final_assets_fd,
                digest,
                created,
                directories,
                Path("assets") / digest,
            )
            with (
                bound_child_directory(staged_fd, digest) as source_fd,
                bound_child_directory(final_assets_fd, digest) as target_fd,
            ):
                source_entries = snapshot_exact_directory(
                    source_fd,
                    regular_names=_BUNDLE_FILES,
                )
                for name in sorted(_BUNDLE_FILES):
                    payload = read_regular_at(
                        source_fd,
                        name,
                        _MAX_ASSET_BYTES,
                        expected_stat=source_entries[name],
                    )
                    _read_or_write_exact(
                        target_fd,
                        name,
                        payload,
                        created,
                        Path("assets") / digest / name,
                    )
                revalidate_entries(source_fd, source_entries)
    for name in ("records.jsonl", "report.json", "checksums.sha256"):
        _read_or_write_exact(
            root_descriptor,
            name,
            payloads[name],
            created,
            Path(name),
        )
    _read_or_write_exact(
        root_descriptor,
        "manifest.json",
        payloads["manifest.json"],
        created,
        Path("manifest.json"),
    )
    sync_directory_fd(root_descriptor)


@contextmanager
def _bound_rollback_parent(
    root_descriptor: int,
    relative: Path,
    directories: Mapping[Path, tuple[int, int]],
) -> Iterator[int]:
    with ExitStack() as stack:
        descriptor = root_descriptor
        opened: list[tuple[int, str, tuple[int, int]]] = []
        prefix = Path()
        for component in relative.parts:
            prefix /= component
            expected = directories.get(prefix)
            if expected is None:
                raise RuntimeError("dataset rollback parent ownership is unknown")
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(before.st_mode)
                or (
                    before.st_dev,
                    before.st_ino,
                )
                != expected
            ):
                raise RuntimeError("dataset rollback parent identity changed")
            parent_descriptor = descriptor
            descriptor = stack.enter_context(
                open_directory(component, dir_fd=parent_descriptor)
            )
            if directory_identity_fd(descriptor) != expected:
                raise RuntimeError("dataset rollback parent binding changed")
            opened.append((parent_descriptor, component, expected))
        yield descriptor
        for parent_descriptor, component, expected in reversed(opened):
            current = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or (
                    current.st_dev,
                    current.st_ino,
                )
                != expected
            ):
                raise RuntimeError("dataset rollback parent binding changed")


def _rollback_created_fd(
    root_descriptor: int,
    created: Mapping[Path, tuple[int, int]],
    directories: Mapping[Path, tuple[int, int]],
) -> None:
    errors: list[BaseException] = []
    for relative, identity in sorted(
        created.items(), key=lambda item: len(item[0].parts), reverse=True
    ):
        try:
            with _bound_rollback_parent(
                root_descriptor,
                relative.parent,
                directories,
            ) as parent_fd:
                item = os.stat(
                    relative.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (item.st_dev, item.st_ino) != identity:
                    raise RuntimeError("dataset rollback entry identity changed")
                if stat.S_ISDIR(item.st_mode):
                    os.rmdir(relative.name, dir_fd=parent_fd)
                elif stat.S_ISREG(item.st_mode):
                    os.unlink(relative.name, dir_fd=parent_fd)
                else:
                    raise RuntimeError("dataset rollback entry type changed")
                sync_directory_fd(parent_fd)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
    if errors:
        raise RuntimeError("dataset public index rollback was incomplete") from errors[
            0
        ]


def _publication_root_binding(
    parent,
    root_name: str,
    root_identity: tuple[int, int],
) -> bool | None:
    try:
        parent.validate()
    except BaseException:  # noqa: BLE001
        return None
    try:
        current = os.stat(
            root_name,
            dir_fd=parent.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except BaseException:  # noqa: BLE001
        return None
    if (
        stat.S_ISDIR(current.st_mode)
        and (
            current.st_dev,
            current.st_ino,
        )
        == root_identity
    ):
        return True
    return None


def _raise_publication_failure(
    root: Path,
    parent,
    root_descriptor: int,
    root_identity: tuple[int, int],
    created: Mapping[Path, tuple[int, int]],
    directories: Mapping[Path, tuple[int, int]],
    active_error: BaseException,
) -> None:
    rollback_error: BaseException | None = None
    try:
        _rollback_created_fd(root_descriptor, created, directories)
    except BaseException as error:  # noqa: BLE001
        rollback_error = error
        active_error.add_note(str(error))
    binding = _publication_root_binding(parent, root.name, root_identity)
    if rollback_error is None and binding is True:
        raise active_error
    if rollback_error is None and binding is False:
        raise CompetitionNativePublicationError(
            root,
            published=False,
            recovery_name=None,
            detail="dataset root moved during public index publication and was rolled back",
        ) from active_error
    detail = (
        "dataset public index rollback could not prove complete cleanup"
        if rollback_error is not None
        else "dataset root binding became foreign during public index publication"
    )
    raise CompetitionNativePublicationError(
        root,
        published=None,
        recovery_name=None,
        detail=detail,
    ) from active_error


def _publish_dataset_index(
    root: Path,
    records_payload: bytes,
    report_payload: bytes,
    manifest: DatasetManifest,
    bundles: Mapping[str, tuple[publication.AssetBundle, Path]],
) -> GenerationReport:
    stage_target = root.parent / f"{root.name}-dataset-index"
    created: dict[Path, tuple[int, int]] = {}
    directories: dict[Path, tuple[int, int]] = {}
    with (
        open_native_output_parent(stage_target) as parent,
        parent.create_staging(label="dataset-index") as transaction,
    ):
        payloads = _stage_public_index(
            transaction,
            records_payload,
            report_payload,
            manifest,
            bundles,
        )
        transaction.fsync()
        seal = transaction.seal()
        staged_manifest, staged_report, _ = _verify_public_index_fd(
            transaction.descriptor,
            with_state=False,
        )
        if staged_manifest != manifest:
            raise RuntimeError("dataset index staging verification changed")
        transaction.validate_seal(seal)
        parent.validate()
        root_entry = os.stat(
            root.name,
            dir_fd=parent.parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(root_entry.st_mode):
            raise ValueError("dataset publication root must be a real directory")
        with open_directory(
            root.name,
            dir_fd=parent.parent_descriptor,
        ) as root_descriptor:
            root_identity = directory_identity_fd(root_descriptor)
            if root_identity != (root_entry.st_dev, root_entry.st_ino):
                raise RuntimeError("dataset publication root binding changed")
            try:
                _copy_staged_public_index(
                    root_descriptor,
                    transaction,
                    payloads,
                    created,
                    directories,
                )
                final_report, _ = _verify_dataset_fd(root_descriptor)
                if final_report != staged_report:
                    raise RuntimeError("dataset index final verification changed")
                transaction.validate_seal(seal)
                if (
                    _publication_root_binding(parent, root.name, root_identity)
                    is not True
                ):
                    raise RuntimeError("dataset publication root binding changed")
                return final_report
            except BaseException as error:  # noqa: BLE001
                _raise_publication_failure(
                    root,
                    parent,
                    root_descriptor,
                    root_identity,
                    created,
                    directories,
                    error,
                )
                raise AssertionError("unreachable")


# Preserve supported public names and pickle lookup.
_checksum_payload.__module__ = "spatialcf.generation.workflows.dataset"
_stage_public_index.__module__ = "spatialcf.generation.workflows.dataset"
_read_or_write_exact.__module__ = "spatialcf.generation.workflows.dataset"
_ensure_exact_directory.__module__ = "spatialcf.generation.workflows.dataset"
_copy_staged_public_index.__module__ = "spatialcf.generation.workflows.dataset"
_bound_rollback_parent.__module__ = "spatialcf.generation.workflows.dataset"
_rollback_created_fd.__module__ = "spatialcf.generation.workflows.dataset"
_publication_root_binding.__module__ = "spatialcf.generation.workflows.dataset"
_raise_publication_failure.__module__ = "spatialcf.generation.workflows.dataset"
_publish_dataset_index.__module__ = "spatialcf.generation.workflows.dataset"
