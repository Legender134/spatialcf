"""Descriptor-safe filesystem primitives for immutable publications."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self

DirectoryIdentity = tuple[int, int]
EntryIdentity = tuple[int, int, int]
CreatedIdentityLedger = dict[tuple[str, ...], EntryIdentity]
StatFingerprint = tuple[int, int, int, int, int, int, int]


class BindingStatus(StrEnum):
    """How one retained parent currently binds an expected object."""

    ABSENT = "absent"
    OWNED = "owned"
    FOREIGN = "foreign"


class RenameLocation(StrEnum):
    """Truthful location of the expected identity after a rename attempt."""

    SOURCE = "source"
    OUTPUT = "output"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RenameReconciliation:
    location: RenameLocation
    source: BindingStatus
    output: BindingStatus


@dataclass(frozen=True, slots=True)
class NativeTreeSeal:
    """Exact created roster and metadata captured after semantic verification."""

    entries: tuple[tuple[tuple[str, ...], StatFingerprint], ...]


class OwnedRenameStateError(RuntimeError):
    """A rename whose expected identity cannot be located unambiguously."""

    def __init__(
        self,
        reconciliation: RenameReconciliation,
        detail: str,
    ) -> None:
        super().__init__(
            f"{detail}; source={reconciliation.source.value}; "
            f"output={reconciliation.output.value}"
        )
        self.reconciliation = reconciliation


class OwnedCleanupError(RuntimeError):
    """Cleanup stopped without deleting an entry of unknown ownership."""

    def __init__(self, detail: str, *, recovery_name: str | None) -> None:
        suffix = (
            f"; recovery binding={recovery_name}" if recovery_name is not None else ""
        )
        super().__init__(detail + suffix)
        self.recovery_name = recovery_name


class CompetitionNativePublicationError(RuntimeError):
    """A publication error with a truthful final-path visibility state."""

    def __init__(
        self,
        output_path: Path,
        *,
        published: bool | None,
        recovery_name: str | None,
        detail: str,
    ) -> None:
        state = (
            "published"
            if published is True
            else ("not published" if published is False else "unknown")
        )
        recovery = (
            f"; recovery binding={recovery_name}" if recovery_name is not None else ""
        )
        super().__init__(
            f"{detail}; publication state={state}: {output_path}{recovery}"
        )
        self.output_path = output_path
        self.published = published
        self.recovery_name = recovery_name
        self.detail = detail

    def _replace_recovery_name(self, recovery_name: str | None) -> None:
        """Refresh recovery telemetry after automatic cleanup has completed."""

        replacement = CompetitionNativePublicationError(
            self.output_path,
            published=self.published,
            recovery_name=recovery_name,
            detail=self.detail,
        )
        self.recovery_name = recovery_name
        self.args = replacement.args


def _directory_flags() -> int:
    required = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("competition native publication requires safe open flags")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise RuntimeError("competition native publication requires openat/statat")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _identity(result: os.stat_result) -> tuple[int, int, int]:
    return result.st_dev, result.st_ino, stat.S_IFMT(result.st_mode)


def _fingerprint(result: os.stat_result) -> StatFingerprint:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _entry_identity(result: os.stat_result) -> EntryIdentity:
    return result.st_dev, result.st_ino, stat.S_IFMT(result.st_mode)


def _checked_directory_identity(
    value: DirectoryIdentity | None,
    label: str,
) -> DirectoryIdentity | None:
    if value is None:
        return None
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(part) is not int for part in value)
    ):
        raise TypeError(
            f"competition native {label} must be an exact directory identity"
        )
    if any(part < 0 for part in value):
        raise ValueError(f"competition native {label} must be non-negative")
    return value


def _require_component(name: str, label: str) -> None:
    if (
        type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ValueError(f"competition native {label} is invalid")


def _relative_parts(relative: str) -> tuple[str, ...]:
    if type(relative) is not str:
        raise TypeError("competition native relative path must be a string")
    path = PurePosixPath(relative)
    parts = path.parts
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in relative
    ):
        raise ValueError("competition native relative path is invalid")
    for part in parts:
        _require_component(part, "relative path component")
    return parts


def _rename_no_replace_at(
    source_parent_descriptor: int,
    source_name: str,
    output_parent_descriptor: int,
    output_name: str,
) -> None:
    """Atomically rename two descriptor-relative bindings without overwrite."""

    _require_component(source_name, "rename source")
    _require_component(output_name, "rename output")
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "competition native publication requires Linux renameat2(RENAME_NOREPLACE)"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_descriptor,
        os.fsencode(source_name),
        output_parent_descriptor,
        os.fsencode(output_name),
        1,
    )
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(output_name)
    if code == errno.ENOENT:
        raise FileNotFoundError(source_name)
    unsupported = {
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.ENOSYS),
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
    if code in unsupported:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) unsupported by this filesystem")
    raise OSError(code, os.strerror(code), source_name)


def _binding_status_at(
    parent_descriptor: int,
    name: str,
    identity: DirectoryIdentity,
) -> BindingStatus:
    try:
        result = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return BindingStatus.ABSENT
    if stat.S_ISDIR(result.st_mode) and (result.st_dev, result.st_ino) == identity:
        return BindingStatus.OWNED
    return BindingStatus.FOREIGN


def _entry_binding_status_at(
    parent_descriptor: int,
    name: str,
    identity: EntryIdentity,
) -> BindingStatus:
    try:
        result = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return BindingStatus.ABSENT
    if _entry_identity(result) == identity:
        return BindingStatus.OWNED
    return BindingStatus.FOREIGN


def reconcile_owned_rename_at(
    parent_descriptor: int,
    source_name: str,
    output_name: str,
    identity: DirectoryIdentity,
) -> RenameReconciliation:
    """Locate an expected directory without mutating either public binding."""

    _require_component(source_name, "rename source")
    _require_component(output_name, "rename output")
    source = _binding_status_at(parent_descriptor, source_name, identity)
    output = _binding_status_at(parent_descriptor, output_name, identity)
    if source is BindingStatus.OWNED and output is not BindingStatus.OWNED:
        location = RenameLocation.SOURCE
    elif output is BindingStatus.OWNED and source is not BindingStatus.OWNED:
        location = RenameLocation.OUTPUT
    else:
        location = RenameLocation.UNKNOWN
    return RenameReconciliation(location=location, source=source, output=output)


def _recovery_name(
    reconciliation: RenameReconciliation,
    source_name: str,
    output_name: str,
) -> str | None:
    if reconciliation.source is BindingStatus.OWNED:
        return source_name
    if reconciliation.output is BindingStatus.OWNED:
        return output_name
    return None


def rename_owned_no_replace_at(
    parent_descriptor: int,
    source_name: str,
    output_name: str,
    identity: DirectoryIdentity,
    *,
    source_descriptor: int | None = None,
) -> RenameReconciliation:
    """Rename an owned directory and reconcile exceptions after the commit point.

    Linux cannot condition ``renameat2`` on a source inode.  A retained source
    descriptor and before/after identity checks therefore detect substitution;
    they do not make an actively hostile same-UID namespace impossible.  An
    unknown binding is reported and is never moved or deleted as compensation.
    """

    if source_name == output_name:
        raise ValueError("competition native rename names must differ")
    if (
        source_descriptor is not None
        and directory_identity_fd(source_descriptor) != identity
    ):
        raise RuntimeError("competition native retained source identity changed")
    before = reconcile_owned_rename_at(
        parent_descriptor,
        source_name,
        output_name,
        identity,
    )
    if before.location is not RenameLocation.SOURCE:
        raise OwnedRenameStateError(before, "competition native source is not owned")
    if before.output is not BindingStatus.ABSENT:
        raise FileExistsError(output_name)
    try:
        _rename_no_replace_at(
            parent_descriptor,
            source_name,
            parent_descriptor,
            output_name,
        )
    except BaseException as error:
        after = reconcile_owned_rename_at(
            parent_descriptor,
            source_name,
            output_name,
            identity,
        )
        if (
            after.location is RenameLocation.OUTPUT
            and after.source is BindingStatus.ABSENT
        ):
            return after
        if after.location is RenameLocation.SOURCE:
            raise
        raise OwnedRenameStateError(
            after,
            "competition native rename state is unknown",
        ) from error
    after = reconcile_owned_rename_at(
        parent_descriptor,
        source_name,
        output_name,
        identity,
    )
    if (
        after.location is not RenameLocation.OUTPUT
        or after.source is not BindingStatus.ABSENT
    ):
        raise OwnedRenameStateError(
            after,
            "competition native rename did not produce one owned output",
        )
    return after


@contextmanager
def bound_absolute_directory(
    path: Path,
    *,
    expected_identity: DirectoryIdentity | None = None,
) -> Iterator[int]:
    """Open every absolute path component and retain its physical binding."""

    if not isinstance(path, Path):
        raise TypeError("competition native root must be a Path")
    checked_expected_identity = _checked_directory_identity(
        expected_identity,
        "expected root identity",
    )
    absolute = Path(os.path.abspath(path))
    components = absolute.parts
    if not components or components[0] != os.path.sep:
        raise ValueError("competition native root must be absolute")
    flags = _directory_flags()
    descriptors: list[int] = [os.open(os.path.sep, flags)]
    bindings: list[tuple[int, str, tuple[int, int, int]]] = []
    try:
        for component in components[1:]:
            if component in {"", ".", ".."} or "/" in component:
                raise ValueError("competition native path component is invalid")
            parent = descriptors[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("competition native path contains a non-directory")
            child = os.open(component, flags, dir_fd=parent)
            try:
                after = os.fstat(child)
            except BaseException:
                os.close(child)
                raise
            if _identity(before) != _identity(after):
                os.close(child)
                raise RuntimeError("competition native path binding changed")
            bindings.append((parent, component, _identity(after)))
            descriptors.append(child)
        opened_root = os.fstat(descriptors[-1])
        if (
            checked_expected_identity is not None
            and (
                opened_root.st_dev,
                opened_root.st_ino,
            )
            != checked_expected_identity
        ):
            raise RuntimeError("competition native root identity changed")
        yield descriptors[-1]
    finally:
        active_error = sys.exc_info()[1]
        binding_error: BaseException | None = None
        try:
            for parent, component, expected in bindings:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
                if _identity(current) != expected:
                    raise RuntimeError("competition native path binding changed")
        except BaseException as error:  # noqa: BLE001
            binding_error = error
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        if binding_error is not None:
            if active_error is not None:
                active_error.add_note(str(binding_error))
            else:
                raise binding_error


@dataclass(slots=True)
class NativeOutputParent:
    """One retained output-parent namespace and all of its lexical ancestors."""

    output_path: Path
    parent_descriptor: int
    parent_identity: EntryIdentity
    ancestor_descriptors: list[int]
    ancestor_bindings: list[tuple[int, str, EntryIdentity]]
    closed: bool = False

    @property
    def output_name(self) -> str:
        return self.output_path.name

    def validate(self) -> None:
        if self.closed:
            raise RuntimeError("competition native output parent is closed")
        if _entry_identity(os.fstat(self.parent_descriptor)) != self.parent_identity:
            raise RuntimeError("competition native output parent identity changed")
        for parent, component, expected in reversed(self.ancestor_bindings):
            current = os.stat(
                component,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if _entry_identity(current) != expected:
                raise RuntimeError("competition native output parent binding changed")

    def ensure_absent(self, name: str) -> None:
        _require_component(name, "output binding")
        self.validate()
        try:
            os.stat(
                name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise FileExistsError(name)

    def fsync(self) -> None:
        self.validate()
        sync_directory_fd(self.parent_descriptor)

    def create_staging(
        self,
        *,
        label: str = "staging",
    ) -> NativeStagingTransaction:
        """Create, open, and own a private staging directory in one operation."""

        _require_component(label, "staging label")
        self.validate()
        for _ in range(4):
            name = f".{self.output_name}.{label}-{uuid.uuid4().hex}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=self.parent_descriptor)
            except FileExistsError:
                continue
            created: os.stat_result | None = None
            descriptor: int | None = None
            try:
                created = os.stat(
                    name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(created.st_mode):
                    raise RuntimeError(
                        "competition native staging creation changed type"
                    )
                descriptor = os.open(
                    name,
                    _directory_flags(),
                    dir_fd=self.parent_descriptor,
                )
                opened = os.fstat(descriptor)
                if _entry_identity(opened) != _entry_identity(created):
                    raise RuntimeError(
                        "competition native staging binding changed while opened"
                    )
                identity = (opened.st_dev, opened.st_ino)
                return NativeStagingTransaction(
                    parent=self,
                    name=name,
                    descriptor=descriptor,
                    identity=identity,
                    created={(): _entry_identity(opened)},
                )
            except BaseException:
                if descriptor is not None:
                    os.close(descriptor)
                if created is not None:
                    error = sys.exc_info()[1]
                    try:
                        _remove_expected_empty_binding_at(
                            self.parent_descriptor,
                            name,
                            _entry_identity(created),
                        )
                        sync_directory_fd(self.parent_descriptor)
                    except BaseException as cleanup_error:  # noqa: BLE001
                        error.add_note(str(cleanup_error))
                raise
        raise RuntimeError("competition native staging name allocation failed")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for descriptor in reversed(self.ancestor_descriptors):
            os.close(descriptor)

    def __enter__(self) -> Self:
        self.validate()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        active_error = sys.exc_info()[1]
        validation_error: BaseException | None = None
        try:
            self.validate()
        except BaseException as error:  # noqa: BLE001
            validation_error = error
        self.close()
        if validation_error is not None:
            if active_error is not None:
                active_error.add_note(str(validation_error))
            else:
                raise validation_error


@dataclass(slots=True)
class NativeStagingTransaction:
    """A retained staging root plus identities created by this transaction."""

    parent: NativeOutputParent
    name: str
    descriptor: int
    identity: DirectoryIdentity
    created: CreatedIdentityLedger
    location: RenameLocation = RenameLocation.SOURCE
    recovery_name: str | None = None
    active_seal: NativeTreeSeal | None = None
    root_relocated: bool = False
    _relocated_root_fingerprint: StatFingerprint | None = None
    closed: bool = False

    def mkdir(self, relative: str) -> None:
        self._require_source()
        parts = _relative_parts(relative)
        if parts in self.created:
            raise FileExistsError(relative)
        parent_descriptor = _open_created_parent(
            self.descriptor,
            parts[:-1],
            self.created,
        )
        try:
            name = parts[-1]
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            created = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            created_identity = _entry_identity(created)
            self.created[parts] = created_identity
            try:
                child = os.open(
                    name,
                    _directory_flags(),
                    dir_fd=parent_descriptor,
                )
            except BaseException:
                try:
                    _remove_expected_empty_binding_at(
                        parent_descriptor,
                        name,
                        created_identity,
                    )
                    del self.created[parts]
                except BaseException as cleanup_error:  # noqa: BLE001
                    sys.exc_info()[1].add_note(str(cleanup_error))
                raise
            try:
                opened = os.fstat(child)
                if _entry_identity(opened) != created_identity:
                    raise RuntimeError(
                        "competition native created directory binding changed"
                    )
            finally:
                os.close(child)
            sync_directory_fd(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def write(self, relative: str, payload: bytes) -> None:
        self._require_source()
        parts = _relative_parts(relative)
        if parts in self.created:
            raise FileExistsError(relative)
        parent_descriptor = _open_created_parent(
            self.descriptor,
            parts[:-1],
            self.created,
        )
        try:
            created = write_regular_sync_at(
                parent_descriptor,
                parts[-1],
                payload,
            )
            self.created[parts] = _entry_identity(created)
            sync_directory_fd(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def adopt_exact_tree(
        self,
        relative: str,
        regular_paths: set[str] | frozenset[str] | tuple[str, ...],
        directory_paths: (set[str] | frozenset[str] | tuple[str, ...]) = (),
    ) -> None:
        """Bind an exact externally-created subtree into the cleanup ledger.

        This is intended for a nested atomic publisher that creates one case
        directory inside this private staging root.  The complete file and
        directory roster is inspected through retained descriptors before any
        identity is adopted.
        """

        self._require_source()
        root_parts = _relative_parts(relative)
        if isinstance(regular_paths, str) or isinstance(directory_paths, str):
            raise TypeError("competition native adopted paths must be collections")
        regular = {_relative_parts(item) for item in regular_paths}
        directories = {_relative_parts(item) for item in directory_paths}
        if regular & directories:
            raise ValueError("competition native adopted tree schema overlaps")
        for parts in regular | directories:
            if parts[:-1] and parts[:-1] not in directories:
                raise ValueError(
                    "competition native adopted tree omits a parent directory"
                )
        planned = {root_parts} | {
            (*root_parts, *parts) for parts in regular | directories
        }
        if any(parts in self.created for parts in planned) or any(
            existing[: len(root_parts)] == root_parts
            for existing in self.created
            if len(existing) > len(root_parts)
        ):
            raise FileExistsError(relative)
        parent_descriptor = _open_created_parent(
            self.descriptor,
            root_parts[:-1],
            self.created,
        )
        captured: CreatedIdentityLedger = {}
        try:
            root_name = root_parts[-1]
            root_before = os.stat(
                root_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(root_before.st_mode):
                raise ValueError("competition native adopted root must be a directory")
            root_descriptor = os.open(
                root_name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
            try:
                root_opened = os.fstat(root_descriptor)
                if _entry_identity(root_opened) != _entry_identity(root_before):
                    raise RuntimeError(
                        "competition native adopted root binding changed"
                    )
                captured[root_parts] = _entry_identity(root_opened)
                _capture_exact_adopted_tree(
                    root_descriptor,
                    root_parts,
                    regular,
                    directories,
                    captured,
                )
            finally:
                os.close(root_descriptor)
            root_after = os.stat(
                root_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _fingerprint(root_after) != _fingerprint(root_before):
                raise RuntimeError(
                    "competition native adopted root changed during capture"
                )
        finally:
            os.close(parent_descriptor)
        self.created.update(captured)

    def fsync(self) -> None:
        self._require_open()
        for parts, expected in sorted(
            self.created.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if parts and expected[2] == stat.S_IFDIR:
                descriptor = _open_created_directory(
                    self.descriptor,
                    parts,
                    self.created,
                )
                try:
                    sync_directory_fd(descriptor)
                finally:
                    os.close(descriptor)
        sync_directory_fd(self.descriptor)

    def validate(self) -> None:
        """Require the retained tree to match every created identity exactly."""

        self._require_open()
        root_expected = self.created.get(())
        if root_expected is None or _entry_identity(os.fstat(self.descriptor)) != (
            root_expected
        ):
            raise RuntimeError("competition native staging root identity changed")
        _validate_created_directory_fd(self.descriptor, (), self.created)

    def seal(self) -> NativeTreeSeal:
        """Freeze the exact verified roster and every regular-file fingerprint."""

        self.validate()
        fingerprints = _capture_tree_seal_fd(
            self.descriptor,
            (),
            self.created,
        )
        seal = NativeTreeSeal(entries=tuple(sorted(fingerprints.items())))
        self.active_seal = seal
        self.root_relocated = False
        self._relocated_root_fingerprint = None
        return seal

    def validate_seal(self, seal: NativeTreeSeal | None = None) -> None:
        """Reject same-inode mutation or any roster/identity change after sealing."""

        self._require_open()
        checked = self.active_seal if seal is None else seal
        if type(checked) is not NativeTreeSeal:
            raise RuntimeError("competition native staging tree is not sealed")
        root_transition = (
            self.root_relocated and self._relocated_root_fingerprint is None
        )
        _validate_tree_seal_fd(
            self.descriptor,
            (),
            self.created,
            dict(checked.entries),
            relocated_root=self._relocated_root_fingerprint,
            allow_root_ctime_transition=root_transition,
        )
        if root_transition:
            self._relocated_root_fingerprint = _fingerprint(os.fstat(self.descriptor))

    def reconcile(self) -> RenameReconciliation:
        """Refresh the expected directory's truthful public-name location."""

        self._require_open()
        result = reconcile_owned_rename_at(
            self.parent.parent_descriptor,
            self.name,
            self.parent.output_name,
            self.identity,
        )
        self.location = result.location
        self.recovery_name = _recovery_name(
            result,
            self.name,
            self.parent.output_name,
        )
        if result.location is RenameLocation.OUTPUT:
            self.root_relocated = True
        return result

    def validate_location(
        self,
        expected: RenameLocation,
    ) -> RenameReconciliation:
        """Require exactly one pure source or output binding via the retained fd."""

        if expected not in {RenameLocation.SOURCE, RenameLocation.OUTPUT}:
            raise ValueError("competition native expected location must be exact")
        result = self.reconcile()
        pure = (
            result.location is expected
            and (result.source if expected is RenameLocation.SOURCE else result.output)
            is BindingStatus.OWNED
            and (result.output if expected is RenameLocation.SOURCE else result.source)
            is BindingStatus.ABSENT
        )
        if pure:
            return result
        published = (
            True
            if result.output is BindingStatus.OWNED
            else (
                False
                if result.source is BindingStatus.OWNED
                and result.output is not BindingStatus.OWNED
                else None
            )
        )
        raise CompetitionNativePublicationError(
            self.parent.output_path,
            published=published,
            recovery_name=self.recovery_name,
            detail="competition native publication binding is not exact",
        )

    def publish(self) -> RenameReconciliation:
        self._require_source()
        self.parent.validate()
        self.validate()
        self.validate_seal()
        self.fsync()
        try:
            result = rename_owned_no_replace_at(
                self.parent.parent_descriptor,
                self.name,
                self.parent.output_name,
                self.identity,
                source_descriptor=self.descriptor,
            )
        except OwnedRenameStateError as error:
            self.location = RenameLocation.UNKNOWN
            self.recovery_name = _recovery_name(
                error.reconciliation,
                self.name,
                self.parent.output_name,
            )
            self.root_relocated = self.recovery_name != self.name
            self._relocated_root_fingerprint = None
            raise CompetitionNativePublicationError(
                self.parent.output_path,
                published=None,
                recovery_name=self.recovery_name,
                detail="competition native publish rename state is unknown",
            ) from error
        self.location = RenameLocation.OUTPUT
        self.root_relocated = True
        self._relocated_root_fingerprint = None
        self.recovery_name = self.parent.output_name
        self.validate_seal()
        self.parent.validate()
        self.parent.fsync()
        self.validate_location(RenameLocation.OUTPUT)
        self.validate_seal()
        return result

    def rollback(self) -> RenameReconciliation:
        self._require_output()
        self.validate_location(RenameLocation.OUTPUT)
        # The public binding is still provably ours even when a post-publish
        # fault changed a sealed child.  Move that owned root out of the
        # canonical output name first; cleanup will independently compare the
        # creation ledger and preserve a recovery tree rather than delete an
        # injected entry.
        try:
            result = rename_owned_no_replace_at(
                self.parent.parent_descriptor,
                self.parent.output_name,
                self.name,
                self.identity,
                source_descriptor=self.descriptor,
            )
        except FileExistsError as error:
            canonical = self.reconcile()
            published = True if canonical.output is BindingStatus.OWNED else None
            publication_error = CompetitionNativePublicationError(
                self.parent.output_path,
                published=published,
                recovery_name=self.recovery_name,
                detail="competition native rollback destination is occupied",
            )
            self._note_parent_validation_error(publication_error)
            raise publication_error from error
        except OwnedRenameStateError as error:
            self.location = RenameLocation.UNKNOWN
            self._relocated_root_fingerprint = None
            self.recovery_name = _recovery_name(
                error.reconciliation,
                self.parent.output_name,
                self.name,
            )
            canonical = reconcile_owned_rename_at(
                self.parent.parent_descriptor,
                self.name,
                self.parent.output_name,
                self.identity,
            )
            published = True if canonical.output is BindingStatus.OWNED else None
            publication_error = CompetitionNativePublicationError(
                self.parent.output_path,
                published=published,
                recovery_name=self.recovery_name,
                detail="competition native rollback rename state is unknown",
            )
            self._note_parent_validation_error(publication_error)
            raise publication_error from error
        self.location = RenameLocation.SOURCE
        self.recovery_name = self.name
        self.root_relocated = True
        self._relocated_root_fingerprint = None
        sync_directory_fd(self.parent.parent_descriptor)
        result = self.validate_location(RenameLocation.SOURCE)
        try:
            self.parent.validate()
        except BaseException as error:
            raise CompetitionNativePublicationError(
                self.parent.output_path,
                published=False,
                recovery_name=self.recovery_name,
                detail=(
                    "competition native rollback completed but the output parent "
                    "binding changed"
                ),
            ) from error
        return result

    def cleanup(self) -> None:
        self._require_source()
        try:
            _cleanup_owned_tree_at(
                self.parent.parent_descriptor,
                self.name,
                self.descriptor,
                self.identity,
                self.created,
            )
        except OwnedCleanupError as error:
            self.location = RenameLocation.UNKNOWN
            self.recovery_name = error.recovery_name
            raise
        self.location = RenameLocation.UNKNOWN
        self.recovery_name = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        os.close(self.descriptor)

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("competition native staging transaction is closed")
        if directory_identity_fd(self.descriptor) != self.identity:
            raise RuntimeError("competition native staging descriptor identity changed")

    def _require_source(self) -> None:
        self._require_open()
        if self.location is not RenameLocation.SOURCE:
            raise RuntimeError("competition native staging is not at its source name")

    def _require_output(self) -> None:
        self._require_open()
        if self.location is not RenameLocation.OUTPUT:
            raise RuntimeError("competition native staging is not at its output name")

    def _note_parent_validation_error(self, error: BaseException) -> None:
        try:
            self.parent.validate()
        except BaseException as validation_error:  # noqa: BLE001
            error.add_note(str(validation_error))

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        active_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        if self.location is RenameLocation.SOURCE:
            try:
                self.cleanup()
            except BaseException as error:  # noqa: BLE001
                cleanup_error = error
        self.close()
        if (
            isinstance(active_error, CompetitionNativePublicationError)
            and active_error.published is False
        ):
            recovery_name = (
                cleanup_error.recovery_name
                if isinstance(cleanup_error, OwnedCleanupError)
                else (None if cleanup_error is None else self.recovery_name)
            )
            active_error._replace_recovery_name(recovery_name)
        if cleanup_error is not None:
            if active_error is not None:
                active_error.add_note(str(cleanup_error))
            else:
                raise cleanup_error


def open_native_output_parent(
    output_path: Path,
    *,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> NativeOutputParent:
    """Retain an output parent and every lexical ancestor without symlinks."""

    if not isinstance(output_path, Path):
        raise TypeError("competition native output must be a Path")
    checked_expected_parent_identity = _checked_directory_identity(
        expected_parent_identity,
        "expected output parent identity",
    )
    output = Path(os.path.abspath(output_path))
    if output == output.parent:
        raise ValueError("competition native output may not be filesystem root")
    _require_component(output.name, "output name")
    parent = output.parent
    flags = _directory_flags()
    descriptors: list[int] = [os.open(os.path.sep, flags)]
    bindings: list[tuple[int, str, EntryIdentity]] = []
    try:
        for component in parent.parts[1:]:
            _require_component(component, "output parent component")
            parent_descriptor = descriptors[-1]
            before = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "competition native output parent contains a non-directory"
                )
            child = os.open(component, flags, dir_fd=parent_descriptor)
            try:
                after = os.fstat(child)
            except BaseException:
                os.close(child)
                raise
            if _entry_identity(before) != _entry_identity(after):
                os.close(child)
                raise RuntimeError(
                    "competition native output parent binding changed while opened"
                )
            bindings.append((parent_descriptor, component, _entry_identity(after)))
            descriptors.append(child)
        parent_descriptor = descriptors[-1]
        parent_stat = os.fstat(parent_descriptor)
        if (
            checked_expected_parent_identity is not None
            and (
                parent_stat.st_dev,
                parent_stat.st_ino,
            )
            != checked_expected_parent_identity
        ):
            raise RuntimeError("competition native output parent identity changed")
        return NativeOutputParent(
            output_path=output,
            parent_descriptor=parent_descriptor,
            parent_identity=_entry_identity(parent_stat),
            ancestor_descriptors=descriptors,
            ancestor_bindings=bindings,
        )
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


@contextmanager
def open_directory(
    path: Path | str,
    *,
    dir_fd: int | None = None,
) -> Iterator[int]:
    """Hold one real directory so later path replacement cannot redirect reads."""

    flags = _directory_flags()
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    try:
        result = os.fstat(descriptor)
        if not stat.S_ISDIR(result.st_mode):
            raise ValueError("competition native path must be a real directory")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def bound_child_directory(parent_descriptor: int, name: str) -> Iterator[int]:
    """Open one child and ensure its parent name stays bound until return."""

    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError("competition native child name is invalid")
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("competition native child must be a directory")
    with open_directory(name, dir_fd=parent_descriptor) as descriptor:
        expected = _identity(os.fstat(descriptor))
        if _identity(before) != expected:
            raise RuntimeError("competition native child binding changed")
        yield descriptor
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _identity(current) != expected:
            raise RuntimeError("competition native child binding changed")


def directory_identity_fd(descriptor: int) -> DirectoryIdentity:
    result = os.fstat(descriptor)
    if not stat.S_ISDIR(result.st_mode):
        raise RuntimeError("competition native descriptor is not a directory")
    return result.st_dev, result.st_ino


def directory_identity(path: Path) -> DirectoryIdentity:
    with open_directory(path) as descriptor:
        return directory_identity_fd(descriptor)


def has_directory_identity(path: Path, identity: DirectoryIdentity) -> bool:
    try:
        return directory_identity(path) == identity
    except (OSError, RuntimeError, ValueError):
        return False


def scan_directory(
    descriptor: int,
    *,
    maximum_entries: int,
) -> dict[str, os.stat_result]:
    """Enumerate one directory with a hard entry bound and no path recursion."""

    if type(maximum_entries) is not int or maximum_entries < 0:
        raise ValueError("maximum_entries must be a non-negative exact integer")
    entries: dict[str, os.stat_result] = {}
    with os.scandir(descriptor) as iterator:
        for entry in iterator:
            if len(entries) >= maximum_entries:
                raise ValueError("competition native directory file set exceeds limit")
            if entry.name in entries or entry.name in {"", ".", ".."}:
                raise ValueError("competition native directory entry is invalid")
            entries[entry.name] = entry.stat(follow_symlinks=False)
    return entries


def snapshot_exact_directory(
    descriptor: int,
    *,
    regular_names: set[str] | frozenset[str],
    directory_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, os.stat_result]:
    expected = set(regular_names) | set(directory_names)
    if len(expected) != len(regular_names) + len(directory_names):
        raise ValueError("competition native directory schema overlaps")
    entries = scan_directory(descriptor, maximum_entries=len(expected))
    if set(entries) != expected:
        raise ValueError("competition native directory file set mismatch")
    for name, result in entries.items():
        if name in regular_names:
            if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
                raise ValueError(
                    "competition native entry must be regular and unlinked"
                )
        elif not stat.S_ISDIR(result.st_mode):
            raise ValueError("competition native entry must be one real directory")
    return entries


def read_regular_at(
    directory_descriptor: int,
    name: str,
    maximum_bytes: int,
    *,
    expected_stat: os.stat_result | None = None,
) -> bytes:
    """Read one direct child through an anchored directory descriptor."""

    if (
        type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ValueError("competition native filename is invalid")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be a positive exact integer")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise RuntimeError("competition native read requires openat/O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
            or (
                expected_stat is not None
                and _fingerprint(before) != _fingerprint(expected_stat)
            )
        ):
            raise ValueError(f"competition native input must be regular: {name}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) > maximum_bytes
            or len(payload) != after.st_size
            or _fingerprint(before) != _fingerprint(after)
            or _fingerprint(after) != _fingerprint(current)
        ):
            raise ValueError(f"competition native input changed while read: {name}")
        return payload
    finally:
        os.close(descriptor)


def revalidate_entries(
    descriptor: int,
    entries: dict[str, os.stat_result],
) -> None:
    for name, expected in entries.items():
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if _fingerprint(current) != _fingerprint(expected):
            raise RuntimeError("competition native directory entry changed")


def write_regular_sync_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> os.stat_result:
    """Create, retain, and durably flush one descriptor-relative file."""

    _require_component(name, "output filename")
    if type(payload) is not bytes:
        raise TypeError("competition native output payload must be bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o644, dir_fd=parent_descriptor)
    created: os.stat_result | None = None
    try:
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise RuntimeError("competition native created file is unsafe")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("competition native file write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            _entry_identity(final) != _entry_identity(created)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_size != len(payload)
        ):
            raise RuntimeError("competition native created file identity changed")
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _fingerprint(current) != _fingerprint(final):
            raise RuntimeError("competition native created file binding changed")
        return final
    except BaseException as error:
        if created is not None:
            try:
                _remove_expected_regular_binding_at(
                    parent_descriptor,
                    name,
                    _entry_identity(created),
                )
            except BaseException as cleanup_error:  # noqa: BLE001
                error.add_note(str(cleanup_error))
        raise
    finally:
        os.close(descriptor)


def write_regular_sync(path: Path, payload: bytes) -> None:
    """Create one file through a retained absolute parent capability."""

    if not isinstance(path, Path):
        raise TypeError("competition native output path must be a Path")
    absolute = Path(os.path.abspath(path))
    with bound_absolute_directory(absolute.parent) as descriptor:
        write_regular_sync_at(descriptor, absolute.name, payload)


def sync_directory_fd(descriptor: int) -> None:
    result = os.fstat(descriptor)
    if not stat.S_ISDIR(result.st_mode):
        raise RuntimeError("competition native fsync target is not a directory")
    os.fsync(descriptor)


def sync_directory(path: Path) -> None:
    with bound_absolute_directory(path) as descriptor:
        sync_directory_fd(descriptor)


def _open_created_directory(
    root_descriptor: int,
    parts: tuple[str, ...],
    created: CreatedIdentityLedger,
) -> int:
    root_expected = created.get(())
    if root_expected is None:
        raise RuntimeError("competition native created ledger has no root")
    descriptor = os.dup(root_descriptor)
    try:
        if _entry_identity(os.fstat(descriptor)) != root_expected:
            raise RuntimeError("competition native created root identity changed")
        prefix: tuple[str, ...] = ()
        for part in parts:
            prefix = (*prefix, part)
            expected = created.get(prefix)
            if expected is None or expected[2] != stat.S_IFDIR:
                raise RuntimeError(
                    "competition native created parent is not in the ledger"
                )
            before = os.stat(
                part,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if _entry_identity(before) != expected:
                raise RuntimeError(
                    "competition native created directory binding changed"
                )
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            try:
                opened = os.fstat(child)
            except BaseException:
                os.close(child)
                raise
            if _entry_identity(opened) != expected:
                os.close(child)
                raise RuntimeError(
                    "competition native created directory identity changed"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_created_parent(
    root_descriptor: int,
    parent_parts: tuple[str, ...],
    created: CreatedIdentityLedger,
) -> int:
    return _open_created_directory(root_descriptor, parent_parts, created)


def _capture_exact_adopted_tree(
    root_descriptor: int,
    root_parts: tuple[str, ...],
    regular_paths: set[tuple[str, ...]],
    directory_paths: set[tuple[str, ...]],
    captured: CreatedIdentityLedger,
) -> None:
    def capture(descriptor: int, relative: tuple[str, ...]) -> None:
        regular_names = {
            parts[-1]
            for parts in regular_paths
            if len(parts) == len(relative) + 1 and parts[:-1] == relative
        }
        directory_names = {
            parts[-1]
            for parts in directory_paths
            if len(parts) == len(relative) + 1 and parts[:-1] == relative
        }
        entries = snapshot_exact_directory(
            descriptor,
            regular_names=regular_names,
            directory_names=directory_names,
        )
        for name in sorted(regular_names):
            captured[(*root_parts, *relative, name)] = _entry_identity(entries[name])
        for name in sorted(directory_names):
            parts = (*relative, name)
            expected = _entry_identity(entries[name])
            captured[(*root_parts, *parts)] = expected
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                if _entry_identity(os.fstat(child)) != expected:
                    raise RuntimeError(
                        "competition native adopted directory identity changed"
                    )
                capture(child, parts)
            finally:
                os.close(child)
        revalidate_entries(descriptor, entries)

    capture(root_descriptor, ())


def _capture_tree_seal_fd(
    descriptor: int,
    prefix: tuple[str, ...],
    created: CreatedIdentityLedger,
) -> dict[tuple[str, ...], StatFingerprint]:
    fingerprints: dict[tuple[str, ...], StatFingerprint] = {}
    if not prefix:
        root = os.fstat(descriptor)
        if _entry_identity(root) != created.get(()) or not stat.S_ISDIR(root.st_mode):
            raise RuntimeError("competition native seal root identity changed")
        fingerprints[()] = _fingerprint(root)
    direct = {
        parts[-1]: (parts, identity)
        for parts, identity in created.items()
        if len(parts) == len(prefix) + 1 and parts[:-1] == prefix
    }
    entries = scan_directory(descriptor, maximum_entries=len(direct))
    if set(entries) != set(direct):
        raise RuntimeError("competition native seal file set changed")
    for name, (parts, expected) in direct.items():
        observed = entries[name]
        if _entry_identity(observed) != expected:
            raise RuntimeError("competition native seal entry identity changed")
        fingerprints[parts] = _fingerprint(observed)
        if expected[2] == stat.S_IFREG:
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError("competition native seal file is unsafe")
            continue
        if expected[2] != stat.S_IFDIR or not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("competition native seal entry type changed")
        child = os.open(name, _directory_flags(), dir_fd=descriptor)
        try:
            opened = os.fstat(child)
            if _fingerprint(opened) != _fingerprint(observed):
                raise RuntimeError(
                    "competition native seal directory changed while opened"
                )
            fingerprints.update(_capture_tree_seal_fd(child, parts, created))
        finally:
            os.close(child)
    revalidate_entries(descriptor, entries)
    return fingerprints


def _root_seal_matches(
    expected: StatFingerprint,
    observed: StatFingerprint,
    *,
    allow_ctime_change: bool,
) -> bool:
    if not allow_ctime_change:
        return observed == expected
    return observed[:-1] == expected[:-1]


def _validate_tree_seal_fd(
    descriptor: int,
    prefix: tuple[str, ...],
    created: CreatedIdentityLedger,
    sealed: dict[tuple[str, ...], StatFingerprint],
    *,
    relocated_root: StatFingerprint | None,
    allow_root_ctime_transition: bool,
) -> None:
    if set(sealed) != set(created):
        raise RuntimeError("competition native seal roster changed")
    if not prefix:
        root = os.fstat(descriptor)
        expected_root = relocated_root or sealed.get(())
        if expected_root is None or not _root_seal_matches(
            expected_root,
            _fingerprint(root),
            allow_ctime_change=allow_root_ctime_transition,
        ):
            raise RuntimeError("competition native sealed root changed")
    direct = {
        parts[-1]: (parts, identity)
        for parts, identity in created.items()
        if len(parts) == len(prefix) + 1 and parts[:-1] == prefix
    }
    entries = scan_directory(descriptor, maximum_entries=len(direct))
    if set(entries) != set(direct):
        raise RuntimeError("competition native sealed tree file set changed")
    for name, (parts, identity) in direct.items():
        observed = entries[name]
        if (
            _entry_identity(observed) != identity
            or _fingerprint(observed) != sealed[parts]
        ):
            raise RuntimeError("competition native sealed entry changed")
        if identity[2] == stat.S_IFREG:
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError("competition native sealed file is unsafe")
            continue
        if identity[2] != stat.S_IFDIR or not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("competition native sealed entry type changed")
        child = os.open(name, _directory_flags(), dir_fd=descriptor)
        try:
            if _fingerprint(os.fstat(child)) != sealed[parts]:
                raise RuntimeError("competition native sealed directory changed")
            _validate_tree_seal_fd(
                child,
                parts,
                created,
                sealed,
                relocated_root=relocated_root,
                allow_root_ctime_transition=allow_root_ctime_transition,
            )
        finally:
            os.close(child)
    revalidate_entries(descriptor, entries)


def _validate_created_directory_fd(
    descriptor: int,
    prefix: tuple[str, ...],
    created: CreatedIdentityLedger,
) -> None:
    direct = {
        parts[-1]: (parts, identity)
        for parts, identity in created.items()
        if len(parts) == len(prefix) + 1 and parts[:-1] == prefix
    }
    entries = scan_directory(descriptor, maximum_entries=len(direct))
    if set(entries) != set(direct):
        raise RuntimeError("competition native staging tree file set changed")
    for name, (parts, expected) in direct.items():
        observed = entries[name]
        if _entry_identity(observed) != expected:
            raise RuntimeError("competition native staging entry identity changed")
        if expected[2] == stat.S_IFREG:
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError("competition native staging file is unsafe")
            continue
        if expected[2] != stat.S_IFDIR or not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("competition native staging entry type changed")
        child = os.open(name, _directory_flags(), dir_fd=descriptor)
        try:
            if _entry_identity(os.fstat(child)) != expected:
                raise RuntimeError(
                    "competition native staging directory identity changed"
                )
            _validate_created_directory_fd(child, parts, created)
        finally:
            os.close(child)
    revalidate_entries(descriptor, entries)


def _restore_isolated_entry_at(
    parent_descriptor: int,
    isolated: str,
    original: str,
    isolated_identity: EntryIdentity,
) -> bool:
    """Best-effort restore of an entry isolated before ownership was known."""

    if (
        _entry_binding_status_at(
            parent_descriptor,
            isolated,
            isolated_identity,
        )
        is not BindingStatus.OWNED
    ):
        return False
    try:
        original_stat = os.stat(
            original,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        # Never overwrite or move either of two foreign bindings.
        del original_stat
        return False
    try:
        _rename_no_replace_at(
            parent_descriptor,
            isolated,
            parent_descriptor,
            original,
        )
    except BaseException:  # noqa: BLE001
        return (
            _entry_binding_status_at(
                parent_descriptor,
                original,
                isolated_identity,
            )
            is BindingStatus.OWNED
            and _entry_binding_status_at(
                parent_descriptor,
                isolated,
                isolated_identity,
            )
            is BindingStatus.ABSENT
        )
    return True


def _isolate_expected_binding_at(
    parent_descriptor: int,
    name: str,
    expected: EntryIdentity,
) -> str:
    """Move one binding to an unpredictable name, then prove its identity.

    ``renameat2`` has no compare-by-inode mode.  A same-UID actor that can
    discover and repeatedly rebind the random isolation name remains outside
    the guarantee.  When a normal substitution race is detected, the foreign
    entry is restored when possible and is never deleted.
    """

    _require_component(name, "cleanup binding")
    before = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if _entry_identity(before) != expected:
        raise OwnedCleanupError(
            "competition native cleanup binding is not owned",
            recovery_name=None,
        )
    for _ in range(4):
        isolated = f".{name}.cleanup-{uuid.uuid4().hex}"
        try:
            os.stat(
                isolated,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            continue
        rename_error: BaseException | None = None
        try:
            _rename_no_replace_at(
                parent_descriptor,
                name,
                parent_descriptor,
                isolated,
            )
        except BaseException as error:  # noqa: BLE001
            rename_error = error
        source = _entry_binding_status_at(parent_descriptor, name, expected)
        output = _entry_binding_status_at(parent_descriptor, isolated, expected)
        if output is BindingStatus.OWNED and source is BindingStatus.ABSENT:
            return isolated
        if source is BindingStatus.OWNED:
            if isinstance(rename_error, FileExistsError):
                continue
            if rename_error is not None:
                raise rename_error
            raise OwnedCleanupError(
                "competition native cleanup isolation did not commit",
                recovery_name=None,
            )

        # The syscall may have moved a replacement installed after the first
        # identity check.  Restore that replacement, but never delete it.
        recovery_name = isolated if output is BindingStatus.FOREIGN else None
        if output is BindingStatus.FOREIGN:
            foreign = os.stat(
                isolated,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _restore_isolated_entry_at(
                parent_descriptor,
                isolated,
                name,
                _entry_identity(foreign),
            ):
                recovery_name = name
        raise OwnedCleanupError(
            "competition native cleanup binding changed during isolation",
            recovery_name=recovery_name,
        ) from rename_error
    raise OwnedCleanupError(
        "competition native cleanup isolation name allocation failed",
        recovery_name=None,
    )


def _remove_expected_empty_binding_at(
    parent_descriptor: int,
    name: str,
    expected: EntryIdentity,
) -> None:
    isolated = _isolate_expected_binding_at(parent_descriptor, name, expected)
    current = os.stat(
        isolated,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if _entry_identity(current) != expected or not stat.S_ISDIR(current.st_mode):
        raise OwnedCleanupError(
            "competition native empty cleanup identity changed",
            recovery_name=isolated,
        )
    try:
        os.rmdir(isolated, dir_fd=parent_descriptor)
    except BaseException as error:
        raise OwnedCleanupError(
            "competition native empty cleanup failed",
            recovery_name=isolated,
        ) from error


def _remove_expected_regular_binding_at(
    parent_descriptor: int,
    name: str,
    expected: EntryIdentity,
) -> None:
    isolated = _isolate_expected_binding_at(parent_descriptor, name, expected)
    current = os.stat(
        isolated,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        _entry_identity(current) != expected
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
    ):
        raise OwnedCleanupError(
            "competition native regular cleanup identity changed",
            recovery_name=isolated,
        )
    try:
        os.unlink(isolated, dir_fd=parent_descriptor)
    except BaseException as error:
        raise OwnedCleanupError(
            "competition native regular cleanup failed",
            recovery_name=isolated,
        ) from error


def _cleanup_created_directory_fd(
    descriptor: int,
    prefix: tuple[str, ...],
    created: CreatedIdentityLedger,
) -> None:
    direct = {
        parts[-1]: (parts, identity)
        for parts, identity in created.items()
        if len(parts) == len(prefix) + 1 and parts[:-1] == prefix
    }
    entries = scan_directory(descriptor, maximum_entries=len(direct))
    if set(entries) != set(direct):
        raise RuntimeError(
            "competition native cleanup tree contains an unfamiliar entry"
        )
    for name in sorted(direct):
        parts, expected = direct[name]
        observed = entries[name]
        if _entry_identity(observed) != expected:
            raise RuntimeError("competition native cleanup child identity changed")
        isolated = _isolate_expected_binding_at(descriptor, name, expected)
        current = os.stat(
            isolated,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(current) != expected:
            raise RuntimeError("competition native isolated child identity changed")
        if expected[2] == stat.S_IFREG:
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise RuntimeError("competition native isolated file is unsafe")
            os.unlink(isolated, dir_fd=descriptor)
            del created[parts]
            continue
        if expected[2] != stat.S_IFDIR:
            raise RuntimeError("competition native cleanup ledger is unsafe")
        child = os.open(isolated, _directory_flags(), dir_fd=descriptor)
        try:
            if _entry_identity(os.fstat(child)) != expected:
                raise RuntimeError(
                    "competition native isolated directory identity changed"
                )
            _cleanup_created_directory_fd(child, parts, created)
        finally:
            os.close(child)
        rebound = os.stat(
            isolated,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(rebound) != expected:
            raise RuntimeError("competition native isolated directory binding changed")
        os.rmdir(isolated, dir_fd=descriptor)
        del created[parts]


def _cleanup_owned_tree_at(
    parent_descriptor: int,
    name: str,
    root_descriptor: int,
    identity: DirectoryIdentity,
    created: CreatedIdentityLedger,
) -> None:
    root_expected = created.get(())
    opened = os.fstat(root_descriptor)
    if (
        root_expected is None
        or root_expected != _entry_identity(opened)
        or identity != (opened.st_dev, opened.st_ino)
    ):
        raise OwnedCleanupError(
            "competition native cleanup root identity changed",
            recovery_name=None,
        )
    isolated = _isolate_expected_binding_at(parent_descriptor, name, root_expected)
    try:
        current = os.stat(
            isolated,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(current) != root_expected:
            raise RuntimeError(
                "competition native isolated cleanup root identity changed"
            )
        _cleanup_created_directory_fd(root_descriptor, (), created)
        current = os.stat(
            isolated,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(current) != root_expected:
            raise RuntimeError("competition native cleanup root binding changed")
        os.rmdir(isolated, dir_fd=parent_descriptor)
        del created[()]
        sync_directory_fd(parent_descriptor)
    except BaseException as error:
        if isinstance(error, OwnedCleanupError) and error.recovery_name == isolated:
            raise
        raise OwnedCleanupError(
            f"competition native cleanup stopped safely: {error}",
            recovery_name=isolated,
        ) from error


def rename_owned_no_replace(
    source: Path,
    destination: Path,
    identity: DirectoryIdentity,
) -> None:
    """Compatibility wrapper over one retained, fd-relative rename."""

    if not isinstance(source, Path) or not isinstance(destination, Path):
        raise TypeError("competition native rename paths must be Paths")
    source = Path(os.path.abspath(source))
    destination = Path(os.path.abspath(destination))
    if source.parent != destination.parent:
        raise ValueError("competition native rename must stay in one parent")
    with bound_absolute_directory(source.parent) as parent_descriptor:
        descriptor = os.open(
            source.name,
            _directory_flags(),
            dir_fd=parent_descriptor,
        )
        try:
            if directory_identity_fd(descriptor) != identity:
                raise RuntimeError("competition native rename source is not owned")
            rename_owned_no_replace_at(
                parent_descriptor,
                source.name,
                destination.name,
                identity,
                source_descriptor=descriptor,
            )
        finally:
            os.close(descriptor)


def rollback_owned_tree(
    output: Path,
    staging: Path,
    identity: DirectoryIdentity,
) -> None:
    """Move an owned visible tree back to staging and durably remove its name."""

    rename_owned_no_replace(output, staging, identity)
    sync_directory(Path(os.path.abspath(output)).parent)


__all__ = (
    "BindingStatus",
    "CompetitionNativePublicationError",
    "CreatedIdentityLedger",
    "DirectoryIdentity",
    "NativeOutputParent",
    "NativeStagingTransaction",
    "NativeTreeSeal",
    "OwnedCleanupError",
    "OwnedRenameStateError",
    "RenameLocation",
    "RenameReconciliation",
    "StatFingerprint",
    "bound_absolute_directory",
    "bound_child_directory",
    "directory_identity",
    "directory_identity_fd",
    "has_directory_identity",
    "open_directory",
    "open_native_output_parent",
    "read_regular_at",
    "reconcile_owned_rename_at",
    "rename_owned_no_replace",
    "rename_owned_no_replace_at",
    "revalidate_entries",
    "rollback_owned_tree",
    "scan_directory",
    "snapshot_exact_directory",
    "sync_directory",
    "sync_directory_fd",
    "write_regular_sync",
    "write_regular_sync_at",
)
