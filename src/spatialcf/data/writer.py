"""Durable, lock-owned publication of immutable dataset versions.

Writers retain advisory ownership locks for their full lifetime. Publication
uses an atomic no-replace rename, and a failed post-rename parent fsync is
rolled back before an error is reported.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from contextlib import suppress
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO

from spatialcf.data.models import FailureRecord, PairRecord
from spatialcf.data.profile import ArtifactProfile, RunProfile
from spatialcf.data.provenance import DATASET_MANIFEST_SCHEMA_VERSION
from spatialcf.data.split import assign_split

_LOCK_SCHEMA_VERSION = 1
_LOCK_OFFSET = 1 << 30
_LOCK_KEYS = frozenset(
    {"dataset_version", "pid", "schema_version", "staging_name", "token"}
)
_WINDOWS_REPARSE_POINT = 0x400


class DatasetDurabilityError(OSError):
    """Publication durability failed, with an explicit visibility outcome."""

    def __init__(
        self,
        final_path: Path,
        *,
        published: bool,
        detail: str,
        recovery_path: Path | None = None,
        recovery_required: bool = False,
    ) -> None:
        canonical_path_visible = _lexists(final_path)
        state = (
            "published"
            if published
            else (
                "not published; recovery required"
                if recovery_required
                else "rolled back"
            )
        )
        recovery = (
            f"; recovery path: {recovery_path}"
            if recovery_path is not None
            else ""
        )
        super().__init__(
            errno.EIO,
            f"{detail}; dataset is {state}: {final_path}{recovery}",
        )
        self.final_path = final_path
        self.published = published
        self.recovery_path = recovery_path
        self.recovery_required = recovery_required
        self.canonical_path_visible = canonical_path_visible


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _unsafe(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        result = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return bool(getattr(result, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)


def _validate_dataset_version(dataset_version: str) -> str:
    if (
        not isinstance(dataset_version, str)
        or not dataset_version.strip()
        or dataset_version in {".", ".."}
        or dataset_version.startswith(".")
    ):
        raise ValueError("dataset_version must be a non-hidden directory name")
    if (
        "/" in dataset_version
        or "\\" in dataset_version
        or PurePosixPath(dataset_version).is_absolute()
        or PureWindowsPath(dataset_version).is_absolute()
        or PureWindowsPath(dataset_version).drive
    ):
        raise ValueError("dataset_version must be a non-hidden directory name")
    return dataset_version


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(_canonical_value(value), **kwargs) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Rename without replacement, or fail closed when unavailable."""
    if os.name == "nt":
        # Windows MoveFile semantics used by os.rename fail when the
        # destination exists.  The precheck only improves the exception path.
        if _lexists(destination):
            raise FileExistsError(destination)
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
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
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        unsupported = {
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.ENOSYS),
            getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
        }
        if code in unsupported:
            raise RuntimeError(
                "renameat2(RENAME_NOREPLACE) unsupported by this filesystem"
            )
        raise OSError(code, os.strerror(code), destination)
    raise RuntimeError("no safe no-replace directory rename on this platform")


def _file_identity(result: os.stat_result) -> tuple[int, int]:
    return (result.st_dev, result.st_ino)


def _meaningful_file_identity(
    result: os.stat_result,
) -> tuple[int, int] | None:
    """Return a usable physical identity, tolerating unavailable zero fields."""
    device = getattr(result, "st_dev", 0)
    inode = getattr(result, "st_ino", 0)
    if (
        type(device) is not int
        or type(inode) is not int
        or device == 0
        or inode == 0
    ):
        return None
    return (device, inode)


def _identity_for_path(path: Path, *, directory: bool) -> tuple[int, int]:
    if not _lexists(path) or _unsafe(path):
        raise RuntimeError(f"unsafe or missing owned path: {path}")
    result = os.stat(path, follow_symlinks=False)
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(result.st_mode):
        raise RuntimeError(f"owned path has the wrong type: {path}")
    return _file_identity(result)


def _windows_open_lock(
    path: Path, *, create: bool, audit: bool = False
) -> BinaryIO:
    """Open an owner/recovery handle, or a read-only compatible audit."""
    import msvcrt

    if create and audit:
        raise ValueError("an audit handle cannot create a lock")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if audit:
        access = 0x80000000  # GENERIC_READ
        # The audit grants all sharing to remain compatible with the retained
        # owner's WRITE and DELETE access; its own access remains read-only.
        sharing = 0x1 | 0x2 | 0x4
    else:
        access = 0x80000000 | 0x40000000 | 0x00010000
        # GENERIC_READ | GENERIC_WRITE | DELETE
        sharing = 0x1  # FILE_SHARE_READ
    disposition = 1 if create else 3  # CREATE_NEW | OPEN_EXISTING
    flags = 0x80 | (0 if create else 0x00200000)
    handle = create_file(
        str(path),
        access,
        sharing,
        None,
        disposition,
        flags,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        code = ctypes.get_last_error()
        if create and code in {80, 183}:
            raise FileExistsError(path)
        if not create and code in {2, 3}:
            raise FileNotFoundError(path)
        raise OSError(code, ctypes.FormatError(code), path)
    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            (os.O_RDONLY if audit else os.O_RDWR)
            | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        close_handle(ctypes.c_void_p(handle))
        raise
    try:
        return os.fdopen(
            descriptor, "rb" if audit else "r+b", buffering=0
        )
    except BaseException:
        os.close(descriptor)
        raise


def _windows_mark_delete_on_close(stream: BinaryIO) -> None:
    """Make the retained owner handle deletion-pending before it is closed."""
    import msvcrt

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_int)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_file_information = kernel32.SetFileInformationByHandle
    set_file_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_file_information.restype = ctypes.c_int
    handle = msvcrt.get_osfhandle(stream.fileno())
    disposition = FileDispositionInfo(1)
    if not set_file_information(
        ctypes.c_void_p(handle),
        4,  # FileDispositionInfo
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        code = ctypes.get_last_error()
        raise OSError(code, ctypes.FormatError(code))


def _windows_live_open_error(error: OSError) -> bool:
    codes = {error.errno, getattr(error, "winerror", None)}
    return os.name == "nt" and bool(codes & {5, 32})


def _open_lock(path: Path, *, create: bool) -> BinaryIO:
    if os.name == "nt":
        return _windows_open_lock(path, create=create)
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError("safe advisory lock support is unavailable")
    flags = os.O_RDWR
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "r+b", buffering=0)


def _try_advisory_lock(stream: BinaryIO) -> bool:
    if os.name == "nt":
        import msvcrt

        stream.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        finally:
            stream.seek(0)
        return True
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import fcntl

        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True
    raise RuntimeError("safe advisory lock support is unavailable")


def _unlock_advisory(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            stream.seek(0)
        return
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return
    raise RuntimeError("safe advisory lock support is unavailable")


def _open_staging_directory(path: Path) -> int:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError("safe staging-directory locking is unavailable")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe staging-directory open flags are unavailable")
    return os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )


def _try_staging_lock(descriptor: int) -> bool:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _unlock_staging(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _pid_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = open_process(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            code = ctypes.get_last_error()
            if code == 87:  # ERROR_INVALID_PARAMETER: no such process
                return False
            if code == 5:  # access denied: fail closed as live
                return True
            raise OSError(code, ctypes.FormatError(code))
        try:
            exit_code = ctypes.c_uint32()
            get_exit = kernel32.GetExitCodeProcess
            get_exit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            get_exit.restype = ctypes.c_int
            if not get_exit(handle, ctypes.byref(exit_code)):
                code = ctypes.get_last_error()
                raise OSError(code, ctypes.FormatError(code))
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            close_handle(handle)
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    raise RuntimeError("safe PID liveness detection is unavailable")


def _read_lock_metadata(stream: BinaryIO) -> dict[str, Any]:
    stream.seek(0)
    payload = stream.read()
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("dataset lock metadata is invalid") from error
    if not isinstance(metadata, dict) or frozenset(metadata) != _LOCK_KEYS:
        raise RuntimeError("dataset lock metadata schema is invalid")
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != _LOCK_SCHEMA_VERSION
        or type(metadata["pid"]) is not int
        or metadata["pid"] <= 0
        or not isinstance(metadata["dataset_version"], str)
        or not isinstance(metadata["staging_name"], str)
        or not isinstance(metadata["token"], str)
        or len(metadata["token"]) != 32
        or any(character not in "0123456789abcdef" for character in metadata["token"])
        or payload != _json_bytes(metadata)
    ):
        raise RuntimeError("dataset lock metadata schema is invalid")
    return metadata


class DatasetWriter:
    def __init__(
        self,
        artifact_root: Path,
        dataset_version: str,
        profile: ArtifactProfile = ArtifactProfile.for_run(RunProfile.EVIDENCE),
    ) -> None:
        version = _validate_dataset_version(dataset_version)
        self._version = version
        self._profile = profile
        self.dataset_root = Path(artifact_root) / "datasets"
        if _lexists(self.dataset_root) and _unsafe(self.dataset_root):
            raise ValueError("unsafe dataset root")
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.final_root = self.dataset_root / version
        self.staging_root = self.dataset_root / f".{version}.tmp"
        self.lock_path = self.dataset_root / f".{version}.lock"
        for path in (self.final_root, self.staging_root, self.lock_path):
            if _lexists(path) and _unsafe(path):
                raise ValueError("unsafe dataset version entry")
        if _lexists(self.final_root) or _lexists(self.staging_root):
            raise FileExistsError(version)

        self._token = uuid.uuid4().hex
        self._lock_stream: BinaryIO | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._lock_metadata: dict[str, Any] | None = None
        self._lock_held = False
        self._owns_lock = False
        self._staging_identity: tuple[int, int] | None = None
        self._staging_descriptor: int | None = None
        self._staging_lock_held = False
        self._finalized = False
        self._pairs: list[PairRecord] = []
        self._failures: list[FailureRecord] = []
        self._generation: dict[str, Any] | None = None
        self._pair_ids: set[str] = set()
        self._request_ids: set[str] = set()
        try:
            self._create_lock()
            self.staging_root.mkdir()
            self._staging_identity = _identity_for_path(
                self.staging_root, directory=True
            )
            self._acquire_staging_lock()
            (self.staging_root / "assets").mkdir()
            (self.staging_root / "topdown").mkdir()
        except BaseException:
            try:
                if self._owns_lock:
                    self._cleanup_owned()
                else:
                    self._discard_initial_lock()
            except BaseException:
                self._owns_lock = False
                try:
                    self._close_staging_lock()
                finally:
                    self._close_lock_without_unlink()
                raise
            raise

    def __del__(self) -> None:
        """Release OS handles when a caller abandons an unfinished writer.

        Deliberately keep the lock and staging paths intact: they are durable
        recovery evidence and may only be removed by the authenticated abort,
        finalize, or stale-recovery paths.  The finalizer merely prevents the
        advisory-lock streams themselves from leaking into later work.
        """
        descriptor = getattr(self, "_staging_descriptor", None)
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
            self._staging_descriptor = None
            self._staging_lock_held = False
        stream = getattr(self, "_lock_stream", None)
        if stream is not None:
            with suppress(OSError, ValueError):
                stream.close()
            self._lock_stream = None
            self._lock_held = False

    def _create_lock(self) -> None:
        stream = _open_lock(self.lock_path, create=True)
        self._lock_stream = stream
        self._lock_identity = _file_identity(os.fstat(stream.fileno()))
        if not _try_advisory_lock(stream):
            raise RuntimeError("newly created dataset lock is unexpectedly locked")
        self._lock_held = True
        metadata = {
            "dataset_version": self._version,
            "pid": os.getpid(),
            "schema_version": _LOCK_SCHEMA_VERSION,
            "staging_name": self.staging_root.name,
            "token": self._token,
        }
        stream.seek(0)
        stream.truncate()
        stream.write(_json_bytes(metadata))
        stream.flush()
        os.fsync(stream.fileno())
        self._lock_metadata = metadata
        self._owns_lock = True
        self._ensure_owner()

    def _discard_initial_lock(self) -> None:
        stream = self._lock_stream
        identity = self._lock_identity
        if (
            stream is not None
            and identity is not None
            and _lexists(self.lock_path)
            and not _unsafe(self.lock_path)
            and _identity_for_path(self.lock_path, directory=False) == identity
        ):
            if os.name == "nt":
                _windows_mark_delete_on_close(stream)
            else:
                self.lock_path.unlink()
        self._close_lock_without_unlink()

    @classmethod
    def recover_stale(cls, artifact_root: Path, dataset_version: str) -> None:
        """Recover a crashed writer while retaining the orphan's lock."""
        writer = cls.__new__(cls)
        version = _validate_dataset_version(dataset_version)
        writer._version = version
        writer.dataset_root = Path(artifact_root) / "datasets"
        if _lexists(writer.dataset_root) and _unsafe(writer.dataset_root):
            raise ValueError("unsafe dataset root")
        writer.dataset_root.mkdir(parents=True, exist_ok=True)
        writer.final_root = writer.dataset_root / version
        writer.staging_root = writer.dataset_root / f".{version}.tmp"
        writer.lock_path = writer.dataset_root / f".{version}.lock"
        writer._token = ""
        writer._lock_stream = None
        writer._lock_identity = None
        writer._lock_metadata = None
        writer._lock_held = False
        writer._owns_lock = False
        writer._staging_identity = None
        writer._staging_descriptor = None
        writer._staging_lock_held = False
        writer._finalized = False
        if _lexists(writer.final_root):
            raise FileExistsError(version)
        if not _lexists(writer.lock_path):
            if _lexists(writer.staging_root):
                raise RuntimeError("orphan staging has no recoverable lock metadata")
            return
        if _unsafe(writer.lock_path):
            raise ValueError("unsafe stale lock entry")

        try:
            stream = _open_lock(writer.lock_path, create=False)
        except OSError as error:
            if _windows_live_open_error(error):
                raise RuntimeError("dataset writer lock is live") from error
            raise
        writer._lock_stream = stream
        writer._lock_identity = _file_identity(os.fstat(stream.fileno()))
        try:
            if not _try_advisory_lock(stream):
                raise RuntimeError("dataset writer lock is live")
            writer._lock_held = True
            metadata = _read_lock_metadata(stream)
            if (
                metadata["dataset_version"] != version
                or metadata["staging_name"] != writer.staging_root.name
            ):
                raise RuntimeError("stale lock metadata targets another dataset")
            if (
                _identity_for_path(writer.lock_path, directory=False)
                != writer._lock_identity
            ):
                raise RuntimeError("stale lock pathname changed during recovery")
            if _pid_is_alive(metadata["pid"]):
                raise RuntimeError("dataset writer PID is still live")
            writer._token = metadata["token"]
            writer._lock_metadata = metadata
            if _lexists(writer.staging_root):
                if _unsafe(writer.staging_root):
                    raise ValueError("unsafe stale staging entry")
                writer._staging_identity = _identity_for_path(
                    writer.staging_root, directory=True
                )
                writer._acquire_staging_lock()
            writer._owns_lock = True
            writer._cleanup_owned()
            writer._fsync_dir(writer.dataset_root)
        except BaseException:
            writer._owns_lock = False
            try:
                writer._close_staging_lock()
            finally:
                writer._close_lock_without_unlink()
            raise

    def write_pair(self, record: PairRecord) -> None:
        self._ensure_open()
        validated = PairRecord.model_validate(record.model_dump(mode="python"))
        if validated.pair_id in self._pair_ids:
            raise ValueError(f"duplicate pair_id: {validated.pair_id}")
        self._claim_request(validated.request_id)
        self._pair_ids.add(validated.pair_id)
        self._pairs.append(validated)

    def write_failure(self, record: FailureRecord) -> None:
        self._ensure_open()
        validated = FailureRecord.model_validate(record.model_dump(mode="python"))
        self._claim_request(validated.request_id)
        self._failures.append(validated)

    def register_generation_attestation(
        self,
        *,
        provenance_path: str,
        attempts_path: str,
        source_scene_paths: tuple[str, ...],
        attempted_requests: int,
        requested_pairs: int | None,
        attempt_limit: int | None,
    ) -> None:
        """Bind the non-pair files needed for deterministic official replay."""
        self._ensure_open()
        if self._generation is not None:
            raise ValueError("generation attestation is already registered")
        if type(attempted_requests) is not int or attempted_requests < 0:
            raise ValueError("attempted_requests must be a non-negative integer")
        for name, value in (
            ("requested_pairs", requested_pairs),
            ("attempt_limit", attempt_limit),
        ):
            if value is not None and (
                type(value) is not int or value <= 0
            ):
                raise ValueError(
                    f"{name} must be null or a positive exact integer"
                )
        if (requested_pairs is None) == (attempt_limit is None):
            raise ValueError(
                "exactly one of requested_pairs and attempt_limit is required"
            )
        if (
            attempt_limit is not None
            and attempted_requests != attempt_limit
        ):
            raise ValueError(
                "attempt_limit requires a complete exact attempt prefix"
            )
        paths = (provenance_path, attempts_path, *source_scene_paths)
        if len(set(paths)) != len(paths):
            raise ValueError("generation attestation paths must be unique")
        for relative in paths:
            if (
                not relative
                or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or PureWindowsPath(relative).is_absolute()
                or PureWindowsPath(relative).drive
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or relative.split("/", 1)[0] != "provenance"
            ):
                raise ValueError(
                    "generation attestation paths must be safe relative "
                    "POSIX paths under provenance/"
                )
        self._generation = {
            "attempt_limit": attempt_limit,
            "attempted_requests": attempted_requests,
            "attempts_path": attempts_path,
            "provenance_path": provenance_path,
            "requested_pairs": requested_pairs,
            "source_scene_paths": list(source_scene_paths),
        }

    def abort(self) -> None:
        self._ensure_owner()
        self._cleanup_owned()

    def finalize(self) -> Path:
        self._ensure_open()
        published_identity: tuple[int, int] | None = None
        try:
            self._validate_dataset()
            self._validate_artifacts()
            self._write_bytes(
                self.staging_root / "pairs.jsonl",
                b"".join(
                    _json_bytes(item.model_dump(mode="python"))
                    for item in self._pairs
                ),
            )
            self._write_bytes(
                self.staging_root / "failures.jsonl",
                b"".join(
                    _json_bytes(item.model_dump(mode="python"))
                    for item in self._failures
                ),
            )
            manifest: dict[str, Any] = {
                "accepted_pairs": len(self._pairs),
                "failures": len(self._failures),
                "required_artifacts": len(self._pairs)
                * len(PairRecord.artifact_path_fields()),
                "schema_version": (
                    DATASET_MANIFEST_SCHEMA_VERSION
                    if self._generation is not None
                    else 1
                ),
                "splits": {
                    split: sum(item.split == split for item in self._pairs)
                    for split in ("train", "dev", "test")
                },
            }
            if self._generation is not None:
                manifest["generation"] = self._generation
                manifest.update(
                    {
                        "run_profile": self._profile.run_profile.value,
                        "evidence_eligible": self._profile.evidence_eligible,
                    }
                )
            self._write_bytes(
                self.staging_root / "manifest.json",
                _json_bytes(manifest, pretty=True),
            )
            self._fsync_staging()
            self._write_checksums()
            self._validate_staged_files()
            self._fsync_dir(self.staging_root)

            self._ensure_owner()
            self._ensure_staging_owner()
            validated_files = self._snapshot_staged_files()
            if _lexists(self.final_root) or _unsafe(self.final_root):
                raise FileExistsError(self.final_root.name)
            _rename_no_replace(self.staging_root, self.final_root)
            expected_identity = self._staging_identity
            try:
                observed_identity = _identity_for_path(
                    self.final_root, directory=True
                )
                if (
                    expected_identity is None
                    or observed_identity != expected_identity
                ):
                    raise RuntimeError(
                        "published directory identity changed"
                    )
                self._validate_published_files(validated_files)
            except BaseException as validation_error:
                self._raise_unvalidated_publication(validation_error)
            published_identity = expected_identity
            self._ensure_staging_handle_owner()
            self._ensure_owner()
            self._fsync_dir(self.dataset_root)
            self._finalized = True
            self._release_lock()
            return self.final_root
        except DatasetDurabilityError:
            raise
        except BaseException as error:
            if published_identity is not None:
                try:
                    self._rollback_publication(published_identity)
                except BaseException as rollback_error:
                    raise DatasetDurabilityError(
                        self.final_root,
                        published=_lexists(self.final_root),
                        detail=f"publication rollback failed: {rollback_error}",
                    ) from error
            try:
                self._cleanup_owned()
            except BaseException as ownership_error:
                if published_identity is not None:
                    raise DatasetDurabilityError(
                        self.final_root,
                        published=_lexists(self.final_root),
                        detail=f"owner cleanup failed: {ownership_error}",
                    ) from error
                raise
            if published_identity is not None:
                try:
                    self._fsync_dir(self.dataset_root)
                except OSError as durability_error:
                    raise DatasetDurabilityError(
                        self.final_root,
                        published=False,
                        detail=f"publication durability failed: {durability_error}",
                    ) from error
            raise

    def _raise_unvalidated_publication(
        self,
        validation_error: BaseException,
    ) -> None:
        """Quarantine any unvalidated final entry and raise a truthful error."""
        recovery_path, quarantine_error, canonical_visible = (
            self._quarantine_unvalidated_publication()
        )
        cleanup_error: BaseException | None = None
        if not canonical_visible:
            try:
                self._cleanup_owned()
            except BaseException as error:
                cleanup_error = error
        if canonical_visible:
            detail = (
                "unvalidated final entry could not be quarantined"
            )
        elif recovery_path is None and quarantine_error is None:
            detail = (
                "unvalidated final entry was already absent before quarantine"
            )
        else:
            detail = "unvalidated final entry was quarantined"
        details = [f"{detail}: {validation_error}"]
        if quarantine_error is not None:
            details.append(f"quarantine failed: {quarantine_error}")
        if cleanup_error is not None:
            details.append(f"owner cleanup failed: {cleanup_error}")
        raise DatasetDurabilityError(
            self.final_root,
            published=False,
            detail="; ".join(details),
            recovery_path=recovery_path,
            recovery_required=True,
        ) from validation_error

    def _quarantine_unvalidated_publication(
        self,
    ) -> tuple[Path | None, BaseException | None, bool]:
        """Move any final-path object to an unpredictable sibling."""
        quarantine_path = self.dataset_root / (
            f".{self._version}.quarantine-{uuid.uuid4().hex}"
        )
        if not _lexists(self.final_root):
            return None, None, False
        try:
            before = os.stat(self.final_root, follow_symlinks=False)
        except BaseException as error:
            return self.final_root, error, _lexists(self.final_root)
        try:
            _rename_no_replace(self.final_root, quarantine_path)
            if _lexists(self.final_root):
                raise RuntimeError(
                    "canonical final path remains visible after quarantine"
                )
            if not _lexists(quarantine_path):
                raise RuntimeError("quarantine path is missing after rename")
            after = os.stat(quarantine_path, follow_symlinks=False)
            if (
                _file_identity(after) != _file_identity(before)
                or stat.S_IFMT(after.st_mode) != stat.S_IFMT(before.st_mode)
            ):
                raise RuntimeError("quarantined entry identity changed")
            self._fsync_dir(self.dataset_root)
        except BaseException as error:
            recovery_path = (
                self.final_root
                if _lexists(self.final_root)
                else (
                    quarantine_path
                    if _lexists(quarantine_path)
                    else None
                )
            )
            return recovery_path, error, _lexists(self.final_root)
        return quarantine_path, None, False

    def _rollback_publication(
        self, published_identity: tuple[int, int]
    ) -> None:
        self._ensure_owner()
        self._ensure_staging_handle_owner()
        if (
            _identity_for_path(self.final_root, directory=True)
            != published_identity
        ):
            raise RuntimeError("published dataset ownership changed")
        if _lexists(self.staging_root):
            raise RuntimeError("cannot roll back over an existing staging path")
        _rename_no_replace(self.final_root, self.staging_root)
        if (
            _identity_for_path(self.staging_root, directory=True)
            != published_identity
        ):
            try:
                _rename_no_replace(self.staging_root, self.final_root)
            finally:
                raise RuntimeError("rolled-back dataset identity changed")
        self._staging_identity = published_identity
        self._ensure_staging_owner()

    def _validate_dataset(self) -> None:
        held_categories: set[str] = set()
        held_combinations: set[tuple[str, str, str]] = set()
        scenes: dict[str, str] = {}
        for item in self._pairs:
            if item.split != assign_split(item.scene_id):
                raise ValueError(
                    "caller-supplied split does not match scene assignment"
                )
            if scenes.setdefault(item.scene_id, item.split) != item.split:
                raise ValueError("scene appears in multiple splits")
            combo = (
                item.subject_category,
                item.relation_after.value,
                item.reference_category,
            )
            if item.split == "test" and "unseen_category" in item.holdout_tags:
                held_categories.update(
                    (item.subject_category, item.reference_category)
                )
            if (
                item.split == "test"
                and "unseen_combination" in item.holdout_tags
            ):
                held_combinations.add(combo)
        for item in self._pairs:
            combo = (
                item.subject_category,
                item.relation_after.value,
                item.reference_category,
            )
            category_match = (
                item.subject_category in held_categories
                or item.reference_category in held_categories
            )
            combo_match = combo in held_combinations
            if item.split != "test" and (category_match or combo_match):
                raise ValueError(
                    "held category or combination leaked outside test"
                )
            if item.split == "test" and (
                category_match != ("unseen_category" in item.holdout_tags)
                or combo_match != ("unseen_combination" in item.holdout_tags)
            ):
                raise ValueError(
                    "holdout tags contradict record categories or combination"
                )

    def _validate_artifacts(self) -> None:
        seen: set[str] = set()
        for item in self._pairs:
            for field in PairRecord.artifact_path_fields():
                relative = getattr(item, field)
                target = self.staging_root.joinpath(
                    *PurePosixPath(relative).parts
                )
                normalized = PurePosixPath(relative).as_posix()
                if normalized in seen:
                    raise ValueError("duplicate artifact path")
                seen.add(normalized)
                parent = target.parent
                while parent != self.staging_root:
                    if _unsafe(parent):
                        raise ValueError(
                            "artifact path traverses symlink/reparse point"
                        )
                    parent = parent.parent
                if not target.is_file() or _unsafe(target):
                    raise ValueError("missing required artifact")
        if self._generation is not None:
            for relative in (
                self._generation["provenance_path"],
                self._generation["attempts_path"],
                *self._generation["source_scene_paths"],
            ):
                if relative in seen:
                    raise ValueError("duplicate attestation/artifact path")
                seen.add(relative)
                target = self.staging_root.joinpath(
                    *PurePosixPath(relative).parts
                )
                parent = target.parent
                while parent != self.staging_root:
                    if _unsafe(parent):
                        raise ValueError(
                            "generation attestation traverses "
                            "symlink/reparse point"
                        )
                    parent = parent.parent
                if not target.is_file() or _unsafe(target):
                    raise ValueError(
                        "missing registered generation attestation file"
                    )

    def _write_checksums(self) -> None:
        checksum_path = self.staging_root / "checksums.sha256"
        files = [
            path
            for path in self._validate_staged_files()
            if path != checksum_path
        ]
        self._write_bytes(
            checksum_path,
            "".join(
                f"{_sha256(path)}  "
                f"{path.relative_to(self.staging_root).as_posix()}\n"
                for path in files
            ).encode("utf-8"),
        )

    def _validate_staged_files(self) -> list[Path]:
        """Validate every staged entry without following links.

        Python exposes ``st_nlink`` on Windows/NTFS, so hardlinks are rejected
        there exactly as on POSIX.  Some filesystems report zero for device or
        inode; those unavailable identity fields are ignored while a meaningful
        link count is still required to be one.
        """
        self._ensure_staging_owner()
        files, _ = self._scan_file_tree(self.staging_root)
        return files

    def _snapshot_staged_files(
        self,
    ) -> dict[str, tuple[int, int] | None]:
        self._ensure_staging_owner()
        _, identities = self._scan_file_tree(self.staging_root)
        return identities

    def _validate_published_files(
        self,
        expected: dict[str, tuple[int, int] | None],
    ) -> None:
        _, observed = self._scan_file_tree(self.final_root)
        if frozenset(observed) != frozenset(expected):
            raise ValueError(
                "published file set differs from the validated staging tree"
            )
        changed = sorted(
            relative
            for relative, identity in expected.items()
            if identity is not None and observed[relative] != identity
        )
        if changed:
            raise ValueError(
                "published physical file identity differs from validated "
                f"staging: {changed[:5]}"
            )

    def _scan_file_tree(
        self,
        root: Path,
    ) -> tuple[list[Path], dict[str, tuple[int, int] | None]]:
        try:
            root_result = os.stat(root, follow_symlinks=False)
        except OSError as error:
            raise ValueError("staged/published tree root is missing") from error
        if (
            not stat.S_ISDIR(root_result.st_mode)
            or stat.S_ISLNK(root_result.st_mode)
            or getattr(root_result, "st_file_attributes", 0)
            & _WINDOWS_REPARSE_POINT
        ):
            raise ValueError(
                "staged/published tree root is not a safe directory"
            )
        files: list[Path] = []
        file_identities: dict[str, tuple[int, int] | None] = {}
        identities: dict[tuple[int, int], str] = {}
        unsafe_link: tuple[str, int] | None = None
        entries = sorted(
            root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in entries:
            relative = path.relative_to(root).as_posix()
            try:
                result = os.stat(path, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"staged path changed during validation: {relative}"
                ) from error
            if (
                stat.S_ISLNK(result.st_mode)
                or getattr(result, "st_file_attributes", 0)
                & _WINDOWS_REPARSE_POINT
            ):
                raise ValueError(
                    f"staged path is a symlink/reparse point: {relative}"
                )
            if stat.S_ISDIR(result.st_mode):
                continue
            if not stat.S_ISREG(result.st_mode):
                raise ValueError(
                    f"staged path is not a regular file: {relative}"
                )
            link_count = getattr(result, "st_nlink", 0)
            if (
                type(link_count) is int
                and link_count > 0
                and link_count != 1
            ):
                if unsafe_link is None:
                    unsafe_link = (relative, link_count)
            identity = _meaningful_file_identity(result)
            if identity is not None:
                previous = identities.setdefault(identity, relative)
                if previous != relative:
                    raise ValueError(
                        "staged regular files have duplicate physical "
                        f"identity: {previous}, {relative}"
                    )
            file_identities[relative] = identity
            files.append(path)
        if unsafe_link is not None:
            relative, link_count = unsafe_link
            raise ValueError(
                "staged regular file has an unsafe hardlink count "
                f"st_nlink={link_count}: {relative}"
            )
        return files, file_identities

    def _write_bytes(self, path: Path, data: bytes) -> None:
        with path.open("wb") as stream:
            stream.write(data)
            stream.flush()
        self._fsync_file(path)

    def _fsync_file(self, path: Path) -> None:
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())

    def _fsync_dir(self, path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            if os.name == "nt" and error.errno in {
                errno.EACCES,
                errno.EINVAL,
                errno.ENOTSUP,
            }:
                return
            raise

    def _fsync_staging(self) -> None:
        for path in self._validate_staged_files():
            self._fsync_file(path)
        directories = sorted(
            (
                item
                for item in self.staging_root.rglob("*")
                if item.is_dir()
            ),
            key=lambda item: (
                -len(item.relative_to(self.staging_root).parts),
                item.relative_to(self.staging_root).as_posix(),
            ),
        )
        for directory in directories:
            if _unsafe(directory):
                raise ValueError("staged directory is a symlink/reparse point")
            self._fsync_dir(directory)
        self._fsync_dir(self.staging_root)

    def _claim_request(self, request_id: str) -> None:
        if request_id in self._request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        self._request_ids.add(request_id)

    def _ensure_open(self) -> None:
        self._ensure_owner()
        self._ensure_staging_owner()
        if (
            self._finalized
            or self._staging_identity is None
            or _identity_for_path(self.staging_root, directory=True)
            != self._staging_identity
        ):
            raise RuntimeError("dataset writer is not open")

    def _ensure_owner(self) -> None:
        stream = self._lock_stream
        if (
            not self._owns_lock
            or not self._lock_held
            or stream is None
            or stream.closed
            or self._lock_identity is None
            or self._lock_metadata is None
        ):
            raise RuntimeError("dataset writer ownership is unavailable")
        try:
            if _file_identity(os.fstat(stream.fileno())) != self._lock_identity:
                raise RuntimeError("dataset writer lock handle changed")
            if (
                _identity_for_path(self.lock_path, directory=False)
                != self._lock_identity
            ):
                raise RuntimeError("dataset writer ownership pathname changed")
            if _read_lock_metadata(stream) != self._lock_metadata:
                raise RuntimeError("dataset writer ownership token changed")
        except OSError as error:
            raise RuntimeError(
                "dataset writer ownership is unavailable"
            ) from error

    def _acquire_staging_lock(self) -> None:
        if os.name == "nt":
            return
        identity = self._staging_identity
        if identity is None:
            raise RuntimeError("staging ownership identity is unavailable")
        descriptor = _open_staging_directory(self.staging_root)
        held = False
        try:
            if _file_identity(os.fstat(descriptor)) != identity:
                raise RuntimeError("staging directory handle identity changed")
            if not _try_staging_lock(descriptor):
                raise RuntimeError("dataset staging directory is live and locked")
            held = True
            if (
                _identity_for_path(self.staging_root, directory=True)
                != identity
            ):
                raise RuntimeError("staging directory pathname changed")
        except BaseException:
            try:
                if held:
                    _unlock_staging(descriptor)
            finally:
                os.close(descriptor)
            raise
        self._staging_descriptor = descriptor
        self._staging_lock_held = True

    def _ensure_staging_owner(self) -> None:
        if (
            self._staging_identity is None
            or _identity_for_path(self.staging_root, directory=True)
            != self._staging_identity
        ):
            raise RuntimeError("staging directory pathname changed")
        self._ensure_staging_handle_owner()

    def _ensure_staging_handle_owner(self) -> None:
        if os.name == "nt":
            return
        descriptor = self._staging_descriptor
        if (
            descriptor is None
            or not self._staging_lock_held
            or self._staging_identity is None
            or _file_identity(os.fstat(descriptor))
            != self._staging_identity
        ):
            raise RuntimeError("dataset staging ownership is unavailable")

    def _release_lock(self) -> None:
        if not self._owns_lock:
            return
        self._ensure_owner()
        if os.name == "nt":
            _windows_mark_delete_on_close(self._lock_stream)
        else:
            release_path = self.dataset_root / (
                f".{self._version}.release-{self._token}"
            )
            if _lexists(release_path):
                raise RuntimeError("owned lock release path already exists")
            _rename_no_replace(self.lock_path, release_path)
            if (
                _identity_for_path(release_path, directory=False)
                != self._lock_identity
                or _read_lock_metadata(self._lock_stream) != self._lock_metadata
            ):
                try:
                    _rename_no_replace(release_path, self.lock_path)
                finally:
                    raise RuntimeError(
                        "dataset writer ownership changed during release"
                    )
            release_path.unlink()
        self._owns_lock = False
        try:
            self._close_lock_without_unlink()
        finally:
            self._close_staging_lock()

    def _close_lock_without_unlink(self) -> None:
        stream = self._lock_stream
        if stream is None:
            return
        try:
            if self._lock_held:
                _unlock_advisory(stream)
        finally:
            self._lock_held = False
            stream.close()
            self._lock_stream = None

    def _close_staging_lock(self) -> None:
        descriptor = self._staging_descriptor
        if descriptor is None:
            return
        try:
            if self._staging_lock_held:
                _unlock_staging(descriptor)
        finally:
            self._staging_lock_held = False
            os.close(descriptor)
            self._staging_descriptor = None

    def _cleanup_owned(self) -> None:
        if not self._owns_lock:
            return
        self._ensure_owner()
        if _lexists(self.staging_root):
            if (
                self._staging_identity is None
                or _identity_for_path(self.staging_root, directory=True)
                != self._staging_identity
            ):
                raise RuntimeError("staging ownership changed")
            self._ensure_staging_owner()
            cleanup_path = self.dataset_root / (
                f".{self._version}.cleanup-{self._token}"
            )
            if _lexists(cleanup_path):
                raise RuntimeError("owned cleanup path already exists")
            _rename_no_replace(self.staging_root, cleanup_path)
            if (
                _identity_for_path(cleanup_path, directory=True)
                != self._staging_identity
            ):
                try:
                    _rename_no_replace(cleanup_path, self.staging_root)
                finally:
                    raise RuntimeError("staging ownership changed during cleanup")
            self._ensure_staging_handle_owner()
            try:
                self._ensure_owner()
                self._ensure_staging_handle_owner()
            except BaseException:
                _rename_no_replace(cleanup_path, self.staging_root)
                raise
            shutil.rmtree(cleanup_path)
            self._close_staging_lock()
            self._staging_identity = None
        self._release_lock()
