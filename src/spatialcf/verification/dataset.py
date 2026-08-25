"""Dataset records, immutable publication, and read-only verification."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import stat
import sys
import uuid
import warnings
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import mean
from typing import Any, BinaryIO, Literal, TypeVar

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from spatialcf.domain.request import (
    InterventionSpec,
    QualityTier,
    Relation,
    SolverStatus,
)
from spatialcf.domain.scene import Scene
from spatialcf.verification.artifacts import _rename_no_replace
from spatialcf.verification.integrity import (
    calculate_candidate_objective,
    calculate_weighted_objective,
    canonical_json_bytes,
    source_corpus_digest,
    topdown_payload,
    validate_generation_budget,
)
from spatialcf.verification.profile import (
    ArtifactProfile,
    RunProfile,
    profile_from_manifest,
)
from spatialcf.verification.provenance import (
    ATTESTED_MANIFEST_SCHEMA_VERSIONS,
    DATASET_MANIFEST_SCHEMA_VERSION,
    DATASET_SEED,
    GENERATOR_VERSION,
    LEGACY_ATTESTED_MANIFEST_SCHEMA_VERSION,
    AttemptEvidence,
    GenerationProvenance,
)
from spatialcf.verification.split import assign_split
from spatialcf.verification.verifier import Verifier

_DATASET_SEED = 20260723
_HOLDOUT_TAGS = frozenset({"unseen_scene", "unseen_category", "unseen_combination"})


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _validate_relative_path(value: str) -> str:
    """Keep artifact references inside the immutable dataset directory."""
    if not value or "\\" in value:
        raise ValueError("artifact path must be a non-empty relative POSIX path")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("artifact path must be a non-empty relative POSIX path")
    if value.split("/", 1)[0] not in {"assets", "scenes", "relations", "topdown"}:
        raise ValueError("artifact path must use an approved artifact prefix")
    if value in {"pairs.jsonl", "failures.jsonl", "manifest.json", "checksums.sha256"}:
        raise ValueError("artifact path may not alias dataset metadata")
    return value


class PairRecord(_FrozenRecord):
    """An accepted, independently verified counterfactual pair."""

    pair_id: str
    request_id: str
    scene_id: str
    split: Literal["train", "dev", "test"]
    holdout_tags: frozenset[str]
    source: str
    seed: int
    generator: str
    subject_id: str
    subject_category: str
    reference_id: str
    reference_category: str
    camera_id: str
    relation_before: Relation
    relation_after: Relation
    question: str
    answer_before: Relation
    answer_after: Relation
    scene_before_path: str
    scene_after_path: str
    rgb_before_path: str
    rgb_after_path: str
    depth_before_path: str
    depth_after_path: str
    instance_before_path: str
    instance_after_path: str
    pointcloud_before_path: str
    pointcloud_after_path: str
    topdown_path: str
    relation_graph_before_path: str
    relation_graph_after_path: str
    relation_diff: tuple[str, ...]
    normalized_edit_distance: float
    leakage_score: float
    visibility_change: float
    inverse_safety_margin: float
    solver_status: SolverStatus
    evaluated_candidates: int
    quality_flags: tuple[str, ...]
    quality: QualityTier
    generator_version: str

    @classmethod
    def artifact_path_fields(cls) -> tuple[str, ...]:
        return tuple(name for name in cls.model_fields if name.endswith("_path"))

    @field_validator(
        "pair_id",
        "request_id",
        "scene_id",
        "source",
        "generator",
        "subject_id",
        "subject_category",
        "reference_id",
        "reference_category",
        "camera_id",
        "question",
        "generator_version",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "required text fields must be non-empty and non-whitespace"
            )
        return value

    @field_validator(
        "scene_before_path",
        "scene_after_path",
        "rgb_before_path",
        "rgb_after_path",
        "depth_before_path",
        "depth_after_path",
        "instance_before_path",
        "instance_after_path",
        "pointcloud_before_path",
        "pointcloud_after_path",
        "topdown_path",
        "relation_graph_before_path",
        "relation_graph_after_path",
    )
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact path must be non-empty")
        return _validate_relative_path(value)

    @field_validator(
        "normalized_edit_distance",
        "leakage_score",
        "visibility_change",
        "inverse_safety_margin",
    )
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("scores must be finite and non-negative")
        return value

    @field_validator("normalized_edit_distance", "leakage_score")
    @classmethod
    def validate_normalized_score(cls, value: float) -> float:
        if value > 1.0:
            raise ValueError("normalized scores must not exceed one")
        return value

    @field_serializer("holdout_tags", when_used="json")
    def serialize_holdout_tags(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_accepted_pair(self) -> PairRecord:
        if any(not value for value in (self.pair_id, self.request_id, self.scene_id)):
            raise ValueError("pair_id, request_id, and scene_id must be non-empty")
        if self.seed != _DATASET_SEED:
            raise ValueError(
                f"seed must be the deterministic dataset seed {_DATASET_SEED}"
            )
        if self.subject_id == self.reference_id:
            raise ValueError("subject_id and reference_id must differ")
        if self.relation_before.opposite is not self.relation_after:
            raise ValueError("accepted pairs must use an opposite relation flip")
        if (
            self.answer_before is not self.relation_before
            or self.answer_after is not self.relation_after
        ):
            raise ValueError("answers must preserve independently verified relations")
        if self.solver_status is not SolverStatus.SUCCESS:
            raise ValueError(
                "accepted pairs require independent verifier status SUCCESS"
            )
        if self.quality is QualityTier.REJECTED:
            raise ValueError("accepted pairs cannot have REJECTED quality")
        if self.quality is QualityTier.PURE and (
            self.leakage_score != 0.0 or self.quality_flags != ("PURE",)
        ):
            raise ValueError("PURE pairs require zero leakage and a PURE quality flag")
        if self.quality is QualityTier.LOW_LEAKAGE and (
            self.leakage_score <= 0.0 or self.quality_flags != ("LOW_LEAKAGE",)
        ):
            raise ValueError(
                "LOW_LEAKAGE pairs require positive leakage and a LOW_LEAKAGE quality flag"
            )
        if not self.holdout_tags.issubset(_HOLDOUT_TAGS):
            raise ValueError("unknown holdout tag")
        if self.split == "test":
            if "unseen_scene" not in self.holdout_tags:
                raise ValueError("test pairs require the unseen_scene holdout tag")
            if self.quality is not QualityTier.PURE:
                raise ValueError("test split must remain PURE")
        elif self.holdout_tags:
            raise ValueError("holdout tags are permitted only on the test split")
        if self.evaluated_candidates < 0:
            raise ValueError("evaluated_candidates must be non-negative")
        return self


class FailureRecord(_FrozenRecord):
    """Append-only evidence for a request that did not become accepted data."""

    failure_id: str
    request_id: str
    scene_id: str
    subject_id: str
    reference_id: str
    relation_before: Relation
    relation_after: Relation
    generator: str
    generator_version: str
    seed: int
    status: SolverStatus
    reason: str
    evaluated_candidates: int

    @field_validator(
        "failure_id",
        "request_id",
        "scene_id",
        "subject_id",
        "reference_id",
        "generator",
        "generator_version",
        "reason",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "required text fields must be non-empty and non-whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_failure(self) -> FailureRecord:
        if any(
            not value
            for value in (
                self.failure_id,
                self.request_id,
                self.scene_id,
                self.reason,
            )
        ):
            raise ValueError(
                "failure_id, request_id, scene_id, and reason must be non-empty"
            )
        if self.seed != _DATASET_SEED:
            raise ValueError(
                f"seed must be the deterministic dataset seed {_DATASET_SEED}"
            )
        if self.subject_id == self.reference_id:
            raise ValueError("subject_id and reference_id must differ")
        if self.relation_before.opposite is not self.relation_after:
            raise ValueError("failures must retain an opposite relation flip")
        if self.status is SolverStatus.SUCCESS:
            raise ValueError("failure status must not be SUCCESS")
        if self.evaluated_candidates < 0:
            raise ValueError("evaluated_candidates must be non-negative")
        return self


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
            f"; recovery path: {recovery_path}" if recovery_path is not None else ""
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
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
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


def _file_identity(result: os.stat_result) -> tuple[int, int]:
    return (result.st_dev, result.st_ino)


def _meaningful_file_identity(
    result: os.stat_result,
) -> tuple[int, int] | None:
    """Return a usable physical identity, tolerating unavailable zero fields."""
    device = getattr(result, "st_dev", 0)
    inode = getattr(result, "st_ino", 0)
    if type(device) is not int or type(inode) is not int or device == 0 or inode == 0:
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


def _windows_open_lock(path: Path, *, create: bool, audit: bool = False) -> BinaryIO:
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
            (os.O_RDONLY if audit else os.O_RDWR) | getattr(os, "O_BINARY", 0),
        )
    except BaseException:
        close_handle(ctypes.c_void_p(handle))
        raise
    try:
        return os.fdopen(descriptor, "rb" if audit else "r+b", buffering=0)
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
        profile: ArtifactProfile = ArtifactProfile.for_run(RunProfile.EVIDENCE),  # noqa: B008
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
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be null or a positive exact integer")
        if (requested_pairs is None) == (attempt_limit is None):
            raise ValueError(
                "exactly one of requested_pairs and attempt_limit is required"
            )
        if attempt_limit is not None and attempted_requests != attempt_limit:
            raise ValueError("attempt_limit requires a complete exact attempt prefix")
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
                    _json_bytes(item.model_dump(mode="python")) for item in self._pairs
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
                observed_identity = _identity_for_path(self.final_root, directory=True)
                if expected_identity is None or observed_identity != expected_identity:
                    raise RuntimeError("published directory identity changed")
                self._validate_published_files(validated_files)
            except BaseException as validation_error:  # noqa: BLE001
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
                except BaseException as rollback_error:  # noqa: BLE001
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
            except BaseException as error:  # noqa: BLE001
                cleanup_error = error
        if canonical_visible:
            detail = "unvalidated final entry could not be quarantined"
        elif recovery_path is None and quarantine_error is None:
            detail = "unvalidated final entry was already absent before quarantine"
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
        except BaseException as error:  # noqa: BLE001
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
            if _file_identity(after) != _file_identity(before) or stat.S_IFMT(
                after.st_mode
            ) != stat.S_IFMT(before.st_mode):
                raise RuntimeError("quarantined entry identity changed")
            self._fsync_dir(self.dataset_root)
        except BaseException as error:  # noqa: BLE001
            recovery_path = (
                self.final_root
                if _lexists(self.final_root)
                else (quarantine_path if _lexists(quarantine_path) else None)
            )
            return recovery_path, error, _lexists(self.final_root)
        return quarantine_path, None, False

    def _rollback_publication(self, published_identity: tuple[int, int]) -> None:
        self._ensure_owner()
        self._ensure_staging_handle_owner()
        if _identity_for_path(self.final_root, directory=True) != published_identity:
            raise RuntimeError("published dataset ownership changed")
        if _lexists(self.staging_root):
            raise RuntimeError("cannot roll back over an existing staging path")
        _rename_no_replace(self.final_root, self.staging_root)
        if _identity_for_path(self.staging_root, directory=True) != published_identity:
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
                held_categories.update((item.subject_category, item.reference_category))
            if item.split == "test" and "unseen_combination" in item.holdout_tags:
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
                raise ValueError("held category or combination leaked outside test")
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
                target = self.staging_root.joinpath(*PurePosixPath(relative).parts)
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
                target = self.staging_root.joinpath(*PurePosixPath(relative).parts)
                parent = target.parent
                while parent != self.staging_root:
                    if _unsafe(parent):
                        raise ValueError(
                            "generation attestation traverses symlink/reparse point"
                        )
                    parent = parent.parent
                if not target.is_file() or _unsafe(target):
                    raise ValueError("missing registered generation attestation file")

    def _write_checksums(self) -> None:
        checksum_path = self.staging_root / "checksums.sha256"
        files = [
            path for path in self._validate_staged_files() if path != checksum_path
        ]
        self._write_bytes(
            checksum_path,
            "".join(
                f"{_sha256(path)}  {path.relative_to(self.staging_root).as_posix()}\n"
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
            or getattr(root_result, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
        ):
            raise ValueError("staged/published tree root is not a safe directory")
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
                or getattr(result, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
            ):
                raise ValueError(f"staged path is a symlink/reparse point: {relative}")
            if stat.S_ISDIR(result.st_mode):
                continue
            if not stat.S_ISREG(result.st_mode):
                raise ValueError(f"staged path is not a regular file: {relative}")
            link_count = getattr(result, "st_nlink", 0)
            if (  # noqa: SIM102
                type(link_count) is int and link_count > 0 and link_count != 1
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
            (item for item in self.staging_root.rglob("*") if item.is_dir()),
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
            raise RuntimeError("dataset writer ownership is unavailable") from error

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
            if _identity_for_path(self.staging_root, directory=True) != identity:
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
            or _file_identity(os.fstat(descriptor)) != self._staging_identity
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
                _identity_for_path(release_path, directory=False) != self._lock_identity
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


_WINDOWS_REPARSE_POINT = 0x400
_METADATA_FILES = frozenset(
    {"checksums.sha256", "failures.jsonl", "manifest.json", "pairs.jsonl"}
)
_MANIFEST_KEYS = frozenset(
    {
        "accepted_pairs",
        "failures",
        "required_artifacts",
        "schema_version",
        "splits",
    }
)
_ATTESTED_MANIFEST_KEYS = _MANIFEST_KEYS | {"generation"}
_GENERATION_MANIFEST_KEYS = frozenset(
    {
        "attempt_limit",
        "attempted_requests",
        "attempts_path",
        "provenance_path",
        "requested_pairs",
        "source_scene_paths",
    }
)
_RecordT = TypeVar("_RecordT", bound=BaseModel)
_DATASET_SEED = DATASET_SEED
_GENERATOR_VERSION = GENERATOR_VERSION
_GENERATOR_NAMES = frozenset({"spatialcf", "random", "target-only"})
_REQUEST_NAMESPACE = uuid.UUID("8ab06aa5-553d-5263-a00a-602d15c53ff5")
_PAIR_NAMESPACE = uuid.UUID("6807e184-f5c3-5812-b467-8783487a9e65")
_FAILURE_NAMESPACE = uuid.UUID("44858018-d898-56fc-afca-c1637cf4e8c6")
_MAX_IMAGE_DIMENSION = 8192
_MAX_IMAGE_PIXELS = 16_777_216
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_DEPTH_BYTES = 128 * 1024 * 1024
_MAX_POINTCLOUD_BYTES = 128 * 1024 * 1024
_MAX_DATASET_FILE_BYTES = 256 * 1024 * 1024
_MAX_IDENTITY_PAYLOAD_BYTES = 128 * 1024 * 1024
_MAX_IDENTITY_PAYLOAD_FILES = 1024


@dataclass(frozen=True)
class _Dataset:
    root: Path
    pairs: tuple[PairRecord, ...]
    failures: tuple[FailureRecord, ...]
    manifest: dict[str, Any]
    provenance: GenerationProvenance | None
    attempts: tuple[AttemptEvidence, ...]
    attempts_authenticated: bool

    @property
    def backend(self) -> str:
        return (
            self.provenance.adapter_backend
            if self.provenance is not None
            else "legacy-json"
        )


@dataclass(frozen=True)
class AuthenticatedDatasetIdentity:
    """Typed identity extracted from one checksummed dataset snapshot."""

    accepted_pairs: int
    run_profile: str
    evidence_eligible: bool
    provenance: GenerationProvenance
    attempts: tuple[AttemptEvidence, ...]
    scenes: tuple[Scene, ...]


@dataclass(frozen=True)
class _StreamedFileDigest:
    sha256: str


def _audit_unsafe(result: os.stat_result) -> bool:
    return bool(getattr(result, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)


def _regular_files(
    root: Path,
    *,
    descriptor_capability: bool = False,
) -> dict[str, Path]:
    if not descriptor_capability:
        current = Path(os.path.abspath(root))
        while True:
            if os.path.lexists(current):
                ancestor_result = current.stat(follow_symlinks=False)
                if current.is_symlink() or _audit_unsafe(ancestor_result):
                    raise ValueError(
                        "dataset root may not traverse a symlink/reparse point"
                    )
            if current == current.parent:
                break
            current = current.parent
    try:
        root_result = root.stat(follow_symlinks=descriptor_capability)
    except FileNotFoundError as error:
        raise ValueError(f"dataset root does not exist: {root}") from error
    if (
        (not descriptor_capability and root.is_symlink())
        or _audit_unsafe(root_result)
        or not stat.S_ISDIR(root_result.st_mode)
    ):
        raise ValueError("dataset root must be an ordinary directory")

    files: dict[str, Path] = {}
    physical_files: dict[tuple[int, int], str] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(
                f"cannot inspect dataset directory: {directory}"
            ) from error
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            try:
                result = Path(entry.path).stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"cannot inspect dataset entry: {relative}") from error
            if entry.is_symlink() or _audit_unsafe(result):
                raise ValueError(f"unsafe symlink/reparse dataset entry: {relative}")
            if stat.S_ISDIR(result.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(result.st_mode):
                if result.st_nlink != 1:
                    raise ValueError(
                        "dataset regular file has unsafe hardlink count; "
                        f"st_nlink={result.st_nlink}: {relative}"
                    )
                if result.st_size > _MAX_DATASET_FILE_BYTES:
                    raise ValueError(
                        f"dataset file exceeds resource byte limit: {relative}"
                    )
                identity = (result.st_dev, result.st_ino)
                duplicate = physical_files.get(identity)
                if duplicate is not None:
                    raise ValueError(
                        "dataset files must have unique physical file identity; "
                        f"hardlink detected: {duplicate} and {relative}"
                    )
                physical_files[identity] = relative
                files[relative] = Path(entry.path)
            else:
                raise ValueError(f"dataset entry is not a regular file: {relative}")
    return files


def _audit_validate_relative_path(value: str) -> str:
    if (
        not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError(f"invalid checksum path: {value!r}")
    return value


def _audit_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum_entries(
    checksum_bytes: bytes,
    actual_paths: set[str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    if b"\r" in checksum_bytes or (
        checksum_bytes and not checksum_bytes.endswith(b"\n")
    ):
        raise ValueError("checksums.sha256 must use canonical LF lines")
    try:
        lines = checksum_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("checksums.sha256 is not UTF-8") from error

    checksums: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"invalid checksum entry at line {line_number}")
        digest, relative = line[:64], _audit_validate_relative_path(line[66:])
        if (
            any(character not in "0123456789abcdef" for character in digest)
            or relative == "checksums.sha256"
        ):
            raise ValueError(f"invalid checksum entry at line {line_number}")
        if relative in checksums:
            raise ValueError(f"duplicate checksum path: {relative}")
        checksums[relative] = digest
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("checksum entries must be sorted by POSIX path")
    if set(checksums) != actual_paths:
        missing = sorted(actual_paths - checksums.keys())
        extra = sorted(checksums.keys() - actual_paths)
        raise ValueError(
            f"checksum file set does not match dataset files; "
            f"unlisted={missing[:5]} missing={extra[:5]}"
        )
    return checksums, tuple(ordered_paths)


def _validate_checksums(
    root: Path,
    files: Mapping[str, Path | bytes | _StreamedFileDigest],
) -> None:
    if not _METADATA_FILES.issubset(files):
        missing = sorted(_METADATA_FILES - files.keys())
        raise ValueError(f"missing dataset metadata files: {missing}")
    checksum_source = files["checksums.sha256"]
    actual_paths = set(files) - {"checksums.sha256"}
    if isinstance(checksum_source, bytes):
        checksum_bytes = checksum_source
    elif isinstance(checksum_source, Path):
        checksum_bytes = checksum_source.read_bytes()
    else:
        raise TypeError(
            "checksums.sha256 must be retained in the authenticated snapshot"
        )
    checksums, ordered_paths = _checksum_entries(
        checksum_bytes,
        actual_paths,
    )

    def digest_for(relative: str) -> str:
        source = files[relative]
        if isinstance(source, bytes):
            return hashlib.sha256(source).hexdigest()
        if isinstance(source, _StreamedFileDigest):
            return source.sha256
        return _audit_sha256(source)

    mismatches = [
        relative
        for relative in ordered_paths
        if digest_for(relative) != checksums[relative]
    ]
    if mismatches:
        raise ValueError(f"checksum mismatch: {mismatches[:5]}")


def _capture_regular_file(
    path: Path,
    *,
    retain_payload: bool,
    retained_byte_limit: int | None = None,
) -> bytes | _StreamedFileDigest:
    try:
        listed = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot inspect dataset file: {path}") from error
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open dataset file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            _audit_unsafe(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > _MAX_DATASET_FILE_BYTES
            or (listed.st_dev, listed.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError(f"dataset file changed before snapshot capture: {path}")
        if (
            retain_payload
            and retained_byte_limit is not None
            and opened.st_size > retained_byte_limit
        ):
            raise ValueError("identity payload byte budget exceeded")
        payload = bytearray() if retain_payload else None
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            byte_count += len(block)
            if (
                payload is not None
                and retained_byte_limit is not None
                and byte_count > retained_byte_limit
            ):
                raise ValueError("identity payload byte budget exceeded")
            digest.update(block)
            if payload is not None:
                payload.extend(block)
            if byte_count > _MAX_DATASET_FILE_BYTES:
                raise ValueError(f"dataset file exceeds resource byte limit: {path}")
        finished = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if byte_count != opened.st_size or any(
            getattr(opened, field) != getattr(finished, field)
            for field in stable_fields
        ):
            raise ValueError(f"dataset file changed during snapshot capture: {path}")
        if payload is not None:
            return bytes(payload)
        return _StreamedFileDigest(sha256=digest.hexdigest())
    finally:
        os.close(descriptor)


def _capture_regular_file_payloads(
    root: Path,
) -> dict[str, bytes | _StreamedFileDigest]:
    files = _regular_files(root)
    if not _METADATA_FILES.issubset(files):
        missing = sorted(_METADATA_FILES - files.keys())
        raise ValueError(f"missing dataset metadata files: {missing}")

    if _MAX_IDENTITY_PAYLOAD_FILES < 2:
        raise ValueError("identity payload file count exceeds limit")
    snapshot: dict[str, bytes | _StreamedFileDigest] = {}
    retained_bytes = 0

    def retain(relative: str) -> bytes:
        nonlocal retained_bytes
        remaining = _MAX_IDENTITY_PAYLOAD_BYTES - retained_bytes
        if remaining < 0:
            raise ValueError("identity payload byte budget exceeded")
        captured = _capture_regular_file(
            files[relative],
            retain_payload=True,
            retained_byte_limit=remaining,
        )
        if not isinstance(captured, bytes):
            raise TypeError("retained dataset payload is not bytes")
        retained_bytes += len(captured)
        snapshot[relative] = captured
        return captured

    checksum_bytes = retain("checksums.sha256")
    checksums, _ = _checksum_entries(
        checksum_bytes,
        set(files) - {"checksums.sha256"},
    )
    manifest_bytes = retain("manifest.json")
    if hashlib.sha256(manifest_bytes).hexdigest() != checksums["manifest.json"]:
        raise ValueError("checksum mismatch: ['manifest.json']")
    manifest = _read_manifest(manifest_bytes)
    if manifest["schema_version"] != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "authenticated dataset identity requires manifest schema version 4"
        )
    generation = manifest["generation"]
    if (
        generation["provenance_path"] != "provenance/generation.json"
        or generation["attempts_path"] != "provenance/attempts.jsonl"
    ):
        raise ValueError(
            "generation provenance paths are not the fixed canonical paths"
        )
    pair_bytes = retain("pairs.jsonl")
    if hashlib.sha256(pair_bytes).hexdigest() != checksums["pairs.jsonl"]:
        raise ValueError("checksum mismatch: ['pairs.jsonl']")
    pair_records = _read_records(pair_bytes, PairRecord, "pair")
    pair_after_paths = {record.scene_after_path for record in pair_records}
    if (
        6 + len(generation["source_scene_paths"]) + len(pair_after_paths)
        > _MAX_IDENTITY_PAYLOAD_FILES
    ):
        raise ValueError("identity payload file count exceeds limit")
    identity_paths = {
        "checksums.sha256",
        "failures.jsonl",
        "manifest.json",
        "pairs.jsonl",
        generation["provenance_path"],
        generation["attempts_path"],
        *generation["source_scene_paths"],
        *pair_after_paths,
    }
    if len(identity_paths) > _MAX_IDENTITY_PAYLOAD_FILES:
        raise ValueError("identity payload file count exceeds limit")
    missing_identity = sorted(identity_paths - files.keys())
    if missing_identity:
        raise ValueError(
            f"generation attestation references missing file: {missing_identity[0]}"
        )

    for relative in sorted(set(files) - snapshot.keys()):
        if relative in identity_paths:
            retain(relative)
        else:
            snapshot[relative] = _capture_regular_file(
                files[relative],
                retain_payload=False,
            )
    return snapshot


def _audit_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return canonical_json_bytes(value, pretty=pretty)


def _read_records(
    path: Path | bytes,
    model: type[_RecordT],
    label: str,
) -> tuple[_RecordT, ...]:
    payload = path if isinstance(path, bytes) else path.read_bytes()
    filename = f"{label}.jsonl" if isinstance(path, bytes) else path.name
    if b"\r" in payload or (payload and not payload.endswith(b"\n")):
        raise ValueError(f"{filename} must use canonical LF JSONL")
    records: list[_RecordT] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank {label} record at line {line_number}")
        try:
            record = model.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"invalid {label} record at line {line_number}: {error}"
            ) from error
        canonical = _audit_json_bytes(record.model_dump(mode="python")).rstrip(b"\n")
        if line != canonical:
            raise ValueError(
                f"invalid {label} record at line {line_number}: "
                "record is not canonical JSON"
            )
        records.append(record)
    return tuple(records)


def _read_manifest(path: Path | bytes) -> dict[str, Any]:
    payload = path if isinstance(path, bytes) else path.read_bytes()
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError("manifest.json must use canonical LF JSON")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json is invalid") from error
    if not isinstance(manifest, dict) or payload != _audit_json_bytes(
        manifest,
        pretty=True,
    ):
        raise ValueError("manifest.json schema or canonical encoding is invalid")
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        expected_keys = _MANIFEST_KEYS
    elif schema_version == LEGACY_ATTESTED_MANIFEST_SCHEMA_VERSION:
        expected_keys = _ATTESTED_MANIFEST_KEYS
    elif schema_version == DATASET_MANIFEST_SCHEMA_VERSION:
        expected_keys = _ATTESTED_MANIFEST_KEYS | {
            "run_profile",
            "evidence_eligible",
        }
    else:
        raise ValueError("unsupported manifest schema_version")
    if frozenset(manifest) != expected_keys:
        raise ValueError("manifest.json schema or canonical encoding is invalid")
    for key in ("accepted_pairs", "failures", "required_artifacts", "schema_version"):
        if type(manifest[key]) is not int or manifest[key] < 0:
            raise ValueError(f"manifest {key} must be a non-negative integer")
    splits = manifest["splits"]
    if (
        not isinstance(splits, dict)
        or frozenset(splits) != {"train", "dev", "test"}
        or any(type(value) is not int or value < 0 for value in splits.values())
    ):
        raise ValueError("manifest splits schema is invalid")
    if manifest["schema_version"] in ATTESTED_MANIFEST_SCHEMA_VERSIONS:
        generation = manifest["generation"]
        if (
            not isinstance(generation, dict)
            or frozenset(generation) != _GENERATION_MANIFEST_KEYS
            or type(generation["attempted_requests"]) is not int
            or generation["attempted_requests"] < 0
            or not isinstance(generation["source_scene_paths"], list)
            or any(
                not isinstance(value, str)
                for value in (
                    generation["attempts_path"],
                    generation["provenance_path"],
                    *generation["source_scene_paths"],
                )
            )
        ):
            raise ValueError("manifest generation attestation schema is invalid")
        validate_generation_budget(
            generation["requested_pairs"],
            generation["attempt_limit"],
            attempted_requests=generation["attempted_requests"],
        )
        paths = (
            generation["attempts_path"],
            generation["provenance_path"],
            *generation["source_scene_paths"],
        )
        for relative in paths:
            _audit_validate_relative_path(relative)
            if relative.split("/", 1)[0] != "provenance":
                raise ValueError(
                    "generation attestation paths must be under provenance/"
                )
        if len(set(paths)) != len(paths):
            raise ValueError("generation attestation paths must be unique")
    if manifest["schema_version"] == DATASET_MANIFEST_SCHEMA_VERSION:
        profile_from_manifest(manifest)
    return manifest


def _relation_diff_errors(record: PairRecord) -> list[str]:
    required = {
        f"-{record.subject_id}:{record.relation_before.value}:{record.reference_id}",
        f"+{record.subject_id}:{record.relation_after.value}:{record.reference_id}",
        (
            f"-{record.reference_id}:{record.relation_before.converse.value}:"
            f"{record.subject_id}"
        ),
        (
            f"+{record.reference_id}:{record.relation_after.converse.value}:"
            f"{record.subject_id}"
        ),
    }
    errors: list[str] = []
    if tuple(sorted(record.relation_diff)) != record.relation_diff:
        errors.append("relation_diff must be sorted")
    if len(set(record.relation_diff)) != len(record.relation_diff):
        errors.append("relation_diff contains duplicates")
    missing = sorted(required - set(record.relation_diff))
    if missing:
        errors.append(f"relation_diff missing target changes {missing}")
    for item in record.relation_diff:
        if not item.startswith(("+", "-")):
            errors.append(f"relation_diff has invalid operation {item!r}")
            continue
        parts = item[1:].split(":")
        if len(parts) != 3:
            errors.append(f"relation_diff has invalid entry {item!r}")
            continue
        try:
            Relation(parts[1])
        except ValueError:
            errors.append(f"relation_diff has invalid relation {item!r}")
    return errors


def _expected_request_id(record: PairRecord) -> str:
    identity = _audit_json_bytes(
        {
            "camera_id": record.camera_id,
            "relation_after": record.relation_after,
            "relation_before": record.relation_before,
            "reference_id": record.reference_id,
            "scene_id": record.scene_id,
            "seed": _DATASET_SEED,
            "subject_id": record.subject_id,
        }
    )
    return str(
        uuid.uuid5(
            _REQUEST_NAMESPACE,
            identity.decode("utf-8").rstrip("\n"),
        )
    )


def _validate_camera_resources(scene: Scene, camera_id: str) -> Any:
    camera = scene.camera_by_id(camera_id)
    pixels = camera.width * camera.height
    if (
        camera.width > _MAX_IMAGE_DIMENSION
        or camera.height > _MAX_IMAGE_DIMENSION
        or pixels > _MAX_IMAGE_PIXELS
    ):
        raise ValueError(
            "camera dimensions exceed the bounded artifact resource contract"
        )
    return camera


def _validate_image(
    path: Path,
    scene: Scene,
    camera_id: str,
    label: str,
    backend: str,
) -> None:
    if path.suffix.lower() != ".png":
        raise ValueError(f"{label} image artifact must use PNG encoding")
    camera = _validate_camera_resources(scene, camera_id)
    if path.stat().st_size > _MAX_IMAGE_BYTES:
        raise ValueError(f"{label} image artifact exceeds resource byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (camera.width, camera.height)
                ):
                    raise ValueError(
                        f"{label} image artifact schema does not match its camera"
                    )
                image.load()
                actual = image.copy()
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(f"{label} image artifact is invalid") from error
    canonical = io.BytesIO()
    actual.save(canonical, format="PNG")
    if canonical.getvalue() != path.read_bytes():
        raise ValueError(f"{label} image artifact is not canonical PNG")
    if backend in {"json", "legacy-json"}:
        instance = "instance" in label.lower()
        expected = Image.new(
            "RGB",
            (camera.width, camera.height),
            "black" if instance else "white",
        )
        draw = ImageDraw.Draw(expected)
        for index, obj in enumerate(scene.objects, start=1):
            view = obj.views.get(camera_id)
            if view is None:
                continue
            box = (
                view.bbox.xmin,
                view.bbox.ymin,
                view.bbox.xmax,
                view.bbox.ymax,
            )
            if instance:
                draw.rectangle(
                    box,
                    fill=(
                        index % 255,
                        (index * 17) % 255,
                        (index * 31) % 255,
                    ),
                )
            else:
                draw.rectangle(box, outline=(40, 90, 180), width=3)
        if actual.tobytes() != expected.tobytes():
            raise ValueError(f"{label} image artifact semantics do not match its scene")


def _validate_depth(
    path: Path,
    scene: Scene,
    camera_id: str,
    label: str,
    backend: str,
) -> None:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{label} depth artifact must use NPY encoding")
    camera = _validate_camera_resources(scene, camera_id)
    if path.stat().st_size > _MAX_DEPTH_BYTES:
        raise ValueError(f"{label} depth artifact exceeds resource byte limit")
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise ValueError("depth artifact must use canonical NPY v1")
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=10_000,
            )
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"{label} depth artifact is invalid") from error
    if (
        fortran_order
        or dtype != np.dtype(np.float32)
        or shape != (camera.height, camera.width)
    ):
        raise ValueError(f"{label} depth artifact schema does not match its camera")
    payload = path.read_bytes()
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"{label} depth artifact is invalid") from error
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.float32
        or array.shape != (camera.height, camera.width)
    ):
        raise ValueError(f"{label} depth artifact schema does not match its camera")
    if backend in {"json", "legacy-json"} and not np.array_equal(
        array,
        np.full(array.shape, 2.0, dtype=np.float32),
    ):
        raise ValueError(f"{label} depth artifact semantics do not match its scene")
    canonical = io.BytesIO()
    np.save(canonical, array, allow_pickle=False)
    if canonical.getvalue() != payload:
        raise ValueError(f"{label} depth artifact is not canonical NPY")


def _validate_pointcloud(
    path: Path,
    scene: Scene,
    label: str,
    backend: str,
) -> None:
    if path.suffix.lower() != ".ply":
        raise ValueError(f"{label} pointcloud artifact must use PLY encoding")
    if path.stat().st_size > _MAX_POINTCLOUD_BYTES:
        raise ValueError(f"{label} pointcloud artifact exceeds resource byte limit")
    payload = path.read_bytes()
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} pointcloud artifact must use canonical LF")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} pointcloud artifact must be ASCII") from error
    try:
        end_header = lines.index("end_header")
    except ValueError as error:
        raise ValueError(f"{label} pointcloud artifact has no PLY header") from error
    header = lines[: end_header + 1]
    vertex_lines = [line for line in header if line.startswith("element vertex ")]
    property_lines = [line for line in header if line.startswith("property ")]
    three_properties = [
        "property float x",
        "property float y",
        "property float z",
    ]
    six_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    valid_properties = (
        [three_properties] if backend in {"json", "legacy-json"} else [six_properties]
    )
    if (
        header[:2] != ["ply", "format ascii 1.0"]
        or len(vertex_lines) != 1
        or property_lines not in valid_properties
        or any(
            line.startswith("element ") for line in header if line not in vertex_lines
        )
    ):
        raise ValueError(f"{label} pointcloud artifact has invalid PLY schema")
    try:
        vertex_count = int(vertex_lines[0].removeprefix("element vertex "))
    except ValueError as error:
        raise ValueError(
            f"{label} pointcloud artifact has invalid vertex count"
        ) from error
    rows = lines[end_header + 1 :]
    if vertex_count < 0 or len(rows) != vertex_count:
        raise ValueError(f"{label} pointcloud artifact vertex count is inconsistent")
    columns = len(property_lines)
    parsed_coordinates: list[tuple[float, float, float]] = []
    for row in rows:
        parts = row.split(" ")
        if len(parts) != columns or any(not part for part in parts):
            raise ValueError(f"{label} pointcloud artifact row is not canonical")
        try:
            coordinates = [float(part) for part in parts[:3]]
            colors = [int(part) for part in parts[3:]]
        except ValueError as error:
            raise ValueError(f"{label} pointcloud artifact row is invalid") from error
        if not all(math.isfinite(value) for value in coordinates) or any(
            value < 0 or value > 255 for value in colors
        ):
            raise ValueError(f"{label} pointcloud artifact row is invalid")
        parsed_coordinates.append(tuple(coordinates))
    if columns == 3:
        expected_coordinates = [
            (obj.obb.center.x, obj.obb.center.y, obj.obb.center.z)
            for obj in scene.objects
        ]
        if parsed_coordinates != expected_coordinates:
            raise ValueError(
                f"{label} pointcloud artifact semantics do not match its scene"
            )
        if backend in {"json", "legacy-json"}:
            expected = (
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(expected_coordinates)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
                + "".join(f"{x} {y} {z}\n" for x, y, z in expected_coordinates)
            ).encode("ascii")
            if payload != expected:
                raise ValueError(
                    f"{label} pointcloud artifact is not canonical JSON-renderer PLY"
                )


def _validate_json_artifact(path: Path, expected: Any, label: str) -> None:
    expected_payload = _audit_json_bytes(expected, pretty=True)
    if path.read_bytes() != expected_payload:
        raise ValueError(f"{label} artifact is not canonical or semantically valid")


def _read_canonical_model(
    path: Path | bytes,
    model: type[_RecordT],
    label: str,
) -> _RecordT:
    payload = path if isinstance(path, bytes) else path.read_bytes()
    try:
        value = model.model_validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"{label} schema is invalid: {error}") from error
    if payload != _audit_json_bytes(value.model_dump(mode="json"), pretty=True):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _retained_snapshot_payload(
    files: dict[str, bytes | _StreamedFileDigest],
    relative: str,
) -> bytes:
    payload = files[relative]
    if not isinstance(payload, bytes):
        raise TypeError(f"authenticated identity payload was not retained: {relative}")
    return payload


def _published_pair_after_scenes(
    pairs: tuple[PairRecord, ...],
    files: Mapping[str, bytes | Path | _StreamedFileDigest],
) -> dict[str, Scene]:
    scenes: dict[str, Scene] = {}
    for pair in pairs:
        captured = files.get(pair.scene_after_path)
        if captured is None:
            raise ValueError(
                f"attempt ledger pair references missing after scene: {pair.pair_id}"
            )
        if isinstance(captured, _StreamedFileDigest):
            raise TypeError(
                "authenticated pair after-scene payload was not retained: "
                f"{pair.scene_after_path}"
            )
        after = _read_canonical_model(
            captured,
            Scene,
            f"pair {pair.pair_id} after scene",
        )
        if (
            after.scene_id != pair.scene_id
            or after.source != pair.source
            or after.generation_seed != pair.seed
        ):
            raise ValueError(
                f"attempt ledger pair after scene identity is invalid: {pair.pair_id}"
            )
        scenes[pair.request_id] = after
    return scenes


def _validate_attempt_ledger(
    *,
    attempts: tuple[AttemptEvidence, ...],
    pairs: tuple[PairRecord, ...],
    failures: tuple[FailureRecord, ...],
    provenance: GenerationProvenance,
    source_scenes: tuple[Scene, ...],
    pair_after_scenes: Mapping[str, Scene],
) -> None:
    """Bind checksummed attempt evidence to its structural dataset outcomes.

    This deliberately performs no generator or simulator replay.  It only
    authenticates relationships already represented by immutable source,
    attempt, pair, and failure records.
    """
    indexes = tuple(item.attempt_index for item in attempts)
    if indexes != tuple(range(1, len(attempts) + 1)):
        raise ValueError("attempt ledger indexes are not a complete ordered prefix")
    request_ids = tuple(item.request_id for item in attempts)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("attempt ledger request IDs are not unique")
    if provenance.attempted_requests != len(attempts):
        raise ValueError("attempt ledger count differs from generation provenance")

    sources_by_id = {scene.scene_id: scene for scene in source_scenes}
    if len(sources_by_id) != len(source_scenes):
        raise ValueError("attempt ledger source scenes are not unique")
    pairs_by_request = {record.request_id: record for record in pairs}
    failures_by_request = {record.request_id: record for record in failures}
    if len(pairs_by_request) != len(pairs) or len(failures_by_request) != len(failures):
        raise ValueError("attempt ledger outcome request IDs are not unique")
    if set(pairs_by_request) & set(failures_by_request):
        raise ValueError("attempt ledger request has both pair and failure outcomes")
    if set(pairs_by_request) | set(failures_by_request) != set(request_ids):
        raise ValueError("pair/failure ledgers do not exactly cover attempt ledger")

    expected_pair_order: list[str] = []
    expected_failure_order: list[str] = []
    known_holdouts = frozenset(
        {"unseen_scene", "unseen_category", "unseen_combination"}
    )
    for attempt in attempts:
        source = sources_by_id.get(attempt.scene_id)
        if source is None:
            raise ValueError(
                "attempt ledger scene is absent from authenticated sources"
            )
        try:
            subject = source.object_by_id(attempt.spec.subject_id)
            reference = source.object_by_id(attempt.spec.reference_id)
            source.camera_by_id(attempt.spec.camera_id)
        except KeyError as error:
            raise ValueError(
                "attempt ledger spec is absent from its authenticated source scene"
            ) from error
        if attempt.spec.subject_id == attempt.spec.reference_id:
            raise ValueError("attempt ledger spec aliases subject and reference")
        if not attempt.holdout_tags.issubset(known_holdouts):
            raise ValueError("attempt ledger contains an unknown holdout tag")

        result = attempt.generator_result
        if attempt.outcome == "pair":
            pair = pairs_by_request.get(attempt.request_id)
            if pair is None or attempt.outcome_id != pair.pair_id:
                raise ValueError("attempt ledger pair outcome binding is invalid")
            if (
                pair.scene_id != attempt.scene_id
                or pair.subject_id != attempt.spec.subject_id
                or pair.reference_id != attempt.spec.reference_id
                or pair.camera_id != attempt.spec.camera_id
                or pair.relation_before is not attempt.spec.relation_before
                or pair.relation_after is not attempt.spec.relation_after
                or pair.holdout_tags != attempt.holdout_tags
                or pair.source != source.source
                or pair.subject_category != subject.category
                or pair.reference_category != reference.category
                or pair.generator != provenance.generator
            ):
                raise ValueError(
                    "attempt ledger pair source/request/scene/spec/holdout binding is invalid"
                )
            score = result.score
            after = pair_after_scenes.get(attempt.request_id)
            if after is None:
                raise ValueError(
                    "attempt ledger pair has no authenticated published after scene"
                )
            try:
                published_subject_position = after.object_by_id(
                    pair.subject_id
                ).position
            except KeyError as error:
                raise ValueError(
                    "attempt ledger pair after scene has no subject"
                ) from error
            expected_score = calculate_weighted_objective(
                pair.normalized_edit_distance,
                pair.leakage_score,
                pair.visibility_change,
                pair.inverse_safety_margin,
                translation_weight=1.0,
                relation_damage_weight=5.0,
                visibility_change_weight=2.0,
                inverse_safety_margin_weight=1.0,
            )
            if (
                result.status is not SolverStatus.SUCCESS
                or result.subject_position != published_subject_position
                or score is None
                or result.quality is not pair.quality
                or result.evaluated_candidates != pair.evaluated_candidates
                or result.reason is not None
                or score.normalized_translation != pair.normalized_edit_distance
                or score.leakage != pair.leakage_score
                or score.visibility_change != pair.visibility_change
                or score.inverse_safety_margin != pair.inverse_safety_margin
                or score.total != expected_score[4]
            ):
                raise ValueError(
                    "attempt ledger pair generator result binding is invalid"
                )
            expected_pair_order.append(attempt.request_id)
            continue

        failure = failures_by_request.get(attempt.request_id)
        if failure is None or attempt.outcome_id != failure.failure_id:
            raise ValueError("attempt ledger failure outcome binding is invalid")
        if (
            failure.scene_id != attempt.scene_id
            or failure.subject_id != attempt.spec.subject_id
            or failure.reference_id != attempt.spec.reference_id
            or failure.relation_before is not attempt.spec.relation_before
            or failure.relation_after is not attempt.spec.relation_after
            or failure.generator != provenance.generator
        ):
            raise ValueError(
                "attempt ledger failure source/request/scene/spec binding is invalid"
            )
        if (
            result.status is not failure.status
            or result.subject_position is not None
            or result.score is not None
            or result.quality is not QualityTier.REJECTED
            or result.evaluated_candidates != failure.evaluated_candidates
            or result.reason != failure.reason
        ):
            raise ValueError(
                "attempt ledger failure generator result binding is invalid"
            )
        expected_failure_order.append(attempt.request_id)

    if tuple(record.request_id for record in pairs) != tuple(expected_pair_order):
        raise ValueError("pair ledger is not an ordered subsequence of attempt ledger")
    if tuple(record.request_id for record in failures) != tuple(expected_failure_order):
        raise ValueError(
            "failure ledger is not an ordered subsequence of attempt ledger"
        )


def read_authenticated_dataset_identity(
    root: Path,
) -> AuthenticatedDatasetIdentity:
    """Read checksummed identity metadata without replaying live sources."""
    dataset_root = Path(root)
    files = _capture_regular_file_payloads(dataset_root)
    _validate_checksums(dataset_root, files)
    manifest = _read_manifest(_retained_snapshot_payload(files, "manifest.json"))
    if manifest["schema_version"] != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "authenticated dataset identity requires manifest schema version 4"
        )

    generation = manifest["generation"]
    if (
        generation["provenance_path"] != "provenance/generation.json"
        or generation["attempts_path"] != "provenance/attempts.jsonl"
    ):
        raise ValueError(
            "generation provenance paths are not the fixed canonical paths"
        )
    referenced_paths = (
        generation["provenance_path"],
        generation["attempts_path"],
        *generation["source_scene_paths"],
    )
    for relative in referenced_paths:
        if relative not in files:
            raise ValueError(
                f"generation attestation references missing file: {relative}"
            )

    provenance = _read_canonical_model(
        _retained_snapshot_payload(
            files,
            generation["provenance_path"],
        ),
        GenerationProvenance,
        "generation provenance",
    )
    attempts = _read_records(
        _retained_snapshot_payload(
            files,
            generation["attempts_path"],
        ),
        AttemptEvidence,
        "generation attempt",
    )
    pairs = _read_records(
        _retained_snapshot_payload(files, "pairs.jsonl"),
        PairRecord,
        "pair",
    )
    failures = _read_records(
        _retained_snapshot_payload(files, "failures.jsonl"),
        FailureRecord,
        "failure",
    )
    if provenance.attempted_requests != len(attempts) or generation[
        "attempted_requests"
    ] != len(attempts):
        raise ValueError(
            "attempted request count differs from the complete attempt ledger"
        )
    if (
        generation["requested_pairs"] != provenance.requested_pairs
        or generation["attempt_limit"] != provenance.attempt_limit
    ):
        raise ValueError(
            "manifest generation budget differs from generation provenance"
        )

    source_paths = tuple(item.path for item in provenance.source_scenes)
    if source_paths != tuple(generation["source_scene_paths"]):
        raise ValueError(
            "manifest source scene paths differ from generation provenance"
        )
    source_ids = tuple(item.scene_id for item in provenance.source_scenes)
    if source_ids != tuple(sorted(source_ids)):
        raise ValueError(
            "generation provenance sources are not in canonical source order"
        )
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("generation provenance contains duplicate source scenes")

    scenes: list[Scene] = []
    for source in provenance.source_scenes:
        expected_path = f"provenance/scenes/{source.sha256}.json"
        if source.path != expected_path:
            raise ValueError(
                "generation provenance source path is not content-addressed"
            )
        payload = _retained_snapshot_payload(files, source.path)
        if hashlib.sha256(payload).hexdigest() != source.sha256:
            raise ValueError("source scene attestation digest mismatch")
        scene = _read_canonical_model(
            payload,
            Scene,
            "source scene attestation",
        )
        if scene.scene_id != source.scene_id:
            raise ValueError("source scene attestation has the wrong identity")
        scenes.append(scene)
    computed_source_digest = source_corpus_digest(scenes)
    if provenance.source_corpus_sha256 != computed_source_digest:
        raise ValueError("source corpus digest differs from authenticated source tree")
    if manifest["accepted_pairs"] != len(pairs) or manifest["failures"] != len(
        failures
    ):
        raise ValueError("manifest outcome counts differ from attempt ledger outcomes")
    _validate_attempt_ledger(
        attempts=attempts,
        pairs=pairs,
        failures=failures,
        provenance=provenance,
        source_scenes=tuple(scenes),
        pair_after_scenes=_published_pair_after_scenes(pairs, files),
    )

    return AuthenticatedDatasetIdentity(
        accepted_pairs=manifest["accepted_pairs"],
        run_profile=manifest["run_profile"],
        evidence_eligible=manifest["evidence_eligible"],
        provenance=provenance,
        attempts=attempts,
        scenes=tuple(scenes),
    )


def _read_dataset(
    root: Path,
    *,
    expected_source_digest: str | None = None,
    descriptor_capability: bool = False,
) -> _Dataset:
    dataset_root = Path(root)
    files = _regular_files(
        dataset_root,
        descriptor_capability=descriptor_capability,
    )
    _validate_checksums(dataset_root, files)
    manifest = _read_manifest(files["manifest.json"])
    pairs = _read_records(files["pairs.jsonl"], PairRecord, "pair")
    failures = _read_records(files["failures.jsonl"], FailureRecord, "failure")
    provenance: GenerationProvenance | None = None
    attempts: tuple[AttemptEvidence, ...] = ()
    attempts_authenticated = False
    attestation_files: set[str] = set()
    if manifest["schema_version"] in ATTESTED_MANIFEST_SCHEMA_VERSIONS:
        generation = manifest["generation"]
        if (
            generation["provenance_path"] != "provenance/generation.json"
            or generation["attempts_path"] != "provenance/attempts.jsonl"
        ):
            raise ValueError(
                "generation provenance paths are not the fixed canonical paths"
            )
        for relative in (
            generation["provenance_path"],
            generation["attempts_path"],
            *generation["source_scene_paths"],
        ):
            if relative not in files:
                raise ValueError(
                    f"generation attestation references missing file: {relative}"
                )
            attestation_files.add(relative)
        provenance = _read_canonical_model(
            files[generation["provenance_path"]],
            GenerationProvenance,
            "generation provenance",
        )
        attempts = _read_records(
            files[generation["attempts_path"]],
            AttemptEvidence,
            "generation attempt",
        )
        source_paths = [item.path for item in provenance.source_scenes]
        if source_paths != generation["source_scene_paths"]:
            raise ValueError(
                "manifest source scene paths differ from generation provenance"
            )
        source_ids = [item.scene_id for item in provenance.source_scenes]
        if source_ids != sorted(source_ids):
            raise ValueError(
                "generation provenance sources are not in canonical source order"
            )
        if any(
            source.path != f"provenance/scenes/{source.sha256}.json"
            for source in provenance.source_scenes
        ):
            raise ValueError(
                "generation provenance source path is not content-addressed"
            )
        if provenance.attempted_requests != len(attempts) or generation[
            "attempted_requests"
        ] != len(attempts):
            raise ValueError(
                "attempted request count differs from the complete attempt ledger"
            )
        if (
            generation["requested_pairs"] != provenance.requested_pairs
            or generation["attempt_limit"] != provenance.attempt_limit
        ):
            raise ValueError(
                "manifest generation budget differs from generation provenance"
            )
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("generation provenance contains duplicate source scenes")
        source_scene_values: list[Scene] = []
        for source in provenance.source_scenes:
            payload = files[source.path].read_bytes()
            if hashlib.sha256(payload).hexdigest() != source.sha256:
                raise ValueError("source scene attestation digest mismatch")
            try:
                scene = Scene.model_validate_json(payload)
            except (ValidationError, ValueError) as error:
                raise ValueError("source scene attestation is invalid") from error
            if scene.scene_id != source.scene_id or payload != _audit_json_bytes(
                scene.model_dump(mode="json"), pretty=True
            ):
                raise ValueError(
                    "source scene attestation is not canonical or has "
                    "the wrong identity"
                )
            source_scene_values.append(scene)
        computed_source_digest = source_corpus_digest(source_scene_values)
        if provenance.source_corpus_sha256 != computed_source_digest:
            raise ValueError(
                "source corpus digest differs from authenticated source tree"
            )
        if (
            expected_source_digest is not None
            and expected_source_digest != computed_source_digest
        ):
            raise ValueError(
                "dataset source corpus digest differs from caller trusted source pin"
            )
        _validate_attempt_ledger(
            attempts=attempts,
            pairs=pairs,
            failures=failures,
            provenance=provenance,
            source_scenes=tuple(source_scene_values),
            pair_after_scenes=_published_pair_after_scenes(pairs, files),
        )
        attempts_authenticated = True

    pair_ids = [record.pair_id for record in pairs]
    duplicate_pairs = sorted(
        value for value, count in Counter(pair_ids).items() if count > 1
    )
    if duplicate_pairs:
        raise ValueError(f"duplicate pair_id: {duplicate_pairs[:5]}")
    failure_ids = [record.failure_id for record in failures]
    duplicate_failures = sorted(
        value for value, count in Counter(failure_ids).items() if count > 1
    )
    if duplicate_failures:
        raise ValueError(f"duplicate failure_id: {duplicate_failures[:5]}")
    request_ids = [record.request_id for record in pairs] + [
        record.request_id for record in failures
    ]
    duplicate_requests = sorted(
        value for value, count in Counter(request_ids).items() if count > 1
    )
    if duplicate_requests:
        raise ValueError(f"duplicate request_id: {duplicate_requests[:5]}")
    if expected_source_digest is not None and provenance is None:
        raise ValueError(
            "expected source pin requires manifest-v3 official replay "
            "attestation; legacy datasets cannot claim a trusted pin"
        )
    unsupported_failure_generators = sorted(
        {
            record.generator
            for record in failures
            if record.generator not in _GENERATOR_NAMES
        }
    )
    if unsupported_failure_generators:
        raise ValueError(
            "failure records contain unsupported generator provenance: "
            f"{unsupported_failure_generators}"
        )
    for record in failures:
        expected_failure_id = str(
            uuid.uuid5(
                _FAILURE_NAMESPACE,
                (f"{record.request_id}:{record.generator}:{_GENERATOR_VERSION}"),
            )
        )
        if (
            record.failure_id != expected_failure_id
            or record.generator_version != _GENERATOR_VERSION
            or record.seed != _DATASET_SEED
        ):
            raise ValueError(
                f"failure {record.failure_id} provenance identity is invalid"
            )
    if failures and (
        len({record.generator for record in failures}) != 1
        or len({record.generator_version for record in failures}) != 1
        or len({record.seed for record in failures}) != 1
    ):
        raise ValueError("failure ledger has heterogeneous generator provenance")

    expected_manifest = {
        "accepted_pairs": len(pairs),
        "failures": len(failures),
        "required_artifacts": len(pairs) * len(PairRecord.artifact_path_fields()),
        "schema_version": manifest["schema_version"],
        "splits": {
            split: sum(record.split == split for record in pairs)
            for split in ("train", "dev", "test")
        },
    }
    for key in ("accepted_pairs", "failures", "required_artifacts", "splits"):
        if manifest[key] != expected_manifest[key]:
            raise ValueError(
                f"manifest {key}={manifest[key]!r} does not match "
                f"recomputed value={expected_manifest[key]!r}"
            )

    referenced: set[str] = set()
    for record in pairs:
        expected_split = assign_split(record.scene_id)
        if record.split != expected_split:
            raise ValueError(
                f"pair {record.pair_id} split assignment={record.split!r} "
                f"does not match recomputed scene split={expected_split!r}"
            )
        errors = _relation_diff_errors(record)
        if errors:
            raise ValueError(
                f"pair {record.pair_id} relation_diff is inconsistent: {errors}"
            )
        for field in PairRecord.artifact_path_fields():
            relative = getattr(record, field)
            if relative not in files:
                raise ValueError(
                    f"pair {record.pair_id} references missing artifact: {relative}"
                )
            if relative in referenced:
                raise ValueError(f"duplicate artifact reference: {relative}")
            referenced.add(relative)
    published_artifacts = set(files) - _METADATA_FILES
    expected_published = referenced | attestation_files
    if expected_published != published_artifacts:
        unreferenced = sorted(published_artifacts - expected_published)
        missing = sorted(expected_published - published_artifacts)
        raise ValueError(
            "manifest/files do not match pair artifact references; "
            f"unreferenced={unreferenced[:5]} missing={missing[:5]}"
        )
    dataset = _Dataset(
        dataset_root,
        pairs,
        failures,
        manifest,
        provenance,
        attempts,
        attempts_authenticated,
    )
    _verify_scenes(dataset)
    return dataset


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return float(value)


def _verify_scenes(dataset: _Dataset) -> None:
    verifier = Verifier()
    for record in dataset.pairs:
        try:
            before_payload = (dataset.root / record.scene_before_path).read_bytes()
            after_payload = (dataset.root / record.scene_after_path).read_bytes()
            before = Scene.model_validate_json(before_payload)
            after = Scene.model_validate_json(after_payload)
            if before_payload != _audit_json_bytes(
                before.model_dump(mode="json"),
                pretty=True,
            ):
                raise ValueError("before scene is not canonical JSON")
            if after_payload != _audit_json_bytes(
                after.model_dump(mode="json"),
                pretty=True,
            ):
                raise ValueError("after scene is not canonical JSON")
            if (
                before.scene_id != record.scene_id
                or after.scene_id != record.scene_id
                or before.source != record.source
                or after.source != record.source
                or before.generation_seed != record.seed
                or after.generation_seed != record.seed
            ):
                raise ValueError("scene identity does not match pair record")
            result = verifier.verify(
                before,
                after,
                InterventionSpec(
                    subject_id=record.subject_id,
                    reference_id=record.reference_id,
                    relation_before=record.relation_before,
                    relation_after=record.relation_after,
                    camera_id=record.camera_id,
                ),
            )
        except (OSError, ValidationError, ValueError, KeyError) as error:
            raise ValueError(
                f"scene re-verification could not execute for {record.pair_id}: {error}"
            ) from error
        if result.status is not SolverStatus.SUCCESS:
            raise ValueError(
                f"scene re-verification failed for {record.pair_id}: "
                f"status={result.status.value} quality={result.quality.value}"
            )
        if result.changed_relations != record.relation_diff:
            raise ValueError(
                f"pair {record.pair_id} relation_diff does not match "
                "recomputed scene graph changes"
            )
        subject = before.object_by_id(record.subject_id)
        reference = before.object_by_id(record.reference_id)
        expected_question = (
            f"What is the relation of the {subject.category} to the "
            f"{reference.category}? Answer with one label."
        )
        if (
            subject.category != record.subject_category
            or reference.category != record.reference_category
            or record.question != expected_question
        ):
            raise ValueError(
                f"pair {record.pair_id} semantic metadata does not match its scene"
            )
        if (
            record.generator not in _GENERATOR_NAMES
            or record.generator_version != _GENERATOR_VERSION
            or record.seed != _DATASET_SEED
        ):
            raise ValueError(f"pair {record.pair_id} generator provenance is invalid")
        expected_request_id = _expected_request_id(record)
        expected_pair_id = str(
            uuid.uuid5(
                _PAIR_NAMESPACE,
                (f"{expected_request_id}:{record.generator}:{_GENERATOR_VERSION}"),
            )
        )
        if (
            record.request_id != expected_request_id
            or record.pair_id != expected_pair_id
        ):
            raise ValueError(f"pair {record.pair_id} provenance identity is invalid")

        score = calculate_candidate_objective(
            before,
            after,
            InterventionSpec(
                subject_id=record.subject_id,
                reference_id=record.reference_id,
                relation_before=record.relation_before,
                relation_after=record.relation_after,
                camera_id=record.camera_id,
            ),
            result.leakage_count,
            verifier.engine,
            translation_weight=1.0,
            relation_damage_weight=5.0,
            visibility_change_weight=2.0,
            inverse_safety_margin_weight=1.0,
        )
        objective_metrics = {
            "normalized_edit_distance": score[0],
            "leakage_score": score[1],
            "visibility_change": score[2],
            "inverse_safety_margin": score[3],
        }
        mismatched_metrics = [
            name
            for name, expected in objective_metrics.items()
            if not math.isclose(
                getattr(record, name),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if mismatched_metrics:
            raise ValueError(
                f"pair {record.pair_id} objective metric mismatch: {mismatched_metrics}"
            )
        if result.quality is not record.quality or record.quality_flags != (
            result.quality.value,
        ):
            raise ValueError(
                f"pair {record.pair_id} quality does not match recomputed evidence"
            )

        spec = InterventionSpec(
            subject_id=record.subject_id,
            reference_id=record.reference_id,
            relation_before=record.relation_before,
            relation_after=record.relation_after,
            camera_id=record.camera_id,
        )
        _validate_json_artifact(
            dataset.root / record.relation_graph_before_path,
            verifier.engine.graph(before, record.camera_id),
            "before relation graph",
        )
        _validate_json_artifact(
            dataset.root / record.relation_graph_after_path,
            verifier.engine.graph(after, record.camera_id),
            "after relation graph",
        )
        _validate_json_artifact(
            dataset.root / record.topdown_path,
            topdown_payload(before, after, spec),
            "topdown",
        )
        _validate_image(
            dataset.root / record.rgb_before_path,
            before,
            record.camera_id,
            "before RGB",
            dataset.backend,
        )
        _validate_image(
            dataset.root / record.rgb_after_path,
            after,
            record.camera_id,
            "after RGB",
            dataset.backend,
        )
        _validate_image(
            dataset.root / record.instance_before_path,
            before,
            record.camera_id,
            "before instance",
            dataset.backend,
        )
        _validate_image(
            dataset.root / record.instance_after_path,
            after,
            record.camera_id,
            "after instance",
            dataset.backend,
        )
        _validate_depth(
            dataset.root / record.depth_before_path,
            before,
            record.camera_id,
            "before",
            dataset.backend,
        )
        _validate_depth(
            dataset.root / record.depth_after_path,
            after,
            record.camera_id,
            "after",
            dataset.backend,
        )
        _validate_pointcloud(
            dataset.root / record.pointcloud_before_path,
            before,
            "before",
            dataset.backend,
        )
        _validate_pointcloud(
            dataset.root / record.pointcloud_after_path,
            after,
            "after",
            dataset.backend,
        )


def audit_dataset(
    root: Path,
    minimum_pairs: int,
    minimum_pure_test: int = 0,
    verify_scenes: bool = False,
    *,
    maximum_mean_leakage: float = 1.0,
    maximum_mean_edit_distance: float = 1.0,
    minimum_pure_rate: float = 0.0,
    minimum_solvable_coverage: float = 0.0,
    expected_source_digest: str | None = None,
    root_descriptor: int | None = None,
    _descriptor_capability: bool = False,
) -> dict[str, Any]:
    """Validate one immutable dataset and enforce configured quality gates."""
    if root_descriptor is not None:
        if type(root_descriptor) is not int or root_descriptor < 0:
            raise ValueError("root_descriptor must be an open directory fd")
        try:
            duplicate = os.dup(root_descriptor)
        except OSError as error:
            raise ValueError("cannot retain dataset root descriptor") from error
        try:
            descriptor_result = os.fstat(duplicate)
        except OSError as error:
            os.close(duplicate)
            raise ValueError("cannot inspect dataset root descriptor") from error
        if not stat.S_ISDIR(descriptor_result.st_mode):
            os.close(duplicate)
            raise ValueError("dataset root descriptor is not a directory")
        descriptor_path = Path("/proc/self/fd") / str(duplicate)
        if not descriptor_path.is_dir():
            os.close(duplicate)
            raise ValueError("descriptor-rooted audit requires Linux procfs")
        try:
            return audit_dataset(
                descriptor_path,
                minimum_pairs,
                minimum_pure_test,
                verify_scenes,
                maximum_mean_leakage=maximum_mean_leakage,
                maximum_mean_edit_distance=maximum_mean_edit_distance,
                minimum_pure_rate=minimum_pure_rate,
                minimum_solvable_coverage=minimum_solvable_coverage,
                expected_source_digest=expected_source_digest,
                _descriptor_capability=True,
            )
        finally:
            os.close(duplicate)
    if _descriptor_capability and not str(root).startswith("/proc/self/fd/"):
        raise ValueError("invalid descriptor-rooted audit capability")
    minimum_pairs = _validate_integer("minimum_pairs", minimum_pairs, minimum=0)
    minimum_pure_test = _validate_integer(
        "minimum_pure_test",
        minimum_pure_test,
        minimum=0,
    )
    if type(verify_scenes) is not bool:
        raise ValueError("verify_scenes must be a boolean")
    maximum_mean_leakage = _validate_rate(
        "maximum_mean_leakage",
        maximum_mean_leakage,
    )
    maximum_mean_edit_distance = _validate_rate(
        "maximum_mean_edit_distance",
        maximum_mean_edit_distance,
    )
    minimum_pure_rate = _validate_rate("minimum_pure_rate", minimum_pure_rate)
    minimum_solvable_coverage = _validate_rate(
        "minimum_solvable_coverage",
        minimum_solvable_coverage,
    )

    if expected_source_digest is not None and (
        type(expected_source_digest) is not str
        or len(expected_source_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_source_digest
        )
    ):
        raise ValueError("expected_source_digest must be lowercase SHA-256")
    dataset = _read_dataset(
        root,
        expected_source_digest=expected_source_digest,
        descriptor_capability=_descriptor_capability,
    )
    records = dataset.pairs
    failures = dataset.failures
    empty_smoke_baseline = (
        not records
        and minimum_pairs == 0
        and dataset.manifest.get("run_profile") == "smoke"
        and dataset.manifest.get("evidence_eligible") is False
        and dataset.provenance is not None
        and dataset.provenance.generator in {"random", "target-only"}
        and dataset.provenance.attempt_limit is not None
        and dataset.provenance.attempt_limit > 0
        and len(dataset.attempts) == dataset.provenance.attempt_limit
    )
    if not records and not empty_smoke_baseline:
        raise ValueError("accepted_pairs=0; empty datasets cannot pass audit")
    if len(records) < minimum_pairs:
        raise ValueError(f"accepted_pairs={len(records)} below minimum={minimum_pairs}")

    pure_test = tuple(
        record
        for record in records
        if record.split == "test" and record.quality is QualityTier.PURE
    )
    if len(pure_test) < minimum_pure_test:
        raise ValueError(
            f"pure_test_pairs={len(pure_test)} below minimum={minimum_pure_test}"
        )
    if minimum_pure_test:
        for tag in ("unseen_scene", "unseen_category", "unseen_combination"):
            if not any(tag in record.holdout_tags for record in pure_test):
                raise ValueError(f"PURE test set has no {tag} pairs")
        pure_relations = {record.relation_after for record in pure_test}
        missing = sorted(relation.value for relation in set(Relation) - pure_relations)
        if missing:
            raise ValueError(f"PURE test set missing target relations: {missing}")
    if minimum_pairs >= 100:
        observed = {record.relation_after for record in records}
        missing = sorted(relation.value for relation in set(Relation) - observed)
        if missing:
            raise ValueError(f"dataset missing target relations: {missing}")

    if dataset.provenance is None and minimum_solvable_coverage > 0.0:
        raise ValueError(
            "solvable coverage gate requires an authenticated attempt ledger"
        )
    total_requests = (
        len(dataset.attempts)
        if dataset.provenance is not None
        else len(records) + len(failures)
    )
    pure_rate = (
        mean(record.quality is QualityTier.PURE for record in records)
        if records
        else 0.0
    )
    mean_leakage = mean(record.leakage_score for record in records) if records else 0.0
    mean_edit_distance = (
        mean(record.normalized_edit_distance for record in records) if records else 0.0
    )
    mean_evaluated = (
        mean(record.evaluated_candidates for record in records) if records else 0.0
    )
    solvable_coverage = len(records) / total_requests if total_requests else 0.0
    gates = (
        (
            "mean_leakage",
            mean_leakage,
            maximum_mean_leakage,
            "above maximum",
            lambda actual, threshold: actual > threshold,
        ),
        (
            "mean_edit_distance",
            mean_edit_distance,
            maximum_mean_edit_distance,
            "above maximum",
            lambda actual, threshold: actual > threshold,
        ),
        (
            "pure_rate",
            pure_rate,
            minimum_pure_rate,
            "below minimum",
            lambda actual, threshold: actual < threshold,
        ),
        (
            "solvable_coverage",
            solvable_coverage,
            minimum_solvable_coverage,
            "below minimum",
            lambda actual, threshold: actual < threshold,
        ),
    )
    for name, actual, threshold, detail, failed in gates:
        if failed(actual, threshold):
            raise ValueError(f"{name}={actual:.6f} {detail}={threshold:.6f}")

    report: dict[str, Any] = {
        "accepted_pairs": len(records),
        "attempts_authenticated": dataset.attempts_authenticated,
        "attempt_limit": (
            dataset.provenance.attempt_limit if dataset.provenance is not None else None
        ),
        "attempted_requests": total_requests,
        "requested_pairs": (
            dataset.provenance.requested_pairs
            if dataset.provenance is not None
            else None
        ),
        "source_corpus_sha256": (
            dataset.provenance.source_corpus_sha256
            if dataset.provenance is not None
            else None
        ),
        "failed_requests": len(failures),
        "solvable_coverage": solvable_coverage,
        "pure_test_pairs": len(pure_test),
        "pure_rate": pure_rate,
        "mean_leakage": mean_leakage,
        "mean_edit_distance": mean_edit_distance,
        "mean_evaluated_candidates": mean_evaluated,
        "failure_statuses": {
            status.value: sum(failure.status is status for failure in failures)
            for status in SolverStatus
            if status is not SolverStatus.SUCCESS
        },
        "by_generator": {},
    }
    for generator in sorted({record.generator for record in records}):
        subset = tuple(record for record in records if record.generator == generator)
        report["by_generator"][generator] = {
            "count": len(subset),
            "mean_leakage": mean(record.leakage_score for record in subset),
            "pure_rate": mean(record.quality is QualityTier.PURE for record in subset),
        }
    return report


def compare_generators(
    spatial_root: Path,
    random_root: Path,
    minimum_matched: int,
    minimum_reduction: float,
    *,
    expected_source_digest: str,
) -> dict[str, float | int | str]:
    """Enforce the matched-request Gate B leakage comparison."""
    minimum_matched = _validate_integer(
        "minimum_matched",
        minimum_matched,
        minimum=1,
    )
    minimum_reduction = _validate_rate(
        "minimum_reduction",
        minimum_reduction,
    )
    if (
        type(expected_source_digest) is not str
        or len(expected_source_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_source_digest
        )
    ):
        raise ValueError("expected_source_digest must be lowercase SHA-256")
    spatial_dataset = _read_dataset(
        spatial_root,
        expected_source_digest=expected_source_digest,
    )
    random_dataset = _read_dataset(
        random_root,
        expected_source_digest=expected_source_digest,
    )
    spatial_profile = profile_from_manifest(spatial_dataset.manifest)
    random_profile = profile_from_manifest(random_dataset.manifest)
    if (
        spatial_profile is None
        or random_profile is None
        or spatial_profile.evidence_eligible is not True
        or random_profile.evidence_eligible is not True
    ):
        raise ValueError("Gate B inputs are not evidence eligible")
    if spatial_dataset.provenance is None or random_dataset.provenance is None:
        raise ValueError(
            "Gate B rejects legacy datasets without official replay attestation"
        )
    spatial_provenance = spatial_dataset.provenance
    random_provenance = random_dataset.provenance
    if (
        spatial_provenance.attempt_limit is None
        or random_provenance.attempt_limit is None
    ):
        raise ValueError(
            "Gate B requires each dataset to use a fixed authenticated attempt prefix"
        )
    if spatial_provenance.attempt_limit != random_provenance.attempt_limit:
        raise ValueError(
            "Gate B datasets must use the same authenticated attempt_limit"
        )
    attempt_limit = spatial_provenance.attempt_limit
    if spatial_provenance.adapter_backend != random_provenance.adapter_backend:
        raise ValueError(
            "Gate B datasets must use the same authenticated adapter backend"
        )
    if (
        spatial_provenance.adapter_implementation
        != random_provenance.adapter_implementation
        or spatial_provenance.adapter_config != random_provenance.adapter_config
    ):
        raise ValueError("Gate B datasets must use identical canonical adapter config")
    spatial_sources = [
        (source.scene_id, source.sha256) for source in spatial_provenance.source_scenes
    ]
    random_sources = [
        (source.scene_id, source.sha256) for source in random_provenance.source_scenes
    ]
    if spatial_sources != random_sources:
        raise ValueError("Gate B datasets use different authenticated inputs")
    spatial_prefix = tuple(attempt.request_id for attempt in spatial_dataset.attempts)
    random_prefix = tuple(attempt.request_id for attempt in random_dataset.attempts)
    if spatial_prefix != random_prefix:
        raise ValueError(
            "Gate B datasets must use an identical authenticated request-ID prefix"
        )
    for label, dataset, expected in (
        ("spatial", spatial_dataset, "spatialcf"),
        ("random", random_dataset, "random"),
    ):
        if dataset.provenance is None or dataset.provenance.generator != expected:
            actual = (
                dataset.provenance.generator if dataset.provenance is not None else None
            )
            raise ValueError(
                f"{label} dataset expected generator={expected!r}; "
                f"found provenance={actual!r}"
            )
        unexpected = sorted(
            {
                record.generator
                for record in dataset.pairs
                if record.generator != expected
            }
        )
        if unexpected:
            raise ValueError(
                f"{label} dataset expected generator={expected!r}; found={unexpected}"
            )

    spatial = {record.request_id: record for record in spatial_dataset.pairs}
    random = {record.request_id: record for record in random_dataset.pairs}
    request_ids = [
        request_id
        for request_id in spatial_prefix
        if request_id in spatial and request_id in random
    ]
    spatial_only = [
        request_id
        for request_id in spatial_prefix
        if request_id in spatial and request_id not in random
    ]
    random_only = [
        request_id
        for request_id in spatial_prefix
        if request_id in random and request_id not in spatial
    ]
    both_failed = [
        request_id
        for request_id in spatial_prefix
        if request_id not in spatial and request_id not in random
    ]
    if len(request_ids) < minimum_matched:
        raise ValueError(
            f"matched_requests={len(request_ids)} below minimum={minimum_matched}"
        )

    match_fields = (
        "request_id",
        "scene_id",
        "split",
        "holdout_tags",
        "source",
        "seed",
        "subject_id",
        "subject_category",
        "reference_id",
        "reference_category",
        "camera_id",
        "relation_before",
        "relation_after",
        "question",
        "answer_before",
        "answer_after",
    )
    for request_id in request_ids:
        mismatches = [
            field
            for field in match_fields
            if getattr(spatial[request_id], field) != getattr(random[request_id], field)
        ]
        spatial_before = (
            spatial_dataset.root / spatial[request_id].scene_before_path
        ).read_bytes()
        random_before = (
            random_dataset.root / random[request_id].scene_before_path
        ).read_bytes()
        if spatial_before != random_before:
            mismatches.append("scene_before_evidence")
        if mismatches:
            raise ValueError(f"request payload mismatch for {request_id}: {mismatches}")

    spatial_mean = mean(spatial[request_id].leakage_score for request_id in request_ids)
    random_mean = mean(random[request_id].leakage_score for request_id in request_ids)
    if random_mean <= 0.0:
        raise ValueError(
            "random matched mean leakage must be a positive reduction denominator"
        )
    reduction = (random_mean - spatial_mean) / random_mean
    if reduction < minimum_reduction:
        raise ValueError(
            f"leakage_reduction={reduction:.6f} below minimum={minimum_reduction:.6f}"
        )
    return {
        "attempt_limit": attempt_limit,
        "both_failed_requests": len(both_failed),
        "matched_coverage": len(request_ids) / attempt_limit,
        "matched_requests": len(request_ids),
        "random_only_requests": len(random_only),
        "random_success_rate": len(random) / attempt_limit,
        "spatial_only_requests": len(spatial_only),
        "spatialcf_success_rate": len(spatial) / attempt_limit,
        "trusted_source_digest": expected_source_digest,
        "spatialcf_mean_leakage": spatial_mean,
        "random_mean_leakage": random_mean,
        "relative_reduction": reduction,
    }
