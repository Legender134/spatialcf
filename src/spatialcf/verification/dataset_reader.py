"""Read-only legacy dataset authentication and independent artifact verification."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import uuid
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from inspect import get_annotations as _get_annotations
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar, get_type_hints as _get_type_hints

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from spatialcf.domain.request import (
    InterventionSpec,
    QualityTier,
    Relation,
    SolverStatus,
)
from spatialcf.domain.scene import Scene
from spatialcf.verification.dataset_models import (
    FailureRecord as FailureRecord,
    PairRecord as PairRecord,
)
from spatialcf.verification.integrity import (
    calculate_candidate_objective,
    calculate_weighted_objective,
    canonical_json_bytes,
    source_corpus_digest,
    topdown_payload,
    validate_generation_budget,
)
from spatialcf.verification.profile import profile_from_manifest
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


# Resolve declared annotations before restoring legacy type identity.
for _owned_type in (
    _Dataset,
    AuthenticatedDatasetIdentity,
    _StreamedFileDigest,
):
    _owned_type.__annotations__ = {
        key: hint
        for key, hint in _get_type_hints(_owned_type, include_extras=True).items()
        if key in _get_annotations(_owned_type)
    }
del _owned_type
_RecordT.__module__ = "spatialcf.verification.dataset"
_Dataset.__module__ = "spatialcf.verification.dataset"
AuthenticatedDatasetIdentity.__module__ = "spatialcf.verification.dataset"
_StreamedFileDigest.__module__ = "spatialcf.verification.dataset"
_audit_unsafe.__module__ = "spatialcf.verification.dataset"
_regular_files.__module__ = "spatialcf.verification.dataset"
_audit_validate_relative_path.__module__ = "spatialcf.verification.dataset"
_audit_sha256.__module__ = "spatialcf.verification.dataset"
_checksum_entries.__module__ = "spatialcf.verification.dataset"
_validate_checksums.__module__ = "spatialcf.verification.dataset"
_capture_regular_file.__module__ = "spatialcf.verification.dataset"
_capture_regular_file_payloads.__module__ = "spatialcf.verification.dataset"
_audit_json_bytes.__module__ = "spatialcf.verification.dataset"
_read_records.__module__ = "spatialcf.verification.dataset"
_read_manifest.__module__ = "spatialcf.verification.dataset"
_relation_diff_errors.__module__ = "spatialcf.verification.dataset"
_expected_request_id.__module__ = "spatialcf.verification.dataset"
_validate_camera_resources.__module__ = "spatialcf.verification.dataset"
_validate_image.__module__ = "spatialcf.verification.dataset"
_validate_depth.__module__ = "spatialcf.verification.dataset"
_validate_pointcloud.__module__ = "spatialcf.verification.dataset"
_validate_json_artifact.__module__ = "spatialcf.verification.dataset"
_read_canonical_model.__module__ = "spatialcf.verification.dataset"
_retained_snapshot_payload.__module__ = "spatialcf.verification.dataset"
_published_pair_after_scenes.__module__ = "spatialcf.verification.dataset"
_validate_attempt_ledger.__module__ = "spatialcf.verification.dataset"
read_authenticated_dataset_identity.__module__ = "spatialcf.verification.dataset"
_read_dataset.__module__ = "spatialcf.verification.dataset"
_verify_scenes.__module__ = "spatialcf.verification.dataset"
