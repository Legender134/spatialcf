"""Read-only descriptor-bound verification for retained artifact trees."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from spatialcf.verification.filesystem import (
    read_regular_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
)


def snapshot_exact_artifact_tree(
    descriptor: int,
    *,
    regular_names: set[str] | frozenset[str],
    directory_names: set[str] | frozenset[str] = frozenset(),
):
    """Bind one exact direct-child roster before semantic verification."""

    return snapshot_exact_directory(
        descriptor,
        regular_names=regular_names,
        directory_names=directory_names,
    )


def scan_retained_directory(
    descriptor: int,
    *,
    maximum_entries: int,
):
    """Enumerate one retained directory without mutation or path recursion."""

    return scan_directory(descriptor, maximum_entries=maximum_entries)


def read_retained_artifact(
    directory_descriptor: int,
    name: str,
    maximum_bytes: int,
    *,
    expected_stat=None,
) -> bytes:
    """Read one direct retained artifact and reject binding changes."""

    return read_regular_at(
        directory_descriptor,
        name,
        maximum_bytes,
        expected_stat=expected_stat,
    )


def revalidate_retained_tree(
    descriptor: int,
    entries: Mapping,
) -> None:
    """Require each retained direct child to preserve its bound fingerprint."""

    revalidate_entries(descriptor, dict(entries))


def retained_sha256_digests(payloads: Mapping[str, bytes]) -> dict[str, str]:
    """Compute an exact direct-child checksum roster for retained bytes."""

    digests: dict[str, str] = {}
    for name in sorted(payloads):
        if (
            type(name) is not str
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError("competition native filename is invalid")
        payload = payloads[name]
        if type(payload) is not bytes:
            raise TypeError("retained artifact payloads must be exact bytes")
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def canonical_checksum_ledger(digests: Mapping[str, str]) -> bytes:
    """Encode one canonical direct-child SHA-256 ledger without writing it."""

    lines: list[str] = []
    for name in sorted(digests):
        if (
            type(name) is not str
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError("competition native filename is invalid")
        digest = digests[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("retained artifact digest is invalid")
        lines.append(f"{digest}  {name}\n")
    return "".join(lines).encode("ascii")


def require_exact_checksum_ledger(
    payload: bytes,
    expected_digests: Mapping[str, str],
) -> None:
    """Reject a changed, missing, extra, or non-canonical checksum roster."""

    if type(payload) is not bytes:
        raise TypeError("retained artifact checksum ledger must be exact bytes")
    if payload != canonical_checksum_ledger(expected_digests):
        raise ValueError("native asset checksum ledger mismatch")


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Rename without replacement, or fail closed when unavailable."""
    if os.name == "nt":
        if os.path.lexists(destination):
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


__all__ = (
    "_rename_no_replace",
    "canonical_checksum_ledger",
    "read_retained_artifact",
    "require_exact_checksum_ledger",
    "retained_sha256_digests",
    "revalidate_retained_tree",
    "scan_retained_directory",
    "snapshot_exact_artifact_tree",
)
