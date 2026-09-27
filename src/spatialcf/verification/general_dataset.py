"""Descriptor-bound structural reader for general dataset artifacts."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from spatialcf.domain import general_dataset as models
from spatialcf.domain.lineage import RuntimeProvenance
from spatialcf.verification.artifact_models import _parse_exact
from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    read_regular_at,
    revalidate_entries,
    snapshot_exact_directory,
)

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024 * 1024
FILE_TYPES = {
    "input.json": models.GeneralDatasetInput,
    "catalog.json": models.GeneralFrozenCatalog,
    "terminals.json": models.GeneralTerminalLedger,
    "records.json": models.GeneralRecords,
    "report.json": models.GeneralDatasetReport,
    "provenance.json": RuntimeProvenance,
    "manifest.json": models.GeneralDatasetManifest,
}


@dataclass(frozen=True)
class GeneralBundle:
    input: models.GeneralDatasetInput
    catalog: models.GeneralFrozenCatalog
    ledger: models.GeneralTerminalLedger
    records: models.GeneralRecords
    report: models.GeneralDatasetReport
    provenance: RuntimeProvenance
    manifest: models.GeneralDatasetManifest
    payloads: Mapping[str, bytes]


@contextmanager
def read_general_input(path: Path):
    path = Path(os.path.abspath(path))
    with bound_absolute_directory(path.parent) as descriptor:
        entry = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        raw = read_regular_at(
            descriptor, path.name, MAX_INPUT_BYTES, expected_stat=entry
        )
        value = _parse_exact(raw, models.GeneralDatasetInput)

        def revalidate():
            revalidate_entries(descriptor, {path.name: entry})

        revalidate()
        yield value, revalidate
        revalidate()


@contextmanager
def read_general_bundle(root: Path, *, expected_identity=None):
    """Read all bytes with retained bindings; proof checking belongs to replay."""
    with bound_absolute_directory(
        Path(os.path.abspath(root)), expected_identity=expected_identity
    ) as descriptor:
        entries = snapshot_exact_directory(
            descriptor, regular_names=set(FILE_TYPES), directory_names=set()
        )
        payloads = {
            name: read_regular_at(
                descriptor,
                name,
                MAX_INPUT_BYTES if name == "input.json" else MAX_FILE_BYTES,
                expected_stat=entries[name],
            )
            for name in FILE_TYPES
        }
        values = {
            name: _parse_exact(payloads[name], cls) for name, cls in FILE_TYPES.items()
        }
        manifest = values["manifest.json"]
        for row in manifest.inventory:
            raw = payloads[row.name]
            if (
                len(raw) != row.byte_length
                or hashlib.sha256(raw).hexdigest() != row.byte_sha256
            ):
                raise ValueError("general dataset inventory byte identity mismatch")
        bundle = GeneralBundle(
            values["input.json"],
            values["catalog.json"],
            values["terminals.json"],
            values["records.json"],
            values["report.json"],
            values["provenance.json"],
            manifest,
            MappingProxyType(payloads),
        )
        ids = tuple(row.candidate_id for row in bundle.catalog.candidates)
        if tuple(row.candidate_id for row in bundle.ledger.terminals) != ids:
            raise ValueError("terminal universe or order differs from catalog")

        def revalidate():
            snapshot_exact_directory(
                descriptor, regular_names=set(FILE_TYPES), directory_names=set()
            )
            revalidate_entries(descriptor, entries)

        revalidate()
        yield bundle
        revalidate()
