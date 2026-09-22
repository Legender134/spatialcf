"""Strict provenance, typed references, and acyclic semantic lineage wires."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from typing import ClassVar, Generic, Literal, Self, TypeVar, get_args

from pydantic import StrictInt, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel, Sha256Digest
from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.serialization import canonical_json_bytes

__all__ = (
    "CounterfactualLineage",
    "DependencyInventoryEntry",
    "InterpreterIdentity",
    "LockMetadata",
    "ModuleFileIdentity",
    "NativeNotRequested",
    "RuntimeProvenance",
    "SemanticObjectReference",
    "SourceLineageIdentity",
)

_ExpectedKindT = TypeVar("_ExpectedKindT", bound=str)


class SemanticObjectReference(CanonicalModel, Generic[_ExpectedKindT]):
    """A strict prior-object reference with a schema-fixed object class."""

    expected_kind: _ExpectedKindT
    version: Literal["1"] = "1"
    semantic_sha256: Sha256Digest

    @classmethod
    def from_model(cls, value: HashBoundCanonicalModel) -> Self:
        """Reference one exact hash-bound value without changing its wire."""

        if not isinstance(value, HashBoundCanonicalModel):
            raise TypeError("reference factory requires a hash-bound model")
        expected_values = get_args(cls.model_fields["expected_kind"].annotation)
        if len(expected_values) != 1 or not isinstance(expected_values[0], str):
            raise TypeError("reference factory requires one literal expected kind")
        expected_kind = expected_values[0]
        actual_kind = type(value).__name__
        if actual_kind != expected_kind:
            raise ValueError(
                f"reference expects {expected_kind}, received {actual_kind}"
            )
        checked = type(value).model_validate(
            value.model_dump(mode="python", warnings="error"), strict=True
        )
        return cls(
            expected_kind=expected_kind,
            semantic_sha256=getattr(checked, checked.SELF_DIGEST_FIELD),
        )


class InterpreterIdentity(CanonicalModel):
    kind: Literal["PYTHON_INTERPRETER"] = "PYTHON_INTERPRETER"
    version: Literal["1"] = "1"
    implementation: CanonicalId
    major: StrictInt
    minor: StrictInt
    micro: StrictInt
    canonical_json_number_grammar: Literal["python-json-3.11-finite-v1"] = (
        "python-json-3.11-finite-v1"
    )

    @model_validator(mode="after")
    def _supported_interpreter(self) -> Self:
        if self.implementation != "CPython" or (self.major, self.minor) != (3, 11):
            raise ValueError("M4 requires the supported CPython 3.11 runtime")
        if self.micro < 0:
            raise ValueError("interpreter micro version must be non-negative")
        return self


class DependencyInventoryEntry(CanonicalModel):
    kind: Literal["INSTALLED_DISTRIBUTION"] = "INSTALLED_DISTRIBUTION"
    version: Literal["1"] = "1"
    distribution_name: CanonicalId
    distribution_version: CanonicalId
    metadata_byte_length: StrictInt
    metadata_byte_sha256: Sha256Digest

    @model_validator(mode="after")
    def _positive_length(self) -> Self:
        if self.metadata_byte_length <= 0:
            raise ValueError("dependency metadata must not be empty")
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.distribution_name) is None:
            raise ValueError("distribution name must use normalized package identity")
        return self


class ModuleFileIdentity(CanonicalModel):
    kind: Literal["PORTABLE_MODULE_FILE"] = "PORTABLE_MODULE_FILE"
    version: Literal["1"] = "1"
    package_relative_path: CanonicalId
    byte_length: StrictInt
    byte_sha256: Sha256Digest

    @model_validator(mode="after")
    def _portable_path(self) -> Self:
        path = self.package_relative_path
        if path.startswith("/") or PureWindowsPath(path).drive or "\\" in path or "//" in path:
            raise ValueError("module file identity must be a portable relative path")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("module file identity contains an unsafe component")
        if not path.startswith("spatialcf/") or not path.endswith(".py"):
            raise ValueError("module file must name portable spatialcf source")
        if self.byte_length < 0:
            raise ValueError("runtime module byte length must be non-negative")
        return self


class LockMetadata(CanonicalModel):
    kind: Literal["LOCK_METADATA"] = "LOCK_METADATA"
    version: Literal["1"] = "1"
    availability: Literal["AVAILABLE", "UNAVAILABLE"]
    scope: Literal["DISTRIBUTION_LOCK", "INSTALLED_PACKAGE_METADATA"]
    byte_length: StrictInt | None = None
    byte_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def _availability_matches_payload(self) -> Self:
        present = self.byte_length is not None and self.byte_sha256 is not None
        if (self.byte_length is None) != (self.byte_sha256 is None):
            raise ValueError("lock length and byte hash must appear together")
        if self.availability == "AVAILABLE":
            if not present or self.byte_length is None or self.byte_length <= 0:
                raise ValueError("available lock metadata requires non-empty bytes")
        elif present:
            raise ValueError("unavailable lock metadata cannot carry byte identity")
        return self


class RuntimeProvenance(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/runtime-provenance/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "runtime_provenance_sha256"

    kind: Literal["SEMANTIC_CONTRAST_RUNTIME_PROVENANCE"] = (
        "SEMANTIC_CONTRAST_RUNTIME_PROVENANCE"
    )
    version: Literal["1"] = "1"
    scope: Literal["LOCAL_SOURCE_SNAPSHOT"] = "LOCAL_SOURCE_SNAPSHOT"
    interpreter: InterpreterIdentity
    dependencies: tuple[DependencyInventoryEntry, ...]
    module_files: tuple[ModuleFileIdentity, ...]
    lock_metadata: LockMetadata
    runtime_provenance_sha256: Sha256Digest

    @model_validator(mode="after")
    def _closed_inventory(self) -> Self:
        if not self.dependencies:
            raise ValueError("runtime provenance requires dependency inventory")
        names = tuple(item.distribution_name for item in self.dependencies)
        if names != tuple(sorted(names, key=canonical_json_bytes)) or len(names) != len(set(names)):
            raise ValueError("distribution inventory must have sorted unique names")
        paths = tuple(item.package_relative_path for item in self.module_files)
        if not paths or paths != tuple(sorted(paths, key=canonical_json_bytes)) or len(paths) != len(set(paths)):
            raise ValueError("runtime module inventory must have sorted unique paths")
        return self


class NativeNotRequested(CanonicalModel):
    """The only native-execution branch permitted by M4."""

    kind: Literal["NATIVE_EXECUTION"] = "NATIVE_EXECUTION"
    version: Literal["1"] = "1"
    status: Literal["NOT_REQUESTED"] = "NOT_REQUESTED"


class SourceLineageIdentity(HashBoundCanonicalModel):
    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/source-lineage/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "source_lineage_sha256"

    kind: Literal["SEMANTIC_CONTRAST_SOURCE_LINEAGE"] = (
        "SEMANTIC_CONTRAST_SOURCE_LINEAGE"
    )
    version: Literal["1"] = "1"
    source_dataset_id: CanonicalId
    source_revision_id: CanonicalId
    source_record_id: CanonicalId
    source_record_byte_sha256: Sha256Digest
    source_files_manifest_sha256: Sha256Digest
    source_group: CanonicalId
    scene_state_sha256: Sha256Digest
    source_lineage_sha256: Sha256Digest


class CounterfactualLineage(HashBoundCanonicalModel):
    """Success-only M4 lineage with no edge back to its containing record."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/semantic-contrast/lineage/1.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "counterfactual_lineage_sha256"

    kind: Literal["COUNTERFACTUAL_LINEAGE"] = "COUNTERFACTUAL_LINEAGE"
    version: Literal["1"] = "1"
    pair_content_sha256: Sha256Digest
    source: SourceLineageIdentity
    solve_request: SemanticObjectReference[Literal["CounterfactualSolveRequest"]]
    selection: SemanticObjectReference[Literal["BackendSelectionRecord"]]
    submission: SemanticObjectReference[Literal["BackendProposalSubmission"]]
    checked_outcome: SemanticObjectReference[Literal["CheckedProofOutcome"]]
    dispatch: SemanticObjectReference[Literal["VerifierDispatchRecord"]]
    certificate: SemanticObjectReference[Literal["CertifiedSolutionCertificate"]]
    result: SemanticObjectReference[Literal["CertifiedSolutionResult"]]
    program: SemanticObjectReference[Literal["EditProgram"]]
    grounded_obligations: SemanticObjectReference[Literal["GroundedObligationSet"]]
    publication_policy_sha256: Sha256Digest
    runtime_provenance: SemanticObjectReference[Literal["RuntimeProvenance"]]
    native_execution: NativeNotRequested
    counterfactual_lineage_sha256: Sha256Digest
