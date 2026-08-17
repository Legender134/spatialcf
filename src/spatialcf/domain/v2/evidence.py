"""Typed provenance and audit evidence outside Canonical v2 semantics.

This module may point at a semantic problem digest, but it never imports or
contains the semantic problem.  Core constraints, objectives, artifacts, and
certificates must likewise never import this evidence layer.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FiniteFloat,
    SchemaIdentityV2,
    Sha256Digest,
    V2Model,
)
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)

_EVIDENCE_HASH_DOMAIN = "spatialcf.evidence-envelope.v2"
_PRE_SEMANTIC_EVIDENCE_HASH_DOMAIN = "spatialcf.pre-semantic-evidence-envelope.v2"
_NativeLocatorValue = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=4096),
]
_PortableRelativePath = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=1024),
]
_NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class RawEvidenceKindV2(StrEnum):
    SOURCE_RECORD = "SOURCE_RECORD"
    RAW_OBSERVATION = "RAW_OBSERVATION"
    MAPPING_INPUT = "MAPPING_INPUT"
    RUNTIME_LOG = "RUNTIME_LOG"
    WRITE_BACK = "WRITE_BACK"
    FINAL_AUDIT = "FINAL_AUDIT"


class EvidenceMediaTypeV2(StrEnum):
    APPLICATION_JSON = "application/json"
    IMAGE_PNG = "image/png"
    APPLICATION_NPY = "application/x-npy"
    TEXT_PLAIN = "text/plain"
    APPLICATION_OCTET_STREAM = "application/octet-stream"


class NativeLocatorKindV2(StrEnum):
    """Closed non-filesystem namespaces for opaque native identities."""

    DATASET_KEY = "DATASET_KEY"
    SCENE_GRAPH_PATH = "SCENE_GRAPH_PATH"
    OBJECT_HANDLE = "OBJECT_HANDLE"


class NativeLocatorV2(V2Model):
    """A typed native identity, never a host filesystem path."""

    kind: NativeLocatorKindV2
    value: _NativeLocatorValue

    @field_validator("value")
    @classmethod
    def validate_locator_value(cls, value: str, info: ValidationInfo) -> str:
        value = _validate_opaque_locator(value)
        kind = info.data.get("kind")
        if re.match(r"(?i)^file:", value):
            raise ValueError("native locator must not be a filesystem URI")
        if kind is NativeLocatorKindV2.SCENE_GRAPH_PATH:
            if not value.startswith("/") or "\\" in value:
                raise ValueError(
                    "SCENE_GRAPH_PATH locator must use absolute graph syntax"
                )
            if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
                raise ValueError(
                    "scene graph locator must not contain empty or traversal nodes"
                )
            return value
        if (
            value.startswith(("/", "~"))
            or "\\" in value
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError(
                "dataset/object locator kind must not contain an absolute filesystem path"
            )
        return value


class MappingProofKindV2(StrEnum):
    ENTITY_IDENTITY = "ENTITY_IDENTITY"
    GEOMETRY = "GEOMETRY"
    SUPPORT = "SUPPORT"
    CAMERA = "CAMERA"
    OBSERVATION_NORMALIZATION = "OBSERVATION_NORMALIZATION"
    RELATION_NORMALIZATION = "RELATION_NORMALIZATION"
    WRITE_BACK = "WRITE_BACK"


class MappingProofStatusV2(StrEnum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    INCOMPLETE = "INCOMPLETE"


class EvidenceMeasurementUnitV2(StrEnum):
    METRE = "METRE"
    SQUARE_METRE = "SQUARE_METRE"
    RADIAN = "RADIAN"
    PIXEL = "PIXEL"
    FRACTION = "FRACTION"


class AuditStatusV2(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


_MAPPING_ALLOWED_RAW_KINDS: dict[
    MappingProofKindV2,
    frozenset[RawEvidenceKindV2],
] = {
    MappingProofKindV2.ENTITY_IDENTITY: frozenset(
        {RawEvidenceKindV2.SOURCE_RECORD, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.GEOMETRY: frozenset(
        {RawEvidenceKindV2.RAW_OBSERVATION, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.SUPPORT: frozenset(
        {RawEvidenceKindV2.RAW_OBSERVATION, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.CAMERA: frozenset(
        {RawEvidenceKindV2.RAW_OBSERVATION, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.OBSERVATION_NORMALIZATION: frozenset(
        {RawEvidenceKindV2.RAW_OBSERVATION, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.RELATION_NORMALIZATION: frozenset(
        {RawEvidenceKindV2.RAW_OBSERVATION, RawEvidenceKindV2.MAPPING_INPUT}
    ),
    MappingProofKindV2.WRITE_BACK: frozenset(
        {RawEvidenceKindV2.WRITE_BACK, RawEvidenceKindV2.RAW_OBSERVATION}
    ),
}

_AUDIT_ALLOWED_RAW_KINDS = frozenset(
    {
        RawEvidenceKindV2.FINAL_AUDIT,
        RawEvidenceKindV2.RUNTIME_LOG,
        RawEvidenceKindV2.RAW_OBSERVATION,
        RawEvidenceKindV2.WRITE_BACK,
    }
)


class AdapterIdentityV2(V2Model):
    """Identity of the fact/write-back adapter that produced this envelope."""

    adapter_id: CanonicalId
    adapter_version: CanonicalId
    platform_id: CanonicalId
    implementation_sha256: Sha256Digest


class SourceIdentityV2(V2Model):
    """Immutable dataset item identity and its opaque native scene locator."""

    source_id: CanonicalId
    source_version: CanonicalId
    source_partition: CanonicalId | None
    native_scene_locator: NativeLocatorV2
    content_sha256: Sha256Digest
    raw_evidence_ids: tuple[CanonicalId, ...]

    @model_validator(mode="after")
    def canonicalize_raw_evidence_ids(self) -> Self:
        object.__setattr__(
            self,
            "raw_evidence_ids",
            _sorted_unique_strings(
                self.raw_evidence_ids,
                label="source raw_evidence_ids",
            ),
        )
        return self


class NativeObjectBindingV2(V2Model):
    """One-to-one evidence mapping from a native entity to a Canonical ID."""

    canonical_object_id: CanonicalId
    native_object_locator: NativeLocatorV2
    mapping_proof_id: CanonicalId


class RawEvidenceRefV2(V2Model):
    """Content-addressed reference to raw bytes outside the semantic payload."""

    evidence_id: CanonicalId
    kind: RawEvidenceKindV2
    media_type: EvidenceMediaTypeV2
    relative_path: _PortableRelativePath
    sha256: Sha256Digest
    byte_length: _NonNegativeStrictInt

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value != value.strip() or _contains_control_character(value):
            raise ValueError("evidence path must be a portable relative path")
        if (
            value.startswith(("/", "~"))
            or "\\" in value
            or "://" in value
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("evidence path must be relative and portable")
        parts = value.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("evidence path must not contain traversal components")
        return value


class EvidenceMeasurementV2(V2Model):
    """One finite, unit-explicit observation optionally enclosed by bounds."""

    measurement_id: CanonicalId
    unit: EvidenceMeasurementUnitV2
    value: FiniteFloat
    lower_bound: FiniteFloat | None
    upper_bound: FiniteFloat | None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if (self.lower_bound is None) != (self.upper_bound is None):
            raise ValueError("measurement bounds must be both present or both absent")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and not self.lower_bound <= self.value <= self.upper_bound
        ):
            raise ValueError("measurement value must lie inside its bounds interval")
        if self.unit is EvidenceMeasurementUnitV2.FRACTION:
            values = (self.value, self.lower_bound, self.upper_bound)
            if any(value is not None and not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("FRACTION measurements and bounds must lie in [0, 1]")
        return self


class MappingProofV2(V2Model):
    """Typed result of one native-to-Canonical mapping check."""

    proof_id: CanonicalId
    kind: MappingProofKindV2
    method_id: CanonicalId
    status: MappingProofStatusV2
    canonical_ids: tuple[CanonicalId, ...] = Field(min_length=1)
    native_locators: tuple[NativeLocatorV2, ...] = Field(min_length=1)
    raw_evidence_ids: tuple[CanonicalId, ...] = Field(min_length=1)
    reason_codes: tuple[CanonicalId, ...]
    measurements: tuple[EvidenceMeasurementV2, ...]
    canonical_edit_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def canonicalize_and_validate_proof(self) -> Self:
        object.__setattr__(
            self,
            "canonical_ids",
            _sorted_unique_strings(self.canonical_ids, label="canonical_ids"),
        )
        object.__setattr__(
            self,
            "native_locators",
            _sorted_unique_native_locators(self.native_locators),
        )
        object.__setattr__(
            self,
            "raw_evidence_ids",
            _sorted_unique_strings(
                self.raw_evidence_ids,
                label="mapping raw_evidence_ids",
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_strings(self.reason_codes, label="mapping reason_codes"),
        )
        object.__setattr__(
            self,
            "measurements",
            _sorted_unique_models(
                self.measurements,
                id_field="measurement_id",
                label="mapping measurements",
            ),
        )
        if self.status is MappingProofStatusV2.VERIFIED and self.reason_codes:
            raise ValueError("VERIFIED mapping proof must not carry reason codes")
        if self.status is not MappingProofStatusV2.VERIFIED and not self.reason_codes:
            raise ValueError(
                "rejected or incomplete mapping proof requires a reason code"
            )
        if self.kind is MappingProofKindV2.ENTITY_IDENTITY and (
            len(self.canonical_ids) != 1 or len(self.native_locators) != 1
        ):
            raise ValueError(
                "ENTITY_IDENTITY proof requires exactly one Canonical ID and "
                "exactly one native locator"
            )
        if self.kind is MappingProofKindV2.WRITE_BACK:
            if self.canonical_edit_sha256 is None:
                raise ValueError("WRITE_BACK proof requires a Canonical Edit hash")
        elif self.canonical_edit_sha256 is not None:
            raise ValueError("only WRITE_BACK proof may carry a Canonical Edit hash")
        return self


class PreSemanticEvidenceEnvelopeV2(V2Model):
    """Traceable source/mapping evidence before a semantic problem can exist.

    A source adapter may fail before it can construct any mapping proof, so the
    proof tuple may be empty.  The source itself must still close at least one
    content-addressed ``SOURCE_RECORD`` reference.  This envelope deliberately
    has no semantic-problem digest and cannot be consumed by the core solver.
    """

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(
            schema_name="pre-semantic-evidence-envelope"
        )
    )
    adapter: AdapterIdentityV2
    source: SourceIdentityV2
    raw_evidence_refs: tuple[RawEvidenceRefV2, ...]
    mapping_proofs: tuple[MappingProofV2, ...]

    @model_validator(mode="after")
    def canonicalize_and_validate_envelope(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(
            schema_name="pre-semantic-evidence-envelope"
        ):
            raise ValueError(
                "pre-semantic-evidence-envelope schema identity must be fixed"
            )
        object.__setattr__(
            self,
            "raw_evidence_refs",
            _sorted_unique_models(
                self.raw_evidence_refs,
                id_field="evidence_id",
                label="pre-semantic raw evidence",
            ),
        )
        paths = tuple(item.relative_path for item in self.raw_evidence_refs)
        if len(set(paths)) != len(paths):
            raise ValueError("pre-semantic raw evidence relative paths must be unique")
        object.__setattr__(
            self,
            "mapping_proofs",
            _sorted_unique_models(
                self.mapping_proofs,
                id_field="proof_id",
                label="pre-semantic mapping proofs",
            ),
        )
        self._validate_references()
        return self

    def _validate_references(self) -> None:
        raw_by_id = {item.evidence_id: item for item in self.raw_evidence_refs}
        raw_ids = set(raw_by_id)
        if any(
            proof.kind is MappingProofKindV2.WRITE_BACK for proof in self.mapping_proofs
        ):
            raise ValueError(
                "pre-semantic evidence must not contain a WRITE_BACK proof"
            )
        if not self.source.raw_evidence_ids:
            raise ValueError("pre-semantic source requires at least one SOURCE_RECORD")
        _require_known_ids(
            self.source.raw_evidence_ids,
            raw_ids,
            label="pre-semantic source references unknown raw evidence",
        )
        _require_raw_kinds(
            self.source.raw_evidence_ids,
            raw_by_id,
            allowed=frozenset({RawEvidenceKindV2.SOURCE_RECORD}),
            label="pre-semantic source evidence must use SOURCE_RECORD kind",
        )
        referenced_raw_ids = set(self.source.raw_evidence_ids)
        for proof in self.mapping_proofs:
            referenced_raw_ids.update(proof.raw_evidence_ids)
            _require_known_ids(
                proof.raw_evidence_ids,
                raw_ids,
                label="pre-semantic mapping proof references unknown raw evidence",
            )
            _require_raw_kinds(
                proof.raw_evidence_ids,
                raw_by_id,
                allowed=_MAPPING_ALLOWED_RAW_KINDS[proof.kind],
                label=(
                    f"pre-semantic mapping {proof.kind.value} raw evidence kind "
                    "is incompatible"
                ),
            )
        orphan_raw_ids = raw_ids.difference(referenced_raw_ids)
        if orphan_raw_ids:
            raise ValueError(
                "orphan pre-semantic raw evidence is not referenced by source or "
                f"mapping proofs: {', '.join(sorted(orphan_raw_ids))}"
            )

    @property
    def pre_semantic_evidence_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_PRE_SEMANTIC_EVIDENCE_HASH_DOMAIN,
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes_v2(self)


class RuntimeIdentityEvidenceV2(V2Model):
    """Opaque execution runtime identity used only by audit evidence."""

    runtime_id: CanonicalId
    platform_id: CanonicalId
    runtime_name: CanonicalId
    runtime_version: CanonicalId
    build_sha256: Sha256Digest | None
    raw_evidence_ids: tuple[CanonicalId, ...]

    @model_validator(mode="after")
    def canonicalize_raw_evidence_ids(self) -> Self:
        object.__setattr__(
            self,
            "raw_evidence_ids",
            _sorted_unique_strings(
                self.raw_evidence_ids,
                label="runtime raw_evidence_ids",
            ),
        )
        return self


class FinalAuditOutcomeV2(V2Model):
    """One final replay/audit outcome; never a core score or solver status."""

    audit_id: CanonicalId
    audit_method_id: CanonicalId
    status: AuditStatusV2
    runtime_id: CanonicalId | None
    canonical_edit_sha256: Sha256Digest | None
    reason_codes: tuple[CanonicalId, ...]
    raw_evidence_ids: tuple[CanonicalId, ...]
    measurements: tuple[EvidenceMeasurementV2, ...]

    @model_validator(mode="after")
    def canonicalize_and_validate_audit(self) -> Self:
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_strings(self.reason_codes, label="audit reason_codes"),
        )
        object.__setattr__(
            self,
            "raw_evidence_ids",
            _sorted_unique_strings(
                self.raw_evidence_ids,
                label="audit raw_evidence_ids",
            ),
        )
        object.__setattr__(
            self,
            "measurements",
            _sorted_unique_models(
                self.measurements,
                id_field="measurement_id",
                label="audit measurements",
            ),
        )
        if self.status is AuditStatusV2.PASS:
            if self.reason_codes:
                raise ValueError("PASS audit must not carry reason codes")
            self._require_executed_audit_fields("PASS")
        elif self.status is AuditStatusV2.FAIL:
            if not self.reason_codes:
                raise ValueError("FAIL audit requires a reason code")
            self._require_executed_audit_fields("FAIL")
        else:
            if not self.reason_codes:
                raise ValueError("NOT_RUN audit requires a reason code")
            if self.measurements:
                raise ValueError("NOT_RUN audit must not carry measurements")
        return self

    def _require_executed_audit_fields(self, status: str) -> None:
        if self.runtime_id is None or self.canonical_edit_sha256 is None:
            raise ValueError(
                f"{status} audit requires runtime and Canonical Edit hashes"
            )
        if not self.raw_evidence_ids:
            raise ValueError(f"{status} audit requires raw evidence")


class EvidenceEnvelopeV2(V2Model):
    """Independent provenance envelope bound only to a semantic problem digest."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="evidence-envelope")
    )
    semantic_problem_sha256: Sha256Digest
    adapter: AdapterIdentityV2
    source: SourceIdentityV2
    native_object_bindings: tuple[NativeObjectBindingV2, ...]
    raw_evidence_refs: tuple[RawEvidenceRefV2, ...]
    mapping_proofs: tuple[MappingProofV2, ...]
    runtime_identities: tuple[RuntimeIdentityEvidenceV2, ...]
    final_audits: tuple[FinalAuditOutcomeV2, ...]

    @model_validator(mode="after")
    def canonicalize_and_validate_envelope(self) -> Self:
        if self.schema_identity != SchemaIdentityV2(schema_name="evidence-envelope"):
            raise ValueError("evidence-envelope schema identity must be fixed")

        object.__setattr__(
            self,
            "native_object_bindings",
            _sorted_unique_models(
                self.native_object_bindings,
                id_field="canonical_object_id",
                label="canonical object bindings",
            ),
        )
        native_locators = tuple(
            binding.native_object_locator for binding in self.native_object_bindings
        )
        if len(set(native_locators)) != len(native_locators):
            raise ValueError("native object binding locators must be unique")
        object.__setattr__(
            self,
            "raw_evidence_refs",
            _sorted_unique_models(
                self.raw_evidence_refs,
                id_field="evidence_id",
                label="raw evidence",
            ),
        )
        relative_paths = tuple(item.relative_path for item in self.raw_evidence_refs)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("raw evidence relative paths must be unique")
        object.__setattr__(
            self,
            "mapping_proofs",
            _sorted_unique_models(
                self.mapping_proofs,
                id_field="proof_id",
                label="mapping proofs",
            ),
        )
        object.__setattr__(
            self,
            "runtime_identities",
            _sorted_unique_models(
                self.runtime_identities,
                id_field="runtime_id",
                label="runtime identities",
            ),
        )
        object.__setattr__(
            self,
            "final_audits",
            _sorted_unique_models(
                self.final_audits,
                id_field="audit_id",
                label="final audits",
            ),
        )
        self._validate_references()
        return self

    @property
    def evidence_envelope_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_EVIDENCE_HASH_DOMAIN)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes_v2(self)

    def _validate_references(self) -> None:
        raw_ids = {item.evidence_id for item in self.raw_evidence_refs}
        raw_by_id = {item.evidence_id: item for item in self.raw_evidence_refs}
        proof_by_id = {item.proof_id: item for item in self.mapping_proofs}
        runtime_ids = {item.runtime_id for item in self.runtime_identities}
        referenced_raw_ids = set(self.source.raw_evidence_ids)

        if not self.source.raw_evidence_ids:
            raise ValueError("post-semantic source requires at least one SOURCE_RECORD")
        _require_known_ids(
            self.source.raw_evidence_ids,
            raw_ids,
            label="source unknown raw evidence",
        )
        _require_raw_kinds(
            self.source.raw_evidence_ids,
            raw_by_id,
            allowed=frozenset({RawEvidenceKindV2.SOURCE_RECORD}),
            label="source evidence must use SOURCE_RECORD kind",
        )
        for proof in self.mapping_proofs:
            referenced_raw_ids.update(proof.raw_evidence_ids)
            _require_known_ids(
                proof.raw_evidence_ids,
                raw_ids,
                label="mapping proof references unknown raw evidence",
            )
            _require_raw_kinds(
                proof.raw_evidence_ids,
                raw_by_id,
                allowed=_MAPPING_ALLOWED_RAW_KINDS[proof.kind],
                label=f"mapping {proof.kind.value} raw evidence kind is incompatible",
            )
        for runtime in self.runtime_identities:
            referenced_raw_ids.update(runtime.raw_evidence_ids)
            _require_known_ids(
                runtime.raw_evidence_ids,
                raw_ids,
                label="runtime references unknown raw evidence",
            )
            _require_raw_kinds(
                runtime.raw_evidence_ids,
                raw_by_id,
                allowed=frozenset({RawEvidenceKindV2.RUNTIME_LOG}),
                label="runtime evidence must use RUNTIME_LOG kind",
            )
        for audit in self.final_audits:
            referenced_raw_ids.update(audit.raw_evidence_ids)
            _require_known_ids(
                audit.raw_evidence_ids,
                raw_ids,
                label="final audit references unknown raw evidence",
            )
            _require_raw_kinds(
                audit.raw_evidence_ids,
                raw_by_id,
                allowed=_AUDIT_ALLOWED_RAW_KINDS,
                label=(
                    "audit raw evidence kind is incompatible; executed audit requires "
                    "FINAL_AUDIT"
                ),
            )
            if audit.status is not AuditStatusV2.NOT_RUN and not any(
                raw_by_id[evidence_id].kind is RawEvidenceKindV2.FINAL_AUDIT
                for evidence_id in audit.raw_evidence_ids
            ):
                raise ValueError(
                    "executed audit requires at least one FINAL_AUDIT raw evidence"
                )
            if audit.runtime_id is not None and audit.runtime_id not in runtime_ids:
                raise ValueError("final audit references unknown runtime identity")
        edit_hashes = {
            audit.canonical_edit_sha256
            for audit in self.final_audits
            if audit.canonical_edit_sha256 is not None
        }
        if len(edit_hashes) > 1:
            raise ValueError(
                "final audits must reference the same final Canonical Edit"
            )
        for binding in self.native_object_bindings:
            proof = proof_by_id.get(binding.mapping_proof_id)
            if proof is None:
                raise ValueError(
                    "native object binding references unknown mapping proof"
                )
            if (
                proof.kind is not MappingProofKindV2.ENTITY_IDENTITY
                or proof.status is not MappingProofStatusV2.VERIFIED
            ):
                raise ValueError(
                    "native object binding requires a verified identity proof"
                )
            if binding.canonical_object_id not in proof.canonical_ids:
                raise ValueError(
                    "binding Canonical object is absent from mapping proof"
                )
            if binding.native_object_locator not in proof.native_locators:
                raise ValueError("binding native locator is absent from mapping proof")
        orphan_raw_ids = raw_ids.difference(referenced_raw_ids)
        if orphan_raw_ids:
            raise ValueError(
                "orphan raw evidence is not referenced by source, mapping, runtime, "
                f"or audit records: {', '.join(sorted(orphan_raw_ids))}"
            )


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _validate_opaque_locator(value: str) -> str:
    if value != value.strip() or _contains_control_character(value):
        raise ValueError("native locator must be a non-blank opaque string")
    return value


def _sorted_unique_strings(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(values))


def _sorted_unique_native_locators(
    values: tuple[NativeLocatorV2, ...],
) -> tuple[NativeLocatorV2, ...]:
    keys = tuple((value.kind.value, value.value) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("native_locators must be unique")
    return tuple(sorted(values, key=lambda value: (value.kind.value, value.value)))


def _sorted_unique_models(
    values: tuple[V2Model, ...],
    *,
    id_field: str,
    label: str,
) -> tuple[V2Model, ...]:
    identities = tuple(getattr(value, id_field) for value in values)
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} IDs must be unique")
    return tuple(sorted(values, key=lambda value: getattr(value, id_field)))


def _require_known_ids(
    referenced_ids: tuple[str, ...],
    known_ids: set[str],
    *,
    label: str,
) -> None:
    unknown = set(referenced_ids).difference(known_ids)
    if unknown:
        raise ValueError(f"{label}: {', '.join(sorted(unknown))}")


def _require_raw_kinds(
    referenced_ids: tuple[str, ...],
    raw_by_id: dict[str, RawEvidenceRefV2],
    *,
    allowed: frozenset[RawEvidenceKindV2],
    label: str,
) -> None:
    incompatible = tuple(
        evidence_id
        for evidence_id in referenced_ids
        if evidence_id in raw_by_id and raw_by_id[evidence_id].kind not in allowed
    )
    if incompatible:
        raise ValueError(f"{label}: {', '.join(sorted(incompatible))}")
