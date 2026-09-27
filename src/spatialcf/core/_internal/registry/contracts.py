"""Static registry contracts validation and explicit dependencies."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)

from pydantic import (
    TypeAdapter,
    ValidationError,
)

from spatialcf.domain.definitions import (
    CapabilityRef,
    HashBoundCanonicalModel,
    canonical_json_bytes,
)

from spatialcf.domain.operators import (
    StateVariableRef,
)

from spatialcf.domain.profiles import (
    OwnerRef,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)


_UNCHANGED_LEAF_DOMAIN = "spatialcf/counterfactual/unchanged-leaves/3.0"


_FROZEN_LEAF_PATH_DOMAIN = "spatialcf/counterfactual/frozen-scene-leaf-path/3.0"


_EXTENSION_FACT_ADDRESS_DOMAIN = "spatialcf/counterfactual/extension-fact-address/3.0"


_ROLE_CERTIFIED_SOLUTION = "definition-kind:claim-certified-solution"


_ROLE_PROVEN_UNSAT = "definition-kind:claim-proven-unsat"


_ROLE_NONCERTIFIED_WITNESS = "definition-kind:claim-noncertified-witness"


_ROLE_UNKNOWN = "definition-kind:claim-unknown"


_ROLE_COMPLETE_DOMAIN = "definition-kind:complete-domain"


_ROLE_SOUND_COMPLETE_DOMAIN = "definition-kind:sound-complete-domain"


_ROLE_PROOF_MATERIAL = "definition-kind:proof-material"


_ROLE_RECORD_BINDING = "definition-kind:record-binding"


_ROLE_RESOURCE_ACCOUNTING = "definition-kind:resource-accounting"


_ROLE_RESOURCE_POLICY = "definition-kind:resource-policy"


_ROLE_ROUTING_POLICY = "definition-kind:routing-policy"


_ROLE_ROUTING_MATCH = "definition-kind:routing-match"


_ROLE_ROUTING_MISMATCH = "definition-kind:routing-mismatch"


_ROLE_ROUTING_SELECTION_DISPOSITION = "definition-kind:routing-selection-disposition"


_ROLE_ROUTING_SELECTION_REASON = "definition-kind:routing-selection-reason"


_MISMATCH_ORDER = (
    "profile",
    "predicate",
    "operator",
    "objective",
    "numeric",
    "proof",
    "resource",
    "backend",
)


_RECORD_KIND_FIELD = "field:definition-kind"


_RECORD_REFERENCE_FIELD = "field:definition-reference"


_RECORD_BOUND_REFERENCE_FIELD = "field:bound-record-ref"


_RECORD_BOUND_SHA256_FIELD = "field:bound-record-sha256"


_STATE_SCHEMA_FIELD = "field:state-variable-schema"


_STATE_LEAF_SCHEMA_FIELD = "field:state-leaf-schema"


_STATE_FACT_FAMILY_FIELD = "field:state-fact-family"


_STATE_ENTITY_KEY_FIELD = "field:state-entity-key"


_STATE_FIELD_PATH_FIELD = "field:state-field-path"


_STATE_VALUE_SCHEMA_FIELD = "field:state-value-schema"


_STATE_FRAME_FIELD = "field:state-frame"


_STATE_UNIT_FIELD = "field:state-unit"


_STATE_TOPOLOGY_FIELD = "field:state-topology"


_PREREQUISITE_FACT_FAMILY_FIELD = "field:prerequisite-fact-family"


_OBJECTIVE_INPUT_SELECTOR_FIELD = "field:objective-input-selector"


_OBJECTIVE_UNIT_FIELD = "field:objective-unit"


_OBJECTIVE_NORMALIZATION_FIELD = "field:objective-normalization"


_PROOF_PAYLOAD_SCHEMA_FIELD = "field:proof-payload-schema"


_PROOF_CHECKER_CAPABILITY_FIELD = "field:proof-checker-capability"


_CLAIM_PROOF_MATERIAL_FIELD = "field:claim-proof-material-definition"


_CLAIM_CHECKER_CAPABILITY_FIELD = "field:claim-checker-capability"


_COMPLETE_DOMAIN_CLAIM_FIELD = "field:complete-domain-claim"


_ROUTING_MATCH_CLAIM_FIELD = "field:routing-match-claim"


_ROUTING_MISMATCH_CLAIM_FIELD = "field:routing-mismatch-claim"


_RESOURCE_ACCOUNTING_CLAIM_FIELD = "field:resource-accounting-claim"


_STATE_METADATA_FIELDS = (
    _STATE_SCHEMA_FIELD,
    _STATE_LEAF_SCHEMA_FIELD,
    _STATE_FACT_FAMILY_FIELD,
    _STATE_ENTITY_KEY_FIELD,
    _STATE_FIELD_PATH_FIELD,
    _STATE_VALUE_SCHEMA_FIELD,
)


_OWNER_REF_ADAPTER = TypeAdapter(OwnerRef)


_CAPABILITY_REF_ADAPTER = TypeAdapter(CapabilityRef)


class DefinitionClosureError(ValueError):
    """A definition, schema, canonical-byte, or typed-value closure failed."""


class SemanticContractError(ValueError):
    """A cross-record semantic, policy, program, or outcome contract failed."""


class ImplementationResolutionError(ValueError):
    """A supplied static owner, build, or capability binding did not resolve."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _ref_key(value: object) -> bytes:
    return canonical_json_bytes(value)


def _state_key(value: StateVariableRef) -> bytes:
    return canonical_json_bytes(
        (
            value.state_schema_ref,
            value.fact_family_ref,
            value.entity_or_fact_key,
            value.field_path_ref,
        )
    )


def _self_digest_matches(model: HashBoundCanonicalModel) -> bool:
    field_name = model.SELF_DIGEST_FIELD
    payload = model.model_dump(
        mode="python",
        by_alias=True,
        exclude={field_name},
        exclude_none=False,
        exclude_defaults=False,
        exclude_unset=False,
        exclude_computed_fields=True,
        round_trip=True,
        # Registry validators intentionally inspect already-decoded hostile
        # wires.  Pydantic's serializer warning is not a semantic verdict and
        # must not prevent the closure checks below from rejecting the wire.
        warnings=False,
    )
    return getattr(model, field_name) == canonical_sha256(
        payload,
        domain=model.HASH_DOMAIN,
    )


@dataclass(frozen=True)
class StaticOwner:
    """One immutable trusted composition row; its object never enters a wire."""

    owner_ref: str
    implementation_build_sha256: str
    capability_refs: tuple[str, ...]
    implementation: object

    def __post_init__(self) -> None:
        try:
            _OWNER_REF_ADAPTER.validate_python(self.owner_ref, strict=True)
        except ValidationError:
            raise ValueError("static owner ref must be canonical")
        if not _is_sha256(self.implementation_build_sha256):
            raise ValueError("static owner build must be an exact sha256")
        if not isinstance(self.capability_refs, tuple):
            raise TypeError("static owner capabilities must be an immutable tuple")
        try:
            validated_capabilities = tuple(
                _CAPABILITY_REF_ADAPTER.validate_python(capability, strict=True)
                for capability in self.capability_refs
            )
        except ValidationError:
            raise ValueError("static owner capabilities must be canonical")
        if any("*" in capability for capability in validated_capabilities):
            raise ValueError("static owner capabilities must be canonical")
        if not validated_capabilities:
            raise ValueError("static owner capabilities must be canonical")
        canonical = tuple(sorted(validated_capabilities, key=_ref_key))
        if validated_capabilities != canonical or len(
            set(validated_capabilities)
        ) != len(validated_capabilities):
            raise ValueError("static owner capabilities must be sorted and unique")
        try:
            hash(self.implementation)
        except TypeError as error:
            raise TypeError("unhashable implementation") from error


# Keep public class, exception and bound-method lookup stable.
DefinitionClosureError.__module__ = "spatialcf.core.registry"
SemanticContractError.__module__ = "spatialcf.core.registry"
ImplementationResolutionError.__module__ = "spatialcf.core.registry"
_is_sha256.__module__ = "spatialcf.core.registry"
_ref_key.__module__ = "spatialcf.core.registry"
_state_key.__module__ = "spatialcf.core.registry"
_self_digest_matches.__module__ = "spatialcf.core.registry"
StaticOwner.__module__ = "spatialcf.core.registry"
