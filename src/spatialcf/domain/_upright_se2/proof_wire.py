"""Upright profile proof wire contracts and intrinsic operations."""

from __future__ import annotations

from enum import (
    StrEnum,
)

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalIdValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntegerValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF,
    UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR,
    UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
    UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF,
    UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR,
    UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.proof_material import (
    UprightSE2ContinuousProofMaterial,
    UprightSE2ProofMaterial,
)

from spatialcf.domain._upright_se2.yaw import (
    ExplicitPoseYawBinding,
    _require_sorted_unique_by_bytes,
)


def _proof_wire_record(
    fields: tuple[tuple[CanonicalId, TypedValue], ...],
) -> TypedValue:
    """Build one canonical record using the sole public proof payload schema."""

    return TypedValue(
        value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (NamedTypedValue(name=name, value=value) for name, value in fields),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


def _proof_wire_encode_node(value: object) -> TypedValue:
    """Encode every canonical proof leaf structurally under one schema identity."""

    if isinstance(value, StrEnum):
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=CanonicalIdValue(value=value.value),
        )
    if value is None:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=EnumSymbolValue(symbol="NULL"),
        )
    if type(value) is bool:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=BooleanValue(value=value),
        )
    if type(value) is int:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=IntegerValue(value=value),
        )
    if type(value) is float:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteRealValue(value=value),
        )
    if type(value) is str:
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=CanonicalIdValue(value=value),
        )
    if type(value) in (tuple, list):
        return TypedValue(
            value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteOrderedTupleValue(
                element_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                items=tuple(_proof_wire_encode_node(item) for item in value),
            ),
        )
    if type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError(
                "proof payload records require canonical string field names"
            )
        return _proof_wire_record(
            tuple((name, _proof_wire_encode_node(item)) for name, item in value.items())
        )
    raise TypeError(f"proof payload cannot encode {type(value).__name__}")


def _proof_wire_decode_node(value: TypedValue) -> object:
    """Decode one structural proof node while rejecting other schema families."""

    if value.value_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
        raise ValueError("proof payload node has the wrong schema")
    payload = value.payload
    if type(payload) is CanonicalIdValue:
        return payload.value
    if type(payload) is BooleanValue:
        return payload.value
    if type(payload) is IntegerValue:
        return payload.value
    if type(payload) is FiniteRealValue:
        return payload.value
    if type(payload) is EnumSymbolValue:
        if payload.symbol != "NULL":
            raise ValueError("proof payload has an unknown scalar discriminator")
        return None
    if type(payload) is FiniteOrderedTupleValue:
        if payload.element_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
            raise ValueError("proof payload tuple has the wrong element schema")
        return tuple(_proof_wire_decode_node(item) for item in payload.items)
    if type(payload) is RecordValue:
        return {
            field.name: _proof_wire_decode_node(field.value) for field in payload.fields
        }
    raise ValueError("proof payload must be structural rather than a digest-only value")


def encode_upright_se2_proof_material(material: UprightSE2ProofMaterial) -> TypedValue:
    """Encode one sealed full cardinal proof into the sole M3 typed payload wire."""

    if type(material) is not UprightSE2ProofMaterial:
        raise TypeError("proof material codec requires UprightSE2ProofMaterial")
    checked = UprightSE2ProofMaterial.model_validate(
        material.model_dump(mode="python", round_trip=True),
        strict=True,
    )
    return _proof_wire_record(
        (
            (
                "definition_ref",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=CanonicalIdValue(
                        value=UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
                    ),
                ),
            ),
            (
                "discriminator",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=EnumSymbolValue(
                        symbol=UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR
                    ),
                ),
            ),
            (
                "proof_material",
                _proof_wire_encode_node(
                    checked.model_dump(mode="python", round_trip=True)
                ),
            ),
        )
    )


def decode_upright_se2_proof_material(value: TypedValue) -> UprightSE2ProofMaterial:
    """Decode the one full canonical proof payload and reject all substitutions."""

    if value.value_schema_ref != UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF:
        raise ValueError("proof payload has the wrong schema")
    if type(value.payload) is not RecordValue:
        raise ValueError("proof payload must use its canonical record")
    fields = {field.name: field.value for field in value.payload.fields}
    if set(fields) != {"definition_ref", "discriminator", "proof_material"}:
        raise ValueError("proof payload record has missing or unknown fields")
    definition = fields["definition_ref"]
    if (
        type(definition.payload) is not CanonicalIdValue
        or definition.payload.value != UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
    ):
        raise ValueError("proof payload has the wrong definition identity")
    discriminator = fields["discriminator"]
    if (
        type(discriminator.payload) is not EnumSymbolValue
        or discriminator.payload.symbol != UPRIGHT_SE2_PROOF_MATERIAL_DISCRIMINATOR
    ):
        raise ValueError("proof payload has an unknown discriminator")
    raw = _proof_wire_decode_node(fields["proof_material"])
    if type(raw) is not dict:
        raise ValueError("proof payload material must be a structural record")
    try:
        # The wire carries enum members as canonical scalar symbols; strict
        # reconstruction would incorrectly demand Python enum instances.  The
        # exact re-encode comparison below still rejects every coercive or
        # non-canonical representation.
        material = UprightSE2ProofMaterial.model_validate(raw, strict=False)
    except Exception as error:
        raise ValueError(
            "proof payload does not decode to sealed proof material"
        ) from error
    if canonical_json_bytes(
        encode_upright_se2_proof_material(material)
    ) != canonical_json_bytes(value):
        raise ValueError("proof payload is not the canonical complete representation")
    return material


def _continuous_proof_wire_record(
    fields: tuple[tuple[CanonicalId, TypedValue], ...],
) -> TypedValue:
    """Encode an additive continuous payload without touching cardinal codec."""

    # ``encode_upright_se2_continuous_proof_material`` has already rebuilt its
    # source material through the exact sealed model.  Constructing this
    # recursive, structural wire from those checked scalar/tuple/record values
    # must not repeatedly re-run Pydantic's recursive payload-union validation:
    # a real proposal embeds its compiler-materialized endpoint and otherwise
    # expands that same checked tree at every nested TypedValue boundary.
    #
    # These constructions retain the schema tags and canonical named-field
    # ordering explicitly.  They do not accept caller-provided wire nodes, and
    # the public decoder still reconstructs and re-encodes through the sealed
    # continuous proof model before accepting any payload.
    return TypedValue.model_construct(
        value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=RecordValue.model_construct(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue.model_construct(name=name, value=value)
                        for name, value in fields
                    ),
                    key=lambda field: canonical_json_bytes(field.name),
                )
            )
        ),
    )


def _continuous_proof_wire_encode_node(value: object) -> TypedValue:
    """Structurally encode only the additive continuous proof schema."""

    if isinstance(value, StrEnum):
        payload = CanonicalIdValue.model_construct(value=value.value)
    elif value is None:
        payload = EnumSymbolValue.model_construct(symbol="NULL")
    elif type(value) is bool:
        payload = BooleanValue.model_construct(value=value)
    elif type(value) is int:
        payload = IntegerValue.model_construct(value=value)
    elif type(value) is float:
        payload = FiniteRealValue.model_construct(value=value)
    elif type(value) is str:
        payload = CanonicalIdValue.model_construct(value=value)
    elif type(value) in (tuple, list):
        return TypedValue.model_construct(
            value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
            payload=FiniteOrderedTupleValue.model_construct(
                element_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                items=tuple(_continuous_proof_wire_encode_node(item) for item in value),
            ),
        )
    elif type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError("continuous proof payload records require string names")
        return _continuous_proof_wire_record(
            tuple(
                (name, _continuous_proof_wire_encode_node(item))
                for name, item in value.items()
            )
        )
    else:
        raise TypeError(
            f"continuous proof payload cannot encode {type(value).__name__}"
        )
    return TypedValue.model_construct(
        value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        payload=payload,
    )


def _continuous_proof_wire_decode_node(value: TypedValue) -> object:
    if (
        value.value_schema_ref
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
    ):
        raise ValueError("continuous proof payload node has the wrong schema")
    payload = value.payload
    if type(payload) is CanonicalIdValue:
        return payload.value
    if type(payload) is BooleanValue:
        return payload.value
    if type(payload) is IntegerValue:
        return payload.value
    if type(payload) is FiniteRealValue:
        return payload.value
    if type(payload) is EnumSymbolValue:
        if payload.symbol != "NULL":
            raise ValueError("continuous proof payload has an unknown scalar")
        return None
    if type(payload) is FiniteOrderedTupleValue:
        if (
            payload.element_schema_ref
            != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
        ):
            raise ValueError("continuous proof payload tuple has wrong schema")
        return tuple(_continuous_proof_wire_decode_node(item) for item in payload.items)
    if type(payload) is RecordValue:
        return {
            field.name: _continuous_proof_wire_decode_node(field.value)
            for field in payload.fields
        }
    raise ValueError("continuous proof payload must be structural")


def encode_upright_se2_continuous_proof_material(
    material: UprightSE2ContinuousProofMaterial,
) -> TypedValue:
    """Encode the independent continuous proof payload wire."""

    if type(material) is not UprightSE2ContinuousProofMaterial:
        raise TypeError("continuous proof codec requires continuous proof material")
    checked = UprightSE2ContinuousProofMaterial.model_validate(
        material.model_dump(mode="python", round_trip=True), strict=True
    )
    return _continuous_proof_wire_record(
        (
            (
                "definition_ref",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=CanonicalIdValue(
                        value=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
                    ),
                ),
            ),
            (
                "discriminator",
                TypedValue(
                    value_schema_ref=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
                    payload=EnumSymbolValue(
                        symbol=UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR
                    ),
                ),
            ),
            (
                "proof_material",
                _continuous_proof_wire_encode_node(
                    checked.model_dump(mode="python", round_trip=True)
                ),
            ),
        )
    )


def decode_upright_se2_continuous_proof_material(
    value: TypedValue,
) -> UprightSE2ContinuousProofMaterial:
    """Decode only the additive continuous proof discriminator/schema pair."""

    if (
        value.value_schema_ref
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
    ):
        raise ValueError("continuous proof payload has the wrong schema")
    if type(value.payload) is not RecordValue:
        raise ValueError("continuous proof payload must use a record")
    fields = {field.name: field.value for field in value.payload.fields}
    if set(fields) != {"definition_ref", "discriminator", "proof_material"}:
        raise ValueError("continuous proof payload has missing or unknown fields")
    if (
        type(fields["definition_ref"].payload) is not CanonicalIdValue
        or fields["definition_ref"].payload.value
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
    ):
        raise ValueError("continuous proof payload has wrong definition identity")
    if (
        type(fields["discriminator"].payload) is not EnumSymbolValue
        or fields["discriminator"].payload.symbol
        != UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DISCRIMINATOR
    ):
        raise ValueError("continuous proof payload has wrong discriminator")
    raw = _continuous_proof_wire_decode_node(fields["proof_material"])
    if type(raw) is not dict:
        raise ValueError("continuous proof material must be a structural record")
    try:
        material = UprightSE2ContinuousProofMaterial.model_validate(raw, strict=False)
    except Exception as error:
        raise ValueError(
            "continuous proof payload does not decode to sealed material"
        ) from error
    if canonical_json_bytes(
        encode_upright_se2_continuous_proof_material(material)
    ) != canonical_json_bytes(value):
        raise ValueError("continuous proof payload is not canonical")
    return material


class UprightSE2VerificationBundle(HashBoundCanonicalModel):
    """A closed profile/availability/pose/proof bundle for later checker replay."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/verification-bundle/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "verification_bundle_sha256"

    profile_registration_sha256: Sha256Digest
    backend_availability_sha256: Sha256Digest
    pose_yaw_bindings: tuple[ExplicitPoseYawBinding, ...]
    proof_material: UprightSE2ProofMaterial
    verification_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_pose_binding_identity(self) -> Self:
        _require_sorted_unique_by_bytes(self.pose_yaw_bindings, "pose/yaw bindings")
        entity_ids = tuple(binding.entity_id for binding in self.pose_yaw_bindings)
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError(
                "verification bundle must reject same entity ID with different bytes"
            )
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2VerificationBundle.model_rebuild()


# Keep supported public import and pickle lookup stable.
_proof_wire_record.__module__ = "spatialcf.domain.upright_se2"
_proof_wire_encode_node.__module__ = "spatialcf.domain.upright_se2"
_proof_wire_decode_node.__module__ = "spatialcf.domain.upright_se2"
encode_upright_se2_proof_material.__module__ = "spatialcf.domain.upright_se2"
decode_upright_se2_proof_material.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_wire_record.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_wire_encode_node.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_wire_decode_node.__module__ = "spatialcf.domain.upright_se2"
encode_upright_se2_continuous_proof_material.__module__ = "spatialcf.domain.upright_se2"
decode_upright_se2_continuous_proof_material.__module__ = "spatialcf.domain.upright_se2"
UprightSE2VerificationBundle.__module__ = "spatialcf.domain.upright_se2"
