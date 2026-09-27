"""Exact typed artifact parsing shared by structural dataset readers."""

from __future__ import annotations

from spatialcf.domain.definitions import HashBoundCanonicalModel
from spatialcf.domain.serialization import canonical_json_bytes


def _parse_exact(payload: bytes, model_type: type[HashBoundCanonicalModel]):
    try:
        value = model_type.model_validate_json(payload, strict=False)
    except Exception as error:
        raise ValueError(f"invalid {model_type.__name__} JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical {model_type.__name__} bytes")
    return model_type.model_validate(
        value.model_dump(mode="python", warnings="error", round_trip=True),
        strict=True,
    )


_parse_exact.__module__ = "spatialcf.verification.contrast"
