"""Deterministic canonical JSON bytes and domain-separated hashes for v2.

Canonical v2 freezes finite float spelling to CPython 3.11's ``json`` encoder.
The golden vectors in ``test_serialization.py`` are the language-neutral wire
contract; another implementation must reproduce them before sharing this hash
prefix. This is intentionally narrower than claiming RFC 8785 compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter

from spatialcf.domain.v2.base import CanonicalId, Sha256Digest, V2Model

CANONICAL_HASH_PREFIX_V2 = b"spatialcf-canonical-json-v2\0"
CANONICAL_JSON_NUMBER_GRAMMAR_V2 = "python-json-3.11-finite-v1"
_CANONICAL_ID_ADAPTER = TypeAdapter(CanonicalId)


def _canonical_json_value_v2(value: Any) -> Any:
    if isinstance(value, V2Model):
        validated = type(value).model_validate(value, strict=True)
        return _canonical_json_value_v2(
            validated.model_dump(
                mode="python",
                by_alias=True,
                exclude_none=False,
                exclude_defaults=False,
                exclude_unset=False,
                exclude_computed_fields=True,
                round_trip=True,
            )
        )
    if isinstance(value, BaseModel):
        raise TypeError("non-v2 Pydantic models cannot enter canonical v2 JSON")
    if isinstance(value, Enum):
        return _canonical_json_value_v2(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: _canonical_json_value_v2(item) for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value_v2(item) for item in value]
    if isinstance(value, AbstractSet):
        normalized = [_canonical_json_value_v2(item) for item in value]
        return sorted(normalized, key=_encoded_sort_key)
    raise TypeError(
        f"value of type {type(value).__name__!r} is not representable as canonical JSON"
    )


def _encoded_sort_key(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_bytes_v2(value: Any) -> bytes:
    """Return compact UTF-8 JSON using the frozen v2 number grammar."""

    normalized = _canonical_json_value_v2(value)
    return _encoded_sort_key(normalized)


def canonical_sha256_v2(value: Any, *, domain: CanonicalId) -> Sha256Digest:
    """Hash canonical bytes with a required semantic domain separator."""

    validated_domain = _CANONICAL_ID_ADAPTER.validate_python(domain, strict=True)
    digest_input = (
        CANONICAL_HASH_PREFIX_V2
        + validated_domain.encode("utf-8")
        + b"\0"
        + canonical_json_bytes_v2(value)
    )
    return hashlib.sha256(digest_input).hexdigest()
