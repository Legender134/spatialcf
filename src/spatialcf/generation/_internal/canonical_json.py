"""Canonical JSON helpers for current generation artifacts."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


def canonical_value(value: Any) -> Any:
    """Convert model values into a deterministic JSON-compatible tree."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(
            (canonical_value(item) for item in value),
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON with the repository's immutable canonical encoding."""
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(canonical_value(value), **options) + "\n").encode("utf-8")
