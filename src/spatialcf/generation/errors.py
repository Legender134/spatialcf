from __future__ import annotations

import json
from typing import TypeVar

T = TypeVar("T")


class UnsupportedArtifactVersion(ValueError):
    """Raised when current code is asked to open a historical artifact."""

    def __init__(
        self,
        artifact_kind: str,
        *,
        expected: str,
        observed: object,
    ) -> None:
        self.artifact_kind = artifact_kind
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"{artifact_kind} version {observed!r} is unsupported; "
            f"expected {expected!r}"
        )


def require_exact_type(value: object, expected_type: type[T], *, label: str) -> T:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be exact {expected_type.__name__}")
    return value


def require_wire_version(
    payload: bytes,
    *,
    artifact_kind: str,
    field: str,
    expected: str,
) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{artifact_kind} is not valid JSON") from error
    if type(decoded) is not dict:
        raise ValueError(f"{artifact_kind} must be a JSON object")
    observed = decoded.get(field)
    if observed != expected:
        raise UnsupportedArtifactVersion(
            artifact_kind,
            expected=expected,
            observed=observed,
        )
    return decoded


__all__ = (
    "UnsupportedArtifactVersion",
    "require_exact_type",
    "require_wire_version",
)
