"""Stateless SHA-256 sampling on the frozen stress grids."""

from __future__ import annotations

from decimal import Decimal
import hashlib
from typing import TypeVar

T = TypeVar("T")


class HashSampler:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace

    def integer(self, label: str, minimum: int, maximum: int) -> int:
        payload = f"{self.namespace}\0{label}".encode("utf-8")
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return minimum + raw % (maximum - minimum + 1)

    def choice(self, label: str, values: tuple[T, ...]) -> T:
        return values[self.integer(label, 0, len(values) - 1)]

    def grid(
        self,
        label: str,
        start: Decimal,
        stop: Decimal,
        step: Decimal,
    ) -> float:
        count = int((stop - start) / step)
        return float(start + step * self.integer(label, 0, count))
