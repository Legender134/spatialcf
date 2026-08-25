"""Shared implementation for capability-local domain-operation budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(slots=True)
class DomainOperationBudgetV2:
    """Mutable exact counter with capability-owned exhaustion semantics."""

    limit: int
    used: int = 0

    _exhaustion_error_type: ClassVar[type[RuntimeError]] = RuntimeError

    def consume(self, amount: int = 1) -> None:
        if type(amount) is not int or amount < 0:
            raise TypeError("domain operation amount must be a non-negative exact int")
        if self.used + amount > self.limit:
            error_type = type(self)._exhaustion_error_type
            if type(error_type) is not type or not issubclass(error_type, RuntimeError):
                raise TypeError(
                    "domain-operation exhaustion type must be a RuntimeError type"
                )
            raise error_type
        self.used += amount


@dataclass(slots=True)
class RemainingDomainOperationBudgetV2:
    """Mutable single-step counter represented by remaining and new usage."""

    remaining: int
    used: int = 0

    _exhaustion_error_type: ClassVar[type[RuntimeError]] = RuntimeError
    _exhaustion_error_message: ClassVar[str] = "domain operation budget exhausted"

    def consume(self) -> None:
        if self.remaining <= 0:
            error_type = type(self)._exhaustion_error_type
            if type(error_type) is not type or not issubclass(error_type, RuntimeError):
                raise TypeError(
                    "domain-operation exhaustion type must be a RuntimeError type"
                )
            message = type(self)._exhaustion_error_message
            if type(message) is not str or not message:
                raise TypeError(
                    "domain-operation exhaustion message must be a non-empty exact str"
                )
            raise error_type(message)
        self.remaining -= 1
        self.used += 1


@dataclass(slots=True)
class BaseUsageDomainOperationBudgetV2:
    """Strict absolute cap over frozen base usage plus newly consumed work."""

    limit: int
    base_used: int
    used: int = 0

    _exhaustion_error_type: ClassVar[type[RuntimeError]] = RuntimeError
    _exhaustion_error_message: ClassVar[str] = "domain operation budget exhausted"

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise TypeError("domain-operation limit must be a positive exact int")
        if (
            type(self.base_used) is not int
            or self.base_used < 0
            or self.base_used > self.limit
        ):
            raise ValueError("base domain-operation usage is outside the limit")
        if type(self.used) is not int or self.used != 0:
            raise ValueError("new domain-operation usage must start at zero")

    def consume(self, amount: int = 1) -> None:
        if type(amount) is not int or amount < 0:
            raise TypeError("domain-operation amount must be a non-negative exact int")
        if self.base_used + self.used + amount > self.limit:
            error_type = type(self)._exhaustion_error_type
            if type(error_type) is not type or not issubclass(error_type, RuntimeError):
                raise TypeError(
                    "domain-operation exhaustion type must be a RuntimeError type"
                )
            message = type(self)._exhaustion_error_message
            if type(message) is not str or not message:
                raise TypeError(
                    "domain-operation exhaustion message must be a non-empty exact str"
                )
            raise error_type(message)
        self.used += amount


@dataclass(slots=True)
class ValidatedLiveDomainOperationBudgetV2:
    """Strict mutable absolute ledger revalidated before every consumption."""

    limit: int
    used: int = 0

    _exhaustion_error_type: ClassVar[type[RuntimeError]] = RuntimeError

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise TypeError("domain-operation limit must be a positive exact int")
        if type(self.used) is not int:
            raise TypeError("domain-operation usage must be an exact int")
        if self.used < 0 or self.used > self.limit:
            raise ValueError("domain-operation usage must be within its limit")

    def consume(self, amount: int = 1) -> None:
        self.validate()
        if type(amount) is not int or amount < 0:
            raise TypeError("domain-operation amount must be a non-negative exact int")
        if self.used + amount > self.limit:
            error_type = type(self)._exhaustion_error_type
            if type(error_type) is not type or not issubclass(error_type, RuntimeError):
                raise TypeError(
                    "domain-operation exhaustion type must be a RuntimeError type"
                )
            raise error_type
        self.used += amount


__all__ = ()
