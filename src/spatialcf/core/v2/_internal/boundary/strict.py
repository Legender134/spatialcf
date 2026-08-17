"""Exact-type strict rebuild helpers for Canonical v2 boundaries."""

from __future__ import annotations

import warnings
from typing import TypeVar

from pydantic import ValidationError

from spatialcf.core.v2._internal.boundary.errors import (
    InvalidCallerInputV2,
    NumericBoundaryGapV2,
)
from spatialcf.domain.v2.base import V2Model
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    ProvenUnsatResultV2,
    UncertifiedResultV2,
)

ModelT = TypeVar("ModelT", bound=V2Model)
_RESULT_TYPES = (
    CertifiedSuccessResultV2,
    ProvenUnsatResultV2,
    UncertifiedResultV2,
)


def strict_input_model_v2(
    value: object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    """Strictly rebuild one exact caller-owned model."""

    if type(value) is not model_type:
        raise InvalidCallerInputV2(f"INVALID_INPUT:{label}:TYPE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return model_type.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise NumericBoundaryGapV2(f"NUMERIC_GAP:{label}_REVALIDATION") from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise InvalidCallerInputV2(f"INVALID_INPUT:{label}") from error


def strict_submitted_solve_result_v2(value: object) -> V2Model:
    """Strictly rebuild one caller-submitted exact solve result."""

    result_type = type(value)
    if result_type not in _RESULT_TYPES:
        raise InvalidCallerInputV2("INVALID_SUBMITTED_RESULT:TYPE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return result_type.model_validate(
                value.model_dump(mode="python"),
                strict=True,
            )
    except (ArithmeticError, RuntimeWarning) as error:
        raise NumericBoundaryGapV2(
            "NUMERIC_GAP:SUBMITTED_RESULT_REVALIDATION"
        ) from error
    except (ValidationError, TypeError, ValueError, Warning) as error:
        raise InvalidCallerInputV2("INVALID_SUBMITTED_RESULT") from error


def strict_fresh_solve_result_v2(value: object) -> V2Model:
    """Strictly rebuild an internal result without hiding invariants."""

    result_type = type(value)
    if result_type not in _RESULT_TYPES:
        raise TypeError("fresh result has the wrong exact type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        return result_type.model_validate(
            value.model_dump(mode="python"),
            strict=True,
        )
