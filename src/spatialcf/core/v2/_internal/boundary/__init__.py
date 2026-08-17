"""Strict private boundaries shared by Canonical v2 entry points."""

from spatialcf.core.v2._internal.boundary.errors import (
    InvalidCallerInputV2,
    NumericBoundaryGapV2,
)
from spatialcf.core.v2._internal.boundary.strict import (
    strict_fresh_solve_result_v2,
    strict_input_model_v2,
    strict_submitted_solve_result_v2,
)

__all__ = (
    "InvalidCallerInputV2",
    "NumericBoundaryGapV2",
    "strict_fresh_solve_result_v2",
    "strict_input_model_v2",
    "strict_submitted_solve_result_v2",
)
