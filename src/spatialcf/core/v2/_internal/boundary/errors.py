"""Typed errors for strict caller-owned Canonical v2 boundaries."""


class InvalidCallerInputV2(RuntimeError):
    """A caller-owned value failed exact-type or strict validation."""

    def __init__(self, finding_code: str) -> None:
        if type(finding_code) is not str or not finding_code.strip():
            raise TypeError("finding_code must be an exact non-blank string")
        self.finding_code = finding_code
        super().__init__(finding_code)


class NumericBoundaryGapV2(RuntimeError):
    """A designated caller-boundary numeric operation could not be certified."""

    def __init__(self, finding_code: str) -> None:
        if type(finding_code) is not str or not finding_code.strip():
            raise TypeError("finding_code must be an exact non-blank string")
        self.finding_code = finding_code
        super().__init__(finding_code)
