"""Independent validation with record imports free of dataset I/O machinery."""

from importlib import import_module

__all__ = [
    "DatasetWriter",
    "FailureRecord",
    "PairRecord",
    "VerificationResult",
    "Verifier",
    "audit_dataset",
    "read_authenticated_dataset_identity",
]

_EXPORTS = {
    "DatasetWriter": "dataset",
    "FailureRecord": "dataset_models",
    "PairRecord": "dataset_models",
    "VerificationResult": "verifier",
    "Verifier": "verifier",
    "audit_dataset": "dataset_audit",
    "read_authenticated_dataset_identity": "dataset_reader",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
