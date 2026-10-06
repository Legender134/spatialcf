"""Version-free public boundary for the current generation pipeline. Lazy exports avoid loading unrelated execution owners."""

from importlib import import_module

__all__ = (
    "DatasetManifest",
    "DatasetRecord",
    "GenerationConfig",
    "GenerationReport",
    "UnsupportedArtifactVersion",
    "generate_dataset",
    "inspect_dataset",
    "load_generation_config",
    "read_dataset_records",
    "verify_dataset",
)

_EXPORTS = {
    "GenerationConfig": ("spatialcf.generation.config", "GenerationConfig"),
    "load_generation_config": ("spatialcf.generation.config", "load_generation_config"),
    "DatasetManifest": ("spatialcf.generation.dataset", "DatasetManifest"),
    "DatasetRecord": ("spatialcf.generation.dataset", "DatasetRecord"),
    "GenerationReport": ("spatialcf.generation.dataset", "GenerationReport"),
    "generate_dataset": ("spatialcf.generation.dataset", "generate_dataset"),
    "inspect_dataset": ("spatialcf.generation.dataset", "inspect_dataset"),
    "read_dataset_records": ("spatialcf.generation.dataset", "read_dataset_records"),
    "verify_dataset": ("spatialcf.generation.dataset", "verify_dataset"),
    "UnsupportedArtifactVersion": (
        "spatialcf.generation.errors",
        "UnsupportedArtifactVersion",
    ),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
