"""Version-free public boundary for the current generation pipeline."""

from spatialcf.generation.config import GenerationConfig, load_generation_config
from spatialcf.generation.dataset import (
    DatasetManifest,
    DatasetRecord,
    GenerationReport,
    generate_dataset,
    inspect_dataset,
    read_dataset_records,
    verify_dataset,
)
from spatialcf.generation.errors import UnsupportedArtifactVersion

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
