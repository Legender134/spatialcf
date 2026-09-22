"""Stable advanced facade for semantic contrast datasets."""

from __future__ import annotations

from pathlib import Path

from spatialcf.domain.contrast import SemanticContrastReport
from spatialcf.generation.workflows.dataset import (
    generate_semantic_contrast_dataset as _generate,
)
from spatialcf.generation.workflows.dataset import (
    verify_semantic_contrast_dataset as _verify,
)

__all__ = (
    "generate_semantic_contrast_dataset",
    "verify_semantic_contrast_dataset",
)


def generate_semantic_contrast_dataset(
    catalog: Path, output: Path
) -> SemanticContrastReport:
    return _generate(catalog, output)


def verify_semantic_contrast_dataset(root: Path) -> SemanticContrastReport:
    return _verify(root)
