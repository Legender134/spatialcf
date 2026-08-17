"""Freshly verify a generated dataset with the public Python API."""

from pathlib import Path

from spatialcf.generation import verify_dataset

report = verify_dataset(Path("dataset"))
print(report.model_dump_json(indent=2))
