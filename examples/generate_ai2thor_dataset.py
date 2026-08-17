"""Generate the quick-start AI2-THOR dataset with the public Python API."""

from pathlib import Path

from spatialcf.generation import generate_dataset

report = generate_dataset(
    config=Path("configs/ai2thor-example.toml"),
    output=Path("dataset"),
)
print(report.model_dump_json(indent=2))
