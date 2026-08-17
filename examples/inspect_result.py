"""Inspect a verified dataset with the public Python API."""

import json
from pathlib import Path

from spatialcf.generation import inspect_dataset

summary = inspect_dataset(Path("dataset"))
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
