# Python API

The supported version-free API is exported from `spatialcf.generation`.

## Generate

```python
from pathlib import Path

from spatialcf.generation import generate_dataset

report = generate_dataset(
    config=Path("configs/ai2thor-example.toml"),
    output=Path("dataset"),
)
print(report.model_dump_json(indent=2))
```

`config` accepts a validated `GenerationConfig` or a path to its TOML
representation. `output` is the dataset root. The optional `adapter_factory`
keyword is available for Adapter implementations and controlled tests.

## Verify

```python
from pathlib import Path

from spatialcf.generation import verify_dataset

report = verify_dataset(Path("dataset"))
```

`verify_dataset` performs fresh verification and returns a `GenerationReport`.
It does not modify the dataset.

## Inspect

```python
from pathlib import Path

from spatialcf.generation import inspect_dataset

summary = inspect_dataset(Path("dataset"))
```

`inspect_dataset` verifies first, then returns the dataset digest, record count,
relation counts, and generation report as JSON-compatible values.

## Records and models

`read_dataset_records(Path("dataset"))` reads the canonical accepted-record
roster under a retained dataset descriptor. The package also exports
`DatasetManifest`, `DatasetRecord`, `GenerationConfig`, `GenerationReport`, and
`UnsupportedArtifactVersion`. Public callers should import these names from
`spatialcf.generation`.
