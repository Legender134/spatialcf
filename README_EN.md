[简体中文](README.md) | [English](README_EN.md)

# SpatialCF

SpatialCF generates verified spatial counterfactual datasets. It freezes requests
from scene observations, plans single-object planar moves with a minimum-cost
solver, executes edits through an Adapter, and freshly verifies the results and
dataset files.

The Schema, solver, and verification logic are platform-neutral. Unity/AI2-THOR
is the first Adapter and connects platform facts and native operations to the
public generation chain.

## Quick start

Python 3.11 is required. These commands create an environment, install the
AI2-THOR Adapter, generate a dataset, and reopen it for verification and
inspection:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "spatialcf[ai2thor]"
spatialcf generate --config configs/ai2thor-example.toml --output ./dataset
spatialcf verify ./dataset
spatialcf inspect ./dataset
```

`generate` never silently replaces a published dataset. `verify` rereads the
metadata, records, assets, and checksums; `inspect` returns a summary only after
full verification succeeds.

## Dataset contents

The generated directory contains `manifest.json`, `records.jsonl`, `report.json`,
`checksums.sha256`, content-addressed `assets/`, and resumable `.spatialcf/`
state. Rejected requests appear only in report counts and never become accepted
records.

## Documentation

- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Concepts](docs/concepts.md)
- [Adapters](docs/adapters.md)
- [Python API](docs/api.md)

SpatialCF is licensed under the [Apache License 2.0](LICENSE).
