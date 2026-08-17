# Quick start

`v0.1.1` is a GitHub release, not a PyPI publication. Clone it and run the
complete workflow from the local checkout:

```bash
git clone --branch v0.1.1 --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[ai2thor]"
spatialcf generate --config configs/ai2thor-example.toml --output ./dataset
spatialcf verify ./dataset
spatialcf inspect ./dataset
```

The example selects the Unity/AI2-THOR Adapter and `FloorPlan2`. It freezes at
most 12 requests using the declared seed. Accepted requests are written as
records; rejected requests remain visible in the report without being promoted
to accepted data.

The output directory must not already contain a published dataset. If generation
is interrupted before publication, rerun the same command with the same
configuration and output path. SpatialCF verifies reusable stage state before
resuming and never changes the frozen request roster.

`verify` performs a fresh, read-only check of metadata, records, assets, stage
state, and checksums. `inspect` performs that same verification before printing a
compact count and relation summary.
