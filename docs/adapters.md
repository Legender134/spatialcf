# Adapters

Adapters are the boundary between the platform-neutral generation system and a
scene runtime. An Adapter provides source-scene loading, normalization, camera
control, settled observation capture, native placement queries, edit execution,
runtime identity, and asset extraction.

The solver owns spatial semantics, feasible-domain construction, scoring, and
selection. An Adapter does not define labels or search for a successful edit by
repeated native execution.

## Unity/AI2-THOR

Unity/AI2-THOR is the first supported Adapter. `v0.1.1` is a historical annotated tag,
not a PyPI publication, so install the Adapter from a local checkout of that
release:

```bash
git clone --branch v0.1.1 --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[ai2thor]"
```

Select it in TOML with `adapter = "ai2thor"` and provide one or more exact scene
names. `width`, `height`, and `seed` control capture dimensions and deterministic
request construction. See `configs/ai2thor-example.toml` for the complete public
configuration.

Adapter or runtime failures are terminal outcomes for their frozen requests; they
do not cause result-dependent candidate ordering or replacement requests.

The Adapter boundary is designed so another simulator or a real-world capture
system can reuse the same Schema, solver, dataset index, and verification flow.

## Generate and verify an Adapter dataset

After installing and configuring the native runtime separately:

```bash
spatialcf generate --config configs/ai2thor-example.toml --output ./dataset
spatialcf verify ./dataset
spatialcf inspect ./dataset
```

This configuration selects `FloorPlan2`; it requires a working native build and
graphics environment. The CPU quick start does not run it. Adapter output uses
`manifest.json`, `records.jsonl`, `report.json`, `checksums.sha256`, `assets/`
and resumable `.spatialcf/` state. Use these commands, not `general verify`, for
that format. Native failures remain recorded outcomes; they are not evidence of
geometric UNSAT.
