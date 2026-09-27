[简体中文](README.md) | [English](README_EN.md)

# SpatialCF

SpatialCF generates **independently verifiable spatial counterfactual data**. Supply scene facts, a goal and authorized edits; the system searches for a false-to-true change and retains replayable evidence. Only certified solutions become before/after pairs. Proven infeasibility, unresolved requests and noncertified feasible witnesses remain distinct outcomes.

## Run a CPU example

Use Python 3.11 and current `main`. The base package needs no Unity, AI2-THOR or GPU. These explicit synthetic facts produce semantic scenes and proofs, not rendered images.

```bash
git clone --branch main --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/general_dataset.py input.json --case placement
spatialcf general generate --input input.json --output dataset
spatialcf general verify dataset
spatialcf general inspect dataset
```

Use new input/output paths. This case places a box from the floor onto a table and certifies its endpoint and minimum displacement. `verify` rereads the dataset and checks retained proofs; `inspect` performs the same verification before summarizing. Complex cases can take minutes.

Continue with the [CPU quick start](docs/quickstart.md) for multi-body, infeasible and unresolved cases and output interpretation.

## Capabilities

| Task | Interface and scope |
| --- | --- |
| Move one object to flip left/right/front/behind/near/far | v2/M2 world XY; fixed height and orientation; existing generation contracts |
| Translate and turn around own/reference pivots | M3 upright SE(2); certified closed subdomains, otherwise UNKNOWN or witnesses |
| Retain semantic pairs and source lineage | M4 semantic contrast; explicit requests and budgets; no rendered native output |
| Place on supports or inside cavities | M5 PLACE_ON/PLACE_IN; exact axis-aligned boxes, horizontal supports, explicit rectangular cavities |
| Edit multiple bodies, 3D poses, joints and contacts | M6 finite exact box models, joint forests, ordered edit sets, prefix and commutativity checks |
| Batch, export and independently verify M5/M6 data | `spatialcf general generate/verify/inspect`; only certified pairs are published |

Current chain: domain/core → adapter protocol → generation → fresh verification. The core owns solving and proofs; adapters translate platform facts; generation and verification consume versioned artifacts.

Proofs cover the declared model and authorized domain. M6 global optimality/UNSAT requires exhaustive finite-universe coverage and independent checking. Unclosed continuous domains, exhausted budgets and weak backend claims cannot establish UNSAT. A feasible witness does not establish optimality.

Swept paths, dynamics, physical stability, arbitrary meshes and closed-loop joints are outside this scope. CPU facts are not authenticated native observations. Schema, solvers and verification are platform-neutral; the Unity/AI2-THOR Adapter connects the native runtime.

## Outputs and routes

A general dataset has seven files: `input.json`, `catalog.json`, `terminals.json`, `records.json`, `report.json`, `provenance.json`, `manifest.json`. Every candidate retains its terminal outcome; only `PUBLISHED_PAIR` contributes a pair.

The existing `spatialcf generate/verify/inspect` commands use the separate Adapter dataset format. See [Adapters](docs/adapters.md); use the verifier matching the dataset route.

## Version and documentation

The user repository is [Legender134/spatialcf](https://github.com/Legender134/spatialcf). Current `main` contains these features. `v0.1.1` is a historical annotated tag without all new interfaces, not a PyPI publication. Package metadata remains `0.1.1`; record `git rev-parse HEAD` for reproducibility. This workflow creates no new tag or GitHub Release.

- [Installation](docs/installation.md)
- [Quick start and outcome interpretation](docs/quickstart.md)
- [Concepts and proof limits](docs/concepts.md)
- [Adapters](docs/adapters.md)
- [Python API](docs/api.md)

SpatialCF is licensed under [Apache License 2.0](LICENSE).
