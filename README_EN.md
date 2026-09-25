[简体中文](README.md) | [English](README_EN.md)

# SpatialCF

SpatialCF generates verified spatial counterfactual datasets. Current chain: domain/core → adapter protocol → generation → fresh verification. It freezes requests from scene observations, plans single-object planar moves with a minimum-cost solver, connects platform facts and Canonical Edits through the Adapter protocol, and freshly verifies the results and dataset files.

The Schema, solver, and verification logic are platform-neutral. Unity/AI2-THOR
is the first Adapter and connects platform facts and native operations to the
public generation chain.

## Current `main` status

The stable generation chain is unchanged. Current public `main` includes the
advanced `spatialcf/planar_translate@2` compatibility embedding: it maps sealed
v2 inputs into general contracts. Its compatibility backend
`spatialcf.core.planar_backend.PlanarTranslateBackend` delegates exactly once to
the existing `spatialcf.core.solver.solve_minimum_cost` and does not call the
verifier. Existing v2 result/certificate verification remains entirely owned by
the existing verifier owner. It is not a second solver or verifier, and it does
not create a new generation route.

This source tree also contains the individually verified CPU-only
`spatialcf/upright_se2@1` General-IR implementation. It gives direct
General-IR callers world XY translation and upright yaw about own or reference
pivots. Exact cardinal closure precedes continuous canonical `ARC`/`FULL_CIRCLE`
directed interval checking; a continuous request can yield a limited
uncertified witness or `UNKNOWN` and is not promised universal certification.
Its backend submits untrusted `BackendSubmission` evidence through the disjoint
`solve_submission`; the checker produces only `CheckedProofOutcome`; and
`core.outcome_assembler` solely dispatches general checking and assembles
certificates and terminal results. This does not change the version-free
generation API, the existing generation route, or the `v0.1.1` tag.

This source tree also includes the direct-GeneralIR CPU profile `spatialcf/rigid_se3_multi@1`. Explicit exact multi-body box, joint and contact facts define ordered atomic edit sets with complete endpoint and prefix checks. A fresh independent checker certifies a minimum over the entire authorized *finite* program universe or complete UNSAT only after exhaustive coverage. Unclosed continuous domains yield a typed UNKNOWN or a feasible witness without an optimality certificate. The profile handles full 3D root poses, noncardinal rational rotations and fixed/prismatic/revolute forest joints; it does not certify swept paths, dynamics or native execution. See the [Python API](docs/api.md#finite-multi-object-rigid-se3-general-ir) for a runnable request and five terminal examples. The existing generation route and M4 data format are unchanged; native execution is `NOT_REQUESTED`.

Current public `main` is distinct from the latest annotated release tag:
`v0.1.1` remains the latest annotated release tag and `v0.2.0` has not been
released. The quick start below therefore continues to use the exact `v0.1.1`
tag.

## Repository and release

The authoritative user-facing repository is
[`Legender134/spatialcf`](https://github.com/Legender134/spatialcf). `v0.1.1`
is the latest annotated release tag. It is not a PyPI publication and is not a
GitHub Release. Clone that tag and install from the local checkout.

Public releases come from a verified deterministic snapshot. Complete
development history, private release manifests, and recovery evidence stay in
separate private development and archive boundaries and are never copied into
the user repository.

## Quick start

Python 3.11 is required. These commands create an environment, install the
AI2-THOR Adapter, generate a dataset, and reopen it for verification and
inspection:

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
