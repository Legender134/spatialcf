# Python API

The supported version-free API is exported from `spatialcf.generation`.

## Advanced planar-translate compatibility

Public `main` also carries the advanced
`spatialcf/planar_translate@2` compatibility embedding. It maps sealed v2
planar-translate inputs into the general contract. Its compatibility backend
`spatialcf.core.planar_backend.PlanarTranslateBackend` delegates exactly once to
existing `spatialcf.core.solver.solve_minimum_cost` and does not call the
verifier. Existing v2 result/certificate verification remains entirely owned by
the existing verifier. This is not a second solver or verifier, and it does not
add a generation entry point or change the supported `spatialcf.generation` API
above.

## Advanced upright SE(2) General-IR implementation

This source tree contains the CPU-only `spatialcf/upright_se2@1` implementation
for direct General-IR integration. It is an advanced source-level contract, not
a package facade, a new CLI, or an addition to the supported version-free
`spatialcf.generation` API. It does not change the existing generation route or
the `v0.1.1` tag.

The domain contract is in `spatialcf.domain.upright_se2`; compilation, proposal
submission, fresh checking, and terminal assembly respectively live in
`spatialcf.core.upright_se2_compiler`, `spatialcf.core.upright_se2_backend`,
`spatialcf.core.upright_se2_verification`, and
`spatialcf.core.outcome_assembler`. The profile supports world XY translation
and upright yaw around an own or named reference pivot. Exact cardinal yaw
closes before continuous yaw. Continuous domains use canonical `ARC` or
`FULL_CIRCLE` with exact-dyadic lifted intervals and checked directed bounds.

`solve_submission` is deliberately disjoint from retained v1 `solve`: it emits
untrusted `BackendSubmission` proposal, complete-domain UNSAT, or UNKNOWN
evidence. The checker produces only `CheckedProofOutcome` and never assembles a
terminal result. `core.outcome_assembler` is the sole general-IR checker
dispatcher, certificate owner, and terminal-result assembler. A finite miss,
numeric gap, unsupported capability, resource exhaustion, or feasible-incomplete
frontier is never fabricated as UNSAT; feasible-incomplete work remains LIMITED
with an uncertified witness.

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
