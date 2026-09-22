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

## Semantic contrast datasets

The advanced CPU-only API is available from `spatialcf.generation.contrast`:

```python
from pathlib import Path
from spatialcf.generation.contrast import (
    generate_semantic_contrast_dataset,
    verify_semantic_contrast_dataset,
)

report = generate_semantic_contrast_dataset(
    catalog=Path("inputs/catalog.json"),
    output=Path("semantic-dataset"),
)
assert verify_semantic_contrast_dataset(Path("semantic-dataset")) == report
```

Both entry points take `Path` arguments and return the strict
`spatialcf.domain.contrast.SemanticContrastReport`. Build a
`SemanticContrastCatalogInput` with its `.seal()` constructor and serialize it
with `spatialcf.domain.serialization.canonical_json_bytes`. The catalog contains
`SourceRecordInput` values, explicit `CandidateRecordInput` requests, and a
`PublicationPolicy`. Source byte references resolve relative to the catalog
file's directory; their declared byte lengths and SHA-256 hashes must match.
A source binds its complete `SceneStateEnvelope`, source identity and file
manifest. Every request, backend profile and budget is fixed before solving.
The standalone public smoke in `tests/public_smoke/test_semantic_contrast.py`
shows a complete synthetic input without private fixtures.

Cardinal and continuous upright-SE(2) requests use the existing compiler,
backend, checker and outcome assembler. The supplied catalog defines the
candidate universe. Source-group splits and opposite-relation eligibility are
deterministic; there is no success-based replacement or retry. The terminal
ledger distinguishes published certified pairs, certified complete-domain
UNSAT, UNKNOWN, noncertified witnesses and policy rejection. Empty and
zero-pair datasets retain truthful counts. A pair records the exact accepted
claim, including whether it is exact-global or finite-gap.

The output contains `manifest.json`, `catalog.json`, `terminals.json`,
`report.json`, `records/`, `objects/` and `sources/`. Typed object identities
are distinct from file-byte hashes. The lineage binds the complete retained
source/request/proof/result/program chain; the outer record and manifest bind
its acyclic file closure. Generation requires an absent output path and uses
atomic no-replace publication after strict rereading and fresh proof replay.
If a filesystem error occurs after a rename, the raised publication error
reports whether the owned output is published, not published or indeterminate,
and carries the recovery binding when known. It does not silently retry.

Verification makes no writes and performs no backend search. It recompiles,
reselects and freshly checks every retained submitted outcome, reconstructs
all expected canonical bytes, and re-evaluates policy rejection and
NO_SELECTION outcomes. It also requires an exact supported runtime provenance
match: CPython 3.11 patch version, the declared portable module-byte closure,
and installed dependency metadata. Provenance has `LOCAL_SOURCE_SNAPSHOT`
scope. Unavailable lock metadata is explicit with
`UNAVAILABLE / INSTALLED_PACKAGE_METADATA`; this is not a hermetic rebuild claim.

These datasets contain semantic commanded before/after states and evidence.
They do not render images or execute an adapter, and native execution remains
`NOT_REQUESTED`. The existing 96 root generation exports and CLI remain unchanged.

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
