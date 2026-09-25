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

## Semantic placement

`spatialcf/semantic_place@1` is a direct-GeneralIR CPU profile. Use a complete
CanonicalScene with exact identity-oriented boxes and closed horizontal
rectangular supports. The existing editable object's real ID must start with
`entity:`; the builder does not rename source objects. Supply finite closed
world X/Y/Z bounds and sorted, unique target IDs. The subject must initially
have valid support and must not already satisfy any requested target goal.

```python
from spatialcf.core.semantic_place_compiler import build_semantic_place_request
from spatialcf.core.semantic_place_backend import SemanticPlaceBackend
from spatialcf.core.outcome_assembler import (
    assemble_counterfactual_outcome, assemble_no_selection_unknown,
)
from spatialcf.domain.semantic_place import SemanticPlaceInterval

# scene is the caller's complete CanonicalScene, with these existing IDs.
request = build_semantic_place_request(
    scene=scene, subject_id="entity:subject", operation="PLACE_ON",
    target_ids=("surface:table",),
    x=SemanticPlaceInterval(lower=-10.0, upper=10.0),
    y=SemanticPlaceInterval(lower=-10.0, upper=10.0),
    z=SemanticPlaceInterval(lower=0.0, upper=10.0),
    cavities=(),
)
backend = SemanticPlaceBackend()
selection = backend.select(request)
if selection.selection_disposition == "NO_SELECTION":
    outcome = assemble_no_selection_unknown(
        solve_request=request, selection=selection,
    )
else:
    compiled = backend.compile(request)
    submission = backend.solve_submission(compiled, request.solver_config)
    outcome = assemble_counterfactual_outcome(
        solve_request=request, selection=selection,
        compilation=compiled, submission=submission,
    )
```

For `PLACE_IN`, pass sorted cavity IDs as targets and explicit
`SemanticPlaceCavityFact.seal(...)` records through `cavities`. Each record
names an existing owner, an identity-oriented owner-local cavity frame, positive
centered interior dimensions, its bottom support surface, and the sorted complete
roster of that owner's collision bodies. The declared open interior must not
intersect those solids, and its bottom must coincide with and fit on the owner's
support. The builder freezes every cavity's full fact address into one complete
inventory; `cavities=()` explicitly declares an empty inventory. A bounding box
of a solid object is not a cavity.

`support_margin_m`, `lateral_margin_m`, and `top_margin_m` are explicit
nonnegative clearances. `limits=SemanticPlaceLimits(...)` freezes numeric,
target, stratum and separate compilation/solve/check budgets. Geometry outside
this exact subset produces typed unsupported evidence after selection. Numeric
gaps and incomplete budget prefixes produce UNKNOWN. Only complete continuous
coverage proves UNSAT. Malformed inputs or tampered proofs raise validation
errors. If the checker cannot finish within its own budget, the assembler
rejects the unverified submission; no certificate or relabelled result is issued.

The objective is squared world-XYZ displacement with a deterministic target-ID,
then exact-XYZ tie break. A winning endpoint must round-trip exactly through
binary64; otherwise it remains a numeric gap. Finite proposal cost bounds may
be unequal outward bounds while the proof retains the exact cost. Only changed
XYZ/support-assignment leaves enter the complete delta; all other source facts
remain unchanged. The checker freshly evaluates containment for every declared
cavity, including cavities that are not targets. Trusted terminal resource usage
adds the independently checked replay cost to the producer's counters.

The standalone `tests/public_smoke/test_semantic_place.py` supplies synthetic
ON, IN and NO_SELECTION examples using only shipped modules. The existing 96
generation exports, M4 format and CLI remain unchanged. This profile performs
no rendering, native execution or motion-path validation.

## Finite multi-object rigid SE(3) General IR

`spatialcf/rigid_se3_multi@1` is an advanced CPU profile for direct General IR callers. Its source must be an explicit, exact extension inventory over an empty `CanonicalScene` carrier. The profile does not convert existing upright v2 objects into tilted rigid bodies. The source names every body, local rectangular collision box, root pose, joint, contact definition and current joint/contact state. The request binds a finite roster of ordered program skeletons with finite or continuous choice domains.

Each atomic step can set several root poses, joint values and contact modes together. The evaluator reconstructs every link's world pose and checks collision, contact and all registered invariants at every program prefix; it checks the after-goal at the final endpoint. For a finite domain, complete checked enumeration certifies the least exact rational objective among *that request's authorized programs*, or certifies UNSAT if every member is refuted. This is not a global optimum over unlisted actions or all physical motions.

The self-contained `tests/public_smoke/test_rigid_se3.py` is a runnable request-building and fresh-assembly example using only shipped runtime modules. After installing SpatialCF in a Python 3.11 environment from a public source checkout or sdist, run `python -m pytest -q tests/public_smoke/test_rigid_se3.py` there. To inspect the selected certified case in an interactive Python session from that checkout:

```python
from pathlib import Path
from runpy import run_path

from spatialcf.core.outcome_assembler import assemble_counterfactual_outcome
from spatialcf.core.rigid_se3_backend import RigidSE3Backend

example = run_path(str(Path("tests/public_smoke/test_rigid_se3.py")))
built = example["build_public_case"]("certified")
backend = RigidSE3Backend()
selection = backend.select(built.request)
compiled = backend.compile(built.request)
submission = backend.solve_submission(compiled, built.request.solver_config)
outcome = assemble_counterfactual_outcome(
    solve_request=built.request,
    selection=selection,
    compilation=compiled,
    submission=submission,
)
assert outcome.result.structural_outcome_class == "CERTIFIED_SOLUTION"
assert outcome.certificate is not None
```

The example's `build_public_case` calls `build_rigid_se3_request` with a complete two-body prismatic/contact source, an atomic edit set, exact finite choices, objective, precondition and goal. The same file exercises complete finite UNSAT, continuous UNKNOWN, a noncertified exact witness and NO_SELECTION. A disabled backend follows `assemble_no_selection_unknown`, with no selected checker or certificate.

Finite root translations and exact rational SO(3) matrices, fixed/prismatic/revolute forest joints, local rectangular boxes, endpoint face contact and registered grounded predicates are supported. Continuous translation boxes, ALL_SO3 rotation, prismatic intervals and full-circle revolute domains are represented but remain unclosed: they yield a typed UNKNOWN or a freshly checked feasible witness without an optimality certificate. Missing/inexact source facts, unsupported registered capability, numeric gaps and resource exhaustion never become UNSAT. Invalid or changed proof material is rejected rather than relabelled.

The result proves complete endpoint states and every discrete edit prefix. It does not certify swept paths, dynamics, stability, friction, rendering, native execution or arbitrary meshes. The version-free `spatialcf.generation` API, its single-object planar route and M4 data format remain unchanged; native execution is `NOT_REQUESTED`.

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
