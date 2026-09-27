# Quick start

## CPU general counterfactual datasets

Start with current `main` and Python 3.11:

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

The input is a synthetic scene and task catalog. The placement case moves a unit
box from the floor onto a raised table: the certified endpoint is
`(3.5, 0, 1.5)` on `surface:table`. Each command uses the same retained input and
proofs. Verification is fresh; it can take minutes for multi-body cases.
`inspect` also verifies before printing, so it is not a cheap unverified peek.
Use new paths: input creation and dataset publication refuse to overwrite them.

## Choose a case

Change `--case placement` and use a different input/output path for each run.

| Case | Task | Expected terminal / pairs |
| --- | --- | --- |
| `placement` | Put a box onto a table | PUBLISHED_PAIR / 1 |
| `multibody` | Close a prismatic joint and engage face contact | PUBLISHED_PAIR / 1 |
| `mixed` (default) | Both preceding tasks | PUBLISHED_PAIR / 2 |
| `unsat` | Require contact while the finite domain only permits release | PROVEN_UNSAT / 0 |
| `unknown` | Continuous root domain without complete proof or a hint | UNKNOWN / 0 |
| `witness` | Same continuous domain with an exact feasible program | NONCERTIFIED_WITNESS / 0 |

For example:

```bash
python examples/general_dataset.py unknown-input.json --case unknown
spatialcf general generate --input unknown-input.json --output unknown-dataset
spatialcf general verify unknown-dataset
```

A separate low-budget planar solver example uses two boxes, a floor and a camera:

```bash
python examples/planar.py
```

It prints the actual solver outcome and fresh v2 replay result. Its bounded
budget may leave the task uncertified; it does not promise a certified pair or
write the general dataset format. The v2 verifier replays its solver; the
M5/M6 general checker independently replays retained proof material without
running backend search.

## Read the result

| File | Meaning |
| --- | --- |
| `input.json` | Explicit source facts, tasks and publication policy |
| `catalog.json` | Frozen candidate requests and budgets |
| `terminals.json` | One outcome per candidate, including all non-published outcomes |
| `records.json` | Certified false-before / true-after pairs and lineage |
| `report.json` | Counts by outcome and published pairs |
| `provenance.json` | Runtime module identities |
| `manifest.json` | Sealed file inventory |

`PUBLISHED_PAIR` means the applicable checker accepted the solution and required
optimality evidence. `PROVEN_UNSAT` means the complete authorized domain was
proved infeasible. `UNKNOWN` means no such conclusion was justified.
`NONCERTIFIED_WITNESS` retains a feasible program without certified optimality;
it is not a published pair. Do not collapse these outcomes into a binary label.

These are semantic endpoints and discrete prefixes, not rendered observations,
physical stability guarantees or collision-free swept paths. M5 uses fixed
axis-aligned boxes; M6 uses finite exact box primitives and acyclic joints.
See [concepts](concepts.md) and [API](api.md#general-counterfactual-datasets).

## Native Adapter route

The separate `spatialcf generate/verify/inspect` route uses an Adapter-specific
configuration and a different dataset format. See [Adapters](adapters.md) for
installation and native prerequisites. The historical `v0.1.1` annotated tag
retains that older workflow; it does not provide `spatialcf general`.
