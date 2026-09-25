# Concepts

SpatialCF separates portable spatial reasoning from environment-specific scene
execution.

## Platform-neutral Schema

The Schema represents scenes, object geometry, relations, interventions,
evidence, and results in canonical coordinates. Core relations are `left`,
`right`, `front`, `behind`, `near`, and `far`. A counterfactual moves exactly one
movable object in world X/Y. The published v2/M2 generation behavior preserves
its height, orientation, scale, category, camera, lighting, and material.

The separate CPU-only `spatialcf/upright_se2@1` General-IR profile extends only
direct General-IR work with upright yaw around the subject's own pivot or a
named reference pivot; it does not change generation behavior. Exact cardinal
yaw closes before continuous yaw. Continuous requests use canonical
`ARC`/`FULL_CIRCLE` intervals lifted with exact dyadic endpoints and checked
directed bounds. They can remain LIMITED with an uncertified witness or
`UNKNOWN`; finite misses, numeric gaps, unsupported capabilities, and resource
exhaustion are never UNSAT.

## Semantic placement

The direct-GeneralIR `spatialcf/semantic_place@1` profile adds `PLACE_ON` and
`PLACE_IN` for exact identity-oriented boxes. Changing support derives the
subject's world Z from bottom contact; it is not free vertical motion. XY is a
continuous finite closed domain. Exact arrangement cells include boundary
lines and points, so a thin feasible component cannot be discarded.

Containment means the entire collision box lies inside a declared closed
cavity interior. Cavity facts bind an owner, bottom support and complete shell
body roster; solid-object bounds do not establish an interior. The checker
reconstructs every declared before/after containment truth, even for an ON
placement and for unselected cavities. State changes are limited to the
subject's existing XYZ and support assignment, with a complete unchanged-leaf
digest for the rest of the scene.

Only complete exact coverage can certify a global squared-displacement minimum
or prove the whole authorized domain empty. A finite search miss, resource
prefix, unsupported source or unrepresentable endpoint remains UNKNOWN or an
explicit rejection. The evidence describes endpoint geometry, not a collision-free
transport path, physical stability or a native rollout. Existing generation
behavior and M4 semantic dataset formats are retained.

## Finite multi-object rigid SE(3)

The direct-GeneralIR `spatialcf/rigid_se3_multi@1` profile represents multiple exact rigid bodies with full three-dimensional root poses. A source-bound forest of fixed, prismatic and revolute joints determines link poses; explicit contact assignments are checked against geometry. An ordered program contains atomic edit sets, so a coordinated multi-body change is evaluated as one step and every subsequent prefix has its own complete state and obligations. Commutativity claims require checked evidence about both adjacent orders.

Its registered geometry is a finite union of rationally anchored rectangular boxes per body. Proper rational SO(3) matrices allow noncardinal pitch and roll. The source and the authorized finite program universe are explicit; no hidden discretization of a continuous domain or implicit expansion to arbitrary meshes is used. Complete enumeration plus fresh checker replay proves a minimum only over that authorized finite universe, or proves its complete infeasibility. A continuous or otherwise unfinished search yields typed UNKNOWN, or a feasible but noncertified witness if every state of that proposed program checks.

The proof covers each discrete program endpoint and prefix. It does not establish a collision-free swept trajectory, dynamics, physical stability, native execution or a rendered observation. Existing single-object generation, M3 upright yaw, M4 semantic contrast and M5 placement remain separate routes.

## Minimum-cost solver

The solver constructs a feasible domain from geometric, collision, support,
visibility, and target-relation constraints. It chooses a deterministic
minimum-cost edit from that domain. Labels are derived from geometry and
independent verification, not supplied by the Adapter.

## Advanced planar compatibility boundary

`spatialcf/planar_translate@2` is an additive compatibility embedding for
sealed current Canonical v2 inputs. It maps those inputs into general
counterfactual contracts, then its thin backend delegates to the one existing v2
solver owner. Existing v2 result and certificate verification remains at the
existing verifier boundary. This is not a second solver or verifier, and it does
not add a generation path or change the supported generation API.

## Adapter

An Adapter loads and normalizes source scenes, captures observations, reports
native placement facts, applies the selected edit, and extracts assets. Unity/
AI2-THOR is the first Adapter. The Schema, solver, and verifier do not depend on
that runtime.

## Fresh verification and publication

Generation freezes request membership before execution. Each planned request has
one accepted or rejected terminal outcome, and runtime results do not reorder or
backfill the roster. Publication creates a stable index only after referenced
evidence passes verification.

The verifier reopens the dataset, recomputes identities and checksums, and rejects
missing, extra, modified, or mismatched records. It is read-only and does not
repair failed output.
