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
