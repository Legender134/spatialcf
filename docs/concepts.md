# Concepts

SpatialCF separates portable spatial reasoning from environment-specific scene
execution.

## Platform-neutral Schema

The Schema represents scenes, object geometry, relations, interventions,
evidence, and results in canonical coordinates. Core relations are `left`,
`right`, `front`, `behind`, `near`, and `far`. A counterfactual moves exactly one
movable object in world X/Y while preserving its height, orientation, scale,
category, camera, lighting, and material.

## Minimum-cost solver

The solver constructs a feasible domain from geometric, collision, support,
visibility, and target-relation constraints. It chooses a deterministic
minimum-cost edit from that domain. Labels are derived from geometry and
independent verification, not supplied by the Adapter.

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
