"""The closed deterministic quick/deep stress schedule."""

from __future__ import annotations

import hashlib
from typing import Final

from spatialcf.solver.stress.models import (
    StressDirection,
    StressFamily,
    StressProfileName,
    StressSlot,
)

DEEP_SEEDS: Final = tuple(range(2026080200, 2026080210))
QUICK_SEEDS: Final = DEEP_SEEDS[:1]
STRESS_DIRECTIONS: Final[tuple[StressDirection, ...]] = ("lr", "fb", "nf")
SAT_FAMILIES: Final[tuple[StressFamily, ...]] = (
    "target_boundary",
    "obstacle_corner",
    "support_boundary",
    "preservation_intersection",
    "tied_optimum",
)
UNSAT_FAMILIES: Final[tuple[StressFamily, ...]] = (
    "room_bounds",
    "support_locus",
    "obstacle_coverage",
    "relation_upper_bound",
)

_SAT_SLOTS_PER_FAMILY: Final = 12
_UNSAT_SLOTS_PER_FAMILY: Final = 10
_SAT_TRANSFORMED_PER_GROUP: Final = 18
_UNSAT_TRANSFORMED_PER_GROUP: Final = 12


def _ranked_transformed_slots(
    seed: int,
    direction: StressDirection,
    raw_slots: range,
    count: int,
) -> frozenset[int]:
    ranked = sorted(
        raw_slots,
        key=lambda raw_slot: hashlib.sha256(
            f"stress-v1/{seed}/{direction}/{raw_slot:03d}/transform".encode("utf-8")
        ).digest(),
    )
    return frozenset(ranked[:count])


def _direction_slots(seed: int, direction: StressDirection) -> tuple[StressSlot, ...]:
    transformed_sat = _ranked_transformed_slots(
        seed,
        direction,
        range(0, 60),
        _SAT_TRANSFORMED_PER_GROUP,
    )
    transformed_unsat = _ranked_transformed_slots(
        seed,
        direction,
        range(60, 100),
        _UNSAT_TRANSFORMED_PER_GROUP,
    )
    slots: list[StressSlot] = []
    for family_index, family in enumerate(SAT_FAMILIES):
        for offset in range(_SAT_SLOTS_PER_FAMILY):
            raw_slot = family_index * _SAT_SLOTS_PER_FAMILY + offset
            slots.append(
                StressSlot(
                    seed=seed,
                    direction=direction,
                    raw_slot=raw_slot,
                    family=family,
                    expected_outcome="SAT",
                    transformed=raw_slot in transformed_sat,
                )
            )
    for family_index, family in enumerate(UNSAT_FAMILIES):
        for offset in range(_UNSAT_SLOTS_PER_FAMILY):
            raw_slot = 60 + family_index * _UNSAT_SLOTS_PER_FAMILY + offset
            slots.append(
                StressSlot(
                    seed=seed,
                    direction=direction,
                    raw_slot=raw_slot,
                    family=family,
                    expected_outcome="UNSAT",
                    transformed=raw_slot in transformed_unsat,
                )
            )
    return tuple(slots)


def stress_slots(profile: StressProfileName) -> tuple[StressSlot, ...]:
    """Return the only supported schedule, in frozen seed/direction/raw order."""
    if profile == "quick":
        seeds = QUICK_SEEDS
    elif profile == "deep":
        seeds = DEEP_SEEDS
    else:
        raise ValueError(f"unknown stress profile: {profile!r}")
    return tuple(
        slot
        for seed in seeds
        for direction in STRESS_DIRECTIONS
        for slot in _direction_slots(seed, direction)
    )
