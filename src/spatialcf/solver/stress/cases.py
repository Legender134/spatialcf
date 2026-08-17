"""Deterministic assembly and identity binding for solver stress cases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from decimal import Decimal
from functools import cache

from spatialcf.domain.models import Vec2
from spatialcf.solver.stress.families_depth import build_depth_case
from spatialcf.solver.stress.families_distance import build_distance_case
from spatialcf.solver.stress.families_horizontal import build_horizontal_case
from spatialcf.solver.stress.models import (
    StressCase,
    StressCaseDraft,
    StressDirection,
    StressProfileName,
    StressSlot,
    StressTransform,
)
from spatialcf.solver.stress.profiles import (
    DEEP_SEEDS,
    QUICK_SEEDS,
    STRESS_DIRECTIONS,
    stress_slots,
)
from spatialcf.solver.stress.sampling import HashSampler
from spatialcf.solver.stress.scene_factory import (
    StressCaseError,
    validate_before_case,
    validate_before_draft,
)
from spatialcf.solver.stress.transforms import apply_stress_transform

_Builder = Callable[[StressSlot, int], StressCaseDraft]
_BUILDERS: dict[StressDirection, _Builder] = {
    "lr": build_horizontal_case,
    "fb": build_depth_case,
    "nf": build_distance_case,
}


class StressGenerationError(ValueError):
    """The frozen schedule could not be generated without weakening a gate."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _placeholder_digest(draft: StressCaseDraft) -> str:
    payload = draft.model_dump(mode="json")
    scene = payload["scene"]
    if not isinstance(scene, dict):
        raise StressGenerationError("stress draft scene payload is not an object")
    scene["scene_id"] = "__CASE_ID__"
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sample_transform(
    slot: StressSlot,
    attempt: int,
    base_case_digest: str,
) -> StressTransform:
    sampler = HashSampler(f"{slot.key}/{attempt}/transform")
    translation = Vec2(
        x=sampler.grid(
            "translation_x",
            Decimal("-2.00"),
            Decimal("2.00"),
            Decimal("0.25"),
        ),
        y=sampler.grid(
            "translation_y",
            Decimal("-2.00"),
            Decimal("2.00"),
            Decimal("0.25"),
        ),
    )
    if slot.expected_outcome == "UNSAT" and slot.direction in {"lr", "fb"}:
        translation = Vec2(x=0.0, y=0.0)
    mirror = sampler.choice(
        "mirror",
        ("none", "camera_horizontal", "camera_depth"),
    )
    rotation = sampler.choice("rotation", (0, 90, 180, 270))
    transform = StressTransform(
        translation_xy=translation,
        mirror=mirror,
        rotation_degrees=rotation,
        base_case_digest=base_case_digest,
    )
    if not transform.transformed:
        transform = transform.model_copy(
            update={
                "mirror": (
                    "camera_horizontal"
                    if slot.expected_outcome == "UNSAT"
                    and slot.direction in {"lr", "fb"}
                    else transform.mirror
                ),
                "translation_xy": (
                    transform.translation_xy
                    if slot.expected_outcome == "UNSAT"
                    and slot.direction in {"lr", "fb"}
                    else Vec2(x=0.25, y=0.0)
                ),
            }
        )
    return transform


def _require_base_matches_slot(base: StressCaseDraft, slot: StressSlot) -> None:
    if (
        base.seed,
        base.direction,
        base.raw_slot,
        base.family,
        base.expected_outcome,
    ) != (
        slot.seed,
        slot.direction,
        slot.raw_slot,
        slot.family,
        slot.expected_outcome,
    ):
        raise StressGenerationError(
            f"builder result does not match requested stress slot: {slot.key}"
        )
    if base.transform != StressTransform():
        raise StressGenerationError(
            f"builder result must use the identity base transform: {slot.key}"
        )


def _first_valid_draft(slot: StressSlot) -> StressCaseDraft:
    builder = _BUILDERS[slot.direction]
    for attempt in range(64):
        try:
            base = builder(slot, attempt)
            _require_base_matches_slot(base, slot)
            validate_before_draft(base)
            if not slot.transformed:
                return base
            base_digest = _placeholder_digest(base)
            transformed = apply_stress_transform(
                base,
                _sample_transform(slot, attempt, base_digest),
            )
            validate_before_draft(transformed)
        except StressCaseError:
            continue
        return transformed
    raise StressGenerationError(f"no valid stress draft after 64 attempts: {slot.key}")


def _bind_case_id(draft: StressCaseDraft, case_id: str) -> StressCase:
    scene = draft.scene.model_copy(update={"scene_id": case_id})
    case = StressCase(
        case_id=case_id,
        scene=scene,
        **draft.model_dump(mode="python", exclude={"scene"}),
    )
    validate_before_case(case)
    return case


def _generate_direction_group_cases(
    seed: int,
    direction: StressDirection,
) -> tuple[StressCase, ...]:
    if isinstance(seed, bool) or seed not in DEEP_SEEDS:
        raise StressGenerationError(f"unknown stress seed: {seed!r}")
    if direction not in STRESS_DIRECTIONS:
        raise StressGenerationError(f"unknown stress direction: {direction!r}")
    slots = tuple(
        slot
        for slot in stress_slots("deep")
        if slot.seed == seed and slot.direction == direction
    )
    drafts = tuple(_first_valid_draft(slot) for slot in slots)
    digests = tuple(_placeholder_digest(draft) for draft in drafts)
    if len(digests) != 100 or len(set(digests)) != 100:
        raise StressGenerationError("stress direction group has non-unique digests")
    ordered = sorted(zip(digests, drafts, strict=True), key=lambda item: item[0])
    return tuple(
        _bind_case_id(
            draft,
            f"stress-{seed}-{direction}-{index:03d}",
        )
        for index, (_, draft) in enumerate(ordered)
    )


@cache
def _cached_direction_group_payloads(
    seed: int,
    direction: StressDirection,
) -> tuple[bytes, ...]:
    return tuple(
        _canonical_json_bytes(case.model_dump(mode="json"))
        for case in _generate_direction_group_cases(seed, direction)
    )


def generate_direction_group(
    seed: int,
    direction: StressDirection,
) -> tuple[StressCase, ...]:
    """Generate a fresh model view of one cached, immutable 100-case group."""
    if isinstance(seed, bool) or seed not in DEEP_SEEDS:
        raise StressGenerationError(f"unknown stress seed: {seed!r}")
    if direction not in STRESS_DIRECTIONS:
        raise StressGenerationError(f"unknown stress direction: {direction!r}")
    return tuple(
        StressCase.model_validate_json(payload)
        for payload in _cached_direction_group_payloads(seed, direction)
    )


def generate_stress_cases(profile: StressProfileName) -> tuple[StressCase, ...]:
    """Generate one exact frozen quick or deep profile in schedule order."""
    if profile == "quick":
        seeds = QUICK_SEEDS
    elif profile == "deep":
        seeds = DEEP_SEEDS
    else:
        raise StressGenerationError(f"unknown stress profile: {profile!r}")
    return tuple(
        case
        for seed in seeds
        for direction in STRESS_DIRECTIONS
        for case in generate_direction_group(seed, direction)
    )


def replay_stress_case(case_id: str) -> StressCase:
    """Regenerate and return exactly one frozen stress case by public ID."""
    if not isinstance(case_id, str):
        raise StressGenerationError("stress case ID must be a string")
    match = re.fullmatch(r"stress-(\d{10})-(lr|fb|nf)-(\d{3})", case_id)
    if match is None:
        raise StressGenerationError(f"invalid stress case ID: {case_id!r}")
    seed = int(match.group(1))
    direction = match.group(2)
    index = int(match.group(3))
    if seed not in DEEP_SEEDS or index >= 100:
        raise StressGenerationError(f"unknown stress case ID: {case_id!r}")
    payloads = _cached_direction_group_payloads(seed, direction)  # type: ignore[arg-type]
    case = StressCase.model_validate_json(payloads[index])
    if case.case_id != case_id:
        raise StressGenerationError(f"stress replay identity mismatch: {case_id!r}")
    return case
