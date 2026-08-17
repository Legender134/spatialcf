"""Independent analytic constructors for distance stress families."""

from __future__ import annotations

import math

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import InterventionSpec, SceneObject, Vec2, Vec3
from spatialcf.geometry.obb import ground_gap
from spatialcf.solver.stress.models import (
    SatStressOracle,
    StressCaseDraft,
    StressSlot,
    UnsatStressOracle,
)
from spatialcf.solver.stress.sampling import HashSampler
from spatialcf.solver.stress.scene_factory import (
    StressCaseError,
    make_camera,
    make_object,
    make_scene,
)


def _camera():
    camera = make_camera(focal_px=40.0, depth_offset=4.0)
    matrix = list(camera.world_to_camera)
    matrix[9] = 0.0
    return camera.model_copy(update={"world_to_camera": tuple(matrix)})


def _pair(
    *,
    subject_x: float = -0.7,
    reference_x: float = 0.0,
    subject_extent_xy: float = 0.4,
    reference_extent_xy: float = 0.4,
    yaw_degrees: int = 0,
) -> tuple[SceneObject, SceneObject]:
    camera = _camera()
    subject = make_object(
        "subject",
        position=Vec3(x=subject_x, y=0.0, z=0.7),
        extent=Vec3(x=subject_extent_xy, y=subject_extent_xy, z=1.0),
        yaw_degrees=yaw_degrees,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=reference_x, y=0.0, z=0.7),
        extent=Vec3(x=reference_extent_xy, y=reference_extent_xy, z=1.0),
        yaw_degrees=yaw_degrees,
        movable=False,
        camera=camera,
    )
    return subject, reference


def _draft(
    slot: StressSlot,
    attempt: int,
    *,
    room_bounds: tuple[float, float, float, float],
    objects: tuple[SceneObject, ...],
    oracle: SatStressOracle | UnsatStressOracle,
) -> StressCaseDraft:
    camera = _camera()
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-nf-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=room_bounds,
            camera=camera,
            objects=objects,
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.NEAR,
            relation_after=Relation.FAR,
            camera_id=camera.camera_id,
        ),
        oracle=oracle,
    )


def _target_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    sampler = HashSampler(f"{slot.key}/{attempt}")
    yaw_degrees = sampler.choice("pair_yaw_degrees", (0, 45))
    if yaw_degrees == 45:
        extent_xy = 0.4
        reference_x = 0.0
        subject_x = -0.9
        witness_x = -(0.4 * math.sqrt(2.0) + 1.5)
    else:
        extent_xy = sampler.choice("pair_extent_xy", (0.2, 0.4, 0.6))
        reference_x = sampler.choice("reference_x", (-0.25, 0.0, 0.25))
        subject_x = reference_x - extent_xy - 0.3
        witness_x = reference_x - extent_xy - 1.5
    subject, reference = _pair(
        subject_x=subject_x,
        reference_x=reference_x,
        subject_extent_xy=extent_xy,
        reference_extent_xy=extent_xy,
        yaw_degrees=yaw_degrees,
    )
    for _ in range(8):
        moved = subject.obb.model_copy(
            update={"center": subject.obb.center.model_copy(update={"x": witness_x})}
        )
        if ground_gap(moved, reference.obb) >= 1.5:
            break
        witness_x = math.nextafter(witness_x, -math.inf)
    else:
        raise StressCaseError("target_boundary_not_enclosed")
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -2.0, 2.0, 2.0),
        objects=(subject, reference),
        oracle=SatStressOracle(
            proof_kind="target_boundary",
            exact_infimum_m=round(abs(witness_x - subject.position.x), 12),
            exact_infimum_points=(Vec2(x=witness_x, y=0.0),),
            derivation="the nearest FAR point extends the initial left radial ray",
        ),
    )


def _obstacle_case(
    slot: StressSlot,
    attempt: int,
    *,
    tied: bool,
) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _pair()
    sampler = HashSampler(f"{slot.key}/{attempt}")
    blocker_y = 0.0 if tied else 0.2
    blocker_x_extent = sampler.choice("blocker_extent_x", (0.4, 0.6, 0.8))
    blocker_y_extent = 0.2
    configuration_half_height = blocker_y_extent / 2.0 + 0.4 / 2.0
    blocker = make_object(
        "blocker",
        position=Vec3(x=-1.9, y=blocker_y, z=0.7),
        extent=Vec3(x=blocker_x_extent, y=blocker_y_extent, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    witness_ys = (
        (-configuration_half_height, configuration_half_height) if tied else (-0.1,)
    )
    witnesses = tuple(Vec2(x=-1.9, y=y) for y in witness_ys)
    exact = round(math.hypot(1.2, abs(witness_ys[0])), 12)
    family = "tied_optimum" if tied else "obstacle_corner"
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -2.0, 2.0, 2.0),
        objects=(subject, reference, blocker),
        oracle=SatStressOracle(
            proof_kind=family,
            exact_infimum_m=exact,
            exact_infimum_points=witnesses,
            derivation=(
                "symmetric target/blocker configuration contacts tie"
                if tied
                else "the lower target/blocker configuration contact is uniquely nearest"
            ),
        ),
    )


def _support_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _pair()
    support_extent_x = HashSampler(f"{slot.key}/{attempt}").choice(
        "support_extent_x",
        (2.5, 3.0, 3.5),
    )
    support = make_object(
        "support",
        position=Vec3(x=-1.2 + support_extent_x / 2.0, y=0.0, z=0.1),
        extent=Vec3(x=support_extent_x, y=4.5, z=0.2),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    subject = subject.model_copy(update={"support_object_id": support.object_id})
    witness_y = 0.4 + math.sqrt(1.5**2 - 0.6**2)
    witnesses = (
        Vec2(x=-1.0, y=-witness_y),
        Vec2(x=-1.0, y=witness_y),
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -2.5, 2.5, 2.5),
        objects=(subject, reference, support),
        oracle=SatStressOracle(
            proof_kind="support_boundary",
            exact_infimum_m=round(math.hypot(0.3, witness_y), 12),
            exact_infimum_points=witnesses,
            derivation="the support left boundary intersects both exact FAR fillets",
        ),
    )


def _preservation_intersection_case(
    slot: StressSlot,
    attempt: int,
) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _pair()
    guard_extent_y = HashSampler(f"{slot.key}/{attempt}").choice(
        "guard_extent_y",
        (0.2, 0.4, 0.6),
    )
    guard = make_object(
        "guard",
        position=Vec3(x=-4.2, y=0.0, z=0.7),
        extent=Vec3(x=0.4, y=guard_extent_y, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    )
    boundary_x = -1.000000000099999
    witness_y = 1.774772708443109
    witnesses = (
        Vec2(x=boundary_x, y=-witness_y),
        Vec2(x=boundary_x, y=witness_y),
    )
    exact = round(math.hypot(boundary_x + 0.7, witness_y), 12)
    return _draft(
        slot,
        attempt,
        room_bounds=(-4.5, -2.5, 2.0, 2.5),
        objects=(subject, reference, guard),
        oracle=SatStressOracle(
            proof_kind="preservation_intersection",
            exact_infimum_m=exact,
            exact_infimum_points=witnesses,
            derivation="the preserved RIGHT boundary intersects both exact FAR fillets",
        ),
    )


def _room_bounds_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    subject_extent, reference_extent = HashSampler(f"{slot.key}/{attempt}").choice(
        "pair_extents",
        ((0.6, 1.0), (0.8, 0.8), (0.8, 1.0)),
    )
    subject, reference = _pair(
        subject_x=-1.2,
        subject_extent_xy=subject_extent,
        reference_extent_xy=reference_extent,
    )
    corner_gap = 2.0 - subject_extent - reference_extent / 2.0
    maximum = round(math.hypot(corner_gap, corner_gap), 12)
    return _draft(
        slot,
        attempt,
        room_bounds=(-2.0, -2.0, 2.0, 2.0),
        objects=(subject, reference),
        oracle=UnsatStressOracle(
            proof_kind="room_bounds",
            maximum_possible_value_m=maximum,
            required_value_m=1.5,
            expected_reason="empty_outer_region",
            derivation="every room center-locus corner is less than FAR from reference",
        ),
    )


def _support_locus_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _pair()
    support_extent_x, support_extent_y = HashSampler(f"{slot.key}/{attempt}").choice(
        "support_extents",
        ((2.0, 2.0), (2.0, 2.25), (2.25, 2.0)),
    )
    support = make_object(
        "support",
        position=Vec3(x=0.0, y=0.0, z=0.1),
        extent=Vec3(x=support_extent_x, y=support_extent_y, z=0.2),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    subject = subject.model_copy(update={"support_object_id": support.object_id})
    maximum = round(
        math.hypot(
            (support_extent_x - 1.2) / 2.0,
            (support_extent_y - 1.2) / 2.0,
        ),
        12,
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(-2.0, -2.0, 2.0, 2.0),
        objects=(subject, reference, support),
        oracle=UnsatStressOracle(
            proof_kind="support_locus",
            maximum_possible_value_m=maximum,
            required_value_m=1.5,
            expected_reason="empty_outer_region",
            derivation="the support center rectangle bounds every reference gap",
        ),
    )


def _relation_upper_bound_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera()
    guard_x = HashSampler(f"{slot.key}/{attempt}").choice(
        "guard_x",
        (0.2, 0.3),
    )
    subject, reference = _pair(subject_x=guard_x - 0.9)
    guard = make_object(
        "guard",
        position=Vec3(x=guard_x, y=0.0, z=1.45),
        extent=Vec3(x=0.4, y=0.4, z=0.4),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(-2.0, -2.0, 2.0, 2.0),
        objects=(subject, reference, guard),
        oracle=UnsatStressOracle(
            proof_kind="relation_upper_bound",
            maximum_possible_value_m=round(guard_x + 0.500000009192, 12),
            required_value_m=1.5,
            expected_reason="empty_outer_region",
            derivation="preserved NEAR to the elevated visible guard caps reference gap",
        ),
    )


def _obstacle_coverage_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _pair(
        subject_x=-0.3,
        reference_x=-1.4,
        subject_extent_xy=0.8,
        reference_extent_xy=0.8,
    )
    blocker_extent_x = HashSampler(f"{slot.key}/{attempt}").choice(
        "blocker_extent_x",
        (0.8, 1.0, 1.2),
    )
    blockers = tuple(
        make_object(
            f"blocker_{index}",
            position=Vec3(x=1.0, y=y, z=0.7),
            extent=Vec3(x=blocker_extent_x, y=0.8, z=1.0),
            yaw_degrees=0,
            movable=False,
            camera=camera,
            request_eligible=False,
        ).model_copy(update={"views": {}})
        for index, y in enumerate((-1.1, 0.0, 1.1))
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(-2.0, -2.0, 2.0, 2.0),
        objects=(subject, reference, *blockers),
        oracle=UnsatStressOracle(
            proof_kind="obstacle_coverage",
            maximum_possible_value_m=round(
                math.hypot(1.2 - blocker_extent_x / 2.0, 0.8),
                12,
            ),
            required_value_m=1.5,
            expected_reason="empty_outer_region",
            derivation="three overlapping configuration obstacles cover the FAR strip",
        ),
    )


def build_distance_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    """Build one deterministic direct NEAR-to-FAR stress draft."""
    if slot.direction != "nf":
        raise StressCaseError("distance_direction_required")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 0 <= attempt < 64
    ):
        raise StressCaseError("invalid_attempt")
    if slot.family == "target_boundary":
        return _target_boundary_case(slot, attempt)
    if slot.family == "obstacle_corner":
        return _obstacle_case(slot, attempt, tied=False)
    if slot.family == "support_boundary":
        return _support_boundary_case(slot, attempt)
    if slot.family == "preservation_intersection":
        return _preservation_intersection_case(slot, attempt)
    if slot.family == "tied_optimum":
        return _obstacle_case(slot, attempt, tied=True)
    if slot.family == "room_bounds":
        return _room_bounds_case(slot, attempt)
    if slot.family == "support_locus":
        return _support_locus_case(slot, attempt)
    if slot.family == "obstacle_coverage":
        return _obstacle_coverage_case(slot, attempt)
    if slot.family == "relation_upper_bound":
        return _relation_upper_bound_case(slot, attempt)
    raise StressCaseError(f"unsupported distance stress family: {slot.family}")
