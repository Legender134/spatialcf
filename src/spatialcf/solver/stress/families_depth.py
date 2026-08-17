"""Independent analytic constructors for depth stress families."""

from __future__ import annotations

import math
from decimal import Decimal

from spatialcf.domain.enums import Relation
from spatialcf.domain.models import Camera, InterventionSpec, SceneObject, Vec2, Vec3
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

# Frozen from the public relation-label boundary contract; constructors derive
# their oracle geometry without importing production relation code.
_OPERATIONAL_COMPARISON_TOLERANCE = 1e-9


def _target_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    sampler = HashSampler(f"{slot.key}/{attempt}")
    focal_px = sampler.choice("focal_px", (40.0, 50.0, 80.0, 100.0))
    depth_offset = sampler.grid(
        "depth_offset",
        Decimal("2.00"),
        Decimal("4.00"),
        Decimal("0.25"),
    )
    subject_extent = sampler.choice("subject_extent", (0.2, 0.4, 0.6))
    reference_extent = sampler.choice("reference_extent", (0.2, 0.4, 0.6))
    subject_yaw = sampler.choice("subject_yaw", (0, 15, 30, 45, 60, 75))
    reference_yaw = sampler.choice("reference_yaw", (0, 30, 45, 60, 90))
    camera = make_camera(focal_px=focal_px, depth_offset=depth_offset)
    subject = make_object(
        "subject",
        position=Vec3(x=-4.0, y=-0.5, z=0.7),
        extent=Vec3(x=subject_extent, y=subject_extent, z=1.0),
        yaw_degrees=subject_yaw,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=0.0, y=0.5, z=0.7),
        extent=Vec3(x=reference_extent, y=reference_extent, z=1.0),
        yaw_degrees=reference_yaw,
        movable=False,
        camera=camera,
    )
    witness_y = 0.7 - _OPERATIONAL_COMPARISON_TOLERANCE
    return _draft(
        slot,
        attempt,
        room_bounds=(-4.5, -1.0, 4.5, 2.0),
        objects=(subject, reference),
        camera=camera,
        oracle=SatStressOracle(
            proof_kind="target_boundary",
            exact_infimum_m=round(1.2 - _OPERATIONAL_COMPARISON_TOLERANCE, 12),
            exact_infimum_points=(Vec2(x=-4.0, y=round(witness_y, 12)),),
            derivation=(
                f"reference depth {depth_offset + 0.5:.2f} m plus the 0.2 m "
                "BEHIND threshold gives world y = 0.7 m"
            ),
        ),
    )


def _pair(
    camera: Camera,
    *,
    subject_x: float = -2.5,
) -> tuple[SceneObject, SceneObject]:
    subject = make_object(
        "subject",
        position=Vec3(x=subject_x, y=-0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=1.0, y=0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    return subject, reference


def _slanted_camera() -> Camera:
    camera = make_camera(focal_px=80.0, depth_offset=4.0)
    matrix = list(camera.world_to_camera)
    matrix[8] = 1.0
    return camera.model_copy(update={"world_to_camera": tuple(matrix)})


def _sampled_room(
    slot: StressSlot,
    attempt: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bounds
    base_depth = max_y - min_y
    maximum_steps = min(8, round((8.0 - base_depth) / 0.25))
    extra_steps = HashSampler(f"{slot.key}/{attempt}").integer(
        "room_depth_extra",
        0,
        maximum_steps,
    )
    return min_x, min_y - 0.25 * extra_steps, max_x, max_y


def _draft(
    slot: StressSlot,
    attempt: int,
    *,
    room_bounds: tuple[float, float, float, float],
    objects: tuple[SceneObject, ...],
    oracle: SatStressOracle | UnsatStressOracle,
    camera: Camera | None = None,
) -> StressCaseDraft:
    scene_camera = camera or make_camera(focal_px=80.0, depth_offset=3.0)
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-fb-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=_sampled_room(slot, attempt, room_bounds),
            camera=scene_camera,
            objects=objects,
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.FRONT,
            relation_after=Relation.BEHIND,
            camera_id=scene_camera.camera_id,
        ),
        oracle=oracle,
    )


def _obstacle_case(
    slot: StressSlot,
    attempt: int,
    *,
    tied: bool,
) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    sampler = HashSampler(f"{slot.key}/{attempt}")
    subject, reference = _pair(camera, subject_x=-2.0 if tied else -2.5)
    blocker_extent_x = sampler.choice("blocker_extent_x", (0.8, 1.0, 1.2))
    blocker_extent_y = sampler.choice("blocker_extent_y", (0.4, 0.6, 0.8))
    blocker = make_object(
        "blocker",
        position=Vec3(x=-2.0, y=0.8, z=0.7),
        extent=Vec3(x=blocker_extent_x, y=blocker_extent_y, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    configuration_half_width = (blocker_extent_x + 0.4) / 2.0
    x_offsets = (
        (-configuration_half_width, configuration_half_width)
        if tied
        else (0.5 - configuration_half_width,)
    )
    witness_y = 0.7 - _OPERATIONAL_COMPARISON_TOLERANCE
    witnesses = tuple(
        Vec2(x=subject.position.x + offset, y=round(witness_y, 12))
        for offset in x_offsets
    )
    exact = round(
        math.hypot(
            abs(x_offsets[0]),
            1.2 - _OPERATIONAL_COMPARISON_TOLERANCE,
        ),
        12,
    )
    family = "tied_optimum" if tied else "obstacle_corner"
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -1.0, 3.0, 2.0),
        objects=(subject, reference, blocker),
        oracle=SatStressOracle(
            proof_kind=family,
            exact_infimum_m=exact,
            exact_infimum_points=witnesses,
            derivation=(
                "symmetric configuration-edge contacts tie"
                if tied
                else "the nearer configuration-edge contact is uniquely optimal"
            ),
        ),
    )


def _room_bounds_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    subject, reference = _pair(camera)
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -2.2, 3.0, 0.8),
        objects=(subject, reference),
        oracle=UnsatStressOracle(
            proof_kind="room_bounds",
            maximum_possible_value_m=0.6,
            required_value_m=round(
                0.7 - _OPERATIONAL_COMPARISON_TOLERANCE,
                12,
            ),
            expected_reason="empty_outer_region",
            derivation="room top 0.8 minus subject half-depth 0.2 gives y <= 0.6",
        ),
    )


def _support_locus_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    subject, reference = _pair(camera)
    support = make_object(
        "support",
        position=Vec3(x=-1.0, y=-0.2, z=0.1),
        extent=Vec3(x=4.0, y=2.0, z=0.2),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    subject = subject.model_copy(update={"support_object_id": "support"})
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -2.0, 3.0, 2.0),
        objects=(subject, reference, support),
        oracle=UnsatStressOracle(
            proof_kind="support_locus",
            maximum_possible_value_m=0.6,
            required_value_m=round(
                0.7 - _OPERATIONAL_COMPARISON_TOLERANCE,
                12,
            ),
            expected_reason="empty_outer_region",
            derivation="support top edge 0.8 minus subject half-depth 0.2 gives y <= 0.6",
        ),
    )


def _obstacle_coverage_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    subject, reference = _pair(camera)
    blockers = tuple(
        make_object(
            f"blocker_{index}",
            position=Vec3(x=x, y=1.35, z=0.7),
            extent=Vec3(x=1.2, y=1.2, z=1.0),
            yaw_degrees=0,
            movable=False,
            camera=camera,
            request_eligible=False,
        ).model_copy(update={"views": {}})
        for index, x in enumerate((-2.2, -0.65, 0.9))
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(-3.0, -1.0, 1.75, 2.0),
        objects=(subject, reference, *blockers),
        oracle=UnsatStressOracle(
            proof_kind="obstacle_coverage",
            maximum_possible_value_m=0.55,
            required_value_m=round(
                0.7 - _OPERATIONAL_COMPARISON_TOLERANCE,
                12,
            ),
            expected_reason="empty_outer_region",
            derivation="three configuration strips cover every center with y >= 0.55",
        ),
    )


def _support_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    matrix = list(camera.world_to_camera)
    matrix[8] = 0.5
    camera = camera.model_copy(update={"world_to_camera": tuple(matrix)})
    subject = make_object(
        "subject",
        position=Vec3(x=-2.5, y=-0.5, z=0.7),
        extent=Vec3(x=0.2, y=0.2, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
        support_object_id="support",
    )
    reference = make_object(
        "reference",
        position=Vec3(x=1.0, y=0.5, z=0.7),
        extent=Vec3(x=0.2, y=0.2, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    support = make_object(
        "support",
        position=Vec3(x=-0.1, y=0.45, z=0.1),
        extent=Vec3(x=5.0, y=2.5, z=0.2),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    witness = Vec2(
        x=round(-0.8 - 2.0 * _OPERATIONAL_COMPARISON_TOLERANCE, 12),
        y=1.6,
    )
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-fb-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=_sampled_room(
                slot,
                attempt,
                (-3.0, -1.0, 3.0, 2.0),
            ),
            camera=camera,
            objects=(subject, reference, support),
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.FRONT,
            relation_after=Relation.BEHIND,
            camera_id=camera.camera_id,
        ),
        oracle=SatStressOracle(
            proof_kind="support_boundary",
            exact_infimum_m=round(
                math.hypot(
                    1.7 - 2.0 * _OPERATIONAL_COMPARISON_TOLERANCE,
                    2.1,
                ),
                12,
            ),
            exact_infimum_points=(witness,),
            derivation=(
                "the operational depth line meets the support center edge y = 1.6"
            ),
        ),
    )


def _preservation_intersection_case(
    slot: StressSlot,
    attempt: int,
) -> StressCaseDraft:
    camera = make_camera(focal_px=80.0, depth_offset=3.0)
    subject = make_object(
        "subject",
        position=Vec3(x=-2.0, y=-0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=-0.2, y=0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    witness = Vec2(
        x=-2.1,
        y=round(0.7 - _OPERATIONAL_COMPARISON_TOLERANCE, 12),
    )
    exact = round(
        math.hypot(0.1, 1.2 - _OPERATIONAL_COMPARISON_TOLERANCE),
        12,
    )
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-fb-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=_sampled_room(
                slot,
                attempt,
                (-3.0, -1.0, 2.0, 2.0),
            ),
            camera=camera,
            objects=(subject, reference),
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.FRONT,
            relation_after=Relation.BEHIND,
            camera_id=camera.camera_id,
        ),
        oracle=SatStressOracle(
            proof_kind="preservation_intersection",
            exact_infimum_m=exact,
            exact_infimum_points=(witness,),
            derivation=(
                "the target depth line y = 0.7 meets the preserved FAR side x = -2.1"
            ),
        ),
    )


def _relation_upper_bound_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _slanted_camera()
    subject = make_object(
        "subject",
        position=Vec3(x=-1.0, y=-0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=1.0, y=0.5, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    reference_depth = 5.5
    reference_u = 320.0 + 80.0 / reference_depth
    preserved_u = reference_u - (32.0 - _OPERATIONAL_COMPARISON_TOLERANCE)
    ratio = (preserved_u - 320.0) / 80.0
    maximum_depth = subject.position.x / ratio
    maximum_linear_value = (maximum_depth - 4.0) / math.sqrt(2.0)
    required_linear_value = (1.7 - _OPERATIONAL_COMPARISON_TOLERANCE) / math.sqrt(2.0)
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-fb-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=_sampled_room(
                slot,
                attempt,
                (-1.2, -1.0, 3.8, 4.0),
            ),
            camera=camera,
            objects=(subject, reference),
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.FRONT,
            relation_after=Relation.BEHIND,
            camera_id=camera.camera_id,
        ),
        oracle=UnsatStressOracle(
            proof_kind="relation_upper_bound",
            maximum_possible_value_m=round(maximum_linear_value, 12),
            required_value_m=round(required_linear_value, 12),
            expected_reason="empty_outer_region",
            derivation="preserved LEFT and room x >= -1 strictly cap calibrated depth",
        ),
    )


def build_depth_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    """Build one deterministic direct FRONT-to-BEHIND stress draft."""
    if slot.direction != "fb":
        raise StressCaseError("depth builder requires direction 'fb'")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 0 <= attempt < 64
    ):
        raise StressCaseError("attempt must be an integer from 0 through 63")
    if slot.family == "target_boundary":
        return _target_boundary_case(slot, attempt)
    if slot.family == "obstacle_corner":
        return _obstacle_case(slot, attempt, tied=False)
    if slot.family == "tied_optimum":
        return _obstacle_case(slot, attempt, tied=True)
    if slot.family == "room_bounds":
        return _room_bounds_case(slot, attempt)
    if slot.family == "support_locus":
        return _support_locus_case(slot, attempt)
    if slot.family == "obstacle_coverage":
        return _obstacle_coverage_case(slot, attempt)
    if slot.family == "support_boundary":
        return _support_boundary_case(slot, attempt)
    if slot.family == "preservation_intersection":
        return _preservation_intersection_case(slot, attempt)
    if slot.family == "relation_upper_bound":
        return _relation_upper_bound_case(slot, attempt)
    raise StressCaseError(f"unsupported depth stress family: {slot.family}")
