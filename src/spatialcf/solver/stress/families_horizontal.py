"""Independent analytic constructors for horizontal stress families."""

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


def _camera(
    *,
    constant_depth: bool = False,
    focal_px: float = 320.0,
    depth_offset: float = 4.0,
) -> Camera:
    camera = make_camera(focal_px=focal_px, depth_offset=depth_offset)
    if constant_depth:
        matrix = list(camera.world_to_camera)
        matrix[9] = 0.0
        camera = camera.model_copy(update={"world_to_camera": tuple(matrix)})
    return camera


def _sampled_y_shift(slot: StressSlot, attempt: int) -> float:
    return HashSampler(f"{slot.key}/{attempt}").grid(
        "world_y_shift",
        Decimal("-0.50"),
        Decimal("0.50"),
        Decimal("0.25"),
    )


def _sampled_room_top(
    slot: StressSlot,
    attempt: int,
    *,
    minimum_y: float,
    minimum_depth: str,
) -> float:
    sampler = HashSampler(f"{slot.key}/{attempt}")
    depth = sampler.grid(
        "room_depth",
        Decimal(minimum_depth),
        Decimal("8.00"),
        Decimal("0.25"),
    )
    return minimum_y + depth


def _draft(
    slot: StressSlot,
    attempt: int,
    *,
    room_bounds: tuple[float, float, float, float],
    camera: Camera,
    objects: tuple[SceneObject, ...],
    oracle: SatStressOracle | UnsatStressOracle,
) -> StressCaseDraft:
    return StressCaseDraft(
        seed=slot.seed,
        direction=slot.direction,
        raw_slot=slot.raw_slot,
        family=slot.family,
        scene=make_scene(
            scene_id=f"draft-{slot.seed}-lr-{slot.raw_slot:03d}-a{attempt:02d}",
            generation_seed=slot.seed,
            room_bounds=room_bounds,
            camera=camera,
            objects=objects,
        ),
        intervention=InterventionSpec(
            subject_id="subject",
            reference_id="reference",
            relation_before=Relation.LEFT,
            relation_after=Relation.RIGHT,
            camera_id=camera.camera_id,
        ),
        oracle=oracle,
    )


def _target_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    sampler = HashSampler(f"{slot.key}/{attempt}")
    focal_px = sampler.choice("focal_px", (320.0, 400.0, 500.0, 640.0))
    depth_offset = sampler.grid(
        "subject_depth",
        Decimal("2.00"),
        Decimal("4.00"),
        Decimal("0.25"),
    )
    subject_extent = sampler.choice("subject_xy_extent", (0.2, 0.4))
    reference_extent = sampler.choice("reference_xy_extent", (0.2, 0.4))
    subject_yaw = sampler.choice("subject_yaw", (0, 15, 30, 45))
    reference_yaw = sampler.choice("reference_yaw", (0, 30, 45, 60))
    camera = _camera(focal_px=focal_px, depth_offset=depth_offset)
    subject = make_object(
        "subject",
        position=Vec3(x=-0.5, y=0.0, z=1.3),
        extent=Vec3(x=subject_extent, y=subject_extent, z=1.0),
        yaw_degrees=subject_yaw,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=0.5, y=1.0, z=1.3),
        extent=Vec3(x=reference_extent, y=reference_extent, z=1.0),
        yaw_degrees=reference_yaw,
        movable=False,
        camera=camera,
    )
    operational_pixels = 32.0 - _OPERATIONAL_COMPARISON_TOLERANCE
    slope = 0.5 / (depth_offset + 1.0) + operational_pixels / focal_px
    deficit = slope * depth_offset + 0.5
    witness = Vec2(
        x=round(-0.5 + deficit / (1.0 + slope * slope), 12),
        y=round(-slope * deficit / (1.0 + slope * slope), 12),
    )
    exact = round(math.hypot(witness.x + 0.5, witness.y), 12)
    return _draft(
        slot,
        attempt,
        room_bounds=(
            -2.5,
            -1.5,
            2.5,
            _sampled_room_top(slot, attempt, minimum_y=-1.5, minimum_depth="4.00"),
        ),
        camera=camera,
        objects=(subject, reference),
        oracle=SatStressOracle(
            proof_kind="target_boundary",
            exact_infimum_m=exact,
            exact_infimum_points=(witness,),
            derivation=(
                f"orthogonal projection onto x - {slope:.12g} y >= "
                f"{slope * depth_offset:.12g}"
            ),
        ),
    )


def _constant_pair(
    camera: Camera,
    *,
    extent_xy: float = 0.4,
    reference_y: float = 0.0,
    y_shift: float = 0.0,
) -> tuple[SceneObject, SceneObject]:
    subject = make_object(
        "subject",
        position=Vec3(x=-0.5, y=y_shift, z=0.7),
        extent=Vec3(x=extent_xy, y=extent_xy, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
    )
    reference = make_object(
        "reference",
        position=Vec3(x=0.0, y=reference_y + y_shift, z=0.7),
        extent=Vec3(x=extent_xy, y=extent_xy, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    return subject, reference


def _obstacle_corner_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera(constant_depth=True)
    sampler = HashSampler(f"{slot.key}/{attempt}")
    y_shift = _sampled_y_shift(slot, attempt)
    obstacle_y_extent = sampler.choice("obstacle_y_extent", (0.4, 0.6, 0.8))
    reference_y = {0.4: 1.4, 0.6: 1.6, 0.8: 1.8}[obstacle_y_extent]
    configuration_half_height = (obstacle_y_extent + 0.4) / 2.0
    subject, reference = _constant_pair(
        camera,
        reference_y=reference_y,
        y_shift=y_shift,
    )
    obstacle = make_object(
        "obstacle",
        position=Vec3(x=0.4, y=y_shift, z=0.7),
        extent=Vec3(x=0.4, y=obstacle_y_extent, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    target_x = 0.4 - _OPERATIONAL_COMPARISON_TOLERANCE / 80.0
    witness_x = round(target_x, 12)
    return _draft(
        slot,
        attempt,
        room_bounds=(
            -2.5,
            y_shift - obstacle_y_extent / 2.0,
            2.5,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=y_shift - obstacle_y_extent / 2.0,
                minimum_depth="3.00",
            ),
        ),
        camera=camera,
        objects=(subject, reference, obstacle),
        oracle=SatStressOracle(
            proof_kind="obstacle_corner",
            exact_infimum_m=round(
                math.hypot(witness_x + 0.5, configuration_half_height),
                12,
            ),
            exact_infimum_points=(
                Vec2(x=witness_x, y=configuration_half_height + y_shift),
            ),
            derivation="nearest reachable RIGHT point is the upper configuration corner",
        ),
    )


def _support_boundary_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera()
    support = make_object(
        "support",
        position=Vec3(x=0.0, y=0.9, z=0.1),
        extent=Vec3(x=4.0, y=2.0, z=0.2),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    subject = make_object(
        "subject",
        position=Vec3(x=-0.5, y=0.0, z=0.7),
        extent=Vec3(x=0.2, y=0.2, z=1.0),
        yaw_degrees=0,
        movable=True,
        camera=camera,
        support_object_id="support",
    )
    reference = make_object(
        "reference",
        position=Vec3(x=0.5, y=1.0, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
    )
    witness_x = 0.8 - _OPERATIONAL_COMPARISON_TOLERANCE / 80.0
    witness = Vec2(x=round(witness_x, 12), y=0.0)
    return _draft(
        slot,
        attempt,
        room_bounds=(
            -2.5,
            -0.5,
            2.5,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=-0.5,
                minimum_depth="3.00",
            ),
        ),
        camera=camera,
        objects=(subject, reference, support),
        oracle=SatStressOracle(
            proof_kind="support_boundary",
            exact_infimum_m=round(
                math.hypot(witness.x + 0.5, witness.y),
                12,
            ),
            exact_infimum_points=(witness,),
            derivation="support y >= 0 clips the operational target boundary",
        ),
    )


def _preservation_intersection_case(
    slot: StressSlot,
    attempt: int,
) -> StressCaseDraft:
    camera = _camera()
    subject, reference = _constant_pair(
        camera,
        extent_xy=0.2,
        reference_y=-0.2,
    )
    operational_pixels = 32.0 - _OPERATIONAL_COMPARISON_TOLERANCE
    witness = Vec2(
        x=round(
            operational_pixels * (4.0 - _OPERATIONAL_COMPARISON_TOLERANCE) / 320.0,
            12,
        ),
        y=-_OPERATIONAL_COMPARISON_TOLERANCE,
    )
    return _draft(
        slot,
        attempt,
        room_bounds=(
            -2.5,
            -1.5,
            2.5,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=-1.5,
                minimum_depth="3.00",
            ),
        ),
        camera=camera,
        objects=(subject, reference),
        oracle=SatStressOracle(
            proof_kind="preservation_intersection",
            exact_infimum_m=round(math.hypot(witness.x + 0.5, witness.y), 12),
            exact_infimum_points=(witness,),
            derivation="target RIGHT and preserved target-pair BEHIND intersect",
        ),
    )


def _tied_optimum_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera(constant_depth=True)
    y_shift = _sampled_y_shift(slot, attempt)
    subject, reference = _constant_pair(
        camera,
        reference_y=1.4,
        y_shift=y_shift,
    )
    obstacle = make_object(
        "obstacle",
        position=Vec3(x=0.4, y=y_shift, z=0.7),
        extent=Vec3(x=0.4, y=0.4, z=1.0),
        yaw_degrees=0,
        movable=False,
        camera=camera,
        request_eligible=False,
    ).model_copy(update={"views": {}})
    target_x = 0.4 - _OPERATIONAL_COMPARISON_TOLERANCE / 80.0
    witness_x = round(target_x, 12)
    exact = round(math.hypot(witness_x + 0.5, 0.4), 12)
    return _draft(
        slot,
        attempt,
        room_bounds=(
            -2.5,
            -1.5 + y_shift,
            2.5,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=-1.5 + y_shift,
                minimum_depth="3.25",
            ),
        ),
        camera=camera,
        objects=(subject, reference, obstacle),
        oracle=SatStressOracle(
            proof_kind="tied_optimum",
            exact_infimum_m=exact,
            exact_infimum_points=(
                Vec2(x=witness_x, y=-0.4 + y_shift),
                Vec2(x=witness_x, y=0.4 + y_shift),
            ),
            derivation="symmetric upper and lower configuration corners tie",
        ),
    )


def _unsat_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    camera = _camera(constant_depth=True)
    y_shift = _sampled_y_shift(slot, attempt)
    if slot.family == "relation_upper_bound":
        subject = make_object(
            "subject",
            position=Vec3(x=3.0, y=y_shift, z=0.7),
            extent=Vec3(x=0.4, y=0.4, z=1.0),
            yaw_degrees=0,
            movable=True,
            camera=camera,
        )
        reference = make_object(
            "reference",
            position=Vec3(x=3.55, y=y_shift, z=0.7),
            extent=Vec3(x=0.4, y=0.4, z=1.0),
            yaw_degrees=0,
            movable=False,
            camera=camera,
        )
        objects = (subject, reference)
        room = (
            0.0,
            -1.5 + y_shift,
            4.0,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=-1.5 + y_shift,
                minimum_depth="3.00",
            ),
        )
        maximum = 3.8
        required = round(
            (3.95 * 80.0 - _OPERATIONAL_COMPARISON_TOLERANCE) / 80.0,
            12,
        )
    else:
        subject, reference = _constant_pair(camera, y_shift=y_shift)
        objects = (subject, reference)
        room = (
            -3.5,
            -1.5 + y_shift,
            0.5,
            _sampled_room_top(
                slot,
                attempt,
                minimum_y=-1.5 + y_shift,
                minimum_depth="3.00",
            ),
        )
        maximum = 0.3
        required = round(
            0.4 - _OPERATIONAL_COMPARISON_TOLERANCE / 80.0,
            12,
        )
        if slot.family == "support_locus":
            room = (
                -2.5,
                -1.5 + y_shift,
                2.5,
                _sampled_room_top(
                    slot,
                    attempt,
                    minimum_y=-1.5 + y_shift,
                    minimum_depth="3.00",
                ),
            )
            support = make_object(
                "support",
                position=Vec3(x=-0.5, y=y_shift, z=0.1),
                extent=Vec3(x=2.0, y=2.0, z=0.2),
                yaw_degrees=0,
                movable=False,
                camera=camera,
                request_eligible=False,
            ).model_copy(update={"views": {}})
            subject = subject.model_copy(update={"support_object_id": "support"})
            objects = (subject, reference, support)
        elif slot.family == "obstacle_coverage":
            room = (-2.25, -1.5 + y_shift, 1.75, 1.5 + y_shift)
            walls = tuple(
                make_object(
                    object_id,
                    position=Vec3(x=1.0, y=wall_y + y_shift, z=0.7),
                    extent=Vec3(x=1.2, y=1.2, z=1.0),
                    yaw_degrees=0,
                    movable=False,
                    camera=camera,
                    request_eligible=False,
                ).model_copy(update={"views": {}})
                for object_id, wall_y in (
                    ("wall_lower", -0.7),
                    ("wall_upper", 0.7),
                )
            )
            objects = (subject, reference, *walls)
            maximum = 0.2
    return _draft(
        slot,
        attempt,
        room_bounds=room,
        camera=camera,
        objects=objects,
        oracle=UnsatStressOracle(
            proof_kind=slot.family,
            maximum_possible_value_m=maximum,
            required_value_m=required,
            expected_reason="empty_outer_region",
            derivation=f"{slot.family} keeps maximum RIGHT value below threshold",
        ),
    )


def build_horizontal_case(slot: StressSlot, attempt: int) -> StressCaseDraft:
    """Build one direct LEFT-to-RIGHT stress draft without production solver code."""
    if slot.direction != "lr":
        raise StressCaseError("horizontal_direction_required")
    if not 0 <= attempt < 64:
        raise StressCaseError("invalid_attempt")
    if slot.family == "target_boundary":
        return _target_boundary_case(slot, attempt)
    if slot.family == "obstacle_corner":
        return _obstacle_corner_case(slot, attempt)
    if slot.family == "support_boundary":
        return _support_boundary_case(slot, attempt)
    if slot.family == "preservation_intersection":
        return _preservation_intersection_case(slot, attempt)
    if slot.family == "tied_optimum":
        return _tied_optimum_case(slot, attempt)
    return _unsat_case(slot, attempt)
