"""Source-only target-reachability evidence for the current roster.

This module deliberately proves only that at least one frozen native placement
coordinate satisfies the requested *target relation*.  It does not compile
support/collision/visibility, call the full solver, or consume native execution
outcomes.  The complete solver and final platform audit remain the only
acceptance authorities.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, model_validator

from spatialcf.core.v2.continuous_yaw_camera_frame import (
    bound_world_point_in_upright_camera_v2_9,
    compile_upright_camera_context_v2_9,
    prepare_camera_independent_candidate_problem_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _divide_positive as _divide_positive_interval_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _extract_target as _extract_directional_target_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _scale as _scale_interval_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _subtract as _subtract_intervals_v2_9,
)
from spatialcf.core.v2.continuous_yaw_directional_relation import (
    _UnsupportedDirectionalTargetV2,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _expanded_universe as _expanded_target_universe_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _extract_target as _extract_shape_gap_target_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _inclusive_complement_complex as _inclusive_shape_gap_complement_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _offset_and_clip_polygon as _offset_and_clip_target_polygon_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _target_complexes as _shape_gap_target_complexes_v2_9,
)
from spatialcf.core.v2.continuous_yaw_target_relation import (
    _UnsupportedTargetRelationV2,
)
from spatialcf.core.v2.convex_translation_domain import (
    ConvexTranslationDomainKindV2,
    RationalPoint2V2,
    compile_convex_translation_obstacle_v2,
)
from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
)
from spatialcf.core.v2.so2_interval import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
)
from spatialcf.core.v2.strict_convex_intersection import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
)
from spatialcf.domain.enums import Relation
from spatialcf.domain.models import InterventionSpec, Scene
from spatialcf.domain.v2.base import Sha256Digest, V2Model
from spatialcf.domain.v2.constraints import (
    MeasurementComparatorV2,
    RelationMeasurementV2,
    RelationV2,
)
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_REACHABLE_POSITIONS_HASH_DOMAIN_V2_9_4 = (
    "spatialcf.competition-native-reachable-native-positions.v2.9.4"
)
_TARGET_REACHABILITY_HASH_DOMAIN_V2_9_4 = (
    "spatialcf.competition-native-candidate-target-reachability.v2.9.4"
)
_PREPARED_SOURCE_CAPABILITY_V2_9_4 = object()


class CompetitionNativeTargetReachabilityStatusV2_9_4(StrEnum):
    REACHABLE = "REACHABLE_NATIVE_TARGET"
    UNREACHABLE = "NO_REACHABLE_NATIVE_TARGET"


@dataclass(frozen=True, slots=True)
class CompetitionNativePreparedTargetReachabilitySourceV2_9_4:
    """Process-local, once-validated source context for candidate derivation."""

    _capability: object = field(repr=False, compare=False)
    capture: object
    source_surface_evidence: object
    camera_evidence: object
    placements_by_id: Mapping[str, object]
    subject_surfaces_by_id: Mapping[str, object]


def _canonical_coordinate_payload_v2_9_4(
    coordinates: Iterable[tuple[Fraction, Fraction]],
) -> tuple[tuple[str, str], ...]:
    checked = tuple(coordinates)
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not Fraction
        or type(item[1]) is not Fraction
        for item in checked
    ):
        raise TypeError("reachable native coordinates must contain exact fractions")
    return tuple(sorted({(str(x), str(y)) for x, y in checked}))


def competition_native_reachable_native_positions_sha256_v2_9_4(
    coordinates: Iterable[tuple[Fraction, Fraction]],
) -> Sha256Digest:
    """Hash one canonical set of target-reachable native XY deltas."""

    return canonical_sha256_v2(
        _canonical_coordinate_payload_v2_9_4(coordinates),
        domain=_REACHABLE_POSITIONS_HASH_DOMAIN_V2_9_4,
    )


class CompetitionNativeCandidateTargetReachabilityV2_9_4(V2Model):
    reachability_version: Literal[
        "competition-native-candidate-target-reachability:2.9.4"
    ] = "competition-native-candidate-target-reachability:2.9.4"
    authority: Literal[
        "SOURCE_ONLY_CANONICAL_TARGET_INNER_NOT_SOLVER_OR_NATIVE_OUTCOME"
    ] = "SOURCE_ONLY_CANONICAL_TARGET_INNER_NOT_SOLVER_OR_NATIVE_OUTCOME"
    candidate_id: str = Field(pattern=r"^candidate-[0-9a-f]{64}$")
    source_id: str = Field(strict=True, min_length=1, max_length=512)
    scene_id: str = Field(strict=True, min_length=1, max_length=512)
    source_capture_sha256: Sha256Digest
    subject_id: str = Field(strict=True, min_length=1, max_length=512)
    reference_id: str = Field(strict=True, min_length=1, max_length=512)
    relation_before: Relation
    relation_after: Relation
    placement_sha256: Sha256Digest
    surface_evidence_sha256: Sha256Digest
    subject_surface_evidence_sha256: Sha256Digest
    camera_evidence_sha256: Sha256Digest
    native_position_count: int = Field(strict=True, ge=1, le=10_000)
    reachable_native_position_count: int = Field(strict=True, ge=0, le=10_000)
    reachable_native_positions_sha256: Sha256Digest
    status: CompetitionNativeTargetReachabilityStatusV2_9_4
    target_reachability_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_reachability(self) -> Self:
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("target reachability relation pair is not opposite")
        if self.reachable_native_position_count > self.native_position_count:
            raise ValueError("reachable native position count exceeds source count")
        reachable = self.reachable_native_position_count > 0
        if reachable != (
            self.status is CompetitionNativeTargetReachabilityStatusV2_9_4.REACHABLE
        ):
            raise ValueError("target reachability terminal status is not closed")
        payload = self.model_dump(
            mode="python",
            exclude={"target_reachability_sha256"},
            warnings="error",
        )
        if self.target_reachability_sha256 != canonical_sha256_v2(
            payload,
            domain=_TARGET_REACHABILITY_HASH_DOMAIN_V2_9_4,
        ):
            raise ValueError("target reachability digest mismatch")
        return self


def build_competition_native_candidate_target_reachability_v2_9_4(
    *,
    candidate_id: str,
    source_id: str,
    scene_id: str,
    source_capture_sha256: str,
    subject_id: str,
    reference_id: str,
    relation_before: Relation,
    placement_sha256: str,
    surface_evidence_sha256: str,
    subject_surface_evidence_sha256: str,
    camera_evidence_sha256: str,
    native_position_count: int,
    reachable_native_coordinates: frozenset[tuple[Fraction, Fraction]],
) -> CompetitionNativeCandidateTargetReachabilityV2_9_4:
    """Build one immutable source-only reachability row."""

    if type(relation_before) is not Relation:
        raise TypeError("target reachability relation_before must be exact")
    if type(reachable_native_coordinates) is not frozenset:
        raise TypeError("reachable native coordinates must be an exact frozenset")
    coordinate_payload = _canonical_coordinate_payload_v2_9_4(
        reachable_native_coordinates
    )
    reachable_count = len(coordinate_payload)
    payload = {
        "reachability_version": (
            "competition-native-candidate-target-reachability:2.9.4"
        ),
        "authority": (
            "SOURCE_ONLY_CANONICAL_TARGET_INNER_NOT_SOLVER_OR_NATIVE_OUTCOME"
        ),
        "candidate_id": candidate_id,
        "source_id": source_id,
        "scene_id": scene_id,
        "source_capture_sha256": source_capture_sha256,
        "subject_id": subject_id,
        "reference_id": reference_id,
        "relation_before": relation_before,
        "relation_after": relation_before.opposite,
        "placement_sha256": placement_sha256,
        "surface_evidence_sha256": surface_evidence_sha256,
        "subject_surface_evidence_sha256": subject_surface_evidence_sha256,
        "camera_evidence_sha256": camera_evidence_sha256,
        "native_position_count": native_position_count,
        "reachable_native_position_count": reachable_count,
        "reachable_native_positions_sha256": canonical_sha256_v2(
            coordinate_payload,
            domain=_REACHABLE_POSITIONS_HASH_DOMAIN_V2_9_4,
        ),
        "status": (
            CompetitionNativeTargetReachabilityStatusV2_9_4.REACHABLE
            if reachable_count
            else CompetitionNativeTargetReachabilityStatusV2_9_4.UNREACHABLE
        ),
    }
    return CompetitionNativeCandidateTargetReachabilityV2_9_4(
        **payload,
        target_reachability_sha256=canonical_sha256_v2(
            payload,
            domain=_TARGET_REACHABILITY_HASH_DOMAIN_V2_9_4,
        ),
    )


def competition_native_target_only_relation_after_native_coordinates_v2_9_4(
    problem: SemanticProblemV2_3,
    native_points: tuple[RationalPoint2V2, ...],
) -> frozenset[tuple[Fraction, Fraction]]:
    """Prove fixed-yaw target membership without solver/native outcomes."""

    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("target-only reachability problem must be exact")
    if type(native_points) is not tuple or any(
        type(point) is not RationalPoint2V2 for point in native_points
    ):
        raise TypeError("target-only native points must be an exact tuple")
    if not native_points:
        return frozenset()
    target = problem.constraints.target_relation
    try:
        if target.relation_after in (
            RelationV2.LEFT,
            RelationV2.RIGHT,
            RelationV2.FRONT,
            RelationV2.BEHIND,
        ):
            return _target_only_directional_native_coordinates_v2_9_4(
                problem, native_points
            )
        if target.relation_after in (RelationV2.NEAR, RelationV2.FAR):
            return _target_only_shape_gap_native_coordinates_v2_9_4(
                problem, native_points
            )
        raise RuntimeError("supported target relation escaped its six-way partition")
    except (
        ArithmeticError,
        RuntimeWarning,
        SO2AtomicBudgetExhaustedV2,
        StrictConvexIntersectionBudgetExhaustedV2,
        _UnsupportedDirectionalTargetV2,
        _UnsupportedTargetRelationV2,
    ):
        return frozenset()


def competition_native_target_only_relation_after_native_coordinate_margins_v2_9_4(
    problem: SemanticProblemV2_3,
    native_points: tuple[RationalPoint2V2, ...],
) -> tuple[tuple[tuple[Fraction, Fraction], Fraction], ...]:
    """Return exact source-only target membership with a stable interior margin.

    Directional margins measure the proven distance from the closed target
    threshold in the relation's native measurement.  Shape-gap relations retain
    their exact membership and use a conservative zero margin until their
    polygonal distance certificate has a separately versioned representation.
    """

    if type(problem) is not SemanticProblemV2_3:
        raise TypeError("target-only reachability problem must be exact")
    if type(native_points) is not tuple or any(
        type(point) is not RationalPoint2V2 for point in native_points
    ):
        raise TypeError("target-only native points must be an exact tuple")
    if not native_points:
        return ()
    target = problem.constraints.target_relation
    try:
        if target.relation_after in (
            RelationV2.LEFT,
            RelationV2.RIGHT,
            RelationV2.FRONT,
            RelationV2.BEHIND,
        ):
            return _target_only_directional_native_coordinate_margins_v2_9_4(
                problem,
                native_points,
            )
        if target.relation_after in (RelationV2.NEAR, RelationV2.FAR):
            return tuple(
                (coordinate, Fraction())
                for coordinate in sorted(
                    _target_only_shape_gap_native_coordinates_v2_9_4(
                        problem,
                        native_points,
                    )
                )
            )
        raise RuntimeError("supported target relation escaped its six-way partition")
    except (
        ArithmeticError,
        RuntimeWarning,
        SO2AtomicBudgetExhaustedV2,
        StrictConvexIntersectionBudgetExhaustedV2,
        _UnsupportedDirectionalTargetV2,
        _UnsupportedTargetRelationV2,
    ):
        return ()


def _target_only_directional_native_coordinates_v2_9_4(
    problem: SemanticProblemV2_3,
    native_points: tuple[RationalPoint2V2, ...],
) -> frozenset[tuple[Fraction, Fraction]]:
    return frozenset(
        coordinate
        for coordinate, _margin in (
            _target_only_directional_native_coordinate_margins_v2_9_4(
                problem,
                native_points,
            )
        )
    )


def _target_only_directional_native_coordinate_margins_v2_9_4(
    problem: SemanticProblemV2_3,
    native_points: tuple[RationalPoint2V2, ...],
) -> tuple[tuple[tuple[Fraction, Fraction], Fraction], ...]:
    projected = prepare_camera_independent_candidate_problem_v2_9(problem)
    atomic = SO2AtomicBudgetV2(limit=10_000 + 100 * len(native_points))
    domain = StrictConvexIntersectionBudgetV2(
        max_domain_operations=10_000 + 20 * len(native_points),
        max_candidate_cells=1,
    )
    camera = compile_upright_camera_context_v2_9(
        problem,
        atomic_budget=atomic,
        domain_budget=domain,
    )
    _, definition, subject, reference = _extract_directional_target_v2_9(
        problem,
        projected,
        camera,
        domain,
    )
    reference_bounds = bound_world_point_in_upright_camera_v2_9(
        camera,
        world_xyz=reference,
        delta_x=(Fraction(), Fraction()),
        delta_y=(Fraction(), Fraction()),
        atomic_budget=atomic,
    )
    threshold = Fraction.from_float(definition.threshold)
    selected: dict[tuple[Fraction, Fraction], Fraction] = {}
    for point in native_points:
        subject_bounds = bound_world_point_in_upright_camera_v2_9(
            camera,
            world_xyz=subject,
            delta_x=(point.x, point.x),
            delta_y=(point.y, point.y),
            atomic_budget=atomic,
        )
        if definition.measurement is RelationMeasurementV2.PROJECTED_CENTER_DELTA_X:
            if not subject_bounds.positive_depth or not reference_bounds.positive_depth:
                continue
            if camera.intrinsics[0] <= 0:
                return ()
            subject_ratio = _divide_positive_interval_v2_9(
                subject_bounds.x_camera,
                subject_bounds.z_camera,
                atomic,
            )
            reference_ratio = _divide_positive_interval_v2_9(
                reference_bounds.x_camera,
                reference_bounds.z_camera,
                atomic,
            )
            measurement = _scale_interval_v2_9(
                _subtract_intervals_v2_9(subject_ratio, reference_ratio, atomic),
                camera.intrinsics[0],
                atomic,
            )
        elif definition.measurement is RelationMeasurementV2.CAMERA_DEPTH_DELTA:
            measurement = _subtract_intervals_v2_9(
                subject_bounds.z_camera,
                reference_bounds.z_camera,
                atomic,
            )
        else:
            raise RuntimeError("directional target escaped its measurement partition")
        if definition.comparator is MeasurementComparatorV2.LESS_THAN:
            margin = threshold - measurement[1]
        else:
            margin = measurement[0] - threshold
        if margin >= 0:
            selected[(point.x, point.y)] = margin
    return tuple(sorted(selected.items()))


def _target_only_shape_gap_native_coordinates_v2_9_4(
    problem: SemanticProblemV2_3,
    native_points: tuple[RationalPoint2V2, ...],
) -> frozenset[tuple[Fraction, Fraction]]:
    projected = prepare_camera_independent_candidate_problem_v2_9(problem)
    atomic = SO2AtomicBudgetV2(limit=100_000)
    domain = StrictConvexIntersectionBudgetV2(
        max_domain_operations=100_000,
        max_candidate_cells=1_000,
    )
    _, relation, threshold, subject, reference = _extract_shape_gap_target_v2_9(
        projected,
        domain,
    )
    universe = _native_point_universe_v2_9_4(native_points)
    expanded = _expanded_target_universe_v2_9(universe, threshold, atomic)
    overlap = compile_convex_translation_obstacle_v2(
        subject[0],
        subject[1],
        reference[0],
        reference[1],
        expanded,
        atomic_budget=atomic,
    )
    if overlap.kind is ConvexTranslationDomainKindV2.IDENTITY:
        inner_near = None
        outer_near = None
    elif (
        overlap.kind is ConvexTranslationDomainKindV2.BRACKET
        and overlap.bracket is not None
    ):
        inner_near = _offset_and_clip_target_polygon_v2_9(
            overlap.bracket.inner_forbidden,
            threshold,
            square=False,
            universe=universe,
            atomic_budget=atomic,
        )
        outer_near = _offset_and_clip_target_polygon_v2_9(
            overlap.bracket.outer_forbidden,
            threshold,
            square=True,
            universe=universe,
            atomic_budget=atomic,
        )
    else:
        return frozenset()
    if relation is RelationV2.FAR:
        inner = _inclusive_shape_gap_complement_v2_9(
            outer_near,
            universe,
            atomic,
            domain,
        )
    else:
        inner, _ = _shape_gap_target_complexes_v2_9(
            relation,
            inner_near,
            outer_near,
            universe,
            atomic,
            domain,
        )
    return frozenset(
        (point.x, point.y) for point in native_points if inner.contains_point(point)
    )


def _native_point_universe_v2_9_4(
    native_points: tuple[RationalPoint2V2, ...],
) -> ExactAxisAlignedRectV2:
    padding = Fraction(1)
    return ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=min(point.x for point in native_points) - padding,
        min_y_m=min(point.y for point in native_points) - padding,
        max_x_m=max(point.x for point in native_points) + padding,
        max_y_m=max(point.y for point in native_points) + padding,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )


def reduce_competition_native_target_proposal_scene_v2_9_4(
    scene: Scene,
    intervention: InterventionSpec,
) -> Scene:
    """Keep subject/reference/support ancestry for target-only compilation."""

    if type(scene) is not Scene or type(intervention) is not InterventionSpec:
        raise TypeError("target proposal inputs must be exact")
    by_id = {item.object_id: item for item in scene.objects}
    keep = {intervention.subject_id, intervention.reference_id}
    frontier = list(keep)
    while frontier:
        object_id = frontier.pop()
        item = by_id.get(object_id)
        if item is None:
            raise ValueError("target proposal object is absent")
        support_id = item.support_object_id
        if support_id is not None and support_id not in keep:
            if support_id not in by_id:
                raise ValueError("target proposal support is absent")
            keep.add(support_id)
            frontier.append(support_id)
    payload = scene.model_dump(mode="python")
    payload["objects"] = tuple(item for item in scene.objects if item.object_id in keep)
    payload["collision_obstacles"] = tuple(
        item for item in scene.collision_obstacles if item.source_object_id in keep
    )
    payload["subject_position_regions"] = tuple(
        item
        for item in scene.subject_position_regions
        if item.subject_object_id in keep
    )
    payload["pinned_object_ids"] = frozenset(
        item for item in scene.pinned_object_ids if item in keep
    )
    return Scene.model_validate(payload, strict=True)


def prepare_competition_native_target_reachability_source_v2_9_4(
    capture: object,
    source_surface_evidence: object,
    camera_evidence: object,
) -> CompetitionNativePreparedTargetReachabilitySourceV2_9_4:
    """Strictly validate and index one frozen source exactly once."""

    from spatialcf.generation._internal.evidence.camera import (
        SourceCameraEvidence,
    )
    from spatialcf.generation._internal.evidence.surface import (
        SourceSurfaceEvidence,
    )
    from spatialcf.generation.capture.models import (
        CompetitionNativeSourceCaptureV2_9,
    )

    if type(capture) is not CompetitionNativeSourceCaptureV2_9:
        raise TypeError("target reachability capture must be exact")
    checked_capture = CompetitionNativeSourceCaptureV2_9.model_validate(
        capture.model_dump(mode="python"), strict=True
    )
    if type(source_surface_evidence) is not SourceSurfaceEvidence:
        raise TypeError("target reachability surface evidence must be exact")
    checked_surface = SourceSurfaceEvidence.model_validate(
        source_surface_evidence.model_dump(mode="python"), strict=True
    )
    if type(camera_evidence) is not SourceCameraEvidence:
        raise TypeError("target reachability camera evidence must be exact")
    checked_camera = SourceCameraEvidence.model_validate(
        camera_evidence.model_dump(mode="python"), strict=True
    )
    if checked_surface.source_id != checked_capture.source.source_id or (
        checked_surface.source_capture_sha256 != checked_capture.source_capture_sha256
    ):
        raise ValueError("target reachability surface evidence does not bind capture")
    if checked_camera.source_id != checked_capture.source.source_id or (
        checked_camera.source_capture_sha256 != checked_capture.source_capture_sha256
    ):
        raise ValueError("target reachability camera evidence does not bind capture")
    placements_by_id = {
        item.object_id: item for item in checked_capture.placement_facts
    }
    subject_surfaces_by_id = {
        item.subject_object_id: item for item in checked_surface.subjects
    }
    if len(placements_by_id) != len(checked_capture.placement_facts) or len(
        subject_surfaces_by_id
    ) != len(checked_surface.subjects):
        raise ValueError("target reachability source indexes are not unique")
    return CompetitionNativePreparedTargetReachabilitySourceV2_9_4(
        _capability=_PREPARED_SOURCE_CAPABILITY_V2_9_4,
        capture=checked_capture,
        source_surface_evidence=checked_surface,
        camera_evidence=checked_camera,
        placements_by_id=MappingProxyType(placements_by_id),
        subject_surfaces_by_id=MappingProxyType(subject_surfaces_by_id),
    )


def derive_competition_native_candidate_target_reachability_from_prepared_v2_9_4(
    *,
    candidate_id: str,
    prepared_source: CompetitionNativePreparedTargetReachabilitySourceV2_9_4,
    subject_id: str,
    reference_id: str,
    relation_before: Relation,
) -> CompetitionNativeCandidateTargetReachabilityV2_9_4:
    """Derive one row from an already validated frozen source context.

    Imports that depend on the candidate-roster module are deliberately local:
    the roster owns source-capture schemas, while this module owns only the
    target-reachability algorithm and ledger schema.
    """

    from spatialcf.generation._internal.planning.proxy import (
        build_target_proposal_problem,
        default_planning_workspace,
    )
    from spatialcf.generation.capture.models import (
        CompetitionNativePlacementAvailabilityV2_9,
    )

    if (
        type(prepared_source)
        is not CompetitionNativePreparedTargetReachabilitySourceV2_9_4
        or prepared_source._capability is not _PREPARED_SOURCE_CAPABILITY_V2_9_4
    ):
        raise TypeError("target reachability prepared source must be exact")
    checked_capture = prepared_source.capture
    checked_surface = prepared_source.source_surface_evidence
    checked_camera = prepared_source.camera_evidence
    if type(relation_before) is not Relation:
        raise TypeError("target reachability relation must be exact")
    try:
        placement = prepared_source.placements_by_id[subject_id]
        subject_surface = prepared_source.subject_surfaces_by_id[subject_id]
    except KeyError as error:
        raise ValueError(
            "target reachability subject evidence is incomplete"
        ) from error
    if (
        placement.availability
        is not CompetitionNativePlacementAvailabilityV2_9.KNOWN_RECEPTACLE_SPAWN
        or not placement.native_positions
        or subject_surface.placement_sha256 != placement.placement_sha256
    ):
        raise ValueError("target reachability requires receptacle native positions")

    scene = checked_capture.scene
    subject = scene.object_by_id(subject_id)
    scene.object_by_id(reference_id)
    intervention = InterventionSpec(
        subject_id=subject_id,
        reference_id=reference_id,
        relation_before=relation_before,
        relation_after=relation_before.opposite,
        camera_id="main",
    )
    native_points = tuple(
        RationalPoint2V2(
            x=Fraction.from_float(position.x - subject.position.x),
            y=Fraction.from_float(position.z - subject.position.y),
        )
        for position in placement.native_positions
    )
    reduced = reduce_competition_native_target_proposal_scene_v2_9_4(
        scene,
        intervention,
    )
    workspace = default_planning_workspace(
        scene,
        subject_id,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        problem = build_target_proposal_problem(
            reduced,
            intervention,
            workspace,
            case_id=f"target-reachability:{candidate_id}",
        )
        reachable = (
            competition_native_target_only_relation_after_native_coordinates_v2_9_4(
                problem,
                native_points,
            )
        )
    return build_competition_native_candidate_target_reachability_v2_9_4(
        candidate_id=candidate_id,
        source_id=checked_capture.source.source_id,
        scene_id=scene.scene_id,
        source_capture_sha256=checked_capture.source_capture_sha256,
        subject_id=subject_id,
        reference_id=reference_id,
        relation_before=relation_before,
        placement_sha256=placement.placement_sha256,
        surface_evidence_sha256=checked_surface.surface_evidence_sha256,
        subject_surface_evidence_sha256=(
            subject_surface.subject_surface_evidence_sha256
        ),
        camera_evidence_sha256=checked_camera.camera_evidence_sha256,
        native_position_count=len(native_points),
        reachable_native_coordinates=reachable,
    )


def derive_competition_native_candidate_target_reachability_v2_9_4(
    *,
    candidate_id: str,
    capture: object,
    source_surface_evidence: object,
    camera_evidence: object,
    subject_id: str,
    reference_id: str,
    relation_before: Relation,
) -> CompetitionNativeCandidateTargetReachabilityV2_9_4:
    """Freshly derive one row from independently strict-validated source facts."""

    prepared = prepare_competition_native_target_reachability_source_v2_9_4(
        capture,
        source_surface_evidence,
        camera_evidence,
    )
    return derive_competition_native_candidate_target_reachability_from_prepared_v2_9_4(
        candidate_id=candidate_id,
        prepared_source=prepared,
        subject_id=subject_id,
        reference_id=reference_id,
        relation_before=relation_before,
    )


CandidateTargetReachability = CompetitionNativeCandidateTargetReachabilityV2_9_4
PreparedTargetReachabilitySource = (
    CompetitionNativePreparedTargetReachabilitySourceV2_9_4
)
TargetReachabilityStatus = CompetitionNativeTargetReachabilityStatusV2_9_4
target_only_relation_after_native_coordinates = (
    competition_native_target_only_relation_after_native_coordinates_v2_9_4
)
reduce_target_proposal_scene = reduce_competition_native_target_proposal_scene_v2_9_4

__all__ = (
    "CandidateTargetReachability",
    "PreparedTargetReachabilitySource",
    "TargetReachabilityStatus",
    "reduce_target_proposal_scene",
    "target_only_relation_after_native_coordinates",
)
