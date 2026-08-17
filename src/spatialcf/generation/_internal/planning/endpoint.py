"""Deterministic current source-only endpoint planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from spatialcf.adapters.ai2thor import (
    AI2ThorNativePosition,
    AI2ThorReceptacleSpawnMap,
    AI2ThorReceptacleSurfacePatch,
    AI2ThorRuntimeIdentity,
    _native_positions_sha256,
    _receptacle_scene_sha256,
    _receptacle_spawn_source_sha256,
    build_receptacle_support_position_region,
)
from spatialcf.core.v2.continuous_yaw_solver_v2_9 import (
    solve_continuous_yaw_minimum_cost_v2_9,
)
from spatialcf.core.v2.convex_translation_domain import RationalPoint2V2
from spatialcf.domain.models import (
    InterventionSpec,
    Scene,
    SceneObject,
    SubjectPositionRegion,
)
from spatialcf.domain.v2.continuous_yaw_camera import SemanticProblemV2_3
from spatialcf.domain.v2.continuous_yaw_solver_v2_9 import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
)
from spatialcf.generation._internal.evidence.reachability import (
    target_only_relation_after_native_coordinates,
)
from spatialcf.generation._internal.evidence.surface import (
    ReceptacleSurfacePatch,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
)
from spatialcf.generation._internal.planning.models import (
    EndpointPlan,
    EndpointWorkspace,
    SubjectPlacementFact,
)
from spatialcf.generation._internal.planning.proxy import (
    _prepare_proxy,
    _PreparedCurrentProxy,
    _project_proxy,
    _project_proxy_problem,
)
from spatialcf.geometry.regions import subject_position_region_geometry
from spatialcf.solver.feasible import FeasibleRegionBuilder

_DEFAULT_RADII_M = (0.02, 0.01, 0.005, 0.001)
_NATIVE_SPAWN_RADIUS_M = 0.000001


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _solver_success_mismatch(
    result: ContinuousYawCertifiedSuccessResultV2_9,
    problem: SemanticProblemV2_3,
    config: ContinuousYawSolverConfigV2_9,
) -> str | None:
    """Reject a success certificate that was not solved for this exact call."""

    if result.semantic_problem_sha256 != problem.semantic_problem_sha256:
        return "endpoint_plan:solver_problem_mismatch"
    if (
        type(result.solver_config) is not ContinuousYawSolverConfigV2_9
        or result.solver_config != config
        or result.solver_config.config_sha256 != config.config_sha256
    ):
        return "endpoint_plan:solver_config_mismatch"
    return None


class EndpointPlanRejected(ValueError):
    """No bounded source-only endpoint proposal reached certified success."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        checked = tuple(sorted(set(reasons)))
        if not checked or any(type(item) is not str or not item for item in checked):
            raise ValueError("endpoint planner reasons must be non-empty strings")
        self.reasons = checked
        super().__init__(";".join(checked))


@dataclass(frozen=True, slots=True)
class _EndpointCandidate:
    """One deterministically sourced endpoint proposal.

    The source tag is operational: only a native receptacle position is an
    exact platform placement fact and therefore receives the micrometre
    workspace below.  Keeping the tag on each point avoids relying on a
    fragile list-prefix convention after the candidate strata are mixed.
    """

    point: RationalPoint2V2
    source: Literal["native", "legacy"]

    def __post_init__(self) -> None:
        if type(self.point) is not RationalPoint2V2:
            raise TypeError("endpoint candidate point must be exact")


@dataclass(frozen=True, slots=True)
class _PatchLocus:
    patch: ReceptacleSurfacePatch
    position_region: SubjectPositionRegion

    def __post_init__(self) -> None:
        if type(self.patch) is not ReceptacleSurfacePatch:
            raise TypeError("patch locus patch must be exact")
        ReceptacleSurfacePatch.model_validate(
            self.patch.model_dump(mode="python"),
            strict=True,
        )
        if type(self.position_region) is not SubjectPositionRegion:
            raise TypeError("patch locus position region must be exact")
        SubjectPositionRegion.model_validate(
            self.position_region.model_dump(mode="python"),
            strict=True,
        )


@dataclass(frozen=True, slots=True)
class _PatchOwnedEndpointCandidate:
    point: RationalPoint2V2
    source: Literal["native", "legacy"]
    patch_index: int
    patch_sha256: str

    def __post_init__(self) -> None:
        if type(self.point) is not RationalPoint2V2:
            raise TypeError("patch-owned candidate point must be exact")
        if self.source not in ("native", "legacy"):
            raise ValueError("patch-owned candidate source is unsupported")
        if type(self.patch_index) is not int or self.patch_index < 0:
            raise TypeError("patch-owned candidate index must be non-negative")
        _require_sha256(self.patch_sha256, "patch-owned candidate patch")


def _single_patch_loci_from_prepared(
    scene: Scene,
    placement: SubjectPlacementFact,
    prepared: _PreparedCurrentProxy,
) -> tuple[_PatchLocus, ...]:
    """Build per-patch loci without repeating capture/evidence verification."""

    if type(scene) is not Scene:
        raise TypeError("patch-bound endpoint scene must be exact")
    if type(placement) is not SubjectPlacementFact:
        raise TypeError("patch-bound endpoint placement must be exact")
    if type(prepared) is not _PreparedCurrentProxy:
        raise TypeError("prepared patch proxy context must be exact")
    checked_placement = SubjectPlacementFact.model_validate(
        placement.model_dump(mode="python"),
        strict=True,
    )
    capture = checked_placement.source_capture
    subject_placement = checked_placement.subject_placement
    checked_evidence = prepared.subject_surface_evidence
    if capture != prepared.source_capture:
        raise ValueError("patch-bound placement changed the prepared capture")
    verified_subjects = tuple(
        item
        for item in prepared.source_surface_evidence.subjects
        if item.subject_object_id == checked_evidence.subject_object_id
    )
    if len(verified_subjects) != 1 or verified_subjects[0] != checked_evidence:
        raise ValueError("patch-bound evidence does not bind verified capture facts")
    if scene != capture.scene:
        raise ValueError("patch-bound placement does not bind the frozen scene")
    if (
        checked_evidence.subject_object_id != subject_placement.object_id
        or checked_evidence.support_object_id != subject_placement.support_object_id
        or checked_evidence.placement_sha256 != subject_placement.placement_sha256
        or checked_evidence.source_capture_sha256 != capture.source_capture_sha256
        or subject_placement.position_region is None
    ):
        raise ValueError("patch-bound evidence does not bind placement lineage")
    patches = tuple(
        AI2ThorReceptacleSurfacePatch(
            x_min=item.x_min,
            x_max=item.x_max,
            native_y=item.native_y,
            z_min=item.z_min,
            z_max=item.z_max,
        )
        for item in checked_evidence.patches
    )
    if checked_evidence.scene_sha256 != _receptacle_scene_sha256(scene, patches):
        raise ValueError("patch-bound evidence does not bind source geometry")
    runtime = AI2ThorRuntimeIdentity(
        **capture.runtime_identity.model_dump(mode="python")
    )
    positions = tuple(
        AI2ThorNativePosition(x=item.x, y=item.y, z=item.z)
        for item in subject_placement.native_positions
    )

    def spawn_map(
        selected: tuple[AI2ThorReceptacleSurfacePatch, ...],
    ) -> AI2ThorReceptacleSpawnMap:
        selected_positions = (
            positions
            if selected == patches
            else tuple(
                AI2ThorNativePosition(
                    x=patch.x_min + (patch.x_max - patch.x_min) * x_index / 20.0,
                    y=patch.native_y,
                    z=patch.z_min + (patch.z_max - patch.z_min) * z_index / 20.0,
                )
                for patch in selected
                for x_index in range(21)
                for z_index in range(21)
            )
        )
        positions_sha256 = (
            checked_evidence.positions_sha256
            if selected == patches
            else _native_positions_sha256(selected_positions)
        )
        scene_sha256 = _receptacle_scene_sha256(scene, selected)
        source_sha256 = _receptacle_spawn_source_sha256(
            scene_id=scene.scene_id,
            subject_object_id=checked_evidence.subject_object_id,
            support_object_id=checked_evidence.support_object_id,
            native_subject_object_id=checked_evidence.native_subject_object_id,
            native_support_object_id=checked_evidence.native_support_object_id,
            runtime_identity=runtime,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            surface_patches=selected,
        )
        return AI2ThorReceptacleSpawnMap(
            scene_id=scene.scene_id,
            subject_object_id=checked_evidence.subject_object_id,
            support_object_id=checked_evidence.support_object_id,
            native_subject_object_id=checked_evidence.native_subject_object_id,
            native_support_object_id=checked_evidence.native_support_object_id,
            runtime_identity=runtime,
            positions=selected_positions,
            positions_sha256=positions_sha256,
            scene_sha256=scene_sha256,
            source_sha256=source_sha256,
            surface_patches=selected,
        )

    full_region = build_receptacle_support_position_region(
        scene,
        spawn_map(patches),
    )
    if full_region != subject_placement.position_region:
        raise ValueError("patch-bound evidence changed the captured placement region")
    return tuple(
        _PatchLocus(
            patch=patch,
            position_region=build_receptacle_support_position_region(
                scene,
                spawn_map((native_patch,)),
            ),
        )
        for patch, native_patch in zip(
            checked_evidence.patches,
            patches,
            strict=True,
        )
    )


def _screening_config(
    config: ContinuousYawSolverConfigV2_9,
    domain_operations: int,
) -> ContinuousYawSolverConfigV2_9:
    candidate = config.candidate_config
    if candidate.max_domain_operations <= domain_operations:
        return config
    payload = config.model_dump(mode="python", warnings="error")
    payload["candidate_config"] = {
        **payload["candidate_config"],
        "max_domain_operations": domain_operations,
        "max_so2_atomic_steps": min(candidate.max_so2_atomic_steps, domain_operations),
        "max_candidate_cells": min(candidate.max_candidate_cells, 5_000),
    }
    payload["max_objective_partition_cells"] = min(
        config.max_objective_partition_cells, 5_000
    )
    return ContinuousYawSolverConfigV2_9.model_validate(payload, strict=True)


def _legacy_source_candidate_points(
    scene: Scene,
    intervention: InterventionSpec,
) -> tuple[RationalPoint2V2, ...]:
    """Use the legacy geometry stack only as a cheap proposal generator.

    These floating-point proposals have no certificate authority.  Every
    returned endpoint is rebuilt and accepted only by the Canonical v2.9
    solver below.
    """

    region = FeasibleRegionBuilder().build(scene, intervention)
    if region.is_empty:
        return ()
    subject = scene.object_by_id(intervention.subject_id)
    points: dict[tuple[Fraction, Fraction], RationalPoint2V2] = {}

    def add(absolute_x: float, absolute_y: float) -> None:
        if not (math.isfinite(absolute_x) and math.isfinite(absolute_y)):
            return
        location = Point(absolute_x, absolute_y)
        if not region.covers(location):
            return
        delta_x = Fraction.from_float(absolute_x - subject.position.x)
        delta_y = Fraction.from_float(absolute_y - subject.position.y)
        point = RationalPoint2V2(x=delta_x, y=delta_y)
        points[(point.x, point.y)] = point

    for polygon in _polygon_parts(region):
        representative = polygon.representative_point()
        add(representative.x, representative.y)
        centroid = polygon.centroid
        add(centroid.x, centroid.y)
        min_x, min_y, max_x, max_y = polygon.bounds
        for x_index in range(7):
            x = min_x + (max_x - min_x) * (x_index + 1) / 8
            for y_index in range(7):
                y = min_y + (max_y - min_y) * (y_index + 1) / 8
                add(x, y)
    return tuple(
        value
        for _, value in sorted(
            points.items(),
            key=lambda item: (
                item[0][0] ** 2 + item[0][1] ** 2,
                item[0][0],
                item[0][1],
            ),
        )
    )


def _neutral_placement_candidate_points(
    scene: Scene,
    intervention: InterventionSpec,
    placement: SubjectPlacementFact | None,
    *,
    require_legacy_region: bool = True,
    require_native_support_region: bool = False,
) -> tuple[RationalPoint2V2, ...]:
    if placement is None:
        return ()
    if (
        type(require_legacy_region) is not bool
        or type(require_native_support_region) is not bool
    ):
        raise TypeError("endpoint placement region policies must be exact")
    capture = placement.source_capture
    subject_placement = placement.subject_placement
    if scene != capture.scene:
        raise ValueError("endpoint placement does not bind the frozen scene")
    subject = scene.object_by_id(intervention.subject_id)
    if (
        intervention.subject_id != subject_placement.object_id
        or subject.object_id != subject_placement.object_id
        or subject.support_object_id != subject_placement.support_object_id
    ):
        raise ValueError("endpoint placement does not bind the frozen source")
    region = (
        FeasibleRegionBuilder().build(scene, intervention)
        if require_legacy_region
        else None
    )
    native_support_region = (
        None
        if not require_native_support_region
        or subject_placement.position_region is None
        else subject_position_region_geometry(subject_placement.position_region)
    )
    if require_native_support_region and native_support_region is None:
        raise EndpointPlanRejected(("endpoint_plan:native_support_locus_missing",))
    points = {
        (point.x, point.y): point
        for item in subject_placement.native_positions
        if (region is None or region.covers(Point(item.x, item.z)))
        and (
            native_support_region is None
            or native_support_region.covers(Point(item.x, item.z))
        )
        for point in (
            RationalPoint2V2(
                x=Fraction.from_float(item.x - subject.position.x),
                y=Fraction.from_float(item.z - subject.position.y),
            ),
        )
    }
    return tuple(
        value
        for _, value in sorted(
            points.items(),
            key=lambda item: (
                item[0][0] ** 2 + item[0][1] ** 2,
                item[0][0],
                item[0][1],
            ),
        )
    )


def endpoint_workspace_within_position_region(
    subject: SceneObject,
    region: SubjectPositionRegion,
    workspace: EndpointWorkspace,
) -> bool:
    """Prove one OBB-center workspace lies in a subject-pivot locus."""

    if type(subject) is not SceneObject:
        raise TypeError("workspace subject must be exact")
    if type(region) is not SubjectPositionRegion:
        raise TypeError("workspace position region must be exact")
    if type(workspace) is not EndpointWorkspace:
        raise TypeError("workspace must be exact")
    if region.subject_object_id != subject.object_id:
        raise ValueError("workspace position region does not bind the subject")
    pivot_locus = subject_position_region_geometry(region)
    center_locus = translate(
        pivot_locus,
        xoff=subject.obb.center.x - subject.position.x,
        yoff=subject.obb.center.y - subject.position.y,
    )
    candidate_workspace = box(
        workspace.min_x_m,
        workspace.min_y_m,
        workspace.max_x_m,
        workspace.max_y_m,
    )
    return bool(center_locus.covers(candidate_workspace))


def _polygon_parts(region: BaseGeometry) -> tuple[Polygon, ...]:
    if isinstance(region, Polygon):
        return (region,)
    if isinstance(region, MultiPolygon):
        return tuple(item for item in region.geoms if not item.is_empty)
    if isinstance(region, GeometryCollection):
        return tuple(
            polygon for item in region.geoms for polygon in _polygon_parts(item)
        )
    return ()


def _tight_workspace(
    subject_x: float,
    subject_y: float,
    point: RationalPoint2V2,
    radius: float,
    planning: EndpointWorkspace,
) -> EndpointWorkspace | None:
    center_x = subject_x + float(point.x)
    center_y = subject_y + float(point.y)
    if not (math.isfinite(center_x) and math.isfinite(center_y)):
        return None
    min_x = max(planning.min_x_m, center_x - radius)
    min_y = max(planning.min_y_m, center_y - radius)
    max_x = min(planning.max_x_m, center_x + radius)
    max_y = min(planning.max_y_m, center_y + radius)
    if min_x >= max_x or min_y >= max_y:
        return None
    return EndpointWorkspace(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
    )


def _current_candidate_roster(
    *,
    native_points: tuple[RationalPoint2V2, ...],
    legacy_points: tuple[RationalPoint2V2, ...],
    max_candidate_points: int,
    target_only_relation_after_coordinates: frozenset[tuple[Fraction, Fraction]],
) -> tuple[_EndpointCandidate, ...]:
    """Freeze the current target-ranked native/legacy source roster."""

    return _bounded_current_candidate_roster(
        native_points=native_points,
        legacy_points=legacy_points,
        max_candidate_points=max_candidate_points,
        native_relation_after_coordinates=target_only_relation_after_coordinates,
    )


def _assign_patch_owned_candidates(
    *,
    subject: SceneObject,
    native_points: tuple[RationalPoint2V2, ...],
    legacy_points: tuple[RationalPoint2V2, ...],
    patch_loci: tuple[_PatchLocus, ...],
    max_candidate_points: int,
    target_only_relation_after_coordinates: frozenset[tuple[Fraction, Fraction]],
) -> tuple[_PatchOwnedEndpointCandidate, ...]:
    """Freeze source-only membership, ordering, and one-patch ownership."""

    if type(subject) is not SceneObject:
        raise TypeError("patch-owned candidate subject must be exact")
    if type(patch_loci) is not tuple or any(
        type(item) is not _PatchLocus for item in patch_loci
    ):
        raise TypeError("patch loci must be an exact tuple")
    canonical_loci = tuple(sorted(patch_loci, key=lambda item: item.patch.patch_index))
    indexes = tuple(item.patch.patch_index for item in canonical_loci)
    if indexes != tuple(range(len(canonical_loci))):
        raise ValueError("patch loci indexes are not canonical and complete")
    geometries = tuple(
        (item, subject_position_region_geometry(item.position_region))
        for item in canonical_loci
    )
    owner_by_coordinate: dict[tuple[Fraction, Fraction], _PatchLocus] = {}

    def freeze_points(
        points: tuple[RationalPoint2V2, ...],
    ) -> tuple[RationalPoint2V2, ...]:
        kept = []
        for point in points:
            absolute = Point(
                subject.position.x + float(point.x),
                subject.position.y + float(point.y),
            )
            owners = tuple(
                item for item, geometry in geometries if geometry.covers(absolute)
            )
            if not owners:
                continue
            owner_by_coordinate[(point.x, point.y)] = owners[0]
            kept.append(point)
        return tuple(kept)

    owned_native = freeze_points(native_points)
    owned_legacy = freeze_points(legacy_points)
    owned_native_coordinates = {(item.x, item.y) for item in owned_native}
    base = _current_candidate_roster(
        native_points=owned_native,
        legacy_points=owned_legacy,
        max_candidate_points=max_candidate_points,
        target_only_relation_after_coordinates=(
            target_only_relation_after_coordinates & owned_native_coordinates
        ),
    )
    return tuple(
        _PatchOwnedEndpointCandidate(
            point=candidate.point,
            source=candidate.source,
            patch_index=owner_by_coordinate[
                (candidate.point.x, candidate.point.y)
            ].patch.patch_index,
            patch_sha256=owner_by_coordinate[
                (candidate.point.x, candidate.point.y)
            ].patch.patch_sha256,
        )
        for candidate in base
    )


def _bounded_current_candidate_roster(
    *,
    native_points: tuple[RationalPoint2V2, ...],
    legacy_points: tuple[RationalPoint2V2, ...],
    max_candidate_points: int,
    native_relation_after_coordinates: frozenset[tuple[Fraction, Fraction]],
) -> tuple[_EndpointCandidate, ...]:
    """Round-robin target-ranked native facts and legacy proposals."""

    sources = {"native": native_points, "legacy": legacy_points}
    if type(max_candidate_points) is not int or max_candidate_points < 1:
        raise TypeError("max_candidate_points must be a positive exact integer")
    if type(native_relation_after_coordinates) is not frozenset or any(
        type(coordinate) is not tuple
        or len(coordinate) != 2
        or any(type(component) is not Fraction for component in coordinate)
        for coordinate in native_relation_after_coordinates
    ):
        raise TypeError("native relation-after coordinates must be exact")
    for label, values in sources.items():
        if type(values) is not tuple or any(
            type(item) is not RationalPoint2V2 for item in values
        ):
            raise TypeError(f"{label} candidate points must be an exact tuple")

    owner_by_coordinate: dict[
        tuple[Fraction, Fraction], Literal["native", "legacy"]
    ] = {}
    for source in ("legacy", "native"):
        for point in sources[source]:
            owner_by_coordinate[(point.x, point.y)] = source
    native_coordinates = {(point.x, point.y) for point in native_points}
    if not native_relation_after_coordinates <= native_coordinates:
        raise ValueError("relation-after priority escaped native membership")

    def coordinate_key(
        source: str,
        coordinate: tuple[Fraction, Fraction],
    ) -> tuple[Fraction, ...] | tuple[int, Fraction, Fraction, Fraction, Fraction]:
        stable_key = (
            Fraction(),
            coordinate[0] ** 2 + coordinate[1] ** 2,
            coordinate[0],
            coordinate[1],
        )
        if source == "native" and native_relation_after_coordinates:
            return (
                0 if coordinate in native_relation_after_coordinates else 1,
                *stable_key,
            )
        return stable_key

    ordered: dict[str, tuple[_EndpointCandidate, ...]] = {}
    for source in ("native", "legacy"):
        coordinates = sorted(
            {
                (point.x, point.y)
                for point in sources[source]
                if owner_by_coordinate[(point.x, point.y)] == source
            },
            key=lambda item: coordinate_key(source, item),
        )
        ordered[source] = tuple(
            _EndpointCandidate(
                point=RationalPoint2V2(x=coordinate[0], y=coordinate[1]),
                source=source,
            )
            for coordinate in coordinates
        )

    result: list[_EndpointCandidate] = []
    index = 0
    while len(result) < max_candidate_points:
        added = False
        for source in ("native", "legacy"):
            candidates = ordered[source]
            if index < len(candidates):
                result.append(candidates[index])
                added = True
                if len(result) == max_candidate_points:
                    break
        if not added:
            break
        index += 1
    return tuple(result)


def plan_endpoint(
    scene: Scene,
    intervention: InterventionSpec,
    workspace: EndpointWorkspace,
    config: ContinuousYawSolverConfigV2_9,
    source_surface_evidence: SourceSurfaceEvidence,
    subject_surface_evidence: SubjectSurfaceEvidence,
    *,
    placement: SubjectPlacementFact,
    case_id: str,
    max_candidate_points: int = 64,
    max_included_collision_obstacles: int = 6,
    screening_domain_operations: int = 100_000,
) -> EndpointPlan:
    """Return the first current solver-certified source-only endpoint."""

    if type(scene) is not Scene or type(intervention) is not InterventionSpec:
        raise TypeError("scene and intervention must be exact legacy values")
    if type(workspace) is not EndpointWorkspace:
        raise TypeError("workspace must be exact")
    if type(config) is not ContinuousYawSolverConfigV2_9:
        raise TypeError("config must be exact")
    if type(placement) is not SubjectPlacementFact:
        raise TypeError("patch-bound endpoint placement must be exact")
    if type(case_id) is not str or not case_id:
        raise TypeError("case_id must be a non-empty exact string")
    if type(max_candidate_points) is not int or max_candidate_points < 1:
        raise TypeError("max_candidate_points must be a positive exact integer")
    if (
        type(max_included_collision_obstacles) is not int
        or max_included_collision_obstacles < 1
    ):
        raise TypeError(
            "max_included_collision_obstacles must be a positive exact integer"
        )
    if type(screening_domain_operations) is not int or screening_domain_operations < 1:
        raise TypeError("screening_domain_operations must be a positive exact integer")

    prepared = _prepare_proxy(
        scene,
        intervention,
        placement.source_capture,
        source_surface_evidence,
        subject_surface_evidence,
        case_id=case_id,
    )
    patch_loci = _single_patch_loci_from_prepared(
        scene,
        placement,
        prepared,
    )
    subject = scene.object_by_id(intervention.subject_id)
    if subject.object_id != subject_surface_evidence.subject_object_id:
        raise ValueError("patch-bound evidence does not bind intervention subject")
    native_points = _neutral_placement_candidate_points(
        scene,
        intervention,
        placement,
        require_legacy_region=False,
        require_native_support_region=False,
    )
    legacy_points = _legacy_source_candidate_points(scene, intervention)
    broad_problem = _project_proxy_problem(prepared.base, workspace)
    target_coordinates = target_only_relation_after_native_coordinates(
        broad_problem,
        native_points,
    )
    if not target_coordinates and not legacy_points:
        raise EndpointPlanRejected(
            ("endpoint_plan:no_relation_after_source_candidate",)
        )
    candidates = _assign_patch_owned_candidates(
        subject=subject,
        native_points=native_points,
        legacy_points=legacy_points,
        patch_loci=patch_loci,
        max_candidate_points=max_candidate_points,
        target_only_relation_after_coordinates=target_coordinates,
    )
    if not candidates:
        raise EndpointPlanRejected(("endpoint_plan:no_single_patch_source_candidate",))
    screening_config = _screening_config(config, screening_domain_operations)
    locus_by_index = {item.patch.patch_index: item for item in patch_loci}
    reasons: set[str] = set()
    attempts = 0
    for candidate_index, candidate in enumerate(candidates):
        radius = (
            _NATIVE_SPAWN_RADIUS_M if candidate.source == "native" else _DEFAULT_RADII_M
        )
        radii = (radius,) if type(radius) is float else radius
        locus = locus_by_index[candidate.patch_index]
        for candidate_radius in radii:
            endpoint_workspace = _tight_workspace(
                subject.obb.center.x,
                subject.obb.center.y,
                candidate.point,
                candidate_radius,
                workspace,
            )
            if endpoint_workspace is None or not (
                endpoint_workspace_within_position_region(
                    subject,
                    locus.position_region,
                    endpoint_workspace,
                )
            ):
                reasons.add("endpoint_plan:outside_single_patch_support_locus")
                continue
            attempts += 1
            proxy = _project_proxy(
                prepared,
                endpoint_workspace,
                patch_index=candidate.patch_index,
            )
            if (
                len(proxy.binding.included_collision_native_object_ids)
                > max_included_collision_obstacles
            ):
                reasons.add("endpoint_plan:collision_roster_too_large")
                continue
            screened = solve_continuous_yaw_minimum_cost_v2_9(
                proxy.semantic_problem,
                screening_config,
            )
            if type(screened.result) is not ContinuousYawCertifiedSuccessResultV2_9:
                reasons.update(screened.finding_codes)
                reasons.update(getattr(screened.result, "finding_codes", ()))
                continue
            screened_mismatch = _solver_success_mismatch(
                screened.result,
                proxy.semantic_problem,
                screening_config,
            )
            if screened_mismatch is not None:
                reasons.add(screened_mismatch)
                continue
            solved = (
                screened
                if screening_config == config
                else solve_continuous_yaw_minimum_cost_v2_9(
                    proxy.semantic_problem,
                    config,
                )
            )
            if type(solved.result) is not ContinuousYawCertifiedSuccessResultV2_9:
                reasons.update(solved.finding_codes)
                reasons.update(getattr(solved.result, "finding_codes", ()))
                continue
            solved_mismatch = _solver_success_mismatch(
                solved.result,
                proxy.semantic_problem,
                config,
            )
            if solved_mismatch is not None:
                reasons.add(solved_mismatch)
                continue
            return EndpointPlan(
                planning_workspace=workspace,
                endpoint_workspace=endpoint_workspace,
                candidate_index=candidate_index,
                attempted_workspace_count=attempts,
                candidate_point_count=len(candidates),
                patch_index=candidate.patch_index,
                patch_sha256=candidate.patch_sha256,
                source_capture_sha256=(subject_surface_evidence.source_capture_sha256),
                placement_sha256=subject_surface_evidence.placement_sha256,
                surface_evidence_sha256=(
                    source_surface_evidence.surface_evidence_sha256
                ),
                subject_surface_evidence_sha256=(
                    subject_surface_evidence.subject_surface_evidence_sha256
                ),
                proxy_bundle_sha256=proxy.proxy_bundle_sha256,
                solve_result_sha256=solved.result.solve_result_sha256,
                runtime_collision_delegated_native_object_ids=(
                    proxy.binding.runtime_collision_delegated_native_object_ids
                ),
            )
    raise EndpointPlanRejected(
        tuple(reasons) or ("endpoint_plan:no_certified_single_patch_workspace",)
    )


__all__ = ("EndpointPlanRejected", "plan_endpoint")
