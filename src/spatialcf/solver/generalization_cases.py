"""Closed synthetic corpus and independent oracle for solver generalization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.domain.enums import Relation, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene, SceneObject, Vec2
from spatialcf.geometry.obb import obb_footprint
from spatialcf.relations.engine import RelationEngine

GENERALIZATION_CASE_IDS = (
    "lr-open-boundary-sat",
    "lr-small-subject-sat",
    "lr-wide-subject-sat",
    "lr-rotated-subject-sat",
    "lr-upper-obstacle-sat",
    "lr-lower-obstacle-sat",
    "lr-two-corridors-tie-sat",
    "lr-room-edge-sat",
    "lr-obstacle-wall-unsat",
    "lr-narrow-room-unsat",
    "fb-open-depth-sat",
    "fb-offset-camera-sat",
    "fb-rotated-subject-sat",
    "fb-wide-support-sat",
    "fb-tight-support-edge-sat",
    "fb-obstacle-side-step-sat",
    "fb-two-obstacle-tie-sat",
    "fb-room-edge-sat",
    "fb-short-support-unsat",
    "fb-shallow-room-unsat",
    "nf-open-room-x-sat",
    "nf-open-room-diagonal-sat",
    "nf-rotated-reference-sat",
    "nf-support-corner-sat",
    "nf-obstacle-detour-sat",
    "nf-small-room-unsat",
    "nf-short-support-unsat",
    "nf-covered-far-zone-unsat",
    "nf-boundary-unsat",
    "nf-large-reference-unsat",
)

_SOURCE = "core-solver-generalization-v1"
_SEED = 20260802
_EXPECTED_DIRECTIONS = (
    *((Relation.LEFT, Relation.RIGHT),) * 10,
    *((Relation.FRONT, Relation.BEHIND),) * 10,
    *((Relation.NEAR, Relation.FAR),) * 10,
)
_EPS = 1e-12
_GEOMETRY_EPS = 1e-8


class GeneralizationCaseError(ValueError):
    """The committed corpus cannot support an independent validation claim."""


class SatOracleSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_outcome: Literal["SAT"]
    proof_kind: Literal[
        "target_boundary",
        "obstacle_corner",
        "support_boundary",
        "far_boundary",
        "target_preservation_intersection",
    ]
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    exact_infimum_points: tuple[Vec2, ...] = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @model_validator(mode="after")
    def finite_points(self) -> SatOracleSpec:
        if not all(
            math.isfinite(value)
            for point in self.exact_infimum_points
            for value in (point.x, point.y)
        ):
            raise ValueError("oracle witness points must be finite")
        return self


class UnsatOracleSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_outcome: Literal["UNSAT"]
    proof_kind: Literal[
        "right_boundary_exceeds_locus",
        "behind_boundary_exceeds_locus",
        "maximum_ground_gap",
        "target_locus_covered",
    ]
    maximum_possible_value_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_value_m: float = Field(allow_inf_nan=False, strict=True)
    expected_reason: Literal["empty_outer_region"]
    derivation: str = Field(min_length=1)


OracleSpec = Annotated[
    SatOracleSpec | UnsatOracleSpec,
    Field(discriminator="expected_outcome"),
]


class GeneralizationCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+$")
    scene: Scene
    intervention: InterventionSpec
    oracle: OracleSpec


class GeneralizationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    cases: tuple[GeneralizationCaseSpec, ...]


@dataclass(frozen=True)
class OracleResult:
    expected_outcome: Literal["SAT", "UNSAT"]
    exact_infimum_m: float | None
    exact_infimum_points: tuple[Vec2, ...]
    maximum_possible_value_m: float | None
    required_value_m: float | None


@dataclass(frozen=True)
class LoadedGeneralizationCase:
    spec: GeneralizationCaseSpec
    scene_sha256: str
    oracle: OracleResult


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise GeneralizationCaseError(f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise GeneralizationCaseError(f"cannot read {label}: {exc}") from exc


def _parse_json(payload: bytes) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeneralizationCaseError(
            f"invalid UTF-8 JSON in cases.json: {exc}"
        ) from exc


def _numeric_values(value: object):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numeric_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _numeric_values(item)


def _relative_subject_vertices(subject: SceneObject) -> tuple[tuple[float, float], ...]:
    footprint = obb_footprint(subject.obb)
    return tuple(
        (float(x) - subject.position.x, float(y) - subject.position.y)
        for x, y in tuple(footprint.exterior.coords)[:-1]
    )


def _rectangular_locus(
    container: BaseGeometry,
    relative_vertices: tuple[tuple[float, float], ...],
    label: str,
) -> BaseGeometry:
    if container.is_empty or not container.equals(box(*container.bounds)):
        raise GeneralizationCaseError(
            f"oracle {label} must be an axis-aligned rectangle"
        )
    min_x, min_y, max_x, max_y = container.bounds
    rel_x = tuple(value[0] for value in relative_vertices)
    rel_y = tuple(value[1] for value in relative_vertices)
    lower_x = min_x - min(rel_x)
    lower_y = min_y - min(rel_y)
    upper_x = max_x - max(rel_x)
    upper_y = max_y - max(rel_y)
    if lower_x > upper_x or lower_y > upper_y:
        return Polygon()
    return box(lower_x, lower_y, upper_x, upper_y)


def _configuration_obstacle(
    obstacle: BaseGeometry,
    relative_vertices: tuple[tuple[float, float], ...],
) -> BaseGeometry:
    obstacle_vertices = tuple(obstacle.exterior.coords)[:-1]
    return MultiPoint(
        [
            (float(ox) - sx, float(oy) - sy)
            for ox, oy in obstacle_vertices
            for sx, sy in relative_vertices
        ]
    ).convex_hull


def _physical_locus(case: GeneralizationCaseSpec) -> BaseGeometry:
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    relative_vertices = _relative_subject_vertices(subject)
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    locus = _rectangular_locus(room, relative_vertices, "room")
    support_id = subject.support_object_id
    if support_id is not None:
        support = scene.object_by_id(support_id)
        support_locus = _rectangular_locus(
            obb_footprint(support.obb),
            relative_vertices,
            "support",
        )
        locus = locus.intersection(support_locus)
    for obstacle in sorted(scene.objects, key=lambda item: item.object_id):
        if obstacle.object_id in {subject.object_id, support_id}:
            continue
        configuration = _configuration_obstacle(
            obb_footprint(obstacle.obb),
            relative_vertices,
        )
        locus = locus.difference(configuration).union(
            locus.intersection(configuration.boundary)
        )
    return locus


def _preserved_distance_locus(
    case: GeneralizationCaseSpec,
    locus: BaseGeometry,
) -> BaseGeometry:
    """Preserve the target pair's original non-target distance-axis label."""
    if case.intervention.relation_after in {Relation.NEAR, Relation.FAR}:
        return locus
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    reference = scene.object_by_id(case.intervention.reference_id)
    labels = RelationEngine().pair_labels(
        scene,
        subject.object_id,
        reference.object_id,
        case.intervention.camera_id,
    )
    configuration = _configuration_obstacle(
        obb_footprint(reference.obb),
        _relative_subject_vertices(subject),
    )
    if Relation.FAR in labels:
        excluded = configuration.buffer(
            RelationEngine.FAR_METERS,
            quad_segs=16384,
        )
        return locus.difference(excluded).union(locus.intersection(excluded.boundary))
    if Relation.NEAR in labels:
        return locus.intersection(
            configuration.buffer(RelationEngine.NEAR_METERS, quad_segs=16384)
        )
    near_zone = configuration.buffer(
        RelationEngine.NEAR_METERS,
        quad_segs=16384,
    )
    far_zone = configuration.buffer(
        RelationEngine.FAR_METERS,
        quad_segs=16384,
    )
    return locus.intersection(far_zone).difference(near_zone)


def _right_boundary(case: GeneralizationCaseSpec) -> float:
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    reference = scene.object_by_id(spec.reference_id)
    camera = scene.camera_by_id(spec.camera_id)
    subject_view = subject.views[spec.camera_id]
    reference_view = reference.views[spec.camera_id]
    fx = camera.intrinsics[0]
    if fx <= 0.0 or subject_view.camera_depth <= 0.0:
        raise GeneralizationCaseError(
            f"{case.case_id}: oracle camera calibration invalid"
        )
    target_u = (
        reference_view.bbox.center_x + camera.width * RelationEngine.LEFT_RIGHT_FRACTION
    )
    return subject.position.x + (
        (target_u - subject_view.bbox.center_x) * subject_view.camera_depth / fx
    )


def _behind_boundary(case: GeneralizationCaseSpec) -> float:
    scene = case.scene
    spec = case.intervention
    subject = scene.object_by_id(spec.subject_id)
    reference = scene.object_by_id(spec.reference_id)
    subject_view = subject.views[spec.camera_id]
    reference_view = reference.views[spec.camera_id]
    world_delta = reference.position.y - subject.position.y
    depth_delta = reference_view.camera_depth - subject_view.camera_depth
    if abs(world_delta) <= _EPS or abs(depth_delta) <= _EPS:
        raise GeneralizationCaseError(
            f"{case.case_id}: oracle depth calibration invalid"
        )
    metres_per_depth = world_delta / depth_delta
    target_depth = reference_view.camera_depth + RelationEngine.FRONT_BEHIND_METERS
    return (
        subject.position.y
        + (target_depth - subject_view.camera_depth) * metres_per_depth
    )


def _target_locus(
    case: GeneralizationCaseSpec,
    physical_locus: BaseGeometry,
) -> tuple[BaseGeometry, float]:
    relation = case.intervention.relation_after
    if physical_locus.is_empty:
        return physical_locus, 0.0
    min_x, min_y, max_x, max_y = physical_locus.bounds
    span = max(max_x - min_x, max_y - min_y, 1.0)
    if relation is Relation.RIGHT:
        boundary = _right_boundary(case)
        if boundary > max_x:
            return Polygon(), boundary
        target = box(boundary, min_y - span, max_x + span, max_y + span)
        return _preserved_distance_locus(
            case,
            physical_locus.intersection(target),
        ), boundary
    if relation is Relation.BEHIND:
        boundary = _behind_boundary(case)
        if boundary > max_y:
            return Polygon(), boundary
        target = box(min_x - span, boundary, max_x + span, max_y + span)
        return _preserved_distance_locus(
            case,
            physical_locus.intersection(target),
        ), boundary
    if relation is not Relation.FAR:
        raise GeneralizationCaseError(f"{case.case_id}: unsupported oracle relation")
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    reference = scene.object_by_id(case.intervention.reference_id)
    configuration = _configuration_obstacle(
        obb_footprint(reference.obb),
        _relative_subject_vertices(subject),
    )
    excluded = configuration.buffer(
        RelationEngine.FAR_METERS,
        quad_segs=16384,
    )
    target = physical_locus.difference(excluded).union(
        physical_locus.intersection(excluded.boundary)
    )
    return target, RelationEngine.FAR_METERS


def _geometry_vertices(geometry: BaseGeometry) -> tuple[Point, ...]:
    if geometry.is_empty:
        return ()
    geometries = getattr(geometry, "geoms", (geometry,))
    points: list[Point] = []
    for item in geometries:
        if hasattr(item, "exterior"):
            points.extend(Point(float(x), float(y)) for x, y in item.exterior.coords)
        elif hasattr(item, "coords"):
            points.extend(Point(float(x), float(y)) for x, y in item.coords)
    return tuple(points)


def _maximum_ground_gap(
    case: GeneralizationCaseSpec,
    physical_locus: BaseGeometry,
) -> float:
    scene = case.scene
    subject = scene.object_by_id(case.intervention.subject_id)
    reference = scene.object_by_id(case.intervention.reference_id)
    configuration = _configuration_obstacle(
        obb_footprint(reference.obb),
        _relative_subject_vertices(subject),
    )
    vertices = _geometry_vertices(physical_locus)
    if not vertices:
        return 0.0
    return max(point.distance(configuration) for point in vertices)


def recompute_oracle(case: GeneralizationCaseSpec) -> OracleResult:
    """Recompute the fixture oracle without production solver construction."""
    physical = _physical_locus(case)
    target, required = _target_locus(case, physical)
    oracle = case.oracle
    origin = Point(
        case.scene.object_by_id(case.intervention.subject_id).position.x,
        case.scene.object_by_id(case.intervention.subject_id).position.y,
    )
    if isinstance(oracle, SatOracleSpec):
        if target.is_empty:
            raise GeneralizationCaseError(f"{case.case_id}: SAT oracle target is empty")
        exact = float(origin.distance(target))
        if not math.isclose(
            oracle.exact_infimum_m,
            exact,
            rel_tol=0.0,
            abs_tol=_EPS,
        ):
            raise GeneralizationCaseError(f"{case.case_id}: oracle infimum mismatch")
        for witness in oracle.exact_infimum_points:
            point = Point(witness.x, witness.y)
            if not target.buffer(_GEOMETRY_EPS).covers(point) or not math.isclose(
                origin.distance(point),
                exact,
                rel_tol=0.0,
                abs_tol=_GEOMETRY_EPS,
            ):
                raise GeneralizationCaseError(
                    f"{case.case_id}: oracle witness mismatch"
                )
        if oracle.proof_kind == "target_boundary" and any(
            obj.object_id not in {"subject", "reference"} for obj in case.scene.objects
        ):
            raise GeneralizationCaseError(f"{case.case_id}: oracle proof kind mismatch")
        if oracle.proof_kind == "obstacle_corner" and len(case.scene.objects) <= 2:
            raise GeneralizationCaseError(f"{case.case_id}: oracle proof kind mismatch")
        if oracle.proof_kind == "support_boundary" and (
            case.scene.object_by_id(case.intervention.subject_id).support_object_id
            is None
        ):
            raise GeneralizationCaseError(f"{case.case_id}: oracle proof kind mismatch")
        if oracle.proof_kind == "far_boundary" and (
            case.intervention.relation_after is not Relation.FAR
        ):
            raise GeneralizationCaseError(f"{case.case_id}: oracle proof kind mismatch")
        if oracle.proof_kind == "target_preservation_intersection" and (
            case.intervention.relation_after in {Relation.NEAR, Relation.FAR}
        ):
            raise GeneralizationCaseError(f"{case.case_id}: oracle proof kind mismatch")
        return OracleResult("SAT", exact, oracle.exact_infimum_points, None, None)

    if not target.is_empty:
        raise GeneralizationCaseError(
            f"{case.case_id}: UNSAT oracle target is not empty"
        )
    if oracle.proof_kind == "right_boundary_exceeds_locus":
        maximum = float(physical.bounds[2]) if not physical.is_empty else 0.0
    elif oracle.proof_kind == "behind_boundary_exceeds_locus":
        maximum = float(physical.bounds[3]) if not physical.is_empty else 0.0
    else:
        maximum = _maximum_ground_gap(case, physical)
    if not math.isclose(
        oracle.maximum_possible_value_m,
        maximum,
        rel_tol=0.0,
        abs_tol=_EPS,
    ):
        raise GeneralizationCaseError(f"{case.case_id}: oracle maximum mismatch")
    if not math.isclose(
        oracle.required_value_m,
        required,
        rel_tol=0.0,
        abs_tol=_EPS,
    ):
        raise GeneralizationCaseError(f"{case.case_id}: oracle requirement mismatch")
    if not maximum < required:
        raise GeneralizationCaseError(f"{case.case_id}: oracle bound is not strict")
    return OracleResult("UNSAT", None, (), maximum, required)


def _validate_common(case: GeneralizationCaseSpec) -> None:
    scene = case.scene
    if scene.scene_id != case.case_id:
        raise GeneralizationCaseError(f"{case.case_id}: scene_id mismatch")
    if scene.source != _SOURCE:
        raise GeneralizationCaseError(f"{case.case_id}: source mismatch")
    if scene.generation_seed != _SEED:
        raise GeneralizationCaseError(f"{case.case_id}: generation seed mismatch")
    if scene.coordinate_system != "RH_METERS_Z_UP":
        raise GeneralizationCaseError(f"{case.case_id}: coordinate system mismatch")
    if scene.pinned_object_ids:
        raise GeneralizationCaseError(f"{case.case_id}: pinned objects mismatch")
    if tuple(camera.camera_id for camera in scene.cameras) != ("camera",):
        raise GeneralizationCaseError(f"{case.case_id}: camera set mismatch")
    if not all(math.isfinite(value) for value in _numeric_values(scene.model_dump())):
        raise GeneralizationCaseError(f"{case.case_id}: scene geometry must be finite")
    try:
        subject = scene.object_by_id(case.intervention.subject_id)
        reference = scene.object_by_id(case.intervention.reference_id)
        scene.camera_by_id(case.intervention.camera_id)
    except KeyError as exc:
        raise GeneralizationCaseError(
            f"{case.case_id}: intervention endpoint missing"
        ) from exc
    if subject.object_id != "subject" or reference.object_id != "reference":
        raise GeneralizationCaseError(f"{case.case_id}: canonical endpoints mismatch")
    if not subject.movable or any(
        obj.movable for obj in scene.objects if obj.object_id != subject.object_id
    ):
        raise GeneralizationCaseError(
            f"{case.case_id}: only the subject may be movable"
        )
    source = RelationEngine().observe(
        scene,
        case.intervention.subject_id,
        case.intervention.reference_id,
        case.intervention.relation_before,
        case.intervention.camera_id,
    )
    if source.status is not SolverStatus.SUCCESS or not source.satisfied:
        raise GeneralizationCaseError(f"{case.case_id}: source relation mismatch")


def load_generalization_cases(root: Path) -> tuple[LoadedGeneralizationCase, ...]:
    """Load and independently validate the exact committed 30-case corpus."""
    if root.is_symlink() or not root.is_dir():
        raise GeneralizationCaseError("generalization root must be a real directory")
    actual_files = {entry.name for entry in root.iterdir()}
    if actual_files != {"cases.json"}:
        raise GeneralizationCaseError("generalization fixture file set mismatch")
    payload = _read_regular_file(root / "cases.json", "cases.json")
    try:
        manifest = GeneralizationManifest.model_validate(_parse_json(payload))
    except ValidationError as exc:
        raise GeneralizationCaseError(
            f"invalid generalization manifest: {exc}"
        ) from exc
    case_ids = tuple(case.case_id for case in manifest.cases)
    if len(case_ids) != len(set(case_ids)):
        raise GeneralizationCaseError("generalization case ids must be unique")
    if set(case_ids) != set(GENERALIZATION_CASE_IDS):
        raise GeneralizationCaseError("manifest must contain canonical case ids")
    if case_ids != GENERALIZATION_CASE_IDS:
        raise GeneralizationCaseError("generalization cases must use canonical order")
    directions = tuple(
        (case.intervention.relation_before, case.intervention.relation_after)
        for case in manifest.cases
    )
    if directions != _EXPECTED_DIRECTIONS:
        raise GeneralizationCaseError("generalization directions mismatch")
    outcomes = tuple(case.oracle.expected_outcome for case in manifest.cases)
    if outcomes.count("SAT") != 21 or outcomes.count("UNSAT") != 9:
        raise GeneralizationCaseError("generalization outcome counts mismatch")

    loaded: list[LoadedGeneralizationCase] = []
    for case in manifest.cases:
        _validate_common(case)
        oracle = recompute_oracle(case)
        scene_payload = canonical_json_bytes(
            case.scene.model_dump(mode="json"),
            pretty=True,
        )
        loaded.append(
            LoadedGeneralizationCase(
                spec=case,
                scene_sha256=hashlib.sha256(scene_payload).hexdigest(),
                oracle=oracle,
            )
        )
    return tuple(loaded)
