"""Dataset-independent challenge contracts for the certified solver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from shapely.geometry import Point, Polygon, box

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene, SceneObject, Vec2
from spatialcf.geometry.obb import (
    footprints_overlap,
    inside_room,
    obb_footprint,
)
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.analytic_motion import (
    AnalyticMotionModel,
    CandidateProjectionError,
)
from spatialcf.solver.certified_models import (
    CertifiedSolveResult,
    CertifiedSolverConfig,
    expected_target_diff,
)
from spatialcf.solver.continuous import CertifiedSpatialCFSolver
from spatialcf.solver.validation import CertificateRecord
from spatialcf.verification.verifier import Verifier


_CASE_IDS = (
    "obstacle-final-placement",
    "finite-support-surface",
    "bounded-far-unsat",
)
_DIRECTIONS = (
    (Relation.LEFT, Relation.RIGHT),
    (Relation.FRONT, Relation.BEHIND),
    (Relation.NEAR, Relation.FAR),
)
_SOURCE = "core-solver-challenges-v1"
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_SAT_COMMON_CHECKS = (
    "source_relation_satisfied",
    "solver_success",
    "certificate_gap_closed",
    "exact_infimum_bracketed",
    "realized_within_tolerance",
    "verifier_success",
    "relation_diff_exact",
    "only_subject_xy_changed",
    "subject_z_unchanged",
    "subject_rotation_unchanged",
    "subject_extent_unchanged",
    "subject_identity_unchanged",
    "support_assignment_unchanged",
    "stationary_objects_unchanged",
    "camera_unchanged",
    "room_unchanged",
    "subject_position_matches_result",
    "subject_view_matches_analytic_motion",
    "inside_room",
    "collision_free",
)
_SAT_CASE_CHECKS = {
    "obstacle-final-placement": (
        "at_one_tied_infimum",
        "obstacle_final_contact_only",
        "floor_contact",
    ),
    "finite-support-surface": (
        "at_one_tied_infimum",
        "support_footprint_covers_subject",
        "support_vertical_contact",
    ),
}
_UNSAT_CHECKS = (
    "source_relation_satisfied",
    "analytic_bound_below_requirement",
    "solver_proved_unsat",
    "quality_rejected",
    "reason_exact",
    "position_absent",
    "score_absent",
    "certificate_absent",
)


class ChallengeValidationError(ValueError):
    """The challenge input or result cannot support an acceptance claim."""

    def __init__(self, message: str, *, failed_checks: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.failed_checks = failed_checks


class SatChallengeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+$")
    scene_file: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+\.json$")
    expected_outcome: Literal["SAT"]
    intervention: InterventionSpec
    exact_infimum_points: tuple[Vec2, ...] = Field(min_length=1)
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    derivation: str = Field(min_length=1)
    expected_relation_diff: tuple[str, str, str, str]

    @model_validator(mode="after")
    def validate_points_are_finite(self) -> "SatChallengeSpec":
        if not all(
            math.isfinite(value)
            for point in self.exact_infimum_points
            for value in (point.x, point.y)
        ):
            raise ValueError("exact_infimum_points must be finite")
        return self


class UnsatChallengeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+$")
    scene_file: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+\.json$")
    expected_outcome: Literal["UNSAT"]
    intervention: InterventionSpec
    maximum_possible_gap_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_gap_m: float = Field(gt=0.0, allow_inf_nan=False, strict=True)
    expected_reason: Literal["empty_outer_region"]
    derivation: str = Field(min_length=1)


ChallengeCaseSpec = Annotated[
    SatChallengeSpec | UnsatChallengeSpec,
    Field(discriminator="expected_outcome"),
]


class ChallengeManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    cases: tuple[ChallengeCaseSpec, ...]


@dataclass(frozen=True)
class LoadedChallengeCase:
    spec: SatChallengeSpec | UnsatChallengeSpec
    scene: Scene
    scene_sha256: str


class SatChallengeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_outcome: Literal["SAT"] = "SAT"
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    relation_before: Relation
    relation_after: Relation
    before_xy: Vec2
    after_xy: Vec2
    exact_infimum_points: tuple[Vec2, ...]
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    realized_displacement_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    realized_error_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    solver_status: SolverStatus
    quality: QualityTier
    verifier_status: SolverStatus
    leakage_count: int = Field(ge=0, strict=True)
    changed_relations: tuple[str, ...]
    certificate: CertificateRecord
    checks: dict[str, bool]

    @model_validator(mode="after")
    def validate_checks(self) -> "SatChallengeRecord":
        expected = set(_SAT_COMMON_CHECKS + _SAT_CASE_CHECKS.get(self.case_id, ()))
        if set(self.checks) != expected or not all(self.checks.values()):
            raise ValueError("SAT challenge record requires every canonical check")
        return self


class UnsatChallengeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_outcome: Literal["UNSAT"] = "UNSAT"
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    relation_before: Relation
    relation_after: Relation
    maximum_possible_gap_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_gap_m: float = Field(gt=0.0, allow_inf_nan=False, strict=True)
    solver_status: SolverStatus
    quality: QualityTier
    reason: str
    checks: dict[str, bool]

    @model_validator(mode="after")
    def validate_checks(self) -> "UnsatChallengeRecord":
        if set(self.checks) != set(_UNSAT_CHECKS) or not all(self.checks.values()):
            raise ValueError("UNSAT challenge record requires every canonical check")
        return self


ChallengeRecord = Annotated[
    SatChallengeRecord | UnsatChallengeRecord,
    Field(discriminator="expected_outcome"),
]
ChallengeScenes = dict[str, tuple[Scene, Scene | None]]


class ChallengeValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["PASS"] = "PASS"
    optimality_tolerance_m: Literal[1e-6] = 1e-6
    cases: tuple[ChallengeRecord, ChallengeRecord, ChallengeRecord]

    @model_validator(mode="after")
    def validate_case_order(self) -> "ChallengeValidationReport":
        if tuple(record.case_id for record in self.cases) != _CASE_IDS:
            raise ValueError("challenge report case order mismatch")
        if tuple(record.expected_outcome for record in self.cases) != (
            "SAT",
            "SAT",
            "UNSAT",
        ):
            raise ValueError("challenge report outcome order mismatch")
        return self


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ChallengeValidationError(f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ChallengeValidationError(f"cannot read {label}: {exc}") from exc


def _parse_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChallengeValidationError(f"invalid UTF-8 JSON in {label}: {exc}") from exc


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ChallengeValidationError(f"{label} mismatch")


def _require_values(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    label: str,
) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(actual, expected, strict=True)
    ):
        raise ChallengeValidationError(f"{label} mismatch")


def _vec3(obj: SceneObject, *, extent: bool = False) -> tuple[float, float, float]:
    value = obj.obb.extent if extent else obj.position
    return value.x, value.y, value.z


def _require_object_geometry(
    obj: SceneObject,
    *,
    position: tuple[float, float, float],
    extent: tuple[float, float, float],
    label: str,
) -> None:
    _require_values(_vec3(obj), position, f"{label} position")
    _require_values(
        (obj.obb.center.x, obj.obb.center.y, obj.obb.center.z),
        position,
        f"{label} OBB centre",
    )
    _require_values(_vec3(obj, extent=True), extent, f"{label} extent")
    _require_values(
        (obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w),
        _IDENTITY,
        f"{label} rotation",
    )
    _require_values(
        (
            obj.obb.rotation.x,
            obj.obb.rotation.y,
            obj.obb.rotation.z,
            obj.obb.rotation.w,
        ),
        _IDENTITY,
        f"{label} OBB rotation",
    )


def _require_bounds(
    geometry: Polygon,
    expected: tuple[float, float, float, float],
    label: str,
) -> None:
    _require_values(tuple(float(value) for value in geometry.bounds), expected, label)


def _require_finite_scene(scene: Scene, case_id: str) -> None:
    def visit(value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ChallengeValidationError(f"{case_id}: scene geometry must be finite")
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(scene.model_dump(mode="python"))


def _validate_common_scene(
    case: SatChallengeSpec | UnsatChallengeSpec,
    scene: Scene,
) -> None:
    if scene.scene_id != case.case_id:
        raise ChallengeValidationError(f"{case.case_id}: scene_id mismatch")
    if scene.source != _SOURCE:
        raise ChallengeValidationError(f"{case.case_id}: source mismatch")
    if scene.coordinate_system != "RH_METERS_Z_UP":
        raise ChallengeValidationError(f"{case.case_id}: coordinate system mismatch")
    if scene.generation_seed != 20260723:
        raise ChallengeValidationError(f"{case.case_id}: generation seed mismatch")
    if scene.pinned_object_ids:
        raise ChallengeValidationError(f"{case.case_id}: pinned objects mismatch")
    object_ids = tuple(obj.object_id for obj in scene.objects)
    if len(object_ids) != len(set(object_ids)):
        raise ChallengeValidationError(f"{case.case_id}: object ids must be unique")
    camera_ids = tuple(camera.camera_id for camera in scene.cameras)
    if camera_ids != ("camera",):
        raise ChallengeValidationError(f"{case.case_id}: camera set mismatch")
    _require_finite_scene(scene, case.case_id)
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    if not room.is_valid or room.is_empty or room.area <= 0.0:
        raise ChallengeValidationError(f"{case.case_id}: room polygon must be valid")

    try:
        subject = scene.object_by_id(case.intervention.subject_id)
        reference = scene.object_by_id(case.intervention.reference_id)
        scene.camera_by_id(case.intervention.camera_id)
    except KeyError as exc:
        raise ChallengeValidationError(
            f"{case.case_id}: intervention endpoint missing"
        ) from exc
    if not subject.movable or reference.movable:
        raise ChallengeValidationError(f"{case.case_id}: movability mismatch")
    if any(obj.movable for obj in scene.objects if obj.object_id != subject.object_id):
        raise ChallengeValidationError(
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
        raise ChallengeValidationError(
            f"{case.case_id}: source relation is not satisfied"
        )


def _validate_obstacle_case(case: SatChallengeSpec, scene: Scene) -> None:
    if {obj.object_id for obj in scene.objects} != {
        "subject",
        "reference",
        "obstacle",
    }:
        raise ChallengeValidationError("obstacle-final-placement: object set mismatch")
    subject = scene.object_by_id("subject")
    reference = scene.object_by_id("reference")
    obstacle = scene.object_by_id("obstacle")
    _require_object_geometry(
        subject,
        position=(0.0, 0.0, 0.5),
        extent=(0.2, 0.2, 1.0),
        label="obstacle subject",
    )
    _require_object_geometry(
        reference,
        position=(2.2, 0.0, 0.5),
        extent=(0.2, 0.2, 1.0),
        label="obstacle reference",
    )
    _require_object_geometry(
        obstacle,
        position=(0.448, 0.0, 0.5),
        extent=(0.4, 0.8, 1.0),
        label="obstacle",
    )
    if obstacle.views or obstacle.support_object_id is not None:
        raise ChallengeValidationError("obstacle geometry mismatch")
    camera = scene.camera_by_id("camera")
    _require_values(camera.intrinsics, (500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0), "obstacle camera intrinsics")
    _require_values(
        camera.world_to_camera,
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0),
        "obstacle camera transform",
    )
    subject_view = subject.views["camera"]
    reference_view = reference.views["camera"]
    target_x = subject.position.x + (
        reference_view.bbox.center_x
        + camera.width * RelationEngine.LEFT_RIGHT_FRACTION
        - subject_view.bbox.center_x
    ) * subject_view.camera_depth / camera.intrinsics[0]
    _require_close(target_x, 0.448, "obstacle target boundary")
    footprint = obb_footprint(obstacle.obb)
    _require_bounds(footprint, (0.248, -0.4, 0.648, 0.4), "obstacle footprint")
    half_x = subject.obb.extent.x / 2.0
    half_y = subject.obb.extent.y / 2.0
    min_x, min_y, max_x, max_y = footprint.bounds
    configuration = box(
        min_x - half_x,
        min_y - half_y,
        max_x + half_x,
        max_y + half_y,
    )
    _require_bounds(
        configuration,
        (0.148, -0.5, 0.748, 0.5),
        "obstacle configuration",
    )
    expected_points = (
        (target_x, float(configuration.bounds[1])),
        (target_x, float(configuration.bounds[3])),
    )
    actual_points = tuple((point.x, point.y) for point in case.exact_infimum_points)
    if actual_points != expected_points:
        raise ChallengeValidationError("obstacle infimum points mismatch")
    infimum = math.hypot(target_x - subject.position.x, 0.5)
    if not math.isclose(case.exact_infimum_m, infimum, rel_tol=0.0, abs_tol=1e-12):
        raise ChallengeValidationError("obstacle infimum mismatch")


def _validate_support_case(case: SatChallengeSpec, scene: Scene) -> None:
    if {obj.object_id for obj in scene.objects} != {
        "subject",
        "reference",
        "support",
    }:
        raise ChallengeValidationError("finite-support-surface: object set mismatch")
    subject = scene.object_by_id("subject")
    reference = scene.object_by_id("reference")
    support = scene.object_by_id("support")
    _require_object_geometry(
        subject,
        position=(0.0, 1.0, 0.3),
        extent=(0.2, 0.2, 0.2),
        label="support subject",
    )
    _require_object_geometry(
        reference,
        position=(4.0, 2.0, 0.5),
        extent=(0.2, 0.2, 1.0),
        label="support reference",
    )
    if subject.support_object_id != "support" or support.views:
        raise ChallengeValidationError("support assignment mismatch")
    subject_bottom = subject.obb.center.z - subject.obb.extent.z / 2.0
    support_top = support.obb.center.z + support.obb.extent.z / 2.0
    if not math.isclose(subject_bottom, support_top, rel_tol=0.0, abs_tol=1e-12):
        raise ChallengeValidationError("support contact mismatch")
    _require_object_geometry(
        support,
        position=(0.0, 1.5, 0.1),
        extent=(2.0, 2.0, 0.2),
        label="support",
    )
    support_footprint = obb_footprint(support.obb)
    _require_bounds(
        support_footprint,
        (-1.0, 0.5, 1.0, 2.5),
        "support footprint",
    )
    half_x = subject.obb.extent.x / 2.0
    half_y = subject.obb.extent.y / 2.0
    min_x, min_y, max_x, max_y = support_footprint.bounds
    locus = box(min_x + half_x, min_y + half_y, max_x - half_x, max_y - half_y)
    _require_bounds(locus, (-0.9, 0.6, 0.9, 2.4), "support centre locus")
    subject_view = subject.views["camera"]
    reference_view = reference.views["camera"]
    world_span = reference.position.y - subject.position.y
    depth_span = reference_view.camera_depth - subject_view.camera_depth
    _require_close(world_span, 1.0, "support calibration world span")
    _require_close(depth_span, 1.0, "support calibration depth span")
    target_depth = reference_view.camera_depth + RelationEngine.FRONT_BEHIND_METERS
    target_y = subject.position.y + (target_depth - subject_view.camera_depth) * (
        world_span / depth_span
    )
    target = Point(subject.position.x, target_y)
    if not locus.contains(target):
        raise ChallengeValidationError("support target is not in finite locus")
    actual_points = tuple((point.x, point.y) for point in case.exact_infimum_points)
    if actual_points != ((0.0, target_y),):
        raise ChallengeValidationError("support infimum points mismatch")
    infimum = math.hypot(target.x - subject.position.x, target.y - subject.position.y)
    if not math.isclose(case.exact_infimum_m, infimum, rel_tol=0.0, abs_tol=1e-12):
        raise ChallengeValidationError("support infimum mismatch")


def _validate_unsat_case(case: UnsatChallengeSpec, scene: Scene) -> None:
    if {obj.object_id for obj in scene.objects} != {"subject", "reference"}:
        raise ChallengeValidationError("bounded-far-unsat: object set mismatch")
    subject = scene.object_by_id("subject")
    reference = scene.object_by_id("reference")
    _require_object_geometry(
        subject,
        position=(0.0, 0.0, 0.5),
        extent=(0.2, 0.2, 1.0),
        label="unsat subject",
    )
    _require_object_geometry(
        reference,
        position=(0.4, 0.0, 0.5),
        extent=(0.2, 0.2, 1.0),
        label="unsat reference",
    )
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    _require_bounds(room, (-0.6, -0.6, 1.0, 0.6), "unsat room")
    half_x = subject.obb.extent.x / 2.0
    half_y = subject.obb.extent.y / 2.0
    min_x, min_y, max_x, max_y = room.bounds
    room_locus = box(
        min_x + half_x,
        min_y + half_y,
        max_x - half_x,
        max_y - half_y,
    )
    _require_bounds(room_locus, (-0.5, -0.5, 0.9, 0.5), "unsat room locus")
    reference_footprint = obb_footprint(reference.obb)
    ref_min_x, ref_min_y, ref_max_x, ref_max_y = reference_footprint.bounds
    configuration = box(
        ref_min_x - half_x,
        ref_min_y - half_y,
        ref_max_x + half_x,
        ref_max_y + half_y,
    )
    _require_bounds(
        configuration,
        (0.2, -0.2, 0.6, 0.2),
        "unsat expanded reference",
    )
    locus_min_x, locus_min_y, locus_max_x, locus_max_y = room_locus.bounds
    maximum = max(
        Point(x, y).distance(configuration)
        for x in (locus_min_x, locus_max_x)
        for y in (locus_min_y, locus_max_y)
    )
    if not math.isclose(
        case.maximum_possible_gap_m,
        maximum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ChallengeValidationError("maximum possible gap mismatch")
    _require_close(
        case.required_gap_m,
        RelationEngine.FAR_METERS,
        "required FAR gap",
    )
    if not maximum < case.required_gap_m:
        raise ChallengeValidationError("UNSAT analytic bound does not close")


def _validate_case_geometry(
    case: SatChallengeSpec | UnsatChallengeSpec,
    scene: Scene,
) -> None:
    _validate_common_scene(case, scene)
    if case.case_id == "obstacle-final-placement" and isinstance(case, SatChallengeSpec):
        _validate_obstacle_case(case, scene)
    elif case.case_id == "finite-support-surface" and isinstance(case, SatChallengeSpec):
        _validate_support_case(case, scene)
    elif case.case_id == "bounded-far-unsat" and isinstance(case, UnsatChallengeSpec):
        _validate_unsat_case(case, scene)
    else:
        raise ChallengeValidationError(f"{case.case_id}: outcome type mismatch")


def load_challenge_cases(root: Path) -> tuple[LoadedChallengeCase, ...]:
    """Load the exact closed challenge corpus and reject any drift."""
    if root.is_symlink() or not root.is_dir():
        raise ChallengeValidationError("challenge root must be a real directory")
    manifest_payload = _read_regular_file(root / "cases.json", "cases.json")
    try:
        manifest = ChallengeManifest.model_validate(
            _parse_json(manifest_payload, "cases.json")
        )
    except ValidationError as exc:
        raise ChallengeValidationError(f"invalid challenge manifest: {exc}") from exc

    case_ids = tuple(case.case_id for case in manifest.cases)
    if len(case_ids) != len(set(case_ids)):
        raise ChallengeValidationError("challenge case ids must be unique")
    if set(case_ids) != set(_CASE_IDS):
        raise ChallengeValidationError("manifest must contain the canonical case ids")
    if case_ids != _CASE_IDS:
        raise ChallengeValidationError("challenge cases must use canonical order")
    directions = tuple(
        (case.intervention.relation_before, case.intervention.relation_after)
        for case in manifest.cases
    )
    if directions != _DIRECTIONS:
        raise ChallengeValidationError("challenge directions mismatch")

    resolved_root = root.resolve()
    scene_names: list[str] = []
    for case in manifest.cases:
        scene_name = Path(case.scene_file)
        if (
            scene_name.name != case.scene_file
            or scene_name.suffix != ".json"
            or case.scene_file == "cases.json"
        ):
            raise ChallengeValidationError(f"{case.case_id}: invalid scene_file")
        if (root / scene_name).resolve().parent != resolved_root:
            raise ChallengeValidationError(f"{case.case_id}: scene_file escapes root")
        if isinstance(case, SatChallengeSpec) and (
            case.expected_relation_diff != expected_target_diff(case.intervention)
        ):
            raise ChallengeValidationError(
                f"{case.case_id}: expected_relation_diff mismatch"
            )
        scene_names.append(case.scene_file)

    expected_files = {"cases.json", *scene_names}
    actual_files = {entry.name for entry in root.iterdir()}
    if actual_files != expected_files:
        raise ChallengeValidationError("challenge fixture file set mismatch")

    loaded: list[LoadedChallengeCase] = []
    for case in manifest.cases:
        scene_payload = _read_regular_file(root / case.scene_file, case.scene_file)
        try:
            scene = Scene.model_validate(_parse_json(scene_payload, case.scene_file))
        except ValidationError as exc:
            raise ChallengeValidationError(
                f"{case.case_id}: invalid scene: {exc}"
            ) from exc
        _validate_case_geometry(case, scene)
        loaded.append(
            LoadedChallengeCase(
                spec=case,
                scene=scene,
                scene_sha256=hashlib.sha256(scene_payload).hexdigest(),
            )
        )
    return tuple(loaded)


def _sat_failed(checks: dict[str, bool], case_id: str) -> ChallengeValidationError:
    order = _SAT_COMMON_CHECKS + _SAT_CASE_CHECKS.get(case_id, ())
    failed_checks = tuple(name for name in order if not checks.get(name, False))
    return ChallengeValidationError(
        f"{case_id}: failed checks: {', '.join(failed_checks)}",
        failed_checks=failed_checks,
    )


def _bottom_z(obj: SceneObject) -> float:
    return obj.obb.center.z - obj.obb.extent.z / 2.0


def _top_z(obj: SceneObject) -> float:
    return obj.obb.center.z + obj.obb.extent.z / 2.0


def validate_sat_challenge(
    case: LoadedChallengeCase,
    solve_result: CertifiedSolveResult,
    after: Scene,
) -> SatChallengeRecord:
    """Independently validate one expected-SAT challenge result."""
    if not isinstance(case.spec, SatChallengeSpec):
        raise ChallengeValidationError(
            f"{case.spec.case_id}: expected a SAT challenge specification"
        )
    before = case.scene
    spec = case.spec.intervention
    tolerance = CertifiedSolverConfig().optimality_tolerance
    before_subject = before.object_by_id(spec.subject_id)
    try:
        after_subject = after.object_by_id(spec.subject_id)
    except KeyError as exc:
        raise ChallengeValidationError(
            f"{case.spec.case_id}: subject missing",
            failed_checks=("only_subject_xy_changed",),
        ) from exc

    source = RelationEngine().observe(
        before,
        spec.subject_id,
        spec.reference_id,
        spec.relation_before,
        spec.camera_id,
    )
    certificate = solve_result.certificate
    result_position = solve_result.subject_position
    solver_success = (
        solve_result.status is SolverStatus.SUCCESS
        and solve_result.quality is QualityTier.PURE
        and solve_result.score is not None
        and result_position is not None
        and certificate is not None
    )
    realized_displacement = math.hypot(
        after_subject.position.x - before_subject.position.x,
        after_subject.position.y - before_subject.position.y,
    )
    realized_error = abs(realized_displacement - case.spec.exact_infimum_m)
    exact_position_error = min(
        math.hypot(
            after_subject.position.x - point.x,
            after_subject.position.y - point.y,
        )
        for point in case.spec.exact_infimum_points
    )

    expected_after_subject = None
    if result_position is not None:
        try:
            expected_after_subject = AnalyticMotionModel().with_object_xy(
                before,
                spec.subject_id,
                result_position.x,
                result_position.y,
            ).object_by_id(spec.subject_id)
        except CandidateProjectionError:
            expected_after_subject = None

    verification = Verifier().verify(before, after, spec)
    before_ids = tuple(obj.object_id for obj in before.objects)
    after_ids = tuple(obj.object_id for obj in after.objects)
    position_dx = after_subject.position.x - before_subject.position.x
    position_dy = after_subject.position.y - before_subject.position.y
    obb_dx = after_subject.obb.center.x - before_subject.obb.center.x
    obb_dy = after_subject.obb.center.y - before_subject.obb.center.y
    only_subject_xy = (
        before_ids == after_ids
        and math.isclose(position_dx, obb_dx, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(position_dy, obb_dy, rel_tol=0.0, abs_tol=1e-12)
    )
    stationary_unchanged = before_ids == after_ids and all(
        before.object_by_id(object_id) == after.object_by_id(object_id)
        for object_id in before_ids
        if object_id != spec.subject_id
    )
    room = Polygon([(point.x, point.y) for point in before.room_polygon_xy])
    support_id = before_subject.support_object_id
    collision_free = all(
        not footprints_overlap(after_subject.obb, obj.obb)
        for obj in after.objects
        if obj.object_id not in {spec.subject_id, support_id}
    )

    checks = {
        "source_relation_satisfied": (
            source.status is SolverStatus.SUCCESS and source.satisfied
        ),
        "solver_success": solver_success,
        "certificate_gap_closed": (
            certificate is not None and certificate.optimality_gap <= tolerance
        ),
        "exact_infimum_bracketed": (
            certificate is not None
            and certificate.distance_lower_bound - certificate.numeric_error_bound
            <= case.spec.exact_infimum_m
            <= certificate.distance_upper_bound + certificate.numeric_error_bound
        ),
        "realized_within_tolerance": realized_error <= tolerance,
        "verifier_success": (
            verification.status is SolverStatus.SUCCESS
            and verification.quality is QualityTier.PURE
            and verification.leakage_count == 0
        ),
        "relation_diff_exact": (
            verification.changed_relations == case.spec.expected_relation_diff
        ),
        "only_subject_xy_changed": only_subject_xy,
        "subject_z_unchanged": (
            after_subject.position.z == before_subject.position.z
            and after_subject.obb.center.z == before_subject.obb.center.z
        ),
        "subject_rotation_unchanged": (
            after_subject.rotation == before_subject.rotation
            and after_subject.obb.rotation == before_subject.obb.rotation
        ),
        "subject_extent_unchanged": (
            after_subject.obb.extent == before_subject.obb.extent
        ),
        "subject_identity_unchanged": (
            after_subject.object_id == before_subject.object_id
            and after_subject.name == before_subject.name
            and after_subject.category == before_subject.category
            and after_subject.movable == before_subject.movable
            and after_subject.request_eligible == before_subject.request_eligible
        ),
        "support_assignment_unchanged": (
            after_subject.support_object_id == before_subject.support_object_id
        ),
        "stationary_objects_unchanged": stationary_unchanged,
        "camera_unchanged": after.cameras == before.cameras,
        "room_unchanged": after.room_polygon_xy == before.room_polygon_xy,
        "subject_position_matches_result": (
            result_position is not None and after_subject.position == result_position
        ),
        "subject_view_matches_analytic_motion": (
            expected_after_subject is not None
            and after_subject == expected_after_subject
        ),
        "inside_room": inside_room(after_subject.obb, room),
        "collision_free": collision_free,
    }

    if case.spec.case_id == "obstacle-final-placement":
        obstacle = after.object_by_id("obstacle")
        checks.update(
            {
                "at_one_tied_infimum": exact_position_error <= tolerance,
                "obstacle_final_contact_only": math.isclose(
                    obb_footprint(after_subject.obb)
                    .intersection(obb_footprint(obstacle.obb))
                    .area,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "floor_contact": (
                    math.isclose(
                        _bottom_z(before_subject),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        _bottom_z(after_subject),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
            }
        )
    elif case.spec.case_id == "finite-support-surface":
        before_support = before.object_by_id("support")
        after_support = after.object_by_id("support")
        checks.update(
            {
                "at_one_tied_infimum": exact_position_error <= tolerance,
                "support_footprint_covers_subject": (
                    obb_footprint(before_support.obb).covers(
                        obb_footprint(before_subject.obb)
                    )
                    and obb_footprint(after_support.obb).covers(
                        obb_footprint(after_subject.obb)
                    )
                ),
                "support_vertical_contact": (
                    math.isclose(
                        _bottom_z(before_subject),
                        _top_z(before_support),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        _bottom_z(after_subject),
                        _top_z(after_support),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ),
            }
        )
    else:
        raise ChallengeValidationError(
            f"{case.spec.case_id}: unsupported SAT challenge"
        )

    if not all(checks.values()):
        raise _sat_failed(checks, case.spec.case_id)

    assert certificate is not None
    return SatChallengeRecord(
        case_id=case.spec.case_id,
        scene_sha256=case.scene_sha256,
        derivation=case.spec.derivation,
        relation_before=spec.relation_before,
        relation_after=spec.relation_after,
        before_xy=Vec2(x=before_subject.position.x, y=before_subject.position.y),
        after_xy=Vec2(x=after_subject.position.x, y=after_subject.position.y),
        exact_infimum_points=case.spec.exact_infimum_points,
        exact_infimum_m=case.spec.exact_infimum_m,
        realized_displacement_m=realized_displacement,
        realized_error_m=realized_error,
        solver_status=solve_result.status,
        quality=solve_result.quality,
        verifier_status=verification.status,
        leakage_count=verification.leakage_count,
        changed_relations=verification.changed_relations,
        certificate=CertificateRecord(
            distance_lower_bound=certificate.distance_lower_bound,
            distance_upper_bound=certificate.distance_upper_bound,
            optimality_gap=certificate.optimality_gap,
            radial_geometry_error=certificate.radial_geometry_error,
            numeric_error_bound=certificate.numeric_error_bound,
            disk_segments=certificate.disk_segments,
            infimum_only=certificate.infimum_only,
        ),
        checks=checks,
    )


def validate_unsat_challenge(
    case: LoadedChallengeCase,
    solve_result: CertifiedSolveResult,
) -> UnsatChallengeRecord:
    """Accept only the exact proof contract for one expected-UNSAT challenge."""
    if not isinstance(case.spec, UnsatChallengeSpec):
        raise ChallengeValidationError(
            f"{case.spec.case_id}: expected an UNSAT challenge specification"
        )
    spec = case.spec
    source = RelationEngine().observe(
        case.scene,
        spec.intervention.subject_id,
        spec.intervention.reference_id,
        spec.intervention.relation_before,
        spec.intervention.camera_id,
    )
    checks = {
        "source_relation_satisfied": (
            source.status is SolverStatus.SUCCESS and source.satisfied
        ),
        "analytic_bound_below_requirement": (
            spec.maximum_possible_gap_m < spec.required_gap_m
        ),
        "solver_proved_unsat": (
            solve_result.status is SolverStatus.UNSATISFIABLE
        ),
        "quality_rejected": solve_result.quality is QualityTier.REJECTED,
        "reason_exact": solve_result.reason == spec.expected_reason,
        "position_absent": solve_result.subject_position is None,
        "score_absent": solve_result.score is None,
        "certificate_absent": solve_result.certificate is None,
    }
    if not all(checks.values()):
        failed_checks = tuple(name for name in _UNSAT_CHECKS if not checks[name])
        raise ChallengeValidationError(
            f"{spec.case_id}: failed checks: {', '.join(failed_checks)}",
            failed_checks=failed_checks,
        )
    return UnsatChallengeRecord(
        case_id=spec.case_id,
        scene_sha256=case.scene_sha256,
        derivation=spec.derivation,
        relation_before=spec.intervention.relation_before,
        relation_after=spec.intervention.relation_after,
        maximum_possible_gap_m=spec.maximum_possible_gap_m,
        required_gap_m=spec.required_gap_m,
        solver_status=solve_result.status,
        quality=solve_result.quality,
        reason=solve_result.reason,
        checks=checks,
    )


def run_challenge_case(
    case: LoadedChallengeCase,
) -> tuple[SatChallengeRecord | UnsatChallengeRecord, Scene | None]:
    """Solve and independently validate one committed challenge case."""
    result = CertifiedSpatialCFSolver().solve(case.scene, case.spec.intervention)
    if isinstance(case.spec, UnsatChallengeSpec):
        return validate_unsat_challenge(case, result), None
    if result.subject_position is None:
        raise ChallengeValidationError(
            f"{case.spec.case_id}: solver did not return a SAT position",
            failed_checks=("solver_success",),
        )
    try:
        after = AnalyticMotionModel().with_object_xy(
            case.scene,
            case.spec.intervention.subject_id,
            result.subject_position.x,
            result.subject_position.y,
        )
    except CandidateProjectionError as exc:
        raise ChallengeValidationError(
            f"{case.spec.case_id}: analytic replay failed",
            failed_checks=("subject_view_matches_analytic_motion",),
        ) from exc
    return validate_sat_challenge(case, result, after), after


def run_challenge_suite(
    root: Path,
) -> tuple[ChallengeValidationReport, ChallengeScenes]:
    """Run all challenges twice and return only byte-repeatable evidence."""
    records: list[SatChallengeRecord | UnsatChallengeRecord] = []
    scenes: ChallengeScenes = {}
    for case in load_challenge_cases(root):
        first_record, first_after = run_challenge_case(case)
        second_record, second_after = run_challenge_case(case)
        if first_record != second_record or first_after != second_after:
            raise ChallengeValidationError(
                f"{case.spec.case_id}: challenge result is not repeatable"
            )
        records.append(first_record)
        scenes[case.spec.case_id] = (case.scene, first_after)
    if len(records) != 3:
        raise ChallengeValidationError("challenge suite did not produce three cases")
    report = ChallengeValidationReport(
        cases=(records[0], records[1], records[2])
    )
    return report, scenes
