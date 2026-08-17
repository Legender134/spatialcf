"""Dataset-independent acceptance contracts for the certified solver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shapely.geometry import Polygon

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene, Vec2
from spatialcf.geometry.obb import footprints_overlap, inside_room
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.analytic_motion import AnalyticMotionModel, CandidateProjectionError
from spatialcf.solver.certified_models import (
    CertifiedSolveResult,
    CertifiedSolverConfig,
    expected_target_diff,
)
from spatialcf.solver.continuous import CertifiedSpatialCFSolver
from spatialcf.verification.verifier import Verifier


_CASE_IDS = ("left-to-right", "front-to-behind", "near-to-far")
_DIRECTIONS = (
    (Relation.LEFT, Relation.RIGHT),
    (Relation.FRONT, Relation.BEHIND),
    (Relation.NEAR, Relation.FAR),
)
_SOURCE = "core-solver-validation-v1"
_CHECK_NAMES = (
    "source_relation_satisfied",
    "solver_success",
    "certificate_gap_closed",
    "exact_infimum_bracketed",
    "realized_within_tolerance",
    "exact_position_within_tolerance",
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
    "floor_contact",
)


class CoreValidationError(ValueError):
    """The validation input or outcome cannot support an acceptance claim."""

    def __init__(self, message: str, *, failed_checks: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.failed_checks = failed_checks


class ValidationCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+$")
    scene_file: str = Field(pattern=r"^[a-z]+(?:-[a-z]+)+\.json$")
    intervention: InterventionSpec
    exact_infimum_xy: Vec2
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    derivation: str = Field(min_length=1)
    expected_relation_diff: tuple[str, str, str, str]

    @model_validator(mode="after")
    def validate_exact_values(self) -> "ValidationCaseSpec":
        if not all(
            math.isfinite(value)
            for value in (self.exact_infimum_xy.x, self.exact_infimum_xy.y)
        ):
            raise ValueError("exact_infimum_xy must be finite")
        return self


class ValidationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    cases: tuple[ValidationCaseSpec, ...]


@dataclass(frozen=True)
class LoadedValidationCase:
    spec: ValidationCaseSpec
    scene: Scene
    scene_sha256: str


class CertificateRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    distance_lower_bound: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    distance_upper_bound: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    optimality_gap: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    radial_geometry_error: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    numeric_error_bound: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    disk_segments: int = Field(ge=4, strict=True)
    infimum_only: bool = Field(strict=True)


class CaseValidationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    relation_before: Relation
    relation_after: Relation
    before_xy: Vec2
    after_xy: Vec2
    exact_infimum_xy: Vec2
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
    def validate_all_checks(self) -> "CaseValidationRecord":
        if set(self.checks) != set(_CHECK_NAMES) or not all(self.checks.values()):
            raise ValueError("validation record requires every canonical check")
        return self


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["PASS"] = "PASS"
    optimality_tolerance_m: Literal[1e-6] = 1e-6
    cases: tuple[CaseValidationRecord, CaseValidationRecord, CaseValidationRecord]

    @model_validator(mode="after")
    def validate_case_order(self) -> "ValidationReport":
        if tuple(case.case_id for case in self.cases) != _CASE_IDS:
            raise ValueError("validation report case order mismatch")
        return self


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CoreValidationError(f"{label} must be a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CoreValidationError(f"cannot read {label}: {exc}") from exc


def _parse_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoreValidationError(f"invalid UTF-8 JSON in {label}: {exc}") from exc


def _require_finite_scene(scene: Scene, case_id: str) -> None:
    def visit(value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise CoreValidationError(f"{case_id}: scene geometry must be finite")
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(scene.model_dump(mode="python"))


def _floor_bottom(scene: Scene, object_id: str) -> float:
    obj = scene.object_by_id(object_id)
    return obj.obb.center.z - obj.obb.extent.z / 2.0


def _validate_scene(case: ValidationCaseSpec, scene: Scene) -> None:
    if scene.scene_id != case.case_id:
        raise CoreValidationError(f"{case.case_id}: scene_id mismatch")
    if scene.source != _SOURCE:
        raise CoreValidationError(f"{case.case_id}: source mismatch")
    object_ids = tuple(obj.object_id for obj in scene.objects)
    if len(object_ids) != len(set(object_ids)):
        raise CoreValidationError(f"{case.case_id}: object ids must be unique")
    camera_ids = tuple(camera.camera_id for camera in scene.cameras)
    if len(camera_ids) != len(set(camera_ids)):
        raise CoreValidationError(f"{case.case_id}: camera ids must be unique")
    _require_finite_scene(scene, case.case_id)
    room = Polygon([(point.x, point.y) for point in scene.room_polygon_xy])
    if not room.is_valid or room.is_empty or room.area <= 0.0:
        raise CoreValidationError(f"{case.case_id}: room polygon must be valid")

    intervention = case.intervention
    try:
        subject = scene.object_by_id(intervention.subject_id)
        reference = scene.object_by_id(intervention.reference_id)
        scene.camera_by_id(intervention.camera_id)
    except KeyError as exc:
        raise CoreValidationError(f"{case.case_id}: intervention endpoint missing") from exc
    if not subject.movable or reference.movable:
        raise CoreValidationError(f"{case.case_id}: movability contract mismatch")
    if any(obj.movable for obj in scene.objects if obj.object_id != subject.object_id):
        raise CoreValidationError(f"{case.case_id}: only the subject may be movable")
    for endpoint in (subject.object_id, reference.object_id):
        if not math.isclose(_floor_bottom(scene, endpoint), 0.0, abs_tol=1e-12):
            raise CoreValidationError(f"{case.case_id}: floor contact mismatch")

    source = RelationEngine().observe(
        scene,
        intervention.subject_id,
        intervention.reference_id,
        intervention.relation_before,
        intervention.camera_id,
    )
    if source.status is not SolverStatus.SUCCESS or not source.satisfied:
        raise CoreValidationError(f"{case.case_id}: source relation is not satisfied")

    distance = math.hypot(
        case.exact_infimum_xy.x - subject.position.x,
        case.exact_infimum_xy.y - subject.position.y,
    )
    if not math.isclose(distance, case.exact_infimum_m, abs_tol=1e-12):
        raise CoreValidationError(f"{case.case_id}: exact_infimum values disagree")


def load_validation_cases(root: Path) -> tuple[LoadedValidationCase, ...]:
    """Load the exact closed-form validation corpus and fail closed."""
    if root.is_symlink() or not root.is_dir():
        raise CoreValidationError("validation root must be a real directory")
    manifest_payload = _read_regular_file(root / "cases.json", "cases.json")
    try:
        manifest = ValidationManifest.model_validate(
            _parse_json(manifest_payload, "cases.json")
        )
    except ValidationError as exc:
        raise CoreValidationError(f"invalid validation manifest: {exc}") from exc

    case_ids = tuple(case.case_id for case in manifest.cases)
    if len(case_ids) != len(set(case_ids)):
        raise CoreValidationError("validation case ids must be unique")
    if case_ids != _CASE_IDS:
        if set(case_ids) != set(_CASE_IDS):
            raise CoreValidationError("manifest must contain all supported directions")
        raise CoreValidationError("validation case ids must use canonical order")
    directions = tuple(
        (case.intervention.relation_before, case.intervention.relation_after)
        for case in manifest.cases
    )
    if directions != _DIRECTIONS:
        raise CoreValidationError("manifest supported directions do not match")

    loaded: list[LoadedValidationCase] = []
    resolved_root = root.resolve()
    for case in manifest.cases:
        scene_name = Path(case.scene_file)
        if (
            scene_name.name != case.scene_file
            or scene_name.suffix != ".json"
            or case.scene_file == "cases.json"
        ):
            raise CoreValidationError(f"{case.case_id}: invalid scene_file")
        if case.expected_relation_diff != expected_target_diff(case.intervention):
            raise CoreValidationError(
                f"{case.case_id}: expected_relation_diff mismatch"
            )
        scene_path = root / scene_name
        if scene_path.resolve().parent != resolved_root:
            raise CoreValidationError(f"{case.case_id}: scene_file escapes root")
        scene_payload = _read_regular_file(scene_path, f"{case.case_id} scene_file")
        try:
            scene = Scene.model_validate(_parse_json(scene_payload, case.scene_file))
        except ValidationError as exc:
            raise CoreValidationError(f"{case.case_id}: invalid scene: {exc}") from exc
        _validate_scene(case, scene)
        loaded.append(
            LoadedValidationCase(
                spec=case,
                scene=scene,
                scene_sha256=hashlib.sha256(scene_payload).hexdigest(),
            )
        )
    return tuple(loaded)


def _failed(checks: dict[str, bool], case_id: str) -> CoreValidationError:
    failed_checks = tuple(name for name in _CHECK_NAMES if not checks[name])
    return CoreValidationError(
        f"{case_id}: failed checks: {', '.join(failed_checks)}",
        failed_checks=failed_checks,
    )


def validate_solver_outcome(
    case: LoadedValidationCase,
    solve_result: CertifiedSolveResult,
    after: Scene,
) -> CaseValidationRecord:
    """Independently validate a solver result against one closed-form case."""
    before = case.scene
    spec = case.spec.intervention
    tolerance = CertifiedSolverConfig().optimality_tolerance
    before_subject = before.object_by_id(spec.subject_id)
    try:
        after_subject = after.object_by_id(spec.subject_id)
    except KeyError as exc:
        raise CoreValidationError(
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
    exact_position_error = math.hypot(
        after_subject.position.x - case.spec.exact_infimum_xy.x,
        after_subject.position.y - case.spec.exact_infimum_xy.y,
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
        and math.isclose(position_dx, obb_dx, abs_tol=1e-12)
        and math.isclose(position_dy, obb_dy, abs_tol=1e-12)
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
    floor_contact = math.isclose(
        _floor_bottom(before, spec.subject_id),
        0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        _floor_bottom(after, spec.subject_id),
        0.0,
        abs_tol=1e-12,
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
        "exact_position_within_tolerance": exact_position_error <= tolerance,
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
            result_position is not None
            and after_subject.position == result_position
        ),
        "subject_view_matches_analytic_motion": (
            expected_after_subject is not None
            and after_subject == expected_after_subject
        ),
        "inside_room": inside_room(after_subject.obb, room),
        "collision_free": collision_free,
        "floor_contact": floor_contact,
    }
    if not all(checks.values()):
        raise _failed(checks, case.spec.case_id)

    assert certificate is not None
    return CaseValidationRecord(
        case_id=case.spec.case_id,
        scene_sha256=case.scene_sha256,
        derivation=case.spec.derivation,
        relation_before=spec.relation_before,
        relation_after=spec.relation_after,
        before_xy=Vec2(x=before_subject.position.x, y=before_subject.position.y),
        after_xy=Vec2(x=after_subject.position.x, y=after_subject.position.y),
        exact_infimum_xy=case.spec.exact_infimum_xy,
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


def run_validation_case(
    case: LoadedValidationCase,
) -> tuple[CaseValidationRecord, Scene]:
    """Solve and independently validate one committed acceptance case."""
    result = CertifiedSpatialCFSolver().solve(case.scene, case.spec.intervention)
    if result.subject_position is None:
        raise CoreValidationError(
            f"{case.spec.case_id}: solver did not return a position",
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
        raise CoreValidationError(
            f"{case.spec.case_id}: analytic replay failed",
            failed_checks=("subject_view_matches_analytic_motion",),
        ) from exc
    return validate_solver_outcome(case, result, after), after


def run_validation_suite(
    root: Path,
) -> tuple[ValidationReport, dict[str, tuple[Scene, Scene]]]:
    """Run the exact three cases twice and return only repeatable results."""
    records: list[CaseValidationRecord] = []
    scenes: dict[str, tuple[Scene, Scene]] = {}
    for case in load_validation_cases(root):
        first_record, first_after = run_validation_case(case)
        second_record, second_after = run_validation_case(case)
        if first_record != second_record or first_after != second_after:
            raise CoreValidationError(
                f"{case.spec.case_id}: validation is not repeatable"
            )
        records.append(first_record)
        scenes[case.spec.case_id] = (case.scene, first_after)
    if len(records) != 3:
        raise CoreValidationError("validation suite did not produce three cases")
    report = ValidationReport(cases=(records[0], records[1], records[2]))
    return report, scenes
