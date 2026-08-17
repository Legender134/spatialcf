"""Execute and independently validate the fixed solver generalization corpus."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely.geometry import Polygon

from spatialcf.domain.enums import QualityTier, Relation, SolverStatus
from spatialcf.domain.models import Scene, Vec2
from spatialcf.geometry.obb import footprints_overlap, inside_room, obb_footprint
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.analytic_motion import (
    AnalyticMotionModel,
    CandidateProjectionError,
)
from spatialcf.solver.certified_models import (
    CertifiedSolverConfig,
    CertifiedSolveResult,
    expected_target_diff,
)
from spatialcf.solver.continuous import CertifiedSpatialCFSolver
from spatialcf.solver.generalization_cases import (
    GENERALIZATION_CASE_IDS,
    LoadedGeneralizationCase,
    SatOracleSpec,
    UnsatOracleSpec,
    load_generalization_cases,
)
from spatialcf.solver.validation import CertificateRecord
from spatialcf.verification.verifier import Verifier

SAT_CHECKS = (
    "source_relation_satisfied",
    "solver_success",
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
    "inside_room",
    "collision_free",
    "support_footprint_valid",
    "support_vertical_contact",
    "oracle_witness_reached",
    "realized_within_tolerance",
    "certificate_gap_closed",
    "oracle_infimum_bracketed",
)

UNSAT_CHECKS = (
    "source_relation_satisfied",
    "independent_bound_strict",
    "solver_proved_unsat",
    "quality_rejected",
    "reason_exact",
    "position_absent",
    "score_absent",
    "certificate_absent",
)

_POSITION_RESULT_CHECK = "result_position_matches_after"
_STRUCTURE_TOLERANCE = 1e-12


class GeneralizationValidationError(ValueError):
    """One controlled case failed an independent acceptance check."""

    def __init__(
        self,
        message: str,
        *,
        failed_checks: tuple[str, ...] = (),
        checks: dict[str, bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_checks = failed_checks
        self.checks = dict(checks or {})


class SatGeneralizationRecord(BaseModel):
    """Passing evidence for one expected-SAT case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_outcome: Literal["SAT"] = "SAT"
    result: Literal["PASS"] = "PASS"
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    relation_before: Relation
    relation_after: Relation
    before_xy: Vec2
    after_xy: Vec2
    exact_infimum_points: tuple[Vec2, ...] = Field(min_length=1)
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
    def validate_pass_contract(self) -> SatGeneralizationRecord:
        if set(self.checks) != set(SAT_CHECKS) or not all(self.checks.values()):
            raise ValueError("SAT record requires every generalization check")
        return self


class UnsatGeneralizationRecord(BaseModel):
    """Passing evidence for one expected-UNSAT case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_outcome: Literal["UNSAT"] = "UNSAT"
    result: Literal["PASS"] = "PASS"
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    relation_before: Relation
    relation_after: Relation
    maximum_possible_value_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_value_m: float = Field(allow_inf_nan=False, strict=True)
    solver_status: SolverStatus
    quality: QualityTier
    reason: str
    checks: dict[str, bool]

    @model_validator(mode="after")
    def validate_pass_contract(self) -> UnsatGeneralizationRecord:
        if set(self.checks) != set(UNSAT_CHECKS) or not all(self.checks.values()):
            raise ValueError("UNSAT record requires every generalization check")
        return self


class FailedGeneralizationRecord(BaseModel):
    """Retained diagnostics for one controlled case that did not validate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    expected_outcome: Literal["SAT", "UNSAT"]
    result: Literal["FAIL"] = "FAIL"
    scene_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation: str = Field(min_length=1)
    solver_status: SolverStatus
    reason: str | None
    failed_checks: tuple[str, ...] = Field(min_length=1)
    checks: dict[str, bool]


GeneralizationRecord: TypeAlias = (
    SatGeneralizationRecord | UnsatGeneralizationRecord | FailedGeneralizationRecord
)
GeneralizationScenes: TypeAlias = dict[str, tuple[Scene, Scene | None]]


class GeneralizationValidationReport(BaseModel):
    """Closed 30-case report whose status is derived from its records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["PASS", "FAIL"]
    optimality_tolerance_m: Literal[1e-6] = 1e-6
    cases: tuple[GeneralizationRecord, ...] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def validate_closed_report(self) -> GeneralizationValidationReport:
        if tuple(record.case_id for record in self.cases) != GENERALIZATION_CASE_IDS:
            raise ValueError("generalization report case order mismatch")
        expected_status = (
            "PASS" if all(record.result == "PASS" for record in self.cases) else "FAIL"
        )
        if self.status != expected_status:
            raise ValueError("generalization report status mismatch")
        return self


def _bottom(scene: Scene, object_id: str) -> float:
    obj = scene.object_by_id(object_id)
    return obj.obb.center.z - obj.obb.extent.z / 2.0


def _top(scene: Scene, object_id: str) -> float:
    obj = scene.object_by_id(object_id)
    return obj.obb.center.z + obj.obb.extent.z / 2.0


def _source_satisfied(case: LoadedGeneralizationCase) -> bool:
    spec = case.spec.intervention
    source = RelationEngine().observe(
        case.spec.scene,
        spec.subject_id,
        spec.reference_id,
        spec.relation_before,
        spec.camera_id,
    )
    return source.status is SolverStatus.SUCCESS and source.satisfied


def _validation_failure(
    case_id: str,
    ordered_checks: tuple[str, ...],
    checks: dict[str, bool],
    *,
    extra_failed: tuple[str, ...] = (),
) -> GeneralizationValidationError:
    failed = tuple(name for name in ordered_checks if not checks[name]) + extra_failed
    return GeneralizationValidationError(
        f"{case_id}: failed checks: {', '.join(failed)}",
        failed_checks=failed,
        checks=checks,
    )


def _certificate_record(solve_result: CertifiedSolveResult) -> CertificateRecord:
    certificate = solve_result.certificate
    assert certificate is not None
    return CertificateRecord(
        distance_lower_bound=certificate.distance_lower_bound,
        distance_upper_bound=certificate.distance_upper_bound,
        optimality_gap=certificate.optimality_gap,
        radial_geometry_error=certificate.radial_geometry_error,
        numeric_error_bound=certificate.numeric_error_bound,
        disk_segments=certificate.disk_segments,
        infimum_only=certificate.infimum_only,
    )


def validate_sat_generalization(
    case: LoadedGeneralizationCase,
    solve_result: CertifiedSolveResult,
    after: Scene,
) -> SatGeneralizationRecord:
    """Validate an expected-SAT result against geometry and the frozen oracle."""
    oracle_spec = case.spec.oracle
    if not isinstance(oracle_spec, SatOracleSpec):
        raise GeneralizationValidationError(
            f"{case.spec.case_id}: expected a SAT oracle",
            failed_checks=("solver_success",),
        )

    before = case.spec.scene
    spec = case.spec.intervention
    tolerance = CertifiedSolverConfig().optimality_tolerance
    before_subject = before.object_by_id(spec.subject_id)
    try:
        after_subject = after.object_by_id(spec.subject_id)
    except KeyError as exc:
        raise GeneralizationValidationError(
            f"{case.spec.case_id}: subject missing from after scene",
            failed_checks=("only_subject_xy_changed",),
        ) from exc

    certificate = solve_result.certificate
    result_position = solve_result.subject_position
    result_position_matches_after = (
        result_position is not None and result_position == after_subject.position
    )
    solver_success = (
        solve_result.status is SolverStatus.SUCCESS
        and solve_result.quality is QualityTier.PURE
        and solve_result.score is not None
        and result_position is not None
        and certificate is not None
        and all(
            math.isfinite(value)
            for value in (
                result_position.x if result_position is not None else math.inf,
                result_position.y if result_position is not None else math.inf,
                result_position.z if result_position is not None else math.inf,
            )
        )
    )

    realized_displacement = math.hypot(
        after_subject.position.x - before_subject.position.x,
        after_subject.position.y - before_subject.position.y,
    )
    realized_error = abs(realized_displacement - oracle_spec.exact_infimum_m)
    witness_error = min(
        math.hypot(
            after_subject.position.x - point.x,
            after_subject.position.y - point.y,
        )
        for point in oracle_spec.exact_infimum_points
    )

    verification = Verifier().verify(before, after, spec)
    before_ids = tuple(obj.object_id for obj in before.objects)
    after_ids = tuple(obj.object_id for obj in after.objects)
    position_dx = after_subject.position.x - before_subject.position.x
    position_dy = after_subject.position.y - before_subject.position.y
    obb_dx = after_subject.obb.center.x - before_subject.obb.center.x
    obb_dy = after_subject.obb.center.y - before_subject.obb.center.y
    only_subject_xy = (
        before_ids == after_ids
        and math.isclose(
            position_dx,
            obb_dx,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        )
        and math.isclose(
            position_dy,
            obb_dy,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        )
    )
    stationary_unchanged = before_ids == after_ids and all(
        before.object_by_id(object_id) == after.object_by_id(object_id)
        for object_id in before_ids
        if object_id != spec.subject_id
    )

    support_id = before_subject.support_object_id
    if support_id is None:
        support_footprint_valid = True
        support_vertical_contact = math.isclose(
            _bottom(before, spec.subject_id),
            0.0,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        ) and math.isclose(
            _bottom(after, spec.subject_id),
            0.0,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        )
    else:
        try:
            support_before = before.object_by_id(support_id)
            support_after = after.object_by_id(support_id)
        except KeyError:
            support_footprint_valid = False
            support_vertical_contact = False
        else:
            support_footprint_valid = obb_footprint(support_before.obb).buffer(
                1e-6
            ).covers(obb_footprint(before_subject.obb)) and obb_footprint(
                support_after.obb
            ).buffer(1e-6).covers(obb_footprint(after_subject.obb))
            support_vertical_contact = math.isclose(
                _bottom(before, spec.subject_id),
                _top(before, support_id),
                rel_tol=0.0,
                abs_tol=_STRUCTURE_TOLERANCE,
            ) and math.isclose(
                _bottom(after, spec.subject_id),
                _top(after, support_id),
                rel_tol=0.0,
                abs_tol=_STRUCTURE_TOLERANCE,
            )

    room = Polygon([(point.x, point.y) for point in before.room_polygon_xy])
    collision_free = all(
        not footprints_overlap(after_subject.obb, obj.obb)
        for obj in after.objects
        if obj.object_id not in {spec.subject_id, support_id}
    )
    checks = {
        "source_relation_satisfied": _source_satisfied(case),
        "solver_success": solver_success,
        "verifier_success": (
            verification.status is SolverStatus.SUCCESS
            and verification.quality is QualityTier.PURE
            and verification.leakage_count == 0
        ),
        "relation_diff_exact": (
            verification.changed_relations == expected_target_diff(spec)
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
        "inside_room": inside_room(after_subject.obb, room),
        "collision_free": collision_free,
        "support_footprint_valid": support_footprint_valid,
        "support_vertical_contact": support_vertical_contact,
        "oracle_witness_reached": witness_error <= tolerance,
        "realized_within_tolerance": realized_error <= tolerance,
        "certificate_gap_closed": (
            certificate is not None and certificate.optimality_gap <= tolerance
        ),
        "oracle_infimum_bracketed": (
            certificate is not None
            and certificate.distance_lower_bound - certificate.numeric_error_bound
            <= oracle_spec.exact_infimum_m
            <= certificate.distance_upper_bound + certificate.numeric_error_bound
        ),
    }
    extra_failed = () if result_position_matches_after else (_POSITION_RESULT_CHECK,)
    if not all(checks.values()) or extra_failed:
        raise _validation_failure(
            case.spec.case_id,
            SAT_CHECKS,
            checks,
            extra_failed=extra_failed,
        )

    assert certificate is not None
    return SatGeneralizationRecord(
        case_id=case.spec.case_id,
        scene_sha256=case.scene_sha256,
        derivation=oracle_spec.derivation,
        relation_before=spec.relation_before,
        relation_after=spec.relation_after,
        before_xy=Vec2(x=before_subject.position.x, y=before_subject.position.y),
        after_xy=Vec2(x=after_subject.position.x, y=after_subject.position.y),
        exact_infimum_points=oracle_spec.exact_infimum_points,
        exact_infimum_m=oracle_spec.exact_infimum_m,
        realized_displacement_m=realized_displacement,
        realized_error_m=realized_error,
        solver_status=solve_result.status,
        quality=solve_result.quality,
        verifier_status=verification.status,
        leakage_count=verification.leakage_count,
        changed_relations=verification.changed_relations,
        certificate=_certificate_record(solve_result),
        checks=checks,
    )


def validate_unsat_generalization(
    case: LoadedGeneralizationCase,
    solve_result: CertifiedSolveResult,
) -> UnsatGeneralizationRecord:
    """Validate a real UNSAT proof; operational failures are never accepted."""
    oracle_spec = case.spec.oracle
    if not isinstance(oracle_spec, UnsatOracleSpec):
        raise GeneralizationValidationError(
            f"{case.spec.case_id}: expected an UNSAT oracle",
            failed_checks=("solver_proved_unsat",),
        )
    maximum = case.oracle.maximum_possible_value_m
    required = case.oracle.required_value_m
    assert maximum is not None and required is not None
    checks = {
        "source_relation_satisfied": _source_satisfied(case),
        "independent_bound_strict": maximum < required,
        "solver_proved_unsat": (solve_result.status is SolverStatus.UNSATISFIABLE),
        "quality_rejected": solve_result.quality is QualityTier.REJECTED,
        "reason_exact": solve_result.reason == oracle_spec.expected_reason,
        "position_absent": solve_result.subject_position is None,
        "score_absent": solve_result.score is None,
        "certificate_absent": solve_result.certificate is None,
    }
    if not all(checks.values()):
        raise _validation_failure(case.spec.case_id, UNSAT_CHECKS, checks)
    assert solve_result.reason is not None
    return UnsatGeneralizationRecord(
        case_id=case.spec.case_id,
        scene_sha256=case.scene_sha256,
        derivation=oracle_spec.derivation,
        relation_before=case.spec.intervention.relation_before,
        relation_after=case.spec.intervention.relation_after,
        maximum_possible_value_m=maximum,
        required_value_m=required,
        solver_status=solve_result.status,
        quality=solve_result.quality,
        reason=solve_result.reason,
        checks=checks,
    )


def _failed_record(
    case: LoadedGeneralizationCase,
    solve_result: CertifiedSolveResult,
    failure: GeneralizationValidationError,
) -> FailedGeneralizationRecord:
    oracle_spec = case.spec.oracle
    return FailedGeneralizationRecord(
        case_id=case.spec.case_id,
        expected_outcome=oracle_spec.expected_outcome,
        scene_sha256=case.scene_sha256,
        derivation=oracle_spec.derivation,
        solver_status=solve_result.status,
        reason=solve_result.reason,
        failed_checks=failure.failed_checks or ("validation_failed",),
        checks=failure.checks,
    )


def run_generalization_suite(
    root: Path,
) -> tuple[GeneralizationValidationReport, GeneralizationScenes]:
    """Solve all 30 canonical cases and aggregate controlled failures."""
    records: list[GeneralizationRecord] = []
    scenes: GeneralizationScenes = {}
    solver = CertifiedSpatialCFSolver()
    motion = AnalyticMotionModel()

    for case in load_generalization_cases(root):
        solve_result = solver.solve(case.spec.scene, case.spec.intervention)
        after: Scene | None = None
        try:
            if isinstance(case.spec.oracle, SatOracleSpec):
                position = solve_result.subject_position
                if position is None or not all(
                    math.isfinite(value)
                    for value in (position.x, position.y, position.z)
                ):
                    raise GeneralizationValidationError(
                        f"{case.spec.case_id}: solver did not return a finite position",
                        failed_checks=("solver_success",),
                        checks={"solver_success": False},
                    )
                try:
                    after = motion.with_object_xy(
                        case.spec.scene,
                        case.spec.intervention.subject_id,
                        position.x,
                        position.y,
                    )
                except CandidateProjectionError as exc:
                    raise GeneralizationValidationError(
                        f"{case.spec.case_id}: analytic replay failed",
                        failed_checks=("solver_success",),
                        checks={"solver_success": False},
                    ) from exc
                record: GeneralizationRecord = validate_sat_generalization(
                    case,
                    solve_result,
                    after,
                )
            else:
                record = validate_unsat_generalization(case, solve_result)
        except GeneralizationValidationError as exc:
            record = _failed_record(case, solve_result, exc)
        records.append(record)
        scenes[case.spec.case_id] = (case.spec.scene, after)

    status: Literal["PASS", "FAIL"] = (
        "PASS" if all(record.result == "PASS" for record in records) else "FAIL"
    )
    report = GeneralizationValidationReport(status=status, cases=tuple(records))
    return report, scenes
