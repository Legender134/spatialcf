"""Run production stress solves through independent acceptance checks."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from shapely.geometry import Polygon

from spatialcf.data.artifacts import canonical_json_bytes
from spatialcf.domain.enums import QualityTier, SolverStatus
from spatialcf.domain.models import Scene, Vec2
from spatialcf.geometry.obb import inside_room, obb_footprint
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver import CertifiedSolverConfig, CertifiedSpatialCFSolver
from spatialcf.solver.analytic_motion import (
    AnalyticMotionModel,
    CandidateProjectionError,
)
from spatialcf.solver.certified_models import CertifiedSolveResult, expected_target_diff
from spatialcf.solver.stress.cases import (
    _first_valid_draft,
    _placeholder_digest,
    generate_stress_cases,
    replay_stress_case,
)
from spatialcf.solver.stress.models import (
    SatStressOracle,
    StressCase,
    StressDirection,
    StressFamily,
    StressProfileName,
    StressTransform,
    UnsatStressOracle,
)
from spatialcf.solver.stress.profiles import (
    DEEP_SEEDS,
    QUICK_SEEDS,
    SAT_FAMILIES,
    STRESS_DIRECTIONS,
    UNSAT_FAMILIES,
    stress_slots,
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
    "result_position_matches_after",
    "oracle_witness_reached",
    "realized_within_tolerance",
    "certificate_consistent",
    "certificate_gap_closed",
    "oracle_infimum_bracketed",
    "transform_invariants",
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
    "transform_invariants",
)

_STRUCTURE_TOLERANCE = 1e-9
_ORACLE_ROUNDING_ERROR_M = 0.5e-12
_NUMERIC_ERROR_BOUND_M = 1e-9


class _FrozenValidationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FrozenCheckMap(Mapping[str, bool]):
    """Insertion-ordered immutable mapping with ordinary JSON serialization."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, bool]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("FrozenCheckMap is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenCheckMap is immutable")

    def __getitem__(self, key: str) -> bool:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))


class _FrozenChecksModel(_FrozenValidationModel):
    solver_result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_scene_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    checks: Mapping[str, bool]

    @field_validator("checks", mode="after")
    @classmethod
    def freeze_checks(cls, value: Mapping[str, bool]) -> FrozenCheckMap:
        return FrozenCheckMap(value)

    @field_serializer("checks")
    def serialize_checks(self, value: Mapping[str, bool]) -> dict[str, bool]:
        return dict(value.items())


class StressValidationError(ValueError):
    """One stress result failed its independent acceptance contract."""

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


class FailedStressValidationRecord(_FrozenChecksModel):
    """Deterministic diagnostics for one controlled validation failure."""

    case_id: str = Field(pattern=r"^stress-\d{10}-(?:lr|fb|nf)-\d{3}$")
    seed: int = Field(ge=2026080200, le=2026080209, strict=True)
    direction: StressDirection
    raw_slot: int = Field(ge=0, le=99, strict=True)
    family: StressFamily
    transform: StressTransform
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_outcome: Literal["SAT", "UNSAT"]
    actual_outcome: SolverStatus
    result: Literal["FAIL"] = "FAIL"
    quality: QualityTier
    reason: str | None
    certificate: CertificateRecord | None
    errors: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_failed_checks(self) -> FailedStressValidationRecord:
        failed = tuple(name for name, passed in self.checks.items() if not passed)
        if failed != self.errors:
            raise ValueError("failure errors must equal failed checks in order")
        return self


class SatStressValidationRecord(_FrozenChecksModel):
    """Passing independent evidence for one expected-SAT stress case."""

    case_id: str = Field(pattern=r"^stress-\d{10}-(?:lr|fb|nf)-\d{3}$")
    seed: int = Field(ge=2026080200, le=2026080209, strict=True)
    direction: StressDirection
    raw_slot: int = Field(ge=0, le=99, strict=True)
    family: StressFamily
    transform: StressTransform
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_outcome: Literal["SAT"] = "SAT"
    actual_outcome: SolverStatus
    result: Literal["PASS"] = "PASS"
    quality: QualityTier
    reason: None = None
    certificate: CertificateRecord
    errors: tuple[str, ...] = ()
    before_xy: Vec2
    after_xy: Vec2
    exact_infimum_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    exact_infimum_points: tuple[Vec2, ...] = Field(min_length=1)
    realized_displacement_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    realized_error_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    witness_error_m: float = Field(ge=0.0, allow_inf_nan=False, strict=True)
    verifier_status: SolverStatus
    leakage_count: int = Field(ge=0, strict=True)
    changed_relations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_pass_contract(self) -> SatStressValidationRecord:
        if (
            tuple(self.checks) != SAT_CHECKS
            or not all(self.checks.values())
            or self.errors
        ):
            raise ValueError("SAT record requires every stress check")
        return self


class UnsatStressValidationRecord(_FrozenChecksModel):
    """Passing independent evidence for one expected-UNSAT stress case."""

    case_id: str = Field(pattern=r"^stress-\d{10}-(?:lr|fb|nf)-\d{3}$")
    seed: int = Field(ge=2026080200, le=2026080209, strict=True)
    direction: StressDirection
    raw_slot: int = Field(ge=0, le=99, strict=True)
    family: StressFamily
    transform: StressTransform
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_outcome: Literal["UNSAT"] = "UNSAT"
    actual_outcome: SolverStatus
    result: Literal["PASS"] = "PASS"
    quality: QualityTier
    reason: str
    certificate: None = None
    errors: tuple[str, ...] = ()
    maximum_possible_value_m: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    required_value_m: float = Field(allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def validate_pass_contract(self) -> UnsatStressValidationRecord:
        if (
            tuple(self.checks) != UNSAT_CHECKS
            or not all(self.checks.values())
            or self.errors
        ):
            raise ValueError("UNSAT record requires every stress check")
        return self


StressValidationRecord = (
    SatStressValidationRecord
    | UnsatStressValidationRecord
    | FailedStressValidationRecord
)


class StressValidationReport(_FrozenValidationModel):
    """Deterministic aggregate result for one requested stress run."""

    schema_version: Literal[1] = 1
    profile: StressProfileName
    requested_case_id: str | None = Field(
        default=None,
        pattern=r"^stress-\d{10}-(?:lr|fb|nf)-\d{3}$",
    )
    status: Literal["PASS", "FAIL"]
    optimality_tolerance_m: Literal[1e-6] = 1e-6
    seeds: tuple[int, ...] = Field(min_length=1)
    case_count: int = Field(ge=1, strict=True)
    failure_count: int = Field(ge=0, strict=True)
    cases: tuple[StressValidationRecord, ...] = Field(min_length=1)

    def failure_summary(self) -> str:
        """Return deterministic concise diagnostics for every failed case."""
        return "; ".join(
            f"{record.case_id}: {record.reason or ','.join(record.errors)}"
            for record in self.cases
            if isinstance(record, FailedStressValidationRecord)
        )

    @model_validator(mode="before")
    @classmethod
    def derive_counts(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cases = value.get("cases")
        if not isinstance(cases, (list, tuple)):
            return value
        expected_cases = len(cases)
        expected_failures = sum(
            (item.get("result") if isinstance(item, dict) else item.result) == "FAIL"
            for item in cases
        )
        for name, expected in (
            ("case_count", expected_cases),
            ("failure_count", expected_failures),
        ):
            if name in value and value[name] != expected:
                raise ValueError(f"stress report {name} mismatch")
        return value | {
            "case_count": expected_cases,
            "failure_count": expected_failures,
        }

    @model_validator(mode="after")
    def validate_closed_report(self) -> StressValidationReport:
        expected_status = "PASS" if self.failure_count == 0 else "FAIL"
        if self.status != expected_status:
            raise ValueError("stress report status mismatch")
        case_ids = tuple(record.case_id for record in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("stress report case IDs must be unique")
        if self.requested_case_id is not None and case_ids != (self.requested_case_id,):
            raise ValueError("single-case stress report identity mismatch")
        observed_seeds = tuple(dict.fromkeys(record.seed for record in self.cases))
        if self.seeds != observed_seeds:
            raise ValueError("stress report seed order mismatch")
        if self.requested_case_id is None:
            expected_seeds = QUICK_SEEDS if self.profile == "quick" else DEEP_SEEDS
            if self.seeds != expected_seeds:
                raise ValueError("stress profile seeds mismatch")
            expected_identity = tuple(
                (seed, direction, f"stress-{seed}-{direction}-{index:03d}")
                for seed in expected_seeds
                for direction in STRESS_DIRECTIONS
                for index in range(100)
            )
            observed_identity = tuple(
                (record.seed, record.direction, record.case_id) for record in self.cases
            )
            if len(self.cases) != len(expected_identity):
                raise ValueError("stress profile case count mismatch")
            if observed_identity != expected_identity:
                raise ValueError("stress profile case order mismatch")
            expected_coverage = Counter(
                {
                    **{(family, "SAT"): 12 for family in SAT_FAMILIES},
                    **{(family, "UNSAT"): 10 for family in UNSAT_FAMILIES},
                }
            )
            for offset in range(0, len(self.cases), 100):
                seed, direction, _ = expected_identity[offset]
                coverage = Counter(
                    (record.family, record.expected_outcome)
                    for record in self.cases[offset : offset + 100]
                )
                if coverage != expected_coverage:
                    raise ValueError("stress profile family/outcome coverage mismatch")
                expected_slots = Counter(
                    (slot.raw_slot, slot.family, slot.expected_outcome)
                    for slot in stress_slots(self.profile)
                    if slot.seed == seed and slot.direction == direction
                )
                observed_slots = Counter(
                    (record.raw_slot, record.family, record.expected_outcome)
                    for record in self.cases[offset : offset + 100]
                )
                if observed_slots != expected_slots:
                    raise ValueError("stress profile raw-slot schedule mismatch")
        return self


class StressValidationEvidence(_FrozenValidationModel):
    """In-memory inputs and outputs retained for later publication."""

    case: StressCase
    solve_result: CertifiedSolveResult
    after_scene: Scene | None


def _case_sha256(case: StressCase) -> str:
    return hashlib.sha256(
        canonical_json_bytes(case.model_dump(mode="json"))
    ).hexdigest()


def _canonical_digest_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0.0:
            label = "+infinity"
        else:
            label = "-infinity"
        return {"non_finite_float": label}
    if isinstance(value, dict):
        return {
            str(key): _canonical_digest_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_digest_value(item) for item in value]
    return value


def stress_solver_result_payload(
    solve_result: CertifiedSolveResult,
) -> dict[str, object]:
    """Return a total canonical JSON payload, including malformed numerics."""
    payload = {
        "status": solve_result.status.value,
        "subject_position": (
            solve_result.subject_position.model_dump(mode="json")
            if solve_result.subject_position is not None
            else None
        ),
        "score": asdict(solve_result.score) if solve_result.score is not None else None,
        "quality": solve_result.quality.value,
        "evaluated_candidates": solve_result.evaluated_candidates,
        "reason": solve_result.reason,
        "certificate": (
            asdict(solve_result.certificate)
            if solve_result.certificate is not None
            else None
        ),
    }
    normalized = _canonical_digest_value(payload)
    assert isinstance(normalized, dict)
    return normalized


def stress_solver_result_digest(solve_result: CertifiedSolveResult) -> str:
    """Bind every solver-result field to one deterministic SHA-256 digest."""
    return hashlib.sha256(
        canonical_json_bytes(stress_solver_result_payload(solve_result))
    ).hexdigest()


def stress_after_scene_digest(after_scene: Scene | None) -> str | None:
    """Bind a retained replay scene, while preserving absence as ``None``."""
    if after_scene is None:
        return None
    payload = _canonical_digest_value(after_scene.model_dump(mode="json"))
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_evidence_digests(
    solve_result: CertifiedSolveResult,
    after_scene: Scene | None,
) -> dict[str, object]:
    return {
        "solver_result_digest": stress_solver_result_digest(solve_result),
        "after_scene_digest": stress_after_scene_digest(after_scene),
    }


def _record_metadata(case: StressCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "direction": case.direction,
        "raw_slot": case.raw_slot,
        "family": case.family,
        "transform": case.transform,
        "case_sha256": _case_sha256(case),
    }


def _source_relation_satisfied(case: StressCase) -> bool:
    spec = case.intervention
    observed = RelationEngine().observe(
        case.scene,
        spec.subject_id,
        spec.reference_id,
        spec.relation_before,
        spec.camera_id,
    )
    return observed.status is SolverStatus.SUCCESS and observed.satisfied


def _transform_invariants(case: StressCase) -> bool:
    slot = next(
        (
            slot
            for slot in stress_slots("deep")
            if slot.seed == case.seed
            and slot.direction == case.direction
            and slot.raw_slot == case.raw_slot
            and slot.family == case.family
            and slot.expected_outcome == case.expected_outcome
        ),
        None,
    )
    if slot is None:
        return False
    expected = _first_valid_draft(slot)
    return _placeholder_digest(expected) == _placeholder_digest(case.as_draft())


def _certificate_consistency(
    solve_result: CertifiedSolveResult,
) -> tuple[bool, float]:
    certificate = solve_result.certificate
    if certificate is None:
        return False, math.inf
    values = (
        certificate.distance_lower_bound,
        certificate.distance_upper_bound,
        certificate.optimality_gap,
        certificate.radial_geometry_error,
        certificate.numeric_error_bound,
    )
    if not all(
        type(value) in {int, float} and math.isfinite(value)
        for value in values
    ):
        return False, math.inf
    recomputed_gap = (
        certificate.distance_upper_bound
        - certificate.distance_lower_bound
        + 2.0 * _NUMERIC_ERROR_BOUND_M
    )
    consistent = (
        certificate.distance_lower_bound >= 0.0
        and certificate.distance_upper_bound >= certificate.distance_lower_bound
        and certificate.radial_geometry_error >= 0.0
        and certificate.numeric_error_bound == _NUMERIC_ERROR_BOUND_M
        and certificate.optimality_gap == recomputed_gap
        and type(certificate.disk_segments) is int
        and 128 <= certificate.disk_segments <= 8192
        and certificate.disk_segments & (certificate.disk_segments - 1) == 0
        and type(certificate.infimum_only) is bool
    )
    return consistent, recomputed_gap


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


def _safe_certificate_record(
    solve_result: CertifiedSolveResult,
) -> CertificateRecord | None:
    certificate = solve_result.certificate
    if certificate is None:
        return None
    scalar_values = (
        certificate.distance_lower_bound,
        certificate.distance_upper_bound,
        certificate.optimality_gap,
        certificate.radial_geometry_error,
        certificate.numeric_error_bound,
    )
    if (
        not all(
            type(value) in {int, float} and math.isfinite(value) and value >= 0.0
            for value in scalar_values
        )
        or type(certificate.disk_segments) is not int
        or certificate.disk_segments < 4
        or type(certificate.infimum_only) is not bool
    ):
        return None
    return _certificate_record(solve_result)


def _bottom(scene: Scene, object_id: str) -> float:
    obj = scene.object_by_id(object_id)
    return obj.obb.center.z - obj.obb.extent.z / 2.0


def _top(scene: Scene, object_id: str) -> float:
    obj = scene.object_by_id(object_id)
    return obj.obb.center.z + obj.obb.extent.z / 2.0


def _collision_free(case: StressCase, after: Scene) -> bool:
    spec = case.intervention
    subject = after.object_by_id(spec.subject_id)
    support_id = subject.support_object_id
    subject_bottom = _bottom(after, spec.subject_id)
    subject_top = _top(after, spec.subject_id)
    footprint = obb_footprint(subject.obb)
    for stationary in after.objects:
        if stationary.object_id in {spec.subject_id, support_id}:
            continue
        vertical_overlap = min(subject_top, _top(after, stationary.object_id)) - max(
            subject_bottom,
            _bottom(after, stationary.object_id),
        )
        if (
            vertical_overlap > _STRUCTURE_TOLERANCE
            and footprint.intersection(obb_footprint(stationary.obb)).area
            > _STRUCTURE_TOLERANCE
        ):
            return False
    return True


def _support_checks(case: StressCase, after: Scene) -> tuple[bool, bool]:
    spec = case.intervention
    before_subject = case.scene.object_by_id(spec.subject_id)
    after_subject = after.object_by_id(spec.subject_id)
    support_id = before_subject.support_object_id
    if support_id is None:
        return True, True
    try:
        support_before = case.scene.object_by_id(support_id)
        support_after = after.object_by_id(support_id)
    except KeyError:
        return False, False
    footprint_valid = obb_footprint(support_before.obb).buffer(
        _STRUCTURE_TOLERANCE
    ).covers(obb_footprint(before_subject.obb)) and obb_footprint(
        support_after.obb
    ).buffer(_STRUCTURE_TOLERANCE).covers(obb_footprint(after_subject.obb))
    vertical_contact = math.isclose(
        _bottom(case.scene, spec.subject_id),
        _top(case.scene, support_id),
        rel_tol=0.0,
        abs_tol=_STRUCTURE_TOLERANCE,
    ) and math.isclose(
        _bottom(after, spec.subject_id),
        _top(after, support_id),
        rel_tol=0.0,
        abs_tol=_STRUCTURE_TOLERANCE,
    )
    return footprint_valid, vertical_contact


def _raise_validation_failure(
    case: StressCase,
    ordered_checks: tuple[str, ...],
    checks: dict[str, bool],
) -> None:
    failed = tuple(name for name in ordered_checks if not checks[name])
    if failed:
        raise StressValidationError(
            f"{case.case_id}: failed checks: {', '.join(failed)}",
            failed_checks=failed,
            checks=checks,
        )


def validate_stress_sat(
    case: StressCase,
    solve_result: CertifiedSolveResult,
    after: Scene,
) -> SatStressValidationRecord:
    """Validate an expected-SAT result independently of solver geometry."""
    oracle = case.oracle
    if not isinstance(oracle, SatStressOracle):
        raise StressValidationError(
            f"{case.case_id}: expected a SAT oracle",
            failed_checks=("solver_success",),
        )
    spec = case.intervention
    before = case.scene
    before_subject = before.object_by_id(spec.subject_id)
    try:
        after_subject = after.object_by_id(spec.subject_id)
    except KeyError as exc:
        raise StressValidationError(
            f"{case.case_id}: subject missing from after scene",
            failed_checks=("only_subject_xy_changed",),
        ) from exc
    certificate = solve_result.certificate
    position = solve_result.subject_position
    solver_success = (
        solve_result.status is SolverStatus.SUCCESS
        and solve_result.quality is QualityTier.PURE
        and solve_result.score is not None
        and position is not None
        and certificate is not None
        and all(
            math.isfinite(value)
            for value in (
                position.x if position is not None else math.inf,
                position.y if position is not None else math.inf,
                position.z if position is not None else math.inf,
            )
        )
    )
    verification = Verifier().verify(before, after, spec)
    before_ids = tuple(obj.object_id for obj in before.objects)
    after_ids = tuple(obj.object_id for obj in after.objects)
    position_dx = after_subject.position.x - before_subject.position.x
    position_dy = after_subject.position.y - before_subject.position.y
    center_dx = after_subject.obb.center.x - before_subject.obb.center.x
    center_dy = after_subject.obb.center.y - before_subject.obb.center.y
    only_subject_xy = (
        before_ids == after_ids
        and math.isclose(
            position_dx,
            center_dx,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        )
        and math.isclose(
            position_dy,
            center_dy,
            rel_tol=0.0,
            abs_tol=_STRUCTURE_TOLERANCE,
        )
    )
    stationary_unchanged = before_ids == after_ids and all(
        before.object_by_id(object_id) == after.object_by_id(object_id)
        for object_id in before_ids
        if object_id != spec.subject_id
    )
    support_footprint_valid, support_vertical_contact = _support_checks(case, after)
    room = Polygon([(point.x, point.y) for point in before.room_polygon_xy])
    realized_displacement = math.hypot(position_dx, position_dy)
    realized_error = abs(realized_displacement - oracle.exact_infimum_m)
    witness_error = min(
        math.hypot(
            after_subject.position.x - witness.x,
            after_subject.position.y - witness.y,
        )
        for witness in oracle.exact_infimum_points
    )
    certificate_consistent, recomputed_gap = _certificate_consistency(solve_result)
    checks = {
        "source_relation_satisfied": _source_relation_satisfied(case),
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
        "collision_free": _collision_free(case, after),
        "support_footprint_valid": support_footprint_valid,
        "support_vertical_contact": support_vertical_contact,
        "result_position_matches_after": (
            position is not None and position == after_subject.position
        ),
        "oracle_witness_reached": witness_error <= 1e-6,
        "realized_within_tolerance": realized_error <= 1e-6,
        "certificate_consistent": certificate_consistent,
        "certificate_gap_closed": certificate_consistent and recomputed_gap <= 1e-6,
        "oracle_infimum_bracketed": (
            certificate_consistent
            and certificate is not None
            and certificate.distance_lower_bound
            - certificate.numeric_error_bound
            - _ORACLE_ROUNDING_ERROR_M
            <= oracle.exact_infimum_m
            <= certificate.distance_upper_bound
            + certificate.numeric_error_bound
            + _ORACLE_ROUNDING_ERROR_M
        ),
        "transform_invariants": _transform_invariants(case),
    }
    _raise_validation_failure(case, SAT_CHECKS, checks)
    assert certificate is not None
    return SatStressValidationRecord(
        **_record_metadata(case),
        **_record_evidence_digests(solve_result, after),
        actual_outcome=solve_result.status,
        quality=solve_result.quality,
        certificate=_certificate_record(solve_result),
        checks=checks,
        before_xy=Vec2(x=before_subject.position.x, y=before_subject.position.y),
        after_xy=Vec2(x=after_subject.position.x, y=after_subject.position.y),
        exact_infimum_m=oracle.exact_infimum_m,
        exact_infimum_points=oracle.exact_infimum_points,
        realized_displacement_m=realized_displacement,
        realized_error_m=realized_error,
        witness_error_m=witness_error,
        verifier_status=verification.status,
        leakage_count=verification.leakage_count,
        changed_relations=verification.changed_relations,
    )


def validate_stress_unsat(
    case: StressCase,
    solve_result: CertifiedSolveResult,
) -> UnsatStressValidationRecord:
    """Accept only a real UNSAT proof with the frozen independent bound."""
    oracle = case.oracle
    if not isinstance(oracle, UnsatStressOracle):
        raise StressValidationError(
            f"{case.case_id}: expected an UNSAT oracle",
            failed_checks=("solver_proved_unsat",),
        )
    checks = {
        "source_relation_satisfied": _source_relation_satisfied(case),
        "independent_bound_strict": (
            oracle.maximum_possible_value_m < oracle.required_value_m
        ),
        "solver_proved_unsat": (solve_result.status is SolverStatus.UNSATISFIABLE),
        "quality_rejected": solve_result.quality is QualityTier.REJECTED,
        "reason_exact": solve_result.reason == oracle.expected_reason,
        "position_absent": solve_result.subject_position is None,
        "score_absent": solve_result.score is None,
        "certificate_absent": solve_result.certificate is None,
        "transform_invariants": _transform_invariants(case),
    }
    _raise_validation_failure(case, UNSAT_CHECKS, checks)
    assert solve_result.reason is not None
    return UnsatStressValidationRecord(
        **_record_metadata(case),
        **_record_evidence_digests(solve_result, None),
        actual_outcome=solve_result.status,
        quality=solve_result.quality,
        reason=solve_result.reason,
        checks=checks,
        maximum_possible_value_m=oracle.maximum_possible_value_m,
        required_value_m=oracle.required_value_m,
    )


def _failed_record(
    case: StressCase,
    solve_result: CertifiedSolveResult,
    failure: StressValidationError,
    after_scene: Scene | None,
) -> FailedStressValidationRecord:
    failed_checks = failure.failed_checks or ("validation_failed",)
    checks = dict(failure.checks)
    for name in failed_checks:
        checks[name] = False
    errors = tuple(name for name, passed in checks.items() if not passed)
    return FailedStressValidationRecord(
        **_record_metadata(case),
        **_record_evidence_digests(solve_result, after_scene),
        expected_outcome=case.expected_outcome,
        actual_outcome=solve_result.status,
        quality=solve_result.quality,
        reason=solve_result.reason,
        certificate=_safe_certificate_record(solve_result),
        checks=checks,
        errors=errors,
    )


def rebuild_stress_validation_record(
    case: StressCase,
    solve_result: CertifiedSolveResult,
    after_scene: Scene | None,
) -> StressValidationRecord:
    """Rebuild one Task 7 record deterministically without invoking a solver."""
    try:
        if isinstance(case.oracle, SatStressOracle):
            position = solve_result.subject_position
            if (
                position is None
                or after_scene is None
                or not all(
                    math.isfinite(value)
                    for value in (position.x, position.y, position.z)
                )
            ):
                raise StressValidationError(
                    f"{case.case_id}: solver did not retain a finite replay",
                    failed_checks=("solver_success",),
                    checks={"solver_success": False},
                )
            return validate_stress_sat(case, solve_result, after_scene)
        return validate_stress_unsat(case, solve_result)
    except StressValidationError as failure:
        return _failed_record(case, solve_result, failure, after_scene)


def _stress_solver_config() -> CertifiedSolverConfig:
    return CertifiedSolverConfig(
        optimality_tolerance=1e-6,
        numeric_tolerance=_NUMERIC_ERROR_BOUND_M,
        target_interior_margin=5e-7,
        initial_disk_segments=128,
        max_disk_segments=8192,
        timeout_seconds=5.0,
    )


def _evaluate_stress_cases(
    cases: tuple[StressCase, ...],
    solver: CertifiedSpatialCFSolver,
) -> tuple[tuple[StressValidationRecord, ...], tuple[StressValidationEvidence, ...]]:
    """Evaluate an already-closed case sequence through an injected solver."""
    motion = AnalyticMotionModel()
    records: list[StressValidationRecord] = []
    evidence: list[StressValidationEvidence] = []

    for case in cases:
        solve_result = solver.solve(case.scene, case.intervention)
        after: Scene | None = None
        try:
            if isinstance(case.oracle, SatStressOracle):
                position = solve_result.subject_position
                if position is None or not all(
                    math.isfinite(value)
                    for value in (position.x, position.y, position.z)
                ):
                    raise StressValidationError(
                        f"{case.case_id}: solver did not return a finite position",
                        failed_checks=("solver_success",),
                        checks={"solver_success": False},
                    )
                after = motion.with_object_xy(
                    case.scene,
                    case.intervention.subject_id,
                    position.x,
                    position.y,
                )
                record: StressValidationRecord = validate_stress_sat(
                    case,
                    solve_result,
                    after,
                )
            else:
                record = validate_stress_unsat(case, solve_result)
        except CandidateProjectionError:
            failure = StressValidationError(
                f"{case.case_id}: analytic replay failed",
                failed_checks=("solver_success",),
                checks={"solver_success": False},
            )
            record = _failed_record(case, solve_result, failure, after)
        except StressValidationError as failure:
            record = _failed_record(case, solve_result, failure, after)
        records.append(record)
        evidence.append(
            StressValidationEvidence(
                case=case,
                solve_result=solve_result,
                after_scene=after,
            )
        )
    return tuple(records), tuple(evidence)


def run_stress_suite(
    profile: StressProfileName,
    case_id: str | None = None,
) -> tuple[StressValidationReport, tuple[StressValidationEvidence, ...]]:
    """Run the requested stress cases and retain deterministic evidence."""
    cases = (
        (replay_stress_case(case_id),)
        if case_id is not None
        else generate_stress_cases(profile)
    )
    solver = CertifiedSpatialCFSolver(_stress_solver_config())
    records, evidence = _evaluate_stress_cases(cases, solver)

    status: Literal["PASS", "FAIL"] = (
        "PASS" if all(record.result == "PASS" for record in records) else "FAIL"
    )
    report = StressValidationReport(
        profile=profile,
        requested_case_id=case_id,
        status=status,
        seeds=tuple(dict.fromkeys(case.seed for case in cases)),
        cases=records,
    )
    return report, evidence
