"""Deterministic coarse-to-fine search over a verified feasible region."""

import math
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Generic, TypeVar

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from spatialcf.adapters.base import XYSceneTransformer
from spatialcf.domain.enums import QualityTier, SolverStatus
from spatialcf.domain.models import InterventionSpec, Scene, Vec3
from spatialcf.solver.analytic_motion import CandidateProjectionError
from spatialcf.solver.execution import (
    CandidateExecution,
    CandidateExecutionStatus,
    CandidateExecutor,
    ExecutionResidual,
)
from spatialcf.solver.feasible import FeasibleRegionBuilder
from spatialcf.solver.objective import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    ObjectiveBreakdown,
    ObjectiveWeights,
    score_candidate,
    score_minimum_cost_candidate,
)
from spatialcf.verification.verifier import VerificationResult, Verifier

MIN_SEARCH_STEP = 1e-9
ExecutionEvidenceT = TypeVar("ExecutionEvidenceT")


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 20260723
    grid_step: float = 0.10
    refine_steps: tuple[float, ...] = (0.05, 0.01)
    max_candidates: int = 10_000
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an exact integer")
        if type(self.grid_step) is not float:
            raise ValueError("grid_step must be an exact float")
        if (
            type(self.refine_steps) is not tuple
            or any(type(step) is not float for step in self.refine_steps)
        ):
            raise ValueError("refine_steps must be a tuple of exact floats")
        if type(self.max_candidates) is not int:
            raise ValueError("max_candidates must be an exact integer")
        if not math.isfinite(self.grid_step):
            raise ValueError("grid_step must be finite")
        if self.grid_step <= 0.0:
            raise ValueError("grid_step must be positive")
        if self.grid_step < MIN_SEARCH_STEP:
            raise ValueError("grid_step must be at least 1e-9 metres")
        if any(not math.isfinite(step) for step in self.refine_steps):
            raise ValueError("refine_steps must be finite")
        if any(step <= 0.0 for step in self.refine_steps):
            raise ValueError("refine_steps must be positive")
        if any(step < MIN_SEARCH_STEP for step in self.refine_steps):
            raise ValueError("refine_steps must be at least 1e-9 metres")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds < 0.0
        ):
            raise ValueError(
                "timeout_seconds must be None or a finite non-negative value"
            )


@dataclass(frozen=True)
class SolveResult:
    status: SolverStatus
    subject_position: Vec3 | None
    score: ObjectiveBreakdown | None
    quality: QualityTier
    evaluated_candidates: int
    reason: str | None
    relation_damage_count: int = 0
    relation_damage_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedCandidateAttempt:
    """Compact audit record for one executed command from the proposal region."""

    commanded_position: Vec3
    analytic_verification: VerificationResult
    analytic_score: ObjectiveBreakdown | None
    execution_status: CandidateExecutionStatus
    execution_errors: tuple[str, ...]
    execution_residuals: tuple[ExecutionResidual, ...]
    observed_before_position: Vec3 | None
    observed_position: Vec3 | None
    observed_score: ObjectiveBreakdown | None
    verification: VerificationResult | None


@dataclass(frozen=True)
class GroundedCandidate(Generic[ExecutionEvidenceT]):
    """The selected command, its actual observation, and immutable evidence."""

    attempt_index: int
    subject_id: str
    commanded_scene: Scene
    observed_before_scene: Scene
    observed_scene: Scene
    analytic_score: ObjectiveBreakdown | None
    observed_score: ObjectiveBreakdown
    verification: VerificationResult
    execution: CandidateExecution[ExecutionEvidenceT]

    @property
    def commanded_position(self) -> Vec3:
        return self.commanded_scene.object_by_id(self.subject_id).position


@dataclass(frozen=True)
class GroundedSolveResult(Generic[ExecutionEvidenceT]):
    """Minimum observed cost over the finite executed episode-command pairs."""

    status: SolverStatus
    candidate: GroundedCandidate[ExecutionEvidenceT] | None
    evaluated_candidates: int
    executed_candidates: int
    budget_exhausted: bool
    attempts: tuple[GroundedCandidateAttempt, ...]
    reason: str | None

    @property
    def selected_attempt_index(self) -> int | None:
        if self.candidate is None:
            return None
        return self.candidate.attempt_index

    @property
    def commanded_position(self) -> Vec3 | None:
        if self.candidate is None:
            return None
        return self.candidate.commanded_position

    @property
    def observed_position(self) -> Vec3 | None:
        if self.candidate is None:
            return None
        return self.candidate.observed_scene.object_by_id(
            self.candidate.subject_id
        ).position

    @property
    def observed_before_position(self) -> Vec3 | None:
        if self.candidate is None:
            return None
        return self.candidate.observed_before_scene.object_by_id(
            self.candidate.subject_id
        ).position

    @property
    def score(self) -> ObjectiveBreakdown | None:
        return None if self.candidate is None else self.candidate.observed_score

    @property
    def quality(self) -> QualityTier:
        if self.candidate is None:
            return QualityTier.REJECTED
        return self.candidate.verification.quality

    @property
    def relation_damage_count(self) -> int:
        if self.candidate is None:
            return 0
        return self.candidate.verification.relation_damage_count

    @property
    def relation_damage_items(self) -> tuple[str, ...]:
        if self.candidate is None:
            return ()
        return self.candidate.verification.relation_damage_items


class SpatialCFSolver:
    """Search feasible X/Y translations and retain only verifier-approved edits."""

    _LOCAL_OFFSETS: tuple[tuple[int, int], ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def __init__(
        self,
        adapter: XYSceneTransformer,
        config: SearchConfig,
        verifier: Verifier | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.verifier = verifier or Verifier()
        self.regions = FeasibleRegionBuilder()

    def _grid(self, bounds: tuple[float, float, float, float], step: float):
        minx, miny, maxx, maxy = bounds
        decimal_step = Decimal(str(step))

        def indices(minimum: float, maximum: float) -> range:
            minimum_index = int((
                Decimal(str(minimum)) / decimal_step
            ).to_integral_value(rounding=ROUND_CEILING))
            maximum_index = int((
                Decimal(str(maximum)) / decimal_step
            ).to_integral_value(rounding=ROUND_FLOOR))
            return range(minimum_index, maximum_index + 1)

        y_indices = indices(miny, maxy)
        for x_index in indices(minx, maxx):
            x = float(decimal_step * x_index)
            for y_index in y_indices:
                y = float(decimal_step * y_index)
                yield x, y

    def _timed_out(self, started: float | None) -> bool:
        return (
            started is not None
            and self.config.timeout_seconds is not None
            and time.monotonic() - started >= self.config.timeout_seconds
        )

    def _initial_candidates(
        self,
        region: BaseGeometry,
        scene: Scene,
        spec: InterventionSpec,
    ) -> tuple[tuple[float, float], ...]:
        return ()

    def _result(
        self,
        status: SolverStatus,
        evaluated: int,
        reason: str,
    ) -> SolveResult:
        return SolveResult(status, None, None, QualityTier.REJECTED, evaluated, reason)

    def _verify_candidate(
        self,
        scene: Scene,
        candidate: Scene,
        spec: InterventionSpec,
    ) -> VerificationResult:
        return self.verifier.verify(scene, candidate, spec)

    def _score_candidate(
        self,
        scene: Scene,
        candidate: Scene,
        spec: InterventionSpec,
        verification: VerificationResult,
    ) -> ObjectiveBreakdown:
        return score_candidate(
            scene,
            candidate,
            spec,
            verification.leakage_count,
            self.verifier.engine,
        )

    @staticmethod
    def _candidate_key(
        score: ObjectiveBreakdown,
        x: float,
        y: float,
    ) -> tuple[float, float, float, float]:
        return score.normalized_translation, score.total, x, y

    def solve(self, scene: Scene, spec: InterventionSpec) -> SolveResult:
        if scene.children_by_support().get(spec.subject_id):
            return self._result(
                SolverStatus.INVALID_SCENE,
                0,
                "subject_has_supported_objects",
            )
        started = (
            time.monotonic()
            if self.config.timeout_seconds is not None
            else None
        )
        region = self.regions.build(scene, spec)
        if self._timed_out(started):
            return self._result(SolverStatus.TIMEOUT, 0, "timeout")
        if region.is_empty:
            return self._result(SolverStatus.UNSATISFIABLE, 0, "empty_region")

        before = scene.object_by_id(spec.subject_id)
        best: tuple[
            tuple[float, float, float, float],
            Vec3,
            ObjectiveBreakdown,
            VerificationResult,
        ] | None = None
        evaluated = 0

        def evaluate(x: float, y: float) -> None:
            nonlocal best, evaluated
            evaluated += 1
            try:
                candidate = self.adapter.with_object_xy(
                    scene, spec.subject_id, x, y
                )
            except CandidateProjectionError:
                return
            verification = self._verify_candidate(scene, candidate, spec)
            if verification.status is not SolverStatus.SUCCESS:
                return
            score = self._score_candidate(scene, candidate, spec, verification)
            key = self._candidate_key(score, x, y)
            position = Vec3(x=x, y=y, z=before.position.z)
            if best is None or key < best[0]:
                best = (key, position, score, verification)

        seeded_positions: set[tuple[float, float]] = set()
        for x, y in self._initial_candidates(region, scene, spec):
            if evaluated >= self.config.max_candidates:
                break
            if self._timed_out(started):
                return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
            if region.covers(Point(x, y)):
                seeded_positions.add((x, y))
                evaluate(x, y)
            if self._timed_out(started):
                return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")

        for x, y in self._grid(region.bounds, self.config.grid_step):
            if evaluated >= self.config.max_candidates:
                break
            if self._timed_out(started):
                return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
            if (x, y) not in seeded_positions and region.covers(Point(x, y)):
                evaluate(x, y)
            if self._timed_out(started):
                return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")

        if best is None:
            if self._timed_out(started):
                return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
            return self._result(
                SolverStatus.UNSATISFIABLE,
                evaluated,
                "no_verified_candidate",
            )

        for step in self.config.refine_steps:
            _, center, _, _ = best
            decimal_step = Decimal(str(step))
            local_positions: set[tuple[float, float]] = set()
            for dx, dy in self._LOCAL_OFFSETS:
                if evaluated >= self.config.max_candidates:
                    break
                if self._timed_out(started):
                    return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
                x = float(Decimal(str(center.x)) + decimal_step * dx)
                y = float(Decimal(str(center.y)) + decimal_step * dy)
                if (x, y) in local_positions:
                    continue
                local_positions.add((x, y))
                if (
                    (x, y) not in seeded_positions
                    and region.covers(Point(x, y))
                ):
                    evaluate(x, y)
                if self._timed_out(started):
                    return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
            if evaluated >= self.config.max_candidates:
                break

        if self._timed_out(started):
            return self._result(SolverStatus.TIMEOUT, evaluated, "timeout")
        verification = best[3]
        return SolveResult(
            SolverStatus.SUCCESS,
            best[1],
            best[2],
            verification.quality,
            evaluated,
            None,
            verification.relation_damage_count,
            verification.relation_damage_items,
        )


class MinimumCostSpatialCFSolver(SpatialCFSolver):
    """Minimize weighted cost over hard-valid, fixed-camera XY interventions.

    Unlike :class:`SpatialCFSolver`, collateral relation changes are measured and
    priced rather than treated as hard infeasibility. Structural validity,
    physical constraints, query visibility, and the requested relation flip stay
    hard requirements.
    """

    def __init__(
        self,
        adapter: XYSceneTransformer,
        config: SearchConfig,
        verifier: Verifier | None = None,
        *,
        objective_weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    ) -> None:
        super().__init__(adapter, config, verifier)
        if type(objective_weights) is not ObjectiveWeights:
            raise ValueError("objective_weights must be ObjectiveWeights")
        self.objective_weights = objective_weights

    def _verify_candidate(
        self,
        scene: Scene,
        candidate: Scene,
        spec: InterventionSpec,
    ) -> VerificationResult:
        return self.verifier.verify_minimum_cost(scene, candidate, spec)

    def _initial_candidates(
        self,
        region: BaseGeometry,
        scene: Scene,
        spec: InterventionSpec,
    ) -> tuple[tuple[float, float], ...]:
        subject = scene.object_by_id(spec.subject_id)
        origin = Point(subject.position.x, subject.position.y)
        components = tuple(getattr(region, "geoms", (region,)))
        components = tuple(sorted(components, key=lambda item: item.bounds))
        points = [nearest_points(origin, region)[1]]
        points.extend(component.representative_point() for component in components)
        unique: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for point in points:
            candidate = float(point.x), float(point.y)
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return tuple(unique)

    def _score_candidate(
        self,
        scene: Scene,
        candidate: Scene,
        spec: InterventionSpec,
        verification: VerificationResult,
    ) -> ObjectiveBreakdown:
        return score_minimum_cost_candidate(
            scene,
            candidate,
            spec,
            verification.relation_damage_count,
            self.verifier.engine,
            self.objective_weights,
        )

    @staticmethod
    def _candidate_key(
        score: ObjectiveBreakdown,
        x: float,
        y: float,
    ) -> tuple[float, float, float, float]:
        return score.total, score.normalized_translation, x, y

    @staticmethod
    def _grounded_position_only_command(
        scene: Scene,
        subject_id: str,
        x: float,
        y: float,
    ) -> Scene:
        """Move only canonical XY geometry when analytic projection is unavailable."""
        subject = scene.object_by_id(subject_id)
        delta_x = x - subject.position.x
        delta_y = y - subject.position.y
        moved = subject.model_copy(
            update={
                "position": subject.position.model_copy(update={"x": x, "y": y}),
                "obb": subject.obb.model_copy(
                    update={
                        "center": subject.obb.center.model_copy(
                            update={
                                "x": subject.obb.center.x + delta_x,
                                "y": subject.obb.center.y + delta_y,
                            }
                        )
                    }
                ),
            }
        )
        return scene.model_copy(
            update={
                "objects": tuple(
                    moved if obj.object_id == subject_id else obj
                    for obj in scene.objects
                )
            }
        )

    def _episode_baseline_errors(
        self,
        nominal: Scene,
        observed: Scene,
        spec: InterventionSpec,
    ) -> tuple[str, ...]:
        """Require one fresh episode to preserve nominal source semantics.

        Physics engines may settle continuous geometry and rendered view
        fractions slightly differently across isolated episodes.  Those values
        belong to the episode and are compared only to its own after-scene.
        Candidate costs remain comparable only when fixed camera, structure,
        support, and the complete discrete relation graph still match the
        nominal planning scene.
        """
        errors: list[str] = []
        for field in (
            "scene_id",
            "source",
            "coordinate_system",
            "generation_seed",
            "pinned_object_ids",
        ):
            if getattr(nominal, field) != getattr(observed, field):
                errors.append(f"episode_{field}_changed")
        if nominal.room_polygon_xy != observed.room_polygon_xy:
            errors.append("episode_room_changed")
        if nominal.cameras != observed.cameras:
            errors.append("episode_camera_changed")
        if nominal.collision_obstacles != observed.collision_obstacles:
            errors.append("episode_collision_obstacles_changed")
        if nominal.subject_position_regions != observed.subject_position_regions:
            errors.append("episode_subject_position_regions_changed")

        nominal_ids = {obj.object_id for obj in nominal.objects}
        observed_ids = {obj.object_id for obj in observed.objects}
        if nominal_ids != observed_ids:
            errors.append("episode_object_set_changed")
            return tuple(sorted(set(errors)))

        for expected in nominal.objects:
            current = observed.object_by_id(expected.object_id)
            if any(
                getattr(expected, field) != getattr(current, field)
                for field in (
                    "name",
                    "category",
                    "movable",
                    "request_eligible",
                    "support_object_id",
                )
            ):
                errors.append(
                    f"episode_object_structure_changed:{expected.object_id}"
                )

        try:
            nominal.camera_by_id(spec.camera_id)
            observed.camera_by_id(spec.camera_id)
            observed.object_by_id(spec.subject_id)
            observed.object_by_id(spec.reference_id)
            source = self.verifier.engine.observe(
                observed,
                spec.subject_id,
                spec.reference_id,
                spec.relation_before,
                spec.camera_id,
            )
            relation_graph_changed = any(
                self.verifier.engine.pair_labels(
                    nominal,
                    first.object_id,
                    second.object_id,
                    spec.camera_id,
                )
                != self.verifier.engine.pair_labels(
                    observed,
                    first.object_id,
                    second.object_id,
                    spec.camera_id,
                )
                for first in nominal.objects
                for second in nominal.objects
                if first.object_id != second.object_id
            )
        except (KeyError, TypeError, ValueError):
            errors.append("episode_relation_graph_invalid")
        else:
            if relation_graph_changed:
                errors.append("episode_relation_graph_changed")
            if not source.satisfied:
                errors.append("episode_source_relation_not_satisfied")
        return tuple(sorted(set(errors)))

    def solve_grounded(
        self,
        scene: Scene,
        spec: InterventionSpec,
        executor: CandidateExecutor[ExecutionEvidenceT],
    ) -> GroundedSolveResult[ExecutionEvidenceT]:
        """Search with an external execution model inside candidate evaluation.

        Every unique command in the analytic proposal region is executed.  An
        analytic verifier failure is recorded but does not veto execution: the
        returned canonical episode before/after pair is independently verified
        and scored by the core.  Search therefore selects the minimum *observed*
        cost over the executed finite candidate set; it never stops at the first
        native success and does not claim a continuous platform optimum or a
        deterministic command-level optimum across stochastic episodes.
        """

        attempts: list[GroundedCandidateAttempt] = []
        evaluated = 0

        def result(
            status: SolverStatus,
            reason: str | None,
            candidate: GroundedCandidate[ExecutionEvidenceT] | None = None,
        ) -> GroundedSolveResult[ExecutionEvidenceT]:
            return GroundedSolveResult(
                status=status,
                candidate=candidate,
                evaluated_candidates=evaluated,
                executed_candidates=len(attempts),
                budget_exhausted=evaluated >= self.config.max_candidates,
                attempts=tuple(attempts),
                reason=reason,
            )

        if scene.children_by_support().get(spec.subject_id):
            return result(SolverStatus.INVALID_SCENE, "subject_has_supported_objects")
        started = (
            time.monotonic()
            if self.config.timeout_seconds is not None
            else None
        )
        region = self.regions.build(scene, spec)
        if self._timed_out(started):
            return result(SolverStatus.TIMEOUT, "timeout")
        if region.is_empty:
            return result(SolverStatus.UNSATISFIABLE, "empty_region")

        best: tuple[
            tuple[float, float, float, float, float, float],
            GroundedCandidate[ExecutionEvidenceT],
        ] | None = None
        proposal_best: tuple[
            tuple[float, float, float],
            Vec3,
        ] | None = None
        fatal_error: str | None = None
        seen_positions: set[tuple[float, float]] = set()

        def evaluate(x: float, y: float) -> None:
            nonlocal best, evaluated, fatal_error, proposal_best
            position_key = (x, y)
            if position_key in seen_positions:
                return
            seen_positions.add(position_key)
            evaluated += 1
            try:
                commanded = self.adapter.with_object_xy(
                    scene,
                    spec.subject_id,
                    x,
                    y,
                )
            except CandidateProjectionError:
                commanded = self._grounded_position_only_command(
                    scene,
                    spec.subject_id,
                    x,
                    y,
                )
                analytic_verification = VerificationResult(
                    SolverStatus.UNCERTIFIED,
                    QualityTier.REJECTED,
                    0,
                    (),
                    ("analytic_projection_failed",),
                )
            else:
                analytic_verification = self._verify_candidate(
                    scene,
                    commanded,
                    spec,
                )
            commanded_position = commanded.object_by_id(spec.subject_id).position
            source_position = scene.object_by_id(spec.subject_id).position
            proposal_key = (
                math.hypot(x - source_position.x, y - source_position.y),
                x,
                y,
            )
            if proposal_best is None or proposal_key < proposal_best[0]:
                proposal_best = (proposal_key, commanded_position)
            analytic_score = (
                self._score_candidate(
                    scene,
                    commanded,
                    spec,
                    analytic_verification,
                )
                if analytic_verification.status is SolverStatus.SUCCESS
                else None
            )

            try:
                execution = executor.execute_candidate(scene, commanded, spec)
                if not isinstance(execution, CandidateExecution):
                    raise TypeError("executor did not return CandidateExecution")
                if execution.commanded_scene != commanded:
                    execution = CandidateExecution.error(
                        commanded,
                        ("executor_commanded_scene_mismatch",),
                    )
            # An executor is an external-platform boundary.  Any ordinary
            # exception means its state/evidence is no longer trustworthy, so
            # convert it to fail-stop UNCERTIFIED rather than leaking it or
            # continuing with a possibly corrupted platform.  BaseException
            # (interrupts and process termination) is deliberately not caught.
            except Exception as error:  # noqa: BLE001
                execution = CandidateExecution.error(
                    commanded,
                    (f"executor_exception:{type(error).__name__}:{error}",),
                )

            episode_before = execution.observed_before_scene
            if execution.status is not CandidateExecutionStatus.ERROR:
                if episode_before is None:  # Defensive against invalid executors.
                    execution = CandidateExecution.error(
                        commanded,
                        ("episode_before_scene_missing",),
                        evidence=execution.evidence,
                    )
                else:
                    episode_errors = self._episode_baseline_errors(
                        scene,
                        episode_before,
                        spec,
                    )
                    if episode_errors:
                        execution = CandidateExecution.error(
                            commanded,
                            episode_errors,
                            evidence=execution.evidence,
                        )
            episode_before = execution.observed_before_scene
            observed = execution.observed_scene
            observed_before_position: Vec3 | None = None
            observed_position: Vec3 | None = None
            if episode_before is not None:
                try:
                    observed_before_position = episode_before.object_by_id(
                        spec.subject_id
                    ).position
                except KeyError:
                    observed_before_position = None
            if observed is not None:
                try:
                    observed_position = observed.object_by_id(
                        spec.subject_id
                    ).position
                except KeyError:
                    observed_position = None
            observed_score: ObjectiveBreakdown | None = None
            observed_verification: VerificationResult | None = None
            selected: GroundedCandidate[ExecutionEvidenceT] | None = None
            if execution.status is CandidateExecutionStatus.ERROR:
                fatal_error = "executor_error:" + ";".join(execution.errors)
            elif execution.status is CandidateExecutionStatus.OBSERVED:
                if episode_before is None or observed is None:
                    fatal_error = "executor_error:episode_observation_missing"
                else:
                    observed_verification = self._verify_candidate(
                        episode_before,
                        observed,
                        spec,
                    )
                    if observed_verification.status is SolverStatus.SUCCESS:
                        observed_score = self._score_candidate(
                            episode_before,
                            observed,
                            spec,
                            observed_verification,
                        )
                        selected = GroundedCandidate(
                            attempt_index=len(attempts),
                            subject_id=spec.subject_id,
                            commanded_scene=commanded,
                            observed_before_scene=episode_before,
                            observed_scene=observed,
                            analytic_score=analytic_score,
                            observed_score=observed_score,
                            verification=observed_verification,
                            execution=execution,
                        )

            attempts.append(
                GroundedCandidateAttempt(
                    commanded_position=commanded_position,
                    analytic_verification=analytic_verification,
                    analytic_score=analytic_score,
                    execution_status=execution.status,
                    execution_errors=execution.errors,
                    execution_residuals=execution.residuals,
                    observed_before_position=observed_before_position,
                    observed_position=observed_position,
                    observed_score=observed_score,
                    verification=observed_verification,
                )
            )
            if selected is not None and observed_position is not None:
                grounded_key = (
                    selected.observed_score.total,
                    selected.observed_score.normalized_translation,
                    x,
                    y,
                    observed_position.x,
                    observed_position.y,
                )
                if best is None or grounded_key < best[0]:
                    best = (grounded_key, selected)

        def consider(x: float, y: float) -> None:
            if (
                fatal_error is None
                and evaluated < self.config.max_candidates
                and (x, y) not in seen_positions
                and region.covers(Point(x, y))
            ):
                evaluate(x, y)

        for x, y in self._initial_candidates(region, scene, spec):
            if fatal_error is not None or evaluated >= self.config.max_candidates:
                break
            if self._timed_out(started):
                return result(SolverStatus.TIMEOUT, "timeout")
            consider(x, y)
            if fatal_error is not None:
                return result(SolverStatus.UNCERTIFIED, fatal_error)
            if self._timed_out(started):
                return result(SolverStatus.TIMEOUT, "timeout")

        for x, y in self._grid(region.bounds, self.config.grid_step):
            if fatal_error is not None or evaluated >= self.config.max_candidates:
                break
            if self._timed_out(started):
                return result(SolverStatus.TIMEOUT, "timeout")
            consider(x, y)
            if fatal_error is not None:
                return result(SolverStatus.UNCERTIFIED, fatal_error)
            if self._timed_out(started):
                return result(SolverStatus.TIMEOUT, "timeout")

        if fatal_error is not None:
            return result(SolverStatus.UNCERTIFIED, fatal_error)

        for step in self.config.refine_steps:
            if best is not None:
                center = best[1].commanded_position
            elif proposal_best is not None:
                center = proposal_best[1]
            else:
                break
            decimal_step = Decimal(str(step))
            for dx, dy in self._LOCAL_OFFSETS:
                if fatal_error is not None or evaluated >= self.config.max_candidates:
                    break
                if self._timed_out(started):
                    return result(SolverStatus.TIMEOUT, "timeout")
                x = float(Decimal(str(center.x)) + decimal_step * dx)
                y = float(Decimal(str(center.y)) + decimal_step * dy)
                consider(x, y)
                if fatal_error is not None:
                    return result(SolverStatus.UNCERTIFIED, fatal_error)
                if self._timed_out(started):
                    return result(SolverStatus.TIMEOUT, "timeout")
            if fatal_error is not None:
                return result(SolverStatus.UNCERTIFIED, fatal_error)
            if evaluated >= self.config.max_candidates:
                break

        if self._timed_out(started):
            return result(SolverStatus.TIMEOUT, "timeout")
        if best is None:
            return result(
                SolverStatus.UNCERTIFIED,
                "no_grounded_candidate_within_budget",
            )
        return result(SolverStatus.SUCCESS, None, best[1])
