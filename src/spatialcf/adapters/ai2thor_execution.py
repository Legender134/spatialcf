"""AI2-THOR implementation of the platform-neutral candidate executor.

The core solver sees only a canonical observed :class:`Scene`.  Fresh isolated
episodes, physics settlement, action failures, same-event assets, and bounded
conversion roundoff remain isolated in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spatialcf.adapters import ai2thor_validation as validation
from spatialcf.adapters.ai2thor import (
    AI2ThorAdapter,
    AI2ThorFloorEnvelope,
    AI2ThorPoseApplication,
    AI2ThorRuntimeError,
    AI2ThorRuntimeIdentity,
    AI2ThorSceneSettlement,
)
from spatialcf.domain.models import InterventionSpec, Scene, Vec3
from spatialcf.geometry.obb import inside_room, obbs_intersect_3d
from spatialcf.relations.engine import RelationEngine
from spatialcf.solver.execution import CandidateExecution, ExecutionResidual


@dataclass(frozen=True)
class AI2ThorCandidateEvidence:
    """Raw immutable evidence retained around one settled native candidate."""

    baseline_settlement: AI2ThorSceneSettlement
    immediate_application: AI2ThorPoseApplication
    settlement: AI2ThorSceneSettlement
    application: AI2ThorPoseApplication
    floor_envelope: AI2ThorFloorEnvelope
    runtime_identity: AI2ThorRuntimeIdentity


_PHYSICAL_NATIVE_ERROR_PREFIXES = (
    "native_stationary_geometry_residual_exceeded:",
    "native_stationary_rotation_residual_exceeded:",
    "native_subject_geometry_residual_exceeded",
    "native_subject_rotation_residual_exceeded",
)


def _is_candidate_physical_error(error: str) -> bool:
    return error.startswith(_PHYSICAL_NATIVE_ERROR_PREFIXES) or (
        error.startswith("native_object_field_changed:")
        and error.endswith(":support_object_id")
    )


def _clean_message(value: object) -> str:
    return " ".join(str(value).split()) or "native adapter failure"


def _position_distance(left: Vec3, right: Vec3) -> float:
    return math.dist(
        (left.x, left.y, left.z),
        (right.x, right.y, right.z),
    )


def _residual(
    name: str,
    value: float,
    limit: float,
) -> ExecutionResidual | None:
    if not math.isfinite(value):
        return None
    return ExecutionResidual(name, float(value), float(limit))


def _baseline_contract_errors(
    nominal_before: Scene,
    settlement: AI2ThorSceneSettlement,
    envelope: AI2ThorFloorEnvelope,
    limits: validation.AI2ThorValidationTolerances,
    expected_runtime: AI2ThorRuntimeIdentity,
    observed_runtime: AI2ThorRuntimeIdentity,
    spec: InterventionSpec,
) -> tuple[str, ...]:
    """Admit a fresh episode only when its task semantics are unchanged."""
    observed = settlement.observed_scene
    errors = list(
        validation.settlement_contract_errors(
            nominal_before.scene_id,
            settlement,
        )
    )
    errors.extend(
        validation.observation_contract_errors(
            settlement.observation,
            spec.camera_id,
        )
    )
    errors.extend(
        validation.floor_envelope_contract_errors(
            observed,
            envelope,
            limits,
            runtime_identity=observed_runtime,
        )
    )
    if observed_runtime != expected_runtime:
        errors.append("runtime_identity_changed")

    for field in (
        "scene_id",
        "source",
        "coordinate_system",
        "generation_seed",
        "pinned_object_ids",
    ):
        if getattr(nominal_before, field) != getattr(observed, field):
            errors.append(f"baseline_{field}_changed")
    errors.extend(
        f"baseline_{error}"
        for error in validation.camera_residual_errors(
            nominal_before,
            observed,
            limits,
        )
    )

    expected_ids = {obj.object_id for obj in nominal_before.objects}
    observed_ids = {obj.object_id for obj in observed.objects}
    if expected_ids != observed_ids:
        errors.append("baseline_object_set_changed")
        return tuple(sorted(set(errors)))
    for expected in nominal_before.objects:
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
            errors.append(f"baseline_object_structure_changed:{expected.object_id}")

    try:
        relation_diff, _ = validation.relation_graph_diff(
            nominal_before,
            observed,
            spec,
        )
        source_labels = RelationEngine().pair_labels(
            observed,
            spec.subject_id,
            spec.reference_id,
            spec.camera_id,
        )
    except (KeyError, ValueError):
        errors.append("baseline_relation_graph_invalid")
    else:
        if relation_diff:
            errors.append("baseline_relation_graph_changed")
        if spec.relation_before not in source_labels:
            errors.append("baseline_source_target_relation_changed")
    return tuple(sorted(set(errors)))


def _native_action_rejection(
    adapter: AI2ThorAdapter,
    runtime_identity: AI2ThorRuntimeIdentity,
    scene_id: str,
    error: RuntimeError,
) -> str | None:
    """Recognize an ordinary failed native command without trusting bad events."""
    if isinstance(error, AI2ThorRuntimeError):
        return None
    try:
        event = adapter.latest_native_event(scene_id)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    metadata = getattr(event, "metadata", None)
    if (
        not isinstance(metadata, dict)
        or metadata.get("lastActionSuccess") is not False
    ):
        return None
    if metadata.get("sceneName") != runtime_identity.native_scene_name:
        return None
    message = metadata.get("errorMessage")
    if type(message) is not str or not message.strip():
        return None
    clean = _clean_message(message)
    if clean != _clean_message(error):
        return None
    return clean


def _native_pose_rejection(
    adapter: AI2ThorAdapter,
    runtime_identity: AI2ThorRuntimeIdentity,
    scene_id: str,
    error: RuntimeError,
) -> str | None:
    """Recognize the adapter's bounded candidate-pose rejection contract."""
    if isinstance(error, AI2ThorRuntimeError):
        return None
    message = _clean_message(error)
    if not (
        message.startswith("object '")
        and message.endswith("' pose changed during pose application")
    ):
        return None
    try:
        event = adapter.latest_native_event(scene_id)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    metadata = getattr(event, "metadata", None)
    if (
        not isinstance(metadata, dict)
        or metadata.get("lastActionSuccess") is not True
        or metadata.get("sceneName") != runtime_identity.native_scene_name
    ):
        return None
    return message


def _did_not_settle(error: RuntimeError) -> bool:
    return _clean_message(error).startswith("AI2-THOR scene did not settle within ")


def _final_application(
    settlement: AI2ThorSceneSettlement,
    commanded: Scene,
    spec: InterventionSpec,
) -> AI2ThorPoseApplication:
    commanded_position = commanded.object_by_id(spec.subject_id).position
    observed_position = settlement.observed_scene.object_by_id(spec.subject_id).position
    return AI2ThorPoseApplication(
        commanded_scene=commanded,
        observed_scene=settlement.observed_scene,
        commanded_position=commanded_position,
        observed_position=observed_position,
        position_residual_m=_position_distance(
            commanded_position,
            observed_position,
        ),
        observation=settlement.observation,
        is_scene_at_rest=settlement.observation.is_scene_at_rest,
        subject_is_moving=False,
    )


def _validate_native_candidate(
    before: Scene,
    commanded: Scene,
    spec: InterventionSpec,
    envelope: AI2ThorFloorEnvelope,
    limits: validation.AI2ThorValidationTolerances,
    runtime_identity: AI2ThorRuntimeIdentity,
    evidence: AI2ThorCandidateEvidence,
) -> tuple[
    Scene | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[ExecutionResidual, ...],
]:
    """Return normalized IR plus contract and candidate-specific failures."""
    application = evidence.application
    contract_errors: list[str] = []
    physical_errors: list[str] = []
    application_errors, commanded_pose_residual = (
        validation.pose_application_contract_errors(commanded, spec, application)
    )
    contract_errors.extend(application_errors)
    contract_errors.extend(
        validation.settlement_contract_errors(before.scene_id, evidence.settlement)
    )
    contract_errors.extend(
        validation.floor_envelope_contract_errors(
            before,
            envelope,
            limits,
            runtime_identity=runtime_identity,
        )
    )
    if application.observation.scene != application.observed_scene:
        contract_errors.append("observation_scene_mismatch")
    contract_errors.extend(
        validation.observation_contract_errors(
            application.observation,
            spec.camera_id,
        )
    )

    (
        normalized,
        structure_errors,
        maximum_geometry_residual,
        maximum_rotation_residual,
    ) = validation.normalized_native_after(
        before,
        application.observed_scene,
        spec,
        limits,
    )
    for error in structure_errors:
        if _is_candidate_physical_error(error):
            physical_errors.append(error)
        else:
            contract_errors.append(error)

    if normalized is not None:
        try:
            raw_relation_diff, raw_leakage = validation.relation_graph_diff(
                before,
                application.observed_scene,
                spec,
            )
            normalized_relation_diff, normalized_leakage = (
                validation.relation_graph_diff(before, normalized, spec)
            )
        except (KeyError, ValueError):
            contract_errors.append("native_relation_graph_invalid")
        else:
            if raw_relation_diff != normalized_relation_diff:
                physical_errors.append("native_relation_diff_mismatch")
            if raw_leakage != normalized_leakage:
                physical_errors.append("native_leakage_count_mismatch")

    if commanded_pose_residual > limits.commanded_pose_residual_m:
        physical_errors.append("commanded_pose_residual_exceeded")
    if not application.is_scene_at_rest or not application.observation.is_scene_at_rest:
        physical_errors.append("scene_not_at_rest")
    if application.subject_is_moving:
        physical_errors.append("subject_is_moving")

    contact_residual = math.inf
    observed = application.observed_scene
    try:
        subject = observed.object_by_id(spec.subject_id)
    except KeyError:
        contract_errors.append("observed_subject_missing")
    else:
        try:
            floor = validation.floor_polygon(envelope)
        except validation.AI2ThorValidationRejected as error:
            contract_errors.extend(error.reasons)
        else:
            if not inside_room(subject.obb, floor, tolerance=0.0):
                physical_errors.append("subject_outside_floor_envelope")
        support_id = subject.support_object_id
        for obstacle in observed.objects:
            if obstacle.object_id in {subject.object_id, support_id}:
                continue
            if obbs_intersect_3d(
                subject.obb,
                obstacle.obb,
                xy_area_tolerance=limits.overlap_xy_area_tolerance,
                z_overlap_tolerance=limits.overlap_z_tolerance,
            ):
                physical_errors.append(f"native_3d_collision:{obstacle.object_id}")
        try:
            contact_residual, footprint_supported = validation.vertical_contact_residual(
                observed,
                spec,
                envelope,
            )
        except KeyError:
            physical_errors.append("observed_support_missing")
        else:
            if contact_residual > limits.vertical_contact_residual_m:
                physical_errors.append("vertical_contact_residual_exceeded")
            if not footprint_supported:
                physical_errors.append("support_footprint_invalid")

    subject_pixels = application.observation.instance_pixel_counts.get(
        spec.subject_id,
        0,
    )
    if type(subject_pixels) is not int or subject_pixels <= 0:
        physical_errors.append("subject_instance_mask_empty")

    residuals = tuple(
        item
        for item in (
            _residual(
                "commanded_pose_m",
                commanded_pose_residual,
                limits.commanded_pose_residual_m,
            ),
            _residual(
                "stationary_geometry_m",
                maximum_geometry_residual,
                limits.stationary_geometry_residual_m,
            ),
            _residual(
                "rotation_degrees",
                maximum_rotation_residual,
                limits.rotation_residual_degrees,
            ),
            _residual(
                "vertical_contact_m",
                contact_residual,
                limits.vertical_contact_residual_m,
            ),
        )
        if item is not None
    )
    return (
        normalized,
        tuple(sorted(set(contract_errors))),
        tuple(sorted(set(physical_errors))),
        residuals,
    )


class AI2ThorCandidateExecutor:
    """Execute every command from a fresh, strictly checked AI2-THOR baseline."""

    def __init__(
        self,
        adapter: AI2ThorAdapter,
        *,
        floor_envelope: AI2ThorFloorEnvelope,
        runtime_identity: AI2ThorRuntimeIdentity,
        tolerances: validation.AI2ThorValidationTolerances | None = None,
        max_settle_steps: int = 30,
    ) -> None:
        if type(floor_envelope) is not AI2ThorFloorEnvelope:
            raise ValueError("floor_envelope must be AI2ThorFloorEnvelope")
        if type(runtime_identity) is not AI2ThorRuntimeIdentity:
            raise ValueError("runtime_identity must be AI2ThorRuntimeIdentity")
        frozen = validation.AI2ThorValidationTolerances()
        if tolerances is not None and tolerances != frozen:
            raise ValueError("AI2-THOR executor requires frozen tolerances")
        if type(max_settle_steps) is not int or max_settle_steps <= 0:
            raise ValueError("max_settle_steps must be an exact positive integer")
        self.adapter = adapter
        self.floor_envelope = floor_envelope
        self.runtime_identity = runtime_identity
        self.tolerances = frozen
        self.max_settle_steps = max_settle_steps

    @staticmethod
    def _exception(
        commanded: Scene,
        stage: str,
        error: BaseException,
    ) -> CandidateExecution[AI2ThorCandidateEvidence]:
        return CandidateExecution.error(
            commanded,
            (f"{stage}_exception:{type(error).__name__}:{_clean_message(error)}",),
        )

    def execute_candidate(
        self,
        before: Scene,
        commanded: Scene,
        spec: InterventionSpec,
    ) -> CandidateExecution[AI2ThorCandidateEvidence]:
        """Run one proposal inside its own fresh controller episode."""
        if type(before) is not Scene or type(commanded) is not Scene:
            raise ValueError("before and commanded must be canonical Scene values")
        if type(spec) is not InterventionSpec:
            raise ValueError("spec must be InterventionSpec")
        if before.scene_id != commanded.scene_id:
            return CandidateExecution.error(
                commanded,
                ("command_scene_mismatch",),
            )

        stage = "baseline"
        try:
            with self.adapter.isolated_scene_observed(
                before,
                max_pass_steps=self.max_settle_steps,
            ) as episode:
                episode_adapter = episode.adapter
                baseline_settlement = episode.baseline_settlement
                if type(baseline_settlement) is not AI2ThorSceneSettlement:
                    raise TypeError("adapter settlement has an invalid type")
                observed_runtime = episode_adapter.runtime_identity()
                if type(observed_runtime) is not AI2ThorRuntimeIdentity:
                    raise TypeError("adapter runtime identity has an invalid type")
                local_before = baseline_settlement.observed_scene
                local_envelope = episode_adapter.conservative_floor_envelope(
                    local_before,
                    self.tolerances.floor_clearance_m,
                )
                if type(local_envelope) is not AI2ThorFloorEnvelope:
                    raise TypeError("adapter floor envelope has an invalid type")

                baseline_errors = _baseline_contract_errors(
                    before,
                    baseline_settlement,
                    local_envelope,
                    self.tolerances,
                    self.runtime_identity,
                    observed_runtime,
                    spec,
                )
                if baseline_errors:
                    result = CandidateExecution.error(
                        commanded,
                        tuple(
                            f"baseline_contract:{error}"
                            for error in baseline_errors
                        ),
                    )
                else:
                    stage = "candidate"
                    result = self._execute_in_episode(
                        episode_adapter,
                        local_before,
                        baseline_settlement,
                        commanded,
                        spec,
                        local_envelope,
                        observed_runtime,
                    )
                stage = "cleanup"
        except Exception as error:  # noqa: BLE001 - fail closed on child cleanup
            return self._exception(commanded, stage, error)
        return result

    def _execute_in_episode(
        self,
        adapter: AI2ThorAdapter,
        local_before: Scene,
        baseline_settlement: AI2ThorSceneSettlement,
        commanded: Scene,
        spec: InterventionSpec,
        floor_envelope: AI2ThorFloorEnvelope,
        runtime_identity: AI2ThorRuntimeIdentity,
    ) -> CandidateExecution[AI2ThorCandidateEvidence]:
        """Apply and validate one proposal against its episode-local source."""
        try:
            requested_position = commanded.object_by_id(spec.subject_id).position
            local_subject = local_before.object_by_id(spec.subject_id)
            immediate = adapter.apply_object_xy_observed(
                local_before,
                spec.subject_id,
                requested_position.x,
                requested_position.y,
            )
            if type(immediate) is not AI2ThorPoseApplication:
                raise TypeError("adapter pose application has an invalid type")
        except RuntimeError as error:
            rejection = _native_action_rejection(
                adapter,
                runtime_identity,
                commanded.scene_id,
                error,
            )
            if rejection is not None:
                return CandidateExecution.rejected(
                    commanded,
                    (f"native_action_rejected:{rejection}",),
                    observed_before_scene=local_before,
                )
            pose_rejection = _native_pose_rejection(
                adapter,
                runtime_identity,
                commanded.scene_id,
                error,
            )
            if pose_rejection is not None:
                return CandidateExecution.rejected(
                    commanded,
                    (f"native_pose_rejected:{pose_rejection}",),
                    observed_before_scene=local_before,
                )
            return self._exception(commanded, "application", error)
        except (KeyError, TypeError, ValueError) as error:
            return self._exception(commanded, "application", error)

        expected_local_position = Vec3(
            x=requested_position.x,
            y=requested_position.y,
            z=local_subject.position.z,
        )
        if (
            immediate.commanded_position != expected_local_position
            or immediate.commanded_scene.object_by_id(spec.subject_id).position
            != expected_local_position
        ):
            return CandidateExecution.error(
                commanded,
                ("commanded_position_mismatch",),
            )
        local_commanded = immediate.commanded_scene

        try:
            settlement = adapter.settle_scene_observed(
                immediate.observed_scene,
                max_pass_steps=self.max_settle_steps,
            )
            if type(settlement) is not AI2ThorSceneSettlement:
                raise TypeError("adapter settlement has an invalid type")
            observed_runtime = adapter.runtime_identity()
            if type(observed_runtime) is not AI2ThorRuntimeIdentity:
                raise TypeError("adapter runtime identity has an invalid type")
        except RuntimeError as error:
            if _did_not_settle(error):
                return CandidateExecution.rejected(
                    commanded,
                    ("post_action_not_settled",),
                    observed_before_scene=local_before,
                )
            return self._exception(commanded, "post_action", error)
        except (KeyError, TypeError, ValueError) as error:
            return self._exception(commanded, "post_action", error)
        if observed_runtime != runtime_identity:
            return CandidateExecution.error(
                commanded,
                ("post_action_runtime_identity_changed",),
            )

        try:
            application = _final_application(
                settlement,
                local_commanded,
                spec,
            )
        except KeyError as error:
            return self._exception(commanded, "post_action", error)
        evidence = AI2ThorCandidateEvidence(
            baseline_settlement=baseline_settlement,
            immediate_application=immediate,
            settlement=settlement,
            application=application,
            floor_envelope=floor_envelope,
            runtime_identity=observed_runtime,
        )
        normalized, contract_errors, physical_errors, residuals = (
            _validate_native_candidate(
                local_before,
                local_commanded,
                spec,
                floor_envelope,
                self.tolerances,
                observed_runtime,
                evidence,
            )
        )
        if contract_errors:
            return CandidateExecution.error(
                commanded,
                contract_errors,
                evidence=evidence,
            )
        if physical_errors:
            return CandidateExecution.rejected(
                commanded,
                physical_errors,
                observed_before_scene=local_before,
                observed_scene=normalized,
                residuals=residuals,
                evidence=evidence,
            )
        if normalized is None:
            return CandidateExecution.error(
                commanded,
                ("normalized_observed_scene_missing",),
                evidence=evidence,
            )
        return CandidateExecution.observed(
            commanded,
            observed_before_scene=local_before,
            observed_scene=normalized,
            residuals=residuals,
            evidence=evidence,
        )


__all__ = (
    "AI2ThorCandidateEvidence",
    "AI2ThorCandidateExecutor",
)
