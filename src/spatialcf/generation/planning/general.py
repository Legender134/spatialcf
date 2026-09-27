"""Static source-to-request construction and complete batch preflight."""

from dataclasses import dataclass

from spatialcf.core.rigid_se3_backend import RigidSE3Backend
from spatialcf.core.rigid_se3_compiler import build_rigid_se3_request
from spatialcf.core.semantic_place_backend import SemanticPlaceBackend
from spatialcf.core.semantic_place_compiler import build_semantic_place_request
from spatialcf.domain.general_dataset import (
    GeneralDatasetInput,
    GeneralFrozenCandidate,
    GeneralFrozenCatalog,
    PlacementSnapshot,
    PlacementTask,
    RigidSnapshot,
    RigidTask,
)
from spatialcf.domain.predicates import BeforePrecondition, NotFormula
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.verification.split import assign_split


@dataclass(frozen=True, slots=True)
class GeneralCatalogPlan:
    input: GeneralDatasetInput
    catalog: GeneralFrozenCatalog
    compilations: tuple[object | None, ...]


def backend_for(profile: str):
    if profile == "spatialcf/semantic_place@1":
        return SemanticPlaceBackend()
    if profile == "spatialcf/rigid_se3_multi@1":
        return RigidSE3Backend()
    raise ValueError("unsupported general dataset profile")


def _request(source, task):
    if type(source) is PlacementSnapshot and type(task) is PlacementTask:
        return build_semantic_place_request(
            scene=source.scene,
            subject_id=task.subject_id,
            operation=task.operation,
            target_ids=task.target_ids,
            x=task.x,
            y=task.y,
            z=task.z,
            cavities=source.cavities,
            support_margin_m=task.support_margin_m,
            lateral_margin_m=task.lateral_margin_m,
            top_margin_m=task.top_margin_m,
            limits=task.limits,
            backend_enabled=task.backend_enabled,
            problem_id="problem:general-dataset",
        )
    if type(source) is RigidSnapshot and type(task) is RigidTask:
        negative = BeforePrecondition(
            formula=NotFormula(formula=task.after_goal.formula)
        )
        preconditions = {
            canonical_json_bytes(value): value
            for value in (*task.before_preconditions, negative)
        }
        return build_rigid_se3_request(
            scene=source.scene,
            source=source.facts,
            domain=task.domain,
            objective=task.objective,
            after_goal=task.after_goal,
            before_preconditions=tuple(
                preconditions[key] for key in sorted(preconditions)
            ),
            preservation_invariants=task.preservation_invariants,
            witness_hints=task.witness_hints,
            limits=task.limits,
            backend_enabled=task.backend_enabled,
            problem_id="problem:general-dataset",
        ).request
    raise ValueError("task and source profile differ")


def build_general_catalog(
    value: GeneralDatasetInput, *, runtime_sha256: str
) -> GeneralCatalogPlan:
    """Freeze every request/selection/compilation before allowing a solve."""
    if type(value) is not GeneralDatasetInput:
        raise TypeError("expected exact GeneralDatasetInput")
    value = GeneralDatasetInput.model_validate(value, strict=True)
    normalized = GeneralDatasetInput.seal(
        sources=tuple(
            sorted(value.sources, key=lambda row: canonical_json_bytes(row.source_id))
        ),
        tasks=tuple(
            sorted(
                value.tasks,
                key=lambda row: (
                    canonical_json_bytes(row.source_id),
                    canonical_json_bytes(row.candidate_id),
                ),
            )
        ),
        policy=value.policy,
    )
    sources = {source.source_id: source for source in normalized.sources}
    candidates = []
    compilations = []
    seen = set()
    for task in normalized.tasks:
        source = sources[task.source_id]
        request = _request(source, task)
        key = (source.source_id, request.solve_request_sha256)
        if key in seen:
            raise ValueError("duplicate source/request candidate")
        seen.add(key)
        profile = request.semantic_problem.action_space_profile_ref
        backend = backend_for(profile)
        selection = backend.select(request)
        compilation = (
            backend.compile(request)
            if selection.selection_disposition == "SELECTED"
            else None
        )
        candidates.append(
            GeneralFrozenCandidate.seal(
                candidate_id=task.candidate_id,
                source_id=source.source_id,
                source_snapshot_sha256=source.source_snapshot_sha256,
                task_sha256=task.task_sha256,
                profile=profile,
                split=assign_split(source.scene.scene_id),
                request=request,
                selection=selection,
            )
        )
        compilations.append(compilation)
    return GeneralCatalogPlan(
        normalized,
        GeneralFrozenCatalog.seal(
            input_sha256=normalized.input_sha256,
            runtime_provenance_sha256=runtime_sha256,
            candidates=tuple(candidates),
        ),
        tuple(compilations),
    )
