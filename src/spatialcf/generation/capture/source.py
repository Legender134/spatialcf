"""Current-only source capture orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from spatialcf.adapters.ai2thor import AI2ThorAdapter, AI2ThorProceduralScene
from spatialcf.domain.enums import Relation
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2
from spatialcf.generation.capture.compiler import compile_roster
from spatialcf.generation.capture.models import (
    CompetitionNativeSourceCaptureOutcomeV2_9,
    CompetitionNativeSourceRefV2_9,
    RosterCompilation,
    RosterPolicy,
)
from spatialcf.generation.capture.plan import (
    CapturePlan,
    CompetitionNativeProceduralDatasetLocatorV2_9,
)

_PROCTHOR_DATASET_ID = "allenai/procthor-10k"
_PROCTHOR_DATASET_NAME = "procthor-10k"
_PROCTHOR_LOADER_ID = "prior"
_PROCTHOR_LOADER_VERSION = "1.0.3"
_EXPECTED_SOURCE_LIFECYCLE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


def load_prior_dataset(name: str, revision: str) -> object:
    try:
        import prior
    except ImportError as error:
        raise RuntimeError("ProcTHOR capture requires spatialcf[procthor]") from error
    try:
        return prior.load_dataset(name, revision=revision)
    except MemoryError:
        raise
    except Exception as error:
        raise RuntimeError("ProcTHOR dataset loader failed") from error


def _dataset_split(dataset: object, split: str):
    if isinstance(dataset, Mapping):
        try:
            values = dataset[split]
        except KeyError as error:
            raise ValueError("ProcTHOR dataset split is absent") from error
    else:
        try:
            values = getattr(dataset, split)
        except AttributeError as error:
            raise ValueError("ProcTHOR dataset split is absent") from error
    if (
        isinstance(values, (str, bytes))
        or not callable(getattr(values, "__len__", None))
        or not callable(getattr(values, "__getitem__", None))
    ):
        raise TypeError("ProcTHOR dataset split must be indexable")
    return values


def _resolve_procedural_source(
    locator: CompetitionNativeProceduralDatasetLocatorV2_9,
    datasets: dict[tuple[str, str], object],
    dataset_loader: Callable[[str, str], object],
) -> AI2ThorProceduralScene:
    if (
        locator.dataset_id != _PROCTHOR_DATASET_ID
        or locator.loader_id != _PROCTHOR_LOADER_ID
        or locator.loader_version != _PROCTHOR_LOADER_VERSION
    ):
        raise ValueError("unsupported ProcTHOR source loader identity")
    key = (_PROCTHOR_DATASET_NAME, locator.revision)
    if key not in datasets:
        datasets[key] = dataset_loader(*key)
    values = _dataset_split(datasets[key], locator.split)
    try:
        house = values[locator.index]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("ProcTHOR source locator does not exist") from error
    if type(house) is not dict:
        raise ValueError("ProcTHOR source house must be an exact dict")
    scene = AI2ThorProceduralScene.create(
        dataset_id=locator.dataset_id,
        revision=locator.revision,
        split=locator.split,
        index=locator.index,
        source_loader_id=locator.loader_id,
        source_loader_version=locator.loader_version,
        house=house,
    )
    if scene.house_sha256 != locator.source_sha256:
        raise ValueError("ProcTHOR source content digest changed")
    return scene


def _rejected_source(
    source: CompetitionNativeSourceRefV2_9,
    reason: str,
) -> CompetitionNativeSourceCaptureOutcomeV2_9:
    return CompetitionNativeSourceCaptureOutcomeV2_9(
        source=source,
        status="rejected",
        capture=None,
        reasons=(reason,),
    )


def _apply_resource_caps(
    policy: RosterPolicy,
    outcomes: tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...],
) -> tuple[CompetitionNativeSourceCaptureOutcomeV2_9, ...]:
    bounded = []
    candidate_count = 0
    for outcome in outcomes:
        capture = outcome.capture
        if outcome.status == "rejected" or capture is None:
            bounded.append(outcome)
            continue
        object_count = len(capture.scene.objects)
        source_candidates = object_count * max(0, object_count - 1) * len(Relation)
        if object_count > policy.max_objects_per_scene:
            bounded.append(
                _rejected_source(
                    outcome.source,
                    "dataset_capture:object_cap_exceeded",
                )
            )
        elif candidate_count + source_candidates > policy.max_candidates_total:
            bounded.append(
                _rejected_source(
                    outcome.source,
                    "dataset_capture:candidate_cap_exceeded",
                )
            )
        else:
            bounded.append(outcome)
            candidate_count += source_candidates
    return tuple(bounded)


def _capture_source_with_adapter(
    source: CompetitionNativeSourceRefV2_9,
    procedural: Mapping[str, AI2ThorProceduralScene] | None,
    plan: CapturePlan,
    adapter_factory: Callable[..., AI2ThorAdapter],
):
    from spatialcf.generation._internal.source_observation import (
        capture_source_observation,
    )

    manager = adapter_factory(
        [source.scene_id],
        width=plan.roster_policy.width,
        height=plan.roster_policy.height,
        seed=plan.roster_policy.seed,
        procedural_scenes=procedural,
    )
    try:
        adapter = manager.__enter__()
    except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
        return (
            _rejected_source(source, "dataset_capture:adapter_lifecycle_failed"),
            None,
            None,
        )
    try:
        result = capture_source_observation(
            adapter,
            source=source,
            settings=plan.capture_settings,
            camera_policy=plan.roster_policy.camera_policy,
        )
    except BaseException as error:
        try:
            manager.__exit__(type(error), error, error.__traceback__)
        except MemoryError:
            raise
        except Exception as cleanup_error:  # noqa: BLE001
            error.add_note(f"AI2-THOR cleanup also failed: {cleanup_error}")
        raise
    try:
        manager.__exit__(None, None, None)
    except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
        return (
            _rejected_source(source, "dataset_capture:adapter_lifecycle_failed"),
            None,
            None,
        )
    return result.outcome, result.surface_evidence, result.camera_evidence


def capture_dataset(
    plan: CapturePlan,
    *,
    adapter_factory: Callable[..., AI2ThorAdapter] = AI2ThorAdapter,
    dataset_loader: Callable[[str, str], object] = load_prior_dataset,
) -> RosterCompilation:
    """Capture every frozen source once, then compile the current roster once."""

    if type(plan) is not CapturePlan:
        raise TypeError("dataset capture plan must be exact")
    checked = CapturePlan.model_validate(
        plan.model_dump(mode="python", warnings="error"),
        strict=True,
    )
    datasets: dict[tuple[str, str], object] = {}
    outcomes: list[CompetitionNativeSourceCaptureOutcomeV2_9] = []
    surface_evidence = []
    camera_evidence = []
    for source, locator in zip(
        checked.roster_policy.sources,
        checked.source_locators,
        strict=True,
    ):
        procedural: dict[str, AI2ThorProceduralScene] = {}
        if isinstance(locator, CompetitionNativeProceduralDatasetLocatorV2_9):
            try:
                procedural[source.scene_id] = _resolve_procedural_source(
                    locator,
                    datasets,
                    dataset_loader,
                )
            except _EXPECTED_SOURCE_LIFECYCLE_ERRORS:
                outcomes.append(
                    _rejected_source(
                        source,
                        "dataset_capture:source_resolution_failed",
                    )
                )
                continue
        outcome, source_surface, source_camera = _capture_source_with_adapter(
            source,
            procedural or None,
            checked,
            adapter_factory,
        )
        outcomes.append(
            CompetitionNativeSourceCaptureOutcomeV2_9.model_validate_json(
                canonical_json_bytes_v2(outcome),
                strict=True,
            )
        )
        if source_surface is not None:
            surface_evidence.append(source_surface)
        if source_camera is not None:
            camera_evidence.append(source_camera)
    bounded = _apply_resource_caps(checked.roster_policy, tuple(outcomes))
    accepted_source_ids = {
        item.source.source_id for item in bounded if item.status == "accepted"
    }
    bounded_surface = tuple(
        item for item in surface_evidence if item.source_id in accepted_source_ids
    )
    bounded_camera = tuple(
        item for item in camera_evidence if item.source_id in accepted_source_ids
    )
    return compile_roster(
        checked.roster_policy,
        bounded,
        bounded_surface,
        bounded_camera,
    )


__all__ = ("capture_dataset", "load_prior_dataset")
