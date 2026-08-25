from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from functools import partialmethod
from types import MappingProxyType
from typing import Any, ContextManager  # noqa: UP035
from weakref import ReferenceType

from spatialcf.adapters.ai2thor.camera import AI2ThorCameraMixin as _AI2ThorCameraMixin
from spatialcf.adapters.ai2thor.capture import (
    AI2ThorCaptureMixin as _AI2ThorCaptureMixin,
)
from spatialcf.adapters.ai2thor.conversion import (
    AI2ThorConversionMixin as _AI2ThorConversionMixin,
)
from spatialcf.adapters.ai2thor.execution import (
    AI2ThorExecutionMixin as _AI2ThorExecutionMixin,
)
from spatialcf.adapters.ai2thor.models import (
    AI2ThorNativeReturnError,
    AI2ThorProceduralScene,
    AI2ThorRuntimeError,
    AI2ThorSettlementTimeout,
    adapter_support_fact_from_native,
    ai2thor_spawn_map_from_adapter,
    applied_certified_edit_from_native,
    settled_readback_from_native,
)
from spatialcf.adapters.ai2thor.support import (
    AI2ThorSupportMixin as _AI2ThorSupportMixin,
)
from spatialcf.adapters.base import (
    AdapterActionRejected,
    AdapterOperationError,
    AdapterProceduralScene,
    AdapterReturnRejected,
    AdapterSettlementTimeout,
    AdapterSupportFact,
    AppliedCertifiedEdit,
    CameraObservationHandle,
    CaptureRequest,
    CertifiedEditApplication,
    SettledReadback,
    SourceCaptureFacts,
    SourceCaptureOptions,
    capture_source_with_scene_adapter,
    capture_support_with_environment_adapter,
)
from spatialcf.domain.scene import Scene

ControllerFactory = Callable[..., Any]


def _load_default_controller_type() -> type[Any]:
    try:
        with warnings.catch_warnings():
            # AI2-THOR 5.0.0 contains escaped spaces in diagnostic-only string
            # literals. Import it inside callers' fail-closed warning scopes
            # without weakening warnings emitted by our code or at runtime.
            warnings.filterwarnings(
                "ignore",
                message=r"invalid escape sequence '\\ '",
                category=DeprecationWarning,
            )
            from ai2thor.controller import Controller
    except ImportError as exc:
        raise RuntimeError(
            "AI2-THOR is not installed. Install spatialcf[sim] with Python 3.11."
        ) from exc
    return Controller


def _default_controller_factory(**kwargs: Any) -> Any:
    return _load_default_controller_type()(**kwargs)


class AI2ThorAdapter(
    _AI2ThorConversionMixin,
    _AI2ThorSupportMixin,
    _AI2ThorCameraMixin,
    _AI2ThorCaptureMixin,
    _AI2ThorExecutionMixin,
):
    def __init__(
        self,
        scene_names: list[str],
        width: int,
        height: int,
        seed: int,
        *,
        controller_factory: ControllerFactory | None = None,
        allow_source_pose_drift: bool = False,
        procedural_scenes: Mapping[str, AI2ThorProceduralScene | AdapterProceduralScene]
        | None = None,
    ) -> None:
        if type(scene_names) is not list or any(
            type(name) is not str for name in scene_names
        ):
            raise ValueError("scene_names must be an exact string list")
        if type(width) is not int or type(height) is not int:
            raise ValueError("render dimensions must be exact integers")
        if type(seed) is not int:
            raise ValueError("seed must be an exact integer")
        if type(allow_source_pose_drift) is not bool:
            raise ValueError("allow_source_pose_drift must be an exact boolean")
        if not scene_names:
            raise ValueError("at least one scene is required")
        if len(set(scene_names)) != len(scene_names):
            raise ValueError("scene names must be unique")
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        if procedural_scenes is None:
            procedural_by_alias: dict[
                str, AI2ThorProceduralScene | AdapterProceduralScene
            ] = {}
        else:
            if not isinstance(procedural_scenes, Mapping):
                raise ValueError("procedural_scenes must be a mapping")
            procedural_by_alias = {}
            for alias, source in procedural_scenes.items():
                if type(alias) is not str:
                    raise ValueError("procedural_scenes keys must be exact strings")
                if type(source) not in (AI2ThorProceduralScene, AdapterProceduralScene):
                    raise ValueError(
                        "procedural_scenes values must be exact "
                        "AI2ThorProceduralScene or AdapterProceduralScene"
                    )
                procedural_by_alias[alias] = source
            if not set(procedural_by_alias).issubset(scene_names):
                raise ValueError("procedural_scenes keys must be a scene_names subset")
        self.scene_names = list(scene_names)
        self.scene_name = self.scene_names[0]
        self.width = width
        self.height = height
        self.seed = seed
        self.allow_source_pose_drift = allow_source_pose_drift
        self.procedural_scenes: Mapping[
            str, AI2ThorProceduralScene | AdapterProceduralScene
        ] = MappingProxyType(dict(procedural_by_alias))
        self.controller: Any | None = None
        self._controller_factory = controller_factory or _default_controller_factory
        self._event: Any | None = None
        self._latest_event: Any | None = None
        self._stopped = False
        self._current_scene: Scene | None = None
        self._pending_protocol_applied: AppliedCertifiedEdit | None = None
        self._pending_protocol_scene: Scene | None = None
        self._pending_protocol_event: Any | None = None
        self._camera_states: dict[
            tuple[str, tuple[float, ...], tuple[float, ...]], dict[str, Any]
        ] = {}
        self._native_rotations: dict[tuple[str, str], dict[str, float]] = {}
        self._isolated_controller_refs: list[ReferenceType[Any]] = []
        self._protocol_camera_handles: dict[
            str, tuple[CameraObservationHandle, ContextManager[Scene]]
        ] = {}
        self._protocol_camera_resumed: set[str] = set()
        self._protocol_camera_token_sequence = 0


def _clear_pending_protocol_application(adapter: AI2ThorAdapter) -> None:
    adapter._pending_protocol_applied = None
    adapter._pending_protocol_scene = None
    adapter._pending_protocol_event = None


def _normalized_protocol_error(
    adapter: AI2ThorAdapter,
    error: Exception,
) -> AdapterOperationError:
    """Classify raw controller failures before they cross the protocol edge."""
    if isinstance(error, AdapterSettlementTimeout):
        return error
    origin = error.__cause__ if isinstance(error.__cause__, Exception) else error
    if isinstance(origin, AI2ThorNativeReturnError):
        return AdapterReturnRejected(
            f"{type(origin).__name__}:{' '.join(str(origin).split())}"
        )
    event = adapter._latest_event
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("lastActionSuccess") is False:
        message = metadata.get("errorMessage")
        if type(message) is str and message.strip():
            return AdapterActionRejected(message)
    return AdapterOperationError(str(error))


def _capture_source_protocol(
    adapter: AI2ThorAdapter,
    request: CaptureRequest,
):
    try:
        return capture_source_with_scene_adapter(adapter, request)
    except (
        AI2ThorNativeReturnError,
        AI2ThorRuntimeError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        raise _normalized_protocol_error(adapter, error) from error


def _observe_source_protocol(
    adapter: AI2ThorAdapter,
    source,
    *,
    options: SourceCaptureOptions,
    settle: bool,
) -> SourceCaptureFacts:
    try:
        return _AI2ThorCaptureMixin.observe_source(
            adapter,
            source,
            options=options,
            settle=settle,
        )
    except AdapterOperationError as error:
        raise _normalized_protocol_error(adapter, error) from error


def _current_adapter_support_facts(
    adapter: AI2ThorAdapter,
    facts: SourceCaptureFacts,
) -> tuple[AdapterSupportFact, ...]:
    current = tuple(
        adapter_support_fact_from_native(item)
        for item in adapter.native_support_facts(facts.scene)
    )
    by_object_id = {item.object_id: item for item in current}
    canonical_object_ids = tuple(item.object_id for item in facts.scene.objects)
    if len(by_object_id) != len(current) or set(by_object_id) != set(
        canonical_object_ids
    ):
        raise AdapterOperationError("current support fact roster changed")
    return tuple(by_object_id[object_id] for object_id in canonical_object_ids)


def _capture_spawn_maps_protocol(
    adapter: AI2ThorAdapter,
    facts: SourceCaptureFacts,
    *,
    subject_object_ids: tuple[str, ...],
):
    try:
        current_support_facts = _current_adapter_support_facts(adapter, facts)
        if current_support_facts != facts.support_facts:
            raise AdapterOperationError(
                "capture support facts changed before spawn-map query"
            )
        return _AI2ThorSupportMixin.capture_spawn_maps(
            adapter,
            facts,
            subject_object_ids=subject_object_ids,
        )
    except AdapterOperationError as error:
        raise _normalized_protocol_error(adapter, error) from error
    except (
        AI2ThorNativeReturnError,
        AI2ThorRuntimeError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        raise _normalized_protocol_error(adapter, error) from error


def _apply_certified_edit_observed(
    adapter: AI2ThorAdapter,
    application: CertifiedEditApplication,
) -> AppliedCertifiedEdit:
    if adapter._pending_protocol_applied is not None:
        raise AdapterOperationError("pending protocol application must settle first")
    _clear_pending_protocol_application(adapter)
    try:
        source = application.source.scene
        subject = source.object_by_id(application.edit.subject_id)
        native = adapter.apply_receptacle_endpoint_observed(
            source,
            ai2thor_spawn_map_from_adapter(application.spawn_map),
            x=subject.position.x + application.edit.translation_xy_m.x,
            y=subject.position.y + application.edit.translation_xy_m.y,
        )
        applied = applied_certified_edit_from_native(native, application=application)
    except AI2ThorSettlementTimeout as error:
        raise AdapterSettlementTimeout(str(error)) from error
    except (
        AI2ThorNativeReturnError,
        AI2ThorRuntimeError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        raise _normalized_protocol_error(adapter, error) from error
    observed_scene = adapter._current_scene
    observed_event = adapter._latest_event
    if (
        type(observed_scene) is not Scene
        or observed_event is None
        or adapter._event is not observed_event
    ):
        raise AdapterOperationError(
            "certified edit did not retain an observed adapter state"
        )
    adapter._pending_protocol_applied = applied
    adapter._pending_protocol_scene = observed_scene
    adapter._pending_protocol_event = observed_event
    return applied


def _settle_readback_observed(
    adapter: AI2ThorAdapter,
    applied: AppliedCertifiedEdit,
) -> SettledReadback:
    pending_scene = adapter._pending_protocol_scene
    pending_event = adapter._pending_protocol_event
    if (
        adapter._pending_protocol_applied is not applied
        or type(pending_scene) is not Scene
        or pending_event is None
        or adapter._current_scene is not pending_scene
        or adapter._event is not pending_event
        or adapter._latest_event is not pending_event
    ):
        _clear_pending_protocol_application(adapter)
        raise AdapterOperationError(
            "pending protocol application is missing, stale, or consumed"
        )
    _clear_pending_protocol_application(adapter)
    try:
        settlement = adapter.settle_scene_observed(
            pending_scene,
            max_pass_steps=applied.application.max_settlement_steps,
        )
        return settled_readback_from_native(settlement, applied=applied)
    except AI2ThorSettlementTimeout as error:
        raise AdapterSettlementTimeout(str(error)) from error
    except (
        AI2ThorNativeReturnError,
        AI2ThorRuntimeError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        raise _normalized_protocol_error(adapter, error) from error


AI2ThorAdapter.capture_source = _capture_source_protocol
AI2ThorAdapter.capture_support = partialmethod(capture_support_with_environment_adapter)
AI2ThorAdapter.observe_source = _observe_source_protocol
AI2ThorAdapter.capture_spawn_maps = _capture_spawn_maps_protocol
AI2ThorAdapter.apply_certified_edit = _apply_certified_edit_observed
AI2ThorAdapter.settle_readback = _settle_readback_observed
