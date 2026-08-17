"""Cold, platform-neutral protocol for Canonical 2.1 scene-fact adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from spatialcf.adapters.canonical_v2_1_models import CanonicalSceneAdaptationV2_1

__all__ = ["CanonicalSceneAdaptationV2_1", "CanonicalSceneFactsAdapterV2_1"]

NativeT_contra = TypeVar("NativeT_contra", contravariant=True)


@runtime_checkable
class CanonicalSceneFactsAdapterV2_1(Protocol[NativeT_contra]):
    """Translate native facts into the exact-cardinal 2.1 scene contract."""

    def adapt_scene_facts(
        self,
        native: NativeT_contra,
        /,
    ) -> CanonicalSceneAdaptationV2_1:
        """Return one closed Canonical scene and independent provenance."""
        ...


def __getattr__(name: str) -> Any:
    if name == "CanonicalSceneAdaptationV2_1":
        from spatialcf.adapters.canonical_v2_1_models import (
            CanonicalSceneAdaptationV2_1,
        )

        globals()[name] = CanonicalSceneAdaptationV2_1
        return CanonicalSceneAdaptationV2_1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
