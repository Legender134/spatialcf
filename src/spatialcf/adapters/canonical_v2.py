"""Cold, platform-neutral protocol for Canonical v2 scene-fact adapters.

The concrete adaptation value model is re-exported lazily so importing this
protocol does not pull the Canonical geometry stack (or legacy adapters) into a
process that only needs to declare an adapter implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from spatialcf.adapters.canonical_v2_models import CanonicalSceneAdaptationV2

__all__ = ["CanonicalSceneAdaptationV2", "CanonicalSceneFactsAdapterV2"]

NativeT_contra = TypeVar("NativeT_contra", contravariant=True)


@runtime_checkable
class CanonicalSceneFactsAdapterV2(Protocol[NativeT_contra]):
    """Structural contract for translating native facts, never request policy."""

    def adapt_scene_facts(
        self,
        native: NativeT_contra,
        /,
    ) -> CanonicalSceneAdaptationV2:
        """Return one closed Canonical scene and its independent provenance."""
        ...


def __getattr__(name: str) -> Any:
    if name == "CanonicalSceneAdaptationV2":
        from spatialcf.adapters.canonical_v2_models import CanonicalSceneAdaptationV2

        globals()[name] = CanonicalSceneAdaptationV2
        return CanonicalSceneAdaptationV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
