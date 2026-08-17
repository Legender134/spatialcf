"""Adapter package with lazy legacy exports.

Keeping these imports lazy lets Canonical v2 fact contracts load without also
loading the legacy Scene model, NumPy, Pillow, or a platform runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialcf.adapters.base import RenderedAssets, SceneAdapter
    from spatialcf.adapters.json_scene import JsonSceneAdapter

__all__ = ["JsonSceneAdapter", "RenderedAssets", "SceneAdapter"]


def __getattr__(name: str) -> Any:
    if name == "JsonSceneAdapter":
        from spatialcf.adapters.json_scene import JsonSceneAdapter

        globals()[name] = JsonSceneAdapter
        return JsonSceneAdapter
    if name in {"RenderedAssets", "SceneAdapter"}:
        from spatialcf.adapters.base import RenderedAssets, SceneAdapter

        values = {
            "RenderedAssets": RenderedAssets,
            "SceneAdapter": SceneAdapter,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
