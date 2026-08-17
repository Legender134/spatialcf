from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from spatialcf.domain.models import Scene


@dataclass(frozen=True)
class RenderedAssets:
    rgb_path: Path
    depth_path: Path
    instance_path: Path
    pointcloud_path: Path


@runtime_checkable
class XYSceneTransformer(Protocol):
    """Create a canonical scene with one object's XY position changed."""

    def with_object_xy(
        self, scene: Scene, object_id: str, x: float, y: float
    ) -> Scene:
        pass


@runtime_checkable
class SceneAdapter(XYSceneTransformer, Protocol):
    def list_scene_ids(self) -> list[str]:
        pass

    def load_scene(self, scene_id: str) -> Scene:
        pass

    def render_assets(
        self,
        scene: Scene,
        camera_id: str,
        destination_root: Path,
        stem: str,
    ) -> RenderedAssets:
        pass
