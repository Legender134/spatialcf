from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spatialcf.adapters.base import RenderedAssets
from spatialcf.domain.models import OBB, Scene, Vec3


class JsonSceneAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list_scene_ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

    def load_scene(self, scene_id: str) -> Scene:
        return Scene.model_validate_json(
            (self.root / f"{scene_id}.json").read_text(encoding="utf-8")
        )

    def with_object_xy(self, scene: Scene, object_id: str, x: float, y: float) -> Scene:
        scene.object_by_id(object_id)
        objects = []
        for obj in scene.objects:
            if obj.object_id != object_id:
                objects.append(obj)
                continue
            dx, dy = x - obj.position.x, y - obj.position.y
            position = Vec3(x=x, y=y, z=obj.position.z)
            obb = OBB(
                center=Vec3(
                    x=obj.obb.center.x + dx,
                    y=obj.obb.center.y + dy,
                    z=obj.obb.center.z,
                ),
                extent=obj.obb.extent,
                rotation=obj.obb.rotation,
            )
            views = {}
            for camera_id, view in obj.views.items():
                views[camera_id] = view.model_copy(update={
                    "bbox": view.bbox.model_copy(update={
                        "xmin": view.bbox.xmin + 100 * dx,
                        "xmax": view.bbox.xmax + 100 * dx,
                    }),
                    "camera_depth": view.camera_depth + dy,
                })
            objects.append(
                obj.model_copy(update={"position": position, "obb": obb, "views": views})
            )
        return scene.model_copy(update={"objects": tuple(objects)})

    def render_assets(
        self,
        scene: Scene,
        camera_id: str,
        destination_root: Path,
        stem: str,
    ) -> RenderedAssets:
        camera = scene.camera_by_id(camera_id)
        destination_root.mkdir(parents=True, exist_ok=True)
        rgb_path = destination_root / f"{stem}.png"
        depth_path = destination_root / f"{stem}-depth.npy"
        instance_path = destination_root / f"{stem}-instance.png"
        pointcloud_path = destination_root / f"{stem}.ply"
        rgb = Image.new("RGB", (camera.width, camera.height), "white")
        instance = Image.new("RGB", (camera.width, camera.height), "black")
        rgb_draw, instance_draw = ImageDraw.Draw(rgb), ImageDraw.Draw(instance)
        for index, obj in enumerate(scene.objects, start=1):
            view = obj.views.get(camera_id)
            if view is None:
                continue
            box = (view.bbox.xmin, view.bbox.ymin, view.bbox.xmax, view.bbox.ymax)
            rgb_draw.rectangle(box, outline=(40, 90, 180), width=3)
            instance_draw.rectangle(
                box,
                fill=(index % 255, (index * 17) % 255, (index * 31) % 255),
            )
        rgb.save(rgb_path)
        instance.save(instance_path)
        np.save(depth_path, np.full((camera.height, camera.width), 2.0, dtype=np.float32))
        points = [obj.obb.center for obj in scene.objects]
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        pointcloud_path.write_text(
            header + "".join(f"{p.x} {p.y} {p.z}\n" for p in points),
            encoding="ascii",
        )
        return RenderedAssets(rgb_path, depth_path, instance_path, pointcloud_path)
