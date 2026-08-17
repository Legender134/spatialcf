"""Calibrated analytic camera projection for continuous XY subject motion."""

from dataclasses import dataclass
import math

from spatialcf.domain.models import BBox2D, Camera, ObjectView, Scene, Vec3


class CandidateProjectionError(ValueError):
    """A proposed XY point cannot be projected into a valid subject view."""


def _camera_coordinate(camera: Camera, row: int, point: Vec3) -> float:
    """Return one homogeneous world-to-camera row evaluated at ``point``."""
    start = row * 4
    matrix = camera.world_to_camera
    return (
        matrix[start] * point.x
        + matrix[start + 1] * point.y
        + matrix[start + 2] * point.z
        + matrix[start + 3]
    )


@dataclass(frozen=True)
class ProjectionCalibration:
    """Affine camera coordinates and residuals fitted to one observed view."""

    camera: Camera
    subject_z: float
    camera_x_coefficients: tuple[float, float, float]
    camera_y_coefficients: tuple[float, float, float]
    depth_coefficients: tuple[float, float, float]
    horizontal_residual: float
    vertical_residual: float
    depth_residual: float

    def camera_x(self, x: float, y: float) -> float:
        return _evaluate_affine(self.camera_x_coefficients, x, y)

    def camera_y(self, x: float, y: float) -> float:
        return _evaluate_affine(self.camera_y_coefficients, x, y)

    def depth(self, x: float, y: float) -> float:
        return _evaluate_affine(self.depth_coefficients, x, y)

    def projected_center(self, x: float, y: float) -> tuple[float, float]:
        depth = self.depth(x, y)
        if depth <= 0.0:
            raise CandidateProjectionError("positive camera depth is required")
        fx, _, cx, _, fy, cy, _, _, _ = self.camera.intrinsics
        return (
            fx * self.camera_x(x, y) / depth + cx,
            cy - fy * self.camera_y(x, y) / depth,
        )

    def calibrated_depth(self, x: float, y: float) -> float:
        return self.depth(x, y) + self.depth_residual


def _evaluate_affine(coefficients: tuple[float, float, float], x: float, y: float) -> float:
    x_coefficient, y_coefficient, constant = coefficients
    return x_coefficient * x + y_coefficient * y + constant


def _camera_affine_coefficients(camera: Camera, row: int, z: float) -> tuple[float, float, float]:
    start = row * 4
    matrix = camera.world_to_camera
    return (
        matrix[start],
        matrix[start + 1],
        matrix[start + 2] * z + matrix[start + 3],
    )


class AnalyticMotionModel:
    """Create independently-verifiable candidate scenes from XY subject motion."""

    def calibration(self, scene: Scene, object_id: str, camera_id: str) -> ProjectionCalibration:
        subject = scene.object_by_id(object_id)
        camera = scene.camera_by_id(camera_id)
        observed = subject.views.get(camera_id)
        if observed is None:
            raise ValueError("subject has no view for camera")

        point = subject.position
        camera_x = _camera_coordinate(camera, 0, point)
        camera_y = _camera_coordinate(camera, 1, point)
        depth = _camera_coordinate(camera, 2, point)
        if depth <= 0.0:
            raise CandidateProjectionError("positive camera depth is required")
        fx, _, cx, _, fy, cy, _, _, _ = camera.intrinsics
        projected_x = fx * camera_x / depth + cx
        projected_y = cy - fy * camera_y / depth
        observed_center_y = (observed.bbox.ymin + observed.bbox.ymax) / 2.0
        camera_x_coefficients = _camera_affine_coefficients(camera, 0, point.z)
        camera_y_coefficients = _camera_affine_coefficients(camera, 1, point.z)
        depth_coefficients = _camera_affine_coefficients(camera, 2, point.z)
        horizontal_residual = observed.bbox.center_x - projected_x
        vertical_residual = observed_center_y - projected_y
        depth_residual = observed.camera_depth - depth
        values = (
            point.x,
            point.y,
            point.z,
            camera_x,
            camera_y,
            depth,
            projected_x,
            projected_y,
            observed.bbox.center_x,
            observed_center_y,
            observed.camera_depth,
            *camera_x_coefficients,
            *camera_y_coefficients,
            *depth_coefficients,
            horizontal_residual,
            vertical_residual,
            depth_residual,
        )
        if not all(math.isfinite(value) for value in values):
            raise CandidateProjectionError(
                "projection calibration values must be finite"
            )
        return ProjectionCalibration(
            camera=camera,
            subject_z=point.z,
            camera_x_coefficients=camera_x_coefficients,
            camera_y_coefficients=camera_y_coefficients,
            depth_coefficients=depth_coefficients,
            horizontal_residual=horizontal_residual,
            vertical_residual=vertical_residual,
            depth_residual=depth_residual,
        )

    def projected_view(
        self, scene: Scene, object_id: str, camera_id: str, x: float, y: float
    ) -> ObjectView:
        self._require_finite_xy(x, y)
        subject = scene.object_by_id(object_id)
        observed = subject.views.get(camera_id)
        if observed is None:
            raise ValueError("subject has no view for camera")
        calibration = self.calibration(scene, object_id, camera_id)
        projected_x, projected_y = calibration.projected_center(x, y)
        depth = calibration.calibrated_depth(x, y)
        if not math.isfinite(depth) or depth <= 0.0:
            raise CandidateProjectionError("positive camera depth is required")

        bbox = _bbox_at_center(observed.bbox, projected_x + calibration.horizontal_residual,
                               projected_y + calibration.vertical_residual)
        return observed.model_copy(update={
            "bbox": bbox,
            "camera_depth": depth,
            "truncated_fraction": _truncated_fraction(bbox, calibration.camera),
        })

    def with_object_xy(self, scene: Scene, object_id: str, x: float, y: float) -> Scene:
        self._require_finite_xy(x, y)
        subject = scene.object_by_id(object_id)
        delta_x = x - subject.position.x
        delta_y = y - subject.position.y
        new_position = subject.position.model_copy(update={"x": x, "y": y})
        new_center = subject.obb.center.model_copy(update={
            "x": subject.obb.center.x + delta_x,
            "y": subject.obb.center.y + delta_y,
        })
        new_views = {
            camera_id: self.projected_view(scene, object_id, camera_id, x, y)
            for camera_id in subject.views
        }
        moved = subject.model_copy(update={
            "position": new_position,
            "obb": subject.obb.model_copy(update={"center": new_center}),
            "views": new_views,
        })
        return scene.model_copy(update={
            "objects": tuple(moved if obj.object_id == object_id else obj for obj in scene.objects),
        })

    @staticmethod
    def _require_finite_xy(x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("candidate coordinates must be finite")


def _bbox_at_center(bbox: BBox2D, center_x: float, center_y: float) -> BBox2D:
    half_width = (bbox.xmax - bbox.xmin) / 2.0
    half_height = (bbox.ymax - bbox.ymin) / 2.0
    return BBox2D(
        xmin=center_x - half_width,
        ymin=center_y - half_height,
        xmax=center_x + half_width,
        ymax=center_y + half_height,
    )


def _truncated_fraction(bbox: BBox2D, camera: Camera) -> float:
    full_area = bbox.area
    if full_area <= 0.0:
        raise CandidateProjectionError("projected bbox must have positive area")
    clipped_width = max(0.0, min(bbox.xmax, camera.width) - max(bbox.xmin, 0.0))
    clipped_height = max(0.0, min(bbox.ymax, camera.height) - max(bbox.ymin, 0.0))
    return 1.0 - clipped_width * clipped_height / full_area
