"""Upright compiler visibility; explicit pure implementation owner."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from spatialcf.core._internal.kernels.projected_visibility import (
    FixedCardinalProjectionBoxV3,
    FixedCardinalVisibilityPolicyV3,
)

from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    FixedCardinalBoxV3,
)

from spatialcf.core.problem import (
    UprightCameraContextV2_9,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    Quaternion,
)

from spatialcf.domain.definitions import (
    FiniteOrderedTupleValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.geometry import (
    GeometryRoleV2,
)

from spatialcf.domain.scene import (
    CanonicalScene,
    PinholeCamera,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _bridge_fraction,
    _rotate_cardinal_xy_components,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _known_exact_source_values,
)

from spatialcf.core._internal.upright_se2.evaluation_data import (
    UprightSE2CardinalVisibilityEvaluationInput,
    UprightSE2ContinuousVisibilityEvaluationInput,
)

from spatialcf.core._internal.upright_se2.evaluation_geometry import (
    _bridge_fixed_box,
    _bridge_world_box_parts,
)

from spatialcf.core._internal.upright_se2.solve_context import (
    _bridge_policy_fields,
    _bridge_policy_id,
    _bridge_policy_real,
    _bridge_policy_symbol,
)


def _bridge_visibility_bound_records(
    bundle: upright.UprightSE2ExecutablePolicyBundle,
) -> tuple[dict[str, TypedValue], ...]:
    fields = _bridge_policy_fields(bundle, "visibility")
    value = fields.get("observation_bounds")
    if (
        type(value) is not TypedValue
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bounds/1.0"
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
    ):
        raise ValueError(
            "cardinal bridge visibility roster is not an exact typed tuple"
        )
    records: list[dict[str, TypedValue]] = []
    expected_names = {
        "observation_id",
        "camera_id",
        "object_id",
        "metric_definition_id",
        "metric_definition_version",
        "comparator",
        "boundary_policy",
        "threshold",
        "tolerance",
    }
    for item in value.payload.items:
        if (
            item.value_schema_ref
            != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
            or type(item.payload) is not RecordValue
        ):
            raise ValueError("cardinal bridge visibility row has the wrong type")
        record = {field.name: field.value for field in item.payload.fields}
        if len(record) != len(item.payload.fields) or set(record) != expected_names:
            raise ValueError("cardinal bridge visibility row is incomplete")
        records.append(record)
    identifiers = tuple(
        _bridge_policy_id(record, "observation_id") for record in records
    )
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or identifiers != tuple(sorted(identifiers, key=canonical_json_bytes))
    ):
        raise ValueError("cardinal bridge visibility rows are not canonical")
    return tuple(records)


def _bridge_projection_box(
    geometry: object,
    objects: dict[str, object],
    *,
    subject_id: str,
    quarter_turns_ccw: int,
    pivot_xy: tuple[Fraction, Fraction],
) -> FixedCardinalProjectionBoxV3:
    center_x, center_y, center_z, half_x, half_y, half_z, _ = _bridge_world_box_parts(
        geometry,
        objects,
    )
    if geometry.owner_object_id == subject_id:
        relative_x = center_x - pivot_xy[0]
        relative_y = center_y - pivot_xy[1]
        rotated_x, rotated_y = _rotate_cardinal_xy_components(
            relative_x,
            relative_y,
            quarter_turns_ccw,
        )
        center_x = pivot_xy[0] + rotated_x
        center_y = pivot_xy[1] + rotated_y
        if quarter_turns_ccw % 2:
            half_x, half_y = half_y, half_x
    return FixedCardinalProjectionBoxV3(
        box_id=geometry.geometry_id,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
    )


def _bridge_camera_context(camera: object) -> UprightCameraContextV2_9:
    rotation = camera.world_to_camera.rotation
    if type(rotation) is not Quaternion or (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("cardinal bridge camera must use identity orientation")
    return UprightCameraContextV2_9(
        camera_id=camera.camera_id,
        width_px=camera.width_px,
        height_px=camera.height_px,
        intrinsics=tuple(
            _bridge_fraction(value, label=f"camera {camera.camera_id} intrinsic")
            for value in camera.intrinsics_row_major
        ),
        near_clip_m=_bridge_fraction(
            camera.near_clip_m,
            label=f"camera {camera.camera_id} near clip",
        ),
        far_clip_m=_bridge_fraction(
            camera.far_clip_m,
            label=f"camera {camera.camera_id} far clip",
        ),
        translation_xyz=tuple(
            _bridge_fraction(value, label=f"camera {camera.camera_id} translation")
            for value in (
                camera.world_to_camera.translation.x,
                camera.world_to_camera.translation.y,
                camera.world_to_camera.translation.z,
            )
        ),
        sine=(Fraction(), Fraction()),
        cosine=(Fraction(1), Fraction(1)),
    )


def _bridge_visibility_inputs(
    scene: CanonicalScene,
    *,
    camera: PinholeCamera,
    objects: dict[str, object],
    subject_id: str,
    quarter_turns_ccw: int,
    pivot_xy: tuple[Fraction, Fraction],
    cell: ClosedXYCellV3,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    resource_cap: int,
) -> tuple[UprightSE2CardinalVisibilityEvaluationInput, ...]:
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    visual_by_object: dict[str, object] = {}
    for geometry in geometries:
        if geometry.role is not GeometryRoleV2.VISUAL:
            continue
        if (
            geometry.owner_object_id is None
            or geometry.owner_object_id in visual_by_object
        ):
            raise ValueError("visual geometry roster must have one owner-bound box")
        visual_by_object[geometry.owner_object_id] = geometry
    moving_geometry = visual_by_object.get(subject_id)
    if moving_geometry is None:
        raise ValueError("cardinal bridge subject has no exact visual geometry")
    projection_by_object = {
        object_id: _bridge_projection_box(
            geometry,
            objects,
            subject_id=subject_id,
            quarter_turns_ccw=quarter_turns_ccw,
            pivot_xy=pivot_xy,
        )
        for object_id, geometry in visual_by_object.items()
    }
    occluders = tuple(sorted(projection_by_object.values(), key=lambda box: box.box_id))
    required_occluder_ids = tuple(box.box_id for box in occluders)
    if len(set(required_occluder_ids)) != len(required_occluder_ids):
        raise ValueError("cardinal bridge visual box IDs are not unique")
    observations = _known_exact_source_values(
        scene.baseline_observations,
        "baseline observations",
    )
    observation_by_id = {
        observation.observation_id: observation for observation in observations
    }
    if len(observation_by_id) != len(observations):
        raise ValueError("cardinal bridge observation roster is not unique")
    bound_records = _bridge_visibility_bound_records(policy_bundle)
    inputs: list[UprightSE2CardinalVisibilityEvaluationInput] = []
    for bound in bound_records:
        observation_id = _bridge_policy_id(bound, "observation_id")
        observation = observation_by_id.get(observation_id)
        if observation is None:
            raise ValueError("visibility policy row does not bind a source observation")
        metric = (
            _bridge_policy_symbol(bound, "metric_definition_id"),
            _bridge_policy_id(bound, "metric_definition_version"),
        )
        if metric != ("visibility:image-area-fraction", "definition:1"):
            raise ValueError("unsupported registered visibility metric family")
        if (
            observation.camera_id != camera.camera_id
            or observation.camera_id != _bridge_policy_id(bound, "camera_id")
            or observation.object_id != _bridge_policy_id(bound, "object_id")
            or (observation.metric_definition_id, observation.metric_definition_version)
            != metric
            or _bridge_policy_symbol(bound, "comparator") != "GEQ"
            or _bridge_policy_symbol(bound, "boundary_policy") != "CLOSED"
        ):
            raise ValueError(
                "visibility policy row does not close its source observation"
            )
        observed_box = projection_by_object.get(observation.object_id)
        if observed_box is None:
            raise ValueError("visibility source geometry is incomplete")
        inputs.append(
            UprightSE2CardinalVisibilityEvaluationInput(
                observation_id=observation_id,
                context=_bridge_camera_context(camera),
                cell=cell.canonical_bounds,
                subject=observed_box,
                moving_subject_id=moving_geometry.geometry_id,
                occluders=occluders,
                required_occluder_ids=required_occluder_ids,
                policy=FixedCardinalVisibilityPolicyV3(
                    metric_definition_id=metric[0],
                    metric_definition_version=metric[1],
                    metric_threshold=_bridge_policy_real(bound, "threshold"),
                    metric_tolerance=_bridge_policy_real(bound, "tolerance"),
                    metric_comparator="GEQ",
                    metric_boundary="CLOSED",
                    atomic_step_limit=resource_cap,
                ),
            )
        )
    if tuple(sorted(observation_by_id, key=canonical_json_bytes)) != tuple(
        _bridge_policy_id(record, "observation_id") for record in bound_records
    ):
        raise ValueError("visibility policy does not cover the complete source roster")
    return tuple(inputs)


def _bridge_continuous_visibility_inputs(
    scene: CanonicalScene,
    *,
    camera: PinholeCamera,
    objects: dict[str, object],
    subject_id: str,
    visibility_subject_box_id: str,
    policy_bundle: upright.UprightSE2ExecutablePolicyBundle,
    resource_cap: int,
) -> tuple[UprightSE2ContinuousVisibilityEvaluationInput, ...]:
    """Bind the one V4 continuous visibility obligation from exact source facts.

    Task 6.1's compound cell seam accepts one caller-supplied visibility DTO
    whose subject ID must equal its primary collision body.  The compiler keeps
    the source visual geometry values and records its original ID, while using
    that collision ID solely as the retained DTO join key.  This is a typed
    source-to-owner mapping, never a backend geometry rewrite.
    """

    geometries = _known_exact_source_values(
        scene.geometry_instances,
        "geometry instances",
    )
    visual_by_object: dict[str, object] = {}
    for geometry in geometries:
        if geometry.role is not GeometryRoleV2.VISUAL:
            continue
        if (
            geometry.owner_object_id is None
            or geometry.owner_object_id in visual_by_object
        ):
            raise ValueError("visual geometry roster must have one owner-bound box")
        visual_by_object[geometry.owner_object_id] = geometry
    moving_geometry = visual_by_object.get(subject_id)
    if moving_geometry is None:
        raise ValueError("continuous bridge subject has no exact visual geometry")
    source_subject = _bridge_fixed_box(moving_geometry, objects)
    subject = FixedCardinalBoxV3(
        box_id=visibility_subject_box_id,
        center_x=source_subject.center_x,
        center_y=source_subject.center_y,
        center_z=source_subject.center_z,
        half_x=source_subject.half_x,
        half_y=source_subject.half_y,
        half_z=source_subject.half_z,
    )
    projection_by_object = {
        object_id: (
            subject if object_id == subject_id else _bridge_fixed_box(geometry, objects)
        )
        for object_id, geometry in visual_by_object.items()
    }
    occluders = tuple(sorted(projection_by_object.values(), key=lambda box: box.box_id))
    required_occluder_ids = tuple(box.box_id for box in occluders)
    if len(set(required_occluder_ids)) != len(required_occluder_ids):
        raise ValueError("continuous bridge visual box IDs are not unique")
    observations = _known_exact_source_values(
        scene.baseline_observations,
        "baseline observations",
    )
    observation_by_id = {
        observation.observation_id: observation for observation in observations
    }
    if len(observation_by_id) != len(observations):
        raise ValueError("continuous bridge observation roster is not unique")
    bound_records = _bridge_visibility_bound_records(policy_bundle)
    if len(bound_records) != 1:
        raise ValueError(
            "unsupported continuous V4 visibility conjunction requires one observation"
        )
    bound = bound_records[0]
    observation_id = _bridge_policy_id(bound, "observation_id")
    observation = observation_by_id.get(observation_id)
    if observation is None:
        raise ValueError("visibility policy row does not bind a source observation")
    metric = (
        _bridge_policy_symbol(bound, "metric_definition_id"),
        _bridge_policy_id(bound, "metric_definition_version"),
    )
    if metric != ("visibility:image-area-fraction", "definition:1"):
        raise ValueError("unsupported registered continuous visibility metric family")
    if (
        observation.camera_id != camera.camera_id
        or observation.camera_id != _bridge_policy_id(bound, "camera_id")
        or observation.object_id != subject_id
        or observation.object_id != _bridge_policy_id(bound, "object_id")
        or (observation.metric_definition_id, observation.metric_definition_version)
        != metric
        or _bridge_policy_symbol(bound, "comparator") != "GEQ"
        or _bridge_policy_symbol(bound, "boundary_policy") != "CLOSED"
    ):
        raise ValueError(
            "unsupported continuous visibility policy does not bind the moving subject"
        )
    return (
        UprightSE2ContinuousVisibilityEvaluationInput(
            observation_id=observation_id,
            source_visual_box_id=source_subject.box_id,
            context=_bridge_camera_context(camera),
            subject=subject,
            moving_subject_id=visibility_subject_box_id,
            occluders=occluders,
            required_occluder_ids=required_occluder_ids,
            policy=FixedCardinalVisibilityPolicyV3(
                metric_definition_id=metric[0],
                metric_definition_version=metric[1],
                metric_threshold=_bridge_policy_real(bound, "threshold"),
                metric_tolerance=_bridge_policy_real(bound, "tolerance"),
                metric_comparator="GEQ",
                metric_boundary="CLOSED",
                atomic_step_limit=resource_cap,
            ),
        ),
    )


# Preserve supported public type/function and pickle lookup.
_bridge_visibility_bound_records.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_projection_box.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_camera_context.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_visibility_inputs.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_continuous_visibility_inputs.__module__ = "spatialcf.core.upright_se2_compiler"
