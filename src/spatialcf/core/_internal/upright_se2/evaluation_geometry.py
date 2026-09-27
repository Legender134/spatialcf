"""Upright compiler evaluation geometry; explicit pure implementation owner."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from spatialcf.core._internal.kernels.upright_box import (
    FixedCardinalBoxV3,
    SupportSurfaceV3,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    Quaternion,
    UncertaintyBudgetV2,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
)

from spatialcf.domain.geometry import (
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)

from spatialcf.domain.predicates import (
    PredicateAtom,
)

from spatialcf.domain.scene import (
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
    CanonicalScene,
    PinholeCamera,
    SupportSurfaceFact,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.core._internal.upright_se2.arithmetic import (
    _bridge_fraction,
    _rotate_cardinal_xy_components,
)

from spatialcf.core._internal.upright_se2.bindings import (
    _direct_predicate_atom,
    _known_exact_source_values,
    _validate_preservation_operands,
    _validate_target_relation_operands,
)

from spatialcf.core._internal.upright_se2.constants import (
    _CARDINAL_TURN_FRACTIONS,
)


def _bridge_compiled_cell_member(
    compilation: upright.UprightSE2Compilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Return one authorized root or its proper exact-dyadic descendant.

    The compilation roster is the sole request-authorized root set.  A backend
    may name deterministic refinement cells, but cannot alter a root's
    authorization, cardinal yaw, domain, or identity by doing so.
    """

    matching = tuple(
        cell
        for cell in compilation.compiled_cells
        if canonical_json_bytes(cell) == canonical_json_bytes(compiled_cell)
    )
    if len(matching) == 1:
        return matching[0]
    if matching:
        raise ValueError("compiled cell matches more than one compilation root")

    roots = tuple(
        root
        for root in compilation.compiled_cells
        if _bridge_is_proper_exact_dyadic_descendant(compiled_cell, root)
    )
    if len(roots) != 1:
        raise ValueError(
            "compiled cell must be one exact compilation root or its unique exact-dyadic descendant"
        )
    return compiled_cell


def _bridge_continuous_compiled_cell_member(
    compilation: upright.UprightSE2ContinuousCompilation,
    compiled_cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Accept only the sole root or one exact lifted `(x,y,u)` descendant."""

    root = compilation.compiled_cells[0]
    if canonical_json_bytes(root) == canonical_json_bytes(compiled_cell):
        return root
    if not _bridge_is_proper_exact_dyadic_continuous_descendant(compiled_cell, root):
        raise ValueError(
            "continuous compiled cell must be its root or a unique exact-dyadic lifted descendant"
        )
    return compiled_cell


def _bridge_is_proper_exact_dyadic_continuous_descendant(
    cell: upright.UprightSE2CompiledCell,
    root: upright.UprightSE2CompiledCell,
) -> bool:
    """Require a strict closed descendant without changing root authorization."""

    if (
        not cell.cell_id.startswith(f"{root.cell_id}/")
        or cell.authorization_sha256 != root.authorization_sha256
        or cell.x_lower.as_fraction < root.x_lower.as_fraction
        or cell.x_upper.as_fraction > root.x_upper.as_fraction
        or cell.y_lower.as_fraction < root.y_lower.as_fraction
        or cell.y_upper.as_fraction > root.y_upper.as_fraction
        or cell.yaw_interval.lower.as_fraction < root.yaw_interval.lower.as_fraction
        or cell.yaw_interval.upper.as_fraction > root.yaw_interval.upper.as_fraction
    ):
        return False
    return (
        cell.x_lower != root.x_lower
        or cell.x_upper != root.x_upper
        or cell.y_lower != root.y_lower
        or cell.y_upper != root.y_upper
        or cell.yaw_interval.lower != root.yaw_interval.lower
        or cell.yaw_interval.upper != root.yaw_interval.upper
    )


def _bridge_is_proper_exact_dyadic_descendant(
    cell: upright.UprightSE2CompiledCell,
    root: upright.UprightSE2CompiledCell,
) -> bool:
    """Require a strict, yaw-bound XY refinement of exactly one root."""

    if (
        not cell.cell_id.startswith(f"{root.cell_id}/")
        or cell.authorization_sha256 != root.authorization_sha256
        or cell.yaw_interval != root.yaw_interval
        or cell.x_lower.as_fraction < root.x_lower.as_fraction
        or cell.x_upper.as_fraction > root.x_upper.as_fraction
        or cell.y_lower.as_fraction < root.y_lower.as_fraction
        or cell.y_upper.as_fraction > root.y_upper.as_fraction
    ):
        return False
    return (
        cell.x_lower != root.x_lower
        or cell.x_upper != root.x_upper
        or cell.y_lower != root.y_lower
        or cell.y_upper != root.y_upper
    )


def _bridge_require_replayed_compilation(
    compilation: upright.UprightSE2Compilation,
    replayed: upright.UprightSE2Compilation,
) -> None:
    """Require source replay to reproduce every normal compiler-owned root.

    A retained M2 q=0 construction adds one provenance record after normal M3
    compilation, so that optional construction root is deliberately checked by
    the source model itself rather than compared to a direct replay.
    """

    for field_name in (
        "solve_request_sha256",
        "source_solve_request",
        "closure",
        "operation",
        "endpoint_construction_recipe",
        "state_footprint",
        "grounded_obligations",
        "semantic_closure",
        "compiled_cells",
    ):
        if canonical_json_bytes(
            getattr(compilation, field_name)
        ) != canonical_json_bytes(getattr(replayed, field_name)):
            raise ValueError(
                "cardinal bridge compilation root or policy closure does not replay"
            )


def _bridge_require_replayed_continuous_compilation(
    compilation: upright.UprightSE2ContinuousCompilation,
    replayed: upright.UprightSE2ContinuousCompilation,
) -> None:
    """Require source replay to reproduce the additive continuous root exactly."""

    for field_name in (
        "solve_request_sha256",
        "source_solve_request",
        "closure",
        "operation",
        "endpoint_construction_recipe",
        "state_footprint",
        "grounded_obligations",
        "semantic_closure",
        "compiled_cells",
        "continuous_yaw_lift",
    ):
        if canonical_json_bytes(
            getattr(compilation, field_name)
        ) != canonical_json_bytes(getattr(replayed, field_name)):
            raise ValueError(
                "continuous bridge compilation root or policy closure does not replay"
            )


def _bridge_validate_cell_yaw(
    cell: upright.UprightSE2CompiledCell,
    quarter_turns_ccw: int,
) -> None:
    expected = _CARDINAL_TURN_FRACTIONS.get(quarter_turns_ccw)
    if expected is None or (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        cell.yaw_interval.seam_ownership,
    ) != (expected, expected, "NONE"):
        raise ValueError("compiled cell yaw interval does not bind the operation")


def _bridge_object_map(scene: CanonicalScene) -> dict[str, object]:
    objects = _known_exact_source_values(scene.objects, "objects")
    result = {object_.object_id: object_ for object_ in objects}
    if len(result) != len(objects):
        raise ValueError("cardinal bridge object roster is not unique")
    return result


def _bridge_exact_identity_camera(scene: CanonicalScene) -> PinholeCamera:
    """Return the sole retained-core camera subset whose XY frame is world XY."""

    cameras = _known_exact_source_values(scene.cameras, "cameras")
    if scene.cameras.uncertainty != UncertaintyBudgetV2() or len(cameras) != 1:
        raise ValueError("cardinal bridge requires one exact fixed camera")
    camera = cameras[0]
    if type(camera) is not PinholeCamera:
        raise ValueError("cardinal bridge camera must use the exact pinhole model")
    if (
        camera.distortion_model is not CameraDistortionModel.NONE
        or camera.brown_conrady_coefficients is not None
    ):
        raise ValueError("cardinal bridge camera distortion must be NONE")
    if camera.calibration_uncertainty != UncertaintyBudgetV2():
        raise ValueError("cardinal bridge camera calibration must be exact")
    if (
        camera.matrix_layout is not CameraMatrixLayout.ROW_MAJOR
        or camera.camera_axes is not CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConvention.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConvention.POSITIVE_Z_FORWARD
    ):
        raise ValueError("cardinal bridge camera convention is unsupported")
    intrinsics = camera.intrinsics_row_major
    if (
        intrinsics[1] != 0.0
        or intrinsics[3] != 0.0
        or intrinsics[6:] != (0.0, 0.0, 1.0)
    ):
        raise ValueError("cardinal bridge camera intrinsics are noncanonical")
    rotation = camera.world_to_camera.rotation
    if type(rotation) is not Quaternion or (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ) != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("cardinal bridge camera must use identity orientation")
    return camera


def _bridge_cardinal_yaw(rotation: Quaternion, *, label: str) -> int:
    """Decode only a source yaw already representable by the fixed DTOs."""

    if type(rotation) is not Quaternion or rotation.x != 0.0 or rotation.y != 0.0:
        raise ValueError(f"{label} must be a cardinal upright rotation")
    if rotation.z == 0.0:
        return 0
    if rotation.z == rotation.w:
        return 1
    if rotation.w == 0.0:
        return 2
    if rotation.z == -rotation.w:
        return 3
    raise ValueError(f"{label} is unsupported by the fixed cardinal DTO bridge")


def _bridge_world_box_parts(
    geometry: object,
    objects: dict[str, object],
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, int]:
    """Translate one exact upright source box to fixed world-box components."""

    if (
        type(getattr(geometry, "shape", None)) is not UprightBox3DV2
        or getattr(geometry, "approximation", None) is not GeometryApproximationV2.EXACT
    ):
        raise ValueError("geometry is unsupported by the exact fixed-box bridge")
    anchor = geometry.anchor_from_geometry
    anchor_q = _bridge_cardinal_yaw(
        anchor.rotation,
        label=f"geometry {geometry.geometry_id} anchor rotation",
    )
    owner_id = geometry.owner_object_id
    if owner_id is None:
        owner_q = 0
        owner_x = Fraction()
        owner_y = Fraction()
        owner_z = Fraction()
    else:
        owner = objects.get(owner_id)
        if owner is None:
            raise ValueError("geometry owner is absent from the exact object roster")
        owner_pose = owner.pose.world_from_object
        owner_q = _bridge_cardinal_yaw(
            owner_pose.rotation,
            label=f"geometry {geometry.geometry_id} owner rotation",
        )
        owner_x = _bridge_fraction(
            owner_pose.translation.x,
            label=f"geometry {geometry.geometry_id} owner x",
        )
        owner_y = _bridge_fraction(
            owner_pose.translation.y,
            label=f"geometry {geometry.geometry_id} owner y",
        )
        owner_z = _bridge_fraction(
            owner_pose.translation.z,
            label=f"geometry {geometry.geometry_id} owner z",
        )
    anchor_x = _bridge_fraction(
        anchor.translation.x,
        label=f"geometry {geometry.geometry_id} anchor x",
    )
    anchor_y = _bridge_fraction(
        anchor.translation.y,
        label=f"geometry {geometry.geometry_id} anchor y",
    )
    relative_x, relative_y = _rotate_cardinal_xy_components(anchor_x, anchor_y, owner_q)
    total_q = (owner_q + anchor_q) % 4
    half_x = (
        _bridge_fraction(
            geometry.shape.size_m.x,
            label=f"geometry {geometry.geometry_id} size x",
        )
        / 2
    )
    half_y = (
        _bridge_fraction(
            geometry.shape.size_m.y,
            label=f"geometry {geometry.geometry_id} size y",
        )
        / 2
    )
    if total_q % 2:
        half_x, half_y = half_y, half_x
    return (
        owner_x + relative_x,
        owner_y + relative_y,
        owner_z
        + _bridge_fraction(
            anchor.translation.z,
            label=f"geometry {geometry.geometry_id} anchor z",
        ),
        half_x,
        half_y,
        _bridge_fraction(
            geometry.shape.size_m.z,
            label=f"geometry {geometry.geometry_id} size z",
        )
        / 2,
        total_q,
    )


def _bridge_fixed_box(
    geometry: object,
    objects: dict[str, object],
) -> FixedCardinalBoxV3:
    center_x, center_y, center_z, half_x, half_y, half_z, _ = _bridge_world_box_parts(
        geometry,
        objects,
    )
    return FixedCardinalBoxV3(
        box_id=geometry.geometry_id,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        half_x=half_x,
        half_y=half_y,
        half_z=half_z,
    )


def _bridge_collision_boxes(
    scene: CanonicalScene,
    *,
    subject_id: str,
    support_surface_id: str | None,
    objects: dict[str, object],
) -> tuple[tuple[FixedCardinalBoxV3, ...], tuple[FixedCardinalBoxV3, ...]]:
    if support_surface_id is None:
        raise ValueError("cardinal bridge subject support assignment is incomplete")
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    geometry_by_id = {geometry.geometry_id: geometry for geometry in geometries}
    if len(geometry_by_id) != len(geometries):
        raise ValueError("cardinal bridge geometry roster is not unique")
    bodies = _known_exact_source_values(scene.collision_bodies, "collision bodies")
    support_sources = tuple(
        surface
        for surface in _known_exact_source_values(
            scene.support_surfaces, "support surfaces"
        )
        if surface.surface_id == support_surface_id
    )
    if len(support_sources) != 1:
        raise ValueError("cardinal bridge support surface is absent or duplicated")
    supporting_body_id = support_sources[0].supporting_body_id
    subject_geometry_ids: list[str] = []
    obstacle_geometry_ids: list[str] = []
    referenced_geometry_ids: list[str] = []
    for body in bodies:
        ids = body.geometry_instance_ids
        if not ids:
            raise ValueError("cardinal bridge collision body must name geometry")
        for geometry_id in ids:
            geometry = geometry_by_id.get(geometry_id)
            if geometry is None or geometry.role is not GeometryRoleV2.COLLISION:
                raise ValueError("collision body must bind exact collision geometry")
            if geometry.owner_object_id != body.owner_object_id:
                raise ValueError("collision body and geometry owners must agree")
            referenced_geometry_ids.append(geometry_id)
        if body.owner_object_id == subject_id:
            subject_geometry_ids.extend(ids)
        elif body.body_id != supporting_body_id:
            obstacle_geometry_ids.extend(ids)
    all_collision_ids = tuple(
        geometry.geometry_id
        for geometry in geometries
        if geometry.role is GeometryRoleV2.COLLISION
    )
    if tuple(sorted(referenced_geometry_ids)) != tuple(sorted(all_collision_ids)):
        raise ValueError("collision bodies do not cover the complete collision roster")
    if (
        not subject_geometry_ids
        or len(set(subject_geometry_ids)) != len(subject_geometry_ids)
        or len(set(obstacle_geometry_ids)) != len(obstacle_geometry_ids)
    ):
        raise ValueError("collision body roster is incomplete or duplicates a geometry")
    subject_boxes = tuple(
        sorted(
            (
                _bridge_fixed_box(geometry_by_id[geometry_id], objects)
                for geometry_id in subject_geometry_ids
            ),
            key=lambda box: box.box_id,
        )
    )
    obstacle_boxes = tuple(
        sorted(
            (
                _bridge_fixed_box(geometry_by_id[geometry_id], objects)
                for geometry_id in obstacle_geometry_ids
            ),
            key=lambda box: box.box_id,
        )
    )
    return subject_boxes, obstacle_boxes


def _bridge_subject_role_boxes(
    scene: CanonicalScene,
    *,
    subject_id: str,
    role: GeometryRoleV2,
    objects: dict[str, object],
) -> tuple[FixedCardinalBoxV3, ...]:
    """Translate every exact subject geometry for one semantic role to fixed boxes."""

    geometries = _known_exact_source_values(
        scene.geometry_instances,
        "geometry instances",
    )
    candidates = tuple(
        geometry
        for geometry in geometries
        if geometry.owner_object_id == subject_id and geometry.role is role
    )
    if not candidates:
        raise ValueError(f"subject {role.value.lower()} geometry is absent")
    return tuple(
        sorted(
            (_bridge_fixed_box(geometry, objects) for geometry in candidates),
            key=lambda box: box.box_id,
        )
    )


def _bridge_xy_role_signature(
    boxes: tuple[FixedCardinalBoxV3, ...],
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
    """Return the complete per-box XY inputs the retained relation owner consumes."""

    return tuple(
        sorted((box.center_x, box.center_y, box.half_x, box.half_y) for box in boxes)
    )


def _bridge_support_role_signature(
    boxes: tuple[FixedCardinalBoxV3, ...],
) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction], ...]:
    """Return the complete per-box support footprint/contact inputs the owner uses."""

    return tuple(
        sorted(
            (
                box.center_x,
                box.center_y,
                box.center_z,
                box.half_x,
                box.half_y,
                box.half_z,
            )
            for box in boxes
        )
    )


def _bridge_validate_subject_role_geometry_closure(
    scene: CanonicalScene,
    *,
    subject_id: str,
    subject_boxes: tuple[FixedCardinalBoxV3, ...],
    objects: dict[str, object],
) -> None:
    """Fail closed unless role-specific subject geometry preserves owner inputs."""

    relation_boxes = _bridge_subject_role_boxes(
        scene,
        subject_id=subject_id,
        role=GeometryRoleV2.RELATION,
        objects=objects,
    )
    if _bridge_xy_role_signature(relation_boxes) != _bridge_xy_role_signature(
        subject_boxes
    ):
        raise ValueError(
            "subject relation geometry must be XY-congruent with collision boxes"
        )
    support_boxes = _bridge_subject_role_boxes(
        scene,
        subject_id=subject_id,
        role=GeometryRoleV2.SUPPORT,
        objects=objects,
    )
    if _bridge_support_role_signature(support_boxes) != _bridge_support_role_signature(
        subject_boxes
    ):
        raise ValueError(
            "subject support geometry must be contact-congruent with collision boxes"
        )


def _bridge_support_surface(
    surface: SupportSurfaceFact,
    objects: dict[str, object],
) -> SupportSurfaceV3:
    if (
        surface.region_approximation is not GeometryApproximationV2.EXACT
        or surface.boundary_policy.value != "CLOSED"
        or (
            surface.normal_in_anchor.x,
            surface.normal_in_anchor.y,
            surface.normal_in_anchor.z,
        )
        != (0.0, 0.0, 1.0)
        or len(surface.region_uv.components) != 1
        or surface.region_uv.components[0].holes
    ):
        raise ValueError("support surface is unsupported by the exact fixed-box bridge")
    anchor = surface.anchor_from_surface
    anchor_q = _bridge_cardinal_yaw(
        anchor.rotation,
        label=f"support surface {surface.surface_id} anchor rotation",
    )
    if surface.owner_object_id is None:
        owner_q = 0
        owner_x = Fraction()
        owner_y = Fraction()
        owner_z = Fraction()
    else:
        owner = objects.get(surface.owner_object_id)
        if owner is None:
            raise ValueError("support surface owner is absent from the object roster")
        pose = owner.pose.world_from_object
        owner_q = _bridge_cardinal_yaw(
            pose.rotation,
            label=f"support surface {surface.surface_id} owner rotation",
        )
        owner_x = _bridge_fraction(pose.translation.x, label="support owner x")
        owner_y = _bridge_fraction(pose.translation.y, label="support owner y")
        owner_z = _bridge_fraction(pose.translation.z, label="support owner z")
    anchor_x = _bridge_fraction(anchor.translation.x, label="support anchor x")
    anchor_y = _bridge_fraction(anchor.translation.y, label="support anchor y")
    translated_x, translated_y = _rotate_cardinal_xy_components(
        anchor_x, anchor_y, owner_q
    )
    origin_x = owner_x + translated_x
    origin_y = owner_y + translated_y
    total_q = (owner_q + anchor_q) % 4
    vertices = surface.region_uv.components[0].exterior.vertices
    if len(vertices) != 4:
        raise ValueError("support surface must be one exact axis-aligned rectangle")
    world_vertices = tuple(
        (
            origin_x
            + _rotate_cardinal_xy_components(
                _bridge_fraction(vertex.x, label="support vertex x"),
                _bridge_fraction(vertex.y, label="support vertex y"),
                total_q,
            )[0],
            origin_y
            + _rotate_cardinal_xy_components(
                _bridge_fraction(vertex.x, label="support vertex x"),
                _bridge_fraction(vertex.y, label="support vertex y"),
                total_q,
            )[1],
        )
        for vertex in vertices
    )
    xs = tuple(sorted({vertex[0] for vertex in world_vertices}))
    ys = tuple(sorted({vertex[1] for vertex in world_vertices}))
    if (
        len(xs) != 2
        or len(ys) != 2
        or set(world_vertices) != {(x, y) for x in xs for y in ys}
    ):
        raise ValueError("support surface must remain an exact world-XY rectangle")
    return SupportSurfaceV3(
        x_lower=xs[0],
        x_upper=xs[1],
        y_lower=ys[0],
        y_upper=ys[1],
        z=owner_z + _bridge_fraction(anchor.translation.z, label="support anchor z"),
    )


def _bridge_target_and_preservation_rows(
    problem: CounterfactualProblemIR,
    *,
    subject_id: str,
    reference_id: str,
) -> tuple[PredicateAtom, PredicateAtom, str]:
    if len(problem.before_preconditions) != 1:
        raise ValueError("cardinal bridge requires one target before relation")
    before = _direct_predicate_atom(problem.before_preconditions[0].formula)
    _validate_target_relation_operands(
        before,
        subject_id=subject_id,
        reference_id=reference_id,
        phase="BEFORE",
    )
    after = _direct_predicate_atom(problem.after_goal.formula)
    after_relation = _validate_target_relation_operands(
        after,
        subject_id=subject_id,
        reference_id=reference_id,
        phase="AFTER",
    )
    if not after_relation.startswith("relation:"):
        raise ValueError("target after relation must be one registered relation")
    if len(problem.preservation_invariants) != 1:
        raise ValueError("cardinal bridge requires one preservation relation row")
    preservation = problem.preservation_invariants[0]
    for phase, formula in (
        ("BEFORE", preservation.before_formula),
        ("AFTER", preservation.after_formula),
    ):
        _validate_preservation_operands(
            _direct_predicate_atom(formula),
            subject_id,
            phase,
        )
    return before, after, after_relation.removeprefix("relation:")


def _bridge_reference_relation_box(
    scene: CanonicalScene,
    *,
    reference_id: str,
    objects: dict[str, object],
) -> FixedCardinalBoxV3:
    geometries = _known_exact_source_values(
        scene.geometry_instances, "geometry instances"
    )
    candidates = tuple(
        geometry
        for geometry in geometries
        if (
            geometry.owner_object_id == reference_id
            and geometry.role is GeometryRoleV2.RELATION
        )
    )
    if len(candidates) != 1:
        raise ValueError("target reference must bind one exact relation geometry")
    return _bridge_fixed_box(candidates[0], objects)


def _bridge_object_pivot_xy(
    objects: dict[str, object],
    object_id: str,
) -> tuple[Fraction, Fraction]:
    object_ = objects.get(object_id)
    if object_ is None:
        raise ValueError("cardinal bridge pivot object is absent from the scene")
    pose = object_.pose.world_from_object
    _bridge_cardinal_yaw(pose.rotation, label=f"pivot object {object_id} rotation")
    return (
        _bridge_fraction(pose.translation.x, label=f"pivot object {object_id} x"),
        _bridge_fraction(pose.translation.y, label=f"pivot object {object_id} y"),
    )


# Preserve supported public type/function and pickle lookup.
_bridge_compiled_cell_member.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_continuous_compiled_cell_member.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_is_proper_exact_dyadic_continuous_descendant.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_is_proper_exact_dyadic_descendant.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_require_replayed_compilation.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_require_replayed_continuous_compilation.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_validate_cell_yaw.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_object_map.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_exact_identity_camera.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_cardinal_yaw.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_world_box_parts.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_fixed_box.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_collision_boxes.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_subject_role_boxes.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_xy_role_signature.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_support_role_signature.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_validate_subject_role_geometry_closure.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_support_surface.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_target_and_preservation_rows.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_reference_relation_box.__module__ = "spatialcf.core.upright_se2_compiler"
_bridge_object_pivot_xy.__module__ = "spatialcf.core.upright_se2_compiler"
