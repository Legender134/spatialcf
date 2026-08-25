# ruff: noqa: F811, I001
"""Private current candidate-family compilers; exact migrated bodies."""

from __future__ import annotations

# Migrated from visibility_domain.py.
"""Sound visibility-domain compilation for a small analytic Canonical v2 subset."""


from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels.rect import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
)
from spatialcf.core._internal.kernels.rectilinear import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    RectilinearTopologyV2,
    intersect_rectilinear_regions_v2,
    normalize_rectilinear_region_v2,
    union_rectilinear_regions_v2,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    Quaternion,
    RigidTransformV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    OccluderSoundnessPolicy,
    VisibilityAreaMeasure,
    VisibilityConstraint,
    VisibilityDepthPolicy,
    VisibilityMaskPolicy,
    VisibilityMetricFormula,
    VisibilityMetricKind,
)
from spatialcf.domain.geometry import (
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.problem import SemanticProblemV2
from spatialcf.domain.scene import (
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
    CanonicalObject,
    PinholeCamera,
)


class VisibilityDomainKindV2(StrEnum):
    BRACKET = "BRACKET"
    IDENTITY = "IDENTITY"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"


@dataclass(frozen=True, slots=True)
class VisibilityDomainOutcomeV2:
    """One exact constant result or an inner/outer allowed-delta bracket."""

    kind: VisibilityDomainKindV2
    inner_allowed_delta: ExactRectilinearRegionV2 | None = None
    outer_allowed_delta: ExactRectilinearRegionV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VisibilityDomainKindV2):
            raise TypeError("kind must be a VisibilityDomainKindV2")
        object.__setattr__(
            self,
            "finding_codes",
            tuple(sorted(set(self.finding_codes))),
        )
        if self.kind is VisibilityDomainKindV2.BRACKET:
            if not isinstance(
                self.inner_allowed_delta, ExactRectilinearRegionV2
            ) or not isinstance(self.outer_allowed_delta, ExactRectilinearRegionV2):
                raise ValueError("BRACKET requires exact inner and outer regions")
            if self.finding_codes:
                raise ValueError("BRACKET cannot carry findings")
            return
        if self.inner_allowed_delta is not None or self.outer_allowed_delta is not None:
            raise ValueError(f"{self.kind.value} must not carry partial regions")
        if self.kind is VisibilityDomainKindV2.IDENTITY:
            if self.finding_codes:
                raise ValueError("IDENTITY cannot carry findings")
            return
        if not self.finding_codes:
            raise ValueError(f"{self.kind.value} requires a finding")


VisibilityDomainCompilationOutcomeV2 = VisibilityDomainOutcomeV2


class _BaselineStateV2(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class _BaselineClassificationV2:
    state: _BaselineStateV2
    failure_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ProjectionCertificateV2:
    min_dx_m: Fraction
    min_dy_m: Fraction
    max_dx_m: Fraction
    max_dy_m: Fraction
    near_face_image_area_fraction: Fraction

    @property
    def has_full_containment_delta(self) -> bool:
        return self.min_dx_m <= self.max_dx_m and self.min_dy_m <= self.max_dy_m


def compile_visibility_domain_v2(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint | str,
    universe: ExactRectilinearRegionV2,
    *,
    max_atomic_cells: int | None = None,
    atomic_budget: RectilinearAtomicBudgetV2 | None = None,
) -> VisibilityDomainOutcomeV2:
    """Compile one hard visibility predicate over a finite XY-delta universe."""

    budget = _resolve_atomic_budget(max_atomic_cells, atomic_budget)
    if not isinstance(problem, SemanticProblemV2):
        raise TypeError("problem must be a SemanticProblemV2")
    if not isinstance(universe, ExactRectilinearRegionV2):
        raise TypeError("universe must be an ExactRectilinearRegionV2")

    checked_problem = SemanticProblemV2.model_validate(
        problem.model_dump(mode="python"),
        strict=True,
    )
    selected = _resolve_constraint(checked_problem, constraint)
    checked_universe = _revalidate_universe(universe, budget)
    if isinstance(checked_universe, VisibilityDomainOutcomeV2):
        return checked_universe
    if isinstance(selected, VisibilityDomainOutcomeV2):
        return selected

    common_findings = _common_findings(checked_problem, selected)
    subject_id = checked_problem.constraints.allowed_edit.subject_id
    moving_subject = subject_id in selected.query_object_ids
    fixed_query_ids = tuple(
        object_id for object_id in selected.query_object_ids if object_id != subject_id
    )

    if moving_subject:
        findings = [*common_findings]
        findings.extend(_moving_subset_findings(checked_problem, selected))
        if fixed_query_ids:
            findings.extend(_baseline_family_findings(checked_problem, selected))
        if findings:
            return _unknown(*findings)
        return _compile_moving_subject(
            checked_problem,
            selected,
            fixed_query_ids,
            checked_universe,
            atomic_budget=budget,
        )

    findings = [*common_findings]
    findings.extend(_fixed_subset_findings(checked_problem, selected))
    findings.extend(_baseline_family_findings(checked_problem, selected))
    if findings:
        return _unknown(*findings)
    return _compile_fixed_queries(
        checked_problem,
        selected,
        checked_universe,
        atomic_budget=budget,
    )


def _compile_fixed_queries(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
    universe: ExactRectilinearRegionV2,
    *,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> VisibilityDomainOutcomeV2:
    classification = _classify_baselines(
        problem,
        constraint,
        constraint.query_object_ids,
    )
    if classification.state is _BaselineStateV2.FAIL:
        return _empty(*classification.failure_codes)
    if classification.state is _BaselineStateV2.PASS:
        return VisibilityDomainOutcomeV2(kind=VisibilityDomainKindV2.IDENTITY)
    if universe.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:VISIBILITY_DOMAIN:{constraint.constraint_id}")
    empty = _make_empty_region(atomic_budget)
    if isinstance(empty, VisibilityDomainOutcomeV2):
        return empty
    return _bracket(empty, universe)


def _compile_moving_subject(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
    fixed_query_ids: tuple[str, ...],
    universe: ExactRectilinearRegionV2,
    *,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> VisibilityDomainOutcomeV2:
    fixed_classification = _classify_baselines(
        problem,
        constraint,
        fixed_query_ids,
    )
    if fixed_classification.state is _BaselineStateV2.FAIL:
        return _empty(*fixed_classification.failure_codes)

    certificate = _projection_certificate(problem, constraint)
    area_threshold = Fraction.from_float(constraint.minimum_image_area_fraction)
    subject_is_guaranteed = certificate.near_face_image_area_fraction >= area_threshold
    clear_inner = (
        fixed_classification.state is _BaselineStateV2.AMBIGUOUS
        or not subject_is_guaranteed
        or not certificate.has_full_containment_delta
    )
    if universe.topology is RectilinearTopologyV2.EMPTY:
        return _empty(f"EXACT_EMPTY:VISIBILITY_DOMAIN:{constraint.constraint_id}")
    if clear_inner:
        empty = _make_empty_region(atomic_budget)
        if isinstance(empty, VisibilityDomainOutcomeV2):
            return empty
        return _bracket(empty, universe)

    containment_rectangle = ExactAxisAlignedRectV2.from_fraction_bounds(
        min_x_m=certificate.min_dx_m,
        min_y_m=certificate.min_dy_m,
        max_x_m=certificate.max_dx_m,
        max_y_m=certificate.max_dy_m,
        coordinate_space=RectCoordinateSpaceV2.TRANSLATION_DELTA_XY_M,
    )
    containment_outcome = normalize_rectilinear_region_v2(
        (containment_rectangle,),
        atomic_budget=atomic_budget,
    )
    failure = _maybe_nested_failure(containment_outcome)
    if failure is not None:
        return failure
    containment = _exact_region(containment_outcome)
    inner_outcome = intersect_rectilinear_regions_v2(
        universe,
        containment,
        atomic_budget=atomic_budget,
    )
    failure = _maybe_nested_failure(inner_outcome)
    if failure is not None:
        return failure
    return _bracket(_exact_region(inner_outcome), universe)


def _resolve_constraint(
    problem: SemanticProblemV2,
    requested: VisibilityConstraint | str,
) -> VisibilityConstraint | VisibilityDomainOutcomeV2:
    if isinstance(requested, VisibilityConstraint):
        checked = VisibilityConstraint.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a VisibilityConstraint or exact str ID")
    registered = next(
        (
            item
            for item in problem.constraints.visibility_constraints
            if item.constraint_id == constraint_id
        ),
        None,
    )
    if registered is None:
        return _unknown(f"UNKNOWN_VISIBILITY_CONSTRAINT:{constraint_id}")
    if checked is not None and checked != registered:
        return _unknown(f"VISIBILITY_CONSTRAINT_MISMATCH:{constraint_id}")
    return registered


def _common_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
) -> tuple[str, ...]:
    findings: list[str] = []
    if constraint.threshold_boundary_policy is not BoundaryPolicy.CLOSED:
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:BOUNDARY_POLICY:"
            f"{constraint.threshold_boundary_policy.value}"
        )
    if constraint.mask_policy is not VisibilityMaskPolicy.FULL_OBJECT:
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:MASK_POLICY:{constraint.mask_policy.value}"
        )
    if (
        constraint.occluder_soundness_policy
        is not OccluderSoundnessPolicy.EXACT_OR_OUTER_SHAPE_BOUND
    ):
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:OCCLUDER_SOUNDNESS_POLICY:"
            f"{constraint.occluder_soundness_policy.value}"
        )
    if not _numeric_policy_is_zero(problem.numeric_policy):
        findings.append("UNSUPPORTED_VISIBILITY_DOMAIN:NUMERIC_POLICY")
    findings.extend(_fact_family_findings("OBJECTS", problem.scene.objects))
    findings.extend(
        _fact_family_findings(
            "GEOMETRY_INSTANCES",
            problem.scene.geometry_instances,
        )
    )
    findings.extend(_visibility_semantics_findings(problem, constraint))
    return tuple(sorted(set(findings)))


def _visibility_semantics_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
    *,
    supported_image_area_formulas: tuple[VisibilityMetricFormula, ...] = (
        VisibilityMetricFormula.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA,
    ),
) -> tuple[str, ...]:
    expected = {
        VisibilityMetricKind.VISIBLE_FRACTION: (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            (
                VisibilityMetricFormula.VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA,
            ),
        ),
        VisibilityMetricKind.IMAGE_AREA_FRACTION: (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            supported_image_area_formulas,
        ),
        VisibilityMetricKind.TRUNCATED_FRACTION: (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            (VisibilityMetricFormula.ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA,),
        ),
    }
    findings: list[str] = []
    for definition in problem.visibility_semantics.definitions:
        definition_id, version, formulas = expected[definition.kind]
        if (
            definition.reference != (definition_id, version)
            or definition.formula not in formulas
            or definition.area_measure
            is not VisibilityAreaMeasure.CONTINUOUS_PIXEL_PLANE_AREA
            or definition.depth_policy
            is not VisibilityDepthPolicy.NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES
        ):
            findings.append(
                f"UNSUPPORTED_VISIBILITY_DOMAIN:METRIC_SEMANTICS:{definition.kind.value}"
            )
    return tuple(findings)


def _fixed_subset_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
) -> tuple[str, ...]:
    subject_id = problem.constraints.allowed_edit.subject_id
    subject_occluders = tuple(
        item.geometry_id
        for item in problem.scene.geometry_instances.values or ()
        if item.owner_object_id == subject_id and item.role is GeometryRoleV2.OCCLUDER
    )
    if subject_occluders:
        return (
            (
                "UNSUPPORTED_VISIBILITY_DOMAIN:SUBJECT_OCCLUDER:"
                f"{constraint.constraint_id}:{','.join(sorted(subject_occluders))}"
            ),
        )
    return ()


def _moving_subset_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
) -> tuple[str, ...]:
    findings: list[str] = []
    geometries = problem.scene.geometry_instances.values or ()
    occluders = tuple(
        item.geometry_id for item in geometries if item.role is GeometryRoleV2.OCCLUDER
    )
    if occluders:
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:OCCLUDER_GEOMETRY:"
            + ",".join(sorted(occluders))
        )

    subject_id = problem.constraints.allowed_edit.subject_id
    objects = {item.object_id: item for item in problem.scene.objects.values or ()}
    subject = objects[subject_id]
    if not _is_identity_rotation(subject.pose.world_from_object.rotation):
        findings.append(f"UNSUPPORTED_VISIBILITY_DOMAIN:OBJECT_ROTATION:{subject_id}")
    visual_geometries = tuple(
        item
        for item in geometries
        if item.owner_object_id == subject_id and item.role is GeometryRoleV2.VISUAL
    )
    if len(visual_geometries) != 1:
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:VISUAL_GEOMETRY_CARDINALITY:"
            f"{subject_id}:{len(visual_geometries)}"
        )
    else:
        visual = visual_geometries[0]
        if visual.approximation is not GeometryApproximationV2.EXACT:
            findings.append(
                "UNSUPPORTED_VISIBILITY_DOMAIN:VISUAL_APPROXIMATION:"
                f"{visual.geometry_id}:{visual.approximation.value}"
            )
        if not _uncertainty_is_zero(visual.uncertainty):
            findings.append(
                f"UNSUPPORTED_VISIBILITY_DOMAIN:VISUAL_UNCERTAINTY:{visual.geometry_id}"
            )
        if not isinstance(visual.shape, UprightBox3DV2):
            findings.append(
                "UNSUPPORTED_VISIBILITY_DOMAIN:VISUAL_SHAPE:"
                f"{visual.geometry_id}:{visual.shape.shape_type}"
            )
        if not _is_identity_rotation(visual.anchor_from_geometry.rotation):
            findings.append(
                f"UNSUPPORTED_VISIBILITY_DOMAIN:VISUAL_ROTATION:{visual.geometry_id}"
            )

    camera_facts = problem.scene.cameras
    findings.extend(_fact_family_findings("CAMERAS", camera_facts))
    cameras = {item.camera_id: item for item in camera_facts.values or ()}
    camera = cameras.get(constraint.camera_id)
    if camera is not None:
        findings.extend(_camera_findings(camera))

    if not findings and camera is not None:
        visual = visual_geometries[0]
        assert isinstance(visual.shape, UprightBox3DV2)
        z_min, z_max = _box_depth_interval(subject, visual)
        near = Fraction.from_float(camera.near_clip_m)
        far = Fraction.from_float(camera.far_clip_m)
        if z_min <= 0 or z_min < near or z_max > far:
            findings.append(
                f"UNSUPPORTED_VISIBILITY_DOMAIN:DEPTH_CLIP:{visual.geometry_id}"
            )
    return tuple(sorted(set(findings)))


def _camera_findings(camera: PinholeCamera) -> tuple[str, ...]:
    findings: list[str] = []
    if not _is_identity_transform(camera.world_to_camera):
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:CAMERA_TRANSFORM:{camera.camera_id}"
        )
    if camera.distortion_model is not CameraDistortionModel.NONE:
        findings.append(f"UNSUPPORTED_VISIBILITY_DOMAIN:DISTORTION:{camera.camera_id}")
    if not _uncertainty_is_zero(camera.calibration_uncertainty):
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:CAMERA_UNCERTAINTY:{camera.camera_id}"
        )
    intrinsics = camera.intrinsics_row_major
    if (
        intrinsics[1] != 0.0
        or intrinsics[3] != 0.0
        or intrinsics[6:] != (0.0, 0.0, 1.0)
    ):
        findings.append(f"UNSUPPORTED_VISIBILITY_DOMAIN:INTRINSICS:{camera.camera_id}")
    if (
        camera.matrix_layout is not CameraMatrixLayout.ROW_MAJOR
        or camera.camera_axes is not CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConvention.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConvention.POSITIVE_Z_FORWARD
    ):
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:CAMERA_CONVENTION:{camera.camera_id}"
        )
    return tuple(findings)


def _baseline_family_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
) -> tuple[str, ...]:
    findings = list(
        _fact_family_findings(
            "BASELINE_OBSERVATIONS",
            problem.scene.baseline_observations,
        )
    )
    if FactCompletenessV2.EXACT not in constraint.accepted_baseline_completeness:
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:BASELINE_COMPLETENESS_POLICY:"
            f"{constraint.constraint_id}"
        )
    return tuple(findings)


def _fact_family_findings(label: str, facts: FactSetV2) -> tuple[str, ...]:
    if facts.availability is FactAvailabilityV2.MISSING:
        return (f"MISSING_FACT:{label}",)
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return (
            (
                "UNSUPPORTED_VISIBILITY_DOMAIN:"
                f"{label}_AVAILABILITY:{facts.availability.value}"
            ),
        )
    findings: list[str] = []
    if facts.completeness is not FactCompletenessV2.EXACT:
        value = facts.completeness.value if facts.completeness is not None else "NONE"
        findings.append(f"UNSUPPORTED_VISIBILITY_DOMAIN:{label}_COMPLETENESS:{value}")
    if facts.uncertainty is None or not _uncertainty_is_zero(facts.uncertainty):
        findings.append(f"UNSUPPORTED_VISIBILITY_DOMAIN:{label}_UNCERTAINTY")
    return tuple(findings)


def _classify_baselines(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
    object_ids: tuple[str, ...],
) -> _BaselineClassificationV2:
    if not object_ids:
        return _BaselineClassificationV2(state=_BaselineStateV2.PASS)
    observations = {
        (
            item.object_id,
            item.camera_id,
            item.metric_definition_id,
            item.metric_definition_version,
        ): item
        for item in problem.scene.baseline_observations.values or ()
    }
    specifications = (
        (
            "VISIBLE_FRACTION",
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            Fraction.from_float(constraint.minimum_visible_fraction),
            True,
        ),
        (
            "IMAGE_AREA_FRACTION",
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            Fraction.from_float(constraint.minimum_image_area_fraction),
            True,
        ),
        (
            "TRUNCATED_FRACTION",
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            Fraction.from_float(constraint.maximum_truncated_fraction),
            False,
        ),
    )
    failures: list[str] = []
    ambiguous = False
    for object_id in object_ids:
        for label, definition_id, version, threshold, is_minimum in specifications:
            observation = observations[
                (object_id, constraint.camera_id, definition_id, version)
            ]
            lower = Fraction.from_float(observation.normalized_lower_bound)
            upper = Fraction.from_float(observation.normalized_upper_bound)
            if is_minimum:
                worst_passes = lower >= threshold
                best_fails = upper < threshold
            else:
                worst_passes = upper <= threshold
                best_fails = lower > threshold
            if best_fails:
                failures.append(
                    "EXACT_EMPTY:VISIBILITY_BASELINE:"
                    f"{constraint.constraint_id}:{object_id}:{label}"
                )
            elif not worst_passes:
                ambiguous = True
    if failures:
        return _BaselineClassificationV2(
            state=_BaselineStateV2.FAIL,
            failure_codes=tuple(sorted(set(failures))),
        )
    if ambiguous:
        return _BaselineClassificationV2(state=_BaselineStateV2.AMBIGUOUS)
    return _BaselineClassificationV2(state=_BaselineStateV2.PASS)


def _projection_certificate(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraint,
) -> _ProjectionCertificateV2:
    subject_id = problem.constraints.allowed_edit.subject_id
    subject = next(
        item
        for item in problem.scene.objects.values or ()
        if item.object_id == subject_id
    )
    visual = next(
        item
        for item in problem.scene.geometry_instances.values or ()
        if item.owner_object_id == subject_id and item.role is GeometryRoleV2.VISUAL
    )
    camera = next(
        item
        for item in problem.scene.cameras.values or ()
        if item.camera_id == constraint.camera_id
    )
    assert isinstance(visual.shape, UprightBox3DV2)

    center_x = Fraction.from_float(
        subject.pose.world_from_object.translation.x
    ) + Fraction.from_float(visual.anchor_from_geometry.translation.x)
    center_y = Fraction.from_float(
        subject.pose.world_from_object.translation.y
    ) + Fraction.from_float(visual.anchor_from_geometry.translation.y)
    center_z = Fraction.from_float(
        subject.pose.world_from_object.translation.z
    ) + Fraction.from_float(visual.anchor_from_geometry.translation.z)
    half_x = Fraction.from_float(visual.shape.size_m.x) / 2
    half_y = Fraction.from_float(visual.shape.size_m.y) / 2
    half_z = Fraction.from_float(visual.shape.size_m.z) / 2
    corners = tuple(
        (center_x + sx * half_x, center_y + sy * half_y, center_z + sz * half_z)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    )

    intrinsics = tuple(
        Fraction.from_float(item) for item in camera.intrinsics_row_major
    )
    fx, cx = intrinsics[0], intrinsics[2]
    fy, cy = intrinsics[4], intrinsics[5]
    left = Fraction(1, 2)
    right = Fraction(camera.width_px) - left
    top = Fraction(1, 2)
    bottom = Fraction(camera.height_px) - top
    min_dx = max((left - cx) * z / fx - x for x, _, z in corners)
    max_dx = min((right - cx) * z / fx - x for x, _, z in corners)
    min_dy = max((top - cy) * z / fy - y for _, y, z in corners)
    max_dy = min((bottom - cy) * z / fy - y for _, y, z in corners)

    near_z = center_z - half_z
    projected_width = fx * Fraction.from_float(visual.shape.size_m.x) / near_z
    projected_height = fy * Fraction.from_float(visual.shape.size_m.y) / near_z
    image_area_fraction = (projected_width * projected_height) / (
        Fraction(camera.width_px) * Fraction(camera.height_px)
    )
    return _ProjectionCertificateV2(
        min_dx_m=min_dx,
        min_dy_m=min_dy,
        max_dx_m=max_dx,
        max_dy_m=max_dy,
        near_face_image_area_fraction=image_area_fraction,
    )


def _box_depth_interval(
    subject: CanonicalObject,
    visual: GeometryInstanceV2,
) -> tuple[Fraction, Fraction]:
    assert isinstance(visual.shape, UprightBox3DV2)
    center = Fraction.from_float(
        subject.pose.world_from_object.translation.z
    ) + Fraction.from_float(visual.anchor_from_geometry.translation.z)
    half = Fraction.from_float(visual.shape.size_m.z) / 2
    return center - half, center + half


def _revalidate_universe(
    universe: ExactRectilinearRegionV2,
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | VisibilityDomainOutcomeV2:
    validation = union_rectilinear_regions_v2(
        universe,
        universe,
        atomic_budget=atomic_budget,
    )
    failure = _maybe_nested_failure(validation)
    if failure is not None:
        return failure
    return _exact_region(validation)


def _make_empty_region(
    atomic_budget: RectilinearAtomicBudgetV2,
) -> ExactRectilinearRegionV2 | VisibilityDomainOutcomeV2:
    outcome = normalize_rectilinear_region_v2((), atomic_budget=atomic_budget)
    failure = _maybe_nested_failure(outcome)
    if failure is not None:
        return failure
    return _exact_region(outcome)


def _is_identity_rotation(rotation: Quaternion) -> bool:
    return (rotation.x, rotation.y, rotation.z, rotation.w) == (0.0, 0.0, 0.0, 1.0)


def _is_identity_transform(transform: RigidTransformV2) -> bool:
    translation = transform.translation
    return (translation.x, translation.y, translation.z) == (
        0.0,
        0.0,
        0.0,
    ) and _is_identity_rotation(transform.rotation)


def _numeric_policy_is_zero(policy: NumericPolicyV2) -> bool:
    return all(
        value == 0.0
        for value in (
            policy.linear_tolerance_m,
            policy.area_tolerance_m2,
            policy.angular_tolerance_rad,
            policy.pixel_tolerance_px,
            policy.fraction_tolerance,
        )
    )


def _uncertainty_is_zero(uncertainty: UncertaintyBudgetV2) -> bool:
    return _numeric_policy_is_zero(
        uncertainty.source_error
    ) and _numeric_policy_is_zero(uncertainty.shape_approximation)


def _resolve_atomic_budget(
    max_atomic_cells: int | None,
    atomic_budget: RectilinearAtomicBudgetV2 | None,
) -> RectilinearAtomicBudgetV2:
    if (max_atomic_cells is None) == (atomic_budget is None):
        raise ValueError("provide exactly one of max_atomic_cells or atomic_budget")
    if atomic_budget is not None:
        if type(atomic_budget) is not RectilinearAtomicBudgetV2:
            raise TypeError("atomic_budget must be a RectilinearAtomicBudgetV2")
        atomic_budget.validate()
        return atomic_budget
    if type(max_atomic_cells) is not int:
        raise TypeError("max_atomic_cells must be an exact int")
    return RectilinearAtomicBudgetV2(limit=max_atomic_cells)


def _exact_region(outcome: RectilinearRegionOutcomeV2) -> ExactRectilinearRegionV2:
    if outcome.kind is not RectilinearOutcomeKindV2.EXACT or outcome.region is None:
        raise RuntimeError("nested rectilinear outcome is not exact")
    return outcome.region


def _maybe_nested_failure(
    outcome: RectilinearRegionOutcomeV2,
) -> VisibilityDomainOutcomeV2 | None:
    if outcome.kind is RectilinearOutcomeKindV2.EXACT:
        return None
    if outcome.kind is RectilinearOutcomeKindV2.RESOURCE_LIMIT:
        return _resource()
    return _unknown(*outcome.finding_codes)


def _bracket(
    inner: ExactRectilinearRegionV2,
    outer: ExactRectilinearRegionV2,
) -> VisibilityDomainOutcomeV2:
    return VisibilityDomainOutcomeV2(
        kind=VisibilityDomainKindV2.BRACKET,
        inner_allowed_delta=inner,
        outer_allowed_delta=outer,
    )


def _empty(*findings: str) -> VisibilityDomainOutcomeV2:
    return VisibilityDomainOutcomeV2(
        kind=VisibilityDomainKindV2.EMPTY,
        finding_codes=tuple(findings),
    )


def _unknown(*findings: str) -> VisibilityDomainOutcomeV2:
    return VisibilityDomainOutcomeV2(
        kind=VisibilityDomainKindV2.UNKNOWN,
        finding_codes=tuple(findings),
    )


def _resource() -> VisibilityDomainOutcomeV2:
    return VisibilityDomainOutcomeV2(
        kind=VisibilityDomainKindV2.RESOURCE_LIMIT,
        finding_codes=("RESOURCE_LIMIT:ATOMIC_CELLS",),
    )


# Migrated from continuous_yaw_visibility.py.
"""Directed fixed-camera visibility projection for continuous-yaw candidates."""


import hashlib
import warnings
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core._internal.kernels import so2 as so2_interval
from spatialcf.core._internal.kernels.convex_translation import (
    _directed_world_corner_boxes,
)
from spatialcf.core._internal.kernels.so2 import (
    SO2AtomicBudgetExhaustedV2,
    SO2AtomicBudgetV2,
    SO2IntervalKindV2,
)
from spatialcf.core._internal.kernels.strict_convex import (
    StrictConvexIntersectionBudgetExhaustedV2,
    StrictConvexIntersectionBudgetV2,
    StrictConvexIntersectionCellV2,
    StrictConvexIntersectionComplexV2,
)
from spatialcf.core._internal.kernels.upright_box import (
    OrientedUprightBoxBoundsV2,
    compile_oriented_upright_box_bounds_v2,
)
from spatialcf.core._internal.compilation.target import (
    _copy_stage as _copy_target_stage,
)
from spatialcf.core._internal.compilation.target import (
    _TargetAwareCandidateStageV2,
)
from spatialcf.core._internal.compilation.collision import (
    MultiObstacleStrictConvexCandidateResourceUsageV2,
    _artifact_bytes,
    _copy_intersection_complex,
    _InvalidInputV2,
    _require_finding_codes,
    _strict_problem,
)
from spatialcf.core._internal.compilation.visibility import (
    _visibility_semantics_findings,
)
from spatialcf.domain.artifacts import (
    CanonicalObjectV2_2,
    GeometryInstanceV2_2,
    PinholeCameraV2_2,
    SemanticProblemV2_2,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    NumericPolicyV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.constraints import (
    BoundaryPolicy,
    OccluderSoundnessPolicy,
    VisibilityConstraint,
    VisibilityMaskPolicy,
)
from spatialcf.domain.geometry import (
    DirectedYawIntervalTransformV2_2,
    GeometryApproximationV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.scene import (
    BaselineObservation,
    CameraAxes,
    CameraDepthConvention,
    CameraDistortionModel,
    CameraMatrixLayout,
    CameraPixelConvention,
)

CONTINUOUS_YAW_VISIBILITY_KERNEL_ID_V2 = (
    "geometry-kernel:rational-continuous-yaw-fixed-camera-visibility-v2"
)
_STAGE_HASH_DOMAIN_V2 = b"spatialcf.continuous-yaw-visibility-stage.v2.8\0"


class _CompleteContinuousYawCandidateKindV2(StrEnum):
    STAGE = "STAGE"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    NUMERIC_GAP = "NUMERIC_GAP"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_INPUT = "INVALID_INPUT"


class _ProjectionClassV2(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class _VisibilityCellMetricBoundsV2:
    cell_id: str
    visible_fraction_lower: Fraction
    visible_fraction_upper: Fraction
    image_area_fraction_lower: Fraction
    image_area_fraction_upper: Fraction
    truncated_fraction_lower: Fraction
    truncated_fraction_upper: Fraction

    def __post_init__(self) -> None:
        if type(self.cell_id) is not str or not self.cell_id.strip():
            raise ValueError("visibility metric cell ID must be non-blank")
        for lower_name, upper_name in (
            ("visible_fraction_lower", "visible_fraction_upper"),
            ("image_area_fraction_lower", "image_area_fraction_upper"),
            ("truncated_fraction_lower", "truncated_fraction_upper"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if type(lower) is not Fraction or type(upper) is not Fraction:
                raise TypeError("visibility metric endpoints must be exact Fractions")
            so2_interval._require_fraction_cap(lower)
            so2_interval._require_fraction_cap(upper)
            if lower < 0 or lower > upper or upper > 1:
                raise ValueError("visibility metric interval must lie within [0, 1]")


@dataclass(frozen=True, slots=True)
class _CompleteContinuousYawCandidateStageV2:
    semantic_problem_sha256: str
    compiler_config_sha256: str
    upstream_t15_artifact_sha256: str
    upstream_target_stage_sha256: str
    subject_id: str
    visibility_constraint_id: str
    inner_allowed: StrictConvexIntersectionComplexV2
    outer_allowed: StrictConvexIntersectionComplexV2
    outer_cell_metrics: tuple[_VisibilityCellMetricBoundsV2, ...]
    resource_usage: MultiObstacleStrictConvexCandidateResourceUsageV2
    remaining_constraint_ids: tuple[str, ...]
    unsat_prefix_eligible: bool

    def __post_init__(self) -> None:
        for label, digest in (
            ("semantic_problem_sha256", self.semantic_problem_sha256),
            ("compiler_config_sha256", self.compiler_config_sha256),
            ("upstream_t15_artifact_sha256", self.upstream_t15_artifact_sha256),
            ("upstream_target_stage_sha256", self.upstream_target_stage_sha256),
        ):
            _require_digest(digest, label)
        for label, value in (
            ("subject_id", self.subject_id),
            ("visibility_constraint_id", self.visibility_constraint_id),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} must be a non-blank exact string")
        inner = _copy_intersection_complex(self.inner_allowed)
        outer = _copy_intersection_complex(self.outer_allowed)
        if inner.universe != outer.universe:
            raise ValueError("visibility inner and outer require one exact universe")
        if not all(outer.contains_point(cell.strict_witness) for cell in inner.cells):
            raise ValueError("visibility inner witness escaped the outer domain")
        if type(self.outer_cell_metrics) is not tuple:
            raise TypeError("outer_cell_metrics must be an exact tuple")
        metrics = tuple(_copy_metric(item) for item in self.outer_cell_metrics)
        if tuple(item.cell_id for item in metrics) != tuple(
            cell.cell_id for cell in outer.cells
        ):
            raise ValueError("visibility metrics must close the outer cell universe")
        if (
            type(self.resource_usage)
            is not MultiObstacleStrictConvexCandidateResourceUsageV2
        ):
            raise TypeError("resource_usage has the wrong exact type")
        usage = MultiObstacleStrictConvexCandidateResourceUsageV2(
            domain_operations=self.resource_usage.domain_operations,
            so2_atomic_steps=self.resource_usage.so2_atomic_steps,
            candidate_cells=self.resource_usage.candidate_cells,
        )
        remaining = _canonical_ids(self.remaining_constraint_ids)
        if self.visibility_constraint_id in remaining:
            raise ValueError("compiled visibility cannot remain in the suffix")
        if type(self.unsat_prefix_eligible) is not bool:
            raise TypeError("unsat_prefix_eligible must be an exact bool")
        expected_unsat = not outer.cells and not remaining
        if self.unsat_prefix_eligible is not expected_unsat:
            raise ValueError(
                "visibility UNSAT eligibility does not match the final outer"
            )
        object.__setattr__(self, "inner_allowed", inner)
        object.__setattr__(self, "outer_allowed", outer)
        object.__setattr__(self, "outer_cell_metrics", metrics)
        object.__setattr__(self, "resource_usage", usage)
        object.__setattr__(self, "remaining_constraint_ids", remaining)

    @property
    def stage_sha256(self) -> str:
        return hashlib.sha256(
            _STAGE_HASH_DOMAIN_V2 + _artifact_bytes(self)  # type: ignore[arg-type]
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _CompleteContinuousYawCandidateOutcomeV2:
    kind: _CompleteContinuousYawCandidateKindV2
    stage: _CompleteContinuousYawCandidateStageV2 | None = None
    finding_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.kind) is not _CompleteContinuousYawCandidateKindV2:
            raise TypeError("kind must be an exact continuous-yaw visibility kind")
        findings = _require_finding_codes(self.finding_codes)
        object.__setattr__(self, "finding_codes", findings)
        if self.kind is _CompleteContinuousYawCandidateKindV2.STAGE:
            if type(self.stage) is not _CompleteContinuousYawCandidateStageV2:
                raise ValueError("STAGE requires one exact visibility stage")
            if findings:
                raise ValueError("STAGE cannot carry findings")
            object.__setattr__(self, "stage", _copy_complete_stage(self.stage))
            return
        if self.stage is not None or not findings:
            raise ValueError("visibility failure requires findings and no stage")


@dataclass(frozen=True, slots=True)
class _ProjectionGeometryV2:
    box: OrientedUprightBoxBoundsV2
    corner_boxes: tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...
    ]
    camera_center_z_lower: Fraction
    camera_center_z_upper: Fraction
    near_face_image_area_fraction: Fraction


@dataclass(frozen=True, slots=True)
class _CellClassificationV2:
    kind: _ProjectionClassV2
    metrics: _VisibilityCellMetricBoundsV2 | None


class _UnsupportedVisibilityProjectionV2(ValueError):
    def __init__(self, finding_code: str) -> None:
        self.finding_code = finding_code
        super().__init__(finding_code)


def _compile_complete_continuous_yaw_candidate_v2(
    problem: SemanticProblemV2_2,
    target_stage: _TargetAwareCandidateStageV2,
    *,
    atomic_budget: SO2AtomicBudgetV2,
    intersection_budget: StrictConvexIntersectionBudgetV2,
) -> _CompleteContinuousYawCandidateOutcomeV2:
    """Complete the T16 candidate by classifying each cell in one fixed view."""

    try:
        checked_problem, checked_target = _strict_inputs(
            problem,
            target_stage,
            atomic_budget,
            intersection_budget,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            constraint, camera, subject, visual = _extract_visibility_subset(
                checked_problem,
                checked_target,
                intersection_budget,
            )
            _reserve_cell_work(checked_target, intersection_budget)
            geometry = _projection_geometry(
                subject,
                visual,
                camera,
                atomic_budget,
            )
            inner_cells = tuple(
                cell
                for cell in checked_target.inner_allowed.cells
                if _classify_cell(
                    cell,
                    camera,
                    constraint,
                    geometry,
                ).kind
                is _ProjectionClassV2.PASS
            )
            outer_pairs = tuple(
                (cell, _classify_cell(cell, camera, constraint, geometry))
                for cell in checked_target.outer_allowed.cells
            )
            outer_cells = tuple(
                cell
                for cell, result in outer_pairs
                if result.kind is not _ProjectionClassV2.FAIL
            )
            outer_metrics = tuple(
                result.metrics
                for _, result in outer_pairs
                if result.kind is not _ProjectionClassV2.FAIL
            )
            if any(metric is None for metric in outer_metrics):
                raise RuntimeError("kept visibility cell lost its metric interval")
            metrics = tuple(metric for metric in outer_metrics if metric is not None)
            intersection_budget.consume_candidate_cells(
                len(inner_cells) + len(outer_cells)
            )
            inner = _filtered_complex(checked_target.inner_allowed, inner_cells)
            outer = _filtered_complex(checked_target.outer_allowed, outer_cells)
            remaining = tuple(
                item
                for item in checked_target.remaining_constraint_ids
                if item != constraint.constraint_id
            )
            intersection_budget.consume_domain(
                18
                + len(metrics) * 7
                + sum(
                    len(cell.half_planes) + len(cell.closure_polygon.vertices_ccw)
                    for complex_ in (inner, outer)
                    for cell in complex_.cells
                )
            )
            stage = _CompleteContinuousYawCandidateStageV2(
                semantic_problem_sha256=checked_problem.semantic_problem_sha256,
                compiler_config_sha256=checked_target.compiler_config_sha256,
                upstream_t15_artifact_sha256=(
                    checked_target.upstream_t15_artifact_sha256
                ),
                upstream_target_stage_sha256=checked_target.stage_sha256,
                subject_id=checked_target.subject_id,
                visibility_constraint_id=constraint.constraint_id,
                inner_allowed=inner,
                outer_allowed=outer,
                outer_cell_metrics=metrics,
                resource_usage=MultiObstacleStrictConvexCandidateResourceUsageV2(
                    domain_operations=intersection_budget.domain_operations_used,
                    so2_atomic_steps=atomic_budget.used,
                    candidate_cells=intersection_budget.candidate_cells_used,
                ),
                remaining_constraint_ids=remaining,
                unsat_prefix_eligible=not outer.cells and not remaining,
            )
        return _CompleteContinuousYawCandidateOutcomeV2(
            kind=_CompleteContinuousYawCandidateKindV2.STAGE,
            stage=stage,
        )
    except _InvalidInputV2:
        return _failure(
            _CompleteContinuousYawCandidateKindV2.INVALID_INPUT,
            "INVALID_INPUT:CONTINUOUS_YAW_VISIBILITY",
        )
    except _UnsupportedVisibilityProjectionV2 as error:
        return _failure(
            _CompleteContinuousYawCandidateKindV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (StrictConvexIntersectionBudgetExhaustedV2, SO2AtomicBudgetExhaustedV2):
        return _failure(
            _CompleteContinuousYawCandidateKindV2.RESOURCE_LIMIT,
            "RESOURCE_LIMIT:CONTINUOUS_YAW_VISIBILITY",
        )
    except (ArithmeticError, RuntimeWarning):
        return _failure(
            _CompleteContinuousYawCandidateKindV2.NUMERIC_GAP,
            "NUMERIC_GAP:CONTINUOUS_YAW_VISIBILITY",
        )


def _strict_inputs(
    problem: object,
    target_stage: object,
    atomic_budget: object,
    intersection_budget: object,
) -> tuple[SemanticProblemV2_2, _TargetAwareCandidateStageV2]:
    if type(atomic_budget) is not SO2AtomicBudgetV2:
        raise TypeError("atomic_budget must be SO2AtomicBudgetV2")
    atomic_budget.validate()
    if type(intersection_budget) is not StrictConvexIntersectionBudgetV2:
        raise TypeError("intersection_budget must be StrictConvexIntersectionBudgetV2")
    if type(target_stage) is not _TargetAwareCandidateStageV2:
        raise TypeError("target_stage must be the exact T16 stage type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        checked_problem = _strict_problem(problem)
        checked_target = _copy_target_stage(target_stage)
    if (
        checked_target.semantic_problem_sha256
        != checked_problem.semantic_problem_sha256
    ):
        raise ValueError("T16 stage problem hash is not closed")
    usage = checked_target.resource_usage
    if (
        atomic_budget.used != usage.so2_atomic_steps
        or intersection_budget.domain_operations_used != usage.domain_operations
        or intersection_budget.candidate_cells_used != usage.candidate_cells
    ):
        raise ValueError("live visibility ledgers must continue exact T16 usage")
    visibility_ids = tuple(
        item.constraint_id
        for item in checked_problem.constraints.visibility_constraints
    )
    if checked_target.remaining_constraint_ids != visibility_ids:
        raise ValueError("T16 suffix must contain exactly the visibility constraint")
    return checked_problem, checked_target


def _extract_visibility_subset(
    problem: SemanticProblemV2_2,
    target_stage: _TargetAwareCandidateStageV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> tuple[
    VisibilityConstraint,
    PinholeCameraV2_2,
    CanonicalObjectV2_2,
    GeometryInstanceV2_2,
]:
    if (
        len(problem.constraints.visibility_constraints) != 1
        or problem.numeric_policy != NumericPolicyV2()
    ):
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_POLICY"
        )
    constraint = problem.constraints.visibility_constraints[0]
    if (
        constraint.constraint_id != target_stage.remaining_constraint_ids[0]
        or constraint.threshold_boundary_policy is not BoundaryPolicy.CLOSED
        or constraint.mask_policy is not VisibilityMaskPolicy.FULL_OBJECT
        or constraint.occluder_soundness_policy
        is not OccluderSoundnessPolicy.EXACT_OR_OUTER_SHAPE_BOUND
        or constraint.occluder_geometry_ids
        or constraint.accepted_baseline_completeness != (FactCompletenessV2.EXACT,)
        or target_stage.subject_id not in constraint.query_object_ids
        or _visibility_semantics_findings(problem, constraint)
    ):
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_POLICY"
        )

    objects = _exact_values(problem.scene.objects, "OBJECTS", budget)
    geometries = _exact_values(
        problem.scene.geometry_instances,
        "GEOMETRY_INSTANCES",
        budget,
    )
    cameras = _exact_values(problem.scene.cameras, "CAMERAS", budget)
    _require_baseline_pass(problem, constraint, budget)
    if len(cameras) != 1 or type(cameras[0]) is not PinholeCameraV2_2:
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_SINGLE_CAMERA"
        )
    camera = cameras[0]
    if camera.camera_id != constraint.camera_id:
        raise RuntimeError("visibility constraint lost its camera reference")
    _require_camera_subset(camera)

    object_by_id = {item.object_id: item for item in objects}
    try:
        subject = object_by_id[target_stage.subject_id]
    except KeyError as error:
        raise RuntimeError("visibility semantic graph lost the subject") from error
    if type(subject) is not CanonicalObjectV2_2 or not subject.movable:
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_SUBJECT"
        )
    visual_geometries = tuple(
        item
        for item in geometries
        if item.owner_object_id == subject.object_id
        and item.role is GeometryRoleV2.VISUAL
    )
    if len(visual_geometries) != 1:
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISUAL_GEOMETRY_CARDINALITY"
        )
    visual = visual_geometries[0]
    anchor = visual.anchor_from_geometry
    if (
        type(visual) is not GeometryInstanceV2_2
        or visual.approximation is not GeometryApproximationV2.EXACT
        or visual.uncertainty != UncertaintyBudgetV2()
        or type(visual.shape) is not UprightBox3DV2
        or anchor.yaw_radians != 0.0
        or (anchor.translation.x, anchor.translation.y) != (0.0, 0.0)
        or any(item.role is GeometryRoleV2.OCCLUDER for item in geometries)
    ):
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISUAL_GEOMETRY"
        )
    return constraint, camera, subject, visual


def _require_camera_subset(camera: PinholeCameraV2_2) -> None:
    transform = camera.world_to_camera
    intrinsics = camera.intrinsics_row_major
    if (
        type(transform) is not DirectedYawIntervalTransformV2_2
        or transform.yaw_radians != 0.0
        or camera.distortion_model is not CameraDistortionModel.NONE
        or camera.calibration_uncertainty != UncertaintyBudgetV2()
        or camera.matrix_layout is not CameraMatrixLayout.ROW_MAJOR
        or camera.camera_axes is not CameraAxes.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConvention.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConvention.POSITIVE_Z_FORWARD
        or intrinsics[1] != 0.0
        or intrinsics[3] != 0.0
        or intrinsics[6:] != (0.0, 0.0, 1.0)
    ):
        raise _UnsupportedVisibilityProjectionV2(
            "UNSUPPORTED_MODEL:CONTINUOUS_YAW_CAMERA"
        )


def _exact_values(facts: object, label: str, budget: StrictConvexIntersectionBudgetV2):
    if (
        getattr(facts, "availability", None) is not FactAvailabilityV2.KNOWN
        or getattr(facts, "completeness", None) is not FactCompletenessV2.EXACT
        or getattr(facts, "uncertainty", None) != UncertaintyBudgetV2()
        or type(getattr(facts, "values", None)) is not tuple
        or getattr(facts, "inner_values", None) is not None
        or getattr(facts, "outer_values", None) is not None
    ):
        raise _UnsupportedVisibilityProjectionV2(
            f"UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_{label}"
        )
    values = facts.values
    budget.consume_domain(len(values) + 1)
    return values


def _require_baseline_pass(
    problem: SemanticProblemV2_2,
    constraint: VisibilityConstraint,
    budget: StrictConvexIntersectionBudgetV2,
) -> None:
    observations = _exact_values(
        problem.scene.baseline_observations,
        "BASELINE_OBSERVATIONS",
        budget,
    )
    by_key = {
        (
            item.object_id,
            item.camera_id,
            item.metric_definition_id,
            item.metric_definition_version,
        ): item
        for item in observations
        if type(item) is BaselineObservation
    }
    specifications = (
        (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            Fraction.from_float(constraint.minimum_visible_fraction),
            True,
        ),
        (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            Fraction.from_float(constraint.minimum_image_area_fraction),
            True,
        ),
        (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            Fraction.from_float(constraint.maximum_truncated_fraction),
            False,
        ),
    )
    budget.consume_domain(len(constraint.query_object_ids) * len(specifications) + 2)
    for object_id in constraint.query_object_ids:
        for definition_id, version, threshold, minimum in specifications:
            observation = by_key.get(
                (object_id, constraint.camera_id, definition_id, version)
            )
            if observation is None:
                raise _UnsupportedVisibilityProjectionV2(
                    "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_BASELINE"
                )
            lower = Fraction.from_float(observation.normalized_lower_bound)
            upper = Fraction.from_float(observation.normalized_upper_bound)
            proven = lower >= threshold if minimum else upper <= threshold
            if not proven:
                raise _UnsupportedVisibilityProjectionV2(
                    "UNSUPPORTED_MODEL:CONTINUOUS_YAW_VISIBILITY_BASELINE"
                )


def _reserve_cell_work(
    target_stage: _TargetAwareCandidateStageV2,
    budget: StrictConvexIntersectionBudgetV2,
) -> None:
    cells = target_stage.inner_allowed.cells + target_stage.outer_allowed.cells
    budget.consume_domain(
        8
        + sum(
            24 + len(cell.half_planes) + 8 * len(cell.closure_polygon.vertices_ccw)
            for cell in cells
        )
    )


def _projection_geometry(
    subject: CanonicalObjectV2_2,
    visual: GeometryInstanceV2_2,
    camera: PinholeCameraV2_2,
    budget: SO2AtomicBudgetV2,
) -> _ProjectionGeometryV2:
    transform = subject.pose.world_from_object
    if type(transform) is not DirectedYawIntervalTransformV2_2:
        raise RuntimeError("continuous-yaw subject lost its directed transform")
    assert type(visual.shape) is UprightBox3DV2
    outcome = compile_oriented_upright_box_bounds_v2(
        transform,
        visual.shape,
        atomic_budget=budget,
    )
    if outcome.kind is SO2IntervalKindV2.RESOURCE_LIMIT:
        raise SO2AtomicBudgetExhaustedV2
    if outcome.kind is SO2IntervalKindV2.NUMERIC_GAP:
        raise ArithmeticError("continuous-yaw visual box numeric gap")
    if outcome.kind is not SO2IntervalKindV2.EXACT or outcome.bounds is None:
        raise RuntimeError("supported visual box did not produce directed bounds")
    box = outcome.bounds
    corners = _directed_world_corner_boxes(box, budget)
    camera_shift = camera.world_to_camera.translation
    center_z = (
        Fraction.from_float(transform.translation.z)
        + Fraction.from_float(visual.anchor_from_geometry.translation.z)
        + Fraction.from_float(camera_shift.z)
    )
    z_lower = center_z - box.half_extent_z
    z_upper = center_z + box.half_extent_z
    for value in (center_z, z_lower, z_upper):
        so2_interval._require_fraction_cap(value)
    if z_lower > 0:
        fx = Fraction.from_float(camera.intrinsics_row_major[0])
        fy = Fraction.from_float(camera.intrinsics_row_major[4])
        image_area = (
            fx
            * fy
            * Fraction.from_float(visual.shape.size_m.x)
            * Fraction.from_float(visual.shape.size_m.y)
            / (
                z_lower
                * z_lower
                * Fraction(camera.width_px)
                * Fraction(camera.height_px)
            )
        )
        image_area = min(Fraction(1), image_area)
    else:
        image_area = Fraction(0)
    so2_interval._require_fraction_cap(image_area)
    return _ProjectionGeometryV2(
        box=box,
        corner_boxes=corners,
        camera_center_z_lower=z_lower,
        camera_center_z_upper=z_upper,
        near_face_image_area_fraction=image_area,
    )


def _classify_cell(
    cell: StrictConvexIntersectionCellV2,
    camera: PinholeCameraV2_2,
    constraint: VisibilityConstraint,
    geometry: _ProjectionGeometryV2,
) -> _CellClassificationV2:
    vertices = cell.closure_polygon.vertices_ccw
    delta_x = (min(item.x for item in vertices), max(item.x for item in vertices))
    delta_y = (min(item.y for item in vertices), max(item.y for item in vertices))
    z_lower = geometry.camera_center_z_lower
    z_upper = geometry.camera_center_z_upper
    near = Fraction.from_float(camera.near_clip_m)
    far = Fraction.from_float(camera.far_clip_m)
    depth_disjoint = z_upper < near or z_lower > far or z_upper <= 0
    visibility_required = (
        constraint.minimum_visible_fraction > 0.0
        or constraint.minimum_image_area_fraction > 0.0
        or constraint.maximum_truncated_fraction < 1.0
    )
    if depth_disjoint and visibility_required:
        return _CellClassificationV2(kind=_ProjectionClassV2.FAIL, metrics=None)
    if z_lower <= 0:
        return _CellClassificationV2(
            kind=_ProjectionClassV2.AMBIGUOUS,
            metrics=_ambiguous_metrics(cell.cell_id),
        )

    projected_u: list[tuple[Fraction, Fraction]] = []
    projected_v: list[tuple[Fraction, Fraction]] = []
    fx = Fraction.from_float(camera.intrinsics_row_major[0])
    fy = Fraction.from_float(camera.intrinsics_row_major[4])
    cx = Fraction.from_float(camera.intrinsics_row_major[2])
    cy = Fraction.from_float(camera.intrinsics_row_major[5])
    camera_shift = camera.world_to_camera.translation
    shift_x = Fraction.from_float(camera_shift.x)
    shift_y = Fraction.from_float(camera_shift.y)
    for corner_x, corner_y in geometry.corner_boxes:
        x_interval = (
            corner_x[0] + delta_x[0] + shift_x,
            corner_x[1] + delta_x[1] + shift_x,
        )
        y_interval = (
            corner_y[0] + delta_y[0] + shift_y,
            corner_y[1] + delta_y[1] + shift_y,
        )
        for z in (z_lower, z_upper):
            u = (fx * x_interval[0] / z + cx, fx * x_interval[1] / z + cx)
            v = (fy * y_interval[0] / z + cy, fy * y_interval[1] / z + cy)
            projected_u.append((min(u), max(u)))
            projected_v.append((min(v), max(v)))
    for interval in (*projected_u, *projected_v):
        so2_interval._require_fraction_cap(interval[0])
        so2_interval._require_fraction_cap(interval[1])

    min_u = min(item[0] for item in projected_u)
    max_u = max(item[1] for item in projected_u)
    min_v = min(item[0] for item in projected_v)
    max_v = max(item[1] for item in projected_v)
    left = Fraction(1, 2)
    right = Fraction(camera.width_px) - left
    top = Fraction(1, 2)
    bottom = Fraction(camera.height_px) - top
    outside = max_u <= left or min_u >= right or max_v <= top or min_v >= bottom
    if outside and visibility_required:
        return _CellClassificationV2(kind=_ProjectionClassV2.FAIL, metrics=None)
    fully_contained = (
        z_lower >= near
        and z_upper <= far
        and min_u >= left
        and max_u <= right
        and min_v >= top
        and max_v <= bottom
    )
    area = geometry.near_face_image_area_fraction
    thresholds_pass = (
        Fraction(1) >= Fraction.from_float(constraint.minimum_visible_fraction)
        and area >= Fraction.from_float(constraint.minimum_image_area_fraction)
        and Fraction(0) <= Fraction.from_float(constraint.maximum_truncated_fraction)
    )
    if fully_contained and thresholds_pass:
        return _CellClassificationV2(
            kind=_ProjectionClassV2.PASS,
            metrics=_VisibilityCellMetricBoundsV2(
                cell_id=cell.cell_id,
                visible_fraction_lower=Fraction(1),
                visible_fraction_upper=Fraction(1),
                image_area_fraction_lower=area,
                image_area_fraction_upper=Fraction(1),
                truncated_fraction_lower=Fraction(0),
                truncated_fraction_upper=Fraction(0),
            ),
        )
    return _CellClassificationV2(
        kind=_ProjectionClassV2.AMBIGUOUS,
        metrics=_ambiguous_metrics(cell.cell_id),
    )


def _ambiguous_metrics(cell_id: str) -> _VisibilityCellMetricBoundsV2:
    return _VisibilityCellMetricBoundsV2(
        cell_id=cell_id,
        visible_fraction_lower=Fraction(0),
        visible_fraction_upper=Fraction(1),
        image_area_fraction_lower=Fraction(0),
        image_area_fraction_upper=Fraction(1),
        truncated_fraction_lower=Fraction(0),
        truncated_fraction_upper=Fraction(1),
    )


def _filtered_complex(
    source: StrictConvexIntersectionComplexV2,
    cells: tuple[StrictConvexIntersectionCellV2, ...],
) -> StrictConvexIntersectionComplexV2:
    return StrictConvexIntersectionComplexV2(
        cells=cells,
        universe=source.universe,
        topology=source.topology,
    )


def _copy_metric(value: _VisibilityCellMetricBoundsV2) -> _VisibilityCellMetricBoundsV2:
    if type(value) is not _VisibilityCellMetricBoundsV2:
        raise TypeError("visibility metric bounds have the wrong exact type")
    return _VisibilityCellMetricBoundsV2(
        cell_id=value.cell_id,
        visible_fraction_lower=value.visible_fraction_lower,
        visible_fraction_upper=value.visible_fraction_upper,
        image_area_fraction_lower=value.image_area_fraction_lower,
        image_area_fraction_upper=value.image_area_fraction_upper,
        truncated_fraction_lower=value.truncated_fraction_lower,
        truncated_fraction_upper=value.truncated_fraction_upper,
    )


def _copy_complete_stage(
    value: _CompleteContinuousYawCandidateStageV2,
) -> _CompleteContinuousYawCandidateStageV2:
    return _CompleteContinuousYawCandidateStageV2(
        semantic_problem_sha256=value.semantic_problem_sha256,
        compiler_config_sha256=value.compiler_config_sha256,
        upstream_t15_artifact_sha256=value.upstream_t15_artifact_sha256,
        upstream_target_stage_sha256=value.upstream_target_stage_sha256,
        subject_id=value.subject_id,
        visibility_constraint_id=value.visibility_constraint_id,
        inner_allowed=value.inner_allowed,
        outer_allowed=value.outer_allowed,
        outer_cell_metrics=value.outer_cell_metrics,
        resource_usage=value.resource_usage,
        remaining_constraint_ids=value.remaining_constraint_ids,
        unsat_prefix_eligible=value.unsat_prefix_eligible,
    )


def _canonical_ids(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise ValueError("remaining constraint IDs must be exact non-blank strings")
    if len(value) != len(set(value)):
        raise ValueError("remaining constraint IDs must be unique")
    return tuple(sorted(value))


def _require_digest(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _failure(
    kind: _CompleteContinuousYawCandidateKindV2,
    finding_code: str,
) -> _CompleteContinuousYawCandidateOutcomeV2:
    return _CompleteContinuousYawCandidateOutcomeV2(
        kind=kind,
        finding_codes=(finding_code,),
    )


__all__ = (
    "CONTINUOUS_YAW_VISIBILITY_KERNEL_ID_V2",
    "_CompleteContinuousYawCandidateKindV2",
    "_CompleteContinuousYawCandidateOutcomeV2",
    "_CompleteContinuousYawCandidateStageV2",
    "_VisibilityCellMetricBoundsV2",
    "_compile_complete_continuous_yaw_candidate_v2",
)
