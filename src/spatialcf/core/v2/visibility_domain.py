"""Sound visibility-domain compilation for a small analytic Canonical v2 subset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from spatialcf.core.v2.rect_kernel import (
    ExactAxisAlignedRectV2,
    RectCoordinateSpaceV2,
)
from spatialcf.core.v2.rectilinear_kernel import (
    ExactRectilinearRegionV2,
    RectilinearAtomicBudgetV2,
    RectilinearOutcomeKindV2,
    RectilinearRegionOutcomeV2,
    RectilinearTopologyV2,
    intersect_rectilinear_regions_v2,
    normalize_rectilinear_region_v2,
    union_rectilinear_regions_v2,
)
from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    QuaternionV2,
    RigidTransformV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.v2.constraints import (
    BoundaryPolicyV2,
    OccluderSoundnessPolicyV2,
    VisibilityAreaMeasureV2,
    VisibilityConstraintV2,
    VisibilityDepthPolicyV2,
    VisibilityMaskPolicyV2,
    VisibilityMetricFormulaV2,
    VisibilityMetricKindV2,
)
from spatialcf.domain.v2.geometry import (
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    UprightBox3DV2,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.scene import (
    CameraAxesV2,
    CameraDepthConventionV2,
    CameraDistortionModelV2,
    CameraMatrixLayoutV2,
    CameraPixelConventionV2,
    CanonicalObjectV2,
    PinholeCameraV2,
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
    constraint: VisibilityConstraintV2 | str,
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
    constraint: VisibilityConstraintV2,
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
    constraint: VisibilityConstraintV2,
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
    requested: VisibilityConstraintV2 | str,
) -> VisibilityConstraintV2 | VisibilityDomainOutcomeV2:
    if isinstance(requested, VisibilityConstraintV2):
        checked = VisibilityConstraintV2.model_validate(
            requested.model_dump(mode="python"),
            strict=True,
        )
        constraint_id = checked.constraint_id
    elif type(requested) is str:
        checked = None
        constraint_id = requested
    else:
        raise TypeError("constraint must be a VisibilityConstraintV2 or exact str ID")
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
    constraint: VisibilityConstraintV2,
) -> tuple[str, ...]:
    findings: list[str] = []
    if constraint.threshold_boundary_policy is not BoundaryPolicyV2.CLOSED:
        findings.append(
            "UNSUPPORTED_VISIBILITY_DOMAIN:BOUNDARY_POLICY:"
            f"{constraint.threshold_boundary_policy.value}"
        )
    if constraint.mask_policy is not VisibilityMaskPolicyV2.FULL_OBJECT:
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:MASK_POLICY:{constraint.mask_policy.value}"
        )
    if (
        constraint.occluder_soundness_policy
        is not OccluderSoundnessPolicyV2.EXACT_OR_OUTER_SHAPE_BOUND
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
    constraint: VisibilityConstraintV2,
    *,
    supported_image_area_formulas: tuple[VisibilityMetricFormulaV2, ...] = (
        VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA,
    ),
) -> tuple[str, ...]:
    expected = {
        VisibilityMetricKindV2.VISIBLE_FRACTION: (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
            (
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA,
            ),
        ),
        VisibilityMetricKindV2.IMAGE_AREA_FRACTION: (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
            supported_image_area_formulas,
        ),
        VisibilityMetricKindV2.TRUNCATED_FRACTION: (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
            (
                VisibilityMetricFormulaV2.ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA,
            ),
        ),
    }
    findings: list[str] = []
    for definition in problem.visibility_semantics.definitions:
        definition_id, version, formulas = expected[definition.kind]
        if (
            definition.reference != (definition_id, version)
            or definition.formula not in formulas
            or definition.area_measure
            is not VisibilityAreaMeasureV2.CONTINUOUS_PIXEL_PLANE_AREA
            or definition.depth_policy
            is not VisibilityDepthPolicyV2.NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES
        ):
            findings.append(
                f"UNSUPPORTED_VISIBILITY_DOMAIN:METRIC_SEMANTICS:{definition.kind.value}"
            )
    return tuple(findings)


def _fixed_subset_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraintV2,
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
    constraint: VisibilityConstraintV2,
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


def _camera_findings(camera: PinholeCameraV2) -> tuple[str, ...]:
    findings: list[str] = []
    if not _is_identity_transform(camera.world_to_camera):
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:CAMERA_TRANSFORM:{camera.camera_id}"
        )
    if camera.distortion_model is not CameraDistortionModelV2.NONE:
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
        camera.matrix_layout is not CameraMatrixLayoutV2.ROW_MAJOR
        or camera.camera_axes is not CameraAxesV2.X_RIGHT_Y_DOWN_Z_FORWARD
        or camera.pixel_convention is not CameraPixelConventionV2.CENTER_AT_HALF
        or camera.depth_convention is not CameraDepthConventionV2.POSITIVE_Z_FORWARD
    ):
        findings.append(
            f"UNSUPPORTED_VISIBILITY_DOMAIN:CAMERA_CONVENTION:{camera.camera_id}"
        )
    return tuple(findings)


def _baseline_family_findings(
    problem: SemanticProblemV2,
    constraint: VisibilityConstraintV2,
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
    constraint: VisibilityConstraintV2,
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
    constraint: VisibilityConstraintV2,
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
    subject: CanonicalObjectV2,
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


def _is_identity_rotation(rotation: QuaternionV2) -> bool:
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
