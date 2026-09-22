"""Standalone public catalog input and installed semantic dataset roundtrip."""

from __future__ import annotations

import spatialcf.domain.compatibility as compatibility
from spatialcf.domain.artifacts import StrictConvexCandidateCompilerConfigV2_7
from spatialcf.domain.base import NumericPolicyV2
from spatialcf.domain.constraints import CanonicalConstraintSet
from spatialcf.domain.objective import ObjectiveSpecV2
from spatialcf.domain.problem import CanonicalSceneV2_3, SemanticProblemV2_3
from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    SemanticsProfile,
    SolverBackendDescriptor,
)
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9

def _sorted(*values: str) -> tuple[str, ...]:
    return tuple(sorted(values, key=canonical_json_bytes))

def planar_field_mapping() -> compatibility.PlanarTranslateFieldMapping:
    """Build the registered compatibility field and kernel closure."""

    return compatibility.PlanarTranslateFieldMapping.seal(
        mapping_definition_ref=compatibility.PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
        problem_field_names=tuple(SemanticProblemV2_3.model_fields),
        scene_field_names=tuple(CanonicalSceneV2_3.model_fields),
        constraint_field_names=tuple(CanonicalConstraintSet.model_fields),
        objective_field_names=tuple(ObjectiveSpecV2.model_fields),
        numeric_policy_field_names=tuple(NumericPolicyV2.model_fields),
        solver_config_field_names=tuple(ContinuousYawSolverConfigV2_9.model_fields),
        candidate_config_field_names=tuple(
            StrictConvexCandidateCompilerConfigV2_7.model_fields
        ),
        algorithm_kernel_bindings=tuple(
            compatibility.PlanarTranslateAlgorithmKernelBinding(
                field_name=field_name,
                registered_value=registered_value,
            )
            for field_name, registered_value in (
                ("algorithm_id", "solver:canonical-branch-and-bound-v2"),
                ("algorithm_version", "algorithm:2.9"),
                (
                    "candidate_config.algorithm_id",
                    "solver:canonical-branch-and-bound-v2",
                ),
                ("candidate_config.algorithm_version", "algorithm:2.7"),
                (
                    "candidate_config.so2_kernel_id",
                    "geometry-kernel:rational-so2-upright-box-directed-v2",
                ),
                (
                    "candidate_config.so2_kernel_version",
                    "kernel:2.2-continuous-yaw-upright-box",
                ),
                (
                    "candidate_config.obstacle_kernel_id",
                    "geometry-kernel:rational-convex-translation-bracket-v2",
                ),
                (
                    "candidate_config.obstacle_kernel_version",
                    "kernel:2.3-convex-translation-bracket",
                ),
                (
                    "candidate_config.partition_kernel_id",
                    "geometry-kernel:rational-convex-complement-partition-v2",
                ),
                (
                    "candidate_config.partition_kernel_version",
                    "kernel:2.4-topology-aware-convex-complement",
                ),
                (
                    "candidate_config.intersection_kernel_id",
                    "geometry-kernel:rational-strict-convex-intersection-v2",
                ),
                (
                    "candidate_config.intersection_kernel_version",
                    "kernel:2.5-strict-convex-intersection",
                ),
                (
                    "candidate_config.support_projection_kernel_id",
                    "geometry-kernel:rational-continuous-yaw-support-projection-v2",
                ),
                (
                    "candidate_config.support_projection_kernel_version",
                    "kernel:2.6-exact-horizontal-support-projection",
                ),
                (
                    "camera_frame_kernel_id",
                    "geometry-kernel:rational-upright-world-to-camera-v2.9",
                ),
                (
                    "target_projection_kernel_id",
                    "geometry-kernel:rational-continuous-yaw-directional-relation-v2.9",
                ),
                (
                    "visibility_projection_kernel_id",
                    "geometry-kernel:rational-continuous-yaw-upright-camera-visibility-v2.9",
                ),
                (
                    "objective_kernel_id",
                    "objective-kernel:rational-continuous-yaw-joint-four-term-v2.9",
                ),
            )
        ),
    )

def planar_registration() -> compatibility.PlanarTranslateProfileRegistration:
    """Build the one immutable profile declaration accepted by the compiler."""

    mapping = planar_field_mapping()
    semantics = SemanticsProfile.seal(
        semantics_profile_ref=compatibility.PLANAR_TRANSLATE_SEMANTICS_PROFILE_REF,
        accepted_scene_and_fact_schema_refs=_sorted(
            "schema:spatialcf/canonical-scene/2.3",
            "schema:spatialcf/semantic-problem/2.3",
        ),
        predicate_definition_refs=(
            compatibility.PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
        ),
        transition_semantics_refs=(
            "definition:spatialcf/planar-translate/transition-translate-xy/2.0",
        ),
        objective_definition_refs=(
            "definition:spatialcf/planar-translate/objective-current-v2/2.0",
        ),
        numeric_semantics_ref="definition:spatialcf/planar-translate/numeric-policy/2.0",
        completeness_policy_ref=(
            "definition:spatialcf/planar-translate/completeness-policy/2.0"
        ),
        uncertainty_policy_ref=(
            "definition:spatialcf/planar-translate/uncertainty-policy/2.0"
        ),
        derived_fact_rule_refs=(),
        observation_obligation_policy_ref=(
            "definition:spatialcf/planar-translate/observation-obligation-policy/2.0"
        ),
    )
    action_space = ActionSpaceProfile.seal(
        action_space_profile_ref=compatibility.PLANAR_TRANSLATE_PROFILE_REF,
        accepted_scene_schema_refs=("schema:spatialcf/canonical-scene/2.3",),
        state_variable_definition_refs=(
            "definition:spatialcf/planar-translate/state-world-xy/2.0",
        ),
        allowed_operator_refs=(
            "definition:spatialcf/planar-translate/operator-translate-xy/2.0",
        ),
        mandatory_invariant_template_refs=(
            "definition:spatialcf/planar-translate/invariants-current-v2/2.0",
        ),
        predicate_capability_refs=(
            compatibility.PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF,
        ),
        objective_capability_refs=(
            compatibility.PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,
        ),
        numeric_semantics_ref="definition:spatialcf/planar-translate/numeric-policy/2.0",
        allowed_claim_definition_refs=_sorted(
            "definition:spatialcf/planar-translate/claim-certified-solution/2.0",
            "definition:spatialcf/planar-translate/claim-proven-unsat/2.0",
            "definition:spatialcf/planar-translate/claim-unknown/2.0",
        ),
        backend_capability_requirements=_sorted(
            compatibility.PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF,
            compatibility.PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,
        ),
        adapter_capability_requirements=(),
        publication_proof_policy_ref=(
            "definition:spatialcf/planar-translate/publication-proof-policy/2.0"
        ),
    )
    descriptor = SolverBackendDescriptor.seal(
        backend_ref=compatibility.PLANAR_TRANSLATE_BACKEND_REF,
        implementation_build_sha256="a" * 64,
        supported_profile_hashes=(action_space.action_space_profile_sha256,),
        supported_predicate_capabilities=(
            compatibility.PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF,
        ),
        supported_operator_capabilities=(
            compatibility.PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF,
        ),
        supported_objective_capabilities=(
            compatibility.PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,
        ),
        supported_numeric_semantics=(
            "definition:spatialcf/planar-translate/numeric-policy/2.0",
        ),
        emitted_proof_material_definition_refs=(
            "definition:spatialcf/planar-translate/proof-material-current-v2/2.0",
        ),
        compatible_checker_capability_refs=(
            compatibility.PLANAR_TRANSLATE_CHECKER_CAPABILITY_REF,
        ),
        resource_definition_refs=(
            "definition:spatialcf/planar-translate/resource-cpu/2.0",
        ),
    )
    return compatibility.PlanarTranslateProfileRegistration.seal(
        semantics_profile=semantics,
        action_space_profile=action_space,
        backend_descriptor=descriptor,
        backend_owner_ref=compatibility.PLANAR_TRANSLATE_BACKEND_OWNER_REF,
        checker_owner_ref=compatibility.PLANAR_TRANSLATE_CHECKER_OWNER_REF,
        profile_capability_ref=compatibility.PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF,
        compiler_capability_ref=compatibility.PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF,
        solver_capability_ref=compatibility.PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,
        checker_capability_ref=compatibility.PLANAR_TRANSLATE_CHECKER_CAPABILITY_REF,
        mapping_definition_ref=compatibility.PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,
        mapping_definition_sha256=mapping.mapping_definition_sha256,
    )


def public_smoke_problem() -> SemanticProblemV2_3:
    """Two synthetic boxes, one floor, one camera and explicit fact families."""
    def xyz(x=0.0, y=0.0, z=0.0):
        return dict(x=x, y=y, z=z)

    def transform(x=0.0, y=0.0, z=0.0):
        return dict(
            kind="DIRECTED_YAW_INTERVAL", translation=xyz(x, y, z), yaw_radians=0.0
        )

    def facts(*values):
        return dict(
            availability="KNOWN", completeness="EXACT", values=values, uncertainty={}
        )

    region = {
        "components": ({
            "exterior": {
                "winding": "COUNTERCLOCKWISE",
                "vertices": tuple(
                    dict(x=x, y=y)
                    for x, y in ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0))
                ),
            },
            "holes": (),
        },),
    }
    objects = tuple(
        dict(
            object_id=f"object:{name}", category_id="category:box",
            movable=name == "subject",
            pose=dict(world_from_object=transform(x, 0.0, 0.5)),
            support_assignment=dict(availability="KNOWN", surface_id="surface:floor"),
        )
        for name, x in (("subject", -1.0), ("reference", 1.0))
    )
    geometries = [
        dict(
            geometry_id=f"geometry:{name}:{role.lower()}",
            owner_object_id=f"object:{name}", role=role,
            anchor_from_geometry=transform(), approximation="EXACT", uncertainty={},
            shape=dict(shape_type="UPRIGHT_BOX_3D", size_m=xyz(0.5, 0.5, 1.0)),
        )
        for name in ("subject", "reference")
        for role in ("VISUAL", "RELATION", "COLLISION", "SUPPORT")
    ]
    geometries.append(dict(
        geometry_id="geometry:floor", owner_object_id=None, role="COLLISION",
        anchor_from_geometry=transform(0.0, 0.0, -0.25),
        approximation="EXACT", uncertainty={},
        shape=dict(shape_type="UPRIGHT_BOX_3D", size_m=xyz(4.0, 4.0, 0.5)),
    ))
    bodies = [
        dict(
            body_id=f"body:{name}", owner_object_id=f"object:{name}",
            geometry_instance_ids=(f"geometry:{name}:collision",),
        )
        for name in ("subject", "reference")
    ]
    bodies.append(dict(
        body_id="body:floor", owner_object_id=None,
        geometry_instance_ids=("geometry:floor",),
    ))
    metrics = (
        ("VISIBLE_FRACTION", "visible-surface-fraction",
         "VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA", 1.0),
        ("IMAGE_AREA_FRACTION", "image-area-fraction",
         "VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA", 0.01),
        ("TRUNCATED_FRACTION", "truncated-fraction",
         "ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA", 0.0),
    )
    observations = tuple(
        dict(
            observation_id=f"observation:{name}:{metric}",
            object_id=f"object:{name}", camera_id="camera:main",
            metric_definition_id=f"visibility:{metric}",
            metric_definition_version="definition:1", normalized_value=value,
            normalized_lower_bound=value, normalized_upper_bound=value,
        )
        for name in ("subject", "reference") for _, metric, _, value in metrics
    )
    scene = dict(
        scene_id="scene:public-semantic-smoke", objects=facts(*objects),
        geometry_instances=facts(*geometries), collision_bodies=facts(*bodies),
        workspace_boundaries=facts(dict(
            fact_id="workspace:floor", region_world_xy=region, boundary_policy="CLOSED",
            region_approximation="EXACT", geometry_uncertainty={},
        )),
        known_free_spaces=facts(),
        support_surfaces=facts(dict(
            surface_id="surface:floor", owner_object_id=None,
            supporting_body_id="body:floor", anchor_from_surface=transform(),
            normal_in_anchor=xyz(z=1.0), region_uv=region,
            region_approximation="EXACT", boundary_policy="CLOSED", geometry_uncertainty={},
        )),
        cameras=facts(dict(
            camera_id="camera:main", width_px=128, height_px=128,
            intrinsics_row_major=(64.0, 0.0, 64.0, 0.0, 64.0, 64.0, 0.0, 0.0, 1.0),
            world_to_camera=dict(
                kind="UPRIGHT_WORLD_TO_CAMERA", azimuth_radians=0.0,
                translation=xyz(z=10.0),
            ),
            near_clip_m=0.1, far_clip_m=100.0, calibration_uncertainty={},
        )),
        baseline_observations=facts(*observations),
    )
    visibility = dict(
        constraint_id="constraint:visibility", visibility_semantics_id="visibility-semantics:smoke",
        camera_id="camera:main", query_object_ids=("object:subject", "object:reference"),
        visible_fraction_metric_definition_id="visibility:visible-surface-fraction",
        visible_fraction_metric_definition_version="definition:1",
        image_area_metric_definition_id="visibility:image-area-fraction",
        image_area_metric_definition_version="definition:1",
        truncated_fraction_metric_definition_id="visibility:truncated-fraction",
        truncated_fraction_metric_definition_version="definition:1",
        mask_policy="FULL_OBJECT", occluder_soundness_policy="EXACT_OR_OUTER_SHAPE_BOUND",
        minimum_visible_fraction=0.0, minimum_image_area_fraction=0.0,
        maximum_truncated_fraction=1.0, threshold_boundary_policy="CLOSED",
        accepted_baseline_completeness=("EXACT",),
    )
    constraints = dict(
        constraint_set_id="constraint-set:smoke",
        allowed_edit=dict(constraint_id="constraint:edit", subject_id="object:subject"),
        position_domain=dict(
            constraint_id="constraint:position", subject_id="object:subject",
            workspace_fact_ids=("workspace:floor",), workspace_aggregation="INTERSECTION",
            region_interpretation="SUBJECT_ANCHOR_LOCUS", boundary_policy="CLOSED",
            required_completeness=("EXACT",), minimum_boundary_clearance_m=0.0,
        ),
        collision_constraints=(dict(
            constraint_id="constraint:collision",
            subject_body_ids=("body:subject",),
            obstacle_body_ids=("body:floor", "body:reference"),
            clearance_metric="SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE",
            boundary_policy="CLOSED", minimum_clearance_m=0.0,
        ),),
        support_constraints=(dict(
            constraint_id="constraint:support", supported_object_id="object:subject",
            surface_id="surface:floor",
            subject_contact_geometry_ids=("geometry:subject:support",),
            contact_feature="LOWEST_FACE_ALONG_SURFACE_NORMAL",
            contact_aggregation="UNION_ALL_SELECTED_FEATURES",
            contact_gap_min_m=0.0, contact_gap_max_m=0.0,
            overlap_metric="PROJECTED_CONTACT_UNION_INTERSECTION_AREA",
            minimum_overlap_area_m2=0.0,
            stability_metric="FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET",
            stability_margin_m=0.0, boundary_policy="CLOSED",
            assignment_policy="EXACT_SURFACE",
        ),),
        visibility_constraints=(visibility,),
        target_relation=dict(
            constraint_id="constraint:target", subject_id="object:subject",
            reference_id="object:reference", relation_before="LEFT", relation_after="RIGHT",
            semantics_id="relation-semantics:smoke", camera_id="camera:main",
        ),
    )
    relation_definitions = []
    for first, second, measurement, unit, low, high in (
        ("LEFT", "RIGHT", "PROJECTED_CENTER_DELTA_X", "PIXEL", -0.1, 0.1),
        ("FRONT", "BEHIND", "CAMERA_DEPTH_DELTA", "METRE", -0.1, 0.1),
        ("NEAR", "FAR", "SHAPE_GAP_XY", "METRE", 0.5, 1.0),
    ):
        for relation, comparator, threshold in (
            (first, "LESS_THAN", low), (second, "GREATER_THAN", high),
        ):
            relation_definitions.append(dict(
                relation=relation, measurement=measurement, comparator=comparator,
                threshold=threshold, unit=unit,
                representative_point=(
                    None if unit == "METRE" and first == "NEAR"
                    else "RELATION_GEOMETRY_VOLUME_CENTROID"
                ),
                operand_order="FIRST_MINUS_SECOND", boundary_policy="CLOSED", tolerance=0.0,
                tolerance_policy="SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET",
                requires_both_visible=False,
            ))
    targets = []
    for constraint, components in (
        ("position", (("POSITION_BOUNDARY_CLEARANCE", "METRE"),)),
        ("collision", (("COLLISION_CLEARANCE", "METRE"),)),
        ("support", (
            ("SUPPORT_CONTACT_GAP_LOWER_MARGIN", "METRE"),
            ("SUPPORT_CONTACT_GAP_UPPER_MARGIN", "METRE"),
            ("SUPPORT_OVERLAP_AREA_MARGIN", "SQUARE_METRE"),
            ("SUPPORT_STABILITY_INSET_MARGIN", "METRE"),
        )),
        ("target", (("TARGET_RELATION_THRESHOLD_MARGIN", "PIXEL"),)),
        ("visibility", tuple(
            (f"VISIBILITY_{name}_MARGIN", "FRACTION")
            for name in ("VISIBLE_FRACTION", "IMAGE_AREA_FRACTION", "TRUNCATED_FRACTION")
        )),
    ):
        targets.append(dict(
            constraint_id=f"constraint:{constraint}", target_slack=0.0,
            component_aggregation="MIN_NORMALIZED_COMPONENT_MARGIN", importance=1.0,
            components=tuple(
                dict(kind=kind, unit=unit, normalizer=1.0) for kind, unit in components
            ),
        ))
    objective = dict(
        objective_id="objective:smoke", mode="NON_PRODUCTION_ABLATION",
        translation=dict(weight=1.0, normalizer_m=1.0, metric="EUCLIDEAN_L2_WORLD_XY_METRE"),
        relation_damage=dict(
            weight=0.0, normalizer=1.0, metric="PAIR_AXIS_SATISFIED_LABEL_SET_CHANGED_INDICATOR",
            aggregation="WEIGHTED_SUM", evaluation_camera_id="camera:main",
            pair_axis_weights=tuple(dict(
                key=dict(
                    first_object_id="object:reference", second_object_id="object:subject", axis=axis
                ),
                damage_weight=1.0,
            ) for axis in ("DEPTH", "DISTANCE")),
        ),
        visibility_change=dict(
            weight=0.0, normalizer=1.0, metric="ABSOLUTE_NORMALIZED_METRIC_DELTA",
            aggregation="WEIGHTED_SUM",
            object_camera_weights=(dict(
                key=dict(
                    object_id="object:subject", camera_id="camera:main",
                    metric_definition_id="visibility:image-area-fraction",
                    metric_definition_version="definition:1",
                ),
                change_weight=1.0,
            ),),
        ),
        safety_margin=dict(
            weight=0.0, normalizer=1.0,
            aggregation=dict(kind="SUM_NORMALIZED_DEFICIT", targets=tuple(targets)),
        ),
    )
    return SemanticProblemV2_3.model_validate(dict(
        scene=scene, constraints=constraints,
        relation_semantics=dict(
            semantics_id="relation-semantics:smoke", definitions=tuple(relation_definitions)
        ),
        visibility_semantics=dict(
            semantics_id="visibility-semantics:smoke",
            definitions=tuple(dict(
                kind=kind, metric_definition_id=f"visibility:{metric}",
                metric_definition_version="definition:1", formula=formula,
                area_measure="CONTINUOUS_PIXEL_PLANE_AREA",
                depth_policy="NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES",
            ) for kind, metric, formula, _ in metrics),
        ),
        objective=objective, numeric_policy={},
    ), strict=False)

def public_smoke_request():
    from spatialcf.core.compiler import compile_planar_translate_v2
    from spatialcf.core.upright_se2_compiler import compile_planar_translate_m2_q0_equivalence
    config=ContinuousYawSolverConfigV2_9(candidate_config=StrictConvexCandidateCompilerConfigV2_7(max_domain_operations=1024,max_so2_atomic_steps=1024,max_candidate_cells=8),max_objective_partition_cells=8,target_optimality_gap=0.0)
    return compile_planar_translate_m2_q0_equivalence(compile_planar_translate_v2(public_smoke_problem(),config,planar_registration())).source_solve_request


def test_installed_semantic_no_selection_roundtrip(tmp_path):
    """Public CPU routing and fresh replay preserve a genuine no-selection result.

    This covers an unsupported backend choice, not selected solve/proof replay.
    The M2 lift retains visibility metrics outside the M3 selected-owner bridge.
    """
    import hashlib
    from spatialcf.domain.contrast import (
        BackendProfile, CandidateRecordInput, PublicationPolicy,
        SemanticContrastCatalogInput, SourceByteReference, SourceFileManifest,
        SourceRecordIdentity, SourceRecordInput,
        TerminalLedger, TerminalStatus,
    )
    from spatialcf.domain.serialization import canonical_json_bytes
    from spatialcf.generation.contrast import (
        generate_semantic_contrast_dataset, verify_semantic_contrast_dataset,
    )

    request = public_smoke_request()
    scene = request.semantic_problem.scene_state
    record_bytes = canonical_json_bytes(scene)
    asset_bytes = b"public semantic source: two synthetic boxes\n"
    identity = SourceRecordIdentity(
        dataset_id="dataset:public-smoke", revision_id="revision:one",
        record_id="record:two-boxes",
    )
    source = SourceRecordInput.seal(
        identity=identity,
        source_record_bytes=SourceByteReference(
            relative_path="source.json", byte_length=len(record_bytes),
            byte_sha256=hashlib.sha256(record_bytes).hexdigest(),
        ),
        source_files=SourceFileManifest.seal(files=(SourceByteReference(
            relative_path="asset.txt", byte_length=len(asset_bytes),
            byte_sha256=hashlib.sha256(asset_bytes).hexdigest(),
        ),)),
        scene_state=scene, scene_state_sha256=scene.scene_state_sha256,
        source_group=scene.base_scene_payload.scene_id,
    )
    candidate = CandidateRecordInput.seal(
        candidate_id="candidate:public-smoke", source_identity=identity,
        solve_request=request, solve_request_sha256=request.solve_request_sha256,
        # The input is cardinal-only; continuous selection must honestly decline.
        backend_profile=BackendProfile.CONTINUOUS,
    )
    catalog = SemanticContrastCatalogInput.seal(
        sources=(source,), candidates=(candidate,),
        publication_policy=PublicationPolicy.seal(
            maximum_catalog_size=1,
            eligibility_claim_definition_ref="definition:spatialcf/semantic-contrast/opposite-target/1.0",
        ),
    )
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "source.json").write_bytes(record_bytes)
    (inputs / "asset.txt").write_bytes(asset_bytes)
    path = inputs / "catalog.json"
    path.write_bytes(canonical_json_bytes(catalog))
    output = tmp_path / "dataset"
    report = generate_semantic_contrast_dataset(path, output)
    assert report.candidate_count == 1
    assert report.policy_rejected_count == 0
    assert report.pair_count == report.proven_unsat_count == 0
    assert report.noncertified_witness_count == 0
    assert report.unknown_count == 1
    assert verify_semantic_contrast_dataset(output) == report
    ledger = TerminalLedger.model_validate_json(
        (output / "terminals.json").read_bytes(), strict=False,
    )
    terminal, = ledger.terminals
    assert terminal.status is TerminalStatus.UNKNOWN
    assert terminal.selection_disposition == "NO_SELECTION"
    assert terminal.submission is terminal.checked_outcome is terminal.dispatch is None
    assert report.native_execution.status == "NOT_REQUESTED"
    assert not (output / "native").exists()
