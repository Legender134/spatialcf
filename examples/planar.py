"""CPU planar input and direct solver replay; no simulator or dataset publication."""
from spatialcf.domain.problem import SemanticProblemV2_3

def example_problem() -> SemanticProblemV2_3:
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


def main():
    import json
    from spatialcf.core.solver import solve_minimum_cost
    from spatialcf.core.verification import verify_solve_result
    from spatialcf.domain.artifacts import StrictConvexCandidateCompilerConfigV2_7
    from spatialcf.domain.solver import ContinuousYawSolverConfigV2_9

    problem = example_problem()
    config = ContinuousYawSolverConfigV2_9(
        candidate_config=StrictConvexCandidateCompilerConfigV2_7(
            max_domain_operations=1024, max_so2_atomic_steps=1024, max_candidate_cells=8),
        max_objective_partition_cells=8, target_optimality_gap=0.0,
    )
    outcome = solve_minimum_cost(problem, config)
    result = outcome.result
    verification = None if result is None else verify_solve_result(problem, config, result)
    if verification is not None and verification.kind != "VERIFIED":
        raise RuntimeError(f"fresh planar replay did not verify: {verification.kind}")
    print(json.dumps({
        "native_execution": "NOT_REQUESTED",
        "outcome": outcome.model_dump(mode="json"),
        "verification": None if verification is None else verification.model_dump(mode="json"),
    }, indent=2))


if __name__ == "__main__":
    main()
