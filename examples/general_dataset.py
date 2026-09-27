"""Self-contained CPU M5/M6 snapshot, batch export and proof replay example."""

from __future__ import annotations

from spatialcf.domain.base import (
    FactAvailabilityV2,
    RigidTransformV2,
    UncertaintyBudgetV2,
    Vec2,
    Vec3,
)

from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryApproximationV2,
    GeometryInstanceV2,
    GeometryRoleV2,
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
    RingWindingV2,
    UprightBox3DV2,
)

from spatialcf.domain.scene import (
    CanonicalObject,
    CanonicalScene,
    ObjectPose,
    ObjectSupportAssignment,
    SupportSurfaceFact,
)

from spatialcf.domain.scene import RegionBoundaryPolicy


def exact(values):
    return dict(
        availability=FactAvailabilityV2.KNOWN,
        values=values,
        uncertainty=UncertaintyBudgetV2(),
    )


def transform(x=0.0, y=0.0, z=0.0):
    return RigidTransformV2.identity().model_copy(
        update={"translation": Vec3(x=x, y=y, z=z)}
    )


def rectangle(x0, y0, x1, y1):
    return PlanarRegionV2(
        components=(
            PlanarPolygonComponentV2(
                exterior=PlanarRingV2(
                    vertices=tuple(
                        Vec2(x=x, y=y)
                        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                    ),
                    winding=RingWindingV2.COUNTERCLOCKWISE,
                ),
                holes=(),
            ),
        )
    )


def placement_scene():
    """A unit subject on the floor; a raised 4x4 target table at x=5."""
    objects = (
        CanonicalObject(
            object_id="entity:subject",
            category_id="category:box",
            movable=True,
            pose=ObjectPose(world_from_object=transform(z=0.5)),
            support_assignment=ObjectSupportAssignment.known("surface:floor"),
        ),
        CanonicalObject(
            object_id="object:table",
            category_id="category:table",
            movable=False,
            pose=ObjectPose(world_from_object=transform(x=5.0)),
            support_assignment=ObjectSupportAssignment.not_applicable(),
        ),
    )
    geometries = []
    for name, owner, role, anchor, size in (
        (
            "subject",
            "entity:subject",
            GeometryRoleV2.COLLISION,
            transform(),
            (1.0, 1.0, 1.0),
        ),
        (
            "subject-support",
            "entity:subject",
            GeometryRoleV2.SUPPORT,
            transform(),
            (1.0, 1.0, 1.0),
        ),
        ("floor", None, GeometryRoleV2.COLLISION, transform(z=-0.5), (20.0, 20.0, 1.0)),
        (
            "table",
            "object:table",
            GeometryRoleV2.COLLISION,
            transform(z=0.5),
            (4.0, 4.0, 1.0),
        ),
    ):
        geometries.append(
            GeometryInstanceV2(
                geometry_id="geometry:" + name,
                owner_object_id=owner,
                role=role,
                anchor_from_geometry=anchor,
                approximation=GeometryApproximationV2.EXACT,
                uncertainty=UncertaintyBudgetV2(),
                shape=UprightBox3DV2(size_m=Vec3(x=size[0], y=size[1], z=size[2])),
            )
        )
    bodies = tuple(
        CollisionBodyFactV2(
            body_id="body:" + name,
            owner_object_id=owner,
            geometry_instance_ids=("geometry:" + name,),
        )
        for name, owner in (
            ("subject", "entity:subject"),
            ("floor", None),
            ("table", "object:table"),
        )
    )
    surfaces = tuple(
        SupportSurfaceFact(
            surface_id="surface:" + name,
            owner_object_id=owner,
            supporting_body_id="body:" + name,
            anchor_from_surface=transform(z=height),
            normal_in_anchor=Vec3(x=0.0, y=0.0, z=1.0),
            region_uv=rectangle(-half, -half, half, half),
            region_approximation=GeometryApproximationV2.EXACT,
            boundary_policy=RegionBoundaryPolicy.CLOSED,
            geometry_uncertainty=UncertaintyBudgetV2(),
        )
        for name, owner, height, half in (
            ("floor", None, 0.0, 10.0),
            ("table", "object:table", 1.0, 2.0),
        )
    )
    return CanonicalScene(
        scene_id="scene:m5-synthetic",
        objects=exact(objects),
        geometry_instances=exact(tuple(geometries)),
        collision_bodies=exact(bodies),
        support_surfaces=exact(surfaces),
        workspace_boundaries=exact(()),
        known_free_spaces=exact(()),
        cameras=exact(()),
        baseline_observations=exact(()),
    )


def table_cavity():
    from spatialcf.domain.semantic_place import SemanticPlaceCavityFact

    return SemanticPlaceCavityFact.seal(
        cavity_id="cavity:table",
        owner_object_id="object:table",
        anchor_from_cavity=transform(z=2.0),
        interior_size_m=Vec3(x=4.0, y=4.0, z=2.0),
        bottom_support_surface_id="surface:table",
        shell_body_ids=("body:table",),
    )


from spatialcf.core.rigid_se3_compiler import build_rigid_se3_state, d

from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    UncertaintyBudgetV2,
)

from spatialcf.domain.definitions import CanonicalIdValue, TypedValue

from spatialcf.domain.predicates import PredicateAtom

from spatialcf.domain.rigid_se3 import (
    RigidSE3BodyFact,
    RigidSE3Box,
    RigidSE3ContactChoiceDomain,
    RigidSE3ContactFact,
    RigidSE3ContactState,
    RigidSE3Domain,
    RigidSE3EditMemberTemplate,
    RigidSE3EditSetTemplate,
    RigidSE3ExactSourceFacts,
    RigidSE3JointChoiceDomain,
    RigidSE3JointFact,
    RigidSE3JointState,
    RigidSE3ObjectivePolicy,
    RigidSE3Pose,
    RigidSE3PrismaticDomain,
    RigidSE3ProgramSkeleton,
    RigidSE3Rational,
    RigidSE3RootChoiceDomain,
    RigidSE3RootRotationDomain,
    RigidSE3RootState,
    RigidSE3RootTranslationDomain,
    RigidSE3Rotation,
    RigidSE3Vector,
)

from spatialcf.domain.scene import CanonicalScene


def _q(n: int, d: int = 1) -> RigidSE3Rational:
    return RigidSE3Rational(numerator=n, denominator=d)


def _vector(x: int = 0, y: int = 0, z: int = 0) -> RigidSE3Vector:
    return RigidSE3Vector(x=_q(x), y=_q(y), z=_q(z))


def _rotation() -> RigidSE3Rotation:
    return RigidSE3Rotation(
        rows=(
            (_q(1), _q(0), _q(0)),
            (_q(0), _q(1), _q(0)),
            (_q(0), _q(0), _q(1)),
        )
    )


def _pose(frame: str = "WORLD") -> RigidSE3Pose:
    return RigidSE3Pose(frame=frame, rotation=_rotation(), translation_m=_vector())


def _empty_scene() -> CanonicalScene:
    exact_empty = FactSetV2(
        availability=FactAvailabilityV2.KNOWN,
        completeness=FactCompletenessV2.EXACT,
        values=(),
        uncertainty=UncertaintyBudgetV2(),
    )
    return CanonicalScene(
        scene_id="scene:public-rigid-se3",
        objects=exact_empty,
        geometry_instances=exact_empty,
        collision_bodies=exact_empty,
        workspace_boundaries=exact_empty,
        known_free_spaces=exact_empty,
        support_surfaces=exact_empty,
        cameras=exact_empty,
        baseline_observations=exact_empty,
    )


def _source() -> RigidSE3ExactSourceFacts:
    return RigidSE3ExactSourceFacts(
        bodies=(
            RigidSE3BodyFact(
                body_id="entity:a",
                category_id="category:box",
                kind="ROOT",
                solid=True,
                primitive_ids=("primitive:a",),
                may_move=True,
                may_change_contact=True,
            ),
            RigidSE3BodyFact(
                body_id="entity:b",
                category_id="category:box",
                kind="CHILD",
                solid=True,
                primitive_ids=("primitive:b",),
                may_move=True,
                may_change_contact=True,
            ),
        ),
        boxes=(
            RigidSE3Box(
                primitive_id="primitive:a",
                local_anchor=_pose("BODY_LOCAL"),
                half_extents_m=_vector(1, 1, 1),
            ),
            RigidSE3Box(
                primitive_id="primitive:b",
                local_anchor=_pose("BODY_LOCAL"),
                half_extents_m=_vector(1, 1, 1),
            ),
        ),
        joints=(
            RigidSE3JointFact(
                joint_id="joint:ab",
                parent_body_id="entity:a",
                child_body_id="entity:b",
                kind="PRISMATIC",
                parent_attachment=_pose("BODY_LOCAL"),
                child_attachment=_pose("JOINT_LOCAL"),
                axis=_vector(1),
            ),
        ),
        contacts=(
            RigidSE3ContactFact(
                contact_id="contact:ab",
                first_body_id="entity:a",
                first_primitive_id="primitive:a",
                first_face="PX",
                second_body_id="entity:b",
                second_primitive_id="primitive:b",
                second_face="NX",
                release_gap_m=_q(1, 3),
            ),
        ),
        roots=(RigidSE3RootState(body_id="entity:a", world_pose=_pose()),),
        joint_states=(
            RigidSE3JointState(
                joint_id="joint:ab",
                kind="PRISMATIC",
                offset_m=_q(3),
            ),
        ),
        contact_states=(
            RigidSE3ContactState(
                contact_id="contact:ab",
                mode="RELEASED",
            ),
        ),
    )


def _domain(
    scene: CanonicalScene,
    source: RigidSE3ExactSourceFacts,
    *,
    continuous: bool,
    only_release: bool,
) -> RigidSE3Domain:
    writable = build_rigid_se3_state(scene, source).writable_value_leaves
    translation = (
        RigidSE3RootTranslationDomain(
            kind="CLOSED_RATIONAL_BOX",
            lower_m=_vector(),
            upper_m=_vector(1, 1, 1),
        )
        if continuous
        else RigidSE3RootTranslationDomain(kind="FINITE", values=(_vector(),))
    )
    return RigidSE3Domain.seal(
        roots=(
            RigidSE3RootChoiceDomain(
                root_body_id="entity:a",
                translation=translation,
                rotation=RigidSE3RootRotationDomain(
                    kind="FINITE", values=(_rotation(),)
                ),
            ),
        ),
        joints=(
            RigidSE3JointChoiceDomain(
                joint_id="joint:ab",
                kind="PRISMATIC",
                prismatic=RigidSE3PrismaticDomain(
                    kind="FINITE",
                    values_m=(_q(3),) if only_release else (_q(2), _q(3)),
                ),
            ),
        ),
        contacts=(
            RigidSE3ContactChoiceDomain(
                contact_id="contact:ab",
                modes=("RELEASED",) if only_release else ("ENGAGED", "RELEASED"),
            ),
        ),
        program_skeletons=(
            RigidSE3ProgramSkeleton(
                skeleton_id="program:atomic",
                steps=(
                    RigidSE3EditSetTemplate(
                        members=(
                            RigidSE3EditMemberTemplate(
                                kind="SET_CONTACT_MODE", target_id="contact:ab"
                            ),
                            RigidSE3EditMemberTemplate(
                                kind="SET_JOINT", target_id="joint:ab"
                            ),
                            RigidSE3EditMemberTemplate(
                                kind="SET_ROOT_SE3", target_id="entity:a"
                            ),
                        )
                    ),
                ),
            ),
        ),
        affected_body_ids=("entity:a", "entity:b"),
        authorized_primary_leaves=writable,
    )


def _objective() -> RigidSE3ObjectivePolicy:
    return RigidSE3ObjectivePolicy(
        translation_weight=_q(1),
        rotation_weight=_q(1),
        prismatic_weight=_q(1),
        revolute_weight=_q(1),
        contact_weight=_q(1),
        step_weight=_q(1),
        translation_length_m=_q(1),
        joint_length_m=_q(1),
    )


def _goal() -> PredicateAtom:
    return PredicateAtom(
        predicate_ref="definition:spatialcf/rigid-se3/1.0/predicate/face_contact",
        operands=(
            TypedValue(
                value_schema_ref="schema:spatialcf/rigid-se3/1.0/contact-id",
                payload=CanonicalIdValue(value="contact:ab"),
            ),
        ),
    )


from pathlib import Path
from spatialcf.domain.general_dataset import (
    GeneralDatasetInput,
    PlacementSnapshot,
    PlacementTask,
    RigidSnapshot,
    RigidTask,
)
from spatialcf.domain.predicates import AfterGoal
from spatialcf.domain.rigid_se3 import RigidSE3Limits
from spatialcf.domain.semantic_place import SemanticPlaceInterval
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.generation.general import (
    generate_general_dataset,
    verify_general_dataset,
)


def _witness_hints(case):
    if case != "witness":
        return ()
    from spatialcf.domain.rigid_se3 import (
        RigidSE3MemberChoice,
        RigidSE3StepChoice,
        RigidSE3WitnessHint,
    )

    return (RigidSE3WitnessHint(
        skeleton_id="program:atomic",
        steps=(RigidSE3StepChoice(members=(
            RigidSE3MemberChoice(
                kind="SET_CONTACT_MODE", target_id="contact:ab", mode="ENGAGED"
            ),
            RigidSE3MemberChoice(
                kind="SET_JOINT", target_id="joint:ab", offset_m=_q(2)
            ),
            RigidSE3MemberChoice(
                kind="SET_ROOT_SE3", target_id="entity:a", pose=_pose()
            ),
        )),),
    ),)


def example_input(case: str = "mixed") -> GeneralDatasetInput:
    """Build fresh facts; only independently certified results publish pairs."""
    if case not in {"mixed", "placement", "multibody", "unknown", "unsat", "witness"}:
        raise ValueError(f"unknown example case: {case}")
    placement = PlacementSnapshot.seal(
        source_id="source:placement",
        dataset_id="dataset:example",
        revision_id="revision:1",
        scene=placement_scene(),
        cavities=(table_cavity(),),
    )
    scene, facts = _empty_scene(), _source()
    rigid = RigidSnapshot.seal(
        source_id="source:rigid",
        dataset_id="dataset:example",
        revision_id="revision:1",
        scene=scene,
        facts=facts,
    )
    tasks = (
        PlacementTask.seal(
            candidate_id="candidate:place-on",
            source_id=placement.source_id,
            subject_id="entity:subject",
            operation="PLACE_ON",
            target_ids=("surface:table",),
            x=SemanticPlaceInterval(lower=-10.0, upper=10.0),
            y=SemanticPlaceInterval(lower=-10.0, upper=10.0),
            z=SemanticPlaceInterval(lower=0.0, upper=10.0),
        ),
        RigidTask.seal(
            candidate_id="candidate:joint-contact",
            source_id=rigid.source_id,
            domain=_domain(
                scene, facts,
                continuous=case in {"unknown", "witness"},
                only_release=case == "unsat",
            ),
            witness_hints=_witness_hints(case),
            objective=_objective(),
            after_goal=AfterGoal(formula=_goal()),
            limits=RigidSE3Limits(max_evaluated_tuples=4),
        ),
    )
    if case == "placement":
        return GeneralDatasetInput.seal(sources=(placement,), tasks=(tasks[0],))
    if case != "mixed":
        return GeneralDatasetInput.seal(sources=(rigid,), tasks=(tasks[1],))
    return GeneralDatasetInput.seal(sources=(placement, rigid), tasks=tasks)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="new input JSON path")
    parser.add_argument(
        "--case",
        choices=("mixed", "placement", "multibody", "unknown", "unsat", "witness"),
        default="mixed",
    )
    parser.add_argument(
        "--output", type=Path, help="optional new output directory; generate and verify"
    )
    arguments = parser.parse_args()
    with arguments.input.open("xb") as stream:
        stream.write(canonical_json_bytes(example_input(arguments.case)))
    if arguments.output is not None:
        generated = generate_general_dataset(arguments.input, arguments.output)
        assert verify_general_dataset(arguments.output) == generated
        print(generated.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
