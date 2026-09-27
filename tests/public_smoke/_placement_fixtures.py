"""Fresh immutable M5 input constructors shared by public and development tests."""
from spatialcf.domain.base import FactAvailabilityV2, FactSetV2, RigidTransformV2, UncertaintyBudgetV2, Vec2, Vec3
from spatialcf.domain.geometry import (
    CollisionBodyFactV2, GeometryApproximationV2, GeometryInstanceV2, GeometryRoleV2,
    PlanarPolygonComponentV2, PlanarRegionV2, PlanarRingV2, RingWindingV2, UprightBox3DV2,
)
from spatialcf.domain.scene import CanonicalObject, CanonicalScene, ObjectPose, ObjectSupportAssignment, SupportSurfaceFact
from spatialcf.domain.scene import RegionBoundaryPolicy


def exact(values):
    return dict(availability=FactAvailabilityV2.KNOWN,values=values,uncertainty=UncertaintyBudgetV2())


def transform(x=0.0, y=0.0, z=0.0):
    return RigidTransformV2.identity().model_copy(update={"translation": Vec3(x=x, y=y, z=z)})


def rectangle(x0, y0, x1, y1):
    return PlanarRegionV2(components=(PlanarPolygonComponentV2(exterior=PlanarRingV2(
        vertices=tuple(Vec2(x=x,y=y) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))),
        winding=RingWindingV2.COUNTERCLOCKWISE,
    ), holes=()),))


def placement_scene():
    """A unit subject on the floor; a raised 4x4 target table at x=5."""
    objects = (
        CanonicalObject(object_id="entity:subject", category_id="category:box", movable=True,
            pose=ObjectPose(world_from_object=transform(z=0.5)), support_assignment=ObjectSupportAssignment.known("surface:floor")),
        CanonicalObject(object_id="object:table", category_id="category:table", movable=False,
            pose=ObjectPose(world_from_object=transform(x=5.0)), support_assignment=ObjectSupportAssignment.not_applicable()),
    )
    geometries = []
    for name, owner, role, anchor, size in (
        ("subject", "entity:subject", GeometryRoleV2.COLLISION, transform(), (1.0,1.0,1.0)),
        ("subject-support", "entity:subject", GeometryRoleV2.SUPPORT, transform(), (1.0,1.0,1.0)),
        ("floor", None, GeometryRoleV2.COLLISION, transform(z=-0.5), (20.0,20.0,1.0)),
        ("table", "object:table", GeometryRoleV2.COLLISION, transform(z=0.5), (4.0,4.0,1.0)),
    ):
        geometries.append(GeometryInstanceV2(geometry_id="geometry:"+name,owner_object_id=owner,role=role,
            anchor_from_geometry=anchor,approximation=GeometryApproximationV2.EXACT,uncertainty=UncertaintyBudgetV2(),
            shape=UprightBox3DV2(size_m=Vec3(x=size[0],y=size[1],z=size[2]))))
    bodies = tuple(CollisionBodyFactV2(body_id="body:"+name,owner_object_id=owner,geometry_instance_ids=("geometry:"+name,))
        for name,owner in (("subject","entity:subject"),("floor",None),("table","object:table")))
    surfaces = tuple(SupportSurfaceFact(surface_id="surface:"+name,owner_object_id=owner,supporting_body_id="body:"+name,
        anchor_from_surface=transform(z=height),normal_in_anchor=Vec3(x=0.0,y=0.0,z=1.0),region_uv=rectangle(-half,-half,half,half),
        region_approximation=GeometryApproximationV2.EXACT,boundary_policy=RegionBoundaryPolicy.CLOSED,geometry_uncertainty=UncertaintyBudgetV2())
        for name,owner,height,half in (("floor",None,0.0,10.0),("table","object:table",1.0,2.0)))
    return CanonicalScene(scene_id="scene:m5-synthetic",objects=exact(objects),geometry_instances=exact(tuple(geometries)),
        collision_bodies=exact(bodies),support_surfaces=exact(surfaces),workspace_boundaries=exact(()),
        known_free_spaces=exact(()),cameras=exact(()),baseline_observations=exact(()))


def table_cavity():
    from spatialcf.domain.semantic_place import SemanticPlaceCavityFact
    return SemanticPlaceCavityFact.seal(cavity_id='cavity:table',owner_object_id='object:table',
        anchor_from_cavity=transform(z=2.0),interior_size_m=Vec3(x=4.0,y=4.0,z=2.0),
        bottom_support_surface_id='surface:table',shell_body_ids=('body:table',))
