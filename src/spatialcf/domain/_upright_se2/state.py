"""Upright profile state contracts and intrinsic operations."""

from __future__ import annotations

from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
)

from pydantic import (
    Field,
    StrictInt,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FactAvailabilityV2,
    FiniteFloat,
    RigidTransformV2,
    Sha256Digest,
    Vec2,
)

from spatialcf.domain.definitions import (
    HashBoundCanonicalModel,
)

from spatialcf.domain.geometry import (
    CollisionBodyFactV2,
    GeometryInstanceV2,
    GeometryRoleV2,
)

from spatialcf.domain.operators import (
    StateDeltaManifest,
    StateVariableRef,
)

from spatialcf.domain.profiles import (
    OwnerRef,
)

from spatialcf.domain.scene import (
    BaselineObservation,
    CanonicalScene,
    SupportSurfaceFact,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_COMPILER_BUILD_SHA256,
    UPRIGHT_SE2_COMPILER_OWNER_REF,
    UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN,
    _UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
)

from spatialcf.domain._upright_se2.source_binding import (
    _DerivedSourceFact,
    _bound_scene_object,
    _derived_source_identifier,
    _exact_source_values,
)

from spatialcf.domain._upright_se2.yaw import (
    CanonicalSO2Angle,
    CardinalYawAuthorization,
    ContinuousYawAuthorization,
    ContinuousYawDomain,
    ExactDyadic,
    FixedPivotBinding,
    PivotMode,
    _require_sorted_unique_by_bytes,
    validate_directed_yaw_quaternion_consistency,
)


class UprightSE2CompilerClosure(HashBoundCanonicalModel):
    """The static profile/definition/owner closure consumed by the pure compiler."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/compiler-closure/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "compiler_closure_sha256"

    profile_registration_sha256: Sha256Digest
    definition_bundle_sha256: Sha256Digest
    solve_policy_definition_bundle_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    policy_bundle_sha256: Sha256Digest
    resource_policy_sha256: Sha256Digest
    compiler_owner_ref: OwnerRef
    compiler_build_sha256: Sha256Digest
    compiler_closure_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_compiler_owner(self) -> Self:
        if self.compiler_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("compiler owner reference must be fixed")
        if self.compiler_build_sha256 != UPRIGHT_SE2_COMPILER_BUILD_SHA256:
            raise ValueError("compiler build digest must be fixed")
        return self


class UprightSE2TranslationDomain(CanonicalModel):
    """The complete exact world-XY search domain for one cardinal tuple."""

    x_lower: ExactDyadic
    x_upper: ExactDyadic
    y_lower: ExactDyadic
    y_upper: ExactDyadic

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.x_lower.as_fraction > self.x_upper.as_fraction:
            raise ValueError("translation x lower bound must not exceed upper")
        if self.y_lower.as_fraction > self.y_upper.as_fraction:
            raise ValueError("translation y lower bound must not exceed upper")
        return self


class UprightSE2CardinalOperation(CanonicalModel):
    """One resolved cardinal *domain* transition before endpoint selection.

    The operation deliberately carries the complete authorized world-XY
    domain, rather than a selected translation.  A later, explicit endpoint
    materializer is the only place that may combine this operation with a
    concrete world-XY value.
    """

    authorization: CardinalYawAuthorization
    translation_domain: UprightSE2TranslationDomain
    inverse_quarter_turns_ccw: Annotated[StrictInt, Field(ge=0, le=3)]
    maximum_program_steps: Annotated[StrictInt, Field(gt=0)]
    maximum_edited_entities: Annotated[StrictInt, Field(gt=0)]

    @property
    def subject_id(self) -> CanonicalId:
        return self.authorization.subject_id

    @property
    def pivot_binding(self) -> FixedPivotBinding:
        return self.authorization.pivot_binding

    @property
    def quarter_turns_ccw(self) -> int:
        return self.authorization.yaw.q

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.authorization.cardinal_yaw_authorization_sha256

    @model_validator(mode="after")
    def _validate_inverse(self) -> Self:
        if self.inverse_quarter_turns_ccw != (-self.authorization.yaw.q) % 4:
            raise ValueError(
                "cardinal inverse must match the exact quarter-turn inverse"
            )
        if self.maximum_program_steps != 1:
            raise ValueError("upright se2 compilation permits exactly one program step")
        if self.maximum_edited_entities != 1:
            raise ValueError(
                "upright se2 compilation permits exactly one edited entity"
            )
        return self


class UprightSE2ContinuousOperation(CanonicalModel):
    """One resolved continuous domain transition before endpoint selection.

    This is intentionally a sibling of ``UprightSE2CardinalOperation``.  It
    does not widen that frozen cardinal record, so existing cardinal canonical
    bytes, discriminators, and hash domains remain unchanged.
    """

    authorization: ContinuousYawAuthorization
    translation_domain: UprightSE2TranslationDomain
    maximum_program_steps: Annotated[StrictInt, Field(gt=0)]
    maximum_edited_entities: Annotated[StrictInt, Field(gt=0)]

    @property
    def subject_id(self) -> CanonicalId:
        return self.authorization.subject_id

    @property
    def pivot_binding(self) -> FixedPivotBinding:
        return self.authorization.pivot_binding

    @property
    def yaw_domain(self) -> ContinuousYawDomain:
        return self.authorization.yaw_domain

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.authorization.continuous_yaw_authorization_sha256

    @model_validator(mode="after")
    def _validate_limits(self) -> Self:
        if self.maximum_program_steps != 1:
            raise ValueError("continuous upright se2 permits exactly one program step")
        if self.maximum_edited_entities != 1:
            raise ValueError("continuous upright se2 permits exactly one edited entity")
        return self


class UprightSE2StateFootprint(HashBoundCanonicalModel):
    """The complete primary/derived/frozen state partition for one transition."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/state-footprint/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "state_footprint_sha256"

    state_delta_manifest: StateDeltaManifest
    frozen_leaf_refs: tuple[StateVariableRef, ...]
    state_footprint_sha256: Sha256Digest

    @property
    def complete_before_leaf_index_sha256(self) -> Sha256Digest:
        return self.state_delta_manifest.complete_before_leaf_index_sha256

    @property
    def complete_after_leaf_index_sha256(self) -> Sha256Digest:
        return self.state_delta_manifest.complete_after_leaf_index_sha256

    @model_validator(mode="after")
    def _validate_frozen_partition(self) -> Self:
        _require_sorted_unique_by_bytes(self.frozen_leaf_refs, "frozen state leaves")
        primary = set(self.state_delta_manifest.authorized_primary_writes)
        derived = set(self.state_delta_manifest.recomputed_derived_writes)
        if primary & set(self.frozen_leaf_refs) or derived & set(self.frozen_leaf_refs):
            raise ValueError("frozen leaves must not overlap primary or derived writes")
        return self


def validate_required_upright_support_surface(
    scene: CanonicalScene,
    subject_id: CanonicalId,
) -> SupportSurfaceFact:
    """Resolve the subject's frozen support under the upright world-+Z rule."""

    subject = next(
        (
            object_
            for object_ in _exact_source_values(scene.objects, "objects")
            if object_.object_id == subject_id
        ),
        None,
    )
    if (
        subject is None
        or subject.support_assignment.availability is not FactAvailabilityV2.KNOWN
        or subject.support_assignment.surface_id is None
    ):
        raise ValueError(
            "subject support assignment must be KNOWN with a support surface"
        )
    sources = tuple(
        surface
        for surface in _exact_source_values(scene.support_surfaces, "support surfaces")
        if surface.surface_id == subject.support_assignment.surface_id
    )
    if len(sources) != 1:
        raise ValueError(
            "subject support assignment must bind one exact support surface"
        )

    surface = sources[0]
    normal = surface.normal_in_anchor
    if (normal.x, normal.y, normal.z) != (0.0, 0.0, 1.0):
        raise ValueError(
            "required support surface must have an exact horizontal world +Z normal"
        )
    surface_rotation = surface.anchor_from_surface.rotation
    if surface_rotation.x != 0.0 or surface_rotation.y != 0.0:
        raise ValueError(
            "required support surface frame must preserve exact horizontal world +Z"
        )
    if surface.owner_object_id is not None:
        owner = next(
            (
                object_
                for object_ in _exact_source_values(scene.objects, "objects")
                if object_.object_id == surface.owner_object_id
            ),
            None,
        )
        if owner is None:
            raise ValueError("required support surface owner must be a scene object")
        owner_rotation = owner.pose.world_from_object.rotation
        if owner_rotation.x != 0.0 or owner_rotation.y != 0.0:
            raise ValueError(
                "required support surface owner pose must preserve exact horizontal world +Z"
            )
    return surface


def _expected_derived_sources(
    scene: CanonicalScene,
    subject_id: CanonicalId,
    fact_kind: Literal["COLLISION", "SUPPORT", "RELATION", "VISIBILITY"],
) -> tuple[_DerivedSourceFact, ...]:
    if fact_kind == "COLLISION":
        return tuple(_exact_source_values(scene.collision_bodies, "collision bodies"))
    if fact_kind == "SUPPORT":
        return (validate_required_upright_support_surface(scene, subject_id),)
    if fact_kind == "RELATION":
        return tuple(
            geometry
            for geometry in _exact_source_values(
                scene.geometry_instances,
                "geometry instances",
            )
            if geometry.role is GeometryRoleV2.RELATION
        )
    return tuple(
        _exact_source_values(scene.baseline_observations, "baseline observations")
    )


class UprightSE2DerivedAfterFact(CanonicalModel):
    """One concrete frozen-source evaluation input at the canonical after pose."""

    fact_kind: Literal["COLLISION", "SUPPORT", "RELATION", "VISIBILITY"]
    source_fact_id: CanonicalId
    source_fact_sha256: Sha256Digest
    source_fact: _DerivedSourceFact
    after_subject_pose: RigidTransformV2

    @model_validator(mode="after")
    def _validate_concrete_source_fact(self) -> Self:
        source_type = {
            "COLLISION": CollisionBodyFactV2,
            "SUPPORT": SupportSurfaceFact,
            "RELATION": GeometryInstanceV2,
            "VISIBILITY": BaselineObservation,
        }[self.fact_kind]
        if not isinstance(self.source_fact, source_type):
            raise TypeError("derived fact kind does not match its concrete source fact")
        if self.fact_kind == "RELATION" and (
            self.source_fact.role is not GeometryRoleV2.RELATION
        ):
            raise ValueError("relation derived fact must use relation-role geometry")
        if _derived_source_identifier(self.source_fact) != self.source_fact_id:
            raise ValueError(
                "derived fact source ID does not match its concrete source fact"
            )
        if (
            canonical_sha256(
                self.source_fact,
                domain=UPRIGHT_SE2_DERIVED_SOURCE_HASH_DOMAIN,
            )
            != self.source_fact_sha256
        ):
            raise ValueError(
                "derived fact source digest does not bind its concrete source fact"
            )
        return self


class UprightSE2AfterStateTemplate(HashBoundCanonicalModel):
    """A complete endpoint pose template without solver or checker conclusions."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/after-state-template/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "after_state_template_sha256"

    subject_id: CanonicalId
    subject_before_pose: RigidTransformV2
    evaluation_scene: CanonicalScene
    subject_pivot_xy_m: Vec2
    subject_pivot_z_m: FiniteFloat
    subject_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    reference_pivot_xy_m: Vec2
    collision_facts: tuple[UprightSE2DerivedAfterFact, ...]
    support_facts: tuple[UprightSE2DerivedAfterFact, ...]
    relation_facts: tuple[UprightSE2DerivedAfterFact, ...]
    visibility_facts: tuple[UprightSE2DerivedAfterFact, ...]
    after_state_template_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_derived_views(self) -> Self:
        subject = next(
            (
                object_
                for object_ in self.evaluation_scene.objects.values or ()
                if object_.object_id == self.subject_id
            ),
            None,
        )
        if subject is None:
            raise ValueError("after-state evaluation scene must contain the subject")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError(
                "after-state subject before pose must bind the evaluation scene"
            )
        translation = self.subject_pose.translation
        if (translation.x, translation.y, translation.z) != (
            self.subject_pivot_xy_m.x,
            self.subject_pivot_xy_m.y,
            self.subject_pivot_z_m,
        ):
            raise ValueError("canonical subject pose must match the after pivot")
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns,
            self.subject_pose.rotation,
        )
        _exact_source_values(self.evaluation_scene.cameras, "cameras")
        for label, facts in (
            ("collision facts", self.collision_facts),
            ("support facts", self.support_facts),
            ("relation facts", self.relation_facts),
            ("visibility facts", self.visibility_facts),
        ):
            _require_sorted_unique_by_bytes(facts, label)
            if any(fact.after_subject_pose != self.subject_pose for fact in facts):
                raise ValueError(
                    "derived fact after poses must match the canonical subject pose"
                )
        for fact_kind, facts in (
            ("COLLISION", self.collision_facts),
            ("SUPPORT", self.support_facts),
            ("RELATION", self.relation_facts),
            ("VISIBILITY", self.visibility_facts),
        ):
            if any(fact.fact_kind != fact_kind for fact in facts):
                raise ValueError("after-state derived fact family is inconsistent")
            expected_sources = _expected_derived_sources(
                self.evaluation_scene,
                self.subject_id,
                fact_kind,
            )
            expected_by_id = {
                _derived_source_identifier(source): source
                for source in expected_sources
            }
            actual_by_id = {fact.source_fact_id: fact.source_fact for fact in facts}
            if len(actual_by_id) != len(facts) or set(actual_by_id) != set(
                expected_by_id
            ):
                raise ValueError(
                    "after-state derived facts must cover the complete affected set"
                )
            if any(
                canonical_json_bytes(actual_by_id[source_id])
                != canonical_json_bytes(source)
                for source_id, source in expected_by_id.items()
            ):
                raise ValueError(
                    "after-state derived facts must bind evaluation-scene sources"
                )
        return self


class UprightSE2EndpointConstructionRecipe(HashBoundCanonicalModel):
    """The complete source-bound recipe for a later selected endpoint.

    This is intentionally not an endpoint and does not carry a translation.
    It records every source value needed by the pure materializer so a
    compilation remains a statement about an authorized domain only.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/endpoint-construction-recipe/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "endpoint_construction_recipe_sha256"

    source_scene_state_sha256: Sha256Digest
    operation_authorization_sha256: Sha256Digest
    subject_id: CanonicalId
    reference_id: CanonicalId
    pivot_binding: FixedPivotBinding
    quarter_turns_ccw: Annotated[StrictInt, Field(ge=0, le=3)]
    translation_domain: UprightSE2TranslationDomain
    subject_before_pose: RigidTransformV2
    reference_before_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    evaluation_scene: CanonicalScene
    endpoint_construction_recipe_sha256: Sha256Digest

    @property
    def authorization_sha256(self) -> Sha256Digest:
        """Expose the bound operation authorization without duplicating it."""

        return self.operation_authorization_sha256

    @property
    def translation_domain_sha256(self) -> Sha256Digest:
        """Return the explicit canonical root of the authorized XY domain."""

        return canonical_sha256(
            self.translation_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        )

    @model_validator(mode="after")
    def _validate_source_bound_recipe(self) -> Self:
        subject = _bound_scene_object(self.evaluation_scene, self.subject_id)
        reference = _bound_scene_object(self.evaluation_scene, self.reference_id)
        if self.subject_id == self.reference_id:
            raise ValueError("endpoint recipe subject and reference must differ")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError(
                "endpoint recipe subject pose must bind its evaluation scene"
            )
        if reference.pose.world_from_object != self.reference_before_pose:
            raise ValueError(
                "endpoint recipe reference pose must bind its evaluation scene"
            )
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns,
            self.subject_before_pose.rotation,
        )
        expected_pivot_id = (
            self.subject_id
            if self.pivot_binding.pivot_mode is PivotMode.OWN
            else self.reference_id
        )
        if self.pivot_binding.pivot_entity_id != expected_pivot_id:
            raise ValueError(
                "endpoint recipe pivot must bind its selected scene object"
            )
        expected_state_sha256 = canonical_sha256(
            {
                "scene_state_sha256": self.source_scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": (
                    subject.pose
                    if self.pivot_binding.pivot_mode is PivotMode.OWN
                    else reference.pose
                ),
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        )
        if self.pivot_binding.pivot_state_sha256 != expected_state_sha256:
            raise ValueError(
                "endpoint recipe pivot digest must bind the source scene state"
            )
        return self


class UprightSE2ContinuousEndpointConstructionRecipe(HashBoundCanonicalModel):
    """Source-bound continuous endpoint recipe, separate from cardinal bytes."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-endpoint-construction-recipe/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_endpoint_construction_recipe_sha256"

    source_scene_state_sha256: Sha256Digest
    operation_authorization_sha256: Sha256Digest
    subject_id: CanonicalId
    reference_id: CanonicalId
    pivot_binding: FixedPivotBinding
    yaw_domain: ContinuousYawDomain
    translation_domain: UprightSE2TranslationDomain
    subject_before_pose: RigidTransformV2
    reference_before_pose: RigidTransformV2
    subject_yaw_turns: CanonicalSO2Angle
    evaluation_scene: CanonicalScene
    continuous_endpoint_construction_recipe_sha256: Sha256Digest

    @property
    def authorization_sha256(self) -> Sha256Digest:
        return self.operation_authorization_sha256

    @property
    def translation_domain_sha256(self) -> Sha256Digest:
        return canonical_sha256(
            self.translation_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        )

    @model_validator(mode="after")
    def _validate_source_bound_recipe(self) -> Self:
        subject = _bound_scene_object(self.evaluation_scene, self.subject_id)
        reference = _bound_scene_object(self.evaluation_scene, self.reference_id)
        if self.subject_id == self.reference_id:
            raise ValueError("continuous endpoint subject and reference must differ")
        if subject.pose.world_from_object != self.subject_before_pose:
            raise ValueError("continuous endpoint subject pose must bind its scene")
        if reference.pose.world_from_object != self.reference_before_pose:
            raise ValueError("continuous endpoint reference pose must bind its scene")
        validate_directed_yaw_quaternion_consistency(
            self.subject_yaw_turns, self.subject_before_pose.rotation
        )
        expected_pivot_id = (
            self.subject_id
            if self.pivot_binding.pivot_mode is PivotMode.OWN
            else self.reference_id
        )
        if self.pivot_binding.pivot_entity_id != expected_pivot_id:
            raise ValueError("continuous endpoint pivot must bind its scene object")
        expected_state_sha256 = canonical_sha256(
            {
                "scene_state_sha256": self.source_scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": (
                    subject.pose
                    if self.pivot_binding.pivot_mode is PivotMode.OWN
                    else reference.pose
                ),
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        )
        if self.pivot_binding.pivot_state_sha256 != expected_state_sha256:
            raise ValueError("continuous endpoint pivot digest does not bind source")
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2CompilerClosure.model_rebuild()
UprightSE2TranslationDomain.model_rebuild()
UprightSE2CardinalOperation.model_rebuild()
UprightSE2ContinuousOperation.model_rebuild()
UprightSE2StateFootprint.model_rebuild()
UprightSE2DerivedAfterFact.model_rebuild()
UprightSE2AfterStateTemplate.model_rebuild()
UprightSE2EndpointConstructionRecipe.model_rebuild()
UprightSE2ContinuousEndpointConstructionRecipe.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2CompilerClosure.__module__ = "spatialcf.domain.upright_se2"
UprightSE2TranslationDomain.__module__ = "spatialcf.domain.upright_se2"
UprightSE2CardinalOperation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousOperation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2StateFootprint.__module__ = "spatialcf.domain.upright_se2"
validate_required_upright_support_surface.__module__ = "spatialcf.domain.upright_se2"
_expected_derived_sources.__module__ = "spatialcf.domain.upright_se2"
UprightSE2DerivedAfterFact.__module__ = "spatialcf.domain.upright_se2"
UprightSE2AfterStateTemplate.__module__ = "spatialcf.domain.upright_se2"
UprightSE2EndpointConstructionRecipe.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousEndpointConstructionRecipe.__module__ = "spatialcf.domain.upright_se2"
