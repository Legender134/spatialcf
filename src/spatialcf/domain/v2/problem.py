"""Closed, platform-neutral Canonical v2 semantic problem contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    NumericPolicyV2,
    SchemaIdentityV2,
    V2Model,
)
from spatialcf.domain.v2.constraints import (
    CanonicalConstraintSetV2,
    MeasurementUnitV2,
    PositionRegionInterpretationV2,
    RelationAxisV2,
    RelationSemanticsV2,
    VisibilityConstraintV2,
    VisibilityMetricKindV2,
    VisibilitySemanticsV2,
)
from spatialcf.domain.v2.geometry import GeometryApproximationV2, GeometryRoleV2
from spatialcf.domain.v2.objective import (
    ObjectiveSpecV2,
    PairAxisKeyV2,
    SafetyComponentKindV2,
    SafetySlackUnitV2,
)
from spatialcf.domain.v2.scene import BaselineObservationV2, CanonicalSceneV2
from spatialcf.domain.v2.serialization import (
    canonical_json_bytes_v2,
    canonical_sha256_v2,
)

_PROBLEM_HASH_DOMAIN = "spatialcf.semantic-problem.v2"


class SemanticProblemV2(V2Model):
    """Everything the production core may consume for one optimization problem.

    Adapter provenance, native locators, runtime outcomes, and audit evidence are
    deliberately absent.  Incomplete fact families remain valid inputs; the core
    must turn the resulting blockers into ``UNCERTIFIED`` rather than silently
    treating them as known-empty facts.
    """

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="semantic-problem")
    )
    scene: CanonicalSceneV2
    constraints: CanonicalConstraintSetV2
    relation_semantics: RelationSemanticsV2
    visibility_semantics: VisibilitySemanticsV2
    objective: ObjectiveSpecV2
    numeric_policy: NumericPolicyV2

    @model_validator(mode="after")
    def validate_semantic_graph(self) -> Self:
        if self.schema_identity != self._expected_schema_identity():
            raise ValueError("semantic-problem schema identity must be fixed")

        objects = {item.object_id: item for item in self.scene.objects.values or ()}
        subject_id = self.constraints.allowed_edit.subject_id
        subject = objects.get(subject_id)
        if subject is None:
            raise ValueError("allowed-edit subject references an unknown object")
        if not subject.movable:
            raise ValueError("allowed-edit subject must be movable")

        target = self.constraints.target_relation
        if target.reference_id not in objects:
            raise ValueError("target reference references an unknown object")
        if target.semantics_id != self.relation_semantics.semantics_id:
            raise ValueError("target semantics ID does not match relation semantics")
        if any(
            constraint.visibility_semantics_id != self.visibility_semantics.semantics_id
            for constraint in self.constraints.visibility_constraints
        ):
            raise ValueError(
                "visibility constraint semantics ID does not match visibility semantics"
            )

        self._validate_position_references()
        self._validate_collision_references(objects)
        self._validate_support_references(objects)
        self._validate_visibility_references(objects)
        self._validate_relation_universe(objects)
        self._validate_visibility_objective(objects)
        self._validate_safety_universe()
        return self

    @classmethod
    def _expected_schema_identity(cls) -> SchemaIdentityV2:
        return SchemaIdentityV2(schema_name="semantic-problem")

    @property
    def semantic_problem_sha256(self) -> str:
        """Domain-separated digest of semantic bytes only."""

        return canonical_sha256_v2(self, domain=_PROBLEM_HASH_DOMAIN)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes_v2(self)

    @property
    def certification_blockers(self) -> tuple[str, ...]:
        """Stable reasons this input cannot immediately claim global coverage.

        These are derived from the semantic payload and are intentionally not a
        self-referential serialized field.  Task 3's compiler will map them to a
        closed ``UNCERTIFIED`` status rather than guessing absent facts.
        """

        blockers: set[str] = set()
        required = self._required_fact_references()
        for family_name, (facts, id_field, required_ids) in required.items():
            if not required_ids:
                continue
            if facts.availability is FactAvailabilityV2.MISSING:
                blockers.add(f"MISSING_FACT:{family_name}")
                continue
            if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
                blockers.add(f"NOT_APPLICABLE_REQUIRED_FACT:{family_name}")
                continue
            guaranteed_ids = _ids(_guaranteed_values(facts), id_field)
            for required_id in required_ids - guaranteed_ids:
                if _known_family_excludes_id(facts, id_field, required_id):
                    # The model validator rejects this case; retain a closed
                    # fallback for unsafe model copies before canonical hashing.
                    blockers.add(f"DANGLING_REFERENCE:{family_name}:{required_id}")
                elif facts.completeness is FactCompletenessV2.OUTER_BOUND:
                    blockers.add(f"OUTER_ONLY_REFERENCE:{family_name}:{required_id}")
                else:
                    blockers.add(f"UNGUARANTEED_REFERENCE:{family_name}:{required_id}")
            if facts.completeness in {
                FactCompletenessV2.INNER_BOUND,
                FactCompletenessV2.OUTER_BOUND,
                FactCompletenessV2.SAMPLED,
            }:
                blockers.add(
                    f"UNBRACKETED_COVERAGE:{family_name}:{facts.completeness.value}"
                )
        blockers.update(self._observation_certification_blockers())
        blockers.update(self._geometry_selection_blockers())
        blockers.update(self._collision_certification_blockers())
        blockers.update(self._support_certification_blockers())
        blockers.update(self._completeness_requirement_blockers())
        return tuple(sorted(blockers))

    def _validate_position_references(self) -> None:
        position = self.constraints.position_domain
        _validate_required_ids(
            self.scene.workspace_boundaries,
            "fact_id",
            set(position.workspace_fact_ids),
            family_name="workspace_boundaries",
        )
        _validate_required_ids(
            self.scene.known_free_spaces,
            "fact_id",
            set(position.known_free_space_fact_ids),
            family_name="known_free_spaces",
        )
        _validate_required_ids(
            self.scene.collision_bodies,
            "body_id",
            set(position.subject_occupancy_body_ids),
            family_name="collision_bodies",
        )
        if (
            position.region_interpretation
            is not PositionRegionInterpretationV2.SUBJECT_OCCUPANCY_CONTAINED
        ):
            return

        subject_id = self.constraints.allowed_edit.subject_id
        bodies = _items_by_id(self.scene.collision_bodies, "body_id")
        for body_id in position.subject_occupancy_body_ids:
            body = bodies.get(body_id)
            if body is not None and body.owner_object_id != subject_id:
                raise ValueError(
                    "position-domain occupancy body must be owned by the subject"
                )
        facts = self.scene.collision_bodies
        if facts.availability is FactAvailabilityV2.KNOWN and facts.completeness in {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }:
            expected = {
                body.body_id
                for body in _possible_values(facts)
                if body.owner_object_id == subject_id
            }
            if set(position.subject_occupancy_body_ids) != expected:
                raise ValueError(
                    "position-domain occupancy coverage must include every possible "
                    "subject collision body"
                )

    def _validate_collision_references(self, objects: dict[str, Any]) -> None:
        required_subject: set[str] = set()
        required_obstacles: set[str] = set()
        for constraint in self.constraints.collision_constraints:
            required_subject.update(constraint.subject_body_ids)
            required_obstacles.update(constraint.obstacle_body_ids)
        required_ids = required_subject | required_obstacles
        _validate_required_ids(
            self.scene.collision_bodies,
            "body_id",
            required_ids,
            family_name="collision_bodies",
        )

        bodies = _items_by_id(self.scene.collision_bodies, "body_id")
        subject_id = self.constraints.allowed_edit.subject_id
        for body_id in required_subject:
            body = bodies.get(body_id)
            if body is not None and body.owner_object_id != subject_id:
                raise ValueError("collision subject body must be owned by the subject")
        for body_id in required_obstacles:
            body = bodies.get(body_id)
            if body is not None and body.owner_object_id == subject_id:
                raise ValueError(
                    "collision obstacle body must not be owned by the subject"
                )

        support_by_id = {
            item.constraint_id: item for item in self.constraints.support_constraints
        }
        surfaces = _items_by_id(self.scene.support_surfaces, "surface_id")
        for collision in self.constraints.collision_constraints:
            for exception in collision.support_contact_exceptions:
                support = support_by_id[exception.support_constraint_id]
                surface = surfaces.get(support.surface_id)
                if (
                    surface is not None
                    and surface.supporting_body_id != exception.obstacle_body_id
                ):
                    raise ValueError(
                        "support-contact exception obstacle body must be the support "
                        "surface supporting body"
                    )

        facts = self.scene.collision_bodies
        if facts.availability is FactAvailabilityV2.KNOWN and facts.completeness in {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }:
            possible_bodies = _items_by_id(facts, "body_id")
            expected_subject = {
                body_id
                for body_id, body in possible_bodies.items()
                if body.owner_object_id == subject_id
            }
            if not expected_subject:
                raise ValueError(
                    "a movable subject requires at least one subject collision body"
                )
            expected_obstacles = set(possible_bodies) - expected_subject
            expected_pairs = {
                (subject_body_id, obstacle_body_id)
                for subject_body_id in expected_subject
                for obstacle_body_id in expected_obstacles
            }
            provided_pairs = {
                (subject_body_id, obstacle_body_id)
                for constraint in self.constraints.collision_constraints
                for subject_body_id in constraint.subject_body_ids
                for obstacle_body_id in constraint.obstacle_body_ids
            }
            if provided_pairs != expected_pairs:
                raise ValueError(
                    "collision constraint coverage must include every possible "
                    "subject-body/obstacle-body pair from the sound outer "
                    "membership bound"
                )

        unknown_owners = {
            body.owner_object_id
            for body in bodies.values()
            if body.owner_object_id is not None and body.owner_object_id not in objects
        }
        if unknown_owners:
            raise ValueError("collision body owner references an unknown object")

    def _validate_support_references(self, objects: dict[str, Any]) -> None:
        constraints_by_object = {
            constraint.supported_object_id: constraint
            for constraint in self.constraints.support_constraints
        }
        if len(constraints_by_object) != len(self.constraints.support_constraints):
            raise ValueError(
                "support constraints require at most one exact-surface predicate "
                "per supported object"
            )
        unknown_supported_objects = set(constraints_by_object) - set(objects)
        if unknown_supported_objects:
            raise ValueError(
                "support constraint references an unknown supported object"
            )

        surface_ids = {
            constraint.surface_id for constraint in self.constraints.support_constraints
        }
        contact_geometry_ids = {
            geometry_id
            for constraint in self.constraints.support_constraints
            for geometry_id in constraint.subject_contact_geometry_ids
        }
        _validate_required_ids(
            self.scene.support_surfaces,
            "surface_id",
            surface_ids,
            family_name="support_surfaces",
        )
        _validate_required_ids(
            self.scene.geometry_instances,
            "geometry_id",
            contact_geometry_ids,
            family_name="geometry_instances",
        )

        for object_id, constraint in constraints_by_object.items():
            assignment = objects[object_id].support_assignment
            if assignment.availability is not FactAvailabilityV2.KNOWN:
                raise ValueError(
                    "support constraint requires a KNOWN baseline support assignment"
                )
            if constraint.surface_id != assignment.surface_id:
                raise ValueError(
                    "support constraint surface must match baseline support assignment"
                )

        subject_id = self.constraints.allowed_edit.subject_id
        subject = objects[subject_id]
        required_supported_objects: set[str] = set()
        if subject.support_assignment.availability is FactAvailabilityV2.KNOWN:
            required_supported_objects.add(subject_id)
        surfaces = _items_by_id(self.scene.support_surfaces, "surface_id")
        for object_id, object_ in objects.items():
            if object_id == subject_id:
                continue
            assignment = object_.support_assignment
            if assignment.availability is not FactAvailabilityV2.KNOWN:
                continue
            assert assignment.surface_id is not None
            surface = surfaces.get(assignment.surface_id)
            if surface is not None and surface.owner_object_id == subject_id:
                required_supported_objects.add(object_id)
        if not required_supported_objects <= set(constraints_by_object):
            raise ValueError(
                "support constraint coverage must include the edited subject and "
                "every fixed child supported by a subject-owned surface"
            )

        geometries = _items_by_id(self.scene.geometry_instances, "geometry_id")
        for constraint in self.constraints.support_constraints:
            for geometry_id in constraint.subject_contact_geometry_ids:
                geometry = geometries.get(geometry_id)
                if geometry is not None and (
                    geometry.owner_object_id != constraint.supported_object_id
                    or geometry.role is not GeometryRoleV2.SUPPORT
                ):
                    raise ValueError(
                        "support contact geometry must be supported-object-owned "
                        "SUPPORT geometry"
                    )

        facts = self.scene.geometry_instances
        if facts.availability is FactAvailabilityV2.KNOWN and facts.completeness in {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }:
            possible = _possible_values(facts)
            for object_id, constraint in constraints_by_object.items():
                expected = {
                    item.geometry_id
                    for item in possible
                    if item.owner_object_id == object_id
                    and item.role is GeometryRoleV2.SUPPORT
                }
                if not expected:
                    raise ValueError(
                        "a supported object requires supported-object-owned SUPPORT "
                        "geometry"
                    )
                if set(constraint.subject_contact_geometry_ids) != expected:
                    raise ValueError(
                        "support constraint coverage must include all possible "
                        "supported-object SUPPORT geometry"
                    )

    def _validate_visibility_references(self, objects: dict[str, Any]) -> None:
        definitions_by_reference = {
            definition.reference: definition
            for definition in self.visibility_semantics.definitions
        }
        required_camera_ids = {self.constraints.target_relation.camera_id}
        required_geometry_ids: set[str] = set()
        required_observations: set[tuple[str, str, str, str]] = set()
        for constraint in self.constraints.visibility_constraints:
            required_camera_ids.add(constraint.camera_id)
            missing_objects = set(constraint.query_object_ids) - set(objects)
            if missing_objects:
                raise ValueError("visibility query references an unknown object")
            required_geometry_ids.update(constraint.occluder_geometry_ids)
            typed_references = (
                (
                    (
                        constraint.visible_fraction_metric_definition_id,
                        constraint.visible_fraction_metric_definition_version,
                    ),
                    VisibilityMetricKindV2.VISIBLE_FRACTION,
                ),
                (
                    (
                        constraint.image_area_metric_definition_id,
                        constraint.image_area_metric_definition_version,
                    ),
                    VisibilityMetricKindV2.IMAGE_AREA_FRACTION,
                ),
                (
                    (
                        constraint.truncated_fraction_metric_definition_id,
                        constraint.truncated_fraction_metric_definition_version,
                    ),
                    VisibilityMetricKindV2.TRUNCATED_FRACTION,
                ),
            )
            for reference, expected_kind in typed_references:
                definition = definitions_by_reference.get(reference)
                if definition is None:
                    raise ValueError(
                        "visibility constraint metric reference is absent from the "
                        "visibility semantics registry"
                    )
                if definition.kind is not expected_kind:
                    raise ValueError(
                        "visibility constraint metric reference resolves to the wrong "
                        "metric kind"
                    )
            for object_id in constraint.query_object_ids:
                for definition_id, version in _visibility_metric_refs(constraint):
                    required_observations.add(
                        (object_id, constraint.camera_id, definition_id, version)
                    )

        for observation in _possible_values(self.scene.baseline_observations):
            if (
                observation.metric_definition_id,
                observation.metric_definition_version,
            ) not in definitions_by_reference:
                raise ValueError(
                    "baseline observation metric reference is absent from the "
                    "visibility semantics registry"
                )

        _validate_required_ids(
            self.scene.cameras,
            "camera_id",
            required_camera_ids,
            family_name="cameras",
        )
        _validate_required_ids(
            self.scene.geometry_instances,
            "geometry_id",
            required_geometry_ids,
            family_name="geometry_instances",
        )
        _validate_required_observations(
            self.scene.baseline_observations,
            required_observations,
        )

        geometries = _items_by_id(self.scene.geometry_instances, "geometry_id")
        for geometry_id in required_geometry_ids:
            geometry = geometries.get(geometry_id)
            if geometry is not None and geometry.role is not GeometryRoleV2.OCCLUDER:
                raise ValueError("visibility occluder must reference OCCLUDER geometry")
            if geometry is not None and geometry.approximation not in {
                GeometryApproximationV2.EXACT,
                GeometryApproximationV2.OUTER,
            }:
                raise ValueError(
                    "visibility occluder geometry must be EXACT or an OUTER shape bound"
                )

        facts = self.scene.geometry_instances
        if facts.availability is FactAvailabilityV2.KNOWN and facts.completeness in {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }:
            expected_occluders = {
                item.geometry_id
                for item in _possible_values(facts)
                if item.role is GeometryRoleV2.OCCLUDER
            }
            for constraint in self.constraints.visibility_constraints:
                if set(constraint.occluder_geometry_ids) != expected_occluders:
                    raise ValueError(
                        "visibility occluder coverage must include all possible "
                        "OCCLUDER geometry from the sound outer membership bound"
                    )

        visual_query_ids = {
            object_id
            for constraint in self.constraints.visibility_constraints
            for object_id in constraint.query_object_ids
        }
        self._validate_geometry_role_coverage(
            visual_query_ids,
            GeometryRoleV2.VISUAL,
            "visibility queries require VISUAL geometry",
        )

    def _validate_relation_universe(self, objects: dict[str, Any]) -> None:
        provided = {
            item.key for item in self.objective.relation_damage.pair_axis_weights
        }
        subject_id = self.constraints.allowed_edit.subject_id
        target = self.constraints.target_relation
        evaluation_camera_id = self.objective.relation_damage.evaluation_camera_id
        if evaluation_camera_id != target.camera_id:
            raise ValueError("relation-damage camera must equal the target camera")
        _validate_required_ids(
            self.scene.cameras,
            "camera_id",
            {evaluation_camera_id},
            family_name="cameras",
        )
        target_key = PairAxisKeyV2(
            first_object_id=target.subject_id,
            second_object_id=target.reference_id,
            axis=target.relation_after.axis,
        )
        if target_key in provided:
            raise ValueError("relation-damage universe must exclude the target axis")

        required = {
            PairAxisKeyV2(
                first_object_id=subject_id,
                second_object_id=other_id,
                axis=axis,
            )
            for other_id in objects
            if other_id != subject_id
            for axis in RelationAxisV2
        } - {target_key}
        if not required <= provided:
            raise ValueError(
                "relation-damage universe must cover every non-target subject pair-axis"
            )

        referenced_objects = {
            object_id
            for key in provided
            for object_id in (key.first_object_id, key.second_object_id)
        }
        if not referenced_objects <= set(objects):
            raise ValueError("relation-damage universe references an unknown object")

        definitions_by_axis = {
            definition.relation.axis: definition
            for definition in self.relation_semantics.definitions
        }
        visibility_required_objects = {
            object_id
            for key in provided
            if definitions_by_axis[key.axis].requires_both_visible
            for object_id in (key.first_object_id, key.second_object_id)
        }
        target_definition = next(
            definition
            for definition in self.relation_semantics.definitions
            if definition.relation is target.relation_after
        )
        if target_definition.requires_both_visible:
            visibility_required_objects.update((target.subject_id, target.reference_id))
        positively_gated_objects = {
            object_id
            for constraint in self.constraints.visibility_constraints
            if constraint.camera_id == evaluation_camera_id
            and constraint.minimum_visible_fraction > 0.0
            and constraint.minimum_image_area_fraction > 0.0
            for object_id in constraint.query_object_ids
        }
        if not visibility_required_objects <= positively_gated_objects:
            raise ValueError(
                "relation visibility prerequisite requires both endpoints of every "
                "visibility-dependent pair to have positive visible-fraction and "
                "image-area gates in the relation-damage camera"
            )
        self._validate_geometry_role_coverage(
            referenced_objects | {target.subject_id, target.reference_id},
            GeometryRoleV2.RELATION,
            "relation universe requires RELATION geometry",
        )

    def _validate_visibility_objective(self, objects: dict[str, Any]) -> None:
        allowed_metric_keys = {
            (object_id, constraint.camera_id, definition_id, version)
            for constraint in self.constraints.visibility_constraints
            for object_id in constraint.query_object_ids
            for definition_id, version in _visibility_metric_refs(constraint)
        }
        required_observations: set[tuple[str, str, str, str]] = set()
        for item in self.objective.visibility_change.object_camera_weights:
            key = item.key
            if key.object_id not in objects:
                raise ValueError("visibility objective references an unknown object")
            metric_key = (
                key.object_id,
                key.camera_id,
                key.metric_definition_id,
                key.metric_definition_version,
            )
            if metric_key not in allowed_metric_keys:
                raise ValueError(
                    "visibility metric must be bound by a visibility constraint"
                )
            required_observations.add(metric_key)
        _validate_required_observations(
            self.scene.baseline_observations,
            required_observations,
        )

    def _validate_safety_universe(self) -> None:
        targets = {
            item.constraint_id: item
            for item in self.objective.safety_margin.aggregation.targets
        }
        if set(targets) != set(self.constraints.slack_constraint_ids):
            raise ValueError(
                "safety target coverage must exactly match slack-bearing constraints"
            )

        expected_components: dict[
            str, set[tuple[SafetyComponentKindV2, SafetySlackUnitV2]]
        ] = {
            self.constraints.position_domain.constraint_id: {
                (
                    SafetyComponentKindV2.POSITION_BOUNDARY_CLEARANCE,
                    SafetySlackUnitV2.METRE,
                )
            },
            **{
                item.constraint_id: {
                    (
                        SafetyComponentKindV2.COLLISION_CLEARANCE,
                        SafetySlackUnitV2.METRE,
                    )
                }
                for item in self.constraints.collision_constraints
            },
            **{
                item.constraint_id: {
                    (
                        SafetyComponentKindV2.SUPPORT_CONTACT_GAP_LOWER_MARGIN,
                        SafetySlackUnitV2.METRE,
                    ),
                    (
                        SafetyComponentKindV2.SUPPORT_CONTACT_GAP_UPPER_MARGIN,
                        SafetySlackUnitV2.METRE,
                    ),
                    (
                        SafetyComponentKindV2.SUPPORT_OVERLAP_AREA_MARGIN,
                        SafetySlackUnitV2.SQUARE_METRE,
                    ),
                    (
                        SafetyComponentKindV2.SUPPORT_STABILITY_INSET_MARGIN,
                        SafetySlackUnitV2.METRE,
                    ),
                }
                for item in self.constraints.support_constraints
            },
            **{
                item.constraint_id: {
                    (
                        SafetyComponentKindV2.VISIBILITY_VISIBLE_FRACTION_MARGIN,
                        SafetySlackUnitV2.FRACTION,
                    ),
                    (
                        SafetyComponentKindV2.VISIBILITY_IMAGE_AREA_FRACTION_MARGIN,
                        SafetySlackUnitV2.FRACTION,
                    ),
                    (
                        SafetyComponentKindV2.VISIBILITY_TRUNCATED_FRACTION_MARGIN,
                        SafetySlackUnitV2.FRACTION,
                    ),
                }
                for item in self.constraints.visibility_constraints
            },
        }
        target_relation = self.constraints.target_relation
        definition = next(
            item
            for item in self.relation_semantics.definitions
            if item.relation is target_relation.relation_after
        )
        expected_components[target_relation.constraint_id] = {
            (
                SafetyComponentKindV2.TARGET_RELATION_THRESHOLD_MARGIN,
                (
                    SafetySlackUnitV2.PIXEL
                    if definition.unit is MeasurementUnitV2.PIXEL
                    else SafetySlackUnitV2.METRE
                ),
            )
        }
        mismatches = {
            constraint_id
            for constraint_id, expected in expected_components.items()
            if {
                (component.kind, component.unit)
                for component in targets[constraint_id].components
            }
            != expected
        }
        if mismatches:
            raise ValueError(
                "safety target components do not match their hard constraints"
            )

    def _validate_geometry_role_coverage(
        self,
        object_ids: set[str],
        role: GeometryRoleV2,
        error: str,
    ) -> None:
        facts = self.scene.geometry_instances
        if facts.availability is not FactAvailabilityV2.KNOWN:
            return
        possible = _possible_values(facts)
        possible_by_owner = {
            object_id: tuple(
                item.geometry_id
                for item in possible
                if item.role is role and item.owner_object_id == object_id
            )
            for object_id in object_ids
        }
        complete_upper = facts.completeness in {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }
        if complete_upper and any(not values for values in possible_by_owner.values()):
            raise ValueError(error)
        if facts.completeness is FactCompletenessV2.EXACT and any(
            len(values) != 1 for values in possible_by_owner.values()
        ):
            raise ValueError(
                f"{error}; v2 requires exactly one selected {role.value} geometry per object"
            )

    def _required_fact_references(
        self,
    ) -> dict[str, tuple[FactSetV2, str, set[str]]]:
        position = self.constraints.position_domain
        collision_ids = {
            body_id
            for constraint in self.constraints.collision_constraints
            for body_id in (*constraint.subject_body_ids, *constraint.obstacle_body_ids)
        } | set(position.subject_occupancy_body_ids)
        surface_ids = {item.surface_id for item in self.constraints.support_constraints}
        geometry_ids = {
            geometry_id
            for item in self.constraints.support_constraints
            for geometry_id in item.subject_contact_geometry_ids
        } | {
            geometry_id
            for item in self.constraints.visibility_constraints
            for geometry_id in item.occluder_geometry_ids
        }
        camera_ids = (
            {
                self.constraints.target_relation.camera_id,
                self.objective.relation_damage.evaluation_camera_id,
            }
            | {item.camera_id for item in self.constraints.visibility_constraints}
            | {
                item.key.camera_id
                for item in self.objective.visibility_change.object_camera_weights
            }
        )
        return {
            "workspace_boundaries": (
                self.scene.workspace_boundaries,
                "fact_id",
                set(position.workspace_fact_ids),
            ),
            "known_free_spaces": (
                self.scene.known_free_spaces,
                "fact_id",
                set(position.known_free_space_fact_ids),
            ),
            "collision_bodies": (
                self.scene.collision_bodies,
                "body_id",
                collision_ids,
            ),
            "support_surfaces": (
                self.scene.support_surfaces,
                "surface_id",
                surface_ids,
            ),
            "geometry_instances": (
                self.scene.geometry_instances,
                "geometry_id",
                geometry_ids,
            ),
            "cameras": (self.scene.cameras, "camera_id", camera_ids),
        }

    def _observation_certification_blockers(self) -> set[str]:
        required = {
            (object_id, constraint.camera_id, definition_id, version)
            for constraint in self.constraints.visibility_constraints
            for object_id in constraint.query_object_ids
            for definition_id, version in _visibility_metric_refs(constraint)
        } | {
            (
                item.key.object_id,
                item.key.camera_id,
                item.key.metric_definition_id,
                item.key.metric_definition_version,
            )
            for item in self.objective.visibility_change.object_camera_weights
        }
        if not required:
            return set()
        facts = self.scene.baseline_observations
        if facts.availability is FactAvailabilityV2.MISSING:
            return {"MISSING_FACT:baseline_observations"}
        if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
            return {"NOT_APPLICABLE_REQUIRED_FACT:baseline_observations"}
        guaranteed = {_observation_key(item) for item in _guaranteed_values(facts)}
        prefix = (
            "OUTER_ONLY_REFERENCE"
            if facts.completeness is FactCompletenessV2.OUTER_BOUND
            else "UNGUARANTEED_REFERENCE"
        )
        blockers = {
            f"{prefix}:baseline_observations:{'|'.join(key)}"
            for key in required - guaranteed
        }
        if facts.completeness in {
            FactCompletenessV2.INNER_BOUND,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.SAMPLED,
        }:
            blockers.add(
                f"UNBRACKETED_COVERAGE:baseline_observations:{facts.completeness.value}"
            )
        return blockers

    def _geometry_selection_blockers(self) -> set[str]:
        facts = self.scene.geometry_instances
        role_objects = self._required_role_objects()
        if facts.availability is FactAvailabilityV2.MISSING:
            return {"MISSING_FACT:geometry_instances"}
        if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
            return {"NOT_APPLICABLE_REQUIRED_FACT:geometry_instances"}
        possible = _possible_values(facts)
        guaranteed = _guaranteed_values(facts)
        blockers: set[str] = set()
        for role, object_ids in role_objects.items():
            for object_id in object_ids:
                possible_ids = {
                    item.geometry_id
                    for item in possible
                    if item.role is role and item.owner_object_id == object_id
                }
                guaranteed_ids = {
                    item.geometry_id
                    for item in guaranteed
                    if item.role is role and item.owner_object_id == object_id
                }
                if len(guaranteed_ids) != 1 or possible_ids != guaranteed_ids:
                    blockers.add(
                        f"UNRESOLVED_GEOMETRY_SELECTION:{role.value}:{object_id}"
                    )
        if role_objects and facts.completeness in {
            FactCompletenessV2.INNER_BOUND,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.SAMPLED,
        }:
            blockers.add(
                f"UNBRACKETED_COVERAGE:geometry_instances:{facts.completeness.value}"
            )
        return blockers

    def _collision_certification_blockers(self) -> set[str]:
        """Close the subject collision model even when no ID is referenced.

        A constraint list cannot define its own coverage universe: otherwise
        deleting both a subject body and its constraint would turn an unknown
        solid object into a point.  Exact/bracketed inputs are validated above;
        this derived gate keeps missing and one-sided inputs explicitly
        uncertifiable.
        """

        facts = self.scene.collision_bodies
        family_name = "collision_bodies"
        if facts.availability is FactAvailabilityV2.MISSING:
            return {f"MISSING_FACT:{family_name}"}
        if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
            return {f"NOT_APPLICABLE_REQUIRED_FACT:{family_name}"}

        subject_id = self.constraints.allowed_edit.subject_id
        possible_subject = {
            body.body_id
            for body in _possible_values(facts)
            if body.owner_object_id == subject_id
        }
        guaranteed_subject = {
            body.body_id
            for body in _guaranteed_values(facts)
            if body.owner_object_id == subject_id
        }
        blockers: set[str] = set()
        if not possible_subject:
            blockers.add(f"MISSING_SUBJECT_COLLISION_BODY:{subject_id}")
        elif not guaranteed_subject:
            blockers.add(f"UNGUARANTEED_SUBJECT_COLLISION_BODY:{subject_id}")
        if facts.completeness in {
            FactCompletenessV2.INNER_BOUND,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.SAMPLED,
        }:
            blockers.add(
                f"UNBRACKETED_COVERAGE:{family_name}:{facts.completeness.value}"
            )
        return blockers

    def _support_certification_blockers(self) -> set[str]:
        """Require support facts sufficient to preserve every fixed object.

        A missing assignment on any object may conceal a direct child of the
        edited subject.  Treating that unknown as ``NOT_APPLICABLE`` could
        therefore certify an edit that silently breaks a frozen support
        relation.
        """

        subject_id = self.constraints.allowed_edit.subject_id
        objects = self.scene.objects.values or ()
        subject = next(item for item in objects if item.object_id == subject_id)
        blockers = {
            f"MISSING_FACT:support_assignment:{item.object_id}"
            for item in objects
            if item.support_assignment.availability is FactAvailabilityV2.MISSING
        }
        assignment = subject.support_assignment
        if assignment.availability is FactAvailabilityV2.MISSING:
            return blockers
        if assignment.availability is FactAvailabilityV2.NOT_APPLICABLE:
            blockers.add(
                f"NOT_APPLICABLE_REQUIRED_FACT:support_assignment:{subject_id}"
            )
            return blockers

        facts = self.scene.geometry_instances
        if facts.availability is FactAvailabilityV2.MISSING:
            blockers.add("MISSING_FACT:subject_support_geometry")
            return blockers
        if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
            blockers.add("NOT_APPLICABLE_REQUIRED_FACT:subject_support_geometry")
            return blockers
        possible = {
            geometry.geometry_id
            for geometry in _possible_values(facts)
            if geometry.owner_object_id == subject_id
            and geometry.role is GeometryRoleV2.SUPPORT
        }
        guaranteed = {
            geometry.geometry_id
            for geometry in _guaranteed_values(facts)
            if geometry.owner_object_id == subject_id
            and geometry.role is GeometryRoleV2.SUPPORT
        }
        if not possible:
            blockers.add(f"MISSING_SUBJECT_SUPPORT_GEOMETRY:{subject_id}")
        elif not guaranteed:
            blockers.add(f"UNGUARANTEED_SUBJECT_SUPPORT_GEOMETRY:{subject_id}")
        if facts.completeness in {
            FactCompletenessV2.INNER_BOUND,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.SAMPLED,
        }:
            blockers.add(
                "UNBRACKETED_COVERAGE:subject_support_geometry:"
                f"{facts.completeness.value}"
            )
        return blockers

    def _completeness_requirement_blockers(self) -> set[str]:
        blockers: set[str] = set()
        position = self.constraints.position_domain
        for family_name, facts, required_ids in (
            (
                "workspace_boundaries",
                self.scene.workspace_boundaries,
                position.workspace_fact_ids,
            ),
            (
                "known_free_spaces",
                self.scene.known_free_spaces,
                position.known_free_space_fact_ids,
            ),
        ):
            if (
                required_ids
                and facts.availability is FactAvailabilityV2.KNOWN
                and facts.completeness not in position.required_completeness
            ):
                blockers.add(
                    f"COMPLETENESS_REQUIREMENT_MISMATCH:{family_name}:"
                    f"{facts.completeness.value}"
                )

        for constraint in self.constraints.visibility_constraints:
            facts = self.scene.baseline_observations
            if (
                facts.availability is FactAvailabilityV2.KNOWN
                and facts.completeness not in constraint.accepted_baseline_completeness
            ):
                blockers.add(
                    "COMPLETENESS_REQUIREMENT_MISMATCH:baseline_observations:"
                    f"{facts.completeness.value}:"
                    f"{constraint.constraint_id}"
                )
        return blockers

    def _required_role_objects(self) -> dict[GeometryRoleV2, set[str]]:
        relation_objects = {
            object_id
            for item in self.objective.relation_damage.pair_axis_weights
            for object_id in (
                item.key.first_object_id,
                item.key.second_object_id,
            )
        } | {
            self.constraints.target_relation.subject_id,
            self.constraints.target_relation.reference_id,
        }
        visual_objects = {
            object_id
            for constraint in self.constraints.visibility_constraints
            for object_id in constraint.query_object_ids
        } | {
            item.key.object_id
            for item in self.objective.visibility_change.object_camera_weights
        }
        return {
            GeometryRoleV2.RELATION: relation_objects,
            GeometryRoleV2.VISUAL: visual_objects,
        }


def _visibility_metric_refs(
    constraint: VisibilityConstraintV2,
) -> tuple[tuple[str, str], ...]:
    return (
        (
            constraint.visible_fraction_metric_definition_id,
            constraint.visible_fraction_metric_definition_version,
        ),
        (
            constraint.image_area_metric_definition_id,
            constraint.image_area_metric_definition_version,
        ),
        (
            constraint.truncated_fraction_metric_definition_id,
            constraint.truncated_fraction_metric_definition_version,
        ),
    )


def _possible_values(facts: FactSetV2) -> tuple[V2Model, ...]:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return ()
    if facts.completeness is FactCompletenessV2.BRACKETED:
        return facts.outer_values or ()
    return facts.values or ()


def _guaranteed_values(facts: FactSetV2) -> tuple[V2Model, ...]:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return ()
    if facts.completeness is FactCompletenessV2.OUTER_BOUND:
        return ()
    if facts.completeness is FactCompletenessV2.BRACKETED:
        return facts.inner_values or ()
    return facts.values or ()


def _ids(values: Iterable[V2Model], id_field: str) -> set[str]:
    return {getattr(item, id_field) for item in values}


def _items_by_id(facts: FactSetV2, id_field: str) -> dict[str, Any]:
    return {getattr(item, id_field): item for item in _possible_values(facts)}


def _known_family_excludes_id(
    facts: FactSetV2,
    id_field: str,
    required_id: str,
) -> bool:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return False
    if facts.completeness in {
        FactCompletenessV2.INNER_BOUND,
        FactCompletenessV2.SAMPLED,
    }:
        return False
    return required_id not in _ids(_possible_values(facts), id_field)


def _validate_required_ids(
    facts: FactSetV2,
    id_field: str,
    required_ids: set[str],
    *,
    family_name: str,
) -> None:
    if not required_ids or facts.availability is FactAvailabilityV2.MISSING:
        return
    if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
        raise ValueError(f"{family_name} cannot be NOT_APPLICABLE when constrained")
    excluded = {
        required_id
        for required_id in required_ids
        if _known_family_excludes_id(facts, id_field, required_id)
    }
    if excluded:
        raise ValueError(f"{family_name} contains dangling required IDs")


def _observation_key(observation: BaselineObservationV2) -> tuple[str, str, str, str]:
    return (
        observation.object_id,
        observation.camera_id,
        observation.metric_definition_id,
        observation.metric_definition_version,
    )


def _validate_required_observations(
    facts: FactSetV2[BaselineObservationV2],
    required_keys: set[tuple[str, str, str, str]],
) -> None:
    if not required_keys or facts.availability is FactAvailabilityV2.MISSING:
        return
    if facts.availability is FactAvailabilityV2.NOT_APPLICABLE:
        raise ValueError(
            "baseline observations cannot be NOT_APPLICABLE when visibility is used"
        )
    possible = {_observation_key(item) for item in _possible_values(facts)}
    if (
        facts.completeness
        not in {FactCompletenessV2.INNER_BOUND, FactCompletenessV2.SAMPLED}
        and not required_keys <= possible
    ):
        raise ValueError("visibility metric lacks a possible normalized baseline")
