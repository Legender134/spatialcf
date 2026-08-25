"""Platform-neutral Canonical relation and hard-constraint contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FactCompletenessV2,
    FiniteFloat,
    NonNegativeFiniteFloat,
    SchemaIdentityV2,
)

UnitInterval = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


def _sort_unique_ids(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(values))


class TranslationAxis(StrEnum):
    X = "X"
    Y = "Y"


class ImmutableField(StrEnum):
    SUBJECT_Z = "SUBJECT_Z"
    SUBJECT_ROTATION = "SUBJECT_ROTATION"
    OTHER_OBJECTS = "OTHER_OBJECTS"
    CAMERAS = "CAMERAS"


class BoundaryPolicy(StrEnum):
    CLOSED = "CLOSED"
    STRICT_INTERIOR = "STRICT_INTERIOR"


class RegionAggregation(StrEnum):
    """How referenced planar fact regions form one semantic region."""

    UNION = "UNION"
    INTERSECTION = "INTERSECTION"


class PositionRegionInterpretation(StrEnum):
    """Whether selected regions constrain the edit anchor or occupied solid."""

    SUBJECT_ANCHOR_LOCUS = "SUBJECT_ANCHOR_LOCUS"
    SUBJECT_OCCUPANCY_CONTAINED = "SUBJECT_OCCUPANCY_CONTAINED"


class GeometrySetAggregation(StrEnum):
    CLOSED_SOLID_UNION = "CLOSED_SOLID_UNION"


class CollisionClearanceMetric(StrEnum):
    """Closed collision predicate understood by the platform-neutral core."""

    SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE = (
        "SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE"
    )


class SupportContactExceptionPolicy(StrEnum):
    RETAIN_SOLID_INTERIOR_DISJOINT_WAIVE_POSITIVE_CLEARANCE_ONLY_WHEN_NAMED_SUPPORT_PREDICATE_HOLDS = "RETAIN_SOLID_INTERIOR_DISJOINT_WAIVE_POSITIVE_CLEARANCE_ONLY_WHEN_NAMED_SUPPORT_PREDICATE_HOLDS"


class SupportAssignmentPolicy(StrEnum):
    EXACT_SURFACE = "EXACT_SURFACE"


class SupportContactFeature(StrEnum):
    LOWEST_FACE_ALONG_SURFACE_NORMAL = "LOWEST_FACE_ALONG_SURFACE_NORMAL"


class SupportContactAggregation(StrEnum):
    UNION_ALL_SELECTED_FEATURES = "UNION_ALL_SELECTED_FEATURES"


class SupportOverlapMetric(StrEnum):
    PROJECTED_CONTACT_UNION_INTERSECTION_AREA = (
        "PROJECTED_CONTACT_UNION_INTERSECTION_AREA"
    )


class SupportStabilityMetric(StrEnum):
    FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET = (
        "FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET"
    )


class VisibilityMaskPolicy(StrEnum):
    FULL_OBJECT = "FULL_OBJECT"


class OccluderSoundnessPolicy(StrEnum):
    EXACT_OR_OUTER_SHAPE_BOUND = "EXACT_OR_OUTER_SHAPE_BOUND"


class VisibilityMetricKind(StrEnum):
    VISIBLE_FRACTION = "VISIBLE_FRACTION"
    IMAGE_AREA_FRACTION = "IMAGE_AREA_FRACTION"
    TRUNCATED_FRACTION = "TRUNCATED_FRACTION"


class VisibilityMetricFormula(StrEnum):
    VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA = (
        "VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA"
    )
    VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA = (
        "VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA"
    )
    VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA = (
        "VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA"
    )
    ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA = (
        "ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA"
    )


class VisibilityAreaMeasure(StrEnum):
    CONTINUOUS_PIXEL_PLANE_AREA = "CONTINUOUS_PIXEL_PLANE_AREA"


class VisibilityDepthPolicy(StrEnum):
    NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES = "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES"


class RelationAxis(StrEnum):
    HORIZONTAL = "HORIZONTAL"
    DEPTH = "DEPTH"
    DISTANCE = "DISTANCE"


class Relation(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FRONT = "FRONT"
    BEHIND = "BEHIND"
    NEAR = "NEAR"
    FAR = "FAR"

    @property
    def axis(self) -> RelationAxis:
        return {
            Relation.LEFT: RelationAxis.HORIZONTAL,
            Relation.RIGHT: RelationAxis.HORIZONTAL,
            Relation.FRONT: RelationAxis.DEPTH,
            Relation.BEHIND: RelationAxis.DEPTH,
            Relation.NEAR: RelationAxis.DISTANCE,
            Relation.FAR: RelationAxis.DISTANCE,
        }[self]

    @property
    def opposite(self) -> Relation:
        return {
            Relation.LEFT: Relation.RIGHT,
            Relation.RIGHT: Relation.LEFT,
            Relation.FRONT: Relation.BEHIND,
            Relation.BEHIND: Relation.FRONT,
            Relation.NEAR: Relation.FAR,
            Relation.FAR: Relation.NEAR,
        }[self]


class RelationMeasurement(StrEnum):
    PROJECTED_CENTER_DELTA_X = "PROJECTED_CENTER_DELTA_X"
    CAMERA_DEPTH_DELTA = "CAMERA_DEPTH_DELTA"
    SHAPE_GAP_XY = "SHAPE_GAP_XY"


class RelationRepresentativePoint(StrEnum):
    RELATION_GEOMETRY_VOLUME_CENTROID = "RELATION_GEOMETRY_VOLUME_CENTROID"


class MeasurementOperandOrder(StrEnum):
    FIRST_MINUS_SECOND = "FIRST_MINUS_SECOND"


class RelationTolerancePolicy(StrEnum):
    SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET = (
        "SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET"
    )


class MeasurementComparator(StrEnum):
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"


class MeasurementUnit(StrEnum):
    PIXEL = "PIXEL"
    METRE = "METRE"


class AllowedEdit(CanonicalModel):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    translation_axes: tuple[TranslationAxis, ...] = (
        TranslationAxis.X,
        TranslationAxis.Y,
    )
    immutable_fields: tuple[ImmutableField, ...] = (
        ImmutableField.SUBJECT_Z,
        ImmutableField.SUBJECT_ROTATION,
        ImmutableField.OTHER_OBJECTS,
        ImmutableField.CAMERAS,
    )

    @model_validator(mode="after")
    def validate_exact_edit_surface(self) -> Self:
        if self.translation_axes != (TranslationAxis.X, TranslationAxis.Y):
            raise ValueError("allowed edit must be exactly XY translation")
        expected = (
            ImmutableField.SUBJECT_Z,
            ImmutableField.SUBJECT_ROTATION,
            ImmutableField.OTHER_OBJECTS,
            ImmutableField.CAMERAS,
        )
        if self.immutable_fields != expected:
            raise ValueError("allowed edit must freeze all required immutable fields")
        return self


class AllowedPositionDomainConstraint(CanonicalModel):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    workspace_fact_ids: tuple[CanonicalId, ...]
    workspace_aggregation: RegionAggregation
    known_free_space_fact_ids: tuple[CanonicalId, ...] = ()
    known_free_space_aggregation: RegionAggregation | None = None
    region_interpretation: PositionRegionInterpretation
    subject_occupancy_body_ids: tuple[CanonicalId, ...] = ()
    subject_occupancy_aggregation: GeometrySetAggregation | None = None
    boundary_policy: BoundaryPolicy
    required_completeness: tuple[FactCompletenessV2, ...]
    minimum_boundary_clearance_m: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        if not self.workspace_fact_ids:
            raise ValueError("position domain requires at least one workspace fact")
        object.__setattr__(
            self,
            "workspace_fact_ids",
            _sort_unique_ids(self.workspace_fact_ids, label="workspace fact IDs"),
        )
        object.__setattr__(
            self,
            "known_free_space_fact_ids",
            _sort_unique_ids(
                self.known_free_space_fact_ids,
                label="known free-space fact IDs",
            ),
        )
        if bool(self.known_free_space_fact_ids) is (
            self.known_free_space_aggregation is None
        ):
            raise ValueError(
                "known free-space aggregation is required exactly when free-space "
                "facts are referenced"
            )
        occupancy_ids = _sort_unique_ids(
            self.subject_occupancy_body_ids,
            label="subject occupancy body IDs",
        )
        object.__setattr__(self, "subject_occupancy_body_ids", occupancy_ids)
        if (
            self.region_interpretation
            is PositionRegionInterpretation.SUBJECT_ANCHOR_LOCUS
        ):
            if occupancy_ids or self.subject_occupancy_aggregation is not None:
                raise ValueError(
                    "anchor locus position semantics forbid occupancy bodies and "
                    "aggregation"
                )
        elif (
            not occupancy_ids
            or self.subject_occupancy_aggregation
            is not GeometrySetAggregation.CLOSED_SOLID_UNION
        ):
            raise ValueError(
                "occupied-space position semantics require subject occupancy body "
                "IDs aggregated as a closed solid union"
            )
        if not self.required_completeness:
            raise ValueError("required completeness must not be empty")
        if len(self.required_completeness) != len(set(self.required_completeness)):
            raise ValueError("required completeness values must be unique")
        allowed = {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.INNER_BOUND,
            FactCompletenessV2.OUTER_BOUND,
            FactCompletenessV2.BRACKETED,
        }
        if not set(self.required_completeness) <= allowed:
            raise ValueError(
                "position domain only accepts sound exact or bounded facts"
            )
        order = {value: index for index, value in enumerate(FactCompletenessV2)}
        object.__setattr__(
            self,
            "required_completeness",
            tuple(sorted(self.required_completeness, key=order.__getitem__)),
        )
        return self


class SupportContactException(CanonicalModel):
    """One exact body pair whose positive clearance alone may be waived.

    The named support predicate must hold, while closed-solid interiors remain
    disjoint.  This is never a penetration or whole-pair collision exemption.
    """

    support_constraint_id: CanonicalId
    subject_body_id: CanonicalId
    obstacle_body_id: CanonicalId
    policy: SupportContactExceptionPolicy

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.support_constraint_id,
            self.subject_body_id,
            self.obstacle_body_id,
            self.policy.value,
        )


class CollisionConstraint(CanonicalModel):
    constraint_id: CanonicalId
    subject_body_ids: tuple[CanonicalId, ...]
    obstacle_body_ids: tuple[CanonicalId, ...]
    clearance_metric: CollisionClearanceMetric
    boundary_policy: BoundaryPolicy
    minimum_clearance_m: NonNegativeFiniteFloat
    support_contact_exceptions: tuple[SupportContactException, ...] = ()

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        if not self.subject_body_ids or not self.obstacle_body_ids:
            raise ValueError("collision body ID sets must not be empty")
        subject = _sort_unique_ids(self.subject_body_ids, label="subject body IDs")
        obstacles = _sort_unique_ids(self.obstacle_body_ids, label="obstacle body IDs")
        if set(subject) & set(obstacles):
            raise ValueError("subject and obstacle body IDs must be disjoint")
        object.__setattr__(self, "subject_body_ids", subject)
        object.__setattr__(self, "obstacle_body_ids", obstacles)
        exceptions = tuple(
            sorted(self.support_contact_exceptions, key=lambda item: item.sort_key)
        )
        if len(exceptions) != len(set(exceptions)):
            raise ValueError("support-contact exceptions must be unique")
        if any(
            item.subject_body_id not in subject
            or item.obstacle_body_id not in obstacles
            for item in exceptions
        ):
            raise ValueError(
                "support-contact exception must reference an exact collision body pair"
            )
        object.__setattr__(
            self,
            "support_contact_exceptions",
            exceptions,
        )
        return self


class SupportConstraint(CanonicalModel):
    constraint_id: CanonicalId
    supported_object_id: CanonicalId
    surface_id: CanonicalId
    subject_contact_geometry_ids: tuple[CanonicalId, ...]
    contact_feature: SupportContactFeature
    contact_aggregation: SupportContactAggregation
    contact_gap_min_m: FiniteFloat
    contact_gap_max_m: FiniteFloat
    overlap_metric: SupportOverlapMetric
    minimum_overlap_area_m2: NonNegativeFiniteFloat
    stability_metric: SupportStabilityMetric
    stability_margin_m: NonNegativeFiniteFloat
    boundary_policy: BoundaryPolicy
    assignment_policy: SupportAssignmentPolicy

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        if not self.subject_contact_geometry_ids:
            raise ValueError("support requires subject contact geometry")
        object.__setattr__(
            self,
            "subject_contact_geometry_ids",
            _sort_unique_ids(
                self.subject_contact_geometry_ids,
                label="subject contact geometry IDs",
            ),
        )
        if self.contact_gap_min_m > self.contact_gap_max_m:
            raise ValueError("contact gap minimum must not exceed maximum")
        return self


class VisibilityMetricDefinition(CanonicalModel):
    """One closed analytic visibility formula, never an adapter callback."""

    metric_definition_id: CanonicalId
    metric_definition_version: CanonicalId
    kind: VisibilityMetricKind
    formula: VisibilityMetricFormula
    area_measure: VisibilityAreaMeasure
    depth_policy: VisibilityDepthPolicy

    @model_validator(mode="after")
    def validate_formula_kind(self) -> Self:
        expected = {
            VisibilityMetricKind.VISIBLE_FRACTION: (
                VisibilityMetricFormula.VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA,
            ),
            VisibilityMetricKind.IMAGE_AREA_FRACTION: (
                VisibilityMetricFormula.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA,
                VisibilityMetricFormula.VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA,
            ),
            VisibilityMetricKind.TRUNCATED_FRACTION: (
                VisibilityMetricFormula.ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA,
            ),
        }[self.kind]
        if self.formula not in expected:
            raise ValueError("visibility metric formula does not match its metric kind")
        return self

    @property
    def reference(self) -> tuple[str, str]:
        return self.metric_definition_id, self.metric_definition_version


class VisibilitySemantics(CanonicalModel):
    """Complete versioned formula registry consumed by the pure core."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="visibility-semantics")
    )
    semantics_id: CanonicalId
    definitions: tuple[VisibilityMetricDefinition, ...]

    @model_validator(mode="after")
    def validate_complete_registry(self) -> Self:
        if self.schema_identity.schema_name != "visibility-semantics":
            raise ValueError("visibility semantics schema identity must be fixed")
        by_kind = {definition.kind: definition for definition in self.definitions}
        if len(by_kind) != len(self.definitions) or set(by_kind) != set(
            VisibilityMetricKind
        ):
            raise ValueError(
                "visibility semantics require exactly one definition per metric kind"
            )
        references = tuple(item.reference for item in self.definitions)
        if len(references) != len(set(references)):
            raise ValueError("visibility metric references must be pairwise distinct")
        object.__setattr__(
            self,
            "definitions",
            tuple(by_kind[kind] for kind in VisibilityMetricKind),
        )
        return self


class VisibilityConstraint(CanonicalModel):
    """Hard visibility policy over three explicitly versioned normalized metrics."""

    constraint_id: CanonicalId
    visibility_semantics_id: CanonicalId
    camera_id: CanonicalId
    query_object_ids: tuple[CanonicalId, ...]
    occluder_geometry_ids: tuple[CanonicalId, ...] = ()
    visible_fraction_metric_definition_id: CanonicalId
    visible_fraction_metric_definition_version: CanonicalId
    image_area_metric_definition_id: CanonicalId
    image_area_metric_definition_version: CanonicalId
    truncated_fraction_metric_definition_id: CanonicalId
    truncated_fraction_metric_definition_version: CanonicalId
    mask_policy: VisibilityMaskPolicy
    occluder_soundness_policy: OccluderSoundnessPolicy
    minimum_visible_fraction: UnitInterval
    minimum_image_area_fraction: UnitInterval
    maximum_truncated_fraction: UnitInterval
    threshold_boundary_policy: BoundaryPolicy
    accepted_baseline_completeness: tuple[FactCompletenessV2, ...]

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        if not self.query_object_ids:
            raise ValueError("visibility query object IDs must not be empty")
        object.__setattr__(
            self,
            "query_object_ids",
            _sort_unique_ids(self.query_object_ids, label="query object IDs"),
        )
        object.__setattr__(
            self,
            "occluder_geometry_ids",
            _sort_unique_ids(
                self.occluder_geometry_ids,
                label="occluder geometry IDs",
            ),
        )
        if not self.accepted_baseline_completeness:
            raise ValueError("accepted baseline completeness must not be empty")
        if len(self.accepted_baseline_completeness) != len(
            set(self.accepted_baseline_completeness)
        ):
            raise ValueError("accepted baseline completeness must be unique")
        allowed_baseline = {
            FactCompletenessV2.EXACT,
            FactCompletenessV2.BRACKETED,
        }
        if not set(self.accepted_baseline_completeness) <= allowed_baseline:
            raise ValueError(
                "visibility baselines require EXACT or BRACKETED completeness"
            )
        order = {value: index for index, value in enumerate(FactCompletenessV2)}
        object.__setattr__(
            self,
            "accepted_baseline_completeness",
            tuple(
                sorted(
                    self.accepted_baseline_completeness,
                    key=order.__getitem__,
                )
            ),
        )
        metric_refs = {
            (
                self.visible_fraction_metric_definition_id,
                self.visible_fraction_metric_definition_version,
            ),
            (
                self.image_area_metric_definition_id,
                self.image_area_metric_definition_version,
            ),
            (
                self.truncated_fraction_metric_definition_id,
                self.truncated_fraction_metric_definition_version,
            ),
        }
        if len(metric_refs) != 3:
            raise ValueError(
                "visible, image-area, and truncated metrics must be pairwise distinct"
            )
        return self


class RelationDefinition(CanonicalModel):
    relation: Relation
    measurement: RelationMeasurement
    comparator: MeasurementComparator
    threshold: FiniteFloat
    unit: MeasurementUnit
    representative_point: RelationRepresentativePoint | None
    operand_order: MeasurementOperandOrder
    boundary_policy: BoundaryPolicy
    tolerance: NonNegativeFiniteFloat
    tolerance_policy: RelationTolerancePolicy
    requires_both_visible: bool

    @model_validator(mode="after")
    def validate_measurement_contract(self) -> Self:
        expected_measurement = {
            RelationAxis.HORIZONTAL: RelationMeasurement.PROJECTED_CENTER_DELTA_X,
            RelationAxis.DEPTH: RelationMeasurement.CAMERA_DEPTH_DELTA,
            RelationAxis.DISTANCE: RelationMeasurement.SHAPE_GAP_XY,
        }[self.relation.axis]
        expected_unit = (
            MeasurementUnit.PIXEL
            if self.relation.axis is RelationAxis.HORIZONTAL
            else MeasurementUnit.METRE
        )
        expected_comparator = (
            MeasurementComparator.LESS_THAN
            if self.relation in {Relation.LEFT, Relation.FRONT, Relation.NEAR}
            else MeasurementComparator.GREATER_THAN
        )
        if self.measurement is not expected_measurement:
            raise ValueError("relation measurement does not match its axis")
        if self.unit is not expected_unit:
            raise ValueError("relation measurement unit does not match its axis")
        if self.comparator is not expected_comparator:
            raise ValueError("relation comparator does not match its direction")
        is_point_measurement = self.measurement in {
            RelationMeasurement.PROJECTED_CENTER_DELTA_X,
            RelationMeasurement.CAMERA_DEPTH_DELTA,
        }
        if is_point_measurement and self.representative_point is not (
            RelationRepresentativePoint.RELATION_GEOMETRY_VOLUME_CENTROID
        ):
            raise ValueError(
                "point relation measurement requires the relation-geometry "
                "representative point"
            )
        if not is_point_measurement and self.representative_point is not None:
            raise ValueError(
                "shape-gap relation measurement must not carry a representative point"
            )
        return self


class RelationSemantics(CanonicalModel):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="relation-semantics")
    )
    semantics_id: CanonicalId
    definitions: tuple[RelationDefinition, ...]

    @model_validator(mode="after")
    def validate_complete_relation_set(self) -> Self:
        if self.schema_identity.schema_name != "relation-semantics":
            raise ValueError("relation semantics schema identity must be fixed")
        by_relation = {
            definition.relation: definition for definition in self.definitions
        }
        if len(by_relation) != len(self.definitions) or set(by_relation) != set(
            Relation
        ):
            raise ValueError(
                "relation semantics require exactly one definition per relation"
            )
        for first, second in (
            (Relation.LEFT, Relation.RIGHT),
            (Relation.FRONT, Relation.BEHIND),
            (Relation.NEAR, Relation.FAR),
        ):
            left = by_relation[first]
            right = by_relation[second]
            if (
                left.measurement is not right.measurement
                or left.unit is not right.unit
                or left.representative_point is not right.representative_point
                or left.operand_order is not right.operand_order
                or left.boundary_policy is not right.boundary_policy
                or left.tolerance != right.tolerance
                or left.tolerance_policy is not right.tolerance_policy
                or left.requires_both_visible is not right.requires_both_visible
            ):
                raise ValueError(
                    "opposite relation definitions must share measurement semantics"
                )
            if left.threshold + left.tolerance >= right.threshold - right.tolerance:
                raise ValueError(
                    "opposite relation thresholds and tolerance brackets must be "
                    "strictly ordered and disjoint"
                )
        object.__setattr__(
            self,
            "definitions",
            tuple(by_relation[relation] for relation in Relation),
        )
        return self


class TargetRelationConstraint(CanonicalModel):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    reference_id: CanonicalId
    camera_id: CanonicalId
    relation_before: Relation
    relation_after: Relation
    semantics_id: CanonicalId

    @model_validator(mode="after")
    def validate_opposite_relation(self) -> Self:
        if self.subject_id == self.reference_id:
            raise ValueError("target subject and reference must differ")
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("target relation must change to its opposite on one axis")
        return self


class CanonicalConstraintSet(CanonicalModel):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="canonical-constraint-set")
    )
    constraint_set_id: CanonicalId
    allowed_edit: AllowedEdit
    position_domain: AllowedPositionDomainConstraint
    collision_constraints: tuple[CollisionConstraint, ...] = ()
    support_constraints: tuple[SupportConstraint, ...] = ()
    visibility_constraints: tuple[VisibilityConstraint, ...]
    target_relation: TargetRelationConstraint

    @model_validator(mode="after")
    def validate_constraint_graph(self) -> Self:
        if self.schema_identity.schema_name != "canonical-constraint-set":
            raise ValueError("constraint-set schema identity must be fixed")
        subject_id = self.allowed_edit.subject_id
        if self.position_domain.subject_id != subject_id:
            raise ValueError("position-domain subject must match allowed edit")
        if self.target_relation.subject_id != subject_id:
            raise ValueError("target subject must match allowed edit")

        object.__setattr__(
            self,
            "collision_constraints",
            tuple(
                sorted(self.collision_constraints, key=lambda item: item.constraint_id)
            ),
        )
        object.__setattr__(
            self,
            "support_constraints",
            tuple(
                sorted(self.support_constraints, key=lambda item: item.constraint_id)
            ),
        )
        object.__setattr__(
            self,
            "visibility_constraints",
            tuple(
                sorted(self.visibility_constraints, key=lambda item: item.constraint_id)
            ),
        )

        ids = (
            self.allowed_edit.constraint_id,
            self.position_domain.constraint_id,
            *(item.constraint_id for item in self.collision_constraints),
            *(item.constraint_id for item in self.support_constraints),
            *(item.constraint_id for item in self.visibility_constraints),
            self.target_relation.constraint_id,
        )
        if len(ids) != len(set(ids)):
            raise ValueError("constraint IDs must be globally unique")

        support_by_id = {item.constraint_id: item for item in self.support_constraints}
        for collision in self.collision_constraints:
            for exception in collision.support_contact_exceptions:
                support = support_by_id.get(exception.support_constraint_id)
                if support is None:
                    raise ValueError(
                        "collision support-contact exceptions must reference support "
                        "constraints"
                    )
                if support.supported_object_id != subject_id:
                    raise ValueError(
                        "collision support-contact exception must reference support "
                        "for the edited subject"
                    )

        required_queries = {
            self.target_relation.subject_id,
            self.target_relation.reference_id,
        }
        if not any(
            item.camera_id == self.target_relation.camera_id
            and required_queries <= set(item.query_object_ids)
            for item in self.visibility_constraints
        ):
            raise ValueError(
                "target subject and reference require query visibility in target camera"
            )
        return self

    @property
    def constraint_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    self.allowed_edit.constraint_id,
                    self.position_domain.constraint_id,
                    *(item.constraint_id for item in self.collision_constraints),
                    *(item.constraint_id for item in self.support_constraints),
                    *(item.constraint_id for item in self.visibility_constraints),
                    self.target_relation.constraint_id,
                )
            )
        )

    @property
    def slack_constraint_ids(self) -> tuple[str, ...]:
        """Hard predicates with a mathematical margin in the objective.

        ``AllowedEdit`` is a permission surface, not a predicate with a
        continuous slack, so it is deliberately excluded.
        """

        return tuple(
            sorted(
                (
                    self.position_domain.constraint_id,
                    *(item.constraint_id for item in self.collision_constraints),
                    *(item.constraint_id for item in self.support_constraints),
                    *(item.constraint_id for item in self.visibility_constraints),
                    self.target_relation.constraint_id,
                )
            )
        )
