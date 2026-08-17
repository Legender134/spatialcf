"""Platform-neutral Canonical v2 relation and hard-constraint contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FactCompletenessV2,
    FiniteFloat,
    NonNegativeFiniteFloat,
    SchemaIdentityV2,
    V2Model,
)

UnitInterval = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False, ge=0.0, le=1.0),
]


def _sort_unique_ids(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(values))


class TranslationAxisV2(StrEnum):
    X = "X"
    Y = "Y"


class ImmutableFieldV2(StrEnum):
    SUBJECT_Z = "SUBJECT_Z"
    SUBJECT_ROTATION = "SUBJECT_ROTATION"
    OTHER_OBJECTS = "OTHER_OBJECTS"
    CAMERAS = "CAMERAS"


class BoundaryPolicyV2(StrEnum):
    CLOSED = "CLOSED"
    STRICT_INTERIOR = "STRICT_INTERIOR"


class RegionAggregationV2(StrEnum):
    """How referenced planar fact regions form one semantic region."""

    UNION = "UNION"
    INTERSECTION = "INTERSECTION"


class PositionRegionInterpretationV2(StrEnum):
    """Whether selected regions constrain the edit anchor or occupied solid."""

    SUBJECT_ANCHOR_LOCUS = "SUBJECT_ANCHOR_LOCUS"
    SUBJECT_OCCUPANCY_CONTAINED = "SUBJECT_OCCUPANCY_CONTAINED"


class GeometrySetAggregationV2(StrEnum):
    CLOSED_SOLID_UNION = "CLOSED_SOLID_UNION"


class CollisionClearanceMetricV2(StrEnum):
    """Closed collision predicate understood by the platform-neutral core."""

    SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE = (
        "SOLID_INTERIOR_DISJOINT_AND_EUCLIDEAN_CLEARANCE"
    )


class SupportContactExceptionPolicyV2(StrEnum):
    RETAIN_SOLID_INTERIOR_DISJOINT_WAIVE_POSITIVE_CLEARANCE_ONLY_WHEN_NAMED_SUPPORT_PREDICATE_HOLDS = "RETAIN_SOLID_INTERIOR_DISJOINT_WAIVE_POSITIVE_CLEARANCE_ONLY_WHEN_NAMED_SUPPORT_PREDICATE_HOLDS"


class SupportAssignmentPolicyV2(StrEnum):
    EXACT_SURFACE = "EXACT_SURFACE"


class SupportContactFeatureV2(StrEnum):
    LOWEST_FACE_ALONG_SURFACE_NORMAL = "LOWEST_FACE_ALONG_SURFACE_NORMAL"


class SupportContactAggregationV2(StrEnum):
    UNION_ALL_SELECTED_FEATURES = "UNION_ALL_SELECTED_FEATURES"


class SupportOverlapMetricV2(StrEnum):
    PROJECTED_CONTACT_UNION_INTERSECTION_AREA = (
        "PROJECTED_CONTACT_UNION_INTERSECTION_AREA"
    )


class SupportStabilityMetricV2(StrEnum):
    FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET = (
        "FULL_CONTACT_UNION_CONTAINED_IN_SURFACE_INSET"
    )


class VisibilityMaskPolicyV2(StrEnum):
    FULL_OBJECT = "FULL_OBJECT"


class OccluderSoundnessPolicyV2(StrEnum):
    EXACT_OR_OUTER_SHAPE_BOUND = "EXACT_OR_OUTER_SHAPE_BOUND"


class VisibilityMetricKindV2(StrEnum):
    VISIBLE_FRACTION = "VISIBLE_FRACTION"
    IMAGE_AREA_FRACTION = "IMAGE_AREA_FRACTION"
    TRUNCATED_FRACTION = "TRUNCATED_FRACTION"


class VisibilityMetricFormulaV2(StrEnum):
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


class VisibilityAreaMeasureV2(StrEnum):
    CONTINUOUS_PIXEL_PLANE_AREA = "CONTINUOUS_PIXEL_PLANE_AREA"


class VisibilityDepthPolicyV2(StrEnum):
    NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES = "NEAREST_POSITIVE_CAMERA_DEPTH_OCCLUDES"


class RelationAxisV2(StrEnum):
    HORIZONTAL = "HORIZONTAL"
    DEPTH = "DEPTH"
    DISTANCE = "DISTANCE"


class RelationV2(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FRONT = "FRONT"
    BEHIND = "BEHIND"
    NEAR = "NEAR"
    FAR = "FAR"

    @property
    def axis(self) -> RelationAxisV2:
        return {
            RelationV2.LEFT: RelationAxisV2.HORIZONTAL,
            RelationV2.RIGHT: RelationAxisV2.HORIZONTAL,
            RelationV2.FRONT: RelationAxisV2.DEPTH,
            RelationV2.BEHIND: RelationAxisV2.DEPTH,
            RelationV2.NEAR: RelationAxisV2.DISTANCE,
            RelationV2.FAR: RelationAxisV2.DISTANCE,
        }[self]

    @property
    def opposite(self) -> RelationV2:
        return {
            RelationV2.LEFT: RelationV2.RIGHT,
            RelationV2.RIGHT: RelationV2.LEFT,
            RelationV2.FRONT: RelationV2.BEHIND,
            RelationV2.BEHIND: RelationV2.FRONT,
            RelationV2.NEAR: RelationV2.FAR,
            RelationV2.FAR: RelationV2.NEAR,
        }[self]


class RelationMeasurementV2(StrEnum):
    PROJECTED_CENTER_DELTA_X = "PROJECTED_CENTER_DELTA_X"
    CAMERA_DEPTH_DELTA = "CAMERA_DEPTH_DELTA"
    SHAPE_GAP_XY = "SHAPE_GAP_XY"


class RelationRepresentativePointV2(StrEnum):
    RELATION_GEOMETRY_VOLUME_CENTROID = "RELATION_GEOMETRY_VOLUME_CENTROID"


class MeasurementOperandOrderV2(StrEnum):
    FIRST_MINUS_SECOND = "FIRST_MINUS_SECOND"


class RelationTolerancePolicyV2(StrEnum):
    SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET = (
        "SYMMETRIC_INNER_OUTER_MEASUREMENT_BRACKET"
    )


class MeasurementComparatorV2(StrEnum):
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"


class MeasurementUnitV2(StrEnum):
    PIXEL = "PIXEL"
    METRE = "METRE"


class AllowedEditV2(V2Model):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    translation_axes: tuple[TranslationAxisV2, ...] = (
        TranslationAxisV2.X,
        TranslationAxisV2.Y,
    )
    immutable_fields: tuple[ImmutableFieldV2, ...] = (
        ImmutableFieldV2.SUBJECT_Z,
        ImmutableFieldV2.SUBJECT_ROTATION,
        ImmutableFieldV2.OTHER_OBJECTS,
        ImmutableFieldV2.CAMERAS,
    )

    @model_validator(mode="after")
    def validate_exact_edit_surface(self) -> Self:
        if self.translation_axes != (TranslationAxisV2.X, TranslationAxisV2.Y):
            raise ValueError("allowed edit must be exactly XY translation")
        expected = (
            ImmutableFieldV2.SUBJECT_Z,
            ImmutableFieldV2.SUBJECT_ROTATION,
            ImmutableFieldV2.OTHER_OBJECTS,
            ImmutableFieldV2.CAMERAS,
        )
        if self.immutable_fields != expected:
            raise ValueError("allowed edit must freeze all required immutable fields")
        return self


class AllowedPositionDomainConstraintV2(V2Model):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    workspace_fact_ids: tuple[CanonicalId, ...]
    workspace_aggregation: RegionAggregationV2
    known_free_space_fact_ids: tuple[CanonicalId, ...] = ()
    known_free_space_aggregation: RegionAggregationV2 | None = None
    region_interpretation: PositionRegionInterpretationV2
    subject_occupancy_body_ids: tuple[CanonicalId, ...] = ()
    subject_occupancy_aggregation: GeometrySetAggregationV2 | None = None
    boundary_policy: BoundaryPolicyV2
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
            is PositionRegionInterpretationV2.SUBJECT_ANCHOR_LOCUS
        ):
            if occupancy_ids or self.subject_occupancy_aggregation is not None:
                raise ValueError(
                    "anchor locus position semantics forbid occupancy bodies and "
                    "aggregation"
                )
        elif (
            not occupancy_ids
            or self.subject_occupancy_aggregation
            is not GeometrySetAggregationV2.CLOSED_SOLID_UNION
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


class SupportContactExceptionV2(V2Model):
    """One exact body pair whose positive clearance alone may be waived.

    The named support predicate must hold, while closed-solid interiors remain
    disjoint.  This is never a penetration or whole-pair collision exemption.
    """

    support_constraint_id: CanonicalId
    subject_body_id: CanonicalId
    obstacle_body_id: CanonicalId
    policy: SupportContactExceptionPolicyV2

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.support_constraint_id,
            self.subject_body_id,
            self.obstacle_body_id,
            self.policy.value,
        )


class CollisionConstraintV2(V2Model):
    constraint_id: CanonicalId
    subject_body_ids: tuple[CanonicalId, ...]
    obstacle_body_ids: tuple[CanonicalId, ...]
    clearance_metric: CollisionClearanceMetricV2
    boundary_policy: BoundaryPolicyV2
    minimum_clearance_m: NonNegativeFiniteFloat
    support_contact_exceptions: tuple[SupportContactExceptionV2, ...] = ()

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


class SupportConstraintV2(V2Model):
    constraint_id: CanonicalId
    supported_object_id: CanonicalId
    surface_id: CanonicalId
    subject_contact_geometry_ids: tuple[CanonicalId, ...]
    contact_feature: SupportContactFeatureV2
    contact_aggregation: SupportContactAggregationV2
    contact_gap_min_m: FiniteFloat
    contact_gap_max_m: FiniteFloat
    overlap_metric: SupportOverlapMetricV2
    minimum_overlap_area_m2: NonNegativeFiniteFloat
    stability_metric: SupportStabilityMetricV2
    stability_margin_m: NonNegativeFiniteFloat
    boundary_policy: BoundaryPolicyV2
    assignment_policy: SupportAssignmentPolicyV2

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


class VisibilityMetricDefinitionV2(V2Model):
    """One closed analytic visibility formula, never an adapter callback."""

    metric_definition_id: CanonicalId
    metric_definition_version: CanonicalId
    kind: VisibilityMetricKindV2
    formula: VisibilityMetricFormulaV2
    area_measure: VisibilityAreaMeasureV2
    depth_policy: VisibilityDepthPolicyV2

    @model_validator(mode="after")
    def validate_formula_kind(self) -> Self:
        expected = {
            VisibilityMetricKindV2.VISIBLE_FRACTION: (
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_OVER_UNOCCLUDED_CLIPPED_PROJECTED_AREA,
            ),
            VisibilityMetricKindV2.IMAGE_AREA_FRACTION: (
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_AREA_OVER_IMAGE_AREA,
                VisibilityMetricFormulaV2.VISIBLE_CLIPPED_PROJECTED_BOUNDING_BOX_AREA_OVER_IMAGE_AREA,
            ),
            VisibilityMetricKindV2.TRUNCATED_FRACTION: (
                VisibilityMetricFormulaV2.ONE_MINUS_CLIPPED_OVER_UNCLIPPED_PROJECTED_AREA,
            ),
        }[self.kind]
        if self.formula not in expected:
            raise ValueError("visibility metric formula does not match its metric kind")
        return self

    @property
    def reference(self) -> tuple[str, str]:
        return self.metric_definition_id, self.metric_definition_version


class VisibilitySemanticsV2(V2Model):
    """Complete versioned formula registry consumed by the pure core."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="visibility-semantics")
    )
    semantics_id: CanonicalId
    definitions: tuple[VisibilityMetricDefinitionV2, ...]

    @model_validator(mode="after")
    def validate_complete_registry(self) -> Self:
        if self.schema_identity.schema_name != "visibility-semantics":
            raise ValueError("visibility semantics schema identity must be fixed")
        by_kind = {definition.kind: definition for definition in self.definitions}
        if len(by_kind) != len(self.definitions) or set(by_kind) != set(
            VisibilityMetricKindV2
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
            tuple(by_kind[kind] for kind in VisibilityMetricKindV2),
        )
        return self


class VisibilityConstraintV2(V2Model):
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
    mask_policy: VisibilityMaskPolicyV2
    occluder_soundness_policy: OccluderSoundnessPolicyV2
    minimum_visible_fraction: UnitInterval
    minimum_image_area_fraction: UnitInterval
    maximum_truncated_fraction: UnitInterval
    threshold_boundary_policy: BoundaryPolicyV2
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


class RelationDefinitionV2(V2Model):
    relation: RelationV2
    measurement: RelationMeasurementV2
    comparator: MeasurementComparatorV2
    threshold: FiniteFloat
    unit: MeasurementUnitV2
    representative_point: RelationRepresentativePointV2 | None
    operand_order: MeasurementOperandOrderV2
    boundary_policy: BoundaryPolicyV2
    tolerance: NonNegativeFiniteFloat
    tolerance_policy: RelationTolerancePolicyV2
    requires_both_visible: bool

    @model_validator(mode="after")
    def validate_measurement_contract(self) -> Self:
        expected_measurement = {
            RelationAxisV2.HORIZONTAL: RelationMeasurementV2.PROJECTED_CENTER_DELTA_X,
            RelationAxisV2.DEPTH: RelationMeasurementV2.CAMERA_DEPTH_DELTA,
            RelationAxisV2.DISTANCE: RelationMeasurementV2.SHAPE_GAP_XY,
        }[self.relation.axis]
        expected_unit = (
            MeasurementUnitV2.PIXEL
            if self.relation.axis is RelationAxisV2.HORIZONTAL
            else MeasurementUnitV2.METRE
        )
        expected_comparator = (
            MeasurementComparatorV2.LESS_THAN
            if self.relation in {RelationV2.LEFT, RelationV2.FRONT, RelationV2.NEAR}
            else MeasurementComparatorV2.GREATER_THAN
        )
        if self.measurement is not expected_measurement:
            raise ValueError("relation measurement does not match its axis")
        if self.unit is not expected_unit:
            raise ValueError("relation measurement unit does not match its axis")
        if self.comparator is not expected_comparator:
            raise ValueError("relation comparator does not match its direction")
        is_point_measurement = self.measurement in {
            RelationMeasurementV2.PROJECTED_CENTER_DELTA_X,
            RelationMeasurementV2.CAMERA_DEPTH_DELTA,
        }
        if is_point_measurement and self.representative_point is not (
            RelationRepresentativePointV2.RELATION_GEOMETRY_VOLUME_CENTROID
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


class RelationSemanticsV2(V2Model):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="relation-semantics")
    )
    semantics_id: CanonicalId
    definitions: tuple[RelationDefinitionV2, ...]

    @model_validator(mode="after")
    def validate_complete_relation_set(self) -> Self:
        if self.schema_identity.schema_name != "relation-semantics":
            raise ValueError("relation semantics schema identity must be fixed")
        by_relation = {
            definition.relation: definition for definition in self.definitions
        }
        if len(by_relation) != len(self.definitions) or set(by_relation) != set(
            RelationV2
        ):
            raise ValueError(
                "relation semantics require exactly one definition per relation"
            )
        for first, second in (
            (RelationV2.LEFT, RelationV2.RIGHT),
            (RelationV2.FRONT, RelationV2.BEHIND),
            (RelationV2.NEAR, RelationV2.FAR),
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
            tuple(by_relation[relation] for relation in RelationV2),
        )
        return self


class TargetRelationConstraintV2(V2Model):
    constraint_id: CanonicalId
    subject_id: CanonicalId
    reference_id: CanonicalId
    camera_id: CanonicalId
    relation_before: RelationV2
    relation_after: RelationV2
    semantics_id: CanonicalId

    @model_validator(mode="after")
    def validate_opposite_relation(self) -> Self:
        if self.subject_id == self.reference_id:
            raise ValueError("target subject and reference must differ")
        if self.relation_after is not self.relation_before.opposite:
            raise ValueError("target relation must change to its opposite on one axis")
        return self


class CanonicalConstraintSetV2(V2Model):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="canonical-constraint-set")
    )
    constraint_set_id: CanonicalId
    allowed_edit: AllowedEditV2
    position_domain: AllowedPositionDomainConstraintV2
    collision_constraints: tuple[CollisionConstraintV2, ...] = ()
    support_constraints: tuple[SupportConstraintV2, ...] = ()
    visibility_constraints: tuple[VisibilityConstraintV2, ...]
    target_relation: TargetRelationConstraintV2

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

        ``AllowedEditV2`` is a permission surface, not a predicate with a
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
