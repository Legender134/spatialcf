"""Platform-neutral Canonical v2 minimum-total-cost objective contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    NonNegativeFiniteFloat,
    PositiveFiniteFloat,
    SchemaIdentityV2,
    V2Model,
)
from spatialcf.domain.v2.constraints import RelationAxisV2
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_OBJECTIVE_HASH_DOMAIN = "spatialcf.objective-spec.v2"


class ObjectiveModeV2(StrEnum):
    PRODUCTION = "PRODUCTION"
    NON_PRODUCTION_ABLATION = "NON_PRODUCTION_ABLATION"


class TieBreakKeyV2(StrEnum):
    TRANSLATION = "TRANSLATION"
    RELATION_DAMAGE = "RELATION_DAMAGE"
    VISIBILITY_CHANGE = "VISIBILITY_CHANGE"
    SAFETY_PENALTY = "SAFETY_PENALTY"
    DELTA_X = "DELTA_X"
    DELTA_Y = "DELTA_Y"


class SafetyAggregationKindV2(StrEnum):
    SUM_NORMALIZED_DEFICIT = "SUM_NORMALIZED_DEFICIT"


class SafetySlackUnitV2(StrEnum):
    METRE = "METRE"
    SQUARE_METRE = "SQUARE_METRE"
    PIXEL = "PIXEL"
    FRACTION = "FRACTION"
    DIMENSIONLESS = "DIMENSIONLESS"


class TranslationMetricV2(StrEnum):
    EUCLIDEAN_L2_WORLD_XY_METRE = "EUCLIDEAN_L2_WORLD_XY_METRE"


class RelationDamageMetricV2(StrEnum):
    PAIR_AXIS_SATISFIED_LABEL_SET_CHANGED_INDICATOR = (
        "PAIR_AXIS_SATISFIED_LABEL_SET_CHANGED_INDICATOR"
    )


class RelationDamageAggregationV2(StrEnum):
    WEIGHTED_SUM = "WEIGHTED_SUM"


class VisibilityChangeMetricV2(StrEnum):
    ABSOLUTE_NORMALIZED_METRIC_DELTA = "ABSOLUTE_NORMALIZED_METRIC_DELTA"


class VisibilityChangeAggregationV2(StrEnum):
    WEIGHTED_SUM = "WEIGHTED_SUM"


class SafetyConstraintAggregationV2(StrEnum):
    MIN_NORMALIZED_COMPONENT_MARGIN = "MIN_NORMALIZED_COMPONENT_MARGIN"


class SafetyComponentKindV2(StrEnum):
    POSITION_BOUNDARY_CLEARANCE = "POSITION_BOUNDARY_CLEARANCE"
    COLLISION_CLEARANCE = "COLLISION_CLEARANCE"
    SUPPORT_CONTACT_GAP_LOWER_MARGIN = "SUPPORT_CONTACT_GAP_LOWER_MARGIN"
    SUPPORT_CONTACT_GAP_UPPER_MARGIN = "SUPPORT_CONTACT_GAP_UPPER_MARGIN"
    SUPPORT_OVERLAP_AREA_MARGIN = "SUPPORT_OVERLAP_AREA_MARGIN"
    SUPPORT_STABILITY_INSET_MARGIN = "SUPPORT_STABILITY_INSET_MARGIN"
    VISIBILITY_VISIBLE_FRACTION_MARGIN = "VISIBILITY_VISIBLE_FRACTION_MARGIN"
    VISIBILITY_IMAGE_AREA_FRACTION_MARGIN = "VISIBILITY_IMAGE_AREA_FRACTION_MARGIN"
    VISIBILITY_TRUNCATED_FRACTION_MARGIN = "VISIBILITY_TRUNCATED_FRACTION_MARGIN"
    TARGET_RELATION_THRESHOLD_MARGIN = "TARGET_RELATION_THRESHOLD_MARGIN"


class PairAxisKeyV2(V2Model):
    """One canonical pair-axis key with a frozen measurement operand order.

    Damage is symmetric as a pair cost, but directional relation measurements
    still require an operand order.  Canonical ID order supplies that order and
    prevents adapters from reversing LEFT/RIGHT or FRONT/BEHIND semantics.
    """

    first_object_id: CanonicalId
    second_object_id: CanonicalId
    axis: RelationAxisV2

    @model_validator(mode="after")
    def canonicalize_unordered_pair(self) -> Self:
        if self.first_object_id == self.second_object_id:
            raise ValueError("pair-axis objects must be distinct")
        if self.second_object_id < self.first_object_id:
            first = self.second_object_id
            second = self.first_object_id
            object.__setattr__(self, "first_object_id", first)
            object.__setattr__(self, "second_object_id", second)
        return self

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return self.first_object_id, self.second_object_id, self.axis.value

    @property
    def measurement_operand_ids(self) -> tuple[str, str]:
        """Return the canonical ``first - second`` measurement operands."""

        return self.first_object_id, self.second_object_id


class ObjectCameraKeyV2(V2Model):
    """One normalized visibility metric, not an ambiguous native observation."""

    object_id: CanonicalId
    camera_id: CanonicalId
    metric_definition_id: CanonicalId
    metric_definition_version: CanonicalId

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.object_id,
            self.camera_id,
            self.metric_definition_id,
            self.metric_definition_version,
        )


class PairAxisWeightV2(V2Model):
    key: PairAxisKeyV2
    damage_weight: PositiveFiniteFloat


class ObjectCameraWeightV2(V2Model):
    key: ObjectCameraKeyV2
    change_weight: PositiveFiniteFloat


class SafetySlackComponentV2(V2Model):
    """One signed raw hard-constraint margin and its unit conversion.

    ``raw_margin / normalizer`` is dimensionless.  Component kinds freeze these
    raw formulas:

    * position: signed distance of candidate XY to the complement of the final
      permitted anchor locus (positive inside);
    * collision: the minimum body-pair Euclidean clearance minus that pair's
      effective required clearance;
    * support gap lower/upper: minimum selected-feature gap minus its lower
      bound, and its upper bound minus maximum selected-feature gap;
    * support overlap: measured union-intersection area minus its minimum;
    * support stability: minimum signed containment distance to the support
      surface boundary minus the required inset;
    * visibility: visible fraction minus its minimum, image-area fraction minus
      its minimum, and its maximum truncated fraction minus the measured value;
    * target relation: comparator-oriented distance from the active
      tolerance-adjusted threshold.
    """

    kind: SafetyComponentKindV2
    unit: SafetySlackUnitV2
    normalizer: PositiveFiniteFloat


class ConstraintSafetyTargetV2(V2Model):
    """Frozen scalar slack and deficit calculus for one hard constraint.

    The constraint slack is the minimum of all ``raw_margin / normalizer``
    components.  Its contribution is
    ``importance * max(0, target_slack - constraint_slack)``.  Both the
    constraint slack and ``target_slack`` are dimensionless.
    """

    constraint_id: CanonicalId
    target_slack: NonNegativeFiniteFloat
    component_aggregation: SafetyConstraintAggregationV2
    components: tuple[SafetySlackComponentV2, ...]
    importance: PositiveFiniteFloat

    @model_validator(mode="after")
    def canonicalize_components(self) -> Self:
        if not self.components:
            raise ValueError("safety target components must not be empty")
        kinds = tuple(component.kind for component in self.components)
        if len(kinds) != len(set(kinds)):
            raise ValueError("safety target component kinds must be unique")
        order = {kind: index for index, kind in enumerate(SafetyComponentKindV2)}
        object.__setattr__(
            self,
            "components",
            tuple(sorted(self.components, key=lambda item: order[item.kind])),
        )
        return self


class TranslationTermV2(V2Model):
    """Weighted Euclidean L2 displacement of the subject in world XY."""

    weight: PositiveFiniteFloat
    normalizer_m: PositiveFiniteFloat
    metric: TranslationMetricV2


class RelationDamageTermV2(V2Model):
    """Weighted sum of pair-axis relation label-set change indicators.

    A component is one iff the set of satisfied labels on that pair-axis
    differs between baseline and candidate, and zero otherwise.
    """

    weight: NonNegativeFiniteFloat
    normalizer: PositiveFiniteFloat
    metric: RelationDamageMetricV2
    aggregation: RelationDamageAggregationV2
    evaluation_camera_id: CanonicalId
    pair_axis_weights: tuple[PairAxisWeightV2, ...]

    @model_validator(mode="after")
    def canonicalize_universe(self) -> Self:
        if not self.pair_axis_weights:
            raise ValueError("pair-axis universe must not be empty")
        keys = tuple(item.key for item in self.pair_axis_weights)
        if len(keys) != len(set(keys)):
            raise ValueError("pair-axis universe must be unique")
        object.__setattr__(
            self,
            "pair_axis_weights",
            tuple(sorted(self.pair_axis_weights, key=lambda item: item.key.sort_key)),
        )
        return self


class VisibilityChangeTermV2(V2Model):
    """Weighted sum of absolute normalized visibility metric deltas."""

    weight: NonNegativeFiniteFloat
    normalizer: PositiveFiniteFloat
    metric: VisibilityChangeMetricV2
    aggregation: VisibilityChangeAggregationV2
    object_camera_weights: tuple[ObjectCameraWeightV2, ...]

    @model_validator(mode="after")
    def canonicalize_universe(self) -> Self:
        if not self.object_camera_weights:
            raise ValueError("object-camera universe must not be empty")
        keys = tuple(item.key for item in self.object_camera_weights)
        if len(keys) != len(set(keys)):
            raise ValueError("object-camera universe must be unique")
        object.__setattr__(
            self,
            "object_camera_weights",
            tuple(
                sorted(
                    self.object_camera_weights,
                    key=lambda item: item.key.sort_key,
                )
            ),
        )
        return self


class SafetyAggregationV2(V2Model):
    """Sum normalized hard-constraint target deficits in canonical ID order."""

    kind: SafetyAggregationKindV2
    targets: tuple[ConstraintSafetyTargetV2, ...]

    @model_validator(mode="after")
    def canonicalize_targets(self) -> Self:
        if not self.targets:
            raise ValueError("safety targets must not be empty")
        ids = tuple(item.constraint_id for item in self.targets)
        if len(ids) != len(set(ids)):
            raise ValueError("safety target constraint IDs must be unique")
        object.__setattr__(
            self,
            "targets",
            tuple(sorted(self.targets, key=lambda item: item.constraint_id)),
        )
        return self


class SafetyMarginTermV2(V2Model):
    weight: NonNegativeFiniteFloat
    normalizer: PositiveFiniteFloat
    aggregation: SafetyAggregationV2


_FIXED_TIE_BREAK = (
    TieBreakKeyV2.TRANSLATION,
    TieBreakKeyV2.RELATION_DAMAGE,
    TieBreakKeyV2.VISIBILITY_CHANGE,
    TieBreakKeyV2.SAFETY_PENALTY,
    TieBreakKeyV2.DELTA_X,
    TieBreakKeyV2.DELTA_Y,
)


class ObjectiveSpecV2(V2Model):
    """The one wire-complete minimum-total-cost formula.

    Each of translation, relation damage, visibility change, and summed safety
    deficit is divided by its term normalizer and multiplied by its term
    weight.  The four values are added, then the fixed tie-break is applied.
    All component metrics and aggregations are serialized closed enums, so an
    independent implementation can recompute the same ``J(edit)``.
    """

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="objective-spec")
    )
    objective_id: CanonicalId
    mode: ObjectiveModeV2
    translation: TranslationTermV2
    relation_damage: RelationDamageTermV2
    visibility_change: VisibilityChangeTermV2
    safety_margin: SafetyMarginTermV2
    tie_break: tuple[TieBreakKeyV2, ...] = _FIXED_TIE_BREAK

    @model_validator(mode="after")
    def validate_objective_contract(self) -> Self:
        if self.schema_identity.schema_name != "objective-spec":
            raise ValueError("objective schema identity must be fixed")
        if self.tie_break != _FIXED_TIE_BREAK:
            raise ValueError(
                "objective tie-break must use the frozen deterministic order"
            )
        if (
            self.mode is ObjectiveModeV2.PRODUCTION
            and self.relation_damage.weight <= 0.0
        ):
            raise ValueError("production objective requires positive relation weight")
        return self

    @property
    def production_eligible(self) -> bool:
        return self.mode is ObjectiveModeV2.PRODUCTION

    @property
    def objective_spec_sha256(self) -> str:
        """Digest under the one frozen domain for objective references."""

        return canonical_sha256_v2(self, domain=_OBJECTIVE_HASH_DOMAIN)
