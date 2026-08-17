"""Auditable Canonical v2 domain and objective partition artifacts.

These immutable contracts publish solver-produced regions, intervals, coverage
claims, and upstream semantic hashes. They do not perform candidate search or
certify that a declared cell union really covers its referenced hard domain;
the pure-core artifact verifier owns that topological proof.
"""

from __future__ import annotations

import math
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BeforeValidator, Field, model_validator

from spatialcf.domain.v2.base import (
    CanonicalId,
    FiniteFloat,
    NonNegativeFiniteFloat,
    SchemaIdentityV2,
    Sha256Digest,
    V2Model,
    Vec2V2,
)
from spatialcf.domain.v2.geometry import PlanarRegionV2
from spatialcf.domain.v2.objective import PairAxisKeyV2, SafetySlackUnitV2
from spatialcf.domain.v2.serialization import canonical_sha256_v2

_CANDIDATE_DOMAIN_HASH_DOMAIN = "candidate-domain-artifact-v2"
_RELATION_COST_PARTITION_HASH_DOMAIN = "relation-cost-partition-v2"
_OBJECTIVE_PARTITION_HASH_DOMAIN = "objective-partition-artifact-v2"


class RegionBoundStatusV2(StrEnum):
    NON_EMPTY = "NON_EMPTY"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"


class DomainCompletenessV2(StrEnum):
    EXACT = "EXACT"
    BRACKETED = "BRACKETED"
    INNER_ONLY = "INNER_ONLY"
    OUTER_ONLY = "OUTER_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class ArtifactCoverageV2(StrEnum):
    """Declared proof coverage, distinct from source-fact completeness."""

    EXACT_OUTER_COVERAGE = "EXACT_OUTER_COVERAGE"
    PARTIAL_OUTER_COVERAGE = "PARTIAL_OUTER_COVERAGE"
    INNER_WITNESS_ONLY = "INNER_WITNESS_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class CandidateDomainVariableKindV2(StrEnum):
    """The one edit variable represented by a candidate domain."""

    SUBJECT_WORLD_XY_TRANSLATION_M = "SUBJECT_WORLD_XY_TRANSLATION_M"


class CandidateConstraintKindV2(StrEnum):
    """Closed hard-constraint kinds understood by candidate compilation."""

    POSITION_DOMAIN = "POSITION_DOMAIN"
    SUPPORT = "SUPPORT"
    COLLISION = "COLLISION"
    TARGET_RELATION = "TARGET_RELATION"
    VISIBILITY = "VISIBILITY"


class ConstraintCompilationDispositionV2(StrEnum):
    """Whether a ledger step compiled a constraint or preserved the universe."""

    APPLIED = "APPLIED"
    UNKNOWN_AS_UNIVERSE = "UNKNOWN_AS_UNIVERSE"


class CandidateCompilationCoverageV2(StrEnum):
    """Coverage of the deterministic ordered hard-constraint sequence."""

    COMPLETE = "COMPLETE"
    EMPTY_OUTER_PREFIX = "EMPTY_OUTER_PREFIX"
    PARTIAL = "PARTIAL"


class PlanarRegionBoundV2(V2Model):
    """A non-empty region, a proven empty region, or an unavailable bound."""

    status: RegionBoundStatusV2
    region: PlanarRegionV2 | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status is RegionBoundStatusV2.NON_EMPTY:
            if self.region is None:
                raise ValueError("NON_EMPTY region bound requires a region")
        elif self.region is not None:
            raise ValueError(
                f"{self.status.value} region bound must not carry a region"
            )
        return self

    @classmethod
    def non_empty(cls, region: PlanarRegionV2) -> Self:
        return cls(status=RegionBoundStatusV2.NON_EMPTY, region=region)

    @classmethod
    def empty(cls) -> Self:
        return cls(status=RegionBoundStatusV2.EMPTY)

    @classmethod
    def unavailable(cls) -> Self:
        return cls(status=RegionBoundStatusV2.UNAVAILABLE)


class PlanarDomainBoundsV2(V2Model):
    """Sound inner/outer description of one planar domain."""

    inner_bound: PlanarRegionBoundV2
    outer_bound: PlanarRegionBoundV2
    completeness: DomainCompletenessV2
    coverage: ArtifactCoverageV2

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        inner_status = self.inner_bound.status
        outer_status = self.outer_bound.status
        has_inner = inner_status is not RegionBoundStatusV2.UNAVAILABLE
        has_outer = outer_status is not RegionBoundStatusV2.UNAVAILABLE

        if self.completeness is DomainCompletenessV2.UNAVAILABLE:
            if (
                has_inner
                or has_outer
                or self.coverage is not ArtifactCoverageV2.UNAVAILABLE
            ):
                raise ValueError(
                    "UNAVAILABLE domain requires unavailable bounds and coverage"
                )
            return self

        if self.completeness is DomainCompletenessV2.INNER_ONLY:
            if not has_inner or has_outer:
                raise ValueError(
                    "INNER_ONLY domain requires an inner bound and unavailable outer"
                )
            if self.coverage is not ArtifactCoverageV2.INNER_WITNESS_ONLY:
                raise ValueError(
                    "INNER_ONLY domain requires INNER_WITNESS_ONLY coverage"
                )
            return self

        if self.completeness is DomainCompletenessV2.OUTER_ONLY:
            if has_inner or not has_outer:
                raise ValueError(
                    "OUTER_ONLY domain requires unavailable inner and an outer bound"
                )
            self._require_outer_coverage()
            return self

        if not has_inner or not has_outer:
            raise ValueError(
                f"{self.completeness.value} domain requires both inner and outer bounds"
            )
        self._require_outer_coverage()

        if self.completeness is DomainCompletenessV2.EXACT:
            if self.inner_bound != self.outer_bound:
                raise ValueError(
                    "EXACT domain requires identical inner and outer bounds"
                )
            return self

        if not _bound_contains(self.outer_bound, self.inner_bound):
            raise ValueError("BRACKETED inner region must be contained in outer region")
        return self

    def _require_outer_coverage(self) -> None:
        if self.coverage not in {
            ArtifactCoverageV2.EXACT_OUTER_COVERAGE,
            ArtifactCoverageV2.PARTIAL_OUTER_COVERAGE,
        }:
            raise ValueError("domain with an outer bound requires outer coverage")

    @property
    def has_complete_outer_bound(self) -> bool:
        return (
            self.coverage is ArtifactCoverageV2.EXACT_OUTER_COVERAGE
            and self.outer_bound.status is not RegionBoundStatusV2.UNAVAILABLE
            and self.completeness
            in {
                DomainCompletenessV2.EXACT,
                DomainCompletenessV2.BRACKETED,
                DomainCompletenessV2.OUTER_ONLY,
            }
        )

    @property
    def is_partition_cell_verification_eligible(self) -> bool:
        return self.has_complete_outer_bound and self.completeness in {
            DomainCompletenessV2.EXACT,
            DomainCompletenessV2.BRACKETED,
        }

    @property
    def is_global_verification_eligible(self) -> bool:
        """Whether bounds meet schema preconditions for later global verification."""

        return (
            self.is_partition_cell_verification_eligible
            and self.inner_bound.status is RegionBoundStatusV2.NON_EMPTY
        )

    @property
    def is_unsat_verification_eligible(self) -> bool:
        """Whether an independent verifier may attempt an empty-outer proof."""

        return (
            self.has_complete_outer_bound
            and self.outer_bound.status is RegionBoundStatusV2.EMPTY
        )


NonNegativeIndex = Annotated[int, Field(strict=True, ge=0)]
NonNegativeResourceCount = Annotated[int, Field(strict=True, ge=0)]


def _reject_boolean_indicator(value: object) -> object:
    if type(value) is bool:
        raise ValueError("relation damage indicator must be a JSON number, not boolean")
    return value


BinaryIndicatorValueV2 = Annotated[
    Literal[0.0, 1.0],
    BeforeValidator(_reject_boolean_indicator),
]


class CandidateDomainVariableV2(V2Model):
    """Platform-neutral meaning and baseline of the two-dimensional variable."""

    variable_kind: CandidateDomainVariableKindV2 = (
        CandidateDomainVariableKindV2.SUBJECT_WORLD_XY_TRANSLATION_M
    )
    subject_id: CanonicalId
    baseline_anchor_world_xy_m: Vec2V2


class CompilationResourceUsageV2(V2Model):
    """Deterministic counters consumed while compiling a candidate domain."""

    domain_operations: NonNegativeResourceCount
    partition_cells: NonNegativeResourceCount
    refinement_steps: NonNegativeResourceCount


class ConstraintDomainShrinkStepV2(V2Model):
    """One ordered hard-constraint intersection ledger entry."""

    step_index: NonNegativeIndex
    constraint_id: CanonicalId
    constraint_kind: CandidateConstraintKindV2
    disposition: ConstraintCompilationDispositionV2
    constraint_domain: PlanarDomainBoundsV2 | None = None
    input_domain: PlanarDomainBoundsV2
    output_domain: PlanarDomainBoundsV2

    @model_validator(mode="after")
    def validate_compilation_step(self) -> Self:
        if self.disposition is ConstraintCompilationDispositionV2.APPLIED:
            if self.constraint_domain is None:
                raise ValueError("APPLIED step requires constraint_domain")
            _require_applied_domain_intersection(
                self.input_domain,
                self.constraint_domain,
                self.output_domain,
            )
            return self

        if self.constraint_domain is not None:
            raise ValueError(
                "UNKNOWN_AS_UNIVERSE step must not carry constraint_domain"
            )
        if self.output_domain.inner_bound.status is not RegionBoundStatusV2.EMPTY:
            raise ValueError("UNKNOWN_AS_UNIVERSE output requires an empty inner bound")
        if not self.input_domain.has_complete_outer_bound:
            raise ValueError(
                "UNKNOWN_AS_UNIVERSE requires a complete input outer bound"
            )
        if self.input_domain.outer_bound.status is RegionBoundStatusV2.UNAVAILABLE:
            raise ValueError("UNKNOWN_AS_UNIVERSE requires a known input outer bound")
        if self.output_domain.outer_bound != self.input_domain.outer_bound:
            raise ValueError(
                "UNKNOWN_AS_UNIVERSE output must preserve the input outer bound"
            )
        if not self.output_domain.has_complete_outer_bound:
            raise ValueError(
                "UNKNOWN_AS_UNIVERSE output requires complete outer coverage"
            )
        expected_completeness = (
            DomainCompletenessV2.EXACT
            if self.output_domain.outer_bound.status is RegionBoundStatusV2.EMPTY
            else DomainCompletenessV2.BRACKETED
        )
        if self.output_domain.completeness is not expected_completeness:
            raise ValueError(
                "UNKNOWN_AS_UNIVERSE output must use the conservative bound shape"
            )
        return self


class CandidateDomainArtifactV2(V2Model):
    """Published hard-domain bracket and deterministic shrink ledger."""

    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(
            schema_name="candidate-domain-artifact"
        )
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    candidate_variable: CandidateDomainVariableV2
    search_universe: PlanarDomainBoundsV2
    ordered_constraint_ids: tuple[CanonicalId, ...] = Field(min_length=1)
    compilation_coverage: CandidateCompilationCoverageV2
    compiled_prefix_length: NonNegativeIndex
    resource_usage: CompilationResourceUsageV2
    hard_domain: PlanarDomainBoundsV2
    shrink_ledger: tuple[ConstraintDomainShrinkStepV2, ...] = ()

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if self.schema_identity.schema_name != "candidate-domain-artifact":
            raise ValueError("candidate domain artifact schema identity must be fixed")
        if (
            self.search_universe.completeness is not DomainCompletenessV2.EXACT
            or not self.search_universe.has_complete_outer_bound
        ):
            raise ValueError(
                "search_universe must be exact with complete outer coverage"
            )
        if len(self.ordered_constraint_ids) != len(set(self.ordered_constraint_ids)):
            raise ValueError("ordered constraint IDs must be unique")
        ordered = tuple(sorted(self.shrink_ledger, key=lambda item: item.step_index))
        indices = tuple(item.step_index for item in ordered)
        if indices != tuple(range(len(ordered))):
            raise ValueError("shrink ledger step indexes must be contiguous from zero")
        constraint_ids = tuple(item.constraint_id for item in ordered)
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("shrink ledger constraint IDs must be unique")
        if self.compiled_prefix_length != len(ordered):
            raise ValueError(
                "compiled_prefix_length must equal the shrink ledger length"
            )
        if self.compiled_prefix_length > len(self.ordered_constraint_ids):
            raise ValueError(
                "compiled_prefix_length exceeds the ordered constraint sequence"
            )
        if constraint_ids != self.ordered_constraint_ids[: self.compiled_prefix_length]:
            raise ValueError(
                "shrink ledger IDs must equal the ordered constraint prefix"
            )
        if ordered and ordered[0].input_domain != self.search_universe:
            raise ValueError("shrink ledger first input must equal search_universe")
        if any(
            left.output_domain != right.input_domain
            for left, right in pairwise(ordered)
        ):
            raise ValueError("shrink ledger domain chain must be closed")
        expected_hard_domain = (
            ordered[-1].output_domain if ordered else self.search_universe
        )
        if expected_hard_domain != self.hard_domain:
            raise ValueError("shrink ledger final output must equal hard_domain")
        self._validate_compilation_coverage(ordered)
        object.__setattr__(self, "shrink_ledger", ordered)
        return self

    def _validate_compilation_coverage(
        self,
        ordered: tuple[ConstraintDomainShrinkStepV2, ...],
    ) -> None:
        all_applied = all(
            step.disposition is ConstraintCompilationDispositionV2.APPLIED
            for step in ordered
        )
        prefix_is_full = self.compiled_prefix_length == len(self.ordered_constraint_ids)
        if self.compilation_coverage is CandidateCompilationCoverageV2.COMPLETE:
            if not prefix_is_full or not all_applied:
                raise ValueError(
                    "COMPLETE compilation requires every ordered constraint APPLIED"
                )
            return
        if (
            self.compilation_coverage
            is CandidateCompilationCoverageV2.EMPTY_OUTER_PREFIX
        ):
            if not all_applied:
                raise ValueError("EMPTY_OUTER_PREFIX requires an APPLIED prefix")
            if self.hard_domain.outer_bound.status is not RegionBoundStatusV2.EMPTY:
                raise ValueError("EMPTY_OUTER_PREFIX requires an empty outer bound")
            return
        if prefix_is_full and all_applied:
            raise ValueError(
                "PARTIAL compilation requires an uncompiled or unknown constraint"
            )

    @property
    def is_global_verification_eligible(self) -> bool:
        """Return structural eligibility, never a global-optimum certification."""

        return (
            self.compilation_coverage is CandidateCompilationCoverageV2.COMPLETE
            and self.hard_domain.is_global_verification_eligible
            and self._ledger_is_structurally_complete
        )

    @property
    def is_unsat_verification_eligible(self) -> bool:
        """Return structural eligibility, never a proven-UNSAT certification."""

        return (
            self.compilation_coverage
            in {
                CandidateCompilationCoverageV2.COMPLETE,
                CandidateCompilationCoverageV2.EMPTY_OUTER_PREFIX,
            }
            and self.hard_domain.is_unsat_verification_eligible
            and self._ledger_is_structurally_complete
        )

    @property
    def _ledger_is_structurally_complete(self) -> bool:
        return all(
            step.disposition is ConstraintCompilationDispositionV2.APPLIED
            and domain.has_complete_outer_bound
            for step in self.shrink_ledger
            for domain in (
                step.input_domain,
                step.constraint_domain,
                step.output_domain,
            )
            if domain is not None
        )

    @property
    def candidate_domain_artifact_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(self, domain=_CANDIDATE_DOMAIN_HASH_DOMAIN)


class ConstraintSlackV2(V2Model):
    """A dimensionless interval after frozen per-component normalization."""

    constraint_id: CanonicalId
    lower_bound: FiniteFloat
    upper_bound: FiniteFloat
    unit: Literal[SafetySlackUnitV2.DIMENSIONLESS]

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.lower_bound > self.upper_bound:
            raise ValueError("constraint slack lower_bound must not exceed upper_bound")
        return self


class RelationDamageBoundV2(V2Model):
    """Sound interval for one binary relation-state change indicator."""

    key: PairAxisKeyV2
    lower_bound: BinaryIndicatorValueV2
    upper_bound: BinaryIndicatorValueV2

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.lower_bound > self.upper_bound:
            raise ValueError("relation damage lower_bound must not exceed upper_bound")
        return self


class RelationCostCellV2(V2Model):
    cell_id: CanonicalId
    domain: PlanarDomainBoundsV2
    relation_damage_vector: tuple[RelationDamageBoundV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_vector(self) -> Self:
        object.__setattr__(
            self,
            "relation_damage_vector",
            _canonical_damage_vector(self.relation_damage_vector),
        )
        return self


class RelationCostPartitionV2(V2Model):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(schema_name="relation-cost-partition")
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    candidate_domain_artifact_sha256: Sha256Digest
    objective_spec_sha256: Sha256Digest
    cells: tuple[RelationCostCellV2, ...]
    cell_outer_union_coverage: ArtifactCoverageV2

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if self.schema_identity.schema_name != "relation-cost-partition":
            raise ValueError("relation cost partition schema identity must be fixed")
        ordered = _canonical_cells(self.cells)
        _require_shared_damage_universe(ordered)
        _validate_union_coverage(ordered, self.cell_outer_union_coverage)
        object.__setattr__(self, "cells", ordered)
        return self

    @property
    def is_global_verification_eligible(self) -> bool:
        """Return structural eligibility; a verifier must prove the cell union."""

        return _partition_is_global_verification_eligible(
            self.cells,
            self.cell_outer_union_coverage,
        )

    @property
    def relation_cost_partition_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_RELATION_COST_PARTITION_HASH_DOMAIN,
        )


class NonNegativeIntervalV2(V2Model):
    lower_bound: NonNegativeFiniteFloat
    upper_bound: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.lower_bound > self.upper_bound:
            raise ValueError("interval lower_bound must not exceed upper_bound")
        return self


def _directed_binary64_sum(
    values: tuple[float, ...],
    *,
    upward: bool,
) -> float:
    exact = sum((Fraction.from_float(value) for value in values), Fraction())
    try:
        published = float(exact)
    except OverflowError as error:
        raise OverflowError("exact sum is outside finite binary64") from error
    if not math.isfinite(published):
        raise OverflowError("exact sum is outside finite binary64")

    published_exact = Fraction.from_float(published)
    if upward and published_exact < exact:
        published = math.nextafter(published, math.inf)
    elif not upward and published_exact > exact:
        published = math.nextafter(published, -math.inf)
    if not math.isfinite(published):
        raise OverflowError("directed sum is outside finite binary64")
    return published


class ObjectiveTermBoundsV2(V2Model):
    translation_loss: NonNegativeIntervalV2
    relation_damage_loss: NonNegativeIntervalV2
    visibility_change_loss: NonNegativeIntervalV2
    safety_margin_loss: NonNegativeIntervalV2

    @model_validator(mode="after")
    def validate_finite_total(self) -> Self:
        try:
            totals = (self.total_lower_bound, self.total_upper_bound)
        except OverflowError as error:
            raise ValueError("objective term-bound totals must be finite") from error
        if not all(math.isfinite(value) for value in totals):
            raise ValueError("objective term-bound totals must be finite")
        return self

    @property
    def total_lower_bound(self) -> float:
        return _directed_binary64_sum(
            (
                self.translation_loss.lower_bound,
                self.relation_damage_loss.lower_bound,
                self.visibility_change_loss.lower_bound,
                self.safety_margin_loss.lower_bound,
            ),
            upward=False,
        )

    @property
    def total_upper_bound(self) -> float:
        return _directed_binary64_sum(
            (
                self.translation_loss.upper_bound,
                self.relation_damage_loss.upper_bound,
                self.visibility_change_loss.upper_bound,
                self.safety_margin_loss.upper_bound,
            ),
            upward=True,
        )


class ObjectivePartitionCellV2(V2Model):
    cell_id: CanonicalId
    parent_relation_cell_id: CanonicalId
    domain: PlanarDomainBoundsV2
    relation_damage_vector: tuple[RelationDamageBoundV2, ...] = Field(min_length=1)
    term_loss_bounds: ObjectiveTermBoundsV2
    constraint_slacks: tuple[ConstraintSlackV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_vectors(self) -> Self:
        damage = _canonical_damage_vector(self.relation_damage_vector)
        slack_ids = tuple(item.constraint_id for item in self.constraint_slacks)
        if len(slack_ids) != len(set(slack_ids)):
            raise ValueError("objective cell constraint slacks must be unique")
        object.__setattr__(self, "relation_damage_vector", damage)
        object.__setattr__(
            self,
            "constraint_slacks",
            tuple(sorted(self.constraint_slacks, key=lambda item: item.constraint_id)),
        )
        return self


class ObjectivePartitionArtifactV2(V2Model):
    schema_identity: SchemaIdentityV2 = Field(
        default_factory=lambda: SchemaIdentityV2(
            schema_name="objective-partition-artifact"
        )
    )
    semantic_problem_sha256: Sha256Digest
    core_solver_config_sha256: Sha256Digest
    candidate_domain_artifact_sha256: Sha256Digest
    relation_cost_partition_sha256: Sha256Digest
    objective_spec_sha256: Sha256Digest
    cells: tuple[ObjectivePartitionCellV2, ...]
    cell_outer_union_coverage: ArtifactCoverageV2

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if self.schema_identity.schema_name != "objective-partition-artifact":
            raise ValueError(
                "objective partition artifact schema identity must be fixed"
            )
        ordered = _canonical_cells(self.cells)
        _require_shared_damage_universe(ordered)
        _require_shared_slack_universe(ordered)
        _validate_union_coverage(ordered, self.cell_outer_union_coverage)
        object.__setattr__(self, "cells", ordered)
        return self

    @property
    def is_global_verification_eligible(self) -> bool:
        """Return structural eligibility; a verifier must resolve all hash refs."""

        return _partition_is_global_verification_eligible(
            self.cells,
            self.cell_outer_union_coverage,
        )

    @property
    def objective_partition_artifact_sha256(self) -> Sha256Digest:
        return canonical_sha256_v2(
            self,
            domain=_OBJECTIVE_PARTITION_HASH_DOMAIN,
        )


def _canonical_damage_vector(
    vector: tuple[RelationDamageBoundV2, ...],
) -> tuple[RelationDamageBoundV2, ...]:
    keys = tuple(item.key for item in vector)
    if len(keys) != len(set(keys)):
        raise ValueError("relation damage vector keys must be unique")
    return tuple(sorted(vector, key=lambda item: item.key.sort_key))


CellT = TypeVar("CellT", RelationCostCellV2, ObjectivePartitionCellV2)


def _canonical_cells(
    cells: tuple[CellT, ...],
) -> tuple[CellT, ...]:
    ids = tuple(cell.cell_id for cell in cells)
    if len(ids) != len(set(ids)):
        raise ValueError("partition cell IDs must be unique")
    return tuple(sorted(cells, key=lambda item: item.cell_id))


def _require_shared_damage_universe(
    cells: tuple[RelationCostCellV2 | ObjectivePartitionCellV2, ...],
) -> None:
    universes = tuple(
        tuple(item.key for item in cell.relation_damage_vector) for cell in cells
    )
    if universes and any(universe != universes[0] for universe in universes[1:]):
        raise ValueError("partition cells must share one relation-damage universe")


def _require_shared_slack_universe(
    cells: tuple[ObjectivePartitionCellV2, ...],
) -> None:
    universes = tuple(
        tuple(item.constraint_id for item in cell.constraint_slacks) for cell in cells
    )
    if universes and any(universe != universes[0] for universe in universes[1:]):
        raise ValueError("objective cells must share one constraint-slack universe")


def _validate_union_coverage(
    cells: tuple[RelationCostCellV2 | ObjectivePartitionCellV2, ...],
    coverage: ArtifactCoverageV2,
) -> None:
    if coverage is ArtifactCoverageV2.EXACT_OUTER_COVERAGE and any(
        not cell.domain.has_complete_outer_bound for cell in cells
    ):
        raise ValueError("exact cell outer coverage requires every cell outer bound")
    if coverage is ArtifactCoverageV2.INNER_WITNESS_ONLY and any(
        cell.domain.completeness is not DomainCompletenessV2.INNER_ONLY
        for cell in cells
    ):
        raise ValueError("inner-witness coverage requires INNER_ONLY cells")
    if coverage is ArtifactCoverageV2.UNAVAILABLE and cells:
        raise ValueError("unavailable cell coverage must not carry cells")


def _partition_is_global_verification_eligible(
    cells: tuple[RelationCostCellV2 | ObjectivePartitionCellV2, ...],
    coverage: ArtifactCoverageV2,
) -> bool:
    return (
        coverage is ArtifactCoverageV2.EXACT_OUTER_COVERAGE
        and bool(cells)
        and all(cell.domain.is_partition_cell_verification_eligible for cell in cells)
        and any(
            cell.domain.inner_bound.status is RegionBoundStatusV2.NON_EMPTY
            for cell in cells
        )
    )


def _require_applied_domain_intersection(
    input_domain: PlanarDomainBoundsV2,
    constraint_domain: PlanarDomainBoundsV2,
    output_domain: PlanarDomainBoundsV2,
) -> None:
    """Require the representable bracket intersection for an APPLIED step.

    This is a structural ledger check, not proof that the referenced semantic
    constraint compiled to ``constraint_domain``.  The independent core
    verifier must resolve the problem hash and recompute that constraint.
    """

    domains = (input_domain, constraint_domain, output_domain)
    if any(
        domain.inner_bound.status is RegionBoundStatusV2.UNAVAILABLE
        or domain.outer_bound.status is RegionBoundStatusV2.UNAVAILABLE
        for domain in domains
    ):
        raise ValueError(
            "APPLIED intersection requires representable inner and outer bounds"
        )
    if any(not domain.has_complete_outer_bound for domain in domains):
        raise ValueError("APPLIED intersection requires complete outer coverage")
    for label, input_bound, constraint_bound, output_bound in (
        (
            "inner",
            input_domain.inner_bound,
            constraint_domain.inner_bound,
            output_domain.inner_bound,
        ),
        (
            "outer",
            input_domain.outer_bound,
            constraint_domain.outer_bound,
            output_domain.outer_bound,
        ),
    ):
        expected = _bound_geometry(input_bound).intersection(
            _bound_geometry(constraint_bound)
        )
        actual = _bound_geometry(output_bound)
        if not expected.equals(actual):
            raise ValueError(
                f"APPLIED output {label} bound must equal the bracket intersection"
            )


def _bound_contains(
    outer: PlanarRegionBoundV2,
    inner: PlanarRegionBoundV2,
) -> bool:
    if inner.status is RegionBoundStatusV2.EMPTY:
        return True
    if outer.status is RegionBoundStatusV2.EMPTY:
        return False
    if (
        inner.status is not RegionBoundStatusV2.NON_EMPTY
        or outer.status is not RegionBoundStatusV2.NON_EMPTY
        or inner.region is None
        or outer.region is None
    ):
        return False
    return _region_geometry(outer.region).covers(_region_geometry(inner.region))


def _bound_geometry(bound: PlanarRegionBoundV2):
    from shapely.geometry import Polygon

    if bound.status is RegionBoundStatusV2.UNAVAILABLE:
        raise ValueError("unavailable region bound has no geometry")
    if bound.status is RegionBoundStatusV2.EMPTY:
        return Polygon()
    if bound.region is None:  # pragma: no cover - defended by model validation
        raise ValueError("non-empty region bound requires geometry")
    return _region_geometry(bound.region)


def _region_geometry(region: PlanarRegionV2):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polygons = []
    for component in region.components:
        exterior = [(point.x, point.y) for point in component.exterior.vertices]
        holes = [
            [(point.x, point.y) for point in hole.vertices] for hole in component.holes
        ]
        polygons.append(Polygon(exterior, holes))
    return unary_union(polygons)
