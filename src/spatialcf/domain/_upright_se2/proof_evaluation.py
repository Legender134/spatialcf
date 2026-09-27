"""Upright profile proof evaluation contracts and intrinsic operations."""

from __future__ import annotations

from typing import (
    ClassVar,
    Literal,
    Self,
)

from pydantic import (
    StrictBool,
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    CapabilityRef,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    HashBoundCanonicalModel,
    IntegerValue,
    TypedValue,
)

from spatialcf.domain.outcomes import (
    ResourceUsage,
)

from spatialcf.domain.profiles import (
    OwnerRef,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.cells import (
    UprightSE2CompiledCell,
    UprightSE2ProofLeafDisposition,
    UprightSE2RetainedOwnerOutcomeKind,
)

from spatialcf.domain._upright_se2.constants import (
    _UPRIGHT_SE2_ENUM_SCHEMA_REF,
    _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
    _UPRIGHT_SE2_INTEGER_SCHEMA_REF,
    _UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF,
    _UPRIGHT_SE2_RETAINED_POINT_TERM_ROSTER_SCHEMA_REF,
    _UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.solve_policy import (
    UprightSE2ExactRational,
)

from spatialcf.domain._upright_se2.values import (
    _materialized_state_record,
)

from spatialcf.domain._upright_se2.yaw import (
    _require_sorted_unique_by_bytes,
)


def _proof_usage_total(usages: tuple[ResourceUsage, ...]) -> ResourceUsage:
    """Derive one canonical shared-ledger total from non-reset stage usages."""

    if not usages:
        raise ValueError("proof resource aggregation requires at least one usage")
    accounting_refs = {usage.accounting_claim_definition_ref for usage in usages}
    if len(accounting_refs) != 1:
        raise ValueError("proof resource usages must use one accounting claim")
    totals: dict[str, float] = {}
    for usage in usages:
        for entry in usage.entries:
            totals[entry.resource_definition_ref] = (
                totals.get(entry.resource_definition_ref, 0.0) + entry.used
            )
    entry_values = tuple(
        {
            "resource_definition_ref": resource_definition_ref,
            "used": used,
        }
        for resource_definition_ref, used in sorted(
            totals.items(), key=lambda item: canonical_json_bytes(item[0])
        )
    )
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": next(iter(accounting_refs)),
            "entries": entry_values,
            "exhausted": any(usage.exhausted for usage in usages),
        }
    )


class UprightSE2RetainedOwnerEvaluation(HashBoundCanonicalModel):
    """One exact retained-owner outcome, bounds, findings, proof rows, and delta."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/retained-owner-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "retained_owner_evaluation_sha256"

    compiled_cell: UprightSE2CompiledCell
    owner_ref: OwnerRef
    evaluator_capability_ref: CapabilityRef
    outcome_kind: UprightSE2RetainedOwnerOutcomeKind
    exact_bounds: tuple[TypedValue, ...]
    finding_codes: tuple[CanonicalId, ...]
    proof_rows: tuple[CanonicalId, ...]
    resource_delta: ResourceUsage
    retained_owner_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_retained_owner_outcome(self) -> Self:
        _require_sorted_unique_by_bytes(self.exact_bounds, "proof exact bounds")
        _require_sorted_unique_by_bytes(self.finding_codes, "proof finding codes")
        _require_sorted_unique_by_bytes(self.proof_rows, "proof rows")
        if not self.proof_rows:
            raise ValueError("retained owner evaluation must retain proof rows")
        if any(type(bound.payload) is DigestValue for bound in self.exact_bounds):
            raise ValueError("proof exact bounds must not be digest-only")
        if self.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT:
            if not self.exact_bounds or self.finding_codes:
                raise ValueError(
                    "exact retained owner evaluations require bounds and no finding"
                )
            return self
        if self.exact_bounds or not self.finding_codes:
            raise ValueError(
                "nonexact retained owner evaluations require findings and no exact bounds"
            )
        expected_prefix = f"finding:spatialcf/upright-se2/proof-transport/{self.outcome_kind.value.lower()}"
        if any(not code.startswith(expected_prefix) for code in self.finding_codes):
            raise ValueError("retained owner finding must match its typed outcome")
        return self


class UprightSE2ProofCellEvaluation(HashBoundCanonicalModel):
    """Every retained-owner result for one exact cell and optional leaf status."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-cell-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_cell_evaluation_sha256"

    compiled_cell: UprightSE2CompiledCell
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...]
    leaf_disposition: UprightSE2ProofLeafDisposition | None
    complete_domain_empty: StrictBool | None
    proof_cell_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_leaf_disposition(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.owner_evaluations,
            "proof cell retained owner evaluations",
        )
        if not self.owner_evaluations:
            raise ValueError("proof cell must retain owner evaluations")
        if any(
            canonical_json_bytes(evaluation.compiled_cell)
            != canonical_json_bytes(self.compiled_cell)
            for evaluation in self.owner_evaluations
        ):
            raise ValueError("proof owner evaluation must bind its exact common cell")
        all_exact = all(
            evaluation.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT
            for evaluation in self.owner_evaluations
        )
        any_nonexact = not all_exact
        if self.leaf_disposition is None:
            if self.complete_domain_empty is not None:
                raise ValueError(
                    "internal proof cells must not claim complete-domain empty"
                )
            return self
        if self.complete_domain_empty is None:
            raise ValueError("proof leaf disposition requires an explicit empty claim")
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE:
            if not all_exact or self.complete_domain_empty:
                raise ValueError(
                    "inward-feasible proof cells require exact nonempty evidence"
                )
            return self
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE:
            if not all_exact or not self.complete_domain_empty:
                raise ValueError(
                    "complete-domain empty proof cells require exact outward-infeasible evidence"
                )
            return self
        if self.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED:
            if self.complete_domain_empty:
                raise ValueError(
                    "pruned proof cells cannot claim complete-domain empty"
                )
            return self
        if self.complete_domain_empty or not any_nonexact:
            raise ValueError(
                "unresolved proof cells require nonexact evidence and no complete-domain empty claim"
            )
        return self


class UprightSE2ProposalPointTerm(CanonicalModel):
    """One ordered exact retained semantic T/A/R/V/S point-term interval."""

    term_id: Literal["T", "A", "R", "V", "S"]
    lower: UprightSE2ExactRational
    upper: UprightSE2ExactRational

    @model_validator(mode="after")
    def _validate_ordered_interval(self) -> Self:
        if self.lower.as_fraction > self.upper.as_fraction:
            raise ValueError("point term interval bounds must be ordered")
        return self


class UprightSE2ProposalPointObjective(CanonicalModel):
    """The complete interval point objective copied from retained owner output."""

    terms: tuple[UprightSE2ProposalPointTerm, ...]
    total_lower: UprightSE2ExactRational
    total_upper: UprightSE2ExactRational

    @model_validator(mode="after")
    def _validate_complete_interval_point_objective(self) -> Self:
        if tuple(term.term_id for term in self.terms) != ("T", "A", "R", "V", "S"):
            raise ValueError("point objective must contain ordered T/A/R/V/S terms")
        if self.total_lower.as_fraction > self.total_upper.as_fraction:
            raise ValueError("point objective total interval bounds must be ordered")
        return self


def _retained_point_objective_value(
    objective: UprightSE2ProposalPointObjective,
) -> TypedValue:
    """Encode the exact owner output transported into a proposal point record."""

    def rational(value: UprightSE2ExactRational) -> TypedValue:
        return _materialized_state_record(
            _UPRIGHT_SE2_EXACT_RATIONAL_SCHEMA_REF,
            (
                (
                    "numerator",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                        payload=IntegerValue(value=value.numerator),
                    ),
                ),
                (
                    "denominator",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_INTEGER_SCHEMA_REF,
                        payload=IntegerValue(value=value.denominator),
                    ),
                ),
            ),
        )

    terms = tuple(
        _materialized_state_record(
            _UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF,
            (
                (
                    "term_id",
                    TypedValue(
                        value_schema_ref=_UPRIGHT_SE2_ENUM_SCHEMA_REF,
                        payload=EnumSymbolValue(symbol=term.term_id),
                    ),
                ),
                ("lower", rational(term.lower)),
                ("upper", rational(term.upper)),
            ),
        )
        for term in objective.terms
    )
    return _materialized_state_record(
        _UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF,
        (
            (
                "terms",
                TypedValue(
                    value_schema_ref=(
                        _UPRIGHT_SE2_RETAINED_POINT_TERM_ROSTER_SCHEMA_REF
                    ),
                    payload=FiniteOrderedTupleValue(
                        element_schema_ref=_UPRIGHT_SE2_RETAINED_POINT_TERM_SCHEMA_REF,
                        items=terms,
                    ),
                ),
            ),
            ("total_lower", rational(objective.total_lower)),
            ("total_upper", rational(objective.total_upper)),
        ),
    )


def _retained_point_owner_objective(
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...],
) -> TypedValue:
    """Require exactly one retained exact-bound payload for the point objective."""

    matches = tuple(
        bound
        for evaluation in owner_evaluations
        for bound in evaluation.exact_bounds
        if bound.value_schema_ref == _UPRIGHT_SE2_RETAINED_POINT_OBJECTIVE_SCHEMA_REF
    )
    if len(matches) != 1:
        raise ValueError(
            "proposal point evaluation requires one retained point owner objective"
        )
    return matches[0]


class UprightSE2ProposalPointEvaluation(HashBoundCanonicalModel):
    """Exact retained point-cell output transported without invoking a kernel."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proposal-point-evaluation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proposal_point_evaluation_sha256"

    point_cell_evaluation: UprightSE2ProofCellEvaluation
    point_objective: UprightSE2ProposalPointObjective
    proposal_point_evaluation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_exact_point_transport(self) -> Self:
        cell_evaluation = self.point_cell_evaluation
        cell = cell_evaluation.compiled_cell
        if (
            cell_evaluation.leaf_disposition is None
            and cell_evaluation.complete_domain_empty is not None
        ):
            raise ValueError("proposal point evaluation must retain an internal cell")
        if cell_evaluation.leaf_disposition is not None and (
            cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
            or cell_evaluation.complete_domain_empty is not False
        ):
            raise ValueError(
                "proposal point evaluation may reuse only an inward-feasible final cell"
            )
        if cell.x_lower != cell.x_upper or cell.y_lower != cell.y_upper:
            raise ValueError(
                "proposal point evaluation requires a degenerate point cell"
            )
        if any(
            evaluation.outcome_kind is not UprightSE2RetainedOwnerOutcomeKind.EXACT
            for evaluation in cell_evaluation.owner_evaluations
        ):
            raise ValueError(
                "proposal point evaluation requires exact retained owner evidence"
            )
        retained_objective = _retained_point_owner_objective(
            cell_evaluation.owner_evaluations
        )
        if canonical_json_bytes(retained_objective) != canonical_json_bytes(
            _retained_point_objective_value(self.point_objective)
        ):
            raise ValueError(
                "proposal point objective must byte-bind retained point owner output"
            )
        return self


class UprightSE2ProofPruneDecision(HashBoundCanonicalModel):
    """One inspectable deterministic prune decision for a retained cell row."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-prune-decision/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_prune_decision_sha256"

    cell_evaluation: UprightSE2ProofCellEvaluation
    prune_reason_codes: tuple[CanonicalId, ...]
    proof_prune_decision_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_prune_decision(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.prune_reason_codes,
            "proof prune reason codes",
        )
        if (
            self.cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.PRUNED
            or not self.prune_reason_codes
        ):
            raise ValueError("proof prune decision must retain one pruned cell reason")
        return self


class UprightSE2ProofFrontierRow(HashBoundCanonicalModel):
    """One final unresolved frontier row with the full nonexact cell evidence."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-frontier-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_frontier_row_sha256"

    cell_evaluation: UprightSE2ProofCellEvaluation
    frontier_reason_codes: tuple[CanonicalId, ...]
    proof_frontier_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_frontier_row(self) -> Self:
        _require_sorted_unique_by_bytes(
            self.frontier_reason_codes,
            "proof frontier reason codes",
        )
        if (
            self.cell_evaluation.leaf_disposition
            is not UprightSE2ProofLeafDisposition.UNRESOLVED
            or not self.frontier_reason_codes
        ):
            raise ValueError(
                "proof frontier row must retain one unresolved cell reason"
            )
        return self


class UprightSE2ProofStageDelta(HashBoundCanonicalModel):
    """One shared-ledger stage that retains its full owner rows and resource delta."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-stage-delta/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_stage_delta_sha256"

    stage_ref: CanonicalId
    owner_evaluations: tuple[UprightSE2RetainedOwnerEvaluation, ...]
    resource_delta: ResourceUsage
    proof_stage_delta_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_stage_delta(self) -> Self:
        if not self.stage_ref.startswith("stage:"):
            raise ValueError("proof stage references must use the stage namespace")
        _require_sorted_unique_by_bytes(
            self.owner_evaluations,
            "proof stage owner evaluations",
        )
        if not self.owner_evaluations:
            raise ValueError("proof stage must retain at least one owner evaluation")
        expected = _proof_usage_total(
            tuple(evaluation.resource_delta for evaluation in self.owner_evaluations)
        )
        if canonical_json_bytes(self.resource_delta) != canonical_json_bytes(expected):
            raise ValueError("proof stage resource delta does not match owner rows")
        return self


class UprightSE2ProofResourceLedger(HashBoundCanonicalModel):
    """The single canonical total resource ledger carried by the proof payload."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-resource-ledger/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_resource_ledger_sha256"

    stage_deltas: tuple[UprightSE2ProofStageDelta, ...]
    canonical_total_resource_usage: ResourceUsage
    proof_resource_ledger_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_shared_ledger(self) -> Self:
        _require_sorted_unique_by_bytes(self.stage_deltas, "proof stage deltas")
        if not self.stage_deltas:
            raise ValueError("proof resource ledger must retain stage deltas")
        if len({stage.stage_ref for stage in self.stage_deltas}) != len(
            self.stage_deltas
        ):
            raise ValueError("proof stage references must be unique")
        expected = _proof_usage_total(
            tuple(stage.resource_delta for stage in self.stage_deltas)
        )
        if canonical_json_bytes(
            self.canonical_total_resource_usage
        ) != canonical_json_bytes(expected):
            raise ValueError(
                "proof resource ledger canonical total does not match stages"
            )
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2RetainedOwnerEvaluation.model_rebuild()
UprightSE2ProofCellEvaluation.model_rebuild()
UprightSE2ProposalPointTerm.model_rebuild()
UprightSE2ProposalPointObjective.model_rebuild()
UprightSE2ProposalPointEvaluation.model_rebuild()
UprightSE2ProofPruneDecision.model_rebuild()
UprightSE2ProofFrontierRow.model_rebuild()
UprightSE2ProofStageDelta.model_rebuild()
UprightSE2ProofResourceLedger.model_rebuild()


# Keep supported public import and pickle lookup stable.
_proof_usage_total.__module__ = "spatialcf.domain.upright_se2"
UprightSE2RetainedOwnerEvaluation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofCellEvaluation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProposalPointTerm.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProposalPointObjective.__module__ = "spatialcf.domain.upright_se2"
_retained_point_objective_value.__module__ = "spatialcf.domain.upright_se2"
_retained_point_owner_objective.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProposalPointEvaluation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofPruneDecision.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofFrontierRow.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofStageDelta.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofResourceLedger.__module__ = "spatialcf.domain.upright_se2"
