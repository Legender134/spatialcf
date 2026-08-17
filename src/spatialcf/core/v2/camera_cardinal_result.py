"""Pull exact cardinal-camera delta artifacts back into semantic world XY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from spatialcf.core.v2._internal.resources.domain_operations import (
    DomainOperationBudgetV2,
)
from spatialcf.core.v2.camera_cardinal_rebase import _rotate_xy
from spatialcf.core.v2.camera_translation import (
    _candidate_variable_for_original_problem_v2_3,
)
from spatialcf.core.v2.minimum_cost_solver import CanonicalMinimumCostSolveOutcomeV2
from spatialcf.core.v2.minimum_cost_solver_v2_1 import _rebind_outcome
from spatialcf.domain.v2.artifacts import (
    CandidateDomainArtifactV2,
    ConstraintDomainShrinkStepV2,
    ObjectivePartitionArtifactV2,
    ObjectivePartitionCellV2,
    PlanarDomainBoundsV2,
    PlanarRegionBoundV2,
    RegionBoundStatusV2,
    RelationCostCellV2,
    RelationCostPartitionV2,
)
from spatialcf.domain.v2.base import Vec2V2
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.certificate import (
    GlobalOptimalityCertificateV2,
    ProvenUnsatCertificateV2,
)
from spatialcf.domain.v2.edit import CanonicalEditV2
from spatialcf.domain.v2.geometry import (
    PlanarPolygonComponentV2,
    PlanarRegionV2,
    PlanarRingV2,
)
from spatialcf.domain.v2.result import (
    CertifiedSuccessResultV2,
    CoreSolverConfigV2,
    ProvenUnsatResultV2,
    UncertifiedResultV2,
)


class CameraCardinalResultResourceLimitV2(RuntimeError):
    pass


@dataclass(slots=True)
class CameraCardinalResultDomainBudgetV2(DomainOperationBudgetV2):
    _exhaustion_error_type: ClassVar[type[RuntimeError]] = (
        CameraCardinalResultResourceLimitV2
    )


def pullback_camera_cardinal_outcome_v2_4(
    replay: CanonicalMinimumCostSolveOutcomeV2,
    original_problem: SemanticProblemV2_1,
    original_config: CoreSolverConfigV2,
    *,
    preprocessing_domain_operations: int,
    original_to_internal_quarter_turns_ccw: int,
    postprocessing_budget: CameraCardinalResultDomainBudgetV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Pull one fresh normalized replay back and rebuild every hash reference."""

    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("camera-cardinal replay has the wrong exact type")
    if type(original_problem) is not SemanticProblemV2_1:
        raise TypeError("camera-cardinal original problem has the wrong exact type")
    if type(original_config) is not CoreSolverConfigV2:
        raise TypeError("camera-cardinal original config has the wrong exact type")
    if type(postprocessing_budget) is not CameraCardinalResultDomainBudgetV2:
        raise TypeError("postprocessing budget has the wrong exact type")
    if (
        type(preprocessing_domain_operations) is not int
        or preprocessing_domain_operations < 0
    ):
        raise TypeError("preprocessing domain operations must be non-negative")
    _inverse_quarter_turns(original_to_internal_quarter_turns_ccw)
    usage = replay.cumulative_generation_usage
    if replay.result is None or usage is None:
        raise ValueError("camera-cardinal replay omitted its result or usage")
    expected_start = preprocessing_domain_operations + usage.domain_operations
    if postprocessing_budget.used != expected_start:
        raise ValueError("camera-cardinal postprocessing ledger did not resume replay")

    start = postprocessing_budget.used
    candidate_post = reserve_camera_cardinal_outcome_structure_v2_4(
        replay,
        postprocessing_budget,
    )
    total_post = postprocessing_budget.used - start
    mapped_replay = _map_replay_outcome(
        replay,
        original_to_internal_quarter_turns_ccw,
    )
    candidate_increment = preprocessing_domain_operations + candidate_post
    if type(replay.result) is ProvenUnsatResultV2:
        candidate_increment = preprocessing_domain_operations + total_post
    return _rebind_outcome(
        mapped_replay,
        original_problem,
        original_config,
        candidate_increment,
        candidate_variable_override=(
            _candidate_variable_for_original_problem_v2_3(original_problem)
        ),
        cumulative_domain_operations_increment=(
            preprocessing_domain_operations + total_post
        ),
    )


def reserve_camera_cardinal_outcome_structure_v2_4(
    replay: CanonicalMinimumCostSolveOutcomeV2,
    budget: CameraCardinalResultDomainBudgetV2,
) -> int:
    """Reserve a complete pullback pass before reading any domain coordinate."""

    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("camera-cardinal replay has the wrong exact type")
    result = replay.result
    if result is None:
        raise ValueError("camera-cardinal replay omitted its result")
    budget.consume()
    candidate = getattr(result, "candidate_domain", None)
    candidate_start = budget.used
    if candidate is not None:
        if type(candidate) is not CandidateDomainArtifactV2:
            raise TypeError("replay candidate has the wrong exact type")
        budget.consume(2 + len(candidate.shrink_ledger))
        _reserve_domain(candidate.search_universe, budget)
        _reserve_domain(candidate.hard_domain, budget)
        for step in candidate.shrink_ledger:
            _reserve_domain(step.input_domain, budget)
            if step.constraint_domain is not None:
                _reserve_domain(step.constraint_domain, budget)
            _reserve_domain(step.output_domain, budget)
    candidate_post = budget.used - candidate_start

    relation = getattr(result, "relation_cost_partition", None)
    if relation is not None:
        if type(relation) is not RelationCostPartitionV2:
            raise TypeError("replay relation partition has the wrong exact type")
        budget.consume(1 + len(relation.cells))
        for cell in relation.cells:
            _reserve_domain(cell.domain, budget)

    objective = getattr(result, "objective_partition", None)
    if objective is not None:
        if type(objective) is not ObjectivePartitionArtifactV2:
            raise TypeError("replay objective partition has the wrong exact type")
        budget.consume(1 + len(objective.cells))
        for cell in objective.cells:
            _reserve_domain(cell.domain, budget)

    if type(result) is CertifiedSuccessResultV2:
        budget.consume(3)
    elif type(result) is ProvenUnsatResultV2:
        budget.consume(2)
    elif type(result) is UncertifiedResultV2:
        budget.consume()
    else:
        raise TypeError("camera-cardinal replay returned an invalid result type")
    budget.consume()
    return candidate_post


def _reserve_domain(
    domain: PlanarDomainBoundsV2,
    budget: CameraCardinalResultDomainBudgetV2,
) -> None:
    if type(domain) is not PlanarDomainBoundsV2:
        raise TypeError("delta domain has the wrong exact type")
    budget.consume()
    for bound in (domain.inner_bound, domain.outer_bound):
        budget.consume()
        if bound.status is not RegionBoundStatusV2.NON_EMPTY:
            continue
        region = bound.region
        if region is None:  # pragma: no cover - model invariant
            raise RuntimeError("NON_EMPTY region bound omitted its region")
        budget.consume(1 + len(region.components))
        for component in region.components:
            budget.consume(1 + len(component.exterior.vertices))
            budget.consume(len(component.holes))
            for hole in component.holes:
                budget.consume(len(hole.vertices))


def _pullback_planar_domain_v2_4(
    domain: PlanarDomainBoundsV2,
    *,
    original_to_internal_quarter_turns_ccw: int,
) -> PlanarDomainBoundsV2:
    if type(domain) is not PlanarDomainBoundsV2:
        raise TypeError("delta domain has the wrong exact type")
    inverse_quarter_turns = _inverse_quarter_turns(
        original_to_internal_quarter_turns_ccw
    )
    return PlanarDomainBoundsV2(
        inner_bound=_pullback_region_bound(domain.inner_bound, inverse_quarter_turns),
        outer_bound=_pullback_region_bound(domain.outer_bound, inverse_quarter_turns),
        completeness=domain.completeness,
        coverage=domain.coverage,
    )


def _map_replay_outcome(
    replay: CanonicalMinimumCostSolveOutcomeV2,
    original_to_internal_quarter_turns_ccw: int,
) -> CanonicalMinimumCostSolveOutcomeV2:
    old_result = replay.result
    if old_result is None:  # pragma: no cover - caller gate
        raise RuntimeError("cannot map a missing replay result")
    old_candidate = getattr(old_result, "candidate_domain", None)
    candidate = (
        None
        if old_candidate is None
        else _pullback_candidate(
            old_candidate,
            original_to_internal_quarter_turns_ccw,
        )
    )
    old_relation = getattr(old_result, "relation_cost_partition", None)
    relation = None
    if old_relation is not None:
        if candidate is None:
            raise RuntimeError("relation replay omitted its candidate")
        relation = _pullback_relation(
            old_relation,
            candidate,
            original_to_internal_quarter_turns_ccw,
        )
    old_objective = getattr(old_result, "objective_partition", None)
    objective = None
    if old_objective is not None:
        if candidate is None or relation is None:
            raise RuntimeError("objective replay omitted its upstream artifacts")
        objective = _pullback_objective(
            old_objective,
            candidate,
            relation,
            original_to_internal_quarter_turns_ccw,
        )

    if type(old_result) is CertifiedSuccessResultV2:
        if candidate is None or relation is None or objective is None:
            raise RuntimeError("success replay omitted its artifact chain")
        edit = _pullback_edit(
            old_result.edit,
            original_to_internal_quarter_turns_ccw,
        )
        certificate = GlobalOptimalityCertificateV2.model_validate(
            old_result.certificate.model_copy(
                update={
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    ),
                    "relation_cost_partition_sha256": (
                        relation.relation_cost_partition_sha256
                    ),
                    "objective_partition_artifact_sha256": (
                        objective.objective_partition_artifact_sha256
                    ),
                    "edit_sha256": edit.edit_sha256,
                }
            ).model_dump(mode="python", warnings="error"),
            strict=True,
        )
        mapped_result = CertifiedSuccessResultV2(
            semantic_problem_sha256=old_result.semantic_problem_sha256,
            core_solver_config=old_result.core_solver_config,
            candidate_domain=candidate,
            relation_cost_partition=relation,
            objective_partition=objective,
            edit=edit,
            global_loss_lower_bound=old_result.global_loss_lower_bound,
            witness_loss_bounds=old_result.witness_loss_bounds,
            certificate=certificate,
        )
    elif type(old_result) is ProvenUnsatResultV2:
        if candidate is None:
            raise RuntimeError("UNSAT replay omitted its candidate")
        certificate = ProvenUnsatCertificateV2.model_validate(
            old_result.certificate.model_copy(
                update={
                    "candidate_domain_artifact_sha256": (
                        candidate.candidate_domain_artifact_sha256
                    )
                }
            ).model_dump(mode="python", warnings="error"),
            strict=True,
        )
        mapped_result = ProvenUnsatResultV2(
            semantic_problem_sha256=old_result.semantic_problem_sha256,
            core_solver_config=old_result.core_solver_config,
            candidate_domain=candidate,
            certificate=certificate,
        )
    elif type(old_result) is UncertifiedResultV2:
        mapped_result = UncertifiedResultV2(
            semantic_problem_sha256=old_result.semantic_problem_sha256,
            core_solver_config=old_result.core_solver_config,
            uncertified_reason=old_result.uncertified_reason,
            candidate_domain=candidate,
            relation_cost_partition=relation,
            objective_partition=objective,
        )
    else:  # pragma: no cover - caller reservation gate
        raise TypeError("camera-cardinal replay returned an invalid result type")
    return CanonicalMinimumCostSolveOutcomeV2(
        result=mapped_result,
        finding_codes=replay.finding_codes,
        cumulative_generation_usage=replay.cumulative_generation_usage,
        proposal_count=replay.proposal_count,
        evaluated_proposal_count=replay.evaluated_proposal_count,
    )


def _pullback_candidate(
    candidate: CandidateDomainArtifactV2,
    quarter_turns: int,
) -> CandidateDomainArtifactV2:
    steps = tuple(
        ConstraintDomainShrinkStepV2(
            step_index=step.step_index,
            constraint_id=step.constraint_id,
            constraint_kind=step.constraint_kind,
            disposition=step.disposition,
            constraint_domain=(
                None
                if step.constraint_domain is None
                else _pullback_planar_domain_v2_4(
                    step.constraint_domain,
                    original_to_internal_quarter_turns_ccw=quarter_turns,
                )
            ),
            input_domain=_pullback_planar_domain_v2_4(
                step.input_domain,
                original_to_internal_quarter_turns_ccw=quarter_turns,
            ),
            output_domain=_pullback_planar_domain_v2_4(
                step.output_domain,
                original_to_internal_quarter_turns_ccw=quarter_turns,
            ),
        )
        for step in candidate.shrink_ledger
    )
    return CandidateDomainArtifactV2.model_validate(
        candidate.model_copy(
            update={
                "search_universe": _pullback_planar_domain_v2_4(
                    candidate.search_universe,
                    original_to_internal_quarter_turns_ccw=quarter_turns,
                ),
                "hard_domain": _pullback_planar_domain_v2_4(
                    candidate.hard_domain,
                    original_to_internal_quarter_turns_ccw=quarter_turns,
                ),
                "shrink_ledger": steps,
            }
        ).model_dump(mode="python", warnings="error"),
        strict=True,
    )


def _pullback_relation(
    relation: RelationCostPartitionV2,
    candidate: CandidateDomainArtifactV2,
    quarter_turns: int,
) -> RelationCostPartitionV2:
    cells = tuple(
        RelationCostCellV2(
            cell_id=cell.cell_id,
            domain=_pullback_planar_domain_v2_4(
                cell.domain,
                original_to_internal_quarter_turns_ccw=quarter_turns,
            ),
            relation_damage_vector=cell.relation_damage_vector,
        )
        for cell in relation.cells
    )
    return RelationCostPartitionV2.model_validate(
        relation.model_copy(
            update={
                "candidate_domain_artifact_sha256": (
                    candidate.candidate_domain_artifact_sha256
                ),
                "cells": cells,
            }
        ).model_dump(mode="python", warnings="error"),
        strict=True,
    )


def _pullback_objective(
    objective: ObjectivePartitionArtifactV2,
    candidate: CandidateDomainArtifactV2,
    relation: RelationCostPartitionV2,
    quarter_turns: int,
) -> ObjectivePartitionArtifactV2:
    cells = tuple(
        ObjectivePartitionCellV2(
            cell_id=cell.cell_id,
            parent_relation_cell_id=cell.parent_relation_cell_id,
            domain=_pullback_planar_domain_v2_4(
                cell.domain,
                original_to_internal_quarter_turns_ccw=quarter_turns,
            ),
            relation_damage_vector=cell.relation_damage_vector,
            term_loss_bounds=cell.term_loss_bounds,
            constraint_slacks=cell.constraint_slacks,
        )
        for cell in objective.cells
    )
    return ObjectivePartitionArtifactV2.model_validate(
        objective.model_copy(
            update={
                "candidate_domain_artifact_sha256": (
                    candidate.candidate_domain_artifact_sha256
                ),
                "relation_cost_partition_sha256": (
                    relation.relation_cost_partition_sha256
                ),
                "cells": cells,
            }
        ).model_dump(mode="python", warnings="error"),
        strict=True,
    )


def _pullback_edit(edit: CanonicalEditV2, quarter_turns: int) -> CanonicalEditV2:
    inverse = _inverse_quarter_turns(quarter_turns)
    translation = edit.translation_xy_m
    x, y = _rotate_xy(translation.x, translation.y, inverse)
    return CanonicalEditV2(
        semantic_problem_sha256=edit.semantic_problem_sha256,
        subject_id=edit.subject_id,
        translation_xy_m=Vec2V2(x=x, y=y),
    )


def _pullback_region_bound(
    bound: PlanarRegionBoundV2,
    inverse_quarter_turns: int,
) -> PlanarRegionBoundV2:
    if type(bound) is not PlanarRegionBoundV2:
        raise TypeError("region bound has the wrong exact type")
    if bound.status is RegionBoundStatusV2.EMPTY:
        return PlanarRegionBoundV2.empty()
    if bound.status is RegionBoundStatusV2.UNAVAILABLE:
        return PlanarRegionBoundV2.unavailable()
    region = bound.region
    if region is None:  # pragma: no cover - model invariant
        raise RuntimeError("NON_EMPTY region bound omitted its region")
    return PlanarRegionBoundV2.non_empty(
        _pullback_region(region, inverse_quarter_turns)
    )


def _pullback_region(
    region: PlanarRegionV2,
    inverse_quarter_turns: int,
) -> PlanarRegionV2:
    if type(region) is not PlanarRegionV2:
        raise TypeError("planar region has the wrong exact type")
    return PlanarRegionV2(
        components=tuple(
            PlanarPolygonComponentV2(
                exterior=_pullback_ring(
                    component.exterior,
                    inverse_quarter_turns,
                ),
                holes=tuple(
                    _pullback_ring(hole, inverse_quarter_turns)
                    for hole in component.holes
                ),
            )
            for component in region.components
        )
    )


def _pullback_ring(
    ring: PlanarRingV2,
    inverse_quarter_turns: int,
) -> PlanarRingV2:
    if type(ring) is not PlanarRingV2:
        raise TypeError("planar ring has the wrong exact type")
    return PlanarRingV2(
        winding=ring.winding,
        vertices=tuple(
            Vec2V2(x=x, y=y)
            for x, y in (
                _rotate_xy(point.x, point.y, inverse_quarter_turns)
                for point in ring.vertices
            )
        ),
    )


def _inverse_quarter_turns(quarter_turns: int) -> int:
    if type(quarter_turns) is not int or not 0 <= quarter_turns <= 3:
        raise TypeError("quarter_turns must be an exact int in 0..3")
    return (-quarter_turns) % 4


__all__ = (
    "CameraCardinalResultDomainBudgetV2",
    "CameraCardinalResultResourceLimitV2",
    "pullback_camera_cardinal_outcome_v2_4",
    "reserve_camera_cardinal_outcome_structure_v2_4",
)
