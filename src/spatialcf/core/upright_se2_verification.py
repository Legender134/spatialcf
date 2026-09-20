"""Fresh, checker-only replay for registered M3 Upright SE(2) submissions."""

from __future__ import annotations

from fractions import Fraction

from spatialcf.core._internal.kernels.projected_visibility import (
    evaluate_continuous_yaw_visibility_v4,
    evaluate_fixed_cardinal_visibility_v3,
)
from spatialcf.core._internal.kernels.so2 import (
    CardinalKernelKindV3,
    ContinuousYawIntervalKindV4,
    SO2AtomicBudgetV2,
    compile_continuous_yaw_lift_v4,
    compile_lifted_turn_sin_cos_bounds_v4,
)
from spatialcf.core._internal.kernels.upright_box import (
    ClosedXYCellV3,
    ContinuousYawCellPolicyV4,
    compile_continuous_yaw_box_bounds_v4,
    evaluate_continuous_yaw_cell_v4,
    evaluate_fixed_cardinal_cell_v3,
)
from spatialcf.core.upright_se2_compiler import (
    build_upright_se2_cardinal_evaluation_inputs,
    build_upright_se2_continuous_evaluation_inputs,
    build_upright_se2_retained_owner_evaluation,
    compile_upright_se2,
    materialize_upright_se2_continuous_endpoint,
    materialize_upright_se2_endpoint,
)
from spatialcf.domain import upright_se2 as upright
from spatialcf.domain.base import Vec2
from spatialcf.domain.counterfactual import CounterfactualSolveRequest
from spatialcf.domain.definitions import TypedValue
from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendSubmission,
    BackendUnknownEvidence,
    CheckedProofOutcome,
    CheckerDisposition,
    ProofMaterialEnvelope,
    ResourceUsage,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = ("verify_upright_se2_submission",)

_CHECKED_FACT_SCHEMA_REF = "schema:spatialcf/upright-se2/checked-replay-fact/1.0"
_CONTINUOUS_BACKEND_REF = "backend:spatialcf/upright-se2/continuous"
_CONTINUOUS_BACKEND_BUILD_SHA256 = "b" * 64
_MAX_CONTINUOUS_EXACT_DYADIC_REFINEMENT_DEPTH = 3
_UNKNOWN_NUMERIC_REASON_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/backend-numeric-gap/1.0"
)
_UNKNOWN_UNSUPPORTED_REASON_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/backend-unsupported-semantic-input/1.0"
)
_UNKNOWN_INCOMPLETE_REASON_CLAIM_DEFINITION_REF = (
    "definition:spatialcf/upright-se2/backend-incomplete-cardinal-coverage/1.0"
)


def verify_upright_se2_submission(
    *,
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    submission: BackendSubmission,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
) -> CheckedProofOutcome:
    """Replay a cardinal submission from roots and return no terminal record.

    The caller supplies the frozen routing record explicitly: a V2 submission
    intentionally carries only its digest and must never cause checker-side
    routing reconstruction.  Every replay input is rebuilt from the source
    request through the compiler bridge; proposal payloads are only compared
    after that independent replay.
    """

    _require_exact(solve_request, CounterfactualSolveRequest, "solve request")
    _require_exact(selection, BackendSelectionRecord, "selection")
    if type(compilation) is upright.UprightSE2ContinuousCompilation:
        return _verify_continuous_submission(
            solve_request=solve_request,
            selection=selection,
            compilation=compilation,
            submission=submission,
            checker_policy=checker_policy,
        )

    _require_exact(compilation, upright.UprightSE2Compilation, "compilation")
    _require_submission(submission)
    _require_exact(
        checker_policy, upright.UprightSE2CheckerReplayPolicy, "checker policy"
    )
    if (
        checker_policy.proof_policy_sha256
        != solve_request.proof_policy.proof_policy_sha256
    ):
        raise ValueError("checker policy does not match solve request")
    solve_policy = upright.decode_upright_se2_solve_policy_definition_payload(
        solve_request.solve_policy_definition_bundle
    )
    if (
        checker_policy.solve_policy_definition_bundle_sha256
        != solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        or checker_policy.requested_gap != solve_policy.requested_gap
        or checker_policy.objective_bound_policy_ref
        != solve_policy.objective_bound_policy_ref
        or checker_policy.exact_global_claim_definition_ref
        != solve_policy.exact_global_claim_definition_ref
        or checker_policy.finite_gap_claim_definition_ref
        != solve_policy.finite_gap_claim_definition_ref
    ):
        raise ValueError("checker policy does not replay solve-policy gap and claims")
    if (
        checker_policy.checker_capability_ref
        != upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
    ):
        raise ValueError("checker policy does not require the cardinal checker")
    if selection.selection_disposition != "SELECTED":
        raise ValueError("selection must be selected for a submitted proof")
    if (
        selection.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or selection.solve_request_sha256 != solve_request.solve_request_sha256
    ):
        raise ValueError("selection roots do not match solve request")

    replayed = compile_upright_se2(solve_request)
    if type(replayed) is not upright.UprightSE2Compilation or canonical_json_bytes(
        replayed
    ) != canonical_json_bytes(compilation):
        raise ValueError("fresh compilation replay does not match submission")

    evidence, envelope, evidence_claim_ref = _submission_evidence(submission)
    if (
        evidence.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or evidence.solve_request_sha256 != solve_request.solve_request_sha256
        or evidence.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or envelope.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
    ):
        raise ValueError("submission selection roots do not match")
    if (
        evidence.proposal_backend_owner_ref == upright.UPRIGHT_SE2_CHECKER_OWNER_REF
        or evidence.proposal_backend_capability_ref
        == upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
    ):
        raise ValueError("proposal owner or capability cannot serve as checker")
    if (
        type(submission) is BackendProposalSubmission
        and evidence_claim_ref != upright.UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF
    ):
        raise ValueError("proposal claim does not bind registered untrusted evidence")
    selected_descriptors = tuple(
        descriptor
        for descriptor in solve_request.backend_descriptor_bundle.backend_descriptors
        if descriptor.backend_ref == selection.selected_backend_ref
    )
    if len(selected_descriptors) != 1:
        raise ValueError("selected backend descriptor does not close")
    if (
        evidence.proposal_backend_ref != selection.selected_backend_ref
        or evidence.proposal_backend_owner_ref != upright.UPRIGHT_SE2_BACKEND_OWNER_REF
        or evidence.proposal_backend_capability_ref
        != upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF
        or evidence.proposal_backend_build_sha256
        != selected_descriptors[0].implementation_build_sha256
    ):
        raise ValueError("proposal backend identity does not bind selected backend")

    proof = _decode_proof(envelope)
    if canonical_json_bytes(proof.compilation) != canonical_json_bytes(compilation):
        raise ValueError("proof compilation does not match supplied compilation")
    _check_cardinal_roster_and_coverage(compilation, proof)
    if canonical_json_bytes(proof.total_resource_usage) != canonical_json_bytes(
        evidence.resource_usage
    ):
        raise ValueError("submission resource usage does not match proof ledger")
    fresh_feasible_leaf_objectives = _replay_cells(compilation, proof)
    fresh_candidate_objectives = _replay_candidates(compilation, proof)
    _check_selected_proposal_bounds(submission, proof, fresh_candidate_objectives)
    _check_terminal_evidence(submission, proof, compilation)
    _check_selected_unknown_reason(
        solve_request=solve_request,
        submission=submission,
        proof=proof,
    )
    checked_claim_ref = _submission_checked_claim(
        submission,
        proof,
        checker_policy,
        evidence_claim_ref,
        fresh_feasible_leaf_objectives,
        fresh_candidate_objectives,
    )
    return CheckedProofOutcome.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proof_material_sha256=envelope.proof_material_sha256,
        checker_capability_ref=upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
        checker_build_sha256=upright.UPRIGHT_SE2_CHECKER_BUILD_SHA256,
        checker_disposition=(
            CheckerDisposition.ACCEPTED
            if checked_claim_ref is not None
            else CheckerDisposition.LIMITED
        ),
        checked_claim_definition_ref=(
            evidence_claim_ref if checked_claim_ref is None else checked_claim_ref
        ),
        checked_fact_refs=_checked_fact_refs(proof, compilation),
    )


def _verify_continuous_submission(
    *,
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2ContinuousCompilation,
    submission: BackendSubmission,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
) -> CheckedProofOutcome:
    """Freshly rebuild one continuous proof without consulting a proposal owner.

    Continuous proof transport deliberately records the proposal backend as the
    producer of its kernel rows.  That is not an authority shortcut here: this
    function rebuilds the compilation, exact-dyadic tree, resource ledger,
    retained-owner rows, concrete endpoints, and terminal strength from the
    request roots before comparing those fresh bytes with the untrusted wire.
    """

    _require_exact(compilation, upright.UprightSE2ContinuousCompilation, "compilation")
    _require_submission(submission)
    _require_exact(
        checker_policy, upright.UprightSE2CheckerReplayPolicy, "checker policy"
    )
    _check_continuous_checker_policy(solve_request, checker_policy)
    if selection.selection_disposition != "SELECTED":
        raise ValueError("selection must be selected for a submitted proof")
    if (
        selection.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or selection.solve_request_sha256 != solve_request.solve_request_sha256
        or selection.selected_backend_ref != _CONTINUOUS_BACKEND_REF
    ):
        raise ValueError("continuous selection roots do not match solve request")

    replayed = compile_upright_se2(solve_request)
    if type(replayed) is not upright.UprightSE2ContinuousCompilation or (
        canonical_json_bytes(replayed) != canonical_json_bytes(compilation)
    ):
        raise ValueError(
            "fresh continuous compilation replay does not match submission"
        )

    evidence, envelope, evidence_claim_ref = _submission_evidence(submission)
    if (
        evidence.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or evidence.solve_request_sha256 != solve_request.solve_request_sha256
        or evidence.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or envelope.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
    ):
        raise ValueError("continuous submission selection roots do not match")
    if (
        evidence.proposal_backend_owner_ref == upright.UPRIGHT_SE2_CHECKER_OWNER_REF
        or evidence.proposal_backend_capability_ref
        == upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF
    ):
        raise ValueError(
            "continuous proposal owner or capability cannot serve as checker"
        )
    if (
        type(submission) is BackendProposalSubmission
        and evidence_claim_ref != upright.UPRIGHT_SE2_EXACT_GLOBAL_CLAIM_DEFINITION_REF
    ):
        raise ValueError("continuous proposal claim does not bind registered evidence")
    selected_descriptors = tuple(
        descriptor
        for descriptor in solve_request.backend_descriptor_bundle.backend_descriptors
        if descriptor.backend_ref == selection.selected_backend_ref
    )
    if len(selected_descriptors) != 1:
        raise ValueError("selected continuous backend descriptor does not close")
    if (
        evidence.proposal_backend_ref != _CONTINUOUS_BACKEND_REF
        or evidence.proposal_backend_ref != selection.selected_backend_ref
        or evidence.proposal_backend_owner_ref != upright.UPRIGHT_SE2_BACKEND_OWNER_REF
        or evidence.proposal_backend_capability_ref
        != upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF
        or evidence.proposal_backend_build_sha256 != _CONTINUOUS_BACKEND_BUILD_SHA256
        or evidence.proposal_backend_build_sha256
        != selected_descriptors[0].implementation_build_sha256
    ):
        raise ValueError("continuous proposal backend identity does not bind selection")

    proof = _decode_continuous_proof(envelope)
    if canonical_json_bytes(proof.compilation) != canonical_json_bytes(compilation):
        raise ValueError(
            "continuous proof compilation does not match supplied compilation"
        )
    _check_continuous_roster_and_coverage(compilation, proof)
    fresh_proof, fresh_cell_total_intervals = _reconstruct_continuous_proof(compilation)
    if canonical_json_bytes(fresh_proof) != canonical_json_bytes(proof):
        raise ValueError(
            "fresh continuous replay does not match proof tree or owner rows"
        )
    if canonical_json_bytes(proof.total_resource_usage) != canonical_json_bytes(
        evidence.resource_usage
    ):
        raise ValueError(
            "continuous submission resource usage does not match proof ledger"
        )
    _check_continuous_selected_proposal_bounds(submission, proof)
    _check_continuous_terminal_evidence(submission, proof, compilation)
    _check_selected_unknown_reason(
        solve_request=solve_request,
        submission=submission,
        proof=fresh_proof,
    )
    checked_claim_ref = _continuous_submission_checked_claim(
        submission=submission,
        proof=fresh_proof,
        checker_policy=checker_policy,
        evidence_claim_ref=evidence_claim_ref,
        fresh_cell_total_intervals=fresh_cell_total_intervals,
    )
    return CheckedProofOutcome.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proof_material_sha256=envelope.proof_material_sha256,
        checker_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
        checker_build_sha256=upright.UPRIGHT_SE2_CHECKER_BUILD_SHA256,
        checker_disposition=(
            CheckerDisposition.ACCEPTED
            if checked_claim_ref is not None
            else CheckerDisposition.LIMITED
        ),
        checked_claim_definition_ref=(
            evidence_claim_ref if checked_claim_ref is None else checked_claim_ref
        ),
        checked_fact_refs=_continuous_checked_fact_refs(proof, compilation),
    )


def _check_continuous_checker_policy(
    solve_request: CounterfactualSolveRequest,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
) -> None:
    """Reuse the frozen policy gap root while selecting the distinct V4 checker."""

    if (
        checker_policy.proof_policy_sha256
        != solve_request.proof_policy.proof_policy_sha256
    ):
        raise ValueError("checker policy does not match solve request")
    solve_policy = upright.decode_upright_se2_solve_policy_definition_payload(
        solve_request.solve_policy_definition_bundle
    )
    if (
        checker_policy.solve_policy_definition_bundle_sha256
        != solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        or checker_policy.requested_gap != solve_policy.requested_gap
        or checker_policy.objective_bound_policy_ref
        != solve_policy.objective_bound_policy_ref
        or checker_policy.exact_global_claim_definition_ref
        != solve_policy.exact_global_claim_definition_ref
        or checker_policy.finite_gap_claim_definition_ref
        != solve_policy.finite_gap_claim_definition_ref
    ):
        raise ValueError("checker policy does not replay continuous solve-policy gap")
    if (
        checker_policy.checker_capability_ref
        != upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF
        or solve_request.proof_policy.required_checker_capability_refs
        != (upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,)
    ):
        raise ValueError("continuous checker capability does not close")


def _decode_continuous_proof(
    envelope: ProofMaterialEnvelope,
) -> upright.UprightSE2ContinuousProofMaterial:
    if (
        envelope.proof_material_definition_ref
        != upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
        or envelope.payload_schema_ref
        != upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
        or len(envelope.typed_payload) != 1
    ):
        raise ValueError("proof envelope is not the registered continuous payload")
    proof = upright.decode_upright_se2_continuous_proof_material(
        envelope.typed_payload[0]
    )
    if proof.continuous_proof_material_sha256 != canonical_sha256(
        proof.model_dump(
            mode="python",
            exclude={"continuous_proof_material_sha256"},
            round_trip=True,
        ),
        domain=proof.HASH_DOMAIN,
    ):
        raise ValueError("continuous proof material digest is not canonical")
    return proof


def _check_continuous_roster_and_coverage(
    compilation: upright.UprightSE2ContinuousCompilation,
    proof: upright.UprightSE2ContinuousProofMaterial,
) -> None:
    """Bind the sole authorized lift root before any fresh owner invocation."""

    if len(proof.continuous_tuple_roster) != 1:
        raise ValueError("checker requires one request-authorized continuous tuple")
    proof_tuple = proof.continuous_tuple_roster[0]
    if (
        canonical_json_bytes(proof_tuple.compiled_cells)
        != canonical_json_bytes(compilation.compiled_cells)
        or proof_tuple.translation_domain
        != compilation.endpoint_construction_recipe.translation_domain
        or proof_tuple.continuous_yaw_lift != compilation.continuous_yaw_lift
        or proof.coverage_artifact.authorization_sha256
        != compilation.operation.authorization_sha256
    ):
        raise ValueError("checker continuous roster or coverage root mismatch")
    leaves = tuple(
        row.compiled_cell
        for row in proof.evaluated_cells
        if row.leaf_disposition is not None
    )
    if canonical_json_bytes(leaves) != canonical_json_bytes(
        proof.coverage_artifact.cells
    ):
        raise ValueError("checker continuous final leaf roster does not match coverage")


def _reconstruct_continuous_proof(
    compilation: upright.UprightSE2ContinuousCompilation,
) -> tuple[
    upright.UprightSE2ContinuousProofMaterial,
    tuple[tuple[Fraction, Fraction], ...],
]:
    """Independently replay the bounded V4 tree from the sole compiled root.

    This intentionally has no backend argument, cache, or solver call.  The
    only computational dependencies are the public compiler bridge and the
    retained V4 kernel owners, all of which are invoked again under one fresh
    shared atomic budget.
    """

    request = compilation.source_solve_request
    root = compilation.compiled_cells[0]
    try:
        root_inputs = build_upright_se2_continuous_evaluation_inputs(compilation, root)
        budget = SO2AtomicBudgetV2(limit=root_inputs.resource_atomic_step_limit)
    except (TypeError, ValueError, ArithmeticError):
        budget = _fresh_continuous_fallback_budget(request)
    rows: list[upright.UprightSE2ProofCellEvaluation] = []
    stages: list[upright.UprightSE2ProofStageDelta] = []
    candidates: list[upright.UprightSE2ContinuousProposalCandidate] = []
    cell_total_intervals: list[tuple[Fraction, Fraction]] = []
    requested_gap = upright.decode_upright_se2_solve_policy_definition_payload(
        request.solve_policy_definition_bundle
    ).requested_gap.as_fraction

    def incumbent_upper() -> Fraction | None:
        if not candidates:
            return None
        return min(
            candidate.point_objective.total_upper.as_fraction
            for candidate in candidates
        )

    def add_unresolved(
        cell: upright.UprightSE2CompiledCell,
        evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
        label: str,
    ) -> None:
        row = _fresh_continuous_unresolved_row(
            request=request,
            cell=cell,
            evaluations=evaluations,
            label=label,
        )
        rows.append(row)
        stages.append(_fresh_continuous_stage(cell, row.owner_evaluations))

    def visit(cell: upright.UprightSE2CompiledCell, depth: int) -> None:
        evaluations, compound, visibility = _fresh_continuous_owner_evaluations(
            compilation=compilation,
            request=request,
            cell=cell,
            budget=budget,
        )
        classification = _fresh_continuous_classification(compound, visibility)
        point_cell = _fresh_continuous_point_cell(cell)
        if canonical_json_bytes(point_cell) != canonical_json_bytes(
            cell
        ) and classification in (
            upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
            None,
        ):
            point_evaluations, point_compound, point_visibility = (
                _fresh_continuous_owner_evaluations(
                    compilation=compilation,
                    request=request,
                    cell=point_cell,
                    budget=budget,
                )
            )
            point_classification = _fresh_continuous_classification(
                point_compound, point_visibility
            )
            point_row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=point_cell,
                owner_evaluations=point_evaluations,
                leaf_disposition=None,
                complete_domain_empty=None,
            )
            rows.append(point_row)
            stages.append(_fresh_continuous_stage(point_cell, point_evaluations))
            if (
                point_classification
                is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
                and point_compound is not None
                and getattr(point_compound, "bounds", None) is not None
            ):
                parent_row = _fresh_continuous_incomplete_parent_row(
                    request=request,
                    cell=cell,
                    evaluations=evaluations,
                    label="continuous-feasible-incomplete-witness",
                )
                rows.append(parent_row)
                stages.append(
                    _fresh_continuous_stage(cell, parent_row.owner_evaluations)
                )
                try:
                    candidates.append(
                        _fresh_continuous_candidate(
                            compilation=compilation,
                            final_inward_cell=parent_row,
                            point_row=point_row,
                            point_bounds=point_compound.bounds,
                        )
                    )
                except (TypeError, ValueError, ArithmeticError):
                    pass
                return
        if classification is upright.UprightSE2ProofLeafDisposition.UNRESOLVED:
            add_unresolved(cell, evaluations, "continuous-owner-nonexact")
            return
        if compound is None or getattr(compound, "bounds", None) is None:
            add_unresolved(cell, evaluations, "continuous-missing-compound-bounds")
            return
        bounds = compound.bounds
        if classification is upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE:
            row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=evaluations,
                leaf_disposition=classification,
                complete_domain_empty=True,
            )
            rows.append(row)
            stages.append(_fresh_continuous_stage(cell, evaluations))
            return
        if classification is None:
            lower, upper = _fresh_continuous_objective_interval(bounds)
            if _fresh_continuous_bounds_authorize_prune(
                cell_lower=lower,
                incumbent_upper=incumbent_upper(),
                requested_gap=requested_gap,
            ):
                row = upright.UprightSE2ProofCellEvaluation.seal(
                    compiled_cell=cell,
                    owner_evaluations=evaluations,
                    leaf_disposition=upright.UprightSE2ProofLeafDisposition.PRUNED,
                    complete_domain_empty=False,
                )
                rows.append(row)
                stages.append(_fresh_continuous_stage(cell, evaluations))
                cell_total_intervals.append((lower, upper))
                return
            children = _fresh_continuous_split_exact_dyadic_cell(cell, depth=depth)
            if children and depth < _MAX_CONTINUOUS_EXACT_DYADIC_REFINEMENT_DEPTH:
                internal = upright.UprightSE2ProofCellEvaluation.seal(
                    compiled_cell=cell,
                    owner_evaluations=evaluations,
                    leaf_disposition=None,
                    complete_domain_empty=None,
                )
                rows.append(internal)
                stages.append(_fresh_continuous_stage(cell, evaluations))
                for child in children:
                    visit(child, depth + 1)
                return
            add_unresolved(cell, evaluations, "continuous-finite-refinement-miss")
            return

        if classification is not upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE:
            raise ValueError(
                "continuous checker encountered an unknown leaf disposition"
            )
        if canonical_json_bytes(point_cell) == canonical_json_bytes(cell):
            final_row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=evaluations,
                leaf_disposition=upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE,
                complete_domain_empty=False,
            )
            rows.append(final_row)
            stages.append(_fresh_continuous_stage(cell, evaluations))
            try:
                candidates.append(
                    _fresh_continuous_candidate(
                        compilation=compilation,
                        final_inward_cell=final_row,
                        point_row=final_row,
                        point_bounds=bounds,
                    )
                )
                cell_total_intervals.append(
                    _fresh_continuous_objective_interval(bounds)
                )
            except (TypeError, ValueError, ArithmeticError, OverflowError):
                rows.pop()
                stages.pop()
                incomplete = _fresh_continuous_incomplete_parent_row(
                    request=request,
                    cell=cell,
                    evaluations=evaluations,
                    label="continuous-point-materialization-incomplete",
                )
                rows.append(incomplete)
                stages.append(
                    _fresh_continuous_stage(cell, incomplete.owner_evaluations)
                )
            return

        point_evaluations, point_compound, point_visibility = (
            _fresh_continuous_owner_evaluations(
                compilation=compilation,
                request=request,
                cell=point_cell,
                budget=budget,
            )
        )
        point_classification = _fresh_continuous_classification(
            point_compound, point_visibility
        )
        point_row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=point_cell,
            owner_evaluations=point_evaluations,
            leaf_disposition=None,
            complete_domain_empty=None,
        )
        rows.append(point_row)
        stages.append(_fresh_continuous_stage(point_cell, point_evaluations))
        if (
            point_classification
            is not upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
            or point_compound is None
            or getattr(point_compound, "bounds", None) is None
        ):
            incomplete = _fresh_continuous_incomplete_parent_row(
                request=request,
                cell=cell,
                evaluations=evaluations,
                label="continuous-inward-point-replay-incomplete",
            )
            rows.append(incomplete)
            stages.append(_fresh_continuous_stage(cell, incomplete.owner_evaluations))
            return
        final_row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=cell,
            owner_evaluations=evaluations,
            leaf_disposition=upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE,
            complete_domain_empty=False,
        )
        rows.append(final_row)
        stages.append(_fresh_continuous_stage(cell, evaluations))
        try:
            candidates.append(
                _fresh_continuous_candidate(
                    compilation=compilation,
                    final_inward_cell=final_row,
                    point_row=point_row,
                    point_bounds=point_compound.bounds,
                )
            )
            cell_total_intervals.append(_fresh_continuous_objective_interval(bounds))
        except (TypeError, ValueError, ArithmeticError, OverflowError):
            rows.pop()
            stages.pop()
            incomplete = _fresh_continuous_incomplete_parent_row(
                request=request,
                cell=cell,
                evaluations=evaluations,
                label="continuous-point-materialization-incomplete",
            )
            rows.append(incomplete)
            stages.append(_fresh_continuous_stage(cell, incomplete.owner_evaluations))

    visit(root, 0)
    return (
        _fresh_continuous_proof_material(
            compilation=compilation,
            rows=tuple(rows),
            stages=tuple(stages),
            proposal_candidates=tuple(candidates),
        ),
        tuple(cell_total_intervals),
    )


def _fresh_continuous_owner_evaluations(
    *,
    compilation: upright.UprightSE2ContinuousCompilation,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    budget: SO2AtomicBudgetV2,
) -> tuple[
    tuple[upright.UprightSE2RetainedOwnerEvaluation, ...], object | None, object | None
]:
    """Call every registered V4 owner for one fresh continuous proof cell."""

    evaluations: list[upright.UprightSE2RetainedOwnerEvaluation] = []

    def record(
        label: str,
        outcome: object,
        start_used: int,
        *,
        additional_exact_bounds: tuple[TypedValue, ...] = (),
    ) -> object:
        evaluations.append(
            _fresh_continuous_transport_owner_outcome(
                request=request,
                cell=cell,
                outcome=outcome,
                evaluator_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                label=label,
                atomic_steps=budget.used - start_used,
                additional_exact_bounds=additional_exact_bounds,
            )
        )
        return outcome

    try:
        inputs = build_upright_se2_continuous_evaluation_inputs(compilation, cell)
        if budget.limit != inputs.resource_atomic_step_limit:
            raise ValueError("continuous compiler bridge resource cap drifted")
        start = budget.used
        lift, yaw_bounds = _fresh_continuous_yaw_owner_inputs(compilation, cell, budget)
        record("continuous-yaw-lift", lift, start)
        if getattr(lift, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        if yaw_bounds is None:
            raise ValueError("exact continuous yaw lift lacks sin/cos bounds")
        start = budget.used
        record("continuous-yaw-sin-cos", yaw_bounds, start)
        if getattr(yaw_bounds, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        if yaw_bounds.bounds is None:
            raise ValueError("exact continuous yaw bounds are absent")

        dynamic_subjects: list[object] = []
        for index, source_box in enumerate(inputs.subject_boxes):
            start = budget.used
            outcome = compile_continuous_yaw_box_bounds_v4(
                source_box,
                cell=inputs.cell,
                pivot_xy=inputs.subject_pivot_xy,
                yaw_bounds=yaw_bounds.bounds,
                atomic_budget=budget,
            )
            record(f"continuous-subject-box-{index}", outcome, start)
            if outcome.kind is not ContinuousYawIntervalKindV4.EXACT:
                return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
            if outcome.bounds is None:
                raise ValueError("exact continuous subject bounds are absent")
            dynamic_subjects.append(outcome.bounds)

        start = budget.used
        static_lift, static_yaw_bounds = _fresh_continuous_static_yaw_owner_inputs(
            budget
        )
        record("continuous-static-yaw-lift", static_lift, start)
        if getattr(static_lift, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        if static_yaw_bounds is None:
            raise ValueError("exact static continuous yaw lift lacks bounds")
        start = budget.used
        record("continuous-static-yaw-sin-cos", static_yaw_bounds, start)
        if (
            getattr(static_yaw_bounds, "kind", None)
            is not ContinuousYawIntervalKindV4.EXACT
        ):
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        if static_yaw_bounds.bounds is None:
            raise ValueError("exact static continuous yaw bounds are absent")
        static_cell = ClosedXYCellV3(Fraction(), Fraction(), Fraction(), Fraction())

        def static_box(label: str, source_box: object) -> object | None:
            start_used = budget.used
            outcome = compile_continuous_yaw_box_bounds_v4(
                source_box,
                cell=static_cell,
                pivot_xy=(Fraction(), Fraction()),
                yaw_bounds=static_yaw_bounds.bounds,
                atomic_budget=budget,
            )
            record(label, outcome, start_used)
            if outcome.kind is not ContinuousYawIntervalKindV4.EXACT:
                return None
            return outcome.bounds

        static_obstacles: list[object] = []
        for index, source_box in enumerate(inputs.obstacle_boxes):
            boxed = static_box(f"continuous-obstacle-box-{index}", source_box)
            if boxed is None:
                return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
            static_obstacles.append(boxed)
        static_reference = static_box("continuous-reference-box", inputs.reference_box)
        if static_reference is None:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None

        visibility_input = inputs.visibility_inputs[0]
        start = budget.used
        visual_subject_outcome = compile_continuous_yaw_box_bounds_v4(
            visibility_input.subject,
            cell=inputs.cell,
            pivot_xy=inputs.subject_pivot_xy,
            yaw_bounds=yaw_bounds.bounds,
            atomic_budget=budget,
        )
        record("continuous-visibility-subject-box", visual_subject_outcome, start)
        if visual_subject_outcome.kind is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        if visual_subject_outcome.bounds is None:
            raise ValueError("exact continuous visibility subject bounds are absent")
        visual_subject = visual_subject_outcome.bounds
        visual_occluders: list[object] = []
        for index, source_box in enumerate(visibility_input.occluders):
            if source_box.box_id == visibility_input.subject.box_id:
                visual_occluders.append(visual_subject)
                continue
            boxed = static_box(f"continuous-visibility-occluder-{index}", source_box)
            if boxed is None:
                return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
            visual_occluders.append(boxed)
        visual_occluders = sorted(
            visual_occluders, key=lambda value: canonical_json_bytes(value.box.box_id)
        )
        start = budget.used
        visibility = evaluate_continuous_yaw_visibility_v4(
            context=visibility_input.context,
            subject=visual_subject,
            moving_subject_id=visibility_input.moving_subject_id,
            occluders=tuple(visual_occluders),
            required_occluder_ids=visibility_input.required_occluder_ids,
            policy=visibility_input.policy,
            atomic_budget=budget,
        )
        record("continuous-visibility", visibility, start)
        if visibility.kind is not ContinuousYawIntervalKindV4.EXACT:
            return (
                tuple(sorted(evaluations, key=canonical_json_bytes)),
                None,
                visibility,
            )

        start = budget.used
        compound = evaluate_continuous_yaw_cell_v4(
            cell=inputs.cell,
            yaw_bounds=yaw_bounds.bounds,
            subject_boxes=tuple(dynamic_subjects),
            obstacle_boxes=tuple(static_obstacles),
            support_surface=inputs.support_surface,
            relation=inputs.relation,
            reference_box=static_reference,
            near_far_threshold=inputs.near_far_threshold,
            policy=ContinuousYawCellPolicyV4(cardinal_policy=inputs.cell_policy),
            visibility=visibility,
            atomic_budget=budget,
            subject_pivot_xy=inputs.subject_pivot_xy,
            objective_subject_pivot_xy=inputs.objective_subject_pivot_xy,
        )
        point_objective_bound: tuple[TypedValue, ...] = ()
        if (
            compound.kind is ContinuousYawIntervalKindV4.EXACT
            and cell.x_lower == cell.x_upper
            and cell.y_lower == cell.y_upper
            and cell.yaw_interval.lower == cell.yaw_interval.upper
        ):
            if compound.bounds is None:
                raise ValueError("exact continuous point cell lacks compound bounds")
            point_objective_bound = (
                upright._retained_point_objective_value(
                    _fresh_continuous_objective(compound.bounds)
                ),
            )
        record(
            "continuous-compound-cell",
            compound,
            start,
            additional_exact_bounds=point_objective_bound,
        )
        return (
            tuple(sorted(evaluations, key=canonical_json_bytes)),
            compound,
            visibility,
        )
    except (TypeError, ValueError, ArithmeticError, OverflowError) as error:
        evaluations.append(
            _fresh_continuous_transport_owner_outcome(
                request=request,
                cell=cell,
                outcome=None,
                evaluator_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                label="continuous-bridge-or-owner-incomplete",
                forced_kind=_fresh_continuous_forced_kind(error),
                atomic_steps=0,
            )
        )
        return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None


def _fresh_continuous_transport_owner_outcome(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    outcome: object | None,
    evaluator_capability_ref: str,
    label: str,
    atomic_steps: int,
    forced_kind: upright.UprightSE2RetainedOwnerOutcomeKind | None = None,
    additional_exact_bounds: tuple[TypedValue, ...] = (),
) -> upright.UprightSE2RetainedOwnerEvaluation:
    """Serialize fresh V4 owner output through the compiler-owned transport seam."""

    if forced_kind is None:
        if outcome is None:
            raise TypeError("continuous retained owner outcome is required")
        kernel_kind = getattr(outcome, "kind", None)
        if type(kernel_kind) is not ContinuousYawIntervalKindV4:
            raise TypeError("retained owner returned an unknown continuous outcome")
        proof_kind = _fresh_continuous_outcome_kind(kernel_kind)
        raw_rows = tuple(getattr(outcome, "proof_rows", ()))
        raw_findings = tuple(getattr(outcome, "finding_codes", ()))
        bounds = getattr(outcome, "bounds", None)
        if (
            proof_kind is upright.UprightSE2RetainedOwnerOutcomeKind.EXACT
            and bounds is None
        ):
            raise ValueError("exact continuous retained owner outcome requires bounds")
    else:
        proof_kind = forced_kind
        raw_rows = (f"proof:spatialcf/upright-se2/{label}/incomplete",)
        raw_findings = (forced_kind.value,)
        bounds = None
    if type(atomic_steps) is not int or atomic_steps < 0:
        raise TypeError("continuous retained owner atomic usage must be exact")
    return build_upright_se2_retained_owner_evaluation(
        compiled_cell=cell,
        owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        evaluator_capability_ref=evaluator_capability_ref,
        outcome_kind=proof_kind,
        raw_proof_rows=raw_rows,
        raw_findings=raw_findings,
        atomic_steps=atomic_steps,
        resource_delta=_resource_delta(
            request,
            used=float(atomic_steps),
            exhausted=(
                proof_kind is upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT
            ),
        ),
        label=label,
        exact_bound_value=bounds,
        additional_exact_bounds=additional_exact_bounds,
        continuous_exact_rationals=True,
    )


def _fresh_continuous_outcome_kind(
    kind: ContinuousYawIntervalKindV4,
) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    return {
        ContinuousYawIntervalKindV4.EXACT: upright.UprightSE2RetainedOwnerOutcomeKind.EXACT,
        ContinuousYawIntervalKindV4.NUMERIC_GAP: upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP,
        ContinuousYawIntervalKindV4.RESOURCE_LIMIT: upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT,
        ContinuousYawIntervalKindV4.UNSUPPORTED: upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED,
    }[kind]


def _fresh_continuous_yaw_owner_inputs(
    compilation: upright.UprightSE2ContinuousCompilation,
    cell: upright.UprightSE2CompiledCell,
    budget: SO2AtomicBudgetV2,
) -> tuple[object, object | None]:
    """Derive the public V4 yaw owner inputs from the root/lift, never proof rows."""

    root = compilation.compiled_cells[0]
    if canonical_json_bytes(cell.yaw_interval) == canonical_json_bytes(
        root.yaw_interval
    ):
        yaw_domain = compilation.operation.yaw_domain
    else:
        lower = cell.yaw_interval.lower.as_fraction
        upper = cell.yaw_interval.upper.as_fraction
        sweep = upper - lower
        if not Fraction() <= sweep < Fraction(1):
            raise ArithmeticError("numeric gap: lifted child sweep is outside ARC")
        yaw_domain = upright.ContinuousYawArc(
            start_angle=upright.CanonicalSO2Angle(
                turns=_fresh_continuous_exact_float(
                    _fresh_canonical_continuous_turn(lower),
                    label="lifted child start",
                )
            ),
            ccw_sweep_turns=_fresh_continuous_exact_float(
                sweep, label="lifted child sweep"
            ),
        )
    lift = compile_continuous_yaw_lift_v4(yaw_domain, atomic_budget=budget)
    if lift.kind is not ContinuousYawIntervalKindV4.EXACT:
        return lift, None
    if lift.bounds is None:
        raise ValueError("exact continuous lift has no bounds")
    return lift, compile_lifted_turn_sin_cos_bounds_v4(
        lift.bounds,
        atomic_budget=budget,
    )


def _fresh_continuous_static_yaw_owner_inputs(
    budget: SO2AtomicBudgetV2,
) -> tuple[object, object | None]:
    lift = compile_continuous_yaw_lift_v4(
        upright.ContinuousYawArc(
            start_angle=upright.CanonicalSO2Angle(turns=0.0),
            ccw_sweep_turns=0.0,
        ),
        atomic_budget=budget,
    )
    if lift.kind is not ContinuousYawIntervalKindV4.EXACT:
        return lift, None
    if lift.bounds is None:
        raise ValueError("exact static continuous lift has no bounds")
    return lift, compile_lifted_turn_sin_cos_bounds_v4(
        lift.bounds,
        atomic_budget=budget,
    )


def _fresh_canonical_continuous_turn(value: Fraction) -> Fraction:
    while value < Fraction(-1, 2):
        value += 1
    while value >= Fraction(1, 2):
        value -= 1
    return value


def _fresh_continuous_exact_float(value: Fraction, *, label: str) -> float:
    result = float(value)
    if Fraction.from_float(result) != value:
        raise ArithmeticError(f"numeric gap: {label} is not an exact binary64 dyadic")
    return 0.0 if result == 0.0 else result


def _fresh_continuous_classification(
    cell_outcome: object | None,
    visibility_outcome: object | None,
) -> upright.UprightSE2ProofLeafDisposition | None:
    if (
        cell_outcome is None
        or visibility_outcome is None
        or getattr(cell_outcome, "kind", None) is not ContinuousYawIntervalKindV4.EXACT
        or getattr(visibility_outcome, "kind", None)
        is not ContinuousYawIntervalKindV4.EXACT
    ):
        return upright.UprightSE2ProofLeafDisposition.UNRESOLVED
    bounds = getattr(cell_outcome, "bounds", None)
    visibility = getattr(visibility_outcome, "bounds", None)
    if bounds is None or visibility is None:
        return upright.UprightSE2ProofLeafDisposition.UNRESOLVED
    if (
        bounds.outer_hard_constraint_failure
        or bounds.relation_outer_failure
        or visibility.classification == "OUTWARD"
    ):
        return upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
    if (
        bounds.inner_hard_constraint_proven
        and bounds.relation_inner_success
        and visibility.classification == "INWARD"
    ):
        return upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
    return None


def _fresh_continuous_objective(
    bounds: object,
) -> upright.UprightSE2ProposalPointObjective:
    """Transport the retained five-term V4 objective interval unchanged."""

    semantic_terms = bounds.semantic_objective_terms
    weighted_terms = bounds.weighted_objective_terms
    terms = tuple(
        upright.UprightSE2ProposalPointTerm(
            term_id=term_id,
            lower=upright.UprightSE2ExactRational(
                numerator=lower.numerator,
                denominator=lower.denominator,
            ),
            upper=upright.UprightSE2ExactRational(
                numerator=upper.numerator,
                denominator=upper.denominator,
            ),
        )
        for term_id, lower, upper in semantic_terms
    )
    total_lower = sum(
        (lower for _term_id, lower, _upper in weighted_terms), start=Fraction(0)
    )
    total_upper = sum(
        (upper for _term_id, _lower, upper in weighted_terms), start=Fraction(0)
    )
    return upright.UprightSE2ProposalPointObjective(
        terms=terms,
        total_lower=upright.UprightSE2ExactRational(
            numerator=total_lower.numerator,
            denominator=total_lower.denominator,
        ),
        total_upper=upright.UprightSE2ExactRational(
            numerator=total_upper.numerator,
            denominator=total_upper.denominator,
        ),
    )


def _fresh_continuous_objective_interval(bounds: object) -> tuple[Fraction, Fraction]:
    weighted_terms = bounds.weighted_objective_terms
    return (
        sum((lower for _term, lower, _upper in weighted_terms), start=Fraction(0)),
        sum((upper for _term, _lower, upper in weighted_terms), start=Fraction(0)),
    )


def _fresh_continuous_point_cell(
    cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Choose the fixed lower-owned exact-dyadic interior witness point."""

    x = (
        cell.x_lower
        if cell.x_lower == cell.x_upper
        else _fresh_continuous_dyadic(
            (cell.x_lower.as_fraction + cell.x_upper.as_fraction) / 2
        )
    )
    y = (
        cell.y_lower
        if cell.y_lower == cell.y_upper
        else _fresh_continuous_dyadic(
            (cell.y_lower.as_fraction + cell.y_upper.as_fraction) / 2
        )
    )
    u = (
        cell.yaw_interval.lower
        if cell.yaw_interval.lower == cell.yaw_interval.upper
        else _fresh_continuous_dyadic(
            (cell.yaw_interval.lower.as_fraction + cell.yaw_interval.upper.as_fraction)
            / 2
        )
    )
    if (
        x == cell.x_lower == cell.x_upper
        and y == cell.y_lower == cell.y_upper
        and u == cell.yaw_interval.lower == cell.yaw_interval.upper
    ):
        return cell
    return upright.UprightSE2CompiledCell.seal(
        cell_id=f"{cell.cell_id}/proposal-point",
        authorization_sha256=cell.authorization_sha256,
        x_lower=x,
        x_upper=x,
        y_lower=y,
        y_upper=y,
        yaw_interval=upright.LiftedYawInterval(
            lower=u,
            upper=u,
            seam_ownership="NONE",
        ),
    )


def _fresh_continuous_split_exact_dyadic_cell(
    cell: upright.UprightSE2CompiledCell,
    *,
    depth: int,
) -> tuple[upright.UprightSE2CompiledCell, ...]:
    """Reconstruct the frozen cyclic X/Y/U split, IDs, and seam ownership."""

    if type(depth) is not int or depth < 0:
        raise ValueError("continuous split depth must be non-negative")
    dimensions = (
        ("x", cell.x_lower.as_fraction, cell.x_upper.as_fraction),
        ("y", cell.y_lower.as_fraction, cell.y_upper.as_fraction),
        (
            "u",
            cell.yaw_interval.lower.as_fraction,
            cell.yaw_interval.upper.as_fraction,
        ),
    )
    selected: str | None = None
    for offset in range(len(dimensions)):
        axis, lower, upper = dimensions[(depth + offset) % len(dimensions)]
        if lower < upper:
            selected = axis
            break
    if selected is None:
        return ()

    def child(
        *,
        lower: upright.ExactDyadic,
        upper: upright.ExactDyadic,
        side: str,
    ) -> upright.UprightSE2CompiledCell:
        if selected == "x":
            return upright.UprightSE2CompiledCell.seal(
                cell_id=f"{cell.cell_id}/split-x-{side}",
                authorization_sha256=cell.authorization_sha256,
                x_lower=lower,
                x_upper=upper,
                y_lower=cell.y_lower,
                y_upper=cell.y_upper,
                yaw_interval=cell.yaw_interval,
            )
        if selected == "y":
            return upright.UprightSE2CompiledCell.seal(
                cell_id=f"{cell.cell_id}/split-y-{side}",
                authorization_sha256=cell.authorization_sha256,
                x_lower=cell.x_lower,
                x_upper=cell.x_upper,
                y_lower=lower,
                y_upper=upper,
                yaw_interval=cell.yaw_interval,
            )
        seam = "NONE"
        if side == "lower" and cell.yaw_interval.seam_ownership == "LOWER_OWNS_SEAM":
            seam = "LOWER_OWNS_SEAM"
        elif (
            side == "upper"
            and cell.yaw_interval.seam_ownership == "UPPER_OWNS_ENDPOINT"
        ):
            seam = "UPPER_OWNS_ENDPOINT"
        return upright.UprightSE2CompiledCell.seal(
            cell_id=f"{cell.cell_id}/split-u-{side}",
            authorization_sha256=cell.authorization_sha256,
            x_lower=cell.x_lower,
            x_upper=cell.x_upper,
            y_lower=cell.y_lower,
            y_upper=cell.y_upper,
            yaw_interval=upright.LiftedYawInterval(
                lower=lower,
                upper=upper,
                seam_ownership=seam,
            ),
        )

    if selected == "x":
        lower, upper = cell.x_lower, cell.x_upper
    elif selected == "y":
        lower, upper = cell.y_lower, cell.y_upper
    else:
        lower, upper = cell.yaw_interval.lower, cell.yaw_interval.upper
    midpoint = _fresh_continuous_dyadic((lower.as_fraction + upper.as_fraction) / 2)
    return (
        child(lower=lower, upper=midpoint, side="lower"),
        child(lower=midpoint, upper=upper, side="upper"),
    )


def _fresh_continuous_dyadic(value: Fraction) -> upright.ExactDyadic:
    if value.denominator & (value.denominator - 1):
        raise ValueError("continuous subdivision endpoint is not dyadic")
    return upright.ExactDyadic(numerator=value.numerator, denominator=value.denominator)


def _fresh_continuous_candidate(
    *,
    compilation: upright.UprightSE2ContinuousCompilation,
    final_inward_cell: upright.UprightSE2ProofCellEvaluation,
    point_row: upright.UprightSE2ProofCellEvaluation,
    point_bounds: object,
) -> upright.UprightSE2ContinuousProposalCandidate:
    """Freshly materialize and bind the concrete endpoint from exact point owners."""

    point_cell = point_row.compiled_cell
    objective = _fresh_continuous_objective(point_bounds)
    point = Vec2(
        x=float(point_cell.x_lower.as_fraction),
        y=float(point_cell.y_lower.as_fraction),
    )
    endpoint = materialize_upright_se2_continuous_endpoint(
        compilation,
        point,
        point_cell.yaw_interval.lower,
    )
    point_evaluation = upright.UprightSE2ProposalPointEvaluation.seal(
        point_cell_evaluation=point_row,
        point_objective=objective,
    )
    return upright.UprightSE2ContinuousProposalCandidate.seal(
        final_inward_cell=final_inward_cell,
        selected_translation_xy_m=point,
        selected_lifted_yaw=point_cell.yaw_interval.lower,
        point_evaluation=point_evaluation,
        point_objective=objective,
        materialized_endpoint=endpoint,
        program=endpoint.program,
    )


def _fresh_continuous_unresolved_row(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
    label: str,
) -> upright.UprightSE2ProofCellEvaluation:
    """Represent a nonexact or finite-search boundary as UNKNOWN-only evidence."""

    owner_evaluations = evaluations
    if not any(
        evaluation.outcome_kind is not upright.UprightSE2RetainedOwnerOutcomeKind.EXACT
        for evaluation in owner_evaluations
    ):
        owner_evaluations = tuple(
            sorted(
                (
                    *owner_evaluations,
                    _fresh_continuous_transport_owner_outcome(
                        request=request,
                        cell=cell,
                        outcome=None,
                        evaluator_capability_ref=(
                            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                        ),
                        label=label,
                        forced_kind=(
                            upright.UprightSE2RetainedOwnerOutcomeKind.FINITE_MISS
                        ),
                        atomic_steps=0,
                    ),
                ),
                key=canonical_json_bytes,
            )
        )
    return upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=cell,
        owner_evaluations=owner_evaluations,
        leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
        complete_domain_empty=False,
    )


def _fresh_continuous_incomplete_parent_row(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
    label: str,
) -> upright.UprightSE2ProofCellEvaluation:
    incomplete = _fresh_continuous_transport_owner_outcome(
        request=request,
        cell=cell,
        outcome=None,
        evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        label=label,
        forced_kind=upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE,
        atomic_steps=0,
    )
    return upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=cell,
        owner_evaluations=tuple(
            sorted((*evaluations, incomplete), key=canonical_json_bytes)
        ),
        leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
        complete_domain_empty=False,
    )


def _fresh_continuous_bounds_authorize_prune(
    *,
    cell_lower: Fraction,
    incumbent_upper: Fraction | None,
    requested_gap: Fraction,
) -> bool:
    if incumbent_upper is None:
        return False
    return cell_lower > incumbent_upper or (
        requested_gap > 0 and cell_lower >= incumbent_upper - requested_gap
    )


def _fresh_continuous_forced_kind(
    error: BaseException,
) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    if isinstance(error, ArithmeticError):
        return upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP
    if "unsupported" in str(error).lower():
        return upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED
    return upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE


def _fresh_continuous_fallback_budget(
    request: CounterfactualSolveRequest,
) -> SO2AtomicBudgetV2:
    limits = request.resource_policy.limits
    if len(limits) != 1:
        raise ValueError("continuous upright se2 requires one resource limit")
    finite_limit = limits[0].finite_limit
    if type(finite_limit) is not float or finite_limit < 1.0:
        raise ValueError("continuous upright se2 resource limit is unusable")
    return SO2AtomicBudgetV2(limit=int(finite_limit))


def _fresh_continuous_stage(
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
) -> upright.UprightSE2ProofStageDelta:
    """Record one non-reset fresh stage for every replayed proof cell."""

    ordered = tuple(sorted(evaluations, key=canonical_json_bytes))
    return upright.UprightSE2ProofStageDelta.seal(
        stage_ref=(
            "stage:spatialcf/upright-se2/continuous-retained-evaluation/"
            f"{cell.compiled_cell_sha256}"
        ),
        owner_evaluations=ordered,
        resource_delta=_fresh_continuous_aggregate_usage(
            tuple(evaluation.resource_delta for evaluation in ordered)
        ),
    )


def _fresh_continuous_proof_material(
    *,
    compilation: upright.UprightSE2ContinuousCompilation,
    rows: tuple[upright.UprightSE2ProofCellEvaluation, ...],
    stages: tuple[upright.UprightSE2ProofStageDelta, ...],
    proposal_candidates: tuple[upright.UprightSE2ContinuousProposalCandidate, ...],
) -> upright.UprightSE2ContinuousProofMaterial:
    """Seal only the independently reconstructed proof transport for comparison."""

    ordered_rows = tuple(
        sorted(
            rows, key=lambda row: _fresh_continuous_cell_order_key(row.compiled_cell)
        )
    )
    leaves = tuple(row for row in ordered_rows if row.leaf_disposition is not None)
    unresolved = tuple(
        row
        for row in leaves
        if row.leaf_disposition is upright.UprightSE2ProofLeafDisposition.UNRESOLVED
    )
    pruned = tuple(
        row
        for row in leaves
        if row.leaf_disposition is upright.UprightSE2ProofLeafDisposition.PRUNED
    )
    proposals = tuple(
        row
        for row in leaves
        if row.leaf_disposition
        is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
    )
    ordered_stages = tuple(sorted(stages, key=canonical_json_bytes))
    ordered_candidates = tuple(
        sorted(proposal_candidates, key=lambda candidate: candidate.canonical_order_key)
    )
    ledger = upright.UprightSE2ProofResourceLedger.seal(
        stage_deltas=ordered_stages,
        canonical_total_resource_usage=_fresh_continuous_aggregate_usage(
            tuple(stage.resource_delta for stage in ordered_stages)
        ),
    )
    return upright.UprightSE2ContinuousProofMaterial.seal(
        solve_request_sha256=compilation.solve_request_sha256,
        semantic_closure_sha256=compilation.semantic_closure.semantic_closure_sha256,
        continuous_upright_se2_compilation_sha256=(
            compilation.continuous_upright_se2_compilation_sha256
        ),
        compilation=compilation,
        continuous_tuple_roster=(
            upright.UprightSE2ContinuousProofTuple.seal(
                authorization=compilation.operation.authorization,
                reference_id=compilation.endpoint_construction_recipe.reference_id,
                translation_domain=compilation.operation.translation_domain,
                continuous_yaw_lift=compilation.continuous_yaw_lift,
                compiled_cells=compilation.compiled_cells,
            ),
        ),
        coverage_artifact=upright.UprightSE2CoverageArtifact.seal(
            authorization_sha256=compilation.operation.authorization_sha256,
            cells=tuple(row.compiled_cell for row in leaves),
            unresolved_cell_sha256s=tuple(
                sorted(row.compiled_cell.compiled_cell_sha256 for row in unresolved)
            ),
        ),
        compiled_cell_sha256s=tuple(
            row.compiled_cell.compiled_cell_sha256 for row in leaves
        ),
        evaluated_cells=ordered_rows,
        proposal_order=proposals,
        proposal_candidates=ordered_candidates,
        incumbent_candidate_sha256=(
            None
            if not ordered_candidates
            else ordered_candidates[0].continuous_proposal_candidate_sha256
        ),
        prune_decisions=tuple(
            upright.UprightSE2ProofPruneDecision.seal(
                cell_evaluation=row,
                prune_reason_codes=(
                    "reason:spatialcf/upright-se2/continuous-objective-bound",
                ),
            )
            for row in pruned
        ),
        unresolved_frontier=tuple(
            upright.UprightSE2ProofFrontierRow.seal(
                cell_evaluation=row,
                frontier_reason_codes=(
                    "reason:spatialcf/upright-se2/continuous-owner-nonexact",
                ),
            )
            for row in unresolved
        ),
        resource_ledger=ledger,
    )


def _fresh_continuous_cell_order_key(
    cell: upright.UprightSE2CompiledCell,
) -> tuple[Fraction, Fraction, bytes]:
    return (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        canonical_json_bytes(cell.cell_id),
    )


def _fresh_continuous_aggregate_usage(
    usages: tuple[ResourceUsage, ...],
) -> ResourceUsage:
    if not usages:
        raise ValueError("resource aggregation requires at least one usage")
    accounting_refs = {usage.accounting_claim_definition_ref for usage in usages}
    if len(accounting_refs) != 1:
        raise ValueError("resource aggregation requires one accounting claim")
    totals: dict[str, float] = {}
    for usage in usages:
        for entry in usage.entries:
            totals[entry.resource_definition_ref] = (
                totals.get(entry.resource_definition_ref, 0.0) + entry.used
            )
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": next(iter(accounting_refs)),
            "entries": tuple(
                {
                    "resource_definition_ref": reference,
                    "used": used,
                }
                for reference, used in sorted(
                    totals.items(), key=lambda item: canonical_json_bytes(item[0])
                )
            ),
            "exhausted": any(usage.exhausted for usage in usages),
        }
    )


def _check_continuous_selected_proposal_bounds(
    submission: BackendSubmission,
    proof: upright.UprightSE2ContinuousProofMaterial,
) -> None:
    if type(submission) is not BackendProposalSubmission:
        return
    if not proof.proposal_candidates:
        raise ValueError("continuous proposal submission lacks a fresh candidate")
    candidate = proof.proposal_candidates[0]
    proposal = submission.proposal
    if (
        proposal.program_sha256 != candidate.program.program_sha256
        or proposal.after_scene_state_sha256
        != candidate.program.after_scene_state_sha256
        or proposal.objective_lower_bound
        != float(candidate.point_objective.total_lower.as_fraction)
        or proposal.objective_upper_bound
        != float(candidate.point_objective.total_upper.as_fraction)
    ):
        raise ValueError(
            "outer continuous proposal does not bind fresh selected witness"
        )


def _check_continuous_terminal_evidence(
    submission: BackendSubmission,
    proof: upright.UprightSE2ContinuousProofMaterial,
    compilation: upright.UprightSE2ContinuousCompilation,
) -> None:
    leaves = tuple(
        row for row in proof.evaluated_cells if row.leaf_disposition is not None
    )
    if type(submission) is BackendCompleteUnsatEvidence:
        if (
            submission.authorized_domain_sha256
            != compilation.endpoint_construction_recipe.translation_domain_sha256
            or submission.complete_domain_coverage_artifact_sha256
            != proof.coverage_artifact.coverage_artifact_sha256
            or not leaves
            or any(
                row.leaf_disposition
                is not upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
                or row.complete_domain_empty is not True
                for row in leaves
            )
        ):
            raise ValueError("complete continuous UNSAT lacks full outward coverage")
    elif type(submission) is BackendProposalSubmission:
        if not proof.proposal_candidates:
            raise ValueError("continuous proposal submission lacks a checked candidate")
    elif (
        type(submission) is BackendUnknownEvidence
        and submission.resource_usage != proof.total_resource_usage
    ):
        raise ValueError("continuous unknown resource usage does not match proof")


def _continuous_submission_checked_claim(
    *,
    submission: BackendSubmission,
    proof: upright.UprightSE2ContinuousProofMaterial,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
    evidence_claim_ref: str,
    fresh_cell_total_intervals: tuple[tuple[Fraction, Fraction], ...],
) -> str | None:
    """Apply fresh exact-equality and directed `U_witness - L_global` criteria."""

    if type(submission) is BackendCompleteUnsatEvidence:
        return evidence_claim_ref
    if type(submission) is not BackendProposalSubmission:
        return None
    if (
        proof.unresolved_frontier
        or not proof.proposal_candidates
        or not fresh_cell_total_intervals
    ):
        return None
    witness = proof.proposal_candidates[0].point_objective
    global_lower = min(lower for lower, _upper in fresh_cell_total_intervals)
    candidate_total_intervals = tuple(
        (
            candidate.point_objective.total_lower.as_fraction,
            candidate.point_objective.total_upper.as_fraction,
        )
        for candidate in proof.proposal_candidates
    )
    comparator_totals = (*fresh_cell_total_intervals, *candidate_total_intervals)
    if (
        comparator_totals
        and all(lower == upper for lower, upper in comparator_totals)
        and len(set(comparator_totals)) == 1
    ):
        return checker_policy.exact_global_claim_definition_ref
    if (
        witness.total_upper.as_fraction - global_lower
        <= checker_policy.requested_gap.as_fraction
    ):
        return checker_policy.finite_gap_claim_definition_ref
    return None


def _continuous_checked_fact_refs(
    proof: upright.UprightSE2ContinuousProofMaterial,
    compilation: upright.UprightSE2ContinuousCompilation,
) -> tuple[dict[str, str], ...]:
    values = (
        proof.continuous_proof_material_sha256,
        compilation.continuous_upright_se2_compilation_sha256,
        proof.coverage_artifact.coverage_artifact_sha256,
        proof.resource_ledger.proof_resource_ledger_sha256,
    )
    return tuple(
        sorted(
            (
                {
                    "artifact_schema_ref": _CHECKED_FACT_SCHEMA_REF,
                    "artifact_sha256": canonical_sha256(
                        value,
                        domain=(
                            "spatialcf/counterfactual/upright-se2/"
                            "continuous-checked-fact/1.0"
                        ),
                    ),
                }
                for value in values
            ),
            key=canonical_json_bytes,
        )
    )


def _submission_evidence(
    submission: BackendSubmission,
) -> tuple[
    BackendProposalSubmission | BackendCompleteUnsatEvidence | BackendUnknownEvidence,
    ProofMaterialEnvelope,
    str,
]:
    if type(submission) is BackendProposalSubmission:
        proposal = submission.proposal
        if submission.backend_proposal_sha256 != proposal.backend_proposal_sha256:
            raise ValueError("proposal wrapper does not bind proposal")
        return proposal, proposal.proof_material, proposal.proposal_claim_definition_ref
    if type(submission) is BackendCompleteUnsatEvidence:
        if (
            submission.complete_domain_claim_definition_ref
            != "definition:spatialcf/upright-se2/claim-proven-unsat/1.0"
        ):
            raise ValueError(
                "complete-domain UNSAT claim does not bind registered evidence"
            )
        return (
            submission,
            submission.proof_material,
            submission.complete_domain_claim_definition_ref,
        )
    if type(submission) is BackendUnknownEvidence:
        return (
            submission,
            submission.proof_material,
            submission.reason_claim_definition_ref,
        )
    raise TypeError("checker accepts only exact M3 backend submissions")


def _require_submission(submission: object) -> None:
    """Reject bypassed/tampered discriminated wires before inspecting evidence."""

    model_type = type(submission)
    if model_type not in (
        BackendProposalSubmission,
        BackendCompleteUnsatEvidence,
        BackendUnknownEvidence,
    ):
        raise TypeError("checker accepts only exact M3 backend submissions")
    checked = model_type.model_validate(
        submission.model_dump(mode="python", round_trip=True), strict=True
    )
    if canonical_json_bytes(checked) != canonical_json_bytes(submission):
        raise ValueError("submission must use canonical bytes")


def _decode_proof(envelope: ProofMaterialEnvelope) -> upright.UprightSE2ProofMaterial:
    if (
        envelope.proof_material_definition_ref
        != upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
        or envelope.payload_schema_ref
        != upright.UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF
        or len(envelope.typed_payload) != 1
    ):
        raise ValueError("proof envelope is not the registered cardinal payload")
    proof = upright.decode_upright_se2_proof_material(envelope.typed_payload[0])
    if proof.proof_material_sha256 != canonical_sha256(
        proof.model_dump(
            mode="python",
            exclude={"proof_material_sha256"},
            round_trip=True,
        ),
        domain=proof.HASH_DOMAIN,
    ):
        raise ValueError("proof material digest is not canonical")
    return proof


def _cardinal_replay_rows(
    proof: upright.UprightSE2ProofMaterial,
) -> tuple[tuple[upright.UprightSE2ProofCellEvaluation, str | None], ...]:
    """Retain canonical cell order while replaying associated points first."""

    rows = {row.compiled_cell.cell_id: row for row in proof.evaluated_cells}
    point_parents: dict[str, str] = {}
    for parent in proof.evaluated_cells:
        cell = parent.compiled_cell
        if cell.x_lower == cell.x_upper and cell.y_lower == cell.y_upper:
            continue
        point = rows.get(f"{cell.cell_id}/proposal-point")
        if point is None:
            continue
        point_cell = point.compiled_cell
        if (
            point.leaf_disposition is not None
            or point.complete_domain_empty is not None
            or point_cell.authorization_sha256 != cell.authorization_sha256
            or point_cell.yaw_interval != cell.yaw_interval
        ):
            raise ValueError("cardinal point-first row does not bind its parent")
        for lower, upper, point_lower, point_upper in (
            (cell.x_lower, cell.x_upper, point_cell.x_lower, point_cell.x_upper),
            (cell.y_lower, cell.y_upper, point_cell.y_lower, point_cell.y_upper),
        ):
            if point_lower != point_upper or not (
                lower == point_lower == upper
                if lower == upper
                else lower.as_fraction < point_lower.as_fraction < upper.as_fraction
            ):
                raise ValueError("cardinal point-first row is not strict interior")
        point_parents[point_cell.cell_id] = cell.cell_id

    ordered = []
    seen: set[str] = set()
    for row in proof.evaluated_cells:
        cell_id = row.compiled_cell.cell_id
        point_id = f"{cell_id}/proposal-point"
        if point_id in point_parents and point_id not in seen:
            ordered.append((rows[point_id], cell_id))
            seen.add(point_id)
        if cell_id not in seen:
            ordered.append((row, point_parents.get(cell_id)))
            seen.add(cell_id)
    return tuple(ordered)


def _fresh_cardinal_incomplete_parent(
    request: CounterfactualSolveRequest,
    row: upright.UprightSE2ProofCellEvaluation,
) -> tuple[upright.UprightSE2RetainedOwnerEvaluation, ...]:
    """Close an unevaluated parent only after its fresh nonexact point."""

    if (
        row.leaf_disposition is not upright.UprightSE2ProofLeafDisposition.UNRESOLVED
        or row.complete_domain_empty is not False
    ):
        raise ValueError("nonexact cardinal point requires an unresolved parent")
    return (
        build_upright_se2_retained_owner_evaluation(
            compiled_cell=row.compiled_cell,
            owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
            evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            outcome_kind=upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE,
            raw_proof_rows=(
                "proof:spatialcf/upright-se2/point-first-parent-incomplete/incomplete",
            ),
            raw_findings=("INCOMPLETE",),
            atomic_steps=0,
            resource_delta=_resource_delta(request, used=0.0, exhausted=False),
            label="point-first-parent-incomplete",
        ),
    )


def _replay_cells(
    compilation: upright.UprightSE2Compilation,
    proof: upright.UprightSE2ProofMaterial,
) -> tuple[upright.UprightSE2ProposalPointObjective, ...]:
    """Replay point-first owner rows under one fresh shared resource budget."""

    budget: SO2AtomicBudgetV2 | None = None
    replayed_steps = 0
    exhausted = False
    incomplete_parents: set[str] = set()
    fresh_feasible_leaf_objectives = []
    for row, point_parent_id in _cardinal_replay_rows(proof):
        if row.compiled_cell.cell_id in incomplete_parents:
            expected_evaluations = _fresh_cardinal_incomplete_parent(
                compilation.source_solve_request, row
            )
            if canonical_json_bytes(row.owner_evaluations) != canonical_json_bytes(
                expected_evaluations
            ):
                raise ValueError(
                    "fresh nonexact point does not bind cardinal incomplete parent"
                )
            continue
        inputs = build_upright_se2_cardinal_evaluation_inputs(
            compilation, row.compiled_cell
        )
        if budget is None:
            budget = SO2AtomicBudgetV2(limit=inputs.resource_atomic_step_limit)
        elif budget.limit != inputs.resource_atomic_step_limit:
            raise ValueError("fresh cardinal bridge resource cap drifted")
        box = evaluate_fixed_cardinal_cell_v3(
            cell=inputs.cell,
            quarter_turns_ccw=inputs.quarter_turns_ccw,
            subject_boxes=inputs.subject_boxes,
            obstacle_boxes=inputs.obstacle_boxes,
            support_surface=inputs.support_surface,
            relation=inputs.relation,
            reference_box=inputs.reference_box,
            near_far_threshold=inputs.near_far_threshold,
            policy=inputs.cell_policy,
            atomic_budget=budget,
            subject_pivot_xy=inputs.subject_pivot_xy,
            objective_subject_pivot_xy=inputs.objective_subject_pivot_xy,
        )
        visibility = tuple(
            evaluate_fixed_cardinal_visibility_v3(
                context=item.context,
                cell=item.cell,
                subject=item.subject,
                moving_subject_id=item.moving_subject_id,
                occluders=item.occluders,
                required_occluder_ids=item.required_occluder_ids,
                policy=item.policy,
                atomic_budget=budget,
            )
            for item in inputs.visibility_inputs
        )
        replayed_steps += sum(
            outcome.atomic_steps_used for outcome in (box, *visibility)
        )
        exhausted = exhausted or any(
            outcome.kind is CardinalKernelKindV3.RESOURCE_LIMIT
            for outcome in (box, *visibility)
        )
        if point_parent_id is not None and any(
            outcome.kind is not CardinalKernelKindV3.EXACT
            for outcome in (box, *visibility)
        ):
            incomplete_parents.add(point_parent_id)
        expected_leaf = _classify_leaf(box, visibility)
        if row.leaf_disposition is not None and not _fresh_leaf_matches_submission(
            submitted=row.leaf_disposition, fresh=expected_leaf
        ):
            raise ValueError("fresh kernel replay does not match retained leaf")
        expected_evaluations = _fresh_owner_evaluations(
            solve_request=compilation.source_solve_request,
            row=row,
            box=box,
            visibility=visibility,
            include_finite_refinement_miss=(
                expected_leaf is None
                and row.leaf_disposition
                is upright.UprightSE2ProofLeafDisposition.UNRESOLVED
            ),
        )
        if canonical_json_bytes(row.owner_evaluations) != canonical_json_bytes(
            expected_evaluations
        ):
            raise ValueError(
                "fresh kernel replay does not match complete retained owner evaluations"
            )
        if (
            row.leaf_disposition
            is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        ):
            fresh_feasible_leaf_objectives.append(
                _objective_from_box_bounds(box.bounds)
            )
    if budget is None or budget.used != replayed_steps:
        raise ValueError("fresh cardinal owner usage does not close shared budget")
    fresh_usage = _resource_delta(
        compilation.source_solve_request,
        used=float(replayed_steps),
        exhausted=exhausted,
    )
    if canonical_json_bytes(fresh_usage) != canonical_json_bytes(
        proof.total_resource_usage
    ):
        raise ValueError("fresh kernel replay resource count does not match ledger")
    return tuple(fresh_feasible_leaf_objectives)


def _replay_candidates(
    compilation: upright.UprightSE2Compilation,
    proof: upright.UprightSE2ProofMaterial,
) -> tuple[upright.UprightSE2ProposalPointObjective, ...]:
    fresh_objectives = []
    for candidate in proof.proposal_candidates:
        endpoint = materialize_upright_se2_endpoint(
            compilation, candidate.selected_translation_xy_m
        )
        if canonical_json_bytes(endpoint) != canonical_json_bytes(
            candidate.materialized_endpoint
        ) or canonical_json_bytes(endpoint.program) != canonical_json_bytes(
            candidate.program
        ):
            raise ValueError("fresh endpoint materialization does not match proposal")
        if (
            endpoint.program.state_delta_manifest
            != compilation.state_footprint.state_delta_manifest
            or endpoint.program.grounded_obligation_set_sha256
            != compilation.grounded_obligations.grounded_obligation_set_sha256
            or endpoint.program.after_scene_state_sha256
            != candidate.program.after_scene_state_sha256
        ):
            raise ValueError("fresh endpoint does not bind state delta or obligations")
        objective = _fresh_point_objective(
            compilation, candidate.point_evaluation.point_cell_evaluation.compiled_cell
        )
        if canonical_json_bytes(objective) != canonical_json_bytes(
            candidate.point_objective
        ):
            raise ValueError("fresh point objective does not match proposal")
        fresh_objectives.append(objective)
    return tuple(fresh_objectives)


def _fresh_point_objective(
    compilation: upright.UprightSE2Compilation,
    point_cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2ProposalPointObjective:
    """Recompute all five raw point intervals through fresh owner calls."""

    inputs = build_upright_se2_cardinal_evaluation_inputs(compilation, point_cell)
    budget = SO2AtomicBudgetV2(limit=inputs.resource_atomic_step_limit)
    box = evaluate_fixed_cardinal_cell_v3(
        cell=inputs.cell,
        quarter_turns_ccw=inputs.quarter_turns_ccw,
        subject_boxes=inputs.subject_boxes,
        obstacle_boxes=inputs.obstacle_boxes,
        support_surface=inputs.support_surface,
        relation=inputs.relation,
        reference_box=inputs.reference_box,
        near_far_threshold=inputs.near_far_threshold,
        policy=inputs.cell_policy,
        atomic_budget=budget,
        subject_pivot_xy=inputs.subject_pivot_xy,
        objective_subject_pivot_xy=inputs.objective_subject_pivot_xy,
    )
    visibility = tuple(
        evaluate_fixed_cardinal_visibility_v3(
            context=item.context,
            cell=item.cell,
            subject=item.subject,
            moving_subject_id=item.moving_subject_id,
            occluders=item.occluders,
            required_occluder_ids=item.required_occluder_ids,
            policy=item.policy,
            atomic_budget=budget,
        )
        for item in inputs.visibility_inputs
    )
    if any(
        outcome.kind is not CardinalKernelKindV3.EXACT for outcome in (box, *visibility)
    ):
        raise ValueError("fresh point replay is not exact")
    semantic_terms = box.bounds.common_cell_semantic_objective_terms
    weighted_terms = box.bounds.common_cell_objective_terms
    terms = tuple(
        upright.UprightSE2ProposalPointTerm(
            term_id=term_id,
            lower=upright.UprightSE2ExactRational(
                numerator=lower.numerator, denominator=lower.denominator
            ),
            upper=upright.UprightSE2ExactRational(
                numerator=upper.numerator, denominator=upper.denominator
            ),
        )
        for term_id, lower, upper in semantic_terms
    )
    lower = sum((item[1] for item in weighted_terms), start=Fraction(0))
    upper = sum((item[2] for item in weighted_terms), start=Fraction(0))
    return upright.UprightSE2ProposalPointObjective(
        terms=terms,
        total_lower=upright.UprightSE2ExactRational(
            numerator=lower.numerator, denominator=lower.denominator
        ),
        total_upper=upright.UprightSE2ExactRational(
            numerator=upper.numerator, denominator=upper.denominator
        ),
    )


def _fresh_owner_evaluations(
    *,
    solve_request: CounterfactualSolveRequest,
    row: upright.UprightSE2ProofCellEvaluation,
    box: object,
    visibility: tuple[object, ...],
    include_finite_refinement_miss: bool,
) -> tuple[upright.UprightSE2RetainedOwnerEvaluation, ...]:
    """Serialize fresh retained DTOs through the shared compiler seam only."""

    point_row = row.compiled_cell.cell_id.endswith("/proposal-point")
    exact = all(
        outcome.kind is CardinalKernelKindV3.EXACT for outcome in (box, *visibility)
    )
    additional_exact_bounds = ()
    if exact:
        additional_exact_bounds = (
            upright._retained_point_objective_value(
                _objective_from_box_bounds(box.bounds)
            ),
        )
    label_prefix = "fixed-cardinal-point" if point_row else "fixed-cardinal"
    outcomes = ((f"{label_prefix}-cell", box),) + tuple(
        (f"{label_prefix}-visibility-{index}", outcome)
        for index, outcome in enumerate(visibility)
    )
    kernel_evaluations = tuple(
        sorted(
            (
                build_upright_se2_retained_owner_evaluation(
                    compiled_cell=row.compiled_cell,
                    owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
                    evaluator_capability_ref=(
                        upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                    ),
                    outcome_kind=_proof_kind(outcome.kind),
                    raw_proof_rows=tuple(outcome.proof_rows),
                    raw_findings=tuple(outcome.finding_codes),
                    atomic_steps=outcome.atomic_steps_used,
                    resource_delta=_resource_delta(
                        solve_request,
                        used=float(outcome.atomic_steps_used),
                        exhausted=(outcome.kind is CardinalKernelKindV3.RESOURCE_LIMIT),
                    ),
                    label=label,
                    exact_bound_value=outcome.bounds,
                    additional_exact_bounds=(
                        additional_exact_bounds if index == 0 else ()
                    ),
                )
                for index, (label, outcome) in enumerate(outcomes)
            ),
            key=canonical_json_bytes,
        )
    )
    if not include_finite_refinement_miss:
        return kernel_evaluations
    if (
        row.leaf_disposition is not upright.UprightSE2ProofLeafDisposition.UNRESOLVED
        or row.complete_domain_empty is not False
    ):
        raise ValueError("finite refinement miss must close one unresolved leaf")
    finite_miss = build_upright_se2_retained_owner_evaluation(
        compiled_cell=row.compiled_cell,
        owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        outcome_kind=upright.UprightSE2RetainedOwnerOutcomeKind.FINITE_MISS,
        raw_proof_rows=(
            "proof:spatialcf/upright-se2/finite-refinement-miss/incomplete",
        ),
        raw_findings=("INCOMPLETE",),
        atomic_steps=0,
        resource_delta=_resource_delta(
            solve_request,
            used=0.0,
            exhausted=False,
        ),
        label="finite-refinement-miss",
    )
    return tuple(sorted((*kernel_evaluations, finite_miss), key=canonical_json_bytes))


def _objective_from_box_bounds(
    bounds: object,
) -> upright.UprightSE2ProposalPointObjective:
    semantic_terms = bounds.common_cell_semantic_objective_terms
    weighted_terms = bounds.common_cell_objective_terms
    total_lower = sum(
        (lower for _term_id, lower, _upper in weighted_terms), start=Fraction(0)
    )
    total_upper = sum(
        (upper for _term_id, _lower, upper in weighted_terms), start=Fraction(0)
    )
    return upright.UprightSE2ProposalPointObjective(
        terms=tuple(
            upright.UprightSE2ProposalPointTerm(
                term_id=term_id,
                lower=upright.UprightSE2ExactRational(
                    numerator=lower.numerator, denominator=lower.denominator
                ),
                upper=upright.UprightSE2ExactRational(
                    numerator=upper.numerator, denominator=upper.denominator
                ),
            )
            for term_id, lower, upper in semantic_terms
        ),
        total_lower=upright.UprightSE2ExactRational(
            numerator=total_lower.numerator,
            denominator=total_lower.denominator,
        ),
        total_upper=upright.UprightSE2ExactRational(
            numerator=total_upper.numerator,
            denominator=total_upper.denominator,
        ),
    )


def _resource_delta(
    solve_request: CounterfactualSolveRequest,
    *,
    used: float,
    exhausted: bool,
) -> ResourceUsage:
    limits = solve_request.resource_policy.limits
    if len(limits) != 1:
        raise ValueError("upright se2 requires one exact resource limit")
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": (
                solve_request.resource_policy.shared_ledger_policy_ref
            ),
            "entries": (
                {
                    "resource_definition_ref": limits[0].definition_ref,
                    "used": used,
                },
            ),
            "exhausted": exhausted,
        }
    )


def _check_terminal_evidence(
    submission: BackendSubmission,
    proof: upright.UprightSE2ProofMaterial,
    compilation: upright.UprightSE2Compilation,
) -> None:
    del compilation
    leaves = tuple(
        row for row in proof.evaluated_cells if row.leaf_disposition is not None
    )
    if type(submission) is BackendCompleteUnsatEvidence:
        if (
            submission.authorized_domain_sha256
            != proof.compilation.endpoint_construction_recipe.translation_domain_sha256
            or submission.complete_domain_coverage_artifact_sha256
            != proof.coverage_artifact.coverage_artifact_sha256
            or not leaves
            or any(
                row.leaf_disposition
                is not upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
                or row.complete_domain_empty is not True
                for row in leaves
            )
        ):
            raise ValueError("complete UNSAT evidence lacks full outward coverage")
    elif type(submission) is BackendProposalSubmission:
        if not proof.proposal_candidates:
            raise ValueError("proposal submission lacks a checked candidate")
    elif (
        type(submission) is BackendUnknownEvidence
        and submission.resource_usage != proof.total_resource_usage
    ):
        raise ValueError("unknown evidence resource usage does not match proof")


def _check_selected_unknown_reason(
    *,
    solve_request: CounterfactualSolveRequest,
    submission: BackendSubmission,
    proof: (
        upright.UprightSE2ProofMaterial | upright.UprightSE2ContinuousProofMaterial
    ),
) -> None:
    """Bind selected UNKNOWN evidence to freshly verified owner outcomes only."""

    if type(submission) is not BackendUnknownEvidence:
        return
    outcome_kinds = {
        evaluation.outcome_kind
        for row in proof.evaluated_cells
        for evaluation in row.owner_evaluations
    }
    if upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT in outcome_kinds:
        expected_reason_claim_definition_ref = (
            solve_request.resource_policy.exhaustion_claim_ref
        )
    elif upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP in outcome_kinds:
        expected_reason_claim_definition_ref = (
            _UNKNOWN_NUMERIC_REASON_CLAIM_DEFINITION_REF
        )
    elif upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED in outcome_kinds:
        expected_reason_claim_definition_ref = (
            _UNKNOWN_UNSUPPORTED_REASON_CLAIM_DEFINITION_REF
        )
    elif (
        upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE in outcome_kinds
        or upright.UprightSE2RetainedOwnerOutcomeKind.FINITE_MISS in outcome_kinds
        or proof.unresolved_frontier
        or any(
            row.leaf_disposition is upright.UprightSE2ProofLeafDisposition.UNRESOLVED
            for row in proof.evaluated_cells
        )
    ):
        expected_reason_claim_definition_ref = (
            _UNKNOWN_INCOMPLETE_REASON_CLAIM_DEFINITION_REF
        )
    else:
        raise ValueError("unknown evidence lacks fresh owner outcome support")
    if submission.reason_claim_definition_ref != expected_reason_claim_definition_ref:
        raise ValueError("unknown evidence reason does not bind fresh owner outcomes")


def _submission_checked_claim(
    submission: BackendSubmission,
    proof: upright.UprightSE2ProofMaterial,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
    evidence_claim_ref: str,
    fresh_feasible_leaf_objectives: tuple[
        upright.UprightSE2ProposalPointObjective, ...
    ],
    fresh_candidate_objectives: tuple[upright.UprightSE2ProposalPointObjective, ...],
) -> str | None:
    """Classify only complete freshly replayed cardinal evidence."""

    if type(submission) is BackendCompleteUnsatEvidence:
        return evidence_claim_ref
    if type(submission) is not BackendProposalSubmission:
        return None
    leaves = tuple(row for row in proof.evaluated_cells if row.leaf_disposition)
    if (
        proof.unresolved_frontier
        or proof.prune_decisions
        or not leaves
        or not fresh_feasible_leaf_objectives
        or len(fresh_candidate_objectives) != len(proof.proposal_candidates)
    ):
        return None
    return _classify_complete_candidate_claim(
        fresh_feasible_leaf_objectives=fresh_feasible_leaf_objectives,
        fresh_candidate_objectives=fresh_candidate_objectives,
        checker_policy=checker_policy,
    )


def _classify_complete_candidate_claim(
    *,
    fresh_feasible_leaf_objectives: tuple[
        upright.UprightSE2ProposalPointObjective, ...
    ],
    fresh_candidate_objectives: tuple[upright.UprightSE2ProposalPointObjective, ...],
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
) -> str | None:
    """Apply the request policy to all fresh leaf and selected-point totals."""

    if not fresh_feasible_leaf_objectives or not fresh_candidate_objectives:
        return None
    global_lower = min(
        objective.total_lower.as_fraction
        for objective in fresh_feasible_leaf_objectives
    )
    totals = tuple(
        (objective.total_lower.as_fraction, objective.total_upper.as_fraction)
        for objective in (
            *fresh_feasible_leaf_objectives,
            *fresh_candidate_objectives,
        )
    )
    if all(lower == upper for lower, upper in totals) and len(set(totals)) == 1:
        return checker_policy.exact_global_claim_definition_ref
    selected = fresh_candidate_objectives[0]
    if (
        selected.total_upper.as_fraction - global_lower
        <= checker_policy.requested_gap.as_fraction
    ):
        return checker_policy.finite_gap_claim_definition_ref
    return None


def _check_selected_proposal_bounds(
    submission: BackendSubmission,
    proof: upright.UprightSE2ProofMaterial,
    fresh_candidate_objectives: tuple[upright.UprightSE2ProposalPointObjective, ...],
) -> None:
    """Bind outer proposal floats to the canonical first fresh candidate only."""

    if type(submission) is not BackendProposalSubmission:
        return
    if not proof.proposal_candidates or not fresh_candidate_objectives:
        raise ValueError("proposal submission lacks a fresh selected candidate")
    selected = proof.proposal_candidates[0]
    fresh_selected = fresh_candidate_objectives[0]
    proposal = submission.proposal
    if (
        proposal.program_sha256 != selected.program.program_sha256
        or proposal.after_scene_state_sha256
        != selected.program.after_scene_state_sha256
    ):
        raise ValueError("outer proposal does not bind the selected candidate program")
    if proposal.objective_lower_bound != float(
        fresh_selected.total_lower.as_fraction
    ) or proposal.objective_upper_bound != float(
        fresh_selected.total_upper.as_fraction
    ):
        raise ValueError(
            "outer proposal objective bounds do not bind fresh selected candidate"
        )


def _classify_interval_claim(
    *,
    witness: upright.UprightSE2ProposalPointObjective,
    global_lower: Fraction,
    checker_policy: upright.UprightSE2CheckerReplayPolicy,
) -> str | None:
    """Classify one freshly replayed objective interval without inventing a tie."""

    witness_lower = witness.total_lower.as_fraction
    witness_upper = witness.total_upper.as_fraction
    if witness_lower == witness_upper and witness_lower == global_lower:
        return checker_policy.exact_global_claim_definition_ref
    if witness_upper - global_lower <= checker_policy.requested_gap.as_fraction:
        return checker_policy.finite_gap_claim_definition_ref
    return None


def _check_cardinal_roster_and_coverage(
    compilation: upright.UprightSE2Compilation,
    proof: upright.UprightSE2ProofMaterial,
) -> None:
    """Bind the decoded proof's sole cardinal root and full XY leaf roster."""

    if len(proof.cardinal_tuple_roster) != 1:
        raise ValueError("checker requires one request-authorized cardinal tuple")
    proof_tuple = proof.cardinal_tuple_roster[0]
    if (
        canonical_json_bytes(proof_tuple.compiled_cells)
        != canonical_json_bytes(compilation.compiled_cells)
        or proof_tuple.translation_domain
        != compilation.endpoint_construction_recipe.translation_domain
        or proof.coverage_artifact.authorization_sha256
        != compilation.operation.authorization_sha256
    ):
        raise ValueError("checker cardinal roster or translation coverage mismatch")
    leaves = tuple(
        row.compiled_cell
        for row in proof.evaluated_cells
        if row.leaf_disposition is not None
    )
    if canonical_json_bytes(leaves) != canonical_json_bytes(
        proof.coverage_artifact.cells
    ):
        raise ValueError("checker final leaf roster does not match coverage artifact")


def _checked_fact_refs(
    proof: upright.UprightSE2ProofMaterial,
    compilation: upright.UprightSE2Compilation,
) -> tuple[dict[str, str], ...]:
    values = (
        proof.proof_material_sha256,
        compilation.upright_se2_compilation_sha256,
        proof.coverage_artifact.coverage_artifact_sha256,
        proof.resource_ledger.proof_resource_ledger_sha256,
    )
    return tuple(
        sorted(
            (
                {
                    "artifact_schema_ref": _CHECKED_FACT_SCHEMA_REF,
                    "artifact_sha256": canonical_sha256(
                        value,
                        domain="spatialcf/counterfactual/upright-se2/checked-fact/3.0",
                    ),
                }
                for value in values
            ),
            key=canonical_json_bytes,
        )
    )


def _proof_kind(
    kind: CardinalKernelKindV3,
) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    return {
        CardinalKernelKindV3.EXACT: upright.UprightSE2RetainedOwnerOutcomeKind.EXACT,
        CardinalKernelKindV3.NUMERIC_GAP: upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP,
        CardinalKernelKindV3.RESOURCE_LIMIT: upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT,
        CardinalKernelKindV3.UNSUPPORTED: upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED,
    }[kind]


def _is_kernel_owner_evaluation(
    evaluation: upright.UprightSE2RetainedOwnerEvaluation,
) -> bool:
    return any(
        "fixed-cardinal-cell" in row or "fixed-cardinal-visibility" in row
        for row in evaluation.proof_rows
    )


def _classify_leaf(box: object, visibility: tuple[object, ...]):
    if any(
        outcome.kind is not CardinalKernelKindV3.EXACT for outcome in (box, *visibility)
    ):
        return upright.UprightSE2ProofLeafDisposition.UNRESOLVED
    if (
        box.bounds.outer_hard_constraint_slack < 0
        or box.bounds.relation_outer_failure
        or any(outcome.bounds.outer_failure for outcome in visibility)
    ):
        return upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
    if (
        box.bounds.inner_hard_constraint_slack >= 0
        and box.bounds.relation_inner_success
        and all(outcome.bounds.inner_success for outcome in visibility)
    ):
        return upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
    return None


def _fresh_leaf_matches_submission(*, submitted: object, fresh: object) -> bool:
    """Bind terminal leaves, reserving ``UNRESOLVED`` for exact ambiguity."""

    return submitted is fresh or (
        submitted is upright.UprightSE2ProofLeafDisposition.UNRESOLVED and fresh is None
    )


def _require_exact(value: object, model_type: type, label: str) -> None:
    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    checked = model_type.model_validate(
        value.model_dump(mode="python", round_trip=True), strict=True
    )
    if canonical_json_bytes(checked) != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical bytes")
