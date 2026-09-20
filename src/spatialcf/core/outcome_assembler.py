"""The sole static M3 checker-dispatch, certificate, and terminal assembler."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcf.core.registry import validate_submission_contract_structure
from spatialcf.core.upright_se2_verification import verify_upright_se2_submission
from spatialcf.domain import upright_se2 as upright
from spatialcf.domain.counterfactual import CounterfactualSolveRequest, EditProgram
from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendSubmission,
    BackendUnknownEvidence,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    CheckedProofOutcome,
    CheckerDisposition,
    CounterfactualCertificate,
    CounterfactualSolveResult,
    NoncertifiedWitnessResult,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    ResourceUsage,
    UnknownResult,
    VerifierDispatchRecord,
)
from spatialcf.domain.predicates import GroundedObligationSet

__all__ = (
    "AssembledCounterfactualOutcome",
    "assemble_counterfactual_outcome",
    "assemble_no_selection_unknown",
)

_COMPLETE_DOMAIN_CLAIM_REF = "definition:spatialcf/upright-se2/complete-domain/1.0"
_SOUND_COMPLETE_DOMAIN_CLAIM_REF = (
    "definition:spatialcf/upright-se2/sound-complete-domain/1.0"
)
_UNKNOWN_CLAIM_REF = "definition:spatialcf/upright-se2/claim-unknown/1.0"


@dataclass(frozen=True, slots=True)
class AssembledCounterfactualOutcome:
    """One assembled result together with the checker records it binds."""

    checked_proof_outcome: CheckedProofOutcome | None
    verifier_dispatch_record: VerifierDispatchRecord | None
    certificate: CounterfactualCertificate | None
    result: CounterfactualSolveResult
    program: EditProgram | None
    grounded_obligations: GroundedObligationSet | None


def assemble_counterfactual_outcome(
    *,
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    submission: BackendSubmission,
) -> AssembledCounterfactualOutcome:
    """Dispatch fresh M3 checking, then assemble exactly one terminal envelope."""

    checked = verify_upright_se2_submission(
        solve_request=solve_request,
        selection=selection,
        compilation=compilation,
        submission=submission,
        checker_policy=upright.build_upright_se2_checker_replay_policy(
            solve_request.proof_policy,
            solve_request.solve_policy_definition_bundle,
        ),
    )
    evidence, proof, _claim_ref = _submission_parts(submission)
    dispatch = VerifierDispatchRecord.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        semantic_definition_bundle_sha256=(
            solve_request.semantic_problem.definition_bundle.definition_bundle_sha256
        ),
        solve_policy_definition_bundle_sha256=(
            solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        ),
        proof_policy_sha256=solve_request.proof_policy.proof_policy_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_owner_ref=evidence.proposal_backend_owner_ref,
        proposal_backend_capability_ref=evidence.proposal_backend_capability_ref,
        proposal_backend_build_sha256=evidence.proposal_backend_build_sha256,
        proof_material_definition_ref=evidence.proof_material.proof_material_definition_ref,
        checker_owner_ref=upright.UPRIGHT_SE2_CHECKER_OWNER_REF,
        checker_capability_ref=checked.checker_capability_ref,
        checker_build_sha256=checked.checker_build_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
    )
    if type(submission) is BackendProposalSubmission:
        if checked.checker_disposition is not CheckerDisposition.ACCEPTED:
            return _validated(
                submission,
                selection,
                _assemble_limited_witness(
                    solve_request,
                    selection,
                    compilation,
                    submission,
                    checked,
                    dispatch,
                    proof,
                ),
            )
        return _validated(
            submission,
            selection,
            _assemble_solution(
                solve_request,
                selection,
                compilation,
                submission,
                checked,
                dispatch,
                proof,
            ),
        )
    if type(submission) is BackendCompleteUnsatEvidence:
        return _validated(
            submission,
            selection,
            _assemble_unsat(
                solve_request, selection, compilation, submission, checked, dispatch
            ),
        )
    if type(submission) is BackendUnknownEvidence:
        result = UnknownResult.seal(
            semantic_problem_sha256=solve_request.semantic_problem_sha256,
            solve_request_sha256=solve_request.solve_request_sha256,
            claim_definition_ref=_UNKNOWN_CLAIM_REF,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,
            checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
            verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
            checker_disposition=checked.checker_disposition,
            resource_usage=submission.resource_usage,
            reason_claim_definition_ref=submission.reason_claim_definition_ref,
            partial_artifact_refs=submission.partial_artifact_refs,
        )
        return _validated(
            submission,
            selection,
            AssembledCounterfactualOutcome(checked, dispatch, None, result, None, None),
        )
    raise TypeError("assembler accepts only M3 V2 backend submissions")


def assemble_no_selection_unknown(
    *,
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
) -> AssembledCounterfactualOutcome:
    """Assemble the only pre-dispatch terminal without inventing a proof."""

    if selection.selection_disposition != "NO_SELECTION":
        raise ValueError("no-selection assembler requires NO_SELECTION routing")
    if (
        selection.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or selection.solve_request_sha256 != solve_request.solve_request_sha256
    ):
        raise ValueError("no-selection roots do not match solve request")
    usage = _zero_resource_usage(solve_request)
    result = UnknownResult.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        claim_definition_ref=_UNKNOWN_CLAIM_REF,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        resource_usage=usage,
        reason_claim_definition_ref=_UNKNOWN_CLAIM_REF,
    )
    return AssembledCounterfactualOutcome(None, None, None, result, None, None)


def _assemble_solution(
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    submission: BackendProposalSubmission,
    checked: CheckedProofOutcome,
    dispatch: VerifierDispatchRecord,
    proof: upright.UprightSE2ProofMaterial | upright.UprightSE2ContinuousProofMaterial,
) -> AssembledCounterfactualOutcome:
    candidate = proof.proposal_candidates[0]
    program = candidate.program
    if (
        submission.proposal.program_sha256 != program.program_sha256
        or submission.proposal.after_scene_state_sha256
        != program.after_scene_state_sha256
    ):
        raise ValueError("checked proposal does not bind selected program")
    certificate = CertifiedSolutionCertificate.seal(
        **_certificate_roots(
            solve_request,
            selection,
            compilation,
            submission.proposal,
            checked,
            dispatch,
        ),
        certificate_kind="CERTIFIED_SOLUTION",
        claim_definition_ref=checked.checked_claim_definition_ref,
        program_sha256=program.program_sha256,
        after_scene_state_sha256=program.after_scene_state_sha256,
        state_delta_manifest_sha256=program.state_delta_manifest.state_delta_manifest_sha256,
        grounded_obligation_set_sha256=(
            compilation.grounded_obligations.grounded_obligation_set_sha256
        ),
    )
    result = CertifiedSolutionResult.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        claim_definition_ref=certificate.claim_definition_ref,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        checker_disposition=checked.checker_disposition,
        resource_usage=submission.proposal.resource_usage,
        accepted_certificate=certificate,
        certificate_sha256=certificate.certificate_sha256,
        program_sha256=program.program_sha256,
        after_scene_state_sha256=program.after_scene_state_sha256,
    )
    return AssembledCounterfactualOutcome(
        checked,
        dispatch,
        certificate,
        result,
        program,
        compilation.grounded_obligations,
    )


def _assemble_limited_witness(
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    submission: BackendProposalSubmission,
    checked: CheckedProofOutcome,
    dispatch: VerifierDispatchRecord,
    proof: upright.UprightSE2ProofMaterial | upright.UprightSE2ContinuousProofMaterial,
) -> AssembledCounterfactualOutcome:
    """Keep a freshly feasible non-global proposal explicitly noncertified."""

    candidate = proof.proposal_candidates[0]
    program = candidate.program
    result = NoncertifiedWitnessResult.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        claim_definition_ref=_UNKNOWN_CLAIM_REF,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        checker_disposition=checked.checker_disposition,
        resource_usage=submission.proposal.resource_usage,
        evidence_claim_definition_ref=submission.proposal.proposal_claim_definition_ref,
        program_sha256=program.program_sha256,
        after_scene_state_sha256=program.after_scene_state_sha256,
    )
    return AssembledCounterfactualOutcome(
        checked, dispatch, None, result, program, compilation.grounded_obligations
    )


def _validated(
    submission: BackendSubmission,
    selection: BackendSelectionRecord,
    assembled: AssembledCounterfactualOutcome,
) -> AssembledCounterfactualOutcome:
    """Invoke the registry-owned additive structural closure after assembly."""

    assert assembled.checked_proof_outcome is not None
    assert assembled.verifier_dispatch_record is not None
    validate_submission_contract_structure(
        submission=submission,
        selection=selection,
        checked_proof_outcome=assembled.checked_proof_outcome,
        verifier_dispatch_record=assembled.verifier_dispatch_record,
        result=assembled.result,
    )
    return assembled


def _assemble_unsat(
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    submission: BackendCompleteUnsatEvidence,
    checked: CheckedProofOutcome,
    dispatch: VerifierDispatchRecord,
) -> AssembledCounterfactualOutcome:
    certificate = ProvenUnsatCertificate.seal(
        **_certificate_roots(
            solve_request, selection, compilation, submission, checked, dispatch
        ),
        certificate_kind="PROVEN_UNSAT",
        claim_definition_ref=submission.complete_domain_claim_definition_ref,
        authorized_domain_sha256=submission.authorized_domain_sha256,
        complete_domain_coverage_artifact_sha256=(
            submission.complete_domain_coverage_artifact_sha256
        ),
        complete_domain_claim_definition_ref=_COMPLETE_DOMAIN_CLAIM_REF,
        sound_complete_domain_claim_definition_ref=_SOUND_COMPLETE_DOMAIN_CLAIM_REF,
    )
    result = ProvenUnsatResult.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        claim_definition_ref=certificate.claim_definition_ref,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        checker_disposition=checked.checker_disposition,
        resource_usage=submission.resource_usage,
        accepted_certificate=certificate,
        certificate_sha256=certificate.certificate_sha256,
        complete_domain_coverage_artifact_sha256=(
            submission.complete_domain_coverage_artifact_sha256
        ),
    )
    return AssembledCounterfactualOutcome(
        checked, dispatch, certificate, result, None, None
    )


def _certificate_roots(
    solve_request: CounterfactualSolveRequest,
    selection: BackendSelectionRecord,
    compilation: upright.UprightSE2Compilation
    | upright.UprightSE2ContinuousCompilation,
    evidence,
    checked: CheckedProofOutcome,
    dispatch: VerifierDispatchRecord,
) -> dict[str, object]:
    return {
        "semantic_problem_sha256": solve_request.semantic_problem_sha256,
        "solve_request_sha256": solve_request.solve_request_sha256,
        "scene_state_sha256": solve_request.semantic_problem.scene_state.scene_state_sha256,
        "backend_selection_record_sha256": selection.backend_selection_record_sha256,
        "checked_proof_outcome_sha256": checked.checked_proof_outcome_sha256,
        "verifier_dispatch_record_sha256": dispatch.verifier_dispatch_record_sha256,
        "semantic_definition_bundle_sha256": (
            solve_request.semantic_problem.definition_bundle.definition_bundle_sha256
        ),
        "solve_policy_definition_bundle_sha256": (
            solve_request.solve_policy_definition_bundle.definition_bundle_sha256
        ),
        "semantics_profile_sha256": (
            compilation.semantic_closure.profile_registration.semantics_profile.semantics_profile_sha256
        ),
        "action_space_profile_sha256": (
            compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
        ),
        "intervention_authorization_sha256": (
            solve_request.semantic_problem.intervention_authorization.intervention_authorization_sha256
        ),
        "objective_expression_sha256": (
            solve_request.semantic_problem.objective_expression.objective_expression_sha256
        ),
        "proof_policy_sha256": solve_request.proof_policy.proof_policy_sha256,
        "resource_policy_sha256": solve_request.resource_policy.resource_policy_sha256,
        "backend_routing_policy_sha256": (
            solve_request.backend_routing_policy.backend_routing_policy_sha256
        ),
        "solver_config_sha256": solve_request.solver_config.solver_config_sha256,
        "implementation_registry_snapshot_sha256": (
            solve_request.implementation_registry_snapshot.implementation_registry_snapshot_sha256
        ),
        "backend_descriptor_bundle_sha256": (
            solve_request.backend_descriptor_bundle.backend_descriptor_bundle_sha256
        ),
        "proposal_backend_build_sha256": evidence.proposal_backend_build_sha256,
        "checker_build_sha256": upright.UPRIGHT_SE2_CHECKER_BUILD_SHA256,
        "proof_material_definition_ref": evidence.proof_material.proof_material_definition_ref,
        "proof_material_sha256": evidence.proof_material_sha256,
        "checker_disposition": checked.checker_disposition,
        "resource_usage": evidence.resource_usage,
    }


def _submission_parts(submission: BackendSubmission):
    if type(submission) is BackendProposalSubmission:
        evidence = submission.proposal
    elif type(submission) in (BackendCompleteUnsatEvidence, BackendUnknownEvidence):
        evidence = submission
    else:
        raise TypeError("assembler accepts only exact M3 V2 backend submissions")
    if len(evidence.proof_material.typed_payload) != 1:
        raise ValueError("submission proof payload is not singular")
    if (
        evidence.proof_material.proof_material_definition_ref
        == upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF
    ):
        proof = upright.decode_upright_se2_proof_material(
            evidence.proof_material.typed_payload[0]
        )
    elif (
        evidence.proof_material.proof_material_definition_ref
        == upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
    ):
        proof = upright.decode_upright_se2_continuous_proof_material(
            evidence.proof_material.typed_payload[0]
        )
    else:
        raise ValueError("submission proof payload is not an installed M3 definition")
    claim_ref = (
        evidence.proposal_claim_definition_ref
        if type(submission) is BackendProposalSubmission
        else (
            evidence.complete_domain_claim_definition_ref
            if type(submission) is BackendCompleteUnsatEvidence
            else evidence.reason_claim_definition_ref
        )
    )
    return evidence, proof, claim_ref


def _zero_resource_usage(solve_request: CounterfactualSolveRequest) -> ResourceUsage:
    limits = solve_request.resource_policy.limits
    if len(limits) != 1:
        raise ValueError("M3 no-selection requires one registered resource limit")
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": (
                solve_request.resource_policy.shared_ledger_policy_ref
            ),
            "entries": (
                {"resource_definition_ref": limits[0].definition_ref, "used": 0.0},
            ),
            "exhausted": False,
        }
    )
