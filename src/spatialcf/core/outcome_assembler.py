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

    if type(solve_request) is CounterfactualSolveRequest:
        profile = solve_request.semantic_problem.action_space_profile_ref
        if profile == "spatialcf/rigid_se3_multi@1":
            return _assemble_rigid_se3(solve_request, selection, compilation, submission)
        if profile == "spatialcf/semantic_place@1":
            return _assemble_semantic_place(solve_request, selection, compilation, submission)
        if profile != upright.UPRIGHT_SE2_PROFILE_REF:
            raise ValueError("unsupported assembler action-space profile")
        if type(compilation) not in (upright.UprightSE2Compilation, upright.UprightSE2ContinuousCompilation):
            raise TypeError("M3 assembler requires its exact compiled problem")
        if compilation.source_solve_request != solve_request:
            raise ValueError("M3 compilation source request does not match")
        if type(submission) not in (BackendProposalSubmission, BackendCompleteUnsatEvidence, BackendUnknownEvidence):
            raise TypeError("checker accepts only exact M3 backend submissions")
        evidence, _proof, _claim = _submission_parts(submission)
        proof_type = (upright.UprightSE2ContinuousProofMaterial
                      if type(compilation) is upright.UprightSE2ContinuousCompilation
                      else upright.UprightSE2ProofMaterial)
        if type(_proof) is not proof_type:
            raise ValueError("M3 proof type does not match compilation")
        for record in (selection, evidence, evidence.proof_material):
            if (record.semantic_problem_sha256, record.solve_request_sha256) != (
                solve_request.semantic_problem_sha256, solve_request.solve_request_sha256,
            ):
                raise ValueError("M3 pre-dispatch source roots do not match")
    else:
        raise TypeError("assembler requires an exact solve request")

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

    if type(solve_request) is CounterfactualSolveRequest and getattr(solve_request.semantic_problem, "action_space_profile_ref", None) == "spatialcf/rigid_se3_multi@1":
        return _assemble_rigid_se3_no_selection(solve_request, selection)
    if type(solve_request) is CounterfactualSolveRequest and getattr(solve_request.semantic_problem, "action_space_profile_ref", None) == "spatialcf/semantic_place@1":
        return _assemble_semantic_place_no_selection(solve_request, selection)

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


def _assemble_semantic_place(solve_request, selection, compilation, submission):
    from spatialcf.core.semantic_place_compiler import (
        BUILDS, OWNERS, SemanticPlaceCompiledProblem, grounded_obligations,
        validate_semantic_place_request,
    )
    from spatialcf.core.semantic_place_verification import (
        decode_semantic_place_proof, verify_semantic_place_submission,
    )
    from spatialcf.domain.semantic_place import definition, schema

    if type(compilation) is not SemanticPlaceCompiledProblem:
        raise TypeError("M5 assembler requires its exact compiled problem")
    if compilation.source_solve_request != solve_request:
        raise ValueError("M5 compilation source request does not match")
    # Reject cross-profile proof types before checker dispatch.
    evidence, proof = decode_semantic_place_proof(submission)
    for record in (selection, evidence, evidence.proof_material, proof):
        if (record.semantic_problem_sha256, record.solve_request_sha256) != (
            solve_request.semantic_problem_sha256, solve_request.solve_request_sha256,
        ):
            raise ValueError("M5 pre-dispatch source roots do not match")
    context = validate_semantic_place_request(solve_request)
    checked = verify_semantic_place_submission(
        solve_request=solve_request, selection=selection,
        compilation=compilation, submission=submission,
    )
    if not any(row.artifact_schema_ref == schema("checked-proof") for row in checked.checked_fact_refs):
        raise ValueError("M5 checker replay budget exhausted; submission is not accepted")
    if type(submission) is not BackendUnknownEvidence and checked.checker_disposition is not CheckerDisposition.ACCEPTED:
        raise ValueError("M5 requires an accepted complete proof")
    dispatch = VerifierDispatchRecord.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        semantic_definition_bundle_sha256=solve_request.semantic_problem.definition_bundle.definition_bundle_sha256,
        solve_policy_definition_bundle_sha256=solve_request.solve_policy_definition_bundle.definition_bundle_sha256,
        proof_policy_sha256=solve_request.proof_policy.proof_policy_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_owner_ref=evidence.proposal_backend_owner_ref,
        proposal_backend_capability_ref=evidence.proposal_backend_capability_ref,
        proposal_backend_build_sha256=evidence.proposal_backend_build_sha256,
        proof_material_definition_ref=evidence.proof_material.proof_material_definition_ref,
        checker_owner_ref=OWNERS["checker"], checker_capability_ref=checked.checker_capability_ref,
        checker_build_sha256=BUILDS["checker"], checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
    )
    usage = _semantic_place_checked_resource_usage(proof,evidence,checked)
    result_fields = dict(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        checker_disposition=checked.checker_disposition, resource_usage=usage,
    )
    if type(submission) is BackendUnknownEvidence:
        result = UnknownResult.seal(**result_fields,claim_definition_ref=definition("unknown"),
            reason_claim_definition_ref=submission.reason_claim_definition_ref,partial_artifact_refs=submission.partial_artifact_refs)
        return _validated(submission,selection,AssembledCounterfactualOutcome(checked,dispatch,None,result,None,None))
    roots = _semantic_place_certificate_roots(context,selection,evidence,checked,dispatch,usage)
    if type(submission) is BackendProposalSubmission:
        program = proof.program
        grounded = grounded_obligations(solve_request)
        certificate = CertifiedSolutionCertificate.seal(**roots,claim_definition_ref=definition("certified"),
            program_sha256=program.program_sha256,after_scene_state_sha256=program.after_scene_state_sha256,
            state_delta_manifest_sha256=program.state_delta_manifest.state_delta_manifest_sha256,
            grounded_obligation_set_sha256=grounded.grounded_obligation_set_sha256)
        result = CertifiedSolutionResult.seal(**result_fields,claim_definition_ref=certificate.claim_definition_ref,
            accepted_certificate=certificate,certificate_sha256=certificate.certificate_sha256,
            program_sha256=program.program_sha256,after_scene_state_sha256=program.after_scene_state_sha256)
        return _validated(submission,selection,AssembledCounterfactualOutcome(checked,dispatch,certificate,result,program,grounded))
    certificate = ProvenUnsatCertificate.seal(**roots,claim_definition_ref=definition("unsat"),
        authorized_domain_sha256=submission.authorized_domain_sha256,
        complete_domain_coverage_artifact_sha256=submission.complete_domain_coverage_artifact_sha256,
        complete_domain_claim_definition_ref=definition("complete-domain"),sound_complete_domain_claim_definition_ref=definition("sound-complete-domain"))
    result = ProvenUnsatResult.seal(**result_fields,claim_definition_ref=certificate.claim_definition_ref,
        accepted_certificate=certificate,certificate_sha256=certificate.certificate_sha256,
        complete_domain_coverage_artifact_sha256=submission.complete_domain_coverage_artifact_sha256)
    return _validated(submission,selection,AssembledCounterfactualOutcome(checked,dispatch,certificate,result,None,None))


def _semantic_place_certificate_roots(context,selection,evidence,checked,dispatch,usage):
    request = context.request
    problem = request.semantic_problem
    return dict(
        semantic_problem_sha256=request.semantic_problem_sha256, solve_request_sha256=request.solve_request_sha256,
        scene_state_sha256=problem.scene_state.scene_state_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        semantic_definition_bundle_sha256=problem.definition_bundle.definition_bundle_sha256,
        solve_policy_definition_bundle_sha256=request.solve_policy_definition_bundle.definition_bundle_sha256,
        semantics_profile_sha256=context.registry_arguments['semantics_profile'].semantics_profile_sha256,
        action_space_profile_sha256=context.registry_arguments['action_space_profile'].action_space_profile_sha256,
        intervention_authorization_sha256=problem.intervention_authorization.intervention_authorization_sha256,
        objective_expression_sha256=problem.objective_expression.objective_expression_sha256,
        proof_policy_sha256=request.proof_policy.proof_policy_sha256,
        resource_policy_sha256=request.resource_policy.resource_policy_sha256,
        backend_routing_policy_sha256=request.backend_routing_policy.backend_routing_policy_sha256,
        solver_config_sha256=request.solver_config.solver_config_sha256,
        implementation_registry_snapshot_sha256=request.implementation_registry_snapshot.implementation_registry_snapshot_sha256,
        backend_descriptor_bundle_sha256=request.backend_descriptor_bundle.backend_descriptor_bundle_sha256,
        proposal_backend_build_sha256=evidence.proposal_backend_build_sha256,
        checker_build_sha256=checked.checker_build_sha256,
        proof_material_definition_ref=evidence.proof_material.proof_material_definition_ref,
        proof_material_sha256=evidence.proof_material_sha256,
        checker_disposition=checked.checker_disposition,resource_usage=usage,
    )


def _assemble_semantic_place_no_selection(solve_request, selection):
    from spatialcf.core.semantic_place_compiler import validate_semantic_place_request
    from spatialcf.core.semantic_place_verification import validate_semantic_place_selection
    from spatialcf.domain.semantic_place import definition

    context=validate_semantic_place_request(solve_request)
    validate_semantic_place_selection(context,selection)
    if selection.selection_disposition!='NO_SELECTION':
        raise ValueError('M5 no-selection assembler requires NO_SELECTION')
    result=UnknownResult.seal(semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,claim_definition_ref=definition('unknown'),
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        resource_usage=_semantic_place_zero_resource_usage(solve_request),reason_claim_definition_ref=definition('unknown'))
    return AssembledCounterfactualOutcome(None,None,None,result,None,None)


def _semantic_place_zero_resource_usage(solve_request):
    from spatialcf.domain.semantic_place import definition

    return ResourceUsage.model_validate(dict(accounting_claim_definition_ref=definition('accounting'),
        entries=tuple(dict(resource_definition_ref=row.definition_ref,used=0.0) for row in solve_request.resource_policy.limits),exhausted=False),strict=True)


def _semantic_place_checked_resource_usage(proof,evidence,checked):
    from spatialcf.domain.semantic_place import SemanticPlaceLedger, definition, schema, HASH_PREFIX
    from spatialcf.domain.serialization import canonical_sha256

    ledgers=[proof.failure_ledger,proof.transition_ledger]
    if proof.compilation is not None:ledgers.append(proof.compilation.ledger)
    if proof.coverage is not None:ledgers.append(proof.coverage.ledger)
    ledgers=[row for row in ledgers if row is not None]
    operations=sum(max((row.replay_operations for row in ledgers if row.stage==stage),default=0) for stage in ('COMPILE','SOLVE'))
    peak=max((row.peak_numeric_bits for row in ledgers),default=0)
    ledger=SemanticPlaceLedger(stage='CHECK',operations=operations,replay_operations=operations,peak_numeric_bits=peak,
        completed_items=('replay:complete',),first_unprocessed_item=None)
    digest=canonical_sha256(ledger,domain=HASH_PREFIX+'/checked-ledger')
    if not any(row.artifact_schema_ref==schema('checked-ledger') and row.artifact_sha256==digest for row in checked.checked_fact_refs):
        raise ValueError('terminal resource total differs from fresh checker ledger')
    usage=evidence.resource_usage
    entries=tuple(row.model_copy(update={'used':float(operations)}) if row.resource_definition_ref==definition('resource/check_operations') else row for row in usage.entries)
    return ResourceUsage(accounting_claim_definition_ref=usage.accounting_claim_definition_ref,entries=entries,exhausted=usage.exhausted)


def _assemble_rigid_se3(solve_request, selection, compilation, submission):
    """The only M6 checker dispatch and terminal assembly boundary."""
    from spatialcf.core.rigid_se3_compiler import (
        BUILDS, OWNERS, RigidSE3CompiledProblem, _grounded_obligations,
        d, c, validate_rigid_se3_request,
    )
    from spatialcf.core.rigid_se3_verification import (
        _verify_rigid_se3_submission_with_ledger,
    )
    from spatialcf.domain.rigid_se3 import decode_proof_material, schema
    from spatialcf.domain.serialization import canonical_sha256

    if type(compilation) is not RigidSE3CompiledProblem:
        raise TypeError("M6 assembler requires its exact compiled problem")
    if compilation.source_solve_request != solve_request:
        raise ValueError("M6 compilation source request does not match")
    if type(submission) is BackendProposalSubmission:
        evidence = submission.proposal
    elif type(submission) in (BackendCompleteUnsatEvidence, BackendUnknownEvidence):
        evidence = submission
    else:
        raise TypeError("M6 assembler requires an exact V2 backend submission")
    if len(evidence.proof_material.typed_payload) != 1 or evidence.proof_material.proof_material_definition_ref != d("proof-material"):
        raise ValueError("M6 pre-dispatch proof schema does not match")
    proof = decode_proof_material(evidence.proof_material.typed_payload[0])
    roots = (solve_request.semantic_problem_sha256, solve_request.solve_request_sha256)
    for record in (selection, evidence, evidence.proof_material, proof):
        if (record.semantic_problem_sha256, record.solve_request_sha256) != roots:
            raise ValueError("M6 pre-dispatch source roots do not match")
    context = validate_rigid_se3_request(solve_request)
    checked, check_ledger = _verify_rigid_se3_submission_with_ledger(
        solve_request=solve_request, selection=selection,
        compilation=compilation, submission=submission)
    if check_ledger.stage != "CHECK" or check_ledger.reason != "NONE":
        raise ValueError("M6 checker replay budget exhausted; no terminal proof")
    digest = canonical_sha256(
        check_ledger, domain="spatialcf/counterfactual/rigid-se3/checked-ledger/1.0")
    facts = {(row.artifact_schema_ref, row.artifact_sha256)
             for row in checked.checked_fact_refs}
    if ((schema("checked-ledger"), digest) not in facts
            or (schema("checked-proof"), proof.rigid_se3_proof_sha256) not in facts):
        raise ValueError("M6 terminal requires the fresh complete CHECK receipt")
    usage = _rigid_se3_checked_resource_usage(evidence.resource_usage, check_ledger, d)
    dispatch = VerifierDispatchRecord.seal(
        semantic_problem_sha256=roots[0], solve_request_sha256=roots[1],
        semantic_definition_bundle_sha256=solve_request.semantic_problem.definition_bundle.definition_bundle_sha256,
        solve_policy_definition_bundle_sha256=solve_request.solve_policy_definition_bundle.definition_bundle_sha256,
        proof_policy_sha256=solve_request.proof_policy.proof_policy_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_owner_ref=evidence.proposal_backend_owner_ref,
        proposal_backend_capability_ref=evidence.proposal_backend_capability_ref,
        proposal_backend_build_sha256=evidence.proposal_backend_build_sha256,
        proof_material_definition_ref=evidence.proof_material.proof_material_definition_ref,
        checker_owner_ref=OWNERS["checker"],
        checker_capability_ref=c("check"), checker_build_sha256=BUILDS["checker"],
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
    )
    result_fields = dict(
        semantic_problem_sha256=roots[0], solve_request_sha256=roots[1],
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        checker_disposition=checked.checker_disposition, resource_usage=usage,
    )
    if type(submission) is BackendUnknownEvidence:
        if (checked.checker_disposition is not CheckerDisposition.LIMITED
                or checked.checked_claim_definition_ref != submission.reason_claim_definition_ref):
            raise ValueError("M6 unknown reason lacks fresh checker support")
        result = UnknownResult.seal(
            **result_fields, claim_definition_ref=d("unknown"),
            reason_claim_definition_ref=submission.reason_claim_definition_ref,
            partial_artifact_refs=submission.partial_artifact_refs)
        return _validated(submission, selection, AssembledCounterfactualOutcome(
            checked, dispatch, None, result, None, None))
    if type(submission) is BackendProposalSubmission:
        if proof.winner_trace is None:
            raise ValueError("M6 proposal has no freshly proved winner")
        program = proof.winner_trace.program
        if (program.program_sha256 != submission.proposal.program_sha256
                or proof.winner_trace.after_state_sha256 != submission.proposal.after_scene_state_sha256):
            raise ValueError("M6 proposal does not bind checked program and endpoint")
        grounded = _grounded_obligations(solve_request.semantic_problem)
        if checked.checker_disposition is CheckerDisposition.LIMITED:
            if checked.checked_claim_definition_ref != d("certified"):
                raise ValueError("M6 partial proposal has unsupported checked claim")
            result = NoncertifiedWitnessResult.seal(
                **result_fields, claim_definition_ref=d("witness"),
                evidence_claim_definition_ref=submission.proposal.proposal_claim_definition_ref,
                program_sha256=program.program_sha256,
                after_scene_state_sha256=proof.winner_trace.after_state_sha256)
            return _validated(submission, selection, AssembledCounterfactualOutcome(
                checked, dispatch, None, result, program, grounded))
        if (checked.checker_disposition is not CheckerDisposition.ACCEPTED
                or checked.checked_claim_definition_ref != d("certified")
                or d("certified") not in solve_request.proof_policy.accepted_claim_definition_refs):
            raise ValueError("M6 certified proposal fails proof policy")
        certificate = CertifiedSolutionCertificate.seal(
            **_rigid_se3_certificate_roots(context, selection, evidence, checked,
                                           dispatch, usage),
            claim_definition_ref=d("certified"),
            program_sha256=program.program_sha256,
            after_scene_state_sha256=proof.winner_trace.after_state_sha256,
            state_delta_manifest_sha256=program.state_delta_manifest.state_delta_manifest_sha256,
            grounded_obligation_set_sha256=grounded.grounded_obligation_set_sha256)
        result = CertifiedSolutionResult.seal(
            **result_fields, claim_definition_ref=certificate.claim_definition_ref,
            accepted_certificate=certificate,
            certificate_sha256=certificate.certificate_sha256,
            program_sha256=program.program_sha256,
            after_scene_state_sha256=proof.winner_trace.after_state_sha256)
        return _validated(submission, selection, AssembledCounterfactualOutcome(
            checked, dispatch, certificate, result, program, grounded))
    if (checked.checker_disposition is not CheckerDisposition.ACCEPTED
            or checked.checked_claim_definition_ref != d("unsat")
            or d("unsat") not in solve_request.proof_policy.accepted_claim_definition_refs
            or proof.coverage is None or proof.coverage.ledger.reason != "NONE"):
        raise ValueError("M6 UNSAT lacks complete accepted coverage and policy")
    certificate = ProvenUnsatCertificate.seal(
        **_rigid_se3_certificate_roots(context, selection, evidence, checked,
                                       dispatch, usage),
        claim_definition_ref=d("unsat"),
        authorized_domain_sha256=submission.authorized_domain_sha256,
        complete_domain_coverage_artifact_sha256=submission.complete_domain_coverage_artifact_sha256,
        complete_domain_claim_definition_ref=d("complete-domain"),
        sound_complete_domain_claim_definition_ref=d("sound-complete-domain"))
    result = ProvenUnsatResult.seal(
        **result_fields, claim_definition_ref=certificate.claim_definition_ref,
        accepted_certificate=certificate, certificate_sha256=certificate.certificate_sha256,
        complete_domain_coverage_artifact_sha256=submission.complete_domain_coverage_artifact_sha256)
    return _validated(submission, selection, AssembledCounterfactualOutcome(
        checked, dispatch, certificate, result, None, None))


def _rigid_se3_checked_resource_usage(usage, ledger, definition):
    operations = float(sum(row.amount for row in ledger.events))
    values = {row.resource_definition_ref: row.used for row in usage.entries}
    values[definition("resource/check_operations")] = operations
    values[definition("resource/max_exact_operations")] = max(
        values[definition("resource/max_exact_operations")], operations)
    values[definition("resource/numeric_bits")] = max(
        values[definition("resource/numeric_bits")], float(ledger.peak_numeric_bits))
    return ResourceUsage.model_validate(dict(
        accounting_claim_definition_ref=usage.accounting_claim_definition_ref,
        entries=tuple(dict(resource_definition_ref=ref, used=amount)
                      for ref, amount in sorted(values.items())),
        exhausted=usage.exhausted), strict=True)


def _rigid_se3_certificate_roots(context, selection, evidence, checked, dispatch, usage):
    request = context.request
    problem = request.semantic_problem
    return dict(
        semantic_problem_sha256=problem.semantic_problem_sha256,
        solve_request_sha256=request.solve_request_sha256,
        scene_state_sha256=problem.scene_state.scene_state_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        checked_proof_outcome_sha256=checked.checked_proof_outcome_sha256,
        verifier_dispatch_record_sha256=dispatch.verifier_dispatch_record_sha256,
        semantic_definition_bundle_sha256=problem.definition_bundle.definition_bundle_sha256,
        solve_policy_definition_bundle_sha256=request.solve_policy_definition_bundle.definition_bundle_sha256,
        semantics_profile_sha256=context.registry_arguments["semantics_profile"].semantics_profile_sha256,
        action_space_profile_sha256=context.registry_arguments["action_space_profile"].action_space_profile_sha256,
        intervention_authorization_sha256=problem.intervention_authorization.intervention_authorization_sha256,
        objective_expression_sha256=problem.objective_expression.objective_expression_sha256,
        proof_policy_sha256=request.proof_policy.proof_policy_sha256,
        resource_policy_sha256=request.resource_policy.resource_policy_sha256,
        backend_routing_policy_sha256=request.backend_routing_policy.backend_routing_policy_sha256,
        solver_config_sha256=request.solver_config.solver_config_sha256,
        implementation_registry_snapshot_sha256=request.implementation_registry_snapshot.implementation_registry_snapshot_sha256,
        backend_descriptor_bundle_sha256=request.backend_descriptor_bundle.backend_descriptor_bundle_sha256,
        proposal_backend_build_sha256=evidence.proposal_backend_build_sha256,
        checker_build_sha256=checked.checker_build_sha256,
        proof_material_definition_ref=evidence.proof_material.proof_material_definition_ref,
        proof_material_sha256=evidence.proof_material_sha256,
        checker_disposition=checked.checker_disposition, resource_usage=usage)


def _assemble_rigid_se3_no_selection(solve_request, selection):
    from spatialcf.core.rigid_se3_compiler import d, validate_rigid_se3_request
    from spatialcf.core.rigid_se3_verification import _selection

    context = validate_rigid_se3_request(solve_request)
    _selection(context, selection, require_selected=False)
    usage = ResourceUsage.model_validate(dict(
        accounting_claim_definition_ref=d("accounting"),
        entries=tuple(dict(resource_definition_ref=row.definition_ref, used=0.0)
                      for row in solve_request.resource_policy.limits),
        exhausted=False), strict=True)
    result = UnknownResult.seal(
        semantic_problem_sha256=solve_request.semantic_problem_sha256,
        solve_request_sha256=solve_request.solve_request_sha256,
        claim_definition_ref=d("unknown"),
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        resource_usage=usage, reason_claim_definition_ref=d("unknown"))
    return AssembledCounterfactualOutcome(None, None, None, result, None, None)
