"""Fresh source and exact arrangement replay; no backend or terminal authority."""
from __future__ import annotations

from spatialcf.core.backends import match_backend_capabilities
from spatialcf.core._internal.kernels import semantic_place as kernel
from spatialcf.core.semantic_place_compiler import (
    BACKEND_REF, BUILDS, OWNERS, SemanticPlaceCompiledProblem, UnsupportedPlacement,
    _compile_validated, _materialize_validated, validate_semantic_place_request,
)
from spatialcf.domain import semantic_place as place
from spatialcf.domain.outcomes import (
    BackendSelectionRecord, CapabilityMatch, ResourceUsage, ProofMaterialEnvelope,
    BackendProposal, BackendProposalSubmission, BackendCompleteUnsatEvidence, BackendUnknownEvidence,
    CheckedProofOutcome, CheckerDisposition,
)
from spatialcf.domain.serialization import canonical_sha256

__all__=("verify_semantic_place_submission",)
d=place.definition
c=place.capability


def _require_exact(value,cls):
    if type(value) is not cls:
        raise TypeError('wrong semantic_place record type')
    cls.model_validate(value.model_dump(mode='python'),strict=True)


def _resource(request,proof=None,*,allocation=False):
    """Reconstruct declared counters independently of producer bookkeeping."""
    entries={v.definition_ref:(v.finite_limit if allocation else 0.0) for v in request.resource_policy.limits}
    if proof is not None:
        ledgers=[proof.failure_ledger,proof.transition_ledger]
        if proof.compilation:
            ledgers.append(proof.compilation.ledger)
            entries[d('resource/max_targets')]=float(len(proof.compilation.targets))
        if proof.coverage:
            ledgers.append(proof.coverage.ledger)
            entries[d('resource/max_strata')]=float(len(proof.coverage.strata))
        for ledger in ledgers:
            if ledger:
                key=d('resource/'+ledger.stage.lower()+'_operations')
                entries[key]=max(entries[key],float(ledger.operations))
                entries[d('resource/numeric_bits')]=max(entries[d('resource/numeric_bits')],float(ledger.peak_numeric_bits))
                if ledger.stage=='COMPILE':
                    completed_targets=sum(item.startswith('target:') for item in ledger.completed_items)
                    entries[d('resource/max_targets')]=max(entries[d('resource/max_targets')],float(completed_targets))
    return ResourceUsage.model_validate(dict(accounting_claim_definition_ref=d('accounting'),
        entries=tuple(dict(resource_definition_ref=k,used=v) for k,v in sorted(entries.items())),
        exhausted=bool(proof and proof.failure_ledger and proof.failure_ledger.reason=='RESOURCE_LIMIT')),strict=True)


def validate_semantic_place_selection(context,selection):
    """Replay static routing without invoking or importing the proposal backend."""
    _require_exact(selection,BackendSelectionRecord)
    request=context.request
    args=context.registry_arguments
    descriptor=request.backend_descriptor_bundle.backend_descriptors[0]
    row=match_backend_capabilities(request,args['action_space_profile'],args['semantics_profile'],args['predicate_definitions'],args['operator_definitions'],descriptor)
    selected=isinstance(row,CapabilityMatch)
    expected=BackendSelectionRecord.seal(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,
        implementation_registry_snapshot_sha256=request.implementation_registry_snapshot.implementation_registry_snapshot_sha256,
        backend_descriptor_bundle_sha256=request.backend_descriptor_bundle.backend_descriptor_bundle_sha256,
        backend_routing_policy_sha256=request.backend_routing_policy.backend_routing_policy_sha256,
        ordered_candidate_backend_refs=(BACKEND_REF,),capability_rows=(row,),selection_disposition='SELECTED' if selected else 'NO_SELECTION',
        selection_disposition_claim_ref=d('selection-disposition'),selected_backend_ref=BACKEND_REF if selected else None,
        selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256 if selected else None,
        resource_allocation=_resource(request,allocation=True) if selected else None,deterministic_selection_reason_ref=d('selection-reason'))
    if selection!=expected:
        raise ValueError('routing does not replay frozen source capabilities')


def decode_semantic_place_proof(submission):
    if type(submission) not in (BackendProposalSubmission,BackendCompleteUnsatEvidence,BackendUnknownEvidence):
        raise TypeError('expected exact submission-v2 record')
    _require_exact(submission,type(submission))
    evidence=submission.proposal if type(submission) is BackendProposalSubmission else submission
    envelope=evidence.proof_material
    if envelope.proof_material_definition_ref!=d('proof-material') or envelope.payload_schema_ref!=place.schema('proof-material') or len(envelope.typed_payload)!=1:
        raise ValueError('wrong profile or proof schema')
    proof=place.decode_proof(envelope.typed_payload[0])
    return evidence,proof


def verify_semantic_place_submission(*,solve_request,selection,compilation,submission):
    """Rebuild every source-bound claim; return only a checked proof outcome.

    Invalid evidence raises explicit validation errors. A separately allocated
    CHECK limit returns LIMITED without validating the stronger submitted claim.
    The assembler must reject that stronger submission, never relabel its DAG.
    """
    context=validate_semantic_place_request(solve_request)
    validate_semantic_place_selection(context,selection)
    if selection.selection_disposition!='SELECTED':
        raise ValueError('proof replay requires SELECTED routing')
    if type(compilation) is not SemanticPlaceCompiledProblem or compilation.source_solve_request!=solve_request:
        raise ValueError('wrong compilation type or source root')
    evidence,submitted=decode_semantic_place_proof(submission)
    from spatialcf.core.registry import _validate_typed_value
    schemas={v.value_schema_ref:v for v in context.registry_arguments['value_schema_definitions']}
    for value in evidence.proof_material.typed_payload:
        if value.value_schema_ref!=evidence.proof_material.payload_schema_ref:
            raise ValueError('proof envelope and typed payload schemas differ')
        _validate_typed_value(value,schemas)
    if (evidence.semantic_problem_sha256,evidence.solve_request_sha256,evidence.backend_selection_record_sha256,
        evidence.proposal_backend_ref,evidence.proposal_backend_owner_ref,evidence.proposal_backend_capability_ref,evidence.proposal_backend_build_sha256)!=(
        solve_request.semantic_problem_sha256,solve_request.solve_request_sha256,selection.backend_selection_record_sha256,
        BACKEND_REF,OWNERS['backend'],c('solve'),BUILDS['backend']):
        raise ValueError('submitted roots, build or owner differ from static authority')
    audit=kernel.Budget(context.limits,'CHECK')
    fresh=None;failure=None;coverage=None;program=None;truths=();transition=None;transport=None
    try:
        try:
            fresh=_compile_validated(context,kernel.Budget(context.limits,'COMPILE',audit=audit))
        except UnsupportedPlacement as error:
            failure=error.ledger
        except kernel.PlacementLimit as error:
            if error.stage=='CHECK':raise
            failure=error.ledger
        if compilation.compilation!=fresh or compilation.failure_ledger!=failure:
            raise ValueError('fresh compilation differs from supplied artifact or failure')
        if fresh is not None:
            budget=kernel.Budget(context.limits,'SOLVE',audit=audit)
            coverage=kernel.solve_targets(fresh.targets,tuple(v.as_fraction for v in fresh.before_xyz),context.limits,budget=budget)
            if coverage.ledger.reason!='NONE':
                failure=coverage.ledger
            elif coverage.winner_xyz is not None:
                try:
                    budget.current='transition:transport'
                    budget.charge()
                    transport=kernel.transport_winner(coverage)
                    if transport is None:
                        raise kernel.PlacementLimit('NUMERIC_GAP',budget.current,'SOLVE')
                    program,_,truths=_materialize_validated(context,fresh,coverage.winner_target_id,transport[0],budget)
                    transition=budget.ledger()
                except kernel.PlacementLimit as error:
                    if error.stage=='CHECK':raise
                    failure=transition=budget.ledger(error)
                    program=None;truths=()
        expected=place.SemanticPlaceProofMaterial.seal(semantic_problem_sha256=solve_request.semantic_problem_sha256,solve_request_sha256=solve_request.solve_request_sha256,
            compilation=fresh,coverage=coverage,program=program,containment=truths,transition_ledger=transition,failure_ledger=failure)
        if submitted!=expected:
            raise ValueError('proof fails fresh source, coverage, minimum, transition or reason replay')
        envelope=ProofMaterialEnvelope.seal(semantic_problem_sha256=solve_request.semantic_problem_sha256,solve_request_sha256=solve_request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,proposal_backend_ref=BACKEND_REF,
            proof_material_definition_ref=d('proof-material'),payload_schema_ref=place.schema('proof-material'),typed_payload=(place.encode_proof(expected),),artifact_refs=())
        common=dict(semantic_problem_sha256=solve_request.semantic_problem_sha256,solve_request_sha256=solve_request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,proposal_backend_ref=BACKEND_REF,proposal_backend_owner_ref=OWNERS['backend'],
            proposal_backend_capability_ref=c('solve'),proposal_backend_build_sha256=BUILDS['backend'],proof_material=envelope,proof_material_sha256=envelope.proof_material_sha256,
            resource_usage=_resource(solve_request,expected))
        if failure is not None:
            claim=d({'RESOURCE_LIMIT':'resource-exhausted','NUMERIC_GAP':'numeric-gap','UNSUPPORTED':'unsupported'}[failure.reason])
            expected_submission=BackendUnknownEvidence.seal(**common,reason_claim_definition_ref=claim)
            disposition=CheckerDisposition.LIMITED
        elif program is not None:
            claim=d('certified')
            proposal=BackendProposal.seal(**common,proposal_claim_definition_ref=claim,program_sha256=program.program_sha256,
                after_scene_state_sha256=program.after_scene_state_sha256,objective_lower_bound=transport[1],objective_upper_bound=transport[2])
            expected_submission=BackendProposalSubmission.seal(proposal=proposal,backend_proposal_sha256=proposal.backend_proposal_sha256)
            disposition=CheckerDisposition.ACCEPTED
        else:
            claim=d('unsat')
            expected_submission=BackendCompleteUnsatEvidence.seal(**common,complete_domain_claim_definition_ref=claim,authorized_domain_sha256=fresh.domain.domain_sha256,
                complete_domain_coverage_artifact_sha256=coverage.coverage_sha256)
            disposition=CheckerDisposition.ACCEPTED
        if submission!=expected_submission:
            raise ValueError('submission type, claim, bounds, resource counters or artifacts do not replay')
        audit.completed.append('replay:complete')
        checked_ledger=audit.ledger()
    except kernel.PlacementLimit as error:
        if error.stage!='CHECK':raise
        claim=d('resource-exhausted')
        disposition=CheckerDisposition.LIMITED
        checked_ledger=audit.ledger(error)
    facts=((place.schema('checked-ledger'),canonical_sha256(checked_ledger,domain=place.HASH_PREFIX+'/checked-ledger')),)
    if checked_ledger.reason=='NONE':
        facts+=((place.schema('checked-proof'),submitted.semantic_place_proof_sha256),)
    return CheckedProofOutcome.seal(semantic_problem_sha256=solve_request.semantic_problem_sha256,solve_request_sha256=solve_request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,proof_material_sha256=evidence.proof_material_sha256,
        checker_capability_ref=c('check'),checker_build_sha256=BUILDS['checker'],checker_disposition=disposition,checked_claim_definition_ref=claim,
        checked_fact_refs=tuple(dict(artifact_schema_ref=schema,artifact_sha256=sha) for schema,sha in sorted(facts)))
