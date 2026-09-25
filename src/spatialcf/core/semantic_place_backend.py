"""Untrusted serial submission-v2 producer for semantic_place@1."""
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
)

__all__=("SemanticPlaceBackend",)
d=place.definition
c=place.capability


def _usage(request,proof=None,*,allocation=False):
    values={limit.definition_ref:(limit.finite_limit if allocation else 0.0) for limit in request.resource_policy.limits}
    if proof is not None:
        for ledger in (proof.compilation.ledger if proof.compilation else None,
                       proof.coverage.ledger if proof.coverage else None,proof.transition_ledger,proof.failure_ledger):
            if ledger is not None:
                key=d('resource/'+ledger.stage.lower()+'_operations')
                values[key]=max(values[key],float(ledger.operations))
                values[d('resource/numeric_bits')]=max(values[d('resource/numeric_bits')],float(ledger.peak_numeric_bits))
                if ledger.stage=='COMPILE':
                    values[d('resource/max_targets')]=max(values[d('resource/max_targets')],float(sum(item.startswith('target:') for item in ledger.completed_items)))
        if proof.compilation:
            values[d('resource/max_targets')]=float(len(proof.compilation.targets))
        if proof.coverage:
            values[d('resource/max_strata')]=float(len(proof.coverage.strata))
    return ResourceUsage.model_validate(dict(accounting_claim_definition_ref=d('accounting'),
        entries=tuple(dict(resource_definition_ref=key,used=value) for key,value in sorted(values.items())),
        exhausted=bool(proof and proof.failure_ledger and proof.failure_ledger.reason=='RESOURCE_LIMIT')),strict=True)


class SemanticPlaceBackend:
    def inspect(self,request):
        return self._inspect_validated(validate_semantic_place_request(request))

    def _inspect_validated(self,context):
        request=context.request
        args=context.registry_arguments
        return match_backend_capabilities(request,args['action_space_profile'],args['semantics_profile'],
            args['predicate_definitions'],args['operator_definitions'],request.backend_descriptor_bundle.backend_descriptors[0])

    def select(self,request):
        return self._select_validated(validate_semantic_place_request(request))

    def _select_validated(self,context):
        request=context.request
        row=self._inspect_validated(context)
        selected=isinstance(row,CapabilityMatch)
        descriptor=request.backend_descriptor_bundle.backend_descriptors[0]
        return BackendSelectionRecord.seal(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,
            implementation_registry_snapshot_sha256=request.implementation_registry_snapshot.implementation_registry_snapshot_sha256,
            backend_descriptor_bundle_sha256=request.backend_descriptor_bundle.backend_descriptor_bundle_sha256,
            backend_routing_policy_sha256=request.backend_routing_policy.backend_routing_policy_sha256,
            ordered_candidate_backend_refs=(BACKEND_REF,),capability_rows=(row,),selection_disposition='SELECTED' if selected else 'NO_SELECTION',
            selection_disposition_claim_ref=d('selection-disposition'),selected_backend_ref=BACKEND_REF if selected else None,
            selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256 if selected else None,
            resource_allocation=_usage(request,allocation=True) if selected else None,deterministic_selection_reason_ref=d('selection-reason'))

    def compile(self,request):
        context=validate_semantic_place_request(request)
        if self._select_validated(context).selection_disposition!='SELECTED':
            raise ValueError('NO_SELECTION cannot compile a selected submission')
        try:
            compilation=_compile_validated(context,kernel.Budget(context.limits,'COMPILE'))
        except (UnsupportedPlacement,kernel.PlacementLimit) as error:
            return SemanticPlaceCompiledProblem(request,None,error.ledger)
        return SemanticPlaceCompiledProblem(request,compilation,None)

    def solve_submission(self,compiled,config):
        if type(compiled) is not SemanticPlaceCompiledProblem:
            raise TypeError('expected exact SemanticPlaceCompiledProblem')
        request=compiled.source_solve_request
        context=validate_semantic_place_request(request)
        if type(config) is not type(request.solver_config) or config!=request.solver_config:
            raise ValueError('solver config substitution')
        selection=self._select_validated(context)
        if selection.selection_disposition!='SELECTED':
            raise ValueError('submission requires selected routing')
        compilation=compiled.compilation
        failure=compiled.failure_ledger
        coverage=program=transition=None
        truths=()
        transport=None
        if compilation is not None:
            budget=kernel.Budget(context.limits,'SOLVE')
            coverage=kernel.solve_targets(compilation.targets,tuple(v.as_fraction for v in compilation.before_xyz),context.limits,budget=budget)
            if coverage.ledger.reason!='NONE':
                failure=coverage.ledger
            elif coverage.winner_xyz is not None:
                try:
                    budget.current='transition:transport'
                    budget.charge()
                    transport=kernel.transport_winner(coverage)
                    if transport is None:
                        raise kernel.PlacementLimit('NUMERIC_GAP',budget.current,'SOLVE')
                    program,_,truths=_materialize_validated(context,compilation,coverage.winner_target_id,transport[0],budget)
                    transition=budget.ledger()
                except kernel.PlacementLimit as error:
                    failure=transition=budget.ledger(error)
                    program=None;truths=()
        if compilation is None and failure is None:
            raise ValueError('compiled problem has neither artifact nor failure ledger')
        proof=place.SemanticPlaceProofMaterial.seal(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,
            compilation=compilation,coverage=coverage,program=program,containment=truths,transition_ledger=transition,failure_ledger=failure)
        material=ProofMaterialEnvelope.seal(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,proposal_backend_ref=BACKEND_REF,
            proof_material_definition_ref=d('proof-material'),payload_schema_ref=place.schema('proof-material'),typed_payload=(place.encode_proof(proof),),artifact_refs=())
        common=dict(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,proposal_backend_ref=BACKEND_REF,
            proposal_backend_owner_ref=OWNERS['backend'],proposal_backend_capability_ref=c('solve'),proposal_backend_build_sha256=BUILDS['backend'],
            proof_material=material,proof_material_sha256=material.proof_material_sha256,resource_usage=_usage(request,proof))
        if failure is not None:
            reason={'RESOURCE_LIMIT':'resource-exhausted','NUMERIC_GAP':'numeric-gap','UNSUPPORTED':'unsupported'}.get(failure.reason)
            if reason is None:raise ValueError('invalid failure ledger cannot become UNKNOWN')
            return BackendUnknownEvidence.seal(**common,reason_claim_definition_ref=d(reason))
        if program is not None:
            proposal=BackendProposal.seal(**common,proposal_claim_definition_ref=d('certified'),program_sha256=program.program_sha256,
                after_scene_state_sha256=program.after_scene_state_sha256,objective_lower_bound=transport[1],objective_upper_bound=transport[2])
            return BackendProposalSubmission.seal(proposal=proposal,backend_proposal_sha256=proposal.backend_proposal_sha256)
        return BackendCompleteUnsatEvidence.seal(**common,complete_domain_claim_definition_ref=d('unsat'),authorized_domain_sha256=compilation.domain.domain_sha256,
            complete_domain_coverage_artifact_sha256=coverage.coverage_sha256)
