"""Static registry outcome validation and explicit dependencies."""

from __future__ import annotations

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    EditProgram,
    SceneStateEnvelope,
)

from spatialcf.domain.definitions import (
    DefinitionBundle,
    ValueSchemaDefinition,
    canonical_json_bytes,
)

from spatialcf.domain.operators import (
    DerivedFactRuleDefinition,
    OperatorDefinition,
    StateVariableDefinition,
)

from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposal,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendSubmission,
    BackendUnknownEvidence,
    CapabilityMatch,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    CheckedProofOutcome,
    CheckerDisposition,
    NoncertifiedWitnessResult,
    ProofMaterialEnvelope,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    UnknownResult,
    VerifierDispatchRecord,
)

from spatialcf.domain.predicates import (
    GroundedObligationSet,
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationRegistrySnapshot,
    ProofPolicy,
    ResourcePolicy,
    SemanticsProfile,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    ImplementationResolutionError,
    SemanticContractError,
    StaticOwner,
    _CLAIM_CHECKER_CAPABILITY_FIELD,
    _CLAIM_PROOF_MATERIAL_FIELD,
    _COMPLETE_DOMAIN_CLAIM_FIELD,
    _PROOF_CHECKER_CAPABILITY_FIELD,
    _PROOF_PAYLOAD_SCHEMA_FIELD,
    _ROLE_CERTIFIED_SOLUTION,
    _ROLE_COMPLETE_DOMAIN,
    _ROLE_NONCERTIFIED_WITNESS,
    _ROLE_PROOF_MATERIAL,
    _ROLE_PROVEN_UNSAT,
    _ROLE_ROUTING_SELECTION_DISPOSITION,
    _ROLE_ROUTING_SELECTION_REASON,
    _ROLE_SOUND_COMPLETE_DOMAIN,
    _ROLE_UNKNOWN,
    _self_digest_matches,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_record_reference,
    _definition_roles,
    _definition_root_maps,
    _require_outcome_definition_role,
    _validate_resource_usage,
    _validate_typed_value,
)

from spatialcf.core._internal.registry.routing import (
    _reconstructed_backend_rows,
)


def validate_submission_contract_structure(
    *,
    submission: BackendSubmission,
    selection: BackendSelectionRecord,
    checked_proof_outcome: CheckedProofOutcome,
    verifier_dispatch_record: VerifierDispatchRecord,
    result: (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ),
) -> None:
    """Close the M3 submission-to-assembled-record hash DAG without execution.

    This is the dispatch-time half of the additive submission validator.  The
    retained instance method above remains the sole full M1 proposal-contract
    validator; its exact signature and behavior are intentionally unchanged.
    """

    if type(checked_proof_outcome) is not CheckedProofOutcome:
        raise TypeError("submission structure requires a typed checked outcome")
    if type(verifier_dispatch_record) is not VerifierDispatchRecord:
        raise TypeError("submission structure requires a typed checker dispatch")
    if (
        verifier_dispatch_record.checked_proof_outcome_sha256
        != checked_proof_outcome.checked_proof_outcome_sha256
        or verifier_dispatch_record.checker_capability_ref
        != checked_proof_outcome.checker_capability_ref
        or verifier_dispatch_record.checker_build_sha256
        != checked_proof_outcome.checker_build_sha256
    ):
        raise SemanticContractError("checker dispatch does not bind checked outcome")
    if (
        selection.selection_disposition != "SELECTED"
        or result.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or result.checked_proof_outcome_sha256
        != checked_proof_outcome.checked_proof_outcome_sha256
        or result.verifier_dispatch_record_sha256
        != verifier_dispatch_record.verifier_dispatch_record_sha256
        or result.checker_disposition != checked_proof_outcome.checker_disposition
    ):
        raise SemanticContractError("assembled submission roots")
    evidence = (
        submission.proposal
        if type(submission) is BackendProposalSubmission
        else submission
    )
    if (
        evidence.backend_selection_record_sha256
        != selection.backend_selection_record_sha256
        or evidence.proof_material_sha256 != checked_proof_outcome.proof_material_sha256
        or evidence.proposal_backend_owner_ref
        != verifier_dispatch_record.proposal_backend_owner_ref
        or evidence.proposal_backend_capability_ref
        != verifier_dispatch_record.proposal_backend_capability_ref
        or evidence.proposal_backend_build_sha256
        != verifier_dispatch_record.proposal_backend_build_sha256
    ):
        raise SemanticContractError("assembled submission evidence")
    if type(submission) is BackendCompleteUnsatEvidence and (
        type(result) is not ProvenUnsatResult
        or result.complete_domain_coverage_artifact_sha256
        != submission.complete_domain_coverage_artifact_sha256
    ):
        raise SemanticContractError("assembled complete-domain evidence")
    if type(submission) is BackendUnknownEvidence and type(result) is not UnknownResult:
        raise SemanticContractError("assembled unknown evidence")
    if type(submission) is BackendProposalSubmission:
        if checked_proof_outcome.checker_disposition is CheckerDisposition.ACCEPTED:
            if type(result) is not CertifiedSolutionResult:
                raise SemanticContractError("assembled accepted proposal")
            certificate = result.accepted_certificate
            if (
                result.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
            ):
                raise SemanticContractError("certificate claim does not bind checker")
            if (
                certificate.proof_material_definition_ref
                != evidence.proof_material.proof_material_definition_ref
            ):
                raise SemanticContractError(
                    "certificate proof does not bind submission"
                )
            if (
                result.program_sha256 != evidence.program_sha256
                or result.after_scene_state_sha256 != evidence.after_scene_state_sha256
                or certificate.program_sha256 != evidence.program_sha256
                or certificate.after_scene_state_sha256
                != evidence.after_scene_state_sha256
            ):
                raise SemanticContractError("terminal payload does not bind submission")
        elif type(result) is not NoncertifiedWitnessResult:
            raise SemanticContractError("assembled limited proposal")


class OutcomeValidation:
    """Stateless validation methods composed by StaticImplementationRegistry."""

    def validate_outcome_contract(
        self,
        semantic_definition_bundle: DefinitionBundle,
        solve_policy_definition_bundle: DefinitionBundle,
        value_schema_definitions: tuple[ValueSchemaDefinition, ...],
        predicate_definitions: tuple[PredicateDefinition, ...],
        state_variable_definitions: tuple[StateVariableDefinition, ...],
        derived_fact_rule_definitions: tuple[DerivedFactRuleDefinition, ...],
        operator_definitions: tuple[OperatorDefinition, ...],
        implementation_registry_snapshot: ImplementationRegistrySnapshot,
        problem: CounterfactualProblemIR,
        semantics_profile: SemanticsProfile,
        action_space_profile: ActionSpaceProfile,
        scene_state: SceneStateEnvelope,
        request: CounterfactualSolveRequest,
        backend_descriptor_bundle: BackendDescriptorBundle,
        solver_config: CounterfactualSolverConfig,
        proof_policy: ProofPolicy,
        resource_policy: ResourcePolicy,
        backend_routing_policy: BackendRoutingPolicy,
        selection: BackendSelectionRecord,
        proposal: BackendProposal | None,
        proof_material: ProofMaterialEnvelope | None,
        checked_proof_outcome: CheckedProofOutcome | None,
        verifier_dispatch_record: VerifierDispatchRecord | None,
        certificate: CertifiedSolutionCertificate | ProvenUnsatCertificate | None,
        result: (
            CertifiedSolutionResult
            | ProvenUnsatResult
            | NoncertifiedWitnessResult
            | UnknownResult
        ),
        program: EditProgram | None,
        grounded_obligations: GroundedObligationSet | None,
    ) -> (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ):
        """Validate the submitted, branch-specific hash DAG without execution.

        The method accepts all four structural terminal envelopes.  A weak
        branch is admissible as a weak branch when policy permits it; only a
        certificate or complete-domain UNSAT branch may claim trusted strength.
        """

        self.validate_solve_request(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
            value_schema_definitions,
            predicate_definitions,
            state_variable_definitions,
            derived_fact_rule_definitions,
            operator_definitions,
            implementation_registry_snapshot,
            problem,
            semantics_profile,
            action_space_profile,
            scene_state,
            request,
            backend_descriptor_bundle,
            solver_config,
            proof_policy,
            resource_policy,
            backend_routing_policy,
        )
        for record in (
            selection,
            proposal,
            proof_material,
            checked_proof_outcome,
            verifier_dispatch_record,
            result,
        ):
            if record is not None and not _self_digest_matches(record):
                raise SemanticContractError("outcome record hash")
        if certificate is not None and not _self_digest_matches(certificate):
            raise SemanticContractError("certificate hash")
        if (
            result.semantic_problem_sha256 != problem.semantic_problem_sha256
            or result.solve_request_sha256 != request.solve_request_sha256
        ):
            raise SemanticContractError("result roots")
        if (
            selection.semantic_problem_sha256 != problem.semantic_problem_sha256
            or selection.solve_request_sha256 != request.solve_request_sha256
            or selection.implementation_registry_snapshot_sha256
            != implementation_registry_snapshot.implementation_registry_snapshot_sha256
            or selection.backend_descriptor_bundle_sha256
            != backend_descriptor_bundle.backend_descriptor_bundle_sha256
            or selection.backend_routing_policy_sha256
            != backend_routing_policy.backend_routing_policy_sha256
        ):
            raise SemanticContractError("selection roots")

        _semantic_definitions, _solve_definitions, definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        roles = _definition_roles(definitions)
        available_backend_refs = {
            descriptor.backend_ref
            for descriptor in backend_descriptor_bundle.backend_descriptors
        }
        unavailable_backend_refs = {
            unavailable.backend_ref
            for unavailable in backend_descriptor_bundle.unavailable_optional_backends
        }
        if not available_backend_refs or (
            available_backend_refs & unavailable_backend_refs
        ):
            # Task 7 has no descriptor digest field for an unavailable row, so
            # a nonempty BackendSelectionRecord cannot safely represent an
            # unavailable-only universe.  It must terminate fail-closed rather
            # than fabricate a CapabilityMismatch.
            raise SemanticContractError("unavailable backend selection")
        if (
            set(selection.ordered_candidate_backend_refs) & unavailable_backend_refs
            or selection.selected_backend_ref in unavailable_backend_refs
        ):
            raise SemanticContractError("unavailable backend selection")
        expected_rows = _reconstructed_backend_rows(
            request,
            action_space_profile,
            semantics_profile,
            predicate_definitions,
            operator_definitions,
            definitions,
            roles,
        )
        if (
            set(selection.ordered_candidate_backend_refs) != available_backend_refs
            or selection.ordered_candidate_backend_refs
            != tuple(row.backend_ref for row in expected_rows)
            or canonical_json_bytes(selection.capability_rows)
            != canonical_json_bytes(expected_rows)
        ):
            raise SemanticContractError("ordered backend candidates")
        if (
            result.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
        ):
            raise SemanticContractError("result roots")
        if (
            roles.get(selection.selection_disposition_claim_ref)
            != _ROLE_ROUTING_SELECTION_DISPOSITION
            or roles.get(selection.deterministic_selection_reason_ref)
            != _ROLE_ROUTING_SELECTION_REASON
        ):
            raise SemanticContractError("selection definition closure")
        descriptor_by_ref = {
            descriptor.backend_ref: descriptor
            for descriptor in backend_descriptor_bundle.backend_descriptors
        }
        matching_rows = tuple(
            row for row in expected_rows if isinstance(row, CapabilityMatch)
        )
        if selection.selection_disposition == "NO_SELECTION":
            if matching_rows:
                raise SemanticContractError("deterministic selected backend")
            if any(
                item is not None
                for item in (
                    proposal,
                    proof_material,
                    checked_proof_outcome,
                    verifier_dispatch_record,
                    certificate,
                    program,
                    grounded_obligations,
                )
            ) or not isinstance(result, UnknownResult):
                raise SemanticContractError("no-selection outcome")
            if (
                result.checked_proof_outcome_sha256 is not None
                or result.verifier_dispatch_record_sha256 is not None
                or result.checker_disposition is not None
            ):
                raise SemanticContractError("no-selection checker")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                result.reason_claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "unknown reason admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or result.reason_claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("unknown claim admissibility")
            _validate_resource_usage(
                result.resource_usage, resource_policy, definitions
            )
            return result
        if selection.selection_disposition != "SELECTED":
            raise SemanticContractError("selection disposition")
        if (
            selection.selected_backend_ref is None
            or selection.selected_backend_descriptor_sha256 is None
            or selection.selected_backend_ref not in descriptor_by_ref
            or descriptor_by_ref[
                selection.selected_backend_ref
            ].backend_descriptor_sha256
            != selection.selected_backend_descriptor_sha256
        ):
            raise SemanticContractError("selected backend descriptor")
        selected_row = next(
            (
                row
                for row in expected_rows
                if row.backend_ref == selection.selected_backend_ref
            ),
            None,
        )
        if not isinstance(selected_row, CapabilityMatch):
            raise SemanticContractError("selected backend capability")
        if not matching_rows or selected_row != matching_rows[0]:
            raise SemanticContractError("deterministic selected backend")
        descriptor = descriptor_by_ref[selection.selected_backend_ref]

        if proposal is None or proof_material is None:
            raise SemanticContractError("selected proposal presence")

        if (
            proposal.semantic_problem_sha256 != problem.semantic_problem_sha256
            or proposal.solve_request_sha256 != request.solve_request_sha256
            or proposal.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or proposal.proposal_backend_ref != selection.selected_backend_ref
            or proposal.proof_material_sha256 != proof_material.proof_material_sha256
        ):
            raise SemanticContractError("proposal roots")
        if (
            proof_material.semantic_problem_sha256 != problem.semantic_problem_sha256
            or proof_material.solve_request_sha256 != request.solve_request_sha256
            or proof_material.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or proof_material.proposal_backend_ref != selection.selected_backend_ref
            or proof_material.proof_material_definition_ref
            not in descriptor.emitted_proof_material_definition_refs
        ):
            raise SemanticContractError("proof material roots")
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        if any(
            value.value_schema_ref != proof_material.payload_schema_ref
            for value in proof_material.typed_payload
        ):
            raise SemanticContractError("proof material schema")
        try:
            for value in proof_material.typed_payload:
                _validate_typed_value(value, schemas)
        except DefinitionClosureError as error:
            raise SemanticContractError("proof material schema") from error
        if proposal.proposal_claim_definition_ref != result.claim_definition_ref:
            raise SemanticContractError("proposal claim")
        if isinstance(result, CertifiedSolutionResult) and (
            program is None
            or proposal.program_sha256 != program.program_sha256
            or proposal.after_scene_state_sha256 != program.after_scene_state_sha256
        ):
            raise SemanticContractError("proposal program")
        capability_owners = self._owners_by_capability()
        proposal_owner = capability_owners.get(proposal.proposal_backend_capability_ref)
        if (
            proposal_owner is None
            or proposal_owner.owner_ref != proposal.proposal_backend_owner_ref
            or proposal_owner.implementation_build_sha256
            != proposal.proposal_backend_build_sha256
            or descriptor.implementation_build_sha256
            != proposal.proposal_backend_build_sha256
            or proposal.proposal_backend_capability_ref
            not in action_space_profile.backend_capability_requirements
        ):
            raise ImplementationResolutionError("proposal backend owner/build")
        result_has_checker = result.checked_proof_outcome_sha256 is not None
        if (checked_proof_outcome is None) != (
            verifier_dispatch_record is None
        ) or result_has_checker != (checked_proof_outcome is not None):
            raise SemanticContractError("checker pair")
        checker_owner: StaticOwner | None = None
        if checked_proof_outcome is not None:
            assert verifier_dispatch_record is not None
            checker_owner = capability_owners.get(
                checked_proof_outcome.checker_capability_ref
            )
            if (
                checker_owner is None
                or checker_owner.implementation_build_sha256
                != checked_proof_outcome.checker_build_sha256
                or checker_owner.owner_ref == proposal_owner.owner_ref
                or checked_proof_outcome.checker_capability_ref
                not in descriptor.compatible_checker_capability_refs
                or checked_proof_outcome.checker_capability_ref
                not in proof_policy.required_checker_capability_refs
            ):
                raise ImplementationResolutionError("checker owner/build")
            if (
                checked_proof_outcome.semantic_problem_sha256
                != problem.semantic_problem_sha256
                or checked_proof_outcome.solve_request_sha256
                != request.solve_request_sha256
                or checked_proof_outcome.backend_selection_record_sha256
                != selection.backend_selection_record_sha256
                or checked_proof_outcome.proof_material_sha256
                != proof_material.proof_material_sha256
                or checked_proof_outcome.checked_claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or result.checked_proof_outcome_sha256
                != checked_proof_outcome.checked_proof_outcome_sha256
                or result.verifier_dispatch_record_sha256
                != verifier_dispatch_record.verifier_dispatch_record_sha256
                or result.checker_disposition
                != checked_proof_outcome.checker_disposition
            ):
                raise SemanticContractError("checked proof roots")
            if (
                verifier_dispatch_record.semantic_problem_sha256
                != problem.semantic_problem_sha256
                or verifier_dispatch_record.solve_request_sha256
                != request.solve_request_sha256
                or verifier_dispatch_record.semantic_definition_bundle_sha256
                != semantic_definition_bundle.definition_bundle_sha256
                or verifier_dispatch_record.solve_policy_definition_bundle_sha256
                != solve_policy_definition_bundle.definition_bundle_sha256
                or verifier_dispatch_record.backend_selection_record_sha256
                != selection.backend_selection_record_sha256
                or verifier_dispatch_record.checked_proof_outcome_sha256
                != checked_proof_outcome.checked_proof_outcome_sha256
                or verifier_dispatch_record.proof_policy_sha256
                != proof_policy.proof_policy_sha256
                or verifier_dispatch_record.proposal_backend_owner_ref
                != proposal.proposal_backend_owner_ref
                or verifier_dispatch_record.proposal_backend_capability_ref
                != proposal.proposal_backend_capability_ref
                or verifier_dispatch_record.proposal_backend_build_sha256
                != proposal.proposal_backend_build_sha256
                or verifier_dispatch_record.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or verifier_dispatch_record.checker_owner_ref != checker_owner.owner_ref
                or verifier_dispatch_record.checker_capability_ref
                != checked_proof_outcome.checker_capability_ref
                or verifier_dispatch_record.checker_build_sha256
                != checked_proof_outcome.checker_build_sha256
            ):
                raise SemanticContractError("verifier dispatch roots")

        for usage in (
            selection.resource_allocation,
            proposal.resource_usage,
            result.resource_usage,
            certificate.resource_usage if certificate is not None else None,
        ):
            if usage is not None:
                _validate_resource_usage(usage, resource_policy, definitions)
        _require_outcome_definition_role(
            roles,
            proof_material.proof_material_definition_ref,
            (_ROLE_PROOF_MATERIAL,),
            "proof material admissibility",
        )
        proof_definition = definitions[proof_material.proof_material_definition_ref]
        if (
            _definition_record_reference(
                proof_definition,
                _PROOF_PAYLOAD_SCHEMA_FIELD,
            )
            != proof_material.payload_schema_ref
        ):
            raise SemanticContractError("proof material schema")
        if (
            checked_proof_outcome is not None
            and _definition_record_reference(
                proof_definition,
                _PROOF_CHECKER_CAPABILITY_FIELD,
            )
            != checked_proof_outcome.checker_capability_ref
        ):
            raise SemanticContractError("proof material checker")

        if isinstance(result, CertifiedSolutionResult):
            if not isinstance(certificate, CertifiedSolutionCertificate):
                raise SemanticContractError("certificate branch")
            if program is None or grounded_obligations is None:
                raise SemanticContractError("certificate program")
            if checked_proof_outcome is None or verifier_dispatch_record is None:
                raise SemanticContractError("certificate checker")
            self.validate_edit_program(
                semantic_definition_bundle,
                solve_policy_definition_bundle,
                value_schema_definitions,
                predicate_definitions,
                state_variable_definitions,
                derived_fact_rule_definitions,
                operator_definitions,
                implementation_registry_snapshot,
                problem,
                semantics_profile,
                action_space_profile,
                scene_state,
                program,
                scene_state,
                program.after_scene_state,
                grounded_obligations,
                problem.intervention_authorization,
            )
            _require_outcome_definition_role(
                roles,
                certificate.claim_definition_ref,
                (_ROLE_CERTIFIED_SOLUTION,),
                "claim admissibility",
            )
            claim_definition = definitions[certificate.claim_definition_ref]
            if (
                _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                != proof_material.proof_material_definition_ref
                or _definition_record_reference(
                    claim_definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
                != checked_proof_outcome.checker_capability_ref
            ):
                raise SemanticContractError("claim admissibility")
            if (
                certificate.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or certificate.proof_material_definition_ref
                != _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
            ):
                raise SemanticContractError("certificate proof material")
            if (
                certificate != result.accepted_certificate
                or certificate.claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref != result.claim_definition_ref
                or certificate.claim_definition_ref
                not in proof_policy.accepted_claim_definition_refs
                or certificate.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or certificate.checker_disposition is not CheckerDisposition.ACCEPTED
                or checked_proof_outcome.checker_disposition
                is not CheckerDisposition.ACCEPTED
                or certificate.program_sha256 != program.program_sha256
                or certificate.after_scene_state_sha256
                != program.after_scene_state_sha256
                or certificate.state_delta_manifest_sha256
                != program.state_delta_manifest.state_delta_manifest_sha256
                or certificate.grounded_obligation_set_sha256
                != grounded_obligations.grounded_obligation_set_sha256
                or result.certificate_sha256 != certificate.certificate_sha256
                or result.program_sha256 != program.program_sha256
                or result.after_scene_state_sha256 != program.after_scene_state_sha256
            ):
                raise SemanticContractError("certificate admissibility")
        elif isinstance(result, ProvenUnsatResult):
            if not isinstance(certificate, ProvenUnsatCertificate):
                raise SemanticContractError("complete-domain certificate")
            if program is not None or grounded_obligations is not None:
                raise SemanticContractError("unsat program presence")
            if checked_proof_outcome is None or verifier_dispatch_record is None:
                raise SemanticContractError("complete-domain checker")
            _require_outcome_definition_role(
                roles,
                certificate.claim_definition_ref,
                (_ROLE_PROVEN_UNSAT,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                certificate.complete_domain_claim_definition_ref,
                (_ROLE_COMPLETE_DOMAIN,),
                "complete-domain coverage",
            )
            _require_outcome_definition_role(
                roles,
                certificate.sound_complete_domain_claim_definition_ref,
                (_ROLE_SOUND_COMPLETE_DOMAIN,),
                "complete-domain coverage",
            )
            claim_definition = definitions[certificate.claim_definition_ref]
            complete_domain_definition = definitions[
                certificate.complete_domain_claim_definition_ref
            ]
            sound_domain_definition = definitions[
                certificate.sound_complete_domain_claim_definition_ref
            ]
            if (
                _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
                != proof_material.proof_material_definition_ref
                or _definition_record_reference(
                    claim_definition,
                    _CLAIM_CHECKER_CAPABILITY_FIELD,
                )
                != checked_proof_outcome.checker_capability_ref
            ):
                raise SemanticContractError("claim admissibility")
            if (
                certificate.proof_material_definition_ref
                != proof_material.proof_material_definition_ref
                or certificate.proof_material_definition_ref
                != _definition_record_reference(
                    claim_definition,
                    _CLAIM_PROOF_MATERIAL_FIELD,
                )
            ):
                raise SemanticContractError("certificate proof material")
            if (
                _definition_record_reference(
                    complete_domain_definition,
                    _COMPLETE_DOMAIN_CLAIM_FIELD,
                )
                != certificate.claim_definition_ref
                or _definition_record_reference(
                    sound_domain_definition,
                    _COMPLETE_DOMAIN_CLAIM_FIELD,
                )
                != certificate.claim_definition_ref
            ):
                raise SemanticContractError("complete-domain coverage")
            if (
                certificate != result.accepted_certificate
                or certificate.claim_definition_ref
                != proposal.proposal_claim_definition_ref
                or certificate.claim_definition_ref
                != checked_proof_outcome.checked_claim_definition_ref
                or certificate.claim_definition_ref != result.claim_definition_ref
                or certificate.claim_definition_ref
                not in proof_policy.accepted_claim_definition_refs
                or certificate.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or certificate.checker_disposition is not CheckerDisposition.ACCEPTED
                or checked_proof_outcome.checker_disposition
                is not CheckerDisposition.ACCEPTED
                or result.certificate_sha256 != certificate.certificate_sha256
                or result.complete_domain_coverage_artifact_sha256
                != certificate.complete_domain_coverage_artifact_sha256
            ):
                raise SemanticContractError("complete-domain coverage")
        elif isinstance(result, NoncertifiedWitnessResult):
            if certificate is not None:
                raise SemanticContractError("witness certificate presence")
            if not proof_policy.permit_noncertified_terminal_records:
                raise SemanticContractError("noncertified policy")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_NONCERTIFIED_WITNESS,),
                "claim admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("claim admissibility")
            if result.evidence_claim_definition_ref not in definitions:
                raise SemanticContractError("witness evidence admissibility")
            if result.program_sha256 is None:
                if (
                    program is not None
                    or grounded_obligations is not None
                    or proposal.program_sha256 is not None
                    or proposal.after_scene_state_sha256 is not None
                ):
                    raise SemanticContractError("witness program presence")
            else:
                if program is None or grounded_obligations is None:
                    raise SemanticContractError("witness program presence")
                self.validate_edit_program(
                    semantic_definition_bundle,
                    solve_policy_definition_bundle,
                    value_schema_definitions,
                    predicate_definitions,
                    state_variable_definitions,
                    derived_fact_rule_definitions,
                    operator_definitions,
                    implementation_registry_snapshot,
                    problem,
                    semantics_profile,
                    action_space_profile,
                    scene_state,
                    program,
                    scene_state,
                    program.after_scene_state,
                    grounded_obligations,
                    problem.intervention_authorization,
                )
                if (
                    result.program_sha256 != program.program_sha256
                    or result.after_scene_state_sha256
                    != program.after_scene_state_sha256
                    or proposal.program_sha256 != program.program_sha256
                    or proposal.after_scene_state_sha256
                    != program.after_scene_state_sha256
                ):
                    raise SemanticContractError("witness program")
            if result.checker_disposition is CheckerDisposition.ACCEPTED:
                raise SemanticContractError("weak claim promotion")
            if (
                checked_proof_outcome is not None
                and checked_proof_outcome.checker_disposition
                is CheckerDisposition.ACCEPTED
            ):
                raise SemanticContractError("weak claim promotion")
        elif isinstance(result, UnknownResult):
            if (
                certificate is not None
                or program is not None
                or grounded_obligations is not None
            ):
                raise SemanticContractError("unknown certificate presence")
            if (
                proposal.program_sha256 is not None
                or proposal.after_scene_state_sha256 is not None
            ):
                raise SemanticContractError("unknown proposal program")
            _require_outcome_definition_role(
                roles,
                result.claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "claim admissibility",
            )
            _require_outcome_definition_role(
                roles,
                result.reason_claim_definition_ref,
                (_ROLE_UNKNOWN,),
                "unknown reason admissibility",
            )
            if (
                result.claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
                or result.reason_claim_definition_ref
                not in action_space_profile.allowed_claim_definition_refs
            ):
                raise SemanticContractError("unknown claim admissibility")
            if result.checker_disposition is CheckerDisposition.ACCEPTED:
                raise SemanticContractError("unknown claim promotion")
            if (
                checked_proof_outcome is not None
                and checked_proof_outcome.checker_disposition
                is CheckerDisposition.ACCEPTED
            ):
                raise SemanticContractError("unknown claim promotion")
        else:  # pragma: no cover - discriminated domain union is exhaustive.
            raise SemanticContractError("result branch")

        # A direct result/certificate root must carry every static dependency;
        # this rejects forward, reverse, self, and mixed-branch digest swaps.
        if certificate is not None:
            certificate_fields = (
                ("semantic_problem_sha256", problem.semantic_problem_sha256),
                ("solve_request_sha256", request.solve_request_sha256),
                ("scene_state_sha256", scene_state.scene_state_sha256),
                (
                    "backend_selection_record_sha256",
                    selection.backend_selection_record_sha256,
                ),
                (
                    "checked_proof_outcome_sha256",
                    checked_proof_outcome.checked_proof_outcome_sha256,
                ),
                (
                    "verifier_dispatch_record_sha256",
                    verifier_dispatch_record.verifier_dispatch_record_sha256,
                ),
                (
                    "semantic_definition_bundle_sha256",
                    semantic_definition_bundle.definition_bundle_sha256,
                ),
                (
                    "solve_policy_definition_bundle_sha256",
                    solve_policy_definition_bundle.definition_bundle_sha256,
                ),
                (
                    "semantics_profile_sha256",
                    semantics_profile.semantics_profile_sha256,
                ),
                (
                    "action_space_profile_sha256",
                    action_space_profile.action_space_profile_sha256,
                ),
                (
                    "intervention_authorization_sha256",
                    problem.intervention_authorization.intervention_authorization_sha256,
                ),
                (
                    "objective_expression_sha256",
                    problem.objective_expression.objective_expression_sha256,
                ),
                ("proof_policy_sha256", proof_policy.proof_policy_sha256),
                ("resource_policy_sha256", resource_policy.resource_policy_sha256),
                (
                    "backend_routing_policy_sha256",
                    backend_routing_policy.backend_routing_policy_sha256,
                ),
                ("solver_config_sha256", solver_config.solver_config_sha256),
                (
                    "implementation_registry_snapshot_sha256",
                    implementation_registry_snapshot.implementation_registry_snapshot_sha256,
                ),
                (
                    "backend_descriptor_bundle_sha256",
                    backend_descriptor_bundle.backend_descriptor_bundle_sha256,
                ),
                (
                    "proposal_backend_build_sha256",
                    proposal.proposal_backend_build_sha256,
                ),
                ("checker_build_sha256", checked_proof_outcome.checker_build_sha256),
                ("proof_material_sha256", proof_material.proof_material_sha256),
            )
            if any(
                getattr(certificate, name) != expected
                for name, expected in certificate_fields
            ):
                raise SemanticContractError("certificate roots")
        return result

    def validate_submission_contract(
        self,
        *,
        submission: BackendSubmission,
        outcome_contract_arguments: dict[str, object],
    ) -> (
        CertifiedSolutionResult
        | ProvenUnsatResult
        | NoncertifiedWitnessResult
        | UnknownResult
    ):
        """Validate one M3 V2 submission without changing the retained API.

        Proposal wrappers add no semantics: after exact wrapper binding they
        delegate verbatim to :meth:`validate_outcome_contract`.  Program-free
        terminal evidence is checked only for its additive structural roots;
        terminal assembly remains the sole owner of certificates and results.
        """

        if type(submission) is BackendProposalSubmission:
            proposal = outcome_contract_arguments.get("proposal")
            if (
                proposal is None
                or canonical_json_bytes(proposal)
                != canonical_json_bytes(submission.proposal)
                or submission.backend_proposal_sha256
                != submission.proposal.backend_proposal_sha256
            ):
                raise SemanticContractError("proposal submission wrapper")
            return self.validate_outcome_contract(**outcome_contract_arguments)  # type: ignore[arg-type]

        if type(submission) not in (
            BackendCompleteUnsatEvidence,
            BackendUnknownEvidence,
        ):
            raise SemanticContractError("submission branch")
        selection = outcome_contract_arguments.get("selection")
        result = outcome_contract_arguments.get("result")
        if type(selection) is not BackendSelectionRecord:
            raise SemanticContractError("submission selection")
        if type(result) not in (ProvenUnsatResult, UnknownResult):
            raise SemanticContractError("terminal submission result")
        if (
            submission.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or result.backend_selection_record_sha256
            != selection.backend_selection_record_sha256
            or submission.proof_material_sha256
            != submission.proof_material.proof_material_sha256
            or canonical_json_bytes(submission.resource_usage)
            != canonical_json_bytes(result.resource_usage)
        ):
            raise SemanticContractError("terminal submission roots")
        if type(submission) is BackendCompleteUnsatEvidence:
            if (
                type(result) is not ProvenUnsatResult
                or result.complete_domain_coverage_artifact_sha256
                != submission.complete_domain_coverage_artifact_sha256
            ):
                raise SemanticContractError("complete-domain submission")
        elif type(result) is not UnknownResult:
            raise SemanticContractError("unknown submission")
        return result


# Keep public class, exception and bound-method lookup stable.
validate_submission_contract_structure.__module__ = "spatialcf.core.registry"
OutcomeValidation.validate_outcome_contract.__module__ = "spatialcf.core.registry"
OutcomeValidation.validate_outcome_contract.__qualname__ = "StaticImplementationRegistry.validate_outcome_contract"
OutcomeValidation.validate_submission_contract.__module__ = "spatialcf.core.registry"
OutcomeValidation.validate_submission_contract.__qualname__ = "StaticImplementationRegistry.validate_submission_contract"
