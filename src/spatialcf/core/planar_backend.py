"""Thin, single-owner bridge for the retained planar-translate v2 solver."""

from __future__ import annotations

import spatialcf.core.solver as solver_owner
from spatialcf.core.registry import ImplementationResolutionError, SemanticContractError
from spatialcf.domain.compatibility import (
    PlanarTranslateCompilation,
    PlanarTranslateDelegatedOutcome,
    _lossless_m1_resource_used,
)
from spatialcf.domain.counterfactual import CounterfactualSolveRequest
from spatialcf.domain.definitions import (
    DigestValue,
    EnumSymbolValue,
    FiniteRealValue,
    TypedValue,
)
from spatialcf.domain.outcomes import (
    BackendProposal,
    BackendSelectionRecord,
    CapabilityMatch,
    CapabilityMismatch,
    ProofMaterialEnvelope,
    ResourceUsage,
)
from spatialcf.domain.profiles import CounterfactualSolverConfig
from spatialcf.domain.serialization import canonical_json_bytes
from spatialcf.domain.solver import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawMinimumCostSolveOutcomeV2_9,
    ContinuousYawProvenUnsatResultV2_9,
    ContinuousYawUncertifiedResultV2_9,
)

_MATCH_CLAIM_REF = "definition:spatialcf/planar-translate/filter/2.0"
_MISMATCH_CLAIM_REF = "definition:spatialcf/planar-translate/filter/2.0"
_SELECTION_CLAIM_REF = "definition:spatialcf/planar-translate/filter/2.0"
_SELECTION_REASON_REF = "definition:spatialcf/planar-translate/order/2.0"
_CERTIFIED_CLAIM_REF = (
    "definition:spatialcf/planar-translate/claim-certified-solution/2.0"
)
_PROOF_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-proof-material/2.0"
)
_STATUS_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-status/2.0"
)
_RESULT_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-result-digest/2.0"
)
_CERTIFICATE_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-certificate-digest/2.0"
)
_PROOF_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-proof-digest/2.0"
)
_CANDIDATE_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-candidate-refs-digest/2.0"
)
_EDIT_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-selected-edit-digest/2.0"
)
_LOWER_BOUND_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-objective-lower-bound/2.0"
)
_UPPER_BOUND_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-objective-upper-bound/2.0"
)


class PlanarTranslateBackend:
    """Delegate one sealed compatibility compilation to the sole retained owner.

    This deliberately is not an M1 ``SolverBackendProtocol`` implementation:
    retained-v2 UNSAT and uncertified outcomes do not have truthful finite M1
    proposal bounds.  Callers receive ``PlanarTranslateDelegatedOutcome`` for
    every source branch instead.
    """

    def __init__(self, compilation: PlanarTranslateCompilation) -> None:
        _require_exact_wire(
            compilation,
            PlanarTranslateCompilation,
            "compilation",
        )
        self._compilation = compilation

    def inspect(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> CapabilityMatch | CapabilityMismatch:
        """Report support from retained records without owner discovery or solving."""

        try:
            request = _require_exact_wire(
                solve_request,
                CounterfactualSolveRequest,
                "solve request",
            )
        except (TypeError, ValueError) as error:
            raise SemanticContractError("solve request failed strict validation") from error
        expected = self._compilation.solve_request
        self._raise_if_request_resolution_drift(request)
        if (
            request.semantic_problem.semantics_profile_ref
            != expected.semantic_problem.semantics_profile_ref
            or request.semantic_problem.action_space_profile_ref
            != expected.semantic_problem.action_space_profile_ref
        ):
            return self._capability_mismatch()
        if canonical_json_bytes(request) != canonical_json_bytes(expected):
            raise SemanticContractError(
                "solve request does not match the retained compatibility closure"
            )
        return self._capability_match()

    def compile(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> PlanarTranslateCompilation:
        """Return only the sealed compilation retained by this wrapper instance."""

        try:
            request = _require_exact_wire(
                solve_request,
                CounterfactualSolveRequest,
                "solve request",
            )
        except (TypeError, ValueError) as error:
            raise SemanticContractError("solve request failed strict validation") from error
        self._raise_if_request_resolution_drift(request)
        if canonical_json_bytes(request) != canonical_json_bytes(
            self._compilation.solve_request
        ):
            raise SemanticContractError(
                "solve request does not match the retained compatibility closure"
            )
        return self._compilation

    def solve(
        self,
        compiled: PlanarTranslateCompilation,
        config: CounterfactualSolverConfig,
    ) -> PlanarTranslateDelegatedOutcome:
        """Validate retained records, call the one v2 owner, then bind its output."""

        compilation = self._require_exact_compilation(compiled)
        self._require_exact_solver_config(config)
        source_outcome = solver_owner.solve_minimum_cost(
            compilation.source_artifacts.problem,
            compilation.source_artifacts.config,
        )
        try:
            _reject_nonlossless_source_usage(source_outcome)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise SemanticContractError(
                "source resource usage is not losslessly representable in M1"
            ) from error
        return self._map_delegated_outcome_without_certifying(
            compilation,
            source_outcome,
        )

    def _capability_match(self) -> CapabilityMatch:
        registration = self._compilation.registration
        descriptor = registration.backend_descriptor
        return CapabilityMatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            matched_capability_refs=tuple(
                sorted(
                    (
                        registration.compiler_capability_ref,
                        registration.solver_capability_ref,
                    ),
                    key=canonical_json_bytes,
                )
            ),
            match_claim_definition_ref=_MATCH_CLAIM_REF,
        )

    def _capability_mismatch(self) -> CapabilityMismatch:
        descriptor = self._compilation.registration.backend_descriptor
        return CapabilityMismatch(
            backend_ref=descriptor.backend_ref,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            missing_capability_refs=(
                self._compilation.registration.solver_capability_ref,
            ),
            reason_claim_definition_ref=_MISMATCH_CLAIM_REF,
        )

    def _raise_if_request_resolution_drift(
        self,
        request: CounterfactualSolveRequest,
    ) -> None:
        expected = self._compilation.solve_request
        if canonical_json_bytes(request.backend_descriptor_bundle) != canonical_json_bytes(
            expected.backend_descriptor_bundle
        ):
            raise ImplementationResolutionError("descriptor drift in solve request")
        if canonical_json_bytes(
            request.implementation_registry_snapshot
        ) != canonical_json_bytes(expected.implementation_registry_snapshot):
            raise ImplementationResolutionError("build drift in solve request")

    def _require_exact_compilation(
        self,
        compiled: PlanarTranslateCompilation,
    ) -> PlanarTranslateCompilation:
        if type(compiled) is not PlanarTranslateCompilation:
            raise SemanticContractError("compilation must be an exact compatibility record")
        expected = self._compilation
        self._raise_if_compilation_resolution_drift(compiled)
        semantic_label = _compilation_semantic_drift_label(compiled, expected)
        if semantic_label is not None:
            raise SemanticContractError(f"{semantic_label} drift in compilation")
        try:
            checked = _require_exact_wire(
                compiled,
                PlanarTranslateCompilation,
                "compilation",
            )
        except (TypeError, ValueError) as error:
            raise SemanticContractError("compilation failed strict semantic closure") from error
        if canonical_json_bytes(checked) != canonical_json_bytes(expected):
            raise SemanticContractError(
                "compilation does not match the retained compatibility closure"
            )
        return compiled

    def _raise_if_compilation_resolution_drift(
        self,
        compiled: PlanarTranslateCompilation,
    ) -> None:
        expected = self._compilation
        try:
            descriptor = compiled.registration.backend_descriptor
            expected_descriptor = expected.registration.backend_descriptor
            if descriptor.backend_ref != expected_descriptor.backend_ref:
                raise ImplementationResolutionError("descriptor drift in compilation")
            if (
                descriptor.implementation_build_sha256
                != expected_descriptor.implementation_build_sha256
            ):
                raise ImplementationResolutionError("build drift in compilation")
            if _wire_bytes(descriptor) != _wire_bytes(
                expected_descriptor
            ):
                raise ImplementationResolutionError("descriptor drift in compilation")
            if compiled.registration.backend_owner_ref != expected.registration.backend_owner_ref:
                raise ImplementationResolutionError("owner drift in compilation")
            if _wire_bytes(
                compiled.solve_request.implementation_registry_snapshot
            ) != _wire_bytes(
                expected.solve_request.implementation_registry_snapshot
            ):
                raise ImplementationResolutionError("build drift in compilation")
        except AttributeError as error:
            raise SemanticContractError("compilation owner closure is malformed") from error

    def _require_exact_solver_config(self, config: CounterfactualSolverConfig) -> None:
        try:
            checked = _require_exact_wire(
                config,
                CounterfactualSolverConfig,
                "solver config",
            )
        except (TypeError, ValueError) as error:
            raise SemanticContractError("solver config failed strict validation") from error
        if canonical_json_bytes(checked) != canonical_json_bytes(
            self._compilation.solve_request.solver_config
        ):
            raise SemanticContractError(
                "solver config does not match the retained solve request"
            )

    def _map_delegated_outcome_without_certifying(
        self,
        compilation: PlanarTranslateCompilation,
        source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
    ) -> PlanarTranslateDelegatedOutcome:
        """Bind source-owned fields without evaluating or certifying them."""

        source_outcome = _require_valid_source_outcome(compilation, source_outcome)
        resource_usage = _resource_usage(compilation, source_outcome)
        selection = self._selection_record(compilation, resource_usage)
        evidence = _source_evidence(source_outcome)
        proof_material = _proof_material(compilation, selection, evidence)
        proposal = _certified_success_proposal(
            compilation,
            selection,
            proof_material,
            resource_usage,
            evidence,
        )
        source = compilation.source_artifacts
        return PlanarTranslateDelegatedOutcome.seal(
            compilation=compilation,
            compilation_sha256=compilation.compilation_sha256,
            source_artifacts_sha256=source.source_artifacts_sha256,
            source_problem_semantic_sha256=source.problem_semantic_sha256,
            source_problem_canonical_sha256=source.problem_canonical_sha256,
            source_config_domain_sha256=source.config_domain_sha256,
            source_config_canonical_sha256=source.config_canonical_sha256,
            v3_semantic_problem_sha256=(
                compilation.semantic_problem.semantic_problem_sha256
            ),
            v3_solve_request_sha256=compilation.solve_request.solve_request_sha256,
            backend_ref=compilation.registration.backend_descriptor.backend_ref,
            backend_owner_ref=compilation.registration.backend_owner_ref,
            backend_build_sha256=(
                compilation.registration.backend_descriptor.implementation_build_sha256
            ),
            backend_selection_record=selection,
            backend_selection_record_sha256=(
                selection.backend_selection_record_sha256
            ),
            proof_material=proof_material,
            proof_material_sha256=proof_material.proof_material_sha256,
            resource_usage=resource_usage,
            source_outcome=source_outcome,
            source_status=evidence["status"],
            source_result_sha256=evidence["result_sha256"],
            source_certificate_sha256=evidence["certificate_sha256"],
            source_proof_sha256=evidence["proof_sha256"],
            source_candidate_refs_sha256=evidence["candidate_refs_sha256"],
            source_selected_edit_sha256=evidence["selected_edit_sha256"],
            objective_lower_bound=evidence["objective_lower_bound"],
            objective_upper_bound=evidence["objective_upper_bound"],
            backend_proposal=proposal,
        )

    def _selection_record(
        self,
        compilation: PlanarTranslateCompilation,
        resource_usage: ResourceUsage,
    ) -> BackendSelectionRecord:
        request = compilation.solve_request
        match = self._capability_match()
        descriptor = compilation.registration.backend_descriptor
        return BackendSelectionRecord.seal(
            semantic_problem_sha256=request.semantic_problem_sha256,
            solve_request_sha256=request.solve_request_sha256,
            implementation_registry_snapshot_sha256=(
                request.implementation_registry_snapshot.implementation_registry_snapshot_sha256
            ),
            backend_descriptor_bundle_sha256=(
                request.backend_descriptor_bundle.backend_descriptor_bundle_sha256
            ),
            backend_routing_policy_sha256=(
                request.backend_routing_policy.backend_routing_policy_sha256
            ),
            ordered_candidate_backend_refs=(descriptor.backend_ref,),
            capability_rows=(match,),
            selection_disposition="SELECTED",
            selection_disposition_claim_ref=_SELECTION_CLAIM_REF,
            selected_backend_ref=descriptor.backend_ref,
            selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            resource_allocation=resource_usage,
            deterministic_selection_reason_ref=_SELECTION_REASON_REF,
        )


def _require_exact_wire(value: object, model_type: type, label: str):
    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    try:
        checked = model_type.model_validate(
            value.model_dump(mode="python", round_trip=True, warnings=False),
            strict=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must pass strict canonical validation") from error
    if canonical_json_bytes(checked) != canonical_json_bytes(value):
        raise ValueError(f"{label} strict canonical bytes do not round trip")
    return checked


def _compilation_semantic_drift_label(
    observed: PlanarTranslateCompilation,
    expected: PlanarTranslateCompilation,
) -> str | None:
    if _wire_bytes(observed.source_artifacts) != _wire_bytes(
        expected.source_artifacts
    ):
        return "source"
    if _wire_bytes(observed.field_mapping) != _wire_bytes(
        expected.field_mapping
    ):
        return "mapping"
    if (
        observed.semantic_problem.semantics_profile_ref
        != expected.semantic_problem.semantics_profile_ref
        or observed.semantic_problem.action_space_profile_ref
        != expected.semantic_problem.action_space_profile_ref
        or _wire_bytes(observed.registration) != _wire_bytes(
            expected.registration
        )
    ):
        return "profile"
    if _wire_bytes(observed.semantic_problem) != _wire_bytes(
        expected.semantic_problem
    ):
        return "v3 root"
    if _wire_bytes(observed.lineage) != _wire_bytes(expected.lineage):
        return "lineage"
    if _wire_bytes(observed.solve_request) != _wire_bytes(
        expected.solve_request
    ):
        return "solve request"
    return None


def _wire_bytes(value: object) -> bytes:
    """Compare decoded hostile wires without treating validation as serialization."""

    return canonical_json_bytes(
        value.model_dump(mode="python", round_trip=True, warnings=False)
    )


def _reject_nonlossless_source_usage(source_outcome: object) -> None:
    """Fail before M1 evidence if exact retained telemetry cannot survive a float wire."""

    if type(source_outcome) is not ContinuousYawMinimumCostSolveOutcomeV2_9:
        return
    source_usage = source_outcome.cumulative_generation_usage
    if source_usage is not None:
        _lossless_m1_resource_used(source_usage.domain_operations)


def _require_valid_source_outcome(
    compilation: PlanarTranslateCompilation,
    source_outcome: object,
) -> ContinuousYawMinimumCostSolveOutcomeV2_9:
    try:
        outcome = _require_exact_wire(
            source_outcome,
            ContinuousYawMinimumCostSolveOutcomeV2_9,
            "source outcome",
        )
    except (TypeError, ValueError) as error:
        label = "certificate" if "certificate" in str(error).lower() else "result"
        raise SemanticContractError(f"source {label} failed strict validation") from error
    result = outcome.result
    if result is None:
        return source_outcome
    if type(result) not in {
        ContinuousYawCertifiedSuccessResultV2_9,
        ContinuousYawProvenUnsatResultV2_9,
        ContinuousYawUncertifiedResultV2_9,
    }:
        raise SemanticContractError("source result type is not the retained v2 terminal")
    if (
        result.semantic_problem_sha256
        != compilation.source_artifacts.problem_semantic_sha256
        or canonical_json_bytes(result.solver_config)
        != canonical_json_bytes(compilation.source_artifacts.config)
    ):
        raise SemanticContractError("source result does not bind the retained source")
    return source_outcome


def _source_evidence(
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
) -> dict[str, object | None]:
    result = source_outcome.result
    if result is None:
        return {
            "status": "MISSING_RESULT",
            "result_sha256": None,
            "certificate_sha256": None,
            "proof_sha256": None,
            "candidate_refs_sha256": None,
            "selected_edit_sha256": None,
            "objective_lower_bound": None,
            "objective_upper_bound": None,
        }
    candidate_refs = result.candidate_refs
    evidence: dict[str, object | None] = {
        "status": result.status,
        "result_sha256": result.solve_result_sha256,
        "certificate_sha256": None,
        "proof_sha256": None,
        "candidate_refs_sha256": (
            None if candidate_refs is None else candidate_refs.candidate_refs_sha256
        ),
        "selected_edit_sha256": None,
        "objective_lower_bound": None,
        "objective_upper_bound": None,
    }
    if type(result) is ContinuousYawCertifiedSuccessResultV2_9:
        return evidence | {
            "certificate_sha256": result.certificate.certificate_sha256,
            "selected_edit_sha256": result.selected_witness.edit.edit_sha256,
            "objective_lower_bound": result.global_loss_lower_bound,
            "objective_upper_bound": result.witness_loss_bounds.total_upper_bound,
        }
    if type(result) is ContinuousYawProvenUnsatResultV2_9:
        return evidence | {"proof_sha256": result.empty_outer_stage_sha256}
    assert type(result) is ContinuousYawUncertifiedResultV2_9
    return evidence


def _resource_usage(
    compilation: PlanarTranslateCompilation,
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
) -> ResourceUsage:
    resource_limit = compilation.solve_request.resource_policy.limits[0]
    source_usage = source_outcome.cumulative_generation_usage
    used = (
        0.0
        if source_usage is None
        else _lossless_m1_resource_used(source_usage.domain_operations)
    )
    return ResourceUsage(
        accounting_claim_definition_ref=(
            compilation.solve_request.resource_policy.exhaustion_claim_ref
        ),
        entries=(
            {
                "resource_definition_ref": resource_limit.definition_ref,
                "used": used,
            },
        ),
        exhausted=used == resource_limit.finite_limit,
    )


def _proof_material(
    compilation: PlanarTranslateCompilation,
    selection: BackendSelectionRecord,
    evidence: dict[str, object | None],
) -> ProofMaterialEnvelope:
    payload = [
        TypedValue(
            value_schema_ref=_STATUS_SCHEMA_REF,
            payload=EnumSymbolValue(symbol=str(evidence["status"])),
        )
    ]
    for schema_ref, key in (
        (_RESULT_SCHEMA_REF, "result_sha256"),
        (_CERTIFICATE_SCHEMA_REF, "certificate_sha256"),
        (_PROOF_SCHEMA_REF, "proof_sha256"),
        (_CANDIDATE_SCHEMA_REF, "candidate_refs_sha256"),
        (_EDIT_SCHEMA_REF, "selected_edit_sha256"),
    ):
        digest = evidence[key]
        if digest is not None:
            payload.append(
                TypedValue(
                    value_schema_ref=schema_ref,
                    payload=DigestValue(value=digest),
                )
            )
    for schema_ref, key in (
        (_LOWER_BOUND_SCHEMA_REF, "objective_lower_bound"),
        (_UPPER_BOUND_SCHEMA_REF, "objective_upper_bound"),
    ):
        bound = evidence[key]
        if bound is not None:
            payload.append(
                TypedValue(
                    value_schema_ref=schema_ref,
                    payload=FiniteRealValue(value=bound),
                )
            )
    return ProofMaterialEnvelope.seal(
        semantic_problem_sha256=selection.semantic_problem_sha256,
        solve_request_sha256=selection.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=compilation.registration.backend_descriptor.backend_ref,
        proof_material_definition_ref=(
            compilation.registration.backend_descriptor.emitted_proof_material_definition_refs[
                0
            ]
        ),
        payload_schema_ref=_PROOF_PAYLOAD_SCHEMA_REF,
        typed_payload=tuple(sorted(payload, key=canonical_json_bytes)),
        artifact_refs=(),
    )


def _certified_success_proposal(
    compilation: PlanarTranslateCompilation,
    selection: BackendSelectionRecord,
    proof_material: ProofMaterialEnvelope,
    resource_usage: ResourceUsage,
    evidence: dict[str, object | None],
) -> BackendProposal | None:
    if evidence["status"] != "CERTIFIED_SUCCESS":
        return None
    lower = evidence["objective_lower_bound"]
    upper = evidence["objective_upper_bound"]
    assert type(lower) is float and type(upper) is float
    descriptor = compilation.registration.backend_descriptor
    return BackendProposal.seal(
        semantic_problem_sha256=selection.semantic_problem_sha256,
        solve_request_sha256=selection.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=descriptor.backend_ref,
        proposal_backend_owner_ref=compilation.registration.backend_owner_ref,
        proposal_backend_capability_ref=compilation.registration.solver_capability_ref,
        proposal_backend_build_sha256=descriptor.implementation_build_sha256,
        proposal_claim_definition_ref=_CERTIFIED_CLAIM_REF,
        proof_material=proof_material,
        proof_material_sha256=proof_material.proof_material_sha256,
        objective_lower_bound=lower,
        objective_upper_bound=upper,
        resource_usage=resource_usage,
    )
