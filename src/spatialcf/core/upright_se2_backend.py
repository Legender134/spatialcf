"""Exact-cardinal proposal orchestration for ``spatialcf/upright_se2@1``.

This module produces untrusted V2 backend submissions only.  It deliberately
owns neither a checker/certificate/result path nor geometry formulas: every
scene-to-kernel value comes through the public compiler bridge and every
geometric result comes from the retained cardinal kernel owners.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TypeAlias

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
    BackendProposal,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendSubmission,
    BackendUnknownEvidence,
    CapabilityMatch,
    CapabilityMismatch,
    ProofMaterialEnvelope,
    ResourceUsage,
    TypedCompilationOutcome,
)
from spatialcf.domain.profiles import (
    CounterfactualSolverConfig,
    SolverBackendDescriptor,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = (
    "UprightSE2Backend",
    "UprightSE2CardinalBackend",
    "UprightSE2ContinuousBackend",
)

_BACKEND_REF = "backend:spatialcf/upright-se2/cardinal"
_CONTINUOUS_BACKEND_REF = "backend:spatialcf/upright-se2/continuous"
_BACKEND_BUILD_SHA256 = "b" * 64
_MATCH_CLAIM_REF = "definition:spatialcf/upright-se2/filter/1.0"
_MISMATCH_CLAIM_REF = "definition:spatialcf/upright-se2/filter/1.0"
_SELECTION_CLAIM_REF = "definition:spatialcf/upright-se2/filter/1.0"
_SELECTION_REASON_REF = "definition:spatialcf/upright-se2/order/1.0"
_UNKNOWN_UNSUPPORTED_REF = (
    "definition:spatialcf/upright-se2/backend-unsupported-semantic-input/1.0"
)
_UNKNOWN_NUMERIC_REF = "definition:spatialcf/upright-se2/backend-numeric-gap/1.0"
_UNKNOWN_INCOMPLETE_REF = (
    "definition:spatialcf/upright-se2/backend-incomplete-cardinal-coverage/1.0"
)
_CONFIG_SUBSTITUTION_REF = (
    "definition:spatialcf/upright-se2/backend-config-substitution/1.0"
)
_UNSAT_CLAIM_REF = "definition:spatialcf/upright-se2/claim-proven-unsat/1.0"
_PROPOSAL_CLAIM_REF = "definition:spatialcf/upright-se2/claim-certified-solution/1.0"
_MAX_EXACT_DYADIC_REFINEMENT_DEPTH = 3
_MAX_CONTINUOUS_EXACT_DYADIC_REFINEMENT_DEPTH = 3

_Submission: TypeAlias = (
    BackendProposalSubmission | BackendCompleteUnsatEvidence | BackendUnknownEvidence
)


class UprightSE2CardinalBackend:
    """A V2-only exact-cardinal backend with no retained ``solve`` member."""

    def inspect(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> CapabilityMatch | CapabilityMismatch:
        """Match only the fully closed registered cardinal request/profile subset."""

        descriptor = _descriptor_or_none(solve_request)
        if not _is_exact_model(solve_request, CounterfactualSolveRequest):
            return _capability_mismatch(descriptor)
        try:
            compilation = compile_upright_se2(solve_request)
        except (TypeError, ValueError, ArithmeticError):
            return _capability_mismatch(descriptor)
        if type(compilation) is not upright.UprightSE2Compilation:
            return _capability_mismatch(descriptor)
        if not _descriptor_closes_cardinal_request(descriptor, compilation):
            return _capability_mismatch(descriptor)
        return CapabilityMatch(
            backend_ref=_BACKEND_REF,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            matched_capability_refs=tuple(
                sorted(
                    (
                        *upright.UPRIGHT_SE2_CARDINAL_CAPABILITY_REFS,
                        upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
                        upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
                    ),
                    key=canonical_json_bytes,
                )
            ),
            match_claim_definition_ref=_MATCH_CLAIM_REF,
        )

    def compile(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> (
        upright.UprightSE2Compilation
        | upright.UprightSE2ContinuousCompilation
        | TypedCompilationOutcome
    ):
        """Delegate to the sole compiler without retaining a compilation cache.

        Direct construction can return the additive continuous sibling; this
        cardinal backend still rejects it at ``inspect`` and only accepts a
        cardinal compilation at ``solve_submission``.
        """

        return compile_upright_se2(solve_request)

    def solve_submission(
        self,
        compiled: upright.UprightSE2Compilation,
        config: CounterfactualSolverConfig,
    ) -> BackendSubmission:
        """Return an untrusted proposal, complete UNSAT evidence, or UNKNOWN.

        The source roots stay on ``compiled`` rather than in an instance cache.
        That makes each submission independently bound to its immutable request
        and leaves later checker dispatch and terminal assembly to their owners.
        """

        compilation = _require_exact_compilation(compiled)
        request = compilation.source_solve_request
        descriptor = _descriptor_or_none(request)
        inspection = self.inspect(request)
        selection = _selection_record(request, descriptor, inspection)
        if not isinstance(inspection, CapabilityMatch):
            proof = _synthetic_incomplete_proof(
                compilation,
                reason="unsupported-cardinal-capability",
            )
            return _unknown_submission(
                compilation,
                selection,
                proof,
                _UNKNOWN_UNSUPPORTED_REF,
            )
        if not _config_matches_compilation(config, compilation):
            proof = _synthetic_incomplete_proof(
                compilation,
                reason="solver-config-substitution",
            )
            return _unknown_submission(
                compilation,
                selection,
                proof,
                _CONFIG_SUBSTITUTION_REF,
            )
        return _evaluate_cardinal_compilation(compilation, selection)


class UprightSE2ContinuousBackend:
    """V2-only continuous submission backend with no checker/result authority."""

    def inspect(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> CapabilityMatch | CapabilityMismatch:
        descriptor = _continuous_descriptor_or_none(solve_request)
        if not _is_exact_model(solve_request, CounterfactualSolveRequest):
            return _continuous_capability_mismatch(descriptor)
        try:
            compilation = compile_upright_se2(solve_request)
        except (TypeError, ValueError, ArithmeticError):
            return _continuous_capability_mismatch(descriptor)
        if type(compilation) is not upright.UprightSE2ContinuousCompilation:
            return _continuous_capability_mismatch(descriptor)
        if not _descriptor_closes_continuous_request(descriptor, compilation):
            return _continuous_capability_mismatch(descriptor)
        assert descriptor is not None
        return CapabilityMatch(
            backend_ref=_CONTINUOUS_BACKEND_REF,
            backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
            matched_capability_refs=tuple(
                sorted(
                    (
                        *upright.UPRIGHT_SE2_CONTINUOUS_CAPABILITY_REFS,
                        upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
                        upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,
                    ),
                    key=canonical_json_bytes,
                )
            ),
            match_claim_definition_ref=_MATCH_CLAIM_REF,
        )

    def compile(
        self,
        solve_request: CounterfactualSolveRequest,
    ) -> upright.UprightSE2ContinuousCompilation | TypedCompilationOutcome:
        return compile_upright_se2(solve_request)

    def solve_submission(
        self,
        compiled: upright.UprightSE2ContinuousCompilation,
        config: CounterfactualSolverConfig,
    ) -> BackendSubmission:
        compilation = _require_exact_continuous_compilation(compiled)
        request = compilation.source_solve_request
        descriptor = _continuous_descriptor_or_none(request)
        inspection = self.inspect(request)
        selection = _continuous_selection_record(request, descriptor, inspection)
        if not isinstance(inspection, CapabilityMatch):
            return _continuous_unknown_submission(
                compilation,
                selection,
                _UNKNOWN_UNSUPPORTED_REF,
                exhausted=False,
            )
        if not _config_matches_continuous_compilation(config, compilation):
            return _continuous_unknown_submission(
                compilation,
                selection,
                _CONFIG_SUBSTITUTION_REF,
                exhausted=False,
            )
        return _evaluate_continuous_compilation(compilation, selection)


# The shorter historical spelling is an additive alias, not another backend.
UprightSE2Backend = UprightSE2CardinalBackend


def _is_exact_model(value: object, model_type: type) -> bool:
    """Require strict self-digest validation without accepting a coerced model."""

    if type(value) is not model_type:
        return False
    try:
        checked = model_type.model_validate(
            value.model_dump(mode="python", round_trip=True),
            strict=True,
        )
    except (TypeError, ValueError):
        return False
    return canonical_json_bytes(checked) == canonical_json_bytes(value)


def _require_exact_compilation(value: object) -> upright.UprightSE2Compilation:
    """Reject a substituted compilation before it reaches a retained owner."""

    if not _is_exact_model(value, upright.UprightSE2Compilation):
        raise TypeError("compiled value must be an exact UprightSE2Compilation")
    assert type(value) is upright.UprightSE2Compilation
    return value


def _require_exact_continuous_compilation(
    value: object,
) -> upright.UprightSE2ContinuousCompilation:
    if not _is_exact_model(value, upright.UprightSE2ContinuousCompilation):
        raise TypeError(
            "compiled value must be an exact UprightSE2ContinuousCompilation"
        )
    assert type(value) is upright.UprightSE2ContinuousCompilation
    return value


def _descriptor_or_none(
    solve_request: object,
) -> SolverBackendDescriptor | None:
    """Return the one concrete descriptor by identity, never by prefix."""

    if type(solve_request) is not CounterfactualSolveRequest:
        return None
    descriptors = tuple(
        descriptor
        for descriptor in solve_request.backend_descriptor_bundle.backend_descriptors
        if descriptor.backend_ref == _BACKEND_REF
    )
    return descriptors[0] if len(descriptors) == 1 else None


def _continuous_descriptor_or_none(
    solve_request: object,
) -> SolverBackendDescriptor | None:
    if type(solve_request) is not CounterfactualSolveRequest:
        return None
    descriptors = tuple(
        descriptor
        for descriptor in solve_request.backend_descriptor_bundle.backend_descriptors
        if descriptor.backend_ref == _CONTINUOUS_BACKEND_REF
    )
    return descriptors[0] if len(descriptors) == 1 else None


def _capability_mismatch(
    descriptor: SolverBackendDescriptor | None,
) -> CapabilityMismatch:
    """Emit a closed mismatch row even when no valid descriptor is available."""

    descriptor_sha256 = (
        descriptor.backend_descriptor_sha256 if descriptor is not None else "0" * 64
    )
    digest = canonical_sha256(
        (
            descriptor_sha256,
            _BACKEND_REF,
            "exact-cardinal-capability-closure",
        ),
        domain="spatialcf/counterfactual/upright-se2/capability-mismatch/3.0",
    )
    return CapabilityMismatch(
        backend_ref=_BACKEND_REF,
        backend_descriptor_sha256=descriptor_sha256,
        missing_capability_refs=(
            f"capability:spatialcf/upright-se2/mismatch/{digest}",
        ),
        reason_claim_definition_ref=_MISMATCH_CLAIM_REF,
    )


def _continuous_capability_mismatch(
    descriptor: SolverBackendDescriptor | None,
) -> CapabilityMismatch:
    descriptor_sha256 = (
        descriptor.backend_descriptor_sha256 if descriptor is not None else "0" * 64
    )
    digest = canonical_sha256(
        (descriptor_sha256, _CONTINUOUS_BACKEND_REF, "continuous-capability-closure"),
        domain="spatialcf/counterfactual/upright-se2/capability-mismatch/3.0",
    )
    return CapabilityMismatch(
        backend_ref=_CONTINUOUS_BACKEND_REF,
        backend_descriptor_sha256=descriptor_sha256,
        missing_capability_refs=(
            f"capability:spatialcf/upright-se2/mismatch/{digest}",
        ),
        reason_claim_definition_ref=_MISMATCH_CLAIM_REF,
    )


def _descriptor_closes_cardinal_request(
    descriptor: SolverBackendDescriptor | None,
    compilation: upright.UprightSE2Compilation,
) -> bool:
    """Compare every descriptor dimension against the compiled exact closure."""

    if descriptor is None:
        return False
    registration = compilation.semantic_closure.profile_registration
    request = compilation.source_solve_request
    resource_refs = tuple(
        sorted(
            (limit.definition_ref for limit in request.resource_policy.limits),
            key=canonical_json_bytes,
        )
    )
    checker_refs = tuple(
        sorted(
            (
                upright.UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
                upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
                upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
            ),
            key=canonical_json_bytes,
        )
    )
    registry_builds = dict(
        request.implementation_registry_snapshot.implementation_build_hashes
    )
    registry_owners = {
        binding.definition_or_capability_ref: binding.implementation_owner_ref
        for binding in request.implementation_registry_snapshot.definition_and_capability_owner_bindings
    }
    return (
        descriptor.backend_ref == _BACKEND_REF
        and descriptor.implementation_build_sha256 == _BACKEND_BUILD_SHA256
        and descriptor.supported_profile_hashes
        == (registration.action_space_profile.action_space_profile_sha256,)
        and descriptor.supported_predicate_capabilities
        == (upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,)
        and descriptor.supported_operator_capabilities
        == (upright.UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,)
        and descriptor.supported_objective_capabilities
        == (upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,)
        and descriptor.supported_numeric_semantics
        == (registration.semantics_profile.numeric_semantics_ref,)
        and descriptor.emitted_proof_material_definition_refs
        == (upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF,)
        and descriptor.compatible_checker_capability_refs == checker_refs
        and descriptor.resource_definition_refs == resource_refs
        and registry_builds.get(upright.UPRIGHT_SE2_BACKEND_OWNER_REF)
        == _BACKEND_BUILD_SHA256
        and registry_owners.get(upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF)
        == upright.UPRIGHT_SE2_BACKEND_OWNER_REF
        and compilation.closure.semantic_closure_sha256
        == compilation.semantic_closure.semantic_closure_sha256
        and compilation.closure.policy_bundle_sha256
        == compilation.semantic_closure.policy_bundle_sha256
    )


def _descriptor_closes_continuous_request(
    descriptor: SolverBackendDescriptor | None,
    compilation: upright.UprightSE2ContinuousCompilation,
) -> bool:
    if descriptor is None:
        return False
    registration = compilation.semantic_closure.profile_registration
    request = compilation.source_solve_request
    resource_refs = tuple(
        sorted(
            (limit.definition_ref for limit in request.resource_policy.limits),
            key=canonical_json_bytes,
        )
    )
    checker_refs = tuple(
        sorted(
            (
                upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
                upright.UPRIGHT_SE2_PREDICATE_VERIFIER_CAPABILITY_REF,
                upright.UPRIGHT_SE2_OBJECTIVE_VERIFIER_CAPABILITY_REF,
            ),
            key=canonical_json_bytes,
        )
    )
    registry_builds = dict(
        request.implementation_registry_snapshot.implementation_build_hashes
    )
    registry_owners = {
        binding.definition_or_capability_ref: binding.implementation_owner_ref
        for binding in request.implementation_registry_snapshot.definition_and_capability_owner_bindings
    }
    return (
        descriptor.backend_ref == _CONTINUOUS_BACKEND_REF
        and descriptor.implementation_build_sha256 == _BACKEND_BUILD_SHA256
        and descriptor.supported_profile_hashes
        == (registration.action_space_profile.action_space_profile_sha256,)
        and descriptor.supported_predicate_capabilities
        == (upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,)
        and descriptor.supported_operator_capabilities
        == (upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,)
        and descriptor.supported_objective_capabilities
        == (upright.UPRIGHT_SE2_OBJECTIVE_EVALUATOR_CAPABILITY_REF,)
        and descriptor.supported_numeric_semantics
        == (registration.semantics_profile.numeric_semantics_ref,)
        and descriptor.emitted_proof_material_definition_refs
        == (upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF,)
        and descriptor.compatible_checker_capability_refs == checker_refs
        and descriptor.resource_definition_refs == resource_refs
        and registry_builds.get(upright.UPRIGHT_SE2_BACKEND_OWNER_REF)
        == _BACKEND_BUILD_SHA256
        and registry_owners.get(upright.UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF)
        == upright.UPRIGHT_SE2_COMPILER_OWNER_REF
        and registry_owners.get(upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF)
        == upright.UPRIGHT_SE2_BACKEND_OWNER_REF
        and registry_owners.get(upright.UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF)
        == upright.UPRIGHT_SE2_CHECKER_OWNER_REF
        and compilation.closure.semantic_closure_sha256
        == compilation.semantic_closure.semantic_closure_sha256
        and compilation.closure.policy_bundle_sha256
        == compilation.semantic_closure.policy_bundle_sha256
    )


def _resource_usage(
    request: CounterfactualSolveRequest,
    *,
    used: float,
    exhausted: bool,
) -> ResourceUsage:
    """Create the sole request-bound resource row without a reset ledger."""

    limits = request.resource_policy.limits
    if len(limits) != 1:
        raise ValueError("upright se2 requires one exact resource limit")
    return ResourceUsage.model_validate(
        {
            "accounting_claim_definition_ref": request.resource_policy.shared_ledger_policy_ref,
            "entries": (
                {
                    "resource_definition_ref": limits[0].definition_ref,
                    "used": used,
                },
            ),
            "exhausted": exhausted,
        }
    )


def _aggregate_usage(usages: tuple[ResourceUsage, ...]) -> ResourceUsage:
    """Reproduce the domain-owned canonical ledger aggregation transparently."""

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


def _selection_record(
    request: CounterfactualSolveRequest,
    descriptor: SolverBackendDescriptor | None,
    inspection: CapabilityMatch | CapabilityMismatch,
) -> BackendSelectionRecord:
    """Bind deterministic routing before untrusted proof transport is built."""

    selected = isinstance(inspection, CapabilityMatch) and descriptor is not None
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
        ordered_candidate_backend_refs=(_BACKEND_REF,),
        capability_rows=(inspection,),
        selection_disposition="SELECTED" if selected else "NO_SELECTION",
        selection_disposition_claim_ref=_SELECTION_CLAIM_REF,
        selected_backend_ref=_BACKEND_REF if selected else None,
        selected_backend_descriptor_sha256=(
            descriptor.backend_descriptor_sha256 if selected and descriptor else None
        ),
        resource_allocation=(
            _resource_usage(request, used=0.0, exhausted=False) if selected else None
        ),
        deterministic_selection_reason_ref=_SELECTION_REASON_REF,
    )


def _continuous_selection_record(
    request: CounterfactualSolveRequest,
    descriptor: SolverBackendDescriptor | None,
    inspection: CapabilityMatch | CapabilityMismatch,
) -> BackendSelectionRecord:
    """Bind continuous routing without reusing cardinal selection bytes."""

    selected = isinstance(inspection, CapabilityMatch) and descriptor is not None
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
        ordered_candidate_backend_refs=(_CONTINUOUS_BACKEND_REF,),
        capability_rows=(inspection,),
        selection_disposition="SELECTED" if selected else "NO_SELECTION",
        selection_disposition_claim_ref=_SELECTION_CLAIM_REF,
        selected_backend_ref=_CONTINUOUS_BACKEND_REF if selected else None,
        selected_backend_descriptor_sha256=(
            descriptor.backend_descriptor_sha256 if selected and descriptor else None
        ),
        resource_allocation=(
            _resource_usage(request, used=0.0, exhausted=False) if selected else None
        ),
        deterministic_selection_reason_ref=_SELECTION_REASON_REF,
    )


def _config_matches_compilation(
    config: object,
    compilation: upright.UprightSE2Compilation,
) -> bool:
    """Refuse a substituted config rather than trusting an ambient invocation."""

    return _is_exact_model(config, CounterfactualSolverConfig) and (
        canonical_json_bytes(config)
        == canonical_json_bytes(compilation.source_solve_request.solver_config)
    )


def _config_matches_continuous_compilation(
    config: object,
    compilation: upright.UprightSE2ContinuousCompilation,
) -> bool:
    return _is_exact_model(config, CounterfactualSolverConfig) and (
        canonical_json_bytes(config)
        == canonical_json_bytes(compilation.source_solve_request.solver_config)
    )


def _cell_order_key(
    cell: upright.UprightSE2CompiledCell,
) -> tuple[Fraction, Fraction, bytes]:
    return (
        cell.yaw_interval.lower.as_fraction,
        cell.yaw_interval.upper.as_fraction,
        canonical_json_bytes(cell.cell_id),
    )


def _sorted_rows(
    rows: tuple[upright.UprightSE2ProofCellEvaluation, ...],
) -> tuple[upright.UprightSE2ProofCellEvaluation, ...]:
    return tuple(sorted(rows, key=lambda row: _cell_order_key(row.compiled_cell)))


def _dyadic(value: Fraction) -> upright.ExactDyadic:
    """Convert an already exact dyadic subdivision endpoint to its domain wire."""

    if value.denominator & (value.denominator - 1):
        raise ValueError("cardinal subdivision endpoint is not dyadic")
    return upright.ExactDyadic(numerator=value.numerator, denominator=value.denominator)


def _split_exact_dyadic_cell(
    cell: upright.UprightSE2CompiledCell,
) -> tuple[upright.UprightSE2CompiledCell, ...]:
    """Deterministically split X first, then Y, without widening degenerate axes."""

    x_lower = cell.x_lower.as_fraction
    x_upper = cell.x_upper.as_fraction
    y_lower = cell.y_lower.as_fraction
    y_upper = cell.y_upper.as_fraction
    x_parts = ((cell.x_lower, cell.x_upper, ""),)
    y_parts = ((cell.y_lower, cell.y_upper, ""),)
    if x_lower < x_upper:
        midpoint = _dyadic((x_lower + x_upper) / 2)
        x_parts = (
            (cell.x_lower, midpoint, "x-lower"),
            (midpoint, cell.x_upper, "x-upper"),
        )
    if y_lower < y_upper:
        midpoint = _dyadic((y_lower + y_upper) / 2)
        y_parts = (
            (cell.y_lower, midpoint, "y-lower"),
            (midpoint, cell.y_upper, "y-upper"),
        )
    if len(x_parts) == len(y_parts) == 1:
        return ()
    return tuple(
        upright.UprightSE2CompiledCell.seal(
            cell_id=(
                f"{cell.cell_id}/{x_name}/{y_name}"
                if x_name and y_name
                else f"{cell.cell_id}/{x_name or y_name}"
            ),
            authorization_sha256=cell.authorization_sha256,
            x_lower=x_lower_part,
            x_upper=x_upper_part,
            y_lower=y_lower_part,
            y_upper=y_upper_part,
            yaw_interval=cell.yaw_interval,
        )
        for x_lower_part, x_upper_part, x_name in x_parts
        for y_lower_part, y_upper_part, y_name in y_parts
    )


def _outcome_kind(
    kind: CardinalKernelKindV3,
) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    return {
        CardinalKernelKindV3.EXACT: upright.UprightSE2RetainedOwnerOutcomeKind.EXACT,
        CardinalKernelKindV3.NUMERIC_GAP: upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP,
        CardinalKernelKindV3.RESOURCE_LIMIT: upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT,
        CardinalKernelKindV3.UNSUPPORTED: upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED,
    }[kind]


def _transport_owner_outcome(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    outcome: object | None,
    evaluator_capability_ref: str,
    label: str,
    budget: SO2AtomicBudgetV2 | None,
    forced_kind: upright.UprightSE2RetainedOwnerOutcomeKind | None = None,
    forced_atomic_steps: int = 0,
    additional_exact_bounds: tuple[TypedValue, ...] = (),
) -> upright.UprightSE2RetainedOwnerEvaluation:
    """Adapt retained kernel output to the compiler-owned proof transport seam."""

    if forced_kind is None:
        assert outcome is not None
        kernel_kind = outcome.kind
        if type(kernel_kind) is not CardinalKernelKindV3:
            raise TypeError("retained owner returned an unknown cardinal outcome")
        proof_kind = _outcome_kind(kernel_kind)
        raw_rows = tuple(outcome.proof_rows)
        raw_findings = tuple(outcome.finding_codes)
        atomic_steps = outcome.atomic_steps_used
        bounds = outcome.bounds
    else:
        proof_kind = forced_kind
        raw_rows = (f"proof:spatialcf/upright-se2/{label}/incomplete",)
        raw_findings = ("INCOMPLETE",)
        atomic_steps = forced_atomic_steps
        bounds = None
    if type(atomic_steps) is not int or atomic_steps < 0:
        raise TypeError("retained owner atomic usage must be a non-negative exact int")
    exhausted = proof_kind is upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT
    return build_upright_se2_retained_owner_evaluation(
        compiled_cell=cell,
        owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        evaluator_capability_ref=evaluator_capability_ref,
        outcome_kind=proof_kind,
        raw_proof_rows=raw_rows,
        raw_findings=raw_findings,
        atomic_steps=atomic_steps,
        resource_delta=_resource_usage(
            request,
            used=float(atomic_steps),
            exhausted=exhausted,
        ),
        label=label,
        exact_bound_value=bounds,
        additional_exact_bounds=additional_exact_bounds,
    )


def _stage_for_cell(
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
) -> upright.UprightSE2ProofStageDelta:
    """Bind every owner call for one common cell into one non-reset stage."""

    ordered = tuple(sorted(evaluations, key=canonical_json_bytes))
    return upright.UprightSE2ProofStageDelta.seal(
        stage_ref=(
            "stage:spatialcf/upright-se2/cardinal-retained-evaluation/"
            f"{cell.compiled_cell_sha256}"
        ),
        owner_evaluations=ordered,
        resource_delta=_aggregate_usage(
            tuple(evaluation.resource_delta for evaluation in ordered)
        ),
    )


def _proof_material(
    compilation: upright.UprightSE2Compilation,
    rows: tuple[upright.UprightSE2ProofCellEvaluation, ...],
    stages: tuple[upright.UprightSE2ProofStageDelta, ...],
    proposal_candidates: tuple[upright.UprightSE2ProposalCandidate, ...] = (),
) -> upright.UprightSE2ProofMaterial:
    """Seal the one request-relative tuple/coverage/frontier/ledger payload."""

    ordered_rows = _sorted_rows(rows)
    leaves = tuple(row for row in ordered_rows if row.leaf_disposition is not None)
    coverage_cells = tuple(row.compiled_cell for row in leaves)
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
    ledger = upright.UprightSE2ProofResourceLedger.seal(
        stage_deltas=ordered_stages,
        canonical_total_resource_usage=_aggregate_usage(
            tuple(stage.resource_delta for stage in ordered_stages)
        ),
    )
    return upright.UprightSE2ProofMaterial.seal(
        solve_request_sha256=compilation.solve_request_sha256,
        semantic_closure_sha256=compilation.semantic_closure.semantic_closure_sha256,
        upright_se2_compilation_sha256=compilation.upright_se2_compilation_sha256,
        compilation=compilation,
        cardinal_tuple_roster=(
            upright.UprightSE2CardinalProofTuple.seal(
                authorization=compilation.operation.authorization,
                reference_id=compilation.endpoint_construction_recipe.reference_id,
                translation_domain=compilation.operation.translation_domain,
                compiled_cells=compilation.compiled_cells,
            ),
        ),
        coverage_artifact=upright.UprightSE2CoverageArtifact.seal(
            authorization_sha256=compilation.operation.authorization_sha256,
            cells=coverage_cells,
            unresolved_cell_sha256s=tuple(
                sorted(row.compiled_cell.compiled_cell_sha256 for row in unresolved)
            ),
        ),
        compiled_cell_sha256s=tuple(
            cell.compiled_cell_sha256 for cell in coverage_cells
        ),
        evaluated_cells=ordered_rows,
        proposal_order=proposals,
        proposal_candidates=tuple(
            sorted(proposal_candidates, key=lambda candidate: candidate.canonical_order_key)
        ),
        prune_decisions=tuple(
            upright.UprightSE2ProofPruneDecision.seal(
                cell_evaluation=row,
                prune_reason_codes=(
                    "reason:spatialcf/upright-se2/cardinal-exact-refinement-cap",
                ),
            )
            for row in pruned
        ),
        unresolved_frontier=tuple(
            upright.UprightSE2ProofFrontierRow.seal(
                cell_evaluation=row,
                frontier_reason_codes=(
                    "reason:spatialcf/upright-se2/cardinal-owner-nonexact",
                ),
            )
            for row in unresolved
        ),
        resource_ledger=ledger,
    )


def _synthetic_incomplete_proof(
    compilation: upright.UprightSE2Compilation,
    *,
    reason: str,
) -> upright.UprightSE2ProofMaterial:
    """Represent pre-owner unsupported/incomplete work as UNKNOWN evidence."""

    request = compilation.source_solve_request
    rows: list[upright.UprightSE2ProofCellEvaluation] = []
    stages: list[upright.UprightSE2ProofStageDelta] = []
    for cell in compilation.compiled_cells:
        evaluation = _transport_owner_outcome(
            request=request,
            cell=cell,
            outcome=None,
            evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
            label=reason,
            budget=None,
            forced_kind=upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE,
        )
        cell_row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=cell,
            owner_evaluations=(evaluation,),
            leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
            complete_domain_empty=False,
        )
        rows.append(cell_row)
        stages.append(_stage_for_cell(cell, (evaluation,)))
    return _proof_material(compilation, tuple(rows), tuple(stages))


def _classify_exact_cell(
    box_outcome: object,
    visibility_outcomes: tuple[object, ...],
) -> upright.UprightSE2ProofLeafDisposition | None:
    """Classify only exact owner outputs; ``None`` requests deterministic refinement."""

    all_outcomes = (box_outcome, *visibility_outcomes)
    if any(
        getattr(outcome, "kind", None) is not CardinalKernelKindV3.EXACT
        for outcome in all_outcomes
    ):
        return upright.UprightSE2ProofLeafDisposition.UNRESOLVED
    box_bounds = box_outcome.bounds
    visibility_bounds = tuple(outcome.bounds for outcome in visibility_outcomes)
    if (
        box_bounds.outer_hard_constraint_slack < 0
        or box_bounds.relation_outer_failure
        or any(bounds.outer_failure for bounds in visibility_bounds)
    ):
        return upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
    if (
        box_bounds.inner_hard_constraint_slack >= 0
        and box_bounds.relation_inner_success
        and all(bounds.inner_success for bounds in visibility_bounds)
    ):
        return upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
    return None


def _point_cell(
    cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Select the lower-owned exact point of one final inward leaf."""

    if cell.x_lower == cell.x_upper and cell.y_lower == cell.y_upper:
        return cell
    x = (
        cell.x_lower
        if cell.x_lower == cell.x_upper
        else _dyadic((cell.x_lower.as_fraction + cell.x_upper.as_fraction) / 2)
    )
    y = (
        cell.y_lower
        if cell.y_lower == cell.y_upper
        else _dyadic((cell.y_lower.as_fraction + cell.y_upper.as_fraction) / 2)
    )
    return upright.UprightSE2CompiledCell.seal(
        cell_id=f"{cell.cell_id}/proposal-point",
        authorization_sha256=cell.authorization_sha256,
        x_lower=x,
        x_upper=x,
        y_lower=y,
        y_upper=y,
        yaw_interval=cell.yaw_interval,
    )


def _point_objective(
    bounds: object, compilation: upright.UprightSE2Compilation
) -> upright.UprightSE2ProposalPointObjective:
    """Transport the retained point owner's exact T/A/R/V/S terms unchanged."""

    del compilation
    semantic_terms = bounds.common_cell_semantic_objective_terms
    weighted_terms = bounds.common_cell_objective_terms
    point_terms = tuple(
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
        (lower for _term_id, lower, _upper in weighted_terms),
        start=Fraction(0),
    )
    total_upper = sum(
        (upper for _term_id, _lower, upper in weighted_terms),
        start=Fraction(0),
    )
    return upright.UprightSE2ProposalPointObjective(
        terms=point_terms,
        total_lower=upright.UprightSE2ExactRational(
            numerator=total_lower.numerator,
            denominator=total_lower.denominator,
        ),
        total_upper=upright.UprightSE2ExactRational(
            numerator=total_upper.numerator,
            denominator=total_upper.denominator,
        ),
    )


def _proposal_candidate(
    *,
    compilation: upright.UprightSE2Compilation,
    request: CounterfactualSolveRequest,
    final_inward_cell: upright.UprightSE2ProofCellEvaluation,
    final_box_bounds: object | None,
    budget: SO2AtomicBudgetV2,
) -> tuple[
    upright.UprightSE2ProposalCandidate,
    upright.UprightSE2ProofCellEvaluation,
    upright.UprightSE2ProofStageDelta | None,
]:
    """Evaluate and transport one retained exact point for an inward leaf."""

    point_cell = _point_cell(final_inward_cell.compiled_cell)
    if point_cell is final_inward_cell.compiled_cell:
        if final_box_bounds is None:
            raise ValueError("degenerate root requires its retained exact box bounds")
        objective = _point_objective(final_box_bounds, compilation)
        point = Vec2(
            x=float(point_cell.x_lower.as_fraction),
            y=float(point_cell.y_lower.as_fraction),
        )
        endpoint = materialize_upright_se2_endpoint(compilation, point)
        point_evaluation = upright.UprightSE2ProposalPointEvaluation.seal(
            point_cell_evaluation=final_inward_cell,
            point_objective=objective,
        )
        return (
            upright.UprightSE2ProposalCandidate.seal(
                final_inward_cell=final_inward_cell,
                selected_translation_xy_m=point,
                point_evaluation=point_evaluation,
                point_objective=objective,
                materialized_endpoint=endpoint,
                program=endpoint.program,
            ),
            final_inward_cell,
            None,
        )
    point_row, point_objective, point_stage = _evaluate_point_cell(
        compilation=compilation,
        request=request,
        point_cell=point_cell,
        budget=budget,
    )
    if point_objective is None:
        raise ValueError("retained point owner did not return exact evidence")
    point = Vec2(
        x=float(point_cell.x_lower.as_fraction),
        y=float(point_cell.y_lower.as_fraction),
    )
    endpoint = materialize_upright_se2_endpoint(compilation, point)
    point_evaluation = upright.UprightSE2ProposalPointEvaluation.seal(
        point_cell_evaluation=point_row,
        point_objective=point_objective,
    )
    candidate = upright.UprightSE2ProposalCandidate.seal(
        final_inward_cell=final_inward_cell,
        selected_translation_xy_m=point,
        point_evaluation=point_evaluation,
        point_objective=point_objective,
        materialized_endpoint=endpoint,
        program=endpoint.program,
    )
    return candidate, point_row, point_stage


def _evaluate_point_cell(
    *,
    compilation: upright.UprightSE2Compilation,
    request: CounterfactualSolveRequest,
    point_cell: upright.UprightSE2CompiledCell,
    budget: SO2AtomicBudgetV2,
) -> tuple[
    upright.UprightSE2ProofCellEvaluation,
    upright.UprightSE2ProposalPointObjective | None,
    upright.UprightSE2ProofStageDelta,
]:
    """Evaluate a strict point through owners without yet materializing a program."""
    inputs = build_upright_se2_cardinal_evaluation_inputs(compilation, point_cell)
    if budget.limit != inputs.resource_atomic_step_limit:
        raise ValueError("public bridge resource cap drifted at proposal point")
    box_outcome = evaluate_fixed_cardinal_cell_v3(
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
    visibility_outcomes = tuple(
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
    exact = all(
        outcome.kind is CardinalKernelKindV3.EXACT
        for outcome in (box_outcome, *visibility_outcomes)
    )
    objective = _point_objective(box_outcome.bounds, compilation) if exact else None
    point_objective_bound = (
        upright._retained_point_objective_value(objective) if objective else None
    )
    evaluations = tuple(
        sorted(
            (
                _transport_owner_outcome(
                    request=request,
                    cell=point_cell,
                    outcome=box_outcome,
                    evaluator_capability_ref=(
                        upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                    ),
                    label="fixed-cardinal-point-cell",
                    budget=budget,
                    additional_exact_bounds=(point_objective_bound,)
                    if point_objective_bound is not None
                    else (),
                ),
                *(
                    _transport_owner_outcome(
                        request=request,
                        cell=point_cell,
                        outcome=outcome,
                        evaluator_capability_ref=(
                            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                        ),
                        label=f"fixed-cardinal-point-visibility-{index}",
                        budget=budget,
                    )
                    for index, outcome in enumerate(visibility_outcomes)
                ),
            ),
            key=canonical_json_bytes,
        )
    )
    point_row = upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=point_cell,
        owner_evaluations=evaluations,
        leaf_disposition=None,
        complete_domain_empty=None,
    )
    return point_row, objective, _stage_for_cell(point_cell, evaluations)


def _evaluate_cardinal_compilation(
    compilation: upright.UprightSE2Compilation,
    selection: BackendSelectionRecord,
) -> _Submission:
    """Evaluate exact cells through public bridge/kernel seams with one budget."""

    request = compilation.source_solve_request
    budget: SO2AtomicBudgetV2 | None = None
    rows: list[upright.UprightSE2ProofCellEvaluation] = []
    stages: list[upright.UprightSE2ProofStageDelta] = []
    candidates: list[upright.UprightSE2ProposalCandidate] = []

    def evaluate_cell(cell: upright.UprightSE2CompiledCell, depth: int) -> None:
        nonlocal budget
        unreported_start_used = 0 if budget is None else budget.used
        try:
            inputs = build_upright_se2_cardinal_evaluation_inputs(compilation, cell)
            if budget is None:
                budget = SO2AtomicBudgetV2(limit=inputs.resource_atomic_step_limit)
            elif budget.limit != inputs.resource_atomic_step_limit:
                raise ValueError("public bridge resource cap drifted between cells")
            point_row: upright.UprightSE2ProofCellEvaluation | None = None
            point_objective: upright.UprightSE2ProposalPointObjective | None = None
            if cell.x_lower != cell.x_upper or cell.y_lower != cell.y_upper:
                point_row, point_objective, point_stage = _evaluate_point_cell(
                    compilation=compilation,
                    request=request,
                    point_cell=_point_cell(cell),
                    budget=budget,
                )
                rows.append(point_row)
                stages.append(point_stage)
                unreported_start_used = budget.used
                if point_objective is None:
                    incomplete = _transport_owner_outcome(
                        request=request,
                        cell=cell,
                        outcome=None,
                        evaluator_capability_ref=(
                            upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                        ),
                        label="point-first-parent-incomplete",
                        budget=None,
                        forced_kind=(
                            upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE
                        ),
                    )
                    parent = upright.UprightSE2ProofCellEvaluation.seal(
                        compiled_cell=cell,
                        owner_evaluations=(incomplete,),
                        leaf_disposition=(
                            upright.UprightSE2ProofLeafDisposition.UNRESOLVED
                        ),
                        complete_domain_empty=False,
                    )
                    rows.append(parent)
                    stages.append(_stage_for_cell(cell, (incomplete,)))
                    return
            box_outcome = evaluate_fixed_cardinal_cell_v3(
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
            visibility_outcomes = tuple(
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
            retained_point_objective = (
                _point_objective(box_outcome.bounds, compilation)
                if all(
                    outcome.kind is CardinalKernelKindV3.EXACT
                    for outcome in (box_outcome, *visibility_outcomes)
                )
                else None
            )
            retained_point_objective_bound = (
                upright._retained_point_objective_value(retained_point_objective)
                if retained_point_objective is not None
                else None
            )
            owner_evaluations = tuple(
                sorted(
                    (
                        _transport_owner_outcome(
                            request=request,
                            cell=cell,
                            outcome=box_outcome,
                            evaluator_capability_ref=(
                                upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                            ),
                            label="fixed-cardinal-cell",
                            budget=budget,
                            additional_exact_bounds=(retained_point_objective_bound,)
                            if retained_point_objective_bound is not None
                            else (),
                        ),
                        *(
                            _transport_owner_outcome(
                                request=request,
                                cell=cell,
                                outcome=outcome,
                                evaluator_capability_ref=(
                                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                                ),
                                label=f"fixed-cardinal-visibility-{index}",
                                budget=budget,
                            )
                            for index, outcome in enumerate(visibility_outcomes)
                        ),
                    ),
                    key=canonical_json_bytes,
                )
            )
            classification = _classify_exact_cell(box_outcome, visibility_outcomes)
        except (TypeError, ValueError, ArithmeticError) as error:
            forced_kind = (
                upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP
                if isinstance(error, ArithmeticError)
                else (
                    upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED
                    if "unsupported" in str(error).lower()
                    else upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE
                )
            )
            evaluation = _transport_owner_outcome(
                request=request,
                cell=cell,
                outcome=None,
                evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
                label="public-bridge-or-owner-incomplete",
                budget=budget,
                forced_kind=forced_kind,
                forced_atomic_steps=(
                    0 if budget is None else budget.used - unreported_start_used
                ),
            )
            row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=(evaluation,),
                leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
                complete_domain_empty=False,
            )
            rows.append(row)
            stages.append(_stage_for_cell(cell, (evaluation,)))
            return

        if classification is None:
            children = _split_exact_dyadic_cell(cell)
            if children and depth < _MAX_EXACT_DYADIC_REFINEMENT_DEPTH:
                internal = upright.UprightSE2ProofCellEvaluation.seal(
                    compiled_cell=cell,
                    owner_evaluations=owner_evaluations,
                    leaf_disposition=None,
                    complete_domain_empty=None,
                )
                rows.append(internal)
                stages.append(_stage_for_cell(cell, owner_evaluations))
                for child in children:
                    evaluate_cell(child, depth + 1)
                return
            finite_miss = _transport_owner_outcome(
                request=request,
                cell=cell,
                outcome=None,
                evaluator_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                label="finite-refinement-miss",
                budget=None,
                forced_kind=upright.UprightSE2RetainedOwnerOutcomeKind.FINITE_MISS,
            )
            row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=tuple(
                    sorted((*owner_evaluations, finite_miss), key=canonical_json_bytes)
                ),
                leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
                complete_domain_empty=False,
            )
            rows.append(row)
            stages.append(_stage_for_cell(cell, row.owner_evaluations))
            return

        row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=cell,
            owner_evaluations=owner_evaluations,
            leaf_disposition=classification,
            complete_domain_empty=(
                classification
                is upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
            ),
        )
        rows.append(row)
        stages.append(_stage_for_cell(cell, owner_evaluations))
        if classification is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE:
            if budget is None:
                raise ValueError("inward point evaluation requires a shared budget")
            if point_row is not None and point_objective is not None:
                point = Vec2(
                    x=float(point_row.compiled_cell.x_lower.as_fraction),
                    y=float(point_row.compiled_cell.y_lower.as_fraction),
                )
                endpoint = materialize_upright_se2_endpoint(compilation, point)
                point_evaluation = upright.UprightSE2ProposalPointEvaluation.seal(
                    point_cell_evaluation=point_row,
                    point_objective=point_objective,
                )
                candidate = upright.UprightSE2ProposalCandidate.seal(
                    final_inward_cell=row,
                    selected_translation_xy_m=point,
                    point_evaluation=point_evaluation,
                    point_objective=point_objective,
                    materialized_endpoint=endpoint,
                    program=endpoint.program,
                )
                point_stage = None
            else:
                candidate, point_row, point_stage = _proposal_candidate(
                    compilation=compilation,
                    request=request,
                    final_inward_cell=row,
                    final_box_bounds=box_outcome.bounds,
                    budget=budget,
                )
            candidates.append(candidate)
            if point_stage is not None:
                rows.append(point_row)
                stages.append(point_stage)

    for root in compilation.compiled_cells:
        evaluate_cell(root, 0)

    proof = _proof_material(
        compilation,
        tuple(rows),
        tuple(stages),
        tuple(candidates),
    )
    if proof.proposal_candidates:
        return _proposal_submission(
            compilation, selection, proof, proof.proposal_candidates[0]
        )
    leaves = tuple(
        row for row in proof.evaluated_cells if row.leaf_disposition is not None
    )
    if leaves and all(
        row.leaf_disposition
        is upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
        for row in leaves
    ):
        return _unsat_submission(compilation, selection, proof)
    outcome_kinds = {
        evaluation.outcome_kind
        for row in proof.evaluated_cells
        for evaluation in row.owner_evaluations
    }
    reason = (
        request.resource_policy.exhaustion_claim_ref
        if upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT in outcome_kinds
        else (
            _UNKNOWN_NUMERIC_REF
            if upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP in outcome_kinds
            else (
                _UNKNOWN_UNSUPPORTED_REF
                if upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED
                in outcome_kinds
                else _UNKNOWN_INCOMPLETE_REF
            )
        )
    )
    return _unknown_submission(compilation, selection, proof, reason)


def _proof_envelope(
    compilation: upright.UprightSE2Compilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ProofMaterial,
) -> ProofMaterialEnvelope:
    """Use the domain codec exactly once for the general proof envelope."""

    return ProofMaterialEnvelope.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_BACKEND_REF,
        proof_material_definition_ref=upright.UPRIGHT_SE2_PROOF_MATERIAL_DEFINITION_REF,
        payload_schema_ref=upright.UPRIGHT_SE2_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        typed_payload=(upright.encode_upright_se2_proof_material(proof),),
        artifact_refs=(),
    )


def _unknown_submission(
    compilation: upright.UprightSE2Compilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ProofMaterial,
    reason_claim_definition_ref: str,
) -> BackendUnknownEvidence:
    """Emit honest terminal UNKNOWN without objective/program/result fields."""

    envelope = _proof_envelope(compilation, selection, proof)
    return BackendUnknownEvidence.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        resource_usage=proof.total_resource_usage,
        reason_claim_definition_ref=reason_claim_definition_ref,
    )


def _continuous_proof_material(
    compilation: upright.UprightSE2ContinuousCompilation,
    rows: tuple[upright.UprightSE2ProofCellEvaluation, ...],
    stages: tuple[upright.UprightSE2ProofStageDelta, ...],
    proposal_candidates: tuple[upright.UprightSE2ContinuousProposalCandidate, ...] = (),
) -> upright.UprightSE2ContinuousProofMaterial:
    """Seal one complete continuous tree/frontier/ledger proof transport."""

    ordered_rows = _sorted_rows(rows)
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
        canonical_total_resource_usage=_aggregate_usage(
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


def _synthetic_continuous_incomplete_proof(
    compilation: upright.UprightSE2ContinuousCompilation,
    *,
    reason: str,
    exhausted: bool,
) -> upright.UprightSE2ContinuousProofMaterial:
    """Retain pre-owner mismatch/config work as typed unresolved evidence."""

    request = compilation.source_solve_request
    root = compilation.compiled_cells[0]
    kind = (
        upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT
        if exhausted
        else upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE
    )
    evaluation = _continuous_transport_owner_outcome(
        request=request,
        cell=root,
        outcome=None,
        evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        label=reason,
        forced_kind=kind,
        atomic_steps=0,
    )
    row = upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=root,
        owner_evaluations=(evaluation,),
        leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
        complete_domain_empty=False,
    )
    return _continuous_proof_material(
        compilation,
        (row,),
        (_continuous_stage_for_cell(root, (evaluation,)),),
    )


def _continuous_proof_envelope(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ContinuousProofMaterial,
) -> ProofMaterialEnvelope:
    return ProofMaterialEnvelope.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_CONTINUOUS_BACKEND_REF,
        proof_material_definition_ref=(
            upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_DEFINITION_REF
        ),
        payload_schema_ref=upright.UPRIGHT_SE2_CONTINUOUS_PROOF_MATERIAL_PAYLOAD_SCHEMA_REF,
        typed_payload=(upright.encode_upright_se2_continuous_proof_material(proof),),
        artifact_refs=(),
    )


def _continuous_unknown_submission(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
    reason_claim_definition_ref: str,
    *,
    exhausted: bool,
) -> BackendUnknownEvidence:
    """Emit untrusted continuous UNKNOWN with complete typed proof evidence."""

    proof = _synthetic_continuous_incomplete_proof(
        compilation,
        reason=reason_claim_definition_ref.rsplit("/", maxsplit=1)[-1],
        exhausted=exhausted,
    )
    envelope = _continuous_proof_envelope(compilation, selection, proof)
    return BackendUnknownEvidence.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_CONTINUOUS_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        resource_usage=proof.total_resource_usage,
        reason_claim_definition_ref=reason_claim_definition_ref,
    )


def _continuous_outcome_kind(
    kind: ContinuousYawIntervalKindV4,
) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    """Map only the retained V4 outcome alphabet into proof transport."""

    return {
        ContinuousYawIntervalKindV4.EXACT: upright.UprightSE2RetainedOwnerOutcomeKind.EXACT,
        ContinuousYawIntervalKindV4.NUMERIC_GAP: upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP,
        ContinuousYawIntervalKindV4.RESOURCE_LIMIT: upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT,
        ContinuousYawIntervalKindV4.UNSUPPORTED: upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED,
    }[kind]


def _continuous_transport_owner_outcome(
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
    """Transport one already-produced V4 owner result through the compiler seam."""

    if forced_kind is None:
        if outcome is None:
            raise TypeError("continuous retained owner outcome is required")
        kernel_kind = getattr(outcome, "kind", None)
        if type(kernel_kind) is not ContinuousYawIntervalKindV4:
            raise TypeError("retained owner returned an unknown continuous outcome")
        proof_kind = _continuous_outcome_kind(kernel_kind)
        raw_rows = tuple(getattr(outcome, "proof_rows", ()))
        raw_findings = tuple(getattr(outcome, "finding_codes", ()))
        bounds = getattr(outcome, "bounds", None)
        if proof_kind is upright.UprightSE2RetainedOwnerOutcomeKind.EXACT and bounds is None:
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
        resource_delta=_resource_usage(
            request,
            used=float(atomic_steps),
            exhausted=(
                proof_kind
                is upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT
            ),
        ),
        label=label,
        exact_bound_value=bounds,
        additional_exact_bounds=additional_exact_bounds,
        continuous_exact_rationals=True,
    )


def _continuous_stage_for_cell(
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
) -> upright.UprightSE2ProofStageDelta:
    """Bind all retained continuous-owner calls for one proof cell to one stage."""

    ordered = tuple(sorted(evaluations, key=canonical_json_bytes))
    return upright.UprightSE2ProofStageDelta.seal(
        stage_ref=(
            "stage:spatialcf/upright-se2/continuous-retained-evaluation/"
            f"{cell.compiled_cell_sha256}"
        ),
        owner_evaluations=ordered,
        resource_delta=_aggregate_usage(
            tuple(evaluation.resource_delta for evaluation in ordered)
        ),
    )


def _continuous_closed_xy_cell(
    cell: upright.UprightSE2CompiledCell,
) -> ClosedXYCellV3:
    return ClosedXYCellV3(
        cell.x_lower.as_fraction,
        cell.x_upper.as_fraction,
        cell.y_lower.as_fraction,
        cell.y_upper.as_fraction,
    )


def _canonical_continuous_turn(value: Fraction) -> Fraction:
    """Choose the registered half-open representative without doing geometry."""

    while value < Fraction(-1, 2):
        value += 1
    while value >= Fraction(1, 2):
        value -= 1
    return value


def _exact_float(value: Fraction, *, label: str) -> float:
    """Refuse a non-lossless binary64 bridge before calling a retained owner."""

    result = float(value)
    if Fraction.from_float(result) != value:
        raise ArithmeticError(f"numeric gap: {label} is not an exact binary64 dyadic")
    return 0.0 if result == 0.0 else result


def _continuous_yaw_owner_inputs(
    compilation: upright.UprightSE2ContinuousCompilation,
    cell: upright.UprightSE2CompiledCell,
    budget: SO2AtomicBudgetV2,
) -> tuple[object, object | None]:
    """Compile one proof cell's yaw owner input from its authorized root/lift."""

    root = compilation.compiled_cells[0]
    if canonical_json_bytes(cell.yaw_interval) == canonical_json_bytes(root.yaw_interval):
        yaw_domain = compilation.operation.yaw_domain
    else:
        lower = cell.yaw_interval.lower.as_fraction
        upper = cell.yaw_interval.upper.as_fraction
        sweep = upper - lower
        if not Fraction() <= sweep < Fraction(1):
            raise ArithmeticError("numeric gap: lifted child sweep is outside ARC")
        yaw_domain = upright.ContinuousYawArc(
            start_angle=upright.CanonicalSO2Angle(
                turns=_exact_float(
                    _canonical_continuous_turn(lower), label="lifted child start"
                )
            ),
            ccw_sweep_turns=_exact_float(sweep, label="lifted child sweep"),
        )
    lift = compile_continuous_yaw_lift_v4(yaw_domain, atomic_budget=budget)
    if lift.kind is not ContinuousYawIntervalKindV4.EXACT:
        return lift, None
    assert lift.bounds is not None
    return lift, compile_lifted_turn_sin_cos_bounds_v4(
        lift.bounds,
        atomic_budget=budget,
    )


def _continuous_static_yaw_owner_inputs(
    budget: SO2AtomicBudgetV2,
) -> tuple[object, object | None]:
    """Compile the one retained static world-frame zero-yaw box basis."""

    lift = compile_continuous_yaw_lift_v4(
        upright.ContinuousYawArc(
            start_angle=upright.CanonicalSO2Angle(turns=0.0),
            ccw_sweep_turns=0.0,
        ),
        atomic_budget=budget,
    )
    if lift.kind is not ContinuousYawIntervalKindV4.EXACT:
        return lift, None
    assert lift.bounds is not None
    return lift, compile_lifted_turn_sin_cos_bounds_v4(
        lift.bounds,
        atomic_budget=budget,
    )


def _continuous_classification(
    cell_outcome: object | None,
    visibility_outcome: object | None,
) -> upright.UprightSE2ProofLeafDisposition | None:
    """Classify only retained exact V4 outer/inner evidence."""

    if (
        cell_outcome is None
        or visibility_outcome is None
        or getattr(cell_outcome, "kind", None)
        is not ContinuousYawIntervalKindV4.EXACT
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


def _continuous_objective(
    bounds: object,
) -> upright.UprightSE2ProposalPointObjective:
    """Copy the V4 retained objective intervals without reconstructing a term."""

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


def _continuous_objective_interval(bounds: object) -> tuple[Fraction, Fraction]:
    """Read the retained weighted interval for deterministic strict pruning."""

    terms = bounds.weighted_objective_terms
    return (
        sum((lower for _term, lower, _upper in terms), start=Fraction(0)),
        sum((upper for _term, _lower, upper in terms), start=Fraction(0)),
    )


def _continuous_point_cell(
    cell: upright.UprightSE2CompiledCell,
) -> upright.UprightSE2CompiledCell:
    """Choose one strict-interior exact-dyadic `(x, y, u)` witness point."""

    x = (
        cell.x_lower
        if cell.x_lower == cell.x_upper
        else _dyadic((cell.x_lower.as_fraction + cell.x_upper.as_fraction) / 2)
    )
    y = (
        cell.y_lower
        if cell.y_lower == cell.y_upper
        else _dyadic((cell.y_lower.as_fraction + cell.y_upper.as_fraction) / 2)
    )
    u = (
        cell.yaw_interval.lower
        if cell.yaw_interval.lower == cell.yaw_interval.upper
        else _dyadic(
            (
                cell.yaw_interval.lower.as_fraction
                + cell.yaw_interval.upper.as_fraction
            )
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


def _continuous_split_exact_dyadic_cell(
    cell: upright.UprightSE2CompiledCell,
    *,
    depth: int,
) -> tuple[upright.UprightSE2CompiledCell, ...]:
    """Split exactly one registered axis in cyclic X, Y, U order."""

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
        *, lower: upright.ExactDyadic,
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
        elif side == "upper" and cell.yaw_interval.seam_ownership == "UPPER_OWNS_ENDPOINT":
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
    midpoint = _dyadic((lower.as_fraction + upper.as_fraction) / 2)
    return (
        child(lower=lower, upper=midpoint, side="lower"),
        child(lower=midpoint, upper=upper, side="upper"),
    )


def _continuous_forced_kind(error: BaseException) -> upright.UprightSE2RetainedOwnerOutcomeKind:
    """Keep bridge/owner failures typed and conservative at the backend boundary."""

    if isinstance(error, ArithmeticError):
        return upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP
    if "unsupported" in str(error).lower():
        return upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED
    return upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE


def _evaluate_continuous_owner_cell(
    *,
    compilation: upright.UprightSE2ContinuousCompilation,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    budget: SO2AtomicBudgetV2,
) -> tuple[
    tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
    object | None,
    object | None,
]:
    """Invoke only Task 6.1 owners for one compiler-bridged continuous cell."""

    evaluations: list[upright.UprightSE2RetainedOwnerEvaluation] = []

    def record(
        label: str,
        outcome: object,
        start_used: int,
        *,
        additional_exact_bounds: tuple[TypedValue, ...] = (),
    ) -> object:
        evaluations.append(
            _continuous_transport_owner_outcome(
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
        lift, yaw_bounds = _continuous_yaw_owner_inputs(compilation, cell, budget)
        record("continuous-yaw-lift", lift, start)
        if getattr(lift, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        assert yaw_bounds is not None
        start = budget.used
        record("continuous-yaw-sin-cos", yaw_bounds, start)
        if getattr(yaw_bounds, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        assert yaw_bounds.bounds is not None

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
            assert outcome.bounds is not None
            dynamic_subjects.append(outcome.bounds)

        start = budget.used
        static_lift, static_yaw_bounds = _continuous_static_yaw_owner_inputs(budget)
        record("continuous-static-yaw-lift", static_lift, start)
        if getattr(static_lift, "kind", None) is not ContinuousYawIntervalKindV4.EXACT:
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        assert static_yaw_bounds is not None
        start = budget.used
        record("continuous-static-yaw-sin-cos", static_yaw_bounds, start)
        if (
            getattr(static_yaw_bounds, "kind", None)
            is not ContinuousYawIntervalKindV4.EXACT
        ):
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None
        assert static_yaw_bounds.bounds is not None
        static_cell = ClosedXYCellV3(
            Fraction(), Fraction(), Fraction(), Fraction()
        )

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
            return outcome.bounds if outcome.kind is ContinuousYawIntervalKindV4.EXACT else None

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
        assert visual_subject_outcome.bounds is not None
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
            return tuple(sorted(evaluations, key=canonical_json_bytes)), None, visibility

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
            assert compound.bounds is not None
            point_objective_bound = (
                upright._retained_point_objective_value(
                    _continuous_objective(compound.bounds)
                ),
            )
        record(
            "continuous-compound-cell",
            compound,
            start,
            additional_exact_bounds=point_objective_bound,
        )
        return tuple(sorted(evaluations, key=canonical_json_bytes)), compound, visibility
    except (TypeError, ValueError, ArithmeticError, OverflowError) as error:
        evaluations.append(
            _continuous_transport_owner_outcome(
                request=request,
                cell=cell,
                outcome=None,
                evaluator_capability_ref=(
                    upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF
                ),
                label="continuous-bridge-or-owner-incomplete",
                forced_kind=_continuous_forced_kind(error),
                atomic_steps=0,
            )
        )
        return tuple(sorted(evaluations, key=canonical_json_bytes)), None, None


def _continuous_fallback_budget(request: CounterfactualSolveRequest) -> SO2AtomicBudgetV2:
    """Keep a failed compiler bridge representable without ambient defaults."""

    limits = request.resource_policy.limits
    if len(limits) != 1:
        raise ValueError("continuous upright se2 requires one resource limit")
    finite_limit = limits[0].finite_limit
    if type(finite_limit) is not float or finite_limit < 1.0:
        raise ValueError("continuous upright se2 resource limit is unusable")
    return SO2AtomicBudgetV2(limit=int(finite_limit))


def _continuous_unresolved_row(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
    label: str,
) -> tuple[
    upright.UprightSE2ProofCellEvaluation,
    upright.UprightSE2ProofStageDelta,
]:
    """Attach explicit nonexact evidence before declaring a final frontier leaf."""

    has_nonexact = any(
        evaluation.outcome_kind is not upright.UprightSE2RetainedOwnerOutcomeKind.EXACT
        for evaluation in evaluations
    )
    owner_evaluations = evaluations
    if not has_nonexact:
        owner_evaluations = tuple(
            sorted(
                (
                    *evaluations,
                    _continuous_transport_owner_outcome(
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
    row = upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=cell,
        owner_evaluations=owner_evaluations,
        leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
        complete_domain_empty=False,
    )
    return row, _continuous_stage_for_cell(cell, owner_evaluations)


def _continuous_incomplete_parent_row(
    *,
    request: CounterfactualSolveRequest,
    cell: upright.UprightSE2CompiledCell,
    evaluations: tuple[upright.UprightSE2RetainedOwnerEvaluation, ...],
    label: str,
) -> tuple[
    upright.UprightSE2ProofCellEvaluation,
    upright.UprightSE2ProofStageDelta,
]:
    """Keep an inward interval unresolved when its concrete replay is not proved."""

    incomplete = _continuous_transport_owner_outcome(
        request=request,
        cell=cell,
        outcome=None,
        evaluator_capability_ref=upright.UPRIGHT_SE2_PREDICATE_EVALUATOR_CAPABILITY_REF,
        label=label,
        forced_kind=upright.UprightSE2RetainedOwnerOutcomeKind.INCOMPLETE,
        atomic_steps=0,
    )
    owner_evaluations = tuple(
        sorted((*evaluations, incomplete), key=canonical_json_bytes)
    )
    row = upright.UprightSE2ProofCellEvaluation.seal(
        compiled_cell=cell,
        owner_evaluations=owner_evaluations,
        leaf_disposition=upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
        complete_domain_empty=False,
    )
    return row, _continuous_stage_for_cell(cell, owner_evaluations)


def _continuous_candidate_from_point(
    *,
    compilation: upright.UprightSE2ContinuousCompilation,
    final_inward_cell: upright.UprightSE2ProofCellEvaluation,
    point_row: upright.UprightSE2ProofCellEvaluation,
    point_bounds: object,
) -> upright.UprightSE2ContinuousProposalCandidate:
    """Select a compiler-materialized endpoint from retained exact point output."""

    point_cell = point_row.compiled_cell
    objective = _continuous_objective(point_bounds)
    point = Vec2(
        x=float(point_cell.x_lower.as_fraction),
        y=float(point_cell.y_lower.as_fraction),
    )
    selected_yaw = point_cell.yaw_interval.lower
    endpoint = materialize_upright_se2_continuous_endpoint(
        compilation,
        point,
        selected_yaw,
    )
    point_evaluation = upright.UprightSE2ProposalPointEvaluation.seal(
        point_cell_evaluation=point_row,
        point_objective=objective,
    )
    return upright.UprightSE2ContinuousProposalCandidate.seal(
        final_inward_cell=final_inward_cell,
        selected_translation_xy_m=point,
        selected_lifted_yaw=selected_yaw,
        point_evaluation=point_evaluation,
        point_objective=objective,
        materialized_endpoint=endpoint,
        program=endpoint.program,
    )


def _continuous_requested_gap(request: CounterfactualSolveRequest) -> Fraction:
    """Read the request-bound finite-gap policy; never invent a solver default."""

    return upright.decode_upright_se2_solve_policy_definition_payload(
        request.solve_policy_definition_bundle
    ).requested_gap.as_fraction


def _continuous_bounds_authorize_prune(
    *,
    cell_lower: Fraction,
    incumbent_upper: Fraction | None,
    requested_gap: Fraction,
) -> bool:
    """Apply the registered strict-exact or closed finite-gap prune comparison."""

    if incumbent_upper is None:
        return False
    return cell_lower > incumbent_upper or (
        requested_gap > 0 and cell_lower >= incumbent_upper - requested_gap
    )


def _evaluate_continuous_compilation(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
) -> _Submission:
    """Run bounded deterministic V4-only `(x, y, u)` branch-and-bound search."""

    request = compilation.source_solve_request
    budget: SO2AtomicBudgetV2 | None = None
    rows: list[upright.UprightSE2ProofCellEvaluation] = []
    stages: list[upright.UprightSE2ProofStageDelta] = []
    candidates: list[upright.UprightSE2ContinuousProposalCandidate] = []
    requested_gap = _continuous_requested_gap(request)

    def ensure_budget(cell: upright.UprightSE2CompiledCell) -> SO2AtomicBudgetV2:
        nonlocal budget
        if budget is not None:
            return budget
        try:
            bridge = build_upright_se2_continuous_evaluation_inputs(compilation, cell)
            budget = SO2AtomicBudgetV2(limit=bridge.resource_atomic_step_limit)
        except (TypeError, ValueError, ArithmeticError):
            budget = _continuous_fallback_budget(request)
        return budget

    def incumbent_upper() -> Fraction | None:
        if not candidates:
            return None
        return min(
            candidate.point_objective.total_upper.as_fraction for candidate in candidates
        )

    def evaluate_cell(cell: upright.UprightSE2CompiledCell, depth: int) -> None:
        active_budget = ensure_budget(cell)
        owner_evaluations, compound, visibility = _evaluate_continuous_owner_cell(
            compilation=compilation,
            request=request,
            cell=cell,
            budget=active_budget,
        )
        classification = _continuous_classification(compound, visibility)
        point_cell = _continuous_point_cell(cell)
        if (
            canonical_json_bytes(point_cell) != canonical_json_bytes(cell)
            and classification
            in (
                upright.UprightSE2ProofLeafDisposition.UNRESOLVED,
                None,
            )
        ):
            point_evaluations, point_compound, point_visibility = (
                _evaluate_continuous_owner_cell(
                    compilation=compilation,
                    request=request,
                    cell=point_cell,
                    budget=active_budget,
                )
            )
            point_classification = _continuous_classification(
                point_compound, point_visibility
            )
            point_row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=point_cell,
                owner_evaluations=point_evaluations,
                leaf_disposition=None,
                complete_domain_empty=None,
            )
            rows.append(point_row)
            stages.append(_continuous_stage_for_cell(point_cell, point_evaluations))
            if (
                point_classification
                is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
                and point_compound is not None
                and getattr(point_compound, "bounds", None) is not None
            ):
                parent_row, parent_stage = _continuous_incomplete_parent_row(
                    request=request,
                    cell=cell,
                    evaluations=owner_evaluations,
                    label="continuous-feasible-incomplete-witness",
                )
                rows.append(parent_row)
                stages.append(parent_stage)
                try:
                    candidates.append(
                        _continuous_candidate_from_point(
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
            row, stage = _continuous_unresolved_row(
                request=request,
                cell=cell,
                evaluations=owner_evaluations,
                label="continuous-owner-nonexact",
            )
            rows.append(row)
            stages.append(stage)
            return
        if compound is None or getattr(compound, "bounds", None) is None:
            row, stage = _continuous_unresolved_row(
                request=request,
                cell=cell,
                evaluations=owner_evaluations,
                label="continuous-missing-compound-bounds",
            )
            rows.append(row)
            stages.append(stage)
            return
        bounds = compound.bounds

        if classification is upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE:
            row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=owner_evaluations,
                leaf_disposition=classification,
                complete_domain_empty=True,
            )
            rows.append(row)
            stages.append(_continuous_stage_for_cell(cell, owner_evaluations))
            return

        if classification is None:
            lower, _upper = _continuous_objective_interval(bounds)
            incumbent = incumbent_upper()
            if _continuous_bounds_authorize_prune(
                cell_lower=lower,
                incumbent_upper=incumbent,
                requested_gap=requested_gap,
            ):
                row = upright.UprightSE2ProofCellEvaluation.seal(
                    compiled_cell=cell,
                    owner_evaluations=owner_evaluations,
                    leaf_disposition=upright.UprightSE2ProofLeafDisposition.PRUNED,
                    complete_domain_empty=False,
                )
                rows.append(row)
                stages.append(_continuous_stage_for_cell(cell, owner_evaluations))
                return
            children = _continuous_split_exact_dyadic_cell(cell, depth=depth)
            if children and depth < _MAX_CONTINUOUS_EXACT_DYADIC_REFINEMENT_DEPTH:
                internal = upright.UprightSE2ProofCellEvaluation.seal(
                    compiled_cell=cell,
                    owner_evaluations=owner_evaluations,
                    leaf_disposition=None,
                    complete_domain_empty=None,
                )
                rows.append(internal)
                stages.append(_continuous_stage_for_cell(cell, owner_evaluations))
                for child in children:
                    evaluate_cell(child, depth + 1)
                return
            row, stage = _continuous_unresolved_row(
                request=request,
                cell=cell,
                evaluations=owner_evaluations,
                label="continuous-finite-refinement-miss",
            )
            rows.append(row)
            stages.append(stage)
            return
        assert classification is upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        if canonical_json_bytes(point_cell) == canonical_json_bytes(cell):
            final_row = upright.UprightSE2ProofCellEvaluation.seal(
                compiled_cell=cell,
                owner_evaluations=owner_evaluations,
                leaf_disposition=upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE,
                complete_domain_empty=False,
            )
            rows.append(final_row)
            stages.append(_continuous_stage_for_cell(cell, owner_evaluations))
            try:
                candidates.append(
                    _continuous_candidate_from_point(
                        compilation=compilation,
                        final_inward_cell=final_row,
                        point_row=final_row,
                        point_bounds=bounds,
                    )
                )
            except (TypeError, ValueError, ArithmeticError):
                # A materialization failure cannot erase exact geometry evidence,
                # so retain a separate unresolved alias rather than certify it.
                rows.pop()
                stages.pop()
                row, stage = _continuous_incomplete_parent_row(
                    request=request,
                    cell=cell,
                    evaluations=owner_evaluations,
                    label="continuous-point-materialization-incomplete",
                )
                rows.append(row)
                stages.append(stage)
            return

        point_evaluations, point_compound, point_visibility = (
            _evaluate_continuous_owner_cell(
                compilation=compilation,
                request=request,
                cell=point_cell,
                budget=active_budget,
            )
        )
        point_classification = _continuous_classification(
            point_compound, point_visibility
        )
        point_row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=point_cell,
            owner_evaluations=point_evaluations,
            leaf_disposition=None,
            complete_domain_empty=None,
        )
        rows.append(point_row)
        stages.append(_continuous_stage_for_cell(point_cell, point_evaluations))
        if (
            point_classification
            is not upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
            or point_compound is None
            or getattr(point_compound, "bounds", None) is None
        ):
            row, stage = _continuous_incomplete_parent_row(
                request=request,
                cell=cell,
                evaluations=owner_evaluations,
                label="continuous-inward-point-replay-incomplete",
            )
            rows.append(row)
            stages.append(stage)
            return
        final_row = upright.UprightSE2ProofCellEvaluation.seal(
            compiled_cell=cell,
            owner_evaluations=owner_evaluations,
            leaf_disposition=upright.UprightSE2ProofLeafDisposition.INWARD_FEASIBLE,
            complete_domain_empty=False,
        )
        rows.append(final_row)
        stages.append(_continuous_stage_for_cell(cell, owner_evaluations))
        try:
            candidates.append(
                _continuous_candidate_from_point(
                    compilation=compilation,
                    final_inward_cell=final_row,
                    point_row=point_row,
                    point_bounds=point_compound.bounds,
                )
            )
        except (TypeError, ValueError, ArithmeticError):
            rows.pop()
            stages.pop()
            row, stage = _continuous_incomplete_parent_row(
                request=request,
                cell=cell,
                evaluations=owner_evaluations,
                label="continuous-point-materialization-incomplete",
            )
            rows.append(row)
            stages.append(stage)

    evaluate_cell(compilation.compiled_cells[0], 0)
    proof = _continuous_proof_material(
        compilation,
        tuple(rows),
        tuple(stages),
        tuple(candidates),
    )
    if proof.proposal_candidates:
        return _continuous_proposal_submission(
            compilation, selection, proof, proof.proposal_candidates[0]
        )
    leaves = tuple(
        row for row in proof.evaluated_cells if row.leaf_disposition is not None
    )
    if leaves and all(
        row.leaf_disposition
        is upright.UprightSE2ProofLeafDisposition.OUTWARD_INFEASIBLE
        for row in leaves
    ):
        return _continuous_unsat_submission(compilation, selection, proof)
    outcome_kinds = {
        evaluation.outcome_kind
        for row in proof.evaluated_cells
        for evaluation in row.owner_evaluations
    }
    reason = (
        request.resource_policy.exhaustion_claim_ref
        if upright.UprightSE2RetainedOwnerOutcomeKind.RESOURCE_LIMIT in outcome_kinds
        else (
            _UNKNOWN_NUMERIC_REF
            if upright.UprightSE2RetainedOwnerOutcomeKind.NUMERIC_GAP in outcome_kinds
            else (
                _UNKNOWN_UNSUPPORTED_REF
                if upright.UprightSE2RetainedOwnerOutcomeKind.UNSUPPORTED
                in outcome_kinds
                else _UNKNOWN_INCOMPLETE_REF
            )
        )
    )
    return _continuous_unknown_from_proof(compilation, selection, proof, reason)


def _continuous_unsat_submission(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ContinuousProofMaterial,
) -> BackendCompleteUnsatEvidence:
    """Emit complete continuous UNSAT only after every proof leaf is outer-empty."""

    envelope = _continuous_proof_envelope(compilation, selection, proof)
    return BackendCompleteUnsatEvidence.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_CONTINUOUS_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        resource_usage=proof.total_resource_usage,
        complete_domain_claim_definition_ref=_UNSAT_CLAIM_REF,
        authorized_domain_sha256=(
            compilation.endpoint_construction_recipe.translation_domain_sha256
        ),
        complete_domain_coverage_artifact_sha256=(
            proof.coverage_artifact.coverage_artifact_sha256
        ),
    )


def _continuous_proposal_submission(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ContinuousProofMaterial,
    selected: upright.UprightSE2ContinuousProposalCandidate,
) -> BackendProposalSubmission:
    """Wrap an untrusted compiler-materialized continuous witness proposal."""

    envelope = _continuous_proof_envelope(compilation, selection, proof)
    proposal = BackendProposal.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_CONTINUOUS_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proposal_claim_definition_ref=_PROPOSAL_CLAIM_REF,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        program_sha256=selected.program.program_sha256,
        after_scene_state_sha256=selected.program.after_scene_state_sha256,
        objective_lower_bound=float(selected.point_objective.total_lower.as_fraction),
        objective_upper_bound=float(selected.point_objective.total_upper.as_fraction),
        resource_usage=proof.total_resource_usage,
    )
    return BackendProposalSubmission.seal(
        proposal=proposal,
        backend_proposal_sha256=proposal.backend_proposal_sha256,
    )


def _continuous_unknown_from_proof(
    compilation: upright.UprightSE2ContinuousCompilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ContinuousProofMaterial,
    reason_claim_definition_ref: str,
) -> BackendUnknownEvidence:
    """Keep complete explored evidence when a continuous terminal is UNKNOWN."""

    envelope = _continuous_proof_envelope(compilation, selection, proof)
    return BackendUnknownEvidence.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_CONTINUOUS_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        resource_usage=proof.total_resource_usage,
        reason_claim_definition_ref=reason_claim_definition_ref,
    )


def _unsat_submission(
    compilation: upright.UprightSE2Compilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ProofMaterial,
) -> BackendCompleteUnsatEvidence:
    """Emit complete UNSAT only when every final exact leaf is outward-empty."""

    envelope = _proof_envelope(compilation, selection, proof)
    return BackendCompleteUnsatEvidence.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        resource_usage=proof.total_resource_usage,
        complete_domain_claim_definition_ref=_UNSAT_CLAIM_REF,
        authorized_domain_sha256=(
            compilation.endpoint_construction_recipe.translation_domain_sha256
        ),
        complete_domain_coverage_artifact_sha256=(
            proof.coverage_artifact.coverage_artifact_sha256
        ),
    )


def _proposal_submission(
    compilation: upright.UprightSE2Compilation,
    selection: BackendSelectionRecord,
    proof: upright.UprightSE2ProofMaterial,
    selected: upright.UprightSE2ProposalCandidate,
) -> BackendProposalSubmission:
    """Materialize only an inward-proven in-domain endpoint via the public recipe."""

    envelope = _proof_envelope(compilation, selection, proof)
    proposal = BackendProposal.seal(
        semantic_problem_sha256=compilation.source_solve_request.semantic_problem_sha256,
        solve_request_sha256=compilation.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=_BACKEND_REF,
        proposal_backend_owner_ref=upright.UPRIGHT_SE2_BACKEND_OWNER_REF,
        proposal_backend_capability_ref=upright.UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
        proposal_backend_build_sha256=_BACKEND_BUILD_SHA256,
        proposal_claim_definition_ref=_PROPOSAL_CLAIM_REF,
        proof_material=envelope,
        proof_material_sha256=envelope.proof_material_sha256,
        program_sha256=selected.program.program_sha256,
        after_scene_state_sha256=selected.program.after_scene_state_sha256,
        objective_lower_bound=float(selected.point_objective.total_lower.as_fraction),
        objective_upper_bound=float(selected.point_objective.total_upper.as_fraction),
        resource_usage=proof.total_resource_usage,
    )
    return BackendProposalSubmission.seal(
        proposal=proposal,
        backend_proposal_sha256=proposal.backend_proposal_sha256,
    )
