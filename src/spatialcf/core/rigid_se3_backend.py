"""Untrusted M6 submission-v2 producer for the source-bound rigid profile."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from spatialcf.core._internal.kernels import rigid_se3 as kernel
from spatialcf.core.backends import match_backend_capabilities
from spatialcf.core.rigid_se3_compiler import (
    BACKEND_REF,
    BUILDS,
    OWNERS,
    RigidSE3CompiledProblem,
    RigidSE3MissingSource,
    RigidSE3RequestContext,
    _definition_field,
    _replay_validated,
    compile_rigid_se3,
    d,
    exact_source_from_cells,
    source_cells_from_state,
    validate_rigid_se3_request,
)
from spatialcf.domain.counterfactual import CounterfactualSolveRequest
from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposal,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendUnknownEvidence,
    CapabilityMatch,
    ProofMaterialEnvelope,
    ResourceUsage,
)
from spatialcf.domain.profiles import CounterfactualSolverConfig
from spatialcf.domain.rigid_se3 import (
    MAX_COVERAGE_BYTES,
    MAX_COVERAGE_ROWS,
    MAX_PROOF_OUTPUT_BOUND_BYTES,
    M6IngressLimit,
    RigidSE3Compilation,
    RigidSE3Coverage,
    RigidSE3Domain,
    RigidSE3FailedPrefix,
    RigidSE3Ledger,
    RigidSE3MemberChoice,
    RigidSE3ProgramTrace,
    RigidSE3ProofMaterial,
    RigidSE3ProposalPolicyPayload,
    RigidSE3Rational,
    RigidSE3StepChoice,
    RigidSE3TupleEvidence,
    decode_tree,
    encode_proof_material,
    m6_output_upper_bound,
    schema,
)
from spatialcf.domain.serialization import canonical_json_bytes

__all__ = ("RigidSE3Backend", "RigidSE3CompiledProblem")


def _usage(request: CounterfactualSolveRequest, *,
           compilation: RigidSE3Compilation | None = None,
           solve_ledger: RigidSE3Ledger | None = None,
           allocation: bool = False) -> ResourceUsage:
    values = {
        row.definition_ref: row.finite_limit if allocation else 0.0
        for row in request.resource_policy.limits
    }
    ledgers = tuple(row for row in (
        compilation.ledger if compilation is not None else None,
        solve_ledger,
    ) if row is not None)
    for ledger in ledgers:
        operations = sum(event.amount for event in ledger.events)
        stage_key = d(f"resource/{ledger.stage.lower()}_operations")
        values[stage_key] = max(values[stage_key], float(operations))
        values[d("resource/max_exact_operations")] = max(
            values[d("resource/max_exact_operations")], float(operations),
        )
        values[d("resource/numeric_bits")] = max(
            values[d("resource/numeric_bits")], float(ledger.peak_numeric_bits),
        )
        for kind, limit_name in (
            ("ENTITY", "max_entities"), ("PRIMITIVE", "max_primitives"),
            ("JOINT", "max_joints"), ("PROGRAM", "max_programs"),
            ("EVALUATED_TUPLE", "max_evaluated_tuples"),
        ):
            used = sum(event.amount for event in ledger.events if event.kind == kind)
            values[d(f"resource/{limit_name}")] = max(
                values[d(f"resource/{limit_name}")], float(used),
            )
    if compilation is not None:
        values[d("resource/max_steps")] = float(max(
            len(row.steps) for row in compilation.domain.program_skeletons
        ))
        if compilation.domain.finite:
            values[d("resource/max_domain_products")] = float(sum(
                _program_count(compilation.domain, row)
                for row in compilation.domain.program_skeletons
            ))
    return ResourceUsage.model_validate({
        "accounting_claim_definition_ref": d("accounting"),
        "entries": tuple({"resource_definition_ref": ref, "used": used}
                      for ref, used in sorted(values.items())),
        "exhausted": bool((compilation and compilation.ledger.reason == "RESOURCE_LIMIT")
                       or (solve_ledger and solve_ledger.reason == "RESOURCE_LIMIT")),
    }, strict=True)


class RigidSE3Backend:
    def inspect(self, request: CounterfactualSolveRequest):
        return self._inspect_validated(validate_rigid_se3_request(request))

    def _inspect_validated(self, context: RigidSE3RequestContext):
        request = context.request
        args = context.registry_arguments
        return match_backend_capabilities(
            request, args["action_space_profile"], args["semantics_profile"],
            args["predicate_definitions"], args["operator_definitions"],
            request.backend_descriptor_bundle.backend_descriptors[0],
        )

    def select(self, request: CounterfactualSolveRequest) -> BackendSelectionRecord:
        return self._select_validated(validate_rigid_se3_request(request))

    def _select_validated(self, context: RigidSE3RequestContext) -> BackendSelectionRecord:
        request = context.request
        row = self._inspect_validated(context)
        selected = isinstance(row, CapabilityMatch)
        descriptor = request.backend_descriptor_bundle.backend_descriptors[0]
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
            ordered_candidate_backend_refs=(BACKEND_REF,), capability_rows=(row,),
            selection_disposition="SELECTED" if selected else "NO_SELECTION",
            selection_disposition_claim_ref=d("selection-disposition"),
            selected_backend_ref=BACKEND_REF if selected else None,
            selected_backend_descriptor_sha256=(
                descriptor.backend_descriptor_sha256 if selected else None
            ),
            resource_allocation=_usage(request, allocation=True) if selected else None,
            deterministic_selection_reason_ref=d("selection-reason"),
        )

    def compile(self, request: CounterfactualSolveRequest) -> RigidSE3CompiledProblem:
        context = validate_rigid_se3_request(request)
        if self._select_validated(context).selection_disposition != "SELECTED":
            raise ValueError("NO_SELECTION cannot compile a selected submission")
        try:
            compilation = compile_rigid_se3(request)
        except RigidSE3MissingSource:
            return RigidSE3CompiledProblem(
                request, None,
                RigidSE3Ledger(
                    stage="COMPILE", reason="UNSUPPORTED",
                    first_unprocessed_item="item:compile:missing-source-fact",
                ), d("missing-source-fact"),
            )
        return RigidSE3CompiledProblem(
            request, compilation if compilation.ledger.reason == "NONE" else None,
            compilation.ledger if compilation.ledger.reason != "NONE" else None,
        )

    def solve_submission(self, compiled: RigidSE3CompiledProblem,
                         config: CounterfactualSolverConfig):
        if type(compiled) is not RigidSE3CompiledProblem:
            raise TypeError("expected exact M6 compiled problem")
        request = compiled.source_solve_request
        context = validate_rigid_se3_request(request)
        if type(config) is not type(request.solver_config) or config != request.solver_config:
            raise ValueError("M6 solver config substitution")
        selection = self._select_validated(context)
        if selection.selection_disposition != "SELECTED":
            raise ValueError("M6 submission requires selected routing")
        fresh = self.compile(request)
        if fresh != compiled:
            raise ValueError("stale or altered M6 compiled problem")
        compilation = compiled.compilation
        failure = compiled.failure_ledger
        coverage = None
        winner = None
        exact_cost = None
        if compilation is not None:
            budget = kernel.ExactBudget(context.limits, "SOLVE")
            if context.domain.finite:
                coverage, winner, exact_cost, failure = _search_finite(
                    context, compilation, budget,
                )
            else:
                winner, exact_cost, failure = _search_hints(
                    context, compilation, budget,
                )
        bounds = None
        if winner is not None:
            assert exact_cost is not None
            try:
                bounds = kernel.outward_float_bounds(exact_cost)
            except kernel.ExactKernelLimit:
                winner = None
                if failure is None or failure.reason == "UNSUPPORTED":
                    prior = coverage.ledger if coverage is not None else failure
                    failure = RigidSE3Ledger(
                        stage="SOLVE", events=prior.events if prior else (),
                        completed_items=prior.completed_items if prior else (),
                        peak_numeric_bits=prior.peak_numeric_bits if prior else 0,
                        reason="NUMERIC_GAP",
                        first_unprocessed_item="item:objective-transport",
                    )
        if m6_output_upper_bound((compilation, coverage, winner, failure)) > MAX_PROOF_OUTPUT_BOUND_BYTES:
            raise M6IngressLimit("M6 ingress emitted proof output ceiling exceeded")
        proof = RigidSE3ProofMaterial.seal(
            semantic_problem_sha256=request.semantic_problem_sha256,
            solve_request_sha256=request.solve_request_sha256,
            source_state_sha256=context.state_view.state.scene_state_sha256,
            domain_sha256=context.domain.domain_sha256,
            compilation=compilation, coverage=coverage, winner_trace=winner,
            swap_evidence=(), failure_ledger=failure,
        )
        envelope = ProofMaterialEnvelope.seal(
            semantic_problem_sha256=request.semantic_problem_sha256,
            solve_request_sha256=request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,
            proposal_backend_ref=BACKEND_REF,
            proof_material_definition_ref=d("proof-material"),
            payload_schema_ref=schema("proof-bytes"),
            typed_payload=(encode_proof_material(proof),), artifact_refs=(),
        )
        usage = _usage(request, compilation=compilation,
                       solve_ledger=failure or (coverage.ledger if coverage else None))
        common = {
            "semantic_problem_sha256": request.semantic_problem_sha256,
            "solve_request_sha256": request.solve_request_sha256,
            "backend_selection_record_sha256": selection.backend_selection_record_sha256,
            "proposal_backend_ref": BACKEND_REF,
            "proposal_backend_owner_ref": OWNERS["backend"],
            "proposal_backend_capability_ref": "capability:spatialcf/rigid-se3/1.0/solve",
            "proposal_backend_build_sha256": BUILDS["backend"],
            "proof_material": envelope,
            "proof_material_sha256": envelope.proof_material_sha256,
            "resource_usage": usage,
        }
        if winner is not None:
            assert bounds is not None
            lower, upper = bounds
            proposal = BackendProposal.seal(
                **common, proposal_claim_definition_ref=d("certified"),
                program_sha256=winner.program.program_sha256,
                after_scene_state_sha256=winner.after_state_sha256,
                objective_lower_bound=lower, objective_upper_bound=upper,
            )
            return BackendProposalSubmission.seal(
                proposal=proposal, backend_proposal_sha256=proposal.backend_proposal_sha256,
            )
        if (coverage is not None and coverage.ledger.reason == "NONE"
                and coverage.winner_tuple_id is None and failure is None):
            return BackendCompleteUnsatEvidence.seal(
                **common, complete_domain_claim_definition_ref=d("unsat"),
                authorized_domain_sha256=context.domain.domain_sha256,
                complete_domain_coverage_artifact_sha256=coverage.coverage_sha256,
            )
        reason_ref = compiled.failure_reason_ref or (
            d("continuous-domain-not-closed")
            if compilation is not None and not context.domain.finite
            and failure is not None and failure.reason == "UNSUPPORTED"
            else _unknown_reason(failure)
        )
        return BackendUnknownEvidence.seal(
            **common, reason_claim_definition_ref=reason_ref,
        )


def _unknown_reason(failure: RigidSE3Ledger | None) -> str:
    if failure is None:
        raise ValueError("M6 unknown requires a typed owner-supported reason")
    return d({
        "RESOURCE_LIMIT": "resource-exhausted",
        "NUMERIC_GAP": "numeric-gap",
        "UNSUPPORTED": "unsupported",
    }[failure.reason])


def _pose_transform(pose) -> kernel.Transform:
    return kernel.Transform(
        rotation=tuple(tuple(value.as_fraction for value in row)
                       for row in pose.rotation.rows),
        translation=pose.translation_m.fractions,
    )


def _trace_cost(trace: RigidSE3ProgramTrace, compilation: RigidSE3Compilation,
                context: RigidSE3RequestContext, budget: kernel.ExactBudget) -> Fraction:
    total = Fraction()
    previous = exact_source_from_cells(source_cells_from_state(compilation.source_state))
    for step in trace.steps:
        current = exact_source_from_cells(source_cells_from_state(step.after_state))
        total = budget.add(total, kernel.step_cost(
            {row.body_id: _pose_transform(row.world_pose) for row in previous.roots},
            {row.body_id: _pose_transform(row.world_pose) for row in current.roots},
            {row.joint_id: row for row in previous.joint_states},
            {row.joint_id: row for row in current.joint_states},
            {row.contact_id: row.mode for row in previous.contact_states},
            {row.contact_id: row.mode for row in current.contact_states},
            context.objective, budget,
        ))
        previous = current
    return total


def _better(cost: Fraction, program_bytes: bytes,
            previous_cost: Fraction | None, previous_bytes: bytes | None,
            budget: kernel.ExactBudget) -> bool:
    if previous_cost is None:
        return True
    assert previous_bytes is not None
    relation = budget.compare(cost, previous_cost)
    return relation < 0 or (relation == 0 and program_bytes < previous_bytes)


def _program_choices(domain: RigidSE3Domain, skeleton):
    # One index dimension per declared finite axis. A root contributes two
    # separate dimensions, so its translation×rotation product is never
    # allocated before the first evaluated-tuple charge.
    from spatialcf.domain.rigid_se3 import RigidSE3Pose

    roots = {row.root_body_id: row for row in domain.roots}
    joints = {row.joint_id: row for row in domain.joints}
    contacts = {row.contact_id: row for row in domain.contacts}
    ranges = []
    for step in skeleton.steps:
        for member in step.members:
            if member.kind == "SET_ROOT_SE3":
                row = roots[member.target_id]
                ranges.extend((range(len(row.translation.values)),
                               range(len(row.rotation.values))))
            elif member.kind == "SET_JOINT":
                row = joints[member.target_id]
                ranges.append(range(len(row.prismatic.values_m if row.prismatic
                                        else row.revolute.values)))
            else:
                ranges.append(range(len(contacts[member.target_id].modes)))

    for indexes in product(*ranges):
        cursor = 0
        steps = []
        for template in skeleton.steps:
            members = []
            for member in template.members:
                if member.kind == "SET_ROOT_SE3":
                    row = roots[member.target_id]
                    translation = row.translation.values[indexes[cursor]]
                    rotation = row.rotation.values[indexes[cursor + 1]]
                    cursor += 2
                    members.append(RigidSE3MemberChoice(
                        kind="SET_ROOT_SE3", target_id=member.target_id,
                        pose=RigidSE3Pose(frame="WORLD", translation_m=translation,
                                          rotation=rotation),
                    ))
                elif member.kind == "SET_JOINT":
                    row = joints[member.target_id]
                    index = indexes[cursor]
                    cursor += 1
                    if row.prismatic is not None:
                        members.append(RigidSE3MemberChoice(
                            kind="SET_JOINT", target_id=member.target_id,
                            offset_m=row.prismatic.values_m[index],
                        ))
                    else:
                        members.append(RigidSE3MemberChoice(
                            kind="SET_JOINT", target_id=member.target_id,
                            circle_point=row.revolute.values[index],
                        ))
                else:
                    row = contacts[member.target_id]
                    members.append(RigidSE3MemberChoice(
                        kind="SET_CONTACT_MODE", target_id=member.target_id,
                        mode=row.modes[indexes[cursor]],
                    ))
                    cursor += 1
            steps.append(RigidSE3StepChoice(members=tuple(members)))
        yield tuple(steps)


def _program_count(domain: RigidSE3Domain, skeleton) -> int:
    roots = {row.root_body_id: row for row in domain.roots}
    joints = {row.joint_id: row for row in domain.joints}
    contacts = {row.contact_id: row for row in domain.contacts}
    count = 1
    for step in skeleton.steps:
        for member in step.members:
            if member.kind == "SET_ROOT_SE3":
                row = roots[member.target_id]
                count *= len(row.translation.values) * len(row.rotation.values)
            elif member.kind == "SET_JOINT":
                row = joints[member.target_id]
                count *= len(row.prismatic.values_m if row.prismatic else row.revolute.values)
            else:
                count *= len(contacts[member.target_id].modes)
    return count


def _search_finite(context: RigidSE3RequestContext,
                   compilation: RigidSE3Compilation,
                   budget: kernel.ExactBudget):
    domain = context.domain
    products = tuple((row.skeleton_id, _program_count(domain, row))
                     for row in domain.program_skeletons)
    expected_total = sum(count for _, count in products)
    rows: list[RigidSE3TupleEvidence] = []
    completed: list[str] = []
    winner = None
    winner_bytes = None
    winner_cost = None
    failure = None
    row_bytes_used = 0
    for skeleton in domain.program_skeletons:
        for choices in _program_choices(domain, skeleton):
            ordinal = len(rows)
            tuple_id = f"tuple:rigid-se3:{ordinal:08d}"
            try:
                if ordinal >= min(context.limits.max_evaluated_tuples, MAX_COVERAGE_ROWS):
                    raise kernel.ExactKernelLimit("RESOURCE_LIMIT", tuple_id)
                budget.charge("EVALUATED_TUPLE")
                replay = _replay_validated(
                    context, compilation, skeleton.skeleton_id, choices, budget,
                )
                if isinstance(replay, RigidSE3FailedPrefix):
                    row = RigidSE3TupleEvidence(
                        tuple_id=tuple_id, skeleton_id=skeleton.skeleton_id,
                        ordinal=ordinal, choices=choices, disposition="INFEASIBLE",
                        failure_prefix_index=replay.failure_prefix_index,
                        failure_obligation_id=replay.failure_obligation_id,
                        failed_prefix=replay,
                    )
                else:
                    cost = _trace_cost(replay, compilation, context, budget)
                    row = RigidSE3TupleEvidence(
                        tuple_id=tuple_id, skeleton_id=skeleton.skeleton_id,
                        ordinal=ordinal, choices=choices, disposition="FEASIBLE",
                        exact_cost=RigidSE3Rational.from_fraction(cost), trace=replay,
                    )
                    program_bytes = canonical_json_bytes(replay.program)
                    improved = _better(cost, program_bytes, winner_cost, winner_bytes, budget)
                # The row model has already precharged its own 16 MiB bound.
                # Admit accumulated coverage by exact serialized bytes: the
                # conservative row bound can greatly exceed its actual size.
                row_bytes = len(canonical_json_bytes(row))
                if row_bytes > MAX_COVERAGE_BYTES - row_bytes_used:
                    raise kernel.ExactKernelLimit("RESOURCE_LIMIT", tuple_id)
                row_bytes_used += row_bytes
                if not isinstance(replay, RigidSE3FailedPrefix) and improved:
                    winner_bytes, winner, winner_cost = program_bytes, replay, cost
                rows.append(row)
                completed.append(tuple_id)
            except kernel.ExactKernelLimit as error:
                failure = budget.as_ledger(
                    completed_items=tuple(completed),
                    first_unprocessed_item=tuple_id, reason=error.reason,
                )
                break
            except M6IngressLimit:
                failure = budget.as_ledger(
                    completed_items=tuple(completed),
                    first_unprocessed_item=tuple_id, reason="RESOURCE_LIMIT",
                )
                break
        if failure is not None:
            break
    ledger = failure or budget.as_ledger(completed_items=tuple(completed))
    coverage = RigidSE3Coverage.seal(
        domain_sha256=domain.domain_sha256,
        program_products=products, expected_total=expected_total,
        tuples=tuple(rows),
        winner_tuple_id=(next(row.tuple_id for row in rows if row.trace == winner)
                         if winner is not None else None),
        ledger=ledger,
    )
    return coverage, winner, winner_cost, failure


def _search_hints(context: RigidSE3RequestContext,
                  compilation: RigidSE3Compilation,
                  budget: kernel.ExactBudget):
    from spatialcf.core.rigid_se3_compiler import d
    proposal = RigidSE3ProposalPolicyPayload.model_validate(decode_tree(
        _definition_field(context.request.solve_policy_definition_bundle,
                          d("proposal-policy"), "field:witness-hints")), strict=True)
    winner = None
    winner_bytes = None
    winner_cost = None
    completed = []
    for ordinal, hint in enumerate(proposal.witness_hints):
        item = f"item:hint:{ordinal:08d}"
        try:
            if ordinal >= context.limits.max_evaluated_tuples:
                raise kernel.ExactKernelLimit("RESOURCE_LIMIT", item)
            budget.charge("EVALUATED_TUPLE")
            replay = _replay_validated(
                context, compilation, hint.skeleton_id, hint.steps, budget,
            )
            if isinstance(replay, RigidSE3ProgramTrace):
                cost = _trace_cost(replay, compilation, context, budget)
                program_bytes = canonical_json_bytes(replay.program)
                if _better(cost, program_bytes, winner_cost, winner_bytes, budget):
                    winner_bytes, winner, winner_cost = program_bytes, replay, cost
            completed.append(item)
        except kernel.ExactKernelLimit as error:
            return winner, winner_cost, budget.as_ledger(
                completed_items=tuple(completed), first_unprocessed_item=item,
                reason=error.reason,
            )
        except M6IngressLimit:
            return winner, winner_cost, budget.as_ledger(
                completed_items=tuple(completed), first_unprocessed_item=item,
                reason="RESOURCE_LIMIT",
            )
        except ValueError:
            # A malformed/out-of-domain hint cannot change the source-bound
            # semantic domain or become a coverage certificate.
            completed.append(item)
    return winner, winner_cost, budget.as_ledger(
        completed_items=tuple(completed),
        first_unprocessed_item="item:continuous-domain-not-closed",
        reason="UNSUPPORTED",
    )
