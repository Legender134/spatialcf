"""Independent source-bound replay of M6 rigid SE(3) proof submissions."""

from __future__ import annotations

from fractions import Fraction

from spatialcf.core._internal.kernels import rigid_se3 as kernel
from spatialcf.core.backends import match_backend_capabilities
from spatialcf.core.rigid_se3_compiler import (
    BACKEND_REF,
    BUILDS,
    OWNERS,
    RigidSE3CompiledProblem,
    RigidSE3MissingSource,
    RigidSE3RequestContext,
    _replay_validated,
    _swap_validated,
    c,
    compile_rigid_se3,
    d,
    exact_source_from_cells,
    source_cells_from_state,
    validate_rigid_se3_request,
)
from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposal,
    BackendProposalSubmission,
    BackendSelectionRecord,
    BackendUnknownEvidence,
    CapabilityMatch,
    CheckedProofOutcome,
    CheckerDisposition,
    ProofMaterialEnvelope,
    ResourceUsage,
)
from spatialcf.domain.rigid_se3 import (
    MAX_COVERAGE_BYTES,
    MAX_COVERAGE_ROWS,
    MAX_PROOF_CHUNKS,
    MAX_PROOF_OUTPUT_BOUND_BYTES,
    M6IngressLimit,
    RigidSE3Coverage,
    RigidSE3FailedPrefix,
    RigidSE3Ledger,
    RigidSE3MemberChoice,
    RigidSE3Pose,
    RigidSE3ProgramTrace,
    RigidSE3ProofMaterial,
    RigidSE3Rational,
    RigidSE3StepChoice,
    RigidSE3TupleEvidence,
    decode_proof_material,
    encode_proof_material,
    m6_output_upper_bound,
    preflight_m6_ingress,
    schema,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = ("verify_rigid_se3_submission",)


def _exact(value, cls):
    if type(value) is not cls:
        raise TypeError(f"M6 checker requires exact {cls.__name__}")
    preflight_m6_ingress(value)
    cls.model_validate(value.model_dump(mode="python"), strict=True)


def _selection(context: RigidSE3RequestContext, supplied: BackendSelectionRecord,
               *, require_selected: bool = True) -> None:
    _exact(supplied, BackendSelectionRecord)
    request = context.request
    descriptor = request.backend_descriptor_bundle.backend_descriptors[0]
    args = context.registry_arguments
    row = match_backend_capabilities(
        request, args["action_space_profile"], args["semantics_profile"],
        args["predicate_definitions"], args["operator_definitions"], descriptor,
    )
    selected = isinstance(row, CapabilityMatch)
    allocation = _resource(request, allocation=True)
    expected = BackendSelectionRecord.seal(
        semantic_problem_sha256=request.semantic_problem_sha256,
        solve_request_sha256=request.solve_request_sha256,
        implementation_registry_snapshot_sha256=request.implementation_registry_snapshot.implementation_registry_snapshot_sha256,
        backend_descriptor_bundle_sha256=request.backend_descriptor_bundle.backend_descriptor_bundle_sha256,
        backend_routing_policy_sha256=request.backend_routing_policy.backend_routing_policy_sha256,
        ordered_candidate_backend_refs=(BACKEND_REF,), capability_rows=(row,),
        selection_disposition="SELECTED" if selected else "NO_SELECTION",
        selection_disposition_claim_ref=d("selection-disposition"),
        selected_backend_ref=BACKEND_REF if selected else None,
        selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256 if selected else None,
        resource_allocation=allocation if selected else None,
        deterministic_selection_reason_ref=d("selection-reason"),
    )
    if supplied != expected or selected != require_selected:
        raise ValueError("M6 routing does not replay source capabilities")


def _resource(request, *, compilation=None, solve_ledger=None, allocation=False):
    """Recompute producer accounting from stage events and the source domain."""
    values = {row.definition_ref: row.finite_limit if allocation else 0.0
              for row in request.resource_policy.limits}
    for ledger in (compilation.ledger if compilation is not None else None, solve_ledger):
        if ledger is None:
            continue
        count = sum(event.amount for event in ledger.events)
        values[d(f"resource/{ledger.stage.lower()}_operations")] = max(
            values[d(f"resource/{ledger.stage.lower()}_operations")], float(count))
        values[d("resource/max_exact_operations")] = max(
            values[d("resource/max_exact_operations")], float(count))
        values[d("resource/numeric_bits")] = max(
            values[d("resource/numeric_bits")], float(ledger.peak_numeric_bits))
        for kind, ref in (
            ("ENTITY", "max_entities"), ("PRIMITIVE", "max_primitives"),
            ("JOINT", "max_joints"), ("PROGRAM", "max_programs"),
            ("EVALUATED_TUPLE", "max_evaluated_tuples"),
        ):
            used = sum(event.amount for event in ledger.events if event.kind == kind)
            key = d(f"resource/{ref}")
            values[key] = max(values[key], float(used))
    if compilation is not None:
        values[d("resource/max_steps")] = float(max(
            len(row.steps) for row in compilation.domain.program_skeletons))
        if compilation.domain.finite:
            values[d("resource/max_domain_products")] = float(sum(
                _cardinality(compilation.domain, row)
                for row in compilation.domain.program_skeletons))
    return ResourceUsage.model_validate({
        "accounting_claim_definition_ref": d("accounting"),
        "entries": tuple({"resource_definition_ref": key, "used": used}
                      for key, used in sorted(values.items())),
        "exhausted": bool((compilation and compilation.ledger.reason == "RESOURCE_LIMIT")
                       or (solve_ledger and solve_ledger.reason == "RESOURCE_LIMIT")),
    }, strict=True)


def _cardinality(domain, skeleton) -> int:
    roots = {row.root_body_id: row for row in domain.roots}
    joints = {row.joint_id: row for row in domain.joints}
    contacts = {row.contact_id: row for row in domain.contacts}
    size = 1
    for step in skeleton.steps:
        for template in step.members:
            if template.kind == "SET_ROOT_SE3":
                row = roots[template.target_id]
                size *= len(row.translation.values) * len(row.rotation.values)
            elif template.kind == "SET_JOINT":
                row = joints[template.target_id]
                size *= len(row.prismatic.values_m if row.prismatic else row.revolute.values)
            else:
                size *= len(contacts[template.target_id].modes)
    return size


def _choices(domain, skeleton):
    """Enumerate one tuple at a time from independent mixed-radix dimensions."""
    roots = {row.root_body_id: row for row in domain.roots}
    joints = {row.joint_id: row for row in domain.joints}
    contacts = {row.contact_id: row for row in domain.contacts}
    dimensions = []
    for step in skeleton.steps:
        for template in step.members:
            if template.kind == "SET_ROOT_SE3":
                row = roots[template.target_id]
                dimensions.extend((len(row.translation.values), len(row.rotation.values)))
            elif template.kind == "SET_JOINT":
                row = joints[template.target_id]
                dimensions.append(len(row.prismatic.values_m if row.prismatic else row.revolute.values))
            else:
                dimensions.append(len(contacts[template.target_id].modes))
    total = _cardinality(domain, skeleton)
    for ordinal in range(total):
        residue = ordinal
        indexes = [0] * len(dimensions)
        for index in range(len(dimensions) - 1, -1, -1):
            residue, indexes[index] = divmod(residue, dimensions[index])
        cursor = 0
        steps = []
        for step in skeleton.steps:
            members = []
            for template in step.members:
                if template.kind == "SET_ROOT_SE3":
                    row = roots[template.target_id]
                    pose = RigidSE3Pose(
                        frame="WORLD", translation_m=row.translation.values[indexes[cursor]],
                        rotation=row.rotation.values[indexes[cursor + 1]],
                    )
                    members.append(RigidSE3MemberChoice(
                        kind="SET_ROOT_SE3", target_id=template.target_id, pose=pose))
                    cursor += 2
                elif template.kind == "SET_JOINT":
                    row = joints[template.target_id]
                    value = ({"offset_m": row.prismatic.values_m[indexes[cursor]]}
                             if row.prismatic else {"circle_point": row.revolute.values[indexes[cursor]]})
                    members.append(RigidSE3MemberChoice(
                        kind="SET_JOINT", target_id=template.target_id, **value))
                    cursor += 1
                else:
                    row = contacts[template.target_id]
                    members.append(RigidSE3MemberChoice(
                        kind="SET_CONTACT_MODE", target_id=template.target_id,
                        mode=row.modes[indexes[cursor]]))
                    cursor += 1
            steps.append(RigidSE3StepChoice(members=tuple(members)))
        yield tuple(steps)


def _cost(trace, compilation, context, budget):
    previous = exact_source_from_cells(source_cells_from_state(compilation.source_state))
    total = Fraction()
    for step in trace.steps:
        current = exact_source_from_cells(source_cells_from_state(step.after_state))
        def poses(source):
            return {row.body_id: kernel.Transform(
                rotation=tuple(tuple(v.as_fraction for v in line)
                               for line in row.world_pose.rotation.rows),
                translation=row.world_pose.translation_m.fractions,
            ) for row in source.roots}
        total = budget.add(total, kernel.step_cost(
            poses(previous), poses(current),
            {row.joint_id: row for row in previous.joint_states},
            {row.joint_id: row for row in current.joint_states},
            {row.contact_id: row.mode for row in previous.contact_states},
            {row.contact_id: row.mode for row in current.contact_states},
            context.objective, budget,
        ))
        previous = current
    return total


def _finite(context, compilation, audit):
    domain = context.domain
    products = tuple((row.skeleton_id, _cardinality(domain, row))
                     for row in domain.program_skeletons)
    rows = []
    completed = []
    best = None
    best_cost = None
    best_bytes = None
    best_id = None
    failure = None
    row_bytes_used = 0
    budget = kernel.ExactBudget(context.limits, "SOLVE", audit=audit)
    for skeleton in domain.program_skeletons:
        for choices in _choices(domain, skeleton):
            ordinal = len(rows)
            item = f"tuple:rigid-se3:{ordinal:08d}"
            try:
                if ordinal >= min(context.limits.max_evaluated_tuples, MAX_COVERAGE_ROWS):
                    raise kernel.ExactKernelLimit("RESOURCE_LIMIT", item)
                budget.charge("EVALUATED_TUPLE")
                result = _replay_validated(
                    context, compilation, skeleton.skeleton_id, choices, budget)
                if isinstance(result, RigidSE3FailedPrefix):
                    row = RigidSE3TupleEvidence(
                        tuple_id=item, skeleton_id=skeleton.skeleton_id,
                        ordinal=ordinal, choices=choices, disposition="INFEASIBLE",
                        failure_prefix_index=result.failure_prefix_index,
                        failure_obligation_id=result.failure_obligation_id,
                        failed_prefix=result,
                    )
                else:
                    cost = _cost(result, compilation, context, budget)
                    row = RigidSE3TupleEvidence(
                        tuple_id=item, skeleton_id=skeleton.skeleton_id,
                        ordinal=ordinal, choices=choices, disposition="FEASIBLE",
                        exact_cost=RigidSE3Rational.from_fraction(cost), trace=result,
                    )
                    key = canonical_json_bytes(result.program)
                    if best_cost is None:
                        improved = True
                    else:
                        relation = budget.compare(cost, best_cost)
                        improved = relation < 0 or (relation == 0 and key < best_bytes)
                # The row model has already precharged its own 16 MiB bound.
                # Compare exact serialized bytes for the cumulative ceiling.
                row_bytes = len(canonical_json_bytes(row))
                if row_bytes > MAX_COVERAGE_BYTES - row_bytes_used:
                    raise kernel.ExactKernelLimit("RESOURCE_LIMIT", item)
                row_bytes_used += row_bytes
                if not isinstance(result, RigidSE3FailedPrefix) and improved:
                    best, best_cost, best_bytes, best_id = result, cost, key, item
                rows.append(row)
                completed.append(item)
            except kernel.CheckKernelLimit:
                raise
            except kernel.ExactKernelLimit as error:
                failure = budget.as_ledger(
                    completed_items=tuple(completed), first_unprocessed_item=item,
                    reason=error.reason,
                )
                break
            except M6IngressLimit:
                failure = budget.as_ledger(
                    completed_items=tuple(completed), first_unprocessed_item=item,
                    reason="RESOURCE_LIMIT",
                )
                break
        if failure is not None:
            break
    ledger = failure or budget.as_ledger(completed_items=tuple(completed))
    coverage = RigidSE3Coverage.seal(
        domain_sha256=domain.domain_sha256, program_products=products,
        expected_total=sum(count for _, count in products), tuples=tuple(rows),
        winner_tuple_id=best_id, ledger=ledger,
    )
    return coverage, best, best_cost, failure


def _hints(context, compilation, audit):
    """Hints may establish a feasible program but never continuous coverage."""
    from spatialcf.core.rigid_se3_compiler import _definition_field
    from spatialcf.domain.rigid_se3 import RigidSE3ProposalPolicyPayload, decode_tree

    policy = RigidSE3ProposalPolicyPayload.model_validate(decode_tree(
        _definition_field(context.request.solve_policy_definition_bundle,
                          d("proposal-policy"), "field:witness-hints")), strict=True)
    budget = kernel.ExactBudget(context.limits, "SOLVE", audit=audit)
    completed = []
    best = None
    best_cost = None
    best_bytes = None
    for ordinal, hint in enumerate(policy.witness_hints):
        item = f"item:hint:{ordinal:08d}"
        try:
            if ordinal >= context.limits.max_evaluated_tuples:
                raise kernel.ExactKernelLimit("RESOURCE_LIMIT", item)
            budget.charge("EVALUATED_TUPLE")
            result = _replay_validated(
                context, compilation, hint.skeleton_id, hint.steps, budget)
            if isinstance(result, RigidSE3ProgramTrace):
                cost = _cost(result, compilation, context, budget)
                key = canonical_json_bytes(result.program)
                if best_cost is None:
                    improved = True
                else:
                    relation = budget.compare(cost, best_cost)
                    improved = relation < 0 or (relation == 0 and key < best_bytes)
                if improved:
                    best, best_cost, best_bytes = result, cost, key
            completed.append(item)
        except kernel.CheckKernelLimit:
            raise
        except kernel.ExactKernelLimit as error:
            return best, best_cost, budget.as_ledger(
                completed_items=tuple(completed), first_unprocessed_item=item,
                reason=error.reason)
        except M6IngressLimit:
            return best, best_cost, budget.as_ledger(
                completed_items=tuple(completed), first_unprocessed_item=item,
                reason="RESOURCE_LIMIT")
        except ValueError:
            completed.append(item)
    return best, best_cost, budget.as_ledger(
        completed_items=tuple(completed),
        first_unprocessed_item="item:continuous-domain-not-closed",
        reason="UNSUPPORTED",
    )


def _proof_parts(submission):
    if type(submission) is BackendProposalSubmission:
        evidence = submission.proposal
    elif type(submission) in (BackendCompleteUnsatEvidence, BackendUnknownEvidence):
        evidence = submission
    else:
        raise TypeError("M6 checker requires an exact V2 backend submission")
    envelope = evidence.proof_material
    if (len(envelope.typed_payload) == 1
            and getattr(envelope.typed_payload[0].payload, "items", ())
            and len(envelope.typed_payload[0].payload.items) > MAX_PROOF_CHUNKS):
        raise ValueError("M6 ingress proof chunk ceiling exceeded")
    _exact(submission, type(submission))
    if (envelope.proof_material_definition_ref != d("proof-material")
            or envelope.payload_schema_ref != schema("proof-bytes")
            or len(envelope.typed_payload) != 1 or envelope.artifact_refs):
        raise ValueError("M6 proof envelope has wrong schema or extra artifacts")
    proof = decode_proof_material(envelope.typed_payload[0])
    return evidence, proof


def _checked(context, selection, evidence, proof, *, disposition, claim, audit,
             complete, exhausted_item=None):
    ledger = audit.as_ledger(
        completed_items=("item:check:complete",) if complete else (),
        first_unprocessed_item=None if complete else f"item:check:{exhausted_item}",
        reason="NONE" if complete else "RESOURCE_LIMIT",
    )
    facts = [(schema("checked-ledger"), canonical_sha256(
        ledger, domain="spatialcf/counterfactual/rigid-se3/checked-ledger/1.0"))]
    if complete:
        facts.append((schema("checked-proof"), proof.rigid_se3_proof_sha256))
    checked = CheckedProofOutcome.seal(
        semantic_problem_sha256=context.request.semantic_problem_sha256,
        solve_request_sha256=context.request.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proof_material_sha256=evidence.proof_material_sha256,
        checker_capability_ref=c("check"), checker_build_sha256=BUILDS["checker"],
        checker_disposition=disposition,
        checked_claim_definition_ref=claim,
        checked_fact_refs=tuple({"artifact_schema_ref": kind, "artifact_sha256": value}
                                for kind, value in sorted(facts)),
    )
    return checked, ledger


def verify_rigid_se3_submission(*, solve_request, selection, compilation, submission):
    """Public checker entry point; only a checked outcome crosses this boundary."""
    checked, _ = _verify_rigid_se3_submission_with_ledger(
        solve_request=solve_request, selection=selection,
        compilation=compilation, submission=submission,
    )
    return checked


def _verify_rigid_se3_submission_with_ledger(*, solve_request, selection,
                                             compilation, submission):
    """Reconstruct selection, source, universe, proof and claim without backend code.

    This private checker-owned seam returns the exact ledger with its digest-
    bound checked outcome for sole-assembler resource accounting. CHECK
    exhaustion has no checked-proof receipt and never validates a stronger
    producer claim, witness, or finite coverage.
    """
    context = validate_rigid_se3_request(solve_request)
    _selection(context, selection)
    if (type(compilation) is not RigidSE3CompiledProblem
            or compilation.source_solve_request != solve_request):
        raise ValueError("M6 compilation carrier or original request differs")
    evidence, submitted = _proof_parts(submission)
    if (
        evidence.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or evidence.solve_request_sha256 != solve_request.solve_request_sha256
        or evidence.backend_selection_record_sha256 != selection.backend_selection_record_sha256
        or evidence.proposal_backend_ref != BACKEND_REF
        or evidence.proposal_backend_owner_ref != OWNERS["backend"]
        or evidence.proposal_backend_capability_ref != c("solve")
        or evidence.proposal_backend_build_sha256 != BUILDS["backend"]
        or evidence.proof_material.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or evidence.proof_material.solve_request_sha256 != solve_request.solve_request_sha256
        or evidence.proof_material.backend_selection_record_sha256 != selection.backend_selection_record_sha256
        or evidence.proof_material.proposal_backend_ref != BACKEND_REF
        or submitted.semantic_problem_sha256 != solve_request.semantic_problem_sha256
        or submitted.solve_request_sha256 != solve_request.solve_request_sha256
    ):
        raise ValueError("M6 proof roots, owner, capability or build are stale")
    audit = kernel.ExactBudget(context.limits, "CHECK")
    try:
        try:
            fresh = compile_rigid_se3(solve_request, audit=audit)
            failed = fresh.ledger if fresh.ledger.reason != "NONE" else None
            if failed is not None:
                fresh = None
            missing_reason = None
        except RigidSE3MissingSource:
            fresh = None
            failed = RigidSE3Ledger(
                stage="COMPILE", reason="UNSUPPORTED",
                first_unprocessed_item="item:compile:missing-source-fact")
            missing_reason = d("missing-source-fact")
        if (compilation.compilation != fresh or compilation.failure_ledger != failed
                or compilation.failure_reason_ref != missing_reason):
            raise ValueError("M6 compilation or source-fact failure does not replay")
        coverage = None
        winner = None
        exact_cost = None
        failure = failed
        if fresh is not None:
            if context.domain.finite:
                coverage, winner, exact_cost, failure = _finite(context, fresh, audit)
            else:
                winner, exact_cost, failure = _hints(context, fresh, audit)
        bounds = None
        if winner is not None:
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
        swaps = []
        if submitted.swap_evidence:
            if fresh is None:
                raise ValueError("M6 swap evidence requires complete compilation")
            if tuple(sorted(submitted.swap_evidence, key=canonical_json_bytes)) != submitted.swap_evidence:
                raise ValueError("M6 swap evidence must use canonical order")
            if len({canonical_json_bytes(row) for row in submitted.swap_evidence}) != len(submitted.swap_evidence):
                raise ValueError("M6 swap evidence cannot repeat a claim")
            for claim in submitted.swap_evidence:
                try:
                    replayed = _swap_validated(
                        context, fresh, claim.source_trace.skeleton_id,
                        claim.source_trace.choices,
                        claim.swapped_trace.skeleton_id,
                        claim.adjacent_index, kind=claim.kind, budget=audit,
                    )
                except kernel.ExactKernelLimit as error:
                    raise kernel.CheckKernelLimit(error.reason, error.item) from error
                if replayed != claim:
                    raise ValueError("M6 swap claim fails two-order source replay")
                swaps.append(replayed)
        if m6_output_upper_bound((fresh, coverage, winner, failure, tuple(swaps))) > MAX_PROOF_OUTPUT_BOUND_BYTES:
            raise M6IngressLimit("M6 ingress expected proof output ceiling exceeded")
        expected = RigidSE3ProofMaterial.seal(
            semantic_problem_sha256=solve_request.semantic_problem_sha256,
            solve_request_sha256=solve_request.solve_request_sha256,
            source_state_sha256=context.state_view.state.scene_state_sha256,
            domain_sha256=context.domain.domain_sha256,
            compilation=fresh, coverage=coverage, winner_trace=winner,
            swap_evidence=tuple(swaps), failure_ledger=failure,
        )
        if submitted != expected:
            raise ValueError("M6 proof fails fresh source, tuple, prefix, objective or ledger replay")
        envelope = ProofMaterialEnvelope.seal(
            semantic_problem_sha256=solve_request.semantic_problem_sha256,
            solve_request_sha256=solve_request.solve_request_sha256,
            backend_selection_record_sha256=selection.backend_selection_record_sha256,
            proposal_backend_ref=BACKEND_REF,
            proof_material_definition_ref=d("proof-material"),
            payload_schema_ref=schema("proof-bytes"),
            typed_payload=(encode_proof_material(expected),), artifact_refs=(),
        )
        common = {
            "semantic_problem_sha256": solve_request.semantic_problem_sha256,
            "solve_request_sha256": solve_request.solve_request_sha256,
            "backend_selection_record_sha256": selection.backend_selection_record_sha256,
            "proposal_backend_ref": BACKEND_REF,
            "proposal_backend_owner_ref": OWNERS["backend"],
            "proposal_backend_capability_ref": c("solve"),
            "proposal_backend_build_sha256": BUILDS["backend"],
            "proof_material": envelope,
            "proof_material_sha256": envelope.proof_material_sha256,
            "resource_usage": _resource(solve_request, compilation=fresh,
                                     solve_ledger=failure or (coverage.ledger if coverage else None)),
        }
        if winner is not None:
            assert bounds is not None
            proposal = BackendProposal.seal(
                **common, proposal_claim_definition_ref=d("certified"),
                program_sha256=winner.program.program_sha256,
                after_scene_state_sha256=winner.after_state_sha256,
                objective_lower_bound=bounds[0], objective_upper_bound=bounds[1],
            )
            expected_submission = BackendProposalSubmission.seal(
                proposal=proposal, backend_proposal_sha256=proposal.backend_proposal_sha256)
            disposition = (CheckerDisposition.ACCEPTED if coverage is not None
                           and coverage.ledger.reason == "NONE" else CheckerDisposition.LIMITED)
            claim = d("certified")
        elif (coverage is not None and coverage.ledger.reason == "NONE"
              and coverage.winner_tuple_id is None and failure is None):
            expected_submission = BackendCompleteUnsatEvidence.seal(
                **common, complete_domain_claim_definition_ref=d("unsat"),
                authorized_domain_sha256=context.domain.domain_sha256,
                complete_domain_coverage_artifact_sha256=coverage.coverage_sha256)
            disposition = CheckerDisposition.ACCEPTED
            claim = d("unsat")
        else:
            if missing_reason is not None:
                reason = missing_reason
            elif (fresh is not None and not context.domain.finite and failure is not None
                  and failure.reason == "UNSUPPORTED"):
                reason = d("continuous-domain-not-closed")
            else:
                if failure is None:
                    raise ValueError("M6 unknown has no fresh supported reason")
                reason = d({
                    "RESOURCE_LIMIT": "resource-exhausted",
                    "NUMERIC_GAP": "numeric-gap",
                    "UNSUPPORTED": "unsupported",
                }[failure.reason])
            expected_submission = BackendUnknownEvidence.seal(
                **common, reason_claim_definition_ref=reason)
            disposition = CheckerDisposition.LIMITED
            claim = reason
        if submission != expected_submission:
            raise ValueError("M6 submission claim, branch, bounds or resource accounting differs")
        return _checked(context, selection, evidence, expected, disposition=disposition,
                        claim=claim, audit=audit, complete=True)
    except kernel.CheckKernelLimit as error:
        return _checked(context, selection, evidence, submitted,
                        disposition=CheckerDisposition.LIMITED,
                        claim=d("resource-exhausted"), audit=audit, complete=False,
                        exhausted_item=error.item)
