"""Self-contained public CPU examples for the rigid SE(3) General-IR profile."""

from __future__ import annotations

from spatialcf.core.outcome_assembler import (
    assemble_counterfactual_outcome,
    assemble_no_selection_unknown,
)
from spatialcf.core.rigid_se3_backend import RigidSE3Backend
from spatialcf.core.rigid_se3_compiler import (
    build_rigid_se3_request,
    build_rigid_se3_state,
    d,
)
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    FactSetV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.definitions import CanonicalIdValue, TypedValue
from spatialcf.domain.outcomes import (
    CertifiedSolutionResult,
    CheckerDisposition,
    NoncertifiedWitnessResult,
    ProvenUnsatResult,
    UnknownResult,
)
from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    NotFormula,
    PredicateAtom,
)
from spatialcf.domain.rigid_se3 import (
    RigidSE3BodyFact,
    RigidSE3Box,
    RigidSE3ContactChoiceDomain,
    RigidSE3ContactFact,
    RigidSE3ContactState,
    RigidSE3Domain,
    RigidSE3EditMemberTemplate,
    RigidSE3EditSetTemplate,
    RigidSE3ExactSourceFacts,
    RigidSE3JointChoiceDomain,
    RigidSE3JointFact,
    RigidSE3JointState,
    RigidSE3Limits,
    RigidSE3MemberChoice,
    RigidSE3ObjectivePolicy,
    RigidSE3Pose,
    RigidSE3PrismaticDomain,
    RigidSE3ProgramSkeleton,
    RigidSE3Rational,
    RigidSE3RootChoiceDomain,
    RigidSE3RootRotationDomain,
    RigidSE3RootState,
    RigidSE3RootTranslationDomain,
    RigidSE3Rotation,
    RigidSE3StepChoice,
    RigidSE3Vector,
    RigidSE3WitnessHint,
    decode_proof_material,
)
from spatialcf.domain.scene import CanonicalScene
from spatialcf.domain.serialization import canonical_json_bytes


def _q(n: int, d: int = 1) -> RigidSE3Rational:
    return RigidSE3Rational(numerator=n, denominator=d)


def _vector(x: int = 0, y: int = 0, z: int = 0) -> RigidSE3Vector:
    return RigidSE3Vector(x=_q(x), y=_q(y), z=_q(z))


def _rotation() -> RigidSE3Rotation:
    return RigidSE3Rotation(rows=(
        (_q(1), _q(0), _q(0)),
        (_q(0), _q(1), _q(0)),
        (_q(0), _q(0), _q(1)),
    ))


def _pose(frame: str = "WORLD") -> RigidSE3Pose:
    return RigidSE3Pose(frame=frame, rotation=_rotation(), translation_m=_vector())


def _empty_scene() -> CanonicalScene:
    exact_empty = FactSetV2(
        availability=FactAvailabilityV2.KNOWN,
        completeness=FactCompletenessV2.EXACT,
        values=(), uncertainty=UncertaintyBudgetV2(),
    )
    return CanonicalScene(
        scene_id="scene:public-rigid-se3",
        objects=exact_empty, geometry_instances=exact_empty,
        collision_bodies=exact_empty, workspace_boundaries=exact_empty,
        known_free_spaces=exact_empty, support_surfaces=exact_empty,
        cameras=exact_empty, baseline_observations=exact_empty,
    )


def _source() -> RigidSE3ExactSourceFacts:
    return RigidSE3ExactSourceFacts(
        bodies=(
            RigidSE3BodyFact(
                body_id="entity:a", category_id="category:box", kind="ROOT",
                solid=True, primitive_ids=("primitive:a",), may_move=True,
                may_change_contact=True,
            ),
            RigidSE3BodyFact(
                body_id="entity:b", category_id="category:box", kind="CHILD",
                solid=True, primitive_ids=("primitive:b",), may_move=True,
                may_change_contact=True,
            ),
        ),
        boxes=(
            RigidSE3Box(
                primitive_id="primitive:a", local_anchor=_pose("BODY_LOCAL"),
                half_extents_m=_vector(1, 1, 1),
            ),
            RigidSE3Box(
                primitive_id="primitive:b", local_anchor=_pose("BODY_LOCAL"),
                half_extents_m=_vector(1, 1, 1),
            ),
        ),
        joints=(RigidSE3JointFact(
            joint_id="joint:ab", parent_body_id="entity:a",
            child_body_id="entity:b", kind="PRISMATIC",
            parent_attachment=_pose("BODY_LOCAL"),
            child_attachment=_pose("JOINT_LOCAL"), axis=_vector(1),
        ),),
        contacts=(RigidSE3ContactFact(
            contact_id="contact:ab", first_body_id="entity:a",
            first_primitive_id="primitive:a", first_face="PX",
            second_body_id="entity:b", second_primitive_id="primitive:b",
            second_face="NX", release_gap_m=_q(1, 3),
        ),),
        roots=(RigidSE3RootState(body_id="entity:a", world_pose=_pose()),),
        joint_states=(RigidSE3JointState(
            joint_id="joint:ab", kind="PRISMATIC", offset_m=_q(3),
        ),),
        contact_states=(RigidSE3ContactState(
            contact_id="contact:ab", mode="RELEASED",
        ),),
    )


def _domain(scene: CanonicalScene, source: RigidSE3ExactSourceFacts,
            *, continuous: bool, only_release: bool) -> RigidSE3Domain:
    writable = build_rigid_se3_state(scene, source).writable_value_leaves
    translation = (
        RigidSE3RootTranslationDomain(
            kind="CLOSED_RATIONAL_BOX", lower_m=_vector(), upper_m=_vector(1, 1, 1),
        ) if continuous else
        RigidSE3RootTranslationDomain(kind="FINITE", values=(_vector(),))
    )
    return RigidSE3Domain.seal(
        roots=(RigidSE3RootChoiceDomain(
            root_body_id="entity:a", translation=translation,
            rotation=RigidSE3RootRotationDomain(kind="FINITE", values=(_rotation(),)),
        ),),
        joints=(RigidSE3JointChoiceDomain(
            joint_id="joint:ab", kind="PRISMATIC",
            prismatic=RigidSE3PrismaticDomain(
                kind="FINITE", values_m=(_q(3),) if only_release else (_q(2), _q(3)),
            ),
        ),),
        contacts=(RigidSE3ContactChoiceDomain(
            contact_id="contact:ab",
            modes=("RELEASED",) if only_release else ("ENGAGED", "RELEASED"),
        ),),
        program_skeletons=(RigidSE3ProgramSkeleton(
            skeleton_id="program:atomic", steps=(RigidSE3EditSetTemplate(members=(
                RigidSE3EditMemberTemplate(kind="SET_CONTACT_MODE", target_id="contact:ab"),
                RigidSE3EditMemberTemplate(kind="SET_JOINT", target_id="joint:ab"),
                RigidSE3EditMemberTemplate(kind="SET_ROOT_SE3", target_id="entity:a"),
            )),),
        ),),
        affected_body_ids=("entity:a", "entity:b"),
        authorized_primary_leaves=writable,
    )


def _objective() -> RigidSE3ObjectivePolicy:
    return RigidSE3ObjectivePolicy(
        translation_weight=_q(1), rotation_weight=_q(1),
        prismatic_weight=_q(1), revolute_weight=_q(1),
        contact_weight=_q(1), step_weight=_q(1),
        translation_length_m=_q(1), joint_length_m=_q(1),
    )


def _goal() -> PredicateAtom:
    return PredicateAtom(
        predicate_ref="definition:spatialcf/rigid-se3/1.0/predicate/face_contact",
        operands=(TypedValue(
            value_schema_ref="schema:spatialcf/rigid-se3/1.0/contact-id",
            payload=CanonicalIdValue(value="contact:ab"),
        ),),
    )


def build_public_case(case: str):
    """Construct one complete public request, without private test fixtures."""
    if case not in {"certified", "unsat", "continuous_unknown", "witness", "no_selection"}:
        raise ValueError("unknown public rigid SE(3) example")
    scene, source = _empty_scene(), _source()
    continuous = case in {"continuous_unknown", "witness"}
    hint = RigidSE3WitnessHint(
        skeleton_id="program:atomic", steps=(RigidSE3StepChoice(members=(
            RigidSE3MemberChoice(kind="SET_CONTACT_MODE", target_id="contact:ab", mode="ENGAGED"),
            RigidSE3MemberChoice(kind="SET_JOINT", target_id="joint:ab", offset_m=_q(2)),
            RigidSE3MemberChoice(kind="SET_ROOT_SE3", target_id="entity:a", pose=_pose()),
        )),),
    )
    goal = _goal()
    return build_rigid_se3_request(
        scene=scene, source=source,
        domain=_domain(scene, source, continuous=continuous, only_release=case == "unsat"),
        objective=_objective(), after_goal=AfterGoal(formula=goal),
        before_preconditions=(BeforePrecondition(formula=NotFormula(formula=goal)),),
        preservation_invariants=(),
        witness_hints=(hint,) if case == "witness" else (),
        limits=RigidSE3Limits(max_evaluated_tuples=4),
        backend_enabled=case != "no_selection",
    )


def run_public_case(case: str):
    """Return the real source request, untrusted submission and assembled endpoint."""
    built = build_public_case(case)
    backend = RigidSE3Backend()
    selection = backend.select(built.request)
    if selection.selection_disposition == "NO_SELECTION":
        assert case == "no_selection"
        return built.request, None, assemble_no_selection_unknown(
            solve_request=built.request, selection=selection,
        )
    compilation = backend.compile(built.request)
    submission = backend.solve_submission(compilation, built.request.solver_config)
    assembled = assemble_counterfactual_outcome(
        solve_request=built.request, selection=selection,
        compilation=compilation, submission=submission,
    )
    return built.request, submission, assembled


def case_bytes(case: str) -> dict[str, bytes]:
    """Canonical source/installed byte parity surface for each public outcome."""
    request, submission, assembled = run_public_case(case)
    return {
        "request": canonical_json_bytes(request),
        "submission": canonical_json_bytes(submission) if submission is not None else b"",
        "result": canonical_json_bytes(assembled.result),
        "program_sha256": (assembled.program.program_sha256.encode()
                           if assembled.program is not None else b""),
        "after_sha256": (assembled.program.after_scene_state_sha256.encode()
                         if assembled.program is not None else b""),
    }


def test_installed_rigid_se3_certified_articulation_and_contact():
    _, submission, assembled = run_public_case("certified")
    assert type(assembled.result) is CertifiedSolutionResult
    assert assembled.certificate is not None and assembled.program is not None
    assert assembled.checked_proof_outcome.checker_disposition is CheckerDisposition.ACCEPTED
    assert len(assembled.program.steps) == 1
    proof = decode_proof_material(submission.proposal.proof_material.typed_payload[0])
    assert proof.coverage is not None and proof.coverage.expected_total == 4
    assert len(proof.coverage.tuples) == 4 and proof.coverage.ledger.reason == "NONE"
    assert proof.winner_trace is not None
    chosen = {(member.kind, member.target_id): member
              for member in proof.winner_trace.choices[0].members}
    assert chosen[("SET_JOINT", "joint:ab")].offset_m == _q(2)
    assert chosen[("SET_CONTACT_MODE", "contact:ab")].mode == "ENGAGED"
    endpoint = proof.winner_trace.steps[-1]
    assert dict(endpoint.world_poses)["entity:b"].translation_m == _vector(2)
    assert endpoint.contact_truth[0].declared_mode == "ENGAGED"
    assert endpoint.contact_truth[0].geometrically_satisfied
    assert assembled.program.program_sha256 == proof.winner_trace.program.program_sha256


def test_installed_rigid_se3_complete_finite_unsat():
    _, submission, assembled = run_public_case("unsat")
    assert type(assembled.result) is ProvenUnsatResult
    assert assembled.certificate is not None and assembled.program is None
    assert assembled.checked_proof_outcome.checker_disposition is CheckerDisposition.ACCEPTED
    proof = decode_proof_material(submission.proof_material.typed_payload[0])
    assert proof.coverage is not None and proof.coverage.expected_total == 1
    assert len(proof.coverage.tuples) == 1 and proof.coverage.ledger.reason == "NONE"
    assert proof.coverage.winner_tuple_id is None
    assert proof.coverage.tuples[0].disposition == "INFEASIBLE"


def test_installed_rigid_se3_continuous_unknown():
    _, _, assembled = run_public_case("continuous_unknown")
    assert type(assembled.result) is UnknownResult
    assert assembled.result.reason_claim_definition_ref == d("continuous-domain-not-closed")
    assert assembled.checked_proof_outcome.checker_disposition is CheckerDisposition.LIMITED
    assert assembled.certificate is None and assembled.program is None


def test_installed_rigid_se3_continuous_feasible_witness():
    _, submission, assembled = run_public_case("witness")
    assert type(assembled.result) is NoncertifiedWitnessResult
    assert assembled.checked_proof_outcome.checker_disposition is CheckerDisposition.LIMITED
    assert assembled.certificate is None and assembled.program is not None
    assert assembled.result.program_sha256 == assembled.program.program_sha256
    assert assembled.program.program_sha256 == submission.proposal.program_sha256
    proof = decode_proof_material(submission.proposal.proof_material.typed_payload[0])
    assert proof.coverage is None and proof.winner_trace is not None
    assert proof.winner_trace.program.program_sha256 == assembled.program.program_sha256


def test_installed_rigid_se3_no_selection():
    _, submission, assembled = run_public_case("no_selection")
    assert submission is None and type(assembled.result) is UnknownResult
    assert assembled.checked_proof_outcome is None
    assert assembled.verifier_dispatch_record is None
    assert assembled.result.checked_proof_outcome_sha256 is None
    assert assembled.result.verifier_dispatch_record_sha256 is None
    assert all(row.used == 0.0 for row in assembled.result.resource_usage.entries)
    assert assembled.certificate is None and assembled.program is None
