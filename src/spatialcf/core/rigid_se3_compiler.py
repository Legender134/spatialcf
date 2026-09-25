"""Source-bound M6 state composition and later program compilation.

M6 physical entities are extension facts over an exact empty CanonicalScene.
The generic registry owns structural state/authorization validation; this
module owns the profile-specific source roster and future program replay.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from spatialcf.core import registry as generic
from spatialcf.core._internal.kernels import rigid_se3 as kernel
from spatialcf.domain.base import (
    FactAvailabilityV2,
    FactCompletenessV2,
    UncertaintyBudgetV2,
)
from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    EditProgram,
    ExtensionFact,
    ExtensionFactBundle,
    SceneStateEnvelope,
)
from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    CanonicalIdValue,
    DefinitionBundle,
    DigestValue,
    EnumSymbolValue,
    NamedTypedValue,
    RecordValue,
    ReferenceValue,
    TypedValue,
    ValueFieldDefinition,
    ValueKind,
    ValueSchemaDefinition,
)
from spatialcf.domain.operators import (
    OperationArgument,
    OperationInvocation,
    OperatorDefinition,
    StateAddressPattern,
    StateDeltaManifest,
    StateLeafIndex,
    StateVariableDefinition,
    StateVariableRef,
    WriteAuthority,
)
from spatialcf.domain.predicates import (
    AfterGoal,
    AllOfFormula,
    AnyOfFormula,
    BeforePrecondition,
    ExactlyKFormula,
    ExistsFormula,
    ForAllFormula,
    GroundedObligation,
    GroundedObligationSet,
    ImpliesFormula,
    NotFormula,
    PredicateAtom,
    PredicateDefinition,
    PreservationInvariant,
)
from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationOwnerBinding,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    ObjectiveExpression,
    ObjectiveTerm,
    ProofPolicy,
    ResourceLimit,
    ResourcePolicy,
    SemanticsProfile,
    SolverBackendDescriptor,
)
from spatialcf.domain.rigid_se3 import (
    INGRESS_POLICY,
    MAX_PROGRAM_TRACE_BYTES,
    MAX_REQUEST_SCALAR_BYTES,
    PREFIX,
    PROFILE_REF,
    M6IngressLimit,
    RigidSE3BodyFact,
    RigidSE3Box,
    RigidSE3CirclePoint,
    RigidSE3Compilation,
    RigidSE3ContactFact,
    RigidSE3ContactState,
    RigidSE3ContactTruth,
    RigidSE3Domain,
    RigidSE3ExactSourceFacts,
    RigidSE3FactCell,
    RigidSE3FailedPrefix,
    RigidSE3Inventory,
    RigidSE3JointFact,
    RigidSE3JointState,
    RigidSE3Ledger,
    RigidSE3Limits,
    RigidSE3ObjectivePolicy,
    RigidSE3Pose,
    RigidSE3PrefixTruth,
    RigidSE3ProfileRegistration,
    RigidSE3ProgramTrace,
    RigidSE3ProposalPolicyPayload,
    RigidSE3Rational,
    RigidSE3RegionFact,
    RigidSE3RootState,
    RigidSE3Rotation,
    RigidSE3SourceCells,
    RigidSE3SourceFactInput,
    RigidSE3StepChoice,
    RigidSE3StepTrace,
    RigidSE3SwapEvidence,
    RigidSE3Vector,
    RigidSE3WitnessHint,
    decode_tree,
    encode_tree,
    m6_output_upper_bound,
    preflight_m6_ingress,
    proof_value_schemas,
    schema,
)
from spatialcf.domain.scene import CanonicalScene
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

BASE_SCHEMA = "schema:spatialcf/canonical-scene/2.0"
FAMILY = {
    name: f"definition:{PREFIX}/fact/{name}"
    for name in (
        "body", "box", "joint", "contact", "region", "root-x", "root-y", "root-z",
        "root-rotation", "joint-state", "contact-mode", "inventory",
    )
}
INVENTORY_KEY = "inventory:rigid-se3"
SEMANTICS_REF = "spatialcf/rigid_se3_multi/semantics@1"
BACKEND_REF = "backend:spatialcf/rigid-se3/exact-finite/1.0"
OWNERS = {name: f"owner:spatialcf/rigid-se3/{name}" for name in
          ("profile", "kernel", "compiler", "backend", "checker")}
BUILDS = {name: canonical_sha256((PREFIX, name, "implementation:1", INGRESS_POLICY),
                                domain=f"{PREFIX}/build") for name in OWNERS}


def d(name: str) -> str:
    return f"definition:{PREFIX}/{name}"


def c(name: str) -> str:
    return f"capability:{PREFIX}/{name}"


def _ordered(values: tuple):
    return tuple(sorted(values, key=canonical_json_bytes))


def _exact_empty_scene(scene: CanonicalScene) -> CanonicalScene:
    scene = CanonicalScene.model_validate(scene.model_dump(mode="python"), strict=True)
    if scene.coordinate_system != "RH_METERS_Z_UP":
        raise ValueError("M6 requires RH metre Z-up coordinates")
    for name in (
        "objects", "geometry_instances", "collision_bodies", "workspace_boundaries",
        "known_free_spaces", "support_surfaces", "cameras", "baseline_observations",
    ):
        facts = getattr(scene, name)
        if (
            facts.availability != FactAvailabilityV2.KNOWN
            or facts.completeness != FactCompletenessV2.EXACT
            or facts.values != ()
            or facts.uncertainty != UncertaintyBudgetV2()
        ):
            raise ValueError("M6 requires each base scene family to be exact and empty")
    return scene


def _fact(name: str, subject: str, key: str, value: object) -> ExtensionFact:
    # One tagged tree includes the explicit availability branch. Nesting a
    # TypedValue inside another tagged tree would make the wire too deep for
    # normal serializers on a box containing a 3x3 exact rotation.
    return ExtensionFact(
        fact_family_ref=FAMILY[name], subject_entity_id=subject,
        fact_key=key, value=encode_tree({"availability": "KNOWN_EXACT", "value": value,
                                        "uncertainty_upper": None}),
    )


@dataclass(frozen=True)
class RigidSE3StateView:
    state: SceneStateEnvelope
    inventory: RigidSE3Inventory
    address_by_owner: Mapping[tuple[str, str, str], str]
    writable_value_leaves: tuple[StateVariableRef, ...]
    unavailable_fact_addresses: tuple[str, ...] = ()


def build_rigid_se3_state(
    scene: CanonicalScene,
    source: RigidSE3ExactSourceFacts | RigidSE3SourceCells,
) -> RigidSE3StateView:
    """Bind all known exact facts and fine primary leaves to one General IR state.

    This structural composition does not assert source collision/contact truth;
    the exact kernel and compiler precondition stage prove those separately.
    """
    preflight_m6_ingress((scene, source), scalar_bytes=MAX_REQUEST_SCALAR_BYTES)
    scene = _exact_empty_scene(scene)
    if isinstance(source, RigidSE3SourceCells):
        return _state_from_cells(scene, source)
    source = RigidSE3ExactSourceFacts.model_validate(source.model_dump(mode="python"), strict=True)
    bodies = {body.body_id: body for body in source.bodies}
    primitive_owner = {
        primitive: body.body_id for body in source.bodies for primitive in body.primitive_ids
    }
    joints = {joint.joint_id: joint for joint in source.joints}
    contacts = {contact.contact_id: contact for contact in source.contacts}
    facts: list[ExtensionFact] = []
    writable: set[tuple[str, str, str]] = set()

    for body in source.bodies:
        facts.append(_fact("body", body.body_id, body.body_id, body))
    for box in source.boxes:
        facts.append(_fact("box", primitive_owner[box.primitive_id], box.primitive_id, box))
    for joint in source.joints:
        facts.append(_fact("joint", joint.child_body_id, joint.joint_id, joint))
    for contact in source.contacts:
        facts.append(_fact("contact", contact.first_body_id, contact.contact_id, contact))
    for region in source.regions:
        facts.append(_fact("region", min(bodies), region.region_id, region))
    for root in source.roots:
        for axis in ("x", "y", "z"):
            facts.append(_fact(f"root-{axis}", root.body_id, root.body_id,
                               getattr(root.world_pose.translation_m, axis)))
            writable.add((FAMILY[f"root-{axis}"], root.body_id, root.body_id))
        facts.append(_fact("root-rotation", root.body_id, root.body_id,
                           root.world_pose.rotation))
        writable.add((FAMILY["root-rotation"], root.body_id, root.body_id))
    for state in source.joint_states:
        joint = joints[state.joint_id]
        value = state.offset_m if state.kind == "PRISMATIC" else state.circle_point
        facts.append(_fact("joint-state", joint.child_body_id, state.joint_id, value))
        writable.add((FAMILY["joint-state"], joint.child_body_id, state.joint_id))
    for state in source.contact_states:
        contact = contacts[state.contact_id]
        facts.append(_fact("contact-mode", contact.first_body_id, state.contact_id, state.mode))
        writable.add((FAMILY["contact-mode"], contact.first_body_id, state.contact_id))

    owner = min(bodies)
    inventory_owner = (FAMILY["inventory"], owner, INVENTORY_KEY)
    owners = tuple(fact.ownership_key for fact in facts) + (inventory_owner,)
    address_by_owner = {key: generic._extension_fact_address(*key) for key in owners}
    if len(set(address_by_owner.values())) != len(owners):
        raise ValueError("M6 extension fact addresses collide")
    inventory = RigidSE3Inventory.seal(
        body_ids=tuple(bodies),
        primitive_ids=tuple(sorted(primitive_owner)),
        joint_ids=tuple(joints),
        contact_ids=tuple(contacts),
        region_ids=tuple(region.region_id for region in source.regions),
        fact_addresses=tuple(sorted(address_by_owner.values())),
    )
    return _finish_state(scene, facts, inventory, address_by_owner, writable)


def _finish_state(
    scene: CanonicalScene, facts: list[ExtensionFact],
    inventory: RigidSE3Inventory,
    address_by_owner: Mapping[tuple[str, str, str], str],
    writable: set[tuple[str, str, str]],
    unavailable: tuple[str, ...] = (),
) -> RigidSE3StateView:
    owner = min(inventory.body_ids)
    facts.append(_fact("inventory", owner, INVENTORY_KEY, inventory))
    leaves: list[StateVariableRef] = []
    writable_leaves: list[StateVariableRef] = []
    for fact in facts:
        key = fact.ownership_key
        address = address_by_owner[key]
        for presence in (True, False):
            token = canonical_sha256(
                (address, presence), domain=f"{PREFIX}/extension-leaf"
            )
            leaf = StateVariableRef(
                state_variable_schema_ref=schema(f"leaf/{token}"),
                state_schema_ref=schema("state"),
                fact_family_ref=fact.fact_family_ref,
                entity_or_fact_key=address,
                field_path_ref=("field-path:presence-" if presence else "field-path:") + token,
            )
            leaves.append(leaf)
            if not presence and key in writable:
                writable_leaves.append(leaf)
    envelope = SceneStateEnvelope.seal(
        base_scene_schema_ref=BASE_SCHEMA,
        base_scene_payload=scene,
        extension_fact_bundles=(ExtensionFactBundle.seal(facts=_ordered(tuple(facts))),),
        closed_entity_index=inventory.body_ids,
        canonical_state_leaf_index=StateLeafIndex.seal(leaves=_ordered(tuple(leaves))),
    )
    return RigidSE3StateView(
        state=envelope, inventory=inventory,
        address_by_owner=address_by_owner,
        writable_value_leaves=_ordered(tuple(writable_leaves)),
        unavailable_fact_addresses=unavailable,
    )


def source_cells_from_exact(source: RigidSE3ExactSourceFacts) -> RigidSE3SourceCells:
    """Expose every exact source address so availability can change explicitly."""
    # The exact composition is the sole owner of topology-to-address mapping.
    # Reuse it, then remove only the generated inventory fact from the rows.
    from spatialcf.domain.base import FactSetV2

    empty = FactSetV2(availability=FactAvailabilityV2.KNOWN,
                      completeness=FactCompletenessV2.EXACT, values=(),
                      uncertainty=UncertaintyBudgetV2())
    scene = CanonicalScene(
        scene_id="scene:m6-source-cell-conversion", objects=empty,
        geometry_instances=empty, collision_bodies=empty,
        workspace_boundaries=empty, known_free_spaces=empty,
        support_surfaces=empty, cameras=empty, baseline_observations=empty,
    )
    view = build_rigid_se3_state(scene, source)
    rows = []
    by_family = {value: name for name, value in FAMILY.items()}
    for fact in view.state.extension_fact_bundles[0].facts:
        name = by_family[fact.fact_family_ref]
        if name == "inventory":
            continue
        raw = decode_tree(fact.value)
        rows.append(RigidSE3SourceFactInput(
            family=name, subject_entity_id=fact.subject_entity_id,
            fact_key=fact.fact_key,
            cell=RigidSE3FactCell(availability="KNOWN_EXACT",
                                   value=encode_tree(raw["value"])),
        ))
    return RigidSE3SourceCells(inventory=view.inventory, facts=tuple(sorted(
        rows, key=lambda row: (row.family, row.subject_entity_id, row.fact_key),
    )))


def _state_from_cells(scene: CanonicalScene, source: RigidSE3SourceCells) -> RigidSE3StateView:
    source = RigidSE3SourceCells.model_validate(source.model_dump(mode="python"), strict=True)
    owner = min(source.inventory.body_ids)
    inventory_owner = (FAMILY["inventory"], owner, INVENTORY_KEY)
    owners = tuple((FAMILY[row.family], row.subject_entity_id, row.fact_key)
                   for row in source.facts) + (inventory_owner,)
    addresses = {key: generic._extension_fact_address(*key) for key in owners}
    if len(addresses) != len(owners) or tuple(sorted(addresses.values())) != source.inventory.fact_addresses:
        raise ValueError("source inventory does not match complete extension ownership")
    writable_names = {"root-x", "root-y", "root-z", "root-rotation", "joint-state", "contact-mode"}
    writable = {key for key, row in zip(owners, source.facts, strict=False)
                if row.family in writable_names}
    unavailable = tuple(sorted(addresses[(FAMILY[row.family], row.subject_entity_id, row.fact_key)]
                               for row in source.facts if row.cell.availability != "KNOWN_EXACT"))
    facts = []
    for row in source.facts:
        cell = row.cell
        raw = {
            "availability": cell.availability,
            "value": decode_tree(cell.value) if cell.value is not None else None,
            "uncertainty_upper": cell.uncertainty_upper,
        }
        facts.append(ExtensionFact(
            fact_family_ref=FAMILY[row.family], subject_entity_id=row.subject_entity_id,
            fact_key=row.fact_key, value=encode_tree(raw),
        ))
    return _finish_state(scene, facts, source.inventory, addresses, writable, unavailable)


@dataclass(frozen=True)
class RigidSE3RequestContext:
    request: CounterfactualSolveRequest
    state_view: RigidSE3StateView
    domain: RigidSE3Domain
    objective: RigidSE3ObjectivePolicy
    limits: RigidSE3Limits
    registry: generic.StaticImplementationRegistry
    registry_arguments: dict


@dataclass(frozen=True)
class RigidSE3CompiledProblem:
    """Passive compilation carrier shared by producer and independent checker."""

    source_solve_request: CounterfactualSolveRequest
    compilation: RigidSE3Compilation | None
    failure_ledger: RigidSE3Ledger | None
    failure_reason_ref: str | None = None

    @property
    def semantic_problem_sha256(self) -> str:
        return self.source_solve_request.semantic_problem_sha256

    @property
    def solve_request_sha256(self) -> str:
        return self.source_solve_request.solve_request_sha256

    @property
    def selected_backend_descriptor_sha256(self) -> str:
        return self.source_solve_request.backend_descriptor_bundle.backend_descriptors[0].backend_descriptor_sha256

    @property
    def grounded_obligation_set_sha256(self) -> str:
        return _grounded_obligations(
            self.source_solve_request.semantic_problem
        ).grounded_obligation_set_sha256

    @property
    def compiled_artifact_sha256s(self) -> tuple[str, ...]:
        return (self.compilation.compilation_sha256,) if self.compilation else ()


def _id_value(value: str, value_schema: str = "id") -> TypedValue:
    return TypedValue(value_schema_ref=schema(value_schema), payload=CanonicalIdValue(value=value))


def _record(fields: Mapping[str, TypedValue], value_schema: str) -> TypedValue:
    return TypedValue(
        value_schema_ref=schema(value_schema),
        payload=RecordValue(fields=tuple(
            NamedTypedValue(name=name, value=value) for name, value in sorted(fields.items())
        )),
    )


def _program_write_leaves(
    view: RigidSE3StateView, domain: RigidSE3Domain,
) -> tuple[StateVariableRef, ...]:
    available = set(view.writable_value_leaves)
    by_address = {address: owner for owner, address in view.address_by_owner.items()}
    required_owners: set[tuple[str, str, str]] = set()
    for program in domain.program_skeletons:
        for step in program.steps:
            for member in step.members:
                if member.kind == "SET_ROOT_SE3":
                    required_owners.update((FAMILY[name], member.target_id, member.target_id)
                                           for name in ("root-x", "root-y", "root-z", "root-rotation"))
                else:
                    name = "joint-state" if member.kind == "SET_JOINT" else "contact-mode"
                    matching = {owner for owner in view.address_by_owner
                                if owner[0] == FAMILY[name] and owner[2] == member.target_id}
                    if len(matching) != 1:
                        raise ValueError("edit target lacks one exact mutable fact")
                    required_owners.update(matching)
    selected = _ordered(tuple(leaf for leaf in available
                              if by_address[leaf.entity_or_fact_key] in required_owners))
    if len(selected) != len(required_owners):
        raise ValueError("program target lacks a complete source primary leaf")
    return selected


_PREDICATE_OPERANDS = {
    "BODY_SEPARATED": ("body-id", "other-body-id"),
    "FACE_CONTACT": ("contact-id",),
    "CONTACT_RELEASED": ("contact-id",),
    "BODY_IN_REGION": ("body-id", "region-id"),
    "FRAME_OFFSET_EQUALS": ("frame-id", "other-frame-id", "tree"),
    "ROOT_POSE_MEMBER": ("body-id",),
    "JOINT_STATE_MEMBER": ("joint-id",),
}

_CORE_INVARIANTS = (
    "all-body-collision", "complete-state-authority", "contact-consistency",
    "endpoint-domain", "kinematics",
)


def _ground_m6_formula(formula: object, inventory: RigidSE3Inventory) -> object:
    """Lower finite entity bindings to the registered canonical-ID operands.

    The retained General IR requires the semantic root to carry a fully
    grounded formula. Its finite quantifier binds ENTITY_REF values, while the
    existing M6 predicate signatures use CANONICAL_ID. This explicit profile
    adapter preserves both contracts and rejects references outside the closed
    source inventory before the semantic root is sealed.
    """
    memberships = {
        "body-id": set(inventory.body_ids),
        "other-body-id": set(inventory.body_ids),
        "frame-id": set(inventory.body_ids) | {"frame:world"},
        "other-frame-id": set(inventory.body_ids) | {"frame:world"},
        "contact-id": set(inventory.contact_ids),
        "joint-id": set(inventory.joint_ids),
        "region-id": set(inventory.region_ids),
    }

    def validate_original(node: object, bindings: Mapping[str, str]) -> None:
        if type(node) is bool:
            return
        if isinstance(node, PredicateAtom):
            name = node.predicate_ref.removeprefix(d("predicate/")).upper()
            expected = _PREDICATE_OPERANDS.get(name)
            if (expected is None
                    or node.predicate_ref != d(f"predicate/{name.lower()}")
                    or len(expected) != len(node.operands)):
                raise ValueError("M6 formula contains an unregistered predicate")
            for label, operand in zip(expected, node.operands, strict=True):
                if operand.value_schema_ref != schema(label):
                    raise ValueError("M6 formula predicate operand has the wrong schema")
                if label == "tree":
                    RigidSE3Vector.model_validate(decode_tree(operand), strict=True)
                    continue
                payload = operand.payload
                if isinstance(payload, CanonicalIdValue):
                    reference = payload.value
                elif (isinstance(payload, ReferenceValue)
                      and payload.kind is ValueKind.ENTITY_REF):
                    reference = payload.reference
                    if reference.startswith("variable:"):
                        if bindings.get(reference) != operand.value_schema_ref:
                            raise ValueError("M6 formula has an unbound or mistyped entity variable")
                        continue
                else:
                    raise ValueError("M6 predicate ID operand has an unsupported value kind")
                if reference not in memberships[label]:
                    raise ValueError("M6 predicate ID operand is outside the closed source inventory")
            return
        if isinstance(node, (ForAllFormula, ExistsFormula)):
            entity_set = node.entity_set
            label = entity_set.entity_schema_ref.removeprefix(schema(""))
            if (entity_set.entity_schema_ref != schema(label)
                    or label not in memberships):
                raise ValueError("M6 finite quantifier has an unsupported entity schema")
            for entity in entity_set.entities:
                if (not isinstance(entity.payload, ReferenceValue)
                        or entity.payload.kind is not ValueKind.ENTITY_REF
                        or entity.payload.reference not in memberships[label]):
                    raise ValueError("M6 finite quantifier contains an unknown entity")
            validate_original(node.formula, bindings | {
                entity_set.variable_ref: entity_set.entity_schema_ref,
            })
            return
        if isinstance(node, NotFormula):
            validate_original(node.formula, bindings)
            return
        if isinstance(node, (AllOfFormula, AnyOfFormula, ExactlyKFormula)):
            for item in node.formulas:
                validate_original(item, bindings)
            return
        if isinstance(node, ImpliesFormula):
            validate_original(node.antecedent, bindings)
            validate_original(node.consequent, bindings)
            return
        raise ValueError("M6 formula contains an unsupported Boolean node")

    validate_original(formula, {})
    grounded = formula if type(formula) is bool else formula.ground()

    def lower(node: object) -> object:
        if type(node) is bool:
            return node
        if isinstance(node, PredicateAtom):
            name = node.predicate_ref.removeprefix(d("predicate/")).upper()
            expected = _PREDICATE_OPERANDS.get(name)
            if (expected is None
                    or node.predicate_ref != d(f"predicate/{name.lower()}")
                    or len(expected) != len(node.operands)):
                raise ValueError("M6 quantified formula has an unregistered predicate")
            operands = []
            for label, operand in zip(expected, node.operands, strict=True):
                if isinstance(operand.payload, ReferenceValue):
                    ref = operand.payload.reference
                    if (operand.payload.kind is not ValueKind.ENTITY_REF
                            or operand.value_schema_ref != schema(label)
                            or label not in memberships
                            or ref not in memberships[label]):
                        raise ValueError("M6 grounded entity reference is outside its exact source roster")
                    operand = TypedValue(
                        value_schema_ref=operand.value_schema_ref,
                        payload=CanonicalIdValue(value=ref),
                    )
                operands.append(operand)
            return PredicateAtom(predicate_ref=node.predicate_ref, operands=tuple(operands))
        if isinstance(node, NotFormula):
            return NotFormula(formula=lower(node.formula))
        if isinstance(node, AllOfFormula):
            return AllOfFormula(formulas=tuple(lower(item) for item in node.formulas))
        if isinstance(node, AnyOfFormula):
            return AnyOfFormula(formulas=tuple(lower(item) for item in node.formulas))
        if isinstance(node, ImpliesFormula):
            return ImpliesFormula(antecedent=lower(node.antecedent),
                                  consequent=lower(node.consequent))
        if isinstance(node, ExactlyKFormula):
            return ExactlyKFormula(k=node.k, formulas=tuple(lower(item) for item in node.formulas))
        raise ValueError("M6 formula is not a registered grounded Boolean node")

    return lower(grounded)


def _known_topology(
    source: RigidSE3ExactSourceFacts | RigidSE3SourceCells,
) -> tuple[dict[str, RigidSE3BodyFact], dict[str, RigidSE3JointFact],
           dict[str, RigidSE3ContactFact]] | None:
    if isinstance(source, RigidSE3ExactSourceFacts):
        return (
            {row.body_id: row for row in source.bodies},
            {row.joint_id: row for row in source.joints},
            {row.contact_id: row for row in source.contacts},
        )
    rows = {family: tuple(row for row in source.facts if row.family == family)
            for family in ("body", "joint", "contact")}
    if any(row.cell.availability != "KNOWN_EXACT" for values in rows.values() for row in values):
        return None
    return (
        {row.fact_key: RigidSE3BodyFact.model_validate(decode_tree(row.cell.value), strict=True)
         for row in rows["body"]},
        {row.fact_key: RigidSE3JointFact.model_validate(decode_tree(row.cell.value), strict=True)
         for row in rows["joint"]},
        {row.fact_key: RigidSE3ContactFact.model_validate(decode_tree(row.cell.value), strict=True)
         for row in rows["contact"]},
    )


def _validate_affected_body_closure(
    source: RigidSE3ExactSourceFacts | RigidSE3SourceCells,
    domain: RigidSE3Domain,
) -> None:
    topology = _known_topology(source)
    if topology is None:
        return  # Selected compilation reports the unavailable fact before search.
    bodies, joints, contacts = topology
    if any(row.root_body_id not in bodies or bodies[row.root_body_id].kind != "ROOT"
           for row in domain.roots):
        raise ValueError("M6 root choice domain disagrees with exact source roots")
    if any(row.joint_id not in joints or joints[row.joint_id].kind != row.kind
           for row in domain.joints):
        raise ValueError("M6 joint choice kind disagrees with immutable source joint")
    if any(row.contact_id not in contacts for row in domain.contacts):
        raise ValueError("M6 contact choice domain lacks an exact source contact")
    children: dict[str, set[str]] = {body_id: set() for body_id in bodies}
    for joint in joints.values():
        children[joint.parent_body_id].add(joint.child_body_id)

    def subtree(root: str) -> set[str]:
        pending = [root]
        found: set[str] = set()
        while pending:
            current = pending.pop()
            if current not in bodies or current in found:
                continue
            found.add(current)
            pending.extend(children[current])
        return found

    affected: set[str] = set()
    for skeleton in domain.program_skeletons:
        for step in skeleton.steps:
            for member in step.members:
                if member.kind == "SET_ROOT_SE3":
                    if member.target_id not in bodies or bodies[member.target_id].kind != "ROOT":
                        raise ValueError("root edit target is not an exact source root")
                    moved = subtree(member.target_id)
                    if any(not bodies[body_id].may_move for body_id in moved):
                        raise ValueError("root edit would move a frozen descendant")
                    affected.update(moved)
                elif member.kind == "SET_JOINT":
                    joint = joints.get(member.target_id)
                    if joint is None or joint.kind == "FIXED":
                        raise ValueError("joint edit target is not an actuated source joint")
                    moved = subtree(joint.child_body_id)
                    if any(not bodies[body_id].may_move for body_id in moved):
                        raise ValueError("joint edit would move a frozen descendant")
                    affected.update(moved)
                else:
                    contact = contacts.get(member.target_id)
                    if contact is None:
                        raise ValueError("contact edit target lacks an exact source definition")
                    participants = {contact.first_body_id, contact.second_body_id}
                    if any(not bodies[body_id].may_change_contact for body_id in participants):
                        raise ValueError("contact edit would change a frozen participant")
                    affected.update(participants)
    if tuple(sorted(affected)) != domain.affected_body_ids:
        raise ValueError("affected body roster must equal the source-bound edit closure")


def build_rigid_se3_request(
    *, scene: CanonicalScene, source: RigidSE3ExactSourceFacts | RigidSE3SourceCells,
    domain: RigidSE3Domain, objective: RigidSE3ObjectivePolicy,
    after_goal: AfterGoal,
    before_preconditions: tuple[BeforePrecondition, ...],
    preservation_invariants: tuple[PreservationInvariant, ...],
    witness_hints: tuple[RigidSE3WitnessHint, ...] = (),
    limits: RigidSE3Limits | None = None,
    backend_enabled: bool = True,
    problem_id: str = "problem:rigid-se3-synthetic",
) -> RigidSE3RequestContext:
    """Compose a source-bound M6 request through the retained General IR.

    Domain and hints remain distinct semantic and operational roots.  This
    function performs structural composition; exact feasibility is checked by
    the M6 kernel and fresh checker in later tasks.
    """
    preflight_m6_ingress(
        (scene, source, domain, objective, after_goal, before_preconditions,
         preservation_invariants, witness_hints, limits, backend_enabled, problem_id),
        scalar_bytes=MAX_REQUEST_SCALAR_BYTES,
    )
    state_view = build_rigid_se3_state(scene, source)
    state = state_view.state
    _validate_affected_body_closure(source, domain)
    after_goal = AfterGoal(formula=_ground_m6_formula(after_goal.formula,
                                                       state_view.inventory))
    before_preconditions = tuple(BeforePrecondition(
        formula=_ground_m6_formula(row.formula, state_view.inventory),
    ) for row in before_preconditions)
    preservation_invariants = tuple(PreservationInvariant(
        before_formula=_ground_m6_formula(row.before_formula, state_view.inventory),
        after_formula=_ground_m6_formula(row.after_formula, state_view.inventory),
        transition_comparator_ref=row.transition_comparator_ref,
    ) for row in preservation_invariants)
    if domain.authorized_primary_leaves != _program_write_leaves(state_view, domain):
        raise ValueError("M6 authorization must exactly match fine program target leaves")
    if not set(domain.affected_body_ids) <= set(state.closed_entity_index):
        raise ValueError("affected body set is outside the exact source inventory")
    limits = limits or RigidSE3Limits()
    for name, value in limits.model_dump(mode="python").items():
        try:
            transported = float(value)
        except OverflowError as error:
            raise ValueError(f"M6 {name} cannot be transported as a finite float") from error
        if not math.isfinite(transported) or int(transported) != value:
            raise ValueError(f"M6 {name} loses exact integer value in policy transport")
    if domain.roots and (not objective.translation_enabled or not objective.rotation_enabled):
        raise ValueError("root pose freedom requires positive translation and rotation weights")
    if any(row.kind == "PRISMATIC" for row in domain.joints) and not objective.prismatic_enabled:
        raise ValueError("prismatic freedom requires a positive joint weight")
    if any(row.kind == "REVOLUTE" for row in domain.joints) and not objective.revolute_enabled:
        raise ValueError("revolute freedom requires a positive joint weight")
    proposal = RigidSE3ProposalPolicyPayload(
        profile_registration_sha256=canonical_sha256(
            RigidSE3ProfileRegistration(), domain=f"{PREFIX}/profile-registration"
        ),
        witness_hints=witness_hints,
    )
    schemas: dict[str, ValueSchemaDefinition] = {
        row.value_schema_ref: row for row in proof_value_schemas()
    }
    for name in ("body-id", "other-body-id", "contact-id", "region-id", "frame-id",
                 "other-frame-id", "joint-id"):
        schemas[schema(name)] = ValueSchemaDefinition.seal(
            value_schema_ref=schema(name), value_kind=ValueKind.CANONICAL_ID,
        )
    for ref in (BASE_SCHEMA, schema("state")):
        schemas[ref] = ValueSchemaDefinition.seal(value_schema_ref=ref, value_kind=ValueKind.RECORD)
    metadata: dict[str, dict[str, TypedValue | str]] = {}
    roles: dict[str, str] = {}
    state_definitions: list[StateVariableDefinition] = []
    for leaf in state.canonical_state_leaf_index.leaves:
        leaf_schema = leaf.state_variable_schema_ref
        schemas[leaf_schema] = ValueSchemaDefinition.seal(
            value_schema_ref=leaf_schema, value_kind=ValueKind.RECORD,
        )
        value_schema = schema("boolean" if leaf.field_path_ref.startswith("field-path:presence-") else "tree")
        ref = d("state/" + leaf_schema.rsplit("/", 1)[-1])
        row = StateVariableDefinition.seal(
            state_variable_ref=ref, state_variable_schema_ref=leaf_schema,
            value_schema_ref=value_schema, frame_ref=d("world"),
            unit_ref=d("metre"), topology_ref=d("closed"),
            write_authority=WriteAuthority.PRIMARY_WRITABLE,
        )
        state_definitions.append(row)
        roles[ref] = "state-variable"
        metadata[ref] = {
            "field:state-variable-schema": leaf_schema,
            "field:state-leaf-schema": leaf.state_schema_ref,
            "field:state-fact-family": leaf.fact_family_ref,
            "field:state-entity-key": leaf.entity_or_fact_key,
            "field:state-field-path": leaf.field_path_ref,
            "field:state-value-schema": value_schema,
            "field:state-frame": d("world"),
            "field:state-unit": d("metre"),
            "field:state-topology": d("closed"),
        }
    state_definitions = list(_ordered(tuple(state_definitions)))
    by_schema = {row.state_variable_schema_ref: row for row in state_definitions}

    predicate_definitions = []
    for name, operands in _PREDICATE_OPERANDS.items():
        ref = d("predicate/" + name.lower())
        roles[ref] = "predicate"
        predicate_definitions.append(PredicateDefinition.seal(
            predicate_ref=ref, operand_schema_refs=tuple(schema(item) for item in operands),
            state_context_policy_ref=d("context"), frame_requirement_ref=d("world"),
            measurement_definition_ref=d("measure/" + name.lower()),
            measurement_unit_ref=d("metre"), comparator_definition_ref=d("comparator/" + name.lower()),
            boundary_policy_ref=d("closed"), tolerance_policy_ref=d("zero-tolerance"),
            uncertainty_policy_ref=d("exact"), observation_prerequisite_template_refs=(),
            evaluator_capability_ref=c("evaluate/" + name.lower()),
            verifier_capability_ref=c("verify/" + name.lower()),
            admissible_claim_definition_refs=(d("certified"),),
        ))
    predicate_definitions = list(_ordered(tuple(predicate_definitions)))
    goal = after_goal.formula
    before = _ordered(before_preconditions)
    core_invariants = tuple(PreservationInvariant(
        before_formula=True, after_formula=True,
        transition_comparator_ref=d("invariant/" + name),
    ) for name in _CORE_INVARIANTS)
    invariants = _ordered((*core_invariants, *preservation_invariants))
    comparators = {d("comparator/preserve_truth"), d("comparator/require_true")}
    if any(row.transition_comparator_ref not in comparators for row in preservation_invariants):
        raise ValueError("M6 preservation invariant requires a registered truth comparator")
    if len({canonical_json_bytes(row) for row in invariants}) != len(invariants):
        raise ValueError("M6 preservation invariant roster must not repeat an obligation")
    for comparator in comparators:
        roles[comparator] = "invariant-comparator"
    for name in _CORE_INVARIANTS:
        ref = d("invariant/" + name)
        roles[ref] = "profile-core-invariant"
        metadata[ref] = {
            "field:core-invariant-kind": d("core-invariant-kind/" + name),
            "field:checker-capability": c("check"),
        }
    if BeforePrecondition(formula=NotFormula(formula=goal)) not in before:
        raise ValueError("M6 counterfactual source must explicitly negate its after goal")
    operator = OperatorDefinition.seal(
        operator_ref=d("operator/apply-edit-set"), parameter_schema_refs=(schema("tree"),),
        read_footprint=_ordered(tuple(StateAddressPattern(state_variable_definition_ref=row.state_variable_ref)
                                      for row in state_definitions)),
        primary_write_footprint=_ordered(tuple(StateAddressPattern(
            state_variable_definition_ref=by_schema[leaf.state_variable_schema_ref].state_variable_ref,
        ) for leaf in domain.authorized_primary_leaves)),
        derived_write_rule_refs=(), required_preconditions=before,
        transition_semantics_ref=d("transition/apply-edit-set"), generated_obligations=(),
        composability_policy_ref=d("ordered-atomic-sets"),
        endpoint_or_path_semantics_ref=d("endpoint-only"),
        compiler_capability_ref=c("compile"), verifier_capability_ref=c("check"),
    )
    roles[operator.operator_ref] = "operator"
    metadata[d("domain")] = {"field:domain": encode_tree(domain)}
    metadata[d("objective-policy")] = {"field:objective-policy": encode_tree(objective)}
    metadata[d("delta-policy")] = {"field:domain-reference": d("domain")}
    auth = InterventionAuthorization.seal(
        editable_entity_ids=domain.affected_body_ids,
        allowed_operator_refs=(operator.operator_ref,),
        authorized_primary_write_set=domain.authorized_primary_leaves,
        variable_bounds=(), maximum_program_steps=max(len(row.steps) for row in domain.program_skeletons),
        maximum_edited_entities=len(domain.affected_body_ids),
        required_derived_rule_refs=(), complete_state_delta_policy_ref=d("delta-policy"),
    )
    objective_names = ("translation", "rotation", "prismatic", "revolute", "contact", "step")
    objective_expression = ObjectiveExpression.seal(
        aggregation_definition_ref=d("objective/sum"),
        terms=tuple(ObjectiveTerm(
            term_id="term:" + name,
            objective_definition_ref=d("objective/" + name),
            input_selector_definition_ref=d("selector/" + name),
            unit_ref=d("dimensionless"), normalization_definition_ref=d("normalization/" + name),
        ) for name in objective_names),
        deterministic_tie_break_definition_ref=d("tie/canonical-program-bytes"),
    )
    for name in objective_names:
        metadata[d("objective/" + name)] = {
            "field:objective-input-selector": d("selector/" + name),
            "field:objective-unit": d("dimensionless"),
            "field:objective-normalization": d("normalization/" + name),
            "field:objective-policy-reference": d("objective-policy"),
        }
    numeric_ref = d("exact-rational")
    semantics = SemanticsProfile.seal(
        semantics_profile_ref=SEMANTICS_REF,
        accepted_scene_and_fact_schema_refs=_ordered((BASE_SCHEMA, schema("tree"))),
        predicate_definition_refs=_ordered(tuple(row.predicate_ref for row in predicate_definitions)),
        transition_semantics_refs=(operator.transition_semantics_ref,),
        objective_definition_refs=_ordered(tuple(d("objective/" + name) for name in objective_names)),
        numeric_semantics_ref=numeric_ref, completeness_policy_ref=d("closed-inventory"),
        uncertainty_policy_ref=d("exact-or-typed-unknown"), derived_fact_rule_refs=(),
        observation_obligation_policy_ref=d("no-observations"),
    )
    action = ActionSpaceProfile.seal(
        action_space_profile_ref=PROFILE_REF,
        accepted_scene_schema_refs=(BASE_SCHEMA,),
        state_variable_definition_refs=_ordered(tuple(row.state_variable_ref for row in state_definitions)),
        allowed_operator_refs=(operator.operator_ref,),
        mandatory_invariant_template_refs=_ordered(tuple({row.transition_comparator_ref for row in invariants})),
        predicate_capability_refs=_ordered(tuple(cap for row in predicate_definitions for cap in
                                                (row.evaluator_capability_ref, row.verifier_capability_ref))),
        objective_capability_refs=(c("objective"),), numeric_semantics_ref=numeric_ref,
        allowed_claim_definition_refs=_ordered((
            d("certified"), d("unsat"), d("witness"), d("unknown"))),
        backend_capability_requirements=(c("solve"),), adapter_capability_requirements=(),
        publication_proof_policy_ref=d("proof-policy"),
    )
    config = CounterfactualSolverConfig.seal(
        solver_config_ref=d("solver-config"), compilation_policy_ref=d("compile-policy"),
        proposal_policy_ref=d("proposal-policy"), objective_bound_policy_ref=d("outward-bounds"),
        determinism_policy_ref=d("deterministic"),
    )
    proof = ProofPolicy.seal(
        proof_policy_ref=d("proof-policy"),
        accepted_claim_definition_refs=_ordered((d("certified"), d("unsat"))),
        required_checker_capability_refs=(c("check"),),
        publication_minimum_claim_ref=d("certified"),
        permit_noncertified_terminal_records=True,
    )
    resource = ResourcePolicy.seal(
        resource_policy_ref=d("resource-policy"),
        limits=_ordered(tuple(ResourceLimit(
            definition_ref=d("resource/" + name), finite_limit=float(value),
        ) for name, value in limits.model_dump(mode="python").items())),
        exhaustion_claim_ref=d("resource-exhausted"),
        shared_ledger_policy_ref=d("cumulative-stage-ledger"),
    )
    routing = BackendRoutingPolicy.seal(
        routing_policy_ref=d("routing"), capability_filter_definition_ref=d("filter"),
        deterministic_order_definition_ref=d("order"),
        portfolio_composition_definition_ref=d("single-backend"),
        stop_condition_definition_ref=d("stop"), resource_partition_definition_ref=d("partition"),
    )
    metadata[d("proof-material")] = {
        "field:proof-payload-schema": schema("proof-bytes"),
        "field:proof-checker-capability": c("check"),
    }
    roles[d("proof-material")] = "proof-material"
    for name, role in (("certified", "claim-certified-solution"), ("unsat", "claim-proven-unsat")):
        roles[d(name)] = role
        metadata[d(name)] = {
            "field:claim-proof-material-definition": d("proof-material"),
            "field:claim-checker-capability": c("check"),
        }
    roles[d("witness")] = "claim-noncertified-witness"
    for name in ("complete-domain", "sound-complete-domain"):
        roles[d(name)] = name
        metadata[d(name)] = {"field:complete-domain-claim": d("unsat")}
    roles[d("resource-policy")] = "resource-policy"
    metadata[d("resource-policy")] = {
        "field:resource-accounting-claim": d("accounting"),
        "field:limits": encode_tree(limits),
    }
    roles[d("accounting")] = "resource-accounting"
    for name in ("resource-exhausted", "numeric-gap", "unsupported", "unknown",
                 "continuous-domain-not-closed", "missing-source-fact"):
        roles[d(name)] = "claim-unknown"
    metadata[d("proposal-policy")] = {
        "field:unknown-claim": d("unknown"),
        "field:numeric-reason": d("numeric-gap"),
        "field:unsupported-reason": d("unsupported"),
        "field:continuous-reason": d("continuous-domain-not-closed"),
        "field:missing-source-reason": d("missing-source-fact"),
        "field:witness-hints": encode_tree(proposal),
    }
    roles[d("routing")] = "routing-policy"
    metadata[d("routing")] = {
        "field:routing-match-claim": d("match"),
        "field:routing-mismatch-claim": d("mismatch"),
        "field:selection-disposition": d("selection-disposition"),
        "field:selection-reason": d("selection-reason"),
    }
    roles.update({
        d("match"): "routing-match", d("mismatch"): "routing-mismatch",
        d("selection-disposition"): "routing-selection-disposition",
        d("selection-reason"): "routing-selection-reason",
    })
    backend_descriptor = SolverBackendDescriptor.seal(
        backend_ref=BACKEND_REF, implementation_build_sha256=BUILDS["backend"],
        supported_profile_hashes=_ordered((semantics.semantics_profile_sha256,
                                           action.action_space_profile_sha256)),
        supported_predicate_capabilities=_ordered(tuple(cap for row in predicate_definitions
                                                         for cap in (row.evaluator_capability_ref,
                                                                     row.verifier_capability_ref))),
        supported_operator_capabilities=_ordered((operator.compiler_capability_ref,
                                                  operator.verifier_capability_ref)) if backend_enabled else (),
        supported_objective_capabilities=action.objective_capability_refs,
        supported_numeric_semantics=(numeric_ref,),
        emitted_proof_material_definition_refs=(d("proof-material"),),
        compatible_checker_capability_refs=(c("check"),),
        resource_definition_refs=_ordered(tuple(row.definition_ref for row in resource.limits)),
    )
    backend_bundle = BackendDescriptorBundle.seal(
        backend_descriptors=(backend_descriptor,), unavailable_optional_backends=(),
    )
    semantic_roots = (
        state, auth, before, AfterGoal(formula=goal), invariants,
        objective_expression, semantics, action, *predicate_definitions,
        *state_definitions, operator,
    )
    semantic_refs = set().union(*(generic._references(root, "definition:") for root in semantic_roots))
    semantic_refs.discard(d("proof-policy"))
    semantic_refs.update((d("proof-material"), d("complete-domain"),
                          d("sound-complete-domain")))
    solve_roots = (config, proof, resource, routing, backend_bundle)
    solve_refs = set().union(*(generic._references(root, "definition:") for root in solve_roots)) - semantic_refs
    solve_refs.update((d("accounting"), d("match"), d("mismatch"),
                       d("selection-disposition"), d("selection-reason")))
    for references in (semantic_refs, solve_refs):
        while True:
            prior = len(references)
            for ref in tuple(references):
                references.update(generic._references(metadata.get(ref, {}), "definition:") -
                                  (semantic_refs if references is solve_refs else set()))
            if len(references) == prior:
                break
    field_schemas: dict[str, str] = {
        "field:definition-kind": schema("definition-kind"),
        "field:definition-reference": schema("id"),
        "field:bound-record-ref": schema("id"),
        "field:bound-record-sha256": schema("digest"),
    }
    for fields in metadata.values():
        for name, value in fields.items():
            expected = value.value_schema_ref if isinstance(value, TypedValue) else schema("id")
            prior = field_schemas.setdefault(name, expected)
            if prior != expected:
                raise ValueError("M6 definition field has inconsistent value schema")
    kinds = _ordered(tuple({"definition-kind:generic", "definition-kind:record-binding",
                            *("definition-kind:" + role for role in roles.values())}))
    schemas[schema("definition-kind")] = ValueSchemaDefinition.seal(
        value_schema_ref=schema("definition-kind"), value_kind=ValueKind.ENUM_SYMBOL,
        enum_symbols=kinds,
    )
    schemas[schema("digest")] = ValueSchemaDefinition.seal(
        value_schema_ref=schema("digest"), value_kind=ValueKind.DIGEST,
    )
    schemas[schema("definition-record")] = ValueSchemaDefinition.seal(
        value_schema_ref=schema("definition-record"), value_kind=ValueKind.RECORD,
        fields=tuple(ValueFieldDefinition(
            field_name=name, value_schema_ref=ref,
            required=name in ("field:definition-kind", "field:definition-reference"),
        ) for name, ref in sorted(field_schemas.items())),
    )

    def envelope(ref: str, anchor: str, kind: str = "generic",
                 fields: Mapping[str, TypedValue | str] | None = None) -> CanonicalDefinitionEnvelope:
        values: dict[str, TypedValue] = {
            "field:definition-kind": TypedValue(
                value_schema_ref=schema("definition-kind"),
                payload=EnumSymbolValue(symbol="definition-kind:" + kind),
            ),
            "field:definition-reference": _id_value(ref),
        }
        values.update({name: value if isinstance(value, TypedValue) else _id_value(value)
                       for name, value in (fields or {}).items()})
        return CanonicalDefinitionEnvelope.seal(
            definition_ref=ref, definition_kind_ref=anchor,
            payload_schema_ref=schema("definition-record"),
            payload=_record(values, "definition-record"),
        )

    semantic_anchor = d("semantic-bootstrap")
    solve_anchor = d("solve-bootstrap")
    semantic_refs.add(semantic_anchor)
    solve_refs.add(solve_anchor)
    semantic_definitions = [envelope(ref, semantic_anchor, roles.get(ref, "generic"), metadata.get(ref))
                            for ref in sorted(semantic_refs)]
    specialized_records = (
        *((row.value_schema_ref, row) for row in schemas.values()),
        *((row.predicate_ref, row) for row in predicate_definitions),
        *((row.state_variable_ref, row) for row in state_definitions),
        (operator.operator_ref, operator),
        (semantics.semantics_profile_ref, semantics),
        (action.action_space_profile_ref, action),
    )
    for ref, record in specialized_records:
        digest = getattr(record, record.SELF_DIGEST_FIELD)
        binding_ref = d("binding/" + canonical_sha256(
            (ref, digest), domain=f"{PREFIX}/record-binding",
        ))
        semantic_definitions.append(envelope(
            binding_ref, semantic_anchor, "record-binding", {
                "field:bound-record-ref": ref,
                "field:bound-record-sha256": TypedValue(
                    value_schema_ref=schema("digest"), payload=DigestValue(value=digest),
                ),
            },
        ))
    semantic_bundle = DefinitionBundle.seal(
        definitions=tuple(sorted(semantic_definitions, key=lambda row: canonical_json_bytes(row.definition_ref))),
    )
    solve_bundle = DefinitionBundle.seal(
        definitions=tuple(envelope(ref, solve_anchor, roles.get(ref, "generic"), metadata.get(ref))
                          for ref in sorted(solve_refs)),
    )
    capabilities = {
        "profile": (c("profile"),),
        "compiler": (c("compile"),),
        "kernel": _ordered((c("objective"), *(cap for row in predicate_definitions
                                                   for cap in (row.evaluator_capability_ref,
                                                               row.verifier_capability_ref)))),
        "backend": (c("solve"),),
        "checker": (c("check"),),
    }
    registry = generic.StaticImplementationRegistry(tuple(
        generic.StaticOwner(OWNERS[name], BUILDS[name], capabilities[name],
                            "spatialcf.rigid_se3." + name)
        for name in sorted(OWNERS)
    ))
    snapshot = ImplementationRegistrySnapshot.seal(
        definition_and_capability_owner_bindings=_ordered(tuple(
            ImplementationOwnerBinding(
                definition_or_capability_ref=cap,
                implementation_owner_ref=OWNERS[name],
            ) for name, values in capabilities.items() for cap in values
        )),
        implementation_build_hashes=_ordered(tuple((OWNERS[name], BUILDS[name])
                                                 for name in OWNERS)),
        dependency_lock_sha256=canonical_sha256(
            ("python:3.11", "canonical-v2", "fraction"),
            domain=f"{PREFIX}/dependency-contract",
        ),
    )
    problem = CounterfactualProblemIR.seal(
        problem_id=problem_id, scene_state=state, definition_bundle=semantic_bundle,
        semantics_profile_ref=SEMANTICS_REF, action_space_profile_ref=PROFILE_REF,
        intervention_authorization=auth, before_preconditions=before,
        after_goal=AfterGoal(formula=goal), preservation_invariants=invariants,
        explicit_observation_obligations=(), objective_expression=objective_expression,
        numeric_semantics_ref=numeric_ref,
    )
    request = CounterfactualSolveRequest.seal(
        semantic_problem=problem, semantic_problem_sha256=problem.semantic_problem_sha256,
        solve_policy_definition_bundle=solve_bundle,
        implementation_registry_snapshot=snapshot,
        backend_descriptor_bundle=backend_bundle,
        solver_config=config, proof_policy=proof, resource_policy=resource,
        backend_routing_policy=routing,
    )
    registry_arguments = {
        "semantic_definition_bundle": semantic_bundle,
        "solve_policy_definition_bundle": solve_bundle,
        "value_schema_definitions": _ordered(tuple(schemas.values())),
        "predicate_definitions": tuple(predicate_definitions),
        "state_variable_definitions": tuple(state_definitions),
        "derived_fact_rule_definitions": (),
        "operator_definitions": (operator,),
        "implementation_registry_snapshot": snapshot,
        "problem": problem, "semantics_profile": semantics, "action_space_profile": action,
        "scene_state": state, "request": request,
        "backend_descriptor_bundle": backend_bundle,
        "solver_config": config, "proof_policy": proof, "resource_policy": resource,
        "backend_routing_policy": routing,
    }
    context = RigidSE3RequestContext(
        request=request, state_view=state_view, domain=domain,
        objective=objective, limits=limits, registry=registry,
        registry_arguments=registry_arguments,
    )
    registry.validate_solve_request(**registry_arguments)
    return context


class RigidSE3MissingSource(ValueError):
    """Typed unresolved source addresses; no infeasibility claim follows."""

    def __init__(self, addresses: tuple[str, ...]):
        super().__init__("M6 exact source facts are unavailable")
        self.addresses = addresses


def source_cells_from_state(state: SceneStateEnvelope) -> RigidSE3SourceCells:
    """Reconstruct every declared source cell from the original General IR."""
    facts = tuple(fact for bundle in state.extension_fact_bundles for fact in bundle.facts)
    inventory_facts = tuple(fact for fact in facts
                            if fact.fact_family_ref == FAMILY["inventory"])
    if len(inventory_facts) != 1 or inventory_facts[0].fact_key != INVENTORY_KEY:
        raise ValueError("M6 state requires exactly one immutable inventory fact")
    inventory_raw = decode_tree(inventory_facts[0].value)
    if (not isinstance(inventory_raw, dict)
            or inventory_raw.get("availability") != "KNOWN_EXACT"
            or inventory_raw.get("uncertainty_upper") is not None):
        raise ValueError("M6 inventory must be exact and available")
    inventory = RigidSE3Inventory.model_validate(inventory_raw["value"], strict=True)
    by_family = {value: name for name, value in FAMILY.items()}
    cells = []
    for fact in facts:
        if fact.fact_family_ref == FAMILY["inventory"]:
            continue
        family = by_family.get(fact.fact_family_ref)
        if family is None:
            raise ValueError("M6 state contains an unknown extension family")
        raw = decode_tree(fact.value)
        if not isinstance(raw, dict) or set(raw) != {
            "availability", "value", "uncertainty_upper",
        }:
            raise ValueError("M6 extension cell is not a complete typed record")
        cells.append(RigidSE3SourceFactInput(
            family=family, subject_entity_id=fact.subject_entity_id,
            fact_key=fact.fact_key,
            cell=RigidSE3FactCell(
                availability=raw["availability"],
                value=encode_tree(raw["value"]) if raw["value"] is not None else None,
                uncertainty_upper=raw["uncertainty_upper"],
            ),
        ))
    source = RigidSE3SourceCells(
        inventory=inventory,
        facts=tuple(sorted(cells, key=lambda row: (
            row.family, row.subject_entity_id, row.fact_key,
        ))),
    )
    if build_rigid_se3_state(state.base_scene_payload, source).state != state:
        raise ValueError("M6 source state differs from its complete source-cell reconstruction")
    return source


def exact_source_from_cells(source: RigidSE3SourceCells) -> RigidSE3ExactSourceFacts:
    """Return a complete typed graph, or preserve the actual missing addresses."""
    if any(row.cell.availability != "KNOWN_EXACT" for row in source.facts):
        addresses = tuple(sorted(generic._extension_fact_address(
            FAMILY[row.family], row.subject_entity_id, row.fact_key,
        ) for row in source.facts if row.cell.availability != "KNOWN_EXACT"))
        raise RigidSE3MissingSource(addresses)
    classes = {
        "body": RigidSE3BodyFact, "box": RigidSE3Box,
        "joint": RigidSE3JointFact, "contact": RigidSE3ContactFact,
        "region": RigidSE3RegionFact,
        "root-x": RigidSE3Rational, "root-y": RigidSE3Rational,
        "root-z": RigidSE3Rational, "root-rotation": RigidSE3Rotation,
    }
    exact = {}
    for row in source.facts:
        assert row.cell.value is not None
        raw = decode_tree(row.cell.value)
        if row.family in classes:
            value = classes[row.family].model_validate(raw, strict=True)
        elif row.family == "joint-state":
            joint = next(item for item in source.facts
                         if item.family == "joint" and item.fact_key == row.fact_key)
            joint_fact = RigidSE3JointFact.model_validate(decode_tree(joint.cell.value), strict=True)
            cls = RigidSE3Rational if joint_fact.kind == "PRISMATIC" else RigidSE3CirclePoint
            value = cls.model_validate(raw, strict=True)
        else:
            value = raw
        exact[(row.family, row.fact_key)] = value

    def rows(family: str) -> tuple:
        return tuple(exact[(family, row.fact_key)] for row in sorted(
            (item for item in source.facts if item.family == family),
            key=lambda item: item.fact_key,
        ))

    root_ids = tuple(row.fact_key for row in source.facts if row.family == "root-x")
    roots = tuple(RigidSE3RootState(
        body_id=body_id,
        world_pose=RigidSE3Pose(
            frame="WORLD",
            translation_m=RigidSE3Vector(
                x=exact[("root-x", body_id)], y=exact[("root-y", body_id)],
                z=exact[("root-z", body_id)],
            ),
            rotation=exact[("root-rotation", body_id)],
        ),
    ) for body_id in sorted(root_ids))
    joints = {row.joint_id: row for row in rows("joint")}
    joint_states = tuple(RigidSE3JointState(
        joint_id=row.fact_key, kind=joints[row.fact_key].kind,
        offset_m=exact[("joint-state", row.fact_key)]
        if joints[row.fact_key].kind == "PRISMATIC" else None,
        circle_point=exact[("joint-state", row.fact_key)]
        if joints[row.fact_key].kind == "REVOLUTE" else None,
    ) for row in sorted((item for item in source.facts if item.family == "joint-state"),
                        key=lambda item: item.fact_key))
    return RigidSE3ExactSourceFacts(
        bodies=rows("body"), boxes=rows("box"), joints=rows("joint"),
        contacts=rows("contact"), regions=rows("region"), roots=roots,
        joint_states=joint_states,
        contact_states=tuple(RigidSE3ContactState(
            contact_id=row.fact_key, mode=exact[("contact-mode", row.fact_key)],
        ) for row in sorted((item for item in source.facts if item.family == "contact-mode"),
                            key=lambda item: item.fact_key)),
    )


def _definition_field(bundle: DefinitionBundle, ref: str, name: str) -> TypedValue:
    definitions = {row.definition_ref: row for row in bundle.definitions}
    try:
        return next(field.value for field in definitions[ref].payload.payload.fields
                    if field.name == name)
    except (KeyError, AttributeError, StopIteration) as error:
        raise ValueError(f"M6 request lacks required source-bound definition {ref}/{name}") from error


def validate_rigid_se3_request(request: CounterfactualSolveRequest) -> RigidSE3RequestContext:
    """Recompose from original roots; submitted metadata cannot define itself."""
    if type(request) is not CounterfactualSolveRequest:
        raise TypeError("M6 requires an exact CounterfactualSolveRequest")
    preflight_m6_ingress(request, scalar_bytes=MAX_REQUEST_SCALAR_BYTES)
    CounterfactualSolveRequest.model_validate(request.model_dump(mode="python"), strict=True)
    problem = request.semantic_problem
    if problem.action_space_profile_ref != PROFILE_REF:
        raise ValueError("request does not select the M6 action-space profile")
    source = source_cells_from_state(problem.scene_state)
    domain = RigidSE3Domain.model_validate(decode_tree(_definition_field(
        problem.definition_bundle, d("domain"), "field:domain",
    )), strict=True)
    objective = RigidSE3ObjectivePolicy.model_validate(decode_tree(_definition_field(
        problem.definition_bundle, d("objective-policy"), "field:objective-policy",
    )), strict=True)
    limits = RigidSE3Limits.model_validate(decode_tree(_definition_field(
        request.solve_policy_definition_bundle, d("resource-policy"), "field:limits",
    )), strict=True)
    proposal = RigidSE3ProposalPolicyPayload.model_validate(decode_tree(_definition_field(
        request.solve_policy_definition_bundle, d("proposal-policy"), "field:witness-hints",
    )), strict=True)
    descriptors = request.backend_descriptor_bundle.backend_descriptors
    if len(descriptors) != 1:
        raise ValueError("M6 requires one exact frozen backend descriptor")
    rebuilt = build_rigid_se3_request(
        scene=problem.scene_state.base_scene_payload, source=source,
        domain=domain, objective=objective, after_goal=problem.after_goal,
        before_preconditions=problem.before_preconditions,
        preservation_invariants=tuple(row for row in problem.preservation_invariants
                                      if row.transition_comparator_ref not in {
                                          d("invariant/" + name) for name in _CORE_INVARIANTS
                                      }),
        witness_hints=proposal.witness_hints, limits=limits,
        backend_enabled=bool(descriptors[0].supported_operator_capabilities),
        problem_id=problem.problem_id,
    )
    if rebuilt.request != request:
        raise ValueError("M6 request differs from trusted source-bound recomposition")
    return rebuilt


def _operand_id(value: TypedValue) -> str:
    if not isinstance(value.payload, CanonicalIdValue):
        raise ValueError("M6 predicate ID operand is not an exact canonical ID")  # noqa: TRY004 - stable validation error
    return value.payload.value


def _member_translation(point: RigidSE3Vector, domain, budget: kernel.ExactBudget) -> bool:
    if domain.kind == "FINITE":
        return point in domain.values
    assert domain.lower_m is not None and domain.upper_m is not None
    return all(budget.compare(lower, value) <= 0 and budget.compare(value, upper) <= 0
               for lower, value, upper in zip(
                   domain.lower_m.fractions, point.fractions,
                   domain.upper_m.fractions, strict=True,
               ))


def _root_member(root: RigidSE3RootState, domain: RigidSE3Domain,
                 budget: kernel.ExactBudget) -> bool:
    choices = {row.root_body_id: row for row in domain.roots}
    option = choices.get(root.body_id)
    if option is None:
        return False
    return (_member_translation(root.world_pose.translation_m, option.translation, budget)
            and (option.rotation.kind == "ALL_SO3"
                 or root.world_pose.rotation in option.rotation.values))


def _joint_member(state: RigidSE3JointState, domain: RigidSE3Domain,
                  budget: kernel.ExactBudget) -> bool:
    choices = {row.joint_id: row for row in domain.joints}
    option = choices.get(state.joint_id)
    if option is None or option.kind != state.kind:
        return False
    if state.kind == "PRISMATIC":
        assert state.offset_m is not None and option.prismatic is not None
        values = option.prismatic
        if values.kind == "FINITE":
            return state.offset_m in values.values_m
        assert values.lower_m is not None and values.upper_m is not None
        return (budget.compare(values.lower_m.as_fraction, state.offset_m.as_fraction) <= 0
                and budget.compare(state.offset_m.as_fraction, values.upper_m.as_fraction) <= 0)
    assert state.circle_point is not None and option.revolute is not None
    return option.revolute.kind == "FULL_CIRCLE" or state.circle_point in option.revolute.values


def _evaluate_atom(atom: PredicateAtom, source: RigidSE3ExactSourceFacts,
                   domain: RigidSE3Domain, poses: Mapping[str, kernel.Transform],
                   boxes: tuple[kernel.WorldBox, ...],
                   budget: kernel.ExactBudget) -> bool:
    prefix = d("predicate/")
    if not atom.predicate_ref.startswith(prefix):
        raise ValueError("unregistered M6 predicate definition")
    name = atom.predicate_ref[len(prefix):].upper()
    if name not in _PREDICATE_OPERANDS or len(atom.operands) != len(_PREDICATE_OPERANDS[name]):
        raise ValueError("M6 predicate signature is outside the registered profile")
    budget.charge("PREDICATE")
    by_box = {row.primitive_id: row for row in boxes}
    contacts = {row.contact_id: row for row in source.contacts}
    contact_modes = {row.contact_id: row.mode for row in source.contact_states}
    if name == "BODY_SEPARATED":
        first, second = (_operand_id(value) for value in atom.operands)
        if first == second or first not in poses or second not in poses:
            raise ValueError("BODY_SEPARATED requires two distinct known bodies")
        separated = True
        for a in boxes:
            if a.body_id != first:
                continue
            for b in boxes:
                if b.body_id == second:
                    budget.charge("COLLISION_PAIR")
                    if kernel.obb_interiors_overlap(a, b, budget):
                        separated = False
        return separated
    if name in {"FACE_CONTACT", "CONTACT_RELEASED"}:
        contact_id = _operand_id(atom.operands[0])
        if contact_id not in contacts:
            raise ValueError("unknown M6 contact predicate operand")
        mode = "ENGAGED" if name == "FACE_CONTACT" else "RELEASED"
        return contact_modes[contact_id] == mode and kernel.contact_satisfied(
            contacts[contact_id], mode, by_box, budget,
        )
    if name == "BODY_IN_REGION":
        body_id, region_id = (_operand_id(value) for value in atom.operands)
        regions = {row.region_id: row for row in source.regions}
        if body_id not in poses or region_id not in regions:
            raise ValueError("unknown M6 body/region predicate operand")
        return kernel.body_in_region(body_id, boxes, regions[region_id], budget)
    if name == "FRAME_OFFSET_EQUALS":
        first, second = (_operand_id(value) for value in atom.operands[:2])
        offset = RigidSE3Vector.model_validate(decode_tree(atom.operands[2]), strict=True)
        origins = dict(poses)
        origins["frame:world"] = kernel.Transform()
        if first not in origins or second not in origins:
            raise ValueError("unknown M6 body/world frame origin")
        return kernel.frame_offset_equals(first, second, offset.fractions, origins, budget)
    if name == "ROOT_POSE_MEMBER":
        body_id = _operand_id(atom.operands[0])
        root = next((row for row in source.roots if row.body_id == body_id), None)
        if root is None:
            raise ValueError("ROOT_POSE_MEMBER requires a known root")
        return _root_member(root, domain, budget)
    if name == "JOINT_STATE_MEMBER":
        joint_id = _operand_id(atom.operands[0])
        state = next((row for row in source.joint_states if row.joint_id == joint_id), None)
        if state is None:
            raise ValueError("JOINT_STATE_MEMBER requires a known actuated joint")
        return _joint_member(state, domain, budget)
    raise ValueError("unregistered M6 predicate definition")


def evaluate_rigid_se3_formula(formula: object, source: RigidSE3ExactSourceFacts,
                               domain: RigidSE3Domain,
                               poses: Mapping[str, kernel.Transform],
                               boxes: tuple[kernel.WorldBox, ...],
                               budget: kernel.ExactBudget) -> bool:
    """Evaluate the full grounded Boolean AST over only registered exact atoms."""
    if isinstance(formula, bool):
        return formula
    if isinstance(formula, (ForAllFormula, ExistsFormula)):
        return evaluate_rigid_se3_formula(formula.ground(), source, domain, poses, boxes, budget)
    if isinstance(formula, PredicateAtom):
        return _evaluate_atom(formula, source, domain, poses, boxes, budget)
    if isinstance(formula, NotFormula):
        return not evaluate_rigid_se3_formula(formula.formula, source, domain, poses, boxes, budget)
    if isinstance(formula, AllOfFormula):
        values = tuple(evaluate_rigid_se3_formula(item, source, domain, poses, boxes, budget)
                       for item in formula.formulas)
        return all(values)
    if isinstance(formula, AnyOfFormula):
        values = tuple(evaluate_rigid_se3_formula(item, source, domain, poses, boxes, budget)
                       for item in formula.formulas)
        return any(values)
    if isinstance(formula, ImpliesFormula):
        antecedent = evaluate_rigid_se3_formula(
            formula.antecedent, source, domain, poses, boxes, budget,
        )
        consequent = evaluate_rigid_se3_formula(
            formula.consequent, source, domain, poses, boxes, budget,
        )
        return not antecedent or consequent
    if isinstance(formula, ExactlyKFormula):
        return sum(evaluate_rigid_se3_formula(item, source, domain, poses, boxes, budget)
                   for item in formula.formulas) == formula.k
    raise ValueError("unsupported M6 Boolean formula")


def _guard_exact_values(value: object, budget: kernel.ExactBudget) -> None:
    if isinstance(value, RigidSE3Rational):
        budget.charge("EXACT_ARITHMETIC")
        budget.guard(value.as_fraction)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _guard_exact_values(getattr(value, name), budget)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _guard_exact_values(item, budget)
    elif isinstance(value, dict):
        for key in sorted(value):
            _guard_exact_values(value[key], budget)


def _state_truth(
    source: RigidSE3ExactSourceFacts,
    domain: RigidSE3Domain,
    problem: CounterfactualProblemIR,
    budget: kernel.ExactBudget,
    *,
    prefix_index: int,
    reference_source: RigidSE3ExactSourceFacts,
    source_invariant_values: tuple[bool, ...] | None = None,
    check_domain: bool = True,
):
    poses = kernel.forward_kinematics(source, budget)
    boxes = kernel.world_boxes(source, poses, budget)
    by_box = {row.primitive_id: row for row in boxes}
    modes = {row.contact_id: row.mode for row in source.contact_states}
    contact_truth = tuple(RigidSE3ContactTruth(
        contact_id=contact.contact_id,
        declared_mode=modes[contact.contact_id],
        geometrically_satisfied=kernel.contact_satisfied(
            contact, modes[contact.contact_id], by_box, budget,
        ),
    ) for contact in source.contacts)
    domain_roots = {row.root_body_id for row in domain.roots}
    domain_joints = {row.joint_id for row in domain.joints}
    domain_modes = {row.contact_id: row.modes for row in domain.contacts}
    reference_roots = {row.body_id: row for row in reference_source.roots}
    reference_joints = {row.joint_id: row for row in reference_source.joint_states}
    reference_contacts = {row.contact_id: row for row in reference_source.contact_states}
    root_truth = tuple(_root_member(row, domain, budget) if row.body_id in domain_roots
                       else row == reference_roots[row.body_id] for row in source.roots)
    joint_truth = tuple(_joint_member(row, domain, budget) if row.joint_id in domain_joints
                        else row == reference_joints[row.joint_id]
                        for row in source.joint_states)
    contact_domain_truth = tuple(
        row.mode in domain_modes[row.contact_id] if row.contact_id in domain_modes
        else row == reference_contacts[row.contact_id]
        for row in source.contact_states
    )
    mutable = all(root_truth) and all(joint_truth) and all(contact_domain_truth)
    immutable = (
        source.bodies == reference_source.bodies
        and source.boxes == reference_source.boxes
        and source.joints == reference_source.joints
        and source.contacts == reference_source.contacts
        and source.regions == reference_source.regions
    )
    core = {
        "kinematics": set(poses) == {row.body_id for row in source.bodies},
        "all-body-collision": kernel.collision_free(boxes, budget),
        "contact-consistency": all(row.geometrically_satisfied for row in contact_truth),
        "endpoint-domain": mutable if check_domain else True,
        "complete-state-authority": immutable,
    }
    truths = [RigidSE3PrefixTruth(
        prefix_index=prefix_index,
        obligation_id=d("invariant/" + name),
        satisfied=core[name],
    ) for name in _CORE_INVARIANTS]
    for index, invariant in enumerate(problem.preservation_invariants):
        if invariant.transition_comparator_ref.startswith(d("invariant/")):
            continue
        before_value = (
            evaluate_rigid_se3_formula(
                invariant.before_formula, reference_source, domain, poses, boxes, budget,
            ) if source_invariant_values is None else source_invariant_values[index]
        )
        after_value = evaluate_rigid_se3_formula(
            invariant.after_formula, source, domain, poses, boxes, budget,
        )
        if invariant.transition_comparator_ref == d("comparator/preserve_truth"):
            satisfied = after_value == before_value
        elif invariant.transition_comparator_ref == d("comparator/require_true"):
            satisfied = before_value and after_value
        else:
            raise ValueError("unregistered M6 preservation comparator")
        truths.append(RigidSE3PrefixTruth(
            prefix_index=prefix_index,
            obligation_id=d(f"preservation/{index:08d}"),
            satisfied=satisfied,
        ))
    world_poses = tuple((body_id, RigidSE3Pose(
        frame="WORLD",
        rotation=RigidSE3Rotation(rows=tuple(tuple(
            RigidSE3Rational.from_fraction(value) for value in row
        ) for row in transform.rotation)),
        translation_m=RigidSE3Vector(**{
            axis: RigidSE3Rational.from_fraction(value)
            for axis, value in zip(("x", "y", "z"), transform.translation, strict=True)
        }),
    )) for body_id, transform in poses.items())
    return tuple(truths), world_poses, contact_truth, poses, boxes


def _compile_product_count(domain: RigidSE3Domain, budget: kernel.ExactBudget,
                           limit: int) -> None:
    if not domain.finite:
        return
    roots = {row.root_body_id: row for row in domain.roots}
    joints = {row.joint_id: row for row in domain.joints}
    contacts = {row.contact_id: row for row in domain.contacts}
    total = 0
    for program in domain.program_skeletons:
        count = 1
        for step in program.steps:
            for member in step.members:
                if member.kind == "SET_ROOT_SE3":
                    selected = roots[member.target_id]
                    width = len(selected.translation.values) * len(selected.rotation.values)
                elif member.kind == "SET_JOINT":
                    selected = joints[member.target_id]
                    width = (len(selected.prismatic.values_m) if selected.prismatic is not None
                             else len(selected.revolute.values))
                else:
                    width = len(contacts[member.target_id].modes)
                if width < 1 or count > limit // width:
                    raise kernel.ExactKernelLimit("RESOURCE_LIMIT", program.skeleton_id)
                count *= width
        if total > limit - count:
            raise kernel.ExactKernelLimit("RESOURCE_LIMIT", program.skeleton_id)
        total += count
        budget.charge("DOMAIN_PRODUCT")


def compile_rigid_se3(
    request: CounterfactualSolveRequest,
    *, audit: kernel.ExactBudget | None = None,
) -> RigidSE3Compilation:
    """Revalidate source and finite program universe without selecting an endpoint."""
    context = validate_rigid_se3_request(request)
    source = exact_source_from_cells(source_cells_from_state(context.state_view.state))
    limits = context.limits
    budget = kernel.ExactBudget(limits, "COMPILE", audit=audit)
    problem = request.semantic_problem
    try:
        for kind, count, cap in (
            ("ENTITY", len(source.bodies), limits.max_entities),
            ("PRIMITIVE", len(source.boxes), limits.max_primitives),
            ("JOINT", len(source.joints), limits.max_joints),
            ("PROGRAM", len(context.domain.program_skeletons), limits.max_programs),
        ):
            if count > cap:
                raise kernel.ExactKernelLimit("RESOURCE_LIMIT", kind)
            if count:
                budget.charge(kind, count)
        for program in context.domain.program_skeletons:
            if len(program.steps) > limits.max_steps:
                raise kernel.ExactKernelLimit("RESOURCE_LIMIT", program.skeleton_id)
            budget.charge("STEP", len(program.steps))
        _guard_exact_values((source, context.domain, context.objective), budget)
        _compile_product_count(context.domain, budget, limits.max_domain_products)
        truths, _, _, poses, boxes = _state_truth(
            source, context.domain, problem, budget,
            prefix_index=0, reference_source=source, check_domain=False,
        )
        if any(not row.satisfied for row in truths):
            raise ValueError("M6 source violates an immutable or requested invariant")
        for before in problem.before_preconditions:
            if not evaluate_rigid_se3_formula(
                before.formula, source, context.domain, poses, boxes, budget,
            ):
                raise ValueError("M6 before precondition fails on the exact source")
        if evaluate_rigid_se3_formula(
            problem.after_goal.formula, source, context.domain, poses, boxes, budget,
        ):
            raise ValueError("M6 requested after goal already holds at the source")
        ledger = budget.as_ledger()
    except kernel.CheckKernelLimit:
        raise
    except kernel.ExactKernelLimit as error:
        ledger = budget.as_ledger(
            first_unprocessed_item=f"item:compile:{error.item}", reason=error.reason,
        )
    return RigidSE3Compilation.seal(
        semantic_problem_sha256=request.semantic_problem_sha256,
        solve_request_sha256=request.solve_request_sha256,
        source_state_sha256=context.state_view.state.scene_state_sha256,
        source_state=context.state_view.state,
        inventory_sha256=context.state_view.inventory.inventory_sha256,
        domain=context.domain, limits=limits, ledger=ledger,
    )


def _checked_skeleton(domain: RigidSE3Domain, skeleton_id: str,
                      choices: tuple[RigidSE3StepChoice, ...],
                      budget: kernel.ExactBudget) -> None:
    skeletons = {row.skeleton_id: row for row in domain.program_skeletons}
    skeleton = skeletons.get(skeleton_id)
    if skeleton is None or len(skeleton.steps) != len(choices):
        raise ValueError("M6 program skeleton or order is outside the domain")
    root_domains = {row.root_body_id: row for row in domain.roots}
    joint_domains = {row.joint_id: row for row in domain.joints}
    contact_domains = {row.contact_id: row for row in domain.contacts}
    for template, choice in zip(skeleton.steps, choices, strict=True):
        if tuple((m.kind, m.target_id) for m in template.members) != tuple(
            (m.kind, m.target_id) for m in choice.members
        ):
            raise ValueError("M6 choice differs from authorized skeleton order")
        for member in choice.members:
            _guard_exact_values(member, budget)
            if member.kind == "SET_ROOT_SE3":
                if member.target_id not in root_domains or not _root_member(
                    RigidSE3RootState(body_id=member.target_id, world_pose=member.pose),
                    domain, budget,
                ):
                    raise ValueError("M6 root choice is outside the domain")
            elif member.kind == "SET_JOINT":
                selected = joint_domains.get(member.target_id)
                if selected is None or (selected.kind == "PRISMATIC") != (member.offset_m is not None):
                    raise ValueError("M6 joint choice has the wrong domain kind")
                if not _joint_member(RigidSE3JointState(
                    joint_id=member.target_id, kind=selected.kind,
                    offset_m=member.offset_m, circle_point=member.circle_point,
                ), domain, budget):
                    raise ValueError("M6 joint choice is outside the domain")
            elif (member.target_id not in contact_domains or
                  member.mode not in contact_domains[member.target_id].modes):
                raise ValueError("M6 contact choice is outside the domain")


def _apply_atomic(source: RigidSE3ExactSourceFacts,
                  choice: RigidSE3StepChoice) -> RigidSE3ExactSourceFacts:
    roots = {row.body_id: row for row in source.roots}
    joints = {row.joint_id: row for row in source.joint_states}
    contacts = {row.contact_id: row for row in source.contact_states}
    joint_kinds = {row.joint_id: row.kind for row in source.joints}
    for member in choice.members:
        if member.kind == "SET_ROOT_SE3":
            roots[member.target_id] = RigidSE3RootState(
                body_id=member.target_id, world_pose=member.pose,
            )
        elif member.kind == "SET_JOINT":
            kind = joint_kinds[member.target_id]
            joints[member.target_id] = RigidSE3JointState(
                joint_id=member.target_id, kind=kind,
                offset_m=member.offset_m, circle_point=member.circle_point,
            )
        else:
            contacts[member.target_id] = RigidSE3ContactState(
                contact_id=member.target_id, mode=member.mode,
            )
    return RigidSE3ExactSourceFacts(
        bodies=source.bodies, boxes=source.boxes, joints=source.joints,
        contacts=source.contacts, regions=source.regions,
        roots=tuple(roots[key] for key in sorted(roots)),
        joint_states=tuple(joints[key] for key in sorted(joints)),
        contact_states=tuple(contacts[key] for key in sorted(contacts)),
    )


def _changed_leaves(before: SceneStateEnvelope, after: SceneStateEnvelope,
                    allowed: tuple[StateVariableRef, ...]) -> tuple[StateVariableRef, ...]:
    if before.canonical_state_leaf_index != after.canonical_state_leaf_index:
        raise ValueError("M6 transition changed complete state leaf inventory")
    prior = generic._leaf_values(before, before.base_scene_payload)
    next_values = generic._leaf_values(after, after.base_scene_payload)
    leaves = before.canonical_state_leaf_index.leaves
    if set(prior) != set(next_values) or set(prior) != {generic._state_key(leaf) for leaf in leaves}:
        raise ValueError("M6 transition lacks complete state leaves")
    changed = _ordered(tuple(leaf for leaf in leaves if canonical_json_bytes(
        prior[generic._state_key(leaf)]
    ) != canonical_json_bytes(next_values[generic._state_key(leaf)])))
    if not set(changed) <= set(allowed):
        raise ValueError("M6 transition changed a frozen or unauthorized fact")
    return changed


def _delta(before: SceneStateEnvelope, after: SceneStateEnvelope,
           changed: tuple[StateVariableRef, ...]) -> StateDeltaManifest:
    old = generic._leaf_values(before, before.base_scene_payload)
    changed_keys = {generic._state_key(row) for row in changed}
    unchanged = tuple((leaf, old[generic._state_key(leaf)])
                      for leaf in before.canonical_state_leaf_index.leaves
                      if generic._state_key(leaf) not in changed_keys)
    return StateDeltaManifest.seal(
        authorized_primary_writes=changed, recomputed_derived_writes=(),
        unchanged_leaves_digest=canonical_sha256(unchanged,
                                                domain=generic._UNCHANGED_LEAF_DOMAIN),
        complete_before_leaf_index_sha256=before.canonical_state_leaf_index.state_leaf_index_sha256,
        complete_after_leaf_index_sha256=after.canonical_state_leaf_index.state_leaf_index_sha256,
    )


def _primary_leaves(view: RigidSE3StateView,
                    choice: RigidSE3StepChoice) -> tuple[StateVariableRef, ...]:
    addresses: set[str] = set()
    for member in choice.members:
        if member.kind == "SET_ROOT_SE3":
            owners = ((FAMILY[name], member.target_id, member.target_id)
                      for name in ("root-x", "root-y", "root-z", "root-rotation"))
        else:
            family = FAMILY["joint-state" if member.kind == "SET_JOINT" else "contact-mode"]
            owners = (owner for owner in view.address_by_owner
                      if owner[0] == family and owner[2] == member.target_id)
        addresses.update(view.address_by_owner[owner] for owner in owners)
    selected = _ordered(tuple(leaf for leaf in view.writable_value_leaves
                              if leaf.entity_or_fact_key in addresses))
    if len(selected) != len(addresses):
        raise ValueError("M6 atomic edit lacks its complete fine write footprint")
    return selected


def _grounded_obligations(problem: CounterfactualProblemIR) -> GroundedObligationSet:
    refs = tuple(row.definition_ref for row in problem.definition_bundle.definitions)
    def rows(items):
        return tuple(GroundedObligation(context=item, source_definition_refs=refs)
                     for item in items)
    return GroundedObligationSet.seal(
        before_preconditions=rows(problem.before_preconditions),
        after_goals=rows((problem.after_goal,)),
        preservation_invariants=rows(problem.preservation_invariants),
        observation_obligations=rows(problem.explicit_observation_obligations),
        grounding_entity_sets=(),
    )


def _replay_validated(context: RigidSE3RequestContext,
                      compilation: RigidSE3Compilation,
                      skeleton_id: str, choices: tuple[RigidSE3StepChoice, ...],
                      budget: kernel.ExactBudget,
                      *, permit_failed: bool = False,
                      ) -> RigidSE3ProgramTrace | RigidSE3FailedPrefix:
    request = context.request
    _checked_skeleton(context.domain, skeleton_id, choices, budget)
    source = exact_source_from_cells(source_cells_from_state(compilation.source_state))
    source_truth, _, _, source_poses, source_boxes = _state_truth(
        source, context.domain, request.semantic_problem, budget,
        prefix_index=0, reference_source=source, check_domain=False,
    )
    source_values = tuple(evaluate_rigid_se3_formula(
        row.before_formula, source, context.domain, source_poses, source_boxes, budget,
    ) for row in request.semantic_problem.preservation_invariants)
    if any(not row.satisfied for row in source_truth):
        raise ValueError("M6 source no longer satisfies its compiled invariants")
    current = source
    current_state = compilation.source_state
    steps: list[RigidSE3StepTrace] = []
    step_trace_bytes = 0
    first_failure: tuple[int, str, SceneStateEnvelope] | None = None
    read_leaves = current_state.canonical_state_leaf_index.leaves
    for index, choice in enumerate(choices):
        budget.charge("STEP")
        after_facts = _apply_atomic(current, choice)
        after = build_rigid_se3_state(current_state.base_scene_payload, after_facts).state
        primary = _primary_leaves(context.state_view, choice)
        changed = _changed_leaves(current_state, after, primary)
        truths, poses, contacts, after_poses, after_boxes = _state_truth(
            after_facts, context.domain, request.semantic_problem, budget,
            prefix_index=index + 1, reference_source=source,
            source_invariant_values=source_values,
        )
        if index + 1 == len(choices):
            goal_satisfied = evaluate_rigid_se3_formula(
                request.semantic_problem.after_goal.formula,
                after_facts, context.domain, after_poses, after_boxes, budget,
            )
            truths += (RigidSE3PrefixTruth(
                prefix_index=index + 1, obligation_id=d("after-goal"),
                satisfied=goal_satisfied,
            ),)
        step = RigidSE3StepTrace(
            step_index=index, choice=choice,
            before_state_sha256=current_state.scene_state_sha256,
            after_state_sha256=after.scene_state_sha256,
            after_state=after, step_delta=_delta(current_state, after, changed),
            read_leaves=read_leaves, primary_write_leaves=primary,
            changed_leaves=changed, prefix_truth=truths,
            world_poses=poses, contact_truth=contacts,
        )
        step_trace_bytes += m6_output_upper_bound(step)
        if step_trace_bytes > MAX_PROGRAM_TRACE_BYTES:
            raise M6IngressLimit("M6 ingress program trace output ceiling exceeded")
        steps.append(step)
        if first_failure is None:
            failed = next((row for row in truths if not row.satisfied), None)
            if failed is not None:
                first_failure = (index + 1, failed.obligation_id, after)
                if not permit_failed:
                    return RigidSE3FailedPrefix(
                        source_state_sha256=compilation.source_state_sha256,
                        choices=choices, completed_steps=tuple(steps),
                        failure_prefix_index=index + 1,
                        failure_obligation_id=failed.obligation_id,
                        failure_state=after,
                    )
        current, current_state = after_facts, after
    if first_failure is not None and not permit_failed:
        prefix_index, obligation_id, failed_state = first_failure
        return RigidSE3FailedPrefix(
            source_state_sha256=compilation.source_state_sha256,
            choices=choices, completed_steps=tuple(steps[:prefix_index]),
            failure_prefix_index=prefix_index,
            failure_obligation_id=obligation_id, failure_state=failed_state,
        )
    before = compilation.source_state
    changed = _changed_leaves(before, current_state, context.domain.authorized_primary_leaves)
    grounded = _grounded_obligations(request.semantic_problem)
    program = EditProgram.seal(
        program_id="program:rigid-se3/" + canonical_sha256(
            (skeleton_id, choices), domain=f"{PREFIX}/program-id",
        ),
        semantic_problem_sha256=request.semantic_problem_sha256,
        action_space_profile_sha256=context.registry_arguments["action_space_profile"].action_space_profile_sha256,
        steps=tuple(OperationInvocation(
            operator_ref=d("operator/apply-edit-set"),
            arguments=(OperationArgument(argument_name="argument:edit-set",
                                         value=encode_tree(choice)),),
        ) for choice in choices),
        before_state_sha256=before.scene_state_sha256,
        after_scene_state=current_state,
        after_scene_state_sha256=current_state.scene_state_sha256,
        state_delta_manifest=_delta(before, current_state, changed),
        grounded_obligation_set_sha256=grounded.grounded_obligation_set_sha256,
    )
    arguments = {key: value for key, value in context.registry_arguments.items()
                 if key not in {"request", "backend_descriptor_bundle", "solver_config",
                                "proof_policy", "resource_policy", "backend_routing_policy"}}
    context.registry.validate_edit_program(
        **arguments, intervention_authorization=request.semantic_problem.intervention_authorization,
        program=program, grounded_obligations=grounded,
        before_state=before, after_state=current_state,
    )
    return RigidSE3ProgramTrace.seal(
        skeleton_id=skeleton_id, choices=choices,
        source_state_sha256=before.scene_state_sha256,
        after_state_sha256=current_state.scene_state_sha256,
        steps=tuple(steps), program=program,
    )


def materialize_rigid_se3_program(
    request: CounterfactualSolveRequest, compilation: RigidSE3Compilation,
    skeleton_id: str, choices: tuple[RigidSE3StepChoice, ...],
) -> RigidSE3ProgramTrace | RigidSE3FailedPrefix:
    """Replay a complete authorized program with fresh source and domain binding."""
    fresh = compile_rigid_se3(request)
    if fresh != compilation:
        raise ValueError("stale or altered M6 compilation")
    if compilation.ledger.reason != "NONE":
        raise ValueError("incomplete M6 compilation cannot materialize a program")
    context = validate_rigid_se3_request(request)
    return _replay_validated(context, compilation, skeleton_id, choices,
                             kernel.ExactBudget(context.limits, "SOLVE"))


def prove_rigid_se3_swap(
    request: CounterfactualSolveRequest, compilation: RigidSE3Compilation,
    source_skeleton_id: str, source_choices: tuple[RigidSE3StepChoice, ...],
    swapped_skeleton_id: str, adjacent_index: int,
    *, kind: str = "SAFE_ENDPOINT_SWAP",
) -> RigidSE3SwapEvidence:
    """Replay both complete authorized orders before claiming adjacent exchange."""
    fresh = compile_rigid_se3(request)
    if fresh != compilation or compilation.ledger.reason != "NONE":
        raise ValueError("swap requires one complete fresh source compilation")
    context = validate_rigid_se3_request(request)
    return _swap_validated(
        context, compilation, source_skeleton_id, source_choices,
        swapped_skeleton_id, adjacent_index, kind=kind,
        budget=kernel.ExactBudget(context.limits, "SOLVE"),
    )


def _swap_validated(
    context: RigidSE3RequestContext, compilation: RigidSE3Compilation,
    source_skeleton_id: str, source_choices: tuple[RigidSE3StepChoice, ...],
    swapped_skeleton_id: str, adjacent_index: int,
    *, kind: str, budget: kernel.ExactBudget,
) -> RigidSE3SwapEvidence:
    """Pure two-order replay; the caller supplies one cumulative stage budget."""
    if kind not in {"STATE_COMMUTES", "SAFE_ENDPOINT_SWAP"}:
        raise ValueError("unregistered M6 swap claim")
    if adjacent_index < 0 or adjacent_index + 1 >= len(source_choices):
        raise ValueError("swap index must name two adjacent edit sets")
    swapped = list(source_choices)
    swapped[adjacent_index], swapped[adjacent_index + 1] = (
        swapped[adjacent_index + 1], swapped[adjacent_index]
    )
    swapped_choices = tuple(swapped)
    source_trace = _replay_validated(
        context, compilation, source_skeleton_id, source_choices,
        budget, permit_failed=True,
    )
    swapped_trace = _replay_validated(
        context, compilation, swapped_skeleton_id, swapped_choices,
        budget, permit_failed=True,
    )
    assert isinstance(source_trace, RigidSE3ProgramTrace)
    assert isinstance(swapped_trace, RigidSE3ProgramTrace)
    if kind == "SAFE_ENDPOINT_SWAP":
        for trace in (source_trace, swapped_trace):
            if any(not truth.satisfied for step in trace.steps for truth in step.prefix_truth):
                raise ValueError("SAFE_ENDPOINT_SWAP requires every prefix and final goal")
    prefix_hash = (source_trace.source_state_sha256 if adjacent_index == 0
                   else source_trace.steps[adjacent_index - 1].after_state_sha256)
    return RigidSE3SwapEvidence(
        kind=kind, source_trace=source_trace, swapped_trace=swapped_trace,
        source_trace_sha256=source_trace.trace_sha256,
        swapped_trace_sha256=swapped_trace.trace_sha256,
        adjacent_index=adjacent_index, common_prefix_state_sha256=prefix_hash,
        domain_sha256=context.domain.domain_sha256,
    )
