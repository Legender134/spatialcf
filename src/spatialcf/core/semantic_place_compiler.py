"""Source-bound static GeneralIR composition and exact placement compilation.

The only mutable scene scalars are the subject's existing XYZ and support ID.
No alias entity, derived cache, native source or ambient plugin is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

from spatialcf.core import registry as generic
from spatialcf.core._internal.kernels import semantic_place as kernel
from spatialcf.domain import semantic_place as place
from spatialcf.domain.base import FactAvailabilityV2, FactCompletenessV2, UncertaintyBudgetV2, Vec3
from spatialcf.domain.counterfactual import CounterfactualProblemIR, CounterfactualSolveRequest, EditProgram, ExtensionFact, ExtensionFactBundle, SceneStateEnvelope
from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope, CanonicalIdValue, DefinitionBundle, DigestValue, EnumSymbolValue,
    FiniteRealValue, FiniteSetValue, IntervalValue, NamedTypedValue, RecordValue, TypedValue,
    ValueFieldDefinition, ValueKind, ValueSchemaDefinition,
)
from spatialcf.domain.geometry import GeometryApproximationV2, GeometryRoleV2, UprightBox3DV2
from spatialcf.domain.operators import OperationArgument, OperationInvocation, OperatorDefinition, StateAddressPattern, StateDeltaManifest, StateLeafIndex, StateVariableDefinition, StateVariableRef, WriteAuthority
from spatialcf.domain.predicates import AfterGoal, AnyOfFormula, BeforePrecondition, GroundedObligation, GroundedObligationSet, NotFormula, PredicateAtom, PredicateDefinition, PreservationInvariant
from spatialcf.domain.profiles import (
    ActionSpaceProfile, BackendDescriptorBundle, BackendRoutingPolicy, CounterfactualSolverConfig,
    ImplementationOwnerBinding, ImplementationRegistrySnapshot, InterventionAuthorization,
    ObjectiveExpression, ObjectiveTerm, ProofPolicy, ResourceLimit, ResourcePolicy, SemanticsProfile, SolverBackendDescriptor,
)
from spatialcf.domain.scene import CanonicalScene, RegionBoundaryPolicy
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

d=place.definition
s=place.schema
c=place.capability
CAVITY_FAMILY=d("cavity-fact")
INVENTORY_FAMILY=d("cavity-inventory")
INVENTORY_KEY="fact-key:semantic-place-cavity-inventory"
BASE_SCHEMA="schema:spatialcf/canonical-scene/2.0"
BACKEND_REF="backend:spatialcf/semantic-place/exact-rectangles/1.0"
SEMANTICS_REF="spatialcf/semantic_place/semantics@1"
OWNERS={name:f"owner:spatialcf/semantic-place/{name}" for name in ("profile","compiler","kernel","backend","checker")}
# Static implementation version identities. Exact source successors are sealed
# separately by architecture/general-counterfactual-m5.toml.
BUILDS={name:canonical_sha256((place.PREFIX,name,"implementation:1"),domain=place.HASH_PREFIX+"/build") for name in OWNERS}


class UnsupportedPlacement(ValueError):
    """A valid source lies outside the exact first-profile fact subset."""


def ordered(values):
    return tuple(sorted(values,key=canonical_json_bytes))


def reseal(model, **updates):
    return type(model).seal(**(model.model_dump(mode="python",exclude={model.SELF_DIGEST_FIELD})|updates))


def _record(fields, schema_ref=None):
    fields=tuple(NamedTypedValue(name=name,value=value) for name,value in sorted(fields.items()))
    shape=tuple((f.name,f.value.value_schema_ref) for f in fields)
    return TypedValue(value_schema_ref=schema_ref or s("record/"+canonical_sha256(shape,domain=place.HASH_PREFIX+"/value-shape")),payload=RecordValue(fields=fields))


def _domain_value(domain):
    fields={k:place.encode_value(v) for k,v in domain.model_dump(mode="python").items() if k not in {"x","y","z","target_ids","support_surface_ids"}}
    for axis in ("x","y","z"):
        interval=getattr(domain,axis)
        fields[axis]=TypedValue(value_schema_ref=s("closed-real-interval"),payload=IntervalValue(endpoint_schema_ref=s("real"),
            lower=FiniteRealValue(value=interval.lower),upper=FiniteRealValue(value=interval.upper),lower_closed=True,upper_closed=True))
    for name in ("target_ids","support_surface_ids"):
        fields[name]=TypedValue(value_schema_ref=s("id-set"),payload=FiniteSetValue(element_schema_ref=s("id"),elements=tuple(place.encode_value(v) for v in getattr(domain,name))))
    return _record(fields)


def _decode_domain(value):
    if not isinstance(value.payload,RecordValue):
        raise ValueError("missing typed SemanticPlaceDomain")
    raw={}
    for field in value.payload.fields:
        payload=field.value.payload
        if field.name in ("x","y","z"):
            if not isinstance(payload,IntervalValue) or not payload.lower_closed or not payload.upper_closed:
                raise ValueError("placement domain intervals must be closed")
            raw[field.name]=dict(lower=payload.lower.value,upper=payload.upper.value,closure="CLOSED")
        elif field.name in ("target_ids","support_surface_ids"):
            if not isinstance(payload,FiniteSetValue):
                raise ValueError("placement target domain must be a finite set")
            raw[field.name]=tuple(v.payload.value for v in payload.elements)
        else:
            raw[field.name]=place.decode_value(field.value)
    domain=place.SemanticPlaceDomain.model_validate(raw,strict=True)
    if _domain_value(domain)!=value:
        raise ValueError("placement domain schema mismatch")
    return domain


def _schema_for_value(value, schemas):
    payload=value.payload
    args=dict(value_schema_ref=value.value_schema_ref,value_kind=payload.kind)
    if isinstance(payload,RecordValue):
        args['fields']=tuple(ValueFieldDefinition(field_name=f.name,value_schema_ref=f.value.value_schema_ref) for f in payload.fields)
        for field in payload.fields:
            _schema_for_value(field.value,schemas)
    elif isinstance(payload,EnumSymbolValue):
        args['enum_symbols']=(payload.symbol,)
    elif isinstance(payload,IntervalValue):
        args.update(endpoint_schema_ref=payload.endpoint_schema_ref,lower_closed=True,upper_closed=True)
        _schema_for_value(place.encode_value(0.0),schemas)
    elif isinstance(payload,FiniteSetValue):
        args.update(element_schema_ref=payload.element_schema_ref)
        _schema_for_value(place.encode_value("id:prototype"),schemas)
        for child in payload.elements:
            _schema_for_value(child,schemas)
    result=ValueSchemaDefinition.seal(**args)
    prior=schemas.get(result.value_schema_ref)
    if prior is not None and prior!=result:
        raise ValueError("same schema ID has different bytes")
    schemas[result.value_schema_ref]=result
    return result.value_schema_ref


def _leaf_rows(scene, facts):
    rows=[]
    for family,identifier,fact in generic._base_fact_rows(scene):
        for path,value,presence in ((('membership',),True,True),*((p,v,False) for p,v in generic._frozen_scalar_values(fact))):
            token=generic._frozen_leaf_token(family,identifier,path)
            rows.append((family,identifier,('field-path:presence-' if presence else 'field-path:')+token,value,token))
    for fact in facts:
        address=generic._extension_fact_address(*fact.ownership_key)
        for presence,value in ((True,True),(False,fact.value)):
            token=canonical_sha256((address,presence),domain=place.HASH_PREFIX+"/extension-leaf")
            rows.append((fact.fact_family_ref,address,('field-path:presence-' if presence else 'field-path:')+token,value,token))
    return tuple(rows)


def _state(scene,facts):
    rows=_leaf_rows(scene,facts)
    leaves=ordered(StateVariableRef(state_variable_schema_ref=s("leaf/"+token),state_schema_ref=s("state"),
        fact_family_ref=family,entity_or_fact_key=identifier,field_path_ref=path) for family,identifier,path,value,token in rows)
    state=SceneStateEnvelope.seal(base_scene_schema_ref=BASE_SCHEMA,base_scene_payload=scene,
        extension_fact_bundles=(ExtensionFactBundle.seal(facts=ordered(facts)),),
        closed_entity_index=ordered({identifier for family,identifier,fact in generic._base_fact_rows(scene)}|{f.subject_entity_id for f in facts}),
        canonical_state_leaf_index=StateLeafIndex.seal(leaves=leaves))
    return state,rows


def _authorized_leaves(state,subject_id):
    family="definition:spatialcf/counterfactual/base-scene/objects/3.0"
    paths=tuple(("pose","world_from_object","translation",axis) for axis in ("x","y","z"))+(("support_assignment","surface_id"),)
    wanted={"field-path:"+generic._frozen_leaf_token(family,subject_id,path) for path in paths}
    leaves=ordered(leaf for leaf in state.canonical_state_leaf_index.leaves if leaf.fact_family_ref==family and leaf.entity_or_fact_key==subject_id and leaf.field_path_ref in wanted)
    if len(leaves)!=4:
        raise ValueError("subject lacks exactly four existing scalar leaves")
    return leaves


def _facts(scene,cavities):
    objects=tuple(o.object_id for o in scene.objects.values)
    if not objects:
        raise ValueError("placement requires an existing subject")
    if len({fact.cavity_id for fact in cavities})!=len(cavities):
        raise ValueError("cavity IDs must be globally unique")
    facts=[]
    for cavity in cavities:
        if cavity.owner_object_id not in objects:
            raise ValueError("cavity owner must be an existing object")
        facts.append(ExtensionFact(fact_family_ref=CAVITY_FAMILY,subject_entity_id=cavity.owner_object_id,fact_key=cavity.cavity_id,value=place.encode_value(cavity)))
    inventory=place.SemanticPlaceCavityInventory.seal(cavity_addresses=tuple(sorted(generic._extension_fact_address(*f.ownership_key) for f in facts)))
    facts.append(ExtensionFact(fact_family_ref=INVENTORY_FAMILY,subject_entity_id=min(objects),fact_key=INVENTORY_KEY,value=place.encode_value(inventory)))
    return ordered(facts)


@dataclass(frozen=True)
class SemanticPlaceContext:
    request: CounterfactualSolveRequest
    domain: place.SemanticPlaceDomain
    limits: place.SemanticPlaceLimits
    registry: generic.StaticImplementationRegistry
    registry_arguments: dict


@dataclass(frozen=True)
class SemanticPlaceCompiledProblem:
    """Ephemeral protocol product retaining roots, never a trusted cache."""
    source_solve_request: CounterfactualSolveRequest
    compilation: place.SemanticPlaceCompilation | None
    failure_ledger: place.SemanticPlaceLedger | None

    @property
    def semantic_problem_sha256(self):
        return self.source_solve_request.semantic_problem_sha256

    @property
    def solve_request_sha256(self):
        return self.source_solve_request.solve_request_sha256

    @property
    def selected_backend_descriptor_sha256(self):
        return self.source_solve_request.backend_descriptor_bundle.backend_descriptors[0].backend_descriptor_sha256

    @property
    def grounded_obligation_set_sha256(self):
        return grounded_obligations(self.source_solve_request).grounded_obligation_set_sha256

    @property
    def compiled_artifact_sha256s(self):
        return (self.compilation.compilation_sha256,) if self.compilation is not None else ()


def grounded_obligations(request):
    problem=request.semantic_problem
    def rows(values):
        return tuple(GroundedObligation(context=value,source_definition_refs=ordered(generic._references(value,"definition:"))) for value in values)
    return GroundedObligationSet.seal(before_preconditions=rows(problem.before_preconditions),after_goals=rows((problem.after_goal,)),preservation_invariants=rows(problem.preservation_invariants))


def _compose(scene,facts,domain,limits,problem_id,backend_enabled=True):
    state,rows=_state(scene,facts)
    if domain.authorized_leaves!=_authorized_leaves(state,domain.subject_id):
        raise ValueError("placement authorization must name the four real scalar leaves")
    schemas={}
    for value in (place.encode_value(True),place.encode_value(0),place.encode_value(0.0),place.encode_value("id:prototype"),_domain_value(domain),place.encode_value(limits)):
        _schema_for_value(value,schemas)
    for reference in (BASE_SCHEMA,s("state"),s("proof-material"),s("subject-id"),s("target-id")):
        schemas[reference]=ValueSchemaDefinition.seal(value_schema_ref=reference,value_kind=ValueKind.CANONICAL_ID if reference in (s("subject-id"),s("target-id")) else ValueKind.RECORD)
    schemas.update({row.value_schema_ref:row for row in place.proof_value_schemas()})
    states=[]; metadata={}; roles={}
    for family,identifier,path,value,token in rows:
        value_schema=_schema_for_value(value if isinstance(value,TypedValue) else place.encode_value(value if value is not None else "id:null"),schemas)
        leaf_schema=s("leaf/"+token)
        schemas[leaf_schema]=ValueSchemaDefinition.seal(value_schema_ref=leaf_schema,value_kind=ValueKind.RECORD)
        ref=d("state/"+token)
        states.append(StateVariableDefinition.seal(state_variable_ref=ref,state_variable_schema_ref=leaf_schema,value_schema_ref=value_schema,
            frame_ref=d("world"),unit_ref=d("metre"),topology_ref=d("closed"),write_authority=WriteAuthority.PRIMARY_WRITABLE))
        roles[ref]="state-variable"
        metadata[ref]={"field:state-variable-schema":leaf_schema,"field:state-leaf-schema":s("state"),"field:state-fact-family":family,
            "field:state-entity-key":identifier,"field:state-field-path":path,"field:state-value-schema":value_schema,
            "field:state-frame":d("world"),"field:state-unit":d("metre"),"field:state-topology":d("closed")}
    states=ordered(states)
    by_schema={v.state_variable_schema_ref:v for v in states}
    def atom(name,target=None):
        operands=(TypedValue(value_schema_ref=s("subject-id"),payload=CanonicalIdValue(value=domain.subject_id)),)
        if target is not None:
            operands+= (TypedValue(value_schema_ref=s("target-id"),payload=CanonicalIdValue(value=target)),)
        return PredicateAtom(predicate_ref=d("predicate/"+name),operands=operands)
    goal=AnyOfFormula(formulas=ordered(atom("on" if domain.operation=="PLACE_ON" else "in",target) for target in domain.target_ids)).ground()
    before=(BeforePrecondition(formula=NotFormula(formula=goal)),)
    invariants=ordered(PreservationInvariant(before_formula=True,after_formula=atom(name),transition_comparator_ref=d("invariant/"+name)) for name in ("contact","collision","endpoint-domain","unchanged"))
    predicates=[]
    for name in ("on","in","contact","collision","endpoint-domain","unchanged"):
        ref=d("predicate/"+name); roles[ref]="predicate"
        predicates.append(PredicateDefinition.seal(predicate_ref=ref,operand_schema_refs=(s("subject-id"),s("target-id")) if name in ("on","in") else (s("subject-id"),),
            state_context_policy_ref=d("context"),frame_requirement_ref=d("world"),measurement_definition_ref=d("measure/"+name),measurement_unit_ref=d("metre"),
            comparator_definition_ref=d("comparator/"+name),boundary_policy_ref=d("closed"),tolerance_policy_ref=d("zero-tolerance"),uncertainty_policy_ref=d("exact"),
            observation_prerequisite_template_refs=(),evaluator_capability_ref=c("evaluate/"+name),verifier_capability_ref=c("verify/"+name),admissible_claim_definition_refs=(d("certified"),)))
    predicates=ordered(predicates)
    metadata[d("measure/endpoint-domain")]={"field:domain-reference":d("domain")}
    metadata[d("delta-policy")]={"field:domain-reference":d("domain")}
    metadata[d("domain")]={"field:domain":_domain_value(domain)}
    operator=OperatorDefinition.seal(operator_ref=d("operator/"+domain.operation),parameter_schema_refs=(s("subject-id"),s("target-id"),s("real"),s("real"),s("real")),
        read_footprint=ordered(StateAddressPattern(state_variable_definition_ref=v.state_variable_ref) for v in states),
        primary_write_footprint=ordered(StateAddressPattern(state_variable_definition_ref=by_schema[leaf.state_variable_schema_ref].state_variable_ref) for leaf in domain.authorized_leaves),
        derived_write_rule_refs=(),required_preconditions=before,transition_semantics_ref=d("transition/"+domain.operation),generated_obligations=(),
        composability_policy_ref=d("one-step"),endpoint_or_path_semantics_ref=d("endpoint-only"),compiler_capability_ref=c("compile"),verifier_capability_ref=c("check"))
    roles[operator.operator_ref]="operator"
    auth=InterventionAuthorization.seal(editable_entity_ids=(domain.subject_id,),allowed_operator_refs=(operator.operator_ref,),authorized_primary_write_set=domain.authorized_leaves,
        variable_bounds=(),maximum_program_steps=1,maximum_edited_entities=1,required_derived_rule_refs=(),complete_state_delta_policy_ref=d("delta-policy"))
    objective=ObjectiveExpression.seal(aggregation_definition_ref=d("sum"),terms=(ObjectiveTerm(term_id="term:squared-displacement",objective_definition_ref=d("squared-displacement"),
        input_selector_definition_ref=d("subject-world-xyz"),unit_ref=d("square-metre"),normalization_definition_ref=d("coefficient-one")),),deterministic_tie_break_definition_ref=d("target-then-exact-xyz"))
    metadata[d("squared-displacement")]={"field:objective-input-selector":d("subject-world-xyz"),"field:objective-unit":d("square-metre"),"field:objective-normalization":d("coefficient-one")}
    semantics=SemanticsProfile.seal(semantics_profile_ref=SEMANTICS_REF,accepted_scene_and_fact_schema_refs=ordered({BASE_SCHEMA,*[f.value.value_schema_ref for f in facts]}),
        predicate_definition_refs=ordered(v.predicate_ref for v in predicates),transition_semantics_refs=(operator.transition_semantics_ref,),objective_definition_refs=(d("squared-displacement"),),
        numeric_semantics_ref=d("exact-dyadic"),completeness_policy_ref=d("closed-inventory"),uncertainty_policy_ref=d("exact"),derived_fact_rule_refs=(),observation_obligation_policy_ref=d("no-observations"))
    action=ActionSpaceProfile.seal(action_space_profile_ref=place.PROFILE_REF,accepted_scene_schema_refs=(BASE_SCHEMA,),state_variable_definition_refs=ordered(v.state_variable_ref for v in states),
        allowed_operator_refs=(operator.operator_ref,),mandatory_invariant_template_refs=ordered(i.transition_comparator_ref for i in invariants),
        predicate_capability_refs=ordered(p.evaluator_capability_ref for p in predicates),objective_capability_refs=(c("objective"),),numeric_semantics_ref=d("exact-dyadic"),
        allowed_claim_definition_refs=ordered((d("certified"),d("unsat"))),backend_capability_requirements=(c("solve"),),adapter_capability_requirements=(),publication_proof_policy_ref=d("proof-policy"))
    config=CounterfactualSolverConfig.seal(solver_config_ref=d("solver-config"),compilation_policy_ref=d("compile-policy"),proposal_policy_ref=d("proposal-policy"),objective_bound_policy_ref=d("outward-bounds"),determinism_policy_ref=d("deterministic"))
    proof=ProofPolicy.seal(proof_policy_ref=d("proof-policy"),accepted_claim_definition_refs=action.allowed_claim_definition_refs,required_checker_capability_refs=(c("check"),),publication_minimum_claim_ref=d("certified"),permit_noncertified_terminal_records=False)
    resource=ResourcePolicy.seal(resource_policy_ref=d("resource-policy"),limits=ordered(ResourceLimit(definition_ref=d("resource/"+name),finite_limit=float(value)) for name,value in limits.model_dump().items()),exhaustion_claim_ref=d("resource-exhausted"),shared_ledger_policy_ref=d("separate-stage-ledgers"))
    routing=BackendRoutingPolicy.seal(routing_policy_ref=d("routing"),capability_filter_definition_ref=d("filter"),deterministic_order_definition_ref=d("order"),portfolio_composition_definition_ref=d("single-backend"),stop_condition_definition_ref=d("stop"),resource_partition_definition_ref=d("partition"))
    metadata[d("proof-material")]={"field:proof-payload-schema":s("proof-material"),"field:proof-checker-capability":c("check")}; roles[d("proof-material")]="proof-material"
    for name,role in (("certified","claim-certified-solution"),("unsat","claim-proven-unsat")):
        roles[d(name)]=role; metadata[d(name)]={"field:claim-proof-material-definition":d("proof-material"),"field:claim-checker-capability":c("check")}
    for name in ("complete-domain","sound-complete-domain"):
        roles[d(name)]=name;metadata[d(name)]={"field:complete-domain-claim":d("unsat")}
    roles[d("resource-policy")]="resource-policy";metadata[d("resource-policy")]={"field:resource-accounting-claim":d("accounting"),"field:limits":place.encode_value(limits)}
    roles[d("accounting")]="resource-accounting"
    for name in ("resource-exhausted","numeric-gap","unsupported","unknown"):
        roles[d(name)]="claim-unknown"
    metadata[d("proposal-policy")]={"field:unknown-claim":d("unknown"),"field:numeric-reason":d("numeric-gap"),"field:unsupported-reason":d("unsupported")}
    roles[d("routing")]="routing-policy";metadata[d("routing")]={"field:routing-match-claim":d("match"),"field:routing-mismatch-claim":d("mismatch"),"field:selection-disposition":d("selection-disposition"),"field:selection-reason":d("selection-reason")}
    roles.update({d("match"):"routing-match",d("mismatch"):"routing-mismatch",d("selection-disposition"):"routing-selection-disposition",d("selection-reason"):"routing-selection-reason"})
    descriptor=SolverBackendDescriptor.seal(backend_ref=BACKEND_REF,implementation_build_sha256=BUILDS['backend'],supported_profile_hashes=ordered((semantics.semantics_profile_sha256,action.action_space_profile_sha256)),
        supported_predicate_capabilities=ordered(v for p in predicates for v in (p.evaluator_capability_ref,p.verifier_capability_ref)),supported_operator_capabilities=ordered((operator.compiler_capability_ref,operator.verifier_capability_ref)) if backend_enabled else (),
        supported_objective_capabilities=action.objective_capability_refs,supported_numeric_semantics=(d("exact-dyadic"),),emitted_proof_material_definition_refs=(d("proof-material"),),compatible_checker_capability_refs=(c("check"),),resource_definition_refs=ordered(limit.definition_ref for limit in resource.limits))
    backend_bundle=BackendDescriptorBundle.seal(backend_descriptors=(descriptor,),unavailable_optional_backends=())
    sem_roots=(state,auth,before,AfterGoal(formula=goal),invariants,objective,semantics,action,*predicates,*states,operator)
    sem_refs=set().union(*(generic._references(root,"definition:") for root in sem_roots))-{d("proof-policy")}
    sem_refs.update((d("proof-material"),d("complete-domain"),d("sound-complete-domain")))
    solve_roots=(config,proof,resource,routing,backend_bundle)
    solve_refs=set().union(*(generic._references(root,"definition:") for root in solve_roots))-sem_refs
    solve_refs.update((d("accounting"),d("match"),d("mismatch"),d("selection-disposition"),d("selection-reason")))
    for references in (sem_refs,solve_refs):
        while True:
            prior=len(references)
            for ref in tuple(references):
                references.update(generic._references(metadata.get(ref,{}),"definition:")- (sem_refs if references is solve_refs else set()))
            if len(references)==prior:break
    # One fixed optional-field envelope schema makes binding metadata part of
    # the reachable schema closure without allowing records to authorize themselves.
    field_schemas={"field:definition-kind":s("definition-kind"),"field:definition-reference":s("id"),"field:bound-record-ref":s("id"),"field:bound-record-sha256":s("digest")}
    for fields in metadata.values():
        for name,value in fields.items():
            field_schemas[name]=value.value_schema_ref if isinstance(value,TypedValue) else s("id")
    kinds=ordered({"definition-kind:generic","definition-kind:record-binding",*("definition-kind:"+v for v in roles.values())})
    schemas[s("definition-kind")]=ValueSchemaDefinition.seal(value_schema_ref=s("definition-kind"),value_kind=ValueKind.ENUM_SYMBOL,enum_symbols=kinds)
    schemas[s("digest")]=ValueSchemaDefinition.seal(value_schema_ref=s("digest"),value_kind=ValueKind.DIGEST)
    schemas[s("definition-record")]=ValueSchemaDefinition.seal(value_schema_ref=s("definition-record"),value_kind=ValueKind.RECORD,
        fields=tuple(ValueFieldDefinition(field_name=name,value_schema_ref=ref,required=name in ("field:definition-kind","field:definition-reference")) for name,ref in sorted(field_schemas.items())))
    def envelope(ref,anchor,kind="generic",fields=None):
        values={"field:definition-kind":TypedValue(value_schema_ref=s("definition-kind"),payload=EnumSymbolValue(symbol="definition-kind:"+kind)),"field:definition-reference":place.encode_value(ref)}
        values.update({k:v if isinstance(v,TypedValue) else place.encode_value(v) for k,v in (fields or {}).items()})
        return CanonicalDefinitionEnvelope.seal(definition_ref=ref,definition_kind_ref=anchor,payload_schema_ref=s("definition-record"),payload=_record(values,s("definition-record")))
    sem_anchor=d("semantic-bootstrap");solve_anchor=d("solve-bootstrap")
    sem_refs.add(sem_anchor);solve_refs.add(solve_anchor)
    semantic_defs=[envelope(ref,sem_anchor,roles.get(ref,"generic"),metadata.get(ref)) for ref in sorted(sem_refs)]
    bindings=[*( (v.value_schema_ref,v) for v in schemas.values()),*((v.predicate_ref,v) for v in predicates),*((v.state_variable_ref,v) for v in states),(operator.operator_ref,operator),(semantics.semantics_profile_ref,semantics),(action.action_space_profile_ref,action)]
    for ref,record in bindings:
        digest=getattr(record,record.SELF_DIGEST_FIELD)
        bind_ref=d("binding/"+canonical_sha256((ref,digest),domain=place.HASH_PREFIX+"/record-binding"))
        semantic_defs.append(envelope(bind_ref,sem_anchor,"record-binding",{"field:bound-record-ref":ref,"field:bound-record-sha256":TypedValue(value_schema_ref=s("digest"),payload=DigestValue(value=digest))}))
    semantic_bundle=DefinitionBundle.seal(definitions=tuple(sorted(semantic_defs,key=lambda v:canonical_json_bytes(v.definition_ref))))
    solve_bundle=DefinitionBundle.seal(definitions=tuple(envelope(ref,solve_anchor,roles.get(ref,"generic"),metadata.get(ref)) for ref in sorted(solve_refs)))
    caps={"profile":(c("profile"),),"compiler":(c("compile"),),"kernel":ordered((c("objective"),*[v for p in predicates for v in (p.evaluator_capability_ref,p.verifier_capability_ref)])),"backend":(c("solve"),),"checker":(c("check"),)}
    registry=generic.StaticImplementationRegistry(tuple(generic.StaticOwner(OWNERS[name],BUILDS[name],caps[name],"spatialcf.semantic_place."+name) for name in sorted(OWNERS)))
    snapshot=ImplementationRegistrySnapshot.seal(definition_and_capability_owner_bindings=ordered(ImplementationOwnerBinding(definition_or_capability_ref=cap,implementation_owner_ref=OWNERS[name]) for name,values in caps.items() for cap in values),
        implementation_build_hashes=ordered((OWNERS[name],BUILDS[name]) for name in OWNERS),dependency_lock_sha256=canonical_sha256(("python:3.11","canonical-v2","fraction"),domain=place.HASH_PREFIX+"/dependency-contract"))
    problem=CounterfactualProblemIR.seal(problem_id=problem_id,scene_state=state,definition_bundle=semantic_bundle,semantics_profile_ref=SEMANTICS_REF,action_space_profile_ref=place.PROFILE_REF,
        intervention_authorization=auth,before_preconditions=before,after_goal=AfterGoal(formula=goal),preservation_invariants=invariants,explicit_observation_obligations=(),objective_expression=objective,numeric_semantics_ref=d("exact-dyadic"))
    request=CounterfactualSolveRequest.seal(semantic_problem=problem,semantic_problem_sha256=problem.semantic_problem_sha256,solve_policy_definition_bundle=solve_bundle,
        implementation_registry_snapshot=snapshot,backend_descriptor_bundle=backend_bundle,solver_config=config,proof_policy=proof,resource_policy=resource,backend_routing_policy=routing)
    arguments=dict(semantic_definition_bundle=semantic_bundle,solve_policy_definition_bundle=solve_bundle,value_schema_definitions=ordered(schemas.values()),predicate_definitions=predicates,state_variable_definitions=states,
        derived_fact_rule_definitions=(),operator_definitions=(operator,),implementation_registry_snapshot=snapshot,problem=problem,semantics_profile=semantics,action_space_profile=action,scene_state=state,
        request=request,backend_descriptor_bundle=backend_bundle,solver_config=config,proof_policy=proof,resource_policy=resource,backend_routing_policy=routing)
    return SemanticPlaceContext(request,domain,limits,registry,arguments)


def build_semantic_place_request(*, scene: CanonicalScene, subject_id: str, operation: str,
        target_ids: tuple[str,...], x: place.SemanticPlaceInterval, y: place.SemanticPlaceInterval,
        z: place.SemanticPlaceInterval, cavities: tuple[place.SemanticPlaceCavityFact,...]=(),
        support_margin_m=0.0,lateral_margin_m=0.0,top_margin_m=0.0,
        limits: place.SemanticPlaceLimits|None=None, backend_enabled=True,
        problem_id="problem:semantic-place") -> CounterfactualSolveRequest:
    """Build the existing GeneralIR root using explicit immutable source facts."""
    if not subject_id.startswith('entity:') or subject_id=='entity:' or '*' in subject_id:
        raise UnsupportedPlacement("existing editable subject ID must use entity: without an alias")
    scene=CanonicalScene.model_validate(scene.model_dump(mode="python"),strict=True)
    place.canonical_numbers(scene)
    if subject_id not in {o.object_id for o in scene.objects.values}:
        raise ValueError("unknown editable subject")
    facts=_facts(scene,cavities)
    state,_=_state(scene,facts)
    if operation=='PLACE_ON':
        surfaces=target_ids
    elif operation=='PLACE_IN':
        by_id={f.cavity_id:f for f in cavities}
        if any(t not in by_id for t in target_ids):
            raise ValueError("unknown cavity target")
        surfaces=tuple(sorted({by_id[t].bottom_support_surface_id for t in target_ids}))
    else:
        raise ValueError("only PLACE_ON or PLACE_IN is supported")
    domain=place.SemanticPlaceDomain.seal(subject_id=subject_id,operation=operation,x=x,y=y,z=z,target_ids=target_ids,
        support_surface_ids=surfaces,authorized_leaves=_authorized_leaves(state,subject_id),support_margin_m=support_margin_m,lateral_margin_m=lateral_margin_m,top_margin_m=top_margin_m)
    context=_compose(scene,facts,domain,limits or place.SemanticPlaceLimits(),problem_id,backend_enabled)
    context.registry.validate_solve_request(**context.registry_arguments)
    return context.request


def validate_semantic_place_request(request: CounterfactualSolveRequest) -> SemanticPlaceContext:
    """Reconstruct the trusted static dossier; submitted metadata cannot own itself."""
    if type(request) is not CounterfactualSolveRequest:
        raise TypeError("expected exact CounterfactualSolveRequest")
    CounterfactualSolveRequest.model_validate(request.model_dump(mode="python"),strict=True)
    definitions={v.definition_ref:v for v in request.semantic_problem.definition_bundle.definitions}
    solve_defs={v.definition_ref:v for v in request.solve_policy_definition_bundle.definitions}
    try:
        domain_fields={f.name:f.value for f in definitions[d("domain")].payload.payload.fields}
        domain=_decode_domain(domain_fields['field:domain'])
        resource_fields={f.name:f.value for f in solve_defs[d("resource-policy")].payload.payload.fields}
        limits=place.SemanticPlaceLimits.model_validate(place.decode_value(resource_fields['field:limits']),strict=True)
    except (KeyError,AttributeError) as error:
        raise ValueError("missing mandatory placement domain or resource definition") from error
    if any(v>2**53 for v in limits.model_dump().values()):
        raise ValueError("resource counters require exact finite transport")
    state=request.semantic_problem.scene_state
    facts=tuple(f for bundle in state.extension_fact_bundles for f in bundle.facts)
    descriptors=request.backend_descriptor_bundle.backend_descriptors
    if len(descriptors)!=1:
        raise ValueError("placement requires the frozen single backend descriptor")
    expected=_compose(state.base_scene_payload,facts,domain,limits,request.semantic_problem.problem_id,bool(descriptors[0].supported_operator_capabilities))
    if expected.request!=request:
        raise ValueError("placement request differs from its trusted source-bound static composition")
    expected.registry.validate_solve_request(**expected.registry_arguments)
    return expected


def _identity(transform):
    rotation=transform.rotation
    if (rotation.x,rotation.y,rotation.z,rotation.w)!=(0.0,0.0,0.0,1.0):
        raise UnsupportedPlacement("required rotation is not exact identity")
    return tuple(F(v) for v in (transform.translation.x,transform.translation.y,transform.translation.z))


def _exact(facts,label):
    if facts.availability is not FactAvailabilityV2.KNOWN or facts.completeness is not FactCompletenessV2.EXACT or facts.uncertainty!=UncertaintyBudgetV2() or facts.values is None:
        raise UnsupportedPlacement(label+" facts must be KNOWN EXACT with zero uncertainty")
    return facts.values


def _geometry(context,budget):
    state=context.request.semantic_problem.scene_state
    scene=state.base_scene_payload
    place.canonical_numbers(scene)
    objects={o.object_id:o for o in _exact(scene.objects,"object")}
    geometries={g.geometry_id:g for g in _exact(scene.geometry_instances,"geometry")}
    bodies={b.body_id:b for b in _exact(scene.collision_bodies,"collision body")}
    source_surfaces={p.surface_id:p for p in _exact(scene.support_surfaces,"support")}
    subject=objects.get(context.domain.subject_id)
    if subject is None or not subject.movable:
        raise ValueError("subject must be an existing movable object")
    local={};world={};offsets={}
    for owner in objects.values():
        budget.current="object:"+owner.object_id
        offsets[owner.object_id]=_identity(owner.pose.world_from_object)
        budget.numbers(*offsets[owner.object_id])
    for geometry in geometries.values():
        if geometry.role is not GeometryRoleV2.COLLISION and not (geometry.owner_object_id==subject.object_id and geometry.role is GeometryRoleV2.SUPPORT):
            continue
        budget.current="geometry:"+geometry.geometry_id
        if geometry.approximation is not GeometryApproximationV2.EXACT or geometry.uncertainty!=UncertaintyBudgetV2() or not isinstance(geometry.shape,UprightBox3DV2):
            raise UnsupportedPlacement("required geometry is not an exact upright box")
        center=_identity(geometry.anchor_from_geometry)
        size=tuple(F(v) for v in (geometry.shape.size_m.x,geometry.shape.size_m.y,geometry.shape.size_m.z))
        lower=tuple(p-a/2 for p,a in zip(center,size,strict=True));upper=tuple(p+a/2 for p,a in zip(center,size,strict=True))
        budget.numbers(*center,*size,*lower,*upper)
        local[geometry.geometry_id]=kernel.box(lower,upper)
        world[geometry.geometry_id]=kernel.translate(local[geometry.geometry_id],offsets.get(geometry.owner_object_id,(F(0),)*3),budget)
    subject_bodies=tuple(b for b in bodies.values() if b.owner_object_id==subject.object_id)
    if len(subject_bodies)!=1 or len(subject_bodies[0].geometry_instance_ids)!=1:
        raise UnsupportedPlacement("subject requires exactly one collision body and box")
    subject_geometry=subject_bodies[0].geometry_instance_ids[0]
    subject_local=local[subject_geometry];sl,su=kernel.bounds(subject_local)
    support_geometries=tuple(g for g in geometries.values() if g.owner_object_id==subject.object_id and g.role is GeometryRoleV2.SUPPORT)
    if not support_geometries:
        raise UnsupportedPlacement("subject requires explicit SUPPORT geometry")
    for g in support_geometries:
        l,u=kernel.bounds(local[g.geometry_id])
        if (l[0],l[1],l[2],u[0],u[1])!=(sl[0],sl[1],sl[2],su[0],su[1]):
            raise UnsupportedPlacement("SUPPORT footprint and bottom must equal collision geometry")
    surfaces={}
    for surface in source_surfaces.values():
        budget.current="surface:"+surface.surface_id
        if surface.boundary_policy is not RegionBoundaryPolicy.CLOSED or surface.region_approximation is not GeometryApproximationV2.EXACT or surface.geometry_uncertainty!=UncertaintyBudgetV2():
            raise UnsupportedPlacement("support must be closed and exact")
        if (surface.normal_in_anchor.x,surface.normal_in_anchor.y,surface.normal_in_anchor.z)!=(0.0,0.0,1.0):
            raise UnsupportedPlacement("support normal must be exactly upward")
        frame=_identity(surface.anchor_from_surface)
        from spatialcf.core._internal.kernels.rect import ExactAxisAlignedRectV2, RectCoordinateSpaceV2, RectKernelErrorV2
        try:
            rect=ExactAxisAlignedRectV2.from_planar_region(surface.region_uv,coordinate_space=RectCoordinateSpaceV2.WORLD_XY_M)
        except RectKernelErrorV2 as error:
            raise UnsupportedPlacement("support region must be one hole-free rectangle") from error
        budget.numbers(*frame,rect.min_x_m,rect.min_y_m,rect.max_x_m,rect.max_y_m)
        shift=tuple(a+b for a,b in zip(frame,offsets.get(surface.owner_object_id,(F(0),)*3),strict=True))
        rectangle=(rect.min_x_m+shift[0],rect.min_y_m+shift[1],rect.max_x_m+shift[0],rect.max_y_m+shift[1])
        budget.numbers(*shift,*rectangle)
        body=bodies[surface.supporting_body_id]
        if body.owner_object_id!=surface.owner_object_id:
            raise ValueError("support body ownership mismatch")
        tops=[]
        for gid in body.geometry_instance_ids:
            l,u=kernel.bounds(world[gid])
            if u[2]==shift[2]:tops.append((l[0],l[1],u[0],u[1]))
        if not kernel.covered_rectangle(rectangle,tuple(tops),budget):
            raise ValueError("support top faces do not cover declared region")
        surfaces[surface.surface_id]=(rectangle,shift[2],surface.owner_object_id)
    budget.current="fact:cavity-inventory"
    facts=tuple(f for bundle in state.extension_fact_bundles for f in bundle.facts)
    inventories=tuple(f for f in facts if f.fact_family_ref==INVENTORY_FAMILY)
    cavity_rows=tuple(f for f in facts if f.fact_family_ref==CAVITY_FAMILY)
    if not inventories:
        raise UnsupportedPlacement("missing closed cavity inventory")
    if len(inventories)!=1 or len(facts)!=1+len(cavity_rows):
        raise ValueError("multiple inventories or unknown extension facts")
    inventory_fact=inventories[0]
    if (inventory_fact.subject_entity_id,inventory_fact.fact_key)!=(min(objects),INVENTORY_KEY):
        raise ValueError("inventory must be on the lexicographically first existing object")
    inventory=place.SemanticPlaceCavityInventory.model_validate(place.decode_value(inventory_fact.value),strict=True)
    addresses=tuple(sorted(generic._extension_fact_address(*f.ownership_key) for f in cavity_rows))
    if addresses!=inventory.cavity_addresses:
        raise ValueError("cavity inventory omits or adds full fact addresses")
    cavities={};cavity_boxes={}
    for fact in cavity_rows:
        cavity=place.SemanticPlaceCavityFact.model_validate(place.decode_value(fact.value),strict=True)
        budget.current="cavity:"+cavity.cavity_id
        if cavity.cavity_id in cavities or (cavity.owner_object_id,cavity.cavity_id)!=(fact.subject_entity_id,fact.fact_key) or cavity.owner_object_id not in objects:
            raise ValueError("duplicate cavity ID or wrong source ownership")
        if cavity.owner_object_id==subject.object_id:
            raise ValueError("edited subject cannot own a cavity")
        if cavity.bottom_support_surface_id not in surfaces:
            raise ValueError("dangling cavity bottom support")
        shell=tuple(sorted(b.body_id for b in bodies.values() if b.owner_object_id==cavity.owner_object_id))
        if shell!=cavity.shell_body_ids:
            raise ValueError("cavity shell roster must equal all owner collision bodies")
        center=_identity(cavity.anchor_from_cavity)
        size=tuple(F(v) for v in (cavity.interior_size_m.x,cavity.interior_size_m.y,cavity.interior_size_m.z))
        budget.numbers(*center,*size)
        center=tuple(a+b for a,b in zip(center,offsets[cavity.owner_object_id],strict=True))
        lo=tuple(p-a/2 for p,a in zip(center,size,strict=True));hi=tuple(p+a/2 for p,a in zip(center,size,strict=True))
        budget.numbers(*center,*size,*lo,*hi)
        volume=kernel.box(lo,hi)
        rectangle,height,owner=surfaces[cavity.bottom_support_surface_id]
        if owner!=cavity.owner_object_id or lo[2]!=height or not (rectangle[0]<=lo[0]<=hi[0]<=rectangle[2] and rectangle[1]<=lo[1]<=hi[1]<=rectangle[3]):
            raise ValueError("cavity bottom must match and be covered by owner support")
        for bid in shell:
            for gid in bodies[bid].geometry_instance_ids:
                if kernel.interiors_overlap(volume,world[gid],budget):
                    raise ValueError("cavity open interior contradicts shell solid")
        cavities[cavity.cavity_id]=cavity;cavity_boxes[cavity.cavity_id]=volume
    for other in objects.values():
        support=source_surfaces.get(other.support_assignment.surface_id)
        if other.object_id!=subject.object_id and support is not None and support.owner_object_id==subject.object_id:
            raise ValueError("edited subject owns a support used by another object")
    obstacles=tuple(sorted((g.geometry_id,world[g.geometry_id]) for g in geometries.values() if g.role is GeometryRoleV2.COLLISION and g.owner_object_id!=subject.object_id))
    return subject,subject_local,world[subject_geometry],surfaces,cavities,cavity_boxes,obstacles


def _on(world_box,surface,assignment,surface_id):
    low,high=kernel.bounds(world_box);rectangle,height,_=surface
    return assignment==surface_id and low[2]==height and rectangle[0]<=low[0]<=high[0]<=rectangle[2] and rectangle[1]<=low[1]<=high[1]<=rectangle[3]


def compile_semantic_place(request: CounterfactualSolveRequest, *, audit=None) -> place.SemanticPlaceCompilation:
    context=validate_semantic_place_request(request)
    return _compile_validated(context,kernel.Budget(context.limits,"COMPILE",audit=audit))


def _compile_validated(context,budget):
    request=context.request
    try:
        subject,local,before_world,surfaces,cavities,cavity_boxes,obstacles=_geometry(context,budget)
        if subject.support_assignment.availability is not FactAvailabilityV2.KNOWN or subject.support_assignment.surface_id not in surfaces:
            raise UnsupportedPlacement("subject source support assignment must be known")
        source_support=subject.support_assignment.surface_id
        if not _on(before_world,surfaces[source_support],source_support,source_support):
            raise ValueError("source support assignment is geometrically invalid")
        if any(kernel.interiors_overlap(before_world,obstacle,budget) for _,obstacle in obstacles):
            raise ValueError("source subject collides with a fixed body")
        domain=context.domain
        targets=[];surface_ids=set()
        for i,tid in enumerate(domain.target_ids):
            budget.current="target:"+tid
            if i>=context.limits.max_targets:
                raise kernel.PlacementLimit("RESOURCE_LIMIT",budget.current)
            if domain.operation=='PLACE_IN':
                if tid not in cavities:raise ValueError("dangling cavity target")
                sid=cavities[tid].bottom_support_surface_id
                if kernel.contains(cavity_boxes[tid],before_world,budget):raise ValueError("before goal already true")
            else:
                sid=tid
            if sid not in surfaces:raise ValueError("dangling support target")
            surface_ids.add(sid)
            rectangle,height,owner=surfaces[sid]
            if owner==subject.object_id:raise ValueError("self-owned target")
            # Existing support graph is acyclic; retain an explicit ancestry guard.
            scene=request.semantic_problem.scene_state.base_scene_payload
            objects={o.object_id:o for o in scene.objects.values}
            seen=set()
            while owner is not None:
                if owner==subject.object_id or owner in seen:raise ValueError("target support ancestry depends on edited subject")
                seen.add(owner)
                parent=objects[owner].support_assignment.surface_id
                owner=surfaces[parent][2] if parent in surfaces else None
            if domain.operation=='PLACE_ON' and _on(before_world,surfaces[sid],source_support,sid):raise ValueError("before goal already true")
            targets.append(kernel.compile_target(target_id=tid,support_surface_id=sid,subject=local,support=rectangle,height=height,
                authorized_xy=(F(domain.x.lower),F(domain.y.lower),F(domain.x.upper),F(domain.y.upper)),authorized_z=(F(domain.z.lower),F(domain.z.upper)),obstacles=obstacles,
                cavity=cavity_boxes[tid] if domain.operation=='PLACE_IN' else None,margins=(F(domain.support_margin_m),F(domain.lateral_margin_m),F(domain.top_margin_m)),budget=budget))
            budget.completed.append("target:"+tid)
        if tuple(sorted(surface_ids))!=domain.support_surface_ids:raise ValueError("typed support roster differs from target domain")
        return place.SemanticPlaceCompilation.seal(semantic_problem_sha256=request.semantic_problem_sha256,solve_request_sha256=request.solve_request_sha256,domain=domain,
            subject_local_box=local,before_xyz=tuple(place.dyadic(v) for v in _identity(subject.pose.world_from_object)),targets=tuple(targets),
            cavity_facts=tuple(cavities[key] for key in sorted(cavities)),source_fact_sha256=request.semantic_problem.scene_state.scene_state_sha256,limits=context.limits,ledger=budget.ledger())
    except UnsupportedPlacement as error:
        error.ledger=budget.ledger(kernel.PlacementLimit("UNSUPPORTED",budget.current))
        raise
    except kernel.PlacementLimit as error:
        error.ledger=budget.ledger(error)
        raise


def materialize_endpoint(request,compilation,target_id,xyz):
    """Replay exactly one endpoint and validate the generic complete delta."""
    fresh=compile_semantic_place(request)
    if fresh!=compilation:raise ValueError("stale or altered placement compilation")
    context=validate_semantic_place_request(request)
    return _materialize_validated(context,compilation,target_id,xyz,kernel.Budget(context.limits,"SOLVE"))


def _materialize_validated(context,compilation,target_id,xyz,budget):
    """One ephemeral transition from a freshly compiled caller-owned context.

    Producer and checker call this independently. No compiled input or result
    is cached; the public entry point always verifies the compilation first.
    """
    request=context.request
    budget.current="transition:endpoint"
    budget.charge()
    place.canonical_numbers(xyz)
    if len(xyz)!=3 or any(type(v) is not float for v in xyz):raise ValueError("endpoint requires finite binary64 XYZ")
    targets={target.target_id:target for target in compilation.targets}
    target=targets.get(target_id)
    if target is None or target.xy_bounds is None:raise ValueError("endpoint target is outside the feasible authorization")
    x,y,z=tuple(F(v) for v in xyz)
    x0,y0,x1,y1=(v.as_fraction for v in target.xy_bounds)
    if not x0<=x<=x1 or not y0<=y<=y1 or z!=target.height.as_fraction:raise ValueError("endpoint violates contact or typed domain")
    obstacles=tuple(tuple(v.as_fraction for v in row) for row in target.forbidden_open_rectangles)
    if kernel._blocked(x,y,obstacles,budget) is not None:raise ValueError("endpoint collides")
    before=request.semantic_problem.scene_state
    scene=before.base_scene_payload
    objects=[]
    for obj in scene.objects.values:
        if obj.object_id==compilation.domain.subject_id:
            pose=obj.pose.model_copy(update={"world_from_object":obj.pose.world_from_object.model_copy(update={"translation":Vec3(x=xyz[0],y=xyz[1],z=xyz[2])})})
            obj=obj.model_copy(update={"pose":pose,"support_assignment":obj.support_assignment.model_copy(update={"surface_id":target.support_surface_id})})
        objects.append(obj)
    after_scene=CanonicalScene.model_validate(scene.model_copy(update={"objects":scene.objects.model_copy(update={"values":tuple(objects)})}).model_dump(mode="python"),strict=True)
    facts=tuple(f for bundle in before.extension_fact_bundles for f in bundle.facts)
    after,_=_state(after_scene,facts)
    old=generic._leaf_values(before,scene);new=generic._leaf_values(after,after_scene)
    changed={key for key in old if canonical_json_bytes(old[key])!=canonical_json_bytes(new[key])}
    leaves=before.canonical_state_leaf_index.leaves
    writes=ordered(leaf for leaf in leaves if generic._state_key(leaf) in changed)
    if not set(writes)<=set(compilation.domain.authorized_leaves):raise ValueError("endpoint changed frozen state")
    unchanged=tuple((leaf,old[generic._state_key(leaf)]) for leaf in leaves if generic._state_key(leaf) not in changed)
    delta=StateDeltaManifest.seal(authorized_primary_writes=writes,recomputed_derived_writes=(),unchanged_leaves_digest=canonical_sha256(unchanged,domain=generic._UNCHANGED_LEAF_DOMAIN),
        complete_before_leaf_index_sha256=before.canonical_state_leaf_index.state_leaf_index_sha256,complete_after_leaf_index_sha256=after.canonical_state_leaf_index.state_leaf_index_sha256)
    problem=request.semantic_problem
    def obligations(values):
        return tuple(GroundedObligation(context=value,source_definition_refs=ordered(generic._references(value,"definition:"))) for value in values)
    grounded=GroundedObligationSet.seal(before_preconditions=obligations(problem.before_preconditions),after_goals=obligations((problem.after_goal,)),preservation_invariants=obligations(problem.preservation_invariants))
    args=tuple(OperationArgument(argument_name='argument:'+name,value=TypedValue(value_schema_ref=ref,payload=CanonicalIdValue(value=value) if isinstance(value,str) else FiniteRealValue(value=value)))
        for name,ref,value in (('subject',s('subject-id'),compilation.domain.subject_id),('target',s('target-id'),target_id),('x',s('real'),xyz[0]),('y',s('real'),xyz[1]),('z',s('real'),xyz[2])))
    program=EditProgram.seal(program_id="program:semantic-place-endpoint",semantic_problem_sha256=request.semantic_problem_sha256,action_space_profile_sha256=context.registry_arguments['action_space_profile'].action_space_profile_sha256,
        steps=(OperationInvocation(operator_ref=d('operator/'+compilation.domain.operation),arguments=args),),before_state_sha256=before.scene_state_sha256,
        after_scene_state=after,after_scene_state_sha256=after.scene_state_sha256,state_delta_manifest=delta,grounded_obligation_set_sha256=grounded.grounded_obligation_set_sha256)
    arguments={key:value for key,value in context.registry_arguments.items() if key not in {'request','backend_descriptor_bundle','solver_config','proof_policy','resource_policy','backend_routing_policy'}}
    context.registry.validate_edit_program(**arguments,intervention_authorization=problem.intervention_authorization,program=program,grounded_obligations=grounded,before_state=before,after_state=after)
    _,local,before_world,_,cavities,cavity_boxes,_=_geometry(context,budget)
    after_world=kernel.translate(local,(x,y,z),budget)
    truths=tuple(place.SemanticPlaceTruthRow(cavity_id=cid,before=kernel.contains(cavity_boxes[cid],before_world,budget),after=kernel.contains(cavity_boxes[cid],after_world,budget)) for cid in sorted(cavities))
    budget.completed.append("transition:endpoint")
    return program,grounded,truths
