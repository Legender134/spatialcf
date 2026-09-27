"""Static registry problem validation and explicit dependencies."""

from __future__ import annotations

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
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
    WriteAuthority,
)

from spatialcf.domain.predicates import (
    PredicateAtom,
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    ImplementationRegistrySnapshot,
    SemanticsProfile,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    SemanticContractError,
    _OBJECTIVE_INPUT_SELECTOR_FIELD,
    _OBJECTIVE_NORMALIZATION_FIELD,
    _OBJECTIVE_UNIT_FIELD,
    _PREREQUISITE_FACT_FAMILY_FIELD,
    _self_digest_matches,
    _state_key,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_record_bindings,
    _definition_record_reference,
    _definition_roles,
    _definition_root_maps,
    _references,
    _semantic_schema_definition_closure,
    _validate_exact_root_definition_closure,
    _validate_typed_value,
    _walk_values,
)

from spatialcf.core._internal.registry.state import (
    _base_fact_rows,
    _declared_extension_leaf_owners,
    _expected_base_leaf_addresses,
    _extension_facts_by_address,
    _state_definition_metadata,
    _state_definition_semantics,
    _state_leaf_owners,
    _validate_scene_state_envelope,
    _validate_state_leaf_value_wires,
)


def _formula_predicate_refs(value: object) -> set[str]:
    return {
        item.predicate_ref
        for item in _walk_values(value)
        if isinstance(item, PredicateAtom)
    }


def _formula_atoms(value: object) -> tuple[PredicateAtom, ...]:
    """Return every atom in a closed formula tree without reinterpreting IDs."""

    return tuple(
        item for item in _walk_values(value) if isinstance(item, PredicateAtom)
    )


def _formula_is_fully_grounded(formula: object) -> bool:
    """Require a semantic root to carry the finite grounded formula, not a template."""

    if type(formula) is bool:
        return True
    ground = getattr(formula, "ground", None)
    if ground is None:
        return False
    try:
        return canonical_json_bytes(formula) == canonical_json_bytes(ground())
    except ValueError:
        return False


class ProblemValidation:
    """Stateless validation methods composed by StaticImplementationRegistry."""

    def validate_semantic_problem(
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
    ) -> CounterfactualProblemIR:
        """Validate problem/profile/scene/action closure on explicit input records."""

        # Specialized records are not roots merely because a caller supplied
        # them alongside a problem.  The profiles and request authorization
        # choose the finite semantic universe; accepting an extra record here
        # would let an otherwise unreachable envelope and binding make itself
        # reachable in the semantic definition bundle below.
        try:
            supplied_predicate_refs = tuple(
                definition.predicate_ref for definition in predicate_definitions
            )
            supplied_operator_refs = tuple(
                definition.operator_ref for definition in operator_definitions
            )
            supplied_state_refs = tuple(
                definition.state_variable_ref
                for definition in state_variable_definitions
            )
            supplied_derived_rule_refs = tuple(
                definition.derived_fact_rule_ref
                for definition in derived_fact_rule_definitions
            )
        except AttributeError as error:
            raise SemanticContractError("profile definition closure") from error

        expected_predicate_refs = set(semantics_profile.predicate_definition_refs)
        expected_operator_refs = set(action_space_profile.allowed_operator_refs)
        expected_state_refs = set(action_space_profile.state_variable_definition_refs)
        if (
            len(set(supplied_predicate_refs)) != len(supplied_predicate_refs)
            or len(set(supplied_operator_refs)) != len(supplied_operator_refs)
            or len(set(supplied_state_refs)) != len(supplied_state_refs)
            or len(set(supplied_derived_rule_refs)) != len(supplied_derived_rule_refs)
        ):
            raise SemanticContractError("profile definition closure")

        # Preserve the semantic source of a malformed cross-record edge before
        # enforcing exact supplied specialized tuples.  A profile omission is
        # not the same violation as a caller-supplied unreferenced record: the
        # former must identify authorization, operator, or derived-output
        # authority, while the latter remains a closed-root failure below.
        authorization = problem.intervention_authorization
        for leaf in authorization.authorized_primary_write_set:
            matching_definitions = tuple(
                definition
                for definition in state_variable_definitions
                if definition.state_variable_schema_ref
                == leaf.state_variable_schema_ref
            )
            if (
                len(matching_definitions) == 1
                and matching_definitions[0].state_variable_ref
                not in expected_state_refs
            ):
                raise SemanticContractError("authorization profile state")

        supplied_derived_rule_ref_set = set(supplied_derived_rule_refs)
        semantic_derived_rule_refs = set(semantics_profile.derived_fact_rule_refs)
        required_derived_rule_refs = set(authorization.required_derived_rule_refs)
        if (
            not required_derived_rule_refs <= semantic_derived_rule_refs
            or not required_derived_rule_refs <= supplied_derived_rule_ref_set
        ):
            raise SemanticContractError("authorization derived rule")
        operator_derived_rule_refs = {
            rule_ref
            for definition in operator_definitions
            for rule_ref in definition.derived_write_rule_refs
        }
        if not required_derived_rule_refs <= operator_derived_rule_refs:
            raise SemanticContractError("authorization derived rule")
        for definition in operator_definitions:
            if not set(definition.derived_write_rule_refs) <= (
                semantic_derived_rule_refs & supplied_derived_rule_ref_set
            ):
                raise SemanticContractError("operator derived rule")

        for rule in derived_fact_rule_definitions:
            for output in rule.fixed_output_set:
                matching_definitions = tuple(
                    definition
                    for definition in state_variable_definitions
                    if definition.state_variable_schema_ref
                    == output.state_variable_schema_ref
                )
                if len(matching_definitions) != 1:
                    continue
                output_definition = matching_definitions[0]
                if output_definition.write_authority is not WriteAuthority.DERIVED_ONLY:
                    raise SemanticContractError("derived output authority")
                if (
                    output_definition.derived_fact_rule_ref
                    != rule.derived_fact_rule_ref
                ):
                    raise SemanticContractError("derived output owner")
        for definition in state_variable_definitions:
            if (
                definition.derived_fact_rule_ref is not None
                and definition.derived_fact_rule_ref
                not in semantic_derived_rule_refs & supplied_derived_rule_ref_set
            ):
                raise SemanticContractError("derived output owner")

        if (
            set(supplied_predicate_refs) != expected_predicate_refs
            or set(supplied_operator_refs) != expected_operator_refs
            or set(supplied_state_refs) != expected_state_refs
        ):
            raise SemanticContractError("profile definition closure")

        supplied_states_by_ref = {
            definition.state_variable_ref: definition
            for definition in state_variable_definitions
        }
        selected_state_definitions = tuple(
            supplied_states_by_ref[reference] for reference in expected_state_refs
        )
        expected_derived_rule_refs = required_derived_rule_refs | {
            definition.derived_fact_rule_ref
            for definition in selected_state_definitions
            if definition.derived_fact_rule_ref is not None
        }
        if (
            set(supplied_derived_rule_refs) != expected_derived_rule_refs
            or not expected_derived_rule_refs <= semantic_derived_rule_refs
        ):
            raise SemanticContractError("profile definition closure")

        self.validate_definition_closure(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
            value_schema_definitions,
            predicate_definitions,
            state_variable_definitions,
            derived_fact_rule_definitions,
            operator_definitions,
            implementation_registry_snapshot,
        )
        if problem.definition_bundle != semantic_definition_bundle:
            raise SemanticContractError("semantic definition bundle")
        if problem.scene_state != scene_state:
            raise SemanticContractError("scene state")
        if (
            problem.semantics_profile_ref != semantics_profile.semantics_profile_ref
            or problem.action_space_profile_ref
            != action_space_profile.action_space_profile_ref
        ):
            raise SemanticContractError("semantic/action profile")
        profile_definitions, solve_definitions, _definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        profile_bindings = _definition_record_bindings(
            profile_definitions,
            _definition_roles(profile_definitions),
        )
        for reference, digest in (
            (
                semantics_profile.semantics_profile_ref,
                semantics_profile.semantics_profile_sha256,
            ),
            (
                action_space_profile.action_space_profile_ref,
                action_space_profile.action_space_profile_sha256,
            ),
        ):
            if profile_bindings.get(reference) != digest:
                raise DefinitionClosureError("definition record binding")
        if (
            not _references(semantics_profile, "definition:")
            <= profile_definitions.keys()
        ):
            raise DefinitionClosureError("semantic definition root")
        action_profile_refs = _references(action_space_profile, "definition:")
        permitted_operational_profile_ref = {
            action_space_profile.publication_proof_policy_ref
        }
        if (
            not action_profile_refs - permitted_operational_profile_ref
            <= profile_definitions.keys()
            or action_space_profile.publication_proof_policy_ref
            not in solve_definitions
        ):
            raise DefinitionClosureError("semantic definition root")

        # Do not let the caller's schema tuple or the problem's embedded
        # definition bundle establish semantic meaning.  The finite schema
        # universe starts at real problem/profile records and the exact
        # profile-selected specialized records, then closes through semantic
        # definition payloads and declared schema dependencies.
        semantic_root_records: tuple[object, ...] = (
            scene_state,
            problem.intervention_authorization,
            problem.before_preconditions,
            problem.after_goal,
            problem.preservation_invariants,
            problem.explicit_observation_obligations,
            problem.objective_expression,
            problem.numeric_semantics_ref,
            semantics_profile,
            action_space_profile,
            *predicate_definitions,
            *state_variable_definitions,
            *derived_fact_rule_definitions,
            *operator_definitions,
        )
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        expected_schema_refs = _semantic_schema_definition_closure(
            profile_definitions,
            schemas,
            semantic_root_records,
        )
        if set(schemas) != expected_schema_refs:
            raise DefinitionClosureError("semantic schema closure")
        _validate_exact_root_definition_closure(
            profile_definitions,
            root_records=semantic_root_records,
            record_binding_targets={
                problem.numeric_semantics_ref,
                semantics_profile.semantics_profile_ref,
                action_space_profile.action_space_profile_ref,
                *expected_schema_refs,
            },
            root_label="semantic",
            include_complete_domain_dependents=True,
        )
        if (
            problem.numeric_semantics_ref != semantics_profile.numeric_semantics_ref
            or action_space_profile.numeric_semantics_ref
            != semantics_profile.numeric_semantics_ref
        ):
            raise SemanticContractError("profile numeric semantics")
        if (
            scene_state.base_scene_schema_ref
            not in semantics_profile.accepted_scene_and_fact_schema_refs
            or scene_state.base_scene_schema_ref
            not in action_space_profile.accepted_scene_schema_refs
        ):
            raise SemanticContractError("scene schema")

        _, leaf_values, base_scene = _validate_scene_state_envelope(
            scene_state,
            state_variable_definitions,
            value_schema_definitions,
            profile_definitions,
            after_state=False,
        )

        expected_entities = {
            identifier for _family, identifier, _value in _base_fact_rows(base_scene)
        } | {
            fact.subject_entity_id
            for bundle in scene_state.extension_fact_bundles
            for fact in bundle.facts
        }
        if set(scene_state.closed_entity_index) != expected_entities:
            raise SemanticContractError("complete scene entity index")
        leaves = scene_state.canonical_state_leaf_index.leaves
        leaf_keys = {_state_key(leaf) for leaf in leaves}
        base_owners = {
            (family, identifier)
            for family, identifier, _fact in _base_fact_rows(base_scene)
        }
        extension_facts = _extension_facts_by_address(scene_state)
        base_addresses = {
            (leaf.fact_family_ref, leaf.entity_or_fact_key, leaf.field_path_ref)
            for leaf in leaves
            if (leaf.fact_family_ref, leaf.entity_or_fact_key) in base_owners
        }
        if base_addresses != _expected_base_leaf_addresses(base_scene):
            raise SemanticContractError("exact state leaf index")
        by_family_identifier = _state_leaf_owners(leaves)
        for family, identifier, _value in _base_fact_rows(base_scene):
            owned = by_family_identifier.get((family, identifier), ())
            if not any(
                leaf.field_path_ref.startswith("field-path:presence-") for leaf in owned
            ):
                raise SemanticContractError("complete state leaf presence")
            if not any(
                not leaf.field_path_ref.startswith("field-path:presence-")
                for leaf in owned
            ):
                raise SemanticContractError("complete state leaf value")

        state_by_schema = {
            definition.state_variable_schema_ref: definition
            for definition in state_variable_definitions
        }
        state_by_ref = {
            definition.state_variable_ref: definition
            for definition in state_variable_definitions
        }
        if len(state_by_schema) != len(state_variable_definitions) or len(
            state_by_ref
        ) != len(state_variable_definitions):
            raise SemanticContractError("state leaf definition")
        if any(
            leaf.state_variable_schema_ref not in state_by_schema for leaf in leaves
        ):
            raise SemanticContractError("state leaf definition")
        if not leaf_keys:
            raise SemanticContractError("complete state leaf index")

        # A submitted leaf is not owned merely because its schema happens to
        # occur in a state definition.  Its complete address and value schema
        # must come from the explicit typed definition record.  This makes the
        # frozen scene index a closed semantic partition, not an opaque list of
        # presence/value placeholders.
        definitions, _solve_definitions, _all_definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        metadata_by_ref = _state_definition_metadata(definitions)
        if set(metadata_by_ref) != set(state_by_ref):
            raise SemanticContractError("schema-owned field path")
        metadata_by_address: dict[
            tuple[str, str, str, str, str], StateVariableDefinition
        ] = {}
        for definition_ref, state_definition in state_by_ref.items():
            (
                state_variable_schema_ref,
                state_schema_ref,
                fact_family_ref,
                entity_or_fact_key,
                field_path_ref,
                value_schema_ref,
            ) = metadata_by_ref[definition_ref]
            if (
                state_variable_schema_ref != state_definition.state_variable_schema_ref
                or value_schema_ref != state_definition.value_schema_ref
            ):
                raise SemanticContractError("schema-owned field path")
            address = (
                state_variable_schema_ref,
                state_schema_ref,
                fact_family_ref,
                entity_or_fact_key,
                field_path_ref,
            )
            if address in metadata_by_address:
                raise SemanticContractError("schema-owned field path")
            metadata_by_address[address] = state_definition
        leaf_by_address = {
            (
                leaf.state_variable_schema_ref,
                leaf.state_schema_ref,
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
                leaf.field_path_ref,
            ): leaf
            for leaf in leaves
        }
        if set(leaf_by_address) != set(metadata_by_address):
            raise SemanticContractError("schema-owned field path")
        extension_leaf_owners = _declared_extension_leaf_owners(
            leaves,
            base_owners,
        )
        if not set(extension_facts) <= set(extension_leaf_owners):
            raise SemanticContractError("extension fact address")
        for owned in extension_leaf_owners.values():
            if (
                len(owned) != 2
                or sum(
                    leaf.field_path_ref.startswith("field-path:presence-")
                    for leaf in owned
                )
                != 1
            ):
                raise SemanticContractError("complete extension state leaf")
        absent_extension_addresses = set(extension_leaf_owners) - set(extension_facts)
        semantics_by_ref = _state_definition_semantics(definitions)
        if set(semantics_by_ref) != set(state_by_ref):
            raise SemanticContractError("state leaf semantics")
        for definition_ref, state_definition in state_by_ref.items():
            if semantics_by_ref[definition_ref] != (
                state_definition.frame_ref,
                state_definition.unit_ref,
                state_definition.topology_ref,
            ):
                raise SemanticContractError("state leaf semantics")
        semantic_definition_refs = {
            reference
            for record in semantic_root_records
            for reference in _references(record, "definition:")
        } - permitted_operational_profile_ref
        if not semantic_definition_refs <= profile_definitions.keys():
            raise SemanticContractError("semantic definition closure")
        profile_capabilities = _references(action_space_profile, "capability:")
        self._validate_snapshot(
            implementation_registry_snapshot,
            profile_capabilities,
            (
                *predicate_definitions,
                *state_variable_definitions,
                *derived_fact_rule_definitions,
                *operator_definitions,
            ),
        )
        try:
            _validate_state_leaf_value_wires(
                leaves,
                leaf_values,
                {
                    _state_key(leaf): metadata_by_address[address]
                    for address, leaf in leaf_by_address.items()
                },
                schemas,
                set(extension_facts),
                absent_extension_addresses,
            )
        except DefinitionClosureError as error:
            raise SemanticContractError("state leaf value schema") from error

        predicate_by_ref = {
            definition.predicate_ref: definition for definition in predicate_definitions
        }
        derived_rules = {
            definition.derived_fact_rule_ref: definition
            for definition in derived_fact_rule_definitions
        }
        if (
            set(predicate_by_ref) != expected_predicate_refs
            or set(derived_rules) != expected_derived_rule_refs
            or set(state_by_ref) != expected_state_refs
        ):
            raise SemanticContractError("profile definition closure")
        predicate_refs = _formula_predicate_refs(
            (
                problem.before_preconditions,
                problem.after_goal,
                problem.preservation_invariants,
                problem.explicit_observation_obligations,
            )
        )
        if not predicate_refs <= predicate_by_ref.keys():
            raise SemanticContractError("formula predicate closure")
        if not predicate_refs <= set(semantics_profile.predicate_definition_refs):
            raise SemanticContractError("profile predicate closure")
        formula_contexts = (
            *(item.formula for item in problem.before_preconditions),
            problem.after_goal.formula,
            *(
                formula
                for invariant in problem.preservation_invariants
                for formula in (invariant.before_formula, invariant.after_formula)
            ),
            *(item.formula for item in problem.explicit_observation_obligations),
        )
        if not all(_formula_is_fully_grounded(formula) for formula in formula_contexts):
            raise SemanticContractError("formula grounding")
        for atom in _formula_atoms(formula_contexts):
            predicate = predicate_by_ref.get(atom.predicate_ref)
            if (
                predicate is None
                or tuple(operand.value_schema_ref for operand in atom.operands)
                != predicate.operand_schema_refs
            ):
                raise SemanticContractError("formula predicate closure")
            try:
                for operand in atom.operands:
                    _validate_typed_value(operand, schemas)
            except DefinitionClosureError as error:
                raise SemanticContractError("formula predicate closure") from error
        actual_fact_families = {
            family for family, _identifier, _fact in _base_fact_rows(base_scene)
        } | {
            fact.fact_family_ref
            for bundle in scene_state.extension_fact_bundles
            for fact in bundle.facts
        }
        for predicate_ref in predicate_refs:
            predicate = predicate_by_ref[predicate_ref]
            for prerequisite_ref in predicate.observation_prerequisite_template_refs:
                prerequisite = definitions.get(prerequisite_ref)
                if prerequisite is None:
                    raise SemanticContractError("prerequisite closure")
                try:
                    required_fact_family = _definition_record_reference(
                        prerequisite,
                        _PREREQUISITE_FACT_FAMILY_FIELD,
                    )
                except DefinitionClosureError as error:
                    raise SemanticContractError("prerequisite closure") from error
                if required_fact_family not in actual_fact_families:
                    raise SemanticContractError("required fact completeness")

        operators = {
            definition.operator_ref: definition for definition in operator_definitions
        }
        if set(operators) != expected_operator_refs:
            raise SemanticContractError("profile definition closure")
        authorization = problem.intervention_authorization
        if (
            not authorization.allowed_operator_refs
            or not set(authorization.allowed_operator_refs)
            <= set(action_space_profile.allowed_operator_refs)
            or not set(authorization.allowed_operator_refs) <= operators.keys()
        ):
            raise SemanticContractError("authorization")
        if not set(authorization.editable_entity_ids) <= set(
            scene_state.closed_entity_index
        ):
            raise SemanticContractError("authorization")
        authorization_keys = {
            _state_key(leaf) for leaf in authorization.authorized_primary_write_set
        }
        if not authorization_keys <= leaf_keys:
            raise SemanticContractError("authorization")
        leaf_by_key = {_state_key(leaf): leaf for leaf in leaves}
        extension_subjects = {
            address: fact.subject_entity_id
            for address, fact in _extension_facts_by_address(scene_state).items()
        }
        for leaf in authorization.authorized_primary_write_set:
            definition = state_by_schema.get(leaf.state_variable_schema_ref)
            if (
                definition is None
                or definition.write_authority is not WriteAuthority.PRIMARY_WRITABLE
                or extension_subjects.get(
                    (leaf.fact_family_ref, leaf.entity_or_fact_key),
                    leaf.entity_or_fact_key,
                )
                not in authorization.editable_entity_ids
            ):
                raise SemanticContractError("authorization")
        authorized_primary_definition_refs = {
            state_by_schema[leaf.state_variable_schema_ref].state_variable_ref
            for leaf in authorization.authorized_primary_write_set
        }
        profile_state_definition_refs = set(
            action_space_profile.state_variable_definition_refs
        )
        if not authorized_primary_definition_refs <= profile_state_definition_refs:
            raise SemanticContractError("authorization profile state")
        for bound in authorization.variable_bounds:
            bound_leaf = leaf_by_key.get(_state_key(bound.state_variable_ref))
            bound_definition = state_by_schema.get(
                bound.state_variable_ref.state_variable_schema_ref
            )
            if (
                _state_key(bound.state_variable_ref) not in authorization_keys
                or bound_definition is None
                or bound_definition.write_authority
                is not WriteAuthority.PRIMARY_WRITABLE
            ):
                raise SemanticContractError("variable bound authorization")
            if bound_leaf != bound.state_variable_ref or (
                bound.value_schema_ref,
                bound.frame_ref,
                bound.unit_ref,
                bound.topology_ref,
            ) != (
                bound_definition.value_schema_ref,
                bound_definition.frame_ref,
                bound_definition.unit_ref,
                bound_definition.topology_ref,
            ):
                raise SemanticContractError("variable bound semantics")
            try:
                _validate_typed_value(bound.typed_domain, schemas)
            except DefinitionClosureError as error:
                raise SemanticContractError("variable bound semantics") from error
        mandatory_invariant_template_refs = set(
            action_space_profile.mandatory_invariant_template_refs
        )
        if not mandatory_invariant_template_refs <= {
            invariant.transition_comparator_ref
            for invariant in problem.preservation_invariants
        }:
            raise SemanticContractError("mandatory invariant closure")
        required_derived_rule_refs = set(authorization.required_derived_rule_refs)
        if not required_derived_rule_refs <= set(
            semantics_profile.derived_fact_rule_refs
        ) or not required_derived_rule_refs <= {
            rule_ref
            for operator_ref in authorization.allowed_operator_refs
            for rule_ref in operators[operator_ref].derived_write_rule_refs
        }:
            raise SemanticContractError("authorization derived rule")
        for operator_ref in action_space_profile.allowed_operator_refs:
            operator = operators[operator_ref]
            if (
                operator.transition_semantics_ref
                not in semantics_profile.transition_semantics_refs
            ):
                raise SemanticContractError("operator transition semantics")
            for pattern in (
                *operator.read_footprint,
                *operator.primary_write_footprint,
            ):
                state_definition = state_by_ref.get(
                    pattern.state_variable_definition_ref
                )
                if state_definition is None:
                    raise SemanticContractError("operator state footprint")
                if (
                    pattern.state_variable_definition_ref
                    not in profile_state_definition_refs
                ):
                    raise SemanticContractError("operator profile state")
                if any(
                    binding.parameter_schema_ref not in operator.parameter_schema_refs
                    for binding in pattern.parameter_bindings
                ):
                    raise SemanticContractError("operator parameter binding")
            if not set(
                operator.derived_write_rule_refs
            ) <= derived_rules.keys() or not set(
                operator.derived_write_rule_refs
            ) <= set(semantics_profile.derived_fact_rule_refs):
                raise SemanticContractError("operator derived rule")
        allowed_primary_footprints = {
            pattern.state_variable_definition_ref
            for operator_ref in authorization.allowed_operator_refs
            for pattern in operators[operator_ref].primary_write_footprint
        }
        if not authorized_primary_definition_refs <= allowed_primary_footprints:
            raise SemanticContractError("authorization operator footprint")
        output_owners: dict[bytes, str] = {}
        for rule in derived_fact_rule_definitions:
            for read in rule.fixed_read_set:
                read_leaf = leaf_by_key.get(_state_key(read))
                read_definition = state_by_schema.get(read.state_variable_schema_ref)
                if read_leaf != read or read_definition is None:
                    raise SemanticContractError("derived read closure")
            for output in rule.fixed_output_set:
                output_leaf = leaf_by_key.get(_state_key(output))
                output_definition = state_by_schema.get(
                    output.state_variable_schema_ref
                )
                if (
                    output_leaf != output
                    or output_definition is None
                    or output_definition.write_authority
                    is not WriteAuthority.DERIVED_ONLY
                ):
                    raise SemanticContractError("derived output authority")
                if (
                    output_definition.derived_fact_rule_ref
                    != rule.derived_fact_rule_ref
                ):
                    raise SemanticContractError("derived output owner")
                output_key = _state_key(output)
                if output_key in output_owners:
                    raise SemanticContractError("derived output owner")
                output_owners[output_key] = rule.derived_fact_rule_ref
        derived_output_definition_refs = {
            state_by_schema[output.state_variable_schema_ref].state_variable_ref
            for rule in derived_fact_rule_definitions
            for output in rule.fixed_output_set
        }
        derived_only_definition_refs = {
            definition.state_variable_ref
            for definition in state_variable_definitions
            if definition.write_authority is WriteAuthority.DERIVED_ONLY
        }
        profile_primary_footprints = {
            pattern.state_variable_definition_ref
            for operator_ref in action_space_profile.allowed_operator_refs
            for pattern in operators[operator_ref].primary_write_footprint
        }
        if profile_primary_footprints & (
            derived_output_definition_refs | derived_only_definition_refs
        ):
            raise SemanticContractError("primary/derived footprint")
        if not required_derived_rule_refs <= {
            rule.derived_fact_rule_ref for rule in derived_fact_rule_definitions
        }:
            raise SemanticContractError("authorization derived rule")
        objective_refs = {
            term.objective_definition_ref for term in problem.objective_expression.terms
        }
        if not objective_refs <= set(semantics_profile.objective_definition_refs):
            raise SemanticContractError("objective closure")
        for term in problem.objective_expression.terms:
            objective_definition = definitions.get(term.objective_definition_ref)
            if objective_definition is None:
                raise SemanticContractError("objective closure")
            try:
                expected_objective_semantics = (
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_INPUT_SELECTOR_FIELD,
                    ),
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_UNIT_FIELD,
                    ),
                    _definition_record_reference(
                        objective_definition,
                        _OBJECTIVE_NORMALIZATION_FIELD,
                    ),
                )
            except DefinitionClosureError as error:
                raise SemanticContractError("objective unit/normalization") from error
            if expected_objective_semantics != (
                term.input_selector_definition_ref,
                term.unit_ref,
                term.normalization_definition_ref,
            ):
                raise SemanticContractError("objective unit/normalization")
        # Hash identity is checked after the explanatory semantic boundary so a
        # malformed authorization/leaf/profile reports that direct violation
        # rather than a stale enclosing root caused by a test-only model copy.
        for record, reason in (
            (problem, "semantic problem hash"),
            (semantics_profile, "semantics profile hash"),
            (action_space_profile, "action profile hash"),
            (scene_state, "scene state hash"),
        ):
            if not _self_digest_matches(record):
                raise SemanticContractError(reason)
        return problem


# Keep public class, exception and bound-method lookup stable.
_formula_predicate_refs.__module__ = "spatialcf.core.registry"
_formula_atoms.__module__ = "spatialcf.core.registry"
_formula_is_fully_grounded.__module__ = "spatialcf.core.registry"
ProblemValidation.validate_semantic_problem.__module__ = "spatialcf.core.registry"
ProblemValidation.validate_semantic_problem.__qualname__ = "StaticImplementationRegistry.validate_semantic_problem"
