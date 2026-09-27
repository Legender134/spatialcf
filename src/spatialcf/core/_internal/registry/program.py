"""Static registry program validation and explicit dependencies."""

from __future__ import annotations

from pydantic import (
    ValidationError,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
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

from spatialcf.domain.predicates import (
    GroundedObligationSet,
    PredicateDefinition,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    SemanticsProfile,
)

from spatialcf.domain.serialization import (
    canonical_sha256,
)

from spatialcf.core._internal.registry.contracts import (
    DefinitionClosureError,
    SemanticContractError,
    _UNCHANGED_LEAF_DOMAIN,
    _ref_key,
    _self_digest_matches,
    _state_key,
)

from spatialcf.core._internal.registry.definitions import (
    _definition_root_maps,
    _validate_typed_value,
)

from spatialcf.core._internal.registry.state import (
    _base_fact_rows,
    _declared_extension_leaf_owners,
    _expected_base_leaf_addresses,
    _extension_facts_by_address,
    _leaf_values,
    _state_definition_metadata,
    _validate_scene_state_envelope,
    _validate_state_leaf_value_wires,
    _validated_base_scene_payload,
)





class ProgramValidation:
    """Stateless validation methods composed by StaticImplementationRegistry."""

    def validate_edit_program(
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
        program: EditProgram,
        before_state: SceneStateEnvelope,
        after_state: SceneStateEnvelope,
        grounded_obligations: GroundedObligationSet,
        intervention_authorization: InterventionAuthorization,
    ) -> EditProgram:
        """Reconstruct the complete state partition for one explicit edit program."""

        self.validate_semantic_problem(
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
        )
        if intervention_authorization != problem.intervention_authorization:
            raise SemanticContractError("intervention authorization")
        if before_state != problem.scene_state:
            raise SemanticContractError("program before scene")
        try:
            before_base_scene = _validated_base_scene_payload(
                before_state.base_scene_payload
            )
        except (TypeError, ValidationError, ValueError) as error:
            raise SemanticContractError("program before scene") from error
        if (
            program.semantic_problem_sha256 != problem.semantic_problem_sha256
            or program.action_space_profile_sha256
            != action_space_profile.action_space_profile_sha256
        ):
            raise SemanticContractError("program roots")
        if (
            program.before_state_sha256 != before_state.scene_state_sha256
            or program.after_scene_state != after_state
            or program.after_scene_state_sha256 != after_state.scene_state_sha256
        ):
            raise SemanticContractError("program state")
        if after_state.base_scene_schema_ref != scene_state.base_scene_schema_ref:
            raise SemanticContractError("complete after state")
        if (
            program.grounded_obligation_set_sha256
            != grounded_obligations.grounded_obligation_set_sha256
        ):
            raise SemanticContractError("program obligations")

        definitions, _solve_definitions, _all_definitions = _definition_root_maps(
            semantic_definition_bundle,
            solve_policy_definition_bundle,
        )
        _, after_values, after_base_scene = _validate_scene_state_envelope(
            after_state,
            state_variable_definitions,
            value_schema_definitions,
            definitions,
            after_state=True,
        )

        manifest = program.state_delta_manifest
        primary_keys = {_state_key(leaf) for leaf in manifest.authorized_primary_writes}
        derived_keys = {_state_key(leaf) for leaf in manifest.recomputed_derived_writes}
        if primary_keys & derived_keys:
            raise SemanticContractError("primary/derived")
        before_leaves = {
            _state_key(leaf): leaf
            for leaf in before_state.canonical_state_leaf_index.leaves
        }
        after_leaves = {
            _state_key(leaf): leaf
            for leaf in after_state.canonical_state_leaf_index.leaves
        }
        complete_keys = set(before_leaves) | set(after_leaves)
        state_by_schema = {
            definition.state_variable_schema_ref: definition
            for definition in state_variable_definitions
        }
        metadata_by_ref = _state_definition_metadata(definitions)
        expected_after_addresses = {
            (
                metadata[0],
                metadata[1],
                metadata[2],
                metadata[3],
                metadata[4],
            )
            for metadata in metadata_by_ref.values()
        }
        actual_after_addresses = {
            (
                leaf.state_variable_schema_ref,
                leaf.state_schema_ref,
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
                leaf.field_path_ref,
            )
            for leaf in after_leaves.values()
        }
        if actual_after_addresses != expected_after_addresses:
            raise SemanticContractError("complete after state")
        if {
            (leaf.fact_family_ref, leaf.entity_or_fact_key, leaf.field_path_ref)
            for leaf in after_leaves.values()
            if (
                leaf.fact_family_ref,
                leaf.entity_or_fact_key,
            )
            in {
                (family, identifier)
                for family, identifier, _fact in _base_fact_rows(after_base_scene)
            }
        } != _expected_base_leaf_addresses(after_base_scene):
            raise SemanticContractError("complete after state")
        after_base_owners = {
            (family, identifier)
            for family, identifier, _fact in _base_fact_rows(after_base_scene)
        }
        after_extension_facts = _extension_facts_by_address(after_state)
        absent_after_extension_addresses = set(
            _declared_extension_leaf_owners(
                tuple(after_leaves.values()),
                after_base_owners,
            )
        ) - set(after_extension_facts)
        schemas = {
            schema.value_schema_ref: schema for schema in value_schema_definitions
        }
        after_definitions_by_leaf: dict[bytes, StateVariableDefinition] = {}
        for leaf in after_leaves.values():
            state_definition = state_by_schema.get(leaf.state_variable_schema_ref)
            if state_definition is None:
                raise SemanticContractError("complete after state")
            after_definitions_by_leaf[_state_key(leaf)] = state_definition
        try:
            _validate_state_leaf_value_wires(
                tuple(after_leaves.values()),
                after_values,
                after_definitions_by_leaf,
                schemas,
                set(after_extension_facts),
                absent_after_extension_addresses,
            )
        except DefinitionClosureError as error:
            raise SemanticContractError("complete after state") from error
        if not (primary_keys | derived_keys) <= complete_keys:
            raise SemanticContractError("complete delta partition")
        authorized_keys = {
            _state_key(leaf)
            for leaf in intervention_authorization.authorized_primary_write_set
        }
        if not primary_keys <= authorized_keys:
            raise SemanticContractError("authorization")
        output_owners: dict[bytes, str] = {}
        for rule in derived_fact_rule_definitions:
            if (
                rule.derived_fact_rule_ref
                not in intervention_authorization.required_derived_rule_refs
            ):
                continue
            for leaf in rule.fixed_output_set:
                key = _state_key(leaf)
                if key in output_owners:
                    raise SemanticContractError("derived output owner")
                output_owners[key] = rule.derived_fact_rule_ref
        rule_outputs = set(output_owners)
        if derived_keys != rule_outputs:
            raise SemanticContractError("derived footprint")
        before_values = _leaf_values(before_state, before_base_scene)
        changed = {
            key
            for key in complete_keys
            if canonical_json_bytes(before_values.get(key))
            != canonical_json_bytes(after_values.get(key))
        }
        if changed != primary_keys | derived_keys:
            raise SemanticContractError("complete delta partition")
        ordered_union_keys = tuple(
            _state_key(leaf) for leaf in before_state.canonical_state_leaf_index.leaves
        ) + tuple(
            key for key in sorted(set(after_leaves) - set(before_leaves), key=_ref_key)
        )
        unchanged_rows = tuple(
            (
                (before_leaves[key] if key in before_leaves else after_leaves[key]),
                before_values.get(key),
            )
            for key in ordered_union_keys
            if key not in changed
            and canonical_json_bytes(before_values.get(key))
            == canonical_json_bytes(after_values.get(key))
        )
        if manifest.unchanged_leaves_digest != canonical_sha256(
            unchanged_rows,
            domain=_UNCHANGED_LEAF_DOMAIN,
        ):
            raise SemanticContractError("unchanged-leaf digest")
        if (
            manifest.complete_before_leaf_index_sha256
            != before_state.canonical_state_leaf_index.state_leaf_index_sha256
            or manifest.complete_after_leaf_index_sha256
            != after_state.canonical_state_leaf_index.state_leaf_index_sha256
        ):
            raise SemanticContractError("complete leaf index")
        operators = {
            definition.operator_ref: definition for definition in operator_definitions
        }
        if len(program.steps) > intervention_authorization.maximum_program_steps:
            raise SemanticContractError("program steps")
        problem_precondition_bytes = {
            canonical_json_bytes(precondition)
            for precondition in problem.before_preconditions
        }
        primary_footprint_definition_refs: set[str] = set()
        derived_rule_refs: set[str] = set()
        for step in program.steps:
            definition = operators.get(step.operator_ref)
            if (
                definition is None
                or step.operator_ref
                not in intervention_authorization.allowed_operator_refs
            ):
                raise SemanticContractError("authorization")
            if (
                not {
                    canonical_json_bytes(precondition)
                    for precondition in definition.required_preconditions
                }
                <= problem_precondition_bytes
            ):
                raise SemanticContractError("operator required precondition")
            if len(step.arguments) != len(definition.parameter_schema_refs):
                raise SemanticContractError("operator parameters")
            if (
                tuple(argument.value.value_schema_ref for argument in step.arguments)
                != definition.parameter_schema_refs
            ):
                raise SemanticContractError("operator parameters")
            for argument in step.arguments:
                try:
                    _validate_typed_value(argument.value, schemas)
                except DefinitionClosureError as error:
                    raise SemanticContractError("operator parameters") from error
            arguments_by_name = {
                argument.argument_name: argument for argument in step.arguments
            }
            for pattern in (
                *definition.read_footprint,
                *definition.primary_write_footprint,
            ):
                for binding in pattern.parameter_bindings:
                    argument = arguments_by_name.get(binding.parameter_ref)
                    if (
                        argument is None
                        or argument.value.value_schema_ref
                        != binding.parameter_schema_ref
                    ):
                        raise SemanticContractError("operator parameter binding")
            primary_footprint_definition_refs.update(
                pattern.state_variable_definition_ref
                for pattern in definition.primary_write_footprint
            )
            derived_rule_refs.update(definition.derived_write_rule_refs)
            if not {
                pattern.state_variable_definition_ref
                for pattern in definition.read_footprint
            } <= {
                state_definition.state_variable_ref
                for state_definition in state_by_schema.values()
            }:
                raise SemanticContractError("operator footprint")
        if (
            not set(intervention_authorization.required_derived_rule_refs)
            <= derived_rule_refs
        ):
            raise SemanticContractError("derived footprint")
        if not derived_rule_refs <= set(
            intervention_authorization.required_derived_rule_refs
        ):
            raise SemanticContractError("authorization derived footprint")
        actual_primary_footprint = {
            key
            for key, leaf in ({**before_leaves, **after_leaves}).items()
            if state_by_schema[leaf.state_variable_schema_ref].state_variable_ref
            in primary_footprint_definition_refs
        }
        if not actual_primary_footprint <= authorized_keys:
            raise SemanticContractError("authorization operator footprint")
        if not primary_keys <= actual_primary_footprint:
            raise SemanticContractError("operator footprint")
        extension_subjects = {
            address: fact.subject_entity_id
            for state in (before_state, after_state)
            for address, fact in _extension_facts_by_address(state).items()
        }
        edited_entities = {
            extension_subjects.get(
                (
                    (before_leaves.get(key) or after_leaves[key]).fact_family_ref,
                    (before_leaves.get(key) or after_leaves[key]).entity_or_fact_key,
                ),
                (before_leaves.get(key) or after_leaves[key]).entity_or_fact_key,
            )
            for key in primary_keys
        }
        if len(edited_entities) > intervention_authorization.maximum_edited_entities:
            raise SemanticContractError("authorization")
        if not _self_digest_matches(grounded_obligations):
            raise SemanticContractError("grounded obligations")
        if not set(grounded_obligations.source_definition_refs) <= definitions.keys():
            raise SemanticContractError("grounded obligations")
        partitions = (
            (grounded_obligations.before_preconditions, problem.before_preconditions),
            (grounded_obligations.after_goals, (problem.after_goal,)),
            (
                grounded_obligations.preservation_invariants,
                problem.preservation_invariants,
            ),
            (
                grounded_obligations.observation_obligations,
                problem.explicit_observation_obligations,
            ),
        )
        if any(
            {canonical_json_bytes(item.context) for item in grounded_partition}
            != {canonical_json_bytes(item) for item in semantic_partition}
            for grounded_partition, semantic_partition in partitions
        ):
            raise SemanticContractError("grounded obligations")
        if any(
            canonical_json_bytes(obligation.context)
            not in {
                canonical_json_bytes(item.context)
                for item in grounded_obligations.after_goals
            }
            for definition in (operators[step.operator_ref] for step in program.steps)
            for obligation in definition.generated_obligations
        ):
            raise SemanticContractError("grounded obligations")
        if not _self_digest_matches(manifest) or not _self_digest_matches(program):
            raise SemanticContractError("program hash")
        return program


# Keep public class, exception and bound-method lookup stable.
ProgramValidation.validate_edit_program.__module__ = "spatialcf.core.registry"
ProgramValidation.validate_edit_program.__qualname__ = "StaticImplementationRegistry.validate_edit_program"
