"""Upright profile semantics contracts and intrinsic operations."""

from __future__ import annotations

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
)

from spatialcf.domain.definitions import (
    CanonicalDefinitionEnvelope,
    DefinitionBundle,
    FiniteOrderedTupleValue,
    HashBoundCanonicalModel,
    TypedValue,
)

from spatialcf.domain.predicates import (
    GroundedObligationSet,
)

from spatialcf.domain.profiles import (
    ObjectiveExpression,
    ObjectiveTerm,
    ResourcePolicy,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS,
    UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
    UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS,
    UPRIGHT_SE2_PREDICATE_DEFINITION_REFS,
    UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF,
    _OBJECTIVE_DEFINITION_KIND_REF,
    _OBJECTIVE_POLICY_SCHEMA_REF,
    _SEMANTIC_CLOSURE_KIND_REF,
    _SEMANTIC_CLOSURE_SCHEMA_REF,
    _SEMANTIC_DEFINITION_KIND_REF,
    _SEMANTIC_KIND_TO_REF,
)

from spatialcf.domain._upright_se2.policies import (
    UprightSE2ExecutablePolicyBundle,
    UprightSE2FiveTermObjectivePolicy,
    UprightSE2ObjectiveTermPolicy,
    UprightSE2SemanticDefinition,
    UprightSE2SemanticOwnerBinding,
    _objective_policy_refs,
    _objective_term_policy_values,
    _semantic_owner_binding,
    _validate_executable_policy_resource_binding,
    decode_upright_se2_objective_policy,
)

from spatialcf.domain._upright_se2.registration import (
    UprightSE2ProfileRegistration,
)

from spatialcf.domain._upright_se2.source_binding import (
    _bound_continuous_grounded_obligations,
    _bound_grounded_obligations,
    validate_upright_se2_executable_policy_visibility_binding,
)

from spatialcf.domain._upright_se2.values import (
    _semantic_definition_body,
    _semantic_definition_ref_for_kind,
    _semantic_definition_schema_for_kind,
    _semantic_typed_digest,
    _semantic_typed_digest_tuple,
    _semantic_typed_id,
    _semantic_typed_real,
    _semantic_typed_record,
    _semantic_typed_symbol,
    _semantic_typed_tuple,
)


class UprightSE2SemanticClosure(HashBoundCanonicalModel):
    """The typed, hash-bound input closure consumed by every future M3 evaluator."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/semantic-closure/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_closure_sha256"

    profile_registration_sha256: Sha256Digest
    profile_registration: UprightSE2ProfileRegistration
    semantic_problem: CounterfactualProblemIR
    definition_bundle: DefinitionBundle
    predicate_definitions: tuple[UprightSE2SemanticDefinition, ...]
    objective_policy: UprightSE2FiveTermObjectivePolicy
    evaluator_bindings: tuple[UprightSE2SemanticOwnerBinding, ...]
    grounded_obligations: GroundedObligationSet
    objective_expression: ObjectiveExpression
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle
    resource_policy: ResourcePolicy
    semantic_closure_sha256: Sha256Digest

    @property
    def definition_bundle_sha256(self) -> Sha256Digest:
        """Expose the nested definition-bundle digest as a typed closure field."""

        return self.definition_bundle.definition_bundle_sha256

    @property
    def grounded_obligation_set_sha256(self) -> Sha256Digest:
        """Expose the nested grounded-obligation digest for evaluator replay."""

        return self.grounded_obligations.grounded_obligation_set_sha256

    @property
    def objective_expression_sha256(self) -> Sha256Digest:
        """Expose the nested objective-expression digest for evaluator replay."""

        return self.objective_expression.objective_expression_sha256

    @model_validator(mode="after")
    def _validate_closed_semantics(self) -> Self:
        if (
            self.profile_registration.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "semantic closure profile digest does not bind its profile"
            )
        _validate_task2_semantic_profile(self.profile_registration)
        if (
            self.semantic_problem.semantics_profile_ref
            != self.profile_registration.semantics_profile.semantics_profile_ref
            or self.semantic_problem.action_space_profile_ref
            != self.profile_registration.action_space_profile.action_space_profile_ref
        ):
            raise ValueError(
                "semantic closure source problem does not bind its profile"
            )
        expected_definitions = _upright_semantic_definitions()
        if tuple(
            canonical_json_bytes(value) for value in self.predicate_definitions
        ) != tuple(canonical_json_bytes(value) for value in expected_definitions):
            raise ValueError(
                "semantic closure must contain every exact predicate definition"
            )
        expected_policy = decode_upright_se2_objective_policy(
            self.executable_policy_bundle
        )
        if canonical_json_bytes(self.objective_policy) != canonical_json_bytes(
            expected_policy
        ):
            raise ValueError(
                "semantic closure must contain the exact five-term objective policy"
            )
        expected_bindings = _upright_semantic_owner_bindings()
        if self.evaluator_bindings != expected_bindings:
            raise ValueError(
                "semantic closure evaluator/verifier bindings do not close"
            )
        expected_bundle = _semantic_definition_bundle_for_digest(
            self.profile_registration_sha256, expected_policy
        )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            expected_bundle
        ):
            raise ValueError("semantic closure definition envelopes do not close")
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            build_upright_se2_objective_expression()
        ):
            raise ValueError("semantic closure objective expression does not close")
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            self.semantic_problem.definition_bundle
        ):
            raise ValueError(
                "semantic closure must bind the source problem definition bundle"
            )
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            self.semantic_problem.objective_expression
        ):
            raise ValueError(
                "semantic closure must bind the source problem objective expression"
            )
        expected_obligations = _bound_grounded_obligations(self.semantic_problem)
        if canonical_json_bytes(self.grounded_obligations) != canonical_json_bytes(
            expected_obligations
        ):
            raise ValueError(
                "semantic closure grounded obligations must bind the complete source roster"
            )
        if (
            self.executable_policy_bundle.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "executable policy bundle does not bind the semantic profile"
            )
        _validate_executable_policy_resource_binding(
            self.executable_policy_bundle,
            self.resource_policy,
        )
        validate_upright_se2_executable_policy_visibility_binding(
            self.executable_policy_bundle,
            self.semantic_problem,
        )
        return self

    @property
    def policy_bundle_sha256(self) -> Sha256Digest:
        """Expose the request-bound executable root to compiler consumers."""

        return self.executable_policy_bundle.policy_bundle_sha256


class UprightSE2ContinuousSemanticClosure(HashBoundCanonicalModel):
    """The additive continuous-yaw semantic closure.

    It intentionally has a distinct hash domain and validation entrypoint.  In
    particular, the frozen cardinal closure is never widened to accept an ARC
    or FULL_CIRCLE compiler-input wire.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-semantic-closure/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_closure_sha256"

    profile_registration_sha256: Sha256Digest
    profile_registration: UprightSE2ProfileRegistration
    semantic_problem: CounterfactualProblemIR
    definition_bundle: DefinitionBundle
    predicate_definitions: tuple[UprightSE2SemanticDefinition, ...]
    objective_policy: UprightSE2FiveTermObjectivePolicy
    evaluator_bindings: tuple[UprightSE2SemanticOwnerBinding, ...]
    grounded_obligations: GroundedObligationSet
    objective_expression: ObjectiveExpression
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle
    resource_policy: ResourcePolicy
    semantic_closure_sha256: Sha256Digest

    @property
    def definition_bundle_sha256(self) -> Sha256Digest:
        return self.definition_bundle.definition_bundle_sha256

    @property
    def grounded_obligation_set_sha256(self) -> Sha256Digest:
        return self.grounded_obligations.grounded_obligation_set_sha256

    @property
    def objective_expression_sha256(self) -> Sha256Digest:
        return self.objective_expression.objective_expression_sha256

    @model_validator(mode="after")
    def _validate_closed_semantics(self) -> Self:
        if (
            self.profile_registration.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "continuous semantic closure profile digest does not bind its profile"
            )
        _validate_task2_semantic_profile(self.profile_registration)
        if (
            self.semantic_problem.semantics_profile_ref
            != self.profile_registration.semantics_profile.semantics_profile_ref
            or self.semantic_problem.action_space_profile_ref
            != self.profile_registration.action_space_profile.action_space_profile_ref
        ):
            raise ValueError(
                "continuous semantic closure source problem does not bind its profile"
            )
        expected_definitions = _upright_semantic_definitions()
        if tuple(
            canonical_json_bytes(value) for value in self.predicate_definitions
        ) != tuple(canonical_json_bytes(value) for value in expected_definitions):
            raise ValueError(
                "continuous semantic closure must contain every exact predicate definition"
            )
        expected_policy = decode_upright_se2_objective_policy(
            self.executable_policy_bundle
        )
        if canonical_json_bytes(self.objective_policy) != canonical_json_bytes(
            expected_policy
        ):
            raise ValueError(
                "continuous semantic closure must contain the exact five-term objective policy"
            )
        if self.evaluator_bindings != _upright_semantic_owner_bindings():
            raise ValueError(
                "continuous semantic closure evaluator/verifier bindings do not close"
            )
        expected_bundle = _semantic_definition_bundle_for_digest(
            self.profile_registration_sha256, expected_policy
        )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            expected_bundle
        ):
            raise ValueError(
                "continuous semantic closure definition envelopes do not close"
            )
        if canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            build_upright_se2_objective_expression()
        ):
            raise ValueError(
                "continuous semantic closure objective expression does not close"
            )
        if canonical_json_bytes(self.definition_bundle) != canonical_json_bytes(
            self.semantic_problem.definition_bundle
        ) or canonical_json_bytes(self.objective_expression) != canonical_json_bytes(
            self.semantic_problem.objective_expression
        ):
            raise ValueError(
                "continuous semantic closure must bind the source semantic roots"
            )
        expected_obligations = _bound_continuous_grounded_obligations(
            self.semantic_problem
        )
        if canonical_json_bytes(self.grounded_obligations) != canonical_json_bytes(
            expected_obligations
        ):
            raise ValueError(
                "continuous semantic closure grounded obligations must bind the complete source roster"
            )
        if (
            self.executable_policy_bundle.profile_registration_sha256
            != self.profile_registration_sha256
        ):
            raise ValueError(
                "continuous executable policy bundle does not bind the semantic profile"
            )
        _validate_executable_policy_resource_binding(
            self.executable_policy_bundle,
            self.resource_policy,
        )
        validate_upright_se2_executable_policy_visibility_binding(
            self.executable_policy_bundle,
            self.semantic_problem,
        )
        return self

    @property
    def policy_bundle_sha256(self) -> Sha256Digest:
        return self.executable_policy_bundle.policy_bundle_sha256


def _upright_semantic_definitions() -> tuple[UprightSE2SemanticDefinition, ...]:
    kinds = tuple(
        sorted(
            _SEMANTIC_KIND_TO_REF,
            key=lambda kind: canonical_json_bytes(_SEMANTIC_KIND_TO_REF[kind]),
        )
    )
    return tuple(
        UprightSE2SemanticDefinition.seal(
            definition_ref=_semantic_definition_ref_for_kind(kind),
            definition_kind=kind,
            payload_schema_ref=_semantic_definition_schema_for_kind(kind),
            semantic_body=_semantic_definition_body(kind),
            owner_binding=_semantic_owner_binding(
                _semantic_definition_ref_for_kind(kind)
            ),
        )
        for kind in kinds
    )


def build_upright_se2_q0_target_objective_policy() -> UprightSE2FiveTermObjectivePolicy:
    """Build the explicit, source-independent unit target policy used by q=0."""

    return UprightSE2FiveTermObjectivePolicy.seal(
        **_objective_policy_refs(),
        terms=tuple(
            UprightSE2ObjectiveTermPolicy(
                term_id=term_id,
                **_objective_term_policy_values(term_id),
                weight=1.0,
                normalizer=1.0,
            )
            for term_id in ("T", "A", "R", "V", "S")
        ),
        owner_binding=_semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
    )


def build_upright_se2_objective_expression() -> ObjectiveExpression:
    """Return the fixed structural expression, without a numeric policy fallback."""

    return ObjectiveExpression.seal(
        aggregation_definition_ref=_objective_policy_refs()[
            "aggregation_definition_ref"
        ],
        terms=tuple(
            ObjectiveTerm(
                term_id=term_id,
                objective_definition_ref=UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
                input_selector_definition_ref=_objective_term_policy_values(term_id)[
                    "input_selector_definition_ref"
                ],
                unit_ref=_objective_term_policy_values(term_id)["unit_ref"],
                normalization_definition_ref=_objective_term_policy_values(term_id)[
                    "normalization_definition_ref"
                ],
            )
            for term_id in ("T", "A", "R", "V", "S")
        ),
        deterministic_tie_break_definition_ref=(
            _objective_policy_refs()["deterministic_tie_break_definition_ref"]
        ),
    )


def _semantic_owner_binding_payload(
    binding: UprightSE2SemanticOwnerBinding,
) -> TypedValue:
    return _semantic_typed_record(
        "schema:spatialcf/upright-se2/semantic-owner-binding/1.0",
        (
            ("definition_ref", _semantic_typed_id(binding.definition_ref)),
            (
                "evaluator_build_sha256",
                _semantic_typed_digest(binding.evaluator_build_sha256),
            ),
            (
                "evaluator_capability_ref",
                _semantic_typed_id(binding.evaluator_capability_ref),
            ),
            ("evaluator_owner_ref", _semantic_typed_id(binding.evaluator_owner_ref)),
            (
                "verifier_build_sha256",
                _semantic_typed_digest(binding.verifier_build_sha256),
            ),
            (
                "verifier_capability_ref",
                _semantic_typed_id(binding.verifier_capability_ref),
            ),
            ("verifier_owner_ref", _semantic_typed_id(binding.verifier_owner_ref)),
        ),
    )


def _semantic_definition_envelope(
    definition: UprightSE2SemanticDefinition,
) -> CanonicalDefinitionEnvelope:
    return CanonicalDefinitionEnvelope.seal(
        definition_ref=definition.definition_ref,
        definition_kind_ref=_SEMANTIC_DEFINITION_KIND_REF,
        payload_schema_ref=definition.payload_schema_ref,
        payload=_semantic_typed_record(
            definition.payload_schema_ref,
            (
                ("definition_kind", _semantic_typed_symbol(definition.definition_kind)),
                ("definition_ref", _semantic_typed_id(definition.definition_ref)),
                (
                    "owner_binding",
                    _semantic_owner_binding_payload(definition.owner_binding),
                ),
                ("semantic_body", definition.semantic_body),
                (
                    "semantic_definition_sha256",
                    _semantic_typed_digest(definition.semantic_definition_sha256),
                ),
            ),
        ),
    )


def _objective_term_payload(term: UprightSE2ObjectiveTermPolicy) -> TypedValue:
    return _semantic_typed_record(
        "schema:spatialcf/upright-se2/objective-term-policy/1.0",
        (
            (
                "input_selector_definition_ref",
                _semantic_typed_id(term.input_selector_definition_ref),
            ),
            ("metric_definition_ref", _semantic_typed_id(term.metric_definition_ref)),
            (
                "normalization_definition_ref",
                _semantic_typed_id(term.normalization_definition_ref),
            ),
            ("normalizer", _semantic_typed_real(term.normalizer)),
            ("normalizer_unit_ref", _semantic_typed_id(term.normalizer_unit_ref)),
            (
                "objective_definition_ref",
                _semantic_typed_id(term.objective_definition_ref),
            ),
            ("term_id", _semantic_typed_symbol(term.term_id)),
            ("unit_ref", _semantic_typed_id(term.unit_ref)),
            ("weight", _semantic_typed_real(term.weight)),
        ),
    )


def _objective_policy_envelope(
    policy: UprightSE2FiveTermObjectivePolicy,
) -> CanonicalDefinitionEnvelope:
    return CanonicalDefinitionEnvelope.seal(
        definition_ref=policy.objective_definition_ref,
        definition_kind_ref=_OBJECTIVE_DEFINITION_KIND_REF,
        payload_schema_ref=_OBJECTIVE_POLICY_SCHEMA_REF,
        payload=_semantic_typed_record(
            _OBJECTIVE_POLICY_SCHEMA_REF,
            (
                (
                    "aggregation_definition_ref",
                    _semantic_typed_id(policy.aggregation_definition_ref),
                ),
                (
                    "deterministic_tie_break_definition_ref",
                    _semantic_typed_id(policy.deterministic_tie_break_definition_ref),
                ),
                (
                    "directed_gap_subtraction_definition_ref",
                    _semantic_typed_id(policy.directed_gap_subtraction_definition_ref),
                ),
                (
                    "exact_equality_definition_ref",
                    _semantic_typed_id(policy.exact_equality_definition_ref),
                ),
                (
                    "exact_prune_comparator_ref",
                    _semantic_typed_id(policy.exact_prune_comparator_ref),
                ),
                (
                    "five_term_objective_policy_sha256",
                    _semantic_typed_digest(policy.five_term_objective_policy_sha256),
                ),
                (
                    "gap_prune_comparator_ref",
                    _semantic_typed_id(policy.gap_prune_comparator_ref),
                ),
                (
                    "interval_boundary_policy_ref",
                    _semantic_typed_id(policy.interval_boundary_policy_ref),
                ),
                (
                    "objective_definition_ref",
                    _semantic_typed_id(policy.objective_definition_ref),
                ),
                (
                    "owner_binding",
                    _semantic_owner_binding_payload(policy.owner_binding),
                ),
                (
                    "safety_penalty_definition_ref",
                    _semantic_typed_id(policy.safety_penalty_definition_ref),
                ),
                (
                    "strict_interval_comparator_ref",
                    _semantic_typed_id(policy.strict_interval_comparator_ref),
                ),
                (
                    "terms",
                    TypedValue(
                        value_schema_ref="schema:spatialcf/upright-se2/objective-term-policy-list/1.0",
                        payload=FiniteOrderedTupleValue(
                            element_schema_ref="schema:spatialcf/upright-se2/objective-term-policy/1.0",
                            items=tuple(
                                _objective_term_payload(term) for term in policy.terms
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _semantic_definition_bundle_for_digest(
    profile_registration_sha256: Sha256Digest,
    objective_policy: UprightSE2FiveTermObjectivePolicy,
) -> DefinitionBundle:
    definitions = _upright_semantic_definitions()
    policy = objective_policy
    predicate_envelopes = tuple(
        _semantic_definition_envelope(definition) for definition in definitions
    )
    objective_envelope = _objective_policy_envelope(policy)
    closure_envelope = CanonicalDefinitionEnvelope.seal(
        definition_ref=UPRIGHT_SE2_SEMANTIC_CLOSURE_DEFINITION_REF,
        definition_kind_ref=_SEMANTIC_CLOSURE_KIND_REF,
        payload_schema_ref=_SEMANTIC_CLOSURE_SCHEMA_REF,
        payload=_semantic_typed_record(
            _SEMANTIC_CLOSURE_SCHEMA_REF,
            (
                (
                    "definition_content_sha256s",
                    _semantic_typed_digest_tuple(
                        tuple(
                            definition.semantic_definition_sha256
                            for definition in definitions
                        )
                        + (policy.five_term_objective_policy_sha256,)
                    ),
                ),
                (
                    "objective_definition_ref",
                    _semantic_typed_id(policy.objective_definition_ref),
                ),
                (
                    "profile_registration_sha256",
                    _semantic_typed_digest(profile_registration_sha256),
                ),
                (
                    "semantic_definition_refs",
                    _semantic_typed_tuple(
                        tuple(definition.definition_ref for definition in definitions)
                        + (policy.objective_definition_ref,)
                    ),
                ),
            ),
        ),
    )
    return DefinitionBundle.seal(
        definitions=tuple(
            sorted(
                (*predicate_envelopes, objective_envelope, closure_envelope),
                key=lambda definition: canonical_json_bytes(definition.definition_ref),
            )
        )
    )


def _validate_task2_semantic_profile(
    registration: UprightSE2ProfileRegistration,
) -> None:
    """Require the complete Task 2 closure only where semantic evaluation begins.

    The Task 1 profile record remains a structural, capability-staged wire so
    its frozen profile tests can still construct the pre-evaluation form.  The
    compiler cannot reach a semantic bundle without this stricter M3 gate.
    """

    if (
        registration.semantics_profile.predicate_definition_refs
        != UPRIGHT_SE2_PREDICATE_DEFINITION_REFS
    ):
        raise ValueError(
            "semantics profile must name the complete upright predicate closure"
        )
    if registration.semantics_profile.objective_definition_refs != (
        UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
    ):
        raise ValueError("semantics profile must name the five-term objective policy")
    if (
        registration.action_space_profile.predicate_capability_refs
        != UPRIGHT_SE2_PREDICATE_CAPABILITY_REFS
    ):
        raise ValueError(
            "action-space profile must name the semantic predicate capabilities"
        )
    if (
        registration.action_space_profile.objective_capability_refs
        != UPRIGHT_SE2_OBJECTIVE_CAPABILITY_REFS
    ):
        raise ValueError(
            "action-space profile must name the semantic objective capabilities"
        )


def build_upright_se2_semantic_definition_bundle(
    registration: UprightSE2ProfileRegistration,
    *,
    objective_policy: UprightSE2FiveTermObjectivePolicy,
) -> DefinitionBundle:
    """Build the only profile/definition bundle accepted by the M3 compiler."""

    _validate_task2_semantic_profile(registration)
    return _semantic_definition_bundle_for_digest(
        registration.profile_registration_sha256, objective_policy
    )


def _upright_semantic_owner_bindings() -> tuple[UprightSE2SemanticOwnerBinding, ...]:
    return tuple(
        sorted(
            (
                *(
                    _semantic_owner_binding(definition_ref)
                    for definition_ref in UPRIGHT_SE2_PREDICATE_DEFINITION_REFS
                ),
                _semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
            ),
            key=canonical_json_bytes,
        )
    )


def build_upright_se2_semantic_closure(
    *,
    profile_registration: UprightSE2ProfileRegistration,
    semantic_problem: CounterfactualProblemIR,
    definition_bundle: DefinitionBundle,
    grounded_obligations: GroundedObligationSet,
    objective_expression: ObjectiveExpression,
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> UprightSE2SemanticClosure:
    """Seal all definitions, operands, objective policy, and owner builds together."""

    objective_policy = decode_upright_se2_objective_policy(executable_policy_bundle)
    return UprightSE2SemanticClosure.seal(
        profile_registration_sha256=profile_registration.profile_registration_sha256,
        profile_registration=profile_registration,
        semantic_problem=semantic_problem,
        definition_bundle=definition_bundle,
        predicate_definitions=_upright_semantic_definitions(),
        objective_policy=objective_policy,
        evaluator_bindings=_upright_semantic_owner_bindings(),
        grounded_obligations=grounded_obligations,
        objective_expression=objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=resource_policy,
    )


def build_upright_se2_continuous_semantic_closure(
    *,
    profile_registration: UprightSE2ProfileRegistration,
    semantic_problem: CounterfactualProblemIR,
    definition_bundle: DefinitionBundle,
    grounded_obligations: GroundedObligationSet,
    objective_expression: ObjectiveExpression,
    executable_policy_bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> UprightSE2ContinuousSemanticClosure:
    """Seal continuous-yaw semantics without widening the cardinal closure."""

    objective_policy = decode_upright_se2_objective_policy(executable_policy_bundle)
    return UprightSE2ContinuousSemanticClosure.seal(
        profile_registration_sha256=profile_registration.profile_registration_sha256,
        profile_registration=profile_registration,
        semantic_problem=semantic_problem,
        definition_bundle=definition_bundle,
        predicate_definitions=_upright_semantic_definitions(),
        objective_policy=objective_policy,
        evaluator_bindings=_upright_semantic_owner_bindings(),
        grounded_obligations=grounded_obligations,
        objective_expression=objective_expression,
        executable_policy_bundle=executable_policy_bundle,
        resource_policy=resource_policy,
    )


# Resolve local model forward references before restoring public identities.
UprightSE2SemanticClosure.model_rebuild()
UprightSE2ContinuousSemanticClosure.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2SemanticClosure.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousSemanticClosure.__module__ = "spatialcf.domain.upright_se2"
_upright_semantic_definitions.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_q0_target_objective_policy.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_objective_expression.__module__ = "spatialcf.domain.upright_se2"
_semantic_owner_binding_payload.__module__ = "spatialcf.domain.upright_se2"
_semantic_definition_envelope.__module__ = "spatialcf.domain.upright_se2"
_objective_term_payload.__module__ = "spatialcf.domain.upright_se2"
_objective_policy_envelope.__module__ = "spatialcf.domain.upright_se2"
_semantic_definition_bundle_for_digest.__module__ = "spatialcf.domain.upright_se2"
_validate_task2_semantic_profile.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_semantic_definition_bundle.__module__ = "spatialcf.domain.upright_se2"
_upright_semantic_owner_bindings.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_semantic_closure.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_continuous_semantic_closure.__module__ = "spatialcf.domain.upright_se2"
