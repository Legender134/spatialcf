"""Closed typed predicate expressions and grounded obligations for M1.

This module owns only local structural validation.  Definition resolution,
semantic evaluation, and implementation capability checks belong to the later
static registry; no adapter, core, geometry, or runtime discovery is reachable
from these immutable domain contracts.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self, TypeAlias

from pydantic import Field, StrictBool, StrictInt, TypeAdapter, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel, Sha256Digest
from spatialcf.domain.definitions import (
    CapabilityRef,
    DefinitionRef,
    HashBoundCanonicalModel,
    ReferenceValue,
    SchemaRef,
    TypedValue,
    ValueKind,
    canonical_json_bytes,
    canonical_sha256,
)

__all__ = (
    "AfterGoal",
    "AllOfFormula",
    "AnyOfFormula",
    "BeforePrecondition",
    "ExactlyKFormula",
    "ExistsFormula",
    "FiniteEntitySet",
    "ForAllFormula",
    "GroundedObligation",
    "GroundedObligationSet",
    "ImpliesFormula",
    "NotFormula",
    "ObservationObligation",
    "PredicateAtom",
    "PredicateDefinition",
    "PreservationInvariant",
)

_GROUNDING_ENTITY_SET_HASH_DOMAIN = "spatialcf/counterfactual/grounding-entity-set/3.0"
_GROUNDED_OBLIGATION_ID_DOMAIN = "spatialcf/counterfactual/grounded-obligation/3.0"


def _require_sorted_unique(
    values: tuple[str, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    if values != tuple(sorted(values, key=canonical_json_bytes)):
        raise ValueError(f"{label} must be sorted")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicate entries")


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicate entries")


class PredicateDefinition(HashBoundCanonicalModel):
    """The complete local meaning closure for one predicate definition."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/predicate-definition/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "definition_sha256"

    predicate_ref: DefinitionRef
    operand_schema_refs: tuple[SchemaRef, ...]
    state_context_policy_ref: DefinitionRef
    frame_requirement_ref: DefinitionRef
    measurement_definition_ref: DefinitionRef
    measurement_unit_ref: DefinitionRef
    comparator_definition_ref: DefinitionRef
    boundary_policy_ref: DefinitionRef
    tolerance_policy_ref: DefinitionRef
    uncertainty_policy_ref: DefinitionRef
    observation_prerequisite_template_refs: tuple[DefinitionRef, ...]
    evaluator_capability_ref: CapabilityRef
    verifier_capability_ref: CapabilityRef
    admissible_claim_definition_refs: tuple[DefinitionRef, ...]
    definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_definition_closure(self) -> Self:
        _require_unique(self.operand_schema_refs, "predicate operand schemas")
        _require_sorted_unique(
            self.observation_prerequisite_template_refs,
            "predicate observation prerequisite templates",
        )
        _require_sorted_unique(
            self.admissible_claim_definition_refs,
            "predicate admissible claim definitions",
        )
        closure_refs = (
            self.state_context_policy_ref,
            self.frame_requirement_ref,
            self.measurement_definition_ref,
            self.measurement_unit_ref,
            self.comparator_definition_ref,
            self.boundary_policy_ref,
            self.tolerance_policy_ref,
            self.uncertainty_policy_ref,
            *self.observation_prerequisite_template_refs,
            *self.admissible_claim_definition_refs,
        )
        _require_unique(closure_refs, "predicate definition closure")
        _require_unique(
            (self.evaluator_capability_ref, self.verifier_capability_ref),
            "predicate capability closure",
        )
        return self


class _GroundableFormula(CanonicalModel):
    """Private common grounding entry point for every closed formula node."""

    def ground(self) -> _StateFormula | bool:
        """Structurally expand finite quantifiers after validating closure."""

        _assert_closed_formula(self, ())
        return _ground_formula(self, ())


class PredicateAtom(_GroundableFormula):
    """One ordered predicate application without a state selector."""

    kind: Literal["PREDICATE_ATOM"] = "PREDICATE_ATOM"
    predicate_ref: DefinitionRef
    operands: tuple[TypedValue, ...]


class FiniteEntitySet(CanonicalModel):
    """An explicit finite domain and its one statically named bound variable."""

    variable_ref: CanonicalId
    entity_schema_ref: SchemaRef
    entities: tuple[TypedValue, ...]

    @model_validator(mode="after")
    def _validate_explicit_entity_domain(self) -> Self:
        if not self.variable_ref.startswith("variable:"):
            raise ValueError("finite entity set variable must start with 'variable:'")
        encoded = tuple(canonical_json_bytes(entity) for entity in self.entities)
        if len(set(encoded)) != len(encoded):
            raise ValueError("finite entity sets must not contain duplicate entities")
        if encoded != tuple(sorted(encoded)):
            raise ValueError("finite entity sets must be sorted")
        for entity in self.entities:
            if entity.value_schema_ref != self.entity_schema_ref:
                raise ValueError(
                    "finite entity set entity schema must match its domain"
                )
            if not isinstance(entity.payload, ReferenceValue) or (
                entity.payload.kind is not ValueKind.ENTITY_REF
            ):
                raise ValueError(
                    "finite entity sets must contain typed entity references"
                )
            if entity.payload.reference.startswith("variable:"):
                raise ValueError("finite entity sets may not discover another variable")
        return self


class NotFormula(_GroundableFormula):
    kind: Literal["NOT"] = "NOT"
    formula: _StateFormula


class AllOfFormula(_GroundableFormula):
    kind: Literal["ALL_OF"] = "ALL_OF"
    formulas: tuple[_StateFormula, ...]

    @model_validator(mode="after")
    def _normalize_formulas(self) -> Self:
        object.__setattr__(
            self,
            "formulas",
            _canonicalize_commutative_formulas(self.formulas, "all-of formulas"),
        )
        return self


class AnyOfFormula(_GroundableFormula):
    kind: Literal["ANY_OF"] = "ANY_OF"
    formulas: tuple[_StateFormula, ...]

    @model_validator(mode="after")
    def _normalize_formulas(self) -> Self:
        object.__setattr__(
            self,
            "formulas",
            _canonicalize_commutative_formulas(self.formulas, "any-of formulas"),
        )
        return self


class ImpliesFormula(_GroundableFormula):
    kind: Literal["IMPLIES"] = "IMPLIES"
    antecedent: _StateFormula
    consequent: _StateFormula


class ExactlyKFormula(_GroundableFormula):
    kind: Literal["EXACTLY_K"] = "EXACTLY_K"
    k: Annotated[StrictInt, Field(ge=0)]
    formulas: tuple[_StateFormula, ...]

    @model_validator(mode="after")
    def _normalize_formulas_and_validate_k(self) -> Self:
        formulas = _canonicalize_commutative_formulas(
            self.formulas,
            "exactly-k formulas",
        )
        if self.k > len(formulas):
            raise ValueError("exactly-k k must not exceed its formula count")
        object.__setattr__(self, "formulas", formulas)
        return self


class ForAllFormula(_GroundableFormula):
    kind: Literal["FOR_ALL"] = "FOR_ALL"
    entity_set: FiniteEntitySet
    formula: _StateFormula


class ExistsFormula(_GroundableFormula):
    kind: Literal["EXISTS"] = "EXISTS"
    entity_set: FiniteEntitySet
    formula: _StateFormula


_StateFormula: TypeAlias = Annotated[
    PredicateAtom
    | NotFormula
    | AllOfFormula
    | AnyOfFormula
    | ImpliesFormula
    | ExactlyKFormula
    | ForAllFormula
    | ExistsFormula,
    Field(discriminator="kind"),
]

# Finite grounding may reduce an otherwise closed formula to True or False.
# Those strict terminals are legal only as a context wrapper's root; every
# recursive formula field above remains the closed eight-case AST.
_GroundedContextFormula: TypeAlias = StrictBool | _StateFormula


class BeforePrecondition(CanonicalModel):
    """A before-state formula or strict terminal produced by finite grounding."""

    kind: Literal["BEFORE_PRECONDITION"] = "BEFORE_PRECONDITION"
    formula: _GroundedContextFormula

    @model_validator(mode="after")
    def _validate_closed_formula(self) -> Self:
        _assert_closed_context_formula(self.formula)
        return self


class AfterGoal(CanonicalModel):
    """An after-state formula or strict terminal produced by finite grounding."""

    kind: Literal["AFTER_GOAL"] = "AFTER_GOAL"
    formula: _GroundedContextFormula

    @model_validator(mode="after")
    def _validate_closed_formula(self) -> Self:
        _assert_closed_context_formula(self.formula)
        return self


class PreservationInvariant(CanonicalModel):
    """An explicit before/after transition obligation with grounded terminals."""

    kind: Literal["PRESERVATION_INVARIANT"] = "PRESERVATION_INVARIANT"
    before_formula: _GroundedContextFormula
    after_formula: _GroundedContextFormula
    transition_comparator_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_closed_formulas(self) -> Self:
        _assert_closed_context_formula(self.before_formula)
        _assert_closed_context_formula(self.after_formula)
        return self


class ObservationObligation(CanonicalModel):
    """A semantic observation with a formula or finite-grounding terminal."""

    kind: Literal["OBSERVATION_OBLIGATION"] = "OBSERVATION_OBLIGATION"
    phase: Literal["BEFORE", "AFTER"]
    formula: _GroundedContextFormula
    evidence_policy_ref: DefinitionRef

    @model_validator(mode="after")
    def _validate_closed_formula(self) -> Self:
        _assert_closed_context_formula(self.formula)
        return self


_ContextWrapper: TypeAlias = Annotated[
    BeforePrecondition | AfterGoal | PreservationInvariant | ObservationObligation,
    Field(discriminator="kind"),
]


class GroundedObligation(CanonicalModel):
    """One fully grounded context plus source refs carried on its wire.

    The context-derived ``obligation_id`` deliberately stays non-wire. Source
    references remain wire data and therefore contribute to the containing
    grounded-obligation set's canonical self digest.
    """

    context: _ContextWrapper
    source_definition_refs: tuple[DefinitionRef, ...]

    @model_validator(mode="after")
    def _validate_grounded_obligation(self) -> Self:
        _require_sorted_unique(
            self.source_definition_refs,
            "grounded obligation source definition references",
            nonempty=True,
        )
        if _contains_quantifier(_context_formulas(self.context)):
            raise ValueError(
                "grounded obligations must not contain unexpanded quantifiers"
            )
        return self

    @property
    def obligation_id(self) -> CanonicalId:
        """Stable context-derived semantic ID, deliberately excluding sources."""

        return f"obligation:{canonical_sha256(self.context, domain=_GROUNDED_OBLIGATION_ID_DOMAIN)}"

    @property
    def _semantic_bytes(self) -> bytes:
        return canonical_json_bytes(self.context)


class GroundedObligationSet(HashBoundCanonicalModel):
    """The complete hash-bound four-way partition and source closure."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/grounded-obligation-set/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "grounded_obligation_set_sha256"

    before_preconditions: tuple[GroundedObligation, ...] = ()
    after_goals: tuple[GroundedObligation, ...] = ()
    preservation_invariants: tuple[GroundedObligation, ...] = ()
    observation_obligations: tuple[GroundedObligation, ...] = ()
    grounding_entity_sets: tuple[FiniteEntitySet, ...] = ()
    source_definition_refs: tuple[DefinitionRef, ...]
    grounding_entity_set_sha256: Sha256Digest
    grounded_obligation_set_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Normalize semantic duplicates before binding the two declared hashes."""

        for field_name in (
            "source_definition_refs",
            "grounding_entity_set_sha256",
        ):
            if field_name in values:
                raise ValueError(f"seal() derives {field_name}")
        return super().seal(**values | cls._canonical_values(values))

    @classmethod
    def _canonical_values(cls, values):
        before_preconditions = _obligation_tuple(values.get("before_preconditions", ()))
        after_goals = _obligation_tuple(values.get("after_goals", ()))
        preservation_invariants = _obligation_tuple(
            values.get("preservation_invariants", ())
        )
        observation_obligations = _obligation_tuple(
            values.get("observation_obligations", ())
        )
        grounding_entity_sets = _finite_entity_set_tuple(
            values.get("grounding_entity_sets", ())
        )
        normalized_before = _normalize_partition(
            before_preconditions,
            BeforePrecondition,
            "before precondition partition",
        )
        normalized_after = _normalize_partition(
            after_goals,
            AfterGoal,
            "after goal partition",
        )
        normalized_preservation = _normalize_partition(
            preservation_invariants,
            PreservationInvariant,
            "preservation invariant partition",
        )
        normalized_observations = _normalize_partition(
            observation_obligations,
            ObservationObligation,
            "observation obligation partition",
        )
        normalized_entity_sets = _normalize_entity_sets(grounding_entity_sets)
        source_definition_refs = _sorted_unique_refs(
            tuple(
                reference
                for obligation in (
                    *normalized_before,
                    *normalized_after,
                    *normalized_preservation,
                    *normalized_observations,
                )
                for reference in obligation.source_definition_refs
            )
        )
        return {
            "before_preconditions": normalized_before,
            "after_goals": normalized_after,
            "preservation_invariants": normalized_preservation,
            "observation_obligations": normalized_observations,
            "grounding_entity_sets": normalized_entity_sets,
            "source_definition_refs": source_definition_refs,
            "grounding_entity_set_sha256": canonical_sha256(
                normalized_entity_sets,
                domain=_GROUNDING_ENTITY_SET_HASH_DOMAIN,
            ),
        }

    @model_validator(mode="after")
    def _validate_canonical_partition(self) -> Self:
        canonical = self._canonical_values(
            {
                "before_preconditions": self.before_preconditions,
                "after_goals": self.after_goals,
                "preservation_invariants": self.preservation_invariants,
                "observation_obligations": self.observation_obligations,
                "grounding_entity_sets": self.grounding_entity_sets,
            }
        )
        for field_name in (
            "before_preconditions",
            "after_goals",
            "preservation_invariants",
            "observation_obligations",
            "grounding_entity_sets",
            "source_definition_refs",
            "grounding_entity_set_sha256",
        ):
            if getattr(self, field_name) != canonical[field_name]:
                raise ValueError(
                    "grounded obligation set must use its canonical partition"
                )
        return self


def _canonicalize_commutative_formulas(
    formulas: tuple[_StateFormula, ...],
    label: str,
) -> tuple[_StateFormula, ...]:
    if not formulas:
        raise ValueError(f"{label} must not be empty")
    encoded = tuple(canonical_json_bytes(formula) for formula in formulas)
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate formulas")
    return tuple(
        formula
        for _encoded, formula in sorted(
            zip(encoded, formulas, strict=True), key=lambda item: item[0]
        )
    )


def _variable_ref(value: TypedValue) -> CanonicalId | None:
    if (
        isinstance(value.payload, ReferenceValue)
        and value.payload.kind is ValueKind.ENTITY_REF
        and value.payload.reference.startswith("variable:")
    ):
        return value.payload.reference
    return None


def _assert_closed_formula(
    formula: _StateFormula,
    bindings: tuple[tuple[CanonicalId, SchemaRef, TypedValue | None], ...],
) -> None:
    if isinstance(formula, PredicateAtom):
        for operand in formula.operands:
            variable_ref = _variable_ref(operand)
            if variable_ref is None:
                continue
            binding = _find_binding(variable_ref, bindings)
            if binding is None:
                raise ValueError(f"unbound variable {variable_ref!r}")
            if operand.value_schema_ref != binding[1]:
                raise ValueError(f"bound variable type mismatch for {variable_ref!r}")
        return
    if isinstance(formula, NotFormula):
        _assert_closed_formula(formula.formula, bindings)
        return
    if isinstance(formula, (AllOfFormula, AnyOfFormula, ExactlyKFormula)):
        for child in formula.formulas:
            _assert_closed_formula(child, bindings)
        return
    if isinstance(formula, ImpliesFormula):
        _assert_closed_formula(formula.antecedent, bindings)
        _assert_closed_formula(formula.consequent, bindings)
        return
    _assert_closed_formula_quantifier(formula, bindings)


def _assert_closed_context_formula(formula: _GroundedContextFormula) -> None:
    if type(formula) is bool:
        return
    _assert_closed_formula(formula, ())


def _assert_closed_formula_quantifier(
    formula: ForAllFormula | ExistsFormula,
    bindings: tuple[tuple[CanonicalId, SchemaRef, TypedValue | None], ...],
) -> None:
    if _find_binding(formula.entity_set.variable_ref, bindings) is not None:
        raise ValueError("quantifier variables must not shadow an outer binding")
    _assert_closed_formula(
        formula.formula,
        bindings
        + (
            (
                formula.entity_set.variable_ref,
                formula.entity_set.entity_schema_ref,
                None,
            ),
        ),
    )


def _find_binding(
    variable_ref: CanonicalId,
    bindings: tuple[tuple[CanonicalId, SchemaRef, TypedValue | None], ...],
) -> tuple[CanonicalId, SchemaRef, TypedValue | None] | None:
    for binding in reversed(bindings):
        if binding[0] == variable_ref:
            return binding
    return None


def _ground_formula(
    formula: _StateFormula,
    bindings: tuple[tuple[CanonicalId, SchemaRef, TypedValue | None], ...],
) -> _StateFormula | bool:
    if isinstance(formula, PredicateAtom):
        operands: list[TypedValue] = []
        for operand in formula.operands:
            variable_ref = _variable_ref(operand)
            if variable_ref is None:
                operands.append(operand)
                continue
            binding = _find_binding(variable_ref, bindings)
            if binding is None:
                raise ValueError(f"unbound variable {variable_ref!r}")
            if operand.value_schema_ref != binding[1]:
                raise ValueError(f"bound variable type mismatch for {variable_ref!r}")
            if binding[2] is None:
                raise ValueError(f"unbound variable {variable_ref!r}")
            operands.append(binding[2])
        return formula.model_copy(update={"operands": tuple(operands)})
    if isinstance(formula, NotFormula):
        grounded = _ground_formula(formula.formula, bindings)
        return not grounded if type(grounded) is bool else NotFormula(formula=grounded)
    if isinstance(formula, AllOfFormula):
        return _ground_all(
            tuple(_ground_formula(child, bindings) for child in formula.formulas)
        )
    if isinstance(formula, AnyOfFormula):
        return _ground_any(
            tuple(_ground_formula(child, bindings) for child in formula.formulas)
        )
    if isinstance(formula, ImpliesFormula):
        antecedent = _ground_formula(formula.antecedent, bindings)
        consequent = _ground_formula(formula.consequent, bindings)
        if antecedent is False or consequent is True:
            return True
        if antecedent is True:
            return consequent
        if consequent is False:
            return NotFormula(formula=antecedent)
        return ImpliesFormula(antecedent=antecedent, consequent=consequent)
    if isinstance(formula, ExactlyKFormula):
        return _ground_exactly_k(formula, bindings)
    if isinstance(formula, ForAllFormula):
        return _ground_all(
            tuple(
                _ground_formula(
                    formula.formula,
                    bindings
                    + (
                        (
                            formula.entity_set.variable_ref,
                            formula.entity_set.entity_schema_ref,
                            entity,
                        ),
                    ),
                )
                for entity in formula.entity_set.entities
            )
        )
    return _ground_any(
        tuple(
            _ground_formula(
                formula.formula,
                bindings
                + (
                    (
                        formula.entity_set.variable_ref,
                        formula.entity_set.entity_schema_ref,
                        entity,
                    ),
                ),
            )
            for entity in formula.entity_set.entities
        )
    )


def _ground_all(values: tuple[_StateFormula | bool, ...]) -> _StateFormula | bool:
    if any(value is False for value in values):
        return False
    formulas = _unique_grounded_formulas(
        tuple(value for value in values if type(value) is not bool)
    )
    if not formulas:
        return True
    if len(formulas) == 1:
        return formulas[0]
    return AllOfFormula(formulas=formulas)


def _ground_any(values: tuple[_StateFormula | bool, ...]) -> _StateFormula | bool:
    if any(value is True for value in values):
        return True
    formulas = _unique_grounded_formulas(
        tuple(value for value in values if type(value) is not bool)
    )
    if not formulas:
        return False
    if len(formulas) == 1:
        return formulas[0]
    return AnyOfFormula(formulas=formulas)


def _ground_exactly_k(
    formula: ExactlyKFormula,
    bindings: tuple[tuple[CanonicalId, SchemaRef, TypedValue | None], ...],
) -> _StateFormula | bool:
    values = tuple(_ground_formula(child, bindings) for child in formula.formulas)
    true_count = sum(value is True for value in values)
    formulas = tuple(value for value in values if type(value) is not bool)
    required_k = formula.k - true_count
    if required_k < 0 or required_k > len(formulas):
        return False
    if not formulas:
        return required_k == 0
    grouped_formulas = _group_grounded_formulas(formulas)
    if all(multiplicity == 1 for _formula, multiplicity in grouped_formulas):
        return ExactlyKFormula(
            k=required_k,
            formulas=tuple(formula for formula, _multiplicity in grouped_formulas),
        )
    return _lower_weighted_exactly_k(grouped_formulas, required_k)


def _unique_grounded_formulas(
    formulas: tuple[_StateFormula, ...],
) -> tuple[_StateFormula, ...]:
    return tuple(
        formula for formula, _multiplicity in _group_grounded_formulas(formulas)
    )


def _group_grounded_formulas(
    formulas: tuple[_StateFormula, ...],
) -> tuple[tuple[_StateFormula, int], ...]:
    grouped: list[tuple[_StateFormula, int]] = []
    last_encoded: bytes | None = None
    for encoded, formula in sorted(
        ((canonical_json_bytes(formula), formula) for formula in formulas),
        key=lambda item: item[0],
    ):
        if encoded == last_encoded:
            previous_formula, multiplicity = grouped[-1]
            grouped[-1] = (previous_formula, multiplicity + 1)
        else:
            grouped.append((formula, 1))
            last_encoded = encoded
    return tuple(grouped)


def _lower_weighted_exactly_k(
    grouped_formulas: tuple[tuple[_StateFormula, int], ...],
    required_k: int,
) -> _StateFormula | bool:
    clauses: list[_StateFormula | bool] = []
    for assignment in range(1 << len(grouped_formulas)):
        weighted_total = sum(
            multiplicity
            for index, (_formula, multiplicity) in enumerate(grouped_formulas)
            if assignment & (1 << index)
        )
        if weighted_total != required_k:
            continue
        clauses.append(
            _ground_all(
                tuple(
                    formula
                    if assignment & (1 << index)
                    else NotFormula(formula=formula)
                    for index, (formula, _multiplicity) in enumerate(grouped_formulas)
                )
            )
        )
    return _ground_any(tuple(clauses))


def _context_formulas(
    context: _ContextWrapper,
) -> tuple[_GroundedContextFormula, ...]:
    if isinstance(context, PreservationInvariant):
        return (context.before_formula, context.after_formula)
    return (context.formula,)


def _contains_quantifier(formulas: tuple[_GroundedContextFormula, ...]) -> bool:
    for formula in formulas:
        if type(formula) is bool:
            continue
        if isinstance(formula, (ForAllFormula, ExistsFormula)):
            return True
        if isinstance(formula, NotFormula) and _contains_quantifier((formula.formula,)):
            return True
        if isinstance(formula, (AllOfFormula, AnyOfFormula, ExactlyKFormula)) and (
            _contains_quantifier(formula.formulas)
        ):
            return True
        if isinstance(formula, ImpliesFormula) and _contains_quantifier(
            (formula.antecedent, formula.consequent)
        ):
            return True
    return False


def _obligation_tuple(value) -> tuple[GroundedObligation, ...]:
    return TypeAdapter(tuple[GroundedObligation, ...]).validate_python(
        value, strict=True
    )


def _finite_entity_set_tuple(value) -> tuple[FiniteEntitySet, ...]:
    return TypeAdapter(tuple[FiniteEntitySet, ...]).validate_python(value, strict=True)


def _normalize_partition(
    obligations: tuple[GroundedObligation, ...],
    wrapper_type: type[CanonicalModel],
    label: str,
) -> tuple[GroundedObligation, ...]:
    for obligation in obligations:
        if not isinstance(obligation.context, wrapper_type):
            raise TypeError(f"{label} contains an obligation in the wrong partition")
    ordered = tuple(
        sorted(obligations, key=lambda obligation: obligation._semantic_bytes)
    )
    normalized: list[GroundedObligation] = []
    for obligation in ordered:
        if normalized and normalized[-1]._semantic_bytes == obligation._semantic_bytes:
            merged_refs = _sorted_unique_refs(
                normalized[-1].source_definition_refs
                + obligation.source_definition_refs
            )
            normalized[-1] = GroundedObligation(
                context=normalized[-1].context,
                source_definition_refs=merged_refs,
            )
        else:
            normalized.append(obligation)
    return tuple(normalized)


def _normalize_entity_sets(
    entity_sets: tuple[FiniteEntitySet, ...],
) -> tuple[FiniteEntitySet, ...]:
    encoded = tuple(canonical_json_bytes(entity_set) for entity_set in entity_sets)
    if len(set(encoded)) != len(encoded):
        raise ValueError("grounding entity sets must not contain duplicate domains")
    return tuple(
        entity_set
        for _encoded, entity_set in sorted(
            zip(encoded, entity_sets, strict=True), key=lambda item: item[0]
        )
    )


def _sorted_unique_refs(
    references: tuple[DefinitionRef, ...],
) -> tuple[DefinitionRef, ...]:
    return tuple(sorted(set(references), key=canonical_json_bytes))


PredicateAtom.model_rebuild()
NotFormula.model_rebuild()
AllOfFormula.model_rebuild()
AnyOfFormula.model_rebuild()
ImpliesFormula.model_rebuild()
ExactlyKFormula.model_rebuild()
ForAllFormula.model_rebuild()
ExistsFormula.model_rebuild()
BeforePrecondition.model_rebuild()
AfterGoal.model_rebuild()
PreservationInvariant.model_rebuild()
ObservationObligation.model_rebuild()
GroundedObligation.model_rebuild()
GroundedObligationSet.model_rebuild()
