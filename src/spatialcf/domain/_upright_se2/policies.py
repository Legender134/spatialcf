"""Upright profile policies contracts and intrinsic operations."""

from __future__ import annotations

import math

from typing import (
    ClassVar,
    Literal,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    FiniteFloat,
    Sha256Digest,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
)

from spatialcf.domain.definitions import (
    CanonicalIdValue,
    CapabilityRef,
    DefinitionRef,
    DigestValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntegerValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.profiles import (
    OwnerRef,
    ResourcePolicy,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_BACKEND_OWNER_REF,
    UPRIGHT_SE2_CHECKER_OWNER_REF,
    UPRIGHT_SE2_COLLISION_PREDICATE_REF,
    UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
    UPRIGHT_SE2_PRESERVATION_PREDICATE_REF,
    UPRIGHT_SE2_SUPPORT_PREDICATE_REF,
    UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF,
    UPRIGHT_SE2_VISIBILITY_PREDICATE_REF,
    _EXECUTABLE_POLICY_BUNDLE_FACT_KEY,
    _EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF,
    _EXECUTABLE_POLICY_FAMILY_REF,
    _EXECUTABLE_POLICY_KEYS,
    _EXECUTABLE_POLICY_SCHEMA_BY_KEY,
    _EXECUTABLE_POLICY_WIRE_SCHEMA_REF,
    _SEMANTIC_ID_SCHEMA_REF,
    _SEMANTIC_REAL_SCHEMA_REF,
    _SEMANTIC_SYMBOL_SCHEMA_REF,
    _UPRIGHT_SE2_ID_SCHEMA_REF,
    _UPRIGHT_SE2_REAL_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.values import (
    _semantic_capabilities_for_definition,
    _semantic_definition_body,
    _semantic_definition_ref_for_kind,
    _semantic_definition_schema_for_kind,
    _semantic_typed_digest,
    _semantic_typed_id,
    _semantic_typed_record,
)


class UprightSE2SemanticOwnerBinding(CanonicalModel):
    """The static evaluator/verifier ownership for one semantic definition."""

    definition_ref: DefinitionRef
    evaluator_capability_ref: CapabilityRef
    verifier_capability_ref: CapabilityRef
    evaluator_owner_ref: OwnerRef
    verifier_owner_ref: OwnerRef
    evaluator_build_sha256: Sha256Digest
    verifier_build_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_static_owner_binding(self) -> Self:
        if self.evaluator_owner_ref != UPRIGHT_SE2_BACKEND_OWNER_REF:
            raise ValueError("semantic evaluator owner must be the upright backend")
        if self.verifier_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF:
            raise ValueError("semantic verifier owner must be the upright checker")
        if self.evaluator_build_sha256 != "b" * 64:
            raise ValueError("semantic evaluator build must be fixed")
        if self.verifier_build_sha256 != "a" * 64:
            raise ValueError("semantic verifier build must be fixed")
        expected_capabilities = _semantic_capabilities_for_definition(
            self.definition_ref
        )
        if (
            self.evaluator_capability_ref,
            self.verifier_capability_ref,
        ) != expected_capabilities:
            raise ValueError("semantic evaluator/verifier capabilities must be fixed")
        return self


class UprightSE2SemanticDefinition(HashBoundCanonicalModel):
    """One typed, versioned predicate body with its executable ownership."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/semantic-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_definition_sha256"

    definition_ref: DefinitionRef
    definition_kind: Literal[
        "COLLISION",
        "SUPPORT",
        "TARGET_RELATION",
        "PRESERVATION",
        "VISIBILITY",
    ]
    payload_schema_ref: CanonicalId
    semantic_body: TypedValue
    owner_binding: UprightSE2SemanticOwnerBinding
    semantic_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_concrete_semantic_body(self) -> Self:
        expected_ref = _semantic_definition_ref_for_kind(self.definition_kind)
        if self.definition_ref != expected_ref:
            raise ValueError(
                "semantic definition reference does not match its body kind"
            )
        expected_schema = _semantic_definition_schema_for_kind(self.definition_kind)
        if self.payload_schema_ref != expected_schema:
            raise ValueError("semantic definition schema does not match its body kind")
        if self.owner_binding.definition_ref != self.definition_ref:
            raise ValueError("semantic owner binding must bind its semantic definition")
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.definition_ref)
        ):
            raise ValueError(
                "semantic definition owner binding does not match the frozen policy"
            )
        expected_body = _semantic_definition_body(self.definition_kind)
        if canonical_json_bytes(self.semantic_body) != canonical_json_bytes(
            expected_body
        ):
            raise ValueError(
                "semantic definition body does not match the frozen upright policy"
            )
        return self


class UprightSE2ObjectiveTermPolicy(CanonicalModel):
    """One concrete term of the fixed five-term M3 objective."""

    term_id: Literal["T", "A", "R", "V", "S"]
    objective_definition_ref: DefinitionRef
    metric_definition_ref: DefinitionRef
    input_selector_definition_ref: DefinitionRef
    unit_ref: DefinitionRef
    normalization_definition_ref: DefinitionRef
    normalizer_unit_ref: DefinitionRef
    weight: FiniteFloat
    normalizer: FiniteFloat

    @model_validator(mode="after")
    def _validate_term_policy(self) -> Self:
        expected = _objective_term_policy_values(self.term_id)
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    "objective term policy does not match the frozen upright policy"
                )
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("objective weights must be finite and non-negative")
        if self.term_id in ("T", "A") and self.weight <= 0.0:
            raise ValueError(
                "translation and angular objective weights must be positive"
            )
        if not math.isfinite(self.normalizer) or self.normalizer <= 0.0:
            raise ValueError(
                "objective normalizers must be finite and strictly positive"
            )
        if self.normalizer_unit_ref != self.unit_ref:
            raise ValueError("objective normalizer units must match term units")
        return self


class UprightSE2FiveTermObjectivePolicy(HashBoundCanonicalModel):
    """The complete non-ambient objective, comparison, tie, and prune policy."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/five-term-objective-policy/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "five_term_objective_policy_sha256"

    objective_definition_ref: DefinitionRef
    aggregation_definition_ref: DefinitionRef
    safety_penalty_definition_ref: DefinitionRef
    strict_interval_comparator_ref: DefinitionRef
    exact_equality_definition_ref: DefinitionRef
    directed_gap_subtraction_definition_ref: DefinitionRef
    exact_prune_comparator_ref: DefinitionRef
    gap_prune_comparator_ref: DefinitionRef
    interval_boundary_policy_ref: DefinitionRef
    deterministic_tie_break_definition_ref: DefinitionRef
    terms: tuple[UprightSE2ObjectiveTermPolicy, ...]
    owner_binding: UprightSE2SemanticOwnerBinding
    five_term_objective_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_objective_policy(self) -> Self:
        expected_refs = _objective_policy_refs()
        for name, expected_value in expected_refs.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    "objective policy reference does not match the frozen upright policy"
                )
        if tuple(term.term_id for term in self.terms) != ("T", "A", "R", "V", "S"):
            raise ValueError("objective policy must contain ordered T/A/R/V/S terms")
        if self.owner_binding.definition_ref != self.objective_definition_ref:
            raise ValueError(
                "objective owner binding must bind the objective definition"
            )
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.objective_definition_ref)
        ):
            raise ValueError("objective owner binding does not match the frozen policy")
        return self

    @property
    def weights_by_term(self) -> dict[str, float]:
        """Return the explicit weight mapping for later pure evaluators."""

        return {term.term_id: term.weight for term in self.terms}


class UprightSE2ExecutablePolicyValue(HashBoundCanonicalModel):
    """One request-owned executable policy value for a registered M3 family.

    The profile owns the permitted families, schemas, and evaluator bindings.
    This model deliberately owns none of the numeric or selector values: those
    are carried by the request-specific ``payload`` and sealed with it.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/executable-policy-value/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "executable_policy_value_sha256"

    policy_key: CanonicalId
    policy_family_ref: DefinitionRef
    definition_ref: DefinitionRef
    payload_schema_ref: CanonicalId
    owner_binding: UprightSE2SemanticOwnerBinding
    payload: TypedValue
    executable_policy_value_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_registered_request_policy(self) -> Self:
        expected = _executable_policy_spec(self.policy_key)
        if (
            self.policy_family_ref,
            self.definition_ref,
            self.payload_schema_ref,
        ) != (
            expected["policy_family_ref"],
            expected["definition_ref"],
            expected["payload_schema_ref"],
        ):
            raise ValueError("executable policy does not match a registered family")
        if canonical_json_bytes(self.owner_binding) != canonical_json_bytes(
            _semantic_owner_binding(self.definition_ref)
        ):
            raise ValueError("executable policy has an unowned evaluator binding")
        _validate_executable_policy_payload(self.policy_key, self.payload)
        return self


class UprightSE2ExecutablePolicyBundle(HashBoundCanonicalModel):
    """The complete non-ambient executable-policy bundle bound by one request."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/executable-policy-bundle/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "policy_bundle_sha256"

    profile_registration_sha256: Sha256Digest
    policies: tuple[UprightSE2ExecutablePolicyValue, ...]
    policy_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_request_policy_bundle(self) -> Self:
        keys = tuple(policy.policy_key for policy in self.policies)
        if keys != _EXECUTABLE_POLICY_KEYS:
            raise ValueError(
                "executable policy bundle must contain every policy exactly once"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("executable policy bundle must not duplicate one policy")
        return self

    def policy_for(self, policy_key: str) -> UprightSE2ExecutablePolicyValue:
        """Return one policy only after the complete roster has been sealed."""

        for policy in self.policies:
            if policy.policy_key == policy_key:
                return policy
        raise ValueError("executable policy bundle is missing the requested policy")


def _semantic_owner_binding(
    definition_ref: DefinitionRef,
) -> UprightSE2SemanticOwnerBinding:
    evaluator_capability_ref, verifier_capability_ref = (
        _semantic_capabilities_for_definition(definition_ref)
    )
    return UprightSE2SemanticOwnerBinding(
        definition_ref=definition_ref,
        evaluator_capability_ref=evaluator_capability_ref,
        verifier_capability_ref=verifier_capability_ref,
        evaluator_owner_ref=UPRIGHT_SE2_BACKEND_OWNER_REF,
        verifier_owner_ref=UPRIGHT_SE2_CHECKER_OWNER_REF,
        evaluator_build_sha256="b" * 64,
        verifier_build_sha256="a" * 64,
    )


def _executable_policy_spec(policy_key: str) -> dict[str, str]:
    """Return profile-owned family metadata for one request-owned policy key."""

    if policy_key not in _EXECUTABLE_POLICY_SCHEMA_BY_KEY:
        raise ValueError("executable policy key is unknown")
    if policy_key == "collision":
        definition_ref = UPRIGHT_SE2_COLLISION_PREDICATE_REF
    elif policy_key == "support":
        definition_ref = UPRIGHT_SE2_SUPPORT_PREDICATE_REF
    elif policy_key.startswith("relation:"):
        definition_ref = UPRIGHT_SE2_TARGET_RELATION_PREDICATE_REF
    elif policy_key == "visibility":
        definition_ref = UPRIGHT_SE2_VISIBILITY_PREDICATE_REF
    elif policy_key == "preservation":
        definition_ref = UPRIGHT_SE2_PRESERVATION_PREDICATE_REF
    else:
        # Numeric, resource, and safety determine objective/search accounting;
        # their evaluator owner is the registered objective authority.
        definition_ref = UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF
    return {
        "policy_family_ref": _EXECUTABLE_POLICY_FAMILY_REF,
        "definition_ref": definition_ref,
        "payload_schema_ref": _EXECUTABLE_POLICY_SCHEMA_BY_KEY[policy_key],
    }


def _policy_payload_fields(
    policy_key: str,
    payload: TypedValue,
) -> dict[str, TypedValue]:
    expected_schema = _executable_policy_spec(policy_key)["payload_schema_ref"]
    if (
        payload.value_schema_ref != expected_schema
        or type(payload.payload) is not RecordValue
    ):
        raise ValueError(
            "executable policy payload must use its registered record schema"
        )
    fields = {field.name: field.value for field in payload.payload.fields}
    if len(fields) != len(payload.payload.fields):
        raise ValueError("executable policy payload contains duplicate fields")
    return fields


def _require_policy_fields(
    policy_key: str,
    fields: dict[str, TypedValue],
    expected: tuple[str, ...],
) -> None:
    if set(fields) != set(expected):
        raise ValueError(
            f"executable policy {policy_key!r} fields are incomplete or unknown"
        )


def _policy_real(fields: dict[str, TypedValue], name: str) -> float:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (_SEMANTIC_REAL_SCHEMA_REF, _UPRIGHT_SE2_REAL_SCHEMA_REF)
        or type(value.payload) is not FiniteRealValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a finite real")
    return value.payload.value


def _policy_integer(fields: dict[str, TypedValue], name: str) -> int:
    value = fields[name]
    if (
        value.value_schema_ref != "schema:spatialcf/upright-se2/integer/1.0"
        or type(value.payload) is not IntegerValue
    ):
        raise ValueError(f"executable policy field {name!r} must be an integer")
    return value.payload.value


def _policy_symbol(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (
            _SEMANTIC_SYMBOL_SCHEMA_REF,
            "schema:spatialcf/upright-se2/enum-symbol/1.0",
        )
        or type(value.payload) is not EnumSymbolValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a symbol")
    return value.payload.symbol


def _policy_id(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref
        not in (_SEMANTIC_ID_SCHEMA_REF, _UPRIGHT_SE2_ID_SCHEMA_REF)
        or type(value.payload) is not CanonicalIdValue
    ):
        raise ValueError(f"executable policy field {name!r} must be an identifier")
    return value.payload.value


def _policy_digest(fields: dict[str, TypedValue], name: str) -> str:
    value = fields[name]
    if (
        value.value_schema_ref != "schema:spatialcf/upright-se2/digest/1.0"
        or type(value.payload) is not DigestValue
    ):
        raise ValueError(f"executable policy field {name!r} must be a digest")
    return value.payload.value


def _policy_tuple(fields: dict[str, TypedValue], name: str) -> FiniteOrderedTupleValue:
    value = fields[name]
    if type(value.payload) is not FiniteOrderedTupleValue:
        raise ValueError(f"executable policy field {name!r} must be an ordered tuple")
    return value.payload


def _policy_id_tuple(fields: dict[str, TypedValue], name: str) -> tuple[str, ...]:
    value = fields[name]
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/resource-limit-values/1.0"
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/policy-item/1.0"
    ):
        raise ValueError(
            f"executable policy field {name!r} must be an exact identifier tuple"
        )
    identifiers: list[str] = []
    for item in value.payload.items:
        if (
            item.value_schema_ref
            not in (_SEMANTIC_ID_SCHEMA_REF, _UPRIGHT_SE2_ID_SCHEMA_REF)
            or type(item.payload) is not CanonicalIdValue
        ):
            raise ValueError(
                f"executable policy field {name!r} must be an exact identifier tuple"
            )
        identifiers.append(item.payload.value)
    return tuple(identifiers)


def _visibility_observation_bound_fields(value: TypedValue) -> dict[str, TypedValue]:
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
        or type(value.payload) is not RecordValue
    ):
        raise ValueError(
            "executable visibility observation bounds must use typed bound records"
        )
    fields = {field.name: field.value for field in value.payload.fields}
    if len(fields) != len(value.payload.fields) or set(fields) != {
        "observation_id",
        "camera_id",
        "object_id",
        "metric_definition_id",
        "metric_definition_version",
        "comparator",
        "boundary_policy",
        "threshold",
        "tolerance",
    }:
        raise ValueError(
            "executable visibility observation bound fields are incomplete or unknown"
        )
    return fields


def _validate_visibility_observation_bounds(fields: dict[str, TypedValue]) -> None:
    value = fields["observation_bounds"]
    if (
        value.value_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bounds/1.0"
        or type(value.payload) is not FiniteOrderedTupleValue
        or value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/visibility-observation-bound/1.0"
    ):
        raise ValueError(
            "executable visibility policy must bind an exact typed observation roster"
        )
    observation_ids: list[str] = []
    for item in value.payload.items:
        bound = _visibility_observation_bound_fields(item)
        observation_ids.append(_policy_id(bound, "observation_id"))
        _policy_id(bound, "camera_id")
        _policy_id(bound, "object_id")
        metric_pair = (
            _policy_symbol(bound, "metric_definition_id"),
            _policy_id(bound, "metric_definition_version"),
        )
        if metric_pair not in {
            ("visibility:image-area-fraction", "definition:1"),
            ("visibility:truncated-fraction", "definition:1"),
            ("visibility:visible-surface-fraction", "definition:1"),
        }:
            raise ValueError("executable visibility observation metric is unsupported")
        for name, expected in (
            ("comparator", _policy_symbol(fields, "comparator")),
            ("boundary_policy", _policy_symbol(fields, "boundary_policy")),
        ):
            if _policy_symbol(bound, name) != expected:
                raise ValueError(
                    f"executable visibility observation bound {name!r} is inconsistent"
                )
        for name in ("threshold", "tolerance"):
            bound_value = _policy_real(bound, name)
            if not 0.0 <= bound_value <= 1.0:
                raise ValueError(
                    f"executable visibility observation bound {name} must be in [0, 1]"
                )
    if not observation_ids or len(set(observation_ids)) != len(observation_ids):
        raise ValueError(
            "executable visibility observation bounds must be nonempty and unique"
        )
    if tuple(observation_ids) != tuple(
        sorted(observation_ids, key=canonical_json_bytes)
    ):
        raise ValueError("executable visibility observation bounds must be ordered")


def _decode_upright_se2_objective_policy_payload(
    payload: TypedValue,
) -> UprightSE2FiveTermObjectivePolicy:
    fields = _policy_payload_fields("objective", payload)
    _require_policy_fields(
        "objective",
        fields,
        (
            "aggregation_rule",
            "comparison_rule",
            "objective_policy_sha256",
            "terms",
            "tie_break_rule",
        ),
    )
    _require_policy_values(
        fields,
        {
            "aggregation_rule": "WEIGHTED_NORMALIZED_SUM",
            "comparison_rule": "INTERVAL_LEXICOGRAPHIC",
            "tie_break_rule": "T_R_V_S_A",
        },
    )
    terms_value = fields["terms"]
    if (
        terms_value.value_schema_ref
        != "schema:spatialcf/upright-se2/objective-term-values/1.0"
        or type(terms_value.payload) is not FiniteOrderedTupleValue
        or terms_value.payload.element_schema_ref
        != "schema:spatialcf/upright-se2/objective-term-value/1.0"
    ):
        raise ValueError(
            "executable objective policy terms must use the exact typed roster"
        )
    terms: list[UprightSE2ObjectiveTermPolicy] = []
    for item in terms_value.payload.items:
        if (
            item.value_schema_ref
            != "schema:spatialcf/upright-se2/objective-term-value/1.0"
            or type(item.payload) is not RecordValue
        ):
            raise ValueError(
                "executable objective policy term must use its typed record"
            )
        term_fields = {field.name: field.value for field in item.payload.fields}
        if len(term_fields) != len(item.payload.fields) or set(term_fields) != {
            "term_id",
            "objective_definition_ref",
            "metric_definition_ref",
            "input_selector_definition_ref",
            "unit_ref",
            "normalization_definition_ref",
            "normalizer_unit_ref",
            "weight",
            "normalizer",
        }:
            raise ValueError(
                "executable objective policy term fields are incomplete or unknown"
            )
        terms.append(
            UprightSE2ObjectiveTermPolicy(
                term_id=_policy_symbol(term_fields, "term_id"),
                objective_definition_ref=_policy_id(
                    term_fields, "objective_definition_ref"
                ),
                metric_definition_ref=_policy_id(term_fields, "metric_definition_ref"),
                input_selector_definition_ref=_policy_id(
                    term_fields, "input_selector_definition_ref"
                ),
                unit_ref=_policy_id(term_fields, "unit_ref"),
                normalization_definition_ref=_policy_id(
                    term_fields, "normalization_definition_ref"
                ),
                normalizer_unit_ref=_policy_id(term_fields, "normalizer_unit_ref"),
                weight=_policy_real(term_fields, "weight"),
                normalizer=_policy_real(term_fields, "normalizer"),
            )
        )
    policy = UprightSE2FiveTermObjectivePolicy.seal(
        **_objective_policy_refs(),
        terms=tuple(terms),
        owner_binding=_semantic_owner_binding(UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF),
    )
    if _policy_digest(fields, "objective_policy_sha256") != (
        policy.five_term_objective_policy_sha256
    ):
        raise ValueError("executable objective policy self digest does not match")
    return policy


def _validate_executable_policy_payload(policy_key: str, payload: TypedValue) -> None:
    """Validate every executable family without substituting a value or default."""

    fields = _policy_payload_fields(policy_key, payload)
    if policy_key == "collision":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "clearance_m",
                "contact_comparator",
                "obstacle_selector",
                "subject_geometry_selector",
            ),
        )
        if _policy_real(fields, "clearance_m") < 0.0:
            raise ValueError(
                "executable collision policy clearance must be non-negative"
            )
        _require_policy_values(
            fields,
            {
                "contact_comparator": "ALLOW_EQUALITY",
                "boundary_policy": "CLOSED",
            },
        )
        _require_policy_ids(
            fields,
            {
                "obstacle_selector": "selector:complete-obstacle-roster",
                "subject_geometry_selector": "selector:compound-subject-geometry",
            },
        )
        return
    if policy_key == "support":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "contact_gap_lower_m",
                "contact_gap_upper_m",
                "containment_boundary_policy",
                "containment_comparator",
                "normal_selector",
                "stability_margin_m",
                "support_frame_selector",
                "support_surface_selector",
            ),
        )
        if _policy_real(fields, "contact_gap_lower_m") > _policy_real(
            fields, "contact_gap_upper_m"
        ):
            raise ValueError(
                "executable support policy contact-gap interval is reversed"
            )
        if _policy_real(fields, "stability_margin_m") < 0.0:
            raise ValueError(
                "executable support policy stability margin must be non-negative"
            )
        _require_policy_ids(
            fields,
            {
                "normal_selector": "selector:world-positive-z",
                "support_frame_selector": "selector:world-xy",
                "support_surface_selector": "selector:assigned-support-surface",
            },
        )
        _require_policy_values(
            fields,
            {
                "containment_boundary_policy": "CLOSED",
                "containment_comparator": "CONTAINS",
            },
        )
        return
    if policy_key.startswith("relation:"):
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "comparator",
                "fixed_camera_selector",
                "frame_selector",
                "measurement",
                "operand_order",
                "relation_symbol",
                "representative_geometry",
                "threshold",
                "tolerance",
                "visibility_gate",
            ),
        )
        if _policy_symbol(fields, "relation_symbol") != policy_key.removeprefix(
            "relation:"
        ):
            raise ValueError("executable relation policy has the wrong relation symbol")
        relation = policy_key.removeprefix("relation:")
        _require_policy_values(
            fields,
            {
                "boundary_policy": "CLOSED",
                "comparator": "LE" if relation in {"LEFT", "FRONT", "NEAR"} else "GE",
                "measurement": (
                    "EXTENT_EUCLIDEAN_SEPARATION"
                    if relation in {"NEAR", "FAR"}
                    else "EXTENT_SIGNED_AXIS_GAP"
                ),
                "operand_order": "SUBJECT_THEN_REFERENCE",
                "representative_geometry": "COMPOUND_BODY",
                "visibility_gate": "NONE",
            },
        )
        _require_policy_ids(
            fields,
            {
                "fixed_camera_selector": "selector:fixed-camera",
                "frame_selector": "selector:world-xy",
            },
        )
        if _policy_real(fields, "threshold") < 0.0:
            raise ValueError(
                "executable relation policy threshold must be non-negative"
            )
        if _policy_real(fields, "tolerance") < 0.0:
            raise ValueError(
                "executable relation policy tolerance must be non-negative"
            )
        return
    if policy_key == "visibility":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "boundary_policy",
                "camera_projection_convention",
                "comparator",
                "depth_policy",
                "mask_policy",
                "observation_bounds",
                "occluder_policy",
                "projected_area_metric",
                "subject_as_occluder",
                "threshold",
                "tolerance",
            ),
        )
        expected_symbols = {
            "boundary_policy": "CLOSED",
            "camera_projection_convention": "UPRIGHT_CAMERA_V2_9",
            "comparator": "GEQ",
            "depth_policy": "NEAR_CLIPPED",
            "mask_policy": "COMPLETE_MASK",
            "occluder_policy": "COMPLETE_ROSTER",
            "projected_area_metric": "PROJECTED_AREA",
            "subject_as_occluder": "INCLUDED",
        }
        for name, expected in expected_symbols.items():
            if _policy_symbol(fields, name) != expected:
                raise ValueError(
                    f"executable visibility policy field {name!r} is unsupported"
                )
        _validate_visibility_observation_bounds(fields)
        if not _policy_tuple(fields, "observation_bounds").items:
            raise ValueError(
                "executable visibility policy must bind observation bounds"
            )
        for name in ("threshold", "tolerance"):
            value = _policy_real(fields, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"executable visibility policy {name} must be in [0, 1]"
                )
        return
    if policy_key == "numeric":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "dyadic_refinement_policy",
                "exact_number_representation",
                "numeric_semantics_ref",
                "tolerance_m",
            ),
        )
        _require_policy_ids(
            fields,
            {
                "numeric_semantics_ref": "definition:spatialcf/upright-se2/numeric-semantics/1.0"
            },
        )
        if _policy_real(fields, "tolerance_m") < 0.0:
            raise ValueError("executable numeric policy tolerance must be non-negative")
        _require_policy_values(
            fields,
            {
                "exact_number_representation": "BINARY64_BITS",
                "dyadic_refinement_policy": "EXACT_DYADIC",
            },
        )
        return
    if policy_key == "objective":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "aggregation_rule",
                "comparison_rule",
                "objective_policy_sha256",
                "terms",
                "tie_break_rule",
            ),
        )
        _decode_upright_se2_objective_policy_payload(payload)
        return
    if policy_key == "safety":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "constraint_slack_target",
                "hard_constraint_selector",
                "safety_penalty_rule",
            ),
        )
        if _policy_real(fields, "constraint_slack_target") < 0.0:
            raise ValueError(
                "executable safety policy slack target must be non-negative"
            )
        _require_policy_ids(
            fields,
            {"hard_constraint_selector": "selector:all-hard-constraints"},
        )
        _require_policy_values(
            fields,
            {"safety_penalty_rule": "FROM_CONSTRAINT_SLACK"},
        )
        return
    if policy_key == "resource":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "atomic_step_limit",
                "deterministic_order",
                "limits",
                "resource_policy_sha256",
                "shared_ledger_policy_ref",
            ),
        )
        atomic_step_limit = _policy_integer(fields, "atomic_step_limit")
        if atomic_step_limit < 0:
            raise ValueError(
                "executable resource policy atomic step limit must be non-negative"
            )
        if not _policy_id_tuple(fields, "limits"):
            raise ValueError("executable resource policy must bind every request limit")
        _policy_digest(fields, "resource_policy_sha256")
        _policy_id(fields, "shared_ledger_policy_ref")
        _require_policy_values(
            fields,
            {"deterministic_order": "LOWER_OWNED_XY"},
        )
        return
    if policy_key == "preservation":
        _require_policy_fields(
            policy_key,
            fields,
            (
                "frozen_observation_policy",
                "grounded_invariant_selector",
                "state_selector",
            ),
        )
        _require_policy_values(
            fields,
            {"frozen_observation_policy": "COMPLETE_GROUNDED"},
        )
        _require_policy_ids(
            fields,
            {
                "grounded_invariant_selector": "selector:grounded-invariants",
                "state_selector": "selector:frozen-nonprimary-state",
            },
        )
        return
    raise ValueError("executable policy key is unknown")


def _require_policy_values(
    fields: dict[str, TypedValue], expected: dict[str, str]
) -> None:
    for name, expected_value in expected.items():
        if _policy_symbol(fields, name) != expected_value:
            raise ValueError(f"executable policy field {name!r} is unsupported")


def _require_policy_ids(
    fields: dict[str, TypedValue], expected: dict[str, str]
) -> None:
    for name, expected_value in expected.items():
        if _policy_id(fields, name) != expected_value:
            raise ValueError(f"executable policy field {name!r} is unsupported")


def _executable_policy_wire(policy: UprightSE2ExecutablePolicyValue) -> TypedValue:
    return _semantic_typed_record(
        _EXECUTABLE_POLICY_WIRE_SCHEMA_REF,
        (
            ("definition_ref", _semantic_typed_id(policy.definition_ref)),
            ("payload", policy.payload),
            ("payload_schema_ref", _semantic_typed_id(policy.payload_schema_ref)),
            ("policy_family_ref", _semantic_typed_id(policy.policy_family_ref)),
            ("policy_key", _semantic_typed_id(policy.policy_key)),
        ),
    )


def executable_policy_bundle_to_typed_value(
    bundle: UprightSE2ExecutablePolicyBundle,
) -> TypedValue:
    """Encode a sealed bundle into the semantic-problem extension-fact wire."""

    return _semantic_typed_record(
        _EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF,
        (
            (
                "policies",
                TypedValue(
                    value_schema_ref=(
                        "schema:spatialcf/upright-se2/executable-policy-wire-tuple/1.0"
                    ),
                    payload=FiniteOrderedTupleValue(
                        element_schema_ref=_EXECUTABLE_POLICY_WIRE_SCHEMA_REF,
                        items=tuple(
                            _executable_policy_wire(policy)
                            for policy in bundle.policies
                        ),
                    ),
                ),
            ),
            (
                "profile_registration_sha256",
                _semantic_typed_digest(bundle.profile_registration_sha256),
            ),
        ),
    )


def _record_field_map(value: TypedValue, schema_ref: str) -> dict[str, TypedValue]:
    if value.value_schema_ref != schema_ref or type(value.payload) is not RecordValue:
        raise ValueError("executable policy wire has the wrong record schema")
    return {field.name: field.value for field in value.payload.fields}


def _wire_policy_value(value: TypedValue) -> UprightSE2ExecutablePolicyValue:
    fields = _record_field_map(value, _EXECUTABLE_POLICY_WIRE_SCHEMA_REF)
    if set(fields) != {
        "definition_ref",
        "payload",
        "payload_schema_ref",
        "policy_family_ref",
        "policy_key",
    }:
        raise ValueError("executable policy wire fields are incomplete or unknown")
    policy_key = _policy_id(fields, "policy_key")
    definition_ref = _policy_id(fields, "definition_ref")
    return UprightSE2ExecutablePolicyValue.seal(
        policy_key=policy_key,
        policy_family_ref=_policy_id(fields, "policy_family_ref"),
        definition_ref=definition_ref,
        payload_schema_ref=_policy_id(fields, "payload_schema_ref"),
        owner_binding=_semantic_owner_binding(definition_ref),
        payload=fields["payload"],
    )


def request_bound_executable_policy_bundle_from_problem(
    semantic_problem: CounterfactualProblemIR,
) -> UprightSE2ExecutablePolicyBundle:
    """Recover the request-owned bundle from the only permitted semantic wire."""

    facts = tuple(
        fact
        for bundle in semantic_problem.scene_state.extension_fact_bundles
        for fact in bundle.facts
        if fact.fact_family_ref == _EXECUTABLE_POLICY_FAMILY_REF
    )
    if len(facts) != 1:
        raise ValueError("executable policy bundle must be present exactly once")
    fact = facts[0]
    if (
        fact.fact_key != _EXECUTABLE_POLICY_BUNDLE_FACT_KEY
        or fact.subject_entity_id != "entity:upright-se2-policy"
    ):
        raise ValueError("executable policy bundle has an unbound extension-fact owner")
    fields = _record_field_map(fact.value, _EXECUTABLE_POLICY_BUNDLE_SCHEMA_REF)
    if set(fields) != {"policies", "profile_registration_sha256"}:
        raise ValueError(
            "executable policy bundle wire fields are incomplete or unknown"
        )
    profile_digest = _policy_digest(fields, "profile_registration_sha256")
    policies_value = fields["policies"]
    if (
        policies_value.value_schema_ref
        != "schema:spatialcf/upright-se2/executable-policy-wire-tuple/1.0"
        or type(policies_value.payload) is not FiniteOrderedTupleValue
        or policies_value.payload.element_schema_ref
        != _EXECUTABLE_POLICY_WIRE_SCHEMA_REF
    ):
        raise ValueError(
            "executable policy bundle must carry the registered policy tuple"
        )
    return UprightSE2ExecutablePolicyBundle.seal(
        profile_registration_sha256=profile_digest,
        policies=tuple(
            _wire_policy_value(value) for value in policies_value.payload.items
        ),
    )


def build_upright_se2_executable_policy_bundle(
    *,
    profile_registration_sha256: Sha256Digest,
    policies: tuple[UprightSE2ExecutablePolicyValue, ...],
) -> UprightSE2ExecutablePolicyBundle:
    """Seal supplied policy values without selecting any ambient values."""

    return UprightSE2ExecutablePolicyBundle.seal(
        profile_registration_sha256=profile_registration_sha256,
        policies=policies,
    )


def decode_upright_se2_objective_policy(
    bundle: UprightSE2ExecutablePolicyBundle,
) -> UprightSE2FiveTermObjectivePolicy:
    """Decode the one request-bound T/A/R/V/S evaluator policy from its wire."""

    return _decode_upright_se2_objective_policy_payload(
        bundle.policy_for("objective").payload
    )


def _validate_executable_policy_resource_binding(
    bundle: UprightSE2ExecutablePolicyBundle,
    resource_policy: ResourcePolicy,
) -> None:
    resource_fields = _policy_payload_fields(
        "resource", bundle.policy_for("resource").payload
    )
    if len(resource_policy.limits) != 1:
        raise ValueError("resource policy must bind one registered limit")
    limit = resource_policy.limits[0]
    if (
        not math.isfinite(limit.finite_limit)
        or limit.finite_limit < 0.0
        or not limit.finite_limit.is_integer()
    ):
        raise ValueError("resource policy finite limit must be a non-negative integer")
    if _policy_id_tuple(resource_fields, "limits") != (limit.definition_ref,):
        raise ValueError(
            "executable resource policy limits do not bind the source request"
        )
    if _policy_integer(resource_fields, "atomic_step_limit") != int(limit.finite_limit):
        raise ValueError(
            "executable resource policy atomic step limit does not bind the source request"
        )
    if (
        _policy_digest(resource_fields, "resource_policy_sha256")
        != resource_policy.resource_policy_sha256
    ):
        raise ValueError(
            "executable policy resource root does not bind the source request"
        )
    if (
        _policy_id(resource_fields, "shared_ledger_policy_ref")
        != resource_policy.shared_ledger_policy_ref
    ):
        raise ValueError(
            "executable policy shared ledger does not bind the source request"
        )


def _objective_term_policy_values(term_id: str) -> dict[str, object]:
    values = {
        "T": (
            "definition:spatialcf/upright-se2/objective-subject-pivot-displacement/1.0",
            "definition:spatialcf/upright-se2/objective-selector-subject-pivot-displacement/1.0",
            "definition:spatialcf/upright-se2/metre/1.0",
            "definition:spatialcf/upright-se2/normalizer-subject-pivot-displacement/1.0",
        ),
        "A": (
            "definition:spatialcf/upright-se2/objective-angular-geodesic/1.0",
            "definition:spatialcf/upright-se2/objective-selector-angular-geodesic/1.0",
            "definition:spatialcf/upright-se2/turn/1.0",
            "definition:spatialcf/upright-se2/normalizer-angular-geodesic/1.0",
        ),
        "R": (
            "definition:spatialcf/upright-se2/objective-nontarget-relation-damage/1.0",
            "definition:spatialcf/upright-se2/objective-selector-nontarget-relation-damage/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-nontarget-relation-damage/1.0",
        ),
        "V": (
            "definition:spatialcf/upright-se2/objective-visibility-change/1.0",
            "definition:spatialcf/upright-se2/objective-selector-visibility-change/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-visibility-change/1.0",
        ),
        "S": (
            "definition:spatialcf/upright-se2/objective-safety-margin-penalty/1.0",
            "definition:spatialcf/upright-se2/objective-selector-safety-margin-penalty/1.0",
            "definition:spatialcf/upright-se2/dimensionless/1.0",
            "definition:spatialcf/upright-se2/normalizer-safety-margin-penalty/1.0",
        ),
    }[term_id]
    return {
        "objective_definition_ref": UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
        "metric_definition_ref": values[0],
        "input_selector_definition_ref": values[1],
        "unit_ref": values[2],
        "normalization_definition_ref": values[3],
        "normalizer_unit_ref": values[2],
    }


def _objective_policy_refs() -> dict[str, str]:
    return {
        "objective_definition_ref": UPRIGHT_SE2_OBJECTIVE_DEFINITION_REF,
        "aggregation_definition_ref": "definition:spatialcf/upright-se2/objective-weighted-normalized-sum/1.0",
        "safety_penalty_definition_ref": "definition:spatialcf/upright-se2/objective-safety-penalty-from-hard-constraints/1.0",
        "strict_interval_comparator_ref": "definition:spatialcf/upright-se2/objective-strict-upper-lower-comparator/1.0",
        "exact_equality_definition_ref": "definition:spatialcf/upright-se2/objective-degenerate-exact-equality/1.0",
        "directed_gap_subtraction_definition_ref": "definition:spatialcf/upright-se2/objective-directed-gap-subtraction/1.0",
        "exact_prune_comparator_ref": "definition:spatialcf/upright-se2/objective-exact-strict-prune/1.0",
        "gap_prune_comparator_ref": "definition:spatialcf/upright-se2/objective-gap-closed-prune/1.0",
        "interval_boundary_policy_ref": "definition:spatialcf/upright-se2/objective-interval-boundary-policy/1.0",
        "deterministic_tie_break_definition_ref": "definition:spatialcf/upright-se2/objective-tie-break-T-R-V-S-A/1.0",
    }


# Resolve local model forward references before restoring public identities.
UprightSE2SemanticOwnerBinding.model_rebuild()
UprightSE2SemanticDefinition.model_rebuild()
UprightSE2ObjectiveTermPolicy.model_rebuild()
UprightSE2FiveTermObjectivePolicy.model_rebuild()
UprightSE2ExecutablePolicyValue.model_rebuild()
UprightSE2ExecutablePolicyBundle.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2SemanticOwnerBinding.__module__ = "spatialcf.domain.upright_se2"
UprightSE2SemanticDefinition.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ObjectiveTermPolicy.__module__ = "spatialcf.domain.upright_se2"
UprightSE2FiveTermObjectivePolicy.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ExecutablePolicyValue.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ExecutablePolicyBundle.__module__ = "spatialcf.domain.upright_se2"
_semantic_owner_binding.__module__ = "spatialcf.domain.upright_se2"
_executable_policy_spec.__module__ = "spatialcf.domain.upright_se2"
_policy_payload_fields.__module__ = "spatialcf.domain.upright_se2"
_require_policy_fields.__module__ = "spatialcf.domain.upright_se2"
_policy_real.__module__ = "spatialcf.domain.upright_se2"
_policy_integer.__module__ = "spatialcf.domain.upright_se2"
_policy_symbol.__module__ = "spatialcf.domain.upright_se2"
_policy_id.__module__ = "spatialcf.domain.upright_se2"
_policy_digest.__module__ = "spatialcf.domain.upright_se2"
_policy_tuple.__module__ = "spatialcf.domain.upright_se2"
_policy_id_tuple.__module__ = "spatialcf.domain.upright_se2"
_visibility_observation_bound_fields.__module__ = "spatialcf.domain.upright_se2"
_validate_visibility_observation_bounds.__module__ = "spatialcf.domain.upright_se2"
_decode_upright_se2_objective_policy_payload.__module__ = "spatialcf.domain.upright_se2"
_validate_executable_policy_payload.__module__ = "spatialcf.domain.upright_se2"
_require_policy_values.__module__ = "spatialcf.domain.upright_se2"
_require_policy_ids.__module__ = "spatialcf.domain.upright_se2"
_executable_policy_wire.__module__ = "spatialcf.domain.upright_se2"
executable_policy_bundle_to_typed_value.__module__ = "spatialcf.domain.upright_se2"
_record_field_map.__module__ = "spatialcf.domain.upright_se2"
_wire_policy_value.__module__ = "spatialcf.domain.upright_se2"
request_bound_executable_policy_bundle_from_problem.__module__ = "spatialcf.domain.upright_se2"
build_upright_se2_executable_policy_bundle.__module__ = "spatialcf.domain.upright_se2"
decode_upright_se2_objective_policy.__module__ = "spatialcf.domain.upright_se2"
_validate_executable_policy_resource_binding.__module__ = "spatialcf.domain.upright_se2"
_objective_term_policy_values.__module__ = "spatialcf.domain.upright_se2"
_objective_policy_refs.__module__ = "spatialcf.domain.upright_se2"
