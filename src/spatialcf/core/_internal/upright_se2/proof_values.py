"""Upright compiler proof values; explicit pure implementation owner."""

from __future__ import annotations

from dataclasses import (
    fields,
    is_dataclass,
)

from enum import (
    StrEnum,
)

from fractions import (
    Fraction,
)

from spatialcf.domain import (
    upright_se2 as upright,
)

from spatialcf.domain.base import (
    CanonicalModel,
)

from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalIdValue,
    EnumSymbolValue,
    FiniteOrderedTupleValue,
    FiniteRealValue,
    IntegerValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.outcomes import (
    ResourceUsage,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.core._internal.upright_se2.constants import (
    _RETAINED_OWNER_BOUND_SCHEMA_REF,
)


def build_upright_se2_retained_owner_evaluation(
    *,
    compiled_cell: upright.UprightSE2CompiledCell,
    owner_ref: str,
    evaluator_capability_ref: str,
    outcome_kind: upright.UprightSE2RetainedOwnerOutcomeKind,
    raw_proof_rows: tuple[str, ...],
    raw_findings: tuple[str, ...],
    atomic_steps: int,
    resource_delta: ResourceUsage,
    label: str,
    exact_bound_value: object | None = None,
    additional_exact_bounds: tuple[TypedValue, ...] = (),
    continuous_exact_rationals: bool = False,
) -> upright.UprightSE2RetainedOwnerEvaluation:
    """Serialize one already-produced retained-owner DTO into proof rows.

    This intentionally performs no scene reconstruction, kernel invocation,
    search, checker dispatch, certificate work, or terminal assembly.  Both
    proposal and fresh-checker owners feed their retained DTO fields through
    this one canonical proof-row transport seam.
    """

    if (
        type(compiled_cell) is not upright.UprightSE2CompiledCell
        or type(owner_ref) is not str
        or not owner_ref
        or type(evaluator_capability_ref) is not str
        or not evaluator_capability_ref
        or type(outcome_kind) is not upright.UprightSE2RetainedOwnerOutcomeKind
        or type(raw_proof_rows) is not tuple
        or any(type(row) is not str for row in raw_proof_rows)
        or type(raw_findings) is not tuple
        or any(type(finding) is not str for finding in raw_findings)
        or type(atomic_steps) is not int
        or atomic_steps < 0
        or type(resource_delta) is not ResourceUsage
        or type(label) is not str
        or not label
        or type(additional_exact_bounds) is not tuple
        or any(type(value) is not TypedValue for value in additional_exact_bounds)
        or type(continuous_exact_rationals) is not bool
    ):
        raise TypeError("retained owner proof transport inputs must be exact")
    if outcome_kind is upright.UprightSE2RetainedOwnerOutcomeKind.EXACT:
        if exact_bound_value is None:
            raise ValueError("exact retained owner outcome requires bounds")
        bound_node = (
            _typed_continuous_bound_node
            if continuous_exact_rationals
            else _typed_bound_node
        )
        exact_bounds = tuple(
            sorted(
                (bound_node(exact_bound_value), *additional_exact_bounds),
                key=canonical_json_bytes,
            )
        )
        finding_codes: tuple[str, ...] = ()
    else:
        exact_bounds = ()
        digest = canonical_sha256(
            (label, outcome_kind.value, raw_findings),
            domain="spatialcf/counterfactual/upright-se2/proof-finding/3.0",
        )
        finding_codes = (
            "finding:spatialcf/upright-se2/proof-transport/"
            + f"{outcome_kind.value.lower()}/{digest}",
        )
    proof_rows = tuple(
        sorted(
            {
                _canonical_proof_row(label, row)
                for row in (*raw_proof_rows, f"proof:spatialcf/upright-se2/{label}")
            },
            key=canonical_json_bytes,
        )
    )
    return upright.UprightSE2RetainedOwnerEvaluation.seal(
        compiled_cell=compiled_cell,
        owner_ref=owner_ref,
        evaluator_capability_ref=evaluator_capability_ref,
        outcome_kind=outcome_kind,
        exact_bounds=exact_bounds,
        finding_codes=finding_codes,
        proof_rows=proof_rows,
        resource_delta=resource_delta,
    )


def _typed_bound_node(value: object) -> TypedValue:
    """Encode retained-owner bounds without interpreting their geometry."""

    return _typed_bound_node_with_rational_mode(value, structural_rationals=False)


def _typed_continuous_bound_node(value: object) -> TypedValue:
    """Encode continuous retained bounds without length-limiting exact rationals."""

    return _typed_bound_node_with_rational_mode(value, structural_rationals=True)


def _typed_bound_node_with_rational_mode(
    value: object,
    *,
    structural_rationals: bool,
) -> TypedValue:
    """Transport retained DTOs while preserving the cardinal scalar wire by default."""

    if isinstance(value, StrEnum):
        payload = CanonicalIdValue(value=value.value)
    elif type(value) is Fraction:
        fraction_id = f"exact-rational:{value.numerator}/{value.denominator}"
        if structural_rationals and len(fraction_id) > 512:
            payload = RecordValue(
                fields=tuple(
                    sorted(
                        (
                            NamedTypedValue(
                                name="denominator",
                                value=TypedValue(
                                    value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
                                    payload=IntegerValue(value=value.denominator),
                                ),
                            ),
                            NamedTypedValue(
                                name="numerator",
                                value=TypedValue(
                                    value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
                                    payload=IntegerValue(value=value.numerator),
                                ),
                            ),
                        ),
                        key=lambda field: canonical_json_bytes(field.name),
                    )
                )
            )
        else:
            payload = CanonicalIdValue(value=fraction_id)
    elif type(value) is bool:
        payload = BooleanValue(value=value)
    elif type(value) is int:
        payload = IntegerValue(value=value)
    elif type(value) is float:
        payload = FiniteRealValue(value=0.0 if value == 0.0 else value)
    elif isinstance(value, str):
        payload = CanonicalIdValue(value=value)
    elif value is None:
        payload = EnumSymbolValue(symbol="NULL")
    elif type(value) in (tuple, list):
        payload = FiniteOrderedTupleValue(
            element_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF,
            items=tuple(
                _typed_bound_node_with_rational_mode(
                    item, structural_rationals=structural_rationals
                )
                for item in value
            ),
        )
    elif type(value) is dict:
        if any(type(name) is not str for name in value):
            raise TypeError("retained bound records require string field names")
        payload = RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=name,
                            value=_typed_bound_node_with_rational_mode(
                                item, structural_rationals=structural_rationals
                            ),
                        )
                        for name, item in value.items()
                    ),
                    key=canonical_json_bytes,
                )
            )
        )
    elif isinstance(value, CanonicalModel):
        return _typed_bound_node_with_rational_mode(
            value.model_dump(mode="python", round_trip=True),
            structural_rationals=structural_rationals,
        )
    elif is_dataclass(value):
        payload = RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=field.name,
                            value=_typed_bound_node_with_rational_mode(
                                getattr(value, field.name),
                                structural_rationals=structural_rationals,
                            ),
                        )
                        for field in fields(value)
                    ),
                    key=canonical_json_bytes,
                )
            )
        )
    else:
        raise TypeError(f"unsupported retained bound value {type(value).__name__}")
    return TypedValue(
        value_schema_ref=_RETAINED_OWNER_BOUND_SCHEMA_REF, payload=payload
    )


def _canonical_proof_row(label: str, raw: str) -> str:
    if raw and not any(character.isspace() for character in raw):
        return raw
    digest = canonical_sha256(
        (label, raw),
        domain="spatialcf/counterfactual/upright-se2/proof-row/3.0",
    )
    return f"proof:spatialcf/upright-se2/{label}/{digest}"


# Preserve supported public type/function and pickle lookup.
build_upright_se2_retained_owner_evaluation.__module__ = "spatialcf.core.upright_se2_compiler"
_typed_bound_node.__module__ = "spatialcf.core.upright_se2_compiler"
_typed_continuous_bound_node.__module__ = "spatialcf.core.upright_se2_compiler"
_typed_bound_node_with_rational_mode.__module__ = "spatialcf.core.upright_se2_compiler"
_canonical_proof_row.__module__ = "spatialcf.core.upright_se2_compiler"
