"""Upright profile m2 contracts and intrinsic operations."""

from __future__ import annotations

from fractions import (
    Fraction,
)

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
    Sha256Digest,
)

from spatialcf.domain.compatibility import (
    PlanarTranslateCompilation,
)

from spatialcf.domain.definitions import (
    DefinitionRef,
    EnumSymbolValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntegerValue,
    NamedTypedValue,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.profiles import (
    OwnerRef,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_COMPILER_OWNER_REF,
    _M2_Q0_DOMAIN_SCHEMA_REF,
    _M2_Q0_DOMAIN_TRANSFORM_REF,
    _M2_Q0_MAPPING_DEFINITION_ROSTER_HASH_DOMAIN,
    _M2_Q0_SHARED_VALUE_HASH_DOMAIN,
    _M2_Q0_SOURCE_PROVENANCE_HASH_DOMAIN,
    _M2_Q0_SUPPORTED_DOMAIN_REF,
    _UPRIGHT_SE2_REAL_SCHEMA_REF,
)

from spatialcf.domain._upright_se2.policies import (
    UprightSE2ExecutablePolicyBundle,
)

from spatialcf.domain._upright_se2.state import (
    UprightSE2TranslationDomain,
)

from spatialcf.domain._upright_se2.yaw import (
    ExactDyadic,
)


def _m2_q0_source_leaf_values(
    source_compilation: PlanarTranslateCompilation,
) -> dict[str, object]:
    """Return every retained M2 leaf under its canonical provenance selector."""

    values: dict[str, object] = {}

    def visit(value: object, selector: str) -> None:
        if type(value) is dict:
            if not value:
                values[selector] = value
                return
            for key in sorted(value, key=canonical_json_bytes):
                visit(value[key], f"{selector}/{key}")
            return
        if type(value) in (list, tuple):
            if not value:
                values[selector] = value
                return
            for index, item in enumerate(value):
                visit(item, f"{selector}/{index}")
            return
        values[selector] = value

    visit(
        source_compilation.model_dump(mode="python", round_trip=True),
        "source:compilation",
    )
    return values


def _m2_q0_source_leaf_selectors(
    source_compilation: PlanarTranslateCompilation,
) -> frozenset[str]:
    """Return every retained M2 leaf selector in canonical source-tree order."""

    return frozenset(_m2_q0_source_leaf_values(source_compilation))


def _m2_q0_dyadic_from_float(value: float) -> ExactDyadic:
    fraction = Fraction.from_float(0.0 if value == 0.0 else value)
    return ExactDyadic(numerator=fraction.numerator, denominator=fraction.denominator)


def _m2_q0_authorized_domain_from_source(
    source_compilation: PlanarTranslateCompilation,
) -> UprightSE2TranslationDomain:
    """Reconstruct the sole supported M2 world-XY delta domain from provenance."""

    source_problem = source_compilation.source_artifacts.problem
    constraints = source_problem.constraints
    position = constraints.position_domain
    if (
        position.region_interpretation.value != "SUBJECT_ANCHOR_LOCUS"
        or position.workspace_aggregation.value != "INTERSECTION"
        or position.boundary_policy.value != "CLOSED"
        or position.known_free_space_fact_ids
        or position.subject_occupancy_body_ids
        or position.minimum_boundary_clearance_m != 0.0
        or len(position.workspace_fact_ids) != 1
        or tuple(item.value for item in position.required_completeness) != ("EXACT",)
    ):
        raise ValueError(
            "q=0 construction source has no supported total world-XY representation"
        )
    workspace = tuple(
        item
        for item in source_problem.scene.workspace_boundaries.values or ()
        if item.fact_id == position.workspace_fact_ids[0]
    )
    if len(workspace) != 1:
        raise ValueError("q=0 construction source must bind one exact workspace fact")
    region = workspace[0].region_world_xy
    if len(region.components) != 1 or region.components[0].holes:
        raise ValueError(
            "q=0 construction source must be one closed axis-aligned rectangle"
        )
    vertices = region.components[0].exterior.vertices
    xs = tuple(sorted({point.x for point in vertices}))
    ys = tuple(sorted({point.y for point in vertices}))
    if (
        len(vertices) != 4
        or len(xs) != 2
        or len(ys) != 2
        or {(point.x, point.y) for point in vertices}
        != {(x, y) for x in xs for y in ys}
    ):
        raise ValueError(
            "q=0 construction source must be one closed axis-aligned rectangle"
        )
    subject = tuple(
        item
        for item in source_problem.scene.objects.values or ()
        if item.object_id == constraints.allowed_edit.subject_id
    )
    if len(subject) != 1:
        raise ValueError("q=0 construction source must bind one source subject pose")
    before = subject[0].pose.world_from_object.translation
    return UprightSE2TranslationDomain(
        x_lower=_m2_q0_dyadic_from_float(xs[0] - before.x),
        x_upper=_m2_q0_dyadic_from_float(xs[1] - before.x),
        y_lower=_m2_q0_dyadic_from_float(ys[0] - before.y),
        y_upper=_m2_q0_dyadic_from_float(ys[1] - before.y),
    )


def _m2_q0_domain_typed_value(domain: UprightSE2TranslationDomain) -> TypedValue:
    return TypedValue(
        value_schema_ref=(
            "schema:spatialcf/upright-se2/executable-m2-q0-domain-policy/1.0"
        ),
        payload=RecordValue(
            fields=tuple(
                sorted(
                    (
                        NamedTypedValue(
                            name=name,
                            value=TypedValue(
                                value_schema_ref=_UPRIGHT_SE2_REAL_SCHEMA_REF,
                                payload=FiniteRealValue(value=float(value.as_fraction)),
                            ),
                        )
                        for name, value in (
                            ("x_lower", domain.x_lower),
                            ("x_upper", domain.x_upper),
                            ("y_lower", domain.y_lower),
                            ("y_upper", domain.y_upper),
                        )
                    ),
                    key=canonical_json_bytes,
                )
            )
        ),
    )


def _m2_q0_non_equivalence_values(
    source_compilation: PlanarTranslateCompilation,
    policy_bundle: UprightSE2ExecutablePolicyBundle,
) -> dict[str, tuple[object, object]]:
    """Return the exact M2 and construction-bound M3 roots for each mismatch."""

    source_problem = source_compilation.source_artifacts.problem
    return {
        "source:policy/objective": (
            source_problem.objective,
            policy_bundle.policy_for("objective").payload,
        ),
        "source:policy/relation": (
            source_problem.relation_semantics,
            tuple(
                policy.payload
                for policy in policy_bundle.policies
                if policy.policy_key.startswith("relation:")
            ),
        ),
        "source:policy/visibility": (
            source_problem.visibility_semantics,
            policy_bundle.policy_for("visibility").payload,
        ),
        "source:policy/support": (
            source_problem.constraints.support_constraints,
            policy_bundle.policy_for("support").payload,
        ),
        "source:policy/numeric": (
            source_problem.numeric_policy,
            policy_bundle.policy_for("numeric").payload,
        ),
        "source:policy/resource": (
            source_compilation.source_artifacts.config,
            policy_bundle.policy_for("resource").payload,
        ),
        "source:policy/tie-break": (
            source_problem.objective.tie_break,
            policy_bundle.policy_for("objective").payload,
        ),
    }


class UprightSE2M2Q0MappingDefinition(HashBoundCanonicalModel):
    """One versioned source-to-M3 construction relation definition.

    This record deliberately describes a relation, never an evaluator.  It is
    carried as compiler-owned evidence for the narrow q=0 construction route
    and cannot alter the Task 1 profile registration or executable M3 policy.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/mapping-definition/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "mapping_definition_sha256"

    mapping_definition_ref: DefinitionRef
    row_kind: Literal["EQUALITY", "SOURCE_CONTEXT", "NON_EQUIVALENCE"]
    source_selector: CanonicalId
    target_selector: CanonicalId | None
    source_value_schema_ref: CanonicalId
    target_value_schema_ref: CanonicalId | None
    source_unit_ref: CanonicalId | None
    target_unit_ref: CanonicalId | None
    transform_ref: DefinitionRef | None
    non_equivalence_reason_ref: DefinitionRef | None
    mapping_owner_ref: OwnerRef
    mapping_version: CanonicalId
    accepted_source_domain_ref: DefinitionRef
    mapping_definition_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_relation_shape(self) -> Self:
        if self.mapping_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("q=0 mapping definitions must use the compiler owner")
        if self.mapping_version != "mapping-version:spatialcf/upright-se2/m2-q0/1":
            raise ValueError("q=0 mapping definitions must use the fixed version")
        expected_mapping_ref = (
            "definition:spatialcf/upright-se2/m2-q0/mapping/"
            f"{self.source_selector.removeprefix('source:').replace('/', '-')}/1.0"
        )
        if self.mapping_definition_ref != expected_mapping_ref:
            raise ValueError(
                "q=0 mapping definition reference must bind its source selector"
            )
        if self.accepted_source_domain_ref != _M2_Q0_SUPPORTED_DOMAIN_REF:
            raise ValueError(
                "q=0 mapping definitions must name the sole supported domain"
            )
        if self.row_kind == "EQUALITY":
            if (
                self.target_selector is None
                or self.target_value_schema_ref is None
                or self.transform_ref is None
                or self.non_equivalence_reason_ref is not None
            ):
                raise ValueError(
                    "equality mapping definition must declare one exact transform"
                )
            if (
                self.source_selector != "source:constraints/position-domain"
                or self.target_selector != "target:operation/translation-domain"
                or self.source_value_schema_ref != _M2_Q0_DOMAIN_SCHEMA_REF
                or self.target_value_schema_ref != _M2_Q0_DOMAIN_SCHEMA_REF
                or self.source_unit_ref
                != "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                or self.target_unit_ref
                != "definition:spatialcf/upright-se2/world-xy/metre/1.0"
                or self.transform_ref != _M2_Q0_DOMAIN_TRANSFORM_REF
            ):
                raise ValueError(
                    "q=0 equality must be the exact world-XY domain mapping"
                )
        elif self.row_kind == "SOURCE_CONTEXT":
            if any(
                value is not None
                for value in (
                    self.target_selector,
                    self.target_value_schema_ref,
                    self.target_unit_ref,
                    self.transform_ref,
                    self.non_equivalence_reason_ref,
                )
            ):
                raise ValueError(
                    "source-context mapping definition must not name an M3 target"
                )
            if (
                not self.source_selector.startswith("source:compilation/")
                or self.source_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/source-leaf/1.0"
                or self.source_unit_ref is not None
            ):
                raise ValueError(
                    "source-context mapping definition must bind one typed source leaf"
                )
        else:
            if (
                self.target_selector is None
                or self.target_value_schema_ref is None
                or self.non_equivalence_reason_ref is None
                or self.transform_ref is not None
            ):
                raise ValueError(
                    "non-equivalence mapping definition must name target and reason"
                )
            expected_non_equivalence = {
                "source:policy/objective": (
                    "target:policy/objective",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-objective/1.0",
                ),
                "source:policy/relation": (
                    "target:policy/relation",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-relation/1.0",
                ),
                "source:policy/visibility": (
                    "target:policy/visibility",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-visibility/1.0",
                ),
                "source:policy/support": (
                    "target:policy/support",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-support/1.0",
                ),
                "source:policy/numeric": (
                    "target:policy/numeric",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-numeric/1.0",
                ),
                "source:policy/resource": (
                    "target:policy/resource",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-resource/1.0",
                ),
                "source:policy/tie-break": (
                    "target:policy/tie-break",
                    "definition:spatialcf/upright-se2/m2-q0/heterogeneous-tie-break/1.0",
                ),
            }
            if (
                self.source_selector not in expected_non_equivalence
                or self.target_selector
                != expected_non_equivalence[self.source_selector][0]
                or self.non_equivalence_reason_ref
                != expected_non_equivalence[self.source_selector][1]
                or self.source_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/source-leaf/1.0"
                or self.target_value_schema_ref
                != "schema:spatialcf/upright-se2/m2-q0/target-leaf/1.0"
                or self.source_unit_ref is not None
                or self.target_unit_ref is not None
            ):
                raise ValueError(
                    "q=0 non-equivalence must bind its fixed heterogeneous policy"
                )
        return self


class UprightSE2M2Q0MappingRow(HashBoundCanonicalModel):
    """One fully bound q=0 relation row over exact source provenance."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/mapping-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "mapping_row_sha256"

    mapping_definition: UprightSE2M2Q0MappingDefinition
    row_kind: Literal["EQUALITY", "SOURCE_CONTEXT", "NON_EQUIVALENCE"]
    source_value_sha256: Sha256Digest
    target_value_sha256: Sha256Digest | None
    shared_value: TypedValue | None
    shared_value_sha256: Sha256Digest | None
    non_equivalence_reason_ref: DefinitionRef | None
    mapping_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_bound_relation(self) -> Self:
        if self.row_kind != self.mapping_definition.row_kind:
            raise ValueError("q=0 mapping row kind must match its definition")
        if self.row_kind == "EQUALITY":
            if (
                self.target_value_sha256 != self.source_value_sha256
                or self.shared_value is None
                or self.shared_value_sha256 is None
                or self.source_value_sha256 != self.shared_value_sha256
                or self.non_equivalence_reason_ref is not None
            ):
                raise ValueError(
                    "equality mapping row must bind one shared normalized value"
                )
            if self.shared_value_sha256 != canonical_sha256(
                self.shared_value,
                domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
            ):
                raise ValueError("equality mapping row shared-value digest is wrong")
        elif self.row_kind == "SOURCE_CONTEXT":
            if any(
                value is not None
                for value in (
                    self.target_value_sha256,
                    self.shared_value,
                    self.shared_value_sha256,
                    self.non_equivalence_reason_ref,
                )
            ):
                raise ValueError("source-context mapping row must retain source only")
        else:
            if (
                self.target_value_sha256 is None
                or self.shared_value is not None
                or self.shared_value_sha256 is not None
                or self.non_equivalence_reason_ref
                != self.mapping_definition.non_equivalence_reason_ref
            ):
                raise ValueError(
                    "non-equivalence mapping row must bind target and reason"
                )
        return self


class UprightSE2M2Q0SourceFreeConstructionRow(HashBoundCanonicalModel):
    """One target-only angular or ordering constant, never a source mapping."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/source-free-construction-row/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "source_free_construction_row_sha256"

    construction_selector: CanonicalId
    target_value: TypedValue
    target_value_sha256: Sha256Digest
    construction_owner_ref: OwnerRef
    construction_version: CanonicalId
    source_free_construction_row_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_source_free_value(self) -> Self:
        if self.construction_owner_ref != UPRIGHT_SE2_COMPILER_OWNER_REF:
            raise ValueError("source-free construction row must use the compiler owner")
        if (
            self.construction_version
            != "construction-version:spatialcf/upright-se2/m2-q0/1"
        ):
            raise ValueError("source-free construction row must use the fixed version")
        if self.target_value_sha256 != canonical_sha256(
            self.target_value,
            domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
        ):
            raise ValueError("source-free construction value digest is wrong")
        return self


class UprightSE2M2Q0Construction(HashBoundCanonicalModel):
    """The narrow domain-only bridge from one exact retained M2 compilation.

    The full source compilation is embedded verbatim as provenance.  The sole
    equality claim is the declared supported world-XY domain transform; every
    heterogeneous policy root remains source context or explicit
    non-equivalence evidence.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/m2-q0/construction/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "m2_q0_construction_sha256"

    source_compilation: PlanarTranslateCompilation
    source_compilation_sha256: Sha256Digest
    source_provenance_sha256: Sha256Digest
    authorized_domain: UprightSE2TranslationDomain
    authorized_domain_sha256: Sha256Digest
    construction_q: Literal[0] = 0
    mapping_definitions: tuple[UprightSE2M2Q0MappingDefinition, ...]
    mapping_definition_roster_sha256: Sha256Digest
    rows: tuple[UprightSE2M2Q0MappingRow, ...]
    source_free_construction_rows: tuple[UprightSE2M2Q0SourceFreeConstructionRow, ...]
    frozen_m3_policy_bundle: UprightSE2ExecutablePolicyBundle
    frozen_m3_policy_bundle_sha256: Sha256Digest
    m2_q0_construction_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_closed_construction(self) -> Self:
        if self.source_compilation_sha256 != self.source_compilation.compilation_sha256:
            raise ValueError("q=0 construction must bind its exact source compilation")
        if self.source_provenance_sha256 != canonical_sha256(
            self.source_compilation,
            domain=_M2_Q0_SOURCE_PROVENANCE_HASH_DOMAIN,
        ):
            raise ValueError("q=0 construction source provenance digest is wrong")
        if self.authorized_domain_sha256 != canonical_sha256(
            self.authorized_domain,
            domain="spatialcf/counterfactual/upright-se2/translation-domain/3.0",
        ):
            raise ValueError("q=0 construction authorized-domain digest is wrong")
        if (
            self.frozen_m3_policy_bundle_sha256
            != self.frozen_m3_policy_bundle.policy_bundle_sha256
        ):
            raise ValueError("q=0 construction must bind its executable M3 policy")
        expected_domain = _m2_q0_authorized_domain_from_source(self.source_compilation)
        if self.authorized_domain != expected_domain:
            raise ValueError(
                "q=0 construction authorized domain must derive from source provenance"
            )
        definitions = tuple(
            sorted(
                self.mapping_definitions, key=lambda item: canonical_json_bytes(item)
            )
        )
        if self.mapping_definitions != definitions:
            raise ValueError("q=0 mapping definitions must use canonical source order")
        if self.mapping_definition_roster_sha256 != canonical_sha256(
            tuple(item.mapping_definition_sha256 for item in definitions),
            domain=_M2_Q0_MAPPING_DEFINITION_ROSTER_HASH_DOMAIN,
        ):
            raise ValueError("q=0 mapping-definition roster digest is wrong")
        definition_bytes = {canonical_json_bytes(item) for item in definitions}
        if any(
            canonical_json_bytes(row.mapping_definition) not in definition_bytes
            for row in self.rows
        ):
            raise ValueError("q=0 mapping rows must use the closed definition roster")
        if (
            len(self.rows) != len(definitions)
            or {canonical_json_bytes(row.mapping_definition) for row in self.rows}
            != definition_bytes
        ):
            raise ValueError(
                "q=0 mapping rows must bind every closed definition exactly once"
            )
        if self.rows != tuple(sorted(self.rows, key=canonical_json_bytes)):
            raise ValueError("q=0 mapping rows must use canonical source order")
        selectors = tuple(row.mapping_definition.source_selector for row in self.rows)
        if len(selectors) != len(set(selectors)):
            raise ValueError(
                "q=0 source selectors must be exhaustive without duplicates"
            )
        source_context_selectors = {
            row.mapping_definition.source_selector
            for row in self.rows
            if row.row_kind == "SOURCE_CONTEXT"
        }
        if source_context_selectors != _m2_q0_source_leaf_selectors(
            self.source_compilation
        ):
            raise ValueError(
                "q=0 construction must retain every source leaf exactly once"
            )
        equality_rows = tuple(row for row in self.rows if row.row_kind == "EQUALITY")
        if len(equality_rows) != 1:
            raise ValueError("q=0 construction must bind one supported-domain equality")
        equality = equality_rows[0]
        expected_domain_value = _m2_q0_domain_typed_value(expected_domain)
        expected_domain_digest = canonical_sha256(
            expected_domain_value,
            domain=_M2_Q0_SHARED_VALUE_HASH_DOMAIN,
        )
        if (
            equality.shared_value != expected_domain_value
            or equality.source_value_sha256 != expected_domain_digest
            or equality.target_value_sha256 != expected_domain_digest
            or equality.shared_value_sha256 != expected_domain_digest
        ):
            raise ValueError(
                "q=0 construction authorized-domain equality must bind source and target"
            )
        source_leaf_values = _m2_q0_source_leaf_values(self.source_compilation)
        for row in self.rows:
            if row.row_kind != "SOURCE_CONTEXT":
                continue
            selector = row.mapping_definition.source_selector
            if row.source_value_sha256 != canonical_sha256(
                source_leaf_values[selector],
                domain="spatialcf/counterfactual/upright-se2/m2-q0/source-leaf/3.0",
            ):
                raise ValueError(
                    "q=0 source-context mapping digest must bind source provenance"
                )
        expected_non_equivalence = _m2_q0_non_equivalence_values(
            self.source_compilation,
            self.frozen_m3_policy_bundle,
        )
        non_equivalence_rows = {
            row.mapping_definition.source_selector: row
            for row in self.rows
            if row.row_kind == "NON_EQUIVALENCE"
        }
        if set(non_equivalence_rows) != set(expected_non_equivalence):
            raise ValueError(
                "q=0 construction must bind every heterogeneous policy root"
            )
        for selector, (source_value, target_value) in expected_non_equivalence.items():
            row = non_equivalence_rows[selector]
            if row.source_value_sha256 != canonical_sha256(
                source_value,
                domain="spatialcf/counterfactual/upright-se2/m2-q0/source-leaf/3.0",
            ) or row.target_value_sha256 != canonical_sha256(
                target_value,
                domain="spatialcf/counterfactual/upright-se2/m2-q0/target-leaf/3.0",
            ):
                raise ValueError(
                    "q=0 non-equivalence mapping digest must bind source and M3 policy"
                )
        if not any(row.row_kind == "SOURCE_CONTEXT" for row in self.rows):
            raise ValueError("q=0 construction must retain source provenance context")
        if not self.source_free_construction_rows:
            raise ValueError("q=0 construction must bind source-free angular constants")
        if self.source_free_construction_rows != tuple(
            sorted(self.source_free_construction_rows, key=canonical_json_bytes)
        ):
            raise ValueError(
                "q=0 source-free construction rows must use canonical order"
            )
        source_free_by_selector = {
            row.construction_selector: row for row in self.source_free_construction_rows
        }
        if set(source_free_by_selector) != {
            "construction:cardinal-own-pivot-q",
            "construction:target-tie-break",
        }:
            raise ValueError(
                "q=0 construction must bind its exact source-free constants"
            )
        q_row = source_free_by_selector["construction:cardinal-own-pivot-q"]
        tie_break_row = source_free_by_selector["construction:target-tie-break"]
        if (
            q_row.target_value.value_schema_ref
            != "schema:spatialcf/upright-se2/integer/1.0"
            or type(q_row.target_value.payload) is not IntegerValue
            or q_row.target_value.payload.value != 0
            or tie_break_row.target_value.value_schema_ref
            != "schema:spatialcf/upright-se2/enum-symbol/1.0"
            or type(tie_break_row.target_value.payload) is not EnumSymbolValue
            or tie_break_row.target_value.payload.symbol != "T_R_V_S_A"
        ):
            raise ValueError(
                "q=0 source-free constants must be fixed construction values"
            )
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2M2Q0MappingDefinition.model_rebuild()
UprightSE2M2Q0MappingRow.model_rebuild()
UprightSE2M2Q0SourceFreeConstructionRow.model_rebuild()
UprightSE2M2Q0Construction.model_rebuild()


# Keep supported public import and pickle lookup stable.
_m2_q0_source_leaf_values.__module__ = "spatialcf.domain.upright_se2"
_m2_q0_source_leaf_selectors.__module__ = "spatialcf.domain.upright_se2"
_m2_q0_dyadic_from_float.__module__ = "spatialcf.domain.upright_se2"
_m2_q0_authorized_domain_from_source.__module__ = "spatialcf.domain.upright_se2"
_m2_q0_domain_typed_value.__module__ = "spatialcf.domain.upright_se2"
_m2_q0_non_equivalence_values.__module__ = "spatialcf.domain.upright_se2"
UprightSE2M2Q0MappingDefinition.__module__ = "spatialcf.domain.upright_se2"
UprightSE2M2Q0MappingRow.__module__ = "spatialcf.domain.upright_se2"
UprightSE2M2Q0SourceFreeConstructionRow.__module__ = "spatialcf.domain.upright_se2"
UprightSE2M2Q0Construction.__module__ = "spatialcf.domain.upright_se2"
