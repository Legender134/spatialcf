"""Upright profile compilation contracts and intrinsic operations."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    CanonicalId,
    Quaternion,
    Sha256Digest,
)

from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
)

from spatialcf.domain.definitions import (
    DigestValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    IntervalValue,
)

from spatialcf.domain.operators import (
    StateVariableRef,
)

from spatialcf.domain.predicates import (
    GroundedObligationSet,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.domain._upright_se2.cells import (
    UprightSE2CompiledCell,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_COMPILER_OWNER_REF,
    UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
    UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
    _UPRIGHT_SE2_CARDINAL_QUATERNIONS,
    _UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS,
    _UPRIGHT_SE2_CLOSED_INTERVAL_TOPOLOGY_REF,
    _UPRIGHT_SE2_DERIVED_ROLES,
    _UPRIGHT_SE2_DERIVED_RULE_REF,
    _UPRIGHT_SE2_METRE_UNIT_REF,
    _UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
    _UPRIGHT_SE2_PRIMARY_ROLES,
    _UPRIGHT_SE2_REAL_SCHEMA_REF,
    _UPRIGHT_SE2_STATE_FAMILY_REF,
    _UPRIGHT_SE2_STATE_SCHEMA_REF,
    _UPRIGHT_SE2_UNCHANGED_LEAVES_HASH_DOMAIN,
    _UPRIGHT_SE2_WORLD_XY_FRAME_REF,
)

from spatialcf.domain._upright_se2.m2 import (
    UprightSE2M2Q0Construction,
)

from spatialcf.domain._upright_se2.semantics import (
    UprightSE2ContinuousSemanticClosure,
    UprightSE2SemanticClosure,
)

from spatialcf.domain._upright_se2.source_binding import (
    _bound_continuous_source_input,
    _bound_scene_object,
    _bound_source_input,
)

from spatialcf.domain._upright_se2.state import (
    UprightSE2CardinalOperation,
    UprightSE2CompilerClosure,
    UprightSE2ContinuousEndpointConstructionRecipe,
    UprightSE2ContinuousOperation,
    UprightSE2EndpointConstructionRecipe,
    UprightSE2StateFootprint,
    UprightSE2TranslationDomain,
)

from spatialcf.domain._upright_se2.yaw import (
    CanonicalSO2Angle,
    ContinuousYawAuthorization,
    ContinuousYawLift,
    ExactDyadic,
    FixedPivotBinding,
    LiftedYawInterval,
    PivotMode,
    _compose_upright_quaternion_from_primary_yaw,
    validate_directed_yaw_quaternion_consistency,
)


class UprightSE2Compilation(HashBoundCanonicalModel):
    """A deterministic cardinal compilation, never a solve/check/certificate result."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/upright-se2/compilation/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "upright_se2_compilation_sha256"

    solve_request_sha256: Sha256Digest
    source_solve_request: CounterfactualSolveRequest
    closure: UprightSE2CompilerClosure
    operation: UprightSE2CardinalOperation
    endpoint_construction_recipe: UprightSE2EndpointConstructionRecipe
    state_footprint: UprightSE2StateFootprint
    grounded_obligations: GroundedObligationSet
    semantic_closure: UprightSE2SemanticClosure
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    m2_q0_construction: UprightSE2M2Q0Construction | None = None
    upright_se2_compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_compilation_cells(self) -> Self:
        if self.source_solve_request.solve_request_sha256 != self.solve_request_sha256:
            raise ValueError(
                "compilation solve request digest does not bind its source request"
            )
        if (
            self.source_solve_request.semantic_problem_sha256
            != self.semantic_closure.semantic_problem.semantic_problem_sha256
            or canonical_json_bytes(self.source_solve_request.semantic_problem)
            != canonical_json_bytes(self.semantic_closure.semantic_problem)
        ):
            raise ValueError(
                "compilation semantic closure does not bind the source request"
            )
        if (
            self.closure.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
        ):
            raise ValueError(
                "compiler and semantic closures must bind the same profile digest"
            )
        if (
            self.semantic_closure.profile_registration.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
        ):
            raise ValueError(
                "semantic closure profile does not bind its profile digest"
            )
        if (
            self.endpoint_construction_recipe.operation_authorization_sha256
            != self.operation.authorization_sha256
            or self.endpoint_construction_recipe.translation_domain
            != self.operation.translation_domain
        ):
            raise ValueError(
                "compilation endpoint recipe must bind its operation domain"
            )
        if (
            self.semantic_closure.grounded_obligation_set_sha256
            != self.grounded_obligations.grounded_obligation_set_sha256
        ):
            raise ValueError(
                "semantic closure must bind the compiled grounded obligations"
            )
        if (
            self.semantic_closure.semantic_closure_sha256
            != self.closure.semantic_closure_sha256
        ):
            raise ValueError("compiler closure must bind the semantic closure")
        if (
            self.semantic_closure.policy_bundle_sha256
            != self.closure.policy_bundle_sha256
        ):
            raise ValueError("compiler closure must bind the executable policy bundle")
        if (
            self.semantic_closure.resource_policy.resource_policy_sha256
            != self.closure.resource_policy_sha256
        ):
            raise ValueError("compiler closure must bind the request resource policy")
        if (
            self.semantic_closure.definition_bundle_sha256
            != self.closure.definition_bundle_sha256
        ):
            raise ValueError(
                "compiler closure must bind the semantic definition bundle"
            )
        if not self.compiled_cells:
            raise ValueError(
                "upright se2 compilation must include one or more canonical cells"
            )
        if tuple(cell.authorization_sha256 for cell in self.compiled_cells) != (
            self.operation.authorization_sha256,
        ):
            raise ValueError(
                "compiled cells must bind the resolved cardinal authorization"
            )
        if self.m2_q0_construction is not None:
            construction = self.m2_q0_construction
            if (
                self.operation.quarter_turns_ccw != 0
                or self.operation.pivot_binding.pivot_mode is not PivotMode.OWN
                or self.operation.translation_domain != construction.authorized_domain
                or self.semantic_closure.policy_bundle_sha256
                != construction.frozen_m3_policy_bundle_sha256
            ):
                raise ValueError(
                    "q=0 construction must bind the domain-only M3 operation"
                )
            construction_facts = tuple(
                fact
                for bundle in self.source_solve_request.semantic_problem.scene_state.extension_fact_bundles
                for fact in bundle.facts
                if fact.fact_family_ref
                == "definition:spatialcf/upright-se2/m2-q0-construction/1.0"
            )
            if len(construction_facts) != 1:
                raise ValueError(
                    "q=0 construction root must be present in the source request"
                )
            fact = construction_facts[0]
            if (
                fact.fact_key != "fact-key:spatialcf/upright-se2/m2-q0-construction"
                or type(fact.value.payload) is not DigestValue
                or fact.value.payload.value != construction.m2_q0_construction_sha256
            ):
                raise ValueError(
                    "q=0 construction request root does not bind construction"
                )
        _validate_compilation_source_binding(self)
        return self


class UprightSE2ContinuousCompilation(HashBoundCanonicalModel):
    """A deterministic continuous compilation, never a checked result.

    This additive model has its own hash domain and carries the canonical
    lifted-yaw roots.  Cardinal compilation remains closed over its existing
    type and wire.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-compilation/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_upright_se2_compilation_sha256"

    solve_request_sha256: Sha256Digest
    source_solve_request: CounterfactualSolveRequest
    closure: UprightSE2CompilerClosure
    operation: UprightSE2ContinuousOperation
    endpoint_construction_recipe: UprightSE2ContinuousEndpointConstructionRecipe
    state_footprint: UprightSE2StateFootprint
    grounded_obligations: GroundedObligationSet
    semantic_closure: UprightSE2ContinuousSemanticClosure
    compiled_cells: tuple[UprightSE2CompiledCell, ...]
    continuous_yaw_lift: ContinuousYawLift
    continuous_upright_se2_compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_continuous_compilation(self) -> Self:
        if self.source_solve_request.solve_request_sha256 != self.solve_request_sha256:
            raise ValueError("continuous compilation must bind its source request")
        if (
            self.closure.profile_registration_sha256
            != self.semantic_closure.profile_registration_sha256
            or self.closure.semantic_closure_sha256
            != self.semantic_closure.semantic_closure_sha256
            or self.closure.policy_bundle_sha256
            != self.semantic_closure.policy_bundle_sha256
            or self.closure.resource_policy_sha256
            != self.semantic_closure.resource_policy.resource_policy_sha256
        ):
            raise ValueError("continuous compiler closure does not bind semantics")
        if (
            self.endpoint_construction_recipe.operation_authorization_sha256
            != self.operation.authorization_sha256
            or self.endpoint_construction_recipe.translation_domain
            != self.operation.translation_domain
            or self.endpoint_construction_recipe.yaw_domain != self.operation.yaw_domain
        ):
            raise ValueError("continuous endpoint recipe does not bind operation")
        if self.continuous_yaw_lift.yaw_domain != self.operation.yaw_domain:
            raise ValueError("continuous lift does not bind operation yaw domain")
        if len(self.compiled_cells) != 1:
            raise ValueError("continuous compilation requires one canonical lift root")
        cell = self.compiled_cells[0]
        interval = self.continuous_yaw_lift.intervals[0]
        if (
            cell.authorization_sha256 != self.operation.authorization_sha256
            or cell.x_lower != self.operation.translation_domain.x_lower
            or cell.x_upper != self.operation.translation_domain.x_upper
            or cell.y_lower != self.operation.translation_domain.y_lower
            or cell.y_upper != self.operation.translation_domain.y_upper
            or cell.yaw_interval != interval
        ):
            raise ValueError("continuous compilation root does not bind operation")
        _validate_continuous_compilation_source_binding(self)
        return self


def _bound_state_leaf(subject_id: CanonicalId, role: str) -> StateVariableRef:
    return StateVariableRef(
        state_variable_schema_ref=f"schema:spatialcf/upright-se2/{role}/1.0",
        state_schema_ref=_UPRIGHT_SE2_STATE_SCHEMA_REF,
        fact_family_ref=_UPRIGHT_SE2_STATE_FAMILY_REF,
        entity_or_fact_key=subject_id,
        field_path_ref=f"field-path:upright-se2-{role}",
    )


def _bound_translation_domain(
    problem: CounterfactualProblemIR,
    subject_id: CanonicalId,
) -> UprightSE2TranslationDomain:
    expected = {
        "subject-world-x": _bound_state_leaf(subject_id, "subject-world-x"),
        "subject-world-y": _bound_state_leaf(subject_id, "subject-world-y"),
    }
    bounds = problem.intervention_authorization.variable_bounds
    if len(bounds) != len(expected):
        raise ValueError("source authorization must bind complete world-XY bounds")
    resolved: dict[str, tuple[ExactDyadic, ExactDyadic]] = {}
    for bound in bounds:
        role = next(
            (
                candidate
                for candidate, state_leaf in expected.items()
                if bound.state_variable_ref == state_leaf
            ),
            None,
        )
        if role is None or role in resolved:
            raise ValueError("source authorization bounds must target subject world XY")
        if (
            bound.value_schema_ref,
            bound.frame_ref,
            bound.unit_ref,
            bound.topology_ref,
        ) != (
            _UPRIGHT_SE2_REAL_SCHEMA_REF,
            _UPRIGHT_SE2_WORLD_XY_FRAME_REF,
            _UPRIGHT_SE2_METRE_UNIT_REF,
            _UPRIGHT_SE2_CLOSED_INTERVAL_TOPOLOGY_REF,
        ):
            raise ValueError("source authorization bounds use the wrong type semantics")
        domain = bound.typed_domain
        if (
            domain.value_schema_ref != _UPRIGHT_SE2_REAL_SCHEMA_REF
            or type(domain.payload) is not IntervalValue
            or domain.payload.endpoint_schema_ref != _UPRIGHT_SE2_REAL_SCHEMA_REF
            or type(domain.payload.lower) is not FiniteRealValue
            or type(domain.payload.upper) is not FiniteRealValue
            or not domain.payload.lower_closed
            or not domain.payload.upper_closed
        ):
            raise ValueError(
                "source authorization bounds must be closed finite-real intervals"
            )
        lower = ExactDyadic(
            numerator=Fraction.from_float(domain.payload.lower.value).numerator,
            denominator=Fraction.from_float(domain.payload.lower.value).denominator,
        )
        upper = ExactDyadic(
            numerator=Fraction.from_float(domain.payload.upper.value).numerator,
            denominator=Fraction.from_float(domain.payload.upper.value).denominator,
        )
        resolved[role] = (lower, upper)
    if set(resolved) != set(expected):
        raise ValueError("source authorization must provide both world-XY bounds")
    return UprightSE2TranslationDomain(
        x_lower=resolved["subject-world-x"][0],
        x_upper=resolved["subject-world-x"][1],
        y_lower=resolved["subject-world-y"][0],
        y_upper=resolved["subject-world-y"][1],
    )


def _bound_cardinal_after_yaw(
    yaw_before: CanonicalSO2Angle,
    q: int,
) -> CanonicalSO2Angle:
    turns = (
        Fraction.from_float(yaw_before.turns) + _UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS[q]
    )
    while turns < Fraction(-1, 2):
        turns += 1
    while turns >= Fraction(1, 2):
        turns -= 1
    return CanonicalSO2Angle(turns=0.0 if turns == 0 else float(turns))


def _bound_rotate_cardinal_xy(x: float, y: float, q: int) -> tuple[float, float]:
    if q == 0:
        return (0.0 if x == 0.0 else x, 0.0 if y == 0.0 else y)
    if q == 1:
        return (0.0 if y == 0.0 else -y, 0.0 if x == 0.0 else x)
    if q == 2:
        return (0.0 if x == 0.0 else -x, 0.0 if y == 0.0 else -y)
    return (0.0 if y == 0.0 else y, 0.0 if x == 0.0 else -x)


def _bound_cardinal_quaternion(
    rotation: Quaternion,
    q: int,
    *,
    primary_expected_yaw: CanonicalSO2Angle,
) -> Quaternion:
    delta_z, delta_w = _UPRIGHT_SE2_CARDINAL_QUATERNIONS[q]
    return _compose_upright_quaternion_from_primary_yaw(
        rotation=rotation,
        delta_z=delta_z,
        delta_w=delta_w,
        primary_expected_yaw=primary_expected_yaw,
    )


def _bound_cardinal_compiled_cell(
    operation: UprightSE2CardinalOperation,
) -> UprightSE2CompiledCell:
    """Derive the sole exact cardinal cell from an already bound operation."""

    yaw = _UPRIGHT_SE2_CARDINAL_TURN_FRACTIONS[operation.quarter_turns_ccw]
    endpoint = ExactDyadic(numerator=yaw.numerator, denominator=yaw.denominator)
    return UprightSE2CompiledCell.seal(
        cell_id=(
            f"cell:spatialcf/upright-se2/cardinal/{operation.authorization_sha256}"
        ),
        authorization_sha256=operation.authorization_sha256,
        x_lower=operation.translation_domain.x_lower,
        x_upper=operation.translation_domain.x_upper,
        y_lower=operation.translation_domain.y_lower,
        y_upper=operation.translation_domain.y_upper,
        yaw_interval=LiftedYawInterval(
            lower=endpoint,
            upper=endpoint,
            seam_ownership="NONE",
        ),
    )


def _validate_compilation_source_binding(compilation: UprightSE2Compilation) -> None:
    """Recompute the Task 2 transition and closure inputs from the bound request."""

    source_request = compilation.source_solve_request
    problem = source_request.semantic_problem
    source_input = _bound_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    authorization = problem.intervention_authorization
    if (
        authorization.editable_entity_ids != (f"entity:{source_input.subject_id}",)
        or authorization.allowed_operator_refs != (source_input.operator_ref,)
        or authorization.authorized_primary_write_set
        != tuple(
            sorted(
                (
                    _bound_state_leaf(source_input.subject_id, role)
                    for role in _UPRIGHT_SE2_PRIMARY_ROLES
                ),
                key=canonical_json_bytes,
            )
        )
        or authorization.maximum_program_steps != 1
        or authorization.maximum_edited_entities != 1
        or authorization.required_derived_rule_refs != (_UPRIGHT_SE2_DERIVED_RULE_REF,)
        or authorization.complete_state_delta_policy_ref
        != "definition:spatialcf/upright-se2/complete-state-delta/1.0"
    ):
        raise ValueError("compilation source authorization does not close")
    operation = compilation.operation
    expected_pivot_mode = (
        PivotMode.OWN
        if source_input.operator_ref == UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF
        else PivotMode.REFERENCE
    )
    expected_pivot_id = (
        source_input.subject_id
        if expected_pivot_mode is PivotMode.OWN
        else source_input.reference_id
    )
    expected_pivot_state_sha256 = canonical_sha256(
        {
            "scene_state_sha256": problem.scene_state.scene_state_sha256,
            "pivot_entity_id": expected_pivot_id,
            "object_pivot_pose": (
                subject.pose if expected_pivot_mode is PivotMode.OWN else reference.pose
            ),
        },
        domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
    )
    if (
        source_input.operator_ref
        not in (
            UPRIGHT_SE2_CARDINAL_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CARDINAL_REFERENCE_PIVOT_OPERATOR_REF,
        )
        or source_input.subject_id == source_input.reference_id
        or operation.subject_id != source_input.subject_id
        or operation.authorization.operator_ref != source_input.operator_ref
        or operation.quarter_turns_ccw != source_input.quarter_turns_ccw
        or operation.pivot_binding.pivot_mode is not expected_pivot_mode
        or operation.pivot_binding.pivot_entity_id != expected_pivot_id
        or operation.pivot_binding.pivot_state_sha256 != expected_pivot_state_sha256
        or operation.translation_domain
        != _bound_translation_domain(problem, source_input.subject_id)
    ):
        raise ValueError("compilation operation does not bind the source request")
    subject_pose = subject.pose.world_from_object
    reference_pose = reference.pose.world_from_object
    validate_directed_yaw_quaternion_consistency(
        source_input.subject_yaw_turns,
        subject_pose.rotation,
    )
    expected_recipe = UprightSE2EndpointConstructionRecipe.seal(
        source_scene_state_sha256=problem.scene_state.scene_state_sha256,
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        pivot_binding=operation.pivot_binding,
        quarter_turns_ccw=source_input.quarter_turns_ccw,
        translation_domain=operation.translation_domain,
        subject_before_pose=subject_pose,
        reference_before_pose=reference_pose,
        subject_yaw_turns=source_input.subject_yaw_turns,
        evaluation_scene=scene,
    )
    if canonical_json_bytes(
        compilation.endpoint_construction_recipe
    ) != canonical_json_bytes(expected_recipe):
        raise ValueError(
            "compilation endpoint recipe does not derive from the source request"
        )
    leaf_index = problem.scene_state.canonical_state_leaf_index
    footprint = compilation.state_footprint
    expected_primary_writes = tuple(
        sorted(
            (
                _bound_state_leaf(source_input.subject_id, role)
                for role in _UPRIGHT_SE2_PRIMARY_ROLES
            ),
            key=canonical_json_bytes,
        )
    )
    expected_derived_writes = tuple(
        sorted(
            (
                _bound_state_leaf(source_input.subject_id, role)
                for role in _UPRIGHT_SE2_DERIVED_ROLES
            ),
            key=canonical_json_bytes,
        )
    )
    expected_written_bytes = {
        *(canonical_json_bytes(leaf) for leaf in expected_primary_writes),
        *(canonical_json_bytes(leaf) for leaf in expected_derived_writes),
    }
    expected_frozen_leaf_refs = tuple(
        leaf
        for leaf in leaf_index.leaves
        if canonical_json_bytes(leaf) not in expected_written_bytes
    )
    manifest = footprint.state_delta_manifest
    partition = tuple(
        sorted(
            (
                *footprint.state_delta_manifest.authorized_primary_writes,
                *footprint.state_delta_manifest.recomputed_derived_writes,
                *footprint.frozen_leaf_refs,
            ),
            key=canonical_json_bytes,
        )
    )
    if (
        partition != leaf_index.leaves
        or manifest.authorized_primary_writes != expected_primary_writes
        or manifest.recomputed_derived_writes != expected_derived_writes
        or footprint.frozen_leaf_refs != expected_frozen_leaf_refs
        or manifest.unchanged_leaves_digest
        != canonical_sha256(
            expected_frozen_leaf_refs,
            domain=_UPRIGHT_SE2_UNCHANGED_LEAVES_HASH_DOMAIN,
        )
        or footprint.complete_before_leaf_index_sha256
        != leaf_index.state_leaf_index_sha256
        or footprint.complete_after_leaf_index_sha256
        != leaf_index.state_leaf_index_sha256
    ):
        raise ValueError(
            "compilation state footprint does not bind the source leaf index"
        )
    expected_cell = _bound_cardinal_compiled_cell(operation)
    if len(compilation.compiled_cells) != 1 or canonical_json_bytes(
        compilation.compiled_cells[0]
    ) != canonical_json_bytes(expected_cell):
        raise ValueError("compiled cells do not derive from the source-bound operation")
    if (
        compilation.closure.definition_bundle_sha256
        != problem.definition_bundle.definition_bundle_sha256
        or compilation.closure.solve_policy_definition_bundle_sha256
        != source_request.solve_policy_definition_bundle.definition_bundle_sha256
        or dict(
            source_request.implementation_registry_snapshot.implementation_build_hashes
        ).get(UPRIGHT_SE2_COMPILER_OWNER_REF)
        != compilation.closure.compiler_build_sha256
    ):
        raise ValueError("compiler closure does not bind the source request")


def _validate_continuous_compilation_source_binding(
    compilation: UprightSE2ContinuousCompilation,
) -> None:
    """Replay continuous operation and recipe roots from the bound request."""

    source_request = compilation.source_solve_request
    problem = source_request.semantic_problem
    source_input = _bound_continuous_source_input(problem)
    scene = problem.scene_state.base_scene_payload
    subject = _bound_scene_object(scene, source_input.subject_id)
    reference = _bound_scene_object(scene, source_input.reference_id)
    expected_pivot_mode = (
        PivotMode.OWN
        if source_input.operator_ref == UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF
        else PivotMode.REFERENCE
    )
    expected_pivot_id = (
        source_input.subject_id
        if expected_pivot_mode is PivotMode.OWN
        else source_input.reference_id
    )
    expected_pivot_pose = (
        subject.pose if expected_pivot_mode is PivotMode.OWN else reference.pose
    )
    expected_pivot = FixedPivotBinding.seal(
        subject_id=source_input.subject_id,
        pivot_mode=expected_pivot_mode,
        pivot_entity_id=expected_pivot_id,
        pivot_state_sha256=canonical_sha256(
            {
                "scene_state_sha256": problem.scene_state.scene_state_sha256,
                "pivot_entity_id": expected_pivot_id,
                "object_pivot_pose": expected_pivot_pose,
            },
            domain=_UPRIGHT_SE2_PIVOT_STATE_HASH_DOMAIN,
        ),
    )
    expected_authorization = ContinuousYawAuthorization.seal(
        subject_id=source_input.subject_id,
        operator_ref=source_input.operator_ref,
        pivot_binding=expected_pivot,
        yaw_domain=source_input.yaw_domain,
    )
    operation = compilation.operation
    if (
        source_input.operator_ref
        not in (
            UPRIGHT_SE2_CONTINUOUS_OWN_PIVOT_OPERATOR_REF,
            UPRIGHT_SE2_CONTINUOUS_REFERENCE_PIVOT_OPERATOR_REF,
        )
        or source_input.subject_id == source_input.reference_id
        or operation.subject_id != source_input.subject_id
        or canonical_json_bytes(operation.authorization)
        != canonical_json_bytes(expected_authorization)
        or operation.pivot_binding != expected_pivot
        or operation.yaw_domain != source_input.yaw_domain
        or operation.translation_domain
        != _bound_translation_domain(problem, source_input.subject_id)
    ):
        raise ValueError(
            "continuous compilation operation does not bind source request"
        )
    subject_pose = subject.pose.world_from_object
    reference_pose = reference.pose.world_from_object
    validate_directed_yaw_quaternion_consistency(
        source_input.subject_yaw_turns,
        subject_pose.rotation,
    )
    expected_recipe = UprightSE2ContinuousEndpointConstructionRecipe.seal(
        source_scene_state_sha256=problem.scene_state.scene_state_sha256,
        operation_authorization_sha256=operation.authorization_sha256,
        subject_id=source_input.subject_id,
        reference_id=source_input.reference_id,
        pivot_binding=expected_pivot,
        yaw_domain=source_input.yaw_domain,
        translation_domain=operation.translation_domain,
        subject_before_pose=subject_pose,
        reference_before_pose=reference_pose,
        subject_yaw_turns=source_input.subject_yaw_turns,
        evaluation_scene=scene,
    )
    if canonical_json_bytes(
        compilation.endpoint_construction_recipe
    ) != canonical_json_bytes(expected_recipe):
        raise ValueError(
            "continuous compilation endpoint recipe does not derive from source request"
        )


# Resolve local model forward references before restoring public identities.
UprightSE2Compilation.model_rebuild()
UprightSE2ContinuousCompilation.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2Compilation.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousCompilation.__module__ = "spatialcf.domain.upright_se2"
_bound_state_leaf.__module__ = "spatialcf.domain.upright_se2"
_bound_translation_domain.__module__ = "spatialcf.domain.upright_se2"
_bound_cardinal_after_yaw.__module__ = "spatialcf.domain.upright_se2"
_bound_rotate_cardinal_xy.__module__ = "spatialcf.domain.upright_se2"
_bound_cardinal_quaternion.__module__ = "spatialcf.domain.upright_se2"
_bound_cardinal_compiled_cell.__module__ = "spatialcf.domain.upright_se2"
_validate_compilation_source_binding.__module__ = "spatialcf.domain.upright_se2"
_validate_continuous_compilation_source_binding.__module__ = "spatialcf.domain.upright_se2"
