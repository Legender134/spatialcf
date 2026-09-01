"""Composed semantic counterfactual problems and operational solve requests.

This additive M1 module composes the frozen current :class:`CanonicalScene`
with typed extension facts.  It intentionally owns only local structural and
self-hash validation: definition/profile resolution, leaf-index completeness,
transition replay, authorization subset checks, and backend compatibility are
static-registry concerns for Task 8.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Self, TypeVar

from pydantic import Field, model_validator

from spatialcf.domain.base import CanonicalId, CanonicalModel, FactSetV2, Sha256Digest
from spatialcf.domain.definitions import (
    DefinitionBundle,
    DefinitionRef,
    HashBoundCanonicalModel,
    SchemaRef,
    TypedValue,
)
from spatialcf.domain.operators import (
    OperationInvocation,
    StateDeltaManifest,
    StateLeafIndex,
)
from spatialcf.domain.predicates import (
    AfterGoal,
    BeforePrecondition,
    ObservationObligation,
    PreservationInvariant,
)
from spatialcf.domain.profiles import (
    BackendDescriptorBundle,
    BackendRoutingPolicy,
    CounterfactualSolverConfig,
    ImplementationRegistrySnapshot,
    InterventionAuthorization,
    ObjectiveExpression,
    ProfileRef,
    ProofPolicy,
    ResourcePolicy,
)
from spatialcf.domain.scene import CanonicalScene
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256

__all__ = (
    "CounterfactualProblemIR",
    "CounterfactualSolveRequest",
    "EditProgram",
    "ExtensionFact",
    "ExtensionFactBundle",
    "SceneStateEnvelope",
)

_ValueT = TypeVar("_ValueT")
_FactT = TypeVar("_FactT", bound=CanonicalModel)
_BASE_SCENE_HASH_DOMAIN = "spatialcf/counterfactual/base-scene-payload/3.0"
_BASE_SCENE_OBJECTS_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/objects/3.0"
)
_BASE_SCENE_GEOMETRY_INSTANCES_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/geometry-instances/3.0"
)
_BASE_SCENE_COLLISION_BODIES_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/collision-bodies/3.0"
)
_BASE_SCENE_WORKSPACE_BOUNDARIES_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/workspace-boundaries/3.0"
)
_BASE_SCENE_KNOWN_FREE_SPACES_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/known-free-spaces/3.0"
)
_BASE_SCENE_SUPPORT_SURFACES_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/support-surfaces/3.0"
)
_BASE_SCENE_CAMERAS_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/cameras/3.0"
)
_BASE_SCENE_BASELINE_OBSERVATIONS_FAMILY_REF: DefinitionRef = (
    "definition:spatialcf/counterfactual/base-scene/baseline-observations/3.0"
)


def _require_sorted_unique_by_bytes(
    values: tuple[_ValueT, ...],
    label: str,
    *,
    nonempty: bool = False,
) -> None:
    if nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if encoded != tuple(sorted(encoded)):
        raise ValueError(f"{label} must be sorted")
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"{label} must not contain duplicate entries")


def _all_fact_values(facts: FactSetV2[_FactT]) -> tuple[_FactT, ...]:
    """Read every stable fact value without trusting a completeness branch."""

    return tuple(
        value
        for values in (facts.values, facts.inner_values, facts.outer_values)
        if values is not None
        for value in values
    )


def _derived_base_fact_ownership_keys(
    scene: CanonicalScene,
) -> frozenset[tuple[DefinitionRef, CanonicalId, CanonicalId]]:
    """Derive non-overridable base ownership from the embedded current scene."""

    ownership: set[tuple[DefinitionRef, CanonicalId, CanonicalId]] = set()
    ownership.update(
        (_BASE_SCENE_OBJECTS_FAMILY_REF, fact.object_id, fact.object_id)
        for fact in _all_fact_values(scene.objects)
    )
    ownership.update(
        (_BASE_SCENE_GEOMETRY_INSTANCES_FAMILY_REF, fact.geometry_id, fact.geometry_id)
        for fact in _all_fact_values(scene.geometry_instances)
    )
    ownership.update(
        (_BASE_SCENE_COLLISION_BODIES_FAMILY_REF, fact.body_id, fact.body_id)
        for fact in _all_fact_values(scene.collision_bodies)
    )
    ownership.update(
        (_BASE_SCENE_WORKSPACE_BOUNDARIES_FAMILY_REF, fact.fact_id, fact.fact_id)
        for fact in _all_fact_values(scene.workspace_boundaries)
    )
    ownership.update(
        (_BASE_SCENE_KNOWN_FREE_SPACES_FAMILY_REF, fact.fact_id, fact.fact_id)
        for fact in _all_fact_values(scene.known_free_spaces)
    )
    ownership.update(
        (_BASE_SCENE_SUPPORT_SURFACES_FAMILY_REF, fact.surface_id, fact.surface_id)
        for fact in _all_fact_values(scene.support_surfaces)
    )
    ownership.update(
        (_BASE_SCENE_CAMERAS_FAMILY_REF, fact.camera_id, fact.camera_id)
        for fact in _all_fact_values(scene.cameras)
    )
    ownership.update(
        (
            _BASE_SCENE_BASELINE_OBSERVATIONS_FAMILY_REF,
            fact.object_id,
            fact.observation_id,
        )
        for fact in _all_fact_values(scene.baseline_observations)
    )
    return frozenset(ownership)


class ExtensionFact(CanonicalModel):
    """One typed extension fact owned by exactly one fact-family triple."""

    fact_family_ref: DefinitionRef
    subject_entity_id: CanonicalId
    fact_key: CanonicalId
    value: TypedValue

    @property
    def ownership_key(self) -> tuple[DefinitionRef, CanonicalId, CanonicalId]:
        return (self.fact_family_ref, self.subject_entity_id, self.fact_key)


class ExtensionFactBundle(HashBoundCanonicalModel):
    """A hash-bound, canonical set of typed extension facts."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/extension-fact-bundle/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "extension_fact_bundle_sha256"

    facts: tuple[ExtensionFact, ...]
    extension_fact_bundle_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_extension_fact_ownership(self) -> Self:
        _require_sorted_unique_by_bytes(self.facts, "extension facts")
        ownership = tuple(fact.ownership_key for fact in self.facts)
        if len(set(ownership)) != len(ownership):
            raise ValueError("extension facts must not duplicate one ownership")
        return self


class SceneStateEnvelope(HashBoundCanonicalModel):
    """The complete current scene plus typed additive facts and state index."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/scene-state/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "scene_state_sha256"

    base_scene_schema_ref: SchemaRef
    base_scene_payload: CanonicalScene
    base_scene_sha256: Sha256Digest
    extension_fact_bundles: tuple[ExtensionFactBundle, ...]
    closed_entity_index: tuple[CanonicalId, ...]
    canonical_state_leaf_index: StateLeafIndex
    scene_state_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Bind the frozen base scene using its additive, non-v2 digest domain."""

        if "base_scene_sha256" in values:
            raise ValueError("seal() derives the base scene digest")
        if "base_scene_payload" not in values:
            return super().seal(**values)
        scene = CanonicalScene.model_validate(values["base_scene_payload"], strict=True)
        return super().seal(
            **(
                values
                | {
                    "base_scene_payload": scene,
                    "base_scene_sha256": canonical_sha256(
                        scene,
                        domain=_BASE_SCENE_HASH_DOMAIN,
                    ),
                }
            )
        )

    @model_validator(mode="after")
    def _validate_scene_composition(self) -> Self:
        expected_base_digest = canonical_sha256(
            self.base_scene_payload,
            domain=_BASE_SCENE_HASH_DOMAIN,
        )
        if self.base_scene_sha256 != expected_base_digest:
            raise ValueError("base scene digest does not match the canonical payload")

        base_keys = _derived_base_fact_ownership_keys(self.base_scene_payload)

        _require_sorted_unique_by_bytes(
            self.extension_fact_bundles,
            "extension fact bundles",
        )
        _require_sorted_unique_by_bytes(
            self.closed_entity_index,
            "closed entity index",
        )
        entities = set(self.closed_entity_index)
        extension_keys: set[tuple[DefinitionRef, CanonicalId, CanonicalId]] = set()
        for bundle in self.extension_fact_bundles:
            for fact in bundle.facts:
                if fact.subject_entity_id not in entities:
                    raise ValueError(
                        "extension facts must belong to the closed entity index"
                    )
                if fact.ownership_key in base_keys:
                    raise ValueError(
                        "extension fact conflicts with base fact ownership"
                    )
                if fact.ownership_key in extension_keys:
                    raise ValueError("duplicate extension ownership")
                extension_keys.add(fact.ownership_key)
        return self


class EditProgram(HashBoundCanonicalModel):
    """A deterministic ordered program with its complete hash-bound after state."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/edit-program/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "program_sha256"

    program_id: CanonicalId
    semantic_problem_sha256: Sha256Digest
    action_space_profile_sha256: Sha256Digest
    steps: tuple[OperationInvocation, ...]
    before_state_sha256: Sha256Digest
    after_scene_state: SceneStateEnvelope
    after_scene_state_sha256: Sha256Digest
    state_delta_manifest: StateDeltaManifest
    grounded_obligation_set_sha256: Sha256Digest
    program_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_complete_after_state(self) -> Self:
        if not self.steps:
            raise ValueError(
                "edit programs must contain at least one ordered invocation"
            )
        if self.after_scene_state_sha256 != self.after_scene_state.scene_state_sha256:
            raise ValueError(
                "after scene state digest does not match the complete state"
            )
        return self


class CounterfactualProblemSchemaIdentity(CanonicalModel):
    """The one exact schema identity permitted for M1 semantic problem roots."""

    schema_name: Literal["canonical-counterfactual-problem"] = (
        "canonical-counterfactual-problem"
    )
    schema_version: Literal["3.0"] = "3.0"


class CounterfactualProblemIR(HashBoundCanonicalModel):
    """A pure semantic counterfactual root without backend or native input."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/semantic-problem/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "semantic_problem_sha256"

    schema_identity: CounterfactualProblemSchemaIdentity = Field(
        default_factory=CounterfactualProblemSchemaIdentity
    )
    problem_id: CanonicalId
    scene_state: SceneStateEnvelope
    definition_bundle: DefinitionBundle
    semantics_profile_ref: ProfileRef
    action_space_profile_ref: ProfileRef
    intervention_authorization: InterventionAuthorization
    before_preconditions: tuple[BeforePrecondition, ...]
    after_goal: AfterGoal
    preservation_invariants: tuple[PreservationInvariant, ...]
    explicit_observation_obligations: tuple[ObservationObligation, ...]
    objective_expression: ObjectiveExpression
    numeric_semantics_ref: DefinitionRef
    semantic_problem_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_semantic_wrapper_sets(self) -> Self:
        before_bytes = tuple(
            canonical_json_bytes(item) for item in self.before_preconditions
        )
        if len(set(before_bytes)) != len(before_bytes):
            raise ValueError("duplicate before precondition")
        if before_bytes != tuple(sorted(before_bytes)):
            raise ValueError("before preconditions must be sorted")
        _require_sorted_unique_by_bytes(
            self.preservation_invariants,
            "preservation invariants",
        )
        _require_sorted_unique_by_bytes(
            self.explicit_observation_obligations,
            "explicit observation obligations",
        )
        return self


class CounterfactualSolveRequest(HashBoundCanonicalModel):
    """An operational root that embeds, but never changes, semantic identity."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/solve-request/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "solve_request_sha256"

    semantic_problem: CounterfactualProblemIR
    semantic_problem_sha256: Sha256Digest
    solve_policy_definition_bundle: DefinitionBundle
    implementation_registry_snapshot: ImplementationRegistrySnapshot
    backend_descriptor_bundle: BackendDescriptorBundle
    solver_config: CounterfactualSolverConfig
    proof_policy: ProofPolicy
    resource_policy: ResourcePolicy
    backend_routing_policy: BackendRoutingPolicy
    solve_request_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_embedded_semantic_root(self) -> Self:
        if (
            self.semantic_problem_sha256
            != self.semantic_problem.semantic_problem_sha256
        ):
            raise ValueError("semantic problem digest does not match the embedded root")
        return self
