"""Hash-bound registration for the retained planar-translate v2 route.

The compatibility records deliberately register data and identities only.  They
do not compile a v2 problem, execute a backend, or replace the M1
``BackendProposal`` proof-material contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import ClassVar, Literal, Self

from pydantic import model_validator

from spatialcf.domain.base import (
    CanonicalId,
    CanonicalModel,
    NonNegativeFiniteFloat,
    Sha256Digest,
)
from spatialcf.domain.counterfactual import (
    CounterfactualProblemIR,
    CounterfactualSolveRequest,
    ExtensionFact,
    ExtensionFactBundle,
)
from spatialcf.domain.definitions import (
    CapabilityRef,
    DefinitionRef,
    DigestValue,
    EnumSymbolValue,
    FiniteRealValue,
    HashBoundCanonicalModel,
    TypedValue,
)
from spatialcf.domain.outcomes import (
    BackendProposal,
    BackendSelectionRecord,
    CapabilityMatch,
    ProofMaterialEnvelope,
    ResourceUsage,
)
from spatialcf.domain.problem import SemanticProblemV2_3
from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    BackendRef,
    OwnerRef,
    SemanticsProfile,
    SolverBackendDescriptor,
)
from spatialcf.domain.serialization import canonical_json_bytes, canonical_sha256
from spatialcf.domain.solver import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawMinimumCostSolveOutcomeV2_9,
    ContinuousYawProvenUnsatResultV2_9,
    ContinuousYawSolveStatusV2,
    ContinuousYawSolverConfigV2_9,
    ContinuousYawUncertifiedResultV2_9,
)

__all__ = (
    "PlanarTranslateV2SourceArtifacts",
    "PlanarTranslateFieldMapping",
    "PlanarTranslateProfileRegistration",
    "PlanarTranslateCompatibilityLineage",
    "PlanarTranslateCompilation",
    "PlanarTranslateDelegatedOutcome",
)

PLANAR_TRANSLATE_PROFILE_REF = "spatialcf/planar_translate@2"
PLANAR_TRANSLATE_SEMANTICS_PROFILE_REF = "spatialcf/planar_translate/semantics@2"
PLANAR_TRANSLATE_MAPPING_DEFINITION_REF = (
    "definition:spatialcf/planar-translate/v2-to-v3-mapping/2.0"
)
PLANAR_TRANSLATE_BACKEND_REF = "backend:spatialcf/planar-translate-v2-owner"
PLANAR_TRANSLATE_BACKEND_OWNER_REF = "owner:spatialcf/core/current-v2-solver"
PLANAR_TRANSLATE_CHECKER_OWNER_REF = "owner:spatialcf/core/current-v2-verifier"
PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF = (
    "capability:spatialcf/counterfactual/profile-planar-translate/2"
)
PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF = (
    "capability:spatialcf/counterfactual/compile-planar-translate-v2/2"
)
PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF = (
    "capability:spatialcf/counterfactual/solve-planar-translate-v2/2"
)
PLANAR_TRANSLATE_CHECKER_CAPABILITY_REF = (
    "capability:spatialcf/counterfactual/verify-planar-translate-v2/2"
)

_MAPPING_DEFINITION_HASH_DOMAIN = (
    "spatialcf/counterfactual/planar-translate/mapping-definition/2.0"
)
_PROBLEM_FIELD_NAMES = (
    "schema_identity",
    "scene",
    "constraints",
    "relation_semantics",
    "visibility_semantics",
    "objective",
    "numeric_policy",
)
_SCENE_FIELD_NAMES = (
    "schema_identity",
    "scene_id",
    "coordinate_system",
    "objects",
    "geometry_instances",
    "collision_bodies",
    "workspace_boundaries",
    "known_free_spaces",
    "support_surfaces",
    "cameras",
    "baseline_observations",
)
_CONSTRAINT_FIELD_NAMES = (
    "schema_identity",
    "constraint_set_id",
    "allowed_edit",
    "position_domain",
    "collision_constraints",
    "support_constraints",
    "visibility_constraints",
    "target_relation",
)
_OBJECTIVE_FIELD_NAMES = (
    "schema_identity",
    "objective_id",
    "mode",
    "translation",
    "relation_damage",
    "visibility_change",
    "safety_margin",
    "tie_break",
)
_NUMERIC_POLICY_FIELD_NAMES = (
    "linear_tolerance_m",
    "area_tolerance_m2",
    "angular_tolerance_rad",
    "pixel_tolerance_px",
    "fraction_tolerance",
)
_SOLVER_CONFIG_FIELD_NAMES = (
    "schema_identity",
    "algorithm_id",
    "algorithm_version",
    "candidate_config",
    "camera_frame_kernel_id",
    "target_projection_kernel_id",
    "visibility_projection_kernel_id",
    "objective_kernel_id",
    "max_objective_partition_cells",
    "max_branch_nodes",
    "max_refinement_steps",
    "target_optimality_gap",
)
_CANDIDATE_CONFIG_FIELD_NAMES = (
    "schema_identity",
    "algorithm_id",
    "algorithm_version",
    "so2_kernel_id",
    "so2_kernel_version",
    "obstacle_kernel_id",
    "obstacle_kernel_version",
    "partition_kernel_id",
    "partition_kernel_version",
    "intersection_kernel_id",
    "intersection_kernel_version",
    "support_projection_kernel_id",
    "support_projection_kernel_version",
    "max_domain_operations",
    "max_so2_atomic_steps",
    "max_candidate_cells",
)
_ALGORITHM_KERNEL_BINDINGS = (
    ("algorithm_id", "solver:canonical-branch-and-bound-v2"),
    ("algorithm_version", "algorithm:2.9"),
    ("candidate_config.algorithm_id", "solver:canonical-branch-and-bound-v2"),
    ("candidate_config.algorithm_version", "algorithm:2.7"),
    (
        "candidate_config.so2_kernel_id",
        "geometry-kernel:rational-so2-upright-box-directed-v2",
    ),
    (
        "candidate_config.so2_kernel_version",
        "kernel:2.2-continuous-yaw-upright-box",
    ),
    (
        "candidate_config.obstacle_kernel_id",
        "geometry-kernel:rational-convex-translation-bracket-v2",
    ),
    (
        "candidate_config.obstacle_kernel_version",
        "kernel:2.3-convex-translation-bracket",
    ),
    (
        "candidate_config.partition_kernel_id",
        "geometry-kernel:rational-convex-complement-partition-v2",
    ),
    (
        "candidate_config.partition_kernel_version",
        "kernel:2.4-topology-aware-convex-complement",
    ),
    (
        "candidate_config.intersection_kernel_id",
        "geometry-kernel:rational-strict-convex-intersection-v2",
    ),
    (
        "candidate_config.intersection_kernel_version",
        "kernel:2.5-strict-convex-intersection",
    ),
    (
        "candidate_config.support_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-support-projection-v2",
    ),
    (
        "candidate_config.support_projection_kernel_version",
        "kernel:2.6-exact-horizontal-support-projection",
    ),
    (
        "camera_frame_kernel_id",
        "geometry-kernel:rational-upright-world-to-camera-v2.9",
    ),
    (
        "target_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-directional-relation-v2.9",
    ),
    (
        "visibility_projection_kernel_id",
        "geometry-kernel:rational-continuous-yaw-upright-camera-visibility-v2.9",
    ),
    (
        "objective_kernel_id",
        "objective-kernel:rational-continuous-yaw-joint-four-term-v2.9",
    ),
)
_NUMERIC_SEMANTICS_REF = "definition:spatialcf/planar-translate/numeric-policy/2.0"
_PROOF_MATERIAL_DEFINITION_REF = (
    "definition:spatialcf/planar-translate/proof-material-current-v2/2.0"
)
_PLANAR_TRANSLATE_SEMANTICS_SCHEMA_REFS = (
    "schema:spatialcf/canonical-scene/2.3",
    "schema:spatialcf/semantic-problem/2.3",
)
_PLANAR_TRANSLATE_ACTION_SCENE_SCHEMA_REFS = (
    "schema:spatialcf/canonical-scene/2.3",
)
_PLANAR_TRANSLATE_STATE_VARIABLE_REFS = (
    "definition:spatialcf/planar-translate/state-world-xy/2.0",
)
_PLANAR_TRANSLATE_OPERATOR_REFS = (
    "definition:spatialcf/planar-translate/operator-translate-xy/2.0",
)
_PLANAR_TRANSLATE_INVARIANT_REFS = (
    "definition:spatialcf/planar-translate/invariants-current-v2/2.0",
)
_PLANAR_TRANSLATE_CLAIM_REFS = (
    "definition:spatialcf/planar-translate/claim-certified-solution/2.0",
    "definition:spatialcf/planar-translate/claim-proven-unsat/2.0",
    "definition:spatialcf/planar-translate/claim-unknown/2.0",
)
_PLANAR_TRANSLATE_BACKEND_CAPABILITIES = (
    PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF,
    PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,
)
_PLANAR_TRANSLATE_PUBLICATION_PROOF_POLICY_REF = (
    "definition:spatialcf/planar-translate/publication-proof-policy/2.0"
)
_PLANAR_TRANSLATE_RESOURCE_REFS = (
    "definition:spatialcf/planar-translate/resource-cpu/2.0",
)
_PLANAR_TRANSLATE_SEMANTICS_PROFILE_SHA256 = (
    "7a1a4ea81f22862fc3789ab7d2fd062379f33dc0dea78c1959bb89228d39c1f2"
)
_PLANAR_TRANSLATE_ACTION_SPACE_PROFILE_SHA256 = (
    "152358ad4ebf7ca114a414a2e92dbeca38c8f88caacc46d16055846e56974935"
)
_SOURCE_ENTITY_ID = "entity:spatialcf/planar-translate-source"
_SOURCE_BINDING_FAMILY = "definition:spatialcf/planar-translate/source-bindings/2.0"
_SOURCE_DIGEST_SCHEMA = "schema:spatialcf/planar-translate/source-digest/2.0"
_MISSING_RESULT_STATUS = "MISSING_RESULT"
_DELEGATED_LOWER_BOUND_HASH_DOMAIN = (
    "spatialcf/counterfactual/planar-translate/delegated-objective-lower-bound/3.0"
)
_DELEGATED_UPPER_BOUND_HASH_DOMAIN = (
    "spatialcf/counterfactual/planar-translate/delegated-objective-upper-bound/3.0"
)
_MATCH_CLAIM_REF = "definition:spatialcf/planar-translate/filter/2.0"
_SELECTION_CLAIM_REF = "definition:spatialcf/planar-translate/filter/2.0"
_SELECTION_REASON_REF = "definition:spatialcf/planar-translate/order/2.0"
_CERTIFIED_CLAIM_REF = (
    "definition:spatialcf/planar-translate/claim-certified-solution/2.0"
)
_PROOF_PAYLOAD_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-proof-material/2.0"
)
_STATUS_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-status/2.0"
)
_RESULT_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-result-digest/2.0"
)
_CERTIFICATE_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-certificate-digest/2.0"
)
_PROOF_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-source-proof-digest/2.0"
)
_CANDIDATE_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-candidate-refs-digest/2.0"
)
_EDIT_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-selected-edit-digest/2.0"
)
_LOWER_BOUND_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-objective-lower-bound/2.0"
)
_UPPER_BOUND_SCHEMA_REF = (
    "schema:spatialcf/planar-translate/delegated-v2-objective-upper-bound/2.0"
)


class PlanarTranslateAlgorithmKernelBinding(CanonicalModel):
    """One literal current-v2 algorithm or kernel binding in mapping data."""

    field_name: CanonicalId
    registered_value: CanonicalId


class PlanarTranslateV2SourceArtifacts(HashBoundCanonicalModel):
    """Exact typed retained-v2 roots plus their raw and domain-separated hashes."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/planar-translate/source-artifacts/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "source_artifacts_sha256"

    problem: SemanticProblemV2_3
    problem_semantic_sha256: Sha256Digest
    problem_canonical_sha256: Sha256Digest
    config: ContinuousYawSolverConfigV2_9
    config_domain_sha256: Sha256Digest
    config_canonical_sha256: Sha256Digest
    source_artifacts_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        cls._require_exact_seal_inputs(values)
        if any(
            name in values
            for name in (
                "problem_semantic_sha256",
                "problem_canonical_sha256",
                "config_domain_sha256",
                "config_canonical_sha256",
            )
        ):
            raise ValueError("seal() derives every source-artifact digest")
        problem = values["problem"]
        config = values["config"]
        return super().seal(
            **(
                values
                | {
                    "problem_semantic_sha256": problem.semantic_problem_sha256,
                    "problem_canonical_sha256": _raw_canonical_sha256(problem),
                    "config_domain_sha256": config.config_sha256,
                    "config_canonical_sha256": _raw_canonical_sha256(config),
                }
            )
        )

    @classmethod
    def _require_exact_seal_inputs(cls, values: Mapping[str, object]) -> None:
        if type(values.get("problem")) is not SemanticProblemV2_3:
            raise TypeError("problem must be an exact SemanticProblemV2_3")
        if type(values.get("config")) is not ContinuousYawSolverConfigV2_9:
            raise TypeError("config must be an exact ContinuousYawSolverConfigV2_9")

    @model_validator(mode="after")
    def _validate_source_artifacts(self) -> Self:
        if type(self.problem) is not SemanticProblemV2_3:
            raise TypeError("problem must be an exact SemanticProblemV2_3")
        if type(self.config) is not ContinuousYawSolverConfigV2_9:
            raise TypeError("config must be an exact ContinuousYawSolverConfigV2_9")
        if self.problem_semantic_sha256 != self.problem.semantic_problem_sha256:
            raise ValueError("problem semantic digest does not match the exact source")
        if self.problem_canonical_sha256 != _raw_canonical_sha256(self.problem):
            raise ValueError("problem canonical digest does not match the exact source")
        if self.config_domain_sha256 != self.config.config_sha256:
            raise ValueError("config domain digest does not match the exact source")
        if self.config_canonical_sha256 != _raw_canonical_sha256(self.config):
            raise ValueError("config canonical digest does not match the exact source")
        return self


class PlanarTranslateFieldMapping(HashBoundCanonicalModel):
    """Literal retained-v2 field and kernel coverage for one mapping definition."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/planar-translate/field-mapping/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "field_mapping_sha256"

    mapping_definition_ref: DefinitionRef
    problem_field_names: tuple[CanonicalId, ...]
    scene_field_names: tuple[CanonicalId, ...]
    constraint_field_names: tuple[CanonicalId, ...]
    objective_field_names: tuple[CanonicalId, ...]
    numeric_policy_field_names: tuple[CanonicalId, ...]
    solver_config_field_names: tuple[CanonicalId, ...]
    candidate_config_field_names: tuple[CanonicalId, ...]
    algorithm_kernel_bindings: tuple[PlanarTranslateAlgorithmKernelBinding, ...]
    mapping_definition_sha256: Sha256Digest
    field_mapping_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        if "mapping_definition_sha256" in values:
            raise ValueError("seal() derives the mapping definition digest")
        payload = {
            name: value
            for name, value in values.items()
            if name != "field_mapping_sha256"
        }
        return super().seal(
            **(
                values
                | {
                    "mapping_definition_sha256": canonical_sha256(
                        payload,
                        domain=_MAPPING_DEFINITION_HASH_DOMAIN,
                    )
                }
            )
        )

    @model_validator(mode="after")
    def _validate_mapping(self) -> Self:
        if self.mapping_definition_ref != PLANAR_TRANSLATE_MAPPING_DEFINITION_REF:
            raise ValueError("mapping definition reference must be fixed")
        _require_exact_fields(self.problem_field_names, _PROBLEM_FIELD_NAMES, "problem")
        _require_exact_fields(self.scene_field_names, _SCENE_FIELD_NAMES, "scene")
        _require_exact_fields(
            self.constraint_field_names,
            _CONSTRAINT_FIELD_NAMES,
            "constraint",
        )
        _require_exact_fields(self.objective_field_names, _OBJECTIVE_FIELD_NAMES, "objective")
        _require_exact_fields(
            self.numeric_policy_field_names,
            _NUMERIC_POLICY_FIELD_NAMES,
            "numeric policy",
        )
        _require_exact_fields(
            self.solver_config_field_names,
            _SOLVER_CONFIG_FIELD_NAMES,
            "solver config",
        )
        _require_exact_fields(
            self.candidate_config_field_names,
            _CANDIDATE_CONFIG_FIELD_NAMES,
            "candidate config",
        )
        bindings = tuple(
            (binding.field_name, binding.registered_value)
            for binding in self.algorithm_kernel_bindings
        )
        if bindings != _ALGORITHM_KERNEL_BINDINGS:
            raise ValueError("algorithm and kernel bindings must be exact")
        declaration = self.model_dump(
            mode="python",
            exclude={"mapping_definition_sha256", "field_mapping_sha256"},
            round_trip=True,
        )
        if self.mapping_definition_sha256 != canonical_sha256(
            declaration,
            domain=_MAPPING_DEFINITION_HASH_DOMAIN,
        ):
            raise ValueError("mapping definition digest does not match its declaration")
        return self


class PlanarTranslateProfileRegistration(HashBoundCanonicalModel):
    """Data-only registration of the retained solver and checker identities."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/planar-translate/profile-registration/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "registration_sha256"

    semantics_profile: SemanticsProfile
    action_space_profile: ActionSpaceProfile
    backend_descriptor: SolverBackendDescriptor
    backend_owner_ref: OwnerRef
    checker_owner_ref: OwnerRef
    profile_capability_ref: CapabilityRef
    compiler_capability_ref: CapabilityRef
    solver_capability_ref: CapabilityRef
    checker_capability_ref: CapabilityRef
    mapping_definition_ref: DefinitionRef
    mapping_definition_sha256: Sha256Digest
    registration_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_registration(self) -> Self:
        semantics = self.semantics_profile
        action_space = self.action_space_profile
        descriptor = self.backend_descriptor
        if semantics.semantics_profile_ref != PLANAR_TRANSLATE_SEMANTICS_PROFILE_REF:
            raise ValueError("semantics profile reference must be fixed")
        _require_exact_values(
            semantics.accepted_scene_and_fact_schema_refs,
            _PLANAR_TRANSLATE_SEMANTICS_SCHEMA_REFS,
            "semantics profile schema",
        )
        _require_exact_values(
            semantics.predicate_definition_refs,
            (PLANAR_TRANSLATE_MAPPING_DEFINITION_REF,),
            "semantics profile predicate definition",
        )
        _require_exact_values(
            semantics.transition_semantics_refs,
            ("definition:spatialcf/planar-translate/transition-translate-xy/2.0",),
            "semantics profile transition",
        )
        _require_exact_values(
            semantics.objective_definition_refs,
            ("definition:spatialcf/planar-translate/objective-current-v2/2.0",),
            "semantics profile objective definition",
        )
        if semantics.numeric_semantics_ref != _NUMERIC_SEMANTICS_REF:
            raise ValueError("semantics profile numeric semantics must be fixed")
        if semantics.completeness_policy_ref != (
            "definition:spatialcf/planar-translate/completeness-policy/2.0"
        ):
            raise ValueError("semantics profile completeness policy must be fixed")
        if semantics.uncertainty_policy_ref != (
            "definition:spatialcf/planar-translate/uncertainty-policy/2.0"
        ):
            raise ValueError("semantics profile uncertainty policy must be fixed")
        _require_exact_values(
            semantics.derived_fact_rule_refs,
            (),
            "semantics profile derived fact rule",
        )
        if semantics.observation_obligation_policy_ref != (
            "definition:spatialcf/planar-translate/observation-obligation-policy/2.0"
        ):
            raise ValueError("semantics profile observation policy must be fixed")
        if (
            semantics.semantics_profile_sha256
            != _PLANAR_TRANSLATE_SEMANTICS_PROFILE_SHA256
        ):
            raise ValueError("semantics profile digest must match the immutable planar profile")

        if action_space.action_space_profile_ref != PLANAR_TRANSLATE_PROFILE_REF:
            raise ValueError("action-space profile reference must be fixed")
        _require_exact_values(
            action_space.accepted_scene_schema_refs,
            _PLANAR_TRANSLATE_ACTION_SCENE_SCHEMA_REFS,
            "action-space profile scene schema",
        )
        _require_exact_values(
            action_space.state_variable_definition_refs,
            _PLANAR_TRANSLATE_STATE_VARIABLE_REFS,
            "action-space profile state variable",
        )
        _require_exact_values(
            action_space.allowed_operator_refs,
            _PLANAR_TRANSLATE_OPERATOR_REFS,
            "action-space profile operator",
        )
        _require_exact_values(
            action_space.mandatory_invariant_template_refs,
            _PLANAR_TRANSLATE_INVARIANT_REFS,
            "action-space profile invariant",
        )
        _require_exact_values(
            action_space.predicate_capability_refs,
            (PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF,),
            "action-space profile predicate capability",
        )
        _require_exact_values(
            action_space.objective_capability_refs,
            (PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,),
            "action-space profile objective capability",
        )
        if action_space.numeric_semantics_ref != _NUMERIC_SEMANTICS_REF:
            raise ValueError("action-space profile numeric semantics must be fixed")
        _require_exact_values(
            action_space.allowed_claim_definition_refs,
            _PLANAR_TRANSLATE_CLAIM_REFS,
            "action-space profile claim",
        )
        _require_exact_values(
            action_space.backend_capability_requirements,
            _PLANAR_TRANSLATE_BACKEND_CAPABILITIES,
            "action-space profile backend capability",
        )
        _require_exact_values(
            action_space.adapter_capability_requirements,
            (),
            "action-space profile adapter capability",
        )
        if (
            action_space.publication_proof_policy_ref
            != _PLANAR_TRANSLATE_PUBLICATION_PROOF_POLICY_REF
        ):
            raise ValueError("action-space profile publication proof policy must be fixed")
        if (
            action_space.action_space_profile_sha256
            != _PLANAR_TRANSLATE_ACTION_SPACE_PROFILE_SHA256
        ):
            raise ValueError("action-space profile digest must match the immutable planar profile")

        if descriptor.backend_ref != PLANAR_TRANSLATE_BACKEND_REF:
            raise ValueError("backend reference must be fixed")
        if self.backend_owner_ref != PLANAR_TRANSLATE_BACKEND_OWNER_REF:
            raise ValueError("backend owner reference must be fixed")
        if self.checker_owner_ref != PLANAR_TRANSLATE_CHECKER_OWNER_REF:
            raise ValueError("checker owner reference must be fixed")
        if self.profile_capability_ref != PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF:
            raise ValueError("profile capability reference must be fixed")
        if self.compiler_capability_ref != PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF:
            raise ValueError("compiler capability reference must be fixed")
        if self.solver_capability_ref != PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF:
            raise ValueError("solver capability reference must be fixed")
        if self.checker_capability_ref != PLANAR_TRANSLATE_CHECKER_CAPABILITY_REF:
            raise ValueError("checker capability reference must be fixed")
        if self.mapping_definition_ref != PLANAR_TRANSLATE_MAPPING_DEFINITION_REF:
            raise ValueError("mapping definition reference must be fixed")
        if descriptor.supported_profile_hashes != (
            _PLANAR_TRANSLATE_ACTION_SPACE_PROFILE_SHA256,
        ):
            raise ValueError("backend descriptor must bind the exact action profile")
        _require_exact_values(
            descriptor.supported_predicate_capabilities,
            (PLANAR_TRANSLATE_PROFILE_CAPABILITY_REF,),
            "backend descriptor predicate capability",
        )
        _require_exact_values(
            descriptor.supported_operator_capabilities,
            (PLANAR_TRANSLATE_COMPILER_CAPABILITY_REF,),
            "backend descriptor operator capability",
        )
        _require_exact_values(
            descriptor.supported_objective_capabilities,
            (PLANAR_TRANSLATE_SOLVER_CAPABILITY_REF,),
            "backend descriptor objective capability",
        )
        _require_exact_values(
            descriptor.supported_numeric_semantics,
            (_NUMERIC_SEMANTICS_REF,),
            "backend descriptor numeric semantics",
        )
        _require_exact_values(
            descriptor.emitted_proof_material_definition_refs,
            (_PROOF_MATERIAL_DEFINITION_REF,),
            "backend descriptor proof material",
        )
        _require_exact_values(
            descriptor.compatible_checker_capability_refs,
            (PLANAR_TRANSLATE_CHECKER_CAPABILITY_REF,),
            "backend descriptor checker capability",
        )
        _require_exact_values(
            descriptor.resource_definition_refs,
            _PLANAR_TRANSLATE_RESOURCE_REFS,
            "backend descriptor resource",
        )
        return self


class PlanarTranslateCompatibilityLineage(HashBoundCanonicalModel):
    """One hash-only bridge between source v2 roots and M1 compilation roots."""

    HASH_DOMAIN: ClassVar[str] = "spatialcf/counterfactual/planar-translate/lineage/3.0"
    SELF_DIGEST_FIELD: ClassVar[str] = "lineage_sha256"

    source_artifacts_sha256: Sha256Digest
    source_problem_semantic_sha256: Sha256Digest
    source_problem_canonical_sha256: Sha256Digest
    source_config_domain_sha256: Sha256Digest
    source_config_canonical_sha256: Sha256Digest
    v3_semantic_problem_sha256: Sha256Digest
    v3_solve_request_sha256: Sha256Digest
    semantics_profile_sha256: Sha256Digest
    action_space_profile_sha256: Sha256Digest
    mapping_definition_sha256: Sha256Digest
    lineage_sha256: Sha256Digest


class PlanarTranslateCompilation(HashBoundCanonicalModel):
    """Closed assembly of source, mapping, profile, M1 roots, and lineage data."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/planar-translate/compilation/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "compilation_sha256"

    source_artifacts: PlanarTranslateV2SourceArtifacts
    field_mapping: PlanarTranslateFieldMapping
    registration: PlanarTranslateProfileRegistration
    semantic_problem: CounterfactualProblemIR
    solve_request: CounterfactualSolveRequest
    lineage: PlanarTranslateCompatibilityLineage
    compilation_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_compilation(self) -> Self:
        if type(self.source_artifacts) is not PlanarTranslateV2SourceArtifacts:
            raise TypeError("source artifacts must be an exact compatibility record")
        if type(self.field_mapping) is not PlanarTranslateFieldMapping:
            raise TypeError("field mapping must be an exact compatibility record")
        if type(self.registration) is not PlanarTranslateProfileRegistration:
            raise TypeError("registration must be an exact compatibility record")
        if type(self.semantic_problem) is not CounterfactualProblemIR:
            raise TypeError("semantic problem must be an exact CounterfactualProblemIR")
        if type(self.solve_request) is not CounterfactualSolveRequest:
            raise TypeError("solve request must be an exact CounterfactualSolveRequest")
        if type(self.lineage) is not PlanarTranslateCompatibilityLineage:
            raise TypeError("lineage must be an exact compatibility record")
        if self.lineage.source_artifacts_sha256 != self.source_artifacts.source_artifacts_sha256:
            raise ValueError("lineage source artifacts digest does not match")
        if self.lineage.source_problem_semantic_sha256 != (
            self.source_artifacts.problem_semantic_sha256
        ):
            raise ValueError("lineage source problem semantic digest does not match")
        if self.lineage.source_problem_canonical_sha256 != (
            self.source_artifacts.problem_canonical_sha256
        ):
            raise ValueError("lineage source problem canonical digest does not match")
        if self.lineage.source_config_domain_sha256 != (
            self.source_artifacts.config_domain_sha256
        ):
            raise ValueError("lineage source config domain digest does not match")
        if self.lineage.source_config_canonical_sha256 != (
            self.source_artifacts.config_canonical_sha256
        ):
            raise ValueError("lineage source config canonical digest does not match")
        if self.lineage.v3_semantic_problem_sha256 != (
            self.semantic_problem.semantic_problem_sha256
        ):
            raise ValueError("lineage semantic problem digest does not match")
        if self.lineage.v3_solve_request_sha256 != self.solve_request.solve_request_sha256:
            raise ValueError("lineage solve request digest does not match")
        if self.solve_request.semantic_problem_sha256 != (
            self.semantic_problem.semantic_problem_sha256
        ):
            raise ValueError("solve request semantic problem digest does not match")
        if self.semantic_problem.semantics_profile_ref != (
            self.registration.semantics_profile.semantics_profile_ref
        ):
            raise ValueError("semantic problem semantics profile must match registration")
        if self.semantic_problem.action_space_profile_ref != (
            self.registration.action_space_profile.action_space_profile_ref
        ):
            raise ValueError("semantic problem action-space profile must match registration")
        if self.semantic_problem.numeric_semantics_ref != (
            self.registration.semantics_profile.numeric_semantics_ref
        ):
            raise ValueError("semantic problem numeric semantics must match registration")
        if self.lineage.semantics_profile_sha256 != (
            self.registration.semantics_profile.semantics_profile_sha256
        ):
            raise ValueError("lineage semantics profile digest does not match")
        if self.lineage.action_space_profile_sha256 != (
            self.registration.action_space_profile.action_space_profile_sha256
        ):
            raise ValueError("lineage action-space profile digest does not match")
        if self.lineage.mapping_definition_sha256 != (
            self.field_mapping.mapping_definition_sha256
        ):
            raise ValueError("lineage mapping definition digest does not match")
        if self.registration.mapping_definition_ref != self.field_mapping.mapping_definition_ref:
            raise ValueError("registration mapping reference does not match")
        if self.registration.mapping_definition_sha256 != (
            self.field_mapping.mapping_definition_sha256
        ):
            raise ValueError("registration mapping digest does not match")
        expected_source_bundle = _derive_planar_translate_source_fact_bundle(
            source_artifacts=self.source_artifacts,
            field_mapping=self.field_mapping,
            registration=self.registration,
        )
        if self.semantic_problem.scene_state.extension_fact_bundles != (
            expected_source_bundle,
        ):
            raise ValueError(
                "materialized source facts do not match the exact source derivation"
            )
        return self


class PlanarTranslateDelegatedOutcome(HashBoundCanonicalModel):
    """One hash-bound retained-v2 result without inventing an M1 terminal claim.

    The sealed M1 :class:`BackendProposal` has mandatory finite objective bounds.
    It can therefore accompany only a retained-v2 certified-success result.  This
    compatibility record carries the exact source outcome and its branch-specific
    evidence for every retained-v2 terminal state without making a v3 certificate
    or creating an artificial finite objective for UNSAT/uncertified branches.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/planar-translate/delegated-outcome/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "delegated_outcome_sha256"

    compilation: PlanarTranslateCompilation
    compilation_sha256: Sha256Digest
    source_artifacts_sha256: Sha256Digest
    source_problem_semantic_sha256: Sha256Digest
    source_problem_canonical_sha256: Sha256Digest
    source_config_domain_sha256: Sha256Digest
    source_config_canonical_sha256: Sha256Digest
    v3_semantic_problem_sha256: Sha256Digest
    v3_solve_request_sha256: Sha256Digest
    backend_ref: BackendRef
    backend_owner_ref: OwnerRef
    backend_build_sha256: Sha256Digest
    backend_selection_record: BackendSelectionRecord
    backend_selection_record_sha256: Sha256Digest
    proof_material: ProofMaterialEnvelope
    proof_material_sha256: Sha256Digest
    resource_usage: ResourceUsage
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9
    source_status: ContinuousYawSolveStatusV2 | Literal["MISSING_RESULT"]
    source_result_sha256: Sha256Digest | None = None
    source_certificate_sha256: Sha256Digest | None = None
    source_proof_sha256: Sha256Digest | None = None
    source_candidate_refs_sha256: Sha256Digest | None = None
    source_selected_edit_sha256: Sha256Digest | None = None
    objective_lower_bound: NonNegativeFiniteFloat | None = None
    objective_upper_bound: NonNegativeFiniteFloat | None = None
    objective_lower_bound_sha256: Sha256Digest | None = None
    objective_upper_bound_sha256: Sha256Digest | None = None
    backend_proposal: BackendProposal | None = None
    delegated_outcome_sha256: Sha256Digest

    @classmethod
    def seal(cls, **values) -> Self:
        """Derive optional bound hashes only when source success supplies bounds."""

        lower = values.get("objective_lower_bound")
        upper = values.get("objective_upper_bound")
        supplied_hashes = {
            name
            for name in (
                "objective_lower_bound_sha256",
                "objective_upper_bound_sha256",
            )
            if name in values
        }
        if lower is None and upper is None:
            if supplied_hashes:
                raise ValueError("absent objective bounds cannot carry bound hashes")
            return super().seal(**values)
        if lower is None or upper is None:
            raise ValueError("delegated objective bounds must appear together")
        if supplied_hashes:
            raise ValueError("seal() derives delegated objective-bound hashes")
        return super().seal(
            **(
                values
                | {
                    "objective_lower_bound_sha256": _delegated_lower_bound_sha256(
                        lower
                    ),
                    "objective_upper_bound_sha256": _delegated_upper_bound_sha256(
                        upper
                    ),
                }
            )
        )

    @model_validator(mode="after")
    def _validate_delegated_outcome(self) -> Self:
        _require_exact_strict_round_trip(
            self.compilation,
            PlanarTranslateCompilation,
            "compilation",
        )
        _require_exact_strict_round_trip(
            self.source_outcome,
            ContinuousYawMinimumCostSolveOutcomeV2_9,
            "source outcome",
        )
        _require_exact_strict_round_trip(
            self.backend_selection_record,
            BackendSelectionRecord,
            "backend selection record",
        )
        _require_exact_strict_round_trip(
            self.proof_material,
            ProofMaterialEnvelope,
            "proof material",
        )
        _require_exact_strict_round_trip(
            self.resource_usage,
            ResourceUsage,
            "resource usage",
        )
        compilation = self.compilation
        source = compilation.source_artifacts
        descriptor = compilation.registration.backend_descriptor
        if (
            self.compilation_sha256 != compilation.compilation_sha256
            or self.source_artifacts_sha256 != source.source_artifacts_sha256
            or self.source_problem_semantic_sha256 != source.problem_semantic_sha256
            or self.source_problem_canonical_sha256 != source.problem_canonical_sha256
            or self.source_config_domain_sha256 != source.config_domain_sha256
            or self.source_config_canonical_sha256 != source.config_canonical_sha256
            or self.v3_semantic_problem_sha256
            != compilation.semantic_problem.semantic_problem_sha256
            or self.v3_solve_request_sha256 != compilation.solve_request.solve_request_sha256
            or self.backend_ref != descriptor.backend_ref
            or self.backend_owner_ref != compilation.registration.backend_owner_ref
            or self.backend_build_sha256 != descriptor.implementation_build_sha256
        ):
            raise ValueError("delegated outer bindings must match the compilation")

        _validate_delegated_source_outcome(compilation, self.source_outcome)
        evidence = _delegated_source_evidence(self.source_outcome)
        expected_lower = evidence["objective_lower_bound"]
        expected_upper = evidence["objective_upper_bound"]
        expected_lower_digest = (
            None
            if expected_lower is None
            else _delegated_lower_bound_sha256(expected_lower)
        )
        expected_upper_digest = (
            None
            if expected_upper is None
            else _delegated_upper_bound_sha256(expected_upper)
        )
        if (
            self.source_status != evidence["status"]
            or self.source_result_sha256 != evidence["result_sha256"]
            or self.source_certificate_sha256 != evidence["certificate_sha256"]
            or self.source_proof_sha256 != evidence["proof_sha256"]
            or self.source_candidate_refs_sha256 != evidence["candidate_refs_sha256"]
            or self.source_selected_edit_sha256 != evidence["selected_edit_sha256"]
            or self.objective_lower_bound != expected_lower
            or self.objective_upper_bound != expected_upper
            or self.objective_lower_bound_sha256 != expected_lower_digest
            or self.objective_upper_bound_sha256 != expected_upper_digest
        ):
            raise ValueError("delegated source evidence must match the source outcome")

        expected_resource_usage = _delegated_resource_usage(
            compilation,
            self.source_outcome,
        )
        _require_exact_canonical_value(
            self.resource_usage,
            expected_resource_usage,
            "delegated resource usage",
        )
        expected_selection = _delegated_selection_record(
            compilation,
            expected_resource_usage,
        )
        _require_exact_canonical_value(
            self.backend_selection_record,
            expected_selection,
            "delegated backend selection record",
        )
        if (
            self.backend_selection_record_sha256
            != expected_selection.backend_selection_record_sha256
        ):
            raise ValueError("delegated backend selection digest does not match")
        expected_proof = _delegated_proof_material(
            compilation,
            expected_selection,
            evidence,
        )
        _require_exact_canonical_value(
            self.proof_material,
            expected_proof,
            "delegated proof material",
        )
        if self.proof_material_sha256 != expected_proof.proof_material_sha256:
            raise ValueError("delegated proof material digest does not match")
        expected_proposal = _delegated_certified_success_proposal(
            compilation,
            expected_selection,
            expected_proof,
            expected_resource_usage,
            evidence,
        )
        if expected_proposal is None:
            if self.backend_proposal is not None:
                raise ValueError("non-success delegated outcome cannot carry a proposal")
        else:
            if type(self.backend_proposal) is not BackendProposal:
                raise ValueError("delegated success requires one existing backend proposal")
            _require_exact_strict_round_trip(
                self.backend_proposal,
                BackendProposal,
                "backend proposal",
            )
            _require_exact_canonical_value(
                self.backend_proposal,
                expected_proposal,
                "delegated success proposal",
            )
        return self


def _require_exact_strict_round_trip(value: object, model_type: type, label: str) -> None:
    if type(value) is not model_type:
        raise TypeError(f"{label} must be an exact {model_type.__name__}")
    try:
        checked = model_type.model_validate(
            value.model_dump(mode="python", round_trip=True, warnings=False),
            strict=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must pass strict canonical validation") from error
    if canonical_json_bytes(checked) != canonical_json_bytes(value):
        raise ValueError(f"{label} strict canonical bytes do not round trip")


def _delegated_lower_bound_sha256(value: NonNegativeFiniteFloat) -> Sha256Digest:
    return canonical_sha256(value, domain=_DELEGATED_LOWER_BOUND_HASH_DOMAIN)


def _delegated_upper_bound_sha256(value: NonNegativeFiniteFloat) -> Sha256Digest:
    return canonical_sha256(value, domain=_DELEGATED_UPPER_BOUND_HASH_DOMAIN)


def _require_exact_canonical_value(
    observed: object,
    expected: object,
    label: str,
) -> None:
    if type(observed) is not type(expected):
        raise TypeError(f"{label} must be an exact {type(expected).__name__}")
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError(f"{label} does not match the retained derivation")


def _validate_delegated_source_outcome(
    compilation: PlanarTranslateCompilation,
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
) -> None:
    result = source_outcome.result
    if result is None:
        return
    if type(result) not in {
        ContinuousYawCertifiedSuccessResultV2_9,
        ContinuousYawProvenUnsatResultV2_9,
        ContinuousYawUncertifiedResultV2_9,
    }:
        raise TypeError("source result must be an exact current-v2 terminal type")
    _require_exact_strict_round_trip(result, type(result), "source result")
    source = compilation.source_artifacts
    if (
        result.semantic_problem_sha256 != source.problem_semantic_sha256
        or canonical_json_bytes(result.solver_config)
        != canonical_json_bytes(source.config)
    ):
        raise ValueError("source result does not bind the retained source")


def _delegated_source_evidence(
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
) -> dict[str, object | None]:
    result = source_outcome.result
    if result is None:
        return {
            "status": _MISSING_RESULT_STATUS,
            "result_sha256": None,
            "certificate_sha256": None,
            "proof_sha256": None,
            "candidate_refs_sha256": None,
            "selected_edit_sha256": None,
            "objective_lower_bound": None,
            "objective_upper_bound": None,
        }
    candidate_refs = result.candidate_refs
    evidence: dict[str, object | None] = {
        "status": result.status,
        "result_sha256": result.solve_result_sha256,
        "certificate_sha256": None,
        "proof_sha256": None,
        "candidate_refs_sha256": (
            None if candidate_refs is None else candidate_refs.candidate_refs_sha256
        ),
        "selected_edit_sha256": None,
        "objective_lower_bound": None,
        "objective_upper_bound": None,
    }
    if type(result) is ContinuousYawCertifiedSuccessResultV2_9:
        return evidence | {
            "certificate_sha256": result.certificate.certificate_sha256,
            "selected_edit_sha256": result.selected_witness.edit.edit_sha256,
            "objective_lower_bound": result.global_loss_lower_bound,
            "objective_upper_bound": result.witness_loss_bounds.total_upper_bound,
        }
    if type(result) is ContinuousYawProvenUnsatResultV2_9:
        return evidence | {"proof_sha256": result.empty_outer_stage_sha256}
    assert type(result) is ContinuousYawUncertifiedResultV2_9
    return evidence


def _lossless_m1_resource_used(domain_operations: int) -> float:
    used = float(domain_operations)
    if int(used) != domain_operations:
        raise ValueError("source resource usage is not losslessly representable in M1")
    return used


def _delegated_resource_usage(
    compilation: PlanarTranslateCompilation,
    source_outcome: ContinuousYawMinimumCostSolveOutcomeV2_9,
) -> ResourceUsage:
    resource_limit = compilation.solve_request.resource_policy.limits[0]
    source_usage = source_outcome.cumulative_generation_usage
    used = (
        0.0
        if source_usage is None
        else _lossless_m1_resource_used(source_usage.domain_operations)
    )
    return ResourceUsage(
        accounting_claim_definition_ref=(
            compilation.solve_request.resource_policy.exhaustion_claim_ref
        ),
        entries=(
            {
                "resource_definition_ref": resource_limit.definition_ref,
                "used": used,
            },
        ),
        exhausted=used == resource_limit.finite_limit,
    )


def _delegated_selection_record(
    compilation: PlanarTranslateCompilation,
    resource_usage: ResourceUsage,
) -> BackendSelectionRecord:
    request = compilation.solve_request
    registration = compilation.registration
    descriptor = registration.backend_descriptor
    match = CapabilityMatch(
        backend_ref=descriptor.backend_ref,
        backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        matched_capability_refs=tuple(
            sorted(
                (
                    registration.compiler_capability_ref,
                    registration.solver_capability_ref,
                ),
                key=canonical_json_bytes,
            )
        ),
        match_claim_definition_ref=_MATCH_CLAIM_REF,
    )
    return BackendSelectionRecord.seal(
        semantic_problem_sha256=request.semantic_problem_sha256,
        solve_request_sha256=request.solve_request_sha256,
        implementation_registry_snapshot_sha256=(
            request.implementation_registry_snapshot.implementation_registry_snapshot_sha256
        ),
        backend_descriptor_bundle_sha256=(
            request.backend_descriptor_bundle.backend_descriptor_bundle_sha256
        ),
        backend_routing_policy_sha256=(
            request.backend_routing_policy.backend_routing_policy_sha256
        ),
        ordered_candidate_backend_refs=(descriptor.backend_ref,),
        capability_rows=(match,),
        selection_disposition="SELECTED",
        selection_disposition_claim_ref=_SELECTION_CLAIM_REF,
        selected_backend_ref=descriptor.backend_ref,
        selected_backend_descriptor_sha256=descriptor.backend_descriptor_sha256,
        resource_allocation=resource_usage,
        deterministic_selection_reason_ref=_SELECTION_REASON_REF,
    )


def _delegated_proof_material(
    compilation: PlanarTranslateCompilation,
    selection: BackendSelectionRecord,
    evidence: dict[str, object | None],
) -> ProofMaterialEnvelope:
    payload = [
        TypedValue(
            value_schema_ref=_STATUS_SCHEMA_REF,
            payload=EnumSymbolValue(symbol=str(evidence["status"])),
        )
    ]
    for schema_ref, key in (
        (_RESULT_SCHEMA_REF, "result_sha256"),
        (_CERTIFICATE_SCHEMA_REF, "certificate_sha256"),
        (_PROOF_SCHEMA_REF, "proof_sha256"),
        (_CANDIDATE_SCHEMA_REF, "candidate_refs_sha256"),
        (_EDIT_SCHEMA_REF, "selected_edit_sha256"),
    ):
        digest = evidence[key]
        if digest is not None:
            payload.append(
                TypedValue(
                    value_schema_ref=schema_ref,
                    payload=DigestValue(value=digest),
                )
            )
    for schema_ref, key in (
        (_LOWER_BOUND_SCHEMA_REF, "objective_lower_bound"),
        (_UPPER_BOUND_SCHEMA_REF, "objective_upper_bound"),
    ):
        bound = evidence[key]
        if bound is not None:
            payload.append(
                TypedValue(
                    value_schema_ref=schema_ref,
                    payload=FiniteRealValue(value=bound),
                )
            )
    return ProofMaterialEnvelope.seal(
        semantic_problem_sha256=selection.semantic_problem_sha256,
        solve_request_sha256=selection.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=compilation.registration.backend_descriptor.backend_ref,
        proof_material_definition_ref=(
            compilation.registration.backend_descriptor.emitted_proof_material_definition_refs[
                0
            ]
        ),
        payload_schema_ref=_PROOF_PAYLOAD_SCHEMA_REF,
        typed_payload=tuple(sorted(payload, key=canonical_json_bytes)),
        artifact_refs=(),
    )


def _delegated_certified_success_proposal(
    compilation: PlanarTranslateCompilation,
    selection: BackendSelectionRecord,
    proof_material: ProofMaterialEnvelope,
    resource_usage: ResourceUsage,
    evidence: dict[str, object | None],
) -> BackendProposal | None:
    if evidence["status"] is not ContinuousYawSolveStatusV2.CERTIFIED_SUCCESS:
        return None
    lower = evidence["objective_lower_bound"]
    upper = evidence["objective_upper_bound"]
    assert type(lower) is float and type(upper) is float
    descriptor = compilation.registration.backend_descriptor
    return BackendProposal.seal(
        semantic_problem_sha256=selection.semantic_problem_sha256,
        solve_request_sha256=selection.solve_request_sha256,
        backend_selection_record_sha256=selection.backend_selection_record_sha256,
        proposal_backend_ref=descriptor.backend_ref,
        proposal_backend_owner_ref=compilation.registration.backend_owner_ref,
        proposal_backend_capability_ref=compilation.registration.solver_capability_ref,
        proposal_backend_build_sha256=descriptor.implementation_build_sha256,
        proposal_claim_definition_ref=_CERTIFIED_CLAIM_REF,
        proof_material=proof_material,
        proof_material_sha256=proof_material.proof_material_sha256,
        objective_lower_bound=lower,
        objective_upper_bound=upper,
        resource_usage=resource_usage,
    )


def _derive_planar_translate_source_fact_bundle(
    *,
    source_artifacts: PlanarTranslateV2SourceArtifacts,
    field_mapping: PlanarTranslateFieldMapping,
    registration: PlanarTranslateProfileRegistration,
) -> ExtensionFactBundle:
    """Derive the complete source-fact materialization for one exact embedding."""

    problem = source_artifacts.problem
    config = source_artifacts.config
    facts: list[ExtensionFact] = []
    groups = (
        ("problem", problem, field_mapping.problem_field_names),
        ("scene", problem.scene, field_mapping.scene_field_names),
        ("constraints", problem.constraints, field_mapping.constraint_field_names),
        ("objective", problem.objective, field_mapping.objective_field_names),
        (
            "numeric-policy",
            problem.numeric_policy,
            field_mapping.numeric_policy_field_names,
        ),
        ("solver-config", config, field_mapping.solver_config_field_names),
        (
            "candidate-config",
            config.candidate_config,
            field_mapping.candidate_config_field_names,
        ),
    )
    for group, owner, field_names in groups:
        family = f"definition:spatialcf/planar-translate/source-{group}-fields/2.0"
        for field_name in field_names:
            facts.append(
                _source_digest_fact(
                    family,
                    f"field:spatialcf/planar-translate/{group}/{field_name}",
                    _source_component_sha256(getattr(owner, field_name)),
                )
            )

    bindings = (
        ("source-problem-semantic-sha256", source_artifacts.problem_semantic_sha256),
        ("source-problem-canonical-sha256", source_artifacts.problem_canonical_sha256),
        ("source-config-domain-sha256", source_artifacts.config_domain_sha256),
        ("source-config-canonical-sha256", source_artifacts.config_canonical_sha256),
        ("source-artifacts-sha256", source_artifacts.source_artifacts_sha256),
        ("mapping-definition-sha256", field_mapping.mapping_definition_sha256),
        (
            "semantics-profile-sha256",
            registration.semantics_profile.semantics_profile_sha256,
        ),
        (
            "action-space-profile-sha256",
            registration.action_space_profile.action_space_profile_sha256,
        ),
        ("registration-sha256", registration.registration_sha256),
    )
    facts.extend(
        _source_digest_fact(
            _SOURCE_BINDING_FAMILY,
            f"binding:spatialcf/planar-translate/{name}",
            digest,
        )
        for name, digest in bindings
    )
    return ExtensionFactBundle.seal(
        facts=tuple(sorted(facts, key=canonical_json_bytes))
    )


def _source_digest_fact(family: str, fact_key: str, digest: str) -> ExtensionFact:
    return ExtensionFact(
        fact_family_ref=family,
        subject_entity_id=_SOURCE_ENTITY_ID,
        fact_key=fact_key,
        value=TypedValue(
            value_schema_ref=_SOURCE_DIGEST_SCHEMA,
            payload=DigestValue(value=digest),
        ),
    )


def _source_component_sha256(value: object) -> Sha256Digest:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_canonical_sha256(value: CanonicalModel) -> Sha256Digest:
    return _source_component_sha256(value)


def _require_exact_fields(
    observed: tuple[CanonicalId, ...],
    expected: tuple[str, ...],
    label: str,
) -> None:
    if observed != expected:
        raise ValueError(f"{label} field coverage must be exact")


def _require_exact_values(
    observed: tuple[CanonicalId, ...],
    expected: tuple[str, ...],
    label: str,
) -> None:
    if observed != expected:
        raise ValueError(f"{label} closure must be exact")
