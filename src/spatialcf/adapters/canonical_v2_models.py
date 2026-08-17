"""Strict value model returned by Canonical v2 scene-fact adapters.

This module contains only the Canonical scene and provenance contract. Request
policy, constraints, objectives, solver configuration, and platform runtimes
stay outside this facts seam.
"""

from __future__ import annotations

from pydantic import model_validator

from spatialcf.domain.v2.base import FactAvailabilityV2, FactSetV2, V2Model
from spatialcf.domain.v2.evidence import (
    MappingProofKindV2,
    MappingProofStatusV2,
    NativeObjectBindingV2,
    PreSemanticEvidenceEnvelopeV2,
)
from spatialcf.domain.v2.scene import CanonicalSceneV2


class CanonicalSceneAdaptationV2(V2Model):
    """Strict, frozen output of one native-to-Canonical fact translation.

    Every Canonical object has exactly one native binding. Each binding is
    backed by exactly one verified ``ENTITY_IDENTITY`` proof whose Canonical ID
    and native locator match exactly; identity proofs without a binding are
    forbidden. Every known non-object scene fact likewise has verified proof
    coverage in the matching facts family.
    """

    scene: CanonicalSceneV2
    pre_semantic_evidence: PreSemanticEvidenceEnvelopeV2
    bindings: tuple[NativeObjectBindingV2, ...]

    @classmethod
    def _scene_model_type(cls) -> type[CanonicalSceneV2]:
        """Return the root scene type owned by this versioned seam."""

        return CanonicalSceneV2

    @model_validator(mode="after")
    def strictly_revalidate_and_close_bindings(self) -> CanonicalSceneAdaptationV2:
        # Nested v2 models already opt into instance revalidation. Revalidating
        # explicitly here also documents and preserves this public trust
        # boundary if their construction path changes later.
        scene = type(self)._scene_model_type().model_validate(self.scene, strict=True)
        evidence = PreSemanticEvidenceEnvelopeV2.model_validate(
            self.pre_semantic_evidence,
            strict=True,
        )
        bindings = tuple(
            NativeObjectBindingV2.model_validate(binding, strict=True)
            for binding in self.bindings
        )

        canonical_ids = tuple(binding.canonical_object_id for binding in bindings)
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("Canonical object binding IDs must be unique")
        native_locator_keys = tuple(
            (
                binding.native_object_locator.kind.value,
                binding.native_object_locator.value,
            )
            for binding in bindings
        )
        if len(set(native_locator_keys)) != len(native_locator_keys):
            raise ValueError("native object binding locators must be unique")
        proof_ids = tuple(binding.mapping_proof_id for binding in bindings)
        if len(set(proof_ids)) != len(proof_ids):
            raise ValueError("native object binding proof IDs must be unique")

        object_ids = tuple(item.object_id for item in (scene.objects.values or ()))
        if set(canonical_ids) != set(object_ids):
            missing = sorted(set(object_ids).difference(canonical_ids))
            extra = sorted(set(canonical_ids).difference(object_ids))
            raise ValueError(
                "bindings must cover every scene object exactly once "
                f"(missing={missing}, extra={extra})"
            )

        mapping_proofs = {proof.proof_id: proof for proof in evidence.mapping_proofs}
        expected_ids_by_kind = _expected_fact_ids_by_proof_kind(scene)
        verified_ids_by_kind = {kind: set() for kind in expected_ids_by_kind}
        unsupported_kinds = {
            MappingProofKindV2.RELATION_NORMALIZATION,
            MappingProofKindV2.WRITE_BACK,
        }
        for proof in evidence.mapping_proofs:
            if proof.kind in unsupported_kinds:
                raise ValueError(
                    f"{proof.kind.value} proof is outside the scene-facts seam"
                )
            expected_ids = expected_ids_by_kind[proof.kind]
            dangling_ids = set(proof.canonical_ids).difference(expected_ids)
            if dangling_ids:
                raise ValueError(
                    f"{proof.kind.value} proof references wrong-family or unknown "
                    "Canonical facts: " + ", ".join(sorted(dangling_ids))
                )
            if proof.status is MappingProofStatusV2.VERIFIED:
                verified_ids_by_kind[proof.kind].update(proof.canonical_ids)

        for kind, expected_ids in expected_ids_by_kind.items():
            uncovered_ids = expected_ids.difference(verified_ids_by_kind[kind])
            if uncovered_ids:
                raise ValueError(
                    f"KNOWN {kind.value} scene facts lack VERIFIED proof coverage: "
                    + ", ".join(sorted(uncovered_ids))
                )

        identity_proof_ids = {
            proof.proof_id
            for proof in evidence.mapping_proofs
            if proof.kind is MappingProofKindV2.ENTITY_IDENTITY
        }
        consumed_identity_proof_ids: set[str] = set()
        for binding in bindings:
            proof = mapping_proofs.get(binding.mapping_proof_id)
            if proof is None:
                raise ValueError(
                    "native object binding references unknown mapping proof"
                )
            if (
                proof.kind is not MappingProofKindV2.ENTITY_IDENTITY
                or proof.status is not MappingProofStatusV2.VERIFIED
            ):
                raise ValueError(
                    "native object binding requires a VERIFIED ENTITY_IDENTITY proof"
                )
            if proof.canonical_ids != (binding.canonical_object_id,):
                raise ValueError(
                    "binding Canonical object must exactly match its identity proof"
                )
            if proof.native_locators != (binding.native_object_locator,):
                raise ValueError(
                    "binding native locator must exactly match its identity proof"
                )
            consumed_identity_proof_ids.add(proof.proof_id)

        orphan_identity_proofs = identity_proof_ids.difference(
            consumed_identity_proof_ids
        )
        if orphan_identity_proofs:
            raise ValueError(
                "orphan ENTITY_IDENTITY proofs have no object binding: "
                + ", ".join(sorted(orphan_identity_proofs))
            )

        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "pre_semantic_evidence", evidence)
        object.__setattr__(
            self,
            "bindings",
            tuple(sorted(bindings, key=lambda item: item.canonical_object_id)),
        )
        return self


def _known_fact_ids(facts: FactSetV2[V2Model], id_field: str) -> set[str]:
    if facts.availability is not FactAvailabilityV2.KNOWN:
        return set()
    values = (
        facts.values
        if facts.values is not None
        else (*(facts.inner_values or ()), *(facts.outer_values or ()))
    )
    return {getattr(value, id_field) for value in values}


def _expected_fact_ids_by_proof_kind(
    scene: CanonicalSceneV2,
) -> dict[MappingProofKindV2, set[str]]:
    return {
        MappingProofKindV2.ENTITY_IDENTITY: _known_fact_ids(
            scene.objects,
            "object_id",
        ),
        MappingProofKindV2.GEOMETRY: set().union(
            _known_fact_ids(scene.geometry_instances, "geometry_id"),
            _known_fact_ids(scene.collision_bodies, "body_id"),
            _known_fact_ids(scene.workspace_boundaries, "fact_id"),
            _known_fact_ids(scene.known_free_spaces, "fact_id"),
        ),
        MappingProofKindV2.SUPPORT: _known_fact_ids(
            scene.support_surfaces,
            "surface_id",
        ),
        MappingProofKindV2.CAMERA: _known_fact_ids(scene.cameras, "camera_id"),
        MappingProofKindV2.OBSERVATION_NORMALIZATION: _known_fact_ids(
            scene.baseline_observations,
            "observation_id",
        ),
    }
