"""Bind Canonical 2.1 scene provenance to an exact matching problem."""

from __future__ import annotations

import warnings

from spatialcf.adapters.canonical_v2_1 import CanonicalSceneAdaptationV2_1
from spatialcf.domain.v2.cardinal import SemanticProblemV2_1
from spatialcf.domain.v2.evidence import EvidenceEnvelopeV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


def bind_canonical_scene_evidence_v2_1(
    adaptation: CanonicalSceneAdaptationV2_1,
    problem: SemanticProblemV2_1,
    /,
) -> EvidenceEnvelopeV2:
    """Bind provenance iff the 2.1 problem embeds the adapted scene exactly."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        trusted_adaptation = CanonicalSceneAdaptationV2_1.model_validate(
            adaptation,
            strict=True,
        )
        trusted_problem = SemanticProblemV2_1.model_validate(problem, strict=True)
        if canonical_json_bytes_v2(trusted_adaptation.scene) != canonical_json_bytes_v2(
            trusted_problem.scene
        ):
            raise ValueError(
                "semantic problem scene does not match adapted Canonical scene"
            )

        pre = trusted_adaptation.pre_semantic_evidence
        return EvidenceEnvelopeV2(
            semantic_problem_sha256=trusted_problem.semantic_problem_sha256,
            adapter=pre.adapter,
            source=pre.source,
            native_object_bindings=trusted_adaptation.bindings,
            raw_evidence_refs=pre.raw_evidence_refs,
            mapping_proofs=pre.mapping_proofs,
            runtime_identities=(),
            final_audits=(),
        )
