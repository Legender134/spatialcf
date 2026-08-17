"""Bind trusted Canonical v2 scene provenance to a complete semantic problem."""

from __future__ import annotations

import warnings

from spatialcf.adapters.canonical_v2 import CanonicalSceneAdaptationV2
from spatialcf.domain.v2.evidence import EvidenceEnvelopeV2
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.serialization import canonical_json_bytes_v2


def bind_canonical_scene_evidence_v2(
    adaptation: CanonicalSceneAdaptationV2,
    problem: SemanticProblemV2,
    /,
) -> EvidenceEnvelopeV2:
    """Bind provenance only when the problem contains the adapted scene exactly."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        trusted_adaptation = CanonicalSceneAdaptationV2.model_validate(
            adaptation,
            strict=True,
        )
        trusted_problem = SemanticProblemV2.model_validate(problem, strict=True)
        adapted_scene_bytes = canonical_json_bytes_v2(trusted_adaptation.scene)
        problem_scene_bytes = canonical_json_bytes_v2(trusted_problem.scene)
        if adapted_scene_bytes != problem_scene_bytes:
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
