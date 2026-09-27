"""Explicit source campaign semantic owner."""

from __future__ import annotations

import hashlib

import json

import logging

import os

from collections.abc import (
    Callable,
    Iterator,
)

from contextlib import (
    ExitStack,
    contextmanager,
)

from dataclasses import (
    dataclass,
)

from pathlib import (
    Path,
)

from typing import (
    Literal,
)

from spatialcf.core.upright_se2_compiler import (
    compile_upright_se2,
)

from spatialcf.domain import (
    contrast as semantic,
)

from spatialcf.domain.constraints import (
    Relation as SemanticRelation,
)

from spatialcf.domain.counterfactual import (
    CounterfactualSolveRequest,
)

from spatialcf.domain.lineage import (
    RuntimeProvenance,
    SemanticObjectReference,
)

from spatialcf.domain.outcomes import (
    TypedCompilationOutcome,
)

from spatialcf.domain.upright_se2 import (
    UprightSE2Compilation,
    UprightSE2ContinuousCompilation,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.verification.filesystem import (
    bound_absolute_directory,
    bound_child_directory,
    read_regular_at,
    revalidate_entries,
)

from spatialcf.verification.split import (
    assign_split,
)

from spatialcf.generation.planning.campaign_contracts import (
    _MAX_PLAN_BYTES,
)


_SemanticCompilation = UprightSE2Compilation | UprightSE2ContinuousCompilation | TypedCompilationOutcome


_SEMANTIC_ELIGIBILITY_CLAUSE = "definition:spatialcf/semantic-contrast/opposite-target/1.0"


@dataclass(frozen=True)
class _SemanticCatalogPlan:
    catalog: semantic.SemanticContrastCatalog
    catalog_input: semantic.SemanticContrastCatalogInput
    normalized_input: bytes
    source_payloads: tuple[tuple[str, bytes], ...]
    compilations: tuple[_SemanticCompilation, ...]


def _semantic_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate semantic catalog JSON key")
        result[key] = value
    return result


def _normalize_semantic_catalog(payload: bytes) -> semantic.SemanticContrastCatalogInput:
    raw = json.loads(payload, object_pairs_hook=_semantic_json_object)
    if canonical_json_bytes(raw) != payload:
        raise ValueError("semantic catalog transport must be canonical JSON")
    if type(raw) is not dict or type(raw.get("sources")) is not list or type(raw.get("candidates")) is not list:
        raise ValueError("semantic catalog requires source and candidate arrays")
    raw["sources"].sort(key=lambda value: canonical_json_bytes(value["identity"]))
    raw["candidates"].sort(key=lambda value: (
        canonical_json_bytes(value["source_identity"]),
        canonical_json_bytes(value["candidate_id"]),
    ))
    normalized = canonical_json_bytes(raw)
    # Retained DefinitionBundle before-validators require JSON array conversion.
    # Exact canonical byte equality rejects every coercion/default insertion;
    # the materialized Python value is then revalidated strictly.
    value = semantic.SemanticContrastCatalogInput.model_validate_json(normalized, strict=False)
    if canonical_json_bytes(value) != normalized:
        raise ValueError("semantic catalog must include its exact canonical model shape")
    return semantic.SemanticContrastCatalogInput.model_validate(
        value.model_dump(mode="python", warnings="error"), strict=True
    )


def _semantic_target_labels(request: CounterfactualSolveRequest):
    # Only called after the retained compiler validates this closed M3 shape.
    before = request.semantic_problem.before_preconditions[0].formula
    after = request.semantic_problem.after_goal.formula
    before_pair = tuple(operand.payload.reference for operand in before.operands[:2])
    after_pair = tuple(operand.payload.reference for operand in after.operands[:2])
    before_relation = SemanticRelation(before.operands[2].payload.symbol.removeprefix("relation:"))
    after_relation = SemanticRelation(after.operands[2].payload.symbol.removeprefix("relation:"))
    return before_pair, after_pair, before_relation, after_relation


def _derive_semantic_catalog(
    value: semantic.SemanticContrastCatalogInput,
    runtime: RuntimeProvenance,
) -> tuple[semantic.SemanticContrastCatalog, tuple[_SemanticCompilation, ...]]:
    if value.publication_policy.eligibility_claim_definition_ref != _SEMANTIC_ELIGIBILITY_CLAUSE:
        raise ValueError("unsupported semantic publication eligibility clause")
    # Validate every profile before establishing any publication-policy outcome.
    compilations = tuple(compile_upright_se2(item.solve_request) for item in value.candidates)
    sources = tuple(semantic.FrozenSourceRecord.seal(
        source=SemanticObjectReference[Literal["SourceRecordInput"]].from_model(source),
        identity=source.identity, source_group=source.source_group,
        scene_state_sha256=source.scene_state_sha256,
        split=semantic.CatalogSplit(assign_split(source.source_group))
    ) for source in value.sources)
    source_map = {canonical_json_bytes(item.identity): item for item in sources}
    candidates = []
    for candidate in value.candidates:
        source = source_map[canonical_json_bytes(candidate.source_identity)]
        before_pair, after_pair, before, after = _semantic_target_labels(candidate.solve_request)
        evidence = semantic.PolicyEligibilityEvidence.seal(
            eligible=before_pair == after_pair and after == before.opposite,
            clause_definition_ref=_SEMANTIC_ELIGIBILITY_CLAUSE,
            solve_request_sha256=candidate.solve_request_sha256,
            source_record_input_sha256=source.source.semantic_sha256,
            publication_policy_sha256=value.publication_policy.publication_policy_sha256,
        )
        candidates.append(semantic.FrozenCandidateRecord.seal(
            candidate=SemanticObjectReference[Literal["CandidateRecordInput"]].from_model(candidate),
            candidate_id=candidate.candidate_id, source_identity=candidate.source_identity,
            source_record_input_sha256=source.source.semantic_sha256,
            solve_request_sha256=candidate.solve_request_sha256,
            backend_profile=candidate.backend_profile, source_group=source.source_group,
            split=source.split, policy_evidence=evidence,
        ))
    normalized = canonical_json_bytes(value)
    return semantic.SemanticContrastCatalog.seal(
        normalized_input_byte_length=len(normalized),
        normalized_input_byte_sha256=hashlib.sha256(normalized).hexdigest(),
        catalog_input=SemanticObjectReference[Literal["SemanticContrastCatalogInput"]].from_model(value),
        sources=sources, candidates=tuple(candidates),
        publication_policy=SemanticObjectReference[Literal["PublicationPolicy"]].from_model(value.publication_policy),
        runtime_provenance=SemanticObjectReference[Literal["RuntimeProvenance"]].from_model(runtime),
    ), compilations


@contextmanager
def _bound_semantic_catalog(
    path: Path, runtime: RuntimeProvenance
) -> Iterator[tuple[_SemanticCatalogPlan, Callable[[], None]]]:
    path = Path(path).absolute()
    with ExitStack() as stack:
        root = stack.enter_context(bound_absolute_directory(path.parent))
        directories = {(): root}
        retained_entries = {}

        def read(relative: str, maximum: int) -> bytes:
            parts = tuple(relative.split("/"))
            parent = ()
            for component in parts[:-1]:
                child = (*parent, component)
                if child not in directories:
                    directories[child] = stack.enter_context(
                        bound_child_directory(directories[parent], component)
                    )
                parent = child
            descriptor = directories[parent]
            current = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
            payload = read_regular_at(descriptor, parts[-1], maximum, expected_stat=current)
            retained_entries.setdefault(descriptor, {})[parts[-1]] = current
            return payload

        transport = read(path.name, _MAX_PLAN_BYTES)
        logging.getLogger(__name__).info(
            "Semantic catalog transport sha256=%s byte_length=%d",
            hashlib.sha256(transport).hexdigest(), len(transport),
        )
        value = _normalize_semantic_catalog(transport)
        references = {}
        for source in value.sources:
            for reference in (source.source_record_bytes, *source.source_files.files):
                references.setdefault(reference.relative_path, reference)
        payloads = {}
        for relative, reference in sorted(references.items()):
            payload = read(relative, max(1, reference.byte_length))
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != reference.byte_length or digest != reference.byte_sha256:
                raise ValueError("semantic source byte identity differs from catalog")
            payloads[digest] = payload
        def revalidate():
            for descriptor, entries in retained_entries.items():
                revalidate_entries(descriptor, entries)

        revalidate()
        catalog, compilations = _derive_semantic_catalog(value, runtime)
        revalidate()
        yield _SemanticCatalogPlan(
            catalog=catalog, catalog_input=value, normalized_input=canonical_json_bytes(value),
            source_payloads=tuple(sorted(payloads.items())), compilations=compilations,
        ), revalidate
        revalidate()


# Preserve supported public names and pickle lookup.
_SemanticCatalogPlan.__module__ = "spatialcf.generation.planning.campaign"
_semantic_json_object.__module__ = "spatialcf.generation.planning.campaign"
_normalize_semantic_catalog.__module__ = "spatialcf.generation.planning.campaign"
_semantic_target_labels.__module__ = "spatialcf.generation.planning.campaign"
_derive_semantic_catalog.__module__ = "spatialcf.generation.planning.campaign"
_bound_semantic_catalog.__module__ = "spatialcf.generation.planning.campaign"
