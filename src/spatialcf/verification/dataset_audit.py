"""Quality gates and matched-generator comparison over authenticated datasets."""

from __future__ import annotations

import math
import os
import stat
from pathlib import Path
from statistics import mean
from typing import Any

from spatialcf.domain.request import QualityTier, Relation, SolverStatus
from spatialcf.verification.dataset_reader import (
    _read_dataset,
)
from spatialcf.verification.profile import profile_from_manifest


def _validate_integer(name: str, value: int, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite value between 0 and 1")
    return float(value)


def audit_dataset(
    root: Path,
    minimum_pairs: int,
    minimum_pure_test: int = 0,
    verify_scenes: bool = False,
    *,
    maximum_mean_leakage: float = 1.0,
    maximum_mean_edit_distance: float = 1.0,
    minimum_pure_rate: float = 0.0,
    minimum_solvable_coverage: float = 0.0,
    expected_source_digest: str | None = None,
    root_descriptor: int | None = None,
    _descriptor_capability: bool = False,
) -> dict[str, Any]:
    """Validate one immutable dataset and enforce configured quality gates."""
    if root_descriptor is not None:
        if type(root_descriptor) is not int or root_descriptor < 0:
            raise ValueError("root_descriptor must be an open directory fd")
        try:
            duplicate = os.dup(root_descriptor)
        except OSError as error:
            raise ValueError("cannot retain dataset root descriptor") from error
        try:
            descriptor_result = os.fstat(duplicate)
        except OSError as error:
            os.close(duplicate)
            raise ValueError("cannot inspect dataset root descriptor") from error
        if not stat.S_ISDIR(descriptor_result.st_mode):
            os.close(duplicate)
            raise ValueError("dataset root descriptor is not a directory")
        descriptor_path = Path("/proc/self/fd") / str(duplicate)
        if not descriptor_path.is_dir():
            os.close(duplicate)
            raise ValueError("descriptor-rooted audit requires Linux procfs")
        try:
            return audit_dataset(
                descriptor_path,
                minimum_pairs,
                minimum_pure_test,
                verify_scenes,
                maximum_mean_leakage=maximum_mean_leakage,
                maximum_mean_edit_distance=maximum_mean_edit_distance,
                minimum_pure_rate=minimum_pure_rate,
                minimum_solvable_coverage=minimum_solvable_coverage,
                expected_source_digest=expected_source_digest,
                _descriptor_capability=True,
            )
        finally:
            os.close(duplicate)
    if _descriptor_capability and not str(root).startswith("/proc/self/fd/"):
        raise ValueError("invalid descriptor-rooted audit capability")
    minimum_pairs = _validate_integer("minimum_pairs", minimum_pairs, minimum=0)
    minimum_pure_test = _validate_integer(
        "minimum_pure_test",
        minimum_pure_test,
        minimum=0,
    )
    if type(verify_scenes) is not bool:
        raise ValueError("verify_scenes must be a boolean")
    maximum_mean_leakage = _validate_rate(
        "maximum_mean_leakage",
        maximum_mean_leakage,
    )
    maximum_mean_edit_distance = _validate_rate(
        "maximum_mean_edit_distance",
        maximum_mean_edit_distance,
    )
    minimum_pure_rate = _validate_rate("minimum_pure_rate", minimum_pure_rate)
    minimum_solvable_coverage = _validate_rate(
        "minimum_solvable_coverage",
        minimum_solvable_coverage,
    )

    if expected_source_digest is not None and (
        type(expected_source_digest) is not str
        or len(expected_source_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_source_digest
        )
    ):
        raise ValueError("expected_source_digest must be lowercase SHA-256")
    dataset = _read_dataset(
        root,
        expected_source_digest=expected_source_digest,
        descriptor_capability=_descriptor_capability,
    )
    records = dataset.pairs
    failures = dataset.failures
    empty_smoke_baseline = (
        not records
        and minimum_pairs == 0
        and dataset.manifest.get("run_profile") == "smoke"
        and dataset.manifest.get("evidence_eligible") is False
        and dataset.provenance is not None
        and dataset.provenance.generator in {"random", "target-only"}
        and dataset.provenance.attempt_limit is not None
        and dataset.provenance.attempt_limit > 0
        and len(dataset.attempts) == dataset.provenance.attempt_limit
    )
    if not records and not empty_smoke_baseline:
        raise ValueError("accepted_pairs=0; empty datasets cannot pass audit")
    if len(records) < minimum_pairs:
        raise ValueError(f"accepted_pairs={len(records)} below minimum={minimum_pairs}")

    pure_test = tuple(
        record
        for record in records
        if record.split == "test" and record.quality is QualityTier.PURE
    )
    if len(pure_test) < minimum_pure_test:
        raise ValueError(
            f"pure_test_pairs={len(pure_test)} below minimum={minimum_pure_test}"
        )
    if minimum_pure_test:
        for tag in ("unseen_scene", "unseen_category", "unseen_combination"):
            if not any(tag in record.holdout_tags for record in pure_test):
                raise ValueError(f"PURE test set has no {tag} pairs")
        pure_relations = {record.relation_after for record in pure_test}
        missing = sorted(relation.value for relation in set(Relation) - pure_relations)
        if missing:
            raise ValueError(f"PURE test set missing target relations: {missing}")
    if minimum_pairs >= 100:
        observed = {record.relation_after for record in records}
        missing = sorted(relation.value for relation in set(Relation) - observed)
        if missing:
            raise ValueError(f"dataset missing target relations: {missing}")

    if dataset.provenance is None and minimum_solvable_coverage > 0.0:
        raise ValueError(
            "solvable coverage gate requires an authenticated attempt ledger"
        )
    total_requests = (
        len(dataset.attempts)
        if dataset.provenance is not None
        else len(records) + len(failures)
    )
    pure_rate = (
        mean(record.quality is QualityTier.PURE for record in records)
        if records
        else 0.0
    )
    mean_leakage = mean(record.leakage_score for record in records) if records else 0.0
    mean_edit_distance = (
        mean(record.normalized_edit_distance for record in records) if records else 0.0
    )
    mean_evaluated = (
        mean(record.evaluated_candidates for record in records) if records else 0.0
    )
    solvable_coverage = len(records) / total_requests if total_requests else 0.0
    gates = (
        (
            "mean_leakage",
            mean_leakage,
            maximum_mean_leakage,
            "above maximum",
            lambda actual, threshold: actual > threshold,
        ),
        (
            "mean_edit_distance",
            mean_edit_distance,
            maximum_mean_edit_distance,
            "above maximum",
            lambda actual, threshold: actual > threshold,
        ),
        (
            "pure_rate",
            pure_rate,
            minimum_pure_rate,
            "below minimum",
            lambda actual, threshold: actual < threshold,
        ),
        (
            "solvable_coverage",
            solvable_coverage,
            minimum_solvable_coverage,
            "below minimum",
            lambda actual, threshold: actual < threshold,
        ),
    )
    for name, actual, threshold, detail, failed in gates:
        if failed(actual, threshold):
            raise ValueError(f"{name}={actual:.6f} {detail}={threshold:.6f}")

    report: dict[str, Any] = {
        "accepted_pairs": len(records),
        "attempts_authenticated": dataset.attempts_authenticated,
        "attempt_limit": (
            dataset.provenance.attempt_limit if dataset.provenance is not None else None
        ),
        "attempted_requests": total_requests,
        "requested_pairs": (
            dataset.provenance.requested_pairs
            if dataset.provenance is not None
            else None
        ),
        "source_corpus_sha256": (
            dataset.provenance.source_corpus_sha256
            if dataset.provenance is not None
            else None
        ),
        "failed_requests": len(failures),
        "solvable_coverage": solvable_coverage,
        "pure_test_pairs": len(pure_test),
        "pure_rate": pure_rate,
        "mean_leakage": mean_leakage,
        "mean_edit_distance": mean_edit_distance,
        "mean_evaluated_candidates": mean_evaluated,
        "failure_statuses": {
            status.value: sum(failure.status is status for failure in failures)
            for status in SolverStatus
            if status is not SolverStatus.SUCCESS
        },
        "by_generator": {},
    }
    for generator in sorted({record.generator for record in records}):
        subset = tuple(record for record in records if record.generator == generator)
        report["by_generator"][generator] = {
            "count": len(subset),
            "mean_leakage": mean(record.leakage_score for record in subset),
            "pure_rate": mean(record.quality is QualityTier.PURE for record in subset),
        }
    return report


def compare_generators(
    spatial_root: Path,
    random_root: Path,
    minimum_matched: int,
    minimum_reduction: float,
    *,
    expected_source_digest: str,
) -> dict[str, float | int | str]:
    """Enforce the matched-request Gate B leakage comparison."""
    minimum_matched = _validate_integer(
        "minimum_matched",
        minimum_matched,
        minimum=1,
    )
    minimum_reduction = _validate_rate(
        "minimum_reduction",
        minimum_reduction,
    )
    if (
        type(expected_source_digest) is not str
        or len(expected_source_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_source_digest
        )
    ):
        raise ValueError("expected_source_digest must be lowercase SHA-256")
    spatial_dataset = _read_dataset(
        spatial_root,
        expected_source_digest=expected_source_digest,
    )
    random_dataset = _read_dataset(
        random_root,
        expected_source_digest=expected_source_digest,
    )
    spatial_profile = profile_from_manifest(spatial_dataset.manifest)
    random_profile = profile_from_manifest(random_dataset.manifest)
    if (
        spatial_profile is None
        or random_profile is None
        or spatial_profile.evidence_eligible is not True
        or random_profile.evidence_eligible is not True
    ):
        raise ValueError("Gate B inputs are not evidence eligible")
    if spatial_dataset.provenance is None or random_dataset.provenance is None:
        raise ValueError(
            "Gate B rejects legacy datasets without official replay attestation"
        )
    spatial_provenance = spatial_dataset.provenance
    random_provenance = random_dataset.provenance
    if (
        spatial_provenance.attempt_limit is None
        or random_provenance.attempt_limit is None
    ):
        raise ValueError(
            "Gate B requires each dataset to use a fixed authenticated attempt prefix"
        )
    if spatial_provenance.attempt_limit != random_provenance.attempt_limit:
        raise ValueError(
            "Gate B datasets must use the same authenticated attempt_limit"
        )
    attempt_limit = spatial_provenance.attempt_limit
    if spatial_provenance.adapter_backend != random_provenance.adapter_backend:
        raise ValueError(
            "Gate B datasets must use the same authenticated adapter backend"
        )
    if (
        spatial_provenance.adapter_implementation
        != random_provenance.adapter_implementation
        or spatial_provenance.adapter_config != random_provenance.adapter_config
    ):
        raise ValueError("Gate B datasets must use identical canonical adapter config")
    spatial_sources = [
        (source.scene_id, source.sha256) for source in spatial_provenance.source_scenes
    ]
    random_sources = [
        (source.scene_id, source.sha256) for source in random_provenance.source_scenes
    ]
    if spatial_sources != random_sources:
        raise ValueError("Gate B datasets use different authenticated inputs")
    spatial_prefix = tuple(attempt.request_id for attempt in spatial_dataset.attempts)
    random_prefix = tuple(attempt.request_id for attempt in random_dataset.attempts)
    if spatial_prefix != random_prefix:
        raise ValueError(
            "Gate B datasets must use an identical authenticated request-ID prefix"
        )
    for label, dataset, expected in (
        ("spatial", spatial_dataset, "spatialcf"),
        ("random", random_dataset, "random"),
    ):
        if dataset.provenance is None or dataset.provenance.generator != expected:
            actual = (
                dataset.provenance.generator if dataset.provenance is not None else None
            )
            raise ValueError(
                f"{label} dataset expected generator={expected!r}; "
                f"found provenance={actual!r}"
            )
        unexpected = sorted(
            {
                record.generator
                for record in dataset.pairs
                if record.generator != expected
            }
        )
        if unexpected:
            raise ValueError(
                f"{label} dataset expected generator={expected!r}; found={unexpected}"
            )

    spatial = {record.request_id: record for record in spatial_dataset.pairs}
    random = {record.request_id: record for record in random_dataset.pairs}
    request_ids = [
        request_id
        for request_id in spatial_prefix
        if request_id in spatial and request_id in random
    ]
    spatial_only = [
        request_id
        for request_id in spatial_prefix
        if request_id in spatial and request_id not in random
    ]
    random_only = [
        request_id
        for request_id in spatial_prefix
        if request_id in random and request_id not in spatial
    ]
    both_failed = [
        request_id
        for request_id in spatial_prefix
        if request_id not in spatial and request_id not in random
    ]
    if len(request_ids) < minimum_matched:
        raise ValueError(
            f"matched_requests={len(request_ids)} below minimum={minimum_matched}"
        )

    match_fields = (
        "request_id",
        "scene_id",
        "split",
        "holdout_tags",
        "source",
        "seed",
        "subject_id",
        "subject_category",
        "reference_id",
        "reference_category",
        "camera_id",
        "relation_before",
        "relation_after",
        "question",
        "answer_before",
        "answer_after",
    )
    for request_id in request_ids:
        mismatches = [
            field
            for field in match_fields
            if getattr(spatial[request_id], field) != getattr(random[request_id], field)
        ]
        spatial_before = (
            spatial_dataset.root / spatial[request_id].scene_before_path
        ).read_bytes()
        random_before = (
            random_dataset.root / random[request_id].scene_before_path
        ).read_bytes()
        if spatial_before != random_before:
            mismatches.append("scene_before_evidence")
        if mismatches:
            raise ValueError(f"request payload mismatch for {request_id}: {mismatches}")

    spatial_mean = mean(spatial[request_id].leakage_score for request_id in request_ids)
    random_mean = mean(random[request_id].leakage_score for request_id in request_ids)
    if random_mean <= 0.0:
        raise ValueError(
            "random matched mean leakage must be a positive reduction denominator"
        )
    reduction = (random_mean - spatial_mean) / random_mean
    if reduction < minimum_reduction:
        raise ValueError(
            f"leakage_reduction={reduction:.6f} below minimum={minimum_reduction:.6f}"
        )
    return {
        "attempt_limit": attempt_limit,
        "both_failed_requests": len(both_failed),
        "matched_coverage": len(request_ids) / attempt_limit,
        "matched_requests": len(request_ids),
        "random_only_requests": len(random_only),
        "random_success_rate": len(random) / attempt_limit,
        "spatial_only_requests": len(spatial_only),
        "spatialcf_success_rate": len(spatial) / attempt_limit,
        "trusted_source_digest": expected_source_digest,
        "spatialcf_mean_leakage": spatial_mean,
        "random_mean_leakage": random_mean,
        "relative_reduction": reduction,
    }


_validate_integer.__module__ = "spatialcf.verification.dataset"
_validate_rate.__module__ = "spatialcf.verification.dataset"
audit_dataset.__module__ = "spatialcf.verification.dataset"
compare_generators.__module__ = "spatialcf.verification.dataset"
