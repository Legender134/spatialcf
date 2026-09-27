"""Stable source campaign API backed by explicit contracts, jobs and storage."""

from __future__ import annotations

import hashlib

import json

import logging

import os

import warnings

from collections import Counter

from collections.abc import Callable, Iterator

from contextlib import ExitStack, contextmanager

from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import dataclass, field

from multiprocessing import get_context

from multiprocessing.connection import Connection

from pathlib import Path

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from spatialcf.core.solver import (
    solve_minimum_cost,
)

from spatialcf.core.upright_se2_compiler import compile_upright_se2

from spatialcf.domain import contrast as semantic

from spatialcf.domain.constraints import Relation as SemanticRelation

from spatialcf.domain.counterfactual import CounterfactualSolveRequest

from spatialcf.domain.lineage import RuntimeProvenance, SemanticObjectReference

from spatialcf.domain.outcomes import TypedCompilationOutcome

from spatialcf.domain.upright_se2 import UprightSE2Compilation, UprightSE2ContinuousCompilation

from spatialcf.domain.base import CanonicalId, CanonicalModel, Sha256Digest

from spatialcf.domain.request import InterventionSpec, Relation

from spatialcf.domain.scene import Scene

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

from spatialcf.domain.solver import (
    ContinuousYawCertifiedSuccessResultV2_9,
    ContinuousYawSolverConfigV2_9,
)

from spatialcf.generation.capture.compiler import (
    compile_roster,
    verify_competition_native_camera_evidence_capture_v2_9_3,
)

from spatialcf.generation.capture.models import (
    CameraPolicy,
    CompetitionNativeCandidateRosterManifestV2_9,
    CompetitionNativePlacementAvailabilityV2_9,
    CompetitionNativeSelectedRequestV2_9,
    CompetitionNativeSourceCaptureV2_9,
    CompetitionNativeSubjectPlacementFactV2_9,
    CompetitionNativeSupportKindV2_9,
    RosterCompilation,
    SourceCameraEvidence,
    SourceSurfaceEvidence,
    SubjectSurfaceEvidence,
    verify_source_surface_evidence,
)

from spatialcf.generation.capture.reachability import (
    CandidateTargetReachability,
    TargetReachabilityStatus,
)

from spatialcf.generation.errors import require_wire_version

from spatialcf.generation.planning.endpoint import (
    EndpointPlanRejected,
    endpoint_workspace_within_position_region,
    plan_endpoint,
)

from spatialcf.generation.planning.models import (
    CollisionDelegation,
    EndpointPlan,
    EndpointWorkspace,
    ProxyBundle,
    SourceViewGuard,
    SubjectPlacementFact,
)

from spatialcf.generation.planning.problem import (
    build_proxy_bundle,
    default_planning_workspace,
    default_solver_config,
)

from spatialcf.generation.planning.view_guard import evaluate_source_view_guard

from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    DirectoryIdentity,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    snapshot_exact_directory,
)

from spatialcf.verification.integrity import competition_legacy_sha256

from spatialcf.verification.split import assign_split

from spatialcf.generation.planning.campaign_contracts import (
    _POLICY_DOMAIN,
    _PLAN_DOMAIN,
    _TARGET_LEDGER_DOMAIN,
    _RUNTIME_DOMAIN,
    _ACCEPTED_ROSTER_DOMAIN,
    _RUNTIME_POSE_POLICY_DOMAIN,
    _SOURCE_POLICY_VERSION,
    _SOURCE_PLAN_VERSION,
    _FILES,
    _MAX_PLAN_BYTES,
    _MAX_POLICY_BYTES,
    _MAX_CHECKSUM_BYTES,
    _MAX_REQUESTS,
    _MAX_SOURCES,
    _MAX_TARGET_ROWS,
    _MAX_REASON_CHARS,
    _SCREENING_DOMAIN_OPERATIONS,
    _RELATIONS,
    _manifest_file_sha256,
    _runtime_identity_sha256,
    _legacy_sha256,
    _accepted_source_capture_roster_sha256,
    _target_reachability_ledger_sha256,
    RuntimePosePolicy,
    SourcePolicy,
    SourceRequestOutcome,
    SourceSlotEntry,
    SourceSlotOutcome,
    BatchRequest,
    BatchManifest,
    _case_id,
    _workspace_is_subset,
    _fresh_solve_result,
    SourcePlan,
    _SOURCE_PLAN_CAPABILITY,
    _SOURCE_PLAN_VERIFICATION_CAPABILITY,
    RetainedSourcePlan,
    RetainedSourcePlanVerification,
)

from spatialcf.generation.planning.campaign_jobs import (
    build_default_source_policy,
    _endpoint_rejection_reason,
    _endpoint_rejection_reasons,
    _rejected_outcome,
    _endpoint_policy_controls,
    _preflight_request,
    _PlanningJob,
    _prepare_job,
    _endpoint_worker,
    _bounded_default_endpoint_plan,
    _run_job,
    _plan_parallel_outcomes,
    plan_source_campaign,
)

from spatialcf.generation.planning.campaign_storage import (
    _parse_source_plan,
    _load_source_policy_fd,
    load_source_policy,
    _load_source_plan_fd,
    prepare_source_plan_verification,
    revalidate_source_plan_verification,
    load_source_plan,
    _retain_checked_source_plan,
    load_source_plan_retained,
    _rollback_transaction,
    _raise_transaction_exit_error,
    publish_source_plan,
)

from spatialcf.generation.planning.semantic import (
    _SemanticCompilation,
    _SEMANTIC_ELIGIBILITY_CLAUSE,
    _SemanticCatalogPlan,
    _semantic_json_object,
    _normalize_semantic_catalog,
    _semantic_target_labels,
    _derive_semantic_catalog,
    _bound_semantic_catalog,
)

__all__ = (
    "BatchManifest",
    "BatchRequest",
    "RetainedSourcePlan",
    "RetainedSourcePlanVerification",
    "RuntimePosePolicy",
    "SourcePlan",
    "SourcePolicy",
    "SourceRequestOutcome",
    "SourceSlotEntry",
    "SourceSlotOutcome",
    "build_default_source_policy",
    "load_source_plan",
    "load_source_plan_retained",
    "load_source_policy",
    "plan_source_campaign",
    "prepare_source_plan_verification",
    "publish_source_plan",
    "revalidate_source_plan_verification",
)
