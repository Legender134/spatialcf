"""Stable dataset workflow imports with explicit implementation owners."""

from __future__ import annotations

import hashlib

import importlib.metadata

import os

import re

import stat

import sys

from collections import Counter

from collections.abc import Callable, Iterator, Mapping

from contextlib import ExitStack, contextmanager

from dataclasses import dataclass

from pathlib import Path, PurePosixPath

from typing import Literal

from spatialcf.adapters.base import EnvironmentAdapter as AI2ThorAdapter

from spatialcf.adapters.base import SettledReadback

from spatialcf.core.outcome_assembler import (
    AssembledCounterfactualOutcome,
    assemble_counterfactual_outcome,
    assemble_no_selection_unknown,
)

from spatialcf.core.upright_se2_backend import (
    UprightSE2CardinalBackend,
    UprightSE2ContinuousBackend,
)

from spatialcf.domain import contrast as semantic

from spatialcf.domain import upright_se2 as upright

from spatialcf.domain.base import Sha256Digest

from spatialcf.domain.definitions import HashBoundCanonicalModel

from spatialcf.domain.lineage import (
    DependencyInventoryEntry,
    InterpreterIdentity,
    LockMetadata,
    ModuleFileIdentity,
    NativeNotRequested,
    RuntimeProvenance,
    SemanticObjectReference,
    SourceLineageIdentity,
)

from spatialcf.domain.outcomes import (
    BackendCompleteUnsatEvidence,
    BackendProposalSubmission,
    BackendSubmission,
    BackendUnknownEvidence,
    CertifiedSolutionCertificate,
    CertifiedSolutionResult,
    NoncertifiedWitnessResult,
    ProvenUnsatCertificate,
    ProvenUnsatResult,
    UnknownResult,
)

from spatialcf.domain.request import InterventionSpec

from spatialcf.domain.serialization import (
    canonical_json_bytes,
    canonical_sha256,
)

import spatialcf.generation.capture as capture
import spatialcf.generation.execution as execution
import spatialcf.generation.planning as planning
import spatialcf.generation.publication as publication

from spatialcf.generation.config import GenerationConfig, load_generation_config

from spatialcf.generation.dataset_models import (
    DatasetManifest,
    DatasetRecord,
    GenerationReport,
    _canonical_model_bytes,
    _safe_relative_path,
)

from spatialcf.generation.execution.audit import (
    EndpointAuditRejected,
    _execute_audit_with_transitions,
)

from spatialcf.generation.planning.campaign import (
    _SemanticCatalogPlan,
    _bound_semantic_catalog,
    _derive_semantic_catalog,
    _semantic_target_labels,
)

from spatialcf.generation.workflows.capture import (
    _capture_and_publish_dataset_with_transitions,
)

from spatialcf.generation.workflows.contracts import (
    CounterfactualRequest,
    ExecutedEdit,
    ExecutionRejection,
    PlanningRejection,
    VerificationRejection,
    VerifiedExample,
    _FreshTransitions,
)

from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    RenameLocation,
    bound_absolute_directory,
    bound_child_directory,
    directory_identity_fd,
    open_directory,
    open_native_output_parent,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
    scan_directory,
    snapshot_exact_directory,
    sync_directory_fd,
    write_regular_sync_at,
)

from spatialcf.verification.contrast import (
    SemanticContrastBundle,
    read_semantic_contrast_bundle,
)

from spatialcf.generation._internal.artifact_runtime import (
    _SEMANTIC_RUNTIME_MODULE_CLOSURE,
    _semantic_retained_bytes,
    _semantic_runtime_provenance,
    _semantic_publication_error,
)

from spatialcf.generation.workflows._dataset.orchestration import (
    _load_or_build_roster,
    _load_or_build_source_plan,
    _plan_source_campaign_with_transitions,
    _run_batches,
    generate_dataset,
)

from spatialcf.generation.workflows._dataset.artifacts import (
    _parse_attempts,
    _verified_bundle_at_path,
    _verified_bundle_fd,
    _path_attempts,
    _descriptor_attempts,
    _parse_records,
    _parse_checksum_ledger,
    _verify_public_index_fd,
    _verify_source_campaign_fd,
    _revalidate_source_campaign_fd,
    _verify_dataset_fd,
)

from spatialcf.generation.workflows._dataset.contracts import (
    _CONFIG_HASH_DOMAIN,
    _DATASET_TREE_HASH_DOMAIN,
    _PUBLIC_FILES,
    _PUBLIC_DIRECTORIES,
    _STATE_DIRECTORIES,
    _BUNDLE_FILES,
    _MAX_METADATA_BYTES,
    _MAX_ASSET_BYTES,
    SourcePlanIncompleteError,
    _VerifiedAttempt,
    _RetainedBatchState,
    _RetainedCampaignState,
)

from spatialcf.generation.workflows._dataset.derivation import (
    _terminal_key,
    _dataset_tree_sha256,
    _derive_dataset,
)

from spatialcf.generation.workflows._dataset.publication import (
    _checksum_payload,
    _stage_public_index,
    _read_or_write_exact,
    _ensure_exact_directory,
    _copy_staged_public_index,
    _bound_rollback_parent,
    _rollback_created_fd,
    _publication_root_binding,
    _raise_publication_failure,
    _publish_dataset_index,
)

from spatialcf.generation.workflows._dataset.state import (
    _require_complete_source_plan,
    _config_sha256,
    _capture_plan,
    _checked_config,
    _absolute_output,
    _initialize_dataset_root,
    _existing_names,
    _validate_generation_root,
    _ensure_batches_root,
    _config_from_capture_plan,
)

from spatialcf.generation.workflows._dataset.semantic import (
    _semantic_ref,
    _semantic_sha,
    _DerivedSemanticBundle,
    _semantic_backend,
    _add_semantic_object,
    _semantic_terminal,
    _semantic_inventory_entry,
    _derive_semantic_bundle,
    _generation_submission,
    _retained_submission_provider,
    _fresh_replay_semantic_bundle,
    _write_semantic_bundle,
    generate_semantic_contrast_dataset,
    verify_semantic_contrast_dataset,
)

__all__ = (
    "generate_dataset",
    "generate_semantic_contrast_dataset",
    "verify_semantic_contrast_dataset",
)
