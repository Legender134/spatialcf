"""Dataset artifact runtime; explicit implementation responsibility."""

from __future__ import annotations

import hashlib

import importlib.metadata

import os

import re

import sys

from pathlib import (
    Path,
)

from spatialcf.domain.lineage import (
    DependencyInventoryEntry,
    InterpreterIdentity,
    LockMetadata,
    ModuleFileIdentity,
    RuntimeProvenance,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.verification.filesystem import (
    CompetitionNativePublicationError,
    RenameLocation,
    bound_absolute_directory,
    read_regular_at,
    reconcile_owned_rename_at,
    revalidate_entries,
)


_SEMANTIC_RUNTIME_MODULE_CLOSURE = (
    "spatialcf/__init__.py",
    "spatialcf/adapters/__init__.py",
    "spatialcf/adapters/ai2thor/__init__.py",
    "spatialcf/adapters/ai2thor/adapter.py",
    "spatialcf/adapters/ai2thor/camera.py",
    "spatialcf/adapters/ai2thor/capture.py",
    "spatialcf/adapters/ai2thor/conversion.py",
    "spatialcf/adapters/ai2thor/execution.py",
    "spatialcf/adapters/ai2thor/geometry_policy.py",
    "spatialcf/adapters/ai2thor/models.py",
    "spatialcf/adapters/ai2thor/support.py",
    "spatialcf/adapters/ai2thor/validation.py",
    "spatialcf/adapters/base.py",
    "spatialcf/composition.py",
    "spatialcf/core/__init__.py",
    "spatialcf/core/_internal/compilation/__init__.py",
    "spatialcf/core/_internal/compilation/collision.py",
    "spatialcf/core/_internal/compilation/support.py",
    "spatialcf/core/_internal/compilation/target.py",
    "spatialcf/core/_internal/compilation/universe.py",
    "spatialcf/core/_internal/compilation/visibility.py",
    "spatialcf/core/_internal/kernels/__init__.py",
    "spatialcf/core/_internal/kernels/camera.py",
    "spatialcf/core/_internal/kernels/convex_partition.py",
    "spatialcf/core/_internal/kernels/convex_translation.py",
    "spatialcf/core/_internal/kernels/projected_visibility.py",
    "spatialcf/core/_internal/kernels/rect.py",
    "spatialcf/core/_internal/kernels/rectilinear.py",
    "spatialcf/core/_internal/kernels/so2.py",
    "spatialcf/core/_internal/kernels/strict_convex.py",
    "spatialcf/core/_internal/kernels/upright_box.py",
    "spatialcf/core/_internal/kernels/visibility_contracts.py",
    "spatialcf/core/_internal/objective/__init__.py",
    "spatialcf/core/_internal/objective/base.py",
    "spatialcf/core/_internal/objective/numeric.py",
    "spatialcf/core/_internal/objective/relation.py",
    "spatialcf/core/_internal/objective/relation_damage.py",
    "spatialcf/core/_internal/objective/relation_partition.py",
    "spatialcf/core/_internal/objective/safety.py",
    "spatialcf/core/_internal/objective/safety_bounds.py",
    "spatialcf/core/_internal/objective/visibility.py",
    "spatialcf/core/_internal/objective/visibility_objective.py",
    "spatialcf/core/_internal/registry/__init__.py",
    "spatialcf/core/_internal/registry/contracts.py",
    "spatialcf/core/_internal/registry/definitions.py",
    "spatialcf/core/_internal/registry/outcome.py",
    "spatialcf/core/_internal/registry/problem.py",
    "spatialcf/core/_internal/registry/program.py",
    "spatialcf/core/_internal/registry/request.py",
    "spatialcf/core/_internal/registry/routing.py",
    "spatialcf/core/_internal/registry/state.py",
    "spatialcf/core/_internal/resources.py",
    "spatialcf/core/_internal/upright_se2/__init__.py",
    "spatialcf/core/_internal/upright_se2/arithmetic.py",
    "spatialcf/core/_internal/upright_se2/bindings.py",
    "spatialcf/core/_internal/upright_se2/compilation.py",
    "spatialcf/core/_internal/upright_se2/constants.py",
    "spatialcf/core/_internal/upright_se2/evaluation.py",
    "spatialcf/core/_internal/upright_se2/evaluation_data.py",
    "spatialcf/core/_internal/upright_se2/evaluation_geometry.py",
    "spatialcf/core/_internal/upright_se2/m2.py",
    "spatialcf/core/_internal/upright_se2/materialization.py",
    "spatialcf/core/_internal/upright_se2/proof_values.py",
    "spatialcf/core/_internal/upright_se2/solve_context.py",
    "spatialcf/core/_internal/upright_se2/visibility.py",
    "spatialcf/core/certificate.py",
    "spatialcf/core/feasibility.py",
    "spatialcf/core/objective.py",
    "spatialcf/core/outcome_assembler.py",
    "spatialcf/core/problem.py",
    "spatialcf/core/registry.py",
    "spatialcf/core/solver.py",
    "spatialcf/core/upright_se2_backend.py",
    "spatialcf/core/upright_se2_compiler.py",
    "spatialcf/core/upright_se2_verification.py",
    "spatialcf/core/verification.py",
    "spatialcf/domain/__init__.py",
    "spatialcf/domain/_upright_se2/__init__.py",
    "spatialcf/domain/_upright_se2/cells.py",
    "spatialcf/domain/_upright_se2/compilation.py",
    "spatialcf/domain/_upright_se2/constants.py",
    "spatialcf/domain/_upright_se2/endpoints.py",
    "spatialcf/domain/_upright_se2/m2.py",
    "spatialcf/domain/_upright_se2/policies.py",
    "spatialcf/domain/_upright_se2/proof_evaluation.py",
    "spatialcf/domain/_upright_se2/proof_material.py",
    "spatialcf/domain/_upright_se2/proof_wire.py",
    "spatialcf/domain/_upright_se2/registration.py",
    "spatialcf/domain/_upright_se2/semantics.py",
    "spatialcf/domain/_upright_se2/solve_policy.py",
    "spatialcf/domain/_upright_se2/source_binding.py",
    "spatialcf/domain/_upright_se2/state.py",
    "spatialcf/domain/_upright_se2/values.py",
    "spatialcf/domain/_upright_se2/yaw.py",
    "spatialcf/domain/artifacts.py",
    "spatialcf/domain/base.py",
    "spatialcf/domain/certificate.py",
    "spatialcf/domain/compatibility.py",
    "spatialcf/domain/constraints.py",
    "spatialcf/domain/contrast.py",
    "spatialcf/domain/counterfactual.py",
    "spatialcf/domain/definitions.py",
    "spatialcf/domain/edit.py",
    "spatialcf/domain/geometry.py",
    "spatialcf/domain/lineage.py",
    "spatialcf/domain/objective.py",
    "spatialcf/domain/operators.py",
    "spatialcf/domain/outcomes.py",
    "spatialcf/domain/predicates.py",
    "spatialcf/domain/problem.py",
    "spatialcf/domain/profiles.py",
    "spatialcf/domain/request.py",
    "spatialcf/domain/result.py",
    "spatialcf/domain/scene.py",
    "spatialcf/domain/serialization.py",
    "spatialcf/domain/solver.py",
    "spatialcf/domain/source.py",
    "spatialcf/domain/upright_se2.py",
    "spatialcf/generation/__init__.py",
    "spatialcf/generation/_internal/__init__.py",
    "spatialcf/generation/_internal/artifact_runtime.py",
    "spatialcf/generation/_internal/audit_geometry.py",
    "spatialcf/generation/_internal/canonical_json.py",
    "spatialcf/generation/_internal/source_manifest.py",
    "spatialcf/generation/capture/__init__.py",
    "spatialcf/generation/capture/_models/__init__.py",
    "spatialcf/generation/capture/_models/camera_contracts.py",
    "spatialcf/generation/capture/_models/camera_evidence.py",
    "spatialcf/generation/capture/_models/camera_geometry.py",
    "spatialcf/generation/capture/_models/camera_scoring.py",
    "spatialcf/generation/capture/_models/constants.py",
    "spatialcf/generation/capture/_models/contracts.py",
    "spatialcf/generation/capture/_models/roster.py",
    "spatialcf/generation/capture/_models/source.py",
    "spatialcf/generation/capture/_models/surfaces.py",
    "spatialcf/generation/capture/compiler.py",
    "spatialcf/generation/capture/models.py",
    "spatialcf/generation/capture/plan.py",
    "spatialcf/generation/capture/reachability.py",
    "spatialcf/generation/capture/reachability_contracts.py",
    "spatialcf/generation/capture/source.py",
    "spatialcf/generation/capture/storage.py",
    "spatialcf/generation/capture/visual_evidence.py",
    "spatialcf/generation/config.py",
    "spatialcf/generation/contrast.py",
    "spatialcf/generation/dataset.py",
    "spatialcf/generation/dataset_models.py",
    "spatialcf/generation/errors.py",
    "spatialcf/generation/execution/__init__.py",
    "spatialcf/generation/execution/audit.py",
    "spatialcf/generation/execution/batch.py",
    "spatialcf/generation/execution/campaign.py",
    "spatialcf/generation/execution/correspondence.py",
    "spatialcf/generation/planning/__init__.py",
    "spatialcf/generation/planning/campaign.py",
    "spatialcf/generation/planning/campaign_contracts.py",
    "spatialcf/generation/planning/campaign_jobs.py",
    "spatialcf/generation/planning/campaign_storage.py",
    "spatialcf/generation/planning/endpoint.py",
    "spatialcf/generation/planning/models.py",
    "spatialcf/generation/planning/native_versions.py",
    "spatialcf/generation/planning/problem.py",
    "spatialcf/generation/planning/proxy_conversion.py",
    "spatialcf/generation/planning/proxy_geometry.py",
    "spatialcf/generation/planning/proxy_preparation.py",
    "spatialcf/generation/planning/proxy_projection.py",
    "spatialcf/generation/planning/reference_persistence.py",
    "spatialcf/generation/planning/semantic.py",
    "spatialcf/generation/planning/view_guard.py",
    "spatialcf/generation/publication/__init__.py",
    "spatialcf/generation/publication/assets.py",
    "spatialcf/generation/request_selection.py",
    "spatialcf/generation/workflows/__init__.py",
    "spatialcf/generation/workflows/_dataset/__init__.py",
    "spatialcf/generation/workflows/_dataset/artifacts.py",
    "spatialcf/generation/workflows/_dataset/contracts.py",
    "spatialcf/generation/workflows/_dataset/derivation.py",
    "spatialcf/generation/workflows/_dataset/orchestration.py",
    "spatialcf/generation/workflows/_dataset/publication.py",
    "spatialcf/generation/workflows/_dataset/semantic.py",
    "spatialcf/generation/workflows/_dataset/state.py",
    "spatialcf/generation/workflows/capture.py",
    "spatialcf/generation/workflows/contracts.py",
    "spatialcf/generation/workflows/dataset.py",
    "spatialcf/generation/workflows/execution.py",
    "spatialcf/geometry/__init__.py",
    "spatialcf/geometry/obb.py",
    "spatialcf/geometry/regions.py",
    "spatialcf/geometry/transforms.py",
    "spatialcf/relations/__init__.py",
    "spatialcf/relations/engine.py",
    "spatialcf/verification/__init__.py",
    "spatialcf/verification/artifact_models.py",
    "spatialcf/verification/artifacts.py",
    "spatialcf/verification/contrast.py",
    "spatialcf/verification/dataset.py",
    "spatialcf/verification/dataset_audit.py",
    "spatialcf/verification/dataset_models.py",
    "spatialcf/verification/dataset_reader.py",
    "spatialcf/verification/filesystem.py",
    "spatialcf/verification/integrity.py",
    "spatialcf/verification/profile.py",
    "spatialcf/verification/provenance.py",
    "spatialcf/verification/split.py",
    "spatialcf/verification/verifier.py",
)


def _semantic_retained_bytes(path: Path) -> bytes:
    with bound_absolute_directory(path.parent) as descriptor:
        before = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        payload = read_regular_at(
            descriptor,
            path.name,
            max(1, before.st_size),
            expected_stat=before,
        )
        revalidate_entries(descriptor, {path.name: before})
        return payload


def _semantic_runtime_provenance() -> RuntimeProvenance:
    """Derive the uncached runtime identity used by generation and replay."""

    module_root = Path(__file__).parents[3]
    modules = []
    for relative in _SEMANTIC_RUNTIME_MODULE_CLOSURE:
        payload = _semantic_retained_bytes(module_root / relative)
        modules.append(
            ModuleFileIdentity(
                package_relative_path=relative,
                byte_length=len(payload),
                byte_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    dependencies: dict[str, DependencyInventoryEntry] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise ValueError("installed dependency has no distribution name")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if name == "spatialcf":
            continue
        candidates = [
            item
            for item in distribution.files or ()
            if item.name in {"METADATA", "PKG-INFO"}
            and len(item.parts) == 2
            and item.parent.name.endswith((".dist-info", ".egg-info"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "installed dependency metadata identity is unavailable or ambiguous"
            )
        payload = _semantic_retained_bytes(
            Path(distribution.locate_file(candidates[0]))
        )
        value = DependencyInventoryEntry(
            distribution_name=name,
            distribution_version=distribution.version,
            metadata_byte_length=len(payload),
            metadata_byte_sha256=hashlib.sha256(payload).hexdigest(),
        )
        previous = dependencies.setdefault(name, value)
        if previous != value:
            raise ValueError("installed dependency identity is ambiguous")
    return RuntimeProvenance.seal(
        interpreter=InterpreterIdentity(
            implementation=(
                sys.implementation.name.capitalize().replace("Cpython", "CPython")
            ),
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            micro=sys.version_info.micro,
        ),
        dependencies=tuple(
            sorted(
                dependencies.values(),
                key=lambda item: canonical_json_bytes(item.distribution_name),
            )
        ),
        module_files=tuple(modules),
        lock_metadata=LockMetadata(
            availability="UNAVAILABLE",
            scope="INSTALLED_PACKAGE_METADATA",
        ),
    )


def _semantic_publication_error(output: Path, transaction, error: BaseException):
    if isinstance(error, CompetitionNativePublicationError):
        raise error
    try:
        transaction.parent.validate()
        reconciliation = (
            reconcile_owned_rename_at(
                transaction.parent.parent_descriptor,
                transaction.name,
                transaction.parent.output_name,
                transaction.identity,
            )
            if transaction.closed else transaction.reconcile()
        )
        transaction.parent.validate()
    except BaseException as reconcile_error:  # noqa: BLE001
        error.add_note(str(reconcile_error))
        published = None
        recovery_name = None
    else:
        published = (
            True
            if reconciliation.location is RenameLocation.OUTPUT
            else False
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
        recovery_name = (
            transaction.parent.output_name
            if reconciliation.location is RenameLocation.OUTPUT
            else transaction.name
            if reconciliation.location is RenameLocation.SOURCE
            else None
        )
    raise CompetitionNativePublicationError(
        output,
        published=published,
        recovery_name=recovery_name,
        detail="semantic dataset publication failed after a publication attempt",
    ) from error


# Preserve supported public names and pickle lookup.
_semantic_retained_bytes.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_runtime_provenance.__module__ = "spatialcf.generation.workflows.dataset"
_semantic_publication_error.__module__ = "spatialcf.generation.workflows.dataset"
