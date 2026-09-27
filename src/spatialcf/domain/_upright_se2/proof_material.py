"""Upright profile proof material contracts and intrinsic operations."""

from __future__ import annotations

from fractions import (
    Fraction,
)

from itertools import (
    pairwise,
)

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    BooleanValue,
    CanonicalIdValue,
    HashBoundCanonicalModel,
    RecordValue,
    TypedValue,
)

from spatialcf.domain.outcomes import (
    ResourceUsage,
)

from spatialcf.domain.serialization import (
    canonical_json_bytes,
)

from spatialcf.domain._upright_se2.cells import (
    UprightSE2CardinalProofTuple,
    UprightSE2CompiledCell,
    UprightSE2ContinuousProofTuple,
    UprightSE2CoverageArtifact,
    UprightSE2ProofLeafDisposition,
    UprightSE2RetainedOwnerOutcomeKind,
    _lifted_cell_order_key,
)

from spatialcf.domain._upright_se2.compilation import (
    UprightSE2Compilation,
    UprightSE2ContinuousCompilation,
)

from spatialcf.domain._upright_se2.endpoints import (
    UprightSE2ContinuousProposalCandidate,
    UprightSE2ProposalCandidate,
)

from spatialcf.domain._upright_se2.proof_evaluation import (
    UprightSE2ProofCellEvaluation,
    UprightSE2ProofFrontierRow,
    UprightSE2ProofPruneDecision,
    UprightSE2ProofResourceLedger,
)

from spatialcf.domain._upright_se2.source_binding import (
    _bound_continuous_source_input,
    _bound_source_input,
)

from spatialcf.domain._upright_se2.yaw import (
    ContinuousYawFullCircle,
)


def _proof_cell_is_within_root(
    cell: UprightSE2CompiledCell,
    root: UprightSE2CompiledCell,
) -> bool:
    """Return whether one proof row is a yaw-bound exact-dyadic root descendant."""

    return (
        cell.authorization_sha256 == root.authorization_sha256
        and cell.yaw_interval == root.yaw_interval
        and root.x_lower.as_fraction <= cell.x_lower.as_fraction
        and cell.x_upper.as_fraction <= root.x_upper.as_fraction
        and root.y_lower.as_fraction <= cell.y_lower.as_fraction
        and cell.y_upper.as_fraction <= root.y_upper.as_fraction
    )


def _proof_root_for_cell(
    cell: UprightSE2CompiledCell,
    roots: tuple[UprightSE2CompiledCell, ...],
) -> UprightSE2CompiledCell:
    """Resolve exactly one compilation root for a retained proof cell."""

    matches = tuple(root for root in roots if _proof_cell_is_within_root(cell, root))
    if len(matches) != 1:
        raise ValueError(
            "proof cells must bind one authorization/yaw compilation root domain"
        )
    return matches[0]


def _proof_validate_leaf_partition(
    roots: tuple[UprightSE2CompiledCell, ...],
    leaves: tuple[UprightSE2CompiledCell, ...],
) -> None:
    """Require final leaves to partition roots with lower-owned split seams.

    Retained-owner geometry remains closed.  Proof coverage instead assigns an
    internal X/Y seam to its lower child, while retaining the root's outer
    upper boundary.  A degenerate axis has one owned coordinate and cannot be
    widened or subdivided into synthetic area.
    """

    leaves_by_root: dict[bytes, list[UprightSE2CompiledCell]] = {
        canonical_json_bytes(root): [] for root in roots
    }
    for leaf in leaves:
        root = _proof_root_for_cell(leaf, roots)
        if (
            root.x_lower.as_fraction < root.x_upper.as_fraction
            and leaf.x_lower.as_fraction >= leaf.x_upper.as_fraction
        ) or (
            root.y_lower.as_fraction < root.y_upper.as_fraction
            and leaf.y_lower.as_fraction >= leaf.y_upper.as_fraction
        ):
            raise ValueError(
                "proof final leaves must not collapse a nondegenerate root axis"
            )
        leaves_by_root[canonical_json_bytes(root)].append(leaf)

    for root in roots:
        root_key = canonical_json_bytes(root)
        root_leaves = tuple(leaves_by_root[root_key])
        if not root_leaves:
            raise ValueError("proof coverage leaves must cover every compilation root")
        x_coordinates = tuple(
            sorted(
                {
                    root.x_lower.as_fraction,
                    root.x_upper.as_fraction,
                    *(
                        coordinate
                        for leaf in root_leaves
                        for coordinate in (
                            leaf.x_lower.as_fraction,
                            leaf.x_upper.as_fraction,
                        )
                    ),
                }
            )
        )
        y_coordinates = tuple(
            sorted(
                {
                    root.y_lower.as_fraction,
                    root.y_upper.as_fraction,
                    *(
                        coordinate
                        for leaf in root_leaves
                        for coordinate in (
                            leaf.y_lower.as_fraction,
                            leaf.y_upper.as_fraction,
                        )
                    ),
                }
            )
        )
        x_segments = (
            ((root.x_lower.as_fraction, root.x_upper.as_fraction),)
            if root.x_lower.as_fraction == root.x_upper.as_fraction
            else tuple(pairwise(x_coordinates))
        )
        y_segments = (
            ((root.y_lower.as_fraction, root.y_upper.as_fraction),)
            if root.y_lower.as_fraction == root.y_upper.as_fraction
            else tuple(pairwise(y_coordinates))
        )
        for x_lower, x_upper in x_segments:
            for y_lower, y_upper in y_segments:
                covering = tuple(
                    leaf
                    for leaf in root_leaves
                    if (
                        leaf.x_lower.as_fraction <= x_lower
                        and x_upper <= leaf.x_upper.as_fraction
                        and leaf.y_lower.as_fraction <= y_lower
                        and y_upper <= leaf.y_upper.as_fraction
                    )
                )
                if len(covering) != 1:
                    raise ValueError(
                        "proof coverage leaves must form one exact-dyadic partition"
                    )
        for x_coordinate in x_coordinates:
            for y_coordinate in y_coordinates:
                owning = tuple(
                    leaf
                    for leaf in root_leaves
                    if _proof_leaf_owns_coordinate(
                        lower=leaf.x_lower.as_fraction,
                        upper=leaf.x_upper.as_fraction,
                        coordinate=x_coordinate,
                        root_lower=root.x_lower.as_fraction,
                    )
                    and _proof_leaf_owns_coordinate(
                        lower=leaf.y_lower.as_fraction,
                        upper=leaf.y_upper.as_fraction,
                        coordinate=y_coordinate,
                        root_lower=root.y_lower.as_fraction,
                    )
                )
                if len(owning) != 1:
                    raise ValueError(
                        "proof coverage leaves must assign each split seam to one lower owner"
                    )


def _proof_leaf_owns_coordinate(
    *,
    lower: Fraction,
    upper: Fraction,
    coordinate: Fraction,
    root_lower: Fraction,
) -> bool:
    """Apply proof-only lower ownership while retaining the root outer lower."""

    return (coordinate == root_lower and lower == root_lower) or (
        lower < coordinate <= upper
    )


class UprightSE2ProofMaterial(HashBoundCanonicalModel):
    """Full hash-bound M3 cardinal proof material prior to independent checking."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/proof-material/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "proof_material_sha256"

    solve_request_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2Compilation
    cardinal_tuple_roster: tuple[UprightSE2CardinalProofTuple, ...]
    coverage_artifact: UprightSE2CoverageArtifact
    compiled_cell_sha256s: tuple[Sha256Digest, ...]
    evaluated_cells: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_order: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_candidates: tuple[UprightSE2ProposalCandidate, ...] = ()
    prune_decisions: tuple[UprightSE2ProofPruneDecision, ...]
    unresolved_frontier: tuple[UprightSE2ProofFrontierRow, ...]
    resource_ledger: UprightSE2ProofResourceLedger
    proof_material_sha256: Sha256Digest

    @property
    def total_resource_usage(self) -> ResourceUsage:
        """Expose the exact inner total callers must copy into their outer record."""

        return self.resource_ledger.canonical_total_resource_usage

    @model_validator(mode="after")
    def _validate_proof_material(self) -> Self:
        compilation = self.compilation
        if (
            self.solve_request_sha256 != compilation.solve_request_sha256
            or self.semantic_closure_sha256
            != compilation.semantic_closure.semantic_closure_sha256
            or self.upright_se2_compilation_sha256
            != compilation.upright_se2_compilation_sha256
        ):
            raise ValueError("proof material roots do not bind its compilation")
        source_input = _bound_source_input(
            compilation.source_solve_request.semantic_problem
        )
        if len(self.cardinal_tuple_roster) != 1:
            raise ValueError(
                "proof tuple roster must contain one request-authorized tuple"
            )
        proof_tuple = self.cardinal_tuple_roster[0]
        if (
            canonical_json_bytes(proof_tuple.authorization)
            != canonical_json_bytes(compilation.operation.authorization)
            or proof_tuple.reference_id != source_input.reference_id
            or proof_tuple.translation_domain
            != compilation.operation.translation_domain
            or tuple(canonical_json_bytes(cell) for cell in proof_tuple.compiled_cells)
            != tuple(canonical_json_bytes(cell) for cell in compilation.compiled_cells)
            or proof_tuple.authorization.operator_ref != source_input.operator_ref
            or proof_tuple.authorization.yaw.q != source_input.quarter_turns_ccw
        ):
            raise ValueError(
                "proof tuple roster does not bind the source authorization"
            )
        root_cells = proof_tuple.compiled_cells
        if any(
            (
                root.x_lower,
                root.x_upper,
                root.y_lower,
                root.y_upper,
            )
            != (
                proof_tuple.translation_domain.x_lower,
                proof_tuple.translation_domain.x_upper,
                proof_tuple.translation_domain.y_lower,
                proof_tuple.translation_domain.y_upper,
            )
            for root in root_cells
        ):
            raise ValueError(
                "proof tuple roots must bind the complete world-XY translation domain"
            )
        if (
            self.coverage_artifact.authorization_sha256
            != compilation.operation.authorization_sha256
        ):
            raise ValueError(
                "proof coverage does not bind the compilation authorization"
            )
        _proof_validate_leaf_partition(root_cells, self.coverage_artifact.cells)
        expected_cell_sha256s = tuple(
            cell.compiled_cell_sha256 for cell in self.coverage_artifact.cells
        )
        if self.compiled_cell_sha256s != expected_cell_sha256s:
            raise ValueError(
                "proof material must bind the complete canonical cell roster"
            )
        if self.evaluated_cells != tuple(
            sorted(
                self.evaluated_cells,
                key=lambda row: _lifted_cell_order_key(row.compiled_cell),
            )
        ):
            raise ValueError(
                "proof material evaluated cell roster must use deterministic order"
            )
        evaluated_cell_ids = tuple(
            row.compiled_cell.cell_id for row in self.evaluated_cells
        )
        if len(set(evaluated_cell_ids)) != len(evaluated_cell_ids):
            raise ValueError(
                "proof material evaluated cells must not duplicate a cell ID"
            )
        if len(
            {canonical_json_bytes(row.compiled_cell) for row in self.evaluated_cells}
        ) != len(self.evaluated_cells):
            raise ValueError("proof material evaluated cells must not duplicate a cell")
        for row in self.evaluated_cells:
            _proof_root_for_cell(row.compiled_cell, root_cells)
        final_leaf_rows = tuple(
            row for row in self.evaluated_cells if row.leaf_disposition is not None
        )
        if tuple(
            canonical_json_bytes(row.compiled_cell) for row in final_leaf_rows
        ) != tuple(canonical_json_bytes(cell) for cell in self.coverage_artifact.cells):
            raise ValueError(
                "proof evaluated cell roster has an incomplete, duplicated, or unordered final leaf roster"
            )

        expected_proposals = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        )
        if tuple(canonical_json_bytes(row) for row in self.proposal_order) != tuple(
            canonical_json_bytes(row) for row in expected_proposals
        ):
            raise ValueError("proof proposal order must be complete and deterministic")
        expected_proposal_cells = tuple(
            canonical_json_bytes(row) for row in expected_proposals
        )
        candidate_cells = tuple(
            canonical_json_bytes(candidate.final_inward_cell)
            for candidate in self.proposal_candidates
        )
        if len(set(candidate_cells)) != len(candidate_cells) or set(
            candidate_cells
        ) != set(expected_proposal_cells):
            raise ValueError("proof proposal candidates must cover final inward cells")
        if self.proposal_candidates != tuple(
            sorted(
                self.proposal_candidates,
                key=lambda candidate: candidate.canonical_order_key,
            )
        ):
            raise ValueError(
                "proof proposal candidates must use canonical witness order"
            )
        for candidate in self.proposal_candidates:
            point_evaluation = candidate.point_evaluation.point_cell_evaluation
            point_cell = point_evaluation.compiled_cell
            if not any(
                canonical_json_bytes(row) == canonical_json_bytes(point_evaluation)
                for row in self.evaluated_cells
            ):
                raise ValueError(
                    "proposal point evaluation must be present in evaluated cells"
                )
            point_root = _proof_root_for_cell(point_cell, root_cells)
            reuses_final_row = canonical_json_bytes(
                point_evaluation
            ) == canonical_json_bytes(candidate.final_inward_cell)
            if reuses_final_row:
                if (
                    canonical_json_bytes(point_cell) != canonical_json_bytes(point_root)
                    or point_cell.x_lower != point_cell.x_upper
                    or point_cell.y_lower != point_cell.y_upper
                ):
                    raise ValueError(
                        "proposal point may reuse only its byte-identical fully degenerate root final row"
                    )
            else:
                if point_evaluation.leaf_disposition is not None:
                    raise ValueError(
                        "nondegenerate proposal points require a distinct internal exact point row"
                    )
                final_cell = candidate.final_inward_cell.compiled_cell
                for lower, upper, coordinate in (
                    (
                        final_cell.x_lower.as_fraction,
                        final_cell.x_upper.as_fraction,
                        point_cell.x_lower.as_fraction,
                    ),
                    (
                        final_cell.y_lower.as_fraction,
                        final_cell.y_upper.as_fraction,
                        point_cell.y_lower.as_fraction,
                    ),
                ):
                    if (lower < upper and not lower < coordinate < upper) or (
                        lower == upper and coordinate != lower
                    ):
                        raise ValueError(
                            "nondegenerate proposal points require a strict-interior exact descendant"
                        )
            owning_leaves = tuple(
                row
                for row in final_leaf_rows
                if _proof_cell_is_within_root(point_cell, row.compiled_cell)
                and _proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.x_lower.as_fraction,
                    upper=row.compiled_cell.x_upper.as_fraction,
                    coordinate=point_cell.x_lower.as_fraction,
                    root_lower=point_root.x_lower.as_fraction,
                )
                and _proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.y_lower.as_fraction,
                    upper=row.compiled_cell.y_upper.as_fraction,
                    coordinate=point_cell.y_lower.as_fraction,
                    root_lower=point_root.y_lower.as_fraction,
                )
            )
            if len(owning_leaves) != 1 or canonical_json_bytes(
                owning_leaves[0]
            ) != canonical_json_bytes(candidate.final_inward_cell):
                raise ValueError(
                    "proposal point cell must be uniquely owned by its final inward leaf"
                )
            policy_terms = compilation.semantic_closure.objective_policy.terms
            if tuple(term.term_id for term in candidate.point_objective.terms) != tuple(
                term.term_id for term in policy_terms
            ):
                raise ValueError(
                    "proposal point objective terms do not bind the request policy"
                )
            expected_lower = sum(
                (
                    point_term.lower.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            expected_upper = sum(
                (
                    point_term.upper.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            if (
                candidate.point_objective.total_lower.as_fraction != expected_lower
                or candidate.point_objective.total_upper.as_fraction != expected_upper
            ):
                raise ValueError(
                    "proposal point objective total does not bind the request policy"
                )
            if (
                candidate.materialized_endpoint.upright_se2_compilation_sha256
                != compilation.upright_se2_compilation_sha256
                or candidate.program.semantic_problem_sha256
                != compilation.source_solve_request.semantic_problem_sha256
                or candidate.program.action_space_profile_sha256
                != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
                or candidate.program.before_state_sha256
                != compilation.source_solve_request.semantic_problem.scene_state.scene_state_sha256
                or candidate.program.grounded_obligation_set_sha256
                != compilation.grounded_obligations.grounded_obligation_set_sha256
                or candidate.program.state_delta_manifest
                != compilation.state_footprint.state_delta_manifest
            ):
                raise ValueError(
                    "proposal candidate program does not bind compilation roots"
                )
        expected_pruned = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED
        )
        if len(self.prune_decisions) != len(expected_pruned) or any(
            canonical_json_bytes(decision.cell_evaluation)
            != canonical_json_bytes(expected)
            for decision, expected in zip(
                self.prune_decisions, expected_pruned, strict=True
            )
        ):
            raise ValueError("proof prune decisions must be complete and deterministic")
        expected_unresolved = tuple(
            row
            for row in final_leaf_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED
        )
        if len(self.unresolved_frontier) != len(expected_unresolved) or any(
            canonical_json_bytes(row.cell_evaluation) != canonical_json_bytes(expected)
            for row, expected in zip(
                self.unresolved_frontier,
                expected_unresolved,
                strict=True,
            )
        ):
            raise ValueError(
                "proof unresolved frontier must be complete and deterministic"
            )
        if self.coverage_artifact.unresolved_cell_sha256s != tuple(
            sorted(
                row.compiled_cell.compiled_cell_sha256 for row in expected_unresolved
            )
        ):
            raise ValueError("proof coverage unresolved roster does not match frontier")

        expected_owner_evaluations = tuple(
            sorted(
                (
                    owner_evaluation
                    for row in self.evaluated_cells
                    for owner_evaluation in row.owner_evaluations
                ),
                key=canonical_json_bytes,
            )
        )
        staged_owner_evaluations = tuple(
            owner_evaluation
            for stage in self.resource_ledger.stage_deltas
            for owner_evaluation in stage.owner_evaluations
        )
        if tuple(sorted(staged_owner_evaluations, key=canonical_json_bytes)) != (
            expected_owner_evaluations
        ):
            raise ValueError(
                "proof shared ledger must account for every owner evaluation"
            )
        return self


def _continuous_proof_cell_within_root(
    cell: UprightSE2CompiledCell,
    root: UprightSE2CompiledCell,
) -> bool:
    """Return whether a proof cell is an exact-dyadic `(x, y, u)` descendant."""

    return (
        cell.authorization_sha256 == root.authorization_sha256
        and root.x_lower.as_fraction <= cell.x_lower.as_fraction
        and cell.x_upper.as_fraction <= root.x_upper.as_fraction
        and root.y_lower.as_fraction <= cell.y_lower.as_fraction
        and cell.y_upper.as_fraction <= root.y_upper.as_fraction
        and root.yaw_interval.lower.as_fraction <= cell.yaw_interval.lower.as_fraction
        and cell.yaw_interval.upper.as_fraction <= root.yaw_interval.upper.as_fraction
    )


def _continuous_proof_leaf_owns_coordinate(
    *,
    lower: Fraction,
    upper: Fraction,
    coordinate: Fraction,
    root_lower: Fraction,
    root_upper: Fraction,
    full_circle_alias: bool,
) -> bool:
    """Apply lower-owned seams, including the full-circle upper closure alias."""

    if full_circle_alias and coordinate == root_upper:
        return lower == root_lower
    return (coordinate == root_lower and lower == root_lower) or (
        lower < coordinate <= upper
    )


def _typed_continuous_bound_proves_inward_feasibility(value: TypedValue) -> bool:
    """Read one complete retained V4 feasibility record from a structural bound."""

    payload = value.payload
    if type(payload) is not RecordValue:
        return False
    fields = {field.name: field.value for field in payload.fields}
    inner_hard_constraint_proven = fields.get("inner_hard_constraint_proven")
    relation_inner_success = fields.get("relation_inner_success")
    classification = fields.get("visibility_classification")
    return (
        inner_hard_constraint_proven is not None
        and type(inner_hard_constraint_proven.payload) is BooleanValue
        and inner_hard_constraint_proven.payload.value is True
        and relation_inner_success is not None
        and type(relation_inner_success.payload) is BooleanValue
        and relation_inner_success.payload.value is True
        and classification is not None
        and type(classification.payload) is CanonicalIdValue
        and classification.payload.value == "INWARD"
    )


def _continuous_point_retains_inward_evidence(
    point_row: UprightSE2ProofCellEvaluation,
) -> bool:
    """Bind feasibility to one exact V4 compound row, not objective transport."""

    return any(
        evaluation.outcome_kind is UprightSE2RetainedOwnerOutcomeKind.EXACT
        and "proof:spatialcf/upright-se2/continuous-compound-cell"
        in evaluation.proof_rows
        and any(
            _typed_continuous_bound_proves_inward_feasibility(bound)
            for bound in evaluation.exact_bounds
        )
        for evaluation in point_row.owner_evaluations
    )


def _continuous_proof_validate_leaf_partition(
    root: UprightSE2CompiledCell,
    leaves: tuple[UprightSE2CompiledCell, ...],
    *,
    full_circle_alias: bool,
) -> None:
    """Require one exact-dyadic, lower-owned `(x, y, u)` leaf partition."""

    if not leaves or any(
        not _continuous_proof_cell_within_root(leaf, root) for leaf in leaves
    ):
        raise ValueError("continuous proof leaves must cover one exact root")
    axes = (
        ("x", root.x_lower.as_fraction, root.x_upper.as_fraction),
        ("y", root.y_lower.as_fraction, root.y_upper.as_fraction),
        ("u", root.yaw_interval.lower.as_fraction, root.yaw_interval.upper.as_fraction),
    )
    leaf_bounds = {
        "x": tuple(
            (leaf.x_lower.as_fraction, leaf.x_upper.as_fraction) for leaf in leaves
        ),
        "y": tuple(
            (leaf.y_lower.as_fraction, leaf.y_upper.as_fraction) for leaf in leaves
        ),
        "u": tuple(
            (
                leaf.yaw_interval.lower.as_fraction,
                leaf.yaw_interval.upper.as_fraction,
            )
            for leaf in leaves
        ),
    }
    coordinates: dict[str, tuple[Fraction, ...]] = {}
    segments: dict[str, tuple[tuple[Fraction, Fraction], ...]] = {}
    for axis, root_lower, root_upper in axes:
        values = tuple(
            sorted(
                {
                    root_lower,
                    root_upper,
                    *(value for pair in leaf_bounds[axis] for value in pair),
                }
            )
        )
        coordinates[axis] = values
        segments[axis] = (
            ((root_lower, root_upper),)
            if root_lower == root_upper
            else tuple(pairwise(values))
        )
    for x_lower, x_upper in segments["x"]:
        for y_lower, y_upper in segments["y"]:
            for u_lower, u_upper in segments["u"]:
                covering = tuple(
                    leaf
                    for leaf in leaves
                    if (
                        leaf.x_lower.as_fraction <= x_lower
                        and x_upper <= leaf.x_upper.as_fraction
                        and leaf.y_lower.as_fraction <= y_lower
                        and y_upper <= leaf.y_upper.as_fraction
                        and leaf.yaw_interval.lower.as_fraction <= u_lower
                        and u_upper <= leaf.yaw_interval.upper.as_fraction
                    )
                )
                if len(covering) != 1:
                    raise ValueError(
                        "continuous proof leaves must form one exact-dyadic partition"
                    )
    for x in coordinates["x"]:
        for y in coordinates["y"]:
            for u in coordinates["u"]:
                owners = tuple(
                    leaf
                    for leaf in leaves
                    if _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.x_lower.as_fraction,
                        upper=leaf.x_upper.as_fraction,
                        coordinate=x,
                        root_lower=root.x_lower.as_fraction,
                        root_upper=root.x_upper.as_fraction,
                        full_circle_alias=False,
                    )
                    and _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.y_lower.as_fraction,
                        upper=leaf.y_upper.as_fraction,
                        coordinate=y,
                        root_lower=root.y_lower.as_fraction,
                        root_upper=root.y_upper.as_fraction,
                        full_circle_alias=False,
                    )
                    and _continuous_proof_leaf_owns_coordinate(
                        lower=leaf.yaw_interval.lower.as_fraction,
                        upper=leaf.yaw_interval.upper.as_fraction,
                        coordinate=u,
                        root_lower=root.yaw_interval.lower.as_fraction,
                        root_upper=root.yaw_interval.upper.as_fraction,
                        full_circle_alias=full_circle_alias,
                    )
                )
                if len(owners) != 1:
                    raise ValueError(
                        "continuous proof leaves must assign every exact seam once"
                    )


class UprightSE2ContinuousProofMaterial(HashBoundCanonicalModel):
    """Complete untrusted continuous branch-and-bound evidence for Task 8.

    This is deliberately a profile proof transport only: fresh geometry replay,
    proof acceptance, certificates, and terminal result assembly remain outside
    this domain record and outside the proposal backend.
    """

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/continuous-proof-material/1.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "continuous_proof_material_sha256"

    solve_request_sha256: Sha256Digest
    semantic_closure_sha256: Sha256Digest
    continuous_upright_se2_compilation_sha256: Sha256Digest
    compilation: UprightSE2ContinuousCompilation
    continuous_tuple_roster: tuple[UprightSE2ContinuousProofTuple, ...]
    coverage_artifact: UprightSE2CoverageArtifact
    compiled_cell_sha256s: tuple[Sha256Digest, ...]
    evaluated_cells: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_order: tuple[UprightSE2ProofCellEvaluation, ...]
    proposal_candidates: tuple[UprightSE2ContinuousProposalCandidate, ...] = ()
    incumbent_candidate_sha256: Sha256Digest | None
    prune_decisions: tuple[UprightSE2ProofPruneDecision, ...]
    unresolved_frontier: tuple[UprightSE2ProofFrontierRow, ...]
    resource_ledger: UprightSE2ProofResourceLedger
    continuous_proof_material_sha256: Sha256Digest

    @property
    def total_resource_usage(self) -> ResourceUsage:
        """Expose the one shared ledger total for the outer submission."""

        return self.resource_ledger.canonical_total_resource_usage

    @model_validator(mode="after")
    def _validate_continuous_proof_material(self) -> Self:
        compilation = self.compilation
        if (
            self.solve_request_sha256 != compilation.solve_request_sha256
            or self.semantic_closure_sha256
            != compilation.semantic_closure.semantic_closure_sha256
            or self.continuous_upright_se2_compilation_sha256
            != compilation.continuous_upright_se2_compilation_sha256
        ):
            raise ValueError("continuous proof material roots do not bind compilation")
        if len(self.continuous_tuple_roster) != 1:
            raise ValueError("continuous proof tuple roster must contain one tuple")
        proof_tuple = self.continuous_tuple_roster[0]
        source_input = _bound_continuous_source_input(
            compilation.source_solve_request.semantic_problem
        )
        if (
            canonical_json_bytes(proof_tuple.authorization)
            != canonical_json_bytes(compilation.operation.authorization)
            or proof_tuple.reference_id != source_input.reference_id
            or proof_tuple.translation_domain
            != compilation.operation.translation_domain
            or proof_tuple.continuous_yaw_lift != compilation.continuous_yaw_lift
            or tuple(canonical_json_bytes(cell) for cell in proof_tuple.compiled_cells)
            != tuple(canonical_json_bytes(cell) for cell in compilation.compiled_cells)
            or proof_tuple.authorization.operator_ref != source_input.operator_ref
        ):
            raise ValueError("continuous proof tuple does not bind the source root")
        root = proof_tuple.compiled_cells[0]
        if (
            self.coverage_artifact.authorization_sha256
            != compilation.operation.authorization_sha256
        ):
            raise ValueError("continuous proof coverage does not bind authorization")
        _continuous_proof_validate_leaf_partition(
            root,
            self.coverage_artifact.cells,
            full_circle_alias=(
                type(compilation.operation.yaw_domain) is ContinuousYawFullCircle
            ),
        )
        expected_cell_sha256s = tuple(
            cell.compiled_cell_sha256 for cell in self.coverage_artifact.cells
        )
        if self.compiled_cell_sha256s != expected_cell_sha256s:
            raise ValueError("continuous proof must bind the canonical leaf roster")
        if self.evaluated_cells != tuple(
            sorted(
                self.evaluated_cells,
                key=lambda row: _lifted_cell_order_key(row.compiled_cell),
            )
        ):
            raise ValueError("continuous evaluated cells must use deterministic order")
        if len({row.compiled_cell.cell_id for row in self.evaluated_cells}) != len(
            self.evaluated_cells
        ) or len(
            {canonical_json_bytes(row.compiled_cell) for row in self.evaluated_cells}
        ) != len(self.evaluated_cells):
            raise ValueError("continuous proof rows must not duplicate cells")
        if any(
            not _continuous_proof_cell_within_root(row.compiled_cell, root)
            for row in self.evaluated_cells
        ):
            raise ValueError("continuous proof rows must be root descendants")
        final_rows = tuple(
            row for row in self.evaluated_cells if row.leaf_disposition is not None
        )
        if tuple(
            canonical_json_bytes(row.compiled_cell) for row in final_rows
        ) != tuple(canonical_json_bytes(cell) for cell in self.coverage_artifact.cells):
            raise ValueError("continuous final rows must exactly bind coverage leaves")
        expected_proposals = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.INWARD_FEASIBLE
        )
        if tuple(canonical_json_bytes(row) for row in self.proposal_order) != tuple(
            canonical_json_bytes(row) for row in expected_proposals
        ):
            raise ValueError("continuous proposal order must be complete and stable")
        candidate_cells = tuple(
            canonical_json_bytes(candidate.final_inward_cell)
            for candidate in self.proposal_candidates
        )
        expected_inward_candidate_cells = tuple(
            canonical_json_bytes(row) for row in expected_proposals
        )
        final_rows_by_bytes = {canonical_json_bytes(row): row for row in final_rows}
        if len(set(candidate_cells)) != len(candidate_cells) or not set(
            expected_inward_candidate_cells
        ).issubset(candidate_cells):
            raise ValueError(
                "continuous candidates must cover inward leaves exactly once"
            )
        if any(
            candidate_cell not in final_rows_by_bytes
            for candidate_cell in candidate_cells
        ):
            raise ValueError("continuous candidate parent must be a final proof leaf")
        if self.proposal_candidates != tuple(
            sorted(
                self.proposal_candidates,
                key=lambda candidate: candidate.canonical_order_key,
            )
        ):
            raise ValueError("continuous proposal candidates use the wrong order")
        expected_incumbent = (
            None
            if not self.proposal_candidates
            else self.proposal_candidates[0].continuous_proposal_candidate_sha256
        )
        if self.incumbent_candidate_sha256 != expected_incumbent:
            raise ValueError("continuous incumbent must be the canonical witness")
        for candidate in self.proposal_candidates:
            point_row = candidate.point_evaluation.point_cell_evaluation
            parent = candidate.final_inward_cell
            if parent.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED:
                if not any(
                    canonical_json_bytes(frontier.cell_evaluation)
                    == canonical_json_bytes(parent)
                    for frontier in self.unresolved_frontier
                ):
                    raise ValueError(
                        "continuous unresolved candidate parent must be in frontier"
                    )
                if point_row.leaf_disposition is not None:
                    raise ValueError(
                        "continuous unresolved candidates require an internal point row"
                    )
                if point_row.compiled_cell.cell_id != (
                    f"{parent.compiled_cell.cell_id}/proposal-point"
                ):
                    raise ValueError(
                        "continuous unresolved candidate point must bind its parent identity"
                    )
                if not _continuous_point_retains_inward_evidence(point_row):
                    raise ValueError(
                        "continuous unresolved candidate requires inward feasibility evidence"
                    )
            if not any(
                canonical_json_bytes(row) == canonical_json_bytes(point_row)
                for row in self.evaluated_cells
            ):
                raise ValueError("continuous candidate point row is absent from proof")
            point_cell = point_row.compiled_cell
            selected_u = candidate.selected_lifted_yaw.as_fraction
            if (
                point_cell.yaw_interval.lower.as_fraction != selected_u
                or point_cell.yaw_interval.upper.as_fraction != selected_u
            ):
                raise ValueError("continuous candidate point yaw is not degenerate")
            if (
                type(compilation.operation.yaw_domain) is ContinuousYawFullCircle
                and selected_u == root.yaw_interval.upper.as_fraction
            ):
                raise ValueError("full-circle upper closure alias cannot be a witness")
            final_cell = candidate.final_inward_cell.compiled_cell
            for lower, upper, coordinate in (
                (
                    final_cell.x_lower.as_fraction,
                    final_cell.x_upper.as_fraction,
                    point_cell.x_lower.as_fraction,
                ),
                (
                    final_cell.y_lower.as_fraction,
                    final_cell.y_upper.as_fraction,
                    point_cell.y_lower.as_fraction,
                ),
                (
                    final_cell.yaw_interval.lower.as_fraction,
                    final_cell.yaw_interval.upper.as_fraction,
                    selected_u,
                ),
            ):
                if (lower < upper and not lower < coordinate < upper) or (
                    lower == upper and coordinate != lower
                ):
                    raise ValueError(
                        "continuous proposal points require strict interior descendants"
                    )
            owners = tuple(
                row
                for row in final_rows
                if _continuous_proof_cell_within_root(point_cell, row.compiled_cell)
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.x_lower.as_fraction,
                    upper=row.compiled_cell.x_upper.as_fraction,
                    coordinate=point_cell.x_lower.as_fraction,
                    root_lower=root.x_lower.as_fraction,
                    root_upper=root.x_upper.as_fraction,
                    full_circle_alias=False,
                )
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.y_lower.as_fraction,
                    upper=row.compiled_cell.y_upper.as_fraction,
                    coordinate=point_cell.y_lower.as_fraction,
                    root_lower=root.y_lower.as_fraction,
                    root_upper=root.y_upper.as_fraction,
                    full_circle_alias=False,
                )
                and _continuous_proof_leaf_owns_coordinate(
                    lower=row.compiled_cell.yaw_interval.lower.as_fraction,
                    upper=row.compiled_cell.yaw_interval.upper.as_fraction,
                    coordinate=selected_u,
                    root_lower=root.yaw_interval.lower.as_fraction,
                    root_upper=root.yaw_interval.upper.as_fraction,
                    full_circle_alias=(
                        type(compilation.operation.yaw_domain)
                        is ContinuousYawFullCircle
                    ),
                )
            )
            if len(owners) != 1 or canonical_json_bytes(
                owners[0]
            ) != canonical_json_bytes(candidate.final_inward_cell):
                raise ValueError("continuous candidate point lacks one owning leaf")
            policy_terms = compilation.semantic_closure.objective_policy.terms
            if tuple(term.term_id for term in candidate.point_objective.terms) != tuple(
                term.term_id for term in policy_terms
            ):
                raise ValueError("continuous candidate terms do not bind policy")
            expected_lower = sum(
                (
                    point_term.lower.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            expected_upper = sum(
                (
                    point_term.upper.as_fraction
                    * Fraction.from_float(policy_term.weight)
                    / Fraction.from_float(policy_term.normalizer)
                    for point_term, policy_term in zip(
                        candidate.point_objective.terms,
                        policy_terms,
                        strict=True,
                    )
                ),
                start=Fraction(0),
            )
            if (
                candidate.point_objective.total_lower.as_fraction != expected_lower
                or candidate.point_objective.total_upper.as_fraction != expected_upper
            ):
                raise ValueError("continuous candidate total does not bind policy")
            if (
                candidate.materialized_endpoint.continuous_upright_se2_compilation_sha256
                != compilation.continuous_upright_se2_compilation_sha256
                or candidate.program.semantic_problem_sha256
                != compilation.source_solve_request.semantic_problem_sha256
                or candidate.program.action_space_profile_sha256
                != compilation.semantic_closure.profile_registration.action_space_profile.action_space_profile_sha256
                or candidate.program.before_state_sha256
                != compilation.source_solve_request.semantic_problem.scene_state.scene_state_sha256
                or candidate.program.grounded_obligation_set_sha256
                != compilation.grounded_obligations.grounded_obligation_set_sha256
                or candidate.program.state_delta_manifest
                != compilation.state_footprint.state_delta_manifest
            ):
                raise ValueError("continuous candidate program does not bind roots")
        expected_pruned = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.PRUNED
        )
        if len(self.prune_decisions) != len(expected_pruned) or any(
            canonical_json_bytes(decision.cell_evaluation)
            != canonical_json_bytes(expected)
            for decision, expected in zip(
                self.prune_decisions, expected_pruned, strict=True
            )
        ):
            raise ValueError("continuous prune decisions must be complete")
        expected_unresolved = tuple(
            row
            for row in final_rows
            if row.leaf_disposition is UprightSE2ProofLeafDisposition.UNRESOLVED
        )
        if len(self.unresolved_frontier) != len(expected_unresolved) or any(
            canonical_json_bytes(frontier.cell_evaluation)
            != canonical_json_bytes(expected)
            for frontier, expected in zip(
                self.unresolved_frontier, expected_unresolved, strict=True
            )
        ):
            raise ValueError("continuous frontier must be complete")
        if self.coverage_artifact.unresolved_cell_sha256s != tuple(
            sorted(
                row.compiled_cell.compiled_cell_sha256 for row in expected_unresolved
            )
        ):
            raise ValueError("continuous coverage frontier does not bind leaves")
        expected_owner_evaluations = tuple(
            sorted(
                (
                    evaluation
                    for row in self.evaluated_cells
                    for evaluation in row.owner_evaluations
                ),
                key=canonical_json_bytes,
            )
        )
        staged_owner_evaluations = tuple(
            evaluation
            for stage in self.resource_ledger.stage_deltas
            for evaluation in stage.owner_evaluations
        )
        if tuple(sorted(staged_owner_evaluations, key=canonical_json_bytes)) != (
            expected_owner_evaluations
        ):
            raise ValueError("continuous shared ledger must account for every owner")
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2ProofMaterial.model_rebuild()
UprightSE2ContinuousProofMaterial.model_rebuild()


# Keep supported public import and pickle lookup stable.
_proof_cell_is_within_root.__module__ = "spatialcf.domain.upright_se2"
_proof_root_for_cell.__module__ = "spatialcf.domain.upright_se2"
_proof_validate_leaf_partition.__module__ = "spatialcf.domain.upright_se2"
_proof_leaf_owns_coordinate.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ProofMaterial.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_cell_within_root.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_leaf_owns_coordinate.__module__ = "spatialcf.domain.upright_se2"
_typed_continuous_bound_proves_inward_feasibility.__module__ = "spatialcf.domain.upright_se2"
_continuous_point_retains_inward_evidence.__module__ = "spatialcf.domain.upright_se2"
_continuous_proof_validate_leaf_partition.__module__ = "spatialcf.domain.upright_se2"
UprightSE2ContinuousProofMaterial.__module__ = "spatialcf.domain.upright_se2"
