"""Analytic candidate-region construction for spatial counterfactuals."""

from spatialcf.solver.certified_constraints import CertifiedConstraintBuilder
from spatialcf.solver.certified_models import (
    CertifiedSolverConfig,
    CertifiedSolveResult,
    ConstraintBracketSummary,
    ConstraintBuildDiagnostics,
    ConstraintDiagnosticStep,
    ConstraintRegionSummary,
    OptimalityCertificate,
)
from spatialcf.solver.continuous import CertifiedSpatialCFSolver
from spatialcf.solver.execution import (
    CandidateExecution,
    CandidateExecutionStatus,
    CandidateExecutor,
    ExecutionResidual,
)
from spatialcf.solver.objective import ObjectiveBreakdown, ObjectiveWeights
from spatialcf.solver.search import (
    GroundedCandidate,
    GroundedCandidateAttempt,
    GroundedSolveResult,
    MinimumCostSpatialCFSolver,
    SearchConfig,
    SolveResult,
    SpatialCFSolver,
)

__all__ = (
    "CandidateExecution",
    "CandidateExecutionStatus",
    "CandidateExecutor",
    "CertifiedConstraintBuilder",
    "CertifiedSolveResult",
    "CertifiedSolverConfig",
    "CertifiedSpatialCFSolver",
    "ConstraintBracketSummary",
    "ConstraintBuildDiagnostics",
    "ConstraintDiagnosticStep",
    "ConstraintRegionSummary",
    "ExecutionResidual",
    "GroundedCandidate",
    "GroundedCandidateAttempt",
    "GroundedSolveResult",
    "MinimumCostSpatialCFSolver",
    "ObjectiveBreakdown",
    "ObjectiveWeights",
    "OptimalityCertificate",
    "SearchConfig",
    "SolveResult",
    "SpatialCFSolver",
)
