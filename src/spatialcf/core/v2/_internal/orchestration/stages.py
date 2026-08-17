"""Private normalization-stage adapters for registered Canonical v2 solves."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import spatialcf.core.v2.camera_translation as _camera_translation
import spatialcf.core.v2.cardinal_yaw as _cardinal_yaw
import spatialcf.core.v2.zero_distortion as _zero_distortion
from spatialcf.core.v2._internal.boundary import (
    InvalidCallerInputV2 as _InvalidInputV2,
)
from spatialcf.core.v2._internal.boundary import (
    NumericBoundaryGapV2 as _NumericInputV2,
)
from spatialcf.core.v2._internal.boundary import strict_input_model_v2
from spatialcf.core.v2._internal.orchestration.capabilities import SolveStageKeyV2
from spatialcf.core.v2.certificate_builder import _ExactCardinalSelectionFrameV2
from spatialcf.core.v2.minimum_cost_solver import (
    CanonicalMinimumCostSolveOutcomeV2,
    _valid_uncertified,
)
from spatialcf.domain.v2.base import FactSetV2, V2Model
from spatialcf.domain.v2.cardinal import (
    CanonicalSceneV2_1,
    SchemaIdentityV2_1,
    SemanticProblemV2_1,
)
from spatialcf.domain.v2.problem import SemanticProblemV2
from spatialcf.domain.v2.result import (
    CoreSolverConfigV2,
    UncertifiedReasonV2,
    UncertifiedResultV2,
)


@dataclass(frozen=True, slots=True)
class PreparedSolveStageV2:
    """One checked outer stage and the normalized inputs for its child."""

    stage_key: SolveStageKeyV2
    checked_problem: SemanticProblemV2 | SemanticProblemV2_1
    checked_config: CoreSolverConfigV2
    normalized_problem: SemanticProblemV2 | SemanticProblemV2_1
    internal_config: CoreSolverConfigV2
    preprocessing_domain_operations: int
    child_selection_frame: _ExactCardinalSelectionFrameV2
    child_numeric_finding: str
    child_missing_finding: str
    rebind_numeric_finding: str
    original_to_internal_quarter_turns_ccw: int | None = None

    def __post_init__(self) -> None:
        if type(self.stage_key) is not SolveStageKeyV2:
            raise TypeError("stage_key must be an exact SolveStageKeyV2")
        if not isinstance(self.checked_problem, V2Model) or not isinstance(
            self.normalized_problem,
            V2Model,
        ):
            raise TypeError("stage problems must be Canonical V2 models")
        if (
            type(self.checked_config) is not CoreSolverConfigV2
            or type(self.internal_config) is not CoreSolverConfigV2
        ):
            raise TypeError("stage configs must be exact CoreSolverConfigV2")
        if (
            type(self.preprocessing_domain_operations) is not int
            or self.preprocessing_domain_operations < 0
        ):
            raise TypeError("preprocessing usage must be a non-negative exact int")
        if type(self.child_selection_frame) is not _ExactCardinalSelectionFrameV2:
            raise TypeError("child selection frame has the wrong exact type")
        for name in (
            "child_numeric_finding",
            "child_missing_finding",
            "rebind_numeric_finding",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise TypeError(f"{name} must be an exact non-blank string")
        quarter_turns = self.original_to_internal_quarter_turns_ccw
        if self.stage_key is SolveStageKeyV2.CAMERA_CARDINAL_REBASE:
            if type(quarter_turns) is not int or quarter_turns not in range(4):
                raise TypeError("camera-cardinal stage requires exact quarter turns")
        elif quarter_turns is not None:
            raise ValueError("only camera-cardinal stage can carry quarter turns")


def prepare_solve_stage_v2(
    stage_key: SolveStageKeyV2,
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> PreparedSolveStageV2 | CanonicalMinimumCostSolveOutcomeV2:
    """Validate and normalize one registered forward stage."""

    if type(stage_key) is not SolveStageKeyV2:
        raise TypeError("stage_key must be an exact SolveStageKeyV2")
    if type(selection_frame) is not _ExactCardinalSelectionFrameV2:
        raise TypeError("selection_frame has the wrong exact type")
    if stage_key is SolveStageKeyV2.CARDINAL_YAW:
        return _prepare_cardinal_stage(problem, config, selection_frame)
    if stage_key is SolveStageKeyV2.ZERO_DISTORTION:
        return _prepare_zero_distortion_stage(problem, config, selection_frame)
    if stage_key is SolveStageKeyV2.CAMERA_TRANSLATION:
        return _prepare_camera_translation_stage(problem, config, selection_frame)
    if stage_key is SolveStageKeyV2.CAMERA_CARDINAL_REBASE:
        return _prepare_camera_cardinal_stage(problem, config, selection_frame)
    raise NotImplementedError(f"solve stage has not been migrated: {stage_key.value}")


def child_numeric_outcome_v2(
    prepared: PreparedSolveStageV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Map an arithmetic failure from a normalized child replay."""

    _require_prepared_stage(prepared)
    return _valid_uncertified(
        prepared.checked_problem,
        prepared.checked_config,
        UncertifiedReasonV2.NUMERIC_GAP,
        prepared.child_numeric_finding,
    )


def child_missing_outcome_v2(
    prepared: PreparedSolveStageV2,
    replay: CanonicalMinimumCostSolveOutcomeV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Close a normalized child replay that did not return a result."""

    _require_prepared_stage(prepared)
    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("child replay has the wrong exact outcome type")
    if replay.result is not None:
        raise ValueError("child replay is not missing its result")
    return _valid_uncertified(
        prepared.checked_problem,
        prepared.checked_config,
        UncertifiedReasonV2.COMPILATION_INCOMPLETE,
        *(replay.finding_codes or (prepared.child_missing_finding,)),
    )


def rebind_solve_stage_v2(
    prepared: PreparedSolveStageV2,
    replay: CanonicalMinimumCostSolveOutcomeV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    """Rebind one normalized child result into its checked outer problem."""

    _require_prepared_stage(prepared)
    if type(replay) is not CanonicalMinimumCostSolveOutcomeV2:
        raise TypeError("child replay has the wrong exact outcome type")
    if prepared.stage_key is SolveStageKeyV2.CAMERA_CARDINAL_REBASE:
        return _rebind_camera_cardinal_stage(prepared, replay)
    if prepared.stage_key not in (
        SolveStageKeyV2.CARDINAL_YAW,
        SolveStageKeyV2.ZERO_DISTORTION,
        SolveStageKeyV2.CAMERA_TRANSLATION,
    ):
        raise NotImplementedError(
            f"solve stage has not been migrated: {prepared.stage_key.value}"
        )

    from spatialcf.core.v2 import minimum_cost_solver_v2_1 as legacy_cardinal

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            candidate_variable_override = None
            if prepared.stage_key is SolveStageKeyV2.CAMERA_TRANSLATION:
                candidate_variable_override = (
                    _camera_translation._candidate_variable_for_original_problem_v2_3(
                        prepared.checked_problem
                    )
                )
            return legacy_cardinal._rebind_outcome(
                replay,
                prepared.checked_problem,
                prepared.checked_config,
                prepared.preprocessing_domain_operations,
                candidate_variable_override=candidate_variable_override,
            )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            prepared.checked_problem,
            prepared.checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            prepared.rebind_numeric_finding,
        )


def _prepare_cardinal_stage(
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> PreparedSolveStageV2 | CanonicalMinimumCostSolveOutcomeV2:
    if (
        type(problem) is not SemanticProblemV2_1
        or type(getattr(problem, "schema_identity", None)) is not SchemaIdentityV2_1
    ):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    if problem.schema_identity.schema_version != "2.1":
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    try:
        checked_config = strict_input_model_v2(
            config,
            CoreSolverConfigV2,
            "CORE_SOLVER_CONFIG",
        )
    except (_NumericInputV2, _InvalidInputV2) as error:
        return _missing_result(error.finding_code)

    budget = _cardinal_yaw.CardinalDomainBudgetV2(
        limit=checked_config.max_domain_operations
    )
    try:
        _cardinal_yaw.reserve_cardinal_problem_structure_v2_1(problem, budget)
    except _cardinal_yaw.CardinalResourceLimitV2:
        return _missing_result("RESOURCE_LIMIT:CARDINAL_PREPROCESSING")
    except (AttributeError, TypeError):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM")
    try:
        checked_problem = strict_input_model_v2(
            problem,
            SemanticProblemV2_1,
            "SEMANTIC_PROBLEM",
        )
    except (_NumericInputV2, _InvalidInputV2) as error:
        return _missing_result(error.finding_code)

    registry_finding = _cardinal_yaw.registry_finding_v2_1(checked_config)
    if registry_finding is not None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            cardinal = _cardinal_yaw.prepare_cardinal_problem_v2_1(
                checked_problem,
                checked_config,
                budget.used,
            )
    except _cardinal_yaw.CardinalResourceLimitV2:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            "RESOURCE_LIMIT:CARDINAL_PREPROCESSING",
        )
    except _cardinal_yaw.CardinalUnsupportedModelV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CARDINAL_NORMALIZATION",
        )

    return PreparedSolveStageV2(
        stage_key=SolveStageKeyV2.CARDINAL_YAW,
        checked_problem=checked_problem,
        checked_config=checked_config,
        normalized_problem=cardinal.normalized_problem,
        internal_config=cardinal.internal_config,
        preprocessing_domain_operations=cardinal.preprocessing_domain_operations,
        child_selection_frame=selection_frame,
        child_numeric_finding="NUMERIC_GAP:V2_0_SOLVE_REPLAY",
        child_missing_finding="NORMALIZED_REPLAY_HAS_NO_RESULT",
        rebind_numeric_finding="NUMERIC_GAP:CARDINAL_RESULT_REBIND",
    )


def _prepare_zero_distortion_stage(
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> PreparedSolveStageV2 | CanonicalMinimumCostSolveOutcomeV2:
    if (
        type(problem) is not SemanticProblemV2_1
        or type(getattr(problem, "schema_identity", None)) is not SchemaIdentityV2_1
    ):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    if problem.schema_identity.schema_version != "2.1":
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    try:
        checked_config = strict_input_model_v2(
            config,
            CoreSolverConfigV2,
            "CORE_SOLVER_CONFIG",
        )
    except (_NumericInputV2, _InvalidInputV2) as error:
        return _missing_result(error.finding_code)

    budget = _zero_distortion.ZeroDistortionDomainBudgetV2(
        limit=checked_config.max_domain_operations
    )
    if not _zero_reservation_input_is_shallow_valid(problem):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM")
    try:
        _zero_distortion.reserve_zero_distortion_problem_structure_v2_2(problem, budget)
    except _zero_distortion.ZeroDistortionResourceLimitV2:
        return _missing_result("RESOURCE_LIMIT:ZERO_DISTORTION_PREPROCESSING")
    try:
        checked_problem = strict_input_model_v2(
            problem,
            SemanticProblemV2_1,
            "SEMANTIC_PROBLEM",
        )
    except (_NumericInputV2, _InvalidInputV2) as error:
        return _missing_result(error.finding_code)

    registry_finding = _zero_distortion.registry_finding_v2_2(checked_config)
    if registry_finding is not None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            zero_distortion = _zero_distortion.prepare_zero_distortion_problem_v2_2(
                checked_problem,
                checked_config,
                budget.used,
            )
    except _zero_distortion.ZeroDistortionResourceLimitV2:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            "RESOURCE_LIMIT:ZERO_DISTORTION_PREPROCESSING",
        )
    except _zero_distortion.ZeroDistortionUnsupportedModelV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:ZERO_DISTORTION_NORMALIZATION",
        )

    return PreparedSolveStageV2(
        stage_key=SolveStageKeyV2.ZERO_DISTORTION,
        checked_problem=checked_problem,
        checked_config=checked_config,
        normalized_problem=zero_distortion.normalized_problem,
        internal_config=zero_distortion.internal_config,
        preprocessing_domain_operations=(
            zero_distortion.preprocessing_domain_operations
        ),
        child_selection_frame=selection_frame,
        child_numeric_finding="NUMERIC_GAP:V2_1_SOLVE_REPLAY",
        child_missing_finding="NORMALIZED_REPLAY_HAS_NO_RESULT",
        rebind_numeric_finding="NUMERIC_GAP:ZERO_DISTORTION_RESULT_REBIND",
    )


def _prepare_camera_translation_stage(
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> PreparedSolveStageV2 | CanonicalMinimumCostSolveOutcomeV2:
    if (
        type(problem) is not SemanticProblemV2_1
        or type(getattr(problem, "schema_identity", None)) is not SchemaIdentityV2_1
    ):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    if problem.schema_identity.schema_version != "2.1":
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    try:
        checked_config = strict_input_model_v2(
            config,
            CoreSolverConfigV2,
            "CORE_SOLVER_CONFIG",
        )
    except (
        _NumericInputV2,
        _InvalidInputV2,
    ) as error:
        return _missing_result(error.finding_code)

    budget = _camera_translation.CameraTranslationDomainBudgetV2(
        limit=checked_config.max_domain_operations
    )
    if not _camera_reservation_input_is_shallow_valid(problem):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM")
    try:
        _camera_translation.reserve_camera_translation_problem_structure_v2_3(
            problem,
            budget,
        )
    except _camera_translation.CameraTranslationResourceLimitV2:
        return _missing_result("RESOURCE_LIMIT:CAMERA_TRANSLATION_PREPROCESSING")
    try:
        checked_problem = strict_input_model_v2(
            problem,
            SemanticProblemV2_1,
            "SEMANTIC_PROBLEM",
        )
    except (
        _NumericInputV2,
        _InvalidInputV2,
    ) as error:
        return _missing_result(error.finding_code)

    registry_finding = _camera_translation.registry_finding_v2_3(checked_config)
    if registry_finding is not None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            camera_translation = (
                _camera_translation.prepare_camera_translation_problem_v2_3(
                    checked_problem,
                    checked_config,
                    budget.used,
                )
            )
    except _camera_translation.CameraTranslationResourceLimitV2:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            "RESOURCE_LIMIT:CAMERA_TRANSLATION_PREPROCESSING",
        )
    except _camera_translation.CameraTranslationUnsupportedModelV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except _camera_translation.CameraTranslationNumericGapV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            error.finding_code,
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CAMERA_TRANSLATION_NORMALIZATION",
        )

    return PreparedSolveStageV2(
        stage_key=SolveStageKeyV2.CAMERA_TRANSLATION,
        checked_problem=checked_problem,
        checked_config=checked_config,
        normalized_problem=camera_translation.normalized_problem,
        internal_config=camera_translation.internal_config,
        preprocessing_domain_operations=(
            camera_translation.preprocessing_domain_operations
        ),
        child_selection_frame=selection_frame,
        child_numeric_finding="NUMERIC_GAP:V2_2_SOLVE_REPLAY",
        child_missing_finding="NORMALIZED_REPLAY_HAS_NO_RESULT",
        rebind_numeric_finding="NUMERIC_GAP:CAMERA_TRANSLATION_RESULT_REBIND",
    )


def _prepare_camera_cardinal_stage(
    problem: object,
    config: object,
    selection_frame: _ExactCardinalSelectionFrameV2,
) -> PreparedSolveStageV2 | CanonicalMinimumCostSolveOutcomeV2:
    from spatialcf.core.v2 import camera_cardinal_rebase as camera_cardinal

    if (
        type(problem) is not SemanticProblemV2_1
        or type(getattr(problem, "schema_identity", None)) is not SchemaIdentityV2_1
    ):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    if problem.schema_identity.schema_version != "2.1":
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM_SCHEMA_VERSION")
    try:
        checked_config = strict_input_model_v2(
            config,
            CoreSolverConfigV2,
            "CORE_SOLVER_CONFIG",
        )
    except (
        _NumericInputV2,
        _InvalidInputV2,
    ) as error:
        return _missing_result(error.finding_code)

    budget = camera_cardinal.CameraCardinalRebaseDomainBudgetV2(
        limit=checked_config.max_domain_operations
    )
    if not _camera_reservation_input_is_shallow_valid(problem):
        return _missing_result("INVALID_INPUT:SEMANTIC_PROBLEM")
    try:
        camera_cardinal.reserve_camera_cardinal_rebase_problem_structure_v2_4(
            problem,
            budget,
        )
    except camera_cardinal.CameraCardinalRebaseResourceLimitV2:
        return _missing_result("RESOURCE_LIMIT:CAMERA_CARDINAL_PREPROCESSING")
    try:
        checked_problem = strict_input_model_v2(
            problem,
            SemanticProblemV2_1,
            "SEMANTIC_PROBLEM",
        )
    except (
        _NumericInputV2,
        _InvalidInputV2,
    ) as error:
        return _missing_result(error.finding_code)

    registry_finding = camera_cardinal.registry_finding_v2_4(checked_config)
    if registry_finding is not None:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            registry_finding,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            camera_cardinal = (
                camera_cardinal.prepare_camera_cardinal_rebase_problem_v2_4(
                    checked_problem,
                    checked_config,
                    budget.used,
                )
            )
    except camera_cardinal.CameraCardinalRebaseResourceLimitV2:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            "RESOURCE_LIMIT:CAMERA_CARDINAL_PREPROCESSING",
        )
    except camera_cardinal.CameraCardinalRebaseUnsupportedModelV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.UNSUPPORTED_MODEL,
            error.finding_code,
        )
    except camera_cardinal.CameraCardinalRebaseNumericGapV2 as error:
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            error.finding_code,
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            checked_problem,
            checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            "NUMERIC_GAP:CAMERA_CARDINAL_NORMALIZATION",
        )

    quarter_turns = camera_cardinal.original_to_internal_quarter_turns_ccw
    return PreparedSolveStageV2(
        stage_key=SolveStageKeyV2.CAMERA_CARDINAL_REBASE,
        checked_problem=checked_problem,
        checked_config=checked_config,
        normalized_problem=camera_cardinal.normalized_problem,
        internal_config=camera_cardinal.internal_config,
        preprocessing_domain_operations=(
            camera_cardinal.preprocessing_domain_operations
        ),
        child_selection_frame=_ExactCardinalSelectionFrameV2(
            normalized_to_semantic_quarter_turns_ccw=(-quarter_turns) % 4
        ),
        child_numeric_finding="NUMERIC_GAP:V2_3_SOLVE_REPLAY",
        child_missing_finding="NORMALIZED_REPLAY_HAS_NO_RESULT",
        rebind_numeric_finding="NUMERIC_GAP:CAMERA_CARDINAL_RESULT_PULLBACK",
        original_to_internal_quarter_turns_ccw=quarter_turns,
    )


def _rebind_camera_cardinal_stage(
    prepared: PreparedSolveStageV2,
    replay: CanonicalMinimumCostSolveOutcomeV2,
) -> CanonicalMinimumCostSolveOutcomeV2:
    from spatialcf.core.v2 import camera_cardinal_result

    if replay.cumulative_generation_usage is None:
        if (
            type(replay.result) is not UncertifiedResultV2
            or replay.result.candidate_domain is not None
            or replay.result.relation_cost_partition is not None
            or replay.result.objective_partition is not None
        ):
            raise ValueError("normalized replay omitted usage for a published artifact")
        return _valid_uncertified(
            prepared.checked_problem,
            prepared.checked_config,
            replay.result.uncertified_reason,
            *(replay.finding_codes or ("NORMALIZED_REPLAY_UNCERTIFIED",)),
        )

    postprocessing_budget = camera_cardinal_result.CameraCardinalResultDomainBudgetV2(
        limit=prepared.checked_config.max_domain_operations,
        used=(
            prepared.preprocessing_domain_operations
            + replay.cumulative_generation_usage.domain_operations
        ),
    )
    quarter_turns = prepared.original_to_internal_quarter_turns_ccw
    if type(quarter_turns) is not int:
        raise RuntimeError("camera-cardinal stage omitted exact quarter turns")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            return camera_cardinal_result.pullback_camera_cardinal_outcome_v2_4(
                replay,
                prepared.checked_problem,
                prepared.checked_config,
                preprocessing_domain_operations=(
                    prepared.preprocessing_domain_operations
                ),
                original_to_internal_quarter_turns_ccw=quarter_turns,
                postprocessing_budget=postprocessing_budget,
            )
    except camera_cardinal_result.CameraCardinalResultResourceLimitV2:
        return _valid_uncertified(
            prepared.checked_problem,
            prepared.checked_config,
            UncertifiedReasonV2.BOUNDED_SEARCH_EXHAUSTED,
            "RESOURCE_LIMIT:CAMERA_CARDINAL_POSTPROCESSING",
        )
    except (ArithmeticError, RuntimeWarning):
        return _valid_uncertified(
            prepared.checked_problem,
            prepared.checked_config,
            UncertifiedReasonV2.NUMERIC_GAP,
            prepared.rebind_numeric_finding,
        )


def _require_prepared_stage(value: PreparedSolveStageV2) -> None:
    if type(value) is not PreparedSolveStageV2:
        raise TypeError("prepared stage has the wrong exact type")


def _missing_result(finding_code: str) -> CanonicalMinimumCostSolveOutcomeV2:
    return CanonicalMinimumCostSolveOutcomeV2(
        result=None,
        finding_codes=(finding_code,),
    )


def _zero_reservation_input_is_shallow_valid(problem: object) -> bool:
    scene = getattr(problem, "scene", None)
    if type(scene) is not CanonicalSceneV2_1:
        return False
    cameras = getattr(scene, "cameras", None)
    if not isinstance(cameras, FactSetV2):
        return False
    return all(
        values is None or type(values) is tuple
        for values in (
            getattr(cameras, "values", None),
            getattr(cameras, "inner_values", None),
            getattr(cameras, "outer_values", None),
        )
    )


def _camera_reservation_input_is_shallow_valid(problem: object) -> bool:
    scene = getattr(problem, "scene", None)
    if type(scene) is not CanonicalSceneV2_1:
        return False
    for family_name in (
        "objects",
        "geometry_instances",
        "collision_bodies",
        "workspace_boundaries",
        "known_free_spaces",
        "support_surfaces",
        "cameras",
        "baseline_observations",
    ):
        facts = getattr(scene, family_name, None)
        if not isinstance(facts, FactSetV2):
            return False
        if not all(
            values is None or type(values) is tuple
            for values in (
                getattr(facts, "values", None),
                getattr(facts, "inner_values", None),
                getattr(facts, "outer_values", None),
            )
        ):
            return False
    return True


__all__ = (
    "PreparedSolveStageV2",
    "child_missing_outcome_v2",
    "child_numeric_outcome_v2",
    "prepare_solve_stage_v2",
    "rebind_solve_stage_v2",
)
