"""Current source-planning authority and operations. Lazy exports avoid loading unrelated execution owners."""

from importlib import import_module

__all__ = (
    "CollisionDelegation",
    "EndpointPlan",
    "EndpointPlanRejected",
    "EndpointWorkspace",
    "ProxyBinding",
    "ProxyBundle",
    "RetainedSourcePlan",
    "RetainedSourcePlanVerification",
    "SourcePlan",
    "SourcePolicy",
    "SubjectPlacementFact",
    "build_default_source_policy",
    "build_proxy_bundle",
    "default_planning_workspace",
    "default_solver_config",
    "load_source_plan",
    "load_source_plan_retained",
    "load_source_policy",
    "plan_endpoint",
    "plan_source_campaign",
    "prepare_source_plan_verification",
    "publish_source_plan",
    "revalidate_source_plan_verification",
)

_EXPORTS = {
    "RetainedSourcePlan": (
        "spatialcf.generation.planning.campaign",
        "RetainedSourcePlan",
    ),
    "RetainedSourcePlanVerification": (
        "spatialcf.generation.planning.campaign",
        "RetainedSourcePlanVerification",
    ),
    "SourcePlan": ("spatialcf.generation.planning.campaign", "SourcePlan"),
    "SourcePolicy": ("spatialcf.generation.planning.campaign", "SourcePolicy"),
    "build_default_source_policy": (
        "spatialcf.generation.planning.campaign",
        "build_default_source_policy",
    ),
    "load_source_plan": ("spatialcf.generation.planning.campaign", "load_source_plan"),
    "load_source_plan_retained": (
        "spatialcf.generation.planning.campaign",
        "load_source_plan_retained",
    ),
    "load_source_policy": (
        "spatialcf.generation.planning.campaign",
        "load_source_policy",
    ),
    "plan_source_campaign": (
        "spatialcf.generation.planning.campaign",
        "plan_source_campaign",
    ),
    "prepare_source_plan_verification": (
        "spatialcf.generation.planning.campaign",
        "prepare_source_plan_verification",
    ),
    "publish_source_plan": (
        "spatialcf.generation.planning.campaign",
        "publish_source_plan",
    ),
    "revalidate_source_plan_verification": (
        "spatialcf.generation.planning.campaign",
        "revalidate_source_plan_verification",
    ),
    "EndpointPlanRejected": (
        "spatialcf.generation.planning.endpoint",
        "EndpointPlanRejected",
    ),
    "plan_endpoint": ("spatialcf.generation.planning.endpoint", "plan_endpoint"),
    "CollisionDelegation": (
        "spatialcf.generation.planning.models",
        "CollisionDelegation",
    ),
    "EndpointPlan": ("spatialcf.generation.planning.models", "EndpointPlan"),
    "EndpointWorkspace": ("spatialcf.generation.planning.models", "EndpointWorkspace"),
    "ProxyBinding": ("spatialcf.generation.planning.models", "ProxyBinding"),
    "ProxyBundle": ("spatialcf.generation.planning.models", "ProxyBundle"),
    "SubjectPlacementFact": (
        "spatialcf.generation.planning.models",
        "SubjectPlacementFact",
    ),
    "build_proxy_bundle": (
        "spatialcf.generation.planning.problem",
        "build_proxy_bundle",
    ),
    "default_planning_workspace": (
        "spatialcf.generation.planning.problem",
        "default_planning_workspace",
    ),
    "default_solver_config": (
        "spatialcf.generation.planning.problem",
        "default_solver_config",
    ),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
