"""Platform-neutral SpatialCF core algorithms."""

__all__ = ("solve_minimum_cost", "verify_solve_result")


def __getattr__(name: str):
    if name == "solve_minimum_cost":
        from spatialcf.core.solver import solve_minimum_cost

        return solve_minimum_cost
    if name == "verify_solve_result":
        from spatialcf.core.verification import verify_solve_result

        return verify_solve_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
