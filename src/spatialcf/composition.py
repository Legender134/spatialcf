"""The sole concrete default-composition leaf for environment adapters."""

from spatialcf.adapters.ai2thor import AI2ThorAdapter

DEFAULT_ENVIRONMENT_ADAPTER_FACTORY = AI2ThorAdapter

__all__ = ("DEFAULT_ENVIRONMENT_ADAPTER_FACTORY",)
