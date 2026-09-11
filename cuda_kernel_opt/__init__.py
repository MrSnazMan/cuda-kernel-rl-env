"""CUDA kernel optimisation environment.

``load_environment`` is imported lazily so that lightweight consumers (task
generation, the TUI's static parts, unit tests) don't drag in verifiers.
"""

from __future__ import annotations

__all__ = ["load_environment", "CudaKernelOptEnv"]


def __getattr__(name: str):
    if name in ("load_environment", "CudaKernelOptEnv"):
        from . import environment

        return getattr(environment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
