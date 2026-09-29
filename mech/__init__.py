"""cadsmith mechanism framework.

Model scripts declare geometry + joints + intent with ``Assembly``; ``run``/``analyze`` do the
rest (kinematics, clearances, mass properties, loads, targets, export). See docs/MECH_SPEC.md.
"""

from __future__ import annotations

from typing import Any

from .assembly import Assembly, ModelError
from .kinematics import Kinematics
from .massprops import assembly_props, part_props
from .materials import MATERIALS, Material

__all__ = ["Assembly", "ModelError", "Material", "MATERIALS", "run", "analyze", "Kinematics",
           "part_props", "assembly_props"]

_LAZY = ("run", "analyze")  # from .runner, imported on first use (runner imports most of mech)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'mech' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
