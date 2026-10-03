"""Material densities (spec §4.2). Densities are g/cm³ (= kg/mm³ × 1e6)."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass

__all__ = ["Material", "MATERIALS", "get_material"]


@dataclass(frozen=True)
class Material:
    name: str
    density: float  # g/cm³


MATERIALS: dict[str, Material] = {
    m.name: m
    for m in (
        Material("PLA", 1.24),
        Material("PETG", 1.27),
        Material("ABS", 1.04),
        Material("ASA", 1.07),
        Material("TPU", 1.21),
        Material("nylon", 1.01),
        Material("PA12", 1.01),
        Material("resin", 1.15),
        Material("POM", 1.41),
        Material("acrylic", 1.18),
        Material("aluminum_6061", 2.70),
        Material("steel", 7.85),
        Material("stainless_304", 8.00),
        Material("brass", 8.50),
        Material("copper", 8.96),
        Material("plywood", 0.68),
        Material("MDF", 0.75),
        Material("carbon_fiber", 1.60),
        Material("rubber", 1.10),
    )
}


def _key(name: str) -> str:
    """Lookup key: case-insensitive, spaces and dashes equivalent to underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# Common spellings that don't normalize onto a canonical name by themselves.
_ALIASES = {"aluminum": "aluminum_6061", "aluminium": "aluminum_6061", "stainless": "stainless_304"}
_LOOKUP: dict[str, Material] = {_key(k): m for k, m in MATERIALS.items()}
_LOOKUP.update({alias: MATERIALS[canonical] for alias, canonical in _ALIASES.items()})


def get_material(name_or_obj: str | Material, density: float | None = None) -> Material:
    """Resolve a material by name (case-insensitive) or pass a ``Material`` through.

    ``density`` (g/cm³) overrides the table value; with an explicit density an unknown name is
    accepted as a custom material. Unknown names without a density raise ``KeyError`` whose
    message suggests the closest match and lists the options.
    """
    if density is not None:
        density = float(density)
        if not (math.isfinite(density) and density > 0):
            raise ValueError(f"density must be a positive number in g/cm³ (got {density})")
    if isinstance(name_or_obj, Material):
        return name_or_obj if density is None else Material(name_or_obj.name, density)
    name = str(name_or_obj)
    base = _LOOKUP.get(_key(name))
    if base is None:
        if density is not None:
            return Material(name, density)
        close = difflib.get_close_matches(_key(name), list(_LOOKUP), n=1, cutoff=0.6)
        hint = f" (did you mean '{_LOOKUP[close[0]].name}'?)" if close else ""
        raise KeyError(
            f"unknown material '{name}'{hint}; options: {', '.join(MATERIALS)} — or pass density= in g/cm³"
        )
    return base if density is None else Material(base.name, density)
