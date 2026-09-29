"""Small, fast solid builders shared by the parts modules (private).

Everything is built directly from topology (wire -> face with holes -> prism) instead of booleans or
the builder API: it is several times faster and keeps each library part well under the 0.3 s budget.
Multi-body parts are returned as a `Part` of non-overlapping solids (touching is fine) rather than
fused, since fusing faceted solids is what would blow the budget.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeVar

from build123d import Edge, Face, Location, Part, Plane, Solid, Wire
from build123d.topology import Shape

# Densities (g/cm^3 == 1e-3 g/mm^3) for bought-in metal parts whose mass the library supplies.
STEEL = 7.85
BRASS = 8.50

T = TypeVar("T")
Bore = tuple[float, float, float]      # (diameter, x, y) of a through hole in a prism


def _circle(d: float, x: float, y: float, z: float) -> Wire:
    return Wire([Edge.make_circle(d / 2, Plane(origin=(x, y, z), z_dir=(0, 0, 1)))])


def prism(outline: Iterable[Sequence[float]], height: float, z0: float = 0.0, *,
          holes: Iterable[Iterable[Sequence[float]]] = (), bores: Iterable[Bore] = ()) -> Solid:
    """Extrude a closed XY polygon from z0 up by `height`, with polygonal `holes`, round `bores`."""
    outer = Wire.make_polygon([(x, y, z0) for x, y in outline], close=True)
    inner = [Wire.make_polygon([(x, y, z0) for x, y in h], close=True) for h in holes]
    inner += [_circle(d, x, y, z0) for d, x, y in bores]
    return Solid.extrude(Face(outer, inner), (0, 0, height))


def disk(d: float, height: float, z0: float = 0.0, *, x: float = 0.0, y: float = 0.0,
         holes: Iterable[Iterable[Sequence[float]]] = (), bores: Iterable[Bore] = ()) -> Solid:
    """Cylinder along +Z, base center (x, y, z0), pierced by polygonal `holes` / round `bores`."""
    inner = [Wire.make_polygon([(hx, hy, z0) for hx, hy in h], close=True) for h in holes]
    inner += [_circle(bd, bx, by, z0) for bd, bx, by in bores]
    return Solid.extrude(Face(_circle(d, x, y, z0), inner), (0, 0, height))


def tube(d_out: float, d_in: float, height: float, z0: float = 0.0) -> Solid:
    """Coaxial hollow cylinder along +Z (d_in == 0 gives a solid cylinder)."""
    return disk(d_out, height, z0, bores=[(d_in, 0.0, 0.0)] if d_in > 0 else [])


def regular_polygon(across_flats: float, sides: int = 6) -> list[tuple[float, float]]:
    """Vertices of a regular polygon centered on the origin, flats parallel to X for hexagons."""
    r = across_flats / 2 / math.cos(math.pi / sides)
    return [(r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]


def metric_size(size: str | float) -> float:
    """'M3' / 'm2.5' / 3 -> nominal diameter in mm."""
    if isinstance(size, str):
        text = size.strip().upper()
        if not text.startswith("M"):
            raise ValueError(f"bad metric size {size!r} (expected e.g. 'M3')")
        size = float(text[1:])
    return float(size)


def lookup_metric(table: dict[float, T], size: str | float, what: str) -> T:
    """Table row for a metric size, with a KeyError listing the sizes available."""
    d = metric_size(size)
    for key, value in table.items():
        if math.isclose(key, d):
            return value
    raise KeyError(f"{what}: no data for M{d:g}; sizes: {', '.join(f'M{k:g}' for k in table)}")


def mass_g(shape: Shape, density_g_cm3: float) -> float:
    """Mass in grams of a solid with uniform density (volume in mm^3)."""
    return shape.volume * density_g_cm3 * 1e-3


def assemble(*shapes: Shape) -> Part:
    """Group the solids of `shapes` into one flat `Part` (no boolean: they must not overlap)."""
    return Part([solid for shape in shapes for solid in shape.solids()])


def z_frame(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Location:
    """Frame at (x, y, z) with world orientation (its Z axis is +Z)."""
    return Location((x, y, z))
