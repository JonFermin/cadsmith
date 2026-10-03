"""Linear-motion stock: ground rods, T8 lead screw and nut, 20-series aluminium extrusions.

All are prismatic along +Z from z=0 (the T8 nut: flange face at z=0, body toward −Z). Frames
`"bottom"` / `"top"` on the axis at the two ends (nut: `"center"` and `"hole_1".."hole_4"`).
Simplified: no threads, extrusions have a single T-slot per slot position and a solid core.
"""
from __future__ import annotations

from build123d import Location

from ._build import BRASS, STEEL, assemble, disk, mass_g, prism, tube, z_frame
from .libpart import LibPart

T8_LEAD = 8.0          # mm per revolution (Tr8x8: 2 mm pitch, 4 starts)
T8_DIAMETER = 8.0

# T8 brass flange nut: flange dia, flange thickness, body dia, overall length, holes, hole PCD.
_T8_NUT = dict(flange_d=22.0, flange_t=3.5, body_d=10.2, length=15.0, hole_d=3.5, pcd=16.0)

# 20-series T-slot (slot 6): opening 6.2 wide through a 1.8 mm lip, cavity 11 wide under the lip
# tapering to the core 6.1 mm below the surface; ø4.2 center bore per 20 mm cell.
_SLOT_OPEN, _SLOT_CAVITY, _LIP, _SLOT_DEPTH, _CORE_BORE = 6.2, 11.0, 1.8, 6.1, 4.2
# Simplified profiles keep the core solid; real profiles are hollower. Scale to catalog linear
# masses (g/mm): 2020 ~ 0.49 kg/m, 2040 ~ 0.87 kg/m.
_EXTRUSION_MASS = {"2020": 0.49, "2040": 0.87}


def _axis_frames(length: float) -> dict[str, Location]:
    return {"bottom": z_frame(), "top": z_frame(z=length)}


def rod(d: float, length: float) -> LibPart:
    """Ground steel rod (linear shaft), ø`d`, along +Z from z=0."""
    shape = assemble(disk(d, length))
    return LibPart(shape, f"Linear shaft {d:g} mm x {length:g} mm, steel", mass_g(shape, STEEL),
                   _axis_frames(length))


def t8_leadscrew(length: float) -> LibPart:
    """T8 lead screw (Tr8x8, lead 8 mm), modeled as a ø8 cylinder along +Z from z=0.

    Pair with `Assembly.screw(rot_joint, slide_joint, lead=T8_LEAD)`. The nut bore is also ø8, so
    declare `allow_contact(screw, nut)`.
    """
    shape = assemble(disk(T8_DIAMETER, length))
    return LibPart(shape, f"T8 lead screw, lead {T8_LEAD:g} mm, L={length:g} mm",
                   mass_g(shape, STEEL), _axis_frames(length))


def t8_nut() -> LibPart:
    """Brass T8 flange nut: ø22 x 3.5 flange (z=0..3.5), ø10.2 body to z=−11.5, 4 x ø3.5 holes.

    The mounting holes sit on a 16 mm PCD at 0/90/180/270°. The bore is ø8 like the screw, so
    declare `allow_contact(screw, nut)`.
    """
    n = _T8_NUT
    r = n["pcd"] / 2
    holes = [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]
    flange_bores = [(T8_DIAMETER, 0.0, 0.0), *((n["hole_d"], x, y) for x, y in holes)]
    shape = assemble(
        disk(n["flange_d"], n["flange_t"], bores=flange_bores),
        tube(n["body_d"], T8_DIAMETER, n["length"] - n["flange_t"], n["flange_t"] - n["length"]),
    )
    frames = {"center": z_frame()}
    frames.update({f"hole_{i}": z_frame(x, y) for i, (x, y) in enumerate(holes, start=1)})
    return LibPart(shape, "T8 brass flange nut, lead 8 mm", mass_g(shape, BRASS), frames)


def _tslot_outline(width: float, height: float, slots_x: list[float],
                   slots_y: list[float]) -> list[tuple[float, float]]:
    """CCW outline of a width x height profile with T-slots at x=slots_x on the ±Y faces and
    y=slots_y on the ±X faces."""
    hw, hh = width / 2, height / 2
    o, c = _SLOT_OPEN / 2, _SLOT_CAVITY / 2
    # One slot in face coordinates (u along the travel direction, v depth into the material).
    slot = [(-o, 0), (-o, _LIP), (-c, _LIP), (-o, _SLOT_DEPTH), (o, _SLOT_DEPTH), (c, _LIP),
            (o, _LIP), (o, 0)]
    # Faces in CCW order: (start corner, travel direction, inward normal, slot centers along u).
    faces = [
        ((hw, -hh), (0, 1), (-1, 0), sorted(y + hh for y in slots_y)),
        ((hw, hh), (-1, 0), (0, -1), sorted(hw - x for x in slots_x)),
        ((-hw, hh), (0, -1), (1, 0), sorted(hh - y for y in slots_y)),
        ((-hw, -hh), (1, 0), (0, 1), sorted(x + hw for x in slots_x)),
    ]
    pts: list[tuple[float, float]] = []
    for (sx, sy), (tx, ty), (nx, ny), centers in faces:
        pts.append((sx, sy))
        for uc in centers:
            pts += [(sx + (uc + u) * tx + v * nx, sy + (uc + u) * ty + v * ny) for u, v in slot]
    return pts


def _extrusion(name: str, width: float, slots_x: list[float], slots_y: list[float],
               cores: list[float], length: float) -> LibPart:
    outline = _tslot_outline(width, 20.0, slots_x, slots_y)
    shape = assemble(prism(outline, length, bores=[(_CORE_BORE, x, 0.0) for x in cores]))
    mass = _EXTRUSION_MASS[name] * length
    bom = f"{name} aluminium extrusion, L={length:g} mm"
    return LibPart(shape, bom, mass, _axis_frames(length))


def extrusion_2020(length: float) -> LibPart:
    """20x20 T-slot extrusion (slot 6), centered on the Z axis, from z=0 to `length`."""
    return _extrusion("2020", 20.0, [0.0], [0.0], [0.0], length)


def extrusion_2040(length: float) -> LibPart:
    """20x40 T-slot extrusion (slot 6), 40 mm along X, centered on the Z axis, z=0..`length`."""
    return _extrusion("2040", 40.0, [-10.0, 10.0], [0.0], [-10.0, 10.0], length)
