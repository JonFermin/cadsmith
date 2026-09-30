"""Motors with standard mounting geometry (simplified bodies, no connectors or D-flats).

Convention: mounting face at z=0, body toward −Z, output shaft along +Z on the origin. Frames
`"shaft"` (origin, +Z) and `"hole_1"`.. (mounting-hole centers on the face, +Z; four for steppers,
two for servos and the N20). For servos the mounting face is the underside of the ears, and
`"horn"` is the spline tip (where a horn seats: z=14.0 for the SG90, 16.4 for the MG996R).
"""
from __future__ import annotations

import math

from build123d import Location

from ._build import assemble, disk, prism, tube, z_frame
from .fits import tap_drill
from .libpart import LibPart

# Effective density of a stepper over its square body envelope; reproduces typical catalog masses
# (NEMA 17 x 40 mm ~ 280 g, NEMA 23 x 56 mm ~ 700 g).
_STEPPER_DENSITY = 3.9          # g/cm^3
_THREAD_DEPTH = 4.5             # mm, blind mounting holes in the stepper face


def _rect(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2)]


def _chamfered_square(a: float, c: float) -> list[tuple[float, float]]:
    """Square of side `a` centered on the origin with 45° corner chamfers of leg `c`."""
    h = a / 2
    return [(h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c),
            (-h + c, -h), (h - c, -h)]


def _hole_frames(points: list[tuple[float, float]]) -> dict[str, Location]:
    return {f"hole_{i}": z_frame(x, y) for i, (x, y) in enumerate(points, start=1)}


def _stepper(nema: int, face: float, spacing: float, hole_d: float, chamfer: float, pilot_d: float,
             pilot_h: float, shaft_d: float, shaft_len: float, length: float) -> LibPart:
    if length <= _THREAD_DEPTH:
        raise ValueError(f"nema{nema}: length must exceed {_THREAD_DEPTH} mm")
    s = spacing / 2
    holes = [(s, s), (-s, s), (-s, -s), (s, -s)]
    outline = _chamfered_square(face, chamfer)
    shape = assemble(
        prism(outline, length - _THREAD_DEPTH, -length),
        prism(outline, _THREAD_DEPTH, -_THREAD_DEPTH, bores=[(hole_d, x, y) for x, y in holes]),
        tube(pilot_d, shaft_d, pilot_h),
        disk(shaft_d, shaft_len),
    )
    mass = _STEPPER_DENSITY * 1e-3 * face * face * length
    frames = {"shaft": z_frame(), **_hole_frames(holes)}
    bom = f"NEMA {nema} stepper motor {face:g}x{face:g}x{length:g} mm"
    return LibPart(shape, bom, mass, frames)


def nema17(length: float = 40) -> LibPart:
    """NEMA 17 stepper: 42.3 mm face, M3 holes on 31 mm square, ø22 x 2 pilot, ø5 x 24 shaft."""
    return _stepper(17, 42.3, 31.0, tap_drill("M3"), 4.0, 22.0, 2.0, 5.0, 24.0, length)


def nema23(length: float = 56) -> LibPart:
    """NEMA 23 stepper: 56.4 face, ø5 holes on 47.14 square, ø38.1 x 1.6 pilot, ø6.35 x 21 shaft."""
    return _stepper(23, 56.4, 47.14, 5.0, 5.0, 38.1, 1.6, 6.35, 21.0, length)


def _servo(name: str, body_l: float, body_w: float, below: float, above: float, ear_span: float,
           ear_t: float, hole_d: float, hole_pitch: float, hole_rows: tuple[float, ...],
           boss_d: float, boss_h: float, spline_d: float, spline_h: float, mass: float) -> LibPart:
    # The output shaft sits on the origin, body_w/2 from the output (+X) end of the body.
    cx = -(body_l / 2 - body_w / 2)
    ear_l = (ear_span - body_l) / 2
    holes = [(cx + sx * hole_pitch / 2, y) for sx in (1, -1) for y in hole_rows]
    ears = [
        prism(_rect(cx + sx * (body_l + ear_l) / 2, 0, ear_l, body_w), ear_t,
              bores=[(hole_d, x, y) for x, y in holes if (x - cx) * sx > 0])
        for sx in (1, -1)
    ]
    shape = assemble(
        prism(_rect(cx, 0, body_l, body_w), below + above, -below),
        *ears,
        disk(boss_d, boss_h, above),
        disk(spline_d, spline_h, above + boss_h),
    )
    frames = {"shaft": z_frame(), "horn": z_frame(0, 0, above + boss_h + spline_h), **_hole_frames(holes)}
    return LibPart(shape, f"{name} servo", mass, frames)


def sg90() -> LibPart:
    """TowerPro SG90 micro servo: body 22.2 x 11.8, ears 32.2 long, ø2 holes 27.8 apart, 9 g."""
    return _servo("SG90 micro", 22.2, 11.8, 15.9, 6.8, 32.2, 2.5, 2.0, 27.8, (0.0,),
                  11.8, 4.0, 4.8, 3.2, 9.0)


def mg996r() -> LibPart:
    """TowerPro MG996R servo: body 40.7 x 19.7, ears 54 long, 4 x ø4.2 holes 49.5 x 10, 55 g."""
    return _servo("MG996R", 40.7, 19.7, 26.5, 10.0, 54.0, 2.5, 4.2, 49.5, (5.0, -5.0),
                  13.0, 3.0, 5.8, 3.4, 55.0)


def n20_gearmotor() -> LibPart:
    """N20 micro gearmotor: 12 x 10 x 9 gearbox, ø12 (10 across flats) x 15 can, ø3 x 10 shaft.

    Two M1.6 holes 9 mm apart on the gearbox face; ~10 g.
    """
    half_flat = math.asin(5 / 6)
    arc = (-half_flat + 2 * half_flat * i / 7 for i in range(8))
    can = [(6 * math.cos(a), 6 * math.sin(a)) for a in arc]
    can += [(-x, -y) for x, y in can]
    holes = [(4.5, 0.0), (-4.5, 0.0)]
    shape = assemble(
        prism(can, 15, -24),
        prism(_rect(0, 0, 12, 10), 7, -9),
        prism(_rect(0, 0, 12, 10), 2, -2, bores=[(1.25, x, y) for x, y in holes]),
        disk(3, 10),
    )
    frames = {"shaft": z_frame(), **_hole_frames(holes)}
    return LibPart(shape, "N20 micro metal gearmotor 12x10 mm", 10.0, frames)
