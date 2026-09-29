"""Rolling bearings by designation: catalog d x D x B and mass, simplified shielded geometry.

Axis Z, centered on the origin (z from −B/2 to +B/2). Frames `"center"` (origin, +Z) and `"top"` /
`"bottom"` (the two side faces, both +Z) for stacking against shoulders.
"""
from __future__ import annotations

import re

from build123d import Location

from ._build import assemble, tube, z_frame
from .libpart import LibPart

# Deep groove ball bearings: bore d, outer D, width B (mm), mass (g, typical catalog).
_BALL = {
    "623": (3, 10, 4, 1.7), "624": (4, 13, 5, 3.1), "625": (5, 16, 5, 4.8), "626": (6, 19, 6, 8.1),
    "608": (8, 22, 7, 12.0), "688": (8, 16, 5, 3.2),
    "6000": (10, 26, 8, 19.0), "6001": (12, 28, 8, 22.0), "6002": (15, 32, 9, 30.0),
    "6003": (17, 35, 10, 39.0), "6004": (20, 42, 12, 69.0), "6005": (25, 47, 12, 80.0),
    "6200": (10, 30, 9, 32.0), "6201": (12, 32, 10, 37.0), "6202": (15, 35, 11, 45.0),
}
# Linear ball bushings: bore dr, outer D, length L, retaining-ring grooves (span B between outer
# groove faces, groove width W, groove diameter D1), mass (g).
_LINEAR = {
    "LM8UU": (8, 15, 24, 17.5, 1.1, 14.3, 16.0),
    "LM10UU": (10, 19, 29, 22.0, 1.3, 18.0, 30.0),
}

_SUFFIX = re.compile(r"(-?(2RS|RS|2Z|ZZ|Z|2RSH|RSH))?")
_RING = 0.3        # ring radial thickness as a fraction of the section height (D − d)/2
_SHIELD_RECESS = 0.3   # mm the shields sit below the ring faces


def _ball_bearing(name: str, d: float, D: float, B: float, mass: float) -> LibPart:
    ring = _RING * (D - d) / 2
    d_inner_land, d_outer_land = d + 2 * ring, D - 2 * ring
    shape = assemble(
        tube(d_inner_land, d, B, -B / 2),
        tube(d_outer_land, d_inner_land, B - 2 * _SHIELD_RECESS, -B / 2 + _SHIELD_RECESS),
        tube(D, d_outer_land, B, -B / 2),
    )
    bom = f"{name} deep groove ball bearing {d:g}x{D:g}x{B:g}"
    return LibPart(shape, bom, mass, _frames(B))


def _linear_bearing(name: str, d: float, D: float, L: float, span: float, groove_w: float,
                    groove_d: float, mass: float) -> LibPart:
    # Stack of coaxial tubes along Z: body | groove | body | groove | body.
    z_groove = span / 2 - groove_w          # inner face of each groove
    breaks = [-L / 2, -span / 2, -z_groove, z_groove, span / 2, L / 2]
    diameters = [D, groove_d, D, groove_d, D]
    segments = zip(diameters, breaks, breaks[1:])
    shape = assemble(*(tube(od, d, z1 - z0, z0) for od, z0, z1 in segments))
    return LibPart(shape, f"{name} linear ball bearing {d:g}x{D:g}x{L:g}", mass, _frames(L))


def _frames(width: float) -> dict[str, Location]:
    return {"center": z_frame(), "top": z_frame(z=width / 2), "bottom": z_frame(z=-width / 2)}


def bearing(designation: str) -> LibPart:
    """Bearing by designation: 623 624 625 626 608 688 6000–6005 6200–6202, LM8UU, LM10UU.

    Shield/seal suffixes are accepted and ignored ("608-2RS", "625ZZ").
    """
    text = designation.strip().upper()
    if text in _LINEAR:
        return _linear_bearing(text, *_LINEAR[text])
    # Longest known designation that prefixes the text, leaving only a shield/seal suffix.
    for key in sorted(_BALL, key=len, reverse=True):
        if text.startswith(key) and _SUFFIX.fullmatch(text[len(key):]):
            return _ball_bearing(key, *_BALL[key])
    raise KeyError(f"unknown bearing {designation!r}; known: {', '.join([*_BALL, *_LINEAR])}")
