"""Involute spur gears, gear pairs, racks and GT2 pulleys (own generator — no bd_warehouse).

Teeth follow the ISO 53 basic rack (addendum m, dedendum 1.25 m). Each flank is the involute from
the form circle to the tip; below it the flank follows whichever cuts deeper of the involute
(continued radially under the base circle) and the trochoid swept by the generating rack's tip
corner — the latter is the real undercut of small gears. The tooth is therefore never fatter than a
rack-generated tooth, so standard gears mesh without overlap. Each flank is a short polyline
(7 involute samples + up to 3 root samples); chords sit inside the convex involute, which only
adds a hair of backlash.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from build123d import Location, Rot

from ._build import assemble, prism, tube, z_frame
from .libpart import LibPart

_ADDENDUM = 1.0          # x module (ISO 53 profile A)
_DEDENDUM = 1.25         # x module (addendum + 0.25 m tip clearance)
_INVOLUTE_SAMPLES = 7    # per flank, form circle -> tip circle
_ROOT_SAMPLES = 3        # per flank, root circle -> form circle (fillet / undercut / radial)

GT2_PITCH = 2.0          # mm
GT2_PLD = 0.254          # pitch line differential per side: OD = PD - 2 * PLD
GT2_TOOTH_DEPTH = 0.75   # groove depth below the OD
GT2_GROOVE_R = 0.555     # groove radius (belt tooth ~ semicircular)
GT2_FLANGE_T = 1.0       # flange thickness
GT2_FLANGE_RISE = 1.5    # flange radius above the OD


def _inv(a: np.ndarray | float) -> np.ndarray | float:
    """Involute function inv(a) = tan(a) - a."""
    return np.tan(a) - a


class _ToothProfile:
    """Half-angle θ(ρ) of one tooth of a rack-generated involute gear, tooth centered on angle 0."""

    def __init__(self, module: float, teeth: int, backlash: float, pressure_angle: float) -> None:
        m, z = module, teeth
        self.alpha = math.radians(pressure_angle)
        self.r = m * z / 2
        self.rb = self.r * math.cos(self.alpha)
        self.ra = self.r + _ADDENDUM * m
        self.rf = self.r - _DEDENDUM * m
        # Each gear is thinned by half the pair backlash at the pitch circle.
        s = math.pi * m / 2 - backlash / 2
        self.psi = s / (2 * self.r)                        # half tooth angle at the pitch circle
        # Generating-rack tooth that cuts the space next to this tooth: its thickness at the pitch
        # line equals the gear's space width; y0 = pitch-line coordinate of its tip corner at φ = 0.
        h_tip = (math.pi * m - s) / 2 - _DEDENDUM * m * math.tan(self.alpha)
        self.y0 = math.pi * m / 2 - h_tip

    def involute(self, rho: np.ndarray) -> np.ndarray:
        """Involute half-angle, continued radially below the base circle."""
        a_rho = np.arccos(self.rb / np.maximum(rho, self.rb))
        return self.psi + _inv(self.alpha) - _inv(a_rho)

    def trochoid(self, rho: np.ndarray) -> np.ndarray:
        """Angular position of the cutter tip corner's path at radius ρ (side nearer the tooth).

        Rolling the gear by φ translates the rack by r·φ; in gear coordinates the corner sits at
        polar angle atan2(Y, rf) − (Y − y0)/r with Y = ±sqrt(ρ² − rf²); −Y is nearer the tooth.
        """
        y = np.sqrt(np.maximum(rho * rho - self.rf * self.rf, 0.0))
        return self.y0 / self.r - (np.arctan2(y, self.rf) - y / self.r)

    def half_angle(self, rho: np.ndarray) -> np.ndarray:
        return np.minimum(self.involute(rho), self.trochoid(rho))

    def form_radius(self) -> float:
        """Radius where the flank stops being involute (top of the undercut, or the base circle)."""
        grid = np.linspace(self.rf, self.ra, 400)
        cut = grid[self.trochoid(grid) < self.involute(grid)]
        if cut.size == 0:
            return max(self.rb, self.rf)
        # refine the upper crossing by bisection
        lo, hi = float(cut.max()), min(float(cut.max()) + (grid[1] - grid[0]), self.ra)
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.trochoid(mid) < self.involute(mid) else (lo, mid)
        return max(lo, self.rb)

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        """(ρ, θ) from root circle to tip circle along one flank."""
        r_form = self.form_radius()
        if r_form - self.rf > 1e-6 * self.r:
            root = np.linspace(self.rf, r_form, _ROOT_SAMPLES, endpoint=False)
        else:  # large gears: root circle above the base circle, the involute starts at the root
            root = np.empty(0)
        # Uniform in roll parameter t = sqrt((ρ/rb)² − 1): denser near the base circle where the
        # involute curves most.
        t0 = math.sqrt(max((r_form / self.rb) ** 2 - 1, 0.0))
        t1 = math.sqrt((self.ra / self.rb) ** 2 - 1)
        flank = self.rb * np.sqrt(1 + np.linspace(t0, t1, _INVOLUTE_SAMPLES) ** 2)
        rho = np.concatenate([root, flank])
        return rho, self.half_angle(rho)


def _gear_outline(module: float, teeth: int, backlash: float,
                  pressure_angle: float) -> list[tuple[float, float]]:
    prof = _ToothProfile(module, teeth, backlash, pressure_angle)
    rho, theta = prof.samples()
    if theta[-1] <= 0:
        raise ValueError(f"spur_gear: {teeth} teeth at {pressure_angle}° gives pointed teeth")
    if theta.min() <= 0:
        raise ValueError(f"spur_gear: {teeth} teeth is too few (undercut consumes the tooth root)")
    pitch = 2 * math.pi / teeth
    # One tooth: up the −θ flank, across the tip, down the +θ flank, then the root-arc midpoint.
    ang = np.concatenate([-theta, theta[::-1], [pitch / 2]])
    rad = np.concatenate([rho, rho[::-1], [prof.rf]])
    pts = []
    for k in range(teeth):
        a = ang + k * pitch
        pts.extend(zip(rad * np.cos(a), rad * np.sin(a)))
    return [(float(x), float(y)) for x, y in pts]


def spur_gear(module: float, teeth: int, width: float, *, bore: float = 0, backlash: float = 0.05,
              pressure_angle: float = 20) -> LibPart:
    """Involute spur gear: axis +Z through the origin, bottom face at z=0, a tooth centered on +X.

    `backlash` is the circumferential backlash (mm, at the pitch circle) of a mesh with a mate that
    uses the same value: each gear's teeth are thinned by backlash/2. Frame `"axis"` at the origin.
    """
    if teeth < 4:
        raise ValueError("spur_gear: need at least 4 teeth")
    outline = _gear_outline(module, teeth, backlash, pressure_angle)
    shape = assemble(prism(outline, width, bores=[(bore, 0.0, 0.0)] if bore > 0 else []))
    bom = f"Spur gear m{module:g} z{teeth} x {width:g} mm, {pressure_angle:g} deg PA"
    if bore > 0:
        bom += f", bore {bore:g}"
    return LibPart(shape, bom, None, {"axis": z_frame()})


class GearPair(NamedTuple):
    g1: LibPart
    g2: LibPart
    ratio: float              # ω2/ω1 for both axes +Z (external mesh: −z1/z2)
    center_distance: float    # mm


def gear_pair(module: float, z1: int, z2: int, width: float, *, backlash: float = 0.05,
              bore1: float = 0, bore2: float = 0) -> GearPair:
    """Meshing external pair at the home pose.

    g1 is `spur_gear(...)` at the origin (tooth on +X); g2 sits at (m(z1+z2)/2, 0, 0) turned by
    180° + 180°/z2 so a tooth *space* faces −X and receives g1's tooth. Both axes +Z; rotating g1
    by θ and g2 by −θ·z1/z2 (i.e. `ratio`·θ) keeps them meshed without overlap.
    """
    a = module * (z1 + z2) / 2
    g1 = spur_gear(module, z1, width, bore=bore1, backlash=backlash)
    g2 = spur_gear(module, z2, width, bore=bore2, backlash=backlash)
    g2 = g2.moved(Location((a, 0, 0)) * Rot(0, 0, 180 + 180 / z2))
    return GearPair(g1, g2, -z1 / z2, a)


def rack(module: float, length: float, width: float, height: float, *, backlash: float = 0.05,
         pressure_angle: float = 20) -> LibPart:
    """Straight rack (ISO 53 profile) along X, centered on x=0, teeth pointing +Y.

    The pitch line is y=0 and the back face is y = −height (height > 1.25·module); z from 0 to
    `width`. A tooth *space* is centered on x=0, so `Pos(0, r, 0) * Rot(0, 0, -90) * spur_gear(...)`
    (r = pitch radius, a tooth pointing −Y) meshes at home; the gear turning +θ° moves a meshing
    rack by +π·r·θ/180 along X. Teeth are thinned by backlash/2 like `spur_gear`. Frame `"pitch"`
    at the origin.
    """
    m = module
    if height <= _DEDENDUM * m:
        raise ValueError(f"rack: height must exceed the dedendum {_DEDENDUM * m:g} mm")
    p = math.pi * m
    tan_a = math.tan(math.radians(pressure_angle))
    h_pitch = (p / 2 - backlash / 2) / 2                  # tooth half-thickness at the pitch line
    h_root, h_tip = h_pitch + _DEDENDUM * m * tan_a, h_pitch - _ADDENDUM * m * tan_a
    y_root, y_tip = -_DEDENDUM * m, _ADDENDUM * m
    # Periodic profile y(x): teeth centered at x = (k + 1/2)·p.
    corners: list[tuple[float, float]] = []
    for k in range(math.floor(-length / 2 / p) - 1, math.ceil(length / 2 / p) + 1):
        c = (k + 0.5) * p
        corners += [(c - h_root, y_root), (c - h_tip, y_tip),
                    (c + h_tip, y_tip), (c + h_root, y_root)]
    xs, ys = np.array(corners).T
    x0, x1 = -length / 2, length / 2
    inside = [(float(x), float(y)) for x, y in corners if x0 < x < x1]
    top = [(x0, float(np.interp(x0, xs, ys)))] + inside + [(x1, float(np.interp(x1, xs, ys)))]
    outline = [(x0, -height)] + top + [(x1, -height)]
    shape = assemble(prism(outline, width))
    bom = f"Rack m{module:g} {length:g} x {width:g} x {height + _ADDENDUM * m:g} mm"
    return LibPart(shape, bom, None, {"pitch": z_frame()})


def _gt2_outline(teeth: int) -> list[tuple[float, float]]:
    """GT2 pulley outline: OD circle with one semicircular-ish groove per tooth, groove 0 on +X."""
    r_o = (GT2_PITCH * teeth / math.pi) / 2 - GT2_PLD
    d_c = r_o - GT2_TOOTH_DEPTH + GT2_GROOVE_R            # groove center radius
    # Groove circle meets the OD at ±beta about the groove center direction.
    beta = math.acos((r_o**2 + d_c**2 - GT2_GROOVE_R**2) / (2 * r_o * d_c))
    # Seen from the groove center, the OD intersection sits at angle ±gamma from the outward ray.
    gamma = math.atan2(r_o * math.sin(beta), r_o * math.cos(beta) - d_c)
    step = 2 * math.pi / teeth
    pts: list[tuple[float, float]] = []
    for k in range(teeth):
        phi = k * step
        # groove (CCW around the pulley = clockwise around its center): from the OD point at
        # phi − beta (local angle −gamma) through the innermost point (local π) to phi + beta
        for s in np.linspace(2 * math.pi - gamma, gamma, 7):
            a = phi + s
            pts.append((d_c * math.cos(phi) + GT2_GROOVE_R * math.cos(a),
                        d_c * math.sin(phi) + GT2_GROOVE_R * math.sin(a)))
        # land between grooves: interior points on the OD
        for a in np.linspace(phi + beta, phi + step - beta, 5)[1:-1]:
            pts.append((r_o * math.cos(a), r_o * math.sin(a)))
    return pts


def gt2_pulley(teeth: int, width: float, bore: float) -> LibPart:
    """GT2 (2 mm pitch) timing pulley, pitch diameter 2·teeth/π, axis +Z.

    Simplified: toothed section of `width` between two 1 mm flanges (three touching solids, no
    hub or set screws); bottom flange face at z=0.
    Frames `"axis"` (origin) and `"belt"` (mid-plane of the toothed section).
    """
    if teeth < 10:
        raise ValueError("gt2_pulley: need at least 10 teeth")
    r_o = GT2_PITCH * teeth / math.pi / 2 - GT2_PLD
    flange_d = 2 * (r_o + GT2_FLANGE_RISE)
    bores = [(bore, 0.0, 0.0)] if bore > 0 else []
    body = assemble(
        tube(flange_d, bore, GT2_FLANGE_T),
        prism(_gt2_outline(teeth), width, GT2_FLANGE_T, bores=bores),
        tube(flange_d, bore, GT2_FLANGE_T, GT2_FLANGE_T + width),
    )
    frames = {"axis": z_frame(), "belt": z_frame(z=GT2_FLANGE_T + width / 2)}
    return LibPart(body, f"GT2 pulley {teeth}T {width:g} mm belt, bore {bore:g}", None, frames)
