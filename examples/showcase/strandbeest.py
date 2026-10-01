"""Theo Jansen "Strandbeest" walker: six Jansen-linkage legs on one three-throw crankshaft.

Jansen's "holy numbers" (a…m) scaled by `scale` mm per unit. Legs are planar in X-Z (hinges ∥ Y):
X = walking direction, Z up, the crankshaft runs along Y through the origin. Station k carries a
mirrored left/right leg pair on one crank pin at phase 360°·k/stations; mirroring maps the crank
angle to 180° − θ, so with 3 stations the six feet are spaced 60° apart in the gait. Each leg is 6
links closed by 3 pins (3 loops): 18 loops are solved together from the single crank angle.

Per station the link layers run (Y, outside → in) upper triangle · {j, f, c} · lower triangle · k,
then mirrored, so the two legs share one crank pin and nothing crosses within a layer. The frame is
laser-cut bulkheads on a 2020 spine with 608 crankshaft bearings, stub pivot axles pressed into the
bulkheads, a NEMA 17 on a coupler at one end and a lightened flywheel at the other.
"""
import math

import numpy as np
from build123d import *
from mech import *
from mech.geom import link
from mech.parts import bearing, extrusion_2020, nema17, socket_head_screw

HOLY = dict(a=38.0, b=41.5, c=39.3, d=40.1, e=55.8, f=39.4, g=36.7, h=65.7, i=49.0, j=50.0, k=61.9, l=7.8, m=15.0)
YAX = (0, -1, 0)                      # hinge axis: +θ turns the crank counter-clockwise seen from −Y
PALETTE = ["#c8102e", "#0f8b8d", "#e39b1b", "#5b4b9a"]


# ---------------------------------------------------------------- Jansen kinematics (drawing + check)
def _circ(c0, r0, c1, r1, side):
    """2-D circle intersection; side=+1 is left of c0→c1 (x right, z up)."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    dv = c1 - c0
    d = math.hypot(*dv)
    x = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    e = dv / d
    return c0 + x * e + side * math.sqrt(max(r0 * r0 - x * x, 0.0)) * np.array([-e[1], e[0]])


def jansen(theta, s=1.0, mirror=1):
    """(x, z) of every joint of one leg at crank angle theta (deg); mirror=−1 flips X (right leg)."""
    H = {k: v * s for k, v in HOLY.items()}
    th = math.radians(180 - theta if mirror < 0 else theta)
    O, P = np.zeros(2), np.array([-H["a"], -H["l"]])
    C = H["m"] * np.array([math.cos(th), math.sin(th)])
    B = _circ(C, H["j"], P, H["b"], -1)            # rod j up to the upper triangle
    D = _circ(P, H["d"], B, H["e"], +1)            # rigid upper triangle P-B-D rocks about P
    E = _circ(C, H["k"], P, H["c"], +1)            # rod k down to the knee E, rocker c from P
    F = _circ(D, H["f"], E, H["g"], -1)            # rod f hangs the lower triangle from D
    G = _circ(E, H["i"], F, H["h"], +1)            # foot: rigid lower triangle E-F-G
    return {n: (mirror * p[0], p[1]) for n, p in dict(O=O, P=P, C=C, B=B, D=D, E=E, F=F, G=G).items()}


# ---------------------------------------------------------------- geometry helpers (all in place, Y = thickness)
def _cyl_y(x, z, y0, y1, d):
    """Cylinder ø d along Y from y0 to y1 at (x, z)."""
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(d / 2, abs(y1 - y0))


def _bar(p0, p1, y, w, t, holes=()):
    """Flat link in an X-Z plane centred on y (thickness t along Y) with ø holes at (x, z) points."""
    shape = link((p0[0], y, p0[1]), (p1[0], y, p1[1]), width=w, thickness=t, z=y - t / 2, normal=(0, 1, 0), hole=0)
    for (x, z), d in holes:
        shape -= _cyl_y(x, z, y - t, y + t, d)
    return shape


def _tri(pa, pb, pc, y, w, t, holes=()):
    """Rigid triangular truss of three bars."""
    shape = _bar(pa, pb, y, w, t) + _bar(pb, pc, y, w, t) + _bar(pc, pa, y, w, t)
    for (x, z), d in holes:
        shape -= _cyl_y(x, z, y - t, y + t, d)
    return shape


def _plate(outline, y, thick, holes=()):
    """Bulkhead: X-Z polygon extruded `thick` along Y, centred on y, with ø holes at (x, z)."""
    shape = Pos(0, y + thick / 2, 0) * Rot(90, 0, 0) * extrude(Polygon(*outline), amount=thick)
    for (x, z), d in holes:
        shape -= _cyl_y(x, z, y - thick, y + thick, d)
    return shape


def _tint(color, f=0.45):
    rgb = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in rgb)


# ---------------------------------------------------------------- callable targets over the foot probe series
def _feet(report):
    probes = report["studies"][0]["series"]["probes"]
    return [np.asarray(v, float) for n, v in probes.items() if n.startswith("foot_")]


def flat_stance(band):
    """Worst foot: share of its stride covered while within `band` mm of its lowest point."""
    def flat_stance(report):
        return min(np.ptp(P[P[:, 2] <= P[:, 2].min() + band, 0]) / np.ptp(P[:, 0]) for P in _feet(report))
    return flat_stance


def feet_down(band):
    """Fewest feet within `band` mm of the common ground plane at any frame."""
    def feet_down(report):
        Z = np.array([P[:, 2] for P in _feet(report)])
        return int((Z <= Z.min() + band).sum(axis=0).min())
    return feet_down


def ground_spread(report):
    """Max − min of the per-foot lowest point: 0 when all feet share one ground plane."""
    return float(np.ptp([P[:, 2].min() for P in _feet(report)]))


# ---------------------------------------------------------------- model
def build(scale=1.0, stations=3, t=3.0, w=7.0, w_tri=8.0, gap=0.8, clearance=0.5, frames=61,
          d_pin=3.0, d_crankpin=4.0, d_pivot=4.0, motor_torque=0.4, band=2.0) -> Assembly:
    asm = Assembly("strandbeest", clearance=clearance)
    s, Y = scale, (0, 1, 0)
    a, l, m = HOLY["a"] * s, HOLY["l"] * s, HOLY["m"] * s
    hole = lambda d: d + 0.4                              # FDM running fit on every pin
    pitch = t + gap                                       # Y pitch of the link layers
    t_web, g_web, t_bulk, g_bulk = 3.0, 1.0, 8.0, 1.5
    half_st = 4 * pitch + g_web + t_web                   # half width of a station: 8 layers + 2 webs
    S = 2 * half_st + t_bulk + 2 * g_bulk                 # station pitch along Y
    ys = [(k - (stations - 1) / 2) * S for k in range(stations)]    # station centres
    yb = [(b - stations / 2) * S for b in range(stations + 1)]      # bulkhead centres
    y_layer = lambda k, i: ys[k] + (i - 3.5) * pitch
    y_web = lambda k, side: ys[k] + side * (half_st - t_web / 2)
    phases = [360.0 * k / stations for k in range(stations)]
    z_top = 34 * s + w_tri / 2 + 2                        # spine underside: above the upper triangle's reach
    y_m = yb[-1] + 40                                     # motor mounting face

    # frame: bulkheads with bearing seats and pivot holes, a motor plate, the 2020 spine, stub pivots
    bulk = [(-a - 8, -l - 7), (a + 8, -l - 7), (a + 8, -l + 5), (14, 10), (14, z_top), (-14, z_top),
            (-14, 10), (-a - 8, -l + 5)]
    mplate = [(-25, -25), (25, -25), (25, 18), (14, 26), (14, z_top), (-14, z_top), (-14, 26), (-25, 18)]
    frame = [_plate(bulk, y, t_bulk, [((0, 0), 22.0), ((-a, -l), d_pivot), ((a, -l), d_pivot)]) for y in yb]
    frame.append(_plate(mplate, y_m - 3, 6.0, [((0, 0), 22.0)] + [((sx * 15.5, sz * 15.5), 3.0) for sx in (-1, 1) for sz in (-1, 1)]))
    asm.part("frame", frame, ground=True, material="acrylic", color="#2f3640", opacity=0.85)
    spine = Pos(0, yb[0] - t_bulk / 2, z_top + 10) * Rot(-90, 0, 0) * extrusion_2020(y_m - yb[0] + t_bulk / 2)
    asm.part("spine", spine, ground=True, color="#c0c5cc")
    stubs = []
    for k in range(stations):                             # each stub crosses only the upper + {j,f,c} layers
        stubs.append(_cyl_y(-a, -l, yb[k] - t_bulk / 2, y_layer(k, 1) + t / 2, d_pivot))
        stubs.append(_cyl_y(a, -l, y_layer(k, 6) - t / 2, yb[k + 1] + t_bulk / 2, d_pivot))
    asm.part("pivots", stubs, ground=True, material="steel", color="#7a828c")
    for b, y in enumerate(yb):
        asm.part(f"bearing_{b}", Pos(0, y, 0) * Rot(90, 0, 0) * bearing("608"), ground=True, color="#8d949c")
    motor = Pos(0, y_m, 0) * Rot(90, 0, 0) * nema17()    # face at y_m, body +Y, ø5 shaft 24 mm toward −Y
    asm.part("motor", motor, ground=True, color="#262626")
    for i in range(1, 5):
        p = motor.frames[f"hole_{i}"].position
        asm.part(f"screw_{i}", Pos(p.X, y_m - 6, p.Z) * Rot(90, 0, 0) * socket_head_screw("M3", 8), ground=True, color="#3a3a3a")
        asm.ignore(f"screw_{i}", "motor")                 # threads engage the motor's tapped holes

    # crankshaft: ø8 main journals outside the throws (rods j/k sweep across the axis inside them)
    crank = _cyl_y(0, 0, yb[0] - t_bulk / 2 - 17, y_web(0, -1), 8.0)
    for k in range(stations):
        C = (m * math.cos(math.radians(phases[k])), m * math.sin(math.radians(phases[k])))
        for side in (-1, 1):
            yw = y_web(k, side)
            crank += link((0, yw, 0), (C[0], yw, C[1]), width=10, thickness=t_web, z=yw - t_web / 2, normal=Y, hole=0)
        crank += _cyl_y(C[0], C[1], y_web(k, -1), y_web(k, 1), d_crankpin)
        if k + 1 < stations:
            crank += _cyl_y(0, 0, y_web(k, 1), y_web(k + 1, -1), 8.0)
    crank += _cyl_y(0, 0, y_web(stations - 1, 1), yb[-1] + 15, 8.0)
    asm.part("crank", crank, material="steel", color="#6b7280")
    yf = yb[0] - t_bulk / 2 - 13
    fly = _cyl_y(0, 0, yf - 3, yf + 3, 56.0) + _cyl_y(0, 0, yf + 3, yf + 6.5, 16.0) - _cyl_y(0, 0, yf - 4, yf + 8, 8.0)
    for i in range(6):
        fly -= _cyl_y(18 * math.cos(i * math.pi / 3), 18 * math.sin(i * math.pi / 3), yf - 4, yf + 4, 10.0)
    asm.part("flywheel", fly, material="aluminum", color="#b0b7bf")
    asm.fix("flywheel", "crank")
    coupler = _cyl_y(0, 0, yb[-1] + 6.5, yb[-1] + 32, 18.0) - _cyl_y(0, 0, yb[-1] + 5, yb[-1] + 15.5, 8.0)
    coupler -= _cyl_y(0, 0, yb[-1] + 15.5, yb[-1] + 34, 5.2)
    asm.part("coupler", coupler, material="aluminum", color="#9aa3ad")
    asm.fix("coupler", "crank")
    asm.revolute("j_crank", "frame", "crank", origin=(0, 0, 0), axis=YAX)

    # legs: layers (outside → in) upper · {j, f, c} · lower · k, mirrored across the station centre
    P3 = lambda p, yy: (p[0], yy, p[1])
    for k in range(stations):
        col = PALETTE[k % len(PALETTE)]
        light = _tint(col)
        for side, mirror, lay in (("L", 1, dict(upper=0, jfc=1, lower=2, k=3)), ("R", -1, dict(upper=7, jfc=6, lower=5, k=4))):
            tag = f"{side}{k}"
            pt = jansen(phases[k], s, mirror)
            P, C, B, D, E, F, G = (pt[n] for n in "PCBDEFG")
            y = {n: y_layer(k, i) for n, i in lay.items()}
            far = lambda frm, to: y[to] + math.copysign(t / 2, y[to] - y[frm])    # far face of layer `to`
            asm.part(f"j_{tag}", _bar(C, B, y["jfc"], w, t, [(C, hole(d_crankpin)), (B, hole(d_pin))]), color=light)
            asm.part(f"k_{tag}", _bar(C, E, y["k"], w, t, [(C, hole(d_crankpin)), (E, hole(d_pin))]), color=light)
            asm.part(f"c_{tag}", _bar(P, E, y["jfc"], w, t, [(P, hole(d_pivot)), (E, hole(d_pin))]), color=light)
            asm.part(f"f_{tag}", _bar(D, F, y["jfc"], w, t, [(D, hole(d_pin)), (F, hole(d_pin))]), color=light)
            upper = _tri(P, B, D, y["upper"], w_tri, t, [(P, hole(d_pivot))])
            upper += _cyl_y(*B, y["upper"], far("upper", "jfc"), d_pin) + _cyl_y(*D, y["upper"], far("upper", "jfc"), d_pin)
            asm.part(f"upper_{tag}", upper, color=col)
            lower = _tri(E, F, G, y["lower"], w_tri, t)
            lower += _cyl_y(*E, far("lower", "k"), far("lower", "jfc"), d_pin) + _cyl_y(*F, y["lower"], far("lower", "jfc"), d_pin)
            asm.part(f"lower_{tag}", lower, color=col)
            gx, gz = G                                    # rubber shoe cupped over the foot vertex
            bot, top = gz - w_tri / 2 - 3, gz + 3
            shoe = Pos(gx, y["lower"], (bot + top) / 2) * Box(1.5 * w_tri, t + 3, top - bot)
            shoe -= Pos(gx, y["lower"], (gz - w_tri / 2 - 0.1 + top + 1) / 2) * Box(2 * w_tri, t + 0.2, top + 1 - gz + w_tri / 2 + 0.1)
            asm.part(f"shoe_{tag}", shoe, material="TPU", color="#1e1e1e")
            asm.fix(f"shoe_{tag}", f"lower_{tag}")
            asm.revolute(f"jC_j_{tag}", "crank", f"j_{tag}", origin=P3(C, y["jfc"]), axis=YAX)
            asm.revolute(f"jC_k_{tag}", "crank", f"k_{tag}", origin=P3(C, y["k"]), axis=YAX)
            asm.revolute(f"jB_{tag}", f"j_{tag}", f"upper_{tag}", origin=P3(B, (y["jfc"] + y["upper"]) / 2), axis=YAX)
            asm.revolute(f"jD_{tag}", f"upper_{tag}", f"f_{tag}", origin=P3(D, (y["jfc"] + y["upper"]) / 2), axis=YAX)
            asm.revolute(f"jE_{tag}", f"k_{tag}", f"lower_{tag}", origin=P3(E, (y["k"] + y["lower"]) / 2), axis=YAX)
            asm.revolute(f"jP_{tag}", "pivots", f"c_{tag}", origin=P3(P, y["jfc"]), axis=YAX)
            asm.pin(f"pP_{tag}", f"upper_{tag}", "pivots", point=P3(P, y["upper"]), axis=YAX)
            asm.pin(f"pE_{tag}", f"c_{tag}", f"lower_{tag}", point=P3(E, (y["jfc"] + y["lower"]) / 2), axis=YAX)
            asm.pin(f"pF_{tag}", f"f_{tag}", f"lower_{tag}", point=P3(F, (y["jfc"] + y["lower"]) / 2), axis=YAX)
            asm.probe(f"foot_{tag}", part=f"lower_{tag}", point=P3(G, y["lower"]))

    asm.actuator("j_crank", capacity=motor_torque)       # NEMA 17 holding torque, N·m
    asm.study("walk", drive={"j_crank": (0, 360)}, frames=frames)
    asm.target("stride", "delta:foot_L0.x", min=60 * s)
    asm.target("step lift", "delta:foot_L0.z", min=15 * s)
    asm.target("flat stance (worst foot)", flat_stance(band * s), min=0.85)
    asm.target("feet on the ground", feet_down(band * s), min=2)
    asm.target("one ground plane", ground_spread, max=0.05 * s)
    asm.target("motor SF", "sf:j_crank", min=3)
    asm.target("no collisions", "clearance", min=clearance)
    return asm


if __name__ == "__main__":
    run(build())
