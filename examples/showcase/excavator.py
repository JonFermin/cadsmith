"""Hydraulic excavator, 1:10 desktop scale (mm): slewing house on a tracked undercarriage; boom,
stick and bucket driven by four hydraulic cylinders (barrel + rod on a prismatic joint, pinned at
both ends), the bucket through the classic H-link / power-link four-bar.

Home pose = maximum reach at ground level (bucket tip on z = 0, bucket rolled out), so frame 0 of
the dig cycle is the pose of the boom-cylinder hand check. The two boom cylinders share one valve:
the left rod is the driver, the right one a second (passive) loop the solver closes — mech rates
one load per loop system, so `cap_boom` is the capacity of the PAIR.
"""
import math
import numpy as np
from scipy.optimize import brentq
from build123d import *
from mech import *
from mech.geom import circle_intersect, from_location, link, to_location, transform_points, unit

Y = (0, 1, 0)
YELLOW, BLACK, STEEL, CHROME, IRON = "#F2B705", "#1C1C1C", "#8D949B", "#DCE0E3", "#2B2B2B"
SOIL, GLASS = "#6B4A2B", "#1E2A31"
PIN, BORE = 12.0, 12.8                      # pin ø and running bore (0.4 mm radial gap)

# ---- modeling helpers: everything is drawn in place, in world mm, at the home pose ---------------
def frame_xz(p0, p1):
    """Location with X along p0→p1 (any direction ∥ the XZ plane), Y = world Y, origin p0."""
    T, x = np.eye(4), unit(np.subtract(p1, p0))
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, (0, 1, 0), np.cross(x, (0, 1, 0)), p0
    return to_location(T)

def heading(p, deg):
    """frame_xz at world point p whose X axis points `deg` above +X."""
    return frame_xz(p, (p[0] + math.cos(math.radians(deg)), p[1], p[2] + math.sin(math.radians(deg))))

def at(loc, x, z):
    """World point of the local (x, z) of a frame_xz Location."""
    return tuple(transform_points(from_location(loc), (x, 0.0, z)))

def prism(pts, y0, y1):
    """(x, z) polygon extruded along Y from y0 to y1."""
    return Pos(0, y1, 0) * Rot(90, 0, 0) * extrude(Polygon(*pts, align=None), amount=y1 - y0)

def box_section(pts, half_w, wall):
    """Welded box beam: (x, z) outline, width 2·half_w, `wall` thick, closed ends."""
    inner = offset(Polygon(*pts, align=None), amount=-wall, kind=Kind.INTERSECTION)
    return prism(pts, -half_w, half_w) - (Pos(0, half_w - wall, 0) * Rot(90, 0, 0)
                                          * extrude(inner, amount=2 * (half_w - wall)))

def xcyl(x0, x1, d):
    return Pos((x0 + x1) / 2, 0, 0) * Rot(0, 90, 0) * Cylinder(d / 2, x1 - x0)

def ycyl(p, d, w):
    return Pos(*p) * Rot(90, 0, 0) * Cylinder(d / 2, w)             # pin / boss along Y, centered

def ring(p, od, bore, w):
    return ycyl(p, od, w) - ycyl(p, bore, w + 2)

def plates(p0, p1, width, y_in, t, hole=0.0):
    """Mirrored pair of rounded side plates between two world points, inner faces at y = ±y_in."""
    return [link(p0, p1, width=width, thickness=t, z=y_in, normal=Y, hole=hole),
            link(p0, p1, width=width, thickness=t, z=-y_in - t, normal=Y, hole=hole)]

def hyd_cylinder(P, Q, closed, stroke, od, rod_d, exposed=12.0):
    """Barrel + rod between base pin P and rod pin Q (world). `closed` = pin-to-pin fully retracted,
    `exposed` = rod showing between barrel head and rod eye when retracted.
    Returns (barrel, rod, axis, (vmin, vmax)) with the prismatic value = extension from home."""
    L0 = float(np.linalg.norm(np.subtract(Q, P)))
    vmin, vmax = closed - L0, closed + stroke - L0
    e, r_eye, bore = 16.0, 12.0, od - 6                 # eye centre → barrel bottom, eye radius, bore ø
    Lt = closed - e - r_eye - exposed                   # barrel tube length
    a0 = e + 6.0 - vmin                                 # piston face at home (6 mm off the bottom when retracted)
    if a0 + vmax > e + Lt - 10:
        raise ValueError(f"cylinder closed length {closed:.0f} too short for stroke {stroke:.0f}")
    body = ring((0, 0, 0), 2 * r_eye + 2, BORE, 20) + xcyl(r_eye - 2, e + 2, 16) + xcyl(e, e + Lt, od)
    barrel = (body + xcyl(e + Lt - 10, e + Lt, od + 4)) - xcyl(e + 4, e + Lt + 1, bore)
    rod = xcyl(a0, L0 - r_eye + 1, rod_d) + xcyl(a0, a0 + 8, bore - 1) + ring((L0, 0, 0), 24, BORE, 20)
    loc = frame_xz(P, Q)
    return loc * barrel, loc * rod, unit(np.subtract(Q, P)), (vmin, vmax)

def tri_len(a, b, phi):
    return math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(phi)))

def ang(v):
    return math.degrees(math.atan2(v[1], v[0]))

# ---- design-target callables (report dict in, number out) -------------------------------------
def tip_reach(report):
    """Largest horizontal distance of the bucket tip from the swing axis over every study (mm)."""
    return max(math.hypot(p[0], p[1]) for s in report["studies"]
               for ok, p in zip(s["series"]["ok"], s["series"]["probes"]["tip"]) if ok)

def dig_depth(report):
    """Deepest bucket-tip position below ground (z = 0) over every study (mm)."""
    return -min(p[2] for s in report["studies"]
                for ok, p in zip(s["series"]["ok"], s["series"]["probes"]["tip"]) if ok)

def build(Lb=570.0, Ls=290.0, Ltip=150.0, alpha=5.0, boom_range=(-30.0, 55.0), bend_range=(30.0, 135.0),
          bucket_open=-5.0, bucket_stroke=114.5, Lh=75.0, Lp=90.0, rk=40.0, payload_g=1000.0,
          supply_mpa=1.5, strict=False) -> Assembly:
    asm = Assembly("excavator", clearance=0.3)                    # 12.8 mm eyes encircle their 12 mm pins: carried
    F = (35.0, 0.0, 215.0)                                        # boom foot pin (world)
    loc_boom = heading(F, alpha)                                  # local x along the boom chord F→S
    S = at(loc_boom, Lb, 0)                                       # boom tip / stick pivot
    tip_z = lambda bend: S[2] - Ls * math.sin(math.radians(bend - alpha)) \
        - Ltip * math.sin(math.radians(bend - alpha - bucket_open))
    bend = brentq(tip_z, bend_range[0], bend_range[1])            # stick bend that puts the tip on the ground
    loc_stick = heading(S, alpha - bend)                          # local x along S→B
    B = at(loc_stick, Ls, 0)                                      # bucket pivot
    loc_bucket = heading(B, alpha - bend + bucket_open)           # local x along B→tooth tip

    # ---- undercarriage (ground) ----
    for side, y0 in (("l", 100.0), ("r", -160.0)):
        asm.part(f"track_{side}", link((-190, 0, 45), (190, 0, 45), width=90, thickness=60, z=y0, normal=Y, hole=0),
                 ground=True, color=BLACK, material="steel", mass_g=3500)
    asm.part("carbody", Pos(-7.5, 0, 63.5) * Box(205, 199, 43), ground=True, color=YELLOW, mass_g=2600)
    asm.part("slew_ring", Pos(0, 0, 94) * Cylinder(70, 12), ground=True, color=STEEL, material="steel")

    # ---- upper structure: house + cab, child of the swing joint ----
    xP, zP, yc = 85.0, 112.0, 45.0                                # boom-cylinder base pins (±yc)
    house = Pos(-45, 0, 109.25) * Box(210, 256, 17.5)                       # deck frame x −150…60
    house += Pos(-106, 0, 176.5) * Box(88, 256, 117)                        # engine deck (behind the cab)
    house += Pos(-6, -82, 165) * Box(112, 92, 94)                           # hydraulic tank / hood
    house += (Pos(-225, 0, 167.75) * Box(150, 256, 134.5)) & Cylinder(285, 400)   # rounded counterweight
    house += Pos(-110, -70, 245) * Cylinder(8, 24)                          # exhaust
    house += Pos(xP - 8, 0, 110) * Box(40, 56, 18)                          # boom-cylinder bracket
    house += ycyl((xP, 0, zP), PIN, 2 * (yc + 13))                          # boom-cylinder base pin
    for p in plates((F[0], 0, 150), F, 50, 28.5, 11.5) + [ycyl(F, PIN, 80)]:   # boom pedestal + foot pin
        house += p
    asm.part("house", house, color=YELLOW, mass_g=7500)
    asm.part("cab", Pos(0, 84.5, 209) * Box(120, 81, 182), color=YELLOW, mass_g=600)
    asm.part("glass", [Pos(62, 84.5, 228) * Box(3, 73, 125), Pos(0, 127, 228) * Box(110, 3, 125),
                       Pos(0, 42, 228) * Box(110, 3, 125)], color=GLASS, mass_g=60)
    asm.fix("cab", "house")
    asm.fix("glass", "house")
    asm.revolute("j_swing", "slew_ring", "house", origin=(0, 0, 100), axis=(0, 0, 1))

    # ---- boom (gooseneck box beam), local x along the chord F→S ----
    Ql, Rl = (330.0, -62.0), (300.0, 92.0)                        # boom-cyl rod pin, stick-cyl base (boom-local)
    boom = box_section([(0, -36), (300, -40), (500, -22), (500, 20), (300, 72), (0, 36)], 28, 2.5)
    for p in (plates((490, 0, 0), (Lb, 0, 0), 44, 22.5, 5.5)                       # long slim tip fork
              + [Pos(0, y, 0) * ring((0, 0, 0), 72, BORE, 8) for y in (24, -24)]     # foot bosses
              + [link((Ql[0], 0, Ql[1]), (Ql[0], 0, -42), width=28, thickness=40, z=-20, normal=Y, hole=0),
                 ycyl((Ql[0], 0, Ql[1]), PIN, 2 * (yc + 13))]                       # boom-cyl lug + pin
              + plates((300, 0, 70), (Rl[0], 0, Rl[1]), 26, 10.5, 6) + [ycyl((Rl[0], 0, Rl[1]), PIN, 34)]
              + [ycyl((Lb, 0, 0), PIN, 56)]):                                       # stick pin
        boom += p
    boom -= ycyl((0, 0, 0), BORE, 120)                            # foot bore (pin belongs to the house)
    asm.part("boom", loc_boom * boom, material="aluminum_6061", color=YELLOW)
    asm.revolute("j_boom", "house", "boom", origin=F, axis=(0, -1, 0), limits=boom_range, home=alpha)

    # ---- stick: box beam, tall heel plates (stick-cylinder fork), slim nose, local x along S→B ----
    Hl, Tl, Gl = (-75.0, 65.0), (0.0, 66.0), (240.0, 14.0)       # stick-cyl rod pin, bucket-cyl base, H-link pivot
    stick = box_section([(-5, -30), (240, -21), (240, 21), (90, 34), (-5, 50)], 22, 2.5)
    heel = [(10, -25), (10, 48), (-22, 70), (-68, 88), (-88, 76), (-90, 50), (-45, -2)]
    for p in ([prism(heel, 11, 22), prism(heel, -22, -11), ycyl((Hl[0], 0, Hl[1]), PIN, 44)]
              + plates((Tl[0], 0, 44), (Tl[0], 0, Tl[1]), 24, 10.5, 6) + [ycyl((Tl[0], 0, Tl[1]), PIN, 34)]
              + [ycyl((Gl[0], 0, Gl[1]), 24, 44), ycyl((Gl[0], 0, Gl[1]), PIN, 84),      # H-link boss + pin
                 link((240, 0, 0), (Ls, 0, 0), width=28, thickness=44, z=-22, normal=Y, hole=0),   # nose
                 ycyl((Ls, 0, 0), 40, 44), ycyl((Ls, 0, 0), PIN, 57)]):                         # boss + bucket pin
        stick += p
    stick -= ycyl((0, 0, 0), BORE, 60)                            # root bore (pin belongs to the boom)
    asm.part("stick", loc_stick * stick, material="aluminum_6061", color=YELLOW)
    asm.revolute("j_stick", "boom", "stick", origin=S, axis=Y, limits=(0, bend_range[1] - bend_range[0]),
                 home=bend - bend_range[0])

    # ---- bucket: arc shell on the far side of the rim chord, side plates, lip, teeth, ears ----
    W = 150.0
    Kl = (rk * math.cos(math.radians(123.0)), rk * math.sin(math.radians(123.0)))   # power-link pin (bucket-local)
    Tz, E, sag = np.array([120.0, 0.0]), np.array([38.0, 14.0]), 50.0           # lip edge, rear rim edge, depth
    c = float(np.linalg.norm(Tz - E))
    R = (c * c / 4 + sag * sag) / (2 * sag)
    u = (Tz - E) / c
    n = np.array([-u[1], u[0]])                                   # rim-chord normal toward the shell
    mid = (E + Tz) / 2
    C = mid + (sag - R) * n                                       # shell arc centre
    th_t, th_e = ang(Tz - C) % 360, ang(E - C) % 360
    fan = [(C[0], C[1])] + [(C[0] + 3 * R * math.cos(math.radians(t)), C[1] + 3 * R * math.sin(math.radians(t)))
                            for t in np.linspace(th_e, th_t, 14)]
    mouth = prism(fan, -W / 2 - 2, W / 2 + 2)                      # everything on the bucket's open side
    shell = (Pos(C[0], 0, C[1]) * Rot(90, 0, 0) * (Cylinder(R, W) - Cylinder(R - 3, W + 2))) - mouth
    beta = ang(u)
    chord_cut = Pos(mid[0] - n[0] * R, 0, mid[1] - n[1] * R) * Rot(0, -beta, 0) * Box(4 * R, W + 10, 2 * R)
    sides = sum((Pos(C[0], y, C[1]) * Rot(90, 0, 0) * Cylinder(R - 0.5, 3) for y in (W / 2 - 1.5, -W / 2 + 1.5)), Part())
    bucket = shell + (sides - chord_cut) + Pos(130, 0, 0) * Box(28, W, 6)       # lip plate
    for y in (-60, -30, 0, 30, 60):                               # five teeth to x = Ltip
        bucket += Pos(Ltip - 6, y, 0) * Box(12, 8, 8)
    for p in plates((0, 0, 0), (Kl[0], 0, Kl[1]), 36, 22.5, 6) + plates((0, 0, 0), (34, 0, 10.5), 28, 22.5, 6):
        bucket += p                                               # ears: pivot → link pin, pivot → shell back
    bucket = bucket + ycyl((Kl[0], 0, Kl[1]), PIN, 70) - ycyl((0, 0, 0), BORE, 80)
    asm.part("bucket", loc_bucket * bucket, material="steel", color=IRON)
    spoil = (Pos(C[0], 0, C[1]) * Rot(90, 0, 0) * Cylinder(R - 8, W - 12)) \
        & (Pos(mid[0] + n[0] * (12 + R), 0, mid[1] + n[1] * (12 + R)) * Rot(0, -beta, 0) * Box(4 * R, W, 2 * R))
    asm.part("spoil", loc_bucket * spoil, color=SOIL, mass_g=payload_g)
    asm.fix("spoil", "bucket")
    asm.revolute("j_bucket", "stick", "bucket", origin=B, axis=Y)

    # ---- bucket four-bar: H-link on the stick, power link to the bucket ear ----
    G, K, T = at(loc_stick, *Gl), at(loc_bucket, *Kl), at(loc_stick, *Tl)
    Hj = max((circle_intersect(G, Lh, K, Lp, side=s, normal=Y) for s in (+1, -1)), key=lambda p: p[2])
    hlink = sum(plates(G, Hj, 26, 36, 6), Part()) + ycyl(Hj, PIN, 84) - ycyl(G, BORE, 100)
    for y in (19.5, -19.5):                                       # pin sleeves rod eye → power link (the "H")
        hlink += Pos(0, y, 0) * ycyl(Hj, 20, 18)
    asm.part("hlink", hlink, material="aluminum_6061", color=YELLOW)
    asm.part("plink", plates(Hj, K, 24, 29, 6, hole=BORE), material="aluminum_6061", color=YELLOW)
    asm.revolute("j_hlink", "stick", "hlink", origin=G, axis=Y)
    asm.revolute("j_plink", "hlink", "plink", origin=Hj, axis=Y)
    asm.pin("p_K", "plink", "bucket", point=K, axis=Y)

    # ---- hydraulic cylinders: closed length / stroke from the required angle ranges ----
    Pw, Qw, Rw, Hw = (xP, 0, zP), at(loc_boom, *Ql), at(loc_boom, *Rl), at(loc_stick, *Hl)
    a, b = np.linalg.norm(np.subtract(Pw, F)), np.linalg.norm(Ql)
    phi = lambda al: al + ang(Ql) - ang((Pw[0] - F[0], Pw[2] - F[2]))          # angle P–F–Q at boom angle al
    closed_b = tri_len(a, b, phi(boom_range[0]))
    stroke_b = tri_len(a, b, phi(boom_range[1])) - closed_b
    SR, SH = np.subtract(Rl, (Lb, 0)), np.array(Hl)
    psi = lambda bd: ang(SR) - ang(SH) + bd                                     # angle R–S–H at stick bend bd
    closed_s = tri_len(np.linalg.norm(SR), np.linalg.norm(SH), psi(bend_range[0]))
    stroke_s = tri_len(np.linalg.norm(SR), np.linalg.norm(SH), psi(bend_range[1])) - closed_s
    closed_k = float(np.linalg.norm(np.subtract(Hj, T))) - 5.0                  # home = 5 mm off fully retracted
    cyls = {"boom_l": ((xP, yc, zP), (Qw[0], yc, Qw[2]), "house", "boom", closed_b, stroke_b, 24, 12, 12),
            "boom_r": ((xP, -yc, zP), (Qw[0], -yc, Qw[2]), "house", "boom", closed_b, stroke_b, 24, 12, 12),
            "stick": (Rw, Hw, "boom", "stick", closed_s, stroke_s, 24, 12, 30),
            "bucket": (T, Hj, "stick", "hlink", closed_k, bucket_stroke, 20, 10, 12)}
    lim = {}
    for name, (P, Q, base, end, closed, stroke, od, rod_d, exposed) in cyls.items():
        barrel, rod, axis, lim[name] = hyd_cylinder(P, Q, closed, stroke, od, rod_d, exposed)
        asm.part(f"barrel_{name}", barrel, material="steel", color=IRON)
        asm.part(f"rod_{name}", rod, material="steel", color=CHROME)
        asm.revolute(f"j_barrel_{name}", base, f"barrel_{name}", origin=P, axis=Y)
        asm.prismatic(f"j_cyl_{name}", f"barrel_{name}", f"rod_{name}", origin=Q, axis=axis, limits=lim[name])
        asm.pin(f"p_{name}", f"rod_{name}", end, point=Q, axis=Y)
    if strict:                                                   # hold these joined pairs to `clearance` too
        for pair in (("bucket", "stick"), ("stick", "boom"), ("boom", "house"), ("hlink", "stick"),
                     ("rod_stick", "stick"), ("barrel_stick", "boom"), ("barrel_bucket", "stick"),
                     ("barrel_boom_l", "house"), ("barrel_boom_r", "house")):
            asm.check_clearance(*pair)

    # ---- intent: probe, actuators, studies, targets ----
    asm.probe("tip", part="bucket", point=at(loc_bucket, Ltip, 0))
    asm.probe("rim", part="bucket", point=at(loc_bucket, *E))                 # rear rim edge, passes behind the pivot
    asm.probe("nose", part="stick", point=at(loc_stick, Ls - 36, 0))          # stick nose axis where the rim passes
    force = lambda od, n=1: n * math.pi / 4 * (od - 6) ** 2 * supply_mpa   # N: bore area × supply pressure
    asm.actuator("j_cyl_boom_l", capacity=force(24, 2))         # the boom pair, rated on the driven rod
    asm.actuator("j_cyl_stick", capacity=force(24))
    asm.actuator("j_cyl_bucket", capacity=force(20))
    f_of = lambda name, v: (v - lim[name][0]) / (lim[name][1] - lim[name][0])     # extension → fraction
    v_of = lambda name, f: lim[name][0] + f * (lim[name][1] - lim[name][0])
    kf = lambda name, pts: [(u, v_of(name, f)) for u, f in pts]
    fb, fs, fk = f_of("boom_l", 0), f_of("stick", 0), f_of("bucket", 0)
    asm.study("dig", frames=101, duration=12.0, drive={                          # lower · curl · lift · swing · dump
        "j_cyl_boom_l": kf("boom_l", [(0, fb), (0.12, 0.0), (0.34, 0.0), (0.50, 0.3), (0.66, 0.85), (0.84, 0.85), (1, 0.6)]),
        "j_cyl_stick": kf("stick", [(0, fs), (0.12, fs), (0.34, 0.30), (0.50, 0.80), (0.66, 0.80), (0.84, 0.55), (1, 0.35)]),
        "j_cyl_bucket": kf("bucket", [(0, fk), (0.12, 0.08), (0.34, 0.12), (0.50, 1.0), (0.84, 1.0), (1, 0.0)]),
        "j_swing": [(0, 0), (0.66, 0), (0.84, 90), (1, 90)]})
    asm.study("boom", drive={"j_cyl_boom_l": lim["boom_l"]}, frames=21)                 # reach envelope
    asm.study("stick", drive={"j_cyl_boom_l": v_of("boom_l", 0.55), "j_cyl_stick": lim["stick"]}, frames=21)
    asm.study("bucket", drive={"j_cyl_boom_l": v_of("boom_l", 0.55), "j_cyl_stick": v_of("stick", 0.5),
                               "j_cyl_bucket": lim["bucket"]}, frames=25)
    asm.study("swing", drive={"j_swing": (0, 360)}, frames=13)
    asm.target("max reach", tip_reach, min=900)
    asm.target("dig depth", dig_depth, min=450)
    asm.target("boom range", "span:j_boom", min=80, study="boom")
    asm.target("stick range", "span:j_stick", min=100, study="stick")
    asm.target("rim clears nose", "min_dist:rim,nose", min=20, study="bucket")
    asm.target("bucket curl", "span:j_bucket", min=155, study="bucket")
    asm.target("bucket rot vs stick", "angle:stick,bucket", min=148, study="bucket")
    for name in ("boom_l", "stick", "bucket"):
        asm.target(f"{name.split('_')[0]} cylinder SF", f"sf:j_cyl_{name}", min=1.5)
    asm.target("no collision", "clearance", min=0.3)
    return asm

if __name__ == "__main__":
    run(build())
