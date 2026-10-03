"""5-cylinder radial aircraft engine: one crank throw, a MASTER rod and four ARTICULATED link rods.

Crank axis = Y (propeller toward -Y). The cylinders fan out in the XZ plane every 72 deg starting
at +Z; cylinder 0 carries the master rod, whose big-end flanges hold four knuckle pins at the
cylinder angles. Link rods hang on those pins; every piston slides radially in its own barrel
(prismatic joint) and is closed onto its rod by a wrist pin (loop closure). Five loops share the
master rod, so the articulated pistons' strokes and TDC timing differ slightly from the master's
(the classic radial-engine effect). Home pose: crank 0 deg = master piston at TDC.
Proportions follow real radials: crankcase radius ~3x the throw so the link rods clear the barrel
mouths, bore ~ stroke, L/r ~ 4.4 for the master rod.
"""
import math
from build123d import *
from mech import *
from mech.geom import link
from mech.parts import bearing, hex_nut, rod

Y = (0, 1, 0)
N_CYL = 5
COL = dict(case="#7d858f", barrel="#aeb6bf", head="#30343a", piston="#dfe3e8", pin="#c9ccd1",
           master="#b08d57", linkrod="#9ea4ab", crank="#3a3d44", hub="#c0c4c8", prop="#8b5a2b", hw="#6c7075")


def ring(ro, ri, z0, z1):
    """Annulus (ri=0: disc) along +Z between z0 and z1 (built in place, no 3D booleans)."""
    prof = Circle(ro) - Circle(ri) if ri > 0 else Circle(ro)
    return Pos(0, 0, z0) * extrude(prof, amount=z1 - z0)


def yring(ro, ri, y0, y1):
    """Annulus / disc along the crank axis (+Y) between y0 and y1."""
    return Rot(-90, 0, 0) * ring(ro, ri, y0, y1)


def flange(ro, ri, y0, y1, holes, d):
    """Y-axis annulus with through holes of diameter d at the given (x, z) offsets."""
    prof = Circle(ro) - Circle(ri)
    for x, z in holes:                      # Rot(-90,0,0) maps sketch (x, y) -> world (x, -y) in XZ
        prof -= Pos(x, -z) * Circle(d / 2)
    return Rot(-90, 0, 0) * Pos(0, 0, y0) * extrude(prof, amount=y1 - y0)


def tdc_phase_error(report) -> float:
    """Worst |crank angle at an articulated piston's TDC - its cylinder angle| in deg (parabolic peak)."""
    s = report["studies"][0]["series"]["joints"]
    th, worst = s["j_crank"], 0.0
    n = len(th) - 1                                              # frame n repeats frame 0 (360 = 0)
    for k in range(1, N_CYL):
        q = s[f"j_pist{k}"]
        i = max(range(n), key=lambda j: q[j])
        q0, q1, q2 = q[(i - 1) % n], q[i], q[(i + 1) % n]
        denom = q0 - 2 * q1 + q2
        tdc = th[i] + (0.5 * (q0 - q2) / denom if denom else 0.0) * (th[1] - th[0])
        worst = max(worst, abs((tdc - 72 * k + 180) % 360 - 180))
    return worst


def build(r=22.0, L=96.0, rho=24.0, l=72.0, bore=40.0, t=6.0, pitch=25.0, capacity=0.25, frames=73) -> Assembly:
    asm = Assembly("radial_engine", clearance=0.3)                # the 6001 bores encircle the crank axis: carried
    phi = [72.0 * k for k in range(N_CYL)]                       # cylinder angles from +Z toward +X (deg)
    d = [(math.sin(math.radians(p)), 0.0, math.cos(math.radians(p))) for p in phi]
    C = (0.0, 0.0, r)                                            # crankpin at home
    K = [(rho * dk[0], 0.0, r + rho * dk[2]) for dk in d]        # knuckle pins (master rod points along +Z)

    def radial_home(k):                                          # piston-pin radius that closes loop k at home
        if k == 0:
            return r + L
        kd = K[k][0] * d[k][0] + K[k][2] * d[k][2]
        return kd + math.sqrt(l ** 2 - (K[k][0] ** 2 + K[k][2] ** 2) + kd ** 2)

    s_home = [radial_home(k) for k in range(N_CYL)]
    P = [(s * dk[0], 0.0, s * dk[2]) for s, dk in zip(s_home, d)]
    R_in, R_out, yc = 68.0, 78.0, 24.0                           # crankcase ring radii, half-width
    spig, pad = 66.0, 80.0                                       # barrel spigot bottom, cylinder pad face
    Rb_i, Rb_o = bore / 2, bore / 2 + 3                          # barrel bore / wall
    s_top = r + L + 16 + 2                                       # barrel top: master crown at TDC + 2 mm squish
    cyl_rot = [Rot(0, p, 0) for p in phi]                        # +Z -> cylinder axis d_k

    # --- ground: crankcase ring with 5 pads; nose and rear covers (with the 6001 seats) bolt on
    case = yring(R_out, R_in, -yc, yc)
    for Rk in cyl_rot:
        case += Rk * ring(27, 0, R_in, pad) & Box(400, 2 * yc, 400)                   # pad, flats at the ring faces
        case -= Rk * ring(Rb_o, 0, R_in - 6, pad + 1)                                  # spigot bore (nominal)
    asm.part("crankcase", case, material="aluminum_6061", ground=True, color=COL["case"], opacity=0.45)
    nose = yring(R_out, 8, -yc - 6, -yc) + yring(22, 8, -32, -yc - 6) + yring(22, 14, -48, -32)    # + 6001 seat
    rear = yring(R_out, 8, yc, yc + 6) + yring(22, 8, yc + 6, 32) + yring(22, 14, 32, 46)
    asm.part("nose_case", nose, material="aluminum_6061", ground=True, color=COL["case"], opacity=0.45)
    asm.part("rear_case", rear, material="aluminum_6061", ground=True, color=COL["case"], opacity=0.45)
    asm.part("bearing_f", Pos(0, -36, 0) * Rot(-90, 0, 0) * bearing("6001"), ground=True)
    asm.part("bearing_r", Pos(0, 36, 0) * Rot(-90, 0, 0) * bearing("6001"), ground=True)
    # barrels (translucent liners), heads, and all cooling fins as one ground part per cylinder
    for k, Rk in enumerate(cyl_rot):
        liner = [ring(Rb_o, Rb_i, spig, s_top), ring(29, Rb_o, pad, pad + 4)]            # liner + flange on the pad
        asm.part(f"barrel{k}", [Rk * s for s in liner], material="steel", ground=True,
                 color=COL["barrel"], opacity=0.35)
        head = [ring(Rb_o + 1, 0, s_top, s_top + 16), Pos(0, 0, s_top + 22) * Box(34, 22, 12),  # head + rocker box
                Pos(0, 0, s_top + 22) * yring(5, 3.5, -36, -9), Pos(0, 0, s_top + 22) * yring(5, 3.5, 9, 36)]  # exhaust/intake
        asm.part(f"head{k}", [Rk * s for s in head], material="aluminum_6061", ground=True, color=COL["head"])
        fins = [ring(32, Rb_o, z, z + 1.5) for z in range(int(pad) + 9, int(s_top) - 2, 4)]
        fins += [ring(34, Rb_o + 1, s_top + z, s_top + z + 1.5) for z in (2, 6, 10, 14)]
        asm.part(f"fins{k}", [Rk * s for s in fins], material="aluminum_6061", ground=True,
                 color=COL["barrel"], opacity=0.35)
        hw = [Pos(26.5 * math.cos(a), 26.5 * math.sin(a), pad + 4) * hex_nut("M3").shape         # hold-down nuts
              for a in (math.radians(45 + 90 * i) for i in range(4))]
        hw += [Pos(sx * 20, -31, 0) * ring(2.2, 0, pad + 2, s_top + 20) for sx in (-1, 1)]       # pushrod tubes
        asm.part(f"hw{k}", [Rk * s for s in hw], material="steel", ground=True, color=COL["hw"])

    # --- crankshaft: front journal (prop shaft) + two counterweighted webs + crankpin + rear journal
    def web(y0, y1):
        arm = link((0, 0, 0), C, width=28, thickness=y1 - y0, normal=Y, z=y0, hole=0)
        counterweight = yring(40, 0, y0, y1) & Pos(0, (y0 + y1) / 2, -25) * Box(90, y1 - y0 + 0.2, 50)
        return arm + counterweight

    crank = yring(6, 0, -75, -16) + web(-16, -10) + Pos(0, 0, r) * yring(6, 0, -10, 10) + web(10, 16) + yring(6, 0, 16, 42)
    asm.part("crank", crank, material="steel", color=COL["crank"])
    spinner = Pos(0, -75, 0) * Rot(90, 0, 0) * Cone(20, 3, 22, align=(Align.CENTER, Align.CENTER, Align.MIN))
    asm.part("hub", [yring(13, 6, -69, -49), spinner], material="aluminum_6061", color=COL["hub"])
    blade = Pos(127, 0, -3) * extrude(SlotCenterToCenter(176, 30), amount=6)         # radius 24..230, chord 30
    blade = Pos(0, -72, 0) * Rot(90 + pitch, 0, 0) * blade                            # pitched, into the XZ plane
    asm.part("prop", [yring(26, 6, -75, -69), blade, Rot(0, 180, 0) * blade], material="plywood", color=COL["prop"])
    asm.fix("hub", "crank")
    asm.fix("prop", "crank")

    # --- master rod: I-shank + big-end boss + two knuckle flanges; link rods run between the flanges
    yf = t / 2 + 0.5                                                                   # flange inner face
    shank = link(C, P[0], width=13, thickness=t, normal=Y, z=-t / 2, hole=0)
    shank -= Pos(0, 0, r) * yring(6, 0, -t, t)                                          # crankpin bore
    shank -= Pos(*P[0]) * yring(2.5, 0, -t, t)                                          # wrist-pin bore
    holes = [(K[k][0], K[k][2] - r) for k in range(1, N_CYL)]
    master = [shank, Pos(0, 0, r) * yring(10, 6, -yf - 3.5, yf + 3.5),
              Pos(0, 0, r) * flange(rho + 4.5, 10, yf, yf + 3.5, holes, 5.0),
              Pos(0, 0, r) * flange(rho + 4.5, 10, -yf - 3.5, -yf, holes, 5.0)]
    asm.part("master_rod", master, material="steel", color=COL["master"])
    asm.revolute("j_crank", "crankcase", "crank", origin=(0, 0, 0), axis=Y)
    asm.revolute("j_master", "crank", "master_rod", origin=C, axis=Y)
    for k in range(1, N_CYL):
        asm.part(f"link{k}", link(K[k], P[k], width=10, thickness=t, normal=Y, z=-t / 2, hole=5),
                 material="steel", color=COL["linkrod"])
        asm.part(f"kpin{k}", Pos(K[k][0], -yf - 4, K[k][2]) * Rot(-90, 0, 0) * rod(5, 2 * (yf + 4)), color=COL["pin"])
        asm.fix(f"kpin{k}", "master_rod")
        asm.revolute(f"j_link{k}", "master_rod", f"link{k}", origin=K[k], axis=Y)

    # --- pistons (hollow, pin bosses, o5 wrist pin) sliding in their barrels
    for k, Rk in enumerate(cyl_rot):
        s, R, rod_k = s_home[k], bore / 2, "master_rod" if k == 0 else f"link{k}"
        body = ring(R, 0, s - 7.5, s + 16) - ring(R - 3, 0, s - 8.5, s + 10)           # skirt + 6 mm crown
        for sy in (-1, 1):
            body += Pos(0, sy * (R + 1) / 2, s + 1.25) * Box(12, R - 7, 17.5)           # pin bosses
        body -= Pos(0, 0, s) * yring(2.5, 0, -R - 1, R + 1)                             # wrist-pin bore
        asm.part(f"piston{k}", Rk * body, material="aluminum_6061", color=COL["piston"])
        asm.part(f"wpin{k}", Rk * (Pos(0, -R + 2, s) * Rot(-90, 0, 0) * rod(5, 2 * R - 4)), color=COL["pin"])
        asm.fix(f"wpin{k}", f"piston{k}")
        asm.prismatic(f"j_pist{k}", "crankcase", f"piston{k}", origin=P[k], axis=d[k], home=s)
        asm.pin(f"p_wrist{k}", rod_k, f"piston{k}", point=P[k], axis=Y)
        asm.probe(f"wrist{k}", part=f"piston{k}", point=P[k])
    # Nominal fits — a piston in its bore, the crank journals in their 6001s — need no declaration:
    # the joints carry them (any overlap is still an interference).

    # --- intent
    asm.actuator("j_crank", capacity=capacity)                                          # hand-propping torque, N*m
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=frames)
    asm.target("master stroke = 2r", "span:j_pist0", min=2 * r - 0.01, max=2 * r + 0.01)
    for k in range(1, N_CYL):
        asm.target(f"link stroke {k} within 2%", f"span:j_pist{k}", min=0.98 * 2 * r, max=1.02 * 2 * r)
    asm.target("TDC phase error", tdc_phase_error, max=5.0)
    asm.target("rods clear case & each other", "clearance", min=0.3)
    asm.target("hand-prop SF", "sf:j_crank", min=2)
    return asm


if __name__ == "__main__":
    run(build())
