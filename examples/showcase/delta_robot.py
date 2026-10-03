"""Clavel delta robot: three NEMA17 upper arms, six carbon rods on ball studs, a level platform.

Layout (world mm, Z up, base plate on top): motor shafts at z = 0 on a circle of radius r_base; leg
i lies in the vertical plane at phi_i = 90°, 210°, 330°, u_i = radial unit, t_i = z × u_i = hinge
axis (+theta = arm down, the delta convention; `home=theta0` makes joint values absolute angles).
Each lower arm is a spatial parallelogram of two rods; every rod end is a rod-end style socket (a
ring around a ø10 ball on an M3 stud, 0.15 mm radial play) that lets the rod swing ±40°.

Tree: motor → arm → [stud spin about t] stud → [swing about t × d] rod. That U-joint's knuckle is
the ball stud itself — a body of revolution, so its spin is invisible. Leg 1a adds [tube spin about
d] between its upper socket and tube, so the platform's tree path carries a full 3-DOF ball, and the
platform hangs from rod 1a through a second U (swing, stud spin). The other five rods close the
spatial loops with ball pins at the platform studs: 15 passive unknowns, 15 equations, mobility 0.

A ball joint built from a stud and two revolutes leaves its outer pair (arm ↔ rod) two joints
apart, so no single joint carries it, although the stud neck swinging inside the socket ring is
the design condition; those pairs are declared `asm.joined` (exempt from clearance, any overlap is
still an error). Every nominal fit a joint or pin itself carries (ball in its ring, shaft in its
bore, a stud spinning in its seat) needs no declaration.

The motors follow the closed-form delta IK so the tool traces a pick-and-place loop and a circle;
`DeltaKin.fk` (three-sphere intersection) checks mech's loop solution independently.
"""
import math

import numpy as np
from build123d import *
from mech import *
from mech.geom import link
from mech.parts import bearing, nema17

PHI = (90.0, 210.0, 330.0)                  # leg planes (deg)
NECK, GAP, BOSS = 10.0, 0.15, 12.0          # stud neck (bar face → ball centre), ball/ring play, tube boss
ORANGE, CARBON, CHROME, ALU, STEEL, DARK, BLUE = ("#E8792B", "#3C414B", "#D9DBDF", "#9AA6B8",
                                                   "#5B6168", "#2A2A2E", "#2F80ED")


# ------------------------------------------------------------------ geometry helpers (world mm)
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def loc_at(p, z_dir, x_dir=None) -> Location:
    """Location at point p with its Z axis along z_dir (X along x_dir when given)."""
    x = None if x_dir is None else tuple(map(float, x_dir))
    return Location(Plane(origin=tuple(map(float, p)), x_dir=x, z_dir=tuple(map(float, z_dir))))


def cyl(p, d, r, a, b):
    """Cylinder of radius r along unit d from p + a·d to p + b·d."""
    p, d = np.asarray(p, float), unit(d)
    return loc_at(p + (a + b) / 2 * d, d) * Cylinder(r, b - a)


def sphere(p, r):
    return Pos(*map(float, p)) * Sphere(r)


def stud(center, neck_dir, rb):
    """Ball stud: ball of radius rb with an M3 neck down to the face of the bar it grows from."""
    return sphere(center, rb) + cyl(center, neck_dir, 1.5, 0, NECK)


def socket_shell(center, axis, boss_dir, rb, rod_r):
    """Rod-end housing: a ring around the ball (its axis = the stud axis, the ball shows 1 mm on
    each face) plus a boss along boss_dir that carries the tube. Cut `socket_bore` after fusing."""
    return cyl(center, axis, rb + 2.0, -(rb - 1.0), rb - 1.0) + cyl(center, boss_dir, rod_r + 1.5, 0, BOSS)


def socket_bore(center, axis, rb):
    return cyl(center, axis, rb + GAP, -(rb + 1.0), rb + 1.0)


# ------------------------------------------------------------------ closed-form delta kinematics
class DeltaKin:
    """Clavel delta: arm angles measured downward from horizontal (deg); platform centre E."""

    def __init__(self, r_base, l_arm, l_rod, r_plat):
        self.r_base, self.l_arm, self.l_rod, self.r_plat = r_base, l_arm, l_rod, r_plat
        rad = np.radians(PHI)
        self.U = np.stack([np.cos(rad), np.sin(rad), np.zeros(3)], axis=1)      # radial units
        self.T = np.stack([-np.sin(rad), np.cos(rad), np.zeros(3)], axis=1)     # hinge axes z × u

    def elbow(self, i, theta):
        """Arm end (centre of the parallelogram hinge line) of leg i at arm angle theta."""
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        return (self.r_base + self.l_arm * c) * self.U[i] + np.array((0.0, 0.0, -self.l_arm * s))

    def ik(self, E):
        """Arm angles (deg) placing the platform centre at E (elbow-out branch)."""
        E, out = np.asarray(E, float), []
        for u, t in zip(self.U, self.T):
            a, b, y = E @ u + self.r_plat - self.r_base, E[2], E @ t
            c = (self.l_arm ** 2 + a * a + b * b + y * y - self.l_rod ** 2) / (2 * self.l_arm)
            rho = math.hypot(a, b)
            if abs(c) > rho:
                raise ValueError(f"effector {np.round(E, 1)} is out of reach")
            out.append(math.degrees(math.atan2(-b, a) - math.acos(c / rho)))
        return np.array(out)

    def fk(self, theta):
        """Platform centre for arm angles (deg): the lower intersection of three spheres of radius
        l_rod about the elbows shifted inward by r_plat (trilateration)."""
        C = [self.elbow(i, q) - self.r_plat * self.U[i] for i, q in enumerate(theta)]
        ex = unit(C[1] - C[0])
        d, i = np.linalg.norm(C[1] - C[0]), ex @ (C[2] - C[0])
        ey = unit(C[2] - C[0] - i * ex)
        j = ey @ (C[2] - C[0])
        x = d / 2                                   # equal radii
        y = (i * i + j * j) / (2 * j) - i * x / j
        z = math.sqrt(self.l_rod ** 2 - x * x - y * y)
        lo, hi = (C[0] + x * ex + y * ey + s * z * np.cross(ex, ey) for s in (-1, 1))
        return lo if lo[2] < hi[2] else hi


def pick_place(u, reach, z_lo, z_hi, y_back):
    """Pick at −x, place at +x, return over +y: 7 smooth-stepped straight segments, u ∈ [0, 1]."""
    W = np.array([(-reach, 0, z_hi), (-reach, 0, z_lo), (-reach, 0, z_hi), (reach, 0, z_hi),
                  (reach, 0, z_lo), (reach, 0, z_hi), (0, y_back, z_hi), (-reach, 0, z_hi)], float)
    n = len(W) - 1
    k = min(int(u * n), n - 1)
    tau = u * n - k
    return W[k] + (tau * tau * (3 - 2 * tau)) * (W[k + 1] - W[k])


# ------------------------------------------------------------------ callable targets
def fk_error(report, kin) -> float:
    """Max distance (mm) between mech's tcp probe and the closed-form FK of the three arm angles."""
    worst = 0.0
    for st in report["studies"]:
        S = st["series"]
        for k, ok in enumerate(S["ok"]):
            if ok:
                q = [S["joints"][f"j_arm_{i}"][k] for i in (1, 2, 3)]
                worst = max(worst, float(np.linalg.norm(kin.fk(q) - np.array(S["probes"]["tcp"][k]))))
    return worst


def path_error(report, study, path) -> float:
    """Max distance (mm) between the tcp probe and the commanded path over a `once` study."""
    S = next(s for s in report["studies"] if s["name"] == study)["series"]
    n = len(S["ok"])
    return max(float(np.linalg.norm(path(k / (n - 1)) - np.array(S["probes"]["tcp"][k])))
               for k, ok in enumerate(S["ok"]) if ok)


def worst_sf(report) -> float:
    sfs = [ld["sf"] for st in report["studies"] for ld in st["loads"].values() if ld.get("sf") is not None]
    return min(sfs) if sfs else math.inf


# ------------------------------------------------------------------ model
def build(r_base=100.0, l_arm=100.0, l_rod=240.0, r_plat=40.0, w=50.0, theta0=30.0, ball_d=10.0,
          rod_d=8.0, reach=80.0, z_pick=-280.0, z_travel=-230.0, y_return=55.0, circle_r=70.0,
          circle_z=-255.0, lift=(5.0, 70.0), payload_g=100.0, capacity=0.55, clearance=1.0) -> Assembly:
    kin = DeltaKin(r_base, l_arm, l_rod, r_plat)
    asm = Assembly("delta_robot", clearance=clearance)             # bearings and housings encircle the arm shafts
    rb, rr = ball_d / 2, rod_d / 2
    E0 = kin.fk([theta0] * 3)                                     # platform centre at home
    zE = float(E0[2])
    legs = [(i + 1, kin.U[i], kin.T[i], kin.elbow(i, theta0), E0 + r_plat * kin.U[i]) for i in range(3)]

    # ---- ground: hexagonal ring plate (z 25…33, open centre); per leg a motor bracket and a
    #      bearing support hang from it, both astride the arm hub
    ring = RegularPolygon(r_base + 50, 6) - RegularPolygon(r_base - 20, 6)
    asm.part("plate", Pos(0, 0, 25) * extrude(ring, 8), ground=True, material="aluminum_6061", color=STEEL)
    for i, u, t, A, P in legs:
        B = r_base * u                                            # motor shaft axis point
        for name, tc, th, bore in ((f"bracket_{i}", -10.0, 4.0, 11.5), (f"support_{i}", 17.5, 7.0, 11.0)):
            plate_side = loc_at(B + tc * t + (0, 0, -0.5), t, u) * Box(50, 51, th)   # z −26 … 25, meets the plate
            asm.part(name, plate_side - cyl(B, t, bore, tc - th, tc + th), ground=True,
                     material="aluminum_6061", color=STEEL)

    # ---- moving platform: hex hub + three spokes with the hinge bars; ball studs are fixed parts
    plat = Pos(0, 0, zE - 3) * extrude(RegularPolygon(32, 6), 6)
    for i, u, t, A, P in legs:
        plat += loc_at((r_plat / 2) * u + (0, 0, zE), (0, 0, 1), u) * Box(r_plat + 4, 12, 6)   # spoke
        plat += cyl(P, t, 4, -(w / 2 - NECK), w / 2 - NECK)
    asm.part("platform", plat, material="aluminum_6061", color=ALU)
    tool = cyl((0, 0, zE - 3), (0, 0, -1), 8, 0, 30) + loc_at((0, 0, zE - 33), (0, 0, -1)) * Cone(
        7, 13, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))                 # vacuum cup
    asm.part("tool", tool, material="aluminum_6061", color=DARK)
    asm.fix("tool", "platform")
    asm.part("payload", Pos(0, 0, zE - 50) * Cylinder(17, 18), mass_g=payload_g, color=BLUE)
    asm.fix("payload", "tool")
    asm.probe("tcp", part="platform", point=tuple(E0))

    # ---- legs
    for i, u, t, A, P in legs:
        B = r_base * u
        d, a2 = unit(P - A), unit(np.cross(t, P - A))             # rod direction, socket swing axis
        motor = nema17(48).mate("shaft", loc_at(B - 12 * t, t, u))  # face on the bracket, shaft along +t
        asm.part(f"motor_{i}", motor, ground=True, color=DARK)
        asm.part(f"bearing_{i}", loc_at(B + 17.5 * t, t) * bearing("608"), ground=True)
        arm = (cyl(B, t, 8, -6, 10) + link(B, A, width=12, thickness=8, z=-4, normal=t, hole=0)
               + cyl(A, t, 4, -(w / 2 - NECK), w / 2 - NECK)      # parallelogram hinge bar
               + cyl(B, t, 3.9, 10, 22) - cyl(B, t, 2.6, -7, 13))  # stub into the 608, motor shaft bore
        asm.part(f"arm_{i}", arm, material="aluminum_6061", color=ORANGE)
        asm.revolute(f"j_arm_{i}", f"motor_{i}", f"arm_{i}", at=motor.frames["shaft"], home=theta0)
        asm.actuator(f"j_arm_{i}", capacity=capacity)
        for s, tag in ((1, "a"), (-1, "b")):
            As, Ps, neck = A + s * w / 2 * t, P + s * w / 2 * t, -s * t
            rod, hi, lo = f"rod_{i}{tag}", f"stud_{i}{tag}", f"stud_{i}{tag}_lo"
            asm.part(hi, stud(As, neck, rb), material="steel", color=CHROME)
            asm.revolute(f"j_stud_{i}{tag}", f"arm_{i}", hi, origin=As, axis=t)
            if (i, tag) == (1, "a"):                              # tree path to the platform
                sock = socket_shell(As, t, d, rb, rr) - socket_bore(As, t, rb) - cyl(As, d, rr, 0, BOSS + 0.1)
                asm.part("socket_1a", sock, material="carbon_fiber", color=CARBON)
                asm.revolute("j_swing_1a", hi, "socket_1a", origin=As, axis=a2)
                body = cyl(As, d, rr, 6.5, l_rod) + socket_shell(Ps, t, -d, rb, rr) - socket_bore(Ps, t, rb)
                asm.part(rod, body, material="carbon_fiber", color=CARBON)
                asm.revolute("j_spin_1a", "socket_1a", rod, origin=As, axis=d)   # 3rd axis of the ball
                asm.part(lo, stud(Ps, neck, rb), material="steel", color=CHROME)
                asm.revolute("j_swing_1a_lo", rod, lo, origin=Ps, axis=a2)
                asm.revolute("j_stud_1a_lo", lo, "platform", origin=Ps, axis=t)
                asm.joined("arm_1", "socket_1a")                  # the U-joints' outer pairs
                asm.joined("platform", rod)
            else:
                body = (cyl(As, d, rr, 0, l_rod) + socket_shell(As, t, d, rb, rr) + socket_shell(Ps, t, -d, rb, rr)
                        - socket_bore(As, t, rb) - socket_bore(Ps, t, rb))
                asm.part(rod, body, material="carbon_fiber", color=CARBON)
                asm.revolute(f"j_swing_{i}{tag}", hi, rod, origin=As, axis=a2)
                asm.joined(f"arm_{i}", rod)                       # stud neck swings inside the upper ring
                asm.part(lo, stud(Ps, neck, rb), material="steel", color=CHROME)
                asm.fix(lo, "platform")
                asm.pin(f"p_{i}{tag}", rod, lo, point=Ps)            # ball joint: point only, no axis

    # ---- intent: IK-driven paths, a pure lift, and the requirements
    pp = lambda u: pick_place(u, reach, z_pick, z_travel, y_return)
    circ = lambda u: np.array((circle_r * math.cos(2 * math.pi * u), circle_r * math.sin(2 * math.pi * u), circle_z))
    ik_drive = lambda path: {f"j_arm_{i + 1}": (lambda u, i=i: float(kin.ik(path(u))[i])) for i in range(3)}
    asm.study("pick_place", drive=ik_drive(pp), frames=71)        # 7 segments × 10 frames
    asm.study("circle", drive=ik_drive(circ), frames=61)
    asm.study("lift", drive={f"j_arm_{i}": lift for i in (1, 2, 3)}, frames=27)
    asm.target("platform stays level", "rot:platform", max=0.02)
    asm.target("tracks IK path", lambda r: path_error(r, "pick_place", pp), max=0.01, study="pick_place")
    asm.target("closed-form FK agrees", lambda r: fk_error(r, kin), max=0.001)
    asm.target("X reach", "delta:tcp.x", min=2 * reach - 0.1, study="pick_place")
    asm.target("Z stroke", "delta:tcp.z", min=120, study="lift")
    asm.target("circle path", "path:tcp", min=2 * math.pi * circle_r * 0.995, study="circle")
    asm.target("motor SF with payload", worst_sf, min=2)
    asm.target("no collisions", "clearance", min=clearance)
    return asm


if __name__ == "__main__":
    run(build())
