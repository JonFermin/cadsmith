"""uArm-style 4-DOF desktop robot arm with every motor in the base (palletizer kinematics).

Yaw: a NEMA17 under the base plate turns the turntable through a GT2 20T:80T belt (608 bearing).
Shoulder and elbow: two NEMA17s stand inside the yoke on the turntable; each drives an 80T pulley
outboard of its yoke plate through a 20T:80T belt (one 250 mm belt size for all three drives).
The shoulder pulley is on the upper arm's shaft (hub in a 6002 bearing); the elbow pulley turns a
crank riding on that shaft whose rod runs parallel to the upper arm and pushes the forearm's heel
(parallelogram S-C-D-E), so the forearm angle is set by the elbow motor independently of the
shoulder. Two more parallelograms (F-G through the elbow triangle, H-K to the wrist) keep the
wrist level; an SG90 on the wrist rolls a two-finger gripper holding a 200 g payload.

Arm hinges turn about -Y so +value raises the link; j_shoulder/j_elbow values are the absolute
upper-arm / forearm angles from +X (deg). Studies drive the motors (4x the arm angles).
"""
import math
import numpy as np
from build123d import *
from mech import *
from mech.geom import link, polygon
from mech.parts import bearing, gt2_pulley, nema17, rod, sg90, socket_head_screw

HINGE = (0, -1, 0)                          # arm hinge axis: right-hand about -Y raises the link
Y = (0, 1, 0)
rp = lambda teeth: teeth * 2 / math.pi / 2  # GT2 pitch radius (mm)


def ycyl(p, y0, y1, r):
    """Cylinder of radius r along Y from y0 to y1, centred on the (x, z) point p."""
    return Pos(float(p[0]), (y0 + y1) / 2, float(p[1])) * Rot(90, 0, 0) * Cylinder(r, y1 - y0)


def along_y(p, y0, libpart):
    """A Z-axis library part (pulley, bearing, motor) re-axed along +Y, its z=0 face at y0."""
    return Pos(float(p[0]), y0, float(p[1])) * Rot(-90, 0, 0) * libpart


def xz_part(sketch, y0, y1):
    """Extrude a 2D sketch drawn in (x, z) coordinates into the slab y0..y1."""
    plane = Plane(origin=(0, y1, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))
    return plane.location * extrude(sketch, y1 - y0)


def disc(p, r):
    return Pos(float(p[0]), float(p[1])) * Circle(r)


def pin_screw(p, y_head, length, toward):
    """M5 socket head screw as a hinge pin: head underside at y_head, shank along ±Y."""
    return Pos(float(p[0]), y_head, float(p[1])) * Rot(90 * toward, 0, 0) * socket_head_screw("M5", length)


def belt_band(p1, r1, p2, r2, n, width, t_in=0.75, t_out=0.63):
    """Closed GT2 belt around two pulleys with parallel axes along n: pitch circles (p1, r1) and
    (p2, r2); the band extends `width` along n from the plane of p1, p2."""
    p1, p2, n = (np.asarray(v, dtype=float) for v in (p1, p2, n))
    d = float(np.linalg.norm(p2 - p1))

    def hull(a, b):                                   # convex hull of two circles on the local X axis
        s = (b - a) / d
        c = math.sqrt(1 - s * s)
        q1, q2 = (-a * s, a * c), (d - b * s, b * c)
        return Circle(a) + Pos(d, 0) * Circle(b) + polygon([q1, (q1[0], -q1[1]), (q2[0], -q2[1]), q2])

    sk = hull(r1 + t_out, r2 + t_out) - hull(r1 - t_in, r2 - t_in)
    plane = Plane(origin=tuple(p1), x_dir=tuple((p2 - p1) / d), z_dir=tuple(n))
    return plane.location * extrude(sk, width)


def tool_reach(report) -> float:
    """Largest horizontal distance (mm) of the tool centre from the yaw axis over all studies."""
    return max(math.hypot(x, y) for s in report["studies"]
               for ok, (x, y, _) in zip(s["series"]["ok"], s["series"]["probes"]["tool"]) if ok)


def tool_low(report) -> float:
    """Lowest tool-centre height (mm) over all studies."""
    return min(p[2] for s in report["studies"]
               for ok, p in zip(s["series"]["ok"], s["series"]["probes"]["tool"]) if ok)


def build(L1=150.0, L2=160.0, a=40.0, b=35.0, th1=65.0, th2=-25.0, zS=115.0, t=5.0,
          belt_cd=72.5, phi_s=262.0, phi_e=220.0, payload_g=200.0,
          nema_nm=0.4, roll_nm=0.176) -> Assembly:
    asm = Assembly("desktop_arm", clearance=0.3)                # the 6002 bores encircle the hubs: carried
    # ---- planar linkage geometry in (x, z) at the home pose -------------------------------------
    c1, s1 = math.cos(math.radians(th1)), math.sin(math.radians(th1))
    c2, s2 = math.cos(math.radians(th2)), math.sin(math.radians(th2))
    u1, u2 = np.array([c1, s1]), np.array([c2, s2])
    S = np.array([0.0, zS])
    E, C = S + L1 * u1, S - a * u2                    # elbow, elbow-crank tip
    W, D = E + L2 * u2, E - a * u2                    # wrist, forearm heel (S-C-D-E parallelogram)
    f, h = b * np.array([-s1, c1]), b * np.array([-s2, c2])   # leveling offsets, ⊥ to each link
    F, G, H, K = S + f, E + f, E + h, W + h           # F-G ∥ S-E (elbow triangle), H-K ∥ E-W (wrist)
    polar = lambda deg: S + belt_cd * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    Ms, Me = polar(phi_s), polar(phi_e)               # shoulder / elbow motor axes (x, z)
    P = lambda p, y=0.0: (float(p[0]), y, float(p[1]))
    lay = lambda k: k * (t + 1)                       # link layers along Y, 1 mm apart
    yR, yC, yU, yT, yL = (lay(k) for k in (-2, -1, 0, 1, 2))   # drive rod · crank+forearm · upper arm+rod_2 · triangle · rod_1
    yPl, tp = 23.0, 8.0                               # yoke plate inner face and thickness
    yB = yPl + tp / 2                                 # plate bearing mid-plane
    yP = yPl + tp + 1                                 # outboard pulleys: y ∈ ±[yP, yP + 8]
    bar = lambda p, q, w, yc, hole=5.2: link(P(p), P(q), w, t, z=yc - t / 2, hole=hole, normal=Y)

    # ---- ground: desk, base plate + skirt, yaw bearing and motor ---------------------------------
    asm.part("desk", Pos(85, 0, -57.5) * Box(430, 400, 5), ground=True, material="plywood", color="#b48a5c")
    yaw_motor = Pos(-belt_cd, 0, -10) * nema17(40)    # face under the plate, shaft up through it
    base = Pos(-15, 0, -5) * Box(210, 170, 10) - Cylinder(11, 30) - Pos(-belt_cd, 0, 0) * Cylinder(11.25, 30)
    for i in range(1, 5):
        v = yaw_motor.frames[f"hole_{i}"].position
        base -= Pos(v.X, v.Y, 0) * Cylinder(1.7, 30)
        asm.part(f"yaw_screw_{i}", yaw_motor.frames[f"hole_{i}"] * Pos(0, 0, 10) * socket_head_screw("M3", 10),
                 ground=True, material="steel")
    asm.part("base", base, ground=True, material="aluminum_6061", color="#a8adb5")
    asm.part("skirt", Pos(-15, 0, -32.5) * (Box(210, 170, 45) - Box(202, 162, 50)), ground=True,
             material="ABS", color="#23272b")
    asm.part("yaw_bearing", Pos(0, 0, -5) * bearing("608"), ground=True, material="steel")
    asm.part("yaw_motor", yaw_motor, ground=True, color="#303030")
    asm.part("yaw_belt", belt_band((-belt_cd, 0, 4), rp(20), (0, 0, 4), rp(80), (0, 0, 1), 6),
             ground=True, material="rubber", color="#141414")

    # ---- yaw stage: shaft, 80T pulley, turntable ------------------------------------------------
    asm.part("yaw_shaft", Pos(0, 0, -12) * rod(8, 32), material="steel")        # flush with the turntable top
    asm.part("yaw_pulley", Pos(0, 0, 3) * gt2_pulley(80, 6, 8), material="aluminum_6061", color="#c7c7c7")
    turntable = Pos(0, 0, 17) * Cylinder(80, 6) + Pos(0, 0, 12.75) * Cylinder(10, 2.5) - Cylinder(4, 40)
    asm.part("turntable", turntable, material="aluminum_6061", color="#a8adb5")
    asm.part("yaw_pinion", Pos(-belt_cd, 0, 3) * gt2_pulley(20, 6, 5), material="aluminum_6061", color="#c7c7c7")
    asm.revolute("j_yaw", "base", "yaw_shaft", origin=(0, 0, 0), axis=(0, 0, 1), limits=(-135, 135))
    asm.revolute("j_yaw_motor", "yaw_motor", "yaw_pinion", at=yaw_motor.frames["shaft"], limits=(-540, 540))
    asm.belt("j_yaw_motor", "j_yaw", 20, 80)
    for p in ("yaw_pulley", "turntable"):
        asm.fix(p, "yaw_shaft")

    # ---- tower on the turntable: yoke plates with hub bearings, two motors inside, belts outside -
    plate = (polygon([(-74, 20), (18, 20), (18, 55), (26, zS - 15), (-30, zS + 21), (-74, zS - 35)])
             + disc(S, 30) + disc(Me, 31) + disc(F, 10) - disc(S, 16))
    asm.part("plate_r", xz_part(plate - disc(Ms, 11.25), yPl, yPl + tp), material="aluminum_6061", color="#a8adb5")
    asm.part("plate_l", xz_part(plate - disc(Me, 11.25), -yPl - tp, -yPl), material="aluminum_6061", color="#a8adb5")
    asm.part("bearing_s", along_y(S, yB, bearing("6002")), material="steel")    # 15x32x9, centred in the plate
    asm.part("bearing_e", along_y(S, -yB, bearing("6002")), material="steel")
    motor_s = along_y(Ms, yPl, nema17(40))                          # face on the right plate, shaft +Y
    motor_e = Pos(float(Me[0]), -yPl, float(Me[1])) * Rot(90, 0, 0) * nema17(40)   # left plate, shaft -Y
    asm.part("shoulder_motor", motor_s, color="#303030")
    asm.part("elbow_motor", motor_e, color="#303030")
    asm.part("belt_s", belt_band(P(Ms, yP + 1), rp(20), P(S, yP + 1), rp(80), Y, 6), material="rubber", color="#141414")
    asm.part("belt_e", belt_band(P(Me, -yP - 7), rp(20), P(S, -yP - 7), rp(80), Y, 6), material="rubber", color="#141414")
    asm.part("post_F", ycyl(F, yL - t / 2 - 0.5, yPl, 2.5) + ycyl(F, yL + t / 2 + 0.5, yPl, 5), material="steel")
    for p in ("plate_r", "plate_l", "bearing_s", "bearing_e", "shoulder_motor", "elbow_motor", "belt_s", "belt_e", "post_F"):
        asm.fix(p, "turntable")
    asm.part("pinion_s", along_y(Ms, yP, gt2_pulley(20, 6, 5)), material="aluminum_6061", color="#c7c7c7")
    asm.part("pinion_e", along_y(Me, -yP - 8, gt2_pulley(20, 6, 5)), material="aluminum_6061", color="#c7c7c7")
    asm.revolute("j_shoulder_motor", "shoulder_motor", "pinion_s", origin=P(Ms), axis=HINGE)
    asm.revolute("j_elbow_motor", "elbow_motor", "pinion_e", origin=P(Me), axis=HINGE)

    # ---- arm: upper arm + shaft + shoulder pulley, elbow crank + pulley, drive rod, forearm ------
    hub = lambda p, y0, y1: ycyl(p, y0, y1, 7.5) - ycyl(p, y0 - 1, y1 + 1, 4)      # ø15 boss, ø8 bore
    asm.part("upper_arm", bar(S, E, 22, yU, hole=8) + hub(S, yU + t / 2, yB + 4.5), material="aluminum_6061", color="#e8772e")
    asm.part("shoulder_shaft", Pos(0, yP + 10, zS) * Rot(90, 0, 0) * rod(8, 2 * (yP + 10)), material="steel")
    asm.part("pulley_s", along_y(S, yP, gt2_pulley(80, 6, 8)), material="aluminum_6061", color="#c7c7c7")
    asm.part("crank", bar(S, C, 16, yC, hole=8) + hub(S, -yB - 4.5, yC - t / 2), material="aluminum_6061", color="#2a9d8f")
    asm.part("pulley_e", along_y(S, -yP - 8, gt2_pulley(80, 6, 8)), material="aluminum_6061", color="#c7c7c7")
    asm.part("drive_rod", bar(C, D, 12, yR), material="aluminum_6061", color="#2a9d8f")
    asm.part("forearm", bar(D, W, 20, yC) - ycyl(E, yC - t, yC + t, 2.6), material="aluminum_6061", color="#e8772e")
    for p in ("shoulder_shaft", "pulley_s"):
        asm.fix(p, "upper_arm")
    asm.fix("pulley_e", "crank")
    asm.revolute("j_shoulder", "turntable", "upper_arm", origin=P(S), axis=HINGE, home=th1, limits=(15, 110))
    asm.revolute("j_elbow", "turntable", "crank", origin=P(S), axis=HINGE, home=th2, limits=(-75, 15))
    asm.revolute("j_rod", "crank", "drive_rod", origin=P(C), axis=HINGE)
    asm.revolute("j_knee", "upper_arm", "forearm", origin=P(E), axis=HINGE)
    asm.pin("p_D", "drive_rod", "forearm", point=P(D), axis=HINGE)         # closes S-C-D-E
    asm.belt("j_shoulder_motor", "j_shoulder", 20, 80)
    asm.belt("j_elbow_motor", "j_elbow", 20, 80)
    for p in ("crank", "pulley_e"):                    # bushing on the arm's shaft: two hinges apart on one axis
        asm.joined(p, "shoulder_shaft")                # (the hubs' nominal bearing seats are carried by their hinges)

    # ---- leveling linkage: elbow triangle, rod_1 (F-G), rod_2 (H-K), wrist -----------------------
    tri = polygon([E, G, H])
    for p in (E, G, H):
        tri += disc(p, 7)
    for p in (E, G, H):
        tri -= disc(p, 2.6)
    asm.part("triangle", xz_part(tri, yT - t / 2, yT + t / 2), material="aluminum_6061", color="#3d5a80")
    asm.part("rod_1", bar(F, G, 12, yL), material="aluminum_6061", color="#3d5a80")
    asm.part("rod_2", bar(H, K, 12, yU), material="aluminum_6061", color="#3d5a80")
    zsv = W[1] - 35.0                                 # servo mounting plane (shaft down)
    yB0, yB1 = yT + t / 2 + 0.5, yT + t / 2 + 0.5 + t  # gripper bracket layer
    wrist = (ycyl(W, yC + t / 2 + 0.5, yB1, 7) + bar(W, K, 14, yT)
             + Pos(W[0], (yB0 + yB1) / 2, (zsv + 3 + W[1]) / 2) * Box(14, t, W[1] - zsv - 3)
             + Pos(W[0] - 5, 0, zsv + 1.5) * Box(40, 28, 3) - Pos(W[0] - 5.2, 0, zsv + 1.5) * Box(23, 12.4, 10)
             - ycyl(W, yC, yB1 + 1, 2.6))
    asm.part("wrist", wrist, material="aluminum_6061", color="#e8772e")
    servo = Pos(W[0], 0, zsv) * Rot(180, 0, 0) * sg90()
    asm.part("servo", servo, color="#303030")
    asm.fix("servo", "wrist")
    asm.revolute("j_tri", "upper_arm", "triangle", origin=P(E), axis=HINGE)
    asm.revolute("j_rod1", "turntable", "rod_1", origin=P(F), axis=HINGE)
    asm.pin("p_G", "rod_1", "triangle", point=P(G), axis=HINGE)            # closes S-E-G-F
    asm.revolute("j_rod2", "triangle", "rod_2", origin=P(H), axis=HINGE)
    asm.revolute("j_wrist", "forearm", "wrist", origin=P(W), axis=HINGE)
    asm.pin("p_K", "rod_2", "wrist", point=P(K), axis=HINGE)               # closes E-W-K-H

    # ---- gripper on the servo horn, payload between the fingers ---------------------------------
    gx, gz = W[0], zsv - 14                           # horn seat (spline tip)
    gripper = (Pos(gx, 0, gz + 1.6) * Cylinder(10, 3.2) - Pos(gx, 0, gz + 1.6) * Cylinder(2.4, 3.2)
               + Pos(gx, 0, gz - 2) * Box(60, 14, 4)
               + Pos(gx - 14.5, 0, gz - 19) * Box(4, 14, 30) + Pos(gx + 14.5, 0, gz - 19) * Box(4, 14, 30))
    asm.part("gripper", gripper, material="PLA", color="#454545")
    tool = (gx, 0.0, gz - 19.5)
    asm.part("payload", Pos(*tool) * Box(25, 25, 25), material="steel", mass_g=payload_g, color="#c0392b")
    asm.fix("payload", "gripper")
    asm.revolute("j_roll", "wrist", "gripper", at=servo.frames["horn"], limits=(-90, 90))
    asm.probe("tool", part="payload", point=tool)

    # ---- hinge pins (M5 socket head screws) ----------------------------------------------------
    for name, parent, p, y_head, length, toward in (
            ("pin_E", "upper_arm", E, yC - t / 2, 18, +1), ("pin_W", "wrist", W, yC - t / 2, 22, +1),
            ("pin_C", "crank", C, yR - t / 2, 12, +1), ("pin_D", "forearm", D, yR - t / 2, 12, +1),
            ("pin_G", "triangle", G, yL + t / 2, 12, -1), ("pin_H", "triangle", H, yU - t / 2, 10, +1),
            ("pin_K", "wrist", K, yT + t / 2, 12, -1)):
        asm.part(name, pin_screw(p, y_head, length, toward), material="steel")
        asm.fix(name, parent)

    # ---- intent --------------------------------------------------------------------------------
    for belt, pulleys in (("yaw_belt", ("yaw_pulley", "yaw_pinion")), ("belt_s", ("pulley_s", "pinion_s")),
                          ("belt_e", ("pulley_e", "pinion_e"))):
        for p in pulleys:
            asm.allow_contact(belt, p, max_depth=None)
    for j in ("j_yaw_motor", "j_shoulder_motor", "j_elbow_motor"):
        asm.actuator(j, capacity=nema_nm)                  # NEMA17 40 mm holding torque
    asm.actuator("j_roll", capacity=roll_nm)               # SG90 stall torque
    ms, me = (lambda th: 4 * (th - th1)), (lambda th: 4 * (th - th2))   # motor angle for an arm angle
    asm.study("yaw", drive={"j_yaw_motor": (-540, 540)}, frames=25)
    asm.study("shoulder", drive={"j_shoulder_motor": (ms(15), ms(110))}, frames=39)
    asm.study("elbow", drive={"j_elbow_motor": (me(-75), me(15))}, frames=37)
    keys = [(0.0, 80, -10), (0.15, 25, -42), (0.25, 20, -58.5), (0.35, 45, -35),  # (u, shoulder°, forearm°)
            (0.65, 45, -35), (0.75, 20, -58.5), (0.85, 45, -35), (1.0, 80, -10)]
    asm.study("pick_place", drive={"j_shoulder_motor": [(u, ms(s)) for u, s, _ in keys],
                                   "j_elbow_motor": [(u, me(e)) for u, _, e in keys],
                                   "j_yaw_motor": [(0, 0), (0.15, -180), (0.35, -180), (0.65, 180), (0.85, 180), (1.0, 0)],
                                   "j_roll": [(0.35, 0), (0.65, 90), (0.85, 90), (1.0, 0)]}, frames=81)
    asm.study("roll", drive={"j_roll": (-90, 90)}, frames=13)
    asm.target("reach", tool_reach, min=280)
    asm.target("reaches desk", tool_low, max=-35)
    asm.target("wrist level", "angle:turntable,wrist", max=0.5)
    asm.target("yaw range", "span:j_yaw", min=270)          # span: the largest sweep of any study
    asm.target("shoulder motor SF", "sf:j_shoulder_motor", min=2)
    asm.target("elbow motor SF", "sf:j_elbow_motor", min=2)
    asm.target("no collision", "clearance", min=0.3)
    return asm


if __name__ == "__main__":
    run(build())
