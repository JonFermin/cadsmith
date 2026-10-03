"""Parallel-jaw gripper: an SG90 turns a gear pair; each jaw rides a parallelogram, so it translates."""
import math
from build123d import *
from mech import *
from mech.geom import link
from mech.parts import gear_pair, sg90

def jaw_tilt(report) -> float:
    """Largest rotation (deg) of jaw_l about Z over the stroke: 0 while the jaws stay parallel."""
    Ts = report["studies"][0]["series"]["transforms"]["jaw_l"]
    return max(abs(math.degrees(math.atan2(T[1][0], T[0][0]))) for T in Ts)

def build(L=35.0, s=16.0, open_deg=120.0, gap_open=30.0, t=4.0, w=6.0, module=1.0, teeth=20) -> Assembly:
    asm = Assembly("parallel_gripper", clearance=0.3)
    gp = gear_pair(module, teeth, teeth, t)                  # g2 at (a, 0, 0), ratio −1
    a = gp.center_distance
    f = a / 2 - L * math.cos(math.radians(open_deg)) - gap_open / 2   # jaw face inset from the bar end
    closed = math.degrees(math.acos((a / 2 - f) / L))        # drawn closed: both faces on x = a/2
    zp, zg = 11.0, 14.5                                       # plate, gear/driver layer (spline top z=14)
    zf, zj = zg + t + 0.5, zg + 2 * t + 1.0                   # follower, jaw layers
    servo = sg90()                                            # ears at z=0 (frames["shaft"]), spline to z=14
    plate = Pos(a / 2, 2, zp + 1.5) * Box(a + 2 * s + 12, 24, 3) - Pos(0, 0, zp) * Cylinder(3.2, 20)
    for x in (-s, a + s):                                     # follower posts
        plate += Pos(x, 0, (zp + 3 + zf - 0.2) / 2) * Cylinder(2, zf - 0.2 - zp - 3)
    asm.part("servo", servo, ground=True, color="#333333")
    asm.part("plate", plate, ground=True, color="#666666")
    for k, ox, ang, fx, sg, gear in (("l", 0.0, closed, -s, 1, gp.g1), ("r", a, 180 - closed, a + s, -1, gp.g2)):
        A = (ox + L * math.cos(math.radians(ang)), L * math.sin(math.radians(ang)), 0.0)
        Af, face, x0 = (fx + A[0] - ox, A[1], 0.0), A[0] + sg * f, A[0] - sg * 4
        asm.part(f"driver_{k}", [Pos(0, 0, zg) * gear.shape, link((ox, 0, 0), A, width=w, thickness=t, z=zg)])
        asm.part(f"follower_{k}", link((fx, 0, 0), Af, width=w, thickness=3, z=zf))
        asm.part(f"jaw_{k}", [link(Af, A, width=8, thickness=3, z=zj),
                              Pos((x0 + face) / 2, A[1] + 11, zj + 1.5) * Box(abs(face - x0), 30, 3),
                              Pos(face - sg * 2, A[1] + 16, (zg + zj + 3) / 2) * Box(4, 20, zj + 3 - zg)])
        if k == "l":
            asm.revolute("j_drive", "servo", "driver_l", at=servo.frames["shaft"], limits=(0, 30))
        else:
            asm.revolute("j_idler", "plate", "driver_r", origin=(a, 0, 0), axis=(0, 0, 1))
        asm.revolute(f"j_jaw_{k}", f"driver_{k}", f"jaw_{k}", origin=A, axis=(0, 0, 1))
        asm.revolute(f"j_fol_{k}", "plate", f"follower_{k}", origin=(fx, 0, 0), axis=(0, 0, 1))
        asm.pin(f"p_{k}", f"follower_{k}", f"jaw_{k}", point=Af, axis=(0, 0, 1))  # closes the parallelogram
        asm.probe(f"pad_{k}", part=f"jaw_{k}", point=(face, A[1] + 16, zg + 4))
    asm.gear("j_drive", "j_idler", gp.ratio)
    asm.allow_contact("jaw_l", "jaw_r")                       # the jaws close on each other
    asm.actuator("j_drive", capacity=0.176)                   # SG90 stall torque 1.8 kg·cm
    asm.study("stroke", drive={"j_drive": (0, open_deg - closed)}, frames=40)
    asm.target("opening", "max_dist:pad_l,pad_r", min=gap_open)
    asm.target("closes", "min_dist:pad_l,pad_r", max=0.5)
    asm.target("jaws parallel", jaw_tilt, max=0.01)
    return asm

if __name__ == "__main__":
    run(build())
