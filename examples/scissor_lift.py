"""Two-stage scissor lift: a NEMA17 turns a T8 lead screw that pulls the bottom sliding pivot.

Arms in the XZ plane pivot about Y; drawn at the lowest pose (theta0). +j_slide = toward the
fixed pivot = raise, so the screw is driven negative (right hand: +360° → −8 mm).
"""
import math
from build123d import *
from mech import *
from mech.geom import link
from mech.parts import T8_LEAD, nema17, t8_leadscrew, t8_nut

def build(L=120.0, theta0=10.0, theta1=40.0, t=4.0, payload_g=1000.0, capacity=0.4) -> Assembly:
    asm = Assembly("scissor_lift", clearance=0.3)
    s, h = L * math.cos(math.radians(theta0)), L * math.sin(math.radians(theta0))
    stroke = s - L * math.cos(math.radians(theta1))
    ly = lambda k: k * (t + 0.5)                             # Y layer of arm k
    P = lambda x, z: (x, 0.0, z)
    ys, zs, xm = 25.0, 16.0, s + 45                           # screw axis (y, z) and motor face x
    base = Pos(s / 2 + 15, 12, -9) * Box(s + 110, 50, 6) + Pos(0, ly(-1) + t / 2, -3) * Box(12, t, 12)
    asm.part("base", base, ground=True, color="#666666")
    asm.part("motor", Pos(xm, ys, zs) * Rot(0, -90, 0) * nema17(), ground=True, color="#333333")
    asm.part("screw", Pos(xm - 26, ys, zs) * Rot(0, -90, 0) * t8_leadscrew(s + 10), material="steel")
    blk = Pos(s, (ly(2) + ys + 10) / 2, (zs + 11) / 2 - 2.75) * Box(16, ys + 10 - ly(2), zs + 16.5)
    asm.part("slider", blk - Pos(s, ys, zs) * Rot(0, 90, 0) * Cylinder(5, 40), material="PETG")
    asm.part("nut", Pos(s + 8, ys, zs) * Rot(0, -90, 0) * t8_nut(), material="brass")
    asm.fix("nut", "slider")
    for name, a, b, k in (("arm1", P(0, 0), P(s, h), 0), ("arm2", P(s, 0), P(0, h), 1),
                          ("arm3", P(0, h), P(s, 2 * h), 0), ("arm4", P(s, h), P(0, 2 * h), -1)):
        asm.part(name, link(a, b, width=10, thickness=t, z=ly(k), normal=(0, 1, 0)), material="aluminum_6061")
    plat = Pos(s / 2, 0, 2 * h + 8) * Box(150, 40, 4) + Pos(0, ly(-2) + t / 2, 2 * h + 1) * Box(12, t, 14)
    asm.part("platform", plat, material="aluminum_6061")
    asm.part("top_slider", Pos(s, ly(1) + t / 2, 2 * h + 1) * Box(12, t, 10), material="POM")
    asm.part("payload", Pos(s / 2, 0, 2 * h + 20) * Box(60, 30, 20), material="steel", mass_g=payload_g)
    asm.fix("payload", "platform")
    Y = (0, 1, 0)
    asm.revolute("j_leadrot", "base", "screw", origin=(xm, ys, zs), axis=(-1, 0, 0))
    asm.prismatic("j_slide", "base", "slider", origin=(s, ys, zs), axis=(-1, 0, 0), limits=(0, stroke))
    asm.screw("j_leadrot", "j_slide", lead=T8_LEAD)
    asm.revolute("j_a1", "base", "arm1", origin=P(0, 0), axis=Y)
    asm.revolute("j_a2", "slider", "arm2", origin=P(s, 0), axis=Y)
    asm.pin("p_x1", "arm1", "arm2", point=P(s / 2, h / 2), axis=Y)       # scissor crossings
    asm.revolute("j_a3", "arm2", "arm3", origin=P(0, h), axis=Y)
    asm.revolute("j_a4", "arm1", "arm4", origin=P(s, h), axis=Y)
    asm.pin("p_x2", "arm3", "arm4", point=P(s / 2, 1.5 * h), axis=Y)
    asm.revolute("j_plat", "arm4", "platform", origin=P(0, 2 * h), axis=Y)
    asm.prismatic("j_top", "platform", "top_slider", origin=P(s, 2 * h), axis=(1, 0, 0))
    asm.pin("p_top", "arm3", "top_slider", point=P(s, 2 * h), axis=Y)
    asm.allow_contact("screw", "nut")
    asm.probe("deck", part="platform", point=P(s / 2, 2 * h + 10))
    asm.actuator("j_leadrot", capacity=capacity)             # NEMA17 holding torque, N·m
    asm.study("lift", drive={"j_leadrot": (0, -360 * stroke / T8_LEAD)}, frames=61)
    asm.target("travel", "delta:deck.z", min=80)
    asm.target("motor SF", "sf:j_leadrot", min=2)
    asm.target("no collision", "clearance", min=0.3)
    return asm

if __name__ == "__main__":
    run(build())
