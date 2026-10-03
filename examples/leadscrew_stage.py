"""Vertical T8 lead-screw stage: the carriage rides two LM8UU bushings on ø8 rods."""
from build123d import *
from mech import *
from mech.parts import T8_LEAD, bearing, rod, t8_leadscrew, t8_nut

def build(span=50.0, height=160.0, z0=20.0, turns=10) -> Assembly:
    asm = Assembly("leadscrew_stage", clearance=0.3)
    xs = (-span / 2, span / 2)                                  # rod positions; screw on the Z axis
    asm.part("base", Pos(0, 0, -5) * Box(span + 30, 36, 10), ground=True, color="#666666")
    for i, x in enumerate(xs):
        asm.part(f"rod{i}", Pos(x, 0, 0) * rod(8, height), ground=True)
    asm.part("screw", Pos(0, 0, 2) * t8_leadscrew(height - 10), material="steel")
    block = Pos(0, 0, z0) * Box(span + 24, 30, 24)              # carriage block, bottom at z0 - 12
    block -= Pos(0, 0, z0) * (Cylinder(5.5, 30) + Pos(xs[0], 0, 0) * Cylinder(7.5, 30)
                              + Pos(xs[1], 0, 0) * Cylinder(7.5, 30))
    asm.part("carriage", block, material="PETG")
    asm.part("nut", Pos(0, 0, z0 + 12) * t8_nut(), material="brass")  # flange on the carriage top
    for i, x in enumerate(xs):
        asm.part(f"bush{i}", Pos(x, 0, z0) * bearing("LM8UU"), material="steel")
        asm.fix(f"bush{i}", "carriage")
    asm.fix("nut", "carriage")
    asm.revolute("j_leadrot", "base", "screw", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_carriage", "base", "carriage", origin=(0, 0, z0), axis=(0, 0, 1),
                  limits=(0, height - 45))
    asm.screw("j_leadrot", "j_carriage", lead=T8_LEAD)          # right hand: +360° moves −8 mm
    asm.allow_contact("screw", "nut")                           # nominal ø8 bore on a ø8 screw
    asm.actuator("j_leadrot", capacity=0.3)                     # NEMA 17 through a coupler, N·m
    asm.study("lift", drive={"j_leadrot": (0, -360 * turns)}, frames=61)
    asm.target("travel", "max:j_carriage", min=75)
    return asm

if __name__ == "__main__":
    run(build())
