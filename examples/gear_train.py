"""2:1 reduction: NEMA 17 pinion driving an output gear on a shaft in a 625 bearing."""
from build123d import *
from mech import *
from mech.parts import bearing, gear_pair, nema17, rod

def build(module=1.0, z1=15, z2=30, width=6.0, plate=5.0) -> Assembly:
    asm = Assembly("gear_train", clearance=0.3)
    gp = gear_pair(module, z1, z2, width, bore1=5, bore2=5)   # g2 already at (a, 0, 0), meshed
    a, zg = gp.center_distance, plate + 3                     # gears 3 mm above the plate
    motor = nema17()                                          # face z=0, shaft +Z at the origin
    brg = Pos(a, 0, plate / 2) * bearing("625")               # 5x16x5, pressed into the plate
    base = Pos(a / 2, 0, plate / 2) * Box(a + 50, 50, plate)
    base -= Cylinder(11.5, 3 * plate) + Pos(a, 0, 0) * Cylinder(8, 3 * plate)  # pilot + bearing seat
    asm.part("plate", base, ground=True, color="#666666")
    asm.part("motor", motor, ground=True, color="#333333")
    asm.part("bearing", brg, ground=True, material="steel")
    asm.part("pinion", Pos(0, 0, zg) * gp.g1, material="POM")
    asm.part("gear_out", Pos(0, 0, zg) * gp.g2, material="POM")
    asm.part("shaft_out", Pos(a, 0, 1) * rod(5, zg + width + 6), material="steel")
    asm.revolute("j_in", "motor", "pinion", at=motor.frames["shaft"])
    asm.revolute("j_out", "plate", "shaft_out", at=brg.frames["center"])
    asm.fix("gear_out", "shaft_out")
    asm.gear("j_in", "j_out", gp.ratio)                       # −z1/z2: both axes +Z, external mesh
    asm.actuator("j_in", capacity=0.4)                        # NEMA 17 holding torque, N·m
    asm.study("turn", drive={"j_in": (0, 720)}, frames=73)
    asm.target("light", "mass_g", max=400)
    return asm

if __name__ == "__main__":
    run(build())
