"""Zero-offset slider-crank: the slider's x equals its joint value (home = r + l)."""
from build123d import *
from mech import *
from mech.geom import link

def build(r=30.0, l=90.0, t=5.0) -> Assembly:
    asm = Assembly("slider_crank", clearance=0.3)
    O, A, C = (0, 0, 0), (r, 0, 0), (r + l, 0, 0)             # crank drawn at 0°, slider at x = r + l
    asm.part("frame", Pos((r + l) / 2, 0, -t) * Box(r + l + 80, 20, t), ground=True, color="#666666")
    asm.part("crank", link(O, A, width=10, thickness=t, z=0), material="aluminum_6061")
    asm.part("rod", link(A, C, width=10, thickness=t, z=t + 0.5), material="aluminum_6061")
    asm.part("slider", Pos(r + l, 0, t / 4) * Box(20, 12, 1.5 * t), material="POM")  # rides on the frame
    asm.revolute("j_crank", "frame", "crank", origin=O, axis=(0, 0, 1))
    asm.revolute("j_rod", "crank", "rod", origin=A, axis=(0, 0, 1))
    asm.prismatic("j_slider", "frame", "slider", origin=C, axis=(1, 0, 0), home=r + l)
    asm.pin("p_C", "rod", "slider", point=C, axis=(0, 0, 1))  # wrist pin closes the loop
    asm.probe("wrist", part="slider", point=C)
    asm.actuator("j_crank", capacity=0.5)
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=73)  # 5° steps
    asm.target("stroke", "span:j_slider", min=55)
    return asm

if __name__ == "__main__":
    run(build())
