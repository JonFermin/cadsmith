"""Crank-rocker four-bar linkage."""
from build123d import *
from mech import *
from mech.geom import link, circle_intersect

def build(ground=100.0, crank=40.0, coupler=90.0, rocker=80.0, t=5.0) -> Assembly:
    asm = Assembly("four_bar", clearance=0.3)
    O2, O4 = (0, 0, 0), (ground, 0, 0)
    A = (crank, 0, 0)                                          # crank drawn at 0°
    B = circle_intersect(A, coupler, O4, rocker, side=+1)      # closes the loop in the drawing
    asm.part("frame", Pos(ground / 2, 0, -2 * t) * Box(ground + 20, 16, t), ground=True, color="#666666")
    asm.part("crank", link(O2, A, width=10, thickness=t, z=0), material="aluminum_6061")
    asm.part("coupler", link(A, B, width=10, thickness=t, z=t + 0.5), material="aluminum_6061")
    asm.part("rocker", link(O4, B, width=10, thickness=t, z=2 * t + 1), material="aluminum_6061")
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1))          # no limits: full turn
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1))
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))                 # closes the loop
    asm.probe("mid", part="coupler", point=[(a + b) / 2 for a, b in zip(A, B)])
    asm.actuator("j_crank", capacity=0.5)
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=72)
    asm.target("rocker swing", "span:j_rocker", min=40)
    return asm

if __name__ == "__main__":
    run(build())
