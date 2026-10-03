"""Box with two lids hinged on opposite top edges, closing to meet with `gap` in the middle."""
from build123d import *
from mech import *

def build(width=80.0, depth=50.0, height=30.0, wall=2.0, gap=0.2) -> Assembly:
    asm = Assembly("hinged_box", clearance=0.3)
    body = Pos(0, 0, height / 2) * Box(width, depth, height)
    body -= Pos(0, 0, height / 2 + wall) * Box(width - 2 * wall, depth - 2 * wall, height)
    lid = (width - gap) / 2                                   # each lid's length along X
    asm.part("body", body, ground=True, color="#666666")
    asm.part("lid_left", Pos(-width / 2 + lid / 2, 0, height + wall / 2) * Box(lid, depth, wall))
    asm.part("lid_right", Pos(width / 2 - lid / 2, 0, height + wall / 2) * Box(lid, depth, wall))
    # hinge lines on the outer top edges; each axis chosen so +θ lifts its lid
    asm.revolute("j_left", "body", "lid_left", origin=(-width / 2, 0, height), axis=(0, -1, 0),
                 limits=(0, 120))
    asm.revolute("j_right", "body", "lid_right", origin=(width / 2, 0, height), axis=(0, 1, 0),
                 limits=(0, 120))
    asm.study("open", drive={"j_left": (0, 110), "j_right": (0, 110)}, frames=23)
    return asm

if __name__ == "__main__":
    run(build())
