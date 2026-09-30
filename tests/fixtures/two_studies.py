"""CLI test model with two studies: an arm swinging about +Z hits a post at 45° in study 'sweep'
(0→90°, 5° steps; the hit spans ≈ 37–53°) but never in study 'back' (0→−90°). A 4-frame
sampling of 'sweep' (0, 30, 60, 90°) samples no frame inside the hit (the sweep's between-frame
check finds it at 45°).
"""
import math

from build123d import Box, Pos

from mech import Assembly, run


def build(post_r=30.0) -> Assembly:
    asm = Assembly("two", clearance=0.3)
    c = post_r / math.sqrt(2)
    asm.part("base", Pos(0, 0, -2) * Box(100, 100, 4), ground=True)
    asm.part("post", Pos(c, c, 10) * Box(6, 6, 20), ground=True)
    asm.part("arm", Pos(20, 0, 10) * Box(40, 4, 4))
    asm.revolute("j", "base", "arm", origin=(0, 0, 10), axis=(0, 0, 1), limits=(-180, 180))
    asm.study("sweep", drive={"j": (0, 90)}, frames=19)
    asm.study("back", drive={"j": (0, -90)}, frames=19)
    return asm


if __name__ == "__main__":
    run(build())
