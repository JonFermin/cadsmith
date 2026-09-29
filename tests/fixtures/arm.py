"""CLI test model: a bar swinging about a vertical axis above a base plate.

Parameters steer the outcome: ``need`` > ``swing`` misses the target (FAIL, or WARN with
``severity="WARN"``), ``bad_joint=True`` names an unknown parent (INVALID), and ``bump=True``
puts a block in the arm's path (interference FAIL).
"""
from build123d import Box, Pos

from mech import Assembly, run


def build(swing=90.0, need=45.0, severity="FAIL", bad_joint=False, bump=False, label="arm") -> Assembly:
    asm = Assembly(label, clearance=0.3)
    asm.part("base", Pos(0, 0, -3) * Box(80, 80, 4), ground=True)
    asm.part("arm", Pos(20, 0, 2) * Box(40, 6, 4), material="aluminum_6061")
    asm.revolute("j_arm", "base" if not bad_joint else "bsae", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    if bump:
        # a second moving part in the arm's path at 45°, held at home by its own slider
        asm.part("block", Pos(21, 21, 2) * Box(6, 6, 4))
        asm.prismatic("j_block", "base", "block", origin=(21, 21, 0), axis=(0, 0, 1), limits=(0, 1))
    asm.probe("tip", part="arm", point=(40, 0, 2))
    asm.study("swing", drive={"j_arm": (0, swing)}, frames=10)
    asm.target("swing", "span:j_arm", min=need, severity=severity)
    return asm


if __name__ == "__main__":
    run(build())
