"""Servo arm swinging a payload in the XZ plane: θ from horizontal, +θ raises the arm."""
from build123d import *
from mech import *
from mech.parts import mg996r

def build(length=120.0, payload_g=100.0, capacity=1.08) -> Assembly:
    asm = Assembly("pendulum_arm", clearance=0.3)
    servo = Rot(90, 0, 0) * mg996r()                        # shaft along −Y at the origin
    y = servo.shape.bounding_box().min.Y - 3                # arm mid-plane, on the spline face
    asm.part("servo", servo, ground=True, color="#333333")
    asm.part("arm", Pos(length / 2 - 9, y, 0) * Box(length - 2, 6, 12), material="aluminum_6061")
    asm.part("payload", Pos(length, y, 0) * Box(20, 20, 20), material="steel", mass_g=payload_g)
    asm.fix("payload", "arm")
    asm.revolute("j_shoulder", "servo", "arm", at=servo.frames["shaft"], limits=(-90, 90))
    asm.probe("tip", part="payload", point=(length + 10, y, 0))
    asm.actuator("j_shoulder", capacity=capacity)           # MG996R stall torque, N·m
    asm.study("swing", drive={"j_shoulder": (-90, 90)}, frames=37)  # 5° steps
    asm.target("servo margin", "sf:j_shoulder", min=2)
    return asm

if __name__ == "__main__":
    run(build())
