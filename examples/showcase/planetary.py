"""Two-stage 25:1 planetary gearbox on a NEMA 17, holding a lever load.

Sun z=12 on the motor shaft, three z=18 planets on a carrier, a FIXED internal ring z=48
(= zs + 2·zp). Each stage reduces 1 + zr/zs = 5:1; the stage-1 carrier carries the stage-2 sun
and the stage-2 carrier is the output, running in a 6001 bearing in the front plate: 25:1 overall.
The ring is `mech.parts.internal_gear` — the exact conjugate of the planets' library involutes,
its tooth tips trimmed against involute interference with the z=18 planets (`pinion=`) — as two
hardened inserts (one per stage) pressed into aluminium housing plates, with spacer rings between.
Planet phasing is derived from the sun mesh (gear_pair logic) and the ring phase from planet 0;
every planet is checked against both meshes before anything is built.

The axis is world +X (motor at −X, output flange at +X) and horizontal, so a lever with a payload
on the output flange loads the whole train: the sun holding torque is the output torque / 25.
With a fixed ring the kinematics are linear in the sun angle, expressed as couplings:
    carrier = sun · zs/(zs+zr)          planet (relative to its carrier) = −(zs/zp)·(zr/(zs+zr)) · sun
Ring/planet pairs cannot be a gear() mesh (the ring is ground, not a revolute child), so they are
declared `asm.mesh(ring, planet)`: contact allowed, any tooth overlap is still an interference.
"""
import math

import numpy as np
from build123d import *
from mech import *
from mech.parts import bearing, internal_gear, nema17, socket_head_screw, spur_gear


# ------------------------------------------------------------------ prism helpers
def _circ(d, x=0.0, y=0.0):
    return Wire([Edge.make_circle(d / 2, Plane(origin=(x, y, 0), z_dir=(0, 0, 1)))])


def prism(outer, inners, z0, h):
    """Extrude a face (outer wire minus inner wires, drawn at z=0) from z0 up by h — no booleans."""
    return Solid.extrude(Face(outer, list(inners)), (0, 0, h)).moved(Pos(0, 0, z0))


def rounded_square(side, radius, extra=None):
    """Outer wire of a rounded square, optionally unioned with another sketch (the stand's leg)."""
    sk = RectangleRounded(side, side, radius)
    if extra is not None:
        sk = sk + extra
    return sk.faces()[0].outer_wire()


def planet_phase(zs, zp, phi_deg):
    """Home rotation (deg) of a planet at carrier angle `phi` so it meshes a sun whose tooth is on +X."""
    return 180 + 180 / zp + phi_deg * (zs + zp) / zp


def ring_phase_for(zs, zp, zr, phi_deg):
    """Ring-space center angle (deg) demanded by the planet at `phi`: its outward-pointing tooth,
    offset e from the pitch point along the planet pitch circle, needs a ring space offset e·zp/zr."""
    pitch = 360 / zp
    e = (phi_deg - planet_phase(zs, zp, phi_deg) + pitch / 2) % pitch - pitch / 2
    return phi_deg + e * zp / zr


def _turn_x(T):
    """Rotation (deg) of a 4x4 transform about world +X (all axes here are +X)."""
    return math.degrees(math.atan2(T[2][1], T[1][1]))


def _spin(report, part):
    """Unwrapped rotation of `part` over the first study, per degree of sun rotation."""
    tr = report["studies"][0]["series"]["transforms"]
    sun = np.unwrap(np.radians([_turn_x(T) for T in tr["sun1"]]))
    out = np.unwrap(np.radians([_turn_x(T) for T in tr[part]]))
    return (out[-1] - out[0]) / (sun[-1] - sun[0])


def output_ratio(report):
    """Measured reduction sun/output from the part transforms (independent of the coupling table)."""
    return 1.0 / _spin(report, "carrier2")


def planet_spin(report):
    """Absolute planet-1 spin per sun turn from the transforms (expect −(zs/(zs+zr))·(zr/zp − 1))."""
    return _spin(report, "planet1_0")


# ------------------------------------------------------------------------------------ the model
def build(module=1.0, zs=12, zp=18, n_planets=3, width=8.0, backlash=0.1, ring_backlash=0.2, ring_rim=1.5,
          payload_g=2500.0, arm=150.0, capacity=0.4, axis_height=60.0, arm_side=1) -> Assembly:
    m, zr = module, zs + 2 * zp
    if (zs + zr) % n_planets:
        raise ValueError(f"equally spaced planets need (zs + zr)/n integer: ({zs} + {zr})/{n_planets}")
    i_stage, a = 1 + zr / zs, m * (zs + zp) / 2                 # per-stage ratio, planet center radius
    i_total = i_stage ** 2
    k_planet = -(zs / zp) * (zr / (zs + zr))                    # planet spin (rel. carrier) per sun degree
    phis1 = [k * 360 / n_planets for k in range(n_planets)]     # stage-1 planet angles (planet 0 at local +X)
    phis2 = [p + 180 / n_planets for p in phis1]                # stage 2 staggered half a spacing
    ring_phase = ring_phase_for(zs, zp, zr, phis1[0])
    for phi in phis1 + phis2:                                   # every planet must mesh sun AND ring
        if abs((ring_phase_for(zs, zp, zr, phi) - ring_phase + 180 / zr) % (360 / zr) - 180 / zr) > 1e-6:
            raise ValueError(f"planet at {phi:g}° cannot mesh both sun and ring with zs={zs}, zp={zp}, zr={zr}")

    H, side, rad = axis_height, 58.0, 8.0                        # axis height; housing section (rounded square)
    r_tip_ring = m * zr / 2 - m                                  # ring addendum (inner) radius
    rc = r_tip_ring - 1.5                                        # carrier plates clear the ring tips by 1.5
    t_st, t_c1, t_c2, t_fp = 8.0, 7.0, 5.0, 8.0                  # stand, carrier 1 (buries the shaft end), carrier 2, front plate
    z1 = t_st + 1.0                                              # stage-1 gears
    z_c1 = z1 + width + 1.0                                      # carrier-1 plate
    z2 = z_c1 + t_c1 + 1.0                                       # stage-2 planets (sun 2 sits on the carrier face)
    z_c2 = z2 + width + 1.0                                      # carrier-2 plate
    z_h0, z_h1 = t_st, z_c2 + t_c2 + 1.0                         # housing span
    z_r1, z_r2 = z1 + width + 0.5, z2 - 0.5                      # ring insert 1 ends / insert 2 starts (0.5 off the carriers)
    z_fp, z_hub = z_h1, z_h1 + t_fp + 2.0                        # front plate, output hub
    mot_holes = [(sx * 15.5, sy * 15.5) for sx in (1, -1) for sy in (1, -1)]      # NEMA 17: 31 mm square
    bolt_holes = [(sx * 23.5, sy * 23.5) for sx in (1, -1) for sy in (1, -1)]     # M3 through-bolts in the corners

    asm = Assembly("planetary", clearance=0.3)
    W = Pos(0, 0, H) * Rot(0, 90, 0)                             # gearbox frame: local +Z (axial) → world +X
    P = lambda x, y, z: (z, y, H - x)                            # local point → world
    X = (1.0, 0.0, 0.0)                                          # every joint axis
    pin_at = lambda phi: (a * math.cos(math.radians(phi)), a * math.sin(math.radians(phi)))

    # ---- ground: base, stand/adapter plate, ring housing, front plate, bearing, bolts, motor
    asm.part("base", Pos(15, 0, 4) * Box(130, 90, 8), ground=True, color="#3d4046", material="aluminum_6061")
    leg = Pos((H - 8) / 2, 0, 0) * Rectangle(H - 8, side)        # down to the base top (local +X = world −Z)
    outer = rounded_square(side, rad, extra=leg)
    stand = [prism(outer, [_circ(22)] + [_circ(3.4, *h) for h in mot_holes] + [_circ(3, *h) for h in bolt_holes], 0, 2),
             prism(outer, [_circ(12)] + [_circ(3.4, *h) for h in mot_holes] + [_circ(3, *h) for h in bolt_holes], 2, 2),
             prism(outer, [_circ(12)] + [_circ(6, *h) for h in mot_holes] + [_circ(3, *h) for h in bolt_holes], 4, t_st - 4)]
    asm.part("stand", [W * s for s in stand], ground=True, color="#6f757d", material="aluminum_6061")
    # housing = a bolted stack: ring insert in its plate, spacer, ring insert in its plate, spacer
    # (each 0.5 mm off the carriers); the inserts sit in nominal (pressed) bores of the plates
    rs, holes = rounded_square(side, rad), [_circ(3.4, *h) for h in bolt_holes]
    d_ring = m * zr + 2.5 * m + 2 * ring_rim                     # internal_gear OD: root circle + rim
    stack = (("ring1", z_h0, z_r1), ("spacer1", z_r1, z_r2), ("ring2", z_r2, z2 + width + 0.5),
             ("spacer2", z2 + width + 0.5, z_h1))
    for name, za, zb in stack:
        if name.startswith("ring"):
            ring = Rot(0, 0, ring_phase) * internal_gear(m, zr, zb - za, rim=ring_rim, backlash=ring_backlash, pinion=zp)
            asm.part(name, W * (Pos(0, 0, za) * ring), ground=True, color="#9ea4ab", material="steel")
            asm.part(f"housing{name[-1]}", W * prism(rs, [_circ(d_ring)] + holes, za, zb - za), ground=True,
                     color="#3f6aa6", material="aluminum_6061")
        else:
            asm.part(name, W * prism(rs, [_circ(2 * r_tip_ring + 1)] + holes, za, zb - za), ground=True,
                     color="#3f6aa6", material="aluminum_6061")
    brg = bearing("6001")                                        # 12 x 28 x 8: the output bearing
    front = [prism(rs, [_circ(28)] + holes, z_fp, t_fp - 3.5),
             prism(rs, [_circ(28)] + [_circ(6, *h) for h in bolt_holes], z_fp + t_fp - 3.5, 3.5)]
    asm.part("front_plate", [W * s for s in front], ground=True, color="#3f6aa6", material="aluminum_6061")
    asm.part("bearing", W * (Pos(0, 0, z_fp + t_fp / 2) * brg), ground=True, color="#c8ccd1")
    bolts = [Pos(x, y, z_fp + t_fp - 3.5) * socket_head_screw("M3", 40) for x, y in bolt_holes]
    asm.part("bolts", [W * b.shape for b in bolts], ground=True, color="#1e1e22", material="steel",
             bom="4 x ISO 4762 M3x40 socket head cap screw")
    motor = W * nema17()
    asm.part("motor", motor, ground=True, color="#2a2a2e")
    screws = [Pos(x, y, 4.0) * socket_head_screw("M3", 8) for x, y in mot_holes]  # heads sunk in the stand
    asm.part("motor_screws", [W * s.shape for s in screws], ground=True, color="#1e1e22", material="steel",
             bom="4 x ISO 4762 M3x8 socket head cap screw")
    asm.fasten("motor_screws", "motor")                          # M3 thread in the motor's 2.5 mm tapped holes

    # ---- stage 1: sun on the motor shaft, planets on carrier 1 (which carries sun 2)
    asm.part("sun1", W * (Pos(0, 0, z1) * spur_gear(m, zs, width, bore=5, backlash=backlash)),
             material="steel", color="#d9dde2")
    asm.revolute("j_sun", "motor", "sun1", at=motor.frames["shaft"])
    pb = bearing("623")                                          # 3 x 10 x 4: one per planet, on a ø3 pin
    stages = (("1", phis1, z1, z_c1, t_c1, "j_sun", [_circ(7)], "#c2552f", "#e39a3b"),
              ("2", phis2, z2, z_c2, t_c2, "j_c1", [], "#1f8a7e", "#3bb3a2"))
    for s, phis, zg, zc, tc, j_in, bores, c_carrier, c_planet in stages:
        carrier = prism(_circ(2 * rc), bores + [_circ(3, *pin_at(p)) for p in phis], zc, tc)
        asm.part(f"carrier{s}", W * carrier, material="aluminum_6061", color=c_carrier)
        asm.revolute(f"j_c{s}", f"spacer{s}", f"carrier{s}", origin=P(0, 0, zc), axis=X)
        pins = [Pos(*pin_at(p), zg - 0.2) * Cylinder(1.5, zc + tc - zg + 0.2, align=(Align.CENTER, Align.CENTER, Align.MIN))
                for p in phis]
        asm.part(f"pins{s}", [W * c for c in pins], material="steel", color="#9aa0a6",
                 bom=f"{len(phis)} x dowel pin 3 x {zc + tc - zg + 0.2:g} mm")
        asm.fix(f"pins{s}", f"carrier{s}")
        for k, phi in enumerate(phis):
            g = Pos(*pin_at(phi), zg) * Rot(0, 0, planet_phase(zs, zp, phi)) * spur_gear(m, zp, width, bore=10, backlash=backlash)
            asm.part(f"planet{s}_{k}", W * g, material="steel", color=c_planet)
            asm.part(f"bearing{s}_{k}", W * (Pos(*pin_at(phi), zg + width / 2) * pb), color="#c8ccd1")
            asm.fix(f"bearing{s}_{k}", f"planet{s}_{k}")
            asm.revolute(f"j_p{s}_{k}", f"carrier{s}", f"planet{s}_{k}", origin=P(*pin_at(phi), zg), axis=X)
            asm.gear(j_in, f"j_p{s}_{k}", k_planet)              # external mesh: this stage's sun / planet
            asm.mesh(f"ring{s}", f"planet{s}_{k}")               # internal mesh: the fixed ring / planet
    asm.part("sun2", W * (Pos(0, 0, z_c1 + t_c1) * spur_gear(m, zs, width + 1, backlash=backlash)),
             material="steel", color="#d9dde2")                  # stage-2 sun, integral with carrier 1
    asm.fix("sun2", "carrier1")
    asm.couple("j_sun", "j_c1", 1 / i_stage)                     # carrier 1 = sun / 5
    asm.couple("j_c1", "j_c2", 1 / i_stage)                      # carrier 2 = carrier 1 / 5

    # ---- output: shaft through the bearing, hub + lever, payload
    shaft = Pos(0, 0, z_c2 + t_c2) * Cylinder(6, z_hub - z_c2 - t_c2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hub = Pos(0, 0, z_hub + 5) * Cylinder(12, 10)
    lever = Pos(0, arm_side * (arm + 10) / 2, z_hub + 5) * Box(8, arm + 10, 10)   # ends inside the weight
    asm.part("output", W * (shaft + hub + lever), material="aluminum_6061", color="#b8bec6")
    asm.fix("output", "carrier2")
    weight = Cylinder(20, 20) - Box(8.4, 44, 10.4)                                 # slotted onto the lever
    asm.part("payload", W * (Pos(0, arm_side * arm, z_hub + 5) * weight), material="steel",
             mass_g=payload_g, color="#2f3237")
    asm.fix("payload", "output")
    asm.probe("tip", part="output", point=P(0, arm_side * arm, z_hub + 5))

    # ---- intent
    asm.actuator("j_sun", capacity=capacity)                     # NEMA 17 holding torque, N·m
    # one sun turn at 5° steps: stage-1 planets roll through ~10 teeth (7 frames per tooth), stage 2 ~2
    asm.study("turn", drive={"j_sun": (0, 360)}, frames=73)
    spin = -(zs / (zs + zr)) * (zr / zp - 1)                     # absolute planet spin per sun turn
    asm.target(f"ratio {i_total:g}:1", output_ratio, min=i_total, max=i_total)
    asm.target(f"planet spin {spin:.4g}", planet_spin, min=spin, max=spin)
    asm.target("motor SF", "sf:j_sun", min=2)
    asm.target("output torque", "load:j_c2", min=3.5)
    asm.target("no collision", "clearance", min=0.3)
    return asm


if __name__ == "__main__":
    run(build())
