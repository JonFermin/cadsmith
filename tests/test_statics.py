"""mech.statics: gravity holding loads by virtual work against closed forms and energy differences."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos

from conftest import FOUR_BAR, make_four_bar
from mech import Assembly, Kinematics, part_props
from mech.assembly import Study
from mech.geom import link
from mech.motion import run_study
from mech.statics import gravity_loads

G = 9.80665


def props_of(asm: Assembly) -> dict:
    return {name: part_props(p) for name, p in asm.parts.items()}


def potential(asm: Assembly, props: dict, transforms: dict) -> float:
    """V = Σ m (−g)·x · 1e-3 J, straight from the definition."""
    return sum(mp.mass_kg * float(-asm.gravity @ (transforms[n][:3, :3] @ mp.com + transforms[n][:3, 3])) * 1e-3
               for n, mp in props.items())


def pendulum(axis=(0, -1, 0)) -> Assembly:
    """A 100 mm bar along +X hinged at the origin; axis (0, −1, 0) makes +θ raise it (θ from horizontal)."""
    asm = Assembly("pendulum")
    asm.part("base", Pos(0, 0, -10) * Box(20, 20, 10), ground=True)
    asm.part("arm", Pos(50, 0, 0) * Box(100, 10, 10), material="aluminum_6061")
    asm.revolute("j_arm", "base", "arm", origin=(0, 0, 0), axis=axis)
    return asm


def test_pendulum_holding_torque():
    asm = pendulum()
    kin, props = Kinematics(asm), props_of(asm)
    res = run_study(asm, kin, Study("lift", {"j_arm": (-90, 90)}, frames=37))
    loads = gravity_loads(asm, kin, res, props)
    m, r = props["arm"].mass_kg, props["arm"].com[0] * 1e-3  # kg, m
    assert r == pytest.approx(0.05)
    load = loads["j_arm"]
    assert (load["unit"], load["reflected_from"]) == ("N·m", None) and list(loads) == ["j_arm"]
    for theta, tau in zip(res.drive["j_arm"], load["series"]):
        exact = m * G * r * math.cos(math.radians(theta))  # positive: the actuator pushes +θ (up)
        assert abs(tau - exact) <= 1e-6 * m * G * r  # spec bound is 0.01·m·g·r
    assert load["frame"] == 18 and load["max_abs"] == pytest.approx(m * G * r, rel=1e-9)  # horizontal
    # reversing the axis reverses joint sign and torque sign together
    asm_r = pendulum(axis=(0, 1, 0))
    kin_r = Kinematics(asm_r)
    res_r = run_study(asm_r, kin_r, Study("lift", {"j_arm": (0, 30)}, frames=3))
    tau_r = gravity_loads(asm_r, kin_r, res_r, props_of(asm_r))["j_arm"]["series"]
    np.testing.assert_allclose(tau_r, [-m * G * r * math.cos(math.radians(t)) for t in (0, 15, 30)], rtol=1e-9)


def leadscrew(hand: str, screw_axis_z: float) -> Assembly:
    """Vertical lead screw (revolute driver) + carriage with a bracket (prismatic +Z, screw coupling)."""
    asm = Assembly("leadscrew")
    asm.part("frame", Pos(0, 0, -5) * Box(60, 60, 10), ground=True)
    asm.part("screw", Pos(0, 0, 100) * Cylinder(4, 200), material="steel")  # COM on its own axis
    asm.part("carriage", Pos(0, 0, 50) * Box(40, 40, 15), material="aluminum_6061")
    asm.part("bracket", Pos(30, 0, 50) * Box(20, 10, 15), material="steel")
    asm.revolute("j_leadrot", "frame", "screw", origin=(0, 0, 0), axis=(0, 0, screw_axis_z))
    asm.prismatic("j_carriage", "frame", "carriage", origin=(0, 0, 50), axis=(0, 0, 1), limits=(-100, 100))
    asm.fix("bracket", "carriage")
    asm.screw("j_leadrot", "j_carriage", lead=8, hand=hand)
    return asm


@pytest.mark.parametrize("hand, screw_axis_z", [("right", 1.0), ("left", 1.0), ("right", -1.0)])
def test_leadscrew_holding_torque_and_reflected_load(hand, screw_axis_z):
    asm = leadscrew(hand, screw_axis_z)
    kin, props = Kinematics(asm), props_of(asm)
    res = run_study(asm, kin, Study("travel", {"j_leadrot": (0, 1080)}, frames=7))
    # +θ moves the carriage by ratio·θ: right hand with co-directional axes goes down (spec §2.1)
    direction = (-1.0 if hand == "right" else 1.0) * screw_axis_z
    assert res.poses[-1].q["j_carriage"] == pytest.approx(direction * 3 * 8.0, abs=1e-9)
    loads = gravity_loads(asm, kin, res, props)
    assert list(loads) == ["j_leadrot", "j_carriage"]
    m = props["carriage"].mass_kg + props["bracket"].mass_kg  # everything moving with the carriage
    torque, carriage = loads["j_leadrot"], loads["j_carriage"]
    assert (torque["unit"], torque["reflected_from"]) == ("N·m", None)
    assert (carriage["unit"], carriage["reflected_from"]) == ("N", "j_leadrot")
    for tau, force in zip(torque["series"], carriage["series"]):
        assert tau == pytest.approx(direction * m * G * 0.008 / (2 * math.pi), rel=1e-6)  # spec: 1%
        assert force == pytest.approx(m * G, rel=1e-6)  # lifting the carriage takes +m·g along +Z
    assert torque["max_abs"] == pytest.approx(m * G * 0.008 / (2 * math.pi), rel=1e-6)


def test_four_bar_load_is_the_energy_gradient_along_the_solved_path():
    """In-plane gravity: τ(θ) equals dV/dθ from re-solved neighbouring poses (loop closure included)."""
    asm = make_four_bar(**FOUR_BAR)
    asm.gravity = np.array([0.0, -G, 0.0])
    kin, props = Kinematics(asm), props_of(asm)
    res = run_study(asm, kin, Study("turn", {"j_crank": (0, 360)}, frames=25))
    load = gravity_loads(asm, kin, res, props)["j_crank"]
    assert load["max_abs"] > 1e-3
    h = 1e-3  # deg
    for k in range(0, 25, 3):
        theta, pose = float(res.drive["j_crank"][k]), res.poses[k]
        up = kin.solve({"j_crank": theta + h}, pose.q)
        down = kin.solve({"j_crank": theta - h}, pose.q)
        assert up.ok and down.ok
        dV = (potential(asm, props, up.transforms) - potential(asm, props, down.transforms)) / (2 * h)
        assert load["series"][k] == pytest.approx(dV * 180 / math.pi, abs=1e-6 * load["max_abs"])
    # the passive joints carry nothing (frictionless pins): only the driver is reported
    assert set(gravity_loads(asm, kin, res, props)) == {"j_crank"}


def test_open_frames_have_no_load():
    asm = make_four_bar(**FOUR_BAR)
    asm.gravity = np.array([0.0, -G, 0.0])
    kin = Kinematics(asm)
    res = run_study(asm, kin, Study("reach", {"j_rocker": (0, 80)}, frames=9))  # beyond the rocker's swing
    series = gravity_loads(asm, kin, res, props_of(asm))["j_rocker"]["series"]
    assert [v is None for v in series] == [not p.ok for p in res.poses]
    assert any(v is None for v in series) and any(v is not None for v in series)


def test_singular_frames_have_no_load():
    """A four-bar drawn at a toggle: J_p is singular at home, so no holding load is defined there."""
    O2, A, O4 = np.array([0.0, 0, 0]), np.array([0.0, 40, 0]), np.array([100.0, 0, 0])
    u = (O4 - A) / np.linalg.norm(O4 - A)
    B = O4 + 80 * u  # coupler and rocker collinear
    asm = Assembly("toggle", gravity=(0, -G, 0))
    asm.part("frame", Pos(50, 0, -10) * Box(220, 16, 5), ground=True)
    asm.part("crank", link(O2, A, 10, 5, z=0))
    asm.part("coupler", link(A, B, 10, 5, z=5.5))
    asm.part("rocker", link(O4, B, 10, 5, z=11))
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1))
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1))
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))
    kin = Kinematics(asm)
    res = run_study(asm, kin, Study("hold", {"j_crank": 0.0}, frames=2))
    assert all(p.ok for p in res.poses)
    load = gravity_loads(asm, kin, res, props_of(asm))["j_crank"]
    assert load["series"] == [None, None] and load["max_abs"] is None and load["frame"] is None


def test_undriven_loop_does_not_block_other_loads():
    """A pendulum driven next to an undriven four-bar: the loop stays at home, the load is exact."""
    asm = make_four_bar(**FOUR_BAR)
    asm.studies.clear()
    asm.part("post", Pos(0, 200, -10) * Box(20, 20, 10), ground=True)
    asm.part("arm", Pos(50, 200, 0) * Box(100, 10, 10), material="aluminum_6061")
    asm.revolute("j_arm", "post", "arm", origin=(0, 200, 0), axis=(0, -1, 0))
    kin, props = Kinematics(asm), props_of(asm)
    assert kin.mobility(["j_arm"]) == 1  # the four-bar is free, held at home
    res = run_study(asm, kin, Study("lift", {"j_arm": (0, 60)}, frames=4))
    series = gravity_loads(asm, kin, res, props)["j_arm"]["series"]
    m, r = props["arm"].mass_kg, 0.05
    np.testing.assert_allclose(series, [m * G * r * math.cos(math.radians(t)) for t in (0, 20, 40, 60)], rtol=1e-9)
