"""mech.kinematics against closed forms: four-bar, slider-crank, couplings, drivers, mobility, Jacobians."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Pos
from scipy.optimize import brentq

from conftest import FOUR_BAR, FOUR_BAR_LINKS, SLIDER_CRANK, four_bar_rocker_deg, make_disc, make_four_bar
from mech import Assembly, Kinematics
from mech.geom import circle_intersect, link, rot_about_line, to_location, transform_points, translation
from mech.kinematics import Pose


def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def rotation_deg_about_z(T: np.ndarray) -> float:
    return math.degrees(math.atan2(T[1, 0], T[0, 0]))


def pin_gap(pose: Pose, asm: Assembly) -> float:
    """Independent check of loop closure: distance between the pin point carried by part a and b."""
    return max(float(np.linalg.norm(transform_points(pose.transforms[p.a], p.point)
                                    - transform_points(pose.transforms[p.b], p.point))) for p in asm.pins)


# ------------------------------------------------------------------------------ four-bar


def test_four_bar_structure(four_bar):
    kin = Kinematics(four_bar)
    assert kin.loops == [["j_coupler", "j_crank", "j_rocker"]]
    assert kin.fixed == set() and kin.coupled == set()
    assert kin.unknowns(["j_crank"]) == ["j_coupler", "j_rocker"]
    assert kin.mobility(["j_crank"]) == 0
    assert kin.mobility([]) == 1
    assert kin.overconstrained(["j_crank"]) == []
    assert not kin.singular_at_home(["j_crank"])
    assert kin.roles_for(four_bar.studies) == {"j_crank": "driver", "j_coupler": "passive", "j_rocker": "passive"}
    lo, hi = four_bar._extent()
    assert kin.L == pytest.approx(float(np.linalg.norm(hi - lo)))


def test_four_bar_rocker_matches_freudenstein_over_full_turn(four_bar):
    kin = Kinematics(four_bar)
    a, b, c, d = (FOUR_BAR[k] for k in ("crank", "coupler", "rocker", "ground"))
    k1, k2, k3 = d / a, d / c, (a * a - b * b + c * c + d * d) / (2 * a * c)
    theta4_home = four_bar_rocker_deg(0.0, **FOUR_BAR_LINKS)
    guess = None
    for k in range(73):  # 0..360 in 5° steps, warm-started like a study
        theta2 = 5.0 * k
        pose = kin.solve({"j_crank": theta2}, guess)
        guess = pose.q
        assert pose.ok and pose.residual < 1e-6 and pin_gap(pose, four_bar) < 1e-6
        assert pose.q["j_crank"] == theta2
        expected = four_bar_rocker_deg(theta2, **FOUR_BAR_LINKS) - theta4_home
        assert abs(wrap180(pose.q["j_rocker"] - expected)) < 1e-3
        t2, t4 = math.radians(theta2), math.radians(theta4_home + pose.q["j_rocker"])
        assert abs(k1 * math.cos(t4) - k2 * math.cos(t2) + k3 - math.cos(t2 - t4)) < 1e-9  # Freudenstein
        # the rocker's joint value is also its actual rotation
        assert abs(wrap180(rotation_deg_about_z(pose.transforms["rocker"]) - expected)) < 1e-3
    assert pose.q["j_rocker"] == pytest.approx(0.0, abs=1e-9)  # back home after a full turn


def test_four_bar_direct_solve_stays_on_drawn_branch(four_bar):
    """No guess: the solver walks from home in ≤10° steps, so it can't jump to the crossed branch."""
    kin = Kinematics(four_bar)
    for theta2 in (90.0, 180.0, 270.0, -135.0):
        pose = kin.solve({"j_crank": theta2})
        expected = four_bar_rocker_deg(theta2, **FOUR_BAR_LINKS) - four_bar_rocker_deg(0.0, **FOUR_BAR_LINKS)
        assert pose.ok and abs(wrap180(pose.q["j_rocker"] - expected)) < 1e-3


def test_four_bar_invariant_under_rigid_placement(four_bar):
    """The same linkage drawn in a tilted plane far from the origin: identical joint values."""
    T = translation((250, -80, 40)) @ rot_about_line((10, 20, 30), (1, -2, 0.5), 57)
    R = T[:3, :3]
    loc = to_location(T)
    moved = Assembly("four_bar_tilted")
    for p in four_bar.parts.values():
        moved.part(p.name, loc * p.shape, ground=p.ground)
    for j in four_bar.joints.values():
        moved.revolute(j.name, j.parent, j.child, origin=transform_points(T, j.origin), axis=R @ j.axis)
    for p in four_bar.pins:
        moved.pin(p.name, p.a, p.b, point=transform_points(T, p.point), axis=R @ p.axis)
    assert moved.validate() == [] and moved.validate_warnings() == []
    flat, tilted = Kinematics(four_bar), Kinematics(moved)
    assert tilted.mobility(["j_crank"]) == 0
    for theta2 in (30.0, 150.0, 290.0):
        a, b = flat.solve({"j_crank": theta2}), tilted.solve({"j_crank": theta2})
        assert b.ok and b.residual < 1e-6
        for name in ("j_coupler", "j_rocker"):
            assert b.q[name] == pytest.approx(a.q[name], abs=1e-6)
        # world motion is the flat motion conjugated by T
        np.testing.assert_allclose(b.transforms["coupler"], T @ a.transforms["coupler"] @ np.linalg.inv(T),
                                   atol=1e-9)


def test_drive_rocker_instead_of_crank(four_bar):
    """Driving the (loop) rocker makes the crank passive; its angle follows the inverse closed form."""
    kin = Kinematics(four_bar)
    a, b, c, d = (FOUR_BAR[k] for k in ("crank", "coupler", "rocker", "ground"))
    swing = [four_bar_rocker_deg(t, **FOUR_BAR_LINKS) for t in np.linspace(0, 360, 721)]
    t4_home = swing[0]
    lo, hi = min(swing) - t4_home, max(swing) - t4_home  # reachable rocker range relative to home
    assert lo < -10 and hi > 10

    def crank_deg(q_rocker: float, sign: float) -> float:
        t4 = math.radians(t4_home + q_rocker)
        bx, by = d + c * math.cos(t4), c * math.sin(t4)
        R = math.hypot(bx, by)
        return math.degrees(math.atan2(by, bx) + sign * math.acos((a * a + R * R - b * b) / (2 * a * R)))

    branch = min((+1.0, -1.0), key=lambda s: abs(wrap180(crank_deg(0.0, s))))  # the drawn crank at 0°
    assert kin.unknowns(["j_rocker"]) == ["j_crank", "j_coupler"]  # declaration order
    assert kin.mobility(["j_rocker"]) == 0 and kin.overconstrained(["j_rocker"]) == []
    for target in (0.6 * lo, 0.6 * hi):
        guess = None
        for q_rocker in np.linspace(0.0, target, 25):
            pose = kin.solve({"j_rocker": q_rocker}, guess)
            guess = pose.q
            assert pose.ok and pose.residual < 1e-6 and pin_gap(pose, four_bar) < 1e-6
            assert abs(wrap180(pose.q["j_crank"] - crank_deg(q_rocker, branch))) < 1e-3


def test_overconstrained_drivers_detected(four_bar):
    kin = Kinematics(four_bar)
    msgs = kin.overconstrained(["j_crank", "j_rocker"])
    assert len(msgs) == 1
    assert all(s in msgs[0] for s in ("j_crank", "j_rocker", "'p_B'", "1 degree"))
    assert kin.unknowns(["j_crank", "j_rocker"]) == ["j_coupler"]
    # solving an over-driven loop fails gracefully at a drive the loop can't reach
    pose = kin.solve({"j_crank": 90.0, "j_rocker": 0.0})
    assert not pose.ok and pose.residual > 1.0


def four_bar_with_arm(on_turntable: bool = False) -> Assembly:
    """The §2 four-bar plus a serial arm on the frame — or the whole linkage on a turntable."""
    asm = make_four_bar(**FOUR_BAR)
    asm.part("arm", Pos(20, 0, -15) * Box(40, 6, 4))
    asm.revolute("j_arm", "frame", "arm", origin=(0, 0, -15), axis=(0, 0, 1))
    if on_turntable:  # re-parent the linkage's ground pivots onto the arm
        asm.joints["j_crank"].parent = asm.joints["j_rocker"].parent = "arm"
    return asm


def test_idle_unknowns_are_loops_no_driver_acts_on():
    kin = Kinematics(four_bar_with_arm())
    loop = ["j_crank", "j_coupler", "j_rocker"]
    assert kin.idle_unknowns(["j_arm"]) == loop and kin.idle_unknowns(["j_crank"]) == []
    assert kin.mobility(["j_arm"]) == 1  # mobility counts every loop...
    assert kin.mobility(["j_arm"] + kin.idle_unknowns(["j_arm"])) == 0  # ...the arm study drives none
    # a loop carried by the driven joint moves rigidly with it and stays at home internally
    carried = Kinematics(four_bar_with_arm(on_turntable=True))
    assert carried.loops == [["j_coupler", "j_crank", "j_rocker"]]
    assert carried.idle_unknowns(["j_arm"]) == loop
    # a coupling ties the arm to the loop: driving the arm now acts on it
    coupled = four_bar_with_arm()
    coupled.couple("j_arm", "j_crank", 1.0)
    kin_c = Kinematics(coupled)
    assert kin_c.idle_unknowns(["j_arm"]) == [] and kin_c.mobility(["j_arm"]) == 0


def test_singular_home_has_full_generic_rank():
    """Four-bar drawn at a toggle (coupler and rocker collinear): J_p is rank-deficient at home only."""
    O2, A, O4 = np.array([0.0, 0, 0]), np.array([0.0, 40, 0]), np.array([100.0, 0, 0])
    u = (O4 - A) / np.linalg.norm(O4 - A)
    B = O4 + 80 * u  # A, O4, B collinear
    asm = Assembly("toggle")
    asm.part("frame", Pos(50, 0, -10) * Box(220, 16, 5), ground=True)
    asm.part("crank", link(O2, A, 10, 5, z=0))
    asm.part("coupler", link(A, B, 10, 5, z=5.5))
    asm.part("rocker", link(O4, B, 10, 5, z=11))
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1))
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1))
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))
    kin = Kinematics(asm)
    Jp, _ = kin.jacobians(kin.solve({}), ["j_crank"])
    assert np.linalg.matrix_rank(Jp, tol=1e-9 * np.abs(Jp).max()) == 1
    assert kin.mobility(["j_crank"]) == 0
    assert kin.singular_at_home(["j_crank"])


def test_limit_violations_reported_for_passive_joints():
    asm = make_four_bar(**FOUR_BAR)
    asm.joints["j_rocker"].limits = (-10.0, 10.0)
    kin = Kinematics(asm)
    expected = four_bar_rocker_deg(180.0, **FOUR_BAR_LINKS) - four_bar_rocker_deg(0.0, **FOUR_BAR_LINKS)
    assert expected > 10
    assert kin.solve({"j_crank": 5.0}).limit_violations == []
    pose = kin.solve({"j_crank": 180.0})
    assert pose.ok
    assert pose.limit_violations == [("j_rocker", pytest.approx(expected, abs=1e-6), (-10.0, 10.0))]
    asm.joints["j_crank"].limits = (0.0, 45.0)  # drivers are check_study's job, not reported here
    assert [v[0] for v in Kinematics(asm).solve({"j_crank": 180.0}).limit_violations] == ["j_rocker"]


def test_solve_never_raises():
    kin = Kinematics(make_four_bar(**FOUR_BAR))
    # rocker +80° from home puts it at θ4 ≈ 181°: |O2B| = 20 mm < b − a, no assembly mode reaches it
    for drive, guess in [({"j_crank": float("nan")}, None), ({"j_crank": 10.0}, {"j_rocker": "garbage"}),
                         ({"j_rocker": 80.0}, None)]:
        pose = kin.solve(drive, guess)
        assert isinstance(pose, Pose) and not pose.ok
        assert set(pose.transforms) == {"frame", "crank", "coupler", "rocker"}
    ignored = kin.solve({"nope": 5.0, "j_crank": 20.0})  # unknown names are ignored (check_study rejects them)
    assert ignored.ok and ignored.q["j_crank"] == 20.0


# ------------------------------------------------------------------------------ slider-crank


def test_slider_crank_position(slider_crank):
    r, l = SLIDER_CRANK["r"], SLIDER_CRANK["l"]
    kin = Kinematics(slider_crank)
    assert kin.unknowns(["j_crank"]) == ["j_rod", "j_slider"] and kin.mobility(["j_crank"]) == 0
    assert slider_crank.validate_warnings() == []
    guess = None
    for theta in np.arange(0.0, 360.0 + 1e-9, 7.5):
        pose = kin.solve({"j_crank": theta}, guess)
        guess = pose.q
        th = math.radians(theta)
        x = r * math.cos(th) + math.sqrt(l * l - (r * math.sin(th)) ** 2)
        assert pose.ok and pose.residual < 1e-6
        assert abs(pose.q["j_slider"] - x) < 1e-4
        assert abs(kin.point(pose, "slider", (r + l, 0, 0))[0] - x) < 1e-4  # probe-style check


# ------------------------------------------------------------------------------ forward kinematics


def serial_arm() -> Assembly:
    asm = Assembly("arm")
    asm.part("base", Box(20, 20, 4), ground=True)
    asm.part("upper", link((0, 0, 0), (50, 0, 0), 8, 4, z=2))
    asm.part("fore", link((50, 0, 0), (100, 0, 0), 8, 4, z=6.5))
    asm.part("slide", Pos(100, 0, 13) * Box(6, 6, 6))
    asm.revolute("j1", "base", "upper", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j2", "upper", "fore", origin=(50, 0, 0), axis=(0, 0, 1), home=30)
    asm.prismatic("j3", "fore", "slide", origin=(100, 0, 10), axis=(1, 0, 0))
    return asm


def test_fk_product_of_exponentials():
    asm = serial_arm()
    kin = Kinematics(asm)
    for T in kin.fk({}).values():
        np.testing.assert_allclose(T, np.eye(4), atol=1e-15)  # home (j2 home = 30) is the drawing
    T = kin.fk({"j1": 90.0, "j2": 120.0, "j3": 10.0})  # j2 moves +90° from its home
    # j1 swings the elbow to (0, 50); the forearm then points along −x; the slide moves 10 mm along it
    np.testing.assert_allclose(transform_points(T["fore"], (100, 0, 0)), [-50, 50, 0], atol=1e-12)
    np.testing.assert_allclose(transform_points(T["slide"], (100, 0, 0)), [-60, 50, 0], atol=1e-12)
    np.testing.assert_allclose(T["upper"], kin.fk({"j1": 90.0})["upper"])
    pose = kin.solve({"j1": 90.0, "j2": 120.0, "j3": 10.0})
    assert pose.ok and pose.residual == 0.0 and kin.mobility(["j1"]) == 0
    assert kin.roles_for([]) == {"j1": "free", "j2": "free", "j3": "free"}
    assert kin.expand({"j1": 5.0}) == {"j1": 5.0, "j2": 30.0, "j3": 0.0}
    with pytest.raises(KeyError, match="did you mean 'j1'"):
        kin.fk({"j_1": 5.0})


# ------------------------------------------------------------------------------ couplings


def two_discs(homes=(0.0, 0.0)) -> Assembly:
    """Discs on parallel +Z axes 30 mm apart, not yet coupled."""
    asm = Assembly("gears")
    asm.part("frame", Pos(15, 0, -2) * Box(60, 30, 4), ground=True)
    asm.part("g1", make_disc((0, 0, 0), r=10))
    asm.part("g2", make_disc((30, 0, 0), r=20))
    asm.revolute("j1", "frame", "g1", origin=(0, 0, 0), axis=(0, 0, 1), home=homes[0])
    asm.revolute("j2", "frame", "g2", origin=(30, 0, 0), axis=(0, 0, 1), home=homes[1])
    return asm


def gear_pair_asm(ratio: float, homes=(0.0, 0.0)) -> Assembly:
    asm = two_discs(homes)
    asm.gear("j1", "j2", ratio)
    return asm


def test_gear_coupling_sign_and_transform():
    kin = Kinematics(gear_pair_asm(-0.5))
    assert kin.coupled == {"j2"}
    T = kin.fk({"j1": 90.0})
    assert kin.expand({"j1": 90.0})["j2"] == pytest.approx(-45.0)
    assert rotation_deg_about_z(T["g2"]) == pytest.approx(-45.0)  # external mesh: opposite sense
    np.testing.assert_allclose(transform_points(T["g2"], (30, 0, 0)), [30, 0, 0], atol=1e-12)
    # the driven value in fk input is overridden by the coupling
    assert rotation_deg_about_z(kin.fk({"j1": 90.0, "j2": 10.0})["g2"]) == pytest.approx(-45.0)
    assert kin.roles_for([]) == {"j1": "free", "j2": "coupled"}


def screw_asm(hand="right", screw_axis=(0, 0, 1), nut_axis=(0, 0, 1)) -> Assembly:
    asm = Assembly("screw")
    asm.part("frame", Pos(0, 0, -2) * Box(40, 40, 4), ground=True)
    asm.part("screw", Pos(0, 0, 40) * Box(8, 8, 80))
    asm.part("nut", Pos(0, 0, 20) * Box(16, 16, 8))
    asm.revolute("j_rot", "frame", "screw", origin=(0, 0, 0), axis=screw_axis)
    asm.prismatic("j_nut", "frame", "nut", origin=(0, 0, 0), axis=nut_axis)
    asm.screw("j_rot", "j_nut", lead=8, hand=hand)
    return asm


@pytest.mark.parametrize("hand, screw_axis, nut_axis, ratio, dz", [
    ("right", (0, 0, 1), (0, 0, 1), -8 / 360, -8.0),  # co-directional: −lead/360
    ("right", (0, 0, 1), (0, 0, -1), 8 / 360, -8.0),  # flipped travel axis: same physical motion
    ("right", (0, 0, -1), (0, 0, 1), 8 / 360, 8.0),  # +360° about −Z is a clockwise turn seen from +Z
    ("left", (0, 0, 1), (0, 0, 1), 8 / 360, 8.0),
])
def test_screw_direction_right_hand_rule(hand, screw_axis, nut_axis, ratio, dz):
    """Right-hand screw turned +360° (right-hand rule about its axis) in a non-rotating nut: the
    screw would advance along +axis, so a nut on an axially fixed screw moves −lead along it."""
    asm = screw_asm(hand, screw_axis, nut_axis)
    kin = Kinematics(asm)
    assert asm.couplings[0].ratio == pytest.approx(ratio)
    T = kin.fk({"j_rot": 360.0})
    np.testing.assert_allclose(T["nut"][:3, 3], [0, 0, dz], atol=1e-12)
    np.testing.assert_allclose(kin.fk({"j_rot": 45.0})["nut"][:3, 3], [0, 0, dz / 8], atol=1e-12)


def rack_asm(rack_y: float, sign=None) -> Assembly:
    asm = Assembly("rack")
    asm.part("frame", Pos(0, 0, -3) * Box(120, 60, 2), ground=True)
    asm.part("pinion", make_disc((0, 0, 0), r=10))
    asm.part("rack", Pos(0, rack_y, 2) * Box(100, 6, 4))
    asm.revolute("j_pinion", "frame", "pinion", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_rack", "frame", "rack", origin=(0, rack_y, 0), axis=(1, 0, 0))
    asm.rack("j_pinion", "j_rack", pitch_radius=10, sign=sign)
    return asm


@pytest.mark.parametrize("rack_y, expected_sign", [(-13.0, 1), (13.0, -1)])
def test_rack_sign_from_geometry(rack_y, expected_sign):
    """The rack moves with the pinion's pitch point at the mesh: ω × r_contact."""
    asm = rack_asm(rack_y)
    kin = Kinematics(asm)
    assert asm.couplings[0].ratio == pytest.approx(expected_sign * math.pi * 10 / 180)
    dtheta = 1e-3  # deg; central differences cancel the radial (second-order) motion of the pitch point
    contact = np.array([0.0, math.copysign(10.0, rack_y), 0.0])
    plus, minus = kin.fk({"j_pinion": dtheta}), kin.fk({"j_pinion": -dtheta})
    pitch_velocity = (transform_points(plus["pinion"], contact) - transform_points(minus["pinion"], contact)) / (
        2 * dtheta)
    rack_velocity = (plus["rack"][:3, 3] - minus["rack"][:3, 3]) / (2 * dtheta)
    np.testing.assert_allclose(rack_velocity, pitch_velocity, rtol=1e-9, atol=1e-12)
    assert abs(rack_velocity[0]) == pytest.approx(10 * math.pi / 180)
    T = kin.fk({"j_pinion": 90.0})
    np.testing.assert_allclose(T["rack"][:3, 3], [expected_sign * 10 * math.pi / 2, 0, 0], atol=1e-12)
    assert Kinematics(rack_asm(rack_y, sign=-expected_sign)).expand({"j_pinion": 90.0})["j_rack"] == \
        pytest.approx(-expected_sign * 5 * math.pi)


def test_couplings_act_on_deltas_from_nonzero_homes():
    asm = two_discs(homes=(30.0, 10.0))
    asm.couple("j1", "j2", ratio=2.0, offset=5.0)
    kin = Kinematics(asm)
    # home stays consistent only with offset 0; here the offset shifts the driven joint by 5°
    assert kin.expand({}) == {"j1": 30.0, "j2": 15.0}
    q = kin.expand({"j1": 40.0})
    assert q["j2"] == pytest.approx(10.0 + 2.0 * (40.0 - 30.0) + 5.0)
    T = kin.fk({"j1": 40.0})
    assert rotation_deg_about_z(T["g1"]) == pytest.approx(10.0)  # motion is relative to the drawing
    assert rotation_deg_about_z(T["g2"]) == pytest.approx(25.0)

    asm = gear_pair_asm(-0.5, homes=(30.0, 10.0))
    kin = Kinematics(asm)
    assert kin.expand({}) == {"j1": 30.0, "j2": 10.0}  # offset 0: home is kinematically consistent
    for T in kin.fk({}).values():
        np.testing.assert_allclose(T, np.eye(4), atol=1e-15)
    assert kin.expand({"j1": 50.0})["j2"] == pytest.approx(10.0 - 0.5 * 20.0)


def test_transitive_coupling_chain():
    asm = gear_pair_asm(-0.5)
    asm.part("g3", make_disc((0, 40, 0), r=10))
    asm.revolute("j3", "frame", "g3", origin=(0, 40, 0), axis=(0, 0, 1), home=5.0)
    asm.gear("j2", "j3", -2.0)
    kin = Kinematics(asm)
    assert kin.coupled == {"j2", "j3"}
    q = kin.expand({"j1": 36.0})
    assert q["j2"] == pytest.approx(-18.0) and q["j3"] == pytest.approx(5.0 + 36.0)


def geared_five_bar() -> Assembly:
    """Five-bar whose right ground joint is coupled to a *passive* joint: j_d = 0.5·Δj_b."""
    O1, P1, O2, P2 = (0, 0, 0), (0, 40, 0), (80, 0, 0), (80, 40, 0)
    Q = circle_intersect(P1, 50, P2, 50, side=+1)  # (40, 70, 0)
    asm = Assembly("five_bar")
    asm.part("frame", Pos(40, 0, -10) * Box(110, 16, 5), ground=True)
    asm.part("l1", link(O1, P1, 10, 5, z=0))
    asm.part("l2", link(P1, Q, 10, 5, z=5.5))
    asm.part("l4", link(O2, P2, 10, 5, z=0))
    asm.part("l3", link(P2, Q, 10, 5, z=11))
    asm.revolute("j_a", "frame", "l1", origin=O1, axis=(0, 0, 1))
    asm.revolute("j_b", "l1", "l2", origin=P1, axis=(0, 0, 1))
    asm.revolute("j_d", "frame", "l4", origin=O2, axis=(0, 0, 1))
    asm.revolute("j_c", "l4", "l3", origin=P2, axis=(0, 0, 1))
    asm.pin("p_Q", "l2", "l3", point=Q, axis=(0, 0, 1))
    asm.couple("j_b", "j_d", ratio=0.5)
    return asm


def five_bar_closure_mm(qa: float, qb: float) -> float:
    """|Q_left − P2| − 50 for the geared five-bar, from plane geometry alone (j_d = 0.5·j_b)."""
    def rot(p, deg, c):
        t = math.radians(deg)
        v = np.subtract(p, c)
        return np.add(c, [v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)])

    q_left = rot(rot((40, 70), qb, (0, 40)), qa, (0, 0))
    p2 = rot((80, 40), 0.5 * qb, (80, 0))
    return float(np.linalg.norm(q_left - p2)) - 50.0


def test_coupling_driven_by_a_passive_loop_joint():
    asm = geared_five_bar()
    assert asm.pins[0].point.tolist() == pytest.approx([40, 70, 0])
    kin = Kinematics(asm)
    assert kin.unknowns(["j_a"]) == ["j_b", "j_c"]  # j_d is coupled, not an unknown
    assert kin.mobility(["j_a"]) == 0 and kin.mobility([]) == 1
    guess, qb_ref = None, 0.0
    for qa in np.linspace(0.0, -25.0, 11):
        pose = kin.solve({"j_a": qa}, guess)
        guess = pose.q
        assert pose.ok and pose.residual < 1e-9 and pin_gap(pose, asm) < 1e-9
        assert pose.q["j_d"] == pytest.approx(0.5 * pose.q["j_b"], abs=1e-12)
        # independent check: the scalar closure root nearest the previous frame's j_b
        qb_ref = brentq(lambda qb: five_bar_closure_mm(qa, qb), qb_ref - 1e-9, qb_ref + 15.0) if qa else 0.0
        assert pose.q["j_b"] == pytest.approx(qb_ref, abs=1e-6)
    assert pose.q["j_b"] > 45.0  # the coupled joint really moved (≈ 50.6° at j_a = −25°)


# ------------------------------------------------------------------------------ Jacobians


def residual_of(kin: Kinematics, asm: Assembly, q: dict) -> np.ndarray:
    """Loop residual recomputed from fk alone (independent of the analytic Jacobian code)."""
    T = kin.fk(q)
    rows = []
    for p in asm.pins:
        rows.append(transform_points(T[p.a], p.point) - transform_points(T[p.b], p.point))
        if p.axis is not None:
            rows.append(kin.L * (T[p.a][:3, :3] @ p.axis - T[p.b][:3, :3] @ p.axis))
    return np.concatenate(rows)


@pytest.mark.parametrize("builder, driven, drive", [
    (lambda: make_four_bar(**FOUR_BAR), ["j_crank"], {"j_crank": 37.0}),
    (geared_five_bar, ["j_a"], {"j_a": -12.0}),
])
def test_jacobians_match_finite_differences(builder, driven, drive):
    asm = builder()
    kin = Kinematics(asm)
    pose = kin.solve(drive)
    assert pose.ok
    Jp, Jd = kin.jacobians(pose, driven)
    unknown = kin.unknowns(driven)
    h = 1e-5
    for J, names in ((Jp, unknown), (Jd, driven)):
        assert J.shape == (len(residual_of(kin, asm, pose.q)), len(names))
        for k, name in enumerate(names):
            plus, minus = dict(pose.q), dict(pose.q)
            plus[name] += h
            minus[name] -= h
            fd = (residual_of(kin, asm, plus) - residual_of(kin, asm, minus)) / (2 * h)
            np.testing.assert_allclose(J[:, k], fd, rtol=1e-6, atol=1e-7)
    # the implicit-function tangent predicts the next pose to first order
    step = 0.01
    dq = -np.linalg.pinv(Jp) @ Jd @ np.array([step])
    nxt = kin.solve({driven[0]: drive[driven[0]] + step}, pose.q)
    for k, name in enumerate(unknown):
        assert nxt.q[name] - pose.q[name] == pytest.approx(dq[k], abs=1e-5)


# ------------------------------------------------------------------------------ review regressions


def planar_four_bar(ground: float, crank: float, coupler: float, rocker: float, th0: float = 0.0,
                    side: int = +1, t: float = 5.0, absolute: bool = False) -> Assembly:
    """Four-bar with the crank drawn at th0 (joint home = th0) and B from circle_intersect(side);
    ``absolute`` makes the rocker's home its drawn angle (else 0: j_rocker = change from home)."""
    O2, O4 = (0.0, 0.0, 0.0), (ground, 0.0, 0.0)
    A = (crank * math.cos(math.radians(th0)), crank * math.sin(math.radians(th0)), 0.0)
    B = circle_intersect(A, coupler, O4, rocker, side=side)
    asm = Assembly("fb")
    asm.part("frame", Pos(ground / 2, 0, -2 * t) * Box(ground + 2 * crank + 20, 16, t), ground=True)
    asm.part("crank", link(O2, A, 10, t, z=0))
    asm.part("coupler", link(A, B, 10, t, z=t + 0.5))
    asm.part("rocker", link(O4, B, 10, t, z=2 * t + 1))
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1), home=th0)
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1),
                 home=math.degrees(math.atan2(B[1], B[0] - ground)) if absolute else 0.0)
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))
    return asm


@pytest.mark.parametrize("frames, singular", [(73, [24, 60]), (37, [12, 30]), (50, [])])
def test_parallelogram_keeps_its_branch_at_the_change_points(frames, singular):
    """crank = rocker = 40, coupler = ground = 100, driven 60 → 420° (review p01): frames landing on
    the change points (180°, 360°) must not flip to the anti-parallelogram, and are listed."""
    from mech.assembly import Study
    from mech.motion import run_study

    asm = planar_four_bar(100, 40, 100, 40, th0=60.0, absolute=True)
    kin = Kinematics(asm)
    res = run_study(asm, kin, Study("turn", {"j_crank": (60, 420)}, frames=frames))
    for pose in res.poses:  # (on a change point the root is double: ~1e-6° is the solver's resolution)
        assert pose.ok and abs(wrap180(pose.q["j_rocker"] - pose.q["j_crank"])) < 1e-4
        assert abs(wrap180(pose.q["j_coupler"] + pose.q["j_crank"] - 60.0)) < 1e-4  # coupler stays parallel
    assert res.branch_jumps == [] and res.singular == singular


def test_redundant_parallel_link_is_mobile():
    """A parallelogram with a third equal parallel link (review p02): Grübler says 0 DOF, but on the
    constraint manifold it moves — one driver, not overconstrained."""
    from mech.assembly import Study
    from mech.motion import check_study, run_study

    th0 = 60.0
    O2, O4, O6 = (0.0, 0, 0), (100.0, 0, 0), (50.0, 0, 0)
    pol = lambda c, r, d: (c[0] + r * math.cos(math.radians(d)), c[1] + r * math.sin(math.radians(d)), 0.0)
    A, B, C = pol(O2, 40, th0), pol(O4, 40, th0), pol(O6, 40, th0)
    asm = Assembly("double_par")
    asm.part("frame", Pos(50, 0, -10) * Box(140, 16, 5), ground=True)
    asm.part("crank", link(O2, A, 10, 5, z=0))
    asm.part("coupler", link(A, B, 10, 5, z=5.5))
    asm.part("rocker", link(O4, B, 10, 5, z=11))
    asm.part("rocker2", link(O6, C, 10, 5, z=11))
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1), home=th0)
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1), home=th0)
    asm.revolute("j_rocker2", "frame", "rocker2", origin=O6, axis=(0, 0, 1), home=th0)
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))
    asm.pin("p_C", "coupler", "rocker2", point=C, axis=(0, 0, 1))
    kin = Kinematics(asm)
    assert kin.mobility(["j_crank"]) == 0 and kin.mobility([]) == 1
    study = Study("open", {"j_crank": (60, 120)}, frames=13)
    assert kin.overconstrained(["j_crank"]) == [] and check_study(kin, study) == []
    for pose in run_study(asm, kin, study).poses:
        assert pose.ok and pose.q["j_rocker"] == pytest.approx(pose.q["j_crank"], abs=1e-6)
        assert pose.q["j_rocker2"] == pytest.approx(pose.q["j_crank"], abs=1e-6)
    # over-driving it for real is still caught
    assert kin.overconstrained(["j_crank", "j_rocker"])


def test_spherical_four_bar_is_mobile():
    """All four hinge axes through one point (review p14): a 1-DOF spherical linkage, not overconstrained."""
    def u(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    R = 50.0
    a1, a4 = np.array([0.0, 0, 1]), u([math.sin(math.radians(60)), 0, math.cos(math.radians(60))])
    a2, a3 = u([0, math.sin(math.radians(30)), math.cos(math.radians(30))]), u([0.5, 0.5, 0.8])
    ball = lambda p: Pos(*(float(x) for x in p)) * Box(4, 4, 4)
    asm = Assembly("spherical")
    asm.part("frame", ball(R * a1 * 0.6) + ball(R * a4 * 0.6), ground=True)
    asm.part("crank", ball(R * a1 * 0.8) + ball(R * a2 * 0.8))
    asm.part("coupler", ball(R * a2) + ball(R * a3))
    asm.part("rocker", ball(R * a4 * 0.9) + ball(R * a3 * 0.9))
    asm.pin_tol = 60.0  # the sketch parts are only markers near the axes
    asm.revolute("j_crank", "frame", "crank", origin=(0, 0, 0), axis=tuple(a1))
    asm.revolute("j_coupler", "crank", "coupler", origin=(0, 0, 0), axis=tuple(a2))
    asm.revolute("j_rocker", "frame", "rocker", origin=(0, 0, 0), axis=tuple(a4))
    asm.pin("p3", "coupler", "rocker", point=tuple(R * a3), axis=tuple(a3))
    kin = Kinematics(asm)
    assert kin.mobility(["j_crank"]) == 0 and kin.overconstrained(["j_crank"]) == []
    pose = None
    for th in np.linspace(0, -60, 13):
        pose = kin.solve({"j_crank": th}, None if pose is None else pose.q)
        assert pose.ok
        # the coupler's far axis keeps its angles to the crank axis a2 and the rocker axis a4
        a2_now = pose.transforms["crank"][:3, :3] @ a2
        a3_now = pose.transforms["coupler"][:3, :3] @ a3
        assert a3_now @ a2_now == pytest.approx(a3 @ a2, abs=1e-9)
        assert a3_now @ a4 == pytest.approx(a3 @ a4, abs=1e-9)


def drag_link_follower_deg(theta: float) -> float:
    """Continuous follower angle change (deg) of the review p26 drag link at crank angle theta."""
    G, a, b, c = 25.0, 50.0, 60.0, 40.0
    B0 = circle_intersect((a, 0, 0), b, (G, 0, 0), c, side=+1)
    prev = math.degrees(math.atan2(B0[1], B0[0] - G))
    start = prev
    for th in np.linspace(0.0, theta, max(2, int(abs(theta) * 10) + 1))[1:]:
        A = (a * math.cos(math.radians(th)), a * math.sin(math.radians(th)), 0.0)
        B = circle_intersect(A, b, (G, 0, 0), c, side=+1)
        f = math.degrees(math.atan2(B[1], B[0] - G))
        prev += wrap180(f - prev)
    return prev - start


@pytest.mark.parametrize("frames", [3, 4, 5, 6])
def test_coarse_full_turn_unwraps_a_fast_passive_joint(frames):
    """A drag link's follower turns > 180° between two coarse frames (review p26): the value is the
    continuous one, not rewound by 360°, and no branch jump is reported."""
    from mech.assembly import Study
    from mech.motion import run_study

    asm = planar_four_bar(25, 50, 60, 40)
    kin = Kinematics(asm)
    res = run_study(asm, kin, Study("turn", {"j_crank": (0, 360)}, frames=frames))
    for theta, pose in zip(res.drive["j_crank"], res.poses):
        assert pose.ok and pose.q["j_rocker"] == pytest.approx(drag_link_follower_deg(theta), abs=1e-3)
    assert res.poses[-1].q["j_rocker"] == pytest.approx(360.0, abs=1e-3)
    assert res.branch_jumps == []
