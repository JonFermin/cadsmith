"""mech.motion: §1 frame sampling, §2.2 default studies, check_study errors, study runs, branch jumps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Pos

from conftest import FOUR_BAR, FOUR_BAR_LINKS, four_bar_rocker_deg, make_four_bar
from mech import Assembly, Kinematics
from mech.assembly import Study
from mech.geom import circle_intersect, link
from mech.motion import _branch_jumps, check_study, default_studies, run_study, sample_drive


def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------ sampling (§1)


def test_once_sampling_hits_both_ends():
    t, drive = sample_drive(Study("s", {"j": (10.0, 50.0)}, frames=5, loop="once", duration=2.0))
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-15)
    np.testing.assert_allclose(drive["j"], [10.0, 20.0, 30.0, 40.0, 50.0], atol=1e-12)
    assert drive["j"][0] == 10.0 and drive["j"][-1] == 50.0  # exact end points
    t, drive = sample_drive(Study("s", {"j": (0.1, 0.3)}, frames=7))
    assert drive["j"][-1] == 0.3 and t[-1] == 3.0


def test_pingpong_sampling_wraps_seamlessly():
    # u_k = 1 − |1 − 2k/N|, t_k = duration·k/N
    t, drive = sample_drive(Study("s", {"j": (0.0, 90.0)}, frames=6, loop="pingpong", duration=3.0))
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], atol=1e-15)
    np.testing.assert_allclose(drive["j"], 90.0 * np.array([0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3]), atol=1e-12)
    # odd frame count: the turn-around frame is skipped, the sequence still wraps to u = 0
    _, drive = sample_drive(Study("s", {"j": (0.0, 10.0)}, frames=5, loop="pingpong"))
    np.testing.assert_allclose(drive["j"], [0.0, 4.0, 8.0, 8.0, 4.0], atol=1e-12)


def test_keyframes_callables_and_constants():
    study = Study("s", {
        "tri": [(1.0, 0.0), (0.0, 0.0), (0.5, 90.0)],  # unsorted keyframes are sorted by u
        "fn": lambda u: 30.0 * math.sin(2 * math.pi * u),
        "hold": 12.5,
        "list": [0, 360],
    }, frames=5)
    _, drive = sample_drive(study)
    np.testing.assert_allclose(drive["tri"], [0.0, 45.0, 90.0, 45.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(drive["fn"], [30.0 * math.sin(2 * math.pi * u) for u in (0, 0.25, 0.5, 0.75, 1.0)])
    np.testing.assert_allclose(drive["hold"], 12.5)
    np.testing.assert_allclose(drive["list"], [0.0, 90.0, 180.0, 270.0, 360.0])
    # keyframes that don't span [0, 1] hold their end values
    _, drive = sample_drive(Study("s", {"j": [(0.25, 10.0), (0.75, 20.0)]}, frames=5))
    np.testing.assert_allclose(drive["j"], [10.0, 10.0, 15.0, 20.0, 20.0])


@pytest.mark.parametrize("study", [
    Study("s", {"j": "abc"}),
    Study("s", {"j": (0.0, 1.0, 2.0)}),
    Study("s", {"j": lambda u: float("nan")}),
    Study("s", {"j": (0, 1)}, frames=1),
    Study("s", {"j": (0, 1)}, loop="bounce"),
])
def test_unsamplable_studies_raise_value_error(study):
    with pytest.raises(ValueError):
        sample_drive(study)


# ------------------------------------------------------------------------------ default studies (§2.2)


def _pendulum_model() -> Assembly:
    """Four-bar (loop) plus serial joints of every kind on a second base."""
    asm = make_four_bar(**FOUR_BAR)
    asm.studies.clear()
    asm.part("post", Pos(0, 100, 0) * Box(10, 10, 10), ground=True)
    asm.part("swing", Pos(20, 100, 10) * Box(30, 4, 4))
    asm.part("spinner", Pos(0, 120, 10) * Box(8, 8, 4))
    asm.part("slide", Pos(0, 140, 10) * Box(8, 8, 4))
    asm.part("follower", Pos(0, 160, 10) * Box(8, 8, 4))
    asm.part("bolt", Pos(0, 100, 7) * Box(2, 2, 4))
    asm.revolute("j_swing", "post", "swing", origin=(0, 100, 10), axis=(0, 1, 0), limits=(-30, 45))
    asm.revolute("j_spin", "post", "spinner", origin=(0, 120, 10), axis=(0, 0, 1))
    asm.prismatic("j_slide", "post", "slide", origin=(0, 140, 10), axis=(1, 0, 0))  # no limits: can't sweep
    asm.prismatic("j_follow", "post", "follower", origin=(0, 160, 10), axis=(1, 0, 0))
    asm.couple("j_spin", "j_follow", ratio=0.1)  # coupled: moves with j_spin, never driven itself
    asm.fix("bolt", "post")
    return asm


def test_default_studies_per_mobility_group():
    asm = _pendulum_model()
    kin = Kinematics(asm)
    studies = default_studies(asm, kin)
    # ordered by the declaration of each study's first driver: j_crank (loop), j_swing, j_spin
    assert [(s.name, s.drive, s.frames, s.loop) for s in studies] == [
        ("sweep_j_crank", {"j_crank": (0.0, 360.0)}, 60, "once"),  # the loop's actuated joint
        ("sweep_j_swing", {"j_swing": (-30.0, 45.0)}, 40, "once"),  # limits
        ("sweep_j_spin", {"j_spin": (0.0, 360.0)}, 40, "once"),  # revolute without limits: full turn
    ]
    assert all(check_study(kin, s) == [] for s in studies)
    assert kin.mobility(["j_crank"]) == 0  # the loop study closes the only loop
    # the serial studies leave the (undriven) loop at home: global mobility counts it
    assert kin.mobility(["j_swing"]) == kin.mobility(["j_spin"]) == 1
    # default_studies ignores declared studies (the runner decides when to use them)
    asm.study("mine", {"j_swing": (0, 10)})
    assert [s.name for s in default_studies(asm, kin)] == [s.name for s in studies]


def test_default_loop_driver_priority():
    asm = make_four_bar(**FOUR_BAR)
    asm.studies.clear()
    asm.actuators.clear()
    kin = Kinematics(asm)
    # no actuator, no limits: the first-declared revolute loop joint, full turn
    assert [(s.name, s.drive) for s in default_studies(asm, kin)] == [("sweep_j_crank", {"j_crank": (0.0, 360.0)})]
    # a loop joint with limits beats a plain revolute
    asm.joints["j_rocker"].limits = (-20.0, 30.0)
    assert [s.drive for s in default_studies(asm, Kinematics(asm))] == [{"j_rocker": (-20.0, 30.0)}]
    # an actuator beats limits
    asm.actuator("j_coupler", capacity=1.0)
    assert [s.drive for s in default_studies(asm, Kinematics(asm))] == [{"j_coupler": (0.0, 360.0)}]


def test_default_study_drives_every_dof_of_a_five_bar():
    """A 2-DOF five-bar gets one driver per degree of freedom (both actuated cranks)."""
    O1, O2 = (0.0, 0.0, 0.0), (60.0, 0.0, 0.0)
    A1, A2 = (0.0, 30.0, 0.0), (60.0, 30.0, 0.0)
    P = circle_intersect(A1, 50.0, A2, 50.0, side=-1)  # above the cranks
    asm = Assembly("five_bar")
    asm.part("frame", Pos(30, 0, -8) * Box(90, 12, 5), ground=True)
    asm.part("crank1", link(O1, A1, 8, 4, z=0))
    asm.part("crank2", link(O2, A2, 8, 4, z=0))
    asm.part("link1", link(A1, P, 8, 4, z=4.5))
    asm.part("link2", link(A2, P, 8, 4, z=9))
    asm.revolute("j1", "frame", "crank1", origin=O1, axis=(0, 0, 1), limits=(-20, 20))
    asm.revolute("j2", "frame", "crank2", origin=O2, axis=(0, 0, 1), limits=(-20, 20))
    asm.revolute("j3", "crank1", "link1", origin=A1, axis=(0, 0, 1))
    asm.revolute("j4", "crank2", "link2", origin=A2, axis=(0, 0, 1))
    asm.pin("p", "link1", "link2", point=P, axis=(0, 0, 1))
    asm.actuator("j1", 1.0)
    asm.actuator("j2", 1.0)
    kin = Kinematics(asm)
    assert kin.mobility([]) == 2
    (study,) = default_studies(asm, kin)
    assert study.drive == {"j1": (-20.0, 20.0), "j2": (-20.0, 20.0)} and study.name == "sweep_j1+j2"
    assert kin.mobility(list(study.drive)) == 0 and check_study(kin, study) == []


# ------------------------------------------------------------------------------ check_study


def _leadscrew() -> Assembly:
    asm = Assembly("leadscrew")
    asm.part("frame", Pos(0, 0, -5) * Box(40, 40, 10), ground=True)
    asm.part("screw", Pos(0, 0, 50) * Box(8, 8, 100))
    asm.part("nut", Pos(0, 0, 30) * Box(20, 20, 10))
    asm.revolute("j_rot", "frame", "screw", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_nut", "frame", "nut", origin=(0, 0, 30), axis=(0, 0, 1), limits=(-50, 50))
    asm.screw("j_rot", "j_nut", lead=8)
    return asm


def test_check_study_accepts_valid_studies(four_bar):
    kin = Kinematics(four_bar)
    assert check_study(kin, four_bar.studies[0]) == []
    assert check_study(kin, Study("pp", {"j_rocker": [(0, 0), (0.5, 10), (1, 0)]}, loop="pingpong")) == []


def test_check_study_reports_every_problem(four_bar):
    four_bar.joints["j_crank"].limits = (0.0, 90.0)
    kin = Kinematics(four_bar)
    errors = check_study(kin, Study("bad", {
        "j_crnk": (0, 10),  # unknown, with suggestion
        "j_crank": (0, 120),  # beyond its limits
        "j_rocker": [(0, 0), (1.5, 3)],  # keyframe outside [0, 1]
        "j_coupler": lambda u: 1 / 0,  # raising callable: reported, never propagated
    }, frames=1, loop="bounce", duration=0.0))
    text = "\n".join(errors)
    assert "frames must be an integer >= 2 (got 1)" in text
    assert "duration must be > 0 s" in text
    assert "unknown joint 'j_crnk' (did you mean 'j_crank'?)" in text
    assert "keyframes for 'j_rocker' need u in [0, 1] (got u = 1.5)" in text
    assert all(e.startswith("study 'bad': ") for e in errors)
    # with valid sampling the limit and callable problems are found too
    errors = check_study(kin, Study("bad", {"j_crank": (0, 120), "j_coupler": lambda u: 1 / 0}))
    text = "\n".join(errors)
    assert "'j_crank' is driven over 0…120°, outside its limits [0, 90]°" in text
    assert "drive for 'j_coupler' failed: ZeroDivisionError" in text
    assert "overconstrained drivers" in text  # j_crank + j_coupler both drive the 1-DOF loop


def test_check_study_rejects_coupled_fixed_and_overconstrained(four_bar):
    asm = _leadscrew()
    asm.part("cap", Pos(0, 0, -12) * Box(10, 10, 4))
    asm.fix("cap", "frame")
    kin = Kinematics(asm)
    errors = check_study(kin, Study("s", {"j_nut": (0, 10), "fix_cap": (0, 1)}))
    assert errors == [
        "study 's': joint 'j_nut' is driven by screw 'screw_j_nut' — drive its driver 'j_rot' instead",
        "study 's': joint 'fix_cap' is fixed and can't be driven",
    ]
    # the driver's own sweep can't push the coupled joint past its limits here: that is kinematics' job
    assert check_study(kin, Study("s", {"j_rot": (0, 360)})) == []
    kin4 = Kinematics(four_bar)
    (msg,) = check_study(kin4, Study("s", {"j_crank": (0, 10), "j_rocker": (0, 5)}))
    assert msg.startswith("study 's': overconstrained drivers: j_crank, j_rocker") and "'p_B'" in msg


# ------------------------------------------------------------------------------ run_study


def test_run_study_four_bar(four_bar):
    kin = Kinematics(four_bar)
    res = run_study(four_bar, kin, four_bar.studies[0])
    assert len(res.poses) == 72 and res.study is four_bar.studies[0]
    np.testing.assert_allclose(res.t, 3.0 * np.arange(72) / 71)
    np.testing.assert_allclose(res.drive["j_crank"], 360.0 * np.arange(72) / 71)
    home = four_bar_rocker_deg(0.0, **FOUR_BAR_LINKS)
    for theta, pose in zip(res.drive["j_crank"], res.poses):
        assert pose.ok and pose.q["j_crank"] == theta
        assert abs(wrap180(pose.q["j_rocker"] - (four_bar_rocker_deg(theta, **FOUR_BAR_LINKS) - home))) < 1e-3
    probe = four_bar.probes[0]
    assert res.probes["mid"].shape == (72, 3)
    np.testing.assert_allclose(res.probes["mid"][0], probe.point, atol=1e-9)  # frame 0 is the drawn pose
    for k in (17, 50):
        np.testing.assert_allclose(res.probes["mid"][k], kin.point(res.poses[k], "coupler", probe.point))
    assert res.branch_jumps == []


def test_run_study_pingpong_returns_to_start(four_bar):
    kin = Kinematics(four_bar)
    res = run_study(four_bar, kin, Study("pp", {"j_crank": (0, 90)}, frames=8, loop="pingpong"))
    q = [p.q["j_crank"] for p in res.poses]
    np.testing.assert_allclose(q, [0, 22.5, 45, 67.5, 90, 67.5, 45, 22.5], atol=1e-12)
    # the way back retraces the way out (same branch, same passive values)
    for k in (1, 2, 3):
        assert res.poses[k].q["j_rocker"] == pytest.approx(res.poses[8 - k].q["j_rocker"], abs=1e-9)
    assert res.branch_jumps == []


def test_coarse_study_is_not_a_branch_jump(four_bar):
    """60° crank steps swing the passive joints by far more than 20°, all explained by the tangent."""
    kin = Kinematics(four_bar)
    res = run_study(four_bar, kin, Study("coarse", {"j_crank": (0, 360)}, frames=7))
    steps = np.abs(np.diff([p.q["j_rocker"] for p in res.poses]))
    assert steps.max() > 30.0 and all(p.ok for p in res.poses)
    assert res.branch_jumps == []


def test_branch_jump_detected(four_bar):
    """A frame solved onto the crossed assembly branch is flagged for both passive joints."""
    kin = Kinematics(four_bar)
    a, b, c, d = (FOUR_BAR[k] for k in ("crank", "coupler", "rocker", "ground"))
    t2 = math.radians(10.0)
    # crossed branch: the other root of Freudenstein's equation
    k1, k2, k3 = d / a, d / c, (a * a - b * b + c * c + d * d) / (2 * a * c)
    P, Q, R = k1 - math.cos(t2), -math.sin(t2), k2 * math.cos(t2) - k3
    t4x = math.atan2(Q, P) - math.acos(R / math.hypot(P, Q))
    A = np.array([a * math.cos(t2), a * math.sin(t2)])
    Bx = np.array([d + c * math.cos(t4x), c * math.sin(t4x)])
    B0 = np.array(circle_intersect((a, 0, 0), b, (d, 0, 0), c, side=+1)[:2])
    t3_home = math.atan2(B0[1], B0[0] - a)
    t4_home = math.radians(four_bar_rocker_deg(0.0, **FOUR_BAR_LINKS))
    guess = {"j_crank": 10.0,
             "j_coupler": wrap180(math.degrees(math.atan2(*(Bx - A)[::-1]) - t3_home) - 10.0),
             "j_rocker": wrap180(math.degrees(t4x - t4_home))}
    open5 = kin.solve({"j_crank": 5.0})
    crossed = kin.solve({"j_crank": 10.0}, guess)
    assert open5.ok and crossed.ok
    assert crossed.q["j_rocker"] == pytest.approx(guess["j_rocker"], abs=1e-6)  # really on the other branch
    jumps = _branch_jumps(kin, ["j_crank"], [open5, crossed])
    assert [(k, j) for k, j, _ in jumps] == [(1, "j_coupler"), (1, "j_rocker")]
    for _, j, size in jumps:
        assert size == pytest.approx(abs(crossed.q[j] - open5.q[j])) and size > 20.0
    # the same frames in branch-consistent order are fine
    assert _branch_jumps(kin, ["j_crank"], [open5, kin.solve({"j_crank": 10.0}, open5.q)]) == []
