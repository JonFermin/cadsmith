"""examples/*.py against their analytic truths (spec §8) — these gate "done".

Statuses, issues, targets and report-level loads come from ``mech.runner.analyze`` (no export);
per-frame truths come from the same pipeline stages (Kinematics → run_study → gravity_loads).
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from conftest import four_bar_rocker_deg
from mech import Kinematics, part_props
from mech.motion import run_study
from mech.report import format_summary
from mech.runner import analyze, check
from mech.statics import gravity_loads

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
G = 9.80665
NAMES = ["four_bar", "slider_crank", "gear_train", "leadscrew_stage", "pendulum_arm", "hinged_box",
         "parallel_gripper", "scissor_lift"]
MAX_LINES = {"parallel_gripper": 60, "scissor_lift": 60}  # promoted dogfood designs; the rest ≤ 50


def example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_of(tmp_path_factory, name: str, **params) -> dict:
    """analyze() without export; out_root is a temp dir so no previous report.json leaks into Δprev."""
    asm = example(name).build(**params)
    return analyze(asm, export=False, out_root=tmp_path_factory.mktemp(name), params=params)


def solved(name: str, **params):
    """(asm, kin, StudyResult of the script's first study)."""
    asm = example(name).build(**params)
    kin = Kinematics(asm)
    return asm, kin, run_study(asm, kin, asm.studies[0])


def codes(report: dict, severity: str | None = None) -> list[str]:
    return [i["code"] for i in report["issues"] if severity is None or i["severity"] == severity]


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("name", NAMES)
def test_example_is_a_short_parametric_template(name):
    text = (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= MAX_LINES.get(name, 50)
    assert 'if __name__ == "__main__":\n    run(build())' in text
    asm = example(name).build()  # every tunable has a keyword default
    assert asm.validate() == []


def test_four_bar_is_byte_identical_to_the_spec_and_the_skill_reference():
    code = (EXAMPLES / "four_bar.py").read_text(encoding="utf-8")
    for doc in (ROOT / "docs" / "MECH_SPEC.md", ROOT / ".claude" / "skills" / "make-mechanism" / "references"
                / "mech-api.md"):
        block = re.search(r"Canonical .*?```python\n(.*?)```", doc.read_text(encoding="utf-8"), re.S).group(1)
        assert block == code, doc.name


def test_four_bar_follows_freudenstein(tmp_path_factory):
    _, _, res = solved("four_bar")
    home = four_bar_rocker_deg(0.0)
    for pose in res.poses:
        exact = four_bar_rocker_deg(pose.q["j_crank"]) - home
        assert abs(wrap180(pose.q["j_rocker"] - exact)) < 1e-3
        assert pose.ok and pose.residual < 1e-6
    report = report_of(tmp_path_factory, "four_bar")
    assert report["status"] == "PASS"
    (target,) = report["targets"]
    rocker = [four_bar_rocker_deg(t) for t in res.drive["j_crank"]]  # the sampled crank angles
    assert target["met"] and target["value"] == pytest.approx(max(rocker) - min(rocker), rel=1e-5)
    assert len(format_summary(report).splitlines()) <= 15


def test_slider_crank_position(tmp_path_factory):
    r, l = 30.0, 90.0
    _, _, res = solved("slider_crank")
    for k, pose in enumerate(res.poses):
        th = math.radians(pose.q["j_crank"])
        x = r * math.cos(th) + math.sqrt(l * l - (r * math.sin(th)) ** 2)
        assert abs(pose.q["j_slider"] - x) < 1e-4
        assert abs(res.probes["wrist"][k][0] - x) < 1e-4
    report = report_of(tmp_path_factory, "slider_crank")
    assert report["status"] == "PASS" and report["targets"][0]["met"]
    lo, hi = report["studies"][0]["joint_ranges"]["j_slider"]
    assert (lo, hi) == (pytest.approx(l - r, abs=1e-6), pytest.approx(l + r, abs=1e-6))  # 5° steps hit 0° and 180°


def test_gear_train_ratio_and_mesh(tmp_path_factory):
    _, _, res = solved("gear_train")
    for pose in res.poses:
        assert pose.q["j_out"] == pytest.approx(-0.5 * pose.q["j_in"], abs=1e-9)
    report = report_of(tmp_path_factory, "gear_train")
    assert report["status"] == "PASS"
    assert not {"interference", "tight_clearance", "static_interference"} & set(codes(report))
    mesh = [i for i in report["issues"] if i["code"] == "gear_mesh"]
    assert len(mesh) == 1 and mesh[0]["severity"] == "INFO" and sorted(mesh[0]["parts"]) == ["gear_out", "pinion"]
    assert report["studies"][0]["joint_ranges"]["j_out"] == [pytest.approx(-360.0), pytest.approx(0.0)]
    # `mech check` runs no study, but j_out follows the study driver j_in: it is not held at home
    checked = check(example("gear_train").build())
    assert checked["status"] == "PASS" and "held_at_home" not in codes(checked)


def test_leadscrew_stage_travel_and_loads(tmp_path_factory):
    asm, kin, res = solved("leadscrew_stage")
    # right-hand thread, co-directional axes: +360° of screw moves the carriage −8 mm
    assert kin.expand({"j_leadrot": 360.0})["j_carriage"] == pytest.approx(-8.0, abs=1e-12)
    for pose in res.poses:
        assert pose.q["j_carriage"] == pytest.approx(-8.0 / 360.0 * pose.q["j_leadrot"], abs=1e-9)
    props = {n: part_props(p) for n, p in asm.parts.items()}
    m = sum(props[n].mass_kg for n in ("carriage", "nut", "bush0", "bush1"))  # everything moving with the carriage
    torque, force = m * G * 0.008 / (2 * math.pi), m * G
    loads = gravity_loads(asm, kin, res, props)
    np.testing.assert_allclose(loads["j_leadrot"]["series"], -torque, rtol=1e-4)  # sign: holds against −θ lift
    np.testing.assert_allclose(loads["j_carriage"]["series"], force, rtol=1e-4)
    assert loads["j_carriage"]["reflected_from"] == "j_leadrot"
    report = report_of(tmp_path_factory, "leadscrew_stage")
    assert report["status"] == "PASS" and report["targets"][0]["met"]
    study = report["studies"][0]
    assert study["loads"]["j_leadrot"]["max_abs"] == pytest.approx(torque, rel=1e-2)  # spec: within 1%
    assert study["loads"]["j_carriage"]["max_abs"] == pytest.approx(force, rel=1e-2)
    assert study["loads"]["j_leadrot"]["frame"] == 0  # constant load: the first frame


def test_pendulum_arm_holding_torque(tmp_path_factory):
    asm, kin, res = solved("pendulum_arm")
    props = {n: part_props(p) for n, p in asm.parts.items()}
    moving = [props["arm"], props["payload"]]
    m = sum(p.mass_kg for p in moving)
    x, z = (sum(p.mass_kg * p.com[i] for p in moving) / m * 1e-3 for i in (0, 2))  # COM (m), axis through origin
    r = math.hypot(x, z)
    assert abs(z) < 1e-9 and props["payload"].mass_kg == pytest.approx(0.1)  # mass_g override
    series = gravity_loads(asm, kin, res, props)["j_shoulder"]["series"]
    for theta, tau in zip(res.drive["j_shoulder"], series):
        exact = m * G * r * math.cos(math.radians(theta))  # +θ raises the arm, so holding is positive
        assert abs(tau - exact) <= 1e-4 * m * G * r  # spec bound is 0.01·m·g·r
    report = report_of(tmp_path_factory, "pendulum_arm")
    assert report["status"] == "PASS" and report["targets"][0]["met"]
    assert report["studies"][0]["loads"]["j_shoulder"]["max_abs"] == pytest.approx(m * G * r, rel=1e-4)


@pytest.mark.parametrize("capacity, severity, status", [(0.1, "FAIL", "FAIL"), (0.18, "WARN", "FAIL")])
def test_pendulum_arm_over_capacity(tmp_path_factory, capacity, severity, status):
    # 0.18 N·m: SF ≈ 1.4 < 1.5 → over_capacity WARN, and the SF ≥ 2 target misses (FAIL)
    report = report_of(tmp_path_factory, "pendulum_arm", capacity=capacity)
    over = [i for i in report["issues"] if i["code"] == "over_capacity"]
    assert [i["severity"] for i in over] == [severity]
    assert report["status"] == status and "target_miss" in codes(report, "FAIL")


def test_hinged_box_default_gap_is_tight(tmp_path_factory):
    report = report_of(tmp_path_factory, "hinged_box")
    assert report["status"] == "WARN"
    assert codes(report, "FAIL") == [] and codes(report, "WARN") == ["tight_clearance"]
    (tight,) = [i for i in report["issues"] if i["code"] == "tight_clearance"]
    assert sorted(tight["parts"]) == ["lid_left", "lid_right"]
    assert tight["value"] == pytest.approx(0.2, abs=1e-6) and tight["frame"] == 0


def test_hinged_box_negative_gap_interferes(tmp_path_factory):
    depth, wall = 50.0, 2.0
    report = report_of(tmp_path_factory, "hinged_box", gap=-1.0)
    assert report["status"] == "FAIL"
    worst = next(i for i in report["issues"] if i["code"] == "interference")
    assert sorted(worst["parts"]) == ["lid_left", "lid_right"] and worst["frame"] == 0
    assert worst["value"] == pytest.approx(1.0 * depth * wall, rel=0.02)  # spec: within 2% of analytic
    np.testing.assert_allclose(worst["extent"], [1.0, depth, wall], atol=1e-6)
    np.testing.assert_allclose(worst["location"], [0.0, 0.0, 30.0 + wall / 2], atol=1e-6)


def test_parallel_gripper_jaws_translate_and_open_to_the_gap(tmp_path_factory):
    """Gear pair ratio −1; each jaw rides a parallelogram, so it never turns (j_jaw = −j_drive); the
    pads meet when drawn closed and are exactly gap_open apart at the end of the stroke."""
    asm, kin, res = solved("parallel_gripper")
    for pose in res.poses:
        assert pose.ok
        assert pose.q["j_idler"] == pytest.approx(-pose.q["j_drive"], abs=1e-9)
        assert pose.q["j_jaw_l"] == pytest.approx(-pose.q["j_drive"], abs=1e-6)
        assert pose.q["j_fol_l"] == pytest.approx(pose.q["j_drive"], abs=1e-6)
    gap = np.linalg.norm(np.asarray(res.probes["pad_l"]) - np.asarray(res.probes["pad_r"]), axis=1)
    assert gap[0] == pytest.approx(0.0, abs=1e-6) and gap[-1] == pytest.approx(30.0, abs=1e-6)
    report = report_of(tmp_path_factory, "parallel_gripper")
    assert report["status"] == "PASS" and all(t["met"] for t in report["targets"])
    tilt = next(t for t in report["targets"] if t["label"] == "jaws parallel")
    assert tilt["value"] <= 1e-9
    assert "no gravity load on j_drive (axis ∥ g or balanced)" in format_summary(report)
    # the opening lands on its bound by construction: float noise must not turn that into a miss
    for L, open_deg in ((35.0, 110.0), (30.0, 110.0)):
        variant = report_of(tmp_path_factory, "parallel_gripper", L=L, open_deg=open_deg)
        opening = next(t for t in variant["targets"] if t["label"] == "opening")
        assert opening["value"] == pytest.approx(30.0, abs=1e-6) and opening["met"], (L, open_deg)
        assert variant["status"] == "PASS"


def test_scissor_lift_travel_and_holding_load(tmp_path_factory):
    """Deck height 2·L·sin θ (+ const), slide x = L·cos θ, right-hand T8 screw: −8 mm per +360°.
    With a payload that dwarfs the linkage, virtual work gives the slide force 2·W·cot θ (max at the
    drawn θ0) and the screw torque F·lead/(2π)."""
    L, th0, th1 = 120.0, 10.0, 40.0
    asm, kin, res = solved("scissor_lift")
    for pose in res.poses:
        assert pose.ok and pose.q["j_slide"] == pytest.approx(-8.0 / 360.0 * pose.q["j_leadrot"], abs=1e-9)
    report = report_of(tmp_path_factory, "scissor_lift")
    assert report["status"] == "PASS" and all(t["met"] for t in report["targets"])
    rise = 2 * L * (math.sin(math.radians(th1)) - math.sin(math.radians(th0)))
    assert np.ptp(np.asarray(res.probes["deck"])[:, 2]) == pytest.approx(rise, abs=1e-6)
    travel = next(t for t in report["targets"] if t["label"] == "travel")
    assert travel["value"] == pytest.approx(rise, rel=5e-6)  # report numbers carry 6 significant figures
    lo, hi = report["studies"][0]["joint_ranges"]["j_slide"]
    assert hi - lo == pytest.approx(L * (math.cos(math.radians(th0)) - math.cos(math.radians(th1))), rel=5e-6)
    heavy = report_of(tmp_path_factory, "scissor_lift", payload_g=100_000.0)
    force = 2 * 100.0 * G / math.tan(math.radians(th0))  # N on the slide at θ0
    loads = heavy["studies"][0]["loads"]
    assert loads["j_slide"]["max_abs"] == pytest.approx(force, rel=1e-2) and loads["j_slide"]["frame"] == 0
    assert loads["j_leadrot"]["max_abs"] == pytest.approx(force * 0.008 / (2 * math.pi), rel=1e-2)
    assert heavy["status"] == "FAIL" and "over_capacity" in codes(heavy, "FAIL")
