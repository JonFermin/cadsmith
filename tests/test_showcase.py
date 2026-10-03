"""examples/showcase/*.py: the six showcase mechanisms stay valid and keep their verified numbers.

Default suite: every showcase passes ``mech check`` (validation, roles/mobility, home-pose
clearance) with the framework defaults — no raised ``pin_tol`` and no ``ignore()``d pairs, the
workarounds the framework made unnecessary. ``-m slow`` / ``--runslow``: a full analysis of each
(every study, clearance sweep, loads, targets) must PASS and reproduce the headline numbers its
hand check verified (README "Showcase").
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from mech.runner import analyze, check

SHOWCASE = Path(__file__).resolve().parents[1] / "examples" / "showcase"
NAMES = ["strandbeest", "radial_engine", "planetary", "desktop_arm", "excavator", "delta_robot"]


def showcase(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_showcase_{name}", SHOWCASE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_showcase_passes_mech_check(name):
    asm = showcase(name).build()
    assert asm.name == name
    assert asm.pin_tol == 5.0 and not asm.ignored  # framework defaults: no workarounds left
    report = check(asm, progress=False)
    assert report["status"] == "PASS", [i["message"] for i in report["issues"] if i["severity"] != "INFO"]


# ------------------------------------------------------------------------------ full runs (slow)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """name -> report of a full analysis (no export), computed once per module."""
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            asm = showcase(name).build()
            cache[name] = analyze(asm, export=False, out_root=tmp_path_factory.mktemp(name), progress=False)
            issues = [i["message"] for i in cache[name]["issues"] if i["severity"] != "INFO"]
            assert cache[name]["status"] == "PASS", issues
            assert all(t["met"] for t in cache[name]["targets"]), cache[name]["targets"]
        return cache[name]

    return get


def target(report: dict, metric_or_label: str) -> float:
    return next(t["value"] for t in report["targets"] if metric_or_label in (t["metric"], t["label"]))


def load(report: dict, study: str, joint: str) -> float:
    return next(s for s in report["studies"] if s["name"] == study)["loads"][joint]["max_abs"]


@pytest.mark.slow
def test_strandbeest_stride_and_lift(full):
    r = full("strandbeest")
    assert target(r, "delta:foot_L0.x") == pytest.approx(67.847, abs=5e-3)   # stride
    assert target(r, "delta:foot_L0.z") == pytest.approx(22.457, abs=5e-3)   # step lift


@pytest.mark.slow
def test_radial_engine_strokes(full):
    r = full("radial_engine")
    strokes = [target(r, f"span:j_pist{k}") for k in range(5)]
    assert strokes == pytest.approx([44.000, 44.049, 44.156, 44.156, 44.049], abs=5e-4)


@pytest.mark.slow
def test_planetary_ratio_and_sun_torque(full):
    r = full("planetary")
    assert target(r, "ratio 25:1") == pytest.approx(25.0, abs=1e-6)
    assert load(r, "turn", "j_sun") == pytest.approx(0.1482, abs=5e-5)       # N·m: output torque / 25


@pytest.mark.slow
def test_desktop_arm_shoulder_load(full):
    r = full("desktop_arm")
    assert load(r, "shoulder", "j_shoulder") == pytest.approx(0.588, abs=5e-4)   # N·m at the 80T pulley


@pytest.mark.slow
def test_excavator_boom_pair_force_at_home(full):
    r = full("excavator")
    # the swing study holds the boom at home (max reach, tip on the ground) in every frame
    assert load(r, "swing", "j_cyl_boom_l") == pytest.approx(340.4, abs=0.05)   # N, the cylinder PAIR


@pytest.mark.slow
def test_delta_robot_max_motor_load(full):
    r = full("delta_robot")
    worst = max(ld["max_abs"] for s in r["studies"] for j, ld in s["loads"].items() if j.startswith("j_arm_"))
    assert worst == pytest.approx(0.233, abs=5e-4)                               # N·m
