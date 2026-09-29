"""targets.py: the §2.3 metric vocabulary, worst-over-studies rule and edge cases.

Synthetic study results pin down the bookkeeping exactly; a slider-crank and a gravity-loaded
pendulum check real runs against closed-form values.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from build123d import Box, Pos

from mech import Assembly
from mech.assembly import Target
from mech.runner import analyze
from mech.targets import evaluate_targets, metric_name

from conftest import SLIDER_CRANK, make_slider_crank


def _result(name, rocker, probes=None, ok=None):
    """A StudyResult stand-in: j_rocker follows ``rocker`` per frame."""
    ok = ok or [True] * len(rocker)
    poses = [SimpleNamespace(q={"j_crank": 0.0, "j_coupler": 0.0, "j_rocker": float(v)}, ok=o)
             for v, o in zip(rocker, ok)]
    return SimpleNamespace(study=SimpleNamespace(name=name), poses=poses,
                           probes={k: np.asarray(v, dtype=float) for k, v in (probes or {}).items()})


def _report(studies=("a", "b"), clearance=None, loads=None, mass=38.2):
    return {"mass": {"total_g": mass},
            "studies": [{"name": s, "min_clearance": (clearance or {}).get(s), "loads": (loads or {}).get(s, {})}
                        for s in studies]}


def _eval(asm, report, results, *targets):
    asm.targets = [Target(label, metric, mn, mx, study, sev) for label, metric, mn, mx, study, sev in targets]
    return {row["label"]: row for row in evaluate_targets(asm, report, results)}


@pytest.fixture
def two_studies():
    return {"a": _result("a", [0.0, 4.0, 10.0]), "b": _result("b", [-5.0, 30.0, 12.0])}


def test_joint_metrics_take_the_worst_study(four_bar, two_studies):
    rows = _eval(four_bar, _report(), two_studies,
                 ("span lo", "span:j_rocker", 20.0, None, None, "FAIL"),  # spans a=10, b=35
                 ("span hi", "span:j_rocker", None, 30.0, None, "WARN"),
                 ("lowest", "min:j_rocker", -10.0, None, None, "FAIL"),  # minima a=0, b=−5
                 ("highest", "max:j_rocker", None, 40.0, None, "FAIL"),  # maxima a=10, b=30
                 ("band", "span:j_rocker", 5.0, 40.0, None, "FAIL"),  # margins 5 and 5: first wins
                 ("only b", "span:j_rocker", 20.0, None, "b", "FAIL"))
    assert (rows["span lo"]["value"], rows["span lo"]["met"], rows["span lo"]["worst_study"]) == (10.0, False, "a")
    assert (rows["span hi"]["value"], rows["span hi"]["met"], rows["span hi"]["worst_study"]) == (35.0, False, "b")
    assert rows["span hi"]["severity"] == "WARN"
    assert (rows["lowest"]["value"], rows["lowest"]["met"], rows["lowest"]["worst_study"]) == (-5.0, True, "b")
    assert (rows["highest"]["value"], rows["highest"]["met"], rows["highest"]["worst_study"]) == (30.0, True, "b")
    assert (rows["band"]["value"], rows["band"]["met"], rows["band"]["worst_study"]) == (10.0, True, "a")
    assert (rows["only b"]["value"], rows["only b"]["met"], rows["only b"]["study"]) == (35.0, True, "b")
    assert set(rows["band"]) == {"label", "metric", "value", "min", "max", "met", "severity", "study",
                                 "worst_study", "error"}


def test_probe_metrics(four_bar):
    # p: (0,0,0) → (3,4,0) → (3,4,12); q sits at the origin; frame 3 is open and must be ignored
    p = [[0, 0, 0], [3, 4, 0], [3, 4, 12], [500, 500, 500]]
    q = [[0, 0, 0]] * 4
    res = {"a": _result("a", [0, 0, 0, 0], {"p": p, "q": q}, ok=[True, True, True, False])}
    rows = _eval(four_bar, _report(["a"]), res,
                 ("path", "path:p", None, None, None, "FAIL"),
                 ("dz", "delta:p.z", None, None, None, "FAIL"),
                 ("dx", "delta: p.x", None, None, None, "FAIL"),
                 ("near", "min_dist:p,q", None, None, None, "FAIL"),
                 ("far", "max_dist:p, q", None, 10.0, None, "FAIL"))
    assert rows["path"]["value"] == pytest.approx(17.0)  # 5 + 12
    assert rows["dz"]["value"] == pytest.approx(12.0)
    assert rows["dx"]["value"] == pytest.approx(3.0)
    assert rows["near"]["value"] == pytest.approx(0.0)
    assert rows["far"]["value"] == pytest.approx(13.0) and rows["far"]["met"] is False
    assert rows["path"]["met"] is True  # no bounds: reported only


def test_report_based_metrics(four_bar):
    loads = {"a": {"j_crank": {"unit": "N·m", "max_abs": 0.25, "frame": 3, "capacity": 0.5, "sf": 2.0}},
             "b": {"j_crank": {"unit": "N·m", "max_abs": 0.0, "frame": 0, "capacity": 0.5, "sf": None},
                   "j_rocker": {"unit": "N·m", "max_abs": 0.1, "frame": 0, "capacity": None, "sf": None}}}
    clear = {"a": {"parts": ["x", "y"], "value": 0.8, "frame": 2}, "b": {"parts": ["x", "y"], "value": 0.2, "frame": 5}}
    res = {"a": _result("a", [0.0, 1.0]), "b": _result("b", [0.0, 1.0])}
    rows = _eval(four_bar, _report(clearance=clear, loads=loads, mass=38.25), res,
                 ("gap", "clearance", 0.3, None, None, "FAIL"),
                 ("load", "load:j_crank", None, 0.3, None, "FAIL"),
                 ("sf", "sf:j_crank", 1.5, None, None, "FAIL"),
                 ("sf b", "sf:j_crank", 1.5, None, "b", "FAIL"),
                 ("sf free", "sf:j_rocker", 1.5, None, "b", "FAIL"),
                 ("mass", "mass_g", None, 40.0, None, "FAIL"),
                 ("f", lambda r: r["mass"]["total_g"] / 2, 20.0, None, None, "FAIL"),
                 ("boom", lambda r: 1 / 0, None, None, None, "WARN"))
    assert (rows["gap"]["value"], rows["gap"]["met"], rows["gap"]["worst_study"]) == (0.2, False, "b")
    assert (rows["load"]["value"], rows["load"]["met"]) == (0.25, True)
    assert (rows["sf"]["value"], rows["sf"]["met"], rows["sf"]["worst_study"]) == (2.0, True, "a")
    # zero load: SF is unbounded — meets any min, JSON value null with the reason
    assert (rows["sf b"]["value"], rows["sf b"]["met"], rows["sf b"]["error"]) == (None, True, "unbounded (∞)")
    assert rows["sf free"]["met"] is False and "no actuator capacity" in rows["sf free"]["error"]
    assert (rows["mass"]["value"], rows["mass"]["met"], rows["mass"]["worst_study"]) == (38.25, True, None)
    assert (rows["f"]["value"], rows["f"]["met"], rows["f"]["metric"]) == (19.125, False, "<lambda>()")
    assert rows["boom"]["met"] is False and rows["boom"]["value"] is None
    assert rows["boom"]["error"] == "ZeroDivisionError: division by zero"


def test_undefined_and_skipped(four_bar):
    res = {"a": _result("a", [0.0, 5.0], ok=[False, False])}
    rows = _eval(four_bar, _report(["a"]), res,
                 ("open", "span:j_rocker", 1.0, None, None, "FAIL"),
                 ("no pairs", "clearance", 0.3, None, None, "FAIL"),
                 ("not run", "span:j_rocker", 1.0, None, "turn", "FAIL"),
                 ("unloaded", "load:j_rocker", None, 1.0, None, "FAIL"))
    assert rows["open"]["met"] is False and rows["open"]["value"] is None
    assert "no closed frame in study 'a'" in rows["open"]["error"]
    assert rows["no pairs"]["met"] is False and "no unjoined part pair" in rows["no pairs"]["error"]
    assert rows["not run"]["met"] is None and rows["not run"]["error"] == "study 'turn' was not run"
    assert rows["unloaded"]["met"] is False and "computes no load for 'j_rocker'" in rows["unloaded"]["error"]


def test_metric_name():
    def rocker_ratio(report):
        return 1.0

    assert metric_name("span:j") == "span:j" and metric_name(rocker_ratio) == "rocker_ratio()"


# ------------------------------------------------------------------------------ real runs


def test_slider_crank_metrics_match_closed_form(tmp_path):
    r, l = SLIDER_CRANK["r"], SLIDER_CRANK["l"]
    asm = make_slider_crank(**SLIDER_CRANK)
    asm.probe("A", part="crank", point=(r, 0, 0))
    asm.probe("C", part="slider", point=(r + l, 0, 0))
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=73)  # 5° steps: hits 0° and 180° exactly
    for label, metric in [("stroke", "span:j_slider"), ("inner", "min:j_slider"), ("outer", "max:j_slider"),
                          ("dx", "delta:C.x"), ("dy", "delta:C.y"), ("rod min", "min_dist:A,C"),
                          ("rod max", "max_dist:A,C"), ("slider path", "path:C"), ("crank path", "path:A"),
                          ("mass", "mass_g")]:
        asm.target(label, metric)
    rep = analyze(asm, export=False, out_root=tmp_path)
    got = {t["label"]: t["value"] for t in rep["targets"]}
    assert got["stroke"] == pytest.approx(2 * r, abs=1e-4)  # x = r cos θ + √(l² − r² sin² θ)
    assert got["inner"] == pytest.approx(l - r, abs=1e-4)
    assert got["outer"] == pytest.approx(l + r, abs=1e-4)
    assert got["dx"] == pytest.approx(2 * r, abs=1e-4)
    assert got["dy"] == pytest.approx(0.0, abs=1e-6)
    assert got["rod min"] == pytest.approx(l, abs=1e-4) and got["rod max"] == pytest.approx(l, abs=1e-4)
    assert got["slider path"] == pytest.approx(4 * r, abs=1e-3)  # out and back
    # 72-gon perimeter; report numbers carry 6 significant figures (relative rounding ≤ 5e-6)
    assert got["crank path"] == pytest.approx(72 * 2 * r * math.sin(math.pi / 72), rel=5e-6)
    assert got["mass"] == pytest.approx(rep["mass"]["total_g"], rel=1e-6)
    assert all(t["met"] for t in rep["targets"])


def test_pendulum_load_and_sf(tmp_path):
    """Arm along +X about axis (0, −1, 0): holding torque m·g·r·cos θ, max m·g·r at θ = 0."""
    asm = Assembly("pendulum")
    asm.part("post", Pos(0, 0, -30) * Box(10, 10, 20), ground=True)
    asm.part("arm", Pos(50, 0, 0) * Box(100, 10, 10), material="PLA")  # COM at r = 50 mm
    asm.revolute("j", "post", "arm", origin=(0, 0, 0), axis=(0, -1, 0), limits=(0, 90))
    asm.actuator("j", capacity=0.01)
    asm.study("lift", drive={"j": (0, 90)}, frames=10)
    asm.target("holding", "load:j", max=0.005)
    asm.target("margin", "sf:j", min=1.5)
    rep = analyze(asm, export=False, out_root=tmp_path)
    m = 100 * 10 * 10 * 1.24e-6  # kg: 10 cm³ of PLA
    tau = m * 9.80665 * 50e-3  # N·m
    got = {t["label"]: t for t in rep["targets"]}
    assert got["holding"]["value"] == pytest.approx(tau, rel=1e-4) and got["holding"]["met"] is False
    assert got["margin"]["value"] == pytest.approx(0.01 / tau, rel=1e-4) and got["margin"]["met"] is True
    assert rep["status"] == "FAIL"  # the missed holding target
    assert [i["code"] for i in rep["issues"] if i["severity"] == "FAIL"] == ["target_miss"]


def test_extent_metrics_skip_studies_that_do_not_move_it(four_bar):
    res = {"a": _result("a", [3.0, 3.0, 3.0]), "b": _result("b", [0.0, 20.0, 50.0]),
           "c": _result("c", [0.0, 25.0, 10.0])}
    rows = _eval(four_bar, _report(["a", "b", "c"]), res,
                 ("swing", "span:j_rocker", 40.0, None, None, "FAIL"),  # a holds j_rocker: skipped
                 ("low", "min:j_rocker", 1.0, None, None, "FAIL"))  # not an extent metric: a counts
    assert (rows["swing"]["value"], rows["swing"]["met"], rows["swing"]["worst_study"]) == (25.0, False, "c")
    assert (rows["low"]["value"], rows["low"]["worst_study"]) == (0.0, "b")
    still = {"a": _result("a", [3.0, 3.0])}
    assert _eval(four_bar, _report(["a"]), still, ("swing", "span:j_rocker", 40.0, None, None, "FAIL"))[
        "swing"]["value"] == 0.0  # nothing moves it anywhere: the honest answer is 0
