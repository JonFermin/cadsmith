"""targets.py: the §2.3 metric vocabulary, how study=None combines studies, and edge cases.

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


def _result(name, rocker, probes=None, ok=None, loop="once", transforms=None):
    """A StudyResult stand-in: j_crank is driven, j_rocker (a loop unknown) follows ``rocker``."""
    ok = ok or [True] * len(rocker)
    transforms = transforms or [{}] * len(rocker)
    poses = [SimpleNamespace(q={"j_crank": 0.0, "j_coupler": 0.0, "j_rocker": float(v)}, ok=o, transforms=T)
             for v, o, T in zip(rocker, ok, transforms)]
    study = SimpleNamespace(name=name, drive={"j_crank": (0, 360)}, loop=loop)
    return SimpleNamespace(study=study, poses=poses,
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


def test_study_none_combines_studies_by_what_the_metric_measures(four_bar, two_studies):
    """Extents (span/delta/path) take the largest value over the studies — what the mechanism can
    do: a joint that sweeps 35° in one study can sweep 35°. Positions (min/max) take the extreme
    over all frames of all studies; the bounds then decide met/missed either way."""
    rows = _eval(four_bar, _report(), two_studies,
                 ("span lo", "span:j_rocker", 20.0, None, None, "FAIL"),  # spans a=10, b=35
                 ("span hi", "span:j_rocker", None, 30.0, None, "WARN"),
                 ("lowest", "min:j_rocker", -10.0, None, None, "FAIL"),  # minima a=0, b=−5
                 ("highest", "max:j_rocker", None, 40.0, None, "FAIL"),  # maxima a=10, b=30
                 ("reaches", "max:j_rocker", 25.0, None, None, "FAIL"),  # some study reaches 25: b
                 ("dips", "min:j_rocker", None, -3.0, None, "FAIL"),  # some study dips below −3: b
                 ("band", "span:j_rocker", 5.0, 40.0, None, "FAIL"),
                 ("only b", "span:j_rocker", 20.0, None, "b", "FAIL"),
                 ("only a", "span:j_rocker", 20.0, None, "a", "FAIL"))
    assert (rows["span lo"]["value"], rows["span lo"]["met"], rows["span lo"]["worst_study"]) == (35.0, True, "b")
    assert (rows["span hi"]["value"], rows["span hi"]["met"], rows["span hi"]["worst_study"]) == (35.0, False, "b")
    assert rows["span hi"]["severity"] == "WARN"
    assert (rows["lowest"]["value"], rows["lowest"]["met"], rows["lowest"]["worst_study"]) == (-5.0, True, "b")
    assert (rows["highest"]["value"], rows["highest"]["met"], rows["highest"]["worst_study"]) == (30.0, True, "b")
    assert (rows["reaches"]["value"], rows["reaches"]["met"]) == (30.0, True)  # worst-of-studies said 10: miss
    assert (rows["dips"]["value"], rows["dips"]["met"]) == (-5.0, True)  # worst-of-studies said 0: miss
    assert (rows["band"]["value"], rows["band"]["met"], rows["band"]["worst_study"]) == (35.0, True, "b")
    assert (rows["only b"]["value"], rows["only b"]["met"], rows["only b"]["study"]) == (35.0, True, "b")
    assert (rows["only a"]["value"], rows["only a"]["met"], rows["only a"]["margin"]) == (10.0, False, -10.0)
    assert set(rows["band"]) == {"label", "metric", "value", "min", "max", "margin", "met", "severity", "study",
                                 "worst_study", "error"}
    assert rows["span lo"]["margin"] == 15.0 and rows["band"]["margin"] == 5.0  # signed: negative = miss


def test_aggregation_table():
    from mech.targets import aggregation

    assert [aggregation(m) for m in ("span:j", "delta:p.z", "path:p")] == ["max"] * 3
    assert [aggregation(m) for m in ("max:j", "max_dist:a,b", "min:j", "min_dist:a,b")] == ["max", "max", "min", "min"]
    assert [aggregation(m) for m in ("clearance", "load:j", "sf:j", "rot:p", "angle:a,b")] == ["worst"] * 5


def test_probe_extents_take_the_largest_study_and_distances_the_extreme(four_bar):
    """delta/path: the study that moves the probe furthest; min_dist/max_dist: the closest and the
    farthest approach over every frame of every study."""
    a = {"p": [[0, 0, 0], [3, 0, 0]], "q": [[10, 0, 0], [10, 0, 0]]}  # Δx 3, distances 10, 7
    b = {"p": [[0, 0, 0], [8, 0, 0]], "q": [[20, 0, 0], [20, 0, 0]]}  # Δx 8, distances 20, 12
    res = {"a": _result("a", [0.0, 1.0], a), "b": _result("b", [0.0, 1.0], b)}
    rows = _eval(four_bar, _report(), res,
                 ("stroke", "delta:p.x", 5.0, None, None, "FAIL"),
                 ("travel", "path:p", None, 6.0, None, "FAIL"),
                 ("closest", "min_dist:p,q", 5.0, None, None, "FAIL"),
                 ("farthest", "max_dist:p,q", None, 15.0, None, "FAIL"))
    assert (rows["stroke"]["value"], rows["stroke"]["met"], rows["stroke"]["worst_study"]) == (8.0, True, "b")
    assert (rows["travel"]["value"], rows["travel"]["met"]) == (8.0, False)  # a limit on the largest travel
    assert (rows["closest"]["value"], rows["closest"]["met"], rows["closest"]["worst_study"]) == (7.0, True, "a")
    assert (rows["farthest"]["value"], rows["farthest"]["met"], rows["farthest"]["worst_study"]) == (20.0, False, "b")


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
    assert (rows["f"]["value"], rows["f"]["met"], rows["f"]["metric"]) == (19.125, False, "callable")
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

    # a callable is shown by its target label: neither ``<lambda>`` nor a function name means anything
    assert metric_name("span:j") == "span:j" and metric_name(rocker_ratio) == "callable"
    assert metric_name(lambda r: 1.0) == "callable"


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
    """A study that does not move the joint/part says nothing about how far it moves: for a
    worst-case rotation limit (rot ≥ 30: "every study that turns it turns it 30°") the held study
    must not count as 0; for the largest-value extents it never wins anyway."""
    from mech.geom import rot_about_line

    def turns(*degs):
        return [{"rocker": rot_about_line((0, 0, 0), (0, 0, 1), d)} for d in degs]

    res = {"a": _result("a", [3.0, 3.0, 3.0], transforms=turns(0, 0, 0)),
           "b": _result("b", [0.0, 20.0, 50.0], transforms=turns(0, 30, 50)),
           "c": _result("c", [0.0, 25.0, 10.0], transforms=turns(0, 25, 10))}
    rows = _eval(four_bar, _report(["a", "b", "c"]), res,
                 ("swing", "span:j_rocker", 40.0, None, None, "FAIL"),  # a holds j_rocker
                 ("tilt", "rot:rocker", 30.0, None, None, "FAIL"),  # a never turns the rocker: skipped
                 ("low", "min:j_rocker", 1.0, None, None, "FAIL"))  # not an extent metric: a counts
    assert (rows["swing"]["value"], rows["swing"]["met"], rows["swing"]["worst_study"]) == (50.0, True, "b")
    assert rows["tilt"]["value"] == pytest.approx(25.0) and rows["tilt"]["met"] is False
    assert rows["tilt"]["worst_study"] == "c"
    assert (rows["low"]["value"], rows["low"]["worst_study"]) == (0.0, "b")
    still = {"a": _result("a", [3.0, 3.0])}
    assert _eval(four_bar, _report(["a"]), still, ("swing", "span:j_rocker", 40.0, None, None, "FAIL"))[
        "swing"]["value"] == 0.0  # nothing moves it anywhere: the honest answer is 0


# ------------------------------------------------------------------------------ bounds, partial runs, series


def test_bounds_allow_float_noise_but_not_real_misses(four_bar):
    """A value that lands on its bound by construction (30 − 4e-15 vs min 30) meets it; a real miss
    of 1e-6 does not, and the row carries the signed margin that says by how much."""
    rows = _eval(four_bar, _report(["a"]), {"a": _result("a", [0.0])},
                 ("exact", lambda r: 30.0 - 4e-15, 30.0, None, None, "FAIL"),
                 ("exact max", lambda r: 1e5 + 1e-11, None, 1e5, None, "FAIL"),
                 ("short", lambda r: 30.0 - 1e-6, 30.0, None, None, "FAIL"),
                 ("band", lambda r: 45.0, 10.0, 40.0, None, "FAIL"))
    assert rows["exact"]["met"] is True and rows["exact max"]["met"] is True
    assert rows["short"]["met"] is False and rows["short"]["margin"] == pytest.approx(-1e-6)
    assert rows["band"]["met"] is False and rows["band"]["margin"] == -5.0


def test_partial_run_leaves_unrun_studies_unevaluated(four_bar):
    """--study a (b skipped): a target of b, a study=None target nothing in a defines or moves, a
    callable that fails on the partial report, and a miss b could still repair (a maximum short of
    its min bound) are not evaluated (met None); a miss more studies cannot repair is real."""
    res = {"a": _result("a", [3.0, 3.0, 3.0])}  # j_rocker never moves in a
    report = _report(["a"])
    report["partial"] = {"studies": ["a"], "skipped": ["b"], "frames": None}
    rows = _eval(four_bar, report, res,
                 ("of b", "span:j_rocker", 40.0, None, "b", "FAIL"),
                 ("swing", "span:j_rocker", 40.0, None, None, "FAIL"),
                 ("gap", "clearance", 0.3, None, None, "FAIL"),
                 ("low", "min:j_rocker", 5.0, None, None, "FAIL"),
                 ("reach", "max:j_rocker", 5.0, None, None, "FAIL"),
                 ("cap", "max:j_rocker", None, 2.0, None, "FAIL"),
                 ("b only", lambda r: r["studies"][1]["frames"], 1.0, None, None, "FAIL"),
                 ("mass", "mass_g", None, 10.0, None, "FAIL"))
    assert rows["of b"]["met"] is None and rows["of b"]["error"] == "study 'b' was not run"
    assert rows["swing"]["met"] is None and rows["swing"]["value"] is None
    assert rows["swing"]["error"] == "not evaluated: nothing moves it in the studies run (studies not run: b)"
    assert rows["gap"]["met"] is None and rows["gap"]["error"].startswith("not evaluated (studies not run: b): ")
    assert rows["b only"]["met"] is None and "IndexError" in rows["b only"]["error"]
    assert (rows["low"]["value"], rows["low"]["met"]) == (3.0, False)  # more studies can only lower a minimum
    assert (rows["reach"]["value"], rows["reach"]["met"]) == (3.0, None)  # b could still reach 5: not evaluated
    assert rows["reach"]["error"] == ("not evaluated (studies not run: b): largest value so far 3, a skipped study "
                                     "could still change it")
    assert (rows["cap"]["value"], rows["cap"]["met"]) == (3.0, False)  # a maximum only grows: a real miss
    assert rows["mass"]["met"] is False  # study-independent
    full = _eval(four_bar, _report(["a"]), res, ("swing", "span:j_rocker", 40.0, None, None, "FAIL"))
    assert (full["swing"]["value"], full["swing"]["met"]) == (0.0, False)  # a full run: nothing moves it


def test_loop_unknowns_are_rewrapped_across_open_frames(four_bar):
    """The solver's guess drifts by a turn while the loop is open (frames 2–3): the closed run after the
    gap continues from the one before it, so span/min/max see the physical swing, not +360°."""
    rocker = [0.0, 131.0, 250.0, 300.0, 319.6, 360.0]
    res = {"a": _result("a", rocker, ok=[True, True, False, False, True, True])}
    rows = _eval(four_bar, _report(["a"]), res,
                 ("swing", "span:j_rocker", None, None, None, "FAIL"),
                 ("low", "min:j_rocker", None, None, None, "FAIL"),
                 ("high", "max:j_rocker", None, None, None, "FAIL"))
    assert rows["swing"]["value"] == pytest.approx(131.0 + 40.4)
    assert rows["low"]["value"] == pytest.approx(-40.4) and rows["high"]["value"] == pytest.approx(131.0)


def test_path_of_a_pingpong_study_is_the_full_cycle(four_bar):
    """Pingpong frames u = 1 − |1 − 2k/N| never repeat frame 0 (seamless wrap): the path adds the chord
    back to frame 0, so it is the there-and-back length; open frames are never bridged."""
    r, n = 25.0, 8
    ang = [math.radians(90 * (1 - abs(1 - 2 * k / n))) for k in range(n)]
    tip = [[r * math.cos(a), 0.0, r * math.sin(a)] for a in ang]
    chord = 2 * r * math.sin(math.radians(22.5) / 2)  # every step is 22.5°
    once = _eval(four_bar, _report(["a"]), {"a": _result("a", [0.0] * n, {"tip": tip})},
                 ("path", "path:tip", None, None, None, "FAIL"))["path"]["value"]
    pp = _eval(four_bar, _report(["a"]), {"a": _result("a", [0.0] * n, {"tip": tip}, loop="pingpong")},
               ("path", "path:tip", None, None, None, "FAIL"))["path"]["value"]
    assert once == pytest.approx(7 * chord, rel=1e-5) and pp == pytest.approx(8 * chord, rel=1e-5)
    gap = _eval(four_bar, _report(["a"]), {"a": _result("a", [0.0] * n, {"tip": tip}, loop="pingpong",
                                                         ok=[True] * 3 + [False] + [True] * 4)},
                ("path", "path:tip", None, None, None, "FAIL"))["path"]["value"]
    assert gap == pytest.approx(6 * chord, rel=1e-5)  # f2→f3 and f3→f4 dropped, wrap f7→f0 kept


def test_rotation_metrics(four_bar):
    """rot:<part> = largest rotation from home (deg); angle:<a>,<b> = largest relative rotation."""
    from mech.geom import rot_about_line
    from mech.targets import rotation_deg

    def T(deg):
        return rot_about_line((5, 0, 0), (0, 0, 1), deg)

    frames = [{"crank": T(0), "rocker": T(0)}, {"crank": T(30), "rocker": T(29.5)},
              {"crank": T(-50), "rocker": T(-50)}, {"crank": T(170), "rocker": T(0)}]
    res = {"a": _result("a", [0.0] * 4, ok=[True, True, True, False], transforms=frames)}
    rows = _eval(four_bar, _report(["a"]), res,
                 ("turns", "rot:crank", None, None, None, "FAIL"),
                 ("parallel", "angle:crank,rocker", None, 1.0, None, "FAIL"),
                 ("still", "rot:frame", None, 0.1, None, "FAIL"))
    assert rows["turns"]["value"] == pytest.approx(50.0)  # the open frame 3 is ignored
    assert rows["parallel"]["value"] == pytest.approx(0.5) and rows["parallel"]["met"] is True
    assert rows["still"]["value"] == 0.0 and rows["still"]["met"] is True
    for deg in (1e-7, 0.5, 90.0, 179.9):
        assert rotation_deg(rot_about_line((0, 0, 0), (1, 2, 3), deg)) == pytest.approx(deg, rel=1e-6)


def test_callables_get_per_frame_series(four_bar):
    """A callable metric sees each study's per-frame ok flags, joint values, probes and transforms."""
    from mech.geom import rot_about_line

    seen = {}

    def jaw_tilt(r):
        s = r["studies"][0]["series"]
        seen.update(s)
        return max(abs(math.degrees(math.atan2(T[1][0], T[0][0]))) for T in s["transforms"]["rocker"])

    frames = [{"rocker": rot_about_line((0, 0, 0), (0, 0, 1), d)} for d in (0.0, 2.0, -3.0)]
    res = {"a": _result("a", [0.0, 4.0, 10.0], {"p": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}, transforms=frames)}
    row = _eval(four_bar, _report(["a"]), res, ("tilt", jaw_tilt, None, 5.0, None, "FAIL"))["tilt"]
    assert row["value"] == pytest.approx(3.0) and row["met"] is True
    assert seen["ok"] == [True, True, True] and seen["joints"]["j_rocker"] == [0.0, 4.0, 10.0]
    assert seen["probes"]["p"][2] == [2.0, 0.0, 0.0]
    assert np.allclose(seen["transforms"]["frame"][1], np.eye(4))  # parts a pose lacks stay at home


def test_a_frames_override_still_evaluates_every_target(tmp_path):
    """--frames only changes the sampling: every study still runs and its targets are evaluated."""
    asm = make_slider_crank(**SLIDER_CRANK)
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=73)
    asm.target("stroke", "span:j_slider", min=55)
    rep = analyze(asm, frames=5, export=False, out_root=tmp_path)
    assert rep["partial"] == {"studies": ["turn"], "skipped": [], "frames": 5}
    (t,) = rep["targets"]
    assert t["met"] is True and t["value"] == pytest.approx(2 * SLIDER_CRANK["r"], abs=1e-4)  # 0°, 180° sampled


def test_partial_run_callable_miss_is_not_evaluated(four_bar):
    """A callable can't say which studies it reads (desktop_arm's tool reach is the max over all
    studies): on a partial run its miss could still be repaired by a skipped study, so it is not
    evaluated, with the value so far; a callable met on the studies run counts as met; the same
    miss on a full run is real."""
    res = {"a": _result("a", [3.0, 3.0, 3.0])}
    report = _report(["a"])
    report["partial"] = {"studies": ["a"], "skipped": ["b"], "frames": None}
    rows = _eval(four_bar, report, res,
                 ("reach", lambda r: 255.0, 280.0, None, None, "FAIL"),
                 ("ok", lambda r: 290.0, 280.0, None, None, "FAIL"))
    assert (rows["reach"]["value"], rows["reach"]["met"]) == (255.0, None)
    assert rows["reach"]["error"] == ("not evaluated (studies not run: b): value so far 255, a skipped study could "
                                     "still change it")
    assert rows["ok"]["met"] is True
    full = _eval(four_bar, _report(["a"]), res, ("reach", lambda r: 255.0, 280.0, None, None, "FAIL"))
    assert full["reach"]["met"] is False and full["reach"]["error"] is None
