"""report.py: number formatting, JSON safety, the text summary (order, dedup, budget, Δprev),
targeted viewer URLs, and actionable issue messages from real runs."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from mech.report import (SUMMARY_LINES, attach_targets, format_check, format_summary, invalid_report, jsonable,
                         sig, vec, extent)
from mech.runner import analyze

from conftest import make_four_bar

M = "−"  # the summary's minus sign
BASE_URL = "http://localhost:3000/mech.html?m=four_bar"


# ------------------------------------------------------------------------------ formatting


@pytest.mark.parametrize("x, text", [
    (0.0, "0"), (51.26, "51.3"), (12.0, "12.0"), (360.0, "360"), (0.041234, "0.0412"), (187.4, "187"),
    (-12.04, M + "12.0"), (9.996, "10.0"), (0.5, "0.500"), (2.1e-9, "2.1e-9"), (123456.0, "1.23e5"),
    (1234.5, "1230"), (0.001, "0.00100"), (None, "n/a"), (math.nan, "n/a"), (math.inf, "n/a"),
])
def test_sig_three_significant_figures(x, text):
    assert sig(x) == text


def test_vectors_share_one_decimal_count():
    assert vec([48.12, 12.0, 1.94]) == "(48.1, 12.0, 1.9)"
    assert vec([62.1, 41.0, 0.0]) == "(62.1, 41.0, 0)"
    assert vec([0.0, -0.78, -0.624]) == f"(0, {M}0.780, {M}0.624)"
    assert vec([0, 0, 0]) == "(0, 0, 0)"
    assert extent([0.4, 2.1, 4.0]) == "0.40×2.10×4.00"
    assert extent([8.45, 4.64, 4.0]) == "8.45×4.64×4.00"


def test_jsonable_has_no_nan_or_numpy():
    raw = {"a": np.float64(math.nan), "b": [np.inf, -np.inf, np.int64(3), np.bool_(True)], "c": np.arange(2.0),
           "d": (1.23456789, None), 4: "key"}
    out = jsonable(raw)
    assert out == {"a": None, "b": [None, None, 3, True], "c": [0.0, 1.0], "d": [1.23457, None], "4": "key"}
    json.dumps(out, allow_nan=False)  # would raise on NaN/Infinity
    assert type(out["b"][2]) is int and type(out["b"][3]) is bool


# ------------------------------------------------------------------------------ synthetic reports


def _study(name="turn", crank=(0.0, 360.0), rocker=(-12.04, 39.31), load=0.0412, frame=18, clearance=None,
           path=187.3):
    return {"name": name, "frames": 72,
            "joint_ranges": {"j_crank": list(crank), "j_coupler": [0.0, 0.0], "j_rocker": list(rocker)},
            "probes": {"mid": {"min": [0.0, 10.0, 5.0], "max": [62.1, 51.0, 5.0], "start": [0.0, 10.0, 5.0],
                               "end": [0.0, 10.0, 5.0], "path_mm": path}},
            "min_clearance": clearance,
            "loads": {"j_crank": {"unit": "N·m", "max_abs": load, "frame": frame, "capacity": 0.5,
                                  "sf": 0.5 / load, "reflected_from": None}},
            "max_residual": 2.1e-9}


def _report(**over):
    r = {"name": "four_bar", "status": "PASS", "params": {}, "parts": 4,
         "joints": {"driver": 1, "coupled": 0, "passive": 2, "free": 0, "fixed": 0},
         "roles": {"j_crank": "driver", "j_coupler": "passive", "j_rocker": "passive"},
         "joint_kinds": {"j_crank": "revolute", "j_coupler": "revolute", "j_rocker": "revolute"},
         "pins": ["p_B"], "mobility": {"turn": 0},
         "mass": {"total_g": 38.24, "com_mm": [48.12, 12.0, 1.94], "parts": {}},
         "issues": [],
         "targets": [{"label": "rocker swing", "metric": "span:j_rocker", "value": 51.26, "min": 40.0, "max": None,
                      "met": True, "severity": "FAIL", "study": None, "worst_study": "turn", "error": None}],
         "studies": [_study()], "home_min_clearance": None,
         "viewer_url": BASE_URL + "&ghost=6&layout=quad&ui=0", "delta_prev": {"first_run": True}}
    r.update(over)
    return r


def _issue(sev, code, msg, study="turn", parts=("crank", "rocker"), value=1.0, frame=3):
    return {"severity": sev, "code": code, "message": msg, "study": study, "frame": frame, "parts": list(parts),
            "value": value, "location": None, "extent": None}


def test_summary_reproduces_spec_example():
    assert format_summary(_report()).splitlines() == [
        "mech four_bar — PASS   4 parts · 3 joints (1 driver, 2 passive) · 1 loop · 38.2 g · CoG (48.1, 12.0, 1.9)",
        "OK   loop p_B closed (max 2.1e-9 mm) · no branch jumps · mobility 0",
        "load j_crank max 0.0412 N·m @f18 · capacity 0.5 → SF 12.1",
        "targets 1/1 · rocker swing span:j_rocker 51.3 ≥ 40",
        f"ranges j_crank 0…360° · j_rocker {M}12.0…39.3° · probe mid Δ(62.1, 41.0, 0) path 187 mm",
        "Δprev: first run",
        "view http://localhost:3000/mech.html?m=four_bar&ghost=6&layout=quad&ui=0 · shot: uv run mech shot four_bar",
    ]


def test_summary_line_order():
    issues = [
        _issue("FAIL", "interference", "crank/rocker 0.84 mm³ overlap"),
        _issue("FAIL", "target_miss", "rocker swing: span:j_rocker 31.3 < min 40", parts=()),
        _issue("WARN", "tight_clearance", "coupler/frame gap 0.1 mm < 0.3", parts=("coupler", "frame")),
        _issue("INFO", "held_at_home", "j_x 0° — not driven by any study", parts=("x",), study=None),
    ]
    lines = format_summary(_report(status="FAIL", issues=issues)).splitlines()
    prefixes = ["mech four_bar — FAIL", "FAIL interference", "FAIL target_miss", "WARN tight", "INFO held_at_home",
                "OK ", "load j_crank", "targets", "ranges", "Δprev", "view"]
    assert len(lines) == len(prefixes)
    for line, prefix in zip(lines, prefixes):
        assert line.startswith(prefix), (line, prefix)


def test_dedup_keeps_worst_and_lists_studies():
    issues = [  # worst first, as build_report sorts them
        _issue("FAIL", "interference", "crank/rocker 2.00 mm³ overlap", study="lift", value=2.0),
        _issue("FAIL", "interference", "crank/rocker 0.84 mm³ overlap", study="turn", value=0.84),
        _issue("FAIL", "interference", "coupler/frame 0.10 mm³ overlap", study="turn", parts=("frame", "coupler")),
    ]
    rep = _report(status="FAIL", issues=issues, studies=[_study("turn"), _study("lift")],
                  mobility={"turn": 0, "lift": 0})
    fails = [line for line in format_summary(rep).splitlines() if line.startswith("FAIL")]
    assert fails == ["FAIL interference crank/rocker 2.00 mm³ overlap [lift, turn]",
                     "FAIL interference coupler/frame 0.10 mm³ overlap [turn]"]


def test_ranges_collapse_across_studies():
    rep = _report(studies=[_study("turn", crank=(0, 90), rocker=(-5, 10), path=50.0),
                           _study("lift", crank=(-30, 45), rocker=(0, 20), path=80.0)],
                  mobility={"turn": 0, "lift": 0})
    ranges = next(line for line in format_summary(rep).splitlines() if line.startswith("ranges"))
    assert ranges == (f"ranges j_crank {M}30.0…90.0° · j_rocker {M}5.00…20.0° · probe mid Δ(62.1, 41.0, 0) "
                      f"path 80.0 mm")


def test_budget_drops_bottom_of_each_tier_and_tallies():
    issues = [_issue("WARN", "tight_clearance", f"p{k}/q{k} gap 0.{k:02d} mm", parts=(f"p{k}", f"q{k}"), value=k)
              for k in range(20)]
    rep = _report(status="WARN", issues=issues)
    lines = format_summary(rep).splitlines()
    assert len(lines) == SUMMARY_LINES
    warn = [line for line in lines if line.startswith("WARN")]
    assert warn == [f"WARN tight p{k}/q{k} gap 0.{k:02d} mm" for k in range(9)]  # the first (worst) nine
    assert lines[-2] == "(+13 more: 11 WARN tight, 1 load, 1 ranges — --verbose)"
    assert lines[0].startswith("mech four_bar — WARN") and lines[-1].startswith("view ")
    for kept in ("OK ", "targets", "Δprev"):  # context lines outrank the tail of a long tier
        assert any(line.startswith(kept) for line in lines)
    full = format_summary(rep, verbose=True).splitlines()
    assert len(full) == 2 + 20 + 5 and not any(line.startswith("(+") for line in full)


def test_budget_keeps_first_fail_warn_and_target_miss():
    issues = ([_issue("FAIL", "interference", f"a{k}/b{k} overlap", parts=(f"a{k}", f"b{k}")) for k in range(30)]
              + [_issue("FAIL", "target_miss", "swing: span:j 1.00 < min 2", parts=())]
              + [_issue("WARN", "tight_clearance", "c/d gap", parts=("c", "d"))]
              + [_issue("INFO", "gear_mesh", "g1/g2 meshing", parts=("g1", "g2"))])
    lines = format_summary(_report(status="FAIL", issues=issues)).splitlines()
    assert len(lines) == SUMMARY_LINES
    assert any(line.startswith("FAIL target_miss") for line in lines)
    assert any(line.startswith("WARN tight c/d") for line in lines)
    assert "1 INFO" in lines[-2] and "FAIL interference" in lines[-2] and lines[-2].endswith(" — --verbose)")


def test_invalid_summary_lists_errors():
    rep = invalid_report("four_bar", ["joint 'j' unknown parent 'fram' (did you mean 'frame'?)", "pin 'p' off part"],
                         {"t": 5.0})
    json.dumps(rep, allow_nan=False)
    assert rep["status"] == "INVALID" and [i["code"] for i in rep["issues"]] == ["invalid_model"] * 2
    assert format_summary(rep).splitlines() == [
        "mech four_bar — INVALID   2 errors",
        "FAIL invalid_model joint 'j' unknown parent 'fram' (did you mean 'frame'?)",
        "FAIL invalid_model pin 'p' off part",
        "nothing analyzed — fix the errors above and re-run",
    ]


# ------------------------------------------------------------------------------ targets, status, URL, Δprev


def _target(label, value, met, severity="FAIL", mn=40.0):
    return {"label": label, "metric": "span:j_rocker", "value": value, "min": mn, "max": None, "met": met,
            "severity": severity, "study": None, "worst_study": "turn", "error": None}


def test_attach_targets_adds_misses_and_retargets_viewer():
    rep = _report(targets=[], issues=[_issue("INFO", "held_at_home", "j_x", parts=("x",), study=None)])
    attach_targets(rep, [_target("rocker swing", 31.26, False), _target("soft", 5.0, False, "WARN")])
    assert rep["status"] == "FAIL"
    assert [(i["severity"], i["code"]) for i in rep["issues"]] == [
        ("FAIL", "target_miss"), ("WARN", "target_miss"), ("INFO", "held_at_home")]
    assert rep["issues"][0]["message"] == "rocker swing: span:j_rocker 31.3 < min 40"
    # target_miss has no parts, so the viewer is pointed at... the first FAIL/WARN issue (index 0)
    assert rep["viewer_url"] == BASE_URL + "&issue=0&ui=0"
    lines = format_summary(rep).splitlines()
    assert lines[1] == "FAIL target_miss rocker swing: span:j_rocker 31.3 < min 40"
    assert "targets 0/2" in lines


def test_viewer_url_prefers_an_issue_with_parts():
    rep = _report(targets=[], issues=[_issue("WARN", "tight_clearance", "a/b gap", parts=("a", "b"))])
    attach_targets(rep, [_target("rocker swing", 31.0, False)])
    assert [i["code"] for i in rep["issues"]] == ["target_miss", "tight_clearance"]
    assert rep["viewer_url"] == BASE_URL + "&issue=1&ui=0"
    attach_targets(rep, [_target("rocker swing", 51.0, True)])  # idempotent, and back to the overview
    assert rep["status"] == "WARN" and rep["viewer_url"] == BASE_URL + "&issue=0&ui=0"
    rep["issues"] = []
    attach_targets(rep, [])
    assert rep["status"] == "PASS" and rep["viewer_url"] == BASE_URL + "&ghost=6&layout=quad&ui=0"


def test_delta_prev_reports_what_changed():
    prev = _report(status="FAIL", issues=[
        _issue("FAIL", "interference", "crank/rocker overlap"),
        _issue("WARN", "tight_clearance", "a/b gap", parts=("b", "a")),
    ], studies=[_study(clearance={"parts": ["a", "b"], "value": 0.1, "frame": 2})],
        targets=[_target("rocker swing", 31.26, False)])
    now = _report(issues=[
        _issue("WARN", "tight_clearance", "a/b gap", parts=("a", "b")),  # same pair, other order: not new
        _issue("WARN", "joint_limit", "j_rocker 52° > 45", parts=("rocker",)),
    ], studies=[_study(clearance={"parts": ["a", "b"], "value": 0.45, "frame": 2})], targets=[])
    now["mass"]["total_g"] = 40.14
    attach_targets(now, [_target("rocker swing", 51.3, True)], prev)
    assert now["delta_prev"] == {"status": ["FAIL", "WARN"], "fixed": ["interference crank/rocker"],
                                 "new": ["joint_limit rocker"], "min_clearance": [0.1, 0.45],
                                 "mass_g": [38.24, 40.14], "targets": {"rocker swing": [31.26, 51.3]}}
    line = next(x for x in format_summary(now).splitlines() if x.startswith("Δprev"))
    assert line == ("Δprev: status FAIL→WARN · fixed interference crank/rocker · new joint_limit rocker · "
                    "min clearance 0.100→0.450 mm · mass 38.2→40.1 g · rocker swing 31.3→51.3")


def test_delta_prev_no_change_first_run_and_invalid_prev():
    prev = _report()
    now = _report(targets=[])
    attach_targets(now, copy.deepcopy(prev["targets"]), prev)
    assert now["delta_prev"] == {} and "Δprev: no change" in format_summary(now)
    attach_targets(now, now["targets"], None)
    assert "Δprev: first run" in format_summary(now)
    attach_targets(now, now["targets"], invalid_report("four_bar", ["x"], {}))
    assert "Δprev: previous run was INVALID" in format_summary(now)
    attach_targets(now, now["targets"], {"status": "PASS", "issues": "garbage"})
    assert "Δprev: previous report.json unreadable" in format_summary(now)


# ------------------------------------------------------------------------------ real runs


def _open_four_bar():
    """The §2 four-bar with a coupler too short for a full crank turn.

    The loop closes while |A − O4| ≤ coupler + rocker: 40² + 100² − 8000·cos θ ≤ 125², i.e.
    cos θ ≥ −0.503125, θ ≤ 120.21° (and ≥ 239.79° on the way back). With 5° frames the open
    frames are θ = 125…235° = f25–f47.
    """
    asm = make_four_bar(coupler=45.0)
    asm.studies[0].frames = 73
    return asm


def test_loop_open_message_names_failing_and_closing_driver_ranges(tmp_path):
    rep = analyze(_open_four_bar(), export=False, out_root=tmp_path)
    json.dumps(rep, allow_nan=False)
    assert rep["status"] == "FAIL"
    loop = [i for i in rep["issues"] if i["code"] == "loop_open"]
    assert len(loop) == 1
    msg = loop[0]["message"]
    assert msg.startswith("loop p_B open for j_crank 125…235° (f25–f47), max residual ")
    assert "; closes 0…120°, 240…360°" in msg
    assert loop[0]["parts"] == ["coupler", "rocker"] and 25 <= loop[0]["frame"] <= 47
    assert loop[0]["value"] > 1.0  # mm: far from closing at θ = 180° (|A − O4| = 140 vs 125)
    summary = format_summary(rep)
    assert "FAIL loop_open loop p_B open for j_crank 125…235°" in summary
    assert "!!   loop p_B OPEN" in summary


def test_interference_message_has_extent_and_home_location(tmp_path):
    from build123d import Box, Pos

    from mech import Assembly

    # a 40×6×4 arm about +Z sweeps into a 6×6×4 block centered at (21, 21, 2)
    asm = Assembly("bump", clearance=0.3)
    asm.part("base", Pos(0, 0, -3) * Box(80, 80, 4), ground=True)
    asm.part("arm", Pos(20, 0, 2) * Box(40, 6, 4))
    asm.part("block", Pos(21, 21, 2) * Box(6, 6, 4))
    asm.revolute("j_arm", "base", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_block", "base", "block", origin=(21, 21, 0), axis=(0, 0, 1), limits=(0, 1))
    asm.study("swing", drive={"j_arm": (0, 90)}, frames=10)
    rep = analyze(asm, export=False, out_root=tmp_path)
    hit = next(i for i in rep["issues"] if i["code"] == "interference")
    assert hit["parts"] == ["arm", "block"] and hit["frame"] == 4  # j_arm = 40°
    # overlap location, mapped into the arm's home frame: rotate the block center by −40° about Z
    c = math.cos(math.radians(40)), math.sin(math.radians(40))
    block_in_arm = np.array([21 * c[0] + 21 * c[1], -21 * c[1] + 21 * c[0], 2.0])
    assert np.linalg.norm(np.array(hit["location"]) - block_in_arm) < 3.0  # inside the 6 mm block
    assert hit["extent"][2] == pytest.approx(4.0, abs=1e-6)  # full block height overlaps
    assert f" mm³ overlap {extent(hit['extent'])} mm @ {vec(hit['location'])} · f4 j_arm=40.0°" in hit["message"]
    held = next(i for i in rep["issues"] if i["code"] == "held_at_home")
    assert held["message"] == "j_block 0 mm — not driven by any study" and held["severity"] == "INFO"


def test_check_report_lists_roles_mobility_and_home_clearance(four_bar):
    from mech.runner import check

    rep = check(four_bar)
    json.dumps(rep, allow_nan=False)
    assert rep["status"] == "PASS" and rep["mobility"] == {"turn": 0} and rep["studies"] == []
    text = format_check(rep)
    assert text.splitlines()[0].startswith("mech check four_bar — PASS   4 parts · 3 joints")
    assert "roles j_crank driver · j_coupler passive · j_rocker passive" in text
    assert "studies turn mobility 0" in text
    assert "home min clearance" in text


def test_a_study_is_underconstrained_only_in_the_loops_it_drives(tmp_path):
    """Default studies (one per mobility group) of a four-bar plus a serial arm: the arm's study
    leaves the loop at home without an underconstrained WARN; a declared arm-only study makes
    the never-moved loop joints a held_at_home INFO instead (in `mech check` too)."""
    from build123d import Box, Pos

    from mech.runner import check

    def model(arm_study: bool):
        asm = make_four_bar()
        asm.studies.clear()
        asm.targets.clear()
        asm.part("arm", Pos(20, 0, -15) * Box(40, 6, 4))
        asm.revolute("j_arm", "frame", "arm", origin=(0, 0, -15), axis=(0, 0, 1), limits=(0, 30))
        if arm_study:
            asm.study("arm_only", drive={"j_arm": (0, 30)}, frames=5)
        return asm

    rep = analyze(model(False), export=False, out_root=tmp_path)
    assert rep["mobility"] == {"sweep_j_crank": 0, "sweep_j_arm": 0}
    assert rep["status"] == "PASS" and not [i for i in rep["issues"] if i["code"] == "underconstrained"]
    for rep in (analyze(model(True), export=False, out_root=tmp_path), check(model(True))):
        assert rep["mobility"] == {"arm_only": 0} and rep["status"] == "PASS"
        (held,) = [i for i in rep["issues"] if i["code"] == "held_at_home"]
        assert held["message"].startswith("j_crank 0°, j_coupler 0°, j_rocker 0°")


def test_static_interference_inside_rigid_groups(tmp_path):
    from build123d import Box, Pos

    from mech import Assembly

    asm = Assembly("static", clearance=0.3)
    asm.part("base", Box(20, 20, 20), ground=True)  # [-10, 10]³
    asm.part("plate", Pos(15, 0, 0) * Box(20, 20, 20), ground=True)  # overlaps base in x ∈ [5, 10]
    asm.part("bracket", Pos(0, 18, 0) * Box(20, 20, 20))  # overlaps base in y ∈ [8, 10]
    asm.fix("bracket", "base")
    rep = analyze(asm, export=False, out_root=tmp_path)
    assert rep["status"] == "FAIL" and rep["studies"] == []
    by_parts = {tuple(i["parts"]): i for i in rep["issues"] if i["code"] == "static_interference"}
    grounds = by_parts[("base", "plate")]  # both ground: one rigid group, not fix-attached -> FAIL
    assert grounds["severity"] == "FAIL" and grounds["frame"] is None
    assert grounds["value"] == pytest.approx(5 * 20 * 20, rel=1e-6)
    np.testing.assert_allclose(grounds["extent"], [5, 20, 20], atol=1e-6)
    np.testing.assert_allclose(grounds["location"], [7.5, 0, 0], atol=1e-6)
    assert "in the same rigid group" in grounds["message"]
    fixed = by_parts[("base", "bracket")]  # fix-attached -> WARN
    assert fixed["severity"] == "WARN" and fixed["value"] == pytest.approx(2 * 20 * 20, rel=1e-6)
    assert "fix-attached by 'fix_bracket'" in fixed["message"]
    assert format_summary(rep).splitlines()[1] == (
        "FAIL static_interference base/plate 2000 mm³ overlap 5.0×20.0×20.0 mm @ (7.50, 0, 0) · at home — parts "
        "are in the same rigid group; union them into one part or leave a gap")
