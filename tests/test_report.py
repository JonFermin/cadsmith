"""report.py: number formatting, JSON safety, the text summary (order, dedup, budget, Δprev),
targeted viewer URLs, and actionable issue messages from real runs."""

from __future__ import annotations

import copy
import json
import math
import re

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
                                  "sf": 0.5 / load if load else None, "reflected_from": None}},
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
         "clearance": {"required": 0.3, "pairs_checked": 6, "allowed_contact": 0},
         "viewer_url": BASE_URL + "&ghost=6&layout=quad&ui=0", "delta_prev": {"first_run": True}}
    r.update(over)
    return r


def _issue(sev, code, msg, study="turn", parts=("crank", "rocker"), value=1.0, frame=3):
    return {"severity": sev, "code": code, "message": msg, "study": study, "frame": frame, "parts": list(parts),
            "value": value, "location": None, "extent": None}


def test_summary_of_a_synthetic_report():
    clear = {"parts": ["frame", "coupler"], "value": 12.96, "frame": 5, "at": {"j_crank": 25.35}}
    assert format_summary(_report(studies=[_study(clearance=clear)])).splitlines() == [
        "mech four_bar — PASS   4 parts · 3 joints (1 driver, 2 passive) · 1 loop · 38.2 g · CoG (48.1, 12.0, 1.9)",
        "OK   loop p_B closed (max 2.1e-9 mm) · no branch jumps · mobility 0",
        "clearance min 13.0 mm (frame/coupler @f5 j_crank=25.4°) · required 0.3 · 6 pairs checked",
        "load j_crank max 0.0412 N·m @f18 · capacity 0.5 → SF 12.1",
        "targets 1/1 · rocker swing span:j_rocker 51.3 ≥ 40",
        f"ranges j_crank 0…360° · j_rocker {M}12.0…39.3° · probe mid Δ(62.1, 41.0, 0) path 187 mm",
        "Δprev: first run",
        "view http://localhost:3000/mech.html?m=four_bar&ghost=6&layout=quad&ui=0 · shot: uv run mech shot four_bar",
    ]


def test_spec_sample_output_is_the_real_four_bar_summary(tmp_path):
    """MECH_SPEC §4.9's sample summary is what `mech run examples/four_bar.py` prints on a first run
    (view line aside: this run is not exported)."""
    import re
    from pathlib import Path

    from test_examples import example

    spec = (Path(__file__).resolve().parents[1] / "docs" / "MECH_SPEC.md").read_text(encoding="utf-8")
    block = re.search(r"\*\*format_summary\*\*.*?```\n(.*?)```", spec, re.S).group(1).splitlines()
    rep = analyze(example("four_bar").build(), export=False, out_root=tmp_path)

    def noise_free(lines):  # the loop residual is float noise (~1e-14 mm): any value will do
        return [re.sub(r"\(max \S+ mm\)", "(max … mm)", line) for line in lines]

    assert rep["studies"][0]["max_residual"] < 1e-9
    assert noise_free(format_summary(rep).splitlines()[:-1]) == noise_free(block[:-1])
    assert block[-1].startswith("view http://localhost:3000/mech.html?m=four_bar&ghost=6&layout=quad&ui=0 · ")


def test_summary_states_clearance_and_folds_zero_loads():
    """A PASS summary always names the min clearance; loads of 0 (axis ∥ g) fold into one line
    without an arbitrary frame, and an undefined load says why."""
    s0 = _study("turn", load=0.0, frame=0)
    s0["loads"]["j_rocker"] = {"unit": "N·m", "max_abs": None, "frame": None, "capacity": None, "sf": None,
                               "reflected_from": None, "why": "1 DOF undetermined"}
    lines = format_summary(_report(studies=[s0], clearance={"required": 0.3, "pairs_checked": 21,
                                                            "allowed_contact": 2})).splitlines()
    assert lines[2] == ("clearance min n/a (no unjoined, non-allowed pair to measure) · required 0.3 · 21 pairs "
                        "checked · 2 allowed-contact pairs")
    assert "load j_rocker n/a (1 DOF undetermined)" in lines
    assert "no gravity load on j_crank (axis ∥ g or balanced)" in lines
    assert not any("@f0" in line for line in lines)
    home = _report(studies=[], mobility={}, home_min_clearance={"parts": ["a", "b"], "value": 1.25})
    assert "clearance min 1.25 mm (a/b at home) · required 0.3 · 6 pairs checked" in format_summary(home)


def test_summary_line_order():
    issues = [
        _issue("FAIL", "interference", "crank/rocker 0.84 mm³ overlap"),
        _issue("FAIL", "target_miss", "rocker swing: span:j_rocker 31.3 < min 40", parts=()),
        _issue("WARN", "tight_clearance", "coupler/frame gap 0.1 mm < 0.3", parts=("coupler", "frame")),
        _issue("INFO", "held_at_home", "j_x 0° — not driven by any study", parts=("x",), study=None),
    ]
    lines = format_summary(_report(status="FAIL", issues=issues)).splitlines()
    prefixes = ["mech four_bar — FAIL", "FAIL interference", "FAIL target_miss", "WARN tight", "INFO held_at_home",
                "OK ", "clearance", "load j_crank", "targets", "ranges", "Δprev", "view"]
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
    assert warn == [f"WARN tight p{k}/q{k} gap 0.{k:02d} mm" for k in range(8)]  # the first (worst) eight
    assert lines[-2] == "(+14 more: 12 WARN tight, 1 load, 1 ranges — --verbose)"
    assert lines[0].startswith("mech four_bar — WARN") and lines[-1].startswith("view ")
    for kept in ("OK ", "clearance", "targets", "Δprev"):  # context lines outrank the tail of a long tier
        assert any(line.startswith(kept) for line in lines)
    full = format_summary(rep, verbose=True).splitlines()
    assert len(full) == 2 + 20 + 6 and not any(line.startswith("(+") for line in full)


def test_budget_keeps_first_fail_warn_and_target_miss():
    issues = ([_issue("FAIL", "interference", f"a{k}/b{k} overlap", parts=(f"a{k}", f"b{k}"), value=30.0 - k)
               for k in range(30)]  # (distinct volumes: alike findings would fold into one line)
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
    # joint-level issues are keyed by the joint their line names, not the joint's child part
    assert now["delta_prev"] == {"status": ["FAIL", "WARN"], "fixed": ["interference crank/rocker"],
                                 "new": ["joint_limit j_rocker"], "min_clearance": [0.1, 0.45],
                                 "mass_g": [38.24, 40.14], "targets": {"rocker swing": [31.26, 51.3]}}
    line = next(x for x in format_summary(now).splitlines() if x.startswith("Δprev"))
    assert line == ("Δprev: status FAIL→WARN · fixed interference crank/rocker · new joint_limit j_rocker · "
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
    # the spot in each part's home geometry: the arm's (rotated back by 40°) and the block's (never moved)
    spot_b = re.search(r", block \((.+?)\) · f4 j_arm=40\.0°", hit["message"])
    assert f" mm³ overlap {extent(hit['extent'])} mm @ arm {vec(hit['location'])}, block (" in hit["message"]
    assert spot_b and all(abs(float(x.replace(M, "-")) - c0) <= 3.0 for x, c0 in zip(spot_b.group(1).split(", "),
                                                                                         (21, 21, 2)))
    held = next(i for i in rep["issues"] if i["code"] == "held_at_home")
    assert held["message"] == "j_block 0 mm — not driven by any study" and held["severity"] == "INFO"


def test_check_report_lists_roles_mobility_and_home_clearance(four_bar):
    from mech.runner import check

    rep = check(four_bar)
    json.dumps(rep, allow_nan=False)
    assert rep["status"] == "PASS" and rep["mobility"] == {"turn": 0} and rep["studies"] == []
    text = format_check(rep)
    assert text.splitlines()[0].startswith("mech check four_bar — PASS   4 parts · 3 joints")
    assert "roles driver j_crank · passive j_coupler, j_rocker" in text
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


# ------------------------------------------------------------------------------ review regressions


def test_target_miss_names_the_bound_and_margin_even_when_rounding_hides_it():
    t = _target("opening", 29.99999, False, mn=30.0)
    t["metric"], t["margin"] = "max_dist:pad_l,pad_r", -1e-5
    rep = _report(targets=[])
    attach_targets(rep, [t])
    assert rep["issues"][0]["message"] == f"opening: max_dist:pad_l,pad_r 30.0 < min 30 (margin {M}1e-5)"
    hi = {**_target("reach", 12.5, False), "min": 5.0, "max": 10.0, "margin": -2.5}
    attach_targets(rep, [hi])
    assert rep["issues"][0]["message"] == f"reach: span:j_rocker 12.5 > max 10 (margin {M}2.50)"


def test_delta_prev_lists_params_added_targets_and_joint_subjects():
    prev = _report(params={"theta0": 10.0, "theta1": 40.0}, issues=[
        _issue("WARN", "over_capacity", "j_leadrot holding load 0.2 N·m near capacity 0.25", parts=("screw",)),
    ])
    now = _report(params={"theta0": 5, "theta1": 45.0}, issues=[
        _issue("FAIL", "over_capacity", "j_leadrot holding load 0.3 N·m exceeds capacity 0.25", parts=("screw",)),
        *[_issue("FAIL", "interference", f"a{k}/b{k} overlap", parts=(f"a{k}", f"b{k}")) for k in range(5)],
    ], targets=[])
    attach_targets(now, [_target("rocker swing", 51.26, True), _target("light", 27.4, True)], prev)
    d = now["delta_prev"]
    assert d["params"] == {"theta0": [10.0, 5], "theta1": [40.0, 45.0]}
    assert d["targets_added"] == ["light"] and "over_capacity j_leadrot" not in d.get("new", [])
    line = next(x for x in format_summary(now).splitlines() if x.startswith("Δprev"))
    assert line.startswith("Δprev: status PASS→FAIL · params theta0 10→5, theta1 40→45 · new interference a0/b0, "
                           "interference a1/b1, interference a2/b2 (+2 more — --verbose) · added target light")
    verbose = next(x for x in format_summary(now, verbose=True).splitlines() if x.startswith("Δprev"))
    assert "interference a4/b4" in verbose and "more — --verbose" not in verbose
    # the FAIL line names the joint, and so does its key (not the child part 'screw')
    assert any(x.startswith("FAIL over_capacity j_leadrot") for x in format_summary(now).splitlines())


def test_delta_prev_compares_only_studies_sampled_alike():
    prev = _report(status="FAIL", studies=[_study("sweep"), _study("back")], mobility={"sweep": 0, "back": 0},
                   issues=[_issue("FAIL", "interference", "arm/post overlap", study="sweep", parts=("arm", "post"))])
    now = _report(studies=[_study("back")], targets=[],
                  partial={"studies": ["back"], "skipped": ["sweep"], "frames": None})
    attach_targets(now, copy.deepcopy(prev["targets"]), prev)
    assert now["delta_prev"] == {"not_compared": {"sweep": "not run"}}  # no "fixed", no status change
    coarse = _report(studies=[{**_study("sweep"), "frames": 4}, {**_study("back"), "frames": 4}], targets=[])
    attach_targets(coarse, copy.deepcopy(prev["targets"]), prev)
    assert coarse["delta_prev"] == {"not_compared": {"sweep": "72→4 frames", "back": "72→4 frames"}}
    assert "Δprev: no change (not compared: sweep (72→4 frames), back (72→4 frames))" in format_summary(coarse)


def test_partial_run_header_and_view_line():
    rep = _report(partial={"studies": ["back"], "skipped": ["sweep", "turn"], "frames": 12}, viewer_url=None)
    lines = format_summary(rep).splitlines()
    assert lines[0].startswith("mech four_bar — PASS (partial run: --study back, skipped sweep, turn; --frames 12)   4 parts")
    assert lines[-1] == ("view: not exported — partial run; four_bar.mech keeps the last full run (drop "
                         "--study/--frames to update it)")
    only_frames = _report(partial={"studies": ["turn"], "skipped": [], "frames": 12}, viewer_url=None)
    assert "— PASS (partial run: --frames 12)   " in format_summary(only_frames)


def test_multi_driver_issue_location_names_every_driver_that_moves_the_parts(tmp_path):
    """Three drivers in one study: j_second swings bar b into cube a (turned by j_first) at its last
    frame. The location names both drivers that move the pair, so the pose can be set up again,
    and leaves out j_third, whose far-away arm moves neither part."""
    from build123d import Box, Pos

    from mech import Assembly

    asm = Assembly("md", clearance=0.3)
    asm.part("base", Box(100, 100, 4), ground=True)
    asm.part("a", Pos(-30, 0, 10) * Box(8, 8, 8))
    asm.part("b", Pos(0, 0, 10) * Box(60, 4, 4))  # pivots at its +X end, (30, 0)
    asm.part("c", Pos(0, 45, 10) * Box(20, 4, 4))
    asm.revolute("j_first", "base", "a", origin=(-30, 0, 0), axis=(0, 0, 1), limits=(-180, 180))
    asm.revolute("j_second", "base", "b", origin=(30, 0, 0), axis=(0, 0, 1), limits=(-180, 180))
    asm.revolute("j_third", "base", "c", origin=(0, 45, 0), axis=(0, 0, 1), limits=(-180, 180))
    asm.study("all", drive={"j_first": (0, 90), "j_second": (90, 0), "j_third": (0, 45)}, frames=10)
    rep = analyze(asm, export=False, out_root=tmp_path)
    hit = next(i for i in rep["issues"] if i["code"] == "interference")
    assert hit["parts"] == ["a", "b"] and hit["frame"] == 9
    assert hit["message"].endswith("· f9 j_first=90.0°, j_second=0°")
    clear = next(line for line in format_summary(rep).splitlines() if line.startswith("clearance"))
    assert "(a/b @f9 j_first=90.0°, j_second=0°" in clear and "j_third" not in clear


def test_open_frames_leave_ranges_targets_and_branch_jumps_alone(tmp_path):
    """four_bar with crank 200 closes only near θ = 0 (±~56°): the rocker's ranges, target and probe
    stats come from the closed frames — re-wrapped across the open gap, so the physical swing, not
    the solver's +360° drift — and no branch_jump is reported at the open/close transitions."""
    asm = make_four_bar(crank=200.0)
    rep = analyze(asm, frames=24, export=False, out_root=tmp_path)
    assert "loop_open" in [i["code"] for i in rep["issues"]]
    assert "branch_jump" not in [i["code"] for i in rep["issues"]]
    lo, hi = rep["studies"][0]["joint_ranges"]["j_rocker"]
    (t,) = rep["targets"]
    assert hi - lo == pytest.approx(t["value"], rel=1e-5) and t["value"] < 180.0
    full = analyze(make_four_bar(crank=200.0), export=False, out_root=tmp_path)
    ks = [k for k in range(72) if abs(((360 * k / 71) + 180) % 360 - 180) < 50]  # well inside the closed arc
    assert full["studies"][0]["probes"]["mid"]["path_mm"] < 1000.0 and len(ks) > 10


def test_undefined_load_says_why(tmp_path):
    """A five-bar driven at one crank closes every frame but keeps 1 DOF: its load is undefined because
    of the leftover mobility, not for lack of a closed frame."""
    from build123d import Box, Pos

    from mech import Assembly
    from mech.geom import circle_intersect, link

    t = 4.0
    asm = Assembly("fivebar", clearance=0.3)
    O1, O5, A, E = (0, 0, 0), (60, 0, 0), (0, 30, 0), (60, 30, 0)
    B = circle_intersect(A, 40, E, 40, side=-1)
    asm.part("frame", Pos(30, 0, -2 * t) * Box(90, 12, t), ground=True)
    asm.part("c1", link(O1, A, width=8, thickness=t, z=0))
    asm.part("l1", link(A, B, width=8, thickness=t, z=t + 0.5))
    asm.part("c2", link(O5, E, width=8, thickness=t, z=0))
    asm.part("l2", link(E, B, width=8, thickness=t, z=2 * t + 1))
    asm.revolute("j1", "frame", "c1", origin=O1, axis=(0, 0, 1))
    asm.revolute("j2", "c1", "l1", origin=A, axis=(0, 0, 1))
    asm.revolute("j5", "frame", "c2", origin=O5, axis=(0, 0, 1))
    asm.revolute("j4", "c2", "l2", origin=E, axis=(0, 0, 1))
    asm.pin("p_B", "l1", "l2", point=B, axis=(0, 0, 1))
    asm.actuator("j1", capacity=0.1)
    asm.study("one", drive={"j1": (0, 20)}, frames=5)
    rep = analyze(asm, export=False, out_root=tmp_path)
    assert rep["mobility"] == {"one": 1}
    load = rep["studies"][0]["loads"]["j1"]
    assert load["max_abs"] is None and load["why"] == "1 DOF undetermined"
    assert "load j1 n/a (1 DOF undetermined)" in format_summary(rep, verbose=True).splitlines()


def test_held_at_home_is_structural(tmp_path):
    """A study that holds its driver at one value still drives its loop: the passive loop joints are
    not 'held at home' (a 2-frame 0→360° study, which samples one pose twice, is INVALID instead).
    In a --study run, the joints of skipped studies are, in so many words."""
    rep = analyze(make_four_bar(), frames=2, export=False, out_root=tmp_path)
    assert rep["status"] == "INVALID"
    assert rep["issues"][0]["message"] == ("study 'turn': 2 frames over 0…360° (0° ≡ 360°) sample one pose — "
                                           "use frames ≥ 3")
    asm = make_four_bar()
    asm.studies[0].drive = {"j_crank": 30.0}
    rep = analyze(asm, frames=2, export=False, out_root=tmp_path)
    assert "held_at_home" not in [i["code"] for i in rep["issues"]] and rep["status"] != "INVALID"
    from build123d import Box, Pos

    asm = make_four_bar()
    asm.part("arm", Pos(20, 0, -15) * Box(40, 6, 4))
    asm.revolute("j_arm", "frame", "arm", origin=(0, 0, -15), axis=(0, 0, 1), limits=(0, 30))
    asm.study("arm", drive={"j_arm": (0, 30)}, frames=4)
    rep = analyze(asm, studies=["arm"], export=False, out_root=tmp_path)
    (held,) = [i for i in rep["issues"] if i["code"] == "held_at_home"]
    assert held["message"] == "j_crank 0°, j_coupler 0°, j_rocker 0° — not driven by the selected studies"
    assert rep["roles"]["j_crank"] == "driver"  # the skipped study still drives it
    assert rep["targets"][0]["met"] is None  # rocker swing: nothing moves j_rocker in 'arm'
    assert rep["status"] == "PASS"


def test_singular_pose_is_one_warn_per_study(tmp_path):
    """A parallelogram driven through its change points (180°, 360°): one WARN naming the first
    singular frame and its drive, counted on the loops line."""
    from test_kinematics import planar_four_bar

    asm = planar_four_bar(100, 40, 100, 40, th0=60.0, absolute=True)
    asm.study("turn", drive={"j_crank": (60, 420)}, frames=37)  # 10° steps: f12 = 180°, f30 = 360°
    rep = analyze(asm, export=False, out_root=tmp_path)
    (hit,) = [i for i in rep["issues"] if i["code"] == "singular_pose"]
    assert hit["severity"] == "WARN" and hit["frame"] == 12 and hit["value"] == 2 and hit["study"] == "turn"
    assert hit["message"].startswith("study 'turn' passes a change/dead point at f12 @ j_crank=180° "
                                     "(2 frames f12, f30) — the loop Jacobian is singular there")
    loops = next(line for line in format_summary(rep).splitlines() if "loop p_B" in line)
    assert loops.startswith("!!   loop p_B closed") and "2 singular frames" in loops


def _bushing(max_depth=0.1):
    """A square bushing (bore 3.9) on a 4.0 rod: overlaps 0.05 mm per side, mean depth ≈ 0.0498 mm."""
    from build123d import Box, Pos

    from mech import Assembly

    asm = Assembly("fit", clearance=0.3)
    asm.part("base", Pos(0, 0, -5) * Box(100, 20, 4), ground=True)
    asm.part("rod", Pos(0, 0, 10) * Box(100, 4, 4), ground=True)
    asm.part("bush", Pos(0, 0, 10) * (Box(10, 10, 10) - Box(20, 3.9, 3.9)))
    asm.prismatic("j", "base", "bush", origin=(0, 0, 10), axis=(1, 0, 0), limits=(0, 20))
    asm.allow_contact("rod", "bush", max_depth=max_depth)
    asm.study("slide", drive={"j": (0, 20)}, frames=3)
    return asm


def test_allowed_overlap_within_max_depth_is_one_contact_info(tmp_path):
    rep = analyze(_bushing(), export=False, out_root=tmp_path)
    assert rep["status"] == "PASS"
    (info,) = [i for i in rep["issues"] if i["code"] == "contact"]
    assert info["severity"] == "INFO" and info["parts"] == ["rod", "bush"]
    assert info["value"] == pytest.approx(0.04975, abs=1e-4)
    assert info["message"] == "rod/bush mean depth 0.0498 mm, 7.90 mm³ @f0 (allowed ≤ 0.1 mm)"


def test_allowed_overlap_deeper_than_max_depth_says_how_to_fix_it(tmp_path):
    from mech.runner import check

    rep = analyze(_bushing(max_depth=0.02), export=False, out_root=tmp_path)
    hit = next(i for i in rep["issues"] if i["code"] == "interference")
    assert hit["message"].endswith("— allowed contact deeper than max_depth 0.02 mm (mean depth 0.0498): fix the "
                                   "fit, or raise max_depth / max_depth=None if the overlap is intended (belt teeth)")
    assert "contact" not in [i["code"] for i in rep["issues"]]
    # the home pose's min clearance is signed like a sweep's: the overlap depth of the interfering pair
    home = check(_bushing(max_depth=0.02))
    assert home["home_min_clearance"]["parts"] == ["rod", "bush"]
    assert home["home_min_clearance"]["value"] == pytest.approx(-0.04975, abs=1e-4)
    assert f"home min clearance {M}0.0498 mm (rod/bush; negative = overlap depth)" in format_check(home)


# ------------------------------------------------------------------------------ compaction (big models)


def _big_report(findings: bool = True) -> dict:
    """A strandbeest-sized synthetic report: 2 drivers, 30 passive + 5 coupled joints, 18 loops, 7 probes
    (six identical feet), 6 meshing pairs, 5 allowed contacts, reflected and round-off loads, 12
    targets (callables among them) and a busy Δprev; with ``findings`` also FAIL/WARN/target_miss
    issues in two studies (a missed callable target among them)."""
    passive = [f"jP_{side}{k}" for k in range(15) for side in "LR"]
    coupled = ["j_out", "j_c0", "j_c1", "j_c2", "j_c3"]
    roles = {"j_crank": "driver", "j_lift": "driver", "j_yaw": "driver", **{j: "passive" for j in passive},
             **{j: "coupled" for j in coupled}, "fix_a": "fixed"}
    kinds = {j: "revolute" for j in roles} | {"j_lift": "prismatic", "fix_a": "fixed"}
    feet = {f"foot_{side}{k}": {"min": [-30.0, 5.0, -10.0], "max": [37.8, 5.0, 12.5], "start": [0, 0, 0],
                                "end": [0, 0, 0], "path_mm": 149.2} for k in range(3) for side in "LR"}
    probes = {**feet, "tip": {"min": [0.0, 0.0, 0.0], "max": [10.0, 20.0, 30.0], "start": [0, 0, 0], "end": [0, 0, 0],
                              "path_mm": 61.0}}

    def ranges(scale: float) -> dict:
        r = {"j_crank": [0.0, 360.0], "j_lift": [0.0, 40.0 * scale], "j_yaw": [0.0, 90.0],
             "j_out": [0.0, 120.0 * scale], **{j: [-0.5 * k, 40.0 + 1.5 * k] for k, j in enumerate(passive)},
             **{f"j_c{k}": [-38.4, 0.0] for k in range(4)}}
        return r

    def load(mx, frame, cap=None, refl=None, unit="N·m"):
        return {"unit": unit, "max_abs": mx, "frame": frame, "capacity": cap,
                "sf": cap / mx if cap and mx else None, "reflected_from": refl}

    loads = {"j_crank": load(0.148, 3, 0.4), "j_lift": load(46.4, 0, 80.0, unit="N"), "j_yaw": load(3.42e-17, 7, 0.2),
             "j_out": load(3.70, 3, None, "j_crank"), "j_c0": load(1.39, 3, None, "j_crank"),
             "j_c1": load(0.741, 3, None, "j_crank"), "j_c2": load(0.278, 3, None, "j_crank"),
             "j_c3": load(0.0, 0, None, "j_crank")}
    walk = {"name": "walk", "frames": 73, "joint_ranges": ranges(1.0), "probes": probes,
            "min_clearance": {"parts": ["c_L0", "upper_L0"], "value": 0.8, "frame": 23,
                              "at": {"j_crank": 138.0, "j_out": 46.0, "j_c0": -14.7}},
            "loads": loads, "max_residual": 1.28e-13,
            "sweep_stats": {"seconds": 96.2, "pairs": 1521, "proximity": 900, "exact": 312, "booleans": 40,
                            "subframe_poses": 12, "slowest": [["c_L0", "upper_L0", 3.1], ["a", "b", 2.0],
                                                              ["c", "d", 1.5], ["e", "f", 1.0]]}}
    lift = {**walk, "name": "lift", "frames": 21, "joint_ranges": ranges(0.5), "min_clearance": None,
            "loads": {k: dict(v) for k, v in loads.items()}, "sweep_stats": None}
    bad = [
        _issue("FAIL", "interference", "c_L0/upper_L0 118 mm³ overlap 5.00×6.00×5.00 mm @ c_L0 (12.0, 3.0, 4.0), "
               "upper_L0 (14.0, 3.0, 4.0) · f78 j_crank=272°, j_lift=12.5 mm (3 frames f76–f78)", study="walk",
               parts=("c_L0", "upper_L0"), value=118.0),
        _issue("FAIL", "interference", "c_L0/upper_L0 40.1 mm³ overlap", study="lift", parts=("c_L0", "upper_L0"),
               value=40.1),
        _issue("FAIL", "interference", "foot_L1/shin_L1 2.00 mm³ overlap 1.00×2.00×1.00 mm @ (4.0, 1.0, 2.0) · f12 "
               "j_crank=59.2°", study="walk", parts=("foot_L1", "shin_L1"), value=2.0),
        _issue("FAIL", "target_miss", "flat stance: 0.812 < min 0.85 (margin −0.0380)", study=None, parts=(),
               value=0.812),
        *[_issue("WARN", "tight_clearance", f"p{k}/q{k} gap 0.{k + 1}0 mm < 0.3 @ (1.0, 2.0, 3.0) dir (0, 0, 1) · f5 "
                 f"j_crank=24.7°", study="walk", parts=(f"p{k}", f"q{k}"), value=0.1 * (k + 1)) for k in range(3)],
        _issue("WARN", "joint_limit", "jP_L3 52.1° > 45 limit @ j_crank=80.0° (f16; 3 frame(s) f15–f17)",
               study="walk", parts=("upper_L3",), value=7.1),
    ]
    infos = [
        *[_issue("INFO", "gear_mesh", f"planet{k}/sun meshing: contact allowed, interference still checked",
                 study=None, parts=(f"planet{k}", "sun"), value=None, frame=None) for k in range(6)],
        *[_issue("INFO", "contact", f"rod{k}/bush{k} mean depth 0.0{9 - k}0 mm, 7.90 mm³ @f0 (allowed ≤ 0.1 mm)",
                 study="walk", parts=(f"rod{k}", f"bush{k}"), value=0.01 * (9 - k), frame=0) for k in range(5)],
        _issue("INFO", "held_at_home", "j_tail 0° — not driven by any study", study=None, parts=("tail",),
               value=None, frame=None),
    ]
    issues = (bad if findings else []) + infos
    targets = [
        {"label": "stride", "metric": "delta:foot_L0.x", "value": 67.8, "min": 60.0, "max": None, "met": True},
        {"label": "step lift", "metric": "delta:foot_L0.z", "value": 22.5, "min": 15.0, "max": None, "met": True},
        {"label": "flat stance", "metric": "callable", "value": 0.812 if findings else 0.912, "min": 0.85, "max": None,
         "met": not findings},
        {"label": "ratio 25:1", "metric": "callable", "value": 25.0, "min": 24.9, "max": 25.1, "met": True},
        {"label": "feet on the ground", "metric": "callable", "value": 2.0, "min": 2.0, "max": None, "met": True},
        {"label": "output swing", "metric": "span:j_out", "value": 120.0, "min": 100.0, "max": None, "met": True},
        {"label": "output torque", "metric": "load:j_out", "value": 3.7, "min": 3.5, "max": None, "met": True},
        {"label": "motor SF", "metric": "sf:j_crank", "value": 2.7, "min": 2.0, "max": None, "met": True},
        {"label": "lift SF", "metric": "sf:j_lift", "value": 1.72, "min": 1.5, "max": None, "met": True},
        {"label": "light", "metric": "mass_g", "value": 744.0, "min": None, "max": 800.0, "met": True},
        {"label": "lift travel", "metric": "span:j_lift", "value": 40.0, "min": 35.0, "max": None, "met": True},
        {"label": "no collision", "metric": "clearance", "value": 0.8, "min": 0.5, "max": None, "met": True},
    ]
    for t in targets:
        t.update({"severity": "FAIL", "study": None, "worst_study": "walk", "error": None,
                  "margin": None if t["value"] is None else t["value"] - (t["min"] or 0)})
    delta = {"params": {"crank": [15.0, 16.0], "leg": [50.0, 52.0]}, "min_clearance": [1.1, 0.8],
             "mass_g": [731.0, 744.0],
             "targets": {"stride": [66.1, 67.8], "step lift": [22.1, 22.5],
                         "flat stance": [0.9, 0.812 if findings else 0.912],
                         "output swing": [118.0, 120.0], "motor SF": [2.8, 2.7], "lift SF": [1.8, 1.72]}}
    if findings:
        delta = {"status": ["PASS", "FAIL"],
                 "new": ["interference c_L0/upper_L0", "interference foot_L1/shin_L1", "tight p0/q0", "tight p1/q1",
                         "tight p2/q2"], "fixed": ["joint_limit jP_R7"], **delta}
    return _report(name="walker", status="FAIL" if findings else "PASS", parts=57, pins=[f"p{k}" for k in range(18)],
                   joints={"driver": 3, "coupled": 5, "passive": 30, "free": 0, "fixed": 1}, roles=roles,
                   joint_kinds=kinds, mobility={"walk": 0, "lift": 0}, studies=[walk, lift], issues=issues,
                   targets=targets, mass={"total_g": 744.0, "com_mm": [0.1, 53.2, 4.3], "parts": {}},
                   clearance={"required": 0.5, "pairs_checked": 1521, "allowed_contact": 11},
                   viewer_url="http://localhost:3000/mech.html?m=walker&" + ("issue=0&ui=0" if findings else
                                                                             "ghost=6&layout=quad&ui=0"),
                   delta_prev=delta)


GOLDEN_BIG_PASS = [
    ('mech walker — PASS   57 parts · 39 joints (3 driver, 5 coupled, 30 passive, 1 fixed) · 18 loops · '
     '744 g · CoG (0.1, 53.2, 4.3)'),
    ('INFO gear_mesh 6 meshing pairs (planet0/sun, planet1/sun, planet2/sun, +3): contact allowed, '
     'interference still checked'),
    ('INFO contact rod0/bush0 mean depth 0.090 mm, 7.90 mm³ @f0 (allowed ≤ 0.1 mm) [walk] · 5 '
     'allowed-contact pairs touch (+4 more — --verbose)'),
    'INFO held_at_home j_tail 0° — not driven by any study',
    'OK   18 loops closed (max 1.28e-13 mm) · no branch jumps · mobility 0',
    ('clearance min 0.800 mm (c_L0/upper_L0 @f23 j_crank=138° [walk]) · required 0.5 · 1521 pairs checked '
     '· 11 allowed-contact pairs'),
    'load j_crank max 0.148 N·m @f3 (walk) · capacity 0.4 → SF 2.70',
    'load j_lift max 46.4 N @f0 (walk) · capacity 80 → SF 1.72',
    'load j_out max 3.70 N·m @f3 (walk) (reflected from j_crank) · no actuator capacity',
    'no gravity load on j_yaw (axis ∥ g or balanced)',
    'reflected loads j_c0 1.39 N·m, j_c1 0.741 N·m, j_c2 0.278 N·m',
    ('targets 12/12 · stride delta:foot_L0.x 67.8 ≥ 60 · step lift delta:foot_L0.z 22.5 ≥ 15 · flat stance '
     '0.912 ≥ 0.85 · ratio 25:1 24.9 ≤ 25.0 ≤ 25.1 · feet on the ground 2.00 ≥ 2 · output swing span:j_out '
     '120 ≥ 100 · (+6 met — --verbose)'),
    ('ranges j_crank 0…360° · j_lift 0…40.0 mm · j_yaw 0…90.0° · j_out 0…120° · probes '
     'foot_L0…foot_R2 Δ(67.8, 0, 22.5) path 149 mm ×6 · probe tip Δ(10.0, 20.0, 30.0) path 61.0 mm · '
     '(+34: 30 passive, 4 '
     'coupled — --verbose)'),
    ('Δprev: params crank 15→16, leg 50→52 · min clearance 1.10→0.800 mm · mass 731→744 g · stride '
     '66.1→67.8 · step lift 22.1→22.5 · flat stance 0.900→0.912 · output swing 118→120 · motor SF '
     '2.80→2.70 · lift SF 1.80→1.72'),
    ('view http://localhost:3000/mech.html?m=walker&ghost=6&layout=quad&ui=0 · shot: uv run mech shot '
     'walker'),
]
GOLDEN_BIG_FAIL = [
    ('mech walker — FAIL   57 parts · 39 joints (3 driver, 5 coupled, 30 passive, 1 fixed) · 18 loops · '
     '744 g · CoG (0.1, 53.2, 4.3)'),
    ('FAIL interference c_L0/upper_L0 118 mm³ overlap 5.00×6.00×5.00 mm @ c_L0 (12.0, 3.0, 4.0), upper_L0 '
     '(14.0, 3.0, 4.0) · f78 j_crank=272°, j_lift=12.5 mm (3 frames f76–f78) [walk, lift]'),
    ('FAIL interference foot_L1/shin_L1 2.00 mm³ overlap 1.00×2.00×1.00 mm @ (4.0, 1.0, 2.0) · f12 '
     'j_crank=59.2° [walk]'),
    'FAIL target_miss flat stance: 0.812 < min 0.85 (margin −0.0380)',
    'WARN tight p0/q0 gap 0.10 mm < 0.3 @ (1.0, 2.0, 3.0) dir (0, 0, 1) · f5 j_crank=24.7° [walk]',
    'WARN tight p1/q1 gap 0.20 mm < 0.3 @ (1.0, 2.0, 3.0) dir (0, 0, 1) · f5 j_crank=24.7° [walk]',
    'WARN tight p2/q2 gap 0.30 mm < 0.3 @ (1.0, 2.0, 3.0) dir (0, 0, 1) · f5 j_crank=24.7° [walk]',
    'WARN joint_limit jP_L3 52.1° > 45 limit @ j_crank=80.0° (f16; 3 frame(s) f15–f17) [walk]',
    'OK   18 loops closed (max 1.28e-13 mm) · no branch jumps · mobility 0',
    ('clearance min 0.800 mm (c_L0/upper_L0 @f23 j_crank=138° [walk]) · required 0.5 · 1521 pairs checked '
     '· 11 allowed-contact pairs'),
    'load j_crank max 0.148 N·m @f3 (walk) · capacity 0.4 → SF 2.70',
    ('targets 11/12 · stride delta:foot_L0.x 67.8 ≥ 60 · step lift delta:foot_L0.z 22.5 ≥ 15 · ratio 25:1 '
     '24.9 ≤ 25.0 ≤ 25.1 · feet on the ground 2.00 ≥ 2 · output swing span:j_out 120 ≥ 100 · output torque '
     'load:j_out 3.70 ≥ 3.5 · (+5 met — --verbose)'),
    ('Δprev: status PASS→FAIL · params crank 15→16, leg 50→52 · fixed joint_limit jP_R7 · new interference '
     'c_L0/upper_L0, interference foot_L1/shin_L1, tight p0/q0 (+2 more — --verbose) · min clearance '
     '1.10→0.800 mm · mass 731→744 g · (+6 more — --verbose)'),
    '(+8 more: 3 INFO, 4 load, 1 ranges — --verbose)',
    'view http://localhost:3000/mech.html?m=walker&issue=0&ui=0 · shot: uv run mech shot walker',
]


def test_big_summary_folds_every_repetitive_kind():
    """The ≤ 15-line summary of a strandbeest-sized model fits ≤ 250 characters a line: loops counted,
    the clearance pose by its driver, rated loads and the targeted reflected load on their own lines
    with the other reflected loads folded and the round-off load folded as zero, gear meshes and
    contacts folded, identical probes collapsed, passive/coupled ranges counted, callables by
    their label. --verbose unfolds all of it."""
    from mech.report import SUMMARY_WIDTH

    lines = format_summary(_big_report(findings=False)).splitlines()
    assert lines == GOLDEN_BIG_PASS
    assert len(lines) == SUMMARY_LINES and max(map(len, lines)) <= SUMMARY_WIDTH
    verbose = format_summary(_big_report(findings=False), verbose=True).splitlines()
    assert not any(line.startswith("(+") or "--verbose" in line for line in verbose)
    assert sum(line.startswith("INFO gear_mesh planet") for line in verbose) == 6
    assert sum(line.startswith("INFO contact rod") for line in verbose) == 5
    assert "load j_c0 max 1.39 N·m @f3 (walk) (reflected from j_crank) · no actuator capacity" in verbose
    assert "no gravity load on j_yaw (axis ∥ g or balanced)" in verbose  # round-off is zero in verbose too
    assert verbose[next(i for i, line in enumerate(verbose) if line.startswith("OK "))].startswith(
        "OK   loops p0, p1, p2, p3, ")
    ranges = next(line for line in verbose if line.startswith("ranges"))
    assert f"jP_R14 {M}14.5…83.5°" in ranges and "probe foot_R2 Δ(67.8, 0, 22.5) path 149 mm" in ranges
    assert [line for line in verbose if line.startswith("sweep ")] == [
        "sweep walk 96.2 s · 1521 pairs · 900 proximity · 312 exact · 40 booleans · 12 sub-frame poses · slowest "
        "c_L0/upper_L0 3.10 s, a/b 2.00 s, c/d 1.50 s"]


def test_big_summary_keeps_every_finding_first():
    """With FAIL/WARN/target_miss findings the budget goes to them and the context lines (loops,
    clearance, targets, Δprev); what does not fit is tallied by kind."""
    from mech.report import SUMMARY_WIDTH

    lines = format_summary(_big_report(findings=True)).splitlines()
    assert lines == GOLDEN_BIG_FAIL
    assert len(lines) == SUMMARY_LINES and max(map(len, lines)) <= SUMMARY_WIDTH
    report = _big_report(findings=True)
    findings = [i for i in report["issues"] if i["severity"] != "INFO"]
    shown = [line for line in lines if line[:4] in ("FAIL", "WARN")]
    assert len(shown) == len(findings) - 1  # the two c_L0/upper_L0 interferences are one finding
    assert "<lambda>" not in "\n".join(lines) and "callable" not in "\n".join(lines)


def test_a_finding_wider_than_the_cap_is_never_cut():
    """A FAIL line keeps its whole message (it is the fix); a long study list shortens instead."""
    from mech.report import SUMMARY_WIDTH

    msg = ("arm/post 5.00 mm³ overlap 1.00×5.00×1.00 mm @ arm (12.0, 3.0, 4.0), post (14.0, 3.0, 4.0) · f78 "
           + ", ".join(f"j_drive{k}={k}0.0°" for k in range(12)) + " — meshing pair: check center distance/backlash")
    studies = [_study(n) for n in ("walk", "trot", "gallop", "turn")]
    issues = [_issue("FAIL", "interference", msg, study=s["name"], parts=("arm", "post")) for s in studies]
    lines = format_summary(_report(status="FAIL", issues=issues, studies=studies,
                                   mobility={s["name"]: 0 for s in studies})).splitlines()
    assert lines[1] == f"FAIL interference {msg} [walk +3]" and len(lines[1]) > SUMMARY_WIDTH
    assert all(len(line) <= SUMMARY_WIDTH for i, line in enumerate(lines) if i != 1)
    assert format_summary(_report(status="FAIL", issues=issues[:1])).splitlines()[1] == f"FAIL interference {msg}"


def test_view_line_never_cuts_the_shot_command():
    """An export outside <repo>/output names the --output-dir in the shot command, whole, however long
    the path: the prose around it shortens to keep the line within the width while it can."""
    from mech.report import SUMMARY_WIDTH

    for n, prose in ((20, "view: exported to C:\\"), (120, "view: exported outside <repo>/output (not served by "
                                                                "the viewer) · "),
                     (170, "view: not served · "), (300, "view: not served · ")):
        out_dir = "C:\\" + "d" * n
        view = format_summary(_report(out_dir=out_dir)).splitlines()[-1]
        assert view.endswith(f" · shot: uv run mech shot four_bar --output-dir {out_dir}"), view
        assert view.startswith(prose) and (len(view) <= SUMMARY_WIDTH) == (n < 300)
    assert format_summary(_report(out_dir="C:\\x")).splitlines()[-1] == (
        "view: exported to C:\\x (the viewer serves <repo>/output only) · shot: uv run mech shot four_bar "
        "--output-dir C:\\x")


def test_invalid_summary_shows_every_error_and_counts_the_rest():
    """Every model error is its own line, even five about one joint; past the budget the rest are
    counted with the --verbose hint."""
    five = invalid_report("arm", [f"revolute 'j_arm': problem {k}" for k in range(5)], {})
    lines = format_summary(five).splitlines()
    assert lines[0] == "mech arm — INVALID   5 errors" and len(lines) == 7
    assert lines[1:6] == [f"FAIL invalid_model revolute 'j_arm': problem {k}" for k in range(5)]
    twenty = invalid_report("arm", [f"revolute 'j_{k}': unknown parent part 'x'" for k in range(20)], {})
    lines = format_summary(twenty).splitlines()
    assert len(lines) == SUMMARY_LINES and lines[0] == "mech arm — INVALID   20 errors"
    assert sum(line.startswith("FAIL invalid_model") for line in lines) == 12
    assert lines[-2:] == ["(+8 more: 8 FAIL invalid_model — --verbose)",
                          "nothing analyzed — fix the errors above and re-run"]
    assert len(format_summary(twenty, verbose=True).splitlines()) == 22


def test_callable_targets_are_shown_by_their_label(tmp_path):
    """A callable target reads as its label in the targets line and in a miss — never ``<lambda>()``
    or a function name — and report.json records its metric as "callable"."""
    asm = make_four_bar()
    asm.target("half the mass", lambda r: r["mass"]["total_g"] / 2, min=1.0)

    def rocker_ratio(r):
        return 0.5

    asm.target("ratio", rocker_ratio, min=1.0)
    rep = analyze(asm, export=False, out_root=tmp_path)
    assert [t["metric"] for t in rep["targets"]] == ["span:j_rocker", "callable", "callable"]
    text = format_summary(rep)
    assert "half the mass 21.2 ≥ 1" in text and f"FAIL target_miss ratio: 0.500 < min 1 (margin {M}0.500)" in text
    assert "<lambda>" not in json.dumps(rep, ensure_ascii=False) and "rocker_ratio" not in text


def test_round_off_loads_are_zero():
    """A holding load at round-off level (gravity along the axis: 3.4e-17 N·m beside a 0.15 N·m
    actuator) is stored as exactly 0 and folds into the no-gravity line; its SF is unbounded."""
    from mech.report import _load_entries
    from mech.targets import is_zero_load, load_scale

    loads = {"turn": {"j_crank": {"unit": "N·m", "max_abs": 0.148, "frame": 3},
                      "j_yaw": {"unit": "N·m", "max_abs": 3.42e-17, "frame": 7},
                      "j_swing": {"unit": "N·m", "max_abs": 1.07e-14, "frame": 2}}}
    scale = load_scale(loads)
    assert scale == 0.148 and is_zero_load(1.07e-14, scale) and not is_zero_load(1e-6, scale)
    assert is_zero_load(1e-12, 0.0) and not is_zero_load(None, scale)  # an absolute floor when all are tiny

    class _Asm:  # what _load_entries reads
        actuators = {"j_crank": type("A", (), {"capacity": 0.4})(), "j_yaw": type("A", (), {"capacity": 0.2})()}

    entries = _load_entries(_Asm(), loads["turn"], scale=scale)
    assert entries["j_yaw"]["max_abs"] == 0.0 and entries["j_yaw"]["sf"] is None
    assert entries["j_swing"]["max_abs"] == 0.0 and entries["j_crank"]["max_abs"] == 0.148
    study = _study(load=0.148, frame=3)
    study["loads"]["j_yaw"] = {"unit": "N·m", "max_abs": 3.42e-17, "frame": 7, "capacity": 0.2, "sf": 5.8e15,
                               "reflected_from": None}  # an older report.json: the summary folds it too
    lines = format_summary(_report(studies=[study])).splitlines()
    assert "no gravity load on j_yaw (axis ∥ g or balanced)" in lines
    assert not any("e-17" in line or "e15" in line for line in lines)


def test_sweep_stats_are_in_the_report_and_the_verbose_summary(tmp_path):
    from build123d import Box, Pos

    from mech import Assembly

    asm = Assembly("stats", clearance=0.3)
    asm.part("base", Pos(0, 0, -3) * Box(80, 80, 4), ground=True)
    asm.part("arm", Pos(20, 0, 2) * Box(40, 6, 4))
    asm.revolute("j", "base", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.study("swing", drive={"j": (0, 90)}, frames=5)
    rep = analyze(asm, export=False, out_root=tmp_path)
    stats = rep["studies"][0]["sweep_stats"]
    assert isinstance(stats, dict) and stats["seconds"] > 0
    json.dumps(stats, allow_nan=False)
    perf = [line for line in format_summary(rep, verbose=True).splitlines() if line.startswith("sweep swing ")]
    assert len(perf) == 1 and perf[0].split(" · ")[0].endswith(" s")
    assert not any(line.startswith("sweep ") for line in format_summary(rep).splitlines())  # --verbose only


def test_pose_of_a_finding_names_the_coupled_joints_on_its_parts_chain(tmp_path):
    """A driver that turns many coupled joints (a planetary's sun) is named with only the coupled
    outputs that pose the finding's parts; the clearance line names drivers only."""
    from build123d import Box, Pos

    from mech import Assembly

    asm = Assembly("coupled", clearance=0.3)
    asm.part("base", Pos(0, 0, -3) * Box(200, 200, 4), ground=True)
    asm.part("drive", Pos(0, 0, 5) * Box(10, 10, 4))
    asm.revolute("j_in", "base", "drive", origin=(0, 0, 0), axis=(0, 0, 1))
    for k, x in enumerate((-60, 60)):
        asm.part(f"arm{k}", Pos(x + 15, 0, 5) * Box(30, 4, 4))
        asm.revolute(f"j_arm{k}", "base", f"arm{k}", origin=(x, 0, 0), axis=(0, 0, 1))
        asm.couple("j_in", f"j_arm{k}", ratio=1.0)
    asm.part("block", Pos(-60, 25, 5) * Box(6, 6, 4), ground=True)  # in arm0's path at 90°
    asm.study("turn", drive={"j_in": (0, 120)}, frames=7)
    rep = analyze(asm, export=False, out_root=tmp_path)
    hit = next(i for i in rep["issues"] if i["code"] == "interference")
    assert hit["parts"] == ["arm0", "block"] or hit["parts"] == ["block", "arm0"]
    assert "(j_arm0=" in hit["message"] and "j_arm1" not in hit["message"]
    clear = next(line for line in format_summary(rep).splitlines() if line.startswith("clearance"))
    assert "j_in=" in clear and "j_arm" not in clear


# ------------------------------------------------------------------------------ review round 3


def test_alike_pair_findings_fold_into_one_line():
    """strandbeest -p gap=0.2: mirrored pairs all tight at the same gap over the same frames are
    one finding — one line naming the worst in full and the others after it — so the load and
    ranges lines keep their room. Different gaps stay apart; --verbose lists every pair."""
    span = "(61 frames f0–f60)"
    alike = [_issue("WARN", "tight_clearance", f"{a}/{b} gap 0.200 mm < 0.5 @ (0, 0, 0) · f{k} j_crank=0° {span}",
                    parts=(a, b), value=0.2 + k * 1e-6, frame=k)
             for k, (a, b) in enumerate([("c_R2", "upper_R2"), ("c_L0", "upper_L0"), ("k_L0", "k_R0"),
                                         ("c_R0", "upper_R0"), ("k_L2", "k_R2")])]
    other = _issue("WARN", "tight_clearance", f"x/y gap 0.400 mm < 0.5 @ (0, 0, 0) · f3 j_crank=0° {span}",
                   parts=("x", "y"), value=0.4)
    rep = _report(status="WARN", issues=alike + [other])
    lines = format_summary(rep).splitlines()
    warn = [line for line in lines if line.startswith("WARN")]
    assert warn == [f"WARN tight 5 pairs alike: {alike[0]['message']} · also c_L0/upper_L0, k_L0/k_R0 "
                    f"(+2 more — --verbose)", f"WARN tight {other['message']}"]
    assert any(line.startswith("load j_crank") for line in lines) and any(line.startswith("ranges") for line in lines)
    assert len([line for line in format_summary(rep, verbose=True).splitlines() if line.startswith("WARN")]) == 6
    assert len([line for line in format_check(rep).splitlines() if line.startswith("WARN")]) == 2


def test_delta_prev_never_says_no_change_over_findings_it_left_out():
    """A --frames run FAILs where the full run passed: the studies are sampled differently, so the
    issue is not compared — and the Δprev line says so instead of "no change"."""
    from mech.report import _delta_prev, _format_delta

    prev = _report(status="PASS", delta_prev=None)
    now = _report(status="FAIL", studies=[dict(_study(), frames=5)],
                  issues=[_issue("FAIL", "interference", "crank/rocker 3.00 mm³ overlap", value=3.0)])
    d = _delta_prev(now, prev)
    assert d["not_compared_issues"] == {"FAIL": 1} and "status" not in d
    assert _format_delta(d) == "Δprev: no change in the shared scope (not compared: 1 FAIL, turn (72→5 frames))"
    d = _delta_prev(_report(studies=[dict(_study(), frames=5)]), prev)
    assert "not_compared_issues" not in d and _format_delta(d) == "Δprev: no change (not compared: turn (72→5 frames))"


def test_unresolved_intervals_are_an_info_line_and_a_verbose_stat():
    """The sweep's sub_unresolved count surfaces: an INFO naming the pairs (the frames alone vouch
    for those intervals) and `· N unresolved` on the --verbose sweep line."""
    from types import SimpleNamespace

    from mech.report import _perf_lines, _unresolved_issues

    stats = {"seconds": 0.3, "pairs": 3, "sub_unresolved": 8, "unresolved_pairs": [["lid_left", "lid_right", 8]]}
    (issue,) = _unresolved_issues(SimpleNamespace(name="open"), SimpleNamespace(stats=stats))
    assert (issue.severity, issue.code, issue.parts, issue.value) == ("INFO", "subframe_unresolved",
                                                                      ["lid_left", "lid_right"], 8.0)
    assert issue.message.startswith("8 between-frame intervals not proven clear (lid_left/lid_right)")
    assert _unresolved_issues(SimpleNamespace(name="open"), SimpleNamespace(stats={"sub_unresolved": 0})) == []
    rep = _report(studies=[dict(_study(), sweep_stats=stats)], issues=[issue.to_dict()])
    assert any(line.startswith("INFO unresolved 8 between-frame intervals") for line in format_summary(rep).splitlines())
    assert _perf_lines(rep)[0].text == "sweep turn 0.300 s · 3 pairs · 8 unresolved"


def test_equality_target_prints_its_value_at_the_bounds_precision():
    """planetary: an equality target met at −0.333333 must not read `−0.333333 ≤ −0.333 ≤ −0.333333`."""
    def target(label, value, lo, hi):
        return {"label": label, "metric": "callable", "value": value, "min": lo, "max": hi, "met": True,
                "severity": "FAIL", "study": None, "worst_study": None, "error": None}

    rep = _report(targets=[target("ratio 25:1", 25.0, 25.0, 25.0), target("planet spin", -0.333333, -0.333333,
                                                                                -0.333333),
                           target("band", 44.0, 43.12, 44.88)])
    line = next(line for line in format_summary(rep).splitlines() if line.startswith("targets"))
    assert line == f"targets 3/3 · ratio 25:1 25.0 = 25 · planet spin {M}0.333333 = {M}0.333333 · band 43.12 ≤ 44.0 ≤ 44.88"


def test_check_roles_line_names_drivers_first_and_counts_the_rest():
    """desktop_arm: 36 joints — the drivers and coupled joints by name, the passive and fixed
    joints counted, never 8 fixed joints shown while 3 of 4 drivers hide behind '+N more'."""
    roles = {f"fix_{k}": "fixed" for k in range(8)}
    roles |= {"j_yaw": "coupled", "j_yaw_motor": "driver", **{f"fix_more_{k}": "fixed" for k in range(15)},
              "j_shoulder_motor": "driver", "j_elbow_motor": "driver", "j_roll": "driver", "j_shoulder": "coupled",
              "j_elbow": "coupled", **{f"j_passive_{k}": "passive" for k in range(6)}}
    text = format_check(_report(roles=roles, joints={"driver": 4, "coupled": 3, "passive": 6, "free": 0, "fixed": 23}))
    (line,) = [line for line in text.splitlines() if line.startswith("roles")]
    assert line == ("roles driver j_yaw_motor, j_shoulder_motor, j_elbow_motor, j_roll · coupled j_yaw, j_shoulder, "
                    "j_elbow · (+29: 6 passive, 23 fixed — --verbose)")
    full = format_check(_report(roles=roles), verbose=True)
    assert "passive j_passive_0, j_passive_1" in full and "fixed fix_0, fix_1" in full
    assert text.splitlines()[-1] == "next: `uv run mech run` the same script to run the studies"
    assert format_check(_report(), command="uv run mech run x.py").splitlines()[-1] == \
        "next: uv run mech run x.py to run the studies"
