"""cli.py: exit codes, -p parsing, trimmed tracebacks, run/check/sweep/list/shot plumbing."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from mech.cli import main
from mech.shot import ShotError, build_query, dist_is_current, resolve_name, serve
from mech.sweep import grid, parse_axis

FIXTURES = Path(__file__).parent / "fixtures"
ARM = str(FIXTURES / "arm.py")
TWO = str(FIXTURES / "two_studies.py")
RAISES = str(FIXTURES / "raises.py")
REPO = Path(__file__).resolve().parents[1]


def _run(capsys, *argv):
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize("params, status, code", [
    ([], "PASS", 0),
    (["-p", "need=100", "severity='WARN'"], "WARN", 1),
    (["-p", "need=100"], "FAIL", 2),
    (["-p", "bump=True"], "FAIL", 2),
    (["-p", "bad_joint=True"], "INVALID", 3),
])
def test_run_exit_codes(capsys, tmp_path, params, status, code):
    got, out, err = _run(capsys, "run", ARM, *params, "--output-dir", tmp_path)
    assert got == code, out + err
    assert out.splitlines()[0].startswith(f"mech arm — {status}")
    assert len(out.splitlines()) <= 15 and err == ""
    report = json.loads((tmp_path / "arm.mech" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == status


def test_invalid_names_the_problem(capsys, tmp_path):
    code, out, _ = _run(capsys, "run", ARM, "-p", "bad_joint=True", "--output-dir", tmp_path)
    assert code == 3
    assert "FAIL invalid_model revolute 'j_arm': unknown parent part 'bsae' (did you mean 'base'?)" in out
    assert not (tmp_path / "arm.mech" / "scene.json").exists()  # nothing analyzed, only the report


def test_params_are_literals_with_string_fallback(capsys, tmp_path):
    code, out, _ = _run(capsys, "run", ARM, "-p", "swing=30", "label=my arm", "-p", "need=12.5",
                        "--json", "--output-dir", tmp_path)
    assert code == 0
    report = json.loads(out)
    assert report["params"] == {"swing": 30, "need": 12.5, "severity": "FAIL", "bad_joint": False, "bump": False,
                                "label": "my arm"}
    assert isinstance(report["params"]["swing"], int)
    assert report["studies"][0]["joint_ranges"]["j_arm"] == [0.0, 30.0]
    assert (tmp_path / "my_arm.mech" / "scene.json").is_file()


def test_unknown_param_lists_valid_ones(capsys, tmp_path):
    code, out, err = _run(capsys, "run", ARM, "-p", "lenght=3", "--output-dir", tmp_path)
    assert code == 3 and out == ""
    assert "unknown param lenght — build() accepts: swing=90.0, need=45.0, severity='FAIL'" in err
    code, _, err = _run(capsys, "run", ARM, "-p", "novalue", "--output-dir", tmp_path)
    assert code == 3 and "bad -p 'novalue'" in err


def test_script_exception_is_trimmed_to_user_frames(capsys, tmp_path):
    code, out, err = _run(capsys, "run", RAISES, "--output-dir", tmp_path)
    assert code == 3 and out == ""
    lines = err.strip().splitlines()
    assert len(lines) <= 6
    assert lines[0] == "mech: error in raises.py build():"
    assert lines[-1] == "ZeroDivisionError: float division by zero"
    assert any('raises.py", line 6, in _arm_length' in line for line in lines)
    assert "cli.py" not in err and "runner.py" not in err
    code, _, err = _run(capsys, "run", RAISES, "--verbose", "--output-dir", tmp_path)
    assert code == 3 and "cli.py" in err and "Traceback (most recent call last)" in err


def test_missing_script_and_bad_arguments(capsys, tmp_path):
    code, _, err = _run(capsys, "run", tmp_path / "nope.py", "--output-dir", tmp_path)
    assert code == 3 and "no such script" in err
    with pytest.raises(SystemExit) as exc:  # argparse usage errors exit 3, not argparse's default 2 (= FAIL)
        main(["run"])
    assert exc.value.code == 3
    capsys.readouterr()


def test_no_export_and_study_filter(capsys, tmp_path):
    code, out, _ = _run(capsys, "run", ARM, "--no-export", "--output-dir", tmp_path)
    assert code == 0 and not (tmp_path / "arm.mech").exists()
    assert out.splitlines()[-1] == "view: not exported (drop --no-export to write output/arm.mech)"
    code, out, _ = _run(capsys, "run", ARM, "--no-export", "--frames", "4", "--output-dir", tmp_path)
    assert code == 0 and not (tmp_path / "arm.mech").exists()
    assert out.splitlines()[0].startswith("mech arm — PASS (partial run: --frames 4)   ")
    assert out.splitlines()[-1].startswith("view: not exported — partial run; arm.mech keeps the last full run")
    code, out, _ = _run(capsys, "run", ARM, "--study", "swnig", "--output-dir", tmp_path)
    assert code == 3 and "study 'swnig' (requested): no such study (did you mean 'swing'?)" in out


def test_delta_prev_across_runs(capsys, tmp_path):
    _run(capsys, "run", ARM, "--output-dir", tmp_path)
    code, out, _ = _run(capsys, "run", ARM, "-p", "bump=True", "--output-dir", tmp_path)
    assert code == 2
    dprev = next(line for line in out.splitlines() if line.startswith("Δprev"))
    assert dprev.startswith("Δprev: status PASS→FAIL · params bump False→True · new interference arm/block")
    _, out, _ = _run(capsys, "run", ARM, "--output-dir", tmp_path)
    assert "Δprev: status FAIL→PASS · params bump True→False · fixed interference arm/block" in out


def test_partial_runs_never_replace_the_last_full_run(capsys, tmp_path):
    """--study / --frames runs say they are partial, keep report.json/scene.json (the last full run),
    and their Δprev never calls an issue of a skipped or differently sampled study "fixed"."""
    mech = tmp_path / "two.mech"
    code, out, _ = _run(capsys, "run", TWO, "--output-dir", tmp_path)
    assert code == 2 and "FAIL interference post/arm" in out
    saved = {f: (mech / f).read_bytes() for f in ("report.json", "scene.json")}

    code, out, _ = _run(capsys, "run", TWO, "--study", "back", "--output-dir", tmp_path)
    lines = out.splitlines()
    assert code == 0 and lines[0].startswith("mech two — PASS (partial run: --study back, skipped sweep)   ")
    dprev = next(line for line in lines if line.startswith("Δprev"))
    assert dprev == "Δprev: no change (not compared: sweep (not run))"  # not "fixed interference arm/post"
    assert lines[-1].startswith("view: not exported — partial run")
    assert {f: (mech / f).read_bytes() for f in saved} == saved
    partial = json.loads((mech / "report.partial.json").read_text(encoding="utf-8"))
    assert partial["partial"] == {"studies": ["back"], "skipped": ["sweep"], "frames": None}
    code, out, _ = _run(capsys, "list", "--output-dir", tmp_path)
    assert out.split()[:2] == ["two", "FAIL"]  # the full run's status

    # the same partial run again compares with the previous partial run in full
    code, out, _ = _run(capsys, "run", TWO, "--study", "back", "-p", "post_r=35", "--output-dir", tmp_path)
    assert "Δprev: params post_r 30→35" in out and "not compared" not in out

    # a coarser sampling (0/30/60/90°) is not compared with the 19-frame run either; the sweep finds
    # the hit between frames 1 and 2, and the issue gives the pose there, not a sampled frame's
    code, out, _ = _run(capsys, "run", TWO, "--frames", "4", "--output-dir", tmp_path)
    assert code == 2 and "(partial run: --frames 4)" in out.splitlines()[0]
    # the coarse run FAILs where the full run did too, but that is not compared: never "no change"
    assert ("Δprev: no change in the shared scope (not compared: 1 FAIL, sweep (19→4 frames), back (19→4 frames))"
            in out)
    assert "· between f1–f2 j=45.0° [sweep]" in out
    assert {f: (mech / f).read_bytes() for f in saved} == saved

    # a full run replaces both files and drops the stale partial report
    _run(capsys, "run", TWO, "-p", "post_r=35", "--output-dir", tmp_path)
    assert not (mech / "report.partial.json").exists()
    assert (mech / "report.json").read_bytes() != saved["report.json"]


def test_partial_run_only_lists_as_partial(capsys, tmp_path):
    _run(capsys, "run", TWO, "--study", "back", "--output-dir", tmp_path)
    code, out, _ = _run(capsys, "list", "--output-dir", tmp_path)
    assert code == 0 and out.split()[:2] == ["two", "PASS"] and "partial run only" in out


def test_invalid_full_run_removes_the_stale_scene(capsys, tmp_path):
    _run(capsys, "run", ARM, "--output-dir", tmp_path)
    mech = tmp_path / "arm.mech"
    assert (mech / "scene.json").is_file() and list((mech / "parts").glob("*.stl"))
    code, _, _ = _run(capsys, "run", ARM, "-p", "bad_joint=True", "--output-dir", tmp_path)
    assert code == 3
    assert not (mech / "scene.json").exists() and not list((mech / "parts").glob("*.stl"))
    code, _, err = _run(capsys, "shot", "arm", "--output-dir", tmp_path)
    assert code == 3 and "the last run of arm.mech was INVALID" in err and "unknown parent part 'bsae'" in err


def test_shot_refuses_a_stale_scene(capsys, tmp_path):
    _run(capsys, "run", ARM, "--output-dir", tmp_path)
    report_path = tmp_path / "arm.mech" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "FAIL"  # e.g. a report.json written by another run than scene.json
    report_path.write_text(json.dumps(report), encoding="utf-8")
    code, _, err = _run(capsys, "shot", "arm", "--output-dir", tmp_path)
    assert code == 3 and "is stale" in err and "status PASS vs FAIL" in err


def test_default_output_dir_is_the_repo_output(tmp_path, monkeypatch):
    """`run(build())` from any working directory exports where the viewer and `mech shot` look; an
    explicit other directory gets a shot hint that points at it."""
    from mech.report import format_summary
    from mech.runner import OUTPUT_DIR, analyze

    assert OUTPUT_DIR == REPO / "output"
    spec = importlib.util.spec_from_file_location("_cli_arm", ARM)
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)
    monkeypatch.chdir(tmp_path)
    report = analyze(arm.build(label="zz_cwd_probe"), export=False)
    assert report["out_dir"] is None and not (tmp_path / "output").exists()
    report = analyze(arm.build(label="zz_cwd_probe"), out_root=tmp_path / "elsewhere")
    assert report["out_dir"] == str((tmp_path / "elsewhere").resolve())
    view = format_summary(report).splitlines()[-1]
    assert view.endswith(f"shot: uv run mech shot zz_cwd_probe --output-dir {report['out_dir']}")


def test_check(capsys, tmp_path):
    code, out, _ = _run(capsys, "check", ARM, "-p", "bump=True", "--output-dir", tmp_path)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("mech check arm — PASS   3 parts · 2 joints (1 driver, 1 free)")
    assert "roles driver j_arm · free j_block" in lines and "studies swing mobility 0" in lines
    # the next step is the real command line, ready to copy: script, -p params, --output-dir
    assert lines[-1] == f"next: uv run mech run {ARM} -p bump=True --output-dir {tmp_path} to run the studies"
    assert not (tmp_path / "arm.mech").exists()  # check never exports
    code, out, _ = _run(capsys, "check", ARM, "-p", "bad_joint=True", "--output-dir", tmp_path)
    assert code == 3 and "INVALID" in out


def test_list(capsys, tmp_path):
    code, out, _ = _run(capsys, "list", "--output-dir", tmp_path)
    assert code == 0 and "no mechanisms" in out
    _run(capsys, "run", ARM, "-p", "need=100", "--output-dir", tmp_path)
    code, out, _ = _run(capsys, "list", "--output-dir", tmp_path)
    assert code == 0 and out.split()[:5] == ["arm", "FAIL", "1", "FAIL", "·"]


def test_sweep_grid_and_table(capsys, tmp_path):
    assert parse_axis("a=1:3:1") == ("a", [1, 2, 3])
    assert parse_axis("a=0:0.3:0.1") == ("a", [0.0, 0.1, 0.2, 0.3])
    assert parse_axis("m=steel,'PLA',2") == ("m", ["steel", "PLA", 2])
    assert grid([("a", [1, 2]), ("b", ["x", "y", "z"])])[:4] == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 1, "b": "z"}, {"a": 2, "b": "x"}]
    for bad in ("a", "a=", "a=3:1:1", "a=0:1:0", "a=x:2:1"):
        with pytest.raises(ValueError):
            parse_axis(bad)
    with pytest.raises(ValueError, match="exceeds the cap of 50"):
        grid([("a", list(range(8))), ("b", list(range(7)))])

    code, out, _ = _run(capsys, "sweep", ARM, "swing=30:90:30", "need=45,60", "--frames", "4",
                        "--output-dir", tmp_path)
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "mech sweep arm.py — 6 variants"
    assert lines[1].split() == ["swing", "need", "status", "F/W", "min", "clr", "worst", "SF", "swing", "why"]
    rows = [line.split() for line in lines[2:8]]
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [
        ("30", "45", "FAIL", "1/0"), ("30", "60", "FAIL", "1/0"), ("60", "45", "PASS", "0/0"),
        ("60", "60", "PASS", "0/0"), ("90", "45", "PASS", "0/0"), ("90", "60", "PASS", "0/0")]
    assert [r[6] for r in rows] == ["30.0", "30.0", "60.0", "60.0", "90.0", "90.0"]  # target value column
    assert [" ".join(r[7:]) for r in rows] == ["target_miss swing"] * 2 + [""] * 4  # why a row fails
    assert lines[8] == "best swing=60 need=45 — PASS (--export-best to export it)"
    assert not (tmp_path / "arm.mech").exists()

    code, out, _ = _run(capsys, "sweep", ARM, "need=100,120", "--export-best", "--frames", "4",
                        "--output-dir", tmp_path)
    assert code == 2 and "best need=100 — FAIL · exported (full run at the declared frames)" in out
    report = json.loads((tmp_path / "arm.mech" / "report.json").read_text(encoding="utf-8"))
    assert report["partial"] is None and report["studies"][0]["frames"] == 10  # not the sweep's 4
    assert (tmp_path / "arm.mech" / "scene.json").is_file()
    code, _, err = _run(capsys, "sweep", ARM, "wobble=1,2", "--output-dir", tmp_path)
    assert code == 3 and "unknown param wobble" in err
    code, _, err = _run(capsys, "sweep", ARM, "swing=1:60:1", "--output-dir", tmp_path)
    assert code == 3 and "cap 50" in err


def test_console_script_utf8_through_a_pipe(tmp_path):
    """The installed `mech` entry point prints UTF-8 even when stdout is a cp1252 pipe."""
    env = {**os.environ, "PYTHONIOENCODING": ""}
    env.pop("PYTHONUTF8", None)
    proc = subprocess.run([sys.executable, "-m", "mech.cli", "run", ARM, "--output-dir", str(tmp_path)],
                          capture_output=True, env=env, cwd=REPO, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")
    assert text.startswith("mech arm — PASS") and "ranges j_arm 0…90.0°" in text
    from importlib.metadata import entry_points

    assert any(ep.value == "mech.cli:main" for ep in entry_points(group="console_scripts", name="mech"))


# ------------------------------------------------------------------------------ shot plumbing


def test_shot_query_and_names(tmp_path):
    assert resolve_name("four_bar", tmp_path) == ("four_bar", tmp_path / "four_bar.mech")
    assert resolve_name("output/Four Bar.mech/", tmp_path) == ("four_bar", tmp_path / "four_bar.mech")
    report = {"viewer_url": "http://localhost:3000/mech.html?m=arm&issue=2&ui=0"}
    assert build_query("arm", report) == {"m": "arm", "issue": "2", "ui": "0"}
    assert build_query("arm", report, view="top", frame=4) == {"m": "arm", "view": "top", "frame": "4", "ui": "0"}
    assert build_query("arm", None) == {"m": "arm", "ui": "0"}


def test_shot_options_become_viewer_params_and_the_file_name():
    """Every viewer parameter is an option (lists joined, booleans 1/0, floats short); raw params pass
    through; values the viewer would reject fail before a browser starts; the file name encodes
    the options in a fixed order."""
    from mech.shot import OPTIONS, shot_suffix

    assert set(OPTIONS) == {"issue", "study", "frame", "q", "view", "cam", "zoom", "section", "focus", "isolate",
                            "hide", "explode", "axes", "ghost", "paths", "layout"}
    q = build_query("exc", {"viewer_url": "http://x/mech.html?m=exc&issue=0&ui=0"}, study="dig",
                    q=["j_boom:30", "j_stick:-10"], view="right", cam="30,20", zoom=2.0, section="-x:12.5",
                    focus=["bucket", "stick"], isolate="bucket,stick,boom", hide=["cab"], explode=0.5, axes=True,
                    ghost=3, paths=False, layout="quad", params={"ui": "1", "extra": "y"})
    assert q == {"m": "exc", "study": "dig", "q": "j_boom:30,j_stick:-10", "view": "right",
                 "cam": "30,20", "zoom": "2", "section": "-x:12.5", "focus": "bucket,stick",
                 "isolate": "bucket,stick,boom", "hide": "cab", "explode": "0.5", "axes": "1", "ghost": "3",
                 "paths": "0", "layout": "quad", "extra": "y", "ui": "0"}  # explicit options replace issue=0
    # q and frame both pick the frame: the viewer would silently drop the q pose, so it is an error
    with pytest.raises(ShotError, match="both pick the frame"):
        build_query("exc", None, study="dig", frame=40, q="j_boom:30")
    assert build_query("exc", None, frame="home", study="dig") == {"m": "exc", "study": "dig", "frame": "home",
                                                                   "ui": "0"}
    assert shot_suffix(build_query("exc", None, study="dig", frame=40, view="top")) == "dig_f40_top"
    assert shot_suffix(build_query("exc", None, cam="30,20", zoom=1.5)) == "cam30_20_z1_5"
    assert shot_suffix(build_query("exc", None, isolate="bucket", axes=True, paths=0)) == "only_bucket_axes_paths0"
    assert shot_suffix(build_query("exc", None, params={"foo": "bar"})) == "foo_bar"
    assert shot_suffix(build_query("exc", None)) == "default"
    long = [build_query("exc", None, hide=f"{'part_' * 12}{k}", frame=k) for k in range(2)]
    names = [shot_suffix(x) for x in long]
    assert all(len(n) <= 48 for n in names) and names[0] != names[1]  # cut, but never shared
    for bad, match in [({"view": "diagonal"}, "bad view"), ({"frame": -1}, "bad frame"), ({"frame": "x"}, "bad frame"),
                       ({"q": "j_boom"}, "bad q"), ({"q": "j:abc"}, "bad q"), ({"cam": "30"}, "bad cam"),
                       ({"cam": "0,95"}, "elevation"), ({"section": "w:3"}, "bad section"), ({"zoom": 0}, "bad zoom"),
                       ({"explode": -1}, "bad explode"), ({"axes": "maybe"}, "bad axes"), ({"ghost": 1.5}, "bad ghost"),
                       ({"layout": "grid"}, "bad layout"), ({"colour": "red"}, "unknown shot option colour")]:
        with pytest.raises(ShotError, match=match):
            build_query("exc", None, **bad)


def test_shot_cli_flags_reach_the_viewer(capsys, monkeypatch, tmp_path):
    """`mech shot` flags (repeatable lists merged, --axes without a value, raw --param k=v) arrive as
    the viewer options; a malformed --param is a usage error."""
    import mech.cli as cli

    seen = {}

    def fake_shot(name, **kw):
        seen.update(kw, name=name)
        return tmp_path / "shot.png"

    monkeypatch.setattr(cli, "shot", fake_shot)
    code, out, _ = _run(capsys, "shot", "exc", "--study", "dig", "--frame", "40", "--q", "j_boom:30", "--q",
                        "j_stick:-10", "--focus", "bucket,stick", "--focus", "boom", "--hide", "cab", "--axes",
                        "--paths", "0", "--zoom", "2", "--cam", "-30,20", "--section", "-x:12.5", "--explode", "0.5",
                        "--ghost", "3", "--layout", "quad", "--view", "right", "--param", "extra=y",
                        "--output-dir", tmp_path)
    assert code == 0 and out.strip() == str(tmp_path / "shot.png")
    assert seen["name"] == "exc" and seen["params"] == {"extra": "y"}
    assert {k: seen[k] for k in ("study", "frame", "q", "focus", "hide", "axes", "paths", "zoom", "cam", "section",
                                 "explode", "ghost", "layout", "view", "isolate", "issue")} == {
        "study": "dig", "frame": "40", "q": "j_boom:30,j_stick:-10", "focus": "bucket,stick,boom", "hide": "cab",
        "axes": "1", "paths": "0", "zoom": 2.0, "cam": "-30,20", "section": "-x:12.5", "explode": 0.5, "ghost": 3,
        "layout": "quad", "view": "right", "isolate": None, "issue": None}
    code, _, err = _run(capsys, "shot", "exc", "--param", "novalue", "--output-dir", tmp_path)
    assert code == 3 and "bad --param 'novalue'" in err
    monkeypatch.undo()
    _run(capsys, "run", ARM, "--output-dir", tmp_path)
    code, _, err = _run(capsys, "shot", "arm", "--cam", "0,95", "--output-dir", tmp_path)
    assert code == 3 and "bad cam=0,95: elevation must be within ±90°" in err  # before any browser starts
    for argv, message in ((["--study", "swnig"], "unknown study 'swnig' — this run has swing"),
                          (["--frame", "10"], "frame 10 is past the last frame of study 'swing' (frames 0…9)"),
                          (["--study", "swing", "--frame", "12"], "frame 12 is past the last frame of study 'swing'"),
                          (["--issue", "0"], "issue 0 does not exist (the report lists 0)")):
        code, _, err = _run(capsys, "shot", "arm", *argv, "--output-dir", tmp_path)
        assert code == 3 and message in err, err


def test_shot_server_serves_dist_and_output_only(tmp_path):
    dist, output = tmp_path / "dist", tmp_path / "output"
    (dist / "assets").mkdir(parents=True)
    (output / "x.mech").mkdir(parents=True)
    (dist / "mech.html").write_text("<html></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("export {};", encoding="utf-8")
    (output / "x.mech" / "scene.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    with serve(dist, output) as base:
        def get(path):
            with urllib.request.urlopen(base + path, timeout=10) as r:
                return r.status, r.headers["Content-Type"], r.read()

        assert get("/mech.html")[0] == 200
        assert get("/assets/app.js")[1].startswith("text/javascript")  # module scripts need a JS type
        status, ctype, body = get("/output/x.mech/scene.json")
        assert (status, body) == (200, b"{}") and ctype.startswith("application/json")
        for sneaky in ("/../secret.txt", "/output/../../secret.txt", "/output/%2e%2e/%2e%2e/secret.txt"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                get(sneaky)
            assert exc.value.code == 404
            exc.value.close()  # an HTTPError holds the response socket open


def test_shot_missing_scene_is_a_clear_error(capsys, tmp_path):
    code, _, err = _run(capsys, "shot", "nothing", "--output-dir", tmp_path)
    assert code == 3 and "no scene at" in err and "mech run" in err
    with pytest.raises(ShotError, match="bad view"):
        from mech.shot import shot

        shot("x", view="diagonal", output_dir=tmp_path)


def test_shot_server_swallows_aborted_connections(capsys, monkeypatch, tmp_path):
    """A browser that drops a connection mid-response (WinError 10053/10054 as the viewer page tears
    down) is no error: no traceback on stderr, and the server keeps serving."""
    import http.client
    from http.server import SimpleHTTPRequestHandler

    from mech.shot import _Server

    dist, output = tmp_path / "dist", tmp_path / "output"
    dist.mkdir()
    output.mkdir()
    (dist / "mech.html").write_text("<html></html>", encoding="utf-8")
    real_get = SimpleHTTPRequestHandler.do_GET

    def flaky_get(self):
        if self.path.startswith("/abort"):
            raise ConnectionAbortedError(10053, "An established connection was aborted by the software in your host")
        if self.path.startswith("/reset"):
            raise ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
        return real_get(self)

    monkeypatch.setattr(SimpleHTTPRequestHandler, "do_GET", flaky_get)
    with serve(dist, output) as base:
        for path in ("/abort", "/reset"):
            with pytest.raises((http.client.RemoteDisconnected, ConnectionError, urllib.error.URLError)):
                urllib.request.urlopen(base + path, timeout=10)
        with urllib.request.urlopen(base + "/mech.html", timeout=10) as r:
            assert r.status == 200
    assert "Traceback" not in capsys.readouterr().err

    server = _Server(("127.0.0.1", 0), SimpleHTTPRequestHandler)  # the server-level net, too
    try:
        for exc in (ConnectionAbortedError(10053, "aborted"), BrokenPipeError(32, "broken pipe")):
            try:
                raise exc
            except OSError:
                server.handle_error(None, ("127.0.0.1", 1))
        assert capsys.readouterr().err == ""
        try:
            raise ValueError("a real bug")
        except ValueError:
            server.handle_error(None, ("127.0.0.1", 1))
        assert "ValueError: a real bug" in capsys.readouterr().err  # real errors still surface
    finally:
        server.server_close()


class _PlaywrightError(Exception):
    pass


class _Page:
    """A playwright page stand-in: ``goto`` fails the first ``fail`` times, then the viewer reports
    ``error`` (None = ready)."""

    def __init__(self, fail: int = 0, error: str | None = None):
        self.fail, self.error, self.gotos, self.listeners = fail, error, 0, 0

    def on(self, event, handler):
        self.listeners += 1

    def remove_listener(self, event, handler):
        self.listeners -= 1

    def goto(self, url, **kw):
        self.gotos += 1
        if self.gotos <= self.fail:
            raise _PlaywrightError("net::ERR_CONNECTION_ABORTED at http://127.0.0.1/mech.html")

    def wait_for_function(self, expr, **kw):
        pass

    def evaluate(self, expr):
        return self.error


def test_shot_page_load_is_retried_once():
    from mech.shot import _load_viewer

    page = _Page(fail=1)
    _load_viewer(page, "http://x/mech.html", 1.0, _PlaywrightError)  # second attempt succeeds
    assert page.gotos == 2 and page.listeners == 0
    page = _Page(fail=5)
    with pytest.raises(ShotError, match="playwright failed: net::ERR_CONNECTION_ABORTED"):
        _load_viewer(page, "http://x/mech.html", 1.0, _PlaywrightError)
    assert page.gotos == 2  # once, not forever
    page = _Page(error="mesh parts/arm.stl: Failed to fetch")
    with pytest.raises(ShotError, match="Failed to fetch"):
        _load_viewer(page, "http://x/mech.html", 1.0, _PlaywrightError)
    assert page.gotos == 2  # a failed data fetch is retried
    page = _Page(error="bad view=diagonal: expected iso|top|front|right|left|back|bottom")
    with pytest.raises(ShotError, match="bad view=diagonal"):
        _load_viewer(page, "http://x/mech.html", 1.0, _PlaywrightError)
    assert page.gotos == 1  # a bad parameter: retrying cannot help


def _viewer_ready() -> bool:
    """playwright importable and previewer/dist current (so the test never runs `npm run build`)."""
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return dist_is_current()


@pytest.mark.skipif(not _viewer_ready(), reason="needs playwright and an up-to-date previewer/dist")
def test_shot_takes_a_screenshot(capsys, tmp_path):
    _run(capsys, "run", ARM, "-p", "bump=True", "--output-dir", tmp_path)
    code, out, err = _run(capsys, "shot", "arm", "--output-dir", tmp_path)
    if code != 0 and "playwright failed" in err:
        pytest.skip(f"browser unavailable: {err.strip()}")
    assert code == 0, err
    png = Path(out.strip())
    assert png == (tmp_path / "arm.mech" / "shot_issue0.png").resolve()
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000


@pytest.mark.skipif(not _viewer_ready(), reason="needs playwright and an up-to-date previewer/dist")
def test_shot_with_options_names_the_file_by_them(capsys, tmp_path):
    _run(capsys, "run", TWO, "--output-dir", tmp_path)
    code, out, err = _run(capsys, "shot", "two", "--study", "back", "--frame", "9", "--view", "top", "--zoom", "2",
                          "--axes", "--output-dir", tmp_path)
    if code != 0 and "playwright failed" in err:
        pytest.skip(f"browser unavailable: {err.strip()}")
    assert code == 0, err
    png = Path(out.strip())
    assert png == (tmp_path / "two.mech" / "shot_back_f9_top_z2_axes.png").resolve()
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000
    assert "Traceback" not in err


def test_export_failure_is_one_line_without_traceback(capsys, tmp_path, monkeypatch):
    import mech.export as ex

    monkeypatch.setattr(ex, "export_stl", lambda shape, path, **kw: False)
    code, out, err = _run(capsys, "run", ARM, "--output-dir", tmp_path)
    assert code == 3 and out == ""
    assert err.startswith("mech: export failed: could not write the STL of part '") and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_verbose_run_prints_progress_on_stderr(capsys, tmp_path):
    """--verbose: the setup, one line per study (fed per frame; where its time went) and the home
    pose on stderr; `mech check --verbose` one line per step. Quiet without a terminal otherwise."""
    code, out, err = _run(capsys, "run", TWO, "--verbose", "--no-export", "--output-dir", tmp_path)
    lines = err.strip().splitlines()
    assert lines[0].startswith("setup (clearance pairs, mass) … ") and lines[0].endswith(" s")
    assert lines[-1].startswith("home pose clearance … ") and lines[-1].endswith(" s")
    studies = lines[1:-1]
    assert [line.split(":")[0] for line in studies] == ["study sweep", "study back"]
    for line in studies:
        assert " 19 frames … " in line
        assert re.search(r" \d+\.\d s \(solve \d+\.\d s, clearance \d+\.\d s, loads \d+\.\d s\)$", line), line
    code, out, err = _run(capsys, "run", TWO, "--no-export", "--output-dir", tmp_path)
    assert err == ""  # not a terminal, not --verbose: quiet
    code, out, err = _run(capsys, "check", TWO, "--verbose")
    assert [line.split(" … ")[0] for line in err.strip().splitlines()] == [
        "check two: validating", "check two: clearance setup", "check two: home pose clearance"]
    _, _, err = _run(capsys, "check", TWO)
    assert err == ""


class _Stream:
    """A stderr stand-in that records writes and says whether it is a terminal."""

    def __init__(self, tty: bool):
        self.tty, self.text = tty, ""

    def isatty(self) -> bool:
        return self.tty

    def write(self, s: str) -> int:
        self.text += s
        return len(s)

    def flush(self) -> None:
        pass


def _clock():
    t = iter(range(100))
    return lambda: float(next(t))


def test_study_progress_ticks_per_frame_and_splits_the_time():
    """Fed progress(stage, done, total) once per frame: off a terminal one dot per 10 % of each
    stage, on a terminal a live counter; the finished line says where the time went."""
    from mech.runner import StudyProgress

    stream = _Stream(tty=False)
    meter = StudyProgress("walk", 4, stream=stream, clock=_clock())  # clock: 0, 1, 2, … s per call
    meter.begin("solve")
    for k in range(1, 5):
        meter("solve", k, 4)
    meter.begin("clearance")
    for k in range(1, 5):
        meter("clearance", k, 4)
    meter.begin("loads")
    meter.finish(9.25)
    assert stream.text == ("study walk: 4 frames … solve ·········· clearance ·········· 9.2 s "
                           "(solve 1.0 s, clearance 1.0 s, loads 1.0 s)\n")
    tty = _Stream(tty=True)
    meter = StudyProgress("walk", 4, stream=tty, clock=_clock())
    meter("solve", 1, 4)
    meter("clearance", 12, 4)  # clamped
    meter.finish(3.0)
    assert tty.text.split("\r")[1:] == ["study walk: 4 frames … solve 1/4",
                                         "study walk: 4 frames … clearance 4/4",
                                         "study walk: 4 frames … 3.0 s (solve 1.0 s, clearance 1.0 s)\n"]


def test_step_status_on_a_terminal_is_cleared_for_check():
    from mech.runner import StepStatus

    tty = _Stream(tty=True)
    steps = StepStatus("check x: ", stream=tty, keep=False, clock=_clock())
    with steps.step("validating"):
        pass
    with steps.step("home"):
        pass
    steps.close()
    first, second, blank, end = tty.text.split("\r")[1:]
    assert (first, second.rstrip(), end) == ("check x: validating …", "check x: home …", "")
    assert len(second) == len(first)  # padded over the longer line before it
    assert blank == " " * len("check x: home …")  # close() blanks the status line
    quiet = _Stream(tty=False)
    with StepStatus(stream=quiet, show=False).step("setup"):
        pass
    assert quiet.text == ""


def test_call_with_progress_tolerates_stages_without_the_parameter():
    from mech.runner import call_with_progress

    seen = []

    def old(x):
        return x + 1

    def new(x, progress=None):
        progress("solve", 1, 1)
        return x + 2

    def kw(x, **options):
        options["progress"]("solve", 1, 1)
        return x + 3

    tick = lambda *a: seen.append(a)  # noqa: E731
    assert call_with_progress(old, 1, progress=tick) == 2 and seen == []
    assert call_with_progress(new, 1, progress=tick) == 3 and call_with_progress(kw, 1, progress=tick) == 4
    assert seen == [("solve", 1, 1)] * 2
    assert call_with_progress(old, 1, progress=None) == 2  # no callback: a plain call


def test_sweep_ranks_a_missing_safety_factor_below_a_measured_one():
    from mech.sweep import Variant, best_variant

    rows = [Variant({"k": 1}, "PASS", min_clearance=1.0, worst_sf=None),
            Variant({"k": 2}, "PASS", min_clearance=1.0, worst_sf=1.6)]
    assert best_variant(rows).params == {"k": 2}


def test_shot_help_says_what_isolate_and_q_do(capsys):
    """--isolate keeps the other parts as faint context (it never hides them: --hide does), and
    --q picks the frame, so it does not go with --frame (rejected before a browser starts)."""
    with pytest.raises(SystemExit):
        main(["shot", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "show only these parts" not in text
    assert "the others are drawn faint" in text and "--hide removes parts" in text
    assert "not with --frame" in text
    code, _, err = _run(capsys, "shot", "nothing_here", "--q", "j:40", "--frame", "3")
    assert code == 3 and "both pick the frame" in err
