"""cli.py: exit codes, -p parsing, trimmed tracebacks, run/check/sweep/list/shot plumbing."""

from __future__ import annotations

import json
import os
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
    code, out, _ = _run(capsys, "run", ARM, "--no-export", "--frames", "4", "--output-dir", tmp_path)
    assert code == 0 and not (tmp_path / "arm.mech").exists()
    assert out.splitlines()[-1] == "view: not exported (drop --no-export to write output/arm.mech)"
    code, out, _ = _run(capsys, "run", ARM, "--study", "swnig", "--output-dir", tmp_path)
    assert code == 3 and "study 'swnig' (requested): no such study (did you mean 'swing'?)" in out


def test_delta_prev_across_runs(capsys, tmp_path):
    _run(capsys, "run", ARM, "--output-dir", tmp_path)
    code, out, _ = _run(capsys, "run", ARM, "-p", "bump=True", "--output-dir", tmp_path)
    assert code == 2
    dprev = next(line for line in out.splitlines() if line.startswith("Δprev"))
    assert dprev.startswith("Δprev: status PASS→FAIL · new interference arm/block")
    _, out, _ = _run(capsys, "run", ARM, "--output-dir", tmp_path)
    assert "Δprev: status FAIL→PASS · fixed interference arm/block" in out


def test_check(capsys, tmp_path):
    code, out, _ = _run(capsys, "check", ARM, "-p", "bump=True", "--output-dir", tmp_path)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("mech check arm — PASS   3 parts · 2 joints (1 driver, 1 free)")
    assert "roles j_arm driver · j_block free" in lines and "studies swing mobility 0" in lines
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
    assert lines[1].split() == ["swing", "need", "status", "F/W", "min", "clr", "worst", "SF", "swing"]
    rows = [line.split() for line in lines[2:8]]
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [
        ("30", "45", "FAIL", "1/0"), ("30", "60", "FAIL", "1/0"), ("60", "45", "PASS", "0/0"),
        ("60", "60", "PASS", "0/0"), ("90", "45", "PASS", "0/0"), ("90", "60", "PASS", "0/0")]
    assert [r[-1] for r in rows] == ["30.0", "30.0", "60.0", "60.0", "90.0", "90.0"]  # target value column
    assert lines[8] == "best swing=60 need=45 — PASS (--export-best to export it)"
    assert not (tmp_path / "arm.mech").exists()

    code, out, _ = _run(capsys, "sweep", ARM, "need=100,120", "--export-best", "--frames", "4",
                        "--output-dir", tmp_path)
    assert code == 2 and "best need=100 — FAIL · exported" in out
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
