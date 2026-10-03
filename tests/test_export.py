"""export.py: output layout, binary STL validity, scene.json schema, JSON safety, slugs."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Pos

from mech import Assembly
from mech.export import mech_dir, read_report, write_json
from mech.runner import analyze


def _strict_load(path: Path):
    """json.load that rejects NaN / Infinity literals."""
    def bad(token):
        raise ValueError(f"{path.name} contains {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


def _stl_triangles(path: Path) -> np.ndarray:
    """(n, 3, 3) vertices of a binary STL, after checking its size is exactly 84 + 50·n."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    assert n > 0 and len(data) == 84 + 50 * n, path.name
    rows = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
                         count=n, offset=84)
    return rows["v"].astype(float)


def test_four_bar_export_layout(tmp_path, four_bar):
    report = analyze(four_bar, out_root=tmp_path)
    out = tmp_path / "four_bar.mech"
    assert mech_dir(tmp_path, "four_bar") == out
    assert sorted(p.name for p in out.iterdir()) == ["parts", "report.json", "scene.json", "series.json"]
    assert sorted(p.name for p in (out / "parts").iterdir()) == ["coupler.stl", "crank.stl", "frame.stl",
                                                                 "rocker.stl"]
    scene = _strict_load(out / "scene.json")
    saved = _strict_load(out / "report.json")
    assert saved == report == scene["report"] and read_report(tmp_path, "four_bar") == report

    assert (scene["version"], scene["name"], scene["units"], scene["up"], scene["clearance"]) == (
        1, "four_bar", "mm", [0, 0, 1], 0.3)
    parts = {p["id"]: p for p in scene["parts"]}
    assert list(parts) == ["frame", "crank", "coupler", "rocker"]
    for name, p in parts.items():
        assert set(p) == {"id", "color", "opacity", "mesh", "ground", "mass_g", "material", "bom", "bbox"}
        assert p["mesh"] == f"parts/{name}.stl"
        tri = _stl_triangles(out / p["mesh"])  # world mm, home pose: inside the part's bbox
        lo, hi = np.array(p["bbox"][0]), np.array(p["bbox"][1])
        assert np.all(tri.reshape(-1, 3) >= lo - 1e-3) and np.all(tri.reshape(-1, 3) <= hi + 1e-3)
        assert p["mass_g"] == pytest.approx(report["mass"]["parts"][name])
    assert parts["frame"]["ground"] is True and parts["crank"]["material"] == "aluminum_6061"
    assert parts["frame"]["bbox"] == [[-10.0, -8.0, -12.5], [110.0, 8.0, -7.5]]

    joints = {j["name"]: j for j in scene["joints"]}
    assert joints["j_crank"] == {"name": "j_crank", "kind": "revolute", "parent": "frame", "child": "crank",
                                 "origin": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "limits": None, "home": 0.0,
                                 "role": "driver"}
    assert joints["j_rocker"]["role"] == "passive"
    assert scene["pins"][0]["name"] == "p_B" and scene["pins"][0]["axis"] == [0.0, 0.0, 1.0]
    assert scene["probes"] == [{"name": "mid", "part": "coupler"}]

    (study,) = scene["studies"]
    n = 72
    assert (study["name"], study["frames"], study["loop"], study["duration"]) == ("turn", n, "once", 3.0)
    assert len(study["t"]) == n and study["t"][0] == 0.0 and study["t"][-1] == 3.0
    assert study["joints"]["j_crank"][0] == 0.0 and study["joints"]["j_crank"][-1] == 360.0
    assert set(study["joints"]) == {"j_crank", "j_coupler", "j_rocker"}
    assert set(study["transforms"]) == {"crank", "coupler", "rocker"}  # the ground frame is omitted (T = I)
    # crank transform at frame 18 of 0…360° in 72 frames: rotation about +Z through the origin
    T = np.array(study["transforms"]["crank"][18]).reshape(4, 4).T  # column-major -> row-major
    th = math.radians(360 * 18 / 71)
    Rz = [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    np.testing.assert_allclose(T[:3, :3], Rz, atol=1e-5)  # 6 significant figures in JSON
    np.testing.assert_allclose(T[:3, 3], 0.0, atol=1e-9)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    assert len(study["probes"]["mid"]) == n and study["issues"] == []
    assert len(study["loads"]["j_crank"]) == n and len(study["residual"]) == n


def test_json_never_contains_nan_and_open_loops_export(tmp_path):
    from conftest import make_four_bar

    asm = make_four_bar(coupler=45.0)  # loop can't close over part of the turn
    report = analyze(asm, out_root=tmp_path)
    assert report["status"] == "FAIL"
    scene = _strict_load(tmp_path / "four_bar.mech" / "scene.json")
    assert any(r > 1.0 for r in scene["studies"][0]["residual"] if r is not None)
    with pytest.raises(ValueError):
        write_json(tmp_path / "x.json", {"bad": math.nan})


def test_slugged_names_stale_files_and_step(tmp_path):
    asm = Assembly("Grip: Ünïcode/Test", clearance=0.3)
    asm.part("CON", Pos(0, 0, -5) * Box(40, 40, 4), ground=True)
    asm.part("con", Pos(0, 0, 10) * Box(10, 10, 4))
    asm.part("a/b:c", Pos(0, 0, 30) * Box(10, 10, 4))
    asm.prismatic("j1", "CON", "con", origin=(0, 0, 0), axis=(0, 0, 1), limits=(0, 5))
    asm.fix("a/b:c", "con")
    out = tmp_path / "grip_unicode_test.mech"
    (out / "parts").mkdir(parents=True)
    (out / "parts" / "old_part.stl").write_bytes(b"stale")
    analyze(asm, out_root=tmp_path, step=True)
    assert sorted(p.name for p in (out / "parts").iterdir()) == ["a_b_c.stl", "con_.stl", "con__2.stl"]
    scene = _strict_load(out / "scene.json")
    assert [p["mesh"] for p in scene["parts"]] == ["parts/con_.stl", "parts/con__2.stl", "parts/a_b_c.stl"]
    for p in scene["parts"]:
        _stl_triangles(out / p["mesh"])
    step = (out / "assembly.step").read_text(encoding="utf-8", errors="replace")
    assert "a/b:c" in step and "CON" in step
    assert scene["studies"][0]["transforms"].keys() == {"con", "a/b:c"}  # the fixed child moves with its parent
    analyze(asm, out_root=tmp_path)  # a later run without step drops the now-stale STEP
    assert not (out / "assembly.step").exists()


def test_partial_runs_write_only_report_partial_json(tmp_path, four_bar):
    """scene.json/report.json always hold the last full run: a --study/--frames run writes
    report.partial.json only; the next full run replaces both and drops the partial report."""
    from conftest import make_four_bar

    analyze(four_bar, out_root=tmp_path, step=True)
    out = tmp_path / "four_bar.mech"
    before = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    partial = analyze(make_four_bar(), studies=["turn"], frames=12, out_root=tmp_path)
    assert partial["partial"] == {"studies": ["turn"], "skipped": [], "frames": 12}
    assert partial["viewer_url"] is None
    after = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file() and p.name != "report.partial.json"}
    assert after == before  # STLs, STEP, scene.json and report.json untouched
    assert _strict_load(out / "report.partial.json") == partial
    assert read_report(tmp_path, "four_bar", partial=True) == partial
    full = analyze(make_four_bar(), out_root=tmp_path)
    assert full["partial"] is None and not (out / "report.partial.json").exists()
    assert full["delta_prev"] == {}  # compared with the previous full run, not the partial one


def test_invalid_full_run_clears_the_scene(tmp_path, four_bar):
    from mech.export import clear_scene

    analyze(four_bar, out_root=tmp_path, step=True)
    out = tmp_path / "four_bar.mech"
    four_bar.revolute("j_bad", "fram", "crank", origin=(0, 0, 0), axis=(0, 0, 1))  # INVALID: unknown parent
    report = analyze(four_bar, out_root=tmp_path)
    assert report["status"] == "INVALID"
    assert sorted(p.name for p in out.iterdir()) == ["parts", "report.json"]
    assert list((out / "parts").iterdir()) == [] and _strict_load(out / "report.json") == report
    clear_scene(tmp_path, "nothing_there")  # a mech that was never exported: no error


def test_stl_failure_is_an_export_error_without_leftovers(tmp_path, four_bar, monkeypatch):
    """export_stl raising (or reporting failure) names the part and leaves no *.tmp.stl behind."""
    import mech.export as ex

    def broken(shape, path, **kw):
        Path(path).write_bytes(b"half a file")
        raise RuntimeError("tessellation failed")

    monkeypatch.setattr(ex, "export_stl", broken)
    with pytest.raises(ex.ExportError, match=r"could not write the STL of part 'frame' \(RuntimeError: tessellation"):
        analyze(four_bar, out_root=tmp_path)
    monkeypatch.setattr(ex, "export_stl", lambda shape, path, **kw: False)
    with pytest.raises(ex.ExportError, match=r"part 'frame' \(export_stl failed\)"):
        analyze(four_bar, out_root=tmp_path)
    assert list((mech_dir(tmp_path, "four_bar") / "parts").glob("*.stl")) == []


# ------------------------------------------------------------------------------ series.json, CoG, virtual parts


def test_series_json_is_the_full_precision_per_frame_record(tmp_path, four_bar):
    """series.json holds, per study and frame, t / ok / every moving joint / probe points / loads /
    residual at full float precision (scene.json rounds to 6 significant figures)."""
    from mech import Kinematics
    from mech.geom import fnum
    from mech.motion import run_study
    from mech.statics import gravity_loads
    from mech.massprops import part_props

    analyze(four_bar, out_root=tmp_path)
    series = _strict_load(tmp_path / "four_bar.mech" / "series.json")
    assert (series["version"], series["name"], series["units"], series["angles"]) == (1, "four_bar", "mm", "deg")
    (st,) = series["studies"]
    assert (st["name"], st["frames"], st["loop"]) == ("turn", 72, "once")
    assert set(st) == {"name", "frames", "loop", "t", "ok", "joints", "probes", "loads", "residual"}

    kin = Kinematics(four_bar)
    res = run_study(four_bar, kin, four_bar.studies[0])
    loads = gravity_loads(four_bar, kin, res, {n: part_props(p) for n, p in four_bar.parts.items()})
    assert st["t"] == [float(t) for t in res.t] and st["ok"] == [True] * 72
    assert set(st["joints"]) == {"j_crank", "j_coupler", "j_rocker"}
    rocker = [p.q["j_rocker"] for p in res.poses]
    assert st["joints"]["j_rocker"] == pytest.approx(rocker, rel=1e-12, abs=1e-12)
    assert any(v != fnum(v) for v in st["joints"]["j_rocker"])  # more than 6 significant figures
    assert st["probes"]["mid"][17] == pytest.approx(list(res.probes["mid"][17]), rel=1e-12)
    assert len(st["probes"]["mid"]) == 72 and all(len(p) == 3 for p in st["probes"]["mid"])
    crank = st["loads"]["j_crank"]
    assert (crank["unit"], crank["reflected_from"]) == ("N·m", None)
    assert crank["series"] == pytest.approx(loads["j_crank"]["series"], rel=1e-12, abs=1e-15)
    assert len(st["residual"]) == 72 and max(st["residual"]) < 1e-9


def test_series_json_open_frames_are_null_not_nan(tmp_path):
    from conftest import make_four_bar

    analyze(make_four_bar(coupler=45.0), out_root=tmp_path)  # the loop opens over part of the turn
    st = _strict_load(tmp_path / "four_bar.mech" / "series.json")["studies"][0]
    assert False in st["ok"] and True in st["ok"]
    open_k = st["ok"].index(False)
    assert st["loads"]["j_crank"]["series"][open_k] is None  # undefined at an open frame
    assert st["residual"][open_k] > 1e-3


def test_report_has_the_cog_of_every_part(tmp_path):
    """mass.parts_com_mm: each part's centre of mass at home (world mm), next to mass.parts (g)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_export_two", Path(__file__).parent / "fixtures" / "two_studies.py")
    two = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(two)
    report = analyze(two.build(), export=False, out_root=tmp_path)
    com = report["mass"]["parts_com_mm"]
    assert set(com) == set(report["mass"]["parts"]) == {"base", "post", "arm"}
    c = 30.0 / math.sqrt(2)  # report numbers carry 6 significant figures
    np.testing.assert_allclose(com["arm"], [20.0, 0.0, 10.0], rtol=1e-5, atol=1e-6)  # Box(40, 4, 4) at (20, 0, 10)
    np.testing.assert_allclose(com["post"], [c, c, 10.0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(com["base"], [0.0, 0.0, -2.0], rtol=1e-5, atol=1e-6)
    # the assembly CoG is the mass-weighted mean of the parts'
    m = report["mass"]["parts"]
    total = sum(m.values())
    mean = sum(np.asarray(com[n]) * m[n] for n in m) / total
    np.testing.assert_allclose(report["mass"]["com_mm"], mean, rtol=1e-5, atol=1e-5)


def test_virtual_parts_are_not_exported_counted_or_weighed(tmp_path):
    """A ``virtual`` part (the internal knuckle of a spherical joint) has no STL, no scene entry, no
    mass, and does not count as a part or a checked pair."""
    asm = Assembly("knuckled", clearance=0.3)
    asm.part("base", Pos(0, 0, -3) * Box(80, 80, 4), ground=True)
    asm.part("arm", Pos(20, 0, 2) * Box(40, 6, 4))
    asm.part("knuckle", Pos(0, 0, 30) * Box(1, 1, 1), material="steel")
    asm.revolute("j_arm", "base", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.fix("knuckle", "arm")
    asm.study("swing", drive={"j_arm": (0, 90)}, frames=4)
    plain = analyze(asm, export=False, out_root=tmp_path)
    asm.parts["knuckle"].virtual = True
    report = analyze(asm, out_root=tmp_path)
    out = tmp_path / "knuckled.mech"
    assert sorted(p.name for p in (out / "parts").iterdir()) == ["arm.stl", "base.stl"]
    scene = _strict_load(out / "scene.json")
    assert [p["id"] for p in scene["parts"]] == ["base", "arm"]
    assert set(scene["studies"][0]["transforms"]) == {"arm"}
    assert report["parts"] == 2 and plain["parts"] == 3
    assert set(report["mass"]["parts"]) == {"base", "arm"}
    knuckle = plain["mass"]["parts"]["knuckle"]  # 1 mm³ of steel
    assert knuckle > 0 and report["mass"]["total_g"] == pytest.approx(plain["mass"]["total_g"] - knuckle, rel=1e-5)
    assert report["clearance"]["pairs_checked"] == plain["clearance"]["pairs_checked"] - 1  # base/knuckle


def test_ball_joints_are_listed_for_the_viewer(tmp_path):
    """scene.json lists each ball() (name, parent, child, home center, its three revolutes): the
    revolutes chain through virtual knuckles that have no mesh or transforms, so the viewer draws
    the ball at its center (carried by the parent) instead of those joints' axes."""
    asm = Assembly("balled", clearance=0.3)
    asm.part("post", Pos(0, 0, -10) * Box(10, 10, 20), ground=True)
    asm.part("arm", Pos(0, 0, 28) * Box(6, 6, 50))  # 3 mm above the ball center on the post top
    assert asm.ball("s", "post", "arm", (0, 0, 0)) == "s"
    asm.study("tilt", drive={"s_1": (0, 30)}, frames=4)
    report = analyze(asm, out_root=tmp_path)
    assert report["status"] == "PASS", report["issues"]
    scene = _strict_load(tmp_path / "balled.mech" / "scene.json")
    assert scene["balls"] == [{"name": "s", "parent": "post", "child": "arm", "center": [0, 0, 0],
                               "joints": ["s_1", "s_2", "s_3"]}]
    assert {j["name"] for j in scene["joints"]} == {"s_1", "s_2", "s_3"}
    assert [p["id"] for p in scene["parts"]] == ["post", "arm"]          # no knuckle meshes …
    assert set(scene["studies"][0]["transforms"]) == {"arm"}             # … nor transforms
