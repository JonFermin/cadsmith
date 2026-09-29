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
    assert sorted(p.name for p in out.iterdir()) == ["parts", "report.json", "scene.json"]
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
