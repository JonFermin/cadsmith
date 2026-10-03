"""Headless regression tests for the mech viewer (previewer/mech.html, MECH_SPEC §7).

A synthetic scene is written straight to a temporary output dir (no `mech` run needed): a ground
plate + post, a 300 mm arm swinging a full turn (so its motion envelope is far larger than the
parts at any one pose), a pin riding on it, a block it passes through at frame 3 (a report
interference issue) and two probes — one near the hub, one sweeping a 300 mm circle. The viewer
is served from previewer/dist (rebuilt when stale) with ``mech.shot.serve`` and driven by
playwright; every check reads the live three.js state through ``window.__mech``.

Run: ``uv run pytest -q -W error previewer/tests``
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from urllib.parse import urlencode

import pytest

FILL_PERSPECTIVE = 0.72  # viewer.js: fraction of the pane the fitted points span (limiting direction)
FILL_ORTHO = 0.78
SLUG = "viewer_t"
FRAMES = 13  # 0..360° in 30° steps


# ------------------------------------------------------------------------------ synthetic scene


def _box_stl(path: Path, lo, hi) -> None:
    """Binary STL of an axis-aligned box (12 outward-wound triangles). The facet normals are
    left zero, as many STL writers do — the viewer must rebuild them from the winding."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    with path.open("wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for i in t:
                f.write(struct.pack("<3f", *v[i]))
            f.write(b"\0\0")


def _rot_z(deg: float) -> list[float]:
    """Column-major 4×4 rotation about +Z (three.js Matrix4.fromArray layout)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _turn(p, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]


PARTS = {  # id: (lo, hi, color, ground)
    "base": ((-100, -100, -10), (100, 100, 0), "#6b7280", True),
    "post": ((-5, -5, 0), (5, 5, 60), "#9aa4b2", True),
    "arm": ((0, -5, 60), (300, 5, 70), "#e8772e", False),
    "pin": ((280, -8, 70), (290, 8, 80), "#c0392b", False),
    "block": ((-10, 140, 55), (10, 160, 75), "#3d5a80", True),
    "dark": ((-60, -60, 0), (-30, -30, 30), "#141414", True),
}


# a ball() joint arm → pin: three revolutes through virtual knuckles b_k1, b_k2 that the scene does
# not carry (no part entry, no transforms) — the viewer draws the ball, never their stale axes
BALL_CENTER = (285.0, 0.0, 70.0)
BALL_JOINTS = [{"name": f"b_{i}", "kind": "revolute", "parent": a, "child": c, "origin": list(BALL_CENTER),
                "axis": ax, "limits": None, "home": 0.0, "role": "passive"}
               for i, (a, c, ax) in enumerate((("arm", "b_k1", [1, 0, 0]), ("b_k1", "b_k2", [0, 1, 0]),
                                               ("b_k2", "pin", [0, 0, 1])), start=1)]


def write_scene(out: Path) -> Path:
    mech = out / f"{SLUG}.mech"
    (mech / "parts").mkdir(parents=True)
    parts = []
    for pid, (lo, hi, color, ground) in PARTS.items():
        _box_stl(mech / "parts" / f"{pid}.stl", lo, hi)
        parts.append({"id": pid, "color": color, "opacity": 1, "mesh": f"parts/{pid}.stl", "ground": ground,
                      "mass_g": 10.0, "material": "pla", "bom": None, "bbox": [list(lo), list(hi)]})
    angles = [30.0 * f for f in range(FRAMES)]
    study = {
        "name": "swing", "frames": FRAMES, "duration": 2.0, "loop": "once",
        "t": [2.0 * f / (FRAMES - 1) for f in range(FRAMES)],
        "joints": {"j_arm": angles, "b_1": [0.0] * FRAMES, "b_2": [0.0] * FRAMES, "b_3": [0.0] * FRAMES},
        "transforms": {pid: [_rot_z(a) for a in angles] for pid in ("arm", "pin")},
        "probes": {"tip": [_turn((300, 0, 65), a) for a in angles], "near": [_turn((20, 0, 65), a) for a in angles]},
        "issues": [{"frame": 3, "a": "arm", "b": "block", "status": "interference", "distance": 0.0, "volume": 2000.0,
                    "pa": [0, 150, 65], "pb": [0, 150, 65]}],
        "loads": {"j_arm": [0.1] * FRAMES}, "residual": [0.0] * FRAMES,
    }
    report = {
        "name": SLUG, "status": "FAIL", "targets": [], "mass": {"total_g": 60.0},
        "issues": [{"code": "interference", "severity": "FAIL", "message": "arm and block overlap 2000 mm³",
                    "parts": ["arm", "block"], "study": "swing", "frame": 3, "location": [150, 0, 65],
                    "extent": [20, 10, 10]}],
        "studies": [{"name": "swing", "loads": {"j_arm": {"unit": "N·m", "capacity": None}}}],
    }
    scene = {
        "version": 1, "name": SLUG, "units": "mm", "up": [0, 0, 1], "clearance": 0.3, "parts": parts,
        "joints": [{"name": "j_arm", "kind": "revolute", "parent": "post", "child": "arm", "origin": [0, 0, 0],
                    "axis": [0, 0, 1], "limits": None, "home": 0.0, "role": "driver"}] + BALL_JOINTS,
        "pins": [], "probes": [{"name": "tip", "part": "arm"}, {"name": "near", "part": "arm"}],
        "balls": [{"name": "b", "parent": "arm", "child": "pin", "center": list(BALL_CENTER),
                   "joints": [j["name"] for j in BALL_JOINTS]}],
        "studies": [study], "report": report,
    }
    (mech / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (mech / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return mech


# ------------------------------------------------------------------------------ browser fixture

# In-page helpers: NDC extents of world points through a pane's camera, hull points of parts.
HELPERS = """
window.__t = {
  app: () => window.__mech,
  ndc(points, pane) {
    const v = window.__mech.viewer;
    const cam = v.panes()[pane].cam;
    cam.updateMatrixWorld();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of points) {
      const q = p.clone().project(cam);
      x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
    }
    return { x0, x1, y0, y1, half: Math.max((x1 - x0) / 2, (y1 - y0) / 2), n: points.length };
  },
  hull(ids) { return window.__mech.parts.hullPoints(ids); },
  probe(name) {
    const a = window.__mech;
    const pr = a.model.probes(a.state.si).find(p => p.name === name);
    return pr.pts.map(p => a.parts.items.get('arm').hull[0].clone().set(p[0], p[1], p[2]));
  },
};
"""


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("needs playwright")
    from mech.shot import ShotError, ensure_dist, serve

    try:
        dist = ensure_dist()
    except ShotError as exc:
        pytest.skip(f"viewer build unavailable: {exc}")
    out = tmp_path_factory.mktemp("output")
    write_scene(out)
    with serve(dist, out) as base, sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001 - browser binaries not installed
            pytest.skip(f"browser unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        def load(**params):
            """Open mech.html with these params; returns window.__mechError (None when ready)."""
            errors.clear()
            query = urlencode({"m": SLUG, **{k: str(v) for k, v in params.items()}, "ui": "0"}, safe=",:")
            page.goto(f"{base}/mech.html?{query}")
            page.wait_for_function("window.__mechReady === true || window.__mechError !== undefined", timeout=90_000)
            err = page.evaluate("window.__mechError ?? null")
            if err is None:
                page.evaluate(HELPERS)
                assert errors == [], errors
            return err

        page.load = load
        yield page
        browser.close()


def js(page, expr):
    return page.evaluate(expr)


# ------------------------------------------------------------------------------ D2: framing


def test_default_fit_frames_visible_parts_at_current_pose_not_the_envelope(viewer):
    assert viewer.load() is None
    m = js(viewer, "__t.ndc(__t.hull(null), 0)")
    # the visible parts span FILL of the pane in their limiting direction, centred, all on screen
    assert m["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.02)
    assert -1 < m["x0"] and m["x1"] < 1 and -1 < m["y0"] and m["y1"] < 1
    assert abs(m["x0"] + m["x1"]) < 0.02 or abs(m["y0"] + m["y1"]) < 0.02
    # the full-turn motion envelope (a 600 mm disc) would not fit: the fit is not the envelope's
    env = js(viewer, """(() => { const e = __t.app().parts.envelope(0); const pts = [];
        for (const x of [e.min.x, e.max.x]) for (const y of [e.min.y, e.max.y]) for (const z of [e.min.z, e.max.z])
          pts.push(e.min.clone().set(x, y, z));
        return __t.ndc(pts, 0); })()""")
    assert env["half"] > 1.2
    # the 300 mm tip circle is cropped by default (only paths near the parts are framed)…
    tip = js(viewer, "__t.ndc(__t.probe('tip'), 0)")
    assert tip["half"] > 1.0
    # …and the default grid is sized to the model (home pose), not the motion envelope
    grid = js(viewer, "__t.app().viewer.floor.scale.x")
    assert grid <= 2.6 * 400  # home extent: x −100…300


def test_ghost_poses_are_framed_only_when_requested(viewer):
    assert viewer.load(ghost=4) is None
    pts = "__t.app().parts.ghostPoints(0, __t.app().model.ghostFrames(0, 4), new Set())"
    m = js(viewer, f"__t.ndc([...__t.hull(null), ...{pts}], 0)")
    assert m["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.02)
    assert viewer.load() is None
    assert js(viewer, f"__t.ndc({pts}, 0)")["half"] > 1.0  # without ghost=, the ghost poses are not framed


def test_paths_param_hides_or_frames_probe_paths(viewer):
    assert viewer.load(paths=1) is None
    m = js(viewer, "__t.ndc([...__t.hull(null), ...__t.probe('tip'), ...__t.probe('near')], 0)")
    assert m["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.02)  # paths=1: the whole circle is framed
    assert js(viewer, "__t.app().probePaths.group.children.length") > 0
    assert viewer.load(paths=0) is None
    assert js(viewer, "[__t.app().state.probes, __t.app().probePaths.group.children.length]") == [False, 0]
    assert js(viewer, "__t.ndc(__t.hull(null), 0)")["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.02)


@pytest.mark.parametrize("layout", ["single", "quad"])
def test_focus_fits_the_parts_in_every_pane(viewer, layout):
    assert viewer.load(focus="pin,block", layout=layout) is None
    panes = 4 if layout == "quad" else 1
    for i in range(panes):
        m = js(viewer, f"__t.ndc(__t.hull(['pin', 'block']), {i})")
        fill = FILL_PERSPECTIVE if i == 0 else FILL_ORTHO
        assert m["half"] == pytest.approx(fill, abs=0.03), (i, m)
        assert -1 < m["x0"] and m["x1"] < 1 and -1 < m["y0"] and m["y1"] < 1
    # paths are not framed with a focus
    assert js(viewer, "__t.app().fitPoints().length") == js(viewer, "__t.hull(['pin', 'block']).length")


@pytest.mark.parametrize("layout", ["single", "quad"])
def test_issue_selects_its_pose_and_keeps_focusing_its_pair(viewer, layout):
    assert viewer.load(issue=0, layout=layout) is None
    s = js(viewer, "(() => { const s = __t.app().state; return [s.si, s.frame, [...s.isolate].sort(), s.focus]; })()")
    assert s == [0, 3, ["arm", "block"], ["arm", "block"]]
    for i in range(4 if layout == "quad" else 1):
        m = js(viewer, f"__t.ndc(__t.hull(['arm', 'block']), {i})")
        assert m["half"] == pytest.approx(FILL_PERSPECTIVE if i == 0 else FILL_ORTHO, abs=0.03), (i, m)
    # a view change (and the F key's fit) still frames the pair
    js(viewer, "__t.app().setView('top')")
    assert js(viewer, "__t.ndc(__t.hull(['arm', 'block']), 0)")["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.03)
    js(viewer, "__t.app().fit()")
    assert js(viewer, "__t.ndc(__t.hull(['arm', 'block']), 0)")["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.03)


def test_zoom_factor_moves_every_camera_in(viewer):
    probe = """(() => { const v = __t.app().viewer; const top = v.orthos[0].cam;
        return [v.camera.position.distanceTo(v.controls.target), top.top - top.bottom]; })()"""
    assert viewer.load(layout="quad") is None
    d1, h1 = js(viewer, probe)
    assert viewer.load(layout="quad", zoom=2) is None
    d2, h2 = js(viewer, probe)
    assert d2 == pytest.approx(d1 / 2, rel=1e-6) and h2 == pytest.approx(h1 / 2, rel=1e-3)
    assert viewer.load(zoom=0.5) is None
    assert js(viewer, probe)[0] == pytest.approx(2 * d1, rel=1e-6)


# ------------------------------------------------------------------------------ D3: section


def test_section_overlay_is_sized_to_the_cut_and_subtle(viewer):
    assert viewer.load(section="y:0") is None
    info = js(viewer, """(() => { const a = __t.app(); const g = a.sectionPlane.group.children[0];
        const [plane, outline] = g.children; const p = plane.geometry.parameters;
        return { w: p.width, h: p.height, fill: plane.material.opacity, line: outline.material.opacity,
                 offset: a.state.section.offset }; })()""")
    # the y=0 cut crosses base (x −100…100, z −10…0), post, arm (x 0…300) and pin (z 70…80):
    # x −100…300 by z −10…80 plus a small margin — not the 600 mm motion envelope
    assert 405 <= info["w"] <= 425 and 92 <= info["h"] <= 108, info
    assert info["fill"] <= 0.04 and info["line"] <= 0.25
    # the caps still carry the cut: back faces drawn solid on every non-context part
    assert js(viewer, "[...__t.app().parts.items.values()].every(it => it.cap.visible)") is True


def test_section_default_offset_cuts_every_part_of_interest(viewer):
    # arm (y −5…5) and pin (y −8…8) overlap in y: the cut goes through both (their box centre
    # would too here, but not for parts of very different extent — see the issue case below)
    assert viewer.load(section="y", isolate="arm,pin") is None
    assert -5 <= js(viewer, "__t.app().state.section.offset") <= 5
    assert viewer.load(section="y", focus="pin") is None
    assert js(viewer, "__t.app().state.section.offset") == pytest.approx(0, abs=1e-6)
    # an issue shown at its pose: through its overlap location (arm home (150,0,65) at 90° → y 150)
    assert viewer.load(issue=0, section="y") is None
    assert js(viewer, "__t.app().state.section.offset") == pytest.approx(150, abs=1e-6)


def test_section_fit_frames_only_what_is_left(viewer):
    # x:−20 keeps x ≤ −20: the arm and pin (x ≥ 0 at frame 0) are cut away and not framed
    assert viewer.load(section="x:-20") is None
    xs = js(viewer, "__t.app().fitPoints().map(p => p.x)")
    assert max(xs) <= -20 + 1e-6
    assert js(viewer, "__t.ndc(__t.app().fitPoints(), 0)")["half"] == pytest.approx(FILL_PERSPECTIVE, abs=0.02)
    # probe paths are cut like the parts: clipped lines, no label for a probe point cut away
    # (both probes sit at x > −20 at frame 0)
    probes = js(viewer, """(() => { const pp = __t.app().probePaths;
        return [pp.group.children.map(o => o.material.clippingPlanes.length), pp.labels.length]; })()""")
    assert probes == [[1, 1, 1, 1], 0]
    assert viewer.load(section="-x:-20") is None  # the other side: both labels back
    assert js(viewer, "__t.app().probePaths.labels.length") == 2


# ------------------------------------------------------------------------------ D1: every param


def test_every_url_param_is_applied(viewer):
    assert viewer.load(study="swing", frame=2, cam="30,20", zoom=1.5, section="-x:10", focus="pin",
                       isolate="arm,pin", hide="block", explode=0.2, axes=1, ghost=3, paths=0, layout="quad") is None
    s = js(viewer, """(() => { const s = __t.app().state; return { si: s.si, frame: s.frame, zoom: s.zoom,
        section: s.section, focus: s.focus, isolate: [...s.isolate].sort(), hidden: [...s.hidden], explode: s.explode,
        axes: s.axes, ghost: s.ghost, probes: s.probes, layout: s.layout,
        blockVisible: __t.app().parts.items.get('block').mesh.visible,
        ghosts: __t.app().parts.ghostGroup.children.length, axesDrawn: __t.app().jointAxes.group.children.length }; })()""")
    assert s == {"si": 0, "frame": 2, "zoom": 1.5, "section": {"axis": "x", "offset": 10, "flip": True},
                 "focus": ["pin"], "isolate": ["arm", "pin"], "hidden": ["block"], "explode": 0.2, "axes": True,
                 "ghost": 3, "probes": False, "layout": "quad", "blockVisible": False, "ghosts": 6, "axesDrawn": 5}
    # cam=30,20: the main camera looks from azimuth 30°, elevation 20°
    d = js(viewer, """(() => { const v = __t.app().viewer;
        return v.camera.position.clone().sub(v.controls.target).normalize().toArray(); })()""")
    az, el = math.radians(30), math.radians(20)
    assert d == pytest.approx([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], abs=1e-6)
    # q= picks the nearest frame; view= aims the camera
    assert viewer.load(q="j_arm:95", view="top") is None
    assert js(viewer, "__t.app().state.frame") == 3
    d = js(viewer, "(() => { const v = __t.app().viewer; return v.camera.position.clone().sub(v.controls.target).normalize().toArray(); })()")
    assert d[2] == pytest.approx(1, abs=1e-6)
    assert viewer.load(frame="home") is None
    assert js(viewer, "__t.app().state.frame") is None


def test_ball_joints_are_drawn_at_their_center_not_as_stale_axes(viewer):
    """The revolutes of a ball() chain through virtual knuckles with no transforms: their axes would
    sit at the home pose. The viewer skips them and draws the ball (three rings + its label) at the
    center carried by the ball's parent — here the arm, turned 90° at frame 3."""
    assert viewer.load(frame=3, axes=1) is None
    info = js(viewer, """(() => { const ja = __t.app().jointAxes;
        return { labels: ja.labels.map(l => l.el.textContent), at: ja.labels.map(l => l.pos.toArray()),
                 children: ja.group.children.length }; })()""")
    assert info["labels"] == ["j_arm", "b"], info
    x, y, z = info["at"][1]
    assert (x, y) == pytest.approx((0.0, 285.0), abs=1e-6) and z > 70.0  # (285, 0, 70) turned 90° about Z
    assert info["children"] == 2 + 3  # j_arm's arrow and ring, the ball's three rings


@pytest.mark.parametrize("params, message", [
    ({"zom": 2}, "unknown parameter 'zom' — did you mean zoom?"),
    ({"zoom": 0}, "bad zoom=0: must be > 0"),
    ({"paths": "maybe"}, "bad paths=maybe: expected 0 or 1"),
    ({"layout": "triple"}, "bad layout=triple: expected single|quad"),
    ({"focus": "pinn"}, "unknown part in focus= 'pinn' — did you mean pin?"),
    ({"study": "swng"}, "unknown study 'swng' — did you mean swing?"),
    ({"q": "j_arm:95", "frame": 2}, "q=j_arm:95 and frame=2 both pick the frame"),
])
def test_bad_params_fail_fast_with_a_hint(viewer, params, message):
    err = viewer.load(**params)
    assert err is not None and message in err, err


# ------------------------------------------------------------------------------ hero-shot quality


def test_zero_stl_normals_are_rebuilt_from_the_winding(viewer):
    assert viewer.load() is None
    normals = js(viewer, """[...__t.app().parts.items.values()].map(it => {
        const n = it.geom.attributes.normal; return [n.getX(0), n.getY(0), n.getZ(0), n.getX(3), n.getY(3), n.getZ(3)]; })""")
    for n in normals:  # first two facets of every box: its bottom (−Z)
        assert n == pytest.approx([0, 0, -1, 0, 0, -1])


def test_lighting_rig_is_z_up_and_supersampled(viewer):
    assert viewer.load() is None
    info = js(viewer, """(() => { const v = __t.app().viewer; const hemi = v.scene.children.find(o => o.isHemisphereLight);
        return { hemi: hemi.position.clone().normalize().toArray(), env: !!v.scene.environment?.isTexture,
                 ratio: v.renderer.getPixelRatio(), shadows: v.renderer.shadowMap.enabled }; })()""")
    # HemisphereLight defaults to Y-up: in this Z-up world its sky must be +Z
    assert info["hemi"] == pytest.approx([0, 0, 1])
    assert info == {**info, "env": True, "ratio": 2, "shadows": True}


def test_dark_parts_keep_their_form(viewer):
    """A near-black part (#141414) must not render as one flat silhouette: its lit top and its
    side faces differ, and it stands out from the background."""
    assert viewer.load(focus="dark", view="iso") is None
    px = js(viewer, """(() => { const a = __t.app(); const v = a.viewer; v.render();
        const cam = v.camera; const gl = v.renderer.getContext(); const r = v.renderer.getPixelRatio();
        const H = gl.drawingBufferHeight;
        const sample = (x, y, z) => { const q = a.parts.items.get('dark').hull[0].clone().set(x, y, z).project(cam);
          const px = Math.round((q.x + 1) / 2 * gl.drawingBufferWidth), py = Math.round((q.y + 1) / 2 * H);
          const out = new Uint8Array(4); gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out); return [...out].slice(0, 3); };
        return { top: sample(-45, -45, 30), front: sample(-45, -60, 15), side: sample(-30, -45, 15),
                 bg: (() => { const out = new Uint8Array(4); gl.readPixels(2, H - 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out); return [...out].slice(0, 3); })() }; })()""")
    lum = {k: 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2] for k, v in px.items()}
    assert lum["top"] > lum["front"] + 4 and lum["front"] > lum["side"] + 2, px
    assert lum["top"] > lum["bg"] + 12, px
