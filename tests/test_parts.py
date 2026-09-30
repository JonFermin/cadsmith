"""Tests for the standard-parts library `mech.parts` (spec §6)."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from build123d import Axis, Location, Pos, Rot, Vector

from mech import parts
from mech.parts import (GearPair, LibPart, bearing, button_head_screw, clearance_hole,
                        extrusion_2020, extrusion_2040, fdm_hole, fit, gear_pair, gt2_pulley,
                        heat_set_insert, hex_nut, it_grade, mg996r, n20_gearmotor, nema17, nema23,
                        rack, rod, sg90, socket_head_screw, spur_gear, t8_leadscrew, t8_nut,
                        tap_drill, tolerance_zone, washer)
from mech.parts.fits import _EI, _ES, _IT, _RANGES
from mech.parts.gears import _ToothProfile


def arr(v: Vector) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z])


def bbox(part: LibPart) -> tuple[np.ndarray, np.ndarray]:
    bb = part.shape.bounding_box()
    return arr(bb.min), arr(bb.max)


def frame_pos(loc: Location) -> np.ndarray:
    return arr(loc.position)


def frame_dir(loc: Location) -> np.ndarray:
    return arr(Axis(loc).direction)


def common_volume(a, b) -> float:
    common = a & b          # None when the solids do not intersect at all
    return 0.0 if common is None else common.volume


# --- ISO 286 fits ---------------------------------------------------------------------------------

def test_fit_10_H7_g6() -> None:
    f = fit(10, "H7/g6")
    assert f["hole"] == (10.000, 10.015)
    assert f["shaft"] == (9.986, 9.995)
    assert f["clearance"] == (0.005, 0.029)


def test_fit_25_H7_p6_is_interference() -> None:
    f = fit(25, "H7/p6")
    assert f["hole"] == (25.000, 25.021)
    assert f["shaft"] == (25.022, 25.035)
    assert f["clearance"] == (-0.035, -0.001)


def test_fit_50_H8_f7() -> None:
    f = fit(50, "H8/f7")
    assert f["hole"] == (50.000, 50.039)
    assert f["shaft"] == (49.950, 49.975)
    assert f["clearance"] == (0.025, 0.089)


@pytest.mark.parametrize("nominal, spec, hole, shaft", [
    (8, "H7/h6", (8.000, 8.015), (7.991, 8.000)),
    (22, "H7/k6", (22.000, 22.021), (22.002, 22.015)),
    (20, "H7/f7", (20.000, 20.021), (19.959, 19.980)),
    (20, "H11/c11", (20.000, 20.130), (19.760, 19.890)),
    (100, "H7/p6", (100.000, 100.035), (100.037, 100.059)),
    (3, "H7/g6", (3.000, 3.010), (2.992, 2.998)),
])
def test_fit_table_values(nominal, spec, hole, shaft) -> None:
    f = fit(nominal, spec)
    assert f["hole"] == hole and f["shaft"] == shaft


def test_size_ranges_are_over_a_up_to_and_including_b() -> None:
    assert it_grade(10, 7) == 15            # 10 belongs to (6, 10]
    assert it_grade(10.001, 7) == 18        # just above -> (10, 18]
    assert it_grade(120, 7) == 35
    with pytest.raises(ValueError):
        it_grade(121, 7)


def test_c_deviation_uses_intermediate_steps() -> None:
    assert tolerance_zone(40, "c11") == (39.720, 39.880)     # (30, 40]: es = −120
    assert tolerance_zone(45, "c11") == (44.710, 44.870)     # (40, 50]: es = −130


def test_js_symmetric_and_odd_it_rounding() -> None:
    assert tolerance_zone(20, "js6") == (19.9935, 20.0065)   # IT6 = 13: ±6.5 (no rounding)
    assert tolerance_zone(20, "js7") == (19.990, 20.010)     # IT7 = 21 -> ±10 (grades 7..11)


def test_k_deviation_only_for_grades_4_to_7() -> None:
    assert tolerance_zone(22, "k6")[0] == 22.002
    assert tolerance_zone(22, "k8")[0] == 22.000


def test_hole_zone_mirrors_shaft_zone() -> None:
    lo, hi = tolerance_zone(10, "G7")
    assert (lo, hi) == (10.005, 10.020)                       # EI = −es(g) = +5 µm


def test_it_table_matches_iso_tolerance_factor() -> None:
    """IT = k·i with i = 0.45·∛D + 0.001·D (µm), D = geometric mean of the range (ISO 286-1)."""
    lows = (1, *_RANGES[:-1])
    factors = {5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100, 12: 160, 13: 250}
    for j, (lo, hi) in enumerate(zip(lows, _RANGES)):
        if j == 0:          # the first range's values are set by convention, not by the formula
            continue
        d = math.sqrt(lo * hi)
        i = 0.45 * d ** (1 / 3) + 0.001 * d
        for grade, k in factors.items():
            assert _IT[grade][j] == pytest.approx(k * i, rel=0.10), (grade, lo, hi)
    for j in range(len(_RANGES)):                             # monotonic in grade and in size
        assert all(_IT[g][j] < _IT[g + 1][j] or _IT[g][j] == _IT[g + 1][j] for g in range(1, 13))
    for g in _IT:
        assert list(_IT[g]) == sorted(_IT[g])


def test_fundamental_deviations_match_iso_formulas() -> None:
    lows = (1, *_RANGES[:-1])
    formulas = {
        "d": lambda D: -16 * D**0.44, "e": lambda D: -11 * D**0.41,
        "f": lambda D: -5.5 * D**0.41, "g": lambda D: -2.5 * D**0.34,
        "n": lambda D: 5 * D**0.34, "k": lambda D: 0.6 * D ** (1 / 3),
    }
    for j, (lo, hi) in enumerate(zip(lows, _RANGES)):
        if j == 0:
            continue
        d = math.sqrt(lo * hi)
        for letter, fn in formulas.items():
            table = (_ES | _EI)[letter][j]
            assert abs(table - fn(d)) <= max(0.6, 0.05 * abs(fn(d))), (letter, lo, hi)
        assert _EI["m"][j] == _IT[7][j] - _IT[6][j]            # m: ei = IT7 − IT6
        assert 0 <= _EI["p"][j] - _IT[7][j] <= 5               # p: ei = IT7 + 0..5


def test_fit_rejects_bad_specs() -> None:
    for bad in ("H7g6", "g6/H7", "H7/x6", "K7/h6"):
        with pytest.raises(ValueError):
            fit(10, bad)


def test_hole_helpers() -> None:
    assert clearance_hole("M3") == 3.4
    assert clearance_hole(3, "close") == 3.2
    assert clearance_hole("M8", "loose") == 10.0
    assert tap_drill("M3") == 2.5
    assert tap_drill("M2.5") == 2.05
    assert tap_drill("M8") == 6.75
    assert fdm_hole(5) == pytest.approx(5.2)
    assert heat_set_insert("M3")["hole_d"] == 4.0
    with pytest.raises(KeyError, match="M3"):
        clearance_hole("M7")


# --- LibPart placement ----------------------------------------------------------------------------

def test_pos_times_libpart_moves_shape_and_frames() -> None:
    m = nema17()
    moved = Pos(10, 20, 30) * Rot(0, 0, 90) * m
    assert isinstance(moved, LibPart) and moved.bom == m.bom and moved.mass_g == m.mass_g
    np.testing.assert_allclose(frame_pos(moved.frames["shaft"]), (10, 20, 30), atol=1e-9)
    # hole_1 at (+15.5, +15.5) rotated 90° about Z -> (−15.5, +15.5), then translated
    np.testing.assert_allclose(frame_pos(moved.frames["hole_1"]), (-5.5, 35.5, 30), atol=1e-9)
    lo, hi = bbox(moved)
    np.testing.assert_allclose(lo, (10 - 21.15, 20 - 21.15, -10), atol=1e-6)
    np.testing.assert_allclose(hi, (10 + 21.15, 20 + 21.15, 54), atol=1e-6)
    # original untouched
    np.testing.assert_allclose(frame_pos(m.frames["shaft"]), (0, 0, 0), atol=1e-12)


def test_rmul_rejects_non_locations() -> None:
    with pytest.raises(TypeError):
        3 * nema17()


def test_mate_lands_frame_on_target() -> None:
    screw = socket_head_screw("M3", 12)
    target = Pos(5, -2, 7) * Rot(90, 0, 0)           # frame Z -> world −Y
    placed = screw.mate("head", target)
    np.testing.assert_allclose(frame_pos(placed.frames["head"]), (5, -2, 7), atol=1e-9)
    np.testing.assert_allclose(frame_dir(placed.frames["head"]), (0, -1, 0), atol=1e-9)
    # the tip is 12 mm down the screw axis (−Z of the head frame -> world +Y)
    np.testing.assert_allclose(frame_pos(placed.frames["tip"]), (5, 10, 7), atol=1e-9)
    lo, hi = bbox(placed)
    assert lo[1] == pytest.approx(-2 - 3.0, abs=1e-6)   # head top 3 mm on the −Y side
    assert hi[1] == pytest.approx(10, abs=1e-6)
    with pytest.raises(KeyError, match="head"):
        screw.mate("nope", target)


def test_mate_motor_shaft_onto_frame() -> None:
    target = Pos(100, 0, 50) * Rot(0, 90, 0)          # shaft along +X
    motor = nema17().mate("shaft", target)
    np.testing.assert_allclose(frame_dir(motor.frames["shaft"]), (1, 0, 0), atol=1e-9)
    assert bbox(motor)[1][0] == pytest.approx(124, abs=1e-6)     # shaft tip 24 mm out
    assert bbox(motor)[0][0] == pytest.approx(60, abs=1e-6)      # 40 mm body behind the face


# --- fasteners ------------------------------------------------------------------------------------

@pytest.mark.parametrize("size, dk, k", [("M2", 3.8, 2.0), ("M3", 5.5, 3.0), ("M5", 8.5, 5.0),
                                         ("M8", 13.0, 8.0)])
def test_socket_head_screw_dimensions(size, dk, k) -> None:
    s = socket_head_screw(size, 16)
    lo, hi = bbox(s)
    np.testing.assert_allclose(hi - lo, (dk, dk, 16 + k), atol=1e-6)
    assert hi[2] == pytest.approx(k) and lo[2] == pytest.approx(-16)
    np.testing.assert_allclose(frame_pos(s.frames["head"]), (0, 0, 0))
    np.testing.assert_allclose(frame_pos(s.frames["tip"]), (0, 0, -16))
    np.testing.assert_allclose(frame_dir(s.frames["tip"]), (0, 0, 1))
    assert "ISO 4762" in s.bom and s.mass_g == pytest.approx(s.shape.volume * 7.85e-3)


def test_button_head_screw() -> None:
    s = button_head_screw("M4", 10)
    lo, hi = bbox(s)
    np.testing.assert_allclose(hi - lo, (7.6, 7.6, 12.2), atol=1e-6)
    with pytest.raises(KeyError):
        button_head_screw("M2", 5)


def test_hex_nut_and_washer() -> None:
    n = hex_nut("M3")
    lo, hi = bbox(n)
    assert hi[1] - lo[1] == pytest.approx(5.5)                         # across flats along Y
    assert hi[0] - lo[0] == pytest.approx(5.5 / math.cos(math.pi / 6))  # across corners along X
    assert (lo[2], hi[2]) == pytest.approx((0, 2.4))
    hexagon = 3 * math.sqrt(3) / 2 * (5.5 / math.sqrt(3)) ** 2
    assert n.shape.volume == pytest.approx((hexagon - math.pi * 1.5**2) * 2.4, rel=1e-6)
    w = washer("M8")
    lo, hi = bbox(w)
    np.testing.assert_allclose(hi - lo, (16, 16, 1.6), atol=1e-6)
    assert w.mass_g == pytest.approx(math.pi / 4 * (16**2 - 8.4**2) * 1.6 * 7.85e-3, rel=1e-6)
    np.testing.assert_allclose(frame_pos(w.frames["top"]), (0, 0, 1.6))


# --- bearings -------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, d, D, B", [("623", 3, 10, 4), ("608", 8, 22, 7), ("688", 8, 16, 5),
                                           ("6001", 12, 28, 8), ("6202", 15, 35, 11),
                                           ("LM8UU", 8, 15, 24), ("LM10UU", 10, 19, 29)])
def test_bearing_dimensions(name, d, D, B) -> None:
    b = bearing(name)
    lo, hi = bbox(b)
    np.testing.assert_allclose(lo, (-D / 2, -D / 2, -B / 2), atol=1e-6)
    np.testing.assert_allclose(hi, (D / 2, D / 2, B / 2), atol=1e-6)
    # the bore is empty: the axis is outside the solid, the bore wall is on it
    assert not b.shape.is_inside(Vector(0, 0, 0))
    assert b.shape.distance_to(Vector(0, 0, 0)) == pytest.approx(d / 2, abs=1e-6)
    np.testing.assert_allclose(frame_pos(b.frames["center"]), (0, 0, 0))
    assert b.mass_g > 0


def test_bearing_suffixes_and_errors() -> None:
    assert bearing("608-2RS").bom == bearing("6082RS").bom == bearing("608ZZ").bom
    assert "6202" in bearing("62022RS").bom
    with pytest.raises(KeyError, match="608"):
        bearing("609")


# --- motors ---------------------------------------------------------------------------------------

def _shaft_solid(part: LibPart):
    return max(part.shape.solids(), key=lambda s: s.bounding_box().max.Z)


@pytest.mark.parametrize("make, face, spacing, shaft_d, length", [
    (lambda: nema17(), 42.3, 31.0, 5.0, 40), (lambda: nema17(48), 42.3, 31.0, 5.0, 48),
    (lambda: nema23(), 56.4, 47.14, 6.35, 56),
])
def test_stepper_geometry(make, face, spacing, shaft_d, length) -> None:
    m = make()
    lo, hi = bbox(m)
    np.testing.assert_allclose((hi - lo)[:2], (face, face), atol=1e-6)
    assert lo[2] == pytest.approx(-length)
    s = _shaft_solid(m).bounding_box()
    assert s.size.X == pytest.approx(shaft_d) and s.min.Z == pytest.approx(0)
    holes = np.array([frame_pos(m.frames[f"hole_{i}"]) for i in range(1, 5)])
    np.testing.assert_allclose(np.abs(holes[:, :2]), spacing / 2, atol=1e-9)
    assert np.all(holes[:, 2] == 0)
    np.testing.assert_allclose(np.linalg.norm(holes[0] - holes[1]), spacing)
    np.testing.assert_allclose(frame_dir(m.frames["shaft"]), (0, 0, 1))
    assert "NEMA" in m.bom and m.mass_g > 200


def test_nema17_mass_is_catalog_like() -> None:
    assert nema17().mass_g == pytest.approx(280, rel=0.05)
    assert nema23().mass_g == pytest.approx(700, rel=0.05)


def test_servos() -> None:
    for make, span, width, pitch, n_holes, mass in [(sg90, 32.2, 11.8, 27.8, 2, 9.0),
                                                    (mg996r, 54.0, 19.7, 49.5, 4, 55.0)]:
        s = make()
        lo, hi = bbox(s)
        assert hi[0] - lo[0] == pytest.approx(span)
        assert hi[1] - lo[1] == pytest.approx(width)
        holes = [frame_pos(s.frames[f"hole_{i}"]) for i in range(1, n_holes + 1)]
        xs = sorted({round(h[0], 9) for h in holes})
        assert xs[-1] - xs[0] == pytest.approx(pitch)
        # holes are through the ears: the hole center is not inside material at mid-ear height
        for h in holes:
            assert not s.shape.is_inside(Vector(h[0], h[1], 1.0))
        assert s.shape.is_inside(Vector(0, 0, -5))                      # body under the shaft
        np.testing.assert_allclose(frame_pos(s.frames["shaft"]), (0, 0, 0))
        assert s.mass_g == mass


def test_sg90_heights() -> None:
    lo, hi = bbox(sg90())
    assert hi[2] - lo[2] == pytest.approx(29.9)
    assert lo[2] == pytest.approx(-15.9)


def test_servo_horn_frame_is_the_spline_tip() -> None:
    for make, tip in [(sg90, 14.0), (mg996r, 16.4)]:
        s = make()
        np.testing.assert_allclose(frame_pos(s.frames["horn"]), (0, 0, tip), atol=1e-9)
        np.testing.assert_allclose(frame_dir(s.frames["horn"]), (0, 0, 1), atol=1e-12)
        assert bbox(s)[1][2] == pytest.approx(tip)  # nothing of the servo above the horn seat


def test_n20() -> None:
    m = n20_gearmotor()
    lo, hi = bbox(m)
    np.testing.assert_allclose(lo, (-6, -5, -24), atol=1e-6)
    np.testing.assert_allclose(hi, (6, 5, 10), atol=1e-6)
    d = frame_pos(m.frames["hole_1"]) - frame_pos(m.frames["hole_2"])
    assert np.linalg.norm(d) == pytest.approx(9.0)


# --- gears ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("m, z", [(1, 15), (1, 30), (1.5, 20), (0.5, 60), (2, 9)])
def test_spur_gear_dimensions(m, z) -> None:
    g = spur_gear(m, z, 6)
    r, ra = m * z / 2, m * z / 2 + m
    lo, hi = bbox(g)
    assert hi[0] == pytest.approx(ra, abs=0.01 * m)      # tooth on +X reaches the tip circle
    assert hi[0] <= ra + 1e-9
    assert (lo[2], hi[2]) == pytest.approx((0, 6))
    # a tooth is centered on +X: pitch point inside, the neighbouring space center outside
    assert g.shape.is_inside(Vector(r, 0, 3))
    a = math.pi / z
    assert not g.shape.is_inside(Vector(r * math.cos(a), r * math.sin(a), 3))
    np.testing.assert_allclose(frame_pos(g.frames["axis"]), (0, 0, 0))


def test_spur_gear_tooth_thickness_includes_backlash_thinning() -> None:
    m, z, b = 2, 20, 0.2
    prof = _ToothProfile(m, z, b, 20)
    r = m * z / 2
    # tooth thickness (arc) at the pitch circle = πm/2 − b/2
    assert 2 * r * prof.half_angle(np.array([r]))[0] == pytest.approx(math.pi * m / 2 - b / 2)
    # base-circle half angle ψ + inv α (involute property)
    alpha = math.radians(20)
    assert prof.involute(np.array([prof.rb]))[0] == pytest.approx(
        (math.pi * m / 2 - b / 2) / (2 * r) + math.tan(alpha) - alpha)
    thick = spur_gear(m, z, 5, backlash=0).shape.volume
    thin = spur_gear(m, z, 5, backlash=b).shape.volume
    # thinning of b/2 over the working depth ~2m of every tooth
    assert thick - thin == pytest.approx(z * 5 * (b / 2) * 2 * m, rel=0.15)


def test_spur_gear_bore() -> None:
    g = spur_gear(1, 30, 6, bore=5)
    assert not g.shape.is_inside(Vector(0, 0, 3))
    assert g.shape.distance_to(Vector(0, 0, 3)) == pytest.approx(2.5, abs=1e-6)


def test_spur_gear_too_few_teeth() -> None:
    with pytest.raises(ValueError):
        spur_gear(1, 3, 5)


PAIRS = [(15, 30), (12, 13), (17, 17), (20, 41), (9, 25), (16, 24)]


@pytest.mark.parametrize("z1, z2", PAIRS)
def test_gear_pair_placement_and_home_mesh(z1, z2) -> None:
    m = 1.25
    gp = gear_pair(m, z1, z2, 5)
    assert isinstance(gp, GearPair)
    assert gp.ratio == pytest.approx(-z1 / z2)
    assert gp.center_distance == pytest.approx(m * (z1 + z2) / 2)
    np.testing.assert_allclose(frame_pos(gp.g2.frames["axis"]), (gp.center_distance, 0, 0),
                               atol=1e-9)
    np.testing.assert_allclose(frame_dir(gp.g2.frames["axis"]), (0, 0, 1), atol=1e-12)
    # g1's tooth on +X sits in the tooth space of g2 facing −X
    pitch_pt = Vector(m * z1 / 2, 0, 2.5)
    assert gp.g1.shape.is_inside(pitch_pt) and not gp.g2.shape.is_inside(pitch_pt)
    assert common_volume(gp.g1.shape, gp.g2.shape) == 0


@pytest.mark.parametrize("z1, z2", [(15, 30), (12, 13), (17, 17), (20, 41)])
def test_gear_pair_stays_meshed_when_rotated_by_ratio(z1, z2) -> None:
    gp = gear_pair(1, z1, z2, 4)
    a = gp.center_distance
    for th in (3.0, 7.0, 11.3, 23.9):
        g1 = Rot(0, 0, th) * gp.g1.shape
        g2 = Pos(a, 0, 0) * Rot(0, 0, gp.ratio * th) * Pos(-a, 0, 0) * gp.g2.shape
        assert common_volume(g1, g2) == 0, (z1, z2, th)


def test_gear_pair_wrong_direction_overlaps() -> None:
    """Negative control: the zero-overlap checks above are sensitive to phasing."""
    gp = gear_pair(1, 15, 30, 4)
    a = gp.center_distance
    g1 = Rot(0, 0, 7) * gp.g1.shape
    g2 = Pos(a, 0, 0) * Rot(0, 0, +7 * 15 / 30) * Pos(-a, 0, 0) * gp.g2.shape
    assert common_volume(g1, g2) > 1.0


def test_gear_pair_home_gap_equals_backlash() -> None:
    """Normal gap on each flank = (b/2)·cos α; polyline chords may only add a hair."""
    b = 0.1
    gp = gear_pair(1, 15, 30, 4, backlash=b)
    gap = gp.g1.shape.distance_to(gp.g2.shape)
    expected = b / 2 * math.cos(math.radians(20))
    assert expected - 1e-4 <= gap <= expected + 0.006


def test_rack_dimensions_and_mesh() -> None:
    m, z = 1, 20
    rk = rack(m, 60, 6, 8)
    lo, hi = bbox(rk)
    np.testing.assert_allclose(lo, (-30, -8, 0), atol=1e-6)
    np.testing.assert_allclose(hi, (30, 1, 6), atol=1e-6)
    r = m * z / 2
    g = spur_gear(m, z, 6)
    expected_gap = 0.05 / 2 * math.cos(math.radians(20))
    for th in (0.0, 5.0, 13.0):
        gear = Pos(0, r, 0) * Rot(0, 0, -90 + th) * g.shape
        rk_s = Pos(math.pi * r * th / 180, 0, 0) * rk.shape
        assert common_volume(gear, rk_s) == 0
        # involute against a straight flank: constant normal gap while rolling
        assert gear.distance_to(rk_s) == pytest.approx(expected_gap, abs=2.5e-3)
    wrong = Pos(-math.pi * r * 5 / 180, 0, 0) * rk.shape
    assert common_volume(Pos(0, r, 0) * Rot(0, 0, -90 + 5) * g.shape, wrong) > 1.0


def test_gt2_pulley() -> None:
    p = gt2_pulley(20, 6, 5)
    pd = 2 * 20 / math.pi
    od = pd - 2 * 0.254
    lo, hi = bbox(p)
    assert hi[2] == pytest.approx(8)
    teeth = next(s for s in p.shape.solids() if s.bounding_box().min.Z == pytest.approx(1))
    radii = [r for v in teeth.vertices() if (r := math.hypot(v.X, v.Y)) > 2.5 + 1e-6]  # not bore
    assert max(radii) == pytest.approx(od / 2, abs=1e-9)
    assert min(radii) == pytest.approx(od / 2 - 0.75, abs=1e-3)   # groove bottom, 0.75 deep
    land = math.pi / 20                                            # between grooves 0 and 1
    r_in = od / 2 - 0.2
    assert p.shape.is_inside(Vector(r_in * math.cos(land), r_in * math.sin(land), 4))
    assert not p.shape.is_inside(Vector(od / 2 - 0.2, 0, 4))      # groove 0 on +X
    assert hi[0] - lo[0] == pytest.approx(od + 3)                  # flanges 1.5 mm proud
    assert not p.shape.is_inside(Vector(0, 0, 4))           # bore
    np.testing.assert_allclose(frame_pos(p.frames["belt"]), (0, 0, 4))


# --- motion parts ---------------------------------------------------------------------------------

def test_rods_and_leadscrew() -> None:
    r = rod(8, 200)
    lo, hi = bbox(r)
    np.testing.assert_allclose(lo, (-4, -4, 0), atol=1e-6)
    np.testing.assert_allclose(hi, (4, 4, 200), atol=1e-6)
    assert r.mass_g == pytest.approx(math.pi * 16 * 200 * 7.85e-3, rel=1e-6)
    np.testing.assert_allclose(frame_pos(r.frames["top"]), (0, 0, 200))
    s = t8_leadscrew(300)
    assert parts.T8_LEAD == 8.0 and "lead 8" in s.bom
    assert bbox(s)[1][2] == pytest.approx(300)


def test_t8_nut() -> None:
    n = t8_nut()
    lo, hi = bbox(n)
    np.testing.assert_allclose(lo, (-11, -11, -11.5), atol=1e-6)
    np.testing.assert_allclose(hi, (11, 11, 3.5), atol=1e-6)
    assert n.shape.distance_to(Vector(0, 0, 0)) == pytest.approx(4.0, abs=1e-6)   # ø8 bore
    holes = [frame_pos(n.frames[f"hole_{i}"]) for i in range(1, 5)]
    np.testing.assert_allclose([np.linalg.norm(h) for h in holes], 8.0)


def test_extrusions() -> None:
    e = extrusion_2020(300)
    lo, hi = bbox(e)
    np.testing.assert_allclose(lo, (-10, -10, 0), atol=1e-6)
    np.testing.assert_allclose(hi, (10, 10, 300), atol=1e-6)
    # slot openings (6.2 wide) on all four faces, core bore on the axis
    for x, y in ((9.5, 0), (-9.5, 0), (0, 9.5), (0, -9.5), (0, 0)):
        assert not e.shape.is_inside(Vector(x, y, 150))
    assert e.shape.is_inside(Vector(9.5, 4.0, 150))          # lip beside the opening
    e2 = extrusion_2040(100)
    lo, hi = bbox(e2)
    np.testing.assert_allclose(hi - lo, (40, 20, 100), atol=1e-6)
    for x, y in ((10, 9.5), (-10, 9.5), (10, -9.5), (19.5, 0), (-19.5, 0), (10, 0), (-10, 0)):
        assert not e2.shape.is_inside(Vector(x, y, 50))
    assert e.mass_g == pytest.approx(0.49 * 300) and e2.mass_g == pytest.approx(0.87 * 100)


# --- validity and speed ---------------------------------------------------------------------------

ALL_PARTS = [
    lambda: socket_head_screw("M3", 10), lambda: button_head_screw("M5", 16),
    lambda: hex_nut("M4"), lambda: washer("M3"), lambda: bearing("608"), lambda: bearing("LM8UU"),
    lambda: nema17(), lambda: nema23(), lambda: sg90(), lambda: mg996r(), lambda: n20_gearmotor(),
    lambda: spur_gear(1, 30, 6, bore=5), lambda: spur_gear(1, 60, 6), lambda: rack(1, 100, 6, 8),
    lambda: gt2_pulley(20, 6, 5), lambda: rod(8, 300), lambda: t8_leadscrew(300),
    lambda: t8_nut(), lambda: extrusion_2020(500), lambda: extrusion_2040(500),
]


@pytest.mark.parametrize("make", ALL_PARTS)
def test_parts_are_valid_fast_and_described(make) -> None:
    make()                                   # warm-up (first-call overheads)
    t0 = time.perf_counter()
    p = make()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.3, f"{p.bom}: {elapsed:.3f} s"
    assert p.shape.is_valid and p.shape.volume > 0
    assert p.bom and p.bom.isascii()
    assert p.frames and all(isinstance(f, Location) for f in p.frames.values())
    # no overlapping solids inside a multi-solid part (mass props sum solid volumes)
    solids = p.shape.solids()
    assert sum(s.volume for s in solids) == pytest.approx(p.shape.volume, rel=1e-9)
    if 1 < len(solids) <= 6:
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                assert common_volume(a, b) == pytest.approx(0, abs=1e-6)


def test_tooth_bearing_parts_are_tagged_and_keep_the_tag_when_moved() -> None:
    """`kind` marks the parts whose teeth mesh (Assembly.meshing_pairs only excuses those)."""
    gp = gear_pair(1, 15, 30, 6)
    assert spur_gear(1, 20, 5).kind == gp.g1.kind == gp.g2.kind == "gear"
    assert rack(1, 60, 6, 8).kind == "rack" and gt2_pulley(20, 6, 5).kind == "pulley"
    assert (Pos(3, 0, 0) * Rot(0, 0, 30) * gp.g2).kind == "gear"
    assert gp.g1.mate("axis", Location((10, 0, 0))).kind == "gear"
    assert nema17().kind is None and bearing("608").kind is None
