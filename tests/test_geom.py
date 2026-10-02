"""mech.geom: vectors, transforms, JSON formatting, slugs and the modeling helpers."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
from build123d import Box, Location, Pos, Rot, Vector

from mech.geom import (
    circle_intersect,
    fnum,
    from_location,
    link,
    rot_about_line,
    slug,
    to_json16,
    to_location,
    transform_aabb,
    transform_points,
    translation,
    unit,
    vec3,
)


def random_rigid(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    return translation(rng.uniform(-50, 50, 3)) @ rot_about_line(rng.uniform(-20, 20, 3), axis, rng.uniform(-180, 180))


# ------------------------------------------------------------------------------ vectors


@pytest.mark.parametrize("v", [(1, 2, 3), [1.0, 2.0, 3.0], np.array([1, 2, 3]), Vector(1, 2, 3)])
def test_vec3_accepts_common_types(v):
    out = vec3(v)
    assert out.dtype == float and out.shape == (3,)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_vec3_copies_and_rejects_bad_input():
    a = np.array([1.0, 2.0, 3.0])
    vec3(a)[0] = 99.0
    assert a[0] == 1.0
    for bad in [(1, 2), (1, 2, 3, 4), "abc", (1, float("nan"), 0), None]:
        with pytest.raises(ValueError):
            vec3(bad)


def test_unit():
    np.testing.assert_allclose(unit((0, 3, 4)), [0, 0.6, 0.8])
    with pytest.raises(ValueError):
        unit((0, 0, 0))


# ------------------------------------------------------------------------------ transforms


def test_rot_about_line_right_hand_rule_and_offset_line():
    np.testing.assert_allclose(transform_points(rot_about_line((0, 0, 0), (0, 0, 1), 90), (1, 0, 0)), [0, 1, 0],
                               atol=1e-15)
    # about the vertical line through (10, 0, 0): (20, 0, 5) swings to (10, 10, 5)
    np.testing.assert_allclose(transform_points(rot_about_line((10, 0, 0), (0, 0, 2), 90), (20, 0, 5)), [10, 10, 5],
                               atol=1e-12)
    # points on the line are fixed; the axis direction's length and sign convention
    T = rot_about_line((1, 2, 3), (1, 1, 1), 120)
    np.testing.assert_allclose(transform_points(T, (2, 3, 4)), [2, 3, 4], atol=1e-12)
    np.testing.assert_allclose(transform_points(T, (2, 2, 3)), [1, 3, 3], atol=1e-12)  # cyclic x->y->z
    np.testing.assert_allclose(rot_about_line((0, 0, 0), (0, 0, -1), 30), rot_about_line((0, 0, 0), (0, 0, 1), -30))
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rot_about_z_keeps_planar_entries_exact():
    R = rot_about_line((3, 4, 0), (0, 0, 1), 37.0)
    assert R[2, 2] == 1.0 and R[2, 0] == R[2, 1] == R[0, 2] == R[1, 2] == 0.0 and R[2, 3] == 0.0


def test_transform_points_single_and_batch():
    T = translation((1, 2, 3)) @ rot_about_line((0, 0, 0), (0, 0, 1), 90)
    pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(transform_points(T, pts), [[1, 3, 3], [0, 2, 3], [1, 2, 4]], atol=1e-12)
    np.testing.assert_allclose(transform_points(T, Vector(1, 0, 0)), [1, 3, 3], atol=1e-12)


def test_to_location_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = random_rigid(rng)
        np.testing.assert_allclose(from_location(to_location(T)), T, atol=1e-12)


def test_to_location_moves_shapes_like_transform_points():
    rng = np.random.default_rng(2)
    T = random_rigid(rng)
    box = Pos(3, -4, 5) * Box(10, 20, 30)
    moved = to_location(T) * box
    np.testing.assert_allclose(vec3(moved.center()), transform_points(T, vec3(box.center())), atol=1e-9)
    got = sorted(tuple(np.round(vec3(v), 6)) for v in moved.vertices())
    want = sorted(tuple(np.round(transform_points(T, vec3(v)), 6)) for v in box.vertices())
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_from_location_matches_build123d_semantics():
    T = from_location(Location((10, 20, 30), (0, 0, 90)))
    np.testing.assert_allclose(T[:3, 3], [10, 20, 30])
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(from_location(Pos(1, 2, 3) * Rot(0, 0, 90)),
                               translation((1, 2, 3)) @ rot_about_line((0, 0, 0), (0, 0, 1), 90), atol=1e-12)
    with pytest.raises(TypeError):
        from_location(np.eye(4))


def test_to_location_orthonormalizes_non_orthonormal_input():
    rng = np.random.default_rng(3)
    T = random_rigid(rng)
    drifted = T.copy()
    drifted[:3, :3] += rng.normal(scale=1e-3, size=(3, 3))
    U, _, Vt = np.linalg.svd(drifted[:3, :3])
    out = from_location(to_location(drifted))
    R = out[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R, U @ Vt, atol=1e-12)  # nearest rotation
    np.testing.assert_allclose(out[:3, 3], T[:3, 3], atol=1e-12)  # translation untouched
    assert np.abs(R - T[:3, :3]).max() < 1e-2

    scaled = T.copy()
    scaled[:3, :3] *= 2.5  # uniform scale is removed
    np.testing.assert_allclose(from_location(to_location(scaled)), T, atol=1e-12)


@pytest.mark.parametrize("bad", [np.diag([1.0, 1.0, -1.0, 1.0]), np.diag([1.0, 1.0, 0.0, 1.0]), np.eye(3),
                                 np.full((4, 4), np.nan)])
def test_to_location_rejects_reflections_singular_and_malformed(bad):
    with pytest.raises(ValueError):
        to_location(bad)


def test_transform_aabb_matches_corner_brute_force():
    rng = np.random.default_rng(4)
    lo, hi = np.array([-1.0, 2.0, 0.5]), np.array([4.0, 3.0, 9.0])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for _ in range(10):
        T = random_rigid(rng)
        moved = transform_points(T, corners)
        a, b = transform_aabb(lo, hi, T)
        np.testing.assert_allclose(a, moved.min(axis=0), atol=1e-12)
        np.testing.assert_allclose(b, moved.max(axis=0), atol=1e-12)


# ------------------------------------------------------------------------------ JSON / slugs


def test_fnum():
    assert fnum(1.23456789) == 1.23457
    assert fnum(123456789) == 123457000.0
    assert fnum(np.float32(0.1)) == 0.1
    assert fnum(-2.5e-7) == -2.5e-7
    assert fnum(float("nan")) is None and fnum(float("inf")) is None and fnum(None) is None


def test_to_json16_is_column_major_rounded_and_json_safe():
    T = translation((1.234567891, -2, 3)) @ rot_about_line((0, 0, 0), (0, 0, 1), 90)
    out = to_json16(T)
    assert len(out) == 16
    assert out[12:15] == [1.23457, -2.0, 3.0] and out[15] == 1.0  # translation is the 4th column
    assert out[0:4] == [0.0, 1.0, 0.0, 0.0]  # first column = image of x = +y (cos 90° snapped to 0)
    assert out[4:8] == [-1.0, 0.0, 0.0, 0.0]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("name", ["CON", "con", "Nul", "aux", "PRN", "COM1", "com9", "LPT1", "lpt3"])
def test_slug_avoids_windows_reserved_names(name):
    s = slug(name)
    assert s.lower() == name.lower() + "_"
    assert re.fullmatch(r"[a-z0-9_-]+", s)


def test_slug_sanitizes_and_dedupes_case_insensitively():
    assert slug("Crank Arm #2") == "crank_arm_2"
    assert slug("Zahnräder: Stufe/1") == "zahnrader_stufe_1"
    assert slug("../../etc/passwd") == "etc_passwd"
    assert slug("   ") == "part" and slug("日本") == "part"
    assert len(slug("x" * 300)) <= 64
    taken = {"Frame"}
    assert slug("frame", taken) == "frame_2"
    assert slug("FRAME", taken) == "frame_3"
    assert slug("con", taken) == "con_" and slug("CON", taken) == "con__2"
    assert {"frame_2", "frame_3", "con_", "con__2"} <= taken
    names = ["Link A", "link-a", "link_a", "LINK A", "link a"]
    seen: set[str] = set()
    slugs = [slug(n, seen) for n in names]
    assert len({s.lower() for s in slugs}) == len(names)
    assert all(re.fullmatch(r"[a-z0-9_-]+", s) for s in slugs)


# ------------------------------------------------------------------------------ modeling helpers


def slot_volume(length, width, thickness, hole):
    return thickness * (length * width + math.pi * width**2 / 4 - 2 * math.pi * hole**2 / 4)


def test_link_geometry_in_place():
    p0, p1 = (10, 5, 0), (10 + 30, 5 + 40, 7)  # 50 mm apart in plan; p1's z is ignored (projected)
    bar = link(p0, p1, width=10, thickness=5, z=2)
    assert bar.volume == pytest.approx(slot_volume(50, 10, 5, 4), rel=1e-9)
    bb = bar.bounding_box()
    assert bb.min.Z == pytest.approx(2) and bb.max.Z == pytest.approx(7)
    assert bb.min.X == pytest.approx(5) and bb.max.X == pytest.approx(45)  # end arcs of radius 5
    # the holes are centered on both points: the hole axes don't touch material
    for x, y in [(10, 5), (40, 45)]:
        assert not bar.is_inside(Vector(x, y, 4.5))
        assert bar.is_inside(Vector(x + 3.5 * 0.6, y - 3.5 * 0.8, 4.5))  # beside the hole, inside the bar


def test_link_defaults_hole_none_and_normal():
    bar = link((0, 0, 3), (40, 0, 3), width=10, thickness=4)  # z defaults to p0's → bottom at 3
    assert bar.bounding_box().min.Z == pytest.approx(3)
    assert bar.volume == pytest.approx(slot_volume(40, 10, 4, 4), rel=1e-9)
    solid = link((0, 0, 0), (40, 0, 0), width=10, thickness=4, hole=0)
    assert solid.volume == pytest.approx(slot_volume(40, 10, 4, 0), rel=1e-9)
    side = link((0, 0, 0), (0, 0, 40), width=10, thickness=4, normal=(1, 0, 0), z=-2)  # in the YZ plane
    bb = side.bounding_box()
    assert (bb.min.X, bb.max.X) == (pytest.approx(-2), pytest.approx(2))
    assert (bb.min.Z, bb.max.Z) == (pytest.approx(-5), pytest.approx(45))
    disc = link((1, 1, 0), (1, 1, 0), width=10, thickness=2, hole=0)
    assert disc.volume == pytest.approx(math.pi * 25 * 2, rel=1e-9)
    with pytest.raises(ValueError):
        link((0, 0, 0), (10, 0, 0), width=4, thickness=2, hole=5)


def test_circle_intersect():
    A, O4 = (40, 0, 0), (100, 0, 0)
    B = circle_intersect(A, 90, O4, 80, side=+1)
    assert isinstance(B, tuple) and len(B) == 3 and all(isinstance(v, float) for v in B)
    assert math.dist(B, A) == pytest.approx(90) and math.dist(B, O4) == pytest.approx(80)
    assert B[1] > 0  # +1 = left of c0→c1 looking down +Z
    B2 = circle_intersect(A, 90, O4, 80, side=-1)
    assert B2[1] == pytest.approx(-B[1]) and B2[0] == pytest.approx(B[0])
    # 3-4-5 triangle in the XZ plane (normal +Y): left of +X looking down +Y is −Z
    P = circle_intersect((0, 0, 0), 4, (5, 0, 0), 3, side=+1, normal=(0, 1, 0))
    np.testing.assert_allclose(P, [3.2, 0, -2.4], atol=1e-12)
    np.testing.assert_allclose(circle_intersect((0, 0, 0), 2, (5, 0, 0), 3), [2, 0, 0], atol=1e-7)  # tangent
    with pytest.raises(ValueError):
        circle_intersect((0, 0, 0), 1, (10, 0, 0), 2)
    with pytest.raises(ValueError):
        circle_intersect((0, 0, 0), 1, (0, 0, 0), 1)


def test_inside_classifies_each_solid_of_a_compound():
    """OCC's classifier on a multi-solid compound calls interior points of later solids outside;
    ``inside`` tests every solid (review p5d: library bearings, motors, list-of-shapes parts)."""
    from build123d import Compound, Plane, Rectangle

    from mech.geom import inside
    from mech.parts import bearing

    two = Compound([Box(10, 10, 10), Pos(20, 0, 0) * Box(10, 10, 10)])
    assert inside(two, (0, 0, 0)) and inside(two, (20, 0, 0)) and inside(two, (5, 0, 0))  # boundary counts
    assert not inside(two, (10, 0, 0))
    ring = bearing("608").shape  # three touching solids; (10, 0, 0) lies in the outer ring
    assert not ring.is_inside(Vector(10, 0, 0)) and inside(ring, (10, 0, 0))
    assert not inside(Plane.XY * Rectangle(10, 10), (0, 0, 0))  # no solid, nothing inside


# ------------------------------------------------------------------------------ polygon (E5)


@pytest.mark.parametrize("reverse", [False, True])
def test_polygon_is_counter_clockwise_whatever_the_point_order(reverse):
    """A clockwise build123d Polygon(..., align=None) has a −Z face: it extrudes downward and its
    fuse with a circle goes wrong (review desktop_arm). geom.polygon fixes the winding."""
    from build123d import Circle, Plane, extrude

    from mech.geom import polygon

    pts = [(0, 0), (30, 0), (30, 10), (20, 10), (20, 5), (10, 5), (10, 10), (0, 10)]  # a U, CCW
    p = polygon(pts[::-1] if reverse else pts)
    assert p.faces()[0].normal_at().Z == pytest.approx(1.0)
    assert p.area == pytest.approx(250.0)
    lo = extrude(p, 5).bounding_box().min
    assert (lo.X, lo.Y, lo.Z) == pytest.approx((0, 0, 0))  # points used as given, extruded +Z
    fused = p + Pos(0, 0) * Circle(2)  # a quarter of the circle sticks out of the corner
    assert fused.area == pytest.approx(250 + 3 * math.pi, rel=1e-6)
    on_xz = polygon(pts, plane=Plane.XZ)
    np.testing.assert_allclose(tuple(on_xz.faces()[0].normal_at()), tuple(Plane.XZ.z_dir), atol=1e-12)
    assert polygon(pts + [pts[0]]).area == pytest.approx(250.0)  # a repeated closing point is dropped


@pytest.mark.parametrize("pts, msg", [
    ([(0, 0), (10, 0)], "at least 3"),
    ([(0, 0), (5, 0), (10, 0)], "no area"),
    ([(0, 0), (10, 10), (10, 0), (0, 12)], "cross"),  # a bow tie
    ([(0, 0), (10, 0), (10, 10), (0, 10), (5, 0)], "cross"),  # back along the bottom edge
    ([(0, 0, 1), (1, 0, 1), (0, 1, 1)], "2-D"),
    ([(0, 0), (1, float("nan")), (0, 1)], "finite"),
])
def test_polygon_rejects_bad_outlines(pts, msg):
    from mech.geom import polygon

    with pytest.raises(ValueError, match=msg):
        polygon(pts)


# ------------------------------------------------------------------------------ tessellations / axis cover


def test_tessellate_keeps_the_shape_and_fits_its_surface():
    from build123d import Cylinder

    from mech.geom import tessellate

    shape = Cylinder(10, 4)
    mesh = tessellate(shape)
    assert mesh.shape is not None and mesh.faces == 3 and mesh.triangles.shape[1] == 3
    r = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    assert r.max() == pytest.approx(10.0, abs=1e-9)  # vertices on the true surface
    assert len(shape.faces()) == 3 and not hasattr(shape.wrapped, "_mesh")  # the shape itself untouched


def test_axis_cover_near_and_around():
    """near = within the radius of the line; around = the section surrounds the line (any bore)."""
    from build123d import Cylinder

    from mech.geom import axis_cover, section_radii, tessellate

    ring = tessellate(Cylinder(10, 4) - Cylinder(8, 6))  # z −2…2, bore r = 8
    near, around = axis_cover(ring, (0, 0, 0), (0, 0, 1), 5.0)
    assert near.size == 0
    np.testing.assert_allclose(around, [[-2.0, 2.0]])
    near, around = axis_cover(ring, (0, 0, 0), (0, 0, 1), 8.5)  # the bore is within 8.5
    np.testing.assert_allclose(near, [[-2.0, 2.0]])
    assert section_radii(ring, (0, 0, 0), (0, 0, 1), 1.0) == pytest.approx((8.0, 10.0), abs=0.01)
    assert section_radii(ring, (0, 0, 0), (0, 0, 1), 3.0) is None
    # a line through the wall: the material pierced by the line surrounds it there (the curved
    # wall has many vertex levels along x, so 128 resampled levels: 20/128 mm resolution)
    near, around = axis_cover(ring, (0, 0, 0), (1, 0, 0), 1.0)
    np.testing.assert_allclose(around, [[-10.0, -8.0], [8.0, 10.0]], atol=20 / 128)
    # a C-ring (slit 1 mm) is not around; neither is a ring beside the line
    slit = tessellate((Cylinder(10, 4) - Cylinder(8, 6)) - Pos(10, 0, 0) * Box(6, 1, 10))
    assert all(a.size == 0 for a in axis_cover(slit, (0, 0, 0), (0, 0, 1), 5.0))
    assert all(a.size == 0 for a in axis_cover(ring, (30, 0, 0), (0, 0, 1), 5.0))


def test_axis_cover_three_pins_round_an_axis_do_not_surround_it():
    """Regression: an angle of −1e-17 rad wrapped to exactly 2π covered every bin."""
    from build123d import Compound, Cylinder

    from mech.geom import axis_cover, tessellate

    pins = Compound([Rot(0, 90, 0) * Pos(15 * math.cos(a), 15 * math.sin(a), 0) * Cylinder(1.5, 14)
                     for a in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)])
    near, around = axis_cover(tessellate(pins), (0, 0, 0), (1, 0, 0), 5.0)
    assert near.size == 0 and around.size == 0


def test_surface_axes_lists_revolution_axes_and_plane_normals_near_a_point():
    """The directions a part may surround a point about (a ball in a socket ring): axes of its
    surfaces of revolution passing within the radius of the point, normals of its planar faces
    within the radius; parallel directions once."""
    from build123d import Cylinder

    from mech.geom import surface_axes

    ring = Rot(90, 0, 0) * (Cylinder(10, 4) - Cylinder(8, 6))  # axis along world Y
    axes = surface_axes(ring, (0, 0, 0), 5.0)  # the bore/OD axis (Y) runs through the point
    assert len(axes) == 1 and abs(axes[0] @ (0, 1, 0)) == pytest.approx(1.0)
    # the end faces' planes (y = ±2) pass within 5 of the point too: the same Y, listed once; far
    # from the axis and from those planes nothing qualifies
    assert surface_axes(ring, (30, 30, 0), 5.0) == []
    box = Box(10, 20, 40)  # faces 5, 10, 20 mm from the centre: only the x faces are within 6
    axes = surface_axes(box, (0, 0, 0), 6.0)
    assert len(axes) == 1 and abs(axes[0] @ (1, 0, 0)) == pytest.approx(1.0)


def test_section_reach_is_how_far_the_section_fills_every_direction():
    """A journal reaches its radius all round; an arm off a hub reaches only the hub's radius in
    most directions, however long it is; a flat bar through the axis reaches its half thickness;
    a ring reaches its outer radius; a part off the axis reaches 0 in the directions it misses."""
    from build123d import Cylinder

    from mech.geom import section_reach, tessellate

    z = ((0, 0, 0), (0, 0, 1))
    assert section_reach(tessellate(Cylinder(6, 10)), *z, 0.0) == pytest.approx(6.0, abs=0.02)
    rotor = tessellate(Cylinder(3, 10) + Pos(9.9, 0, 0) * Box(19.8, 2, 4))
    assert section_reach(rotor, *z, 0.0) == pytest.approx(3.0, abs=0.02)
    assert section_reach(tessellate(Box(40, 2, 4)), *z, 0.0) == pytest.approx(1.0, abs=0.05)
    assert section_reach(tessellate(Cylinder(10, 6) - Cylinder(8, 8)), *z, 0.0) == pytest.approx(10.0, abs=0.02)
    assert section_reach(tessellate(Pos(20, 0, 0) * Box(4, 4, 4)), *z, 0.0) == 0.0
    assert section_reach(tessellate(Cylinder(6, 10)), *z, 20.0) is None  # the plane misses it
